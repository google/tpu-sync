// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "torch/headeronly/core/DeviceType.h"
#include "csrc/api/tensor_buffer.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "tpu_sync/core/layout_utils.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/xla_compat.h"

namespace tpu_raiden {
namespace torch {
namespace {

std::vector<int64_t> TensorDimensions(const at::Tensor& tensor) {
  std::vector<int64_t> dimensions;
  dimensions.reserve(tensor.dim());
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    dimensions.push_back(tensor.size(dim));
  }
  return dimensions;
}

// Returns OK if `buffer`, whose leading dimensions `tensor_dims` folds, keeps
// every element at the device byte offset where a buffer of shape `tensor_dims`
// keeps it: raw copies move the bytes as they are, to and from peers that hold
// the tensor in buffers of its own shape.
absl::Status CheckFoldedBufferLayout(const xla::PjRtBuffer& buffer,
                                     absl::Span<const int64_t> tensor_dims) {
  const xla::Shape& buffer_shape = buffer.on_device_shape();
  ABSL_ASSIGN_OR_RETURN(
      const xla::Layout tensor_layout,
      raiden::PjRtClientDefaultLayout(buffer.client(),
                                      buffer_shape.element_type(), tensor_dims),
      _.SetPrepend() << "could not get the device layout of a "
                     << DimsToString(tensor_dims)
                     << " buffer to compare it with: ");
  // Not ABSL_RETURN_IF_ERROR: the open-source Abseil backport
  // (third_party/abseil/status_macros.patch) cannot annotate its error.
  if (absl::Status status =
          CheckFoldedDeviceLayout(buffer_shape, tensor_dims, tensor_layout);
      !status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the buffer does not hold its bytes as a ", DimsToString(tensor_dims),
        " buffer would: ", status.message()));
  }
  return absl::OkStatus();
}

}  // namespace

UnpackedTensor UnpackTorchTensor(const at::Tensor& tensor,
                                 bool unsafe_skip_buffer_lock,
                                 FoldedViews folded_views) {
  if (tensor.device().type() != at::DeviceType::PrivateUse1) {
    throw std::invalid_argument(
        "Tensor must reside on TPU device private use space");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument("Tensor must be contiguous");
  }
  if (tensor.storage_offset() != 0) {
    throw std::invalid_argument(
        "Tensor must start at storage offset 0: raw transfer block-addresses "
        "the base storage buffer, so an offset (sliced) view would read/write "
        "the wrong blocks.");
  }

  // Raw transfer DMAs physical bytes directly into/out of the buffer, so it
  // must operate on the tensor's actual STORAGE buffer -- the one the model
  // reads and writes -- NOT a materialized copy. Materializing a
  // layout-reinterpreting view (e.g. vLLM's
  // `empty().set_(kv_cache.untyped_storage()).view(5D)`, where the model's
  // on-device layout is tiled so the 5D reshape's layout differs from the
  // base) yields a SEPARATE buffer; DMAing into such a copy silently drops
  // H2d (the model never sees the reload) and makes D2h read a one-off
  // snapshot. GetBaseTensorBuffer always returns the buffer backing the
  // tensor's storage, so the DMA lands in the live cache regardless of view.
  auto status_or_ref = torch_tpu::GetBaseTensorBuffer(tensor);
  if (!status_or_ref.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to resolve base device buffer: ", status_or_ref.status().message()));
  }
  torch_tpu::TensorBufferHandle base_ref = std::move(status_or_ref.value());

  if (tensor.dim() == 0 || tensor.size(0) <= 0) {
    throw std::invalid_argument(
        "Tensor must have a non-empty leading block dimension");
  }

  // The base buffer is the storage and may carry different logical dimensions
  // than the tensor's view. Raw transfer slices it by the tensor's logical
  // major (block) dimension while DMAing the base storage buffer. Accept either
  // a matching base major dimension, vLLM's raw rank-1 byte storage exposed
  // as a full rank-2 block-major view: (num_blocks, bytes_per_block), or, where
  // `folded_views` allows it, a view that folds the buffer's leading dimensions
  // into its own. The last includes clones: torch_tpu backs
  // `x.reshape(E * F, D).clone()` with x's [E, F, D] buffer.
  const absl::Span<const int64_t> base_dims = base_ref.dimensions();
  if (base_dims.empty()) {
    throw std::invalid_argument("Tensor base storage buffer has no dimensions");
  }
  std::vector<int64_t> tensor_dims = TensorDimensions(tensor);
  const bool base_major_matches = base_dims[0] == tensor.size(0);
  const bool is_rank1_raw_storage_block_view =
      base_dims.size() == 1 && tensor.dim() == 2;
  const bool folds_leading_dims = !base_major_matches &&
                                  !is_rank1_raw_storage_block_view &&
                                  FoldsLeadingDims(base_dims, tensor_dims);
  if (folds_leading_dims && folded_views == FoldedViews::kReject) {
    throw std::invalid_argument(absl::StrCat(
        "Tensor's leading (block) dimension does not match its base storage "
        "buffer: the tensor's shape ",
        DimsToString(tensor_dims), " folds leading dimensions of the buffer's ",
        DimsToString(base_dims),
        ". Raw transfer and KVCacheManager address the buffer in units of its "
        "own leading dimension, so they reject such tensors; "
        "WeightSynchronizer accepts them if the buffer's device layout "
        "matches."));
  }
  if (!base_major_matches && !is_rank1_raw_storage_block_view &&
      !folds_leading_dims) {
    throw std::invalid_argument(absl::StrCat(
        "Tensor's leading (block) dimension does not match its base storage "
        "buffer; raw transfer requires a layout-preserving full view of the "
        "block-major storage (tensor shape ",
        DimsToString(tensor_dims), ", base storage buffer shape ",
        DimsToString(base_dims), ")."));
  }
  // torch packs float4_e2m1fn_x2 two values per byte while the base buffer
  // (an XLA F4E2M1FN array of twice the minor dimension) is accounted at one
  // byte per element, so the same storage counts twice over there.
  const int64_t base_bytes_per_tensor_byte =
      tensor.scalar_type() == at::kFloat4_e2m1fn_x2 ? 2 : 1;
  if (static_cast<int64_t>(base_ref.size_bytes()) !=
      static_cast<int64_t>(tensor.nbytes()) * base_bytes_per_tensor_byte) {
    throw std::invalid_argument(
        "Tensor does not span its entire base storage buffer; raw transfer "
        "requires a full reinterpret of the storage, not a partial view.");
  }
  const size_t logical_physical_size = static_cast<size_t>(tensor.nbytes());
  const size_t logical_slice_byte_size =
      logical_physical_size / static_cast<size_t>(tensor.size(0));

  // Materialize deferred tensor so AwaitBuffer() won't hang. No-op if already
  // materialized. This should never return a separate buffer otherwise
  // all DMA operations will go to the wrong buffer.
  if (auto status = base_ref.Materialize(); !status.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to materialize base device buffer: ", status.message()));
  }

  auto status_or_buf = base_ref.AwaitBuffer();
  if (!status_or_buf.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to fetch PjRtBuffer from TPU reference: ", status_or_buf.status().message()));
  }
  if (folds_leading_dims) {
    if (absl::Status status =
            CheckFoldedBufferLayout(*status_or_buf.value(), tensor_dims);
        !status.ok()) {
      const std::string message = absl::StrCat(
          "Tensor of shape ", DimsToString(tensor_dims),
          " folds the leading dimensions of its base storage buffer of shape ",
          DimsToString(base_dims), ", but ", status.message());
      if (absl::IsInvalidArgument(status)) {
        throw std::invalid_argument(message);
      }
      throw std::runtime_error(message);
    }
  }

  auto handle_or = raiden::RaidenBufferHandle::Acquire(
      status_or_buf.value(), /*c_api=*/nullptr, /*extension=*/nullptr,
      unsafe_skip_buffer_lock);
  if (!handle_or.ok()) {
    throw std::runtime_error(
        absl::StrCat("Failed to acquire RaidenBufferHandle: ",
                     handle_or.status().message()));
  }

  // Return the buffer AND the owning base ref. The ref pins the storage buffer
  // backing the tensor for as long as the caller keeps it (manager lifetime /
  // transfer future), so the raw PjRtBuffer* stays valid.
  return UnpackedTensor{
      .buffer = std::move(handle_or.value()),
      .ref = std::move(base_ref),
      .logical_dimensions = std::move(tensor_dims),
      .logical_slice_byte_size = logical_slice_byte_size,
      .logical_physical_size = logical_physical_size,
  };
}

}  // namespace torch
}  // namespace tpu_raiden

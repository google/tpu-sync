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

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "ATen/core/TensorBody.h"
#include "torch/headeronly/core/DeviceType.h"

#include "torch_tpu/csrc/eager/materialize.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"

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

}  // namespace

UnpackedTensor UnpackTorchTensor(const at::Tensor& tensor,
                                 bool unsafe_skip_buffer_lock) {
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
  // reads and writes -- NOT a materialized copy. MaterializeAndReturn returns
  // the storage buffer only for a base tensor; for a layout-reinterpreting view
  // (e.g. vLLM's `empty().set_(kv_cache.untyped_storage()).view(5D)`, where the
  // model's on-device layout is tiled so the 5D reshape's layout differs from
  // the base) it returns a SEPARATE materialized buffer. DMAing into that copy
  // silently drops H2d (the model never sees the reload) and makes D2h read a
  // one-off snapshot. GetBaseBuffer always returns the buffer backing the
  // tensor's storage, so the DMA lands in the live cache regardless of view.
  auto status_or_ref = torch_tpu::GetBaseBuffer(tensor);
  if (!status_or_ref.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to resolve base device buffer: ", status_or_ref.status().message()));
  }
  torch_tpu::DeviceBufferRef base_ref = std::move(status_or_ref.value());

  if (tensor.dim() == 0 || tensor.size(0) <= 0) {
    throw std::invalid_argument(
        "Tensor must have a non-empty leading block dimension");
  }

  // The base buffer is the storage and may carry different logical dimensions
  // than the tensor's view. Raw transfer slices it by the tensor's logical
  // major (block) dimension while DMAing the base storage buffer. Accept either
  // a matching base major dimension or vLLM's raw rank-1 byte storage exposed
  // as a full rank-2 block-major view: (num_blocks, bytes_per_block).
  if (base_ref.dimensions().empty()) {
    throw std::invalid_argument("Tensor base storage buffer has no dimensions");
  }
  const bool base_major_matches = base_ref.dimensions()[0] == tensor.size(0);
  const bool is_rank1_raw_storage_block_view =
      base_ref.dimensions().size() == 1 && tensor.dim() == 2;
  if (!base_major_matches && !is_rank1_raw_storage_block_view) {
    throw std::invalid_argument(
        "Tensor's leading (block) dimension does not match its base storage "
        "buffer; raw transfer requires a layout-preserving full view of the "
        "block-major storage.");
  }
  if (static_cast<int64_t>(base_ref.size_bytes()) !=
      static_cast<int64_t>(tensor.nbytes())) {
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
  if (auto status = torch_tpu::Materialize(
          base_ref, torch_tpu::MaterializationReason::kExplicitSync);
      !status.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to materialize base device buffer: ", status.message()));
  }

  auto status_or_buf = base_ref.AwaitBuffer();
  if (!status_or_buf.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to fetch PjRtBuffer from TPU reference: ", status_or_buf.status().message()));
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
      .logical_dimensions = TensorDimensions(tensor),
      .logical_slice_byte_size = logical_slice_byte_size,
      .logical_physical_size = logical_physical_size,
  };
}

}  // namespace torch
}  // namespace tpu_raiden

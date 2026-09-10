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

#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"

namespace tpu_raiden {
namespace torch {

namespace {
std::unordered_map<void*, xla::PjRtBuffer*>& GetMockMap() {
  static auto* m = new std::unordered_map<void*, xla::PjRtBuffer*>();
  return *m;
}

std::vector<int64_t> TensorDimensions(const at::Tensor& tensor) {
  std::vector<int64_t> dimensions;
  dimensions.reserve(tensor.dim());
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    dimensions.push_back(tensor.size(dim));
  }
  return dimensions;
}
}  // namespace

void RegisterMockTensor(const at::Tensor& tensor, xla::PjRtBuffer* buffer) {
  GetMockMap()[tensor.unsafeGetTensorImpl()] = buffer;
}

UnpackedTensor UnpackTorchTensor(const at::Tensor& tensor,
                                 bool unsafe_skip_buffer_lock) {
  auto it = GetMockMap().find(tensor.unsafeGetTensorImpl());
  if (it == GetMockMap().end()) {
    throw std::runtime_error(
        "Mock tensor not registered. Call RegisterMockTensor first.");
  }

  auto handle_or = raiden::RaidenBufferHandle::Acquire(
      it->second, /*c_api=*/nullptr, /*extension=*/nullptr,
      unsafe_skip_buffer_lock);
  if (!handle_or.ok()) {
    throw std::runtime_error(
        "Failed to acquire RaidenBufferHandle for mock buffer");
  }

  // Mock has no real materialization, hence no TensorBufferHandle to hand
  // back.
  if (tensor.dim() == 0 || tensor.size(0) <= 0) {
    return UnpackedTensor{.buffer = std::move(handle_or.value()),
                          .ref = std::nullopt};
  }
  const size_t logical_physical_size = static_cast<size_t>(tensor.nbytes());
  return UnpackedTensor{
      .buffer = std::move(handle_or.value()),
      .ref = std::nullopt,
      .logical_dimensions = TensorDimensions(tensor),
      .logical_slice_byte_size =
          logical_physical_size / static_cast<size_t>(tensor.size(0)),
      .logical_physical_size = logical_physical_size,
  };
}

}  // namespace torch
}  // namespace tpu_raiden

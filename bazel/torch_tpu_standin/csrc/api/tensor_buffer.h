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

#ifndef TORCH_TPU_CSRC_API_TENSOR_BUFFER_H_
#define TORCH_TPU_CSRC_API_TENSOR_BUFFER_H_

#include <cstdint>
#include <memory>

#include "ATen/core/TensorBody.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

// Lifecycle state of the underlying TPU device buffer.
enum class DeviceBufferState {
  kPlaceholder,
  kDeferred,
  kExecuting,
  kMaterializing,
  kMaterialized,
};

// Public-facing handle representing the underlying device buffer of a TPU
// tensor. Provides query, materialization, and synchronization methods without
// exposing internal eager graph abstractions directly.
class TensorBufferHandle final {
 public:
  // Move-only semantics to enforce const-correctness.
  TensorBufferHandle(const TensorBufferHandle&) = delete;
  TensorBufferHandle& operator=(const TensorBufferHandle&) = delete;
  TensorBufferHandle(TensorBufferHandle&& other) noexcept;
  TensorBufferHandle& operator=(TensorBufferHandle&& other) noexcept;
  ~TensorBufferHandle();

  // Returns the logical size in bytes of the referenced buffer.
  [[nodiscard]] int64_t size_bytes() const;

  // Returns the logical dimensions of the referenced buffer.
  [[nodiscard]] absl::Span<const int64_t> dimensions() const;

  // Returns the total number of elements in the referenced buffer.
  [[nodiscard]] int64_t num_elements() const;

  // Returns the current lifecycle state of the referenced buffer.
  [[nodiscard]] DeviceBufferState state() const;

  // Triggers materialization of the referenced buffer in-place.
  absl::Status Materialize();

  // Awaits materialization and returns the underlying PjRtBuffer pointer.
  absl::StatusOr<xla::PjRtBuffer* absl_nonnull> AwaitBuffer();

  // Waits for the device buffer to be ready to read and execution to complete.
  absl::Status Synchronize();

 private:
  class Impl;
  explicit TensorBufferHandle(absl_nonnull std::unique_ptr<Impl> impl);

  absl_nonnull std::unique_ptr<Impl> impl_;

  friend absl::StatusOr<TensorBufferHandle> GetBaseTensorBuffer(
      const at::Tensor& tensor);
};

// Extracts the TensorBufferHandle for the given tensor's base buffer.
// Returns an error if the tensor is not a valid TPU tensor.
absl::StatusOr<TensorBufferHandle> GetBaseTensorBuffer(
    const at::Tensor& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_API_TENSOR_BUFFER_H_

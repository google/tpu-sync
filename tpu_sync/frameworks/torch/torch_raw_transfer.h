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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace raiden {

class RawHostBuffer {
 public:
  explicit RawHostBuffer(int64_t size_bytes);
  ~RawHostBuffer() = default;

  uintptr_t DataPtr() const;
  void* MutableData() const;
  const void* Data() const;
  size_t SizeBytes() const;
  bool IsPjRtBacked() const;
  void EnsureBoundToDevice(xla::PjRtDevice* device);

 private:
  size_t size_bytes_ = 0;
  void* data_ptr_ = nullptr;
  std::unique_ptr<xla::PjRtBuffer> pjrt_buffer_;
  c10::DataPtr data_;
};

class PreparedTorchRawTransfer
    : public std::enable_shared_from_this<PreparedTorchRawTransfer> {
 public:
  PreparedTorchRawTransfer(const at::Tensor& tpu_tensor,
                           std::shared_ptr<RawHostBuffer> host_buffer,
                           bool unsafe_skip_buffer_lock);

  size_t PhysicalSizeBytes() const;
  std::shared_ptr<RawHostBuffer> HostBuffer() const;

  PjRtCopyFuture D2HAsync();
  PjRtCopyFuture H2DAsync();
  void D2H();
  void H2D();

 private:
  std::shared_ptr<RawHostBuffer> host_buffer_;
  // Owns the materialized buffer behind pjrt_buffer_ for this object's lifetime
  // (required when tpu_tensor is a view -> separate materialized buffer).
  std::optional<torch_tpu::DeviceBufferRef> buffer_ref_;
  size_t physical_size_ = 0;
  RaidenBufferHandle buffer_;
};

PjRtCopyFuture TransferD2HBatchAsync(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim);

PjRtCopyFuture TransferH2DBatchAsync(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim);

PjRtCopyFuture TransferD2HAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim);

PjRtCopyFuture TransferH2DAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_H_

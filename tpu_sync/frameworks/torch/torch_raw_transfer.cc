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

#include "tpu_sync/frameworks/torch/torch_raw_transfer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/utils.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"

namespace raiden {
namespace {

using TensorList = std::vector<at::Tensor>;

using ::tpu_raiden::torch::UnpackTorchTensor;

void DeleteHostBufferAllocation(void* ctx) {
  delete static_cast<std::shared_ptr<tpu_raiden::HostBufferAllocation>*>(ctx);
}
}  // namespace

RawHostBuffer::RawHostBuffer(int64_t size_bytes) {
  if (size_bytes < 0) {
    throw std::invalid_argument(
        "RawHostBuffer size_bytes must be non-negative");
  }
  size_bytes_ = static_cast<size_t>(size_bytes);
}

uintptr_t RawHostBuffer::DataPtr() const {
  return reinterpret_cast<uintptr_t>(data_ptr_);
}

void* RawHostBuffer::MutableData() const { return data_ptr_; }

const void* RawHostBuffer::Data() const { return data_ptr_; }

size_t RawHostBuffer::SizeBytes() const { return size_bytes_; }

bool RawHostBuffer::IsPjRtBacked() const { return pjrt_buffer_ != nullptr; }

void RawHostBuffer::EnsureBoundToDevice(xla::PjRtDevice* device) {
  if (data_ptr_ != nullptr || size_bytes_ == 0) {
    return;
  }
  if (device == nullptr) {
    throw std::invalid_argument("Cannot bind RawHostBuffer to null device");
  }
  xla::PjRtMemorySpace* pinned_host = nullptr;
  auto memory_or = device->memory_space_by_kind("pinned_host");
  if (memory_or.ok()) {
    pinned_host = memory_or.value();
  } else {
    for (xla::PjRtMemorySpace* memory : device->memory_spaces()) {
      std::string kind(memory->kind());
      if (kind == "pinned_host" || kind == "PINNED_HOST") {
        pinned_host = memory;
        break;
      }
    }
  }

  if (pinned_host != nullptr) {
    xla::Shape shape =
        xla::ShapeUtil::MakeShape(xla::U8, {static_cast<int64_t>(size_bytes_)});
    auto buffer_or =
        device->client()->CreateUninitializedBuffer(shape, pinned_host);
    if (buffer_or.ok()) {
      pjrt_buffer_ = std::move(buffer_or.value());
      auto ptr_or =
          pjrt_buffer_->client()->UnsafeBufferPointer(pjrt_buffer_.get());
      if (!ptr_or.ok()) {
        throw std::runtime_error(
            std::string("Failed to get pinned host buffer pointer: ") +
            std::string(ptr_or.status().message()));
      }
      data_ptr_ = reinterpret_cast<void*>(ptr_or.value());
      return;
    }
  }

  auto allocator_or = tpu_raiden::HostMemoryAllocator::Create(device->client());
  if (!allocator_or.ok()) {
    throw std::runtime_error("Failed to create TPU pinned host allocator: " +
                             allocator_or.status().ToString());
  }
  auto allocator = std::move(allocator_or).value();
  auto status_or_alloc = allocator->Allocate(size_bytes_);
  if (!status_or_alloc.ok()) {
    throw std::runtime_error("Failed to allocate TPU pinned host buffer: " +
                             status_or_alloc.status().ToString());
  }
  auto alloc = std::move(status_or_alloc).value();
  auto* ctx = new std::shared_ptr<tpu_raiden::HostBufferAllocation>(
      std::make_shared<tpu_raiden::HostBufferAllocation>(std::move(alloc)));
  data_ = c10::DataPtr((*ctx)->ptr, ctx, &DeleteHostBufferAllocation,
                       c10::Device(c10::DeviceType::CPU));
  if (data_.get() == nullptr) {
    throw std::runtime_error("Failed to allocate TPU pinned host buffer");
  }
  data_ptr_ = data_.get();
}

namespace {
[[noreturn]] void ThrowStatus(absl::string_view context,
                              const absl::Status& status) {
  throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
}

template <typename T>
T ValueOrThrow(absl::string_view context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

void ValidateCpuTensor(const at::Tensor& tensor, absl::string_view role) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(absl::StrCat(role, " must be a CPU tensor"));
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(absl::StrCat(role, " must be contiguous"));
  }
}

void AwaitReady(xla::PjRtBuffer* buffer, absl::string_view role) {
  (void)buffer;
  (void)role;
}

// Raw transfer addresses the device buffer as a flat array of equal-size
// major-dimension slices ("blocks"): block i lives at byte offset
// i * GetMajorSliceByteSize(shape). That mapping is only correct when logical
// dimension 0 is the most-major physical dimension and the buffer's physical
// size is an exact multiple of the slice size (the blocks tile it with no
// remainder). Assert both so a buffer with an unexpected on-device layout fails
// loudly here instead of silently transferring the wrong bytes.
void ValidateMajorDimLayout(const RaidenBufferHandle& buffer,
                            absl::string_view role) {
  const xla::Shape& shape = buffer.shape;
  const int rank = shape.dimensions().size();
  if (rank < 1) {
    throw std::invalid_argument(
        absl::StrCat(role, " buffer must have rank >= 1 for block transfer"));
  }
  // In xla::Layout, minor_to_major(rank - 1) is the most-major physical dim.
  if (buffer.buffer && buffer.buffer->layout() &&
      buffer.buffer->layout()->xla_layout().minor_to_major(rank - 1) != 0) {
    throw std::invalid_argument(
        absl::StrCat(role,
                     " buffer layout must place logical dimension 0 as the "
                     "most-major "
                     "physical dimension; block offsetting assumes blocks are "
                     "the "
                     "outermost, physically contiguous dimension."));
  }
  const int64_t slice = GetMajorSliceByteSize(shape);
  // Fallback to buffer->GetOnDeviceSizeInBytes() if available, but for now we
  // might not have it easily without deprecated buffer pointer if it's not
  // cached in handle.
  // Assuming shape gives enough info or buffer pointer is available as fallback
  // in handle for now.
  const int64_t physical_size =
      buffer.buffer
          ? ValueOrThrow(
                absl::StrCat(role, " physical buffer size for layout check"),
                buffer.buffer->GetOnDeviceSizeInBytes())
          : xla::ShapeUtil::ByteSizeOf(shape);  // Fallback

  if (slice <= 0 || physical_size % slice != 0) {
    throw std::invalid_argument(
        absl::StrCat(role,
                     " buffer physical size is not an exact multiple of its "
                     "major-dimension "
                     "slice size; the block-layout assumption does not hold."));
  }
}

PjRtCopyFuture IssueD2HCopy(const RaidenBufferHandle& src_buffer,
                            uint8_t* dst_data, size_t dst_size,
                            const std::vector<int64_t>& src_offsets_major_dim,
                            const std::vector<int64_t>& dst_offsets_major_dim,
                            const std::vector<int64_t>& copy_sizes_major_dim,
                            std::shared_ptr<void> user_hold = nullptr) {
  ValidateMajorDimLayout(src_buffer, "Source");
  const bool is_partial =
      tpu_raiden::IsPartialCopy(src_buffer.shape, src_offsets_major_dim,
                                dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size =
      src_buffer.buffer
          ? ValueOrThrow("Failed to get source physical buffer size",
                         src_buffer.buffer->GetOnDeviceSizeInBytes())
          : xla::ShapeUtil::ByteSizeOf(src_buffer.shape);
  const int64_t slice_byte_size = GetMajorSliceByteSize(src_buffer.shape);

  if (is_partial) {
    tpu_raiden::ValidatePartialAlignment(src_buffer.shape, slice_byte_size);
  }

  std::vector<tpu_raiden::RawCopyChunk> chunks =
      tpu_raiden::ComputeAndValidateChunks(
          slice_byte_size, physical_size, dst_size, is_partial,
          src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim,
          /*is_d2h=*/true);

  std::vector<xla::Future<>> futures;
  futures.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    futures.push_back(src_buffer.CopyRawDeviceToHost(
        dst_data + chunk.dst_offset, chunk.src_offset, chunk.size_bytes));
  }
  return PjRtCopyFuture(
      xla::JoinFutures(absl::MakeSpan(futures)),
      {BufferHolder{src_buffer.c_hold, src_buffer.common_hold,
                    /*ext_hold=*/nullptr, std::move(user_hold)}});
}

PjRtCopyFuture IssueH2DCopy(const uint8_t* src_data, size_t src_size,
                            const RaidenBufferHandle& dst_buffer,
                            const std::vector<int64_t>& src_offsets_major_dim,
                            const std::vector<int64_t>& dst_offsets_major_dim,
                            const std::vector<int64_t>& copy_sizes_major_dim,
                            std::shared_ptr<void> user_hold = nullptr) {
  ValidateMajorDimLayout(dst_buffer, "Destination");
  const bool is_partial =
      tpu_raiden::IsPartialCopy(dst_buffer.shape, src_offsets_major_dim,
                                dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size =
      dst_buffer.buffer
          ? ValueOrThrow("Failed to get destination physical buffer size",
                         dst_buffer.buffer->GetOnDeviceSizeInBytes())
          : xla::ShapeUtil::ByteSizeOf(dst_buffer.shape);
  const int64_t slice_byte_size = GetMajorSliceByteSize(dst_buffer.shape);

  if (is_partial) {
    tpu_raiden::ValidatePartialAlignment(dst_buffer.shape, slice_byte_size);
  }

  std::vector<tpu_raiden::RawCopyChunk> chunks =
      tpu_raiden::ComputeAndValidateChunks(
          slice_byte_size, physical_size, src_size, is_partial,
          src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim,
          /*is_d2h=*/false);

  std::vector<xla::Future<>> futures;
  futures.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    futures.push_back(dst_buffer.CopyRawHostToDevice(
        src_data + chunk.src_offset, chunk.dst_offset, chunk.size_bytes));
  }
  return PjRtCopyFuture(
      xla::JoinFutures(absl::MakeSpan(futures)),
      {BufferHolder{dst_buffer.c_hold, dst_buffer.common_hold,
                    /*ext_hold=*/nullptr, std::move(user_hold)}});
}
}  // namespace

PjRtCopyFuture TransferD2HBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(src_arrs.size());
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(dst_arrs[i], "Destination");
    auto unpacked = UnpackTorchTensor(src_arrs[i]);
    const RaidenBufferHandle& src_buffer = unpacked.buffer;

    auto torch_holds = std::make_shared<std::vector<at::Tensor>>();
    torch_holds->push_back(src_arrs[i]);
    torch_holds->push_back(dst_arrs[i]);

    auto fut = IssueD2HCopy(
        src_buffer, reinterpret_cast<uint8_t*>(dst_arrs[i].data_ptr()),
        dst_arrs[i].nbytes(), src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, std::move(torch_holds));
    // Keep the materialized (possibly view) buffer alive until the copy is
    // done.
    if (unpacked.ref) {
      fut.AddKeepAlive(std::make_shared<torch_tpu::DeviceBufferRef>(
          std::move(*unpacked.ref)));
    }
    futures.push_back(std::move(fut));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture TransferH2DBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(src_arrs.size());
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(src_arrs[i], "Source");
    auto unpacked = UnpackTorchTensor(dst_arrs[i]);
    const RaidenBufferHandle& dst_buffer = unpacked.buffer;

    auto torch_holds = std::make_shared<std::vector<at::Tensor>>();
    torch_holds->push_back(src_arrs[i]);
    torch_holds->push_back(dst_arrs[i]);

    auto fut = IssueH2DCopy(
        reinterpret_cast<const uint8_t*>(src_arrs[i].data_ptr()),
        src_arrs[i].nbytes(), dst_buffer, src_offsets_major_dim,
        dst_offsets_major_dim, copy_sizes_major_dim, std::move(torch_holds));
    // Keep the materialized (possibly view) buffer alive until the copy is
    // done.
    if (unpacked.ref) {
      fut.AddKeepAlive(std::make_shared<torch_tpu::DeviceBufferRef>(
          std::move(*unpacked.ref)));
    }
    futures.push_back(std::move(fut));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture TransferD2HAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferD2HBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

PjRtCopyFuture TransferH2DAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferH2DBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

PreparedTorchRawTransfer::PreparedTorchRawTransfer(
    const at::Tensor& tpu_tensor, std::shared_ptr<RawHostBuffer> host_buffer,
    bool unsafe_skip_buffer_lock)
    : host_buffer_(std::move(host_buffer)) {
  if (!host_buffer_) {
    throw std::invalid_argument("host_buffer must not be None");
  }
  auto unpacked = UnpackTorchTensor(tpu_tensor, unsafe_skip_buffer_lock);
  buffer_ = std::move(unpacked.buffer);
  buffer_ref_ = std::move(unpacked.ref);  // keep the materialized buffer alive
  host_buffer_->EnsureBoundToDevice(buffer_.device);
  physical_size_ = static_cast<size_t>(
      ValueOrThrow("Failed to get TPU physical buffer size",
                   buffer_.buffer ? buffer_.buffer->GetOnDeviceSizeInBytes()
                                  : xla::ShapeUtil::ByteSizeOf(buffer_.shape)));
  if (host_buffer_->SizeBytes() < physical_size_) {
    throw std::invalid_argument(
        "RawHostBuffer is smaller than TPU physical size");
  }
}

size_t PreparedTorchRawTransfer::PhysicalSizeBytes() const {
  return physical_size_;
}

std::shared_ptr<RawHostBuffer> PreparedTorchRawTransfer::HostBuffer() const {
  return host_buffer_;
}

PjRtCopyFuture PreparedTorchRawTransfer::D2HAsync() {
  xla::Future<> copy_future = buffer_.CopyRawDeviceToHost(
      host_buffer_->MutableData(), 0, physical_size_);
  return PjRtCopyFuture(
      std::move(copy_future),
      {BufferHolder{buffer_.c_hold, buffer_.common_hold, /*ext_hold=*/nullptr,
                    shared_from_this()}});
}

PjRtCopyFuture PreparedTorchRawTransfer::H2DAsync() {
  xla::Future<> copy_future =
      buffer_.CopyRawHostToDevice(host_buffer_->Data(), 0, physical_size_);
  return PjRtCopyFuture(
      std::move(copy_future),
      {BufferHolder{buffer_.c_hold, buffer_.common_hold, /*ext_hold=*/nullptr,
                    shared_from_this()}});
}

void PreparedTorchRawTransfer::D2H() {
  PjRtCopyFuture future = D2HAsync();
  absl::Status status = future.Await();
  if (!status.ok()) {
    ThrowStatus("D2H copy failed", status);
  }
}

void PreparedTorchRawTransfer::H2D() {
  PjRtCopyFuture future = H2DAsync();
  absl::Status status = future.Await();
  if (!status.ok()) {
    ThrowStatus("H2D copy failed", status);
  }
}

}  // namespace raiden

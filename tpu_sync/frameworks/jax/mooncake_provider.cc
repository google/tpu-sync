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

#include "tpu_sync/frameworks/jax/mooncake_provider.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"

namespace raiden::mooncake {

// The ABI's binary layout (LP64). This file implements version 1, so a new
// version must fail the build until its table is re-checked field by field.
static_assert(kAbiVersion == 1);
static_assert(sizeof(BufferHandle) == 8);
static_assert(offsetof(ProviderTable, abi_version) == 0);
static_assert(offsetof(ProviderTable, name) == 8);
static_assert(offsetof(ProviderTable, size) == 16);
static_assert(offsetof(ProviderTable, device_index) == 24);
static_assert(offsetof(ProviderTable, numa_node) == 32);
static_assert(offsetof(ProviderTable, copy_to_host) == 40);
static_assert(offsetof(ProviderTable, copy_from_host) == 48);
static_assert(offsetof(ProviderTable, release) == 56);
static_assert(sizeof(ProviderTable) == 64);

namespace {

std::atomic<int64_t> live_buffer_count{0};

// TPU DMA rejects host buffers that are not 32-byte aligned (measured on TPU
// v6e and v7x; the runtime's dma_requirements().host_alignment_bytes, which
// PJRT does not expose). Copies through other host pointers are staged through
// an aligned bounce buffer. 64 is a power of two above the measured value.
constexpr size_t kHostDmaAlignment = 64;

bool IsDmaAligned(const void* host) {
  return reinterpret_cast<uintptr_t>(host) % kHostDmaAlignment == 0;
}

// A DMA-aligned scratch host buffer of at least `size` bytes.
class BounceBuffer {
 public:
  explicit BounceBuffer(size_t size)
      : data_(std::aligned_alloc(kHostDmaAlignment,
                                 (size + kHostDmaAlignment - 1) /
                                     kHostDmaAlignment * kHostDmaAlignment)) {}
  ~BounceBuffer() { std::free(data_); }
  BounceBuffer(const BounceBuffer&) = delete;
  BounceBuffer& operator=(const BounceBuffer&) = delete;

  void* data() const { return data_; }

 private:
  void* const data_;
};

// The state behind one handle. Immutable after construction, so any number of
// threads may copy through the same buffer at once.
class ProviderBuffer {
 public:
  ProviderBuffer(RaidenBufferHandle buffer, int device_index)
      : buffer_(std::move(buffer)),
        size_(buffer_.GetOnDeviceSizeInBytes()),
        device_index_(device_index),
        numa_node_(tpu_raiden::GetPjRtDeviceNumaNode(buffer_.device)) {
    live_buffer_count.fetch_add(1, std::memory_order_relaxed);
  }
  ~ProviderBuffer() {
    live_buffer_count.fetch_sub(1, std::memory_order_relaxed);
  }

  ProviderBuffer(const ProviderBuffer&) = delete;
  ProviderBuffer& operator=(const ProviderBuffer&) = delete;

  size_t size() const { return size_; }
  int device_index() const { return device_index_; }
  int numa_node() const { return numa_node_; }

  // Blocking copies of bytes [offset, offset + len) of the on-device
  // representation. Any host pointer works; `offset` must meet the device's
  // DMA alignment (32 bytes on TPU v6e and v7x), or the copy fails.
  absl::Status CopyToHost(size_t offset, void* host_dst, size_t len) const {
    ABSL_RETURN_IF_ERROR(CheckRange(offset, host_dst, len));
    if (len == 0) return absl::OkStatus();
    if (IsDmaAligned(host_dst)) return DeviceToHost(offset, host_dst, len);
    BounceBuffer bounce(len);
    if (bounce.data() == nullptr) {
      return absl::ResourceExhaustedError("cannot allocate a bounce buffer");
    }
    ABSL_RETURN_IF_ERROR(DeviceToHost(offset, bounce.data(), len));
    std::memcpy(host_dst, bounce.data(), len);
    return absl::OkStatus();
  }

  absl::Status CopyFromHost(size_t offset, const void* host_src,
                            size_t len) const {
    ABSL_RETURN_IF_ERROR(CheckRange(offset, host_src, len));
    if (len == 0) return absl::OkStatus();
    if (IsDmaAligned(host_src)) return HostToDevice(offset, host_src, len);
    BounceBuffer bounce(len);
    if (bounce.data() == nullptr) {
      return absl::ResourceExhaustedError("cannot allocate a bounce buffer");
    }
    std::memcpy(bounce.data(), host_src, len);
    return HostToDevice(offset, bounce.data(), len);
  }

 private:
  absl::Status DeviceToHost(size_t offset, void* host_dst, size_t len) const {
    ABSL_ASSIGN_OR_RETURN(
        PjRtCopyFuture copy,
        IssueD2hShard(buffer_, {D2hCopy{host_dst, static_cast<int64_t>(offset),
                                        static_cast<int64_t>(len)}}));
    return copy.Await();
  }

  absl::Status HostToDevice(size_t offset, const void* host_src,
                            size_t len) const {
    ABSL_ASSIGN_OR_RETURN(
        PjRtCopyFuture copy,
        IssueH2dShard(buffer_, {H2dCopy{host_src, static_cast<int64_t>(offset),
                                        static_cast<int64_t>(len)}}));
    return copy.Await();
  }

  // The adapter promises in-range copies; this guards against a broken one.
  absl::Status CheckRange(size_t offset, const void* host, size_t len) const {
    if (offset > size_ || len > size_ - offset) {
      return absl::OutOfRangeError(absl::StrCat("range [", offset, ", +", len,
                                                ") exceeds buffer of ", size_,
                                                " bytes"));
    }
    if (host == nullptr && len > 0) {
      return absl::InvalidArgumentError("null host pointer");
    }
    return absl::OkStatus();
  }

  const RaidenBufferHandle buffer_;
  const size_t size_;
  const int device_index_;
  const int numa_node_;
};

const ProviderBuffer* FromHandle(BufferHandle handle) {
  return reinterpret_cast<const ProviderBuffer*>(
      static_cast<uintptr_t>(handle));
}

bool HoldsDeviceBuffer(const RaidenBufferHandle& buffer) {
  return buffer.is_common_buffer ? buffer.common_raw_buffer != nullptr
                                 : buffer.c_hold != nullptr;
}

// --- The C table. Called by Mooncake from many threads at once. ------------

size_t Size(BufferHandle handle) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return buffer == nullptr ? 0 : buffer->size();
}

int DeviceIndex(BufferHandle handle) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return buffer == nullptr ? -1 : buffer->device_index();
}

int NumaNode(BufferHandle handle) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return buffer == nullptr ? -1 : buffer->numa_node();
}

// The C boundary for both copy directions: no status or exception crosses it,
// only 0 (success) or 1 (failure, logged here).
int RunCopy(const char* direction, BufferHandle handle, size_t offset,
            size_t len,
            absl::FunctionRef<absl::Status(const ProviderBuffer&)> copy) {
  absl::Status status;
  const ProviderBuffer* buffer = FromHandle(handle);
  if (buffer == nullptr) {
    status = absl::InvalidArgumentError("invalid handle 0");
  } else {
    try {
      status = copy(*buffer);
    } catch (const std::exception& e) {
      status = absl::InternalError(e.what());
    } catch (...) {
      status = absl::InternalError("unknown exception");
    }
  }
  if (status.ok()) return 0;
  LOG(ERROR) << "tpu-sync Mooncake provider: " << direction << " of " << len
             << " bytes at offset " << offset << " of "
             << (buffer == nullptr ? std::string("handle 0")
                                   : absl::StrCat("a ", buffer->size(),
                                                  "-byte buffer on tpu:",
                                                  buffer->device_index()))
             << " failed: " << status;
  return 1;
}

int CopyToHost(BufferHandle handle, size_t offset, void* host_dst, size_t len) {
  return RunCopy("copy_to_host", handle, offset, len,
                 [&](const ProviderBuffer& buffer) {
                   return buffer.CopyToHost(offset, host_dst, len);
                 });
}

int CopyFromHost(BufferHandle handle, size_t offset, const void* host_src,
                 size_t len) {
  return RunCopy("copy_from_host", handle, offset, len,
                 [&](const ProviderBuffer& buffer) {
                   return buffer.CopyFromHost(offset, host_src, len);
                 });
}

void Release(BufferHandle handle) { delete FromHandle(handle); }

constexpr ProviderTable kProviderTable = {
    .abi_version = kAbiVersion,
    .name = "tpu-sync",
    .size = &Size,
    .device_index = &DeviceIndex,
    .numa_node = &NumaNode,
    .copy_to_host = &CopyToHost,
    .copy_from_host = &CopyFromHost,
    .release = &Release,
};

}  // namespace

absl::StatusOr<std::vector<BufferHandle>> ExportBuffers(
    std::vector<ExportRequest> requests) {
  std::vector<std::unique_ptr<ProviderBuffer>> buffers;
  buffers.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    ExportRequest& request = requests[i];
    if (!HoldsDeviceBuffer(request.buffer)) {
      return absl::InvalidArgumentError(
          absl::StrCat("export request ", i, " holds no device buffer"));
    }
    if (request.device_index < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "export request ", i, " has device index ", request.device_index));
    }
    buffers.push_back(std::make_unique<ProviderBuffer>(
        std::move(request.buffer), request.device_index));
  }
  // Reserved first so that nothing can throw once ownership is handed out.
  std::vector<BufferHandle> handles;
  handles.reserve(buffers.size());
  for (std::unique_ptr<ProviderBuffer>& buffer : buffers) {
    handles.push_back(reinterpret_cast<uintptr_t>(buffer.release()));
  }
  return handles;
}

int64_t LiveBufferCount() {
  return live_buffer_count.load(std::memory_order_relaxed);
}

}  // namespace raiden::mooncake

extern "C" int mc_tpu_buffer_provider_get(
    uint32_t requested_version, raiden::mooncake::ProviderTable* out) {
  if (out == nullptr || requested_version != raiden::mooncake::kAbiVersion) {
    return 1;
  }
  *out = raiden::mooncake::kProviderTable;
  return 0;
}

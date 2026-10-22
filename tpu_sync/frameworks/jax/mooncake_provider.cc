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
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/mooncake/tpu_buffer_provider.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"

// This file implements version 1 of the table. A new upstream header must be
// re-checked field by field, so fail the build rather than serve it silently.
static_assert(MC_TPU_BUFFER_PROVIDER_ABI_VERSION == 1u);
static_assert(sizeof(mc_tpu_buffer_t) == 8);
static_assert(offsetof(mc_tpu_buffer_provider_t, abi_version) == 0);
static_assert(offsetof(mc_tpu_buffer_provider_t, name) == 8);
static_assert(offsetof(mc_tpu_buffer_provider_t, size) == 16);
static_assert(offsetof(mc_tpu_buffer_provider_t, device_index) == 24);
static_assert(offsetof(mc_tpu_buffer_provider_t, numa_node) == 32);
static_assert(offsetof(mc_tpu_buffer_provider_t, copy_to_host) == 40);
static_assert(offsetof(mc_tpu_buffer_provider_t, copy_from_host) == 48);
static_assert(offsetof(mc_tpu_buffer_provider_t, release) == 56);
static_assert(sizeof(mc_tpu_buffer_provider_t) == 64);
static_assert(std::is_same_v<decltype(&mc_tpu_buffer_provider_get),
                             mc_tpu_buffer_provider_get_fn>);

namespace raiden::mooncake {
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

const ProviderBuffer* FromHandle(mc_tpu_buffer_t handle) {
  return reinterpret_cast<const ProviderBuffer*>(
      static_cast<uintptr_t>(handle));
}

bool HoldsDeviceBuffer(const RaidenBufferHandle& buffer) {
  return buffer.is_common_buffer ? buffer.common_raw_buffer != nullptr
                                 : buffer.c_hold != nullptr;
}

// --- The C table. Called by Mooncake from many threads at once. ------------

size_t Size(mc_tpu_buffer_t handle) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return buffer == nullptr ? 0 : buffer->size();
}

int DeviceIndex(mc_tpu_buffer_t handle) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return buffer == nullptr ? -1 : buffer->device_index();
}

int NumaNode(mc_tpu_buffer_t handle) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return buffer == nullptr ? -1 : buffer->numa_node();
}

// The C boundary for both copy directions: no status or exception crosses it,
// only 0 (success) or 1 (failure, logged here).
int RunCopy(const char* direction, mc_tpu_buffer_t handle, size_t offset,
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

int CopyToHost(mc_tpu_buffer_t handle, size_t offset, void* host_dst,
               size_t len) {
  return RunCopy("copy_to_host", handle, offset, len,
                 [&](const ProviderBuffer& buffer) {
                   return buffer.CopyToHost(offset, host_dst, len);
                 });
}

int CopyFromHost(mc_tpu_buffer_t handle, size_t offset, const void* host_src,
                 size_t len) {
  return RunCopy("copy_from_host", handle, offset, len,
                 [&](const ProviderBuffer& buffer) {
                   return buffer.CopyFromHost(offset, host_src, len);
                 });
}

void Release(mc_tpu_buffer_t handle) { delete FromHandle(handle); }

constexpr mc_tpu_buffer_provider_t kProviderTable = {
    .abi_version = MC_TPU_BUFFER_PROVIDER_ABI_VERSION,
    .name = "tpu-sync",
    .size = &Size,
    .device_index = &DeviceIndex,
    .numa_node = &NumaNode,
    .copy_to_host = &CopyToHost,
    .copy_from_host = &CopyFromHost,
    .release = &Release,
};

}  // namespace

absl::StatusOr<std::vector<mc_tpu_buffer_t>> ExportBuffers(
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
  std::vector<mc_tpu_buffer_t> handles;
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

extern "C" int mc_tpu_buffer_provider_get(uint32_t requested_version,
                                          mc_tpu_buffer_provider_t* out) {
  if (out == nullptr ||
      requested_version != MC_TPU_BUFFER_PROVIDER_ABI_VERSION) {
    return 1;
  }
  *out = raiden::mooncake::kProviderTable;
  return 0;
}

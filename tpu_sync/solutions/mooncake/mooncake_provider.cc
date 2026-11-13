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

#include "tpu_sync/solutions/mooncake/mooncake_provider.h"

#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/core/xla_compat.h"

namespace raiden::mooncake {

// The ABI's binary layout (LP64), as in Mooncake's tpu_buffer_provider.h.
static_assert(sizeof(BufferHandle) == 8);
static_assert(offsetof(ProviderTable, name) == 0);
static_assert(offsetof(ProviderTable, size) == 8);
static_assert(offsetof(ProviderTable, device_index) == 16);
static_assert(offsetof(ProviderTable, numa_node) == 24);
static_assert(offsetof(ProviderTable, copy_to_host) == 32);
static_assert(offsetof(ProviderTable, copy_from_host) == 40);
static_assert(offsetof(ProviderTable, release) == 48);
static_assert(offsetof(ProviderTable, dma_map) == 56);
static_assert(offsetof(ProviderTable, dma_unmap) == 64);
static_assert(sizeof(ProviderTable) == 72);

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
  ProviderBuffer(RaidenBufferHandle buffer, int device_index,
                 std::shared_ptr<void> keep_alive)
      : keep_alive_(std::move(keep_alive)),
        buffer_(std::move(buffer)),
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

  // Destroyed after `buffer_`, whose PjRtBuffer pointer it keeps valid for a
  // torch tensor (see UnpackTorchTensor).
  const std::shared_ptr<void> keep_alive_;
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

// Host memory mapped through dma_map: each range, by its start, with the PJRT
// client that mapped it and must unmap it. No two ranges overlap.
//
// One mutex guards all of it and is held across the PJRT calls. A process maps
// only a few ranges, so the serialization costs nothing, and it makes
// MarkProcessExiting() a barrier: once it returns, no PJRT call is running and
// none can start.
class DmaMappings {
 public:
  // Makes `client` the one that later Map() calls map with.
  void SetClient(xla::PjRtClient* client) {
    absl::MutexLock lock(mu_);
    client_ = client;
  }

  absl::Status Map(void* host, size_t len) {
    static const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t start = reinterpret_cast<uintptr_t>(host);
    if (host == nullptr || len == 0 || start % page_size != 0 ||
        len % page_size != 0 ||
        len > std::numeric_limits<uintptr_t>::max() - start) {
      return absl::InvalidArgumentError(absl::StrCat(
          "the range must be non-empty, aligned to the ", page_size,
          "-byte page size, and inside the address space"));
    }
    absl::MutexLock lock(mu_);
    if (process_exiting_) {
      return absl::FailedPreconditionError("the process is exiting");
    }
    if (client_ == nullptr) {
      return absl::FailedPreconditionError(
          "no buffer has been exported, so there is no PJRT client to map "
          "with; export the arrays before creating the TransferEngine");
    }
    if (!PjRtClientSupportsDma(client_)) {
      return absl::FailedPreconditionError(
          "the PJRT client of the exported buffers cannot DMA-map host memory "
          "(that takes a PJRT C API client of a platform other than CPU)");
    }
    // The TPU client does not reject a range that overlaps a mapped one itself,
    // though its interval map of mapped ranges requires that none do.
    for (const auto& [mapped_host, mapping] : mappings_) {
      const uintptr_t mapped_start = reinterpret_cast<uintptr_t>(mapped_host);
      if (start < mapped_start + mapping.len && mapped_start < start + len) {
        return absl::AlreadyExistsError(
            "the range overlaps one that dma_map already mapped");
      }
    }
    // Reserved first, so that recording a successful map cannot fail.
    mappings_.reserve(mappings_.size() + 1);
    if (absl::Status status = PjRtClientDmaMap(client_, host, len);
        !status.ok()) {
      // The TPU client maps the range for every device, and when some of them
      // fail it keeps the mappings on the others. Mooncake will free the memory
      // without unmapping it, so undo them; devices without a mapping ignore
      // this.
      (void)PjRtClientDmaUnmap(client_, host);
      return status;
    }
    mappings_.emplace(host, Mapping{len, client_});
    return absl::OkStatus();
  }

  absl::Status Unmap(void* host) {
    absl::MutexLock lock(mu_);
    auto it = mappings_.find(host);
    if (it == mappings_.end()) {
      return absl::NotFoundError("no range mapped by dma_map starts there");
    }
    xla::PjRtClient* const client = it->second.client;
    // Forgotten even if PJRT fails: Mooncake frees the memory either way.
    mappings_.erase(it);
    // The framework may already have destroyed the client, and the exiting
    // process takes the mapping with it.
    if (process_exiting_) return absl::OkStatus();
    return PjRtClientDmaUnmap(client, host);
  }

  void MarkProcessExiting() {
    absl::MutexLock lock(mu_);
    process_exiting_ = true;
  }

  int64_t count() {
    absl::MutexLock lock(mu_);
    return mappings_.size();
  }

 private:
  struct Mapping {
    size_t len;
    xla::PjRtClient* client;
  };

  absl::Mutex mu_;
  xla::PjRtClient* client_ ABSL_GUARDED_BY(mu_) = nullptr;
  bool process_exiting_ ABSL_GUARDED_BY(mu_) = false;
  absl::flat_hash_map<void*, Mapping> mappings_ ABSL_GUARDED_BY(mu_);
};

// Never destroyed, so that a dma_unmap during static destruction still finds
// its mapping.
DmaMappings& ProcessDmaMappings() {
  static absl::NoDestructor<DmaMappings> mappings;
  return *mappings;
}

// --- The C table. Called by Mooncake from many threads at once. ------------

// The C boundary: a status or exception from `body` does not cross it. Returns
// 0 on success, and 1 on failure, which it logs as `operation()` failing.
int RunAtCBoundary(absl::FunctionRef<absl::Status()> body,
                   absl::FunctionRef<std::string()> operation) {
  absl::Status status;
  try {
    status = body();
  } catch (const std::exception& e) {
    status = absl::InternalError(e.what());
  } catch (...) {
    status = absl::InternalError("unknown exception");
  }
  if (status.ok()) return 0;
  LOG(ERROR) << "tpu-sync Mooncake provider: " << operation()
             << " failed: " << status;
  return 1;
}

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

// Both copy directions.
int RunCopy(const char* direction, BufferHandle handle, size_t offset,
            size_t len,
            absl::FunctionRef<absl::Status(const ProviderBuffer&)> copy) {
  const ProviderBuffer* buffer = FromHandle(handle);
  return RunAtCBoundary(
      [&] {
        return buffer == nullptr
                   ? absl::InvalidArgumentError("invalid handle 0")
                   : copy(*buffer);
      },
      [&] {
        return absl::StrCat(
            direction, " of ", len, " bytes at offset ", offset, " of ",
            buffer == nullptr
                ? std::string("handle 0")
                : absl::StrCat("a ", buffer->size(),
                               "-byte buffer on tpu:", buffer->device_index()));
      });
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

int DmaMap(void* host, size_t len) {
  return RunAtCBoundary(
      [&] { return ProcessDmaMappings().Map(host, len); },
      [&] { return absl::StrFormat("dma_map of %d bytes at %p", len, host); });
}

int DmaUnmap(void* host) {
  return RunAtCBoundary(
      [&] { return ProcessDmaMappings().Unmap(host); },
      [&] { return absl::StrFormat("dma_unmap at %p", host); });
}

constexpr ProviderTable kProviderTable = {
    .name = "tpu-sync",
    .size = &Size,
    .device_index = &DeviceIndex,
    .numa_node = &NumaNode,
    .copy_to_host = &CopyToHost,
    .copy_from_host = &CopyFromHost,
    .release = &Release,
    .dma_map = &DmaMap,
    .dma_unmap = &DmaUnmap,
};

}  // namespace

absl::StatusOr<std::vector<BufferHandle>> ExportBuffers(
    std::vector<ExportRequest> requests) {
  std::vector<std::unique_ptr<ProviderBuffer>> buffers;
  buffers.reserve(requests.size());
  xla::PjRtClient* client = nullptr;
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
    if (request.buffer.device != nullptr) {
      client = request.buffer.device->client();
    }
    buffers.push_back(std::make_unique<ProviderBuffer>(
        std::move(request.buffer), request.device_index,
        std::move(request.keep_alive)));
  }
  // Reserved first so that nothing can throw once ownership is handed out.
  std::vector<BufferHandle> handles;
  handles.reserve(buffers.size());
  for (std::unique_ptr<ProviderBuffer>& buffer : buffers) {
    handles.push_back(reinterpret_cast<uintptr_t>(buffer.release()));
  }
  // JAX and TorchTPU each have one TPU client per process, so dma_map simply
  // maps with the client of the latest export.
  if (client != nullptr) ProcessDmaMappings().SetClient(client);
  return handles;
}

int64_t LiveBufferCount() {
  return live_buffer_count.load(std::memory_order_relaxed);
}

int64_t LiveDmaMappingCount() { return ProcessDmaMappings().count(); }

void MarkProcessExiting() { ProcessDmaMappings().MarkProcessExiting(); }

int GetProviderTable(ProviderTable* out) {
  if (out == nullptr) return 1;
  *out = kProviderTable;
  return 0;
}

}  // namespace raiden::mooncake

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

#include "tpu_sync/core/host_memory_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/tpu_utils.h"

namespace tpu_raiden {
namespace {
thread_local const xla::PjRtDevice* g_current_device = nullptr;

// PJRT_Client_DmaMap aborts inside libtpu on multi-process TPUv7 pods
// (tpu::System::DmaMap raw_hash_map::at out-of-range), so DMA mapping must be
// skipped there.
bool ShouldSkipDmaMap(xla::PjRtClient* client) {
  bool is_tpuv7 = false;
  if (client != nullptr) {
    absl::string_view version = client->platform_version();
    if (absl::StrContains(version, "7") || absl::StrContains(version, "v7")) {
      is_tpuv7 = true;
    }
    if (!client->devices().empty() && client->devices()[0] != nullptr) {
      absl::string_view kind = client->devices()[0]->device_kind();
      if (absl::StrContains(kind, "7") || absl::StrContains(kind, "v7")) {
        is_tpuv7 = true;
      }
    }
  }
  const bool is_multi_process =
      (std::getenv("PJRT_LOCAL_PROCESS_RANK") != nullptr ||
       std::getenv("RANK") != nullptr || std::getenv("LOCAL_RANK") != nullptr);
  return is_tpuv7 && is_multi_process;
}
}  // namespace

absl::StatusOr<std::unique_ptr<HostMemoryAllocator>>
HostMemoryAllocator::Create(xla::PjRtClient* pjrt_client) {
  if (pjrt_client != nullptr && pjrt_client->platform_name() != "cpu") {
    return XlaHostMemoryAllocator::Create(pjrt_client);
  }
  return std::unique_ptr<HostMemoryAllocator>(
      std::make_unique<MallocHostMemoryAllocator>());
}

absl::StatusOr<std::unique_ptr<XlaHostMemoryAllocator>>
XlaHostMemoryAllocator::Create(xla::PjRtClient* absl_nonnull pjrt_client) {
  return std::unique_ptr<XlaHostMemoryAllocator>(
      new XlaHostMemoryAllocator(pjrt_client));
}

XlaHostMemoryAllocator::XlaHostMemoryAllocator(xla::PjRtClient* client)
    : client_(client) {}

absl::StatusOr<HostBufferAllocation> XlaHostMemoryAllocator::AllocateDmaMapped(
    size_t size_bytes) {
  return Allocate(size_bytes);
}

absl::StatusOr<HostBufferAllocation>
XlaHostMemoryAllocator::AllocateDmaMappedForDevice(
    size_t size_bytes, const xla::PjRtDevice* device) {
  g_current_device = device;
  int numa_node = GetPjRtDeviceNumaNode(device);
  VLOG(1) << "[ALLOCATOR] Device: "
          << (device ? device->DebugString() : "nullptr")
          << ", resolved NUMA node: " << numa_node;
  if (numa_node >= 0) {
    // Bind this thread's allocations to the TPU's local NUMA node
    SetThreadMempolicy(2, numa_node);  // MPOL_BIND

    auto alloc_or = AllocateDmaMapped(size_bytes);
    if (alloc_or.ok()) {
      // Touch one byte per page (4KB) using a volatile pointer to force
      // physical page allocation (first-touch) on the bound NUMA node.
      volatile uint8_t* p =
          static_cast<volatile uint8_t*>(alloc_or.value().ptr);
      for (size_t i = 0; i < size_bytes; i += 4096) {
        p[i] = 0;
      }
    }

    // Restore default policy
    SetThreadMempolicy(0);  // MPOL_DEFAULT

    g_current_device = nullptr;
    return alloc_or;
  }
  auto alloc_or = AllocateDmaMapped(size_bytes);
  g_current_device = nullptr;
  return alloc_or;
}

absl::StatusOr<HostBufferAllocation> XlaHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  if (size_bytes == 0) {
    HostBufferAllocation alloc;
    alloc.ptr = nullptr;
    alloc.size = 0;
    return alloc;
  }

  const bool skip_dma_map = ShouldSkipDmaMap(client_);

  // HYBRID PATH: When skipping DmaMap on multi-process TPUv7 pods, delegate
  // directly to libtpu's internal host memory allocator pool. This yields
  // pre-mapped VFIO staged DMA memory (~175 GB/s H2D / ~128 GB/s D2H) rather
  // than raw unpinned OS memory (~106 GB/s), saving ~70 GB/s on MPMD workloads.
  if (skip_dma_map && client_ != nullptr &&
      client_->GetHostMemoryAllocator() != nullptr) {
    xla::HostMemoryAllocator::AllocateOptions alloc_opts;
    if (g_current_device != nullptr) {
      alloc_opts.numa_node = GetPjRtDeviceNumaNode(g_current_device);
      alloc_opts.local_device_id = g_current_device->local_device_id();
    }
    xla::HostMemoryAllocator* host_alloc = client_->GetHostMemoryAllocator();
    xla::HostMemoryAllocator::OwnedPtr data =
        host_alloc->Allocate(size_bytes, alloc_opts);
    if (data != nullptr) {
      HostBufferAllocation alloc;
      alloc.ptr = data.get();
      alloc.size = size_bytes;
      alloc.owner = std::shared_ptr<void>(
          data.get(), [d = std::move(data)](void*) mutable { d.reset(); });
      return alloc;
    }
  }

  size_t aligned_size = (size_bytes + 4095) & ~4095;

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return absl::ResourceExhaustedError(
        absl::StrCat("mmap failed for size: ", aligned_size));
  }

  bool dma_mapped = false;

  if (!skip_dma_map) {
    auto status = client_->DmaMap(ptr, aligned_size);
    if (status.ok()) {
      dma_mapped = true;
    } else {
      munmap(ptr, aligned_size);
      return absl::InternalError(
          absl::StrCat("DmaMap failed: ", status.message()));
    }
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;

  xla::PjRtClient* client = client_;
  alloc.owner =
      std::shared_ptr<void>(ptr, [client, aligned_size, dma_mapped](void* p) {
        if (dma_mapped) {
          (void)client->DmaUnmap(p);
        }
        munmap(p, aligned_size);
      });

  return alloc;
}

absl::StatusOr<HostBufferAllocation> MallocHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  void* ptr = nullptr;
  if (size_bytes > 0) {
    int err = posix_memalign(&ptr, 64, size_bytes);
    if (err != 0) {
      return absl::ResourceExhaustedError(
          absl::StrCat("posix_memalign failed with error: ", err));
    }
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;
  alloc.owner = std::shared_ptr<void>(ptr, std::free);
  return alloc;
}

absl::StatusOr<std::unique_ptr<SharedMemoryHostMemoryAllocator>>
SharedMemoryHostMemoryAllocator::Create(
    xla::PjRtClient* client, absl::string_view shm_key,
    const SharedMemoryHeader& expected_schema) {
  return std::unique_ptr<SharedMemoryHostMemoryAllocator>(
      new SharedMemoryHostMemoryAllocator(client, shm_key, expected_schema));
}

SharedMemoryHostMemoryAllocator::SharedMemoryHostMemoryAllocator(
    xla::PjRtClient* client, absl::string_view shm_key,
    const SharedMemoryHeader& expected_schema)
    : client_(client), shm_key_(shm_key), expected_schema_(expected_schema) {}

SharedMemoryHostMemoryAllocator::~SharedMemoryHostMemoryAllocator() {
  if (mapped_ptr_ != nullptr && mapped_ptr_ != MAP_FAILED) {
    SharedMemoryHeader* header = static_cast<SharedMemoryHeader*>(mapped_ptr_);
    if (header->reference_count > 0) {
      header->reference_count--;
    }
    if (dma_mapped_ && client_ != nullptr) {
      (void)client_->DmaUnmap(mapped_ptr_);
    }
    munmap(mapped_ptr_, mapped_size_);
  }
  if (shm_fd_ >= 0) {
    close(shm_fd_);
  }
}

absl::StatusOr<HostBufferAllocation> SharedMemoryHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  if (size_bytes == 0) {
    HostBufferAllocation alloc;
    alloc.ptr = nullptr;
    alloc.size = 0;
    return alloc;
  }

  if (mapped_ptr_ != nullptr && mapped_ptr_ != MAP_FAILED) {
    HostBufferAllocation alloc;
    alloc.ptr = static_cast<uint8_t*>(mapped_ptr_) + sizeof(SharedMemoryHeader);
    alloc.size = size_bytes;
    alloc.owner = std::shared_ptr<void>(mapped_ptr_, [](void*) {});
    return alloc;
  }

  size_t aligned_payload_size = (size_bytes + 4095) & ~4095;
  size_t total_size =
      (sizeof(SharedMemoryHeader) + aligned_payload_size + 4095) & ~4095;
  mapped_size_ = total_size;

  std::string full_shm_key = shm_key_;
  const char* server_name_env = std::getenv("RAIDEN_SHM_SERVER_NAME");
  if (server_name_env != nullptr && std::strlen(server_name_env) > 0) {
    absl::StrAppend(&full_shm_key, "_", server_name_env);
  }
  if (g_current_device != nullptr) {
    absl::StrAppend(&full_shm_key, "_dev_",
                    g_current_device->local_device_id().value());
  }
  if (full_shm_key.empty() || full_shm_key[0] != '/') {
    full_shm_key.insert(0, "/");
  }

  VLOG(1) << "[SHM_ALLOCATOR] Attempting to open/create shm key: "
          << full_shm_key << " of size " << total_size;

  shm_fd_ = shm_open(full_shm_key.c_str(), O_RDWR, 0666);
  bool is_warm_boot = (shm_fd_ >= 0);
  VLOG(2) << "[SHM_DEBUG] shm_open for warm boot key: " << full_shm_key
          << ", fd: " << shm_fd_ << ", errno: " << errno << " ("
          << std::strerror(errno) << ")";

  if (is_warm_boot) {
    VLOG(1)
        << "[SHM_ALLOCATOR] Found existing shm segment. Validating schema...";
    void* header_ptr = mmap(nullptr, sizeof(SharedMemoryHeader),
                            PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (header_ptr != MAP_FAILED) {
      SharedMemoryHeader* header = static_cast<SharedMemoryHeader*>(header_ptr);
      bool compatible = true;
      if (header->magic != expected_schema_.magic) {
        VLOG(1) << "magic mismatch: " << header->magic << " vs "
                << expected_schema_.magic;
        compatible = false;
      }
      if (header->version != expected_schema_.version) {
        VLOG(1) << "version mismatch: " << header->version << " vs "
                << expected_schema_.version;
        compatible = false;
      }
      if (std::strcmp(header->model_uid, expected_schema_.model_uid) != 0) {
        VLOG(1) << "model_uid mismatch: " << header->model_uid << " vs "
                << expected_schema_.model_uid;
        compatible = false;
      }
      if (header->num_blocks != expected_schema_.num_blocks) {
        VLOG(1) << "num_blocks mismatch: " << header->num_blocks << " vs "
                << expected_schema_.num_blocks;
        compatible = false;
      }
      if (header->block_size != expected_schema_.block_size) {
        VLOG(1) << "block_size mismatch: " << header->block_size << " vs "
                << expected_schema_.block_size;
        compatible = false;
      }
      if (header->num_heads != expected_schema_.num_heads) {
        VLOG(1) << "num_heads mismatch: " << header->num_heads << " vs "
                << expected_schema_.num_heads;
        compatible = false;
      }
      if (header->head_dim != expected_schema_.head_dim) {
        VLOG(1) << "head_dim mismatch: " << header->head_dim << " vs "
                << expected_schema_.head_dim;
        compatible = false;
      }
      if (header->itemsize != expected_schema_.itemsize) {
        VLOG(1) << "itemsize mismatch: " << header->itemsize << " vs "
                << expected_schema_.itemsize;
        compatible = false;
      }
      munmap(header_ptr, sizeof(SharedMemoryHeader));

      if (!compatible) {
        VLOG(1) << "[SHM_ALLOCATOR] Existing shm schema incompatible. "
                   "Re-creating...";
        close(shm_fd_);
        shm_fd_ = -1;
        shm_unlink(full_shm_key.c_str());
        is_warm_boot = false;
      }
    } else {
      LOG(ERROR) << "[SHM_ALLOCATOR] Failed to map shm header for verification";
      close(shm_fd_);
      shm_fd_ = -1;
      is_warm_boot = false;
    }
  }

  if (!is_warm_boot) {
    shm_fd_ = shm_open(full_shm_key.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    if (shm_fd_ < 0) {
      shm_fd_ = shm_open(full_shm_key.c_str(), O_CREAT | O_RDWR, 0666);
    }
    if (shm_fd_ < 0) {
      return absl::InternalError(absl::StrCat(
          "shm_open failed to create segment: ", std::strerror(errno)));
    }

    if (ftruncate(shm_fd_, total_size) != 0) {
      close(shm_fd_);
      shm_fd_ = -1;
      shm_unlink(full_shm_key.c_str());
      return absl::InternalError(absl::StrCat(
          "ftruncate failed on shm segment: ", std::strerror(errno)));
    }
  }

  mapped_ptr_ =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (mapped_ptr_ == MAP_FAILED) {
    close(shm_fd_);
    shm_fd_ = -1;
    return absl::ResourceExhaustedError(
        absl::StrCat("mmap failed on shm segment: ", std::strerror(errno)));
  }

  SharedMemoryHeader* header = static_cast<SharedMemoryHeader*>(mapped_ptr_);
  if (!is_warm_boot) {
    VLOG(1) << "[SHM_ALLOCATOR] Initializing fresh shm header schema...";
    std::memcpy(header, &expected_schema_, sizeof(SharedMemoryHeader));
    header->total_payload_bytes = size_bytes;
    header->reference_count = 1;

    volatile uint8_t* p = static_cast<volatile uint8_t*>(mapped_ptr_);
    for (size_t i = 4096; i < total_size; i += 4096) {
      p[i] = 0;
    }
  } else {
    header->reference_count++;
    VLOG(1) << "[SHM_ALLOCATOR] Successfully attached to existing shm segment. "
               "Reference count: "
            << header->reference_count;
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(mapped_ptr_) + sizeof(SharedMemoryHeader);
  alloc.size = size_bytes;
  alloc.owner = std::shared_ptr<void>(mapped_ptr_, [](void*) {});
  return alloc;
}

absl::StatusOr<HostBufferAllocation>
SharedMemoryHostMemoryAllocator::AllocateDmaMapped(size_t size_bytes) {
  auto alloc_or = Allocate(size_bytes);
  if (!alloc_or.ok()) return alloc_or;

  if (!dma_mapped_ && client_ != nullptr && client_->platform_name() != "cpu" &&
      !ShouldSkipDmaMap(client_)) {
    VLOG(1) << "[SHM_ALLOCATOR] Registering shared memory mapping with PjRt "
               "DMA engine...";
    auto status = client_->DmaMap(mapped_ptr_, mapped_size_);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "DmaMap failed on shared memory segment: ", status.message()));
    }
    dma_mapped_ = true;
  }

  return alloc_or;
}

absl::StatusOr<HostBufferAllocation>
SharedMemoryHostMemoryAllocator::AllocateDmaMappedForDevice(
    size_t size_bytes, const xla::PjRtDevice* device) {
  g_current_device = device;
  int numa_node = GetPjRtDeviceNumaNode(device);
  VLOG(1) << "[SHM_ALLOCATOR] Allocation device: "
          << (device ? device->DebugString() : "nullptr")
          << ", NUMA node: " << numa_node;

  if (numa_node >= 0) {
    SetThreadMempolicy(2, numa_node);
    auto alloc_or = AllocateDmaMapped(size_bytes);
    SetThreadMempolicy(0);
    g_current_device = nullptr;
    return alloc_or;
  }

  auto alloc_or = AllocateDmaMapped(size_bytes);
  g_current_device = nullptr;
  return alloc_or;
}

absl::StatusOr<SharedMemoryInfo>
SharedMemoryHostMemoryAllocator::GetSharedMemoryInfo(const void* ptr) const {
  if (mapped_ptr_ == nullptr || mapped_ptr_ == MAP_FAILED) {
    return absl::FailedPreconditionError("Shared memory is not mapped");
  }
  const uint8_t* p = static_cast<const uint8_t*>(ptr);
  const uint8_t* start = static_cast<const uint8_t*>(mapped_ptr_);
  const uint8_t* end = start + mapped_size_;
  if (p < start || p >= end) {
    return absl::InvalidArgumentError(
        "Pointer does not belong to this allocator");
  }
  SharedMemoryInfo info;
  info.shm_key = shm_key_;
  info.size = mapped_size_;
  info.offset = p - start;
  info.base_ptr = mapped_ptr_;
  info.fd = shm_fd_;
  return info;
}

HostBufferAllocator CreateHostMemoryAllocator(xla::PjRtClient* client,
                                              size_t num_blocks,
                                              size_t block_size) {
  SharedMemoryHeader expected_schema;
  expected_schema.magic = 0x52414944454E5348;  // "RAIDENSH"
  expected_schema.version = 1;
  std::strncpy(expected_schema.model_uid, "mock_model_uid",
               sizeof(expected_schema.model_uid));
  expected_schema.global_mesh_shape[0] = 1;  // Single rank dummy mesh
  expected_schema.num_blocks = num_blocks;
  expected_schema.block_size = block_size;
  expected_schema.total_payload_bytes = num_blocks * block_size;
  expected_schema.reference_count = 0;

  const char* shm_key_env = std::getenv("RAIDEN_SHM_KEY");
  std::string shm_key =
      (shm_key_env != nullptr) ? shm_key_env : "raiden_shm_key";

  auto shm_alloc_or =
      SharedMemoryHostMemoryAllocator::Create(client, shm_key, expected_schema);
  if (!shm_alloc_or.ok()) {
    LOG(ERROR) << "Failed to create SharedMemoryHostMemoryAllocator: "
               << shm_alloc_or.status().ToString();
    return nullptr;
  }
  auto shm_alloc = std::shared_ptr<SharedMemoryHostMemoryAllocator>(
      std::move(shm_alloc_or).value());
  return HostBufferAllocator(std::move(shm_alloc));
}

}  // namespace tpu_raiden

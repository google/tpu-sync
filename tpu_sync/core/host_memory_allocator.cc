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
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/tpu_utils.h"

namespace tpu_raiden {
namespace {
thread_local const xla::PjRtDevice* g_current_device = nullptr;
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

  size_t aligned_size = (size_bytes + 4095) & ~4095;

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return absl::ResourceExhaustedError(
        absl::StrCat("mmap failed for size: ", aligned_size));
  }

  auto status = client_->DmaMap(ptr, aligned_size);
  if (!status.ok()) {
    munmap(ptr, aligned_size);
    return absl::InternalError(
        absl::StrCat("DmaMap failed: ", status.message()));
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;

  xla::PjRtClient* client = client_;
  alloc.owner = std::shared_ptr<void>(ptr, [client, aligned_size](void* p) {
    (void)client->DmaUnmap(p);
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

std::string SharedMemoryHostMemoryAllocator::ComposeSegmentName(
    absl::string_view shm_key, std::optional<int64_t> global_device_id) {
  std::string name(shm_key);
  const char* server_name_env = std::getenv("RAIDEN_SHM_SERVER_NAME");
  if (server_name_env != nullptr && std::strlen(server_name_env) > 0) {
    absl::StrAppend(&name, "_", server_name_env);
  }
  if (global_device_id.has_value()) {
    absl::StrAppend(&name, "_dev_", *global_device_id);
  }
  if (name.empty() || name[0] != '/') {
    name.insert(0, "/");
  }
  return name;
}

absl::Status SharedMemoryHostMemoryAllocator::ValidateShmNameParts(
    absl::string_view shm_key) {
  // The same charset SanitizeForShmName (kv_cache_store_wrapper.cc) keeps;
  // there internal identities are rewritten to it, here user configuration
  // is rejected against it.
  auto validate_part = [](absl::string_view part,
                          absl::string_view what) -> absl::Status {
    constexpr size_t kMaxPartLength = 200;
    if (part.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat(what, " must not be empty"));
    }
    if (part.size() > kMaxPartLength) {
      return absl::InvalidArgumentError(absl::StrCat(
          what, "=\"", part, "\" is ", part.size(),
          " characters; a shm segment name allows at most ", kMaxPartLength,
          " here, leaving room for the composed suffixes"));
    }
    for (char c : part) {
      if (!absl::ascii_isalnum(c) && c != '_' && c != '-' && c != '.') {
        return absl::InvalidArgumentError(absl::StrCat(
            what, "=\"", part, "\" contains '",
            absl::CEscape(absl::string_view(&c, 1)),
            "'; a shm segment name allows only letters, digits, '.', '_' and "
            "'-'"));
      }
    }
    return absl::OkStatus();
  };

  absl::Status valid =
      validate_part(absl::StripPrefix(shm_key, "/"), "RAIDEN_SHM_KEY");
  if (!valid.ok()) {
    return valid;
  }
  const char* server_name_env = std::getenv("RAIDEN_SHM_SERVER_NAME");
  if (server_name_env != nullptr && std::strlen(server_name_env) > 0) {
    return validate_part(server_name_env, "RAIDEN_SHM_SERVER_NAME");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SharedMemoryHostMemoryAllocator>>
SharedMemoryHostMemoryAllocator::Create(
    xla::PjRtClient* client, absl::string_view shm_key,
    const SharedMemoryHeader& expected_schema) {
  // Failing here gives the misconfiguration a clear boot-time error instead
  // of a bare shm_open errno at the first allocation.
  absl::Status valid = ValidateShmNameParts(shm_key);
  if (!valid.ok()) {
    return valid;
  }
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

  const std::string full_shm_key = ComposeSegmentName(
      shm_key_, g_current_device != nullptr
                    ? std::optional<int64_t>(
                          g_current_device->global_device_id().value())
                    : std::nullopt);

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

  if (!dma_mapped_ && client_ != nullptr && client_->platform_name() != "cpu") {
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

}  // namespace tpu_raiden

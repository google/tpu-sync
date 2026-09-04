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
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// Reserves [offset, offset + length) of the segment file, extending it if
// needed, so /dev/shm exhaustion surfaces here as an error instead of a
// SIGBUS at first touch.
absl::Status ReserveSegmentRange(int fd, size_t offset, size_t length,
                                 const std::string& segment_name) {
  int err;
  do {
    err = posix_fallocate(fd, offset, length);
  } while (err == EINTR);
  if (err != 0) {
    return absl::ResourceExhaustedError(
        absl::StrCat("posix_fallocate failed on shm segment ", segment_name,
                     ": ", std::strerror(err)));
  }
  return absl::OkStatus();
}

// Compares the identity fields. block_size is deliberately absent: the
// allocator derives it from the first allocation and validates it against
// that derivation (OpenSegment), not against the caller schema, which
// leaves it zero.
bool HeaderMatchesSchema(const SharedMemoryHeader& header,
                         const SharedMemoryHeader& expected) {
  bool compatible = true;
  if (header.magic != expected.magic) {
    VLOG(1) << "magic mismatch: " << header.magic << " vs " << expected.magic;
    compatible = false;
  }
  if (header.version != expected.version) {
    VLOG(1) << "version mismatch: " << header.version << " vs "
            << expected.version;
    compatible = false;
  }
  if (std::strcmp(header.model_uid, expected.model_uid) != 0) {
    VLOG(1) << "model_uid mismatch: " << header.model_uid << " vs "
            << expected.model_uid;
    compatible = false;
  }
  if (header.num_blocks != expected.num_blocks) {
    VLOG(1) << "num_blocks mismatch: " << header.num_blocks << " vs "
            << expected.num_blocks;
    compatible = false;
  }
  if (header.num_heads != expected.num_heads) {
    VLOG(1) << "num_heads mismatch: " << header.num_heads << " vs "
            << expected.num_heads;
    compatible = false;
  }
  if (header.head_dim != expected.head_dim) {
    VLOG(1) << "head_dim mismatch: " << header.head_dim << " vs "
            << expected.head_dim;
    compatible = false;
  }
  if (header.itemsize != expected.itemsize) {
    VLOG(1) << "itemsize mismatch: " << header.itemsize << " vs "
            << expected.itemsize;
    compatible = false;
  }
  return compatible;
}
}  // namespace

absl::StatusOr<ScopedShmLock> ScopedShmLock::Acquire(
    absl::string_view segment_name) {
  const std::string lock_name = absl::StrCat(segment_name, ".lock");
  int fd = shm_open(lock_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd < 0) {
    return absl::InternalError(absl::StrCat("shm_open failed on lock file ",
                                            lock_name, ": ",
                                            std::strerror(errno)));
  }
  int err;
  do {
    err = flock(fd, LOCK_EX);
  } while (err != 0 && errno == EINTR);
  if (err != 0) {
    absl::Status status = absl::InternalError(absl::StrCat(
        "flock failed on lock file ", lock_name, ": ", std::strerror(errno)));
    close(fd);
    return status;
  }
  return ScopedShmLock(fd);
}

ScopedShmLock::~ScopedShmLock() {
  if (fd_ >= 0) {
    // Closing the descriptor releases the flock.
    close(fd_);
  }
}

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
  // The same charset SanitizeForShmName (kv_cache_metadata_shm.cc) keeps;
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
  for (auto& [name, segment] : segments_) {
    for (PayloadRegion& region : segment.regions) {
      if (region.dma_mapped && client_ != nullptr) {
        (void)client_->DmaUnmap(region.ptr);
      }
      munmap(region.ptr, region.size_bytes);
    }
    if (segment.header != nullptr) {
      // The shared refcount needs the same cross-process lock as
      // OpenSegment's increment. A destructor cannot fail, so a failed
      // acquire only costs the decrement's atomicity.
      absl::StatusOr<ScopedShmLock> lock = ScopedShmLock::Acquire(name);
      if (!lock.ok()) {
        LOG(WARNING) << "[SHM_ALLOCATOR] Releasing segment " << name
                     << " without its creation lock: " << lock.status();
      }
      if (segment.header->reference_count > 0) {
        segment.header->reference_count--;
      }
      munmap(segment.header, kPayloadOffset);
    }
    if (segment.fd >= 0) {
      close(segment.fd);
    }
    // Never shm_unlink: the segment must outlive this process for the next
    // run to recover from.
  }
}

absl::StatusOr<SharedMemoryHostMemoryAllocator::Segment*>
SharedMemoryHostMemoryAllocator::OpenSegment(const std::string& segment_name,
                                             size_t first_request_bytes) {
  VLOG(1) << "[SHM_ALLOCATOR] Opening shm segment: " << segment_name;

  // The whole warm-or-cold decision -- open, validate, possibly unlink and
  // re-create -- runs under the segment's creation lock, so a concurrent
  // opener cannot unlink a live segment whose header is not written yet.
  absl::StatusOr<ScopedShmLock> lock = ScopedShmLock::Acquire(segment_name);
  if (!lock.ok()) {
    return lock.status();
  }

  const size_t aligned_request = (first_request_bytes + 4095) & ~4095;
  // The request is num_blocks * bytes-per-block, so a pinned block count
  // recovers the bytes-per-block factor of the region stride that the
  // identity fields alone cannot.
  uint64_t bytes_per_block = 0;
  if (expected_schema_.num_blocks > 0 &&
      first_request_bytes % expected_schema_.num_blocks == 0) {
    bytes_per_block = first_request_bytes / expected_schema_.num_blocks;
    if (bytes_per_block > std::numeric_limits<uint32_t>::max()) {
      bytes_per_block = 0;
    }
  }

  int fd = shm_open(segment_name.c_str(), O_RDWR, 0666);
  bool warm = (fd >= 0);
  SharedMemoryHeader* header = nullptr;

  if (warm) {
    VLOG(1)
        << "[SHM_ALLOCATOR] Found existing shm segment. Validating schema...";
    struct stat st;
    bool compatible =
        fstat(fd, &st) == 0 && st.st_size >= static_cast<off_t>(kPayloadOffset);
    if (compatible) {
      void* header_ptr = mmap(nullptr, kPayloadOffset, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, 0);
      if (header_ptr == MAP_FAILED) {
        LOG(ERROR)
            << "[SHM_ALLOCATOR] Failed to map shm header for verification";
        compatible = false;
      } else {
        header = static_cast<SharedMemoryHeader*>(header_ptr);
        compatible = HeaderMatchesSchema(*header, expected_schema_);
        // The file must still hold the recorded payload. Unsigned space:
        // the recorded total is untrusted, and a huge value cast to off_t
        // would slip past a signed comparison.
        if (compatible &&
            header->total_payload_bytes >
                static_cast<uint64_t>(st.st_size) - kPayloadOffset) {
          VLOG(1) << "recorded payload " << header->total_payload_bytes
                  << " exceeds the file size " << st.st_size;
          compatible = false;
        }
        if (compatible) {
          if (bytes_per_block > 0) {
            // Exact stride check: the predecessor's regions must have been
            // this run's size, or their bytes belong to other offsets.
            if (header->block_size != bytes_per_block) {
              VLOG(1) << "block size mismatch: " << header->block_size
                      << " vs derived " << bytes_per_block;
              compatible = false;
            }
          } else if (header->total_payload_bytes % aligned_request != 0) {
            // Without a pinned block count, whole-multiple divisibility is
            // the only replay tripwire available.
            VLOG(1) << "recorded payload " << header->total_payload_bytes
                    << " is not a multiple of the request "
                    << aligned_request;
            compatible = false;
          }
        }
        if (!compatible) {
          munmap(header_ptr, kPayloadOffset);
          header = nullptr;
        }
      }
    }
    if (!compatible) {
      VLOG(1) << "[SHM_ALLOCATOR] Existing shm segment incompatible. "
                 "Re-creating...";
      close(fd);
      fd = -1;
      shm_unlink(segment_name.c_str());
      warm = false;
    } else {
      header->reference_count++;
      VLOG(1) << "[SHM_ALLOCATOR] Successfully attached to existing shm "
                 "segment. Reference count: "
              << header->reference_count;
    }
  }

  if (!warm) {
    // Under the creation lock an EEXIST can only be an uncooperative
    // writer: fail loudly rather than overwrite its header.
    fd = shm_open(segment_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
      return absl::InternalError(
          absl::StrCat("shm_open failed to create segment ", segment_name,
                       ": ", std::strerror(errno)));
    }
    absl::Status reserved =
        ReserveSegmentRange(fd, 0, kPayloadOffset, segment_name);
    if (!reserved.ok()) {
      close(fd);
      shm_unlink(segment_name.c_str());
      return reserved;
    }
    void* header_ptr = mmap(nullptr, kPayloadOffset, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
    if (header_ptr == MAP_FAILED) {
      close(fd);
      shm_unlink(segment_name.c_str());
      return absl::ResourceExhaustedError(absl::StrCat(
          "mmap failed on shm segment ", segment_name, ": ",
          std::strerror(errno)));
    }
    VLOG(1) << "[SHM_ALLOCATOR] Initializing fresh shm header schema...";
    header = static_cast<SharedMemoryHeader*>(header_ptr);
    std::memcpy(header, &expected_schema_, sizeof(SharedMemoryHeader));
    header->block_size = static_cast<uint32_t>(bytes_per_block);
    header->total_payload_bytes = 0;
    header->reference_count = 1;
  }

  Segment& segment = segments_[segment_name];
  segment.fd = fd;
  segment.warm = warm;
  segment.header = header;
  return &segment;
}

absl::StatusOr<SharedMemoryHostMemoryAllocator::PayloadRegion*>
SharedMemoryHostMemoryAllocator::AllocateRegion(size_t size_bytes) {
  const size_t aligned_size = (size_bytes + 4095) & ~4095;
  const std::string segment_name = ComposeSegmentName(
      shm_key_, g_current_device != nullptr
                    ? std::optional<int64_t>(
                          g_current_device->global_device_id().value())
                    : std::nullopt);

  Segment* segment;
  auto it = segments_.find(segment_name);
  if (it != segments_.end()) {
    // A later allocation against a device whose segment an earlier call
    // already opened (e.g. the next array or shard on that device): reuse
    // the open segment and its warm/cold decision.
    segment = &it->second;
  } else {
    // The first allocation against this device: open its segment, deciding
    // warm or cold for the rest of the run.
    auto segment_or = OpenSegment(segment_name, size_bytes);
    if (!segment_or.ok()) {
      return segment_or.status();
    }
    segment = *segment_or;
  }

  if (segment->region_size == 0) {
    segment->region_size = aligned_size;
  } else if (aligned_size != segment->region_size) {
    return absl::InternalError(absl::StrCat(
        "shm segment ", segment_name, " serves ", segment->region_size,
        "-byte regions; refusing a ", aligned_size,
        "-byte allocation, the recovery layout assumes one uniform size"));
  }

  const size_t region_offset = kPayloadOffset + segment->payload_cursor;
  bool zero_fill = !segment->warm;
  if (segment->warm && segment->payload_cursor + aligned_size >
                           segment->header->total_payload_bytes) {
    if (!segment->degradation_warned) {
      LOG(WARNING) << "[SHM_ALLOCATOR] Segment " << segment_name
                   << " recorded " << segment->header->total_payload_bytes
                   << " payload bytes; allocations beyond them are served "
                      "zeroed instead of recovered";
      segment->degradation_warned = true;
    } else {
      VLOG(1) << "[SHM_ALLOCATOR] Region at offset " << region_offset
              << " of segment " << segment_name << " served zeroed";
    }
    zero_fill = true;
  }
  if (zero_fill) {
    absl::Status reserved = ReserveSegmentRange(segment->fd, region_offset,
                                                aligned_size, segment_name);
    if (!reserved.ok()) {
      return reserved;
    }
  }

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   segment->fd, region_offset);
  if (ptr == MAP_FAILED) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "mmap failed on shm segment ", segment_name, ": ",
        std::strerror(errno)));
  }

  if (zero_fill) {
    // First-touch under the caller's NUMA mempolicy places the pages; a warm
    // region is never touched, its bytes are the recovered data.
    volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
    for (size_t i = 0; i < aligned_size; i += 4096) {
      p[i] = 0;
    }
  }

  segment->payload_cursor += aligned_size;
  if (segment->payload_cursor > segment->header->total_payload_bytes) {
    segment->header->total_payload_bytes = segment->payload_cursor;
  }
  segment->regions.push_back(PayloadRegion{ptr, aligned_size, false});
  return &segment->regions.back();
}

absl::StatusOr<HostBufferAllocation> SharedMemoryHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  if (size_bytes == 0) {
    HostBufferAllocation alloc;
    alloc.ptr = nullptr;
    alloc.size = 0;
    return alloc;
  }

  auto region_or = AllocateRegion(size_bytes);
  if (!region_or.ok()) return region_or.status();

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>((*region_or)->ptr);
  alloc.size = size_bytes;
  alloc.owner = std::shared_ptr<void>((*region_or)->ptr, [](void*) {});
  return alloc;
}

absl::StatusOr<HostBufferAllocation>
SharedMemoryHostMemoryAllocator::AllocateDmaMapped(size_t size_bytes) {
  if (size_bytes == 0) {
    HostBufferAllocation alloc;
    alloc.ptr = nullptr;
    alloc.size = 0;
    return alloc;
  }

  auto region_or = AllocateRegion(size_bytes);
  if (!region_or.ok()) return region_or.status();
  PayloadRegion* region = *region_or;

  if (!region->dma_mapped && client_ != nullptr &&
      client_->platform_name() != "cpu") {
    VLOG(1) << "[SHM_ALLOCATOR] Registering shared memory region with PjRt "
               "DMA engine...";
    auto status = client_->DmaMap(region->ptr, region->size_bytes);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "DmaMap failed on shared memory region: ", status.message()));
    }
    region->dma_mapped = true;
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(region->ptr);
  alloc.size = size_bytes;
  alloc.owner = std::shared_ptr<void>(region->ptr, [](void*) {});
  return alloc;
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

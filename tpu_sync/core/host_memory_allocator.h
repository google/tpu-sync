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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

struct HostBufferAllocation {
  uint8_t* ptr = nullptr;
  size_t size = 0;
  std::shared_ptr<void> owner;
};

// Returns a HostBufferAllocation of at least the requested size for a given
// device. If device is nullptr, it allocates on the default/current NUMA node.
// On failure, returns a non-OK status.
using HostBufferAllocator = std::function<absl::StatusOr<HostBufferAllocation>(
    size_t, const xla::PjRtDevice*)>;

// High-performance host memory allocator that allocates DMA-capable pinned
// memory using PJRT APIs or standard fallback allocations.
class HostMemoryAllocator {
 public:
  virtual ~HostMemoryAllocator() = default;

  static absl::StatusOr<std::unique_ptr<HostMemoryAllocator>> Create(
      xla::PjRtClient* pjrt_client);

  virtual absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) = 0;

  // Allocates host memory registered with the device DMA engine
  // (PjRtClient::DmaMap), so raw C-API D2H/H2D into it run as a true async DMA
  // instead of a synchronous staged copy (the producer-D2H bottleneck). The
  // default implementation falls back to Allocate (e.g. CPU / no DMA support).
  virtual absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) {
    return Allocate(size_bytes);
  }

  // Device-aware version of AllocateDmaMapped. Attempts to allocate host
  // memory on the NUMA node local to the given device.
  virtual absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) {
    return AllocateDmaMapped(size_bytes);
  }
};

class XlaHostMemoryAllocator : public HostMemoryAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<XlaHostMemoryAllocator>> Create(
      xla::PjRtClient* absl_nonnull pjrt_client);

  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;

  absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) override;

  absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) override;

 private:
  explicit XlaHostMemoryAllocator(xla::PjRtClient* client);

  xla::PjRtClient* client_ = nullptr;
};

class MallocHostMemoryAllocator : public HostMemoryAllocator {
 public:
  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;
};

// Serializes attach-or-create decisions on a shared-memory segment across
// processes: without it, a process can open a peer's segment before its
// header is written, judge it incompatible, and unlink a live segment.
// Locks via flock(2) on a companion "<name>.lock" file that is never
// unlinked, so its identity survives the guarded segment's re-creation;
// the kernel releases the lock when its holder exits, so a crashed holder
// cannot wedge later boots. Taken only when a segment is opened or
// released -- once per segment per process lifetime, at boot and shutdown
// -- so it adds no cost to the allocation or serving paths; contention
// only serializes same-host processes booting against the same segment
// name, for the microseconds the peer needs to set up a header.
class ScopedShmLock {
 public:
  // Blocks until the lock guarding `segment_name` is held.
  static absl::StatusOr<ScopedShmLock> Acquire(absl::string_view segment_name);

  ScopedShmLock(ScopedShmLock&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
  }
  ScopedShmLock(const ScopedShmLock&) = delete;
  ScopedShmLock& operator=(const ScopedShmLock&) = delete;
  ScopedShmLock& operator=(ScopedShmLock&&) = delete;
  ~ScopedShmLock();

 private:
  explicit ScopedShmLock(int fd) : fd_(fd) {}
  int fd_ = -1;
};

struct alignas(64) SharedMemoryHeader {
  uint64_t magic = 0x52414944454E5348;  // "RAIDENSH"
  // Segment layout revision; a mismatch cold-starts the segment. Version 2
  // placed each allocation in its own page-aligned region after the header
  // page (version 1 segments held a single payload right after the header).
  uint32_t version = 2;
  char model_uid[256] = {0};
  uint32_t global_mesh_shape[5] = {0};
  uint32_t shard_layout[5] = {0};
  // Number of blocks in the host pool this segment backs;
  uint32_t num_blocks = 0;
  // Bytes per block, derived by the allocator from its first allocation
  // (0 when num_blocks is unknown).
  uint32_t block_size = 0;
  uint32_t num_heads = 0;
  uint32_t head_dim = 0;
  uint32_t itemsize = 0;
  // Cumulative payload bytes ever allocated from this segment. A warm boot
  // replays its allocations in order against this bound.
  uint64_t total_payload_bytes = 0;
  uint32_t reference_count = 0;
};

// Persists host KV mirrors across process restarts in POSIX shared memory.
// Each device gets one segment (ComposeSegmentName); within it, allocations
// occupy consecutive page-aligned regions after the header page, in
// allocation order. Whether a run is warm (re-attaching to a predecessor's
// data) or cold is decided once, when the segment is first opened; every
// later allocation follows that decision.
class SharedMemoryHostMemoryAllocator : public HostMemoryAllocator {
 public:
  // Payload regions start page-aligned after the header so that each can be
  // mmapped individually; the header owns the segment's first page.
  static constexpr size_t kPayloadOffset = 4096;

  static absl::StatusOr<std::unique_ptr<SharedMemoryHostMemoryAllocator>>
  Create(xla::PjRtClient* client, absl::string_view shm_key,
         const SharedMemoryHeader& expected_schema);

  // The shm_open name for this allocator's segment: `shm_key` plus an
  // optional "_<RAIDEN_SHM_SERVER_NAME>" suffix from the environment, plus
  // "_dev_<global_device_id>" when a device is given, "/"-prefixed.
  static std::string ComposeSegmentName(
      absl::string_view shm_key, std::optional<int64_t> global_device_id);

  // Validates the user-supplied segment-name inputs: `shm_key` (a
  // RAIDEN_SHM_KEY value; one leading '/' is tolerated as the shm_open name
  // prefix) and, when set, the RAIDEN_SHM_SERVER_NAME environment value.
  // Each must be non-empty, at most 200 characters (leaving NAME_MAX room
  // for the composed suffixes), and hold only letters, digits, '.', '_' and
  // '-' -- shm names are single path components, so '/' in particular makes
  // shm_open fail. Violations are an InvalidArgumentError naming the
  // variable and the offending character; user configuration is never
  // silently rewritten.
  static absl::Status ValidateShmNameParts(absl::string_view shm_key);

  ~SharedMemoryHostMemoryAllocator() override;

  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;
  absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) override;
  absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) override;

 private:
  SharedMemoryHostMemoryAllocator(xla::PjRtClient* client,
                                  absl::string_view shm_key,
                                  const SharedMemoryHeader& expected_schema);

  // One payload region per Allocate() call, mmapped individually at its
  // page-aligned offset within the segment.
  struct PayloadRegion {
    void* ptr = nullptr;
    size_t size_bytes = 0;
    bool dma_mapped = false;
  };

  struct Segment {
    int fd = -1;
    bool warm = false;
    // Payload bytes handed out from this segment so far in this run.
    size_t payload_cursor = 0;
    // The uniform page-aligned per-allocation size, set by the first
    // allocation; the recovery layout assumes it, so later allocations must
    // match it.
    size_t region_size = 0;
    // The warm->degraded transition has been logged for this segment.
    bool degradation_warned = false;
    // The segment's first page, mapped for the allocator's lifetime.
    SharedMemoryHeader* header = nullptr;
    std::vector<PayloadRegion> regions;
  };

  // Opens the named segment, deciding warm (exists and matches
  // expected_schema_) or cold (created, or reformatted on a mismatch) for
  // the rest of the run. first_request_bytes is the first allocation's
  // requested size, for validating a warm segment's recorded layout against
  // this run's replay.
  absl::StatusOr<Segment*> OpenSegment(const std::string& segment_name,
                                       size_t first_request_bytes);

  // Bump-allocates the next page-aligned region of the per-device segment,
  // opening the segment on its first allocation.
  absl::StatusOr<PayloadRegion*> AllocateRegion(size_t size_bytes);

  xla::PjRtClient* client_ = nullptr;
  std::string shm_key_;
  SharedMemoryHeader expected_schema_;
  // Keyed by segment name; absl::node_hash_map for pointer stability of the
  // values.
  absl::node_hash_map<std::string, Segment> segments_;
};

static_assert(sizeof(SharedMemoryHeader) <=
                  SharedMemoryHostMemoryAllocator::kPayloadOffset,
              "the segment header must fit in the page before the payload");

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_

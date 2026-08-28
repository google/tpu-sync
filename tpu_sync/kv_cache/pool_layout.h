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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_POOL_LAYOUT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_POOL_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

// A strided run of live bytes inside one pool block. Byte-level and
// model-agnostic: callers describe interior layout, raiden only stores and
// checks it.
struct RegionSpec {
  std::string name;
  int64_t offset_bytes = 0;
  int64_t stride_bytes = 0;
  int64_t unit_bytes = 0;
  int64_t num_units = 0;
  int64_t units_per_stride = 1;

  int64_t live_bytes() const;
  int64_t extent_end_bytes() const;
  absl::Status Validate(int64_t slot_bytes) const;
};

// One block pool inside a wrapped storage. A storage is a whole wrapped
// allocation (constructor order gives storage_index); a pool is an array of
// num_blocks equally strided blocks starting at base_offset_bytes, each
// carrying the same regions of live bytes. tag and dtype_tag are opaque:
// raiden stores, filters, and echoes them but never parses them.
struct PoolSpec {
  std::string tag;
  size_t storage_index = 0;
  int64_t base_offset_bytes = 0;
  int64_t block_stride_bytes = 0;
  int64_t num_blocks = 0;
  std::vector<RegionSpec> regions;
  std::string dtype_tag;
  // Max blocks of this pool one transfer can touch (FA: pages per max-length
  // request; GDN state: one block per group). When every pool of a storage
  // declares a positive value and RegisterPools is given a lease count, that
  // storage's host mirror becomes a bounded staging arena of
  // leases x max(hint) slots instead of a full shadow of the device pool.
  // 0 (default) keeps the full mirror.
  int64_t staging_blocks_per_request = 0;

  // Absolute end of the last declared live byte in the backing storage.
  // Inter-block and trailing padding are not part of the pool's addressable
  // storage extent.
  int64_t storage_extent_end_bytes() const;

  // Validates internal consistency and, when storage_bytes >= 0, that the
  // live regions of every block fit inside the storage.
  absl::Status Validate(int64_t storage_bytes) const;
};

// Host-side reference to one block of one pool. Only pool->regions are
// addressable; the final block's trailing stride padding need not be backed.
struct PoolBlockRef {
  uint8_t* ptr = nullptr;
  int64_t block_stride_bytes = 0;
  const PoolSpec* pool = nullptr;
  size_t pool_idx = 0;
  size_t shard_idx = 0;
  int64_t block_id = 0;
};

// True when regions jointly cover every byte of [start, end) within a block.
bool RegionsCoverRange(const std::vector<RegionSpec>& regions, size_t start,
                       size_t end);

// One contiguous byte range inside a pool's storage. Host mirrors share the
// storage layout, so the same offset addresses both sides of a D2H/H2D copy.
struct PoolBlockCopyExtent {
  int64_t offset_bytes = 0;
  int64_t size_bytes = 0;
};

// Declared-live copy extents for the given block ids of one pool. Adjacent or
// overlapping regions are coalesced; padding is never returned. Rejects
// out-of-range ids and overflowing byte geometry.
absl::StatusOr<std::vector<PoolBlockCopyExtent>> ComputePoolBlockCopyExtents(
    const PoolSpec& pool, absl::Span<const int64_t> block_ids);

// One coalesced run of live bytes inside a pool block, mapping compact-live
// (logical) offsets to physical block offsets. Logical offsets concatenate
// the runs in physical-address order — the byte-span plan's destination
// coordinate space.
struct PoolLiveSegment {
  int64_t logical_offset = 0;
  int64_t physical_offset = 0;
  int64_t size = 0;

  friend bool operator==(const PoolLiveSegment& lhs,
                         const PoolLiveSegment& rhs) {
    return lhs.logical_offset == rhs.logical_offset &&
           lhs.physical_offset == rhs.physical_offset && lhs.size == rhs.size;
  }
};

// Expands a pool's regions into non-overlapping physical live runs with
// compact-live offsets. Adjacent physical runs are coalesced; overlapping
// regions and runs exceeding the block stride are rejected.
absl::StatusOr<std::vector<PoolLiveSegment>> ExpandPoolLiveSegments(
    const PoolSpec& pool);

// Maps one gap-free physical byte range to its compact-live [start, end)
// interval. Fails when the range crosses padding or lies outside the
// declared live runs.
absl::StatusOr<std::pair<int64_t, int64_t>> PhysicalLiveRangeToLogical(
    absl::Span<const PoolLiveSegment> segments, int64_t physical_offset,
    int64_t size);

// One physical chunk of a compact-live copy split at source or destination
// physical gaps (the forward direction of PhysicalLiveRangeToLogical; the
// reshard planner's translation step).
struct LiveCopyChunk {
  int64_t src_physical = 0;
  int64_t dst_physical = 0;
  int64_t size = 0;

  friend bool operator==(const LiveCopyChunk& lhs, const LiveCopyChunk& rhs) {
    return lhs.src_physical == rhs.src_physical &&
           lhs.dst_physical == rhs.dst_physical && lhs.size == rhs.size;
  }
};

// Splits one compact-live copy of `size` bytes from src_offset/dst_offset
// (both compact-live coordinates) into physical chunks, breaking at every
// source or destination segment boundary. Port of
// rpc/raiden_controller.py::_translate_live_copy, byte-identical semantics
// including the 2^20 chunk expansion bound.
absl::StatusOr<std::vector<LiveCopyChunk>> TranslateLiveCopy(
    absl::Span<const PoolLiveSegment> src_segments,
    absl::Span<const PoolLiveSegment> dst_segments, int64_t src_offset,
    int64_t dst_offset, int64_t size);

tpu_sync::rpc::RegionSpecProto ToProto(const RegionSpec& region);
absl::StatusOr<RegionSpec> RegionSpecFromProto(
    const tpu_sync::rpc::RegionSpecProto& proto);

tpu_sync::rpc::PoolSpecProto ToProto(const PoolSpec& pool);
absl::StatusOr<PoolSpec> PoolSpecFromProto(
    const tpu_sync::rpc::PoolSpecProto& proto);

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_POOL_LAYOUT_H_

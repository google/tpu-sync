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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_LOGICAL_RESHARD_PLANNER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_LOGICAL_RESHARD_PLANNER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

// How the Trainer seeds the destination replicas (see `SeedLayout`).
struct SeedingOptions {
  // Number of stripes G. 0 picks `min(#bundles, D / replication)`, which
  // seeds every replica (when `D` is a multiple of the replication) with one
  // stripe.
  int32_t num_stripes = 0;
  // Number of replicas every stripe is pushed to (R), clamped to [1, D].
  int32_t replication = 2;
  // Number of concurrent Trainer push streams k (pushes per wave). 0 derives
  // it from the broadcast host ratio (`ComputeSeedSamplerCount`).
  int32_t trainer_streams = 0;
  // Optional permutation of [0, D): `replica_order[i]` is the replica index of
  // the `i`-th destination unit in sorted unit order. Seeds are assigned by
  // position in this order, so they stay stable when callers list the same
  // units in a different order. Empty means the identity.
  std::vector<int32_t> replica_order;
};

// Computes logical N-D shard intersections, strided/tile-aware byte copy
// schedules, cross-variable plan deduplication, byte-balanced variable
// bundle partitioning and the Trainer seed layout.
//
// Thread-safe: all methods are stateless or synchronize internal plan caches.
class LogicalReshardPlanner {
 public:
  LogicalReshardPlanner() = default;

  // Computes how many destination replicas $k$ the Trainer can push to
  // concurrently so that aggregated Trainer egress bandwidth matches the
  // receivers' aggregated ingress bandwidth:
  //   k = max(1, min(num_dst_replicas,
  //                  floor(broadcast_host_ratio * H_trainer / H_sampler)))
  // When |broadcast_host_ratio| <= 0.0, returns |num_dst_replicas| (push to
  // all replicas at once). Used as the default number of Trainer streams.
  static int32_t ComputeSeedSamplerCount(int32_t num_dst_replicas,
                                         double broadcast_host_ratio,
                                         int32_t trainer_hosts,
                                         int32_t sampler_hosts);

  // Converts a linear logical shard index |idx| into row-major N-D mesh
  // coordinates for |shape|.
  static std::vector<int64_t> GetGlobalIndices(int64_t idx,
                                               absl::Span<const int64_t> shape);

  // Computes the N-D tensor slice owned by each logical shard in `[0,
  // prod(mesh_shape))` (row-major over |mesh_shape|) for a tensor of
  // |global_shape| sharded according to |layout| across |mesh_shape|.
  static absl::StatusOr<std::vector<NdSlice>> ComputeNdShardSlices(
      absl::Span<const int64_t> global_shape,
      absl::Span<const int64_t> mesh_shape, absl::Span<const int64_t> layout);

  // Returns the N-D intersection of |slice_a| and |slice_b|, or `std::nullopt`
  // if the two slices do not overlap.
  static std::optional<NdSlice> IntersectNdSlices(const NdSlice& slice_a,
                                                  const NdSlice& slice_b);

  // Permutes |slice| dimensions from logical tensor order into major-to-minor
  // physical layout order according to minor-to-major |layout|.
  static NdSlice ToPhysical(const NdSlice& slice,
                            absl::Span<const int64_t> layout);

  // Returns true if |intersection| is aligned to TPU `(8, 128)` tiles within
  // both |src_slice| and |dst_slice|.
  static bool IsNdSliceTileAligned(const NdSlice& src_slice,
                                   const NdSlice& dst_slice,
                                   const NdSlice& intersection,
                                   absl::Span<const int64_t> src_layout,
                                   absl::Span<const int64_t> dst_layout);

  // Generates row-major contiguous or 2D-strided byte copy chunks for moving
  // |intersection| from |src_slice| to |dst_slice|.
  static std::vector<StridedCopyChunk> GenerateStridedCopyChunks(
      const NdSlice& src_slice, const NdSlice& dst_slice,
      const NdSlice& intersection, absl::Span<const int64_t> src_layout,
      absl::Span<const int64_t> dst_layout, int64_t itemsize);

  // Generates TPU `(8, 128)` tile-aware byte copy chunks for moving
  // |intersection| from |src_slice| to |dst_slice|.
  static std::vector<StridedCopyChunk> GenerateStridedCopyChunksTileAware(
      const NdSlice& src_slice, const NdSlice& dst_slice,
      const NdSlice& intersection, absl::Span<const int64_t> src_layout,
      absl::Span<const int64_t> dst_layout, int64_t itemsize);

  // Partitions |num_layers| variables with per-shard byte sizes
  // |dst_layer_shard_bytes| into at most |num_bundle_groups| contiguous
  // variable bundles balanced by byte volume.
  static std::vector<VariableBundleSpec> PartitionVariableBundles(
      int32_t num_layers, absl::Span<const int64_t> dst_layer_shard_bytes,
      int32_t num_bundle_groups = 8);

  // Builds the stripes, the stable seed assignment and the replica-major
  // Trainer waves for bundles of |bundle_bytes| and |num_dst_replicas|
  // destination replicas:
  //   - replication R = clamp(options.replication, 1, D);
  //   - stripes G = options.num_stripes (0 = auto: min(B, D / R)), clamped to
  //     [1, min(B, D / R)], each a contiguous byte-balanced bundle range;
  //   - seed slot `g * R + c` (copy `c` of stripe `g`) is the replica at
  //     position `(g * R + c) * D / (G * R)` of the sorted replica order
  //     (`options.replica_order`, identity if empty), so seeds only change
  //     when the membership changes;
  //   - waves hold at most |trainer_streams| (>= 1) pushes, copy 0 of every
  //     stripe first.
  static absl::StatusOr<SeedLayout> BuildSeedLayout(
      absl::Span<const int64_t> bundle_bytes, int32_t num_dst_replicas,
      const SeedingOptions& options, int32_t trainer_streams);

  // Computes a complete `LogicalReshardSchedule`: deduplicated per-variable
  // copy plans, `num_bundle_groups` bundles (raised to at least the requested
  // number of stripes) and the seed layout. The number of Trainer streams is
  // |seeding.trainer_streams| or, if 0, `ComputeSeedSamplerCount(
  // num_dst_replicas, broadcast_host_ratio, trainer_hosts, sampler_hosts)`.
  static absl::StatusOr<LogicalReshardSchedule> ComputeLogicalSchedule(
      absl::Span<const VariableSpec> src_variables,
      absl::Span<const VariableSpec> dst_variables, int32_t num_src_shards,
      int32_t num_dst_shards, int32_t num_dst_replicas,
      double broadcast_host_ratio, int32_t trainer_hosts, int32_t sampler_hosts,
      int32_t num_bundle_groups = 8, const SeedingOptions& seeding = {});
};

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_LOGICAL_RESHARD_PLANNER_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_LOGICAL_TYPES_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_LOGICAL_TYPES_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

// Represents a logical N-dimensional tensor slice via per-dimension start
// |offsets| and extent |sizes|.
struct NdSlice {
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;

  bool operator==(const NdSlice& other) const {
    return offsets == other.offsets && sizes == other.sizes;
  }

  bool operator!=(const NdSlice& other) const { return !(*this == other); }

  bool IsEmpty() const {
    if (sizes.empty() && offsets.empty()) return true;
    for (int64_t s : sizes) {
      if (s <= 0) return true;
    }
    return false;
  }

  int64_t NumElements() const {
    if (sizes.empty()) return 1;
    int64_t prod = 1;
    for (int64_t s : sizes) {
      if (s <= 0) return 0;
      prod *= s;
    }
    return prod;
  }

  template <typename H>
  friend H AbslHashValue(H h, const NdSlice& slice) {
    return H::combine(std::move(h), slice.offsets, slice.sizes);
  }
};

// Represents a contiguous or 2D-strided byte copy operation between a source
// shard buffer and a destination shard buffer.
struct StridedCopyChunk {
  int64_t src_offset_bytes = 0;
  int64_t dst_offset_bytes = 0;
  int64_t size_bytes = 0;
  int64_t src_stride_bytes = 0;
  int64_t dst_stride_bytes = 0;
  int64_t count = 1;

  bool operator==(const StridedCopyChunk& other) const {
    return std::tie(src_offset_bytes, dst_offset_bytes, size_bytes,
                    src_stride_bytes, dst_stride_bytes, count) ==
           std::tie(other.src_offset_bytes, other.dst_offset_bytes,
                    other.size_bytes, other.src_stride_bytes,
                    other.dst_stride_bytes, other.count);
  }

  bool operator!=(const StridedCopyChunk& other) const {
    return !(*this == other);
  }

  template <typename H>
  friend H AbslHashValue(H h, const StridedCopyChunk& chunk) {
    return H::combine(std::move(h), chunk.src_offset_bytes,
                      chunk.dst_offset_bytes, chunk.size_bytes,
                      chunk.src_stride_bytes, chunk.dst_stride_bytes,
                      chunk.count);
  }
};

// Describes the sharding topology and byte layout of a single weight variable.
// `global_shape`, `mesh_shape` and `layout` define the logical shards
// (row-major over `mesh_shape`), and `global_shard_indices[i]` is the logical
// shard held by the unit's local shard `i`.
struct VariableSpec {
  std::string name;
  std::vector<int64_t> global_shape;
  std::vector<int64_t> mesh_shape;
  std::vector<int64_t> layout;
  int64_t itemsize = 0;
  int32_t layer_idx = 0;
  std::vector<int64_t> global_shard_indices;
  bool skip_tiling = false;

  bool operator==(const VariableSpec& other) const {
    return std::tie(name, global_shape, mesh_shape, layout, itemsize, layer_idx,
                    global_shard_indices, skip_tiling) ==
           std::tie(other.name, other.global_shape, other.mesh_shape,
                    other.layout, other.itemsize, other.layer_idx,
                    other.global_shard_indices, other.skip_tiling);
  }

  bool operator!=(const VariableSpec& other) const { return !(*this == other); }
};

// Groups a contiguous subset of model weight variables (`layer_indices`) into a
// coarse-grained unit of Trainer seeding (stripes are bundle ranges) and of
// sampler-to-sampler pulls.
struct VariableBundleSpec {
  int32_t bundle_id = 0;
  std::vector<int32_t> layer_indices;
  int64_t total_bytes = 0;
  std::vector<int64_t> layer_byte_sizes;
};

// Represents one shard-to-shard copy action targeting a specific destination
// sampler replica and destination global shard index.
struct ShardCopyAction {
  int32_t dst_replica_idx = 0;
  int32_t dst_shard_idx = 0;
  int64_t src_offset_bytes = 0;
  int64_t dst_offset_bytes = 0;
  int64_t size_bytes = 0;
  int64_t src_stride_bytes = 0;
  int64_t dst_stride_bytes = 0;
  int64_t count = 1;
  int32_t plan_id = 0;

  bool operator==(const ShardCopyAction& other) const {
    return std::tie(dst_replica_idx, dst_shard_idx, src_offset_bytes,
                    dst_offset_bytes, size_bytes, src_stride_bytes,
                    dst_stride_bytes, count, plan_id) ==
           std::tie(other.dst_replica_idx, other.dst_shard_idx,
                    other.src_offset_bytes, other.dst_offset_bytes,
                    other.size_bytes, other.src_stride_bytes,
                    other.dst_stride_bytes, other.count, other.plan_id);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ShardCopyAction& action) {
    return H::combine(std::move(h), action.dst_replica_idx,
                      action.dst_shard_idx, action.src_offset_bytes,
                      action.dst_offset_bytes, action.size_bytes,
                      action.src_stride_bytes, action.dst_stride_bytes,
                      action.count, action.plan_id);
  }
};

// Which destination replicas the Trainer seeds with which bundles, and in what
// order. The bundles are split into `num_stripes` stripes and every stripe is
// pushed to `replication` replicas, so the Trainer sends `replication` copies
// of the model in total, and no replica receives more than one stripe from the
// Trainer. `num_stripes = 1, replication = n` is "n full seeds".
struct SeedLayout {
  // Number of stripes (G).
  int32_t num_stripes = 1;
  // Number of replicas each stripe is pushed to (R).
  int32_t replication = 2;
  // `stripe_bundles[g]`: contiguous, byte-balanced bundle ids of stripe `g`.
  std::vector<std::vector<int32_t>> stripe_bundles;
  // `stripe_seeds[g]`: the `replication` replica indices that the Trainer
  // pushes stripe `g` to.
  std::vector<std::vector<int32_t>> stripe_seeds;
  // `trainer_waves[w]`: `(replica_idx, stripe)` pushes that run concurrently
  // in wave `w` (at most the number of Trainer streams). Replica-major: copy 0
  // of every stripe comes before copy 1 of any stripe.
  std::vector<std::vector<std::pair<int32_t, int32_t>>> trainer_waves;

  bool operator==(const SeedLayout& other) const {
    return std::tie(num_stripes, replication, stripe_bundles, stripe_seeds,
                    trainer_waves) ==
           std::tie(other.num_stripes, other.replication, other.stripe_bundles,
                    other.stripe_seeds, other.trainer_waves);
  }
};

// Complete logical reshard schedule produced by `LogicalReshardPlanner`.
// Contains only logical shard indices (`src_shard_idx`, `dst_replica_idx`,
// `dst_shard_idx`) and deduplicated `plan_id` tables so that cached plans are
// completely independent of ephemeral physical IP:port addresses.
struct LogicalReshardSchedule {
  int32_t num_src_shards = 0;
  int32_t num_dst_shards = 0;
  int32_t num_dst_replicas = 0;
  int32_t num_unique_plans = 0;

  // Ordered variable names and layer indices.
  std::vector<std::string> variable_names;
  // Maps layer_idx -> deduplicated plan_id.
  std::vector<int32_t> variable_to_plan_id;
  // Per-layer destination shard byte sizes (`[layer_idx]`).
  std::vector<int64_t> dst_layer_shard_bytes;
  // Per-layer skip_tiling decision (`[layer_idx]`).
  std::vector<bool> skip_tiling_by_layer;
  // Coarse-grained variable bundles (`num_bundle_groups`).
  std::vector<VariableBundleSpec> variable_bundles;

  // Stripes, seeds and Trainer waves. The sampler-to-sampler pulls that
  // complete every replica are decided while the transfer runs
  // (`PullScheduler`).
  SeedLayout seed_layout;

  // Indexed by `[src_shard_idx][plan_id]`: replica-independent copy actions
  // (`dst_replica_idx == 0`) that trainer shard `src_shard_idx` executes for
  // the variables of `plan_id`. They are emitted once for every seed of the
  // stripe that contains the variable, and for no other replica.
  std::vector<std::vector<std::vector<ShardCopyAction>>>
      trainer_actions_by_shard_and_plan;

  // Indexed by `[dst_shard_idx][layer_idx]`: number of push chunks that
  // `dst_shard_idx` of a seed of the stripe containing `layer_idx` expects to
  // receive from the Trainer for that layer. Other replicas expect none.
  std::vector<std::vector<int32_t>> dst_shard_expected_layer_chunks;
};

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_LOGICAL_TYPES_H_

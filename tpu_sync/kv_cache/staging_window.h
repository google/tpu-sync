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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STAGING_WINDOW_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STAGING_WINDOW_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tpu_raiden {

// Configuration for bounded-window host staging buffer management.
struct StagingWindowConfig {
  // CONSTRUCTION-TIME FIELDS (cannot be changed after session init):
  int32_t window_chunks = 0;  // W (1..255; 0 = off).
  bool serve_windowed_pulls = true;

  // MUTABLE FIELDS (reconfigurable before first transfer):
  int32_t chunk_blocks = 0;               // C in blocks (0 = auto).
  int64_t chunk_bytes_target = 8 << 20;   // 8 MiB target for auto C.
  int64_t stream_bytes_target = 4 << 20;  // 4 MiB target for stream count.
  enum class Admission { kFailFast, kWait } admission = Admission::kFailFast;
  int64_t admission_timeout_ms = 0;
  bool assume_peer_support = false;
  enum class UnitOrder { kChunkMajor } order = UnitOrder::kChunkMajor;

  bool enabled() const { return window_chunks > 0; }
};

// Reads config from TPU_RAIDEN_STAGING_* env vars; unset -> default.
absl::StatusOr<StagingWindowConfig> StagingWindowConfigFromEnv(
    absl::FunctionRef<const char*(const char*)> getenv_fn);
absl::StatusOr<StagingWindowConfig> StagingWindowConfigFromEnv();

// Resolves config: returns explicit config if provided, else reads from env.
absl::StatusOr<StagingWindowConfig> ResolveStagingWindowConfig(
    std::optional<StagingWindowConfig> config = std::nullopt);

// Builds a StagingWindowConfig from optional kwargs.
// If all optionals are nullopt, returns std::nullopt (caller falls back to
// env). Otherwise, starts from the env-resolved configuration
// (ResolveStagingWindowConfig()) and overlays only the explicitly provided
// fields, so explicit kwargs always win and omitted ones keep env/defaults.
absl::StatusOr<std::optional<StagingWindowConfig>> BuildStagingWindowConfig(
    std::optional<int32_t> window_chunks = std::nullopt,
    std::optional<int32_t> chunk_blocks = std::nullopt,
    std::optional<int64_t> chunk_bytes_target = std::nullopt,
    std::optional<int64_t> stream_bytes_target = std::nullopt,
    std::optional<std::string> admission = std::nullopt,
    std::optional<int64_t> admission_timeout_ms = std::nullopt,
    std::optional<bool> assume_peer_support = std::nullopt);

// Computes auto chunk size in blocks: clamp(ceil(target / bytes), 1, 65535).
int32_t AutoChunkBlocks(int64_t chunk_bytes_target,
                        int64_t bytes_per_block_all_shards);

// Computes effective chunk size in blocks (config.chunk_blocks if >0, else
// AutoChunkBlocks).
int32_t EffectiveChunkBlocks(const StagingWindowConfig& config,
                             int64_t bytes_per_block_all_shards = 0);

// Resolves effective chunk size in blocks given per-host shards and per-shard
// bytes.
inline int32_t ResolveEffectiveChunkBlocks(
    const StagingWindowConfig& config, int64_t num_shards_per_host,
    int64_t bytes_per_block_layer_shard) {
  return EffectiveChunkBlocks(
      config, num_shards_per_host * bytes_per_block_layer_shard);
}

// Validates config against pool_blocks (W*C_eff <= pool_blocks when enabled).
absl::Status ValidateStagingWindowConfig(
    const StagingWindowConfig& config, int64_t pool_blocks,
    int64_t bytes_per_block_all_shards = 0);

inline absl::Status ValidateStagingWindowConfig(
    const StagingWindowConfig& config, int64_t pool_blocks,
    int64_t num_shards_per_host, int64_t bytes_per_block_layer_shard) {
  return ValidateStagingWindowConfig(
      config, pool_blocks, num_shards_per_host * bytes_per_block_layer_shard);
}

// Re-validates mutable fields against pool_blocks, rejecting construction-time
// field mutations (window_chunks, serve_windowed_pulls).
absl::Status RevalidateMutableStagingWindowConfig(
    const StagingWindowConfig& current, const StagingWindowConfig& update,
    int64_t pool_blocks, int64_t bytes_per_block_all_shards = 0);

inline absl::Status RevalidateMutableStagingWindowConfig(
    const StagingWindowConfig& current, const StagingWindowConfig& update,
    int64_t pool_blocks, int64_t num_shards_per_host,
    int64_t bytes_per_block_layer_shard) {
  return RevalidateMutableStagingWindowConfig(
      current, update, pool_blocks,
      num_shards_per_host * bytes_per_block_layer_shard);
}

// Computes stream count per unit: clamp(floor(bytes / target), 1, min(P, C_c)).
int32_t StreamsPerUnit(int64_t unit_bytes_per_group,
                       int64_t stream_bytes_target, int32_t parallelism,
                       int32_t chunk_blocks_c);

// Half-open interval [begin, end) of transfer positions for a chunk.
struct ChunkRange {
  int64_t begin = 0;
  int64_t end = 0;

  int64_t size() const { return end - begin; }
  friend bool operator==(const ChunkRange& a, const ChunkRange& b) {
    return a.begin == b.begin && a.end == b.end;
  }
};

// Execution plan for transferring k blocks in chunks through a staging window.
struct ChunkPlan {
  int64_t k = 0;
  int32_t chunk_blocks = 0;      // C
  int32_t n_chunks = 0;          // ceil(k / C)
  int32_t window_chunks = 0;     // W
  int32_t effective_window = 0;  // W_eff = reuse_slots ? W : max(W, n_chunks)
  int64_t ring_blocks = 0;       // min(k, W_eff * C)

  ::tpu_raiden::ChunkRange ChunkRange(int32_t c) const;

  int32_t ChunkLen(int32_t c) const {
    return static_cast<int32_t>(ChunkRange(c).size());
  }

  int32_t SlotOf(int32_t c) const {
    if (c < 0 || effective_window <= 0) return 0;
    return c % effective_window;
  }

  int64_t TotalUnits(int32_t num_layers) const {
    return static_cast<int64_t>(num_layers) * n_chunks;
  }

  absl::StatusOr<std::vector<int64_t>> ExpandRing(
      absl::Span<const int64_t> ring_ids) const;

  absl::StatusOr<int32_t> ChunkOfPosition(int64_t i) const;

  static bool HasDuplicates(absl::Span<const int64_t> ids);
};

// Builds a validated ChunkPlan for transfer size k.
absl::StatusOr<ChunkPlan> BuildChunkPlan(int64_t k, int32_t chunk_blocks,
                                         int32_t window_chunks,
                                         bool reuse_slots);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STAGING_WINDOW_H_

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

#include "tpu_sync/kv_cache/staging_window.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace tpu_raiden {

namespace {

template <typename T>
absl::Status ParseIntEnv(const char* name,
                         absl::FunctionRef<const char*(const char*)> getenv_fn,
                         T* out) {
  const char* raw = getenv_fn(name);
  if (raw == nullptr) return absl::OkStatus();
  if (!absl::SimpleAtoi(raw, out)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid integer for ", name, ": \"", raw, "\""));
  }
  return absl::OkStatus();
}

absl::Status ParseBoolEnv(const char* name,
                          absl::FunctionRef<const char*(const char*)> getenv_fn,
                          bool* out) {
  const char* raw = getenv_fn(name);
  if (raw == nullptr) return absl::OkStatus();
  const std::string lowered = absl::AsciiStrToLower(raw);
  if (lowered == "1" || lowered == "true") {
    *out = true;
    return absl::OkStatus();
  }
  if (lowered == "0" || lowered == "false") {
    *out = false;
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Invalid boolean for ", name, ": \"", raw, "\", expected 1 or 0"));
}

}  // namespace

absl::StatusOr<StagingWindowConfig> StagingWindowConfigFromEnv(
    absl::FunctionRef<const char*(const char*)> getenv_fn) {
  StagingWindowConfig config;
  ABSL_RETURN_IF_ERROR(ParseIntEnv("TPU_RAIDEN_STAGING_WINDOW_CHUNKS",
                                   getenv_fn, &config.window_chunks));
  ABSL_RETURN_IF_ERROR(ParseIntEnv("TPU_RAIDEN_STAGING_CHUNK_BLOCKS", getenv_fn,
                                   &config.chunk_blocks));
  ABSL_RETURN_IF_ERROR(ParseIntEnv("TPU_RAIDEN_STAGING_CHUNK_BYTES", getenv_fn,
                                   &config.chunk_bytes_target));
  ABSL_RETURN_IF_ERROR(ParseIntEnv("TPU_RAIDEN_STAGING_STREAM_BYTES", getenv_fn,
                                   &config.stream_bytes_target));
  if (const char* raw = getenv_fn("TPU_RAIDEN_STAGING_ADMISSION")) {
    const std::string lowered = absl::AsciiStrToLower(raw);
    if (lowered == "fail_fast") {
      config.admission = StagingWindowConfig::Admission::kFailFast;
    } else if (lowered == "wait") {
      config.admission = StagingWindowConfig::Admission::kWait;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid value for TPU_RAIDEN_STAGING_ADMISSION: \"",
                       raw, "\", expected \"fail_fast\" or \"wait\""));
    }
  }
  ABSL_RETURN_IF_ERROR(ParseIntEnv("TPU_RAIDEN_STAGING_ADMISSION_TIMEOUT_MS",
                                   getenv_fn, &config.admission_timeout_ms));
  ABSL_RETURN_IF_ERROR(ParseBoolEnv("TPU_RAIDEN_STAGING_WINDOW_SERVE",
                                    getenv_fn, &config.serve_windowed_pulls));
  ABSL_RETURN_IF_ERROR(ParseBoolEnv("TPU_RAIDEN_STAGING_WINDOW_ASSUME_PEER",
                                    getenv_fn, &config.assume_peer_support));
  return config;
}

absl::StatusOr<StagingWindowConfig> StagingWindowConfigFromEnv() {
  return StagingWindowConfigFromEnv(
      [](const char* name) { return std::getenv(name); });
}

absl::StatusOr<StagingWindowConfig> ResolveStagingWindowConfig(
    std::optional<StagingWindowConfig> config) {
  if (config.has_value()) {
    return *config;
  }
  return StagingWindowConfigFromEnv();
}

absl::StatusOr<std::optional<StagingWindowConfig>> BuildStagingWindowConfig(
    std::optional<int32_t> window_chunks, std::optional<int32_t> chunk_blocks,
    std::optional<int64_t> chunk_bytes_target,
    std::optional<int64_t> stream_bytes_target,
    std::optional<std::string> admission,
    std::optional<int64_t> admission_timeout_ms,
    std::optional<bool> assume_peer_support) {
  if (!window_chunks.has_value() && !chunk_blocks.has_value() &&
      !chunk_bytes_target.has_value() && !stream_bytes_target.has_value() &&
      !admission.has_value() && !admission_timeout_ms.has_value() &&
      !assume_peer_support.has_value()) {
    return std::nullopt;
  }
  auto base_cfg_result = ResolveStagingWindowConfig();
  if (!base_cfg_result.ok()) {
    return base_cfg_result.status();
  }
  StagingWindowConfig cfg = *std::move(base_cfg_result);
  if (window_chunks.has_value()) {
    cfg.window_chunks = *window_chunks;
  }
  if (chunk_blocks.has_value()) {
    cfg.chunk_blocks = *chunk_blocks;
  }
  if (chunk_bytes_target.has_value()) {
    cfg.chunk_bytes_target = *chunk_bytes_target;
  }
  if (stream_bytes_target.has_value()) {
    cfg.stream_bytes_target = *stream_bytes_target;
  }
  if (admission.has_value()) {
    if (*admission == "fail_fast" || admission->empty()) {
      cfg.admission = StagingWindowConfig::Admission::kFailFast;
    } else if (*admission == "wait") {
      cfg.admission = StagingWindowConfig::Admission::kWait;
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid staging_admission: must be 'fail_fast' or 'wait', got '",
          *admission, "'"));
    }
  }
  if (admission_timeout_ms.has_value()) {
    cfg.admission_timeout_ms = *admission_timeout_ms;
  }
  if (assume_peer_support.has_value()) {
    cfg.assume_peer_support = *assume_peer_support;
  }
  return std::make_optional(cfg);
}

int32_t AutoChunkBlocks(int64_t chunk_bytes_target,
                        int64_t bytes_per_block_all_shards) {
  if (bytes_per_block_all_shards <= 0 || chunk_bytes_target <= 0) return 1;
  const int64_t ceil_val =
      1 + (chunk_bytes_target - 1) / bytes_per_block_all_shards;
  return static_cast<int32_t>(std::clamp<int64_t>(ceil_val, 1, 65535));
}

int32_t EffectiveChunkBlocks(const StagingWindowConfig& config,
                             int64_t bytes_per_block_all_shards) {
  return config.chunk_blocks > 0 ? config.chunk_blocks
                                 : AutoChunkBlocks(config.chunk_bytes_target,
                                                   bytes_per_block_all_shards);
}

absl::Status ValidateStagingWindowConfig(const StagingWindowConfig& config,
                                         int64_t pool_blocks,
                                         int64_t bytes_per_block_all_shards) {
  if (config.chunk_blocks < 0 || config.chunk_blocks > 65535) {
    return absl::InvalidArgumentError(absl::StrCat(
        "chunk_blocks must be in [0, 65535], got ", config.chunk_blocks));
  }
  if (config.chunk_bytes_target <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("chunk_bytes_target must be positive, got ",
                     config.chunk_bytes_target));
  }
  if (config.stream_bytes_target <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("stream_bytes_target must be positive, got ",
                     config.stream_bytes_target));
  }
  if (config.admission_timeout_ms < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("admission_timeout_ms must be non-negative, got ",
                     config.admission_timeout_ms));
  }
  if (config.window_chunks < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "window_chunks must be non-negative, got ", config.window_chunks));
  }
  if (!config.enabled()) return absl::OkStatus();

  if (config.window_chunks > 255) {
    return absl::InvalidArgumentError(
        absl::StrCat("window_chunks must be in [1, 255] when enabled, got ",
                     config.window_chunks));
  }
  const int32_t effective_chunk_blocks =
      EffectiveChunkBlocks(config, bytes_per_block_all_shards);
  const int64_t req =
      static_cast<int64_t>(config.window_chunks) * effective_chunk_blocks;
  if (pool_blocks <= 0 || req > pool_blocks) {
    return absl::InvalidArgumentError(
        absl::StrCat("Required staging window blocks W*C (", req,
                     ") exceeds pool_blocks (", pool_blocks, ")"));
  }
  return absl::OkStatus();
}

absl::Status RevalidateMutableStagingWindowConfig(
    const StagingWindowConfig& current, const StagingWindowConfig& update,
    int64_t pool_blocks, int64_t bytes_per_block_all_shards) {
  if (update.window_chunks != current.window_chunks) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot mutate construction-time field window_chunks (current ",
        current.window_chunks, ", update ", update.window_chunks, ")"));
  }
  if (update.serve_windowed_pulls != current.serve_windowed_pulls) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot mutate construction-time field serve_windowed_pulls (current ",
        current.serve_windowed_pulls, ", update ", update.serve_windowed_pulls,
        ")"));
  }
  return ValidateStagingWindowConfig(update, pool_blocks,
                                     bytes_per_block_all_shards);
}

int32_t StreamsPerUnit(int64_t unit_bytes_per_group,
                       int64_t stream_bytes_target, int32_t parallelism,
                       int32_t chunk_blocks_c) {
  int64_t raw_streams = 1;
  if (stream_bytes_target > 0 && unit_bytes_per_group > 0) {
    raw_streams = unit_bytes_per_group / stream_bytes_target;
  }
  int32_t upper_bound = std::max(1, std::min(parallelism, chunk_blocks_c));
  return static_cast<int32_t>(std::clamp<int64_t>(raw_streams, 1, upper_bound));
}

ChunkRange ChunkPlan::ChunkRange(int32_t c) const {
  if (c < 0 || c >= n_chunks) return {0, 0};
  const int64_t begin = static_cast<int64_t>(c) * chunk_blocks;
  const int64_t end = std::min<int64_t>(begin + chunk_blocks, k);
  return {begin, end};
}

absl::StatusOr<std::vector<int64_t>> ChunkPlan::ExpandRing(
    absl::Span<const int64_t> ring_ids) const {
  if (chunk_blocks <= 0 || effective_window <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid ChunkPlan: chunk_blocks=", chunk_blocks,
                     ", effective_window=", effective_window));
  }
  if (static_cast<int64_t>(ring_ids.size()) != ring_blocks) {
    return absl::InvalidArgumentError(
        absl::StrCat("ring_ids size mismatch: expected ", ring_blocks, ", got ",
                     ring_ids.size()));
  }
  std::vector<int64_t> out(k);
  for (int64_t i = 0; i < k; ++i) {
    const int64_t slot = (i / chunk_blocks) % effective_window;
    const int64_t offset = i % chunk_blocks;
    out[i] = ring_ids[slot * chunk_blocks + offset];
  }
  return out;
}

absl::StatusOr<int32_t> ChunkPlan::ChunkOfPosition(int64_t i) const {
  if (chunk_blocks <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid ChunkPlan: chunk_blocks=", chunk_blocks));
  }
  if (i < 0 || i >= k) {
    return absl::InvalidArgumentError(
        absl::StrCat("Position ", i, " out of range [0, ", k, ")"));
  }
  return static_cast<int32_t>(i / chunk_blocks);
}

bool ChunkPlan::HasDuplicates(absl::Span<const int64_t> ids) {
  absl::flat_hash_set<int64_t> seen;
  seen.reserve(ids.size());
  for (int64_t id : ids) {
    if (!seen.insert(id).second) return true;
  }
  return false;
}

absl::StatusOr<ChunkPlan> BuildChunkPlan(int64_t k, int32_t chunk_blocks,
                                         int32_t window_chunks,
                                         bool reuse_slots) {
  if (k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("k must be positive, got ", k));
  }
  if (chunk_blocks <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("chunk_blocks must be positive, got ", chunk_blocks));
  }
  if (window_chunks <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("window_chunks must be positive, got ", window_chunks));
  }
  const int64_t n_chunks_64 = 1 + (k - 1) / chunk_blocks;
  if (n_chunks_64 > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat("n_chunks exceeds int32_t max: ", n_chunks_64));
  }
  const int32_t n_chunks = static_cast<int32_t>(n_chunks_64);
  const int32_t effective_window =
      reuse_slots ? window_chunks : std::max(window_chunks, n_chunks);
  const int64_t ring_blocks = std::min<int64_t>(
      k, static_cast<int64_t>(effective_window) * chunk_blocks);

  return ChunkPlan{
      k, chunk_blocks, n_chunks, window_chunks, effective_window, ring_blocks};
}

}  // namespace tpu_raiden

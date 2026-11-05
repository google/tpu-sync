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

#include "tpu_sync/kv_cache/backends/storage/storage_lookup_helper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

StorageLookupHelper::StorageLookupHelper(std::string data_name,
                                         std::string dp_rank,
                                         size_t lookup_batch_size)
    : data_name_(std::move(data_name)),
      dp_rank_(std::move(dp_rank)),
      lookup_batch_size_(lookup_batch_size) {
  CHECK_GT(lookup_batch_size_, 0u);
}

RaidenBlockId StorageLookupHelper::MakeSharedStorageBlock() const {
  RaidenBlockId block;
  block.status = BlockStatus::SHARED_STORAGE;
  block.raiden_id.job_replica_id = "shared";
  block.raiden_id.data_name = data_name_;
  return block;
}

absl::StatusOr<std::vector<BlockKey>> StorageLookupHelper::ExpandShardKeys(
    const BlockKeyMapper& mapper, const std::string& block_hash) {
  const int pp_size = mapper.pp_size();
  const int pcp_size = mapper.pcp_size();
  const int tp_size = mapper.tp_size();
  // In secondary storage every (pp_rank, pcp_rank, tp_rank) worker writes its
  // own shard of each block. The block is available only if every shard is
  // present. An undeclared axis spans a single shard and is left unset in the
  // mapping options, so it adds no path segment.
  auto declare = [](int size, int rank, int* size_field, int* rank_field) {
    if (size == kAxisUndeclared) return;
    *size_field = size;
    *rank_field = rank;
  };
  std::vector<BlockKey> keys;
  keys.reserve(mapper.shards_per_block());
  for (int pp_rank = 0; pp_rank < std::max(1, pp_size); ++pp_rank) {
    for (int pcp_rank = 0; pcp_rank < std::max(1, pcp_size); ++pcp_rank) {
      for (int tp_rank = 0; tp_rank < std::max(1, tp_size); ++tp_rank) {
        KeyMappingOptions lookup_opts;
        ParallelismConfig& p = lookup_opts.parallelism;
        declare(pp_size, pp_rank, &p.pp_size, &p.pp_rank);
        declare(pcp_size, pcp_rank, &p.pcp_size, &p.pcp_rank);
        declare(tp_size, tp_rank, &p.tp_size, &p.tp_rank);
        ABSL_ASSIGN_OR_RETURN(BlockKey key,
                              mapper.MapKey(block_hash, lookup_opts));
        keys.push_back(std::move(key));
      }
    }
  }
  return keys;
}

std::vector<bool> StorageLookupHelper::ProbeExists(
    KVBackend& backend, absl::Span<const BlockKey> keys) {
  std::vector<bool> exists(keys.size(), false);
  std::promise<std::vector<absl::StatusOr<bool>>> promise;
  auto future = promise.get_future();
  backend.BatchExistsAsync(keys,
                           [&promise](std::vector<absl::StatusOr<bool>> res) {
                             promise.set_value(std::move(res));
                           });
  const std::vector<absl::StatusOr<bool>> answers = future.get();
  // A short answer cannot be attributed to keys reliably; treat it as absent.
  if (answers.size() != keys.size()) return exists;
  for (size_t i = 0; i < answers.size(); ++i) {
    exists[i] = answers[i].ok() && *answers[i];
  }
  return exists;
}

absl::Status StorageLookupHelper::ValidateLookupOptions(
    const LookupOptions& options) {
  if (options.match_mode == LookupMatchMode::kSubset && options.enable_global) {
    return absl::InvalidArgumentError(
        "LookupMatchMode::kSubset is not supported with enable_global=true");
  }
  if (options.host_resident_only && options.enable_global) {
    return absl::InvalidArgumentError(
        "host_resident_only is not supported with enable_global=true");
  }
  if (options.include_eviction_candidates && options.pin_found) {
    return absl::InvalidArgumentError(
        "include_eviction_candidates is not supported with pin_found=true");
  }
  return absl::OkStatus();
}

absl::StatusOr<BlockSliceList> StorageLookupHelper::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options,
    KeyExpander expand_keys, ExistenceProber probe_exists) const {
  // Without a cache no key is fresh, and confirmed keys are not recorded.
  return Lookup(
      block_hashes, options, expand_keys, probe_exists,
      [](absl::Span<const std::vector<BlockKey>> shard_keys) {
        std::vector<std::vector<bool>> fresh;
        fresh.reserve(shard_keys.size());
        for (const std::vector<BlockKey>& keys : shard_keys) {
          fresh.emplace_back(keys.size(), false);
        }
        return fresh;
      },
      [](absl::Span<const BlockKey* const> keys) {});
}

// Lookup phases:
// - Phase 0 (map): expands each block hash into its shard keys (`kPrefix`
//   stops at the first unmappable hash; `kSubset` records an empty key list
//   and continues to preserve index alignment).
// - Phase 1 (cache): marks the shard keys that `cached_fresh` reports as
//   present.
// - Phase 2 (storage): probes the other shard keys in chunks of
//   `lookup_batch_size_` (`kPrefix` exits early after the first chunk with an
//   absent key; `kSubset` probes all chunks) and passes the confirmed keys to
//   `cache_insert`.
// - Phase 3 (merge): emits SHARED_STORAGE entries (`kPrefix` up to the first
//   unavailable block; `kSubset` for all available blocks).
absl::StatusOr<BlockSliceList> StorageLookupHelper::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options,
    KeyExpander expand_keys, ExistenceProber probe_exists,
    CachedFreshFn cached_fresh, CacheInsertFn cache_insert) const {
  ABSL_RETURN_IF_ERROR(ValidateLookupOptions(options));

  BlockSliceList results;
  if (options.host_resident_only || block_hashes.empty()) {
    return results;
  }

  // Phase 0 (map).
  std::vector<std::vector<BlockKey>> shard_keys;
  shard_keys.reserve(block_hashes.size());
  for (const std::string& hash : block_hashes) {
    absl::StatusOr<std::vector<BlockKey>> keys = expand_keys(hash);
    if (!keys.ok()) {
      if (options.match_mode == LookupMatchMode::kSubset) {
        shard_keys.emplace_back();
        continue;
      }
      break;
    }
    shard_keys.push_back(*std::move(keys));
  }
  if (shard_keys.empty()) return results;

  // Phase 1 (cache). The answer is indexed below, so its shape must match.
  std::vector<std::vector<bool>> present = cached_fresh(shard_keys);
  CHECK_EQ(present.size(), shard_keys.size());
  for (size_t i = 0; i < shard_keys.size(); ++i) {
    CHECK_EQ(present[i].size(), shard_keys[i].size());
  }

  // Phase 2 (storage): flat list of cache-missed shard keys across all blocks,
  // probed in chunks of lookup_batch_size_.
  struct ShardRef {
    size_t block;
    size_t shard;
  };
  uint64_t cache_hits = 0;
  std::vector<ShardRef> misses;
  for (size_t i = 0; i < shard_keys.size(); ++i) {
    for (size_t j = 0; j < shard_keys[i].size(); ++j) {
      if (present[i][j]) {
        ++cache_hits;
      } else {
        misses.push_back({i, j});
      }
    }
  }
  uint64_t stat_hits = 0;
  uint64_t stat_misses = 0;
  std::vector<const BlockKey*> confirmed;
  for (size_t offset = 0; offset < misses.size();
       offset += lookup_batch_size_) {
    const size_t chunk_len =
        std::min(lookup_batch_size_, misses.size() - offset);
    std::vector<BlockKey> chunk_keys;
    chunk_keys.reserve(chunk_len);
    for (size_t k = 0; k < chunk_len; ++k) {
      const ShardRef& miss = misses[offset + k];
      chunk_keys.push_back(shard_keys[miss.block][miss.shard]);
    }
    const std::vector<bool> exists = probe_exists(chunk_keys);
    // A short answer cannot be attributed to keys reliably; treat it as absent.
    const bool answer_complete = exists.size() == chunk_len;
    bool chunk_has_absent = false;
    for (size_t k = 0; k < chunk_len; ++k) {
      const ShardRef& miss = misses[offset + k];
      if (answer_complete && exists[k]) {
        present[miss.block][miss.shard] = true;
        confirmed.push_back(&shard_keys[miss.block][miss.shard]);
        ++stat_hits;
      } else {
        chunk_has_absent = true;
        ++stat_misses;
      }
    }
    if (options.match_mode == LookupMatchMode::kPrefix && chunk_has_absent) {
      break;  // Early exit; see Phase 2 above.
    }
  }
  cache_insert(confirmed);

  telemetry::RaidenMetricStore& metric_store =
      telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (metric_store.HasBackends()) {
    auto record = [&](absl::string_view result, uint64_t count) {
      if (count == 0) return;
      const telemetry::MetricLabel labels[] = {
          {telemetry::metric_labels::kDpRank, dp_rank_},
          {telemetry::metric_labels::kResult, result},
      };
      metric_store.IncrementCounter(
          telemetry::metric_names::kStorageMetadataCheckTotal, labels, count);
    };
    record(telemetry::metric_labels::kResultCacheHit, cache_hits);
    record(telemetry::metric_labels::kResultStatHit, stat_hits);
    record(telemetry::metric_labels::kResultStatMiss, stat_misses);
  }

  // Phase 3 (merge).
  auto all_present = [](const std::vector<bool>& shards) {
    return !shards.empty() && absl::c_all_of(shards, [](bool shard_present) {
      return shard_present;
    });
  };
  if (options.match_mode == LookupMatchMode::kSubset) {
    for (size_t i = 0; i < shard_keys.size(); ++i) {
      if (all_present(present[i])) {
        results.push_back(
            std::make_pair(block_hashes[i], MakeSharedStorageBlock()));
      }
    }
  } else {
    for (size_t i = 0; i < shard_keys.size() && all_present(present[i]); ++i) {
      results.push_back(
          std::make_pair(block_hashes[i], MakeSharedStorageBlock()));
    }
  }
  return results;
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

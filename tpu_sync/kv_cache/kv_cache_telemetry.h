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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_TELEMETRY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_TELEMETRY_H_

// Telemetry helper functions for KVCacheStore and its offload backends.

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {
namespace kv_cache {

// Increments `kv_recall_blocks_total` by `count` with the given store identity
// and recall labels. `dp_rank` must be a pre-rendered decimal string, and
// `tier` and `outcome` must be `telemetry::metric_labels::kTier*` and
// `kOutcome*` constants respectively. No-op when `count == 0`.
inline void IncrementKvRecallBlocks(absl::string_view dp_rank,
                                    absl::string_view tier,
                                    absl::string_view outcome, uint64_t count) {
  if (count == 0) return;
  const telemetry::MetricLabel labels[] = {
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kTier, tier},
      {telemetry::metric_labels::kOutcome, outcome},
  };
  telemetry::RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
      telemetry::metric_names::kKvRecallBlocksTotal, labels, count);
}

// Increments `kv_save_blocks_total` by `count` with the given store identity
// and save labels. `dp_rank` must be a pre-rendered decimal string, and
// `tier` and `outcome` must be `telemetry::metric_labels::kTier*` and
// `kOutcome*` constants respectively (`outcome` must be `kOutcomeOk`,
// `kOutcomeRejected`, or `kOutcomeFailed`). No-op when `count == 0`.
inline void IncrementKvSaveBlocks(absl::string_view dp_rank,
                                  absl::string_view tier,
                                  absl::string_view outcome, uint64_t count) {
  if (count == 0) return;
  const telemetry::MetricLabel labels[] = {
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kTier, tier},
      {telemetry::metric_labels::kOutcome, outcome},
  };
  telemetry::RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
      telemetry::metric_names::kKvSaveBlocksTotal, labels, count);
}

// Sets the `host_pool_blocks` gauge for each `state` (`total`, `used`,
// `pinned`) with the given store identity labels. `dp_rank` must be a
// pre-rendered decimal string.
inline void RecordKvHostPoolBlocks(absl::string_view dp_rank, size_t total,
                                   size_t used, size_t pinned) {
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  const telemetry::MetricLabel total_labels[] = {
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kState, telemetry::metric_labels::kStateTotal},
  };
  const telemetry::MetricLabel used_labels[] = {
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kState, telemetry::metric_labels::kStateUsed},
  };
  const telemetry::MetricLabel pinned_labels[] = {
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kState,
       telemetry::metric_labels::kStatePinned},
  };
  store.SetGauge(telemetry::metric_names::kHostPoolBlocks, total_labels,
                 static_cast<double>(total));
  store.SetGauge(telemetry::metric_names::kHostPoolBlocks, used_labels,
                 static_cast<double>(used));
  store.SetGauge(telemetry::metric_names::kHostPoolBlocks, pinned_labels,
                 static_cast<double>(pinned));
}

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_TELEMETRY_H_

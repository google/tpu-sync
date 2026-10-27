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
inline void IncrementKvRecallBlocks(absl::string_view host_ip,
                                    absl::string_view dp_rank,
                                    absl::string_view tier,
                                    absl::string_view outcome, uint64_t count) {
  if (count == 0) return;
  const telemetry::MetricLabel labels[] = {
      {telemetry::metric_labels::kHostIp, host_ip},
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kTier, tier},
      {telemetry::metric_labels::kOutcome, outcome},
  };
  telemetry::RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
      telemetry::metric_names::kKvRecallBlocksTotal, labels, count);
}

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_TELEMETRY_H_

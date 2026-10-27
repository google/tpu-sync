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

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <ratio>  // NOLINT(build/c++11)

#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {
namespace kv_cache {

// Monotonic clock used to time KV cache recall batches.
using RecallClock = std::chrono::steady_clock;

// Records one settled recall batch of `count` blocks with the given store
// identity and recall labels: increments `kv_recall_blocks_total` by `count`
// and observes the time elapsed since `start` (in milliseconds) in
// `kv_recall_duration_ms`, with the same labels. `dp_rank` must be a
// pre-rendered decimal string, and `tier` and `outcome` must be
// `telemetry::metric_labels::kTier*` and `kOutcome*` constants respectively.
// No-op when `count == 0`, so an empty batch adds no duration sample.
inline void RecordKvRecall(absl::string_view host_ip, absl::string_view dp_rank,
                           absl::string_view tier, absl::string_view outcome,
                           uint64_t count, RecallClock::time_point start) {
  if (count == 0) return;
  const double duration_ms =
      std::chrono::duration<double, std::milli>(RecallClock::now() - start)
          .count();
  const telemetry::MetricLabel labels[] = {
      {telemetry::metric_labels::kHostIp, host_ip},
      {telemetry::metric_labels::kDpRank, dp_rank},
      {telemetry::metric_labels::kTier, tier},
      {telemetry::metric_labels::kOutcome, outcome},
  };
  const telemetry::RaidenMetricStore& store =
      telemetry::RaidenMetricStore::GetGlobalMetricStore();
  store.IncrementCounter(telemetry::metric_names::kKvRecallBlocksTotal, labels,
                         count);
  store.ObserveHistogram(telemetry::metric_names::kKvRecallDurationMs, labels,
                         duration_ms);
}

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_TELEMETRY_H_

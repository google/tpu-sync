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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_COLLECTOR_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_COLLECTOR_H_

#include <string>

#include "absl/container/flat_hash_map.h"

namespace tpu_raiden::telemetry {

// Configuration options for the shared-memory telemetry collector.
struct ShmCollectorOptions {
  // Directory where shared-memory segment files (.mmap) are stored (e.g.
  // "/dev/shm" or "/tmp").
  std::string shm_dir;
};

// Thread-safe shared-memory telemetry collector.
// Scans for memory-mapped segment files in /dev/shm created by active TPU
// workers, acquires shared reader locks, aggregates counters, gauges, and
// histograms across worker processes, and cleans up dead worker files.
//
// Thread-safety note: `ShmCollector` instances are thread-safe and const
// member methods may be called concurrently from multiple threads, provided
// each calling thread passes its own distinct `totals` map instance. Sharing a
// single `totals` instance across threads requires external synchronization.
class ShmCollector {
 public:
  explicit ShmCollector(ShmCollectorOptions options);

  // Scans the configured shared-memory directory, aggregates metric totals
  // across all live worker processes into `totals`, and reaps dead worker
  // files. Clears and populates `totals`, reusing existing bucket allocation
  // across calls.
  //
  // Dead worker processes (whose exclusive lock can be acquired) are reaped
  // by unlinking their segment files. Dead worker metrics are intentionally
  // discarded upon reaping; only metrics from live processes are aggregated.
  // Downstream consumers should note that cumulative counters from reaped
  // workers are not retained.
  //
  // Gauges are aggregated across workers by addition. For non-additive gauges
  // (e.g. utilization or fraction), callers should supply rank-distinguishing
  // labels to avoid cross-worker aggregation.
  //
  // Metric keys in `totals` are formatted as follows:
  // - Counters & Gauges: "metric_name" (unlabeled) or
  //   "metric_name/label1=val1;label2=val2" (labeled).
  // - Histograms:
  //   * Count: "metric_name_count" or "metric_name_count/labels"
  //   * Sum: "metric_name_sum" or "metric_name_sum/labels"
  //   * Buckets: "metric_name_bucket/le=<bound>" (unlabeled) or
  //     "metric_name_bucket/labels;le=<bound>" (labeled).
  //     Bucket values are cumulative counts across kDefaultHistogramBuckets
  //     and "+Inf", strictly non-decreasing.
  //
  // Thread safety: Concurrent invocations from multiple threads are safe
  // provided each calling thread supplies its own distinct `totals` container.
  void CollectMetrics(absl::flat_hash_map<std::string, double>& totals) const;

 private:
  ShmCollectorOptions options_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_COLLECTOR_H_

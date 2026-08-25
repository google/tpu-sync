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
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

namespace internal {

// Aggregates metric entries from a single memory-mapped segment layout into
// `totals`. Exposed for testing only.
void AggregateSegment(const ShmSegmentLayout* seg,
                      absl::flat_hash_map<std::string, double>& totals);

}  // namespace internal

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
class ShmCollector {
 public:
  explicit ShmCollector(ShmCollectorOptions options);
  ~ShmCollector() = default;

  ShmCollector(const ShmCollector&) = default;
  ShmCollector& operator=(const ShmCollector&) = default;
  ShmCollector(ShmCollector&&) noexcept = default;
  ShmCollector& operator=(ShmCollector&&) noexcept = default;

  // Scans the configured shared-memory directory, aggregates metric totals
  // across all live worker processes into `totals`, and reaps dead worker
  // files. Clears and populates `totals`, reusing existing bucket allocation
  // across calls.
  //
  // Metric keys in `totals` are formatted as:
  // - "metric_name" (unlabeled) or "metric_name/label1=val1;label2=val2"
  // - For histograms, suffixes "/count", "/sum", and "/bucket_<N>" are
  //   appended (e.g. "transfer_duration_ms/bucket_0").
  void CollectMetrics(absl::flat_hash_map<std::string, double>& totals) const;

 private:
  ShmCollectorOptions options_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_COLLECTOR_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {

// Base class for multi-process shared-memory telemetry exporters.
// Manages an underlying ShmWriter for low-overhead metric publishing and an
// ShmCollector for multi-worker aggregation.
//
// Preconditions:
// Requires valid local_rank (via options.local_rank) and shm_dir (via
// options.shm_dir). Fails fast with CHECK if either is missing or invalid.
// Trailing slashes in shm_dir are normalized while preserving "/".
//
// Thread-safety & Destruction contract:
// Callers must ensure all concurrent metric recording has finished prior to
// exporter destruction.
class BaseShmExporter : public MetricsBackend {
 public:
  explicit BaseShmExporter(ExporterOptions options);
  ~BaseShmExporter() override;

  BaseShmExporter(const BaseShmExporter&) = delete;
  BaseShmExporter& operator=(const BaseShmExporter&) = delete;
  BaseShmExporter(BaseShmExporter&&) = delete;
  BaseShmExporter& operator=(BaseShmExporter&&) = delete;

  // Metric recording methods. Safe for concurrent execution across worker
  // threads. Writes directly to the memory-mapped shared segment.
  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val) const override;
  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override;
  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override;

  // Returns an empty string. Shared-memory exporters intentionally do not
  // produce in-process text snapshots because metrics are gathered
  // out-of-process via memory-mapped segments.
  std::string GetTextSnapshot() const override;

  // Scans the shared-memory directory and aggregates metric totals across all
  // local worker processes into `totals`.
  void CollectMetrics(absl::flat_hash_map<std::string, double>& totals) const;

  const ExporterOptions& GetOptions() const { return options_; }

 private:
  ExporterOptions options_;
  std::unique_ptr<ShmWriter> shm_writer_;
  std::unique_ptr<ShmCollector> collector_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_

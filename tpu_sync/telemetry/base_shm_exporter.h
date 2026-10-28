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
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {

// Default rank identifier used in shared-memory segment filenames when
// `ExporterOptions::local_rank` is unset (e.g., store-level coordinator
// processes).
inline constexpr absl::string_view kDefaultShmSegmentRank = "store";

// Base class for multi-process shared-memory telemetry exporters.
// Manages an underlying ShmWriter for low-overhead metric publishing and an
// ShmCollector for multi-worker aggregation.
//
// Preconditions:
// Requires valid shm_dir (via options.shm_dir) and, when options.local_rank is
// provided, a valid non-negative integer rank string. Fails fast with CHECK if
// shm_dir is missing or invalid, or if local_rank is provided and invalid.
// Trailing slashes in shm_dir are stripped; the root path "/" is prohibited.
// Note: custom_buckets is ignored by shared-memory backends in favor of fixed
// compile-time bucket layouts (kDefaultHistogramBuckets).
//
// Thread-safety & Destruction contract:
// Callers must ensure all concurrent metric recording and CollectMetrics
// calls have finished prior to exporter destruction.
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
  // Records a gauge value. Automatically attaches the "local_rank" label for
  // worker processes (when options.local_rank is set) unless "local_rank" or
  // "dp_rank" is already present in caller labels.
  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override;
  // Records a histogram observation. Safe for concurrent execution across
  // worker threads. Writes directly to the memory-mapped shared segment.
  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override;

  // Returns an empty string. BaseShmExporter is un-opinionated and does not
  // prescribe an exposition format (Prometheus vs OTel); this method is
  // intended to be overridden by format-specific subclasses.
  std::string GetTextSnapshot() const override;

  // Scans the shared-memory directory and returns aggregated metrics across
  // all local worker processes.
  AggregatedMetrics CollectMetrics() const;

  // Returns the validated exporter configuration options.
  const ExporterOptions& options() const { return options_; }

 private:
  MetricLabel LocalRankLabel() const {
    DCHECK(options_.local_rank.has_value());
    return MetricLabel{metric_labels::kLocalRank, *options_.local_rank};
  }

  const ExporterOptions options_;
  ShmWriter shm_writer_;
  ShmCollector shm_collector_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_

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

#include "tpu_sync/telemetry/base_shm_exporter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {

BaseShmExporter::BaseShmExporter(ExporterOptions options)
    : options_(std::move(options)) {
  CHECK(options_.local_rank.has_value() && !options_.local_rank->empty())
      << "options.local_rank must be specified and non-empty for "
         "BaseShmExporter";

  int rank_val = -1;
  CHECK(absl::SimpleAtoi(*options_.local_rank, &rank_val) && rank_val >= 0)
      << "options.local_rank must be a valid non-negative integer for "
         "BaseShmExporter, got: '"
      << *options_.local_rank << "'";

  CHECK(options_.shm_dir.has_value() && !options_.shm_dir->empty())
      << "options.shm_dir must be specified and non-empty for BaseShmExporter";

  while (options_.shm_dir->size() > 1 && options_.shm_dir->back() == '/') {
    options_.shm_dir->pop_back();
  }

  shm_writer_ = std::make_unique<ShmWriter>(ShmWriterOptions{
      .shm_dir = *options_.shm_dir,
      .local_rank = *options_.local_rank,
  });

  collector_ = std::make_unique<ShmCollector>(ShmCollectorOptions{
      .shm_dir = *options_.shm_dir,
  });
}

BaseShmExporter::~BaseShmExporter() = default;

void BaseShmExporter::IncrementCounter(absl::string_view name, LabelSpan labels,
                                       uint64_t val) const {
  if (shm_writer_ != nullptr) {
    shm_writer_->IncrementCounter(name, labels, val);
  }
}

void BaseShmExporter::SetGauge(absl::string_view name, LabelSpan labels,
                               double val) const {
  if (shm_writer_ != nullptr) {
    shm_writer_->SetGauge(name, labels, val);
  }
}

void BaseShmExporter::ObserveHistogram(absl::string_view name, LabelSpan labels,
                                       double val) const {
  if (shm_writer_ != nullptr) {
    shm_writer_->ObserveHistogram(name, labels, val);
  }
}

std::string BaseShmExporter::GetTextSnapshot() const {
  // Shared-memory exporters intentionally do not produce in-process text
  // snapshots because metrics are gathered out-of-process via memory-mapped
  // segments.
  return "";
}

void BaseShmExporter::CollectMetrics(
    absl::flat_hash_map<std::string, double>& totals) const {
  if (collector_ != nullptr) {
    collector_->CollectMetrics(totals);
  }
}

}  // namespace tpu_raiden::telemetry

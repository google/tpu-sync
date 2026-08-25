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
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {

BaseShmExporter::BaseShmExporter(const ExporterOptions& options)
    : options_(options) {
  std::string rank;
  if (options_.local_rank.has_value() && !options_.local_rank->empty()) {
    rank = *options_.local_rank;
  } else {
    const char* env_rank = std::getenv("LOCAL_RANK");
    if (env_rank != nullptr && *env_rank != '\0') {
      rank = env_rank;
    }
  }

  if (rank.empty()) {
    LOG(ERROR) << "LOCAL_RANK must be specified for BaseShmExporter (via "
                  "options.local_rank or LOCAL_RANK environment variable).";
    throw std::invalid_argument(
        "LOCAL_RANK must be specified for BaseShmExporter (via "
        "options.local_rank or LOCAL_RANK environment variable).");
  }
  options_.local_rank = rank;
  const std::string dir = options_.GetShmDir();

  ShmWriterOptions w_opts;
  w_opts.shm_dir = dir;
  w_opts.local_rank = rank;
  shm_writer_ = std::make_unique<ShmWriter>(w_opts);

  ShmCollectorOptions c_opts;
  c_opts.shm_dir = dir;
  collector_ = std::make_unique<ShmCollector>(c_opts);
}

BaseShmExporter::~BaseShmExporter() { Stop(); }

void BaseShmExporter::IncrementCounter(absl::string_view name, LabelSpan labels,
                                       uint64_t val) const {
  if (shm_writer_) shm_writer_->IncrementCounter(name, labels, val);
}

void BaseShmExporter::SetGauge(absl::string_view name, LabelSpan labels,
                               double val) const {
  if (shm_writer_) shm_writer_->SetGauge(name, labels, val);
}

void BaseShmExporter::ObserveHistogram(absl::string_view name, LabelSpan labels,
                                       double val) const {
  if (shm_writer_) shm_writer_->ObserveHistogram(name, labels, val);
}

std::string BaseShmExporter::GetTextSnapshot() const {
  return "";
}

void BaseShmExporter::CollectMetrics(
    absl::flat_hash_map<std::string, double>& totals) const {
  if (collector_) collector_->CollectMetrics(totals);
}

}  // namespace tpu_raiden::telemetry

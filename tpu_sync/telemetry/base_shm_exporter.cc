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

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {
namespace {

// Validates that `options.local_rank` is a valid non-negative integer
// (canonicalizing any leading zeros) and that `options.shm_dir` is an absolute,
// non-empty directory path with trailing slashes stripped, rejecting the root
// path "/".
ExporterOptions NormalizeAndValidateOptions(ExporterOptions options) {
  CHECK(options.local_rank.has_value() && !options.local_rank->empty())
      << "options.local_rank must be specified and non-empty for "
         "BaseShmExporter";

  int rank_val = -1;
  CHECK(absl::SimpleAtoi(*options.local_rank, &rank_val) && rank_val >= 0)
      << "options.local_rank must be a valid non-negative integer for "
         "BaseShmExporter, got: '"
      << *options.local_rank << "'";
  options.local_rank = absl::StrCat(rank_val);

  CHECK(options.shm_dir.has_value() && !options.shm_dir->empty())
      << "options.shm_dir must be specified and non-empty for BaseShmExporter";
  CHECK(options.shm_dir->starts_with('/'))
      << "options.shm_dir must be an absolute path, got: '" << *options.shm_dir
      << "'";

  while (options.shm_dir->ends_with('/')) {
    options.shm_dir->pop_back();
  }
  CHECK(!options.shm_dir->empty())
      << "options.shm_dir must not be the root path '/'";

  return options;
}

}  // namespace

BaseShmExporter::BaseShmExporter(ExporterOptions options)
    : options_(NormalizeAndValidateOptions(std::move(options))),
      shm_writer_(ShmWriterOptions{
          .shm_dir = *options_.shm_dir,
          .local_rank = *options_.local_rank,
      }),
      shm_collector_(ShmCollectorOptions{
          .shm_dir = *options_.shm_dir,
      }) {}

BaseShmExporter::~BaseShmExporter() = default;

void BaseShmExporter::IncrementCounter(absl::string_view name, LabelSpan labels,
                                       uint64_t val) const {
  shm_writer_.IncrementCounter(name, labels, val);
}

void BaseShmExporter::SetGauge(absl::string_view name, LabelSpan labels,
                               double val) const {
  if (!std::isfinite(val)) return;

  const MetricLabel rank_label = LocalRankLabel();
  if (labels.empty()) {
    shm_writer_.SetGauge(name, LabelSpan(&rank_label, 1), val);
    return;
  }
  if (absl::c_any_of(labels, [](const MetricLabel& label) {
        return label.key == metric_labels::kLocalRank;
      })) {
    shm_writer_.SetGauge(name, labels, val);
    return;
  }
  absl::InlinedVector<MetricLabel, kDefaultInlinedLabelCapacity + 1>
      augmented_labels(labels.begin(), labels.end());
  if (absl::c_is_sorted(labels)) {
    const auto it = absl::c_lower_bound(augmented_labels, rank_label);
    augmented_labels.insert(it, rank_label);
  } else {
    augmented_labels.push_back(rank_label);
    absl::c_sort(augmented_labels);
  }
  shm_writer_.SetGauge(name, augmented_labels, val);
}

void BaseShmExporter::ObserveHistogram(absl::string_view name, LabelSpan labels,
                                       double val) const {
  shm_writer_.ObserveHistogram(name, labels, val);
}

std::string BaseShmExporter::GetTextSnapshot() const { return std::string(); }

AggregatedMetrics BaseShmExporter::CollectMetrics() const {
  return shm_collector_.CollectMetrics();
}

}  // namespace tpu_raiden::telemetry

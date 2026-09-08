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

#include "tpu_sync/telemetry/prometheus_shm_exporter.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "prometheus/client_metric.h"
#include "prometheus/collectable.h"
#include "prometheus/exposer.h"
#include "prometheus/metric_family.h"
#include "prometheus/metric_type.h"
#include "prometheus/text_serializer.h"
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/base_shm_exporter.h"
#include "tpu_sync/telemetry/exporter_util.h"
#include "tpu_sync/telemetry/label_util.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {
namespace {

// Parses compact semicolon-encoded SHM labels into Prometheus ClientMetric
// labels, reusing ParseShmLabels() and sorting label pairs lexicographically by
// name per the Prometheus / OpenMetrics exposition standard.
std::vector<prometheus::ClientMetric::Label> ParseLabels(
    absl::string_view encoded_labels) {
  if (encoded_labels.empty()) {
    return {};
  }
  std::vector<std::pair<std::string, std::string>> raw_label_pairs =
      ParseShmLabels(encoded_labels);
  std::vector<prometheus::ClientMetric::Label> labels;
  labels.reserve(raw_label_pairs.size());
  for (auto& [key, value] : raw_label_pairs) {
    labels.push_back({.name = std::move(key), .value = std::move(value)});
  }
  absl::c_sort(labels);
  return labels;
}

prometheus::MetricFamily CreateFamily(const MetricMetadata& metadata) {
  prometheus::MetricFamily family;
  family.name = absl::StrCat(kPrometheusMetricPrefix, metadata.name);
  family.help = metadata.description;
  switch (metadata.type) {
    case MetricType::kCounter:
      family.type = prometheus::MetricType::Counter;
      break;
    case MetricType::kGauge:
      family.type = prometheus::MetricType::Gauge;
      break;
    case MetricType::kHistogram:
      family.type = prometheus::MetricType::Histogram;
      break;
  }
  return family;
}

// Converts aggregated shared memory metrics into Prometheus MetricFamily
// structures for exposition.
std::vector<prometheus::MetricFamily> CollectMetricFamilies(
    const AggregatedMetrics& metrics) {
  if (metrics.empty()) {
    return {};
  }

  std::vector<prometheus::MetricFamily> families;
  families.reserve(std::size(metric_metadata::kAllMetrics));

  for (const MetricMetadata& metadata : metric_metadata::kAllMetrics) {
    switch (metadata.type) {
      case MetricType::kCounter: {
        auto counter_it = metrics.counters.find(metadata.name);
        if (counter_it == metrics.counters.end() ||
            counter_it->second.empty()) {
          break;
        }
        const auto& label_counts = counter_it->second;
        prometheus::MetricFamily family = CreateFamily(metadata);
        family.metric.reserve(label_counts.size());
        for (const auto& [encoded_labels, count] : label_counts) {
          prometheus::ClientMetric& metric = family.metric.emplace_back();
          metric.label = ParseLabels(encoded_labels);
          metric.counter.value = static_cast<double>(count);
        }
        families.push_back(std::move(family));
        break;
      }
      case MetricType::kGauge: {
        auto gauge_it = metrics.gauges.find(metadata.name);
        if (gauge_it == metrics.gauges.end() || gauge_it->second.empty()) {
          break;
        }
        const auto& label_values = gauge_it->second;
        prometheus::MetricFamily family = CreateFamily(metadata);
        family.metric.reserve(label_values.size());
        for (const auto& [encoded_labels, gauge_value] : label_values) {
          prometheus::ClientMetric& metric = family.metric.emplace_back();
          metric.label = ParseLabels(encoded_labels);
          metric.gauge.value = gauge_value;
        }
        families.push_back(std::move(family));
        break;
      }
      case MetricType::kHistogram: {
        auto hist_it = metrics.histograms.find(metadata.name);
        if (hist_it == metrics.histograms.end() || hist_it->second.empty()) {
          break;
        }
        const auto& label_histograms = hist_it->second;
        prometheus::MetricFamily family = CreateFamily(metadata);
        family.metric.reserve(label_histograms.size());
        for (const auto& [encoded_labels, histogram_data] : label_histograms) {
          prometheus::ClientMetric& metric = family.metric.emplace_back();
          metric.label = ParseLabels(encoded_labels);
          std::erase_if(metric.label,
                        [](const prometheus::ClientMetric::Label& label) {
                          return label.name == kPrometheusLeLabel;
                        });

          metric.histogram.sample_sum = histogram_data.sample_sum;
          metric.histogram.bucket.reserve(kNumHistogramBuckets + 1);

          uint64_t cumulative_count = 0;
          for (size_t bucket_idx = 0; bucket_idx < kNumHistogramBuckets;
               ++bucket_idx) {
            cumulative_count += histogram_data.bucket_counts[bucket_idx];
            metric.histogram.bucket.push_back({
                .cumulative_count = cumulative_count,
                .upper_bound = kDefaultHistogramBuckets[bucket_idx],
            });
          }
          // Add +Inf bucket
          cumulative_count +=
              histogram_data.bucket_counts[kNumHistogramBuckets];
          metric.histogram.bucket.push_back({
              .cumulative_count = cumulative_count,
              .upper_bound = std::numeric_limits<double>::infinity(),
          });
          metric.histogram.sample_count = cumulative_count;
        }
        families.push_back(std::move(family));
        break;
      }
    }
  }

  // Sort metrics within families
  for (prometheus::MetricFamily& family : families) {
    absl::c_sort(family.metric, [](const prometheus::ClientMetric& metric_a,
                                   const prometheus::ClientMetric& metric_b) {
      return metric_a.label < metric_b.label;
    });
  }

  return families;
}

// Adapter class implementing prometheus::Collectable by querying the exporter's
// aggregated metrics and converting them via CollectMetricFamilies().
class PrometheusShmCollectable final : public prometheus::Collectable {
 public:
  explicit PrometheusShmCollectable(const PrometheusShmExporter& exporter)
      : exporter_(exporter) {}
  std::vector<prometheus::MetricFamily> Collect() const override {
    return CollectMetricFamilies(exporter_.CollectMetrics());
  }

 private:
  const PrometheusShmExporter& exporter_;
};

}  // namespace

// Holds HTTP exposition state.
// Note: `collectable` is declared before `exposer` so that `exposer` destructs
// first (in reverse declaration order), synchronously stopping the CivetWeb
// server and joining worker threads before `collectable` is released.
struct PrometheusShmExporter::ExposerState {
  std::shared_ptr<prometheus::Collectable> collectable;
  std::unique_ptr<prometheus::Exposer> exposer;
};

PrometheusShmExporter::PrometheusShmExporter(ExporterOptions exporter_options)
    : BaseShmExporter(std::move(exporter_options)) {
  const int port = options().port;
  if (port >= kMinPort && port <= kMaxPort) {
    const std::string endpoint = JoinHostPort(options().bind_address, port);
    try {
      std::unique_ptr<prometheus::Exposer> exposer =
          std::make_unique<prometheus::Exposer>(endpoint);
      std::shared_ptr<prometheus::Collectable> collectable =
          std::make_shared<PrometheusShmCollectable>(*this);
      exposer->RegisterCollectable(collectable);
      exposer_state_ = std::make_unique<ExposerState>();
      exposer_state_->collectable = std::move(collectable);
      exposer_state_->exposer = std::move(exposer);
      LOG(INFO) << "Prometheus SHM exporter listening on http://" << endpoint
                << "/metrics";
    } catch (const std::exception& e) {
      LOG(INFO) << "Failed to bind Prometheus HTTP exposer on " << endpoint
                << ": " << e.what();
      exposer_state_.reset();
    }
  } else if (port > 0) {
    LOG(WARNING) << "Invalid port configured for Prometheus SHM HTTP exposer: "
                 << port << ". Expected port in range [" << kMinPort << ", "
                 << kMaxPort << "].";
  }
}

PrometheusShmExporter::~PrometheusShmExporter() = default;

bool PrometheusShmExporter::IsServerRunningForTesting() const {
  return exposer_state_ != nullptr && exposer_state_->exposer != nullptr;
}

std::string PrometheusShmExporter::GetTextSnapshot() const {
  return prometheus::TextSerializer().Serialize(
      CollectMetricFamilies(CollectMetrics()));
}

}  // namespace tpu_raiden::telemetry

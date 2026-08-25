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

#include <algorithm>
#include <cstdint>
#include <exception>
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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/telemetry/base_shm_exporter.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {
namespace {

std::string JoinHostPort(absl::string_view host, int port) {
  if (absl::StrContains(host, ':') && !absl::StartsWith(host, "[")) {
    return absl::StrCat("[", host, "]:", port);
  }
  return absl::StrCat(host, ":", port);
}

std::vector<prometheus::ClientMetric::Label> ParseLabels(
    absl::string_view enc) {
  std::vector<prometheus::ClientMetric::Label> labels;
  if (enc.empty()) return labels;
  for (absl::string_view p : absl::StrSplit(enc, ';')) {
    std::pair<absl::string_view, absl::string_view> kv =
        absl::StrSplit(p, absl::MaxSplits('=', 1));
    if (!kv.first.empty()) {
      labels.push_back({std::string(kv.first), std::string(kv.second)});
    }
  }
  std::sort(labels.begin(), labels.end());
  return labels;
}

class PrometheusShmCollectable : public prometheus::Collectable {
 public:
  explicit PrometheusShmCollectable(const PrometheusShmExporter* exp)
      : exp_(exp) {}
  std::vector<prometheus::MetricFamily> Collect() const override {
    return exp_ ? exp_->CollectMetricFamilies()
                : std::vector<prometheus::MetricFamily>{};
  }

 private:
  const PrometheusShmExporter* exp_;
};

}  // namespace

PrometheusShmExporter::PrometheusShmExporter(const ExporterOptions& options)
    : BaseShmExporter(options) {
  if (options_.port > 0) Start();
}

PrometheusShmExporter::~PrometheusShmExporter() { Stop(); }

void PrometheusShmExporter::Start() {
  BaseShmExporter::Start();
  absl::MutexLock lock(&mutex_);
  if (exposer_ != nullptr || options_.port <= 0) return;

  if (options_.port >= kMinPort && options_.port <= kMaxPort) {
    std::string endpoint = JoinHostPort(options_.bind_address, options_.port);
    try {
      exposer_ = std::make_unique<prometheus::Exposer>(endpoint);
      collectable_ = std::make_shared<PrometheusShmCollectable>(this);
      exposer_->RegisterCollectable(collectable_);
      LOG(INFO) << "Prometheus SHM exporter listening on http://" << endpoint
                << "/metrics";
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to bind Prometheus HTTP exposer on " << endpoint
                 << ": " << e.what();
      exposer_.reset();
      collectable_.reset();
    }
  } else {
    LOG(WARNING) << "Invalid port configured for Prometheus SHM HTTP exposer: "
                 << options_.port;
  }
}

void PrometheusShmExporter::Stop() {
  absl::MutexLock lock(&mutex_);
  exposer_.reset();
  collectable_.reset();
  BaseShmExporter::Stop();
}

bool PrometheusShmExporter::IsExposerRunning() const {
  absl::MutexLock lock(&mutex_);
  return exposer_ != nullptr;
}

std::vector<prometheus::MetricFamily>
PrometheusShmExporter::CollectMetricFamilies() const {
  std::vector<prometheus::MetricFamily> families;
  if (!collector_) return families;

  absl::flat_hash_map<std::string, double> totals;
  collector_->CollectMetrics(totals);

  for (const MetricMetadata& meta : metric_metadata::kAllMetrics) {
    prometheus::MetricFamily family;
    family.name = absl::StrCat("tpu_raiden_", meta.name);
    family.help = std::string(meta.description);
    family.type =
        (meta.type == MetricType::kCounter) ? prometheus::MetricType::Counter
        : (meta.type == MetricType::kGauge) ? prometheus::MetricType::Gauge
                                            : prometheus::MetricType::Histogram;

    std::string prefix = absl::StrCat(meta.name, "/");

    if (meta.type == MetricType::kCounter || meta.type == MetricType::kGauge) {
      for (const auto& [raw_key, val] : totals) {
        if (!absl::StartsWith(raw_key, prefix)) continue;
        prometheus::ClientMetric cm;
        cm.label = ParseLabels(raw_key.substr(prefix.length()));
        if (meta.type == MetricType::kCounter) {
          cm.counter.value = val;
        } else {
          cm.gauge.value = val;
        }
        family.metric.push_back(std::move(cm));
      }
    } else if (meta.type == MetricType::kHistogram) {
      absl::flat_hash_set<std::string> seen_enc;
      for (const auto& [raw_key, _] : totals) {
        if (!absl::StartsWith(raw_key, prefix)) continue;
        std::string sub = raw_key.substr(prefix.length());
        size_t slash_pos = sub.find('/');
        std::string enc =
            (slash_pos != std::string::npos) ? sub.substr(0, slash_pos) : "";
        if (!seen_enc.insert(enc).second) continue;

        prometheus::ClientMetric cm;
        cm.label = ParseLabels(enc);
        std::string base_key = absl::StrCat(prefix, enc);
        auto count_it = totals.find(absl::StrCat(base_key, "/count"));
        if (count_it != totals.end()) {
          cm.histogram.sample_count = static_cast<uint64_t>(count_it->second);
        }
        auto sum_it = totals.find(absl::StrCat(base_key, "/sum"));
        if (sum_it != totals.end()) cm.histogram.sample_sum = sum_it->second;
        for (size_t b = 0; b < kNumHistogramBuckets; ++b) {
          auto b_it = totals.find(absl::StrCat(base_key, "/bucket_", b));
          uint64_t b_count =
              (b_it != totals.end()) ? static_cast<uint64_t>(b_it->second) : 0;
          cm.histogram.bucket.push_back({b_count, kDefaultHistogramBuckets[b]});
        }
        family.metric.push_back(std::move(cm));
      }
    }

    std::sort(
        family.metric.begin(), family.metric.end(),
        [](const prometheus::ClientMetric& a,
           const prometheus::ClientMetric& b) { return a.label < b.label; });
    families.push_back(std::move(family));
  }
  return families;
}

std::string PrometheusShmExporter::GetTextSnapshot() const {
  return prometheus::TextSerializer().Serialize(CollectMetricFamilies());
}

}  // namespace tpu_raiden::telemetry

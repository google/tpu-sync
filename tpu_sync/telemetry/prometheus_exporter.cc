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

#include "tpu_sync/telemetry/prometheus_exporter.h"

#include <cstdint>
#include <exception>
#include <map>  // NOLINT: Required by prometheus-cpp client API.
#include <memory>
#include <string>
#include <utility>

#include "prometheus/counter.h"
#include "prometheus/exposer.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

namespace {

constexpr absl::string_view kMetricPrefix = "tpu_raiden_";

std::map<std::string, std::string> ConvertLabels(LabelSpan labels) {
  if (labels.empty()) {
    return {};
  }
  std::map<std::string, std::string> result;
  for (const auto& [key, value] : labels) {
    result.emplace(key, value);
  }
  return result;
}

std::string JoinHostPort(absl::string_view host, int port) {
  if (absl::StrContains(host, ':') && !absl::StartsWith(host, "[")) {
    return absl::StrCat("[", host, "]:", port);
  }
  return absl::StrCat(host, ":", port);
}

}  // namespace

void PrometheusExporter::RegisterKnownFamilies() {
  for (const auto& meta : metric_metadata::kAllMetrics) {
    if (counter_families_.contains(meta.name) ||
        gauge_families_.contains(meta.name) ||
        histogram_families_.contains(meta.name)) {
      continue;
    }
    std::string prometheus_name = absl::StrCat(kMetricPrefix, meta.name);
    switch (meta.type) {
      case MetricType::kCounter: {
        auto* family = &prometheus::BuildCounter()
                            .Name(prometheus_name)
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        family->Add({});
        counter_families_.emplace(meta.name, family);
        break;
      }
      case MetricType::kGauge: {
        auto* family = &prometheus::BuildGauge()
                            .Name(prometheus_name)
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        family->Add({});
        gauge_families_.emplace(meta.name, family);
        break;
      }
      case MetricType::kHistogram: {
        auto* family = &prometheus::BuildHistogram()
                            .Name(prometheus_name)
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        family->Add({}, default_buckets_);
        histogram_families_.emplace(meta.name, family);
        break;
      }
    }
  }
}

PrometheusExporter::PrometheusExporter(const ExporterOptions& options)
    : registry_(std::make_shared<prometheus::Registry>()),
      default_buckets_(options.custom_buckets.begin(),
                       options.custom_buckets.end()),
      options_(options) {
  RegisterKnownFamilies();

  if (options_.port >= kMinPort && options_.port <= kMaxPort) {
    std::string endpoint = JoinHostPort(options_.bind_address, options_.port);
    // prometheus-cpp Exposer throws std::runtime_error on socket binding or
    // initialization failure. Catching here prevents abnormal termination and
    // allows graceful degradation with IsServerRunning() reporting false.
    try {
      exposer_ = std::make_unique<prometheus::Exposer>(endpoint);
      exposer_->RegisterCollectable(registry_);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to start Prometheus HTTP exporter on " << endpoint
                 << ": " << e.what();
      exposer_.reset();
    }
  } else if (options_.port != 0) {
    LOG(WARNING) << "Invalid port configured for Prometheus HTTP exporter: "
                 << options_.port << ". Expected port in range [" << kMinPort
                 << ", " << kMaxPort << "].";
  }
}

PrometheusExporter::~PrometheusExporter() = default;

bool PrometheusExporter::IsServerRunning() const { return exposer_ != nullptr; }

prometheus::Family<prometheus::Counter>* PrometheusExporter::GetCounterFamily(
    absl::string_view name) const {
  auto it = counter_families_.find(name);
  if (it == counter_families_.end()) {
    return nullptr;
  }
  return it->second;
}

prometheus::Family<prometheus::Gauge>* PrometheusExporter::GetGaugeFamily(
    absl::string_view name) const {
  auto it = gauge_families_.find(name);
  if (it == gauge_families_.end()) {
    return nullptr;
  }
  return it->second;
}

prometheus::Family<prometheus::Histogram>*
PrometheusExporter::GetHistogramFamily(absl::string_view name) const {
  auto it = histogram_families_.find(name);
  if (it == histogram_families_.end()) {
    return nullptr;
  }
  return it->second;
}

void PrometheusExporter::IncrementCounter(absl::string_view name,
                                          LabelSpan labels,
                                          uint64_t val) const {
  prometheus::Family<prometheus::Counter>* family = GetCounterFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Counter& counter = family->Add(ConvertLabels(labels));
  counter.Increment(static_cast<double>(val));
}

void PrometheusExporter::SetGauge(absl::string_view name, LabelSpan labels,
                                  double val) const {
  prometheus::Family<prometheus::Gauge>* family = GetGaugeFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Gauge& gauge = family->Add(ConvertLabels(labels));
  gauge.Set(val);
}

void PrometheusExporter::ObserveHistogram(absl::string_view name,
                                          LabelSpan labels, double val) const {
  prometheus::Family<prometheus::Histogram>* family = GetHistogramFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Histogram& histogram =
      family->Add(ConvertLabels(labels), default_buckets_);
  histogram.Observe(val);
}

std::string PrometheusExporter::GetTextSnapshot() const {
  prometheus::TextSerializer serializer;
  return serializer.Serialize(registry_->Collect());
}

}  // namespace tpu_raiden::telemetry

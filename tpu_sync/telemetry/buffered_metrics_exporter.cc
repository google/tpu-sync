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

#include "tpu_sync/telemetry/buffered_metrics_exporter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

namespace {

constexpr absl::string_view kMetricPrefix = "tpu_raiden_";
constexpr size_t kDefaultInlinedLabelCapacity = 4;

std::string EscapeLabelValue(absl::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
        escaped.push_back('\\');
        escaped.push_back('\\');
        break;
      case '"':
        escaped.push_back('\\');
        escaped.push_back('"');
        break;
      case '\n':
        escaped.push_back('\\');
        escaped.push_back('n');
        break;
      default:
        escaped.push_back(c);
        break;
    }
  }
  return escaped;
}

}  // namespace

std::string FormatCanonicalLabels(LabelSpan labels) {
  if (labels.empty()) {
    return "";
  }
  absl::InlinedVector<MetricLabel, kDefaultInlinedLabelCapacity> sorted_labels(
      labels.begin(), labels.end());
  std::sort(sorted_labels.begin(), sorted_labels.end());

  std::string out;
  out.push_back('{');
  bool first = true;
  for (const auto& [key, value] : sorted_labels) {
    if (!first) {
      out.push_back(',');
    }
    first = false;
    out.append(key);
    out.push_back('=');
    out.push_back('"');
    out.append(EscapeLabelValue(value));
    out.push_back('"');
  }
  out.push_back('}');
  return out;
}

BufferedMetricsExporter::BufferedMetricsExporter(
    absl::Span<const MetricMetadata> metrics) {
  for (const auto& meta : metrics) {
    switch (meta.type) {
      case MetricType::kCounter:
        counters_.emplace(
            meta.name,
            std::make_unique<
                MetricFamilyBuffer<LockFreeCounterAccumulator>>());
        break;
      case MetricType::kGauge:
        gauges_.emplace(
            meta.name, std::make_unique<MetricFamilyBuffer<QueueBuffer<>>>());
        break;
      case MetricType::kHistogram:
        histograms_.emplace(
            meta.name, std::make_unique<MetricFamilyBuffer<QueueBuffer<>>>());
        break;
    }
  }
}

void BufferedMetricsExporter::IncrementCounter(absl::string_view name,
                                               LabelSpan labels,
                                               uint64_t val) const {
  auto it = counters_.find(name);
  if (it == counters_.end()) {
    return;
  }
  if (auto* acc = it->second->GetOrCreate(labels)) {
    acc->Add(val);
  }
}

void BufferedMetricsExporter::SetGauge(absl::string_view name, LabelSpan labels,
                                       double val) const {
  auto it = gauges_.find(name);
  if (it == gauges_.end()) {
    return;
  }
  if (auto* buf = it->second->GetOrCreate(labels)) {
    buf->Push(val);
  }
}

void BufferedMetricsExporter::ObserveHistogram(absl::string_view name,
                                               LabelSpan labels,
                                               double val) const {
  auto it = histograms_.find(name);
  if (it == histograms_.end()) {
    return;
  }
  if (auto* buf = it->second->GetOrCreate(labels)) {
    buf->Push(val);
  }
}

std::map<std::string, std::vector<double>>
BufferedMetricsExporter::GetAndResetMetricSamples() {
  std::map<std::string, std::vector<double>> result;

  for (const auto& [name, family_buffer] : counters_) {
    family_buffer->ForEachAccumulator(
        [&](absl::string_view canonical_labels,
            LockFreeCounterAccumulator* counter) {
          uint64_t delta = counter->ExchangeAndReset();
          if (delta > 0) {
            std::string full_name =
                absl::StrCat(kMetricPrefix, name, canonical_labels);
            result[full_name].push_back(static_cast<double>(delta));
          }
        });
  }

  for (const auto& [name, family_buffer] : gauges_) {
    family_buffer->ForEachAccumulator(
        [&](absl::string_view canonical_labels, QueueBuffer<>* gauge) {
          std::vector<double> samples = gauge->ExtractAndReset();
          if (!samples.empty()) {
            std::string full_name =
                absl::StrCat(kMetricPrefix, name, canonical_labels);
            result[full_name] = std::move(samples);
          }
        });
  }

  for (const auto& [name, family_buffer] : histograms_) {
    family_buffer->ForEachAccumulator(
        [&](absl::string_view canonical_labels, QueueBuffer<>* histogram) {
          std::vector<double> samples = histogram->ExtractAndReset();
          if (!samples.empty()) {
            std::string full_name =
                absl::StrCat(kMetricPrefix, name, canonical_labels);
            result[full_name] = std::move(samples);
          }
        });
  }

  return result;
}

}  // namespace tpu_raiden::telemetry

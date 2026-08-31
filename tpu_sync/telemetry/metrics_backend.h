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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_BACKEND_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden::telemetry {

enum class MetricType : uint8_t {
  kCounter,
  kGauge,
  kHistogram,
};

inline constexpr double kDefaultHistogramBuckets[] = {
    0.1,    0.25,   0.5,    1.0,     2.5,     5.0,    10.0,
    25.0,   50.0,   100.0,  250.0,   500.0,   750.0,  1000.0,
    2500.0, 5000.0, 7500.0, 10000.0, 25000.0, 50000.0};

inline constexpr int kDefaultExporterPort = 0;
inline constexpr absl::string_view kDefaultExporterHost = "0.0.0.0";
inline constexpr int kMinPort = 1;
inline constexpr int kMaxPort = 65535;

// Configuration options for metric exporters.
struct ExporterOptions {
  // Bind address for HTTP metric exporter. Defaults to "0.0.0.0".
  std::string bind_address{kDefaultExporterHost};
  // TCP port for HTTP metric exporter. If <= 0, HTTP serving is disabled.
  int port = kDefaultExporterPort;
  // Non-owning view of histogram bucket boundaries. Defaults to
  // kDefaultHistogramBuckets and is copied by the exporter during construction.
  absl::Span<const double> custom_buckets = kDefaultHistogramBuckets;
  // Worker local rank identifier in distributed multi-rank environments. If
  // unset (std::nullopt) or empty, telemetry initialization falls back to the
  // LOCAL_RANK environment variable.
  std::optional<std::string> local_rank;
};

// Structure defining centralized metadata for a Raiden metric.
struct MetricMetadata {
  absl::string_view name;
  absl::string_view description;
  MetricType type;
  absl::Span<const double> buckets = kDefaultHistogramBuckets;
  absl::Span<const absl::string_view> label_names = {};

  // Enables compile-time member-wise value equality comparisons and test
  // assertions with zero runtime overhead.
  bool operator==(const MetricMetadata& other) const = default;
};

namespace metric_names {

inline constexpr absl::string_view kSentBytesTotal = "sent_bytes_total";
inline constexpr absl::string_view kReceivedBytesTotal = "received_bytes_total";
inline constexpr absl::string_view kTransferFailuresTotal =
    "transfer_failures_total";
inline constexpr absl::string_view kTransferDurationMs = "transfer_duration_ms";

inline constexpr absl::string_view kH2dTransferTimeMs = "h2d_transfer_time_ms";
inline constexpr absl::string_view kD2hTransferTimeMs = "d2h_transfer_time_ms";
inline constexpr absl::string_view kBufferAllocatedBytes =
    "buffer_allocated_bytes";

}  // namespace metric_names

namespace metric_descriptions {

inline constexpr absl::string_view kSentBytesTotal =
    "Total count of bytes sent over TPU Raiden interfaces.";
inline constexpr absl::string_view kReceivedBytesTotal =
    "Total count of bytes received over TPU Raiden interfaces.";
inline constexpr absl::string_view kTransferFailuresTotal =
    "Cumulative total count of transfer failures across all interfaces.";
inline constexpr absl::string_view kTransferDurationMs =
    "Measures End-to-End (E2E) latency bound around the entire request in "
    "milliseconds, including setup delays.";
inline constexpr absl::string_view kBufferAllocatedBytes =
    "Current host DRAM buffer capacity allocated in bytes for KV cache staging "
    "across all layers and shards.";

inline constexpr absl::string_view kH2dTransferTimeMs =
    "Host-to-Device transfer latency in milliseconds.";
inline constexpr absl::string_view kD2hTransferTimeMs =
    "Device-to-Host transfer latency in milliseconds.";

}  // namespace metric_descriptions

namespace metric_metadata {

inline constexpr MetricMetadata kSentBytesTotal{
    .name = metric_names::kSentBytesTotal,
    .description = metric_descriptions::kSentBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kReceivedBytesTotal{
    .name = metric_names::kReceivedBytesTotal,
    .description = metric_descriptions::kReceivedBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kTransferFailuresTotal{
    .name = metric_names::kTransferFailuresTotal,
    .description = metric_descriptions::kTransferFailuresTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kTransferDurationMs{
    .name = metric_names::kTransferDurationMs,
    .description = metric_descriptions::kTransferDurationMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kH2dTransferTimeMs{
    .name = metric_names::kH2dTransferTimeMs,
    .description = metric_descriptions::kH2dTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kD2hTransferTimeMs{
    .name = metric_names::kD2hTransferTimeMs,
    .description = metric_descriptions::kD2hTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kBufferAllocatedBytes{
    .name = metric_names::kBufferAllocatedBytes,
    .description = metric_descriptions::kBufferAllocatedBytes,
    .type = MetricType::kGauge};

inline constexpr MetricMetadata kAllMetrics[] = {
    kSentBytesTotal,       kReceivedBytesTotal, kTransferFailuresTotal,
    kTransferDurationMs,   kH2dTransferTimeMs,  kD2hTransferTimeMs,
    kBufferAllocatedBytes,
};
}  // namespace metric_metadata

namespace metric_labels {

inline constexpr absl::string_view kDirection = "direction";
inline constexpr absl::string_view kDirectionPush = "push";
inline constexpr absl::string_view kDirectionPull = "pull";
inline constexpr absl::string_view kDirectionPullResponse = "pull_response";

inline constexpr absl::string_view kErrorCode = "error_code";
}  // namespace metric_labels

// Structure defining a metric key-value label pair.
struct MetricLabel {
  absl::string_view key;
  absl::string_view value;

  auto operator<=>(const MetricLabel& other) const = default;
};

// Allocation-free label view span type definition
using LabelSpan = absl::Span<const MetricLabel>;

// Abstract Dual-Backend Interface
class MetricsBackend {
 public:
  MetricsBackend() = default;
  MetricsBackend(const MetricsBackend&) = delete;
  MetricsBackend& operator=(const MetricsBackend&) = delete;
  MetricsBackend(MetricsBackend&&) = delete;
  MetricsBackend& operator=(MetricsBackend&&) = delete;

  virtual ~MetricsBackend() = default;

  virtual void IncrementCounter(absl::string_view name, LabelSpan labels,
                                uint64_t val) const = 0;

  virtual void SetGauge(absl::string_view name, LabelSpan labels,
                        double val) const = 0;

  virtual void ObserveHistogram(absl::string_view name, LabelSpan labels,
                                double val) const = 0;

  virtual std::string GetTextSnapshot() const = 0;

  virtual std::map<std::string, std::vector<double>>
  GetAndResetMetricSamples() {
    return {};
  }
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_BACKEND_H_

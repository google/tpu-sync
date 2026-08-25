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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_API_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_API_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

// Environment Variables START.
inline constexpr char kTelemetryBackendsEnvVar[] =
    "TPU_RAIDEN_TELEMETRY_BACKENDS";
inline constexpr char kPrometheusPortEnvVar[] = "TPU_RAIDEN_PROMETHEUS_PORT";
inline constexpr char kPrometheusHostEnvVar[] = "TPU_RAIDEN_PROMETHEUS_HOST";
inline constexpr char kLocalRankEnvVar[] = "LOCAL_RANK";
// Environment Variables END.

// Backend names START.
inline constexpr absl::string_view kPrometheus = "prometheus";
inline constexpr absl::string_view kBuffered = "buffered";
// Backend names END.

// Central Telemetry Facade for managing metrics across registered backends.
// This class is thread-safe for all concurrent operations.
class RaidenMetricStore {
 public:
  static RaidenMetricStore& GetGlobalMetricStore();

  RaidenMetricStore() = default;
  ~RaidenMetricStore() = default;

  RaidenMetricStore(const RaidenMetricStore&) = delete;
  RaidenMetricStore& operator=(const RaidenMetricStore&) = delete;
  RaidenMetricStore(RaidenMetricStore&&) = delete;
  RaidenMetricStore& operator=(RaidenMetricStore&&) = delete;

  // Inspects the TPU_RAIDEN_TELEMETRY_BACKENDS environment variable and
  // initializes backends. Returns OkStatus() if backends are already
  // configured, if the variable is unset, or on success. Returns
  // InvalidArgumentError if any backend token in the environment variable is
  // invalid.
  [[nodiscard]] absl::Status InitializeFromEnvironment();

  // Validates, instantiates, and sets backends matching the provided names.
  // Returns absl::InvalidArgumentError if any backend name is unrecognized.
  // Performs all-or-nothing assignment: backends remain unchanged on error.
  [[nodiscard]] absl::Status InitializeFromBackendNames(
      absl::Span<const absl::string_view> backend_names);

  void SetBackends(std::vector<std::unique_ptr<MetricsBackend>> backends);
  bool HasBackends() const;

  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val = 1) const;

  void SetGauge(absl::string_view name, LabelSpan labels, double val) const;

  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const;

  std::string GetTextSnapshot() const;

  std::vector<MetricMetadata> GetMetricMetadata() const;

  std::map<std::string, std::vector<double>> GetAndResetMetricSamples();

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::unique_ptr<MetricsBackend>> backends_
      ABSL_GUARDED_BY(mutex_);
  std::atomic<bool> has_backends_{false};
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_API_H_

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

#include "tpu_sync/telemetry/metrics_api.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/buffered_metrics_exporter.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/prometheus_exporter.h"

namespace tpu_raiden::telemetry {

namespace {

int ResolveExporterPort() {
  const char* env_port = std::getenv(kPrometheusPortEnvVar);
  if (env_port != nullptr && *env_port != '\0') {
    int parsed_port = 0;
    if (absl::SimpleAtoi(env_port, &parsed_port) && parsed_port >= kMinPort &&
        parsed_port <= kMaxPort) {
      return parsed_port;
    }
    LOG(WARNING) << "Invalid port specified in " << kPrometheusPortEnvVar
                 << ": '" << env_port << "'. Expected integer in range ["
                 << kMinPort << ", " << kMaxPort
                 << "]. Falling back to default port (" << kDefaultExporterPort
                 << ").";
  }
  return kDefaultExporterPort;
}

std::string ResolveExporterHost() {
  const char* env_host = std::getenv(kPrometheusHostEnvVar);
  if (env_host != nullptr && *env_host != '\0') {
    return std::string(env_host);
  }
  return std::string(kDefaultExporterHost);
}

std::optional<std::string> ResolveLocalRank() {
  const char* env_rank = std::getenv(kLocalRankEnvVar);
  if (env_rank != nullptr && *env_rank != '\0') {
    absl::string_view trimmed = absl::StripAsciiWhitespace(env_rank);
    if (!trimmed.empty()) {
      return std::string(trimmed);
    }
  }
  return std::nullopt;
}

}  // namespace

RaidenMetricStore& RaidenMetricStore::GetGlobalMetricStore() {
  static absl::NoDestructor<RaidenMetricStore> global_store;
  static const bool initialized = [&] {
    absl::Status status = global_store->InitializeFromEnvironment();
    if (!status.ok()) {
      LOG(WARNING)
          << "Failed to initialize telemetry backends from environment: "
          << status;
    }
    return true;
  }();
  (void)initialized;
  return *global_store;
}

void RaidenMetricStore::SetBackends(
    std::vector<std::unique_ptr<MetricsBackend>> backends) {
  std::erase_if(backends, [](const std::unique_ptr<MetricsBackend>& backend) {
    return backend == nullptr;
  });
  absl::MutexLock lock(mutex_);
  backends_ = std::move(backends);
  has_backends_.store(!backends_.empty(), std::memory_order_release);
}

bool RaidenMetricStore::HasBackends() const {
  return has_backends_.load(std::memory_order_acquire);
}

absl::Status RaidenMetricStore::InitializeFromBackendNames(
    absl::Span<const absl::string_view> backend_names) {
  std::vector<std::unique_ptr<MetricsBackend>> new_backends;
  absl::flat_hash_set<std::string> seen_backends;

  for (absl::string_view backend_key : backend_names) {
    std::string name =
        absl::AsciiStrToLower(absl::StripAsciiWhitespace(backend_key));
    if (name.empty() || !seen_backends.insert(name).second) {
      continue;
    }
    if (name == kPrometheus) {
      new_backends.push_back(
          std::make_unique<PrometheusExporter>(ExporterOptions{
              .bind_address = ResolveExporterHost(),
              .port = ResolveExporterPort(),
              .local_rank = ResolveLocalRank(),
          }));
    } else if (name == kBuffered) {
      new_backends.push_back(std::make_unique<BufferedMetricsExporter>());
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown telemetry backend: ", backend_key));
    }
  }

  SetBackends(std::move(new_backends));
  return absl::OkStatus();
}

absl::Status RaidenMetricStore::InitializeFromEnvironment() {
  if (HasBackends()) {
    return absl::OkStatus();
  }
  const char* env = std::getenv(kTelemetryBackendsEnvVar);
  if (env == nullptr) {
    return absl::OkStatus();
  }
  std::vector<absl::string_view> backend_names =
      absl::StrSplit(env, absl::ByChar(','), absl::SkipWhitespace());
  return InitializeFromBackendNames(backend_names);
}

void RaidenMetricStore::IncrementCounter(absl::string_view name,
                                         LabelSpan labels, uint64_t val) const {
  if (!HasBackends()) return;
  // TODO: Explore RCU optimization for lock-free reads.
  absl::ReaderMutexLock lock(mutex_);
  for (const std::unique_ptr<MetricsBackend>& backend : backends_) {
    backend->IncrementCounter(name, labels, val);
  }
}

void RaidenMetricStore::SetGauge(absl::string_view name, LabelSpan labels,
                                 double val) const {
  if (!HasBackends()) return;
  absl::ReaderMutexLock lock(mutex_);
  for (const std::unique_ptr<MetricsBackend>& backend : backends_) {
    backend->SetGauge(name, labels, val);
  }
}

void RaidenMetricStore::ObserveHistogram(absl::string_view name,
                                         LabelSpan labels, double val) const {
  if (!HasBackends()) return;
  absl::ReaderMutexLock lock(mutex_);
  for (const std::unique_ptr<MetricsBackend>& backend : backends_) {
    backend->ObserveHistogram(name, labels, val);
  }
}

std::string RaidenMetricStore::GetTextSnapshot() const {
  if (!HasBackends()) return "";
  absl::ReaderMutexLock lock(mutex_);
  std::string result;
  for (const std::unique_ptr<MetricsBackend>& backend : backends_) {
    // TODO: Consider adding a separator between backends.
    absl::StrAppend(&result, backend->GetTextSnapshot());
  }
  return result;
}

std::vector<MetricMetadata> RaidenMetricStore::GetMetricMetadata() const {
  if (!HasBackends()) return {};
  return {std::begin(metric_metadata::kAllMetrics),
          std::end(metric_metadata::kAllMetrics)};
}

std::map<std::string, std::vector<double>>
RaidenMetricStore::GetAndResetMetricSamples() {
  if (!HasBackends()) return {};
  absl::MutexLock lock(mutex_);
  std::map<std::string, std::vector<double>> result;
  for (const std::unique_ptr<MetricsBackend>& backend : backends_) {
    auto samples = backend->GetAndResetMetricSamples();
    for (auto& [name, vals] : samples) {
      auto& result_vals = result[name];
      result_vals.insert(result_vals.end(), vals.begin(), vals.end());
    }
  }
  return result;
}

}  // namespace tpu_raiden::telemetry

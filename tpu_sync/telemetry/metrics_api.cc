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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/buffered_metrics_exporter.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/prometheus_exporter.h"
#include "tpu_sync/telemetry/prometheus_shm_exporter.h"

namespace tpu_raiden::telemetry {

std::optional<std::string> ResolveEnvVar(const char* env_var) {
  if (env_var == nullptr) {
    return std::nullopt;
  }
  const char* value = std::getenv(env_var);
  if (value != nullptr && *value != '\0') {
    absl::string_view trimmed = absl::StripAsciiWhitespace(value);
    if (!trimmed.empty()) {
      return std::string(trimmed);
    }
  }
  return std::nullopt;
}

namespace {

int ResolveExporterPort() {
  if (std::optional<std::string> port_str =
          ResolveEnvVar(kPrometheusPortEnvVar)) {
    int parsed_port = 0;
    if (absl::SimpleAtoi(*port_str, &parsed_port) && parsed_port >= kMinPort &&
        parsed_port <= kMaxPort) {
      return parsed_port;
    }
    LOG(WARNING) << "Invalid port specified in " << kPrometheusPortEnvVar
                 << ": '" << *port_str << "'. Expected integer in range ["
                 << kMinPort << ", " << kMaxPort
                 << "]. Falling back to default port (" << kDefaultExporterPort
                 << ").";
  }
  return kDefaultExporterPort;
}

std::string ResolveEnvVar(const char* env_var,
                          absl::string_view default_value) {
  return telemetry::ResolveEnvVar(env_var).value_or(std::string(default_value));
}

std::optional<std::string> ResolveWorkloadName() {
  for (const char* env_var :
       {kWorkloadNameEnvVar, "JOBSET_NAME", "WORKLOAD_NAME", "M_RUN_NAME"}) {
    if (std::optional<std::string> val = telemetry::ResolveEnvVar(env_var)) {
      return val;
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
  cloud_logging_enabled_.store(false, std::memory_order_release);
}

bool RaidenMetricStore::HasBackends() const {
  return has_backends_.load(std::memory_order_acquire);
}

bool RaidenMetricStore::IsCloudLoggingEnabled() const {
  return cloud_logging_enabled_.load(std::memory_order_acquire);
}

absl::Status RaidenMetricStore::InitializeFromBackendNames(
    absl::Span<const absl::string_view> backend_names) {
  // Pre-validate all backend names first to guarantee all-or-nothing semantics:
  // if any backend name is unrecognized, existing backends remain unchanged.
  for (absl::string_view backend_key : backend_names) {
    std::string name =
        absl::AsciiStrToLower(absl::StripAsciiWhitespace(backend_key));
    if (name.empty()) {
      continue;
    }
    if (name == kPrometheus || name == kBuffered ||
        name == kCloudLogging
    ) {
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown telemetry backend: ", backend_key));
  }

  // Clear any existing backends so that previous global metric registrations
  // are unregistered before new ones are built.
  SetBackends({});

  std::vector<std::unique_ptr<MetricsBackend>> new_backends;
  absl::flat_hash_set<std::string> seen_backends;
  bool enable_cloud_logging = false;
  bool has_buffered_exporter = false;

  for (absl::string_view backend_key : backend_names) {
    std::string name =
        absl::AsciiStrToLower(absl::StripAsciiWhitespace(backend_key));
    if (name.empty() || !seen_backends.insert(name).second) {
      continue;
    }
    if (name == kPrometheus) {
      ExporterOptions exporter_options = {
          .bind_address =
              ResolveEnvVar(kPrometheusHostEnvVar, kDefaultExporterHost),
          .port = ResolveExporterPort(),
          .local_rank = ResolveEnvVar(kLocalRankEnvVar),
          .shm_dir = ResolveEnvVar(kTelemetryMultiprocDirEnvVar),
      };
      if (exporter_options.shm_dir) {
        if (!exporter_options.local_rank) {
          LOG_FIRST_N(INFO, 1) << kLocalRankEnvVar << " is not set while "
                               << kTelemetryMultiprocDirEnvVar
                               << " is set; PrometheusShmExporter will omit "
                                  "the local_rank label.";
        }
        new_backends.push_back(
            std::make_unique<PrometheusShmExporter>(
                std::move(exporter_options)));
      } else {
        new_backends.push_back(
            std::make_unique<PrometheusExporter>(std::move(exporter_options)));
      }
    } else if (name == kBuffered || name == kCloudLogging) {
      if (!has_buffered_exporter) {
        new_backends.push_back(std::make_unique<BufferedMetricsExporter>());
        has_buffered_exporter = true;
      }
      if (name == kCloudLogging) {
        enable_cloud_logging = true;
      }
    }
  }

  SetBackends(std::move(new_backends));
  cloud_logging_enabled_.store(enable_cloud_logging, std::memory_order_release);
  return absl::OkStatus();
}

absl::Status RaidenMetricStore::InitializeFromEnvironment() {
  if (HasBackends()) {
    return absl::OkStatus();
  }
  const char* env = std::getenv(kTelemetryBackendsEnvVar);
  if (env == nullptr) {
    return InitializeFromBackendNames({kBuffered});
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

std::string RaidenMetricStore::FlushToCloudLogging(absl::string_view phase,
                                                   uint64_t uuid,
                                                   absl::string_view req_id) {
  if (!IsCloudLoggingEnabled()) {
    return "";
  }
  auto samples = GetAndResetMetricSamples();
  std::string metrics_json;
  for (const auto& [key, vals] : samples) {
    size_t finite_count = 0;
    double sum = 0.0;
    double min_v = 0.0;
    double max_v = 0.0;
    double latest = 0.0;
    for (double v : vals) {
      if (!std::isfinite(v)) {
        continue;
      }
      if (finite_count == 0) {
        min_v = v;
        max_v = v;
      } else {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
      }
      sum += v;
      latest = v;
      ++finite_count;
    }
    if (finite_count == 0) {
      continue;
    }
    if (!metrics_json.empty()) {
      absl::StrAppend(&metrics_json, ",");
    }
    absl::StrAppend(&metrics_json, "\"", absl::CEscape(key),
                    "\":{\"count\":", finite_count, ",\"sum\":", sum,
                    ",\"min\":", min_v, ",\"max\":", max_v,
                    ",\"latest\":", latest, "}");
  }
  if (metrics_json.empty()) {
    return "";
  }
  std::string workload_field;
  if (std::optional<std::string> workload = ResolveWorkloadName()) {
    workload_field =
        absl::StrCat(",\"workload\":\"", absl::CEscape(*workload), "\"");
  }
  std::string req_id_field;
  if (!req_id.empty()) {
    req_id_field = absl::StrCat(",\"req_id\":\"", absl::CEscape(req_id), "\"");
  }
  std::string line = absl::StrCat(
      "{\"severity\":\"INFO\",\"component\":\"tpu_raiden.weight_synchronizer\"",
      workload_field, ",\"phase\":\"", absl::CEscape(phase),
      "\",\"uuid\":", uuid, req_id_field, ",\"raiden_telemetry\":{",
      metrics_json, "}}\n");
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fflush(stdout);
  return line;
}

}  // namespace tpu_raiden::telemetry

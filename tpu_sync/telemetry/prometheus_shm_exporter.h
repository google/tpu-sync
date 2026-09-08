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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_SHM_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_SHM_EXPORTER_H_

#include <memory>
#include <string>

#include "tpu_sync/telemetry/base_shm_exporter.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

// PrometheusShmExporter exposes shared memory metrics collected across
// multi-worker processes via an HTTP /metrics endpoint for Prometheus scrapers.
//
// Inherits from BaseShmExporter to aggregate counters, gauges, and histograms
// from shared memory segments. The first process to bind the designated port
// runs the HTTP exposer, while secondary workers gracefully disable their HTTP
// server and continue writing/aggregating metrics via shared memory. All
// instances can generate text snapshots independently.
//
// Operational note: In this decentralized port-contention model, the HTTP
// exposer is bound during construction. If the primary worker serving the HTTP
// endpoint terminates, surviving workers continue aggregating and serving
// metrics via shared memory and GetTextSnapshot(). A newly initialized exporter
// or worker can bind the port once the primary terminates.
//
// Thread safety: Thread-safe for concurrent calls to public methods.
class PrometheusShmExporter final : public BaseShmExporter {
 public:
  explicit PrometheusShmExporter(ExporterOptions exporter_options);
  ~PrometheusShmExporter() override;
  PrometheusShmExporter(const PrometheusShmExporter&) = delete;
  PrometheusShmExporter& operator=(const PrometheusShmExporter&) = delete;
  PrometheusShmExporter(PrometheusShmExporter&&) = delete;
  PrometheusShmExporter& operator=(PrometheusShmExporter&&) = delete;

  // Collects and serializes all shared memory metrics into Prometheus text
  // exposition format (version 0.0.4).
  std::string GetTextSnapshot() const override;

  // Returns true if the HTTP server is actively running and listening for
  // scrape requests. For testing purposes only.
  bool IsServerRunningForTesting() const;

 private:
  struct ExposerState;
  std::unique_ptr<ExposerState> exposer_state_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_SHM_EXPORTER_H_

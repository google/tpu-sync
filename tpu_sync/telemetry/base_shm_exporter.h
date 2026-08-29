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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {

// Abstract base class for multi-process shared-memory telemetry exporters.
// Manages underlying ShmWriter for low-overhead metric publishing and
// ShmCollector for multi-worker aggregation.
class BaseShmExporter : public MetricsBackend {
 public:
  explicit BaseShmExporter(const ExporterOptions& options = {});
  ~BaseShmExporter() override;

  BaseShmExporter(const BaseShmExporter&) = delete;
  BaseShmExporter& operator=(const BaseShmExporter&) = delete;
  BaseShmExporter(BaseShmExporter&&) = delete;
  BaseShmExporter& operator=(BaseShmExporter&&) = delete;

  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val) const override;
  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override;
  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override;

  std::string GetTextSnapshot() const override;
  void CollectMetrics(absl::flat_hash_map<std::string, double>& totals) const;

  virtual void Start() { is_running_.store(true, std::memory_order_release); }
  virtual void Stop() { is_running_.store(false, std::memory_order_release); }
  virtual bool IsRunning() const {
    return is_running_.load(std::memory_order_acquire);
  }

  const ExporterOptions& GetOptions() const { return options_; }
  const ShmWriter* GetWriter() const { return shm_writer_.get(); }
  const ShmCollector* GetCollector() const { return collector_.get(); }

 protected:
  ExporterOptions options_;
  std::unique_ptr<ShmWriter> shm_writer_;
  std::unique_ptr<ShmCollector> collector_;
  std::atomic<bool> is_running_{false};
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BASE_SHM_EXPORTER_H_

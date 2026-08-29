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
#include <vector>

#include "prometheus/collectable.h"
#include "prometheus/exposer.h"
#include "prometheus/metric_family.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/telemetry/base_shm_exporter.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

class PrometheusShmExporter : public BaseShmExporter {
 public:
  explicit PrometheusShmExporter(const ExporterOptions& options = {});
  ~PrometheusShmExporter() override;
  PrometheusShmExporter(const PrometheusShmExporter&) = delete;
  PrometheusShmExporter& operator=(const PrometheusShmExporter&) = delete;

  void Start() override;
  void Stop() override;

  std::string GetTextSnapshot() const override;
  bool IsExposerRunning() const;
  std::vector<prometheus::MetricFamily> CollectMetricFamilies() const;

 private:
  mutable absl::Mutex mutex_;
  std::unique_ptr<prometheus::Exposer> exposer_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<prometheus::Collectable> collectable_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_SHM_EXPORTER_H_

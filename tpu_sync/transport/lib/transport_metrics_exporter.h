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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TRANSPORT_METRICS_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TRANSPORT_METRICS_EXPORTER_H_

#include "peregrine/src/api/transport_metrics.h"

namespace tpu_raiden::transport::lib {

// Stateful delta calculator that exports cumulative peregrine::TransportMetrics
// snapshots to RaidenMetricStore::GetGlobalMetricStore().
// Holds no pointers or threads.
class TransportMetricsExporter final {
 public:
  TransportMetricsExporter() = default;

  // Computes deltas between `curr` and `prev_metrics_`, exports deltas to
  // RaidenMetricStore::GetGlobalMetricStore(), and updates `prev_metrics_`.
  void Export(const ::peregrine::TransportMetrics& curr);

 private:
  ::peregrine::TransportMetrics prev_metrics_{};
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TRANSPORT_METRICS_EXPORTER_H_

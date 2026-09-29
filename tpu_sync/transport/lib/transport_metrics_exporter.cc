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

#include "tpu_sync/transport/lib/transport_metrics_exporter.h"

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "peregrine/src/api/transport_metrics.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::transport::lib {
namespace {

namespace metric_labels = ::tpu_raiden::telemetry::metric_labels;
namespace metric_names = ::tpu_raiden::telemetry::metric_names;
using ::tpu_raiden::telemetry::LabelSpan;
using ::tpu_raiden::telemetry::MetricLabel;
using ::tpu_raiden::telemetry::RaidenMetricStore;

constexpr MetricLabel kWriteLabels[] = {
    {.key = metric_labels::kDirection, .value = metric_labels::kDirectionWrite},
};
constexpr MetricLabel kReadLabels[] = {
    {.key = metric_labels::kDirection, .value = metric_labels::kDirectionRead},
};

void ExportHistogramDelta(RaidenMetricStore& store, absl::string_view name,
                          LabelSpan labels,
                          const ::peregrine::Log2Histogram<32>& curr_hist,
                          const ::peregrine::Log2Histogram<32>& prev_hist) {
  for (size_t i = 0; i < curr_hist.buckets.size(); ++i) {
    const uint64_t delta = curr_hist.buckets[i] - prev_hist.buckets[i];
    // Log2Histogram bucket 0 covers [0, 1) (lower bound 0.0), and bucket i
    // (i >= 1) covers [2^(i-1), 2^i) (lower bound 2^(i-1)).
    double bucket_val = 0.0;
    if (i > 0) {
      bucket_val = 1ULL << (i - 1);
    }
    for (uint64_t s = 0; s < delta; ++s) {
      store.ObserveHistogram(name, labels, bucket_val);
    }
  }
}

void ExportOpMetricsDelta(RaidenMetricStore& store,
                          const ::peregrine::OpMetrics& curr_op,
                          const ::peregrine::OpMetrics& prev_op,
                          LabelSpan labels) {
  ExportHistogramDelta(store, metric_names::kPeregrineE2eLatencyUs, labels,
                       curr_op.e2e_latency_us, prev_op.e2e_latency_us);
  ExportHistogramDelta(store, metric_names::kPeregrineRequestSizeBytes, labels,
                       curr_op.request_size_bytes, prev_op.request_size_bytes);
  store.IncrementCounter(metric_names::kPeregrineBytesTotal, labels,
                         curr_op.bytes - prev_op.bytes);
  store.IncrementCounter(metric_names::kPeregrineErrorsTotal, labels,
                         curr_op.errors - prev_op.errors);
}

}  // namespace

void TransportMetricsExporter::Export(
    const ::peregrine::TransportMetrics& curr) {
  RaidenMetricStore& store = RaidenMetricStore::GetGlobalMetricStore();
  if (!store.HasBackends()) {
    prev_metrics_ = curr;
    return;
  }

  ExportOpMetricsDelta(store, curr.write, prev_metrics_.write, kWriteLabels);
  ExportOpMetricsDelta(store, curr.read, prev_metrics_.read, kReadLabels);

  prev_metrics_ = curr;
}

}  // namespace tpu_raiden::transport::lib

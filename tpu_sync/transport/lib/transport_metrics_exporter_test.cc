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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "peregrine/src/api/transport_metrics.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"

namespace tpu_raiden::transport::lib {
namespace {

namespace metric_labels = ::tpu_raiden::telemetry::metric_labels;
namespace metric_names = ::tpu_raiden::telemetry::metric_names;
using ::testing::_;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::tpu_raiden::telemetry::MetricLabel;
using ::tpu_raiden::telemetry::MetricsBackend;
using ::tpu_raiden::telemetry::MockMetricsBackend;
using ::tpu_raiden::telemetry::RaidenMetricStore;

class TransportMetricsExporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  }

  void TearDown() override {
    RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  }

  MockMetricsBackend* RegisterMockBackend() {
    auto mock_backend = std::make_unique<MockMetricsBackend>();
    MockMetricsBackend* raw_mock = mock_backend.get();
    std::vector<std::unique_ptr<MetricsBackend>> backends;
    backends.push_back(std::move(mock_backend));
    RaidenMetricStore::GetGlobalMetricStore().SetBackends(std::move(backends));
    return raw_mock;
  }
};

TEST_F(TransportMetricsExporterTest,
       FastPathExitWhenNoBackendsStillUpdatesBaseline) {
  TransportMetricsExporter exporter;

  ::peregrine::TransportMetrics m1{};
  m1.write.bytes = 1024;
  m1.write.errors = 2;
  m1.write.e2e_latency_us.buckets[0] = 3;

  // Export with no backends registered; should update prev_metrics_ to m1.
  exporter.Export(m1);

  MockMetricsBackend* mock = RegisterMockBackend();

  ::peregrine::TransportMetrics m2 = m1;
  m2.write.bytes = 1524;  // delta = 500

  const MetricLabel write_label{.key = metric_labels::kDirection,
                                .value = metric_labels::kDirectionWrite};
  const MetricLabel read_label{.key = metric_labels::kDirection,
                               .value = metric_labels::kDirectionRead};
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineBytesTotal),
                                      ElementsAre(write_label), 500))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineErrorsTotal),
                                      ElementsAre(write_label), 0))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineBytesTotal),
                                      ElementsAre(read_label), 0))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineErrorsTotal),
                                      ElementsAre(read_label), 0))
      .Times(1);
  EXPECT_CALL(*mock, ObserveHistogram(_, _, _)).Times(0);

  exporter.Export(m2);
}

TEST_F(TransportMetricsExporterTest,
       ExportsWriteAndReadCounterAndHistogramDeltas) {
  MockMetricsBackend* mock = RegisterMockBackend();
  TransportMetricsExporter exporter;

  const MetricLabel write_label{.key = metric_labels::kDirection,
                                .value = metric_labels::kDirectionWrite};
  const MetricLabel read_label{.key = metric_labels::kDirection,
                               .value = metric_labels::kDirectionRead};

  ::peregrine::TransportMetrics m1{};
  m1.write.bytes = 4096;
  m1.write.errors = 1;
  m1.write.e2e_latency_us.buckets[0] = 2;      // bucket 0 -> 0.0
  m1.write.e2e_latency_us.buckets[4] = 1;      // bucket 4 -> 2^(4-1) = 8.0
  m1.write.request_size_bytes.buckets[1] = 1;  // bucket 1 -> 2^(1-1) = 1.0
  m1.read.bytes = 8192;
  m1.read.errors = 3;
  m1.read.e2e_latency_us.buckets[10] = 2;      // bucket 10 -> 2^9 = 512.0
  m1.read.request_size_bytes.buckets[31] = 1;  // bucket 31 -> 2^30

  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineBytesTotal),
                                      ElementsAre(write_label), 4096))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineErrorsTotal),
                                      ElementsAre(write_label), 1))
      .Times(1);
  EXPECT_CALL(*mock, ObserveHistogram(Eq(metric_names::kPeregrineE2eLatencyUs),
                                      ElementsAre(write_label), DoubleEq(0.0)))
      .Times(2);
  EXPECT_CALL(*mock, ObserveHistogram(Eq(metric_names::kPeregrineE2eLatencyUs),
                                      ElementsAre(write_label), DoubleEq(8.0)))
      .Times(1);
  EXPECT_CALL(*mock,
              ObserveHistogram(Eq(metric_names::kPeregrineRequestSizeBytes),
                               ElementsAre(write_label), DoubleEq(1.0)))
      .Times(1);

  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineBytesTotal),
                                      ElementsAre(read_label), 8192))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineErrorsTotal),
                                      ElementsAre(read_label), 3))
      .Times(1);
  EXPECT_CALL(*mock, ObserveHistogram(Eq(metric_names::kPeregrineE2eLatencyUs),
                                      ElementsAre(read_label), DoubleEq(512.0)))
      .Times(2);
  EXPECT_CALL(
      *mock, ObserveHistogram(Eq(metric_names::kPeregrineRequestSizeBytes),
                              ElementsAre(read_label),
                              DoubleEq(static_cast<double>(uint64_t{1} << 30))))
      .Times(1);

  exporter.Export(m1);
  ::testing::Mock::VerifyAndClearExpectations(mock);

  // Second Export with identical snapshot emits 0 counter deltas and 0
  // histogram observations.
  EXPECT_CALL(*mock, IncrementCounter(_, _, 0)).Times(4);
  EXPECT_CALL(*mock, ObserveHistogram(_, _, _)).Times(0);
  exporter.Export(m1);
  ::testing::Mock::VerifyAndClearExpectations(mock);

  // Third Export with incremental changes.
  ::peregrine::TransportMetrics m2 = m1;
  m2.write.bytes += 1024;
  m2.write.e2e_latency_us.buckets[4] += 3;

  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineBytesTotal),
                                      ElementsAre(write_label), 1024))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineErrorsTotal),
                                      ElementsAre(write_label), 0))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineBytesTotal),
                                      ElementsAre(read_label), 0))
      .Times(1);
  EXPECT_CALL(*mock, IncrementCounter(Eq(metric_names::kPeregrineErrorsTotal),
                                      ElementsAre(read_label), 0))
      .Times(1);
  EXPECT_CALL(*mock, ObserveHistogram(Eq(metric_names::kPeregrineE2eLatencyUs),
                                      ElementsAre(write_label), DoubleEq(8.0)))
      .Times(3);

  exporter.Export(m2);
}

}  // namespace
}  // namespace tpu_raiden::transport::lib

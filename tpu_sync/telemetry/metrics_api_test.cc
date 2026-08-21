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

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/test_util.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

class ScopedEnvironmentVariable {
 public:
  ScopedEnvironmentVariable(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) {
      old_value_ = old;
      has_old_value_ = true;
    }
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  ~ScopedEnvironmentVariable() {
    if (has_old_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) =
      delete;

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_ = false;
};

static_assert(!std::is_copy_constructible_v<MetricsBackend>,
              "MetricsBackend must not be copy constructible");
static_assert(!std::is_copy_assignable_v<MetricsBackend>,
              "MetricsBackend must not be copy assignable");
static_assert(!std::is_move_constructible_v<MetricsBackend>,
              "MetricsBackend must not be move constructible");
static_assert(!std::is_move_assignable_v<MetricsBackend>,
              "MetricsBackend must not be move assignable");

class MetricsApiTest : public testing::Test {
 protected:
  void SetUp() override { store_.SetBackends({}); }

  void TearDown() override { store_.SetBackends({}); }

  RaidenMetricStore store_;
};

TEST_F(MetricsApiTest, GlobalMetricStoreSingleton) {
  RaidenMetricStore& global1 = RaidenMetricStore::GetGlobalMetricStore();
  RaidenMetricStore& global2 = RaidenMetricStore::GetGlobalMetricStore();
  EXPECT_EQ(&global1, &global2);
}

TEST_F(MetricsApiTest, MetricMetadataConstants) {
  EXPECT_EQ(kPrometheus, "prometheus");

  // SentBytesTotal
  EXPECT_EQ(metric_names::kSentBytesTotal, "sent_bytes_total");
  EXPECT_EQ(metric_descriptions::kSentBytesTotal,
            "Total count of bytes sent over TPU Raiden interfaces.");
  EXPECT_EQ(metric_metadata::kSentBytesTotal.name, "sent_bytes_total");
  EXPECT_EQ(metric_metadata::kSentBytesTotal.description,
            "Total count of bytes sent over TPU Raiden interfaces.");
  EXPECT_EQ(metric_metadata::kSentBytesTotal.type, MetricType::kCounter);

  // ReceivedBytesTotal
  EXPECT_EQ(metric_names::kReceivedBytesTotal, "received_bytes_total");
  EXPECT_EQ(metric_descriptions::kReceivedBytesTotal,
            "Total count of bytes received over TPU Raiden interfaces.");
  EXPECT_EQ(metric_metadata::kReceivedBytesTotal.name, "received_bytes_total");
  EXPECT_EQ(metric_metadata::kReceivedBytesTotal.description,
            "Total count of bytes received over TPU Raiden interfaces.");
  EXPECT_EQ(metric_metadata::kReceivedBytesTotal.type, MetricType::kCounter);

  // TransferFailuresTotal
  EXPECT_EQ(metric_names::kTransferFailuresTotal, "transfer_failures_total");
  EXPECT_EQ(
      metric_descriptions::kTransferFailuresTotal,
      "Cumulative total count of transfer failures across all interfaces.");
  EXPECT_EQ(metric_metadata::kTransferFailuresTotal.name,
            "transfer_failures_total");
  EXPECT_EQ(
      metric_metadata::kTransferFailuresTotal.description,
      "Cumulative total count of transfer failures across all interfaces.");
  // H2dTransferTimeMs
  EXPECT_EQ(metric_names::kH2dTransferTimeMs, "h2d_transfer_time_ms");
  EXPECT_EQ(metric_descriptions::kH2dTransferTimeMs,
            "Host-to-Device transfer latency in milliseconds.");
  EXPECT_EQ(metric_metadata::kH2dTransferTimeMs.name, "h2d_transfer_time_ms");
  EXPECT_EQ(metric_metadata::kH2dTransferTimeMs.description,
            "Host-to-Device transfer latency in milliseconds.");
  EXPECT_EQ(metric_metadata::kH2dTransferTimeMs.type, MetricType::kHistogram);

  // D2hTransferTimeMs
  EXPECT_EQ(metric_names::kD2hTransferTimeMs, "d2h_transfer_time_ms");
  EXPECT_EQ(metric_descriptions::kD2hTransferTimeMs,
            "Device-to-Host transfer latency in milliseconds.");
  EXPECT_EQ(metric_metadata::kD2hTransferTimeMs.name, "d2h_transfer_time_ms");
  EXPECT_EQ(metric_metadata::kD2hTransferTimeMs.description,
            "Device-to-Host transfer latency in milliseconds.");
  EXPECT_EQ(metric_metadata::kD2hTransferTimeMs.type, MetricType::kHistogram);

  // TransferDurationMs
  EXPECT_EQ(metric_names::kTransferDurationMs, "transfer_duration_ms");
  EXPECT_EQ(
      metric_descriptions::kTransferDurationMs,
      "Measures End-to-End (E2E) latency bound around the entire request in "
      "milliseconds, including setup delays.");
  EXPECT_EQ(metric_metadata::kTransferDurationMs.name, "transfer_duration_ms");
  EXPECT_EQ(
      metric_metadata::kTransferDurationMs.description,
      "Measures End-to-End (E2E) latency bound around the entire request in "
      "milliseconds, including setup delays.");
  EXPECT_EQ(metric_metadata::kTransferDurationMs.type, MetricType::kHistogram);

  // BufferAllocatedBytes
  EXPECT_EQ(metric_names::kBufferAllocatedBytes, "buffer_allocated_bytes");
  EXPECT_EQ(metric_descriptions::kBufferAllocatedBytes,
            "Current host DRAM buffer capacity allocated in bytes for KV cache "
            "staging "
            "across all layers and shards.");
  EXPECT_EQ(metric_metadata::kBufferAllocatedBytes.name,
            "buffer_allocated_bytes");
  EXPECT_EQ(metric_metadata::kBufferAllocatedBytes.description,
            "Current host DRAM buffer capacity allocated in bytes for KV cache "
            "staging "
            "across all layers and shards.");
  EXPECT_EQ(metric_metadata::kBufferAllocatedBytes.type, MetricType::kGauge);

  // Direction Labels
  EXPECT_EQ(metric_labels::kDirection, "direction");
  EXPECT_EQ(metric_labels::kDirectionPush, "push");
  EXPECT_EQ(metric_labels::kDirectionPull, "pull");
  EXPECT_EQ(metric_labels::kDirectionPullResponse, "pull_response");

  // Error Code Labels
  EXPECT_EQ(metric_labels::kErrorCode, "error_code");

  // All Metrics
  EXPECT_THAT(metric_metadata::kAllMetrics,
              ElementsAre(metric_metadata::kSentBytesTotal,
                          metric_metadata::kReceivedBytesTotal,
                          metric_metadata::kTransferFailuresTotal,
                          metric_metadata::kTransferDurationMs,
                          metric_metadata::kH2dTransferTimeMs,
                          metric_metadata::kD2hTransferTimeMs,
                          metric_metadata::kBufferAllocatedBytes));
}

TEST_F(MetricsApiTest, FastPathExitWhenNoBackends) {
  EXPECT_FALSE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 1024);
  store_.IncrementCounter(metric_names::kReceivedBytesTotal, {}, 1024);
  store_.IncrementCounter(metric_names::kTransferFailuresTotal, {}, 1);
  EXPECT_EQ(store_.GetTextSnapshot(), "");
  EXPECT_TRUE(store_.GetMetricMetadata().empty());
  EXPECT_TRUE(store_.GetAndResetMetricSamples().empty());
}

TEST_F(MetricsApiTest, DispatchesToRegisteredBackend) {
  auto mock_backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_mock = mock_backend.get();

  EXPECT_CALL(*raw_mock,
              IncrementCounter(Eq(metric_names::kSentBytesTotal), _, 2048))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              IncrementCounter(Eq(metric_names::kReceivedBytesTotal), _, 4096))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              IncrementCounter(Eq(metric_names::kTransferFailuresTotal), _, 1))
      .Times(1);
  EXPECT_CALL(*raw_mock, GetTextSnapshot()).WillOnce(Return("# HELP mock\n"));
  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(mock_backend));
  store_.SetBackends(std::move(backends));
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 2048);
  store_.IncrementCounter(metric_names::kReceivedBytesTotal, {}, 4096);
  store_.IncrementCounter(metric_names::kTransferFailuresTotal, {}, 1);
  EXPECT_EQ(store_.GetTextSnapshot(), "# HELP mock\n");
}

TEST_F(MetricsApiTest, ObserveHistogramPrecision) {
  auto mock_backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_mock = mock_backend.get();

  // Verify that fractional values (microseconds precision in ms metric) are
  // passed correctly.
  static constexpr double kFractionalValue = 0.001234;  // 1.234 microseconds

  EXPECT_CALL(*raw_mock, ObserveHistogram(Eq(metric_names::kH2dTransferTimeMs),
                                          _, Eq(kFractionalValue)))
      .Times(1);
  EXPECT_CALL(*raw_mock, ObserveHistogram(Eq(metric_names::kD2hTransferTimeMs),
                                          _, Eq(kFractionalValue)))
      .Times(1);

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(mock_backend));
  store_.SetBackends(std::move(backends));

  store_.ObserveHistogram(metric_names::kH2dTransferTimeMs, {},
                          kFractionalValue);
  store_.ObserveHistogram(metric_names::kD2hTransferTimeMs, {},
                          kFractionalValue);
}

TEST_F(MetricsApiTest, SetBackendsReplacesExistingBackends) {
  auto backend1 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend1 = backend1.get();
  std::vector<std::unique_ptr<MetricsBackend>> initial_backends;
  initial_backends.push_back(std::move(backend1));
  store_.SetBackends(std::move(initial_backends));

  auto backend2 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend2 = backend2.get();

  EXPECT_CALL(*raw_backend1, IncrementCounter(_, _, _)).Times(0);
  EXPECT_CALL(*raw_backend2,
              IncrementCounter(Eq("tpu_raiden_sent_bytes_total"), _, 4096))
      .Times(1);

  std::vector<std::unique_ptr<MetricsBackend>> new_backends;
  new_backends.push_back(std::move(backend2));
  store_.SetBackends(std::move(new_backends));

  EXPECT_TRUE(store_.HasBackends());
  store_.IncrementCounter("tpu_raiden_sent_bytes_total", {}, 4096);
}

TEST_F(MetricsApiTest, SetBackendsEmptyClearsBackends) {
  auto backend = std::make_unique<MockMetricsBackend>();
  std::vector<std::unique_ptr<MetricsBackend>> initial_backends;
  initial_backends.push_back(std::move(backend));
  store_.SetBackends(std::move(initial_backends));
  EXPECT_TRUE(store_.HasBackends());

  store_.SetBackends({});
  EXPECT_FALSE(store_.HasBackends());

  store_.IncrementCounter("tpu_raiden_sent_bytes_total", {}, 1024);
  EXPECT_EQ(store_.GetTextSnapshot(), "");
  EXPECT_TRUE(store_.GetMetricMetadata().empty());
  EXPECT_TRUE(store_.GetAndResetMetricSamples().empty());
}

TEST_F(MetricsApiTest, SetBackendsMultipleBackends) {
  auto backend1 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend1 = backend1.get();
  auto backend2 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend2 = backend2.get();

  EXPECT_CALL(*raw_backend1, IncrementCounter(Eq("counter"), _, 100)).Times(1);
  EXPECT_CALL(*raw_backend2, IncrementCounter(Eq("counter"), _, 100)).Times(1);
  EXPECT_CALL(*raw_backend1, GetTextSnapshot()).WillOnce(Return("snap1\n"));
  EXPECT_CALL(*raw_backend2, GetTextSnapshot()).WillOnce(Return("snap2\n"));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend1));
  backends.push_back(std::move(backend2));
  store_.SetBackends(std::move(backends));

  EXPECT_TRUE(store_.HasBackends());
  store_.IncrementCounter("counter", {}, 100);
  EXPECT_EQ(store_.GetTextSnapshot(), "snap1\nsnap2\n");
}

TEST_F(MetricsApiTest, GetMetricMetadataReturnsAllMetricsWhenBackendsActive) {
  auto backend = std::make_unique<MockMetricsBackend>();
  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend));
  store_.SetBackends(std::move(backends));

  std::vector<MetricMetadata> result = store_.GetMetricMetadata();
  EXPECT_THAT(result, ElementsAre(metric_metadata::kSentBytesTotal,
                                  metric_metadata::kReceivedBytesTotal,
                                  metric_metadata::kTransferFailuresTotal,
                                  metric_metadata::kTransferDurationMs,
                                  metric_metadata::kH2dTransferTimeMs,
                                  metric_metadata::kD2hTransferTimeMs,
                                  metric_metadata::kBufferAllocatedBytes));
}

TEST_F(MetricsApiTest, GetMetricMetadataEmptyWhenNoBackends) {
  EXPECT_FALSE(store_.HasBackends());
  EXPECT_TRUE(store_.GetMetricMetadata().empty());

  auto backend = std::make_unique<MockMetricsBackend>();
  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend));
  store_.SetBackends(std::move(backends));
  EXPECT_FALSE(store_.GetMetricMetadata().empty());

  store_.SetBackends({});
  EXPECT_TRUE(store_.GetMetricMetadata().empty());
}

TEST_F(MetricsApiTest, GetAndResetMetricSamplesSingleBackend) {
  auto mock_backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_mock = mock_backend.get();

  std::map<std::string, std::vector<double>> expected_samples = {
      {"sent_bytes_total", {1024.0, 2048.0}},
      {"transfer_failures_total", {1.0}},
  };
  EXPECT_CALL(*raw_mock, GetAndResetMetricSamples())
      .WillOnce(Return(expected_samples));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(mock_backend));
  store_.SetBackends(std::move(backends));

  EXPECT_EQ(store_.GetAndResetMetricSamples(), expected_samples);
}

TEST_F(MetricsApiTest, GetAndResetMetricSamplesMultipleBackendsMerging) {
  auto backend1 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend1 = backend1.get();
  auto backend2 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend2 = backend2.get();

  std::map<std::string, std::vector<double>> samples1 = {
      {"sent_bytes_total", {100.0, 200.0}},
      {"unique_to_backend1", {1.0}},
  };
  std::map<std::string, std::vector<double>> samples2 = {
      {"sent_bytes_total", {300.0}},
      {"unique_to_backend2", {2.0, 3.0}},
  };

  EXPECT_CALL(*raw_backend1, GetAndResetMetricSamples())
      .WillOnce(Return(samples1));
  EXPECT_CALL(*raw_backend2, GetAndResetMetricSamples())
      .WillOnce(Return(samples2));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend1));
  backends.push_back(std::move(backend2));
  store_.SetBackends(std::move(backends));

  std::map<std::string, std::vector<double>> result =
      store_.GetAndResetMetricSamples();

  std::map<std::string, std::vector<double>> expected = {
      {"sent_bytes_total", {100.0, 200.0, 300.0}},
      {"unique_to_backend1", {1.0}},
      {"unique_to_backend2", {2.0, 3.0}},
  };
  EXPECT_EQ(result, expected);
}

TEST_F(MetricsApiTest, GetAndResetMetricSamplesConsecutiveCalls) {
  auto mock_backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_mock = mock_backend.get();

  std::map<std::string, std::vector<double>> first_samples = {
      {"sent_bytes_total", {100.0}},
  };
  std::map<std::string, std::vector<double>> empty_samples = {};

  EXPECT_CALL(*raw_mock, GetAndResetMetricSamples())
      .WillOnce(Return(first_samples))
      .WillOnce(Return(empty_samples));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(mock_backend));
  store_.SetBackends(std::move(backends));

  EXPECT_EQ(store_.GetAndResetMetricSamples(), first_samples);
  EXPECT_TRUE(store_.GetAndResetMetricSamples().empty());
}

TEST_F(MetricsApiTest, GetAndResetMetricSamplesEmptySampleVectors) {
  auto mock_backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_mock = mock_backend.get();

  std::map<std::string, std::vector<double>> empty_entry = {
      {"empty_metric", {}},
  };
  EXPECT_CALL(*raw_mock, GetAndResetMetricSamples())
      .WillOnce(Return(empty_entry));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(mock_backend));
  store_.SetBackends(std::move(backends));

  std::map<std::string, std::vector<double>> result =
      store_.GetAndResetMetricSamples();
  EXPECT_EQ(result, empty_entry);
}

TEST_F(MetricsApiTest, SetBackendsFiltersNullptrs) {
  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(nullptr);
  store_.SetBackends(std::move(backends));

  EXPECT_FALSE(store_.HasBackends());
  EXPECT_TRUE(store_.GetMetricMetadata().empty());
  EXPECT_TRUE(store_.GetAndResetMetricSamples().empty());
}

class MinimalMetricsBackend : public MetricsBackend {
 public:
  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val) const override {}
  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override {}
  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override {}
  std::string GetTextSnapshot() const override { return ""; }
};

TEST_F(MetricsApiTest, MetricsBackendDefaultVirtualMethods) {
  MinimalMetricsBackend backend;
  EXPECT_TRUE(backend.GetAndResetMetricSamples().empty());
}

TEST_F(MetricsApiTest, ConcurrentTelemetryEmissions) {
  auto backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend = backend.get();

  static constexpr int kNumThreads = 8;
  static constexpr int kIterations = 100;
  static constexpr int kTotalCalls = kNumThreads * kIterations;

  EXPECT_CALL(*raw_backend, IncrementCounter(Eq("counter"), _, 1))
      .Times(kTotalCalls);
  EXPECT_CALL(*raw_backend, SetGauge(Eq("gauge"), _, 42)).Times(kTotalCalls);
  EXPECT_CALL(*raw_backend, ObserveHistogram(Eq("histogram"), _, 3.14))
      .Times(kTotalCalls);
  EXPECT_CALL(*raw_backend, GetTextSnapshot())
      .Times(kTotalCalls)
      .WillRepeatedly(Return("snapshot\n"));
  EXPECT_CALL(*raw_backend, GetAndResetMetricSamples())
      .Times(kTotalCalls)
      .WillRepeatedly(Return(std::map<std::string, std::vector<double>>{}));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend));
  store_.SetBackends(std::move(backends));

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this] {
      for (int j = 0; j < kIterations; ++j) {
        store_.IncrementCounter("counter", {}, 1);
        store_.SetGauge("gauge", {}, 42);
        store_.ObserveHistogram("histogram", {}, 3.14);
        (void)store_.GetTextSnapshot();
        (void)store_.GetMetricMetadata();
        (void)store_.GetAndResetMetricSamples();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(MetricsApiTest, ConstMetricsBackendReference) {
  MockMetricsBackend backend;
  const MetricsBackend& const_ref = backend;

  EXPECT_CALL(backend, IncrementCounter(Eq("counter"), _, 5)).Times(1);
  EXPECT_CALL(backend, SetGauge(Eq("gauge"), _, 10)).Times(1);
  EXPECT_CALL(backend, ObserveHistogram(Eq("histogram"), _, 1.23)).Times(1);
  EXPECT_CALL(backend, GetTextSnapshot()).WillOnce(Return("snapshot\n"));

  const_ref.IncrementCounter("counter", {}, 5);
  const_ref.SetGauge("gauge", {}, 10);
  const_ref.ObserveHistogram("histogram", {}, 1.23);
  EXPECT_EQ(const_ref.GetTextSnapshot(), "snapshot\n");
}

TEST_F(MetricsApiTest, InitializeFromBackendNamesPrometheus) {
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);
  EXPECT_THAT(store_.GetTextSnapshot(),
              HasSubstr("tpu_raiden_sent_bytes_total 10"));
  EXPECT_THAT(store_.GetAndResetMetricSamples(), IsEmpty());
}

TEST_F(MetricsApiTest, InitializeFromBackendNamesBuffered) {
  EXPECT_THAT(store_.InitializeFromBackendNames({"buffered"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);
  EXPECT_THAT(store_.GetTextSnapshot(), IsEmpty());
  EXPECT_THAT(store_.GetAndResetMetricSamples(),
              UnorderedElementsAre(
                  Pair("tpu_raiden_sent_bytes_total", ElementsAre(10.0))));
}

TEST_F(MetricsApiTest, InitializeFromBackendNamesMultipleBackends) {
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus", "buffered"}),
              IsOk());
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);
  EXPECT_THAT(store_.GetTextSnapshot(),
              HasSubstr("tpu_raiden_sent_bytes_total 10"));
  EXPECT_THAT(store_.GetAndResetMetricSamples(),
              UnorderedElementsAre(
                  Pair("tpu_raiden_sent_bytes_total", ElementsAre(10.0))));
}

TEST_F(MetricsApiTest, InitializeFromBackendNamesCaseInsensitiveAndWhitespace) {
  EXPECT_THAT(store_.InitializeFromBackendNames({"   pRoMeThEuS   "}), IsOk());
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);
  EXPECT_THAT(store_.GetTextSnapshot(),
              HasSubstr("tpu_raiden_sent_bytes_total 10"));
  EXPECT_THAT(store_.GetAndResetMetricSamples(), IsEmpty());
}

TEST_F(MetricsApiTest, InitializeFromBackendNamesDeduplicates) {
  EXPECT_THAT(store_.InitializeFromBackendNames(
                  {"prometheus", "Prometheus", " PROMETHEUS "}),
              IsOk());
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 1);
  EXPECT_THAT(store_.GetTextSnapshot(),
              HasSubstr("tpu_raiden_sent_bytes_total 1\n"));
  EXPECT_THAT(store_.GetAndResetMetricSamples(), IsEmpty());
}

TEST_F(MetricsApiTest, InitializeFromBackendNamesEmptyClearsBackends) {
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());

  EXPECT_THAT(store_.InitializeFromBackendNames({}), IsOk());
  EXPECT_FALSE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);
  EXPECT_THAT(store_.GetTextSnapshot(), IsEmpty());
  EXPECT_THAT(store_.GetAndResetMetricSamples(), IsEmpty());
}

TEST_F(MetricsApiTest,
       InitializeFromBackendNamesUnknownBackendFailsAllOrNothing) {
  EXPECT_FALSE(store_.HasBackends());
  EXPECT_THAT(
      store_.InitializeFromBackendNames({"unknown_backend"}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unknown telemetry backend: unknown_backend")));
  EXPECT_FALSE(store_.HasBackends());

  EXPECT_THAT(
      store_.InitializeFromBackendNames({"prometheus", "unknown_backend"}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unknown telemetry backend: unknown_backend")));
  EXPECT_FALSE(store_.HasBackends());
}

TEST_F(MetricsApiTest, InitializeFromEnvironmentReadsEnvVar) {
  ScopedEnvironmentVariable env_var(kTelemetryBackendsEnvVar,
                                    "prometheus,buffered");
  EXPECT_THAT(store_.InitializeFromEnvironment(), IsOk());
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);
  EXPECT_THAT(store_.GetTextSnapshot(),
              HasSubstr("tpu_raiden_sent_bytes_total 10"));
  EXPECT_THAT(store_.GetAndResetMetricSamples(),
              UnorderedElementsAre(
                  Pair("tpu_raiden_sent_bytes_total", ElementsAre(10.0))));
}

TEST_F(MetricsApiTest, InitializeFromEnvironmentEmptyOrUnsetNoOp) {
  EXPECT_THAT(store_.InitializeFromEnvironment(), IsOk());
  EXPECT_FALSE(store_.HasBackends());

  {
    ScopedEnvironmentVariable env_var(kTelemetryBackendsEnvVar, "   ,  , ");
    EXPECT_THAT(store_.InitializeFromEnvironment(), IsOk());
    EXPECT_FALSE(store_.HasBackends());
  }
}

TEST_F(MetricsApiTest, InitializeFromEnvironmentUnknownFails) {
  ScopedEnvironmentVariable env_var(kTelemetryBackendsEnvVar,
                                    "invalid_backend");
  EXPECT_THAT(
      store_.InitializeFromEnvironment(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unknown telemetry backend: invalid_backend")));
  EXPECT_FALSE(store_.HasBackends());
}

TEST_F(MetricsApiTest, InitializeFromEnvironmentNoOpIfAlreadyInitialized) {
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());

  ScopedEnvironmentVariable env_var(kTelemetryBackendsEnvVar,
                                    "invalid_backend");
  EXPECT_THAT(store_.InitializeFromEnvironment(), IsOk());
  EXPECT_TRUE(store_.HasBackends());
}

TEST_F(MetricsApiTest, InitializeWithPrometheusPortEnvironmentVariable) {
  int port = PickUnusedPort();
  std::string port_str = absl::StrCat(port);
  ScopedEnvironmentVariable port_env(kPrometheusPortEnvVar, port_str.c_str());
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());
}

TEST_F(MetricsApiTest, InitializeWithPrometheusHostEnvironmentVariable) {
  int port = PickUnusedPort();
  std::string port_str = absl::StrCat(port);
  ScopedEnvironmentVariable port_env(kPrometheusPortEnvVar, port_str.c_str());
  ScopedEnvironmentVariable host_env(kPrometheusHostEnvVar, "127.0.0.1");
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());
}

TEST_F(MetricsApiTest, InitializeWithInvalidPrometheusPortFallsBackGracefully) {
  ScopedEnvironmentVariable invalid_port_env(kPrometheusPortEnvVar,
                                             "invalid_port");
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());
}

TEST_F(MetricsApiTest,
       InitializeWithOutOfRangePrometheusPortFallsBackGracefully) {
  ScopedEnvironmentVariable out_of_range_env(kPrometheusPortEnvVar, "99999");
  EXPECT_THAT(store_.InitializeFromBackendNames({"prometheus"}), IsOk());
  EXPECT_TRUE(store_.HasBackends());
}

}  // namespace
}  // namespace tpu_raiden::telemetry

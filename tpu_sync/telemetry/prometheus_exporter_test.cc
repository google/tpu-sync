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

#include "tpu_sync/telemetry/prometheus_exporter.h"

#include <iterator>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "third_party/prometheus_cpp_client/core/include/prometheus/histogram.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/test_util.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::HasSubstr;

TEST(PrometheusExporterTest, ExporterOptionsDefaultHistogramBuckets) {
  ExporterOptions options;
  std::vector<double> expected(std::begin(kDefaultHistogramBuckets),
                               std::end(kDefaultHistogramBuckets));
  std::vector<double> actual(options.custom_buckets.begin(),
                             options.custom_buckets.end());
  EXPECT_EQ(actual, expected);
}

TEST(PrometheusExporterTest, RecordAndExportFormat) {
  PrometheusExporter exporter;

  MetricLabel label1{"interface", "ICI"};

  exporter.IncrementCounter(metric_names::kSentBytesTotal, {label1}, 1024);

  // Record transfer failures (Counter)
  const MetricLabel failure_labels[] = {
      {metric_labels::kErrorCode, "DEADLINE_EXCEEDED"},
      {metric_labels::kDirection, metric_labels::kDirectionPull},
  };
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal,
                            failure_labels, 2);

  // Record histogram observation
  exporter.ObserveHistogram(metric_names::kH2dTransferTimeMs, {}, 12.5);
  std::string output = exporter.GetTextSnapshot();

  EXPECT_THAT(output, HasSubstr("# TYPE tpu_raiden_sent_bytes_total counter"));
  EXPECT_THAT(output,
              HasSubstr("tpu_raiden_sent_bytes_total{interface=\"ICI\"} 1024"));

  EXPECT_THAT(output,
              HasSubstr("# TYPE tpu_raiden_transfer_failures_total counter"));
  EXPECT_THAT(output,
              HasSubstr("tpu_raiden_transfer_failures_total{direction=\"pull\","
                        "error_code=\"DEADLINE_EXCEEDED\"} 2"));

  EXPECT_THAT(output,
              HasSubstr("# TYPE tpu_raiden_h2d_transfer_time_ms histogram"));
  EXPECT_THAT(
      output,
      HasSubstr("tpu_raiden_h2d_transfer_time_ms_bucket{le=\"25\"} 1"));
  EXPECT_THAT(
      output,
      HasSubstr(
          "tpu_raiden_h2d_transfer_time_ms_bucket{le=\"5000\"} 1"));
}

TEST(PrometheusExporterTest, MetricSpecificCustomBuckets) {
  PrometheusExporter exporter;
  exporter.ObserveHistogram(metric_names::kH2dTransferTimeMs, {}, 1.5);
  exporter.ObserveHistogram(metric_names::kD2hTransferTimeMs, {}, 250.0);

  std::string output = exporter.GetTextSnapshot();

  // Verify custom millisecond buckets for H2D
  EXPECT_THAT(
      output,
      HasSubstr("tpu_raiden_h2d_transfer_time_ms_bucket{le=\"0.1\"} 0"));
  EXPECT_THAT(
      output,
      HasSubstr("tpu_raiden_h2d_transfer_time_ms_bucket{le=\"2.5\"} 1"));
  EXPECT_THAT(
      output,
      HasSubstr(
          "tpu_raiden_h2d_transfer_time_ms_bucket{le=\"5000\"} 1"));

  // Verify custom millisecond buckets for D2H
  EXPECT_THAT(
      output,
      HasSubstr("tpu_raiden_d2h_transfer_time_ms_bucket{le=\"100\"} 0"));
  EXPECT_THAT(
      output,
      HasSubstr("tpu_raiden_d2h_transfer_time_ms_bucket{le=\"250\"} 1"));
  EXPECT_THAT(
      output,
      HasSubstr(
          "tpu_raiden_d2h_transfer_time_ms_bucket{le=\"5000\"} 1"));
}

TEST(PrometheusExporterTest, UnmappedMetricIgnored) {
  PrometheusExporter exporter;
  exporter.IncrementCounter("custom_unmapped_counter", {}, 42);
  exporter.SetGauge("custom_unmapped_gauge", {}, 99);
  exporter.ObserveHistogram("custom_unmapped_histogram", {}, 1.23);

  std::string output = exporter.GetTextSnapshot();

  EXPECT_FALSE(absl::StrContains(output, "custom_unmapped_counter"));
  EXPECT_FALSE(absl::StrContains(output, "custom_unmapped_gauge"));
  EXPECT_FALSE(absl::StrContains(output, "custom_unmapped_histogram"));
}

TEST(PrometheusExporterTest, ConstReferenceAccess) {
  PrometheusExporter exporter;
  const PrometheusExporter& const_exporter = exporter;
  const MetricsBackend& const_backend = exporter;

  const_exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 500);

  std::string snapshot = const_backend.GetTextSnapshot();
  EXPECT_THAT(snapshot, HasSubstr("tpu_raiden_sent_bytes_total 500"));
}

TEST(PrometheusExporterTest, ConcurrentMetricUpdates) {
  PrometheusExporter exporter;
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&exporter]() {
      for (int j = 0; j < kIterations; ++j) {
        exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 1);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  std::string snapshot = exporter.GetTextSnapshot();
  EXPECT_THAT(snapshot, HasSubstr(absl::StrCat("tpu_raiden_sent_bytes_total ",
                                               kNumThreads * kIterations)));
}

TEST(PrometheusExporterTest, ServerDisabledWhenPortIsZero) {
  PrometheusExporter exporter_no_port;
  EXPECT_FALSE(exporter_no_port.IsServerRunning());
  EXPECT_EQ(exporter_no_port.GetBoundPort(), 0);
}

TEST(PrometheusExporterTest, ServerStartsWhenPortConfigured) {
  int port = PickUnusedPort();
  ExporterOptions options{
      .bind_address = "127.0.0.1",
      .port = port,
  };
  PrometheusExporter exporter_with_port(options);
  EXPECT_TRUE(exporter_with_port.IsServerRunning());
  EXPECT_EQ(exporter_with_port.GetBoundPort(), port);
}

TEST(PrometheusExporterTest, ServerDisabledWhenPortIsOutOfRange) {
  ExporterOptions options{
      .bind_address = "127.0.0.1",
      .port = 99999,
  };
  PrometheusExporter exporter(options);
  EXPECT_FALSE(exporter.IsServerRunning());
  EXPECT_EQ(exporter.GetBoundPort(), 0);
}

TEST(PrometheusExporterTest, ServerHandlesPortCollisionGracefully) {
  int port = PickUnusedPort();
  ExporterOptions options{
      .bind_address = "127.0.0.1",
      .port = port,
  };
  PrometheusExporter first_exporter(options);
  EXPECT_TRUE(first_exporter.IsServerRunning());

  // A second distinct exporter instance attempting to bind to the already
  // occupied port will fail socket binding (EADDRINUSE). It catches the
  // exception gracefully, leaves IsServerRunning() as false, and reports bound
  // port as 0.
  PrometheusExporter second_exporter(options);
  EXPECT_FALSE(second_exporter.IsServerRunning());
  EXPECT_EQ(second_exporter.GetBoundPort(), 0);
}

}  // namespace
}  // namespace tpu_raiden::telemetry

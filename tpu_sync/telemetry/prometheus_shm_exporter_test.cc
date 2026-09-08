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

#include "tpu_sync/telemetry/prometheus_shm_exporter.h"

#include <unistd.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/test_util.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

// Test fixture for PrometheusShmExporter validating shared memory metric
// aggregation, HTTP exposer lifecycle, port contention, and format conversions
// (counters, gauges, histograms).
class PrometheusShmExporterTest : public testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path temp_dir = testing::TempDir();
    test_dir_ = (temp_dir /
                 absl::StrCat("prom_shm_test_", getpid(), "_", absl::Hex(this)))
                    .string();
    std::error_code ec;
    std::filesystem::create_directories(test_dir_, ec);
    ASSERT_FALSE(ec) << ec.message();
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  ExporterOptions DefaultOptions(absl::string_view rank = "0",
                                 int port = 0) const {
    ExporterOptions options;
    options.shm_dir = test_dir_;
    options.local_rank = std::string(rank);
    options.bind_address = "127.0.0.1";
    options.port = port;
    return options;
  }

  std::string test_dir_;
};

// Options Validation & Port Edge Cases

TEST_F(PrometheusShmExporterTest, InvalidOrZeroPortGracefullyDisablesExposer) {
  for (int port : {0, -100, -1, 65536, 100000}) {
    PrometheusShmExporter exporter(DefaultOptions("0", port));
    EXPECT_FALSE(exporter.IsServerRunningForTesting());
  }

  // Metric emission and local snapshot function even when exposer is disabled.
  PrometheusShmExporter exporter(DefaultOptions("0", 0));
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 42);
  EXPECT_THAT(exporter.GetTextSnapshot(),
              HasSubstr(R"(tpu_raiden_sent_bytes_total{direction="push"} 42)"));
}

// HTTP Port Lifecycle, Collision, Recovery & Multi-threaded Contention

TEST_F(PrometheusShmExporterTest,
       HttpExposerLifecyclePortCollisionAndFallbackRecovery) {
  int port = 0;
  std::unique_ptr<PrometheusShmExporter> exp1;
  for (int attempt = 0; attempt < 3; ++attempt) {
    port = PickUnusedPort();
    if (port <= 0) continue;
    exp1 = std::make_unique<PrometheusShmExporter>(DefaultOptions("0", port));
    if (exp1->IsServerRunningForTesting()) break;
    exp1.reset();
  }
  if (!exp1 || !exp1->IsServerRunningForTesting()) {
    GTEST_SKIP() << "No free port available for HTTP exposer test";
  }

  ExporterOptions opt2 = DefaultOptions("1", port);

  // Secondary exporter detects collision, degrades gracefully without crash.
  PrometheusShmExporter exp2(opt2);
  EXPECT_FALSE(exp2.IsServerRunningForTesting());

  // Both workers can record metrics into shared memory.
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exp1->IncrementCounter(metric_names::kSentBytesTotal, labels, 100);
  exp2.IncrementCounter(metric_names::kSentBytesTotal, labels, 200);

  // Primary serves aggregated metrics from both workers.
  EXPECT_TRUE(exp1->IsServerRunningForTesting());
  EXPECT_THAT(exp1->GetTextSnapshot(),
              HasSubstr(R"(tpu_raiden_sent_bytes_total{)"
                        R"(direction="push"} 300)"));

  // In-process snapshot from secondary also reflects the aggregated total.
  EXPECT_THAT(exp2.GetTextSnapshot(),
              HasSubstr(R"(tpu_raiden_sent_bytes_total{)"
                        R"(direction="push"} 300)"));

  // Destroy primary exporter to release the port.
  exp1.reset();

  // A newly constructed exporter on that port can now bind and recover.
  PrometheusShmExporter exp3(DefaultOptions("0", port));
  EXPECT_TRUE(exp3.IsServerRunningForTesting());
  EXPECT_THAT(exp3.GetTextSnapshot(),
              HasSubstr(R"(tpu_raiden_sent_bytes_total{)"
                        R"(direction="push"} 200)"));
}

TEST_F(PrometheusShmExporterTest, MultiThreadedPortContentionExactlyOneBinds) {
  constexpr int kNumExporters = 8;
  const int port = PickUnusedPort();
  if (port <= 0) {
    GTEST_SKIP() << "No free port available for contention test";
  }

  std::vector<std::unique_ptr<PrometheusShmExporter>> exporters(kNumExporters);
  absl::Notification start_gate;
  std::vector<std::thread> threads;
  threads.reserve(kNumExporters);

  for (int i = 0; i < kNumExporters; ++i) {
    threads.emplace_back([this, port, i, &exporters, &start_gate]() {
      start_gate.WaitForNotification();
      exporters[i] = std::make_unique<PrometheusShmExporter>(
          DefaultOptions(absl::StrCat(i), port));
    });
  }

  start_gate.Notify();
  for (std::thread& t : threads) {
    t.join();
  }

  std::vector<int> running_ranks;
  for (int i = 0; i < kNumExporters; ++i) {
    if (exporters[i] != nullptr && exporters[i]->IsServerRunningForTesting()) {
      running_ranks.push_back(i);
    }
  }

  // Exactly 1 exporter must succeed in binding; the other 7 fail gracefully.
  EXPECT_EQ(running_ranks.size(), 1)
      << "Port contention test failed on port " << port << "; running ranks: ["
      << absl::StrJoin(running_ranks, ", ") << "]";
}

// Metric Translation, Aggregation & Exposition Snapshots

TEST_F(PrometheusShmExporterTest, MetricFamilyTranslationWithHistograms) {
  PrometheusShmExporter exporter(DefaultOptions());

  const MetricLabel push_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 1024);

  const MetricLabel fail_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull},
      {metric_labels::kErrorCode, "DEADLINE_EXCEEDED"}};
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, fail_labels,
                            3);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 512);
  exporter.SetGauge(metric_names::kBufferAllocatedBytes, {}, 65536.0);

  // Histograms: boundary values and rejection of non-finite values (NaN, Inf).
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            0.05);  // bucket <= 0.1
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            0.1);  // bucket <= 0.1 exact boundary
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            5.5);  // bucket <= 10.0
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            50000.0);  // bucket <= 50000.0 exact boundary
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            60000.0);  // bucket <= +Inf
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            std::numeric_limits<double>::quiet_NaN());
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            std::numeric_limits<double>::infinity());
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                            -std::numeric_limits<double>::infinity());

  // Unlabelled histogram observation.
  exporter.ObserveHistogram(metric_names::kH2dTransferTimeMs, {}, 15.5);

  const std::string snapshot = exporter.GetTextSnapshot();

  // Counters: unlabelled and labelled translations.
  EXPECT_THAT(snapshot, HasSubstr("# HELP tpu_raiden_sent_bytes_total"));
  EXPECT_THAT(snapshot,
              HasSubstr("# TYPE tpu_raiden_sent_bytes_total counter"));
  EXPECT_THAT(snapshot, HasSubstr("tpu_raiden_sent_bytes_total 512\n"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_sent_bytes_total{direction="push"} 1024)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_failures_total{direction="pull",)"
                R"(error_code="DEADLINE_EXCEEDED"} 3)"));

  // Gauge: automatic local_rank label attachment.
  EXPECT_THAT(snapshot,
              HasSubstr("# TYPE tpu_raiden_buffer_allocated_bytes gauge"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_buffer_allocated_bytes{local_rank="0"} 65536)"));

  // Histogram: sample count, sum, and cumulative bucket progression.
  EXPECT_THAT(snapshot,
              HasSubstr("# TYPE tpu_raiden_transfer_duration_ms histogram"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(
          R"(tpu_raiden_transfer_duration_ms_count{direction="push"} 5)"));
  EXPECT_THAT(snapshot, HasSubstr(R"(tpu_raiden_transfer_duration_ms_sum{)"
                                  R"(direction="push"} 110005.65)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(le="0.1"} 2)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(le="5"} 2)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(le="10"} 3)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(le="25000"} 3)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(le="50000"} 4)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(le="+Inf"} 5)"));

  // Non-finite values (NaN, -Inf) must not pollute the snapshot. Note: +Inf
  // rejection cannot be verified by substring check because the Prometheus
  // histogram bucket label le="+Inf" is legitimately present; its rejection is
  // verified above by sample_count == 5 and the exact finite sample_sum.
  EXPECT_THAT(snapshot, Not(HasSubstr("nan")));
  EXPECT_THAT(snapshot, Not(HasSubstr("NaN")));
  EXPECT_THAT(snapshot, Not(HasSubstr("-Inf")));

  // Unlabelled histogram exposition.
  EXPECT_THAT(snapshot,
              HasSubstr("# TYPE tpu_raiden_h2d_transfer_time_ms histogram"));
  EXPECT_THAT(snapshot, HasSubstr("tpu_raiden_h2d_transfer_time_ms_count 1"));
  EXPECT_THAT(snapshot, HasSubstr("tpu_raiden_h2d_transfer_time_ms_sum 15.5"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_h2d_transfer_time_ms_bucket{le="25"} 1)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_h2d_transfer_time_ms_bucket{le="+Inf"} 1)"));
}

TEST_F(PrometheusShmExporterTest, LabelParsingSortingAndSanitization) {
  PrometheusShmExporter exporter(DefaultOptions());

  const MetricLabel unsorted_labels[] = {
      {"zebra", "last"},  {"combo", R"(val;1=2\3)"},  {"empty_val", ""},
      {"apple", "first"}, {"path", R"(C:\dir\file)"},
  };
  exporter.IncrementCounter(metric_names::kSentBytesTotal, unsorted_labels, 42);

  const MetricLabel hist_labels[] = {
      {"direction", "push"},
      {"le", "custom_val"},
      {"test_tag", "esc;le=val"},
  };
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, hist_labels,
                            10.0);

  const std::string snapshot = exporter.GetTextSnapshot();

  // Escaped delimiters, empty values, and lexicographical label sorting:
  // apple < combo < empty_val < path < zebra.
  EXPECT_THAT(
      snapshot,
      HasSubstr(
          R"(tpu_raiden_sent_bytes_total{apple="first",combo="val;1=2\\3",)"
          R"(empty_val="",path="C:\\dir\\file",zebra="last"} 42)"));

  // Reserved "le" label must be sanitized out while preserving escaped
  // delimiters in other labels.
  EXPECT_THAT(snapshot, Not(HasSubstr("custom_val")));
  EXPECT_THAT(snapshot, HasSubstr(R"(test_tag="esc;le=val")"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_count{direction="push",)"
                R"(test_tag="esc;le=val"} 1)"));
  EXPECT_THAT(
      snapshot,
      HasSubstr(R"(tpu_raiden_transfer_duration_ms_bucket{direction="push",)"
                R"(test_tag="esc;le=val",le="10"} 1)"));
}

TEST_F(PrometheusShmExporterTest, EmptyExporterReturnsNoFamiliesOrSnapshot) {
  PrometheusShmExporter exporter(DefaultOptions());
  EXPECT_EQ(exporter.GetTextSnapshot(), "");
}

}  // namespace
}  // namespace tpu_raiden::telemetry

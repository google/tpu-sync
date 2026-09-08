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

#include "tpu_sync/telemetry/base_shm_exporter.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <array>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_collector.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {
namespace {

bool HasWorkerSegment(absl::string_view dir, absl::string_view prefix) {
  std::error_code ec;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(std::filesystem::path(dir), ec)) {
    if (ec) break;
    if (absl::StartsWith(entry.path().filename().string(), prefix)) {
      return true;
    }
  }
  return false;
}

class BaseShmExporterTest : public testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = absl::StrCat(
        testing::TempDir(), "/base_shm_exporter_test_", getpid(), "_",
        testing::UnitTest::GetInstance()->current_test_info()->name());
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
    std::filesystem::create_directories(test_dir_, ec);
    ASSERT_FALSE(ec);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  std::string test_dir_;
};

TEST_F(BaseShmExporterTest, CheckFailsWhenLocalRankMissingOrInvalid) {
  ExporterOptions options;
  options.shm_dir = test_dir_;
  options.local_rank = std::nullopt;

  EXPECT_DEATH((BaseShmExporter(options)), "local_rank");

  options.local_rank = "";
  EXPECT_DEATH((BaseShmExporter(options)), "local_rank");

  options.local_rank = "-1";
  EXPECT_DEATH((BaseShmExporter(options)), "local_rank");

  options.local_rank = "../0";
  EXPECT_DEATH((BaseShmExporter(options)), "local_rank");

  options.local_rank = "abc";
  EXPECT_DEATH((BaseShmExporter(options)), "local_rank");
}

TEST_F(BaseShmExporterTest, NormalizesAndCanonicalizesOptions) {
  ExporterOptions options;
  options.shm_dir = absl::StrCat(test_dir_, "///");
  options.local_rank = "007";

  BaseShmExporter exporter(options);
  EXPECT_EQ(exporter.options().shm_dir, test_dir_);
  EXPECT_EQ(exporter.options().local_rank, "7");
  EXPECT_TRUE(HasWorkerSegment(test_dir_, absl::StrCat(kShmFilePrefix, "7_")));
}

TEST_F(BaseShmExporterTest, CheckFailsWhenShmDirMissingOrInvalid) {
  ExporterOptions options;
  options.local_rank = "0";

  options.shm_dir = std::nullopt;
  EXPECT_DEATH((BaseShmExporter(options)), "shm_dir");

  options.shm_dir = "";
  EXPECT_DEATH((BaseShmExporter(options)), "shm_dir");

  options.shm_dir = "relative/path";
  EXPECT_DEATH((BaseShmExporter(options)), "shm_dir");

  options.shm_dir = "/";
  EXPECT_DEATH((BaseShmExporter(options)), "shm_dir");

  options.shm_dir = "///";
  EXPECT_DEATH((BaseShmExporter(options)), "shm_dir");
}

TEST_F(BaseShmExporterTest, RecordsAndAggregatesMetricsAcrossWorkers) {
  BaseShmExporter exporter0(ExporterOptions{
      .local_rank = "0",
      .shm_dir = test_dir_,
  });

  constexpr std::array<MetricLabel, 1> push_labels = {
      MetricLabel{.key = metric_labels::kDirection,
                  .value = metric_labels::kDirectionPush}};
  exporter0.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 1024);
  exporter0.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 512);

  constexpr std::array<MetricLabel, 1> pull_labels = {
      MetricLabel{.key = metric_labels::kDirection,
                  .value = metric_labels::kDirectionPull}};
  exporter0.SetGauge(metric_names::kBufferAllocatedBytes, pull_labels, 1024.0);
  exporter0.SetGauge("worker_state", {}, 1.0);

  exporter0.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                             12.5);
  EXPECT_EQ(exporter0.GetTextSnapshot(), "");

  {
    BaseShmExporter exporter1(ExporterOptions{
        .local_rank = "1",
        .shm_dir = test_dir_,
    });
    exporter1.IncrementCounter(metric_names::kSentBytesTotal, push_labels,
                               2000);
    exporter1.SetGauge(metric_names::kBufferAllocatedBytes, pull_labels,
                       2048.0);
    exporter1.SetGauge("worker_state", {}, 2.0);
    exporter1.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                               25.0);

    AggregatedMetrics metrics = exporter0.CollectMetrics();

    EXPECT_EQ(metrics.counters["sent_bytes_total"]["direction=push"], 3536);
    EXPECT_DOUBLE_EQ(
        metrics.gauges["buffer_allocated_bytes"]["direction=pull;local_rank=0"],
        1024.0);
    EXPECT_DOUBLE_EQ(
        metrics.gauges["buffer_allocated_bytes"]["direction=pull;local_rank=1"],
        2048.0);
    EXPECT_DOUBLE_EQ(metrics.gauges["worker_state"]["local_rank=0"], 1.0);
    EXPECT_DOUBLE_EQ(metrics.gauges["worker_state"]["local_rank=1"], 2.0);

    const HistogramData& hist =
        metrics.histograms["transfer_duration_ms"]["direction=push"];
    EXPECT_EQ(hist.sample_count, 2);
    EXPECT_DOUBLE_EQ(hist.sample_sum, 37.5);
    // Both 12.5 and 25.0 fall into bucket 7 (le=25.0)
    EXPECT_EQ(hist.bucket_counts[7], 2);
  }

  AggregatedMetrics metrics = exporter0.CollectMetrics();

  EXPECT_EQ(metrics.counters["sent_bytes_total"]["direction=push"], 1536);
  EXPECT_DOUBLE_EQ(
      metrics.gauges["buffer_allocated_bytes"]["direction=pull;local_rank=0"],
      1024.0);
  EXPECT_FALSE(metrics.gauges["buffer_allocated_bytes"].contains(
      "direction=pull;local_rank=1"));
  EXPECT_DOUBLE_EQ(metrics.gauges["worker_state"]["local_rank=0"], 1.0);
  EXPECT_FALSE(metrics.gauges["worker_state"].contains("local_rank=1"));

  const HistogramData& hist =
      metrics.histograms["transfer_duration_ms"]["direction=push"];
  EXPECT_EQ(hist.sample_count, 1);
  EXPECT_DOUBLE_EQ(hist.sample_sum, 12.5);
  // 12.5 falls into bucket 7 (le=25.0)
  EXPECT_EQ(hist.bucket_counts[7], 1);
}

TEST_F(BaseShmExporterTest, ReapsDeadWorkerSegmentOnCollection) {
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  absl::Cleanup pipe_cleanup_both = [&pipe_fds] {
    close(pipe_fds[0]);
    close(pipe_fds[1]);
  };
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  std::move(pipe_cleanup_both).Cancel();
  if (pid == 0) {
    close(pipe_fds[0]);
    ExporterOptions dead_options;
    dead_options.shm_dir = test_dir_;
    dead_options.local_rank = "99";
    BaseShmExporter dead_exporter(dead_options);
    constexpr std::array<MetricLabel, 1> labels = {
        MetricLabel{.key = metric_labels::kDirection,
                    .value = metric_labels::kDirectionPush}};
    dead_exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 500);
    char ready = 'R';
    if (write(pipe_fds[1], &ready, 1) != 1) {
      _exit(1);
    }
    close(pipe_fds[1]);
    _exit(0);  // Terminate without running ~BaseShmExporter() so segment file
               // remains.
  }
  close(pipe_fds[1]);
  absl::Cleanup pipe_cleanup = [read_fd = pipe_fds[0]] { close(read_fd); };
  bool child_reaped = false;
  int status = 0;
  absl::Cleanup child_cleanup = [pid, &status, &child_reaped] {
    if (!child_reaped && pid > 0) {
      waitpid(pid, &status, 0);
    }
  };
  char ready = 0;
  ssize_t bytes_read = read(pipe_fds[0], &ready, 1);
  std::move(pipe_cleanup).Cancel();
  close(pipe_fds[0]);
  pid_t waited_pid = waitpid(pid, &status, 0);
  child_reaped = true;
  ASSERT_EQ(bytes_read, 1);
  ASSERT_EQ(waited_pid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  ASSERT_TRUE(HasWorkerSegment(test_dir_, absl::StrCat(kShmFilePrefix, "99_")));

  ExporterOptions live_options;
  live_options.shm_dir = test_dir_;
  live_options.local_rank = "0";
  BaseShmExporter live_exporter(live_options);

  constexpr std::array<MetricLabel, 1> labels = {
      MetricLabel{.key = metric_labels::kDirection,
                  .value = metric_labels::kDirectionPush}};
  live_exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);

  EXPECT_EQ(live_exporter.CollectMetrics()
                .counters["sent_bytes_total"]["direction=push"],
            100);
  EXPECT_FALSE(
      HasWorkerSegment(test_dir_, absl::StrCat(kShmFilePrefix, "99_")));
}

TEST_F(BaseShmExporterTest, SetGaugeDropsNonFiniteValues) {
  BaseShmExporter exporter(ExporterOptions{
      .local_rank = "0",
      .shm_dir = test_dir_,
  });

  exporter.SetGauge("test_nan", {}, std::numeric_limits<double>::quiet_NaN());
  exporter.SetGauge("test_inf", {}, std::numeric_limits<double>::infinity());
  exporter.SetGauge("test_neg_inf", {},
                    -std::numeric_limits<double>::infinity());

  constexpr std::array<MetricLabel, 1> labels = {
      MetricLabel{.key = metric_labels::kDirection,
                  .value = metric_labels::kDirectionPush}};
  exporter.SetGauge("test_nan_labels", labels,
                    std::numeric_limits<double>::quiet_NaN());
  exporter.SetGauge("test_inf_labels", labels,
                    std::numeric_limits<double>::infinity());
  exporter.SetGauge("test_neg_inf_labels", labels,
                    -std::numeric_limits<double>::infinity());

  AggregatedMetrics metrics = exporter.CollectMetrics();
  EXPECT_FALSE(metrics.gauges.contains("test_nan"));
  EXPECT_FALSE(metrics.gauges.contains("test_inf"));
  EXPECT_FALSE(metrics.gauges.contains("test_neg_inf"));
  EXPECT_FALSE(metrics.gauges.contains("test_nan_labels"));
  EXPECT_FALSE(metrics.gauges.contains("test_inf_labels"));
  EXPECT_FALSE(metrics.gauges.contains("test_neg_inf_labels"));
}

TEST_F(BaseShmExporterTest, ConcurrentMetricRecording) {
  BaseShmExporter exporter(ExporterOptions{
      .local_rank = "0",
      .shm_dir = test_dir_,
  });

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&exporter, t]() {
      std::string t_str = absl::StrCat(t);
      MetricLabel thread_label{.key = "thread", .value = t_str};
      for (int i = 0; i < kOpsPerThread; ++i) {
        exporter.IncrementCounter(metric_names::kSentBytesTotal,
                                  LabelSpan(&thread_label, 1), 1);
        exporter.SetGauge(metric_names::kBufferAllocatedBytes,
                          LabelSpan(&thread_label, 1), static_cast<double>(i));
        exporter.ObserveHistogram(metric_names::kTransferDurationMs,
                                  LabelSpan(&thread_label, 1), 5.0);
      }
    });
  }

  for (std::thread& thread : threads) {
    thread.join();
  }

  AggregatedMetrics metrics = exporter.CollectMetrics();
  for (int t = 0; t < kNumThreads; ++t) {
    std::string counter_key = absl::StrCat("thread=", t);
    EXPECT_EQ(metrics.counters[metric_names::kSentBytesTotal][counter_key],
              kOpsPerThread);
    std::string gauge_key = absl::StrCat("local_rank=0;thread=", t);
    EXPECT_TRUE(metrics.gauges[metric_names::kBufferAllocatedBytes].contains(
        gauge_key));
    EXPECT_DOUBLE_EQ(
        metrics.gauges[metric_names::kBufferAllocatedBytes][gauge_key],
        static_cast<double>(kOpsPerThread - 1));
    EXPECT_EQ(metrics.histograms[metric_names::kTransferDurationMs][counter_key]
                  .sample_count,
              kOpsPerThread);
    EXPECT_DOUBLE_EQ(
        metrics.histograms[metric_names::kTransferDurationMs][counter_key]
            .sample_sum,
        static_cast<double>(kOpsPerThread) * 5.0);
  }
}

TEST_F(BaseShmExporterTest, SetGaugeLabelHandlingAndCapacity) {
  BaseShmExporter exporter(ExporterOptions{
      .local_rank = "2",
      .shm_dir = test_dir_,
  });

  // 1. Empty labels -> attaches local_rank.
  exporter.SetGauge("empty_label_gauge", {}, 10.0);

  // 2. Caller already provides local_rank -> preserves caller's local_rank.
  constexpr std::array<MetricLabel, 1> custom_rank_labels = {
      MetricLabel{.key = metric_labels::kLocalRank, .value = "99"}};
  exporter.SetGauge("custom_rank_gauge", custom_rank_labels, 30.0);

  // 3. Pre-sorted 8-label set (capacity boundary: 8 -> 9 slots, sorted
  // insertion).
  const std::string expected = "a=1;b=2;c=3;d=4;local_rank=2;m=5;n=6;y=7;z=8";
  const std::array<MetricLabel, 8> sorted_labels = {
      MetricLabel{"a", "1"}, MetricLabel{"b", "2"}, MetricLabel{"c", "3"},
      MetricLabel{"d", "4"}, MetricLabel{"m", "5"}, MetricLabel{"n", "6"},
      MetricLabel{"y", "7"}, MetricLabel{"z", "8"},
  };
  exporter.SetGauge("sorted_gauge", sorted_labels, 42.0);

  // 4. Unsorted 8-label set (triggers in-place sort path).
  const std::array<MetricLabel, 8> unsorted_labels = {
      MetricLabel{"z", "8"}, MetricLabel{"y", "7"}, MetricLabel{"n", "6"},
      MetricLabel{"m", "5"}, MetricLabel{"d", "4"}, MetricLabel{"c", "3"},
      MetricLabel{"b", "2"}, MetricLabel{"a", "1"},
  };
  exporter.SetGauge("unsorted_gauge", unsorted_labels, 42.0);

  AggregatedMetrics metrics = exporter.CollectMetrics();
  EXPECT_DOUBLE_EQ(metrics.gauges["empty_label_gauge"]["local_rank=2"], 10.0);
  EXPECT_DOUBLE_EQ(metrics.gauges["custom_rank_gauge"]["local_rank=99"], 30.0);
  EXPECT_FALSE(metrics.gauges["custom_rank_gauge"].contains("local_rank=2"));
  EXPECT_DOUBLE_EQ(metrics.gauges["sorted_gauge"][expected], 42.0);
  EXPECT_DOUBLE_EQ(metrics.gauges["unsorted_gauge"][expected], 42.0);
}

}  // namespace
}  // namespace tpu_raiden::telemetry

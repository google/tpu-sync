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

#include <unistd.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using testing::DoubleEq;
using testing::Pair;

class BaseShmExporterTest : public testing::Test {
 protected:
  void SetUp() override {
    test_dir_ =
        absl::StrCat(testing::TempDir(), "/base_shm_exporter_test_", getpid());
    std::filesystem::remove_all(test_dir_);
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  std::string test_dir_;
};

TEST_F(BaseShmExporterTest, CheckFailsWhenLocalRankMissingOrInvalid) {
  ExporterOptions options;
  options.shm_dir = test_dir_;
  options.local_rank = std::nullopt;

  EXPECT_DEATH(BaseShmExporter exporter(options), "local_rank");

  options.local_rank = "";
  EXPECT_DEATH(BaseShmExporter exporter(options), "local_rank");

  options.local_rank = "-1";
  EXPECT_DEATH(BaseShmExporter exporter(options), "local_rank");

  options.local_rank = "../0";
  EXPECT_DEATH(BaseShmExporter exporter(options), "local_rank");

  options.local_rank = "abc";
  EXPECT_DEATH(BaseShmExporter exporter(options), "local_rank");
}

TEST_F(BaseShmExporterTest, ValidatesAndUsesLocalRank) {
  ExporterOptions options;
  options.shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  EXPECT_EQ(exporter.GetOptions().local_rank, "0");
  EXPECT_EQ(exporter.GetOptions().shm_dir, test_dir_);

  bool found_rank_0 = false;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    const std::string filename = entry.path().filename().string();
    if (filename.rfind("worker_rank_0_", 0) == 0) {
      found_rank_0 = true;
    }
  }
  EXPECT_TRUE(found_rank_0);
}

TEST_F(BaseShmExporterTest, CheckFailsWhenShmDirMissingOrInvalid) {
  ExporterOptions options;
  options.local_rank = "0";
  options.shm_dir = std::nullopt;

  EXPECT_DEATH(BaseShmExporter exporter(options), "shm_dir");

  options.shm_dir = "";
  EXPECT_DEATH(BaseShmExporter exporter(options), "shm_dir");

  // When neither local_rank nor shm_dir is specified in options, death occurs
  // on local_rank first.
  ExporterOptions default_options;
  EXPECT_DEATH(BaseShmExporter exporter(default_options), "local_rank");
}

TEST_F(BaseShmExporterTest, NormalizesShmDir) {
  // Normalization of trailing slashes on option and segment creation.
  ExporterOptions slash_options;
  slash_options.shm_dir = absl::StrCat(test_dir_, "///");
  slash_options.local_rank = "0";
  BaseShmExporter slash_exporter(slash_options);
  EXPECT_EQ(slash_exporter.GetOptions().shm_dir, test_dir_);

  bool found_segment = false;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    const std::string filename = entry.path().filename().string();
    if (filename.rfind("worker_rank_0_", 0) == 0) {
      found_segment = true;
    }
  }
  EXPECT_TRUE(found_segment);

  // Preserves root directory "/" when trailing slashes stripped from option.
  ExporterOptions root_slash_options;
  root_slash_options.shm_dir = "///";
  root_slash_options.local_rank = "0";
  BaseShmExporter root_slash_exporter(root_slash_options);
  EXPECT_EQ(root_slash_exporter.GetOptions().shm_dir, "/");

  // Preserves root directory "/" directly on option.
  ExporterOptions root_options;
  root_options.shm_dir = "/";
  root_options.local_rank = "0";
  BaseShmExporter root_exporter(root_options);
  EXPECT_EQ(root_exporter.GetOptions().shm_dir, "/");
}

TEST_F(BaseShmExporterTest, SnapshotExtractionMethodsReturnEmpty) {
  ExporterOptions options;
  options.shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);
  exporter.SetGauge(metric_names::kBufferAllocatedBytes, labels, 2048.0);
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, labels, 10.0);

  EXPECT_EQ(exporter.GetTextSnapshot(), "");
  EXPECT_TRUE(exporter.GetAndResetMetricSamples().empty());
}

TEST_F(BaseShmExporterTest, RecordsAndAggregatesMetricsAcrossWorkers) {
  ExporterOptions options0;
  options0.shm_dir = test_dir_;
  options0.local_rank = "0";
  BaseShmExporter exporter0(options0);

  auto exporter1 = std::make_unique<BaseShmExporter>(ExporterOptions{
      .local_rank = "1",
      .shm_dir = test_dir_,
  });

  const MetricLabel push_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter0.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 1024);
  exporter0.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 512);
  exporter1->IncrementCounter(metric_names::kSentBytesTotal, push_labels, 2000);

  const MetricLabel pull_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull}};
  exporter0.SetGauge(metric_names::kBufferAllocatedBytes, pull_labels, 1024.0);
  exporter1->SetGauge(metric_names::kBufferAllocatedBytes, pull_labels, 2048.0);

  exporter0.ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                             12.5);
  exporter1->ObserveHistogram(metric_names::kTransferDurationMs, push_labels,
                              25.0);

  absl::flat_hash_map<std::string, double> totals;
  exporter0.CollectMetrics(totals);

  // Both ranks are live; metrics (counters, gauges, histograms) are aggregated
  // across segments.
  EXPECT_THAT(
      totals,
      testing::AllOf(
          testing::Contains(
              Pair("sent_bytes_total/direction=push", DoubleEq(3536.0))),
          testing::Contains(
              Pair("buffer_allocated_bytes/direction=pull", DoubleEq(3072.0))),
          testing::Contains(
              Pair("transfer_duration_ms/direction=push/count", DoubleEq(2.0))),
          testing::Contains(
              Pair("transfer_duration_ms/direction=push/sum", DoubleEq(37.5))),
          testing::Contains(Pair("transfer_duration_ms/direction=push/bucket_7",
                                 DoubleEq(2.0)))));

  // Terminate rank 1 process (releasing its flock on segment).
  exporter1.reset();

  // Collector reaps the dead worker segment and aggregates only live workers.
  exporter0.CollectMetrics(totals);
  EXPECT_THAT(
      totals,
      testing::AllOf(
          testing::Contains(
              Pair("sent_bytes_total/direction=push", DoubleEq(1536.0))),
          testing::Contains(
              Pair("buffer_allocated_bytes/direction=pull", DoubleEq(1024.0))),
          testing::Contains(
              Pair("transfer_duration_ms/direction=push/count", DoubleEq(1.0))),
          testing::Contains(
              Pair("transfer_duration_ms/direction=push/sum", DoubleEq(12.5))),
          testing::Contains(Pair("transfer_duration_ms/direction=push/bucket_7",
                                 DoubleEq(1.0)))));
}

}  // namespace
}  // namespace tpu_raiden::telemetry

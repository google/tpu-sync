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

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <stdexcept>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::NotNull;

class BaseShmExporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = absl::StrCat(::testing::TempDir(), "/base_shm_exporter_test_",
                             getpid());
    std::filesystem::remove_all(test_dir_);
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    unsetenv("LOCAL_RANK");
    unsetenv("SHM_DIR");
    std::filesystem::remove_all(test_dir_);
  }

  std::string test_dir_;
};

TEST_F(BaseShmExporterTest, ThrowsWhenLocalRankMissing) {
  unsetenv("LOCAL_RANK");
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = std::nullopt;

  EXPECT_THROW(BaseShmExporter exporter(options), std::invalid_argument);

  options.local_rank = "";
  EXPECT_THROW(BaseShmExporter exporter(options), std::invalid_argument);

  setenv("LOCAL_RANK", "", 1);
  options.local_rank = std::nullopt;
  EXPECT_THROW(BaseShmExporter exporter(options), std::invalid_argument);
}

TEST_F(BaseShmExporterTest, ResolvesLocalRankFromOptions) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  EXPECT_THAT(exporter.GetWriter(), NotNull());
  EXPECT_THAT(exporter.GetCollector(), NotNull());
  EXPECT_EQ(exporter.GetOptions().local_rank, "0");
}

TEST_F(BaseShmExporterTest, ResolvesLocalRankFromEnvironment) {
  setenv("LOCAL_RANK", "3", 1);
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = std::nullopt;

  BaseShmExporter exporter(options);
  EXPECT_THAT(exporter.GetWriter(), NotNull());
  EXPECT_THAT(exporter.GetCollector(), NotNull());
  EXPECT_EQ(exporter.GetOptions().local_rank, "3");
}

TEST_F(BaseShmExporterTest, ResolvesShmDirFromOption) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  EXPECT_EQ(options.GetShmDir(), test_dir_);
  BaseShmExporter exporter(options);
  EXPECT_EQ(exporter.GetOptions().GetShmDir(), test_dir_);
}

TEST_F(BaseShmExporterTest, ResolvesShmDirFromEnvironment) {
  setenv("SHM_DIR", test_dir_.c_str(), 1);
  ExporterOptions options;
  options.local_rank = "0";

  EXPECT_EQ(options.GetShmDir(), test_dir_);
  BaseShmExporter exporter(options);
  EXPECT_EQ(exporter.GetOptions().GetShmDir(), test_dir_);
}

TEST_F(BaseShmExporterTest, ResolvesShmDirDefaultFallback) {
  unsetenv("SHM_DIR");
  ExporterOptions options;
  EXPECT_EQ(options.GetShmDir(), "/dev/shm");

  setenv("SHM_DIR", "", 1);
  EXPECT_EQ(options.GetShmDir(), "/dev/shm");
}

TEST_F(BaseShmExporterTest, LifecycleStartStopIsRunning) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  EXPECT_FALSE(exporter.IsRunning());

  exporter.Start();
  EXPECT_TRUE(exporter.IsRunning());

  exporter.Stop();
  EXPECT_FALSE(exporter.IsRunning());
}

TEST_F(BaseShmExporterTest, GetTextSnapshotReturnsEmpty) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);

  EXPECT_EQ(exporter.GetTextSnapshot(), "");
}

TEST_F(BaseShmExporterTest, GetAndResetMetricSamplesReturnsEmpty) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);

  EXPECT_TRUE(exporter.GetAndResetMetricSamples().empty());
}

TEST_F(BaseShmExporterTest, MetricRecordingAndCollection) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);

  const MetricLabel sent_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, sent_labels, 1024);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, sent_labels, 512);

  const MetricLabel gauge_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull}};
  exporter.SetGauge(metric_names::kBufferAllocatedBytes, gauge_labels, 4096.0);

  exporter.ObserveHistogram(metric_names::kTransferDurationMs, sent_labels,
                            12.5);

  absl::flat_hash_map<std::string, double> totals;
  exporter.CollectMetrics(totals);

  EXPECT_EQ(totals["sent_bytes_total/direction=push"], 1536.0);
  EXPECT_EQ(totals["buffer_allocated_bytes/direction=pull"], 4096.0);
  EXPECT_EQ(totals["transfer_duration_ms/direction=push/count"], 1.0);
  EXPECT_EQ(totals["transfer_duration_ms/direction=push/sum"], 12.5);
}

}  // namespace
}  // namespace tpu_raiden::telemetry

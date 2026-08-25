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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "third_party/prometheus_cpp_client/core/include/prometheus/metric_family.h"
#include "tpu_sync/telemetry/base_shm_exporter.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/test_util.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::HasSubstr;

std::string HttpGet(int port, const std::string& path) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return "";

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return "";
  }

  std::string req = absl::StrCat("GET ", path,
                                 " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: "
                                 "close\r\n\r\n");
  send(fd, req.data(), req.size(), 0);

  std::string response;
  char buf[4096];
  ssize_t n = 0;
  while ((n = recv(fd, buf, sizeof(buf), 0)) > 0) {
    response.append(buf, n);
  }
  close(fd);
  return response;
}

class PrometheusShmExporterTest : public testing::Test {
 protected:
  void SetUp() override {
    const char* tmp = std::getenv("TEST_TMPDIR");
    test_dir_ = tmp ? absl::StrCat(tmp, "/prom_shm_test_", getpid())
                    : absl::StrCat("/tmp/prom_shm_test_", getpid());
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  std::string test_dir_;
};

TEST_F(PrometheusShmExporterTest, ThrowsWhenLocalRankNotSpecified) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = std::nullopt;
  EXPECT_THROW(PrometheusShmExporter exporter(options), std::invalid_argument);
}

TEST_F(PrometheusShmExporterTest, BaseExporterLifecycle) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  EXPECT_EQ(exporter.GetOptions().port, 0);
  EXPECT_NE(exporter.GetWriter(), nullptr);
  EXPECT_NE(exporter.GetCollector(), nullptr);
  EXPECT_FALSE(exporter.IsRunning());

  exporter.Start();
  EXPECT_TRUE(exporter.IsRunning());

  exporter.Stop();
  EXPECT_FALSE(exporter.IsRunning());
}

TEST_F(PrometheusShmExporterTest, BaseExporterMetricForwarding) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";

  BaseShmExporter exporter(options);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 2048);

  absl::flat_hash_map<std::string, double> totals;
  exporter.CollectMetrics(totals);
  EXPECT_EQ(totals["sent_bytes_total/direction=push"], 2048);
}

TEST_F(PrometheusShmExporterTest, TextSnapshotContainsMetricFormatting) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";
  options.port = 0;

  PrometheusShmExporter exporter(options);
  const MetricLabel sent_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, sent_labels, 1024);

  const MetricLabel fail_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull},
      {metric_labels::kErrorCode, "DEADLINE_EXCEEDED"}};
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, fail_labels,
                            3);

  std::string snapshot = exporter.GetTextSnapshot();
  EXPECT_THAT(snapshot, HasSubstr("# HELP tpu_raiden_sent_bytes_total"));
  EXPECT_THAT(snapshot,
              HasSubstr("# TYPE tpu_raiden_sent_bytes_total counter"));
  EXPECT_THAT(
      snapshot,
      HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 1024"));
  EXPECT_THAT(
      snapshot,
      HasSubstr("tpu_raiden_transfer_failures_total{direction=\"pull\",error_"
                "code=\"DEADLINE_EXCEEDED\"} 3"));
}

TEST_F(PrometheusShmExporterTest, ExposerDisabledWhenPortZero) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";
  options.port = 0;

  PrometheusShmExporter exporter(options);
  EXPECT_FALSE(exporter.IsExposerRunning());
  exporter.Start();
  EXPECT_FALSE(exporter.IsExposerRunning());
  exporter.Stop();
}

TEST_F(PrometheusShmExporterTest, ExposerStartsAndServesHttpScrape) {
  int port = PickUnusedPort();
  if (port <= 0) {
    GTEST_SKIP() << "No free port available for HTTP exposer test";
  }

  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";
  options.bind_address = "127.0.0.1";
  options.port = port;

  PrometheusShmExporter exporter(options);
  EXPECT_TRUE(exporter.IsExposerRunning());

  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 777);

  std::string response = HttpGet(port, "/metrics");
  EXPECT_THAT(response, HasSubstr("HTTP/1.1 200 OK"));
  EXPECT_THAT(response,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 777"));

  exporter.Stop();
  EXPECT_FALSE(exporter.IsExposerRunning());
}

TEST_F(PrometheusShmExporterTest, ExposerGracefulDegradationOnPortCollision) {
  int port = PickUnusedPort();
  if (port <= 0) {
    GTEST_SKIP() << "No free port available for port collision test";
  }

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  int on = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  ASSERT_EQ(bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  ASSERT_EQ(listen(sock, 1), 0);

  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";
  options.bind_address = "127.0.0.1";
  options.port = port;

  PrometheusShmExporter exporter(options);
  EXPECT_FALSE(exporter.IsExposerRunning());

  close(sock);
}

TEST_F(PrometheusShmExporterTest, CollectMetricFamiliesDataIntegrity) {
  ExporterOptions options;
  options.base_shm_dir = test_dir_;
  options.local_rank = "0";
  options.port = 0;

  PrometheusShmExporter exporter(options);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 500);

  std::vector<prometheus::MetricFamily> families =
      exporter.CollectMetricFamilies();
  EXPECT_FALSE(families.empty());

  bool found_sent = false;
  for (const auto& fam : families) {
    if (fam.name == "tpu_raiden_sent_bytes_total") {
      found_sent = true;
      EXPECT_EQ(fam.type, prometheus::MetricType::Counter);
      ASSERT_FALSE(fam.metric.empty());
      bool found_push = false;
      for (const auto& m : fam.metric) {
        for (const auto& lbl : m.label) {
          if (lbl.name == "direction" && lbl.value == "push") {
            found_push = true;
            EXPECT_EQ(m.counter.value, 500.0);
          }
        }
      }
      EXPECT_TRUE(found_push);
    }
  }
  EXPECT_TRUE(found_sent);
}

TEST_F(PrometheusShmExporterTest, DecentralizedApproachANoLeaderLock) {
  ExporterOptions options1;
  options1.base_shm_dir = test_dir_;
  options1.local_rank = "0";
  options1.port = 0;

  ExporterOptions options2;
  options2.base_shm_dir = test_dir_;
  options2.local_rank = "1";
  options2.port = 0;

  PrometheusShmExporter exporter1(options1);
  PrometheusShmExporter exporter2(options2);

  EXPECT_FALSE(std::filesystem::exists(test_dir_ + "/leader.lock"));
}

TEST_F(PrometheusShmExporterTest, MultiWorkerAggregationInTextSnapshot) {
  ExporterOptions options1;
  options1.base_shm_dir = test_dir_;
  options1.local_rank = "0";
  options1.port = 0;

  ExporterOptions options2;
  options2.base_shm_dir = test_dir_;
  options2.local_rank = "1";
  options2.port = 0;

  PrometheusShmExporter exporter1(options1);
  PrometheusShmExporter exporter2(options2);

  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter1.IncrementCounter(metric_names::kSentBytesTotal, labels, 150);
  exporter2.IncrementCounter(metric_names::kSentBytesTotal, labels, 350);

  std::string snapshot = exporter1.GetTextSnapshot();
  EXPECT_THAT(snapshot,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 500"));
}

TEST_F(PrometheusShmExporterTest, DeadWorkerUnlinkedOnSweepInTextSnapshot) {
  ExporterOptions options1;
  options1.base_shm_dir = test_dir_;
  options1.local_rank = "0";
  options1.port = 0;

  PrometheusShmExporter exporter1(options1);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exporter1.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);

  std::string file2;
  {
    ExporterOptions options2;
    options2.base_shm_dir = test_dir_;
    options2.local_rank = "1";
    options2.port = 0;

    PrometheusShmExporter exporter2(options2);
    file2 = exporter2.GetWriter()->file_path();
    exporter2.IncrementCounter(metric_names::kSentBytesTotal, labels, 400);
  }

  std::string snapshot = exporter1.GetTextSnapshot();
  // Exporter 1 remains (100); dead exporter 2 unlinked
  EXPECT_THAT(snapshot,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 100"));
  EXPECT_FALSE(std::filesystem::exists(file2));
}

}  // namespace
}  // namespace tpu_raiden::telemetry

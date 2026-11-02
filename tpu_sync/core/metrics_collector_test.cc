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

#include "tpu_sync/core/metrics_collector.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

TEST(MetricsCollectorTest, BasicSequenceAndJsonDump) {
  MetricsCollector collector;
  uint64_t uuid = 12345;

  collector.RecordStart(uuid, "req_1", 10, 10240);
  absl::SleepFor(absl::Milliseconds(5));
  collector.RecordFirstPacket(uuid);
  absl::SleepFor(absl::Milliseconds(10));
  collector.RecordLastPacket(uuid);
  collector.RecordH2dEnqueue(uuid);
  absl::SleepFor(absl::Milliseconds(5));
  collector.RecordH2dComplete(uuid);
  collector.RecordEnd(uuid);

  std::string test_path = "/tmp/test_raiden_metrics.json";
  collector.WriteJsonReport(test_path);

  std::ifstream f(test_path);
  ASSERT_TRUE(f.is_open());
  std::stringstream ss;
  ss << f.rdbuf();
  std::string content = ss.str();

  EXPECT_TRUE(absl::StrContains(content, R"("uuid": 12345)"));
  EXPECT_TRUE(absl::StrContains(content, R"("req_id": "req_1")"));
  EXPECT_TRUE(absl::StrContains(content, "connection_setup_us"));
  EXPECT_TRUE(absl::StrContains(content, "network_transit_us"));
  EXPECT_TRUE(absl::StrContains(content, "h2d_copy_us"));
}

TEST(MetricsCollectorTest, GetTransferMetricsReturnsRecordedTimestamps) {
  MetricsCollector collector;
  constexpr uint64_t kUuid = 4242;
  EXPECT_FALSE(collector.GetTransferMetrics(kUuid).has_value());

  collector.RecordStart(kUuid, "typed", /*num_blocks=*/3, /*total_bytes=*/384);
  collector.RecordFirstPacket(kUuid);
  collector.RecordLastPacket(kUuid);
  collector.RecordEnd(kUuid);

  std::optional<TransferMetrics> metrics = collector.GetTransferMetrics(kUuid);
  ASSERT_TRUE(metrics.has_value());
  EXPECT_EQ(metrics->uuid, kUuid);
  EXPECT_EQ(metrics->req_id, "typed");
  EXPECT_EQ(metrics->num_blocks, 3);
  EXPECT_EQ(metrics->total_bytes, 384);
  EXPECT_LE(metrics->start_time, metrics->first_packet_time);
  EXPECT_LE(metrics->first_packet_time, metrics->last_packet_time);
  EXPECT_LE(metrics->last_packet_time, metrics->end_time);
}

TEST(MetricsCollectorTest, AsynchronousNicBandwidthTracking) {
  // 1. Create a temporary directory for mock sysfs
  std::string mock_sysfs = "/tmp/mock_sysfs";
  std::filesystem::create_directories(mock_sysfs + "/eth0/statistics");

  // Write initial stats: rx=1000, tx=2000
  {
    std::ofstream rx(mock_sysfs + "/eth0/statistics/rx_bytes");
    rx << 1000;
    std::ofstream tx(mock_sysfs + "/eth0/statistics/tx_bytes");
    tx << 2000;
  }

  // 2. Start collector pointing to mock sysfs
  MetricsCollector collector(mock_sysfs);
  uint64_t uuid = 999;

  collector.RecordStart(uuid, "req_async", 1, 1000000);  // 1MB

  // Simulating transfer time
  absl::SleepFor(absl::Milliseconds(10));

  // Write final stats: rx=2001000 (2MB transferred), tx=4000
  {
    std::ofstream rx(mock_sysfs + "/eth0/statistics/rx_bytes");
    rx << 2001000;
    std::ofstream tx(mock_sysfs + "/eth0/statistics/tx_bytes");
    tx << 4000;
  }

  collector.RecordEnd(uuid);

  // Allow some time for background thread to process the End event and capture
  // stats
  absl::SleepFor(absl::Milliseconds(20));

  std::string test_path = "/tmp/test_raiden_metrics_async.json";
  collector.WriteJsonReport(test_path);

  std::ifstream f(test_path);
  ASSERT_TRUE(f.is_open());
  std::stringstream ss;
  ss << f.rdbuf();
  std::string content = ss.str();

  EXPECT_TRUE(absl::StrContains(content, R"("uuid": 999)"));
  EXPECT_TRUE(absl::StrContains(content, R"("req_id": "req_async")"));
  EXPECT_TRUE(absl::StrContains(content, "nic_wire_bandwidth_gbps"));
  EXPECT_TRUE(absl::StrContains(content, "eth0_rx"));

  // Cleanup
  std::filesystem::remove_all(mock_sysfs);
}

}  // namespace
}  // namespace tpu_raiden

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_METRICS_COLLECTOR_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_METRICS_COLLECTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"

namespace tpu_raiden {

enum class MetricsEventType {
  kStart,
  kFirstPacket,
  kLastPacket,
  kH2dEnqueue,
  kH2dComplete,
  kEnd
};

struct NicBytes {
  uint64_t rx_bytes = 0;
  uint64_t tx_bytes = 0;
};

// Trivial, fixed-size event struct for lock-free queue safety
struct MetricsEvent {
  uint64_t uuid = 0;
  char req_id[64] = {0};
  MetricsEventType type;
  absl::Time timestamp;
  int64_t num_blocks = 0;
  int64_t total_bytes = 0;
};

struct TransferMetrics {
  uint64_t uuid;
  std::string req_id;

  absl::Time start_time;
  absl::Time first_packet_time;
  absl::Time last_packet_time;
  absl::Time h2d_enqueue_time;
  absl::Time h2d_complete_time;
  absl::Time end_time;

  // Interface-level byte snapshots
  absl::flat_hash_map<std::string, NicBytes> initial_nic_bytes;
  absl::flat_hash_map<std::string, NicBytes> final_nic_bytes;

  int64_t total_bytes = 0;
  int64_t num_blocks = 0;
};

class MetricsCollector {
 public:
  explicit MetricsCollector(std::string sysfs_dir = "/sys/class/net");
  ~MetricsCollector();

  // Lock-free, non-blocking event push (takes nanoseconds)
  void RecordStart(uint64_t uuid, const std::string& req_id, int64_t num_blocks,
                   int64_t total_bytes);
  void RecordFirstPacket(uint64_t uuid);
  void RecordLastPacket(uint64_t uuid);
  void RecordH2dEnqueue(uint64_t uuid);
  void RecordH2dComplete(uint64_t uuid);
  void RecordEnd(uint64_t uuid);

  void WriteJsonReport(
      const std::string& filepath = "/tmp/raiden_metrics.json");
  std::string DumpMetricsToString();

  // The metrics recorded for `uuid` (the same ones `DumpMetricsToString`
  // reports), or nullopt before `RecordStart` for it. A slot is shared by the
  // uuids congruent modulo kMaxMetrics, so the result is the latest of them.
  // Read it once the transfer has settled, as the Record* calls race with it.
  std::optional<TransferMetrics> GetTransferMetrics(uint64_t uuid) const;

 private:
  absl::flat_hash_map<std::string, NicBytes> SnapshotAllNics();

  std::string sysfs_dir_;
  std::vector<std::string> monitored_interfaces_;

  static constexpr size_t kMaxMetrics = 1024;
  std::atomic<bool> metrics_valid_[kMaxMetrics];
  TransferMetrics metrics_array_[kMaxMetrics];
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_METRICS_COLLECTOR_H_

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

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/prometheus_cpp_client/core/include/prometheus/metric_family.h"
#include "tpu_sync/telemetry/base_shm_exporter.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/prometheus_shm_exporter.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"
#include "tpu_sync/telemetry/test_util.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::Le;
using ::testing::Not;

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

class PrometheusShmExporterChallengerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* tmp = std::getenv("TEST_TMPDIR");
    test_dir_ = tmp ? absl::StrCat(tmp, "/prom_challenger_", getpid(), "_",
                                   reinterpret_cast<uintptr_t>(this))
                    : absl::StrCat("/tmp/prom_challenger_", getpid(), "_",
                                   reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  std::string test_dir_;
};

// ============================================================================
// CHALLENGE AREA 1: Port Collisions & Binding Resilience
// ============================================================================

TEST_F(PrometheusShmExporterChallengerTest,
       PortCollisionTwoExportersGracefulDegradation) {
  int port = PickUnusedPort();
  ASSERT_GT(port, 0);

  ExporterOptions opt1;
  opt1.base_shm_dir = test_dir_;
  opt1.local_rank = "0";
  opt1.bind_address = "127.0.0.1";
  opt1.port = port;

  ExporterOptions opt2;
  opt2.base_shm_dir = test_dir_;
  opt2.local_rank = "1";
  opt2.bind_address = "127.0.0.1";
  opt2.port = port;

  PrometheusShmExporter exp1(opt1);
  EXPECT_TRUE(exp1.IsExposerRunning());

  // Second exporter attempts to bind to the identical port.
  // Must catch exception internally, log ERROR, and NOT crash.
  PrometheusShmExporter exp2(opt2);
  EXPECT_FALSE(exp2.IsExposerRunning());

  // Both writers must still function normally for shared-memory writes.
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exp1.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);
  exp2.IncrementCounter(metric_names::kSentBytesTotal, labels, 200);

  // Exporter 1 serves HTTP request containing aggregated metrics.
  std::string http_res = HttpGet(port, "/metrics");
  EXPECT_THAT(http_res, HasSubstr("HTTP/1.1 200 OK"));
  EXPECT_THAT(http_res,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 300"));

  // In-process snapshot from exp2 still works despite exposer failure.
  std::string snapshot2 = exp2.GetTextSnapshot();
  EXPECT_THAT(snapshot2,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 300"));
}

TEST_F(PrometheusShmExporterChallengerTest,
       PortCollisionRecoveryAfterPrimaryStops) {
  int port = PickUnusedPort();
  ASSERT_GT(port, 0);

  ExporterOptions opt1;
  opt1.base_shm_dir = test_dir_;
  opt1.local_rank = "0";
  opt1.bind_address = "127.0.0.1";
  opt1.port = port;

  ExporterOptions opt2;
  opt2.base_shm_dir = test_dir_;
  opt2.local_rank = "1";
  opt2.bind_address = "127.0.0.1";
  opt2.port = port;

  auto exp1 = std::make_unique<PrometheusShmExporter>(opt1);
  EXPECT_TRUE(exp1->IsExposerRunning());

  auto exp2 = std::make_unique<PrometheusShmExporter>(opt2);
  EXPECT_FALSE(exp2->IsExposerRunning());

  // Stop primary exporter to release port.
  exp1->Stop();
  EXPECT_FALSE(exp1->IsExposerRunning());

  // Now retry starting secondary exporter.
  exp2->Start();
  EXPECT_TRUE(exp2->IsExposerRunning());

  std::string http_res = HttpGet(port, "/metrics");
  EXPECT_THAT(http_res, HasSubstr("HTTP/1.1 200 OK"));

  exp2->Stop();
}

TEST_F(PrometheusShmExporterChallengerTest,
       MassivePortCollisionContentionMultiThreaded) {
  int port = PickUnusedPort();
  ASSERT_GT(port, 0);

  constexpr int kNumExporters = 8;
  std::vector<std::unique_ptr<PrometheusShmExporter>> exporters(kNumExporters);
  std::vector<std::thread> threads;
  threads.reserve(kNumExporters);

  for (int i = 0; i < kNumExporters; ++i) {
    threads.emplace_back([this, port, i, &exporters]() {
      ExporterOptions opt;
      opt.base_shm_dir = test_dir_;
      opt.local_rank = absl::StrCat(i);
      opt.bind_address = "127.0.0.1";
      opt.port = port;
      exporters[i] = std::make_unique<PrometheusShmExporter>(opt);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  int running_count = 0;
  for (int i = 0; i < kNumExporters; ++i) {
    if (exporters[i]->IsExposerRunning()) {
      running_count++;
    }
  }

  // Exactly 1 exporter must succeed in binding; the other 7 fail gracefully.
  EXPECT_EQ(running_count, 1);

  // All exporters clean up cleanly without hanging or crashing.
  for (int i = 0; i < kNumExporters; ++i) {
    exporters[i]->Stop();
  }
}

TEST_F(PrometheusShmExporterChallengerTest,
       InvalidPortConfigurationsHandledGracefully) {
  const int invalid_ports[] = {-100, -1, 65536, 100000};
  for (int p : invalid_ports) {
    ExporterOptions opt;
    opt.base_shm_dir = test_dir_;
    opt.local_rank = "0";
    opt.port = p;

    PrometheusShmExporter exp(opt);
    EXPECT_FALSE(exp.IsExposerRunning());
  }
}

// ============================================================================
// CHALLENGE AREA 2: Disabled Exposer (port = 0)
// ============================================================================

TEST_F(PrometheusShmExporterChallengerTest,
       DisabledExposerZeroPortFullLocalFunctionality) {
  ExporterOptions opt;
  opt.base_shm_dir = test_dir_;
  opt.local_rank = "0";
  opt.port = 0;

  PrometheusShmExporter exp(opt);
  EXPECT_FALSE(exp.IsExposerRunning());

  // Start() explicitly called should not start exposer if port == 0.
  exp.Start();
  EXPECT_FALSE(exp.IsExposerRunning());

  // Metric emission works.
  const MetricLabel sent_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  exp.IncrementCounter(metric_names::kSentBytesTotal, sent_labels, 555);

  const MetricLabel recv_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull}};
  exp.IncrementCounter(metric_names::kReceivedBytesTotal, recv_labels, 888);

  std::string snapshot = exp.GetTextSnapshot();
  EXPECT_THAT(snapshot,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 555"));
  EXPECT_THAT(
      snapshot,
      HasSubstr("tpu_raiden_received_bytes_total{direction=\"pull\"} 888"));

  auto families = exp.CollectMetricFamilies();
  EXPECT_FALSE(families.empty());

  exp.Stop();
  EXPECT_FALSE(exp.IsExposerRunning());
}

TEST_F(PrometheusShmExporterChallengerTest,
       MultipleZeroPortExportersConcurrentSharedMemoryAggregation) {
  constexpr int kWorkers = 4;
  std::vector<std::unique_ptr<PrometheusShmExporter>> exps;
  exps.reserve(kWorkers);

  for (int i = 0; i < kWorkers; ++i) {
    ExporterOptions opt;
    opt.base_shm_dir = test_dir_;
    opt.local_rank = absl::StrCat(i);
    opt.port = 0;
    exps.push_back(std::make_unique<PrometheusShmExporter>(opt));
  }

  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};

  for (int i = 0; i < kWorkers; ++i) {
    exps[i]->IncrementCounter(metric_names::kSentBytesTotal, labels, 1000);
  }

  // Any worker querying snapshot must see the full aggregate (4 * 1000 = 4000).
  for (int i = 0; i < kWorkers; ++i) {
    std::string snap = exps[i]->GetTextSnapshot();
    EXPECT_THAT(
        snap,
        HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 4000"));
  }
}

// ============================================================================
// CHALLENGE AREA 3: Dead Worker Reaping Integration via Snapshot Calls
// ============================================================================

TEST_F(PrometheusShmExporterChallengerTest,
       DeadWorkerReapingIntegratedInTextSnapshotAndMetricFamilies) {
  ExporterOptions live_opt;
  live_opt.base_shm_dir = test_dir_;
  live_opt.local_rank = "0";
  live_opt.port = 0;

  PrometheusShmExporter live_exp(live_opt);

  // Simulate dead worker by creating it in a scope and destroying it.
  {
    ExporterOptions dead_opt;
    dead_opt.base_shm_dir = test_dir_;
    dead_opt.local_rank = "1";
    dead_opt.port = 0;

    PrometheusShmExporter dead_exp(dead_opt);
    const MetricLabel sent_labels[] = {
        {metric_labels::kDirection, metric_labels::kDirectionPush}};
    dead_exp.IncrementCounter(metric_names::kSentBytesTotal, sent_labels, 450);

    const MetricLabel fail_labels[] = {
        {metric_labels::kDirection, metric_labels::kDirectionPull},
        {metric_labels::kErrorCode, "UNAVAILABLE"}};
    dead_exp.IncrementCounter(metric_names::kTransferFailuresTotal, fail_labels,
                              7);
  }

  // Live worker emits metrics before calling snapshot.
  const MetricLabel live_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};
  live_exp.IncrementCounter(metric_names::kSentBytesTotal, live_labels, 150);

  // Calling CollectMetricFamilies() should trigger dead worker reaping.
  auto families = live_exp.CollectMetricFamilies();
  bool found_sent = false;

  for (const auto& fam : families) {
    if (fam.name == "tpu_raiden_sent_bytes_total") {
      found_sent = true;
      for (const auto& m : fam.metric) {
        if (!m.label.empty() && m.label[0].value == "push") {
          // Expected: Live (150); dead worker unlinked.
          EXPECT_DOUBLE_EQ(m.counter.value, 150.0);
        }
      }
    }
  }

  EXPECT_TRUE(found_sent);

  // Text snapshot format reflects live worker total.
  std::string snapshot = live_exp.GetTextSnapshot();
  EXPECT_THAT(snapshot,
              HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 150"));
}

TEST_F(PrometheusShmExporterChallengerTest,
       SequentialDeadWorkerCascadesUnlinkedOnSweep) {
  ExporterOptions survivor_opt;
  survivor_opt.base_shm_dir = test_dir_;
  survivor_opt.local_rank = "0";
  survivor_opt.port = 0;

  PrometheusShmExporter survivor(survivor_opt);
  const MetricLabel labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPush}};

  // 5 workers start, record 100, and die sequentially.
  for (int i = 1; i <= 5; ++i) {
    std::string dead_file;
    {
      ExporterOptions dead_opt;
      dead_opt.base_shm_dir = test_dir_;
      dead_opt.local_rank = absl::StrCat(i);
      dead_opt.port = 0;
      PrometheusShmExporter dead_exp(dead_opt);
      dead_exp.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);
    }
    // Snapshot unlinks dead worker.
    std::string snap = survivor.GetTextSnapshot();
    survivor.IncrementCounter(metric_names::kSentBytesTotal, labels, 100);
    snap = survivor.GetTextSnapshot();
    EXPECT_THAT(snap, HasSubstr(absl::StrCat(
                          "tpu_raiden_sent_bytes_total{direction=\"push\"} ",
                          i * 100)));
  }
}

TEST_F(PrometheusShmExporterChallengerTest, DeadWorkerReapingOptionPreserved) {
  ExporterOptions live_opt;
  live_opt.base_shm_dir = test_dir_;
  live_opt.local_rank = "0";
  live_opt.port = 0;

  PrometheusShmExporter live_exp(live_opt);

  {
    ExporterOptions dead_opt;
    dead_opt.base_shm_dir = test_dir_;
    dead_opt.local_rank = "1";
    dead_opt.port = 0;
    PrometheusShmExporter dead_exp(dead_opt);
    const MetricLabel labels[] = {
        {metric_labels::kDirection, metric_labels::kDirectionPush}};
    dead_exp.IncrementCounter(metric_names::kSentBytesTotal, labels, 500);
  }

  // Dead worker is unlinked and excluded from live snapshot.
  std::string snapshot = live_exp.GetTextSnapshot();
  EXPECT_THAT(
      snapshot,
      Not(HasSubstr("tpu_raiden_sent_bytes_total{direction=\"push\"} 500")));
}

// ============================================================================
// CHALLENGE AREA 4: Histogram Bucket Ordering, Boundary Values & Formatting
// ============================================================================

TEST_F(PrometheusShmExporterChallengerTest,
       HistogramSlotCumulativeBucketOrderingAndBoundaries) {
  ShmHistogramSlot slot;
  EXPECT_EQ(slot.sample_count.load(), 0);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));

  // Default bucket boundaries:
  // [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
  //  750.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0, 25000.0, 50000.0]

  // Value 0.05 falls into bucket 0 (<= 0.1) and all cumulative buckets.
  slot.Observe(0.05);
  EXPECT_EQ(slot.sample_count.load(), 1);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.05));
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 1);
  }

  // Value 5.5 falls into bucket 6 (<= 10.0), skipping buckets 0..5.
  slot.Observe(5.5);
  EXPECT_EQ(slot.sample_count.load(), 2);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(5.55));
  for (size_t b = 0; b <= 5; ++b) EXPECT_EQ(slot.bucket_counts[b].load(), 1);
  for (size_t b = 6; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 2);
  }

  // Value 60000.0 exceeds highest boundary (50000.0), updating only +Inf bucket
  // (index 20).
  slot.Observe(60000.0);
  EXPECT_EQ(slot.sample_count.load(), 3);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(60005.55));
  for (size_t b = 0; b <= 5; ++b) EXPECT_EQ(slot.bucket_counts[b].load(), 1);
  for (size_t b = 6; b < kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 2);
  }
  EXPECT_EQ(slot.bucket_counts[kNumHistogramBuckets].load(), 3);

  // Monotonic cumulative invariant:
  for (size_t i = 0; i < kNumHistogramBuckets; ++i) {
    EXPECT_LE(slot.bucket_counts[i].load(), slot.bucket_counts[i + 1].load());
  }
}

TEST_F(PrometheusShmExporterChallengerTest,
       HistogramSlotExactBoundaryMatching) {
  ShmHistogramSlot slot;
  slot.Observe(0.1);         // Exactly on bucket 0 boundary
  slot.Observe(0.1000001);   // Just above bucket 0 -> bucket 1
  slot.Observe(50000.0);     // Exactly on bucket 19 boundary
  slot.Observe(50000.0001);  // Just above bucket 19 -> bucket 20 (+Inf)

  EXPECT_EQ(slot.sample_count.load(), 4);
  EXPECT_EQ(slot.bucket_counts[0].load(), 1);
  EXPECT_EQ(slot.bucket_counts[1].load(), 2);
  EXPECT_EQ(slot.bucket_counts[19].load(), 3);
  EXPECT_EQ(slot.bucket_counts[20].load(), 4);

  for (size_t i = 0; i < kNumHistogramBuckets; ++i) {
    EXPECT_LE(slot.bucket_counts[i].load(), slot.bucket_counts[i + 1].load());
  }
}

TEST_F(PrometheusShmExporterChallengerTest,
       HistogramSlotNonFiniteAndAdversarialValues) {
  ShmHistogramSlot slot;

  // 1. Non-finite values discarded
  slot.Observe(std::numeric_limits<double>::quiet_NaN());
  slot.Observe(std::numeric_limits<double>::signaling_NaN());
  slot.Observe(-std::numeric_limits<double>::quiet_NaN());
  slot.Observe(std::numeric_limits<double>::infinity());
  slot.Observe(-std::numeric_limits<double>::infinity());

  EXPECT_EQ(slot.sample_count.load(), 0);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 0);
  }

  // 2. Signed zeros
  slot.Observe(+0.0);
  slot.Observe(-0.0);
  EXPECT_EQ(slot.sample_count.load(), 2);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 2);
  }

  // 3. Subnormals
  slot.Observe(std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(slot.sample_count.load(), 3);
  EXPECT_GT(slot.sample_sum.load(), 0.0);

  // 4. Negative numbers
  slot.Observe(-50.0);
  EXPECT_EQ(slot.sample_count.load(), 4);
  EXPECT_LT(slot.sample_sum.load(), 0.0);
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 4);
  }
}

TEST_F(PrometheusShmExporterChallengerTest,
       HistogramSlotConcurrentContentionCasLoop) {
  ShmHistogramSlot slot;
  constexpr int kNumThreads = 16;
  constexpr int kNumIters = 25000;  // Total 400,000 observations
  std::atomic<bool> start_signal{false};

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&slot, &start_signal, t]() {
      while (!start_signal.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kNumIters; ++i) {
        double val = ((t * 17 + i) % 20) * 2.5 + 0.05;
        slot.Observe(val);
      }
    });
  }

  start_signal.store(true, std::memory_order_release);
  for (auto& th : threads) th.join();

  EXPECT_EQ(slot.sample_count.load(),
            static_cast<uint64_t>(kNumThreads) * kNumIters);
  EXPECT_EQ(slot.bucket_counts[kNumHistogramBuckets].load(),
            static_cast<uint64_t>(kNumThreads) * kNumIters);

  // Monotonic cumulative invariant:
  for (size_t i = 0; i < kNumHistogramBuckets; ++i) {
    EXPECT_LE(slot.bucket_counts[i].load(), slot.bucket_counts[i + 1].load());
  }
}

TEST_F(PrometheusShmExporterChallengerTest,
       HistogramDeadWorkerUnlinkedOnSweep) {
  ShmCollectorOptions collector_opts;
  collector_opts.shm_dir = test_dir_;
  ShmCollector collector(collector_opts);

  // Step 2: Create a dead worker file with a histogram slot.
  std::string worker_path = absl::StrCat(test_dir_, "/worker_9999_hist.mmap");
  int fd =
      open(worker_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, kSegmentTotalFileSize), 0);

  void* addr = mmap(nullptr, kSegmentTotalFileSize, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
  ASSERT_NE(addr, MAP_FAILED);

  auto* seg = reinterpret_cast<ShmSegmentLayout*>(addr);
  std::memset(seg, 0, kSegmentTotalFileSize);
  ShmTocHeader& h = seg->header;
  h.version = kSupportedVersion;
  h.pid = 9999;
  h.max_toc_entries = kMaxTocEntries;
  h.data_pool_offset = sizeof(ShmSegmentLayout);

  ShmTocEntry& e = seg->toc[h.toc_entry_count++];
  snprintf(e.metric_name, sizeof(e.metric_name), "sent_bytes_total");
  snprintf(e.encoded_labels, sizeof(e.encoded_labels), "direction=push");
  e.type = MetricType::kCounter;
  e.offset = h.data_pool_offset + h.data_pool_bytes;
  e.size = sizeof(std::atomic<uint64_t>);
  e.entry_state.store(TocEntryState::kCommitted);
  uint8_t* raw = reinterpret_cast<uint8_t*>(seg) + e.offset;
  new (raw) std::atomic<uint64_t>(12345);
  h.data_pool_bytes += e.size;

  h.magic.store(kRaidenShmMagic, std::memory_order_release);
  munmap(addr, kSegmentTotalFileSize);
  close(fd);

  // Sweep unlinks dead worker
  absl::flat_hash_map<std::string, double> totals;
  collector.CollectMetrics(totals);

  EXPECT_EQ(totals["sent_bytes_total/direction=push"], 0.0);
  EXPECT_FALSE(std::filesystem::exists(worker_path));
}

}  // namespace
}  // namespace tpu_raiden::telemetry

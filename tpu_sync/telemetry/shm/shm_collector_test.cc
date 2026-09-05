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

#include "tpu_sync/telemetry/shm/shm_collector.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/inotify.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::Contains;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Key;
using ::testing::Not;
using ::testing::Pair;
using ::testing::SizeIs;

constexpr MetricLabel kPush{metric_labels::kDirection,
                            metric_labels::kDirectionPush};
constexpr MetricLabel kPull{metric_labels::kDirection,
                            metric_labels::kDirectionPull};
constexpr std::array<MetricLabel, 1> kPushLabels = {kPush};
constexpr std::array<MetricLabel, 1> kPullLabels = {kPull};
constexpr absl::string_view kPushKey = "sent_bytes_total/direction=push";
// Number of metrics emitted per histogram: count, sum, and the
// overflow bucket (+1 over kNumHistogramBuckets).
constexpr size_t kMetricsPerHistogram = kNumHistogramBuckets + 3;

class ShmCollectorTest : public testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = absl::StrCat(testing::TempDir(), "shm_col_", getpid(), "_",
                             reinterpret_cast<uintptr_t>(this));
    std::error_code ec;
    std::filesystem::create_directories(test_dir_, ec);
    ASSERT_FALSE(ec);
  }

  void TearDown() override {
    for (int fd : held_fds_) {
      if (fd >= 0) close(fd);
    }
    held_fds_.clear();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  ShmWriterOptions WriterOptions(absl::string_view rank) const {
    return {.shm_dir = test_dir_, .local_rank = std::string(rank)};
  }

  ShmCollectorOptions CollectorOptions() const {
    return {.shm_dir = test_dir_};
  }

  std::string FilePath(absl::string_view filename) const {
    return absl::StrCat(test_dir_, "/", filename);
  }

  std::string ShmName(absl::string_view name,
                      absl::string_view ext = kShmFileExtension) const {
    return absl::StrCat(kShmFilePrefix, name, ext);
  }

  absl::flat_hash_map<std::string, double> Collect(
      absl::string_view sub_dir = "") const {
    absl::flat_hash_map<std::string, double> totals;
    const std::string dir = sub_dir.empty() ? test_dir_ : FilePath(sub_dir);
    ShmCollector(ShmCollectorOptions{.shm_dir = dir}).CollectMetrics(totals);
    return totals;
  }

  void CollectInto(absl::flat_hash_map<std::string, double>& totals) const {
    ShmCollector(CollectorOptions()).CollectMetrics(totals);
  }

  void ExpectMetrics(const absl::flat_hash_map<std::string, double>& totals,
                     std::initializer_list<std::pair<absl::string_view, double>>
                         expected) const {
    for (const auto& [key, value] : expected) {
      EXPECT_THAT(totals, Contains(Pair(key, DoubleEq(value))));
    }
  }

  void ExpectKeysAbsent(const absl::flat_hash_map<std::string, double>& totals,
                        std::initializer_list<absl::string_view> keys) const {
    for (absl::string_view key : keys) {
      EXPECT_THAT(totals, Not(Contains(Key(key))));
    }
  }

  void WriteFile(absl::string_view filename, absl::string_view data = "",
                 size_t file_size = 0) {
    const std::string path = FilePath(filename);
    const int fd =
        open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    ASSERT_GE(fd, 0);
    absl::Cleanup close_fd = [fd] { close(fd); };
    if (!data.empty()) {
      ASSERT_EQ(write(fd, data.data(), data.size()),
                static_cast<ssize_t>(data.size()));
    }
    if (file_size > data.size()) {
      ASSERT_EQ(ftruncate(fd, file_size), 0);
    }
  }

  int HoldSharedLock(absl::string_view filename) {
    const std::string path = FilePath(filename);
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_SH | LOCK_NB) != 0) {
      close(fd);
      return -1;
    }
    held_fds_.push_back(fd);
    return fd;
  }

  int WriteAndLockFile(absl::string_view filename, absl::string_view data = "",
                       size_t file_size = 0) {
    WriteFile(filename, data, file_size);
    return HoldSharedLock(filename);
  }

  void ReleaseHeldLock(int fd) {
    auto it = std::find(held_fds_.begin(), held_fds_.end(), fd);
    if (it != held_fds_.end()) {
      close(*it);
      held_fds_.erase(it);
    }
  }

  bool FileExists(absl::string_view filename) const {
    std::error_code ec;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(FilePath(filename), ec);
    return !ec && std::filesystem::status_known(status) &&
           status.type() != std::filesystem::file_type::not_found;
  }

  int CountMatchingFiles(absl::string_view substring) const {
    int count = 0;
    std::error_code ec;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(test_dir_, ec)) {
      if (absl::StrContains(entry.path().filename().string(), substring)) {
        ++count;
      }
    }
    return count;
  }

  template <typename F>
  void CreateDeadWorkerSegment(absl::string_view rank, F&& populate) {
    std::array<int, 2> pipe_fds;
    ASSERT_EQ(pipe2(pipe_fds.data(), O_CLOEXEC), 0);
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      close(pipe_fds[0]);
      // Allocate writer on heap without deleting so ~ShmWriter() does not
      // unlink the segment file, simulating an abrupt process termination.
      auto* writer = new ShmWriter(WriterOptions(rank));
      populate(*writer);
      char ready = 'R';
      if (write(pipe_fds[1], &ready, 1) != 1) {
        _exit(1);
      }
      close(pipe_fds[1]);
      _exit(0);
    }
    close(pipe_fds[1]);
    absl::Cleanup cleanup_parent = [pipe_fd = pipe_fds[0], pid] {
      close(pipe_fd);
      kill(pid, SIGKILL);
      int status = 0;
      waitpid(pid, &status, 0);
    };
    char sync = 0;
    ASSERT_EQ(read(pipe_fds[0], &sync, 1), 1);
    std::move(cleanup_parent).Cancel();
    close(pipe_fds[0]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
  }

  std::string test_dir_;
  std::vector<int> held_fds_;
};

using ShmCollectorDeathTest = ShmCollectorTest;

// Lightweight helper to build memory-mapped segments for unit testing.
struct MockSegment {
  struct alignas(64) AlignedSegment {
    alignas(64) char data[kSegmentTotalFileSize];
  };

  std::unique_ptr<AlignedSegment> storage;
  ShmSegmentLayout* layout;
  uint32_t next_offset = sizeof(ShmSegmentLayout);

  MockSegment()
      : storage(std::make_unique<AlignedSegment>()),
        layout(reinterpret_cast<ShmSegmentLayout*>(storage->data)) {
    std::memset(storage->data, 0, sizeof(storage->data));
    layout->header.magic.store(kRaidenShmMagic);
    layout->header.max_toc_entries = kMaxTocEntries;
    layout->header.data_pool_offset = sizeof(ShmSegmentLayout);
  }

  uint32_t Alloc(uint32_t size, uint32_t align = 64) {
    next_offset = (next_offset + align - 1) & ~(align - 1);
    const uint32_t offset = next_offset;
    next_offset += size;
    return offset;
  }

  ShmTocEntry& AddEntry(absl::string_view name, MetricType type,
                        std::optional<uint32_t> offset = std::nullopt,
                        uint32_t size = 8, absl::string_view labels = "",
                        TocEntryState state = TocEntryState::kCommitted) {
    const size_t idx =
        layout->header.toc_entry_count.load(std::memory_order_relaxed);
    ShmTocEntry& entry = layout->toc[idx];
    entry.type = type;
    entry.offset = offset.has_value() ? *offset : Alloc(size);
    entry.size = size;
    SetBounded(entry.metric_name, name);
    SetBounded(entry.encoded_labels, labels);
    entry.entry_state.store(state, std::memory_order_release);
    layout->header.toc_entry_count.store(idx + 1, std::memory_order_release);
    return entry;
  }

  template <typename T>
  void StoreValue(uint32_t offset, T value) {
    if (offset % alignof(T) == 0 &&
        offset + sizeof(T) <= sizeof(storage->data)) {
      ::new (static_cast<void*>(storage->data + offset)) std::atomic<T>(value);
    }
  }

  void AddCounter(absl::string_view name, uint64_t value,
                  absl::string_view labels = "",
                  TocEntryState state = TocEntryState::kCommitted,
                  std::optional<uint32_t> offset = std::nullopt,
                  uint32_t size = sizeof(std::atomic<uint64_t>)) {
    const ShmTocEntry& entry =
        AddEntry(name, MetricType::kCounter, offset, size, labels, state);
    StoreValue<uint64_t>(entry.offset, value);
  }

  void AddGauge(absl::string_view name, double value,
                absl::string_view labels = "",
                TocEntryState state = TocEntryState::kCommitted,
                std::optional<uint32_t> offset = std::nullopt,
                uint32_t size = sizeof(std::atomic<double>)) {
    const ShmTocEntry& entry =
        AddEntry(name, MetricType::kGauge, offset, size, labels, state);
    StoreValue<double>(entry.offset, value);
  }

  void AddHistogram(absl::string_view name, double sum, uint64_t count,
                    absl::string_view labels = "",
                    std::optional<uint32_t> offset = std::nullopt) {
    const ShmTocEntry& entry = AddEntry(name, MetricType::kHistogram, offset,
                                        sizeof(ShmHistogramSlot), labels);
    if (entry.offset % alignof(ShmHistogramSlot) == 0 &&
        entry.offset + sizeof(ShmHistogramSlot) <= sizeof(storage->data)) {
      auto* slot = ::new (static_cast<void*>(storage->data + entry.offset))
          ShmHistogramSlot();
      slot->sample_sum.store(sum, std::memory_order_relaxed);
      slot->sample_count.store(count, std::memory_order_relaxed);
    }
  }

  absl::string_view data() const {
    return absl::string_view(storage->data, sizeof(storage->data));
  }

 private:
  template <size_t N>
  static void SetBounded(char (&dest)[N], absl::string_view src) {
    std::memset(dest, 0, N);
    if (!src.empty()) {
      std::memcpy(dest, src.data(), std::min(src.size(), N - 1));
    }
  }
};

TEST_F(ShmCollectorDeathTest, RejectsEmptyOrUnsetShmDir) {
  EXPECT_DEATH(ShmCollector(ShmCollectorOptions{}),
               "ShmCollector requires a non-empty shm_dir");
  EXPECT_DEATH(ShmCollector(ShmCollectorOptions{.shm_dir = ""}),
               "ShmCollector requires a non-empty shm_dir");
}

TEST_F(ShmCollectorTest, HandlesNonExistentShmDir) {
  EXPECT_THAT(Collect("nonexistent"), IsEmpty());
}

TEST_F(ShmCollectorTest, AggregatesLiveWorkersAndReapsDead) {
  ASSERT_NO_FATAL_FAILURE(
      CreateDeadWorkerSegment("dead", [](const ShmWriter& dead_writer) {
        dead_writer.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels,
                                     500);
        dead_writer.SetGauge(metric_names::kBufferAllocatedBytes, {}, 512.0);
      }));

  ShmWriter writer_0(WriterOptions("0"));
  ShmWriter writer_1(WriterOptions("1"));
  const std::array<MetricLabel, 2> failure_labels = {
      kPull, MetricLabel{metric_labels::kErrorCode, "DEADLINE_EXCEEDED"}};
  const std::array<MetricLabel, 2> hist_labels = {
      kPush, MetricLabel{metric_labels::kErrorCode, "RESOURCE_EXHAUSTED"}};

  writer_0.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels, 100);
  writer_1.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels, 200);
  writer_0.IncrementCounter(metric_names::kSentBytesTotal, kPullLabels, 50);
  writer_1.IncrementCounter(metric_names::kSentBytesTotal, kPullLabels, 75);
  writer_1.IncrementCounter(metric_names::kTransferFailuresTotal,
                            failure_labels, 5);
  writer_0.SetGauge(metric_names::kBufferAllocatedBytes, {}, 1000.0);
  writer_1.SetGauge(metric_names::kBufferAllocatedBytes, {}, 2000.0);
  writer_0.ObserveHistogram(metric_names::kTransferDurationMs, {}, 0.05);
  writer_1.ObserveHistogram(metric_names::kTransferDurationMs, {}, 0.15);
  writer_1.ObserveHistogram(metric_names::kTransferDurationMs, hist_labels,
                            0.25);

  absl::flat_hash_map<std::string, double> totals = Collect();
  constexpr absl::string_view kFailureKey =
      "transfer_failures_total/direction=pull;error_code=DEADLINE_EXCEEDED";
  constexpr absl::string_view kHistCountKey =
      "transfer_duration_ms_count/direction=push;error_code=RESOURCE_EXHAUSTED";
  constexpr absl::string_view kHistSumKey =
      "transfer_duration_ms_sum/direction=push;error_code=RESOURCE_EXHAUSTED";
  constexpr absl::string_view kHistBucketPrefix =
      "transfer_duration_ms_bucket/"
      "direction=push;error_code=RESOURCE_EXHAUSTED";
  // 4 scalar metrics (push, pull, failure, buffer) + 2 histograms * 23 metrics
  // = 50.
  EXPECT_THAT(totals, SizeIs(50));
  ExpectMetrics(totals, {
                            {kPushKey, 300.0},
                            {"sent_bytes_total/direction=pull", 125.0},
                            {kFailureKey, 5.0},
                            {"buffer_allocated_bytes", 3000.0},
                            {"transfer_duration_ms_count", 2.0},
                            {"transfer_duration_ms_sum", 0.20},
                            {"transfer_duration_ms_bucket/le=0.1", 1.0},
                            {"transfer_duration_ms_bucket/le=0.25", 2.0},
                            {kHistCountKey, 1.0},
                            {kHistSumKey, 0.25},
                            {absl::StrCat(kHistBucketPrefix, ";le=0.25"), 1.0},
                        });

  EXPECT_EQ(CountMatchingFiles("worker_rank_dead_"), 0);
  EXPECT_EQ(CountMatchingFiles("worker_rank_0_"), 1);
  EXPECT_EQ(CountMatchingFiles("worker_rank_1_"), 1);
}

TEST_F(ShmCollectorTest, ConcurrencyAndProcessCrashHandling) {
  ASSERT_NO_FATAL_FAILURE(
      CreateDeadWorkerSegment("crashed", [](const ShmWriter& writer) {
        writer.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels,
                                777);
      }));
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(CreateDeadWorkerSegment(
        absl::StrCat("dead_", i), [](const ShmWriter& writer) {
          writer.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
        }));
  }
  ShmWriter live(WriterOptions("live"));
  live.IncrementCounter(metric_names::kSentBytesTotal, {}, 42);

  absl::Notification start_notification;
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([this, &start_notification]() {
      start_notification.WaitForNotification();
      absl::flat_hash_map<std::string, double> thread_totals;
      ShmCollector(CollectorOptions()).CollectMetrics(thread_totals);
      EXPECT_THAT(thread_totals, Not(Contains(Key(kPushKey))));
      EXPECT_THAT(thread_totals,
                  Contains(Pair("sent_bytes_total", DoubleEq(42.0))));
    });
  }
  start_notification.Notify();
  for (std::thread& thread : threads) {
    thread.join();
  }

  absl::flat_hash_map<std::string, double> final_totals = Collect();
  EXPECT_THAT(final_totals, Contains(Pair("sent_bytes_total", DoubleEq(42.0))));
  ExpectKeysAbsent(final_totals, {kPushKey});
  EXPECT_EQ(CountMatchingFiles("crashed"), 0);
  EXPECT_EQ(CountMatchingFiles("dead_"), 0);
  EXPECT_EQ(CountMatchingFiles("live"), 1);
}

TEST_F(ShmCollectorTest, AggregatesMultiChunkWorker) {
  ShmWriter writer(WriterOptions("multi"));
  for (size_t i = 0; i < kMaxTocEntries + 20; ++i) {
    writer.IncrementCounter(absl::StrCat("metric_", i), {}, 1);
  }

  absl::flat_hash_map<std::string, double> totals = Collect();
  EXPECT_THAT(totals, SizeIs(kMaxTocEntries + 20));
  ExpectMetrics(totals, {
                            {"metric_0", 1.0},
                            {absl::StrCat("metric_", kMaxTocEntries - 1), 1.0},
                            {absl::StrCat("metric_", kMaxTocEntries), 1.0},
                            {absl::StrCat("metric_", kMaxTocEntries + 19), 1.0},
                        });
}

TEST_F(ShmCollectorTest, AggregatesMultiChunkHistogramsOnPoolExhaustion) {
  constexpr size_t kNumHistograms = 350;
  ShmWriter writer(WriterOptions("multi_hist"));
  for (size_t i = 0; i < kNumHistograms; ++i) {
    writer.ObserveHistogram(absl::StrCat("hist_", i), {}, 0.05);
  }

  EXPECT_GE(CountMatchingFiles("multi_hist"), 2);
  absl::flat_hash_map<std::string, double> totals = Collect();
  EXPECT_THAT(totals, SizeIs(kNumHistograms * kMetricsPerHistogram));
  ExpectMetrics(totals,
                {
                    {"hist_0_count", 1.0},
                    {"hist_0_bucket/le=0.1", 1.0},
                    {absl::StrCat("hist_", kNumHistograms - 1, "_count"), 1.0},
                });
}

TEST_F(ShmCollectorTest, ReapsAllChunksOfMultiChunkDeadWorker) {
  ASSERT_NO_FATAL_FAILURE(
      CreateDeadWorkerSegment("dead_multi", [](const ShmWriter& writer) {
        for (size_t i = 0; i < kMaxTocEntries + 20; ++i) {
          writer.IncrementCounter(absl::StrCat("dead_metric_", i), {}, 1);
        }
      }));

  ASSERT_GE(CountMatchingFiles("dead_multi"), 2);
  EXPECT_THAT(Collect(), IsEmpty());
  EXPECT_EQ(CountMatchingFiles("dead_multi"), 0);
}

TEST_F(ShmCollectorTest, ClearsPrePopulatedTotalsMap) {
  ShmWriter live(WriterOptions("live"));
  live.IncrementCounter(metric_names::kSentBytesTotal, {}, 10);

  absl::flat_hash_map<std::string, double> totals = {{"stale_a", 999.0},
                                                     {"stale_b", 123.0}};
  CollectInto(totals);
  EXPECT_THAT(totals, ElementsAre(Pair("sent_bytes_total", DoubleEq(10.0))));
}

TEST_F(ShmCollectorTest, SkipsDefensiveSlotValidationFailures) {
  constexpr uint32_t kBase = sizeof(ShmSegmentLayout);
  MockSegment mock;

  mock.AddCounter("valid_counter", 100);
  // 1. Unaligned slot pointer (offset + 1 is not 8-byte aligned)
  mock.AddCounter("unaligned_counter", 0, "", TocEntryState::kCommitted,
                  kBase + 65);
  // 2. TOC offset below data_pool_offset
  mock.AddCounter("below_pool_offset", 0, "", TocEntryState::kCommitted,
                  kBase - 64);
  // 3. TOC offset + size exceeds kSegmentTotalFileSize
  mock.AddCounter("exceeds_filesize", 0, "", TocEntryState::kCommitted,
                  kSegmentTotalFileSize - 4);
  // 3b. 32-bit unsigned offset wraparound overflow
  mock.AddCounter("overflow_offset", 0, "", TocEntryState::kCommitted,
                  0xFFFFFFC0, 64);
  // 4. Non-finite gauge value (NaN)
  mock.AddGauge("nan_gauge", std::numeric_limits<double>::quiet_NaN());
  // 5. Non-finite gauge value (Inf)
  mock.AddGauge("inf_gauge", std::numeric_limits<double>::infinity());
  // 6. Non-finite histogram sample sum (Inf)
  mock.AddHistogram("inf_hist", std::numeric_limits<double>::infinity(), 5);
  // 7. Undersized slot descriptor for Counter (< 8 bytes)
  mock.AddCounter("undersized_counter", 0, "", TocEntryState::kCommitted,
                  std::nullopt, 4);
  // 8. Histogram with 0 observations
  mock.AddHistogram("zero_hist", 0.0, 0);
  // 9. Non-null-terminated metric name filling all 64 bytes
  ShmTocEntry& bad_name = mock.AddEntry("", MetricType::kCounter, std::nullopt,
                                        8, "", TocEntryState::kWriting);
  std::memset(bad_name.metric_name, 'x', sizeof(bad_name.metric_name));
  bad_name.entry_state.store(TocEntryState::kCommitted,
                             std::memory_order_release);
  // 10. Non-null-terminated encoded labels filling all 128 bytes
  ShmTocEntry& bad_labels =
      mock.AddEntry("valid_name_bad_labels", MetricType::kCounter, std::nullopt,
                    8, "", TocEntryState::kWriting);
  std::memset(bad_labels.encoded_labels, 'y',
              sizeof(bad_labels.encoded_labels));
  bad_labels.entry_state.store(TocEntryState::kCommitted,
                               std::memory_order_release);
  mock.AddCounter("uninit_metric", 0, "", TocEntryState::kUninitialized);
  mock.AddCounter("writing_metric", 0, "", TocEntryState::kWriting);

  ASSERT_GE(WriteAndLockFile(ShmName("defensive_slots"), mock.data()), 0);

  absl::flat_hash_map<std::string, double> totals = Collect();
  constexpr size_t kExpectedDefensiveMetricsCount = 1 + kMetricsPerHistogram;
  EXPECT_THAT(totals, SizeIs(kExpectedDefensiveMetricsCount));
  ExpectMetrics(totals, {
                            {"valid_counter", 100.0},
                            {"zero_hist_count", 0.0},
                            {"zero_hist_sum", 0.0},
                            {"zero_hist_bucket/le=0.1", 0.0},
                        });
  ExpectKeysAbsent(
      totals, {"unaligned_counter", "below_pool_offset", "exceeds_filesize",
               "overflow_offset", "nan_gauge", "inf_gauge", "inf_hist_count",
               "undersized_counter", "valid_name_bad_labels", "uninit_metric",
               "writing_metric"});
  EXPECT_THAT(
      totals,
      Not(Contains(Key(std::string(sizeof(bad_name.metric_name), 'x')))));
}

TEST_F(ShmCollectorTest, ResilientToCorruptFiles) {
  ShmWriter live(WriterOptions("good"));
  live.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels, 500);

  WriteFile(ShmName("zero"));
  WriteFile(ShmName("truncated"), "", sizeof(ShmSegmentLayout) + 128);
  EXPECT_GE(WriteAndLockFile(ShmName("live_truncated"), "short"), 0);

  auto write_bad_header = [&](absl::string_view name, uint32_t magic,
                              uint32_t max_toc, uint32_t pool_offset,
                              uint32_t toc_count = 0) {
    ShmTocHeader header{};
    header.magic.store(magic);
    header.max_toc_entries = max_toc;
    header.data_pool_offset = pool_offset;
    header.toc_entry_count.store(toc_count);
    const absl::string_view data(reinterpret_cast<const char*>(&header),
                                 sizeof(header));
    WriteFile(ShmName(name), data, kSegmentTotalFileSize);
    EXPECT_GE(WriteAndLockFile(ShmName(absl::StrCat("live_", name)), data,
                               kSegmentTotalFileSize),
              0);
  };

  constexpr uint32_t kInvalidMagic = 0xDEADBEEF;
  write_bad_header("bad_magic", kInvalidMagic, kMaxTocEntries,
                   sizeof(ShmSegmentLayout));
  write_bad_header("bad_max_toc", kRaidenShmMagic, kMaxTocEntries + 1,
                   sizeof(ShmSegmentLayout));
  write_bad_header("bad_offset", kRaidenShmMagic, kMaxTocEntries,
                   kSegmentTotalFileSize);
  write_bad_header("bad_offset_low", kRaidenShmMagic, kMaxTocEntries,
                   sizeof(ShmSegmentLayout) - 1);
  write_bad_header("bad_offset_unaligned", kRaidenShmMagic, kMaxTocEntries,
                   sizeof(ShmSegmentLayout) + 1);
  write_bad_header("bad_count", kRaidenShmMagic, kMaxTocEntries,
                   sizeof(ShmSegmentLayout), kMaxTocEntries + 1);

  WriteFile("other_file.txt", "non-shm content");
  std::error_code ec;
  std::filesystem::create_directories(FilePath(ShmName("dir")), ec);
  ASSERT_FALSE(ec);
  ASSERT_EQ(symlink("/dev/null", FilePath(ShmName("symlink")).c_str()), 0);

  const std::string unreadable = ShmName("unreadable");
  WriteFile(unreadable, "", kSegmentTotalFileSize);
  const std::string unreadable_path = FilePath(unreadable);
  ASSERT_EQ(chmod(unreadable_path.c_str(), 0000), 0);
  absl::Cleanup restore_permissions = [unreadable_path] {
    chmod(unreadable_path.c_str(), 0644);
  };

  EXPECT_THAT(Collect(), ElementsAre(Pair(kPushKey, DoubleEq(500.0))));

  EXPECT_TRUE(FileExists("other_file.txt"));
  EXPECT_TRUE(FileExists(ShmName("dir")));
  EXPECT_TRUE(FileExists(ShmName("symlink")));
  EXPECT_TRUE(FileExists(unreadable));

  for (absl::string_view name :
       {"zero", "truncated", "bad_magic", "bad_max_toc", "bad_offset",
        "bad_offset_low", "bad_offset_unaligned", "bad_count"}) {
    SCOPED_TRACE(absl::StrCat("Checking deletion of corrupt file: ", name));
    EXPECT_FALSE(FileExists(ShmName(name)));
  }
  for (absl::string_view name :
       {"bad_magic", "bad_max_toc", "bad_offset", "bad_offset_low",
        "bad_offset_unaligned", "bad_count"}) {
    SCOPED_TRACE(absl::StrCat("Checking preservation of live file: ", name));
    EXPECT_TRUE(FileExists(ShmName(absl::StrCat("live_", name))));
  }
  EXPECT_TRUE(FileExists(ShmName("live_truncated")));
}

TEST_F(ShmCollectorTest, SkipsFifoWithoutBlocking) {
  const std::string fifo = ShmName("fifo");
  ASSERT_EQ(mkfifo(FilePath(fifo).c_str(), 0644), 0);

  ShmWriter live(WriterOptions("live"));
  live.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels, 100);

  EXPECT_THAT(Collect(), ElementsAre(Pair(kPushKey, DoubleEq(100.0))));
  EXPECT_TRUE(FileExists(fifo));
}

TEST_F(ShmCollectorTest, ReapsOrphanedTmpFilesAndPreservesActiveTmpFiles) {
  ShmWriter live(WriterOptions("live"));
  live.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels, 100);

  const std::string dead_chunk = ShmName("dead_chunk_0", kShmTmpFileExtension);
  const std::string dead_plain = ShmName("dead_plain", ".tmp");
  const std::string active_chunk =
      ShmName("active_chunk_0", kShmTmpFileExtension);

  WriteFile(dead_chunk, "orphaned chunk tmp");
  WriteFile(dead_plain, "orphaned plain tmp");

  MockSegment mock;
  mock.AddCounter("tmp_uncommitted_counter", 999);
  const int active_fd = WriteAndLockFile(active_chunk, mock.data());
  ASSERT_GE(active_fd, 0);

  EXPECT_THAT(Collect(), ElementsAre(Pair(kPushKey, DoubleEq(100.0))));
  EXPECT_FALSE(FileExists(dead_chunk));
  EXPECT_TRUE(FileExists(dead_plain));
  EXPECT_TRUE(FileExists(active_chunk));

  ReleaseHeldLock(active_fd);

  EXPECT_THAT(Collect(), ElementsAre(Pair(kPushKey, DoubleEq(100.0))));
  EXPECT_FALSE(FileExists(active_chunk));
  EXPECT_TRUE(FileExists(dead_plain));
}

#if defined(__linux__)
struct UnlinkContext {
  static inline std::atomic<const char*> path{nullptr};
  static inline std::atomic<int> invoked{0};

  static void HandleSigio(int /*signum*/) {
    const char* target = path.exchange(nullptr, std::memory_order_acq_rel);
    if (target != nullptr) {
      unlink(target);
      invoked.store(1, std::memory_order_release);
    }
  }
};

TEST_F(ShmCollectorTest, SkipsUnlinkedSegmentUnderSharedLock) {
  // Execute in an isolated child subprocess so asynchronous signals and
  // handler registrations never leak into the test harness or interfere with
  // thread sanitizers and test watchdog threads.
  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    const std::string unlinked_file = ShmName("unlinked");
    const std::string path = FilePath(unlinked_file);

    MockSegment mock;
    mock.AddCounter("unlinked_counter", 999);
    const int holder_fd = WriteAndLockFile(unlinked_file, mock.data());
    if (holder_fd < 0) {
      _exit(1);
    }

    ShmWriter live(WriterOptions("live"));
    live.IncrementCounter(metric_names::kSentBytesTotal, kPushLabels, 100);

    UnlinkContext::path.store(path.c_str(), std::memory_order_release);
    UnlinkContext::invoked.store(0, std::memory_order_release);

    struct sigaction sa{};
    sa.sa_handler = UnlinkContext::HandleSigio;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGIO, &sa, nullptr) != 0) {
      _exit(2);
    }

    const int inotify_fd = inotify_init1(IN_CLOEXEC);
    if (inotify_fd < 0) {
      _exit(3);
    }
    if (inotify_add_watch(inotify_fd, path.c_str(), IN_OPEN) < 0) {
      _exit(4);
    }
    if (fcntl(inotify_fd, F_SETOWN, getpid()) != 0) {
      _exit(5);
    }
    const int flags = fcntl(inotify_fd, F_GETFL);
    if (flags < 0 || fcntl(inotify_fd, F_SETFL, flags | O_ASYNC) != 0) {
      _exit(6);
    }

    const auto totals = Collect();
    const bool correct_metrics =
        totals.size() == 1 && totals.contains(kPushKey) &&
        std::abs(totals.at(kPushKey) - 100.0) < 1e-6;
    const bool unlinked =
        UnlinkContext::invoked.load(std::memory_order_acquire) == 1;

    struct stat holder_stat{};
    const bool link_zero =
        fstat(holder_fd, &holder_stat) == 0 && holder_stat.st_nlink == 0;

    close(inotify_fd);
    close(holder_fd);

    _exit((correct_metrics && unlinked && link_zero) ? 0 : 7);
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
#endif

}  // namespace
}  // namespace tpu_raiden::telemetry

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
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <memory>
#include <new>
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
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"
#include "tpu_sync/telemetry/shm/shm_writer.h"

namespace tpu_raiden::telemetry {

class ShmCollectorTest : public testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = absl::StrCat(testing::TempDir(), "/shm_col_", getpid(), "_",
                             reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(test_dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }
  ShmWriterOptions WriterOptions(absl::string_view rank) const {
    return {.shm_dir = test_dir_, .local_rank = std::string(rank)};
  }
  ShmCollectorOptions CollectorOptions() const {
    return {.shm_dir = test_dir_};
  }

  std::string test_dir_;
};

namespace {

using ::testing::DoubleEq;

TEST_F(ShmCollectorTest, RejectsEmptyShmDir) {
  EXPECT_DEATH(ShmCollector(ShmCollectorOptions{.shm_dir = ""}),
               "ShmCollector requires a non-empty shm_dir");
}

TEST_F(ShmCollectorTest, AggregatesLiveWorkersAndReapsDead) {
  // Dead worker segment created and destroyed before collection.
  {
    ShmWriter dead_writer(WriterOptions("dead"));
    MetricLabel l{metric_labels::kDirection, metric_labels::kDirectionPush};
    dead_writer.IncrementCounter(metric_names::kSentBytesTotal, {&l, 1}, 500);
    dead_writer.SetGauge(metric_names::kBufferAllocatedBytes, {}, 512.0);
  }

  // Active live workers.
  ShmWriter w0(WriterOptions("0")), w1(WriterOptions("1"));
  MetricLabel push{metric_labels::kDirection, metric_labels::kDirectionPush};
  MetricLabel pull{metric_labels::kDirection, metric_labels::kDirectionPull};
  const std::array<MetricLabel, 2> fail = {
      MetricLabel{metric_labels::kDirection, metric_labels::kDirectionPull},
      MetricLabel{metric_labels::kErrorCode, "DEADLINE_EXCEEDED"}};

  w0.IncrementCounter(metric_names::kSentBytesTotal, {&push, 1}, 100);
  w1.IncrementCounter(metric_names::kSentBytesTotal, {&push, 1}, 200);
  w0.IncrementCounter(metric_names::kSentBytesTotal, {&pull, 1}, 50);
  w1.IncrementCounter(metric_names::kSentBytesTotal, {&pull, 1}, 75);
  w1.IncrementCounter(metric_names::kTransferFailuresTotal,
                      {fail.data(), fail.size()}, 5);
  w0.SetGauge(metric_names::kBufferAllocatedBytes, {}, 1000.0);
  w1.SetGauge(metric_names::kBufferAllocatedBytes, {}, 2000.0);
  w0.ObserveHistogram(metric_names::kTransferDurationMs, {}, 0.05);
  w1.ObserveHistogram(metric_names::kTransferDurationMs, {}, 0.15);

  ShmCollector initial(CollectorOptions());
  ShmCollector collector(std::move(initial));
  absl::flat_hash_map<std::string, double> totals;
  collector.CollectMetrics(totals);

  EXPECT_THAT(totals["sent_bytes_total/direction=push"], DoubleEq(300.0));
  EXPECT_THAT(totals["sent_bytes_total/direction=pull"], DoubleEq(125.0));
  EXPECT_THAT(totals["transfer_failures_total/"
                     "direction=pull;error_code=DEADLINE_EXCEEDED"],
              DoubleEq(5.0));
  EXPECT_THAT(totals["buffer_allocated_bytes"], DoubleEq(3000.0));
  EXPECT_THAT(totals["transfer_duration_ms/count"], DoubleEq(2.0));
  EXPECT_THAT(totals["transfer_duration_ms/sum"], DoubleEq(0.20));
  EXPECT_THAT(totals["transfer_duration_ms/bucket_0"], DoubleEq(1.0));
  EXPECT_THAT(totals["transfer_duration_ms/bucket_1"], DoubleEq(1.0));

  // Dead worker file is reaped (unlinked) while live worker files remain.
  int dead_files = 0;
  int live_files = 0;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    const std::string fn = entry.path().filename().string();
    dead_files += absl::StrContains(fn, "worker_rank_dead_");
    live_files += absl::StrContains(fn, "worker_rank_0_") ||
                  absl::StrContains(fn, "worker_rank_1_");
  }
  EXPECT_EQ(dead_files, 0);
  EXPECT_EQ(live_files, 2);
}

TEST_F(ShmCollectorTest, ConcurrencyAndProcessCrashHandling) {
  int pfd[2];
  ASSERT_EQ(pipe(pfd), 0);
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {
    close(pfd[0]);
    ShmWriter child(WriterOptions("crashed"));
    MetricLabel l{metric_labels::kDirection, metric_labels::kDirectionPush};
    child.IncrementCounter(metric_names::kSentBytesTotal, {&l, 1}, 777);
    char ready = 'R';
    ASSERT_EQ(write(pfd[1], &ready, 1), 1);
    close(pfd[1]);
    while (true) pause();
  }

  close(pfd[1]);
  char sync = 0;
  ASSERT_EQ(read(pfd[0], &sync, 1), 1);
  close(pfd[0]);
  kill(pid, SIGKILL);
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);

  for (int i = 0; i < 3; ++i) {
    ShmWriter(WriterOptions(absl::StrCat("dead_", i)))
        .IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
  }
  ShmWriter live(WriterOptions("live"));
  live.IncrementCounter(metric_names::kSentBytesTotal, {}, 42);

  // Concurrent collector threads race to aggregate and reap dead files.
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([this]() {
      absl::flat_hash_map<std::string, double> t;
      ShmCollector(CollectorOptions()).CollectMetrics(t);
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }

  absl::flat_hash_map<std::string, double> final_totals;
  ShmCollector(CollectorOptions()).CollectMetrics(final_totals);
  EXPECT_THAT(final_totals["sent_bytes_total"], DoubleEq(42.0));
  EXPECT_FALSE(final_totals.contains("sent_bytes_total/direction=push"));

  int dead_files = 0;
  for (const auto& e : std::filesystem::directory_iterator(test_dir_)) {
    const std::string fn = e.path().filename().string();
    dead_files +=
        absl::StrContains(fn, "crashed") || absl::StrContains(fn, "dead_");
  }
  EXPECT_EQ(dead_files, 0);
}

TEST_F(ShmCollectorTest, AggregatesMultiChunkWorker) {
  ShmWriter writer(WriterOptions("multi"));
  for (size_t i = 0; i < kMaxTocEntries + 20; ++i) {
    writer.IncrementCounter(absl::StrCat("metric_", i), {}, 1);
  }

  absl::flat_hash_map<std::string, double> totals;
  ShmCollector(CollectorOptions()).CollectMetrics(totals);

  EXPECT_THAT(totals["metric_0"], DoubleEq(1.0));
  EXPECT_THAT(totals[absl::StrCat("metric_", kMaxTocEntries + 19)],
              DoubleEq(1.0));
}

TEST_F(ShmCollectorTest, AggregatesLabeledHistograms) {
  ShmWriter writer(WriterOptions("hist_worker"));
  MetricLabel push{metric_labels::kDirection, metric_labels::kDirectionPush};
  MetricLabel err{metric_labels::kErrorCode, "RESOURCE_EXHAUSTED"};
  const std::array<MetricLabel, 2> labels = {push, err};

  writer.ObserveHistogram(metric_names::kTransferDurationMs,
                          {labels.data(), labels.size()}, 0.05);
  writer.ObserveHistogram(metric_names::kTransferDurationMs,
                          {labels.data(), labels.size()}, 0.25);

  absl::flat_hash_map<std::string, double> totals;
  ShmCollector(CollectorOptions()).CollectMetrics(totals);

  std::string prefix =
      "transfer_duration_ms/direction=push;error_code=RESOURCE_EXHAUSTED";
  EXPECT_THAT(totals[absl::StrCat(prefix, "/count")], DoubleEq(2.0));
  EXPECT_THAT(totals[absl::StrCat(prefix, "/sum")], DoubleEq(0.30));
  EXPECT_THAT(totals[absl::StrCat(prefix, "/bucket_0")], DoubleEq(1.0));
  EXPECT_THAT(totals[absl::StrCat(prefix, "/bucket_1")], DoubleEq(1.0));
}

TEST_F(ShmCollectorTest, ReapsDeadTruncatedFiles) {
  const std::string trunc_path = absl::StrCat(test_dir_, "/", kShmFilePrefix,
                                              "dead_trunc", kShmFileExtension);
  {
    std::ofstream ofs(trunc_path, std::ios::binary);
    ofs.write("short", 5);
  }
  absl::flat_hash_map<std::string, double> totals;
  ShmCollector(CollectorOptions()).CollectMetrics(totals);

  EXPECT_FALSE(std::filesystem::exists(trunc_path));
}

TEST_F(ShmCollectorTest, SkipsLiveUndersizedFileWithoutReaping) {
  const std::string live_trunc_path = absl::StrCat(
      test_dir_, "/", kShmFilePrefix, "live_trunc", kShmFileExtension);
  int fd = open(live_trunc_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  ASSERT_GE(fd, 0);
  absl::Cleanup close_fd = [fd] { close(fd); };
  ASSERT_EQ(write(fd, "short", 5), 5);
  ASSERT_EQ(flock(fd, LOCK_SH | LOCK_NB), 0);

  absl::flat_hash_map<std::string, double> totals;
  ShmCollector(CollectorOptions()).CollectMetrics(totals);

  EXPECT_TRUE(totals.empty());
  EXPECT_TRUE(std::filesystem::exists(live_trunc_path));
}

TEST_F(ShmCollectorTest, ResilientToCorruptedFilesAndNonExistentDir) {
  // Non-existent directory handling.
  absl::flat_hash_map<std::string, double> missing_totals;
  ShmCollector({.shm_dir = absl::StrCat(test_dir_, "/missing")})
      .CollectMetrics(missing_totals);
  EXPECT_TRUE(missing_totals.empty());

  ShmWriter live_writer(WriterOptions("good"));
  MetricLabel push{metric_labels::kDirection, metric_labels::kDirectionPush};
  live_writer.IncrementCounter(metric_names::kSentBytesTotal, {&push, 1}, 500);

  auto write_file = [&](absl::string_view name, absl::string_view data,
                        size_t pad = 0) {
    std::ofstream ofs(absl::StrCat(test_dir_, "/", name), std::ios::binary);
    ofs.write(data.data(), data.size());
    if (pad > 0) ofs.write(std::string(pad, '\0').data(), pad);
  };
  auto write_hdr = [&](absl::string_view name, uint32_t magic) {
    ShmTocHeader h{};
    h.magic.store(magic);
    write_file(absl::StrCat(kShmFilePrefix, name),
               {reinterpret_cast<const char*>(&h), sizeof(h)},
               kSegmentTotalFileSize - sizeof(h));
  };

  write_file(absl::StrCat(kShmFilePrefix, "zero.mmap"), "");
  write_file(absl::StrCat(kShmFilePrefix, "short.mmap"), "short_hdr");
  write_file(absl::StrCat(kShmFilePrefix, "truncated.mmap"), "",
             sizeof(ShmSegmentLayout) + 128);
  write_hdr("bad_magic.mmap", 0xDEADBEEF);
  write_file("other_file.txt", "non-shm content");
  std::filesystem::create_directories(
      absl::StrCat(test_dir_, "/", kShmFilePrefix, "dir.mmap"));
  symlink("/dev/null",
          absl::StrCat(test_dir_, "/", kShmFilePrefix, "symlink.mmap").c_str());

  const std::string unreadable =
      absl::StrCat(test_dir_, "/", kShmFilePrefix, "unreadable.mmap");
  write_file(absl::StrCat(kShmFilePrefix, "unreadable.mmap"), "",
             kSegmentTotalFileSize);
  chmod(unreadable.c_str(), 0000);
  absl::Cleanup restore_perms = [&] { chmod(unreadable.c_str(), 0644); };

  absl::flat_hash_map<std::string, double> totals;
  ShmCollector(CollectorOptions()).CollectMetrics(totals);
  EXPECT_THAT(totals["sent_bytes_total/direction=push"], DoubleEq(500.0));
  EXPECT_TRUE(
      std::filesystem::exists(absl::StrCat(test_dir_, "/other_file.txt")));
}

TEST_F(ShmCollectorTest, DirectAggregateSegmentEdgeCases) {
  absl::flat_hash_map<std::string, double> totals;

  internal::AggregateSegment(nullptr, totals);
  EXPECT_TRUE(totals.empty());

  struct alignas(64) Buffer {
    uint8_t data[kSegmentTotalFileSize]{};
  };
  auto mem = std::make_unique<Buffer>();
  auto* seg = reinterpret_cast<ShmSegmentLayout*>(mem->data);
  seg->header.magic.store(kRaidenShmMagic, std::memory_order_release);
  seg->header.max_toc_entries = kMaxTocEntries;
  seg->header.data_pool_offset = sizeof(ShmSegmentLayout);

  // Unaligned segment pointer (safe to pass as const void*).
  const void* unaligned_seg =
      reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(seg) + 1);
  internal::AggregateSegment(unaligned_seg, totals);
  EXPECT_TRUE(totals.empty());

  // Corrupt magic.
  seg->header.magic.store(0xDEADBEEF, std::memory_order_release);
  internal::AggregateSegment(seg, totals);
  EXPECT_TRUE(totals.empty());
  seg->header.magic.store(kRaidenShmMagic, std::memory_order_release);

  // Corrupt max_toc_entries.
  seg->header.max_toc_entries = kMaxTocEntries + 1;
  internal::AggregateSegment(seg, totals);
  EXPECT_TRUE(totals.empty());
  seg->header.max_toc_entries = kMaxTocEntries;

  // Corrupt data_pool_offset below sizeof(ShmSegmentLayout).
  seg->header.data_pool_offset = sizeof(ShmSegmentLayout) - 1;
  internal::AggregateSegment(seg, totals);
  EXPECT_TRUE(totals.empty());

  // Corrupt data_pool_offset >= kSegmentTotalFileSize.
  seg->header.data_pool_offset = kSegmentTotalFileSize;
  internal::AggregateSegment(seg, totals);
  EXPECT_TRUE(totals.empty());

  // Restore valid data_pool_offset.
  seg->header.data_pool_offset = sizeof(ShmSegmentLayout);

  const uint32_t pool = sizeof(ShmSegmentLayout);
  new (mem->data + pool) std::atomic<double>(42.5);
  new (mem->data + pool + kMetricSlotAlignment) std::atomic<uint64_t>(100);
  auto* histogram_slot = reinterpret_cast<ShmHistogramSlot*>(
      mem->data + pool + 2 * kMetricSlotAlignment);
  histogram_slot->sample_count.store(1);
  histogram_slot->sample_sum.store(0.5);
  histogram_slot->bucket_counts[2].store(1);

  struct EntrySpec {
    absl::string_view name;
    MetricType type;
    uint32_t off;
    uint32_t size;
    TocEntryState state;
  };
  const std::array specs = {
      EntrySpec{"uncomm", MetricType::kCounter, pool, 8,
                TocEntryState::kWriting},
      EntrySpec{"gauge_m", MetricType::kGauge, pool, 8,
                TocEntryState::kCommitted},
      EntrySpec{"counter_m", MetricType::kCounter, pool + kMetricSlotAlignment,
                8, TocEntryState::kCommitted},
      EntrySpec{"hist_m", MetricType::kHistogram,
                pool + 2 * kMetricSlotAlignment, sizeof(ShmHistogramSlot),
                TocEntryState::kCommitted},
      EntrySpec{"oob", MetricType::kCounter, kSegmentTotalFileSize + 100, 8,
                TocEntryState::kCommitted},
      EntrySpec{"overflow", MetricType::kCounter, pool, UINT32_MAX,
                TocEntryState::kCommitted},
      EntrySpec{"unaligned", MetricType::kCounter, pool + 3, 8,
                TocEntryState::kCommitted},
      EntrySpec{"", MetricType::kCounter, pool + kMetricSlotAlignment, 8,
                TocEntryState::kCommitted},
      EntrySpec{"underflow", MetricType::kCounter,
                sizeof(ShmSegmentLayout) - 64, 8, TocEntryState::kCommitted},
  };
  for (size_t i = 0; i < specs.size(); ++i) {
    ShmTocEntry& toc_entry = seg->toc[i];
    snprintf(toc_entry.metric_name, sizeof(toc_entry.metric_name), "%.*s",
             static_cast<int>(specs[i].name.size()), specs[i].name.data());
    toc_entry.type = specs[i].type;
    toc_entry.offset = specs[i].off;
    toc_entry.size = specs[i].size;
    toc_entry.entry_state.store(specs[i].state);
  }
  seg->header.toc_entry_count.store(specs.size());

  internal::AggregateSegment(seg, totals);
  EXPECT_FALSE(totals.contains("uncomm"));
  EXPECT_FALSE(totals.contains("oob"));
  EXPECT_FALSE(totals.contains("overflow"));
  EXPECT_FALSE(totals.contains("unaligned"));
  EXPECT_FALSE(totals.contains(""));
  EXPECT_FALSE(totals.contains("underflow"));
  EXPECT_THAT(totals["gauge_m"], DoubleEq(42.5));
  EXPECT_THAT(totals["counter_m"], DoubleEq(100.0));
  EXPECT_THAT(totals["hist_m/count"], DoubleEq(1.0));
  EXPECT_THAT(totals["hist_m/sum"], DoubleEq(0.5));
  EXPECT_THAT(totals["hist_m/bucket_2"], DoubleEq(1.0));
}

}  // namespace
}  // namespace tpu_raiden::telemetry

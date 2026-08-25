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

#include "tpu_sync/telemetry/shm/shm_writer.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

class ShmWriterTest : public testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = absl::StrCat(testing::TempDir(), "/shm_writer_test_", getpid());
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  ShmWriterOptions DefaultOptions(absl::string_view rank = "0") const {
    return ShmWriterOptions{.shm_dir = test_dir_,
                            .local_rank = std::string(rank)};
  }

  static bool IsValid(const ShmWriter& writer) {
    absl::MutexLock lock(writer.mutex_);
    return !writer.chunks_.empty() && writer.chunks_[0].segment != nullptr;
  }

  static const ShmSegmentLayout* GetSegment(const ShmWriter& writer) {
    absl::MutexLock lock(writer.mutex_);
    return writer.chunks_.empty() ? nullptr : writer.chunks_[0].segment;
  }

  static std::string GetFilePath(const ShmWriter& writer) {
    absl::MutexLock lock(writer.mutex_);
    return writer.chunks_.empty() ? "" : writer.chunks_[0].file_path;
  }

  static std::string GetUuid(const ShmWriter& writer) { return writer.uuid_; }

  static int GetFd(const ShmWriter& writer) {
    absl::MutexLock lock(writer.mutex_);
    return writer.chunks_.empty() ? -1 : writer.chunks_[0].fd;
  }

  static const ShmTocEntry* FindTocEntry(const ShmSegmentLayout* segment,
                                         absl::string_view name,
                                         LabelSpan labels = {}) {
    if (!segment) return nullptr;
    std::string encoded = EncodeLabels(labels);
    uint32_t count =
        segment->header.toc_entry_count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; ++i) {
      if (segment->toc[i].metric_name == name) {
        if (labels.empty() || segment->toc[i].encoded_labels == encoded) {
          return &segment->toc[i];
        }
      }
    }
    return nullptr;
  }

  template <typename T>
  static const T* ReadSlot(const ShmWriter& writer, absl::string_view name,
                           LabelSpan labels = {}) {
    const ShmSegmentLayout* segment = GetSegment(writer);
    const ShmTocEntry* entry = FindTocEntry(segment, name, labels);
    if (!entry) return nullptr;
    return reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(segment) + entry->offset);
  }

  static uint64_t ReadCounter(const ShmWriter& writer, absl::string_view name,
                              LabelSpan labels = {}) {
    const std::atomic<uint64_t>* slot =
        ReadSlot<std::atomic<uint64_t>>(writer, name, labels);
    return slot ? slot->load(std::memory_order_relaxed) : 0;
  }

  static double ReadGauge(const ShmWriter& writer, absl::string_view name,
                          LabelSpan labels = {}) {
    const std::atomic<double>* slot =
        ReadSlot<std::atomic<double>>(writer, name, labels);
    return slot ? slot->load(std::memory_order_relaxed) : 0.0;
  }

  static const ShmHistogramSlot* ReadHistogram(const ShmWriter& writer,
                                               absl::string_view name,
                                               LabelSpan labels = {}) {
    return ReadSlot<ShmHistogramSlot>(writer, name, labels);
  }

  std::string test_dir_;
};

namespace {

using ::testing::DoubleEq;
using ::testing::IsNull;
using ::testing::NotNull;

TEST_F(ShmWriterTest, InitializesEmptyTOCAndExpandsDynamically) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));
  EXPECT_EQ(GetFilePath(writer),
            absl::StrCat(test_dir_, "/", kShmFilePrefix, "0_", GetUuid(writer),
                         "_chunk_0", kShmFileExtension));

  const ShmSegmentLayout* segment = GetSegment(writer);
  ASSERT_THAT(segment, NotNull());

  EXPECT_EQ(segment->header.magic.load(), kRaidenShmMagic);
  EXPECT_EQ(segment->header.version, kSupportedVersion);
  EXPECT_EQ(segment->header.pid, getpid());
  EXPECT_EQ(segment->header.max_toc_entries, kMaxTocEntries);
  EXPECT_EQ(segment->header.data_pool_offset, sizeof(ShmSegmentLayout));
  EXPECT_EQ(segment->header.toc_entry_count.load(), 0);
  EXPECT_EQ(segment->header.data_pool_bytes.load(), 0);

  // Dynamic allocation on demand.
  MetricLabel sent_label{metric_labels::kDirection,
                         metric_labels::kDirectionPush};
  MetricLabel fail_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull},
      {metric_labels::kErrorCode, "INTERNAL"}};

  writer.IncrementCounter(metric_names::kSentBytesTotal, {&sent_label, 1}, 100);
  writer.IncrementCounter(metric_names::kTransferFailuresTotal, fail_labels, 1);
  writer.SetGauge(metric_names::kBufferAllocatedBytes, {}, 5.0);

  EXPECT_EQ(segment->header.toc_entry_count.load(), 3);
  EXPECT_GT(segment->header.data_pool_bytes.load(), 0);

  EXPECT_EQ(segment->toc[0].type, MetricType::kCounter);
  EXPECT_EQ(absl::string_view(segment->toc[0].metric_name),
            metric_names::kSentBytesTotal);
  EXPECT_TRUE(
      absl::StrContains(segment->toc[0].encoded_labels, "direction=push"));

  EXPECT_EQ(segment->toc[1].type, MetricType::kCounter);
  EXPECT_EQ(absl::string_view(segment->toc[1].metric_name),
            metric_names::kTransferFailuresTotal);
  EXPECT_TRUE(
      absl::StrContains(segment->toc[1].encoded_labels, "direction=pull"));
  EXPECT_TRUE(
      absl::StrContains(segment->toc[1].encoded_labels, "error_code=INTERNAL"));

  EXPECT_EQ(segment->toc[2].type, MetricType::kGauge);
  EXPECT_EQ(absl::string_view(segment->toc[2].metric_name),
            metric_names::kBufferAllocatedBytes);
}

TEST_F(ShmWriterTest, DataPoolOffsetMonotonicityAndContiguity) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  MetricLabel push{metric_labels::kDirection, metric_labels::kDirectionPush};
  MetricLabel pull{metric_labels::kDirection, metric_labels::kDirectionPull};
  writer.IncrementCounter(metric_names::kSentBytesTotal, {&push, 1}, 1);
  writer.IncrementCounter(metric_names::kSentBytesTotal, {&pull, 1}, 2);
  writer.ObserveHistogram(metric_names::kTransferDurationMs, {}, 0.05);

  const ShmSegmentLayout* segment = GetSegment(writer);
  ASSERT_THAT(segment, NotNull());
  EXPECT_EQ(segment->header.toc_entry_count.load(), 3);

  uint32_t expected_offset = sizeof(ShmSegmentLayout);
  for (uint32_t i = 0; i < segment->header.toc_entry_count.load(); ++i) {
    const ShmTocEntry& entry = segment->toc[i];
    uint32_t align = kMetricSlotAlignment;
    expected_offset = (expected_offset + align - 1) & ~(align - 1);
    EXPECT_EQ(entry.offset, expected_offset);
    EXPECT_GT(entry.size, 0);
    expected_offset += entry.size;
  }
}

TEST_F(ShmWriterTest, PointerCacheResolutionAndMultiLabelUpdates) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  MetricLabel push{metric_labels::kDirection, metric_labels::kDirectionPush};
  MetricLabel pull{metric_labels::kDirection, metric_labels::kDirectionPull};

  writer.IncrementCounter(metric_names::kSentBytesTotal, {&push, 1}, 42);
  writer.IncrementCounter(metric_names::kSentBytesTotal, {&pull, 1}, 100);

  EXPECT_EQ(GetSegment(writer)->header.toc_entry_count.load(), 2);
  EXPECT_EQ(ReadCounter(writer, metric_names::kSentBytesTotal, {&push, 1}), 42);
  EXPECT_EQ(ReadCounter(writer, metric_names::kSentBytesTotal, {&pull, 1}),
            100);

  // Subsequent updates to cached entries do not allocate new TOC slots.
  writer.IncrementCounter(metric_names::kSentBytesTotal, {&push, 1}, 58);
  EXPECT_EQ(GetSegment(writer)->header.toc_entry_count.load(), 2);
  EXPECT_EQ(ReadCounter(writer, metric_names::kSentBytesTotal, {&push, 1}),
            100);
  EXPECT_EQ(ReadCounter(writer, metric_names::kSentBytesTotal, {&pull, 1}),
            100);
}

TEST_F(ShmWriterTest, HoldsAdvisorySharedFlockAndReleasesOnDestruction) {
  std::string path;
  {
    ShmWriter writer(DefaultOptions());
    ASSERT_TRUE(IsValid(writer));
    path = GetFilePath(writer);
    EXPECT_TRUE(std::filesystem::exists(path));

    // Verify O_CLOEXEC flag on writer descriptor.
    int flags = fcntl(GetFd(writer), F_GETFD);
    EXPECT_TRUE(flags & FD_CLOEXEC);

    // Open duplicate descriptor to probe advisory locking.
    int probe_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    ASSERT_GE(probe_fd, 0);

    // Exclusive lock must fail with EWOULDBLOCK while writer is active.
    EXPECT_EQ(flock(probe_fd, LOCK_EX | LOCK_NB), -1);
    EXPECT_EQ(errno, EWOULDBLOCK);

    // Concurrent shared lock must succeed.
    EXPECT_EQ(flock(probe_fd, LOCK_SH | LOCK_NB), 0);
    flock(probe_fd, LOCK_UN);
    close(probe_fd);
  }

  // File remains on disk after destruction for ShmCollector dead reaping.
  EXPECT_TRUE(std::filesystem::exists(path));

  // Lock must now be completely free, allowing exclusive acquisition.
  int dead_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
  ASSERT_GE(dead_fd, 0);
  EXPECT_EQ(flock(dead_fd, LOCK_EX | LOCK_NB), 0);
  flock(dead_fd, LOCK_UN);
  close(dead_fd);
}

TEST_F(ShmWriterTest, RecordsMetricsWithMixedLabelOrdering) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  MetricLabel labels_canonical[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull},
      {metric_labels::kErrorCode, "INTERNAL"}};
  MetricLabel labels_reversed[] = {
      {metric_labels::kErrorCode, "INTERNAL"},
      {metric_labels::kDirection, metric_labels::kDirectionPull}};

  writer.IncrementCounter(metric_names::kTransferFailuresTotal,
                          labels_canonical, 10);
  writer.IncrementCounter(metric_names::kTransferFailuresTotal, labels_reversed,
                          15);

  EXPECT_EQ(GetSegment(writer)->header.toc_entry_count.load(), 1);
  EXPECT_EQ(ReadCounter(writer, metric_names::kTransferFailuresTotal), 25);
}

TEST_F(ShmWriterTest, DynamicMetricsAndLabelsAllocatedSafely) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  MetricLabel label{"custom_key", "custom_val"};
  writer.IncrementCounter("custom_metric", {&label, 1}, 100);
  writer.SetGauge("custom_gauge", {&label, 1}, 3.14);
  writer.ObserveHistogram("custom_hist", {&label, 1}, 1.5);

  EXPECT_EQ(ReadCounter(writer, "custom_metric"), 100);
  EXPECT_THAT(ReadGauge(writer, "custom_gauge"), DoubleEq(3.14));
  const ShmHistogramSlot* histogram_slot = ReadHistogram(writer, "custom_hist");
  ASSERT_THAT(histogram_slot, NotNull());
  EXPECT_EQ(histogram_slot->sample_count.load(), 1);
}

TEST(ShmWriterStaticTest, NonCopyableAndNonMovable) {
  static_assert(!std::is_copy_constructible_v<ShmWriter>);
  static_assert(!std::is_copy_assignable_v<ShmWriter>);
  static_assert(!std::is_move_constructible_v<ShmWriter>);
  static_assert(!std::is_move_assignable_v<ShmWriter>);
}

TEST_F(ShmWriterTest, ConcurrentMultiThreadedIncrementCounter) {
  constexpr int kNumThreads = 16;
  constexpr int kItersPerThread = 25000;

  MetricLabel sent_label{metric_labels::kDirection,
                         metric_labels::kDirectionPush};
  MetricLabel fail_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPull},
      {metric_labels::kErrorCode, "RESOURCE_EXHAUSTED"}};

  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    workers.emplace_back([&writer, &sent_label, &fail_labels]() {
      for (int j = 0; j < kItersPerThread; ++j) {
        writer.IncrementCounter(metric_names::kSentBytesTotal, {&sent_label, 1},
                                1);
        writer.IncrementCounter(metric_names::kTransferFailuresTotal,
                                fail_labels, 2);
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  EXPECT_EQ(ReadCounter(writer, metric_names::kSentBytesTotal),
            static_cast<uint64_t>(kNumThreads) * kItersPerThread);
  EXPECT_EQ(ReadCounter(writer, metric_names::kTransferFailuresTotal),
            static_cast<uint64_t>(kNumThreads) * kItersPerThread * 2);
}

TEST_F(ShmWriterTest, ConcurrentMultiThreadedMixedWorkloadStress) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  constexpr int kNumThreads = 32;
  constexpr int kIterations = 10000;

  MetricLabel push_label{metric_labels::kDirection,
                         metric_labels::kDirectionPush};
  MetricLabel pull_label{metric_labels::kDirection,
                         metric_labels::kDirectionPull};
  MetricLabel fail_labels[] = {
      {metric_labels::kDirection, metric_labels::kDirectionPullResponse},
      {metric_labels::kErrorCode, "UNAVAILABLE"}};

  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    workers.emplace_back([&writer, &push_label, &pull_label, &fail_labels,
                          t]() {
      for (int i = 0; i < kIterations; ++i) {
        int op = (t + i) % 4;
        switch (op) {
          case 0:
            writer.IncrementCounter(metric_names::kSentBytesTotal,
                                    {&push_label, 1}, 1);
            break;
          case 1:
            writer.IncrementCounter(metric_names::kTransferFailuresTotal,
                                    fail_labels, 5);
            break;
          case 2:
            writer.IncrementCounter(metric_names::kReceivedBytesTotal,
                                    {&push_label, 1}, 10);
            break;
          case 3:
            writer.IncrementCounter("dynamic_metric", {&pull_label, 1}, 100);
            writer.SetGauge("dynamic_metric", {&pull_label, 1}, 3.14);
            writer.ObserveHistogram("dynamic_metric", {&pull_label, 1}, 1.0);
            break;
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  EXPECT_GT(ReadCounter(writer, metric_names::kSentBytesTotal), 0);
  EXPECT_GT(ReadCounter(writer, metric_names::kTransferFailuresTotal), 0);
  EXPECT_GT(ReadCounter(writer, metric_names::kReceivedBytesTotal), 0);
  EXPECT_GT(ReadCounter(writer, "dynamic_metric"), 0);
}

TEST_F(ShmWriterTest, BoundaryLabelsAndAdversarialCases) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  // Empty label span allocates valid slot.
  writer.IncrementCounter(metric_names::kSentBytesTotal, {}, 1);

  // Exactly 1 label (dedicated fast-path branch in EncodeLabels).
  MetricLabel label_1[1] = {{"k1", "v1"}};
  writer.IncrementCounter(metric_names::kSentBytesTotal, {label_1, 1}, 2);

  // Exactly 8 labels (stack array upper bound in EncodeLabels).
  MetricLabel labels_8[8] = {{"k1", "v1"}, {"k2", "v2"}, {"k3", "v3"},
                             {"k4", "v4"}, {"k5", "v5"}, {"k6", "v6"},
                             {"k7", "v7"}, {"k8", "v8"}};
  writer.IncrementCounter(metric_names::kSentBytesTotal, labels_8, 3);

  // > 8 labels (dynamically allocated vector fallback in EncodeLabels).
  MetricLabel labels_20[20] = {
      {"l01", "1"}, {"l02", "2"}, {"l03", "3"}, {"l04", "4"}, {"l05", "5"},
      {"l06", "6"}, {"l07", "7"}, {"l08", "8"}, {"l09", "9"}, {"l10", "0"},
      {"l11", "1"}, {"l12", "2"}, {"l13", "3"}, {"l14", "4"}, {"l15", "5"},
      {"l16", "6"}, {"l17", "7"}, {"l18", "8"}, {"l19", "9"}, {"l20", "0"}};
  writer.IncrementCounter(metric_names::kSentBytesTotal, labels_20, 4);

  // Verify memory boundaries for all registered slots
  const ShmSegmentLayout* segment = GetSegment(writer);
  ASSERT_THAT(segment, NotNull());
  const uint8_t* seg_start = reinterpret_cast<const uint8_t*>(segment);
  const uint8_t* seg_end = seg_start + kSegmentTotalFileSize;

  EXPECT_EQ(segment->header.toc_entry_count.load(), 4);
  for (uint32_t i = 0; i < segment->header.toc_entry_count.load(); ++i) {
    const ShmTocEntry& entry = segment->toc[i];
    EXPECT_GE(entry.offset, sizeof(ShmSegmentLayout));
    EXPECT_LT(entry.offset + entry.size, kSegmentTotalFileSize);
    const uint8_t* slot_addr = seg_start + entry.offset;
    EXPECT_GE(slot_addr, seg_start);
    EXPECT_LE(slot_addr + entry.size, seg_end);
  }
}

TEST_F(ShmWriterTest, EscapedDelimiterLabelsAndCollisionFreeTOC) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  // Case 1: Label value containing ';' and '='
  MetricLabel label1[1] = {{"k", "v1;k2=v2"}};
  // Case 2: Two discrete labels that would collide with unescaped
  // representation
  MetricLabel label2[2] = {{"k", "v1"}, {"k2", "v2"}};

  writer.IncrementCounter("delim_test", {label1, 1}, 10);
  writer.IncrementCounter("delim_test", label2, 20);

  const ShmSegmentLayout* segment = GetSegment(writer);
  ASSERT_THAT(segment, NotNull());
  EXPECT_EQ(segment->header.toc_entry_count.load(), 2);

  EXPECT_EQ(ReadCounter(writer, "delim_test", {label1, 1}), 10);
  EXPECT_EQ(ReadCounter(writer, "delim_test", label2), 20);
}

TEST_F(ShmWriterTest, OversizedMetricNameAndLabelsRejection) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  std::string huge_name(70, 'x');  // exceeds 64-byte buffer
  writer.IncrementCounter(huge_name, {}, 1);

  const ShmSegmentLayout* segment = GetSegment(writer);
  ASSERT_THAT(segment, NotNull());
  EXPECT_EQ(segment->header.toc_entry_count.load(), 0);

  std::string huge_label_val(150, 'y');  // exceeds 128-byte buffer
  MetricLabel huge_label[1] = {{"key", huge_label_val}};
  writer.IncrementCounter("valid_name", {huge_label, 1}, 1);
  EXPECT_EQ(segment->header.toc_entry_count.load(), 0);
}

TEST_F(ShmWriterTest, FlockLivenessAcrossMetricMutations) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));
  const std::string path = GetFilePath(writer);

  int probe_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
  ASSERT_GE(probe_fd, 0);

  // Probe: Exclusive lock must fail immediately with EWOULDBLOCK.
  EXPECT_EQ(flock(probe_fd, LOCK_EX | LOCK_NB), -1);
  EXPECT_EQ(errno, EWOULDBLOCK);

  // Perform continuous mutations while verifying lock remains held.
  MetricLabel label{metric_labels::kDirection, metric_labels::kDirectionPush};
  for (int i = 0; i < 500; ++i) {
    writer.IncrementCounter(metric_names::kSentBytesTotal, {&label, 1}, 10);
    if (i % 50 == 0) {
      EXPECT_EQ(flock(probe_fd, LOCK_EX | LOCK_NB), -1);
      EXPECT_EQ(errno, EWOULDBLOCK);
      EXPECT_EQ(flock(probe_fd, LOCK_SH | LOCK_NB), 0);
    }
  }

  close(probe_fd);
}

TEST_F(ShmWriterTest, CrossProcessExclusiveLockContentionFork) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));
  const std::string path = GetFilePath(writer);

  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {
    int child_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (child_fd < 0) _exit(1);
    if (flock(child_fd, LOCK_EX | LOCK_NB) != -1 || errno != EWOULDBLOCK) {
      close(child_fd);
      _exit(2);
    }
    if (flock(child_fd, LOCK_SH | LOCK_NB) != 0) {
      close(child_fd);
      _exit(3);
    }
    flock(child_fd, LOCK_UN);
    close(child_fd);
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(ShmWriterTest, CrossProcessLockReleaseOnProcessExit) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {
    ShmWriter child_writer(DefaultOptions());
    if (!IsValid(child_writer)) _exit(1);
    MetricLabel label{metric_labels::kDirection, metric_labels::kDirectionPush};
    child_writer.IncrementCounter(metric_names::kSentBytesTotal, {&label, 1},
                                  777);
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  std::string expected_file;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    if (absl::StrContains(entry.path().string(), "worker_rank_0_") &&
        absl::EndsWith(entry.path().string(), "_chunk_0.mmap")) {
      expected_file = entry.path().string();
      break;
    }
  }
  ASSERT_FALSE(expected_file.empty());
  EXPECT_TRUE(std::filesystem::exists(expected_file));

  // The OS kernel automatically releases advisory locks on child exit.
  int dead_fd = open(expected_file.c_str(), O_RDWR | O_CLOEXEC);
  ASSERT_GE(dead_fd, 0);
  EXPECT_EQ(flock(dead_fd, LOCK_EX | LOCK_NB), 0);
  flock(dead_fd, LOCK_UN);
  close(dead_fd);
}

TEST_F(ShmWriterTest, InvalidDoubleValuesNaNAndInfResilience) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  writer.ObserveHistogram("hist", {}, std::numeric_limits<double>::quiet_NaN());
  writer.ObserveHistogram("hist", {}, std::numeric_limits<double>::infinity());
  writer.ObserveHistogram("hist", {}, -std::numeric_limits<double>::infinity());

  const ShmHistogramSlot* slot = ReadHistogram(writer, "hist");
  ASSERT_THAT(slot, NotNull());
  EXPECT_EQ(slot->sample_count.load(), 0);
  EXPECT_THAT(slot->sample_sum.load(), DoubleEq(0.0));

  writer.ObserveHistogram("hist", {}, 10.5);
  EXPECT_EQ(slot->sample_count.load(), 1);
  EXPECT_THAT(slot->sample_sum.load(), DoubleEq(10.5));

  // Subnormal / extreme finite value resilience.
  writer.ObserveHistogram("hist", {},
                          std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(slot->sample_count.load(), 2);

  // Gauge ignores non-finite values.
  writer.SetGauge("gauge_test", {}, 42.0);
  EXPECT_THAT(ReadGauge(writer, "gauge_test"), DoubleEq(42.0));
  writer.SetGauge("gauge_test", {}, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THAT(ReadGauge(writer, "gauge_test"), DoubleEq(42.0));
  writer.SetGauge("gauge_test", {}, std::numeric_limits<double>::infinity());
  EXPECT_THAT(ReadGauge(writer, "gauge_test"), DoubleEq(42.0));
  writer.SetGauge("gauge_test", {}, -std::numeric_limits<double>::infinity());
  EXPECT_THAT(ReadGauge(writer, "gauge_test"), DoubleEq(42.0));
}

TEST_F(ShmWriterTest, InitializationFailureGracefulDegradation) {
  // 1. Empty shm_dir
  ShmWriter empty_dir(ShmWriterOptions{.shm_dir = "", .local_rank = "0"});
  EXPECT_FALSE(IsValid(empty_dir));
  EXPECT_THAT(GetSegment(empty_dir), IsNull());

  // 2. Empty local_rank
  ShmWriter empty_rank(
      ShmWriterOptions{.shm_dir = test_dir_, .local_rank = ""});
  EXPECT_FALSE(IsValid(empty_rank));
  EXPECT_THAT(GetSegment(empty_rank), IsNull());

  // 3. Impossible directory path
  std::string regular_file = absl::StrCat(test_dir_, "/regular_file");
  int fd = open(regular_file.c_str(), O_CREAT | O_WRONLY, 0644);
  ASSERT_GE(fd, 0);
  close(fd);

  ShmWriter invalid_dir_shm_writer(
      ShmWriterOptions{.shm_dir = absl::StrCat(regular_file, "/impossible_dir"),
                       .local_rank = "0"});
  EXPECT_FALSE(IsValid(invalid_dir_shm_writer));
  EXPECT_EQ(GetFd(invalid_dir_shm_writer), -1);
  EXPECT_THAT(GetSegment(invalid_dir_shm_writer), IsNull());

  // Operations on invalid writer must safely no-op without crash.
  MetricLabel label{metric_labels::kDirection, metric_labels::kDirectionPush};
  invalid_dir_shm_writer.IncrementCounter(metric_names::kSentBytesTotal,
                                          {&label, 1}, 10);
  invalid_dir_shm_writer.SetGauge("any_gauge", {&label, 1}, 1.0);
  invalid_dir_shm_writer.ObserveHistogram("any_hist", {&label, 1}, 2.0);
}

TEST_F(ShmWriterTest, RapidConstructionDestructionCycle) {
  MetricLabel label{metric_labels::kDirection, metric_labels::kDirectionPush};
  for (int i = 0; i < 50; ++i) {
    ShmWriter writer(DefaultOptions());
    ASSERT_TRUE(IsValid(writer));
    writer.IncrementCounter(metric_names::kSentBytesTotal, {&label, 1}, i);
  }
}

TEST_F(ShmWriterTest, DynamicMultiChunkExpansionOnOverflow) {
  ShmWriter writer(DefaultOptions());
  ASSERT_TRUE(IsValid(writer));

  std::string chunk0_path = GetFilePath(writer);
  std::string chunk1_path =
      absl::StrCat(test_dir_, "/", kShmFilePrefix, "0", "_", GetUuid(writer),
                   "_chunk_1", kShmFileExtension);

  EXPECT_TRUE(std::filesystem::exists(chunk0_path));
  EXPECT_FALSE(std::filesystem::exists(chunk1_path));

  // Allocate kMaxTocEntries + 5 unique metrics to force chunk 0 exhaustion and
  // allocate chunk 1.
  for (size_t i = 0; i < kMaxTocEntries + 5; ++i) {
    std::string name = absl::StrCat("metric_", i);
    writer.IncrementCounter(name, {}, static_cast<uint64_t>(i + 1));
  }

  EXPECT_TRUE(std::filesystem::exists(chunk1_path));
  EXPECT_EQ(ReadCounter(writer, "metric_0"), 1);

  writer.IncrementCounter("metric_0", {}, 10);
  EXPECT_EQ(ReadCounter(writer, "metric_0"), 11);
}

TEST_F(ShmWriterTest, CustomFileModePermissions) {
  ShmWriter writer(ShmWriterOptions{
      .shm_dir = test_dir_, .local_rank = "0", .file_mode = 0600});
  ASSERT_TRUE(IsValid(writer));
  std::string path = GetFilePath(writer);

  struct stat st;
  ASSERT_EQ(stat(path.c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 0777, 0600);
}

}  // namespace
}  // namespace tpu_raiden::telemetry

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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::DoubleEq;
using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

inline constexpr mode_t kDefaultFileMode =
    S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;                        // 0644
inline constexpr mode_t kRestrictedFileMode = S_IRUSR | S_IWUSR;  // 0600

static_assert(!std::is_copy_constructible_v<ShmWriter> &&
              !std::is_copy_assignable_v<ShmWriter> &&
              !std::is_move_constructible_v<ShmWriter> &&
              !std::is_move_assignable_v<ShmWriter>);

// Streamlined RAII helper to discover, mmap, and inspect ShmWriter segments.
class MappedSegment {
 public:
  MappedSegment() = default;
  explicit MappedSegment(absl::string_view path, absl::string_view rank = "0",
                         uint32_t chunk_idx = 0) {
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec) || ec) {
      return;
    }
    std::string file_prefix = absl::StrCat(kShmFilePrefix, rank, "_");
    std::string file_suffix =
        absl::StrCat("_chunk_", chunk_idx, kShmFileExtension);
    auto dir_it = std::filesystem::directory_iterator(path, ec);
    if (ec) {
      return;
    }
    for (const std::filesystem::directory_entry& entry : dir_it) {
      if (ec) {
        break;
      }
      std::string filename = entry.path().filename().string();
      if (absl::StartsWith(filename, file_prefix) &&
          absl::EndsWith(filename, file_suffix)) {
        file_path_ = entry.path().string();
        break;
      }
    }
    if (file_path_.empty()) {
      return;
    }
    fd_ = open(file_path_.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat file_stat;
    if (fd_ >= 0 && fstat(fd_, &file_stat) == 0 &&
        file_stat.st_size >= kSegmentTotalFileSize) {
      void* addr =
          mmap(nullptr, kSegmentTotalFileSize, PROT_READ, MAP_SHARED, fd_, 0);
      if (addr != MAP_FAILED) {
        segment_ = static_cast<const ShmSegmentLayout*>(addr);
      }
    }
  }
  ~MappedSegment() {
    if (segment_ != nullptr) {
      munmap(const_cast<ShmSegmentLayout*>(segment_), kSegmentTotalFileSize);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }
  MappedSegment(const MappedSegment&) = delete;
  MappedSegment& operator=(const MappedSegment&) = delete;

  bool is_valid() const {
    return segment_ && segment_->header.magic.load(std::memory_order_acquire) ==
                           kRaidenShmMagic;
  }
  const ShmSegmentLayout* segment() const { return segment_; }
  const std::string& file_path() const { return file_path_; }
  int fd() const { return fd_; }

  static absl::string_view BoundedString(const char* buf, size_t max_len) {
    return absl::string_view(buf, strnlen(buf, max_len));
  }

  const ShmTocEntry* FindToc(
      absl::string_view name, LabelSpan labels = {},
      std::optional<MetricType> type = std::nullopt) const {
    if (!is_valid()) {
      return nullptr;
    }
    std::string encoded = EncodeLabels(labels);
    uint32_t entry_count = std::min(
        segment_->header.toc_entry_count.load(std::memory_order_acquire),
        static_cast<uint32_t>(kMaxTocEntries));
    for (uint32_t i = 0; i < entry_count; ++i) {
      const ShmTocEntry& entry = segment_->toc[i];
      if (entry.entry_state.load(std::memory_order_acquire) ==
              TocEntryState::kCommitted &&
          (!type || entry.type == *type) &&
          BoundedString(entry.metric_name, sizeof(entry.metric_name)) == name &&
          BoundedString(entry.encoded_labels, sizeof(entry.encoded_labels)) ==
              encoded) {
        return &entry;
      }
    }
    return nullptr;
  }

  template <typename T, MetricType ExpectedType>
  const T* ReadSlot(absl::string_view name, LabelSpan labels = {}) const {
    const ShmTocEntry* entry = FindToc(name, labels, ExpectedType);
    if (entry == nullptr) {
      return nullptr;
    }
    if (entry->offset % alignof(T) != 0 ||
        entry->offset < sizeof(ShmSegmentLayout) ||
        entry->offset + sizeof(T) > kSegmentTotalFileSize ||
        entry->size < sizeof(T)) {
      return nullptr;
    }
    return reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(segment_) + entry->offset);
  }

  uint64_t ReadCounter(absl::string_view name, LabelSpan labels = {}) const {
    const std::atomic<uint64_t>* slot =
        ReadSlot<std::atomic<uint64_t>, MetricType::kCounter>(name, labels);
    return slot != nullptr ? slot->load(std::memory_order_relaxed) : 0;
  }
  double ReadGauge(absl::string_view name, LabelSpan labels = {}) const {
    const std::atomic<double>* slot =
        ReadSlot<std::atomic<double>, MetricType::kGauge>(name, labels);
    return slot != nullptr ? slot->load(std::memory_order_relaxed) : 0.0;
  }
  const ShmHistogramSlot* ReadHistogram(absl::string_view name,
                                        LabelSpan labels = {}) const {
    return ReadSlot<ShmHistogramSlot, MetricType::kHistogram>(name, labels);
  }

 private:
  int fd_ = -1;
  const ShmSegmentLayout* segment_ = nullptr;
  std::string file_path_;
};

class ShmWriterTest : public testing::Test {
 protected:
  void SetUp() override {
    old_umask_ = umask(0022);
    test_dir_ = absl::StrCat(testing::TempDir(), "/shm_writer_test_", getpid());
    std::error_code ec;
    std::filesystem::create_directories(test_dir_, ec);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
    umask(old_umask_);
  }
  ShmWriterOptions DefaultOptions(absl::string_view rank = "0") const {
    return ShmWriterOptions{.shm_dir = test_dir_,
                            .local_rank = std::string(rank)};
  }
  int FilePermissions(const std::string& path) const {
    struct stat file_stat;
    return stat(path.c_str(), &file_stat) == 0 ? (file_stat.st_mode & 0777)
                                               : -1;
  }
  std::string test_dir_;
  mode_t old_umask_ = 0;
};

// 1. Validates header magic, pid, initial TOC limits, contiguity, and
// permissions (0600 vs 0644).
TEST_F(ShmWriterTest, LifecycleAndSegmentLayout) {
  {
    ShmWriter writer(ShmWriterOptions{.shm_dir = test_dir_,
                                      .local_rank = "c",
                                      .file_mode = kRestrictedFileMode});
    MappedSegment mapped(test_dir_, "c");
    ASSERT_TRUE(mapped.is_valid());
    EXPECT_EQ(FilePermissions(mapped.file_path()), kRestrictedFileMode);
  }

  ShmWriter writer(DefaultOptions());
  MappedSegment mapped(test_dir_);
  ASSERT_TRUE(mapped.is_valid());
  EXPECT_EQ(FilePermissions(mapped.file_path()), kDefaultFileMode);
  EXPECT_THAT(mapped.file_path(), HasSubstr("worker_rank_0_"));
  EXPECT_THAT(mapped.file_path(), EndsWith("_chunk_0.mmap"));

  const ShmSegmentLayout* segment = mapped.segment();
  ASSERT_THAT(segment, NotNull());
  EXPECT_EQ(segment->header.magic.load(), kRaidenShmMagic);
  EXPECT_EQ(segment->header.pid, getpid());
  EXPECT_EQ(segment->header.max_toc_entries, kMaxTocEntries);
  EXPECT_EQ(segment->header.data_pool_offset, sizeof(ShmSegmentLayout));
  EXPECT_EQ(segment->header.toc_entry_count.load(), 0);
  EXPECT_EQ(segment->header.data_pool_bytes.load(), 0);

  MetricLabel push_label{metric_labels::kDirection,
                         metric_labels::kDirectionPush};
  MetricLabel pull_label{metric_labels::kDirection,
                         metric_labels::kDirectionPull};
  writer.IncrementCounter(metric_names::kSentBytesTotal, {&push_label, 1}, 100);
  writer.SetGauge(metric_names::kBufferAllocatedBytes, {&pull_label, 1}, 5.0);
  writer.ObserveHistogram(metric_names::kTransferDurationMs, {}, 0.05);

  EXPECT_EQ(segment->header.toc_entry_count.load(), 3);
  EXPECT_GT(segment->header.data_pool_bytes.load(), 0);

  uint32_t expected_offset = sizeof(ShmSegmentLayout);
  for (uint32_t i = 0; i < 3; ++i) {
    expected_offset = (expected_offset + kMetricSlotAlignment - 1) &
                      ~(kMetricSlotAlignment - 1);
    EXPECT_EQ(segment->toc[i].offset, expected_offset);
    expected_offset += segment->toc[i].size;
  }

  for (int i = 0; i < 5; ++i) {
    std::string rank = absl::StrCat("r", i);
    ShmWriter rank_writer(DefaultOptions(rank));
    MappedSegment rank_mapped(test_dir_, rank);
    ASSERT_TRUE(rank_mapped.is_valid());
    rank_writer.IncrementCounter(metric_names::kSentBytesTotal,
                                 {&push_label, 1}, i);
    EXPECT_EQ(rank_mapped.ReadCounter(metric_names::kSentBytesTotal,
                                      {&push_label, 1}),
              i);
  }
}

// 2. Consolidates counter, gauge, histogram mutations, label sorting, delimiter
// escaping, and multi-type same-name resolution.
TEST_F(ShmWriterTest, MetricRecordingAndLabelEncoding) {
  ShmWriter writer(DefaultOptions());
  MappedSegment mapped(test_dir_);
  ASSERT_TRUE(mapped.is_valid());

  MetricLabel custom_label{"k", "v"};
  writer.IncrementCounter("c", {&custom_label, 1}, 100);
  writer.SetGauge("g", {&custom_label, 1}, 3.14);
  writer.ObserveHistogram("h", {&custom_label, 1}, 1.5);
  EXPECT_EQ(mapped.ReadCounter("c", {&custom_label, 1}), 100);
  EXPECT_EQ(mapped.ReadCounter("c", {}), 0);
  EXPECT_THAT(mapped.ReadGauge("g", {&custom_label, 1}), DoubleEq(3.14));
  EXPECT_THAT(mapped.ReadGauge("g", {}), DoubleEq(0.0));
  const ShmHistogramSlot* hist = mapped.ReadHistogram("h", {&custom_label, 1});
  ASSERT_THAT(hist, NotNull());
  EXPECT_EQ(hist->sample_count.load(), 1);
  EXPECT_THAT(hist->sample_sum.load(), DoubleEq(1.5));
  EXPECT_THAT(mapped.ReadHistogram("h", {}), IsNull());

  MetricLabel wrong_label{"k", "w"};
  EXPECT_EQ(mapped.ReadCounter("c", {&wrong_label, 1}), 0);

  uint32_t toc_count = mapped.segment()->header.toc_entry_count.load();
  writer.IncrementCounter("c", {&custom_label, 1}, 50);
  EXPECT_EQ(mapped.ReadCounter("c", {&custom_label, 1}), 150);
  EXPECT_EQ(mapped.segment()->header.toc_entry_count.load(), toc_count);

  std::array<MetricLabel, 2> canonical_labels = {
      {{metric_labels::kDirection, metric_labels::kDirectionPull},
       {metric_labels::kErrorCode, "INTERNAL"}}};
  std::array<MetricLabel, 2> reversed_labels = {
      {{metric_labels::kErrorCode, "INTERNAL"},
       {metric_labels::kDirection, metric_labels::kDirectionPull}}};
  writer.IncrementCounter("sorted", canonical_labels, 10);
  writer.IncrementCounter("sorted", reversed_labels, 15);
  EXPECT_EQ(mapped.ReadCounter("sorted", canonical_labels), 25);
  EXPECT_EQ(mapped.ReadCounter("sorted", reversed_labels), 25);

  std::array<MetricLabel, 1> delimiter_label = {{{"k", "v1;k2=v2"}}};
  std::array<MetricLabel, 2> multi_labels = {{{"k", "v1"}, {"k2", "v2"}}};
  writer.IncrementCounter("delim", delimiter_label, 10);
  writer.IncrementCounter("delim", multi_labels, 20);
  EXPECT_EQ(mapped.ReadCounter("delim", delimiter_label), 10);
  EXPECT_EQ(mapped.ReadCounter("delim", multi_labels), 20);

  writer.IncrementCounter("same", {}, 10);
  writer.SetGauge("same", {}, 2.718);
  writer.ObserveHistogram("same", {}, 42.0);
  EXPECT_EQ(mapped.ReadCounter("same"), 10);
  EXPECT_THAT(mapped.ReadGauge("same"), DoubleEq(2.718));
  const ShmHistogramSlot* same_histogram = mapped.ReadHistogram("same");
  ASSERT_THAT(same_histogram, NotNull());
  EXPECT_THAT(same_histogram->sample_sum.load(), DoubleEq(42.0));
}

// 3. Verifies bucket threshold binning, cumulative counts, and boundary
// placement.
TEST_F(ShmWriterTest, HistogramDistributionAndBoundaries) {
  ShmWriter writer(DefaultOptions());
  MappedSegment mapped(test_dir_);
  ASSERT_TRUE(mapped.is_valid());

  writer.ObserveHistogram("hb", {}, 0.05);      // bucket 0 (< 0.1)
  writer.ObserveHistogram("hb", {}, 0.1);       // bucket 0 (== 0.1)
  writer.ObserveHistogram("hb", {}, 25.0);      // bucket 7 (== 25.0)
  writer.ObserveHistogram("hb", {}, 100000.0);  // overflow bucket

  const ShmHistogramSlot* slot = mapped.ReadHistogram("hb");
  ASSERT_THAT(slot, NotNull());
  EXPECT_EQ(slot->sample_count.load(), 4);
  EXPECT_THAT(slot->sample_sum.load(), DoubleEq(100025.15));
  EXPECT_EQ(slot->bucket_counts[0].load(), 2);
  EXPECT_EQ(slot->bucket_counts[7].load(), 1);
  EXPECT_EQ(slot->bucket_counts[kNumHistogramBuckets].load(), 1);

  for (size_t bucket = 0; bucket <= kNumHistogramBuckets; ++bucket) {
    if (bucket != 0 && bucket != 7 && bucket != kNumHistogramBuckets) {
      EXPECT_EQ(slot->bucket_counts[bucket].load(), 0);
    }
  }
}

// 4. Multi-threaded stress: 32 writer threads + 4 reader threads verifying
// deterministic sums (80k / 400k / 800k / 8M) and barrier integrity.
TEST_F(ShmWriterTest, ConcurrentWriterAndReaderObservationStress) {
  ShmWriter writer(DefaultOptions());
  MappedSegment mapped(test_dir_);
  ASSERT_TRUE(mapped.is_valid());

  constexpr int kNumWriterThreads = 32;
  constexpr int kNumReaderThreads = 4;
  constexpr int kIterationsPerWriter = 10000;

  MetricLabel push_label{metric_labels::kDirection,
                         metric_labels::kDirectionPush};
  MetricLabel pull_label{metric_labels::kDirection,
                         metric_labels::kDirectionPull};
  std::array<MetricLabel, 2> fail_labels = {
      {{metric_labels::kDirection, metric_labels::kDirectionPullResponse},
       {metric_labels::kErrorCode, "UNAVAILABLE"}}};

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> invariant_failures{0};
  std::atomic<uint64_t> read_cycles{0};
  std::vector<std::thread> readers;
  for (int reader_idx = 0; reader_idx < kNumReaderThreads; ++reader_idx) {
    readers.emplace_back([&]() {
      const ShmSegmentLayout* segment = mapped.segment();
      while (!stop.load(std::memory_order_relaxed)) {
        read_cycles.fetch_add(1, std::memory_order_relaxed);
        if (segment->header.magic.load(std::memory_order_acquire) !=
            kRaidenShmMagic) {
          continue;
        }
        uint32_t entry_count = std::min(
            segment->header.toc_entry_count.load(std::memory_order_acquire),
            static_cast<uint32_t>(kMaxTocEntries));
        for (uint32_t i = 0; i < entry_count; ++i) {
          const ShmTocEntry& entry = segment->toc[i];
          if (entry.entry_state.load(std::memory_order_acquire) ==
              TocEntryState::kCommitted) {
            if (entry.type != MetricType::kCounter &&
                entry.type != MetricType::kGauge &&
                entry.type != MetricType::kHistogram) {
              invariant_failures.fetch_add(1, std::memory_order_relaxed);
            }
            if (entry.offset < sizeof(ShmSegmentLayout) ||
                entry.offset + entry.size > kSegmentTotalFileSize ||
                entry.offset % kMetricSlotAlignment != 0 ||
                MappedSegment::BoundedString(entry.metric_name,
                                             sizeof(entry.metric_name))
                    .empty()) {
              invariant_failures.fetch_add(1, std::memory_order_relaxed);
            }
            if (entry.type == MetricType::kCounter &&
                entry.offset % alignof(std::atomic<uint64_t>) == 0 &&
                entry.offset + sizeof(std::atomic<uint64_t>) <=
                    kSegmentTotalFileSize) {
              const std::atomic<uint64_t>* counter_slot =
                  reinterpret_cast<const std::atomic<uint64_t>*>(
                      reinterpret_cast<const uint8_t*>(segment) + entry.offset);
              uint64_t value = counter_slot->load(std::memory_order_relaxed);
              absl::string_view name = MappedSegment::BoundedString(
                  entry.metric_name, sizeof(entry.metric_name));
              if (name == metric_names::kSentBytesTotal && value > 80000) {
                invariant_failures.fetch_add(1, std::memory_order_relaxed);
              } else if (name == metric_names::kTransferFailuresTotal &&
                         (value > 400000 || value % 5 != 0)) {
                invariant_failures.fetch_add(1, std::memory_order_relaxed);
              } else if (name == metric_names::kReceivedBytesTotal &&
                         (value > 800000 || value % 10 != 0)) {
                invariant_failures.fetch_add(1, std::memory_order_relaxed);
              } else if (name == "dyn_c" &&
                         (value > 8000000 || value % 100 != 0)) {
                invariant_failures.fetch_add(1, std::memory_order_relaxed);
              }
            }
          }
        }
      }
    });
  }

  std::vector<std::thread> writers;
  for (int thread_idx = 0; thread_idx < kNumWriterThreads; ++thread_idx) {
    writers.emplace_back([&, thread_idx]() {
      for (int i = 0; i < kIterationsPerWriter; ++i) {
        switch ((thread_idx + i) % 4) {
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
          default:
            writer.IncrementCounter("dyn_c", {&pull_label, 1}, 100);
            writer.SetGauge("dyn_g", {&pull_label, 1}, 3.14);
            writer.ObserveHistogram("dyn_h", {&pull_label, 1}, 1.0);
            break;
        }
      }
    });
  }

  for (std::thread& writer_thread : writers) {
    writer_thread.join();
  }
  stop.store(true, std::memory_order_relaxed);
  for (std::thread& reader_thread : readers) {
    reader_thread.join();
  }

  EXPECT_EQ(invariant_failures.load(), 0);
  EXPECT_GT(read_cycles.load(), 1000);
  EXPECT_EQ(mapped.ReadCounter(metric_names::kSentBytesTotal, {&push_label, 1}),
            80000);
  EXPECT_EQ(
      mapped.ReadCounter(metric_names::kTransferFailuresTotal, fail_labels),
      400000);
  EXPECT_EQ(
      mapped.ReadCounter(metric_names::kReceivedBytesTotal, {&push_label, 1}),
      800000);
  EXPECT_EQ(mapped.ReadCounter("dyn_c", {&pull_label, 1}), 8000000);
  EXPECT_THAT(mapped.ReadGauge("dyn_g", {&pull_label, 1}), DoubleEq(3.14));
  const ShmHistogramSlot* dyn_hist =
      mapped.ReadHistogram("dyn_h", {&pull_label, 1});
  ASSERT_THAT(dyn_hist, NotNull());
  EXPECT_EQ(dyn_hist->sample_count.load(), 80000);
  EXPECT_THAT(dyn_hist->sample_sum.load(), DoubleEq(80000.0));
}

// 5. Consolidates flock contention, mutation persistence, and release on
// process destruction.
TEST_F(ShmWriterTest, AdvisoryLockingAndCrossProcessLifecycle) {
  auto run_child = [](auto&& fn) -> bool {
    pid_t pid = fork();
    if (pid < 0) {
      return false;
    }
    if (pid == 0) {
      _exit(fn() ? 0 : 1);
    }
    int child_status = 0;
    if (waitpid(pid, &child_status, 0) != pid) {
      return false;
    }
    return WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
  };
  auto can_lock_exclusive = [](const std::string& file_path) -> bool {
    int fd = open(file_path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      return false;
    }
    bool lock_succeeded = flock(fd, LOCK_EX | LOCK_NB) == 0;
    if (lock_succeeded) {
      flock(fd, LOCK_UN);
    }
    close(fd);
    return lock_succeeded;
  };

  std::string path;
  {
    ShmWriter writer(DefaultOptions());
    MappedSegment mapped(test_dir_);
    ASSERT_TRUE(mapped.is_valid());
    path = mapped.file_path();
    std::error_code ec;
    EXPECT_TRUE(std::filesystem::exists(path, ec));
    EXPECT_FALSE(ec);

    auto has_cloexec_descriptor = [&](const std::string& target_path) -> bool {
      std::error_code dir_ec;
      auto dir_it =
          std::filesystem::directory_iterator("/proc/self/fd", dir_ec);
      if (dir_ec) {
        return false;
      }
      for (const std::filesystem::directory_entry& entry : dir_it) {
        if (dir_ec) {
          return false;
        }
        int fd_number = 0;
        std::error_code equiv_ec;
        if (std::filesystem::equivalent(entry.path(), target_path, equiv_ec) &&
            !equiv_ec &&
            absl::SimpleAtoi(entry.path().filename().string(), &fd_number) &&
            fd_number != mapped.fd()) {
          if ((fcntl(fd_number, F_GETFD) & FD_CLOEXEC) != 0) {
            return true;
          }
        }
      }
      return false;
    };
    EXPECT_TRUE(has_cloexec_descriptor(path));

    int probe_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    ASSERT_GE(probe_fd, 0);
    auto assert_contention = [&](int fd) {
      EXPECT_EQ(flock(fd, LOCK_EX | LOCK_NB), -1);
      EXPECT_EQ(errno, EWOULDBLOCK);
      EXPECT_EQ(flock(fd, LOCK_SH | LOCK_NB), 0);
      flock(fd, LOCK_UN);
    };
    assert_contention(probe_fd);

    MetricLabel label{metric_labels::kDirection, metric_labels::kDirectionPush};
    for (int i = 0; i < 100; ++i) {
      writer.IncrementCounter(metric_names::kSentBytesTotal, {&label, 1}, 10);
      if (i % 25 == 0) {
        assert_contention(probe_fd);
      }
    }
    close(probe_fd);

    EXPECT_TRUE(run_child([&]() {
      int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
      bool ok = fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == -1 &&
                errno == EWOULDBLOCK && flock(fd, LOCK_SH | LOCK_NB) == 0;
      if (fd >= 0) {
        flock(fd, LOCK_UN);
        close(fd);
      }
      return ok;
    }));
  }

  std::error_code exists_ec;
  EXPECT_TRUE(std::filesystem::exists(path, exists_ec));
  EXPECT_FALSE(exists_ec);
  EXPECT_TRUE(can_lock_exclusive(path));

  EXPECT_TRUE(run_child([&]() {
    ShmWriter child_writer(DefaultOptions("child"));
    MetricLabel child_label{metric_labels::kDirection,
                            metric_labels::kDirectionPush};
    child_writer.IncrementCounter(metric_names::kSentBytesTotal,
                                  {&child_label, 1}, 777);
    return true;
  }));

  MappedSegment child_mapped(test_dir_, "child");
  ASSERT_TRUE(child_mapped.is_valid());
  MetricLabel child_label{metric_labels::kDirection,
                          metric_labels::kDirectionPush};
  EXPECT_EQ(child_mapped.ReadCounter(metric_names::kSentBytesTotal,
                                     {&child_label, 1}),
            777);
  EXPECT_TRUE(can_lock_exclusive(child_mapped.file_path()));
}

// 6. Tests TOC slot overflow (1024), 64 KB data pool overflow (342 histograms),
// and 16 chunks limit.
TEST_F(ShmWriterTest, MultiChunkExpansionAndLimitHandling) {
  {
    ShmWriter writer(DefaultOptions());
    MappedSegment chunk0_mapped(test_dir_);
    ASSERT_TRUE(chunk0_mapped.is_valid());

    constexpr size_t total_metrics = ShmWriter::kMaxChunks * kMaxTocEntries;
    for (size_t i = 0; i < total_metrics; ++i) {
      writer.IncrementCounter(absl::StrCat("m_", i), {}, i + 1);
    }
    EXPECT_EQ(chunk0_mapped.ReadCounter("m_0"), 1);
    writer.IncrementCounter("m_0", {}, 10);
    EXPECT_EQ(chunk0_mapped.ReadCounter("m_0"), 11);

    MappedSegment chunk1_mapped(test_dir_, "0", 1);
    ASSERT_TRUE(chunk1_mapped.is_valid());
    EXPECT_EQ(chunk1_mapped.segment()->header.chunk_index, 1);
    EXPECT_EQ(chunk1_mapped.ReadCounter("m_1024"), 1025);

    MappedSegment chunk15_mapped(test_dir_, "0", 15);
    ASSERT_TRUE(chunk15_mapped.is_valid());
    EXPECT_EQ(chunk15_mapped.segment()->header.chunk_index, 15);
    EXPECT_EQ(chunk15_mapped.ReadCounter(absl::StrCat("m_", total_metrics - 1)),
              total_metrics);

    writer.IncrementCounter("overflow_m", {}, 10);
    writer.SetGauge("overflow_g", {}, 3.14);
    writer.ObserveHistogram("overflow_h", {}, 1.0);
    EXPECT_EQ(chunk15_mapped.ReadCounter("overflow_m"), 0);
    EXPECT_THAT(chunk15_mapped.ReadGauge("overflow_g"), DoubleEq(0.0));
    EXPECT_THAT(chunk15_mapped.ReadHistogram("overflow_h"), IsNull());
    EXPECT_EQ(chunk15_mapped.segment()->header.toc_entry_count.load(),
              kMaxTocEntries);

    std::error_code exists_ec;
    std::string chunk16_path =
        absl::StrReplaceAll(chunk0_mapped.file_path(),
                            {{absl::StrCat("_chunk_0", kShmFileExtension),
                              absl::StrCat("_chunk_16", kShmFileExtension)}});
    EXPECT_FALSE(std::filesystem::exists(chunk16_path, exists_ec));
    EXPECT_FALSE(exists_ec);
  }

  {
    ShmWriter hist_writer(DefaultOptions("hist"));
    MappedSegment hist_chunk0(test_dir_, "hist", 0);
    ASSERT_TRUE(hist_chunk0.is_valid());
    for (size_t i = 0; i < 342; ++i) {
      hist_writer.ObserveHistogram(absl::StrCat("h_", i), {},
                                   static_cast<double>(i + 1));
    }
    EXPECT_EQ(hist_chunk0.segment()->header.toc_entry_count.load(), 341);
    EXPECT_LT(hist_chunk0.segment()->header.toc_entry_count.load(),
              kMaxTocEntries);

    MappedSegment hist_chunk1(test_dir_, "hist", 1);
    ASSERT_TRUE(hist_chunk1.is_valid());
    EXPECT_EQ(hist_chunk1.segment()->header.chunk_index, 1);
    EXPECT_EQ(hist_chunk1.segment()->header.toc_entry_count.load(), 1);

    const ShmHistogramSlot* h0_slot = hist_chunk0.ReadHistogram("h_0");
    ASSERT_THAT(h0_slot, NotNull());
    EXPECT_THAT(h0_slot->sample_sum.load(), DoubleEq(1.0));

    const ShmHistogramSlot* h341_slot = hist_chunk1.ReadHistogram("h_341");
    ASSERT_THAT(h341_slot, NotNull());
    EXPECT_THAT(h341_slot->sample_sum.load(), DoubleEq(342.0));
  }
}

// 7. Consolidates 63/64-byte name boundaries, 127/128-byte label boundaries,
// NaN/Inf rejection, and invalid dir.
TEST_F(ShmWriterTest, InputValidationAndFailureResilience) {
  ShmWriter writer(DefaultOptions());
  MappedSegment mapped(test_dir_);
  ASSERT_TRUE(mapped.is_valid());

  std::array<MetricLabel, 1> label_1 = {{{"k1", "v1"}}};
  std::array<MetricLabel, 8> labels_8 = {{{"k1", "v1"},
                                          {"k2", "v2"},
                                          {"k3", "v3"},
                                          {"k4", "v4"},
                                          {"k5", "v5"},
                                          {"k6", "v6"},
                                          {"k7", "v7"},
                                          {"k8", "v8"}}};
  std::array<std::string, 20> label_keys;
  std::array<MetricLabel, 20> labels_20;
  for (int i = 0; i < 20; ++i) {
    label_keys[i] = std::string(1, 'a' + i);
    labels_20[i] = {label_keys[i], "1"};
  }
  writer.IncrementCounter("c0", {}, 1);
  writer.IncrementCounter("c1", label_1, 2);
  writer.IncrementCounter("c8", labels_8, 3);
  writer.IncrementCounter("c20", labels_20, 4);
  EXPECT_EQ(mapped.ReadCounter("c0"), 1);
  EXPECT_EQ(mapped.ReadCounter("c1", label_1), 2);
  EXPECT_EQ(mapped.ReadCounter("c8", labels_8), 3);
  EXPECT_EQ(mapped.ReadCounter("c20", labels_20), 4);
  for (uint32_t i = 0; i < 4; ++i) {
    const ShmTocEntry& entry = mapped.segment()->toc[i];
    EXPECT_GE(entry.offset, sizeof(ShmSegmentLayout));
    EXPECT_LT(entry.offset + entry.size, kSegmentTotalFileSize);
  }

  std::string name_63(63, 'a');
  std::string name_64(64, 'b');
  std::string name_oversized(74, 'c');
  writer.IncrementCounter(name_63, {}, 10);
  writer.IncrementCounter(name_64, {}, 20);
  writer.IncrementCounter(name_oversized, {}, 30);
  EXPECT_EQ(mapped.ReadCounter(name_63), 10);
  EXPECT_EQ(mapped.ReadCounter(name_64), 0);
  EXPECT_EQ(mapped.ReadCounter(name_oversized), 0);

  std::string val_125(125, 'v');
  std::string val_126(126, 'w');
  std::string val_oversized(148, 'x');
  std::array<MetricLabel, 1> label_127 = {{{"k", val_125}}};
  std::array<MetricLabel, 1> label_128 = {{{"k", val_126}}};
  std::array<MetricLabel, 1> label_oversized = {{{"k", val_oversized}}};
  writer.IncrementCounter("b_lbl", label_127, 30);
  writer.IncrementCounter("b_lbl", label_128, 40);
  writer.IncrementCounter("b_lbl", label_oversized, 50);
  EXPECT_EQ(mapped.ReadCounter("b_lbl", label_127), 30);
  EXPECT_EQ(mapped.ReadCounter("b_lbl", label_128), 0);
  EXPECT_EQ(mapped.ReadCounter("b_lbl", label_oversized), 0);

  writer.SetGauge("g_nan", {}, 42.0);
  for (double val : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity()}) {
    writer.ObserveHistogram("h_nan", {}, val);
    writer.SetGauge("g_nan", {}, val);
  }
  const ShmHistogramSlot* nan_histogram = mapped.ReadHistogram("h_nan");
  ASSERT_THAT(nan_histogram, NotNull());
  EXPECT_EQ(nan_histogram->sample_count.load(), 0);
  EXPECT_THAT(mapped.ReadGauge("g_nan"), DoubleEq(42.0));

  writer.ObserveHistogram("h_nan", {}, 10.5);
  writer.ObserveHistogram("h_nan", {},
                          std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(nan_histogram->sample_count.load(), 2);
  EXPECT_THAT(nan_histogram->sample_sum.load(), DoubleEq(10.5));

  std::string regular_file = absl::StrCat(test_dir_, "/regular_file");
  int fd = open(regular_file.c_str(), O_CREAT | O_WRONLY, kDefaultFileMode);
  ASSERT_GE(fd, 0);
  close(fd);

  MetricLabel dummy_label{metric_labels::kDirection,
                          metric_labels::kDirectionPush};
  for (const ShmWriterOptions& opt :
       {ShmWriterOptions{.shm_dir = "", .local_rank = "0"},
        ShmWriterOptions{.shm_dir = test_dir_, .local_rank = ""},
        ShmWriterOptions{.shm_dir = absl::StrCat(regular_file, "/x"),
                         .local_rank = "0"}}) {
    ShmWriter bad_writer(opt);
    bad_writer.IncrementCounter(metric_names::kSentBytesTotal,
                                {&dummy_label, 1}, 10);
    bad_writer.SetGauge("any_g", {&dummy_label, 1}, 1.0);
    bad_writer.ObserveHistogram("any_h", {&dummy_label, 1}, 2.0);
  }

  EXPECT_FALSE(MappedSegment("").is_valid());
  EXPECT_FALSE(MappedSegment(test_dir_, "").is_valid());
  EXPECT_FALSE(MappedSegment(regular_file).is_valid());
}

}  // namespace
}  // namespace tpu_raiden::telemetry

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

#include "tpu_sync/telemetry/shm/shm_layout.h"

#include <atomic>
#include <chrono>  // NOLINT
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>  // NOLINT
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::DoubleEq;

TEST(ShmLayoutTest, StaticLayoutInvariants) {
  EXPECT_TRUE(std::is_standard_layout_v<ShmTocEntry>);
  EXPECT_TRUE(std::is_standard_layout_v<ShmTocHeader>);
  EXPECT_TRUE(std::is_standard_layout_v<ShmHistogramSlot>);
  EXPECT_TRUE(std::is_standard_layout_v<ShmSegmentLayout>);

  EXPECT_TRUE(std::is_trivially_copyable_v<ShmTocEntry>);
  EXPECT_TRUE(std::is_trivially_copyable_v<ShmTocHeader>);
  EXPECT_TRUE(std::is_trivially_copyable_v<ShmHistogramSlot>);
  EXPECT_TRUE(std::is_trivially_copyable_v<ShmSegmentLayout>);

  EXPECT_EQ(alignof(ShmTocHeader), 64);
  EXPECT_EQ(sizeof(ShmTocHeader), 64);
  EXPECT_EQ(alignof(ShmSegmentLayout), 64);
  EXPECT_EQ(sizeof(ShmSegmentLayout), 262208);
  EXPECT_EQ(alignof(ShmTocEntry), 64);
  EXPECT_EQ(sizeof(ShmTocEntry), 256);
  EXPECT_EQ(alignof(ShmHistogramSlot), 64);
  EXPECT_EQ(sizeof(ShmHistogramSlot), 192);

  EXPECT_EQ(sizeof(ShmTocEntry::metric_name), 64);
  EXPECT_EQ(sizeof(ShmTocEntry::encoded_labels), 128);
}

TEST(ShmLayoutTest, StructMemberOffsetsAndNoInternalPadding) {
  // ShmTocEntry offset checks
  EXPECT_EQ(offsetof(ShmTocEntry, metric_name), 0);
  EXPECT_EQ(offsetof(ShmTocEntry, encoded_labels), 64);
  EXPECT_EQ(offsetof(ShmTocEntry, type), 192);
  EXPECT_EQ(offsetof(ShmTocEntry, offset), 196);
  EXPECT_EQ(offsetof(ShmTocEntry, size), 200);
  EXPECT_EQ(offsetof(ShmTocEntry, entry_state), 204);
  EXPECT_EQ(offsetof(ShmTocEntry, padding), 208);
  EXPECT_EQ(sizeof(ShmTocEntry), 256);

  // ShmTocHeader offset checks
  EXPECT_EQ(offsetof(ShmTocHeader, magic), 0);
  EXPECT_EQ(offsetof(ShmTocHeader, version), 4);
  EXPECT_EQ(offsetof(ShmTocHeader, pid), 8);
  EXPECT_EQ(offsetof(ShmTocHeader, toc_entry_count), 16);
  EXPECT_EQ(offsetof(ShmTocHeader, max_toc_entries), 20);
  EXPECT_EQ(offsetof(ShmTocHeader, data_pool_offset), 24);
  EXPECT_EQ(offsetof(ShmTocHeader, data_pool_bytes), 28);
  EXPECT_EQ(offsetof(ShmTocHeader, chunk_index), 32);
  EXPECT_EQ(offsetof(ShmTocHeader, padding), 36);
  EXPECT_EQ(sizeof(ShmTocHeader), 64);

  // ShmHistogramSlot offset checks
  EXPECT_EQ(offsetof(ShmHistogramSlot, sample_count), 0);
  EXPECT_EQ(offsetof(ShmHistogramSlot, sample_sum), 8);
  EXPECT_EQ(offsetof(ShmHistogramSlot, bucket_counts), 16);
  EXPECT_EQ(offsetof(ShmHistogramSlot, padding), 184);
  EXPECT_EQ(sizeof(ShmHistogramSlot), 192);

  // ShmSegmentLayout offset checks
  EXPECT_EQ(offsetof(ShmSegmentLayout, header), 0);
  EXPECT_EQ(offsetof(ShmSegmentLayout, toc), 64);
  EXPECT_EQ(sizeof(ShmSegmentLayout), 262208);
}

TEST(ShmLayoutTest, ConstantsAndOffsets) {
  EXPECT_EQ(kRaidenShmMagic, 0xABCD1234);
  EXPECT_EQ(kSupportedVersion, 2);
  EXPECT_EQ(kShmFilePrefix, "worker_rank_");
  EXPECT_EQ(kShmFileExtension, ".mmap");
  EXPECT_EQ(kMaxTocEntries, 1024);
  EXPECT_EQ(kMaxDataPoolBytes, 65536);
  EXPECT_EQ(kMetricSlotAlignment, 64);
  EXPECT_EQ(kNumHistogramBuckets, 20);

  size_t expected_total_size = sizeof(ShmTocHeader) +
                               (sizeof(ShmTocEntry) * kMaxTocEntries) +
                               kMaxDataPoolBytes;
  EXPECT_EQ(kSegmentTotalFileSize, expected_total_size);
  EXPECT_EQ(kSegmentTotalFileSize, 327744);
  EXPECT_EQ(CalculateChunkFileSize(kMaxTocEntries, kMaxDataPoolBytes),
            expected_total_size);
}

TEST(ShmLayoutTest, TocEntryStateEnumValues) {
  EXPECT_EQ(static_cast<uint32_t>(TocEntryState::kUninitialized), 0);
  EXPECT_EQ(static_cast<uint32_t>(TocEntryState::kWriting), 1);
  EXPECT_EQ(static_cast<uint32_t>(TocEntryState::kCommitted), 2);
}

TEST(ShmLayoutTest, HistogramSingleThreadObservation) {
  ShmHistogramSlot slot;
  EXPECT_EQ(slot.sample_count.load(), 0);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));

  // Value 0.05 falls into bucket 0 (<= 0.1).
  slot.Observe(0.05);
  EXPECT_EQ(slot.sample_count.load(), 1);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.05));
  EXPECT_EQ(slot.bucket_counts[0].load(), 1);
  for (size_t b = 1; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 0);
  }

  // Value 5.5 falls into bucket 6 (<= 10.0, > 5.0).
  slot.Observe(5.5);
  EXPECT_EQ(slot.sample_count.load(), 2);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(5.55));
  EXPECT_EQ(slot.bucket_counts[0].load(), 1);
  EXPECT_EQ(slot.bucket_counts[6].load(), 1);
  for (size_t b = 1; b <= kNumHistogramBuckets; ++b) {
    if (b != 6) {
      EXPECT_EQ(slot.bucket_counts[b].load(), 0);
    }
  }

  // Value 60000.0 exceeds highest boundary (50000.0), falling into bucket 20
  // (+Inf).
  slot.Observe(60000.0);
  EXPECT_EQ(slot.sample_count.load(), 3);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(60005.55));
  EXPECT_EQ(slot.bucket_counts[0].load(), 1);
  EXPECT_EQ(slot.bucket_counts[6].load(), 1);
  EXPECT_EQ(slot.bucket_counts[kNumHistogramBuckets].load(), 1);

  // Cumulative view verification (as computed by exporter)
  uint64_t cumulative_sum = 0;
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    cumulative_sum += slot.bucket_counts[b].load();
    if (b < 6) {
      EXPECT_EQ(cumulative_sum, 1);
    } else if (b < kNumHistogramBuckets) {
      EXPECT_EQ(cumulative_sum, 2);
    } else {
      EXPECT_EQ(cumulative_sum, 3);
    }
  }
}

TEST(ShmLayoutTest, HistogramNonFiniteObservationsIgnored) {
  ShmHistogramSlot slot;
  slot.Observe(std::numeric_limits<double>::quiet_NaN());
  slot.Observe(std::numeric_limits<double>::infinity());
  slot.Observe(-std::numeric_limits<double>::infinity());

  EXPECT_EQ(slot.sample_count.load(), 0);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 0);
  }
}

TEST(ShmLayoutTest, HistogramConcurrentObservations) {
  ShmHistogramSlot slot;
  constexpr int kNumThreads = 8;
  constexpr int kNumIters = 1000;
  constexpr double kObsValue = 0.5;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&slot]() {
      for (int i = 0; i < kNumIters; ++i) {
        slot.Observe(kObsValue);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(slot.sample_count.load(), kNumThreads * kNumIters);
  EXPECT_THAT(slot.sample_sum.load(),
              DoubleEq(kNumThreads * kNumIters * kObsValue));
  // 0.5 falls into bucket 2 (0.25 < 0.5 <= 0.5).
  EXPECT_EQ(slot.bucket_counts[0].load(), 0);
  EXPECT_EQ(slot.bucket_counts[1].load(), 0);
  EXPECT_EQ(slot.bucket_counts[2].load(), kNumThreads * kNumIters);
  for (size_t b = 3; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 0);
  }
}

TEST(ShmLayoutTest, HistogramBucketSemantics) {
  ShmHistogramSlot slot;
  slot.Observe(0.1);         // Exactly on bucket 0 boundary
  slot.Observe(0.1000001);   // Just above bucket 0 -> bucket 1
  slot.Observe(50000.0);     // Exactly on bucket 19 boundary
  slot.Observe(50000.0001);  // Just above bucket 19 -> bucket 20 (+Inf)

  EXPECT_EQ(slot.sample_count.load(), 4);
  EXPECT_EQ(slot.bucket_counts[0].load(), 1);
  EXPECT_EQ(slot.bucket_counts[1].load(), 1);
  EXPECT_EQ(slot.bucket_counts[19].load(), 1);
  EXPECT_EQ(slot.bucket_counts[20].load(), 1);

  // Cumulative view is monotonic
  uint64_t prev_cumulative = 0;
  uint64_t cumulative = 0;
  for (size_t i = 0; i <= kNumHistogramBuckets; ++i) {
    cumulative += slot.bucket_counts[i].load();
    EXPECT_GE(cumulative, prev_cumulative);
    prev_cumulative = cumulative;
  }
  EXPECT_EQ(cumulative, 4);
}

TEST(ShmLayoutTest, AdversarialNonFiniteAndEdgeValues) {
  ShmHistogramSlot slot;

  // 1. NaN variants
  slot.Observe(std::numeric_limits<double>::quiet_NaN());
  slot.Observe(std::numeric_limits<double>::signaling_NaN());
  slot.Observe(-std::numeric_limits<double>::quiet_NaN());

  // 2. Infinities
  slot.Observe(std::numeric_limits<double>::infinity());
  slot.Observe(-std::numeric_limits<double>::infinity());

  EXPECT_EQ(slot.sample_count.load(), 0);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    EXPECT_EQ(slot.bucket_counts[b].load(), 0);
  }

  // 3. Signed Zeros
  slot.Observe(+0.0);
  slot.Observe(-0.0);
  EXPECT_EQ(slot.sample_count.load(), 2);
  EXPECT_THAT(slot.sample_sum.load(), DoubleEq(0.0));
  // 0.0 <= 0.1 so placed in bucket 0
  EXPECT_EQ(slot.bucket_counts[0].load(), 2);

  // 4. Subnormals
  slot.Observe(std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(slot.sample_count.load(), 3);
  EXPECT_GT(slot.sample_sum.load(), 0.0);
  EXPECT_EQ(slot.bucket_counts[0].load(), 3);

  // 5. Negative finite numbers
  slot.Observe(-50.0);
  EXPECT_EQ(slot.sample_count.load(), 4);
  EXPECT_LT(slot.sample_sum.load(), 0.0);
  // -50.0 <= 0.1, placed in bucket 0
  EXPECT_EQ(slot.bucket_counts[0].load(), 4);

  // 6. Extreme max finite number
  ShmHistogramSlot max_slot;
  max_slot.Observe(std::numeric_limits<double>::max());
  EXPECT_EQ(max_slot.sample_count.load(), 1);
  EXPECT_EQ(max_slot.sample_sum.load(), std::numeric_limits<double>::max());
  // max exceeds 50000.0, placed in +Inf bucket (index 20)
  for (size_t b = 0; b < kNumHistogramBuckets; ++b) {
    EXPECT_EQ(max_slot.bucket_counts[b].load(), 0);
  }
  EXPECT_EQ(max_slot.bucket_counts[kNumHistogramBuckets].load(), 1);

  // 7. Overflow to +Inf in sum accumulation
  max_slot.Observe(std::numeric_limits<double>::max());
  EXPECT_EQ(max_slot.sample_count.load(), 2);
  EXPECT_TRUE(std::isinf(max_slot.sample_sum.load()));
}

TEST(ShmLayoutTest, HighContentionThreadScalingAndCASLivelock) {
  ShmHistogramSlot slot;
  constexpr int kNumThreads = 32;
  constexpr int kNumIters = 25000;  // Total 800,000 observations
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
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(slot.sample_count.load(),
            static_cast<uint64_t>(kNumThreads) * kNumIters);

  uint64_t total_bucket_samples = 0;
  for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
    total_bucket_samples += slot.bucket_counts[b].load();
  }
  EXPECT_EQ(total_bucket_samples,
            static_cast<uint64_t>(kNumThreads) * kNumIters);
}

TEST(ShmLayoutTest, FloatingPointPrecisionAndAbsorptionStress) {
  ShmHistogramSlot slot;

  // Case 1: Absorption under extreme dynamic range (1e16 + 1.0)
  slot.Observe(1e16);
  constexpr int kSmallAdds = 10000;
  for (int i = 0; i < kSmallAdds; ++i) {
    slot.Observe(1.0);
  }
  EXPECT_EQ(slot.sample_count.load(), 1 + kSmallAdds);

  // Case 2: Multi-precision comparison for normal dynamic range
  ShmHistogramSlot normal_slot;
  constexpr int kNormalIters = 100000;
  for (int i = 0; i < kNormalIters; ++i) {
    normal_slot.Observe(0.125);  // Exact power of 2 fraction (1/8)
  }
  EXPECT_EQ(normal_slot.sample_count.load(), kNormalIters);
  EXPECT_THAT(normal_slot.sample_sum.load(), DoubleEq(kNormalIters * 0.125));
}

TEST(ShmLayoutTest, ConcurrentReaderCumulativeMonotonicityRace) {
  ShmHistogramSlot slot;
  constexpr int kNumWriters = 8;
  constexpr int kNumReaders = 4;
  constexpr int kWritesPerThread = 50000;
  std::atomic<bool> stop_readers{false};
  std::atomic<bool> start_signal{false};
  std::atomic<uint64_t> non_monotonic_snapshots{0};
  std::atomic<uint64_t> total_snapshots{0};

  std::vector<std::thread> writers;
  writers.reserve(kNumWriters);
  for (int w = 0; w < kNumWriters; ++w) {
    writers.emplace_back([&slot, &start_signal, w]() {
      while (!start_signal.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kWritesPerThread; ++i) {
        double val = ((w * 31 + i) % 22) * 2500.0 + 0.05;
        slot.Observe(val);
      }
    });
  }

  std::vector<std::thread> readers;
  readers.reserve(kNumReaders);
  for (int r = 0; r < kNumReaders; ++r) {
    readers.emplace_back([&slot, &start_signal, &stop_readers,
                          &non_monotonic_snapshots, &total_snapshots]() {
      while (!start_signal.load(std::memory_order_acquire)) {
      }
      uint64_t local_counts[kNumHistogramBuckets + 1];
      while (!stop_readers.load(std::memory_order_relaxed)) {
        for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
          local_counts[b] =
              slot.bucket_counts[b].load(std::memory_order_relaxed);
        }
        total_snapshots.fetch_add(1, std::memory_order_relaxed);
        // Compute cumulative counts on-the-fly and check monotonicity
        uint64_t prev_cum = 0;
        uint64_t cum = 0;
        bool monotonic = true;
        for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
          cum += local_counts[b];
          if (cum < prev_cum) {
            monotonic = false;
            break;
          }
          prev_cum = cum;
        }
        if (!monotonic) {
          non_monotonic_snapshots.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start_signal.store(true, std::memory_order_release);
  for (auto& w : writers) {
    w.join();
  }
  stop_readers.store(true, std::memory_order_release);
  for (auto& r : readers) {
    r.join();
  }

  EXPECT_EQ(non_monotonic_snapshots.load(), 0);
  EXPECT_GT(total_snapshots.load(), 0);
}

TEST(ShmLayoutTest, BinarySerializationAndNoGaps) {
  alignas(64) uint8_t raw_buffer[sizeof(ShmTocEntry)];
  std::memset(raw_buffer, 0xAA, sizeof(raw_buffer));

  auto* entry = reinterpret_cast<ShmTocEntry*>(raw_buffer);
  std::memset(entry->metric_name, 'A', sizeof(entry->metric_name));
  entry->metric_name[sizeof(entry->metric_name) - 1] = '\0';
  std::memset(entry->encoded_labels, 'B', sizeof(entry->encoded_labels));
  entry->encoded_labels[sizeof(entry->encoded_labels) - 1] = '\0';
  entry->type = MetricType::kHistogram;
  entry->offset = 0x12345678;
  entry->size = 0x87654321;
  entry->entry_state.store(TocEntryState::kCommitted);

  EXPECT_EQ(raw_buffer[0], 'A');
  EXPECT_EQ(raw_buffer[63], '\0');
  EXPECT_EQ(raw_buffer[64], 'B');
  EXPECT_EQ(raw_buffer[191], '\0');
  EXPECT_EQ(*reinterpret_cast<const MetricType*>(&raw_buffer[192]),
            MetricType::kHistogram);
  EXPECT_EQ(*reinterpret_cast<const uint32_t*>(&raw_buffer[196]), 0x12345678);
  EXPECT_EQ(*reinterpret_cast<const uint32_t*>(&raw_buffer[200]), 0x87654321);
  EXPECT_EQ(*reinterpret_cast<const TocEntryState*>(&raw_buffer[204]),
            TocEntryState::kCommitted);

  EXPECT_EQ(sizeof(ShmHistogramSlot), 8 + 8 + (21 * 8) + 8);
}

TEST(ShmLayoutTest, CacheLineBoundaryAndFalseSharingAnalysis) {
  EXPECT_EQ(alignof(ShmTocHeader), 64);
  EXPECT_EQ(sizeof(ShmTocHeader), 64);
  EXPECT_EQ(sizeof(ShmTocHeader) % 64, 0);

  EXPECT_EQ(alignof(ShmSegmentLayout), 64);
  EXPECT_EQ(sizeof(ShmSegmentLayout), 262208);
  EXPECT_EQ(sizeof(ShmSegmentLayout) % 64, 0);

  EXPECT_EQ(kSegmentTotalFileSize, 327744);
  EXPECT_EQ(kSegmentTotalFileSize % 64, 0);
  EXPECT_EQ(kMetricSlotAlignment, 64);
}

TEST(ShmLayoutTest, FullSegmentArrayNoOverlap) {
  auto segment = std::make_unique<ShmSegmentLayout>();
  std::memset(segment.get(), 0, sizeof(ShmSegmentLayout));

  segment->header.magic.store(kRaidenShmMagic);
  segment->header.version = kSupportedVersion;
  segment->header.pid = 123456;
  segment->header.toc_entry_count = kMaxTocEntries;
  segment->header.max_toc_entries = kMaxTocEntries;
  segment->header.data_pool_offset = sizeof(ShmSegmentLayout);
  segment->header.data_pool_bytes = kMaxDataPoolBytes;

  for (size_t i = 0; i < kMaxTocEntries; ++i) {
    snprintf(segment->toc[i].metric_name, sizeof(segment->toc[i].metric_name),
             "metric_name_stream_%04zu", i);
    snprintf(segment->toc[i].encoded_labels,
             sizeof(segment->toc[i].encoded_labels),
             "cluster=tpuv4,task=%04zu,direction=push", i);
    segment->toc[i].type =
        (i % 2 == 0) ? MetricType::kCounter : MetricType::kHistogram;
    segment->toc[i].offset =
        static_cast<uint32_t>(segment->header.data_pool_offset + (i * 192));
    segment->toc[i].size = (i % 2 == 0) ? 8 : 192;
    segment->toc[i].entry_state.store(TocEntryState::kCommitted);
  }

  EXPECT_EQ(segment->header.magic.load(), kRaidenShmMagic);
  EXPECT_EQ(segment->header.version, kSupportedVersion);
  EXPECT_EQ(segment->header.pid, 123456);
  EXPECT_EQ(segment->header.toc_entry_count, kMaxTocEntries);

  for (size_t i = 0; i < kMaxTocEntries; ++i) {
    char expected_name[64];
    char expected_labels[128];
    snprintf(expected_name, sizeof(expected_name), "metric_name_stream_%04zu",
             i);
    snprintf(expected_labels, sizeof(expected_labels),
             "cluster=tpuv4,task=%04zu,direction=push", i);

    EXPECT_STREQ(segment->toc[i].metric_name, expected_name);
    EXPECT_STREQ(segment->toc[i].encoded_labels, expected_labels);
    EXPECT_EQ(segment->toc[i].offset,
              segment->header.data_pool_offset + (i * 192));
    EXPECT_EQ(segment->toc[i].size, (i % 2 == 0) ? 8 : 192);
    EXPECT_EQ(segment->toc[i].entry_state.load(), TocEntryState::kCommitted);
  }
}

}  // namespace
}  // namespace tpu_raiden::telemetry

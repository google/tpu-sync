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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_LAYOUT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_LAYOUT_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

inline constexpr uint32_t kRaidenShmMagic = 0xABCD1234;
inline constexpr uint32_t kSupportedVersion = 2;
inline constexpr absl::string_view kShmFilePrefix = "worker_rank_";
inline constexpr absl::string_view kShmFileExtension = ".mmap";

inline constexpr size_t kMaxTocEntries = 1024;
// Data pool capacity for metric slots in a single chunk (64 KB).
// Sized to accommodate 1024 64-byte aligned counter/gauge slots, or up to 341
// 192-byte histogram slots. Multi-chunk dynamic expansion allocates additional
// chunk files on demand if data pool capacity is exceeded.
inline constexpr size_t kMaxDataPoolBytes = 64 * 1024;  // 64 KB
inline constexpr size_t kMetricSlotAlignment = 64;
inline constexpr size_t kNumHistogramBuckets =
    std::size(kDefaultHistogramBuckets);

// State machine transitions for a Table of Contents (TOC) entry descriptor:
// - kUninitialized: Entry slot is empty or unallocated.
// - kWriting: Writer is populating entry fields (name, labels, type, offset).
// - kCommitted: Entry is fully published and valid for reader consumption.
//
// Writers transition from kWriting to kCommitted using release memory ordering
// after all entry fields and data pool slot initialization are complete.
// Readers load entry_state using acquire memory ordering to safely synchronize
// and read published metric descriptors and slot contents without locks.
enum class TocEntryState : uint32_t {
  kUninitialized = 0,
  kWriting = 1,
  kCommitted = 2,
};

// Table of Contents entry descriptor for a single metric stream.
//
// Publication protocol: The writer fills metric_name, encoded_labels, type,
// offset, and size, then publishes the entry by setting entry_state to
// TocEntryState::kCommitted with release memory ordering. Readers must load
// entry_state with acquire memory ordering before accessing the metric slot at
// offset.
struct alignas(64) ShmTocEntry {
  char metric_name[64];
  char encoded_labels[128];
  MetricType type;
  // Explicitly reserve 3 bytes to ensure 64-byte alignment of the TOC entry.
  uint8_t reserved[3]{};
  uint32_t offset;
  uint32_t size;
  std::atomic<TocEntryState> entry_state{TocEntryState::kUninitialized};
  uint8_t padding[48]{};
};

// 64-byte aligned header for the shared-memory telemetry segment.
//
// Initialization and publication protocol:
// The creator process zeroes or mmaps the shared memory segment, sets all
// non-atomic fields (version, pid, max_toc_entries, data_pool_offset,
// chunk_index), and finally publishes the segment by writing kRaidenShmMagic to
// magic using release memory ordering. Readers must verify that magic loaded
// with acquire memory ordering equals kRaidenShmMagic before reading any header
// or TOC fields.
//
// Data pool metric slots are aligned to kMetricSlotAlignment (64 bytes) to
// prevent intra-process false sharing between multiple threads concurrently
// updating distinct metric streams on different CPU cores.
struct alignas(64) ShmTocHeader {
  std::atomic<uint32_t> magic{0};
  uint32_t version{kSupportedVersion};
  int64_t pid{0};
  std::atomic<uint32_t> toc_entry_count{0};
  uint32_t max_toc_entries{kMaxTocEntries};
  uint32_t data_pool_offset{0};
  std::atomic<uint32_t> data_pool_bytes{0};
  uint32_t chunk_index{0};
  uint8_t padding[28]{};
};

// Lock-free shared-memory slot for a single histogram metric stream.
//
// Uses the standard uniform histogram bucket distribution
// (`kDefaultHistogramBuckets`) to ensure a fixed 192-byte standard layout with
// 64-byte alignment and zero serialization overhead across heterogeneous
// exporter pipelines.
//
// Stores non-cumulative (differential) bucket counts to eliminate write
// amplification (3 atomic operations per Observe instead of 23) and prevent
// reader-writer monotonicity race conditions. Exporters compute cumulative
// bucket counts on-the-fly, which is inherently monotonic since all bucket
// counts are non-negative.
struct alignas(64) ShmHistogramSlot {
  std::atomic<uint64_t> sample_count{0};
  std::atomic<double> sample_sum{0.0};
  std::atomic<uint64_t> bucket_counts[kNumHistogramBuckets + 1]{};
  uint8_t padding[8]{};

  // Records an observation value into the histogram. Non-finite values (NaN,
  // +/-Inf) are ignored. Updates sample count, sample sum, and the specific
  // non-cumulative bucket count atomically with relaxed memory ordering.
  void Observe(double value) {
    if (!std::isfinite(value)) {
      return;
    }
    sample_count.fetch_add(1, std::memory_order_relaxed);
    sample_sum.fetch_add(value, std::memory_order_relaxed);
    auto it = std::lower_bound(std::begin(kDefaultHistogramBuckets),
                               std::end(kDefaultHistogramBuckets), value);
    size_t bucket_idx = static_cast<size_t>(
        std::distance(std::begin(kDefaultHistogramBuckets), it));
    bucket_counts[bucket_idx].fetch_add(1, std::memory_order_relaxed);
  }
};

// Complete 64-byte aligned segment layout containing header and TOC entries.
struct alignas(64) ShmSegmentLayout {
  ShmTocHeader header;
  ShmTocEntry toc[kMaxTocEntries];
};

// Calculates the total shared-memory segment file size in bytes required for
// `max_toc_entries` metric stream descriptors and `data_pool_bytes` of metric
// slot storage. Returns the total segment size including header, TOC, and data
// pool.
constexpr size_t CalculateChunkFileSize(size_t max_toc_entries,
                                        size_t data_pool_bytes) {
  return sizeof(ShmTocHeader) + (sizeof(ShmTocEntry) * max_toc_entries) +
         data_pool_bytes;
}

inline constexpr size_t kSegmentTotalFileSize =
    CalculateChunkFileSize(kMaxTocEntries, kMaxDataPoolBytes);

static_assert(std::is_standard_layout_v<ShmTocEntry> &&
              std::is_standard_layout_v<ShmTocHeader> &&
              std::is_standard_layout_v<ShmHistogramSlot> &&
              std::is_standard_layout_v<ShmSegmentLayout>);

static_assert(std::is_trivially_copyable_v<ShmTocEntry> &&
              std::is_trivially_copyable_v<ShmTocHeader> &&
              std::is_trivially_copyable_v<ShmHistogramSlot> &&
              std::is_trivially_copyable_v<ShmSegmentLayout>);

static_assert(alignof(ShmTocHeader) == 64 && sizeof(ShmTocHeader) == 64);
static_assert(alignof(ShmTocEntry) == 64 && sizeof(ShmTocEntry) == 256);
static_assert(alignof(ShmHistogramSlot) == 64 &&
              sizeof(ShmHistogramSlot) == 192);
static_assert(alignof(ShmSegmentLayout) == 64 &&
              sizeof(ShmSegmentLayout) == 262208);

static_assert(std::atomic<uint64_t>::is_always_lock_free &&
              std::atomic<int64_t>::is_always_lock_free &&
              std::atomic<double>::is_always_lock_free &&
              std::atomic<uint32_t>::is_always_lock_free &&
              std::atomic<TocEntryState>::is_always_lock_free);

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_LAYOUT_H_

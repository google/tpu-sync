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

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

namespace {

constexpr absl::string_view kCountSuffix = "_count";
constexpr absl::string_view kSumSuffix = "_sum";
constexpr absl::string_view kBucketPrefix = "_bucket/";
constexpr absl::string_view kLePrefix = "le=";
constexpr absl::string_view kDelimitedLePrefix = ";le=";
constexpr absl::string_view kInfBound = "+Inf";

const absl::NoDestructor<std::array<std::string, kNumHistogramBuckets + 1>>
    kDefaultBucketBoundaries([] {
      std::array<std::string, kNumHistogramBuckets + 1> array;
      for (size_t i = 0; i < kNumHistogramBuckets; ++i) {
        array[i] = absl::StrCat(kDefaultHistogramBuckets[i]);
      }
      array[kNumHistogramBuckets] = kInfBound;
      return array;
    }());

// Compile-time upper bound on formatted metric key length derived directly
// from ShmTocEntry buffer capacities in shm_layout.h:
// - metric_name: sizeof(ShmTocEntry::metric_name) (64 bytes)
// - encoded_labels: sizeof(ShmTocEntry::encoded_labels) (128 bytes)
// - longest formatting overhead: "_bucket/;le=+Inf" (16 bytes)
// Headroom buffer is added to prevent any reallocation under future expansions.
constexpr size_t kMaxHistogramSuffixLength =
    kBucketPrefix.size() + kDelimitedLePrefix.size() + kInfBound.size();
constexpr size_t kKeyBufferHeadroomBytes = 64;

constexpr size_t kKeyBufferInitialCapacity =
    sizeof(ShmTocEntry::metric_name) + sizeof(ShmTocEntry::encoded_labels) +
    kMaxHistogramSuffixLength + kKeyBufferHeadroomBytes;

// Validates that the memory region beginning at `slot_bytes` has at least
// `sizeof(T)` byte capacity and meets the `alignof(T)` natural alignment
// required for atomic operations on slot type `T`.
template <typename T>
bool IsSlotValid(const uint8_t* slot_bytes, uint32_t entry_size) {
  return entry_size >= sizeof(T) &&
         (reinterpret_cast<uintptr_t>(slot_bytes) % alignof(T) == 0);
}

// Aggregates metrics from a mapped shared-memory segment into `totals`.
//
// Preconditions:
//   - `raw_segment` must be non-null and aligned to
//   `alignof(ShmSegmentLayout)`.
//   - `raw_segment` must point to at least `kSegmentTotalFileSize` bytes of
//     valid readable memory.
//
// Reads TOC entries using acquire barriers to guard against uninitialized or
// partially written entries. Defensive checks reject corrupt offsets,
// non-null-terminated strings, and invalid scalar values.
void AggregateSegment(const void* raw_segment,
                      absl::flat_hash_map<std::string, double>& totals) {
  if (raw_segment == nullptr ||
      (reinterpret_cast<uintptr_t>(raw_segment) % alignof(ShmSegmentLayout) !=
       0)) {
    return;
  }
  const ShmSegmentLayout* segment =
      static_cast<const ShmSegmentLayout*>(raw_segment);

  if (segment->header.magic.load(std::memory_order_acquire) !=
      kRaidenShmMagic) {
    return;
  }
  if (segment->header.max_toc_entries != kMaxTocEntries) {
    return;
  }

  const uint32_t data_pool_offset = segment->header.data_pool_offset;
  if (data_pool_offset < sizeof(ShmSegmentLayout) ||
      data_pool_offset >= kSegmentTotalFileSize ||
      data_pool_offset % kMetricSlotAlignment != 0) {
    return;
  }
  const size_t num_entries =
      segment->header.toc_entry_count.load(std::memory_order_acquire);
  if (num_entries > kMaxTocEntries) {
    return;
  }

  std::string key_buffer;
  key_buffer.reserve(kKeyBufferInitialCapacity);

  for (size_t i = 0; i < num_entries; ++i) {
    const ShmTocEntry& toc_entry = segment->toc[i];
    if (toc_entry.entry_state.load(std::memory_order_acquire) !=
        TocEntryState::kCommitted) {
      continue;
    }

    const MetricType metric_type = toc_entry.type;
    const uint32_t entry_offset = toc_entry.offset;
    const uint32_t entry_size = toc_entry.size;

    if (metric_type != MetricType::kCounter &&
        metric_type != MetricType::kGauge &&
        metric_type != MetricType::kHistogram) {
      continue;
    }
    if (entry_offset < data_pool_offset ||
        static_cast<uint64_t>(entry_offset) + entry_size >
            kSegmentTotalFileSize) {
      continue;
    }

    const void* const name_nul =
        std::memchr(toc_entry.metric_name, '\0', sizeof(toc_entry.metric_name));
    if (name_nul == nullptr || name_nul == toc_entry.metric_name) {
      continue;
    }
    const absl::string_view metric_name(
        toc_entry.metric_name,
        static_cast<const char*>(name_nul) - toc_entry.metric_name);

    const void* const labels_nul = std::memchr(
        toc_entry.encoded_labels, '\0', sizeof(toc_entry.encoded_labels));
    if (labels_nul == nullptr) {
      continue;
    }
    const absl::string_view encoded_labels(
        toc_entry.encoded_labels,
        static_cast<const char*>(labels_nul) - toc_entry.encoded_labels);

    const uint8_t* const slot_bytes =
        reinterpret_cast<const uint8_t*>(segment) + entry_offset;

    if (metric_type != MetricType::kHistogram) {
      key_buffer.clear();
      key_buffer.append(metric_name);
      if (!encoded_labels.empty()) {
        key_buffer.push_back('/');
        key_buffer.append(encoded_labels);
      }
    }

    switch (metric_type) {
      case MetricType::kCounter: {
        if (IsSlotValid<std::atomic<uint64_t>>(slot_bytes, entry_size)) {
          totals[key_buffer] += static_cast<double>(
              reinterpret_cast<const std::atomic<uint64_t>*>(slot_bytes)
                  ->load(std::memory_order_relaxed));
        }
        break;
      }
      case MetricType::kGauge: {
        if (IsSlotValid<std::atomic<double>>(slot_bytes, entry_size)) {
          const double gauge_value =
              reinterpret_cast<const std::atomic<double>*>(slot_bytes)
                  ->load(std::memory_order_relaxed);
          if (std::isfinite(gauge_value)) {
            totals[key_buffer] += gauge_value;
          }
        }
        break;
      }
      case MetricType::kHistogram: {
        if (IsSlotValid<ShmHistogramSlot>(slot_bytes, entry_size)) {
          const ShmHistogramSlot* const histogram_slot =
              reinterpret_cast<const ShmHistogramSlot*>(slot_bytes);
          const double sum =
              histogram_slot->sample_sum.load(std::memory_order_relaxed);
          if (!std::isfinite(sum)) {
            break;
          }

          auto append_histogram_scalar = [&](absl::string_view suffix,
                                             double value) {
            key_buffer.clear();
            key_buffer.append(metric_name);
            key_buffer.append(suffix);
            if (!encoded_labels.empty()) {
              key_buffer.push_back('/');
              key_buffer.append(encoded_labels);
            }
            totals[key_buffer] += value;
          };
          append_histogram_scalar(
              kCountSuffix,
              static_cast<double>(histogram_slot->sample_count.load(
                  std::memory_order_relaxed)));
          append_histogram_scalar(kSumSuffix, sum);

          key_buffer.clear();
          key_buffer.append(metric_name);
          key_buffer.append(kBucketPrefix);
          if (!encoded_labels.empty()) {
            key_buffer.append(encoded_labels);
            key_buffer.append(kDelimitedLePrefix);
          } else {
            key_buffer.append(kLePrefix);
          }
          const size_t base_bucket_key_length = key_buffer.size();

          uint64_t cumulative_count = 0;
          for (size_t bucket_index = 0; bucket_index <= kNumHistogramBuckets;
               ++bucket_index) {
            cumulative_count +=
                histogram_slot->bucket_counts[bucket_index].load(
                    std::memory_order_relaxed);
            key_buffer.resize(base_bucket_key_length);
            key_buffer.append((*kDefaultBucketBoundaries)[bucket_index]);
            totals[key_buffer] += static_cast<double>(cumulative_count);
          }
        }
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace

ShmCollector::ShmCollector(ShmCollectorOptions options)
    : options_(std::move(options)) {
  CHECK(!options_.shm_dir.empty())
      << "ShmCollector requires a non-empty shm_dir";
}

void ShmCollector::CollectMetrics(
    absl::flat_hash_map<std::string, double>& totals) const {
  totals.clear();

  DIR* const dir_stream = opendir(options_.shm_dir.c_str());
  if (dir_stream == nullptr) {
    PLOG_EVERY_N_SEC(WARNING, 10)
        << "Failed to open shared-memory directory " << options_.shm_dir;
    return;
  }
  absl::Cleanup close_dir = [dir_stream] { closedir(dir_stream); };

  while (const dirent* dir_entry = readdir(dir_stream)) {
    absl::string_view filename(dir_entry->d_name);
    const bool is_tmp = absl::EndsWith(filename, kShmTmpFileExtension);
    if (!absl::StartsWith(filename, kShmFilePrefix) ||
        (!absl::EndsWith(filename, kShmFileExtension) && !is_tmp)) {
      continue;
    }

    const int fd = openat(dirfd(dir_stream), dir_entry->d_name,
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
      continue;
    }
    absl::Cleanup close_fd = [fd] { close(fd); };

    struct stat file_stat{};
    if (fstat(fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
      continue;
    }

    if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
      // Process is dead. Dead worker metrics are intentionally discarded upon
      // reaping; only metrics from live processes are aggregated.
      // Re-verify fstat under lock: if already unlinked by a concurrent reaper,
      // abort early.
      if (fstat(fd, &file_stat) != 0 || file_stat.st_nlink == 0) {
        continue;
      }
      // Security note: The shared memory directory is expected to be protected
      // with sticky-bit or restricted permissions to prevent unprivileged
      // symlink replacement attacks. We verify inode equality under exclusive
      // lock and use unlinkat() relative to dir_stream to avoid path traversal
      // TOCTOU.
      struct stat current_stat{};
      if (fstatat(dirfd(dir_stream), dir_entry->d_name, &current_stat,
                  AT_SYMLINK_NOFOLLOW) == 0 &&
          current_stat.st_ino == file_stat.st_ino &&
          current_stat.st_dev == file_stat.st_dev) {
        if (unlinkat(dirfd(dir_stream), dir_entry->d_name, 0) != 0 &&
            errno != ENOENT) {
          PLOG_EVERY_N_SEC(WARNING, 10)
              << "Failed to unlink dead worker shared-memory segment at "
              << options_.shm_dir << "/" << filename;
        }
      }
      // Do NOT call flock(LOCK_UN); close_fd will release the lock.
    } else if (!is_tmp && flock(fd, LOCK_SH | LOCK_NB) == 0) {
      // Memory Safety / Fault Tolerance:
      // In this architecture, TPU workers and the collector operate in a
      // trusted local domain. Shared advisory locks protect against collecting
      // partially initialized segments. We re-verify file size and link count
      // under the lock before mapping to guard against undersized or
      // concurrently unlinked dead worker files.
      if (fstat(fd, &file_stat) != 0 || file_stat.st_nlink == 0 ||
          file_stat.st_size < static_cast<off_t>(kSegmentTotalFileSize)) {
        continue;
      }
      void* const mapped_address =
          mmap(nullptr, kSegmentTotalFileSize, PROT_READ, MAP_SHARED, fd, 0);
      if (mapped_address == MAP_FAILED) {
        PLOG_EVERY_N_SEC(WARNING, 10)
            << "Failed to mmap shared-memory segment at " << options_.shm_dir
            << "/" << filename;
        continue;
      }
      absl::Cleanup unmap_segment = [mapped_address] {
        munmap(mapped_address, kSegmentTotalFileSize);
      };
      AggregateSegment(mapped_address, totals);
    }
  }
}

}  // namespace tpu_raiden::telemetry

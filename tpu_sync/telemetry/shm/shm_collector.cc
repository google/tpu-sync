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
#include <stdio.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

namespace {

// Validates that the memory region beginning at `slot_bytes` has at least
// `sizeof(T)` byte capacity and meets the `alignof(T)` natural alignment
// required for atomic operations on slot type `T`.
template <typename T>
bool IsSlotValid(const uint8_t* slot_bytes, uint32_t entry_size) {
  return entry_size >= sizeof(T) &&
         (reinterpret_cast<uintptr_t>(slot_bytes) % alignof(T) == 0);
}

// Retrieves a reference to an existing metric value in `map` without heap
// allocations via string_view lookup, or emplacing a new entry if absent.
template <typename ValueType>
ValueType& GetOrEmplaceMetric(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, ValueType>>& map,
    absl::string_view metric_name, absl::string_view encoded_labels) {
  absl::flat_hash_map<std::string, ValueType>& label_map =
      map.try_emplace(metric_name).first->second;
  return label_map.try_emplace(encoded_labels).first->second;
}

// Aggregates metrics from a mapped shared-memory segment into `metrics`.
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
void AggregateSegment(const void* raw_segment, AggregatedMetrics& metrics) {
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
    absl::string_view metric_name(
        toc_entry.metric_name,
        static_cast<const char*>(name_nul) - toc_entry.metric_name);

    const void* const labels_nul = std::memchr(
        toc_entry.encoded_labels, '\0', sizeof(toc_entry.encoded_labels));
    if (labels_nul == nullptr) {
      continue;
    }
    absl::string_view encoded_labels(
        toc_entry.encoded_labels,
        static_cast<const char*>(labels_nul) - toc_entry.encoded_labels);

    const uint8_t* const slot_bytes =
        reinterpret_cast<const uint8_t*>(segment) + entry_offset;

    switch (metric_type) {
      case MetricType::kCounter: {
        if (IsSlotValid<std::atomic<uint64_t>>(slot_bytes, entry_size)) {
          GetOrEmplaceMetric(metrics.counters, metric_name, encoded_labels) +=
              reinterpret_cast<const std::atomic<uint64_t>*>(slot_bytes)
                  ->load(std::memory_order_relaxed);
        }
        break;
      }
      case MetricType::kGauge: {
        if (IsSlotValid<std::atomic<double>>(slot_bytes, entry_size)) {
          const double gauge_value =
              reinterpret_cast<const std::atomic<double>*>(slot_bytes)
                  ->load(std::memory_order_relaxed);
          if (std::isfinite(gauge_value)) {
            GetOrEmplaceMetric(metrics.gauges, metric_name, encoded_labels) +=
                gauge_value;
          }
        }
        break;
      }
      case MetricType::kHistogram: {
        if (IsSlotValid<ShmHistogramSlot>(slot_bytes, entry_size)) {
          const ShmHistogramSlot* const histogram_slot =
              reinterpret_cast<const ShmHistogramSlot*>(slot_bytes);
          const uint64_t count =
              histogram_slot->sample_count.load(std::memory_order_relaxed);
          const double sum =
              histogram_slot->sample_sum.load(std::memory_order_relaxed);
          if (!std::isfinite(sum)) {
            break;
          }

          HistogramData& hist_data = GetOrEmplaceMetric(
              metrics.histograms, metric_name, encoded_labels);
          hist_data.sample_count += count;
          hist_data.sample_sum += sum;

          for (size_t bucket_index = 0;
               bucket_index < hist_data.bucket_counts.size(); ++bucket_index) {
            hist_data.bucket_counts[bucket_index] +=
                histogram_slot->bucket_counts[bucket_index].load(
                    std::memory_order_relaxed);
          }
        }
        break;
      }
    }
  }
}

}  // namespace

ShmCollector::ShmCollector(ShmCollectorOptions options)
    : options_(std::move(options)) {
  CHECK(!options_.shm_dir.empty())
      << "ShmCollector requires a non-empty shm_dir";
}

AggregatedMetrics ShmCollector::CollectMetrics() const {
  AggregatedMetrics metrics;

  DIR* const dir_stream = opendir(options_.shm_dir.c_str());
  if (dir_stream == nullptr) {
    PLOG_EVERY_N_SEC(WARNING, 10)
        << "Failed to open shared-memory directory " << options_.shm_dir;
    return metrics;
  }
  absl::Cleanup close_dir = [dir_stream] { closedir(dir_stream); };

  while (const dirent* dir_entry = readdir(dir_stream)) {
    absl::string_view filename(dir_entry->d_name);
    // Skip non-shm or .tmp shm files.
    if (!absl::StartsWith(filename, kShmFilePrefix) ||
        !absl::EndsWith(filename, kShmFileExtension)) {
      continue;
    }

    const int fd = openat(dirfd(dir_stream), dir_entry->d_name,
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
      if (errno != ENOENT && errno != ELOOP) {
        PLOG_EVERY_N_SEC(WARNING, 10)
            << "Failed to open shared-memory segment at " << options_.shm_dir
            << "/" << filename;
      }
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
    } else if (flock(fd, LOCK_SH | LOCK_NB) == 0) {
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
      AggregateSegment(mapped_address, metrics);
    }
  }

  return metrics;
}

}  // namespace tpu_raiden::telemetry

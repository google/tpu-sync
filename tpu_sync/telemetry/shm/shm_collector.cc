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

#include <algorithm>
#include <atomic>
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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

ShmCollector::ShmCollector(ShmCollectorOptions options)
    : options_(std::move(options)) {
  CHECK(!options_.shm_dir.empty())
      << "ShmCollector requires a non-empty shm_dir";
}

namespace internal {

void AggregateSegment(const void* raw_seg,
                      absl::flat_hash_map<std::string, double>& totals) {
  if (raw_seg == nullptr ||
      (reinterpret_cast<uintptr_t>(raw_seg) % alignof(ShmSegmentLayout) != 0)) {
    return;
  }
  const auto* seg = static_cast<const ShmSegmentLayout*>(raw_seg);

  if (seg->header.magic.load(std::memory_order_acquire) != kRaidenShmMagic) {
    return;
  }
  if (seg->header.max_toc_entries != kMaxTocEntries) {
    return;
  }

  const uint32_t data_pool_offset = seg->header.data_pool_offset;
  if (data_pool_offset < sizeof(ShmSegmentLayout) ||
      data_pool_offset >= kSegmentTotalFileSize) {
    return;
  }
  size_t count = seg->header.toc_entry_count.load(std::memory_order_acquire);
  size_t num_entries = std::min(count, kMaxTocEntries);

  for (size_t i = 0; i < num_entries; ++i) {
    const ShmTocEntry& toc_entry = seg->toc[i];
    if (toc_entry.entry_state.load(std::memory_order_acquire) !=
        TocEntryState::kCommitted) {
      continue;
    }
    if (toc_entry.type != MetricType::kCounter &&
        toc_entry.type != MetricType::kGauge &&
        toc_entry.type != MetricType::kHistogram) {
      continue;
    }
    if (toc_entry.offset < data_pool_offset ||
        static_cast<uint64_t>(toc_entry.offset) + toc_entry.size >
            kSegmentTotalFileSize) {
      continue;
    }

    absl::string_view metric_name(
        toc_entry.metric_name,
        strnlen(toc_entry.metric_name, sizeof(toc_entry.metric_name)));
    if (metric_name.empty()) {
      continue;
    }

    absl::string_view encoded_labels(
        toc_entry.encoded_labels,
        strnlen(toc_entry.encoded_labels, sizeof(toc_entry.encoded_labels)));

    const uint8_t* raw =
        reinterpret_cast<const uint8_t*>(seg) + toc_entry.offset;
    auto get_key = [&] {
      return encoded_labels.empty()
                 ? std::string(metric_name)
                 : absl::StrCat(metric_name, "/", encoded_labels);
    };

    switch (toc_entry.type) {
      case MetricType::kCounter: {
        if (toc_entry.size >= sizeof(std::atomic<uint64_t>) &&
            (reinterpret_cast<uintptr_t>(raw) %
                 alignof(std::atomic<uint64_t>) ==
             0)) {
          totals[get_key()] += static_cast<double>(
              reinterpret_cast<const std::atomic<uint64_t>*>(raw)->load(
                  std::memory_order_relaxed));
        }
        break;
      }
      case MetricType::kGauge: {
        if (toc_entry.size >= sizeof(std::atomic<double>) &&
            (reinterpret_cast<uintptr_t>(raw) % alignof(std::atomic<double>) ==
             0)) {
          double gauge_value =
              reinterpret_cast<const std::atomic<double>*>(raw)->load(
                  std::memory_order_relaxed);
          if (std::isfinite(gauge_value)) {
            totals[get_key()] += gauge_value;
          }
        }
        break;
      }
      case MetricType::kHistogram: {
        if (toc_entry.size >= sizeof(ShmHistogramSlot) &&
            (reinterpret_cast<uintptr_t>(raw) % alignof(ShmHistogramSlot) ==
             0)) {
          auto* histogram_slot = reinterpret_cast<const ShmHistogramSlot*>(raw);
          double sum =
              histogram_slot->sample_sum.load(std::memory_order_relaxed);
          if (!std::isfinite(sum)) {
            break;
          }
          std::string key = get_key();
          totals[absl::StrCat(key, "/count")] += static_cast<double>(
              histogram_slot->sample_count.load(std::memory_order_relaxed));
          totals[absl::StrCat(key, "/sum")] += sum;

          // TODO: Optimize dynamic string allocation churn in
          // hot scrape loop across histogram buckets.
          std::string bucket_prefix = absl::StrCat(key, "/bucket_");
          for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
            totals[absl::StrCat(bucket_prefix, b)] +=
                static_cast<double>(histogram_slot->bucket_counts[b].load(
                    std::memory_order_relaxed));
          }
        }
        break;
      }
    }
  }
}

}  // namespace internal

void ShmCollector::CollectMetrics(
    absl::flat_hash_map<std::string, double>& totals) const {
  totals.clear();

  DIR* dir = opendir(options_.shm_dir.c_str());
  if (dir == nullptr) {
    return;
  }
  absl::Cleanup close_dir = [dir] { closedir(dir); };

  struct dirent* entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    absl::string_view filename(entry->d_name);
    if (!absl::StartsWith(filename, kShmFilePrefix) ||
        !absl::EndsWith(filename, kShmFileExtension)) {
      continue;
    }

    const std::string path = absl::StrCat(options_.shm_dir, "/", filename);
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
      continue;
    }
    absl::Cleanup close_fd = [fd] { close(fd); };

    struct stat file_stat = {};
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
      // Verify inode before unlinking to prevent TOCTOU.
      struct stat current_stat = {};
      if (lstat(path.c_str(), &current_stat) == 0 &&
          current_stat.st_ino == file_stat.st_ino &&
          current_stat.st_dev == file_stat.st_dev) {
        if (unlink(path.c_str()) != 0) {
          PLOG(WARNING)
              << "Failed to unlink dead worker shared-memory segment at "
              << path;
        }
      }
      // Do NOT call flock(LOCK_UN); close_fd will release the lock.
    } else if (flock(fd, LOCK_SH | LOCK_NB) == 0) {
      // Process is live. Re-verify fstat after acquiring shared lock to prevent
      // SIGBUS from newly created, undersized files during worker startup.
      if (fstat(fd, &file_stat) != 0 ||
          file_stat.st_size < static_cast<off_t>(kSegmentTotalFileSize)) {
        continue;
      }
      void* addr =
          mmap(nullptr, kSegmentTotalFileSize, PROT_READ, MAP_SHARED, fd, 0);
      if (addr != MAP_FAILED) {
        absl::Cleanup unmap_addr = [addr] {
          munmap(addr, kSegmentTotalFileSize);
        };
        internal::AggregateSegment(addr, totals);
      }
    }
  }
}

}  // namespace tpu_raiden::telemetry

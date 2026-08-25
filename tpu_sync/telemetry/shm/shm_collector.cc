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
#include <filesystem>  // NOLINT(build/c++17)
#include <iterator>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
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

void AggregateSegment(const ShmSegmentLayout* seg,
                      absl::flat_hash_map<std::string, double>& totals) {
  if (seg == nullptr ||
      (reinterpret_cast<uintptr_t>(seg) % alignof(ShmSegmentLayout) != 0)) {
    return;
  }
  if (seg->header.data_pool_offset < sizeof(ShmSegmentLayout) ||
      seg->header.data_pool_offset >= kSegmentTotalFileSize) {
    return;
  }
  uint32_t count = seg->header.toc_entry_count.load(std::memory_order_acquire);
  uint32_t num_entries = std::min(count, static_cast<uint32_t>(kMaxTocEntries));

  for (uint32_t i = 0; i < num_entries; ++i) {
    const ShmTocEntry& e = seg->toc[i];
    if (e.entry_state.load(std::memory_order_acquire) !=
        TocEntryState::kCommitted) {
      continue;
    }
    if (e.offset < seg->header.data_pool_offset ||
        static_cast<uint64_t>(e.offset) + e.size > kSegmentTotalFileSize) {
      continue;
    }

    absl::string_view metric_name(
        e.metric_name, strnlen(e.metric_name, sizeof(e.metric_name)));
    if (metric_name.empty()) {
      continue;
    }

    absl::string_view encoded_labels(
        e.encoded_labels, strnlen(e.encoded_labels, sizeof(e.encoded_labels)));

    const uint8_t* raw = reinterpret_cast<const uint8_t*>(seg) + e.offset;
    std::string key(metric_name);
    if (!encoded_labels.empty()) {
      absl::StrAppend(&key, "/", encoded_labels);
    }

    switch (e.type) {
      case MetricType::kCounter: {
        if (e.size >= sizeof(std::atomic<uint64_t>) &&
            (reinterpret_cast<uintptr_t>(raw) %
                 alignof(std::atomic<uint64_t>) ==
             0)) {
          totals[key] += static_cast<double>(
              reinterpret_cast<const std::atomic<uint64_t>*>(raw)->load(
                  std::memory_order_relaxed));
        }
        break;
      }
      case MetricType::kGauge: {
        if (e.size >= sizeof(std::atomic<double>) &&
            (reinterpret_cast<uintptr_t>(raw) % alignof(std::atomic<double>) ==
             0)) {
          double val = reinterpret_cast<const std::atomic<double>*>(raw)->load(
              std::memory_order_relaxed);
          if (std::isfinite(val)) {
            totals[key] += val;
          }
        }
        break;
      }
      case MetricType::kHistogram: {
        if (e.size >= sizeof(ShmHistogramSlot) &&
            (reinterpret_cast<uintptr_t>(raw) % alignof(ShmHistogramSlot) ==
             0)) {
          auto* h = reinterpret_cast<const ShmHistogramSlot*>(raw);
          totals[absl::StrCat(key, "/count")] += static_cast<double>(
              h->sample_count.load(std::memory_order_relaxed));
          double sum = h->sample_sum.load(std::memory_order_relaxed);
          if (std::isfinite(sum)) {
            totals[absl::StrCat(key, "/sum")] += sum;
          }

          // TODO: Optimize dynamic string allocation churn in
          // hot scrape loop across histogram buckets.
          std::string bucket_prefix = absl::StrCat(key, "/bucket_");
          for (size_t b = 0; b <= kNumHistogramBuckets; ++b) {
            totals[absl::StrCat(bucket_prefix, b)] += static_cast<double>(
                h->bucket_counts[b].load(std::memory_order_relaxed));
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
  std::error_code ec;
  if (!std::filesystem::exists(options_.shm_dir, ec) || ec) {
    return;
  }

  auto it = std::filesystem::directory_iterator(options_.shm_dir, ec);
  if (ec) {
    return;
  }

  for (; it != std::filesystem::directory_iterator();) {
    absl::Cleanup advance = [&] {
      it.increment(ec);
      if (ec) {
        ec.clear();
      }
    };

    std::error_code entry_ec;
    if (!it->is_regular_file(entry_ec) || entry_ec) {
      continue;
    }
    std::string fn = it->path().filename().string();
    if (!absl::StartsWith(fn, kShmFilePrefix) ||
        !absl::EndsWith(fn, kShmFileExtension)) {
      continue;
    }

    std::string path = it->path().string();
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
      continue;
    }
    absl::Cleanup close_fd = [fd] { close(fd); };

    struct stat st = {};
    if (fstat(fd, &st) != 0) {
      continue;
    }

    if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
      // Process is dead. Dead worker metrics are intentionally discarded upon
      // reaping; only metrics from live processes are aggregated.
      // Verify inode before unlinking to prevent TOCTOU.
      struct stat st_current = {};
      if (st.st_nlink > 0 && lstat(path.c_str(), &st_current) == 0 &&
          st_current.st_ino == st.st_ino && st_current.st_dev == st.st_dev) {
        unlink(path.c_str());
      }
      // Do NOT call flock(LOCK_UN); close_fd will release the lock.
    } else if (flock(fd, LOCK_SH | LOCK_NB) == 0) {
      // Process is live. Prevent SIGBUS: must be at least
      // kSegmentTotalFileSize.
      if (st.st_size < static_cast<off_t>(kSegmentTotalFileSize)) {
        continue;
      }
      void* addr =
          mmap(nullptr, kSegmentTotalFileSize, PROT_READ, MAP_SHARED, fd, 0);
      if (addr != MAP_FAILED) {
        absl::Cleanup unmap_addr = [addr] {
          munmap(addr, kSegmentTotalFileSize);
        };
        auto* live = reinterpret_cast<const ShmSegmentLayout*>(addr);
        if (live->header.magic.load(std::memory_order_acquire) ==
                kRaidenShmMagic &&
            live->header.version == kSupportedVersion) {
          internal::AggregateSegment(live, totals);
        }
      }
    }
  }
}

}  // namespace tpu_raiden::telemetry

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
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/label_util.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

ShmWriter::ShmWriter(const ShmWriterOptions& options) : options_(options) {
  if (options_.shm_dir.empty() || options_.local_rank.empty()) {
    LOG(WARNING) << "ShmWriter disabled: shm_dir or local_rank is empty";
    return;
  }

  std::error_code ec;
  std::filesystem::create_directories(options_.shm_dir, ec);
  if (ec) {
    LOG(ERROR) << "ShmWriter failed to create directory " << options_.shm_dir
               << ": " << ec.message();
    return;
  }

  absl::BitGen bitgen;
  uuid_ = absl::StrFormat("%08x", absl::Uniform<uint32_t>(bitgen));

  absl::MutexLock lock(mutex_);
  chunks_.reserve(kMaxChunks);
  if (!AllocateNewChunk()) {
    LOG(ERROR) << "ShmWriter failed to allocate initial shared-memory chunk in "
               << options_.shm_dir;
  }
}

ShmWriter::~ShmWriter() {
  absl::MutexLock lock(mutex_);
  for (auto& chunk : chunks_) {
    if (chunk.segment) {
      munmap(chunk.segment, kSegmentTotalFileSize);
      chunk.segment = nullptr;
    }
    if (chunk.fd >= 0) {
      close(chunk.fd);
      chunk.fd = -1;
    }
    if (!chunk.file_path.empty()) {
      unlink(chunk.file_path.c_str());
    }
  }
  chunks_.clear();
  counter_cache_.clear();
  gauge_cache_.clear();
  histogram_cache_.clear();
}

bool ShmWriter::AllocateNewChunk() const {
  if (chunks_.size() >= kMaxChunks) {
    LOG_EVERY_N_SEC(ERROR, 10)
        << "ShmWriter reached maximum chunk limit of " << kMaxChunks
        << " chunks. Cannot allocate additional shared-memory chunks.";
    return false;
  }

  uint32_t chunk_idx = static_cast<uint32_t>(chunks_.size());
  std::string base_path =
      absl::StrCat(options_.shm_dir, "/", kShmFilePrefix, options_.local_rank,
                   "_", uuid_, "_chunk_", chunk_idx);
  std::string path = absl::StrCat(base_path, kShmFileExtension);
  std::string tmp_path = absl::StrCat(base_path, kShmTmpFileExtension);

  int fd = open(tmp_path.c_str(),
                O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC,
                options_.file_mode);
  if (fd < 0) {
    PLOG_EVERY_N_SEC(ERROR, 10)
        << "ShmWriter failed to open temporary file " << tmp_path;
    return false;
  }

  if (fchmod(fd, options_.file_mode) != 0) {
    PLOG_EVERY_N_SEC(WARNING, 10)
        << "ShmWriter failed to fchmod " << tmp_path << " to "
        << absl::StrFormat("0%o", options_.file_mode);
  }

  if (int err = posix_fallocate(fd, 0, kSegmentTotalFileSize); err != 0) {
    LOG_EVERY_N_SEC(ERROR, 10) << "ShmWriter failed to posix_fallocate file "
                               << tmp_path << ": " << std::strerror(err);
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  void* addr = mmap(nullptr, kSegmentTotalFileSize, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    PLOG_EVERY_N_SEC(ERROR, 10) << "ShmWriter failed to mmap file " << tmp_path;
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  auto* segment = reinterpret_cast<ShmSegmentLayout*>(addr);
  std::memset(segment, 0, kSegmentTotalFileSize);

  ShmTocHeader& header = segment->header;
  header.pid = static_cast<int64_t>(getpid());
  header.chunk_index = chunk_idx;
  header.toc_entry_count.store(0, std::memory_order_relaxed);
  header.max_toc_entries = static_cast<uint32_t>(kMaxTocEntries);
  header.data_pool_offset = static_cast<uint32_t>(sizeof(ShmSegmentLayout));
  header.data_pool_bytes.store(0, std::memory_order_relaxed);

  header.magic.store(kRaidenShmMagic, std::memory_order_release);

  if (flock(fd, LOCK_SH | LOCK_NB) != 0) {
    PLOG_EVERY_N_SEC(ERROR, 10)
        << "ShmWriter failed to flock file " << tmp_path;
    munmap(addr, kSegmentTotalFileSize);
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  if (rename(tmp_path.c_str(), path.c_str()) != 0) {
    PLOG_EVERY_N_SEC(ERROR, 10) << "ShmWriter failed to rename file from "
                                << tmp_path << " to " << path;
    munmap(addr, kSegmentTotalFileSize);
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  chunks_.push_back({fd, segment, path, chunk_idx});
  return true;
}

template <typename SlotT, MetricType Type>
SlotT* ShmWriter::AllocateSlotAndPublishEntry(absl::string_view name,
                                              absl::string_view encoded_labels,
                                              MetricKey key) const {
  constexpr uint32_t kAlign = alignof(SlotT) < kMetricSlotAlignment
                                  ? static_cast<uint32_t>(kMetricSlotAlignment)
                                  : static_cast<uint32_t>(alignof(SlotT));
  constexpr uint32_t kSlotSize =
      sizeof(SlotT) < kMetricSlotAlignment
          ? static_cast<uint32_t>(kMetricSlotAlignment)
          : static_cast<uint32_t>(sizeof(SlotT));
  Chunk* active_chunk = &chunks_.back();
  ShmTocHeader* header = &active_chunk->segment->header;
  uint32_t current_count =
      header->toc_entry_count.load(std::memory_order_relaxed);
  uint32_t current_bytes =
      header->data_pool_bytes.load(std::memory_order_relaxed);
  uint32_t aligned_bytes = (current_bytes + kAlign - 1) & ~(kAlign - 1);

  if (current_count >= header->max_toc_entries ||
      aligned_bytes + kSlotSize > kMaxDataPoolBytes) {
    if (!AllocateNewChunk()) return nullptr;
    active_chunk = &chunks_.back();
    header = &active_chunk->segment->header;
    current_count = header->toc_entry_count.load(std::memory_order_relaxed);
    current_bytes = header->data_pool_bytes.load(std::memory_order_relaxed);
    aligned_bytes = (current_bytes + kAlign - 1) & ~(kAlign - 1);
  }

  uint32_t offset = header->data_pool_offset + aligned_bytes;
  uint8_t* raw = reinterpret_cast<uint8_t*>(active_chunk->segment) + offset;
  auto* slot = new (raw) SlotT();

  ShmTocEntry& entry = active_chunk->segment->toc[current_count];
  entry.entry_state.store(TocEntryState::kWriting, std::memory_order_relaxed);
  std::memcpy(entry.metric_name, name.data(), name.size());
  entry.metric_name[name.size()] = '\0';
  std::memcpy(entry.encoded_labels, encoded_labels.data(),
              encoded_labels.size());
  entry.encoded_labels[encoded_labels.size()] = '\0';
  entry.type = Type;
  entry.offset = offset;
  entry.size = kSlotSize;

  header->data_pool_bytes.store(aligned_bytes + kSlotSize,
                                std::memory_order_relaxed);
  entry.entry_state.store(TocEntryState::kCommitted, std::memory_order_release);
  header->toc_entry_count.fetch_add(1, std::memory_order_release);

  auto& cache = GetMutableCache<SlotT>();
  cache.emplace(std::move(key), slot);
  return slot;
}

template <typename SlotT, MetricType Type>
SlotT* ShmWriter::GetOrCreateSlot(absl::string_view name,
                                  LabelSpan labels) const {
  if (name.empty() || name.size() >= sizeof(ShmTocEntry::metric_name)) {
    LOG_EVERY_N_SEC(ERROR, 10) << "Metric name rejected: '" << name
                               << "' is empty or exceeds buffer size "
                               << sizeof(ShmTocEntry::metric_name);
    return nullptr;
  }

  char label_buf[sizeof(ShmTocEntry::encoded_labels)];
  std::optional<absl::string_view> encoded_labels =
      FormatShmLabelsToBuffer(labels, absl::MakeSpan(label_buf));
  if (!encoded_labels.has_value() ||
      encoded_labels->size() >= sizeof(ShmTocEntry::encoded_labels)) {
    LOG_EVERY_N_SEC(ERROR, 10)
        << "Metric " << name
        << " labels rejected: encoded length exceeds buffer size "
        << sizeof(ShmTocEntry::encoded_labels);
    return nullptr;
  }

  MetricKeyView view{name, *encoded_labels};

  // 1. Fast-path: check cache under reader lock with zero-allocation view.
  {
    absl::ReaderMutexLock read_lock(mutex_);
    const auto& cache = GetCache<SlotT>();
    auto it = cache.find(view);
    if (it != cache.end()) return it->second;
  }

  // 2. Slow-path: acquire exclusive lock to allocate and publish.
  absl::MutexLock write_lock(mutex_);
  const auto& cache = GetCache<SlotT>();
  auto it = cache.find(view);
  if (it != cache.end()) return it->second;

  if (chunks_.empty()) return nullptr;

  MetricKey key{std::string(name), std::string(*encoded_labels)};
  return AllocateSlotAndPublishEntry<SlotT, Type>(name, *encoded_labels,
                                                  std::move(key));
}

void ShmWriter::IncrementCounter(absl::string_view name, LabelSpan labels,
                                 uint64_t val) const {
  std::atomic<uint64_t>* slot =
      GetOrCreateSlot<std::atomic<uint64_t>, MetricType::kCounter>(name,
                                                                   labels);
  if (slot) {
    slot->fetch_add(val, std::memory_order_relaxed);
  }
}

void ShmWriter::SetGauge(absl::string_view name, LabelSpan labels,
                         double val) const {
  if (!std::isfinite(val)) return;
  std::atomic<double>* slot =
      GetOrCreateSlot<std::atomic<double>, MetricType::kGauge>(name, labels);
  if (slot && std::isfinite(val)) {
    slot->store(val, std::memory_order_relaxed);
  }
}

void ShmWriter::ObserveHistogram(absl::string_view name, LabelSpan labels,
                                 double val) const {
  if (!std::isfinite(val)) return;
  ShmHistogramSlot* slot =
      GetOrCreateSlot<ShmHistogramSlot, MetricType::kHistogram>(name, labels);
  if (slot) {
    slot->Observe(val);
  }
}

}  // namespace tpu_raiden::telemetry

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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

namespace {

// Maximum number of label pairs sorted inline on the stack before falling back
// to dynamic heap allocation. 8 labels * 32 bytes = 256 bytes stack overhead.
constexpr size_t kInlineLabelCapacity = 8;

void AppendEscaped(absl::string_view s, std::string& out) {
  for (char c : s) {
    if (c == '\\' || c == '=' || c == ';') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
}

}  // namespace

std::string EncodeLabels(LabelSpan labels) {
  if (labels.empty()) return "";
  if (labels.size() == 1) {
    std::string encoded;
    AppendEscaped(labels[0].key, encoded);
    encoded.push_back('=');
    AppendEscaped(labels[0].value, encoded);
    return encoded;
  }

  absl::InlinedVector<std::pair<absl::string_view, absl::string_view>,
                      kInlineLabelCapacity>
      sorted;
  sorted.reserve(labels.size());
  for (const auto& label : labels) {
    sorted.emplace_back(label.key, label.value);
  }
  absl::c_sort(sorted);

  std::string encoded;
  bool first = true;
  for (const auto& [key, value] : sorted) {
    if (!first) {
      encoded.push_back(';');
    }
    first = false;
    AppendEscaped(key, encoded);
    encoded.push_back('=');
    AppendEscaped(value, encoded);
  }
  return encoded;
}

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
  }
  chunks_.clear();
  counter_cache_.clear();
  gauge_cache_.clear();
  histogram_cache_.clear();
}

bool ShmWriter::AllocateNewChunk() const {
  if (chunks_.size() >= kMaxChunks) {
    LOG(ERROR) << "ShmWriter reached maximum chunk limit of " << kMaxChunks
               << " chunks. Cannot allocate additional shared-memory chunks.";
    return false;
  }

  uint32_t chunk_idx = static_cast<uint32_t>(chunks_.size());
  std::string path =
      absl::StrCat(options_.shm_dir, "/", kShmFilePrefix, options_.local_rank,
                   "_", uuid_, "_chunk_", chunk_idx, kShmFileExtension);
  std::string tmp_path = absl::StrCat(path, ".tmp");

  int fd = open(tmp_path.c_str(),
                O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC,
                options_.file_mode);
  if (fd < 0) {
    PLOG(ERROR) << "ShmWriter failed to open temporary file " << tmp_path;
    return false;
  }

  if (ftruncate(fd, kSegmentTotalFileSize) != 0) {
    PLOG(ERROR) << "ShmWriter failed to ftruncate file " << tmp_path;
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  void* addr = mmap(nullptr, kSegmentTotalFileSize, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    PLOG(ERROR) << "ShmWriter failed to mmap file " << tmp_path;
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
    PLOG(ERROR) << "ShmWriter failed to flock file " << tmp_path;
    munmap(addr, kSegmentTotalFileSize);
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  if (rename(tmp_path.c_str(), path.c_str()) != 0) {
    PLOG(ERROR) << "ShmWriter failed to rename file from " << tmp_path << " to "
                << path;
    munmap(addr, kSegmentTotalFileSize);
    close(fd);
    unlink(tmp_path.c_str());
    return false;
  }

  chunks_.push_back({fd, segment, path, chunk_idx});
  return true;
}

bool ShmWriter::ValidateMetric(absl::string_view name,
                               absl::string_view encoded_labels) const {
  if (chunks_.empty()) return false;

  if (name.size() >= sizeof(ShmTocEntry::metric_name)) {
    LOG(ERROR) << "Metric name rejected: " << name << " exceeds buffer size "
               << sizeof(ShmTocEntry::metric_name);
    return false;
  }
  if (encoded_labels.size() >= sizeof(ShmTocEntry::encoded_labels)) {
    LOG(ERROR) << "Metric " << name << " labels rejected: encoded length "
               << encoded_labels.size() << " exceeds buffer size "
               << sizeof(ShmTocEntry::encoded_labels);
    return false;
  }
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
  snprintf(entry.metric_name, sizeof(entry.metric_name), "%.*s",
           static_cast<int>(name.size()), name.data());
  snprintf(entry.encoded_labels, sizeof(entry.encoded_labels), "%.*s",
           static_cast<int>(encoded_labels.size()), encoded_labels.data());
  entry.type = Type;
  entry.offset = offset;
  entry.size = kSlotSize;

  entry.entry_state.store(TocEntryState::kCommitted, std::memory_order_release);
  header->data_pool_bytes.store(aligned_bytes + kSlotSize,
                                std::memory_order_relaxed);
  header->toc_entry_count.fetch_add(1, std::memory_order_release);

  auto& cache = GetCache<SlotT>();
  cache.emplace(std::move(key), slot);
  return slot;
}

template <typename SlotT, MetricType Type>
SlotT* ShmWriter::GetOrCreateSlot(absl::string_view name,
                                  LabelSpan labels) const {
  // TODO: Performance: GetOrCreateSlot calls EncodeLabels(labels)
  // unconditionally on every invocation, allocating a dynamic std::string on
  // the heap even when the metric stream is already present in the cache.
  // Consider formatting the encoded labels into a stack-allocated scratch
  // buffer (e.g. char scratch[128] or absl::InlinedVector<char, 128>) so that
  // MetricKeyView can perform transparent hash lookups on cache hits with zero
  // heap allocations.
  std::string encoded_labels = EncodeLabels(labels);
  MetricKey key{std::string(name), encoded_labels};

  // 1. Fast-path: check cache under reader lock.
  {
    absl::ReaderMutexLock read_lock(mutex_);
    const auto& cache = GetCache<SlotT>();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }

  // 2. Slow-path: acquire exclusive lock to validate, allocate, and publish.
  absl::MutexLock write_lock(mutex_);
  auto& cache = GetCache<SlotT>();
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  if (!ValidateMetric(name, encoded_labels)) return nullptr;

  return AllocateSlotAndPublishEntry<SlotT, Type>(name, encoded_labels,
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
  std::atomic<double>* slot =
      GetOrCreateSlot<std::atomic<double>, MetricType::kGauge>(name, labels);
  if (slot && std::isfinite(val)) {
    slot->store(val, std::memory_order_relaxed);
  }
}

void ShmWriter::ObserveHistogram(absl::string_view name, LabelSpan labels,
                                 double val) const {
  ShmHistogramSlot* slot =
      GetOrCreateSlot<ShmHistogramSlot, MetricType::kHistogram>(name, labels);
  if (slot) {
    slot->Observe(val);
  }
}

}  // namespace tpu_raiden::telemetry

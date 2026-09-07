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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_BLOCK_TRACKER_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_BLOCK_TRACKER_H_

#include <cstddef>
#include <list>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace tpu_raiden::kv_cache {

class BlockTracker {
 public:
  struct StatusResult {
    std::vector<std::string> done;
    std::vector<std::string> failed;
    std::vector<std::string> pending;
    std::vector<std::string> existing;
    std::vector<std::string> unregistered;
  };

  BlockTracker() = default;
  ~BlockTracker() = default;

  BlockTracker(const BlockTracker&) = delete;
  BlockTracker& operator=(const BlockTracker&) = delete;
  BlockTracker(BlockTracker&&) = delete;
  BlockTracker& operator=(BlockTracker&&) = delete;

  void AddPending(absl::Span<const std::string> block_hashes);
  void AddPending(const std::string& block_hash);

  void RemovePending(absl::Span<const std::string> block_hashes);
  void RemovePending(const std::string& block_hash);

  bool IsPending(absl::string_view block_hash) const;
  bool Contains(absl::string_view block_hash) const;

  std::vector<std::string> GetPending() const;

  void MarkDone(absl::Span<const std::string> block_hashes);
  void MarkDone(const std::string& block_hash);

  void MarkFailed(absl::Span<const std::string> block_hashes);
  void MarkFailed(const std::string& block_hash);

  void MarkExisting(absl::Span<const std::string> block_hashes);
  void MarkExisting(const std::string& block_hash);

  void MarkUnregistered(absl::Span<const std::string> block_hashes);
  void MarkUnregistered(const std::string& block_hash);

  // Atomically records failed blocks alongside existing or unregistered blocks.
  void MarkFailedWithExisting(absl::Span<const std::string> failed,
                              absl::Span<const std::string> existing);
  void MarkFailedWithUnregistered(absl::Span<const std::string> failed,
                                  absl::Span<const std::string> unregistered);

  void Update(absl::Span<const std::string> done,
              absl::Span<const std::string> failed = {});

  // Atomically merges another tracker's state into this tracker, removing
  // completed/failed blocks from pending, accumulating done, failed, existing,
  // and unregistered records, and merging any pending blocks not already
  // present. Clears |other|.
  void Merge(BlockTracker&& other);

  StatusResult Poll(absl::Duration timeout = absl::ZeroDuration());

  void Clear();
  bool Empty() const;
  size_t PendingCount() const;
  size_t DoneCount() const;
  size_t FailedCount() const;

 private:
  void MarkDoneLocked(absl::Span<const std::string> block_hashes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void MarkFailedLocked(absl::Span<const std::string> block_hashes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void MarkExistingLocked(absl::Span<const std::string> block_hashes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void MarkUnregisteredLocked(absl::Span<const std::string> block_hashes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::list<std::string>::iterator>
      pending_map_ ABSL_GUARDED_BY(mutex_);
  std::list<std::string> pending_list_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> existing_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> unregistered_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tpu_raiden::kv_cache

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_BLOCK_TRACKER_H_

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

#include "tpu_sync/kv_cache/block_tracker.h"

#include <cstddef>
#include <iterator>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace tpu_raiden::kv_cache {

void BlockTracker::AddPending(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  for (const auto& hash : block_hashes) {
    if (!pending_map_.contains(hash)) {
      pending_list_.push_back(hash);
      auto it = std::prev(pending_list_.end());
      pending_map_.emplace(hash, it);
    }
  }
}

void BlockTracker::AddPending(const std::string& block_hash) {
  AddPending(absl::MakeConstSpan(&block_hash, 1));
}

void BlockTracker::RemovePending(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  for (const auto& hash : block_hashes) {
    auto it = pending_map_.find(hash);
    if (it != pending_map_.end()) {
      pending_list_.erase(it->second);
      pending_map_.erase(it);
    }
  }
}

void BlockTracker::RemovePending(const std::string& block_hash) {
  RemovePending(absl::MakeConstSpan(&block_hash, 1));
}

bool BlockTracker::IsPending(absl::string_view block_hash) const {
  absl::MutexLock lock(mutex_);
  return pending_map_.contains(block_hash);
}

bool BlockTracker::Contains(absl::string_view block_hash) const {
  return IsPending(block_hash);
}

std::vector<std::string> BlockTracker::GetPending() const {
  absl::MutexLock lock(mutex_);
  return std::vector<std::string>(pending_list_.begin(), pending_list_.end());
}

void BlockTracker::MarkDoneLocked(absl::Span<const std::string> block_hashes) {
  for (const auto& hash : block_hashes) {
    auto it = pending_map_.find(hash);
    if (it != pending_map_.end()) {
      pending_list_.erase(it->second);
      pending_map_.erase(it);
    }
    done_.push_back(hash);
  }
}

void BlockTracker::MarkFailedLocked(
    absl::Span<const std::string> block_hashes) {
  for (const auto& hash : block_hashes) {
    auto it = pending_map_.find(hash);
    if (it != pending_map_.end()) {
      pending_list_.erase(it->second);
      pending_map_.erase(it);
    }
    failed_.push_back(hash);
  }
}

void BlockTracker::MarkExistingLocked(
    absl::Span<const std::string> block_hashes) {
  for (const auto& hash : block_hashes) {
    existing_.push_back(hash);
  }
}

void BlockTracker::MarkUnregisteredLocked(
    absl::Span<const std::string> block_hashes) {
  for (const auto& hash : block_hashes) {
    unregistered_.push_back(hash);
  }
}

void BlockTracker::MarkDone(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  MarkDoneLocked(block_hashes);
}

void BlockTracker::MarkDone(const std::string& block_hash) {
  MarkDone(absl::MakeConstSpan(&block_hash, 1));
}

void BlockTracker::MarkFailed(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  MarkFailedLocked(block_hashes);
}

void BlockTracker::MarkFailed(const std::string& block_hash) {
  MarkFailed(absl::MakeConstSpan(&block_hash, 1));
}

void BlockTracker::MarkExisting(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  MarkExistingLocked(block_hashes);
}

void BlockTracker::MarkExisting(const std::string& block_hash) {
  MarkExisting(absl::MakeConstSpan(&block_hash, 1));
}

void BlockTracker::MarkUnregistered(
    absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  MarkUnregisteredLocked(block_hashes);
}

void BlockTracker::MarkUnregistered(const std::string& block_hash) {
  MarkUnregistered(absl::MakeConstSpan(&block_hash, 1));
}

void BlockTracker::Update(absl::Span<const std::string> done,
                          absl::Span<const std::string> failed) {
  absl::MutexLock lock(mutex_);
  if (!done.empty()) {
    MarkDoneLocked(done);
  }
  if (!failed.empty()) {
    MarkFailedLocked(failed);
  }
}

void BlockTracker::Merge(BlockTracker&& other) {
  if (this == &other) {
    return;
  }
  std::vector<std::string> other_done;
  std::vector<std::string> other_failed;
  std::vector<std::string> other_existing;
  std::vector<std::string> other_unregistered;
  std::vector<std::string> other_pending;
  {
    absl::MutexLock other_lock(other.mutex_);
    other_done = std::move(other.done_);
    other.done_.clear();
    other_failed = std::move(other.failed_);
    other.failed_.clear();
    other_existing = std::move(other.existing_);
    other.existing_.clear();
    other_unregistered = std::move(other.unregistered_);
    other.unregistered_.clear();
    other_pending.assign(other.pending_list_.begin(),
                         other.pending_list_.end());
    other.pending_list_.clear();
    other.pending_map_.clear();
  }

  absl::MutexLock this_lock(mutex_);
  MarkDoneLocked(other_done);
  MarkFailedLocked(other_failed);
  existing_.insert(existing_.end(), other_existing.begin(),
                   other_existing.end());
  unregistered_.insert(unregistered_.end(), other_unregistered.begin(),
                       other_unregistered.end());
  for (const auto& hash : other_pending) {
    if (!pending_map_.contains(hash)) {
      pending_list_.push_back(hash);
      auto it = std::prev(pending_list_.end());
      pending_map_.emplace(hash, it);
    }
  }
}

BlockTracker::StatusResult BlockTracker::Poll(absl::Duration timeout) {
  absl::MutexLock lock(mutex_);
  if (timeout > absl::ZeroDuration() && done_.empty() && failed_.empty() &&
      existing_.empty() && unregistered_.empty()) {
    mutex_.AwaitWithTimeout(absl::Condition(
                                +[](BlockTracker* tracker) {
                                  tracker->mutex_.AssertHeld();
                                  return !tracker->done_.empty() ||
                                         !tracker->failed_.empty() ||
                                         !tracker->existing_.empty() ||
                                         !tracker->unregistered_.empty();
                                },
                                this),
                            timeout);
  }
  StatusResult result;
  result.pending.assign(pending_list_.begin(), pending_list_.end());
  result.done = std::move(done_);
  done_.clear();
  result.failed = std::move(failed_);
  failed_.clear();
  result.existing = std::move(existing_);
  existing_.clear();
  result.unregistered = std::move(unregistered_);
  unregistered_.clear();
  return result;
}

void BlockTracker::Clear() {
  absl::MutexLock lock(mutex_);
  pending_map_.clear();
  pending_list_.clear();
  done_.clear();
  failed_.clear();
  existing_.clear();
  unregistered_.clear();
}

bool BlockTracker::Empty() const {
  absl::MutexLock lock(mutex_);
  return pending_map_.empty() && done_.empty() && failed_.empty();
}

size_t BlockTracker::PendingCount() const {
  absl::MutexLock lock(mutex_);
  return pending_map_.size();
}

size_t BlockTracker::DoneCount() const {
  absl::MutexLock lock(mutex_);
  return done_.size();
}

size_t BlockTracker::FailedCount() const {
  absl::MutexLock lock(mutex_);
  return failed_.size();
}

}  // namespace tpu_raiden::kv_cache

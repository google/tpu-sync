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

#include "tpu_sync/kv_cache/staging_manager.h"

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <deque>
#include <numeric>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {

namespace {

absl::Time ToAbslDeadline(std::chrono::steady_clock::time_point deadline) {
  const auto remaining = std::max(std::chrono::steady_clock::duration::zero(),
                                  deadline - std::chrono::steady_clock::now());
  return absl::Now() + absl::FromChrono(remaining);
}

}  // namespace

StagingLayerManager::StagingLayerManager(size_t num_slots) : free_(num_slots) {
  std::iota(free_.begin(), free_.end(), 0);
}

absl::StatusOr<size_t> StagingLayerManager::Acquire(
    size_t unit, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool>* cancelled) {
  absl::MutexLock lock(mu_);
  if (free_.empty() && leases_.empty()) {
    return absl::FailedPreconditionError("StagingLayerManager has no slots");
  }
  auto ready = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return shutdown_ ||
           (cancelled != nullptr &&
            cancelled->load(std::memory_order_relaxed)) ||
           leases_.contains(unit) || !free_.empty();
  };
  if (!mu_.AwaitWithDeadline(absl::Condition(&ready),
                             ToAbslDeadline(deadline))) {
    return absl::DeadlineExceededError(absl::StrCat(
        "Timed out waiting for a free staging slot for unit ", unit));
  }
  if (shutdown_ ||
      (cancelled != nullptr && cancelled->load(std::memory_order_relaxed))) {
    return absl::CancelledError("StagingLayerManager is shutting down");
  }
  if (auto it = leases_.find(unit); it != leases_.end()) {
    return it->second;
  }
  const size_t slot = free_.front();
  free_.pop_front();
  leases_.emplace(unit, slot);
  return slot;
}

std::optional<size_t> StagingLayerManager::Lookup(size_t unit) const {
  absl::MutexLock lock(mu_);
  auto it = leases_.find(unit);
  if (it == leases_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<size_t> StagingLayerManager::Release(size_t unit) {
  absl::MutexLock lock(mu_);
  auto it = leases_.find(unit);
  if (it == leases_.end()) {
    return std::nullopt;
  }
  const size_t slot = it->second;
  leases_.erase(it);
  free_.push_back(slot);
  return slot;
}

void StagingLayerManager::ReleaseAll() {
  absl::MutexLock lock(mu_);
  for (const auto& [unit, slot] : leases_) {
    (void)unit;
    free_.push_back(slot);
  }
  leases_.clear();
}

void StagingLayerManager::Shutdown() {
  absl::MutexLock lock(mu_);
  shutdown_ = true;
}

size_t StagingLayerManager::num_free() const {
  absl::MutexLock lock(mu_);
  return free_.size();
}

}  // namespace tpu_raiden

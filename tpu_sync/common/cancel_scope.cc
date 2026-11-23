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

#include "tpu_sync/common/cancel_scope.h"

#include <cstdint>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

// How often ~CancelScope() logs while it waits.
constexpr absl::Duration kDestructionWaitLogInterval = absl::Seconds(10);

}  // namespace

CancelScope::Registration::~Registration() {
  if (scope_ != nullptr) scope_->Deregister(id_);
}

CancelScope::~CancelScope() {
  // Interrupts the operations that still hold registrations, so that they
  // return and destroy them.
  Cancel();
  absl::MutexLock lock(mu_);
  CHECK(!cancel_running_ || running_thread_ != std::this_thread::get_id())
      << "CancelScope destroyed by one of its own callbacks";
  const auto done = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return num_live_registrations_ == 0 && !cancel_running_;
  };
  const absl::Time start = absl::Now();
  while (!mu_.AwaitWithTimeout(absl::Condition(&done),
                               kDestructionWaitLogInterval)) {
    LOG(WARNING) << "~CancelScope() (" << this << ") still waiting after "
                 << absl::FormatDuration(absl::Now() - start) << " for "
                 << num_live_registrations_
                 << " registration(s) to be destroyed"
                 << (cancel_running_ ? " and for Cancel() to return" : "");
  }
}

void CancelScope::Cancel() {
  {
    absl::MutexLock lock(mu_);
    if (cancelled_) return;
    cancelled_ = true;
    cancel_running_ = true;
    running_thread_ = std::this_thread::get_id();
  }
  // From here on OnCancel() refuses new callbacks, so `callbacks_` only
  // shrinks. Run the callbacks one at a time without holding `mu_`: a callback
  // may call into the scope (e.g. destroy its own Registration) or take a lock
  // whose holder is waiting for `mu_`. Taking one at a time also lets a
  // concurrent ~Registration remove a callback that has not started.
  while (true) {
    absl::AnyInvocable<void() &&> callback;
    {
      absl::MutexLock lock(mu_);
      running_id_ = 0;
      if (callbacks_.empty()) {
        cancel_running_ = false;
        return;
      }
      auto it = callbacks_.begin();
      running_id_ = it->first;
      callback = std::move(it->second);
      callbacks_.erase(it);
    }
    std::move(callback)();
  }
}

bool CancelScope::cancelled() const {
  absl::ReaderMutexLock lock(mu_);
  return cancelled_;
}

absl::StatusOr<CancelScope::Registration> CancelScope::OnCancel(
    absl_nonnull absl::AnyInvocable<void() &&> on_cancel) {
  if (on_cancel == nullptr) {
    return absl::InvalidArgumentError(
        "CancelScope::OnCancel() called with an empty callback");
  }
  {
    absl::MutexLock lock(mu_);
    if (!cancelled_) {
      const uint64_t id = ++last_id_;
      callbacks_.emplace(id, std::move(on_cancel));
      ++num_live_registrations_;
      return Registration(this, id);
    }
  }
  // `on_cancel` is destroyed, with its captures, after `mu_` is released.
  return absl::CancelledError("CancelScope is already cancelled");
}

void CancelScope::Deregister(uint64_t id) {
  absl::AnyInvocable<void() &&> never_ran_callback;
  {
    absl::MutexLock lock(mu_);
    auto it = callbacks_.find(id);
    if (it != callbacks_.end()) {
      // The callback never ran, and now never will.
      never_ran_callback = std::move(it->second);
      callbacks_.erase(it);
    } else if (!(running_id_ == id &&
                 running_thread_ == std::this_thread::get_id())) {
      // Unless the callback is destroying its own registration, wait for it to
      // return.
      const auto done = [this, id]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return running_id_ != id;
      };
      mu_.Await(absl::Condition(&done));
    }
  }
  // Destroys a callback that never ran, with its captures, without `mu_` (their
  // destructors may use the scope) but while this registration still counts as
  // live, so that the scope's destructor waits for them.
  never_ran_callback = nullptr;
  absl::MutexLock lock(mu_);
  --num_live_registrations_;
}

}  // namespace tpu_raiden

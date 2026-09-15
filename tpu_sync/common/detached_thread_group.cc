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

#include "tpu_sync/common/detached_thread_group.h"

#include <memory>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {

struct DetachedThreadGroup::State {
  bool NoneInFlight() const ABSL_SHARED_LOCKS_REQUIRED(mu) {
    return in_flight == 0;
  }

  mutable absl::Mutex mu;
  int in_flight ABSL_GUARDED_BY(mu) = 0;
};

DetachedThreadGroup::DetachedThreadGroup(absl::string_view name)
    : name_(name), state_(std::make_shared<State>()) {}

DetachedThreadGroup::~DetachedThreadGroup() { AwaitAllDone(); }

bool DetachedThreadGroup::Spawn(absl::AnyInvocable<void() &&> task) {
  // Incremented before the thread is created so AwaitAllDone() cannot observe
  // zero while a task is still starting.
  {
    absl::MutexLock lock(state_->mu);
    ++state_->in_flight;
  }
  try {
    std::thread([state = state_, task = std::move(task)]() mutable {
      absl::Cleanup task_done = [&state] {
        absl::MutexLock lock(state->mu);
        --state->in_flight;
      };
      std::move(task)();
    }).detach();
    return true;
  } catch (const std::system_error& e) {
    // std::thread's constructor throws when pthread_create fails. Letting that
    // escape would terminate the process, so report it and let the caller
    // decide how to shed the work.
    LOG_EVERY_N_SEC(ERROR, 1)
        << "Failed to create " << name_ << " thread: " << e.what();
    absl::MutexLock lock(state_->mu);
    --state_->in_flight;
    return false;
  }
}

void DetachedThreadGroup::AwaitAllDone() {
  absl::MutexLock lock(state_->mu);
  state_->mu.Await(absl::Condition(state_.get(), &State::NoneInFlight));
}

}  // namespace tpu_raiden

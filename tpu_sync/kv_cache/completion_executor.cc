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

#include "tpu_sync/kv_cache/completion_executor.h"

#include <cstdint>
#include <deque>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {
namespace kv_cache {

CompletionExecutor& CompletionExecutor::Instance() {
  static absl::NoDestructor<CompletionExecutor> executor(kCompletionThreads);
  return *executor;
}

CompletionExecutor::CompletionExecutor(int threads) {
  workers_.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    workers_.emplace_back(&CompletionExecutor::WorkerLoop, this);
  }
  // Detached: the singleton is never destroyed (NoDestructor), so the workers
  // run for the life of the process and there is nothing to join.
  for (std::thread& worker : workers_) worker.detach();
}

void CompletionExecutor::Execute(Task task) noexcept {
  absl::MutexLock lock(mu_);
  queue_.push_back(std::move(task));
}

void CompletionExecutor::WorkerLoop() {
  for (;;) {
    Task task;
    {
      absl::MutexLock lock(mu_);
      mu_.Await(absl::Condition(
          +[](std::deque<Task>* q) { return !q->empty(); }, &queue_));
      ++wakeups_;
      task = std::move(queue_.front());
      queue_.pop_front();
      ++running_;
    }
    std::move(task)();
    absl::MutexLock lock(mu_);
    --running_;
  }
}

void CompletionExecutor::DrainForTesting() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(this, &CompletionExecutor::IsIdle));
}

int64_t CompletionExecutor::WakeupsForTesting() const {
  absl::MutexLock lock(mu_);
  return wakeups_;
}

}  // namespace kv_cache
}  // namespace tpu_raiden

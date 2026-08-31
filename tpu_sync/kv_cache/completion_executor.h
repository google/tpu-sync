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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_COMPLETION_EXECUTOR_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_COMPLETION_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/concurrency/executor.h"

namespace tpu_raiden {
namespace kv_cache {

// Worker threads in the process-wide pool.
inline constexpr int kCompletionThreads = 4;

// Process-wide thread pool for completion callbacks; gRPC and transfer
// threads are shared, so store work must not run on them. Schedule always
// enqueues and never runs the task on the calling thread, so scheduling
// under a lock or from an already-resolved future is safe. (Exception: if
// enqueueing itself throws, Execute runs the task inline.) The pool outlives
// every store, so tasks must check a lifetime fence (KVCacheStore::Lifetime).
class CompletionExecutor final : public tsl::Executor {
 public:
  static CompletionExecutor& Instance();

  static void Schedule(absl::AnyInvocable<void() &&> task) {
    Instance().Execute(std::move(task));
  }

  static size_t DefaultPoolSize() { return kCompletionThreads; }

  void Execute(Task task) noexcept final;

  // Blocks until the queue is empty and no task is running. TESTS ONLY.
  void DrainForTesting();

  // Number of tasks workers have dequeued. TESTS ONLY.
  int64_t WakeupsForTesting() const;

 private:
  friend class absl::NoDestructor<CompletionExecutor>;

  explicit CompletionExecutor(int threads);
  ~CompletionExecutor() final = default;

  bool IsIdle() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return queue_.empty() && running_ == 0;
  }
  void WorkerLoop();

  mutable absl::Mutex mu_;
  // Tasks waiting to run.
  std::deque<Task> queue_ ABSL_GUARDED_BY(mu_);
  // Tasks currently running.
  int running_ ABSL_GUARDED_BY(mu_) = 0;
  // Total tasks dequeued; read by WakeupsForTesting.
  int64_t wakeups_ ABSL_GUARDED_BY(mu_) = 0;
  // Detached worker threads; see the constructor.
  std::vector<std::thread> workers_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_COMPLETION_EXECUTOR_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_DETACHED_THREAD_GROUP_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_DETACHED_THREAD_GROUP_H_

#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden {

// Runs one-shot tasks on detached threads and lets the owner wait for all of
// them to finish.
//
// A server that spawns a std::thread per accepted connection must not retain
// the thread objects: a joinable std::thread whose function has already
// returned still owns its stack, so glibc cannot unmap it and the process
// leaks two virtual memory areas (an 8 MB stack plus its guard page) per
// connection until vm.max_map_count is exhausted. Detaching fixes that but
// leaves no handles to join at teardown, so this class reproduces the join
// guarantee by counting the tasks that have not returned yet.
//
// Thread-safe.
class DetachedThreadGroup final {
 public:
  // |name| appears in the log message emitted when a thread cannot be created.
  explicit DetachedThreadGroup(absl::string_view name);

  // Waits for in-flight tasks, so destruction is safe even if the owner forgot
  // to call AwaitAllDone(). Owners must still call it explicitly before
  // tearing down state the tasks touch, because members are destroyed only
  // after the owner's destructor body has run.
  ~DetachedThreadGroup();

  DetachedThreadGroup(const DetachedThreadGroup&) = delete;
  DetachedThreadGroup& operator=(const DetachedThreadGroup&) = delete;

  // Runs |task| on a new detached thread. Returns false if the thread could
  // not be created, in which case |task| never runs and the caller still owns
  // whatever it captured.
  bool Spawn(absl::AnyInvocable<void() &&> task);

  // Blocks until every task spawned so far has returned. Tasks spawned while
  // this call is blocked are also waited for, so callers must stop accepting
  // new work first.
  void AwaitAllDone();

 private:
  // Held by the group and by every in-flight task. Sharing ownership keeps the
  // mutex alive until the last task has finished unlocking it, which a mutex
  // owned directly by the group would not: AwaitAllDone() returns as soon as
  // the final decrement is visible, while that task is still inside Unlock().
  struct State;

  const std::string name_;
  const std::shared_ptr<State> state_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_DETACHED_THREAD_GROUP_H_

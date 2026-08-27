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

#include <cstddef>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/functional/any_invocable.h"
#include "tpu_sync/core/numa_thread_pool.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {
constexpr size_t kDefaultCompletionThreadPoolSize = 8;

NumaThreadPool& GlobalThreadPool() {
  static absl::NoDestructor<NumaThreadPool> pool(
      kDefaultCompletionThreadPoolSize);
  return *pool;
}
}  // namespace

void CompletionExecutor::Schedule(absl::AnyInvocable<void() &&> task) {
  GlobalThreadPool().Schedule(
      [task = std::move(task)]() mutable { std::move(task)(); });
}

size_t CompletionExecutor::DefaultPoolSize() {
  return kDefaultCompletionThreadPoolSize;
}

}  // namespace kv_cache
}  // namespace tpu_raiden

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

#include "absl/functional/any_invocable.h"

namespace tpu_raiden {
namespace kv_cache {

// CompletionExecutor provides a process-wide thread pool for executing
// completion callbacks off gRPC completion queue threads, preventing
// handler threads from stalling on store mutexes.
class CompletionExecutor {
 public:
  // Schedules a task to run on the background completion executor.
  static void Schedule(absl::AnyInvocable<void() &&> task);

  // Returns default pool thread count.
  static size_t DefaultPoolSize();
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_COMPLETION_EXECUTOR_H_

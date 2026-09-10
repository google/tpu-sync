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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xla/tsl/platform/logging.h"
#include "tpu_sync/core/tpu_utils.h"

namespace tpu_raiden {

class NumaThreadPool {
 public:
  explicit NumaThreadPool(size_t threads);
  ~NumaThreadPool();

  // Schedule a task, optionally targeting a specific NUMA node.
  // The pool automatically handles pinning the worker thread to the target node
  // before executing the task.
  template <class F, class... Args>
  auto Schedule(std::optional<int> numa_node, F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    VLOG(1) << "NumaThreadPool::Schedule: scheduling task. Target NUMA: "
            << (numa_node.has_value() ? std::to_string(*numa_node) : "none")
            << ", from thread: " << std::this_thread::get_id();

    // Wrap the user task to inject the pinning logic before execution.
    auto task_wrapper =
        [numa_node, f = std::forward<F>(f),
         args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
          if (numa_node.has_value() && *numa_node >= 0) {
            PinCurrentThreadToNumaNode(*numa_node);
          }
          return std::apply(f, std::move(args_tuple));
        };

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::move(task_wrapper));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) {
        throw std::runtime_error("Schedule on stopped NumaThreadPool");
      }
      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  // Overload for scheduling without a specific NUMA node.
  template <class F, class... Args>
  auto Schedule(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    return Schedule(std::nullopt, std::forward<F>(f),
                    std::forward<Args>(args)...);
  }

  // Returns true if the current thread is a worker thread of ANY
  // NumaThreadPool.
  static bool IsCurrentThreadWorker();

  // Returns the number of worker threads in the pool.
  size_t num_threads() const { return workers.size(); }

  // Attempts to dequeue and execute one pending task.
  // Returns true if a task was executed, false if the queue was empty or pool
  // stopped.
  bool ExecuteOneTask();

 private:
  void WorkerLoop();

  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_

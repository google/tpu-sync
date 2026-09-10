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

#include "tpu_sync/core/numa_thread_pool.h"

#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "xla/tsl/platform/logging.h"

namespace tpu_raiden {

namespace {
// Thread-local flag to identify worker threads.
thread_local bool is_worker_thread = false;
}  // namespace

NumaThreadPool::NumaThreadPool(size_t threads) : stop(false) {
  VLOG(1) << "NumaThreadPool created with " << threads
          << " threads. Pool instance: " << this;
  for (size_t i = 0; i < threads; ++i) {
    workers.emplace_back([this, i] {
      is_worker_thread = true;  // Mark this thread as a worker
      VLOG(1) << "NumaThreadPool worker thread " << i
              << " started. Thread ID: " << std::this_thread::get_id()
              << ", Pool: " << this;
      this->WorkerLoop();
      VLOG(1) << "NumaThreadPool worker thread " << i
              << " exiting. Thread ID: " << std::this_thread::get_id();
    });
  }
}

void NumaThreadPool::WorkerLoop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(this->queue_mutex);
      this->condition.wait(
          lock, [this] { return this->stop || !this->tasks.empty(); });
      if (this->stop && this->tasks.empty()) {
        return;
      }
      task = std::move(this->tasks.front());
      this->tasks.pop();
    }
    VLOG(1) << "NumaThreadPool worker " << std::this_thread::get_id()
            << " executing task. Pool: " << this;
    task();
    VLOG(1) << "NumaThreadPool worker " << std::this_thread::get_id()
            << " finished task. Pool: " << this;
  }
}

NumaThreadPool::~NumaThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

bool NumaThreadPool::IsCurrentThreadWorker() { return is_worker_thread; }

bool NumaThreadPool::ExecuteOneTask() {
  std::function<void()> task;
  {
    std::unique_lock<std::mutex> lock(queue_mutex);  // NOLINT(build/c++11)
    if (stop || tasks.empty()) {
      return false;
    }
    task = std::move(tasks.front());
    tasks.pop();
  }
  task();
  return true;
}

}  // namespace tpu_raiden

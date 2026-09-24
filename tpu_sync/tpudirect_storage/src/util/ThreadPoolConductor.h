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

#ifndef TDSUL_SRC_UTIL_THREAD_POOL_CONDUCTOR_H_
#define TDSUL_SRC_UTIL_THREAD_POOL_CONDUCTOR_H_

#include <pthread.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "util/IoConductor.h"
#include "util/IoQueue.h"

namespace tdsul {

class WorkerThread {
 public:
  WorkerThread(int id);
  ~WorkerThread();

  // Disallow copy and move semantics
  WorkerThread(const WorkerThread&) = delete;
  WorkerThread& operator=(const WorkerThread&) = delete;

  void Start(int core_id = -1);
  void Stop();

  void EnqueueTask(WorkRequest* request);
  size_t GetPendingTaskCount() const;

 private:
  void Run();
  static void* ThreadEntry(void* arg);

  int id_;
  std::queue<WorkRequest*> task_queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  pthread_t thread_;
  bool thread_valid_;
  std::atomic<bool> stop_flag_;
};

class ThreadPoolConductor : public IoConductor {
 public:
  explicit ThreadPoolConductor(int num_threads, int core_start = -1,
                               int core_end = -1);
  ~ThreadPoolConductor();

  // Disallow copy and move semantics
  ThreadPoolConductor(const ThreadPoolConductor&) = delete;
  ThreadPoolConductor& operator=(const ThreadPoolConductor&) = delete;

  void Start() override;
  void Stop() override;

  void Dispatch(WorkRequest* request) override;
  void AssignQueue(IoQueue* queue);
  IoConductorType GetType() const override {
    return IoConductorType::THREAD_POOL;
  }

 private:
  std::vector<std::unique_ptr<WorkerThread>> workers_;
  int num_threads_;
  int core_start_;
  int core_end_;
};

// Dispatcher logic for routing WorkRequests
void DispatchWorkRequest(WorkRequest* req);

}  // namespace tdsul

#endif  // TDSUL_SRC_UTIL_THREAD_POOL_CONDUCTOR_H_

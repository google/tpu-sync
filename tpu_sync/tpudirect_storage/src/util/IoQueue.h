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

#ifndef TDSUL_SRC_UTIL_IO_QUEUE_H_
#define TDSUL_SRC_UTIL_IO_QUEUE_H_

#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>

#include "../def_internal.h"

namespace tdsul {

/**
 * @brief IoQueue manages pending and scheduled WorkRequests, ensuring FIFO
 * order is obeyed given a mixed workload of single and batch requests.
 *
 * Batch requests are required to obey strict FIFO ordering while still
 * utilizing the worker thread pool as much as possible. Therefore, a batch
 * request, as well as any single request that comes after a batch, must wait
 * for a clear signal that the previous work requests have finished. Without
 * batch requests, single requests could be scheduled to a worker thread's local
 * queue immediately. But batch requests make it complex.
 */
class IoQueue {
 public:
  IoQueue();
  ~IoQueue();

  // Disallow copy and move semantics for safety and simplicity
  IoQueue(const IoQueue&) = delete;
  IoQueue& operator=(const IoQueue&) = delete;
  IoQueue(IoQueue&&) = delete;
  IoQueue& operator=(IoQueue&&) = delete;

  // Returns true if the queue transitions from unassigned to assigned.
  bool TestAndSetAssigned();

  // Used by the worker thread to get the next request without holding a lock
  // long.
  WorkRequest* CheckAndGetNextSchedulableRequest();

  /**
   * @brief Enqueues a work request into the queue in a thread-safe manner.
   */
  void Push(WorkRequest* request);

  /**
   * @brief Dequeues a work request from the queue in a thread-safe manner.
   * @return true if a request was popped, false if queue was empty.
   */
  bool Pop(WorkRequest*& request);

  /**
   * @brief Retrieves the work request at the front of the queue without
   * removing it.
   * @return true if a request was retrieved, false if queue was empty.
   */
  bool Front(WorkRequest*& request) const;

  /**
   * @brief Checks if the queue is empty in a thread-safe manner.
   */
  bool Empty() const;

  /**
   * @brief Returns the number of requests in the queue in a thread-safe manner.
   */
  std::size_t Size() const;

  void SetWorker(class WorkerThread* worker);
  size_t GetTotalTransferSize() const;

 private:
  // Mutex to guard internal queue state for thread-safety.
  mutable std::mutex mutex_;
  std::deque<WorkRequest*> queue_;
  std::atomic<bool> is_assigned_{false};
  class WorkerThread* assigned_worker_ = nullptr;
  size_t total_transfer_size_ = 0;
};

}  // namespace tdsul

#endif  // TDSUL_SRC_UTIL_IO_QUEUE_H_

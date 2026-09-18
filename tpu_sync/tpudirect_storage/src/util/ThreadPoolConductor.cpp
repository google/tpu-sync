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

#include "util/ThreadPoolConductor.h"

#include <pthread.h>
#include <unistd.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include "absl/log/log.h"
#include "def_internal.h"
#include "tdsul/def.h"
#include "util/IoConductor.h"

namespace tdsul {

extern std::unique_ptr<tdsul::IoConductor> g_io_conductor;

WorkerThread::WorkerThread(int id)
    : id_(id), thread_valid_(false), stop_flag_(false) {}

WorkerThread::~WorkerThread() { Stop(); }

void* WorkerThread::ThreadEntry(void* arg) {
  static_cast<WorkerThread*>(arg)->Run();
  return nullptr;
}

void WorkerThread::Start(int core_id) {
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  if (core_id >= 0) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
  }

  int rc = pthread_create(&thread_, &attr, ThreadEntry, this);
  pthread_attr_destroy(&attr);

  if (rc != 0 && core_id >= 0) {
    // `pthread_create` fails with EINVAL when the requested CPU lies outside
    // the process's allowed cpuset. That happens routinely under restricted
    // cpusets (Forge test sandboxes, Borg jobs with a CPU mask), because
    // `sysconf(_SC_NPROCESSORS_ONLN)` reports the machine's cores rather than
    // the cores this process may actually run on.
    //
    // Falling back to an unpinned thread keeps the pool functional. Without
    // this, no worker starts at all and the pool silently accepts work it will
    // never execute.
    LOG(WARNING) << "pthread_create with affinity to core " << core_id
                 << " failed for worker " << id_ << " (" << rc
                 << "); retrying without CPU affinity";
    pthread_attr_t unpinned_attr;
    pthread_attr_init(&unpinned_attr);
    rc = pthread_create(&thread_, &unpinned_attr, ThreadEntry, this);
    pthread_attr_destroy(&unpinned_attr);
  }

  if (rc != 0) {
    LOG(ERROR) << "Error calling pthread_create for worker " << id_ << ": "
               << rc;
  } else {
    thread_valid_ = true;
    if (core_id >= 0) {
      LOG(INFO) << "Set worker " << id_ << " affinity to core " << core_id;
    }
  }
}

void WorkerThread::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_flag_ = true;
  }
  cv_.notify_one();
  if (thread_valid_) {
    pthread_join(thread_, nullptr);
    thread_valid_ = false;
  }

  // `Run()` drains the queue before exiting, so normally nothing is left here.
  // Requests do remain when the worker thread never started (see the affinity
  // fallback in `Start()`); without this the pool would accept work, never
  // execute it, and leak every queued `WorkRequest`.
  //
  // Ownership mirrors `Run()`: requests owned by an `IoQueue` are freed by that
  // queue, so only unowned requests are deleted here.
  std::queue<WorkRequest*> abandoned;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    abandoned.swap(task_queue_);
  }
  while (!abandoned.empty()) {
    WorkRequest* request = abandoned.front();
    abandoned.pop();
    if (request == nullptr) continue;
    if (request->future_) {
      request->future_->status.status = TDS_ERROR_IO;
    }
    if (!request->queue_) {
      delete request;
    }
  }
}

void WorkerThread::EnqueueTask(WorkRequest* request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    task_queue_.push(request);
  }
  cv_.notify_one();
}

size_t WorkerThread::GetPendingTaskCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return task_queue_.size();
}

void WorkerThread::Run() {
  while (true) {
    WorkRequest* request = nullptr;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_flag_ || !task_queue_.empty(); });

      if (stop_flag_ && task_queue_.empty()) {
        break;
      }

      request = task_queue_.front();
      task_queue_.pop();
    }

    if (request->type_ == RequestType::SINGLE) {
      // TODO: Process the IO request synchronously here
      VLOG(3) << "Worker " << std::to_string(id_) << " processing request.";
      if (request->future_) {
        request->future_->status.status = TDS_SUCCESS;
      }
    }
    if (!request->queue_) {
      // Free the request after processing only if it's not managed by a queue
      delete request;
    }
  }
}

ThreadPoolConductor::ThreadPoolConductor(int num_threads, int core_start,
                                         int core_end)
    : num_threads_(num_threads), core_start_(core_start), core_end_(core_end) {
  for (int i = 0; i < num_threads_; ++i) {
    workers_.push_back(std::make_unique<WorkerThread>(i));
  }
}

ThreadPoolConductor::~ThreadPoolConductor() { Stop(); }

void ThreadPoolConductor::Start() {
  int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (num_cores <= 0) {
    num_cores = 1;
  }

  int actual_core_start =
      (core_start_ >= 0 && core_start_ < num_cores) ? core_start_ : 0;
  int actual_core_end = (core_end_ >= 0 && core_end_ < num_cores &&
                         core_end_ >= actual_core_start)
                            ? core_end_
                            : num_cores - 1;
  int core_range = actual_core_end - actual_core_start + 1;

  if (core_start_ >= 0 && core_end_ >= 0 && core_range != num_threads_) {
    LOG(FATAL) << "ThreadPool size (" << num_threads_
               << ") does not match configured core range size (" << core_range
               << ").";
  }

  // Only pin when the caller explicitly configured a core range via
  // `tds_init`'s "worker_core_start"/"worker_core_end". Pinning by default is
  // actively harmful: the core indices are derived from
  // `sysconf(_SC_NPROCESSORS_ONLN)` (machine-wide), which says nothing about
  // which CPUs this process is permitted to run on under a restricted cpuset.
  const bool pin_to_cores = (core_start_ >= 0 && core_end_ >= 0);

  for (int i = 0; i < num_threads_; ++i) {
    // TODO: Set thread affinity according to the PCIe topology, which should be
    // attached to the closest core to NIC or SSD. For now, simple round-robin
    // assignment within the specified core range
    workers_[i]->Start(pin_to_cores ? actual_core_start + (i % core_range)
                                    : -1);
  }
}

void ThreadPoolConductor::Stop() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

void ThreadPoolConductor::Dispatch(WorkRequest* request) {
  if (workers_.empty()) {
    LOG(ERROR) << "No worker threads available in ThreadPool.";
    // Clean up request if we can't process it to avoid leaks
    delete request;
    return;
  }
  size_t min_pending = static_cast<size_t>(-1);
  int best_worker_idx = 0;

  for (int i = 0; i < num_threads_; ++i) {
    size_t pending = workers_[i]->GetPendingTaskCount();
    if (pending < min_pending) {
      min_pending = pending;
      best_worker_idx = i;
    }
  }

  workers_[best_worker_idx]->EnqueueTask(request);
}

}  // namespace tdsul

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

#include "IoQueue.h"

#include <cstddef>
#include <mutex>

#include "def_internal.h"

namespace tdsul {

IoQueue::IoQueue() = default;

IoQueue::~IoQueue() = default;

void IoQueue::Push(WorkRequest* request) {
  std::lock_guard<std::mutex> lock(mutex_);
  request->state_ = RequestState::PENDING;
  total_transfer_size_ += request->GetTransferSize();
  queue_.push_back(request);
}

bool IoQueue::Pop(WorkRequest*& request) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!queue_.empty() && queue_.front()->state_ == RequestState::FINISHED) {
    request = queue_.front();
    total_transfer_size_ -= request->GetTransferSize();
    queue_.pop_front();
    return true;
  }
  return false;
}

bool IoQueue::Front(WorkRequest*& request) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return false;
  }
  request = queue_.front();
  return true;
}

bool IoQueue::Empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

std::size_t IoQueue::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

bool IoQueue::TestAndSetAssigned() {
  bool expected = false;
  return is_assigned_.compare_exchange_strong(expected, true);
}

WorkRequest* IoQueue::CheckAndGetNextSchedulableRequest() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (queue_.empty()) {
    is_assigned_.store(false);
    return nullptr;
  }

  WorkRequest* req = queue_.front();
  if (req->state_ == RequestState::PENDING) {
    req->state_ = RequestState::SCHEDULED;
    return req;
  }

  is_assigned_.store(false);
  return nullptr;
}

void IoQueue::SetWorker(class WorkerThread* worker) {
  std::lock_guard<std::mutex> lock(mutex_);
  assigned_worker_ = worker;
}

size_t IoQueue::GetTotalTransferSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_transfer_size_;
}

}  // namespace tdsul

// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check_worker.h"

#include <cassert>
#include <chrono>
#include <utility>

namespace tpu_raiden::recv_nan_internal {
Worker::Worker() : thread_([this] { Run(); }) {}
Worker::~Worker() { Drain(); }

bool Worker::Schedule(Submit submit, std::function<void()> submitted) {
  {
    std::lock_guard lock(mu_);
    if (stopping_ || checks_ == kMaxChecks) return false;
    tasks_.push_back({std::move(submit), std::move(submitted)});
    ++checks_;
  }
  ready_.notify_one();
  return true;
}

void Worker::Drain() {
  {
    std::lock_guard lock(mu_);
    stopping_ = true;
  }
  ready_.notify_one();
  if (thread_.joinable()) thread_.join();
}

void Worker::Run() {
  std::deque<Poll> pending;
  for (;;) {
    Work work;
    {
      std::unique_lock lock(mu_);
      if (tasks_.empty()) {
        if (pending.empty()) {
          ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
          if (tasks_.empty()) return;
        } else {
          ready_.wait_for(lock, std::chrono::milliseconds(1),
                          [this] { return !tasks_.empty(); });
        }
      }
      if (!tasks_.empty()) {
        work = std::move(tasks_.front());
        tasks_.pop_front();
      }
    }
    if (work.submit) {
      auto poll = work.submit();
      assert(poll && "receive NaN submission must return a completion poll");
      pending.push_back(std::move(poll));
      work.submitted();
    }
    // FIFO consumption preserves each pool's receive-check window even if
    // different devices finish bool readback out of order.
    while (!pending.empty() && pending.front()()) {
      pending.pop_front();
      std::lock_guard lock(mu_);
      --checks_;
    }
  }
}
}  // namespace tpu_raiden::recv_nan_internal

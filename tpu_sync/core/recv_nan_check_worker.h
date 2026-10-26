// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#ifndef TPU_SYNC_CORE_RECV_NAN_CHECK_WORKER_H_
#define TPU_SYNC_CORE_RECV_NAN_CHECK_WORKER_H_

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace tpu_raiden::recv_nan_internal {
// Private to receive diagnostics. A poll must retain its device resources and
// return true only when they may be released. Callbacks must not throw.
class Worker {
 public:
  using Poll = std::function<bool()>;
  using Submit = std::function<Poll()>;
  Worker();
  ~Worker();
  // The callback publishes receive completion after actual device submission,
  // independently of bool readiness. False means no job was accepted.
  bool Schedule(Submit submit, std::function<void()> submitted);
  void Drain();

 private:
  struct Work {
    Submit submit;
    std::function<void()> submitted;
  };
  void Run();
  static constexpr size_t kMaxChecks = 128;
  std::mutex mu_;
  std::condition_variable ready_;
  std::deque<Work> tasks_;
  size_t checks_ = 0;
  bool stopping_ = false;
  std::thread thread_;
};
}  // namespace tpu_raiden::recv_nan_internal
#endif

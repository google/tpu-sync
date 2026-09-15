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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_

#include <atomic>
#include <string>
#include <thread>  // NOLINT

#include "tpu_sync/common/detached_thread_group.h"

namespace tpu_raiden {
namespace weight_sync {

class WeightSynchronizerBase;

// TCP Socket Server Daemon that runs natively in C++ to accept Control-Plane
// management RPC commands (like PushWeights and Shutdown) directly from the
// RL Coordinator or Controller task, bypassing Python servicer overhead.
//
// Connection threads are detached; the destructor blocks until every in-flight
// connection has returned instead of joining retained thread objects.
class WeightSynchronizerListener final {
 public:
  WeightSynchronizerListener(WeightSynchronizerBase* engine, int listener_port);
  ~WeightSynchronizerListener();

  WeightSynchronizerListener(const WeightSynchronizerListener&) = delete;
  WeightSynchronizerListener& operator=(const WeightSynchronizerListener&) =
      delete;

  int listener_port() const { return listener_port_; }
  bool is_active() const { return !stopping_; }

 private:
  void ListenerLoop();
  void ConnectionWorker(int client_fd);

  WeightSynchronizerBase* engine_;
  int listener_port_;
  std::atomic<int> server_fd_{-1};
  std::atomic<bool> stopping_{false};

  std::thread listener_thread_;

  // The destructor drains this so |engine_| and `this` outlive every in-flight
  // connection.
  DetachedThreadGroup connection_threads_{
      "WeightSynchronizerListener connection"};
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_

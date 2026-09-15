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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_LISTENER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_LISTENER_H_

#include <atomic>
#include <string>
#include <thread>  // NOLINT

#include "tpu_sync/common/detached_thread_group.h"

namespace tpu_raiden {
namespace kv_cache {

class KVCacheManagerBase;

// Connection threads are detached; the destructor blocks until every in-flight
// connection has returned instead of joining retained thread objects.
class KVCacheListener final {
 public:
  KVCacheListener(KVCacheManagerBase* engine, int listener_port);
  ~KVCacheListener();

  KVCacheListener(const KVCacheListener&) = delete;
  KVCacheListener& operator=(const KVCacheListener&) = delete;

  int listener_port() const { return listener_port_; }
  bool is_active() const { return !stopping_; }

 private:
  void ListenerLoop();
  void ConnectionWorker(int client_fd);

  KVCacheManagerBase* engine_;
  int listener_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  std::thread listener_thread_;

  // The destructor drains this so |engine_| and `this` outlive every in-flight
  // connection.
  DetachedThreadGroup connection_threads_{"KVCacheListener connection"};
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_LISTENER_H_

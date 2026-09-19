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
#include <cstdint>
#include <functional>
#include <string>
#include <thread>  // NOLINT

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tpu_sync/common/detached_thread_group.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
namespace kv_cache {

// Connection threads are detached; the destructor blocks until every in-flight
// connection has returned instead of joining retained thread objects.
class KVCacheListener final {
 public:
  template <typename Engine>
  KVCacheListener(Engine* engine, int listener_port)
      : KVCacheListener(
            EngineCallbacks{
                .pool_reshard_push =
                    [engine](const tpu_sync::rpc::StartTransferRequest& req,
                             absl::Span<const int64_t> src_block_ids,
                             int parallelism) {
                      return engine->PoolReshardPush(req, src_block_ids,
                                                     parallelism);
                    },
                .pool_reshard_register_recv =
                    [engine](const tpu_sync::rpc::StartTransferRequest& req,
                             absl::Span<const int64_t> chip_block_ids) {
                      return engine->PoolReshardRegisterRecv(req,
                                                             chip_block_ids);
                    },
                .push_kv_cache_resharded =
                    [engine](const tpu_sync::rpc::StartTransferRequest& req) {
                      if constexpr (requires {
                                      engine->PushKVCacheResharded(req);
                                    }) {
                        return engine->PushKVCacheResharded(req);
                      } else {
                        return engine->base()->PushKVCacheResharded(req);
                      }
                    },
                .register_active_plan =
                    [engine](uint64_t uuid,
                             const tpu_sync::rpc::StartTransferRequest& req,
                             bool is_sender) {
                      return engine->RegisterActivePlan(uuid, req, is_sender);
                    },
                .wait_for_pending_work =
                    [engine]() { return engine->WaitForPendingWork(); },
            },
            listener_port) {}
  ~KVCacheListener();

  KVCacheListener(const KVCacheListener&) = delete;
  KVCacheListener& operator=(const KVCacheListener&) = delete;

  int listener_port() const { return listener_port_; }
  bool is_active() const { return !stopping_; }

 private:
  struct EngineCallbacks {
    std::function<absl::Status(const tpu_sync::rpc::StartTransferRequest&,
                               absl::Span<const int64_t>, int)>
        pool_reshard_push;
    std::function<absl::Status(const tpu_sync::rpc::StartTransferRequest&,
                               absl::Span<const int64_t>)>
        pool_reshard_register_recv;
    std::function<absl::Status(const tpu_sync::rpc::StartTransferRequest&)>
        push_kv_cache_resharded;
    std::function<absl::Status(
        uint64_t, const tpu_sync::rpc::StartTransferRequest&, bool)>
        register_active_plan;
    std::function<absl::Status()> wait_for_pending_work;
  };

  KVCacheListener(EngineCallbacks callbacks, int listener_port);
  void ListenerLoop();
  void ConnectionWorker(int client_fd);

  EngineCallbacks callbacks_;
  int listener_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  std::thread listener_thread_;

  // The destructor drains this so |callbacks_| and `this` outlive every
  // in-flight connection.
  DetachedThreadGroup connection_threads_{"KVCacheListener connection"};
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_LISTENER_H_

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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
namespace kv_cache {

// Control-plane server daemon for KV cache transfers backed by
// ControlPipeServer (supporting TCP, gRPC, and ZMQ via
// `TPU_RAIDEN_CONTROL_PLANE_BACKEND`).
class KVCacheListener final {
 public:
  template <typename Engine>
  KVCacheListener(
      Engine* engine, int listener_port,
      std::optional<ControlPipeBackendType> backend_type = std::nullopt)
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
                .pool_host_addrs = [engine](uint64_t uuid, size_t pool_idx)
                    -> absl::StatusOr<tpu_sync::rpc::PoolHostAddrsProto> {
                  if constexpr (requires {
                                  engine->PoolHostBaseAddrs(uuid, pool_idx);
                                }) {
                    return engine->PoolHostBaseAddrs(uuid, pool_idx);
                  } else {
                    return engine->base()->PoolHostBaseAddrs(uuid, pool_idx);
                  }
                },
                .layer_host_addrs = [engine](uint64_t uuid)
                    -> std::vector<tpu_sync::rpc::PoolHostAddrsProto> {
                  if constexpr (requires { engine->LayerHostAddrs(uuid); }) {
                    return engine->LayerHostAddrs(uuid);
                  } else {
                    return engine->base()->LayerHostAddrs(uuid);
                  }
                },
            },
            listener_port, backend_type) {}
  ~KVCacheListener();

  KVCacheListener(const KVCacheListener&) = delete;
  KVCacheListener& operator=(const KVCacheListener&) = delete;

  int listener_port() const { return listener_port_; }
  bool is_active() const { return !stopping_.load(); }
  ControlPipeBackendType backend_type() const { return backend_type_; }

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
    // Pool base address per local shard (KVCacheManagerBase::
    // PoolHostBaseAddrs) and the pool layout, reported in the pool-reshard
    // receiver arm reply.
    std::function<absl::StatusOr<tpu_sync::rpc::PoolHostAddrsProto>(uint64_t,
                                                                    size_t)>
        pool_host_addrs;
    // Host addresses of every layer (KVCacheManagerBase::LayerHostAddrs),
    // reported in the planned-transfer receiver arm reply.
    std::function<std::vector<tpu_sync::rpc::PoolHostAddrsProto>(uint64_t)>
        layer_host_addrs;
  };

  KVCacheListener(EngineCallbacks callbacks, int listener_port,
                  std::optional<ControlPipeBackendType> backend_type);
  // Stops the server (waiting for in-flight handlers to return), then drains
  // pending engine work unless a COMMAND_SHUTDOWN already did.
  void Shutdown();
  void HandleControlRequest(const tpu_sync::rpc::ControlRequest& req,
                            tpu_sync::rpc::ControlResponse* resp);

  EngineCallbacks callbacks_;
  // Port actually bound by |pipe_server_| (0 until Start() returns), which
  // differs from the requested port when the caller asks for port 0.
  int listener_port_ = 0;
  ControlPipeBackendType backend_type_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> work_drained_{false};

  std::unique_ptr<ControlPipeServer> pipe_server_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_LISTENER_H_

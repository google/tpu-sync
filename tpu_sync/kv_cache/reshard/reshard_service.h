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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_SERVICE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/kv_cache/reshard/reshard_coordinator.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"
#include "tpu_sync/rpc/controller_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// The aggregate reshard control plane: owns the work-unit
// directory, request-block registry, planner (stateless), and coordinator,
// and serves RaidenControllerServer's ControllerRequest / ControlRequest
// surface over ControlPipeServer (supporting TCP, gRPC, and ZMQ via
// `TPU_RAIDEN_CONTROL_PLANE_BACKEND`).
class ReshardService {
 public:
  struct Options {
    int port = 0;
    double request_registry_ttl_s = 600.0;
    // Injectable for unit tests; defaults to a ControlPipeClient resolved from
    // `backend_type` / `TPU_RAIDEN_CONTROL_PLANE_BACKEND`.
    ControlPipeClient* client = nullptr;
    std::optional<ControlPipeBackendType> backend_type = std::nullopt;
    RequestBlockRegistry::Clock clock;  // defaults to steady_clock seconds
    WorkerDelivery delivery = WorkerDelivery{};
  };

  explicit ReshardService(const Options& options);
  ~ReshardService();

  ReshardService(const ReshardService&) = delete;
  ReshardService& operator=(const ReshardService&) = delete;

  // Strongly-typed control-plane handlers used by ControlPipeServer.
  tpu_sync::rpc::ControllerResponse HandleControllerCommand(
      const tpu_sync::rpc::ControllerRequest& req);
  tpu_sync::rpc::ControlResponse HandleRaidenCommand(
      const tpu_sync::rpc::ControlRequest& req);

  // Legacy raw frame fallback (raw request body in, raw response body out).
  std::string HandleFrame(const std::string& request_bytes);

  // ControlPipeServer hosting (sidecar mode).
  absl::Status StartServer();
  void StopServer();
  int port() const;
  // TTL (seconds) of unclaimed request-block registrations, as configured
  // through Options.
  double request_registry_ttl_s() const { return request_registry_ttl_s_; }

  // Set before StartServer(): invoked when a COMMAND_SHUTDOWN arrives (the
  // sidecar main uses it to exit its wait loop).
  void set_shutdown_callback(std::function<void()> callback) {
    shutdown_callback_ = std::move(callback);
  }

  WorkUnitDirectory* directory() { return directory_.get(); }
  RequestBlockRegistry* registry() { return registry_.get(); }
  ReshardCoordinator* coordinator() { return coordinator_.get(); }

 private:
  tpu_sync::rpc::ControlResponse HandleRaidenStartTransfer(
      const tpu_sync::rpc::ControlRequest& req);

  absl::Mutex state_mu_;  // the controller-wide lock (Python self._lock)
  std::unique_ptr<ControlPipeClient> default_client_;
  ControlPipeClient* client_ = nullptr;  // injected or default
  std::optional<ControlPipeBackendType> backend_type_;
  WorkerDelivery delivery_;
  std::unique_ptr<WorkUnitDirectory> directory_;
  std::unique_ptr<RequestBlockRegistry> registry_;
  std::unique_ptr<ReshardCoordinator> coordinator_;
  std::unique_ptr<ControlPipeServer> server_;
  std::function<void()> shutdown_callback_;
  int requested_port_ = 0;
  double request_registry_ttl_s_ = 600.0;
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_SERVICE_H_

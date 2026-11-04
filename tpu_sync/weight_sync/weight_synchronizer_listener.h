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
#include <functional>
#include <memory>

#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {

class WeightSynchronizerBase;

// Control-Plane Server Daemon that runs natively in C++ to accept management
// RPC commands (like PushWeights and Shutdown) directly via ControlPipeServer.
class WeightSynchronizerListener final {
 public:
  WeightSynchronizerListener(
      WeightSynchronizerBase* engine, int listener_port,
      ControlPipeBackendType backend_type = ControlPipeBackendType::kTcp);
  ~WeightSynchronizerListener();

  WeightSynchronizerListener(const WeightSynchronizerListener&) = delete;
  WeightSynchronizerListener& operator=(const WeightSynchronizerListener&) =
      delete;

  int listener_port() const { return listener_port_; }
  bool is_active() const { return !stopping_.load(); }

  void Shutdown();

  // Executes a single ControlRequest against |engine| and populates |resp|.
  // Invokes |shutdown_callback| if a COMMAND_SHUTDOWN request is processed.
  static void ExecuteControlRequest(
      WeightSynchronizerBase* engine,
      const ::tpu_sync::rpc::ControlRequest& req,
      ::tpu_sync::rpc::ControlResponse* resp,
      std::function<void()> shutdown_callback = nullptr);

 private:
  WeightSynchronizerBase* engine_;
  int listener_port_ = 0;
  std::atomic<bool> stopping_{false};

  std::unique_ptr<ControlPipeServer> pipe_server_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_

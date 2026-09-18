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

#include <memory>

#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/control_pipe/grpc_control_pipe.h"
#include "tpu_sync/common/control_pipe/tcp_control_pipe.h"
#include "tpu_sync/common/control_pipe/zmq_control_pipe.h"

namespace tpu_raiden {

std::unique_ptr<ControlPipeServer> CreateControlPipeServer(
    const ControlPipeConfig& config) {
  switch (config.backend_type) {
    case ControlPipeBackendType::kGrpc:
      return std::make_unique<GrpcControlPipeServer>(config);
    case ControlPipeBackendType::kZmq:
      return std::make_unique<ZmqControlPipeServer>(config);
    case ControlPipeBackendType::kTcp:
      return std::make_unique<TcpControlPipeServer>(config);
  }
  return std::make_unique<TcpControlPipeServer>(config);
}

std::unique_ptr<ControlPipeClient> CreateControlPipeClient(
    const ControlPipeConfig& config) {
  switch (config.backend_type) {
    case ControlPipeBackendType::kGrpc:
      return std::make_unique<GrpcControlPipeClient>(config);
    case ControlPipeBackendType::kZmq:
      return std::make_unique<ZmqControlPipeClient>(config);
    case ControlPipeBackendType::kTcp:
      return std::make_unique<TcpControlPipeClient>(config);
  }
  return std::make_unique<TcpControlPipeClient>(config);
}

}  // namespace tpu_raiden

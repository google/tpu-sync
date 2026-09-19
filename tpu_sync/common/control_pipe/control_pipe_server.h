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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_SERVER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"

namespace tpu_raiden {

class ControlPipeServer {
 public:
  virtual ~ControlPipeServer() = default;

  // Binds and starts listening on `requested_port` (0 for OS auto-allocation).
  // Returns the actual bound port.
  virtual absl::StatusOr<int> Start(int requested_port) = 0;

  // Gracefully stops the server and waits for all in-flight handler tasks
  // to complete. Safe to call multiple times.
  virtual void Stop() = 0;

  // Immediately stops accepting new connections (closing the listening socket)
  // while allowing currently executing handler tasks to finish sending their
  // responses without deadlocking if called from inside a handler callback.
  virtual void StopAccepting() {}

  virtual int bound_port() const = 0;
  virtual ControlDispatcher& dispatcher() = 0;
  virtual ControlPipeBackendType backend_type() const = 0;
};

std::unique_ptr<ControlPipeServer> CreateControlPipeServer(
    const ControlPipeConfig& config);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_SERVER_H_

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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/impl/service_type.h"
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
  // to complete. Safe to call multiple times. Must not be called from inside
  // a handler executing on this server (including registered gRPC services).
  virtual void Stop() = 0;

  // Registers |service| on the server so that it is served on the same port
  // as the control pipe, and one client channel reaches both. Must be called
  // before Start(). |service| is not owned and must outlive Stop(). Handlers
  // of |service| must not call Stop() on this server.
  //
  // Only the gRPC backend can serve an extra gRPC service. Every other
  // backend returns an Unimplemented error naming the backend; callers must
  // not ignore it.
  virtual absl::Status RegisterGrpcService(grpc::Service* service) {
    return absl::UnimplementedError(
        absl::StrCat("ControlPipeServer backend '",
                     ControlPipeBackendTypeName(backend_type()),
                     "' cannot serve a gRPC service"));
  }

  virtual int bound_port() const = 0;
  virtual ControlDispatcher& dispatcher() = 0;
  virtual ControlPipeBackendType backend_type() const = 0;
};

std::unique_ptr<ControlPipeServer> CreateControlPipeServer(
    const ControlPipeConfig& config);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_SERVER_H_

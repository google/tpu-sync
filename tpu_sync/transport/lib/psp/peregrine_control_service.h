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

#ifndef TPU_SYNC_TRANSPORT_LIB_PSP_PEREGRINE_CONTROL_SERVICE_H_
#define TPU_SYNC_TRANSPORT_LIB_PSP_PEREGRINE_CONTROL_SERVICE_H_

#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "peregrine/src/internal/control/message.pb.h"
#include "peregrine/src/internal/control/service.grpc.pb.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"

namespace tpu_raiden::transport::lib {

// Server-side gRPC implementation for PeregrineService.
// Handles incoming RPCs from connecting peers
class PeregrineControlServiceImpl final
    : public ::peregrine::internal::control::PeregrineService::Service {
 public:
  explicit PeregrineControlServiceImpl(RawBufferTransport* transport)
      : transport_(transport) {}

  grpc::Status ProcessUnary(
      grpc::ServerContext* context,
      const ::peregrine::internal::proto::ReqMsg* request,
      ::peregrine::internal::proto::RespMsg* response) override;

 private:
  RawBufferTransport* const transport_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_PSP_PEREGRINE_CONTROL_SERVICE_H_

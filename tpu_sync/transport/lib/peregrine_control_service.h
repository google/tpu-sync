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

#ifndef TPU_SYNC_TRANSPORT_LIB_PEREGRINE_CONTROL_SERVICE_H_
#define TPU_SYNC_TRANSPORT_LIB_PEREGRINE_CONTROL_SERVICE_H_

#include "absl/status/statusor.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "peregrine/src/internal/control/message.pb.h"
#include "peregrine/src/internal/control/message_internal.pb.h"
#include "peregrine/src/internal/control/service.grpc.pb.h"
#include "tpu_sync/common/grpc_util.h"
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
      ::peregrine::internal::proto::RespMsg* response) override {
    if (transport_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "RawBufferTransport is not initialized");
    }
    if (!request->has_psp_tcp_req()) {
      return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                          "Unsupported request type");
    }
    const auto& psp_req = request->psp_tcp_req();
    if (!psp_req.has_psp()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "psp token must be present");
    }
    const auto& psp = psp_req.psp();
    if (psp.spi() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "psp.spi must be non-zero");
    }
    if (!psp.has_gen()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "psp.gen must be present");
    }
    if (psp.key().size() != 16) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "psp.key must be exactly 16 bytes");
    }
    if (psp_req.peer_target().ip_port().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "peer_target.ip_port must be non-empty");
    }

    auto server_rx_key = transport_->RegisterPspPeer(
        psp.spi(), psp.key(), psp_req.peer_target().ip_port());
    if (!server_rx_key.ok()) {
      return ToGrpcStatus(server_rx_key.status());
    }

    auto* resp_psp = response->mutable_psp_tcp_resp()->mutable_psp();
    resp_psp->set_spi(server_rx_key->spi);
    resp_psp->set_gen(server_rx_key->gen);
    resp_psp->set_key(server_rx_key->key);
    return grpc::Status::OK;
  }

 private:
  RawBufferTransport* const transport_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_PEREGRINE_CONTROL_SERVICE_H_

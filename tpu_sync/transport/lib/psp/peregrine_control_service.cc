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

#include "tpu_sync/transport/lib/psp/peregrine_control_service.h"

#include <string_view>

#include "absl/status/statusor.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "peregrine/src/api/v1/psp/psp.h"
#include "peregrine/src/internal/control/message.pb.h"
#include "peregrine/src/internal/control/message_internal.pb.h"
#include "tpu_sync/common/grpc_util.h"

namespace tpu_raiden::transport::lib {

namespace {
using ::peregrine::internal::proto::ReqMsg;
using ::peregrine::internal::proto::RespMsg;
using ::peregrine::v1::PspToken;
}  // namespace

grpc::Status PeregrineControlServiceImpl::ProcessUnary(
    grpc::ServerContext* context, const ReqMsg* request, RespMsg* response) {
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

  const absl::StatusOr<PspToken> token = transport_->RegisterPspPeer(
      psp.spi(), psp.key(), psp_req.peer_target().ip_port());
  if (!token.ok()) {
    return ToGrpcStatus(token.status());
  }

  auto* resp_psp = response->mutable_psp_tcp_resp()->mutable_psp();
  resp_psp->set_spi(token->spi.value());
  resp_psp->set_gen(token->gen.value());
  const char* key = reinterpret_cast<const char*>(token->key.data());
  resp_psp->set_key(std::string_view(key, token->key.size()));
  return grpc::Status::OK;
}

}  // namespace tpu_raiden::transport::lib

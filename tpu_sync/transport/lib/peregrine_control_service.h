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
#include "absl/strings/str_cat.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/service.grpc.pb.h"
#include "tpu_sync/transport/lib/service.pb.h"

namespace tpu_raiden::transport::lib {

// Server-side gRPC implementation for PeregrineService.
// Handles incoming RPCs from connecting peers
class PeregrineControlServiceImpl final : public PeregrineService::Service {
 public:
  explicit PeregrineControlServiceImpl(RawBufferTransport* transport)
      : transport_(transport) {}

  grpc::Status ExchangePspKey(grpc::ServerContext* context,
                              const PspKeyExchangeRequest* request,
                              PspKeyExchangeResponse* response) override {
    if (transport_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "RawBufferTransport is not initialized");
    }
    if (request->client_spi() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "client_spi must be non-zero");
    }
    if (request->client_key().size() != 16) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "client_key must be exactly 16 bytes");
    }

    auto server_rx_key = transport_->RegisterPspPeer(request->client_spi(),
                                                     request->client_key());
    if (!server_rx_key.ok()) {
      return grpc::Status(
          grpc::StatusCode::INTERNAL,
          absl::StrCat("Failed to register PSP peer: ",
                       server_rx_key.status().message()));
    }

    response->set_server_spi(server_rx_key->spi);
    response->set_server_key(server_rx_key->key);
    return grpc::Status::OK;
  }

 private:
  RawBufferTransport* const transport_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_PEREGRINE_CONTROL_SERVICE_H_

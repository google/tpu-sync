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

#include "tpu_sync/core/transfer_control_server.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpc/impl/channel_arg_names.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"

namespace tpu_raiden {

TransferControlServiceImpl::TransferControlServiceImpl(
    TransferControlDelegate* delegate)
    : delegate_(delegate) {}

grpc::Status TransferControlServiceImpl::PullStream(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::PullStreamRequest* request,
    ::tpu_sync::proto::PullStreamResponse* response) {
  if (request == nullptr || response == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Null request or response");
  }
  if (delegate_ == nullptr) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        "No delegate configured");
  }
  absl::Status status =
      delegate_->HandleGrpcPullStream(context, *request, response);
  if (!status.ok()) {
    if (response->status() == ::tpu_sync::proto::STATUS_UNSPECIFIED ||
        response->status() == ::tpu_sync::proto::STATUS_OK) {
      response->set_status(::tpu_sync::proto::STATUS_INTERNAL_ERROR);
    }
    response->set_success(false);
    if (response->error_message().empty()) {
      response->set_error_message(std::string(status.message()));
    }
  } else {
    if (response->status() == ::tpu_sync::proto::STATUS_UNSPECIFIED) {
      response->set_status(::tpu_sync::proto::STATUS_OK);
    }
    if (response->status() == ::tpu_sync::proto::STATUS_OK) {
      response->set_success(true);
    }
  }
  return grpc::Status::OK;
}

grpc::Status TransferControlServiceImpl::Ack(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::TransferAckRequest* request,
    ::tpu_sync::proto::TransferAckResponse* response) {
  if (request == nullptr || response == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Null request or response");
  }
  if (delegate_ == nullptr) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        "No delegate configured");
  }
  absl::Status status = delegate_->HandleGrpcAck(context, *request, response);
  if (!status.ok()) {
    response->set_success(false);
    if (response->message().empty()) {
      response->set_message(std::string(status.message()));
    }
  } else {
    if (!response->success() && response->message().empty()) {
      response->set_success(true);
      response->set_message("SUCCESS");
    }
  }
  return grpc::Status::OK;
}

grpc::Status TransferControlServiceImpl::CheckLiveness(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::TransferLivenessRequest* request,
    ::tpu_sync::proto::TransferLivenessResponse* response) {
  if (request == nullptr || response == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Null request or response");
  }
  if (delegate_ == nullptr) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        "No delegate configured");
  }
  absl::Status status =
      delegate_->HandleGrpcCheckLiveness(context, *request, response);
  if (!status.ok()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string(status.message()));
  }
  return grpc::Status::OK;
}

TransferControlServer::TransferControlServer(TransferControlDelegate* delegate,
                                             int port)
    : delegate_(delegate), port_(port) {
  service_impl_ = std::make_unique<TransferControlServiceImpl>(delegate_);

  grpc::ServerBuilder builder;

  // Symmetric keepalive ping configuration: eliminates ping strike disconnects.
  builder.AddChannelArgument(
      GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 10000);
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  builder.RegisterService(service_impl_.get());

  if (port_ >= 0) {
    std::string server_address = "0.0.0.0:" + std::to_string(port_);
    int selected_port = 0;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                             &selected_port);
    grpc_server_ = builder.BuildAndStart();
    if (!grpc_server_ || selected_port == 0) {
      LOG(FATAL) << "Failed to start TransferControlServer on port " << port_;
    }
    port_ = selected_port;
    LOG(INFO) << "TransferControlServer actively listening on gRPC port: "
              << port_;
  } else {
    // In-Process Channel Mode (port == kInProcessPort == -1): No TCP sockets
    // bound.
    grpc_server_ = builder.BuildAndStart();
    if (!grpc_server_) {
      LOG(FATAL) << "Failed to start in-process TransferControlServer";
    }
    LOG(INFO) << "TransferControlServer running in IN-PROCESS mode (port="
              << port_ << ")";
  }
}

TransferControlServer::~TransferControlServer() { Shutdown(); }

std::shared_ptr<grpc::Channel> TransferControlServer::InProcessChannel(
    const grpc::ChannelArguments& args) {
  if (!grpc_server_) return nullptr;
  return grpc_server_->InProcessChannel(args);
}

void TransferControlServer::Shutdown(absl::Duration timeout) {
  if (stopping_.exchange(true)) return;
  if (grpc_server_) {
    grpc_server_->Shutdown(absl::ToChronoTime(absl::Now() + timeout));
  }
}

}  // namespace tpu_raiden

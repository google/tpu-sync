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

#include "tpu_sync/common/control_pipe/grpc_control_pipe.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpc/impl/channel_arg_names.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.grpc.pb.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"

namespace tpu_raiden {
namespace {

absl::Status GrpcStatusToAbsl(const grpc::Status& status) {
  if (status.ok()) return absl::OkStatus();
  switch (status.error_code()) {
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      return absl::DeadlineExceededError(status.error_message());
    case grpc::StatusCode::UNAVAILABLE:
      return absl::UnavailableError(status.error_message());
    case grpc::StatusCode::CANCELLED:
      return absl::CancelledError(status.error_message());
    case grpc::StatusCode::INVALID_ARGUMENT:
      return absl::InvalidArgumentError(status.error_message());
    case grpc::StatusCode::NOT_FOUND:
      return absl::NotFoundError(status.error_message());
    case grpc::StatusCode::ALREADY_EXISTS:
      return absl::AlreadyExistsError(status.error_message());
    case grpc::StatusCode::PERMISSION_DENIED:
      return absl::PermissionDeniedError(status.error_message());
    case grpc::StatusCode::UNAUTHENTICATED:
      return absl::UnauthenticatedError(status.error_message());
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return absl::ResourceExhaustedError(status.error_message());
    case grpc::StatusCode::FAILED_PRECONDITION:
      return absl::FailedPreconditionError(status.error_message());
    case grpc::StatusCode::ABORTED:
      return absl::AbortedError(status.error_message());
    case grpc::StatusCode::OUT_OF_RANGE:
      return absl::OutOfRangeError(status.error_message());
    case grpc::StatusCode::UNIMPLEMENTED:
      return absl::UnimplementedError(status.error_message());
    default:
      return absl::InternalError(status.error_message());
  }
}

std::string ExtractIpFromGrpcPeer(absl::string_view peer) {
  absl::string_view addr_port = peer;
  if (absl::StartsWithIgnoreCase(peer, "ipv4:") ||
      absl::StartsWithIgnoreCase(peer, "ipv6:")) {
    addr_port = peer.substr(5);
  }

  absl::string_view ip;
  if (absl::StartsWith(addr_port, "[")) {
    if (absl::StrContains(addr_port, ']')) {
      std::pair<absl::string_view, absl::string_view> split =
          absl::StrSplit(addr_port.substr(1), absl::MaxSplits(']', 1));
      ip = split.first;
    }
  } else if (absl::StartsWithIgnoreCase(addr_port, "%5B")) {
    std::string upper = absl::AsciiStrToUpper(addr_port);
    if (absl::StrContains(upper, "%5D")) {
      std::pair<absl::string_view, absl::string_view> split =
          absl::StrSplit(addr_port.substr(3), absl::MaxSplits("%5D", 1));
      ip = split.first;
    }
  } else if (absl::StrContains(addr_port, ':')) {
    size_t last_colon = addr_port.rfind(':');
    ip = addr_port.substr(0, last_colon);
  }

  if (ip.empty()) return "127.0.0.1";
  if (absl::StartsWithIgnoreCase(ip, "::ffff:")) {
    return std::string(ip.substr(7));
  }
  return std::string(ip);
}

}  // namespace

// =============================================================================
// Service Implementations
// =============================================================================

class GrpcControlPipeServer::ControlPipeServiceImpl final
    : public control_pipe::proto::ControlPipeService::Service {
 public:
  explicit ControlPipeServiceImpl(ControlDispatcher* dispatcher)
      : dispatcher_(dispatcher) {}

  grpc::Status SendControl(
      grpc::ServerContext* context,
      const control_pipe::proto::ControlEnvelope* request,
      control_pipe::proto::ControlResponseEnvelope* response) override {
    ControlContext ctx;
    ctx.peer_ip = ExtractIpFromGrpcPeer(context->peer());
    ctx.request_id = request->request_id();
    ctx.backend_type = ControlPipeBackendType::kGrpc;
    ctx.deadline = absl::FromChrono(context->deadline());
    for (const auto& [k, v] : request->metadata()) {
      ctx.metadata[k] = v;
    }

    *response = dispatcher_->Dispatch(ctx, *request);
    return grpc::Status::OK;
  }

  grpc::Status ControlStream(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<control_pipe::proto::ControlResponseEnvelope,
                               control_pipe::proto::ControlEnvelope>* stream)
      override {
    std::string peer_ip = ExtractIpFromGrpcPeer(context->peer());
    control_pipe::proto::ControlEnvelope req;
    while (stream->Read(&req)) {
      ControlContext ctx;
      ctx.peer_ip = peer_ip;
      ctx.request_id = req.request_id();
      ctx.backend_type = ControlPipeBackendType::kGrpc;
      ctx.deadline = absl::FromChrono(context->deadline());
      for (const auto& [k, v] : req.metadata()) {
        ctx.metadata[k] = v;
      }

      control_pipe::proto::ControlResponseEnvelope resp =
          dispatcher_->Dispatch(ctx, req);
      if (!stream->Write(resp)) {
        break;
      }
    }
    return grpc::Status::OK;
  }

 private:
  ControlDispatcher* dispatcher_;
};

class GrpcControlPipeServer::LegacyKVCacheServiceImpl final
    : public control_plane::proto::KVCacheControlPlaneService::Service {
 public:
  explicit LegacyKVCacheServiceImpl(ControlDispatcher* dispatcher)
      : dispatcher_(dispatcher) {}

  grpc::Status PullStream(
      grpc::ServerContext* context,
      const control_plane::proto::PullStreamRequest* request,
      control_plane::proto::PullStreamResponse* response) override {
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(
        control_plane::proto::PullStreamRequest::descriptor()->full_name());
    if (!request->SerializeToString(env.mutable_payload())) {
      response->set_status(-1);
      response->set_message("Failed to serialize PullStreamRequest");
      return grpc::Status::OK;
    }

    ControlContext ctx;
    ctx.peer_ip = ExtractIpFromGrpcPeer(context->peer());
    ctx.backend_type = ControlPipeBackendType::kGrpc;
    ctx.deadline = absl::FromChrono(context->deadline());

    control_pipe::proto::ControlResponseEnvelope resp_env =
        dispatcher_->Dispatch(ctx, env);
    if (resp_env.status_code() == 0) {
      if (!response->ParseFromString(resp_env.payload())) {
        response->set_status(-1);
        response->set_message("Failed to parse PullStreamResponse");
      }
    } else {
      response->set_status(-1);
      response->set_message(resp_env.error_message());
    }
    return grpc::Status::OK;
  }

  grpc::Status Ack(grpc::ServerContext* context,
                   const control_plane::proto::AckRequest* request,
                   control_plane::proto::AckResponse* response) override {
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(
        control_plane::proto::AckRequest::descriptor()->full_name());
    if (!request->SerializeToString(env.mutable_payload())) {
      response->set_status(-1);
      response->set_message("Failed to serialize AckRequest");
      return grpc::Status::OK;
    }

    ControlContext ctx;
    ctx.peer_ip = ExtractIpFromGrpcPeer(context->peer());
    ctx.backend_type = ControlPipeBackendType::kGrpc;
    ctx.deadline = absl::FromChrono(context->deadline());

    control_pipe::proto::ControlResponseEnvelope resp_env =
        dispatcher_->Dispatch(ctx, env);
    if (resp_env.status_code() == 0) {
      response->set_status(0);
    } else {
      response->set_status(-1);
      response->set_message(resp_env.error_message());
    }
    return grpc::Status::OK;
  }

 private:
  ControlDispatcher* dispatcher_;
};

// =============================================================================
// GrpcControlPipeServer
// =============================================================================

GrpcControlPipeServer::GrpcControlPipeServer(const ControlPipeConfig& config)
    : config_(config),
      pipe_service_(std::make_unique<ControlPipeServiceImpl>(&dispatcher_)),
      kv_cache_service_(
          std::make_unique<LegacyKVCacheServiceImpl>(&dispatcher_)) {}

GrpcControlPipeServer::~GrpcControlPipeServer() { Stop(); }

absl::StatusOr<int> GrpcControlPipeServer::Start(int requested_port) {
  absl::MutexLock lock(mu_);
  if (grpc_server_) {
    return absl::FailedPreconditionError(
        "GrpcControlPipeServer already started");
  }

  grpc::ServerBuilder builder;
  int max_msg_bytes = static_cast<int>(config_.max_frame_bytes);
  builder.SetMaxReceiveMessageSize(max_msg_bytes);
  builder.SetMaxSendMessageSize(max_msg_bytes);

  int selected_port = 0;
  std::string server_address = absl::StrCat("[::]:", requested_port);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(pipe_service_.get());
  builder.RegisterService(kv_cache_service_.get());

  grpc_server_ = builder.BuildAndStart();
  if (!grpc_server_ || selected_port <= 0) {
    grpc_server_.reset();
    return absl::InternalError(absl::StrCat(
        "Failed to start GrpcControlPipeServer on port ", requested_port));
  }
  bound_port_ = selected_port;
  return bound_port_;
}

void GrpcControlPipeServer::StopAccepting() {
  std::unique_ptr<grpc::Server> server_to_stop;
  {
    absl::MutexLock lock(mu_);
    server_to_stop = std::move(grpc_server_);
  }
  if (server_to_stop) {
    std::thread([server = std::move(server_to_stop)]() mutable {
      server->Shutdown(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(500));
      server->Wait();
    }).detach();
  }
}

void GrpcControlPipeServer::Stop() {
  std::unique_ptr<grpc::Server> server_to_stop;
  {
    absl::MutexLock lock(mu_);
    server_to_stop = std::move(grpc_server_);
  }
  if (server_to_stop) {
    server_to_stop->Shutdown(std::chrono::system_clock::now() +
                             std::chrono::seconds(2));
    server_to_stop->Wait();
  }
}

// =============================================================================
// GrpcControlPipeClient
// =============================================================================

GrpcControlPipeClient::GrpcControlPipeClient(const ControlPipeConfig& config)
    : config_(config) {}

std::shared_ptr<control_pipe::proto::ControlPipeService::Stub>
GrpcControlPipeClient::GetOrCreateStub(absl::string_view endpoint) {
  absl::MutexLock lock(stub_mu_);
  auto it = stubs_.find(endpoint);
  if (it != stubs_.end()) {
    return it->second;
  }

  grpc::ChannelArguments args;
  int max_msg_bytes = static_cast<int>(config_.max_frame_bytes);
  args.SetMaxReceiveMessageSize(max_msg_bytes);
  args.SetMaxSendMessageSize(max_msg_bytes);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 20000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
      std::string(endpoint), grpc::InsecureChannelCredentials(), args);
  std::shared_ptr<control_pipe::proto::ControlPipeService::Stub> stub =
      control_pipe::proto::ControlPipeService::NewStub(channel);
  stubs_[std::string(endpoint)] = stub;
  return stub;
}

absl::StatusOr<control_pipe::proto::ControlResponseEnvelope>
GrpcControlPipeClient::SendRaw(
    absl::string_view endpoint,
    const control_pipe::proto::ControlEnvelope& envelope,
    absl::Duration timeout) {
  absl::Duration effective_timeout =
      timeout > absl::ZeroDuration() ? timeout : config_.default_timeout;

  auto stub = GetOrCreateStub(endpoint);
  grpc::ClientContext ctx;
  if (effective_timeout > absl::ZeroDuration() &&
      effective_timeout < absl::InfiniteDuration()) {
    ctx.set_deadline(absl::ToChronoTime(absl::Now() + effective_timeout));
  }

  control_pipe::proto::ControlResponseEnvelope resp_env;
  grpc::Status status = stub->SendControl(&ctx, envelope, &resp_env);
  if (!status.ok()) {
    return GrpcStatusToAbsl(status);
  }
  return resp_env;
}

}  // namespace tpu_raiden

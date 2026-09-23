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

#include "tpu_sync/core/grpc_control_plane_backend.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"
#include "xla/tsl/concurrency/future.h"

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

control_plane::proto::PullStreamRequest ToProto(
    const PullStreamRequestSpec& req) {
  control_plane::proto::PullStreamRequest proto_req;
  proto_req.set_uuid(req.uuid);
  proto_req.set_ep_idx(req.ep_idx);
  proto_req.set_consumer_data_port(req.consumer_data_port);
  for (const auto& ip : req.consumer_ips) {
    proto_req.add_consumer_ips(ip);
  }
  for (int64_t id : req.src_block_ids) {
    proto_req.add_src_block_ids(id);
  }
  for (int64_t id : req.dst_block_ids) {
    proto_req.add_dst_block_ids(id);
  }
  return proto_req;
}

absl::StatusOr<PullStreamResponseSpec> FromProto(
    const grpc::Status& rpc_status,
    const control_plane::proto::PullStreamResponse& proto_resp) {
  if (!rpc_status.ok()) {
    return GrpcStatusToAbsl(rpc_status);
  }
  return PullStreamResponseSpec{
      .status = proto_resp.status(),
      .num_layers = proto_resp.num_layers(),
      .data_port = proto_resp.data_port(),
      .message = proto_resp.message(),
  };
}

}  // namespace

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
      size_t close_pos = upper.find("%5D");
      ip = addr_port.substr(3, close_pos - 3);
    }
  } else if (absl::StrContains(addr_port, ':')) {
    size_t last_colon = addr_port.rfind(':');
    ip = addr_port.substr(0, last_colon);
  }

  if (ip.empty()) return "";
  if (absl::StartsWithIgnoreCase(ip, "::ffff:")) {
    return std::string(ip.substr(7));
  }
  return std::string(ip);
}

grpc::Status KVCacheControlPlaneServiceImpl::PullStream(
    grpc::ServerContext* context,
    const control_plane::proto::PullStreamRequest* request,
    control_plane::proto::PullStreamResponse* response) {
  if (!handler_) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        "ControlPlaneHandler not initialized");
  }

  PullStreamRequestSpec spec;
  spec.uuid = request->uuid();
  spec.ep_idx = request->ep_idx();
  spec.consumer_data_port = request->consumer_data_port();
  spec.consumer_ips.assign(request->consumer_ips().begin(),
                           request->consumer_ips().end());
  spec.src_block_ids.assign(request->src_block_ids().begin(),
                            request->src_block_ids().end());
  spec.dst_block_ids.assign(request->dst_block_ids().begin(),
                            request->dst_block_ids().end());

  std::string fallback_peer_ip;
  if (spec.consumer_ips.empty()) {
    fallback_peer_ip = ExtractIpFromGrpcPeer(context->peer());
  }

  absl::StatusOr<PullStreamResponseSpec> result =
      handler_->OnPullStream(spec, fallback_peer_ip);
  absl::Status injected = FaultInjectStatus(hooks::kGrpcControlPlanePullReply);
  if (result.ok() && !injected.ok()) {
    result = injected;
  }
  if (!result.ok()) {
    response->set_status(-1);
    response->set_message(std::string(result.status().message()));
    return grpc::Status::OK;
  }

  response->set_status(result->status);
  response->set_num_layers(result->num_layers);
  response->set_data_port(result->data_port);
  response->set_message(result->message);
  return grpc::Status::OK;
}

grpc::Status KVCacheControlPlaneServiceImpl::Ack(
    grpc::ServerContext* context,
    const control_plane::proto::AckRequest* request,
    control_plane::proto::AckResponse* response) {
  if (!handler_) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        "ControlPlaneHandler not initialized");
  }
  absl::Status status = handler_->OnAck(request->uuid());
  if (!status.ok()) {
    response->set_status(-1);
    response->set_message(std::string(status.message()));
    return grpc::Status::OK;
  }
  response->set_status(0);
  return grpc::Status::OK;
}

GrpcControlPlaneBackend::GrpcControlPlaneBackend(size_t max_cached_stubs)
    : max_cached_stubs_(max_cached_stubs) {}

GrpcControlPlaneBackend::~GrpcControlPlaneBackend() {
  StopServer();
  CancelPendingPulls();
}

size_t GrpcControlPlaneBackend::TEST_CachedStubCount() const {
  absl::MutexLock lock(stub_mu_);
  return stubs_.size();
}

bool GrpcControlPlaneBackend::TEST_HasCachedStub(
    absl::string_view endpoint) const {
  absl::MutexLock lock(stub_mu_);
  return stubs_.contains(endpoint);
}

absl::StatusOr<int> GrpcControlPlaneBackend::StartServer(
    int requested_port, ControlPlaneHandler* handler) {
  if (server_) {
    StopServer();
  }
  service_impl_ = std::make_unique<KVCacheControlPlaneServiceImpl>(handler);

  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  builder.SetMaxSendMessageSize(64 * 1024 * 1024);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  builder.AddChannelArgument(
      GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 5000);

  int bound_port = 0;
  std::string listen_addr = absl::StrCat("[::]:", requested_port);
  builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials(),
                           &bound_port);
  builder.RegisterService(service_impl_.get());

  server_ = builder.BuildAndStart();
  if (!server_ || bound_port <= 0) {
    return absl::InternalError(
        absl::StrCat("Failed to bind gRPC control server on ", listen_addr));
  }
  LOG(INFO) << "gRPC Control Plane listening on [::]:" << bound_port;
  return bound_port;
}

void GrpcControlPlaneBackend::StopServer() {
  if (server_) {
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(5));
    server_->Wait();
    server_.reset();
  }
  service_impl_.reset();
}

std::shared_ptr<control_plane::proto::KVCacheControlPlaneService::Stub>
GrpcControlPlaneBackend::GetOrCreateStub(absl::string_view endpoint) {
  {
    absl::MutexLock lock(stub_mu_);
    if (auto it = stubs_.find(endpoint); it != stubs_.end()) {
      lru_order_.splice(lru_order_.begin(), lru_order_, it->second.lru_it);
      return it->second.stub;
    }
  }

  std::string ep_str(endpoint);
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  args.SetMaxSendMessageSize(64 * 1024 * 1024);

  auto channel = grpc::CreateCustomChannel(
      ep_str, grpc::InsecureChannelCredentials(), args);
  std::shared_ptr<control_plane::proto::KVCacheControlPlaneService::Stub> stub =
      control_plane::proto::KVCacheControlPlaneService::NewStub(channel);

  absl::MutexLock lock(stub_mu_);
  if (auto it = stubs_.find(ep_str); it != stubs_.end()) {
    lru_order_.splice(lru_order_.begin(), lru_order_, it->second.lru_it);
    return it->second.stub;
  }
  if (max_cached_stubs_ == 0) {
    return stub;
  }
  while (stubs_.size() >= max_cached_stubs_ && !lru_order_.empty()) {
    stubs_.erase(lru_order_.back());
    lru_order_.pop_back();
  }
  lru_order_.push_front(ep_str);
  stubs_.emplace(std::move(ep_str), StubCacheEntry{stub, lru_order_.begin()});
  return stub;
}

tsl::Future<PullStreamResponseSpec> GrpcControlPlaneBackend::SendPullRequest(
    absl::string_view remote_endpoint, const PullStreamRequestSpec& req,
    absl::Duration timeout) {
  auto [promise, future] = tsl::MakePromise<PullStreamResponseSpec>();
  auto owned = std::make_shared<PendingPull>();
  PendingPull* call = owned.get();
  call->stub = GetOrCreateStub(remote_endpoint);
  if (timeout > absl::ZeroDuration()) {
    call->context.set_deadline(std::chrono::system_clock::now() +
                               absl::ToChronoMilliseconds(timeout));
  }
  call->request = ToProto(req);
  call->promise = std::move(promise);
  {
    absl::MutexLock lock(pending_mu_);
    pending_pulls_.emplace(call, std::move(owned));
  }
  call->stub->async()->PullStream(
      &call->context, &call->request, &call->response,
      [this, call](grpc::Status status) { CompletePendingPull(call, status); });
  return future;
}

void GrpcControlPlaneBackend::CompletePendingPull(PendingPull* call,
                                                  const grpc::Status& status) {
  // Set runs the future's OnReady callbacks on this thread. The entry stays
  // registered until they return, so CancelPendingPulls cannot let the
  // destructor finish while one is still running.
  call->promise.Set(FromProto(status, call->response));
  std::shared_ptr<PendingPull> finished;
  {
    absl::MutexLock lock(pending_mu_);
    auto node = pending_pulls_.extract(call);
    finished = std::move(node.mapped());
  }
}

void GrpcControlPlaneBackend::CancelPendingPulls() {
  // TryCancel may run the completion callback on this thread, and that takes
  // pending_mu_, so cancel outside the lock; the copies keep each context
  // alive until its TryCancel returns.
  std::vector<std::shared_ptr<PendingPull>> to_cancel;
  {
    absl::MutexLock lock(pending_mu_);
    to_cancel.reserve(pending_pulls_.size());
    for (const auto& [call, owned] : pending_pulls_) {
      to_cancel.push_back(owned);
    }
  }
  for (const auto& call : to_cancel) {
    call->context.TryCancel();
  }
  to_cancel.clear();
  absl::MutexLock lock(pending_mu_);
  pending_mu_.Await(absl::Condition(
      +[](absl::flat_hash_map<PendingPull*, std::shared_ptr<PendingPull>>*
              pulls) { return pulls->empty(); },
      &pending_pulls_));
}

absl::Status GrpcControlPlaneBackend::SendAck(absl::string_view remote_endpoint,
                                              uint64_t uuid,
                                              absl::Duration timeout) {
  auto stub = GetOrCreateStub(remote_endpoint);
  grpc::ClientContext context;
  if (timeout > absl::ZeroDuration()) {
    context.set_deadline(std::chrono::system_clock::now() +
                         absl::ToChronoMilliseconds(timeout));
  }

  control_plane::proto::AckRequest proto_req;
  proto_req.set_uuid(uuid);

  control_plane::proto::AckResponse proto_resp;
  grpc::Status rpc_status = stub->Ack(&context, proto_req, &proto_resp);
  if (!rpc_status.ok()) {
    return GrpcStatusToAbsl(rpc_status);
  }
  if (proto_resp.status() != 0) {
    return absl::InternalError(
        absl::StrCat("Ack rejected by peer: ", proto_resp.message()));
  }
  return absl::OkStatus();
}

}  // namespace tpu_raiden

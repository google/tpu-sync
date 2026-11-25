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

#include "tpu_sync/weight_sync/lib/grpc_transport_service.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/platform/grpc_credential_factory.h"
#include "tpu_sync/proto/grpc_transport_service.grpc.pb.h"
#include "tpu_sync/proto/grpc_transport_service.pb.h"
#include "tpu_sync/rpc/weight_transfer_common.pb.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {
namespace {

constexpr char kGrpcArgKeepaliveTimeMs[] = "grpc.keepalive_time_ms";
constexpr char kGrpcArgKeepaliveTimeoutMs[] = "grpc.keepalive_timeout_ms";
constexpr char kGrpcArgKeepalivePermitWithoutCalls[] =
    "grpc.keepalive_permit_without_calls";
constexpr char kGrpcArgHttp2MinRecvPingIntervalWithoutDataMs[] =
    "grpc.http2.min_ping_interval_without_data_ms";
constexpr char kGrpcArgHttp2MaxPingStrikes[] = "grpc.http2.max_ping_strikes";
constexpr char kGrpcArgTcpTxZerocopyEnabled[] =
    "grpc.experimental.tcp_tx_zerocopy_enabled";

void SetStatusProto(const absl::Status& status,
                    ::tpu_sync::rpc::StatusProto* proto) {
  proto->set_code(static_cast<int32_t>(status.code()));
  if (!status.ok()) {
    proto->set_message(std::string(status.message()));
  }
}

absl::Cord MakeOwnedZeroCopyCord(absl::Span<const uint8_t> data,
                                 std::shared_ptr<const void> owner) {
  if (data.empty()) {
    return absl::Cord();
  }
  return absl::MakeCordFromExternal(
      absl::string_view(reinterpret_cast<const char*>(data.data()),
                        data.size()),
      [owner = std::move(owner)](absl::string_view) {});
}

// Self-deleting ServerWriteReactor that streams a single GrpcPullResponse
// message and keeps |owner| alive until OnDone().
class PullChunksReactor final
    : public grpc::ServerWriteReactor<::tpu_sync::proto::GrpcPullResponse> {
 public:
  explicit PullChunksReactor(::tpu_sync::proto::GrpcPullResponse response,
                             std::shared_ptr<const void> owner = nullptr)
      : owner_(std::move(owner)), response_(std::move(response)) {
    StartWriteAndFinish(&response_, grpc::WriteOptions(), grpc::Status::OK);
  }

  void OnDone() override { delete this; }

 private:
  std::shared_ptr<const void> owner_;
  ::tpu_sync::proto::GrpcPullResponse response_;
};

}  // namespace

class GrpcTransportServer::GrpcTransportServiceImpl final
    : public ::tpu_sync::proto::GrpcTransportService::CallbackService {
 public:
  explicit GrpcTransportServiceImpl(GrpcTransportDelegate* delegate)
      : delegate_(delegate) {}
  ~GrpcTransportServiceImpl() override = default;

  grpc::ServerUnaryReactor* PushChunks(
      grpc::CallbackServerContext* context,
      const ::tpu_sync::proto::GrpcPushRequest* request,
      ::tpu_sync::proto::GrpcPushResponse* response) override {
    SetStatusProto(HandlePushChunks(*request), response->mutable_status());
    grpc::ServerUnaryReactor* reactor = context->DefaultReactor();
    reactor->Finish(grpc::Status::OK);
    return reactor;
  }

  grpc::ServerWriteReactor<::tpu_sync::proto::GrpcPullResponse>* PullChunks(
      grpc::CallbackServerContext* /*context*/,
      const ::tpu_sync::proto::GrpcPullRequest* request) override {
    ::tpu_sync::proto::GrpcPullResponse response;
    if (delegate_ == nullptr) {
      SetStatusProto(
          absl::FailedPreconditionError("GrpcTransportDelegate is null"),
          response.mutable_status());
      return new PullChunksReactor(std::move(response));
    }

    PullChunksRequest pull_req{
        .uuid = request->uuid(),
        .buffer_id = request->buffer_id(),
        .shard_id = request->shard_id(),
        .chunk_batch_id = request->chunk_batch_id(),
    };
    absl::StatusOr<PullChunksResponse> pulled =
        delegate_->ResolvePullChunks(pull_req);
    if (!pulled.ok()) {
      SetStatusProto(pulled.status(), response.mutable_status());
      return new PullChunksReactor(std::move(response));
    }

    response.mutable_chunks()->Reserve(static_cast<int>(pulled->chunks.size()));
    for (const ChunkSpec& spec : pulled->chunks) {
      ::tpu_sync::proto::ChunkProto* chunk = response.add_chunks();
      if (const auto* scatter = std::get_if<ScatterChunkSpec>(&spec)) {
        ::tpu_sync::proto::ScatterChunkProto* dst =
            chunk->mutable_scatter_chunk();
        dst->set_byte_offset(scatter->byte_offset);
        dst->set_byte_size(scatter->byte_size);
        dst->set_byte_stride(scatter->byte_stride);
        dst->mutable_payload()->set_data(
            MakeOwnedZeroCopyCord(scatter->payload, pulled->owner));
      } else {
        const auto& gather = std::get<GatherChunkSpec>(spec);
        ::tpu_sync::proto::GatherChunkProto* dst =
            chunk->mutable_gather_chunk();
        dst->set_byte_offset(gather.byte_offset);
        dst->mutable_payloads()->Reserve(
            static_cast<int>(gather.payloads.size()));
        for (absl::Span<const uint8_t> span : gather.payloads) {
          dst->add_payloads()->set_data(
              MakeOwnedZeroCopyCord(span, pulled->owner));
        }
      }
    }
    SetStatusProto(absl::OkStatus(), response.mutable_status());
    return new PullChunksReactor(std::move(response), std::move(pulled->owner));
  }

 private:
  absl::Status HandlePushChunks(
      const ::tpu_sync::proto::GrpcPushRequest& request) {
    if (delegate_ == nullptr) {
      return absl::FailedPreconditionError("GrpcTransportDelegate is null");
    }

    ABSL_ASSIGN_OR_RETURN(
        absl::Span<uint8_t> host_buffer,
        delegate_->GetHostBuffer(request.buffer_id(), request.shard_id()));

    // Pass 1: validate every chunk upfront so rejected batches are atomic.
    for (int i = 0; i < request.chunks_size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          ValidateChunkProto(request.chunks(i), host_buffer.size(), i));
    }

    // Pass 2: apply all validated chunks into the host buffer.
    for (int i = 0; i < request.chunks_size(); ++i) {
      ApplyChunkProto(request.chunks(i), host_buffer);
    }

    return delegate_->OnPushChunksComplete(request.uuid(), request.buffer_id(),
                                           request.shard_id(),
                                           request.chunk_batch_id());
  }

  GrpcTransportDelegate* const delegate_;
};

GrpcTransportServer::GrpcTransportServer(
    GrpcTransportDelegate* delegate, GrpcTransportServerConfig config,
    std::shared_ptr<GrpcCredentialFactory> credential_factory)
    : config_(config),
      credential_factory_(credential_factory ? std::move(credential_factory)
                                             : CreateGrpcCredentialFactory()),
      service_(std::make_unique<GrpcTransportServiceImpl>(delegate)) {}

GrpcTransportServer::~GrpcTransportServer() { Stop(); }

absl::StatusOr<int> GrpcTransportServer::Start(int requested_port) {
  absl::MutexLock lock(mu_);
  if (server_ != nullptr) {
    return absl::FailedPreconditionError("GrpcTransportServer already started");
  }
  if (!credential_factory_) {
    return absl::FailedPreconditionError(
        "GrpcTransportServer credential factory is null");
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<grpc::ServerCredentials> server_creds,
                        credential_factory_->CreateServerCredentials());
  if (!server_creds) {
    return absl::FailedPreconditionError(
        "GrpcCredentialFactory::CreateServerCredentials returned null");
  }

  if (config_.max_message_size_bytes <= 0) {
    return absl::InvalidArgumentError(
        "GrpcTransportServerConfig::max_message_size_bytes must be positive");
  }

  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(config_.max_message_size_bytes);
  builder.SetMaxSendMessageSize(config_.max_message_size_bytes);
  builder.AddChannelArgument(kGrpcArgKeepaliveTimeMs,
                             config_.keepalive_time_ms);
  builder.AddChannelArgument(kGrpcArgKeepaliveTimeoutMs,
                             config_.keepalive_timeout_ms);
  builder.AddChannelArgument(kGrpcArgKeepalivePermitWithoutCalls, 1);
  builder.AddChannelArgument(kGrpcArgHttp2MinRecvPingIntervalWithoutDataMs,
                             config_.min_recv_ping_interval_without_data_ms);
  builder.AddChannelArgument(kGrpcArgHttp2MaxPingStrikes, 0);
  if (config_.enable_tcp_tx_zerocopy) {
    builder.AddChannelArgument(kGrpcArgTcpTxZerocopyEnabled, 1);
  }

  int selected_port = 0;
  builder.AddListeningPort(absl::StrCat("[::]:", requested_port),
                           std::move(server_creds), &selected_port);
  builder.RegisterService(service_.get());

  server_ = builder.BuildAndStart();
  if (!server_ || selected_port <= 0) {
    server_.reset();
    return absl::InternalError(absl::StrCat(
        "Failed to start GrpcTransportServer on port ", requested_port));
  }
  bound_port_ = selected_port;
  return bound_port_;
}

void GrpcTransportServer::Stop() {
  std::unique_ptr<grpc::Server> server_to_stop;
  {
    absl::MutexLock lock(mu_);
    server_to_stop = std::move(server_);
    bound_port_ = 0;
  }
  if (server_to_stop != nullptr) {
    server_to_stop->Shutdown(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(100));
    server_to_stop->Wait();
  }
}

int GrpcTransportServer::local_port() const {
  absl::MutexLock lock(mu_);
  return bound_port_;
}

std::string GrpcTransportServer::address() const {
  absl::MutexLock lock(mu_);
  return absl::StrCat("127.0.0.1:", bound_port_);
}

}  // namespace tpu_raiden::weight_sync

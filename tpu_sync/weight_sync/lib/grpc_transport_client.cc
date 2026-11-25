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

#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "tpu_sync/platform/grpc_credential_factory.h"
#include "tpu_sync/proto/grpc_transport_service.grpc.pb.h"
#include "tpu_sync/proto/grpc_transport_service.pb.h"
#include "tpu_sync/rpc/weight_transfer_common.pb.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {

struct GrpcTransportClient::PeerStubPool {
  std::vector<
      std::shared_ptr<::tpu_sync::proto::GrpcTransportService::StubInterface>>
      stubs;
  std::atomic<size_t> next_idx{0};
};

namespace {

constexpr char kGrpcArgEnableHttpProxy[] = "grpc.enable_http_proxy";
constexpr char kGrpcArgKeepaliveTimeMs[] = "grpc.keepalive_time_ms";
constexpr char kGrpcArgKeepaliveTimeoutMs[] = "grpc.keepalive_timeout_ms";
constexpr char kGrpcArgKeepalivePermitWithoutCalls[] =
    "grpc.keepalive_permit_without_calls";
constexpr char kGrpcArgHttp2MaxPingsWithoutData[] =
    "grpc.http2.max_pings_without_data";
constexpr char kGrpcArgUseLocalSubchannelPool[] =
    "grpc.use_local_subchannel_pool";
constexpr char kGrpcArgChannelId[] = "grpc.channel_id";
constexpr char kGrpcArgTcpTxZerocopyEnabled[] =
    "grpc.experimental.tcp_tx_zerocopy_enabled";
constexpr char kGrpcArgTcpTxZerocopySendBytesThreshold[] =
    "grpc.experimental.tcp_tx_zerocopy_send_bytes_threshold";

// Tracks outstanding external absl::Cord references created for a synchronous
// zero-copy PushChunks call so the call does not return until gRPC has
// released every slice referencing caller memory.
struct ExternalCordGuard {
  std::atomic<size_t> pending{1};
  absl::Notification all_released;

  void AddRef() { pending.fetch_add(1, std::memory_order_relaxed); }

  void Release() {
    if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      all_released.Notify();
    }
  }

  void Wait() { all_released.WaitForNotification(); }
};

absl::Cord MakeGuardedZeroCopyCord(
    absl::Span<const uint8_t> data,
    const std::shared_ptr<ExternalCordGuard>& guard) {
  if (data.empty()) {
    return absl::Cord();
  }
  guard->AddRef();
  return absl::MakeCordFromExternal(
      absl::string_view(reinterpret_cast<const char*>(data.data()),
                        data.size()),
      [guard](absl::string_view) { guard->Release(); });
}

absl::Status GrpcStatusToAbsl(const grpc::Status& status) {
  if (status.ok()) {
    return absl::OkStatus();
  }
  return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                      status.error_message());
}

absl::Status StatusProtoToAbsl(const ::tpu_sync::rpc::StatusProto& proto) {
  if (proto.code() == 0) {
    return absl::OkStatus();
  }
  return absl::Status(static_cast<absl::StatusCode>(proto.code()),
                      proto.message());
}

}  // namespace

GrpcTransportClient::GrpcTransportClient(
    GrpcTransportClientConfig config,
    std::shared_ptr<GrpcCredentialFactory> credential_factory)
    : config_(config),
      credential_factory_(credential_factory ? std::move(credential_factory)
                                             : CreateGrpcCredentialFactory()) {
  config_.num_channels_per_peer = std::max(1, config_.num_channels_per_peer);
}

GrpcTransportClient::~GrpcTransportClient() = default;

absl::StatusOr<std::shared_ptr<GrpcTransportClient::PeerStubPool>>
GrpcTransportClient::GetOrCreatePool(absl::string_view peer) {
  if (peer.empty()) {
    return absl::InvalidArgumentError("Peer address cannot be empty");
  }
  if (!credential_factory_) {
    return absl::FailedPreconditionError(
        "GrpcTransportClient credential factory is null");
  }

  {
    absl::MutexLock lock(mu_);
    auto it = stub_pools_.find(peer);
    if (it != stub_pools_.end()) {
      return it->second;
    }
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<grpc::ChannelCredentials> creds,
                        credential_factory_->CreateClientCredentials());
  if (!creds) {
    return absl::FailedPreconditionError(
        "GrpcCredentialFactory::CreateClientCredentials returned null");
  }

  if (config_.max_message_size_bytes <= 0) {
    return absl::InvalidArgumentError(
        "GrpcTransportClientConfig::max_message_size_bytes must be positive");
  }

  auto new_pool = std::make_shared<PeerStubPool>();
  new_pool->stubs.reserve(config_.num_channels_per_peer);
  const std::string peer_str(peer);
  for (int channel_idx = 0; channel_idx < config_.num_channels_per_peer;
       ++channel_idx) {
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(config_.max_message_size_bytes);
    args.SetMaxSendMessageSize(config_.max_message_size_bytes);
    args.SetInt(kGrpcArgEnableHttpProxy, 0);
    args.SetInt(kGrpcArgKeepaliveTimeMs, config_.keepalive_time_ms);
    args.SetInt(kGrpcArgKeepaliveTimeoutMs, config_.keepalive_timeout_ms);
    args.SetInt(kGrpcArgKeepalivePermitWithoutCalls, 1);
    args.SetInt(kGrpcArgHttp2MaxPingsWithoutData, 0);
    args.SetInt(kGrpcArgUseLocalSubchannelPool, 1);
    args.SetInt(kGrpcArgChannelId, channel_idx);
    if (config_.enable_tcp_tx_zerocopy) {
      args.SetInt(kGrpcArgTcpTxZerocopyEnabled, 1);
      if (config_.tcp_tx_zerocopy_send_bytes_threshold > 0) {
        args.SetInt(kGrpcArgTcpTxZerocopySendBytesThreshold,
                    config_.tcp_tx_zerocopy_send_bytes_threshold);
      }
    }
    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateCustomChannel(peer_str, creds, args);
    new_pool->stubs.push_back(
        ::tpu_sync::proto::GrpcTransportService::NewStub(channel));
  }

  absl::MutexLock lock(mu_);
  auto [it, inserted] = stub_pools_.try_emplace(peer_str, std::move(new_pool));
  return it->second;
}

absl::Status GrpcTransportClient::PushChunks(absl::string_view peer,
                                             const PushChunksRequest& request,
                                             absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<PeerStubPool> pool,
                        GetOrCreatePool(peer));
  const size_t idx = pool->next_idx.fetch_add(1, std::memory_order_relaxed) %
                     pool->stubs.size();
  const std::shared_ptr<::tpu_sync::proto::GrpcTransportService::StubInterface>&
      stub = pool->stubs[idx];

  auto guard = std::make_shared<ExternalCordGuard>();
  absl::Status status;
  {
    ::tpu_sync::proto::GrpcPushRequest proto_request;
    proto_request.set_uuid(request.uuid);
    proto_request.set_buffer_id(request.buffer_id);
    proto_request.set_shard_id(request.shard_id);
    proto_request.set_chunk_batch_id(request.chunk_batch_id);
    proto_request.mutable_chunks()->Reserve(
        static_cast<int>(request.chunks.size()));

    for (const ChunkSpec& spec : request.chunks) {
      ::tpu_sync::proto::ChunkProto* chunk = proto_request.add_chunks();
      if (const auto* scatter = std::get_if<ScatterChunkSpec>(&spec)) {
        ::tpu_sync::proto::ScatterChunkProto* dst =
            chunk->mutable_scatter_chunk();
        dst->set_byte_offset(scatter->byte_offset);
        dst->set_byte_size(scatter->byte_size);
        dst->set_byte_stride(scatter->byte_stride);
        dst->mutable_payload()->set_data(
            MakeGuardedZeroCopyCord(scatter->payload, guard));
      } else {
        const auto& gather = std::get<GatherChunkSpec>(spec);
        ::tpu_sync::proto::GatherChunkProto* dst =
            chunk->mutable_gather_chunk();
        dst->set_byte_offset(gather.byte_offset);
        dst->mutable_payloads()->Reserve(
            static_cast<int>(gather.payloads.size()));
        for (absl::Span<const uint8_t> span : gather.payloads) {
          dst->add_payloads()->set_data(MakeGuardedZeroCopyCord(span, guard));
        }
      }
    }

    const absl::Duration effective_timeout =
        timeout > absl::ZeroDuration() ? timeout : config_.default_timeout;
    grpc::ClientContext context;
    context.set_deadline(absl::ToChronoTime(absl::Now() + effective_timeout));

    ::tpu_sync::proto::GrpcPushResponse response;
    grpc::Status rpc_status =
        stub->PushChunks(&context, proto_request, &response);
    status = GrpcStatusToAbsl(rpc_status);
    if (status.ok()) {
      status = StatusProtoToAbsl(response.status());
    }
  }
  guard->Release();
  guard->Wait();
  return status;
}

absl::Status GrpcTransportClient::PullChunks(absl::string_view peer,
                                             const PullChunksRequest& request,
                                             absl::Span<uint8_t> dst_buffer,
                                             absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<PeerStubPool> pool,
                        GetOrCreatePool(peer));
  const size_t idx = pool->next_idx.fetch_add(1, std::memory_order_relaxed) %
                     pool->stubs.size();
  const std::shared_ptr<::tpu_sync::proto::GrpcTransportService::StubInterface>&
      stub = pool->stubs[idx];

  ::tpu_sync::proto::GrpcPullRequest proto_request;
  proto_request.set_uuid(request.uuid);
  proto_request.set_buffer_id(request.buffer_id);
  proto_request.set_shard_id(request.shard_id);
  proto_request.set_chunk_batch_id(request.chunk_batch_id);

  const absl::Duration effective_timeout =
      timeout > absl::ZeroDuration() ? timeout : config_.default_timeout;
  grpc::ClientContext context;
  context.set_deadline(absl::ToChronoTime(absl::Now() + effective_timeout));

  std::unique_ptr<
      grpc::ClientReaderInterface<::tpu_sync::proto::GrpcPullResponse>>
      reader = stub->PullChunks(&context, proto_request);

  ::tpu_sync::proto::GrpcPullResponse response;
  while (reader->Read(&response)) {
    ABSL_RETURN_IF_ERROR(StatusProtoToAbsl(response.status()));
    for (int i = 0; i < response.chunks_size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          ValidateChunkProto(response.chunks(i), dst_buffer.size(), i));
    }
    for (int i = 0; i < response.chunks_size(); ++i) {
      ApplyChunkProto(response.chunks(i), dst_buffer);
    }
    response.Clear();
  }

  return GrpcStatusToAbsl(reader->Finish());
}

size_t GrpcTransportClient::CachedChannelCount(absl::string_view peer) const {
  absl::MutexLock lock(mu_);
  auto it = stub_pools_.find(peer);
  return it == stub_pools_.end() ? 0 : it->second->stubs.size();
}

}  // namespace tpu_raiden::weight_sync

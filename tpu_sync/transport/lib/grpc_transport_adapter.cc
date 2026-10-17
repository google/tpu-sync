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

#include "tpu_sync/transport/lib/grpc_transport_adapter.h"

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "tpu_sync/telemetry/label_util.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/grpc_transport_service.grpc.pb.h"
#include "tpu_sync/transport/lib/grpc_transport_service.pb.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

namespace {

using ::tpu_raiden::telemetry::ExtractFirstEndpointIp;
using ::tpu_raiden::telemetry::MetricLabel;
using ::tpu_raiden::telemetry::RaidenMetricStore;
namespace metric_labels = ::tpu_raiden::telemetry::metric_labels;
namespace metric_names = ::tpu_raiden::telemetry::metric_names;

constexpr MetricLabel kPushLabels[] = {
    {.key = metric_labels::kDirection, .value = metric_labels::kDirectionPush},
};

constexpr absl::Duration kDefaultRpcDeadline = absl::Seconds(120);

constexpr char kGrpcArgEnableHttpProxy[] = "grpc.enable_http_proxy";
constexpr char kGrpcArgKeepaliveTimeMs[] = "grpc.keepalive_time_ms";
constexpr char kGrpcArgKeepaliveTimeoutMs[] = "grpc.keepalive_timeout_ms";
constexpr char kGrpcArgKeepalivePermitWithoutCalls[] =
    "grpc.keepalive_permit_without_calls";
constexpr char kGrpcArgHttp2MaxPingsWithoutData[] =
    "grpc.http2.max_pings_without_data";
constexpr char kGrpcArgHttp2MinRecvPingIntervalWithoutDataMs[] =
    "grpc.http2.min_ping_interval_without_data_ms";
constexpr char kGrpcArgUseLocalSubchannelPool[] =
    "grpc.use_local_subchannel_pool";
constexpr char kGrpcArgChannelId[] = "grpc.channel_id";

void RecordP2pTransferTime(std::chrono::steady_clock::time_point start_ts,
                           std::chrono::steady_clock::time_point end_ts,
                           absl::string_view src_ip, absl::string_view dst_ip) {
  const absl::Duration duration = absl::FromChrono(end_ts - start_ts);
  const double duration_ms =
      std::max(0.0, absl::ToDoubleMilliseconds(duration));
  const MetricLabel p2p_labels[] = {
      {.key = metric_labels::kSrcIp, .value = src_ip},
      {.key = metric_labels::kDstIp, .value = dst_ip},
  };
  RaidenMetricStore::GetGlobalMetricStore().ObserveHistogram(
      metric_names::kP2pTransferTimeMs, p2p_labels, duration_ms);
}

absl::Status ReportError(CompletionCallback& on_complete, absl::Status status) {
  if (on_complete) {
    on_complete(status);
  }
  return status;
}

}  // namespace

bool UseGrpcTransportAdapter() {
  const char* v = std::getenv("TPU_RAIDEN_DATA_TRANSPORT");
  if (v == nullptr) return false;
  absl::string_view sv(v);
  return sv == "1" || absl::EqualsIgnoreCase(sv, "grpc") ||
         absl::EqualsIgnoreCase(sv, "true");
}

proto::ChunkProto ChunkHeaderToProto(const ChunkHeader& header) {
  proto::ChunkProto proto;
  proto.set_version(header.version);
  proto.set_op(header.op);
  proto.set_flags(header.flags);
  proto.set_buffer_id(header.buffer_id);
  proto.set_reserved(header.reserved);
  proto.set_metadata_size(header.metadata_size);
  proto.set_remote_id(header.remote_id);
  proto.set_local_id(header.local_id);
  proto.set_count_or_size(header.count_or_size);
  proto.set_uuid(header.uuid);
  return proto;
}

ChunkHeader ProtoToChunkHeader(const proto::ChunkProto& proto) {
  ChunkHeader header = {};
  header.version = static_cast<uint16_t>(proto.version());
  header.op = static_cast<uint8_t>(proto.op());
  header.flags = static_cast<uint8_t>(proto.flags());
  header.buffer_id = static_cast<uint16_t>(proto.buffer_id());
  header.reserved = static_cast<uint16_t>(proto.reserved());
  header.metadata_size = static_cast<uint16_t>(proto.metadata_size());
  header.remote_id = proto.remote_id();
  header.local_id = proto.local_id();
  header.count_or_size = proto.count_or_size();
  header.uuid = proto.uuid();
  return header;
}

proto::ChunkProto RequestToChunkProto(const Request& req) {
  ChunkHeader header = {};
  header.version = 1;
  header.op = req.socket_opcode;
  header.flags = req.major_order;
  header.buffer_id = 0;
  header.reserved = static_cast<uint16_t>(req.parallelism);
  header.metadata_size = 0;
  header.remote_id = static_cast<uint32_t>(req.remote_id);
  header.local_id = req.local_id;
  header.count_or_size = req.count_or_size;
  header.uuid = req.uuid;
  return ChunkHeaderToProto(header);
}

absl::Status GrpcStatusToAbsl(const grpc::Status& status) {
  if (status.ok()) {
    return absl::OkStatus();
  }
  return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                      status.error_message());
}

grpc::Status AbslStatusToGrpc(const absl::Status& status) {
  if (status.ok()) {
    return grpc::Status::OK;
  }
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      std::string(status.message()));
}

GrpcTransportServiceImpl::GrpcTransportServiceImpl(
    GrpcCustomPushHandler push_handler)
    : push_handler_(std::move(push_handler)) {}

grpc::Status GrpcTransportServiceImpl::PushBlocks(
    grpc::ServerContext* context, const proto::GrpcPushRequest* request,
    proto::GrpcPushResponse* response) {
  if (!push_handler_) {
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                        "Push handler is not configured");
  }

  if (request->chunks().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Missing ChunkProto in GrpcPushRequest");
  }

  const ChunkHeader header = ProtoToChunkHeader(request->chunks(0));
  const std::vector<int> dst_block_ids(request->dst_block_ids().begin(),
                                       request->dst_block_ids().end());
  const std::vector<int> src_block_ids(request->src_block_ids().begin(),
                                       request->src_block_ids().end());

  auto send_handshake =
      [&](absl::Span<const int> allocated_ids) -> absl::Status {
    response->clear_allocated_block_ids();
    for (int id : allocated_ids) {
      response->add_allocated_block_ids(id);
    }
    return absl::OkStatus();
  };

  int payload_idx = 0;
  auto read_next_payload = [&]() -> absl::StatusOr<absl::Cord> {
    if (payload_idx >= request->chunks_size()) {
      return absl::UnavailableError("Unexpected EOF reading push payload");
    }
    return request->chunks(payload_idx++).payload();
  };

  absl::Status status =
      push_handler_(header, dst_block_ids, src_block_ids,
                    std::move(send_handshake), std::move(read_next_payload));
  if (!status.ok()) {
    response->set_ack(false);
    response->set_error_message(std::string(status.message()));
    return AbslStatusToGrpc(status);
  }

  response->set_ack(true);
  return grpc::Status::OK;
}

GrpcTransportServer::GrpcTransportServer(GrpcCustomPushHandler push_handler,
                                         int requested_port)
    : owned_service_(
          std::make_unique<GrpcTransportServiceImpl>(std::move(push_handler))) {
  StartServer(owned_service_.get(), requested_port);
}

GrpcTransportServer::GrpcTransportServer(
    proto::GrpcTransportService::Service* service, int requested_port) {
  StartServer(service, requested_port);
}

GrpcTransportServer::~GrpcTransportServer() { Shutdown(); }

void GrpcTransportServer::StartServer(
    proto::GrpcTransportService::Service* service, int requested_port) {
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(-1);
  builder.SetMaxSendMessageSize(-1);
  builder.AddChannelArgument(kGrpcArgKeepaliveTimeMs, 10000);
  builder.AddChannelArgument(kGrpcArgKeepaliveTimeoutMs, 5000);
  builder.AddChannelArgument(kGrpcArgKeepalivePermitWithoutCalls, 1);
  builder.AddChannelArgument(kGrpcArgHttp2MaxPingsWithoutData, 0);
  builder.AddChannelArgument(kGrpcArgHttp2MinRecvPingIntervalWithoutDataMs,
                             5000);
  builder.AddListeningPort(absl::StrCat("[::]:", requested_port),
                           grpc::InsecureServerCredentials(), &bound_port_);
  builder.RegisterService(service);
  server_ = builder.BuildAndStart();
}

void GrpcTransportServer::Shutdown() {
  if (server_) {
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::milliseconds(100));
    server_->Wait();
    server_.reset();
  }
}

std::string GrpcTransportServer::address() const {
  return absl::StrCat("127.0.0.1:", bound_port_);
}

GrpcTransportAdapter::GrpcTransportAdapter(RawBufferTransport* raw_transport,
                                           int parallelism)
    : raw_transport_(raw_transport), parallelism_(std::max(1, parallelism)) {}

GrpcTransportAdapter::~GrpcTransportAdapter() = default;

std::shared_ptr<proto::GrpcTransportService::StubInterface>
GrpcTransportAdapter::GetOrCreateStub(absl::string_view peer) {
  const std::string peer_str(peer);
  PeerStubPool* pool = nullptr;
  {
    absl::MutexLock lock(stub_mu_);
    auto it = stub_pools_.find(peer_str);
    if (it != stub_pools_.end()) {
      pool = it->second.get();
    }
  }

  if (pool == nullptr) {
    auto new_pool = std::make_unique<PeerStubPool>();
    new_pool->stubs.reserve(parallelism_);
    for (int channel_idx = 0; channel_idx < parallelism_; ++channel_idx) {
      grpc::ChannelArguments args;
      args.SetMaxReceiveMessageSize(-1);
      args.SetMaxSendMessageSize(-1);
      args.SetInt(kGrpcArgEnableHttpProxy, 0);
      args.SetInt(kGrpcArgKeepaliveTimeMs, 10000);
      args.SetInt(kGrpcArgKeepaliveTimeoutMs, 5000);
      args.SetInt(kGrpcArgKeepalivePermitWithoutCalls, 1);
      args.SetInt(kGrpcArgHttp2MaxPingsWithoutData, 0);
      // Force each channel in the pool to establish its own dedicated TCP/HTTP2
      // connection rather than sharing the process-global subchannel pool.
      args.SetInt(kGrpcArgUseLocalSubchannelPool, 1);
      args.SetInt(kGrpcArgChannelId, channel_idx);

      auto channel = grpc::CreateCustomChannel(
          peer_str, grpc::InsecureChannelCredentials(), args);
      new_pool->stubs.push_back(proto::GrpcTransportService::NewStub(channel));
    }

    absl::MutexLock lock(stub_mu_);
    auto [it, inserted] =
        stub_pools_.try_emplace(peer_str, std::move(new_pool));
    pool = it->second.get();
  }

  const size_t idx = pool->next_idx.fetch_add(1, std::memory_order_relaxed) %
                     pool->stubs.size();
  return pool->stubs[idx];
}

absl::StatusOr<Handle> GrpcTransportAdapter::Post(
    absl::Span<const std::string> peers, absl::Span<const Request> requests,
    absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
    CompletionCallback on_complete) {
  if (requests.empty()) {
    return ReportError(on_complete,
                       absl::InvalidArgumentError("Requests cannot be empty"));
  }

  const uint8_t opcode = requests.front().socket_opcode;
  switch (opcode) {
    case 1:
    case 6:
      return PostGrpcPush(peers, requests, src_block_ids, dst_block_ids,
                          std::move(on_complete));
    default:
      return ReportError(on_complete,
                         absl::UnimplementedError(absl::StrCat(
                             "Unsupported socket opcode: ", opcode)));
  }
}

absl::StatusOr<Status> GrpcTransportAdapter::Poll(Handle handle) {
  return absl::UnimplementedError(
      "GrpcTransportAdapter::Poll not implemented yet");
}

absl::StatusOr<Handle> GrpcTransportAdapter::PostGrpcPush(
    absl::Span<const std::string> peers, absl::Span<const Request> requests,
    absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
    CompletionCallback on_complete) {
  const size_t num_blocks = src_block_ids.size();
  const auto& req = requests.front();
  const int P = req.parallelism;
  if (P <= 0) {
    return ReportError(on_complete, absl::InvalidArgumentError(
                                        "parallelism must be positive"));
  }
  if (peers.empty()) {
    return ReportError(
        on_complete, absl::InvalidArgumentError("peers list cannot be empty"));
  }

  const absl::Span<const std::string> local_ips =
      raw_transport_ != nullptr ? raw_transport_->local_ips()
                                : absl::Span<const std::string>();
  const std::string src_ip(ExtractFirstEndpointIp(local_ips));
  const std::string dst_ip(ExtractFirstEndpointIp(peers));
  const std::chrono::steady_clock::time_point push_start_ts =
      std::chrono::steady_clock::now();

  std::vector<int> allocated_ids(num_blocks, 0);
  const size_t base_blocks_per_stream = num_blocks / P;
  const size_t remainder = num_blocks % P;
  size_t req_offset = 0;
  for (int i = 0; i < P; ++i) {
    const size_t block_offset =
        i * base_blocks_per_stream + std::min<size_t>(i, remainder);

    size_t req_end = req_offset;
    while (req_end < requests.size() && requests[req_end].stream_idx == i) {
      ++req_end;
    }

    const std::string remote_peer = peers[i % peers.size()];
    absl::Span<const Request> stream_requests =
        requests.subspan(req_offset, req_end - req_offset);
    req_offset = req_end;

    absl::Status status =
        PostGrpcPushInternal(remote_peer, stream_requests, src_block_ids,
                             dst_block_ids, block_offset, allocated_ids);
    if (!status.ok()) {
      return ReportError(on_complete, status);
    }
  }

  const std::chrono::steady_clock::time_point push_end_ts =
      std::chrono::steady_clock::now();
  RecordP2pTransferTime(push_start_ts, push_end_ts, src_ip, dst_ip);
  if (on_complete) {
    on_complete(std::move(allocated_ids));
  }
  return 0;
}

absl::Status GrpcTransportAdapter::PostGrpcPushInternal(
    absl::string_view peer, absl::Span<const Request> requests,
    absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
    size_t block_offset, std::vector<int>& allocated_ids) {
  if (requests.empty()) {
    return absl::OkStatus();
  }

  const auto& first = requests.front();
  const uint8_t socket_opcode = first.socket_opcode;
  const uint32_t count_or_size = first.count_or_size;
  const size_t block_count = static_cast<size_t>(count_or_size);

  auto stub = GetOrCreateStub(peer);
  grpc::ClientContext context;
  context.set_deadline(absl::ToChronoTime(absl::Now() + kDefaultRpcDeadline));

  proto::GrpcPushRequest req_msg;
  if (socket_opcode == 6) {
    ABSL_DCHECK_LE(block_offset + block_count, dst_block_ids.size());
    ABSL_DCHECK_LE(block_offset + block_count, src_block_ids.size());
    for (size_t k = 0; k < block_count; ++k) {
      req_msg.add_dst_block_ids(dst_block_ids[block_offset + k]);
      req_msg.add_src_block_ids(src_block_ids[block_offset + k]);
    }
  }

  uint64_t stream_bytes_sent = 0;
  for (size_t i = 0; i < requests.size();) {
    size_t j = i;
    uint32_t total_size = 0;
    while (j < requests.size() &&
           requests[j].request_id == requests[i].request_id) {
      total_size += static_cast<uint32_t>(requests[j].len);
      ++j;
    }

    proto::ChunkProto* chunk = req_msg.add_chunks();
    *chunk = RequestToChunkProto(requests[i]);
    if (block_count > 0 && total_size > 0) {
      absl::Cord payload_cord;
      for (size_t idx = i; idx < j; ++idx) {
        if (requests[idx].len > 0) {
          payload_cord.Append(absl::MakeCordFromExternal(
              absl::string_view(
                  reinterpret_cast<const char*>(requests[idx].laddr),
                  requests[idx].len),
              [](absl::string_view) {}));
        }
      }
      *chunk->mutable_payload() = std::move(payload_cord);
      stream_bytes_sent += total_size;
    }
    i = j;
  }

  proto::GrpcPushResponse resp_msg;
  grpc::Status rpc_status = stub->PushBlocks(&context, req_msg, &resp_msg);
  if (!rpc_status.ok()) {
    return GrpcStatusToAbsl(rpc_status);
  }

  if (!resp_msg.ack()) {
    return absl::InternalError(!resp_msg.error_message().empty()
                                   ? resp_msg.error_message()
                                   : "Push verification failed");
  }

  if (socket_opcode == 6) {
    for (size_t k = 0; k < block_count; ++k) {
      allocated_ids[block_offset + k] = dst_block_ids[block_offset + k];
    }
  } else {
    const auto& resp_ids = resp_msg.allocated_block_ids();
    if (static_cast<size_t>(resp_ids.size()) < block_count) {
      return absl::InternalError("Insufficient allocated block IDs returned");
    }
    for (size_t k = 0; k < block_count; ++k) {
      ABSL_DCHECK_LT(block_offset + k, allocated_ids.size());
      allocated_ids[block_offset + k] = resp_ids[k];
    }
  }

  if (stream_bytes_sent > 0) {
    RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
        metric_names::kSentBytesTotal, kPushLabels, stream_bytes_sent);
  }
  return absl::OkStatus();
}

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_GRPC_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_GRPC_TRANSPORT_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/grpc_transport_service.grpc.pb.h"
#include "tpu_sync/transport/lib/grpc_transport_service.pb.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

// Returns true if gRPC transport adapter is enabled via environment variable
// TPU_RAIDEN_DATA_TRANSPORT=grpc (case-insensitive) or "1".
bool UseGrpcTransportAdapter();

// Converts between Request / ChunkHeader and protobuf ChunkProto.
proto::ChunkProto ChunkHeaderToProto(const ChunkHeader& header);
ChunkHeader ProtoToChunkHeader(const proto::ChunkProto& proto);
proto::ChunkProto RequestToChunkProto(const Request& req);

// Converts between grpc::Status and absl::Status.
absl::Status GrpcStatusToAbsl(const grpc::Status& status);
grpc::Status AbslStatusToGrpc(const absl::Status& status);

// Custom server-side push handler invoked by
// GrpcTransportServiceImpl::PushBlocks.
// - `header`: Initial ChunkHeader from GrpcPushRequest.
// - `dst_block_ids`: Destination block IDs (populated for Op 6).
// - `src_block_ids`: Source block IDs (populated for Op 6).
// - `send_handshake`: Callback to send the handshake ACK and (for Op 1) the
//   allocated destination block IDs before reading payload chunks.
// - `read_next_payload`: Callback to read the next coalesced payload buffer.
using GrpcCustomPushHandler = std::function<absl::Status(
    const ChunkHeader& header, absl::Span<const int> dst_block_ids,
    absl::Span<const int> src_block_ids,
    std::function<absl::Status(absl::Span<const int>)> send_handshake,
    std::function<absl::StatusOr<absl::Cord>()> read_next_payload)>;

// gRPC service implementation for GrpcTransportService.
class GrpcTransportServiceImpl final
    : public proto::GrpcTransportService::Service {
 public:
  explicit GrpcTransportServiceImpl(GrpcCustomPushHandler push_handler);
  ~GrpcTransportServiceImpl() override = default;

  grpc::Status PushBlocks(grpc::ServerContext* context,
                          const proto::GrpcPushRequest* request,
                          proto::GrpcPushResponse* response) override;

 private:
  GrpcCustomPushHandler push_handler_;
};

// Embedded gRPC server wrapper for GrpcTransportService.
class GrpcTransportServer {
 public:
  explicit GrpcTransportServer(GrpcCustomPushHandler push_handler,
                               int requested_port = 0);
  explicit GrpcTransportServer(proto::GrpcTransportService::Service* service,
                               int requested_port = 0);
  ~GrpcTransportServer();

  void Shutdown();
  int local_port() const { return bound_port_; }
  std::string address() const;

 private:
  void StartServer(proto::GrpcTransportService::Service* service,
                   int requested_port);

  std::unique_ptr<GrpcTransportServiceImpl> owned_service_;
  std::unique_ptr<grpc::Server> server_;
  int bound_port_ = 0;
};

// gRPC implementation of TransportAdapter.
class GrpcTransportAdapter : public TransportAdapter {
 public:
  explicit GrpcTransportAdapter(RawBufferTransport* raw_transport = nullptr,
                                int parallelism = 1);
  ~GrpcTransportAdapter() override;

  absl::StatusOr<Handle> Post(
      absl::Span<const std::string> peers, absl::Span<const Request> requests,
      absl::Span<const int> src_block_ids = {},
      absl::Span<const int> dst_block_ids = {},
      CompletionCallback on_complete = nullptr) override;

  absl::StatusOr<Status> Poll(Handle handle) override;

 private:
  std::shared_ptr<proto::GrpcTransportService::StubInterface> GetOrCreateStub(
      absl::string_view peer);

  // Block-level gRPC Push Operations (Op 1, Op 6).
  absl::StatusOr<Handle> PostGrpcPush(absl::Span<const std::string> peers,
                                      absl::Span<const Request> requests,
                                      absl::Span<const int> src_block_ids,
                                      absl::Span<const int> dst_block_ids,
                                      CompletionCallback on_complete);

  absl::Status PostGrpcPushInternal(absl::string_view peer,
                                    absl::Span<const Request> requests,
                                    absl::Span<const int> src_block_ids,
                                    absl::Span<const int> dst_block_ids,
                                    size_t block_offset,
                                    std::vector<int>& allocated_ids);

 private:
  RawBufferTransport* const raw_transport_;
  const int parallelism_;

  struct PeerStubPool {
    std::vector<std::shared_ptr<proto::GrpcTransportService::StubInterface>>
        stubs;
    std::atomic<size_t> next_idx{0};
  };

  absl::Mutex stub_mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<PeerStubPool>> stub_pools_
      ABSL_GUARDED_BY(stub_mu_);
};

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_GRPC_TRANSPORT_ADAPTER_H_

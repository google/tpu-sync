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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_SERVICE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/server.h"
#include "tpu_sync/platform/grpc_credential_factory.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {

// Delegate used by GrpcTransportServer to resolve destination host buffers,
// receive push completion notifications, and serve pull requests.
// Implementations must be thread-safe for concurrent RPC handlers.
class GrpcTransportDelegate {
 public:
  virtual ~GrpcTransportDelegate() = default;

  virtual absl::StatusOr<absl::Span<uint8_t>> GetHostBuffer(
      uint64_t buffer_id, uint32_t shard_id) = 0;

  virtual absl::Status OnPushChunks(uint64_t uuid, uint64_t buffer_id,
                                    uint32_t shard_id,
                                    uint64_t chunk_batch_id) {
    return absl::OkStatus();
  }

  virtual absl::StatusOr<PullChunksResponse> OnPullChunks(
      const PullChunksRequest& request) {
    return absl::UnimplementedError(
        "GrpcTransportDelegate::OnPullChunks is not implemented");
  }
};

struct GrpcTransportServerConfig {
  // TODO(justinlu): Auto-chunk large transfers to bound per-RPC receiver memory
  // instead of relying on a large max message size.
  static constexpr int kDefaultMaxMessageSizeBytes = 2'000'000'000;  // 2 GB

  int keepalive_time_ms = 10000;
  int keepalive_timeout_ms = 5000;
  int min_recv_ping_interval_without_data_ms = 5000;
  int max_message_size_bytes = kDefaultMaxMessageSizeBytes;
  bool enable_tcp_tx_zerocopy = true;
};

// Standalone callback-based asynchronous gRPC server hosting
// GrpcTransportService. Thread-safe for Start()/Stop()/local_port()/address().
class GrpcTransportServer {
 public:
  explicit GrpcTransportServer(
      GrpcTransportDelegate* delegate, GrpcTransportServerConfig config = {},
      std::shared_ptr<GrpcCredentialFactory> credential_factory = nullptr);
  ~GrpcTransportServer();

  GrpcTransportServer(const GrpcTransportServer&) = delete;
  GrpcTransportServer& operator=(const GrpcTransportServer&) = delete;

  absl::StatusOr<int> Start(int requested_port = 0);
  void Stop();

  int local_port() const;
  std::string address() const;

 private:
  class GrpcTransportServiceImpl;

  GrpcTransportServerConfig config_;
  std::shared_ptr<GrpcCredentialFactory> credential_factory_;
  std::unique_ptr<GrpcTransportServiceImpl> service_;

  mutable absl::Mutex mu_;
  std::unique_ptr<grpc::Server> server_ ABSL_GUARDED_BY(mu_);
  int bound_port_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_SERVICE_H_

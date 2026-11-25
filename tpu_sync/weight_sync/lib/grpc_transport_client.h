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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/platform/grpc_credential_factory.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {

struct GrpcTransportClientConfig {
  // TODO(justinlu): Auto-chunk large transfers to bound per-RPC receiver memory
  // instead of relying on a large max message size.
  static constexpr int kDefaultMaxMessageSizeBytes = 2'000'000'000;  // 2 GB

  int num_channels_per_peer = 1;
  absl::Duration default_timeout = absl::Seconds(120);
  int keepalive_time_ms = 10000;
  int keepalive_timeout_ms = 5000;
  int max_message_size_bytes = kDefaultMaxMessageSizeBytes;
  bool enable_tcp_tx_zerocopy = true;
  int tcp_tx_zerocopy_send_bytes_threshold = 0;  // 0 = gRPC default.
};

// Thread-safe gRPC client for GrpcTransportService with per-peer channel
// pooling.
class GrpcTransportClient {
 public:
  explicit GrpcTransportClient(
      GrpcTransportClientConfig config = {},
      std::shared_ptr<GrpcCredentialFactory> credential_factory = nullptr);
  ~GrpcTransportClient();

  GrpcTransportClient(const GrpcTransportClient&) = delete;
  GrpcTransportClient& operator=(const GrpcTransportClient&) = delete;

  // Sends |request| via zero-copy external Cords referencing caller-owned spans
  // until return.
  absl::Status PushChunks(absl::string_view peer,
                          const PushChunksRequest& request,
                          absl::Duration timeout = absl::ZeroDuration());

  // Streams chunks from |peer| and writes them directly into |dst_buffer|.
  absl::Status PullChunks(absl::string_view peer,
                          const PullChunksRequest& request,
                          absl::Span<uint8_t> dst_buffer,
                          absl::Duration timeout = absl::ZeroDuration());

  size_t CachedChannelCount(absl::string_view peer) const;

 private:
  struct PeerStubPool;

  absl::StatusOr<std::shared_ptr<PeerStubPool>> GetOrCreatePool(
      absl::string_view peer);

  GrpcTransportClientConfig config_;
  std::shared_ptr<GrpcCredentialFactory> credential_factory_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<PeerStubPool>> stub_pools_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_CLIENT_H_

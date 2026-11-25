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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_GRPC_TRANSPORT_DELEGATE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_GRPC_TRANSPORT_DELEGATE_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_service.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {

// Concrete thread-safe GrpcTransportDelegate that resolves destination host
// buffers via a caller-supplied provider, tracks active TransferSessions by
// transfer |uuid|, and routes incoming PushChunks and PullChunks RPCs by first
// resolving the session via GetSession() and delegating to session APIs.
//
// Lock ordering: internal |mu_| is a leaf lock and is never held across
// |host_buffer_provider_| or TransferSession/WeightSendSession calls.
class WeightGrpcTransportDelegate final : public GrpcTransportDelegate {
 public:
  using HostBufferProvider = std::function<absl::StatusOr<absl::Span<uint8_t>>(
      uint64_t buffer_id, uint32_t shard_id)>;

  explicit WeightGrpcTransportDelegate(HostBufferProvider host_buffer_provider);
  ~WeightGrpcTransportDelegate() override = default;

  WeightGrpcTransportDelegate(const WeightGrpcTransportDelegate&) = delete;
  WeightGrpcTransportDelegate& operator=(const WeightGrpcTransportDelegate&) =
      delete;

  // Registers a transfer |session| (send or receive) for |uuid|.
  // Returns InvalidArgumentError if |session| is null, or AlreadyExistsError if
  // |uuid| is already registered.
  absl::Status RegisterSession(
      uint64_t uuid, std::shared_ptr<::tpu_raiden::TransferSession> session);

  // Unregisters the transfer session registered for |uuid|.
  // Returns NotFoundError if no session is registered for |uuid|.
  absl::Status UnregisterSession(uint64_t uuid);

  // GrpcTransportDelegate overrides:
  absl::StatusOr<absl::Span<uint8_t>> GetHostBuffer(uint64_t buffer_id,
                                                    uint32_t shard_id) override;

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> GetSession(
      uint64_t uuid) const override;

  absl::Status OnPushChunks(uint64_t uuid, uint64_t buffer_id,
                            uint32_t shard_id,
                            uint64_t chunk_batch_id) override;

  absl::StatusOr<PullChunksResponse> OnPullChunks(
      const PullChunksRequest& request) override;

 private:
  HostBufferProvider host_buffer_provider_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<::tpu_raiden::TransferSession>>
      sessions_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_GRPC_TRANSPORT_DELEGATE_H_

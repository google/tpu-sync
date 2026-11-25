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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SEND_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SEND_SESSION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {

struct WeightSendSessionOptions {
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
  // Optional owner keeping source host buffers alive until transfer completion.
  std::shared_ptr<const void> host_buffer_owner = nullptr;
};

// Base class for whole-model weight send sessions implementing TransferSession
// from a StartTransferRequest spec (host-buffer transfer only; D2H is handled
// externally). Thread-safe.
class WeightSendSession : public ::tpu_raiden::TransferSession {
 public:
  using ConstHostBufferProvider =
      std::function<absl::StatusOr<absl::Span<const uint8_t>>(
          uint64_t buffer_id, uint32_t shard_id)>;

  // Creates an active push-based send session from |spec| using non-owning
  // |client| (which must outlive the session) and |host_buffer_provider| to
  // resolve source host buffers for each (layer_idx, src_shard_idx).
  static absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
  CreatePushSendSession(GrpcTransportClient* client,
                        const ::tpu_sync::rpc::StartTransferRequest& spec,
                        ConstHostBufferProvider host_buffer_provider,
                        WeightSendSessionOptions options = {});

  // Creates a passive pull-based send session from |spec| using
  // |host_buffer_provider| to resolve source host buffers for each (layer_idx,
  // src_shard_idx).
  static absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
  CreatePullSendSession(const ::tpu_sync::rpc::StartTransferRequest& spec,
                        ConstHostBufferProvider host_buffer_provider,
                        WeightSendSessionOptions options = {});

  ~WeightSendSession() override = default;

  uint64_t uuid() const { return uuid_; }

  // Serves an incoming PullChunks RPC for passive pull-based send sessions.
  // Push send sessions return FailedPreconditionError. Thread-safe.
  virtual absl::StatusOr<PullChunksResponse> OnPullChunks(
      const PullChunksRequest& request);

  bool Done() const override;
  void Finish(const absl::Status& status = absl::OkStatus()) override;
  // Send sessions never accept external incoming pushes.
  bool TryBeginOp() override { return false; }
  void EndOp() override;
  absl::Status GetStatus() const override;
  absl::Status AwaitForDone() override;
  bool IsDraining() const override;
  absl::StatusOr<size_t> ResolveStagingSlot(size_t unit) override;
  // Send sessions never receive data; always returns FailedPreconditionError.
  absl::Status OnDataReceived(size_t major_dimension_idx,
                              absl::Span<const int> minor_dimension_indices,
                              absl::Span<const int> shard_ids) override;
  tsl::Future<> GetDataReadyFuture(size_t major_dimension_idx) override;
  absl::optional<std::shared_ptr<const ::tpu_raiden::TransferPlan>> plan()
      const override;
  std::chrono::steady_clock::time_point deadline() const override;

 protected:
  WeightSendSession(uint64_t uuid,
                    std::chrono::steady_clock::time_point deadline,
                    std::shared_ptr<const void> host_buffer_owner = nullptr);

  absl::Status MarkStarted(bool allow_already_succeeded = false);
  bool TryBeginInternalOp();
  void EndInternalOp();

  const uint64_t uuid_;
  const std::chrono::steady_clock::time_point deadline_;
  std::shared_ptr<const void> host_buffer_owner_;

  mutable absl::Mutex mu_;
  bool started_ ABSL_GUARDED_BY(mu_) = false;
  size_t in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SEND_SESSION_H_

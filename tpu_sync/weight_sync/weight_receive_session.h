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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_RECEIVE_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_RECEIVE_SESSION_H_

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

namespace tpu_raiden::weight_sync {

struct WeightReceiveSessionOptions {
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
  std::function<absl::Status(uint64_t layer_idx)> on_layer_received = nullptr;
};

// Base class for whole-model weight receive sessions implementing
// TransferSession from a StartTransferRequest spec (host-buffer transfer only;
// H2D is handled externally). Thread-safe.
class WeightReceiveSession : public ::tpu_raiden::TransferSession {
 public:
  using HostBufferProvider = std::function<absl::StatusOr<absl::Span<uint8_t>>(
      uint64_t buffer_id, uint32_t shard_id)>;
  using OnLayerReceivedCallback =
      std::function<absl::Status(uint64_t layer_idx)>;

  // Creates an active pull-based receive session from |spec| using non-owning
  // |client| (which must outlive the session) and |host_buffer_provider| to
  // resolve destination host buffers for each (layer_idx, dst_shard_idx).
  static absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
  CreatePullReceiveSession(GrpcTransportClient* client,
                           const ::tpu_sync::rpc::StartTransferRequest& spec,
                           HostBufferProvider host_buffer_provider,
                           WeightReceiveSessionOptions options = {});

  // Creates a passive push-based receive session from |spec|, tracking expected
  // chunk batches per layer from |spec.expected_layer_chunk_counts()|.
  static absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
  CreatePushReceiveSession(const ::tpu_sync::rpc::StartTransferRequest& spec,
                           WeightReceiveSessionOptions options = {});

  ~WeightReceiveSession() override = default;

  uint64_t uuid() const { return uuid_; }

  bool Done() const override;
  void Finish(const absl::Status& status = absl::OkStatus()) override;
  bool TryBeginOp() override;
  void EndOp() override;
  absl::Status GetStatus() const override;
  absl::Status AwaitForDone() override;
  bool IsDraining() const override;
  absl::StatusOr<size_t> ResolveStagingSlot(size_t unit) override;
  tsl::Future<> GetDataReadyFuture(size_t major_dimension_idx) override;
  absl::optional<std::shared_ptr<const ::tpu_raiden::TransferPlan>> plan()
      const override;
  std::chrono::steady_clock::time_point deadline() const override;

 protected:
  WeightReceiveSession(uint64_t uuid,
                       std::chrono::steady_clock::time_point deadline,
                       OnLayerReceivedCallback on_layer_received = nullptr);

  // Marks the session as started; returns FailedPreconditionError if already
  // started, draining, or done.
  absl::Status MarkStarted();

  // Internal operation reference counting for subclasses that reject external
  // TryBeginOp() calls.
  bool TryBeginInternalOp();
  void EndInternalOp();

  // Invokes |on_layer_received_| if set.
  absl::Status NotifyLayerReceived(uint64_t layer_idx);

  const uint64_t uuid_;
  const std::chrono::steady_clock::time_point deadline_;
  OnLayerReceivedCallback on_layer_received_;

  mutable absl::Mutex mu_;
  bool started_ ABSL_GUARDED_BY(mu_) = false;
  size_t in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_RECEIVE_SESSION_H_

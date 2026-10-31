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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_RECEIVE_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_RECEIVE_SESSION_H_

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/kv_cache/transfer_session.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {

// Encapsulates the per-transfer state and execution lifecycle of a consumer
// receive operation (pull/push H2D receives), including host staging
// ownership, reference-counted drain and sticky-failure tracking, block/layer
// readiness accounting, and H2D copy execution.
//
// Thread-safe: all mutable session state is synchronized via internal |mu_|.
class TransferReceiveSession
    : public TransferSession,
      public std::enable_shared_from_this<TransferReceiveSession> {
 public:
  static absl::StatusOr<std::shared_ptr<TransferReceiveSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      const std::string& req_id, const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids,
      const std::optional<std::vector<int64_t>>& local_host_block_ids,
      std::chrono::steady_clock::time_point deadline,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  static absl::StatusOr<std::shared_ptr<TransferReceiveSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      std::string req_id, int32_t total_blocks,
      std::chrono::steady_clock::time_point deadline,
      bool acquire_staging = false,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  static absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
  CreateFromActivePlan(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      const ::tpu_sync::rpc::StartTransferRequest& request,
      std::chrono::steady_clock::time_point deadline,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  ~TransferReceiveSession() override { ReleaseStaging(); }

  bool Done() const override {
    absl::MutexLock lock(mu_);
    return done_;
  }

  void Finish(const absl::Status& status = absl::OkStatus()) override;

  absl::Status GetStatus() const override {
    absl::MutexLock lock(mu_);
    return status_;
  }

  absl::Status AwaitForDone() override;

  // Returns transfer completion and failure details. Only valid after Done().
  TransferDetails Details() const;

  bool IsDraining() const override {
    absl::MutexLock lock(mu_);
    return draining_;
  }

  absl::optional<std::shared_ptr<const TransferPlan>> plan() const override {
    if (transfer_plan_ == nullptr || Done()) {
      return absl::nullopt;
    }
    return transfer_plan_;
  }

  bool TryBeginOp() override {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) return false;
    ++in_flight_;
    return true;
  }
  void EndOp() override;

  // Returns true if network/layer transfer is complete and all H2D futures are
  // ready.
  bool IsReadyToComplete() const;

  // Starts the consumer pull handshake through the control backend without
  // blocking, and updates session state upon completion or error.
  void ExecutePullRequest(KVCacheManagerWithTransfer& manager,
                          const std::string& remote_endpoint);

  // Peregrine receive hook: records first-packet timing the first time any
  // Post for this session lands.
  absl::Status OnRecvProgress();

  // Peregrine receive hook: marks the network phase complete and records
  // last-packet timing once every layer's Posts have landed on this receiver.
  absl::Status OnRecvNetworkComplete();

  // Handles a transport report that |shard_ids| of |block_ids| landed. Each
  // shard's progress is tracked separately; a shard completes once it has
  // received every (layer, block) of the session, and the network phase
  // completes once every shard has. Routed pushes report a block range once
  // per shard subset, so a shard's count only advances on streams carrying it.
  absl::Status OnBlockShardsReceived(const std::vector<int>& block_ids,
                                     absl::Span<const int> shard_ids);

  // Issues H2D copy for |layer_idx| using |base_| and registers the completion
  // callback. When fewer host layer buffers exist than layers, the copy reads
  // the host layer buffer leased to |layer_idx| and the callback releases the
  // lease.
  absl::Status ExecuteLayerH2d(size_t layer_idx);

  // Leases a buffer to |layer_idx| on first use.
  absl::StatusOr<size_t> ResolveStagingSlot(size_t layer_idx) override {
    return AcquireStagingSlot(layer_idx);
  }

  const StagingLayerManager& staging_layer_manager() const {
    return staging_layer_manager_;
  }

  std::string req_id() const {
    absl::MutexLock lock(mu_);
    return req_id_;
  }
  int32_t total_blocks() const {
    absl::MutexLock lock(mu_);
    return total_blocks_;
  }
  bool tracks_completion() const {
    absl::MutexLock lock(mu_);
    return tracks_completion_;
  }
  bool HasPendingWork() const {
    absl::MutexLock lock(mu_);
    return (tracks_completion_ && !done_) || in_flight_ > 0;
  }
  std::chrono::steady_clock::time_point deadline() const {
    absl::MutexLock lock(mu_);
    return deadline_;
  }

 private:
  // Returns the host layer buffer leased to |layer_idx|, leasing one when it
  // has none and blocking while every buffer is in use. A timeout fails the
  // receive. Callable from any thread; concurrent calls for the same layer
  // (one per shard stream) observe one lease.
  absl::StatusOr<size_t> AcquireStagingSlot(size_t layer_idx);

  // Returns the host layer buffer currently leased to |layer_idx|, or
  // kFailedPrecondition if |layer_idx| holds no lease.
  absl::StatusOr<size_t> LookupStagingSlot(size_t layer_idx) const;

  TransferReceiveSession(kv_cache::KVCacheManagerBase* base,
                         StagingBlockAllocator* staging_allocator,
                         uint64_t uuid,
                         std::shared_ptr<MetricsCollector> metrics_collector,
                         std::shared_ptr<const TransferPlan> plan = nullptr)
      : base_(base),
        staging_allocator_(staging_allocator),
        uuid_(uuid),
        metrics_collector_(std::move(metrics_collector)),
        transfer_plan_(std::move(plan)),
        blocks_received_per_shard_(base != nullptr ? base->num_shards() : 0, 0),
        staging_layer_manager_(base != nullptr && base->num_host_layers() <
                                                      base->num_layers()
                                   ? base->num_host_layers()
                                   : 0) {}
  TransferReceiveSession(kv_cache::KVCacheManagerBase* base,
                         StagingBlockAllocator* staging_allocator,
                         uint64_t uuid, std::string req_id,
                         int32_t total_blocks,
                         std::chrono::steady_clock::time_point deadline,
                         bool acquire_staging,
                         std::shared_ptr<MetricsCollector> metrics_collector)
      : base_(base),
        staging_allocator_(staging_allocator),
        uuid_(uuid),
        metrics_collector_(std::move(metrics_collector)),
        req_id_(std::move(req_id)),
        total_blocks_(total_blocks),
        blocks_received_per_shard_(base != nullptr ? base->num_shards() : 0, 0),
        deadline_(deadline),
        start_time_(std::chrono::steady_clock::now()),
        staging_layer_manager_(base != nullptr && base->num_host_layers() <
                                                      base->num_layers()
                                   ? base->num_host_layers()
                                   : 0) {
    if (acquire_staging) {
      absl::StatusOr<StagingAllocation> acquired =
          staging_allocator_->Acquire(1);
      if (acquired.ok()) {
        staging_ = *std::move(acquired);
      }
    }
  }

  // Initializes this receive session for an HBM destination active plan.
  void InitFromActivePlan(
      const ::tpu_sync::rpc::StartTransferRequest& request,
      std::chrono::steady_clock::time_point deadline, StagingAllocation staging,
      const absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>&
          host_block_of);
  // Allocates host staging for a consumer StartRead load plan via
  // |staging_allocator_| (unless |local_host_block_ids| is explicitly provided)
  // and populates |*host_block_ids|. Returns false if staging capacity is
  // unavailable.
  bool AllocateStagingForLoad(
      const std::string& req_id, const std::vector<int64_t>& local_block_ids,
      const std::optional<std::vector<int64_t>>& local_host_block_ids,
      std::vector<int64_t>* host_block_ids);

  // Builds the transport and H2D copy plan for a consumer StartRead request.
  static CopyPlan BuildLoadCopyPlan(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids,
      const std::vector<int64_t>& local_host_block_ids);

  // Initializes this receive session for a consumer StartRead load plan.
  void InitFromLoadPlan(const std::string& req_id, CopyPlan load_plan,
                        std::chrono::steady_clock::time_point deadline);
  using H2dIssueFuture =
      std::shared_future<absl::StatusOr<raiden::PjRtCopyFuture>>;

  void ReleaseStaging();
  void ReleaseStagingLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Transitions the session into draining/completed state. If |status| is
  // non-OK and this is the first failure, records |status| and attributes the
  // failure to |phase| (or infers the current phase via CurrentPhaseLocked()
  // if |phase| is omitted).
  void FinishLocked(const absl::Status& status = absl::OkStatus(),
                    std::optional<absl::string_view> phase = std::nullopt)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns the earliest execution phase of this receive still in progress,
  // as named in TransferDetails::phase.
  absl::string_view CurrentPhaseLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EndRecvOpLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  bool AllH2dDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordBlockShardsReceivedLocked(const std::vector<int>& block_ids,
                                       absl::Span<const int> shard_ids,
                                       bool* first_packet,
                                       bool* network_just_completed)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ResetShardProgressLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordNetworkCompleteLocked(
      bool all_complete, std::string* session_req_id,
      std::chrono::steady_clock::time_point* session_start_time)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* base_ = nullptr;
  StagingBlockAllocator* staging_allocator_ = nullptr;
  uint64_t uuid_ = 0;
  std::shared_ptr<MetricsCollector> metrics_collector_;
  std::string req_id_ ABSL_GUARDED_BY(mu_);
  StagingAllocation staging_ ABSL_GUARDED_BY(mu_);
  CopyPlan load_plan_ ABSL_GUARDED_BY(mu_);
  CopySpec h2d_copy_ ABSL_GUARDED_BY(mu_);
  std::vector<int64_t> chip_block_ids_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<kv_cache::HostBlockId, kv_cache::DeviceBlockId>
      host_to_chip_ ABSL_GUARDED_BY(mu_);
  std::vector<H2dIssueFuture> h2d_dispatch_futures_ ABSL_GUARDED_BY(mu_);
  int32_t total_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  bool tracks_completion_ ABSL_GUARDED_BY(mu_) = true;
  // Per-shard count of (layer, block) units reported so far, indexed by shard;
  // shard `s` is complete at total_blocks_ * num_layers.
  std::vector<int64_t> blocks_received_per_shard_ ABSL_GUARDED_BY(mu_);
  // Shards whose count reached completion; network done at num_shards.
  int32_t num_completed_shards_ ABSL_GUARDED_BY(mu_) = 0;
  bool any_blocks_received_ ABSL_GUARDED_BY(mu_) = false;
  int32_t num_completed_layers_ ABSL_GUARDED_BY(mu_) = 0;
  bool network_completed_ ABSL_GUARDED_BY(mu_) = false;
  bool h2d_started_ ABSL_GUARDED_BY(mu_) = false;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  // The transfer phase in which the receive failed; populated when `status_` is
  // set to a non-OK status.
  absl::string_view failed_phase_ ABSL_GUARDED_BY(mu_);
  // True while a pulled receive is waiting for the producer to accept its pull
  // request.
  bool handshake_pending_ ABSL_GUARDED_BY(mu_) = false;
  // The producer control endpoint to which a pulled receive sent its request.
  std::string peer_endpoint_ ABSL_GUARDED_BY(mu_);
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  std::chrono::steady_clock::time_point completed_at_ ABSL_GUARDED_BY(mu_);
  std::vector<int> accumulated_host_block_ids_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point deadline_ ABSL_GUARDED_BY(mu_);
  const std::shared_ptr<const TransferPlan> transfer_plan_;
  std::chrono::steady_clock::time_point start_time_ ABSL_GUARDED_BY(mu_);
  std::vector<raiden::PjRtCopyFuture> h2d_futures_ ABSL_GUARDED_BY(mu_);
  // staging_layer_manager_ synchronizes itself; cancelled_ mirrors draining_
  // so waiters in staging_layer_manager_.Acquire() can bail out without taking
  // mu_.
  StagingLayerManager staging_layer_manager_;
  std::atomic<bool> cancelled_{false};
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_RECEIVE_SESSION_H_

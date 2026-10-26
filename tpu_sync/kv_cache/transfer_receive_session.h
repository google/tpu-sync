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

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/staging_window.h"
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
      std::optional<StagingWindowConfig> staging_window_config = std::nullopt);

  static std::shared_ptr<TransferReceiveSession> CreateQueued(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      const std::string& req_id, const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids,
      const std::optional<std::vector<int64_t>>& local_host_block_ids,
      std::chrono::steady_clock::time_point deadline,
      std::optional<StagingWindowConfig> staging_window_config = std::nullopt);

  static absl::StatusOr<std::shared_ptr<TransferReceiveSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      std::string req_id, int32_t total_blocks,
      std::chrono::steady_clock::time_point deadline,
      bool acquire_staging = false);

  static absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
  CreateFromActivePlan(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      const ::tpu_sync::rpc::StartTransferRequest& request, uint64_t generation,
      std::chrono::steady_clock::time_point deadline,
      absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>*
          host_block_of);

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

  // Releases any held staging slot or dynamic host blocks. Safe to call
  // multiple times.
  bool HasStaging() const {
    absl::MutexLock lock(mu_);
    return !staging_.empty();
  }

  void ReleaseStaging();

  bool TryBeginRecvOp() {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) return false;
    ++in_flight_;
    return true;
  }
  void EndRecvOp();

  // Marks the plan to be unregistered when the receive settles, returning true
  // if the receive is still active and will unregister on settle, or false if
  // it is already done.
  bool DeferUnregisterOnSettle();

  // Atomically claims any pending settle-unregister request and writes its
  // plan generation to |generation|.
  bool TakePendingUnregister(uint64_t* generation);

  // Returns true if all units have completed and their H2D copies are ready
  // (units_complete == L * n_chunks && h2d_issued == L * n_chunks &&
  // h2d_ready == h2d_issued).
  bool IsReadyToComplete() const;

  // Configures bounded-window chunking parameters and per-chunk H2D sub-specs.
  // |n_chunks| must be >= 1. When per-chunk sub-specs are provided,
  // |chunk_h2d_specs[c][layer_idx]| is dispatched for unit (c, layer_idx);
  // otherwise the base H2D copy plan is used.
  absl::Status SetChunkH2dSpecs(
      int32_t n_chunks, std::vector<std::vector<CopySpec>> chunk_h2d_specs = {},
      int32_t chunk_blocks = 0, int32_t window_chunks = 0,
      std::optional<ChunkPlan> chunk_plan = std::nullopt);

  bool IsWindowed() const {
    absl::MutexLock lock(mu_);
    return window_chunks_ > 0 && chunk_plan_.has_value();
  }

  std::vector<int64_t> transport_host_block_ids() const {
    absl::MutexLock lock(mu_);
    return load_plan_.transport_host_block_ids;
  }

  // Gates an incoming push stream for a windowed receive session before the
  // handshake ack is sent. Validates that the stream's destination blocks
  // belong to the current unit's ring slot and that no (position, shard)
  // pair is claimed more than once within the unit.
  absl::Status AwaitIncomingPushSlot(int32_t layer_idx,
                                     absl::Span<const int64_t> dst_block_ids,
                                     absl::Span<const int32_t> stream_shards,
                                     absl::Time deadline);

  int32_t ExpectedPushesPerLayer() const;

  int32_t units_complete() const {
    absl::MutexLock lock(mu_);
    return units_complete_;
  }
  int32_t h2d_issued() const {
    absl::MutexLock lock(mu_);
    return h2d_issued_;
  }
  int32_t h2d_ready() const {
    absl::MutexLock lock(mu_);
    return h2d_ready_;
  }
  int32_t n_chunks() const {
    absl::MutexLock lock(mu_);
    return n_chunks_;
  }
  int32_t chunk_blocks() const {
    absl::MutexLock lock(mu_);
    return chunk_blocks_;
  }
  int32_t window_chunks() const {
    absl::MutexLock lock(mu_);
    return window_chunks_;
  }

  // Starts the consumer pull handshake through the control backend without
  // blocking, and updates session state upon completion or error.
  void ExecutePullRequest(KVCacheManagerWithTransfer& manager,
                          const std::string& remote_endpoint);

  // Downgrades a windowed receive session to exact-k.
  // Releases the staging allocation, acquires |k| blocks from
  // staging_allocator_, rebuilds the exact-k load plan, and clears chunking.
  absl::Status DowngradeToExactK();

  bool handshake_pending() const {
    absl::MutexLock lock(mu_);
    return handshake_pending_;
  }
  int in_flight() const {
    absl::MutexLock lock(mu_);
    return in_flight_;
  }

  // Handles a transport report that |shard_ids| of |block_ids| landed. Each
  // shard's progress is tracked separately; a shard completes once it has
  // received every (layer, block) of the session, and the network phase
  // completes once every shard has. Routed pushes report a block range once
  // per shard subset, so a shard's count only advances on streams carrying it.
  absl::Status OnBlockShardsReceived(KVCacheManagerWithTransfer& manager,
                                     const std::vector<int>& block_ids,
                                     absl::Span<const int> shard_ids);

  // Issues H2D copy for |layer_idx| using |base_| and registers the completion
  // callback.
  absl::Status ExecuteLayerH2d(KVCacheManagerWithTransfer& manager,
                               size_t layer_idx);

  std::string req_id() const {
    absl::MutexLock lock(mu_);
    return req_id_;
  }
  int32_t total_blocks() const {
    absl::MutexLock lock(mu_);
    return total_blocks_;
  }
  std::chrono::steady_clock::time_point deadline() const {
    absl::MutexLock lock(mu_);
    return deadline_;
  }
  int32_t h2d_done(int32_t s, int32_t l) const {
    absl::MutexLock lock(mu_);
    if (s >= 0 && static_cast<size_t>(s) < h2d_done_.size() && l >= 0 &&
        static_cast<size_t>(l) < h2d_done_[s].size()) {
      return h2d_done_[s][l];
    }
    return 0;
  }
  // Seats a queued receive session once host staging memory is available:
  // acquires min(k, W*C) blocks, builds the load plan and chunk specs,
  // converts the admission op to the pull op, and starts the pull request.
  absl::Status SeatFromQueue(KVCacheManagerWithTransfer& manager);

  // Initializes this receive session for a consumer StartRead load plan.
  void InitFromLoadPlan(const std::string& req_id, CopyPlan load_plan,
                        std::chrono::steady_clock::time_point deadline);
  void InitFromLoadPlanLocked(const std::string& req_id, CopyPlan load_plan,
                              std::chrono::steady_clock::time_point deadline)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Mutates staging window config for fault-injection testing.
  void SetStagingWindowConfigForTesting(StagingWindowConfig config) {
    absl::MutexLock lock(mu_);
    staging_window_config_ = config;
  }

  bool is_queued() const {
    absl::MutexLock lock(mu_);
    return is_queued_;
  }
  int64_t ring_blocks_needed() const {
    absl::MutexLock lock(mu_);
    return ring_blocks_needed_;
  }
  std::chrono::steady_clock::time_point admission_deadline() const {
    absl::MutexLock lock(mu_);
    return admission_deadline_;
  }
  std::chrono::steady_clock::time_point start_time() const {
    absl::MutexLock lock(mu_);
    return start_time_;
  }

  int64_t gate_waits_over_1ms() const {
    absl::MutexLock lock(mu_);
    return gate_waits_over_1ms_;
  }
  absl::Time last_unit_progress() const {
    absl::MutexLock lock(mu_);
    return last_unit_progress_;
  }

 private:
  TransferReceiveSession(kv_cache::KVCacheManagerBase* base,
                         StagingBlockAllocator* staging_allocator,
                         uint64_t uuid = 0,
                         StagingWindowConfig staging_window_config = {})
      : base_(base),
        staging_allocator_(staging_allocator),
        uuid_(uuid),
        blocks_received_per_shard_(base != nullptr ? base->num_shards() : 0, 0),
        recv_gen_(base != nullptr ? base->num_layers() : 0, 0),
        staging_window_config_(staging_window_config) {}
  TransferReceiveSession(kv_cache::KVCacheManagerBase* base,
                         StagingBlockAllocator* staging_allocator,
                         uint64_t uuid, std::string req_id,
                         int32_t total_blocks,
                         std::chrono::steady_clock::time_point deadline,
                         bool acquire_staging = false)
      : base_(base),
        staging_allocator_(staging_allocator),
        uuid_(uuid),
        req_id_(std::move(req_id)),
        total_blocks_(total_blocks),
        blocks_received_per_shard_(base != nullptr ? base->num_shards() : 0, 0),
        recv_gen_(base != nullptr ? base->num_layers() : 0, 0),
        deadline_(deadline),
        start_time_(std::chrono::steady_clock::now()) {
    if (acquire_staging) {
      absl::StatusOr<StagingAllocation> acquired =
          staging_allocator_->Acquire(1);
      if (acquired.ok()) {
        staging_ = *std::move(acquired);
      }
    }
  }

  // Initializes this receive session for an HBM destination active plan,
  // allocating dynamic host staging via |staging_allocator_| when enabled.
  absl::Status InitFromActivePlan(
      const ::tpu_sync::rpc::StartTransferRequest& request, uint64_t generation,
      std::chrono::steady_clock::time_point deadline,
      absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>*
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

  absl::Status SetChunkH2dSpecsLocked(
      int32_t n_chunks, std::vector<std::vector<CopySpec>> chunk_h2d_specs = {},
      int32_t chunk_blocks = 0, int32_t window_chunks = 0,
      std::optional<ChunkPlan> chunk_plan = std::nullopt)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
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

  bool AllUnitsAndH2dDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordBlockShardsReceivedLocked(const std::vector<int>& block_ids,
                                       absl::Span<const int> shard_ids,
                                       bool* first_packet,
                                       bool* network_just_completed)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ResetShardProgressLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordNetworkCompleteLocked(
      MetricsCollector* absl_nullable metrics, bool all_complete,
      std::string* session_req_id,
      std::chrono::steady_clock::time_point* session_start_time)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* base_ = nullptr;
  StagingBlockAllocator* staging_allocator_ = nullptr;
  uint64_t uuid_ = 0;
  std::string req_id_ ABSL_GUARDED_BY(mu_);
  StagingAllocation staging_ ABSL_GUARDED_BY(mu_);
  CopyPlan load_plan_ ABSL_GUARDED_BY(mu_);
  CopySpec h2d_copy_ ABSL_GUARDED_BY(mu_);
  std::vector<int64_t> chip_block_ids_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<kv_cache::HostBlockId, kv_cache::DeviceBlockId>
      host_to_chip_ ABSL_GUARDED_BY(mu_);
  int32_t total_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t units_complete_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t h2d_issued_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t h2d_ready_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t n_chunks_ ABSL_GUARDED_BY(mu_) = 1;
  int32_t chunk_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t window_chunks_ ABSL_GUARDED_BY(mu_) = 0;
  StagingWindowConfig staging_window_config_ ABSL_GUARDED_BY(mu_);
  std::vector<int32_t> recv_gen_ ABSL_GUARDED_BY(mu_);
  std::vector<std::vector<CopySpec>> chunk_h2d_specs_ ABSL_GUARDED_BY(mu_);
  std::optional<ChunkPlan> chunk_plan_ ABSL_GUARDED_BY(mu_);
  // Per-layer set of claimed (position, shard) pairs for the current unit,
  // encoded as ((int64_t)position << 32) | (uint32_t)shard_id.
  std::vector<absl::flat_hash_set<int64_t>> claimed_ ABSL_GUARDED_BY(mu_);
  // Per-shard count of (layer, block) units reported so far, indexed by shard;
  // shard `s` is complete at total_blocks_ * num_layers.
  std::vector<int64_t> blocks_received_per_shard_ ABSL_GUARDED_BY(mu_);
  // Shards whose count reached completion; network done at num_shards.
  int32_t num_completed_shards_ ABSL_GUARDED_BY(mu_) = 0;
  bool any_blocks_received_ ABSL_GUARDED_BY(mu_) = false;
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
  std::chrono::steady_clock::time_point start_time_ ABSL_GUARDED_BY(mu_);
  // The plan is dropped when this receive settles: set for every
  // demand-staged receiver plan (whose mapping would otherwise outlive its
  // freed blocks) and when an unregister arrives while the receive is in
  // flight (the plan stays mapped until then so late pushes resolve
  // through its blocks).
  bool unregister_on_settle_ ABSL_GUARDED_BY(mu_) = false;
  // Generation of the plan this receive belongs to; settlement cleanup
  // only touches that registration.
  uint64_t plan_generation_ ABSL_GUARDED_BY(mu_) = 0;
  // Per-slot, per-layer completed H2D generation count. [slot s][layer l].
  std::vector<std::vector<int32_t>> h2d_done_ ABSL_GUARDED_BY(mu_);
  int64_t gate_waits_over_1ms_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<int32_t> last_gate_log_chunk_per_layer_ ABSL_GUARDED_BY(mu_);
  absl::Time last_unit_progress_ ABSL_GUARDED_BY(mu_) = absl::InfinitePast();
  bool is_queued_ ABSL_GUARDED_BY(mu_) = false;
  int64_t ring_blocks_needed_ ABSL_GUARDED_BY(mu_) = 0;
  std::chrono::steady_clock::time_point admission_deadline_
      ABSL_GUARDED_BY(mu_);
  std::vector<int64_t> queued_remote_block_ids_ ABSL_GUARDED_BY(mu_);
  std::vector<int64_t> queued_local_block_ids_ ABSL_GUARDED_BY(mu_);
  std::optional<std::vector<int64_t>> queued_local_host_block_ids_
      ABSL_GUARDED_BY(mu_);
  std::atomic<int64_t> total_gate_wait_us_{0};
  std::atomic<int64_t> total_h2d_dispatch_us_{0};
  std::atomic<int64_t> total_h2d_dma_us_{0};
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_RECEIVE_SESSION_H_

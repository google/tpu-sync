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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_RECEIVE_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_RECEIVE_SESSION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/kv_cache/transfer_session.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {

class StagingBlockAllocator;

// Encapsulates the consumer-side state and execution lifecycle of a multi-tag
// pool-reshard receive operation, including bounded pool staging lease
// teardown, reference-counted drain and sticky-failure tracking, per-pool
// readiness accounting, and order-ranked H2D copy execution.
//
// Thread-safe: all mutable session state is synchronized via internal |mu_|.
class ReshardReceiveSession
    : public TransferSession,
      public std::enable_shared_from_this<ReshardReceiveSession> {
 public:
  // Creates and initializes a consumer pool-reshard receive session for |plan|
  // and acquires bounded pool staging leases per storage via
  // |staging_allocator|.
  static absl::StatusOr<std::shared_ptr<ReshardReceiveSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_blocks,
      std::chrono::steady_clock::time_point deadline);

  ~ReshardReceiveSession() override { ReleaseStaging(); }

  bool Done() const override {
    absl::MutexLock lock(mu_);
    return done_;
  }

  void Finish(const absl::Status& status = absl::OkStatus()) override;

  bool TryBeginOp() override {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) return false;
    ++in_flight_;
    return true;
  }
  void EndOp() override;

  absl::Status GetStatus() const override {
    absl::MutexLock lock(mu_);
    return status_;
  }

  absl::Status AwaitForDone() override;

  bool IsDraining() const override {
    absl::MutexLock lock(mu_);
    return draining_;
  }

  const std::string& req_id() const { return req_id_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }
  // Position in the request's sub-transfer chain (both zero for a
  // single-plan request); see StartTransferRequest.sub_plan_index.
  int32_t sub_plan_index() const { return sub_plan_index_; }
  int32_t sub_plan_count() const { return sub_plan_count_; }

  // Handles completion of pool |pool_idx|.
  absl::Status OnDataReceived(size_t pool_idx) override {
    return OnPoolReceived(pool_idx);
  }

  // Leases a buffer to |pool_idx| on first use.
  absl::StatusOr<size_t> ResolveStagingSlot(size_t pool_idx) override {
    return AcquireStagingSlot(pool_idx);
  }

  absl::optional<std::shared_ptr<const TransferPlan>> plan() const override {
    if (transfer_plan_ == nullptr || Done()) {
      return absl::nullopt;
    }
    return transfer_plan_;
  }

 private:
  friend struct PoolReshardRecvTestPeer;
  friend struct ReshardReceiveSessionTestPeer;

  // Handles pool completion notifications for this pool-reshard receive
  // session.
  absl::Status OnPoolReceived(size_t pool_idx);

  // Returns the host staging buffer leased to |pool_idx|, leasing one when it
  // has none and waiting until deadline when every slot is leased.
  absl::StatusOr<size_t> AcquireStagingSlot(size_t pool_idx);

  // Returns the host staging buffer currently leased to |pool_idx|, or
  // kFailedPrecondition if |pool_idx| holds no lease.
  absl::StatusOr<size_t> LookupStagingSlot(size_t pool_idx) const;

  ReshardReceiveSession(kv_cache::KVCacheManagerBase* base,
                        StagingBlockAllocator* staging_allocator, uint64_t uuid,
                        TransferPlan transfer_plan,
                        const ::tpu_sync::rpc::StartTransferRequest& plan,
                        absl::Span<const int64_t> chip_blocks,
                        std::chrono::steady_clock::time_point deadline);

  static absl::Status ValidatePlan(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_blocks);
  static absl::Status ValidateReceiverCoverage(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan);

  absl::Status AcquireStagingLeases(
      const ::tpu_sync::rpc::StartTransferRequest& plan);

  void ReleaseStaging();
  void ReleaseStagingLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FinishLocked(const absl::Status& status = absl::OkStatus())
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EndRecvOpLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Handles completion of |pool_idx|'s H2D upload and finalizes the
  // pool-reshard receive when all pools have completed or on error.
  void FinishPoolH2d(size_t pool_idx, const absl::Status& status);

  absl::Status RecordPoolReceivedLocked(size_t pool_idx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::vector<std::pair<size_t, std::vector<int64_t>>>
  CollectEligiblePoolH2dsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordPoolH2dResultLocked(size_t pool_idx, const absl::Status& status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ExecuteEligiblePoolH2ds();

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  StagingBlockAllocator* const staging_allocator_ = nullptr;
  const uint64_t uuid_ = 0;
  const std::string req_id_;
  const int32_t sub_plan_index_ = 0;
  const int32_t sub_plan_count_ = 0;
  const std::chrono::steady_clock::time_point deadline_;
  const std::chrono::steady_clock::time_point start_time_;
  const std::shared_ptr<const TransferPlan> transfer_plan_;
  std::vector<int64_t> chip_block_ids_ ABSL_GUARDED_BY(mu_);
  bool staging_released_ ABSL_GUARDED_BY(mu_) = true;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  bool reshard_finalizing_ ABSL_GUARDED_BY(mu_) = false;
  absl::flat_hash_set<size_t> expected_pool_indices_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<size_t> started_pool_indices_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<size_t> completed_pool_indices_ ABSL_GUARDED_BY(mu_);
  // Multi-tag plans: per-pool H2D upload ordering. A pool's mirror is
  // uploaded only after every expected pool of a strictly lower order
  // rank has completed its upload (FA at rank 0, state classes at rank
  // 1, so state bytes land last on aliased arena pages). Single-tag
  // plans leave every rank 0 (upload immediately on wire completion).
  absl::flat_hash_map<size_t, int> pool_order_ranks_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<size_t> h2d_launched_pools_ ABSL_GUARDED_BY(mu_);
  // Multi-tag plans: each pool uploads only its own group's destination
  // block ids (the flat chip_block_ids list concatenates all groups).
  absl::flat_hash_map<size_t, std::vector<int64_t>> pool_dst_block_ids_
      ABSL_GUARDED_BY(mu_);
  StagingLayerManager staging_layer_manager_;
  std::atomic<bool> cancelled_{false};
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_RECEIVE_SESSION_H_

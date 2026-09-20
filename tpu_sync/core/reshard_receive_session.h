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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_RESHARD_RECEIVE_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_RESHARD_RECEIVE_SESSION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {

class KVCacheManagerWithTransfer;
class StagingBlockAllocator;

// Encapsulates the consumer-side state and execution lifecycle of a multi-tag
// pool-reshard receive operation, including bounded pool staging lease
// teardown, reference-counted drain and sticky-failure tracking, per-pool
// readiness accounting, and order-ranked H2D copy execution.
//
// Thread-safe: all mutable session state is synchronized via internal |mu_|.
class ReshardReceiveSession {
 public:
  // Creates and initializes a consumer pool-reshard receive session for |plan|,
  // acquiring bounded pool staging leases per storage via |staging_allocator|.
  static absl::StatusOr<std::shared_ptr<ReshardReceiveSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* staging_allocator,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_blocks,
      std::chrono::steady_clock::time_point deadline);

  ~ReshardReceiveSession() { ReleaseStaging(); }

  const std::string& req_id() const { return req_id_; }
  uint64_t uuid() const { return uuid_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }

  bool draining() const {
    absl::MutexLock lock(mu_);
    return draining_;
  }
  bool failed() const {
    absl::MutexLock lock(mu_);
    return failed_;
  }
  bool done() const {
    absl::MutexLock lock(mu_);
    return done_;
  }

  // Decides the receive's outcome; marks it done and releases its pool staging
  // leases once nothing issued for it is still running.
  void FinishRecv(bool has_failed, bool unregister_on_settle = false);

  // Releases any held pool staging leases for |uuid_|. Safe to call multiple
  // times.
  void ReleaseStaging();

  // Atomically claims any pending settle-unregister request and writes its
  // plan generation (0 for pool-reshard plans) to |generation|.
  bool TakePendingUnregister(uint64_t* generation);

  // Returns true if network/pool transfer is complete and all H2D futures are
  // ready.
  bool IsReadyToComplete() const;

  // Returns true if this session still has pending network or H2D work.
  bool HasPendingWork() const;

  // Handles pool completion notifications for this pool-reshard receive
  // session.
  absl::Status OnPoolReceived(KVCacheManagerWithTransfer& manager,
                              size_t pool_idx);

 private:
  friend struct PoolReshardRecvTestPeer;

  ReshardReceiveSession(kv_cache::KVCacheManagerBase* base,
                        StagingBlockAllocator* staging_allocator, uint64_t uuid,
                        const ::tpu_sync::rpc::StartTransferRequest& plan,
                        absl::Span<const int64_t> chip_blocks,
                        std::chrono::steady_clock::time_point deadline);

  absl::Status AcquireStagingLeases(
      const ::tpu_sync::rpc::StartTransferRequest& plan);

  void ReleaseStagingLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FinishRecvLocked(bool has_failed, bool unregister_on_settle = false)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EndRecvOpLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Decrements the count of in-flight operations; marks the receive done and
  // releases its staging resources if it was draining and waiting for this op.
  void EndRecvOp();

  // Handles completion of |pool_idx|'s H2D upload and finalizes the
  // pool-reshard receive when all pools have completed or on error.
  void FinishPoolH2d(KVCacheManagerWithTransfer& manager, size_t pool_idx,
                     const absl::Status& status);

  bool AllH2dDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status RecordPoolReceivedLocked(size_t pool_idx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::vector<std::pair<size_t, std::vector<int64_t>>>
  CollectEligiblePoolH2dsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordPoolH2dResultLocked(size_t pool_idx, const absl::Status& status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ExecuteEligiblePoolH2ds(KVCacheManagerWithTransfer& manager);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  StagingBlockAllocator* const staging_allocator_ = nullptr;
  const uint64_t uuid_ = 0;
  const std::string req_id_;
  const std::chrono::steady_clock::time_point deadline_;
  const std::chrono::steady_clock::time_point start_time_;
  std::vector<int64_t> chip_block_ids_ ABSL_GUARDED_BY(mu_);
  bool staging_released_ ABSL_GUARDED_BY(mu_) = false;
  bool network_completed_ ABSL_GUARDED_BY(mu_) = false;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool failed_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  bool reshard_finalizing_ ABSL_GUARDED_BY(mu_) = false;
  bool unregister_on_settle_ ABSL_GUARDED_BY(mu_) = true;
  std::vector<raiden::PjRtCopyFuture> h2d_futures_ ABSL_GUARDED_BY(mu_);
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
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_RESHARD_RECEIVE_SESSION_H_

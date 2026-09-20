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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_RECEIVE_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_RECEIVE_SESSION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

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
class TransferReceiveSession {
 public:
  static absl::StatusOr<std::shared_ptr<TransferReceiveSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
      const std::string& req_id, const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids,
      const std::optional<std::vector<int64_t>>& local_host_block_ids,
      std::chrono::steady_clock::time_point deadline);

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

  ~TransferReceiveSession() { ReleaseStaging(); }

  // Decides a receive's outcome; marks it done and releases its staging
  // resources once nothing issued for it is still running.
  void FinishRecv(bool has_failed, bool unregister_on_settle = false);

  // Releases any held staging slot or dynamic host blocks. Safe to call
  // multiple times.
  void ReleaseStaging();

  // Marks the plan to be unregistered when the receive settles, returning true
  // if the receive is still active and will unregister on settle, or false if
  // it is already done.
  bool DeferUnregisterOnSettle();

  // Atomically claims any pending settle-unregister request and writes its
  // plan generation to |generation|.
  bool TakePendingUnregister(uint64_t* generation);

  // Returns true if network/layer transfer is complete and all H2D futures are
  // ready.
  bool IsReadyToComplete() const;

  // Returns true if this session still has pending network or H2D work.
  bool HasPendingWork() const;

  // Schedules the consumer pull handshake on |base_->push_pool()| and updates
  // session state upon completion or error.
  void ExecutePullRequest(KVCacheManagerWithTransfer& manager,
                          const std::string& remote_endpoint);

  // Handles block completion notifications for this receive session.
  absl::Status OnBlocksReceived(KVCacheManagerWithTransfer& manager,
                                const std::vector<int>& block_ids);

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
  std::chrono::steady_clock::time_point deadline() const {
    absl::MutexLock lock(mu_);
    return deadline_;
  }

 private:
  TransferReceiveSession(kv_cache::KVCacheManagerBase* base,
                         StagingBlockAllocator* staging_allocator,
                         uint64_t uuid = 0)
      : base_(base), staging_allocator_(staging_allocator), uuid_(uuid) {}
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

  // Initializes this receive session for a consumer StartRead load plan.
  void InitFromLoadPlan(const std::string& req_id, CopyPlan load_plan,
                        std::chrono::steady_clock::time_point deadline);
  using H2dIssueFuture =
      std::shared_future<absl::StatusOr<raiden::PjRtCopyFuture>>;

  void ReleaseStagingLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FinishRecvLocked(bool has_failed, bool unregister_on_settle = false)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EndRecvOpLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Decrements the count of in-flight operations; marks the receive done and
  // releases its staging resources if it was draining and waiting for this op.
  void EndRecvOp();

  bool AllH2dDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool RecordBlocksReceivedLocked(const std::vector<int>& block_ids,
                                  bool* first_packet,
                                  bool* network_just_completed)
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
  std::vector<H2dIssueFuture> h2d_dispatch_futures_ ABSL_GUARDED_BY(mu_);
  int32_t total_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t num_completed_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t num_completed_layers_ ABSL_GUARDED_BY(mu_) = 0;
  bool network_completed_ ABSL_GUARDED_BY(mu_) = false;
  bool h2d_started_ ABSL_GUARDED_BY(mu_) = false;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool failed_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<int> accumulated_host_block_ids_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point deadline_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point start_time_ ABSL_GUARDED_BY(mu_);
  std::vector<raiden::PjRtCopyFuture> h2d_futures_ ABSL_GUARDED_BY(mu_);
  // The plan is dropped when this receive settles: set for every
  // demand-staged receiver plan (whose mapping would otherwise outlive its
  // freed blocks) and when an unregister arrives while the receive is in
  // flight (the plan stays mapped until then so late pushes resolve
  // through its blocks).
  bool unregister_on_settle_ ABSL_GUARDED_BY(mu_) = false;
  // Generation of the plan this receive belongs to; settlement cleanup
  // only touches that registration.
  uint64_t plan_generation_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_RECEIVE_SESSION_H_

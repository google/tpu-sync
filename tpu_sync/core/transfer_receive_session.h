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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_send_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {

struct CopyPlan {
  int64_t num_blocks = 0;
  std::vector<int64_t> requested_remote_block_ids;
  std::vector<int64_t> requested_local_block_ids;
  std::vector<int64_t> producer_remote_block_ids;
  std::vector<int64_t> h2d_local_block_ids;
  std::vector<int64_t> h2d_host_block_ids;
  std::vector<int64_t> transport_host_block_ids;
  std::vector<size_t> host_dst_to_src;
  CopySpec d2h_copy;
  CopySpec h2d_copy;

  bool RequiresHostReorder() const { return !host_dst_to_src.empty(); }
};

struct PendingCopy {
  int64_t host_block_id;
  int64_t chip_block_id;
};

class KVCacheManagerWithTransfer;

// Encapsulates the per-transfer state and execution lifecycle of a consumer
// receive operation (both legacy pull/push H2D receives and multi-tag
// pool-reshard receives), including host staging ownership, reference-counted
// drain and sticky-failure tracking, block/layer/pool readiness accounting,
// and order-ranked H2D copy execution.
class TransferReceiveSession
    : public std::enable_shared_from_this<TransferReceiveSession> {
 public:
  explicit TransferReceiveSession(kv_cache::KVCacheManagerBase* base_in,
                                  uint64_t uuid_in = 0)
      : base_(base_in), uuid_(uuid_in) {}

  // Initializes this receive session for an HBM destination active plan.
  void InitFromActivePlan(
      const ::tpu_sync::rpc::StartTransferRequest& request,
      const absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>&
          host_block_of,
      std::vector<int> staged_blocks, uint64_t generation,
      std::chrono::steady_clock::time_point deadline_in,
      const CopySpec& coalesced_h2d_copy);

  // Initializes this receive session for a multi-tag pool-reshard plan.
  void InitFromPoolReshardPlan(
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_blocks,
      std::chrono::steady_clock::time_point deadline_in);

  // Initializes this receive session for a consumer StartRead load plan.
  void InitFromLoadPlan(const std::string& req_id_in, const CopyPlan& load_plan,
                        std::chrono::steady_clock::time_point deadline_in);

  // Increments the count of in-flight handshake or H2D operations.
  void BeginOpLocked() { ++in_flight_; }

  // Decrements the count of in-flight operations. Returns true when the session
  // is draining and the last in-flight operation has just finished.
  bool EndOpLocked();

  // Transitions a legacy receive session into draining mode and latches
  // |has_failed|. Returns true if the session should be retired immediately.
  bool FinishLocked(bool has_failed);

  // Returns true if network/layer transfer is complete and all H2D futures are
  // ready.
  bool IsReadyToComplete() const;

  // Returns true if this session still has pending network or H2D work.
  bool HasPendingWork() const;

  // Schedules the consumer pull handshake on |base_->push_pool()| and updates
  // |manager| upon completion or error.
  void ExecutePullRequest(KVCacheManagerWithTransfer& manager,
                          const std::string& remote_endpoint,
                          CopyPlan load_plan);

  // Handles block completion notifications for this receive session.
  absl::Status OnBlocksReceived(KVCacheManagerWithTransfer& manager,
                                const std::vector<int>& block_ids);

  // Issues H2D copy for |layer_idx| using |base_| and registers the completion
  // callback to update |manager|.
  absl::Status ExecuteLayerH2d(KVCacheManagerWithTransfer& manager,
                               size_t layer_idx);

  // Handles pool completion notifications for a pool-reshard receive session.
  absl::Status OnPoolReceived(KVCacheManagerWithTransfer& manager,
                              size_t pool_idx);

  // Handles completion of |pool_idx|'s H2D upload and finalizes the
  // pool-reshard receive when all pools have completed or on error.
  void FinishPoolH2d(KVCacheManagerWithTransfer& manager, size_t pool_idx,
                     const absl::Status& status);

  uint64_t uuid() const { return uuid_; }
  const std::string& req_id() const { return req_id_; }
  void set_req_id(std::string req_id) { req_id_ = std::move(req_id); }
  int64_t slot_idx() const { return slot_idx_; }
  void set_slot_idx(int64_t slot_idx) { slot_idx_ = slot_idx; }
  std::vector<int>* mutable_staged_host_blocks() {
    return &staged_host_blocks_;
  }
  void set_staged_host_blocks(std::vector<int> blocks) {
    staged_host_blocks_ = std::move(blocks);
  }
  std::vector<int> TakeStagedHostBlocks() {
    std::vector<int> out = std::move(staged_host_blocks_);
    staged_host_blocks_.clear();
    return out;
  }
  int32_t total_blocks() const { return total_blocks_; }
  void set_total_blocks(int32_t total_blocks) { total_blocks_ = total_blocks; }
  void set_num_completed_blocks(int32_t count) {
    num_completed_blocks_ = count;
  }
  int in_flight() const { return in_flight_; }
  bool draining() const { return draining_; }
  bool failed() const { return failed_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }
  void set_deadline(std::chrono::steady_clock::time_point deadline) {
    deadline_ = deadline;
  }
  void set_start_time(std::chrono::steady_clock::time_point start_time) {
    start_time_ = start_time;
  }
  bool is_pool_reshard() const { return is_pool_reshard_; }
  bool unregister_on_settle() const { return unregister_on_settle_; }
  void set_unregister_on_settle(bool val) { unregister_on_settle_ = val; }
  uint64_t plan_generation() const { return plan_generation_; }

 private:
  using H2dIssueFuture =
      std::shared_future<absl::StatusOr<raiden::PjRtCopyFuture>>;

  bool AllH2dDone() const;
  bool RecordBlocksReceivedLocked(const std::vector<int>& block_ids,
                                  bool* first_packet,
                                  bool* network_just_completed);
  absl::Status RecordPoolReceivedLocked(size_t pool_idx);
  std::vector<std::pair<size_t, std::vector<int64_t>>>
  CollectEligiblePoolH2dsLocked();
  bool RecordPoolH2dResultLocked(size_t pool_idx, const absl::Status& status);
  void ExecuteEligiblePoolH2ds(KVCacheManagerWithTransfer& manager);
  kv_cache::KVCacheManagerBase* base_ = nullptr;
  uint64_t uuid_ = 0;
  std::string req_id_;
  int64_t slot_idx_ = -1;  // host staging slot to release on completion
  // Host blocks held under demand staging, released on completion. Empty
  // when the transfer holds a fixed slot instead.
  std::vector<int> staged_host_blocks_;
  CopySpec h2d_copy_;
  std::vector<int64_t> chip_block_ids_;
  absl::flat_hash_map<kv_cache::HostBlockId, kv_cache::DeviceBlockId>
      host_to_chip_;
  std::map<std::pair<size_t, size_t>, std::vector<PendingCopy>>
      pending_h2d_copies_;
  std::vector<H2dIssueFuture> h2d_dispatch_futures_;
  int32_t total_blocks_ = 0;
  int32_t num_completed_blocks_ = 0;
  int32_t num_completed_layers_ = 0;
  bool network_completed_ = false;
  bool h2d_started_ = false;
  int in_flight_ = 0;
  bool draining_ = false;
  bool failed_ = false;
  bool reshard_finalizing_ = false;
  std::vector<int> accumulated_host_block_ids_;
  std::chrono::steady_clock::time_point deadline_;
  std::chrono::steady_clock::time_point start_time_;
  std::vector<raiden::PjRtCopyFuture> h2d_futures_;
  bool is_pool_reshard_ = false;
  // The plan is dropped when this receive settles: set for every
  // demand-staged receiver plan (whose mapping would otherwise outlive its
  // freed blocks) and when an unregister arrives while the receive is in
  // flight (the plan stays mapped until then so late pushes resolve
  // through its blocks).
  bool unregister_on_settle_ = false;
  // Generation of the plan this receive belongs to; settlement cleanup
  // only touches that registration.
  uint64_t plan_generation_ = 0;
  std::set<size_t> expected_pool_indices_;
  std::set<size_t> started_pool_indices_;
  std::set<size_t> completed_pool_indices_;
  // Multi-tag plans: per-pool H2D upload ordering. A pool's mirror is
  // uploaded only after every expected pool of a strictly lower order
  // rank has completed its upload (FA at rank 0, state classes at rank
  // 1, so state bytes land last on aliased arena pages). Single-tag
  // plans leave every rank 0 (upload immediately on wire completion).
  std::map<size_t, int> pool_order_ranks_;
  std::set<size_t> h2d_launched_pools_;
  // Multi-tag plans: each pool uploads only its own group's destination
  // block ids (the flat chip_block_ids list concatenates all groups).
  std::map<size_t, std::vector<int64_t>> pool_dst_block_ids_;
};

using ReceiveSession = TransferReceiveSession;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_RECEIVE_SESSION_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/data_connection_policy.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/kv_cache/transfer_session.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

class KVCacheManagerWithTransfer;
class StagingBlockAllocator;

// Encapsulates the producer-side state and execution logic for a single
// push-driven pool-reshard send transfer (identified by |uuid|).
//
// Thread-safety: This class is thread-safe. Immutable fields (|base_|,
// |staging_allocator_|, |req_id_|, |uuid_|, |parallelism_|, |deadline_|,
// |plan_|, |dst_peers_|, |data_connection_reuse_|) are set at construction
// time and read without locking. Mutable state is protected by internal |mu_|.
class ReshardSendSession
    : public TransferSession,
      public std::enable_shared_from_this<ReshardSendSession> {
 public:
  // Validates the sender schedule in |plan|, resolves the data connection
  // policy (TPU_RAIDEN_DATA_CONN_REUSE), initializes the transport server,
  // and creates a producer pool-reshard send session that Start() pushes on
  // |manager|. |manager| is null only for sessions that are never started
  // (tests that drive the executor directly).
  static absl::StatusOr<std::shared_ptr<ReshardSendSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      KVCacheManagerWithTransfer* absl_nullable manager,
      StagingBlockAllocator* absl_nullable staging_allocator,
      absl::Span<const int64_t> src_block_ids, int parallelism,
      std::chrono::steady_clock::time_point deadline,
      ::tpu_sync::rpc::StartTransferRequest plan);

  // Releases any pool staging leases still held (a session dropped before it
  // settled); a settled session has already released them.
  ~ReshardSendSession() override;

  bool Done() const override {
    absl::MutexLock lock(mu_);
    return done_;
  }

  void Finish(const absl::Status& status = absl::OkStatus()) override;

  bool TryBeginOp() override { return false; }

  // Counts one op out. Under kPerTransfer, the last op of a finalizing send
  // closes the idle data connections to |dst_peers_| while it is still
  // counted, then settles the send (which also retires its plan).
  void EndOp() override;

  absl::Status GetStatus() const override {
    absl::MutexLock lock(mu_);
    return status_;
  }

  absl::Status AwaitForDone() override;

  bool IsDraining() const override {
    absl::MutexLock lock(mu_);
    return finalizing_;
  }

  const std::string& req_id() const { return req_id_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }
  // Position in the request's sub-transfer chain (both zero for a
  // single-plan request); see StartTransferRequest.sub_plan_index.
  int32_t sub_plan_index() const { return plan_.sub_plan_index(); }
  int32_t sub_plan_count() const { return plan_.sub_plan_count(); }

  // Leases host staging per pool, issues D2H copies for each pool in |plan_|
  // from the source blocks given to Create(), and chains async H2H peer
  // pushes upon D2H completion. Non-blocking preconditions (manager,
  // transport, host staging layers, pool indices) are checked synchronously
  // and returned; the staging waits, copies and pushes run on a pipeline
  // thread and report through GetStatus() / Done().
  absl::Status Start() override;

  // Returns the buffer the device-to-host copy leased to |pool_idx|.
  absl::StatusOr<size_t> ResolveStagingSlot(size_t pool_idx) override {
    return LookupStagingSlot(pool_idx);
  }

  // Always returns kFailedPrecondition: a send session never receives data.
  absl::Status OnDataReceived(size_t major_dimension_idx,
                              absl::Span<const int> minor_dimension_indices,
                              absl::Span<const int> shard_ids) override;

  // Pool-reshard sends push each pool themselves once its device-to-host copy
  // lands, so no transport read gates on them: the future is already OK.
  tsl::Future<> GetDataReadyFuture(size_t major_dimension_idx) override {
    return tsl::Future<>(absl::OkStatus());
  }

  absl::optional<std::shared_ptr<const TransferPlan>> plan() const override {
    if (transfer_plan_ == nullptr || Done()) {
      return absl::nullopt;
    }
    return transfer_plan_;
  }

 private:
  friend struct ReshardSendSessionTestPeer;

  // Returns the host staging buffer leased to |pool_idx|, leasing one when it
  // has none and waiting until deadline when every slot is leased.
  absl::StatusOr<size_t> AcquireStagingSlot(size_t pool_idx);

  // Returns the host staging buffer currently leased to |pool_idx|. Fails
  // with the session's status (or kCancelled) once the session is draining
  // and holds no lease for |pool_idx|, and kFailedPrecondition when a live
  // session holds none.
  absl::StatusOr<size_t> LookupStagingSlot(size_t pool_idx) const;

  // Checks the preconditions of Start that need no waiting: a running
  // transport server, at least one host staging layer, and every transfer
  // pool index resolving on |base_|.
  absl::Status ValidatePushPreconditions() const;

  void ExecutePushPipeline(std::vector<int64_t> src_block_ids);

  ReshardSendSession(kv_cache::KVCacheManagerBase* base,
                     KVCacheManagerWithTransfer* manager,
                     StagingBlockAllocator* staging_allocator,
                     std::vector<int64_t> src_block_ids, std::string req_id,
                     uint64_t uuid, int parallelism,
                     int remaining_pool_peer_pushes,
                     std::chrono::steady_clock::time_point deadline,
                     TransferPlan transfer_plan,
                     ::tpu_sync::rpc::StartTransferRequest plan,
                     std::vector<std::string> dst_peers,
                     DataConnectionReuse data_connection_reuse);

  // Validates |plan| against |base| and |src_block_ids|.
  static absl::Status ValidatePlan(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> src_block_ids);

  // Records completion or failure of one (pool, peer) push operation and
  // releases staging resources once all scheduled pushes settle.
  void RecordPushCompletion(const absl::Status& status);
  void StartPoolPush(size_t pool_idx);
  void SettleLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseLeasesLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  // The manager this send executes on; null for a session never started.
  KVCacheManagerWithTransfer* const manager_ = nullptr;
  StagingBlockAllocator* const staging_allocator_ = nullptr;
  // Source device blocks Start() copies from.
  const std::vector<int64_t> src_block_ids_;
  const std::string req_id_;
  const uint64_t uuid_ = 0;
  const int parallelism_ = 8;
  const std::chrono::steady_clock::time_point deadline_;
  const ::tpu_sync::rpc::StartTransferRequest plan_;
  // Data endpoints the sender schedule pushes to.
  const std::vector<std::string> dst_peers_;
  // TPU_RAIDEN_DATA_CONN_REUSE as resolved when the session was created.
  const DataConnectionReuse data_connection_reuse_;
  const std::shared_ptr<const TransferPlan> transfer_plan_;

  mutable absl::Mutex mu_;
  int remaining_pool_peer_pushes_ ABSL_GUARDED_BY(mu_) = 0;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  bool finalizing_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<raiden::PjRtCopyFuture> d2h_futures_ ABSL_GUARDED_BY(mu_);
  bool leases_released_ ABSL_GUARDED_BY(mu_) = false;
  // The last op already closed the idle data connections (kPerTransfer).
  bool connections_released_ ABSL_GUARDED_BY(mu_) = false;
  StagingLayerManager staging_layer_manager_;
  std::atomic<bool> cancelled_{false};
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_

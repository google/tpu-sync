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
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/data_connection_policy.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
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
  // registers the active plan on |base|, and creates a producer pool-reshard
  // send session.
  static absl::StatusOr<std::shared_ptr<ReshardSendSession>> Create(
      kv_cache::KVCacheManagerBase* base,
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
  uint64_t uuid() const { return uuid_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }
  // Position in the request's sub-transfer chain (both zero for a
  // single-plan request); see StartTransferRequest.sub_plan_index.
  int32_t sub_plan_index() const { return plan_.sub_plan_index(); }
  int32_t sub_plan_count() const { return plan_.sub_plan_count(); }

  // Leases host staging per pool, issues D2H copies for each pool in |plan_|,
  // and chains async H2H peer pushes upon D2H completion.
  absl::Status ExecutePush(KVCacheManagerWithTransfer& manager,
                           absl::Span<const int64_t> src_block_ids);

 private:
  friend struct ReshardSendSessionTestPeer;

  ReshardSendSession(kv_cache::KVCacheManagerBase* base,
                     StagingBlockAllocator* staging_allocator,
                     std::string req_id, uint64_t uuid, int parallelism,
                     int remaining_pool_peer_pushes,
                     std::chrono::steady_clock::time_point deadline,
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
  void RecordPushCompletion(KVCacheManagerWithTransfer& manager,
                            const absl::Status& status);
  void StartPoolPush(KVCacheManagerWithTransfer& manager, size_t pool_idx);
  // Counts one op out. The last op of a finalizing send first drops the plan
  // and, under kPerTransfer, closes the idle data connections to |dst_peers_|
  // while it is still counted, then settles the send.
  void EndOp();
  void SettleLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseLeasesLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  StagingBlockAllocator* const staging_allocator_ = nullptr;
  const std::string req_id_;
  const uint64_t uuid_ = 0;
  const int parallelism_ = 8;
  const std::chrono::steady_clock::time_point deadline_;
  const ::tpu_sync::rpc::StartTransferRequest plan_;
  // Data endpoints the sender schedule pushes to.
  const std::vector<std::string> dst_peers_;
  // TPU_RAIDEN_DATA_CONN_REUSE as resolved when the session was created.
  const DataConnectionReuse data_connection_reuse_;

  mutable absl::Mutex mu_;
  int remaining_pool_peer_pushes_ ABSL_GUARDED_BY(mu_) = 0;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  bool finalizing_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<raiden::PjRtCopyFuture> d2h_futures_ ABSL_GUARDED_BY(mu_);
  // The manager this send executes on, set by ExecutePush; a send finished
  // from outside (deadline, cancellation) drops its plan on it once its
  // last push ends.
  KVCacheManagerWithTransfer* manager_ ABSL_GUARDED_BY(mu_) = nullptr;
  bool leases_released_ ABSL_GUARDED_BY(mu_) = false;
  bool plan_unregistered_ ABSL_GUARDED_BY(mu_) = false;
  // The last op already closed the idle data connections (kPerTransfer).
  bool connections_released_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_

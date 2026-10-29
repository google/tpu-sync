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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SEND_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SEND_SESSION_H_

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/data_connection_policy.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/transfer_session.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

// Encapsulates the per-transfer state and execution lifecycle of a producer
// pull-serve send operation, including block registration/validation,
// reference-counted drain and sticky-failure tracking, block readiness
// callbacks, and pipelined multi-layer D2H -> H2H execution.
//
// Thread-safe: all mutable session state is synchronized via internal |mu_|.
// Lock ordering: when both |KVCacheManagerWithTransfer::mu_| and |mu_| are
// acquired, |KVCacheManagerWithTransfer::mu_| must be acquired first. |mu_| is
// a leaf lock and is never held across callbacks or manager calls.
class TransferSendSession
    : public TransferSession,
      public std::enable_shared_from_this<TransferSendSession> {
 public:
  // Describes a block ID appearing more than once, with its first 2 indices.
  struct DuplicateBlock {
    int64_t block_id;
    size_t first_index, duplicate_index;
  };
  // Returns the first duplicate block ID and its first two indices, or nullopt.
  static std::optional<DuplicateBlock> FindDuplicateBlock(
      absl::Span<const int64_t> block_ids);

  // Resolves the data connection policy (TPU_RAIDEN_DATA_CONN_REUSE) and
  // creates a send session for |block_ids|.
  static absl::StatusOr<std::shared_ptr<TransferSendSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator,
      std::string req_id, uint64_t uuid, absl::Span<const int64_t> block_ids,
      std::chrono::steady_clock::time_point deadline,
      std::chrono::steady_clock::time_point register_start,
      KVCacheManagerWithTransfer* absl_nullable manager = nullptr,
      uint64_t base_uuid = 0, int32_t sub_plan_index = 0);

  ~TransferSendSession() override { ReleaseStagingBlocks(); }

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

  // Returns summary metadata and failure diagnostics for this send session.
  // Only meaningful after `Done()` returns true.
  TransferDetails Details() const;

  bool IsDraining() const override {
    absl::MutexLock lock(mu_);
    return draining_;
  }

  bool HasStaging() const {
    absl::MutexLock lock(mu_);
    return !staging_.empty();
  }

  // Validates deadline, requested blocks, and single-pull invariant, then marks
  // |pull_started_| true. Throws std::runtime_error or std::invalid_argument on
  // violation.
  void ValidateAndBeginPull(const std::vector<int64_t>& requested_block_ids,
                            std::chrono::steady_clock::time_point now);

  // Returns true if |block_id| is registered in this session and StartPush has
  // created |layer_idx|'s D2H future (its copy may not have been issued yet).
  bool OwnsBlockWithReadyFuture(int block_id, size_t layer_idx) const;

  // Registers |cb| to be invoked with |layer_idx|'s D2H copy status once the
  // copy completes, or with the send's failure if the copy is never issued.
  // Invokes |cb(OkStatus())| immediately if the session holds no future for
  // |layer_idx| (StartPush has not run, or the send has settled).
  void RegisterLayerReadinessCallback(
      size_t layer_idx,
      transport::BlockTransportDelegate::HostBlockReadyCallback cb);

  // Counts one finished copy or push; marks the send done and releases its
  // staging resources if it was draining and waiting for this operation.
  // Under kPerTransfer that last operation first closes the idle data
  // connections to the consumer's endpoints.
  void EndSendOp();

  // Stages producer device blocks into host staging and executes the
  // multi-layer D2H copy and pipelined H2H push.
  void StartPush(const std::vector<std::string>& remote_data_endpoints,
                 const std::vector<int64_t>& src_block_ids,
                 const std::vector<int64_t>& dst_block_ids);

  // Coalesces contiguous (src_block_ids, dst_block_ids) block runs into
  // CopySpec segments.
  static CopySpec BuildCoalescedCopySpec(
      const std::vector<int64_t>& src_block_ids,
      const std::vector<int64_t>& dst_block_ids);

  // Returns the host layer buffer leased to |layer_idx|, leasing one when it
  // has none and blocking while every buffer is in use. There is one buffer
  // per layer on a full host mirror, so the lease is then layer_idx itself
  // and never waits. A timeout fails the send in the staging phase.
  absl::StatusOr<size_t> AcquireStagingSlot(size_t layer_idx);

  // Returns the host layer buffer currently leased to |layer_idx|, or
  // kFailedPrecondition if |layer_idx| holds no lease.
  absl::StatusOr<size_t> LookupStagingSlot(size_t layer_idx) const;

  const StagingLayerManager& staging_layer_manager() const {
    return staging_layer_manager_;
  }

  const std::string& req_id() const { return req_id_; }
  uint64_t uuid() const { return uuid_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }
  std::chrono::steady_clock::time_point register_start() const {
    return register_start_;
  }

  int64_t num_blocks() const {
    absl::MutexLock lock(mu_);
    return static_cast<int64_t>(pulled_blocks_);
  }
  int64_t total_bytes() const {
    absl::MutexLock lock(mu_);
    return total_bytes_;
  }
  std::chrono::steady_clock::time_point d2h_done() const {
    absl::MutexLock lock(mu_);
    return d2h_done_;
  }

  uint64_t base_uuid() const { return base_uuid_; }
  int32_t sub_plan_index() const { return sub_plan_index_; }
  size_t pulled_blocks() const {
    absl::MutexLock l(mu_);
    return pulled_blocks_;
  }
  void BeginOrQueueH2h();
  void StartGatedH2h();
  void DropH2hGateHold();
  void CancelFromManagerLocked(const absl::Status& status);

 private:
  TransferSendSession(kv_cache::KVCacheManagerBase* base,
                      StagingBlockAllocator* staging_allocator,
                      std::string req_id, uint64_t uuid,
                      std::chrono::steady_clock::time_point deadline,
                      std::chrono::steady_clock::time_point register_start,
                      DataConnectionReuse data_connection_reuse,
                      KVCacheManagerWithTransfer* absl_nullable manager,
                      uint64_t base_uuid, int32_t sub_plan_index);

  std::optional<DuplicateBlock> PopulateRegisteredBlocks(
      absl::Span<const int64_t> block_ids);
  void ValidateRequestedBlocksLocked(
      const std::vector<int64_t>& requested_block_ids) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseStagingBlocks();
  void ReleaseStagingBlocksLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Marks the session as draining with `status`. If `status` is not OK,
  // attributes the failure to `phase` if provided, or else to the earliest
  // step still running (see `CurrentPhaseLocked()`).
  void FinishLocked(const absl::Status& status = absl::OkStatus(),
                    std::optional<absl::string_view> phase = std::nullopt)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns the earliest step of this send still running, as named in
  // `TransferDetails::phase`.
  absl::string_view CurrentPhaseLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EndSendOpLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status AcquireStagingWithRetry(int64_t num_blocks);
  void SendNextLayer(size_t l);
  void BeginStageTraceLocked(std::optional<ScopedTrace>& trace,
                             absl::string_view name, uint32_t stage_tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  StagingBlockAllocator* const staging_allocator_ = nullptr;
  KVCacheManagerWithTransfer* const manager_ = nullptr;
  const std::string req_id_;
  const uint64_t uuid_ = 0;
  const uint64_t base_uuid_ = 0;
  const int32_t sub_plan_index_ = 0;
  const std::chrono::steady_clock::time_point deadline_;
  const std::chrono::steady_clock::time_point register_start_;
  // TPU_RAIDEN_DATA_CONN_REUSE as resolved when the session was created.
  const DataConnectionReuse data_connection_reuse_;
  StagingAllocation staging_ ABSL_GUARDED_BY(mu_);
  int64_t total_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_set<int64_t> registered_block_set_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point d2h_done_ ABSL_GUARDED_BY(mu_);
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  // The transfer phase in which the send failed; populated when `status_` is
  // set to a non-OK status.
  absl::string_view failed_phase_ ABSL_GUARDED_BY(mu_);
  bool pull_started_ ABSL_GUARDED_BY(mu_) = false;
  size_t pulled_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  bool has_h2h_gate_hold_ ABSL_GUARDED_BY(mu_) = false;
  // Copies and pushes issued for this send whose completion has not been
  // observed. The staging they read and write is held until it is zero.
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  // The outcome is decided (failed, or every push done) and the send
  // waits for its in-flight work before it is reported.
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  // The last op already closed the idle data connections (kPerTransfer).
  bool connections_released_ ABSL_GUARDED_BY(mu_) = false;
  std::chrono::steady_clock::time_point completed_at_ ABSL_GUARDED_BY(mu_);
  std::vector<raiden::PjRtCopyFuture> d2h_layer_futures_ ABSL_GUARDED_BY(mu_);
  // The consumer's data endpoints, populated once `StartPush` serves its pull.
  std::vector<std::string> remote_data_endpoints_ ABSL_GUARDED_BY(mu_);
  std::vector<int> src_ints_ ABSL_GUARDED_BY(mu_);
  std::vector<int> dst_ints_ ABSL_GUARDED_BY(mu_);
  std::atomic<size_t> remaining_d2h_layers_{0};
  std::atomic<size_t> remaining_h2h_layers_{0};
  std::atomic<bool> cancelled_{false};
  StagingLayerManager staging_layer_manager_;
  std::optional<ScopedTrace> d2h_stage_trace_ ABSL_GUARDED_BY(mu_);
  std::optional<ScopedTrace> h2h_gate_wait_trace_ ABSL_GUARDED_BY(mu_);
  std::optional<ScopedTrace> h2h_push_trace_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SEND_SESSION_H_

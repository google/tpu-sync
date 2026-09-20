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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SEND_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SEND_SESSION_H_

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
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
class TransferSendSession : public TransferSession {
 public:
  static absl::StatusOr<std::shared_ptr<TransferSendSession>> Create(
      kv_cache::KVCacheManagerBase* base,
      StagingBlockAllocator* absl_nullable staging_allocator,
      std::string req_id, uint64_t uuid, absl::Span<const int64_t> block_ids,
      std::chrono::steady_clock::time_point deadline,
      std::chrono::steady_clock::time_point register_start, int in_flight = 0,
      bool pull_started = false);

  ~TransferSendSession() override { ReleaseSlot(); }

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

  // Returns true if |block_id| is registered in this session and |layer_idx|
  // has an issued D2H future.
  bool OwnsBlockWithReadyFuture(int block_id, size_t layer_idx) const;

  // Registers |cb| to be invoked when |layer_idx|'s D2H copy completes, or
  // invokes |cb(OkStatus())| immediately if no copy is pending for |layer_idx|.
  void RegisterLayerReadinessCallback(
      size_t layer_idx,
      transport::BlockTransportDelegate::HostBlockReadyCallback cb);

  // Counts one finished copy or push; marks the send done and releases its
  // staging resources if it was draining and waiting for this operation.
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

  const std::string& req_id() const { return req_id_; }
  uint64_t uuid() const { return uuid_; }
  std::chrono::steady_clock::time_point deadline() const { return deadline_; }
  std::chrono::steady_clock::time_point register_start() const {
    return register_start_;
  }

  int64_t num_blocks() const {
    absl::MutexLock lock(mu_);
    return num_blocks_;
  }
  int64_t total_bytes() const {
    absl::MutexLock lock(mu_);
    return total_bytes_;
  }
  std::chrono::steady_clock::time_point d2h_done() const {
    absl::MutexLock lock(mu_);
    return d2h_done_;
  }

 private:
  TransferSendSession(kv_cache::KVCacheManagerBase* base,
                      StagingBlockAllocator* staging_allocator,
                      std::string req_id, uint64_t uuid,
                      std::chrono::steady_clock::time_point deadline,
                      std::chrono::steady_clock::time_point register_start,
                      int in_flight = 0, bool pull_started = false);

  // Populates |registered_block_ids_| and |registered_block_set_| from
  // |block_ids|. Returns the first duplicate block ID if any duplicate is
  // present, or std::nullopt when all IDs are unique.
  std::optional<int64_t> PopulateRegisteredBlocks(
      absl::Span<const int64_t> block_ids);
  void ValidateRequestedBlocksLocked(
      const std::vector<int64_t>& requested_block_ids) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseSlot();
  void ReleaseSlotLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FinishLocked(const absl::Status& status = absl::OkStatus())
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EndSendOpLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::StatusOr<StagingAllocation> AcquireStagingWithRetry(int64_t num_blocks);
  void SendNextLayer(size_t l);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  StagingBlockAllocator* const staging_allocator_ = nullptr;
  const std::string req_id_;
  const uint64_t uuid_ = 0;
  const std::chrono::steady_clock::time_point deadline_;
  const std::chrono::steady_clock::time_point register_start_;
  StagingAllocation staging_ ABSL_GUARDED_BY(mu_);
  int64_t num_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t registered_num_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t total_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<int64_t> registered_block_ids_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<int64_t> registered_block_set_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point d2h_done_ ABSL_GUARDED_BY(mu_);
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  bool pull_started_ ABSL_GUARDED_BY(mu_) = false;
  // Copies and pushes issued for this send whose completion has not been
  // observed. The staging they read and write is held until it is zero.
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  // The outcome is decided (failed, or every push done) and the send
  // waits for its in-flight work before it is reported.
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<raiden::PjRtCopyFuture> d2h_layer_futures_ ABSL_GUARDED_BY(mu_);
  std::vector<std::string> remote_data_endpoints_ ABSL_GUARDED_BY(mu_);
  std::vector<int> src_ints_ ABSL_GUARDED_BY(mu_);
  std::vector<int> dst_ints_ ABSL_GUARDED_BY(mu_);
  std::atomic<size_t> remaining_h2h_layers_{0};
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SEND_SESSION_H_

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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

struct CopySpec {
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;
};

class KVCacheManagerWithTransfer;

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
    : public std::enable_shared_from_this<TransferSendSession> {
 public:
  TransferSendSession(kv_cache::KVCacheManagerBase* base_in,
                      std::string req_id_in, uint64_t uuid_in,
                      std::chrono::steady_clock::time_point deadline_in,
                      std::chrono::steady_clock::time_point register_start_in);

  // Populates |registered_block_ids_| and |registered_block_set_| from
  // |block_ids|. Returns the first duplicate block ID if any duplicate is
  // present, or std::nullopt when all IDs are unique.
  std::optional<int64_t> PopulateRegisteredBlocks(
      absl::Span<const int64_t> block_ids);

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

  // Counts one copy or push issued for the send.
  void BeginSendOpLocked() {
    absl::MutexLock lock(mu_);
    ++in_flight_;
  }

  // Counts one finished copy or push; retires the send on |manager| if it was
  // draining and waiting for this operation.
  void EndSendOpLocked(KVCacheManagerWithTransfer& manager);

  // Acquires |manager|'s mutex and records one finished copy or push.
  void EndSendOp(KVCacheManagerWithTransfer& manager);

  // Decides a pull-serve send's outcome and retires it on |manager| once
  // nothing issued for it is still running.
  void FinishSendLocked(KVCacheManagerWithTransfer& manager, bool has_failed);

  // Executes the multi-layer D2H copy and pipelined H2H push using |base_|.
  void ExecutePush(KVCacheManagerWithTransfer& manager,
                   const CopySpec& d2h_copy,
                   const std::vector<std::string>& remote_endpoints,
                   const std::vector<int64_t>& host_block_ids,
                   const std::vector<int64_t>& dst_block_ids);

  const std::string& req_id() const { return req_id_; }
  uint64_t uuid() const { return uuid_; }

  int64_t slot_idx() const {
    absl::MutexLock lock(mu_);
    return slot_idx_;
  }
  void set_slot_idx(int64_t slot_idx) {
    absl::MutexLock lock(mu_);
    slot_idx_ = slot_idx;
  }
  std::vector<int> staged_host_blocks() const {
    absl::MutexLock lock(mu_);
    return staged_host_blocks_;
  }
  void set_staged_host_blocks(std::vector<int> blocks) {
    absl::MutexLock lock(mu_);
    staged_host_blocks_ = std::move(blocks);
  }
  void ClearStagedHostBlocks() {
    absl::MutexLock lock(mu_);
    staged_host_blocks_.clear();
  }
  int64_t num_blocks() const {
    absl::MutexLock lock(mu_);
    return num_blocks_;
  }
  int64_t total_bytes() const {
    absl::MutexLock lock(mu_);
    return total_bytes_;
  }
  std::vector<int64_t> registered_block_ids() const {
    absl::MutexLock lock(mu_);
    return registered_block_ids_;
  }
  std::chrono::steady_clock::time_point register_start() const {
    absl::MutexLock lock(mu_);
    return register_start_;
  }
  std::chrono::steady_clock::time_point d2h_done() const {
    absl::MutexLock lock(mu_);
    return d2h_done_;
  }
  bool failed() const {
    absl::MutexLock lock(mu_);
    return failed_;
  }
  void set_pull_started(bool pull_started) {
    absl::MutexLock lock(mu_);
    pull_started_ = pull_started;
  }
  bool slot_released() const {
    absl::MutexLock lock(mu_);
    return slot_released_;
  }
  void set_slot_released(bool released) {
    absl::MutexLock lock(mu_);
    slot_released_ = released;
  }
  void set_in_flight(int in_flight) {
    absl::MutexLock lock(mu_);
    in_flight_ = in_flight;
  }
  bool draining() const {
    absl::MutexLock lock(mu_);
    return draining_;
  }
  std::chrono::steady_clock::time_point deadline() const {
    absl::MutexLock lock(mu_);
    return deadline_;
  }
  void set_deadline(std::chrono::steady_clock::time_point deadline) {
    absl::MutexLock lock(mu_);
    deadline_ = deadline;
  }

 private:
  void ValidateRequestedBlocksLocked(
      const std::vector<int64_t>& requested_block_ids) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void RetireSendLocked(KVCacheManagerWithTransfer& manager);
  void SendNextLayer(KVCacheManagerWithTransfer& manager, size_t l);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  const std::string req_id_;
  const uint64_t uuid_ = 0;
  int64_t slot_idx_ ABSL_GUARDED_BY(mu_) = -1;
  // Host blocks held under demand staging, released on completion. Empty
  // when the transfer holds a fixed slot instead.
  std::vector<int> staged_host_blocks_ ABSL_GUARDED_BY(mu_);
  int64_t num_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t registered_num_blocks_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t total_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<int64_t> registered_block_ids_ ABSL_GUARDED_BY(mu_);
  std::set<int64_t> registered_block_set_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point register_start_ ABSL_GUARDED_BY(mu_);
  std::chrono::steady_clock::time_point d2h_done_ ABSL_GUARDED_BY(mu_);
  bool failed_ ABSL_GUARDED_BY(mu_) = false;
  bool pull_started_ ABSL_GUARDED_BY(mu_) = false;
  bool slot_released_ ABSL_GUARDED_BY(mu_) = false;
  // Copies and pushes issued for this send whose completion has not been
  // observed. The staging they read and write is held until it is zero.
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  // The outcome is decided (failed, or every push done) and the send
  // waits for its in-flight work before it is reported.
  bool draining_ ABSL_GUARDED_BY(mu_) = false;
  std::chrono::steady_clock::time_point deadline_ ABSL_GUARDED_BY(mu_);
  std::vector<raiden::PjRtCopyFuture> d2h_layer_futures_ ABSL_GUARDED_BY(mu_);
  std::vector<std::string> remote_data_endpoints_ ABSL_GUARDED_BY(mu_);
  std::vector<int> src_ints_ ABSL_GUARDED_BY(mu_);
  std::vector<int> dst_ints_ ABSL_GUARDED_BY(mu_);
  std::atomic<size_t> remaining_h2h_layers_{0};
};

using SendSession = TransferSendSession;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SEND_SESSION_H_

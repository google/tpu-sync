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

#include "tpu_sync/core/reshard_receive_session.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ratio>  // NOLINT(build/c++11)
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {

namespace {

double DurationMs(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void RecordTransferDuration(double duration_ms) {
  telemetry::RaidenMetricStore::GetGlobalMetricStore().ObserveHistogram(
      telemetry::metric_names::kTransferDurationMs, {}, duration_ms);
}

}  // namespace

absl::StatusOr<std::shared_ptr<ReshardReceiveSession>>
ReshardReceiveSession::Create(kv_cache::KVCacheManagerBase* base,
                              StagingBlockAllocator* staging_allocator,
                              const ::tpu_sync::rpc::StartTransferRequest& plan,
                              absl::Span<const int64_t> chip_blocks,
                              std::chrono::steady_clock::time_point deadline) {
  auto session =
      std::shared_ptr<ReshardReceiveSession>(new ReshardReceiveSession(
          base, staging_allocator, plan.uuid(), plan, chip_blocks, deadline));
  TF_RETURN_IF_ERROR(session->AcquireStagingLeases(plan));
  return session;
}

ReshardReceiveSession::ReshardReceiveSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, uint64_t uuid,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_blocks,
    std::chrono::steady_clock::time_point deadline)
    : base_(base),
      staging_allocator_(staging_allocator),
      uuid_(uuid),
      req_id_(plan.req_id()),
      deadline_(deadline),
      start_time_(std::chrono::steady_clock::now()) {
  absl::MutexLock lock(mu_);
  chip_block_ids_.assign(chip_blocks.begin(), chip_blocks.end());
  for (int32_t pool_idx : plan.transfer_pool_indices()) {
    expected_pool_indices_.insert(static_cast<size_t>(pool_idx));
    pool_order_ranks_[static_cast<size_t>(pool_idx)] = 0;
  }
  for (const auto& group : plan.pool_groups()) {
    std::vector<int64_t> group_dst_ids(group.dst_device_block_ids().begin(),
                                       group.dst_device_block_ids().end());
    for (int32_t pool_idx : group.pool_indices()) {
      pool_order_ranks_[static_cast<size_t>(pool_idx)] = group.order_rank();
      pool_dst_block_ids_[static_cast<size_t>(pool_idx)] = group_dst_ids;
    }
  }
}

absl::Status ReshardReceiveSession::AcquireStagingLeases(
    const ::tpu_sync::rpc::StartTransferRequest& plan) {
  // Bounded host staging: the wire still lands at device (chip) block ids,
  // but on a bounded storage those ids are remapped to arena slots leased to
  // this uuid. Lease the union of every pool's destination ids per storage
  // before arming; a failure here refuses the arm cleanly (the coordinator
  // abandons the claim and no sender is dispatched). Full-mirror storages are
  // no-ops. Released in FinishPoolH2d / the deadline sweep /
  // ~ReshardReceiveSession.
  std::map<size_t, std::set<int64_t>> dst_ids_by_storage;
  {
    absl::MutexLock lock(mu_);
    for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
      const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
      const kv_cache::PoolSpec* pool_spec = base_->pool(pool_idx);
      if (pool_spec == nullptr ||
          !base_->PoolStorageStagingBounded(pool_spec->storage_index)) {
        continue;
      }
      auto ids_it = pool_dst_block_ids_.find(pool_idx);
      const std::vector<int64_t>& ids = ids_it == pool_dst_block_ids_.end()
                                            ? chip_block_ids_
                                            : ids_it->second;
      dst_ids_by_storage[pool_spec->storage_index].insert(ids.begin(),
                                                          ids.end());
    }
  }
  for (const auto& [storage_idx, ids] : dst_ids_by_storage) {
    std::vector<int64_t> id_list(ids.begin(), ids.end());
    absl::Status lease_status = staging_allocator_->AcquirePoolStagingLease(
        uuid_, storage_idx, id_list);
    if (!lease_status.ok()) {
      return lease_status;
    }
  }
  return absl::OkStatus();
}

void ReshardReceiveSession::ReleaseStagingLocked() {
  if (staging_released_) return;
  staging_released_ = true;
  if (staging_allocator_ != nullptr) {
    staging_allocator_->ReleasePoolStagingLeases(uuid_);
  }
}

void ReshardReceiveSession::ReleaseStaging() {
  absl::MutexLock lock(mu_);
  ReleaseStagingLocked();
}

void ReshardReceiveSession::FinishRecvLocked(bool has_failed,
                                             bool unregister_on_settle) {
  if (unregister_on_settle) unregister_on_settle_ = true;
  if (has_failed) failed_ = true;
  if (draining_) return;
  draining_ = true;
  if (in_flight_ == 0 && !done_) {
    ReleaseStagingLocked();
    done_ = true;
  }
}

void ReshardReceiveSession::FinishRecv(bool has_failed,
                                       bool unregister_on_settle) {
  absl::MutexLock lock(mu_);
  FinishRecvLocked(has_failed, unregister_on_settle);
}

void ReshardReceiveSession::EndRecvOpLocked() {
  if (in_flight_ <= 0) {
    LOG(DFATAL) << "Receive operation count underflow for UUID " << uuid_;
    return;
  }
  --in_flight_;
  if (draining_ && in_flight_ == 0 && !done_) {
    ReleaseStagingLocked();
    done_ = true;
  }
}

void ReshardReceiveSession::EndRecvOp() {
  absl::MutexLock lock(mu_);
  EndRecvOpLocked();
}

bool ReshardReceiveSession::TakePendingUnregister(uint64_t* generation) {
  absl::MutexLock lock(mu_);
  if (!unregister_on_settle_) return false;
  unregister_on_settle_ = false;
  if (generation != nullptr) {
    *generation = 0;
  }
  return true;
}

bool ReshardReceiveSession::AllH2dDoneLocked() const {
  for (const auto& f : h2d_futures_) {
    if (!f.IsReady()) return false;
  }
  return true;
}

bool ReshardReceiveSession::IsReadyToComplete() const {
  absl::MutexLock lock(mu_);
  return network_completed_ && AllH2dDoneLocked();
}

bool ReshardReceiveSession::HasPendingWork() const {
  absl::MutexLock lock(mu_);
  if (done_) return false;
  if (!network_completed_) return true;
  return !AllH2dDoneLocked();
}

absl::Status ReshardReceiveSession::RecordPoolReceivedLocked(size_t pool_idx) {
  if (expected_pool_indices_.find(pool_idx) == expected_pool_indices_.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "received undeclared pool ", pool_idx, " for UUID ", uuid_));
  }
  if (started_pool_indices_.find(pool_idx) != started_pool_indices_.end()) {
    return absl::AlreadyExistsError(
        absl::StrCat("pool completed more than once: ", pool_idx));
  }
  started_pool_indices_.insert(pool_idx);
  return absl::OkStatus();
}

std::vector<std::pair<size_t, std::vector<int64_t>>>
ReshardReceiveSession::CollectEligiblePoolH2dsLocked() {
  std::vector<std::pair<size_t, std::vector<int64_t>>> to_launch;
  if (reshard_finalizing_) return to_launch;
  for (size_t pool_idx : started_pool_indices_) {
    if (h2d_launched_pools_.count(pool_idx)) continue;
    const auto rank_it = pool_order_ranks_.find(pool_idx);
    const int rank = rank_it == pool_order_ranks_.end() ? 0 : rank_it->second;
    bool prerequisites_uploaded = true;
    for (size_t other : expected_pool_indices_) {
      const auto other_it = pool_order_ranks_.find(other);
      const int other_rank =
          other_it == pool_order_ranks_.end() ? 0 : other_it->second;
      if (other_rank < rank && completed_pool_indices_.find(other) ==
                                   completed_pool_indices_.end()) {
        prerequisites_uploaded = false;
        break;
      }
    }
    if (!prerequisites_uploaded) continue;
    h2d_launched_pools_.insert(pool_idx);
    const auto ids_it = pool_dst_block_ids_.find(pool_idx);
    to_launch.emplace_back(pool_idx, ids_it == pool_dst_block_ids_.end()
                                         ? chip_block_ids_
                                         : ids_it->second);
  }
  std::sort(to_launch.begin(), to_launch.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  return to_launch;
}

bool ReshardReceiveSession::RecordPoolH2dResultLocked(
    size_t pool_idx, const absl::Status& status) {
  if (reshard_finalizing_) return false;
  if (!status.ok()) {
    reshard_finalizing_ = true;
    return true;
  }
  completed_pool_indices_.insert(pool_idx);
  if (completed_pool_indices_ == expected_pool_indices_) {
    reshard_finalizing_ = true;
    return true;
  }
  return false;
}

absl::Status ReshardReceiveSession::OnPoolReceived(
    KVCacheManagerWithTransfer& manager, size_t pool_idx) {
  {
    absl::MutexLock lock(mu_);
    if (done_) {
      return absl::NotFoundError(
          absl::StrCat("no active receiver for UUID ", uuid_));
    }
    TF_RETURN_IF_ERROR(RecordPoolReceivedLocked(pool_idx));
  }
  ExecuteEligiblePoolH2ds(manager);
  return absl::OkStatus();
}

void ReshardReceiveSession::ExecuteEligiblePoolH2ds(
    KVCacheManagerWithTransfer& manager) {
  std::vector<std::pair<size_t, std::vector<int64_t>>> to_launch;
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return;
    }
    to_launch = CollectEligiblePoolH2dsLocked();
    in_flight_ += static_cast<int32_t>(to_launch.size());
  }
  for (auto& [pool_idx, dst_chip_block_ids] : to_launch) {
    auto future_or = base_->H2dPoolBlocks(pool_idx, dst_chip_block_ids,
                                          /*shard_idx=*/std::nullopt, uuid_);
    if (!future_or.ok()) {
      FinishPoolH2d(manager, pool_idx, future_or.status());
      EndRecvOp();
      continue;
    }
    raiden::PjRtCopyFuture future = *std::move(future_or);
    {
      absl::MutexLock lock(mu_);
      h2d_futures_.push_back(future);
    }
    future.OnReady([this, &manager, pool_idx = pool_idx](auto status_or) {
      FinishPoolH2d(manager, pool_idx,
                    status_or.ok() ? absl::OkStatus() : status_or.status());
      EndRecvOp();
    });
  }
}

void ReshardReceiveSession::FinishPoolH2d(KVCacheManagerWithTransfer& manager,
                                          size_t pool_idx,
                                          const absl::Status& status) {
  bool finished = false;
  {
    absl::MutexLock lock(mu_);
    if (done_) {
      return;
    }
    finished = RecordPoolH2dResultLocked(pool_idx, status);
  }
  if (!finished && status.ok()) {
    ExecuteEligiblePoolH2ds(manager);
  }
  std::chrono::steady_clock::time_point session_start_time;
  bool should_record_duration = false;
  if (finished) {
    absl::Status unregister = manager.UnregisterActivePlan(uuid_);
    if (!unregister.ok() && !absl::IsNotFound(unregister)) {
      LOG(ERROR) << "Failed to unregister pool reshard receiver plan " << uuid_
                 << ": " << unregister;
    }
    bool has_failed = false;
    {
      absl::MutexLock lock(mu_);
      unregister_on_settle_ = false;
      if (!status.ok() || (!unregister.ok() && !absl::IsNotFound(unregister))) {
        has_failed = true;
      } else {
        session_start_time = start_time_;
        should_record_duration = true;
        network_completed_ = true;
      }
    }
    FinishRecv(has_failed);
  }
  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(session_start_time, std::chrono::steady_clock::now()));
  }
}

}  // namespace tpu_raiden

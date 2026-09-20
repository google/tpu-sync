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

#include "tpu_sync/core/reshard_send_session.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
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
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {

ReshardSendSession::ReshardSendSession(
    kv_cache::KVCacheManagerBase* base_in, std::string req_id_in,
    uint64_t uuid_in, int parallelism_in, int remaining_pool_peer_pushes_in,
    std::chrono::steady_clock::time_point deadline_in,
    ::tpu_sync::rpc::StartTransferRequest plan_in)
    : base_(base_in),
      req_id_(std::move(req_id_in)),
      uuid_(uuid_in),
      parallelism_(parallelism_in),
      deadline_(deadline_in),
      plan_(std::move(plan_in)),
      remaining_pool_peer_pushes_(remaining_pool_peer_pushes_in) {}

absl::Status ReshardSendSession::ExecutePush(
    KVCacheManagerWithTransfer& manager,
    absl::Span<const int64_t> src_block_ids) {
  {
    absl::MutexLock lock(mu_);
    ++in_flight_;
  }
  // Multi-tag plans scope each pool's staging and pushes to its group's
  // entries; the flat src_block_ids argument is the legacy single-tag
  // whole-plan block list.
  const auto pool_group_index = [this](size_t pool_idx) -> int {
    for (int group_idx = 0; group_idx < plan_.pool_groups_size(); ++group_idx) {
      const auto& indices = plan_.pool_groups(group_idx).pool_indices();
      if (std::find(indices.begin(), indices.end(),
                    static_cast<int32_t>(pool_idx)) != indices.end()) {
        return group_idx;
      }
    }
    return -1;
  };
  auto local_schedule_it = plan_.shard_push_schedules().find(0);
  if (local_schedule_it == plan_.shard_push_schedules().end() &&
      plan_.shard_push_schedules().size() == 1) {
    local_schedule_it = plan_.shard_push_schedules().begin();
  }

  for (int32_t encoded_pool_idx : plan_.transfer_pool_indices()) {
    const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
    std::vector<int64_t> pool_src_block_ids(src_block_ids.begin(),
                                            src_block_ids.end());
    if (local_schedule_it != plan_.shard_push_schedules().end()) {
      const int group_idx = pool_group_index(pool_idx);
      std::set<int64_t> group_src_ids;
      for (const auto& entry : local_schedule_it->second.entries()) {
        if (entry.pool_group() == group_idx) {
          group_src_ids.insert(static_cast<int64_t>(entry.src_block_id()));
        }
      }
      pool_src_block_ids.assign(group_src_ids.begin(), group_src_ids.end());
      if (pool_src_block_ids.empty()) {
        // This sender owns none of the group's bytes (e.g. a PCP rank whose
        // interleave slices all fall past a short prefix, or a state group
        // this sender routes to no destination): the pool produces no push
        // and CountPoolReshardSendSlots counted no slot for it. The
        // receiver's expected pushes count only senders with scheduled pairs.
        continue;
      }
    }
    // Bounded host staging: lease one arena slot per source page of this
    // pool's storage for the transfer before staging its bytes (no-op on
    // full-mirror storages). Released in SettleLocked.
    if (const kv_cache::PoolSpec* pool_spec = base_->pool(pool_idx);
        pool_spec != nullptr) {
      absl::Status lease_status = base_->AcquirePoolStagingLease(
          uuid_, pool_spec->storage_index, pool_src_block_ids,
          base_->pool_staging_lease_timeout());
      if (!lease_status.ok()) {
        Finish(manager, lease_status);
        EndOp();
        return lease_status;
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> future = base_->D2hPoolBlocks(
        pool_idx, pool_src_block_ids, /*shard_idx=*/std::nullopt, uuid_);
    if (!future.ok()) {
      Finish(manager, future.status());
      EndOp();
      return future.status();
    }
    raiden::PjRtCopyFuture pool_future = *std::move(future);
    {
      absl::MutexLock lock(mu_);
      d2h_futures_.push_back(pool_future);
      ++in_flight_;
    }
    pool_future.OnReady([this, &manager, pool_idx](auto status_or) {
      if (!status_or.ok()) {
        Finish(manager, status_or.status());
        EndOp();
        return;
      }
      StartPoolPush(manager, pool_idx);
      EndOp();
    });
  }
  EndOp();
  return absl::OkStatus();
}

void ReshardSendSession::StartPoolPush(KVCacheManagerWithTransfer& manager,
                                       size_t pool_idx) {
  RAIDEN_TRACE_FN("KVTransfer::StartPoolReshardPush", [&]() {
    return absl::StrCat("uuid=", uuid_, " pool=", pool_idx);
  });
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      return;
    }
  }

  auto schedule_it = plan_.shard_push_schedules().find(0);
  if (schedule_it == plan_.shard_push_schedules().end()) {
    schedule_it = plan_.shard_push_schedules().begin();
  }
  // A pool pushes only its own group's (src, dst) pairs.
  int pool_group_idx = -1;
  for (int group_idx = 0; group_idx < plan_.pool_groups_size(); ++group_idx) {
    const auto& indices = plan_.pool_groups(group_idx).pool_indices();
    if (std::find(indices.begin(), indices.end(),
                  static_cast<int32_t>(pool_idx)) != indices.end()) {
      pool_group_idx = group_idx;
      break;
    }
  }
  std::map<std::string, std::vector<std::pair<int, int>>> transfers_by_peer;
  std::map<std::string, std::set<std::pair<int, int>>> seen_by_peer;
  for (const auto& entry : schedule_it->second.entries()) {
    if (entry.pool_group() != pool_group_idx) {
      continue;
    }
    const std::pair<int, int> pair{static_cast<int>(entry.src_block_id()),
                                   static_cast<int>(entry.dst_block_id())};
    if (seen_by_peer[entry.dst_peer()].insert(pair).second) {
      transfers_by_peer[entry.dst_peer()].push_back(pair);
    }
  }

  transport::BlockTransport* transport_srv = base_->transport_server();
  if (transport_srv == nullptr) {
    Finish(manager,
           absl::FailedPreconditionError("transport server is not running"));
    return;
  }

  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      return;
    }
    in_flight_ += static_cast<int>(transfers_by_peer.size());
  }

  for (const auto& [peer, transfers] : transfers_by_peer) {
    std::vector<int> src_ids;
    std::vector<int> dst_ids;
    src_ids.reserve(transfers.size());
    dst_ids.reserve(transfers.size());
    for (const auto& [src_id, dst_id] : transfers) {
      src_ids.push_back(src_id);
      dst_ids.push_back(dst_id);
    }
    transport_srv->AsyncPush(
        {peer}, src_ids, dst_ids, parallelism_,
        transport::MajorOrder::kLayerMajor, uuid_, static_cast<int>(pool_idx),
        [this, &manager](absl::StatusOr<std::vector<int>> result) {
          Finish(manager, result.ok() ? absl::OkStatus() : result.status());
          EndOp();
        });
  }
}

void ReshardSendSession::Finish(KVCacheManagerWithTransfer& manager,
                                const absl::Status& status) {
  RAIDEN_TRACE_FN("KVTransfer::FinishPoolReshardSend", [&]() {
    return absl::StrCat("uuid=", uuid_, " status=", status.code());
  });
  bool should_unregister = false;
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) return;
    if (!status.ok()) {
      LOG(ERROR) << "Pool reshard send failed uuid=" << uuid_
                 << " req_id=" << req_id_ << ": " << status;
      failed_ = true;
      finalizing_ = true;
      should_unregister = true;
    } else if (--remaining_pool_peer_pushes_ == 0) {
      finalizing_ = true;
      should_unregister = true;
    }
  }
  if (!should_unregister) {
    return;
  }
  // All callers of Finish hold an active in_flight_ count and invoke EndOp()
  // after Finish returns, so |this| remains live while UnregisterActivePlan
  // runs outside |mu_|.
  absl::Status unregister = manager.UnregisterActivePlan(uuid_);
  if (!unregister.ok() && !absl::IsNotFound(unregister)) {
    LOG(ERROR) << "Failed to unregister pool reshard sender plan " << uuid_
               << ": " << unregister;
    absl::MutexLock lock(mu_);
    failed_ = true;
  }
}

void ReshardSendSession::FinishTimeout() {
  absl::MutexLock lock(mu_);
  if (done_ || finalizing_) return;
  failed_ = true;
  finalizing_ = true;
  if (in_flight_ == 0) {
    SettleLocked();
  }
}

void ReshardSendSession::EndOp() {
  absl::MutexLock lock(mu_);
  --in_flight_;
  if (finalizing_ && in_flight_ == 0 && !done_) {
    SettleLocked();
  }
}

void ReshardSendSession::SettleLocked() {
  // Every push of every pool has completed (or the send failed): release the
  // host staging arena slots and mark done atomically under |mu_|.
  base_->ReleasePoolStagingLeases(uuid_);
  done_ = true;
}

}  // namespace tpu_raiden

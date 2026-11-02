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

#include "tpu_sync/kv_cache/reshard_receive_session.h"

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ratio>  // NOLINT(build/c++11)
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/utils.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
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

absl::Status ReshardReceiveSession::ValidatePlan(
    const kv_cache::KVCacheManagerBase& base,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_blocks) {
  TF_RETURN_IF_ERROR(ValidateCommonPoolReshardPlan(&base, plan, chip_blocks));

  size_t cursor = 0;
  for (const auto& group : plan.pool_groups()) {
    absl::flat_hash_set<int64_t> group_ids;
    for (int64_t block_id : group.dst_device_block_ids()) {
      if (!group_ids.insert(block_id).second) {
        return absl::InvalidArgumentError(
            "group destination block ids must be unique");
      }
      if (cursor >= chip_blocks.size() || chip_blocks[cursor] != block_id) {
        return absl::InvalidArgumentError(
            "pool group block ids must concatenate to the plan's "
            "local block ids");
      }
      ++cursor;
    }
  }
  if (cursor != chip_blocks.size()) {
    return absl::InvalidArgumentError(
        "pool group block ids must cover the plan's local block ids");
  }

  TF_RETURN_IF_ERROR(ValidatePoolBlockBounds(&base, plan, chip_blocks));

  absl::flat_hash_set<int64_t> local_ids(chip_blocks.begin(),
                                         chip_blocks.end());
  absl::flat_hash_set<int64_t> receiver_blocks_with_zero_start;
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      if (static_cast<size_t>(entry.dst_shard_idx()) >= base.num_shards()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "destination shard index ", entry.dst_shard_idx(),
            " is out of range: receiver has ", base.num_shards(), " shards"));
      }
      const int64_t local_id = entry.dst_block_id();
      if (local_ids.find(local_id) == local_ids.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("destination block id ", local_id,
                         " is absent from the local block-id list"));
      }
      const int64_t local_offset = entry.dst_offset_bytes();
      const int64_t local_stride = entry.dst_stride_bytes();
      if (entry.dst_offset_bytes() == 0) {
        receiver_blocks_with_zero_start.insert(entry.dst_block_id());
      }
      const int32_t group_idx = entry.pool_group();
      for (int32_t encoded_pool_idx :
           plan.pool_groups(group_idx).pool_indices()) {
        const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
        const kv_cache::PoolSpec* spec = base.pool(pool_idx);
        if (!StridedSpanFitsRegions(local_offset, local_stride,
                                    entry.size_bytes(), entry.count(),
                                    spec->block_stride_bytes, spec->regions)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "destination span exceeds declared pool ", pool_idx,
              " live regions in block ", local_id, ": offset=", local_offset,
              " stride=", local_stride, " size=", entry.size_bytes(), " count=",
              entry.count(), " block_stride_bytes=", spec->block_stride_bytes));
        }
      }
    }
  }
  for (int64_t block_id : local_ids) {
    if (receiver_blocks_with_zero_start.find(block_id) ==
        receiver_blocks_with_zero_start.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "destination block ", block_id,
          " has no transfer entry starting at offset 0; partial-page "
          "destination preservation is not implemented"));
    }
  }
  return ValidateReceiverCoverage(base, plan);
}

absl::Status ReshardReceiveSession::ValidateReceiverCoverage(
    const kv_cache::KVCacheManagerBase& base,
    const ::tpu_sync::rpc::StartTransferRequest& plan) {
  constexpr int64_t kMaxExpandedRepeats = 1 << 24;

  struct GroupView {
    std::vector<size_t> pool_indices;
    std::vector<int64_t> dst_ids;
    std::vector<int64_t> extents;
    int64_t expected_pushes = 0;
  };
  std::vector<GroupView> groups;
  absl::flat_hash_set<size_t> grouped_pools;
  for (const auto& group : plan.pool_groups()) {
    GroupView view;
    for (int32_t pool_idx : group.pool_indices()) {
      if (pool_idx < 0 ||
          !grouped_pools.insert(static_cast<size_t>(pool_idx)).second) {
        return absl::InvalidArgumentError(
            "group pool indices must be unique and non-negative");
      }
      view.pool_indices.push_back(static_cast<size_t>(pool_idx));
    }
    view.dst_ids.assign(group.dst_device_block_ids().begin(),
                        group.dst_device_block_ids().end());
    view.extents.assign(group.dst_expected_extent_bytes().begin(),
                        group.dst_expected_extent_bytes().end());
    view.expected_pushes = group.expected_pushes();
    groups.push_back(std::move(view));
  }
  if (grouped_pools.size() !=
      static_cast<size_t>(plan.transfer_pool_indices_size())) {
    return absl::InvalidArgumentError(
        "group pool indices do not partition the plan's transfer pools");
  }
  for (int32_t pool_idx : plan.transfer_pool_indices()) {
    if (!grouped_pools.contains(static_cast<size_t>(pool_idx))) {
      return absl::InvalidArgumentError(
          "group pool indices do not partition the plan's transfer pools");
    }
  }

  const int64_t parallelism = plan.parallelism();
  if (parallelism <= 0) {
    return absl::InvalidArgumentError(
        "receiver plans require positive parallelism for push accounting");
  }

  struct GroupState {
    std::vector<kv_cache::PoolLiveSegment> segments;
    int64_t live_bytes = 0;
    absl::flat_hash_map<int64_t, size_t> ordinals;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> coverage;
    absl::flat_hash_map<
        int32_t, absl::flat_hash_set<std::tuple<std::string, int64_t, int64_t>>>
        pairs_by_sender;
  };
  std::vector<GroupState> states(groups.size());
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const GroupView& view = groups[group_idx];
    GroupState& state = states[group_idx];
    if (view.pool_indices.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("group ", group_idx, " declares no pools"));
    }
    for (size_t pool_idx : view.pool_indices) {
      const kv_cache::PoolSpec* spec = base.pool(pool_idx);
      if (spec == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("group pool index out of range: ", pool_idx));
      }
      absl::StatusOr<std::vector<kv_cache::PoolLiveSegment>> segments =
          kv_cache::ExpandPoolLiveSegments(*spec);
      if (!segments.ok()) return segments.status();
      if (state.segments.empty()) {
        state.segments = *std::move(segments);
      } else if (state.segments != *segments) {
        return absl::InvalidArgumentError(absl::StrCat(
            "group ", group_idx, " pools must share one live-region map; pool ",
            pool_idx, " disagrees"));
      }
    }
    for (const kv_cache::PoolLiveSegment& segment : state.segments) {
      state.live_bytes += segment.size;
    }
    if (state.live_bytes <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("group ", group_idx, " has no live destination bytes"));
    }
    if (view.extents.empty()) {
      return absl::InvalidArgumentError(
          "receiver plans require dst_expected_extent_bytes");
    }
    if (view.extents.size() != view.dst_ids.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "group ", group_idx,
          " extents do not match its destination block count: got ",
          view.extents.size(), ", expected ", view.dst_ids.size()));
    }
    for (size_t ordinal = 0; ordinal < view.extents.size(); ++ordinal) {
      const int64_t extent = view.extents[ordinal];
      if (extent <= 0 || extent > state.live_bytes) {
        return absl::InvalidArgumentError(
            absl::StrCat("extent ", extent, " for destination block ordinal ",
                         ordinal, " of group ", group_idx, " is outside (0, ",
                         state.live_bytes, "]"));
      }
      if (ordinal != view.extents.size() - 1 && extent != state.live_bytes) {
        return absl::InvalidArgumentError(
            "extents must cover every destination block fully except the "
            "final one");
      }
    }
    for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal) {
      state.ordinals[view.dst_ids[ordinal]] = ordinal;
    }
    state.coverage.resize(view.dst_ids.size());
  }

  int64_t expanded_repeats = 0;
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      const size_t group_idx = static_cast<size_t>(entry.pool_group());
      if (group_idx >= states.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry declares an unknown pool group ", group_idx));
      }
      GroupState& state = states[group_idx];
      const auto ordinal_it = state.ordinals.find(entry.dst_block_id());
      if (ordinal_it == state.ordinals.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry targets destination block ", entry.dst_block_id(),
            " outside its group ", group_idx, " destination set"));
      }
      const int64_t extent = groups[group_idx].extents[ordinal_it->second];
      expanded_repeats += entry.count();
      if (expanded_repeats > kMaxExpandedRepeats) {
        return absl::InvalidArgumentError(
            "receiver plan exceeds the repeat expansion bound");
      }
      for (int64_t repeat = 0; repeat < entry.count(); ++repeat) {
        const int64_t physical =
            entry.dst_offset_bytes() + repeat * entry.dst_stride_bytes();
        absl::StatusOr<std::pair<int64_t, int64_t>> range =
            kv_cache::PhysicalLiveRangeToLogical(state.segments, physical,
                                                 entry.size_bytes());
        if (!range.ok()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "reshard entry for destination block ", entry.dst_block_id(),
              " crosses padding or lies outside declared live regions: ",
              range.status().message()));
        }
        if (range->second > extent) {
          return absl::InvalidArgumentError(absl::StrCat(
              "reshard entry exceeds destination block ", entry.dst_block_id(),
              " declared live tail: end=", range->second, " extent=", extent));
        }
        state.coverage[ordinal_it->second].push_back(*range);
      }
      state.pairs_by_sender[source_rank].insert(std::make_tuple(
          entry.dst_peer(), static_cast<int64_t>(entry.src_block_id()),
          static_cast<int64_t>(entry.dst_block_id())));
    }
  }

  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const GroupView& view = groups[group_idx];
    GroupState& state = states[group_idx];
    int64_t calculated_pushes = 0;
    for (const auto& [source_rank, pairs] : state.pairs_by_sender) {
      calculated_pushes +=
          std::min(parallelism, static_cast<int64_t>(pairs.size()));
    }
    if (calculated_pushes != view.expected_pushes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "expected pushes for group ", group_idx,
          " do not match the received schedules: declared=",
          view.expected_pushes, " recomputed=", calculated_pushes));
    }
    for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal) {
      std::vector<std::pair<int64_t, int64_t>>& intervals =
          state.coverage[ordinal];
      std::sort(intervals.begin(), intervals.end());
      int64_t covered_until = 0;
      for (const auto& [start_bytes, end_bytes] : intervals) {
        if (start_bytes != covered_until) {
          return absl::InvalidArgumentError(absl::StrCat(
              "receiver schedule has a destination coverage ",
              start_bytes < covered_until ? "overlap" : "gap", " for block ",
              view.dst_ids[ordinal], " at byte ", start_bytes));
        }
        covered_until = end_bytes;
      }
      if (covered_until != view.extents[ordinal]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "receiver schedule does not cover the exact live bytes for "
            "destination block ",
            view.dst_ids[ordinal], ": covered=", covered_until,
            " expected=", view.extents[ordinal]));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<ReshardReceiveSession>>
ReshardReceiveSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_blocks,
    std::chrono::steady_clock::time_point deadline) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  TF_RETURN_IF_ERROR(ValidatePlan(*base, plan, chip_blocks));
  // Device-only executor (see PoolReshardPush): arming a receive on a
  // host-only manager is refused rather than silently landing in mirrors.
  if (!base->has_device_buffers()) {
    return absl::FailedPreconditionError(
        "pool reshard receive requires a device-attached manager; host-only "
        "managers are not supported");
  }
  if (plan.dst_mem_type() != ::tpu_sync::rpc::MEMORY_TYPE_HBM) {
    return absl::InvalidArgumentError(
        "pool reshard receiver requires dst_mem_type=HBM");
  }
  ABSL_ASSIGN_OR_RETURN(
      TransferPlan transfer_plan,
      TransferPlan::Create(plan, /*is_sender=*/false, base->num_pools(),
                           base->has_explicit_pools(), base->pools()));
  auto session =
      std::shared_ptr<ReshardReceiveSession>(new ReshardReceiveSession(
          base, std::move(transfer_plan), chip_blocks, deadline));
  if (absl::Status status = session->AcquireStagingLeases(plan); !status.ok()) {
    session->ReleaseStaging();
    return status;
  }
  return session;
}

ReshardReceiveSession::ReshardReceiveSession(
    kv_cache::KVCacheManagerBase* base, TransferPlan transfer_plan,
    absl::Span<const int64_t> chip_blocks,
    std::chrono::steady_clock::time_point deadline)
    : base_(base),
      deadline_(deadline),
      start_time_(std::chrono::steady_clock::now()),
      transfer_plan_(
          std::make_shared<const TransferPlan>(std::move(transfer_plan))),
      staging_layer_manager_(base != nullptr ? base->num_host_layers() : 0) {
  absl::MutexLock lock(mu_);
  chip_block_ids_.assign(chip_blocks.begin(), chip_blocks.end());
  for (int32_t pool_idx : request().transfer_pool_indices()) {
    expected_pool_indices_.insert(static_cast<size_t>(pool_idx));
    pool_order_ranks_[static_cast<size_t>(pool_idx)] = 0;
  }
  for (const auto& group : request().pool_groups()) {
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
    if (!dst_ids_by_storage.empty()) {
      staging_released_ = false;
    }
  }
  for (const auto& [storage_idx, ids] : dst_ids_by_storage) {
    std::vector<int64_t> id_list(ids.begin(), ids.end());
    absl::Status lease_status = base_->AcquirePoolStagingLease(
        uuid(), storage_idx, id_list, base_->pool_staging_lease_timeout());
    if (!lease_status.ok()) {
      return lease_status;
    }
  }
  return absl::OkStatus();
}

void ReshardReceiveSession::ReleaseStagingLocked() {
  staging_layer_manager_.ReleaseAll();
  if (staging_released_) return;
  staging_released_ = true;
  base_->ReleasePoolStagingLeases(uuid());
}

void ReshardReceiveSession::ReleaseStaging() {
  absl::MutexLock lock(mu_);
  ReleaseStagingLocked();
}

void ReshardReceiveSession::FinishLocked(const absl::Status& status) {
  if (!status.ok() && status_.ok()) {
    status_ = status;
  }
  if (draining_) return;
  draining_ = true;
  cancelled_.store(true, std::memory_order_relaxed);
  staging_layer_manager_.Shutdown();
  if (in_flight_ == 0 && !done_) {
    ReleaseStagingLocked();
    done_ = true;
  }
}

absl::StatusOr<size_t> ReshardReceiveSession::AcquireStagingSlot(
    size_t pool_idx) {
  std::chrono::steady_clock::time_point deadline;
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return absl::CancelledError(
          absl::StrCat("Receive session ", uuid(), " is draining"));
    }
    ++in_flight_;
    deadline = deadline_;
  }
  absl::StatusOr<size_t> slot =
      staging_layer_manager_.Acquire(pool_idx, deadline, &cancelled_);
  absl::MutexLock lock(mu_);
  if (!slot.ok() && !absl::IsCancelled(slot.status())) {
    FinishLocked(slot.status());
  } else if (slot.ok() && (done_ || draining_)) {
    staging_layer_manager_.Release(pool_idx);
    slot = absl::CancelledError(
        absl::StrCat("Receive session ", uuid(), " is draining"));
  }
  EndRecvOpLocked();
  return slot;
}

absl::StatusOr<size_t> ReshardReceiveSession::LookupStagingSlot(
    size_t pool_idx) const {
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return absl::CancelledError(
          absl::StrCat("Receive session ", uuid(), " is draining"));
    }
  }
  std::optional<size_t> slot = staging_layer_manager_.Lookup(pool_idx);
  if (!slot.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Pool ", pool_idx, " has no leased staging slot"));
  }
  return *slot;
}

void ReshardReceiveSession::Finish(const absl::Status& status) {
  absl::MutexLock lock(mu_);
  FinishLocked(status);
}

absl::Status ReshardReceiveSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

void ReshardReceiveSession::EndRecvOpLocked() {
  if (in_flight_ <= 0) {
    LOG(DFATAL) << "Receive operation count underflow for UUID " << uuid();
    return;
  }
  --in_flight_;
  if (draining_ && in_flight_ == 0 && !done_) {
    ReleaseStagingLocked();
    done_ = true;
  }
}

void ReshardReceiveSession::EndOp() {
  absl::MutexLock lock(mu_);
  EndRecvOpLocked();
}

absl::Status ReshardReceiveSession::RecordPoolReceivedLocked(size_t pool_idx) {
  if (expected_pool_indices_.find(pool_idx) == expected_pool_indices_.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "received undeclared pool ", pool_idx, " for UUID ", uuid()));
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

absl::Status ReshardReceiveSession::OnPoolReceived(size_t pool_idx) {
  {
    absl::MutexLock lock(mu_);
    if (done_) {
      return absl::NotFoundError(
          absl::StrCat("no active receiver for UUID ", uuid()));
    }
    TF_RETURN_IF_ERROR(RecordPoolReceivedLocked(pool_idx));
  }
  ExecuteEligiblePoolH2ds();
  return absl::OkStatus();
}

void ReshardReceiveSession::ExecuteEligiblePoolH2ds() {
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
    {
      // The decision to upload was taken under |mu_| above; the receive may
      // have been finished (deadline, cancellation) since. Re-check
      // immediately before issuing so a finished receive starts no upload
      // it does not need. A copy already handed to the device cannot be
      // revoked; its in-flight count keeps the receive -- and its blocks --
      // owned until the copy ends.
      absl::MutexLock lock(mu_);
      if (done_ || draining_) {
        staging_layer_manager_.Release(pool_idx);
        EndRecvOpLocked();
        continue;
      }
    }
    if (dst_chip_block_ids.empty()) {
      // ValidateReceiverCoverage rejects groups without destination blocks,
      // so an empty list here is a broken invariant, not a legal no-op.
      FinishPoolH2d(pool_idx, absl::InternalError(absl::StrCat(
                                  "pool ", pool_idx, " of UUID ", uuid(),
                                  " has no destination blocks to upload")));
      staging_layer_manager_.Release(pool_idx);
      EndOp();
      continue;
    }
    absl::StatusOr<size_t> slot = LookupStagingSlot(pool_idx);
    if (!slot.ok()) {
      FinishPoolH2d(pool_idx, slot.status());
      EndOp();
      continue;
    }
    const size_t host_layer_idx = *slot;
    absl::StatusOr<raiden::PjRtCopyFuture> copy_future = base_->H2dPoolBlocks(
        pool_idx, dst_chip_block_ids,
        /*shard_idx=*/std::nullopt, uuid(), host_layer_idx);
    if (!copy_future.ok()) {
      FinishPoolH2d(pool_idx, copy_future.status());
      staging_layer_manager_.Release(pool_idx);
      EndOp();
      continue;
    }
    raiden::PjRtCopyFuture future = *std::move(copy_future);
    future.OnReady(
        [self = shared_from_this(), pool_idx = pool_idx](auto copy_status) {
          if (copy_status.ok()) {
            self->staging_layer_manager_.Release(pool_idx);
            self->FinishPoolH2d(pool_idx, absl::OkStatus());
          } else {
            self->FinishPoolH2d(pool_idx, copy_status.status());
            self->staging_layer_manager_.Release(pool_idx);
          }
          self->EndOp();
        });
  }
}

void ReshardReceiveSession::FinishPoolH2d(size_t pool_idx,
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
    ExecuteEligiblePoolH2ds();
  }
  std::chrono::steady_clock::time_point session_start_time;
  bool should_record_duration = false;
  if (finished) {
    // The plan lives in this session, so it stays until the receive settles
    // (every upload and every admitted push ended) and goes with its staging:
    // it never outlives the staging or is torn down under a push still
    // resolving through it.
    absl::Status terminal_status = absl::OkStatus();
    {
      absl::MutexLock lock(mu_);
      if (!status.ok()) {
        terminal_status = status;
      } else {
        session_start_time = start_time_;
        should_record_duration = true;
      }
    }
    Finish(terminal_status);
  }
  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(session_start_time, std::chrono::steady_clock::now()));
  }
}

}  // namespace tpu_raiden

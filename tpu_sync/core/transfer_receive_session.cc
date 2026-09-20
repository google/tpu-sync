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

#include "tpu_sync/core/transfer_receive_session.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <ratio>  // NOLINT(build/c++11)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/metrics_collector.h"  // IWYU pragma: keep
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_send_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {

namespace {

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  if (status.code() == absl::StatusCode::kInvalidArgument) {
    throw std::invalid_argument(context + ": " + std::string(status.message()));
  }
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

void CheckStatus(const std::string& context, const absl::Status& status) {
  if (!status.ok()) {
    ThrowStatus(context, status);
  }
}

double DurationMs(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void RecordTransferDuration(double duration_ms) {
  telemetry::RaidenMetricStore::GetGlobalMetricStore().ObserveHistogram(
      telemetry::metric_names::kTransferDurationMs, {}, duration_ms);
}

}  // namespace

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, uint64_t uuid,
    const std::string& req_id, const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::optional<std::vector<int64_t>>& local_host_block_ids,
    std::chrono::steady_clock::time_point deadline) {
  auto session = std::shared_ptr<TransferReceiveSession>(
      new TransferReceiveSession(base, staging_allocator, uuid));
  std::vector<int64_t> host_block_ids;
  if (!session->AllocateStagingForLoad(req_id, local_block_ids,
                                       local_host_block_ids, &host_block_ids)) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Failed to allocate staging for load req_id=", req_id,
                     ", uuid=", uuid));
  }
  CopyPlan load_plan =
      BuildLoadCopyPlan(remote_block_ids, local_block_ids, host_block_ids);
  session->InitFromLoadPlan(req_id, std::move(load_plan), deadline);
  return session;
}

std::shared_ptr<TransferReceiveSession> TransferReceiveSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, uint64_t uuid, std::string req_id,
    int32_t total_blocks, std::chrono::steady_clock::time_point deadline,
    bool acquire_staging) {
  return std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
      base, staging_allocator, uuid, std::move(req_id), total_blocks, deadline,
      acquire_staging));
}

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::CreateFromActivePlan(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, uint64_t uuid,
    const ::tpu_sync::rpc::StartTransferRequest& request, uint64_t generation,
    std::chrono::steady_clock::time_point deadline,
    absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>*
        host_block_of) {
  auto session = std::shared_ptr<TransferReceiveSession>(
      new TransferReceiveSession(base, staging_allocator, uuid));
  ABSL_RETURN_IF_ERROR(session->InitFromActivePlan(request, generation,
                                                   deadline, host_block_of));
  return session;
}

std::shared_ptr<TransferReceiveSession>
TransferReceiveSession::CreateFromPoolReshardPlan(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_blocks,
    std::chrono::steady_clock::time_point deadline) {
  auto session = std::shared_ptr<TransferReceiveSession>(
      new TransferReceiveSession(base, staging_allocator, plan.uuid()));
  session->InitFromPoolReshardPlan(plan, chip_blocks, deadline);
  return session;
}

absl::Status TransferReceiveSession::InitFromActivePlan(
    const ::tpu_sync::rpc::StartTransferRequest& request, uint64_t generation,
    std::chrono::steady_clock::time_point deadline,
    absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>*
        host_block_of) {
  absl::MutexLock lock(mu_);
  if (staging_allocator_ != nullptr &&
      staging_allocator_->dynamic_host_staging() &&
      request.pool_groups_size() == 0) {
    std::vector<int64_t> device_blocks;
    absl::flat_hash_set<int64_t> seen;
    for (const auto& [src_shard, schedule] : request.shard_push_schedules()) {
      for (const auto& e : schedule.entries()) {
        int64_t id = e.dst_block_id();
        if (seen.insert(id).second) device_blocks.push_back(id);
      }
    }
    if (!device_blocks.empty()) {
      absl::StatusOr<StagingAllocation> allocated =
          staging_allocator_->AcquireDynamicBlocks(
              static_cast<int64_t>(device_blocks.size()));
      if (!allocated.ok()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "cannot stage ", device_blocks.size(), " blocks for plan ", uuid_,
            ": ", allocated.status().message()));
      }
      staging_ = *std::move(allocated);
      absl::Span<const int> plan_blocks = staging_.block_ids();
      for (size_t i = 0; i < device_blocks.size(); ++i) {
        (*host_block_of)[device_blocks[i]] = plan_blocks[i];
      }
    }
  }
  unregister_on_settle_ = !staging_.empty();
  plan_generation_ = generation;
  req_id_ = request.req_id().empty()
                ? absl::StrCat("resharded_transfer_", uuid_)
                : request.req_id();

  absl::flat_hash_set<int> unique_dst_blocks;
  int64_t expected_blocks = 0;
  for (const auto& [src_replica_idx, schedule] :
       request.shard_push_schedules()) {
    absl::flat_hash_set<std::pair<int, int>> unique_transfers_from_this_source;
    for (const auto& push_entry : schedule.entries()) {
      const int64_t dst = push_entry.dst_block_id();
      unique_dst_blocks.insert(dst);
      auto hb = host_block_of->find(dst);
      host_to_chip_[hb == host_block_of->end() ? dst : hb->second] = dst;
      unique_transfers_from_this_source.insert(
          {push_entry.src_block_id(), push_entry.dst_block_id()});
    }
    expected_blocks += unique_transfers_from_this_source.size();
  }
  std::vector<int64_t> h2d_local_block_ids(unique_dst_blocks.begin(),
                                           unique_dst_blocks.end());
  std::vector<int64_t> h2d_host_block_ids;
  h2d_host_block_ids.reserve(h2d_local_block_ids.size());
  for (int64_t dst : h2d_local_block_ids) {
    auto hb = host_block_of->find(dst);
    h2d_host_block_ids.push_back(hb == host_block_of->end() ? dst : hb->second);
  }
  total_blocks_ = expected_blocks;
  num_completed_blocks_ = 0;
  deadline_ = deadline;
  start_time_ = std::chrono::steady_clock::now();
  h2d_copy_ = TransferSendSession::BuildCoalescedCopySpec(h2d_host_block_ids,
                                                          h2d_local_block_ids);
  if (total_blocks_ == 0) {
    ReleaseStagingLocked();
  }
  return absl::OkStatus();
}

void TransferReceiveSession::InitFromPoolReshardPlan(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_blocks,
    std::chrono::steady_clock::time_point deadline) {
  absl::MutexLock lock(mu_);
  req_id_ = plan.req_id();
  is_pool_reshard_ = true;
  unregister_on_settle_ = true;
  deadline_ = deadline;
  start_time_ = std::chrono::steady_clock::now();
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

bool TransferReceiveSession::AllocateStagingForLoad(
    const std::string& req_id, const std::vector<int64_t>& local_block_ids,
    const std::optional<std::vector<int64_t>>& local_host_block_ids,
    std::vector<int64_t>* host_block_ids) {
  absl::MutexLock lock(mu_);
  if (local_host_block_ids.has_value()) {
    *host_block_ids = *local_host_block_ids;
    return true;
  }
  if (local_block_ids.empty()) {
    host_block_ids->clear();
    return true;
  }
  absl::flat_hash_set<int64_t> unique_local_bids(local_block_ids.begin(),
                                                 local_block_ids.end());
  std::optional<StagingAllocation> acquired = staging_allocator_->Acquire(
      static_cast<int64_t>(unique_local_bids.size()));
  if (!acquired.has_value()) {
    LOG(ERROR) << "StartRead: cannot stage " << unique_local_bids.size()
               << " blocks for req_id=" << req_id
               << " (dynamic=" << staging_allocator_->dynamic_host_staging()
               << ", free_host_blocks="
               << base_->host_block_manager()->num_free_blocks()
               << ", free_slots=" << staging_allocator_->num_free_slots()
               << ", max_blocks=" << staging_allocator_->max_blocks() << ")";
    return false;
  }
  staging_ = *std::move(acquired);
  absl::Span<const int> staged_blocks = staging_.block_ids();
  absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
      local_to_host;
  size_t host_block_idx = 0;
  host_block_ids->clear();
  host_block_ids->reserve(local_block_ids.size());
  for (size_t k = 0; k < local_block_ids.size(); ++k) {
    int64_t local_bid = local_block_ids[k];
    auto it = local_to_host.find(local_bid);
    if (it == local_to_host.end()) {
      int64_t host_bid = staged_blocks[host_block_idx++];
      local_to_host[local_bid] = host_bid;
      host_block_ids->push_back(host_bid);
    } else {
      host_block_ids->push_back(it->second);
    }
  }
  return true;
}

CopyPlan TransferReceiveSession::BuildLoadCopyPlan(
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::vector<int64_t>& local_host_block_ids) {
  if (remote_block_ids.size() != local_block_ids.size() ||
      local_block_ids.size() != local_host_block_ids.size()) {
    throw std::invalid_argument(
        "remote_block_ids, local_block_ids, and local_host_block_ids must have "
        "same length");
  }
  CopyPlan plan;
  plan.num_blocks = static_cast<int64_t>(remote_block_ids.size());
  plan.requested_remote_block_ids = remote_block_ids;
  plan.requested_local_block_ids = local_block_ids;
  if (remote_block_ids.empty()) {
    return plan;
  }

  // 1. Determine transport order (sorted by remote_block_ids)
  std::vector<size_t> remote_order(remote_block_ids.size());
  for (size_t i = 0; i < remote_order.size(); ++i) {
    remote_order[i] = i;
  }
  std::stable_sort(remote_order.begin(), remote_order.end(),
                   [&](size_t a, size_t b) {
                     return remote_block_ids[a] < remote_block_ids[b];
                   });

  plan.producer_remote_block_ids.reserve(remote_order.size());
  plan.transport_host_block_ids.reserve(remote_order.size());
  for (size_t i = 0; i < remote_order.size(); ++i) {
    const size_t original_idx = remote_order[i];
    plan.producer_remote_block_ids.push_back(remote_block_ids[original_idx]);
    plan.transport_host_block_ids.push_back(local_host_block_ids[original_idx]);
  }

  // 2. Determine H2D copy plan (sorted by local_block_ids for opt)
  std::vector<size_t> local_order(local_block_ids.size());
  for (size_t i = 0; i < local_order.size(); ++i) {
    local_order[i] = i;
  }
  std::stable_sort(local_order.begin(), local_order.end(),
                   [&](size_t a, size_t b) {
                     return local_block_ids[a] < local_block_ids[b];
                   });

  plan.h2d_local_block_ids.reserve(local_order.size());
  plan.h2d_host_block_ids.reserve(local_order.size());
  for (size_t i = 0; i < local_order.size(); ++i) {
    const size_t original_idx = local_order[i];
    int64_t local_bid = local_block_ids[original_idx];
    int64_t host_bid = local_host_block_ids[original_idx];
    if (plan.h2d_local_block_ids.empty() ||
        plan.h2d_local_block_ids.back() != local_bid) {
      plan.h2d_local_block_ids.push_back(local_bid);
      plan.h2d_host_block_ids.push_back(host_bid);
    } else {
      if (plan.h2d_host_block_ids.back() != host_bid) {
        throw std::invalid_argument(
            "Duplicate local block IDs must map to the same host block ID");
      }
    }
  }

  plan.h2d_copy = TransferSendSession::BuildCoalescedCopySpec(
      plan.h2d_host_block_ids, plan.h2d_local_block_ids);
  plan.host_dst_to_src.clear();
  return plan;
}

void TransferReceiveSession::InitFromLoadPlan(
    const std::string& req_id, CopyPlan load_plan,
    std::chrono::steady_clock::time_point deadline) {
  absl::MutexLock lock(mu_);
  req_id_ = req_id;
  deadline_ = deadline;
  start_time_ = std::chrono::steady_clock::now();
  chip_block_ids_ = load_plan.h2d_local_block_ids;
  total_blocks_ = load_plan.num_blocks;
  num_completed_blocks_ = 0;
  num_completed_layers_ = 0;
  in_flight_ = load_plan.num_blocks > 0 ? 1 : 0;
  h2d_copy_ = load_plan.h2d_copy;
  for (size_t i = 0; i < load_plan.transport_host_block_ids.size(); ++i) {
    host_to_chip_[load_plan.transport_host_block_ids[i]] =
        load_plan.h2d_local_block_ids[i];
  }
  h2d_dispatch_futures_.reserve(load_plan.h2d_local_block_ids.size());
  load_plan_ = std::move(load_plan);
}

void TransferReceiveSession::ReleaseStagingLocked() {
  staging_.Reset();
  if (base_ != nullptr && is_pool_reshard_) {
    base_->ReleasePoolStagingLeases(uuid_);
  }
}

void TransferReceiveSession::ReleaseStaging() {
  absl::MutexLock lock(mu_);
  ReleaseStagingLocked();
}

void TransferReceiveSession::FinishRecvLocked(bool has_failed,
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

void TransferReceiveSession::FinishRecv(bool has_failed,
                                        bool unregister_on_settle) {
  absl::MutexLock lock(mu_);
  FinishRecvLocked(has_failed, unregister_on_settle);
}

void TransferReceiveSession::EndRecvOpLocked() {
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

void TransferReceiveSession::EndRecvOp() {
  absl::MutexLock lock(mu_);
  EndRecvOpLocked();
}

bool TransferReceiveSession::DeferUnregisterOnSettle() {
  absl::MutexLock lock(mu_);
  if (is_pool_reshard_ || done_) {
    return false;
  }
  unregister_on_settle_ = true;
  return true;
}

bool TransferReceiveSession::TakePendingUnregister(uint64_t* generation) {
  absl::MutexLock lock(mu_);
  if (!unregister_on_settle_) return false;
  unregister_on_settle_ = false;
  if (generation != nullptr) {
    *generation = plan_generation_;
  }
  return true;
}

bool TransferReceiveSession::AllH2dDoneLocked() const {
  for (const auto& f : h2d_futures_) {
    if (!f.IsReady()) return false;
  }
  return true;
}

bool TransferReceiveSession::IsReadyToComplete() const {
  absl::MutexLock lock(mu_);
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  return (network_completed_ ||
          num_completed_layers_ == static_cast<int32_t>(total_layers)) &&
         AllH2dDoneLocked();
}

bool TransferReceiveSession::HasPendingWork() const {
  absl::MutexLock lock(mu_);
  if (done_) return false;
  if (!is_pool_reshard_ || !network_completed_) return true;
  return !AllH2dDoneLocked();
}

bool TransferReceiveSession::RecordBlocksReceivedLocked(
    const std::vector<int>& block_ids, bool* first_packet,
    bool* network_just_completed) {
  *first_packet = false;
  *network_just_completed = false;
  num_completed_blocks_ += block_ids.size();
  if (num_completed_blocks_ == static_cast<int32_t>(block_ids.size())) {
    *first_packet = true;
  }
  accumulated_host_block_ids_.insert(accumulated_host_block_ids_.end(),
                                     block_ids.begin(), block_ids.end());
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  if (num_completed_blocks_ >=
      total_blocks_ * static_cast<int32_t>(total_layers)) {
    network_completed_ = true;
    *network_just_completed = true;
    return num_completed_layers_ == static_cast<int32_t>(total_layers);
  }
  return false;
}

absl::Status TransferReceiveSession::RecordPoolReceivedLocked(size_t pool_idx) {
  if (!is_pool_reshard_) {
    return absl::FailedPreconditionError(
        absl::StrCat("pool completion for UUID ", uuid_,
                     " but the receiver was armed on the legacy path"));
  }
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
TransferReceiveSession::CollectEligiblePoolH2dsLocked() {
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

bool TransferReceiveSession::RecordPoolH2dResultLocked(
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

void TransferReceiveSession::ExecutePullRequest(
    KVCacheManagerWithTransfer& manager, const std::string& remote_endpoint) {
  std::optional<int> target_node = base_->assigned_numa_node();
  std::string session_req_id;
  CopyPlan load_plan;
  {
    absl::MutexLock lock(mu_);
    session_req_id = req_id_;
    load_plan = load_plan_;
  }

  base_->push_pool()->Schedule(
      target_node, [this, &manager, remote_endpoint, session_req_id,
                    load_plan = std::move(load_plan)]() {
        bool pull_failed = false;
        try {
          LOG(INFO) << "StartRead (connecting): req_id=" << session_req_id
                    << ", uuid=" << uuid_
                    << ", numa=" << base_->assigned_numa_node().value_or(-1);
          PullStreamRequestSpec req_spec;
          req_spec.uuid = uuid_;
          req_spec.ep_idx = 0;
          req_spec.consumer_data_port =
              static_cast<uint32_t>(manager.local_data_port_);
          req_spec.consumer_ips = base_->local_ips();
          req_spec.src_block_ids = load_plan.producer_remote_block_ids;
          req_spec.dst_block_ids = load_plan.transport_host_block_ids;

          absl::StatusOr<PullStreamResponseSpec> response =
              manager.control_backend_->SendPullRequest(
                  remote_endpoint, req_spec, absl::Seconds(manager.timeout_s_));
          CheckStatus("control pull request", response.status());
          if (response->status != 0) {
            throw std::runtime_error(absl::StrCat(
                "Remote producer rejected Hybrid Bridge read request: ",
                response->message));
          }
          VLOG(1) << "StartRead (Hybrid Bridge) successfully registered pull "
                     "request with Producer. req_id: "
                  << session_req_id;
        } catch (const std::exception& e) {
          pull_failed = true;
          LOG(ERROR) << "Raiden consumer error during Hybrid Bridge StartRead "
                        "connect: "
                     << e.what();
        }

        absl::MutexLock lock(mu_);
        if (pull_failed) {
          FinishRecvLocked(/*has_failed=*/true);
        }
        EndRecvOpLocked();
      });
}

absl::Status TransferReceiveSession::OnBlocksReceived(
    KVCacheManagerWithTransfer& manager, const std::vector<int>& block_ids) {
  const uint64_t uuid = uuid_;
  const int numa_node = base_->assigned_numa_node().value_or(-1);
  std::chrono::steady_clock::time_point session_start_time;
  bool should_record_duration = false;
  bool first_packet = false;
  bool network_just_completed = false;
  std::string session_req_id;
  bool all_complete = false;
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_ || is_pool_reshard_) {
      return absl::OkStatus();
    }
    all_complete = RecordBlocksReceivedLocked(block_ids, &first_packet,
                                              &network_just_completed);
    MetricsCollector* const metrics = manager.metrics_collector_.get();
    if (first_packet && metrics != nullptr) {
      metrics->RecordFirstPacket(uuid_);
    }
    if (!network_just_completed) {
      VLOG(1) << "OnBlocksReceived: Partial blocks received for uuid " << uuid_
              << ", completed: " << num_completed_blocks_ << " / "
              << total_blocks_ * base_->num_layers();
      return absl::OkStatus();
    }
    session_req_id = req_id_;
    if (metrics != nullptr) {
      metrics->RecordLastPacket(uuid_);
    }
    if (all_complete) {
      session_start_time = start_time_;
      should_record_duration = true;
      if (metrics != nullptr) {
        metrics->RecordEnd(uuid_);
      }
      FinishRecvLocked(/*has_failed=*/false);
    }
  }

  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(session_start_time, std::chrono::steady_clock::now()));
    LOG(INFO) << "OnBlocksReceived (Network + H2D complete): req_id="
              << session_req_id << ", uuid=" << uuid << ", numa=" << numa_node;
  }
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::ExecuteLayerH2d(
    KVCacheManagerWithTransfer& manager, size_t layer_idx) {
  CopySpec copy_spec;
  std::string session_req_id;
  bool trigger_enqueue = false;
  {
    absl::MutexLock lock(mu_);
    if (done_ || is_pool_reshard_ || draining_) {
      return absl::OkStatus();
    }
    ++in_flight_;
    copy_spec = h2d_copy_;
    session_req_id = req_id_;
    if (!h2d_started_) {
      h2d_started_ = true;
      trigger_enqueue = true;
    }
  }
  if (trigger_enqueue && manager.metrics_collector_) {
    manager.metrics_collector_->RecordH2dEnqueue(uuid_);
  }

  LOG(INFO) << "OnLayerReceived (H2D copy start) layer " << layer_idx
            << ": req_id=" << session_req_id << ", uuid=" << uuid_
            << ", numa=" << base_->assigned_numa_node().value_or(-1);

  auto future_or =
      base_->H2dSyncDispatch(copy_spec.src_offsets, copy_spec.dst_offsets,
                             copy_spec.sizes, /*slot_idx=*/std::nullopt,
                             /*layer_idx=*/layer_idx);
  if (!future_or.ok()) {
    absl::MutexLock lock(mu_);
    FinishRecvLocked(/*has_failed=*/true);
    EndRecvOpLocked();
    return future_or.status();
  }

  auto future = *future_or;
  {
    absl::MutexLock lock(mu_);
    h2d_futures_.push_back(future);
  }
  const uint64_t uuid = uuid_;
  const int numa_node = base_->assigned_numa_node().value_or(-1);
  future.OnReady([this, &manager, uuid, numa_node, layer_idx, session_req_id,
                  metrics_collector =
                      manager.metrics_collector_](auto status_or) {
    bool all_layers_done = false;
    std::chrono::steady_clock::time_point session_start_time;
    bool should_unregister = false;
    uint64_t generation = 0;
    {
      absl::MutexLock lock(mu_);
      if (done_) {
        LOG(DFATAL) << "H2D callback for retired receive UUID " << uuid;
        return;
      }
      if (status_or.ok()) {
        LOG(INFO) << "OnLayerReceived (H2D copy complete) layer " << layer_idx
                  << ": req_id=" << session_req_id << ", numa=" << numa_node;
        num_completed_layers_++;
        if (num_completed_layers_ ==
                static_cast<int32_t>(base_->num_layers()) &&
            !draining_) {
          all_layers_done = true;
          session_start_time = start_time_;
          FinishRecvLocked(/*has_failed=*/false);
        }
      } else {
        LOG(ERROR) << "OnLayerReceived (H2D copy failed) layer " << layer_idx
                   << " for req_id: " << session_req_id
                   << ", error: " << status_or.status().ToString();
        FinishRecvLocked(/*has_failed=*/true);
      }
      EndRecvOpLocked();
      if (done_ && unregister_on_settle_) {
        unregister_on_settle_ = false;
        generation = plan_generation_;
        should_unregister = true;
      }
    }
    if (all_layers_done) {
      RecordTransferDuration(
          DurationMs(session_start_time, std::chrono::steady_clock::now()));
      if (metrics_collector) {
        metrics_collector->RecordH2dComplete(uuid);
      }
      LOG(INFO) << "All layers H2D copy complete: req_id=" << session_req_id;
      if (metrics_collector) {
        metrics_collector->RecordEnd(uuid);
      }
    }
    if (should_unregister) {
      manager.UnregisterSettledPlan(uuid, generation);
    }
  });

  return absl::OkStatus();
}

absl::Status TransferReceiveSession::OnPoolReceived(
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

void TransferReceiveSession::ExecuteEligiblePoolH2ds(
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

void TransferReceiveSession::FinishPoolH2d(KVCacheManagerWithTransfer& manager,
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

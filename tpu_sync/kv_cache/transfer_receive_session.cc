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

#include "tpu_sync/kv_cache/transfer_receive_session.h"

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <ratio>  // NOLINT(build/c++11)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/metrics_collector.h"  // IWYU pragma: keep
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/kv_cache/transfer_send_session.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {

namespace {

// The steps of a receive, as named in `TransferDetails::phase`.
constexpr absl::string_view kConnectPhase = "connect";
constexpr absl::string_view kReceivePhase = "receive";
constexpr absl::string_view kH2dPhase = "h2d";

double DurationMs(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void RecordTransferDuration(double duration_ms) {
  telemetry::RaidenMetricStore::GetGlobalMetricStore().ObserveHistogram(
      telemetry::metric_names::kTransferDurationMs, {}, duration_ms);
}

absl::Time ToAbslDeadline(std::chrono::steady_clock::time_point deadline) {
  const auto remaining = std::max(std::chrono::steady_clock::duration::zero(),
                                  deadline - std::chrono::steady_clock::now());
  return absl::Now() + absl::FromChrono(remaining);
}

}  // namespace

StagingLayerManager::StagingLayerManager(size_t num_slots) : free_(num_slots) {
  std::iota(free_.begin(), free_.end(), 0);
}

absl::StatusOr<size_t> StagingLayerManager::Acquire(
    size_t unit, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool>* cancelled) {
  absl::MutexLock lock(mu_);
  if (free_.empty() && leases_.empty()) {
    return absl::FailedPreconditionError("StagingLayerManager has no slots");
  }
  auto ready = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return shutdown_ ||
           (cancelled != nullptr &&
            cancelled->load(std::memory_order_relaxed)) ||
           leases_.contains(unit) || !free_.empty();
  };
  if (!mu_.AwaitWithDeadline(absl::Condition(&ready),
                             ToAbslDeadline(deadline))) {
    return absl::DeadlineExceededError(absl::StrCat(
        "Timed out waiting for a free staging slot for unit ", unit));
  }
  if (shutdown_ ||
      (cancelled != nullptr && cancelled->load(std::memory_order_relaxed))) {
    return absl::CancelledError("StagingLayerManager is shutting down");
  }
  if (auto it = leases_.find(unit); it != leases_.end()) {
    return it->second;
  }
  const size_t slot = free_.front();
  free_.pop_front();
  leases_.emplace(unit, slot);
  return slot;
}

std::optional<size_t> StagingLayerManager::Lookup(size_t unit) const {
  absl::MutexLock lock(mu_);
  auto it = leases_.find(unit);
  if (it == leases_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<size_t> StagingLayerManager::Release(size_t unit) {
  absl::MutexLock lock(mu_);
  auto it = leases_.find(unit);
  if (it == leases_.end()) {
    return std::nullopt;
  }
  const size_t slot = it->second;
  leases_.erase(it);
  free_.push_back(slot);
  return slot;
}

void StagingLayerManager::ReleaseAll() {
  absl::MutexLock lock(mu_);
  for (const auto& [unit, slot] : leases_) {
    (void)unit;
    free_.push_back(slot);
  }
  leases_.clear();
}

void StagingLayerManager::Shutdown() {
  absl::MutexLock lock(mu_);
  shutdown_ = true;
}

size_t StagingLayerManager::num_free() const {
  absl::MutexLock lock(mu_);
  return free_.size();
}

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    const std::string& req_id, const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::optional<std::vector<int64_t>>& local_host_block_ids,
    std::chrono::steady_clock::time_point deadline,
    std::shared_ptr<MetricsCollector> metrics_collector) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  auto session =
      std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
          base, staging_allocator, uuid, std::move(metrics_collector)));
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

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    std::string req_id, int32_t total_blocks,
    std::chrono::steady_clock::time_point deadline, bool acquire_staging,
    std::shared_ptr<MetricsCollector> metrics_collector) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  return std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
      base, staging_allocator, uuid, std::move(req_id), total_blocks, deadline,
      acquire_staging, std::move(metrics_collector)));
}

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::CreateFromActivePlan(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    const ::tpu_sync::rpc::StartTransferRequest& request,
    std::chrono::steady_clock::time_point deadline,
    std::shared_ptr<MetricsCollector> metrics_collector) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  StagingAllocation staging;
  absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
      host_block_of;
  if (staging_allocator->dynamic_host_staging() &&
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
      absl::StatusOr<StagingAllocation> allocated = staging_allocator->Acquire(
          static_cast<int64_t>(device_blocks.size()));
      if (!allocated.ok()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "cannot stage ", device_blocks.size(), " blocks for plan ", uuid,
            ": ", allocated.status().message()));
      }
      staging = *std::move(allocated);
      absl::Span<const int> plan_blocks = staging.block_ids();
      for (size_t i = 0; i < device_blocks.size(); ++i) {
        host_block_of[device_blocks[i]] = plan_blocks[i];
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      TransferPlan plan,
      TransferPlan::Create(request, /*is_sender=*/false, base->num_pools(),
                           base->has_explicit_pools(), base->pools(),
                           host_block_of));
  auto session =
      std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
          base, staging_allocator, uuid, std::move(metrics_collector),
          std::make_shared<const TransferPlan>(std::move(plan))));
  session->InitFromActivePlan(request, deadline, std::move(staging),
                              host_block_of);
  return session;
}

void TransferReceiveSession::InitFromActivePlan(
    const ::tpu_sync::rpc::StartTransferRequest& request,
    std::chrono::steady_clock::time_point deadline, StagingAllocation staging,
    const absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>&
        host_block_of) {
  absl::MutexLock lock(mu_);
  staging_ = std::move(staging);
  const bool hbm_receiver =
      request.dst_mem_type() == ::tpu_sync::rpc::MEMORY_TYPE_HBM;
  req_id_ = request.req_id().empty()
                ? absl::StrCat("resharded_transfer_", uuid_)
                : request.req_id();
  deadline_ = deadline;
  start_time_ = std::chrono::steady_clock::now();
  if (!hbm_receiver) {
    total_blocks_ = 0;
    tracks_completion_ = false;
    deadline_ = std::chrono::steady_clock::time_point::max();
    return;
  }

  absl::flat_hash_set<int> unique_dst_blocks;
  int64_t expected_blocks = 0;
  for (const auto& [src_replica_idx, schedule] :
       request.shard_push_schedules()) {
    absl::flat_hash_set<std::pair<int, int>> unique_transfers_from_this_source;
    for (const auto& push_entry : schedule.entries()) {
      const int64_t dst = push_entry.dst_block_id();
      unique_dst_blocks.insert(dst);
      auto hb = host_block_of.find(dst);
      host_to_chip_[hb == host_block_of.end() ? dst : hb->second] = dst;
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
    auto hb = host_block_of.find(dst);
    h2d_host_block_ids.push_back(hb == host_block_of.end() ? dst : hb->second);
  }
  total_blocks_ = expected_blocks;
  ResetShardProgressLocked();
  deadline_ = deadline;
  start_time_ = std::chrono::steady_clock::now();
  h2d_copy_ = TransferSendSession::BuildCoalescedCopySpec(h2d_host_block_ids,
                                                          h2d_local_block_ids);
  if (total_blocks_ == 0) {
    ReleaseStagingLocked();
  }
  tracks_completion_ = total_blocks_ > 0;
  if (!tracks_completion_) {
    deadline_ = std::chrono::steady_clock::time_point::max();
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
  absl::StatusOr<StagingAllocation> acquired = staging_allocator_->Acquire(
      static_cast<int64_t>(unique_local_bids.size()));
  if (!acquired.ok()) {
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
  handshake_pending_ = true;
  chip_block_ids_ = load_plan.h2d_local_block_ids;
  total_blocks_ = load_plan.num_blocks;
  ResetShardProgressLocked();
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
  staging_layer_manager_.ReleaseAll();
  staging_.Reset();
}

void TransferReceiveSession::ReleaseStaging() {
  absl::MutexLock lock(mu_);
  ReleaseStagingLocked();
}

void TransferReceiveSession::FinishLocked(
    const absl::Status& status, std::optional<absl::string_view> phase) {
  if (!status.ok() && status_.ok()) {
    status_ = status;
    failed_phase_ = phase.has_value() ? *phase : CurrentPhaseLocked();
  }
  if (draining_) return;
  draining_ = true;
  cancelled_.store(true, std::memory_order_relaxed);
  staging_layer_manager_.Shutdown();
  if (in_flight_ == 0 && !done_) {
    ReleaseStagingLocked();
    done_ = true;
    completed_at_ = std::chrono::steady_clock::now();
  }
}

void TransferReceiveSession::Finish(const absl::Status& status) {
  absl::MutexLock lock(mu_);
  FinishLocked(status);
}

absl::Status TransferReceiveSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

TransferDetails TransferReceiveSession::Details() const {
  absl::MutexLock lock(mu_);
  TransferDetails details = DetailsFromStatus(status_);
  details.completed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             completed_at_.time_since_epoch())
                             .count();
  details.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           completed_at_ - start_time_)
                           .count();
  if (!peer_endpoint_.empty()) {
    details.peer = peer_endpoint_;
  }
  if (!status_.ok()) {
    details.phase = std::string(failed_phase_);
  }
  return details;
}

absl::string_view TransferReceiveSession::CurrentPhaseLocked() const {
  if (handshake_pending_) return kConnectPhase;
  return network_completed_ ? kH2dPhase : kReceivePhase;
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
    completed_at_ = std::chrono::steady_clock::now();
  }
}

void TransferReceiveSession::EndOp() {
  absl::MutexLock lock(mu_);
  EndRecvOpLocked();
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

void TransferReceiveSession::ResetShardProgressLocked() {
  blocks_received_per_shard_.assign(base_ != nullptr ? base_->num_shards() : 0,
                                    0);
  num_completed_shards_ = 0;
  any_blocks_received_ = false;
}

bool TransferReceiveSession::RecordBlockShardsReceivedLocked(
    const std::vector<int>& block_ids, absl::Span<const int> shard_ids,
    bool* first_packet, bool* network_just_completed) {
  *first_packet = !any_blocks_received_;
  any_blocks_received_ = true;
  *network_just_completed = false;
  accumulated_host_block_ids_.insert(accumulated_host_block_ids_.end(),
                                     block_ids.begin(), block_ids.end());
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  const int64_t per_shard_target =
      static_cast<int64_t>(total_blocks_) * static_cast<int64_t>(total_layers);
  for (const int sh : shard_ids) {
    int64_t& received = blocks_received_per_shard_[sh];
    const bool was_complete = received >= per_shard_target;
    received += static_cast<int64_t>(block_ids.size());
    if (!was_complete && received >= per_shard_target) {
      ++num_completed_shards_;
    }
  }
  if (!network_completed_ &&
      num_completed_shards_ ==
          static_cast<int32_t>(blocks_received_per_shard_.size())) {
    network_completed_ = true;
    *network_just_completed = true;
    return num_completed_layers_ == static_cast<int32_t>(total_layers);
  }
  return false;
}

bool TransferReceiveSession::RecordNetworkCompleteLocked(
    bool all_complete, std::string* session_req_id,
    std::chrono::steady_clock::time_point* session_start_time) {
  network_completed_ = true;
  *session_req_id = req_id_;
  if (metrics_collector_ != nullptr) {
    metrics_collector_->RecordLastPacket(uuid_);
  }
  if (!all_complete) {
    return false;
  }
  *session_start_time = start_time_;
  if (metrics_collector_ != nullptr) {
    metrics_collector_->RecordEnd(uuid_);
  }
  FinishLocked();
  return true;
}

void TransferReceiveSession::ExecutePullRequest(
    KVCacheManagerWithTransfer& manager, const std::string& remote_endpoint) {
  std::optional<int> target_node = base_->assigned_numa_node();
  std::string session_req_id;
  PullStreamRequestSpec req_spec;
  {
    absl::MutexLock lock(mu_);
    session_req_id = req_id_;
    req_spec.src_block_ids = load_plan_.producer_remote_block_ids;
    req_spec.dst_block_ids = load_plan_.transport_host_block_ids;
    peer_endpoint_ = remote_endpoint;
  }
  req_spec.uuid = uuid_;
  req_spec.ep_idx = 0;
  req_spec.consumer_data_port = static_cast<uint32_t>(manager.local_data_port_);
  req_spec.consumer_ips = base_->local_ips();
  req_spec.layer_host_addrs = base_->LayerHostAddrs(uuid_);

  LOG(INFO) << "StartRead (connecting): req_id=" << session_req_id
            << ", uuid=" << uuid_ << ", numa=" << target_node.value_or(-1);

  // May run on a transport thread, so it must not block.
  auto on_response = [self = shared_from_this(), session_req_id](
                         absl::StatusOr<PullStreamResponseSpec> response) {
    absl::Cleanup end_op = [self]() { self->EndOp(); };
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferRecvSessionPullReply);
    if (response.ok() && !injected.ok()) {
      response = injected;
    }
    // Preserve the transport status code (e.g. DEADLINE_EXCEEDED or
    // UNAVAILABLE) when the handshake itself fails, distinguishing a transport
    // failure from an application-level rejection by the producer.
    absl::Status pull_status = absl::OkStatus();
    if (!response.ok()) {
      pull_status = absl::Status(
          response.status().code(),
          absl::StrCat("control pull request: ", response.status().message()));
    } else if (response->status != 0) {
      pull_status = absl::FailedPreconditionError(
          absl::StrCat("Remote producer rejected Hybrid Bridge read request: ",
                       response->message));
    }
    if (pull_status.ok()) {
      VLOG(1) << "StartRead (Hybrid Bridge) successfully registered pull "
                 "request with Producer. req_id: "
              << session_req_id;
    } else {
      LOG(ERROR) << "Raiden consumer error during Hybrid Bridge StartRead "
                    "connect: "
                 << pull_status.message();
    }

    absl::MutexLock lock(self->mu_);
    if (!pull_status.ok()) {
      self->FinishLocked(pull_status);
    }
    self->handshake_pending_ = false;
  };
  // Issued from push_pool_ so a backend that blocks (TCP) does not block the
  // StartRead caller. An async backend (gRPC) returns as soon as the RPC is
  // issued, so the worker is not held while the handshake waits.
  base_->push_pool()->Schedule(
      target_node, [&manager, remote_endpoint, req_spec = std::move(req_spec),
                    on_response = std::move(on_response)]() mutable {
        try {
          FaultInjectThrow(hooks::kTransferRecvSessionPullRequest);
        } catch (const std::exception& e) {
          on_response(absl::InternalError(e.what()));
          return;
        }
        manager.control_backend_
            ->SendPullRequest(remote_endpoint, req_spec,
                              absl::Seconds(manager.control_timeout_s_))
            .OnReady(std::move(on_response));
      });
}

absl::Status TransferReceiveSession::OnRecvProgress() {
  absl::MutexLock lock(mu_);
  if (done_ || draining_) {
    return absl::OkStatus();
  }
  if (!any_blocks_received_) {
    any_blocks_received_ = true;
    if (metrics_collector_ != nullptr) {
      metrics_collector_->RecordFirstPacket(uuid_);
    }
  }
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::OnRecvNetworkComplete() {
  absl::MutexLock lock(mu_);
  if (done_ || draining_ || network_completed_) {
    return absl::OkStatus();
  }
  network_completed_ = true;
  if (metrics_collector_ != nullptr) {
    metrics_collector_->RecordLastPacket(uuid_);
  }
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::OnBlockShardsReceived(
    const std::vector<int>& block_ids, absl::Span<const int> shard_ids) {
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
    if (done_ || draining_ || total_blocks_ == 0) {
      return absl::OkStatus();
    }
    for (const int sh : shard_ids) {
      if (sh < 0 || sh >= static_cast<int>(blocks_received_per_shard_.size())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "shard ", sh, " out of range for uuid ", uuid_,
            " (num_shards=", blocks_received_per_shard_.size(), ")"));
      }
    }
    all_complete = RecordBlockShardsReceivedLocked(
        block_ids, shard_ids, &first_packet, &network_just_completed);
    if (first_packet && metrics_collector_ != nullptr) {
      metrics_collector_->RecordFirstPacket(uuid_);
    }
    if (!network_just_completed) {
      VLOG(1) << "OnBlockShardsReceived: partial for uuid " << uuid_
              << ", shards complete: " << num_completed_shards_ << " / "
              << blocks_received_per_shard_.size() << ", per-shard blocks: ["
              << absl::StrJoin(blocks_received_per_shard_, ",") << "] / "
              << total_blocks_ * base_->num_layers();
      return absl::OkStatus();
    }
    should_record_duration = RecordNetworkCompleteLocked(
        all_complete, &session_req_id, &session_start_time);
  }

  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(session_start_time, std::chrono::steady_clock::now()));
    LOG(INFO) << "OnBlockShardsReceived (Network + H2D complete): req_id="
              << session_req_id << ", uuid=" << uuid << ", numa=" << numa_node;
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> TransferReceiveSession::AcquireStagingSlot(
    size_t layer_idx) {
  // A full host mirror stages every layer in place: there is no slot to lease
  // and callers address the layer's own buffer.
  if (base_ == nullptr || base_->num_host_layers() >= base_->num_layers()) {
    return absl::NotFoundError("no bounded host staging");
  }
  std::chrono::steady_clock::time_point deadline;
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return absl::CancelledError(
          absl::StrCat("Receive session ", uuid_, " is draining"));
    }
    ++in_flight_;
    deadline = deadline_;
  }
  absl::StatusOr<size_t> slot =
      staging_layer_manager_.Acquire(layer_idx, deadline, &cancelled_);
  absl::MutexLock lock(mu_);
  if (!slot.ok() && !absl::IsCancelled(slot.status())) {
    FinishLocked(slot.status(), kReceivePhase);
  } else if (slot.ok() && (done_ || draining_)) {
    staging_layer_manager_.Release(layer_idx);
    slot = absl::CancelledError(
        absl::StrCat("Receive session ", uuid_, " is draining"));
  }
  EndRecvOpLocked();
  return slot;
}

absl::StatusOr<size_t> TransferReceiveSession::LookupStagingSlot(
    size_t layer_idx) const {
  std::optional<size_t> slot = staging_layer_manager_.Lookup(layer_idx);
  if (!slot.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Layer ", layer_idx, " has no leased staging slot"));
  }
  return *slot;
}

absl::Status TransferReceiveSession::OnDataReceived(size_t layer_idx) {
  // |transfer_plan_| is immutable after construction, so only the Done()
  // check takes |mu_|, and only on this misuse path.
  if (transfer_plan_ != nullptr &&
      transfer_plan_->request().pool_groups_size() > 0 && !Done()) {
    return absl::FailedPreconditionError(
        absl::StrCat("pool completion for UUID ", uuid_,
                     " but the receiver was armed on the legacy path"));
  }
  return ExecuteLayerH2d(layer_idx);
}

absl::Status TransferReceiveSession::ExecuteLayerH2d(size_t layer_idx) {
  // Host source of the copy: the layer's own buffer on a full mirror, or its
  // leased buffer when fewer host layer buffers exist than layers (acquired
  // when the layer's bytes were addressed in GetBlockChunks).
  size_t host_layer_idx = layer_idx;
  if (base_->num_host_layers() < base_->num_layers()) {
    ABSL_ASSIGN_OR_RETURN(host_layer_idx, LookupStagingSlot(layer_idx));
  }
  CopySpec copy_spec;
  std::string session_req_id;
  bool trigger_enqueue = false;
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_ || total_blocks_ == 0) {
      staging_layer_manager_.Release(layer_idx);
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
  if (trigger_enqueue && metrics_collector_ != nullptr) {
    metrics_collector_->RecordH2dEnqueue(uuid_);
  }

  LOG(INFO) << "OnLayerReceived (H2D copy start) layer " << layer_idx
            << ": req_id=" << session_req_id << ", uuid=" << uuid_
            << ", numa=" << base_->assigned_numa_node().value_or(-1);

  {
    // The receive may have been finished (deadline, cancellation) between
    // the check above and here. Re-check immediately before issuing so a
    // finished receive starts no copy it does not need. A copy already
    // handed to the device cannot be revoked; its in-flight count keeps the
    // receive -- and its blocks -- owned until the copy ends.
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      staging_layer_manager_.Release(layer_idx);
      EndRecvOpLocked();
      return absl::OkStatus();
    }
  }
  absl::Status injected =
      FaultInjectStatus(hooks::kTransferRecvSessionH2dDispatch);
  auto future_or =
      injected.ok()
          ? base_->H2dSyncDispatch(copy_spec.src_offsets, copy_spec.dst_offsets,
                                   copy_spec.sizes, /*slot_idx=*/std::nullopt,
                                   /*layer_idx=*/layer_idx,
                                   /*host_layer_idx=*/host_layer_idx)
          : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
  if (!future_or.ok()) {
    staging_layer_manager_.Release(layer_idx);
    absl::MutexLock lock(mu_);
    FinishLocked(future_or.status(), kH2dPhase);
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
  future.OnReady([self = shared_from_this(), uuid, numa_node, layer_idx,
                  session_req_id](auto status_or) {
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferRecvSessionH2dComplete);
    if (status_or.ok() && !injected.ok()) {
      status_or = injected;
    }
    absl::Cleanup end_op = [self]() { self->EndOp(); };
    bool all_layers_done = false;
    std::chrono::steady_clock::time_point session_start_time;
    {
      absl::MutexLock lock(self->mu_);
      if (self->done_) {
        LOG(DFATAL) << "H2D callback for retired receive UUID " << uuid;
        return;
      }
      if (status_or.ok()) {
        LOG(INFO) << "OnLayerReceived (H2D copy complete) layer " << layer_idx
                  << ": req_id=" << session_req_id << ", numa=" << numa_node;
        self->num_completed_layers_++;
        if (self->num_completed_layers_ ==
                static_cast<int32_t>(self->base_->num_layers()) &&
            !self->draining_) {
          all_layers_done = true;
          session_start_time = self->start_time_;
          self->FinishLocked();
        }
      } else {
        LOG(ERROR) << "OnLayerReceived (H2D copy failed) layer " << layer_idx
                   << " for req_id: " << session_req_id
                   << ", error: " << status_or.status().ToString();
        self->FinishLocked(status_or.status(), kH2dPhase);
      }
    }
    // The device has the bytes (or never will): the staging buffer is free
    // for the next layer either way. Release after FinishLocked() so a failed
    // H2D shuts down the manager before waking any blocked layer.
    self->staging_layer_manager_.Release(layer_idx);
    if (all_layers_done) {
      RecordTransferDuration(
          DurationMs(session_start_time, std::chrono::steady_clock::now()));
      if (self->metrics_collector_ != nullptr) {
        self->metrics_collector_->RecordH2dComplete(uuid);
      }
      LOG(INFO) << "All layers H2D copy complete: req_id=" << session_req_id;
      if (self->metrics_collector_ != nullptr) {
        self->metrics_collector_->RecordEnd(uuid);
      }
    }
  });

  return absl::OkStatus();
}

}  // namespace tpu_raiden

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

#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
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
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/metrics_collector.h"  // IWYU pragma: keep
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_send_session.h"
#include "tpu_sync/fault_injection/fault_injector.h"
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
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    const std::string& req_id, const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::optional<std::vector<int64_t>>& local_host_block_ids,
    std::chrono::steady_clock::time_point deadline) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
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

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    std::string req_id, int32_t total_blocks,
    std::chrono::steady_clock::time_point deadline, bool acquire_staging) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  return std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
      base, staging_allocator, uuid, std::move(req_id), total_blocks, deadline,
      acquire_staging));
}

absl::StatusOr<std::shared_ptr<TransferReceiveSession>>
TransferReceiveSession::CreateFromActivePlan(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    const ::tpu_sync::rpc::StartTransferRequest& request, uint64_t generation,
    std::chrono::steady_clock::time_point deadline,
    absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>*
        host_block_of) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  auto session = std::shared_ptr<TransferReceiveSession>(
      new TransferReceiveSession(base, staging_allocator, uuid));
  ABSL_RETURN_IF_ERROR(session->InitFromActivePlan(request, generation,
                                                   deadline, host_block_of));
  return session;
}

absl::Status TransferReceiveSession::InitFromActivePlan(
    const ::tpu_sync::rpc::StartTransferRequest& request, uint64_t generation,
    std::chrono::steady_clock::time_point deadline,
    absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>*
        host_block_of) {
  absl::MutexLock lock(mu_);
  if (staging_allocator_->dynamic_host_staging() &&
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
      absl::StatusOr<StagingAllocation> allocated = staging_allocator_->Acquire(
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
}

void TransferReceiveSession::ReleaseStaging() {
  absl::MutexLock lock(mu_);
  ReleaseStagingLocked();
}

void TransferReceiveSession::FinishLocked(const absl::Status& status) {
  if (!status.ok() && status_.ok()) {
    status_ = status;
  }
  if (draining_) return;
  draining_ = true;
  if (in_flight_ == 0 && !done_) {
    ReleaseStagingLocked();
    done_ = true;
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
  if (done_) {
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
  }
  req_spec.uuid = uuid_;
  req_spec.ep_idx = 0;
  req_spec.consumer_data_port = static_cast<uint32_t>(manager.local_data_port_);
  req_spec.consumer_ips = base_->local_ips();

  LOG(INFO) << "StartRead (connecting): req_id=" << session_req_id
            << ", uuid=" << uuid_ << ", numa=" << target_node.value_or(-1);

  // May run on a transport thread, so it must not block.
  auto on_response = [self = shared_from_this(), session_req_id](
                         absl::StatusOr<PullStreamResponseSpec> response) {
    absl::Cleanup end_op = [self]() { self->EndRecvOp(); };
    absl::Status pull_status = absl::OkStatus();
    try {
      absl::Status injected =
          FaultInjectStatus(hooks::kTransferRecvSessionPullReply);
      if (response.ok() && !injected.ok()) {
        response = injected;
      }
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
      pull_status = absl::InternalError(e.what());
      LOG(ERROR) << "Raiden consumer error during Hybrid Bridge StartRead "
                    "connect: "
                 << e.what();
    }

    if (!pull_status.ok()) {
      self->Finish(pull_status);
    }
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
    if (done_ || draining_) {
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
      FinishLocked();
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
    if (done_ || draining_) {
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

  {
    // The receive may have been finished (deadline, cancellation) between
    // the check above and here. Re-check immediately before issuing so a
    // finished receive starts no copy it does not need. A copy already
    // handed to the device cannot be revoked; its in-flight count keeps the
    // receive -- and its blocks -- owned until the copy ends.
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
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
                                   /*layer_idx=*/layer_idx)
          : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
  if (!future_or.ok()) {
    absl::MutexLock lock(mu_);
    FinishLocked(future_or.status());
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
  future.OnReady([self = shared_from_this(), &manager, uuid, numa_node,
                  layer_idx, session_req_id,
                  metrics_collector =
                      manager.metrics_collector_](auto status_or) {
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferRecvSessionH2dComplete);
    if (status_or.ok() && !injected.ok()) {
      status_or = injected;
    }
    absl::Cleanup end_op = [self]() { self->EndRecvOp(); };
    bool all_layers_done = false;
    std::chrono::steady_clock::time_point session_start_time;
    bool should_unregister = false;
    uint64_t generation = 0;
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
        self->FinishLocked(status_or.status());
      }
      if (self->draining_ && self->in_flight_ == 1 &&
          self->unregister_on_settle_) {
        self->unregister_on_settle_ = false;
        generation = self->plan_generation_;
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
    if (should_unregister && !manager.IsShuttingDown()) {
      manager.UnregisterSettledPlan(uuid, generation);
    }
  });

  return absl::OkStatus();
}

}  // namespace tpu_raiden

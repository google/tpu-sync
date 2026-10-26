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
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
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
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/metrics_collector.h"  // IWYU pragma: keep
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/staging_window.h"
#include "tpu_sync/kv_cache/transfer_send_session.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {

namespace {

// The steps of a receive, as named in `TransferDetails::phase`.
constexpr absl::string_view kAdmissionPhase = "admission";
constexpr absl::string_view kConnectPhase = "connect";
constexpr absl::string_view kReceivePhase = "receive";
constexpr absl::string_view kH2dPhase = "h2d";

// TODO(jcgu): Make threshold configurable via flag/user input defaulting to 5s.
constexpr absl::Duration kUnitProgressStallThreshold = absl::Seconds(5);

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
    std::chrono::steady_clock::time_point deadline,
    std::optional<StagingWindowConfig> staging_window_config) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }

  // Window mode is disabled (falls back to exact-k) when caller provides
  // explicit local_host_block_ids, when local_block_ids is empty or has
  // duplicates, or when window_chunks <= 0.
  const bool is_windowed =
      staging_window_config.has_value() &&
      staging_window_config->window_chunks > 0 && !local_block_ids.empty() &&
      (!local_host_block_ids.has_value() || local_host_block_ids->empty()) &&
      !ChunkPlan::HasDuplicates(local_block_ids);
  if (!is_windowed) {
    staging_window_config = StagingWindowConfig{};
  }

  auto session =
      std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
          base, staging_allocator, uuid,
          staging_window_config.value_or(StagingWindowConfig{})));

  if (is_windowed) {
    const int64_t num_shards = base != nullptr ? base->num_shards() : 0;
    const int64_t block_bytes =
        base != nullptr ? static_cast<int64_t>(base->block_bytes(0)) : 0;
    const int32_t C = ResolveEffectiveChunkBlocks(*staging_window_config,
                                                  num_shards, block_bytes);
    const int64_t k = static_cast<int64_t>(local_block_ids.size());
    ABSL_ASSIGN_OR_RETURN(
        const ChunkPlan plan,
        BuildChunkPlan(k, C, staging_window_config->window_chunks,
                       /*reuse_slots=*/true));

    // Lease min(k, W * C) blocks from staging_allocator.
    // In CL-5b, W_eff = W (reuse_slots = true), so plan.ring_blocks == min(k, W
    // * C).
    std::vector<int64_t> ring_block_ids;
    {
      absl::MutexLock lock(session->mu_);
      absl::StatusOr<StagingAllocation> acquired =
          staging_allocator->Acquire(plan.ring_blocks);
      if (!acquired.ok()) {
        const int64_t free_blocks =
            base != nullptr && base->host_block_manager() != nullptr
                ? base->host_block_manager()->num_free_blocks()
                : 0;
        const int64_t cap = staging_allocator->capacity();
        LOG(ERROR) << "StartRead: cannot stage " << plan.ring_blocks
                   << " blocks for req_id=" << req_id
                   << " (dynamic=" << staging_allocator->dynamic_host_staging()
                   << ", free_host_blocks=" << free_blocks
                   << ", free_slots=" << staging_allocator->num_free_slots()
                   << ", max_blocks=" << staging_allocator->max_blocks() << ")";
        return absl::ResourceExhaustedError(absl::StrCat(
            "staging window admission rejected: ", plan.ring_blocks,
            " blocks needed, ", free_blocks, " free (budget ", cap, ")"));
      }
      ring_block_ids.assign(acquired->block_ids().begin(),
                            acquired->block_ids().end());
      session->staging_ = *std::move(acquired);
    }

    ABSL_ASSIGN_OR_RETURN(std::vector<int64_t> expanded_ring,
                          plan.ExpandRing(ring_block_ids));

    std::vector<size_t> remote_order(remote_block_ids.size());
    std::iota(remote_order.begin(), remote_order.end(), 0);
    std::stable_sort(remote_order.begin(), remote_order.end(),
                     [&](size_t a, size_t b) {
                       return remote_block_ids[a] < remote_block_ids[b];
                     });

    std::vector<int64_t> host_block_ids(k);
    for (size_t i = 0; i < k; ++i) {
      host_block_ids[remote_order[i]] = expanded_ring[i];
    }

    CopyPlan load_plan =
        BuildLoadCopyPlan(remote_block_ids, local_block_ids, host_block_ids);
    {
      absl::MutexLock lock(session->mu_);
      session->in_flight_ = 1;
    }
    session->InitFromLoadPlan(req_id, std::move(load_plan), deadline);

    const size_t total_layers = base != nullptr ? base->num_layers() : 0;
    std::vector<std::vector<CopySpec>> chunk_h2d_specs(plan.n_chunks);
    for (int32_t c = 0; c < plan.n_chunks; ++c) {
      ChunkRange range = plan.ChunkRange(c);
      std::vector<int64_t> chunk_host_bids;
      std::vector<int64_t> chunk_chip_bids;
      chunk_host_bids.reserve(range.size());
      chunk_chip_bids.reserve(range.size());
      for (int64_t i = range.begin; i < range.end; ++i) {
        const size_t original_idx = remote_order[i];
        chunk_host_bids.push_back(host_block_ids[original_idx]);
        chunk_chip_bids.push_back(local_block_ids[original_idx]);
      }
      std::vector<size_t> chunk_order(chunk_chip_bids.size());
      std::iota(chunk_order.begin(), chunk_order.end(), 0);
      std::stable_sort(chunk_order.begin(), chunk_order.end(),
                       [&](size_t a, size_t b) {
                         return chunk_chip_bids[a] < chunk_chip_bids[b];
                       });
      std::vector<int64_t> sorted_chip_bids, sorted_host_bids;
      sorted_chip_bids.reserve(chunk_order.size());
      sorted_host_bids.reserve(chunk_order.size());
      for (size_t idx : chunk_order) {
        sorted_chip_bids.push_back(chunk_chip_bids[idx]);
        sorted_host_bids.push_back(chunk_host_bids[idx]);
      }
      CopySpec chunk_spec = TransferSendSession::BuildCoalescedCopySpec(
          sorted_host_bids, sorted_chip_bids);
      chunk_h2d_specs[c] = std::vector<CopySpec>(total_layers, chunk_spec);
    }

    ABSL_RETURN_IF_ERROR(
        session->SetChunkH2dSpecs(plan.n_chunks, std::move(chunk_h2d_specs), C,
                                  staging_window_config->window_chunks, plan));
  } else {
    std::vector<int64_t> host_block_ids;
    if (!session->AllocateStagingForLoad(
            req_id, local_block_ids, local_host_block_ids, &host_block_ids)) {
      return absl::ResourceExhaustedError(
          absl::StrCat("Failed to allocate staging for load req_id=", req_id,
                       ", uuid=", uuid));
    }
    CopyPlan load_plan =
        BuildLoadCopyPlan(remote_block_ids, local_block_ids, host_block_ids);
    {
      absl::MutexLock lock(session->mu_);
      session->in_flight_ = 1;
    }
    session->InitFromLoadPlan(req_id, std::move(load_plan), deadline);
  }

  return session;
}

std::shared_ptr<TransferReceiveSession> TransferReceiveSession::CreateQueued(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    const std::string& req_id, const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::optional<std::vector<int64_t>>& local_host_block_ids,
    std::chrono::steady_clock::time_point deadline,
    std::optional<StagingWindowConfig> staging_window_config) {
  auto session =
      std::shared_ptr<TransferReceiveSession>(new TransferReceiveSession(
          base, staging_allocator, uuid,
          staging_window_config.value_or(StagingWindowConfig{})));
  absl::MutexLock lock(session->mu_);
  session->req_id_ = req_id;
  session->peer_endpoint_ = remote_endpoint;
  session->deadline_ = deadline;
  session->start_time_ = std::chrono::steady_clock::now();
  session->is_queued_ = true;
  session->in_flight_ = 1;
  session->failed_phase_ = kAdmissionPhase;
  session->total_blocks_ = static_cast<int32_t>(remote_block_ids.size());
  session->queued_remote_block_ids_ = remote_block_ids;
  session->queued_local_block_ids_ = local_block_ids;
  session->queued_local_host_block_ids_ = local_host_block_ids;

  const int64_t num_shards = base != nullptr ? base->num_shards() : 0;
  const int64_t block_bytes =
      base != nullptr ? static_cast<int64_t>(base->block_bytes(0)) : 0;
  const int32_t C = ResolveEffectiveChunkBlocks(session->staging_window_config_,
                                                num_shards, block_bytes);
  const int64_t k = static_cast<int64_t>(local_block_ids.size());
  auto chunk_plan_result =
      BuildChunkPlan(k, C, session->staging_window_config_.window_chunks,
                     /*reuse_slots=*/true);
  if (chunk_plan_result.ok()) {
    session->ring_blocks_needed_ = chunk_plan_result->ring_blocks;
  } else {
    session->ring_blocks_needed_ = k;
  }

  if (session->staging_window_config_.admission_timeout_ms > 0) {
    session->admission_deadline_ =
        std::min(deadline,
                 session->start_time_ +
                     std::chrono::milliseconds(
                         session->staging_window_config_.admission_timeout_ms));
  } else {
    session->admission_deadline_ = deadline;
  }
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
  ResetShardProgressLocked();
  units_complete_ = 0;
  h2d_issued_ = 0;
  h2d_ready_ = 0;
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  recv_gen_.assign(total_layers, 0);
  const int32_t w = window_chunks_ > 0 ? window_chunks_ : 1;
  h2d_done_.assign(w, std::vector<int32_t>(total_layers, 0));
  last_gate_log_chunk_per_layer_.assign(total_layers, -1);
  gate_waits_over_1ms_ = 0;
  last_unit_progress_ = absl::Now();
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

void TransferReceiveSession::InitFromLoadPlanLocked(
    const std::string& req_id, CopyPlan load_plan,
    std::chrono::steady_clock::time_point deadline) {
  ABSL_CHECK_EQ(in_flight_, 1);
  req_id_ = req_id;
  deadline_ = deadline;
  start_time_ = std::chrono::steady_clock::now();
  handshake_pending_ = true;
  chip_block_ids_ = load_plan.h2d_local_block_ids;
  total_blocks_ = load_plan.num_blocks;
  ResetShardProgressLocked();
  units_complete_ = 0;
  h2d_issued_ = 0;
  h2d_ready_ = 0;
  recv_gen_.assign(base_ != nullptr ? base_->num_layers() : 0, 0);
  claimed_.assign(base_ != nullptr ? base_->num_layers() : 0, {});
  h2d_copy_ = load_plan.h2d_copy;
  host_to_chip_.clear();
  for (size_t i = 0; i < load_plan.h2d_host_block_ids.size(); ++i) {
    host_to_chip_[load_plan.h2d_host_block_ids[i]] =
        load_plan.h2d_local_block_ids[i];
  }
  ABSL_CHECK_EQ(load_plan.transport_host_block_ids.size(),
                load_plan.requested_local_block_ids.size());
  if (staging_window_config_.window_chunks == 0) {
    absl::flat_hash_set<int64_t> unique_local_bids(
        load_plan.requested_local_block_ids.begin(),
        load_plan.requested_local_block_ids.end());
    ABSL_CHECK_EQ(host_to_chip_.size(), unique_local_bids.size());
  } else {
    absl::flat_hash_set<int64_t> leased_ring(staging_.block_ids().begin(),
                                             staging_.block_ids().end());
    for (int64_t host_id : load_plan.transport_host_block_ids) {
      ABSL_CHECK(leased_ring.contains(host_id));
    }
  }
  load_plan_ = std::move(load_plan);
}

void TransferReceiveSession::InitFromLoadPlan(
    const std::string& req_id, CopyPlan load_plan,
    std::chrono::steady_clock::time_point deadline) {
  absl::MutexLock lock(mu_);
  InitFromLoadPlanLocked(req_id, std::move(load_plan), deadline);
}

void TransferReceiveSession::ReleaseStagingLocked() {
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
  if (is_queued_) {
    is_queued_ = false;
    EndRecvOpLocked();
    return;
  }
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
  if (is_queued_) return kAdmissionPhase;
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

absl::Status TransferReceiveSession::SetChunkH2dSpecsLocked(
    int32_t n_chunks, std::vector<std::vector<CopySpec>> chunk_h2d_specs,
    int32_t chunk_blocks, int32_t window_chunks,
    std::optional<ChunkPlan> chunk_plan) {
  if (units_complete_ > 0 || h2d_issued_ > 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Cannot mutate chunk configuration after units have "
        "started: units_complete=",
        units_complete_, ", h2d_issued=", h2d_issued_, ", uuid=", uuid_));
  }
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  if (!chunk_h2d_specs.empty()) {
    if (chunk_h2d_specs.size() != static_cast<size_t>(n_chunks)) {
      return absl::InvalidArgumentError(
          absl::StrCat("chunk_h2d_specs outer size ", chunk_h2d_specs.size(),
                       " != n_chunks ", n_chunks, " for uuid=", uuid_));
    }
    for (size_t c = 0; c < chunk_h2d_specs.size(); ++c) {
      if (chunk_h2d_specs[c].size() != total_layers) {
        return absl::InvalidArgumentError(absl::StrCat(
            "chunk_h2d_specs[", c, "] size ", chunk_h2d_specs[c].size(),
            " != total_layers ", total_layers, " for uuid=", uuid_));
      }
    }
  }
  n_chunks_ = n_chunks;
  chunk_h2d_specs_ = std::move(chunk_h2d_specs);
  if (chunk_blocks < 0 || window_chunks < 0) {
    return absl::InvalidArgumentError(
        "chunk_blocks and window_chunks must be non-negative");
  }
  chunk_blocks_ = staging_window_config_.chunk_blocks = chunk_blocks;
  window_chunks_ = staging_window_config_.window_chunks = window_chunks;
  chunk_plan_ = std::move(chunk_plan);
  if (base_ != nullptr && recv_gen_.size() != base_->num_layers()) {
    recv_gen_.assign(base_->num_layers(), 0);
  }
  const int32_t w = window_chunks_ > 0 ? window_chunks_ : 1;
  h2d_done_.assign(w, std::vector<int32_t>(total_layers, 0));
  last_gate_log_chunk_per_layer_.assign(total_layers, -1);
  gate_waits_over_1ms_ = 0;
  last_unit_progress_ = absl::Now();
  claimed_.assign(total_layers, {});
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::SetChunkH2dSpecs(
    int32_t n_chunks, std::vector<std::vector<CopySpec>> chunk_h2d_specs,
    int32_t chunk_blocks, int32_t window_chunks,
    std::optional<ChunkPlan> chunk_plan) {
  if (n_chunks < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("n_chunks must be >= 1, got ", n_chunks));
  }
  absl::MutexLock lock(mu_);
  return SetChunkH2dSpecsLocked(n_chunks, std::move(chunk_h2d_specs),
                                chunk_blocks, window_chunks,
                                std::move(chunk_plan));
}

absl::Status TransferReceiveSession::SeatFromQueue(
    KVCacheManagerWithTransfer& manager) {
  std::string remote_endpoint;
  {
    absl::MutexLock lock(mu_);
    if (draining_ || done_) {
      return absl::CancelledError("session cancelled while queued");
    }
    if (!is_queued_) {
      return absl::FailedPreconditionError("session is not queued");
    }
    if (staging_allocator_ == nullptr) {
      return absl::FailedPreconditionError("staging allocator is null");
    }

    absl::StatusOr<StagingAllocation> acquired =
        staging_allocator_->Acquire(ring_blocks_needed_);
    if (!acquired.ok()) {
      if (!absl::IsResourceExhausted(acquired.status())) {
        FinishLocked(acquired.status(), kAdmissionPhase);
      }
      return acquired.status();
    }

    staging_ = *std::move(acquired);
    std::vector<int64_t> ring_block_ids(staging_.block_ids().begin(),
                                        staging_.block_ids().end());

    const int64_t num_shards = base_ != nullptr ? base_->num_shards() : 0;
    const int64_t block_bytes =
        base_ != nullptr ? static_cast<int64_t>(base_->block_bytes(0)) : 0;
    const int32_t C = ResolveEffectiveChunkBlocks(staging_window_config_,
                                                  num_shards, block_bytes);
    const int64_t k = static_cast<int64_t>(queued_local_block_ids_.size());
    auto chunk_plan_result =
        BuildChunkPlan(k, C, staging_window_config_.window_chunks,
                       /*reuse_slots=*/true);
    if (!chunk_plan_result.ok()) {
      FinishLocked(chunk_plan_result.status(), kAdmissionPhase);
      return chunk_plan_result.status();
    }
    const ChunkPlan plan = *std::move(chunk_plan_result);

    auto expanded_ring_result = plan.ExpandRing(ring_block_ids);
    if (!expanded_ring_result.ok()) {
      FinishLocked(expanded_ring_result.status(), kAdmissionPhase);
      return expanded_ring_result.status();
    }
    std::vector<int64_t> expanded_ring = *std::move(expanded_ring_result);

    std::vector<int64_t> remote_bids = queued_remote_block_ids_;
    std::vector<size_t> remote_order(remote_bids.size());
    std::iota(remote_order.begin(), remote_order.end(), 0);
    std::stable_sort(remote_order.begin(), remote_order.end(),
                     [&remote_bids](size_t a, size_t b) {
                       return remote_bids[a] < remote_bids[b];
                     });

    std::vector<int64_t> host_block_ids(k);
    for (size_t i = 0; i < k; ++i) {
      host_block_ids[remote_order[i]] = expanded_ring[i];
    }

    CopyPlan load_plan = BuildLoadCopyPlan(
        queued_remote_block_ids_, queued_local_block_ids_, host_block_ids);

    InitFromLoadPlanLocked(req_id_, std::move(load_plan), deadline_);

    const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
    std::vector<std::vector<CopySpec>> chunk_h2d_specs(plan.n_chunks);
    for (int32_t c = 0; c < plan.n_chunks; ++c) {
      ChunkRange range = plan.ChunkRange(c);
      std::vector<int64_t> chunk_host_bids;
      std::vector<int64_t> chunk_chip_bids;
      chunk_host_bids.reserve(range.size());
      chunk_chip_bids.reserve(range.size());
      for (int64_t i = range.begin; i < range.end; ++i) {
        const size_t original_idx = remote_order[i];
        chunk_host_bids.push_back(host_block_ids[original_idx]);
        chunk_chip_bids.push_back(queued_local_block_ids_[original_idx]);
      }
      std::vector<size_t> chunk_order(chunk_chip_bids.size());
      std::iota(chunk_order.begin(), chunk_order.end(), 0);
      std::stable_sort(chunk_order.begin(), chunk_order.end(),
                       [&](size_t a, size_t b) {
                         return chunk_chip_bids[a] < chunk_chip_bids[b];
                       });
      std::vector<int64_t> sorted_chip_bids, sorted_host_bids;
      sorted_chip_bids.reserve(chunk_order.size());
      sorted_host_bids.reserve(chunk_order.size());
      for (size_t idx : chunk_order) {
        sorted_chip_bids.push_back(chunk_chip_bids[idx]);
        sorted_host_bids.push_back(chunk_host_bids[idx]);
      }
      CopySpec chunk_spec = TransferSendSession::BuildCoalescedCopySpec(
          sorted_host_bids, sorted_chip_bids);
      chunk_h2d_specs[c] = std::vector<CopySpec>(total_layers, chunk_spec);
    }

    absl::Status status =
        SetChunkH2dSpecsLocked(plan.n_chunks, std::move(chunk_h2d_specs), C,
                               staging_window_config_.window_chunks, plan);
    if (!status.ok()) {
      FinishLocked(status, kAdmissionPhase);
      if (in_flight_ > 0) {
        EndRecvOpLocked();
      }
      return status;
    }

    is_queued_ = false;

    queued_remote_block_ids_.clear();
    queued_local_block_ids_.clear();
    queued_local_host_block_ids_.reset();
    remote_endpoint = peer_endpoint_;
  }
  ExecutePullRequest(manager, remote_endpoint);
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::DowngradeToExactK() {
  absl::MutexLock lock(mu_);
  if (draining_ || done_) {
    return absl::FailedPreconditionError("session is already draining or done");
  }
  if (units_complete_ > 0 || h2d_issued_ > 0) {
    return absl::FailedPreconditionError(
        "cannot downgrade active session with units in flight");
  }
  if (window_chunks_ == 0 && chunk_blocks_ == 0 && n_chunks_ == 1) {
    return absl::OkStatus();
  }
  ReleaseStagingLocked();

  const auto& local_bids = load_plan_.requested_local_block_ids;
  absl::flat_hash_set<int64_t> unique_bids(local_bids.begin(),
                                           local_bids.end());
  const int64_t k = static_cast<int64_t>(unique_bids.size());
  if (k > 0) {
    if (staging_allocator_ == nullptr) {
      return absl::FailedPreconditionError("staging_allocator_ is null");
    }
    absl::StatusOr<StagingAllocation> acquired = staging_allocator_->Acquire(k);
    if (!acquired.ok()) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Cannot downgrade session %v to exact-k (%d blocks): %s", uuid_, k,
          acquired.status().message()));
    }
    staging_ = *std::move(acquired);
  }

  absl::Span<const int> staged_blocks = staging_.block_ids();
  absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
      local_to_host;
  size_t host_block_idx = 0;
  std::vector<int64_t> host_block_ids;
  host_block_ids.reserve(local_bids.size());
  for (int64_t local_bid : local_bids) {
    auto [it, inserted] = local_to_host.try_emplace(local_bid, 0);
    if (inserted) it->second = staged_blocks[host_block_idx++];
    host_block_ids.push_back(it->second);
  }

  load_plan_ = BuildLoadCopyPlan(load_plan_.requested_remote_block_ids,
                                 local_bids, host_block_ids);
  chip_block_ids_ = load_plan_.h2d_local_block_ids;
  total_blocks_ = load_plan_.num_blocks;
  h2d_copy_ = load_plan_.h2d_copy;
  host_to_chip_.clear();
  for (size_t i = 0; i < load_plan_.h2d_host_block_ids.size(); ++i) {
    host_to_chip_[load_plan_.h2d_host_block_ids[i]] =
        load_plan_.h2d_local_block_ids[i];
  }
  ABSL_CHECK_EQ(host_to_chip_.size(),
                load_plan_.transport_host_block_ids.size());
  chunk_blocks_ = window_chunks_ = 0;
  staging_window_config_.chunk_blocks = staging_window_config_.window_chunks =
      0;
  n_chunks_ = 1;
  chunk_h2d_specs_.clear();
  chunk_plan_ = std::nullopt;
  claimed_.clear();
  h2d_done_.clear();
  last_gate_log_chunk_per_layer_.clear();
  gate_waits_over_1ms_ = 0;
  last_unit_progress_ = absl::Now();
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::AwaitIncomingPushSlot(
    int32_t layer_idx, absl::Span<const int64_t> dst_block_ids,
    absl::Span<const int32_t> stream_shards, absl::Time deadline) {
  absl::MutexLock lock(mu_);
  if (window_chunks_ <= 0 || !chunk_plan_.has_value()) {
    return absl::OkStatus();
  }
  if (draining_ || done_) {
    return status_.ok() ? absl::CancelledError("receive session draining")
                        : status_;
  }
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= total_layers) {
    absl::Status status = absl::FailedPreconditionError(
        absl::StrCat("layer_idx ", layer_idx, " out of range [0, ",
                     total_layers, ") for uuid ", uuid_));
    FinishLocked(status, "gate");
    return status;
  }
  const int32_t c = recv_gen_[layer_idx];
  if (c < 0 || c >= n_chunks_) {
    absl::Status status = absl::FailedPreconditionError(absl::StrCat(
        "push for uuid ", uuid_, " layer ", layer_idx, " arrives at unit ", c,
        " >= n_chunks ", n_chunks_, "; peer window mismatch"));
    FinishLocked(status, "gate");
    return status;
  }
  RAIDEN_TRACE_FN("Recv::Gate", [&]() {
    return absl::StrCat("uuid=", uuid_, " l=", layer_idx, " c=", c);
  });

  const ChunkRange range = chunk_plan_->ChunkRange(c);
  const int64_t C_c = range.size();
  const int32_t s = chunk_plan_->SlotOf(c);

  if (dst_block_ids.empty() ||
      dst_block_ids.size() > static_cast<size_t>(C_c) ||
      stream_shards.empty()) {
    absl::Status status = absl::FailedPreconditionError(absl::StrCat(
        "push for uuid ", uuid_, " layer ", layer_idx,
        " re-claims staging (position, shard) pairs already admitted to unit (",
        c, ",", layer_idx, ") / outside slot ", s, " / lists ",
        dst_block_ids.size(), " > ", C_c, " positions; peer window mismatch"));
    FinishLocked(status, "gate");
    return status;
  }

  // Pre-validate that all dst_ids map to valid positions in chunk c's slot.
  std::vector<int32_t> positions;
  positions.reserve(dst_block_ids.size());
  for (int64_t dst_id : dst_block_ids) {
    int32_t found_pos = -1;
    for (int64_t p = 0; p < C_c; ++p) {
      if (load_plan_.transport_host_block_ids[range.begin + p] == dst_id) {
        found_pos = static_cast<int32_t>(p);
        break;
      }
    }
    if (found_pos < 0) {
      absl::Status status = absl::FailedPreconditionError(
          absl::StrCat("push for uuid ", uuid_, " layer ", layer_idx,
                       " re-claims staging (position, shard) pairs already "
                       "admitted to unit (",
                       c, ",", layer_idx, ") / outside slot ", s, " / lists ",
                       dst_block_ids.size(), " > ", C_c,
                       " positions; peer window mismatch"));
      FinishLocked(status, "gate");
      return status;
    }
    positions.push_back(found_pos);
  }

  const size_t num_shards = base_ != nullptr ? base_->num_shards() : 0;
  absl::flat_hash_set<int64_t> new_claims;
  new_claims.reserve(positions.size() * stream_shards.size());
  for (int32_t p : positions) {
    for (int32_t sh : stream_shards) {
      if (sh < 0 || (num_shards > 0 && static_cast<size_t>(sh) >= num_shards)) {
        absl::Status status = absl::FailedPreconditionError(
            absl::StrCat("push for uuid ", uuid_, " layer ", layer_idx,
                         " invalid shard ", sh, "; peer window mismatch"));
        FinishLocked(status, "gate");
        return status;
      }
      const int64_t key =
          (static_cast<int64_t>(p) << 32) | static_cast<uint32_t>(sh);
      if (!new_claims.insert(key).second ||
          (static_cast<size_t>(layer_idx) < claimed_.size() &&
           claimed_[layer_idx].contains(key))) {
        absl::Status status = absl::FailedPreconditionError(
            absl::StrCat("push for uuid ", uuid_, " layer ", layer_idx,
                         " re-claims staging (position, shard) pairs already "
                         "admitted to unit (",
                         c, ",", layer_idx, ") / outside slot ", s, " / lists ",
                         dst_block_ids.size(), " > ", C_c,
                         " positions; peer window mismatch"));
        FinishLocked(status, "gate");
        return status;
      }
    }
  }

  if (static_cast<size_t>(layer_idx) >= claimed_.size()) {
    claimed_.resize(layer_idx + 1);
  }
  for (int64_t key : new_claims) {
    claimed_[layer_idx].insert(key);
  }

  const int32_t needed_done = (window_chunks_ > 0) ? (c / window_chunks_) : 0;
  if (static_cast<size_t>(s) >= h2d_done_.size() ||
      static_cast<size_t>(layer_idx) >= h2d_done_[s].size()) {
    absl::Status status = absl::InternalError(absl::StrFormat(
        "h2d_done_ dimensions mismatch: slot %d, layer %d (dims %dx%d)", s,
        layer_idx, h2d_done_.size(),
        h2d_done_.empty() ? 0 : h2d_done_[0].size()));
    FinishLocked(status, "gate");
    return status;
  }

  if (h2d_done_[s][layer_idx] < needed_done) {
    RAIDEN_TRACE_FN("Recv::GateWait", [&]() {
      return absl::StrCat("uuid=", uuid_, " l=", layer_idx, " c=", c, " s=", s);
    });
    const absl::Time wait_start = absl::Now();
    if (last_unit_progress_ != absl::InfinitePast() &&
        wait_start - last_unit_progress_ > kUnitProgressStallThreshold) {
      LOG(WARNING) << "Windowed receive session uuid=" << uuid_
                   << " made no unit progress for "
                   << absl::FormatDuration(wait_start - last_unit_progress_)
                   << " (layer=" << layer_idx << " chunk=" << c << " slot=" << s
                   << ")";
    }

    absl::Time session_deadline = absl::InfiniteFuture();
    if (deadline_ != std::chrono::steady_clock::time_point::max() &&
        deadline_ != std::chrono::steady_clock::time_point{}) {
      const auto now_steady = std::chrono::steady_clock::now();
      if (deadline_ <= now_steady) {
        session_deadline = absl::Now();
      } else {
        session_deadline =
            absl::Now() +
            absl::Nanoseconds(
                std::chrono::duration_cast<std::chrono::nanoseconds>(deadline_ -
                                                                     now_steady)
                    .count());
      }
    }
    const absl::Time effective_deadline =
        (deadline == absl::InfiniteFuture())
            ? session_deadline
            : std::min(deadline, session_deadline);

    auto can_proceed = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return draining_ || h2d_done_[s][layer_idx] >= needed_done;
    };
    mu_.AwaitWithDeadline(absl::Condition(&can_proceed), effective_deadline);

    const absl::Duration waited = absl::Now() - wait_start;
    total_gate_wait_us_.fetch_add(absl::ToInt64Microseconds(waited),
                                  std::memory_order_relaxed);
    if (waited > absl::Milliseconds(1)) {
      ++gate_waits_over_1ms_;
      if (static_cast<size_t>(layer_idx) <
              last_gate_log_chunk_per_layer_.size() &&
          last_gate_log_chunk_per_layer_[layer_idx] < c) {
        last_gate_log_chunk_per_layer_[layer_idx] = c;
        LOG(INFO) << "staging window gate wait > 1ms for uuid=" << uuid_
                  << " layer=" << layer_idx << " chunk=" << c << " slot=" << s
                  << " waited=" << absl::ToInt64Microseconds(waited) << " us";
      }
    }

    if (draining_ || done_ || !status_.ok()) {
      absl::Status status =
          status_.ok() ? absl::CancelledError(
                             "receive session draining during gate wait")
                       : status_;
      FinishLocked(status, "gate");
      return status;
    }

    if (h2d_done_[s][layer_idx] < needed_done) {
      absl::Status status = absl::DeadlineExceededError(absl::StrFormat(
          "staging window gate timed out for uuid=%v layer=%d chunk=%d slot=%d "
          "(h2d_done=%d < needed=%d)",
          uuid_, layer_idx, c, s, h2d_done_[s][layer_idx], needed_done));
      FinishLocked(status, "gate");
      return status;
    }
  } else {
    if (draining_ || done_ || !status_.ok()) {
      absl::Status status =
          status_.ok() ? absl::CancelledError(
                             "receive session draining during gate wait")
                       : status_;
      FinishLocked(status, "gate");
      return status;
    }
  }

  return absl::OkStatus();
}

int32_t TransferReceiveSession::ExpectedPushesPerLayer() const {
  absl::MutexLock lock(mu_);
  return n_chunks_;
}

bool TransferReceiveSession::AllUnitsAndH2dDoneLocked() const {
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  const int32_t total_units = static_cast<int32_t>(total_layers) * n_chunks_;
  return units_complete_ == total_units && h2d_issued_ == total_units &&
         h2d_ready_ == h2d_issued_;
}

bool TransferReceiveSession::IsReadyToComplete() const {
  absl::MutexLock lock(mu_);
  if (handshake_pending_ || in_flight_ > 0) return false;
  return AllUnitsAndH2dDoneLocked();
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
    return AllUnitsAndH2dDoneLocked();
  }
  return false;
}

bool TransferReceiveSession::RecordNetworkCompleteLocked(
    MetricsCollector* absl_nullable metrics, bool all_complete,
    std::string* session_req_id,
    std::chrono::steady_clock::time_point* session_start_time) {
  network_completed_ = true;
  *session_req_id = req_id_;
  if (metrics != nullptr) {
    metrics->RecordLastPacket(uuid_);
  }
  if (!all_complete || !AllUnitsAndH2dDoneLocked()) {
    return false;
  }
  *session_start_time = start_time_;
  if (metrics != nullptr) {
    metrics->RecordEnd(uuid_);
  }
  FinishLocked();
  return true;
}

void TransferReceiveSession::ExecutePullRequest(
    KVCacheManagerWithTransfer& manager, const std::string& remote_endpoint) {
  RAIDEN_TRACE_FN("Recv::Handshake",
                  [&]() { return absl::StrCat("uuid=", uuid_); });
  std::optional<int> target_node = base_->assigned_numa_node();
  {
    absl::MutexLock lock(mu_);
    peer_endpoint_ = remote_endpoint;
    handshake_pending_ = true;
  }

  LOG(INFO) << "StartRead (connecting): initiating pull request to peer "
            << remote_endpoint << " for uuid " << uuid_;

  base_->push_pool()->Schedule(target_node, [self = shared_from_this(),
                                             &manager, remote_endpoint,
                                             target_node]() mutable {
    auto end_handshake = [self](const absl::Status& status) {
      {
        absl::MutexLock lock(self->mu_);
        if (!status.ok()) {
          LOG(ERROR) << "StartRead handshake failed for uuid " << self->uuid_
                     << ": " << status;
          self->FinishLocked(status, kConnectPhase);
        } else {
          VLOG(1) << "StartRead handshake completed for uuid " << self->uuid_;
        }
        self->handshake_pending_ = false;
        self->EndRecvOpLocked();
      }
    };

    bool is_windowed = false;
    int32_t chunk_blocks = 0, window_chunks = 0;
    int64_t total_blocks = 0;
    bool assume = false;
    absl::Duration timeout = absl::Seconds(manager.control_timeout_s_);
    bool early_exit = false;
    {
      absl::MutexLock lock(self->mu_);
      if (self->draining_ || self->done_) {
        early_exit = true;
      } else {
        is_windowed = self->window_chunks_ > 0;
        chunk_blocks = self->chunk_blocks_;
        window_chunks = self->window_chunks_;
        total_blocks = self->total_blocks_;
        assume = self->staging_window_config_.assume_peer_support;
      }
    }
    if (early_exit) {
      end_handshake(absl::OkStatus());
      return;
    }

    if (is_windowed && !assume) {
      auto cached = manager.GetCachedCapabilities(remote_endpoint);
      if (!cached) {
        auto probe = manager.ProbeCapabilities(remote_endpoint, timeout);
        if (!probe.ok()) return end_handshake(probe.status());
        manager.SetCachedCapabilities(remote_endpoint, *probe);
        cached = *probe;
      }
      const int64_t ring_blocks = std::min<int64_t>(
          total_blocks, static_cast<int64_t>(window_chunks) * chunk_blocks);
      if (!cached->supported || !cached->serve_staging_window ||
          cached->max_ring_blocks < ring_blocks ||
          cached->max_ring_blocks <= 0) {
        if (absl::Status s = self->DowngradeToExactK(); !s.ok()) {
          return end_handshake(s);
        }
      }
    }

    auto make_req = [self, &manager](int32_t c_blocks, int32_t w_chunks) {
      PullStreamRequestSpec req;
      {
        absl::MutexLock lock(self->mu_);
        req.src_block_ids = self->load_plan_.producer_remote_block_ids;
        req.dst_block_ids = self->load_plan_.transport_host_block_ids;
      }
      req.uuid = self->uuid_;
      req.consumer_data_port = static_cast<uint32_t>(manager.local_data_port_);
      req.consumer_ips = self->base_->local_ips();
      req.layer_host_addrs = self->base_->LayerHostAddrs(self->uuid_);
      req.chunk_blocks = c_blocks;
      req.window_chunks = w_chunks;
      return req;
    };

    auto send_pull = [&manager, remote_endpoint](
                         const PullStreamRequestSpec& req, auto on_reply) {
      try {
        FaultInjectThrow(hooks::kTransferRecvSessionPullRequest);
      } catch (const std::exception& e) {
        return on_reply(absl::InternalError(e.what()));
      }
      manager.control_backend_
          ->SendPullRequest(remote_endpoint, req,
                            absl::Seconds(manager.control_timeout_s_))
          .OnReady(std::move(on_reply));
    };

    auto status_from_reply = [](const absl::StatusOr<PullStreamResponseSpec>&
                                    resp) {
      if (!resp.ok()) {
        return absl::Status(
            resp.status().code(),
            absl::StrCat("control pull request: ", resp.status().message()));
      }
      return resp->status != 0
                 ? absl::FailedPreconditionError(absl::StrCat(
                       "Remote producer rejected Hybrid Bridge read request: ",
                       resp->message))
                 : absl::OkStatus();
    };

    auto on_terminal = [self, end_handshake, status_from_reply](
                           absl::StatusOr<PullStreamResponseSpec> resp) {
      RAIDEN_TRACE_FN("Recv::FinalAck",
                      [&]() { return absl::StrCat("uuid=", self->uuid_); });
      if (auto s = FaultInjectStatus(hooks::kTransferRecvSessionPullReply);
          resp.ok() && !s.ok()) {
        resp = s;
      }
      end_handshake(status_from_reply(resp));
    };

    int32_t sent_c = 0, sent_w = 0;
    {
      absl::MutexLock lock(self->mu_);
      if (self->window_chunks_ > 0) {
        sent_c = self->chunk_blocks_;
        sent_w = self->window_chunks_;
      }
    }
    if (sent_w == 0) {
      send_pull(make_req(0, 0), std::move(on_terminal));
      return;
    }

    auto on_window_reply =
        [self, &manager, remote_endpoint, target_node, sent_c, sent_w,
         end_handshake, on_terminal, send_pull, make_req, status_from_reply](
            absl::StatusOr<PullStreamResponseSpec> resp) mutable {
          RAIDEN_TRACE_FN("Recv::WindowAck",
                          [&]() { return absl::StrCat("uuid=", self->uuid_); });
          absl::Status injected =
              FaultInjectStatus(hooks::kTransferRecvSessionPullReply);
          if (resp.ok() && !injected.ok()) resp = injected;

          if (resp.ok() && resp->status != 0) {
            if (resp->staging_window_rejected_reason.has_value() ||
                ParseStagingWindowRejectedReason(resp->message).has_value()) {
              manager.InvalidateCachedCapabilities(remote_endpoint);
              self->base_->push_pool()->Schedule(
                  target_node, [self, end_handshake, on_terminal, send_pull,
                                make_req]() mutable {
                    if (absl::Status s = self->DowngradeToExactK(); !s.ok()) {
                      return end_handshake(s);
                    }
                    send_pull(make_req(0, 0), std::move(on_terminal));
                  });
              return;
            }
          }

          if (resp.ok() && resp->status == 0) {
            auto echo =
                resp->staging_window_ack.has_value()
                    ? std::make_pair(resp->staging_window_ack->chunk_blocks,
                                     resp->staging_window_ack->window_chunks)
                    : ParseStagingWindowAck(resp->message)
                          .value_or(std::make_pair(-1, -1));
            if (echo.first != sent_c || echo.second != sent_w) {
              manager.InvalidateCachedCapabilities(remote_endpoint);
              return end_handshake(absl::FailedPreconditionError(absl::StrCat(
                  "peer ignored staging window for endpoint=", remote_endpoint,
                  ", uuid=", self->uuid_, ", sent=(", sent_c, ", ", sent_w,
                  "), echo=(", echo.first, ", ", echo.second, ")")));
            }
            return end_handshake(absl::OkStatus());
          }

          end_handshake(status_from_reply(resp));
        };

    send_pull(make_req(sent_c, sent_w), std::move(on_window_reply));
  });
}

absl::Status TransferReceiveSession::OnBlockShardsReceived(
    KVCacheManagerWithTransfer& manager, const std::vector<int>& block_ids,
    absl::Span<const int> shard_ids) {
  const uint64_t uuid = uuid_;
  RAIDEN_TRACE_FN("Recv::Payload", [&]() {
    return absl::StrCat("uuid=", uuid, " blocks=", block_ids.size());
  });
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
    for (const int sh : shard_ids) {
      if (sh < 0 || sh >= static_cast<int>(blocks_received_per_shard_.size())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "shard ", sh, " out of range for uuid ", uuid_,
            " (num_shards=", blocks_received_per_shard_.size(), ")"));
      }
    }
    all_complete = RecordBlockShardsReceivedLocked(
        block_ids, shard_ids, &first_packet, &network_just_completed);
    MetricsCollector* const metrics = manager.metrics_collector_.get();
    if (first_packet && metrics != nullptr) {
      metrics->RecordFirstPacket(uuid_);
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
        metrics, all_complete, &session_req_id, &session_start_time);
  }

  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(session_start_time, std::chrono::steady_clock::now()));
    LOG(INFO) << "OnBlockShardsReceived (Network + H2D complete): req_id="
              << session_req_id << ", uuid=" << uuid << ", numa=" << numa_node;
  }
  return absl::OkStatus();
}

absl::Status TransferReceiveSession::ExecuteLayerH2d(
    KVCacheManagerWithTransfer& manager, size_t layer_idx) {
  CopySpec copy_spec;
  std::string session_req_id;
  bool trigger_enqueue = false;
  int32_t chunk_idx = -1;
  bool is_multi_shard_windowed = false;
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return absl::OkStatus();
    }
    is_multi_shard_windowed =
        window_chunks_ > 0 && base_ != nullptr && base_->num_shards() >= 4;
    const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
    if (layer_idx >= total_layers) {
      return absl::InvalidArgumentError(absl::StrCat(
          "layer_idx ", layer_idx, " >= num_layers ", total_layers));
    }
    const int32_t c = recv_gen_[layer_idx];
    if (c >= n_chunks_) {
      absl::Status err = absl::FailedPreconditionError(absl::StrCat(
          "OnLayerReceived: layer ", layer_idx, " already received all ",
          n_chunks_, " units for uuid=", uuid_));
      FinishLocked(err, kReceivePhase);
      return err;
    }
    chunk_idx = c;
    recv_gen_[layer_idx] = c + 1;
    ++units_complete_;
    if (layer_idx < claimed_.size()) {
      claimed_[layer_idx].clear();
    }
    last_unit_progress_ = absl::Now();

    if (!chunk_h2d_specs_.empty()) {
      if (c < 0 || static_cast<size_t>(c) >= chunk_h2d_specs_.size() ||
          layer_idx >= chunk_h2d_specs_[c].size()) {
        absl::Status err = absl::FailedPreconditionError(absl::StrCat(
            "ExecuteLayerH2d: chunk index ", c, " or layer_idx ", layer_idx,
            " out of bounds for chunk_h2d_specs_ (size ",
            chunk_h2d_specs_.size(), ") for uuid=", uuid_));
        FinishLocked(err, kReceivePhase);
        return err;
      }
      copy_spec = chunk_h2d_specs_[c][layer_idx];
    } else {
      copy_spec = h2d_copy_;
    }

    ++in_flight_;
    session_req_id = req_id_;
    if (!h2d_started_) {
      h2d_started_ = true;
      trigger_enqueue = true;
    }
  }
  if (trigger_enqueue && manager.metrics_collector_) {
    manager.metrics_collector_->RecordH2dEnqueue(uuid_);
  }

  RAIDEN_TRACE_FN("Recv::UnitComplete->H2dDispatch", [&]() {
    return absl::StrCat("uuid=", session_req_id, " l=", layer_idx,
                        " c=", chunk_idx);
  });

  VLOG(1) << "OnLayerReceived (H2D copy start) layer " << layer_idx
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
  const uint64_t uuid = uuid_;
  absl::Status injected =
      FaultInjectStatus(hooks::kTransferRecvSessionH2dDispatch);

  auto attach_completion = [self = shared_from_this(), &manager, uuid,
                            layer_idx, chunk_idx,
                            session_req_id](raiden::PjRtCopyFuture future,
                                            absl::Time dma_start) {
    const int numa_node = self->base_ != nullptr
                              ? self->base_->assigned_numa_node().value_or(-1)
                              : -1;
    future.OnReady([self, &manager, uuid, numa_node, layer_idx, chunk_idx,
                    session_req_id, dma_start,
                    metrics_collector =
                        manager.metrics_collector_](auto copy_status) {
      self->total_h2d_dma_us_.fetch_add(
          absl::ToInt64Microseconds(absl::Now() - dma_start),
          std::memory_order_relaxed);
      absl::Status injected =
          FaultInjectStatus(hooks::kTransferRecvSessionH2dComplete);
      if (copy_status.ok() && !injected.ok()) {
        copy_status = injected;
      }
      absl::Cleanup end_op = [self]() { self->EndRecvOp(); };
      bool all_layers_done = false;
      std::chrono::steady_clock::time_point session_start_time;
      bool should_unregister = false;
      uint64_t generation = 0;
      int64_t gate_waits_over_1ms = 0;
      {
        absl::MutexLock lock(self->mu_);
        if (self->done_) {
          LOG(DFATAL) << "H2D callback for retired receive UUID " << uuid;
          return;
        }
        if (copy_status.ok()) {
          VLOG(1) << "OnLayerReceived (H2D copy complete) layer " << layer_idx
                  << ": req_id=" << session_req_id << ", numa=" << numa_node;
          ++self->h2d_ready_;
          if (self->window_chunks_ > 0 && !self->h2d_done_.empty()) {
            const int32_t s = chunk_idx % self->window_chunks_;
            if (static_cast<size_t>(s) < self->h2d_done_.size() &&
                layer_idx < self->h2d_done_[s].size()) {
              ++self->h2d_done_[s][layer_idx];
            }
          }
          self->last_unit_progress_ = absl::Now();
          if (self->AllUnitsAndH2dDoneLocked() && !self->draining_) {
            all_layers_done = true;
            session_start_time = self->start_time_;
            gate_waits_over_1ms = self->gate_waits_over_1ms_;
            self->FinishLocked();
          }
        } else {
          LOG(ERROR) << "OnLayerReceived (H2D copy failed) layer " << layer_idx
                     << " for req_id: " << session_req_id
                     << ", error: " << copy_status.status().ToString();
          self->FinishLocked(copy_status.status(), kH2dPhase);
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
        LOG(INFO) << "All layers H2D copy complete: req_id=" << session_req_id
                  << " total_gate_wait_us=" << self->total_gate_wait_us_.load()
                  << " gate_waits_over_1ms=" << gate_waits_over_1ms
                  << " total_h2d_dispatch_us="
                  << self->total_h2d_dispatch_us_.load()
                  << " total_h2d_dma_us=" << self->total_h2d_dma_us_.load();
        if (metrics_collector) {
          metrics_collector->RecordEnd(uuid);
        }
      }
      if (should_unregister && !manager.IsShuttingDown()) {
        manager.UnregisterSettledPlan(uuid, generation);
      }
    });
  };

  if (is_multi_shard_windowed) {
    {
      absl::MutexLock lock(mu_);
      ++h2d_issued_;
    }
    const auto target_node = base_->assigned_numa_node();
    base_->push_pool()->Schedule(
        target_node, [self = shared_from_this(), &manager, uuid, layer_idx,
                      copy_spec, injected, attach_completion]() mutable {
          {
            absl::MutexLock lock(self->mu_);
            if (self->done_ || self->draining_) {
              bool should_unregister = false;
              uint64_t generation = 0;
              if (self->draining_ && self->in_flight_ == 1 &&
                  self->unregister_on_settle_) {
                self->unregister_on_settle_ = false;
                generation = self->plan_generation_;
                should_unregister = true;
              }
              self->EndRecvOpLocked();
              if (should_unregister && !manager.IsShuttingDown()) {
                manager.UnregisterSettledPlan(uuid, generation);
              }
              return;
            }
          }
          RAIDEN_TRACE_FN("Recv::H2D", [&]() {
            return absl::StrCat("uuid=", uuid, " l=", layer_idx);
          });
          const absl::Time dispatch_start = absl::Now();
          auto future_or =
              injected.ok() ? self->base_->H2dSyncDispatch(
                                  copy_spec.src_offsets, copy_spec.dst_offsets,
                                  copy_spec.sizes, /*slot_idx=*/std::nullopt,
                                  /*layer_idx=*/layer_idx)
                            : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
          self->total_h2d_dispatch_us_.fetch_add(
              absl::ToInt64Microseconds(absl::Now() - dispatch_start),
              std::memory_order_relaxed);
          if (!future_or.ok()) {
            bool should_unregister = false;
            uint64_t generation = 0;
            {
              absl::MutexLock lock(self->mu_);
              self->FinishLocked(future_or.status(), kH2dPhase);
              if (self->draining_ && self->in_flight_ == 1 &&
                  self->unregister_on_settle_) {
                self->unregister_on_settle_ = false;
                generation = self->plan_generation_;
                should_unregister = true;
              }
              self->EndRecvOpLocked();
            }
            if (should_unregister && !manager.IsShuttingDown()) {
              manager.UnregisterSettledPlan(uuid, generation);
            }
            return;
          }
          attach_completion(*std::move(future_or), absl::Now());
        });
    return absl::OkStatus();
  }

  RAIDEN_TRACE_FN("Recv::H2D", [&]() {
    return absl::StrCat("uuid=", uuid, " l=", layer_idx);
  });
  const absl::Time dispatch_start = absl::Now();
  auto future_or =
      injected.ok()
          ? base_->H2dSyncDispatch(copy_spec.src_offsets, copy_spec.dst_offsets,
                                   copy_spec.sizes, /*slot_idx=*/std::nullopt,
                                   /*layer_idx=*/layer_idx)
          : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
  total_h2d_dispatch_us_.fetch_add(
      absl::ToInt64Microseconds(absl::Now() - dispatch_start),
      std::memory_order_relaxed);
  if (!future_or.ok()) {
    bool should_unregister = false;
    uint64_t generation = 0;
    {
      absl::MutexLock lock(mu_);
      FinishLocked(future_or.status(), kH2dPhase);
      if (draining_ && in_flight_ == 1 && unregister_on_settle_) {
        unregister_on_settle_ = false;
        generation = plan_generation_;
        should_unregister = true;
      }
      EndRecvOpLocked();
    }
    if (should_unregister && !manager.IsShuttingDown()) {
      manager.UnregisterSettledPlan(uuid, generation);
    }
    return future_or.status();
  }

  {
    absl::MutexLock lock(mu_);
    ++h2d_issued_;
  }
  attach_completion(*std::move(future_or), absl::Now());
  return absl::OkStatus();
}

}  // namespace tpu_raiden

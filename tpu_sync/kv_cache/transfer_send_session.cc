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

#include "tpu_sync/kv_cache/transfer_send_session.h"

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/data_connection_policy.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/transport/peer_endpoint.h"

namespace tpu_raiden {

namespace {

// The steps of a send, as named in `TransferDetails::phase`.
constexpr absl::string_view kAwaitPullPhase = "await_pull";
constexpr absl::string_view kStagingPhase = "staging";
constexpr absl::string_view kD2hPhase = "d2h";
constexpr absl::string_view kPushPhase = "push";

}  // namespace

absl::StatusOr<std::shared_ptr<TransferSendSession>>
TransferSendSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    absl::Span<const int64_t> block_ids,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start,
    BlockLocation source) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  ABSL_ASSIGN_OR_RETURN(const DataConnectionReuse data_connection_reuse,
                        ResolveDataConnectionReuse());
  auto session = std::shared_ptr<TransferSendSession>(new TransferSendSession(
      base, staging_allocator, uuid, deadline, register_start,
      data_connection_reuse, /*plan=*/nullptr, source));
  std::optional<DuplicateBlock> duplicate_block =
      session->PopulateRegisteredBlocks(block_ids);
  if (duplicate_block.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "NotifyForRead rejected duplicate block ", duplicate_block->block_id,
        " for uuid=", uuid, ": positions ", duplicate_block->first_index,
        " and ", duplicate_block->duplicate_index, " of ", block_ids.size(),
        " blocks"));
  }
  return session;
}

absl::StatusOr<std::shared_ptr<TransferSendSession>>
TransferSendSession::CreateFromActivePlan(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator, uint64_t uuid,
    const ::tpu_sync::rpc::StartTransferRequest& request) {
  StagingAllocation staging;
  absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
      host_block_of;
  if (staging_allocator != nullptr &&
      staging_allocator->dynamic_host_staging() &&
      request.pool_groups_size() == 0) {
    std::vector<int64_t> device_blocks;
    absl::flat_hash_set<int64_t> seen;
    for (const auto& [src_shard, schedule] : request.shard_push_schedules()) {
      for (const auto& e : schedule.entries()) {
        int64_t id = e.src_block_id();
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
  ABSL_ASSIGN_OR_RETURN(const DataConnectionReuse data_connection_reuse,
                        ResolveDataConnectionReuse());
  ABSL_ASSIGN_OR_RETURN(
      TransferPlan plan,
      TransferPlan::Create(request, /*is_sender=*/true, base->num_pools(),
                           base->has_explicit_pools(), base->pools(),
                           std::move(host_block_of)));
  const auto now = std::chrono::steady_clock::now();
  auto session = std::shared_ptr<TransferSendSession>(new TransferSendSession(
      base, staging_allocator, uuid,
      std::chrono::steady_clock::time_point::max(), now, data_connection_reuse,
      std::make_shared<const TransferPlan>(std::move(plan))));
  {
    absl::MutexLock lock(session->mu_);
    session->staging_ = std::move(staging);
  }
  return session;
}

TransferSendSession::TransferSendSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, uint64_t uuid,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start,
    DataConnectionReuse data_connection_reuse,
    std::shared_ptr<const TransferPlan> plan, BlockLocation source)
    : base_(base),
      staging_allocator_(staging_allocator),
      uuid_(uuid),
      deadline_(deadline),
      register_start_(register_start),
      data_connection_reuse_(data_connection_reuse),
      transfer_plan_(std::move(plan)),
      source_(source),
      staging_layer_manager_(base != nullptr ? base->num_host_layers() : 0) {}

std::optional<TransferSendSession::DuplicateBlock>
TransferSendSession::PopulateRegisteredBlocks(
    absl::Span<const int64_t> block_ids) {
  absl::MutexLock lock(mu_);
  registered_block_set_.clear();
  for (size_t i = 0; i < block_ids.size(); ++i) {
    if (registered_block_set_.insert(block_ids[i]).second) continue;
    size_t first_index = 0;
    while (block_ids[first_index] != block_ids[i]) ++first_index;
    return DuplicateBlock{
        .block_id = block_ids[i],
        .first_index = first_index,
        .duplicate_index = i,
    };
  }
  return std::nullopt;
}

void TransferSendSession::ValidateRequestedBlocksLocked(
    const std::vector<int64_t>& requested_block_ids) const {
  if (requested_block_ids.empty()) {
    throw std::invalid_argument(
        "pull stream requested no blocks; use ack-only path");
  }
  absl::flat_hash_set<int64_t> seen;
  for (int64_t block_id : requested_block_ids) {
    if (registered_block_set_.find(block_id) == registered_block_set_.end()) {
      throw std::invalid_argument(
          "pull stream requested block not registered by producer");
    }
    if (!seen.insert(block_id).second) {
      throw std::invalid_argument(
          "pull stream requested duplicate producer block id");
    }
  }
}

void TransferSendSession::ValidateAndBeginPull(
    const std::vector<int64_t>& requested_block_ids,
    std::chrono::steady_clock::time_point now) {
  absl::MutexLock lock(mu_);
  if (deadline_ <= now) {
    throw std::runtime_error(
        absl::StrCat("read registration for uuid ", uuid_, " expired"));
  }
  ValidateRequestedBlocksLocked(requested_block_ids);
  if (pull_started_) {
    throw std::runtime_error(
        absl::StrCat("pull already started for uuid ", uuid_));
  }
  pull_started_ = true;
}

tsl::Future<> TransferSendSession::GetDataReadyFuture(
    size_t major_dimension_idx) {
  absl::MutexLock lock(mu_);
  if (major_dimension_idx >= d2h_layer_futures_.size()) {
    return tsl::Future<>(absl::OkStatus());
  }
  return d2h_layer_futures_[major_dimension_idx].future;
}

void TransferSendSession::ReleaseStagingBlocksLocked() {
  staging_layer_manager_.ReleaseAll();
  staging_.Reset();
  d2h_unit_futures_.clear();
  d2h_layer_futures_.clear();
}

void TransferSendSession::ReleaseStagingBlocks() {
  absl::MutexLock lock(mu_);
  ReleaseStagingBlocksLocked();
}

absl::StatusOr<size_t> TransferSendSession::AcquireStagingSlot(
    size_t layer_idx) {
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
  }
  absl::StatusOr<size_t> slot =
      staging_layer_manager_.Acquire(layer_idx, deadline_, &cancelled_);
  if (!slot.ok()) {
    // A timeout (or a manager with no buffers) fails the send while it waits
    // for host staging; a cancellation means the send is already draining.
    if (!absl::IsCancelled(slot.status())) {
      absl::MutexLock lock(mu_);
      FinishLocked(slot.status(), kStagingPhase);
    }
    return slot.status();
  }
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      staging_layer_manager_.Release(layer_idx);
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
  }
  return *slot;
}

absl::StatusOr<size_t> TransferSendSession::LookupStagingSlot(
    size_t layer_idx) const {
  {
    absl::MutexLock lock(mu_);
    if (done_ || draining_) {
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
  }
  std::optional<size_t> slot = staging_layer_manager_.Lookup(layer_idx);
  if (!slot.has_value()) {
    // On a full host mirror a layer with no lease is addressed in place.
    if (base_ == nullptr || base_->num_host_layers() >= base_->num_layers()) {
      return absl::NotFoundError("no bounded host staging");
    }
    return absl::FailedPreconditionError(absl::StrCat(
        "No staging slot leased for layer ", layer_idx, " (uuid=", uuid_, ")"));
  }
  return *slot;
}

void TransferSendSession::FinishLocked(const absl::Status& status,
                                       std::optional<absl::string_view> phase) {
  if (draining_ || done_) return;
  status_ = status;
  if (!status.ok()) {
    failed_phase_ = phase.has_value() ? *phase : CurrentPhaseLocked();
  }
  cancelled_.store(true, std::memory_order_relaxed);
  staging_layer_manager_.Shutdown();
  draining_ = true;
  if (in_flight_ == 0) {
    ReleaseStagingBlocksLocked();
    done_ = true;
    completed_at_ = std::chrono::steady_clock::now();
  }
}

void TransferSendSession::Finish(const absl::Status& status) {
  // Settling here, with no op in flight, means no push was ever issued:
  // pushes start in a push_pool task (AdmitPush, or the task that resumes a
  // parked unit), each stays counted until its completion has run, and the
  // completion of the last unit calls Finish() while still counted (a failed
  // copy or push finishes under its own op). A send that pushed therefore
  // settles in EndOp, which releases its data connections first.
  absl::MutexLock lock(mu_);
  FinishLocked(status);
}

absl::Status TransferSendSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

absl::Status TransferSendSession::OnDataReceived(
    size_t major_dimension_idx, absl::Span<const int> minor_dimension_indices,
    absl::Span<const int> shard_ids) {
  return absl::FailedPreconditionError(absl::StrCat(
      "send session does not receive data: layer ", major_dimension_idx,
      " reported received on send session for UUID ", uuid_));
}

TransferDetails TransferSendSession::Details() const {
  absl::MutexLock lock(mu_);
  TransferDetails details = DetailsFromStatus(status_);
  details.completed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             completed_at_.time_since_epoch())
                             .count();
  details.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           completed_at_ - register_start_)
                           .count();
  if (!remote_data_endpoints_.empty()) {
    details.peer =
        absl::StrJoin(remote_data_endpoints_, ",",
                      [](std::string* out, const transport::PeerEndpoint& ep) {
                        absl::StrAppend(out, ep.data);
                      });
  }
  if (!status_.ok()) {
    details.phase = std::string(failed_phase_);
  }
  return details;
}

absl::string_view TransferSendSession::CurrentPhaseLocked() const {
  if (remote_data_endpoints_.empty()) return kAwaitPullPhase;
  // A host-resident send has nothing to stage or copy.
  if (source_ == BlockLocation::kHost) return kPushPhase;
  if (staging_.empty()) return kStagingPhase;
  if (d2h_layer_futures_.size() < base_->num_layers()) return kD2hPhase;
  for (const raiden::PjRtCopyFuture& future : d2h_layer_futures_) {
    if (!future.IsReady()) return kD2hPhase;
  }
  return kPushPhase;
}

void TransferSendSession::EndSendOpLocked() {
  --in_flight_;
  if (draining_ && in_flight_ == 0 && !done_) {
    ReleaseStagingBlocksLocked();
    done_ = true;
    completed_at_ = std::chrono::steady_clock::now();
  }
}

void TransferSendSession::EndOp() {
  // Under kPerTransfer the last op of a draining send closes the idle data
  // connections to the consumer's endpoints before it is counted out: every
  // push has returned its socket by now, and done_ cannot be set meanwhile,
  // so ~KVCacheManagerWithTransfer is still parked in AwaitForDone() and
  // |base_| and its transport are alive. transport_server() cannot contend on
  // server_init_mu_ either: every StopTransportServer() call is in a
  // destructor that runs only after that wait. The check and the decrement
  // share a critical section so a Finish() that starts draining in between
  // cannot settle the send unreleased.
  while (true) {
    std::vector<transport::PeerEndpoint> peers;
    {
      absl::MutexLock lock(mu_);
      const bool release_due =
          data_connection_reuse_ == DataConnectionReuse::kPerTransfer &&
          in_flight_ == 1 && draining_ && !done_ && !connections_released_ &&
          !remote_data_endpoints_.empty();
      if (!release_due) {
        EndSendOpLocked();
        return;
      }
      connections_released_ = true;
      peers = remote_data_endpoints_;
    }
    ReleaseDataConnectionsAfterSend(base_->transport_server(), peers, uuid_);
  }
}

CopySpec TransferSendSession::BuildCoalescedCopySpec(
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  if (src_block_ids.size() != dst_block_ids.size()) {
    throw std::invalid_argument(
        "src and dst block lists must have same length");
  }
  CopySpec spec;
  if (src_block_ids.empty()) {
    return spec;
  }
  const int64_t n = static_cast<int64_t>(src_block_ids.size());
  spec.src_offsets.reserve(n);
  spec.dst_offsets.reserve(n);
  spec.sizes.reserve(n);

  for (int64_t start = 0; start < n;) {
    int64_t end = start + 1;
    while (end < n && src_block_ids[end] == src_block_ids[end - 1] + 1 &&
           dst_block_ids[end] == dst_block_ids[end - 1] + 1) {
      ++end;
    }
    const int64_t run_size = end - start;
    spec.src_offsets.push_back(src_block_ids[start]);
    spec.dst_offsets.push_back(dst_block_ids[start]);
    spec.sizes.push_back(run_size);
    start = end;
  }
  return spec;
}

std::vector<std::pair<size_t, size_t>> TransferSendSession::BlockGroupBounds(
    size_t num_blocks, size_t block_bytes, size_t group_bytes) {
  if (num_blocks == 0) return {};
  size_t blocks_per_group = num_blocks;
  if (group_bytes > 0 && block_bytes > 0) {
    blocks_per_group = std::max<size_t>(1, group_bytes / block_bytes);
  }
  const size_t count = (num_blocks + blocks_per_group - 1) / blocks_per_group;
  const size_t size = num_blocks / count;
  const size_t larger = num_blocks % count;
  std::vector<std::pair<size_t, size_t>> bounds;
  bounds.reserve(count);
  size_t begin = 0;
  for (size_t g = 0; g < count; ++g) {
    const size_t end = begin + size + (g < larger ? 1 : 0);
    bounds.emplace_back(begin, end);
    begin = end;
  }
  return bounds;
}

void TransferSendSession::PlanUnitsLocked(size_t num_blocks) {
  const size_t total_layers = base_->num_layers();
  const kv_cache::KVCacheManagerBase::PushGroups push_groups =
      base_->push_groups();
  const size_t group_bytes = push_groups.group_bytes;
  const size_t parallelism =
      static_cast<size_t>(std::max(1, base_->parallelism()));
  units_.clear();
  layer_streams_.assign(total_layers, std::nullopt);
  unpushed_units_.assign(total_layers, 0);
  for (size_t l = 0; l < total_layers; ++l) {
    const std::vector<std::pair<size_t, size_t>> groups = BlockGroupBounds(
        num_blocks, base_->block_bytes(l) * base_->num_shards(), group_bytes);
    // A layer that moves in several pushes declares in each the streams of
    // all of them, so the consumer completes it only on the last.
    if (groups.size() > 1) {
      size_t streams = 0;
      for (const auto& [begin, end] : groups) {
        streams += std::min(parallelism, end - begin);
      }
      layer_streams_[l] = streams;
    }
    unpushed_units_[l] = groups.size();
    for (const auto& [begin, end] : groups) {
      units_.push_back(PushUnit{.layer = l, .begin = begin, .end = end});
    }
  }
  push_window_ = push_groups.window;
}

absl::Status TransferSendSession::AcquireStagingWithRetry(int64_t num_blocks) {
  while (true) {
    {
      absl::MutexLock lock(mu_);
      if (draining_ || done_) {
        return !status_.ok()
                   ? status_
                   : absl::CancelledError(
                         "Send session cancelled while waiting for staging");
      }
    }
    const auto now = std::chrono::steady_clock::now();
    const auto slice_deadline =
        std::min(deadline_, now + std::chrono::milliseconds(1));
    absl::StatusOr<StagingAllocation> acquired =
        staging_allocator_->AcquireWithTimeout(num_blocks, slice_deadline);
    if (acquired.ok()) {
      absl::MutexLock lock(mu_);
      if (draining_ || done_) {
        return !status_.ok()
                   ? status_
                   : absl::CancelledError(
                         "Send session cancelled while waiting for staging");
      }
      staging_ = *std::move(acquired);
      return absl::OkStatus();
    }
    if (!absl::IsResourceExhausted(acquired.status())) {
      return acquired.status();
    }
    if (std::chrono::steady_clock::now() >= deadline_) {
      return acquired.status();
    }
  }
}

void TransferSendSession::StartPush(
    const std::vector<transport::PeerEndpoint>& remote_data_endpoints,
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::StartPush", [&]() {
    return absl::StrCat("uuid=", uuid_, " blocks=", src_block_ids.size());
  });
  if (src_block_ids.empty() || src_block_ids.size() != dst_block_ids.size() ||
      remote_data_endpoints.empty()) {
    Finish(absl::InvalidArgumentError(
        absl::StrCat("Invalid StartPush arguments for uuid=", uuid_)));
    return;
  }
  {
    absl::MutexLock lock(mu_);
    remote_data_endpoints_ = remote_data_endpoints;
  }

  const size_t total_layers = base_->num_layers();
  const size_t num_blocks = src_block_ids.size();
  if (source_ == BlockLocation::kHost) {
    // The registered blocks already are host blocks: give every unit an
    // already-OK D2H future, so SendNextUnit and the transport's data-ready
    // wait (GetDataReadyFuture) proceed without staging or D2H copies.
    size_t num_units = 0;
    {
      absl::MutexLock lock(mu_);
      if (draining_ || done_) {
        return;
      }
      if (total_layers == 0) {
        FinishLocked(absl::OkStatus());
        return;
      }
      src_ints_.assign(src_block_ids.begin(), src_block_ids.end());
      dst_ints_.assign(dst_block_ids.begin(), dst_block_ids.end());
      PlanUnitsLocked(num_blocks);
      num_units = units_.size();
      const raiden::PjRtCopyFuture ready(tsl::Future<>(absl::OkStatus()),
                                         raiden::BufferHolders{});
      d2h_unit_futures_.assign(num_units, ready);
      d2h_layer_futures_.assign(total_layers, ready);
    }
    remaining_h2h_units_.store(num_units, std::memory_order_relaxed);
    SendNextUnit(0);
    return;
  }

  // Stage the producer's device KV into a host slot (slot.block_ids) and send
  // those host blocks to the consumer, keeping host offsets within the staging
  // pool. Writing D2H straight to host[src_block_id] overflows the host buffer
  // once a device block id exceeds num_host_blocks.
  absl::Status acquire_status =
      AcquireStagingWithRetry(static_cast<int64_t>(num_blocks));
  if (!acquire_status.ok()) {
    Finish(acquire_status);
    return;
  }

  std::vector<int64_t> host_block_ids;
  std::vector<PushUnit> units;
  std::vector<tsl::Promise<>> d2h_promises;
  {
    absl::MutexLock lock(mu_);
    if (draining_ || done_) {
      return;
    }
    if (total_layers == 0) {
      FinishLocked(absl::OkStatus());
      return;
    }
    absl::Span<const int> blocks = staging_.block_ids();
    host_block_ids.assign(blocks.begin(), blocks.begin() + num_blocks);
    PlanUnitsLocked(num_blocks);
    units = units_;
    // Every unit's future exists before any copy is issued, so the H2H chain
    // and readiness callbacks can wait on a unit whose copy has not started;
    // a layer's future joins those of its units.
    d2h_promises.reserve(units.size());
    d2h_unit_futures_.reserve(units.size());
    for (size_t u = 0; u < units.size(); ++u) {
      auto [promise, future] = tsl::MakePromise<>();
      d2h_promises.push_back(std::move(promise));
      d2h_unit_futures_.emplace_back(std::move(future),
                                     raiden::BufferHolders{});
    }
    d2h_layer_futures_.reserve(total_layers);
    for (size_t u = 0; u < units.size();) {
      size_t next = u + 1;
      while (next < units.size() && units[next].layer == units[u].layer) {
        ++next;
      }
      d2h_layer_futures_.push_back(
          next == u + 1 ? d2h_unit_futures_[u]
                        : raiden::JoinPjRtCopyFutures(absl::MakeConstSpan(
                              d2h_unit_futures_.data() + u, next - u)));
      u = next;
    }
    src_ints_.assign(host_block_ids.begin(), host_block_ids.end());
    dst_ints_.assign(dst_block_ids.begin(), dst_block_ids.end());
  }

  // Coalesce contiguous (device,host) block runs into a few large copies. With
  // per-block segments (sizes=1) a contiguous KV range becomes n device copies
  // that flood the command queue with small ops and serialize against prefill
  // GEMMs on the shared TensorCore; coalescing collapses a contiguous range to
  // one copy, matching the pre-Hybrid-Push pull path. A unit that is a block
  // group copies its own range.
  CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);
  remaining_h2h_units_.store(units.size(), std::memory_order_relaxed);

  // 1. Arm the H2H chain on unit 0's future before any copy is issued, so
  // pushes start as copies complete while this (pull worker) thread is still
  // issuing the later units. When fewer host layer buffers exist than layers
  // the loop below blocks in AcquireStagingSlot until an earlier layer's last
  // push frees its buffer, which is only possible because the chain is
  // running.
  SendNextUnit(0);

  // 2. Issue D2H copies unit by unit, each into the host layer buffer its
  // layer leases (layer l's own buffer on a full mirror; the units of a layer
  // share its lease). Each copy is counted against the send before it starts
  // and released by its own completion, so the staging it writes stays owned
  // for as long as it runs.
  //
  // Units whose copy is never issued (the send failed or expired first)
  // resolve with the send's status so the H2H chain and the readiness
  // callbacks waiting on them settle instead of hanging on an unset promise.
  size_t next_unit = 0;
  absl::Cleanup fail_unissued_units = [this, &d2h_promises, &next_unit]() {
    if (next_unit >= d2h_promises.size()) return;
    absl::Status status;
    {
      absl::MutexLock lock(mu_);
      status = status_.ok() ? absl::CancelledError("Send session is draining")
                            : status_;
    }
    for (size_t u = next_unit; u < d2h_promises.size(); ++u) {
      d2h_promises[u].Set(status);
    }
  };
  for (size_t u = 0; u < units.size(); ++u) {
    const PushUnit& unit = units[u];
    CopySpec group_copy;
    const bool whole_layer = unit.begin == 0 && unit.end == num_blocks;
    if (!whole_layer) {
      group_copy = BuildCoalescedCopySpec(
          std::vector<int64_t>(src_block_ids.begin() + unit.begin,
                               src_block_ids.begin() + unit.end),
          std::vector<int64_t>(host_block_ids.begin() + unit.begin,
                               host_block_ids.begin() + unit.end));
    }
    const CopySpec& unit_copy = whole_layer ? d2h_copy : group_copy;
    absl::StatusOr<size_t> host_layer_idx = AcquireStagingSlot(unit.layer);
    if (!host_layer_idx.ok()) {
      Finish(host_layer_idx.status());
      return;
    }
    {
      absl::MutexLock session_lock(mu_);
      // The send expired or failed while its copies were being issued:
      // what was issued drains, nothing more starts.
      if (draining_) {
        staging_layer_manager_.Release(unit.layer);
        return;
      }
      ++in_flight_;
    }
    LOG(INFO) << "StartPush (D2H start) layer " << unit.layer << " blocks ["
              << unit.begin << ", " << unit.end << "): uuid=" << uuid_
              << ", numa=" << base_->assigned_numa_node().value_or(-1);
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferSendSessionD2hDispatch);
    absl::StatusOr<raiden::PjRtCopyFuture> future =
        injected.ok()
            ? base_->D2hSyncDispatch(unit_copy.src_offsets,
                                     unit_copy.dst_offsets, unit_copy.sizes,
                                     /*slot_idx=*/std::nullopt,
                                     /*layer_idx=*/unit.layer, *host_layer_idx)
            : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
    if (!future.ok()) {
      // A copy that cannot even be issued fails the transfer; the worker
      // thread has no caller for an exception to reach. Its promise, and
      // those of the units after it, are failed by `fail_unissued_units`.
      LOG(ERROR) << "StartPush: failed to issue D2H for layer " << unit.layer
                 << " blocks [" << unit.begin << ", " << unit.end
                 << "): " << future.status();
      staging_layer_manager_.Release(unit.layer);
      {
        absl::MutexLock lock(mu_);
        FinishLocked(future.status());
      }
      // Earlier units may already have been pushed, so this can be the
      // send's last op: EndOp (unlike EndSendOpLocked) first closes the
      // idle data connections under kPerTransfer.
      EndOp();  // this copy never started
      return;
    }
    raiden::PjRtCopyFuture unit_future = *std::move(future);
    unit_future.OnReady(
        [self = shared_from_this(), promise = std::move(d2h_promises[u])](
            absl::StatusOr<raiden::BufferHolders> status_or) mutable {
          promise.Set(status_or.status());
          self->EndOp();
        });
    next_unit = u + 1;
  }
}

void TransferSendSession::SendNextUnit(size_t u) {
  RAIDEN_TRACE_FN("KVTransfer::SendNextUnit",
                  [&]() { return absl::StrCat("uuid=", uuid_, " unit=", u); });
  std::optional<raiden::PjRtCopyFuture> unit_future;
  size_t layer = 0;
  {
    absl::MutexLock lock(mu_);
    // Past the last unit: the pushes in flight finish the send.
    if (draining_ || u >= d2h_unit_futures_.size()) {
      return;
    }
    unit_future = d2h_unit_futures_[u];
    layer = units_[u].layer;
    ++in_flight_;
  }

  unit_future->OnReady([self = shared_from_this(), u, layer](auto status_or) {
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferSendSessionD2hComplete);
    if (status_or.ok() && !injected.ok()) {
      status_or = injected;
    }
    absl::Cleanup end_op = [self]() { self->EndOp(); };
    if (!status_or.ok()) {
      LOG(ERROR) << "StartPush: D2H copy failed for layer " << layer
                 << " (unit " << u
                 << "), status: " << status_or.status().ToString();
      {
        absl::MutexLock lock(self->mu_);
        self->FinishLocked(status_or.status(), kD2hPhase);
      }
      self->staging_layer_manager_.Release(layer);
      return;
    }
    {
      absl::MutexLock session_lock(self->mu_);
      // The send expired or failed while the copy ran: nothing is pushed.
      if (self->draining_) {
        self->staging_layer_manager_.Release(layer);
        return;
      }
    }

    std::move(end_op).Cancel();
    self->base_->push_pool()->Schedule([self, u]() { self->AdmitPush(u); });
  });
}

void TransferSendSession::AdmitPush(size_t u) {
  absl::Cleanup end_op = [this]() { EndOp(); };
  {
    absl::MutexLock lock(mu_);
    if (draining_) {
      staging_layer_manager_.Release(units_[u].layer);
      return;
    }
    // The window is full: the push that frees it issues this unit and chains
    // the next. The pushes in flight keep the send open meanwhile.
    if (push_window_ > 0 && pushes_in_flight_ >= push_window_) {
      parked_unit_ = u;
      return;
    }
    ++pushes_in_flight_;
  }
  IssuePush(u);
  // Queue the next unit's push without waiting for this one to finish.
  SendNextUnit(u + 1);
}

void TransferSendSession::IssuePush(size_t u) {
  std::vector<transport::PeerEndpoint> remote_data_endpoints;
  std::vector<int> src_ints;
  std::vector<int> dst_ints;
  PushUnit unit;
  std::optional<size_t> layer_streams;
  {
    absl::MutexLock lock(mu_);
    unit = units_[u];
    if (draining_) {
      --pushes_in_flight_;  // admitted, never issued
      staging_layer_manager_.Release(unit.layer);
      return;
    }
    remote_data_endpoints = remote_data_endpoints_;
    src_ints.assign(src_ints_.begin() + unit.begin,
                    src_ints_.begin() + unit.end);
    dst_ints.assign(dst_ints_.begin() + unit.begin,
                    dst_ints_.begin() + unit.end);
    layer_streams = layer_streams_[unit.layer];
    ++in_flight_;
  }
  LOG(INFO) << "StartPush (H2H start layer " << unit.layer << " blocks ["
            << unit.begin << ", " << unit.end << ")): uuid=" << uuid_
            << ", numa=" << base_->assigned_numa_node().value_or(-1);
  base_->H2hWriteDirectAsync(
      remote_data_endpoints, src_ints, dst_ints, uuid_,
      static_cast<int>(unit.layer), layer_streams,
      [self = shared_from_this(),
       unit](absl::StatusOr<std::vector<int>> push_res) {
        absl::Cleanup end_op = [self]() { self->EndOp(); };
        bool release_layer = false;
        std::optional<size_t> resume;
        {
          absl::MutexLock lock(self->mu_);
          --self->pushes_in_flight_;
          if (!push_res.ok()) {
            self->FinishLocked(push_res.status(), kPushPhase);
          } else {
            // The layer's last push has read it out of its host buffer:
            // return the buffer so a later layer can D2H into it.
            release_layer = --self->unpushed_units_[unit.layer] == 0;
            if (self->parked_unit_.has_value() && !self->draining_) {
              resume = *self->parked_unit_;
              self->parked_unit_.reset();
              ++self->pushes_in_flight_;
              ++self->in_flight_;  // held by the resume task below
            }
          }
        }
        if (!push_res.ok()) {
          LOG(ERROR) << "H2hWrite failed for layer " << unit.layer
                     << " blocks [" << unit.begin << ", " << unit.end
                     << "): " << push_res.status().ToString();
          self->staging_layer_manager_.Release(unit.layer);
          return;
        }
        if (release_layer) {
          self->staging_layer_manager_.Release(unit.layer);
        }

        LOG(INFO) << "StartPush (H2H complete layer " << unit.layer
                  << " blocks [" << unit.begin << ", " << unit.end
                  << ")): uuid=" << self->uuid_ << ", numa="
                  << self->base_->assigned_numa_node().value_or(-1);

        const bool last = self->remaining_h2h_units_.fetch_sub(1) == 1;
        if (last) {
          LOG(INFO) << "StartPush (All H2H complete): uuid=" << self->uuid_;
          self->Finish();
        }
        if (resume.has_value()) {
          self->base_->push_pool()->Schedule([self, next = *resume]() {
            absl::Cleanup end_op = [self]() { self->EndOp(); };
            self->IssuePush(next);
            self->SendNextUnit(next + 1);
          });
        }
      });
}

}  // namespace tpu_raiden

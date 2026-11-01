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
#include "tpu_sync/transport/block_transport_delegate.h"

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
    StagingBlockAllocator* absl_nullable staging_allocator, std::string req_id,
    uint64_t uuid, absl::Span<const int64_t> block_ids,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  ABSL_ASSIGN_OR_RETURN(const DataConnectionReuse data_connection_reuse,
                        ResolveDataConnectionReuse());
  auto session = std::shared_ptr<TransferSendSession>(
      new TransferSendSession(base, staging_allocator, std::move(req_id), uuid,
                              deadline, register_start, data_connection_reuse));
  std::optional<DuplicateBlock> duplicate_block =
      session->PopulateRegisteredBlocks(block_ids);
  if (duplicate_block.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "NotifyForRead rejected duplicate block ", duplicate_block->block_id,
        " for req_id=", session->req_id(), ", uuid=", uuid, ": positions ",
        duplicate_block->first_index, " and ", duplicate_block->duplicate_index,
        " of ", block_ids.size(), " blocks"));
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
  std::string req_id = request.req_id().empty()
                           ? absl::StrCat("resharded_transfer_", uuid)
                           : request.req_id();
  auto session = std::shared_ptr<TransferSendSession>(new TransferSendSession(
      base, staging_allocator, std::move(req_id), uuid,
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
    StagingBlockAllocator* staging_allocator, std::string req_id, uint64_t uuid,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start,
    DataConnectionReuse data_connection_reuse,
    std::shared_ptr<const TransferPlan> plan)
    : base_(base),
      staging_allocator_(staging_allocator),
      req_id_(std::move(req_id)),
      uuid_(uuid),
      deadline_(deadline),
      register_start_(register_start),
      data_connection_reuse_(data_connection_reuse),
      transfer_plan_(std::move(plan)),
      staging_layer_manager_(base != nullptr ? base->num_host_layers() : 0) {}

std::optional<TransferSendSession::DuplicateBlock>
TransferSendSession::PopulateRegisteredBlocks(
    absl::Span<const int64_t> block_ids) {
  absl::MutexLock lock(mu_);
  registered_block_ids_.assign(block_ids.begin(), block_ids.end());
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

bool TransferSendSession::OwnsBlockWithReadyFuture(int block_id,
                                                   size_t layer_idx) const {
  absl::MutexLock lock(mu_);
  return registered_block_set_.find(block_id) != registered_block_set_.end() &&
         layer_idx < d2h_layer_futures_.size();
}

void TransferSendSession::RegisterLayerReadinessCallback(
    size_t layer_idx,
    transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
  std::optional<raiden::PjRtCopyFuture> future;
  {
    absl::MutexLock lock(mu_);
    if (layer_idx < d2h_layer_futures_.size()) {
      future = d2h_layer_futures_[layer_idx];
    }
  }
  if (!future.has_value()) {
    cb(absl::OkStatus());
    return;
  }
  future->OnReady(
      [cb = std::move(cb)](auto status_or) { cb(status_or.status()); });
}

void TransferSendSession::ReleaseStagingBlocksLocked() {
  staging_layer_manager_.ReleaseAll();
  staging_.Reset();
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
  // pushes start in SendNextLayer's push_pool task, each stays counted until
  // its completion has run, and the completion of the last layer calls
  // Finish() while still counted (a failed copy or push finishes under its
  // own op). A send that pushed therefore settles in EndOp, which
  // releases its data connections first.
  absl::MutexLock lock(mu_);
  FinishLocked(status);
}

absl::Status TransferSendSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

absl::Status TransferSendSession::OnDataReceived(size_t layer_idx) {
  return absl::FailedPreconditionError(
      absl::StrCat("send session does not receive data: layer ", layer_idx,
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
    details.peer = absl::StrJoin(remote_data_endpoints_, ",");
  }
  if (!status_.ok()) {
    details.phase = std::string(failed_phase_);
  }
  return details;
}

absl::string_view TransferSendSession::CurrentPhaseLocked() const {
  if (remote_data_endpoints_.empty()) return kAwaitPullPhase;
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
    std::vector<std::string> peers;
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
    const std::vector<std::string>& remote_data_endpoints,
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::StartPush", [&]() {
    return absl::StrCat("uuid=", uuid_, " blocks=", src_block_ids.size());
  });
  // Stage the producer's device KV into a host slot (slot.block_ids) and send
  // those host blocks to the consumer, keeping host offsets within the staging
  // pool. Writing D2H straight to host[src_block_id] overflows the host buffer
  // once a device block id exceeds num_host_blocks.
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

  absl::Status acquire_status =
      AcquireStagingWithRetry(static_cast<int64_t>(src_block_ids.size()));
  if (!acquire_status.ok()) {
    Finish(acquire_status);
    return;
  }

  const size_t total_layers = base_->num_layers();
  std::vector<int64_t> host_block_ids;
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
    host_block_ids.assign(blocks.begin(),
                          blocks.begin() + src_block_ids.size());
    // Every layer's future exists before any copy is issued, so the H2H chain
    // and readiness callbacks can wait on a layer whose copy has not started.
    d2h_promises.reserve(total_layers);
    d2h_layer_futures_.reserve(total_layers);
    for (size_t l = 0; l < total_layers; ++l) {
      auto [promise, future] = tsl::MakePromise<>();
      d2h_promises.push_back(std::move(promise));
      d2h_layer_futures_.emplace_back(std::move(future),
                                      raiden::BufferHolders{});
    }
    src_ints_.assign(host_block_ids.begin(), host_block_ids.end());
    dst_ints_.assign(dst_block_ids.begin(), dst_block_ids.end());
  }

  // Coalesce contiguous (device,host) block runs into a few large copies. With
  // per-block segments (sizes=1) a contiguous KV range becomes n device copies
  // that flood the command queue with small ops and serialize against prefill
  // GEMMs on the shared TensorCore; coalescing collapses a contiguous range to
  // one copy, matching the pre-Hybrid-Push pull path.
  CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);
  remaining_h2h_layers_.store(total_layers, std::memory_order_relaxed);

  // 1. Arm the H2H chain on layer 0's future before any copy is issued, so
  // pushes start as copies complete while this (pull worker) thread is still
  // issuing the later layers. When fewer host layer buffers exist than layers
  // the loop below blocks in AcquireStagingSlot until an earlier layer's push
  // frees its buffer, which is only possible because the chain is running.
  SendNextLayer(0);

  // 2. Issue D2H copies layer-by-layer, each into the host layer buffer its
  // layer leases (layer l's own buffer on a full mirror). Each copy is
  // counted against the send before it starts and released by its own
  // completion, so the staging it writes stays owned for as long as it runs.
  //
  // Layers whose copy is never issued (the send failed or expired first)
  // resolve with the send's status so the H2H chain and the readiness
  // callbacks waiting on them settle instead of hanging on an unset promise.
  size_t next_layer = 0;
  absl::Cleanup fail_unissued_layers = [this, &d2h_promises, &next_layer,
                                        total_layers]() {
    if (next_layer >= total_layers) return;
    absl::Status status;
    {
      absl::MutexLock lock(mu_);
      status = status_.ok() ? absl::CancelledError("Send session is draining")
                            : status_;
    }
    for (size_t l = next_layer; l < total_layers; ++l) {
      d2h_promises[l].Set(status);
    }
  };
  for (size_t l = 0; l < total_layers; ++l) {
    absl::StatusOr<size_t> host_layer_idx = AcquireStagingSlot(l);
    if (!host_layer_idx.ok()) {
      Finish(host_layer_idx.status());
      return;
    }
    {
      absl::MutexLock session_lock(mu_);
      // The send expired or failed while its copies were being issued:
      // what was issued drains, nothing more starts.
      if (draining_) {
        staging_layer_manager_.Release(l);
        return;
      }
      ++in_flight_;
    }
    LOG(INFO) << "StartPush (D2H start) layer " << l << ": uuid=" << uuid_
              << ", numa=" << base_->assigned_numa_node().value_or(-1);
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferSendSessionD2hDispatch);
    absl::StatusOr<raiden::PjRtCopyFuture> future =
        injected.ok()
            ? base_->D2hSyncDispatch(d2h_copy.src_offsets, d2h_copy.dst_offsets,
                                     d2h_copy.sizes, /*slot_idx=*/std::nullopt,
                                     /*layer_idx=*/l, *host_layer_idx)
            : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
    if (!future.ok()) {
      // A copy that cannot even be issued fails the transfer; the worker
      // thread has no caller for an exception to reach. Its promise, and
      // those of the layers after it, are failed by `fail_unissued_layers`.
      LOG(ERROR) << "StartPush: failed to issue D2H for layer " << l << ": "
                 << future.status();
      staging_layer_manager_.Release(l);
      {
        absl::MutexLock lock(mu_);
        FinishLocked(future.status());
      }
      // Earlier layers may already have been pushed, so this can be the
      // send's last op: EndOp (unlike EndSendOpLocked) first closes the
      // idle data connections under kPerTransfer.
      EndOp();  // this copy never started
      return;
    }
    raiden::PjRtCopyFuture layer_future = *std::move(future);
    layer_future.OnReady(
        [self = shared_from_this(), promise = std::move(d2h_promises[l])](
            absl::StatusOr<raiden::BufferHolders> status_or) mutable {
          promise.Set(status_or.status());
          self->EndOp();
        });
    next_layer = l + 1;
  }
}

void TransferSendSession::SendNextLayer(size_t l) {
  RAIDEN_TRACE_FN("KVTransfer::SendNextLayer",
                  [&]() { return absl::StrCat("uuid=", uuid_, " layer=", l); });
  if (l >= base_->num_layers()) {
    // Reached end of layer loop. Background asynchronous transfers will clean
    // up when remaining_h2h_layers_ reaches 0.
    return;
  }
  std::optional<raiden::PjRtCopyFuture> layer_future;
  {
    absl::MutexLock lock(mu_);
    if (draining_ || l >= d2h_layer_futures_.size()) {
      return;
    }
    layer_future = d2h_layer_futures_[l];
    ++in_flight_;
  }

  layer_future->OnReady([self = shared_from_this(), l](auto status_or) {
    absl::Status injected =
        FaultInjectStatus(hooks::kTransferSendSessionD2hComplete);
    if (status_or.ok() && !injected.ok()) {
      status_or = injected;
    }
    absl::Cleanup end_op = [self]() { self->EndOp(); };
    if (!status_or.ok()) {
      LOG(ERROR) << "StartPush: D2H copy failed for layer " << l
                 << ", status: " << status_or.status().ToString();
      {
        absl::MutexLock lock(self->mu_);
        self->FinishLocked(status_or.status(), kD2hPhase);
      }
      self->staging_layer_manager_.Release(l);
      return;
    }
    {
      absl::MutexLock session_lock(self->mu_);
      // The send expired or failed while the copy ran: nothing is pushed.
      if (self->draining_) {
        self->staging_layer_manager_.Release(l);
        return;
      }
    }

    std::move(end_op).Cancel();
    self->base_->push_pool()->Schedule([self, l]() {
      absl::Cleanup end_op = [self]() { self->EndOp(); };
      std::vector<std::string> remote_data_endpoints;
      std::vector<int> src_ints;
      std::vector<int> dst_ints;
      {
        absl::MutexLock lock(self->mu_);
        if (self->draining_) {
          self->staging_layer_manager_.Release(l);
          return;
        }
        remote_data_endpoints = self->remote_data_endpoints_;
        src_ints = self->src_ints_;
        dst_ints = self->dst_ints_;
        ++self->in_flight_;
      }
      LOG(INFO) << "StartPush (H2H start layer " << l
                << "): uuid=" << self->uuid_
                << ", numa=" << self->base_->assigned_numa_node().value_or(-1);
      self->base_->H2hWriteDirectAsync(
          remote_data_endpoints, src_ints, dst_ints, self->uuid_,
          static_cast<int>(l),
          [self, l](absl::StatusOr<std::vector<int>> push_res) {
            absl::Cleanup end_op = [self]() { self->EndOp(); };
            if (!push_res.ok()) {
              LOG(ERROR) << "H2hWrite failed for layer " << l << ": "
                         << push_res.status().ToString();
              {
                absl::MutexLock lock(self->mu_);
                self->FinishLocked(push_res.status(), kPushPhase);
              }
              self->staging_layer_manager_.Release(l);
              return;
            }
            // The push has read the layer out of its host buffer: return
            // the buffer so a later layer can D2H into it.
            self->staging_layer_manager_.Release(l);

            LOG(INFO) << "StartPush (H2H complete layer " << l
                      << "): uuid=" << self->uuid_ << ", numa="
                      << self->base_->assigned_numa_node().value_or(-1);

            const bool last = self->remaining_h2h_layers_.fetch_sub(1) == 1;
            if (last) {
              LOG(INFO) << "StartPush (All H2H complete): uuid=" << self->uuid_;
              self->Finish();
            }
          });

      // Immediately queue the next layer's push without waiting for this one to
      // finish.
      self->SendNextLayer(l + 1);
    });
  });
}

}  // namespace tpu_raiden

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
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
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
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/data_connection_policy.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/staging_window.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

namespace {

// The steps of a send, as named in `TransferDetails::phase`.
constexpr absl::string_view kAwaitPullPhase = "await_pull";
constexpr absl::string_view kStagingPhase = "staging";
constexpr absl::string_view kD2hPhase = "d2h";
constexpr absl::string_view kPushPhase = "push";

// TODO(jcgu): Make threshold configurable via flag/user input defaulting to 5s.
constexpr absl::Duration kUnitProgressStallThreshold = absl::Seconds(5);

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

TransferSendSession::TransferSendSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, std::string req_id, uint64_t uuid,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start,
    DataConnectionReuse data_connection_reuse)
    : base_(base),
      staging_allocator_(staging_allocator),
      req_id_(std::move(req_id)),
      uuid_(uuid),
      deadline_(deadline),
      register_start_(register_start),
      data_connection_reuse_(data_connection_reuse),
      last_unit_progress_(absl::Now()) {}

std::optional<TransferSendSession::DuplicateBlock>
TransferSendSession::PopulateRegisteredBlocks(
    absl::Span<const int64_t> block_ids) {
  absl::MutexLock lock(mu_);
  registered_num_blocks_ = static_cast<int64_t>(block_ids.size());
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

absl::Status TransferSendSession::ConfigureWindowedSend(
    int32_t window_chunks, std::vector<std::vector<CopySpec>> chunk_d2h_specs,
    std::vector<std::vector<int32_t>> chunk_src_host_block_ids,
    std::vector<std::vector<int32_t>> chunk_dst_host_block_ids,
    int64_t block_bytes_per_layer_per_group, int64_t stream_bytes_target) {
  if (block_bytes_per_layer_per_group < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "block_bytes_per_layer_per_group must be non-negative, got ",
        block_bytes_per_layer_per_group));
  }
  if (stream_bytes_target <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "stream_bytes_target must be positive, got ", stream_bytes_target));
  }
  const size_t n_chunks = chunk_d2h_specs.size();
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  if (window_chunks < 1 || n_chunks == 0 ||
      chunk_src_host_block_ids.size() != n_chunks ||
      chunk_dst_host_block_ids.size() != n_chunks) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid window/chunk dims: W=", window_chunks,
                     ", n=", n_chunks, " for uuid=", uuid_));
  }
  for (size_t c = 0; c < n_chunks; ++c) {
    if (chunk_d2h_specs[c].size() != total_layers ||
        chunk_src_host_block_ids[c].size() !=
            chunk_dst_host_block_ids[c].size() ||
        chunk_src_host_block_ids[c].empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid chunk ", c, " for uuid=", uuid_));
    }
  }
  absl::MutexLock lock(mu_);
  if (draining_ || done_ || pull_started_ || d2h_started_) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot configure active session uuid=", uuid_));
  }
  window_chunks_ = window_chunks;
  n_chunks_ = static_cast<int32_t>(n_chunks);
  chunk_d2h_specs_ = std::move(chunk_d2h_specs);
  chunk_src_host_block_ids_ = std::move(chunk_src_host_block_ids);
  chunk_dst_host_block_ids_ = std::move(chunk_dst_host_block_ids);
  block_bytes_per_layer_per_group_ = block_bytes_per_layer_per_group;
  stream_bytes_target_ = stream_bytes_target;
  return absl::OkStatus();
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
  return !IsWindowedLocked() && registered_block_set_.contains(block_id) &&
         layer_idx < d2h_unit_futures_.size() &&
         !d2h_unit_futures_[layer_idx].empty();
}

void TransferSendSession::RegisterLayerReadinessCallback(
    size_t layer_idx,
    transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
  std::optional<raiden::PjRtCopyFuture> future;
  {
    absl::MutexLock lock(mu_);
    if (IsWindowedLocked()) {
      cb(absl::FailedPreconditionError(
          absl::StrCat("Windowed session uuid=", uuid_, " has no readiness")));
      return;
    }
    if (layer_idx < d2h_unit_futures_.size() &&
        !d2h_unit_futures_[layer_idx].empty()) {
      future = d2h_unit_futures_[layer_idx][0];
    }
  }
  if (!future.has_value()) {
    cb(absl::OkStatus());
    return;
  }
  future->OnReady(
      [cb = std::move(cb)](auto copy_status) { cb(copy_status.status()); });
}

void TransferSendSession::ReleaseStagingBlocksLocked() {
  staging_.Reset();
  d2h_unit_futures_.clear();
}

void TransferSendSession::ReleaseStagingBlocks() {
  absl::MutexLock lock(mu_);
  ReleaseStagingBlocksLocked();
}

void TransferSendSession::FinishLocked(const absl::Status& status,
                                       std::optional<absl::string_view> phase) {
  if (draining_ || done_) return;
  status_ = status;
  if (!status.ok()) {
    failed_phase_ = phase.has_value() ? *phase : CurrentPhaseLocked();
  }
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
  // own op). A send that pushed therefore settles in EndSendOp, which
  // releases its data connections first.
  absl::MutexLock lock(mu_);
  FinishLocked(status);
}

absl::Status TransferSendSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
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

bool TransferSendSession::IsLayerAckWaitingLocked(size_t l) const {
  if (l >= h2h_acked_units_.size() || h2h_acked_units_[l] >= n_chunks_) {
    return false;
  }
  const int32_t next_push = h2h_issued_units_[l];
  if (next_push < n_chunks_ && d2h_unit_ready_[l][next_push] &&
      h2h_acked_units_[l] < next_push) {
    return true;
  }
  const int32_t next_d2h = d2h_issued_units_[l];
  return window_chunks_ > 0 && next_d2h < n_chunks_ &&
         next_d2h >= h2h_acked_units_[l] + window_chunks_ &&
         std::all_of(
             d2h_unit_ready_[l].begin(), d2h_unit_ready_[l].begin() + next_d2h,
             [](bool ready) { return ready; });
}

absl::string_view TransferSendSession::CurrentPhaseLocked() const {
  if (remote_data_endpoints_.empty()) return kAwaitPullPhase;
  if (staging_.empty()) return kStagingPhase;
  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  for (size_t l = 0; l < total_layers; ++l) {
    const int32_t issued =
        l < d2h_issued_units_.size() ? d2h_issued_units_[l] : 0;
    for (int32_t c = 0; c < issued; ++c) {
      if (l >= d2h_unit_ready_.size() || !d2h_unit_ready_[l][c]) {
        return kD2hPhase;
      }
    }
    const int32_t acked = l < h2h_acked_units_.size() ? h2h_acked_units_[l] : 0;
    if (issued < n_chunks_ &&
        (window_chunks_ == 0 || issued < acked + window_chunks_)) {
      return kD2hPhase;
    }
  }
  if (d2h_started_) {
    for (size_t l = 0; l < total_layers; ++l) {
      if (IsLayerAckWaitingLocked(l)) return kAckWaitPhase;
    }
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

void TransferSendSession::EndSendOp() {
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
    const std::vector<int64_t>& dst_block_ids,
    std::optional<StagingWindowConfig> window_config) {
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

  int64_t blocks_to_acquire = static_cast<int64_t>(src_block_ids.size());
  std::optional<ChunkPlan> chunk_plan;
  if (window_config.has_value() && window_config->window_chunks > 0 &&
      window_config->chunk_blocks > 0) {
    auto chunk_plan_result = BuildChunkPlan(
        static_cast<int64_t>(src_block_ids.size()), window_config->chunk_blocks,
        window_config->window_chunks,
        /*reuse_slots=*/true);
    if (!chunk_plan_result.ok()) {
      Finish(chunk_plan_result.status());
      return;
    }
    chunk_plan = *chunk_plan_result;
    blocks_to_acquire = chunk_plan->ring_blocks;
  }

  absl::Status acquire_status = AcquireStagingWithRetry(blocks_to_acquire);
  if (!acquire_status.ok()) {
    Finish(acquire_status);
    return;
  }

  const size_t total_layers = base_ != nullptr ? base_->num_layers() : 0;
  int32_t init_units = 1;
  bool is_windowed = false;
  int32_t first_chunk_blocks = 0;
  int64_t block_bytes = 0;
  int64_t stream_bytes_target = StagingWindowConfig{}.stream_bytes_target;
  {
    absl::MutexLock lock(mu_);
    if (draining_ || done_) return;
    if (total_layers == 0) {
      FinishLocked(absl::OkStatus());
      return;
    }
    last_unit_progress_ = absl::Now();
    if (chunk_d2h_specs_.empty()) {
      absl::Span<const int> blocks = staging_.block_ids();
      if (chunk_plan.has_value()) {
        std::vector<int64_t> ring_bids(
            blocks.begin(), blocks.begin() + chunk_plan->ring_blocks);
        auto expanded_ring_result = chunk_plan->ExpandRing(ring_bids);
        if (!expanded_ring_result.ok()) {
          FinishLocked(expanded_ring_result.status());
          return;
        }
        const std::vector<int64_t>& expanded_ring = *expanded_ring_result;
        const int32_t n_chunks = chunk_plan->n_chunks;
        chunk_d2h_specs_.resize(n_chunks);
        chunk_src_host_block_ids_.resize(n_chunks);
        chunk_dst_host_block_ids_.resize(n_chunks);
        for (int32_t c = 0; c < n_chunks; ++c) {
          const ChunkRange range = chunk_plan->ChunkRange(c);
          std::vector<int64_t> chunk_src_dev(
              src_block_ids.begin() + range.begin,
              src_block_ids.begin() + range.end);
          std::vector<int64_t> chunk_host(expanded_ring.begin() + range.begin,
                                          expanded_ring.begin() + range.end);
          chunk_d2h_specs_[c] = std::vector<CopySpec>(
              total_layers, BuildCoalescedCopySpec(chunk_src_dev, chunk_host));
          chunk_src_host_block_ids_[c] =
              std::vector<int32_t>(chunk_host.begin(), chunk_host.end());
          chunk_dst_host_block_ids_[c] =
              std::vector<int32_t>(dst_block_ids.begin() + range.begin,
                                   dst_block_ids.begin() + range.end);
        }
        window_chunks_ = chunk_plan->window_chunks;
        n_chunks_ = n_chunks;
        stream_bytes_target_ = window_config->stream_bytes_target;
      } else {
        std::vector<int64_t> host_blocks(blocks.begin(),
                                         blocks.begin() + src_block_ids.size());
        chunk_d2h_specs_ = {std::vector<CopySpec>(
            total_layers, BuildCoalescedCopySpec(src_block_ids, host_blocks))};
        chunk_src_host_block_ids_ = {
            {blocks.begin(), blocks.begin() + src_block_ids.size()}};
        chunk_dst_host_block_ids_ = {
            {dst_block_ids.begin(), dst_block_ids.end()}};
      }
    }
    d2h_started_ = true;
    pull_started_ = true;
    is_windowed = IsWindowedLocked();
    if (!chunk_src_host_block_ids_.empty()) {
      first_chunk_blocks =
          static_cast<int32_t>(chunk_src_host_block_ids_[0].size());
    }
    block_bytes = block_bytes_per_layer_per_group_;
    stream_bytes_target = stream_bytes_target_;
    init_units = window_chunks_ > 0 ? std::min(window_chunks_, n_chunks_) : 1;
    remaining_h2h_units_.store(total_layers * n_chunks_,
                               std::memory_order_relaxed);
    d2h_issued_units_.assign(total_layers, init_units);
    in_flight_ += static_cast<int>(total_layers) * init_units;
    d2h_unit_ready_.assign(total_layers, std::vector<bool>(n_chunks_, false));
    h2h_issued_units_.assign(total_layers, 0);
    h2h_acked_units_.assign(total_layers, 0);
    d2h_unit_futures_.assign(total_layers,
                             std::vector<raiden::PjRtCopyFuture>(n_chunks_));
  }

  if (is_windowed) {
    const int P = base_ != nullptr ? base_->parallelism() : 1;
    const int G = base_ != nullptr ? base_->num_numa_groups() : 1;
    if (block_bytes == 0 && base_ != nullptr) {
      block_bytes = (static_cast<int64_t>(base_->block_bytes(0)) *
                     static_cast<int64_t>(base_->num_shards())) /
                    std::max(1, G);
    }
    const int64_t unit_bytes_0 =
        static_cast<int64_t>(first_chunk_blocks) * block_bytes;
    const int n_u_first_chunk = StreamsPerUnit(
        unit_bytes_0, stream_bytes_target, P, first_chunk_blocks);
    if (P < 16) {
      LOG(WARNING) << "Windowed send running with parallelism = " << P
                   << " < 16 for uuid=" << uuid_;
    }
    if (static_cast<int64_t>(total_layers) * n_u_first_chunk <
        static_cast<int64_t>(P) * G) {
      LOG(WARNING) << "Windowed send with total_layers * n_u ("
                   << (total_layers * n_u_first_chunk) << ") < P * G ("
                   << (P * G)
                   << ") cannot saturate all peer stream slots for uuid="
                   << uuid_;
    }
  }

  for (size_t l = 0; l < total_layers; ++l) {
    for (int32_t c = 0; c < init_units; ++c) {
      DispatchD2h(l, c);
    }
  }
}

void TransferSendSession::DispatchD2h(size_t l, int32_t c) {
  RAIDEN_TRACE_FN("Window::D2H", [&]() {
    return absl::StrCat("uuid=", uuid_, " l=", l, " c=", c);
  });
  absl::Cleanup end_op = [this]() { EndSendOp(); };
  CopySpec copy_spec;
  {
    absl::MutexLock lock(mu_);
    if (draining_ || done_) return;
    copy_spec = chunk_d2h_specs_[c][l];
  }

  VLOG(1) << "StartPush (D2H start layer " << l << " unit " << c
          << "): uuid=" << uuid_;

  absl::Status injected =
      FaultInjectStatus(hooks::kTransferSendSessionD2hDispatch);
  absl::StatusOr<raiden::PjRtCopyFuture> future =
      injected.ok()
          ? base_->D2hSyncDispatch(copy_spec.src_offsets, copy_spec.dst_offsets,
                                   copy_spec.sizes, /*slot_idx=*/std::nullopt,
                                   /*layer_idx=*/l)
          : absl::StatusOr<raiden::PjRtCopyFuture>(injected);

  if (!future.ok()) {
    LOG(ERROR) << "DispatchD2h failed for layer " << l << " unit " << c << ": "
               << future.status();
    absl::MutexLock lock(mu_);
    FinishLocked(future.status(), kD2hPhase);
    return;
  }

  raiden::PjRtCopyFuture copy_future = *std::move(future);
  {
    absl::MutexLock lock(mu_);
    d2h_unit_futures_[l][c] = copy_future;
  }
  std::move(end_op).Cancel();
  copy_future.OnReady([self = shared_from_this(), l, c](auto copy_status) {
    self->OnD2hComplete(l, c, copy_status.status());
  });
}

void TransferSendSession::OnD2hComplete(size_t l, int32_t c,
                                        const absl::Status& status) {
  absl::Cleanup end_op = [this]() { EndSendOp(); };
  absl::Status injected =
      FaultInjectStatus(hooks::kTransferSendSessionD2hComplete);
  const absl::Status s = injected.ok() ? status : injected;
  if (!s.ok()) {
    LOG(ERROR) << "D2H copy failed for layer " << l << " unit " << c << ": "
               << s;
    absl::MutexLock lock(mu_);
    FinishLocked(s, kD2hPhase);
    return;
  }

  absl::MutexLock lock(mu_);
  if (draining_ || done_) return;
  d2h_unit_ready_[l][c] = true;
  last_unit_progress_ = absl::Now();
  if (d2h_done_ == std::chrono::steady_clock::time_point{}) {
    d2h_done_ = std::chrono::steady_clock::now();
  }
  MaybeScheduleNextLayerUnitsLocked(l);
}

void TransferSendSession::DispatchH2h(size_t l, int32_t c) {
  absl::Cleanup end_op = [this]() { EndSendOp(); };
  std::vector<std::string> endpoints;
  std::vector<int> src_ints, dst_ints;
  std::optional<int> push_parallelism = std::nullopt;
  int n_u_log = 0;
  {
    absl::MutexLock lock(mu_);
    if (draining_ || done_) return;
    endpoints = remote_data_endpoints_;
    src_ints = chunk_src_host_block_ids_[c];
    dst_ints = chunk_dst_host_block_ids_[c];
    const int P = base_ != nullptr ? base_->parallelism() : 1;
    if (IsWindowedLocked()) {
      const int32_t chunk_blocks_c = static_cast<int32_t>(src_ints.size());
      int64_t block_bytes_per_group = block_bytes_per_layer_per_group_;
      if (block_bytes_per_group == 0 && base_ != nullptr) {
        block_bytes_per_group = (static_cast<int64_t>(base_->block_bytes(l)) *
                                 static_cast<int64_t>(base_->num_shards())) /
                                std::max(1, base_->num_numa_groups());
      }
      const int64_t unit_bytes =
          static_cast<int64_t>(chunk_blocks_c) * block_bytes_per_group;
      const int32_t n_u =
          StreamsPerUnit(unit_bytes, stream_bytes_target_, P, chunk_blocks_c);
      push_parallelism = n_u;
      n_u_log = n_u;
    } else {
      push_parallelism = P;
      n_u_log = P;
    }
  }

  VLOG(1) << "StartPush (H2H start layer " << l << " unit " << c
          << " n_u=" << n_u_log << "): uuid=" << uuid_ << ", numa="
          << (base_ != nullptr ? base_->assigned_numa_node().value_or(-1) : -1);

  std::move(end_op).Cancel();
  base_->H2hWriteDirectAsync(
      endpoints, src_ints, dst_ints, uuid_, static_cast<int>(l),
      [self = shared_from_this(), l, c](auto push_res) {
        self->OnH2hComplete(l, c, push_res);
      },
      push_parallelism);
}

void TransferSendSession::OnH2hComplete(
    size_t l, int32_t c, const absl::StatusOr<std::vector<int>>& push_res) {
  absl::Cleanup end_op = [this]() { EndSendOp(); };
  if (!push_res.ok()) {
    LOG(ERROR) << "H2hWrite failed for layer " << l << " unit " << c << ": "
               << push_res.status();
    absl::MutexLock lock(mu_);
    FinishLocked(push_res.status(), kPushPhase);
    return;
  }

  VLOG(1) << "StartPush (H2H complete layer " << l << " unit " << c
          << "): uuid=" << uuid_
          << ", numa=" << base_->assigned_numa_node().value_or(-1);

  absl::MutexLock lock(mu_);
  h2h_acked_units_[l] = c + 1;
  last_unit_progress_ = absl::Now();
  if (remaining_h2h_units_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    LOG(INFO) << "StartPush (All H2H complete): uuid=" << uuid_;
    FinishLocked(absl::OkStatus());
    return;
  }
  MaybeScheduleNextLayerUnitsLocked(l);
}

void TransferSendSession::MaybeScheduleNextLayerUnitsLocked(size_t l) {
  if (draining_ || done_ || !d2h_started_) return;
  const absl::Time now = absl::Now();
  if (IsWindowedLocked() && last_unit_progress_ != absl::InfinitePast() &&
      now - last_unit_progress_ > kUnitProgressStallThreshold) {
    LOG(WARNING) << "Windowed send session uuid=" << uuid_
                 << " made no unit progress for "
                 << absl::FormatDuration(now - last_unit_progress_);
  }
  const int32_t max_d2h =
      window_chunks_ > 0 ? h2h_acked_units_[l] + window_chunks_ : n_chunks_;
  while (d2h_issued_units_[l] < n_chunks_ && d2h_issued_units_[l] < max_d2h) {
    const int32_t c = d2h_issued_units_[l]++;
    ++in_flight_;
    base_->push_pool()->Schedule(
        [self = shared_from_this(), l, c]() { self->DispatchD2h(l, c); });
  }
  if (d2h_issued_units_[l] < n_chunks_ && d2h_issued_units_[l] >= max_d2h) {
    const int32_t acked = h2h_acked_units_[l];
    auto meta_fn = [this, acked, l]() {
      return absl::StrCat("uuid=", uuid_, " l=", l, " acked=", acked);
    };
    RAIDEN_TRACE_FN("Window::AckWaitEnter", meta_fn);
  }

  const int32_t c = h2h_issued_units_[l];
  if (c < n_chunks_ && d2h_unit_ready_[l][c] && h2h_acked_units_[l] == c) {
    h2h_issued_units_[l]++;
    ++in_flight_;
    base_->push_pool()->Schedule(
        [self = shared_from_this(), l, c]() { self->DispatchH2h(l, c); });
  }
}

}  // namespace tpu_raiden

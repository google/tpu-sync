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

#include "tpu_sync/core/transfer_send_session.h"

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
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

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
  auto session = std::shared_ptr<TransferSendSession>(
      new TransferSendSession(base, staging_allocator, std::move(req_id), uuid,
                              deadline, register_start));
  std::optional<int64_t> duplicate_block =
      session->PopulateRegisteredBlocks(block_ids);
  if (duplicate_block.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "NotifyForRead rejected duplicate block ", *duplicate_block,
        " for req_id=", session->req_id(), ", uuid=", uuid));
  }
  return session;
}

TransferSendSession::TransferSendSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, std::string req_id, uint64_t uuid,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start)
    : base_(base),
      staging_allocator_(staging_allocator),
      req_id_(std::move(req_id)),
      uuid_(uuid),
      deadline_(deadline),
      register_start_(register_start) {}

std::optional<int64_t> TransferSendSession::PopulateRegisteredBlocks(
    absl::Span<const int64_t> block_ids) {
  absl::MutexLock lock(mu_);
  registered_num_blocks_ = static_cast<int64_t>(block_ids.size());
  registered_block_ids_.assign(block_ids.begin(), block_ids.end());
  registered_block_set_.clear();
  std::optional<int64_t> duplicate_block;
  for (int64_t block_id : block_ids) {
    if (!registered_block_set_.insert(block_id).second) {
      duplicate_block = block_id;
    }
  }
  return duplicate_block;
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
  staging_.Reset();
  d2h_layer_futures_.clear();
}

void TransferSendSession::ReleaseStagingBlocks() {
  absl::MutexLock lock(mu_);
  ReleaseStagingBlocksLocked();
}

void TransferSendSession::FinishLocked(const absl::Status& status) {
  if (draining_ || done_) return;
  status_ = status;
  draining_ = true;
  if (in_flight_ == 0) {
    ReleaseStagingBlocksLocked();
    done_ = true;
  }
}

void TransferSendSession::Finish(const absl::Status& status) {
  absl::MutexLock lock(mu_);
  FinishLocked(status);
}

absl::Status TransferSendSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

void TransferSendSession::EndSendOpLocked() {
  --in_flight_;
  if (draining_ && in_flight_ == 0 && !done_) {
    ReleaseStagingBlocksLocked();
    done_ = true;
  }
}

void TransferSendSession::EndSendOp() {
  absl::MutexLock lock(mu_);
  EndSendOpLocked();
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

  absl::Status acquire_status =
      AcquireStagingWithRetry(static_cast<int64_t>(src_block_ids.size()));
  if (!acquire_status.ok()) {
    Finish(acquire_status);
    return;
  }

  const size_t total_layers = base_->num_layers();
  std::vector<int64_t> host_block_ids;
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
    d2h_layer_futures_.reserve(total_layers);
    remote_data_endpoints_ = remote_data_endpoints;
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

  // 1. Issue D2H copies layer-by-layer. Each copy is counted against the
  // send before it starts and released by its own completion, so the
  // staging it writes stays owned for as long as it runs.
  for (size_t l = 0; l < total_layers; ++l) {
    {
      absl::MutexLock session_lock(mu_);
      // The send expired or failed while its copies were being issued:
      // what was issued drains, nothing more starts.
      if (draining_) return;
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
                                     /*layer_idx=*/l)
            : absl::StatusOr<raiden::PjRtCopyFuture>(injected);
    if (!future.ok()) {
      // A copy that cannot even be issued fails the transfer; the worker
      // thread has no caller for an exception to reach.
      LOG(ERROR) << "StartPush: failed to issue D2H for layer " << l << ": "
                 << future.status();
      absl::MutexLock lock(mu_);
      FinishLocked(future.status());
      EndSendOpLocked();  // this copy never started
      return;
    }
    raiden::PjRtCopyFuture layer_future = *std::move(future);
    {
      absl::MutexLock lock(mu_);
      d2h_layer_futures_.push_back(layer_future);
    }
    layer_future.OnReady(
        [self = shared_from_this()](absl::StatusOr<raiden::BufferHolders>) {
          self->EndSendOp();
        });
  }

  SendNextLayer(0);
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
    absl::Cleanup end_op = [self]() { self->EndSendOp(); };
    if (!status_or.ok()) {
      LOG(ERROR) << "StartPush: D2H copy failed for layer " << l
                 << ", status: " << status_or.status().ToString();
      self->Finish(status_or.status());
      return;
    }
    {
      absl::MutexLock session_lock(self->mu_);
      // The send expired or failed while the copy ran: nothing is pushed.
      if (self->draining_) {
        return;
      }
    }

    std::move(end_op).Cancel();
    self->base_->push_pool()->Schedule([self, l]() {
      FaultInjectThrow(hooks::kTransferSendSessionPushTask);
      absl::Cleanup end_op = [self]() { self->EndSendOp(); };
      std::vector<std::string> remote_data_endpoints;
      std::vector<int> src_ints;
      std::vector<int> dst_ints;
      {
        absl::MutexLock lock(self->mu_);
        if (self->draining_) {
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
            absl::Cleanup end_op = [self]() { self->EndSendOp(); };
            if (!push_res.ok()) {
              LOG(ERROR) << "H2hWrite failed for layer " << l << ": "
                         << push_res.status().ToString();
              self->Finish(push_res.status());
              return;
            }

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

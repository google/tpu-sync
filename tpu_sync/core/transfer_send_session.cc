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

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

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
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

absl::StatusOr<std::shared_ptr<TransferSendSession>>
TransferSendSession::Create(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* staging_allocator, std::string req_id, uint64_t uuid,
    absl::Span<const int64_t> block_ids,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point register_start, int in_flight,
    bool pull_started) {
  auto session = std::shared_ptr<TransferSendSession>(new TransferSendSession(
      base, staging_allocator, std::move(req_id), uuid, deadline,
      register_start, in_flight, pull_started));
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
    std::chrono::steady_clock::time_point register_start, int in_flight,
    bool pull_started)
    : base_(base),
      staging_allocator_(staging_allocator),
      req_id_(std::move(req_id)),
      uuid_(uuid),
      deadline_(deadline),
      register_start_(register_start),
      pull_started_(pull_started),
      in_flight_(in_flight) {
  if (in_flight > 0 && staging_allocator_ != nullptr) {
    std::optional<StagingAllocation> acquired = staging_allocator_->Acquire(1);
    if (acquired.has_value()) {
      staging_ = *std::move(acquired);
    }
  }
}

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

void TransferSendSession::ReleaseSlotLocked() { staging_.Reset(); }

void TransferSendSession::ReleaseSlot() {
  absl::MutexLock lock(mu_);
  ReleaseSlotLocked();
}

void TransferSendSession::FinishSendLocked(bool has_failed) {
  if (has_failed) failed_ = true;
  if (draining_) return;
  draining_ = true;
  if (in_flight_ == 0 && !done_) {
    ReleaseSlotLocked();
    done_ = true;
  }
}

void TransferSendSession::FinishSend(bool has_failed) {
  absl::MutexLock lock(mu_);
  FinishSendLocked(has_failed);
}

void TransferSendSession::EndSendOpLocked() {
  --in_flight_;
  if (draining_ && in_flight_ == 0 && !done_) {
    ReleaseSlotLocked();
    done_ = true;
  }
}

void TransferSendSession::EndSendOp() {
  absl::MutexLock lock(mu_);
  EndSendOpLocked();
}

bool TransferSendSession::AcquireStagingWithRetry(
    KVCacheManagerWithTransfer& manager,
    const std::vector<int64_t>& src_block_ids,
    std::vector<int64_t>* host_block_ids) {
  // The entry's deadline, set when the producer registered the request,
  // bounds the whole transfer; the wait for staging shares it rather than
  // starting a later one of its own.
  while (true) {
    if (manager.shutting_down_.load(std::memory_order_relaxed)) {
      return false;  // the manager is being destroyed; its state goes with it
    }
    {
      absl::MutexLock session_lock(mu_);
      if (draining_) {
        return false;  // request cancelled while waiting for staging
      }
    }
    // Staging that can never seat this request fails it now rather than
    // after the deadline: a fixed slot holds max_blocks_ pages, the
    // per-transfer pool holds total_blocks() pages.
    const int64_t capacity = staging_allocator_->capacity();
    if (static_cast<int64_t>(src_block_ids.size()) > capacity) {
      LOG(ERROR) << "StartPush: request " << req_id_ << " needs "
                 << src_block_ids.size() << " blocks but "
                 << (staging_allocator_->dynamic_host_staging()
                         ? "the host staging pool holds "
                         : "a staging slot holds ")
                 << capacity;
      FinishSend(/*has_failed=*/true);
      return false;
    }
    std::optional<StagingAllocation> acquired =
        staging_allocator_->Acquire(static_cast<int64_t>(src_block_ids.size()));
    if (acquired.has_value()) {
      absl::Span<const int> blocks = acquired->block_ids();
      host_block_ids->clear();
      host_block_ids->reserve(src_block_ids.size());
      for (size_t i = 0; i < src_block_ids.size(); ++i) {
        host_block_ids->push_back(blocks[i]);
      }
      absl::MutexLock session_lock(mu_);
      staging_ = *std::move(acquired);
      return true;
    }
    // Staging exhausted: wait for in-flight sends to hand blocks back
    // instead of reporting a send that never happened. The consumer's own
    // deadline still bounds the total wait.
    if (std::chrono::steady_clock::now() >= deadline_) {
      bool already_draining = false;
      {
        absl::MutexLock session_lock(mu_);
        already_draining = draining_;
      }
      if (!already_draining) {
        LOG(ERROR) << "StartPush: staging exhausted serving " << req_id_ << " ("
                   << src_block_ids.size() << " blocks; free_host_blocks="
                   << base_->host_block_manager()->num_free_blocks()
                   << ", total_host_blocks="
                   << base_->host_block_manager()->total_blocks()
                   << ", free_slots="
                   << manager.staging_allocator_->num_free_slots()
                   << "); reporting transfer failure";
        FinishSend(/*has_failed=*/true);
      }
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

void TransferSendSession::StartPush(
    KVCacheManagerWithTransfer& manager,
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
  std::vector<int64_t> host_block_ids;
  if (!AcquireStagingWithRetry(manager, src_block_ids, &host_block_ids)) {
    return;
  }

  // Coalesce contiguous (device,host) block runs into a few large copies. With
  // per-block segments (sizes=1) a contiguous KV range becomes n device copies
  // that flood the command queue with small ops and serialize against prefill
  // GEMMs on the shared TensorCore; coalescing collapses a contiguous range to
  // one copy, matching the pre-Hybrid-Push pull path.
  CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);
  const size_t total_layers = base_->num_layers();
  {
    absl::MutexLock lock(mu_);
    d2h_layer_futures_.reserve(total_layers);
    remote_data_endpoints_ = remote_data_endpoints;
    src_ints_.assign(host_block_ids.begin(), host_block_ids.end());
    dst_ints_.assign(dst_block_ids.begin(), dst_block_ids.end());
  }
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
    absl::StatusOr<raiden::PjRtCopyFuture> future =
        base_->D2hSyncDispatch(d2h_copy.src_offsets, d2h_copy.dst_offsets,
                               d2h_copy.sizes, /*slot_idx=*/std::nullopt,
                               /*layer_idx=*/l);
    if (!future.ok()) {
      // A copy that cannot even be issued fails the transfer; the worker
      // thread has no caller for an exception to reach.
      LOG(ERROR) << "StartPush: failed to issue D2H for layer " << l << ": "
                 << future.status();
      absl::MutexLock lock(mu_);
      FinishSendLocked(/*has_failed=*/true);
      EndSendOpLocked();  // this copy never started
      return;
    }
    raiden::PjRtCopyFuture layer_future = *std::move(future);
    {
      absl::MutexLock lock(mu_);
      d2h_layer_futures_.push_back(layer_future);
    }
    layer_future.OnReady(
        [this](absl::StatusOr<raiden::BufferHolders>) { EndSendOp(); });
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

  layer_future->OnReady([this, l](auto status_or) {
    if (!status_or.ok()) {
      LOG(ERROR) << "StartPush: D2H copy failed for layer " << l
                 << ", status: " << status_or.status().ToString();
      absl::MutexLock lock(mu_);
      FinishSendLocked(/*has_failed=*/true);
      EndSendOpLocked();
      return;
    }
    bool is_draining = false;
    {
      absl::MutexLock session_lock(mu_);
      // The send expired or failed while the copy ran: nothing is pushed.
      is_draining = draining_;
      if (is_draining) {
        EndSendOpLocked();
      }
    }
    if (is_draining) {
      return;
    }

    base_->push_pool()->Schedule([this, l]() {
      std::vector<std::string> remote_data_endpoints;
      std::vector<int> src_ints;
      std::vector<int> dst_ints;
      {
        absl::MutexLock lock(mu_);
        if (draining_) {
          EndSendOpLocked();
          return;
        }
        remote_data_endpoints = remote_data_endpoints_;
        src_ints = src_ints_;
        dst_ints = dst_ints_;
        ++in_flight_;
      }
      LOG(INFO) << "StartPush (H2H start layer " << l << "): uuid=" << uuid_
                << ", numa=" << base_->assigned_numa_node().value_or(-1);
      base_->H2hWriteDirectAsync(
          remote_data_endpoints, src_ints, dst_ints, uuid_, static_cast<int>(l),
          [this, l](absl::StatusOr<std::vector<int>> push_res) {
            if (!push_res.ok()) {
              LOG(ERROR) << "H2hWrite failed for layer " << l << ": "
                         << push_res.status().ToString();
              absl::MutexLock lock(mu_);
              FinishSendLocked(/*has_failed=*/true);
              EndSendOpLocked();
              return;
            }

            LOG(INFO) << "StartPush (H2H complete layer " << l
                      << "): uuid=" << uuid_
                      << ", numa=" << base_->assigned_numa_node().value_or(-1);

            const bool last = remaining_h2h_layers_.fetch_sub(1) == 1;
            if (last) {
              LOG(INFO) << "StartPush (All H2H complete): uuid=" << uuid_;
            }
            absl::MutexLock lock(mu_);
            if (last) {
              FinishSendLocked(/*has_failed=*/false);
            }
            EndSendOpLocked();
          });

      // Immediately queue the next layer's push without waiting for this one to
      // finish.
      SendNextLayer(l + 1);
      EndSendOp();
    });
  });
}

}  // namespace tpu_raiden

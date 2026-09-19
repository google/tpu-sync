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

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

TransferSendSession::TransferSendSession(
    kv_cache::KVCacheManagerBase* base_in, std::string req_id_in,
    uint64_t uuid_in, std::chrono::steady_clock::time_point deadline_in,
    std::chrono::steady_clock::time_point register_start_in)
    : base_(base_in),
      req_id_(std::move(req_id_in)),
      uuid_(uuid_in),
      register_start_(register_start_in),
      deadline_(deadline_in) {}

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

void TransferSendSession::RetireSendLocked(
    KVCacheManagerWithTransfer& manager) {
  std::shared_ptr<TransferSendSession> self = shared_from_this();
  (failed() ? manager.failed_recving_ : manager.done_sending_).insert(req_id_);
  manager.ReleaseEntrySlotLocked(self);
  auto it = manager.send_entries_.find(uuid_);
  if (it != manager.send_entries_.end() && it->second == self) {
    manager.send_entries_.erase(it);
  }
}

void TransferSendSession::FinishSendLocked(KVCacheManagerWithTransfer& manager,
                                           bool has_failed) {
  bool should_retire = false;
  {
    absl::MutexLock lock(mu_);
    if (has_failed) failed_ = true;
    if (draining_) return;
    draining_ = true;
    should_retire = (in_flight_ == 0);
  }
  if (should_retire) {
    RetireSendLocked(manager);
  }
}

void TransferSendSession::EndSendOpLocked(KVCacheManagerWithTransfer& manager) {
  bool should_retire = false;
  {
    absl::MutexLock lock(mu_);
    --in_flight_;
    should_retire = (draining_ && in_flight_ == 0);
  }
  if (should_retire) {
    RetireSendLocked(manager);
  }
}

void TransferSendSession::EndSendOp(KVCacheManagerWithTransfer& manager) {
  absl::MutexLock lock(manager.mu_);
  EndSendOpLocked(manager);
}

void TransferSendSession::ExecutePush(
    KVCacheManagerWithTransfer& manager, const CopySpec& d2h_copy,
    const std::vector<std::string>& remote_endpoints,
    const std::vector<int64_t>& host_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  std::shared_ptr<TransferSendSession> self = shared_from_this();
  const size_t total_layers = base_->num_layers();
  {
    absl::MutexLock lock(mu_);
    d2h_layer_futures_.reserve(total_layers);
    remote_data_endpoints_ = remote_endpoints;
    src_ints_.assign(host_block_ids.begin(), host_block_ids.end());
    dst_ints_.assign(dst_block_ids.begin(), dst_block_ids.end());
    num_blocks_ = static_cast<int64_t>(host_block_ids.size());
  }
  remaining_h2h_layers_.store(total_layers, std::memory_order_relaxed);

  // 1. Issue D2H copies layer-by-layer. Each copy is counted against the
  // send before it starts and released by its own completion, so the
  // staging it writes stays owned for as long as it runs.
  for (size_t l = 0; l < total_layers; ++l) {
    {
      absl::MutexLock lock(manager.mu_);
      // The send expired or failed while its copies were being issued:
      // what was issued drains, nothing more starts.
      if (draining()) return;
      BeginSendOpLocked();
    }
    LOG(INFO) << "StartPushInternal (D2H start) layer " << l
              << ": uuid=" << uuid_
              << ", numa=" << base_->assigned_numa_node().value_or(-1);
    absl::StatusOr<raiden::PjRtCopyFuture> future =
        base_->D2hSyncDispatch(d2h_copy.src_offsets, d2h_copy.dst_offsets,
                               d2h_copy.sizes, /*slot_idx=*/std::nullopt,
                               /*layer_idx=*/l);
    if (!future.ok()) {
      // A copy that cannot even be issued fails the transfer; the worker
      // thread has no caller for an exception to reach.
      LOG(ERROR) << "StartPushInternal: failed to issue D2H for layer " << l
                 << ": " << future.status();
      absl::MutexLock lock(manager.mu_);
      FinishSendLocked(manager, /*has_failed=*/true);
      EndSendOpLocked(manager);  // this copy never started
      return;
    }
    raiden::PjRtCopyFuture layer_future = *std::move(future);
    {
      absl::MutexLock lock(mu_);
      d2h_layer_futures_.push_back(layer_future);
    }
    layer_future.OnReady(
        [&manager, self](absl::StatusOr<raiden::BufferHolders>) {
          self->EndSendOp(manager);
        });
  }

  SendNextLayer(manager, 0);
}

void TransferSendSession::SendNextLayer(KVCacheManagerWithTransfer& manager,
                                        size_t l) {
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
    if (l < d2h_layer_futures_.size()) {
      layer_future = d2h_layer_futures_[l];
    }
  }
  if (!layer_future.has_value()) {
    return;
  }

  std::shared_ptr<TransferSendSession> self = shared_from_this();
  layer_future->OnReady([&manager, self, l](auto status_or) {
    if (!status_or.ok()) {
      LOG(ERROR) << "StartPushInternal: D2H copy failed for layer " << l
                 << ", status: " << status_or.status().ToString();
      absl::MutexLock lock(manager.mu_);
      self->FinishSendLocked(manager, /*has_failed=*/true);
      return;
    }
    {
      absl::MutexLock lock(manager.mu_);
      // The send expired or failed while the copy ran: nothing is pushed.
      if (self->draining()) return;
      self->BeginSendOpLocked();
    }

    self->base_->push_pool()->Schedule([&manager, self, l]() {
      std::vector<std::string> remote_data_endpoints;
      std::vector<int> src_ints;
      std::vector<int> dst_ints;
      {
        absl::MutexLock lock(self->mu_);
        remote_data_endpoints = self->remote_data_endpoints_;
        src_ints = self->src_ints_;
        dst_ints = self->dst_ints_;
      }
      LOG(INFO) << "StartPushInternal (H2H start layer " << l
                << "): uuid=" << self->uuid_
                << ", numa=" << self->base_->assigned_numa_node().value_or(-1);
      self->base_->H2hWriteDirectAsync(
          remote_data_endpoints, src_ints, dst_ints, self->uuid_,
          static_cast<int>(l),
          [&manager, self, l](absl::StatusOr<std::vector<int>> push_res) {
            if (!push_res.ok()) {
              LOG(ERROR) << "H2hWrite failed for layer " << l << ": "
                         << push_res.status().ToString();
              absl::MutexLock lock(manager.mu_);
              self->FinishSendLocked(manager, /*has_failed=*/true);
              self->EndSendOpLocked(manager);
              return;
            }

            LOG(INFO) << "StartPushInternal (H2H complete layer " << l
                      << "): uuid=" << self->uuid_ << ", numa="
                      << self->base_->assigned_numa_node().value_or(-1);

            const bool last = self->remaining_h2h_layers_.fetch_sub(1) == 1;
            absl::MutexLock lock(manager.mu_);
            if (last) {
              LOG(INFO) << "StartPushInternal (All H2H complete): uuid="
                        << self->uuid_;
              self->FinishSendLocked(manager, /*has_failed=*/false);
            }
            self->EndSendOpLocked(manager);
          });

      // Immediately queue the next layer's push without waiting for this one to
      // finish.
      self->SendNextLayer(manager, l + 1);
    });
  });
}

}  // namespace tpu_raiden

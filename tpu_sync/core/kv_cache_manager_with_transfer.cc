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

// Dummy change to force Kokoro retry.
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/metrics_collector.h"
#include "tpu_sync/core/pool_reshard_send_slots.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/core/transfer_receive_session.h"
#include "tpu_sync/core/transfer_send_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

namespace {

constexpr absl::Duration kPendingWorkTimeout = absl::Seconds(30);

// How long a pull request waits for the producer to register the read it
// names. The registration normally precedes the announcement the consumer
// acts on, so this only covers reordering between the two; a pull whose
// registration expired, or never happened, is rejected once it lapses.
constexpr absl::Duration kPullRegistrationGrace = absl::Seconds(5);

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

void EmitTimingLog(const std::string& message) { LOG(INFO) << message; }

bool StridedSpanFitsBlock(int64_t offset, int64_t stride, int64_t size,
                          int64_t count, int64_t block_size) {
  if (offset < 0 || stride < 0 || size <= 0 || count <= 0 || block_size <= 0 ||
      offset > block_size || size > block_size - offset) {
    return false;
  }
  if (count == 1) return true;

  // Division avoids overflowing (count - 1) * stride.
  const int64_t remaining = block_size - offset - size;
  return stride <= remaining / (count - 1);
}

bool StridedSpanFitsRegions(int64_t offset, int64_t stride, int64_t size,
                            int64_t count, int64_t block_size,
                            const std::vector<kv_cache::RegionSpec>& regions) {
  if (!StridedSpanFitsBlock(offset, stride, size, count, block_size)) {
    return false;
  }
  for (int64_t repeat = 0; repeat < count; ++repeat) {
    const int64_t start = offset + repeat * stride;
    if (!kv_cache::RegionsCoverRange(regions, static_cast<size_t>(start),
                                     static_cast<size_t>(start + size))) {
      return false;
    }
  }
  return true;
}

template <typename T>
T ValueOrThrow(const std::string& context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

static CopySpec OffsetsImpl(const std::vector<int64_t>& block_ids,
                            bool source_is_compact) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  CopySpec spec;
  spec.src_offsets.reserve(block_ids.size());
  spec.dst_offsets.reserve(block_ids.size());
  spec.sizes.reserve(block_ids.size());
  for (int64_t start = 0; start < n;) {
    int64_t end = start + 1;
    while (end < n && block_ids[end] == block_ids[end - 1] + 1) {
      ++end;
    }
    const int64_t run_size = end - start;
    if (source_is_compact) {
      spec.src_offsets.push_back(start);
      spec.dst_offsets.push_back(block_ids[start]);
    } else {
      spec.src_offsets.push_back(block_ids[start]);
      spec.dst_offsets.push_back(start);
    }
    spec.sizes.push_back(run_size);
    start = end;
  }
  return spec;
}

static kv_cache::KVCacheCopySpec ToKVCacheCopySpecImpl(const CopySpec& spec) {
  return {.src_offsets = spec.src_offsets,
          .dst_offsets = spec.dst_offsets,
          .sizes = spec.sizes};
}

static CopyPlan BuildLoadCopyPlan(
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
  plan.host_dst_to_src.clear();  // No host reordering needed!
  return plan;
}

double DurationMs(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

CopySpec KVCacheManagerWithTransfer::Offsets(
    const std::vector<int64_t>& block_ids, bool source_is_compact) {
  return OffsetsImpl(block_ids, source_is_compact);
}

kv_cache::KVCacheCopySpec KVCacheManagerWithTransfer::ToKVCacheCopySpec(
    const CopySpec& spec) {
  return ToKVCacheCopySpecImpl(spec);
}

void KVCacheManagerWithTransfer::InitializeBaseHooks() {
  kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
  hooks.register_block_readiness_callback =
      [this](size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
             transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
        RegisterBlockReadinessCallback(layer_idx, shard_idx, block_id, uuid,
                                       std::move(cb));
      };
  hooks.on_blocks_received = [this](const std::vector<int>& block_ids,
                                    uint64_t uuid) {
    return OnBlocksReceived(block_ids, uuid);
  };
  hooks.on_layer_received = [this](size_t layer_idx, uint64_t uuid) {
    RAIDEN_TRACE_FN("KVTransfer::OnLayerReceived", [&]() {
      return absl::StrCat("layer=", layer_idx, " uuid=", uuid);
    });
    std::shared_ptr<ReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_entries_.find(uuid);
      if (it == active_recv_entries_.end()) {
        return absl::OkStatus();
      }
      session = it->second;
    }
    absl::Status status = session->ExecuteLayerH2d(*this, layer_idx);
    MaybeUnregisterSettledRecv(uuid, *session);
    return status;
  };
  hooks.on_pool_received = [this](size_t pool_idx, uint64_t uuid) {
    RAIDEN_TRACE_FN("KVTransfer::OnPoolReceived", [&]() {
      return absl::StrCat("pool=", pool_idx, " uuid=", uuid);
    });
    std::shared_ptr<ReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_entries_.find(uuid);
      if (it == active_recv_entries_.end()) {
        return absl::NotFoundError(
            absl::StrCat("no active receiver for UUID ", uuid));
      }
      session = it->second;
    }
    return session->OnPoolReceived(*this, pool_idx);
  };
  hooks.pool_reshard_push =
      [this](const ::tpu_sync::rpc::StartTransferRequest& plan,
             absl::Span<const int64_t> src_block_ids, int parallelism) {
        return PoolReshardPush(plan, src_block_ids, parallelism);
      };
  hooks.pool_reshard_register_recv =
      [this](const ::tpu_sync::rpc::StartTransferRequest& plan,
             absl::Span<const int64_t> chip_block_ids) {
        return PoolReshardRegisterRecv(plan, chip_block_ids);
      };
  hooks.wait_for_pending_work = [this]() { return WaitForPendingWork(); };
  hooks.register_active_plan =
      [this](uint64_t uuid,
             const ::tpu_sync::rpc::StartTransferRequest& request,
             bool is_sender) {
        return RegisterActivePlan(uuid, request, is_sender);
      };
  hooks.unregister_active_plan = [this](uint64_t uuid) {
    return UnregisterActivePlan(uuid);
  };
  hooks.get_node_id = [this]() { return node_id(); };
  base_->SetTransferEventHooks(std::move(hooks));
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : base_(std::make_unique<kv_cache::KVCacheManagerBase>(
          layer_buffers, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks),
          unsafe_skip_buffer_lock, parallelism, host_allocator)),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock),
      metrics_collector_(std::move(metrics_collector)) {
  InitializeBaseHooks();
  InitializeControlPlane();
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = base_->ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (base_->num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    // Demand staging allocates per request, so the pool stays whole rather
    // than being carved into fixed slots.
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    size_t slice_byte_size, const std::vector<int64_t>& dimensions,
    size_t physical_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::optional<int> assigned_numa_node,
    std::shared_ptr<MetricsCollector> metrics_collector)
    : base_(std::make_unique<kv_cache::KVCacheManagerBase>(
          layer_buffers, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks),
          unsafe_skip_buffer_lock, parallelism, host_allocator,
          /*bind_ip=*/std::nullopt,
          slice_byte_size > 0 ? std::make_optional(slice_byte_size)
                              : std::nullopt,
          dimensions,
          physical_size > 0 ? std::make_optional(physical_size) : std::nullopt,
          assigned_numa_node)),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock),
      metrics_collector_(std::move(metrics_collector)) {
  InitializeBaseHooks();
  InitializeControlPlane();
  if (base_->num_layers() == 0 || base_->num_shards() == 0) {
    return;
  }
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = base_->ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (base_->num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    // Demand staging allocates per request, so the pool stays whole rather
    // than being carved into fixed slots.
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerWithTransfer(
          num_layers, num_shards,
          std::vector<size_t>(num_layers, slice_byte_size), local_port,
          host_blocks_to_allocate, parallelism, node_id, local_control_port,
          max_blocks, num_slots, timeout_s, std::move(metrics_collector)) {}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    size_t num_layers, size_t num_shards, std::vector<size_t> slice_byte_sizes,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    std::shared_ptr<MetricsCollector> metrics_collector)
    : base_(std::make_unique<kv_cache::KVCacheManagerBase>(
          num_layers, num_shards, std::move(slice_byte_sizes), local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks), parallelism,
          nullptr)),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(false),
      metrics_collector_(std::move(metrics_collector)) {
  InitializeBaseHooks();
  InitializeControlPlane();
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = base_->ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    // Demand staging allocates per request, so the pool stays whole rather
    // than being carved into fixed slots.
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer() {
  StopControlServer();
  // Pull-serve workers read this object's state; nothing may be torn down
  // while one is still running.
  shutting_down_.store(true, std::memory_order_relaxed);
  {
    absl::MutexLock lock(pull_workers_mu_);
    pull_workers_mu_.Await(absl::Condition(
        +[](int* active) { return *active == 0; }, &active_pull_workers_));
  }
  if (base_) {
    base_->StopTransportServer();
    base_->ShutdownTransferPools();
    base_->SetTransferEventHooks({});
  }
  control_backend_.reset();
  control_handler_.reset();
  {
    absl::MutexLock lock(mu_);
    send_entries_.clear();
    active_recv_entries_.clear();
  }
  if (base_->host_block_manager() && !all_slots_.empty()) {
    std::vector<int> blocks_to_unlock;
    blocks_to_unlock.reserve(all_slots_.size() * max_blocks_);
    for (const Slot& slot : all_slots_) {
      for (int block_id : slot.block_ids) {
        blocks_to_unlock.push_back(block_id);
      }
    }
    (void)base_->host_block_manager()->Unlock(blocks_to_unlock);
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    std::unique_ptr<kv_cache::KVCacheManagerBase> base, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : base_(std::move(base)),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(false),
      metrics_collector_(std::move(metrics_collector)) {
  InitializeBaseHooks();
  InitializeControlPlane();
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = base_->ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (base_->num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
}

int64_t KVCacheManagerWithTransfer::NotifyForRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<int64_t>& block_ids,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  RAIDEN_TRACE_FN("KVTransfer::NotifyForRead", [&]() {
    return absl::StrCat("req=", req_id, " uuid=", uuid,
                        " blocks=", block_ids.size());
  });
  const auto register_start = std::chrono::steady_clock::now();
  if (block_ids.empty()) {
    return 0;
  }

  auto entry = std::make_shared<TransferSendSession>(
      base_.get(), req_id, uuid, deadline.value_or(DeadlineFromNow()),
      register_start);
  std::optional<int64_t> duplicate_block =
      entry->PopulateRegisteredBlocks(block_ids);

  {
    absl::MutexLock lock(mu_);
    if (duplicate_block.has_value()) {
      LOG(ERROR) << "NotifyForRead rejected duplicate block "
                 << *duplicate_block << " for req_id=" << req_id
                 << ", uuid=" << uuid;
      return 0;
    }
    if (pending_acks_.erase(uuid) > 0) {
      done_sending_.insert(req_id);
      return 0;
    }
    if (!send_entries_.try_emplace(uuid, entry).second) {
      LOG(ERROR) << "NotifyForRead rejected duplicate uuid=" << uuid
                 << " for req_id=" << req_id;
      return 0;
    }
  }
  cv_.SignalAll();

  std::ostringstream timing;
  timing << "RAIDEN_TIMING event=producer_register"
         << " req_id=" << req_id << " uuid=" << uuid << " node_id=" << node_id_
         << " blocks=" << block_ids.size() << " enqueue_ms="
         << DurationMs(register_start, std::chrono::steady_clock::now())
         << " failed=0";
  EmitTimingLog(timing.str());
  return static_cast<int64_t>(uuid);
}

absl::Status KVCacheManagerWithTransfer::EmplaceRecvEntryLocked(
    uint64_t uuid, const std::shared_ptr<RecvEntry>& entry) {
  auto existing = active_recv_entries_.find(uuid);
  if (existing != active_recv_entries_.end() && existing->second->done()) {
    (existing->second->failed() ? failed_recving_ : done_recving_)
        .insert(existing->second->req_id());
    active_recv_entries_.erase(existing);
  }
  // try_emplace leaves entry untouched on a duplicate. Callers rely on that
  // guarantee to release staging owned by the rejected entry.
  if (!active_recv_entries_.try_emplace(uuid, entry).second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Receive with UUID ", uuid, " is already registered"));
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::RegisterActivePlan(
    uint64_t uuid, const ::tpu_sync::rpc::StartTransferRequest& request,
    bool is_sender) {
  // Registration is one indivisible step: a concurrent unregister or a
  // second registration of the same uuid waits for it, so a plan is never
  // visible without the staging and receive state that belong to it.
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  if (base_->HasActivePlan(uuid)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Plan with UUID ", uuid, " is already registered!"));
  }
  const uint64_t generation = ++plan_generation_counter_;
  // Under demand staging a plan's device blocks are staged in host blocks
  // allocated for the plan, so the host mirror no longer has to span the
  // device block space. Pool-addressed plans keep their own addressing.
  absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
      host_block_of;
  std::vector<int> plan_blocks;
  if (dynamic_host_staging_ && request.pool_groups_size() == 0) {
    std::vector<int64_t> device_blocks;
    absl::flat_hash_set<int64_t> seen;
    for (const auto& [src_shard, schedule] : request.shard_push_schedules()) {
      for (const auto& e : schedule.entries()) {
        int64_t id = is_sender ? e.src_block_id() : e.dst_block_id();
        if (seen.insert(id).second) device_blocks.push_back(id);
      }
    }
    if (!device_blocks.empty()) {
      absl::MutexLock lock(mu_);
      auto allocated = base_->host_block_manager()->Allocate(
          static_cast<int>(device_blocks.size()), /*lock=*/true);
      if (!allocated.ok()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "cannot stage ", device_blocks.size(), " blocks for plan ", uuid,
            ": ", allocated.status().message()));
      }
      plan_blocks = *allocated;
      for (size_t i = 0; i < device_blocks.size(); ++i) {
        host_block_of[device_blocks[i]] = plan_blocks[i];
      }
    }
  }

  // Staging ownership is settled before the plan is published. An HBM
  // receiver's blocks belong to its receive entry and return when the
  // upload settles; a sender's blocks, and a host-memory receiver's,
  // belong to the plan and return when it is unregistered.
  const bool hbm_receiver =
      !is_sender && request.dst_mem_type() == ::tpu_sync::rpc::MEMORY_TYPE_HBM;
  if (!plan_blocks.empty() && !hbm_receiver) {
    absl::MutexLock lock(mu_);
    plan_staging_[uuid] = plan_blocks;
  }

  // 2. If we are the receiver and the destination memory type is HBM,
  //    populate active_recv_entries_ to enable automatic H2D copy!
  if (hbm_receiver) {
    absl::MutexLock lock(mu_);
    absl::flat_hash_set<int> unique_dst_blocks;
    for (const auto& [src_replica_idx, schedule] :
         request.shard_push_schedules()) {
      for (const auto& push_entry : schedule.entries()) {
        unique_dst_blocks.insert(push_entry.dst_block_id());
      }
    }
    std::vector<int64_t> h2d_local_block_ids(unique_dst_blocks.begin(),
                                             unique_dst_blocks.end());
    std::vector<int64_t> h2d_host_block_ids;
    h2d_host_block_ids.reserve(h2d_local_block_ids.size());
    for (int64_t dst : h2d_local_block_ids) {
      auto hb = host_block_of.find(dst);
      h2d_host_block_ids.push_back(hb == host_block_of.end() ? dst
                                                             : hb->second);
    }
    auto recv_entry = std::make_shared<ReceiveSession>(base_.get(), uuid);
    recv_entry->InitFromActivePlan(
        request, host_block_of, std::move(plan_blocks), generation,
        DeadlineFromNow(),
        TransferSendSession::BuildCoalescedCopySpec(h2d_host_block_ids,
                                                    h2d_local_block_ids));

    if (recv_entry->total_blocks() == 0) {
      recv_entry->ReleaseStaging();
    }
    if (recv_entry->total_blocks() > 0) {
      absl::Status inserted = EmplaceRecvEntryLocked(uuid, recv_entry);
      if (!inserted.ok()) {
        recv_entry->ReleaseStaging();
        return inserted;
      }
      LOG(INFO) << "RegisterActivePlan (Receiver): Populated "
                   "active_recv_entries_ for UUID "
                << uuid << " with " << recv_entry->total_blocks()
                << " total physical block-pushes (including duplicates across "
                   "sources) for automatic H2D.";
    }
  }

  // Publish the plan last: pushes resolve through it, so everything they
  // may touch exists by the time it is visible.
  absl::Status registered = base_->RegisterActivePlan(
      uuid, request, is_sender, host_block_of, generation);
  if (!registered.ok()) {
    absl::MutexLock lock(mu_);
    auto staged = plan_staging_.find(uuid);
    if (staged != plan_staging_.end()) {
      (void)base_->host_block_manager()->Unlock(staged->second);
      (void)base_->host_block_manager()->Deallocate(staged->second);
      plan_staging_.erase(staged);
    }
    auto recv = active_recv_entries_.find(uuid);
    if (recv != active_recv_entries_.end()) {
      recv->second->ReleaseStaging();
      active_recv_entries_.erase(recv);
    }
    return registered;
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::RegisterRecv(
    uint64_t uuid, const std::string& req_id, int64_t expected_block_count,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  absl::MutexLock lock(mu_);
  auto recv_entry = std::make_shared<ReceiveSession>(
      base_.get(), uuid, req_id, expected_block_count,
      deadline.value_or(DeadlineFromNow()));
  // host_to_chip is left empty -> defaults to 1-to-1 mapping in
  // OnBlocksReceived
  absl::Status inserted = EmplaceRecvEntryLocked(uuid, recv_entry);
  if (!inserted.ok()) {
    return inserted;
  }
  VLOG(1)
      << "RegisterRecv (Receiver): Registered expected block count for UUID "
      << uuid << " with " << expected_block_count << " expected blocks.";
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::ValidatePoolReshardPlan(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> local_block_ids, bool is_sender) {
  if (plan.req_id().empty()) {
    return absl::InvalidArgumentError("reshard plan req_id must be non-empty");
  }
  if (plan.uuid() <= 0) {
    return absl::InvalidArgumentError("reshard plan uuid must be positive");
  }
  if (!plan.use_block_chunks()) {
    return absl::InvalidArgumentError(
        "pool reshard requires use_block_chunks=true");
  }
  if (plan.transfer_pool_indices().empty()) {
    return absl::InvalidArgumentError(
        "reshard plan must declare transfer_pool_indices");
  }
  if (plan.pool_groups().empty()) {
    return absl::InvalidArgumentError(
        "reshard plans must declare pool_groups (a plan is a list of "
        "groups; entries name their group)");
  }
  for (const auto& group : plan.pool_groups()) {
    if (group.expected_pushes() <= 0) {
      return absl::InvalidArgumentError(
          "every pool group must expect a positive push count");
    }
  }
  if (plan.pool_dtype_tags_size() != static_cast<int>(base_->num_pools())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "reshard plan must declare one dtype tag per pool: plan=",
        plan.pool_dtype_tags_size(), " local=", base_->num_pools()));
  }
  if (local_block_ids.empty()) {
    return absl::InvalidArgumentError("local block ids must not be empty");
  }

  absl::flat_hash_set<int64_t> local_ids(local_block_ids.begin(),
                                         local_block_ids.end());
  // Different groups address different pools, so numerically equal ids
  // across groups are legitimate on both sides. Receiver: the flat list
  // must concatenate the groups' destination runs (uniqueness holds within
  // each group). Sender: the flat list is the union of per-tag source
  // blocks; only bounds are checked here — per-group scoping happens at
  // entry resolution.
  for (int64_t block_id : local_block_ids) {
    if (block_id < 0 || block_id > std::numeric_limits<int>::max()) {
      return absl::InvalidArgumentError(
          "local block ids must be non-negative and fit in int");
    }
  }
  if (!is_sender) {
    size_t cursor = 0;
    for (const auto& group : plan.pool_groups()) {
      absl::flat_hash_set<int64_t> group_ids;
      for (int64_t block_id : group.dst_device_block_ids()) {
        if (!group_ids.insert(block_id).second) {
          return absl::InvalidArgumentError(
              "group destination block ids must be unique");
        }
        if (cursor >= local_block_ids.size() ||
            local_block_ids[cursor] != block_id) {
          return absl::InvalidArgumentError(
              "pool group block ids must concatenate to the plan's "
              "local block ids");
        }
        ++cursor;
      }
    }
    if (cursor != local_block_ids.size()) {
      return absl::InvalidArgumentError(
          "pool group block ids must cover the plan's local block ids");
    }
  }

  // The executor validates the plan's *declared* pool set against this
  // manager's pool table — explicit or implicit — and its geometry. Which
  // tags should move is request data resolved by the controller; no tag name
  // means anything here.
  absl::flat_hash_set<size_t> declared_pools;
  for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
    if (encoded_pool_idx < 0) {
      return absl::InvalidArgumentError(
          "transfer pool index must be non-negative");
    }
    const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
    if (!declared_pools.insert(pool_idx).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate transfer pool index ", pool_idx));
    }
    const kv_cache::PoolSpec* spec = base_->pool(pool_idx);
    if (spec == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("transfer pool index out of range: ", pool_idx));
    }
    if (plan.pool_dtype_tags(pool_idx) != spec->dtype_tag) {
      return absl::InvalidArgumentError(
          absl::StrCat("plan dtype tag mismatch for pool ", pool_idx, " (",
                       spec->tag, "): plan=", plan.pool_dtype_tags(pool_idx),
                       " local=", spec->dtype_tag));
    }
    for (int64_t block_id : local_block_ids) {
      if (block_id >= spec->num_blocks) {
        return absl::InvalidArgumentError(
            absl::StrCat("local block id ", block_id,
                         " is out of range for pool ", pool_idx));
      }
    }
  }

  if (plan.shard_push_schedules().empty()) {
    return absl::InvalidArgumentError(
        "reshard plan must contain shard push schedules");
  }

  size_t entry_count = 0;
  absl::flat_hash_set<int64_t> receiver_blocks_with_zero_start;
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    if (source_rank < 0) {
      return absl::InvalidArgumentError(
          "reshard schedule source rank must be non-negative");
    }
    for (const auto& entry : schedule.entries()) {
      ++entry_count;
      if (entry.dst_peer().empty()) {
        return absl::InvalidArgumentError(
            "reshard entry dst_peer must be non-empty");
      }
      if (entry.src_block_id() < 0 ||
          entry.src_block_id() > std::numeric_limits<int>::max() ||
          entry.dst_block_id() < 0 ||
          entry.dst_block_id() > std::numeric_limits<int>::max() ||
          entry.dst_shard_idx() < 0 || entry.src_offset_bytes() < 0 ||
          entry.dst_offset_bytes() < 0 || entry.size_bytes() <= 0 ||
          entry.src_stride_bytes() < 0 || entry.dst_stride_bytes() < 0 ||
          entry.count() <= 0 || entry.count() > (1 << 20)) {
        return absl::InvalidArgumentError(
            "reshard entry contains invalid ids, offsets, sizes, or strides");
      }
      if (entry.count() > 1 &&
          (entry.src_stride_bytes() == 0 || entry.dst_stride_bytes() == 0)) {
        return absl::InvalidArgumentError(
            "multi-chunk reshard entries require positive strides");
      }
      if (!is_sender &&
          static_cast<size_t>(entry.dst_shard_idx()) >= base_->num_shards()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "destination shard index ", entry.dst_shard_idx(),
            " is out of range: receiver has ", base_->num_shards(), " shards"));
      }
      const int64_t local_id =
          is_sender ? entry.src_block_id() : entry.dst_block_id();
      if (local_ids.find(local_id) == local_ids.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat(is_sender ? "source" : "destination", " block id ",
                         local_id, " is absent from the local block-id list"));
      }
      const int64_t local_offset =
          is_sender ? entry.src_offset_bytes() : entry.dst_offset_bytes();
      const int64_t local_stride =
          is_sender ? entry.src_stride_bytes() : entry.dst_stride_bytes();
      if (!is_sender && entry.dst_offset_bytes() == 0) {
        receiver_blocks_with_zero_start.insert(entry.dst_block_id());
      }
      const int32_t group_idx = entry.pool_group();
      if (group_idx < 0 || group_idx >= plan.pool_groups_size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry declares an unknown pool group ", group_idx));
      }
      std::vector<size_t> entry_pools;
      for (int32_t pool_idx : plan.pool_groups(group_idx).pool_indices()) {
        entry_pools.push_back(static_cast<size_t>(pool_idx));
      }
      for (size_t pool_idx : entry_pools) {
        const kv_cache::PoolSpec* spec = base_->pool(pool_idx);
        if (!StridedSpanFitsRegions(local_offset, local_stride,
                                    entry.size_bytes(), entry.count(),
                                    spec->block_stride_bytes, spec->regions)) {
          return absl::InvalidArgumentError(absl::StrCat(
              is_sender ? "source" : "destination",
              " span exceeds declared pool ", pool_idx,
              " live regions in block ", local_id, ": offset=", local_offset,
              " stride=", local_stride, " size=", entry.size_bytes(), " count=",
              entry.count(), " block_stride_bytes=", spec->block_stride_bytes));
        }
      }
    }
  }
  if (entry_count == 0) {
    return absl::InvalidArgumentError("reshard plan contains no entries");
  }
  if (!is_sender) {
    for (int64_t block_id : local_ids) {
      if (receiver_blocks_with_zero_start.find(block_id) ==
          receiver_blocks_with_zero_start.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "destination block ", block_id,
            " has no transfer entry starting at offset 0; partial-page "
            "destination preservation is not implemented"));
      }
    }
    TF_RETURN_IF_ERROR(ValidatePoolReshardReceiverCoverage(plan));
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::ValidatePoolReshardReceiverCoverage(
    const ::tpu_sync::rpc::StartTransferRequest& plan) {
  constexpr int64_t kMaxExpandedRepeats = 1 << 20;

  struct GroupView {
    std::vector<size_t> pool_indices;
    std::vector<int64_t> dst_ids;
    std::vector<int64_t> extents;
    int64_t expected_pushes = 0;
  };
  std::vector<GroupView> groups;
  absl::flat_hash_set<size_t> grouped_pools;
  for (const auto& group : plan.pool_groups()) {
    GroupView view;
    for (int32_t pool_idx : group.pool_indices()) {
      if (pool_idx < 0 ||
          !grouped_pools.insert(static_cast<size_t>(pool_idx)).second) {
        return absl::InvalidArgumentError(
            "group pool indices must be unique and non-negative");
      }
      view.pool_indices.push_back(static_cast<size_t>(pool_idx));
    }
    view.dst_ids.assign(group.dst_device_block_ids().begin(),
                        group.dst_device_block_ids().end());
    view.extents.assign(group.dst_expected_extent_bytes().begin(),
                        group.dst_expected_extent_bytes().end());
    view.expected_pushes = group.expected_pushes();
    groups.push_back(std::move(view));
  }
  if (grouped_pools.size() !=
      static_cast<size_t>(plan.transfer_pool_indices_size())) {
    return absl::InvalidArgumentError(
        "group pool indices do not partition the plan's transfer pools");
  }
  for (int32_t pool_idx : plan.transfer_pool_indices()) {
    if (!grouped_pools.contains(static_cast<size_t>(pool_idx))) {
      return absl::InvalidArgumentError(
          "group pool indices do not partition the plan's transfer pools");
    }
  }

  const int64_t parallelism = plan.parallelism();
  if (parallelism <= 0) {
    return absl::InvalidArgumentError(
        "receiver plans require positive parallelism for push accounting");
  }

  struct GroupState {
    std::vector<kv_cache::PoolLiveSegment> segments;
    int64_t live_bytes = 0;
    absl::flat_hash_map<int64_t, size_t> ordinals;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> coverage;
    absl::flat_hash_map<
        int32_t, absl::flat_hash_set<std::tuple<std::string, int64_t, int64_t>>>
        pairs_by_sender;
  };
  std::vector<GroupState> states(groups.size());
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const GroupView& view = groups[group_idx];
    GroupState& state = states[group_idx];
    if (view.pool_indices.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("group ", group_idx, " declares no pools"));
    }
    for (size_t pool_idx : view.pool_indices) {
      const kv_cache::PoolSpec* spec = base_->pool(pool_idx);
      if (spec == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("group pool index out of range: ", pool_idx));
      }
      absl::StatusOr<std::vector<kv_cache::PoolLiveSegment>> segments =
          kv_cache::ExpandPoolLiveSegments(*spec);
      if (!segments.ok()) return segments.status();
      if (state.segments.empty()) {
        state.segments = *std::move(segments);
      } else if (state.segments != *segments) {
        return absl::InvalidArgumentError(absl::StrCat(
            "group ", group_idx, " pools must share one live-region map; pool ",
            pool_idx, " disagrees"));
      }
    }
    for (const kv_cache::PoolLiveSegment& segment : state.segments) {
      state.live_bytes += segment.size;
    }
    if (state.live_bytes <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("group ", group_idx, " has no live destination bytes"));
    }
    if (view.extents.empty()) {
      return absl::InvalidArgumentError(
          "receiver plans require dst_expected_extent_bytes");
    }
    if (view.extents.size() != view.dst_ids.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "group ", group_idx,
          " extents do not match its destination block count: got ",
          view.extents.size(), ", expected ", view.dst_ids.size()));
    }
    for (size_t ordinal = 0; ordinal < view.extents.size(); ++ordinal) {
      const int64_t extent = view.extents[ordinal];
      if (extent <= 0 || extent > state.live_bytes) {
        return absl::InvalidArgumentError(
            absl::StrCat("extent ", extent, " for destination block ordinal ",
                         ordinal, " of group ", group_idx, " is outside (0, ",
                         state.live_bytes, "]"));
      }
      if (ordinal != view.extents.size() - 1 && extent != state.live_bytes) {
        return absl::InvalidArgumentError(
            "extents must cover every destination block fully except the "
            "final one");
      }
    }
    for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal) {
      state.ordinals[view.dst_ids[ordinal]] = ordinal;
    }
    state.coverage.resize(view.dst_ids.size());
  }

  int64_t expanded_repeats = 0;
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      const size_t group_idx = static_cast<size_t>(entry.pool_group());
      if (group_idx >= states.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry declares an unknown pool group ", group_idx));
      }
      GroupState& state = states[group_idx];
      const auto ordinal_it = state.ordinals.find(entry.dst_block_id());
      if (ordinal_it == state.ordinals.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry targets destination block ", entry.dst_block_id(),
            " outside its group ", group_idx, " destination set"));
      }
      const int64_t extent = groups[group_idx].extents[ordinal_it->second];
      expanded_repeats += entry.count();
      if (expanded_repeats > kMaxExpandedRepeats) {
        return absl::InvalidArgumentError(
            "receiver plan exceeds the repeat expansion bound");
      }
      for (int64_t repeat = 0; repeat < entry.count(); ++repeat) {
        const int64_t physical =
            entry.dst_offset_bytes() + repeat * entry.dst_stride_bytes();
        absl::StatusOr<std::pair<int64_t, int64_t>> range =
            kv_cache::PhysicalLiveRangeToLogical(state.segments, physical,
                                                 entry.size_bytes());
        if (!range.ok()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "reshard entry for destination block ", entry.dst_block_id(),
              " crosses padding or lies outside declared live regions: ",
              range.status().message()));
        }
        if (range->second > extent) {
          return absl::InvalidArgumentError(absl::StrCat(
              "reshard entry exceeds destination block ", entry.dst_block_id(),
              " declared live tail: end=", range->second, " extent=", extent));
        }
        state.coverage[ordinal_it->second].push_back(*range);
      }
      state.pairs_by_sender[source_rank].insert(std::make_tuple(
          entry.dst_peer(), static_cast<int64_t>(entry.src_block_id()),
          static_cast<int64_t>(entry.dst_block_id())));
    }
  }

  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const GroupView& view = groups[group_idx];
    GroupState& state = states[group_idx];
    int64_t calculated_pushes = 0;
    for (const auto& [source_rank, pairs] : state.pairs_by_sender) {
      calculated_pushes +=
          std::min(parallelism, static_cast<int64_t>(pairs.size()));
    }
    if (calculated_pushes != view.expected_pushes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "expected pushes for group ", group_idx,
          " do not match the received schedules: declared=",
          view.expected_pushes, " recomputed=", calculated_pushes));
    }
    for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal) {
      std::vector<std::pair<int64_t, int64_t>>& intervals =
          state.coverage[ordinal];
      std::sort(intervals.begin(), intervals.end());
      int64_t covered_until = 0;
      for (const auto& [start_bytes, end_bytes] : intervals) {
        if (start_bytes != covered_until) {
          return absl::InvalidArgumentError(absl::StrCat(
              "receiver schedule has a destination coverage ",
              start_bytes < covered_until ? "overlap" : "gap", " for block ",
              view.dst_ids[ordinal], " at byte ", start_bytes));
        }
        covered_until = end_bytes;
      }
      if (covered_until != view.extents[ordinal]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "receiver schedule does not cover the exact live bytes for "
            "destination block ",
            view.dst_ids[ordinal], ": covered=", covered_until,
            " expected=", view.extents[ordinal]));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::PoolReshardPush(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> src_block_ids, int parallelism) {
  RAIDEN_TRACE_FN("KVTransfer::PoolReshardPush", [&]() {
    return absl::StrCat("uuid=", plan.uuid(),
                        " src_blocks=", src_block_ids.size());
  });
  TF_RETURN_IF_ERROR(
      ValidatePoolReshardPlan(plan, src_block_ids, /*is_sender=*/true));
  // Device-only executor: without device attachments there are no bytes this
  // path could legitimately move; host-only managers fail closed with no
  // host-mode branch to mask device bugs.
  if (!base_->has_device_buffers()) {
    return absl::FailedPreconditionError(
        "pool reshard push requires a device-attached manager; host-only "
        "managers are not supported");
  }
  if (parallelism <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }

  auto schedule_it = plan.shard_push_schedules().find(0);
  if (schedule_it == plan.shard_push_schedules().end()) {
    if (plan.shard_push_schedules().size() != 1) {
      return absl::InvalidArgumentError(
          "sender plan must use local schedule key 0");
    }
    schedule_it = plan.shard_push_schedules().begin();
  }
  std::set<std::string> peers;
  for (const auto& entry : schedule_it->second.entries()) {
    peers.insert(entry.dst_peer());
  }
  if (peers.empty()) {
    return absl::InvalidArgumentError("sender plan contains no peers");
  }

  base_->InitTransportServer();
  TF_RETURN_IF_ERROR(
      base_->RegisterActivePlanDirect(plan.uuid(), plan, /*is_sender=*/true));

  auto state = std::make_shared<PoolReshardSendEntry>();
  state->req_id = plan.req_id();
  state->uuid = plan.uuid();
  state->parallelism = parallelism;
  // One completion per (pool, peer-with-entries): with sharded destinations
  // a group may push each of its pools to a single peer, so pools x peers
  // would wait for completions that never come (tpu-sync follow-up on #744).
  state->remaining_pool_peer_pushes =
      static_cast<int>(CountPoolReshardSendSlots(plan, schedule_it->second));
  if (state->remaining_pool_peer_pushes <= 0) {
    (void)base_->UnregisterActivePlanDirect(plan.uuid());
    return absl::InvalidArgumentError("sender plan schedules no pushes");
  }
  state->plan = plan;
  state->deadline = DeadlineFromNow();
  {
    absl::MutexLock lock(mu_);
    if (active_pool_reshard_sends_.contains(plan.uuid())) {
      (void)base_->UnregisterActivePlanDirect(plan.uuid());
      return absl::AlreadyExistsError(
          absl::StrCat("pool reshard send UUID already active: ", plan.uuid()));
    }
    active_pool_reshard_sends_[plan.uuid()] = state;
  }

  // Multi-tag plans scope each pool's staging and pushes to its group's
  // entries; the flat src_block_ids argument is the legacy single-tag
  // whole-plan block list.
  const auto pool_group_index = [&plan](size_t pool_idx) -> int {
    for (int group_idx = 0; group_idx < plan.pool_groups_size(); ++group_idx) {
      const auto& indices = plan.pool_groups(group_idx).pool_indices();
      if (std::find(indices.begin(), indices.end(),
                    static_cast<int32_t>(pool_idx)) != indices.end()) {
        return group_idx;
      }
    }
    return -1;
  };
  auto local_schedule_it = plan.shard_push_schedules().find(0);
  if (local_schedule_it == plan.shard_push_schedules().end() &&
      plan.shard_push_schedules().size() == 1) {
    local_schedule_it = plan.shard_push_schedules().begin();
  }

  for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
    const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
    std::vector<int64_t> pool_src_block_ids(src_block_ids.begin(),
                                            src_block_ids.end());
    if (local_schedule_it != plan.shard_push_schedules().end()) {
      const int group_idx = pool_group_index(pool_idx);
      std::set<int64_t> group_src_ids;
      for (const auto& entry : local_schedule_it->second.entries()) {
        if (entry.pool_group() == group_idx) {
          group_src_ids.insert(static_cast<int64_t>(entry.src_block_id()));
        }
      }
      pool_src_block_ids.assign(group_src_ids.begin(), group_src_ids.end());
      if (pool_src_block_ids.empty()) {
        // This sender owns none of the group's bytes (e.g. a PCP rank whose
        // interleave slices all fall past a short prefix, or a state group
        // this sender routes to no destination): the pool produces no push
        // and CountPoolReshardSendSlots counted no slot for it. The
        // receiver's expected pushes count only senders with scheduled pairs.
        continue;
      }
    }
    // Bounded host staging: lease one arena slot per source page of this
    // pool's storage for the transfer before staging its bytes (no-op on
    // full-mirror storages). Released in FinishPoolReshardSend.
    if (const kv_cache::PoolSpec* pool_spec = base_->pool(pool_idx);
        pool_spec != nullptr) {
      absl::Status lease_status = base_->AcquirePoolStagingLease(
          plan.uuid(), pool_spec->storage_index, pool_src_block_ids,
          base_->pool_staging_lease_timeout());
      if (!lease_status.ok()) {
        FinishPoolReshardSend(plan.uuid(), lease_status);
        return lease_status;
      }
    }
    auto future_or = base_->D2hPoolBlocks(pool_idx, pool_src_block_ids,
                                          /*shard_idx=*/std::nullopt,
                                          static_cast<uint64_t>(plan.uuid()));
    if (!future_or.ok()) {
      FinishPoolReshardSend(plan.uuid(), future_or.status());
      return future_or.status();
    }
    raiden::PjRtCopyFuture future = std::move(future_or).value();
    state->d2h_futures.push_back(future);
    future.OnReady([this, uuid = static_cast<uint64_t>(plan.uuid()),
                    pool_idx](auto status_or) {
      if (!status_or.ok()) {
        FinishPoolReshardSend(uuid, status_or.status());
        return;
      }
      StartPoolReshardPush(uuid, pool_idx);
    });
  }
  return absl::OkStatus();
}

void KVCacheManagerWithTransfer::StartPoolReshardPush(uint64_t uuid,
                                                      size_t pool_idx) {
  RAIDEN_TRACE_FN("KVTransfer::StartPoolReshardPush", [&]() {
    return absl::StrCat("uuid=", uuid, " pool=", pool_idx);
  });
  std::shared_ptr<PoolReshardSendEntry> state;
  {
    absl::MutexLock lock(mu_);
    auto it = active_pool_reshard_sends_.find(uuid);
    if (it == active_pool_reshard_sends_.end()) return;
    state = it->second;
  }

  auto schedule_it = state->plan.shard_push_schedules().find(0);
  if (schedule_it == state->plan.shard_push_schedules().end()) {
    schedule_it = state->plan.shard_push_schedules().begin();
  }
  // A pool pushes only its own group's (src, dst) pairs.
  int pool_group_idx = -1;
  for (int group_idx = 0; group_idx < state->plan.pool_groups_size();
       ++group_idx) {
    const auto& indices = state->plan.pool_groups(group_idx).pool_indices();
    if (std::find(indices.begin(), indices.end(),
                  static_cast<int32_t>(pool_idx)) != indices.end()) {
      pool_group_idx = group_idx;
      break;
    }
  }
  std::map<std::string, std::vector<std::pair<int, int>>> transfers_by_peer;
  std::map<std::string, std::set<std::pair<int, int>>> seen_by_peer;
  for (const auto& entry : schedule_it->second.entries()) {
    if (entry.pool_group() != pool_group_idx) {
      continue;
    }
    const std::pair<int, int> pair{static_cast<int>(entry.src_block_id()),
                                   static_cast<int>(entry.dst_block_id())};
    if (seen_by_peer[entry.dst_peer()].insert(pair).second) {
      transfers_by_peer[entry.dst_peer()].push_back(pair);
    }
  }

  transport::BlockTransport* transport_srv = base_->transport_server();
  if (transport_srv == nullptr) {
    FinishPoolReshardSend(
        uuid, absl::FailedPreconditionError("transport server is not running"));
    return;
  }

  for (const auto& [peer, transfers] : transfers_by_peer) {
    std::vector<int> src_ids;
    std::vector<int> dst_ids;
    src_ids.reserve(transfers.size());
    dst_ids.reserve(transfers.size());
    for (const auto& [src_id, dst_id] : transfers) {
      src_ids.push_back(src_id);
      dst_ids.push_back(dst_id);
    }
    transport_srv->AsyncPush(
        {peer}, src_ids, dst_ids, state->parallelism,
        transport::MajorOrder::kLayerMajor, uuid, static_cast<int>(pool_idx),
        [this, uuid](absl::StatusOr<std::vector<int>> result) {
          FinishPoolReshardSend(
              uuid, result.ok() ? absl::OkStatus() : result.status());
        });
  }
}

void KVCacheManagerWithTransfer::FinishPoolReshardSend(
    uint64_t uuid, const absl::Status& status) {
  RAIDEN_TRACE_FN("KVTransfer::FinishPoolReshardSend", [&]() {
    return absl::StrCat("uuid=", uuid, " status=", status.code());
  });
  bool finished = false;
  {
    absl::MutexLock lock(mu_);
    auto it = active_pool_reshard_sends_.find(uuid);
    if (it == active_pool_reshard_sends_.end()) return;
    auto& state = *it->second;
    if (state.finalizing) return;
    if (!status.ok()) {
      LOG(ERROR) << "Pool reshard send failed uuid=" << uuid
                 << " req_id=" << state.req_id << ": " << status;
      state.failed = true;
      state.finalizing = true;
      finished = true;
    } else if (--state.remaining_pool_peer_pushes == 0) {
      state.finalizing = true;
      finished = true;
    }
  }
  if (finished) {
    absl::Status unregister = UnregisterActivePlan(uuid);
    if (!unregister.ok() && !absl::IsNotFound(unregister)) {
      LOG(ERROR) << "Failed to unregister pool reshard sender plan " << uuid
                 << ": " << unregister;
    }
    // Every push of every pool has completed (or the send failed): the host
    // staging bytes are no longer read, so the arena slots go back.
    base_->ReleasePoolStagingLeases(uuid);
    absl::MutexLock lock(mu_);
    auto it = active_pool_reshard_sends_.find(uuid);
    if (it == active_pool_reshard_sends_.end()) return;
    if (it->second->failed ||
        (!unregister.ok() && !absl::IsNotFound(unregister))) {
      failed_recving_.insert(it->second->req_id);
    } else {
      done_sending_.insert(it->second->req_id);
    }
    active_pool_reshard_sends_.erase(it);
  }
}

absl::Status KVCacheManagerWithTransfer::PoolReshardRegisterRecv(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::PoolReshardRegisterRecv", [&]() {
    return absl::StrCat("uuid=", plan.uuid(),
                        " chip_blocks=", chip_block_ids.size());
  });
  TF_RETURN_IF_ERROR(
      ValidatePoolReshardPlan(plan, chip_block_ids, /*is_sender=*/false));
  // Device-only executor (see PoolReshardPush): arming a receive on a
  // host-only manager is refused rather than silently landing in mirrors.
  if (!base_->has_device_buffers()) {
    return absl::FailedPreconditionError(
        "pool reshard receive requires a device-attached manager; host-only "
        "managers are not supported");
  }
  if (plan.dst_mem_type() != ::tpu_sync::rpc::MEMORY_TYPE_HBM) {
    return absl::InvalidArgumentError(
        "pool reshard receiver requires dst_mem_type=HBM");
  }
  {
    absl::MutexLock lock(mu_);
    auto existing = active_recv_entries_.find(plan.uuid());
    if (existing != active_recv_entries_.end()) {
      if (existing->second->done()) {
        (existing->second->failed() ? failed_recving_ : done_recving_)
            .insert(existing->second->req_id());
        active_recv_entries_.erase(existing);
      } else {
        return absl::AlreadyExistsError(absl::StrCat(
            "pool reshard recv UUID already active: ", plan.uuid()));
      }
    }
  }

  // Bounded host staging: the wire still lands at device (chip) block ids,
  // but on a bounded storage those ids are remapped to arena slots leased to
  // this uuid. Lease the union of every pool's destination ids per storage
  // before arming; a failure here refuses the arm cleanly (the coordinator
  // abandons the claim and no sender is dispatched). Full-mirror storages are
  // no-ops. Released in FinishPoolReshardRecvPool / the deadline sweep.
  {
    std::map<size_t, std::set<int64_t>> dst_ids_by_storage;
    std::map<size_t, std::vector<int64_t>> group_dst_by_pool;
    for (const auto& group : plan.pool_groups()) {
      std::vector<int64_t> group_dst_ids(group.dst_device_block_ids().begin(),
                                         group.dst_device_block_ids().end());
      for (int32_t pool_idx : group.pool_indices()) {
        group_dst_by_pool[static_cast<size_t>(pool_idx)] = group_dst_ids;
      }
    }
    for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
      const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
      const kv_cache::PoolSpec* pool_spec = base_->pool(pool_idx);
      if (pool_spec == nullptr ||
          !base_->PoolStorageStagingBounded(pool_spec->storage_index)) {
        continue;
      }
      auto ids_it = group_dst_by_pool.find(pool_idx);
      const std::vector<int64_t> fallback(chip_block_ids.begin(),
                                          chip_block_ids.end());
      const std::vector<int64_t>& ids =
          ids_it == group_dst_by_pool.end() ? fallback : ids_it->second;
      dst_ids_by_storage[pool_spec->storage_index].insert(ids.begin(),
                                                          ids.end());
    }
    for (const auto& [storage_idx, ids] : dst_ids_by_storage) {
      std::vector<int64_t> id_list(ids.begin(), ids.end());
      absl::Status lease_status =
          base_->AcquirePoolStagingLease(plan.uuid(), storage_idx, id_list,
                                         base_->pool_staging_lease_timeout());
      if (!lease_status.ok()) {
        base_->ReleasePoolStagingLeases(plan.uuid());
        return lease_status;
      }
    }
  }

  absl::Status register_status =
      base_->RegisterActivePlanDirect(plan.uuid(), plan, /*is_sender=*/false);
  if (!register_status.ok()) {
    base_->ReleasePoolStagingLeases(plan.uuid());
    return register_status;
  }
  auto recv_entry = std::make_shared<ReceiveSession>(base_.get(), plan.uuid());
  recv_entry->InitFromPoolReshardPlan(plan, chip_block_ids, DeadlineFromNow());
  {
    absl::MutexLock lock(mu_);
    active_recv_entries_[plan.uuid()] = std::move(recv_entry);
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::UnregisterActivePlan(uint64_t uuid) {
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  bool deferred = false;
  {
    absl::MutexLock lock(mu_);
    auto it = plan_staging_.find(uuid);
    if (it != plan_staging_.end()) {
      (void)base_->host_block_manager()->Unlock(it->second);
      (void)base_->host_block_manager()->Deallocate(it->second);
      plan_staging_.erase(it);
    }
    // A receive still in flight keeps its plan: pushes the transport has
    // already accepted must keep resolving into the plan's staging blocks.
    // The plan is dropped when the receive completes, fails, or times out.
    auto recv = active_recv_entries_.find(uuid);
    if (recv != active_recv_entries_.end() &&
        recv->second->DeferUnregisterOnSettle()) {
      deferred = true;
    }
  }
  if (deferred) return absl::OkStatus();
  return base_->UnregisterActivePlanDirect(uuid);
}

void KVCacheManagerWithTransfer::UnregisterSettledPlan(uint64_t uuid,
                                                       uint64_t generation) {
  // Serialized with registration so cleanup for one registration can never
  // remove a newer one reusing the uuid, or race its progress counters.
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  if (generation != 0) {
    std::optional<uint64_t> current = base_->ActivePlanGeneration(uuid);
    if (!current.has_value() || *current != generation) {
      return;  // the plan is gone, or the uuid already belongs to a newer one
    }
  }
  absl::Status status = base_->UnregisterActivePlanDirect(uuid);
  if (!status.ok() && !absl::IsNotFound(status)) {
    LOG(ERROR) << "Failed to unregister settled transfer plan " << uuid << ": "
               << status;
  }
}

void KVCacheManagerWithTransfer::MaybeUnregisterSettledRecv(
    uint64_t uuid, RecvEntry& session) {
  uint64_t generation = 0;
  if (session.done() && session.TakePendingUnregister(&generation)) {
    UnregisterSettledPlan(uuid, generation);
  }
}

std::vector<RaidenTransferEndpoint>
KVCacheManagerWithTransfer::get_local_endpoints() const {
  // NOTE: prefers the CONTROL port when a control server is running, because
  // StartRead speaks the control protocol. Callers that need the block-
  // transport data protocol (H2hRead/H2dRead pulls, H2hWrite pushes, and
  // therefore worker registration) must use get_local_data_endpoints()
  // instead: aiming a data-protocol connection at the control port hangs both
  // sides with no error, since each waits for the other's framing.
  return BuildEndpoints(local_control_port_ > 0
                            ? local_control_port_
                            : base_->local_port().value_or(0));
}

std::vector<RaidenTransferEndpoint>
KVCacheManagerWithTransfer::get_local_data_endpoints() const {
  return BuildEndpoints(base_->local_port().value_or(0));
}

std::vector<RaidenTransferEndpoint> KVCacheManagerWithTransfer::BuildEndpoints(
    int64_t port) const {
  std::vector<int64_t> all_shards(base_->num_shards());
  for (size_t i = 0; i < base_->num_shards(); ++i) {
    all_shards[i] = static_cast<int64_t>(i);
  }
  std::vector<RaidenTransferEndpoint> eps;
  for (const auto& ip : base_->local_ips()) {
    std::string endpoint = absl::StrContains(ip, ':')
                               ? absl::StrCat("[", ip, "]:", port)
                               : absl::StrCat(ip, ":", port);
    eps.push_back({endpoint, all_shards});
  }
  return eps;
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<std::string>& remote_endpoints,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::StartRead", [&]() {
    return absl::StrCat("req=", req_id, " uuid=", uuid,
                        " blocks=", remote_block_ids.size());
  });
  std::string target_ep;
  if (!remote_endpoints.empty()) {
    target_ep = remote_endpoints[0];
  }
  StartRead(req_id, uuid, target_ep, remote_block_ids, local_block_ids,
            parallelism, local_host_block_ids);
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::StartRead", [&]() {
    return absl::StrCat("req=", req_id, " uuid=", uuid,
                        " blocks=", remote_block_ids.size());
  });
  if (remote_descriptors.empty()) {
    return;
  }
  // TODO: Deal with the case where the shards on both sides don't
  // perfectly match. KVCacheManagerWithTransfer is bound to a single NUMA node
  // / single endpoint. Multi-endpoint routing across sockets is orchestrated by
  // the JAX facade.
  if (remote_descriptors.size() != 1) {
    VLOG(1) << "KVCacheManagerWithTransfer::StartRead received "
            << remote_descriptors.size()
            << " descriptors, selecting first endpoint.";
  }
  StartRead(req_id, uuid, remote_descriptors[0].endpoint, remote_block_ids,
            local_block_ids, parallelism, local_host_block_ids);
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  StartRead(req_id, uuid, remote_endpoint, remote_block_ids, local_block_ids,
            parallelism, std::move(local_host_block_ids), DeadlineFromNow());
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids,
    std::chrono::steady_clock::time_point deadline) {
  LOG(INFO) << "StartRead (initiate): req_id=" << req_id << ", uuid=" << uuid
            << ", numa=" << base_->assigned_numa_node().value_or(-1);
  VLOG(1) << "KVCacheManagerWithTransfer::StartRead (Hybrid Bridge) called. "
             "req_id: "
          << req_id << ", uuid: " << uuid << ", remote: " << remote_endpoint
          << ", Thread: " << std::this_thread::get_id();
  if (remote_block_ids.size() != local_block_ids.size() ||
      (local_host_block_ids.has_value() &&
       local_host_block_ids->size() != local_block_ids.size())) {
    throw std::invalid_argument(
        "remote_block_ids, local_block_ids, and local_host_block_ids must have "
        "same length");
  }
  // local_block_ids index the consumer's DEVICE KV cache, not the host staging
  // pool; reusing them as host indices overflows the host buffer once a device
  // block id exceeds num_host_blocks. If the caller didn't supply explicit host
  // indices, borrow a staging slot and stage into its reserved host blocks
  // (slot.block_ids -- the real, possibly non-contiguous host blocks).
  CopyPlan load_plan;
  auto entry = std::make_shared<ReceiveSession>(base_.get(), uuid);
  {
    absl::MutexLock lock(mu_);
    auto incumbent = active_recv_entries_.find(uuid);
    if (incumbent != active_recv_entries_.end()) {
      if (incumbent->second->done()) {
        (incumbent->second->failed() ? failed_recving_ : done_recving_)
            .insert(incumbent->second->req_id());
        active_recv_entries_.erase(incumbent);
      } else {
        LOG(ERROR) << "StartRead rejected duplicate uuid=" << uuid
                   << " for req_id=" << req_id;
        // Re-delivery of the same announcement is idempotent. The incumbent
        // remains responsible for producing this request's one terminal report.
        if (incumbent->second->req_id() != req_id) {
          failed_recving_.insert(req_id);
        }
        return;
      }
    }

    std::unique_ptr<Slot> slot;
    std::vector<int> staged_blocks;
    std::vector<int64_t> host_block_ids;
    if (local_host_block_ids.has_value()) {
      host_block_ids = *local_host_block_ids;
    } else if (!local_block_ids.empty()) {
      absl::flat_hash_set<int64_t> unique_local_bids(local_block_ids.begin(),
                                                     local_block_ids.end());
      auto staged = AcquireRecvStagingLocked(
          static_cast<int64_t>(unique_local_bids.size()), &slot,
          &staged_blocks);
      if (!staged.has_value()) {
        // Request larger than the staging pool can seat: surface as a recv
        // failure (the connector can recompute) rather than throwing.
        LOG(ERROR) << "StartRead: cannot stage " << unique_local_bids.size()
                   << " blocks for req_id=" << req_id
                   << " (dynamic=" << dynamic_host_staging_
                   << ", free_host_blocks="
                   << base_->host_block_manager()->num_free_blocks()
                   << ", free_slots=" << free_slots_.size()
                   << ", max_blocks=" << max_blocks_ << ")";
        failed_recving_.insert(req_id);
        return;
      }
      absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
          local_to_host;
      size_t host_block_idx = 0;
      host_block_ids.reserve(local_block_ids.size());
      for (size_t k = 0; k < local_block_ids.size(); ++k) {
        int64_t local_bid = local_block_ids[k];
        auto it = local_to_host.find(local_bid);
        if (it == local_to_host.end()) {
          int64_t host_bid = (*staged)[host_block_idx++];
          local_to_host[local_bid] = host_bid;
          host_block_ids.push_back(host_bid);
        } else {
          host_block_ids.push_back(it->second);
        }
      }
    }
    load_plan =
        BuildLoadCopyPlan(remote_block_ids, local_block_ids, host_block_ids);

    entry->InitFromLoadPlan(req_id, load_plan, deadline, std::move(slot),
                            std::move(staged_blocks));
    absl::Status inserted = EmplaceRecvEntryLocked(uuid, entry);
    if (!inserted.ok()) {
      entry->ReleaseStaging();
      failed_recving_.insert(req_id);
      LOG(ERROR) << "StartRead failed to register req_id=" << req_id
                 << ", uuid=" << uuid << ": " << inserted.message();
      return;
    }
  }

  if (metrics_collector_) {
    uint64_t total_bytes = static_cast<uint64_t>(load_plan.num_blocks) *
                           base_->num_layers() * base_->num_shards() *
                           base_->slice_byte_size();
    metrics_collector_->RecordStart(uuid, req_id, load_plan.num_blocks,
                                    total_bytes);
  }

  if (load_plan.num_blocks == 0) {
    entry->FinishRecv(/*has_failed=*/false);
    return;
  }

  entry->ExecutePullRequest(*this, remote_endpoint, std::move(load_plan));
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheManagerWithTransfer::CompleteReadRaw() {
  RAIDEN_TRACE("KVTransfer::CompleteReadRaw");
  std::vector<std::string> done_sending;
  std::vector<std::string> done_recving;
  std::vector<std::string> failed_recving;
  std::vector<std::pair<uint64_t, uint64_t>> settled_plans;
  {
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto it = send_entries_.begin(); it != send_entries_.end();) {
      const std::shared_ptr<SendEntry> entry = it->second;
      const uint64_t uuid = it->first;
      if (!entry->draining() && entry->deadline() <= now) {
        // Past its deadline the transfer failed. One nobody pulled is
        // reported now; one whose copies or pushes still run keeps its
        // staging until they end, so the next transfer is never seated on
        // memory a copy still writes.
        entry->FinishSend(/*has_failed=*/true);
      }
      if (entry->done()) {
        (entry->failed() ? failed_recving_ : done_sending_)
            .insert(entry->req_id());
        if (entry->failed()) {
          settled_plans.emplace_back(uuid, 0);
        }
        it = send_entries_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = active_pool_reshard_sends_.begin();
         it != active_pool_reshard_sends_.end();) {
      const auto& entry = it->second;
      if (entry->deadline <= now) {
        failed_recving_.insert(entry->req_id);
        settled_plans.emplace_back(it->first, 0);
        auto erase_it = it++;
        active_pool_reshard_sends_.erase(erase_it);
      } else {
        ++it;
      }
    }
    // Reclaim recv entries whose transfer never completed (e.g. the producer
    // died or never finished pushing). Without this the entry and its host
    // staging slot leak forever, eventually exhausting the slot pool. Surface
    // the timeout as a recv failure so the connector can recompute the blocks.
    for (auto it = active_recv_entries_.begin();
         it != active_recv_entries_.end();) {
      const uint64_t uuid = it->first;
      const std::shared_ptr<ReceiveSession> entry = it->second;
      if (!entry->draining()) {
        if (entry->IsReadyToComplete()) {
          LOG(INFO) << "CompleteReadRaw (polling completion): req_id="
                    << entry->req_id();
          entry->FinishRecv(/*has_failed=*/false);
        } else if (entry->deadline() <= now) {
          // Preserve the pre-existing timeout cleanup behavior even for
          // receives registered without a plan: UnregisterSettledPlan also
          // clears any transport-side progress associated with the UUID.
          entry->FinishRecv(/*has_failed=*/true,
                            /*unregister_on_settle=*/true);
        }
      }

      if (entry->done()) {
        (entry->failed() ? failed_recving_ : done_recving_)
            .insert(entry->req_id());
        uint64_t generation = 0;
        if (entry->TakePendingUnregister(&generation)) {
          settled_plans.emplace_back(uuid, generation);
        }
        active_recv_entries_.erase(it++);
      } else {
        ++it;
      }
    }
    done_sending.assign(done_sending_.begin(), done_sending_.end());
    done_recving.assign(done_recving_.begin(), done_recving_.end());
    failed_recving.assign(failed_recving_.begin(), failed_recving_.end());
    done_sending_.clear();
    done_recving_.clear();
    failed_recving_.clear();
  }
  // Unregistering drops the plan and its transport receive-progress counters
  // (ForgetPushProgress), so a settled uuid is reusable.
  for (const auto& [uuid, generation] : settled_plans) {
    UnregisterSettledPlan(uuid, generation);
    // A settled (completed or timed-out) pool-reshard sender/receiver may
    // still hold bounded-staging arena slots.
    base_->ReleasePoolStagingLeases(uuid);
  }
  return {done_sending, done_recving, failed_recving};
}

StageResult KVCacheManagerWithTransfer::IssueH2D(
    int64_t slot_idx, int64_t num_blocks,
    const std::vector<int64_t>& local_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::IssueH2D", [&]() {
    return absl::StrCat("slot=", slot_idx, " blocks=", num_blocks);
  });
  if (base_->num_layers() == 0) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (slot_idx < 0 || slot_idx >= num_slots_) {
    throw std::out_of_range("slot_idx out of range");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
    throw std::invalid_argument("num_blocks must match len(local_block_ids)");
  }

  // Get the actual host block IDs for the first num_blocks in the slot
  const Slot& slot = all_slots_[slot_idx];
  std::vector<int64_t> host_block_ids;
  host_block_ids.reserve(num_blocks);
  for (int64_t i = 0; i < num_blocks; ++i) {
    host_block_ids.push_back(slot.block_ids[i]);
  }

  // Coalesce contiguous (host, device) block runs
  CopySpec copy_spec = TransferSendSession::BuildCoalescedCopySpec(
      host_block_ids, local_block_ids);
  kv_cache::KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);

  // We still calculate host_spans for the result, but we don't use slot_idx
  // in H2d call to avoid slot-based double offsetting in the base class.
  std::vector<kv_cache::KVCacheHostSpan> host_spans =
      LayerSpans(slot_idx, num_blocks);

  auto future = std::make_shared<TransferFuture>();
  int64_t total_bytes = 0;
  for (const kv_cache::KVCacheHostSpan& span : host_spans) {
    total_bytes += static_cast<int64_t>(span.nbytes);
  }

  // Call H2dSyncDispatch with slot_idx = std::nullopt to use actual host block
  // IDs
  auto fut_or = base_->H2dSyncDispatch(
      transfer_spec.src_offsets, transfer_spec.dst_offsets, transfer_spec.sizes,
      /*slot_idx=*/std::nullopt);
  if (!fut_or.ok()) {
    throw std::runtime_error("Failed to issue H2D transfer: " +
                             std::string(fut_or.status().message()));
  }
  future->Add(std::move(fut_or.value()));

  return {.future = std::move(future),
          .host_spans = std::move(host_spans),
          .total_bytes = total_bytes,
          .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
}

std::vector<kv_cache::KVCacheHostSpan> KVCacheManagerWithTransfer::LayerSpans(
    int64_t slot_idx, int64_t num_blocks) {
  if (base_->num_layers() == 0) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  std::vector<kv_cache::KVCacheHostSpan> spans;
  const Slot& slot = all_slots_[slot_idx];

  // Coalesce contiguous runs of block IDs in the slot
  struct Run {
    int64_t start_block_id;
    int64_t size;
  };
  std::vector<Run> runs;
  for (int64_t start = 0; start < num_blocks;) {
    int64_t end = start + 1;
    while (end < num_blocks &&
           slot.block_ids[end] == slot.block_ids[end - 1] + 1) {
      ++end;
    }
    runs.push_back({slot.block_ids[start], end - start});
    start = end;
  }

  spans.reserve(base_->num_layers() * base_->num_shards() * runs.size());
  for (size_t layer_idx = 0; layer_idx < base_->num_layers(); ++layer_idx) {
    const int64_t per_layer = base_->LayerBlockByteSize(layer_idx);
    const size_t layer_bytes = per_layer > 0 ? static_cast<size_t>(per_layer)
                                             : base_->slice_byte_size();
    for (size_t shard_idx = 0; shard_idx < base_->num_shards(); ++shard_idx) {
      uint8_t* host_ptr = base_->GetHostPointer(layer_idx, shard_idx);
      for (const auto& run : runs) {
        const size_t byte_offset =
            static_cast<size_t>(run.start_block_id) * layer_bytes;
        const size_t nbytes = static_cast<size_t>(run.size) * layer_bytes;
        spans.push_back(
            kv_cache::KVCacheHostSpan{.ptr = host_ptr + byte_offset,
                                      .nbytes = nbytes,
                                      .slot_idx = slot_idx,
                                      .base_major = run.start_block_id,
                                      .num_major = run.size,
                                      .layer_idx = layer_idx,
                                      .shard_idx = shard_idx});
      }
    }
  }
  return spans;
}

absl::Status KVCacheManagerWithTransfer::InitializeSlotPool(int64_t num_slots) {
  if (base_->host_block_manager()->num_free_blocks() <
      num_slots * max_blocks_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Insufficient free host blocks to initialize slot pool. Required: ",
        num_slots * max_blocks_,
        ", Available: ", base_->host_block_manager()->num_free_blocks()));
  }
  absl::MutexLock lock(slot_mu_);
  free_slots_.clear();
  all_slots_.clear();
  all_slots_.reserve(num_slots);
  for (int64_t i = 0; i < num_slots; ++i) {
    ABSL_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                          base_->host_block_manager()->Allocate(max_blocks_,
                                                                /*lock=*/true));
    if (allocated_ids.size() != max_blocks_) {
      return absl::InternalError(absl::StrCat(
          "Slot pool allocation returned incorrect number of blocks: ",
          allocated_ids.size(), ", expected: ", max_blocks_));
    }
    all_slots_.push_back(
        Slot{/*slot_idx=*/i, /*block_ids=*/allocated_ids, /*manager=*/nullptr});
    free_slots_.push_back(Slot{/*slot_idx=*/i,
                               /*block_ids=*/std::move(allocated_ids),
                               /*manager=*/nullptr});
  }
  return absl::OkStatus();
}

KVCacheManagerWithTransfer::Slot::~Slot() { Reset(); }

void KVCacheManagerWithTransfer::Slot::Reset() {
  if (manager != nullptr && slot_idx >= 0) {
    manager->ReleaseSlotLocked(slot_idx);
  }
  manager = nullptr;
  slot_idx = -1;
  block_ids.clear();
}

bool KVCacheManagerWithTransfer::DynamicHostStagingEnabled() {
  const char* raw = std::getenv("TPU_RAIDEN_DYNAMIC_HOST_STAGING");
  return raw != nullptr && std::string(raw) == "1";
}

std::optional<std::vector<int64_t>>
KVCacheManagerWithTransfer::AcquireRecvStagingLocked(
    int64_t num_blocks, std::unique_ptr<Slot>* slot_out,
    std::vector<int>* staged_blocks_out) {
  if (num_blocks <= 0) return std::vector<int64_t>();
  if (!dynamic_host_staging_) {
    {
      absl::MutexLock slot_lock(slot_mu_);
      if (num_blocks > max_blocks_ || free_slots_.empty()) return std::nullopt;
    }
    auto slot = std::make_unique<Slot>(AcquireSlotLocked());
    std::vector<int64_t> blocks;
    blocks.reserve(num_blocks);
    for (int64_t i = 0; i < num_blocks; ++i) {
      blocks.push_back(slot->block_ids[i]);
    }
    *slot_out = std::move(slot);
    return blocks;
  }
  // Lock the pages so the pool's LRU cannot evict staging that is mid-flight.
  // An allocation miss is only reported back; this helper neither retries
  // nor logs. The producer retries it until its deadline, the consumer
  // fails it at once, and each logs its own verdict.
  auto allocated =
      base_->host_block_manager()->Allocate(static_cast<int>(num_blocks),
                                            /*lock=*/true);
  if (!allocated.ok()) {
    return std::nullopt;
  }
  *staged_blocks_out = *allocated;
  std::vector<int64_t> blocks;
  blocks.reserve(allocated->size());
  for (int id : *allocated) blocks.push_back(static_cast<int64_t>(id));
  return blocks;
}

KVCacheManagerWithTransfer::Slot KVCacheManagerWithTransfer::AcquireSlot() {
  return AcquireSlotLocked();
}

KVCacheManagerWithTransfer::Slot
KVCacheManagerWithTransfer::AcquireSlotLocked() {
  absl::MutexLock lock(slot_mu_);
  if (free_slots_.empty()) {
    throw std::runtime_error("Raiden host slot pool exhausted");
  }
  Slot slot = std::move(free_slots_.front());
  free_slots_.pop_front();
  slot.manager = this;
  return slot;
}

std::unique_ptr<KVCacheManagerWithTransfer::Slot>
KVCacheManagerWithTransfer::TryAcquireSlot() {
  absl::MutexLock lock(slot_mu_);
  if (free_slots_.empty()) {
    return nullptr;
  }
  auto slot = std::make_unique<Slot>(std::move(free_slots_.front()));
  free_slots_.pop_front();
  slot->manager = this;
  return slot;
}

size_t KVCacheManagerWithTransfer::num_free_slots() const {
  absl::MutexLock lock(slot_mu_);
  return free_slots_.size();
}

void KVCacheManagerWithTransfer::ReleaseSlotLocked(int64_t slot_idx) {
  absl::MutexLock lock(slot_mu_);
  if (slot_idx < 0 || slot_idx >= num_slots_) {
    return;
  }
  const Slot& prototype = all_slots_[slot_idx];
  free_slots_.push_back(Slot{/*slot_idx=*/prototype.slot_idx,
                             /*block_ids=*/prototype.block_ids,
                             /*manager=*/nullptr});
}

std::shared_ptr<KVCacheManagerWithTransfer::StagingReadinessState>
KVCacheManagerWithTransfer::CreateStagingReadiness(int64_t slot_idx,
                                                   int64_t num_blocks) {
  auto state = std::make_shared<StagingReadinessState>();
  state->slot_idx = slot_idx;
  state->num_blocks = num_blocks;
  state->num_layers = base_->num_layers();
  state->num_shards = base_->num_shards();
  state->layers.resize(state->num_layers * state->num_shards);
  {
    absl::MutexLock lock(mu_);
    staging_readiness_[slot_idx] = state;
  }
  return state;
}

void KVCacheManagerWithTransfer::MarkStagingLayerReady(
    const std::shared_ptr<StagingReadinessState>& state, size_t layer_idx,
    size_t shard_idx, absl::Status status) {
  if (!state) return;
  const size_t layer_state_idx = layer_idx * state->num_shards + shard_idx;
  if (layer_state_idx >= state->layers.size()) return;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    state->layers[layer_state_idx].done = true;
    state->layers[layer_state_idx].status = std::move(status);
  }
  state->cv.notify_all();
}

void KVCacheManagerWithTransfer::RemoveStagingReadinessLocked(
    int64_t slot_idx) {
  auto it = staging_readiness_.find(slot_idx);
  if (it == staging_readiness_.end()) return;
  std::shared_ptr<StagingReadinessState> state = it->second;
  staging_readiness_.erase(it);
  {
    std::lock_guard<std::mutex> state_lock(state->mu);
    for (StagingLayerReady& layer : state->layers) {
      if (!layer.done) {
        layer.done = true;
        layer.status = absl::CancelledError("staging slot was released");
      }
    }
  }
  state->cv.notify_all();
}

void KVCacheManagerWithTransfer::RegisterBlockReadinessCallback(
    size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
    transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
  if (block_id < 0 || max_blocks_ <= 0) {
    cb(absl::OkStatus());
    return;
  }
  if (uuid == kLeaseAuthorizedPullUuid) {
    // A lease-authorised pull. The reader holds a lease over blocks this node
    // already published as host-resident, so their device-to-host copy
    // provably completed before the lease was granted, and the pin keeps them
    // from being reused for the duration of the read. Gating here would add
    // nothing -- and would be actively wrong, because the fallback scan below
    // can only match some OTHER transfer's entry, making this read wait on a
    // future that has nothing to do with it.
    cb(absl::OkStatus());
    return;
  }
  std::shared_ptr<TransferSendSession> entry;
  {
    absl::MutexLock lock(mu_);
    // Exact match: the transfer named itself, so gate on its own D2H. Reached
    // by uuid-carrying senders; a pull that did not identify itself falls
    // through to the scan below.
    auto it = send_entries_.find(uuid);
    if (it != send_entries_.end() && !it->second->done()) {
      entry = it->second;
    } else {
      // Fallback: no entry owns this uuid, so look for any live transfer that
      // registered this block id and wait on ITS copy. Conservative and
      // imprecise -- the match is by block id alone, so an unrelated transfer
      // can gate this one, and a stale entry whose future never resolves would
      // stall it until the reader's own deadline fires.
      for (const auto& [u, e] : send_entries_) {
        if (!e->done() && e->OwnsBlockWithReadyFuture(block_id, layer_idx)) {
          entry = e;
          break;
        }
      }
    }
  }
  if (!entry) {
    cb(absl::OkStatus());
    return;
  }
  entry->RegisterLayerReadinessCallback(layer_idx, std::move(cb));
}

class KVCacheManagerWithTransfer::ControlPlaneHandlerImpl
    : public ControlPlaneHandler {
 public:
  explicit ControlPlaneHandlerImpl(KVCacheManagerWithTransfer* manager)
      : manager_(manager) {}

  absl::StatusOr<PullStreamResponseSpec> OnPullStream(
      const PullStreamRequestSpec& req,
      absl::string_view fallback_peer_ip) override {
    return manager_->HandlePullStream(req, fallback_peer_ip);
  }

  absl::Status OnAck(uint64_t uuid) override {
    return manager_->HandleAck(uuid);
  }

  uint64_t MaxPullStreamBlocks() const override {
    return manager_->MaxPullStreamBlocks();
  }

 private:
  KVCacheManagerWithTransfer* manager_;
};

void KVCacheManagerWithTransfer::InitializeControlPlane() {
  control_handler_ = std::make_unique<ControlPlaneHandlerImpl>(this);
  auto executor = [this](std::function<void()> task) {
    base_->pull_pool()->Schedule(base_->assigned_numa_node(), std::move(task));
  };
  control_backend_ =
      CreateControlPlaneBackend(ResolveControlPlaneBackendType(),
                                std::move(executor), absl::Seconds(timeout_s_));
}

void KVCacheManagerWithTransfer::StartControlServer() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = false;
  }
  absl::StatusOr<int> bound_port = control_backend_->StartServer(
      local_control_port_, control_handler_.get());
  CheckStatus("StartControlServer", bound_port.status());
  local_control_port_ = *bound_port;
}

void KVCacheManagerWithTransfer::StopControlServer() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
    while (!staging_readiness_.empty()) {
      RemoveStagingReadinessLocked(staging_readiness_.begin()->first);
    }
  }
  // Wake workers parked in HandlePullStream waiting for a send entry that
  // will never arrive, so their loops observe stopping_ and exit.
  cv_.SignalAll();
  if (control_backend_) {
    control_backend_->StopServer();
  }
}

absl::StatusOr<PullStreamResponseSpec>
KVCacheManagerWithTransfer::HandlePullStream(
    const PullStreamRequestSpec& req, absl::string_view fallback_peer_ip) {
  RAIDEN_TRACE_FN("KVTransfer::HandlePullStream", [&]() {
    return absl::StrCat("uuid=", req.uuid,
                        " blocks=", req.src_block_ids.size());
  });
  try {
    const uint64_t block_capacity = MaxPullStreamBlocks();
    if (req.src_block_ids.size() > block_capacity) {
      throw std::invalid_argument(
          absl::StrCat("pull stream block count ", req.src_block_ids.size(),
                       " exceeds configured maximum ", block_capacity));
    }

    const absl::Duration grace =
        std::min(kPullRegistrationGrace, absl::Seconds(timeout_s_));
    std::shared_ptr<TransferSendSession> entry;
    {
      absl::MutexLock lock(mu_);
      const absl::Time give_up = absl::Now() + grace;
      while (true) {
        auto it = send_entries_.find(req.uuid);
        if (it != send_entries_.end()) {
          entry = it->second;
          break;
        }
        const absl::Duration left = give_up - absl::Now();
        if (stopping_.load() || left <= absl::ZeroDuration()) {
          break;
        }
        cv_.WaitWithTimeout(&mu_, left);
      }
      if (stopping_.load()) {
        return PullStreamResponseSpec{
            .status = -1, .message = "Producer control server is stopping"};
      }
      if (!entry) {
        throw std::runtime_error(
            absl::StrCat("no read registered for uuid ", req.uuid, " within ",
                         absl::FormatDuration(grace),
                         ": the producer expired it or never registered it"));
      }
      // Registration guards prevent a live entry from being replaced, so an
      // expired entry cannot become valid while this pull waits out the grace.
      entry->ValidateAndBeginPull(req.src_block_ids,
                                  std::chrono::steady_clock::now());
    }

    std::vector<std::string> peer_ips = req.consumer_ips;
    if (peer_ips.empty()) {
      if (control_backend_->Name() != "tcp") {
        LOG(WARNING) << "No consumer IPs specified in PullStreamRequest.";
      }
      if (!fallback_peer_ip.empty()) {
        peer_ips.push_back(std::string(fallback_peer_ip));
      }
    }

    std::vector<std::string> remote_data_endpoints;
    for (const auto& peer_ip : peer_ips) {
      if (absl::StrContains(peer_ip, ':')) {
        remote_data_endpoints.push_back(
            absl::StrCat("[", peer_ip, "]:", req.consumer_data_port));
      } else {
        remote_data_endpoints.push_back(
            absl::StrCat(peer_ip, ":", req.consumer_data_port));
      }
    }

    VLOG(1) << "HandlePullStream (Hybrid Bridge) successfully acknowledged "
               "consumer. Intercepting and launching StartPush to "
            << (remote_data_endpoints.empty() ? "" : remote_data_endpoints[0])
            << (remote_data_endpoints.size() > 1 ? " and others" : "");

    {
      absl::MutexLock lock(pull_workers_mu_);
      ++active_pull_workers_;
    }
    std::thread([this, entry, remote_data_endpoints,
                 src_block_ids = req.src_block_ids,
                 dst_block_ids = req.dst_block_ids]() {
      entry->StartPush(*this, remote_data_endpoints, src_block_ids,
                       dst_block_ids);
      absl::MutexLock lock(pull_workers_mu_);
      --active_pull_workers_;
    }).detach();

    return PullStreamResponseSpec{
        .status = 0,
        .num_layers =
            static_cast<uint32_t>(base_->num_layers() * base_->num_shards()),
        .data_port = static_cast<uint32_t>(local_data_port_),
        .message = "",
    };
  } catch (const std::exception& e) {
    LOG(ERROR) << "Raiden producer rejected PullStream request: " << e.what();
    return PullStreamResponseSpec{
        .status = -1,
        .num_layers = 0,
        .data_port = 0,
        .message = e.what(),
    };
  }
}

absl::Status KVCacheManagerWithTransfer::HandleAck(uint64_t uuid) {
  AckSend(uuid);
  return absl::OkStatus();
}

uint64_t KVCacheManagerWithTransfer::MaxPullStreamBlocks() const {
  const int64_t block_capacity =
      dynamic_host_staging_ ? base_->host_block_manager()->total_blocks()
                            : max_blocks_;
  return static_cast<uint64_t>(std::max<int64_t>(0, block_capacity));
}

absl::Status KVCacheManagerWithTransfer::WaitForPendingWork() {
  RAIDEN_TRACE("KVTransfer::WaitForPendingWork");
  LOG(INFO) << "Waiting for pending transfer work to complete...";
  const absl::Time start = absl::Now();
  while (true) {
    {
      absl::MutexLock lock(mu_);
      bool recv_pending = false;
      for (const auto& [uuid, entry] : active_recv_entries_) {
        (void)uuid;
        if (entry->HasPendingWork()) {
          recv_pending = true;
          break;
        }
      }
      if (!recv_pending && active_pool_reshard_sends_.empty()) {
        break;
      }
      const absl::Duration elapsed = absl::Now() - start;
      if (elapsed > kPendingWorkTimeout) {
        return absl::DeadlineExceededError(
            "Timeout waiting for pending transfer work");
      }
    }
    absl::SleepFor(absl::Milliseconds(100));
  }
  LOG(INFO) << "All pending transfer work completed.";
  return absl::OkStatus();
}

std::string KVCacheManagerWithTransfer::EndpointWithPort(
    const std::string& endpoint, int port) const {
  if (endpoint.empty()) {
    throw std::invalid_argument("endpoint is empty");
  }
  std::string host;
  if (absl::StartsWith(endpoint, "[")) {
    size_t closing = endpoint.find("]:");
    if (closing == std::string::npos) {
      throw std::invalid_argument("invalid IPv6 endpoint: " + endpoint);
    }
    host = endpoint.substr(1, closing - 1);
  } else {
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("endpoint must be host:port");
    }
    host = endpoint.substr(0, colon);
  }
  if (absl::StrContains(host, ':')) {
    return absl::StrCat("[", host, "]:", port);
  }
  return absl::StrCat(host, ":", port);
}

void KVCacheManagerWithTransfer::AckRemote(const std::string& remote_endpoint,
                                           uint64_t uuid) {
  CheckStatus("AckRemote",
              control_backend_->SendAck(remote_endpoint, uuid,
                                        absl::Seconds(timeout_s_)));
}

void KVCacheManagerWithTransfer::AckSend(uint64_t uuid) {
  std::shared_ptr<SendEntry> entry;
  {
    absl::MutexLock lock(mu_);
    auto it = send_entries_.find(uuid);
    if (it == send_entries_.end()) {
      pending_acks_.insert(uuid);
      return;
    }
    entry = it->second;
  }
  entry->FinishSend(/*has_failed=*/false);
  const auto ack_done = std::chrono::steady_clock::now();
  std::ostringstream timing;
  timing << "RAIDEN_TIMING event=producer_ack"
         << " req_id=" << entry->req_id() << " uuid=" << entry->uuid()
         << " node_id=" << node_id_ << " blocks=" << entry->num_blocks()
         << " bytes=" << entry->total_bytes()
         << " stage_to_ack_ms=" << DurationMs(entry->d2h_done(), ack_done)
         << " register_to_ack_ms="
         << DurationMs(entry->register_start(), ack_done)
         << " failed=" << (entry->failed() ? 1 : 0);
  EmitTimingLog(timing.str());
}

std::chrono::steady_clock::time_point
KVCacheManagerWithTransfer::DeadlineFromNow() const {
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(static_cast<int64_t>(timeout_s_ * 1000.0));
}

void KVCacheManagerWithTransfer::ConfigureDataPortFromKvTransfer() {
  if (base_->num_layers() == 0) {
    local_data_port_ = 0;
    return;
  }
  std::optional<int> data_port = base_->local_port();
  if (!data_port.has_value()) {
    throw std::runtime_error("KVCacheManager BlockTransport is not running");
  }
  local_data_port_ = *data_port;
}

std::vector<int> KVCacheManagerWithTransfer::ContiguousBlockIds(
    uint64_t base, uint64_t count) const {
  if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::out_of_range("block count exceeds int range");
  }
  if (base > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      count >
          static_cast<uint64_t>(std::numeric_limits<int>::max()) - base + 1) {
    throw std::out_of_range("block id range exceeds int range");
  }
  std::vector<int> ids;
  ids.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    ids.push_back(static_cast<int>(base + i));
  }
  return ids;
}

std::optional<int> KVCacheManagerWithTransfer::GetLocalTpuNumaNode(
    xla::PjRtBuffer* buf) const {
  if (buf && buf->device()) {
    int node = GetPjRtDeviceNumaNode(buf->device());
    if (node >= 0) {
      return node;
    }
  }
  return std::nullopt;
}

absl::Status KVCacheManagerWithTransfer::OnBlocksReceived(
    const std::vector<int>& block_ids, uint64_t uuid) {
  RAIDEN_TRACE_FN("KVTransfer::OnBlocksReceived", [&]() {
    return absl::StrCat("blocks=", block_ids.size(), " uuid=", uuid);
  });
  VLOG(1) << "KVCacheManagerWithTransfer::OnBlocksReceived called. uuid: "
          << uuid << ", received blocks count: " << block_ids.size();

  std::shared_ptr<ReceiveSession> session;
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) {
      return absl::OkStatus();
    }
    session = it->second;
  }
  absl::Status status = session->OnBlocksReceived(*this, block_ids);
  MaybeUnregisterSettledRecv(uuid, *session);
  return status;
}

}  // namespace tpu_raiden

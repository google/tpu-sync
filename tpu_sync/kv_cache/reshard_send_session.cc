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

#include "tpu_sync/kv_cache/reshard_send_session.h"

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/pool_reshard_send_slots.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/utils.h"
#include "tpu_sync/kv_cache/data_connection_policy.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/kv_cache/transfer_session.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/block_transport_session.h"

namespace tpu_raiden {

namespace {

// Encapsulates the producer-side state and execution logic for a single
// push-driven pool-reshard send transfer (identified by |uuid|).
//
// Thread-safety: This class is thread-safe. Immutable fields (|base_|,
// |parallelism_|, |deadline_|, |dst_peers_|, |data_connection_reuse_|, and
// |transfer_plan_|, whose request supplies the uuid and plan) are set
// at construction time and read without locking. Mutable state is protected
// by internal |mu_|.
class ReshardSendSession final
    : public TransferSession,
      public std::enable_shared_from_this<ReshardSendSession> {
 public:
  // Built only by CreateReshardSendSession(), which validates |transfer_plan|
  // and the other arguments first.
  ReshardSendSession(kv_cache::KVCacheManagerBase* base,
                     std::vector<int64_t> src_block_ids, int parallelism,
                     int remaining_pool_peer_pushes,
                     std::chrono::steady_clock::time_point deadline,
                     TransferPlan transfer_plan,
                     std::vector<std::string> dst_peers,
                     DataConnectionReuse data_connection_reuse);

  // Validates |plan| against |base| and |src_block_ids|.
  static absl::Status ValidatePlan(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> src_block_ids);

  // Releases any pool staging leases still held (a session dropped before it
  // settled); a settled session has already released them.
  ~ReshardSendSession() override;

  bool Done() const override {
    absl::MutexLock lock(mu_);
    return done_;
  }

  void Finish(const absl::Status& status = absl::OkStatus()) override;

  bool TryBeginOp() override { return false; }

  // Counts one op out. Under kPerTransfer, the last op of a finalizing send
  // closes the idle data connections to |dst_peers_| while it is still
  // counted, then settles the send (which also retires its plan).
  void EndOp() override;

  absl::Status GetStatus() const override {
    absl::MutexLock lock(mu_);
    return status_;
  }

  absl::Status AwaitForDone() override;

  bool IsDraining() const override {
    absl::MutexLock lock(mu_);
    return finalizing_;
  }

  std::chrono::steady_clock::time_point deadline() const override {
    return deadline_;
  }

  // Leases host staging per pool, issues D2H copies for each pool in the plan
  // from the source blocks given to CreateReshardSendSession(), and chains
  // async H2H peer pushes upon D2H completion. Non-blocking preconditions
  // (manager, transport, host staging layers, pool indices) are checked
  // synchronously and returned; the staging waits, copies and pushes run on a
  // pipeline thread and report through GetStatus() / Done().
  absl::Status Start() override;

  // Returns the buffer the device-to-host copy leased to |pool_idx|.
  absl::StatusOr<size_t> ResolveStagingSlot(size_t pool_idx) override {
    return LookupStagingSlot(pool_idx);
  }

  // Always returns kFailedPrecondition: a send session never receives data.
  absl::Status OnDataReceived(size_t major_dimension_idx,
                              absl::Span<const int> minor_dimension_indices,
                              absl::Span<const int> shard_ids) override;

  // Pool-reshard sends push each pool themselves once its device-to-host copy
  // lands, so no transport read gates on them: the future is already OK.
  tsl::Future<> GetDataReadyFuture(size_t major_dimension_idx) override {
    return tsl::Future<>(absl::OkStatus());
  }

  absl::optional<std::shared_ptr<const TransferPlan>> plan() const override {
    if (transfer_plan_ == nullptr || Done()) {
      return absl::nullopt;
    }
    return transfer_plan_;
  }

 private:
  // Returns the host staging buffer leased to |pool_idx|, leasing one when it
  // has none and waiting until deadline when every slot is leased.
  absl::StatusOr<size_t> AcquireStagingSlot(size_t pool_idx);

  // Returns the host staging buffer currently leased to |pool_idx|. Fails
  // with the session's status (or kCancelled) once the session is draining
  // and holds no lease for |pool_idx|, and kFailedPrecondition when a live
  // session holds none.
  absl::StatusOr<size_t> LookupStagingSlot(size_t pool_idx) const;

  // Checks the preconditions of Start that need no waiting: a running
  // transport server, at least one host staging layer, and every transfer
  // pool index resolving on |base_|.
  absl::Status ValidatePushPreconditions() const;

  void ExecutePushPipeline(std::vector<int64_t> src_block_ids);

  // The sender plan this session executes; immutable for its lifetime.
  const ::tpu_sync::rpc::StartTransferRequest& request() const {
    return transfer_plan_->request();
  }
  uint64_t uuid() const { return request().uuid(); }

  // Records completion or failure of one (pool, peer) push operation and
  // releases staging resources once all scheduled pushes settle.
  void RecordPushCompletion(const absl::Status& status);
  void StartPoolPush(size_t pool_idx);
  void SettleLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseLeasesLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  // Source device blocks Start() copies from.
  const std::vector<int64_t> src_block_ids_;
  const int parallelism_ = 8;
  const std::chrono::steady_clock::time_point deadline_;
  // Data endpoints the sender schedule pushes to.
  const std::vector<std::string> dst_peers_;
  // TPU_RAIDEN_DATA_CONN_REUSE as resolved when the session was created.
  const DataConnectionReuse data_connection_reuse_;
  const std::shared_ptr<const TransferPlan> transfer_plan_;

  mutable absl::Mutex mu_;
  int remaining_pool_peer_pushes_ ABSL_GUARDED_BY(mu_) = 0;
  int in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  bool finalizing_ ABSL_GUARDED_BY(mu_) = false;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<raiden::PjRtCopyFuture> d2h_futures_ ABSL_GUARDED_BY(mu_);
  bool leases_released_ ABSL_GUARDED_BY(mu_) = false;
  // The last op already closed the idle data connections (kPerTransfer).
  bool connections_released_ ABSL_GUARDED_BY(mu_) = false;
  PoolStagingLayers staging_layer_manager_;
  std::atomic<bool> cancelled_{false};
};

}  // namespace

absl::Status ReshardSendSession::ValidatePlan(
    const kv_cache::KVCacheManagerBase& base,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> src_block_ids) {
  ABSL_RETURN_IF_ERROR(
      ValidateCommonPoolReshardPlan(&base, plan, src_block_ids));
  ABSL_RETURN_IF_ERROR(ValidatePoolBlockBounds(&base, plan, src_block_ids));

  absl::flat_hash_set<int64_t> local_ids(src_block_ids.begin(),
                                         src_block_ids.end());
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      const int64_t local_id = entry.src_block_id();
      if (local_ids.find(local_id) == local_ids.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("source block id ", local_id,
                         " is absent from the local block-id list"));
      }
      const int64_t local_offset = entry.src_offset_bytes();
      const int64_t local_stride = entry.src_stride_bytes();
      const int32_t group_idx = entry.pool_group();
      for (int32_t encoded_pool_idx :
           plan.pool_groups(group_idx).pool_indices()) {
        const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
        const kv_cache::PoolSpec* spec = base.pool(pool_idx);
        if (!StridedSpanFitsRegions(local_offset, local_stride,
                                    entry.size_bytes(), entry.count(),
                                    spec->block_stride_bytes, spec->regions)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "source span exceeds declared pool ", pool_idx,
              " live regions in block ", local_id, ": offset=", local_offset,
              " stride=", local_stride, " size=", entry.size_bytes(), " count=",
              entry.count(), " block_stride_bytes=", spec->block_stride_bytes));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<TransferSession>> CreateReshardSendSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator,
    absl::Span<const int64_t> src_block_ids, int parallelism,
    std::chrono::steady_clock::time_point deadline,
    ::tpu_sync::rpc::StartTransferRequest plan) {
  if (staging_allocator == nullptr) {
    return absl::InvalidArgumentError("staging_allocator must not be null");
  }
  ABSL_ASSIGN_OR_RETURN(const DataConnectionReuse data_connection_reuse,
                        ResolveDataConnectionReuse());
  ABSL_RETURN_IF_ERROR(
      ReshardSendSession::ValidatePlan(*base, plan, src_block_ids));
  // Device-only executor: without device attachments there are no bytes this
  // path could legitimately move; host-only managers fail closed with no
  // host-mode branch to mask device bugs.
  if (!base->has_device_buffers()) {
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

  // One completion per (pool, peer-with-entries): with sharded destinations
  // a group may push each of its pools to a single peer, so pools x peers
  // would wait for completions that never come (tpu-sync follow-up on #744).
  const int remaining_pool_peer_pushes =
      static_cast<int>(CountPoolReshardSendSlots(plan, schedule_it->second));
  if (remaining_pool_peer_pushes <= 0) {
    return absl::InvalidArgumentError("sender plan schedules no pushes");
  }

  base->InitTransportServer();
  ABSL_ASSIGN_OR_RETURN(
      TransferPlan transfer_plan,
      TransferPlan::Create(plan, /*is_sender=*/true, base->num_pools(),
                           base->has_explicit_pools(), base->pools()));

  auto session = std::shared_ptr<ReshardSendSession>(new ReshardSendSession(
      base, std::vector<int64_t>(src_block_ids.begin(), src_block_ids.end()),
      parallelism, remaining_pool_peer_pushes, deadline,
      std::move(transfer_plan),
      std::vector<std::string>(peers.begin(), peers.end()),
      data_connection_reuse));
  return session;
}

ReshardSendSession::ReshardSendSession(
    kv_cache::KVCacheManagerBase* base, std::vector<int64_t> src_block_ids,
    int parallelism, int remaining_pool_peer_pushes,
    std::chrono::steady_clock::time_point deadline, TransferPlan transfer_plan,
    std::vector<std::string> dst_peers,
    DataConnectionReuse data_connection_reuse)
    : base_(base),
      src_block_ids_(std::move(src_block_ids)),
      parallelism_(parallelism),
      deadline_(deadline),
      dst_peers_(std::move(dst_peers)),
      data_connection_reuse_(data_connection_reuse),
      transfer_plan_(
          std::make_shared<const TransferPlan>(std::move(transfer_plan))),
      remaining_pool_peer_pushes_(remaining_pool_peer_pushes),
      staging_layer_manager_(PoolStagingLayers::ForManager(base)) {}

ReshardSendSession::~ReshardSendSession() {
  absl::MutexLock lock(mu_);
  ReleaseLeasesLocked();
}

absl::StatusOr<size_t> ReshardSendSession::AcquireStagingSlot(size_t pool_idx) {
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
  }
  absl::StatusOr<size_t> slot =
      staging_layer_manager_.Acquire(pool_idx, deadline_, &cancelled_);
  if (!slot.ok()) {
    return slot.status();
  }
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      staging_layer_manager_.Release(pool_idx);
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
  }
  return *slot;
}

absl::StatusOr<size_t> ReshardSendSession::LookupStagingSlot(
    size_t pool_idx) const {
  // A lease outlives the session's finalizing_ flip: a push whose completion
  // started finalizing may still be addressing another pool's host bytes.
  // SettleLocked releases every lease, so a done session never resolves one.
  if (std::optional<size_t> slot = staging_layer_manager_.Lookup(pool_idx);
      slot.has_value()) {
    return *slot;
  }
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
  }
  return absl::FailedPreconditionError(
      absl::StrCat("Pool ", pool_idx, " has no leased staging slot"));
}

absl::Status ReshardSendSession::ValidatePushPreconditions() const {
  if (base_->transport_server() == nullptr) {
    return absl::FailedPreconditionError("transport server is not running");
  }
  if (base_->num_host_layers() == 0) {
    return absl::FailedPreconditionError(
        "pool reshard push requires at least one host staging layer");
  }
  for (int32_t encoded_pool_idx : request().transfer_pool_indices()) {
    const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
    if (base_->pool(pool_idx) == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("transfer pool index out of range: ", pool_idx));
    }
  }
  return absl::OkStatus();
}

absl::Status ReshardSendSession::Start() {
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      return !status_.ok() ? status_
                           : absl::CancelledError("Send session is draining");
    }
    ++in_flight_;
  }
  absl::Cleanup end_op = [this]() { EndOp(); };
  // Fail fast on the plan-shape and transport preconditions that need no
  // waiting, so the START_TRANSFER reply still carries them. Staging slot
  // and arena leases can block on earlier pools' pushes and stay on the
  // pipeline thread below.
  if (absl::Status precondition = ValidatePushPreconditions();
      !precondition.ok()) {
    RecordPushCompletion(precondition);
    return precondition;
  }
  std::vector<int64_t> src_ids = src_block_ids_;
  try {
    std::thread([self = shared_from_this(),
                 src_ids = std::move(src_ids)]() mutable {
      self->ExecutePushPipeline(std::move(src_ids));
      self->EndOp();
    }).detach();
  } catch (const std::system_error& e) {
    // Thread exhaustion: fail the send on the synchronous path so the session
    // settles and the plan is dropped instead of waiting for the deadline
    // sweep with the exception unwinding through the RPC handler.
    absl::Status status = absl::InternalError(
        absl::StrCat("failed to start pool reshard push thread: ", e.what()));
    RecordPushCompletion(status);
    return status;
  }
  std::move(end_op).Cancel();
  return absl::OkStatus();
}

void ReshardSendSession::ExecutePushPipeline(
    std::vector<int64_t> src_block_ids) {
  // Multi-tag plans scope each pool's staging and pushes to its group's
  // entries; the flat src_block_ids argument is the legacy single-tag
  // whole-plan block list.
  const auto pool_group_index = [this](size_t pool_idx) -> int {
    for (int group_idx = 0; group_idx < request().pool_groups_size();
         ++group_idx) {
      const auto& indices = request().pool_groups(group_idx).pool_indices();
      if (std::find(indices.begin(), indices.end(),
                    static_cast<int32_t>(pool_idx)) != indices.end()) {
        return group_idx;
      }
    }
    return -1;
  };
  auto local_schedule_it = request().shard_push_schedules().find(0);
  if (local_schedule_it == request().shard_push_schedules().end() &&
      request().shard_push_schedules().size() == 1) {
    local_schedule_it = request().shard_push_schedules().begin();
  }

  std::vector<size_t> ordered_pools;
  ordered_pools.reserve(request().transfer_pool_indices_size());
  for (int32_t encoded_pool_idx : request().transfer_pool_indices()) {
    ordered_pools.push_back(static_cast<size_t>(encoded_pool_idx));
  }
  std::stable_sort(
      ordered_pools.begin(), ordered_pools.end(), [&](size_t a, size_t b) {
        const int ga = pool_group_index(a);
        const int gb = pool_group_index(b);
        const int rank_a = (ga >= 0 && ga < request().pool_groups_size())
                               ? request().pool_groups(ga).order_rank()
                               : 0;
        const int rank_b = (gb >= 0 && gb < request().pool_groups_size())
                               ? request().pool_groups(gb).order_rank()
                               : 0;
        return rank_a < rank_b;
      });

  for (size_t pool_idx : ordered_pools) {
    std::vector<int64_t> pool_src_block_ids(src_block_ids.begin(),
                                            src_block_ids.end());
    if (local_schedule_it != request().shard_push_schedules().end()) {
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

    absl::StatusOr<size_t> host_slot = AcquireStagingSlot(pool_idx);
    if (!host_slot.ok()) {
      RecordPushCompletion(host_slot.status());
      return;
    }
    {
      absl::MutexLock lock(mu_);
      if (done_ || finalizing_) {
        staging_layer_manager_.Release(pool_idx);
        return;
      }
    }

    // Bounded host staging: lease one arena slot per source page of this
    // pool's storage for the transfer before staging its bytes (no-op on
    // full-mirror storages). Released in SettleLocked.
    if (const kv_cache::PoolSpec* pool_spec = base_->pool(pool_idx);
        pool_spec != nullptr) {
      absl::Status lease_status = base_->AcquirePoolStagingLease(
          uuid(), pool_spec->storage_index, pool_src_block_ids,
          base_->pool_staging_lease_timeout());
      if (!lease_status.ok()) {
        RecordPushCompletion(lease_status);
        staging_layer_manager_.Release(pool_idx);
        return;
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> future =
        base_->D2hPoolBlocks(pool_idx, pool_src_block_ids,
                             /*shard_idx=*/std::nullopt, uuid(), *host_slot);
    if (!future.ok()) {
      RecordPushCompletion(future.status());
      staging_layer_manager_.Release(pool_idx);
      return;
    }
    raiden::PjRtCopyFuture pool_future = *std::move(future);
    {
      absl::MutexLock lock(mu_);
      d2h_futures_.push_back(pool_future);
      ++in_flight_;
    }
    pool_future.OnReady(
        [self = shared_from_this(), pool_idx](auto copy_status) {
          absl::Cleanup end_op = [self]() { self->EndOp(); };
          if (!copy_status.ok()) {
            self->RecordPushCompletion(copy_status.status());
            self->staging_layer_manager_.Release(pool_idx);
            return;
          }
          self->StartPoolPush(pool_idx);
        });
  }
}

void ReshardSendSession::StartPoolPush(size_t pool_idx) {
  RAIDEN_TRACE_FN("KVTransfer::StartPoolReshardPush", [&]() {
    return absl::StrCat("uuid=", uuid(), " pool=", pool_idx);
  });
  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      staging_layer_manager_.Release(pool_idx);
      return;
    }
  }

  auto schedule_it = request().shard_push_schedules().find(0);
  if (schedule_it == request().shard_push_schedules().end()) {
    schedule_it = request().shard_push_schedules().begin();
  }
  // A pool pushes only its own group's (src, dst) pairs.
  int pool_group_idx = -1;
  for (int group_idx = 0; group_idx < request().pool_groups_size();
       ++group_idx) {
    const auto& indices = request().pool_groups(group_idx).pool_indices();
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

  auto remaining_pushes =
      std::make_shared<std::atomic<int>>(transfers_by_peer.size());
  if (transfers_by_peer.empty()) {
    staging_layer_manager_.Release(pool_idx);
    return;
  }

  transport::BlockTransport* transport_srv = base_->transport_server();
  if (transport_srv == nullptr) {
    RecordPushCompletion(
        absl::FailedPreconditionError("transport server is not running"));
    staging_layer_manager_.Release(pool_idx);
    return;
  }

  {
    absl::MutexLock lock(mu_);
    if (done_ || finalizing_) {
      staging_layer_manager_.Release(pool_idx);
      return;
    }
    in_flight_ += static_cast<int>(transfers_by_peer.size());
  }

  // The plan names this pool in the sender's own index space; the receiver
  // resolves the wire index against the destination's pool table.
  std::optional<int> wire_pool_idx;
  auto wire_it =
      request().wire_pool_indices().find(static_cast<int32_t>(pool_idx));
  if (wire_it != request().wire_pool_indices().end()) {
    wire_pool_idx = wire_it->second;
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
    transport_srv
        ->AsyncPush({peer}, src_ids, dst_ids, parallelism_,
                    transport::MajorOrder::kLayerMajor, uuid(),
                    static_cast<int>(pool_idx), wire_pool_idx)
        .OnReady([self = shared_from_this(), pool_idx, remaining_pushes](
                     const absl::StatusOr<std::vector<int>>& result) {
          absl::Cleanup end_op = [self]() { self->EndOp(); };
          const bool last_for_pool = remaining_pushes->fetch_sub(1) == 1;
          if (!result.ok()) {
            self->RecordPushCompletion(result.status());
            if (last_for_pool) {
              self->staging_layer_manager_.Release(pool_idx);
            }
            return;
          }
          if (last_for_pool) {
            self->staging_layer_manager_.Release(pool_idx);
          }
          self->RecordPushCompletion(absl::OkStatus());
        });
  }
}

void ReshardSendSession::RecordPushCompletion(const absl::Status& status) {
  RAIDEN_TRACE_FN("KVTransfer::FinishPoolReshardSend", [&]() {
    return absl::StrCat("uuid=", uuid(), " status=", status.code());
  });
  absl::MutexLock lock(mu_);
  if (done_ || finalizing_) return;
  if (!status.ok()) {
    LOG(ERROR) << "Pool reshard send failed uuid=" << uuid() << ": " << status;
    if (status_.ok()) {
      status_ = status;
    }
    finalizing_ = true;
    cancelled_.store(true, std::memory_order_relaxed);
    staging_layer_manager_.Shutdown();
  } else if (--remaining_pool_peer_pushes_ == 0) {
    finalizing_ = true;
    cancelled_.store(true, std::memory_order_relaxed);
    staging_layer_manager_.Shutdown();
  }
}

void ReshardSendSession::Finish(const absl::Status& status) {
  absl::MutexLock lock(mu_);
  if (done_ || finalizing_) return;
  if (!status.ok() && status_.ok()) {
    status_ = status;
  }
  finalizing_ = true;
  cancelled_.store(true, std::memory_order_relaxed);
  staging_layer_manager_.Shutdown();
  // Settling here, with no op in flight, means no push was ever issued:
  // Start holds an op while it schedules, StartPoolPush counts every
  // push before issuing it, and each push stays counted until its completion
  // has run RecordPushCompletion, which starts finalizing on the first
  // failure or the last scheduled success. A send that pushed therefore
  // settles in EndOp, which releases its data connections first.
  if (in_flight_ == 0) {
    SettleLocked();
  }
}

absl::Status ReshardSendSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

absl::Status ReshardSendSession::OnDataReceived(
    size_t major_dimension_idx, absl::Span<const int> minor_dimension_indices,
    absl::Span<const int> shard_ids) {
  return absl::FailedPreconditionError(absl::StrCat(
      "send session does not receive data: pool ", major_dimension_idx,
      " reported received on send session for UUID ", uuid()));
}

void ReshardSendSession::EndOp() {
  // Under kPerTransfer the last op of a send closes the idle data
  // connections to |dst_peers_| before it is counted out, also while the
  // manager is being destroyed: every push has returned its socket by now,
  // and ~KVCacheManagerWithTransfer is still parked in AwaitForDone(), which
  // this op releases, so the manager, |base_| and its transport are alive.
  // transport_server() cannot contend on server_init_mu_ either: every
  // StopTransportServer() call is in a destructor that runs only after that
  // wait. The check and the decrement share a critical section so a Finish()
  // that starts finalizing in between cannot settle the send unreleased.
  while (true) {
    {
      absl::MutexLock lock(mu_);
      const bool release_due =
          data_connection_reuse_ == DataConnectionReuse::kPerTransfer &&
          in_flight_ == 1 && finalizing_ && !done_ && !connections_released_;
      if (!release_due) {
        --in_flight_;
        if (finalizing_ && in_flight_ == 0 && !done_) {
          SettleLocked();
        }
        return;
      }
      connections_released_ = true;
    }
    ReleaseDataConnectionsAfterSend(base_->transport_server(), dst_peers_,
                                    uuid());
  }
}

void ReshardSendSession::ReleaseLeasesLocked() {
  staging_layer_manager_.ReleaseAll();
  if (leases_released_) return;
  leases_released_ = true;
  base_->ReleasePoolStagingLeases(uuid());
}

void ReshardSendSession::SettleLocked() {
  // Every push of every pool has completed (or the send failed): release the
  // host staging arena slots and mark done atomically under |mu_|.
  ReleaseLeasesLocked();
  cancelled_.store(true, std::memory_order_relaxed);
  staging_layer_manager_.Shutdown();
  done_ = true;
}

}  // namespace tpu_raiden

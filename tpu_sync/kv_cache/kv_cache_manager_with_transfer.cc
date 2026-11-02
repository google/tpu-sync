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

#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <ratio>  // NOLINT(build/c++11)
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/metrics_collector.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/reshard_receive_session.h"
#include "tpu_sync/kv_cache/reshard_send_session.h"
#include "tpu_sync/kv_cache/transfer_receive_session.h"
#include "tpu_sync/kv_cache/transfer_send_session.h"
#include "tpu_sync/kv_cache/transfer_session.h"

namespace tpu_raiden {

namespace {

constexpr absl::Duration kPendingWorkTimeout = absl::Seconds(30);

// How long a pull request waits for the producer to register the read it
// names. The registration normally precedes the announcement the consumer
// acts on, so this only covers reordering between the two; a pull whose
// registration expired, or never happened, is rejected once it lapses.
constexpr absl::Duration kPullRegistrationGrace = absl::Seconds(5);

// Room above kPullRegistrationGrace for the rest of a handshake once the
// producer has registered. Deriving the control deadline from the grace
// rather than picking a number keeps the two from drifting apart: a deadline
// at or below the grace would be a dead heat with the producer's own
// legitimate wait, turning ordinary reordering between the announcement and
// the registration into a hard failure instead of something waited through.
constexpr absl::Duration kControlDeadlineSlack = absl::Seconds(5);

// Bound on one control handshake, end to end, unless the caller's transfer
// budget is tighter. A handshake is a few hundred bytes and a round trip; it
// has no business inheriting the bulk-transfer timeout, because until it
// returns it is holding one of a small number of workers.
double DeriveControlTimeoutS(double timeout_s) {
  return absl::ToDoubleSeconds(
      std::min(absl::Seconds(timeout_s),
               kPullRegistrationGrace + kControlDeadlineSlack));
}

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

// Formats a single-line summary of why transfer |req_id| failed for logging.
std::string DescribeFailure(absl::string_view req_id,
                            const TransferDetails& details) {
  std::string line = absl::StrCat("Transfer failed: req_id=", req_id);
  if (details.code.has_value()) {
    absl::StrAppend(&line, " code=", *details.code);
  }
  if (details.phase.has_value()) {
    absl::StrAppend(&line, " phase=", *details.phase);
  }
  if (details.peer.has_value()) {
    absl::StrAppend(&line, " peer=", *details.peer);
  }
  if (details.elapsed_ns.has_value()) {
    absl::StrAppend(&line, " elapsed_ms=", *details.elapsed_ns / 1e6);
  }
  if (details.reason.has_value()) {
    absl::StrAppend(&line, " reason=", *details.reason);
  }
  return line;
}

}  // namespace

TransferDetails DetailsFromStatus(const absl::Status& status) {
  TransferDetails details;
  if (!status.ok()) {
    details.code = absl::StatusCodeToString(status.code());
    details.reason = std::string(status.message());
  }
  return details;
}

void KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked(
    bool is_recv, const std::string& req_id, uint64_t uuid,
    int32_t sub_plan_index, int32_t sub_plan_count) {
  if (sub_plan_count <= 1) return;
  auto& chains = is_recv ? recv_chains_ : send_chains_;
  if (sub_plan_index == 0) {
    // Sub-plan 0 is armed before any other sub-plan of its chain: a new
    // chain for this req_id (e.g. a retry) starts from a fresh entry.
    auto it = chains.find(req_id);
    if (it != chains.end() && !it->second.failed) {
      LOG(WARNING) << "Pool reshard " << (is_recv ? "receive" : "send")
                   << " req_id=" << req_id
                   << ": sub-plan 0 of a new chain registered while the "
                      "previous chain was still unsettled ("
                   << it->second.settled.size() << " of " << it->second.count
                   << " sub-plans settled); tracking the new chain";
    }
    chains[req_id] = SubPlanChainProgress{};
    chains[req_id].anchored = true;
  }
  SubPlanChainProgress& progress = chains[req_id];
  progress.plan_uuids.insert(uuid);
  progress.count = std::max(progress.count, sub_plan_count);
  progress.last_activity = std::chrono::steady_clock::now();
}

void KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked(
    bool is_recv, const std::string& req_id, uint64_t uuid,
    int32_t sub_plan_index, int32_t sub_plan_count,
    const absl::Status& status) {
  absl::flat_hash_set<std::string>& done =
      is_recv ? done_recving_ : done_sending_;
  if (sub_plan_count <= 1) {
    // A single-plan request: the session's terminal is the request's.
    (status.ok() ? done : failed_recving_).insert(req_id);
    if (!status.ok()) transfer_details_[req_id] = DetailsFromStatus(status);
    return;
  }
  auto& chains = is_recv ? recv_chains_ : send_chains_;
  if (auto it = chains.find(req_id);
      it != chains.end() && it->second.anchored &&
      !it->second.plan_uuids.contains(uuid)) {
    // A sub-plan of an earlier chain under the same req_id settling after a
    // new chain (a retry) started: that chain was already reported.
    VLOG(1) << "Pool reshard " << (is_recv ? "receive" : "send")
            << " req_id=" << req_id << " uuid=" << uuid
            << ": ignoring sub-plan " << sub_plan_index
            << " of a superseded chain (" << status << ")";
    return;
  }
  SubPlanChainProgress& progress = chains[req_id];
  progress.last_activity = std::chrono::steady_clock::now();
  if (progress.failed) return;  // the request's failure was already reported
  progress.count = std::max(progress.count, sub_plan_count);
  if (!status.ok()) {
    progress.failed = true;
    failed_recving_.insert(req_id);
    TransferDetails details = DetailsFromStatus(status);
    details.reason = absl::StrCat("sub-plan ", sub_plan_index, " of ",
                                  sub_plan_count, ": ", status.message());
    transfer_details_[req_id] = std::move(details);
    LOG(WARNING) << "Pool reshard " << (is_recv ? "receive" : "send")
                 << " req_id=" << req_id << " failed at sub-plan "
                 << sub_plan_index << " of " << sub_plan_count << ": "
                 << status;
    return;
  }
  progress.settled.insert(sub_plan_index);
  VLOG(1) << "Pool reshard " << (is_recv ? "receive" : "send")
          << " req_id=" << req_id << " sub-plan " << sub_plan_index << " of "
          << sub_plan_count << " settled (" << progress.settled.size()
          << " settled so far)";
  if (static_cast<int32_t>(progress.settled.size()) >= progress.count) {
    done.insert(req_id);
    chains.erase(req_id);
  }
}

void KVCacheManagerWithTransfer::SweepSubPlanChainsLocked(
    std::chrono::steady_clock::time_point now) {
  if (recv_chains_.empty() && send_chains_.empty()) return;
  const auto stall = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(timeout_s_));
  for (auto* chains : {&recv_chains_, &send_chains_}) {
    for (auto it = chains->begin(); it != chains->end();) {
      SubPlanChainProgress& progress = it->second;
      if (now - progress.last_activity <= stall) {
        ++it;
        continue;
      }
      if (progress.failed) {
        // The failure was reported; late sub-plans of the doomed request
        // were swallowed for one timeout, which is as long as any of them
        // can live.
        chains->erase(it++);
        continue;
      }
      // Quiet for a whole timeout: unless a sub-plan of the request is
      // still active, its chain has stalled (the coordinator never armed or
      // dispatched the next sub-plan).
      bool active = false;
      for (uint64_t uuid : progress.plan_uuids) {
        if (sessions_.contains(uuid)) {
          active = true;
          break;
        }
      }
      if (active) {
        progress.last_activity = now;
        ++it;
        continue;
      }
      const absl::Status status = absl::DeadlineExceededError(absl::StrCat(
          "pool reshard sub-plan chain stalled: ", progress.settled.size(),
          " of ", progress.count, " sub-plans settled, none active for ",
          timeout_s_, "s"));
      failed_recving_.insert(it->first);
      transfer_details_[it->first] = DetailsFromStatus(status);
      LOG(WARNING) << "Pool reshard req_id=" << it->first << ": "
                   << status.message();
      progress.failed = true;
      progress.last_activity = now;
      ++it;
    }
  }
}

void KVCacheManagerWithTransfer::InitializeBaseHooks() {
  kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
  hooks.get_session = [this](uint64_t uuid) { return GetSession(uuid); };
  base_->SetTransferEventHooks(std::move(hooks));
}

absl::StatusOr<std::shared_ptr<TransferSession>>
KVCacheManagerWithTransfer::GetSession(uint64_t uuid) const {
  absl::MutexLock lock(mu_);
  if (auto it = sessions_.find(uuid); it != sessions_.end()) {
    return it->second;
  }
  return absl::NotFoundError(absl::StrCat("Session ", uuid, " not found"));
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerWithTransfer(
          std::make_unique<kv_cache::KVCacheManagerBase>(
              layer_buffers, local_port,
              host_blocks_to_allocate.value_or(num_slots * max_blocks),
              unsafe_skip_buffer_lock, parallelism, host_allocator),
          node_id, local_control_port, max_blocks, num_slots, timeout_s,
          std::move(metrics_collector)) {}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    size_t slice_byte_size, const std::vector<int64_t>& dimensions,
    size_t physical_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector,
    std::optional<int> num_staging_layers)
    : KVCacheManagerWithTransfer(
          std::make_unique<kv_cache::KVCacheManagerBase>(
              layer_buffers, local_port,
              host_blocks_to_allocate.value_or(num_slots * max_blocks),
              unsafe_skip_buffer_lock, parallelism, host_allocator,
              /*bind_ip=*/std::nullopt,
              slice_byte_size > 0 ? std::make_optional(slice_byte_size)
                                  : std::nullopt,
              dimensions,
              physical_size > 0 ? std::make_optional(physical_size)
                                : std::nullopt,
              num_staging_layers),
          node_id, local_control_port, max_blocks, num_slots, timeout_s,
          std::move(metrics_collector)) {}

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
    : KVCacheManagerWithTransfer(
          std::make_unique<kv_cache::KVCacheManagerBase>(
              num_layers, num_shards, std::move(slice_byte_sizes), local_port,
              host_blocks_to_allocate.value_or(num_slots * max_blocks),
              parallelism, nullptr),
          node_id, local_control_port, max_blocks, num_slots, timeout_s,
          std::move(metrics_collector)) {}

KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer() {
  // Stop admitting transfer requests, then drain what is in flight before any
  // state is torn down: Finish(Cancelled) every session so it settles (a
  // passive receive would otherwise hold this destructor until its deadline),
  // end staging waits, wait for the pull-serve workers that read this
  // object's state, and wait for every session to be done. Only then stop
  // the transport, which those sessions' last ops still use.
  StopControlServer();
  std::vector<std::shared_ptr<TransferSession>> sessions_to_wait;
  {
    absl::MutexLock lock(mu_);
    sessions_to_wait.reserve(sessions_.size());
    const absl::Status cancel_status =
        absl::CancelledError("KVCacheManagerWithTransfer shutting down");
    for (const auto& [uuid, session] : sessions_) {
      (void)uuid;
      session->Finish(cancel_status);
      sessions_to_wait.push_back(session);
    }
  }
  if (staging_allocator_) {
    staging_allocator_->Shutdown();
  }
  {
    absl::MutexLock lock(pull_workers_mu_);
    pull_workers_mu_.Await(absl::Condition(
        +[](int* active) { return *active == 0; }, &active_pull_workers_));
  }
  for (const auto& session : sessions_to_wait) {
    session->AwaitForDone().IgnoreError();
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
    sessions_.clear();
  }
  staging_allocator_.reset();
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    std::unique_ptr<kv_cache::KVCacheManagerBase> base, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : base_(std::move(base)),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      timeout_s_(timeout_s),
      control_timeout_s_(DeriveControlTimeoutS(timeout_s)),
      metrics_collector_(std::move(metrics_collector)) {
  base_->set_node_id(node_id_);
  InitializeBaseHooks();
  InitializeControlPlane();
  if (base_->num_shards() == 0) {
    staging_allocator_ =
        CreateStagingBlockAllocator(base_.get(), /*num_slots=*/0, max_blocks);
    return;
  }
  if (local_control_port_ >= 0) {
    if (max_blocks <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    if (base_->num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
  }
  staging_allocator_ =
      CreateStagingBlockAllocator(base_.get(), num_slots, max_blocks);
  if (local_control_port_ >= 0) {
    StartControlServer();
  }
}

bool KVCacheManagerWithTransfer::RetireSettledSessionLocked(
    absl::flat_hash_map<uint64_t, std::shared_ptr<TransferSession>>::iterator
        it,
    bool* retired_push_progress, bool* retired_pool_staging) {
  const uint64_t uuid = it->first;
  if (auto recv =
          std::dynamic_pointer_cast<TransferReceiveSession>(it->second)) {
    if (!recv->Done()) {
      return false;
    }
    if (recv->tracks_completion()) {
      (!recv->GetStatus().ok() ? failed_recving_ : done_recving_)
          .insert(recv->req_id());
      transfer_details_[recv->req_id()] = recv->Details();
    }
    if (retired_push_progress != nullptr) {
      *retired_push_progress = true;
    }
    sessions_.erase(it);
    return true;
  }
  if (auto reshard_recv =
          std::dynamic_pointer_cast<ReshardReceiveSession>(it->second)) {
    if (!reshard_recv->Done()) {
      return false;
    }
    RecordPoolReshardTerminalLocked(
        /*is_recv=*/true, reshard_recv->req_id(), uuid,
        reshard_recv->sub_plan_index(), reshard_recv->sub_plan_count(),
        reshard_recv->GetStatus());
    if (retired_push_progress != nullptr) {
      *retired_push_progress = true;
    }
    if (retired_pool_staging != nullptr) {
      *retired_pool_staging = true;
    }
    sessions_.erase(it);
    return true;
  }
  if (auto send = std::dynamic_pointer_cast<TransferSendSession>(it->second)) {
    if (!send->Done()) {
      return false;
    }
    const bool failed = !send->GetStatus().ok();
    (failed ? failed_recving_ : done_sending_).insert(send->req_id());
    transfer_details_[send->req_id()] = send->Details();
    if (failed && retired_push_progress != nullptr) {
      *retired_push_progress = true;
    }
    base_->ClearRemoteLayerAddrs(uuid);
    sessions_.erase(it);
    return true;
  }
  if (auto reshard_send =
          std::dynamic_pointer_cast<ReshardSendSession>(it->second)) {
    if (!reshard_send->Done()) {
      return false;
    }
    const bool failed = !reshard_send->GetStatus().ok();
    RecordPoolReshardTerminalLocked(
        /*is_recv=*/false, reshard_send->req_id(), uuid,
        reshard_send->sub_plan_index(), reshard_send->sub_plan_count(),
        reshard_send->GetStatus());
    if (failed) {
      if (retired_push_progress != nullptr) {
        *retired_push_progress = true;
      }
      if (retired_pool_staging != nullptr) {
        *retired_pool_staging = true;
      }
    }
    sessions_.erase(it);
    return true;
  }
  return false;
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

  absl::StatusOr<std::shared_ptr<TransferSendSession>> created =
      TransferSendSession::Create(
          base_.get(), staging_allocator_.get(), req_id, uuid, block_ids,
          deadline.value_or(DeadlineFromNow()), register_start);
  // An invalid registration is a caller bug rather than an empty transfer:
  // throw immediately instead of returning 0, which callers interpret as
  // "nothing to send" and then wait until the consumer's pull times out.
  if (!created.ok()) {
    throw std::invalid_argument(std::string(created.status().message()));
  }
  std::shared_ptr<TransferSendSession> session = *std::move(created);

  {
    absl::MutexLock lock(mu_);
    if (pending_acks_.erase(uuid) > 0) {
      done_sending_.insert(req_id);
      return 0;
    }
    if (!sessions_.try_emplace(uuid, session).second) {
      throw std::runtime_error(
          absl::StrCat("NotifyForRead rejected duplicate uuid=", uuid,
                       " for req_id=", req_id, ": a read is still registered"));
    }
  }
  // A send session only waits for the consumer's pull; its Start() is a no-op
  // today, but any failure it reports follows this function's throw contract.
  if (absl::Status started = session->Start(); !started.ok()) {
    throw std::runtime_error(
        absl::StrCat("NotifyForRead failed to start uuid=", uuid,
                     " for req_id=", req_id, ": ", started.message()));
  }
  cv_.SignalAll();

  LOG(INFO) << "RAIDEN_TIMING event=producer_register"
            << " req_id=" << req_id << " uuid=" << uuid
            << " node_id=" << node_id_ << " blocks=" << block_ids.size()
            << " enqueue_ms="
            << DurationMs(register_start, std::chrono::steady_clock::now())
            << " failed=0";
  return static_cast<int64_t>(uuid);
}

absl::Status KVCacheManagerWithTransfer::EmplaceRecvSessionLocked(
    uint64_t uuid, const std::shared_ptr<TransferReceiveSession>& session) {
  if (auto existing = sessions_.find(uuid); existing != sessions_.end()) {
    if (auto recv =
            std::dynamic_pointer_cast<TransferReceiveSession>(existing->second);
        recv != nullptr && !recv->IsDraining() && recv->IsReadyToComplete()) {
      recv->Finish();
    }
    RetireSettledSessionLocked(existing);
  }
  // try_emplace leaves session untouched on a duplicate. Callers rely on that
  // guarantee to release staging owned by the rejected session.
  if (!sessions_.try_emplace(uuid, session).second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Receive with UUID ", uuid, " is already registered"));
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::RegisterActivePlan(
    uint64_t uuid, const ::tpu_sync::rpc::StartTransferRequest& request,
    bool is_sender) {
  bool retired_existing = false;
  bool retired_pool_staging = false;
  std::shared_ptr<TransferSession> registered;
  {
    absl::MutexLock lock(mu_);
    auto existing = sessions_.find(uuid);
    if ((existing != sessions_.end() && existing->second->plan().has_value()) ||
        pending_pool_reshard_recvs_.contains(uuid)) {
      return absl::AlreadyExistsError(
          absl::StrCat("Plan with UUID ", uuid, " is already registered!"));
    }
    if (existing != sessions_.end()) {
      if (auto recv = std::dynamic_pointer_cast<TransferReceiveSession>(
              existing->second);
          recv != nullptr && !recv->IsDraining() && recv->IsReadyToComplete()) {
        recv->Finish();
      }
      if (!existing->second->Done()) {
        return absl::AlreadyExistsError(
            absl::StrCat(is_sender ? "Send" : "Receive", " with UUID ", uuid,
                         " is already registered"));
      }
    }

    if (is_sender) {
      ABSL_ASSIGN_OR_RETURN(
          std::shared_ptr<TransferSendSession> send_session,
          TransferSendSession::CreateFromActivePlan(
              base_.get(), staging_allocator_.get(), uuid, request));
      if (existing != sessions_.end()) {
        RetireSettledSessionLocked(existing, &retired_existing,
                                   &retired_pool_staging);
      }
      registered = send_session;
      sessions_.emplace(uuid, std::move(send_session));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          std::shared_ptr<TransferReceiveSession> recv_session,
          TransferReceiveSession::CreateFromActivePlan(
              base_.get(), staging_allocator_.get(), uuid, request,
              DeadlineFromNow(), metrics_collector_));
      if (existing != sessions_.end()) {
        RetireSettledSessionLocked(existing, &retired_existing,
                                   &retired_pool_staging);
      }
      absl::Status inserted = EmplaceRecvSessionLocked(uuid, recv_session);
      if (!inserted.ok()) {
        return inserted;
      }
      registered = recv_session;
      if (recv_session->total_blocks() > 0) {
        LOG(INFO) << "RegisterActivePlan (Receiver): Populated "
                     "sessions_ for UUID "
                  << uuid << " with " << recv_session->total_blocks()
                  << " total physical block-pushes (including duplicates "
                     "across sources) for automatic H2D.";
      }
    }
  }
  if (retired_existing) {
    if (auto* transport = base_->transport_server()) {
      transport->ForgetPushProgress(uuid);
    }
  }
  if (retired_pool_staging) {
    base_->ReleasePoolStagingLeases(uuid);
  }
  return registered->Start();
}

absl::Status KVCacheManagerWithTransfer::RegisterRecv(
    uint64_t uuid, const std::string& req_id, int64_t expected_block_count,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  std::shared_ptr<TransferReceiveSession> recv_session;
  {
    absl::MutexLock lock(mu_);
    ASSIGN_OR_RETURN(
        recv_session,
        TransferReceiveSession::Create(
            base_.get(), staging_allocator_.get(), uuid, req_id,
            expected_block_count, deadline.value_or(DeadlineFromNow()),
            /*acquire_staging=*/false, metrics_collector_));
    absl::Status inserted = EmplaceRecvSessionLocked(uuid, recv_session);
    if (!inserted.ok()) {
      return inserted;
    }
    VLOG(1)
        << "RegisterRecv (Receiver): Registered expected block count for UUID "
        << uuid << " with " << expected_block_count << " expected blocks.";
  }
  return recv_session->Start();
}

absl::Status KVCacheManagerWithTransfer::PoolReshardPush(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> src_block_ids, int parallelism) {
  RAIDEN_TRACE_FN("KVTransfer::PoolReshardPush", [&]() {
    return absl::StrCat("uuid=", plan.uuid(),
                        " src_blocks=", src_block_ids.size());
  });
  bool retired_push_progress = false;
  bool retired_pool_staging = false;
  {
    absl::MutexLock lock(mu_);
    auto existing = sessions_.find(plan.uuid());
    if (existing != sessions_.end() &&
        !RetireSettledSessionLocked(existing, &retired_push_progress,
                                    &retired_pool_staging)) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool reshard send UUID already active: ", plan.uuid()));
    }
  }
  if (retired_push_progress) {
    if (auto* transport = base_->transport_server()) {
      transport->ForgetPushProgress(plan.uuid());
    }
  }
  if (retired_pool_staging) {
    base_->ReleasePoolStagingLeases(plan.uuid());
  }

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<ReshardSendSession> state,
      ReshardSendSession::Create(base_.get(), staging_allocator_.get(),
                                 src_block_ids, parallelism, DeadlineFromNow(),
                                 plan));
  {
    absl::MutexLock lock(mu_);
    sessions_[plan.uuid()] = state;
    NoteSubPlanRegisteredLocked(/*is_recv=*/false, plan.req_id(), plan.uuid(),
                                plan.sub_plan_index(), plan.sub_plan_count());
  }

  return state->Start();
}

absl::Status KVCacheManagerWithTransfer::PoolReshardRegisterRecv(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::PoolReshardRegisterRecv", [&]() {
    return absl::StrCat("uuid=", plan.uuid(),
                        " chip_blocks=", chip_block_ids.size());
  });
  bool retired_push_progress = false;
  bool retired_pool_staging = false;
  {
    absl::MutexLock lock(mu_);
    auto existing = sessions_.find(plan.uuid());
    if (existing != sessions_.end() &&
        !RetireSettledSessionLocked(existing, &retired_push_progress,
                                    &retired_pool_staging)) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool reshard recv UUID already active: ", plan.uuid()));
    }
    if (!pending_pool_reshard_recvs_.insert(plan.uuid()).second) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool reshard recv UUID already active: ", plan.uuid()));
    }
  }
  if (retired_push_progress) {
    if (auto* transport = base_->transport_server()) {
      transport->ForgetPushProgress(plan.uuid());
    }
  }
  if (retired_pool_staging) {
    base_->ReleasePoolStagingLeases(plan.uuid());
  }
  absl::Cleanup clear_pending = [this, uuid = plan.uuid()]() {
    absl::MutexLock lock(mu_);
    pending_pool_reshard_recvs_.erase(uuid);
  };

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<ReshardReceiveSession> recv_session,
      ReshardReceiveSession::Create(base_.get(), staging_allocator_.get(), plan,
                                    chip_block_ids, DeadlineFromNow()));
  {
    absl::MutexLock lock(mu_);
    pending_pool_reshard_recvs_.erase(plan.uuid());
    sessions_[plan.uuid()] = recv_session;
    NoteSubPlanRegisteredLocked(/*is_recv=*/true, plan.req_id(), plan.uuid(),
                                plan.sub_plan_index(), plan.sub_plan_count());
  }
  return recv_session->Start();
}

absl::Status KVCacheManagerWithTransfer::UnregisterActivePlan(uint64_t uuid) {
  std::shared_ptr<TransferSession> session;
  std::shared_ptr<TransferSendSession> send;
  std::shared_ptr<TransferReceiveSession> recv;
  std::shared_ptr<ReshardSendSession> reshard_send;
  {
    absl::MutexLock lock(mu_);
    auto it = sessions_.find(uuid);
    if (it == sessions_.end() || !it->second->plan().has_value()) {
      return absl::NotFoundError(
          absl::StrCat("No active plan with UUID ", uuid));
    }
    session = it->second;
    send = std::dynamic_pointer_cast<TransferSendSession>(session);
    recv = std::dynamic_pointer_cast<TransferReceiveSession>(session);
    reshard_send = std::dynamic_pointer_cast<ReshardSendSession>(session);
    if (send != nullptr) {
      // The plan's registrant, not the CompleteReadWithDetails sweep, owns
      // its send session: drop it before finishing it so the sweep never
      // reports it as a transfer.
      sessions_.erase(it);
    }
  }
  if (send != nullptr || reshard_send != nullptr) {
    if (!session->Done()) {
      session->Finish(absl::CancelledError("send plan unregistered"));
    }
    return absl::OkStatus();
  }
  if (!session->Done()) {
    session->Finish(absl::CancelledError("receive plan unregistered"));
  }
  if (!session->Done()) {
    return absl::OkStatus();
  }
  if (recv != nullptr && !recv->tracks_completion()) {
    absl::MutexLock lock(mu_);
    auto it = sessions_.find(uuid);
    if (it != sessions_.end() && it->second == recv) {
      sessions_.erase(it);
    }
  }
  if (auto* transport = base_->transport_server()) {
    transport->ForgetPushProgress(uuid);
  }
  return absl::OkStatus();
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
  std::vector<::tpu_sync::rpc::PoolHostAddrsProto> layer_host_addrs =
      base_->LayerHostAddrs(/*uuid=*/0);
  std::vector<RaidenTransferEndpoint> eps;
  for (const auto& ip : base_->local_ips()) {
    std::string endpoint = absl::StrContains(ip, ':')
                               ? absl::StrCat("[", ip, "]:", port)
                               : absl::StrCat(ip, ":", port);
    // Shards whose device is NUMA-local to this NIC; every shard when the
    // host has no usable NUMA topology.
    eps.push_back({endpoint, base_->shards_for_local_ip(ip), layer_host_addrs});
  }
  return eps;
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
            local_block_ids, parallelism, std::move(local_host_block_ids));
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  RAIDEN_TRACE_FN("KVTransfer::StartRead", [&]() {
    return absl::StrCat("req=", req_id, " uuid=", uuid,
                        " blocks=", remote_block_ids.size());
  });
  const std::chrono::steady_clock::time_point effective_deadline =
      deadline.value_or(DeadlineFromNow());
  LOG(INFO) << "StartRead (initiate): req_id=" << req_id << ", uuid=" << uuid
            << ", numa=" << base_->assigned_numa_node().value_or(-1);
  VLOG(1) << "KVCacheManagerWithTransfer::StartRead (Hybrid Bridge) called. "
             "req_id: "
          << req_id << ", uuid: " << uuid << ", remote: " << remote_endpoint
          << ", Thread: " << std::this_thread::get_id();
  FaultInjectThrow(hooks::kKvCacheManagerApiStartRead);
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
  std::shared_ptr<TransferReceiveSession> session;
  {
    absl::MutexLock lock(mu_);
    auto incumbent = sessions_.find(uuid);
    if (incumbent != sessions_.end() &&
        !RetireSettledSessionLocked(incumbent)) {
      LOG(ERROR) << "StartRead rejected duplicate uuid=" << uuid
                 << " for req_id=" << req_id;
      // Re-delivery of the same announcement is idempotent. The incumbent
      // remains responsible for producing this request's one terminal report.
      auto recv_incumbent =
          std::dynamic_pointer_cast<TransferReceiveSession>(incumbent->second);
      if (recv_incumbent == nullptr || recv_incumbent->req_id() != req_id) {
        failed_recving_.insert(req_id);
        transfer_details_[req_id] = DetailsFromStatus(absl::AlreadyExistsError(
            recv_incumbent != nullptr
                ? absl::StrCat(
                      "StartRead rejected duplicate uuid=", uuid,
                      ", already receiving req_id=", recv_incumbent->req_id())
                : absl::StrCat("StartRead rejected duplicate uuid=", uuid)));
      }
      return;
    }

    absl::StatusOr<std::shared_ptr<TransferReceiveSession>> created =
        TransferReceiveSession::Create(
            base_.get(), this, staging_allocator_.get(), uuid, req_id,
            remote_endpoint, remote_block_ids, local_block_ids,
            local_host_block_ids, effective_deadline, metrics_collector_);
    if (!created.ok()) {
      failed_recving_.insert(req_id);
      transfer_details_[req_id] = DetailsFromStatus(created.status());
      return;
    }
    session = *std::move(created);

    absl::Status inserted = EmplaceRecvSessionLocked(uuid, session);
    if (!inserted.ok()) {
      failed_recving_.insert(req_id);
      transfer_details_[req_id] = DetailsFromStatus(inserted);
      LOG(ERROR) << "StartRead failed to register req_id=" << req_id
                 << ", uuid=" << uuid << ": " << inserted.message();
      return;
    }
  }

  const int64_t num_blocks = session->total_blocks();
  if (metrics_collector_) {
    uint64_t total_bytes = static_cast<uint64_t>(num_blocks) *
                           base_->num_layers() * base_->num_shards() *
                           base_->slice_byte_size();
    metrics_collector_->RecordStart(uuid, req_id, num_blocks, total_bytes);
  }

  if (num_blocks == 0) {
    session->Finish();
    return;
  }

  if (absl::Status status =
          FaultInjectStatus(hooks::kKvCacheManagerApiStartReadSubmitPull);
      !status.ok()) {
    session->Finish(status);
    session->EndOp();
    return;
  }

  // The pull handshake reports its own outcome through the session. A Start()
  // failure means the handshake never ran, so release the op it would have.
  if (absl::Status started = session->Start(); !started.ok()) {
    session->Finish(started);
    session->EndOp();
  }
}

CompleteReadResult KVCacheManagerWithTransfer::CompleteReadWithDetails() {
  RAIDEN_TRACE("KVTransfer::CompleteReadWithDetails");
  FaultInjectThrow(hooks::kKvCacheManagerApiCompleteRead);
  CompleteReadResult completions;
  std::vector<uint64_t> retired_uuids;
  {
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto it = sessions_.begin(); it != sessions_.end();) {
      const uint64_t uuid = it->first;
      const std::shared_ptr<TransferSession> session = it->second;
      if (!session->IsDraining()) {
        if (auto send =
                std::dynamic_pointer_cast<TransferSendSession>(session)) {
          if (send->deadline() <= now) {
            // Past its deadline the transfer failed. One nobody pulled is
            // reported now; one whose copies or pushes still run keeps its
            // staging until they end, so the next transfer is never seated on
            // memory a copy still writes.
            send->Finish(absl::DeadlineExceededError("Send session timed out"));
          }
        } else if (auto reshard_send =
                       std::dynamic_pointer_cast<ReshardSendSession>(session)) {
          if (reshard_send->deadline() <= now) {
            reshard_send->Finish(
                absl::DeadlineExceededError("Pool reshard send timed out"));
          }
        } else if (auto recv =
                       std::dynamic_pointer_cast<TransferReceiveSession>(
                           session)) {
          if (recv->IsReadyToComplete()) {
            LOG(INFO) << "CompleteReadRaw (polling completion): req_id="
                      << recv->req_id();
            recv->Finish();
          } else if (recv->deadline() <= now) {
            recv->Finish(
                absl::DeadlineExceededError("Receive session timed out"));
          }
        } else if (auto reshard_recv =
                       std::dynamic_pointer_cast<ReshardReceiveSession>(
                           session)) {
          if (reshard_recv->deadline() <= now) {
            reshard_recv->Finish(
                absl::DeadlineExceededError("Pool reshard receive timed out"));
          }
        }
      }
      auto current = it++;
      bool retired_push_progress = false;
      if (RetireSettledSessionLocked(current, &retired_push_progress) &&
          retired_push_progress) {
        retired_uuids.push_back(uuid);
      }
    }
    SweepSubPlanChainsLocked(now);
    std::erase_if(retired_uuids,
                  [this](uint64_t uuid) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                    return sessions_.contains(uuid) ||
                           pending_pool_reshard_recvs_.contains(uuid);
                  });
    completions.sent.assign(done_sending_.begin(), done_sending_.end());
    completions.received.assign(done_recving_.begin(), done_recving_.end());
    completions.failed.assign(failed_recving_.begin(), failed_recving_.end());
    completions.details.swap(transfer_details_);
    done_sending_.clear();
    done_recving_.clear();
    failed_recving_.clear();
  }
  // Drop transport receive-progress counters (ForgetPushProgress) and any
  // remaining pool staging leases, so a settled uuid is reusable.
  for (uint64_t uuid : retired_uuids) {
    if (auto* transport = base_->transport_server()) {
      transport->ForgetPushProgress(uuid);
    }
    base_->ReleasePoolStagingLeases(uuid);
  }
  for (const std::string& req_id : completions.failed) {
    auto details_it = completions.details.find(req_id);
    LOG(WARNING) << DescribeFailure(
        req_id, details_it != completions.details.end() ? details_it->second
                                                        : TransferDetails{});
  }
  return completions;
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheManagerWithTransfer::CompleteReadRaw() {
  CompleteReadResult completions = CompleteReadWithDetails();
  return {std::move(completions.sent), std::move(completions.received),
          std::move(completions.failed)};
}

std::unique_ptr<StagingBlockAllocator> CreateStagingBlockAllocator(
    kv_cache::KVCacheManagerBase* base, int64_t num_slots, int64_t max_blocks,
    std::optional<bool> dynamic_host_staging) {
  if (num_slots > 0 && max_blocks > 0) {
    absl::Status status =
        base->ConfigureHostStagingSlots(num_slots, max_blocks);
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }
  }
  absl::StatusOr<std::unique_ptr<StagingBlockAllocator>> allocator =
      StagingBlockAllocator::Create(base->host_block_manager(), num_slots,
                                    max_blocks, dynamic_host_staging);
  if (!allocator.ok()) {
    throw std::runtime_error(std::string(allocator.status().message()));
  }
  return *std::move(allocator);
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
  // The inbound side gets the same short bound as the outbound one: a handler
  // reading a request off a stalled consumer is holding a pull-pool worker for
  // as long as it blocks, which is the same defect seen from the other end.
  control_backend_ = CreateControlPlaneBackend(
      ResolveControlPlaneBackendType(), std::move(executor),
      absl::Seconds(control_timeout_s_));
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
  }
  // Wake workers parked in HandlePullStream waiting for a send session that
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

    // Delays the control thread ahead of the grace wait for the producer's
    // NotifyForRead registration.
    FaultInjectDelay(hooks::kKvCacheManagerPullRegisterWait);

    const absl::Duration grace =
        std::min(kPullRegistrationGrace, absl::Seconds(timeout_s_));
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      const absl::Time give_up = absl::Now() + grace;
      while (true) {
        session = FindSessionLocked<TransferSendSession>(req.uuid);
        if (session != nullptr) {
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
      if (!session) {
        throw std::runtime_error(
            absl::StrCat("no read registered for uuid ", req.uuid, " within ",
                         absl::FormatDuration(grace),
                         ": the producer expired it or never registered it"));
      }
      // Registration guards prevent a live session from being replaced, so an
      // expired session cannot become valid while this pull waits out the
      // grace.
      session->ValidateAndBeginPull(req.src_block_ids,
                                    std::chrono::steady_clock::now());
      base_->SetRemoteLayerAddrs(req.uuid, req.layer_host_addrs);
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
    absl::Cleanup counter_cleanup = [this, &session]() {
      {
        absl::MutexLock lock(pull_workers_mu_);
        --active_pull_workers_;
      }
      session->Finish(
          absl::InternalError("PullStream worker failed to launch"));
    };
    FaultInjectThrow(hooks::kKvCacheManagerPullSpawn);
    std::thread([this, session, remote_data_endpoints,
                 src_block_ids = req.src_block_ids,
                 dst_block_ids = req.dst_block_ids]() {
      try {
        session->StartPush(remote_data_endpoints, src_block_ids, dst_block_ids);
      } catch (const std::exception& e) {
        session->Finish(absl::InternalError(e.what()));
      } catch (...) {
        session->Finish(
            absl::InternalError("Unknown exception in pull worker"));
      }
      absl::MutexLock lock(pull_workers_mu_);
      --active_pull_workers_;
    }).detach();
    std::move(counter_cleanup).Cancel();

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
  return static_cast<uint64_t>(
      std::max<int64_t>(0, staging_allocator_->capacity()));
}

absl::Status KVCacheManagerWithTransfer::WaitForPendingWork() {
  RAIDEN_TRACE("KVTransfer::WaitForPendingWork");
  LOG(INFO) << "Waiting for pending transfer work to complete...";
  const absl::Time start = absl::Now();
  while (true) {
    {
      absl::MutexLock lock(mu_);
      bool pending = false;
      for (const auto& [uuid, session] : sessions_) {
        (void)uuid;
        if (auto recv =
                std::dynamic_pointer_cast<TransferReceiveSession>(session)) {
          if (recv->HasPendingWork()) {
            pending = true;
            break;
          }
        } else if (!session->Done()) {
          pending = true;
          break;
        }
      }
      if (!pending) {
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

void KVCacheManagerWithTransfer::AckSend(uint64_t uuid) {
  std::shared_ptr<TransferSendSession> session;
  {
    absl::MutexLock lock(mu_);
    session = FindSessionLocked<TransferSendSession>(uuid);
    if (session == nullptr) {
      pending_acks_.insert(uuid);
      return;
    }
  }
  session->Finish();
  LOG(INFO) << "RAIDEN_TIMING event=producer_ack"
            << " req_id=" << session->req_id() << " uuid=" << uuid
            << " node_id=" << node_id_
            << " failed=" << (!session->GetStatus().ok() ? 1 : 0);
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

}  // namespace tpu_raiden

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
#include <sstream>
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
#include "absl/strings/numbers.h"
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
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/reshard_receive_session.h"
#include "tpu_sync/kv_cache/reshard_send_session.h"
#include "tpu_sync/kv_cache/transfer_receive_session.h"
#include "tpu_sync/kv_cache/transfer_send_session.h"
#include "tpu_sync/kv_cache/transfer_session.h"
#include "tpu_sync/transport/block_transport_delegate.h"

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

void EmitTimingLog(const std::string& message) { LOG(INFO) << message; }

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

std::optional<int64_t> ReadEnvStagingWindowBlocks() {
  const char* raw = std::getenv("TPU_RAIDEN_STAGING_WINDOW_BLOCKS");
  if (raw == nullptr || raw[0] == '\0') return std::nullopt;
  if (absl::EqualsIgnoreCase(raw, "auto") || absl::string_view(raw) == "-1") {
    return -1;
  }
  if (int64_t val = 0; absl::SimpleAtoi(raw, &val) && val >= 0) return val;
  throw std::invalid_argument(absl::StrCat(
      "Invalid TPU_RAIDEN_STAGING_WINDOW_BLOCKS value: '", raw, "'"));
}

std::optional<int32_t> ReadEnvStagingWindowDepth() {
  const char* raw = std::getenv("TPU_RAIDEN_STAGING_WINDOW_DEPTH");
  if (raw == nullptr || raw[0] == '\0') return std::nullopt;
  if (int32_t val = 0; absl::SimpleAtoi(raw, &val) && val > 0) return val;
  throw std::invalid_argument(absl::StrCat(
      "Invalid TPU_RAIDEN_STAGING_WINDOW_DEPTH value: '", raw, "'"));
}

uint64_t ReadEnvStagingWindowTargetBytesPerNic(uint64_t default_bytes) {
  const char* raw =
      std::getenv("TPU_RAIDEN_STAGING_WINDOW_TARGET_BYTES_PER_NIC");
  if (raw == nullptr || raw[0] == '\0') return default_bytes;
  if (uint64_t val = 0; absl::SimpleAtoi(raw, &val) && val > 0) return val;
  throw std::invalid_argument(absl::StrCat(
      "Invalid TPU_RAIDEN_STAGING_WINDOW_TARGET_BYTES_PER_NIC value: '", raw,
      "'"));
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

int64_t SubPlanUuid(int64_t uuid, int32_t index) {
  if (index <= 0) return uuid;
  // Keep the splitmix64 result positive (63-bit) for engine request
  // compatibility.
  uint64_t z = static_cast<uint64_t>(uuid) +
               0x9E3779B97F4A7C15ull * static_cast<uint64_t>(index);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return std::max<int64_t>(
      1, static_cast<int64_t>((z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFFull));
}

size_t ResolveActiveDataNicCount(std::optional<size_t> num_ext_nics) {
  if (num_ext_nics.has_value()) return std::max<size_t>(1, *num_ext_nics);
  size_t data_nics = 0;
  for (const auto& nic : GetLocalHostNicAddresses()) {
    if (nic.classification == NicClassification::kDataPlane) ++data_nics;
  }
  return std::max<size_t>(1, data_nics);
}

int64_t ResolveDefaultStagingWindowBlocks(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    size_t num_nics, int64_t max_blocks, uint64_t target_bytes_per_nic,
    std::optional<size_t> host_pool_blocks, int32_t window_depth) {
  if (max_blocks <= 0 || num_layers == 0 || num_shards == 0 ||
      slice_byte_size == 0 || target_bytes_per_nic == 0) {
    return 0;
  }
  const uint64_t target_window_bytes =
      static_cast<uint64_t>(std::max<size_t>(1, num_nics)) *
      target_bytes_per_nic;
  const uint64_t bytes_per_block = static_cast<uint64_t>(num_layers) *
                                   static_cast<uint64_t>(num_shards) *
                                   static_cast<uint64_t>(slice_byte_size);
  const uint64_t raw_c =
      (target_window_bytes + bytes_per_block - 1) / bytes_per_block;
  int64_t c = static_cast<int64_t>(
      std::clamp<uint64_t>(raw_c, 1ULL, static_cast<uint64_t>(max_blocks)));
  if (host_pool_blocks.has_value() && *host_pool_blocks > 0) {
    const int64_t pool_cap = std::max<int64_t>(
        1, static_cast<int64_t>(
               *host_pool_blocks /
               static_cast<size_t>(std::max<int32_t>(1, window_depth))));
    c = std::min<int64_t>(c, pool_cap);
  }
  return std::max<int64_t>(1, c);
}

int64_t ResolveDefaultStagingWindowBlocks(
    const kv_cache::KVCacheManagerBase* base, int64_t max_blocks,
    uint64_t target_bytes_per_nic, int32_t window_depth) {
  if (base == nullptr) return 0;
  return ResolveDefaultStagingWindowBlocks(
      base->num_layers(), base->num_shards(), base->slice_byte_size(),
      std::max<size_t>(1, base->local_ips().size()), max_blocks,
      target_bytes_per_nic,
      base->host_block_manager()
          ? std::make_optional(base->host_block_manager()->total_blocks())
          : std::nullopt,
      window_depth);
}

int64_t ResolveHostBlocksToAllocate(std::optional<int> host_blocks_to_allocate,
                                    int64_t num_slots, int64_t max_blocks,
                                    size_t num_layers, size_t num_shards,
                                    size_t slice_byte_size, size_t num_nics,
                                    std::optional<int64_t> window_blocks,
                                    std::optional<int32_t> window_depth) {
  if (window_blocks.value_or(0) < -1) {
    throw std::invalid_argument(
        absl::StrCat("window_blocks must be >= -1, got ", *window_blocks));
  }
  if (window_depth.value_or(1) <= 0) {
    throw std::invalid_argument(
        absl::StrCat("window_depth must be > 0, got ", *window_depth));
  }
  if (host_blocks_to_allocate.has_value()) return *host_blocks_to_allocate;

  const int32_t effective_depth = window_depth.value_or(
      ReadEnvStagingWindowDepth().value_or(kDefaultStagingWindowDepth));
  int64_t c = window_blocks.value_or(ReadEnvStagingWindowBlocks().value_or(0));
  if (c == -1) {
    c = ResolveDefaultStagingWindowBlocks(
        num_layers, num_shards, slice_byte_size, num_nics, max_blocks,
        ReadEnvStagingWindowTargetBytesPerNic(
            kDefaultStagingWindowTargetBytesPerNic),
        /*host_pool_blocks=*/std::nullopt, effective_depth);
  }
  return c > 0 ? c * effective_depth : num_slots * max_blocks;
}

void KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked(
    bool is_recv, const std::string& req_id, uint64_t uuid,
    int32_t sub_plan_index, int32_t sub_plan_count) {
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
    int32_t sub_plan_index, int32_t sub_plan_count, const absl::Status& status,
    std::optional<TransferDetails> details) {
  absl::flat_hash_set<std::string>& done =
      is_recv ? done_recving_ : done_sending_;
  auto& chains = is_recv ? recv_chains_ : send_chains_;
  if (sub_plan_count <= 1) {
    // A single-plan request: the session's terminal is the request's.
    (status.ok() ? done : failed_recving_).insert(req_id);
    if (!status.ok()) {
      TransferDetails d = details.value_or(DetailsFromStatus(status));
      if (!d.code) d.code = absl::StatusCodeToString(status.code());
      if (!d.reason) d.reason = std::string(status.message());
      transfer_details_[req_id] = std::move(d);
      if (auto it = chains.find(req_id); it != chains.end()) {
        it->second.failed = true;
      }
    } else {
      if (details) transfer_details_[req_id] = *details;
      chains.erase(req_id);
    }
    return;
  }
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
    TransferDetails d = details.value_or(DetailsFromStatus(status));
    if (!d.code) d.code = absl::StatusCodeToString(status.code());
    const std::string base_reason =
        d.reason.value_or(std::string(status.message()));
    if (!absl::StartsWith(base_reason, "sub-plan ")) {
      d.reason = absl::StrCat("sub-plan ", sub_plan_index, " of ",
                              sub_plan_count, ": ", base_reason);
    }
    transfer_details_[req_id] = std::move(d);
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
    if (details) transfer_details_[req_id] = *details;
    chains.erase(req_id);
  }
}

void KVCacheManagerWithTransfer::SweepSubPlanChainsLocked(
    std::chrono::steady_clock::time_point now) {
  if (recv_chains_.empty() && send_chains_.empty()) return;
  const auto stall = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(timeout_s_));
  auto sweep = [&](bool is_recv) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto& chains = is_recv ? recv_chains_ : send_chains_;
    for (auto it = chains.begin(); it != chains.end();) {
      SubPlanChainProgress& progress = it->second;
      if (now - progress.last_activity <= stall) {
        ++it;
        continue;
      }
      if (progress.failed) {
        // The failure was reported; late sub-plans of the doomed request
        // were swallowed for one timeout, which is as long as any of them
        // can live.
        chains.erase(it++);
        continue;
      }
      // Quiet for a whole timeout: unless a sub-plan of the request is
      // still active, its chain has stalled (the coordinator never armed or
      // dispatched the next sub-plan).
      bool active = false;
      if (is_recv) {
        for (const auto& [uuid, session] : active_pool_reshard_recvs_) {
          if (session->req_id() == it->first) {
            active = true;
            break;
          }
        }
        if (!active) {
          for (const auto& [_, s] : active_recv_sessions_) {
            if (s->req_id() == it->first && !s->Done()) active = true;
          }
        }
        if (!active) {
          if (auto c = active_recv_chains_.find(it->first);
              c != active_recv_chains_.end()) {
            active = !c->second.aborted &&
                     (c->second.in_flight_sub_plans > 0 ||
                      c->second.next_sub_plan_index < c->second.sub_plan_count);
          }
        }
      } else {
        for (const auto& [uuid, session] : active_pool_reshard_sends_) {
          if (session->req_id() == it->first) {
            active = true;
            break;
          }
        }
        if (!active) {
          for (const auto& [_, s] : send_sessions_) {
            if (s->req_id() == it->first && !s->Done()) active = true;
          }
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
      LOG(WARNING) << "Pool reshard " << (is_recv ? "receive" : "send")
                   << " req_id=" << it->first << ": " << status.message();
      progress.failed = true;
      progress.last_activity = now;
      ++it;
    }
  };
  sweep(/*is_recv=*/true);
  sweep(/*is_recv=*/false);
}

void KVCacheManagerWithTransfer::InitializeBaseHooks() {
  kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
  hooks.register_block_readiness_callback =
      [this](size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
             transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
        RegisterBlockReadinessCallback(layer_idx, shard_idx, block_id, uuid,
                                       std::move(cb));
      };
  hooks.on_block_shards_received = [this](const std::vector<int>& block_ids,
                                          absl::Span<const int> shard_ids,
                                          uint64_t uuid) {
    return OnBlockShardsReceived(block_ids, shard_ids, uuid);
  };
  hooks.on_layer_received = [this](size_t layer_idx, uint64_t uuid) {
    RAIDEN_TRACE_FN("KVTransfer::OnLayerReceived", [&]() {
      return absl::StrCat("layer=", layer_idx, " uuid=", uuid);
    });
    std::shared_ptr<TransferReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_sessions_.find(uuid);
      if (it == active_recv_sessions_.end()) {
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
    std::shared_ptr<ReshardReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_pool_reshard_recvs_.find(uuid);
      if (it == active_pool_reshard_recvs_.end()) {
        auto legacy_it = active_recv_sessions_.find(uuid);
        if (legacy_it != active_recv_sessions_.end() &&
            !legacy_it->second->Done()) {
          return absl::FailedPreconditionError(
              absl::StrCat("pool completion for UUID ", uuid,
                           " but the receiver was armed on the legacy path"));
        }
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
  hooks.begin_incoming_push = [this](uint64_t uuid) -> absl::Status {
    std::shared_ptr<TransferReceiveSession> recv_session;
    std::shared_ptr<ReshardReceiveSession> reshard_session;
    {
      absl::MutexLock lock(mu_);
      if (auto it = active_recv_sessions_.find(uuid);
          it != active_recv_sessions_.end()) {
        recv_session = it->second;
      } else if (auto reshard_it = active_pool_reshard_recvs_.find(uuid);
                 reshard_it != active_pool_reshard_recvs_.end()) {
        reshard_session = reshard_it->second;
      } else if (uuid == 0 || base_->HasActivePlan(uuid)) {
        return absl::OkStatus();
      } else {
        return absl::NotFoundError(
            absl::StrCat("No active receive session for uuid=", uuid));
      }
    }
    FaultInjectDelay(hooks::kBlockTransportRecvBeginPush);
    const bool started = recv_session != nullptr
                             ? recv_session->TryBeginRecvOp()
                             : reshard_session->TryBeginRecvOp();
    return started ? absl::OkStatus()
                   : absl::CancelledError(absl::StrCat(
                         "Receive session for uuid=", uuid, " is draining"));
  };
  hooks.end_incoming_push = [this](uint64_t uuid,
                                   const absl::Status& status) -> absl::Status {
    std::shared_ptr<TransferReceiveSession> recv_session;
    std::shared_ptr<ReshardReceiveSession> reshard_session;
    {
      absl::MutexLock lock(mu_);
      if (auto it = active_recv_sessions_.find(uuid);
          it != active_recv_sessions_.end()) {
        recv_session = it->second;
      } else if (auto reshard_it = active_pool_reshard_recvs_.find(uuid);
                 reshard_it != active_pool_reshard_recvs_.end()) {
        reshard_session = reshard_it->second;
      } else {
        return absl::OkStatus();
      }
    }
    if (recv_session != nullptr) {
      if (!status.ok()) {
        recv_session->DeferUnregisterOnSettle();
        recv_session->Finish(status);
      }
      recv_session->EndRecvOp();
      MaybeUnregisterSettledRecv(uuid, *recv_session);
      return !recv_session->GetStatus().ok()
                 ? absl::CancelledError(absl::StrCat(
                       "Receive session for uuid=", uuid,
                       " failed: ", recv_session->GetStatus().message()))
                 : absl::OkStatus();
    }
    // A failed push is terminal for the pool receive (the sender does not
    // retry): fail the session so EndRecvOp settles it now, releasing its
    // staging lease and plan claim, instead of waiting for the deadline sweep.
    if (!status.ok()) {
      reshard_session->Finish(status);
    }
    reshard_session->EndRecvOp();
    uint64_t generation = 0;
    if (reshard_session->Done() &&
        reshard_session->TakePendingUnregister(&generation)) {
      UnregisterSettledPlan(uuid, generation);
    }
    return !reshard_session->GetStatus().ok()
               ? absl::CancelledError(absl::StrCat(
                     "ReshardReceiveSession for uuid=", uuid,
                     " failed: ", reshard_session->GetStatus().message()))
               : absl::OkStatus();
  };
  hooks.on_recv_progress = [this](uint64_t uuid) -> absl::Status {
    std::shared_ptr<TransferReceiveSession> recv_session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_sessions_.find(uuid);
      if (it == active_recv_sessions_.end()) {
        return absl::OkStatus();
      }
      recv_session = it->second;
    }
    return recv_session->OnRecvProgress();
  };
  hooks.on_recv_network_complete = [this](uint64_t uuid) -> absl::Status {
    std::shared_ptr<TransferReceiveSession> recv_session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_sessions_.find(uuid);
      if (it == active_recv_sessions_.end()) {
        return absl::OkStatus();
      }
      recv_session = it->second;
    }
    return recv_session->OnRecvNetworkComplete();
  };
  hooks.acquire_recv_staging =
      [this](uint64_t uuid, size_t layer_idx) -> absl::StatusOr<size_t> {
    std::shared_ptr<TransferReceiveSession> recv_session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_sessions_.find(uuid);
      if (it == active_recv_sessions_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Receive session ", uuid, " not found"));
      }
      recv_session = it->second;
    }
    return recv_session->AcquireStagingSlot(layer_idx);
  };
  hooks.lookup_send_staging =
      [this](uint64_t uuid, size_t layer_idx) -> absl::StatusOr<size_t> {
    std::shared_ptr<TransferSendSession> send_session;
    {
      absl::MutexLock lock(mu_);
      auto it = send_sessions_.find(uuid);
      if (it == send_sessions_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Send session ", uuid, " not found"));
      }
      send_session = it->second;
    }
    return send_session->LookupStagingSlot(layer_idx);
  };
  base_->SetTransferEventHooks(std::move(hooks));
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerWithTransfer(
          layer_buffers, /*slice_byte_size=*/0, /*dimensions=*/{},
          /*physical_size=*/0, local_port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism, std::move(host_allocator),
          node_id, local_control_port, max_blocks, num_slots, timeout_s,
          std::move(metrics_collector)) {}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    size_t slice_byte_size, const std::vector<int64_t>& dimensions,
    size_t physical_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerWithTransfer(
          std::make_unique<kv_cache::KVCacheManagerBase>(
              layer_buffers, local_port,
              static_cast<int>(ResolveHostBlocksToAllocate(
                  host_blocks_to_allocate, num_slots, max_blocks,
                  layer_buffers.size(),
                  layer_buffers.empty() ? 0 : layer_buffers[0].size(),
                  slice_byte_size > 0
                      ? slice_byte_size
                      : ((layer_buffers.empty() || layer_buffers[0].empty())
                             ? 0
                             : raiden::GetMajorSliceByteSize(
                                   layer_buffers[0][0].shape)),
                  ResolveActiveDataNicCount())),
              unsafe_skip_buffer_lock, parallelism, host_allocator,
              /*bind_ip=*/std::nullopt,
              slice_byte_size > 0 ? std::make_optional(slice_byte_size)
                                  : std::nullopt,
              dimensions,
              physical_size > 0 ? std::make_optional(physical_size)
                                : std::nullopt),
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
          [&] {
            const size_t first_slice_byte_size =
                slice_byte_sizes.empty() ? 0 : slice_byte_sizes[0];
            const int host_blocks =
                static_cast<int>(ResolveHostBlocksToAllocate(
                    host_blocks_to_allocate, num_slots, max_blocks, num_layers,
                    num_shards, first_slice_byte_size,
                    ResolveActiveDataNicCount()));
            return std::make_unique<kv_cache::KVCacheManagerBase>(
                num_layers, num_shards, std::move(slice_byte_sizes), local_port,
                host_blocks, parallelism, nullptr);
          }(),
          node_id, local_control_port, max_blocks, num_slots, timeout_s,
          std::move(metrics_collector)) {}

KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer() {
  StopControlServer();
  // Pull-serve workers read this object's state; nothing may be torn down
  // while one is still running.
  shutting_down_.store(true, std::memory_order_relaxed);
  std::vector<std::shared_ptr<TransferSession>> sessions_to_wait;
  std::vector<std::shared_ptr<TransferSendSession>> gated_send_sessions;
  const absl::Status cancel_status =
      absl::CancelledError("KVCacheManagerWithTransfer shutting down");
  {
    absl::MutexLock lock(mu_);
    for (auto& [_, parent] : parent_send_registrations_) {
      for (auto& s : parent.pending_h2h_sessions) {
        gated_send_sessions.push_back(std::move(s));
      }
    }
    parent_send_registrations_.clear();
    sub_uuid_to_parent_.clear();

    for (auto& [_, chain] : active_recv_chains_) {
      AbortRecvChainLocked(chain, /*terminal_status=*/nullptr,
                           /*siblings_to_cancel=*/nullptr);
    }
    active_recv_chains_.clear();

    sessions_to_wait.reserve(
        send_sessions_.size() + active_recv_sessions_.size() +
        active_pool_reshard_sends_.size() + active_pool_reshard_recvs_.size());
    for (const auto& [uuid, session] : send_sessions_) {
      (void)uuid;
      session->CancelFromManagerLocked(cancel_status);
      sessions_to_wait.push_back(session);
    }
    for (const auto& [uuid, session] : active_recv_sessions_) {
      (void)uuid;
      session->Finish(cancel_status);
      sessions_to_wait.push_back(session);
    }
    for (const auto& [uuid, session] : active_pool_reshard_sends_) {
      (void)uuid;
      session->Finish(cancel_status);
      sessions_to_wait.push_back(session);
    }
    for (const auto& [uuid, session] : active_pool_reshard_recvs_) {
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
  for (const auto& s : gated_send_sessions) {
    if (!s->IsDraining()) s->Finish(cancel_status);
    s->DropH2hGateHold();
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
    send_sessions_.clear();
    active_recv_sessions_.clear();
    active_pool_reshard_sends_.clear();
    active_pool_reshard_recvs_.clear();
    plan_staging_.clear();
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
  const int64_t env_blocks = ReadEnvStagingWindowBlocks().value_or(0);
  const bool has_env_window = (env_blocks > 0 || env_blocks == -1);
  const bool bounded_host_pool =
      base_->host_block_manager() != nullptr && num_slots > 0 &&
      max_blocks > 0 &&
      base_->host_block_manager()->total_blocks() <
          static_cast<size_t>(num_slots * max_blocks);
  staging_allocator_ = CreateStagingBlockAllocator(
      base_.get(), num_slots, max_blocks,
      (has_env_window || bounded_host_pool) ? std::make_optional(true)
                                            : std::nullopt);
  if (local_control_port_ >= 0) StartControlServer();
  if (has_env_window) {
    SetStagingWindowConfig(env_blocks, ReadEnvStagingWindowDepth().value_or(
                                           kDefaultStagingWindowDepth));
  }
}

void KVCacheManagerWithTransfer::SetStagingWindowConfig(int64_t window_blocks,
                                                        int32_t window_depth) {
  if (window_blocks < -1) {
    throw std::invalid_argument(
        absl::StrCat("window_blocks must be >= -1, got ", window_blocks));
  }
  if (window_depth <= 0) {
    throw std::invalid_argument(
        absl::StrCat("window_depth must be > 0, got ", window_depth));
  }
  window_depth_ = window_depth;
  const int64_t num_slots =
      staging_allocator_ ? staging_allocator_->num_slots() : 0;
  const int64_t max_blocks =
      staging_allocator_ ? staging_allocator_->max_blocks() : 0;
  window_blocks_ = window_blocks == -1
                       ? ResolveDefaultStagingWindowBlocks(
                             base_.get(), max_blocks,
                             ReadEnvStagingWindowTargetBytesPerNic(
                                 kDefaultStagingWindowTargetBytesPerNic),
                             window_depth_)
                       : std::max<int64_t>(0, window_blocks);

  if (window_blocks_ > 0 && staging_allocator_ != nullptr &&
      !staging_allocator_->dynamic_host_staging()) {
    std::vector<int> slot_block_ids;
    for (int64_t s = 0; s < num_slots; ++s) {
      auto blocks = staging_allocator_->slot_blocks(s);
      slot_block_ids.insert(slot_block_ids.end(), blocks.begin(), blocks.end());
    }
    staging_allocator_.reset();
    if (base_->host_block_manager() != nullptr && !slot_block_ids.empty()) {
      CheckStatus("SetStagingWindowConfig",
                  base_->host_block_manager()->Deallocate(slot_block_ids));
    }
    staging_allocator_ = CreateStagingBlockAllocator(
        base_.get(), num_slots, max_blocks, /*dynamic_host_staging=*/true);
  }
  LOG(INFO) << "Staging window config: window_blocks=" << window_blocks_
            << " window_depth=" << window_depth_ << " host_pool_blocks="
            << (base_ != nullptr && base_->host_block_manager() != nullptr
                    ? base_->host_block_manager()->total_blocks()
                    : 0)
            << " host_pool_mib="
            << (base_ != nullptr
                    ? static_cast<double>(base_->GetAllocatedHostDramBytes()) /
                          (1024.0 * 1024.0)
                    : 0.0);
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

  const auto effective_deadline = deadline.value_or(DeadlineFromNow());
  absl::StatusOr<std::shared_ptr<TransferSendSession>> created =
      TransferSendSession::Create(base_.get(), staging_allocator_.get(), req_id,
                                  uuid, block_ids, effective_deadline,
                                  register_start, /*manager=*/this);
  // An invalid registration is a caller bug rather than an empty transfer:
  // throw immediately instead of returning 0, which callers interpret as
  // "nothing to send" and then wait until the consumer's pull times out.
  if (!created.ok()) {
    throw std::invalid_argument(std::string(created.status().message()));
  }
  std::shared_ptr<TransferSendSession> session = *std::move(created);
  {
    absl::MutexLock lock(mu_);
    bool has_pending_ack = (pending_acks_.erase(uuid) > 0);
    for (int32_t c = 1;
         !pending_acks_.empty() && c < static_cast<int32_t>(block_ids.size());
         ++c) {
      has_pending_ack |= (pending_acks_.erase(SubPlanUuid(uuid, c)) > 0);
    }
    if (has_pending_ack) {
      done_sending_.insert(req_id);
      return 0;
    }
    if (send_sessions_.contains(uuid) ||
        parent_send_registrations_.contains(uuid)) {
      throw std::runtime_error(
          absl::StrCat("NotifyForRead rejected duplicate uuid=", uuid,
                       " for req_id=", req_id, ": a read is still registered"));
    }
    send_sessions_[uuid] = session;
    parent_send_registrations_[uuid] = ParentSendRegistration{
        .req_id = req_id,
        .base_uuid = uuid,
        .registered_block_set = {block_ids.begin(), block_ids.end()},
        .total_blocks = block_ids.size(),
        .deadline = effective_deadline,
        .register_start = register_start,
    };
    // Register every possible sub-plan: the consumer chooses the chunking.
    for (int32_t c = 0; c < static_cast<int32_t>(block_ids.size()); ++c) {
      sub_uuid_to_parent_[SubPlanUuid(uuid, c)] = {uuid, c};
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

absl::Status KVCacheManagerWithTransfer::EmplaceRecvSessionLocked(
    uint64_t uuid, const std::shared_ptr<TransferReceiveSession>& session) {
  RetireSettledRecvSessionLocked(uuid, /*settled_plans=*/nullptr,
                                 /*siblings_to_cancel=*/nullptr);
  // try_emplace leaves session untouched on a duplicate. Callers rely on that
  // guarantee to release staging owned by the rejected session.
  if (!active_recv_sessions_.try_emplace(uuid, session).second) {
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
  // Staging ownership is settled before the plan is published. An HBM
  // receiver's blocks belong to its receive session and return when the
  // upload settles; a sender's blocks, and a host-memory receiver's,
  // belong to the plan and return when it is unregistered.
  const bool hbm_receiver =
      !is_sender && request.dst_mem_type() == ::tpu_sync::rpc::MEMORY_TYPE_HBM;
  // 2. If we are the receiver and the destination memory type is HBM,
  //    populate active_recv_sessions_ to enable automatic H2D copy!
  if (hbm_receiver) {
    absl::MutexLock lock(mu_);
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<TransferReceiveSession> recv_session,
        TransferReceiveSession::CreateFromActivePlan(
            base_.get(), staging_allocator_.get(), uuid, request, generation,
            DeadlineFromNow(), &host_block_of, metrics_collector_));

    if (recv_session->total_blocks() > 0) {
      absl::Status inserted = EmplaceRecvSessionLocked(uuid, recv_session);
      if (!inserted.ok()) {
        recv_session->ReleaseStaging();
        return inserted;
      }
      LOG(INFO) << "RegisterActivePlan (Receiver): Populated "
                   "active_recv_sessions_ for UUID "
                << uuid << " with " << recv_session->total_blocks()
                << " total physical block-pushes (including duplicates across "
                   "sources) for automatic H2D.";
    }
  } else if (staging_allocator_->dynamic_host_staging() &&
             request.pool_groups_size() == 0) {
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
      absl::StatusOr<StagingAllocation> allocated = staging_allocator_->Acquire(
          static_cast<int64_t>(device_blocks.size()));
      if (!allocated.ok()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "cannot stage ", device_blocks.size(), " blocks for plan ", uuid,
            ": ", allocated.status().message()));
      }
      StagingAllocation plan_staging = *std::move(allocated);
      absl::Span<const int> plan_blocks = plan_staging.block_ids();
      for (size_t i = 0; i < device_blocks.size(); ++i) {
        host_block_of[device_blocks[i]] = plan_blocks[i];
      }
      plan_staging_[uuid] = std::move(plan_staging);
    }
  }

  // Publish the plan last: pushes resolve through it, so everything they
  // may touch exists by the time it is visible.
  absl::Status registered = base_->RegisterActivePlan(
      uuid, request, is_sender, host_block_of, generation);
  if (!registered.ok()) {
    absl::MutexLock lock(mu_);
    plan_staging_.erase(uuid);
    auto recv = active_recv_sessions_.find(uuid);
    if (recv != active_recv_sessions_.end()) {
      recv->second->ReleaseStaging();
      active_recv_sessions_.erase(recv);
    }
    return registered;
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::RegisterRecv(
    uint64_t uuid, const std::string& req_id, int64_t expected_block_count,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  absl::MutexLock lock(mu_);
  ASSIGN_OR_RETURN(
      std::shared_ptr<TransferReceiveSession> recv_session,
      TransferReceiveSession::Create(
          base_.get(), staging_allocator_.get(), uuid, req_id,
          expected_block_count, deadline.value_or(DeadlineFromNow()),
          /*acquire_staging=*/false, metrics_collector_));
  // host_to_chip is left empty -> defaults to 1-to-1 mapping in
  // OnBlockShardsReceived
  absl::Status inserted = EmplaceRecvSessionLocked(uuid, recv_session);
  if (!inserted.ok()) {
    return inserted;
  }
  VLOG(1)
      << "RegisterRecv (Receiver): Registered expected block count for UUID "
      << uuid << " with " << expected_block_count << " expected blocks.";
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::PoolReshardPush(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> src_block_ids, int parallelism) {
  RAIDEN_TRACE_FN("KVTransfer::PoolReshardPush", [&]() {
    return absl::StrCat("uuid=", plan.uuid(),
                        " src_blocks=", src_block_ids.size());
  });
  {
    absl::MutexLock lock(mu_);
    auto existing = active_pool_reshard_sends_.find(plan.uuid());
    if (existing != active_pool_reshard_sends_.end()) {
      if (existing->second->Done()) {
        RecordPoolReshardTerminalLocked(
            /*is_recv=*/false, existing->second->req_id(),
            existing->second->uuid(),
            existing->second->sub_plan_index(),
            existing->second->sub_plan_count(), existing->second->GetStatus());
        active_pool_reshard_sends_.erase(existing);
      } else {
        return absl::AlreadyExistsError(absl::StrCat(
            "pool reshard send UUID already active: ", plan.uuid()));
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<ReshardSendSession> state,
      ReshardSendSession::Create(base_.get(), staging_allocator_.get(),
                                 src_block_ids, parallelism, DeadlineFromNow(),
                                 plan));
  {
    absl::MutexLock lock(mu_);
    active_pool_reshard_sends_[plan.uuid()] = state;
    NoteSubPlanRegisteredLocked(/*is_recv=*/false, plan.req_id(), plan.uuid(),
                                plan.sub_plan_index(), plan.sub_plan_count());
  }

  return state->ExecutePush(*this, src_block_ids);
}

absl::Status KVCacheManagerWithTransfer::PoolReshardRegisterRecv(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_block_ids) {
  RAIDEN_TRACE_FN("KVTransfer::PoolReshardRegisterRecv", [&]() {
    return absl::StrCat("uuid=", plan.uuid(),
                        " chip_blocks=", chip_block_ids.size());
  });
  {
    absl::MutexLock lock(mu_);
    auto existing_reshard = active_pool_reshard_recvs_.find(plan.uuid());
    if (existing_reshard != active_pool_reshard_recvs_.end()) {
      if (existing_reshard->second->Done()) {
        RecordPoolReshardTerminalLocked(
            /*is_recv=*/true, existing_reshard->second->req_id(),
            existing_reshard->second->uuid(),
            existing_reshard->second->sub_plan_index(),
            existing_reshard->second->sub_plan_count(),
            existing_reshard->second->GetStatus());
        active_pool_reshard_recvs_.erase(existing_reshard);
      } else {
        return absl::AlreadyExistsError(absl::StrCat(
            "pool reshard recv UUID already active: ", plan.uuid()));
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<ReshardReceiveSession> recv_session,
      ReshardReceiveSession::Create(base_.get(), staging_allocator_.get(), plan,
                                    chip_block_ids, DeadlineFromNow()));
  {
    absl::MutexLock lock(mu_);
    active_pool_reshard_recvs_[plan.uuid()] = std::move(recv_session);
    NoteSubPlanRegisteredLocked(/*is_recv=*/true, plan.req_id(), plan.uuid(),
                                plan.sub_plan_index(), plan.sub_plan_count());
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::UnregisterActivePlan(uint64_t uuid) {
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  std::shared_ptr<TransferReceiveSession> recv;
  std::shared_ptr<ReshardReceiveSession> reshard_recv;
  {
    absl::MutexLock lock(mu_);
    plan_staging_.erase(uuid);
    if (auto it = active_recv_sessions_.find(uuid);
        it != active_recv_sessions_.end() && !it->second->Done()) {
      recv = it->second;
    } else if (auto reshard_it = active_pool_reshard_recvs_.find(uuid);
               reshard_it != active_pool_reshard_recvs_.end() &&
               !reshard_it->second->Done()) {
      reshard_recv = reshard_it->second;
    }
  }
  // Unregistering a live receive, on either the block-addressed or the
  // pool-reshard path, cancels it: no further push is admitted
  // (TryBeginRecvOp) and no further H2D starts. Work the transport already
  // accepted keeps the plan -- its writes resolve through the plan's staging
  // -- until it ends; the plan then goes with the session, together with
  // the staging it leased, instead of outliving the receive until the
  // deadline sweep. An idle receive settles here and its plan is dropped
  // at once.
  if (recv != nullptr) {
    recv->DeferUnregisterOnSettle();
    recv->Finish(absl::CancelledError("receive plan unregistered"));
    if (!recv->Done()) return absl::OkStatus();
    uint64_t generation = 0;
    recv->TakePendingUnregister(&generation);
  } else if (reshard_recv != nullptr) {
    reshard_recv->Finish(absl::CancelledError("receive plan unregistered"));
    if (!reshard_recv->Done()) return absl::OkStatus();
    uint64_t generation = 0;
    reshard_recv->TakePendingUnregister(&generation);
  }
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
    uint64_t uuid, TransferReceiveSession& session) {
  if (!session.Done()) return;
  uint64_t generation = 0;
  if (session.TakePendingUnregister(&generation)) {
    UnregisterSettledPlan(uuid, generation);
  }
  OnRecvSessionSettled(uuid, session);
}

void KVCacheManagerWithTransfer::RetireSettledRecvSessionLocked(
    uint64_t uuid, std::vector<std::pair<uint64_t, uint64_t>>* settled_plans,
    std::vector<std::shared_ptr<TransferReceiveSession>>* siblings_to_cancel) {
  auto it = active_recv_sessions_.find(uuid);
  if (it == active_recv_sessions_.end() || !it->second->Done()) return;
  std::shared_ptr<TransferReceiveSession> session = it->second;
  if (settled_plans != nullptr) {
    uint64_t gen = 0;
    if (session->TakePendingUnregister(&gen)) {
      settled_plans->emplace_back(uuid, gen);
    }
  }
  active_recv_sessions_.erase(it);

  if (auto rc = recv_chains_.find(session->req_id());
      session->sub_plan_count() > 1 || rc == recv_chains_.end() ||
      !rc->second.anchored || rc->second.plan_uuids.contains(uuid)) {
    RecordPoolReshardTerminalLocked(
        /*is_recv=*/true, session->req_id(), uuid, session->sub_plan_index(),
        session->sub_plan_count(), session->GetStatus(), session->Details());
  }

  if (auto chain_it = active_recv_chains_.find(session->req_id());
      chain_it != active_recv_chains_.end()) {
    ActiveRecvChain& c = chain_it->second;
    if (uuid == SubPlanUuid(c.base_uuid, session->sub_plan_index())) {
      c.in_flight_sub_plans = std::max<int32_t>(0, c.in_flight_sub_plans - 1);
      if (!session->GetStatus().ok()) {
        AbortRecvChainLocked(c, /*terminal_status=*/nullptr,
                             siblings_to_cancel);
      }
      if (c.in_flight_sub_plans == 0 &&
          (c.aborted || c.next_sub_plan_index >= c.sub_plan_count)) {
        active_recv_chains_.erase(chain_it);
      }
    }
  }
}

void KVCacheManagerWithTransfer::AbortRecvChainLocked(
    ActiveRecvChain& chain, const absl::Status* terminal_status,
    std::vector<std::shared_ptr<TransferReceiveSession>>* siblings_to_cancel) {
  chain.aborted = true;
  if (terminal_status != nullptr) {
    RecordPoolReshardTerminalLocked(
        /*is_recv=*/true, chain.req_id,
        SubPlanUuid(chain.base_uuid, chain.next_sub_plan_index),
        chain.next_sub_plan_index, chain.sub_plan_count, *terminal_status);
  }
  if (siblings_to_cancel != nullptr) {
    for (int32_t c = 0; c < chain.next_sub_plan_index; ++c) {
      auto sib_it = active_recv_sessions_.find(SubPlanUuid(chain.base_uuid, c));
      if (sib_it != active_recv_sessions_.end() && !sib_it->second->Done()) {
        siblings_to_cancel->push_back(sib_it->second);
      }
    }
  }
}

void KVCacheManagerWithTransfer::CancelAndUnregisterRecvSiblings(
    absl::Span<const std::shared_ptr<TransferReceiveSession>> siblings) {
  for (const auto& sib : siblings) {
    if (!sib->IsDraining()) {
      sib->Finish(absl::CancelledError("Receive chain aborted"));
    }
    MaybeUnregisterSettledRecv(sib->uuid(), *sib);
  }
}

void KVCacheManagerWithTransfer::DispatchNextRecvSubPlansLocked(
    ActiveRecvChain& chain, std::vector<DispatchedSubRecv>* to_start,
    std::vector<std::shared_ptr<TransferReceiveSession>>* siblings_to_cancel) {
  const size_t total_blocks = chain.remote_block_ids.size();
  const size_t c_blocks =
      static_cast<size_t>(std::max<int64_t>(1, chain.window_blocks));
  while (!IsShuttingDown() && !chain.aborted &&
         chain.in_flight_sub_plans < chain.window_depth &&
         chain.next_sub_plan_index < chain.sub_plan_count) {
    const int32_t c = chain.next_sub_plan_index;
    const size_t begin = static_cast<size_t>(c) * c_blocks;
    if (begin >= total_blocks) break;
    const size_t count = std::min(total_blocks, begin + c_blocks) - begin;
    const uint64_t sub_uuid = SubPlanUuid(chain.base_uuid, c);
    RetireSettledRecvSessionLocked(sub_uuid, nullptr, siblings_to_cancel);

    auto created = TransferReceiveSession::Create(
        base_.get(), staging_allocator_.get(), sub_uuid, chain.req_id,
        {chain.remote_block_ids.begin() + begin,
         chain.remote_block_ids.begin() + begin + count},
        {chain.local_block_ids.begin() + begin,
         chain.local_block_ids.begin() + begin + count},
        std::nullopt, chain.deadline, c, chain.sub_plan_count,
        metrics_collector_);
    absl::Status status = created.ok()
                              ? EmplaceRecvSessionLocked(sub_uuid, *created)
                              : created.status();
    if (!status.ok()) {
      if (created.ok()) (*created)->ReleaseStaging();
      if (!created.ok() && absl::IsResourceExhausted(status)) break;
      AbortRecvChainLocked(chain, &status, siblings_to_cancel);
      break;
    }
    ++chain.next_sub_plan_index;
    ++chain.in_flight_sub_plans;
    if (to_start)
      to_start->push_back({*std::move(created), chain.remote_endpoint});
  }
}

void KVCacheManagerWithTransfer::ResumeOrExpireRecvChainsLocked(
    std::chrono::steady_clock::time_point now,
    std::vector<DispatchedSubRecv>* to_start,
    std::vector<std::shared_ptr<TransferReceiveSession>>* siblings_to_cancel) {
  for (auto it = active_recv_chains_.begin();
       it != active_recv_chains_.end();) {
    ActiveRecvChain& chain = it->second;
    if (!chain.aborted && chain.next_sub_plan_index < chain.sub_plan_count) {
      if (chain.deadline <= now) {
        const absl::Status s =
            absl::DeadlineExceededError("Receive session timed out");
        AbortRecvChainLocked(chain, &s, siblings_to_cancel);
      } else {
        DispatchNextRecvSubPlansLocked(chain, to_start, siblings_to_cancel);
      }
    }
    if (chain.in_flight_sub_plans == 0 &&
        (chain.aborted || chain.next_sub_plan_index >= chain.sub_plan_count)) {
      active_recv_chains_.erase(it++);
    } else {
      ++it;
    }
  }
}

void KVCacheManagerWithTransfer::StartDispatchedRecvSessions(
    std::vector<DispatchedSubRecv> to_start) {
  for (auto& item : to_start) {
    const int64_t num_blocks = item.session->total_blocks();
    if (metrics_collector_) {
      const uint64_t bytes = static_cast<uint64_t>(num_blocks) *
                             base_->num_layers() * base_->num_shards() *
                             base_->slice_byte_size();
      metrics_collector_->RecordStart(
          item.session->uuid(), item.session->req_id(), num_blocks, bytes);
    }
    if (num_blocks == 0) {
      item.session->Finish();
      MaybeUnregisterSettledRecv(item.session->uuid(), *item.session);
      continue;
    }
    if (absl::Status status =
            FaultInjectStatus(hooks::kKvCacheManagerApiStartReadSubmitPull);
        !status.ok()) {
      item.session->Finish(status);
      item.session->EndRecvOp();
      MaybeUnregisterSettledRecv(item.session->uuid(), *item.session);
      continue;
    }
    item.session->ExecutePullRequest(*this, item.remote_endpoint);
  }
}

void KVCacheManagerWithTransfer::OnRecvSessionSettled(
    uint64_t uuid, TransferReceiveSession& session) {
  if (IsShuttingDown()) return;
  std::vector<std::pair<uint64_t, uint64_t>> settled_plans;
  std::vector<std::shared_ptr<TransferReceiveSession>> siblings_to_cancel;
  std::vector<DispatchedSubRecv> to_start;
  {
    absl::MutexLock lock(mu_);
    if (session.sub_plan_count() > 1 ||
        active_recv_chains_.contains(session.req_id())) {
      RetireSettledRecvSessionLocked(uuid, &settled_plans, &siblings_to_cancel);
    }
    ResumeOrExpireRecvChainsLocked(std::chrono::steady_clock::now(), &to_start,
                                   &siblings_to_cancel);
  }
  for (const auto& [plan_uuid, generation] : settled_plans) {
    UnregisterSettledPlan(plan_uuid, generation);
  }
  CancelAndUnregisterRecvSiblings(siblings_to_cancel);
  StartDispatchedRecvSessions(std::move(to_start));
}

void KVCacheManagerWithTransfer::MaybeUnregisterSettledReshardRecv(
    uint64_t uuid, ReshardReceiveSession& session) {
  uint64_t generation = 0;
  if (session.Done() && session.TakePendingUnregister(&generation)) {
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
  std::vector<std::pair<uint64_t, uint64_t>> settled_plans;
  std::vector<std::shared_ptr<TransferReceiveSession>> siblings_to_cancel;
  std::vector<DispatchedSubRecv> to_start;
  {
    absl::MutexLock lock(mu_);
    const int64_t c_blocks = std::max<int64_t>(1, window_blocks_);
    const int32_t w_depth = std::max<int32_t>(1, window_depth_);
    const size_t k = remote_block_ids.size();
    const bool use_bounded_window = window_blocks_ > 0 &&
                                    !local_host_block_ids.has_value() &&
                                    !remote_block_ids.empty();
    const int32_t sub_plan_count =
        use_bounded_window
            ? static_cast<int32_t>((k + static_cast<size_t>(c_blocks) - 1) /
                                   static_cast<size_t>(c_blocks))
            : 1;

    std::string incumbent_req_id;
    for (int32_t c = 0; c < sub_plan_count && incumbent_req_id.empty(); ++c) {
      const uint64_t cand_uuid = SubPlanUuid(uuid, c);
      RetireSettledRecvSessionLocked(cand_uuid, &settled_plans,
                                     &siblings_to_cancel);
      if (auto it = active_recv_sessions_.find(cand_uuid);
          it != active_recv_sessions_.end() && !it->second->Done()) {
        incumbent_req_id = it->second->req_id();
        break;
      }
      for (const auto& [_, chain] : active_recv_chains_) {
        for (int32_t sc = 0; sc < chain.sub_plan_count; ++sc) {
          if (static_cast<uint64_t>(SubPlanUuid(chain.base_uuid, sc)) ==
              cand_uuid) {
            incumbent_req_id = chain.req_id;
            break;
          }
        }
        if (!incumbent_req_id.empty()) break;
      }
    }

    if (!incumbent_req_id.empty()) {
      LOG(ERROR) << "StartRead rejected duplicate uuid=" << uuid
                 << " for req_id=" << req_id;
      if (incumbent_req_id != req_id) {
        failed_recving_.insert(req_id);
        transfer_details_[req_id] = DetailsFromStatus(absl::AlreadyExistsError(
            absl::StrCat("StartRead rejected duplicate uuid=", uuid,
                         ", already receiving req_id=", incumbent_req_id)));
      }
    } else if (use_bounded_window) {
      if (auto old_it = active_recv_chains_.find(req_id);
          old_it != active_recv_chains_.end()) {
        AbortRecvChainLocked(old_it->second, /*terminal_status=*/nullptr,
                             &siblings_to_cancel);
      }
      for (int32_t c = 0; c < sub_plan_count; ++c) {
        NoteSubPlanRegisteredLocked(/*is_recv=*/true, req_id,
                                    SubPlanUuid(uuid, c), c, sub_plan_count);
      }
      ActiveRecvChain& chain = (active_recv_chains_[req_id] = ActiveRecvChain{
                                    .req_id = req_id,
                                    .remote_endpoint = remote_endpoint,
                                    .base_uuid = uuid,
                                    .remote_block_ids = remote_block_ids,
                                    .local_block_ids = local_block_ids,
                                    .deadline = effective_deadline,
                                    .window_blocks = c_blocks,
                                    .window_depth = w_depth,
                                    .sub_plan_count = sub_plan_count});
      DispatchNextRecvSubPlansLocked(chain, &to_start, &siblings_to_cancel);
      if (chain.in_flight_sub_plans == 0 &&
          (chain.aborted ||
           chain.next_sub_plan_index >= chain.sub_plan_count)) {
        active_recv_chains_.erase(req_id);
      }
    } else {
      NoteSubPlanRegisteredLocked(/*is_recv=*/true, req_id, uuid, 0, 1);
      auto created = TransferReceiveSession::Create(
          base_.get(), staging_allocator_.get(), uuid, req_id, remote_block_ids,
          local_block_ids, local_host_block_ids, effective_deadline,
          /*sub_plan_index=*/0, /*sub_plan_count=*/1, metrics_collector_);
      absl::Status inserted = created.ok()
                                  ? EmplaceRecvSessionLocked(uuid, *created)
                                  : created.status();
      if (!inserted.ok()) {
        if (created.ok()) (*created)->ReleaseStaging();
        failed_recving_.insert(req_id);
        transfer_details_[req_id] = DetailsFromStatus(inserted);
        LOG(ERROR) << "StartRead failed to register req_id=" << req_id
                   << ", uuid=" << uuid << ": " << inserted.message();
      } else {
        to_start.push_back({*std::move(created), remote_endpoint});
      }
    }
  }

  for (const auto& [plan_uuid, generation] : settled_plans) {
    UnregisterSettledPlan(plan_uuid, generation);
  }
  CancelAndUnregisterRecvSiblings(siblings_to_cancel);
  StartDispatchedRecvSessions(std::move(to_start));
}

CompleteReadResult KVCacheManagerWithTransfer::CompleteReadWithDetails() {
  RAIDEN_TRACE("KVTransfer::CompleteReadWithDetails");
  FaultInjectThrow(hooks::kKvCacheManagerApiCompleteRead);
  CompleteReadResult completions;
  std::vector<std::pair<uint64_t, uint64_t>> settled_plans;
  std::vector<std::shared_ptr<TransferSendSession>> sessions_to_release;
  std::vector<std::shared_ptr<TransferReceiveSession>> siblings_to_cancel;
  std::vector<DispatchedSubRecv> to_start;
  const auto now = std::chrono::steady_clock::now();
  {
    absl::MutexLock lock(mu_);
    std::vector<uint64_t> send_uuids;
    for (const auto& [uuid, session] : send_sessions_) {
      if (!session->IsDraining() && session->deadline() <= now) {
        session->CancelFromManagerLocked(
            absl::DeadlineExceededError("Send session timed out"));
      }
      send_uuids.push_back(uuid);
    }
    for (auto it = parent_send_registrations_.begin();
         it != parent_send_registrations_.end();) {
      std::erase_if(it->second.pending_h2h_sessions, [&](auto& s) {
        if (!s->IsDraining()) return false;
        sessions_to_release.push_back(std::move(s));
        return true;
      });
      if (it->second.deadline <= now &&
          std::none_of(send_sessions_.begin(), send_sessions_.end(),
                       [&](const auto& kv) {
                         return kv.second->base_uuid() == it->first;
                       })) {
        if (!it->second.failed) {
          failed_recving_.insert(it->second.req_id);
          transfer_details_[it->second.req_id] = DetailsFromStatus(
              absl::DeadlineExceededError("Send session timed out"));
        }
        for (auto& s : it->second.pending_h2h_sessions) {
          sessions_to_release.push_back(std::move(s));
        }
        CleanupParentSendRegistrationLocked((it++)->first);
      } else {
        ++it;
      }
    }
    for (uint64_t uuid : send_uuids) {
      RetireSettledSendSessionLocked(uuid, &settled_plans,
                                     &sessions_to_release);
    }
    for (auto it = active_pool_reshard_sends_.begin();
         it != active_pool_reshard_sends_.end();) {
      const auto& session = it->second;
      if (!session->IsDraining() && session->deadline() <= now) {
        session->Finish(
            absl::DeadlineExceededError("Pool reshard send timed out"));
      }
      if (session->Done()) {
        const bool failed = !session->GetStatus().ok();
        RecordPoolReshardTerminalLocked(
            /*is_recv=*/false, session->req_id(), session->uuid(),
            session->sub_plan_index(),
            session->sub_plan_count(), session->GetStatus());
        if (failed) {
          settled_plans.emplace_back(it->first, 0);
        }
        active_pool_reshard_sends_.erase(it++);
      } else {
        ++it;
      }
    }
    // Reclaim recv sessions whose transfer never completed (e.g. the producer
    // died or never finished pushing). Without this the session and its host
    // staging slot leak forever, eventually exhausting the slot pool. Surface
    // the timeout as a recv failure so the connector can recompute the blocks.
    std::vector<uint64_t> recv_uuids;
    for (const auto& [uuid, session] : active_recv_sessions_) {
      if (!session->IsDraining()) {
        if (session->IsReadyToComplete()) {
          LOG(INFO) << "CompleteReadRaw (polling completion): req_id="
                    << session->req_id();
          session->Finish();
        } else if (session->deadline() <= now) {
          // Preserve the pre-existing timeout cleanup behavior even for
          // receives registered without a plan: UnregisterSettledPlan also
          // clears any transport-side progress associated with the UUID.
          session->DeferUnregisterOnSettle();
          session->Finish(
              absl::DeadlineExceededError("Receive session timed out"));
        }
      }
      recv_uuids.push_back(uuid);
    }
    for (uint64_t uuid : recv_uuids) {
      RetireSettledRecvSessionLocked(uuid, &settled_plans, &siblings_to_cancel);
    }
    for (auto it = active_pool_reshard_recvs_.begin();
         it != active_pool_reshard_recvs_.end();) {
      const uint64_t uuid = it->first;
      const std::shared_ptr<ReshardReceiveSession> session = it->second;
      if (!session->IsDraining()) {
        if (session->IsReadyToComplete()) {
          LOG(INFO) << "CompleteReadRaw (polling completion): req_id="
                    << session->req_id();
          session->Finish();
        } else if (session->deadline() <= now) {
          session->Finish(
              absl::DeadlineExceededError("Pool reshard receive timed out"));
        }
      }

      if (session->Done()) {
        RecordPoolReshardTerminalLocked(
            /*is_recv=*/true, session->req_id(), session->uuid(),
            session->sub_plan_index(),
            session->sub_plan_count(), session->GetStatus());
        uint64_t generation = 0;
        if (session->TakePendingUnregister(&generation)) {
          settled_plans.emplace_back(uuid, generation);
        }
        active_pool_reshard_recvs_.erase(it++);
      } else {
        ++it;
      }
    }
    ResumeOrExpireRecvChainsLocked(now, &to_start, &siblings_to_cancel);
    SweepSubPlanChainsLocked(now);
    completions.sent.assign(done_sending_.begin(), done_sending_.end());
    completions.received.assign(done_recving_.begin(), done_recving_.end());
    completions.failed.assign(failed_recving_.begin(), failed_recving_.end());
    completions.details.swap(transfer_details_);
    done_sending_.clear();
    done_recving_.clear();
    failed_recving_.clear();
  }
  for (const auto& s : sessions_to_release) {
    if (!s->IsDraining()) s->Finish(absl::CancelledError("Send chain aborted"));
    s->DropH2hGateHold();
  }
  CancelAndUnregisterRecvSiblings(siblings_to_cancel);
  // Unregistering drops the plan and its transport receive-progress counters
  // (ForgetPushProgress), so a settled uuid is reusable.
  for (const auto& [uuid, generation] : settled_plans) {
    UnregisterSettledPlan(uuid, generation);
    // A settled (completed or timed-out) pool-reshard sender/receiver may
    // still hold bounded-staging arena slots.
    base_->ReleasePoolStagingLeases(uuid);
  }
  StartDispatchedRecvSessions(std::move(to_start));
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

void KVCacheManagerWithTransfer::RegisterBlockReadinessCallback(
    size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
    transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
  if (block_id < 0 || staging_allocator_ == nullptr ||
      staging_allocator_->max_blocks() <= 0) {
    cb(absl::OkStatus());
    return;
  }
  if (uuid == kLeaseAuthorizedPullUuid) {
    // A lease-authorised pull. The reader holds a lease over blocks this node
    // already published as host-resident, so their device-to-host copy
    // provably completed before the lease was granted, and the pin keeps them
    // from being reused for the duration of the read. Gating here would add
    // nothing -- and would be actively wrong, because the fallback scan below
    // can only match some OTHER transfer's session, making this read wait on a
    // future that has nothing to do with it.
    cb(absl::OkStatus());
    return;
  }
  std::shared_ptr<TransferSendSession> session;
  {
    absl::MutexLock lock(mu_);
    // Exact match: the transfer named itself, so gate on its own D2H. Reached
    // by uuid-carrying senders; a pull that did not identify itself falls
    // through to the scan below.
    auto it = send_sessions_.find(uuid);
    if (it != send_sessions_.end() && !it->second->Done()) {
      session = it->second;
    } else {
      // Fallback: no session owns this uuid, so look for any live transfer that
      // registered this block id and wait on ITS copy. Conservative and
      // imprecise -- the match is by block id alone, so an unrelated transfer
      // can gate this one, and a stale session whose future never resolves
      // would stall it until the reader's own deadline fires.
      for (const auto& [u, s] : send_sessions_) {
        if (!s->Done() && s->OwnsBlockWithReadyFuture(block_id, layer_idx)) {
          session = s;
          break;
        }
      }
    }
  }
  if (!session) {
    cb(absl::OkStatus());
    return;
  }
  session->RegisterLayerReadinessCallback(layer_idx, std::move(cb));
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
        if (sub_uuid_to_parent_.contains(req.uuid)) break;
        const absl::Duration left = give_up - absl::Now();
        if (stopping_.load() || left <= absl::ZeroDuration()) break;
        cv_.WaitWithTimeout(&mu_, left);
      }
      if (stopping_.load()) {
        return PullStreamResponseSpec{
            .status = -1, .message = "Producer control server is stopping"};
      }

      const auto now = std::chrono::steady_clock::now();
      auto sub_it = sub_uuid_to_parent_.find(req.uuid);
      auto parent_it =
          sub_it != sub_uuid_to_parent_.end()
              ? parent_send_registrations_.find(sub_it->second.base_uuid)
              : parent_send_registrations_.end();
      if (parent_it == parent_send_registrations_.end()) {
        throw std::runtime_error(
            absl::StrCat("no read registered for uuid ", req.uuid));
      }
      ParentSendRegistration& parent = parent_it->second;
      if (parent.failed) {
        throw std::runtime_error(absl::StrCat("read registration for uuid ",
                                              req.uuid, " has failed"));
      }
      if (now >= parent.deadline) {
        throw std::runtime_error(
            absl::StrCat("read registration for uuid ", req.uuid, " expired"));
      }
      for (int64_t b : req.src_block_ids) {
        if (!parent.registered_block_set.contains(b)) {
          throw std::invalid_argument(
              "pull stream requested block not registered by producer");
        }
        if (parent.claimed_blocks.contains(b)) {
          throw std::invalid_argument(
              "pull stream requested block already claimed");
        }
      }
      const int32_t sub_idx = sub_it->second.sub_plan_index;
      auto it = send_sessions_.find(req.uuid);
      if (sub_idx == 0) {
        if (it == send_sessions_.end()) {
          throw std::runtime_error(
              absl::StrCat("pull already started for uuid ", req.uuid));
        }
        session = it->second;
      } else {
        if (it != send_sessions_.end()) {
          throw std::runtime_error(
              absl::StrCat("pull already started for uuid ", req.uuid));
        }
        auto created = TransferSendSession::Create(
            base_.get(), staging_allocator_.get(), parent.req_id, req.uuid,
            req.src_block_ids, parent.deadline, parent.register_start, this,
            parent.base_uuid, sub_idx);
        if (!created.ok()) {
          throw std::runtime_error(std::string(created.status().message()));
        }
        session = *std::move(created);
      }
      session->ValidateAndBeginPull(req.src_block_ids, now);
      if (sub_idx > 0) send_sessions_.emplace(req.uuid, session);
      parent.claimed_blocks.insert(req.src_block_ids.begin(),
                                   req.src_block_ids.end());
      ++parent.sub_pulls_dispatched;
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
    std::vector<DispatchedSubRecv> to_start;
    std::vector<std::shared_ptr<TransferReceiveSession>> siblings_to_cancel;
    {
      absl::MutexLock lock(mu_);
      ResumeOrExpireRecvChainsLocked(std::chrono::steady_clock::now(),
                                     &to_start, &siblings_to_cancel);
      bool recv_pending = false;
      for (const auto& [uuid, session] : active_recv_sessions_) {
        (void)uuid;
        if (!session->Done()) {
          recv_pending = true;
          break;
        }
      }
      if (!recv_pending) {
        for (const auto& [_, chain] : active_recv_chains_) {
          if (!chain.aborted &&
              (chain.in_flight_sub_plans > 0 ||
               chain.next_sub_plan_index < chain.sub_plan_count)) {
            recv_pending = true;
            break;
          }
        }
      }
      if (!recv_pending) {
        for (const auto& [uuid, session] : active_pool_reshard_recvs_) {
          (void)uuid;
          if (session->HasPendingWork()) {
            recv_pending = true;
            break;
          }
        }
      }
      bool send_pending = false;
      for (const auto& [uuid, session] : active_pool_reshard_sends_) {
        (void)uuid;
        if (!session->Done()) {
          send_pending = true;
          break;
        }
      }
      if (!send_pending) {
        for (const auto& [uuid, session] : send_sessions_) {
          (void)uuid;
          if (!session->Done()) {
            send_pending = true;
            break;
          }
        }
      }
      if (!recv_pending && !send_pending && to_start.empty()) {
        break;
      }
      const absl::Duration elapsed = absl::Now() - start;
      if (elapsed > kPendingWorkTimeout) {
        return absl::DeadlineExceededError(
            "Timeout waiting for pending transfer work");
      }
    }
    CancelAndUnregisterRecvSiblings(siblings_to_cancel);
    StartDispatchedRecvSessions(std::move(to_start));
    absl::SleepFor(absl::Milliseconds(100));
  }
  LOG(INFO) << "All pending transfer work completed.";
  return absl::OkStatus();
}

void KVCacheManagerWithTransfer::AckSend(uint64_t uuid) {
  std::shared_ptr<TransferSendSession> session;
  {
    absl::MutexLock lock(mu_);
    auto it = send_sessions_.find(uuid);
    if (it == send_sessions_.end()) {
      pending_acks_.insert(uuid);
      return;
    }
    session = it->second;
  }
  session->Finish();
  const auto ack_done = std::chrono::steady_clock::now();
  std::ostringstream timing;
  timing << "RAIDEN_TIMING event=producer_ack"
         << " req_id=" << session->req_id() << " uuid=" << session->uuid()
         << " node_id=" << node_id_ << " blocks=" << session->num_blocks()
         << " bytes=" << session->total_bytes()
         << " stage_to_ack_ms=" << DurationMs(session->d2h_done(), ack_done)
         << " register_to_ack_ms="
         << DurationMs(session->register_start(), ack_done)
         << " failed=" << (!session->GetStatus().ok() ? 1 : 0);
  EmitTimingLog(timing.str());
}

bool KVCacheManagerWithTransfer::TryEnterSendH2hGate(
    uint64_t base_uuid, int32_t sub_plan_index,
    std::shared_ptr<TransferSendSession> session, bool* rejected) {
  absl::MutexLock lock(mu_);
  auto parent_it = parent_send_registrations_.find(base_uuid);
  if (IsShuttingDown() || session->IsDraining() ||
      (parent_it != parent_send_registrations_.end() &&
       parent_it->second.failed)) {
    *rejected = true;
    return false;
  }
  *rejected = false;
  if (parent_it == parent_send_registrations_.end()) return true;
  ParentSendRegistration& parent = parent_it->second;
  if (!parent.h2h_in_flight &&
      sub_plan_index == parent.next_h2h_sub_plan_index) {
    parent.h2h_in_flight = true;
    return true;
  }
  parent.pending_h2h_sessions.insert(
      std::upper_bound(parent.pending_h2h_sessions.begin(),
                       parent.pending_h2h_sessions.end(), session,
                       [](const auto& a, const auto& b) {
                         return a->sub_plan_index() < b->sub_plan_index();
                       }),
      std::move(session));
  return false;
}

void KVCacheManagerWithTransfer::OnSendSessionSettled(
    uint64_t uuid, TransferSendSession& session) {
  if (IsShuttingDown()) return;
  std::vector<std::pair<uint64_t, uint64_t>> settled_plans;
  std::vector<std::shared_ptr<TransferSendSession>> sessions_to_release;
  std::shared_ptr<TransferSendSession> next_to_start;
  {
    absl::MutexLock lock(mu_);
    RetireSettledSendSessionLocked(uuid, &settled_plans, &sessions_to_release);
    if (auto it = parent_send_registrations_.find(session.base_uuid());
        it != parent_send_registrations_.end() && !it->second.failed &&
        session.sub_plan_index() == it->second.next_h2h_sub_plan_index) {
      ParentSendRegistration& p = it->second;
      p.h2h_in_flight = false;
      p.next_h2h_sub_plan_index = session.sub_plan_index() + 1;
      while (
          !p.pending_h2h_sessions.empty() &&
          (p.pending_h2h_sessions.front()->IsDraining() || IsShuttingDown())) {
        if (p.pending_h2h_sessions.front()->sub_plan_index() ==
            p.next_h2h_sub_plan_index) {
          ++p.next_h2h_sub_plan_index;
        }
        sessions_to_release.push_back(
            std::move(p.pending_h2h_sessions.front()));
        p.pending_h2h_sessions.pop_front();
      }
      if (!p.pending_h2h_sessions.empty() &&
          p.pending_h2h_sessions.front()->sub_plan_index() ==
              p.next_h2h_sub_plan_index) {
        next_to_start = std::move(p.pending_h2h_sessions.front());
        p.pending_h2h_sessions.pop_front();
        p.h2h_in_flight = true;
      }
    }
  }
  for (const auto& [plan_uuid, gen] : settled_plans) {
    UnregisterSettledPlan(plan_uuid, gen);
    base_->ReleasePoolStagingLeases(plan_uuid);
  }
  const absl::Status chain_status =
      session.GetStatus().ok() ? absl::CancelledError("Send chain aborted")
                               : session.GetStatus();
  for (const auto& s : sessions_to_release) {
    if (!s->IsDraining()) s->Finish(chain_status);
    s->DropH2hGateHold();
  }
  if (next_to_start != nullptr) next_to_start->StartGatedH2h();
}

void KVCacheManagerWithTransfer::CleanupParentSendRegistrationLocked(
    uint64_t base_uuid) {
  auto it = parent_send_registrations_.find(base_uuid);
  if (it == parent_send_registrations_.end()) return;
  for (size_t c = 0; c < it->second.total_blocks; ++c) {
    sub_uuid_to_parent_.erase(SubPlanUuid(base_uuid, static_cast<int32_t>(c)));
  }
  parent_send_registrations_.erase(it);
}

void KVCacheManagerWithTransfer::RetireSettledSendSessionLocked(
    uint64_t uuid, std::vector<std::pair<uint64_t, uint64_t>>* settled_plans,
    std::vector<std::shared_ptr<TransferSendSession>>* sessions_to_release) {
  auto it = send_sessions_.find(uuid);
  if (it == send_sessions_.end() || !it->second->Done()) return;

  std::shared_ptr<TransferSendSession> session = it->second;
  base_->ClearRemoteLayerAddrs(uuid);
  send_sessions_.erase(it);

  auto parent_it = parent_send_registrations_.find(session->base_uuid());
  if (parent_it == parent_send_registrations_.end()) {
    const bool failed = !session->GetStatus().ok();
    if (failed && settled_plans) settled_plans->emplace_back(uuid, 0);
    if (!done_sending_.contains(session->req_id()) &&
        !failed_recving_.contains(session->req_id())) {
      (failed ? failed_recving_ : done_sending_).insert(session->req_id());
      transfer_details_[session->req_id()] = session->Details();
    }
    return;
  }

  ParentSendRegistration& parent = parent_it->second;
  const bool has_remaining_siblings = std::any_of(
      send_sessions_.begin(), send_sessions_.end(), [&](const auto& kv) {
        return kv.second->base_uuid() == parent.base_uuid;
      });

  if (!session->GetStatus().ok()) {
    if (settled_plans) settled_plans->emplace_back(uuid, 0);
    if (!parent.failed) {
      parent.failed = true;
      failed_recving_.insert(parent.req_id);
      transfer_details_[parent.req_id] = session->Details();
    }
    if (sessions_to_release) {
      for (auto& s : parent.pending_h2h_sessions) {
        sessions_to_release->push_back(std::move(s));
      }
      for (auto& [_, s] : send_sessions_) {
        if (s->base_uuid() == parent.base_uuid && !s->Done()) {
          sessions_to_release->push_back(s);
        }
      }
    }
    parent.pending_h2h_sessions.clear();
  }
  if (parent.failed) {
    if (!has_remaining_siblings) {
      CleanupParentSendRegistrationLocked(parent.base_uuid);
    }
    return;
  }

  parent.settled_blocks += session->pulled_blocks();
  const bool single_pull_complete =
      uuid == parent.base_uuid && parent.sub_pulls_dispatched == 1 &&
      (window_blocks_ <= 0 ||
       session->pulled_blocks() < static_cast<size_t>(window_blocks_) ||
       parent.total_blocks <= static_cast<size_t>(window_blocks_));
  if ((parent.sub_pulls_dispatched == 0 && session->pulled_blocks() == 0 &&
       uuid == parent.base_uuid) ||
      single_pull_complete || parent.settled_blocks >= parent.total_blocks) {
    if (!done_sending_.contains(parent.req_id) &&
        !failed_recving_.contains(parent.req_id)) {
      done_sending_.insert(parent.req_id);
      transfer_details_[parent.req_id] = session->Details();
    }
    if (!has_remaining_siblings) {
      CleanupParentSendRegistrationLocked(parent.base_uuid);
    }
  }
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

absl::Status KVCacheManagerWithTransfer::OnBlockShardsReceived(
    const std::vector<int>& block_ids, absl::Span<const int> shard_ids,
    uint64_t uuid) {
  RAIDEN_TRACE_FN("KVTransfer::OnBlockShardsReceived", [&]() {
    return absl::StrCat("blocks=", block_ids.size(),
                        " shards=", shard_ids.size(), " uuid=", uuid);
  });
  VLOG(1) << "KVCacheManagerWithTransfer::OnBlockShardsReceived called. uuid: "
          << uuid << ", received blocks count: " << block_ids.size()
          << ", shards: " << shard_ids.size();

  std::shared_ptr<TransferReceiveSession> session;
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_sessions_.find(uuid);
    if (it == active_recv_sessions_.end()) {
      return absl::OkStatus();
    }
    session = it->second;
  }
  absl::Status status =
      session->OnBlockShardsReceived(*this, block_ids, shard_ids);
  MaybeUnregisterSettledRecv(uuid, *session);
  return status;
}

}  // namespace tpu_raiden

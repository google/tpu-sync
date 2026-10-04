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

#include "tpu_sync/weight_sync/manager/v3/controller_v3.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/async_control_server.h"
#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"
#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"
#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

std::string GetRoutableLocalIp() {
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd >= 0) {
    struct sockaddr_in remote{};
    remote.sin_family = AF_INET;
    remote.sin_port = htons(80);
    ::inet_pton(AF_INET, "10.255.255.255", &remote.sin_addr);
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&remote),
                  sizeof(remote)) == 0) {
      struct sockaddr_in local{};
      socklen_t len = sizeof(local);
      if (::getsockname(fd, reinterpret_cast<struct sockaddr*>(&local), &len) ==
          0) {
        char buf[INET_ADDRSTRLEN] = {0};
        if (::inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf)) !=
                nullptr &&
            std::strlen(buf) > 0 && std::string(buf) != "0.0.0.0" &&
            std::strncmp(buf, "127.", 4) != 0) {
          ::close(fd);
          return std::string(buf);
        }
      }
    }
    ::close(fd);
  }

  int fd6 = ::socket(AF_INET6, SOCK_DGRAM, 0);
  if (fd6 >= 0) {
    struct sockaddr_in6 remote6{};
    remote6.sin6_family = AF_INET6;
    remote6.sin6_port = htons(80);
    ::inet_pton(AF_INET6, "2001:4860:4860::8888", &remote6.sin6_addr);
    if (::connect(fd6, reinterpret_cast<struct sockaddr*>(&remote6),
                  sizeof(remote6)) == 0) {
      struct sockaddr_in6 local6{};
      socklen_t len6 = sizeof(local6);
      if (::getsockname(fd6, reinterpret_cast<struct sockaddr*>(&local6),
                        &len6) == 0) {
        char buf6[INET6_ADDRSTRLEN] = {0};
        if (::inet_ntop(AF_INET6, &local6.sin6_addr, buf6, sizeof(buf6)) !=
                nullptr &&
            std::strlen(buf6) > 0 && std::string(buf6) != "::" &&
            std::string(buf6) != "::1") {
          ::close(fd6);
          return absl::StrCat("[", buf6, "]");
        }
      }
    }
    ::close(fd6);
  }
  return "127.0.0.1";
}

}  // namespace

RaidenControllerV3::PlanOptions RaidenControllerV3::PlanOptions::Make(
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h, int32_t parallelism,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling) {
  PlanOptions out;
  out.dst_mem_type = dst_mem_type;
  out.skip_d2h = skip_d2h;
  out.parallelism = std::max<int32_t>(1, parallelism);
  for (const auto& [layer, skip] : skip_tiling) {
    if (skip) out.skip_tiling_layers.push_back(layer);
  }
  std::sort(out.skip_tiling_layers.begin(), out.skip_tiling_layers.end());
  return out;
}

RaidenControllerV3::RaidenControllerV3(const Options& options)
    : requested_port_(options.port),
      broadcast_host_ratio_(options.broadcast_host_ratio),
      num_bundle_groups_(std::max<int32_t>(1, options.num_bundle_groups)),
      request_registry_ttl_s_(options.request_registry_ttl_s),
      min_unobserved_ttl_s_(options.min_unobserved_ttl_s),
      enable_plan_cache_(options.enable_plan_cache),
      backend_type_(options.backend_type),
      custom_rpc_sender_(options.custom_rpc_sender),
      pull_phase_(options.pull_phase),
      requested_pull_server_port_(options.pull_server_port),
      pull_server_threads_(std::max<int32_t>(1, options.pull_server_threads)),
      pull_wire_codec_(options.pull_wire_codec),
      num_stripes_(std::max<int32_t>(0, options.num_stripes)),
      seed_replication_(std::max<int32_t>(1, options.seed_replication)),
      grant_batch_size_(std::max<int32_t>(1, options.grant_batch_size)),
      max_concurrent_uploads_per_source_(
          std::max<int32_t>(1, options.max_concurrent_uploads_per_source)),
      lease_timeout_ms_(std::max<int64_t>(1, options.lease_timeout_ms)),
      long_poll_timeout_ms_(std::max<int64_t>(1, options.long_poll_timeout_ms)),
      transfer_timeout_ms_(std::max<int64_t>(1, options.transfer_timeout_ms)) {
  ControlPipeConfig client_cfg;
  client_cfg.backend_type = ResolveControlPipeBackendType(backend_type_);
  client_cfg.enable_tcp_connection_pooling = true;
  client_ = CreateControlPipeClient(client_cfg);
}

RaidenControllerV3::~RaidenControllerV3() { StopServer(); }

absl::StatusOr<int> RaidenControllerV3::StartServer() {
  if (server_ != nullptr) {
    return server_->bound_port();
  }
  ControlPipeConfig cfg;
  cfg.backend_type = ResolveControlPipeBackendType(backend_type_);
  cfg.requested_port = requested_port_;

  server_ = CreateControlPipeServer(cfg);
  server_->dispatcher()
      .RegisterHandler<tpu_sync::rpc::ControllerRequest,
                       tpu_sync::rpc::ControllerResponse>(
          [this](const ControlContext& /*ctx*/,
                 const tpu_sync::rpc::ControllerRequest& req)
              -> absl::StatusOr<tpu_sync::rpc::ControllerResponse> {
            return HandleControllerRequest(req);
          },
          HandlerOptions<tpu_sync::rpc::ControllerRequest>()
              .WithMaxPayloadBytes(cfg.max_frame_bytes));

  server_->dispatcher().RegisterRawHandler(
      ::tpu_sync::rpc::ControlRequest::descriptor()->full_name(),
      [this](const ControlContext& /*ctx*/, absl::string_view req_bytes)
          -> absl::StatusOr<std::string> { return HandleRawFrame(req_bytes); },
      cfg.max_frame_bytes);

  ABSL_ASSIGN_OR_RETURN(const int port, server_->Start(requested_port_));
  if (requested_pull_server_port_ >= 0 && pull_server_ == nullptr) {
    std::unique_ptr<AsyncControlServer> pull_server = CreateEpollControlServer(
        {.num_threads = pull_server_threads_,
         .max_frame_bytes = cfg.max_frame_bytes},
        [this](const ControlContext& ctx,
               control_pipe::proto::ControlEnvelope request,
               AsyncControlReply reply) {
          HandlePullEnvelope(ctx, std::move(request), std::move(reply));
        });
    absl::StatusOr<int> pull_port =
        pull_server->Start(requested_pull_server_port_);
    if (!pull_port.ok()) {
      server_->Stop();
      server_.reset();
      return pull_port.status();
    }
    pull_server_ = std::move(pull_server);
  }
  return port;
}

void RaidenControllerV3::StopServer() {
  if (pull_server_ != nullptr) {
    pull_server_->Stop();
    pull_server_.reset();
  }
  if (server_ != nullptr) {
    server_->Stop();
    server_.reset();
  }
  transfer_threads_.AwaitAllDone();
}

int RaidenControllerV3::pull_server_port() const {
  return pull_server_ != nullptr ? pull_server_->bound_port() : 0;
}

std::optional<AsyncControlServer::Stats>
RaidenControllerV3::GetPullServerStats() const {
  if (pull_server_ == nullptr) return std::nullopt;
  return pull_server_->GetStats();
}

int RaidenControllerV3::port() const {
  return server_ != nullptr ? server_->bound_port() : requested_port_;
}

std::string RaidenControllerV3::controller_address() const {
  {
    absl::MutexLock lock(mu_);
    if (!controller_address_override_.empty()) {
      return controller_address_override_;
    }
  }
  const int p = port();
  if (p > 0) {
    return absl::StrCat(GetRoutableLocalIp(), ":", p);
  }
  return "";
}

void RaidenControllerV3::set_controller_address(absl::string_view address) {
  absl::MutexLock lock(mu_);
  controller_address_override_ = std::string(address);
}

void RaidenControllerV3::set_broadcast_host_ratio(double ratio) {
  absl::MutexLock lock(mu_);
  broadcast_host_ratio_ = ratio;
  schedule_cache_.clear();
  ++plan_inputs_generation_;
}

void RaidenControllerV3::set_num_bundle_groups(int32_t groups) {
  absl::MutexLock lock(mu_);
  num_bundle_groups_ = std::max<int32_t>(1, groups);
  schedule_cache_.clear();
  ++plan_inputs_generation_;
}

int32_t RaidenControllerV3::num_stripes() const {
  absl::MutexLock lock(mu_);
  return num_stripes_;
}

void RaidenControllerV3::set_num_stripes(int32_t num_stripes) {
  absl::MutexLock lock(mu_);
  num_stripes_ = std::max<int32_t>(0, num_stripes);
  schedule_cache_.clear();
  ++plan_inputs_generation_;
}

int32_t RaidenControllerV3::seed_replication() const {
  absl::MutexLock lock(mu_);
  return seed_replication_;
}

void RaidenControllerV3::set_seed_replication(int32_t replication) {
  absl::MutexLock lock(mu_);
  seed_replication_ = std::max<int32_t>(1, replication);
  schedule_cache_.clear();
  ++plan_inputs_generation_;
}

double RaidenControllerV3::request_registry_ttl_s() const {
  absl::MutexLock lock(mu_);
  return request_registry_ttl_s_;
}

void RaidenControllerV3::set_request_registry_ttl_s(double ttl_s) {
  absl::MutexLock lock(mu_);
  request_registry_ttl_s_ = ttl_s;
  EvictExpiredTransfersLocked(absl::Now());
}

// The pull options are execution options: they are read when an execution
// starts, so changing them never invalidates schedules or stored plans.
int32_t RaidenControllerV3::grant_batch_size() const {
  absl::MutexLock lock(mu_);
  return grant_batch_size_;
}

void RaidenControllerV3::set_grant_batch_size(int32_t grant_batch_size) {
  absl::MutexLock lock(mu_);
  grant_batch_size_ = std::max<int32_t>(1, grant_batch_size);
}

int32_t RaidenControllerV3::max_concurrent_uploads_per_source() const {
  absl::MutexLock lock(mu_);
  return max_concurrent_uploads_per_source_;
}

void RaidenControllerV3::set_max_concurrent_uploads_per_source(
    int32_t max_uploads) {
  absl::MutexLock lock(mu_);
  max_concurrent_uploads_per_source_ = std::max<int32_t>(1, max_uploads);
}

int64_t RaidenControllerV3::lease_timeout_ms() const {
  absl::MutexLock lock(mu_);
  return lease_timeout_ms_;
}

void RaidenControllerV3::set_lease_timeout_ms(int64_t timeout_ms) {
  absl::MutexLock lock(mu_);
  lease_timeout_ms_ = std::max<int64_t>(1, timeout_ms);
}

int64_t RaidenControllerV3::long_poll_timeout_ms() const {
  absl::MutexLock lock(mu_);
  return long_poll_timeout_ms_;
}

void RaidenControllerV3::set_long_poll_timeout_ms(int64_t timeout_ms) {
  absl::MutexLock lock(mu_);
  long_poll_timeout_ms_ = std::max<int64_t>(1, timeout_ms);
}

int64_t RaidenControllerV3::transfer_timeout_ms() const {
  absl::MutexLock lock(mu_);
  return transfer_timeout_ms_;
}

void RaidenControllerV3::set_transfer_timeout_ms(int64_t timeout_ms) {
  absl::MutexLock lock(mu_);
  // An execution option: cached schedules and stored plans stay valid, so
  // `plan_inputs_generation_` is not bumped.
  transfer_timeout_ms_ = std::max<int64_t>(1, timeout_ms);
}

void RaidenControllerV3::ClearPlanCache() {
  absl::MutexLock lock(mu_);
  schedule_cache_.clear();
}

size_t RaidenControllerV3::GetPlanCacheSize() const {
  absl::MutexLock lock(mu_);
  return schedule_cache_.size();
}

size_t RaidenControllerV3::GetTransferRecordCount() {
  absl::MutexLock lock(mu_);
  EvictExpiredTransfersLocked(absl::Now());
  return transfers_.size();
}

absl::flat_hash_map<std::string, uint64_t>
RaidenControllerV3::GetRetainedPlanUuids() {
  absl::MutexLock lock(mu_);
  EvictExpiredTransfersLocked(absl::Now());
  absl::flat_hash_map<std::string, uint64_t> out;
  for (const auto& [req_id, rec] : transfers_) {
    if (rec.plan.has_value()) out[req_id] = rec.plan->uuid;
  }
  return out;
}

void RaidenControllerV3::InvalidateUnitInScheduleCache(const RaidenId& unit) {
  // Stored plans embed the unit's concrete endpoints, so they are invalidated
  // as well (even ones whose logical schedule is unaffected).
  ++plan_inputs_generation_;
  for (auto it = schedule_cache_.begin(); it != schedule_cache_.end();) {
    const bool in_src =
        std::find(it->first.src_units.begin(), it->first.src_units.end(),
                  unit) != it->first.src_units.end();
    const bool in_dst =
        std::find(it->first.dst_units.begin(), it->first.dst_units.end(),
                  unit) != it->first.dst_units.end();
    if (in_src || in_dst) {
      auto erase_it = it++;
      schedule_cache_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

void RaidenControllerV3::EvictExpiredTransfersLocked(absl::Time now,
                                                     absl::string_view keep) {
  if (request_registry_ttl_s_ < 0.0) {
    return;
  }
  const absl::Duration ttl = absl::Seconds(request_registry_ttl_s_);
  const absl::Duration unobserved_ttl =
      absl::Seconds(std::max(request_registry_ttl_s_, min_unobserved_ttl_s_));
  for (auto it = transfers_.begin(); it != transfers_.end();) {
    const TransferSessionRecord& rec = it->second;
    bool expired = false;
    switch (rec.status) {
      case tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED:
      case tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED:
        // A result nobody has seen yet is kept (bounded) so that pollers and
        // late waiters still get it, even with a zero TTL.
        expired =
            rec.completed_at <= now &&
            (now - rec.completed_at) >= (rec.observed ? ttl : unobserved_ttl);
        break;
      case tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED:
        // A materialized plan that was never executed.
        expired =
            rec.planned_at <= now && (now - rec.planned_at) >= unobserved_ttl;
        break;
      default:
        // In-progress transfers are never evicted.
        break;
    }
    if (expired && it->first != keep) {
      auto erase_it = it++;
      transfers_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

absl::Status RaidenControllerV3::RegisterWorkUnit(
    const tpu_sync::rpc::RegisterWorkUnitRequest& req) {
  ABSL_ASSIGN_OR_RETURN(bool topology_changed,
                        entity_registry_.RegisterWorkUnit(req));
  if (topology_changed) {
    const RaidenId unit = RaidenIdFromProto(req.unit());
    absl::MutexLock lock(mu_);
    InvalidateUnitInScheduleCache(unit);
  }
  return absl::OkStatus();
}

absl::Status RaidenControllerV3::AttachHost(
    const RaidenId& unit, absl::string_view control_address,
    absl::Span<const std::string> host_shards) {
  ABSL_RETURN_IF_ERROR(
      entity_registry_.AttachHost(unit, control_address, host_shards));
  absl::MutexLock lock(mu_);
  InvalidateUnitInScheduleCache(unit);
  return absl::OkStatus();
}

absl::StatusOr<LogicalReshardSchedule>
RaidenControllerV3::GetOrComputeLogicalSchedule(
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling,
    bool use_cached_plan) {
  if (src_units.empty() || dst_units.empty()) {
    return absl::InvalidArgumentError(
        "src_units and dst_units must not be empty");
  }
  ABSL_ASSIGN_OR_RETURN(WorkUnitEntity src_ent,
                        entity_registry_.BuildCompositeEntity(src_units));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<WorkUnitEntity> dst_entities,
      entity_registry_.BuildDestinationReplicaEntities(dst_units));
  return ComputeScheduleForEntities(src_units, dst_units, src_ent, dst_entities,
                                    skip_tiling, use_cached_plan);
}

absl::StatusOr<LogicalReshardSchedule>
RaidenControllerV3::ComputeScheduleForEntities(
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    const WorkUnitEntity& src_ent,
    absl::Span<const WorkUnitEntity> dst_entities,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling,
    bool use_cached_plan) {
  if (dst_entities.empty()) {
    return absl::InvalidArgumentError("dst_units must not be empty");
  }
  const WorkUnitEntity& dst_rep0 = dst_entities[0];
  const int32_t num_dst_replicas = static_cast<int32_t>(dst_entities.size());

  std::vector<std::pair<int32_t, bool>> skip_sorted(skip_tiling.begin(),
                                                    skip_tiling.end());
  std::sort(skip_sorted.begin(), skip_sorted.end());

  double eff_ratio = 1.0;
  int32_t eff_bundle_groups = 8;
  SeedingOptions seeding;
  {
    absl::MutexLock lock(mu_);
    eff_ratio = broadcast_host_ratio_;
    eff_bundle_groups = num_bundle_groups_;
    seeding.num_stripes = num_stripes_;
    seeding.replication = seed_replication_;
  }

  ScheduleCacheKey cache_key;
  cache_key.src_units.assign(src_units.begin(), src_units.end());
  cache_key.dst_units.assign(dst_units.begin(), dst_units.end());
  cache_key.ratio_millis =
      static_cast<int64_t>(std::llround(eff_ratio * 1000.0));
  cache_key.num_bundle_groups = eff_bundle_groups;
  cache_key.num_stripes = seeding.num_stripes;
  cache_key.seed_replication = seeding.replication;
  cache_key.skip_tiling_sorted = std::move(skip_sorted);

  if (enable_plan_cache_ && use_cached_plan) {
    absl::MutexLock lock(mu_);
    auto it = schedule_cache_.find(cache_key);
    if (it != schedule_cache_.end()) {
      return it->second;
    }
  }

  std::vector<VariableSpec> src_vars = src_ent.variables;
  std::vector<VariableSpec> dst_vars = dst_rep0.variables;
  for (size_t i = 0; i < src_vars.size(); ++i) {
    auto it = skip_tiling.find(static_cast<int32_t>(i));
    if (it != skip_tiling.end() && it->second) {
      src_vars[i].skip_tiling = true;
      if (i < dst_vars.size()) dst_vars[i].skip_tiling = true;
    }
  }

  // Seeds are assigned by position in sorted unit order, so they stay put
  // when callers list the same units in a different order.
  seeding.replica_order.resize(num_dst_replicas);
  std::iota(seeding.replica_order.begin(), seeding.replica_order.end(), 0);
  std::sort(seeding.replica_order.begin(), seeding.replica_order.end(),
            [&](int32_t a, int32_t b) {
              const RaidenId& ua = dst_entities[a].unit;
              const RaidenId& ub = dst_entities[b].unit;
              return std::tie(ua.job_name, ua.job_replica_id, ua.data_name,
                              ua.data_replica_idx, a) <
                     std::tie(ub.job_name, ub.job_replica_id, ub.data_name,
                              ub.data_replica_idx, b);
            });

  ABSL_ASSIGN_OR_RETURN(
      LogicalReshardSchedule schedule,
      LogicalReshardPlanner::ComputeLogicalSchedule(
          src_vars, dst_vars, static_cast<int32_t>(src_ent.shards.size()),
          static_cast<int32_t>(dst_rep0.shards.size()), num_dst_replicas,
          eff_ratio, src_ent.NumHosts(), dst_rep0.NumHosts(), eff_bundle_groups,
          seeding));

  if (enable_plan_cache_) {
    absl::MutexLock lock(mu_);
    schedule_cache_[cache_key] = schedule;
  }
  return schedule;
}

absl::StatusOr<MaterializedTransferPlan>
RaidenControllerV3::BuildMaterializedPlan(
    absl::string_view req_id, uint64_t uuid,
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h, int32_t parallelism,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling,
    bool use_cached_plan) {
  const uint64_t eff_uuid =
      uuid > 0 ? uuid : next_uuid_.fetch_add(1, std::memory_order_relaxed);
  const std::string eff_req_id =
      !req_id.empty() ? std::string(req_id) : absl::StrCat("req_", eff_uuid);
  ABSL_ASSIGN_OR_RETURN(
      StoredPlan stored,
      MaterializeAndStorePlan(eff_req_id, eff_uuid, src_units, dst_units,
                              dst_mem_type, skip_d2h, parallelism, skip_tiling,
                              use_cached_plan, /*claimed=*/false));
  return std::move(stored.plan);
}

absl::StatusOr<RaidenControllerV3::StoredPlan>
RaidenControllerV3::MaterializeAndStorePlan(
    absl::string_view req_id, uint64_t uuid,
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h, int32_t parallelism,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling, bool use_cached_plan,
    bool claimed) {
  auto reject_if_in_progress = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = transfers_.find(req_id);
    if (it != transfers_.end() &&
        it->second.status ==
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Transfer ", req_id, " is in progress; cannot re-materialize it"));
    }
    return absl::OkStatus();
  };
  // Captured before any setting or registration is read, so a concurrent
  // change can only make the stored plan look stale (and get rebuilt).
  uint64_t inputs_generation = 0;
  {
    absl::MutexLock lock(mu_);
    if (!claimed) {
      ABSL_RETURN_IF_ERROR(reject_if_in_progress());
    }
    inputs_generation = plan_inputs_generation_;
  }
  if (src_units.empty() || dst_units.empty()) {
    return absl::InvalidArgumentError(
        "src_units and dst_units must not be empty");
  }

  // Resolve the entities once and use them for both the logical schedule and
  // the materialization.
  ABSL_ASSIGN_OR_RETURN(WorkUnitEntity src_ent,
                        entity_registry_.BuildCompositeEntity(src_units));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<WorkUnitEntity> dst_entities,
      entity_registry_.BuildDestinationReplicaEntities(dst_units));
  ABSL_ASSIGN_OR_RETURN(
      LogicalReshardSchedule schedule,
      ComputeScheduleForEntities(src_units, dst_units, src_ent, dst_entities,
                                 skip_tiling, use_cached_plan));

  plan_materialization_count_.fetch_add(1, std::memory_order_relaxed);
  ABSL_ASSIGN_OR_RETURN(MaterializedTransferPlan plan,
                        DynamicPullEngine::MaterializeTransferPlan(
                            req_id, uuid, src_ent, dst_entities, schedule,
                            dst_mem_type, skip_d2h, parallelism));

  absl::MutexLock lock(mu_);
  if (!claimed) {
    ABSL_RETURN_IF_ERROR(reject_if_in_progress());
  }
  const absl::Time now = absl::Now();
  EvictExpiredTransfersLocked(now, /*keep=*/req_id);
  TransferSessionRecord& rec = transfers_[req_id];
  if (!claimed) {
    rec.status = tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED;
    rec.result_status = absl::OkStatus();
    rec.completed_at = absl::InfiniteFuture();
    rec.observed = false;
  }
  rec.plan = plan;
  rec.src_units.assign(src_units.begin(), src_units.end());
  rec.dst_units.assign(dst_units.begin(), dst_units.end());
  rec.dst_entities = dst_entities;
  rec.plan_options =
      PlanOptions::Make(dst_mem_type, skip_d2h, parallelism, skip_tiling);
  rec.plan_inputs_generation = inputs_generation;
  rec.planned_at = now;
  rec.pull_service = nullptr;
  std::shared_ptr<TransferPullService> pull_service;
  if (claimed) {
    ABSL_ASSIGN_OR_RETURN(pull_service, NewPullServiceLocked(rec));
  }
  return StoredPlan{std::move(plan), std::move(dst_entities),
                    std::move(pull_service)};
}

absl::StatusOr<tpu_sync::rpc::ControlResponse>
RaidenControllerV3::SendWorkerRpc(absl::string_view endpoint,
                                  const tpu_sync::rpc::ControlRequest& req,
                                  absl::Time deadline) {
  const absl::Duration remaining = deadline - absl::Now();
  if (remaining <= absl::ZeroDuration()) {
    return absl::DeadlineExceededError(
        absl::StrCat("Transfer deadline passed before the RPC to ", endpoint));
  }
  if (custom_rpc_sender_) {
    return custom_rpc_sender_(endpoint, req);
  }
  return client_
      ->Call<tpu_sync::rpc::ControlRequest, tpu_sync::rpc::ControlResponse>(
          endpoint, req, std::min(kWorkerRpcTimeout, remaining));
}

bool RaidenControllerV3::TryClaimTransferLocked(absl::string_view req_id,
                                                absl::Time deadline) {
  TransferSessionRecord& rec = transfers_[req_id];
  if (rec.status ==
      tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS) {
    return false;
  }
  rec.status = tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS;
  rec.result_status = absl::OkStatus();
  rec.completed_at = absl::InfiniteFuture();
  rec.observed = false;
  rec.deadline = deadline;
  return true;
}

absl::StatusOr<absl::Time> RaidenControllerV3::ExecutionDeadlineLocked(
    std::optional<absl::Duration> transfer_timeout) const {
  if (transfer_timeout.has_value() &&
      *transfer_timeout <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Transfer timeout must be positive, got ",
                     absl::FormatDuration(*transfer_timeout)));
  }
  return absl::Now() +
         transfer_timeout.value_or(absl::Milliseconds(transfer_timeout_ms_));
}

void RaidenControllerV3::FinishTransfer(absl::string_view req_id,
                                        const absl::Status& status,
                                        bool observed) {
  absl::MutexLock lock(mu_);
  const absl::Time now = absl::Now();
  TransferSessionRecord& rec = transfers_[req_id];
  rec.result_status = status;
  rec.completed_at = now;
  rec.observed = observed;
  rec.status = status.ok()
                   ? tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED
                   : tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED;
  // Never evict the record that just finished here: its result must stay
  // observable by `WaitForTransfer`/`GetTransferStatus` until the next pass.
  EvictExpiredTransfersLocked(now, /*keep=*/req_id);
}

PullSchedulerOptions RaidenControllerV3::PullSchedulerOptionsLocked() const {
  PullSchedulerOptions options;
  options.grant_batch_size = grant_batch_size_;
  options.max_concurrent_uploads_per_source =
      max_concurrent_uploads_per_source_;
  options.lease_timeout = absl::Milliseconds(lease_timeout_ms_);
  options.long_poll_timeout = absl::Milliseconds(long_poll_timeout_ms_);
  return options;
}

absl::StatusOr<std::shared_ptr<TransferPullService>>
RaidenControllerV3::NewPullServiceLocked(TransferSessionRecord& rec) {
  if (!rec.plan.has_value()) {
    return absl::FailedPreconditionError("Transfer record holds no plan");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<TransferPullService> service,
      TransferPullService::Create(*rec.plan, rec.dst_entities,
                                  PullSchedulerOptionsLocked()));
  // The previous execution's service (if any) answers its parked requests
  // with `kAborted` when it is destroyed; its callbacks never take `mu_`.
  rec.pull_service = std::move(service);
  return rec.pull_service;
}

absl::StatusOr<std::shared_ptr<TransferPullService>>
RaidenControllerV3::FindPullService(absl::string_view req_id) const {
  absl::MutexLock lock(mu_);
  auto it = transfers_.find(req_id);
  if (it == transfers_.end() || it->second.pull_service == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("No pull service for transfer ", req_id));
  }
  return it->second.pull_service;
}

std::shared_ptr<TransferPullService> RaidenControllerV3::GetPullServiceForTest(
    absl::string_view req_id) const {
  absl::StatusOr<std::shared_ptr<TransferPullService>> service =
      FindPullService(req_id);
  return service.ok() ? *std::move(service) : nullptr;
}

absl::StatusOr<std::vector<PullShardStats>> RaidenControllerV3::GetPullStats(
    absl::string_view req_id) const {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<TransferPullService> service,
                        FindPullService(req_id));
  return service->scheduler().GetStats();
}

void RaidenControllerV3::HandlePullRequest(PullServiceRequest request,
                                           PullServiceCallback done) {
  absl::StatusOr<std::shared_ptr<TransferPullService>> service =
      FindPullService(request.req_id);
  if (!service.ok()) {
    std::move(done)(PullServiceReply{.kind = PullReplyKind::kAborted,
                                     .status = service.status()});
    return;
  }
  // The service may be dropped (record evicted or re-executed) while the
  // request is parked: its scheduler then answers it before it goes away.
  (*service)->Handle(std::move(request), std::move(done));
}

void RaidenControllerV3::AcquirePulls(PullServiceRequest request,
                                      PullServiceCallback done) {
  if (!request.completed.empty() || !request.failed.empty()) {
    std::move(done)(PullServiceReply{
        .kind = PullReplyKind::kAborted,
        .status = absl::InvalidArgumentError(
            "AcquirePulls carries no reports; use ReportPulls")});
    return;
  }
  HandlePullRequest(std::move(request), std::move(done));
}

void RaidenControllerV3::ReportPulls(PullServiceRequest request,
                                     PullServiceCallback done) {
  HandlePullRequest(std::move(request), std::move(done));
}

void RaidenControllerV3::HandlePullEnvelope(
    const ControlContext& /*ctx*/, control_pipe::proto::ControlEnvelope request,
    AsyncControlReply reply) {
  control_pipe::proto::ControlResponseEnvelope response;
  if (pull_wire_codec_ != nullptr &&
      request.message_type() == pull_wire_codec_->message_type()) {
    absl::StatusOr<PullServiceRequest> decoded =
        pull_wire_codec_->DecodeRequest(request.payload());
    if (!decoded.ok()) {
      response.set_status_code(static_cast<int32_t>(decoded.status().code()));
      response.set_error_message(std::string(decoded.status().message()));
      std::move(reply)(std::move(response));
      return;
    }
    HandlePullRequest(*std::move(decoded),
                      [codec = pull_wire_codec_, reply = std::move(reply)](
                          PullServiceReply pull_reply) mutable {
                        control_pipe::proto::ControlResponseEnvelope out;
                        out.set_payload(codec->EncodeReply(pull_reply));
                        std::move(reply)(std::move(out));
                      });
    return;
  }
  // TODO(justinlu): Add the pull RPCs to `raiden_service.proto` and decode
  // them here (a `PullWireCodec` for `tpu_sync.rpc.ControlRequest`), then
  // start the pull server by default:
  //   COMMAND_ACQUIRE_PULLS: AcquirePullsRequest {req_id, uuid, unit,
  //     host_idx, max_grants, long_poll_ms, seeded, lost_data}
  //   COMMAND_REPORT_PULLS: ReportPullsRequest {req_id, uuid, unit, host_idx,
  //     completed: [lease_id], failed: [{lease_id, reason}], want_more
  //     (max_grants), long_poll_ms, seeded, lost_data}
  //   Both answered by PullsResponse {oneof {grants: [{bundle_index,
  //     source_unit, source_data_endpoint, lease_id, expiry_ms}], done,
  //     abort: status}}, mapped to `AcquirePulls` / `ReportPulls`.
  response.set_status_code(
      static_cast<int32_t>(absl::StatusCode::kUnimplemented));
  response.set_error_message(
      absl::StrCat("UNIMPLEMENTED: the pull server does not serve ",
                   request.message_type()));
  std::move(reply)(std::move(response));
}

absl::Status RaidenControllerV3::RunPlan(
    const MaterializedTransferPlan& plan,
    absl::Span<const WorkUnitEntity> dst_entities,
    TransferPullService* pull_service,
    const DynamicPullEngine::RpcSenderFn& send_rpc, absl::Time deadline) {
  absl::Status status;
  if (pull_phase_ == PullPhaseKind::kScheduled) {
    ScheduledPullPhase phase(pull_service);
    status = DynamicPullEngine::ExecuteTransfer(plan, dst_entities, phase,
                                                send_rpc, deadline);
  } else {
    ControllerDrivenPullPhase phase(pull_service);
    status = DynamicPullEngine::ExecuteTransfer(plan, dst_entities, phase,
                                                send_rpc, deadline);
  }
  // E.g. an RPC cut short by the deadline.
  if (!status.ok() && !absl::IsDeadlineExceeded(status) &&
      absl::Now() >= deadline) {
    return absl::DeadlineExceededError(absl::StrCat(
        "Transfer ", plan.req_id, " passed its deadline: ", status.message()));
  }
  return status;
}

absl::Status RaidenControllerV3::ExecuteMaterializedTransferSync(
    absl::string_view req_id, uint64_t expected_uuid,
    const DynamicPullEngine::RpcSenderFn& rpc_sender_override,
    std::optional<absl::Duration> transfer_timeout) {
  if (req_id.empty()) {
    return absl::InvalidArgumentError("req_id must not be empty");
  }
  const std::string key(req_id);
  MaterializedTransferPlan plan;
  std::vector<WorkUnitEntity> dst_entities;
  std::shared_ptr<TransferPullService> pull_service;
  absl::Time deadline;
  {
    absl::MutexLock lock(mu_);
    ABSL_ASSIGN_OR_RETURN(deadline, ExecutionDeadlineLocked(transfer_timeout));
    EvictExpiredTransfersLocked(absl::Now());
    auto it = transfers_.find(key);
    if (it == transfers_.end() || !it->second.plan.has_value()) {
      return absl::NotFoundError(
          absl::StrCat("No materialized plan found for req_id: ", req_id));
    }
    if (expected_uuid != 0 && it->second.plan->uuid != expected_uuid) {
      return absl::FailedPreconditionError(
          absl::StrCat("Plan ", req_id, " (uuid=", expected_uuid,
                       ") is no longer retained: it was replaced by uuid ",
                       it->second.plan->uuid));
    }
    TransferSessionRecord& rec = it->second;
    if (rec.status ==
        tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS) {
      return absl::FailedPreconditionError(
          absl::StrCat("Transfer ", req_id, " is already in progress"));
    }
    ABSL_ASSIGN_OR_RETURN(pull_service, NewPullServiceLocked(rec));
    TryClaimTransferLocked(key, deadline);
    plan = *rec.plan;
    dst_entities = rec.dst_entities;
  }

  DynamicPullEngine::RpcSenderFn eff_sender =
      rpc_sender_override != nullptr
          ? rpc_sender_override
          : [this, deadline](absl::string_view ep,
                             const tpu_sync::rpc::ControlRequest& r) {
              return SendWorkerRpc(ep, r, deadline);
            };
  absl::Status exec_status =
      RunPlan(plan, dst_entities, pull_service.get(), eff_sender, deadline);
  FinishTransfer(key, exec_status, /*observed=*/true);
  return exec_status;
}

absl::Status RaidenControllerV3::PlanAndRunClaimed(
    absl::string_view req_id, uint64_t uuid, bool uuid_was_given,
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h, int32_t parallelism,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling,
    absl::Time deadline) {
  std::optional<StoredPlan> stored;
  {
    const PlanOptions requested_options =
        PlanOptions::Make(dst_mem_type, skip_d2h, parallelism, skip_tiling);
    absl::MutexLock lock(mu_);
    auto it = transfers_.find(req_id);
    if (it != transfers_.end() && it->second.plan.has_value()) {
      TransferSessionRecord& rec = it->second;
      const bool same_units =
          std::equal(rec.src_units.begin(), rec.src_units.end(),
                     src_units.begin(), src_units.end()) &&
          std::equal(rec.dst_units.begin(), rec.dst_units.end(),
                     dst_units.begin(), dst_units.end());
      // A plan built for different per-transfer options carries different
      // worker commands (memory type, D2H, parallelism, tiling) and must not
      // be reused; neither may one built before a controller setting or a
      // registration (e.g. a worker endpoint) changed. The transfer timeout
      // is not part of the plan and never prevents reuse.
      if (same_units && rec.plan_options == requested_options &&
          rec.plan_inputs_generation == plan_inputs_generation_ &&
          (!uuid_was_given || rec.plan->uuid == uuid)) {
        ABSL_ASSIGN_OR_RETURN(std::shared_ptr<TransferPullService> service,
                              NewPullServiceLocked(rec));
        stored = StoredPlan{*rec.plan, rec.dst_entities, std::move(service)};
      }
    }
  }
  if (!stored.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        stored, MaterializeAndStorePlan(
                    req_id, uuid, src_units, dst_units, dst_mem_type, skip_d2h,
                    parallelism, skip_tiling,
                    /*use_cached_plan=*/true, /*claimed=*/true));
  }
  return RunPlan(
      stored->plan, stored->dst_entities, stored->pull_service.get(),
      [this, deadline](absl::string_view ep,
                       const tpu_sync::rpc::ControlRequest& r) {
        return SendWorkerRpc(ep, r, deadline);
      },
      deadline);
}

absl::Status RaidenControllerV3::ExecuteTransferSync(
    absl::string_view req_id, uint64_t uuid,
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h, int32_t parallelism,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling,
    std::optional<absl::Duration> transfer_timeout) {
  const uint64_t eff_uuid =
      uuid > 0 ? uuid : next_uuid_.fetch_add(1, std::memory_order_relaxed);
  const std::string eff_req_id =
      !req_id.empty() ? std::string(req_id) : absl::StrCat("req_", eff_uuid);
  absl::Time deadline;
  {
    absl::MutexLock lock(mu_);
    ABSL_ASSIGN_OR_RETURN(deadline, ExecutionDeadlineLocked(transfer_timeout));
    EvictExpiredTransfersLocked(absl::Now());
    if (!TryClaimTransferLocked(eff_req_id, deadline)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Transfer ", eff_req_id, " is already in progress"));
    }
  }
  absl::Status status = PlanAndRunClaimed(
      eff_req_id, eff_uuid, /*uuid_was_given=*/uuid > 0, src_units, dst_units,
      dst_mem_type, skip_d2h, parallelism, skip_tiling, deadline);
  FinishTransfer(eff_req_id, status, /*observed=*/true);
  return status;
}

absl::Status RaidenControllerV3::StartTransferAsync(
    absl::string_view req_id, uint64_t uuid,
    absl::Span<const RaidenId> src_units, absl::Span<const RaidenId> dst_units,
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h, int32_t parallelism,
    const absl::flat_hash_map<int32_t, bool>& skip_tiling,
    std::optional<absl::Duration> transfer_timeout) {
  const uint64_t eff_uuid =
      uuid > 0 ? uuid : next_uuid_.fetch_add(1, std::memory_order_relaxed);
  const std::string eff_req_id =
      !req_id.empty() ? std::string(req_id) : absl::StrCat("req_", eff_uuid);
  absl::Time deadline;
  {
    absl::MutexLock lock(mu_);
    ABSL_ASSIGN_OR_RETURN(deadline, ExecutionDeadlineLocked(transfer_timeout));
    EvictExpiredTransfersLocked(absl::Now());
    if (!TryClaimTransferLocked(eff_req_id, deadline)) {
      return absl::OkStatus();
    }
  }

  std::vector<RaidenId> src_vec(src_units.begin(), src_units.end());
  std::vector<RaidenId> dst_vec(dst_units.begin(), dst_units.end());
  transfer_threads_.Spawn(
      [this, eff_req_id, eff_uuid, uuid_was_given = uuid > 0,
       src_vec = std::move(src_vec), dst_vec = std::move(dst_vec), dst_mem_type,
       skip_d2h, parallelism, skip_tiling, deadline]() {
        absl::Status s = PlanAndRunClaimed(
            eff_req_id, eff_uuid, uuid_was_given, src_vec, dst_vec,
            dst_mem_type, skip_d2h, parallelism, skip_tiling, deadline);
        if (!s.ok()) {
          LOG(ERROR) << "RaidenControllerV3 async transfer " << eff_req_id
                     << " failed: " << s;
        }
        // Always records a terminal state, including failures before execution
        // (e.g. unknown units), so pollers never hang on IN_PROGRESS. The
        // result stays retained until a poller or waiter has observed it.
        FinishTransfer(eff_req_id, s, /*observed=*/false);
      });
  return absl::OkStatus();
}

absl::Status RaidenControllerV3::WaitForTransfer(
    absl::string_view req_id, std::optional<absl::Duration> timeout) {
  const std::string key(req_id);
  bool seen = false;
  std::optional<absl::Status> result;
  // The condition runs under `mu_` (possibly on the finishing thread) right
  // after the record turns terminal, so the result is captured before any
  // later eviction can drop the record.
  auto is_done = [this, &key, &seen,
                  &result]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    auto it = transfers_.find(key);
    if (it == transfers_.end()) return true;
    seen = true;
    const auto status = it->second.status;
    if (status == tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED ||
        status == tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED) {
      result = it->second.result_status;
      return true;
    }
    return false;
  };
  absl::MutexLock lock(mu_);
  if (timeout.has_value()) {
    if (!mu_.AwaitWithTimeout(absl::Condition(&is_done), *timeout)) {
      return absl::DeadlineExceededError(
          absl::StrCat("Timed out waiting for transfer ", req_id));
    }
  } else {
    // Follows the deadline of the transfer, which is set anew whenever an
    // execution is claimed while waiting.
    constexpr auto kInProgress =
        tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS;
    absl::Time followed_deadline = absl::InfinitePast();
    auto done_or_new_deadline = [this, &key, &is_done, &followed_deadline]()
                                    ABSL_SHARED_LOCKS_REQUIRED(mu_) {
                                      if (is_done()) return true;
                                      // `is_done()` returned false, so the
                                      // record exists.
                                      const TransferSessionRecord& rec =
                                          transfers_.find(key)->second;
                                      return rec.status == kInProgress &&
                                             rec.deadline != followed_deadline;
                                    };
    while (!is_done()) {
      const TransferSessionRecord& rec = transfers_.find(key)->second;
      absl::Time wait_until;
      if (rec.status == kInProgress) {
        followed_deadline = rec.deadline;
        wait_until = rec.deadline + kWaitGraceAfterDeadline;
      } else {
        // Not started yet: as long as an execution claimed now may run. A
        // claim sets a deadline, which wakes the loop to follow it.
        followed_deadline = absl::InfinitePast();
        wait_until = absl::Now() + absl::Milliseconds(transfer_timeout_ms_) +
                     kWaitGraceAfterDeadline;
      }
      if (!mu_.AwaitWithDeadline(absl::Condition(&done_or_new_deadline),
                                 wait_until)) {
        return absl::DeadlineExceededError(absl::StrCat(
            "Timed out waiting for transfer ", req_id, " past its deadline"));
      }
    }
  }
  if (result.has_value()) {
    auto it = transfers_.find(key);
    if (it != transfers_.end()) it->second.observed = true;
    return *result;
  }
  if (!seen) {
    return absl::NotFoundError(absl::StrCat("Transfer not found: ", req_id));
  }
  return absl::NotFoundError(absl::StrCat(
      "Transfer ", req_id, " record was evicted before its result was seen"));
}

tpu_sync::rpc::GetTransferStatusResponse::Status
RaidenControllerV3::GetTransferStatus(absl::string_view req_id) {
  absl::MutexLock lock(mu_);
  EvictExpiredTransfersLocked(absl::Now());
  auto it = transfers_.find(req_id);
  if (it == transfers_.end()) {
    return tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED;
  }
  TransferSessionRecord& rec = it->second;
  if (rec.status ==
          tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED ||
      rec.status == tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED) {
    rec.observed = true;
  }
  return rec.status;
}

tpu_sync::rpc::ControlResponse RaidenControllerV3::HandleControlRequest(
    const tpu_sync::rpc::ControlRequest& req) {
  tpu_sync::rpc::ControlResponse resp;
  resp.set_success(false);

  switch (req.command()) {
    case tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER: {
      const auto& st = req.start_transfer_request();
      if (st.src_units_size() == 0 || st.dst_units_size() == 0) {
        resp.set_message("src_units and dst_units must not be empty");
        break;
      }
      std::vector<RaidenId> src_us;
      src_us.reserve(st.src_units_size());
      for (const auto& su : st.src_units()) {
        src_us.push_back(RaidenIdFromProto(su));
      }
      std::vector<RaidenId> dst_us;
      dst_us.reserve(st.dst_units_size());
      for (const auto& du : st.dst_units()) {
        dst_us.push_back(RaidenIdFromProto(du));
      }
      const tpu_sync::rpc::MemoryType eff_mem_type =
          st.dst_mem_type() == tpu_sync::rpc::MEMORY_TYPE_UNSPECIFIED
              ? tpu_sync::rpc::MEMORY_TYPE_DRAM
              : st.dst_mem_type();
      absl::flat_hash_map<int32_t, bool> skip_tiling;
      for (const auto& [k, v] : st.skip_tiling()) {
        skip_tiling[k] = v;
      }
      absl::Status s = StartTransferAsync(
          st.req_id(), static_cast<uint64_t>(std::max<int64_t>(0, st.uuid())),
          src_us, dst_us, eff_mem_type, st.skip_d2h(),
          std::max<int32_t>(1, st.parallelism()), skip_tiling);
      if (!s.ok()) {
        resp.set_message(std::string(s.message()));
        break;
      }
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT: {
      absl::Status s = RegisterWorkUnit(req.register_work_unit_request());
      if (!s.ok()) {
        resp.set_message(std::string(s.message()));
        break;
      }
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA: {
      for (auto& meta : entity_registry_.GetAllMetadata()) {
        *resp.mutable_get_metadata_response()->add_metadata() = std::move(meta);
      }
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN: {
      absl::flat_hash_set<std::string> endpoints;
      for (const RaidenId& u : entity_registry_.GetRegisteredUnits()) {
        auto ent = entity_registry_.GetEntity(u);
        if (ent.ok()) {
          for (const std::string& ep : ent->control_endpoints) {
            if (!ep.empty()) endpoints.insert(ep);
          }
        }
      }
      tpu_sync::rpc::ControlRequest shutdown_req;
      shutdown_req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN);
      for (const std::string& ep : endpoints) {
        SendWorkerRpc(ep, shutdown_req).status().IgnoreError();
      }
      resp.set_success(true);
      break;
    }
    default:
      // The pull RPCs go to the pull server (`HandlePullEnvelope`).
      resp.set_message(
          "UNIMPLEMENTED: unsupported ControlRequest command in ControllerV3");
      break;
  }
  return resp;
}

tpu_sync::rpc::ControllerResponse RaidenControllerV3::HandleControllerRequest(
    const tpu_sync::rpc::ControllerRequest& req) {
  tpu_sync::rpc::ControllerResponse resp;
  resp.set_success(false);

  switch (req.command()) {
    case tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER: {
      const auto& coord = req.coordinate_transfer_request();
      if (coord.src_units_size() == 0 || coord.dst_units_size() == 0) {
        resp.set_message("src_units and dst_units must not be empty");
        break;
      }
      std::vector<RaidenId> src_us;
      src_us.reserve(coord.src_units_size());
      for (const auto& su : coord.src_units()) {
        src_us.push_back(RaidenIdFromProto(su));
      }
      std::vector<RaidenId> dst_us;
      dst_us.reserve(coord.dst_units_size());
      for (const auto& du : coord.dst_units()) {
        dst_us.push_back(RaidenIdFromProto(du));
      }
      const tpu_sync::rpc::MemoryType eff_mem_type =
          coord.dst_mem_type() == tpu_sync::rpc::MEMORY_TYPE_UNSPECIFIED
              ? tpu_sync::rpc::MEMORY_TYPE_DRAM
              : coord.dst_mem_type();
      absl::Status s = StartTransferAsync(
          coord.req_id(),
          static_cast<uint64_t>(std::max<int64_t>(0, coord.uuid())), src_us,
          dst_us, eff_mem_type);
      if (!s.ok()) {
        resp.set_message(std::string(s.message()));
        break;
      }
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS: {
      const auto& st_req = req.get_transfer_status_request();
      resp.mutable_get_transfer_status_response()->set_status(
          GetTransferStatus(st_req.req_id()));
      resp.set_success(true);
      break;
    }
    default:
      resp.set_message("Unsupported ControllerRequest command in ControllerV3");
      break;
  }
  return resp;
}

std::string RaidenControllerV3::HandleRawFrame(
    absl::string_view request_bytes) {
  tpu_sync::rpc::ControllerRequest ctrl_req;
  if (ctrl_req.ParseFromString(request_bytes)) {
    const bool is_controller_cmd =
        (ctrl_req.command() ==
             tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER &&
         ctrl_req.has_coordinate_transfer_request()) ||
        (ctrl_req.command() ==
             tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS &&
         ctrl_req.has_get_transfer_status_request());
    if (is_controller_cmd) {
      return HandleControllerRequest(ctrl_req).SerializeAsString();
    }
  }

  tpu_sync::rpc::ControlRequest raiden_req;
  if (!raiden_req.ParseFromString(request_bytes)) {
    tpu_sync::rpc::ControlResponse err;
    err.set_success(false);
    err.set_message("Failed to parse ControlRequest");
    return err.SerializeAsString();
  }
  return HandleControlRequest(raiden_req).SerializeAsString();
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

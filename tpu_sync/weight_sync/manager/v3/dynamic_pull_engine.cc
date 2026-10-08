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

#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

// Runs tasks on a fixed number of threads.
class BoundedExecutor {
 public:
  explicit BoundedExecutor(int32_t threads) {
    for (int32_t i = 0; i < std::max(1, threads); ++i) {
      threads_.emplace_back([this] { Work(); });
    }
  }

  ~BoundedExecutor() { Shutdown(/*drain=*/true); }

  // Runs |task| on a pool thread. Dropped after `Shutdown()`.
  void Schedule(absl::AnyInvocable<void() &&> task) {
    absl::MutexLock lock(mu_);
    if (stopping_) return;
    tasks_.push_back(std::move(task));
  }

  // Stops accepting tasks, runs (|drain|) or drops the queued ones, and joins
  // the threads once the running ones finished.
  void Shutdown(bool drain) {
    {
      absl::MutexLock lock(mu_);
      if (stopping_ && threads_.empty()) return;
      stopping_ = true;
      if (!drain) tasks_.clear();
    }
    for (std::thread& t : threads_) t.join();
    threads_.clear();
  }

 private:
  void Work() {
    while (true) {
      absl::AnyInvocable<void() &&> task;
      {
        absl::MutexLock lock(mu_);
        auto ready = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
          return stopping_ || !tasks_.empty();
        };
        mu_.Await(absl::Condition(&ready));
        if (tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      std::move(task)();
    }
  }

  absl::Mutex mu_;
  std::deque<absl::AnyInvocable<void() &&>> tasks_ ABSL_GUARDED_BY(mu_);
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<std::thread> threads_;
};

// Sends every command in |commands| as a `COMMAND_START_TRANSFER` on at most
// |max_parallel| threads and returns the first error, after all of them have
// finished.
absl::Status DispatchStartTransfers(
    const std::vector<HostTransferCommand>& commands,
    const ControlRpcSender& send_rpc, int32_t max_parallel = 64);

absl::Status SendStartTransfer(const HostTransferCommand& cmd,
                               const ControlRpcSender& send_rpc) {
  tpu_sync::rpc::ControlRequest ctrl_req;
  ctrl_req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER);
  *ctrl_req.mutable_start_transfer_request() = cmd.request;
  ABSL_ASSIGN_OR_RETURN(tpu_sync::rpc::ControlResponse resp,
                        send_rpc(cmd.control_endpoint, ctrl_req));
  if (!resp.success()) {
    return absl::InternalError(absl::StrCat("Worker StartTransfer failed on ",
                                            cmd.control_endpoint, ": ",
                                            resp.message()));
  }
  return absl::OkStatus();
}

absl::Status DispatchStartTransfers(
    const std::vector<HostTransferCommand>& commands,
    const ControlRpcSender& send_rpc, int32_t max_parallel) {
  if (commands.empty()) return absl::OkStatus();
  if (commands.size() == 1) return SendStartTransfer(commands[0], send_rpc);
  absl::Mutex mu;
  absl::Status first_error;
  {
    BoundedExecutor executor(
        std::min<int32_t>(max_parallel, static_cast<int32_t>(commands.size())));
    for (const HostTransferCommand& cmd : commands) {
      executor.Schedule([&send_rpc, &cmd, &mu, &first_error] {
        absl::Status status = SendStartTransfer(cmd, send_rpc);
        absl::MutexLock lock(mu);
        if (!status.ok() && first_error.ok()) first_error = std::move(status);
      });
    }
  }
  return first_error;
}

RaidenId HostUnit(const WorkUnitEntity& ent, const HostAttachment& host) {
  return !host.unit.job_name.empty() ? host.unit : ent.unit;
}

void SetSkipTiling(const std::vector<bool>& skip_tiling_by_layer,
                   tpu_sync::rpc::StartTransferRequest& req) {
  for (size_t l = 0; l < skip_tiling_by_layer.size(); ++l) {
    (*req.mutable_skip_tiling())[static_cast<int32_t>(l)] =
        skip_tiling_by_layer[l];
  }
}

// Sorted layer indices of |bundle_ids|.
absl::StatusOr<std::vector<int32_t>> BundleLayers(
    absl::Span<const VariableBundleSpec> bundles,
    absl::Span<const int32_t> bundle_ids) {
  std::vector<int32_t> layers;
  for (int32_t b : bundle_ids) {
    if (b < 0 || b >= static_cast<int32_t>(bundles.size())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Bundle id ", b, " out of range for ", bundles.size(), " bundles"));
    }
    layers.insert(layers.end(), bundles[b].layer_indices.begin(),
                  bundles[b].layer_indices.end());
  }
  std::sort(layers.begin(), layers.end());
  return layers;
}

// Sets the per-layer chunk counts a sampler host expects: Trainer chunk
// counts (`expected_by_shard[global_shard][layer]`) for its seeded layers and
// one whole-layer chunk per owned shard for every pulled layer.
void SetExpectedChunkCounts(
    const HostAttachment& host, int32_t num_layers,
    absl::Span<const int32_t> seeded_layers,
    absl::Span<const std::vector<int32_t>> expected_by_shard,
    tpu_sync::rpc::StartTransferRequest& req) {
  const int32_t shards_on_host =
      static_cast<int32_t>(host.owned_global_shard_indices.size());
  std::vector<bool> seeded(num_layers, false);
  for (int32_t l : seeded_layers) {
    if (l >= 0 && l < num_layers) seeded[l] = true;
  }
  int64_t host_total_chunks = 0;
  auto* layer_counts = req.mutable_expected_layer_chunk_counts();
  layer_counts->clear();
  for (int32_t l = 0; l < num_layers; ++l) {
    int32_t layer_chunks = 0;
    if (seeded[l]) {
      for (int32_t g_shard : host.owned_global_shard_indices) {
        if (g_shard >= 0 &&
            g_shard < static_cast<int32_t>(expected_by_shard.size()) &&
            l < static_cast<int32_t>(expected_by_shard[g_shard].size())) {
          layer_chunks += expected_by_shard[g_shard][l];
        }
      }
    } else {
      layer_chunks = shards_on_host;
    }
    (*layer_counts)[l] = layer_chunks;
    host_total_chunks += layer_chunks;
  }
  req.set_expected_block_count(host_total_chunks);
}

// Replicas the Trainer seeds, in stripe order.
std::vector<int32_t> AllSeeds(const SeedLayout& layout) {
  std::vector<int32_t> seeds;
  for (const std::vector<int32_t>& stripe_seeds : layout.stripe_seeds) {
    seeds.insert(seeds.end(), stripe_seeds.begin(), stripe_seeds.end());
  }
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  return seeds;
}

// The error of transfer |req_id| once its deadline passed; |step| names what
// the transfer was doing.
absl::Status TransferDeadlineError(absl::string_view req_id,
                                   absl::string_view step) {
  return absl::DeadlineExceededError(
      absl::StrCat("Transfer ", req_id, " passed its deadline ", step));
}

// Fails with `TransferDeadlineError` if |deadline| passed.
absl::Status CheckTransferDeadline(absl::string_view req_id,
                                   absl::Time deadline,
                                   absl::string_view step) {
  if (absl::Now() < deadline) return absl::OkStatus();
  return TransferDeadlineError(req_id, step);
}

// Waits until |scheduler| completes, fails or |deadline| passes (which then
// fails the transfer).
absl::Status AwaitPullPhase(PullScheduler& scheduler, absl::string_view req_id,
                            absl::Time deadline) {
  absl::Status status = scheduler.WaitForCompletion(deadline);
  if (status.ok()) return status;
  if (absl::IsDeadlineExceeded(status) && !scheduler.complete() &&
      scheduler.status().ok()) {
    status = TransferDeadlineError(req_id, "waiting for the pull phase");
  }
  // Answers the requests of the samplers that are still asking.
  scheduler.Abort(status);
  return scheduler.complete() ? absl::OkStatus() : scheduler.status();
}

// Acts as the client of a `PullScheduler` for every sampler host of a
// transfer: each grant becomes a push from the source host, reported back
// when it finished. See `ControllerDrivenPullPhase`.
class PushDriver {
 public:
  PushDriver(const MaterializedTransferPlan& plan, TransferPullService& service,
             const ControlRpcSender& send_rpc, int32_t max_concurrent_pushes)
      : plan_(plan),
        service_(service),
        send_rpc_(send_rpc),
        grant_batch_(EffectiveGrantBatchSize(service.scheduler().options())),
        num_hosts_(service.num_hosts()),
        num_clients_(static_cast<int32_t>(service.dst_entities().size()) *
                     service.num_hosts()),
        clients_(num_clients_),
        executor_(max_concurrent_pushes) {}

  // Sends the first request of every host.
  void Start() {
    for (int32_t c = 0; c < num_clients_; ++c) Send(c, {}, {});
  }

  // Stops pushing and waits until every request was answered. The scheduler
  // must have completed or failed.
  void Stop() {
    executor_.Shutdown(/*drain=*/false);
    absl::MutexLock lock(mu_);
    auto answered = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return outstanding_ == 0;
    };
    mu_.Await(absl::Condition(&answered));
  }

  int64_t pushes() const {
    absl::MutexLock lock(mu_);
    return pushes_;
  }

 private:
  struct Client {
    uint32_t seq = 0;
    int32_t inflight = 0;
    bool finished = false;
  };

  int32_t Replica(int32_t client) const { return client / num_hosts_; }
  int32_t Host(int32_t client) const { return client % num_hosts_; }

  void Send(int32_t client, absl::InlinedVector<uint64_t, 8> completed,
            absl::InlinedVector<uint64_t, 4> failed) {
    PullRequest req;
    req.replica = Replica(client);
    req.host = Host(client);
    req.completed = std::move(completed);
    req.failed = std::move(failed);
    uint32_t seq = 0;
    {
      absl::MutexLock lock(mu_);
      Client& c = clients_[client];
      req.max_grants = std::max(0, grant_batch_ - c.inflight);
      seq = ++c.seq;
      ++outstanding_;
    }
    service_.scheduler().Submit(std::move(req),
                                [this, client, seq](PullReply reply) {
                                  OnReply(client, seq, std::move(reply));
                                });
  }

  // Runs on a scheduler thread: must not block.
  void OnReply(int32_t client, uint32_t seq, PullReply reply) {
    bool ask_again = false;
    {
      absl::MutexLock lock(mu_);
      Client& c = clients_[client];
      if (reply.kind != PullReplyKind::kGrants) c.finished = true;
      // Grants of a superseded request are leases too.
      c.inflight += static_cast<int32_t>(reply.grants.size());
      ask_again = reply.kind == PullReplyKind::kGrants && seq == c.seq &&
                  c.inflight < grant_batch_;
    }
    for (const PullGrant& grant : reply.grants) {
      executor_.Schedule([this, client, grant] { Push(client, grant); });
    }
    if (ask_again) Send(client, {}, {});
    absl::MutexLock lock(mu_);
    --outstanding_;
  }

  // Runs on a pool thread.
  void Push(int32_t client, const PullGrant& grant) {
    const absl::Span<const WorkUnitEntity> dst = service_.dst_entities();
    absl::StatusOr<HostTransferCommand> cmd =
        DynamicPullEngine::BuildPushCommand(plan_, dst[grant.source_replica],
                                            dst[Replica(client)], Host(client),
                                            grant.bundle_id);
    absl::Status status = cmd.status();
    if (status.ok() && !cmd->request.shard_push_schedules().empty()) {
      status = SendStartTransfer(*cmd, send_rpc_);
    }
    if (!status.ok()) {
      LOG_EVERY_N_SEC(WARNING, 1)
          << "Transfer " << plan_.req_id << ": host " << Host(client)
          << " of replica " << grant.source_replica << " failed to push bundle "
          << grant.bundle_id << " to replica " << Replica(client) << ": "
          << status;
    }
    {
      absl::MutexLock lock(mu_);
      --clients_[client].inflight;
      ++pushes_;
    }
    if (status.ok()) {
      Send(client, {grant.lease_id}, {});
    } else {
      Send(client, {}, {grant.lease_id});
    }
  }

  const MaterializedTransferPlan& plan_;
  TransferPullService& service_;
  const ControlRpcSender& send_rpc_;
  const int32_t grant_batch_;
  const int32_t num_hosts_;
  const int32_t num_clients_;

  mutable absl::Mutex mu_;
  std::vector<Client> clients_ ABSL_GUARDED_BY(mu_);
  // Requests submitted and not answered yet.
  int64_t outstanding_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t pushes_ ABSL_GUARDED_BY(mu_) = 0;

  // Last member: its threads stop before the state above goes away.
  BoundedExecutor executor_;
};

}  // namespace

absl::StatusOr<MaterializedTransferPlan>
DynamicPullEngine::MaterializeTransferPlan(
    absl::string_view req_id, uint64_t uuid, const WorkUnitEntity& src_entity,
    absl::Span<const WorkUnitEntity> dst_entities,
    const LogicalReshardSchedule& schedule,
    tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h,
    int32_t parallelism) {
  if (dst_entities.empty()) {
    return absl::InvalidArgumentError("dst_entities must not be empty");
  }
  const int32_t num_dst = static_cast<int32_t>(dst_entities.size());
  if (schedule.num_dst_replicas != num_dst) {
    return absl::InvalidArgumentError(
        absl::StrCat("Schedule was computed for ", schedule.num_dst_replicas,
                     " destination replicas, got ", num_dst));
  }
  const SeedLayout& layout = schedule.seed_layout;
  const int32_t num_layers =
      static_cast<int32_t>(schedule.variable_names.size());

  MaterializedTransferPlan plan;
  plan.req_id = std::string(req_id);
  plan.uuid = uuid;
  plan.dst_layer_shard_bytes = schedule.dst_layer_shard_bytes;
  plan.skip_tiling_by_layer = schedule.skip_tiling_by_layer;
  plan.dst_mem_type = dst_mem_type;
  plan.parallelism = std::max<int32_t>(1, parallelism);
  plan.variable_bundles = schedule.variable_bundles;
  plan.seed_layout = layout;

  // Bundles the Trainer pushes to every replica.
  std::vector<std::vector<int32_t>> seeded(num_dst);
  for (size_t g = 0; g < layout.stripe_seeds.size(); ++g) {
    if (g >= layout.stripe_bundles.size()) break;
    for (int32_t r : layout.stripe_seeds[g]) {
      if (r < 0 || r >= num_dst) {
        return absl::InvalidArgumentError(
            absl::StrCat("Seed ", r, " of stripe ", g, " out of range for ",
                         num_dst, " destination replicas"));
      }
      seeded[r].insert(seeded[r].end(), layout.stripe_bundles[g].begin(),
                       layout.stripe_bundles[g].end());
    }
  }

  // 1. One receiver command per sampler host.
  for (int32_t r = 0; r < num_dst; ++r) {
    const WorkUnitEntity& ent = dst_entities[r];
    std::sort(seeded[r].begin(), seeded[r].end());
    ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> seeded_layers,
                          BundleLayers(schedule.variable_bundles, seeded[r]));
    for (const HostAttachment& host : ent.hosts) {
      if (host.control_address.empty()) continue;
      SamplerHostCommand sampler_cmd;
      sampler_cmd.replica_idx = r;
      sampler_cmd.seeded_bundles = seeded[r];
      sampler_cmd.seeded_layers = seeded_layers;

      HostTransferCommand& cmd = sampler_cmd.command;
      cmd.unit = HostUnit(ent, host);
      cmd.host_idx = host.host_idx;
      cmd.control_endpoint = host.control_address;
      auto& req = cmd.request;
      *req.add_src_units() = RaidenIdToProto(src_entity.unit);
      *req.add_dst_units() = RaidenIdToProto(cmd.unit);
      req.set_uuid(static_cast<int64_t>(uuid));
      req.set_is_sender(false);
      req.set_dst_mem_type(dst_mem_type);
      req.set_req_id(std::string(req_id));
      req.set_skip_d2h(skip_d2h);
      req.set_parallelism(plan.parallelism);
      SetExpectedChunkCounts(host, num_layers, seeded_layers,
                             schedule.dst_shard_expected_layer_chunks, req);
      SetSkipTiling(schedule.skip_tiling_by_layer, req);
      plan.sampler_commands.push_back(std::move(sampler_cmd));
    }
  }

  // 2. Trainer sender commands, one set per wave. Every push carries only the
  // layers of the seed's own stripe.
  std::vector<std::vector<int32_t>> stripe_layers;
  stripe_layers.reserve(layout.stripe_bundles.size());
  for (const std::vector<int32_t>& bundle_ids : layout.stripe_bundles) {
    ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> layers,
                          BundleLayers(schedule.variable_bundles, bundle_ids));
    stripe_layers.push_back(std::move(layers));
  }
  for (const std::vector<std::pair<int32_t, int32_t>>& wave :
       layout.trainer_waves) {
    const int32_t wave_size = static_cast<int32_t>(wave.size());
    for (const auto& [seed, stripe] : wave) {
      if (seed < 0 || seed >= num_dst || stripe < 0 ||
          stripe >= static_cast<int32_t>(stripe_layers.size())) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid Trainer push of stripe ", stripe,
                         " to destination replica ", seed));
      }
    }
    std::vector<HostTransferCommand> wave_cmds;
    for (size_t t = 0; t < src_entity.hosts.size(); ++t) {
      const HostAttachment& host = src_entity.hosts[t];
      if (host.control_address.empty() || wave_size == 0) continue;
      HostTransferCommand cmd;
      cmd.unit = HostUnit(src_entity, host);
      cmd.host_idx = host.host_idx;
      cmd.control_endpoint = host.control_address;

      auto& req = cmd.request;
      *req.add_src_units() = RaidenIdToProto(cmd.unit);
      for (const auto& [seed, stripe] : wave) {
        *req.add_dst_units() = RaidenIdToProto(dst_entities[seed].unit);
      }
      req.set_uuid(static_cast<int64_t>(uuid));
      req.set_is_sender(true);
      req.set_dst_mem_type(dst_mem_type);
      req.set_req_id(std::string(req_id));
      req.set_skip_d2h(skip_d2h);
      req.set_parallelism(plan.parallelism);
      SetSkipTiling(schedule.skip_tiling_by_layer, req);

      auto* schedules_map = req.mutable_shard_push_schedules();
      for (size_t k = 0; k < host.owned_global_shard_indices.size(); ++k) {
        const int32_t s_shard = host.owned_global_shard_indices[k];
        const int32_t key_shard = (k < host.unit_local_shard_indices.size())
                                      ? host.unit_local_shard_indices[k]
                                      : s_shard;
        if (s_shard < 0 ||
            s_shard >= static_cast<int32_t>(
                           schedule.trainer_actions_by_shard_and_plan.size())) {
          continue;
        }
        const auto& actions_by_plan =
            schedule.trainer_actions_by_shard_and_plan[s_shard];
        tpu_sync::rpc::ShardPushScheduleProto shard_sched;
        // Trainer host `t` starts with push `t % wave_size`, so that the
        // hosts spread over the wave's seeds.
        for (int32_t i = 0; i < wave_size; ++i) {
          const auto& [seed, stripe] =
              wave[(static_cast<int32_t>(t) + i) % wave_size];
          const WorkUnitEntity& target_ent = dst_entities[seed];
          for (int32_t l : stripe_layers[stripe]) {
            if (l < 0 || l >= static_cast<int32_t>(
                                  schedule.variable_to_plan_id.size())) {
              continue;
            }
            const int32_t pid = schedule.variable_to_plan_id[l];
            if (pid < 0 ||
                pid >= static_cast<int32_t>(actions_by_plan.size())) {
              continue;
            }
            for (const ShardCopyAction& act : actions_by_plan[pid]) {
              if (act.dst_shard_idx < 0 ||
                  act.dst_shard_idx >=
                      static_cast<int32_t>(target_ent.shards.size())) {
                return absl::InvalidArgumentError(
                    absl::StrCat("Destination shard index ", act.dst_shard_idx,
                                 " out of range for replica ",
                                 target_ent.unit.job_replica_id));
              }
              auto* entry = shard_sched.add_entries();
              entry->add_dst_peers(target_ent.shards[act.dst_shard_idx]);
              entry->set_dst_shard_idx(act.dst_shard_idx);
              entry->set_src_offset_bytes(act.src_offset_bytes);
              entry->set_dst_offset_bytes(act.dst_offset_bytes);
              entry->set_size_bytes(act.size_bytes);
              entry->set_src_stride_bytes(act.src_stride_bytes);
              entry->set_dst_stride_bytes(act.dst_stride_bytes);
              entry->set_count(static_cast<int32_t>(act.count));
              entry->set_layer_idx(l);
            }
          }
        }
        if (shard_sched.entries_size() > 0) {
          (*schedules_map)[key_shard] = std::move(shard_sched);
        }
      }
      if (!schedules_map->empty()) {
        wave_cmds.push_back(std::move(cmd));
      }
    }
    plan.trainer_wave_commands.push_back(std::move(wave_cmds));
  }
  return plan;
}

absl::StatusOr<HostTransferCommand> DynamicPullEngine::BuildPushCommand(
    const MaterializedTransferPlan& plan, const WorkUnitEntity& source,
    const WorkUnitEntity& puller, int32_t host_idx, int32_t bundle_id) {
  if (bundle_id < 0 ||
      bundle_id >= static_cast<int32_t>(plan.variable_bundles.size())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Bundle id ", bundle_id, " out of range for ",
                     plan.variable_bundles.size(), " bundles"));
  }
  if (host_idx < 0 || host_idx >= static_cast<int32_t>(source.hosts.size()) ||
      host_idx >= static_cast<int32_t>(puller.hosts.size())) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Host ", host_idx, " cannot push from a replica with ",
        source.hosts.size(), " hosts to one with ", puller.hosts.size(),
        "; destination replicas must have the same topology"));
  }
  const VariableBundleSpec& bundle = plan.variable_bundles[bundle_id];
  const HostAttachment& src_host = source.hosts[host_idx];
  HostTransferCommand cmd;
  cmd.unit = HostUnit(source, src_host);
  cmd.host_idx = src_host.host_idx;
  cmd.control_endpoint = src_host.control_address;

  auto& req = cmd.request;
  *req.add_src_units() = RaidenIdToProto(cmd.unit);
  *req.add_dst_units() =
      RaidenIdToProto(HostUnit(puller, puller.hosts[host_idx]));
  req.set_uuid(static_cast<int64_t>(plan.uuid));
  req.set_is_sender(true);
  req.set_dst_mem_type(plan.dst_mem_type);
  req.set_req_id(plan.req_id);
  req.set_skip_d2h(true);
  req.set_parallelism(plan.parallelism);
  SetSkipTiling(plan.skip_tiling_by_layer, req);

  auto* schedules_map = req.mutable_shard_push_schedules();
  for (size_t k = 0; k < src_host.owned_global_shard_indices.size(); ++k) {
    const int32_t g_shard = src_host.owned_global_shard_indices[k];
    const int32_t key_shard = (k < src_host.unit_local_shard_indices.size())
                                  ? src_host.unit_local_shard_indices[k]
                                  : g_shard;
    if (g_shard < 0 || g_shard >= static_cast<int32_t>(puller.shards.size())) {
      continue;
    }
    tpu_sync::rpc::ShardPushScheduleProto shard_sched;
    for (size_t bi = 0; bi < bundle.layer_indices.size(); ++bi) {
      const int32_t l_idx = bundle.layer_indices[bi];
      int64_t layer_bytes = 0;
      if (l_idx >= 0 &&
          l_idx < static_cast<int32_t>(plan.dst_layer_shard_bytes.size())) {
        layer_bytes = plan.dst_layer_shard_bytes[l_idx];
      } else if (bi < bundle.layer_byte_sizes.size()) {
        layer_bytes = bundle.layer_byte_sizes[bi];
      }
      auto* entry = shard_sched.add_entries();
      entry->add_dst_peers(puller.shards[g_shard]);
      entry->set_dst_shard_idx(g_shard);
      entry->set_src_offset_bytes(0);
      entry->set_dst_offset_bytes(0);
      entry->set_size_bytes(layer_bytes);
      entry->set_src_stride_bytes(layer_bytes);
      entry->set_dst_stride_bytes(layer_bytes);
      entry->set_count(1);
      entry->set_layer_idx(l_idx);
    }
    if (shard_sched.entries_size() > 0) {
      (*schedules_map)[key_shard] = std::move(shard_sched);
    }
  }
  return cmd;
}

absl::Status DynamicPullEngine::ExecuteTransfer(
    const MaterializedTransferPlan& plan,
    absl::Span<const WorkUnitEntity> dst_entities, PullPhase& pull_phase,
    const RpcSenderFn& send_rpc, absl::Time deadline) {
  // Step 1: Arm every sampler host for the Trainer push and its pulls.
  ABSL_RETURN_IF_ERROR(CheckTransferDeadline(plan.req_id, deadline,
                                             "before arming the samplers"));
  std::vector<HostTransferCommand> receivers;
  receivers.reserve(plan.sampler_commands.size());
  for (const SamplerHostCommand& sampler_cmd : plan.sampler_commands) {
    receivers.push_back(sampler_cmd.command);
  }
  ABSL_RETURN_IF_ERROR(DispatchStartTransfers(receivers, send_rpc));
  // Step 2.
  ABSL_RETURN_IF_ERROR(CheckTransferDeadline(plan.req_id, deadline,
                                             "before arming the pull phase"));
  ABSL_RETURN_IF_ERROR(pull_phase.Arm(plan, dst_entities, send_rpc, deadline));
  // Step 3: Trainer waves, one after another.
  for (size_t w = 0; w < plan.trainer_wave_commands.size(); ++w) {
    ABSL_RETURN_IF_ERROR(CheckTransferDeadline(
        plan.req_id, deadline, absl::StrCat("before Trainer wave ", w)));
    ABSL_RETURN_IF_ERROR(
        DispatchStartTransfers(plan.trainer_wave_commands[w], send_rpc));
  }
  // Step 4: Complete every replica.
  ABSL_RETURN_IF_ERROR(
      CheckTransferDeadline(plan.req_id, deadline, "before the pull phase"));
  return pull_phase.Run(plan, dst_entities, send_rpc, deadline);
}

absl::StatusOr<std::unique_ptr<TransferPullService>>
TransferPullService::Create(const MaterializedTransferPlan& plan,
                            std::vector<WorkUnitEntity> dst_entities,
                            PullSchedulerOptions options) {
  if (dst_entities.empty()) {
    return absl::InvalidArgumentError("dst_entities must not be empty");
  }
  const int32_t num_hosts = static_cast<int32_t>(dst_entities[0].hosts.size());
  for (size_t r = 0; r < dst_entities.size(); ++r) {
    if (static_cast<int32_t>(dst_entities[r].hosts.size()) != num_hosts ||
        num_hosts == 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Destination replica ", r, " has ", dst_entities[r].hosts.size(),
          " hosts, replica 0 has ", num_hosts,
          "; destination replicas must have the same topology"));
    }
  }
  for (size_t b = 0; b < plan.variable_bundles.size(); ++b) {
    if (plan.variable_bundles[b].bundle_id != static_cast<int32_t>(b)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Bundle ", b, " has bundle id ", plan.variable_bundles[b].bundle_id));
    }
  }
  const int32_t num_replicas = static_cast<int32_t>(dst_entities.size());
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PullScheduler> scheduler,
      PullScheduler::Create(
          num_replicas, num_hosts,
          static_cast<int32_t>(plan.variable_bundles.size()),
          PullScheduler::SeededBundles(plan.seed_layout, num_replicas),
          std::move(options)));
  return absl::WrapUnique(new TransferPullService(
      plan, std::move(dst_entities), num_hosts, std::move(scheduler)));
}

TransferPullService::TransferPullService(
    const MaterializedTransferPlan& plan,
    std::vector<WorkUnitEntity> dst_entities, int32_t num_hosts,
    std::unique_ptr<PullScheduler> scheduler)
    : req_id_(plan.req_id),
      uuid_(plan.uuid),
      dst_entities_(std::move(dst_entities)),
      num_hosts_(num_hosts),
      scheduler_(std::move(scheduler)) {
  for (size_t r = 0; r < dst_entities_.size(); ++r) {
    const WorkUnitEntity& ent = dst_entities_[r];
    replica_of_unit_.try_emplace(ent.unit, static_cast<int32_t>(r));
    for (const HostAttachment& host : ent.hosts) {
      if (!host.unit.job_name.empty()) {
        replica_of_unit_.try_emplace(host.unit, static_cast<int32_t>(r));
      }
    }
  }
}

absl::Status TransferPullService::CheckIds(absl::string_view req_id,
                                           uint64_t uuid) const {
  if (req_id != req_id_ || uuid != uuid_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Pull request for transfer ", req_id, " (uuid ", uuid,
        ") does not match transfer ", req_id_, " (uuid ", uuid_, ")"));
  }
  return absl::OkStatus();
}

absl::StatusOr<int32_t> TransferPullService::ReplicaOfUnit(
    const RaidenId& unit) const {
  auto it = replica_of_unit_.find(unit);
  if (it == replica_of_unit_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Unit ", unit.job_name, "/", unit.job_replica_id, "/",
                     unit.data_name, "/", unit.data_replica_idx,
                     " is not a destination of transfer ", req_id_));
  }
  return it->second;
}

PhysicalPullGrant TransferPullService::Bind(const PullGrant& grant,
                                            int32_t host_idx) const {
  const WorkUnitEntity& source = dst_entities_[grant.source_replica];
  const HostAttachment& host = source.hosts[host_idx];
  return PhysicalPullGrant{
      .bundle_id = grant.bundle_id,
      .source_replica = grant.source_replica,
      .source_unit = HostUnit(source, host),
      .source_data_endpoint =
          host.data_endpoints.empty() ? "" : host.data_endpoints[0],
      .lease_id = grant.lease_id,
      .expiry = grant.expiry,
  };
}

void TransferPullService::Handle(PullServiceRequest request,
                                 PullServiceCallback done) {
  absl::StatusOr<int32_t> replica = ReplicaOfUnit(request.unit);
  absl::Status status = CheckIds(request.req_id, request.uuid);
  if (status.ok()) status = replica.status();
  if (status.ok() && (request.host_idx < 0 || request.host_idx >= num_hosts_)) {
    status = absl::InvalidArgumentError(
        absl::StrCat("Host index ", request.host_idx, " out of range for ",
                     num_hosts_, " hosts"));
  }
  if (!status.ok()) {
    PullServiceReply reply;
    reply.kind = PullReplyKind::kAborted;
    reply.status = std::move(status);
    std::move(done)(std::move(reply));
    return;
  }
  for (const FailedPull& failed : request.failed) {
    LOG_EVERY_N_SEC(WARNING, 1)
        << "Transfer " << req_id_ << ": host " << request.host_idx
        << " of replica " << *replica << " failed to pull bundle "
        << PullScheduler::LeaseBundle(failed.lease_id) << ": " << failed.reason;
  }
  PullRequest req;
  req.replica = *replica;
  req.host = request.host_idx;
  req.completed.assign(request.completed.begin(), request.completed.end());
  for (const FailedPull& failed : request.failed) {
    req.failed.push_back(failed.lease_id);
  }
  req.max_grants = request.max_grants;
  req.long_poll = request.long_poll;
  req.seeded = request.seeded;
  req.lost_data = request.lost_data;
  const int32_t host = request.host_idx;
  scheduler_->Submit(std::move(req), [this, host, done = std::move(done)](
                                         PullReply reply) mutable {
    PullServiceReply out;
    out.kind = reply.kind;
    out.status = std::move(reply.status);
    out.grants.reserve(reply.grants.size());
    for (const PullGrant& grant : reply.grants) {
      out.grants.push_back(Bind(grant, host));
    }
    std::move(done)(std::move(out));
  });
}

absl::Status ScheduledPullPhase::Arm(
    const MaterializedTransferPlan& /*plan*/,
    absl::Span<const WorkUnitEntity> /*dst_entities*/,
    const ControlRpcSender& /*send_rpc*/, absl::Time /*deadline*/) {
  if (service_ == nullptr) {
    return absl::InvalidArgumentError("ScheduledPullPhase needs a service");
  }
  // The samplers start acquiring once armed; nothing else to do.
  return absl::OkStatus();
}

absl::Status ScheduledPullPhase::Run(
    const MaterializedTransferPlan& plan,
    absl::Span<const WorkUnitEntity> /*dst_entities*/,
    const ControlRpcSender& /*send_rpc*/, absl::Time deadline) {
  if (service_ == nullptr) {
    return absl::InvalidArgumentError("ScheduledPullPhase needs a service");
  }
  return AwaitPullPhase(service_->scheduler(), plan.req_id, deadline);
}

absl::Status ControllerDrivenPullPhase::Arm(
    const MaterializedTransferPlan& /*plan*/,
    absl::Span<const WorkUnitEntity> /*dst_entities*/,
    const ControlRpcSender& /*send_rpc*/, absl::Time /*deadline*/) {
  if (service_ == nullptr) {
    return absl::InvalidArgumentError(
        "ControllerDrivenPullPhase needs a service");
  }
  // The samplers were armed with `plan.sampler_commands`; nothing else to do.
  return absl::OkStatus();
}

absl::Status ControllerDrivenPullPhase::Run(
    const MaterializedTransferPlan& plan,
    absl::Span<const WorkUnitEntity> dst_entities,
    const ControlRpcSender& send_rpc, absl::Time deadline) {
  if (service_ == nullptr) {
    return absl::InvalidArgumentError(
        "ControllerDrivenPullPhase needs a service");
  }
  if (dst_entities.size() != service_->dst_entities().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Pull service has ", service_->dst_entities().size(),
        " destination replicas, got ", dst_entities.size(), " entities"));
  }
  PullScheduler& scheduler = service_->scheduler();
  // The Trainer push to the seeds was accepted.
  for (int32_t seed : AllSeeds(plan.seed_layout)) scheduler.MarkSeeded(seed);
  PushDriver driver(plan, *service_, send_rpc, max_concurrent_pushes_);
  driver.Start();
  absl::Status status = AwaitPullPhase(scheduler, plan.req_id, deadline);
  driver.Stop();
  return status;
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

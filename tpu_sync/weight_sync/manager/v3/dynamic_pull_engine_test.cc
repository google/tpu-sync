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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"
#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// One Trainer host with 2 shards.
WorkUnitEntity MakeTrainer() {
  WorkUnitEntity trainer;
  trainer.unit = RaidenId("trainer", "0", "weights");
  trainer.shards = {"127.0.0.1:8000", "127.0.0.1:8001"};
  trainer.control_endpoints = {"127.0.0.1:9000"};
  EntityRegistry::RebuildHostAttachments(&trainer);
  return trainer;
}

// |n| single-host, single-shard destination replicas.
std::vector<WorkUnitEntity> MakeDstEntities(int n) {
  std::vector<WorkUnitEntity> out(n);
  for (int i = 0; i < n; ++i) {
    out[i].unit = RaidenId(absl::StrCat("sampler_", i), "0", "weights", i);
    out[i].shards = {absl::StrCat("127.0.0.1:", 8100 + i)};
    out[i].control_endpoints = {absl::StrCat("127.0.0.1:", 9100 + i)};
    EntityRegistry::RebuildHostAttachments(&out[i]);
  }
  return out;
}

// One variable per entry of |rows| (`[rows, 64]` bf16), resharded from 2
// Trainer shards onto 1 sampler shard, one bundle per variable.
absl::StatusOr<LogicalReshardSchedule> MakeSchedule(
    int32_t num_dst, absl::Span<const int64_t> rows, int32_t trainer_streams,
    int32_t replication = 2) {
  std::vector<VariableSpec> src;
  std::vector<VariableSpec> dst;
  for (size_t l = 0; l < rows.size(); ++l) {
    VariableSpec v;
    v.name = absl::StrCat("w", l);
    v.global_shape = {rows[l], 64};
    v.mesh_shape = {2, 1};
    v.layout = {1, 0};
    v.itemsize = 2;
    v.layer_idx = static_cast<int32_t>(l);
    v.global_shard_indices = {0, 1};
    src.push_back(v);
    v.mesh_shape = {1, 1};
    v.global_shard_indices = {0};
    dst.push_back(v);
  }
  SeedingOptions seeding;
  seeding.replication = replication;
  seeding.trainer_streams = trainer_streams;
  return LogicalReshardPlanner::ComputeLogicalSchedule(
      src, dst, /*num_src_shards=*/2, /*num_dst_shards=*/1, num_dst,
      /*broadcast_host_ratio=*/1.0, /*trainer_hosts=*/1, /*sampler_hosts=*/1,
      /*num_bundle_groups=*/static_cast<int32_t>(rows.size()), seeding);
}

// The 8-sampler, 4-bundle example: stripe `g` = bundle `g`, seeded on
// replicas `2g` and `2g + 1`.
struct Fixture {
  WorkUnitEntity trainer = MakeTrainer();
  std::vector<WorkUnitEntity> dst;
  MaterializedTransferPlan plan;
};

absl::StatusOr<Fixture> MakeFixture(int32_t num_dst, std::vector<int64_t> rows,
                                    int32_t trainer_streams = 4) {
  Fixture fixture;
  fixture.dst = MakeDstEntities(num_dst);
  ABSL_ASSIGN_OR_RETURN(LogicalReshardSchedule schedule,
                        MakeSchedule(num_dst, rows, trainer_streams));
  ABSL_ASSIGN_OR_RETURN(
      fixture.plan,
      DynamicPullEngine::MaterializeTransferPlan(
          "req", /*uuid=*/42, fixture.trainer, fixture.dst, schedule,
          tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true,
          /*parallelism=*/1));
  return fixture;
}

absl::StatusOr<Fixture> MakeWorkedExample() {
  return MakeFixture(8, {32, 32, 32, 32});
}

absl::StatusOr<tpu_sync::rpc::ControlResponse> OkResponse() {
  tpu_sync::rpc::ControlResponse resp;
  resp.set_success(true);
  return resp;
}

TEST(DynamicPullEngineTest, MaterializesSamplerCommandsAndTrainerWaves) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  const MaterializedTransferPlan& plan = fixture->plan;
  EXPECT_THAT(plan.seed_layout.stripe_seeds,
              ElementsAre(ElementsAre(0, 1), ElementsAre(2, 3),
                          ElementsAre(4, 5), ElementsAre(6, 7)));

  // One receiver command per sampler host. Replica `r` expects
  // the Trainer's 2 chunks only for the layer of its own stripe, and one
  // whole-layer chunk for every pulled layer.
  ASSERT_THAT(plan.sampler_commands, SizeIs(8));
  for (int32_t r = 0; r < 8; ++r) {
    const SamplerHostCommand& cmd = plan.sampler_commands[r];
    EXPECT_EQ(cmd.replica_idx, r);
    EXPECT_EQ(cmd.command.control_endpoint,
              absl::StrCat("127.0.0.1:", 9100 + r));
    EXPECT_THAT(cmd.seeded_bundles, ElementsAre(r / 2));
    EXPECT_THAT(cmd.seeded_layers, ElementsAre(r / 2));
    const tpu_sync::rpc::StartTransferRequest& req = cmd.command.request;
    EXPECT_FALSE(req.is_sender());
    EXPECT_EQ(req.src_units(0).job_name(), "trainer");
    for (int32_t l = 0; l < 4; ++l) {
      EXPECT_EQ(req.expected_layer_chunk_counts().at(l), l == r / 2 ? 2 : 1)
          << "replica " << r << " layer " << l;
    }
    EXPECT_EQ(req.expected_block_count(), 5);
  }

  // Two Trainer waves of 4 pushes: copy 0 of every stripe, then copy 1. Each
  // push carries only the seed's own stripe.
  ASSERT_THAT(plan.trainer_wave_commands, SizeIs(2));
  for (int32_t w = 0; w < 2; ++w) {
    ASSERT_THAT(plan.trainer_wave_commands[w], SizeIs(1));
    const tpu_sync::rpc::StartTransferRequest& req =
        plan.trainer_wave_commands[w][0].request;
    EXPECT_TRUE(req.is_sender());
    ASSERT_EQ(req.dst_units_size(), 4);
    for (int32_t g = 0; g < 4; ++g) {
      EXPECT_EQ(req.dst_units(g).job_name(),
                absl::StrCat("sampler_", 2 * g + w));
    }
    ASSERT_EQ(req.shard_push_schedules_size(), 2);
    for (const auto& [shard, schedule] : req.shard_push_schedules()) {
      ASSERT_EQ(schedule.entries_size(), 4);
      for (const auto& entry : schedule.entries()) {
        EXPECT_EQ(entry.dst_peers(0),
                  fixture->dst[2 * entry.layer_idx() + w].shards[0]);
      }
    }
  }

  // The schedule must match the destination replicas.
  absl::StatusOr<LogicalReshardSchedule> schedule =
      MakeSchedule(8, {32, 32, 32, 32}, 4);
  ASSERT_OK(schedule);
  EXPECT_EQ(DynamicPullEngine::MaterializeTransferPlan(
                "req", 1, fixture->trainer, MakeDstEntities(7), *schedule)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

// Thread-safe, ordered record of control RPC endpoints and pull phase calls.
class EventLog {
 public:
  void Add(std::string event) {
    absl::MutexLock lock(mu_);
    events_.push_back(std::move(event));
  }
  std::vector<std::string> events() const {
    absl::MutexLock lock(mu_);
    return events_;
  }

 private:
  mutable absl::Mutex mu_;
  std::vector<std::string> events_ ABSL_GUARDED_BY(mu_);
};

class RecordingPullPhase final : public PullPhase {
 public:
  RecordingPullPhase(EventLog* log, absl::Status arm_status)
      : log_(log), arm_status_(std::move(arm_status)) {}

  absl::Status Arm(const MaterializedTransferPlan&,
                   absl::Span<const WorkUnitEntity>, const ControlRpcSender&,
                   absl::Time) override {
    log_->Add("Arm");
    return arm_status_;
  }
  absl::Status Run(const MaterializedTransferPlan&,
                   absl::Span<const WorkUnitEntity>, const ControlRpcSender&,
                   absl::Time) override {
    log_->Add("Run");
    return absl::OkStatus();
  }

 private:
  EventLog* log_;
  absl::Status arm_status_;
};

HostTransferCommand MakeCommand(absl::string_view endpoint) {
  HostTransferCommand cmd;
  cmd.control_endpoint = std::string(endpoint);
  return cmd;
}

TEST(DynamicPullEngineTest, ExecuteTransferArmsSamplersThenRunsWavesInOrder) {
  MaterializedTransferPlan plan;
  plan.req_id = "req_pull_phase";
  plan.sampler_commands.resize(2);
  plan.sampler_commands[0].command = MakeCommand("sampler_0");
  plan.sampler_commands[1].command = MakeCommand("sampler_1");
  plan.trainer_wave_commands = {
      {MakeCommand("wave_0")},
      {MakeCommand("wave_1a"), MakeCommand("wave_1b")}};

  EventLog log;
  auto sender = [&log](absl::string_view ep,
                       const tpu_sync::rpc::ControlRequest& req) {
    EXPECT_EQ(req.command(),
              tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER);
    log.Add(std::string(ep));
    return OkResponse();
  };
  RecordingPullPhase pull_phase(&log, absl::OkStatus());
  ASSERT_OK(DynamicPullEngine::ExecuteTransfer(plan, {}, pull_phase, sender,
                                               absl::InfiniteFuture()));
  // Samplers are armed (in parallel) before `Arm()`; the waves run one after
  // another (each in parallel) before `Run()`.
  const std::vector<std::string> events = log.events();
  ASSERT_THAT(events, SizeIs(7));
  EXPECT_THAT(std::vector<std::string>(events.begin(), events.begin() + 2),
              UnorderedElementsAre("sampler_0", "sampler_1"));
  EXPECT_THAT(std::vector<std::string>(events.begin() + 2, events.begin() + 4),
              ElementsAre("Arm", "wave_0"));
  EXPECT_THAT(std::vector<std::string>(events.begin() + 4, events.begin() + 6),
              UnorderedElementsAre("wave_1a", "wave_1b"));
  EXPECT_EQ(events[6], "Run");

  // A failing `Arm()` stops the transfer before the Trainer push.
  EventLog failed_log;
  auto failed_sender = [&failed_log](absl::string_view ep,
                                     const tpu_sync::rpc::ControlRequest&) {
    failed_log.Add(std::string(ep));
    return OkResponse();
  };
  RecordingPullPhase failing_phase(&failed_log,
                                   absl::UnavailableError("arm failed"));
  EXPECT_EQ(DynamicPullEngine::ExecuteTransfer(
                plan, {}, failing_phase, failed_sender, absl::InfiniteFuture())
                .code(),
            absl::StatusCode::kUnavailable);
  EXPECT_THAT(failed_log.events(),
              UnorderedElementsAre("sampler_0", "sampler_1", "Arm"));
}

TEST(DynamicPullEngineTest, ExecuteTransferStopsAtTheTransferDeadline) {
  MaterializedTransferPlan plan;
  plan.req_id = "req_deadline";
  plan.sampler_commands.resize(1);
  plan.sampler_commands[0].command = MakeCommand("sampler_0");
  plan.trainer_wave_commands = {{MakeCommand("wave_0")},
                                {MakeCommand("wave_1")}};

  // A transfer past its deadline sends nothing.
  EventLog expired_log;
  auto expired_sender = [&expired_log](absl::string_view ep,
                                       const tpu_sync::rpc::ControlRequest&) {
    expired_log.Add(std::string(ep));
    return OkResponse();
  };
  RecordingPullPhase expired_phase(&expired_log, absl::OkStatus());
  absl::Status expired = DynamicPullEngine::ExecuteTransfer(
      plan, {}, expired_phase, expired_sender, absl::Now() - absl::Seconds(1));
  EXPECT_EQ(expired.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(expired.message(),
              HasSubstr("Transfer req_deadline passed its deadline"));
  EXPECT_THAT(expired_log.events(), IsEmpty());

  // A deadline that passes during a wave stops the transfer before the next
  // wave.
  const absl::Time deadline = absl::Now() + absl::Seconds(2);
  EventLog log;
  auto slow_sender = [&log, deadline](absl::string_view ep,
                                      const tpu_sync::rpc::ControlRequest&) {
    log.Add(std::string(ep));
    if (ep == "wave_0") {
      absl::SleepFor(deadline - absl::Now() + absl::Milliseconds(50));
    }
    return OkResponse();
  };
  RecordingPullPhase pull_phase(&log, absl::OkStatus());
  absl::Status status = DynamicPullEngine::ExecuteTransfer(
      plan, {}, pull_phase, slow_sender, deadline);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(), HasSubstr("before Trainer wave 1"));
  EXPECT_THAT(log.events(), ElementsAre("sampler_0", "Arm", "wave_0"));
}

TEST(DynamicPullEngineTest, BuildPushCommandTargetsTheSameHostOfThePuller) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  absl::StatusOr<HostTransferCommand> cmd = DynamicPullEngine::BuildPushCommand(
      fixture->plan, fixture->dst[1], fixture->dst[5], /*host_idx=*/0,
      /*bundle_id=*/0);
  ASSERT_OK(cmd);
  EXPECT_EQ(cmd->control_endpoint, "127.0.0.1:9101");
  const tpu_sync::rpc::StartTransferRequest& req = cmd->request;
  EXPECT_TRUE(req.is_sender());
  EXPECT_TRUE(req.skip_d2h());
  EXPECT_EQ(req.req_id(), "req");
  ASSERT_EQ(req.dst_units_size(), 1);
  EXPECT_EQ(req.dst_units(0).job_name(), "sampler_5");
  ASSERT_EQ(req.shard_push_schedules_size(), 1);
  const auto& schedule = req.shard_push_schedules().begin()->second;
  ASSERT_EQ(schedule.entries_size(), 1);
  EXPECT_EQ(schedule.entries(0).layer_idx(), 0);
  EXPECT_EQ(schedule.entries(0).dst_peers(0), fixture->dst[5].shards[0]);
  EXPECT_EQ(schedule.entries(0).size_bytes(),
            fixture->plan.dst_layer_shard_bytes[0]);

  EXPECT_EQ(DynamicPullEngine::BuildPushCommand(fixture->plan, fixture->dst[1],
                                                fixture->dst[5], 1, 0)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(DynamicPullEngine::BuildPushCommand(fixture->plan, fixture->dst[1],
                                                fixture->dst[5], 0, 4)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

// Inline scheduler options (replies are delivered before `Handle` returns).
PullSchedulerOptions InlineOptions() {
  PullSchedulerOptions options;
  options.inline_execution = true;
  options.grant_batch_size = 2;
  options.max_concurrent_uploads_per_source = 2;
  return options;
}

// Calls `Handle` and returns the reply (inline scheduler only).
PullServiceReply Call(TransferPullService& service,
                      PullServiceRequest request) {
  std::optional<PullServiceReply> out;
  service.Handle(std::move(request),
                 [&out](PullServiceReply reply) { out = std::move(reply); });
  CHECK(out.has_value());
  return *std::move(out);
}

TEST(TransferPullServiceTest, TranslatesUnitsAndBindsGrantsToEndpoints) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  std::vector<WorkUnitEntity> uneven = fixture->dst;
  uneven[3].hosts.push_back(uneven[3].hosts[0]);
  EXPECT_EQ(TransferPullService::Create(fixture->plan, uneven, InlineOptions())
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      TransferPullService::Create(fixture->plan, fixture->dst, InlineOptions());
  ASSERT_OK(service);
  TransferPullService& s = **service;
  EXPECT_EQ(s.num_hosts(), 1);

  // Wrong transfer ids, unknown units and bad host indices are refused.
  PullServiceReply reply =
      Call(s, {.req_id = "other", .uuid = 42, .unit = fixture->dst[0].unit});
  EXPECT_EQ(reply.kind, PullReplyKind::kAborted);
  EXPECT_EQ(reply.status.code(), absl::StatusCode::kFailedPrecondition);
  reply = Call(s, {.req_id = "req", .uuid = 42, .unit = fixture->trainer.unit});
  EXPECT_EQ(reply.status.code(), absl::StatusCode::kNotFound);
  reply = Call(s, {.req_id = "req",
                   .uuid = 42,
                   .unit = fixture->dst[0].unit,
                   .host_idx = 1});
  EXPECT_EQ(reply.status.code(), absl::StatusCode::kInvalidArgument);

  // Replica 0 (a seed of bundle 0) reports `seeded`; a zero long poll is
  // answered right away.
  reply = Call(s, {.req_id = "req",
                   .uuid = 42,
                   .unit = fixture->dst[0].unit,
                   .max_grants = 2,
                   .long_poll = absl::ZeroDuration(),
                   .seeded = true});
  EXPECT_EQ(reply.kind, PullReplyKind::kGrants);
  EXPECT_THAT(reply.grants, IsEmpty());  // Nothing else is seeded yet.

  // Replica 4 is granted bundle 0 from replica 0, with its endpoint.
  reply = Call(s, {.req_id = "req",
                   .uuid = 42,
                   .unit = fixture->dst[4].unit,
                   .max_grants = 2,
                   .long_poll = absl::ZeroDuration()});
  ASSERT_EQ(reply.kind, PullReplyKind::kGrants);
  ASSERT_THAT(reply.grants, SizeIs(1));
  const PhysicalPullGrant& grant = reply.grants[0];
  EXPECT_EQ(grant.bundle_id, 0);
  EXPECT_EQ(grant.source_replica, 0);
  EXPECT_EQ(grant.source_unit, fixture->dst[0].unit);
  EXPECT_EQ(grant.source_data_endpoint, fixture->dst[0].shards[0]);
  EXPECT_EQ(PullScheduler::LeaseBundle(grant.lease_id), 0);

  // A failed pull excludes replica 0 for replica 4; the report's reply has
  // nothing else to grant.
  reply = Call(
      s,
      {.req_id = "req",
       .uuid = 42,
       .unit = fixture->dst[4].unit,
       .max_grants = 2,
       .long_poll = absl::ZeroDuration(),
       .failed = {{.lease_id = grant.lease_id, .reason = "connection reset"}}});
  EXPECT_EQ(reply.kind, PullReplyKind::kGrants);
  EXPECT_THAT(reply.grants, IsEmpty());
  EXPECT_EQ(s.scheduler().GetStats()[0].failures, 1);
}

// Runs one sampler host's pull loop against |service| on its own thread:
// every grant is pulled instantly.
class FakeSampler {
 public:
  FakeSampler(TransferPullService* service, RaidenId unit, bool seed)
      : service_(service), unit_(std::move(unit)), seed_(seed) {}

  void Start() { Ask({}, seed_); }

  // Waits for the final reply.
  PullReplyKind Wait() {
    absl::MutexLock lock(mu_);
    mu_.Await(absl::Condition(&finished_));
    return final_kind_;
  }

 private:
  void Ask(std::vector<uint64_t> completed, bool seeded) {
    service_->Handle({.req_id = service_->req_id(),
                      .uuid = service_->uuid(),
                      .unit = unit_,
                      .max_grants = 2,
                      .long_poll = absl::Seconds(10),
                      .seeded = seeded,
                      .completed = std::move(completed)},
                     [this](PullServiceReply reply) {
                       if (reply.kind != PullReplyKind::kGrants) {
                         absl::MutexLock lock(mu_);
                         final_kind_ = reply.kind;
                         finished_ = true;
                         return;
                       }
                       std::vector<uint64_t> done;
                       for (const PhysicalPullGrant& g : reply.grants) {
                         done.push_back(g.lease_id);
                       }
                       Ask(std::move(done), false);
                     });
  }

  TransferPullService* const service_;
  const RaidenId unit_;
  const bool seed_;
  absl::Mutex mu_;
  bool finished_ ABSL_GUARDED_BY(mu_) = false;
  PullReplyKind final_kind_ ABSL_GUARDED_BY(mu_) = PullReplyKind::kGrants;
};

TEST(ScheduledPullPhaseTest, CompletesWhenEverySamplerHoldsEveryBundle) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  PullSchedulerOptions options;
  options.grant_batch_size = 2;
  options.max_concurrent_uploads_per_source = 2;
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      TransferPullService::Create(fixture->plan, fixture->dst, options);
  ASSERT_OK(service);
  std::vector<std::unique_ptr<FakeSampler>> samplers;
  for (int32_t r = 0; r < 8; ++r) {
    samplers.push_back(std::make_unique<FakeSampler>(
        service->get(), fixture->dst[r].unit, /*seed=*/true));
  }
  ScheduledPullPhase phase(service->get());
  for (auto& sampler : samplers) sampler->Start();
  ASSERT_OK(DynamicPullEngine::ExecuteTransfer(
      fixture->plan, fixture->dst, phase,
      [](absl::string_view, const tpu_sync::rpc::ControlRequest&) {
        return OkResponse();
      },
      absl::Now() + absl::Seconds(60)));
  for (auto& sampler : samplers)
    EXPECT_EQ(sampler->Wait(), PullReplyKind::kDone);
  int64_t grants = 0;
  for (const PullShardStats& stats : (*service)->scheduler().GetStats()) {
    grants += stats.grants;
  }
  EXPECT_EQ(grants, 8 * 3);
}

TEST(ScheduledPullPhaseTest, TransferDeadlineAbortsTheSamplers) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      TransferPullService::Create(fixture->plan, fixture->dst,
                                  PullSchedulerOptions());
  ASSERT_OK(service);
  // Only replica 4 asks; the seeds never report `seeded`.
  FakeSampler sampler(service->get(), fixture->dst[4].unit, /*seed=*/false);
  sampler.Start();
  ScheduledPullPhase phase(service->get());
  absl::Status status = phase.Run(fixture->plan, fixture->dst, {},
                                  absl::Now() + absl::Milliseconds(100));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer req passed its deadline waiting for the "
                        "pull phase"));
  EXPECT_EQ(sampler.Wait(), PullReplyKind::kAborted);
}

// Fake workers for `ControllerDrivenPullPhase`: tracks which replica holds
// which bundle (bundle `l` is layer `l`) and checks that sources only push
// bundles they hold.
class FakeWorkers {
 public:
  explicit FakeWorkers(const SeedLayout& layout, int32_t replicas)
      : holds_(PullScheduler::SeededBundles(layout, replicas)) {}

  // Makes pushes from these replicas fail.
  void FailPushesFrom(std::vector<int32_t> replicas) {
    absl::MutexLock lock(mu_);
    failing_ = std::move(replicas);
  }
  void SetPushDelay(absl::Duration delay) {
    absl::MutexLock lock(mu_);
    delay_ = delay;
  }

  ControlRpcSender sender() {
    return [this](absl::string_view ep,
                  const tpu_sync::rpc::ControlRequest& req)
               -> absl::StatusOr<tpu_sync::rpc::ControlResponse> {
      const tpu_sync::rpc::StartTransferRequest& st =
          req.start_transfer_request();
      if (!st.is_sender() || ep == "127.0.0.1:9000") return OkResponse();
      int32_t source = 0;
      CHECK(absl::SimpleAtoi(ep.substr(ep.size() - 4), &source));
      source -= 9100;
      int32_t puller = 0;
      CHECK(absl::SimpleAtoi(
          absl::string_view(st.dst_units(0).job_name()).substr(8), &puller));
      const int32_t bundle =
          st.shard_push_schedules().begin()->second.entries(0).layer_idx();
      absl::Duration delay;
      {
        absl::MutexLock lock(mu_);
        ++pushes_;
        if (std::find(failing_.begin(), failing_.end(), source) !=
            failing_.end()) {
          return absl::UnavailableError(
              absl::StrCat("Replica ", source, " is unreachable"));
        }
        if (std::find(holds_[source].begin(), holds_[source].end(), bundle) ==
            holds_[source].end()) {
          ++pushes_from_non_holders_;
        }
        delay = delay_;
      }
      absl::SleepFor(delay);
      absl::MutexLock lock(mu_);
      holds_[puller].push_back(bundle);
      return OkResponse();
    };
  }

  int32_t pushes() const {
    absl::MutexLock lock(mu_);
    return pushes_;
  }
  int32_t pushes_from_non_holders() const {
    absl::MutexLock lock(mu_);
    return pushes_from_non_holders_;
  }
  // Distinct bundles replica |r| holds.
  int32_t held(int32_t r) const {
    absl::MutexLock lock(mu_);
    std::vector<int32_t> b = holds_[r];
    std::sort(b.begin(), b.end());
    return static_cast<int32_t>(std::unique(b.begin(), b.end()) - b.begin());
  }

 private:
  mutable absl::Mutex mu_;
  std::vector<std::vector<int32_t>> holds_ ABSL_GUARDED_BY(mu_);
  std::vector<int32_t> failing_ ABSL_GUARDED_BY(mu_);
  absl::Duration delay_ ABSL_GUARDED_BY(mu_);
  int32_t pushes_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t pushes_from_non_holders_ ABSL_GUARDED_BY(mu_) = 0;
};

absl::StatusOr<std::unique_ptr<TransferPullService>> MakeService(
    const Fixture& fixture, int32_t grant_batch = 2) {
  PullSchedulerOptions options;
  options.grant_batch_size = grant_batch;
  options.max_concurrent_uploads_per_source = grant_batch;
  return TransferPullService::Create(fixture.plan, fixture.dst, options);
}

TEST(ControllerDrivenPullPhaseTest, PushesEveryGrantFromAHolder) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      MakeService(*fixture);
  ASSERT_OK(service);
  FakeWorkers workers(fixture->plan.seed_layout, 8);
  ControllerDrivenPullPhase phase(service->get(), /*max_concurrent_pushes=*/4);
  ASSERT_OK(DynamicPullEngine::ExecuteTransfer(
      fixture->plan, fixture->dst, phase, workers.sender(),
      absl::Now() + absl::Seconds(60)));
  EXPECT_TRUE((*service)->scheduler().complete());
  EXPECT_EQ(workers.pushes(), 8 * 3);
  EXPECT_EQ(workers.pushes_from_non_holders(), 0);
  for (int32_t r = 0; r < 8; ++r) EXPECT_EQ(workers.held(r), 4) << r;
}

TEST(ControllerDrivenPullPhaseTest, FailingSourceIsAvoided) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      MakeService(*fixture);
  ASSERT_OK(service);
  FakeWorkers workers(fixture->plan.seed_layout, 8);
  // Replica 0 seeds bundle 0 with replica 1.
  workers.FailPushesFrom({0});
  ControllerDrivenPullPhase phase(service->get());
  ASSERT_OK(DynamicPullEngine::ExecuteTransfer(
      fixture->plan, fixture->dst, phase, workers.sender(),
      absl::Now() + absl::Seconds(60)));
  for (int32_t r = 1; r < 8; ++r) EXPECT_EQ(workers.held(r), 4) << r;
  const PullShardStats stats = (*service)->scheduler().GetStats()[0];
  EXPECT_GT(stats.failures, 0);
  EXPECT_EQ(workers.pushes_from_non_holders(), 0);
}

TEST(ControllerDrivenPullPhaseTest, LosingBothSeedsOfABundleAborts) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      MakeService(*fixture);
  ASSERT_OK(service);
  FakeWorkers workers(fixture->plan.seed_layout, 8);
  workers.FailPushesFrom({0, 1});
  ControllerDrivenPullPhase phase(service->get());
  absl::Status status = DynamicPullEngine::ExecuteTransfer(
      fixture->plan, fixture->dst, phase, workers.sender(),
      absl::Now() + absl::Seconds(60));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.message(), HasSubstr("No live replica holds bundle 0"));
}

TEST(ControllerDrivenPullPhaseTest, TransferDeadlineStopsThePushes) {
  absl::StatusOr<Fixture> fixture = MakeWorkedExample();
  ASSERT_OK(fixture);
  absl::StatusOr<std::unique_ptr<TransferPullService>> service =
      MakeService(*fixture);
  ASSERT_OK(service);
  FakeWorkers workers(fixture->plan.seed_layout, 8);
  workers.SetPushDelay(absl::Milliseconds(300));
  ControllerDrivenPullPhase phase(service->get(), /*max_concurrent_pushes=*/2);
  const absl::Time start = absl::Now();
  absl::Status status = DynamicPullEngine::ExecuteTransfer(
      fixture->plan, fixture->dst, phase, workers.sender(),
      start + absl::Milliseconds(500));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(), HasSubstr("Transfer req passed its deadline"));
  // Queued pushes were dropped; only the running ones finished.
  EXPECT_LT(workers.pushes(), 8 * 3);
  EXPECT_LT(absl::Now() - start, absl::Seconds(5));
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

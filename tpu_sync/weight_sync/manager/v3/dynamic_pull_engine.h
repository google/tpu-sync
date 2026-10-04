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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_DYNAMIC_PULL_ENGINE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_DYNAMIC_PULL_ENGINE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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

// Pairs a target host control-plane endpoint with the `StartTransferRequest`
// payload to dispatch to that host.
struct HostTransferCommand {
  RaidenId unit;
  int32_t host_idx = 0;
  std::string control_endpoint;
  tpu_sync::rpc::StartTransferRequest request;
};

// The receiver command of one sampler host: it expects the Trainer push for
// the layers of its seeded bundles and one whole-layer chunk per owned shard
// for every other layer, which it pulls.
struct SamplerHostCommand {
  int32_t replica_idx = 0;
  HostTransferCommand command;
  std::vector<int32_t> seeded_bundles;
  // Sorted layer indices of `seeded_bundles`.
  std::vector<int32_t> seeded_layers;
};

// Complete set of materialized control-plane transfer commands for a V3
// weight synchronization: the Trainer seeds stripes of the model onto a few
// replicas, and the samplers complete each other with pulls that a
// `PullScheduler` decides while the transfer runs.
struct MaterializedTransferPlan {
  std::string req_id;
  uint64_t uuid = 0;
  std::vector<int64_t> dst_layer_shard_bytes;
  std::vector<bool> skip_tiling_by_layer;
  tpu_sync::rpc::MemoryType dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_DRAM;
  int32_t parallelism = 1;
  std::vector<VariableBundleSpec> variable_bundles;
  SeedLayout seed_layout;

  // One command per sampler host of every destination replica, in replica
  // order.
  std::vector<SamplerHostCommand> sampler_commands;

  // `trainer_wave_commands[w]`: one sender command per Trainer host for wave
  // `w` of `seed_layout.trainer_waves`, pushing only the layers of each
  // seed's own stripe. Waves run one after another.
  std::vector<std::vector<HostTransferCommand>> trainer_wave_commands;
};

// Sends one control-plane request to a worker endpoint.
using ControlRpcSender =
    std::function<absl::StatusOr<tpu_sync::rpc::ControlResponse>(
        absl::string_view endpoint, const tpu_sync::rpc::ControlRequest& req)>;

// A grant bound to physical endpoints: host `h` of the requesting replica
// pulls `bundle_id` from host `h` of `source_unit` (all destination replicas
// share one shard layout, so host `h` of the source holds the same shards).
struct PhysicalPullGrant {
  int32_t bundle_id = 0;
  int32_t source_replica = 0;
  RaidenId source_unit;
  std::string source_data_endpoint;
  uint64_t lease_id = 0;
  absl::Time expiry;
};

// A lease the host could not pull.
struct FailedPull {
  uint64_t lease_id = 0;
  std::string reason;
};

// `AcquirePulls` (no reports) or `ReportPulls` (reports; its reply carries the
// next grants) of one sampler host.
struct PullServiceRequest {
  std::string req_id;
  uint64_t uuid = 0;
  // Replica or host unit of the requesting sampler.
  RaidenId unit;
  int32_t host_idx = 0;
  // New grants the host can take now (0: report only).
  int32_t max_grants = 0;
  // How long the request may stay parked until something can be granted
  // (capped by the scheduler's `long_poll_timeout`). Zero answers right away.
  absl::Duration long_poll = absl::ZeroDuration();
  // The Trainer push to the host landed: it holds its seeded bundles.
  bool seeded = false;
  // The host lost everything it received for this transfer (e.g. it
  // restarted) and pulls every bundle again.
  bool lost_data = false;
  std::vector<uint64_t> completed;
  std::vector<FailedPull> failed;
};

struct PullServiceReply {
  PullReplyKind kind = PullReplyKind::kGrants;
  std::vector<PhysicalPullGrant> grants;
  absl::Status status;
};

// Receives the reply of one request, exactly once. It may run on a scheduler
// thread and must not block.
using PullServiceCallback = absl::AnyInvocable<void(PullServiceReply) &&>;

// Serves the pull requests of one transfer: translates units and host
// indices to the replica indices of its `PullScheduler` and binds the grants
// to the endpoints of the sources. Data never goes through the controller.
//
// Thread-safe.
class TransferPullService {
 public:
  // Fails if |dst_entities| do not have the same number of hosts (at least
  // one), if the bundle ids of |plan| are not `[0, #bundles)`, or if the seed
  // layout does not seed every bundle on a replica of |dst_entities|.
  static absl::StatusOr<std::unique_ptr<TransferPullService>> Create(
      const MaterializedTransferPlan& plan,
      std::vector<WorkUnitEntity> dst_entities, PullSchedulerOptions options);

  TransferPullService(const TransferPullService&) = delete;
  TransferPullService& operator=(const TransferPullService&) = delete;

  // Handles |request| asynchronously (see `PullScheduler::Submit`). Requests
  // for another transfer, from units that are not destinations of this one
  // or with an invalid host index are answered right away with `kAborted`.
  void Handle(PullServiceRequest request, PullServiceCallback done);

  // Host |host_idx| of the source of |grant|.
  PhysicalPullGrant Bind(const PullGrant& grant, int32_t host_idx) const;

  // Returns the destination replica index of |unit| (a replica or host unit).
  absl::StatusOr<int32_t> ReplicaOfUnit(const RaidenId& unit) const;

  PullScheduler& scheduler() { return *scheduler_; }
  const PullScheduler& scheduler() const { return *scheduler_; }
  const std::string& req_id() const { return req_id_; }
  uint64_t uuid() const { return uuid_; }
  int32_t num_hosts() const { return num_hosts_; }
  absl::Span<const WorkUnitEntity> dst_entities() const {
    return dst_entities_;
  }

 private:
  TransferPullService(const MaterializedTransferPlan& plan,
                      std::vector<WorkUnitEntity> dst_entities,
                      int32_t num_hosts,
                      std::unique_ptr<PullScheduler> scheduler);

  absl::Status CheckIds(absl::string_view req_id, uint64_t uuid) const;

  const std::string req_id_;
  const uint64_t uuid_;
  const std::vector<WorkUnitEntity> dst_entities_;
  const int32_t num_hosts_;
  absl::flat_hash_map<RaidenId, int32_t, RaidenIdHash> replica_of_unit_;
  std::unique_ptr<PullScheduler> scheduler_;
};

// Phase 2 of a V3 transfer: completing every destination replica after (and,
// in the target, while) the Trainer seeds its stripes. `DynamicPullEngine`
// drives a transfer as
//   1. arm the receiver of every sampler host (`sampler_commands`),
//   2. `Arm()`,
//   3. dispatch the Trainer waves in order (the only resharding hop and the
//      only traffic on the Trainer's NICs),
//   4. `Run()`: return once every replica holds every bundle, or on the first
//      error.
// |deadline| is the deadline of the whole transfer (Trainer waves and pull
// phase). Both calls fail with `DeadlineExceededError` once it passed.
class PullPhase {
 public:
  virtual ~PullPhase() = default;

  // Called after the samplers were armed, before the Trainer push.
  virtual absl::Status Arm(const MaterializedTransferPlan& plan,
                           absl::Span<const WorkUnitEntity> dst_entities,
                           const ControlRpcSender& send_rpc,
                           absl::Time deadline) = 0;

  // Called after every Trainer wave was accepted.
  virtual absl::Status Run(const MaterializedTransferPlan& plan,
                           absl::Span<const WorkUnitEntity> dst_entities,
                           const ControlRpcSender& send_rpc,
                           absl::Time deadline) = 0;
};

// Target `PullPhase`: every sampler host runs its own pull loop against the
// controller's `TransferPullService` (`AcquirePulls` / `ReportPulls`), which
// lets pulls overlap the Trainer push. `Run()` waits until the scheduler
// completes or fails, or until the transfer deadline (which then aborts the
// transfer, so that the samplers still asking learn about it).
//
// TODO(justinlu): Implement the sampler side, then make this the
// controller's default: one transfer per sampler host that receives the
// Trainer push for its seeded layers (`SamplerHostCommand::seeded_bundles`,
// to be encoded in `StartTransferRequest`) and pulls the others with the same
// host tiling mode; a pull loop that acquires up to `grant_batch_size` grants,
// pulls each from the granted source, reports completions and failures
// (piggybacked on the next acquire), reports `seeded` once the Trainer push
// landed, and after a restart re-arms its receiver for every layer and
// acquires with `lost_data`; and a source-side cap of
// `max_concurrent_uploads_per_source` concurrent uploads.
class ScheduledPullPhase final : public PullPhase {
 public:
  // |service| must outlive this object.
  explicit ScheduledPullPhase(TransferPullService* service)
      : service_(service) {}

  absl::Status Arm(const MaterializedTransferPlan& plan,
                   absl::Span<const WorkUnitEntity> dst_entities,
                   const ControlRpcSender& send_rpc,
                   absl::Time deadline) override;
  absl::Status Run(const MaterializedTransferPlan& plan,
                   absl::Span<const WorkUnitEntity> dst_entities,
                   const ControlRpcSender& send_rpc,
                   absl::Time deadline) override;

 private:
  TransferPullService* service_;
};

// Placeholder `PullPhase` for workers without a pull loop: the controller
// acts as the client of the same `PullScheduler` on behalf of every sampler
// host. `Run()` marks the seeds as seeded (the Trainer waves were accepted),
// acquires grants for every host, and for each grant asks host `h` of the
// source to push the bundle to host `h` of the puller (one `StartTransfer`,
// whole layers per shard) on a bounded pool of threads. It reports each push
// as completed or failed and receives the next grants with that report.
//
// It intentionally differs from `ScheduledPullPhase`: sources push instead of
// pullers pulling, every pull costs a controller round trip to a worker, and
// pulls cannot overlap the Trainer push. Do not use it for performance
// comparisons.
//
// TODO(justinlu): Remove once the samplers run the pull loop
// (`ScheduledPullPhase`).
class ControllerDrivenPullPhase final : public PullPhase {
 public:
  // |service| must outlive this object. At most |max_concurrent_pushes|
  // pushes run at a time (>= 1).
  explicit ControllerDrivenPullPhase(TransferPullService* service,
                                     int32_t max_concurrent_pushes = 64)
      : service_(service), max_concurrent_pushes_(max_concurrent_pushes) {}

  absl::Status Arm(const MaterializedTransferPlan& plan,
                   absl::Span<const WorkUnitEntity> dst_entities,
                   const ControlRpcSender& send_rpc,
                   absl::Time deadline) override;
  absl::Status Run(const MaterializedTransferPlan& plan,
                   absl::Span<const WorkUnitEntity> dst_entities,
                   const ControlRpcSender& send_rpc,
                   absl::Time deadline) override;

 private:
  TransferPullService* service_;
  const int32_t max_concurrent_pushes_;
};

// Materializes logical reshard schedules into physical `StartTransferRequest`s
// and executes them: Phase 1 (Trainer -> seeds, in waves) and Phase 2
// (`PullPhase`: samplers complete each other).
class DynamicPullEngine {
 public:
  using RpcSenderFn = ControlRpcSender;

  DynamicPullEngine() = default;

  // Materializes one `SamplerHostCommand` per sampler host and the Trainer
  // wave commands by binding |schedule| to the physical endpoints in
  // |src_entity| and |dst_entities| (one per destination replica of
  // |schedule|).
  static absl::StatusOr<MaterializedTransferPlan> MaterializeTransferPlan(
      absl::string_view req_id, uint64_t uuid, const WorkUnitEntity& src_entity,
      absl::Span<const WorkUnitEntity> dst_entities,
      const LogicalReshardSchedule& schedule,
      tpu_sync::rpc::MemoryType dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_DRAM,
      bool skip_d2h = false, int32_t parallelism = 1);

  // The command that makes host |host_idx| of |source| push |bundle_id|'s
  // layers (whole layers per shard, from its post-reshard buffers) to the
  // same host of |puller|. Fails if either replica has no such host or the
  // bundle id is out of range.
  static absl::StatusOr<HostTransferCommand> BuildPushCommand(
      const MaterializedTransferPlan& plan, const WorkUnitEntity& source,
      const WorkUnitEntity& puller, int32_t host_idx, int32_t bundle_id);

  // Executes a materialized weight sync transfer plan:
  // 1. Arms every sampler host (`plan.sampler_commands`).
  // 2. `pull_phase.Arm()`.
  // 3. Dispatches the Trainer waves, one after another.
  // 4. `pull_phase.Run()`.
  // Stops at the first failing step and returns its error. Fails with
  // `DeadlineExceededError` if |deadline| passed before a step starts; the
  // pull phase honors it within its steps. |send_rpc| should bound each RPC
  // by |deadline| as well.
  static absl::Status ExecuteTransfer(
      const MaterializedTransferPlan& plan,
      absl::Span<const WorkUnitEntity> dst_entities, PullPhase& pull_phase,
      const RpcSenderFn& send_rpc, absl::Time deadline);
};

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_DYNAMIC_PULL_ENGINE_H_

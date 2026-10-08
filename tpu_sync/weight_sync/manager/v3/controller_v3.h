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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_CONTROLLER_V3_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_CONTROLLER_V3_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/detached_thread_group.h"
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

// Encoding of the pull RPCs (`AcquirePulls` / `ReportPulls`) in the payload
// of a `ControlEnvelope` on the pull server.
class PullWireCodec {
 public:
  virtual ~PullWireCodec() = default;
  // `ControlEnvelope::message_type` of the pull requests.
  virtual absl::string_view message_type() const = 0;
  virtual absl::StatusOr<PullServiceRequest> DecodeRequest(
      absl::string_view payload) const = 0;
  virtual std::string EncodeReply(const PullServiceReply& reply) const = 0;
};

// V3 Weight Synchronization Controller: the Trainer seeds stripes of the model
// onto a few destination replicas, and the samplers complete each other with
// pulls that the controller decides while the transfer runs. Every execution
// of a transfer gets a `TransferPullService` (one `PullScheduler`) that
// answers `AcquirePulls` / `ReportPulls`; it is replaced when the plan is
// executed again and dropped when the transfer record is evicted.
class RaidenControllerV3 {
 public:
  enum class PullPhaseKind {
    // The controller makes the sources push on behalf of the samplers
    // (`ControllerDrivenPullPhase`), for workers without a pull loop.
    kControllerDriven,
    // The samplers run the pull loop against the controller
    // (`ScheduledPullPhase`).
    kScheduled,
  };

  struct Options {
    int port = 0;
    double broadcast_host_ratio = 1.0;
    int32_t num_bundle_groups = 8;
    // Number of stripes G (0 = auto: `min(#bundles, D / seed_replication)`).
    int32_t num_stripes = 0;
    // Number of destination replicas every stripe is pushed to (R).
    int32_t seed_replication = 2;
    // Pull scheduling (see `PullSchedulerOptions`): leases a sampler host
    // holds at a time (k_b, at most `max_concurrent_uploads_per_source`),
    // concurrent uploads per source host (c), lease and long-poll timeouts.
    // Execution options: they apply to transfers started afterwards. Clamped
    // to >= 1.
    int32_t grant_batch_size = 8;
    int32_t max_concurrent_uploads_per_source = 8;
    int64_t lease_timeout_ms = 30000;
    int64_t long_poll_timeout_ms = 5000;
    PullPhaseKind pull_phase = PullPhaseKind::kControllerDriven;
    // Port of the pull server started by `StartServer()` (0: ephemeral), or
    // negative for none. It serves the pull RPCs asynchronously: a parked
    // long poll holds no thread. The connections are spread over
    // `pull_server_threads` event loops; one loop served ~40K pull RPCs/s on
    // an 8-vCPU VM (`controller_pull_rpc_benchmark`).
    int pull_server_port = -1;
    int32_t pull_server_threads = 2;
    // Encoding of the pull RPCs on the pull server. Without one, the server
    // answers them with UNIMPLEMENTED (see `HandlePullEnvelope`). For tests
    // and benchmarks until the wire commands exist.
    std::shared_ptr<const PullWireCodec> pull_wire_codec;
    // Deadline of a whole transfer (re-planning, Trainer waves and pull
    // phase), from the start of its execution; a per-transfer timeout
    // overrides it. A transfer still running at its deadline fails with
    // `DeadlineExceededError` and its record turns FAILED. Clamped to >= 1.
    int64_t transfer_timeout_ms = 600000;
    bool enable_plan_cache = true;
    // Retention of terminal (COMPLETED/FAILED) transfer records after they
    // finish. A record is only evicted once its result has been observed
    // (returned by a synchronous execute call, `WaitForTransfer` or
    // `GetTransferStatus`), so 0 means "evict at the next eviction pass after
    // the result was delivered". Negative disables eviction.
    double request_registry_ttl_s = 600.0;
    // Minimum retention of records nobody has consumed yet: materialized
    // plans that were never executed (NOT_STARTED, measured from
    // materialization) and terminal records whose result was never observed
    // (measured from completion). Such records are kept for
    // max(request_registry_ttl_s, min_unobserved_ttl_s), which bounds memory
    // for fire-and-forget callers.
    double min_unobserved_ttl_s = 60.0;
    std::optional<ControlPipeBackendType> backend_type = std::nullopt;
    DynamicPullEngine::RpcSenderFn custom_rpc_sender = nullptr;
  };

  explicit RaidenControllerV3(const Options& options);
  ~RaidenControllerV3();

  // Starts the embedded C++ ControlPipeServer on `options.port` (or an OS
  // ephemeral port if `port == 0`), and the pull server if configured.
  // Returns the bound port.
  absl::StatusOr<int> StartServer();

  // Stops the embedded C++ ControlPipeServer and the pull server, and waits
  // for background tasks.
  void StopServer();

  // Bound port of the pull server, or 0 if it is not running.
  int pull_server_port() const;
  std::optional<AsyncControlServer::Stats> GetPullServerStats() const;

  // Returns the bound port if `StartServer()` has been called, or the requested
  // port otherwise.
  int port() const;

  // Returns `"127.0.0.1:<port>"` for local/loopback worker callbacks, or an
  // explicitly configured controller address override.
  std::string controller_address() const;
  void set_controller_address(absl::string_view address);

  double broadcast_host_ratio() const { return broadcast_host_ratio_; }
  void set_broadcast_host_ratio(double ratio);

  int32_t num_bundle_groups() const { return num_bundle_groups_; }
  void set_num_bundle_groups(int32_t groups);

  // Seeding options (see `SeedingOptions`). Changing them invalidates cached
  // schedules and stored plans.
  int32_t num_stripes() const;
  void set_num_stripes(int32_t num_stripes);
  int32_t seed_replication() const;
  void set_seed_replication(int32_t replication);

  double request_registry_ttl_s() const;
  void set_request_registry_ttl_s(double ttl_s);

  // Pull scheduling options (see `Options`). Execution options: changes
  // apply to transfers started afterwards and never invalidate cached
  // schedules or stored plans.
  int32_t grant_batch_size() const;
  void set_grant_batch_size(int32_t grant_batch_size);
  int32_t max_concurrent_uploads_per_source() const;
  void set_max_concurrent_uploads_per_source(int32_t max_uploads);
  int64_t lease_timeout_ms() const;
  void set_lease_timeout_ms(int64_t timeout_ms);
  int64_t long_poll_timeout_ms() const;
  void set_long_poll_timeout_ms(int64_t timeout_ms);

  // Default transfer timeout (see `Options::transfer_timeout_ms`). Changes
  // apply to transfers started afterwards. An execution option: it never
  // invalidates cached schedules or stored plans.
  int64_t transfer_timeout_ms() const;
  void set_transfer_timeout_ms(int64_t timeout_ms);

  // Registers or updates a work unit in the `EntityRegistry`. Invalidates
  // cached logical schedules referencing the unit if its logical sharding
  // topology changed.
  absl::Status RegisterWorkUnit(
      const tpu_sync::rpc::RegisterWorkUnitRequest& req);

  // Attaches a host endpoint and owned shards to an existing work unit and
  // invalidates cached schedules referencing that unit.
  absl::Status AttachHost(const RaidenId& unit,
                          absl::string_view control_address,
                          absl::Span<const std::string> host_shards);

  // Computes (or retrieves from cache) the `LogicalReshardSchedule` for
  // transferring weights from |src_units| to |dst_units|.
  absl::StatusOr<LogicalReshardSchedule> GetOrComputeLogicalSchedule(
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling = {},
      bool use_cached_plan = true);

  // Materializes a `MaterializedTransferPlan` for |src_units| -> |dst_units|
  // and stores it under |req_id| (status NOT_STARTED). |uuid| == 0
  // auto-assigns one and an empty
  // |req_id| becomes `"req_<uuid>"`, drawn from the same counter as
  // `ExecuteTransferSync`/`StartTransferAsync` so ids never collide; the
  // returned plan carries the effective ids. Fails with
  // `FailedPreconditionError` while a transfer for |req_id| is in progress.
  absl::StatusOr<MaterializedTransferPlan> BuildMaterializedPlan(
      absl::string_view req_id, uint64_t uuid,
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units,
      tpu_sync::rpc::MemoryType dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_DRAM,
      bool skip_d2h = false, int32_t parallelism = 1,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling = {},
      bool use_cached_plan = true);

  // Synchronously executes the plan previously materialized under |req_id|
  // (against the destination units it was materialized for) without
  // re-materializing it, with a fresh `TransferPullService`. If
  // |expected_uuid| != 0 the stored plan must
  // have that uuid (otherwise `FailedPreconditionError`, e.g. the plan was
  // replaced). Fails with `NotFoundError` if no plan is stored and with
  // `FailedPreconditionError` while a transfer for |req_id| is in progress.
  // |transfer_timeout| overrides `transfer_timeout_ms()` for this execution
  // (`InvalidArgumentError` unless positive).
  absl::Status ExecuteMaterializedTransferSync(
      absl::string_view req_id, uint64_t expected_uuid = 0,
      const DynamicPullEngine::RpcSenderFn& rpc_sender_override = nullptr,
      std::optional<absl::Duration> transfer_timeout = std::nullopt);

  // Synchronously executes a full V3 weight sync transfer (`src_units ->
  // dst_units`). Reuses the plan stored under |req_id| if it was built for the
  // same units, uuid (or |uuid| == 0) and plan-affecting options
  // (|dst_mem_type|, |skip_d2h|, |parallelism|, |skip_tiling|) and no
  // controller setting (broadcast host ratio, bundle groups or seeding) or
  // work-unit registration changed since it was materialized;
  // otherwise materializes a new one (replacing the stored plan). Empty
  // |req_id| / zero |uuid| are auto-assigned. |transfer_timeout| overrides
  // `transfer_timeout_ms()` for this transfer (`InvalidArgumentError` unless
  // positive); it does not affect plan reuse.
  absl::Status ExecuteTransferSync(
      absl::string_view req_id, uint64_t uuid,
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units,
      tpu_sync::rpc::MemoryType dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_DRAM,
      bool skip_d2h = false, int32_t parallelism = 1,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling = {},
      std::optional<absl::Duration> transfer_timeout = std::nullopt);

  // Launches an asynchronous V3 weight sync transfer on the background thread
  // group and records its status under |req_id|. Plan reuse and
  // |transfer_timeout| follow `ExecuteTransferSync`. Returns OK without
  // starting anything if a transfer for |req_id| is already in progress.
  absl::Status StartTransferAsync(
      absl::string_view req_id, uint64_t uuid,
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units,
      tpu_sync::rpc::MemoryType dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_DRAM,
      bool skip_d2h = false, int32_t parallelism = 1,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling = {},
      std::optional<absl::Duration> transfer_timeout = std::nullopt);

  // Waits for transfer |req_id| to complete and returns its result (which
  // marks the result as observed). Returns `NotFoundError` if no record for
  // |req_id| exists, and `DeadlineExceededError` if the wait timed out (which
  // does not affect the transfer). Without |timeout| it waits until the
  // transfer's deadline plus `kWaitGraceAfterDeadline`, by which time the
  // transfer has failed if it did not complete; for a transfer that has not
  // started yet, until `transfer_timeout_ms()` plus that grace from now, or
  // until the deadline of its execution once it starts.
  absl::Status WaitForTransfer(
      absl::string_view req_id,
      std::optional<absl::Duration> timeout = std::nullopt);

  // How long a default `WaitForTransfer` waits past the transfer deadline, so
  // that the transfer records its own failure first.
  static constexpr absl::Duration kWaitGraceAfterDeadline = absl::Seconds(5);

  // Returns the status enum (`GetTransferStatusResponse::Status`) for |req_id|
  // after evicting expired records (unknown/evicted ids report NOT_STARTED).
  // Reporting a terminal status marks the result as observed, so it becomes
  // evictable once `request_registry_ttl_s` has passed.
  tpu_sync::rpc::GetTransferStatusResponse::Status GetTransferStatus(
      absl::string_view req_id);

  // The pull RPCs of the sampler hosts, routed by `req_id` to the
  // `TransferPullService` of the transfer's current (or last) execution.
  // `AcquirePulls` asks for grants; `ReportPulls` reports completed and
  // failed leases and receives the next grants in its reply. |done| runs
  // exactly once, possibly later on a scheduler thread (a request that cannot
  // be granted anything yet is parked for up to its long poll); it must not
  // block. Requests for transfers without a pull service are answered with
  // `kAborted` (`NotFoundError`), `AcquirePulls` with reports with `kAborted`
  // (`InvalidArgumentError`).
  void AcquirePulls(PullServiceRequest request, PullServiceCallback done);
  void ReportPulls(PullServiceRequest request, PullServiceCallback done);

  // Scheduler counters of the current (or last) execution of |req_id|.
  absl::StatusOr<std::vector<PullShardStats>> GetPullStats(
      absl::string_view req_id) const;

  // Handles an incoming `ControlRequest` (start transfer, work unit
  // registration, metadata, shutdown).
  tpu_sync::rpc::ControlResponse HandleControlRequest(
      const tpu_sync::rpc::ControlRequest& req);

  // Handles an incoming `ControllerRequest` (`COMMAND_COORDINATE_TRANSFER`,
  // `COMMAND_GET_TRANSFER_STATUS`).
  tpu_sync::rpc::ControllerResponse HandleControllerRequest(
      const tpu_sync::rpc::ControllerRequest& req);

  // Handles a raw wire frame that may be either a `ControllerRequest` or
  // `ControlRequest` (supporting both Mode 1 and Mode 3 TCP framing).
  std::string HandleRawFrame(absl::string_view request_bytes);

  void ClearPlanCache();
  size_t GetPlanCacheSize() const;
  size_t GetTransferRecordCount();
  // Returns `req_id -> uuid` for every retained transfer record that holds a
  // materialized plan (after evicting expired records).
  absl::flat_hash_map<std::string, uint64_t> GetRetainedPlanUuids();
  int64_t GetPlanMaterializationCountForTest() const {
    return plan_materialization_count_.load(std::memory_order_relaxed);
  }

  EntityRegistry& entity_registry() { return entity_registry_; }
  const EntityRegistry& entity_registry() const { return entity_registry_; }

  // Returns the current pull service of |req_id|, or nullptr if none is
  // retained.
  std::shared_ptr<TransferPullService> GetPullServiceForTest(
      absl::string_view req_id) const;

  // Sends |req| to the worker at |endpoint|. The RPC is bounded by
  // `kWorkerRpcTimeout` and by |deadline| (`DeadlineExceededError` without
  // sending if it already passed).
  absl::StatusOr<tpu_sync::rpc::ControlResponse> SendWorkerRpc(
      absl::string_view endpoint, const tpu_sync::rpc::ControlRequest& req,
      absl::Time deadline = absl::InfiniteFuture());

  static constexpr absl::Duration kWorkerRpcTimeout = absl::Seconds(120);

 private:
  // Key of the logical schedule cache. Together with the invalidation on
  // registration changes it covers every input of
  // `LogicalReshardPlanner::ComputeLogicalSchedule` (including `skip_tiling`).
  // `dst_mem_type`, `skip_d2h` and `parallelism` do not affect the logical
  // schedule; they are applied per transfer by
  // `DynamicPullEngine::MaterializeTransferPlan` and guarded by `PlanOptions`.
  struct ScheduleCacheKey {
    std::vector<RaidenId> src_units;
    std::vector<RaidenId> dst_units;
    int64_t ratio_millis = 1000;
    int32_t num_bundle_groups = 8;
    int32_t num_stripes = 0;
    int32_t seed_replication = 2;
    std::vector<std::pair<int32_t, bool>> skip_tiling_sorted;

    bool operator==(const ScheduleCacheKey& o) const {
      return std::tie(src_units, dst_units, ratio_millis, num_bundle_groups,
                      num_stripes, seed_replication, skip_tiling_sorted) ==
             std::tie(o.src_units, o.dst_units, o.ratio_millis,
                      o.num_bundle_groups, o.num_stripes, o.seed_replication,
                      o.skip_tiling_sorted);
    }

    template <typename H>
    friend H AbslHashValue(H h, const ScheduleCacheKey& k) {
      h = H::combine(std::move(h), k.src_units.size(), k.dst_units.size());
      for (const RaidenId& u : k.src_units) {
        h = H::combine(std::move(h), u.job_name, u.job_replica_id, u.data_name,
                       u.data_replica_idx);
      }
      for (const RaidenId& u : k.dst_units) {
        h = H::combine(std::move(h), u.job_name, u.job_replica_id, u.data_name,
                       u.data_replica_idx);
      }
      return H::combine(std::move(h), k.ratio_millis, k.num_bundle_groups,
                        k.num_stripes, k.seed_replication,
                        k.skip_tiling_sorted);
    }
  };

  // Normalized per-transfer options a materialized plan was built with. A
  // stored plan is only reused when the requested options are equal. The
  // transfer timeout is an execution option, not a plan option: it is
  // deliberately absent here and from `ScheduleCacheKey`.
  struct PlanOptions {
    tpu_sync::rpc::MemoryType dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_DRAM;
    bool skip_d2h = false;
    // Clamped to >= 1, as in the materialized commands.
    int32_t parallelism = 1;
    // Sorted layer indices with `skip_tiling == true` (false entries are
    // equivalent to absent ones).
    std::vector<int32_t> skip_tiling_layers;

    static PlanOptions Make(
        tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h,
        int32_t parallelism,
        const absl::flat_hash_map<int32_t, bool>& skip_tiling);

    bool operator==(const PlanOptions& o) const {
      return std::tie(dst_mem_type, skip_d2h, parallelism,
                      skip_tiling_layers) == std::tie(o.dst_mem_type,
                                                      o.skip_d2h, o.parallelism,
                                                      o.skip_tiling_layers);
    }
  };

  struct TransferSessionRecord {
    tpu_sync::rpc::GetTransferStatusResponse::Status status =
        tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED;
    absl::Status result_status = absl::OkStatus();
    // Materialized plan and the units/entities/options it was built for.
    std::optional<MaterializedTransferPlan> plan;
    std::vector<RaidenId> src_units;
    std::vector<RaidenId> dst_units;
    std::vector<WorkUnitEntity> dst_entities;
    PlanOptions plan_options;
    // Value of `plan_inputs_generation_` when the plan was materialized.
    uint64_t plan_inputs_generation = 0;
    // Pull service of the current (or last) execution; serves the pull
    // RPCs. Created when an execution is claimed.
    std::shared_ptr<TransferPullService> pull_service;
    // True once a terminal result has been delivered to a caller (synchronous
    // execute, `WaitForTransfer` or `GetTransferStatus`). Unobserved terminal
    // records are retained for at least `min_unobserved_ttl_s_`.
    bool observed = false;
    absl::Time planned_at = absl::InfinitePast();
    absl::Time completed_at = absl::InfiniteFuture();
    // Deadline of the current (or last) execution, set when it is claimed.
    absl::Time deadline = absl::InfiniteFuture();
  };

  struct StoredPlan {
    MaterializedTransferPlan plan;
    std::vector<WorkUnitEntity> dst_entities;
    std::shared_ptr<TransferPullService> pull_service;
  };

  void InvalidateUnitInScheduleCache(const RaidenId& unit)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Evicts expired records (except |keep|), dropping their pull services.
  void EvictExpiredTransfersLocked(absl::Time now, absl::string_view keep = "")
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Scheduler options of an execution started now.
  PullSchedulerOptions PullSchedulerOptionsLocked() const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Creates the pull service of a new execution of |rec|'s plan (replacing
  // the previous one).
  absl::StatusOr<std::shared_ptr<TransferPullService>> NewPullServiceLocked(
      TransferSessionRecord& rec) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the pull service of transfer |req_id| (`NotFoundError` if none).
  absl::StatusOr<std::shared_ptr<TransferPullService>> FindPullService(
      absl::string_view req_id) const;

  void HandlePullRequest(PullServiceRequest request, PullServiceCallback done);

  // Serves one request of the pull server.
  void HandlePullEnvelope(const ControlContext& ctx,
                          control_pipe::proto::ControlEnvelope request,
                          AsyncControlReply reply);

  absl::StatusOr<LogicalReshardSchedule> ComputeScheduleForEntities(
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units, const WorkUnitEntity& src_ent,
      absl::Span<const WorkUnitEntity> dst_entities,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling,
      bool use_cached_plan);

  // Materializes and stores a plan for |req_id|. If |claimed| is true the
  // caller already marked the record IN_PROGRESS and will execute the plan
  // (with the returned pull service); otherwise the record is reset to
  // NOT_STARTED, holds no pull service, and in-progress transfers are
  // rejected.
  absl::StatusOr<StoredPlan> MaterializeAndStorePlan(
      absl::string_view req_id, uint64_t uuid,
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units,
      tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h,
      int32_t parallelism,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling,
      bool use_cached_plan, bool claimed);

  // Marks |req_id| IN_PROGRESS with |deadline|. Returns false (and changes
  // nothing) if it already is.
  bool TryClaimTransferLocked(absl::string_view req_id, absl::Time deadline)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The deadline of an execution claimed now: |transfer_timeout| if set
  // (`InvalidArgumentError` unless positive), else `transfer_timeout_ms_`.
  absl::StatusOr<absl::Time> ExecutionDeadlineLocked(
      std::optional<absl::Duration> transfer_timeout) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Body of a claimed `ExecuteTransferSync`/`StartTransferAsync`: reuses or
  // materializes the plan and executes it by |deadline|.
  absl::Status PlanAndRunClaimed(
      absl::string_view req_id, uint64_t uuid, bool uuid_was_given,
      absl::Span<const RaidenId> src_units,
      absl::Span<const RaidenId> dst_units,
      tpu_sync::rpc::MemoryType dst_mem_type, bool skip_d2h,
      int32_t parallelism,
      const absl::flat_hash_map<int32_t, bool>& skip_tiling,
      absl::Time deadline);

  // Executes |plan| with |pull_service| by |deadline|, with the pull phase
  // of `Options::pull_phase`. A failure at or after |deadline| is reported as
  // `DeadlineExceededError`.
  absl::Status RunPlan(const MaterializedTransferPlan& plan,
                       absl::Span<const WorkUnitEntity> dst_entities,
                       TransferPullService* pull_service,
                       const DynamicPullEngine::RpcSenderFn& send_rpc,
                       absl::Time deadline);

  // Records the terminal state of a claimed transfer. |observed| is true when
  // the result is returned directly to the caller (synchronous execution).
  void FinishTransfer(absl::string_view req_id, const absl::Status& status,
                      bool observed);

  int requested_port_ = 0;
  double broadcast_host_ratio_ = 1.0;
  int32_t num_bundle_groups_ = 8;
  double request_registry_ttl_s_ = 600.0;
  double min_unobserved_ttl_s_ = 60.0;
  bool enable_plan_cache_ = true;
  std::optional<ControlPipeBackendType> backend_type_;
  DynamicPullEngine::RpcSenderFn custom_rpc_sender_;
  const PullPhaseKind pull_phase_;
  const int requested_pull_server_port_;
  const int32_t pull_server_threads_;
  const std::shared_ptr<const PullWireCodec> pull_wire_codec_;

  EntityRegistry entity_registry_;

  mutable absl::Mutex mu_;
  int32_t num_stripes_ ABSL_GUARDED_BY(mu_) = 0;
  int32_t seed_replication_ ABSL_GUARDED_BY(mu_) = 2;
  int32_t grant_batch_size_ ABSL_GUARDED_BY(mu_) = 8;
  int32_t max_concurrent_uploads_per_source_ ABSL_GUARDED_BY(mu_) = 8;
  int64_t lease_timeout_ms_ ABSL_GUARDED_BY(mu_) = 30000;
  int64_t long_poll_timeout_ms_ ABSL_GUARDED_BY(mu_) = 5000;
  int64_t transfer_timeout_ms_ ABSL_GUARDED_BY(mu_) = 600000;
  std::string controller_address_override_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<ScheduleCacheKey, LogicalReshardSchedule> schedule_cache_
      ABSL_GUARDED_BY(mu_);
  // Bumped whenever a controller setting or work-unit registration that
  // materialized plans are built from changes (i.e. whenever schedule cache
  // entries are invalidated). Stored plans from an
  // older generation are never reused by
  // `ExecuteTransferSync`/`StartTransferAsync`.
  uint64_t plan_inputs_generation_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<std::string, TransferSessionRecord> transfers_
      ABSL_GUARDED_BY(mu_);
  std::atomic<uint64_t> next_uuid_{1000};
  std::atomic<int64_t> plan_materialization_count_{0};

  std::unique_ptr<ControlPipeClient> client_;
  std::unique_ptr<ControlPipeServer> server_;
  std::unique_ptr<AsyncControlServer> pull_server_;
  DetachedThreadGroup transfer_threads_{"RaidenControllerV3 transfers"};
};

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_CONTROLLER_V3_H_

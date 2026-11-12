import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Model
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Invariant
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Traces

/-!
# Reshard sub-plan chains (`PlanSubTransfers`, `StartChain`, `SubPlanChainProgress`)

A pool reshard whose destination has less staging than the request
(`dst_window_blocks`) runs as a chain of sub-plans, one per window. This module
models the three places that know about the chain:
1. **The planner** — `PlanSubTransfers` and `SubPlanUuid` in `coord.cc`: how
   a request's destination block lists are split by window, how shorter and
   unwindowed tags are right-aligned into the last sub-transfers, when a
   windowed tag that shares destination block IDs with another tag disables
   windowing, and how `index >= 1` sub-plans get their UUIDs.
2. **The worker** — `KVCacheManagerWithTransfer::SubPlanChainProgress`,
   `NoteSubPlanRegisteredLocked`, `RecordPoolReshardTerminalLocked`, and
   `SweepSubPlanChainsLocked` in `mgr.{h,cc}`: how one worker folds the
   terminals of a request's sub-plan sessions into the single `done_recving` /
   `done_sending` / `failed_recving` that `CompleteReadWithDetails` publishes.
3. **The coordinator** — `ReshardCoordinator::StartPoolReshard`,
   `ExecutePoolReshard`, `StartChain`, and `RunSubPlan(ctx, 0 .. K - 1)` in
   `coord.cc`, composed with the request's `RequestBlockRegistry` generation
   (`Reshard/Registry.lean`): one claim across the windows, each window's
   arm-then-dispatch over the per-window `plan.src_units`, the
   `ArmRefusedForBusyStaging` re-arm loop, and when `transfer_status_` reads
   `kCompleted` or `kFailed`.

One sub-plan on its own — its two-stage barrier, its receiver and sender
sessions — is `Reshard/Coordinator.lean`, `Reshard/Receive.lean`,
`Reshard/Send.lean`, and `Reshard/Pipeline.lean`; this module takes those as
given and models what the chain adds.

`coord.h`/`coord.cc` are `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`;
`mgr.h`/`mgr.cc` are `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`;
`planner.cc` is `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`;
`reg.cc` is `tpu_sync/kv_cache/reshard/request_block_registry.cc`;
`pool_reshard_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`; and
`reshard_service_test.cc` is
`tpu_sync/kv_cache/reshard/reshard_service_test.cc`. Exact line ranges for
every symbol, sub-block, and test are in the generated `## Source map` table
below.

## State

### Worker (`WorkerState`, one worker, one `req_id`)

| Field | C++ | Role |
|---|---|---|
| `chain` | `recv_chains_[req_id]` / `send_chains_[req_id]` (`SubPlanChainProgress`: `count`, `settled`, `failed`, `anchored`, `plan_uuids`) | `present = false` is absence from the map |
| `active` | `active_pool_reshard_recvs_` / `active_pool_reshard_sends_` entries for `req_id` | each with its `sub_plan_index()`, `sub_plan_count()`, and once `Done()` the status it settled with |
| `doneReports`, `failReports` | insertions of `req_id` into `done_recving_` / `done_sending_` and into `failed_recving_` | what `CompleteReadWithDetails` publishes |
| `doneLog` (ghost) | — | for each `done` insertion, the chain `count` and `settled` set at that moment (`DoneAfterAllSubPlans`) |
| `seenUuids` (ghost) | — | UUIDs registered so far (A2) |

### Coordinator (`CoordChainState`, one request)

| Field | C++ | Role |
|---|---|---|
| `allSrcUnits`, `winSrcUnits`, `dstUnits` | `args.src_units`; per-window `plan.src_units` (`BuildPoolReshardPlan` keeps the ranks with spans in the window); `args.dst_units` | the topology; `winSrcUnits[w] ⊆ allSrcUnits` |
| `transferStatus` | `transfer_status_[req_id]` | `notStarted`, `inProgress`, `completed`, `failed` |
| `winIdx`, `winPhase`, `armAttempts` | the running `RunSubPlan(ctx, index)`, its stage, and `attempt` of the re-arm loop | `RunSubPlan(ctx, 0)` runs inline, `1 .. K - 1` on `StartChain`'s thread |
| `maxArmAttempts` | `kChainArmAttempts` | `attempts = index == 0 ? 1 : kChainArmAttempts` |
| `armResults`, `sendResults` | `arm_status[i]` / `armed[i]`, `sender_status[i]` of the running window | an arm result is `ok`, `refusedBusy` (`ArmRefusedForBusyStaging`), or `failed` |
| `dstStates` | per destination: an earlier window's `ReshardReceiveSession` still holding its `PoolStagingLease` (`prevSessionActive`), the running window's receiver armed (`curWinArmed`) | a busy destination refuses the arm |
| `activeSenders` | ranks with a dispatched `ReshardSendSession` that have not voted `Complete` | — |
| `activeSenderExposed` (ghost) | — | latched if a rank in `activeSenders` is left without a registry claim (`NoSenderExposed`) |
| `reg` | the request's `RequestBlockRegistry` generation (`Registry.RegGen`) | shared by all windows through `ctx->claim_owner` |

## Events

| System | Event | C++ |
|---|---|---|
| worker | `registerSubPlan uuid index count` | `PoolReshardRegisterRecv` / `PoolReshardPush` seating the session and calling `NoteSubPlanRegisteredLocked` |
| worker | `settleSession uuid ok` | the session reaching `Done()` |
| worker | `poll stalled` | `CompleteReadWithDetails`: `RecordPoolReshardTerminalLocked` for each `Done()` session, then `SweepSubPlanChainsLocked(now)`; `stalled` is `now - progress.last_activity > stall` |
| coordinator | `registerProducer srcUnit` | `RequestBlockRegistry::Register` |
| coordinator | `startChainCoordination` | `StartPoolReshard` setting `kInProgress` |
| coordinator | `startWindowPlan planOk` | `RunSubPlan` Stage 0: `LookupAndClaim`, `BuildPoolReshardPlan` |
| coordinator | `armDstResult dstUnit outcome` | one Stage 1 `PoolReshardRegisterRecv` RPC returning |
| coordinator | `retryBusyArm` | the Stage 1 join setting `retry = true` and re-arming the unarmed destinations |
| coordinator | `finishWindowArm` | the Stage 1 join dispatching, or `return arm_status[i]` |
| coordinator | `sendSrcResult srcUnit ok` | one Stage 2 `PoolReshardPush` RPC returning |
| coordinator | `finishWindowSend` | the Stage 2 join: next window, final status, or `return status` |
| coordinator | `settleDispatchedRecv dstUnit` | a dispatched window's receiver settling and releasing its lease |
| coordinator | `senderVoteComplete srcUnit` | the producer calling `RequestBlockRegistry::Complete` for a rank that reported `done_sending` |
| coordinator | `cancelIfUnclaimed` | the producer calling `RequestBlockRegistry::CancelIfUnclaimed` |

## Lock scopes

| Event | Lock scopes |
|---|---|
| `registerSubPlan` | one `mu_` scope of `PoolReshardRegisterRecv` / `PoolReshardPush`: insert into `active_pool_reshard_*`, `NoteSubPlanRegisteredLocked` |
| `settleSession` | the session's own `mu_` (`Transfer/Session.lean`); the worker reads `Done()` under `mu_` in `poll` |
| `poll` | one `mu_` scope of `CompleteReadWithDetails`: the sweep of `Done()` sessions, `SweepSubPlanChainsLocked`, and the swap of `done_*_` / `failed_recving_` into the result |
| `registerProducer`, `senderVoteComplete`, `cancelIfUnclaimed` | `RequestBlockRegistry::mu_` |
| `startChainCoordination`, `finishWindowSend` (last window), `finishWindowArm` / `startWindowPlan` / `finishWindowSend` (abort) | `status_mu_` for `transfer_status_`; `RunSubPlan`'s registry calls under `RequestBlockRegistry::mu_` |
| `startWindowPlan`, `armDstResult`, `retryBusyArm`, `finishWindowArm`, `sendSrcResult` | `RunSubPlan`'s own thread joins (`armers`, `senders`); the RPCs run unlocked and report through per-index slots |
| `settleDispatchedRecv` | the receiver's `mu_` and `PoolStagingStorage`'s lock on release |

`RunSubPlan(ctx, 0)` returns before `StartChain` spawns `RunSubPlan(ctx, 1)`
(`ExecutePoolReshard`), so the windows of one request never overlap on the
coordinator; what overlaps is a dispatched window's data plane with the next
window's arming, which is what `prevSessionActive` and `retryBusyArm` model.

## Assumptions

* **A1 — `0 <= sub_plan_index < max(1, sub_plan_count)`.** `RunSubPlan` stamps
  every sub-plan with its `index` in `0 .. chain.count - 1`
  (`BuildStartTransferForTarget`). Guard: `registerSubPlan` refuses other
  indices. Without it `settled.size() >= count` could fire on indices outside
  the chain.
* **A2 — sub-plan UUIDs are fresh.** `SubPlanUuid` derives distinct UUIDs per
  index (`subPlanUuid_pos_of_pos`, `trace_sub_plan_uuid_positive_and_distinct`)
  and a retry under the same `req_id` uses a new request UUID. Guard:
  `registerSubPlan` refuses a UUID already in `seenUuids`. A repeated UUID is
  `PoolReshardRegisterRecv`'s duplicate-plan path, modelled in
  `Reshard/PoolStaging.lean`.
* **A3 — a rank votes `Complete` only after its last window.** The producer
  calls `RequestBlockRegistry::Complete` for a rank once that rank's worker
  reports `done_sending`, which `RecordPoolReshardTerminalLocked` emits only
  after every sub-plan of the chain on that worker settled. Guard:
  `senderVoteComplete` refuses a rank still in `plan.src_units` of the running
  or a later window. Without it a rank's vote could retire the claim while
  another window is being armed.
* **Not modelled:** `!stopping_.load()` in the re-arm condition (a coordinator
  shutting down aborts instead of re-arming; `retryBusyArm` assumes it is not
  stopping), the `kController` delivery mode's null-controller error path, and
  `transfer_details_`. A worker holds one `recv_chains_` and one `send_chains_`
  entry per `req_id`; `WorkerState` is one of them.
* **A4 — producers register before `Coordinate`.** `registerProducer` is
  enabled only before `startChainCoordination`. The registry refuses `Register`
  once the claim is taken (`Reshard/Registry.lean`); a registration landing
  between `StartPoolReshard` and window 0's `LookupAndClaim` is the
  single-window case of `Reshard/Coordinator.lean`.

## Properties

* `subPlanUuid_zero`, `subPlanUuid_pos_of_pos`: `SubPlanUuid(uuid, 0) = uuid`
  and `0 < SubPlanUuid(uuid, index) <= 0x7FFFFFFFFFFFFFFF` for `index > 0`.
* `planSubTransfers_no_aliased_window_split`: a multi-window chain is planned
  only when no windowed tag shares a destination block ID with another tag, so
  the per-plan `RankWaveOrder` on aliased device pages (`Reshard/Receive.lean`)
  is never split across sub-plans.
* `reachable_worker_safe` (`DoneAfterAllSubPlans`): in every reachable state of
  `WorkerState.step`, every `done_recving` / `done_sending` insertion for a
  `count`-plan request was made with `count` distinct sub-plan indices in
  `[0, count)` settled OK.
* `reachable_coordChain_safe` (`CoordChainSafe`), for every topology whose
  windows are non-empty subsets of the claimed snapshot (`ValidChainCfg`):
  `Registry.Inv`; no dispatched rank is ever left without the registry claim
  (`NoSenderExposed` — the `if (index == 0)` guard on `AbandonClaim`); no
  sender of a window is dispatched while the window arms
  (`NoPushBeforeAllArmed`); every destination is armed while a window
  dispatches (`AllArmedWhileDispatching`); a destination never has two windows'
  receivers staged at once (`OneWindowStagedPerDestination`);
  `transfer_status_` reads `kCompleted` only after the last window's Stage 2
  (`CompletedOnlyAtChainEnd`); and the claim is held from window 0's Stage 0
  to the end of the chain (`ClaimHeldAcrossWindows`).
* What the shipping code does beyond those guarantees is shown by traces
  (`Traces.lean`): a rank with spans in fewer than `K` windows never reaches
  `K` settled sub-plans and is failed by the stall sweep
  (`trace_sender_in_first_window_only`, `trace_sender_in_last_window_only`);
  the first failing sub-plan publishes `failed_recving` while sibling sessions
  are still active (`trace_first_failure_reports_while_sibling_active`); a
  sub-plan arriving after the failed entry was swept is failed a second time
  (`trace_late_sub_plan_after_sweep_reports_failure_twice`); and a window
  `>= 1` failure keeps the claim until `ttl_s_` even after every dispatched rank
  has voted (`trace_chained_arm_failure_keeps_claim_past_last_vote`).
* Mutants: counting a superseded chain's sub-plan
  (`mutant_counting_stale_sub_plan_reports_done_early`), not re-anchoring on
  sub-plan 0 (`mutant_no_reanchor_never_reports_retry_done`), and
  `AbandonClaim` at every index
  (`mutant_abandon_at_every_index_exposes_active_sender`).
* Searches (`workerViolates`, `coordViolates`, both `!decide (Safe s)`) exhaust
  a two-window worker and two two-window coordinator topologies.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `SubTransferSlice` | `:97-102` |
| `coord.h` | `SubTransferChain` | `:108-111` |
| `coord.h` | `ReshardCoordinator` | `:138-207` |
| `coord.cc` | `*` (`struct ReshardCoordinator::ChainContext {` .. `std::atomic<bool> finished{false};`) | `:310-321` |
| `coord.cc` | `SubPlanUuid` | `:356-367` |
| `coord.cc` | `PlanSubTransfers` | `:369-472` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:474-526` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:528-601` |
| `coord.cc` | `ReshardCoordinator::ReapFinishedChainsLocked` | `:603-612` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:614-649` |
| `coord.cc` | `ReshardCoordinator::WaitForSubTransferChains` | `:651-660` |
| `coord.cc` | `ArmRefusedForBusyStaging` | `:667-670` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:678-932` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`const int attempts = index == 0 ? 1 : kChainArmAttempts;` .. `if (!retry) break;`) | `:749-790` |
| `planner.cc` | `BuildPoolReshardPlan` (`PoolReshardPlan plan;` .. `plan.dst_units = request.dst_units;`) | `:1289-1295` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`struct SubPlanChainProgress {` .. `void SweepSubPlanChainsLocked(std::chrono::steady_clock::time_point now)`) | `:350-379` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked` | `:151-175` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:177-227` |
| `mgr.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:229-288` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:833-871` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:873-910` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1151-1292` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
| `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` | `:373-405` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-491` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:493-509` |
| `pool_reshard_test.cc` | `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan` | `:823-859` |
| `pool_reshard_test.cc` | `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce` | `:861-884` |
| `pool_reshard_test.cc` | `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain` | `:886-929` |
| `pool_reshard_test.cc` | `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout` | `:931-953` |
| `pool_reshard_test.cc` | `SubPlanChainTest.SinglePlanRequestsReportAtOnce` | `:955-967` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.NoWindowsPlanOneTransfer` | `:941-959` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.WindowSplitsPagedTagInDestinationOrder` | `:961-978` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.UnwindowedTagsRideInTheLastSubTransfer` | `:980-1008` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.WindowedTagSharingIdsDisablesWindows` | `:1010-1029` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.MalformedWindowsFailClosed` | `:1031-1043` |
| `reshard_service_test.cc` | `SubPlanUuidTest.DerivedUuidsArePositiveAndDistinct` | `:1045-1055` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowedRequestChainsOneSubTransferPerWindow` | `:1057-1117` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries` | `:1119-1204` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowComposesWithPrefixClip` | `:1206-1243` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedArmRetriesWhileReceiverStagingIsBusy` | `:1245-1270` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedRequestStaysInProgressUntilTheChainEnds` | `:1272-1294` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedArmFailureMarksTheTransferFailed` | `:1296-1318` |
<!-- source-map:end -->
-/

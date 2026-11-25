import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Model
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Invariant
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Traces

/-!
# Reshard sub-plan chains (`PlanSubTransfers`, `StartChain`, `SubPlanChainProgress`)

Windowed pool reshards (`dst_window_blocks`) run as a chain of sub-plans across
(1) `PlanSubTransfers` and `SubPlanUuid` (`coord.cc`), (2) worker terminal folding
in `SubPlanChainProgress`, `NoteSubPlanRegisteredLocked`,
`RecordPoolReshardTerminalLocked`, and `SweepSubPlanChainsLocked` (`mgr.{h,cc}`),
and (3) coordinator `StartPoolReshard` / `ExecutePoolReshard` / `StartChain` /
`RunSubPlan(ctx, 0 .. K - 1)` over `RequestBlockRegistry` (`Reshard/Registry.lean`);
single-sub-plan sessions and barriers live in `Reshard/{Coordinator,Receive,Send,Pipeline}.lean` (`docs/transfer/reshard.md`).
File aliases: `coord.{h,cc}` = `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`,
`planner.cc` = `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`,
`mgr.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`,
`reshard_client.cc` = `tpu_sync/kv_cache/reshard/reshard_client.cc`,
`reshard_service.h` = `tpu_sync/kv_cache/reshard/reshard_service.h`,
`pool_reshard_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`,
`reshard_service_test.cc` = `tpu_sync/kv_cache/reshard/reshard_service_test.cc`.

## State

| System | Field | C++ / Role |
|---|---|---|
| worker | `chain` | `sub_plan_chains_[req_id]` (`SubPlanChainProgress`: `sub_sessions`, `completed`, `failed`, `anchored`), shared by send/recv |
| worker | `sessions` | `sessions_` entries (`kPoolReshardSend` / `kPoolReshardReceive`) for `req_id` with `(index, count)`, `draining`, `inFlight`, `done`, `ok`, `retired` |
| worker | `doneReports`, `failReports` | `done_recving_` / `done_sending_` and `failed_recving_` insertions from `CompleteReadWithDetails` |
| worker | `doneLog`, `seenUuids` (ghost) | `(sub_sessions.size(), settled)` at each `done` (`DoneAfterAllSubPlans`); UUIDs registered so far (A2) |
| coordinator | `allSrcUnits`, `winSrcUnits`, `dstUnits` | `args.src_units`, per-window `plan.src_units` (`⊆ allSrcUnits`), `args.dst_units` |
| coordinator | `transferStatus` | `transfer_status_[req_id]` (`notStarted`, `inProgress`, `completed`, `failed`) |
| coordinator | `winIdx`, `winPhase`, `armAttempts`, `maxArmAttempts` | `RunSubPlan(ctx, index)` stage and `attempt` (`kChainArmAttempts = 12`; default `2` for search) |
| coordinator | `armResults`, `sendResults`, `dstStates` | Stage 1/2 per-target outcomes; per-destination `prevSessionActive` and `curWinArmed` |
| coordinator | `activeSenders`, `activeSenderExposed` (ghost) | dispatched ranks not yet voted `Complete`; latched if an active sender loses its claim (`NoSenderExposed`) |
| coordinator | `reg` | `RequestBlockRegistry` generation (`Registry.RegGen`) shared under `ctx->claim_owner` |

## Events

| System | Event | C++ |
|---|---|---|
| worker | `registerSubPlan uuid index count` | `PoolReshardRegisterRecv` / `PoolReshardPush` seating session + `NoteSubPlanRegisteredLocked` |
| worker | `beginOp uuid`, `latchDrain uuid ok`, `settleSession uuid ok` | entering `in_flight_ > 0`; latching drain while `in_flight_ > 0`; settling via `EndOp` / `EndRecvOpLocked` or idle `Finish` |
| worker | `poll stalled`, `pollMidSettle stalled midSettle` | `CompleteReadWithDetails`: `!IsDraining()` deadline arm, retire (`RecordPoolReshardTerminalLocked`), optional mid-poll `EndOp`s, `SweepSubPlanChainsLocked` |
| coordinator | `registerProducer srcUnit`, `startChainCoordination` | `RequestBlockRegistry::Register`; `StartPoolReshard` setting `kInProgress` |
| coordinator | `startWindowPlan planOk` | `RunSubPlan` Stage 0: `LookupAndClaim` + `BuildPoolReshardPlan` |
| coordinator | `armDstResult dstUnit outcome`, `retryBusyArm`, `finishWindowArm` | Stage 1 `PoolReshardRegisterRecv` reply; `ArmRefusedForBusyStaging` re-arm; Stage 1 join / abort |
| coordinator | `sendSrcResult srcUnit ok`, `finishWindowSend` | Stage 2 `PoolReshardPush` reply; Stage 2 join (next window, `kCompleted`, or abort) |
| coordinator | `settleDispatchedRecv dstUnit`, `senderVoteComplete srcUnit`, `cancelIfUnclaimed` | receiver lease release; producer `Complete` after rank's last window (A3); `CancelIfUnclaimed` |

## Lock scopes

| Event | Lock scopes |
|---|---|
| `registerSubPlan` | manager `mu_` on `sessions_` insert + `NoteSubPlanRegisteredLocked`; pre-insert UUID gap covered by A2 (`PoolStaging.lean`, `Send.lean`) |
| `beginOp`, `latchDrain`, `settleSession` | 1 `session->mu_` scope each; post-settle recv `FinishLocked(error)` is OK-to-error only, never adding a `done` |
| `poll`, `pollMidSettle` | 1 manager `mu_` scope in `CompleteReadWithDetails`, taking `session->mu_` separately in `IsDraining`, retire, and sweep |
| `registerProducer`, `senderVoteComplete`, `cancelIfUnclaimed` | 1 `RequestBlockRegistry::mu_` scope (`Registry.lean`) |
| `startChainCoordination`, `startWindowPlan`, `finishWindowArm`, `finishWindowSend` | `status_mu_` on `transfer_status_` writes; `RequestBlockRegistry::mu_` on `LookupAndClaim` / `AbandonClaim` |
| `armDstResult`, `retryBusyArm`, `sendSrcResult` | unlocked worker RPCs writing disjoint per-target slots, joined by `RunSubPlan` |
| `settleDispatchedRecv` | receiver `mu_` and `PoolStagingStorage` `staging_mu_` on lease release (`Receive.lean`) |

`pollMidSettle` models `EndOp` completing under `session->mu_` between retire and
sweep (A5c), with registration-order `sessions_` iteration. Windows `0 .. K - 1`
run sequentially on the coordinator; only a dispatched window's data plane
overlaps the next window's arming (`prevSessionActive`, `retryBusyArm`).

## Assumptions

`WorkerState.step` (`sys`) encodes caller guards A1, A2, A6, A7 and C++ deadline
arm A5a; `WorkerState.stepRestricted` (`sysRestricted`) adds timing guards A5b
and A5c (**not** enforced by C++; `docs/transfer/reshard.md` §"Executions the C++
permits"). `CoordChainState.step` encodes A3, A4, and A8.

* **A1 — `0 <= sub_plan_index < max(1, sub_plan_count)`.** Stamped by
  `BuildStartTransferForTarget` in `RunSubPlan`; guarded in `registerSubPlan`.
* **A2 — fresh sub-plan UUIDs (partial).** `registerSubPlan` rejects `seenUuids`.
  `SubPlanUuid` is positive (`subPlanUuid_pos_of_pos`) and live duplicates fail
  `ALREADY_EXISTS`, while retry `args.uuid` freshness, post-settle reuse, and
  concurrent `PoolReshardPush` (`Send.lean` A2; `PoolStaging.lean`) are assumed.
* **A3 — rank votes `Complete` only after its last window (caller contract).**
  `ReshardClient::CompleteRequestBlocks` is external; `senderVoteComplete` enables
  votes after a rank's last planned window (sparse-window workers report stall
  `failed_recving`, not `done_sending`: `trace_sender_in_first_window_only`).
* **A4 — producers register before `StartPoolReshard`.** Guarded in
  `registerProducer` (post-claim `Register` refused in `Registry.lean`; pre-claim
  single-window arrival in `Coordinator.lean`).
* **A5 — deadline arm (`A5a`), tombstone sweep (`A5b`), intra-poll `EndOp` (`A5c`).**
  - *A5a (`Guards.deadlineArm`, in `step`):* when `stalled`, `!IsDraining()`
    sessions past `deadline()` fail before retire/sweep (`mutant_no_deadline_arm_reports_done_after_sweep`).
  - *A5b (`Guards.noDrainingOkAtSweep`, `stepRestricted` only — **not** in C++):*
    `SweepSubPlanChainsLocked` erases `failed || completed` tombstones without
    checking `!Done()`, so an `IsDraining() && !Done()` OK sibling later emits
    `done` after failure (`trace_draining_ok_at_sweep_reports_done_after_failure`).
  - *A5c (`Guards.noSettleBetweenRetireAndSweep`, `stepRestricted` only — **not**
    in C++):* mid-poll `EndOp` (`pollMidSettle`) stalls an unanchored chain and
    clears `sub_sessions`, so the next poll hits `!anchored` before `failed` and
    emits `done` (`trace_mid_poll_endop_unanchored_reports_done_after_stall`).
    Alternative `stepSweepWaitsForOpen` closes A5b and A5c (`reachable_sweepWaitsForOpen_safe`).
* **A6 — no single-plan registration while a sub-plan is seated
  (`Guards.noSingleWhileSeated`, not in C++).** `NoteSubPlanRegisteredLocked`
  erases `sub_plan_chains_[req_id]` on `sub_plan_count <= 1`, orphaning seated
  sub-plans (`trace_single_plan_registration_erases_live_chain`).
* **A7 — one session per `sub_plan_index >= 1` slot (`Guards.oneSessionPerSlot`,
  not in C++).** Unconditional `sub_sessions[index]` overwrite on an unanchored
  sparse-sender retry orphans the first session into `!anchored` premature `done`
  (`trace_slot_overwrite_reports_stray_done`).
* **A8 — single-window exclusive destination staging.** `CoordChainState` equates
  `busy` with `prevSessionActive` (dedicated bounded `PoolStagingStorage`).
* **Not modelled:** coordinator `stopping_`; mid-chain registry TTL expiry or
  forced `Release`; overlapping pipelined stages on one worker; `kController`
  null-controller error; `done_sending_` vs. `done_recving_` split.

## Properties

* `subPlanUuid_zero`, `subPlanUuid_pos_of_pos`, and
  `planSubTransfers_no_aliased_window_split` (windowing disabled when a windowed
  tag shares destination block IDs, preserving intra-plan `RankWaveOrder`).
* `reachable_worker_safe` (`WorkerSafe` / `DoneAfterAllSubPlans` on
  `WorkerState.sysRestricted`; violated on `WorkerState.sys` in A5b/A5c and
  outside A6/A7) and `reachable_sweepWaitsForOpen_safe` (on alternative
  `WorkerState.sysSweepWaitsForOpen` without A5b/A5c).
* `reachable_coordChain_safe` (`CoordChainSafe` under `ValidChainCfg`):
  `Registry.Inv`, `NoSenderExposed`, `NoPushBeforeAllArmed`,
  `AllArmedWhileDispatching`, `OneWindowStagedPerDestination` (A8),
  `CompletedOnlyAtChainEnd`, and `ClaimHeldAcrossWindows`.
* `Traces.lean`: replays over 6 `SubPlanChainTest` + 6 windowed
  `ReshardStackTest` cases, A5b–A5c / A6–A7 traces, 4 mutants, and `#guard`
  searches.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `SubTransferSlice` | `:97-102` |
| `coord.h` | `SubTransferChain` | `:108-111` |
| `coord.cc` | `*` (`constexpr absl::Duration kWorkerRpcTimeout = absl::Seconds(60);`) | `:60` |
| `coord.cc` | `*` (`struct ReshardCoordinator::ChainContext {` .. `std::atomic<bool> finished{false};`) | `:313-324` |
| `coord.cc` | `SubPlanUuid` | `:359-370` |
| `coord.cc` | `PlanSubTransfers` | `:372-475` |
| `coord.cc` | `ReshardCoordinator::ReapFinishedChainsLocked` | `:606-615` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:617-652` |
| `coord.cc` | `ReshardCoordinator::WaitForSubTransferChains` | `:654-663` |
| `coord.cc` | `ArmRefusedForBusyStaging` | `:670-673` |
| `coord.cc` | `*` (`constexpr int kChainArmAttempts = 12;`) | `:677` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`const int attempts = index == 0 ? 1 : kChainArmAttempts;` .. `if (!retry) break;`) | `:752-793` |
| `planner.cc` | `BuildPoolReshardPlan` (`PoolReshardPlan plan;` .. `plan.dst_units = request.dst_units;`) | `:1308-1314` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`enum class SessionRole {` .. `SessionRole role;`) | `:295-307` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`struct SubPlanChainProgress {` .. `void SweepSubPlanChainsLocked(std::chrono::steady_clock::time_point now)`) | `:382-409` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked` | `:148-184` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:186-252` |
| `mgr.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:254-303` |
| `reshard_client.cc` | `ReshardClient::CompleteRequestBlocks` | `:355-360` |
| `reshard_service.h` | `*` (`double request_registry_ttl_s = 600.0;`) | `:47` |
| `pool_reshard_test.cc` | `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan` | `:921-957` |
| `pool_reshard_test.cc` | `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce` | `:959-982` |
| `pool_reshard_test.cc` | `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain` | `:984-1027` |
| `pool_reshard_test.cc` | `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout` | `:1029-1051` |
| `pool_reshard_test.cc` | `SubPlanChainTest.SinglePlanRequestsReportAtOnce` | `:1053-1065` |
| `pool_reshard_test.cc` | `SubPlanChainTest.SendTerminalWaitsForTheLastSubPlan` | `:1067-1095` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.NoWindowsPlanOneTransfer` | `:1020-1038` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.WindowSplitsPagedTagInDestinationOrder` | `:1040-1057` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.UnwindowedTagsRideInTheLastSubTransfer` | `:1059-1087` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.WindowedTagSharingIdsDisablesWindows` | `:1089-1108` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.MalformedWindowsFailClosed` | `:1110-1122` |
| `reshard_service_test.cc` | `SubPlanUuidTest.DerivedUuidsArePositiveAndDistinct` | `:1124-1134` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowedRequestChainsOneSubTransferPerWindow` | `:1136-1196` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries` | `:1198-1283` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowComposesWithPrefixClip` | `:1285-1322` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedArmRetriesWhileReceiverStagingIsBusy` | `:1324-1349` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedRequestStaysInProgressUntilTheChainEnds` | `:1351-1373` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedArmFailureMarksTheTransferFailed` | `:1375-1397` |
<!-- source-map:end -->
-/

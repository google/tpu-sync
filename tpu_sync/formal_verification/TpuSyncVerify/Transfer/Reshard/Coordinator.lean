import TpuSyncVerify.Transfer.Reshard.Coordinator.Model
import TpuSyncVerify.Transfer.Reshard.Coordinator.Lemmas
import TpuSyncVerify.Transfer.Reshard.Coordinator.Invariant
import TpuSyncVerify.Transfer.Reshard.Coordinator.Traces

/-!
# Two-stage reshard coordinator (`ReshardCoordinator::RunSubPlan`)

Formal model of one sub-transfer of `ReshardCoordinator::ExecutePoolReshard`
(`tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`) — the plan, arm and
dispatch stages of `ReshardCoordinator::RunSubPlan` — composed with
`RequestBlockRegistry` (`TpuSyncVerify.Transfer.Reshard.Registry`).

`coord.h`/`coord.cc` are `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`;
`mgr.cc` is `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`;
`reshard_recv.cc` is `tpu_sync/kv_cache/reshard_receive_session.cc`;
`reg.cc` is `tpu_sync/kv_cache/reshard/request_block_registry.cc`;
`planner.cc` is `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`;
`kv_cache_listener.cc` is `tpu_sync/kv_cache/kv_cache_listener.cc`;
`reshard_service.cc` is `tpu_sync/kv_cache/reshard/reshard_service.cc`; and
`reshard_service_test.cc` is
`tpu_sync/kv_cache/reshard/reshard_service_test.cc`. Exact line
ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

  C++ source under analysis:
  * `PoolReshardArgs` and `ReshardCoordinator` (`coord.h`)
  * `ReshardCoordinator::ExecutePoolReshard`, `ReshardCoordinator::RunSubPlan`
    and `ReshardCoordinator::StartChain` (`coord.cc`)
  * `KVCacheListener::HandleControlRequest` (`tpu_sync/kv_cache/kv_cache_listener.cc`, `PoolReshardRegisterRecv` control-pipe handler)
  * `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` (`mgr.cc`: the `pending_pool_reshard_recvs_` claim, `CreateReshardReceiveSession`, insertion into `sessions_`)
  * `CreateReshardReceiveSession`, `ReshardReceiveSession::AcquireStagingLeases`, and `ReshardReceiveSession::ReleaseStaging` / `ReshardReceiveSession::ReleaseStagingLocked` (`reshard_recv.cc`)

  ## Scope: one sub-plan of a request

  `ReshardCoordinator::ExecutePoolReshard` fetches the source and destination
  metadata once, picks the request's `uuid`, plans the request's sub-transfer
  chain (the free function `PlanSubTransfers`: one sub-plan per
  `dst_window_blocks` window of the destination block lists, `chain.count == 1`
  when no tag is windowed), draws one `claim_owner` for the whole chain
  (`claim_counter.fetch_add(1)`, so the registry sees one claim per request),
  stores all of it in a `ReshardCoordinator::ChainContext`, and runs sub-plan 0
  inline (`RunSubPlan(ctx, 0)`). When `ctx->chain.count > 1`,
  `ReshardCoordinator::StartChain` runs sub-plans `1..count-1` one after another
  on a coordinator thread (`chains_`), which records the request's final
  `transfer_status_` (`kCompleted` once every sub-plan has been dispatched,
  `kFailed` at the first failure or on shutdown); `StartPoolReshard` records
  that status itself only `if (!chain_started)`.

  Each `RunSubPlan(ctx, index)` call is one plan → arm → dispatch sequence:
  `BuildPoolReshardPlan(plan_request, registry_, ctx->claim_owner)` claims the
  source snapshot in `RequestBlockRegistry` (`LookupAndClaim`); the `armers`
  threads arm every destination (`PoolReshardRegisterRecv`) and are joined; the
  senders are dispatched (`PoolReshardPush`) and joined. This module models one
  such call with `index == 0` in which every claimed source rank emits at least
  one schedule entry (`plan.src_units = request.src_units`; when prefix clipping
  or pipelined stage filtering leaves a claimed rank with no surviving entries,
  `if (schedules.find(unit) != schedules.end())` omits it from `plan.src_units`).
  Later sub-plans of a chain (`index >= 1`, `StartChain`, sparse
  `plan.src_units`, `ArmRefusedForBusyStaging` re-arming up to
  `kChainArmAttempts` times while `!stopping_.load()`, and `if (index == 0)`
  registry-claim ownership across sub-plans) together with worker-side
  per-request chain folding (`KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked`,
  `RecordPoolReshardTerminalLocked`, `SweepSubPlanChainsLocked`) are modelled
  in `Reshard/SubPlanChain.lean`. `WorkerDelivery::Mode::kController`'s
  `delivery_.controller == nullptr` post-arm error path (which calls
  `AbandonClaim` after Phase 1 arming and returns `InternalError` before
  dispatching senders) is excluded from `Ev` alongside transport and
  serialization setup errors (matching `SubPlanChain.lean`). The sender-dispatch
  failure path is the same for every index: `for (const absl::Status& status : sender_status)`
  returns the first error without disarming receivers and without `AbandonClaim`.

  Not modelled: the registry's clock and forced release. `Ev` has no TTL-expiry
  event (`RequestBlockRegistry::PurgeExpiredLifecycleLocked` /
  `PurgeExpiredRequestBlocksLocked`, run on entry by `Register`,
  `LookupAndClaim`, `Complete`, `Release`, and the status probe;
  `CancelIfUnclaimed` runs `PurgeExpiredLifecycleLocked` only, and `AbandonClaim`
  runs neither) and no `RequestBlockRegistry::Release`
  (`COMMAND_RELEASE_REQUEST_BLOCKS`, which retires a claimed generation
  unconditionally). The claim is refreshed to `now + ttl_s_` only by the one
  `LookupAndClaim` at the start of the call, so `ClaimedWhileArming` and the
  traces below are statements about a call that finishes arming within `ttl_s_`
  (`request_registry_ttl_s`, default 600 s; one `index == 0` arm round is
  bounded by `kWorkerRpcTimeout`, 60 s) and during which nobody releases the
  generation.

  What this file proves:
  1. **Universal Protocol Guarantees (`CoordInv`, `reachable_coord_inv`)**:
     Across all reachable states:
     * `Registry.Inv s.reg` holds at every step.
     * `NoPushBeforeAllArmed` (`ReshardCoordinator::RunSubPlan`, `for (std::thread& t : armers) t.join();` before `sender_dispatch_ns`): No sender
       is dispatched (`s.sendResults = []`) and no receiver receives any push
       while `s.phase = .idle ∨ s.phase = .arming`.
     * `AllArmedInDispatching` (`ReshardCoordinator::RunSubPlan`, `for (std::thread& t : armers) t.join();` and the `if (armed[i]) continue;` scan that ends in `return arm_status[i];`): Entering
       `.dispatching` requires `s.armFailed = false` and all `dstUnits`
       reported in `s.armResults`.
     * `ClaimedWhileArming` (`ReshardCoordinator::RunSubPlan`, `BuildPoolReshardPlan(plan_request, registry_, ctx->claim_owner)` through the `armed[i]` scan): Throughout
       `.arming`, the source snapshot remains claimed in `RequestBlockRegistry`
       (`s.reg.claimedUnits = some s.srcUnits` and `s.reg.claimOwners = [s.claimOwner]`),
       preventing `CancelIfUnclaimed` and conflicting `Register` calls.
     * `LeaseMatchesSession`: Every destination unit holds a `PoolStagingLease`
       if and only if its `ReshardReceiveSession` is currently armed.
  2. **What the two abort paths leave behind (`Traces.lean`)**:
     * `trace_partial_arm_orphans_armed_receiver`
       (`ReshardCoordinator::RunSubPlan`, `return arm_status[i];`;
       `ReshardStackTest.TwoDestinationsOneArmRefusalAbandonsClaimAndSkipsSenders`
       drives it): When `dst_unit 10` succeeds in arming
       and `dst_unit 11` fails, `RunSubPlan` calls `AbandonClaim` (`if (index == 0)`) and
       aborts without disarming `dst_unit 10`, leaving `dst_unit 10`'s
       `ReshardReceiveSession` and `PoolStagingLease` orphaned until
       `timeout_s_`.
     * `trace_dispatch_failure_orphans_receivers_and_claim`
       (`ReshardCoordinator::RunSubPlan`, `for (const absl::Status& status : sender_status)`;
       `ReshardStackTest.SenderDispatchFailurePropagatesErrorAfterReceiverArm`
       is its 2×1 form, asserting the registry still reads claimed): When all `dst_units` arm and
       `src_unit 0` succeeds but `src_unit 1` fails during Stage 2 dispatch,
       `RunSubPlan` returns `sender_status` without disarming
       `dst_units` AND without calling `registry_->AbandonClaim(...)`. Both
       `dst_units` remain orphaned with active sessions/leases AND
       `claimed_[(req_id, uuid)]` remains pinned in `RequestBlockRegistry`
       until `ttl_s_`.
  3. **Searches and the disarm-and-abandon alternative (`Traces.lean`)**:
     `violatesCoordInv = !decide (CoordInv s)` finds no violation over the 2×2
     instance; `stepDisarmAndAbandon`, which also disarms the armed
     destinations and calls `AbandonClaim` on both aborts, is the one `step`
     change under which `NoOrphansWhenAborted` holds on that instance
     (`trace_disarm_and_abandon_cleans_up_both_aborts`); the shipping `step`
     violates it on both abort traces.

  ## Lock scopes

  `ReshardCoordinator::StartPoolReshard` writes `transfer_status_[args.req_id]`
  under `ReshardCoordinator::status_mu_` before `ExecutePoolReshard` and, for a
  request that did not continue as a chain (`if (!chain_started)`), again after
  it; a chained request's final status is written by `StartChain`'s thread under
  the same `status_mu_`. `ExecutePoolReshard` reads `directory_->LocalMetadata`
  under `*directory_->mu()` (shared with `RequestBlockRegistry::mu_`) on entry.
  `StartChain`, `ReapFinishedChainsLocked`, `WaitForSubTransferChains` and
  `~ReshardCoordinator` hold `ReshardCoordinator::chains_mu_` only to spawn,
  reap and join chain threads; `RunSubPlan` itself takes neither `status_mu_`
  nor `chains_mu_` (it reads the `stopping_` atomic only when deciding whether
  to re-arm). Stage-result vectors (`arm_replies` / `arm_status` / `armed`,
  `sender_status`) are synchronized across worker threads by the
  `std::thread::join` barriers, while shared registry and receiver state are
  guarded by `RequestBlockRegistry::mu_` (`Registry.lean`) and each
  destination's `KVCacheManagerWithTransfer::mu_`:

  | Event | Scopes | Between them |
  |---|---|---|
  | `registerProducer srcUnit` | 1 `RequestBlockRegistry::mu_` scope (`RequestBlockRegistry::Register`) | — |
  | `startPlan planOk` | 1 `RequestBlockRegistry::mu_` scope in `BuildPoolReshardPlan` (`LookupAndClaim`), plus a second scope (`AbandonClaim`, `if (index == 0)`) if `planOk = false` after claiming | — |
  | `armDstResult dstUnit ok` | 1 `armers` worker thread calling `PoolReshardRegisterRecv` on `dstUnit` (2 `KVCacheManagerWithTransfer::mu_` scopes, `Pipeline.lean`, plus a third acquisition at function exit by the `clear_pending` cleanup, which only erases the UUID from `pending_pool_reshard_recvs_`) and writing its own `arm_replies[i]` / `arm_status[i]` / `armed[i]` slot | worker threads write disjoint slots of `arm_replies` / `arm_status` / `armed`, joined at `finishArmBarrier` |
  | `finishArmBarrier` | `for (std::thread& t : armers) t.join();`, followed for an unarmed destination (`if (armed[i]) continue;`) by 1 `RequestBlockRegistry::mu_` scope (`registry_->AbandonClaim`, `if (index == 0)`) before `return arm_status[i];` | `join()` orders all `arm_status[i]` / `armed[i]` writes before the scan that inspects them; with `index == 0` there is a single arm attempt (`attempts = 1`), so the re-arm loop is not entered |
  | `sendSrcResult srcUnit ok` | 1 `senders` worker thread (or `SubmitTransferProgram` future) calling `PoolReshardPush` on `srcUnit` and writing `sender_status[i]` on failure; on `ok = true`, folds the sender RPC, receiver settlement (`Receive.lean`), and the producer's `registry_->Complete` vote (1 `RequestBlockRegistry::mu_` scope) | folding `Complete` into `ok = true` is sound for `CoordInv` because whenever any `srcUnit` fails (`sendFailed = true`), that unit never calls `Complete`, keeping `allVoted = false` |
  | `finishSendBarrier` | `for (std::thread& t : senders) t.join();` (or the final `futures[i].Await()`) before `for (const absl::Status& status : sender_status)` | `join()` / `Await()` orders all `sender_status[i]` writes before the status check |

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `WorkerDelivery` | `:62-67` |
| `coord.h` | `PoolReshardArgs` | `:70-92` |
| `coord.h` | `ReshardCoordinator` | `:138-207` |
| `coord.cc` | `*` (`struct ReshardCoordinator::ChainContext {` .. `std::atomic<bool> finished{false};`) | `:313-324` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:477-529` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:531-604` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`static std::atomic<uintptr_t> claim_counter` .. `reinterpret_cast<const void*>(claim_counter.fetch_add(1));`) | `:569-571` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`absl::Status first = RunSubPlan(ctx, 0);` .. `return absl::OkStatus();`) | `:595-603` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:617-652` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:681-935` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`auto plan_or = BuildPoolReshardPlan(plan_request, registry_, ctx->claim_owner);` .. `PoolReshardPlan& plan = *plan_or;`) | `:726-731` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`std::vector<absl::Status> arm_status` .. `for (std::thread& t : armers) t.join();`) | `:746-779` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`bool retry = false;` .. `return arm_status[i];`) | `:780-791` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`std::vector<absl::Status> sender_status` .. `for (std::thread& t : senders) t.join();`) | `:816-876` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`if (delivery_.controller == nullptr)`) | `:818-824` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`for (const absl::Status& status : sender_status)`) | `:879-881` |
| `kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` (`} else if (is_pool_reshard) {` .. `(*resp->mutable_receiver_pool_addrs())[pool_idx] =`) | `:138-169` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:733-775` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` |
| `reshard_recv.cc` | `CreateReshardReceiveSession` | `:541-575` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:604-643` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:645-650` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:652-655` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolH2dResultLocked` | `:796-809` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:890-926` |
| `planner.cc` | `SplitGlobalSpaceSpans` | `:267-338` |
| `planner.cc` | `ValidateSpanCoverage` | `:350-487` |
| `planner.cc` | `BuildPoolReshardPlan` (`auto registrations_or =` .. `if (!registrations_or.ok()) return registrations_or.status();`) | `:1128-1130` |
| `planner.cc` | `BuildPoolReshardPlan` (`if (unique_peers.size() != request.dst_units.size())`) | `:1155-1161` |
| `planner.cc` | `BuildPoolReshardPlan` (`PoolReshardPlan plan;` .. `plan.dst_units = request.dst_units;`) | `:1308-1314` |
| `reg.cc` | `RequestBlockRegistry::Register` | `:118-300` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-501` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:503-519` |
| `reshard_service.cc` | `ReshardService::HandleControllerCommand` (`case tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS:`) | `:293-305` |
| `reshard_service_test.cc` | `ReshardStackTest.FullPoolReshardFlowEmitsArmThenDispatch` | `:806-880` |
| `reshard_service_test.cc` | `ReshardStackTest.RegisteredDestinationPeregrineEndpointsEmittedOnScheduleEntries` | `:882-913` |
| `reshard_service_test.cc` | `ReshardStackTest.GlobalSpaceSpansSplitAtPageBoundaries` | `:915-956` |
| `reshard_service_test.cc` | `ReshardStackTest.ClipDropsTrimsAndRebasesGlobalSpans` | `:958-1014` |
| `reshard_service_test.cc` | `ReshardStackTest.ClipValidationFailsClosed` | `:1399-1432` |
| `reshard_service_test.cc` | `ReshardStackTest.ClipRejectsPageIndexedDeclarations` | `:1434-1446` |
| `reshard_service_test.cc` | `ReshardStackTest.MissingRegistrationKeepsContractString` | `:1448-1458` |
| `reshard_service_test.cc` | `ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders` | `:1460-1476` |
| `reshard_service_test.cc` | `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle` | `:1508-1572` |
| `reshard_service_test.cc` | `ReshardStackTest.CompletionVotesRetireAfterClaim` | `:1574-1597` |
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsArmEachThenDispatchSendersOnce` | `:1610-1680` |
| `reshard_service_test.cc` | `ReshardStackTest.SubsetSourceManifestsPairPoolsByTag` | `:1682-1821` |
| `reshard_service_test.cc` | `ReshardStackTest.SourceTagsAbsentOnTheDestinationAreIgnored` | `:1935-2016` |
| `reshard_service_test.cc` | `ReshardStackTest.MismatchedDestinationsFailClosed` | `:2018-2056` |
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsShardedSpansRouteBytesPerReceiver` | `:2061-2126` |
| `reshard_service_test.cc` | `ReshardStackTest.StridedGlobalSpansStayOneEntryPerDestination` | `:2128-2183` |
| `reshard_service_test.cc` | `ReshardStackTest.StridedGlobalSpansMustStayInsideOnePage` | `:2218-2228` |
| `reshard_service_test.cc` | `ReshardStackTest.ShardedSpanOrdinalOutOfRangeFailsClosed` | `:2230-2245` |
| `reshard_service_test.cc` | `ReshardStackTest.ShardedDestinationWithoutCoverageFailsClosed` | `:2247-2260` |
| `reshard_service_test.cc` | `ReshardStackTest.ShardedNonUniformExtentsFailClosed` | `:2262-2276` |
| `reshard_service_test.cc` | `ReshardStackTest.ReplicatedAndRoutedSpansCombine` | `:2289-2321` |
| `reshard_service_test.cc` | `ReshardStackTest.UnsetTransferStreamsPlansOneStreamPerSourceRank` | `:2363-2383` |
| `reshard_service_test.cc` | `ReshardStackTest.TransferStreamsEnvCapsPlanStreamsAndExpectedPushes` | `:2385-2405` |
| `reshard_service_test.cc` | `ReshardStackTest.DuplicateRegistrationRefreshesTtlWhileConflictsAndPostClaimWritesFail` | `:2443-2474` |
| `reshard_service_test.cc` | `ReshardStackTest.CompletionNoOpsAndPreClaimGuard` | `:2507-2569` |
| `reshard_service_test.cc` | `ReshardStackTest.ForcedReleaseClearsEveryLifecycleState` | `:2600-2663` |
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsOneArmRefusalAbandonsClaimAndSkipsSenders` | `:2665-2703` |
| `reshard_service_test.cc` | `ReshardStackTest.SenderDispatchFailurePropagatesErrorAfterReceiverArm` | `:2705-2741` |
| `reshard_service_test.cc` | `ReshardStackTest.FailedClaimDoesNotRefreshPartialRegistrationTtl` | `:2788-2806` |
<!-- source-map:end -->
-/

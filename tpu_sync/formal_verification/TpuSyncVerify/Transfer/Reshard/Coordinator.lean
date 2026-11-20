import TpuSyncVerify.Transfer.Reshard.Coordinator.Model
import TpuSyncVerify.Transfer.Reshard.Coordinator.Lemmas
import TpuSyncVerify.Transfer.Reshard.Coordinator.Invariant
import TpuSyncVerify.Transfer.Reshard.Coordinator.Traces

/-!
# Two-stage reshard coordinator (`ReshardCoordinator::RunSubPlan`)

Models one `index == 0` sub-plan of `ReshardCoordinator::ExecutePoolReshard`
(`ReshardCoordinator::RunSubPlan`: Stage 0 `LookupAndClaim` / `BuildPoolReshardPlan`,
Stage 1 parallel `PoolReshardRegisterRecv` arming, Stage 2 parallel
`PoolReshardPush` dispatch) composed with `RequestBlockRegistry`
(`Reshard/Registry.lean`) and per-destination receiver/lease state
(`Reshard/{Receive,PoolStaging}.lean`). Multi-window chains (`index >= 1`,
`StartChain`, sparse `plan.src_units`, `ArmRefusedForBusyStaging` retries) and
worker chain folding live in `Reshard/SubPlanChain.lean` (`docs/transfer/reshard.md`).

File aliases: `coord.{h,cc}` = `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`,
`planner.cc` = `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`,
`kv_cache_listener.cc` = `tpu_sync/kv_cache/kv_cache_listener.cc`,
`reshard_service_test.cc` = `tpu_sync/kv_cache/reshard/reshard_service_test.cc`.

## State (`CoordState`)

| Field | C++ | Role |
|---|---|---|
| `srcUnits`, `dstUnits`, `claimOwner` | `plan.src_units`, `plan.dst_units`, `ctx->claim_owner` | sub-plan topology and unique registry owner token |
| `phase` | `RunSubPlan` stage | `idle`, `arming`, `dispatching`, `succeeded`, `aborted` |
| `dstStates` | per-destination `DstUnitState` | `sessionArmed`, `stagingLeaseHeld`, `pushesReceived`, `completed` |
| `armResults`, `armFailed` | `arm_status[i]` / `armed[i]` | completed Stage 1 `PoolReshardRegisterRecv` outcomes |
| `sendResults`, `sendFailed` | `sender_status[i]` | completed Stage 2 `PoolReshardPush` outcomes |
| `reg` | `RequestBlockRegistry` generation (`Registry.RegGen`) | `(args.req_id, args.uuid)` claim and completion state |

## Events (`Ev`)

| Event | C++ |
|---|---|
| `registerProducer srcUnit` | `RequestBlockRegistry::Register` |
| `startPlan planOk` | Stage 0 `BuildPoolReshardPlan` (`LookupAndClaim`), plus `AbandonClaim` if `!planOk` |
| `armDstResult dstUnit ok` | one Stage 1 `PoolReshardRegisterRecv` RPC returning (`KVCacheListener::HandleControlRequest`) |
| `finishArmBarrier` | Stage 1 `armers` `join()` barrier: enter `.dispatching` or `AbandonClaim` + `return arm_status[i]` |
| `sendSrcResult srcUnit ok` | one Stage 2 `PoolReshardPush` RPC returning (folding receiver settlement + `Complete` vote when `ok`) |
| `finishSendBarrier` | Stage 2 `senders` `join()` / `Await()` barrier: `.succeeded` or `return status` |

## Lock scopes

| Event | Lock scopes |
|---|---|
| `registerProducer` | 1 `RequestBlockRegistry::mu_` scope (`Register`) |
| `startPlan` | `RequestBlockRegistry::mu_` in `LookupAndClaim`, plus a second scope (`AbandonClaim`) when `!planOk` |
| `armDstResult` | `armers` thread calling `PoolReshardRegisterRecv` (`PoolStaging.lean`) and writing its disjoint `armed[i]` slot |
| `finishArmBarrier` | `armers` `join()` barrier, plus 1 `RequestBlockRegistry::mu_` scope (`AbandonClaim`) on any unarmed destination |
| `sendSrcResult` | `senders` thread calling `PoolReshardPush` (`Send.lean`), folding receiver settle (`Receive.lean`) + `Complete` on `ok` |
| `finishSendBarrier` | `senders` `join()` / `Await()` barrier ordering all `sender_status[i]` writes before inspection |

`StartPoolReshard` updates `transfer_status_` under `status_mu_` (`if (!chain_started)`);
`RunSubPlan` itself takes neither `status_mu_` nor `chains_mu_`. Folding `Complete`
into `sendSrcResult _ true` is sound for `CoordInv` because any failed sender
never votes, keeping `allVoted = false`.

## Assumptions

* **A1 — Single `index == 0` sub-plan with full source set
  (`plan.src_units = request.src_units`).** Every claimed rank emits at least one
  schedule entry; sparse `plan.src_units` and `index >= 1` chains are in
  `Reshard/SubPlanChain.lean`.
* **A2 — Data plane folded into `sendSrcResult _ true`.** Session-level D2H/push/H2D
  interleavings and `PoolStagingStorage` leases are proved in
  `Reshard/{Send,Receive,Pipeline,PoolStaging}.lean`.
* **A3 — No mid-call registry TTL expiry or forced `Release`, and framed delivery.**
  Excludes `PurgeExpired*` / `Release` during arming (`request_registry_ttl_s = 600` s
  vs. 60 s `kWorkerRpcTimeout`) and the `WorkerDelivery::Mode::kController`
  null-controller post-arm error.

## Properties

* `reachable_coord_inv` (`CoordInv`): `Registry.Inv`, `NoPushBeforeAllArmed`,
  `AllArmedInDispatching`, `ClaimedWhileArming` (blocks `CancelIfUnclaimed` and
  conflicting `Register`), and `LeaseMatchesSession`.
* Abort traces vs. alternative (`Traces.lean`):
  `trace_partial_arm_orphans_armed_receiver` (armed receiver + lease orphaned
  until `timeout_s_`) and `trace_dispatch_failure_orphans_receivers_and_claim`
  (receivers + claim orphaned until `timeout_s_` / `ttl_s_`), contrasted with
  `stepDisarmAndAbandon` (`NoOrphansWhenAborted`,
  `trace_disarm_and_abandon_cleans_up_both_aborts`) and
  `mutant_dispatch_without_arm_barrier_drops_push`.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `WorkerDelivery` | `:62-67` |
| `coord.h` | `PoolReshardArgs` | `:70-92` |
| `coord.h` | `ReshardCoordinator` | `:138-207` |
| `coord.cc` | `BuildStartTransferForTarget` | `:111-274` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:477-529` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:531-604` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:681-935` |
| `kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` (`} else if (is_pool_reshard) {` .. `(*resp->mutable_receiver_pool_addrs())[pool_idx] =`) | `:138-169` |
| `planner.cc` | `SplitGlobalSpaceSpans` | `:267-338` |
| `planner.cc` | `BuildPoolReshardPlan` (`if (unique_src.size() != request.src_units.size())`) | `:660-663` |
| `planner.cc` | `BuildPoolReshardPlan` (`auto registrations_or =` .. `if (!registrations_or.ok()) return registrations_or.status();`) | `:1128-1130` |
| `planner.cc` | `BuildPoolReshardPlan` (`if (unique_peers.size() != request.dst_units.size())`) | `:1155-1161` |
| `reshard_service_test.cc` | `ReshardStackTest.FullPoolReshardFlowEmitsArmThenDispatch` | `:806-880` |
| `reshard_service_test.cc` | `ReshardStackTest.RegisteredDestinationPeregrineEndpointsEmittedOnScheduleEntries` | `:882-913` |
| `reshard_service_test.cc` | `ReshardStackTest.GlobalSpaceSpansSplitAtPageBoundaries` | `:915-956` |
| `reshard_service_test.cc` | `ReshardStackTest.ClipDropsTrimsAndRebasesGlobalSpans` | `:958-1014` |
| `reshard_service_test.cc` | `ReshardStackTest.ClipValidationFailsClosed` | `:1399-1432` |
| `reshard_service_test.cc` | `ReshardStackTest.ClipRejectsPageIndexedDeclarations` | `:1434-1446` |
| `reshard_service_test.cc` | `ReshardStackTest.MissingRegistrationKeepsContractString` | `:1448-1458` |
| `reshard_service_test.cc` | `ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders` | `:1460-1476` |
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
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsOneArmRefusalAbandonsClaimAndSkipsSenders` | `:2665-2703` |
| `reshard_service_test.cc` | `ReshardStackTest.SenderDispatchFailurePropagatesErrorAfterReceiverArm` | `:2705-2741` |
<!-- source-map:end -->
-/

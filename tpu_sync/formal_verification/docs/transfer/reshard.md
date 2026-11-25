# Pool-reshard transfer: model and results

Reference map from the pool-addressed resharding KV-cache transfer C++ to its
Lean models under `TpuSyncVerify/Transfer/Reshard/`. Each module's docstring
owns its state/event tables, lock scopes, assumptions, properties, and direct
`## Source map`; this page summarizes subsystem boundaries, cross-module
assumptions, design alternatives, permitted C++ executions, counterfactual
mutants, and the full C++ → Lean routing index. Every `trace_*` theorem that
replays a C++ test names `Suite.TestName` in its docstring (`grep -rn <TestName> TpuSyncVerify`).

File aliases: `reshard_recv.{h,cc}` = `tpu_sync/kv_cache/reshard_receive_session.{h,cc}`,
`reshard_send.{h,cc}` = `tpu_sync/kv_cache/reshard_send_session.{h,cc}`,
`slots.cc` = `tpu_sync/core/pool_reshard_send_slots.cc`,
`mgr.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`,
`base.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`,
`raw_core.h` = `tpu_sync/core/raw_transfer_core.h`,
`bt.cc` = `tpu_sync/transport/block_transport.cc`,
`bts.cc` = `tpu_sync/transport/block_transport_session.cc`,
`planner.cc` = `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`,
`reg.{h,cc}` = `tpu_sync/kv_cache/reshard/request_block_registry.{h,cc}`,
`coord.{h,cc}` = `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`,
`pool_reshard_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`,
`reshard_send_test.cc` = `tpu_sync/kv_cache/reshard_send_session_test.cc`,
`reshard_recv_test.cc` = `tpu_sync/kv_cache/reshard_receive_session_test.cc`,
`reshard_service_test.cc` = `tpu_sync/kv_cache/reshard/reshard_service_test.cc`.

## What is proved

Every theorem below holds on all reachable states of its model (carrying that
module's guarded assumptions; see
[Assumptions and the status of each](#assumptions-and-the-status-of-each)).

| Property | Theorems | Statement |
|---|---|---|
| Publication correctness and aliased-page rank-wave safety | `Pipeline.reachable_safe`, `Pipeline.reachable_unarmed_safe` (`PublicationCorrect`, `AliasedPageCorrect`), `Pipeline.attention_safe`, `Recv.RankWaveOrder` (`rankWaveOrder_iff`) | No pool is queued, issued, or completed for H2D while any strictly lower-rank pool is incomplete, so lower-rank H2D never clobbers completed higher-rank data; when `done_recving` is published, `dstHbm = good numPools` and stays invariant. Covers device-overlapping pools **of distinct `order_rank`s** under `HostLayersBounded()` (isolated per-pool host buffers; `Pipeline` A1); same-rank device overlap and shared-byte full-mirror overlap are outside the model. |
| Data correctness and post-publication safety across transfers | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe`, `Pipeline.system_unarmed_data_correct`, `Pipeline.system_unarmed_attention_safe` | Across concurrent, overlapped, and recycled transfers (`multiSys` / `multiSysUnarmed`), no transfer writes any buffer after release (`wroteReleased = false`) or reads recycled `srcHbm` / staging, and published `dstHbm = good ranks.length` persists across all subsequent steps (scoped by `MultiRequest` A1–A2). |
| Progress across transfers | `Pipeline.system_progress`, `Pipeline.system_unarmed_progress` | From any reachable multi-transfer state, any transfer can settle, release both staging leases, and publish both outcomes (`HandedOff`) in finitely many steps, enabling `.recycleSrc` and `.nextRequest`. |
| Sub-block byte-interval coverage and reconstruction | `Coverage.partition_reconstructs_block`, `Coverage.partition_reconstructs_perm`, `CoveragePipeline.step_pipe`, `CoveragePipeline.reachable_safe` | Any sub-block interval set accepted by `ValidateReceiverCoverage` forms a complete disjoint partition of `[0, extent)` and reconstructs `good extent` under every stream arrival permutation. |
| Receiver plan-push count (validator contract) | `Coverage.stream_count_matches`, `Coverage.senderStreams_pos_iff`, `Pipeline.StreamAccountingSafe` | Every accepted plan has `expected_pushes` equal to $\sum_s \min(\text{parallelism}, \lvert\text{pairs}_s\rvert)$ per pool group (`0` when `pairs_s` is empty), and `OnPoolReceived(p)` fires only after `peerCounts[p]` landings (sender wire fan-out matching `expected_pushes` is `Pipeline` A3). |
| Destination and source HBM quietness | `Pipeline.DstHbmSafe`, `Pipeline.dstHbm_quiet`, `Pipeline.SrcHbmSafe`, `Pipeline.srcHbm_quiet` | Once receive publishes no H2D is pending or unretired (`dstHbm` stays constant); once send publishes no D2H is pending or unretired (`srcHbm` may be reclaimed). |
| Bounded host-staging slot conservation, disjointness, and registration safety | `PoolStaging.no_slot_aliasing_between_distinct_uuids`, `PoolStaging.acquireLease_idempotent`, `PoolStaging.allocateMissing_slot_sum`, `PoolStaging.reachable_reserveBeforeLease_safe`, `Pipeline.StagingSafe` | Distinct active UUIDs hold disjoint `PoolStagingStorage` slots; same-UUID shared-storage acquisition is idempotent; `reserveBeforeLease` preserves active receive leases across duplicate arms; and released staging is never accessed by its former session. |
| Termination and drain-to-settle liveness | `Recv.NoOpLeak`, `Send.NoOpLeak`, `Pipeline.NoOpLeak`, `Recv.reachable_can_settle`, `Send.reachable_can_settle`, `Pipeline.reachable_can_settle`, `Pipeline.reachable_unarmed_can_settle` | Every unit of `in_flight_` has an enabled transition that advances or retires it, and every reachable state drains to `done ∧ ¬hasStaging` in finitely many steps. |
| Sender slot conservation and active-until-finalizing invariant | `Send.reachable_safe` (`ActiveOrDraining`, `Publication`), `Send.reachable_inv` (`not_draining`) | While `!finalizing_`, `remaining_pool_peer_pushes_` equals $\sum_p \text{remainingSlots}_p$, and `Start`'s caller op keeps `0 < inFlight` until `finalizing_`; `done_sending` requires all peer pushes to complete OK. |
| Plan visibility | `Recv.SettleSafe`, `Recv.NoRetiredCallback`, `Send.SettleSafe`, `Send.NoRetiredCallback`, `Send.trace_partial_failure_keeps_plan` | `plan()` is non-empty iff `!Done()`; `SettleSafe` + `NoRetiredCallback` prove no chunk/staging lookup callback remains in flight once `Done()` holds (`trace_partial_failure_keeps_plan` shows the plan retained while sibling pools drain after first push error). |
| Two-phase Arm-then-Dispatch barrier and producer block registry | `Coordinator.reachable_coord_inv`, `Coordinator.no_push_before_all_armed`, `Coordinator.claimed_while_arming`, `Pipeline.NoUnarmedPush`, `Registry.reachable_safe`, `Registry.tombstone_blocks_register_and_claim`, `Registry.claimed_blocks_register_and_cancel`, `Registry.abandonClaim_preserves_when_other_owner_present`, `Registry.complete_retires_iff_all_claimed_voted`, `Registry.purgeExpiredAll_inv` | No sender dispatches and no stream lands before all destinations arm; source blocks stay claimed in `RequestBlockRegistry` throughout arming (blocking `CancelIfUnclaimed` and conflicting `Register`); single-owner `AbandonClaim` preserves sibling owners; `Complete` retires iff all claimed units voted; and `CancelIfUnclaimed` tombstones block late registration and claims. |
| Multi-window sub-plan chains: planner, worker chain folding, coordinator chain | `SubPlanUuid` / `PlanSubTransfers` lemmas, `SubPlanChain.reachable_worker_safe`, `SubPlanChain.reachable_sweepWaitsForOpen_safe`, `SubPlanChain.reachable_coordChain_safe` | `SubPlanUuid` fixes index 0 and emits positive `splitmix64` IDs for `index > 0`; `PlanSubTransfers` disables windowing when a windowed tag shares destination block IDs (preserving `RankWaveOrder`). On `WorkerState.sysRestricted` (shipping `step` restricted to timing guards A5b/A5c, plus caller guards A1/A2/A6/A7), `done` requires all `sub_plan_count` slots settled OK (`DoneAfterAllSubPlans`, violated on `WorkerState.sys` in A5b/A5c and outside A6/A7; closed without A5b/A5c by alternative `stepSweepWaitsForOpen`). Across windows (under A3, A4, A8), `CoordChainState` preserves `NoSenderExposed` (`if (index == 0)` on `AbandonClaim`), `NoPushBeforeAllArmed`, `AllArmedWhileDispatching`, `OneWindowStagedPerDestination`, `CompletedOnlyAtChainEnd`, and `ClaimHeldAcrossWindows`. |
| Engine caller contract above `poll_stats()` (single- and multi-worker) | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.twoChannel_contract`, `Caller.earlyLoadError_fail_eq_finish`, `Caller.earlyLoadError_fail_holds_blocks`, `Caller.leakP_absorbing`, `Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing` | `connectorTwoChannel` (Stage-3 `tpu_connector.py` over `AggChannel.eagerBlock` and `KVOutputAggregator`) never frees HBM while any worker is in flight (`mwEarlyFree = false`), leaks no block, and never reads unwritten KV; early `invalid_block_ids` staging (`_stage3_fail_uncertain_loads_early` under `Policy.fail`) holds destination blocks until the receiver settles. |

## Modules

Higher modules compose lower `step` functions directly (`Pipeline` over `Send` +
`Recv`; `Coverage` over `Pipeline` via `CoveragePipeline.step_pipe`; `Coordinator`
and `SubPlanChain` over `Registry.RegGen.step`).

| Module | Scope | Main theorems |
|---|---|---|
| `Common/System.lean`, `Common/ModelCheck.lean` | Transition-system core (`System`, `run`, `Reachable`) and bounded `#guard` checker (`ModelCheck.check`) | `System.settles_of_rank` |
| `Common/ListAux.lean`, `Transfer/Session.lean` | List helpers and shared settle protocol (`in_flight_`, `draining_`, `done_`, staging lease) | `Lifecycle.Consistent` preserved by `beginOp` / `finishLocked` / `endOpLocked` |
| `Reshard/Receive.lean` | `ReshardReceiveSession` + manager deadline/retire: plan validation, rank-gated `CollectEligiblePoolH2dsLocked`, `ExecuteEligiblePoolH2ds`, `FinishPoolH2d`, `PushEndPolicy` | `Recv.reachable_safe`, `Recv.rankWaveOrder_iff`, `Recv.reachable_can_settle` |
| `Reshard/Send.lean` | `ReshardSendSession` (`CountPoolReshardSendSlots`): `Start` caller op, `ExecutePushPipeline`, `StartPoolPush`, `RecordPushCompletion`, `EndOp` | `Send.reachable_safe`, `Send.reachable_can_settle`, `Send.trace_partial_failure_keeps_plan`, `Send.trace_start_fails_fast`, `Send.trace_finish_after_last_push` |
| `Reshard/Pipeline.lean` | Single reshard transfer: `Send` + `Recv` + stream counters + five memories + aliased-page `writePoolHbm` + Arm-then-Dispatch barrier (`init` vs. `initUnarmed`) | `Pipeline.reachable_safe`, `Pipeline.reachable_unarmed_safe`, `Pipeline.attention_safe`, `Pipeline.dstHbm_quiet`, `Pipeline.srcHbm_quiet`, `Pipeline.reachable_can_settle`, `Pipeline.reachable_unarmed_can_settle` |
| `Reshard/MultiRequest.lean` | Concurrent and overlapped reshard transfers (`multiSys`, `multiSysUnarmed`), `recycleSrc` / `nextRequest`, and `wroteReleased` checks | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe`, `Pipeline.system_progress` (and `unarmed` counterparts), `Pipeline.trace_mutant_write_after_release_corrupts_all` |
| `Reshard/Coverage.lean` | Validator plan-push count, `CountPoolReshardSendSlots`, `DefaultTransferStreams`, `ValidateReceiverCoverage` interval partition, and `CoveragePipeline` refinement | `Coverage.stream_count_matches`, `Coverage.senderStreams_pos_iff`, `Coverage.partition_reconstructs_block`, `Coverage.partition_reconstructs_perm`, `CoveragePipeline.step_pipe`, `CoveragePipeline.reachable_safe` |
| `Reshard/PoolStaging.lean` | `PoolStagingStorage` bounded arena, per-`(storage, uuid)` lease idempotency & disjointness, and `RegistrationPolicy.reserveBeforeLease` vs. `.leaseBeforeReserve` | `PoolStaging.acquireLease_idempotent`, `PoolStaging.no_slot_aliasing_between_distinct_uuids`, `PoolStaging.allocateMissing_slot_sum`, `PoolStaging.reachable_reserveBeforeLease_safe`, `PoolStaging.trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot` |
| `Reshard/Registry.lean` | `RequestBlockRegistry` generation lifecycle (`Register`, multi-owner `LookupAndClaim` with row TTL refresh, `AbandonClaim`, `Complete`, `CancelIfUnclaimed`, `Release`, `purgeExpiredAll`) | `Registry.reachable_safe`, `Registry.tombstone_blocks_register_and_claim`, `Registry.claimed_blocks_register_and_cancel`, `Registry.abandonClaim_preserves_when_other_owner_present`, `Registry.complete_retires_iff_all_claimed_voted`, `Registry.purgeExpiredAll_inv`, `Registry.mutant_split_ttl_purges_request_blocks_while_claim_active` |
| `Reshard/Coordinator.lean` | Single `index == 0` `ReshardCoordinator::RunSubPlan` composed with `RequestBlockRegistry` and per-destination receiver/lease state | `Coordinator.reachable_coord_inv`, `Coordinator.no_push_before_all_armed`, `Coordinator.claimed_while_arming`, `Coordinator.trace_partial_arm_orphans_armed_receiver`, `Coordinator.trace_dispatch_failure_orphans_receivers_and_claim`, `Coordinator.trace_disarm_and_abandon_cleans_up_both_aborts` |
| `Reshard/SubPlanChain.lean` | Multi-window chains: `PlanSubTransfers` / `SubPlanUuid`, worker `sub_plan_chains_` folding (`SubPlanChainProgress`), and coordinator `StartChain` / `RunSubPlan(0 .. K-1)` with busy re-arming and sparse `plan.src_units` | `SubPlanChain.subPlanUuid_pos_of_pos`, `SubPlanChain.planSubTransfers_no_aliased_window_split`, `SubPlanChain.reachable_worker_safe`, `SubPlanChain.reachable_sweepWaitsForOpen_safe`, `SubPlanChain.reachable_coordChain_safe`, A5b/A5c/A6/A7 `trace_*` & 4 mutants |
| `Transfer/Caller.lean` | Serving engine above `poll_stats()` (`connectorTwoChannel`, `KVOutputAggregator` in `MultiWorker.lean`, `_stage3_fail_uncertain_loads_early`) | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.twoChannel_contract`, `Caller.applyRecv_eager_eq_latched`, `Caller.earlyLoadError_fail_holds_blocks`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing` |

## Boundaries

### Subsystem boundary

* **In scope:** pool-addressed KV-cache resharding — `ReshardReceiveSession`,
  `ReshardSendSession`, `CountPoolReshardSendSlots`, `DefaultTransferStreams`,
  `ValidatePlan` / `ValidateReceiverCoverage` contracts, `PoolStagingStorage`
  leases, `RequestBlockRegistry`, `PlanSubTransfers` / `SubPlanUuid`,
  `ReshardCoordinator`, and worker `SubPlanChainProgress` folding.
* **Out of scope:** symmetric prefill-to-decode transfer
  ([prefill_decode.md](prefill_decode.md)); pure planner geometry and
  transfer-program lowering (`pool_reshard_planner.cc`,
  `transfer_program_reshard.cc`); `RaidenController`, `kv_cache_store`, and
  `weight_sync/`.

### Coordinator and engine boundary (above the sessions)

* **Modelled:** `RequestBlockRegistry` and `ReshardCoordinator` single- and
  multi-window control planes (`PlanSubTransfers`, `SubPlanUuid`, `Register`,
  multi-owner `LookupAndClaim`, `CancelIfUnclaimed`, Stage 1 `PoolReshardRegisterRecv`
  with `ArmRefusedForBusyStaging` retries, Stage 2 `PoolReshardPush` over sparse
  `plan.src_units`, `Complete` votes), worker `SubPlanChainProgress` folding in
  `CompleteReadWithDetails`, and the Stage-3 engine block contract in
  `Transfer/Caller.lean` (`connectorTwoChannel`, `KVOutputAggregator`,
  `earlyLoadError_fail_holds_blocks`).
* **Not modelled:** Python submit-worker threading, topology-admission tables,
  partial-prefix load failures, and preemption/prefix caching (`Caller` B2–B3);
  cross-transfer destination page disjointness (`MultiRequest` A1) and full-mirror
  intra-transfer host staging overlap (`Pipeline` A1); mid-call/mid-chain
  registry TTL expiry or forced `Release` and post-settle same-UUID reuse
  (`Coordinator` A3, `SubPlanChain` A2); and `src_units ∩ dst_units = ∅`
  (when a unit is in both lists, `BuildStartTransferForTarget` sets
  `is_sender = true` so `KVCacheListener::HandleControlRequest` routes Phase 1
  to `PoolReshardPush` instead of `PoolReshardRegisterRecv`).

### Transport and staging boundary (below the session callbacks)

* **Modelled:** incoming-push ops (`BeginIncomingPush` / `EndIncomingPush` in
  `bts.cc`), `CompleteIncomingPush` counting plan pushes to `expected_pushes`
  before calling `OnDataReceived` once (`on_layer_received_called`), sub-block
  byte-interval coverage (`Reshard/Coverage.lean`), and `PoolStagingStorage` slot
  arenas (`PoolStaging.lean`).
* **Not modelled:** raw TCP socket I/O, connection pooling / `EndOp` idle-conn
  release, and PJRT DMA internals; sender NUMA shard-group stream fan-out
  (`AsyncPushShards` / `AccumulateLandedShards`, `Pipeline` A3); TCP wire ack
  order (`StartStream` completes before `AsyncPush(...).OnReady`, `Pipeline` A2);
  unreserved `PoolReshardPush` registration (`Send` A2); and `staging_cv_` timed
  waits in `AcquirePoolStagingLease` (`Arena.acquireLease = none` stands for
  timeout).

### Bounded host layer staging

* **Full host mirror (`!HostLayersBounded()`):** When
  `num_host_layers() == num_layers()` (default), both sessions construct
  `StagingLayerManager` with `0` slots and address
  `StaticHostLayer(pool_idx) = base_->pool(pool_idx)->storage_index` in place.
  Because `CopyPoolBlocks` uses identical intra-block offsets on host and
  device, pools with overlapping declared-live regions on one storage also share
  host staging bytes on a full mirror (outside per-pool `srcStaging` /
  `dstStaging`, `Pipeline` A1).
* **Bounded host layers (`HostLayersBounded()`):** When
  `num_host_layers() < num_layers()`, each session leases per-**pool** buffers
  through its own `StagingLayerManager` (`Transfer/LayerStaging.lean`), with
  `Shutdown()` on drain and `ReleaseAll()` at settle. Within one transfer this
  isolates overlapping pools of distinct `order_rank`s (`Pipeline.AliasedPageCorrect`);
  across concurrent transfers `MultiRequest` A1 assumes no cross-session byte
  collisions. When active pools exceed `num_host_layers()`, all `Safe` invariants
  still hold (waits fail into existing error/cancel arms), while multi-rank
  receiver progress can block until `deadline_`. (`PoolStagingStorage` bounds
  host *blocks* per storage, orthogonal to host layer buffer count.)

## Assumptions and the status of each

| Assumption | One-line statement | Status |
|---|---|---|
| Pipeline / MultiRequest engine contract | Destination reads `dstHbm` only after `done_recving`; producer reclaims `srcHbm` only after settlement. | External contract; encoded as guards on `.recycleSrc` / `.nextRequest` (`MultiRequest.lean`) and proved in the engine direction in `Transfer/Caller.lean`. |
| MultiRequest A1 — cross-transfer page and host-staging disjointness | Destination device pages (`dstHbm`) and host staging bytes of concurrent transfers are disjoint. | Device bytes: uniform-stride storages place block $b$ inside $[b \cdot S, (b+1) \cdot S)$ (`PoolSpec::Validate`), and the external engine block pool hands out distinct uncached `dst_device_block_ids` (caller layout assumed on non-uniform full-mirror fallback). Host bytes: disjoint via `PoolStaging.no_slot_aliasing_between_distinct_uuids` on bounded arenas and device offsets on full mirrors; under `HostLayersBounded()`, per-session `StagingLayerManager`s require assuming no cross-session byte collisions in shared host layer buffers. |
| Pipeline A1 / MultiRequest A2 — intra-transfer per-pool host staging disjointness and no same-rank device overlap | Each pool has an isolated host staging cell, and only strictly lower `order_rank`s clobber higher ranks on device (`clobberHigherRanks`). | Not enforced by C++. Scope limit: aliased-page theorems cover device-overlapping pools **of distinct `order_rank`s** under `HostLayersBounded()` (isolated per-pool host buffers, subject to `MultiRequest` A1 across sessions) and disjoint-host layouts on a full mirror; same-rank device overlap is unmodelled everywhere. |
| Pipeline A2 — stream lands after sender push completion | `landStream p` follows `send (.pushDone p true)`, reversing TCP's `StartStream` ack → `PostSocketPush` `OnReady` wire order. | Soundness argument in `Pipeline.lean`: `srcStaging[p]` is constant while the push op is held (`StagingSafe`), so successful runs reorder; streams landed while push is in flight or before sender ack-failure are outside model states. |
| Pipeline A3 — one stream per sender peer push | `peerCounts[p]` is both sender peer pushes and receiver landings for pool `p`. | Assumed: NUMA shard-group stream fan-out (`AsyncPushShards` / `AccumulateLandedShards`) is unmodelled; `Coverage.stream_count_matches` is the receiver validator's check only. |
| Receive A1 — at-most-once `OnPoolReceived` per declared pool | `CompleteIncomingPush` fires `OnDataReceived` once at `plan_pushes == expected_chunks` (`on_layer_received_called`), and `RecordPoolReceivedLocked` rejects duplicates. | Encoded as guards (`pool.stage == .waiting`, `landedStreams[p] = peerCounts[p]`). |
| Receive A2 — registration before visibility | `PoolReshardRegisterRecv` claims `uuid` in `pending_pool_reshard_recvs_` before `CreateReshardReceiveSession` / `AcquireStagingLeases` and seats `sessions_` only on success. | Discharged for receive registration order (`PoolStaging.reachable_reserveBeforeLease_safe`, `Coordinator.no_push_before_all_armed`); fresh UUIDs across post-poll unlocked lease cleanup are assumed. |
| Receive A3 / Send A3 — no double end; folded `StartPoolPush` check | Each push/copy op ends once; `StartPoolPush`'s two `done_ \|\| finalizing_` checks inspect only immutable fields between them. | C++ callback and control-flow discipline. |
| Receive A4 — fold of `RecordPool*Locked` with `ExecuteEligiblePoolH2ds` | External `Finish` between recording a pool and `ExecuteEligiblePoolH2ds`'s lock matches `cancel` before (or after `in_flight_ += to_launch.size()`, after) the step. | Soundness argument in `Receive.lean` (`trace_cancel_between_batch_queue_and_issue`). |
| Send A1 — positive peer counts and one endpoint per `dst_peer` | Active pools have `> 0` peers (`CountPoolReshardSendSlots`), and entries sharing `dst_peer` share `dst_peregrine_peer` (`EntryPeer`). | Positive slots enforced by `CreateReshardSendSession` / `ExecutePushPipeline` (`Send.validSchedule`); single endpoint per `dst_peer` holds for `BuildPoolReshardPlan` plans (not re-checked by `ReshardSendSession::ValidatePlan`). |
| Send A2 — `Start` before external `Finish`; one send per UUID | `PoolReshardPush` seats `sessions_[uuid]` and immediately calls `Start()` (`++in_flight_`); concurrent sends on one UUID are absent. | `Start` order from `PoolReshardPush` (`ReshardSendSessionTest.StartFailsFastWithoutTransportServer`); fresh UUID per send assumed (no pending-UUID reservation on send). |
| Send A4 — `kPerTransfer` connection-release window in `EndOp` | Dropping `mu_` for `ReleaseDataConnectionsAfterSend` at `in_flight_ == 1 && finalizing_` before `--in_flight_` / `SettleLocked()` changes no modelled state. | Soundness argument in `Send.lean`. |
| SubPlanChain A1–A2 — valid `sub_plan_index` and fresh sub-plan UUIDs | `0 <= sub_plan_index < max(1, sub_plan_count)` (A1) and each sub-plan UUID registered at most once per worker (A2). | A1 established by `RunSubPlan`. A2 partial: `SubPlanUuid` positivity proved (`subPlanUuid_pos_of_pos`) and live duplicates fail `ALREADY_EXISTS`, while caller retry `args.uuid` freshness, post-settle reuse, and concurrent `PoolReshardPush` are assumed (`seenUuids` guard). |
| SubPlanChain A5–A7 — deadline arm (A5a), no draining-OK sibling across tombstone erase (A5b), no mid-poll `EndOp` (A5c); caller guards A6 and A7 | A5a deadline-arms `!IsDraining()` sessions before sweep; A5b/A5c exclude timing windows in `CompleteReadWithDetails` / `SweepSubPlanChainsLocked`; A6 forbids `count <= 1` while `count > 1` is seated; A7 forbids overwriting a live `index >= 1` slot. | `WorkerState.Guards` in `stepG`: A5a is shipping C++ (`WorkerState.step`; mutant `mutant_no_deadline_arm_reports_done_after_sweep`). A6 and A7 are unverified caller guards (`trace_single_plan_registration_erases_live_chain`, `trace_slot_overwrite_reports_stray_done`). A5b and A5c are **not** enforced by C++ and **not** in `WorkerState.step` (violated in `trace_draining_ok_at_sweep_reports_done_after_failure` and `trace_mid_poll_endop_unanchored_reports_done_after_stall`); they guard only `WorkerState.stepRestricted` (`reachable_worker_safe`), whereas alternative `stepSweepWaitsForOpen` is safe without them (`reachable_sweepWaitsForOpen_safe`). |
| SubPlanChain A3–A4, A8 — `Complete` after last window, pre-start `Register`, exclusive single-window staging | Producer calls `Complete` only after a rank's last window (A3) and `Register` before `StartPoolReshard` (A4); `busy` iff `prevSessionActive` (A8). | A3 is an external caller contract on `ReshardClient::CompleteRequestBlocks` (sparse-window workers report stall `failed_recving`, `trace_sender_in_first_window_only`). A4 enforced post-claim by `Registry.lean` (pre-claim in `Coordinator.lean`). A8 scopes `OneWindowStagedPerDestination`. |
| Caller B1–B3 — `KVOutputAggregator` threshold, whole-transfer failure, no preemption/prefix caching | One settled outcome per worker (`ENABLE_MULTI_NUMA` unset/`0`, or `Caller/NumaBarrier.lean` for 1–2 sub-managers), whole-request failure, no block sharing. | B1 proved in `Caller/MultiWorker.lean` (`mw_contract_safe`, `_sync_stage3_tp_submit_states`) and worker chain folding in `SubPlanChain.lean`; B2–B3 scope `Transfer/Caller.lean`. |

## Shipping C++ vs. modelled alternatives

Two modules parameterize `step` by a shipping C++ guard vs. its omission
(`PushEndPolicy`, `RegistrationPolicy`). Every other module transcribes shipping
C++ with its caller assumptions as guards, except the `SubPlanChain` worker:
`WorkerState.step` (shipping C++ under caller guards A1, A2, A6, A7 and deadline
arm A5a) violates `DoneAfterAllSubPlans` in two internal timing windows (A5b,
A5c), so `reachable_worker_safe` is proved on `WorkerState.sysRestricted` (which
adds A5b and A5c). Two non-shipping alternatives show what closing the abort and
sweep gaps buys: `Coordinator.stepDisarmAndAbandon` (disarming armed receivers
and calling `AbandonClaim` on both aborts, satisfying `NoOrphansWhenAborted` on
the 2×2 search) and `SubPlanChain.WorkerState.stepSweepWaitsForOpen` (retaining
`failed || completed` tombstones while unretired `req_id` sessions remain and
checking `failed || completed` before `!anchored`, proved safe without A5b or
A5c by `reachable_sweepWaitsForOpen_safe`).

| Module | Parameter | Shipping C++ behavior | Modelled alternative |
|---|---|---|---|
| `Reshard/Receive.lean` | `PushEndPolicy` | **`PushEndPolicy.finishOnError`**: `EndIncomingPush` (`bts.cc`) calls `session->Finish(status)` before `EndOp()` on `!status.ok()`, draining and releasing staging immediately (`trace_failed_incoming_push`, `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline`). | **`PushEndPolicy.endOpOnly`** (mutant): calls `EndOp()` only, leaving a failed receiver `!IsDraining() && !Done()` with its `PoolStagingLease` pinned until `timeout_s_`. `Recv.reachable_safe` holds under both. |
| `Reshard/PoolStaging.lean` | `RegistrationPolicy` | **`RegistrationPolicy.reserveBeforeLease`**: `PoolReshardRegisterRecv` reserves `uuid` in `pending_pool_reshard_recvs_` before `CreateReshardReceiveSession` acquires `PoolStagingStorage` leases (`reachable_reserveBeforeLease_safe`). | **`RegistrationPolicy.leaseBeforeReserve`** (mutant): leases before reserving `uuid`; a rejected duplicate's `~ReshardReceiveSession()` calls `ReleasePoolStagingLeases(uuid)` and frees the survivor's slots (`trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`). |

### Executions the C++ permits

Each trace below is a valid execution of shipping C++ (not a mutant): the first
five run the module's shipping `step` (`WorkerState.sysRestricted` refuses the
A5b and A5c traces), and the last two run `WorkerState.step` with one unverified
caller guard (A6 or A7) turned off.

| Trace | What the C++ does | Outcome |
|---|---|---|
| `Coordinator.trace_partial_arm_orphans_armed_receiver` | On Stage 1 arm failure `RunSubPlan` calls `AbandonClaim` (`if (index == 0)`) and returns without disarming already-armed destinations | Armed destination retains its `ReshardReceiveSession` and `PoolStagingLease` until `timeout_s_` (cleaned up by alternative `Coordinator.stepDisarmAndAbandon`) |
| `Coordinator.trace_dispatch_failure_orphans_receivers_and_claim` | On Stage 2 sender failure `RunSubPlan` returns without disarming receivers or calling `AbandonClaim` | Armed destinations hold leases until `timeout_s_` and the registry claim blocks `CancelIfUnclaimed` until `ttl_s_` (cleaned up by `Coordinator.stepDisarmAndAbandon`) |
| `Registry.trace_single_stage_votes_retire_multi_owner_claim` | `Complete` keys `completed_units_` by `(req_id, uuid)` rather than `claim_owner`, retiring once every `claimed_units_` rank votes | Stage 1's `Complete` votes retire the generation, erasing sibling stage 2's claim and source registrations |
| `SubPlanChain.trace_draining_ok_at_sweep_reports_done_after_failure` | A5b window (`WorkerState.step`): `SweepSubPlanChainsLocked` erases `failed \|\| completed` tombstones without checking `!Done()`, leaving an `IsDraining() && !Done()` OK sibling open | `DoneAfterAllSubPlans` violated: sibling settles into `end()` and emits `done` after `failed_recving` (prevented by `WorkerState.stepSweepWaitsForOpen`) |
| `SubPlanChain.trace_mid_poll_endop_unanchored_reports_done_after_stall` | A5c window (`WorkerState.step`): `EndOp` sets `done_` between `RetireSettledSessionLocked` and `SweepSubPlanChainsLocked` (`session->mu_` gap) | `DoneAfterAllSubPlans` violated: unanchored chain stalls and clears `sub_sessions`, then hits `!anchored` before `failed` to emit `done` after `failed_recving` (prevented by `stepSweepWaitsForOpen`) |
| `SubPlanChain.trace_single_plan_registration_erases_live_chain` | Outside caller guard A6: `NoteSubPlanRegisteredLocked` erases `sub_plan_chains_[req_id]` on `sub_plan_count <= 1` while chained sessions are seated | `DoneAfterAllSubPlans` violated: orphaned sub-plan settles directly as the request's `done` |
| `SubPlanChain.trace_slot_overwrite_reports_stray_done` | Outside caller guard A7: `NoteSubPlanRegisteredLocked` overwrites `sub_sessions[sub_plan_index]` without checking for an unretired occupant | `DoneAfterAllSubPlans` violated: overwritten session on an unanchored chain hits `!anchored` stray arm and emits `done` with one slot settled |

## Counterfactual mutants

Each mutant removes one C++ check/ordering (or caller contract where noted) — or,
for `Coverage.mutant_*`, mutates the plan — and is killed by an executable trace
or `#guard` search in `TpuSyncVerify/Transfer/Reshard/`:

| Mutant | Guard / invariant removed | Property violated |
|---|---|---|
| `Recv.onPoolReceivedIgnoringRank` | `CollectEligiblePoolH2dsLocked` lower-`rank` completion check | `RankWaveOrder` (higher-rank H2D launches before lower-rank H2D finishes) |
| `Recv.h2dIssueLeak` | `ExecuteEligiblePoolH2ds` `EndRecvOpLocked()` cleanup on `draining_` early exit | `NoOpLeak` (`in_flight_ > 0` forever after cancellation) |
| `Send.initOvercountedSlots` | `CountPoolReshardSendSlots` counting active `(pool, peer)` pairs instead of `pools × peers` | `ActiveOrDraining` (`in_flight_ == 0` with `!finalizing_` after all pushes) |
| `Send.stepCancelRecordsStatusFirst` (`trace_finish_after_last_push_mutant`) | `ReshardSendSession::Finish` checking `done_ \|\| finalizing_` before recording `status_` | `published = some true` when `Finish(Cancelled)` races after last OK push |
| `Send.startFailWithoutRecord` | `ReshardSendSession::Start` calling `RecordPushCompletion(status)` on fast failure | `ActiveOrDraining` (`in_flight_ == 0` with `!finalizing_`) |
| `Send.d2hDoneLeakOnDraining` | `ExecutePushPipeline` `pool_future.OnReady` calling `EndOp()` on `done_ \|\| finalizing_` exit | `NoOpLeak` (leaked pool op prevents settlement) |
| `Pipeline.mutant_ignoring_rank_corrupts_aliased_page` | Rank-ordered H2D wave gate on shared device pages (`CollectEligiblePoolH2dsLocked`) | `AliasedPageCorrect` / `PublicationCorrect` (rank 0 H2D clobbers rank 1 to `.junk`) |
| `Pipeline.mutant_onPoolReceived_before_all_streams` | `CompleteIncomingPush` waiting for `plan_pushes == expected_chunks` before `OnDataReceived` | `StreamAccountingSafe` / `PublicationCorrect` (H2D copies incomplete staging) |
| `Pipeline.mutant_reclaim_before_publish_corrupts_dstHbm` | Caller contract: reclaiming `srcHbm` only after send publication (`SrcHbmSafe`) | `PublicationCorrect` (early reclaim copies `.junk` into `dstHbm`) |
| `Pipeline.trace_mutant_write_after_release_corrupts_all` | `ReshardSendSession::SettleLocked` releasing `PoolStagingLease` only at `in_flight_ == 0` | `wroteReleased = false` (`d2hDone` writes after `srcStaging` release) |
| `Coverage.mutant_gap_in_coverage_leaves_unwritten_byte` | `ValidateReceiverCoverage` rejecting coverage gaps | Sub-block reconstruction (`.blank` bytes remain in `byteStaging`) |
| `Coverage.mutant_overlap_in_coverage_rejected` | `ValidateReceiverCoverage` rejecting overlapping intervals | Disjoint interval partition (`validBlockCoverage = false`) |
| `Coverage.mutant_mismatched_expected_pushes_rejected` | `ValidateReceiverCoverage` checking `expected_pushes` against $\sum_s \min(\text{parallelism}, \lvert\text{pairs}_s\rvert)$ | `stream_count_matches` (`validateExpectedPushes = false`) |
| `PoolStaging.mutant_non_idempotent_acquire_exhausts_shared_storage` | `AcquirePoolStagingLease` skipping blocks already leased by `uuid` | Shared-storage lease admission (duplicate allocation exhausts arena) |
| `PoolStaging.mutant_leaking_release_breaks_conservation_and_starves` | `ReleasePoolStagingLeases` returning all leased slots to `free_slots` | Slot conservation (`totalSlots = numSlots`) |
| `Registry.mutant_premature_complete_retires_while_peer_active` | `RequestBlockRegistry::Complete` checking `all_voted` across `claimed_units_` | Multi-sender claim safety (first vote retires peer's active blocks) |
| `Registry.mutant_unconditional_abandon_exposes_sibling_owner_to_cancel` | `RequestBlockRegistry::AbandonClaim` retaining `claimed_` until `claimed_owners_` is empty | Multi-owner claim safety (one stage abandoning exposes sibling to `CancelIfUnclaimed`) |
| `Registry.mutant_split_ttl_purges_request_blocks_while_claim_active` | `LookupAndClaim` refreshing `registration.expires_at` on claimed `request_blocks_` rows | `ClaimOwnersConsistent` (rows expire while `claimed_` stays active, breaking sibling claims) |
| `Coordinator.mutant_dispatch_without_arm_barrier_drops_push` | `RunSubPlan` joining Stage 1 armers and verifying `armed[i]` before Stage 2 | `NoPushBeforeAllArmed` (push to unarmed `dst_unit 11` is dropped) |
| `SubPlanChain.stepCountStale` (`mutant_counting_stale_sub_plan_reports_done_early`) | `RecordPoolReshardTerminalLocked` ignoring settled sessions absent from anchored `sub_sessions` | `DoneAfterAllSubPlans` (superseded sub-plan triggers early `done` on retry) |
| `SubPlanChain.stepNoReanchor` (`mutant_no_reanchor_never_reports_retry_done`) | `NoteSubPlanRegisteredLocked` re-anchoring on `sub_plan_index == 0` | Same-`req_id` retry liveness (retry trapped in failed entry) |
| `SubPlanChain.mutant_no_deadline_arm_reports_done_after_sweep` | `CompleteReadWithDetails` `!IsDraining()` deadline arm before `SweepSubPlanChainsLocked` (A5a) | `DoneAfterAllSubPlans` (un-failed open sibling settles into `end()` as `done`) |
| `SubPlanChain.stepAbandonAtEveryIndex` (`mutant_abandon_at_every_index_exposes_active_sender`) | `RunSubPlan` guarding `AbandonClaim` with `if (index == 0)` | `NoSenderExposed` (`index >= 1` abort drops claim while window-0 sender is active) |

## C++ → Lean routing index

Per C++ symbol: owning Lean module, modelled events/definitions, and primary
theorems/mutants (`Lines` generated by `tools/resolve_anchors.py`).

### Receive session (`reshard_recv.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `tpu_sync/transport/transfer_session.h` | `TransferSession` | `:36-115` | `Transfer/Session.lean` | Common session interface (`Start`, `Done`, `Finish`, `TryBeginOp` / `EndOp`, `IsDraining`, `ResolveStagingSlot`, `OnDataReceived`, `plan()`, `deadline()`) | `Lifecycle.Consistent` |
| `reshard_recv.cc` | `ReshardReceiveSession` | `:68-243` | `Transfer/Session.lean`, `Reshard/Receive.lean` | Receive session state (`in_flight_`, `draining_`, `done_`, `staging_released_`, `pool_order_ranks_`, `transfer_plan_`, `TryBeginOp`, `~ReshardReceiveSession`) | `Recv.reachable_safe` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidatePlan` | `:257-334` | `Reshard/Receive.lean` | `Recv.validPlan`, `Recv.validPlanPoolsAndBlocks` | `Recv.trace_reshard_receive_session_test_parity` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` | `:336-539` | `Reshard/Coverage.lean` | `Coverage.validBlockCoverage`, `Coverage.validateExpectedPushes` | `Coverage.partition_reconstructs_perm`, `Coverage.stream_count_matches` |
| `reshard_recv.cc` | `CreateReshardReceiveSession` | `:541-575` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean` | Factory (`ValidatePlan`, HBM gates, `TransferPlan::Create`, `AcquireStagingLeases` with rollback on error) under `pending_pool_reshard_recvs_` | `PoolStaging.reachable_reserveBeforeLease_safe` |
| `reshard_recv.cc` | `ReshardReceiveSession::StaticHostLayer` | `:178-186` | `Reshard/Pipeline.lean` | Full-mirror host layer (`pool->storage_index`, `Pipeline` A1) | `Pipeline.AliasedPageCorrect` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingSlot` | `:671-702` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | `StaticHostLayer(pool_idx)` when `!HostLayersBounded()`, else per-pool `StagingLayerManager` lease (`Pipeline` A1) | `Pipeline.AliasedPageCorrect` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:604-643` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean` | Per-storage bounded lease acquisition (`AcquirePoolStagingLease`) | `PoolStaging.acquireLease_idempotent` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishLocked` | `:657-669` | `Transfer/Session.lean`, `Reshard/Receive.lean` | `Lifecycle.finishLocked`, `Recv.Ev.cancel` | `Recv.SettleSafe` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndRecvOpLocked` | `:735-745` | `Transfer/Session.lean`, `Reshard/Receive.lean` | `Lifecycle.endOpLocked` | `Recv.StagingIntegrity` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndOp` | `:747-750` | `Transfer/Session.lean`, `Reshard/Receive.lean` | `Recv.Ev.pushEnd` (`EndIncomingPush`: `Finish(status)` on error, then `EndOp`) | `Recv.SettleSafe`, `Recv.trace_failed_incoming_push` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolReceivedLocked` | `:752-763` | `Reshard/Receive.lean` | Rejects undeclared or duplicate pool indices (`Receive` A1) | `Recv.reachable_safe` |
| `reshard_recv.cc` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` | `:765-794` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | Rank gate + batch `in_flight_ += to_launch.size()` | `Recv.RankWaveOrder`, `Pipeline.AliasedPageCorrect`, mutants `Recv.onPoolReceivedIgnoringRank`, `Pipeline.mutant_ignoring_rank_corrupts_aliased_page` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnPoolReceived` | `:811-822` | `Reshard/Receive.lean` | `Recv.Ev.onPoolReceived` (delivered by `CompleteIncomingPush` via `OnDataReceived`) | `Recv.RankWaveOrder` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` | `:824-888` | `Reshard/Receive.lean` | `Recv.Ev.h2dIssue` (`draining_` early exit calls `EndRecvOpLocked()`) | `Recv.NoOpLeak`, mutant `Recv.h2dIssueLeak` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:890-926` | `Reshard/Receive.lean` | `Recv.Ev.h2dReady`, `Recv.Ev.h2dDone` (last pool calls `Finish`) | `Recv.reachable_safe` (`Publication`), `Recv.reachable_can_settle` |
<!-- source-map:end -->

### Send session (`reshard_send.{h,cc}`, `slots.cc`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `reshard_send.cc` | `ReshardSendSession` | `:72-230` | `Transfer/Session.lean`, `Reshard/Send.lean` | Send session state (`remaining_pool_peer_pushes_`, `finalizing_`, `in_flight_`, `done_`, `transfer_plan_`, `TryBeginOp() == false`) | `Send.reachable_safe` |
| `slots.cc` | `CountPoolReshardSendSlots` | `:27-51` | `Reshard/Send.lean`, `Reshard/Coverage.lean` | Deduplicated `(pool, peer)` slot count (`Coverage.countPoolReshardSendSlots`) | `Send.ActiveOrDraining`, `Coverage.trace_pool_reshard_send_slots_test_parity`, mutant `Send.initOvercountedSlots` |
| `reshard_send.cc` | `CreateReshardSendSession` | `:279-341` | `Reshard/Send.lean` | Factory (`ValidatePlan`, HBM gate, `CountPoolReshardSendSlots > 0`, `InitTransportServer`) | `Send.reachable_safe` |
| `reshard_send.cc` | `ReshardSendSession::Start` | `:437-474` | `Reshard/Send.lean` | Caller op (`++in_flight_`), `ExecutePushPipeline` thread, and fast-fail path (`Send.Ev.startFail`) | `Send.ActiveOrDraining`, `Send.trace_start_fails_fast`, mutant `Send.startFailWithoutRecord` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePushPipeline` | `:476-590` | `Reshard/Send.lean` | `Send.Ev.d2hIssue` per pool (after slot + arena lease), `Send.Ev.execFinish` | `Send.ActiveOrDraining` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:592-697` | `Reshard/Send.lean` | `Send.Ev.d2hDone` | `Send.NoOpLeak`, mutant `Send.d2hDoneLeakOnDraining` |
| `reshard_send.cc` | `ReshardSendSession::RecordPushCompletion` | `:699-718` | `Reshard/Send.lean` | `Send.Ev.pushDone`, `Send.recordPushCompletion` (latches `finalizing_` on first error or last push) | `Send.Publication`, `Send.ActiveOrDraining` |
| `reshard_send.cc` | `ReshardSendSession::Finish` | `:720-738` | `Reshard/Send.lean` | `Send.Ev.cancel` (`if (done_ \|\| finalizing_) return;` before recording `status_`) | `Send.SettlesPromptly`, `Send.trace_finish_after_last_push`, mutant `Send.stepCancelRecordsStatusFirst` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:754-782` | `Reshard/Send.lean` | `kPerTransfer` idle-conn release (`Send` A4), then `--in_flight_` and `SettleLocked()` on last op | `Send.SettleSafe`, `Send.trace_cancel_settles_on_last_endOp` |
<!-- source-map:end -->

### Manager, pool staging, and copy futures (`mgr.{h,cc}`, `base.{h,cc}`, `raw_core.h`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `mgr.h` | `KVCacheManagerWithTransfer` (`struct SubPlanChainProgress {` .. `void SweepSubPlanChainsLocked(std::chrono::steady_clock::time_point now)`) | `:382-409` | `Reshard/SubPlanChain.lean` | `SubPlanChain.ChainProgress` (`sub_plan_chains_[req_id]`: `sub_sessions`, `completed`, `failed`, `anchored`, `last_activity`) | `SubPlanChain.reachable_worker_safe` (`WorkerState.sysRestricted`) |
| `mgr.cc` | `KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked` | `:148-184` | `Reshard/SubPlanChain.lean` | `WorkerEv.registerSubPlan`, `noteSubPlanRegistered` (`count <= 1` erases entry; `index == 0` re-anchors; unguarded under A6/A7) | `SubPlanChain.reachable_worker_safe`, `trace_single_plan_registration_erases_live_chain`, `trace_slot_overwrite_reports_stray_done`, mutant `stepNoReanchor` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:186-252` | `Reshard/SubPlanChain.lean` | `WorkerEv.poll`, `recordTerminal`, `recordTerminalWaitOpen` (folds `sub_sessions` into single `done` / `failed_recving`) | `SubPlanChain.reachable_worker_safe`, `reachable_sweepWaitsForOpen_safe`, `trace_mid_poll_endop_unanchored_reports_done_after_stall`, mutant `stepCountStale` |
| `mgr.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:254-303` | `Reshard/SubPlanChain.lean` | `WorkerEv.poll` / `pollMidSettle` (`stalled = true`), `sweepStalledChain`, `sweepStalledChainWaitOpen` | `SubPlanChain.trace_draining_ok_at_sweep_reports_done_after_failure`, `trace_mid_poll_endop_unanchored_reports_done_after_stall`, `reachable_worker_safe`, `reachable_sweepWaitsForOpen_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` | `:305-309` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | `hooks.get_session` wiring `BeginIncomingPush` / `EndIncomingPush` to active sessions | `Recv.trace_failed_incoming_push` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean` | `sessions_.find(uuid)` under `mu_` | — |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` | `Reshard/Receive.lean`, `Reshard/Send.lean`, `Reshard/SubPlanChain.lean` | Folds `Done()` `kPoolReshard*` entries via `RecordPoolReshardTerminalLocked` (`publish`) and queues post-poll lease release | `SubPlanChain.reachable_worker_safe`, `PoolStaging.RegEv.settleSession` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:733-775` | `Reshard/Send.lean`, `Reshard/SubPlanChain.lean` | Checks/retires incumbent, constructs send session unlocked (no pending claim, `Send` A2), seats `SessionEntry`, and calls `Start()` | `Send.reachable_safe`, `SubPlanChain.reachable_worker_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean`, `Reshard/SubPlanChain.lean` | Reserves `pending_pool_reshard_recvs_` (`beginArm`), acquires leases (`acquireArmLease`), and seats `SessionEntry` + `NoteSubPlanRegisteredLocked` (`finishArm`) | `PoolStaging.reachable_reserveBeforeLease_safe`, `trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`, `SubPlanChain.reachable_worker_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:828-874` | `Reshard/Receive.lean`, `Reshard/Send.lean` | Coordinator disarm calling `Finish(Cancelled)` on a live session | `Recv.Ev.cancel`, `Send.Ev.cancel` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1076-1139` | `Reshard/Receive.lean`, `Reshard/Send.lean`, `Reshard/Pipeline.lean`, `Reshard/MultiRequest.lean`, `Reshard/SubPlanChain.lean`, `Transfer/Caller.lean` | `!IsDraining()` deadline arm, `RetireSettledSessionLocked`, `SweepSubPlanChainsLocked`, and unlocked `ForgetPushProgress` / `ReleasePoolStagingLeases` | `Pipeline.PublicationCorrect`, `Pipeline.system_data_correct`, `SubPlanChain.reachable_worker_safe`, `Caller.contract_safe` |
| `base.h` | `KVCacheManagerBase` | `:120-1117` | `Reshard/PoolStaging.lean` | `PoolStagingStorage` and aliased hybrid-model pool declarations | `PoolStaging.ArenaInv` |
| `base.h` | `KVCacheManagerBase::AttachPlaceholderDeviceHoldForTest` | `:767` | `Reshard/Receive.lean`, `Reshard/Send.lean` | Placeholder `buffer_holds_` entry for CPU validation/peer tests | — |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` | `:2572-2629` | `Reshard/PoolStaging.lean` | `PoolStaging.Arena.acquireLease` | `PoolStaging.acquireLease_idempotent`, `PoolStaging.no_slot_aliasing_between_distinct_uuids`, mutant `mutant_non_idempotent_acquire_exhausts_shared_storage` |
| `base.cc` | `KVCacheManagerBase::ReleasePoolStagingLeases` | `:2631-2647` | `Reshard/PoolStaging.lean` | `PoolStaging.Arena.releaseLease` | `PoolStaging.allocateMissing_slot_sum`, mutant `mutant_leaking_release_breaks_conservation_and_starves` |
| `base.cc` | `KVCacheManagerBase::CopyPoolBlocks` | `:2917-3058` | `Reshard/Receive.lean`, `Reshard/Send.lean`, `Reshard/Pipeline.lean` | Pool-extent D2H/H2D dispatch (`D2hPoolBlocks`, `H2dPoolBlocks`) over declared-live regions (`Pipeline` A1) | `Recv.NoOpLeak`, `Send.NoOpLeak`, `Pipeline.AliasedPageCorrect` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnDataReceived` | `:128-135` | `Reshard/Receive.lean` | Routes pool completions (`minor_dimension_indices` empty) to `OnPoolReceived` | `Recv.RankWaveOrder` |
| `base.cc` | `KVCacheManagerBase::GetSession` | `:3081-3087` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean` | Transport delegate lookup (`transfer_hooks_.get_session`) | — |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` | `Reshard/Receive.lean`, `Reshard/Send.lean` | Resolves `host_layer_idx` via `GetSession(uuid)` → `session->ResolveStagingSlot` | — |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` | `Reshard/Receive.lean`, `Reshard/Send.lean` | Host buffer span lookup per chunk for `ValidateChunks` | — |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` | `:3399-3590` | `Reshard/PoolStaging.lean`, `Reshard/Send.lean` | `PoolStaging.Arena.resolveBlockSlot`, `ResolveStagingSlot`, and sender sub-block chunk resolution | `PoolStaging.no_slot_aliasing_between_distinct_uuids`, `Send.trace_partial_failure_keeps_plan` |
| `raw_core.h` | `PjRtCopyFuture::OnReady` | `:514-568` | `Reshard/Receive.lean`, `Reshard/Send.lean` | Runs callback inline with `OkStatus()` when `!future.IsValid()` and `event_bundles.empty()` | — |
| `raw_core.h` | `IssueD2hShard` | `:657-685` | `Reshard/Send.lean` | Returns ready no-op `PjRtCopyFuture` on null `RaidenBufferHandle` (`DeviceBackedTestBase`) | — |
| `raw_core.h` | `IssueH2dShard` | `:687-715` | `Reshard/Receive.lean` | Returns ready no-op `PjRtCopyFuture` on null `RaidenBufferHandle` (`DeviceBackedTestBase`) | — |
| `reshard_send_test.cc` | `FakeD2hBase` | `:341-427` | `Reshard/Send.lean` | CPU test base with promise-backed `D2hPoolBlocks` (`HoldNextCopy` / `CompleteCopy` / `FillOnCopy`) driving `Start` → `StartPoolPush` to `StubRecvSession` or `RecordingH2dReceiverBase` | — |
| `pool_reshard_test.cc` | `TestManager::FinishPoolReshardRecvPool` | `:127-137` | `Reshard/Receive.lean` | CPU helper arming `FakeH2dBase`, resolving staging, and calling `OnDataReceived(pool_idx, {}, {})` | — |
| `pool_reshard_test.cc` | `FakeH2dBase` | `:75-109` | `Reshard/Receive.lean` | CPU test base answering `H2dPoolBlocks` with a ready future carrying the armed status | — |
| `pool_reshard_test.cc` | `DeviceBackedTestBase` | `:692-705` | `Reshard/Receive.lean`, `Reshard/Send.lean` | Populates a default-constructed `RaidenBufferHandle` (`BoundedPoolTestManager`) | — |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline` | `:862-900` | `Reshard/Receive.lean` | Replayed by `trace_failed_incoming_push` (`PushEndPolicy.finishOnError`) | `Recv.trace_failed_incoming_push` |
<!-- source-map:end -->

### Transport and planner (`bt.cc`, `bts.cc`, `planner.cc`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` | `Reshard/Pipeline.lean` | Delegates incoming stream to `StartStream` and deletes push session once `IsDone()` | `Pipeline.StagingSafe` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | `Recv.Ev.pushBegin`: `GetSession(uuid)` + `TryBeginOp()` | `Recv.StagingIntegrity`, `Pipeline.StagingSafe` |
| `bts.cc` | `EndIncomingPush` | `:258-271` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | `Recv.Ev.pushEnd`: `Finish(status)` on `!status.ok()`, then `EndOp()` (`PushEndPolicy.finishOnError`) | `Recv.trace_failed_incoming_push`, `Pipeline.StagingSafe` |
| `bts.cc` | `ValidateChunks` | `:275-310` | `Reshard/Receive.lean`, `Reshard/Send.lean`, `Reshard/Coverage.lean` | Per-chunk host span validation (`GetBlockArrayHostSpan` → `ResolveStagingSlot`) | — |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` | `Reshard/Pipeline.lean` | `BeginIncomingPush` / `EndIncomingPush` around stream read, writing 1-byte ack afterwards (`Pipeline` A2) | `Pipeline.StagingSafe` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` | `Reshard/Pipeline.lean`, `Reshard/Coverage.lean` | Counts landed plan pushes to `expected_pushes` and fires `OnDataReceived` once (`on_layer_received_called`; `Pipeline.Ev.landStream`) | `Pipeline.StreamAccountingSafe`, mutant `Pipeline.mutant_onPoolReceived_before_all_streams` |
| `bts.cc` | `BlockTransportServerPushSession::AccumulateLandedShards` | `:715-722` | `Reshard/Pipeline.lean`, `Reshard/Coverage.lean` | Folds NUMA shard groups into plan pushes; not modelled (`Pipeline` A3) | — |
| `bt.cc` | `BlockTransport::AsyncPushShards` | `:571-644` | `Reshard/Coverage.lean`, `Reshard/Send.lean`, `Reshard/Pipeline.lean` | Clamps `min(parallelism, num_blocks)` streams per NUMA shard group (fan-out unmodelled, `Pipeline` A3) | — |
| `tpu_sync/transport/lib/adapter/socket_transport_adapter.cc` | `SocketTransportAdapter::PostSocketPush` (`if (remaining_workers->fetch_sub(1) == 1) {`) | `:474-495` | `Reshard/Pipeline.lean` | Fires `AsyncPush(...).OnReady` (`pushDone`) once all `P` stream tasks complete (`Pipeline` A2) | `Pipeline.StagingSafe` |
| `tpu_sync/transport/lib/adapter/socket_transport_adapter.cc` | `SocketTransportAdapter::PostSocketPushInternal` (`ReadExact(fd, &ack, 1, config_.final_ack_read_timeout_ms);`) | `:706` | `Reshard/Pipeline.lean` | Reads receiver's 1-byte ack before completing stream (`Pipeline` A2) | `Pipeline.StagingSafe` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` | `:755-873` | `Reshard/Coverage.lean`, `Reshard/Send.lean` | Sub-block request construction, `GetBlockChunks`, and `ValidateChunks` | `Coverage.partition_reconstructs_perm` |
| `planner.cc` | `DefaultTransferStreams` | `:620-637` | `Reshard/Coverage.lean` | `Coverage.defaultTransferStreams`, `Coverage.resolvePlanParallelism` | `Coverage.trace_default_transfer_streams_test_parity` |
| `planner.cc` | `BuildPoolReshardPlan` | `:639-1359` | `Reshard/Coverage.lean` | `Coverage.senderStreams`, `Coverage.recomputedExpectedPushes` | `Coverage.stream_count_matches`, `Coverage.senderStreams_pos_iff` |
<!-- source-map:end -->

### Registry and coordinator (`reg.{h,cc}`, `coord.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `reg.h` | `RequestBlockRegistry` | `:45-151` | `Reshard/Registry.lean` | `RegGen` (`request_blocks_`, `claimed_`, `claimed_owners_`, `completed_units_`, `cancelled_`) | `Registry.reachable_safe` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredRequestBlocksLocked` | `:61-69` | `Reshard/Registry.lean` | `RegGen.purgeExpiredRequestBlocks` | mutant `Registry.mutant_split_ttl_purges_request_blocks_while_claim_active` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredLifecycleLocked` | `:71-96` | `Reshard/Registry.lean` | `RegGen.purgeExpiredAll` | `Registry.purgeExpiredAll_inv` |
| `reg.cc` | `RequestBlockRegistry::RetireLocked` | `:98-116` | `Reshard/Registry.lean` | Erases `request_blocks_` and claim maps on completion | `Registry.complete_retires_iff_all_claimed_voted` |
| `reg.cc` | `RequestBlockRegistry::Register` | `:118-300` | `Reshard/Registry.lean` | `Ev.register` | `Registry.tombstone_blocks_register_and_claim`, `Registry.claimed_blocks_register_and_cancel` |
| `reg.cc` | `RequestBlockRegistry::Release` | `:302-315` | `Reshard/Registry.lean` | `Ev.release` (forced release; excluded from `Coordinator` / `SubPlanChain` mid-call steps) | `Registry.reachable_safe` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` | `Reshard/Registry.lean` | `Ev.complete` | `Registry.complete_retires_iff_all_claimed_voted`, `trace_single_stage_votes_retire_multi_owner_claim`, mutant `mutant_premature_complete_retires_while_peer_active` |
| `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` | `:373-405` | `Reshard/Registry.lean` | `Ev.cancelIfUnclaimed` | `Registry.tombstone_blocks_register_and_claim` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-501` | `Reshard/Registry.lean` | `Ev.lookupAndClaim` (refreshes claimed row TTLs alongside `claimed_`) | `Registry.claimed_blocks_register_and_cancel`, `trace_claim_refreshes_registration_ttl_while_failed_claim_does_not`, mutant `mutant_split_ttl_purges_request_blocks_while_claim_active` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:503-519` | `Reshard/Registry.lean` | `Ev.abandonClaim` | `Registry.abandonClaim_preserves_when_other_owner_present`, mutant `mutant_unconditional_abandon_exposes_sibling_owner_to_cancel` |
| `reg.cc` | `RequestBlockRegistry::InvalidateUnitLocked` | `:521-529` | `Reshard/Registry.lean` | Worker re-registration erasing `request_blocks_` rows while `claimed_` stays; not modelled (`Registry.lean` A3) | — |
| `tpu_sync/kv_cache/reshard/reshard_client.cc` | `ReshardClient::CompleteRequestBlocks` | `:355-360` | `Reshard/SubPlanChain.lean` | Sender of `COMMAND_COMPLETE_REQUEST_BLOCKS` (`ChainEv.senderVoteComplete`, `SubPlanChain` A3) | — |
| `coord.h` | `SubTransferChain` | `:108-111` | `Reshard/SubPlanChain.lean` | `SubPlanChain.SubTransferSlice`, `SubPlanChain.SubTransferChain` | `SubPlanChain.planSubTransfers_no_aliased_window_split` |
| `coord.h` | `ReshardCoordinator` | `:138-207` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean` | `CoordState`, `CoordInv`, `CoordChainState`, `CoordChainSafe` | `Coordinator.reachable_coord_inv`, `SubPlanChain.reachable_coordChain_safe` |
| `tpu_sync/kv_cache/kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` | `:103-228` | `Reshard/Coordinator.lean` | Framed listener routing `COMMAND_START_TRANSFER` to `PoolReshardPush` (`is_sender()`) or `PoolReshardRegisterRecv` | — |
| `coord.cc` | `BuildStartTransferForTarget` | `:111-274` | `Reshard/Coordinator.lean` | Target request builder (`is_sender` when unit in `src_units`) | — |
| `coord.cc` | `SubPlanUuid` | `:359-370` | `Reshard/SubPlanChain.lean` | `SubPlanChain.subPlanUuid` (63-bit positive `splitmix64` for `index > 0`) | `SubPlanChain.subPlanUuid_zero`, `SubPlanChain.subPlanUuid_pos_of_pos` |
| `coord.cc` | `PlanSubTransfers` | `:372-475` | `Reshard/SubPlanChain.lean` | `SubPlanChain.planSubTransfers`, `hasWindowedTagSharingIds` | `SubPlanChain.planSubTransfers_no_aliased_window_split` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:477-529` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean` | `ChainEv.startChainCoordination` (`transfer_status_` writes `if (!chain_started)`) | `SubPlanChain.reachable_coordChain_safe` (`CompletedOnlyAtChainEnd`) |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:531-604` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean` | Draws `claim_owner`, builds `ChainContext`, runs `RunSubPlan(ctx, 0)`, spawns `StartChain` | `Coordinator.reachable_coord_inv`, `SubPlanChain.reachable_coordChain_safe` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:617-652` | `Reshard/SubPlanChain.lean` | Multi-window continuation (`RunSubPlan(ctx, 1 .. count - 1)`) and final `transfer_status_` | `SubPlanChain.reachable_coordChain_safe` (`ClaimHeldAcrossWindows`), `trace_chained_arm_failure_keeps_claim_past_last_vote` |
| `coord.cc` | `ArmRefusedForBusyStaging` | `:670-673` | `Reshard/SubPlanChain.lean` | `ChainEv.armDstResult _ .refusedBusy`, `ChainEv.retryBusyArm` (`kChainArmAttempts` loop) | `SubPlanChain.reachable_coordChain_safe` (`OneWindowStagedPerDestination`), `trace_chained_arm_retries_while_receiver_staging_busy` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:681-935` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean`, `Reshard/Pipeline.lean` | Stage 0 `LookupAndClaim`, Stage 1 arming & busy retries, Stage 2 dispatch, `if (index == 0)` `AbandonClaim` | `Coordinator.no_push_before_all_armed`, `Coordinator.claimed_while_arming`, `SubPlanChain.reachable_coordChain_safe`, mutant `stepAbandonAtEveryIndex` |
<!-- source-map:end -->

*External reshard caller pair (`Transfer/Caller.lean`):*
vllm-torchtpu `src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py`,
vLLM `vllm/distributed/kv_transfer/kv_connector/utils.py`, and
vLLM `vllm/v1/core/sched/scheduler.py`
(`TPURaidenConnectorWorker.get_finished`, `_poll_finished`,
`_apply_stage3_submit_outcome`, `_stage3_fail_uncertain_loads_early`,
`get_block_ids_with_load_errors`, `_sync_stage3_tp_submit_states`,
`get_finished_count`; `KVOutputAggregator.aggregate`;
scheduler `_handle_invalid_blocks`, `_update_from_kv_xfer_finished`,
`finish_requests`, `_update_waiting_for_remote_kv`, `kv_load_failure_policy`) →
`Caller.contract_safe`, `Caller.mw_contract_safe`,
`Caller.applyRecv_eager_eq_latched`, `Caller.applyRecv_staggered_eq_batch`,
`Caller.twoChannel_contract`, `Caller.earlyLoadError_fail_holds_blocks`,
`Caller.mw_earlyLoadError_fail_holds_blocks`,
`Caller.trace_twoChannel_early_uncertain_load_error`, `Caller.trace_mw_staggered_*`.

## Where the rest lives

| Content | File |
|---|---|
| `trace_*` witnesses, replayed tests, and `#guard` searches | `grep -rn "theorem trace_\|#guard" TpuSyncVerify/Transfer/Reshard` |
| Modelling conventions (`step`, ghost fields, replay, mutants) | [conventions.md](../conventions.md) |

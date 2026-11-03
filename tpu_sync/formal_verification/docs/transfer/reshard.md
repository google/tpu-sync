# Pool-reshard transfer: model and results

The reference map from the pool-addressed resharding KV-cache transfer C++ to
its Lean models under `TpuSyncVerify/Transfer/Reshard/`.

Each module's docstring owns its state table, event table, assumptions,
properties and `## Source map`; this page routes between them. It says what is
proved and where, lists the modules, draws the three boundaries of the model,
tabulates the assumptions and how each is discharged, sets the shipping code
beside the design alternatives modelled next to it, lists the mutants, and ends
with the per-symbol C++ → Lean index. Each `trace_*` theorem's docstring names
the C++ test it replays, so `grep -rn <TestName> TpuSyncVerify` answers "is
test X modelled".

File aliases: `reshard_recv.{h,cc}` = `tpu_sync/kv_cache/reshard_receive_session.{h,cc}`,
`reshard_send.{h,cc}` = `tpu_sync/kv_cache/reshard_send_session.{h,cc}`,
`slots.cc` = `tpu_sync/core/pool_reshard_send_slots.cc`,
`mgr.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`,
`base.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`,
`bt.cc` = `tpu_sync/transport/block_transport.cc`,
`bts.cc` = `tpu_sync/transport/block_transport_session.cc`,
`planner.cc` = `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`,
`reg.{h,cc}` = `tpu_sync/kv_cache/reshard/request_block_registry.{h,cc}`,
`coord.{h,cc}` = `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`,
`pool_reshard_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.

## What is proved

Every theorem below is proved on every reachable state of its model; the
statement column is the one-line form, the module docstring has the exact one.

| Property | Theorems | Statement |
|---|---|---|
| Publication correctness and aliased-page rank-wave safety | `Pipeline.reachable_safe`, `Pipeline.reachable_unarmed_safe` (`PublicationCorrect`, `AliasedPageCorrect`), `Pipeline.attention_safe`, `Recv.RankWaveOrder` (`rankWaveOrder_iff`) | No pool is queued, issued, or completed for H2D while any strictly lower-rank pool is not yet `completed`, so lower-rank H2D waves never clobber completed higher-rank pool bytes on shared hybrid-model device pages; when `poll_stats()` publishes `done_recving`, `dstHbm = good numPools` and stays invariant afterwards. |
| Data correctness and post-publication safety across transfers | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe`, `Pipeline.system_unarmed_data_correct`, `Pipeline.system_unarmed_attention_safe` | In any reachable state of `multiSys` or `multiSysUnarmed` (concurrent, overlapped and recycled reshard transfers), no transfer ever writes any of its four buffers after releasing them (`wroteReleased = false`) or reads from recycled `srcHbm`, `srcStaging`, or `dstStaging`; whenever a transfer publishes `done_recving`, its `dstHbm` holds `good ranks.length` and stays equal to `good ranks.length` across every subsequent transition of any transfer. |
| Progress across transfers | `Pipeline.system_progress`, `Pipeline.system_unarmed_progress` | From any reachable multi-transfer state (`multiSys` or `multiSysUnarmed`), any transfer can settle, release both staging leases to `PoolStagingStorage`, and publish both outcomes (`HandedOff`) in finitely many steps, enabling both `.recycleSrc` and `.nextRequest`. |
| Sub-block byte-interval coverage and reconstruction | `Coverage.partition_reconstructs_block`, `Coverage.partition_reconstructs_perm`, `CoveragePipeline.step_pipe`, `CoveragePipeline.reachable_safe` | Any sub-block interval set accepted by `ValidateReceiverCoverage` forms a complete disjoint partition of `[0, extent)` and reconstructs `good extent` under every stream arrival permutation. |
| Multi-sender stream count conservation | `Coverage.stream_count_matches`, `Coverage.senderStreams_pos_iff`, `Pipeline.StreamAccountingSafe` | For every pool group, the receiver's `expected_pushes` validated by `ValidateReceiverCoverage` equals $\sum_s \min(\text{parallelism}, |\text{pairs}_s|)$; a sender with zero scheduled pairs opens zero streams; and `OnPoolReceived(p)` fires only after all `expected_pushes` streams for pool `p` have landed. |
| Destination and source HBM quietness | `Pipeline.DstHbmSafe`, `Pipeline.dstHbm_quiet`, `Pipeline.SrcHbmSafe`, `Pipeline.srcHbm_quiet` | Once the receive is published no H2D copy is pending or unretired (`dstHbm` never changes afterwards); once the send is published no D2H copy is pending or unretired (`srcHbm` may be reclaimed). |
| Bounded host-staging slot conservation, disjointness, and registration safety | `PoolStaging.no_slot_aliasing_between_distinct_uuids`, `PoolStaging.acquireLease_idempotent`, `PoolStaging.allocateMissing_slot_sum`, `PoolStaging.reachable_reserveBeforeLease_safe`, `Pipeline.StagingSafe` | In `PoolStagingStorage`, distinct active UUIDs hold disjoint physical slot lists and never resolve any block IDs to the same staging slot; re-acquiring a shared storage for the same `uuid` is idempotent; under `reserveBeforeLease`, every active session retains its staging lease until it settles; and released staging is never read or written by the session that held it. |
| Termination and drain-to-settle liveness | `Recv.NoOpLeak`, `Send.NoOpLeak`, `Pipeline.NoOpLeak`, `Recv.reachable_can_settle`, `Send.reachable_can_settle`, `Pipeline.reachable_can_settle`, `Pipeline.reachable_unarmed_can_settle` | Every unit of `in_flight_` has an enabled transition that advances or retires it, and every reachable state drains to `done ∧ ¬hasStaging` in finitely many steps. |
| Sender slot conservation and active-until-finalizing invariant | `Send.reachable_safe` (`ActiveOrDraining`, `Publication`) | While `!finalizing_`, `remaining_pool_peer_pushes_` equals $\sum_p \text{remainingSlots}_p$, and `ExecutePush`'s caller op keeps `0 < inFlight` until `finalizing_` is latched; `done_sending` requires every active pool to have completed all peer pushes. |
| Plan unregistration and poll-finish equivalence | `Recv.PlanIntegrity`, `Recv.reachable_pollReady_disabled`, `Send.PlanIntegrity`, `Send.trace_early_unregister_on_error` | On the receiver, `(planUnregistered → done) ∧ (published ≠ none → planUnregistered)` and `IsReadyToComplete() && !IsDraining()` is disabled in every reachable state because `FinishPoolH2d` sets `network_completed_` and calls `Finish` in the same step; on the sender, `done → planUnregistered` and `planUnregistered → draining`, and `RecordPushCompletion` unregisters the plan on the first failed push while sibling pools still have ops in flight (`Send/Traces.lean` §"The `EndOp`-only unregistration alternative" shows the `step` change under which `planUnregistered = done`). |
| Two-phase Arm-then-Dispatch barrier and producer block registry | `Coordinator.reachable_coord_inv`, `Coordinator.no_push_before_all_armed`, `Coordinator.claimed_while_arming`, `Pipeline.NoUnarmedPush`, `Registry.reachable_safe`, `Registry.tombstone_blocks_register_and_claim`, `Registry.claimed_blocks_register_and_cancel`, `Registry.abandonClaim_preserves_when_other_owner_present`, `Registry.complete_retires_iff_all_claimed_voted`, `Registry.purgeExpiredAll_inv` | No sender is dispatched and no stream lands until every destination unit has armed its `ReshardReceiveSession`; producer blocks remain claimed in `RequestBlockRegistry` throughout arming, blocking `CancelIfUnclaimed` and duplicate `Register`; abandoning one owner's claim preserves sibling owners' claims; `Complete` retires a generation iff every claimed source unit has voted; and cancelling an unclaimed generation leaves a tombstone blocking late registration and claims. |
| Multi-window sub-plan chains: planner, worker chain folding, coordinator chain | `SubPlanChain.subPlanUuid_zero`, `SubPlanChain.subPlanUuid_pos_of_pos`, `SubPlanChain.planSubTransfers_no_aliased_window_split`, `SubPlanChain.reachable_worker_safe`, `SubPlanChain.reachable_coordChain_safe` | `SubPlanUuid` fixes index 0 and emits strictly positive `splitmix64` IDs for `index > 0`; `PlanSubTransfers` never splits a request whose windowed tag shares destination block IDs with another tag (preserving intra-plan `order_rank` H2D wave ordering); on each worker, `done_recving` / `done_sending` for a `count`-plan request is inserted only with `count` distinct sub-plan indices settled OK (`DoneAfterAllSubPlans`); across the chain the coordinator never leaves a dispatched rank without the registry claim (`NoSenderExposed`, the `if (index == 0)` guard), dispatches no sender while a window arms, has every destination armed while it dispatches, never stages two windows on one destination, publishes `kCompleted` only after the last window, and holds the claim from window 0 to the end of the chain. |
| Engine caller contract above `poll_stats()` (single- and multi-worker) | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.twoChannel_contract`, `Caller.earlyLoadError_fail_eq_finish`, `Caller.earlyLoadError_fail_holds_blocks`, `Caller.leakP_absorbing`, `Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing` | A connector that forwards every publication (`connectorTwoChannel`, matching `vllm-torchtpu`'s Stage-3 `tpu_connector.py` over `AggChannel.eagerBlock`) never frees HBM while any worker on that node is still in flight (`mwEarlyFree = false`), leaks no held block on either node, and never lets decode read unwritten KV, both for a single worker and across workers aggregated by `KVOutputAggregator`; staging `invalid_block_ids` early on an uncertain post-arm RPC error under `Policy.fail` defers the block free until the native receiver settles (`dHeld = true`). |

## Modules

Higher modules compose the lower `step` functions directly: `Pipeline` composes
`Send.step` and `Recv.step` with per-pool stream counters, five pool-indexed
memories (`srcHbm`, `srcStaging`, `wire`, `dstStaging`, `dstHbm`), aliased-page
overwrite (`writePoolHbm`), and the coordinator's Arm-then-Dispatch barrier;
`Coverage` refines `Pipeline`'s scalar stream counter into sub-block byte
intervals via a lockstep forward simulation (`CoveragePipeline.step_pipe`);
`Coordinator` composes `Registry.step` with multi-destination receiver arming
and `PoolStagingLease` lifecycles.

| Module | Scope | Main theorems |
|---|---|---|
| `Common/ListAux.lean`, `Transfer/Session.lean` | List indexing helpers and the shared settle protocol (`in_flight_`, `draining_`, `done_`, staging lease) | `Lifecycle.Consistent` preserved by `beginOp` / `finishLocked` / `endOpLocked` |
| `Reshard/Receive.lean` | One `ReshardReceiveSession` + manager poll/unregister hooks: static plan/pool validation (`validPlan`, `validPlanPoolsAndBlocks`), rank-gated `CollectEligiblePoolH2dsLocked`, batch `in_flight_ += to_launch.size()`, `ExecuteEligiblePoolH2ds` (`draining_` early exit), `FinishPoolH2d`, `TakePendingUnregister`, `PushEndPolicy` (`finishOnError` vs. `endOpOnly`) | `Recv.reachable_safe` (`SettleSafe`, `NoRetiredCallback`, `StagingIntegrity`, `PlanIntegrity`, `SettlesPromptly`, `RankWaveOrder`, `ReadinessSound`, `Publication`, `NoOpLeak`), `Recv.rankWaveOrder_iff`, `Recv.reachable_pollReady_disabled`, `Recv.reachable_can_settle` |
| `Reshard/Send.lean` | One `ReshardSendSession` (`CountPoolReshardSendSlots`): per-pool D2H (`ExecutePush`), per-peer `StartPoolPush`, `remaining_pool_peer_pushes_` conservation, plan unregistration in `RecordPushCompletion` and `EndOp` | `Send.reachable_safe` (`SettleSafe`, `NoRetiredCallback`, `StagingIntegrity`, `SettlesPromptly`, `ActiveOrDraining`, `PlanIntegrity`, `Publication`, `NoOpLeak`), `Send.reachable_can_settle`, `Send.trace_early_unregister_on_error` |
| `Reshard/Pipeline.lean` | Single reshard transfer: `Send` + `Recv` + per-pool `BlockTransport` stream counters + five pool-indexed memories + aliased-page `writePoolHbm` (`clobberHigherRanks`) + Arm-then-Dispatch barrier (`init` vs. `initUnarmed`) | `Pipeline.reachable_safe`, `Pipeline.reachable_unarmed_safe` (`PublicationCorrect`, `AliasedPageCorrect`, `DstHbmSafe`, `SrcHbmSafe`, `StagingSafe`, `StreamAccountingSafe`, `NoUnarmedPush`, `NoOpLeak`), `Pipeline.attention_safe`, `Pipeline.dstHbm_quiet`, `Pipeline.srcHbm_quiet`, `Pipeline.reachable_can_settle`, `Pipeline.reachable_unarmed_can_settle` |
| `Reshard/MultiRequest.lean` | `multiSys` and `multiSysUnarmed`: concurrent and overlapped reshard transfers (`recycleSrc`, `nextRequest`), per-buffer write-after-release checks (`wroteReleased`), and post-publication HBM safety across transfers | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe`, `Pipeline.system_progress`, `Pipeline.system_unarmed_data_correct`, `Pipeline.system_unarmed_attention_safe`, `Pipeline.system_unarmed_progress`, `Pipeline.trace_multi_request`, `Pipeline.trace_overlapped_requests`, `Pipeline.trace_unarmed_multi_request`, `Pipeline.trace_mutant_write_after_release_corrupts_all` |
| `Reshard/Coverage.lean` | Multi-sender stream count conservation (`Σ_s min(parallelism, \|pairs_s\|) = expected_pushes`), the sender's slot count (`CountPoolReshardSendSlots`), `DefaultTransferStreams` (`TPU_RAIDEN_TRANSFER_STREAMS`), `ValidateReceiverCoverage` interval-partition completeness, and byte-level refinement `CoveragePipeline` over `Pipeline` | `Coverage.stream_count_matches`, `Coverage.senderStreams_pos_iff`, `Coverage.partition_reconstructs_block`, `Coverage.partition_reconstructs_perm`, `CoveragePipeline.step_pipe`, `CoveragePipeline.reachable_safe`, `Coverage.trace_pool_reshard_send_slots_test_parity`, `Coverage.trace_reshard_stack_routing_test_parity` |
| `Reshard/PoolStaging.lean` | `PoolStagingStorage` bounded slot arena, per-`(storage, uuid)` lease idempotency & cross-UUID disjointness, and receive registration order (`RegistrationPolicy.reserveBeforeLease` vs. the `.leaseBeforeReserve` mutant) | `PoolStaging.acquireLease_idempotent`, `PoolStaging.no_slot_aliasing_between_distinct_uuids`, `PoolStaging.allocateMissing_slot_sum`, `PoolStaging.reachable_reserveBeforeLease_safe`, `PoolStaging.trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`, `PoolStaging.trace_pool_staging_recv_registration_tests` |
| `Reshard/Registry.lean` | `RequestBlockRegistry` generation lifecycle: per-unit `Register`, multi-owner `LookupAndClaim`, `AbandonClaim`, full-set `Complete` voting, `CancelIfUnclaimed` tombstones, and TTL expiration (`purgeExpiredAll` vs. split `purgeExpiredRequestBlocks`) | `Registry.reachable_safe`, `Registry.tombstone_blocks_register_and_claim`, `Registry.claimed_blocks_register_and_cancel`, `Registry.abandonClaim_preserves_when_other_owner_present`, `Registry.complete_retires_iff_all_claimed_voted`, `Registry.purgeExpiredAll_inv`, `Registry.trace_single_stage_votes_retire_multi_owner_claim`, `Registry.trace_split_ttl_purges_request_blocks_while_claim_active` |
| `Reshard/Coordinator.lean` | One sub-plan (`ReshardCoordinator::RunSubPlan`) of `ReshardCoordinator::ExecutePoolReshard` composed with `RequestBlockRegistry` and per-destination receiver sessions | `Coordinator.reachable_coord_inv`, `Coordinator.no_push_before_all_armed`, `Coordinator.claimed_while_arming`, `Coordinator.trace_partial_arm_orphans_armed_receiver`, `Coordinator.trace_dispatch_failure_orphans_receivers_and_claim`, `Coordinator.trace_disarm_and_abandon_cleans_up_both_aborts` |
| `Reshard/SubPlanChain.lean` | Multi-window chains: `PlanSubTransfers` and `SubPlanUuid` (`splitmix64`), the worker's `SubPlanChainProgress` folding (`NoteSubPlanRegisteredLocked`, `RecordPoolReshardTerminalLocked`, `SweepSubPlanChainsLocked`), and `StartPoolReshard` / `StartChain` / `RunSubPlan(0 .. K-1)` over `RequestBlockRegistry` with the `ArmRefusedForBusyStaging` re-arm loop and sparse per-window `plan.src_units` | `SubPlanChain.subPlanUuid_zero`, `SubPlanChain.subPlanUuid_pos_of_pos`, `SubPlanChain.planSubTransfers_no_aliased_window_split`, `SubPlanChain.reachable_worker_safe`, `SubPlanChain.reachable_coordChain_safe`, `SubPlanChain.trace_sender_in_first_window_only`, `SubPlanChain.trace_first_failure_reports_while_sibling_active`, `SubPlanChain.trace_chained_arm_failure_keeps_claim_past_last_vote`, `SubPlanChain.mutant_abandon_at_every_index_exposes_active_sender` |
| `Transfer/Caller.lean` | Serving engine above `poll_stats()` shared with `PrefillDecode` (`vllm-torchtpu` `tpu_connector.py` and `tpu-inference` `tpu_raiden_connector.py` on vLLM): `connectorTwoChannel` models Stage-3 reshard outcome forwarding, `MultiWorker.lean` models `KVOutputAggregator` across worker ranks (`AggChannel.eagerBlock` and `AggChannel.latchedReq`), and `earlyLoadError_fail_holds_blocks` models `_stage3_fail_uncertain_loads_early` | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.twoChannel_contract`, `Caller.applyRecv_eager_eq_latched`, `Caller.earlyLoadError_fail_eq_finish`, `Caller.earlyLoadError_fail_holds_blocks`, `Caller.leakP_absorbing`, `Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing`, `Caller.trace_mw_staggered_*` |

## Boundaries

The model sits between three boundaries. Each side names the module that owns
it; the arguments are in those modules.

### Subsystem boundary

In scope is pool-addressed KV-cache resharding: `ReshardReceiveSession`,
`ReshardSendSession`, `CountPoolReshardSendSlots`, `DefaultTransferStreams`,
the output contracts of `ValidatePlan` / `ValidateReceiverCoverage`,
`PoolStagingStorage` bounded leases, `RequestBlockRegistry`, and
`ReshardCoordinator` (`PlanSubTransfers`, `SubPlanUuid`, `ExecutePoolReshard`,
`StartChain`, `RunSubPlan`) together with worker-side `SubPlanChainProgress`
terminal folding (`NoteSubPlanRegisteredLocked`,
`RecordPoolReshardTerminalLocked`, `SweepSubPlanChainsLocked`).

Out of scope: prefill-to-decode ("symmetric") transfer — see
[prefill_decode.md](prefill_decode.md); pure planner geometry and transfer-program
lowering (`tpu_sync/kv_cache/reshard/pool_reshard_planner.cc` and
`tpu_sync/core/transfer_program_reshard.cc`, which compile topology geometry
into `StartTransferRequest` plans validated by `ValidatePlan` /
`ValidateReceiverCoverage`); `RaidenController`, `kv_cache_store`, and
`weight_sync/` model-weight resharding.

### Coordinator and engine boundary (above the sessions)

Modelled: the `RequestBlockRegistry` and `ReshardCoordinator` two-stage and
multi-window protocols — per-tag `dst_window_blocks` chain planning
(`PlanSubTransfers`, `SubPlanUuid`), producer block registration (`Register`),
multi-owner `LookupAndClaim`, `CancelIfUnclaimed` tombstones, Stage 1
destination arming (`PoolReshardRegisterRecv`, including
`ArmRefusedForBusyStaging` retries across `StartChain` windows), Stage 2 sender
dispatch (`PoolReshardPush`, including sparse per-window `plan.src_units`),
per-unit `Complete` voting, worker `SubPlanChainProgress` terminal folding
(`NoteSubPlanRegisteredLocked`, `RecordPoolReshardTerminalLocked`,
`SweepSubPlanChainsLocked`) and `CompleteReadWithDetails` (`poll_stats()`)
outcome publication (`Reshard/SubPlanChain.lean`), and the connector/scheduler
block lifecycle above `poll_stats()` (`Transfer/Caller.lean`: when the engine
may recycle is `.recycleSrc` / `.nextRequest` in `Reshard/MultiRequest.lean`;
what the engine owes a publication is `Caller.Contract`, satisfied by
`connectorTwoChannel` for `vllm-torchtpu`'s Stage-3 connector both for a single
worker and across workers aggregated by `KVOutputAggregator` in
`Caller/MultiWorker.lean` (`AggChannel.eagerBlock`), including its early
`invalid_block_ids` path `_stage3_fail_uncertain_loads_early` via
`earlyLoadError_fail_holds_blocks` and `mw_earlyLoadError_fail_holds_blocks`).

Not modelled: Python submit-worker threading and topology-admission tables in
the external connector above `ReshardService`, partial-prefix load failures and
preemption/prefix caching (`Caller` B2–B3), disjointness of destination device
pages across concurrent transfers (`MultiRequest` A1: all pools sharing a
`storage_index` use a uniform `block_stride_bytes` in `ConfigurePoolStaging` so
blocks $b \neq b'$ occupy disjoint device byte ranges $[b \cdot S, (b+1) \cdot S)$,
and the engine's unified `[0, num_blocks)` block pool hands out distinct
uncached `dst_device_block_ids` across all cache groups to concurrent transfers),
and disjointness of source and destination units (`src_units ∩ dst_units = ∅`),
which is treated as a caller/planner precondition (`BuildStartTransferForTarget`
sets `is_sender = true` when a unit appears in both lists, which
`WorkerServiceImpl::StartTransfer` dispatches only to `PoolReshardPush`).

### Transport and staging boundary (below the session callbacks)

Modelled: the per-`(uuid, pool)` stream accounting, sub-block interval coverage,
and `PoolStagingStorage` slot maps — `begin_incoming_push` / `end_incoming_push`
op leases, `CompleteIncomingPush` counting completed streams against
`expected_pushes` before firing `OnPoolReceived(pool_idx)` once, sub-block
byte-interval partition on destination staging (`Coverage.lean`), and
per-`(storage_idx, uuid)` slot allocation and release in `PoolStagingStorage`
(`PoolStaging.lean`).

Not modelled: raw TCP socket loops and wire framing (Peregrine `ReadExact` /
`WriteExact`, TCP connection pooling, and per-transfer idle connection release
in `ReshardSendSession::EndOp`), and raw pointer `memcpy` / PJRT
`CopyFromHostToDevice` / `CopyToHostAsync` internals, which are collapsed into
atomic state-machine transitions (`d2hDone`, `landStream`, `h2dIssue`,
`h2dDone`).

## Assumptions and how each is discharged

The session and sub-plan chain modules number their assumptions (`Receive` A1–A4,
`Send` A1–A4, `SubPlanChain` A1–A4); the full statements and arguments are in the
module docstrings.

| Assumption | One-line statement | Status |
|---|---|---|
| Pipeline / MultiRequest engine contract | Destination reads `dstHbm` only after `done_recving`; producer reclaims `srcHbm` only after transfer settlement. | External contract; encoded as the guards of `.recycleSrc` / `.nextRequest` in `Reshard/MultiRequest.lean`. The other direction is `Transfer/Caller.lean`. |
| MultiRequest A1 — cross-transfer destination page disjointness | Destination device pages (`dstHbm`) of distinct concurrent transfers are disjoint, guaranteed by the allocator. | All pools sharing a `storage_index` have a uniform `block_stride_bytes` and `num_blocks` (`ConfigurePoolStaging`) and fit inside `[b * stride, (b + 1) * stride)` (`PoolSpec::Validate(storage_bytes)` when `storage_bytes = num_blocks * stride`, and the caller's manifest builder), so blocks $b \neq b'$ never overlap in device bytes; the engine's unified `[0, num_blocks)` block pool hands out distinct uncached `dst_device_block_ids` across all cache groups to concurrent transfers, and host arena slots are disjoint across `uuid`s (`PoolStaging.no_slot_aliasing_between_distinct_uuids`). |
| Receive A1 — at-most-once `OnPoolReceived` per declared pool | `CompleteIncomingPush` fires `OnPoolReceived(l, uuid)` only when `plan_pushes == expected_chunks`, and `RecordPoolReceivedLocked` rejects undeclared or duplicate pool indices. | Discharged: `Pipeline.StreamAccountingSafe`, `Coverage.stream_count_matches`, `CoveragePipeline.step_pipe`. |
| Receive A2 — registration before visibility | `RegisterActivePlanDirect` and `AcquireStagingLeases` in `ReshardReceiveSession::Create` succeed before insertion into `active_pool_reshard_recvs_`, rolling back both on failure. | Discharged: `PoolStaging.reachable_reserveBeforeLease_safe`, `Coordinator.no_push_before_all_armed`. |
| Receive A3 / Send A3 — no double end and single `StartPoolPush` lock step | Each admitted push and each H2D/D2H op ends at most once; `StartPoolPush`'s two `done_ \|\| finalizing_` checks inspect only immutable plan fields in between and fold into `d2hDone`. | C++ callback and control-flow discipline. |
| Receive A4 — fold of `RecordPool*Locked` with `ExecuteEligiblePoolH2ds` first lock | An external `Finish` between unlocking `mu_` after recording a pool and locking `mu_` in `ExecuteEligiblePoolH2ds` produces the same state as `cancel` arriving before the event, whereas one arriving after `in_flight_ += to_launch.size()` is `cancel` after the event. | Soundness argument in `Receive.lean` (`trace_cancel_between_batch_queue_and_issue`). |
| Send A1 — non-empty schedule with positive peer counts per active pool | `ReshardSendSession::Create` rejects `remaining_pool_peer_pushes <= 0`, and `ExecutePush` skips pools with zero scheduled entries on this sender, matching `CountPoolReshardSendSlots`. | Discharged: `Coverage.senderStreams_pos_iff`, `Send.validPools`. |
| Send A2 — `ExecutePush` starts before external `Finish` | `PoolReshardPush` inserts the session into `active_pool_reshard_sends_` and immediately invokes `ExecutePush`, which takes its caller op (`++in_flight_`). | C++ init discipline (`KVCacheManagerWithTransfer::PoolReshardPush`). |
| Send A4 — per-transfer connection-release window in `EndOp` | Under `DataConnectionReuse::kPerTransfer`, the last op (`in_flight_ == 1 && finalizing_`) drops `mu_` in `EndOp` to call `ReleaseDataConnectionsAfterSend` before `--in_flight_` and `SettleLocked()`; nothing modelled changes in the window. | Soundness argument in `Send.lean`. |
| SubPlanChain A1–A2 — valid `sub_plan_index` and fresh per-sub-plan UUIDs | `RunSubPlan` stamps `0 <= sub_plan_index < max(1, sub_plan_count)` (`BuildStartTransferForTarget`), `SubPlanUuid` derives distinct positive UUIDs per index, and retries use a fresh request UUID. | Discharged for `SubPlanUuid` (`subPlanUuid_zero`, `subPlanUuid_pos_of_pos`, `trace_sub_plan_uuid_positive_and_distinct`); duplicate-UUID admission is `Reshard/PoolStaging.lean`. |
| SubPlanChain A3–A4 — sender `Complete` after last window and producer `Register` before `Coordinate` | A producer calls `RequestBlockRegistry::Complete` for a rank only after its worker emits `done_sending` (once all `count` sub-plans settle), and registers blocks before `StartPoolReshard`. | Discharged for `done_sending` (`SubPlanChain.reachable_worker_safe` / `DoneAfterAllSubPlans`) and post-claim `Register` refusal (`Reshard/Registry.lean`); single-window pre-claim arrival is `Reshard/Coordinator.lean`. |
| Caller B1–B3 — multi-worker `KVOutputAggregator` threshold (`expectedP = 2`, `expectedD = 2`) with one settled outcome per worker (`TPU_RAIDEN_ENABLE_MULTI_NUMA=0`), whole-transfer failure, no preemption/prefix caching | `Transfer/Caller.lean` §Assumptions. | B1's `KVOutputAggregator` combining step is modelled directly in `Caller/MultiWorker.lean` (`mw_contract_safe`, `applyRecv_eager_eq_latched`, `applyRecv_staggered_eq_batch`, `mw_earlyLoadError_fail_holds_blocks`; enforced across TP ranks by `_sync_stage3_tp_submit_states`), while per-worker folding of `sub_plan_count > 1` chains into a single worker outcome is modelled in `Reshard/SubPlanChain.lean` (`WorkerState`); B2 matches single-outcome `CompleteReadWithDetails` publication; B3 does not change which event frees a transfer-held block. |

## Shipping C++ vs. modelled alternatives

Two modules parameterize `step` by a **shipping C++ guard** and the same
`step` **without it**, so the guard's role is shown by what breaks
(`PushEndPolicy`, `RegistrationPolicy`). Every other module has one shipping
`step`; where a trace shows what the code leaves behind, `Traces.lean` also
carries the one `step` change that would not — `Send.stepUnregisterOnSettle`
(`RecordPushCompletion` without `UnregisterActivePlan`; `planUnregistered = done`
on the 1- and 2-pool instances), `Coordinator.stepDisarmAndAbandon` (disarm and
`AbandonClaim` on both aborts; `NoOrphansWhenAborted` on the 2×2 instance) and
the `if (index == 0)` mutant `SubPlanChain.stepAbandonAtEveryIndex` (which shows
why the coordinator does *not* abandon at every index). These alternatives are
not the shipping code and are not what the inductive theorems are about.

| Module | Parameter | Shipping C++ behavior | Modelled alternative |
|---|---|---|---|
| `Reshard/Receive.lean` | `PushEndPolicy` | **`PushEndPolicy.finishOnError`**: `hooks.end_incoming_push` (`KVCacheManagerWithTransfer::InitializeBaseHooks`) calls `reshard_session->Finish(status)` before `EndRecvOp()` when `!status.ok()` (matching the `TransferReceiveSession` branch), so a failed push drains and releases staging at once (`trace_failed_incoming_push`; regression test `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline`). | **`PushEndPolicy.endOpOnly`** (mutant): calls `EndRecvOp()` without checking `status.ok()`, so after a mid-transfer push failure the receiver stays `!IsDraining() && !Done()`, pinning its `PoolStagingLease` and active plan until `timeout_s_` (`trace_failed_incoming_push`). `Recv.reachable_safe` holds under both. |
| `Reshard/PoolStaging.lean` | `RegistrationPolicy` | **`RegistrationPolicy.reserveBeforeLease`**: `ReshardReceiveSession::Create` claims `uuid` via `RegisterActivePlanDirect` before constructing the session and acquiring `PoolStagingStorage` leases. Proves inductively (`reachable_reserveBeforeLease_safe`) that every active session retains its allocated staging slots (`ActiveSessionsHoldLeases`). | **`RegistrationPolicy.leaseBeforeReserve`** (mutant): acquires `st.leases[uuid]` before `RegisterActivePlanDirect`. A rejected duplicate-UUID call destroys its temporary session, whose destructor calls `ReleasePoolStagingLeases(uuid)` and frees the surviving session's physical staging slots (`trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`). |

## Counterfactual mutants

Each mutant removes a specific C++ guard or ordering constraint and is refuted
by an executable counterexample trace or `#guard` check in
`TpuSyncVerify/Transfer/Reshard/`:

| Mutant | C++ guard / invariant removed | Property violated |
|---|---|---|
| `Recv.onPoolReceivedIgnoringRank` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` checks that all pools with strictly lower `rank` are `completed` before queuing a pool for H2D | `RankWaveOrder` (higher-rank pool launches H2D before lower-rank pool finishes) |
| `Recv.h2dIssueLeak` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` calls `EndRecvOpLocked()` on each remaining queued pool when exiting early on `draining_` | `NoOpLeak` (`in_flight_` stays positive after cancellation; session never settles) |
| `Recv.cancelUnregisterEarly` | `unregister_on_settle_ = true` (`ReshardReceiveSession`, `KVCacheManagerWithTransfer::MaybeUnregisterSettledReshardRecv`) defers `UnregisterActivePlan` until `Done()` (`in_flight_ == 0`) | `PlanIntegrity` (`planUnregistered = true` while `done = false` and H2D/push ops are still in flight) |
| `Send.initOvercountedSlots` | `CountPoolReshardSendSlots` (`ReshardSendSession::Create`) deduplicates `(pool, peer)` slots instead of summing raw schedule entries | `NoOpLeak` / `Publication` (`remaining_pool_peer_pushes_` never reaches 0; session hangs with `done = false`) |
| `Send.stepWithoutEndOpUnregister` | `ReshardSendSession::EndOp` calls `UnregisterActivePlan` when the last op drains after external cancellation (`Finish`) | `PlanIntegrity` (`done = true` while `planUnregistered = false`) |
| `Send.d2hDoneLeakOnDraining` | `ReshardSendSession::StartPoolPush` calls `EndOp()` when returning early on `finalizing_` after a D2H callback | `NoOpLeak` (leaked pool op keeps `in_flight_ > 0` forever on cancellation) |
| `Pipeline.mutant_ignoring_rank_corrupts_aliased_page` | Rank-ordered H2D wave gate on shared device pages (`ReshardReceiveSession::CollectEligiblePoolH2dsLocked`) | `AliasedPageCorrect` / `PublicationCorrect` (out-of-order arrival writes rank 1 first, then rank 0's `writePoolHbm` clobbers rank 1 to `.junk`, yielding `dstHbm = [kv 0, junk]`) |
| `Pipeline.mutant_onPoolReceived_before_all_streams` | `BlockTransportServerPushSession::CompleteIncomingPush` waits until `completed_streams == expected_pushes` before calling `OnPoolReceived(pool_idx)` | `StreamAccountingSafe` / `PublicationCorrect` (H2D reads incomplete staging before all sender streams land) |
| `Pipeline.mutant_reclaim_before_publish_corrupts_dstHbm` | Caller/coordinator retires producer blocks only after transfer completion (`SrcHbmSafe`) | `PublicationCorrect` (reclaiming `srcHbm` before D2H copies `.junk` into `dstHbm`) |
| `Pipeline.trace_mutant_write_after_release_corrupts_all` | `ReshardSendSession::SettleLocked` releases `PoolStagingLease` only when `in_flight_ == 0` (`StagingSafe`) | `wroteReleased = false` (releasing `srcStaging` at `execFinish` while D2H is in flight causes `d2hDone` to write after release and corrupt all transfers' buffers) |
| `Coverage.mutant_gap_in_coverage_leaves_unwritten_byte` | `ValidateReceiverCoverage` rejects coverage intervals with a gap (`cursor != extent`) | Sub-block reconstruction (`dstStaging` retains `.uninit` bytes after all streams land) |
| `Coverage.mutant_overlap_in_coverage_rejected` | `ValidateReceiverCoverage` rejects overlapping sub-block intervals (`seg.start < cursor`) | Disjoint interval partition (`validBlockCoverage = false`) |
| `Coverage.mutant_mismatched_expected_pushes_rejected` | `ValidateReceiverCoverage` rejects plans where `expected_pushes` disagrees with $\sum_s \min(\text{parallelism}, |\text{pairs}_s|)$ | `stream_count_matches` (`validExpectedPushes = false`) |
| `PoolStaging.mutant_non_idempotent_acquire_exhausts_shared_storage` | `KVCacheManagerBase::AcquirePoolStagingLease` skips block IDs already leased by the same `uuid` | Shared-storage lease admission (second pool sharing `storage_index` re-allocates slots and exhausts bounded capacity) |
| `PoolStaging.mutant_leaking_release_breaks_conservation_and_starves` | `KVCacheManagerBase::ReleasePoolStagingLeases` returns all leased slots to `free_slots` when erasing `leases[uuid]` | Slot conservation (`freeSlots.length + leasedSlots = capacity`; subsequent transfer starves) |
| `Registry.mutant_premature_complete_retires_while_peer_active` | `RequestBlockRegistry::Complete` checks `completed_units_ == claimed_units_` before calling `RetireLocked` | Multi-sender claim safety (first sender's `Complete` retires blocks while second sender is still reading) |
| `Registry.mutant_unconditional_abandon_exposes_sibling_owner_to_cancel` | `RequestBlockRegistry::AbandonClaim` retains `claimed_` until `claimed_owners_` is empty | Pipelined multi-owner claim safety (stage 1's `AbandonClaim` exposes stage 2's claimed blocks to `CancelIfUnclaimed`) |
| `Registry.trace_single_stage_votes_retire_multi_owner_claim` | `RequestBlockRegistry::Complete` tracks `completed_units_` per `(req_id, uuid)` rather than per `(req_id, uuid, claim_owner)` | Multi-owner claim isolation (stage 1's senders voting `Complete` retire the generation while stage 2's claim is still active) |
| `Registry.trace_split_ttl_purges_request_blocks_while_claim_active` | `RequestBlockRegistry::LookupAndClaim` sets `claimed_[(req_id, uuid)] = t_1 + ttl_s_` without refreshing `registration.expires_at` on the claimed `request_blocks_` rows | Claimed registration liveness (`PurgeExpiredRequestBlocksLocked` erases `request_blocks_` in `[t_0 + ttl_s_, t_1 + ttl_s_)` while `claimed_` is still active) |
| `Coordinator.mutant_dispatch_without_arm_barrier_drops_push` | `ReshardCoordinator::RunSubPlan` joins all Stage 1 receiver-arming threads and verifies every `armed[i]` before launching Stage 2 senders | Two-phase Arm-then-Dispatch (`NoPushBeforeAllArmed`: premature sender push arrives at unarmed `dst_unit 11` and is dropped) |
| `SubPlanChain.stepCountStale` (`mutant_counting_stale_sub_plan_reports_done_early`) | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` skips settled sessions whose `uuid` is absent from `plan_uuids` | `DoneAfterAllSubPlans` (a stale sub-plan from a superseded chain counts toward a retry's `settled` set and emits `done_recving` / `done_sending` while the retry's own sub-plan is still active) |
| `SubPlanChain.stepNoReanchor` (`mutant_no_reanchor_never_reports_retry_done`) | `KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked` replaces the chain entry on `sub_plan_index == 0` | Retry liveness under same `req_id` (a retry's sub-plans land in the earlier failed entry and never emit `done`) |
| `SubPlanChain.stepAbandonAtEveryIndex` (`mutant_abandon_at_every_index_exposes_active_sender`) | `ReshardCoordinator::RunSubPlan` guards `registry_->AbandonClaim(...)` with `if (index == 0)` on arm / plan aborts | `NoSenderExposed` (`AbandonClaim` on a window `>= 1` failure drops the registry claim while window `0`'s dispatched sender is still reading source HBM, letting `CancelIfUnclaimed` tombstone the generation mid-transfer) |

## C++ → Lean routing index

Per C++ symbol: the Lean module that owns it, the model events or definitions
it becomes, and the theorems to open first. The `Lines` column is generated by
`tools/resolve_anchors.py`.

### Receive session (`reshard_recv.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `reshard_recv.h` | `ReshardReceiveSession` | `:54-201` | `Transfer/Session.lean`, `Reshard/Receive.lean` | `in_flight_`, `draining_`, `done_`, `staging_released_`, `unregister_on_settle_`, `pool_order_ranks_`; `~ReshardReceiveSession` calls `ReleasePoolStagingLeases(uuid_)` | `Recv.reachable_safe` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidatePlan` | `:67-144` | `Reshard/Receive.lean` | `Recv.validPlan`, `Recv.validPlanPoolsAndBlocks` | `Recv.trace_reshard_receive_session_test_parity` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` | `:146-349` | `Reshard/Coverage.lean` | `Coverage.validBlockCoverage`, `Coverage.validateExpectedPushes` | `Coverage.partition_reconstructs_perm`, `Coverage.stream_count_matches`, `Coverage.trace_reshard_stack_routing_test_parity` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` | `:351-390` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean` | `PoolStaging.RegistrationPolicy` (`reserveBeforeLease` claims `uuid` via `RegisterActivePlanDirect` before `AcquireStagingLeases`) | `PoolStaging.reachable_reserveBeforeLease_safe` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:422-461` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean` | Per-storage bounded lease acquisition (`AcquirePoolStagingLease`) | `PoolStaging.acquireLease_idempotent` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishLocked` | `:474-484` | `Transfer/Session.lean`, `Reshard/Receive.lean` | `Lifecycle.finishLocked`, `Recv.Ev.cancel` | `Recv.SettleSafe` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndRecvOpLocked` | `:497-507` | `Transfer/Session.lean`, `Reshard/Receive.lean` | `Lifecycle.endOpLocked` | `Recv.StagingIntegrity` |
| `reshard_recv.cc` | `ReshardReceiveSession::TakePendingUnregister` | `:514-522` | `Reshard/Receive.lean` | `Recv.maybeUnregisterSettled` | `Recv.PlanIntegrity`, mutant `Recv.cancelUnregisterEarly` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolReceivedLocked` | `:543-554` | `Reshard/Receive.lean` | Rejects undeclared or duplicate pool indices (`Receive` A1) | `Recv.reachable_safe` |
| `reshard_recv.cc` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` | `:556-585` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | Rank gate + batch `in_flight_ += to_launch.size()` | `Recv.RankWaveOrder` (`rankWaveOrder_iff`), `Pipeline.AliasedPageCorrect`, mutants `Recv.onPoolReceivedIgnoringRank`, `Pipeline.mutant_ignoring_rank_corrupts_aliased_page` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnPoolReceived` | `:602-614` | `Reshard/Receive.lean` | `Recv.Ev.onPoolReceived` | `Recv.RankWaveOrder` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` | `:616-668` | `Reshard/Receive.lean` | `Recv.Ev.h2dIssue` (`draining_` early exit calls `EndRecvOpLocked()`) | `Recv.NoOpLeak`, mutant `Recv.h2dIssueLeak` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:670-708` | `Reshard/Receive.lean` | `Recv.Ev.h2dReady`, `Recv.Ev.h2dDone` (sets `network_completed_ = true` before `Finish`) | `Recv.reachable_pollReady_disabled` |
<!-- source-map:end -->

### Send session (`reshard_send.{h,cc}`, `slots.cc`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `reshard_send.h` | `ReshardSendSession` | `:48-156` | `Transfer/Session.lean`, `Reshard/Send.lean` | `remaining_pool_peer_pushes_`, `finalizing_`, `in_flight_`, `done_` | `Send.reachable_safe` |
| `slots.cc` | `CountPoolReshardSendSlots` | `:27-51` | `Reshard/Send.lean`, `Reshard/Coverage.lean` | Deduplicated `(pool, peer)` slot count (`Coverage.countPoolReshardSendSlots`) | `Send.ActiveOrDraining`, `Coverage.trace_pool_reshard_send_slots_test_parity`, mutant `Send.initOvercountedSlots` |
| `reshard_send.cc` | `ReshardSendSession::Create` | `:92-151` | `Reshard/Send.lean` | Calls `CountPoolReshardSendSlots`, rejects `remaining_pool_peer_pushes <= 0`, registers active plan and staging leases | `Send.reachable_safe` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` | `:177-265` | `Reshard/Send.lean` | `Send.Ev.d2hIssue`, `Send.Ev.execFinish` | `Send.ActiveOrDraining` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:267-349` | `Reshard/Send.lean` | `Send.Ev.d2hDone` | `Send.NoOpLeak`, mutant `Send.d2hDoneLeakOnDraining` |
| `reshard_send.cc` | `ReshardSendSession::RecordPushCompletion` | `:351-392` | `Reshard/Send.lean` | `Send.Ev.pushDone`, `Send.recordPushCompletion` (`UnregisterActivePlan` on the first error) | `Send.trace_early_unregister_on_error`, alternative `Send.stepUnregisterOnSettle` |
| `reshard_send.cc` | `ReshardSendSession::Finish` | `:394-410` | `Reshard/Send.lean` | `Send.Ev.cancel` | `Send.SettlesPromptly` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:418-468` | `Reshard/Send.lean` | Settles on `in_flight_ == 1 && finalizing_`, calls `UnregisterActivePlan` | `Send.PlanIntegrity`, mutant `Send.stepWithoutEndOpUnregister` |
<!-- source-map:end -->

### Manager and pool staging (`mgr.{h,cc}`, `base.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `mgr.h` | `KVCacheManagerWithTransfer` (`struct SubPlanChainProgress {` .. `void SweepSubPlanChainsLocked(std::chrono::steady_clock::time_point now)`) | `:350-379` | `Reshard/SubPlanChain.lean` | `SubPlanChain.ChainProgress`, `recv_chains_`, `send_chains_` | `SubPlanChain.reachable_worker_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked` | `:151-175` | `Reshard/SubPlanChain.lean` | `SubPlanChain.WorkerEv.registerSubPlan`, `noteSubPlanRegistered` | `SubPlanChain.reachable_worker_safe`, mutant `SubPlanChain.stepNoReanchor` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:177-227` | `Reshard/SubPlanChain.lean` | `SubPlanChain.WorkerEv.poll`, `recordTerminal` | `SubPlanChain.reachable_worker_safe` (`DoneAfterAllSubPlans`), `SubPlanChain.trace_first_failure_reports_while_sibling_active`, mutant `SubPlanChain.stepCountStale` |
| `mgr.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:229-288` | `Reshard/SubPlanChain.lean` | `SubPlanChain.WorkerEv.poll` (`stalled = true`), `sweepStalledChain` | `SubPlanChain.trace_sender_in_first_window_only`, `SubPlanChain.trace_late_sub_plan_after_sweep_reports_failure_twice` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` | `:290-490` | `Reshard/Receive.lean`, `Reshard/Pipeline.lean` | `begin_incoming_push` / `end_incoming_push` reshard branches (`Recv.Ev.pushBegin`, `Recv.Ev.pushEnd`, `Recv.PushEndPolicy.finishOnError` vs. `.endOpOnly`) | `Recv.trace_failed_incoming_push` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:833-871` | `Reshard/Send.lean`, `Reshard/SubPlanChain.lean` | Inserts sender into `active_pool_reshard_sends_`, records sub-plan registration, and calls `ExecutePush` (`Send` A2) | `Send.reachable_safe`, `SubPlanChain.reachable_worker_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:873-910` | `Reshard/Receive.lean`, `Reshard/PoolStaging.lean`, `Reshard/SubPlanChain.lean` | `PoolStaging.RegEv` (`beginArm`, `acquireArmLease`, `finishArm`, `settleSession`) and `NoteSubPlanRegisteredLocked` | `PoolStaging.reachable_reserveBeforeLease_safe`, `PoolStaging.trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`, `SubPlanChain.reachable_worker_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::MaybeUnregisterSettledReshardRecv` | `:977-983` | `Reshard/Receive.lean` | `Recv.maybeUnregisterSettled` (`TakePendingUnregister` → `UnregisterSettledPlan`) | `Recv.PlanIntegrity` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1151-1292` | `Reshard/Receive.lean`, `Reshard/Send.lean`, `Reshard/Pipeline.lean`, `Reshard/MultiRequest.lean`, `Reshard/SubPlanChain.lean`, `Transfer/Caller.lean` | Reshard send/recv sweeps (`Send.Ev.publish`, `Recv.Ev.publish`, `SubPlanChain.WorkerEv.poll`, `Caller.Ev.pOutcome`, `Caller.Ev.dOutcome`) | `Pipeline.PublicationCorrect`, `Pipeline.attention_safe`, `Pipeline.system_data_correct`, `SubPlanChain.reachable_worker_safe`, `Caller.contract_safe` |
| `base.h` | `KVCacheManagerBase` | `:117-1204` | `Reshard/PoolStaging.lean` | `PoolStagingStorage` and aliased hybrid-model pool declarations | `PoolStaging.ArenaInv` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` | `:2136-2193` | `Reshard/PoolStaging.lean` | `PoolStaging.Arena.acquireLease` | `PoolStaging.acquireLease_idempotent`, `PoolStaging.no_slot_aliasing_between_distinct_uuids`, mutant `mutant_non_idempotent_acquire_exhausts_shared_storage` |
| `base.cc` | `KVCacheManagerBase::ReleasePoolStagingLeases` | `:2195-2211` | `Reshard/PoolStaging.lean` | `PoolStaging.Arena.releaseLease` | `PoolStaging.allocateMissing_slot_sum`, mutant `mutant_leaking_release_breaks_conservation_and_starves` |
| `base.cc` | `KVCacheManagerBase::RegisterActivePlanDirect` | `:3092-3096` | `Reshard/PoolStaging.lean` | Direct plan registration before staging lease acquisition | `PoolStaging.reachable_reserveBeforeLease_safe` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` | `:3271-3582` | `Reshard/PoolStaging.lean`, `Reshard/Send.lean` | `PoolStaging.Arena.resolveBlockSlot` (`block_host_base`) and sub-block staging resolution on sender | `PoolStaging.no_slot_aliasing_between_distinct_uuids`, `Send.trace_early_unregister_on_error` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline` | `:766-802` | `Reshard/Receive.lean` | Replayed by `trace_failed_incoming_push` (`PushEndPolicy.finishOnError`) | `Recv.trace_failed_incoming_push` |
<!-- source-map:end -->

### Transport and planner (`bt.cc`, `bts.cc`, `planner.cc`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:458-463` | `Reshard/Pipeline.lean` | delegates stream to the uuid's `BlockTransportServerPushSession::StartStream` | `Pipeline.StagingSafe` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:321-455` | `Reshard/Pipeline.lean` | `begin_incoming_push` / `end_incoming_push` guard around incoming push stream | `Pipeline.StagingSafe` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:506-702` | `Reshard/Pipeline.lean`, `Reshard/Coverage.lean` | Plan-declared progress: counts completed streams against `expected_pushes` and fires `OnPoolReceived(pool_idx)` once (`Pipeline.Ev.landStream`) | `Pipeline.StreamAccountingSafe`, mutant `Pipeline.mutant_onPoolReceived_before_all_streams` |
| `bt.cc` | `BlockTransport::AsyncPushShards` | `:550-614` | `Reshard/Coverage.lean`, `Reshard/Send.lean` | Sender stream fan-out (`min(parallelism, num_blocks)`) and `BuildBlockRequests` call | `Coverage.stream_count_matches` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` | `:719-833` | `Reshard/Coverage.lean`, `Reshard/Send.lean` | Sub-block request construction and `GetBlockChunks` call | `Coverage.partition_reconstructs_perm` |
| `planner.cc` | `DefaultTransferStreams` | `:611-628` | `Reshard/Coverage.lean` | `Coverage.defaultTransferStreams`, `Coverage.resolvePlanParallelism` | `Coverage.trace_default_transfer_streams_test_parity` |
| `planner.cc` | `BuildPoolReshardPlan` | `:630-1340` | `Reshard/Coverage.lean` | `Coverage.senderStreams`, `Coverage.recomputedExpectedPushes` | `Coverage.stream_count_matches`, `Coverage.senderStreams_pos_iff` |
<!-- source-map:end -->

### Registry and coordinator (`reg.{h,cc}`, `coord.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `reg.h` | `RequestBlockRegistry` | `:45-151` | `Reshard/Registry.lean` | `RegGen`, `request_blocks_`, `claimed_`, `claimed_owners_`, `completed_units_`, `cancelled_` | `Registry.reachable_safe` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredRequestBlocksLocked` | `:61-69` | `Reshard/Registry.lean` | `RegGen.purgeExpiredRequestBlocks` | `Registry.trace_split_ttl_purges_request_blocks_while_claim_active` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredLifecycleLocked` | `:71-96` | `Reshard/Registry.lean` | `RegGen.purgeExpiredAll` | `Registry.purgeExpiredAll_inv` |
| `reg.cc` | `RequestBlockRegistry::RetireLocked` | `:98-116` | `Reshard/Registry.lean` | Erases `request_blocks_` and claim state on completion | `Registry.complete_retires_iff_all_claimed_voted` |
| `reg.cc` | `RequestBlockRegistry::Register` | `:118-300` | `Reshard/Registry.lean` | `Ev.register` | `Registry.tombstone_blocks_register_and_claim`, `Registry.claimed_blocks_register_and_cancel` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` | `Reshard/Registry.lean` | `Ev.complete` | `Registry.complete_retires_iff_all_claimed_voted`, `Registry.trace_single_stage_votes_retire_multi_owner_claim`, mutant `mutant_premature_complete_retires_while_peer_active` |
| `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` | `:373-405` | `Reshard/Registry.lean` | `Ev.cancelIfUnclaimed` | `Registry.tombstone_blocks_register_and_claim` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-491` | `Reshard/Registry.lean` | `Ev.lookupAndClaim` | `Registry.claimed_blocks_register_and_cancel`, `Registry.trace_split_ttl_purges_request_blocks_while_claim_active` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:493-509` | `Reshard/Registry.lean` | `Ev.abandonClaim` | `Registry.abandonClaim_preserves_when_other_owner_present`, mutant `mutant_unconditional_abandon_exposes_sibling_owner_to_cancel` |
| `coord.h` | `SubTransferChain` | `:108-111` | `Reshard/SubPlanChain.lean` | `SubPlanChain.SubTransferSlice`, `SubPlanChain.SubTransferChain` | `SubPlanChain.planSubTransfers_no_aliased_window_split` |
| `coord.h` | `ReshardCoordinator` | `:138-207` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean` | `CoordState`, `CoordInv`, `CoordChainState`, `CoordChainSafe` | `Coordinator.reachable_coord_inv`, `SubPlanChain.reachable_coordChain_safe` |
| `coord.cc` | `BuildStartTransferForTarget` | `:111-271` | `Reshard/Coordinator.lean` | Target request construction (`is_sender` when unit in `src_units`) | — |
| `coord.cc` | `SubPlanUuid` | `:356-367` | `Reshard/SubPlanChain.lean` | `SubPlanChain.subPlanUuid` (63-bit positive `splitmix64` for `index > 0`) | `SubPlanChain.subPlanUuid_zero`, `SubPlanChain.subPlanUuid_pos_of_pos` |
| `coord.cc` | `PlanSubTransfers` | `:369-472` | `Reshard/SubPlanChain.lean` | `SubPlanChain.planSubTransfers`, `hasWindowedTagSharingIds` | `SubPlanChain.planSubTransfers_no_aliased_window_split` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:474-526` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean` | `ChainEv.startChainCoordination` (`transfer_status_` transitions `if (!chain_started)`) | `SubPlanChain.reachable_coordChain_safe` (`CompletedOnlyAtChainEnd`) |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:528-601` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean` | Draws `claim_owner`, builds `ChainContext`, runs `RunSubPlan(ctx, 0)`, and spawns `StartChain` when `chain.count > 1` | `Coordinator.reachable_coord_inv`, `SubPlanChain.reachable_coordChain_safe` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:614-649` | `Reshard/SubPlanChain.lean` | Multi-window chain continuation (`RunSubPlan(ctx, 1 .. count - 1)`) and final `transfer_status_` publication | `SubPlanChain.reachable_coordChain_safe` (`ClaimHeldAcrossWindows`), `SubPlanChain.trace_chained_request_stays_in_progress_until_chain_ends`, `SubPlanChain.trace_chained_arm_failure_keeps_claim_past_last_vote` |
| `coord.cc` | `ArmRefusedForBusyStaging` | `:667-670` | `Reshard/SubPlanChain.lean` | `ChainEv.armDstResult _ .refusedBusy`, `ChainEv.retryBusyArm` (`kChainArmAttempts` re-arm loop while an earlier window holds staging) | `SubPlanChain.reachable_coordChain_safe` (`OneWindowStagedPerDestination`), `SubPlanChain.trace_chained_arm_retries_while_receiver_staging_busy`, `SubPlanChain.trace_busy_refusals_exhaust_arm_attempts` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:678-932` | `Reshard/Coordinator.lean`, `Reshard/SubPlanChain.lean`, `Reshard/Pipeline.lean` | Stage 0 `LookupAndClaim`, Stage 1 parallel destination arming & busy-staging retries, Stage 2 parallel sender dispatch, `if (index == 0)` on `AbandonClaim` | `Coordinator.no_push_before_all_armed`, `Coordinator.claimed_while_arming`, `Coordinator.trace_partial_arm_orphans_armed_receiver`, `Coordinator.trace_dispatch_failure_orphans_receivers_and_claim`, `SubPlanChain.reachable_coordChain_safe` (`NoSenderExposed`), mutant `SubPlanChain.stepAbandonAtEveryIndex` |
<!-- source-map:end -->

*External, the production reshard caller pair (`Transfer/Caller.lean`):*
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
`Caller.twoChannel_contract`, `Caller.earlyLoadError_fail_eq_finish`,
`Caller.earlyLoadError_fail_holds_blocks`,
`Caller.mw_earlyLoadError_fail_eq_finish`,
`Caller.mw_earlyLoadError_fail_holds_blocks`,
`Caller.trace_twoChannel_early_uncertain_load_error`, `Caller.trace_twoChannel_*`,
`Caller.trace_mw_staggered_*`, `Caller.trace_mw_overcounted_expected_leaks`.

## Where the rest lives

| Content | File |
|---|---|
| `trace_*` witnesses and the tests they replay; `#guard` bounded searches | the theorem docstrings and `#guard` lines in the Lean modules (`grep -rn "theorem trace_\|#guard" TpuSyncVerify/Transfer/Reshard`) |
| how the models are written (one executable `step`, ghost fields, replay, mutants) | [conventions.md](../conventions.md) |

# Prefill-to-decode transfer: model and results

The reference map from the prefill-to-decode KV-cache transfer C++ to its Lean
models under `TpuSyncVerify/Transfer/`.

Each module's docstring owns its state table, event table, assumptions,
properties and `## Source map`; this page routes between them, defines the
three model boundaries, summarizes every assumption and counterfactual mutant,
and indexes C++ symbols to Lean modules. A `trace_*` theorem that replays a C++
or end-to-end test names it in its docstring (`grep -rn <TestName> TpuSyncVerify`).

File aliases: `recv.{h,cc}` = `tpu_sync/kv_cache/transfer_receive_session.{h,cc}`,
`send.{h,cc}` = `tpu_sync/kv_cache/transfer_send_session.{h,cc}`,
`staging.{h,cc}` = `tpu_sync/kv_cache/staging_manager.{h,cc}` (`StagingLayerManager`),
`tpu_sync/transport/transfer_session.h` (the `TransferSession` interface, cited by
its full path),
`mgr.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`,
`bt.{h,cc}` = `tpu_sync/transport/block_transport.{h,cc}`,
`bts.{h,cc}` = `tpu_sync/transport/block_transport_session.{h,cc}`,
`base.cc` = `tpu_sync/kv_cache/kv_cache_manager_base.cc`,
`tcp_control_plane_backend.cc` = `tpu_sync/core/tcp_control_plane_backend.cc`,
`grpc_control_plane_backend.cc` = `tpu_sync/core/grpc_control_plane_backend.cc`.

## What is proved

Every theorem below is proved on every reachable state of its model; the
module docstring gives the exact statement.

| Property | Theorems | Statement |
|---|---|---|
| Data correctness and attention safety across requests | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe` | In `multiSys` (concurrent, overlapped and recycled requests), a request publishing `done_recving` has `decodeHbm = good n` across every later transition. |
| Progress across requests | `Pipeline.system_progress` | From any reachable state, any request can settle, return both staging buffers and publish both outcomes (`HandedOff`) in finitely many steps. |
| Publication correctness, one request | `Pipeline.reachable_safe` (`PublicationCorrect`), `LayerStaging.reachable_safe` (`PublicationCorrect`) | `recv.published = some true → decodeHbm = good n` on both the full host mirror and bounded `numSendSlots` / `numRecvSlots` under `num_staging_layers`. |
| Bounded host layer staging (`num_staging_layers`) | `StagingLayerManager.lookup_injective`, `LayerStaging.reachable_safe`, `LayerStaging.sendLookupStagingSlot_active`, `LayerStaging.recvLookupStagingSlot_active`, `LayerStaging.reachable_can_settle`, `LayerStaging.no_slots_cannot_publish_ok`, `LayerStaging.exists_complete_trace` | Host layer slots are conserved, mutually exclusive across layers, live and in-bounds on every active lookup (`slotFault = false`), and returned in full at settle; `0` slots on either side cannot publish `done_recving` for `n > 0`, while `1` slot per side suffices for arbitrary `n` without deadlock. |
| Within-layer blocks: gather, reordering, coalescing, custom staging | `BlockOrdering.execCoalesced_buildCoalescedSpec`, `BlockOrdering.landStage_get_requested`, `BlockOrdering.h2dStage_get_requested`, `BlockOrdering.BlockPipeline.reachable_safe` | Coalesced DMA equals elementwise copy; `BuildLoadCopyPlan`'s `remote_order` and `local_order` permutations cancel, delivering `.kv l t.remote` to every requested block while unrequested blocks stay blank. |
| Decode HBM safety and quietness | `Recv.PublishedQuiet`, `Pipeline.DecodeHbmSafe`, `decodeHbm_quiet`, `attention_safe` | Once the receive is published no H2D copy is pending or unretired, so `decodeHbm` never changes afterwards. |
| Prefill HBM safety and quietness | `Send.PublishedQuiet`, `Pipeline.PrefillHbmSafe`, `prefillHbm_quiet` | Once the send is published no D2H copy is pending or unretired, so the caller may reclaim prefill HBM. |
| Staging integrity and quietness | `Recv.StagingIntegrity`, `Send.StagingIntegrity`, `Pipeline.StagingSafe`, `wroteReleased_eq_false` | `hasStaging = !done`; released staging is neither written nor read by the session that held it. |
| Handshake rendezvous and single pull claim | `Send.PullClaimed`, `Pipeline.HandshakeSafe`, `Pipeline.reachable_unregistered_safe`, `BlockOrdering.validateRegistration_iff`, `BlockOrdering.validateRequestedBlocks_iff` | `StartPush` and data movement require `NotifyForRead` registration and `ValidateAndBeginPull`'s claim of a non-empty, duplicate-free registered subset; while `HandlePullStream` waits, the pull stays pending and unclaimed. |
| Termination | `Recv.NoOpLeak`, `Send.NoOpLeak`, `Pipeline.NoOpLeak`, `Recv.reachable_can_settle`, `Send.reachable_can_settle`, `Pipeline.reachable_can_settle` | Every unit of `in_flight_` has an enabled step that advances or retires it, and every reachable state drains to `done ∧ ¬hasStaging` and publishes. |
| Per-side counters | `Recv.Publication`, `Send.Publication`, `Recv.ReadinessSound`, `Recv.SettleSafe`, `Send.SettleSafe`, `Send.NoUnderflow` | `done_recving → every H2D callback ran OK`; `done_sending → every push completed OK`; `IsReadyToComplete → ready = numLayers`; `done → inFlight = 0`; `EndSendOpLocked` never underflows. |
| Transport accounting | `Transport.reachable_safe` (`OnceDispatched`, `NetworkSound`, `CompleteOnSuccess`) | With honest senders, `OnLayerReceived` fires once per layer, the block target is reached only after every layer was dispatched, and a successful receive copied every layer once. |
| Engine caller contract | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.numa_settledCounts_safe`, `Caller.numa_settledCounts_maps_clean`, `Caller.leakP_absorbing`, `Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing` | Forwarding every publication leaks no held block, never frees HBM while a worker or NUMA sub-manager is in flight (`mwEarlyFree = false`, `numaEarlyFree = false`), and never decodes unwritten KV; a missed publication is unrecoverable. |
| Peer isolation and uuid tables | `PeerIsolation.reachable_safe`, `PeerIsolation.tcp_healthy_blocked_when_pool_full`, `PeerIsolation.grpc_healthy_can_complete`, `UuidTable.reachable_inv` | Staging slots and `push_pool_` workers are conserved; TCP blocking pulls can exhaust `push_pool_` across peers while gRPC async pulls hold no worker; a uuid is refused until its draining incumbent settles. |

## Modules

Higher modules compose the lower `step` functions as-is: `Pipeline` = `Send.step` +
`Recv.step` + per-layer memories and guards; `LayerStaging` refines `Pipeline`
(`LayerStaging.step_pipe_steps`) over `numSendSlots` / `numRecvSlots` host
buffers through `StagingLayerManager`; `BlockOrdering` refines `Pipeline`'s
layer cell into a block array (`BlockPipeline.step_pipe`); `MultiRequest` runs
`reqs : List Pipeline` over the four shared buffers.

| Module | Scope | Main theorems |
|---|---|---|
| `Common/System.lean` | transition-system core (`System`, `run`, `Reachable`) | `System.reachable_induction`, `System.settles_of_rank` |
| `Common/ModelCheck.lean` | breadth-first search (`ModelCheck.check`) evaluated by `#guard` | — |
| `Common/ListAux.lean` | `List` indexing and `countTrue` pigeonhole helpers | `all_true_of_countTrue_eq_length`, `exists_false_lt` |
| `Transfer/Session.lean` | settle protocol (`in_flight_`, `draining_`, `done_`, staging) shared by all sessions | `Lifecycle.Consistent` preserved by `beginOp` / `finish*` / `endOpLocked` |
| `Transfer/LayerStaging.lean` | `StagingLayerManager` FIFO free queue, active leases, `Shutdown`, `ReleaseAll`, `ResolveNumStagingLayers` | `StagingLayerManager.acquire_inv`, `release_inv`, `releaseAll_inv`, `lookup_injective`, `lookup_lt`, `acquire_idempotent`, `shutdown_cancels` |
| `PrefillDecode/Receive.lean` | one `TransferReceiveSession` + manager poll/publication | `Recv.reachable_safe`, `Recv.reachable_can_settle` |
| `PrefillDecode/ReceivePoll.lean` | poll-side `IsReadyToComplete` finish (`stepM true` shipping vs. `stepM false`); `metrics` ghost | `Recv.reachable_callbackFinishes`, `Recv.pollReady_window`, `Recv.noPoll_safe`, `Recv.noPoll_metrics_on_success`, `Recv.noPoll_zero_layers_never_succeeds` |
| `PrefillDecode/Send.lean` | one `TransferSendSession`: pull claim, `StartPush`, D2H loop, H2H chain | `Send.reachable_safe` (`PullClaimed`), `Send.reachable_can_settle` |
| `PrefillDecode/Pipeline.lean` | single request: handshake rendezvous + five layer-indexed memories, reclaim, reseat, handoff | `Pipeline.reachable_safe`, `Pipeline.reachable_unregistered_safe`, `Pipeline.attention_safe`, `Pipeline.reachable_can_settle`, `Pipeline.handedOff_quiet`, `Pipeline.inv_can_handoff` |
| `PrefillDecode/LayerStaging.lean` | bounded host layer staging (`num_staging_layers`): `BoundedPipeline` over `numSendSlots` / `numRecvSlots` host buffers and unlocked `AcquireStagingSlot` gaps | `LayerStaging.reachable_safe`, `reachable_unregistered_safe`, `sendLookupStagingSlot_injective`, `recvLookupStagingSlot_injective`, `reachable_can_settle`, `no_slots_cannot_publish_ok`, `exists_complete_trace` |
| `PrefillDecode/MultiRequest.lean` | `multiSys`: concurrent/overlapped requests, read- and write-after-release checks (`wroteReleased`) | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe`, `Pipeline.system_progress` |
| `PrefillDecode/PeerIsolation.lean` | one consumer, `Peer.sick` / `Peer.healthy`, `StagingBlockAllocator` slots, `push_pool_` (`tcpBlocking` vs `grpcAsync`) | `PeerIsolation.reachable_safe`, `PeerIsolation.reachable_inv`, `tcp_healthy_blocked_when_pool_full`, `grpc_healthy_can_complete`, `trace_sick_peer_starves_staging_slots` |
| `PrefillDecode/UuidTable.lean` | `sessions_` (`SessionEntry{session, req_id, role}`), drain-before-reuse, `req_id` idempotency, `RetireSettledSessionLocked`, `CompleteReadWithDetails` sweep | `UuidTable.reachable_inv`, `active_recv_preserved`, `active_send_preserved` |
| `PrefillDecode/BlockOrdering.lean` | within-layer blocks: `PopulateRegisteredBlocks`, `ValidateRequestedBlocksLocked`, `BuildCoalescedCopySpec`, `BuildLoadCopyPlan`, custom host staging, `BlockPipeline` | `execCoalesced_buildCoalescedSpec`, `landStage_get_requested`, `h2dStage_get_requested`, `BlockPipeline.step_pipe`, `BlockPipeline.reachable_safe` |
| `PrefillDecode/PipelineChecks.lean` | `Pipeline` / `multiSys` traces, `#guard` searches, mutants | `Pipeline.trace_normal`, `trace_layers_out_of_order`, `trace_pull_ahead_of_registration`, `trace_multi_request`, `trace_overlapped_requests` |
| `PrefillDecode/Transport.lean` | receiver-side stream accounting (`BlockTransportServerPushSession`: `layer_progress_`, chunk-count and sender-declared contracts, once-latch, retirement) discharging `Receive` A1 and A4 | `Transport.reachable_safe`, `trace_dup_chunk_publishes_missing_layer`, `trace_dup_after_retirement`, `trace_dup_declared_rejected`, bounded `searchHardened` |
| `Transfer/Caller.lean` | engine contract above `poll_stats()`: `NumaAwareKVCacheManager::CompleteReadRaw` (`NumaBarrier.lean`), worker connector (`connectorTwoChannel` + mutants), `KVOutputAggregator` (`MultiWorker.lean`), and scheduler block lifecycle | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.numa_settledCounts_safe`, `Caller.numa_settledCounts_maps_clean`, `twoChannel_contract`, `applyRecv_eager_eq_latched`, `leakP_absorbing`, `leakD_absorbing` |

## Boundaries

### Subsystem boundary

In scope is the prefill-to-decode ("symmetric") transfer: producer registration
(`NotifyForRead`), consumer pull and load (`StartRead`), the D2H → H2H → H2D
pipeline in `TransferSendSession` / `TransferReceiveSession`, and outcome
polling (`CompleteReadWithDetails` / `poll_stats()`).

Out of scope: pool resharding (`ReshardSendSession` / `ReshardReceiveSession`,
see [reshard.md](reshard.md)), `RaidenController`, `kv_cache_store`, and
`host_offload_backend`. Three symmetric variants are documented boundaries
rather than branches of `step`:

* **Host-memory source (`source = BlockLocation::kHost` on `NotifyForRead` /
  `TransferSendSession::Create`).** `StartPush` skips staging and D2H, seeds
  `d2h_layer_futures_` OK, and calls `SendNextLayer(0)` directly. That state
  matches the device-source model after `Send.tr_host_source_prefix`
  (`Send.trace_host_source_prefix`), so all `Send` safety and settlement
  theorems carry over.
* **Host-memory destination (`destination = BlockLocation::kHost` on `StartRead`
  / `TransferReceiveSession::Create`).** `StartRead` writes directly to
  `local_host_block_ids` (no staging lease) and `OnDataReceived` routes layer
  completions to `SettleHostLayer` instead of `ExecuteLayerH2d` (leaving
  `decodeHbm` untouched; `Transfer/Session.lean` governs handshake and push
  leases unchanged). `CheckHostMemoryTransferSupported` restricts both `kHost`
  paths to a full host mirror, so `LayerStaging` never meets `kHost`.
* **Plan-registered sends (`RegisterActivePlan` `is_sender` →
  `TransferSendSession::CreateFromActivePlan`, Send A9).** Registers no blocks
  for pull claims, sets `deadline() = time_point::max()`, and is erased from
  `sessions_` by `UnregisterActivePlan` before `Finish(CancelledError)` (never
  published by poll). External plan readers (`PlanHostBlocks` / `GetBlockChunks`)
  take no session op (`TryBeginOp` returns `false`), so draining reads before
  `UnregisterActivePlan` is the registrant's obligation.

### Engine boundary (above `poll_stats()`)

Modelled: `NotifyForRead`, `StartRead`, and `CompleteReadWithDetails` →
`done_sending` / `done_recving` / `failed_recving`. What tpu-sync owes the
engine is HBM/staging quietness (§What is proved); when the engine may recycle
buffers is `MultiRequest.lean` (`.recyclePrefill` after send publication,
`.nextRequest` after receive publication, checked by `wroteReleased`). What the
engine owes a publication is `Caller.lean` (`connectorTwoChannel` over
single-worker and `KVOutputAggregator` multi-worker configurations in
`Caller/MultiWorker.lean`, plus intra-worker `NumaAwareKVCacheManager::CompleteReadRaw`
`settledCounts` barrier aggregation across sub-managers in
`Caller/NumaBarrier.lean`).

Not modelled: Python bootstrap/proxy servers, scheduler queues, external block
allocators, engine-side timeouts, caller-initiated cancel (sessions end only on
deadline, transport error, shutdown, or `UnregisterActivePlan`), and `Caller`
B2–B3 (partial-prefix load failures, preemption, prefix caching, chunked prefill).

### Transport boundary (below the session callbacks)

Modelled behind `TransferSession` (`tpu_sync/transport/transfer_session.h`) via
`KVCacheManagerBase::GetSession(uuid)` (`hooks.get_session` → `sessions_`):

* **Incoming-push op lease:** `BeginIncomingPush` (`GetSession` + `TryBeginOp`)
  and `EndIncomingPush` (`Finish(status)` on error, then `EndOp`) in `bts.cc`
  around `StartStream`, pinning staging across expiry (refusing send sessions
  with `CancelledError`).
* **Bounded host layer staging:** `GetBlockChunks` and `ValidateChunks` →
  `GetBlockArrayHostSpan` → `ResolveBlockArrayHostLayer` → `ResolveStagingSlot`
  (`AcquireStagingSlot` / `LookupStagingSlot`), proved in
  `PrefillDecode/LayerStaging.lean` and `Transfer/LayerStaging.lean`.
* **`OnDataReceived` dispatch and network completion:** `OnLayerReceived` →
  `ExecuteLayerH2d` precedes `OnBlockShardsReceived` → `RecordBlockShards` on
  the completing stream (`Receive` A4), and `ExecuteLayerH2d`'s third `mu_`
  scope also sets `network_completed_` (`Recv.netIssued`) once
  `dispatched_layers_` reaches `num_layers()`. Plan-declared pushes call
  `session->OnDataReceived(l, {}, {})` directly (rejected with
  `FailedPrecondition` by a `TransferReceiveSession` on a pool plan).
* **Pull readiness, block plans, worker pool, and stream accounting:**
  `BlockTransportServerPullSession::TriggerNextSendStep` waiting on
  `GetDataReadyFuture(l)`, within-layer block plans (`BlockOrdering.lean`),
  control-plane `push_pool_` (`PeerIsolation.lean`), and per-uuid
  `BlockTransportServerPushSession` stream accounting (`Transport.lean`).

Not modelled: raw socket framing (`ReadExact` / `WriteExact`, headers,
connection pooling, `memcpy`), collapsed into atomic `land l`, `h2hDone l ok`,
and `cancel` + `pushEnd`; and the opt-in Peregrine data plane
(`TPU_RAIDEN_ENABLE_PEREGRINE` / `TransportBackend::kPeregrine`, default off:
sender `BlockTransport::AsyncPush` → `PeregrineTransportAdapter::PostExplicitPush`,
and receiver `BlockTransport::ResolveRecvBuffer` / `OnRecvComplete` →
`BlockTransportServerPushSession::OnPeregrinePostComplete` with `is_peregrine = true`
under a `PostLease` from `BeginIncomingPush` / `EndIncomingPush`).

## Assumptions and the status of each

Every module numbers its assumptions in its docstring; this table indexes each
assumption and how it is discharged or scoped.

| Assumption | One-line statement | Status |
|---|---|---|
| Pipeline A4 — engine contract | Decode reads destination HBM only after `done_recving`; prefill frees source HBM only after send publication. | External contract; encoded on `.recyclePrefill` / `.nextRequest` in `MultiRequest.lean`, verified on the caller side in `Caller.lean`. |
| Receive A4 / Pipeline A3 — transport ordering | On the stream completing a layer, `ExecuteLayerH2d` runs before `RecordBlockShards` on the same thread; blocks are accounted once the copy is issued. | Ordering: C++ control flow in `StartStream`. Multi-stream coarsening: `Transport.NetworkSound` under `Transport` A4 (mutant: `Recv.netAccountUnordered`). Needed only for `RecordBlockShardsReceivedLocked`, not `ExecuteLayerH2d`'s `netIssued` setter. |
| Receive A1 — one dispatch per layer | `CompleteIncomingPush` fires `OnLayerReceived` at most once per layer, retiring the push session at `all_layers_called`. | Discharged by `Transport.OnceDispatched` under `Transport` A2, A4, A5 and `Cfg.WF.chunkSingle` (enforced by single-endpoint `StartRead`; caller obligation on `RegisterRecv` and plans without `shard_push_schedules`; exceptions: `trace_dup_after_retirement`, `trace_chunk_two_senders`). |
| Receive A2 — registration | `total_blocks_ > 0`, seated in `sessions_`, and `ReleaseStaging` runs only in `~TransferReceiveSession`. | C++ control flow (candidates rejected by `EmplaceRecvSessionLocked` are never shared); `sessions_` modelled in `UuidTable.lean`. |
| Receive A3 / Send A4 — no double end | Each `in_flight_` op ends at most once. | C++ callback and RAII discipline. |
| Send A1 — at-most-once `StartPush` | `StartPush` runs only after `NotifyForRead` and `ValidateAndBeginPull`'s `pull_started_` claim. | Discharged: `Send.PullClaimed`, `Pipeline.HandshakeSafe`, `BlockOrdering.validateRequestedBlocks_iff`. |
| Send A2, A6 — sequential D2H loop and `SendNextLayer(0)` fold | One thread creates all D2H futures, calls `SendNextLayer(0)`, then issues copies in layer order. | C++ control flow (`TransferSendSession::StartPush`). |
| Send A3 — layers interchangeable | `Send.lean` counts copies and pushes without layer indices. | Discharged: `Pipeline.lean` indexes every memory event by layer. |
| Send A5 — no consumer Ack | `SendAck` has no non-test caller, so `HandleAck → AckSend → Finish()` is excluded. | Dead path. |
| Send A7 — connection-release window | Under `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer`, the last `EndOp` drops `mu_` for `ReleaseDataConnectionsAfterSend` before `EndSendOpLocked`. | Soundness argument in `Send.lean` (`in_flight_ == 1`, `draining_`, `!done_` freeze all modelled state). |
| Send A8 — no incoming push on a send | `TransferSendSession::TryBeginOp` returns `false` and `OnDataReceived` returns `FailedPreconditionError`. | Enforced by C++ (`TransferSendSession` / `BeginIncomingPush`). |
| Send A9 — plan-registered sends excluded | `CreateFromActivePlan` sends (`RegisterActivePlan` `is_sender`) never run `StartPush` and are erased unreported by `UnregisterActivePlan`. | Scope limit (§Boundaries); registrant must finish external plan reads before `UnregisterActivePlan`. |
| Pipeline A1 — whole-layer cells | A layer moves as one unit. | Discharged: `BlockOrdering.BlockPipeline.step_pipe` (lockstep block simulation). |
| Pipeline A2 — delivery ordered after push callback | Model lands a layer at or after `h2hDone`; C++ lands before the ack. | Soundness argument in `Pipeline.lean` (the push op pins `prefillStaging[l]` across `[h2hIssue, h2hDone]`). |
| LayerStaging LS1–LS2 — single consumer stream per layer; atomic `Shutdown` + `Release` | LS1: at most one push stream per consumer layer calls `AcquireStagingSlot(l)` before `CompleteIncomingPush`. LS2: `releaseSync` combines `Shutdown()` and `Release(l)`. | LS1: configuration obligation (`parallelism == 1` or single-block transfer with one NUMA shard group; `AsyncPushShards` otherwise opens `P * num_groups` streams). LS2: exact on callback errors; C++ soundness argument in `LayerStaging.lean` §Assumptions for synchronous dispatch failures (`Release(l)` before `FinishLocked`). |
| Transport A1–A5 — lock, latched contract, uniform declaration, honest senders, fresh `uuid` | Progress mutated under push-session `mu_`; contract latched on first stream; uniform per-sender stream count; honest stream counts; single generation per `uuid`. | `Transport.lean` §Assumptions: A2 holds when senders wait for `RegisterActivePlan` (as on controller `COMMAND_START_TRANSFER` hops); A3 is unchecked on legacy non-pool `RegisterActivePlan` plans; A4 is dropped in dishonest-sender traces; A5 excludes cross-generation `ForgetPushProgress(uuid)` races. |
| Caller B1–B3 — worker/NUMA thresholds, whole-transfer failure, no preemption/prefix-cache/chunked-prefill | Multi-worker `KVOutputAggregator` (`expected = 2`) and intra-worker `NumaAwareKVCacheManager` (`1` or `2` sub-managers); single-outcome failure; static block set per transfer. | `Caller.lean` §Assumptions: `Caller/MultiWorker.lean` (`mw_contract_safe`) and `Caller/NumaBarrier.lean` (`numa_settledCounts_safe` under one report per sub-manager, N1); `> 2` NUMA sub-managers under `ENABLE_MULTI_NUMA=1` not proved. |

## Shipping C++ vs. modelled alternatives

Two modules parameterize both the **shipping C++** and an **alternative**
(`stepM` in `ReceivePoll.lean`; `NumaPolicy` in `Caller/NumaBarrier.lean`).
Elsewhere, `Backend.tcpBlocking` and `Backend.grpcAsync` in `PeerIsolation.lean`
are both shipping backends, while `PeerIsolation.stepPerPeerQuota` and
`Transport.stepHardened` sit alongside `step` as non-shipping wrapper systems.

| Module | Parameter | Shipping C++ behavior | Modelled alternative (not in shipping C++) |
|---|---|---|---|
| `PrefillDecode/ReceivePoll.lean` | `stepM (poll : Bool)` | **`stepM true`**: `CompleteReadWithDetails` calls `Finish()` on `IsReadyToComplete()`. Safe, but firing between ready futures and the last H2D callback skips end-of-transfer metrics (`trace_poll_skips_metrics`). | **`stepM false`**: only the last H2D callback finishes positive-layer receives (zero-layer receives would finish at creation, like sends). Preserves safety and settlement (`noPoll_safe`, `noPoll_can_settle`) and guarantees `noPoll_metrics_on_success`. |
| `Caller/NumaBarrier.lean` | `NumaPolicy` | **`settledCounts`**: `NumaAwareKVCacheManager::CompleteReadRaw` counts `done_sending_counts_`, `done_recving_counts_`, and `failed_recving_counts_` per `req_id`, waits until their sum reaches `req_expected_counts_` (or `sub_managers_.size()`), emits `failed_recving` if any failed, and erases all four map entries (`numa_settledCounts_safe`, `numa_settledCounts_maps_clean`; 8 `KVCacheManagerWrapperTest`s replayed). | **`eagerFailSet`** (mutant): emits `failed_recving` on the first failing sub-manager via `failedRecvingSet`, freeing HBM while a sibling sub-manager's DMA is still in flight (`trace_eagerFailSet_recv_early_free`, `trace_eagerFailSet_send_early_free`) and leaving stale map entries on `req_id` reuse (`trace_eagerFailSet_single_sub_reuse_leaks`). |

## Counterfactual mutants

Each mutant removes a C++ guard, ordering constraint, or assumption and yields a
`.counterexample` or refutation trace; the two rows marked "alternative" add a
check shipping C++ lacks:

| Mutant | C++ guard / invariant removed | Property violated |
|---|---|---|
| `Recv.cancelEager`, `Send.cancelEager` | `in_flight_ == 0` in `FinishLocked`'s settle condition: `cancel` settles at once | `SettleSafe` (`done → inFlight = 0`), `NoRetiredCallback` / `Drained`, staging & HBM safety |
| `Recv.netAccountUnordered` | `ExecuteLayerH2d` runs before `RecordBlockShards` on the same thread in `StartStream` (Receive A4) | `ReadinessSound` (`IsReadyToComplete → ready = numLayers`) |
| `Send.trySendNextUncounted` (`sendNextUncounted` / `startUncounted`) | `++in_flight_` op taken by `SendNextLayer(0)` and `SendNextLayer(l + 1)` | `NoUnderflow` / `Drained` (`end_op` decrements `inFlight` to `0` early) |
| `Send.stepWithoutFailUnissued` (`trace_without_fail_unissued_leaks`) | `absl::Cleanup fail_unissued_layers` in `StartPush` resolving unissued `d2h_promises` on early exit | `NoOpLeak` (armed `SendNextLayer` callback hangs with `in_flight_ > 0`) |
| `Recv.h2dIssueLeak` | `EndRecvOpLocked()` on the second-lock `if (done_ \|\| draining_)` early return in `ExecuteLayerH2d` | `NoOpLeak` (leaked `in_flight_` prevents `done_`) |
| `Send.sendNextUnbounded` | `l >= base_->num_layers()` / `l >= d2h_layer_futures_.size()` bounds checks in `SendNextLayer` | `NoOpLeak` (`SendNextLayer(numLayers)` claims an op with no future) |
| `Pipeline.dispatchEarly` (`trace_dispatch_early`) | `ExecuteLayerH2d(l)` dispatched only after layer `l` landed in `CompleteIncomingPush` | `PublicationCorrect` (`decodeHbm` receives `.junk`) |
| `Pipeline.h2dReadyByRank` (`trace_h2d_by_rank`) | H2D copy for layer `l` reads staging slot `l` rather than completion-rank index | `PublicationCorrect` (out-of-order landing yields `[kv 1, junk]`) |
| `Pipeline.reseatAtFinish` (`trace_reseat_at_finish`) | Prefill staging released only at settle (`done_ = true`), not on `FinishLocked` (`draining_ = true`) | `PublicationCorrect` / `StagingSafe` |
| `LayerStaging.sysEarlySendRelease` (`mutant_early_send_release_corrupts_hbm`) | Producer host slot released in `h2hDone`, not at `H2hWriteDirectAsync` dispatch (`sendH2hIssue`) | `PublicationCorrect` (`decodeHbmB` receives `[.kv 1, .kv 1]`) |
| `LayerStaging.sysEarlyRecvRelease` (`mutant_early_recv_release_corrupts_hbm`) | Consumer host slot released in `h2dDone`, not right after `H2dSyncDispatch` (`h2dIssue`) | `PublicationCorrect` (`decodeHbmB` receives `[.kv 1, .kv 1]`) |
| `LayerStaging.sysNoReleaseAll` (`mutant_missing_release_all_leaks_slot`) | `ReleaseStagingLocked` / `ReleaseStagingBlocksLocked` call `staging_layer_manager_.ReleaseAll()` at settle | `SettledSlotsFree` (aborted push settles with `leases = [(0, 0)]` leaked) |
| `UuidTable.stepOverwriteDraining` | `EmplaceRecvSessionLocked` rejects duplicate UUIDs until `RetireSettledSessionLocked` sees `Done()` | Staging slot conservation (`freeSlots + activeStaging = numSlots`) |
| `PeerIsolation.stepReleaseOnCancel` | `TransferReceiveSession` releases its `StagingBlockAllocator` slot at settle (`in_flight_ == 0`), not on `.cancel` | `PeerIsolation.Safe` (`freeSlots + activeStaging = numSlots`) |
| `PeerIsolation.stepPerPeerQuota` (alternative, not a mutant) | adds per-peer cap `peerStaging peer < maxPerPeer` on `StartRead` (`DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`) | none within search depth (`= .outOfFuel`), vs. `starvesFirstHealthy` at depth 3 on shipping `step` |
| `BlockOrdering.trace_mutant_mismatched_transport_order` (`buildBuggyCopyPlan`) | `BuildLoadCopyPlan` permutes both `producer_remote_block_ids` and `transport_host_block_ids` by `remote_order` | Within-layer block publication correctness (`[.kv 0 0, .kv 0 1]` instead of `[.kv 0 1, .kv 0 0]`) |
| `Transport` with `Cfg.honest = false` (`trace_dup_chunk_publishes_missing_layer`, `trace_dup_after_retirement`) | `Transport` A4 (honest senders): chunk-count lacks over-count rejection, and `all_layers_called` retirement seats a fresh push session on later streams | `NetworkSound` / `CompleteOnSuccess` (missing layer published) and `OnceDispatched` (duplicate copy after retirement; `trace_dup_copy_fails_late`) |
| `Transport.stepHardened` (alternative, not a mutant) | adds chunk-count over-count rejection and a session-side per-layer once-latch in `ExecuteLayerH2d` | none within search depth (`searchHardened … = .outOfFuel`); honest traces unchanged (`trace_hardened_honest`) |
| `Caller.connectorDropFailed`, `Caller.connectorNaive`, `Caller.connectorLoadErrorOnly` | `connectorTwoChannel` forwards `failed_recving` on both P (`finished_sending`) and D (`finished_recving` + load-error channel) | `connectorDropFailed` leaks P/D blocks; `connectorNaive` decodes unwritten KV; `connectorLoadErrorOnly` leaks D blocks |

## C++ → Lean routing index

Per C++ symbol: the Lean module that owns it, the model events or definitions
it becomes, and the theorems to open first. The `Lines` column is generated by
`tools/resolve_anchors.py`.

### Receive session and slot manager (`staging.{h,cc}`, `recv.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `staging.h` | `StagingLayerManager` | `:39-74` | `Transfer/LayerStaging.lean`, `PrefillDecode/LayerStaging.lean` | `StagingLayerManager` (`free`, `leases`, `shutdown`, `acquire`, `lookup`, `release`, `releaseAll`, `doShutdown`) | `StagingLayerManager.lookup_injective`, `lookup_lt`, `acquire_idempotent`, `shutdown_cancels`, `LayerStaging.reachable_safe` |
| `tpu_sync/transport/transfer_session.h` | `TransferSession` | `:36-115` | `Transfer/Session.lean` | common session interface (`Start`, `Done`, `Finish`, `TryBeginOp`, `EndOp`, `IsDraining`, `ResolveStagingSlot`, `OnDataReceived`, `GetDataReadyFuture`, `plan`, `deadline`) | `Lifecycle.Consistent` |
| `recv.cc` | `TransferReceiveSession::TryBeginOp` | `:456-467` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `Lifecycle.beginOp`, `Recv.pushBegin` (`BeginIncomingPush`) | `Lifecycle.Consistent`, `Recv.reachable_safe` |
| `recv.cc` | `TransferReceiveSession::EndOp` | `:469-472` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `Recv.pushEnd` (`EndIncomingPush`) | `Recv.StagingIntegrity`, `Recv.trace_push_lease_pins_staging_on_cancel` |
| `recv.h` | `TransferReceiveSession` (`std::vector<int64_t> blocks_received_per_shard_` .. `std::vector<raiden::PjRtCopyFuture> h2d_futures_`) | `:329-357` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `in_flight_`, `draining_`, `done_`, `failed_phase_`, `handshake_pending_`, `blocks_received_per_shard_`, `num_completed_layers_`, `network_completed_`, `h2d_futures_` | `Recv.reachable_safe` (`SettleSafe`, `StagingIntegrity`, `SettlesPromptly`) |
| `recv.cc` | `TransferReceiveSession::Create(remote_block_ids)` | `:81-123` | `PrefillDecode/Receive.lean`, `PrefillDecode/BlockOrdering.lean` | `Recv.initLoad` (`kDevice` default; `kHost` in §Boundaries) | `BlockOrdering.landStage_get_requested` |
| `recv.cc` | `TransferReceiveSession::Create(total_blocks)` | `:125-137` | `PrefillDecode/Receive.lean` | `Recv.initPush` | `Recv.reachable_safe` |
| `recv.cc` | `TransferReceiveSession::InitFromActivePlan` | `:191-243` | `PrefillDecode/Receive.lean` | `Recv.initPush` (`handshake_pending_ = false`, `in_flight_ = 0`) | `Recv.reachable_safe`, `Recv.trace_normal` |
| `recv.cc` | `TransferReceiveSession::AllocateStagingForLoad` | `:245-294` | `PrefillDecode/BlockOrdering.lean` | custom vs. dynamic host staging | `BlockOrdering.BlockPipeline.CustomHostStagingCorrect` |
| `recv.cc` | `TransferReceiveSession::BuildLoadCopyPlan` | `:296-364` | `PrefillDecode/BlockOrdering.lean` | `buildCopyPlanFromTriples` (`remoteSorted`, `localSorted`, `h2dPairs`) | `BlockOrdering.landStage_get_requested`, `BlockOrdering.h2dStage_get_requested`, mutant `trace_mutant_mismatched_transport_order` |
| `recv.cc` | `TransferReceiveSession::InitFromLoadPlan` | `:366-378` | `PrefillDecode/Receive.lean` | `Recv.initLoad` (`handshake_pending_ = true`, `in_flight_ = 1`) | `Recv.reachable_safe` |
| `recv.cc` | `TransferReceiveSession::ReleaseStagingLocked` | `:380-383` | `Transfer/Session.lean`, `PrefillDecode/LayerStaging.lean` | `Lifecycle.settleLocked`, `LayerStaging.syncMgr` (`releaseAll`) | `Recv.StagingIntegrity`, `LayerStaging.SettledSlotsFree`, mutant `mutant_missing_release_all_leaks_slot` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:390-405` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean`, `PrefillDecode/LayerStaging.lean` | `Lifecycle.finishLocked`, `Recv.failedPhase`, `LayerStaging.syncMgr` (`doShutdown`) | `Recv.reachable_safe`, `LayerStaging.DrainingShutsDown`, mutant `Recv.cancelEager` |
| `recv.cc` | `TransferReceiveSession::Details` | `:418-434` | `PrefillDecode/Receive.lean` | `TransferDetails` reporting | `Recv.trace_details_*` |
| `recv.cc` | `TransferReceiveSession::CurrentPhaseLocked` | `:436-441` | `PrefillDecode/Receive.lean` | `Recv.Phase`, `Recv.phase` | `Recv.trace_details_*` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:443-454` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `Lifecycle.endOpLocked` | `Recv.SettleSafe`, `Recv.SettlesPromptly` |
| `recv.cc` | `TransferReceiveSession::AllH2dDoneLocked` | `:474-479` | `PrefillDecode/Receive.lean` | `Recv.allH2dDone` | `Recv.ReadinessSound` |
| `recv.cc` | `TransferReceiveSession::IsReadyToComplete` | `:481-487` | `PrefillDecode/Receive.lean`, `PrefillDecode/ReceivePoll.lean` | `Recv.isReadyToComplete`, `stepM poll` | `Recv.ReadinessSound`, `Recv.pollReady_window`, `Recv.trace_poll_skips_metrics` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShardsReceivedLocked` | `:496-521` | `PrefillDecode/Receive.lean`, `PrefillDecode/Transport.lean` | `Recv.networkCompleted`, `Recv.Ev.netAccount` | `Recv.ReadinessSound`, mutant `Recv.netAccountUnordered` |
| `recv.cc` | `TransferReceiveSession::RecordNetworkCompleteLocked` | `:523-539` | `PrefillDecode/Receive.lean`, `PrefillDecode/ReceivePoll.lean` | `Recv.Ev.netAccount` finish & metrics | `Recv.reachable_callbackFinishes` |
| `recv.cc` | `TransferReceiveSession::Start` | `:541-627` | `PrefillDecode/Receive.lean`, `PrefillDecode/PeerIsolation.lean` | pull handshake on `push_pool_` (`on_response` checks `!self->draining_` before error `FinishLocked`, then `EndOp`): `Recv.Ev.pullReply`, `PeerIsolation.Ev.dispatchPull` | `Recv.trace_deadline_during_handshake`, `Recv.trace_details_rejected_pull`, `PeerIsolation.trace_consumer_gives_up_and_drains` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShards` | `:629-678` | `PrefillDecode/Receive.lean`, `PrefillDecode/ReceivePoll.lean` | `Recv.Ev.netAccount` | `Recv.netAccount_frame`, `Recv.trace_late_net_account_after_retire` |
| `recv.cc` | `TransferReceiveSession::OnDataReceived` | `:721-740` | `PrefillDecode/Receive.lean` | routes block reports to `RecordBlockShards` and layer completions to `ExecuteLayerH2d` (or `SettleHostLayer` on `kHost`, §Boundaries) | `Recv.ReadinessSound` |
| `recv.cc` | `TransferReceiveSession::SettleHostLayer` | `:742-769` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `BlockLocation::kHost` layer completion (`RecordNetworkCompleteLocked` on last layer; §Boundaries) | `Lifecycle.Consistent.done_or_draining` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:680-709` | `PrefillDecode/LayerStaging.lean` | `recvAcquireBegin`, `recvAcquireWait`, `recvAcquireFinish` | `LayerStaging.reachable_safe`, `recvLookupStagingSlot_injective`, `reachable_can_settle` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:771-903` | `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/LayerStaging.lean` | `h2dBegin`, `h2dIssue` (sets `Recv.netIssued` via `dispatched_layers_` on last layer), `h2dReady`, `h2dDone`, `recvLookupStagingSlot` | `Recv.NoOpLeak`, `Recv.NoRetiredCallback`, `Pipeline.DecodeHbmSafe`, `LayerStaging.PublicationCorrect`, mutants `Recv.h2dIssueLeak`, `mutant_early_recv_release_corrupts_hbm` |
<!-- source-map:end -->

### Send session (`send.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `send.h` | `TransferSendSession` (`absl::string_view failed_phase_` .. `bool done_`) | `:238-246` | `Transfer/Session.lean`, `PrefillDecode/Send.lean` | `failed_phase_`, `in_flight_`, `draining_`, `done_` | `Send.reachable_safe` (`SettleSafe`, `Drained`, `StagingIntegrity`, `NoUnderflow`) |
| `send.cc` | `TransferSendSession::Create` | `:65-91` | `PrefillDecode/Send.lean` | `Send.init` (`kDevice` default; `kHost` in §Boundaries) | `Send.reachable_safe` |
| `send.cc` | `TransferSendSession::CreateFromActivePlan` | `:93-144` | `PrefillDecode/Send.lean` | unmodelled plan-registered send (Send A9, §Boundaries) | — |
| `send.cc` | `TransferSendSession::PopulateRegisteredBlocks` | `:163-179` | `PrefillDecode/BlockOrdering.lean` | `validateRegistration` | `BlockOrdering.validateRegistration_iff` |
| `send.cc` | `TransferSendSession::ValidateRequestedBlocksLocked` | `:181-198` | `PrefillDecode/BlockOrdering.lean` | `validateRequestedBlocks` | `BlockOrdering.validateRequestedBlocks_iff` |
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:200-214` | `PrefillDecode/Send.lean`, `PrefillDecode/Pipeline.lean` | `Send.Ev.beginPull` (`pull_started_` claim) | `Send.PullClaimed`, `Pipeline.HandshakeSafe` |
| `send.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:225-229` | `Transfer/Session.lean`, `PrefillDecode/LayerStaging.lean` | `Lifecycle.settleLocked`, `LayerStaging.syncMgr` (`releaseAll`) | `Send.StagingIntegrity`, `LayerStaging.SettledSlotsFree` |
| `send.cc` | `TransferSendSession::AcquireStagingSlot` | `:236-265` | `PrefillDecode/LayerStaging.lean` | `sendAcquireBegin`, `sendAcquireDone`, `sendD2hBegin` | `LayerStaging.reachable_safe`, `reachable_can_settle` |
| `send.cc` | `TransferSendSession::LookupStagingSlot` | `:267-286` | `PrefillDecode/LayerStaging.lean` | `sendLookupStagingSlot` | `LayerStaging.sendLookupStagingSlot_injective`, `NoSlotFault` |
| `send.cc` | `TransferSendSession::FinishLocked` | `:288-303` | `Transfer/Session.lean`, `PrefillDecode/Send.lean`, `PrefillDecode/LayerStaging.lean` | `Lifecycle.finishOnceLocked`, `Send.Ev.cancel`, `LayerStaging.syncMgr` (`doShutdown`) | `Lifecycle.finishOnceLocked_decided`, `LayerStaging.DrainingShutsDown`, mutant `Send.cancelEager` |
| `send.cc` | `TransferSendSession::Details` | `:330-350` | `PrefillDecode/Send.lean` | `TransferDetails` reporting | `Send.trace_details_*` |
| `send.cc` | `TransferSendSession::CurrentPhaseLocked` | `:352-362` | `PrefillDecode/Send.lean` | `Send.Phase`, `Send.phase`, `Send.failedPhase` | `Send.trace_details_*` |
| `send.cc` | `TransferSendSession::EndSendOpLocked` | `:364-371` | `Transfer/Session.lean`, `PrefillDecode/Send.lean` | `Lifecycle.endOpLocked` | `Send.SettleSafe`, `Send.Drained` |
| `send.cc` | `TransferSendSession::EndOp` | `:373-400` | `PrefillDecode/Send.lean` | `EndOp` (`per_transfer` connection-close window, Send A7) | `Send.reachable_safe` |
| `send.cc` | `TransferSendSession::GetDataReadyFuture` | `:216-223` | `PrefillDecode/Send.lean` | per-layer D2H future waited on by `TriggerNextSendStep` (`d2hReady`) | `Send.reachable_can_settle` |
| `send.cc` | `TransferSendSession::BuildCoalescedCopySpec` | `:402-431` | `PrefillDecode/BlockOrdering.lean` | `buildCoalescedSpec` | `BlockOrdering.execCoalesced_buildCoalescedSpec` |
| `send.cc` | `TransferSendSession::StartPush` | `:469-640` | `PrefillDecode/Send.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/LayerStaging.lean` | `start` (`d2h_promises` + `SendNextLayer(0)`; `kHost` arm in §Boundaries), `sendAcquire*`, `sendD2hBegin`, `d2hIssue`, `d2hReady`, `d2hEnd`, `fail_unissued_layers` | `Send.reachable_safe`, `Pipeline.PrefillHbmSafe`, `LayerStaging.exists_complete_trace`, mutants `Send.startUncounted`, `Send.stepWithoutFailUnissued` |
| `send.cc` | `TransferSendSession::SendNextLayer` | `:642-741` | `PrefillDecode/Send.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/LayerStaging.lean` | `wake`, `h2hIssue`, `sendNext`, `h2hDone` | `Send.reachable_can_settle`, `LayerStaging.PublicationCorrect`, mutants `Send.sendNextUncounted`, `Send.sendNextUnbounded`, `mutant_early_send_release_corrupts_hbm` |
<!-- source-map:end -->

### Manager (`mgr.cc`, `base.cc`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` | `:305-309` | `PrefillDecode/UuidTable.lean` | wires `hooks.get_session` | — |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` | `PrefillDecode/UuidTable.lean` | `sessions_.find(uuid)` under `mu_` (`NotFound` when absent) | `UuidTable.reachable_inv` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Receive.lean`, `PrefillDecode/Send.lean`, `Transfer/Caller.lean` | `publish`, `sweepSend`, `sweepRecv`: publishes `Done()` entries under `SessionEntry::req_id` and erases from `sessions_` | `Recv.Publication`, `Send.Publication`, `UuidTable.trace_duplicate_uuid_rejected_until_drained` |
| `base.cc` | `KVCacheManagerBase::GetSession` | `:3081-3087` | `PrefillDecode/Transport.lean` | transport delegate lookup (`transfer_hooks_.get_session`) | — |
| `base.cc` | `KVCacheManagerBase::OnLayerReceived` | `:3345-3354` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean` | `h2dBegin l` (`OnDataReceived(layer_idx, {}, {})`) | `Pipeline.dispatchEarly` |
| `base.cc` | `KVCacheManagerBase::OnBlockShardsReceived` | `:3323-3341` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean` | `netAccount` (`OnDataReceived(0, block_ids, shard_ids)`, non-empty only) | `Recv.ReadinessSound` |
| `bts.cc` | `BlockTransportServerPullSession::TriggerNextSendStep` | `:1055-1151` | `PrefillDecode/Send.lean` | pull server's `GetDataReadyFuture(l)` readiness wait (`d2hReady`) | — |
| `mgr.cc` | `KVCacheManagerWithTransfer::NotifyForRead` | `:532-603` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/BlockOrdering.lean` | `notifyForRead`, `validateRegistration`, `sessions_.try_emplace` (`kHost` in §Boundaries) | `UuidTable.active_send_preserved`, `BlockOrdering.validateRegistration_iff` |
| `mgr.cc` | `KVCacheManagerWithTransfer::EmplaceRecvSessionLocked` | `:605-626` | `PrefillDecode/UuidTable.lean` | drain-before-reuse on `sessions_[uuid]` | `UuidTable.active_recv_preserved`, mutant `UuidTable.stepOverwriteDraining` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterActivePlan` | `:628-707` | `PrefillDecode/UuidTable.lean` | receiver arm: `UuidTable.Ev.registerRecv`; `is_sender` arm unmodelled (Send A9) | `UuidTable.reachable_inv` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:828-874` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Receive.lean` | receiver arm: `Recv.Ev.cancel`; send arm erases unreported before `Finish` (Send A9) | `Recv.reachable_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterRecv` | `:709-731` | `PrefillDecode/UuidTable.lean` | `UuidTable.Ev.registerRecv` | `UuidTable.reachable_inv` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:942-1074` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/PeerIsolation.lean` | `startRead`, `stepPerPeerQuota`, `SessionEntry::req_id` idempotency (`kHost` in §Boundaries) | `UuidTable.trace_duplicate_receive_different_req_id`, `PeerIsolation.trace_sick_peer_starves_staging_slots` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1076-1139` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Receive.lean`, `PrefillDecode/Send.lean`, `Transfer/Caller.lean` | sweeps `sessions_`: `pollReady`, `deadline() <= now` `cancel`, `RetireSettledSessionLocked` (`publish`) | `Send.trace_deadline_during_copy`, `Recv.trace_deadline_during_copy`, `Caller.contract_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadRaw` | `:1142-1147` | `PrefillDecode/UuidTable.lean` | wrapper around `CompleteReadWithDetails` | `Recv.ReadinessSound` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1229-1358` | `PrefillDecode/Pipeline.lean`, `PrefillDecode/PipelineChecks.lean` | `Pipeline.Ev.pullWait`, `Send.Ev.beginPull`, `Send.Ev.cancel` | `Pipeline.HandshakeSafe`, `Pipeline.reachable_unregistered_safe`, `Pipeline.trace_pull_ahead_of_registration` |
| `base.cc` | `ResolveNumStagingLayers(std::optional<int> num_staging_layers)` | `:718-735` | `Transfer/LayerStaging.lean` | `Transfer.resolveNumStagingLayers` | `Transfer.resolveNumStagingLayers_bounds` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:3112-3120` | `PrefillDecode/Transport.lean` | sender-declared vs. chunk-count contract | `Transport.trace_chunk_two_senders`, `Transport.trace_dup_declared_rejected` |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` | `PrefillDecode/LayerStaging.lean` | resolves `host_layer_idx` via `GetSession(uuid)` → `ResolveStagingSlot` | `LayerStaging.reachable_safe`, `NoSlotFault` |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` | `PrefillDecode/LayerStaging.lean` | host buffer span lookup for `ValidateChunks` | `LayerStaging.reachable_safe`, `NoSlotFault` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`size_t host_layer_idx = layer_idx;` .. `base_host_ptr = host_buffer->data();`) | `:3498-3512` | `PrefillDecode/LayerStaging.lean` | `ResolveStagingSlot`: `sendLookupStagingSlot`, `recvAcquire*` | `LayerStaging.reachable_safe`, `NoSlotFault` |
| `tpu_sync/frameworks/jax/kv_cache_manager.h` | `NumaAwareKVCacheManager` (`std::map<std::string, int> done_sending_counts_;` .. `std::map<std::string, int> req_expected_counts_;`) | `:295-298` | `Transfer/Caller.lean` | `NumaMgr` per-request outcome counters | `Caller.numa_settledCounts_maps_clean` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::StartRead(remote_descriptors)` | `:580-624` | `Transfer/Caller.lean` | `NumaCfg.setExpected = true`, `numaInitDExpected` | `Caller.numa_settledCounts_safe` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::StartRead(remote_endpoint)` | `:626-641` | `Transfer/Caller.lean` | `NumaCfg.setExpected = false`, `numaInitDExpected` | `Caller.numa_settledCounts_safe` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::CompleteReadRaw` | `:644-706` | `Transfer/Caller.lean` | `NumaPolicy.settledCounts` (`process_completion`); mutant `NumaPolicy.eagerFailSet` | `Caller.numa_settledCounts_safe`, `Caller.numa_settledCounts_maps_clean`, mutants `Caller.trace_eagerFailSet_*` |
<!-- source-map:end -->

### Transport (`bt.{h,cc}`, `bts.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `bts.h` | `BlockTransportServerPushSession` (`struct LayerProgress` .. `absl::flat_hash_map<int, LayerProgress> layer_progress_ ABSL_GUARDED_BY(mu_);`) | `:158-201` | `PrefillDecode/Transport.lean` | `LayerProgress` / `layer_progress_` / `done_` under push-session `mu_` | `Transport.reachable_safe` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean` | `pushBegin`: `GetSession(uuid)` + `TryBeginOp()` | `Recv.StagingIntegrity`, `Recv.trace_push_lease_pins_staging_on_cancel` |
| `bts.cc` | `EndIncomingPush` | `:258-271` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean` | `pushEnd`: `Finish(status)` on error, then `EndOp()` | `Recv.StagingIntegrity`, `Pipeline.StagingSafe` |
| `bts.cc` | `ValidateChunks` | `:275-310` | `PrefillDecode/Transport.lean`, `PrefillDecode/LayerStaging.lean` | per-chunk span check via `GetBlockArrayHostSpan` → `ResolveStagingSlot` | `LayerStaging.reachable_safe`, `NoSlotFault` |
| `bt.h` | `BlockTransport` (`absl::Mutex server_sessions_mu_;` .. `active_push_sessions_ ABSL_GUARDED_BY(server_sessions_mu_);`) | `:310-313` | `PrefillDecode/Transport.lean` | uuid → push-session map (`GetOrCreatePushSession`, `ForgetPushProgress`) | — |
| `bt.cc` | `BlockTransport::GetOrCreatePushSession` | `:432-443` | `PrefillDecode/Transport.lean` | creates or returns live push session; seats fresh session after retirement | `Transport.reachable_safe` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` | `PrefillDecode/Transport.lean` | `GetOrCreatePushSession`, `StartStream`, then `DeletePushSession` if `IsDone()` | — |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean` | `pushBegin`, `land l`, `CompleteIncomingPush`, `OnBlockShardsReceived`, `pushEnd`; `Transport.Ev.stream l s` | `Recv.StagingIntegrity`, `Pipeline.StagingSafe` |
| `bt.cc` | `BlockTransport::OnRecvComplete` | `:485-497` | `PrefillDecode/Transport.lean` | unmodelled Peregrine receive entry point (§Boundaries) | — |
| `bts.cc` | `BlockTransportServerPushSession::OnPeregrinePostComplete` | `:663-713` | `PrefillDecode/Transport.lean` | unmodelled Peregrine `is_peregrine = true` completion path (§Boundaries) | — |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean` | `Transport.Ev.stream l s`, `completePush`, `retire`, `nextLayer` | `Transport.reachable_safe`, `trace_dup_chunk_publishes_missing_layer`, `trace_dup_after_retirement`, `stepHardened` |
| `bt.cc` | `BlockTransport::ForgetPushProgress` | `:1075-1089` | `PrefillDecode/Transport.lean` | erases and `Cancel()`s `active_push_sessions_[uuid]` outside manager `mu_` (`Transport` A5) | — |
<!-- source-map:end -->

### Control plane

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `tcp_control_plane_backend.cc` | `TcpControlPlaneBackend::SendPullRequestBlocking` | `:592-660` | `PrefillDecode/PeerIsolation.lean` | `Backend.tcpBlocking` | `PeerIsolation.tcp_healthy_blocked_when_pool_full` |
| `grpc_control_plane_backend.cc` | `GrpcControlPlaneBackend::SendPullRequest` | `:308-336` | `PrefillDecode/PeerIsolation.lean` | `Backend.grpcAsync` | `PeerIsolation.grpc_healthy_can_complete`, `PeerIsolation.grpc_freeWorkers_eq_poolSize` |
<!-- source-map:end -->

*External caller pair (`Transfer/Caller.lean`):*
tpu-inference `tpu_inference/distributed/tpu_raiden_connector.py` and vLLM
`vllm/distributed/kv_transfer/kv_connector/utils.py`,
`vllm/v1/core/sched/scheduler.py` (`TPUConnectorWorker.get_transfer_results`,
`get_finished`, `KVOutputAggregator.aggregate`, scheduler
`_handle_invalid_blocks`, `_handle_failed_recving`, `_update_from_kv_xfer_finished`,
`finish_requests`, `kv_load_failure_policy`) → `Caller.contract_safe`,
`Caller.mw_contract_safe`, `Caller.twoChannel_contract`,
`Caller.applyRecv_eager_eq_latched`, `Caller.leakP_absorbing`,
`Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing`.

## Where the rest lives

| Content | File |
|---|---|
| `trace_*` witnesses and replayed tests; `#guard` bounded searches | theorem docstrings and `#guard` lines in `TpuSyncVerify/Transfer/` |
| modelling conventions (executable `step`, ghost fields, replay, mutants) | [conventions.md](../conventions.md) |

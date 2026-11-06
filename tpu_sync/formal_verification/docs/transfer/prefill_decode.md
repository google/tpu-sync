# Prefill-to-decode transfer: model and results

The reference map from the prefill-to-decode KV-cache transfer C++ to its Lean
models under `TpuSyncVerify/Transfer/`.

Each module's docstring owns its state table, event table, assumptions,
properties and `## Source map`; this page routes between them. It says what is
proved and where, lists the modules, draws the three boundaries of the model,
tabulates the assumptions and how each is discharged, sets the shipping code
beside the design alternatives modelled next to it, lists the mutants, and ends
with the per-symbol C++ → Lean index. Each `trace_*` theorem's docstring names
the C++ or end-to-end test it replays, so `grep -rn <TestName> TpuSyncVerify`
answers "is test X modelled".

File aliases: `recv.{h,cc}` = `tpu_sync/kv_cache/transfer_receive_session.{h,cc}`,
`send.{h,cc}` = `tpu_sync/kv_cache/transfer_send_session.{h,cc}`,
`mgr.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`,
`bt.{h,cc}` = `tpu_sync/transport/block_transport.{h,cc}`,
`bts.{h,cc}` = `tpu_sync/transport/block_transport_session.{h,cc}`,
`base.cc` = `tpu_sync/kv_cache/kv_cache_manager_base.cc`,
`tcp_control_plane_backend.cc` = `tpu_sync/core/tcp_control_plane_backend.cc`,
`grpc_control_plane_backend.cc` = `tpu_sync/core/grpc_control_plane_backend.cc`.

## What is proved

Every theorem below is proved on every reachable state of its model; the
statement column is the one-line form, the module docstring has the exact one.

| Property | Theorems | Statement |
|---|---|---|
| Data correctness and attention safety across requests | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe` | In any reachable state of `multiSys` (concurrent, overlapped and recycled requests), a request that publishes `done_recving` has `decodeHbm = good n` and keeps it across every later transition. |
| Progress across requests | `Pipeline.system_progress` | From any reachable state, any request can settle, return both staging buffers and publish both outcomes (`HandedOff`) in finitely many steps. |
| Publication correctness, one request | `Pipeline.reachable_safe` (`PublicationCorrect`) | `recv.published = some true → decodeHbm = good n`. |
| Within-layer blocks: gather, reordering, coalescing, custom staging | `BlockOrdering.execCoalesced_buildCoalescedSpec`, `BlockOrdering.landStage_get_requested`, `BlockOrdering.h2dStage_get_requested`, `BlockOrdering.BlockPipeline.reachable_safe` | Coalesced DMA equals elementwise copy; the two permutations of `BuildLoadCopyPlan` (`remote_order`, `local_order`) cancel, so on `done_recving` every requested block `t` of layer `l` holds `.kv l t.remote` and unrequested blocks stay blank; custom host staging holds the same until reseated. |
| Decode HBM safety and quietness | `Recv.PublishedQuiet`, `Pipeline.DecodeHbmSafe`, `decodeHbm_quiet`, `attention_safe` | Once the receive is published no H2D copy is pending or unretired, so `decodeHbm` never changes afterwards. |
| Prefill HBM safety and quietness | `Send.PublishedQuiet`, `Pipeline.PrefillHbmSafe`, `prefillHbm_quiet` | Once the send is published no D2H copy is pending or unretired, so the caller may reclaim prefill HBM. |
| Staging integrity and quietness | `Recv.StagingIntegrity`, `Send.StagingIntegrity`, `Pipeline.StagingSafe`, `wroteReleased_eq_false` | `hasStaging = !done`; released staging is neither written nor read by the session that held it. |
| Handshake rendezvous and single pull claim | `Send.PullClaimed`, `Pipeline.HandshakeSafe`, `Pipeline.reachable_unregistered_safe`, `BlockOrdering.validateRegistration_iff`, `BlockOrdering.validateRequestedBlocks_iff` | `StartPush` and all data movement need `NotifyForRead`'s registration and `ValidateAndBeginPull`'s claim of a non-empty, duplicate-free subset of the registered blocks; while `HandlePullStream` waits, the pull stays pending and unclaimed. |
| Termination | `Recv.NoOpLeak`, `Send.NoOpLeak`, `Pipeline.NoOpLeak`, `Recv.reachable_can_settle`, `Send.reachable_can_settle`, `Pipeline.reachable_can_settle` | Every unit of `in_flight_` has an enabled step that advances or retires it, and every reachable state drains to `done ∧ ¬hasStaging` and publishes. |
| Per-side counters | `Recv.Publication`, `Send.Publication`, `Recv.ReadinessSound`, `Recv.SettleSafe`, `Send.SettleSafe`, `Send.NoUnderflow` | `done_recving → every H2D callback ran OK`; `done_sending → every push completed OK`; `IsReadyToComplete → ready = numLayers`; `done → inFlight = 0`; `EndSendOpLocked` never underflows. |
| Transport accounting | `Transport.reachable_safe` (`OnceDispatched`, `NetworkSound`, `CompleteOnSuccess`) | With honest senders, `OnLayerReceived` fires once per layer, the block target is reached only after every layer was dispatched, and a successful receive copied every layer exactly once. |
| Engine caller contract (single- and multi-worker) | `Caller.contract_safe`, `Caller.mw_contract_safe`, `Caller.leakP_absorbing`, `Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing` | A connector that forwards every publication leaks no held block on either node, never frees HBM while a worker on that node is still in flight (`mwEarlyFree = false`), and never lets decode read unwritten KV, both for one worker and across workers aggregated by `KVOutputAggregator`; a missed publication is unrecoverable by any later engine action. |
| Peer isolation and uuid tables | `PeerIsolation.reachable_safe`, `PeerIsolation.tcp_healthy_blocked_when_pool_full`, `PeerIsolation.grpc_healthy_can_complete`, `UuidTable.reachable_inv` | Host staging slots and `push_pool_` workers are conserved across peers. On the blocking TCP control plane a wedged pull holds a `push_pool_` worker, so `poolSize` pulls to a sick peer stop every further handshake, healthy peers included; on the gRPC backend a waiting pull holds no worker and a healthy peer completes once admitted (while `numSlots` wedged reads still exhaust the shared staging pool). A uuid is refused until its draining incumbent has settled, so staging slots are conserved. |

## Modules

Higher modules compose the lower `step` functions as-is: `Pipeline` = `Send.step` +
`Recv.step` + per-layer memories and guards; `BlockOrdering` refines `Pipeline`'s
layer cell into a block array (`BlockPipeline.step_pipe` is a lockstep
simulation over `Pipeline.step`); `MultiRequest` runs `reqs : List Pipeline`
over the four shared buffers.

| Module | Scope | Main theorems |
|---|---|---|
| `Common/ListAux.lean` | `List` indexing / `countTrue` pigeonhole helpers | `all_true_of_countTrue_eq_length`, `exists_false_lt` |
| `Transfer/Session.lean` | settle protocol (`in_flight_`, `draining_`, `done_`, staging) shared by all session classes | `Lifecycle.Consistent` preserved by `beginOp` / `finish*` / `endOpLocked` |
| `PrefillDecode/Receive.lean` | one `TransferReceiveSession` + manager poll/publication | `Recv.reachable_safe`, `Recv.reachable_can_settle` |
| `PrefillDecode/ReceivePoll.lean` | the poll-side `IsReadyToComplete` finish, with (`stepM true`, shipping) and without (`stepM false`) it; `metrics` ghost | `Recv.reachable_callbackFinishes`, `Recv.pollReady_window`, `Recv.noPoll_safe`, `Recv.noPoll_metrics_on_success`, `Recv.noPoll_zero_layers_never_succeeds` |
| `PrefillDecode/Send.lean` | one `TransferSendSession`: pull claim, `StartPush`, D2H loop, H2H chain | `Send.reachable_safe` (`PullClaimed`), `Send.reachable_can_settle` |
| `PrefillDecode/Pipeline.lean` | single request: handshake rendezvous + five layer-indexed memories, reclaim, reseat, handoff | `Pipeline.reachable_safe`, `Pipeline.reachable_unregistered_safe`, `Pipeline.attention_safe`, `Pipeline.reachable_can_settle`, `Pipeline.handedOff_quiet`, `Pipeline.inv_can_handoff` |
| `PrefillDecode/MultiRequest.lean` | `multiSys`: concurrent/overlapped requests, read- and write-after-release checks (`wroteReleased`) | `Pipeline.system_data_correct`, `Pipeline.system_attention_safe`, `Pipeline.system_progress` |
| `PrefillDecode/PeerIsolation.lean` | one consumer, `Peer.sick` / `Peer.healthy` producers, `StagingBlockAllocator` slots, `push_pool_` (`tcpBlocking` vs `grpcAsync`) | `PeerIsolation.reachable_safe`, `PeerIsolation.reachable_inv`, `tcp_healthy_blocked_when_pool_full`, `grpc_healthy_can_complete`, `trace_sick_peer_starves_staging_slots` |
| `PrefillDecode/UuidTable.lean` | `active_recv_sessions_` / `send_sessions_` tables, drain-before-reuse, `req_id` idempotency, `CompleteReadWithDetails` sweep | `UuidTable.reachable_inv`, `active_recv_preserved`, `active_send_preserved` |
| `PrefillDecode/BlockOrdering.lean` | within-layer blocks: `PopulateRegisteredBlocks`, `ValidateRequestedBlocksLocked`, `BuildCoalescedCopySpec`, `BuildLoadCopyPlan` (`remote_order` / `local_order`), custom host staging, `BlockPipeline` | `execCoalesced_buildCoalescedSpec`, `landStage_get_requested`, `h2dStage_get_requested`, `BlockPipeline.step_pipe`, `BlockPipeline.reachable_safe` |
| `PrefillDecode/PipelineChecks.lean` | `Pipeline` / `multiSys` traces, `#guard` searches, mutants | `Pipeline.trace_normal`, `trace_layers_out_of_order`, `trace_pull_ahead_of_registration`, `trace_multi_request`, `trace_overlapped_requests` |
| `PrefillDecode/Transport.lean` | `BlockTransport` receiver-side stream accounting (the per-uuid `BlockTransportServerPushSession`: `layer_progress_`, chunk-count and sender-declared contracts, once-latch, retirement) feeding the receiver's block and layer counters; discharges `Receive` A1 and the A4 coarsening under one explicit assumption (honest senders) and shows what one extra stream does without it | `Transport.reachable_safe`, `trace_dup_chunk_publishes_missing_layer`, `trace_dup_after_retirement`, `trace_dup_declared_rejected`, bounded `searchHardened` |
| `Transfer/Caller.lean` | the engine above `poll_stats()` shared by both transfer paths (tpu-inference `tpu_raiden_connector.py` and vllm-torchtpu `tpu_connector.py` on vLLM): the worker-side connector as a `Connector` (`connectorTwoChannel` and three counterfactual mutants `connectorDropFailed`, `connectorNaive`, `connectorLoadErrorOnly`), `KVOutputAggregator` multi-worker counting over both `AggChannel.latchedReq` (`_failed_recving_pending`) and `AggChannel.eagerBlock` (`invalid_block_ids`) (`MultiWorker.lean`), and the scheduler's block lifecycle under both `kv_load_failure_policy` values; the other direction of `Pipeline` A4 (`Contract`, `mwEarlyFree = false`) | `Caller.contract_safe`, `Caller.mw_contract_safe`, `twoChannel_contract`, `applyRecv_eager_eq_latched`, `earlyLoadError_fail_eq_finish`, `earlyLoadError_fail_holds_blocks`, `leakP_absorbing`, `leakD_absorbing`, `mwLeakP_absorbing`, `mwLeakD_absorbing`, `trace_twoChannel_*`, `trace_mw_latched_*`, `trace_dropFailed_*`, `trace_mw_dropFailed_partial_failure_leaks`, `trace_naive_decodes`, `trace_loadErrorOnly_*`, exhaustive `search` and `mwSearch` per configuration |

## Boundaries

The model sits between three boundaries. Each side names the module that owns
it; the arguments are in those modules.

### Subsystem boundary

In scope is the prefill-to-decode ("symmetric") transfer: producer
registration (`NotifyForRead` / `register_read`), consumer pull and load
(`StartRead` / `start_read`), the D2H → H2H → H2D pipeline in
`TransferSendSession` / `TransferReceiveSession`, and outcome polling
(`CompleteReadWithDetails` / `poll_stats()`).

Out of scope: pool resharding (`ReshardSendSession` / `ReshardReceiveSession`,
`active_pool_reshard_{sends,recvs}_` in the same poll loop) — see
[reshard.md](reshard.md); and `RaidenController`, `kv_cache_store`, and
`host_offload_backend`.

### Engine boundary (above `poll_stats()`)

Modelled: the API the engine sees (`NotifyForRead`, `StartRead`,
`CompleteReadWithDetails` → `done_sending` / `done_recving` / `failed_recving`)
and both directions of the contract across it. What tpu-sync owes the engine
is proved as HBM and staging safety (§What is proved); when the engine may
recycle is `MultiRequest.lean` (`.recyclePrefill` only after the send is
published, `.nextRequest` only after the receive is; `wroteReleased` catches a
premature recycle). What the engine owes a publication is `Caller.lean`, for
the production caller pair tpu-inference `tpu_raiden_connector.py` on vLLM
(both for a single worker and across workers aggregated by `KVOutputAggregator`
over `AggChannel.latchedReq` and `AggChannel.eagerBlock` in
`Caller/MultiWorker.lean`): a two-channel connector (`connectorTwoChannel`) that
forwards every outcome and reports a failed receive on both the completion and
load-failure channels never frees HBM while a worker on that node is still in
flight (`mwEarlyFree = false`), leaks no held block, and never decodes on
unwritten KV (`Caller.contract_safe`, `Caller.mw_contract_safe`); dropping
`failed_recving` or omitting either channel on failure violates safety
(`trace_dropFailed_*`, `trace_mw_dropFailed_partial_failure_leaks`,
`trace_naive_decodes`, `trace_loadErrorOnly_*`).

Not modelled: the Python orchestration itself (bootstrap and proxy servers,
scheduler queues, external block allocators, engine-side timeouts). An engine
that frees HBM before `poll_stats()` publishes violates `Pipeline` A4, which
`wroteReleased` makes visible. tpu-sync has no caller-invoked cancel on this
path: a session ends only on its own deadline, a transport error, manager
shutdown or plan unregister. `Caller.lean` leaves out (its B2–B3) partial-prefix
load failures and preemption, prefix caching and chunked prefill.

### Transport boundary (below the session callbacks)

Modelled: the session–transport callback protocol — the incoming-push lease
(`begin_incoming_push` / `end_incoming_push`) that pins staging across session
expiry, `OnLayerReceived` before `OnBlockShardsReceived` (Receive A4), the
within-layer block plans (`BlockOrdering.lean`), the control-plane worker pool
(`PeerIsolation.lean`), and `BlockTransport`'s receiver-side stream accounting
(`Transport.lean`: the per-uuid `BlockTransportServerPushSession` that
`BlockTransport::HandleIncomingPush` delegates each stream to; it discharges
Receive A1 and the coarsening behind A4 under one named assumption, honest
senders).

Not modelled: raw socket I/O and wire framing (`ReadExact` / `WriteExact` and
their timeouts, headers, length checks, connection pooling, `memcpy`),
collapsed into the atomic `land l`, `h2hDone l ok` and `cancel` + `pushEnd`;
the Peregrine data plane (`BlockTransport::AsyncPush` when
`TPU_RAIDEN_ENABLE_PEREGRINE` selects it, and the receive path
`BlockTransport::OnRecvComplete` →
`BlockTransportServerPushSession::OnPeregrinePostComplete`, `is_peregrine`,
which has no production caller yet); and bounded host layer staging
(`TPU_RAIDEN_NUM_STAGING_LAYERS`, `StagingLayerManager`,
`TransferSendSession::AcquireStagingSlot` /
`TransferReceiveSession::AcquireStagingSlot`), under which successive layers
reuse one host buffer — the models are the full per-layer mirror; `Receive.lean`
and `Send.lean` §"Bounded host layer staging" argue the session properties
carry over. Read `block_transport.cc` and `block_transport_session.cc` directly
for wire behaviour.

## Assumptions and how each is discharged

Every module numbers its assumptions (`Receive` A1–A4, `Send` A1–A7, `Pipeline`
A1–A4, `Transport` A1–A4, `Caller` B1–B3); the full statements and arguments
are in the module docstrings. This table is the index: what each one says in a
line, and whether it is discharged by another Lean module, enforced by C++
control flow, or an external contract.

| Assumption | One-line statement | Status |
|---|---|---|
| Pipeline A4 — engine contract | Decode reads destination HBM only after `done_recving`; prefill frees source HBM only after every registered transfer settled. | External contract; encoded as the guards of `.recyclePrefill` / `.nextRequest` in `MultiRequest.lean`. The other direction is `Caller.lean`. |
| Receive A4 / Pipeline A3 — transport ordering, one sender per layer | For the stream that completes a layer, `OnLayerReceived` runs before `OnBlockShardsReceived` on the same thread; a layer's blocks are accounted once its copy is issued. | Ordering: C++ control flow in `BlockTransportServerPushSession::StartStream`. Coarsening: discharged by `Transport.NetworkSound` under `Transport` A4 (honest senders); mutant `Recv.netAccountUnordered` shows what breaks without it. |
| Receive A1 — one dispatch per layer | `CompleteIncomingPush` fires `OnLayerReceived` once per layer (`on_layer_received_called`). | Discharged by `Transport.OnceDispatched` under `Transport` A2 (a layer's contract is latched by its first stream), A4 (honest senders) and `Cfg.WF.chunkSingle`; `trace_dup_after_retirement` and `trace_chunk_two_senders` are the exceptions. |
| Receive A2 — registration | External `ReleaseStaging()` runs only on registration failure, before the plan is published. | C++ init discipline; the tables are `UuidTable.lean`. |
| Receive A3 / Send A4 — no double end | Each `in_flight_` op ends at most once. | C++ callback and RAII discipline. |
| Send A1 — at-most-once `StartPush` | After `NotifyForRead` registration and the `pull_started_` claim in `ValidateAndBeginPull`. | Discharged: `Send.PullClaimed`, `Pipeline.HandshakeSafe`, `BlockOrdering.validateRequestedBlocks_iff`. |
| Send A2, A6 — sequential D2H loop, `SendNextLayer(0)` fold | One thread creates every layer's D2H future, calls `SendNextLayer(0)`, then issues the copies in layer order. | C++ control flow (`TransferSendSession::StartPush`). |
| Send A3 — layers interchangeable | `Send.lean` counts copies and pushes without layer indices. | Discharged: `Pipeline.lean` indexes every memory event by layer. |
| Send A5 — no consumer Ack | `SendAck` has no non-test caller, so `HandleAck → AckSend → Finish()` is excluded. | Dead path. |
| Send A7 — connection-release window | Under `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer` the last `EndSendOp` drops `mu_` before `EndSendOpLocked`; nothing modelled changes in the window. | Soundness argument in `Send.lean`. |
| Pipeline A1 — whole-layer cells | A layer moves as one unit. | Discharged: `BlockOrdering.BlockPipeline.step_pipe` (lockstep simulation over blocks). |
| Pipeline A2 — delivery after the push callback | The model lands a layer at or after its push callback; C++ lands it before the ack. Sound because the push pins staging. | Soundness argument in `Pipeline.lean`. |
| Transport A1–A4 — one lock, contract latched by the first stream, uniform declaration, honest senders | `Transport.lean` §Assumptions. | Stated there; A2 rests on controller-side ordering outside this repository, A4 is the one the C++ does not enforce and the traces drop. |
| Caller B1–B3 — multi-worker `KVOutputAggregator` threshold (`expectedP = 2`, `expectedD = 2`) with one settled outcome per worker (`TPU_RAIDEN_ENABLE_MULTI_NUMA=0`), whole-transfer failure, no preemption/prefix caching/chunked prefill | `Caller.lean` §Assumptions. | B1's `KVOutputAggregator` combining step is modelled directly for both `AggChannel.latchedReq` (`_failed_recving_pending`) and `AggChannel.eagerBlock` (`invalid_block_ids`) in `Caller/MultiWorker.lean` (`mw_contract_safe`, `applyRecv_eager_eq_latched`, `applyRecv_staggered_eq_batch`; `trace_mw_overcounted_expected_leaks` shows what breaks when `get_finished_count()` exceeds participating workers); B2 matches single-outcome `CompleteReadWithDetails` publication; B3 does not change which event frees a transfer-held block. |

## Shipping C++ vs. modelled design alternatives

One module parameterizes both the **shipping C++ behavior** and a
**non-shipping design alternative** so both can be compared formally (`stepM` in
`ReceivePoll.lean`). Every other module has one shipping `step`
(`Backend.tcpBlocking` and `Backend.grpcAsync` in `PeerIsolation.lean` are both
shipping C++ control-plane backends); where a trace shows what the shipping code
leaves open, an alternative `step` at the end of the module or in `Traces.lean`
records the candidate change — `PeerIsolation.stepPerPeerQuota` (a per-peer
staging cap on `StartRead` for
`DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`) and
`Transport.stepHardened` (chunk-count over-count rejection and a session-side
per-layer once-latch) — without parameterizing the main model or its inductive
theorems.

| Module | Parameter | Shipping C++ behavior | Modelled design alternative (not in shipping C++) |
|---|---|---|---|
| `PrefillDecode/ReceivePoll.lean` | `stepM (poll : Bool)` | **`stepM true`**: the poll in `CompleteReadWithDetails` finishes a session whose `IsReadyToComplete()` holds. Safe, but when it fires between the last future becoming ready and the last callback running, that callback sees `draining_` and skips the end-of-transfer metrics (`trace_poll_skips_metrics`). | **`stepM false`**: only the last H2D callback finishes a session with layers (a zero-layer receive would have to finish at creation, as a zero-layer send does). Keeps every safety and settlement property (`noPoll_safe`, `noPoll_can_settle`) and guarantees metrics on success (`noPoll_metrics_on_success`). |

## Counterfactual mutants

Each mutant removes a specific C++ guard or ordering constraint and yields a
`.counterexample` (or refutation trace) in `TpuSyncVerify`:

| Mutant | C++ guard / invariant removed | Property violated |
|---|---|---|
| `Recv.cancelEager`, `Send.cancelEager` | `FinishLocked` / `EndRecvOpLocked` / `EndSendOpLocked` settle only when `in_flight_ == 0`, and `CompleteReadWithDetails` publishes only once `Done()` | Settle safety (`done → inFlight = 0`), no retired callback (`NoRetiredCallback`) / drained before settle (`Drained`), staging & HBM safety |
| `Recv.netAccountUnordered` | `OnLayerReceived` runs before `OnBlockShardsReceived` on the same thread in `BlockTransportServerPushSession::StartStream` (Receive A4) | Readiness soundness (`IsReadyToComplete → ready = numLayers`) |
| `Send.trySendNextUncounted` (`Send.sendNextUncounted` / `Send.startUncounted`) | `++in_flight_` op taken by `SendNextLayer`, called both from `StartPush` before its D2H loop (`SendNextLayer(0)`, folded into `start`) and from the H2H pool task (`SendNextLayer(l+1)`) | No underflow (`NoUnderflow`) / drained before settle (`Drained`: the pool task still ends the uncounted op via `absl::Cleanup end_op`, so `inFlight` hits 0 early) |
| `Send.stepWithoutFailUnissued` (`Send.trace_without_fail_unissued_leaks`) | `absl::Cleanup fail_unissued_layers` in `StartPush` resolves unissued layers' `d2h_promises` when the D2H loop exits early while `draining_` | No op leak (`NoOpLeak`: an armed `SendNextLayer` callback on an unissued layer hangs with `in_flight_ > 0` forever) |
| `Recv.h2dIssueLeak` | `EndRecvOpLocked()` on the `if (done_ \|\| draining_)` early return under the second lock in `ExecuteLayerH2d` | No op leak (`NoOpLeak`; leaked `in_flight_` prevents `done_` forever) |
| `Send.sendNextUnbounded` | `l >= base_->num_layers()` and `l >= d2h_layer_futures_.size()` bounds checks in `SendNextLayer` | No op leak (`NoOpLeak`; `SendNextLayer(numLayers)` takes an op whose future never exists) |
| `Pipeline.dispatchEarly` (`trace_dispatch_early`) | `ExecuteLayerH2d(l)` only dispatched after layer `l` landed in `CompleteIncomingPush` | Publication correctness (`decodeHbm` receives `.junk`) |
| `Pipeline.h2dReadyByRank` (`trace_h2d_by_rank`) | H2D copy for layer `l` reads staging slot `l` rather than a completion-counter index | Publication correctness (out-of-order landing produces `[kv 1, junk]`) |
| `Pipeline.reseatAtFinish` (`trace_reseat_at_finish`) | Prefill staging released only when settling (`done_ = true`, `in_flight_ == 0`), not immediately on `FinishLocked` (`draining_ = true`) | Publication correctness / staging safety |
| `UuidTable.stepOverwriteDraining` | `EmplaceRecvSessionLocked` rejects a duplicate UUID while an expired incumbent is still draining (`draining = true, done = false`) rather than overwriting it | Staging slot conservation (`freeSlots + activeStaging = numSlots`) |
| `PeerIsolation.stepReleaseOnCancel` | `TransferReceiveSession` releases its `StagingBlockAllocator` slot only on `Lifecycle.settleLocked` (`in_flight_ == 0`), not on `.cancel` while the pull handshake (`inFlight = 1`) is still outstanding | Staging slot conservation (`PeerIsolation.Safe`: `freeSlots + activeStaging = numSlots`) |
| `PeerIsolation.stepPerPeerQuota` (candidate fix, not a mutant) | adds a per-peer cap `peerStaging peer < maxPerPeer` on `StartRead` staging admission (`DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`) | none within the search depth (`= .outOfFuel`), where shipping `step` hits `starvesFirstHealthy` at depth 3 (`trace_per_peer_quota_admits_healthy`) |
| `BlockOrdering.trace_mutant_mismatched_transport_order` (`buildBuggyCopyPlan`) | `BuildLoadCopyPlan` applies `remote_order` to `producer_remote_block_ids` **and** to `transport_host_block_ids`; the mutant keeps `h2hPairs` in the unsorted request order | Within-layer block publication correctness: on `test_host_reordering` (`remote = [1, 0]`, `local = [0, 1]`) `h2dStage` yields `[.kv 0 0, .kv 0 1]` instead of the correct `[.kv 0 1, .kv 0 0]` |
| `Transport` with `Cfg.honest = false` (`trace_dup_chunk_publishes_missing_layer`, `trace_dup_after_retirement`) | `Transport` A4, honest senders: the chunk-count contract has no over-count rejection, every accepted stream is counted, and `layer_progress_` is erased on `all_layers_called` while the receive is live | `NetworkSound` / `CompleteOnSuccess`: success published with a layer never copied (chunk-count, before retirement); `OnceDispatched`: a layer copied twice after retirement on either contract (`trace_dup_copy_fails_late`: a failing duplicate fails the receive) |
| `Transport.stepHardened` (candidate fix, not a mutant) | adds the missing over-count rejection to the chunk-count contract and a per-layer once-latch in `ExecuteLayerH2d` | none within the search depth on the dishonest instances (`searchHardened … = .outOfFuel`); honest traces unchanged (`trace_hardened_honest`) |
| `Caller.connectorDropFailed`, `Caller.connectorNaive`, `Caller.connectorLoadErrorOnly` | `connectorTwoChannel` (`TPUConnectorWorker.get_transfer_results` / `TPURaidenConnectorWorker.get_finished`) forwards `failed_recving` on both P (`finished_sending`) and D (`finished_recving` + load-failure channel) | `connectorDropFailed` leaks held blocks on both nodes (`trace_dropFailed_*`, `trace_mw_dropFailed_partial_failure_leaks`); `connectorNaive` (including `get_finished()` without `get_block_ids_with_load_errors`) decodes on unwritten KV (`trace_naive_decodes`); `connectorLoadErrorOnly` leaks D blocks (`trace_loadErrorOnly_*`, `trace_mw_latched_loadErrorOnly_never_emits`) |

## C++ → Lean routing index

Per C++ symbol: the Lean module that owns it, the model events or definitions
it becomes, and the theorems to open first. The `Lines` column is generated by
`tools/resolve_anchors.py`.

### Receive session (`recv.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `recv.h` | `TransferReceiveSession::TryBeginRecvOp` | `:121-126` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `Lifecycle.beginOp`, `Recv.pushBegin` | `Lifecycle.Consistent`, `Recv.reachable_safe` |
| `recv.h` | `TransferReceiveSession` (`std::vector<int64_t> blocks_received_per_shard_` .. `std::vector<raiden::PjRtCopyFuture> h2d_futures_`) | `:308-331` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `in_flight_`, `draining_`, `done_`, `failed_phase_`, `handshake_pending_`, `blocks_received_per_shard_`, `num_completed_shards_`, `num_completed_layers_`, `network_completed_`, `h2d_futures_` | `Recv.reachable_safe` (`SettleSafe`, `StagingIntegrity`, `SettlesPromptly`) |
| `recv.cc` | `TransferReceiveSession::Create(remote_block_ids)` | `:164-190` | `PrefillDecode/Receive.lean`, `PrefillDecode/BlockOrdering.lean` | `Recv.initLoad` | `BlockOrdering.landStage_get_requested` |
| `recv.cc` | `TransferReceiveSession::Create(total_blocks)` | `:192-205` | `PrefillDecode/Receive.lean` | `Recv.initPush` | `Recv.reachable_safe` |
| `recv.cc` | `TransferReceiveSession::InitFromActivePlan` | `:227-297` | `PrefillDecode/Receive.lean` | `Recv.initPush` (`handshake_pending_ = false`, `in_flight_ = 0`) | `Recv.reachable_safe`, `Recv.trace_normal`, `Recv.trace_push_lease_pins_staging_on_cancel` |
| `recv.cc` | `TransferReceiveSession::AllocateStagingForLoad` | `:299-345` | `PrefillDecode/BlockOrdering.lean` | custom vs. dynamic host staging | `BlockOrdering.CustomHostStagingCorrect` |
| `recv.cc` | `TransferReceiveSession::BuildLoadCopyPlan` | `:347-415` | `PrefillDecode/BlockOrdering.lean` | `buildCopyPlanFromTriples` (`CopyPlan.remoteSorted`, `.producerRemoteBlocks`, `.transportHostBlocks`, `.localSorted`, `.h2dPairs`) | `BlockOrdering.landStage_get_requested`, `BlockOrdering.h2dStage_get_requested`, mutant `trace_mutant_mismatched_transport_order` |
| `recv.cc` | `TransferReceiveSession::InitFromLoadPlan` | `:417-437` | `PrefillDecode/Receive.lean` | `Recv.initLoad` (`handshake_pending_ = true`, `in_flight_ = 1`) | `Recv.reachable_safe` |
| `recv.cc` | `TransferReceiveSession::ReleaseStagingLocked` | `:439-442` | `Transfer/Session.lean` | `Lifecycle.settleLocked` | `Recv.StagingIntegrity` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:449-464` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `Lifecycle.finishLocked`, `Recv.failedPhase` | `Recv.reachable_safe`, `Recv.trace_details_*`, mutant `Recv.cancelEager` |
| `recv.cc` | `TransferReceiveSession::Details` | `:477-493` | `PrefillDecode/Receive.lean` | `TransferDetails` reporting | `Recv.trace_details_*` |
| `recv.cc` | `TransferReceiveSession::CurrentPhaseLocked` | `:495-498` | `PrefillDecode/Receive.lean` | `Recv.Phase`, `Recv.phase` | `Recv.trace_details_*` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:500-511` | `Transfer/Session.lean`, `PrefillDecode/Receive.lean` | `Lifecycle.endOpLocked` | `Recv.SettleSafe`, `Recv.SettlesPromptly` |
| `recv.cc` | `TransferReceiveSession::AllH2dDoneLocked` | `:537-542` | `PrefillDecode/Receive.lean` | `Recv.allH2dDone` | `Recv.ReadinessSound` |
| `recv.cc` | `TransferReceiveSession::IsReadyToComplete` | `:544-550` | `PrefillDecode/Receive.lean`, `PrefillDecode/ReceivePoll.lean` | `Recv.isReadyToComplete`, `stepM poll` | `Recv.ReadinessSound`, `Recv.pollReady_window`, `Recv.trace_poll_skips_metrics` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShardsReceivedLocked` | `:559-586` | `PrefillDecode/Receive.lean`, `PrefillDecode/Transport.lean` | `Recv.networkCompleted`, `Recv.Ev.netAccount` | `Recv.ReadinessSound`, mutant `Recv.netAccountUnordered` |
| `recv.cc` | `TransferReceiveSession::RecordNetworkCompleteLocked` | `:588-605` | `PrefillDecode/Receive.lean`, `PrefillDecode/ReceivePoll.lean` | `Recv.Ev.netAccount` finish & metrics | `Recv.reachable_callbackFinishes` |
| `recv.cc` | `TransferReceiveSession::ExecutePullRequest` | `:607-683` | `PrefillDecode/Receive.lean`, `PrefillDecode/PeerIsolation.lean` | `Recv.Ev.pullReply`, `PeerIsolation.Ev.dispatchPull`, `PeerIsolation.Ev.sessStep idx (.pullReply ok)` | `Recv.trace_deadline_during_handshake`, `Recv.trace_details_rejected_pull`, `PeerIsolation.trace_consumer_gives_up_and_drains` |
| `recv.cc` | `TransferReceiveSession::OnRecvProgress` | `:685-697` | `PrefillDecode/Transport.lean` | Peregrine per-push progress hook (`is_peregrine` path, unmodelled) | — |
| `recv.cc` | `TransferReceiveSession::OnRecvNetworkComplete` | `:699-709` | `PrefillDecode/Receive.lean`, `PrefillDecode/Transport.lean` | Peregrine network-completion callback (`RecordNetworkCompleteLocked`) | `Recv.ReadinessSound`, `Recv.reachable_callbackFinishes` |
| `recv.cc` | `TransferReceiveSession::OnBlockShardsReceived` | `:711-758` | `PrefillDecode/Receive.lean`, `PrefillDecode/ReceivePoll.lean` | `Recv.Ev.netAccount` | `Recv.netAccount_frame`, `Recv.trace_late_net_account_after_retire` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:760-784` | `PrefillDecode/Receive.lean` | bounded host layer staging: an op nested inside the push lease while a layer waits for a host buffer (unmodelled; `Receive.lean` §"Bounded host layer staging") | — |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:786-925` | `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean` | `h2dBegin`, `h2dIssue`, `h2dReady`, `h2dDone` | `Recv.NoOpLeak`, `Recv.NoRetiredCallback`, `Recv.trace_finish_between_locks`, `Pipeline.DecodeHbmSafe`, mutant `Recv.h2dIssueLeak` |
<!-- source-map:end -->

### Send session (`send.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `send.h` | `TransferSendSession` (`absl::string_view failed_phase_` .. `bool done_`) | `:226-234` | `Transfer/Session.lean`, `PrefillDecode/Send.lean` | `failed_phase_`, `in_flight_`, `draining_`, `done_` | `Send.reachable_safe` (`SettleSafe`, `Drained`, `StagingIntegrity`, `NoUnderflow`) |
| `send.cc` | `TransferSendSession::Create` | `:64-89` | `PrefillDecode/Send.lean` | `Send.init` | `Send.reachable_safe` |
| `send.cc` | `TransferSendSession::PopulateRegisteredBlocks` | `:106-124` | `PrefillDecode/BlockOrdering.lean` | `validateRegistration` | `BlockOrdering.validateRegistration_iff` |
| `send.cc` | `TransferSendSession::ValidateRequestedBlocksLocked` | `:126-143` | `PrefillDecode/BlockOrdering.lean` | `validateRequestedBlocks` | `BlockOrdering.validateRequestedBlocks_iff` |
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:145-159` | `PrefillDecode/Send.lean`, `PrefillDecode/Pipeline.lean` | `Send.Ev.beginPull` (`pull_started_` claim) | `Send.PullClaimed`, `Pipeline.HandshakeSafe` |
| `send.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:186-190` | `Transfer/Session.lean` | `Lifecycle.settleLocked` | `Send.StagingIntegrity` |
| `send.cc` | `TransferSendSession::AcquireStagingSlot` | `:197-226` | `PrefillDecode/Send.lean` | bounded host layer staging: D2H loop waits for a host layer buffer before `++in_flight_` (unmodelled; `Send.lean` §"Bounded host layer staging") | — |
| `send.cc` | `TransferSendSession::FinishLocked` | `:245-260` | `Transfer/Session.lean`, `PrefillDecode/Send.lean` | `Lifecycle.finishOnceLocked`, `Send.Ev.cancel` | `Lifecycle.finishOnceLocked_decided`, `Send.trace_cancel_after_ok_finish`, `Send.trace_ok_after_cancel_keeps_failure`, mutant `Send.cancelEager` |
| `send.cc` | `TransferSendSession::Details` | `:279-295` | `PrefillDecode/Send.lean` | `TransferDetails` reporting | `Send.trace_details_*` |
| `send.cc` | `TransferSendSession::CurrentPhaseLocked` | `:297-305` | `PrefillDecode/Send.lean` | `Send.Phase`, `Send.phase`, `Send.failedPhase` | `Send.trace_details_*` |
| `send.cc` | `TransferSendSession::EndSendOpLocked` | `:307-314` | `Transfer/Session.lean`, `PrefillDecode/Send.lean` | `Lifecycle.endOpLocked` | `Send.SettleSafe`, `Send.Drained` |
| `send.cc` | `TransferSendSession::EndSendOp` | `:316-343` | `PrefillDecode/Send.lean` | `EndSendOp` (`TPU_RAIDEN_DATA_CONN_REUSE=per_transfer` connection-close window, Send A7) | `Send.reachable_safe` |
| `send.cc` | `TransferSendSession::BuildCoalescedCopySpec` | `:345-374` | `PrefillDecode/BlockOrdering.lean` | `buildCoalescedSpec` | `BlockOrdering.execCoalesced_buildCoalescedSpec` |
| `send.cc` | `TransferSendSession::StartPush` | `:412-559` | `PrefillDecode/Send.lean`, `PrefillDecode/Pipeline.lean` | `start` (pre-creates `d2h_promises` and calls `SendNextLayer(0)`), `d2hBegin`, `d2hIssue`, `d2hReady`, `d2hEnd`, `fail_unissued_layers` | `Send.reachable_safe`, `Pipeline.PrefillHbmSafe`, mutants `Send.startUncounted`, `Send.stepWithoutFailUnissued` |
| `send.cc` | `TransferSendSession::SendNextLayer` | `:561-660` | `PrefillDecode/Send.lean`, `PrefillDecode/Pipeline.lean` | `wake`, `h2hIssue`, `sendNext`, `h2hDone` | `Send.reachable_can_settle`, mutants `Send.sendNextUncounted`, `Send.sendNextUnbounded` |
<!-- source-map:end -->

### Manager (`mgr.cc`, `base.cc`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.begin_incoming_push =`) | `:364-389` | `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean` | `pushBegin` | `Recv.StagingIntegrity`, `Recv.trace_push_lease_pins_staging_on_cancel`, `Recv.trace_push_lease_outlives_h2d`, `Pipeline.StagingSafe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.end_incoming_push =`) | `:390-436` | `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean` | `pushEnd` | `Recv.StagingIntegrity`, `Recv.trace_push_lease_pins_staging_on_cancel`, `Recv.trace_push_lease_outlives_h2d`, `Pipeline.StagingSafe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NotifyForRead` | `:654-701` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/BlockOrdering.lean` | `UuidTable.Ev.notifyForRead`, `Pipeline.Ev.notifyForRead`, `validateRegistration` | `UuidTable.active_send_preserved`, `UuidTable.trace_duplicate_send_cannot_replace_live_offer`, `BlockOrdering.validateRegistration_iff` |
| `mgr.cc` | `KVCacheManagerWithTransfer::EmplaceRecvSessionLocked` | `:703-719` | `PrefillDecode/UuidTable.lean` | drain-before-reuse on `active_recv_sessions_[uuid]` | `UuidTable.active_recv_preserved`, `UuidTable.trace_duplicate_uuid_rejected_until_drained`, mutant `UuidTable.stepOverwriteDraining` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterActivePlan` | `:721-809` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Receive.lean` | `UuidTable.Ev.registerRecv`, Receive A2 | `UuidTable.reachable_inv` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterRecv` | `:811-831` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/Receive.lean` | `UuidTable.Ev.registerRecv` | `UuidTable.reachable_inv` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:1045-1149` | `PrefillDecode/UuidTable.lean`, `PrefillDecode/PeerIsolation.lean`, `PrefillDecode/Receive.lean` | `UuidTable.Ev.startRead`, `PeerIsolation.Ev.startRead peer`, `stepPerPeerQuota` | `UuidTable.trace_duplicate_receive_different_req_id`, `UuidTable.trace_repeated_receive_same_req_id_idempotent`, `PeerIsolation.trace_sick_peer_starves_staging_slots` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1151-1292` | `PrefillDecode/Send.lean`, `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/UuidTable.lean`, `Transfer/Caller.lean` | `cancel`, `pollReady`, `publish`, `sweepSend`, `sweepRecv`, `Caller.Ev.pOutcome`, `Caller.Ev.dOutcome` | `Send.trace_deadline_during_copy`, `Recv.trace_deadline_during_copy`, `Pipeline.PrefillHbmSafe`, `Pipeline.DecodeHbmSafe`, `Pipeline.StagingSafe`, `Caller.contract_safe` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadRaw` | `:1295-1300` | `PrefillDecode/Receive.lean` | `CompleteReadRaw` wrapper around `CompleteReadWithDetails` | `Recv.ReadinessSound` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1431-1557` | `PrefillDecode/Pipeline.lean`, `PrefillDecode/PipelineChecks.lean`, `PrefillDecode/Send.lean` | `Pipeline.Ev.pullWait`, `Send.Ev.beginPull`, `Send.Ev.cancel` | `Pipeline.HandshakeSafe`, `Pipeline.reachable_unregistered_safe`, `Pipeline.trace_pull_ahead_of_registration`, `Pipeline.trace_unregistered_pull_rejected`, `Send.trace_pull_spawn_failure_cleanup` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:2745-2762` | `PrefillDecode/Transport.lean` | sender-declared vs. chunk-count contract | `Transport.trace_chunk_two_senders`, `Transport.trace_dup_declared_rejected` |
<!-- source-map:end -->

### Transport (`bt.{h,cc}`, `bts.{h,cc}`)

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `bts.h` | `BlockTransportServerPushSession` (`struct LayerProgress` .. `size_t layers_dispatched_ ABSL_GUARDED_BY(mu_) = 0;`) | `:146-172` | `PrefillDecode/Transport.lean` | `LayerProgress` / `layer_progress_` / `layers_dispatched_` under the push session's `mu_`, one session per uuid | `Transport.reachable_safe` |
| `bt.h` | `BlockTransport` (`absl::Mutex server_sessions_mu_;` .. `active_push_sessions_ ABSL_GUARDED_BY(server_sessions_mu_);`) | `:280-283` | `PrefillDecode/Transport.lean` | the uuid → push-session map (`GetOrCreatePushSession`, `ForgetPushProgress`) | — |
| `bt.cc` | `BlockTransport::GetOrCreatePushSession` | `:443-456` | `PrefillDecode/Transport.lean` | first stream of a uuid creates its session; later streams share it | `Transport.reachable_safe` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:458-463` | `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/Transport.lean` | entry point of a stream: looks the uuid's push session up and delegates to `StartStream` | — |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:321-455` | `PrefillDecode/Receive.lean`, `PrefillDecode/Pipeline.lean`, `PrefillDecode/Transport.lean` | `pushBegin`, `land l`, `h2dBegin l`, `netAccount`, `pushEnd`; `Transport.Ev.stream l s` | `Recv.StagingIntegrity`, `Pipeline.StagingSafe` |
| `bt.cc` | `BlockTransport::OnRecvComplete` | `:468-474` | `PrefillDecode/Transport.lean` | Peregrine receive entry point (→ `OnPeregrinePostComplete`, unmodelled) | — |
| `bts.cc` | `BlockTransportServerPushSession::OnPeregrinePostComplete` | `:460-495` | `PrefillDecode/Transport.lean` | Peregrine receive path (`CompleteIncomingPush(..., /*is_peregrine=*/true)`, unmodelled) | — |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:506-702` | `PrefillDecode/Transport.lean`, `PrefillDecode/Receive.lean` | `Transport.Ev.stream l s`, `completePush`, `retire`, `nextLayer` | `Transport.reachable_safe` (`OnceDispatched`, `NetworkSound`, `CompleteOnSuccess`), `trace_dup_chunk_publishes_missing_layer`, `trace_dup_after_retirement`, `stepHardened` |
| `bt.cc` | `BlockTransport::ForgetPushProgress` | `:1009-1023` | `PrefillDecode/Transport.lean` | plan retirement drops the uuid's push session (`Cancel()`); not an event | — |
<!-- source-map:end -->

### Control plane

<!-- source-map:begin -->
| File | Symbol / anchor | Lines | Lean module | Model events / definitions | Key theorems / mutants |
|---|---|---|---|---|---|
| `tcp_control_plane_backend.cc` | `TcpControlPlaneBackend::SendPullRequestBlocking` | `:592-660` | `PrefillDecode/PeerIsolation.lean` | `Backend.tcpBlocking` | `PeerIsolation.tcp_healthy_blocked_when_pool_full` |
| `grpc_control_plane_backend.cc` | `GrpcControlPlaneBackend::SendPullRequest` | `:306-334` | `PrefillDecode/PeerIsolation.lean` | `Backend.grpcAsync` | `PeerIsolation.grpc_healthy_can_complete`, `PeerIsolation.grpc_freeWorkers_eq_poolSize` |
<!-- source-map:end -->

*External, the production caller pair (`Transfer/Caller.lean`):*
tpu-inference `tpu_inference/distributed/tpu_raiden_connector.py`,
vLLM `vllm/distributed/kv_transfer/kv_connector/utils.py`, and
vLLM `vllm/v1/core/sched/scheduler.py` (`TPUConnectorWorker.get_transfer_results`,
`get_finished`, `TPUConnectorScheduler.request_finished`,
`update_state_after_alloc`, `get_finished_count`; `KVOutputAggregator.aggregate`;
scheduler `_handle_invalid_blocks`, `_handle_failed_recving`,
`_update_from_kv_xfer_finished`, `finish_requests`,
`_update_waiting_for_remote_kv`, `kv_load_failure_policy`) →
`Caller.contract_safe`, `Caller.mw_contract_safe`,
`Caller.twoChannel_contract`, `Caller.applyRecv_eager_eq_latched`,
`Caller.applyRecv_staggered_eq_batch`, `Caller.leakP_absorbing`,
`Caller.leakD_absorbing`, `Caller.mwLeakP_absorbing`, `Caller.mwLeakD_absorbing`,
`trace_twoChannel_*`, `trace_mw_latched_*`, `trace_mw_staggered_*`,
`trace_dropFailed_*`, `trace_mw_dropFailed_partial_failure_leaks`,
`trace_naive_decodes`, `trace_loadErrorOnly_*`,
`trace_mw_overcounted_expected_leaks`.

## Where the rest lives

| Content | File |
|---|---|
| `trace_*` witnesses and the tests they replay; `#guard` bounded searches | the theorem docstrings and `#guard` lines in the Lean modules (`grep -rn "theorem trace_\|#guard" TpuSyncVerify/Transfer`) |
| how the models are written (one executable `step`, ghost fields, replay, mutants) | [conventions.md](../conventions.md) |

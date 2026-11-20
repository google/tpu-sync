import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Model
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Lemmas
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.SendInv
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.RecvInv
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Progress
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.SlotBounds
import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Traces

/-!
# Prefill-to-decode transfer under bounded host layer staging

Refines `Pipeline.lean` (`step_pipe_steps`) when `num_staging_layers` allocates
`numSendSlots` producer and `numRecvSlots` consumer host layer buffers
(`num_host_layers() < num_layers()`), arbitrated by each session's
`StagingLayerManager` (`Transfer/LayerStaging.lean`). `BlockLocation::kHost` is
excluded by `CheckHostMemoryTransferSupported` (`docs/transfer/prefill_decode.md`
§Boundaries).

File aliases: `send.{h,cc}` = `tpu_sync/kv_cache/transfer_send_session.{h,cc}`,
`recv.{h,cc}` = `tpu_sync/kv_cache/transfer_receive_session.{h,cc}`,
`base.cc` = `tpu_sync/kv_cache/kv_cache_manager_base.cc`,
`bt.cc` = `tpu_sync/transport/block_transport.cc`,
`send_drain_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_send_drain_test.cc`.

## State

Embeds `pipe : Pipeline` and adds:

| Field | C++ | Role |
|---|---|---|
| `sendMgr` | `TransferSendSession::staging_layer_manager_` | producer `StagingLayerManager(numSendSlots)` |
| `recvMgr` | `TransferReceiveSession::staging_layer_manager_` | consumer `StagingLayerManager(numRecvSlots)` |
| `prefillSlots` | producer `GetHostBuffer(0..numSendSlots-1, shard)` | physical producer host layer buffers |
| `wireB` | wire payload read from `prefillSlots[slot]` | layer-indexed wire cells |
| `decodeSlots` | consumer `GetHostBuffer(0..numRecvSlots-1, shard)` | physical consumer host layer buffers |
| `decodeHbmB` | decode HBM written from `decodeSlots[slot]` | consumer's `numLayers` HBM layers |
| `sendLoop` | ghost (`SendLoopPhase`) | `StartPush` D2H loop pc (`.nextIter`, `.acquiring`, `.leasedBeforeD2h`, `.exited`) |
| `recvPhaseL` | ghost (`RecvLayerPhase` per layer) | consumer lease phase (`.idle` .. `.h2dRetired`, `.aborted`) |
| `activeStreams` | ghost | open `BeginIncomingPush` .. `EndIncomingPush` streams |
| `slotFault` | ghost | latches `true` if any active step sees `Lookup = none` or `slot ≥ numSlots` |

## Events

| Event | C++ |
|---|---|
| `sendAcquireBegin` / `sendAcquireDone timedOut` | `TransferSendSession::AcquireStagingSlot(l)` pre-check, `Acquire`, and post-check |
| `sendD2hBegin` / `sendD2hIssue ok` | `StartPush` loop `draining_` check (releases `l` on exit) and `D2hSyncDispatch` into `prefillSlots[slot]` |
| `sendH2hIssue` / `h2hDone l ok` | `LookupStagingSlot(l)` (`GetBlockChunks` / `ValidateChunks`), push from `prefillSlots[slot]`, and `Release(l)` in callback |
| `recvAcquireBegin l` / `recvAcquireWait l timedOut` / `recvAcquireFinish l` | `TransferReceiveSession::AcquireStagingSlot(l)` op + `Acquire` + `draining_` re-check + `EndRecvOpLocked` |
| `land l` / `h2dBegin l` / `h2dIssue l ok` / `h2dDone l ok` | write `decodeSlots[slot]`, `LookupStagingSlot(l)` + `ExecuteLayerH2d(l)` into `decodeHbmB[l]`, and `Release(l)` in callback |
| `pipe e` | remaining `Pipeline.Ev` steps (`syncMgr` runs `Shutdown()` on `draining` and `ReleaseAll()` at `done`) |

## Lock scopes

| Event | Scopes | Gap note |
|---|---|---|
| `sendAcquireBegin` .. `sendD2hBegin` | session `mu_`, `StagingLayerManager::mu_`, session `mu_` | split across `.acquiring` and `.leasedBeforeD2h` so concurrent `Finish` / callbacks interleave freely |
| `recvAcquireBegin` .. `recvAcquireFinish` | session `mu_` (`++in_flight_`), `StagingLayerManager::mu_`, session `mu_` (`EndRecvOpLocked`) | split across `.waiting` and `.leasedInAcquire`; nested inside the stream's push lease |
| `sendH2hIssue`, `h2dBegin` | manager `mu_` (`GetSession`), `StagingLayerManager::mu_` (`Lookup`), session `mu_` | `sendLookupStagingSlot_active` / `recvLookupStagingSlot_active` prove `Lookup` succeeds |

## Assumptions

Inherits `Pipeline` A1–A4 (`Pipeline.lean`), plus:

* **LS1 (one push stream per consumer layer).** At most one push stream at a time
  acquires and lands layer `l` before `CompleteIncomingPush` (`landedL[l] = false`).
  Configuration obligation: `BlockTransport::AsyncPushShards` opens
  `parallelism * num_groups` streams per layer, so LS1 holds for `parallelism == 1`
  (or single-block transfers) with one source NUMA shard group; repeated
  per-chunk `AcquireStagingSlot(l)` calls on that stream are modelled.
* **LS2 (`Shutdown()` and `Release(l)` combined in `releaseSync`).** Exact on
  callback errors (`FinishLocked`'s `Shutdown()` precedes `Release(l)`,
  `doShutdown_release`). On synchronous dispatch failures (`D2hSyncDispatch`,
  `H2dSyncDispatch`), C++ calls `Release(l)` before `FinishLocked(error)`; that
  the intermediate state cannot break `Safe` is argued from C++ and not proved:
  the producer has no concurrent `Acquire` waiter, and on the consumer the
  failing thread holds `in_flight_ > 0` until `FinishLocked` records the error
  and shuts down, while any layer `l'` leasing the freed slot holds it
  exclusively and returns it on drain or `ReleaseAll()`.

## Properties

Proved on every reachable state (`reachable_safe`, `reachable_unregistered_safe`),
with deadlock-free settlement (`reachable_can_settle`) and tight `1`-slot bounds
(`no_slots_cannot_publish_ok`, `exists_complete_trace`):

* `Pipeline.Safe s.pipe`: HBM/staging quietness, handshake rendezvous, and `NoOpLeak`.
* `PublicationCorrect`: `s.pipe.recv.published = some true → s.decodeHbmB = Pipeline.good n`.
* `NoSlotFault`: `s.slotFault = false`, with active lookup resolution and non-aliasing (`sendLookupStagingSlot_active`, `recvLookupStagingSlot_active`, `sendLookupStagingSlot_injective`, `recvLookupStagingSlot_injective`).
* `SlotManagersValid` & `SettledSlotsFree`: `free.length + leases.length = numSlots`, `allSlots.Nodup`, and `done = true → leases = [] ∧ numFree = numSlots`.
* `DrainingShutsDown`: `draining = true → shutdown = true` on both sessions.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `send.h` | `TransferSendSession::ResolveStagingSlot` | `:158-160` |
| `recv.h` | `TransferReceiveSession::ResolveStagingSlot` | `:160-162` |
| `recv.h` | `TransferReceiveSession` (`absl::StatusOr<size_t> AcquireStagingSlot(size_t layer_idx);`) | `:202` |
| `recv.cc` | `TransferReceiveSession::ReleaseStagingLocked` | `:380-383` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:680-709` |
| `recv.cc` | `TransferReceiveSession::LookupStagingSlot` | `:711-719` |
| `send.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:225-229` |
| `send.cc` | `TransferSendSession::AcquireStagingSlot` | `:236-265` |
| `send.cc` | `TransferSendSession::LookupStagingSlot` | `:267-286` |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`size_t host_layer_idx = layer_idx;` .. `base_host_ptr = host_buffer->data();`) | `:3498-3512` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(` .. `ValidateChunks(block_delegate_, l, sh, uuid, chunks));`) | `:836-846` |
| `send_drain_test.cc` | `RecvLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` | `:1722-1807` |
| `send_drain_test.cc` | `RecvLifecycleTest.BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock` | `:1809-1917` |
| `send_drain_test.cc` | `SendLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` | `:1924-2033` |
| `send_drain_test.cc` | `SendLifecycleTest.BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock` | `:2035-2172` |
<!-- source-map:end -->
-/

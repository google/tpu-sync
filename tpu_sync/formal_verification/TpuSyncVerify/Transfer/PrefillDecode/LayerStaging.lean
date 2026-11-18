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

End-to-end single-request prefill-to-decode model when fewer host layer buffers
are allocated than device layers (the `num_staging_layers` constructor
parameter, `ResolveNumStagingLayers` in `base.cc`;
`KVCacheManagerBase::num_host_layers() < num_layers()`). Refines `Pipeline.lean`
by replacing the full per-layer host mirrors with `numSendSlots` physical
producer host layer buffers (`prefillSlots`) and `numRecvSlots` physical
consumer host layer buffers (`decodeSlots`), arbitrated independently by each
session's `StagingLayerManager` (`TpuSyncVerify/Transfer/LayerStaging.lean`).

* **Producer (`TransferSendSession`).** Before each iteration `l` of `StartPush`'s
  D2H loop takes `++in_flight_`, `AcquireStagingSlot(l)` checks `done_ || draining_`
  under `mu_`, blocks in `StagingLayerManager::Acquire(l, deadline_, &cancelled_)`
  until a host buffer is free, `deadline_` expires or `cancelled_` / `Shutdown()`
  wakes it, calls `FinishLocked(slot.status(), kStagingPhase)` on a non-cancelled
  error, and re-checks `done_ || draining_` under `mu_` (releasing `l` if the
  session started draining in the unlocked gap). Once `D2hSyncDispatch` runs,
  layer `l` owns its slot across D2H completion (`d2hReady l`), `SendNextLayer`'s
  `push_pool` task (`LookupStagingSlot(l)`, reached as
  `TransferSendSession::ResolveStagingSlot` from both
  `KVCacheManagerBase::GetBlockChunks` and `ValidateChunks` →
  `KVCacheManagerBase::GetBlockArrayHostSpan` →
  `KVCacheManagerBase::ResolveBlockArrayHostLayer` in
  `BlockTransport::BuildBlockRequests`, each taking
  `KVCacheManagerWithTransfer::mu_` in `GetSession(uuid)`),
  and `H2hWriteDirectAsync`, releasing it only inside the H2H
  completion callback (`h2hDone l ok`), on an earlier error/draining exit
  (`sendD2hBegin`, `sendD2hIssue false`, `sendWake`, `sendH2hIssue`), or
  wholesale at settle (`ReleaseAll()` in `ReleaseStagingBlocksLocked`). Because
  `StartPush` arms `SendNextLayer(0)` *before* entering the D2H loop, earlier
  layers' H2H pushes complete and return host buffers while the D2H loop is
  blocked in `AcquireStagingSlot` for a later layer.
* **Consumer (`TransferReceiveSession`).** When an incoming push stream addresses
  and validates layer `l`'s host memory in
  `BlockTransportServerPushSession::StartStream` — each chunk running
  `KVCacheManagerBase::GetBlockChunks` followed by `ValidateChunks` →
  `KVCacheManagerBase::GetBlockArrayHostSpan` →
  `KVCacheManagerBase::ResolveBlockArrayHostLayer`, each taking
  `KVCacheManagerWithTransfer::mu_` in `GetSession(uuid)` and calling
  `session->ResolveStagingSlot(layer_idx)` =
  `TransferReceiveSession::AcquireStagingSlot` —
  `AcquireStagingSlot(l)` refuses if `done_ || draining_`, takes an `in_flight_`
  op, waits in `StagingLayerManager::Acquire(l, deadline, &cancelled_)` (returning
  the existing slot immediately on `ValidateChunks` and on additional
  chunks/shards of an already-leased layer), fails the session on timeout or
  zero-slot error, releases `l` if the session started draining in the unlocked
  gap, and ends its `in_flight_` op. (On a full host mirror `GetBlockChunks` and
  `ResolveBlockArrayHostLayer` address the layer's own buffer directly without
  calling `ResolveStagingSlot` when `!explicit_pools_`;
  `TransferReceiveSession::ResolveStagingSlot` returns `NotFound` unconditionally
  on a full mirror, while `TransferSendSession::ResolveStagingSlot` returns
  `*slot` while leased and `NotFound` when unleased.)
  `ForEachPayload` lands layer `l` into `decodeSlots[slot]` inside the outer
  `BeginIncomingPush` .. `EndIncomingPush` lease (`GetSession` + `TryBeginOp`,
  `EndOp`); still inside that stream lease,
  `CompleteIncomingPush` → `KVCacheManagerBase::OnLayerReceived(l)` →
  `OnDataReceived(l, {}, {})` → `ExecuteLayerH2d(l)` first
  resolves `Lookup(l)` (`recv.cc`), then acquires `mu_` and either releases `l`
  without dispatching if `done_ || draining_ || total_blocks_ == 0` (the last
  disjunct never holds here: zero-block reads are finished by `StartRead`
  without a pull) or dispatches `H2dSyncDispatch`
  from `decodeSlots[slot]` into `decodeHbmB[l]`, releasing `l` inside
  `future.OnReady` (`h2dDone l ok`) or wholesale at settle (`ReleaseAll()` in
  `ReleaseStagingLocked`).

Host-memory transfers (`BlockLocation::kHost` as `NotifyForRead`'s `source` or
`StartRead`'s `destination`, `Pipeline.lean` §"Host-memory source and
destination") are refused unless `num_host_layers() == num_layers()`
(`KVCacheManagerWithTransfer::CheckHostMemoryTransferSupported`), so they never
run under bounded layer staging and are outside this module; on a host-source
send the `staging_layer_manager_.Release(l)` calls in `SendNextLayer` find no
lease and return `nullopt`.

`staging.h`/`staging.cc` are `tpu_sync/kv_cache/staging_manager.{h,cc}`; `send.h`/`send.cc` are
`tpu_sync/kv_cache/transfer_send_session.{h,cc}`; `recv.h`/`recv.cc` are
`tpu_sync/kv_cache/transfer_receive_session.{h,cc}`; `mgr.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`; `base.cc` is
`tpu_sync/kv_cache/kv_cache_manager_base.cc`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`; `bts.cc` is
`tpu_sync/transport/block_transport_session.cc`; `send_drain_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_send_drain_test.cc`. Exact line ranges for every
symbol, sub-block, and test are in the generated `## Source map` table below.

## State and lock scopes

`BoundedPipeline` embeds `pipe : Pipeline` (preserving every session counter,
lifecycle bit, and handshake step via the lockstep simulation `step_pipe_steps`)
and adds:

| Field | C++ | Role |
|---|---|---|
| `sendMgr` | `TransferSendSession::staging_layer_manager_` | producer `StagingLayerManager(numSendSlots)` |
| `recvMgr` | `TransferReceiveSession::staging_layer_manager_` | consumer `StagingLayerManager(numRecvSlots)` |
| `prefillSlots` | `GetHostBuffer(0..numSendSlots-1, shard)` on producer | producer's physical host layer buffers |
| `wireB` | payload delivered onto the wire from `prefillSlots[slot]` | layer-indexed wire cells |
| `decodeSlots` | `GetHostBuffer(0..numRecvSlots-1, shard)` on consumer | consumer's physical host layer buffers |
| `decodeHbmB` | decode HBM written from `decodeSlots[slot]` | consumer's `numLayers` HBM layers |
| `sendLoop` | control pc of `StartPush`'s D2H loop (`SendLoopPhase`) | `.nextIter`, `.acquiring`, `.leasedBeforeD2h`, `.exited` |
| `recvPhaseL` | per-layer consumer lease state (`RecvLayerPhase`) | `.idle`, `.waiting hadLease`, `.leasedInAcquire`, `.leasedForStream`, `.h2dPending`, `.h2dRunning`, `.h2dRetired`, `.aborted` |
| `activeStreams` | open `BeginIncomingPush` .. `EndIncomingPush` streams | separates stream leases from `AcquireStagingSlot`'s transient `in_flight_` op |
| `slotFault` | ghost fault latch | set if any D2H, H2H, stream land, or H2D step sees `Lookup = none` or `slot ≥ numSlots` |

**Unlocked gaps around `StagingLayerManager::Acquire`.** Both `AcquireStagingSlot`
implementations drop session `mu_` before calling `StagingLayerManager::Acquire`
and re-acquire session `mu_` afterwards. Splitting producer acquisition into
`sendAcquireBegin` → `sendAcquireDone` → `sendD2hBegin` (via `.acquiring` and
`.leasedBeforeD2h`) and consumer acquisition into `recvAcquireBegin` →
`recvAcquireWait` → `recvAcquireFinish` (via `.waiting hadLease` and
`.leasedInAcquire`) admits arbitrary concurrent session callbacks and external
`Finish(error)` calls in both unlocked gaps. On callback error (`sendWake false`,
`h2hDone l false`, `h2dDone l false`), `FinishLocked` calls
`staging_layer_manager_.Shutdown()` *before* `Release(l)` (`releaseSync`;
`doShutdown_release` proves both orders yield the identical `StagingLayerManager`),
so a thread blocked in `Acquire` wakes with `Cancelled` instead of claiming the
freed slot on a failing session. The two synchronous dispatch failures
(`sendD2hIssue false` in `StartPush`, `h2dIssue l false` in `ExecuteLayerH2d`)
run in the other order — `staging_layer_manager_.Release(l)` first, then
`FinishLocked(error)` under `mu_` — and the model takes each as one
`releaseSync` step; LS2 says what that leaves out.

## Assumptions

Inherits `Pipeline` A1–A4 (`Pipeline.lean` §Assumptions), plus:

* **LS1 (one push stream per consumer layer; a configuration assumption the
  transport does not enforce).** Each layer reaches the consumer on one push
  stream, so at most one thread calls `AcquireStagingSlot(l)` for a given `l`
  at a time, and `GetBlockChunks(l)` and its companion `ValidateChunks` →
  `GetBlockArrayHostSpan` → `ResolveBlockArrayHostLayer` are called only before
  `CompleteIncomingPush` marks layer `l` landed (`landedL[l] = false`).
  `recvPhaseL[l]` is one acquisition state per layer and cannot represent two
  threads acquiring the same layer. This is `Pipeline` A3's one-stream case and
  does not follow from the C++: `BlockTransport::AsyncPushShards` declares
  `P * num_groups` streams for a layer (`P` is the manager's `parallelism`,
  capped at the number of blocks; `num_groups` is the number of source NUMA
  shard groups), and each stream runs `StartStream`'s `GetBlockChunks(l)` for
  itself. LS1 therefore holds when `parallelism == 1` (or the transfer is a
  single block) with one shard group; several concurrent acquirers of one
  layer are not modelled. Multiple chunks or shards of layer `l` within the
  one stream (and `ValidateChunks` after each `GetBlockChunks`) call
  `AcquireStagingSlot(l)` repeatedly (`recvAcquireBegin` from
  `.leasedForStream`, exercised in `trace_recv_cycles_host_layer_buffers`).
* **LS2 (`Shutdown()` and `Release(l)` are one step).** In C++ `FinishLocked`
  (`Shutdown()`) and `Release(l)` take `StagingLayerManager::mu_` in separate
  scopes; the model's `releaseSync` does both at once.
  - *Callback errors* (`sendWake false`, `h2hDone l false`, `h2dDone l false`).
    `Shutdown()` runs first and sets `shutdown_ = true` monotonically
    (`doShutdown_release`), so any waiter woken by `Shutdown()` or `Release(l)`
    observes `shutdown_ == true` under `StagingLayerManager::mu_` and returns
    `Cancelled`. The single step is exact.
  - *Dispatch failures* (`sendD2hIssue false`: `D2hSyncDispatch` fails in
    `StartPush`; `h2dIssue l false`: `H2dSyncDispatch` fails in
    `ExecuteLayerH2d`). Here the C++ calls `Release(l)` first and
    `FinishLocked(error)` afterwards, and the state between the two calls is
    not a state of the model. On the producer nothing can observe it:
    `StartPush`'s loop is the only caller of
    `TransferSendSession::AcquireStagingSlot` and it is the thread running the
    failure path, so no thread is waiting in `Acquire`. On the consumer a push
    thread waiting in `Acquire` for another layer `l'` can take the freed
    buffer before `Shutdown()`. That this cannot break `Safe` is argued from
    the C++ here and is not a Lean theorem: (1) the failing thread still holds
    the `in_flight_` op it took in `ExecuteLayerH2d`'s first `mu_` scope, so
    the receive can neither settle nor be published until its
    `FinishLocked(error)` and `EndRecvOpLocked()` run, and that `FinishLocked`
    records the error (the first error wins even on a receive already
    draining) and shuts the manager down whatever happened in between, so the
    receive ends failed; (2) layer `l`'s copy was never issued, so nothing
    reads or writes the freed buffer for `l`, and `l'` holds it as an ordinary
    exclusive lease — its landing and its H2D copy, if they run before the
    shutdown, use a buffer no other layer holds; (3) after the shutdown `l'`'s
    lease is returned by `AcquireStagingSlot`'s `done_ || draining_` re-check,
    by one of `ExecuteLayerH2d`'s draining exits, by its copy's callback, or by
    `ReleaseAll()` at settle. The window can change which layers were copied
    before the receive fails; it cannot publish a success, alias a buffer or
    leak a slot. No trace here and no C++ test holds a waiter across it.

## Main theorems

* **Safety (`reachable_safe`, `reachable_unregistered_safe`).** Every state
  reachable from `init n numSendSlots numRecvSlots` or
  `initUnregistered n numSendSlots numRecvSlots` (for *any* `n, numSendSlots,
  numRecvSlots : Nat`, including `1` slot or `0` slots) satisfies `Safe`:
  - `Pipeline.Safe s.pipe` (decode/prefill HBM quietness, staging lifetime,
    handshake rendezvous, `NoOpLeak`);
  - `PublicationCorrect`: `s.pipe.recv.published = some true → s.decodeHbmB = Pipeline.good n`
    — multiplexing `n` layers over arbitrarily few host buffers never aliases or
    corrupts another layer's KV data;
  - `NoSlotFault`: `s.slotFault = false` — every active D2H write, `GetBlockChunks` /
    `ValidateChunks` send lookup, H2H read, stream landing, `ExecuteLayerH2d` lookup, and H2D read
    finds a live in-bounds physical host slot;
  - `SlotManagersValid`: `free.length + leases.length = numSlots` and `allSlots.Nodup`
    on both sessions' `StagingLayerManager`s;
  - `SettledSlotsFree`: `done = true → leases = [] ∧ numFree = numSlots` on both
    sessions;
  - `DrainingShutsDown`: `draining = true → shutdown = true` on both sessions.
* **Lookup resolution & exclusivity (`sendLookupStagingSlot_active`,
  `recvLookupStagingSlot_active`, `sendLookupStagingSlot_injective`,
  `recvLookupStagingSlot_injective`).** Every active non-draining `GetBlockChunks` /
  `ValidateChunks` producer lookup and every open-stream `ExecuteLayerH2d` consumer precheck (even
  on a concurrently draining session) resolves an in-bounds physical host slot
  holding `.kv l`, and distinct live layers never alias to the same physical slot.
* **Eventual settlement (`reachable_can_settle`).** From every reachable state
  (registered or unregistered, for any `n, numSendSlots, numRecvSlots`), there
  exists a finite trace reaching a fully settled, published state where both
  sessions have `done = true`, `hasStaging = false`, `leases = []`, and
  `numFree = numSlots` — bounded host staging introduces no deadlock even when
  `numSendSlots < n`, `numRecvSlots < n`, or `0`.
* **Necessary and sufficient slot bounds (`no_slots_cannot_publish_ok`,
  `exists_complete_trace`).** For `0 < n`, if `numSendSlots = 0` or
  `numRecvSlots = 0` then `recv.published = some true` is unreachable; conversely,
  whenever `n = 0` or (`0 < numSendSlots ∧ 0 < numRecvSlots`), there exists an
  explicit finite trace (`completeTrace n`) publishing `done_sending`,
  `done_recving`, and `decodeHbmB = Pipeline.good n` — **1 host layer buffer on
  each side suffices for arbitrary `n`**.

## Replay traces and mutants

* `LayerStaging/Traces.lean` replays the four bounded-staging GTests in
  `send_drain_test.cc` (nine traces: the two cycling tests and the seven
  failure subcases of
  `BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock`) at reduced
  size: the tests run 64 or 16 layers over 4 host layer buffers, the traces 4
  or 3 layers over 2. The producer tests drive every event of their traces
  except the pull claim `beginPull`, which `TestManager::ServePull` skips by
  calling `StartPush` directly.
  The consumer tests bypass the transport (`RecvTestManager::AddRecv`, then
  `GetBlockChunks` and `OnLayerReceived` called directly), so their traces
  add the producer run, the pull reply, the stream lease, the payload landing
  and the block report as framing the tests do not perform.
  The file also holds eight traces no C++ test drives (lock-gap
  cancellation races, both dispatch failures, zero-slot rejection,
  unregistered handshake + poll + recycle) and refutes three mutants
  (`mutant_early_send_release_corrupts_hbm`,
  `mutant_early_recv_release_corrupts_hbm`, `mutant_missing_release_all_leaks_slot`).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `staging.h` | `StagingLayerManager` | `:39-74` |
| `staging.cc` | `StagingLayerManager::Acquire` | `:50-79` |
| `staging.cc` | `StagingLayerManager::Lookup` | `:81-88` |
| `staging.cc` | `StagingLayerManager::Release` | `:90-100` |
| `staging.cc` | `StagingLayerManager::ReleaseAll` | `:102-109` |
| `staging.cc` | `StagingLayerManager::Shutdown` | `:111-114` |
| `recv.cc` | `TransferReceiveSession::ReleaseStagingLocked` | `:380-383` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:390-405` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:680-709` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`if (base_->num_host_layers() < base_->num_layers())` .. `self->staging_layer_manager_.Release(layer_idx);`) | `:776-876` |
| `send.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:225-229` |
| `send.cc` | `TransferSendSession::AcquireStagingSlot` | `:236-265` |
| `send.cc` | `TransferSendSession::LookupStagingSlot` | `:267-286` |
| `send.cc` | `TransferSendSession::FinishLocked` | `:288-303` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::StatusOr<size_t> host_layer_idx = AcquireStagingSlot(l);` .. `next_layer = l + 1;`) | `:589-638` |
| `send.cc` | `TransferSendSession::SendNextLayer` | `:642-741` |
| `send.h` | `TransferSendSession::ResolveStagingSlot` | `:158-160` |
| `recv.h` | `TransferReceiveSession::ResolveStagingSlot` | `:160-162` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.get_session =`) | `:307` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CheckHostMemoryTransferSupported` | `:1402-1414` |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`size_t host_layer_idx = layer_idx;` .. `base_host_ptr = host_buffer->data();`) | `:3498-3512` |
| `base.cc` | `KVCacheManagerBase::OnLayerReceived` | `:3345-3354` |
| `base.cc` | `*` (`size_t ResolveNumStagingLayers(size_t num_layers,` .. `return requested;`) | `:718-734` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(` .. `ValidateChunks(block_delegate_, l, sh, uuid, chunks));`) | `:836-846` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `bts.cc` | `ValidateChunks` | `:275-310` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`std::vector<BlockChunk> chunks = delegate_->GetBlockChunks(` .. `ValidateChunks(delegate_, l, sh, header.uuid, chunks));`) | `:449-459` |
| `send_drain_test.cc` | `RecvLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` | `:1722-1807` |
| `send_drain_test.cc` | `RecvLifecycleTest.BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock` | `:1809-1917` |
| `send_drain_test.cc` | `SendLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` | `:1924-2033` |
| `send_drain_test.cc` | `SendLifecycleTest.BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock` | `:2035-2172` |
<!-- source-map:end -->
-/

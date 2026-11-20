import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Model
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Lemmas
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Progress

/-!
# Prefill-to-decode pipeline

End-to-end single-request prefill-to-decode model composing `Send.lean` and
`Receive.lean` with the `HandlePullStream` handshake and five layer-indexed
memories on the full host mirror (`num_host_layers() == num_layers()`). Bounded
host layer staging refines this model in `LayerStaging.lean`; within-layer block
gather/reordering refines it in `BlockOrdering.lean`; `BlockLocation::kHost` is
documented in `docs/transfer/prefill_decode.md` §Boundaries.

File aliases: `mgr.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`,
`base.h` = `tpu_sync/kv_cache/kv_cache_manager_base.h`.

## State

Composes `send : Send` and `recv : Recv` with layer-indexed `List Cell` memories
(`.kv l`, `.junk`, `.blank`) and per-layer ghost sets:

| Field | C++ / ghost | Role |
|---|---|---|
| `prefillHbm` | caller prefill HBM | initialized to `.kv l`; overwritten by `reclaim` |
| `prefillStaging` | producer host staging (`AcquireStagingWithRetry`) | written by `d2hReady l` and `reseatPrefillStaging` |
| `wire` | H2H payload (`H2hWriteDirectAsync`) | written by `h2hDone l true` |
| `decodeStaging` | consumer host staging | written by `land l` (`ForEachPayload`) and `reseatDecodeStaging` |
| `decodeHbm` | caller decode HBM | written by `h2dReady l` (`ExecuteLayerH2d`) |
| `registered`, `pullWaiting` | `sessions_[uuid]` offer & `HandlePullStream` `cv_.WaitWithTimeout` | producer registration and pull grace-wait state |
| `reclaimed` | ghost | engine freed `prefillHbm` after send publication |
| `d2hReadyL`, `h2hRetiredL` | ghost (`List Bool`) | per-layer producer D2H completion and H2H callback sets |
| `landedL`, `claimedL`, `h2dPendingL`, `h2dIssuedL`, `h2dReadyL` | ghost (`List Bool`) | per-layer consumer land, `OnLayerReceived` claim, and H2D sets |

Copies read their source at completion (`d2hReady l`, `h2hDone l true`, `h2dReady l`),
the pessimistic choice against mid-copy overwrites.

## Events

| Event | C++ |
|---|---|
| `notifyForRead` | `NotifyForRead` seats `SessionEntry` in `sessions_[uuid]` and calls `cv_.SignalAll()` |
| `pullWait` | `HandlePullStream` sees `!registered` and enters `cv_.WaitWithTimeout(&mu_, left)` |
| `send e` | `Send.Ev` step (`beginPull` requires `registered`; `wake` requires layer `woken`'s D2H ready) |
| `d2hReady l` | D2H copy `l` (`l < d2hIssued`) finishes: `prefillStaging[l] := prefillHbm[l]` |
| `h2hDone l ok` | push callback `l` (`l < h2hIssued`) runs; on `ok`, `wire[l] := prefillStaging[l]` |
| `recv e` | `Recv.Ev` step (`pullReply true` requires `send.pullStarted = true`) |
| `land l` | `StartStream`'s `ForEachPayload` writes `decodeStaging[l] := wire[l]` under `pushes > 0` |
| `h2dBegin l` | `OnLayerReceived(l)` enters `ExecuteLayerH2d(l)` once after `landedL[l]` |
| `h2dIssue l ok` | `ExecuteLayerH2d(l)` re-check and `H2dSyncDispatch` |
| `h2dReady l` | H2D copy `l` finishes: `decodeHbm[l] := decodeStaging[l]` |
| `reclaim` | engine frees `prefillHbm` after `send.published ≠ none` |
| `reseatPrefillStaging` | allocator reuses send staging after `!send.life.hasStaging` |
| `reseatDecodeStaging` | allocator reuses receive staging after `!recv.life.hasStaging` |

## Lock scopes

| Event | Scopes | Gap note |
|---|---|---|
| `notifyForRead` | producer manager `mu_`, then `cv_.SignalAll()` | `HandlePullStream` taking `mu_` before `SignalAll` already sees `sessions_[uuid]` |
| `pullWait` | producer manager `mu_` into `WaitWithTimeout` | atomically releases `mu_` while `registered = false` |
| `send .beginPull` | producer manager `mu_` nesting send `mu_` | `FindSessionLocked` and `ValidateAndBeginPull` run in one manager `mu_` scope |
| `recv (.pullReply ok)` | 2 consumer `mu_` scopes (`Receive.lean`) | RPC causality orders `send .beginPull` before `ok = true` |
| `land l` | unlocked `ForEachPayload` inside push lease | `pushes > 0` pins `decodeStaging` across the write and precedes `h2dBegin l` |
| `send .publish`, `recv .publish`, `recv .pollReady` | manager `mu_` nesting session `mu_` | `Done()` is monotone (`Receive.lean`, `Send.lean`) |
| `reclaim`, `reseat*` | 0 (engine / `StagingBlockAllocator`) | gated by published / released flags established under session `mu_` |

## Assumptions

* **A1 (whole-layer cells).** A layer moves as one unit; refined to per-block
  arrays in `BlockOrdering.lean` (`BlockPipeline.step_pipe`).
* **A2 (delivery ordered after push callback).** `land l` is ordered at or after
  `h2hDone l true` (C++ lands before the ack); sound because the push op pins
  `prefillStaging[l]` unchanged across `[h2hIssue, h2hDone]`.
* **A3 (one sender per layer).** One push stream delivers each layer (with
  multi-stream or NUMA-sharded pushes, `CompleteIncomingPush` waits for every
  stream of the layer, `Transport.lean`).
* **A4 (engine contract).** Decode reads `decodeHbm` only after `done_recving`,
  and prefill frees `prefillHbm` only after send publication (`MultiRequest.lean`,
  `Caller.lean`).

## Properties

Proved on every reachable state (`reachable_safe`, `reachable_unregistered_safe`),
with finite drain and handoff (`reachable_can_settle`, `inv_can_handoff`):

* `PublicationCorrect`: `recv.published = some true → decodeHbm = good n`.
* `DecodeHbmSafe` & `attention_safe`: once `recv.published ≠ none`, no H2D copy is pending or unretired, and `done_recving` preserves `decodeHbm = good n` forever.
* `PrefillHbmSafe`: `send.published ≠ none → send.d2hPending = false ∧ send.d2hRetired = send.d2hIssued`.
* `StagingSafe`: released `prefillStaging` / `decodeStaging` has no copy or push reading or writing it.
* `HandshakeSafe`: `StartPush` and data movement require `registered ∧ send.pullStarted`, and `pullWaiting` implies `recv.pullPending ∧ ¬send.pullStarted`.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `base.h` | `KVCacheManagerBase::num_host_layers` | `:713` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`// ids name blocks of this manager's host pool, pushed as they are (no` .. `BlockLocation source = BlockLocation::kDevice);`) | `:187-199` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1229-1358` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` (`const absl::Time give_up = absl::Now() + grace;` .. `session->ValidateAndBeginPull`) | `:1253-1278` |
<!-- source-map:end -->
-/

import TpuSyncVerify.Transfer.PrefillDecode.Receive.Model
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Frame
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Traces

/-!
# Receive session

Consumer side of the prefill-to-decode model: one `TransferReceiveSession` on
the decode (consumer) side and the slice of `KVCacheManagerWithTransfer` that
drives it. Models the session lifecycle — how in-flight work is counted,
how the session drains and settles, when its host staging is released — along
with the transport's block accounting, the readiness predicate
`IsReadyToComplete`, and the manager's poll that publishes `done_recving` /
`failed_recving`. Layer data, memory contents and the producer are modelled in
`Send.lean` and `Pipeline.lean`.

`recv.h`/`recv.cc` are `tpu_sync/kv_cache/transfer_receive_session.{h,cc}`;
`mgr.h`/`mgr.cc` are `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`; `base.cc`
is `tpu_sync/kv_cache/kv_cache_manager_base.cc`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`; `bts.cc` is
`tpu_sync/transport/block_transport_session.cc`;
`tpu_sync/transport/transfer_session.h` is the `TransferSession` interface
(`TryBeginOp`, `EndOp`, `Finish`, `Done`, `OnDataReceived`, `Start`, …) that
`TransferReceiveSession` implements and through which the transport and
`KVCacheManagerBase` reach it. Exact line ranges for every symbol, sub-block,
and test are in the generated `## Source map` table below.

## State

The settle protocol itself is `Transfer.Lifecycle`. On top of it:

| Field             | C++                                        | Role |
|-------------------|--------------------------------------------|------|
| `numLayers`       | `base_->num_layers()`                      | fixed |
| `issued`          | `h2d_futures_.size()`, and by A1 `dispatched_layers_.size()` | H2D copies handed to the device |
| `ready`           | futures in `h2d_futures_` with `IsReady()` | copies finished on the device |
| `completed`       | `num_completed_layers_`                    | callbacks that ran with an OK status |
| `layersAccounted` | layers' worth of the per-shard threshold `total_blocks_ * num_layers` absorbed by the counters `blocks_received_per_shard_` (`RecordBlockShardsReceivedLocked`) | layers whose blocks the transport has reported |
| `netIssued`       | `network_completed_` as set by `ExecuteLayerH2d`'s last scope (`dispatched_layers_.size() == base_->num_layers()`) | every layer's copy has been issued |
| `published`       | membership in `done_recving_` / `failed_recving_` (`RetireSettledSessionLocked`, from `CompleteReadWithDetails`; a settled incumbent is also retired when displaced in `EmplaceRecvSessionLocked`, `RegisterActivePlan` and `StartRead`) | what `poll_stats()` shows the engine |
| `pushes`          | ghost                                      | incoming pushes between `TryBeginOp` and `EndOp` |
| `pullPending`     | `handshake_pending_` (set in `InitFromLoadPlan`, cleared in `Start`'s `on_response`) | the StartRead pull handshake still holds its op |
| `pending`         | ghost                                      | `ExecuteLayerH2d` calls between their first and second lock |
| `retired`         | ghost                                      | H2D callbacks that have run, OK or not |

Ghost fields have no single C++ variable. They record where each unit of
`in_flight_` came from, which is what `Accounted` is about.

`network_completed_` has two setters and is `networkCompleted`,
`layersAccounted = numLayers ∨ netIssued`. The block-accounting setter is
`RecordBlockShardsReceivedLocked`: the C++ counts per shard
(`blocks_received_per_shard_[s]`, one `int64_t` per shard), each incoming push
stream adds its block count to every shard it carries
(`received += static_cast<int64_t>(block_ids.size())`), a shard completes at
`total_blocks_ * total_layers`, and `network_completed_` is set when every
shard has (`num_completed_shards_ == static_cast<int32_t>(blocks_received_per_shard_.size())`).
The model keeps the per-layer view: one `netAccount` per layer, which is the
coarsening A4 justifies. The dispatch setter is the last `mu_` scope of
`ExecuteLayerH2d`: after `h2d_futures_.push_back(future)` the layer joins
`dispatched_layers_`, and when that set holds every layer and the receive is
neither done nor draining, `network_completed_ = true` — the only setter a
receive fed by Peregrine posts (no block reports) ever sees. Under A1 the set's
size is `issued`, so `netIssued` is set by the `h2dIssue true` that makes
`issued = numLayers` (`Inv.netIssued_issued`).

Not state: `any_blocks_received_` (first-packet timing, set by the first
`TryBeginOp` or the first `RecordBlockShards`), `tracks_completion_` (`true`
for every receive here; `InitFromActivePlan` clears it for a receiver with no
blocks or a non-HBM destination, which `RetireSettledSessionLocked` then
erases without reporting), `destination_` (`BlockLocation::kDevice` for every
receive here; see §"Host destination" below), and `pull_manager_` /
`pull_endpoint_` (set by the pulled-receive `Create` so that `Start` knows
which manager's control backend to send the pull through). The request id a
receive is reported under is not the session's: `sessions_` maps each uuid to
a `SessionEntry {session, req_id, role}` and `RetireSettledSessionLocked`
reads `it->second.req_id`.

## Events

| Event          | C++ |
|----------------|-----|
| `pushBegin`    | `TryBeginOp` from `BeginIncomingPush` in `bts.cc` (`delegate->GetSession(uuid)` then `session->TryBeginOp()`; a `false` becomes `CancelledError`), called in `BlockTransportServerPushSession::StartStream` — the stream's entry point is `BlockTransport::HandleIncomingPush`, which looks the uuid's push session up and delegates to it |
| `pushEnd`      | `EndOp` from `EndIncomingPush` in `bts.cc`, called at the end of `StartStream` and from its `absl::Cleanup end_incoming_push` on any early return. `EndIncomingPush` takes the push's status and, when it is not OK, runs `session->Finish(status)` before `session->EndOp()` (the cleanup passes `InternalError("Incoming push failed")`) — in the model, `cancel` followed by `pushEnd`, see `trace_push_lease_pins_staging_on_cancel` |
| `pullReply ok` | `on_response` of `Start`: a transport failure keeps its status code, a producer rejection (`response->status != 0`) becomes `FailedPrecondition`; under `mu_` the error runs `FinishLocked(pull_status)` only while the receive is not already draining (`if (!pull_status.ok() && !self->draining_)` — a receive that settled before its reply keeps its verdict) and clears `handshake_pending_ = false`, then the `absl::Cleanup end_op` ends the op. In `StartRead`, the fault-injected `kKvCacheManagerApiStartReadSubmitPull` path (and the defensive `!started.ok()` check after `session->Start()`) runs `session->Finish(status); session->EndOp();` after `EmplaceRecvSessionLocked` released manager `mu_` — calling `Finish(status)` unconditionally without `on_response`'s `!self->draining_` guard, which in the model is `cancel` followed by `pullReply false` (matching `pullReply false` alone whenever `draining = false`, and forcing `statusOk := false` even if a concurrent caller already set `draining = true`) |
| `h2dBegin`     | `ExecuteLayerH2d`, first critical section (`++in_flight_`), from `OnDataReceived` with no block ids (`KVCacheManagerBase::OnLayerReceived` → `session->OnDataReceived(layer_idx, {}, {})`, called from `BlockTransportServerPushSession::CompleteIncomingPush` → `delegate_->OnLayerReceived`); `OnDataReceived` routes a host destination to `SettleHostLayer` instead (§"Host destination") |
| `h2dIssue ok`  | `ExecuteLayerH2d` from the `done_ \|\| draining_` re-check through `H2dSyncDispatch`, `h2d_futures_.push_back(future)` and the `dispatched_layers_` scope that may set `network_completed_` |
| `h2dReady`     | the device finishes a copy: its future becomes `IsReady()` |
| `h2dDone ok`   | `ExecuteLayerH2d`'s `future.OnReady` H2D completion callback |
| `netAccount`   | `RecordBlockShards` from `OnDataReceived` with block ids (`KVCacheManagerBase::OnBlockShardsReceived` → `session->OnDataReceived(0, block_ids, shard_ids)`, called from `StartStream` after `CompleteIncomingPush`): accounts a stream's blocks per shard (`RecordBlockShardsReceivedLocked`), may set `network_completed_`, and finishes the session if every layer is also complete (`RecordNetworkCompleteLocked`) |
| `pollReady`    | `CompleteReadWithDetails` / `CompleteReadRaw` sees `IsReadyToComplete()` on a `TransferReceiveSession` that is not draining and calls `recv->Finish()` |
| `cancel`       | any `Finish(error)` from outside the session: deadline (`CompleteReadWithDetails`'s `session->deadline() <= now` arm, one for every session kind, after the readiness arm: `Finish(DeadlineExceededError("Transfer session timed out"))`), shutdown (`~KVCacheManagerWithTransfer` finishes every session in `sessions_` with `CancelledError` and then `AwaitForDone()`s each), plan unregister (`UnregisterActivePlan`), a failed push (`EndIncomingPush`), and a ready incumbent being finished so that a new registration can retire it (`EmplaceRecvSessionLocked` / `RegisterActivePlan`: `recv->Finish()` if `!recv->IsDraining() && recv->IsReadyToComplete()` — an OK finish, which is `pollReady` from the poll window) |
| `publish`      | `RetireSettledSessionLocked` moves a settled session (`if (!recv->Done()) return false;`) into `done_recving_` or `failed_recving_` by its status under the entry's `req_id`, stores its `Details()` in `transfer_details_[req_id]`, and erases it from `sessions_`; called for every entry by `CompleteReadWithDetails` / `CompleteReadRaw`, and for a displaced incumbent by `StartRead`, `EmplaceRecvSessionLocked` and `RegisterActivePlan` |

## Lock scopes

Every mutable field of `TransferReceiveSession` is `ABSL_GUARDED_BY(mu_)`
except two that serve the bounded-staging wait below: `staging_layer_manager_`
(a `StagingLayerManager` with its own `mu_`) and the atomic `cancelled_`, a
mirror of `draining_` written only by `FinishLocked`
(`cancelled_.store(true, std::memory_order_relaxed)`). `base_`, `uuid_`,
`staging_allocator_`, `metrics_collector_`, `transfer_plan_`, `pull_manager_`
and `pull_endpoint_` are set before the session is shared and never written
again, and they (with `base_` / manager configuration getters) are the only
other state an event reads outside `mu_`.

| Event          | `mu_` scopes | Between them |
|----------------|--------------|--------------|
| `pushBegin`    | 1: `TryBeginOp` | preceded by `BeginIncomingPush`'s `delegate->GetSession(uuid)` (`KVCacheManagerBase::GetSession` → `hooks.get_session` → `KVCacheManagerWithTransfer::GetSession`, a `sessions_` lookup under manager `mu_`); a session retired in between is `done_` and refuses the op (`none`); a send session seated under the uuid refuses it too (`TransferSendSession::TryBeginOp` returns `false`) and is not a state of this model |
| `pushEnd`      | 1: `EndOp` (`EndRecvOpLocked`), plus one read-only `GetStatus()` scope in `EndIncomingPush` for its return value | on a failed push, `EndIncomingPush`'s `session->Finish(status)` is an earlier scope (`cancel` before `pushEnd`) |
| `pullReply`    | 2: `on_response`'s `absl::MutexLock lock(self->mu_)` (`FinishLocked(pull_status)` on a failure that finds `!self->draining_`, `handshake_pending_ = false`), then `EndOp()` from `absl::Cleanup end_op` after `lock` is released (`Start` also takes an initial `mu_` scope before scheduling to read `load_plan_` and set `peer_endpoint_`; the fault-injected `session->Finish(status); session->EndOp();` in `StartRead`, and the same pair after a `Start()` that returns an error — which `Start` never does — run an unconditional `Finish(status)` scope followed by `EndOp()`, i.e. `cancel` then `pullReply false`, leaving `handshake_pending_ = true` on a draining or settled session) | an `EndOp` tail, below |
| `h2dBegin`     | 1: first scope of `ExecuteLayerH2d` (after the unlocked `LookupStagingSlot` under bounded staging, below) | — |
| `h2dIssue`     | 1–3 in `ExecuteLayerH2d`: the `done_ \|\| draining_` re-check (`EndRecvOpLocked()` and return in 1 scope when draining); then `h2d_futures_.push_back(future)` after the unlocked `H2dSyncDispatch` (or, on dispatch failure, `FinishLocked(future_or.status(), kH2dPhase); EndRecvOpLocked()`); then, after `future.OnReady` is registered, `dispatched_layers_.insert(layer_idx)` and `network_completed_ = true` if the set is full and `!network_completed_ && !done_ && !draining_` | during `H2dSyncDispatch` the copy is not yet in `h2d_futures_`, so `AllH2dDoneLocked` does not see it. Harmless by A4: this thread calls `RecordBlockShards` only after `ExecuteLayerH2d` returns, so `network_completed_` cannot be set in the gap, and `num_completed_layers_ == total_layers` is false while a layer is unissued, keeping `IsReadyToComplete` false. A `Finish(error)` before the re-check is `cancel` then `h2dIssue` (`trace_finish_between_locks`); one during the dispatch finds the op `h2dBegin` took, cannot settle, and commutes with `push_back`, i.e. `h2dIssue true` then `cancel` (`trace_deadline_during_copy`, `RecvDrainTest.TimeoutDuringH2dDispatchKeepsStaging`). Between `push_back` and the third scope a `cancel`, or the copy's own `h2dDone` (the future may already be ready), finds exactly the model's state after `h2dIssue true`: the copy counted in `h2d_futures_` and `in_flight_`. The one thing the third scope then does differently is to leave `network_completed_` false (`!done_ && !draining_` fails) where the model has set `netIssued`; a draining session's flag is read by nothing the model tracks — `RecordBlockShards` and `TryBeginOp` refuse, `IsReadyToComplete` is consulted only behind `!IsDraining()` (`CompleteReadWithDetails`, `EmplaceRecvSessionLocked`, `RegisterActivePlan`), and `CurrentPhaseLocked` reads it only for `Details()` — so `h2dIssue true` followed by that `cancel` / `h2dDone` reaches the same lifecycle state with the same enabled events, and folding the third scope into `h2dIssue true` is exact for everything proved (`ReadinessSound` holds from `netIssued` through `Inv.netIssued_issued`, independently of the flag's timing) |
| `h2dReady`     | 0 (device) | — |
| `h2dDone`      | 2: the callback's `absl::MutexLock lock(self->mu_)` (`done_` check, `num_completed_layers_++`, `FinishLocked`), then `EndOp()` from `absl::Cleanup end_op` | an `EndOp` tail, below (the callback also returns the layer's host buffer lease, `self->staging_layer_manager_.Release(layer_idx)`, in the gap) |
| `netAccount`   | 1: `RecordBlockShards` (`RecordBlockShardsReceivedLocked`, `RecordNetworkCompleteLocked`, `FinishLocked`), preceded by `KVCacheManagerBase::OnBlockShardsReceived`'s `GetSession` lookup under manager `mu_` (an empty block list returns before any session: `if (block_ids.empty())`) | — |
| `pollReady`    | on the poll thread under manager `mu_`: `IsDraining()`, `IsReadyToComplete()`, `Finish()` (the log line reads the entry's `req_id`) | the poll, below |
| `cancel`       | 1: `Finish(error)`, after its condition was checked in earlier scopes (`!IsDraining()`, then `!IsReadyToComplete()` and `deadline() <= now` under manager `mu_` for the sweep; `plan().has_value()` under manager `mu_` and then `!session->Done()` with no lock for `UnregisterActivePlan`; none for `~KVCacheManagerWithTransfer` and `EndIncomingPush`) | `cancel` has no guard in the model, so it admits every such interleaving; late errors, below |
| `publish`      | under manager `mu_` in `RetireSettledSessionLocked`: `Done()`, `tracks_completion()`, `GetStatus()`, `Details()`, then `sessions_.erase(it)` (the `req_id` is the entry's, read under the same lock) | on a settled session only `status_` can still change; late errors, below |

**`EndOp` tails.** `pullReply` and `h2dDone` run their decision under `mu_`
and then call `EndOp` in a second scope. Between the two scopes, `in_flight_`
still counts the op. Because only the settle checks in `FinishLocked` and
`EndRecvOpLocked` inspect `in_flight_`, any interleaved step makes the same
decision as it would after `EndOp`, except that an interleaved `Finish`
leaves the final settle to `EndOp` with the same resulting `draining_`,
`done_` and `status_`. Every such interleaving therefore reaches the same state
as running the whole event first, which is why the ghost `pending` and `retired`
counters need no intermediate state for the tail.

**The poll.** `pollReady` checks `IsDraining()` and `IsReadyToComplete()` and
calls `Finish()`. Once `IsReadyToComplete()` is true it stays
true: when `network_completed_ || num_completed_layers_ == total_layers` holds,
`issued = numLayers` (`CountersOrdered` with A1, or `Inv.netIssued_issued`), so
no new future can join `h2d_futures_` and each ingredient of
`IsReadyToComplete` is monotone. If `Finish(error)` runs between the check and
`Finish()`, the session is already draining when the poll's OK `Finish()` runs,
so `FinishLocked` returns at `if (draining_) return;` without touching
`status_` — matching `pollReady` being disabled after `cancel`.

**Late errors.** `FinishLocked` records a non-OK status before testing
`draining_`, so a `Finish(error)` after an OK settle still flips `status_`;
`Lifecycle.finishLocked` does the same, which is why `cancel` is unguarded and
a subsequent `publish` reports `some false`. Four callers can do this: the
deadline sweep's `Finish(DeadlineExceeded)` if the last H2D callback settled the
session after `IsReadyToComplete()` returned false; `UnregisterActivePlan`,
whose `!session->Done()` check and `Finish(CancelledError)` both run after it
released manager `mu_`; `~KVCacheManagerWithTransfer`, which finishes every
seated session unconditionally; and `StartRead`'s synchronous pre-`Start()`
`kKvCacheManagerApiStartReadSubmitPull` (or `!started.ok()`) error path, which
calls `session->Finish(status)` unconditionally after `EmplaceRecvSessionLocked`
published the session in `sessions_` (`cancel` before `pullReply false`, flipping
`status_` if a concurrent caller already finished the session OK in that unlocked
window). In the `UnregisterActivePlan` (or `StartRead`) case, `Finish(error)`
can land during `RetireSettledSessionLocked` between the `GetStatus()` that
selects `done_recving_` and the `Details()` stored in `transfer_details_`,
producing a `done_recving_` entry whose details carry an error
(`transfer_details_` is outside the model's state; on the modelled fields every
interleaving is `cancel` before or after `publish`). By contrast, `Start`'s
`on_response` callback is not such a caller: `on_response` tests
`!self->draining_` before `FinishLocked(pull_status)` under the same `mu_`, so
once `Start()` has scheduled the pull, a receive whose data (or zero-layer
readiness poll) outran a failing control reply keeps its OK verdict
(`pullReply false` on a draining session only ends the op;
`trace_late_reply_failure_keeps_verdict`).

**Bounded host layer staging.** The model is the full host mirror, one host
layer buffer per layer (`KVCacheManagerBase::num_host_layers() == num_layers()`,
the default; the `num_staging_layers` constructor argument, checked by
`ResolveNumStagingLayers`, bounds it). When fewer buffers than layers exist, a
layer's payload is written into a buffer leased from the session's
`StagingLayerManager`: each chunk's `KVCacheManagerBase::GetBlockChunks` and
companion `ValidateChunks` → `KVCacheManagerBase::GetBlockArrayHostSpan` →
`KVCacheManagerBase::ResolveBlockArrayHostLayer` (both inside the push lease,
from `StartStream`'s payload loop, each taking `KVCacheManagerWithTransfer::mu_`
in `GetSession(uuid)`) call `session->ResolveStagingSlot(layer_idx)` →
`TransferReceiveSession::AcquireStagingSlot` (on a full mirror `GetBlockChunks`
and `ResolveBlockArrayHostLayer` address the layer's own buffer directly when
`!explicit_pools_`, and `AcquireStagingSlot` itself returns `NotFound`, "no
bounded host staging"), which takes an op (`if (done_ || draining_)` refusal, `++in_flight_`,
the pattern of `TryBeginOp`), waits in `StagingLayerManager::Acquire` until a
buffer is free, the layer already holds one, the session's `deadline_` passes
or `cancelled_` is set (`FinishLocked` also calls
`staging_layer_manager_.Shutdown()` to wake waiters), then under `mu_` fails
the receive on a timeout (`FinishLocked(slot.status(), kReceivePhase)`),
returns a lease won by a draining session, and ends the op
(`EndRecvOpLocked()`). `ExecuteLayerH2d` then reads the leased buffer
(`LookupStagingSlot`, before its first lock, failing with `FailedPrecondition`
if the lease is gone) and dispatches with it as `host_layer_idx`; the lease is
returned on each early return, by the H2D callback after its `FinishLocked`
(`self->staging_layer_manager_.Release(layer_idx)`), and wholesale at settle
(`ReleaseAll()` in `ReleaseStagingLocked`). Nothing proved here changes under
that mode: the extra op is nested inside the push lease on the push thread
(taken after `pushBegin`, ended before the stream's `CompleteIncomingPush`), so
`in_flight_` is transiently one higher while `pushes > 0` already holds and
every settle decision is the one the model makes; a timeout, a cancelled wait
or a failed `LookupStagingSlot` fails the stream, which is `cancel` before
`pushEnd` (the unguarded `cancel` admits it), and a lookup can only fail on a
draining session (its lease was released by an early return or `ReleaseAll`).
`PrefillDecode/LayerStaging.lean` (with `Transfer/LayerStaging.lean`) models
`StagingLayerManager`, `TransferReceiveSession::AcquireStagingSlot`,
`decodeSlots` reuse across layers, and the unlocked gaps around `Acquire`,
proving publication correctness, slot conservation/exclusivity,
drain-to-settle termination, and the necessary and sufficient `1`-slot bound
(`LayerStaging.reachable_safe`, `LayerStaging.reachable_can_settle`,
`LayerStaging.no_slots_cannot_publish_ok`, `LayerStaging.exists_complete_trace`).

### Host destination (`BlockLocation::kHost`)

The model is a receive into device blocks (`destination_ ==
BlockLocation::kDevice`, the default of every `Create`). A `StartRead` with
`destination = BlockLocation::kHost` lands the pull in the caller's own host
blocks (`StartRead` copies `local_block_ids` into `local_host_block_ids`, which
`AllocateStagingForLoad` returns without leasing staging) and settles without an
H2D copy: `OnDataReceived` with no block ids routes to `SettleHostLayer` instead
of `ExecuteLayerH2d`, which under `mu_` returns on `done_ || draining_`, counts
`++num_completed_layers_`, and on the last layer runs
`RecordNetworkCompleteLocked(/*all_complete=*/true, ...)` — setting
`network_completed_ = true`, recording the last-packet and end metrics, and
calling `FinishLocked()`. Nothing is issued, no H2D op is taken (`in_flight_`
counts only pushes and the handshake), `h2d_futures_` stays empty, and
`CurrentPhaseLocked` never answers `h2d`. The later block reports (the transport
reports a layer before its blocks) find the receive draining and return at
`RecordBlockShards`'s first check.

This path is not a branch of `step`: the H2D events have no counterpart and
`issued`, `ready` and `retired` stay `0` while `completed` climbs, which
`CountersOrdered` forbids. Its properties follow by inspection:
- `SettleSafe`, `StagingIntegrity`, `SettlesPromptly` and `NoOpLeak` are the
  `Lifecycle` facts over `pushBegin`, `pushEnd`, `pullReply`, `cancel`,
  `publish` and the last layer's `FinishLocked()`.
- For `num_layers() > 0`, `IsReadyToComplete` stays false until that
  `FinishLocked()` (`num_completed_layers_ < num_layers` and
  `network_completed_` unset) and the receive is draining thereafter, so the
  readiness poll never finishes a positive-layer host receive. At
  `num_layers() == 0`, `num_completed_layers_ == 0 == total_layers` makes
  `IsReadyToComplete()` true from construction, so an intervening poll calls
  `Finish(OkStatus())` and sets `draining_ = true` while the pull handshake op
  is still in flight, causing `Start`'s `on_response`
  (`if (!pull_status.ok() && !self->draining_)`) to ignore a failed pull
  handshake (`trace_zero_layers_poll`).
- `done_recving` requires `num_completed_layers_ == num_layers`, counted once
  per `OnLayerReceived`; A1 makes that "every layer landed" (`SettleHostLayer`
  keeps no `dispatched_layers_`-style set, so a re-delivered layer would count
  twice, just as in the device callback's `num_completed_layers_++`).

`step` does include `on_response`'s `!self->draining_` check, which ignores a
failed `pullReply` after a fast settle (shared with device receives, and easiest
to hit when a host receive has no H2D wait). The `HostMemoryTransferTest`s
(`kv_cache_manager_with_transfer_host_memory_test.cc`) drive this path end to
end; a dedicated model would replace `Recv`'s H2D events with one
`hostLayerSettle`.

## Assumptions

* **A1 (one dispatch per layer).** `BlockTransportServerPushSession::CompleteIncomingPush`
  fires `OnLayerReceived` once per layer
  (`if (array_complete && !progress.on_layer_received_called)`), so
  `ExecuteLayerH2d` is entered at most `numLayers` times. Encoded as the
  `h2dBegin` guard `issued + pending < numLayers`. The once-latch lives in the
  layer's entry of the uuid's push-session table `layer_progress_`; once every
  layer has been called (`if (all_layers_called)`) `CompleteIncomingPush` marks
  the push session `done_`, `BlockTransport::HandleIncomingPush` drops it
  (`if (session->IsDone())` → `DeletePushSession(session)`), and the next push
  for the uuid gets a fresh table from `GetOrCreatePushSession`; A1 therefore
  also assumes no sender re-pushes a uuid after its push session was retired
  while the receive is still live.
  `Transport.lean` models the table and proves A1 (`Transport.OnceDispatched`)
  from exactly that assumption (its A4, honest senders);
  `Transport.trace_dup_after_retirement` is the re-push. The session itself
  keeps the network phase honest against a re-delivered layer
  (`dispatched_layers_` is a set, so a duplicate does not complete it —
  `RecvLifecycleTest.RedeliveredLayerDoesNotCompleteNetworkPhase`), but still
  issues the duplicate's copy, so A1 remains what bounds `issued`.
* **A2 (registration).** The session has blocks to receive
  (`total_blocks_ > 0`, which is also what the `total_blocks_ == 0` disjuncts
  of the first checks in `ExecuteLayerH2d` and `RecordBlockShards` test) and is
  seated in `sessions_`. `ReleaseStaging()` is called only by the destructor
  (`~TransferReceiveSession`): a candidate that `EmplaceRecvSessionLocked`
  rejects (`try_emplace` fails; from `StartRead`, `RegisterRecv` or
  `RegisterActivePlan`) is dropped by its creator before it was ever shared, so
  its release is not an event here.
* **A3 (no double end).** Every op ends at most once, so `in_flight_` never
  underflows: `h2dDone` is only enabled while a callback is outstanding,
  `pushEnd` only while a push is open.
* **A4 (transport ordering).** For the request that completes a layer,
  `BlockTransportServerPushSession::StartStream` (reached from
  `BlockTransport::HandleIncomingPush`, dispatched from `HandleCustomRequest`)
  calls `CompleteIncomingPush` → `delegate_->OnLayerReceived` →
  `KVCacheManagerBase::OnLayerReceived` → `session->OnDataReceived` →
  `ExecuteLayerH2d` (a plan-declared stream calls `session->OnDataReceived`
  directly) before `delegate_->OnBlockShardsReceived` →
  `KVCacheManagerBase::OnBlockShardsReceived` → `session->OnDataReceived` →
  `RecordBlockShards`, on the same thread, and `ExecuteLayerH2d` pushes the
  future into `h2d_futures_` (`h2d_futures_.push_back(future)`) before
  returning. A layer's blocks are accounted as one event once its copy is
  issued: guard `layersAccounted < issued`. With several streams per layer
  (several senders, or one sender's push split by source NUMA node into shard
  groups, each stream reporting only the shards it carried) the per-shard
  counters grow before `OnLayerReceived`, but every shard can only *reach*
  `total_blocks_ * num_layers` in `RecordBlockShardsReceivedLocked` once every
  stream has reported, so the call that sets `network_completed_` runs after
  every layer's completing stream has reported, and each such report follows
  that stream's own `OnLayerReceived` on the same thread; lumping each layer's
  accounting into that event is sound for everything proved here. This relies
  on streams reporting exactly their planned block count once (the counters in
  `RecordBlockShardsReceivedLocked` are plain sums compared with
  `received >= per_shard_target`): over-reporting would complete a shard
  early, which is precisely the `netAccountUnordered` mutant. `Transport.lean`
  proves the coarsening (`Transport.NetworkSound`) from honest senders and
  shows the over-report (`Transport.trace_dup_chunk_publishes_missing_layer`).

## Properties

All proved on every reachable state (`reachable_safe`):

* **Settle safety.** `done → inFlight = 0`: a settled session owns no work, so
  its blocks and staging can be reused. This is what the comment in
  `ExecuteLayerH2d` relies on ("A copy already handed to the device cannot be
  revoked; its in-flight count keeps the receive -- and its blocks -- owned
  until the copy ends").
* **No retired callback.** `done → retired = issued`: the
  `LOG(DFATAL) << "H2D callback for retired receive"` in `ExecuteLayerH2d`'s
  `future.OnReady` callback is unreachable.
* **Staging integrity.** `hasStaging = !done`.
* **Prompt settle.** `draining → inFlight = 0 → done`.
* **Readiness is sound.** `IsReadyToComplete → ready = numLayers`: when the
  poll decides the receive is complete, every layer's copy has been issued
  *and* has finished on the device. For the dispatch setter of
  `network_completed_` this holds directly: `netIssued` is set only by the
  issue that makes `issued = numLayers` (`Inv.netIssued_issued`), and
  `AllH2dDoneLocked` then says `ready = issued`. For the block-accounting
  setter it is sound only because of A4 (the mutant `netAccountUnordered` below
  shows what breaks without it).
* **Publication.** `published = some true → completed = numLayers`: when the
  engine is told `done_recving`, every layer's H2D callback has run with an
  OK status. Counter-level form of *publication correctness*;
  `Pipeline.lean` adds the memory contents.
* **Counters.** `completed ≤ retired ≤ ready ≤ issued ≤ numLayers` and
  `layersAccounted ≤ issued`.
* **No op leak.** `0 < inFlight →` some event in `drainEvents` is enabled:
  every accounted unit of `in_flight_` has an owner that can advance or retire
  it (`NoOpLeak`), and every reachable state can drain to `done = true` and
  release its staging in a finite number of steps (`reachable_can_settle`).
* **Published is quiet (`PublishedQuiet`).** `published ≠ none → pending = 0 ∧ retired = issued`:
  once `poll_stats()` has reported the receive, as `done_recving` or as
  `failed_recving`, no H2D copy is dispatching or still writing decode HBM, so
  the engine may reuse it. The receiver's promise to the engine at the session
  level (the pipeline restates it as `DecodeHbmSafe`), and the only property of this
  module that dropping `Done()` before publishing a *failed* receive violates.

Note on the readiness predicate. One might expect `IsReadyToComplete` to
coincide with "every layer's callback has run OK"; it does not.
`IsReadyToComplete` tests futures (`ready`), and callbacks (`retired`,
`completed`) can lag behind, so the shipping predicate can be true before
`num_completed_layers_` reaches `numLayers` (`trace_poll_before_callbacks`).
That is harmless — `done` still waits for every callback through `in_flight_`
— and the publication property above is the one that matters.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `recv.h` | `TransferReceiveSession::~TransferReceiveSession` | `:97` |
| `recv.h` | `TransferReceiveSession::Done` | `:104-107` |
| `recv.h` | `TransferReceiveSession::GetStatus` | `:111-114` |
| `recv.h` | `TransferReceiveSession::IsDraining` | `:121-124` |
| `recv.h` | `TransferReceiveSession` (`bool TryBeginOp() override;` .. `void EndOp() override;`) | `:135-136` |
| `recv.h` | `TransferReceiveSession` (`absl::Status OnDataReceived(size_t major_dimension_idx,` .. `absl::Span<const int> shard_ids) override;`) | `:150-152` |
| `recv.h` | `TransferReceiveSession::ResolveStagingSlot` | `:160-162` |
| `recv.h` | `TransferReceiveSession` (`const BlockLocation destination_ = BlockLocation::kDevice;`) | `:321` |
| `recv.h` | `TransferReceiveSession::tracks_completion` | `:168-171` |
| `recv.h` | `TransferReceiveSession::deadline` | `:176-179` |
| `recv.h` | `TransferReceiveSession` (`std::vector<int64_t> blocks_received_per_shard_` .. `std::vector<raiden::PjRtCopyFuture> h2d_futures_`) | `:329-357` |
| `recv.h` | `TransferReceiveSession` (`mutable absl::Mutex mu_;` .. `std::atomic<bool> cancelled_{false};`) | `:309-362` |
| `recv.h` | `TransferReceiveSession` (`absl::StatusOr<size_t> AcquireStagingSlot(size_t layer_idx);`) | `:202` |
| `recv.cc` | `TransferReceiveSession::InitFromActivePlan` | `:191-243` |
| `recv.cc` | `TransferReceiveSession::InitFromLoadPlan` | `:366-378` |
| `recv.cc` | `TransferReceiveSession::ReleaseStaging` | `:385-388` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:390-405` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` (`if (!status.ok() && status_.ok())`) | `:392-395` |
| `recv.cc` | `TransferReceiveSession::Finish` | `:407-410` |
| `recv.cc` | `TransferReceiveSession::AwaitForDone` | `:412-416` |
| `recv.cc` | `TransferReceiveSession::Details` | `:418-434` |
| `recv.cc` | `TransferReceiveSession::CurrentPhaseLocked` | `:436-441` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:443-454` |
| `recv.cc` | `TransferReceiveSession::TryBeginOp` | `:456-467` |
| `recv.cc` | `TransferReceiveSession::EndOp` | `:469-472` |
| `recv.cc` | `TransferReceiveSession::AllH2dDoneLocked` | `:474-479` |
| `recv.cc` | `TransferReceiveSession::IsReadyToComplete` | `:481-487` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShardsReceivedLocked` | `:496-521` |
| `recv.cc` | `TransferReceiveSession::RecordNetworkCompleteLocked` | `:523-539` |
| `recv.cc` | `TransferReceiveSession::Start` | `:541-627` |
| `recv.cc` | `TransferReceiveSession::Start` (`auto on_response =`) | `:568-608` |
| `recv.cc` | `TransferReceiveSession::Start` (`absl::Cleanup end_op`) | `:570` |
| `recv.cc` | `TransferReceiveSession::Start` (`absl::MutexLock lock(self->mu_);` .. `self->handshake_pending_ = false;`) | `:599-607` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShards` | `:629-678` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShards` (`if (done_ || draining_ || total_blocks_ == 0)`) | `:644-646` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:680-709` |
| `recv.cc` | `TransferReceiveSession::LookupStagingSlot` | `:711-719` |
| `recv.cc` | `TransferReceiveSession::OnDataReceived` | `:721-740` |
| `recv.cc` | `TransferReceiveSession::SettleHostLayer` | `:742-769` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:771-903` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`ABSL_ASSIGN_OR_RETURN(host_layer_idx, LookupStagingSlot(layer_idx));`) | `:777` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`if (done_ || draining_ || total_blocks_ == 0)` .. `trigger_enqueue = true;`) | `:783-791` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`// The receive may have been finished` .. `h2d_futures_.push_back(future);`) | `:803-835` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`if (!future_or.ok())`) | `:824-830` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`future.OnReady`) | `:839-888` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`absl::Cleanup end_op`) | `:846` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`absl::MutexLock lock(self->mu_);` .. `self->FinishLocked(status_or.status(), kH2dPhase);`) | `:850-870` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`dispatched_layers_.insert(layer_idx);` .. `network_completed_ = true;`) | `:894-897` |
| `mgr.h` | `*` (`enum class BlockLocation { kDevice, kHost };`) | `:99` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`struct SessionEntry {` .. `SessionRole role;`) | `:304-307` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` | `:305-309` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` |
| `mgr.cc` | `KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer` (`for (const auto& [uuid, entry] : sessions_)` .. `session->AwaitForDone().IgnoreError();`) | `:399-414` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` (`if (auto recv =` .. `transfer_details_[req_id] = recv->Details();`) | `:476-484` |
| `mgr.cc` | `KVCacheManagerWithTransfer::EmplaceRecvSessionLocked` | `:605-626` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterActivePlan` | `:628-707` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterRecv` | `:709-731` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:828-874` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` (`session->Finish(absl::CancelledError("receive plan unregistered"));`) | `:858` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:942-1074` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` (`if (destination == BlockLocation::kHost)` .. `local_host_block_ids = local_block_ids;`) | `:964-974` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` (`FaultInjectStatus(hooks::kKvCacheManagerApiStartReadSubmitPull);` .. `session->Finish(started);`) | `:1061-1071` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = sessions_.begin();`) | `:1084-1109` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`if (auto recv =` .. `absl::DeadlineExceededError("Transfer session timed out"));`) | `:1088-1100` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`auto current = it++;` .. `retired_uuids.push_back(uuid);`) | `:1103-1107` |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` |
| `base.cc` | `KVCacheManagerBase::GetSession` | `:3081-3087` |
| `base.cc` | `KVCacheManagerBase::OnBlockShardsReceived` | `:3323-3341` |
| `base.cc` | `KVCacheManagerBase::OnLayerReceived` | `:3345-3354` |
| `base.cc` | `*` (`size_t ResolveNumStagingLayers(size_t num_layers,` .. `return requested;`) | `:718-734` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`absl::StatusOr<size_t> slot = session->ResolveStagingSlot(layer_idx);`) | `:3503` |
| `bt.cc` | `BlockTransport::HandleCustomRequest` | `:407-430` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `bts.cc` | `ValidateChunks` | `:275-310` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`absl::Cleanup end_incoming_push`) | `:385-390` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`std::vector<BlockChunk> chunks = delegate_->GetBlockChunks(` .. `ValidateChunks(delegate_, l, sh, header.uuid, chunks));`) | `:449-459` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` |
| `send_drain_test.cc` | `RecvLifecycleTest.OutOfOrderLayersSettleAfterEveryH2d` | `:965-984` |
| `send_drain_test.cc` | `RecvLifecycleTest.DetailsReportWhenTheLastH2dLanded` | `:880-895` |
| `send_drain_test.cc` | `RecvLifecycleTest.DetailsSayAReceiveTimedOutWaitingForData` | `:897-910` |
| `send_drain_test.cc` | `RecvLifecycleTest.DetailsSayAFailedUploadFailedInH2dWhileDataArrives` | `:912-926` |
| `send_drain_test.cc` | `RecvLifecycleTest.SingleFailedH2dReportsFailureAndReturnsStaging` | `:986-997` |
| `send_drain_test.cc` | `RecvLifecycleTest.NetworkCompletionWaitsForH2d` | `:840-858` |
| `send_drain_test.cc` | `RecvLifecycleTest.EmptyBlockReportIssuesNoH2d` | `:860-878` |
| `send_drain_test.cc` | `RecvLifecycleTest.LateBlockAccountingAfterRetirementIsANoOp` | `:941-963` |
| `send_drain_test.cc` | `RecvLifecycleTest.SessionRemainsValidAcrossAsyncH2dCallbacksAfterCallerDropsPtr` | `:1237-1264` |
| `send_drain_test.cc` | `RecvLifecycleTest.ReceiveWithoutTrafficFailsAtItsDeadline` | `:999-1010` |
| `send_drain_test.cc` | `RecvDrainTest.ExpiredReceiveKeepsStagingUntilH2dEnds` | `:1012-1039` |
| `send_drain_test.cc` | `RecvDrainTest.FailedLayerWaitsForOtherH2dCopies` | `:1149-1175` |
| `send_drain_test.cc` | `RecvDrainTest.TimeoutDuringH2dDispatchKeepsStaging` | `:1177-1213` |
| `send_drain_test.cc` | `RecvLifecycleTest.IncomingPushLeasePinsStagingDuringWriteAndRejectsWhenDraining` | `:1266-1290` |
| `send_drain_test.cc` | `RecvLifecycleTest.IncomingPushLeaseSpansLayerH2dAndBlockAccountingBeforeReleasing` | `:1593-1621` |
| `send_drain_test.cc` | `RecvLifecycleTest.FailedIncomingPushImmediatelyFailsSessionAndReleasesStagingBeforeDeadline` | `:1623-1652` |
| `send_drain_test.cc` | `RecvLifecycleTest.FirstAdmittedPushRecordsFirstPacketOnce` | `:1292-1323` |
| `send_drain_test.cc` | `RecvLifecycleTest.AdmissionRejectedByDoneSessionRecordsNoFirstPacket` | `:1325-1357` |
| `send_drain_test.cc` | `RecvLifecycleTest.AdmissionRejectedByDrainingSessionRecordsNoFirstPacket` | `:1359-1400` |
| `send_drain_test.cc` | `RecvLifecycleTest.PeregrineLayerCompletionsRecordLastPacketOnceAndFinishAfterH2d` | `:1419-1459` |
| `send_drain_test.cc` | `RecvLifecycleTest.TcpLastLayerThenItsBlockReportRecordLastPacketOnceAndFinishOnce` | `:1461-1507` |
| `send_drain_test.cc` | `RecvLifecycleTest.TcpH2dFinishingBeforeLastBlockReportRecordsLastPacketAndFinishesOnce` | `:1509-1553` |
| `send_drain_test.cc` | `RecvLifecycleTest.RedeliveredLayerDoesNotCompleteNetworkPhase` | `:1555-1591` |
| `send_drain_test.cc` | `RecvLifecycleTest.RedeliveredStartReadLeavesAReadyReceiveToCompleteRead` | `:1092-1121` |
| `send_drain_test.cc` | `RecvLifecycleTest.RegisterRecvFinishesAReadyIncumbentBeforeReplacingIt` | `:1123-1147` |
| `send_drain_test.cc` | `RecvLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` | `:1722-1807` |
| `send_drain_test.cc` | `RecvLifecycleTest.BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock` | `:1809-1917` |
| `send_drain_test.cc` | `RecvLifecycleTest.GetSessionResolvesActiveRecvAndReturnsNotFoundAfterRetirement` | `:2203-2228` |
| `send_drain_test.cc` | `RecvLifecycleTest.FailedH2dDispatchFailsAndSettlesTheReceive` | `:2233-2252` |
| `send_drain_test.cc` | `RecvLifecycleTest.SecondFinishOnAFailedReceiveIsANoOp` | `:2256-2272` |
| `send_drain_test.cc` | `RecvLifecycleTest.PushLeaseClosingBeforeTheCopyEndsDoesNotSettle` | `:2278-2296` |
| `send_drain_test.cc` | `RecvDrainTest.CancelledReceiveWhoseCopyThenFailsStaysCancelled` | `:2301-2314` |
| `control_test.cc` | `ControlHandshakeTest.ExpiredReceiveKeepsStagingUntilHandshakeEnds` | `:816-856` |
| `control_test.cc` | `ControlHandshakeTest.AcceptedPullAdvancesReceivePhaseAndReleasesHandshakeOp` | `:895-928` |
| `control_test.cc` | `ControlHandshakeTest.AcceptedPullAfterDeadlineSweepSettlesExpiredReceive` | `:930-971` |
| `control_test.cc` | `ControlHandshakeTest.RejectedPullReportsWhyAndWhichProducerRejectedIt` | `:985-1003` |
| `control_test.cc` | `ControlHandshakeTest.RefusedConnectionKeepsItsStatusCode` | `:1005-1030` |
| `pool_reshard_test.cc` | `DemandStagingTest.UnregisteringIdleReceiverReleasesPlanAtOnce` | `:535-560` |
| `pool_reshard_test.cc` | `DemandStagingTest.UnregisteringInFlightReceiverDefersUntilItSettles` | `:562-603` |
| `pool_reshard_test.cc` | `DemandStagingTest.DemandStagedReceiverPlanUnregistersWhenItSettles` | `:665-688` |
<!-- source-map:end -->
-/

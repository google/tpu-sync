import TpuSyncVerify.Transfer.PrefillDecode.Send.Model
import TpuSyncVerify.Transfer.PrefillDecode.Send.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Send.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Send.Frame
import TpuSyncVerify.Transfer.PrefillDecode.Send.Traces

/-!
# Send session

Producer side of the prefill-to-decode model: one `TransferSendSession` on the
prefill (producer) side and the slice of `KVCacheManagerWithTransfer` that
drives it. The sender stages each layer's device blocks into host staging
(D2H), pushes the staging to the consumer (H2H) layer by layer, and must not
let the engine reuse the device blocks, or the allocator reseat the staging,
while any copy or push is still running. Memory contents are in `Pipeline.lean`.

`send.h`/`send.cc` are `tpu_sync/kv_cache/transfer_send_session.{h,cc}`;
`mgr.cc` is `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`; `base.cc` is
`tpu_sync/kv_cache/kv_cache_manager_base.cc`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`. Exact line ranges for every symbol,
sub-block, and test are in the generated `## Source map` table below.

## How a send runs

`StartPush` runs once on a worker thread spawned by the consumer's pull request
(`HandlePullStream`'s `std::thread`). It acquires staging
(`AcquireStagingWithRetry`), pre-creates every layer's D2H promise/future pair
(`d2h_promises`, `d2h_layer_futures_`) in its locked section, calls
`SendNextLayer(0)` so pushes can overlap with D2H dispatch, and then issues one
D2H copy per layer in a loop (`for (size_t l = 0; l < total_layers; ++l)`),
each counted in `in_flight_` from before its dispatch until its future's
`layer_future.OnReady` callback resolves `d2h_promises[l]` and ends the op
(`self->EndSendOp()`). When the loop returns early on a draining session, a
failed layer-buffer lease, or a failed dispatch, its `absl::Cleanup
fail_unissued_layers` resolves `d2h_promises[next_layer..total_layers-1]` with
the session's error status so an already-armed `SendNextLayer` callback wakes
with that error instead of hanging on an unset promise.

`SendNextLayer(l)` takes an op (`layer_future = d2h_layer_futures_[l]; ++in_flight_`)
and waits on layer `l`'s D2H future. When it resolves (`layer_future->OnReady`)
the layer is handed to the push pool (`self->base_->push_pool()->Schedule`),
whose task takes a second op (`++self->in_flight_`) for the H2H push,
issues it (`H2hWriteDirectAsync`), calls `SendNextLayer(l + 1)`, and only then
releases the first op via its `absl::Cleanup end_op`. The push's callback
releases the second op and, when the last layer's push completes
(`remaining_h2h_layers_.fetch_sub(1) == 1`), finishes the session OK
(`self->Finish()`).

So two chains run concurrently against the same `in_flight_`: the D2H copies,
issued in layer order by `StartPush`, and the pushes, chained strictly one
layer after another starting from `SendNextLayer(0)`. The session settles when
both have drained after a `Finish`.

## State

The settle protocol is `Transfer.Lifecycle`. On top of it:

| Field        | C++                                             | Role |
|--------------|-------------------------------------------------|------|
| `numLayers`  | `base_->num_layers()`                           | fixed |
| `pullStarted`| `pull_started_` (`TransferSendSession`)         | `ValidateAndBeginPull` claimed the offer |
| `started`    | `StartPush` passed its locked section (`host_block_ids.assign` .. `dst_ints_.assign`) | staging acquired, `d2h_layer_futures_` populated, copies may start |
| `d2hPending` | ghost                                           | the D2H loop is between `++in_flight_` and `D2hSyncDispatch`'s outcome |
| `d2hIssued`  | `next_layer` in `StartPush` (`size_t next_layer = 0;` .. `next_layer = l + 1;`) | copies handed to the device |
| `d2hReady`   | issued copies whose `d2h_promises[l]` is set (`IsReady()`) | copies finished |
| `d2hRetired` | ghost                                           | copies whose `layer_future.OnReady` in `StartPush` has ended their op |
| `queued`     | ghost                                           | `SendNextLayer(l)` calls that took their op (`++in_flight_`) |
| `woken`      | ghost                                           | of those, the future callback (`layer_future->OnReady`) has run |
| `pooled`     | ghost                                           | layers scheduled on `push_pool()->Schedule` whose task has not run |
| `h2hIssued`  | ghost                                           | pushes handed to the transport (`H2hWriteDirectAsync`) |
| `chaining`   | ghost                                           | pool tasks that issued their push and still hold the `SendNextLayer` op (`H2hWriteDirectAsync` .. `SendNextLayer(l + 1)`) |
| `h2hRetired` | ghost                                           | push callbacks (`H2hWriteDirectAsync` callback) that have run |
| `h2hOk`      | `num_layers - remaining_h2h_layers_` (`TransferSendSession`) | pushes that completed OK |
| `published`  | membership in `done_sending_` / `failed_recving_` (`CompleteReadWithDetails`) | what `poll_stats()` shows the engine |
| `underflow`  | —                                               | `EndSendOpLocked` ran with `in_flight_ == 0` |

Ghost fields record where each unit of `in_flight_` came from, which is what
`Accounted` is about. `underflow` exists because `EndSendOpLocked`
decrements `--in_flight_` without a guard; the model records the event
instead of silently saturating so that `NoUnderflow` is a theorem.

## Events

| Event          | C++ |
|----------------|-----|
| `beginPull`    | `ValidateAndBeginPull` (called from `HandlePullStream`): rejects if `deadline_ <= now` (modelled together with pre-pull cancellation via `draining`/`done`) or `pull_started_` is already set, otherwise sets `pull_started_ = true` |
| `start`        | `StartPush` from the acquisition (`AcquireStagingWithRetry`) through its locked section and `SendNextLayer(0)` (A6), spawned after `ValidateAndBeginPull`: a zero-layer send (`total_layers == 0`) calls `FinishLocked(absl::OkStatus())` at once; a positive-layer send populates `d2h_promises` / `d2h_layer_futures_` and attempts `SendNextLayer(0)` |
| `d2hBegin`     | one iteration of `StartPush`'s D2H loop up to `++in_flight_` (`if (draining_) { staging_layer_manager_.Release(l); return; } ++in_flight_;`) |
| `d2hIssue ok`  | the dispatch (`D2hSyncDispatch`): failure finishes the session and ends the op (`FinishLocked(future.status());` then `EndSendOp();`); success registers `layer_future.OnReady` to resolve `d2h_promises[l]` and call `EndSendOp()`, and advances `next_layer = l + 1` |
| `d2hReady`     | the device finishes an issued copy: its `layer_future.OnReady` resolves `d2h_promises[l]` (`IsReady()`) |
| `d2hEnd`       | the copy's `layer_future.OnReady` in `StartPush` ends its op via `EndSendOp` |
| `wake ok`      | `SendNextLayer`'s `layer_future->OnReady` callback (either after an issued copy resolved `d2h_promises[l]`, or — with `ok = false` — after the D2H loop exited early while `draining_` and `fail_unissued_layers` resolved `d2h_promises[next_layer..total_layers-1]` with an error): a failed copy or failed unissued promise finishes the session (`self->FinishLocked(status_or.status(), kD2hPhase)`); a draining session drops the layer (`if (self->draining_)`); otherwise the layer is scheduled on `push_pool()->Schedule` and `std::move(end_op).Cancel()` carries the op over |
| `h2hIssue`     | the `push_pool` task up to the push: a draining session (`if (self->draining_)`) drops the layer and ends the op via `absl::Cleanup end_op`; otherwise a second op is taken (`++self->in_flight_`) and the push issued (`H2hWriteDirectAsync` → `BlockTransport::AsyncPush`, which fans a layer out into one stream group per source NUMA node via `AsyncPushShards` and joins them via `tsl::JoinFutures` into this one callback; the callback may even run inline on an argument error) |
| `sendNext`     | the pool task's `SendNextLayer(l + 1)` followed by the release of the carried-over op (`absl::Cleanup end_op`) |
| `h2hDone ok`   | the `H2hWriteDirectAsync` push callback: failure finishes the session (`self->FinishLocked(push_res.status(), kPushPhase)`); the last OK push (`remaining_h2h_layers_.fetch_sub(1) == 1`) finishes it OK (`self->Finish()`); the op ends either way |
| `cancel`       | any `Finish(error)` from outside the data path: deadline (`CompleteReadWithDetails`), shutdown (`~KVCacheManagerWithTransfer`), pull-worker spawn failure (`HandlePullStream`'s `counter_cleanup`), staging acquisition failure (`!acquire_status.ok()`), layer-buffer lease failure (`!host_layer_idx.ok()`), bad arguments in `StartPush` (`src_block_ids.empty() || ...`), pull-worker exceptions (`HandlePullStream`'s `catch`) |
| `publish`      | `CompleteReadWithDetails` / `CompleteReadRaw` moves a settled session (`session->Done()`) into `done_sending_` or `failed_recving_` by its status and erases it from `send_sessions_` |

## Lock scopes

Every mutable field of `TransferSendSession` is `ABSL_GUARDED_BY(mu_)` except
`remaining_h2h_layers_` (an atomic read only by its own `fetch_sub`),
`cancelled_` (an atomic set in `FinishLocked` to wake `StagingLayerManager`),
and `staging_layer_manager_` (internally synchronized); `base_`,
`staging_allocator_`, `req_id_`, `uuid_`, `deadline_`, `register_start_` and
`data_connection_reuse_` are `const`. Every `EndSendOp` below is one `mu_` scope
under the default `pooled` connection policy and the A7 window under
`per_transfer`.

| Event          | `mu_` scopes | Between them |
|----------------|--------------|--------------|
| `beginPull`    | 1: `ValidateAndBeginPull` | — (`deadline_` is `const`) |
| `start`        | `StartPush`'s locked section (`if (draining_ || done_) return;` through `dst_ints_.assign`, populating `d2h_promises` and `d2h_layer_futures_`) plus `SendNextLayer(0)`'s scope (A6), preceded by `remote_data_endpoints_ =` and the scopes of `AcquireStagingWithRetry` (which touch only `remote_data_endpoints_` and `staging_`) | a `Finish` during `AcquireWithTimeout` hits `if (draining_ || done_)` in `AcquireStagingWithRetry` and drops the local `StagingAllocation` before `staging_` is assigned; a `Finish` between `staging_ = *std::move(acquired)` and `StartPush`'s locked section settles the idle session and releases `staging_` (`FinishLocked` runs `ReleaseStagingBlocksLocked()` at `in_flight_ == 0`), after which the locked section returns — both match `cancel` before `start`. A failed acquisition or bad arguments call `Finish(error)` (`cancel`) |
| `d2hBegin`     | 1 deciding scope: the loop's `session_lock` (`if (draining_) { staging_layer_manager_.Release(l); return; } ++in_flight_;`), preceded by `AcquireStagingSlot` (below) | if the loop returns here (or earlier on `!host_layer_idx.ok()`), `fail_unissued_layers` resolves `d2h_promises[next_layer..total_layers-1]` with the session's error status so any armed `SendNextLayer` callback can run `wake false` |
| `d2hIssue`     | on success, 0 `mu_` scopes (`D2hSyncDispatch` and `layer_future.OnReady` run outside `mu_`, since `d2h_promises[l]` was pre-allocated in `start`); on failure, `FinishLocked(future.status())` (1 scope) followed by `EndSendOp()` | a `Finish` during `D2hSyncDispatch` finds the op `d2hBegin` took and cannot settle; `layer_future.OnReady` is registered regardless of `draining_` (matching `d2hIssue true`), and the copy drains through `d2hEnd` (`SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds`). On `!future.ok()`, `fail_unissued_layers` resolves `d2h_promises[l..total_layers-1]` after `EndSendOp()`. If `layer_future` is already ready when `OnReady` is registered, the callback runs inline (`d2hReady` then `d2hEnd`) |
| `d2hReady`     | 0 (`d2h_promises[l].Set(status_or.status())` in `StartPush`'s `layer_future.OnReady`) | — |
| `d2hEnd`       | 1: `EndSendOp` | — |
| `wake`         | on `ok = false` or `draining_`: 2 (`FinishLocked` or the `if (self->draining_)` check, then `EndSendOp` from `absl::Cleanup end_op`); otherwise 1 (`if (self->draining_)` check, then `std::move(end_op).Cancel()` carries the op into the unlocked `push_pool()->Schedule`) | a `Finish` after the check and before the pool task runs is caught by the task's own `if (self->draining_)` check (`wake true`, `cancel`, `h2hIssue` ending the op) |
| `h2hIssue`     | when draining: 2 (the task's `if (self->draining_)` check, then `EndSendOp` from `end_op`); otherwise 1 (the check through `++self->in_flight_`, followed by the unlocked `H2hWriteDirectAsync`) | a `Finish` between `++self->in_flight_` and `H2hWriteDirectAsync` does not stop the push (matching `h2hIssue`), and the push callback later retires the op (`Finish()` on the last OK push is a no-op once draining). Splitting the pool task into `h2hIssue` and `sendNext` via `chaining` also admits a push callback that runs before `SendNextLayer(l + 1)`, including inline (`h2hIssue`, `h2hDone`, `sendNext`) |
| `sendNext`     | for `l + 1 < num_layers`: 2 (`SendNextLayer(l + 1)`'s scope from `if (draining_ || l >= d2h_layer_futures_.size()) return;` to `++in_flight_;`, then `EndSendOp` from the pool task's `end_op`); for the last layer, `SendNextLayer` returns before its lock (`if (l >= base_->num_layers()) return;`), leaving only `EndSendOp` | an `EndSendOp` tail, below |
| `h2hDone`      | on `ok = false`: 2 (`FinishLocked(push_res.status(), kPushPhase)`, then `EndSendOp` from `end_op`); on `ok = true`: lock-free `remaining_h2h_layers_.fetch_sub(1)`, then `Finish()` (1 scope) if it was the last layer, then `EndSendOp` | nothing else reads `remaining_h2h_layers_`, so any step between `fetch_sub` and `Finish()` orders before `h2hDone` (a `cancel` in that gap wins, matching `cancel` then `h2hDone true` because `finishOnceLocked` ignores the later OK); after `Finish()`, an `EndSendOp` tail |
| `cancel`       | 1: `Finish(error)` (for the deadline sweep, `!IsDraining()` is checked in an earlier scope; `deadline()` is `const`) | `TransferSendSession::FinishLocked` ignores every call after the first (`if (draining_ || done_) return;`), so a stale sweep is a no-op (`trace_cancel_after_ok_finish`, `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess`) |
| `publish`      | under manager `mu_`: `Done()`, `GetStatus()`, `Details()` | nothing changes on a settled sender (`TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone`), so no late error can split the published set from `transfer_details_` |

**`EndSendOp` tails.** `d2hIssue false`, `wake`, `h2hIssue`, `sendNext` and
`h2hDone` all run `EndSendOp` after releasing the deciding scope's lock.
Between the two scopes, `in_flight_` still counts the op. Because only the
settle checks in `FinishLocked` and `EndSendOpLocked` (and A7's `release_due`)
read `in_flight_`, any step in the gap makes the same decision as after
`EndSendOp`, except that a `Finish` in the gap leaves the final settle to
`EndSendOp` with the same resulting `draining_`, `done_` and `status_`. Every
such interleaving therefore reaches the same state as running the whole event
first, keeping the ghost counters (`d2hRetired`, `woken`, `pooled`, `chaining`,
`h2hRetired`) sound.

**Unchecked dispatches.** Neither `d2hIssue` (when registering
`layer_future.OnReady`) nor `h2hIssue` (when calling `H2hWriteDirectAsync`)
re-reads `draining_`; the model matches this directly, so a `cancel` between
`d2hBegin` and `d2hIssue`, or between `h2hIssue`'s lock and the transport call,
is admitted in the same order. The only multi-scope fold across distinct
operations is A6 (`SendNextLayer(0)` folded into `start`).

### Bounded host layer staging (`TPU_RAIDEN_NUM_STAGING_LAYERS`)

When `TPU_RAIDEN_NUM_STAGING_LAYERS` is set below `num_layers`
(`ResolveEnvNumStagingLayers` in `base.cc`), the host mirror has fewer layer
buffers than `num_layers` and each `TransferSendSession` arbitrates them
through `staging_layer_manager_` (`StagingLayerManager`). Before `d2hBegin`,
each D2H loop iteration calls `TransferSendSession::AcquireStagingSlot(l)`,
which checks `if (done_ || draining_)` under `mu_`, blocks in
`StagingLayerManager::Acquire(l, deadline_, &cancelled_)` until a buffer is
free, `deadline_` expires or `cancelled_` is set (`FinishLocked` sets
`cancelled_` and calls `staging_layer_manager_.Shutdown()` to wake waiters),
calls `FinishLocked(slot.status(), kStagingPhase)` on a non-cancelled failure,
and under `mu_` returns a lease won after `done_ || draining_`; if
`AcquireStagingSlot` fails, `StartPush` calls `Finish(host_layer_idx.status())`
and returns (triggering `fail_unissued_layers`). While pushing, `BlockTransport`
resolves layer `l`'s host pointer through `hooks.lookup_send_staging` →
`TransferSendSession::LookupStagingSlot(l)` (`staging_layer_manager_.Lookup`).
Each leased buffer is released on early return (`d2hBegin` `if (draining_)`,
`d2hIssue` `!future.ok()`, `wake` error or `draining_`, `h2hIssue` `draining_`),
by the H2H callback (`h2hDone` on both error and success, before reusing the
buffer for a waiting layer), and wholesale at settle (`ReleaseAll()` in
`ReleaseStagingBlocksLocked`). Nothing proved here changes under that mode:
`AcquireStagingSlot` takes no `in_flight_` op, its failure is `cancel` before
`d2hBegin`, and because `SendNextLayer(0)` was armed in `start` before the D2H
loop, pushes of earlier layers still complete and release buffers while the D2H
loop is blocked in `AcquireStagingSlot` (`SendLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers`,
`SendLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes`).
What this module does not model is the host-buffer reuse across layers itself
and the bounded wait in `Acquire`; `Pipeline.lean` keeps one `prefillStaging`
cell per layer.

## Assumptions

* **A1 (at-most-once `StartPush`).** `ValidateAndBeginPull`
  rejects a duplicate pull or an expired offer and sets `pull_started_ = true`;
  the model's `beginPull` does the same and gates `start`
  (`PullClaimed`), so a second `StartPush` on one session is unreachable.
* **A2 (sequential loop).** The D2H loop is one thread, so at most one
  dispatch is between its `++in_flight_` and its outcome: `d2hPending` is a
  `Bool`.
* **A3 (layers are interchangeable).** The model counts copies and pushes
  rather than naming layers. `SendNextLayer` is chained in layer order
  (`0, 1, …`), so when `SendNextLayer(k)`'s callback wakes on an issued layer in
  C++, layers `0 … k-1` have already woken (and were issued because the D2H loop
  issues in layer order `0, 1, …`) and layer `k`'s future is ready — hence at
  least `k + 1` futures are ready (`woken < d2hReady`), even if later layers'
  D2H copies finished earlier; if `k ≥ next_layer` (`d2hIssued ≤ woken`), layer
  `k` was never issued and its promise can only be resolved with an error by
  `fail_unissued_layers` once the D2H loop has exited (`draining ∧ ¬d2hPending`).
  `Pipeline.lean` indexes every layer's readiness and memory cell explicitly.
* **A4 (no double end).** Each op ends once: every op-ending event is guarded
  by the ghost counter that opened it.
* **A5 (no consumer Ack).** `HandleAck → AckSend → session->Finish()`
  would finish a send OK from outside. No code in the repository
  sends an Ack (`SendAck` has only test callers), so it is not an
  event; nor is the ack-first path of `NotifyForRead`, which reports
  `done_sending` without registering the session when `pending_acks_.erase(uuid) > 0`
  (filled only by `AckSend` when `it == send_sessions_.end()`). With it,
  `Publication` below would read "the consumer acked", not "every layer was
  pushed".
* **A6 (fold of `SendNextLayer(0)` into `start`).** Between `StartPush`'s locked
  section (which populates `d2h_layer_futures_` for all `total_layers`) and
  `SendNextLayer(0)`'s lock (`if (draining_ || l >= d2h_layer_futures_.size()) return;`),
  only `BuildCoalescedCopySpec` and `remaining_h2h_layers_.store(total_layers)`
  run, and only an external `Finish` can intervene: if `Finish` lands between
  the two locks, `StartPush` has no `in_flight_` op yet, so `FinishLocked`
  settles the session at once (`done_ = true`) and `SendNextLayer(0)` returns at
  `if (draining_)` — the exact state reached by `cancel` before `start`. The
  locked section and `SendNextLayer(0)` therefore fold into one event `start`.

* **A7 (connection release window).** `EndSendOp`
  is modelled as the atomic `endOp`. Under `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer`
  (default `pooled`, which never enters this branch) the last op of a draining
  send first drops `mu_` to close the idle data connections to the consumer
  (`ReleaseDataConnectionsAfterSend`; the branch `release_due` also needs
  `!remote_data_endpoints_.empty()` and `!connections_released_`) and only then
  runs `EndSendOpLocked`. In that window `in_flight_ == 1`, `draining_` and
  `!done_` hold: no new op can begin (`beginOp` refuses while `draining_` in
  `StartPush` and `SendNextLayer`), no other op can end (this is the only one),
  `FinishLocked` returns at `if (draining_ || done_) return;`,
  `ValidateAndBeginPull` throws on `if (pull_started_)`, and the poll cannot
  publish (`Done()` is false), so no modelled variable changes between the
  check and the decrement; `connections_released_` only stops a second pass.
  Every op exit — including `!future.ok()` in `StartPush`, which can be the
  last op when earlier layers already pushed
  (`TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenFailedDispatchEndsSend`) —
  goes through `EndSendOp()`. The window delays `done_`, which every trace
  already allows between events.

## Properties

All proved on every reachable state (`reachable_safe`):

* **Settle safety.** `done → inFlight = 0`.
* **Drained.** `done →` no copy is pending or unretired, no `SendNextLayer`
  callback, pool task or push is outstanding. In particular no D2H copy still
  reads the device blocks (so the engine may reuse them) or writes the staging,
  and no push still reads the staging (so the allocator may reseat it). This is
  what the comments on `in_flight_` (`TransferSendSession`) and the
  send-deadline loop in `CompleteReadWithDetails` promise.
* **Staging integrity.** `hasStaging = !done`: staging, once acquired, is held
  exactly until the session settles.
* **Prompt settle.** `draining → inFlight = 0 → done`.
* **No underflow.** `EndSendOpLocked` never runs on an idle session.
* **Publication.** `published = some true → h2hOk = numLayers`: when the
  engine is told `done_sending`, every layer's push completed OK. Counter-level
  form of *publication correctness* for the producer.
* **Counters.** The two chains are ordered:
  `d2hRetired ≤ d2hReady ≤ d2hIssued ≤ numLayers`,
  `woken ≤ queued ≤ numLayers`, `woken < d2hIssued → woken < d2hReady`,
  `pooled + h2hIssued ≤ woken`, `h2hOk ≤ h2hRetired ≤ h2hIssued`.
* **Pull claim (`PullClaimed`).** `(started → pullStarted) ∧ ((d2hPending ∨ 0 < d2hIssued) → started)`:
  `StartPush` never runs unless `ValidateAndBeginPull` claimed `pull_started_`,
  and no D2H copy is pending or issued unless `StartPush` started.
* **No op leak.** `0 < inFlight →` some event in `drainEvents` is enabled:
  every accounted unit of `in_flight_` has an owner that can advance or retire
  it (`NoOpLeak`), and every reachable state can drain to `done = true` and
  release its staging in a finite number of steps (`reachable_can_settle`).

Correspondence notes. (1) The sender's `FinishLocked` ignores every call after
the first (`Lifecycle.finishOnceLocked`), unlike the receiver's. An OK finish
happens inside the last push callback while D2H `OnReady`s may still be
outstanding, so a shutdown or deadline in that window leaves the status OK
(`trace_cancel_after_ok_finish`); harmless, since the send did complete.
(2) A failed send is reported in `failed_recving_` (`CompleteReadWithDetails`),
there is no `failed_sending_`. (3) When the D2H loop exits early (`!host_layer_idx.ok()`,
`if (draining_) return;`, or `!future.ok()`), `fail_unissued_layers` resolves
`d2h_promises[next_layer..total_layers-1]` with the session's error status so an
already-armed `SendNextLayer` callback wakes via `wake false` and retires its op
(`trace_d2h_dispatch_fail_drains_issued`, `trace_without_fail_unissued_leaks`).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `send.h` | `TransferSendSession::Done` | `:69-72` |
| `send.h` | `TransferSendSession::GetStatus` | `:76-79` |
| `send.h` | `TransferSendSession::IsDraining` | `:87-90` |
| `send.h` | `TransferSendSession::deadline` | `:149` |
| `send.h` | `TransferSendSession` (`mutable absl::Mutex mu_;` .. `const DataConnectionReuse data_connection_reuse_;`) | `:207-215` |
| `send.h` | `TransferSendSession` (`bool pull_started_` .. `StagingLayerManager staging_layer_manager_;`) | `:227-245` |
| `send.cc` | `*` (`constexpr absl::string_view kAwaitPullPhase` .. `constexpr absl::string_view kPushPhase`) | `:57-60` |
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:145-159` |
| `send.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:186-190` |
| `send.cc` | `TransferSendSession::AcquireStagingSlot` | `:197-226` |
| `send.cc` | `TransferSendSession::LookupStagingSlot` | `:228-243` |
| `send.cc` | `TransferSendSession::FinishLocked` | `:245-260` |
| `send.cc` | `TransferSendSession::Finish` | `:262-271` |
| `send.cc` | `TransferSendSession::Details` | `:279-295` |
| `send.cc` | `TransferSendSession::CurrentPhaseLocked` | `:297-305` |
| `send.cc` | `TransferSendSession::EndSendOpLocked` | `:307-314` |
| `send.cc` | `TransferSendSession::EndSendOp` | `:316-343` |
| `send.cc` | `TransferSendSession::AcquireStagingWithRetry` | `:376-410` |
| `send.cc` | `TransferSendSession::StartPush` | `:412-559` |
| `send.cc` | `TransferSendSession::StartPush` (`if (src_block_ids.empty()`) | `:423-428` |
| `send.cc` | `TransferSendSession::StartPush` (`remote_data_endpoints_ = remote_data_endpoints;`) | `:431` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Status acquire_status =` .. `dst_ints_.assign`) | `:434-467` |
| `send.cc` | `TransferSendSession::StartPush` (`SendNextLayer(0);` .. `for (size_t l = next_layer; l < total_layers; ++l)`) | `:483-503` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::StatusOr<size_t> host_layer_idx = AcquireStagingSlot(l);` .. `next_layer = l + 1;`) | `:508-557` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::MutexLock session_lock(mu_);` .. `++in_flight_;`) | `:514-521` |
| `send.cc` | `TransferSendSession::StartPush` (`if (!future.ok())`) | `:533-549` |
| `send.cc` | `TransferSendSession::StartPush` (`layer_future.OnReady`) | `:551-556` |
| `send.cc` | `TransferSendSession::SendNextLayer` | `:561-660` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`if (l >= base_->num_layers())` .. `++in_flight_;`) | `:564-576` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`layer_future->OnReady`) | `:579-659` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`if (!status_or.ok())`) | `:586-595` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`absl::MutexLock session_lock(self->mu_);` .. `std::move(end_op).Cancel();`) | `:597-605` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->push_pool()->Schedule`) | `:606-658` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`remote_data_endpoints = self->remote_data_endpoints_;` .. `++self->in_flight_;`) | `:617-620` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:625-653` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`if (!push_res.ok())`) | `:630-639` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`const bool last = self->remaining_h2h_layers_.fetch_sub(1) == 1;`) | `:648` |
| `base.cc` | `*` (`size_t ResolveEnvNumStagingLayers(size_t num_layers) {` .. `return requested;`) | `:420-429` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`slot = transfer_hooks_.lookup_send_staging(uuid, layer_idx);`) | `:3341` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.lookup_send_staging =`) | `:475-488` |
| `mgr.cc` | `KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer` (`for (const auto& [uuid, session] : send_sessions_)`) | `:568-572` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NotifyForRead` | `:654-701` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = send_sessions_.begin();`) | `:1159-1181` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1431-1557` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` (`absl::Cleanup counter_cleanup` .. `std::move(counter_cleanup).Cancel();`) | `:1516-1539` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandleAck` | `:1559-1562` |
| `mgr.cc` | `KVCacheManagerWithTransfer::AckSend` | `:1625-1648` |
| `bt.cc` | `BlockTransport::AsyncPush` | `:499-548` |
| `send_drain_test.cc` | `SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds` | `:509-533` |
| `send_drain_test.cc` | `SendDrainTest.SendNobodyPulledFailsAtItsDeadline` | `:555-564` |
| `send_drain_test.cc` | `SendDrainTest.FailedLayerWaitsForTheOtherLayersCopies` | `:535-553` |
| `send_drain_test.cc` | `SendLifecycleTest.FailedSendWithoutWorkSettlesImmediately` | `:620-630` |
| `send_drain_test.cc` | `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess` | `:652-664` |
| `send_drain_test.cc` | `SendLifecycleTest.SuccessCannotOverrideAnEarlierFailure` | `:666-678` |
| `send_drain_test.cc` | `SendLifecycleTest.DetailsSayAnUnpulledSendTimedOutAwaitingItsPull` | `:698-719` |
| `send_drain_test.cc` | `SendDrainTest.DetailsSayAnExpiredSendWasStillCopying` | `:721-738` |
| `send_drain_test.cc` | `SendDrainTest.DetailsCarryTheErrorOfAFailedCopy` | `:740-753` |
| `send_drain_test.cc` | `SendDrainTest.DetailsSayAFailedPushFailedInPushWhileCopiesRun` | `:755-782` |
| `send_drain_test.cc` | `SendLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` | `:1487-1585` |
| `send_drain_test.cc` | `SendLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` | `:1587-1735` |
| `pool_reshard_test.cc` | `SendDeadlineTest.ExpiredSendSessionFailsInsteadOfReportingDone` | `:369-378` |
| `control_test.cc` | `ControlHandshakeTest.PullAfterRegistrationDeadlineIsRejected` | `:311-333` |
| `control_test.cc` | `ControlHandshakeTest.DuplicatePullIsRejectedBeforeAcknowledgement` | `:370-382` |
| `transfer_send_session_test.cc` | `FakeSendBase::CompleteH2h` | `:97-104` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeDone` | `:219-242` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnections` | `:244-264` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush` | `:266-298` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush` | `:300-326` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CreateRejectsUnknownDataConnReuseValue` | `:328-343` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.SendSessionImplementsTransferSessionInterface` | `:345-375` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator` | `:410-453` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.FinishBeforeStartPushDoesNotAcquireStagingOrAccessHbm` | `:455-482` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.DoneGuaranteesAllResourcesReleasedAndNoHbmOrTransportAccessAfterDone` | `:484-541` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.ZeroLayerSessionCompletesAndReleasesStagingWithoutHang` | `:580-599` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone` | `:622-652` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.D2hDispatchFailureFailsUnissuedLayersAndDrainsIssuedCopies` | `:683-736` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenFailedDispatchEndsSend` | `:742-797` |
<!-- source-map:end -->
-/

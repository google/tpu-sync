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
`mgr.cc` is `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`. Exact line ranges for every symbol,
sub-block, and test are in the generated `## Source map` table below.

## How a send runs

`StartPush` runs once on a worker thread spawned by the
consumer's pull request (`HandlePullStream`'s `std::thread`). It acquires staging
(`AcquireStagingWithRetry`), then issues one D2H copy per layer in a loop
(`for (size_t l = 0; l < total_layers; ++l)`), each counted in `in_flight_` from
before its dispatch until its future's `layer_future.OnReady` callback ends the
op (`self->EndSendOp()`). After the loop it calls `SendNextLayer(0)`.

`SendNextLayer(l)` takes an op (`layer_future = d2h_layer_futures_[l]; ++in_flight_`)
and waits on layer `l`'s D2H future. When it is ready (`layer_future->OnReady`)
the layer is handed to the push pool (`self->base_->push_pool()->Schedule`),
whose task takes a second op (`++self->in_flight_`) for the H2H push,
issues it (`H2hWriteDirectAsync`), calls `SendNextLayer(l + 1)`, and only then
releases the first op via its `absl::Cleanup end_op`. The push's callback
releases the second op and, when the last layer's push completes
(`remaining_h2h_layers_.fetch_sub(1) == 1`), finishes the session OK
(`self->Finish()`).

So two chains run against the same `in_flight_`: the D2H copies, all issued up
front, and the pushes, strictly one layer after another. The session settles
when both have drained after a `Finish`.

## State

The settle protocol is `Transfer.Lifecycle`. On top of it:

| Field        | C++                                             | Role |
|--------------|-------------------------------------------------|------|
| `numLayers`  | `base_->num_layers()`                           | fixed |
| `pullStarted`| `pull_started_` (`TransferSendSession`)         | `ValidateAndBeginPull` claimed the offer |
| `started`    | `StartPush` passed its locked section (`host_block_ids.assign` .. `dst_ints_.assign`) | staging acquired, copies may start |
| `d2hPending` | ghost                                           | the D2H loop is between `++in_flight_` and `D2hSyncDispatch`'s outcome |
| `d2hIssued`  | `d2h_layer_futures_.size()` (`TransferSendSession`) until the settle clears it (`ReleaseStagingBlocksLocked`) | copies handed to the device |
| `d2hReady`   | futures with `IsReady()`                        | copies finished |
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
| `start`        | `StartPush` from the acquisition (`AcquireStagingWithRetry`) through its locked section, spawned after `ValidateAndBeginPull`: a zero-layer send (`total_layers == 0`) calls `FinishLocked(absl::OkStatus())` at once |
| `d2hBegin`     | one iteration of `StartPush`'s D2H loop up to `++in_flight_` (`if (draining_) return; ++in_flight_;`) |
| `d2hIssue ok`  | the dispatch (`D2hSyncDispatch`): failure finishes the session and ends the op (`FinishLocked(future.status()); EndSendOpLocked()`); success records the future (`d2h_layer_futures_.push_back(layer_future)`). For the last layer, `SendNextLayer(0)` (up to its lock) is attempted as part of the event |
| `d2hReady`     | the device finishes a copy: its future becomes `IsReady()` |
| `d2hEnd`       | the copy's `layer_future.OnReady` in `StartPush` ends its op via `EndSendOp` (see the lock-window note below) |
| `wake ok`      | `SendNextLayer`'s `layer_future->OnReady` callback: a failed copy finishes the session (`self->FinishLocked(status_or.status(), kD2hPhase)`); a draining session drops the layer (`if (self->draining_) return;`); otherwise the layer is scheduled on `push_pool()->Schedule` and `std::move(end_op).Cancel()` carries the op over |
| `h2hIssue`     | the `push_pool` task up to the push: a draining session (`if (self->draining_) return;`) drops the layer and ends the op via `absl::Cleanup end_op`; otherwise a second op is taken (`++self->in_flight_`) and the push issued (`H2hWriteDirectAsync` → `BlockTransport::AsyncPush`, which fans a layer out into one stream group per source NUMA node via `AsyncPushShards` and joins them via `tsl::JoinFutures` into this one callback; the callback may even run inline on an argument error) |
| `sendNext`     | the pool task's `SendNextLayer(l + 1)` followed by the release of the carried-over op (`absl::Cleanup end_op`) |
| `h2hDone ok`   | the `H2hWriteDirectAsync` push callback: failure finishes the session (`self->FinishLocked(push_res.status(), kPushPhase)`); the last OK push (`remaining_h2h_layers_.fetch_sub(1) == 1`) finishes it OK (`self->Finish()`); the op ends either way |
| `cancel`       | any `Finish(error)` from outside the data path: deadline (`CompleteReadWithDetails`), shutdown (`~KVCacheManagerWithTransfer`), pull-worker spawn failure (`HandlePullStream`'s `counter_cleanup`), staging acquisition failure (`!acquire_status.ok()`), bad arguments in `StartPush` (`src_block_ids.empty() || ...`), pull-worker exceptions (`HandlePullStream`'s `catch`) |
| `publish`      | `CompleteReadWithDetails` / `CompleteReadRaw` moves a settled session (`session->Done()`) into `done_sending_` or `failed_recving_` by its status and erases it from `send_sessions_` |

## Lock scopes

Every mutable field of `TransferSendSession` is `ABSL_GUARDED_BY(mu_)` except
`remaining_h2h_layers_`, an atomic read only by its own `fetch_sub`; `base_`,
`staging_allocator_`, `req_id_`, `uuid_`, `deadline_`, `register_start_` and
`data_connection_reuse_` are `const`. Every `EndSendOp` below is one `mu_` scope
under the default `pooled` connection policy and the A7 window under
`per_transfer`.

| Event          | `mu_` scopes | Between them |
|----------------|--------------|--------------|
| `beginPull`    | 1: `ValidateAndBeginPull` | — (`deadline_` is `const`) |
| `start`        | 1 deciding scope: `StartPush`'s locked section (`if (draining_ || done_) return;` through `dst_ints_.assign`), preceded by `remote_data_endpoints_ =` and the scopes of `AcquireStagingWithRetry` (which touch only `remote_data_endpoints_` and `staging_`) | a `Finish` during `AcquireWithTimeout` hits `if (draining_ || done_)` in `AcquireStagingWithRetry` and drops the local `StagingAllocation` before `staging_` is assigned; a `Finish` between `staging_ = *std::move(acquired)` and `StartPush`'s locked section settles the idle session and releases `staging_` (`FinishLocked` runs `ReleaseStagingBlocksLocked()` at `in_flight_ == 0`), after which the locked section returns — both match `cancel` before `start`. A failed acquisition or bad arguments call `Finish(error)` (`cancel`) |
| `d2hBegin`     | 1: the loop's `session_lock` (`if (draining_) return; ++in_flight_;`) | — |
| `d2hIssue`     | on success, 1 scope (`d2h_layer_futures_.push_back(layer_future)`) after the unlocked `D2hSyncDispatch`, then `layer_future.OnReady` outside `mu_` and, for the last layer, `SendNextLayer(0)`'s scope (A6); on failure, 1 scope (`FinishLocked(future.status()); EndSendOpLocked();`) | a `Finish` during `D2hSyncDispatch` finds the op `d2hBegin` took and cannot settle; the future is pushed regardless of `draining_` (matching `d2hIssue`), and the copy drains through `d2hEnd` (`SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds`). If `layer_future` is already ready when `OnReady` is registered, the callback runs inline (`d2hReady` then `d2hEnd`) |
| `d2hReady`     | 0 (device) | — |
| `d2hEnd`       | 1: `EndSendOp` | — |
| `wake`         | on `ok = false` or `draining_`: 2 (`FinishLocked` or the `if (self->draining_)` check, then `EndSendOp` from `absl::Cleanup end_op`); otherwise 1 (`if (self->draining_)` check, then `std::move(end_op).Cancel()` carries the op into the unlocked `push_pool()->Schedule`) | a `Finish` after the check and before the pool task runs is caught by the task's own `if (self->draining_)` check (`wake true`, `cancel`, `h2hIssue` ending the op) |
| `h2hIssue`     | when draining: 2 (the task's `if (self->draining_)` check, then `EndSendOp` from `end_op`); otherwise 1 (the check through `++self->in_flight_`, followed by the unlocked `H2hWriteDirectAsync`) | a `Finish` between `++self->in_flight_` and `H2hWriteDirectAsync` does not stop the push (matching `h2hIssue`), and the push callback later retires the op (`Finish()` on the last OK push is a no-op once draining). Splitting the pool task into `h2hIssue` and `sendNext` via `chaining` also admits a push callback that runs before `SendNextLayer(l + 1)`, including inline (`h2hIssue`, `h2hDone`, `sendNext`) |
| `sendNext`     | for `l + 1 < num_layers`: 2 (`SendNextLayer(l + 1)`'s scope from `if (draining_ || l >= d2h_layer_futures_.size()) return;` to `++in_flight_;`, then `EndSendOp` from the pool task's `end_op`); for the last layer, `SendNextLayer` returns before its lock (`if (l >= base_->num_layers()) return;`), leaving only `EndSendOp` | an `EndSendOp` tail, below |
| `h2hDone`      | on `ok = false`: 2 (`FinishLocked(push_res.status(), kPushPhase)`, then `EndSendOp` from `end_op`); on `ok = true`: lock-free `remaining_h2h_layers_.fetch_sub(1)`, then `Finish()` (1 scope) if it was the last layer, then `EndSendOp` | nothing else reads `remaining_h2h_layers_`, so any step between `fetch_sub` and `Finish()` orders before `h2hDone` (a `cancel` in that gap wins, matching `cancel` then `h2hDone true` because `finishOnceLocked` ignores the later OK); after `Finish()`, an `EndSendOp` tail |
| `cancel`       | 1: `Finish(error)` (for the deadline sweep, `!IsDraining()` is checked in an earlier scope; `deadline()` is `const`) | `TransferSendSession::FinishLocked` ignores every call after the first (`if (draining_ || done_) return;`), so a stale sweep is a no-op (`trace_cancel_after_ok_finish`, `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess`) |
| `publish`      | under manager `mu_`: `Done()`, `GetStatus()`, `Details()` | nothing changes on a settled sender (`TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone`), so no late error can split the published set from `transfer_details_` |

**`EndSendOp` tails.** `wake`, `h2hIssue`, `sendNext` and `h2hDone` all run
`EndSendOp` from an `absl::Cleanup` after releasing the deciding scope's lock.
Between the two scopes, `in_flight_` still counts the op. Because only the
settle checks in `FinishLocked` and `EndSendOpLocked` (and A7's `release_due`)
read `in_flight_`, any step in the gap makes the same decision as after
`EndSendOp`, except that a `Finish` in the gap leaves the final settle to
`EndSendOp` with the same resulting `draining_`, `done_` and `status_`. Every
such interleaving therefore reaches the same state as running the whole event
first, keeping the ghost counters (`d2hRetired`, `woken`, `pooled`, `chaining`,
`h2hRetired`) sound.

**Unchecked dispatches.** Neither `d2hIssue` (when pushing the future) nor
`h2hIssue` (when calling `H2hWriteDirectAsync`) re-reads `draining_`; the model
matches this directly, so a `cancel` between `d2hBegin` and `d2hIssue`, or
between `h2hIssue`'s lock and the transport call, is admitted in the same order.
The only multi-scope fold across distinct operations is A6 (`SendNextLayer(0)`
folded into the last `d2hIssue`).

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
  (`0, 1, …`), so when `SendNextLayer(k)`'s callback wakes in C++, layers
  `0 … k-1` have already woken and layer `k`'s future is ready — hence at
  least `k + 1` futures are ready (`woken < d2hReady`), even if later layers'
  D2H copies finished earlier. Since no event's effect on the counters depends
  on which layer finished, any real interleaving maps onto one the model
  admits.
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
* **A6 (fold of `SendNextLayer(0)`).** Between recording the last future
  (`d2h_layer_futures_.push_back(layer_future)`) and `SendNextLayer(0)`'s lock,
  only a `Finish` can change the outcome of the attempt, and it has the same
  effect ordered between `d2hBegin` and `d2hIssue true` (`d2hIssue true` still
  records `d2hIssued = numLayers` while `trySendNext` returns `none` on
  `draining`); the two are one event.

* **A7 (connection release window).** `EndSendOp`
  is modelled as the atomic `endOp`. Under `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer`
  (default `pooled`, which never enters this branch) the last op of a draining
  send first drops `mu_` to close the idle data connections to the consumer
  (`ReleaseDataConnectionsAfterSend`; the branch `release_due` also needs
  `!remote_data_endpoints_.empty()` and `!connections_released_`) and only then
  runs `EndSendOpLocked`. In that window `in_flight_ == 1`, `draining_` and
  `!done_` hold: no new op can begin (`beginOp` refuses while `draining_` in
  `StartPush` and `SendNextLayer`), no other op can end (this is the only one,
  A3), `FinishLocked` returns at `if (draining_ || done_) return;`,
  `ValidateAndBeginPull` throws on `if (pull_started_)`, and the poll cannot
  publish (`Done()` is false), so no modelled variable changes between the
  check and the decrement; `connections_released_` only stops a second pass.
  The direct `EndSendOpLocked()` on a copy that could not be issued in
  `StartPush` bypasses the window, but no push exists then. The window delays
  `done_`, which every trace already allows between events.

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
  `woken ≤ queued ≤ d2hIssued`, `woken ≤ d2hReady`,
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
there is no `failed_sending_`. (3) The D2H loop's `if (draining_) return;` in
`StartPush` also skips `SendNextLayer(0)`, which is fine because every later
`SendNextLayer` re-checks `draining_`.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `send.h` | `TransferSendSession::Done` | `:69-72` |
| `send.h` | `TransferSendSession::GetStatus` | `:76-79` |
| `send.h` | `TransferSendSession::IsDraining` | `:87-90` |
| `send.h` | `TransferSendSession::deadline` | `:133` |
| `send.h` | `TransferSendSession` (`mutable absl::Mutex mu_;` .. `const DataConnectionReuse data_connection_reuse_;`) | `:191-199` |
| `send.h` | `TransferSendSession` (`bool pull_started_` .. `std::atomic<size_t> remaining_h2h_layers_`) | `:211-227` |
| `send.cc` | `*` (`constexpr absl::string_view kAwaitPullPhase` .. `constexpr absl::string_view kPushPhase`) | `:55-58` |
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:142-156` |
| `send.cc` | `TransferSendSession::FinishLocked` | `:193-206` |
| `send.cc` | `TransferSendSession::Finish` | `:208-217` |
| `send.cc` | `TransferSendSession::Details` | `:225-241` |
| `send.cc` | `TransferSendSession::CurrentPhaseLocked` | `:243-251` |
| `send.cc` | `TransferSendSession::EndSendOpLocked` | `:253-260` |
| `send.cc` | `TransferSendSession::EndSendOp` | `:262-289` |
| `send.cc` | `TransferSendSession::AcquireStagingWithRetry` | `:322-356` |
| `send.cc` | `TransferSendSession::StartPush` | `:358-457` |
| `send.cc` | `TransferSendSession::StartPush` (`if (src_block_ids.empty()`) | `:369-374` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Status acquire_status =` .. `dst_ints_.assign`) | `:380-403` |
| `send.cc` | `TransferSendSession::StartPush` (`for (size_t l = 0; l < total_layers; ++l)`) | `:417-454` |
| `send.cc` | `TransferSendSession::StartPush` (`layer_future.OnReady`) | `:450-453` |
| `send.cc` | `TransferSendSession::StartPush` (`remote_data_endpoints_ = remote_data_endpoints;`) | `:377` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::MutexLock session_lock(mu_);` .. `++in_flight_;`) | `:419-423` |
| `send.cc` | `TransferSendSession::StartPush` (`if (!future.ok())`) | `:435-444` |
| `send.cc` | `TransferSendSession::StartPush` (`d2h_layer_futures_.push_back(layer_future);`) | `:448` |
| `send.cc` | `TransferSendSession::SendNextLayer` | `:459-547` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`if (l >= base_->num_layers())` .. `++in_flight_;`) | `:462-474` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`layer_future->OnReady`) | `:477-546` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->push_pool()->Schedule`) | `:500-545` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:518-540` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`if (!status_or.ok())`) | `:484-490` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`absl::MutexLock session_lock(self->mu_);` .. `std::move(end_op).Cancel();`) | `:492-499` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`remote_data_endpoints = self->remote_data_endpoints_;` .. `++self->in_flight_;`) | `:510-513` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`if (!push_res.ok())`) | `:523-529` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`const bool last = self->remaining_h2h_layers_.fetch_sub(1) == 1;`) | `:535` |
| `send.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:183-186` |
| `mgr.cc` | `KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer` (`for (const auto& [uuid, session] : send_sessions_)`) | `:400-404` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NotifyForRead` | `:486-533` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = send_sessions_.begin();`) | `:988-1010` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1260-1386` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` (`absl::Cleanup counter_cleanup` .. `std::move(counter_cleanup).Cancel();`) | `:1345-1368` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandleAck` | `:1388-1391` |
| `mgr.cc` | `KVCacheManagerWithTransfer::AckSend` | `:1454-1477` |
| `bt.cc` | `BlockTransport::AsyncPush` | `:1052-1084` |
| `send_drain_test.cc` | `SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds` | `:324-348` |
| `send_drain_test.cc` | `SendDrainTest.SendNobodyPulledFailsAtItsDeadline` | `:370-379` |
| `send_drain_test.cc` | `SendDrainTest.FailedLayerWaitsForTheOtherLayersCopies` | `:350-368` |
| `send_drain_test.cc` | `SendLifecycleTest.FailedSendWithoutWorkSettlesImmediately` | `:435-445` |
| `send_drain_test.cc` | `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess` | `:467-479` |
| `send_drain_test.cc` | `SendLifecycleTest.SuccessCannotOverrideAnEarlierFailure` | `:481-493` |
| `send_drain_test.cc` | `SendLifecycleTest.DetailsSayAnUnpulledSendTimedOutAwaitingItsPull` | `:513-534` |
| `send_drain_test.cc` | `SendDrainTest.DetailsSayAnExpiredSendWasStillCopying` | `:536-553` |
| `send_drain_test.cc` | `SendDrainTest.DetailsCarryTheErrorOfAFailedCopy` | `:555-568` |
| `send_drain_test.cc` | `SendDrainTest.DetailsSayAFailedPushFailedInPushWhileCopiesRun` | `:570-597` |
| `pool_reshard_test.cc` | `SendDeadlineTest.ExpiredSendSessionFailsInsteadOfReportingDone` | `:369-378` |
| `control_test.cc` | `ControlHandshakeTest.PullAfterRegistrationDeadlineIsRejected` | `:311-333` |
| `control_test.cc` | `ControlHandshakeTest.DuplicatePullIsRejectedBeforeAcknowledgement` | `:370-382` |
| `transfer_send_session_test.cc` | `FakeSendBase::CompleteH2h` | `:82-89` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeDone` | `:189-212` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnections` | `:214-234` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush` | `:236-268` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush` | `:270-296` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CreateRejectsUnknownDataConnReuseValue` | `:298-313` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.SendSessionImplementsTransferSessionInterface` | `:315-345` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator` | `:380-423` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.FinishBeforeStartPushDoesNotAcquireStagingOrAccessHbm` | `:425-452` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.DoneGuaranteesAllResourcesReleasedAndNoHbmOrTransportAccessAfterDone` | `:454-511` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.ZeroLayerSessionCompletesAndReleasesStagingWithoutHang` | `:550-569` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone` | `:592-622` |
<!-- source-map:end -->
-/

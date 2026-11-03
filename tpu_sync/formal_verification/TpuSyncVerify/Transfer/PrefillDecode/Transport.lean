import TpuSyncVerify.Transfer.PrefillDecode.Transport.Model
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Lemmas
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Traces

/-!
# Receiver-side stream accounting in `BlockTransport`

Models the per-uuid layer progress table the transport keeps for incoming
pushes (`BlockTransportServerPushSession::layer_progress_`) and the two
receiver counters of `TransferReceiveSession` it feeds, to discharge the
assumption `Receive.lean` calls **A1** ("one dispatch per layer") and the
coarsening behind **A4** ("a layer's blocks are accounted once its copy is
issued"), and to state exactly what they rest on.

Abbreviations: `bt.cc` = `tpu_sync/transport/block_transport.cc`, `bt.h` =
`tpu_sync/transport/block_transport.h`, `bts.cc` =
`tpu_sync/transport/block_transport_session.cc`, `bts.h` =
`tpu_sync/transport/block_transport_session.h`, `recv.cc` =
`tpu_sync/kv_cache/transfer_receive_session.cc`, `mgr.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`, `send.cc` =
`tpu_sync/kv_cache/transfer_send_session.cc`, `base.cc` =
`tpu_sync/kv_cache/kv_cache_manager_base.cc`. Exact line ranges for every
symbol and sub-block are in the generated `## Source map` table below.

## What is modelled

One receive (`uuid`) with `numLayers` block arrays, each pushed by
`numSenders` senders that each declare `numStreams` streams per layer
(`header.reserved` passed as `num_streams` to `CompleteIncomingPush`) carrying
`blocksPerStream` blocks. `BlockTransport` keeps one
`BlockTransportServerPushSession` per uuid in `active_push_sessions_`
(`BlockTransport::GetOrCreatePushSession`, under `server_sessions_mu_`); the
session owns the uuid's progress table `layer_progress_` (keyed by layer,
guarded by the session's own `mu_`), so the one-uuid view taken here is the
structure of the C++, not an abstraction of a shared table. A stream is
`BlockTransport::HandleIncomingPush` → `GetOrCreatePushSession(header.uuid)` →
`BlockTransportServerPushSession::StartStream` from the admission gate to the
final ack, collapsed into one event: the progress-table update is one
critical section of the push session's `mu_` in
`BlockTransportServerPushSession::CompleteIncomingPush`
(`auto& progress = layer_progress_[l]`) and the two delegate callbacks that
follow run on the same reader thread, so interleaving other streams between
them changes nothing the properties here observe. The sender always names one
layer per stream (`TransferSendSession::SendNextLayer`'s `H2hWriteDirectAsync`
→ `BlockTransport::BuildBlockRequests`'s `header_layer_idx`), so the
all-layers form `local_id == 0xFFFFFFFF` in `ResolveIncomingPushSpec` and
`CompleteIncomingPush` does not arise and `l` ranges over `0 .. numLayers - 1`.

Two of the four completion contracts of
`BlockTransportServerPushSession::CompleteIncomingPush` are modelled, selected
by `Cfg.declared`:

* **chunk-count** (`expected_senders = nullopt`,
  `array_complete = progress.completed_chunks == expected_chunks`): the layer
  completes when `completed_chunks == num_streams`. This is the contract of
  the prefill-to-decode path whenever no receiver plan with
  `shard_push_schedules` is registered (`KVCacheManagerBase::ExpectedPushSenders`)
  — in particular always on the `StartRead` pull path, which registers no plan
  (`KVCacheManagerWithTransfer::StartRead(remote_endpoint)`).
* **sender-declared** (`expected_senders = some n`,
  `else if (expected_senders.has_value())`): per sender `landed`/`declared`,
  `senders_complete`, and three rejections (`AlreadyExists` when a sender lands
  more than it declared or more senders than expected appear,
  `FailedPrecondition` when a sender re-declares).

The plan-declared contract (`if (plan_declared)` in `CompleteIncomingPush`, pool
resharding) is out of scope; see `Reshard/`. So is the Peregrine contract
(`else if (is_peregrine)`): it is reached only from
`BlockTransportServerPushSession::OnPeregrinePostComplete`, the receiver side of
the Peregrine data plane, whose only entry point `BlockTransport::OnRecvComplete`
nothing outside the transport's own tests calls yet (the sender side,
`BlockTransport::AsyncPush` routing a push to the Peregrine adapter when
`config_.backend_for(op) == TransportBackend::kPeregrine`, is likewise outside
the model; the socket path `StartStream` is what every production push takes
and it leaves `is_peregrine` at its default). The delegate hooks that branch
adds (`OnRecvProgress`, and `OnRecvNetworkComplete` once the session's
`layers_dispatched_` reaches `num_layers()`) therefore never fire on the
modelled path; on the socket path `network_completed_` is set only by
`RecordBlockShardsReceivedLocked`.

The session side is `Transfer/Session.lean`'s `Lifecycle`, transcribed as
`Receive.lean` does: the push lease is `beginOp` (`TryBeginRecvOp` via
`hooks.begin_incoming_push`) and is returned inside the same event (`EndRecvOp`
via `hooks.end_incoming_push` / `StartStream`); the H2D callback and the
poll finish through `finishLocked`/`endOpLocked`
(`TransferReceiveSession::FinishLocked`, `EndRecvOpLocked`). `inFlight` counts
only H2D copies whose callback has not run.

## State

| Field | C++ | Role |
|---|---|---|
| `Layer.entry` | `layer_progress_[l]` of the uuid's push session (`BlockTransportServerPushSession::LayerProgress`, `layer_progress_`) | `none` = no entry (never created, or cleared by retirement) |
| `Progress.completedChunks` | `LayerProgress::completed_chunks` | chunk-count contract |
| `Progress.landed` | `SenderStreams::landed` per sender | sender-declared contract |
| `Progress.sendersComplete` | `LayerProgress::senders_complete` | sender-declared contract |
| `Progress.called` | `LayerProgress::on_layer_received_called` | the once-latch |
| `Layer.sent` | ghost | streams accepted per sender for this layer |
| `Layer.dispatched` | ghost | `OnLayerReceived` → `ExecuteLayerH2d` entries for this layer |
| `Layer.pending` | subset of `h2d_futures_` (`ExecuteLayerH2d`: `h2d_futures_.push_back(future)`) | copies issued for this layer whose callback has not run |
| `blocks` | `blocks_received_per_shard_` collapsed to one shard (`RecordBlockShardsReceivedLocked`) | block units reported by `OnBlockShardsReceived` |
| `completedLayers` | `num_completed_layers_` (`ExecuteLayerH2d`: `self->num_completed_layers_++`) | H2D callbacks that ran OK |
| `life` | `in_flight_`, `draining_`, `done_`, `status_`, staging (`Lifecycle`) | settle protocol |

The shard dimension is collapsed: every shard shares the one target
`total_blocks_ * num_layers` (`RecordBlockShardsReceivedLocked`:
`const int64_t per_shard_target = total_blocks_ * num_layers`) and a stream adds
its block count to every shard it carries, so one counter against
`numSenders * numStreams * blocksPerStream * numLayers` has the same
completion behaviour (A4 in `Receive.lean` restates this over shards).

## Events

| Event | C++ |
|---|---|
| `stream l s` | `HandleIncomingPush` → `GetOrCreatePushSession` → `StartStream`: `BeginIncomingPush` → `TryBeginRecvOp` (`hooks.begin_incoming_push`, refuses once `done_`/`draining_`), payload, `CompleteIncomingPush`, on completion `OnLayerReceived` → `ExecuteLayerH2d` (`hooks.on_layer_received`, `++in_flight_`), then `OnBlockShardsReceived`, `EndIncomingPush`. A rejection inside `CompleteIncomingPush` returns early and `absl::Cleanup end_incoming_push` ends the push with `InternalError("Incoming push failed")`, which `hooks.end_incoming_push` turns into `Finish(status)` then `EndRecvOp()`. |
| `h2dDone l ok` | the `future.OnReady` callback of a layer's copy in `ExecuteLayerH2d`: on OK `num_completed_layers_++` and, at `num_layers()` while not draining, the success `FinishLocked()`; on error `FinishLocked(status_or.status(), kH2dPhase)`; then `absl::Cleanup end_op`'s `EndRecvOp()` |
| `publish` | `CompleteReadWithDetails`: a receive that is not draining and `IsReadyToComplete()` (`network_completed_ || num_completed_layers_ == num_layers`, and every issued future ready) is `Finish()`ed. "Every future ready" is taken as "every callback has run"; the window between the two is `ReceivePoll.lean`'s subject (`pollReady`) and changes no property here. |
| `cancel` | the deadline sweep in `CompleteReadWithDetails` (`else if (session->deadline() <= now)`) or any other `Finish(error)` on a receive that is not yet draining |

`published` (`life.done && life.statusOk`) is what the poll reports in
`done_recving_` (`CompleteReadWithDetails`).

## Lock scopes

Three mutexes meet on the push thread: `BlockTransport::server_sessions_mu_`
(the uuid → push-session map `active_push_sessions_`), the push session's
`BlockTransportServerPushSession::mu_` (its `layer_progress_`, `done_`,
`layers_dispatched_`), and the receive session's `TransferReceiveSession::mu_`.
`stream` is the one deliberately coarse event in the prefill-to-decode models:
it spans all three. In order on the push thread: `GetOrCreatePushSession`
(`server_sessions_mu_`, creating the session on the uuid's first stream),
`TryBeginRecvOp` (receive-session `mu_`), the payload write (unlocked; under
bounded host layer staging each chunk's `GetBlockChunks` also runs
`TransferReceiveSession::AcquireStagingSlot`, audited in `Receive.lean`), the
`expected_senders` lookup (push-session `mu_`), the unlocked
`ExpectedPushSenders` check (A2), the `layer_progress_` update with the
once-latch and retirement (push-session `mu_`, A1), then `OnLayerReceived` →
`ExecuteLayerH2d` (receive-session `mu_` scopes, audited in `Receive.lean`),
on retirement the session's `on_completed_` callback (`server_sessions_mu_`,
erasing the uuid from `active_push_sessions_`), `OnBlockShardsReceived`
(receive-session `mu_`, plus `MaybeUnregisterSettledRecv`), and
`EndIncomingPush` → `EndRecvOp` (receive-session `mu_`, plus
`MaybeUnregisterSettledRecv` and `GetStatus()`). If `CompleteIncomingPush`
rejects the push, the `absl::Cleanup end_incoming_push` runs
`EndIncomingPush(InternalError(...))`, which calls `DeferUnregisterOnSettle()`
and `Finish(status)` before `EndRecvOp()`. `h2dDone`, `publish` and `cancel`
are `Receive.lean`'s `h2dDone`, `pollReady` and `cancel`.

| Event | Scopes | Between them |
|---|---|---|
| `stream l s` | `server_sessions_mu_`, push-session `mu_` and receive-session `mu_` scopes above | concurrent streams of one uuid share one push session (`GetOrCreatePushSession` returns the existing entry) and serialize their `layer_progress_` updates on its `mu_` in the order of the model's `stream` events; streams of other uuids touch other sessions. Their receive-session calls are covered by `Receive.lean`. If `Finish(error)` lands after `TryBeginRecvOp` and before the delegate calls finish, `ExecuteLayerH2d` and `OnBlockShardsReceived` return at `if (done_ || draining_)` and `EndRecvOp` releases the lease while `layer_progress_` keeps its latch; `Safe` still holds (`OnceDispatched` by the latch, `NetworkSound` vacuously because `blocks < c.target`, and `published = false`). Likewise, if an OK `Finish()` lands before `OnBlockShardsReceived`, every layer has already been dispatched and completed its H2D copy (`CompleteOnSuccess`), and `blocks < c.target` keeps `NetworkSound` vacuous |
| `h2dDone l ok` | 2 (`Receive.lean`) | an `EndRecvOp` tail (`Receive.lean`) |
| `publish` | `IsDraining()`, `IsReadyToComplete()`, `Finish()` (`Receive.lean`, `pollReady`) | monotone readiness; an OK `Finish()` on a draining session is a no-op |
| `cancel` | 1: `Finish(error)` | guarded here by `!draining`; a stale sweep is a no-op |

Retirement (`if (all_layers_called)` in `CompleteIncomingPush`) runs
`layer_progress_.clear()` and sets the push session's `done_` in the same
critical section as the latch; the session is dropped from
`active_push_sessions_` by its `on_completed_` callback after the delegate
callbacks have returned. `StartStream` and `CompleteIncomingPush` never test
`done_`, so a stream that reaches the session in between, or a later stream
that creates a fresh session, starts a fresh entry: `Layer.entry = none` covers
both. `BlockTransport::ForgetPushProgress` removes the uuid's session from
`active_push_sessions_` (under `server_sessions_mu_`) and `Cancel()`s it, from
`KVCacheManagerBase::UnregisterActivePlanDirect`, reached through
`UnregisterSettledPlan` (the receive draining with its last op, or settled) and
`UnregisterActivePlan` (only once `Done()`); no stream passes `TryBeginRecvOp`
in either state, and a stream already past the gate that recreates an entry
finds the receive session draining. It is not an event.

## Assumptions

* **A1 (one lock).** All progress-table mutation is under the push session's
  `mu_` in `CompleteIncomingPush`, and a session holds exactly one uuid's
  table; the latch is set in the same critical section that decides completion
  (`if (array_complete && !progress.on_layer_received_called)`).
  Encoded by making the table update one step.
* **A2 (stable expectation).** `ExpectedPushSenders(uuid)` is read outside
  the lock and latched on the layer's first stream (`if (!progress.expected_senders_resolved)`),
  so the contract a layer runs under is whatever the first stream saw. Encoded as
  the fixed `Cfg.declared`. On the `StartRead` path there is never a plan;
  on the plan path (`RegisterActivePlan`) the session is seated (`EmplaceRecvSessionLocked`)
  *before* the plan is published (`base_->RegisterActivePlan`), so a push landing in that window would
  latch chunk-count on a layer of a sender-declared plan. That the sender
  does not start before the receiver's registration has returned is a
  controller-side ordering outside this repository's code; it is assumed,
  not proved. `trace_chunk_two_senders` is what the window would produce
  on a multi-sender plan.
* **A3 (uniform declaration, one node per schedule).** Every sender declares
  the same `numStreams`, and no sender outside the expected set appears; the
  `FailedPrecondition` and "more senders" rejections in `CompleteIncomingPush`
  (`else if (streams.declared != num_streams)` and
  `if (progress.sender_streams.size() > *expected_senders)`) are
  therefore never taken and are not modelled. Senders are told apart by
  `header.remote_id = node_id()` (`BuildBlockRequests`:
  `remote_id = static_cast<uint32_t>(block_delegate_->node_id())`) while
  `expected_senders` counts `shard_push_schedules`, keyed by source shard
  partition (`StartTransferRequest.shard_push_schedules`); the contract
  therefore presumes one source shard per node, which the receiver never checks.
* **A4 (honest senders) — `Cfg.honest`.** Every sender sends at most the
  streams it declared for each layer, for the life of the receive. Encoded as
  a guard on `stream`: a stream beyond the declaration is not enabled. This
  is the assumption the C++ does **not** enforce on the chunk-count contract,
  nor on any contract after the table entries are retired
  (`if (all_layers_called)` in `CompleteIncomingPush`); the traces at the end drop it.

## Properties

* **`OnceDispatched`** — `ExecuteLayerH2d` runs at most once per layer. This
  is `Receive.lean` A1 and what its guard `issued + pending < numLayers`
  encodes.
* **`NetworkSound`** — when the block count reaches its target, every layer's
  copy has been issued. This is the coarsening behind `Receive.lean` A4
  (`netAccount` guarded by `layersAccounted < issued`).
* **`CompleteOnSuccess`** — a receive published as successful issued every
  layer's copy exactly once and no copy is still in flight. This is what the
  engine reads out of `done_recving`.

`reachable_safe`: under A4 both contracts satisfy all three, for any
positive `numLayers`, `numSenders`, `numStreams`, `blocksPerStream`.

Without A4 (`Cfg.honest = false`) the traces at the end show what one extra
stream does: on the chunk-count contract before retirement the receive is
published with a layer never copied; on either contract after retirement the
layer is copied twice (the session's `in_flight_` still holds `done_` off
until both copies return, and a failing duplicate copy fails the receive).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `bts.h` | `BlockTransportServerPushSession` (`struct LayerProgress` .. `size_t layers_dispatched_ ABSL_GUARDED_BY(mu_) = 0;`) | `:146-172` |
| `bt.h` | `BlockTransport` (`absl::Mutex server_sessions_mu_;` .. `active_push_sessions_ ABSL_GUARDED_BY(server_sessions_mu_);`) | `:280-283` |
| `bts.cc` | `ResolveIncomingPushSpec` | `:228-283` |
| `bt.cc` | `BlockTransport::GetOrCreatePushSession` | `:443-456` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:458-463` |
| `bt.cc` | `BlockTransport::OnRecvComplete` | `:468-474` |
| `bt.cc` | `BlockTransport::AsyncPush` (`if (config_.backend_for(op) == TransportBackend::kPeregrine)`) | `:509-522` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:321-455` |
| `bts.cc` | `BlockTransportServerPushSession::OnPeregrinePostComplete` | `:460-495` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:506-702` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`absl::Cleanup end_incoming_push`) | `:334-341` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`std::optional<size_t> expected_senders;` .. `expected_senders = delegate_->ExpectedPushSenders(uuid_);`) | `:523-535` |
| `bts.cc` | `BlockTransportServerPushSession::Cancel` | `:309-319` |
| `bt.cc` | `BlockTransport::ForgetPushProgress` | `:1009-1023` |
| `base.cc` | `KVCacheManagerBase::UnregisterActivePlanDirect` (`transport->ForgetPushProgress(uuid);`) | `:3245` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`bool array_complete = false;` .. `array_complete = progress.completed_chunks == expected_chunks;`) | `:556-632` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`bool all_layers_called = true;` .. `completion_cb = std::move(on_completed_);`) | `:651-668` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`const uint32_t remote_id =` .. `static_cast<uint32_t>(header_layer_idx);`) | `:729-735` |
| `recv.h` | `TransferReceiveSession::Done` | `:90-93` |
| `recv.h` | `TransferReceiveSession::IsDraining` | `:107-110` |
| `recv.h` | `TransferReceiveSession::TryBeginRecvOp` | `:121-126` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:449-464` |
| `recv.cc` | `TransferReceiveSession::Finish` | `:466-469` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:500-511` |
| `recv.cc` | `TransferReceiveSession::EndRecvOp` | `:513-516` |
| `recv.cc` | `TransferReceiveSession::IsReadyToComplete` | `:544-550` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShardsReceivedLocked` | `:559-586` |
| `recv.cc` | `TransferReceiveSession::OnRecvProgress` | `:685-697` |
| `recv.cc` | `TransferReceiveSession::OnRecvNetworkComplete` | `:699-709` |
| `recv.cc` | `TransferReceiveSession::OnBlockShardsReceived` | `:711-758` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:786-925` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`future.OnReady`) | `:862-922` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.on_layer_received =`) | `:303-319` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.begin_incoming_push =`) | `:364-389` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.end_incoming_push =`) | `:390-436` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterActivePlan` | `:721-809` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:912-949` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterSettledPlan` | `:951-967` |
| `mgr.cc` | `KVCacheManagerWithTransfer::MaybeUnregisterSettledRecv` | `:969-975` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_descriptors)` | `:1019-1043` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:1045-1149` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_recv_sessions_.begin();`) | `:1207-1238` |
| `mgr.cc` | `KVCacheManagerWithTransfer::OnBlockShardsReceived` | `:1668-1692` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:625-653` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:2745-2762` |
| `tpu_sync/rpc/raiden_service.proto` | `StartTransferRequest` (`map<int32, ShardPushScheduleProto> shard_push_schedules`) | `:180` |
<!-- source-map:end -->
-/

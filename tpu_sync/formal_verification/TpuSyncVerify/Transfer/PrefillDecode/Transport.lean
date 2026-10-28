import TpuSyncVerify.Transfer.PrefillDecode.Transport.Model
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Lemmas
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Traces

/-!
# Receiver-side stream accounting in `BlockTransport`

Models the `{uuid, layer}` progress table of `BlockTransport` and the two
receiver counters of `TransferReceiveSession` it feeds, to discharge the
assumption `Receive.lean` calls **A1** ("one dispatch per layer") and the
coarsening behind **A4** ("a layer's blocks are accounted once its copy is
issued"), and to state exactly what they rest on.

Abbreviations: `bt.cc` = `tpu_sync/transport/block_transport.cc`, `bt.h` =
`tpu_sync/transport/block_transport.h`, `recv.cc` =
`tpu_sync/kv_cache/transfer_receive_session.cc`, `mgr.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`, `send.cc` =
`tpu_sync/kv_cache/transfer_send_session.cc`, `base.cc` =
`tpu_sync/kv_cache/kv_cache_manager_base.cc`. Exact line ranges for every
symbol and sub-block are in the generated `## Source map` table below.

## What is modelled

One receive (`uuid`) with `numLayers` block arrays, each pushed by
`numSenders` senders that each declare `numStreams` streams per layer
(`header.reserved` passed as `num_streams` to `CompleteIncomingPush`) carrying
`blocksPerStream` blocks. A stream is `BlockTransport::HandleIncomingPush` from
the admission gate to the final ack, collapsed into one event: the
progress-table update is one `progress_mu_` critical section in
`CompleteIncomingPush` (`auto& progress = layer_progress_[{uuid, l}]`) and the
two delegate callbacks that follow run on the same reader thread, so
interleaving other streams between them changes nothing the properties here
observe. The sender always names one layer per stream
(`TransferSendSession::SendNextLayer`'s `H2hWriteDirectAsync` →
`BlockTransport::BuildBlockRequests`'s `header_layer_idx`), so the all-layers
form `local_id == 0xFFFFFFFF` in `ResolveIncomingPushSpec` and
`CompleteIncomingPush` does not arise and `l` ranges over `0 .. numLayers - 1`.

Two of the four completion contracts of `BlockTransport::CompleteIncomingPush`
are modelled, selected by `Cfg.declared`:

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
(`else if (is_peregrine)`): it is reached only from `BlockTransport::OnRecvComplete`,
which nothing outside the transport's own tests calls yet, while the socket path
`BlockTransport::HandleIncomingPush` leaves `is_peregrine` at its default. The
delegate hooks that branch adds (`OnRecvProgress`, and `OnRecvNetworkComplete`
once `layers_dispatched_[uuid]` reaches `num_layers()`) therefore never fire on
the modelled path; on the socket path `network_completed_` is set only by
`RecordBlockShardsReceivedLocked`.

The session side is `Transfer/Session.lean`'s `Lifecycle`, transcribed as
`Receive.lean` does: the push lease is `beginOp` (`TryBeginRecvOp` via
`hooks.begin_incoming_push`) and is returned inside the same event (`EndRecvOp`
via `hooks.end_incoming_push` / `HandleIncomingPush`); the H2D callback and the
poll finish through `finishLocked`/`endOpLocked`
(`TransferReceiveSession::FinishLocked`, `EndRecvOpLocked`). `inFlight` counts
only H2D copies whose callback has not run.

## State

| Field | C++ | Role |
|---|---|---|
| `Layer.entry` | `layer_progress_[{uuid, l}]` (`BlockTransport::LayerProgress`, `layer_progress_`) | `none` = no entry (never created, or erased by retirement) |
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
| `stream l s` | `HandleIncomingPush`: `BeginIncomingPush` → `TryBeginRecvOp` (`hooks.begin_incoming_push`, refuses once `done_`/`draining_`), payload, `CompleteIncomingPush`, on completion `OnLayerReceived` → `ExecuteLayerH2d` (`hooks.on_layer_received`, `++in_flight_`), then `OnBlockShardsReceived`, `EndIncomingPush`. A rejection inside `CompleteIncomingPush` returns early and `absl::Cleanup end_incoming_push` ends the push with `InternalError("Incoming push failed")`, which `hooks.end_incoming_push` turns into `Finish(status)` then `EndRecvOp()`. |
| `h2dDone l ok` | the `future.OnReady` callback of a layer's copy in `ExecuteLayerH2d`: on OK `num_completed_layers_++` and, at `num_layers()` while not draining, the success `FinishLocked()`; on error `FinishLocked(status_or.status(), kH2dPhase)`; then `absl::Cleanup end_op`'s `EndRecvOp()` |
| `publish` | `CompleteReadWithDetails`: a receive that is not draining and `IsReadyToComplete()` (`network_completed_ || num_completed_layers_ == num_layers`, and every issued future ready) is `Finish()`ed. "Every future ready" is taken as "every callback has run"; the window between the two is `ReceivePoll.lean`'s subject (`pollReady`) and changes no property here. |
| `cancel` | the deadline sweep in `CompleteReadWithDetails` (`else if (session->deadline() <= now)`) or any other `Finish(error)` on a receive that is not yet draining |

`published` (`life.done && life.statusOk`) is what the poll reports in
`done_recving_` (`CompleteReadWithDetails`).

## Lock scopes

`stream` is the one deliberately coarse event in the prefill-to-decode models:
it spans both `BlockTransport::progress_mu_` and `TransferReceiveSession::mu_`.
In order on the push thread: `TryBeginRecvOp` (session `mu_`), the payload
write (unlocked), the `expected_senders` lookup (`progress_mu_`), the unlocked
`ExpectedPushSenders` check (A2), the `layer_progress_` update with the
once-latch and retirement erase (`progress_mu_`, A1), then `OnLayerReceived` →
`ExecuteLayerH2d` (session `mu_` scopes, audited in `Receive.lean`),
`OnBlockShardsReceived` (session `mu_`, plus `MaybeUnregisterSettledRecv`), and
`EndIncomingPush` → `EndRecvOp` (session `mu_`, plus `MaybeUnregisterSettledRecv`
and `GetStatus()`). If `CompleteIncomingPush` rejects the push, the
`absl::Cleanup end_incoming_push` runs `EndIncomingPush(InternalError(...))`,
which calls `DeferUnregisterOnSettle()` and `Finish(status)` before `EndRecvOp()`.
`h2dDone`, `publish` and `cancel` are `Receive.lean`'s `h2dDone`, `pollReady`
and `cancel`.

| Event | Scopes | Between them |
|---|---|---|
| `stream l s` | `progress_mu_` and session `mu_` scopes above | concurrent streams serialize their `layer_progress_` updates on `progress_mu_` in the order of the model's `stream` events, and their session calls are covered by `Receive.lean`. If `Finish(error)` lands after `TryBeginRecvOp` and before the delegate calls finish, `ExecuteLayerH2d` and `OnBlockShardsReceived` return at `if (done_ || draining_)` and `EndRecvOp` releases the lease while `layer_progress_` keeps its latch; `Safe` still holds (`OnceDispatched` by the latch, `NetworkSound` vacuously because `blocks < c.target`, and `published = false`). Likewise, if an OK `Finish()` lands before `OnBlockShardsReceived`, every layer has already been dispatched and completed its H2D copy (`CompleteOnSuccess`), and `blocks < c.target` keeps `NetworkSound` vacuous |
| `h2dDone l ok` | 2 (`Receive.lean`) | an `EndRecvOp` tail (`Receive.lean`) |
| `publish` | `IsDraining()`, `IsReadyToComplete()`, `Finish()` (`Receive.lean`, `pollReady`) | monotone readiness; an OK `Finish()` on a draining session is a no-op |
| `cancel` | 1: `Finish(error)` | guarded here by `!draining`; a stale sweep is a no-op |

`BlockTransport::ForgetPushProgress` erases a uuid's entries under
`progress_mu_` from `KVCacheManagerBase::UnregisterActivePlanDirect`, reached
through `UnregisterSettledPlan` (the receive draining with its last op, or
settled) and `UnregisterActivePlan` (only once `Done()`); no stream passes
`TryBeginRecvOp` in either state, and a stream already past the gate that
recreates an entry finds the session draining. It is not an event.

## Assumptions

* **A1 (one lock).** All progress-table mutation is under `progress_mu_` in
  `CompleteIncomingPush`; the latch is set in the same critical section that
  decides completion (`if (array_complete && !progress.on_layer_received_called)`).
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
| `bt.h` | `BlockTransport` (`struct LayerProgress` .. `layers_dispatched_`) | `:295-343` |
| `bt.cc` | `ResolveIncomingPushSpec` | `:352-407` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:496-639` |
| `bt.cc` | `BlockTransport::OnRecvComplete` | `:644-680` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` | `:691-887` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` (`absl::Cleanup end_incoming_push`) | `:510-517` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` (`std::optional<size_t> expected_senders;` .. `expected_senders = block_delegate_->ExpectedPushSenders(uuid);`) | `:711-723` |
| `bt.cc` | `BlockTransport::ForgetPushProgress` | `:1524-1537` |
| `base.cc` | `KVCacheManagerBase::UnregisterActivePlanDirect` (`transport_server->ForgetPushProgress(uuid);`) | `:3209` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` (`bool array_complete = false;` .. `array_complete = progress.completed_chunks == expected_chunks;`) | `:737-815` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` (`bool all_layers_called = true;` .. `layer_progress_.erase({uuid, layer});`) | `:839-851` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`const uint32_t remote_id =` .. `static_cast<uint32_t>(header_layer_idx);`) | `:1263-1269` |
| `recv.h` | `TransferReceiveSession::Done` | `:86-89` |
| `recv.h` | `TransferReceiveSession::IsDraining` | `:103-106` |
| `recv.h` | `TransferReceiveSession::TryBeginRecvOp` | `:117-122` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:357-370` |
| `recv.cc` | `TransferReceiveSession::Finish` | `:372-375` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:406-417` |
| `recv.cc` | `TransferReceiveSession::EndRecvOp` | `:419-422` |
| `recv.cc` | `TransferReceiveSession::IsReadyToComplete` | `:450-456` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShardsReceivedLocked` | `:465-492` |
| `recv.cc` | `TransferReceiveSession::OnRecvProgress` | `:592-606` |
| `recv.cc` | `TransferReceiveSession::OnRecvNetworkComplete` | `:608-620` |
| `recv.cc` | `TransferReceiveSession::OnBlockShardsReceived` | `:622-670` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:672-793` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`future.OnReady`) | `:732-790` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.on_layer_received =`) | `:163-179` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.begin_incoming_push =`) | `:224-249` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.end_incoming_push =`) | `:250-296` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterActivePlan` | `:553-641` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:741-778` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterSettledPlan` | `:780-796` |
| `mgr.cc` | `KVCacheManagerWithTransfer::MaybeUnregisterSettledRecv` | `:798-804` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_descriptors)` | `:848-872` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:874-978` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_recv_sessions_.begin();`) | `:1035-1066` |
| `mgr.cc` | `KVCacheManagerWithTransfer::OnBlockShardsReceived` | `:1497-1521` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:518-540` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:2700-2717` |
| `tpu_sync/rpc/raiden_service.proto` | `StartTransferRequest` (`map<int32, ShardPushScheduleProto> shard_push_schedules`) | `:180` |
<!-- source-map:end -->
-/

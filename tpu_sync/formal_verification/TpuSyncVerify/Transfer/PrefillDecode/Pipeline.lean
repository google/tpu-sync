import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Model
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Lemmas
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Progress

/-!
# Prefill-to-decode pipeline

End-to-end single-request prefill-to-decode model: one `Send` session, one
`Recv` session and the five memories the KV data moves through between them.
The sessions are the `Send.lean` and `Receive.lean` models, used as-is; this
file adds what they abstract away — *which bytes* each copy moves, and *which
layer* each copy is for — and proves publication correctness: when the decode
engine is told `done_recving`, its HBM holds the prefill's KV cache. Buffer
safety properties (prefill HBM reclaimed, staging released) follow as
corollaries of the sessions' settle protocol and are stated here too.

`send.cc` is `tpu_sync/kv_cache/transfer_send_session.cc`; `recv.cc` is
`tpu_sync/kv_cache/transfer_receive_session.cc`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`; `bts.cc` is
`tpu_sync/transport/block_transport_session.cc`; `mgr.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`; `base.h` is
`tpu_sync/kv_cache/kv_cache_manager_base.h`. Exact line ranges for
every symbol and sub-block are in the generated `## Source map` table below.

## Memories

A memory is a list indexed by layer. A `Cell` is `.kv l` (layer `l`'s KV
data), `.junk` (whatever a buffer held before, or holds after it was reused)
or `.blank` (never written). Layer `l` of a memory is correct iff it is
`.kv l`.

| Memory           | Role | Written by |
|------------------|------|------------|
| `prefillHbm`     | the request's KV cache in prefill HBM | `reclaim` (the engine frees prefill HBM) |
| `prefillStaging` | the send's host staging (`AcquireStagingWithRetry` in `TransferSendSession::StartPush`) | `d2hReady l` (D2H copy, `base_->D2hSyncDispatch` in `StartPush`), `reseatPrefillStaging` |
| `wire`           | data delivered by a direct H2H write (`base_->H2hWriteDirectAsync` in `TransferSendSession::SendNextLayer`) | `h2hDone l true` |
| `decodeStaging`  | the receive's host staging | `land l` (transport, `ForEachPayload` in `BlockTransportServerPushSession::StartStream`), `reseatDecodeStaging` |
| `decodeHbm`      | the decode request's KV cache in decode HBM | `h2dReady l` (H2D copy, `TransferReceiveSession::ExecuteLayerH2d`) |

Both host stagings are one cell per layer: the full host mirror
(`KVCacheManagerBase::num_host_layers() == num_layers()`, the default). With
fewer host layer buffers than layers (`TPU_RAIDEN_NUM_STAGING_LAYERS`) each
session leases a buffer per layer from its `StagingLayerManager`
(`TransferSendSession::AcquireStagingSlot` before the layer's D2H,
`TransferReceiveSession::AcquireStagingSlot` from `GetBlockChunks` inside the
push lease) and returns it once the layer has left the buffer (the push
callback on the send side, the H2D callback on the receive side); that reuse
of one buffer by successive layers is outside this model, see `Receive.lean`
§"Bounded host layer staging".

## Layers complete in any order

The sessions count copies; they do not say which layer a copy was for. The
real system issues D2H copies in layer order (the loop over
`AcquireStagingSlot(l)` and `D2hSyncDispatch` in `TransferSendSession::StartPush`,
after `StartPush` has created every layer's D2H future
(`d2h_layer_futures_.emplace_back`) and armed the push chain with
`SendNextLayer(0)`) and pushes in layer order (`self->SendNextLayer(l + 1)` in
`TransferSendSession::SendNextLayer`), but D2H futures resolve in any order,
pushes travel over separate connections and land in any order
(`BlockTransportServerPushSession::CompleteIncomingPush` sets
`if (array_complete && !progress.on_layer_received_called)` and reports each
layer when *its* last block is in), and H2D futures resolve in any order. So
every event that touches a memory carries the layer it is for, and per-layer
ghost sets record which layers have passed each stage:

| Ghost         | Layers that … | Agrees with |
|---------------|---------------|-------------|
| `d2hReadyL`   | have finished their D2H copy (`prefillStaging[l]` written) | `send.d2hReady` (`cnt_d2hReady`) |
| `h2hRetiredL` | have run their push callback | `send.h2hRetired` (`cnt_h2hRetired`) |
| `landedL`     | the transport has landed in `decodeStaging` | — |
| `claimedL`    | `OnLayerReceived` has entered `ExecuteLayerH2d` for | — |
| `h2dPendingL` | are between the first and second lock of `ExecuteLayerH2d` | `recv.pending` (`cnt_h2dPending`) |
| `h2dIssuedL`  | `ExecuteLayerH2d` has dispatched to the device | `recv.issued` (`cnt_h2dIssued`) |
| `h2dReadyL`   | have finished their H2D copy (`decodeHbm[l]` written) | `recv.ready` (`cnt_h2dReady`) |

`reclaimed` records that the engine has freed prefill HBM.

## Events

| Event                   | What it is |
|-------------------------|-----------|
| `notifyForRead`         | `KVCacheManagerWithTransfer::NotifyForRead` registers the offer in `send_sessions_[uuid]` (`registered := true`) and calls `cv_.SignalAll()` |
| `pullWait`              | `KVCacheManagerWithTransfer::HandlePullStream` arrives before `NotifyForRead` (`registered = false`) and enters `cv_.WaitWithTimeout(&mu_, left)` (`pullWaiting := true`) |
| `send e`                | the send session's event `e`, no memory effect. `d2hReady` and `h2hDone` are disabled here (they are the layer-indexed events below). `beginPull` (`ValidateAndBeginPull`) requires `registered = true` and clears `pullWaiting`. `wake` additionally needs layer `woken`'s copy to have finished: `SendNextLayer(l)` waits on layer `l`'s future (`layer_future = d2h_layer_futures_[l]` .. `layer_future->OnReady`), not on any future. `StartPush` creates every layer's future before issuing any copy and calls `SendNextLayer(0)` before its D2H loop, so layer 0's wait is armed first; a layer whose copy is never issued has its promise failed with the send's status by the `absl::Cleanup fail_unissued_layers` when `StartPush` returns, so the waiting `SendNextLayer` runs its error arm |
| `d2hReady l`            | the D2H copy of layer `l` finishes: `prefillStaging[l] := prefillHbm[l]`. Issued iff `l < d2hIssued` (copies are issued in order) |
| `h2hDone l ok`          | the push callback for layer `l` in `SendNextLayer` (`H2hWriteDirectAsync`). Push `l` exists iff `l < h2hIssued` (pushes are issued in order). On success `wire[l] := prefillStaging[l]` |
| `recv e`                | the receive session's event `e`, no memory effect. `h2dBegin`, `h2dIssue` and `h2dReady` are disabled here. `pullReply ok` requires `ok = false ∨ send.pullStarted = true` and clears `pullWaiting` |
| `h2dBegin l`            | `OnLayerReceived(l)` → `ExecuteLayerH2d(l)` up to its first unlock: once per layer (A1), only after layer `l` landed (`if (array_complete && !progress.on_layer_received_called)` in `BlockTransportServerPushSession::CompleteIncomingPush`) |
| `h2dIssue l ok`         | `ExecuteLayerH2d(l)` from the `if (done_ || draining_)` re-check through `h2d_futures_.push_back(future)`: dispatches layer `l`'s H2D copy unless draining or failed |
| `h2dReady l`            | the H2D copy of layer `l` finishes: `decodeHbm[l] := decodeStaging[l]` |
| `land l`                | the transport writes layer `l` from the wire into `decodeStaging[l]`, inside an accepted push (`BeginIncomingPush` .. `EndIncomingPush` in `BlockTransportServerPushSession::StartStream`, reached from `BlockTransport::HandleIncomingPush`) |
| `reclaim`               | the engine frees prefill HBM once `poll_stats()` (`CompleteReadWithDetails`) has reported the send (`if (session->Done())` in the `send_sessions_` loop) |
| `reseatPrefillStaging`  | the host staging pool hands the send's released staging to someone else, who writes to it |
| `reseatDecodeStaging`   | likewise for the receive's staging |

A copy reads its source when it *completes*, not when it is issued. This is
the pessimistic choice: a source that is overwritten at any point during the
copy may be read after the overwrite. Corruption is never undone, so a model
that reads at issue time admits no violation this one does not.

## Lock scopes

Session events inherit the `TransferSendSession::mu_` and
`TransferReceiveSession::mu_` scopes from `Send.lean` and `Receive.lean`
(indexing copy and callback events by layer `l` only names which entry of
`d2h_layer_futures_[l]` or `OnLayerReceived(l)` runs). The pipeline-level events
and cross-node guards map to C++ synchronization as follows:

| Event | Scopes | Between them |
|---|---|---|
| `notifyForRead` | 1 producer-manager `mu_` scope in `NotifyForRead` (`send_sessions_.try_emplace(uuid, session)`), then `cv_.SignalAll()` outside `mu_` | a `HandlePullStream` call that acquires `mu_` between the unlock and `SignalAll` already finds the entry in `send_sessions_` and skips the wait |
| `pullWait` | 1 producer-manager `mu_` scope in `HandlePullStream` up to `cv_.WaitWithTimeout(&mu_, left)`, which atomically releases `mu_` to sleep | while asleep (`pullWaiting = true`), either `send_sessions_` has no entry (`registered = false`) or, after `notifyForRead`, the thread has not yet re-acquired `mu_` to run `ValidateAndBeginPull` |
| `send .beginPull` | `ValidateAndBeginPull`'s session `mu_` scope, nested inside `HandlePullStream`'s producer-manager `mu_` scope once `send_sessions_.find(req.uuid)` succeeds | no gap between waking from `WaitWithTimeout` with the entry present and `ValidateAndBeginPull`: both run in the same loop iteration under producer-manager `mu_` |
| `recv (.pullReply ok)` | 2 consumer-session `mu_` scopes (`Receive.lean`); `ok = true` additionally requires `send.pullStarted = true` | RPC causality: `HandlePullStream` returns `.status = 0` only after `ValidateAndBeginPull` (`send .beginPull`) succeeded; any exception, `stopping_` check, or timeout returns `.status = -1` (`ok = false`, clearing `pullWaiting`) |
| `land l` | unlocked payload write (`ForEachPayload` in `StartStream`) between `BeginIncomingPush` (`recv .pushBegin`, which holds a push lease `pushes > 0`) and `CompleteIncomingPush` → `OnLayerReceived` (`h2dBegin l`) | the push lease keeps `in_flight_ > 0` on the receive session throughout `ForEachPayload`, so `decodeStaging` cannot be released (`reseatDecodeStaging` is disabled) while `land l` writes it, and `h2dBegin l` cannot run until `ForEachPayload` returns |
| `reclaim`, `reseatPrefillStaging`, `reseatDecodeStaging` | 0 (caller engine or `StagingBlockAllocator`, outside session `mu_`) | enabled only after `send.published ≠ none` or `hasStaging = false`, which `Send.lean` and `Receive.lean` establish under their session locks |

## Assumptions

* **A1 (layers move whole).** Blocks are not modelled: a layer is copied as a
  unit. The transport delivers a layer's blocks individually but reports the
  layer only once the last block is in
  (`if (array_complete && !progress.on_layer_received_called)` in
  `BlockTransportServerPushSession::CompleteIncomingPush`), and both device
  copies are per layer.
* **A2 (delivery ordered after the callback).** In the model, the sender's
  push callback (`h2hDone l true`) puts layer `l` on the wire and the
  receiver lands it (`land l`) at or after that step. In the real system the
  write lands *before* the callback fires (the receiver acks after
  `EndIncomingPush(header.uuid)` in `BlockTransportServerPushSession::StartStream`). This
  reversal is sound because `prefillStaging[l]` is invariant throughout
  `[h2hIssue, h2hDone]`: the push holds an op (`in_flight_ > 0`), so the send
  cannot settle and its staging cannot be reseated, and `d2hReady l` has
  already run before `SendNextLayer(l)` woke and cannot run a second time.
  Reading `prefillStaging[l]` at `h2hDone` therefore reads the exact value that
  was present when the real transport streamed the data, so every real trace
  with a successful callback has a model trace with the same receiver-side
  states. The model cannot express a layer that landed whose push callback then
  reports failure (lost ack); that trace is safe for the same reason, but it
  is not in the model.
* **A3 (one sender per layer).** Each layer's data comes from one producer in
  one push stream. With several streams per layer (several producers, or one
  producer's push split by source NUMA node into shard groups — producer side
  `ShardGroupsByNuma` in `BlockTransport::AsyncPush`, joined into the layer's
  single callback via `tsl::JoinFutures`; receiver side `ShardsFromMask` in
  `BlockTransportServerPushSession::StartStream`) the transport's per-stream
  completion (`BlockTransportServerPushSession::CompleteIncomingPush`) and the
  receiver's per-shard counters
  decide when the layer is complete; each contributing stream still read a
  correct source, which is all the proof uses.
* **A4 (engine contract).** The decode engine reads the KV cache only after
  `poll_stats()` reports `done_recving`; the prefill engine frees prefill HBM
  only after it reports the send. Both are outside tpu-sync.

## Properties

All proved on every reachable state (`reachable_safe` and
`reachable_unregistered_safe`):

* **Publication correctness.** `recv.published = some true → decodeHbm = good n`:
  when the engine is told `done_recving`, decode HBM holds every layer of the
  prefill's KV cache (`PublicationCorrect`).
* **Decode HBM safety** (`DecodeHbmSafe`, `attention_safe`). Once the engine is
  told `done_recving` or `failed_recving` (`recv.published ≠ none`), no H2D copy
  is dispatching or writing to decode HBM (`pending = 0 ∧ retired = issued`);
  moreover, once `done_recving` is reported, `decodeHbm = good n` holds in every
  state reachable afterwards (`attention_safe`).
* **Prefill HBM safety.** `reclaimed → d2hPending = false ∧ d2hRetired = d2hIssued`:
  when the engine frees prefill HBM no D2H copy is dispatching or reading it,
  and (since the send has settled) none will be issued (`PrefillHbmSafe`,
  source buffer safety).
* **Staging safety.** A send whose staging was released has no copy writing
  it and no push reading it; a receive whose staging was released has no push
  writing it and no copy reading it (`StagingSafe`, staging
  integrity & quietness). Both follow from `done → inFlight = 0` and the
  sessions' accounting of `in_flight_`.
* **Handshake rendezvous safety** (`HandshakeSafe`). `StartPush`
  (`send.started = true`) and all D2H/H2H activity require the offer to have
  been registered by `NotifyForRead` (`registered = true`) and atomically
  claimed by `ValidateAndBeginPull` (`send.pullStarted = true`), and while
  `HandlePullStream` is in its grace wait (`pullWaiting = true`), the consumer's
  `PullStream` RPC remains pending (`recv.pullPending = true`) and
  `pull_started_` remains unclaimed (`send.pullStarted = false`).

Why the proof goes through, in one paragraph. A copy that completes is still
in flight, so its session is not settled; a session that is not settled still
owns its staging (nobody has reseated it) and, on the send side, the engine
has not reclaimed prefill HBM (it does that only after publication, which
needs settling). So every copy reads a buffer that nobody else has touched,
and each copy for layer `l` reads slot `l` and writes slot `l`, so by
induction on the chain HBM → staging → wire → staging → HBM each layer that
arrives is the right one, whatever order the layers arrive in. The guards that
make this true are exactly the sessions' refusal to begin an op once draining
and their refusal to settle while an op is in flight — the settle protocol of
stage 1 — plus the per-layer guards that a push waits for its own layer's
copy and an H2D dispatch for its own layer's landing.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:145-159` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Status acquire_status =` .. `Finish(acquire_status);`) | `:434-437` |
| `send.cc` | `TransferSendSession::StartPush` (`d2h_layer_futures_.emplace_back(std::move(future),`) | `:463-464` |
| `send.cc` | `TransferSendSession::StartPush` (`SendNextLayer(0);`) | `:483` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Cleanup fail_unissued_layers =`) | `:494-506` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::StatusOr<size_t> host_layer_idx = AcquireStagingSlot(l);` .. `next_layer = l + 1;`) | `:508-557` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`layer_future = d2h_layer_futures_[l];` .. `layer_future->OnReady`) | `:575-579` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:625-653` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->SendNextLayer(l + 1);`) | `:657` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:786-925` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`// The receive may have been finished` .. `h2d_futures_.push_back(future);`) | `:826-858` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:458-463` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:321-455` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`ABSL_RETURN_IF_ERROR(delegate_->BeginIncomingPush(header.uuid));` .. `FaultInjectSocket(hooks::kBlockTransportRecvSendAck, client_fd);`) | `:332-451` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`ShardsFromMask`) | `:371` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`ABSL_RETURN_IF_ERROR(ForEachPayload(`) | `:381-429` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:506-702` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`if (array_complete && !progress.on_layer_received_called)`) | `:635-666` |
| `bt.cc` | `BlockTransport::AsyncPush` | `:499-548` |
| `base.h` | `KVCacheManagerBase::num_host_layers` | `:799` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NotifyForRead` (`if (!send_sessions_.try_emplace(uuid, session).second)` .. `cv_.SignalAll();`) | `:685-691` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = send_sessions_.begin();` .. `send_sessions_.erase(it++);`) | `:1159-1177` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1431-1557` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` (`const absl::Time give_up = absl::Now() + grace;` .. `session->ValidateAndBeginPull`) | `:1455-1481` |
<!-- source-map:end -->
-/

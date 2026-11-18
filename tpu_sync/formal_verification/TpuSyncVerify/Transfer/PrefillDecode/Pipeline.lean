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

`send.h`/`send.cc` are `tpu_sync/kv_cache/transfer_send_session.{h,cc}`;
`recv.h`/`recv.cc` are `tpu_sync/kv_cache/transfer_receive_session.{h,cc}`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`; `bts.cc` is
`tpu_sync/transport/block_transport_session.cc`; `mgr.h`/`mgr.cc` are
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`; `base.h`/`base.cc` are
`tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`. Exact line ranges for
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

Both host stagings here are one cell per layer: the full host mirror
(`KVCacheManagerBase::num_host_layers() == num_layers()`, the default). With
fewer host layer buffers than layers (the `num_staging_layers` constructor
parameter, `ResolveNumStagingLayers` in `base.cc`), each session leases a
buffer per layer from its `StagingLayerManager`
(`TransferSendSession::AcquireStagingSlot` before the layer's D2H, then
resolved as `ResolveStagingSlot` → `TransferSendSession::LookupStagingSlot`
from `KVCacheManagerBase::GetBlockChunks` and `ValidateChunks` →
`KVCacheManagerBase::GetBlockArrayHostSpan` →
`KVCacheManagerBase::ResolveBlockArrayHostLayer` in
`BlockTransport::BuildBlockRequests`, each taking
`KVCacheManagerWithTransfer::mu_` in `GetSession(uuid)`; on the consumer,
`TransferReceiveSession::AcquireStagingSlot`, reached as `ResolveStagingSlot`
from both `KVCacheManagerBase::GetBlockChunks` and `ValidateChunks` →
`KVCacheManagerBase::GetBlockArrayHostSpan` →
`KVCacheManagerBase::ResolveBlockArrayHostLayer` under `GetSession(uuid)`
inside the push lease) and returns it once the layer has left the buffer (the push
callback on the send side, the H2D callback on the receive side);
`PrefillDecode/LayerStaging.lean` refines `Pipeline` with bounded `prefillSlots`
and `decodeSlots` and proves publication correctness, slot conservation,
settlement, and `1`-slot sufficiency over this `Pipeline.step`.

**Host-memory source and destination (boundary, not modelled).** Both ends
of the protocol take a `BlockLocation` (`mgr.h`: `enum class BlockLocation {
kDevice, kHost }`), defaulting to `kDevice`, which is the path above. With
`source = kHost` on `NotifyForRead` / `TransferSendSession::Create`, the
registered ids name blocks of the producer's host pool and `StartPush` takes
its `if (source_ == BlockLocation::kHost)` branch: no `AcquireStagingWithRetry`,
no D2H loop, every entry of `d2h_layer_futures_` is an already-OK future and
`SendNextLayer(0)` pushes straight from those host blocks — `prefillHbm` and
`prefillStaging` collapse into one memory that tpu-sync never writes. With
`destination = kHost` on `StartRead` / `TransferReceiveSession::Create`, the
pull lands in the host blocks `local_block_ids` and
`TransferReceiveSession::OnDataReceived` routes the layer report to
`SettleHostLayer` instead of `ExecuteLayerH2d`: no H2D, and the last layer's
report closes the receive from inside the transport's push lease
(`RecordNetworkCompleteLocked(/*all_complete=*/true)` → `FinishLocked()`)
rather than through `pollReady` or the last `h2dIssue` — `decodeStaging` is
the final destination and `decodeHbm` is never written. Both branches are
admitted only on a full host mirror with no leased host blocks
(`KVCacheManagerWithTransfer::CheckHostMemoryTransferSupported`:
`num_host_layers() == num_layers()` and `!staging_allocator_->LeasesHostBlocks()`),
so bounded layer staging (`PrefillDecode/LayerStaging.lean`) never meets
them. `step` has no event or `Cfg` flag for either branch: the session models
this file composes (`Send.lean`, `Receive.lean`) model the device path, and
the host path changes which memories exist and where the receive settles, not
the settle protocol the properties below rest on. The properties are stated
for the device path; for a host-source send, "`prefillStaging[l]` is read by
push `l`" reads "the caller's host block for layer `l`", whose stability until
the send settles is the caller's contract (`NotifyForRead`'s declaration in
`mgr.h`), an instance of A4.

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
| `notifyForRead`         | `KVCacheManagerWithTransfer::NotifyForRead` registers the offer in `sessions_[uuid]` (a `SessionEntry{session, req_id, SessionRole::kTransfer}`; `registered := true`) and calls `cv_.SignalAll()` |
| `pullWait`              | `KVCacheManagerWithTransfer::HandlePullStream` arrives before `NotifyForRead` (`registered = false`) and enters `cv_.WaitWithTimeout(&mu_, left)` (`pullWaiting := true`) |
| `send e`                | the send session's event `e`, no memory effect. `d2hReady` and `h2hDone` are disabled here (they are the layer-indexed events below). `beginPull` (`ValidateAndBeginPull`) requires `registered = true` and clears `pullWaiting`. `wake` additionally needs layer `woken`'s copy to have finished: `SendNextLayer(l)` waits on layer `l`'s future (`layer_future = d2h_layer_futures_[l]` .. `layer_future->OnReady`), not on any future. `StartPush` creates every layer's future before issuing any copy and calls `SendNextLayer(0)` before its D2H loop, so layer 0's wait is armed first; a layer whose copy is never issued has its promise failed with the send's status by the `absl::Cleanup fail_unissued_layers` when `StartPush` returns, so the waiting `SendNextLayer` runs its error arm. `publish` is `KVCacheManagerWithTransfer::CompleteReadWithDetails` retiring the settled send through `RetireSettledSessionLocked` (`if (!send->Done()) return false;`): the outcome goes to `done_sending_` / `failed_recving_` and the entry leaves `sessions_` |
| `d2hReady l`            | the D2H copy of layer `l` finishes: `prefillStaging[l] := prefillHbm[l]`. Issued iff `l < d2hIssued` (copies are issued in order) |
| `h2hDone l ok`          | the push callback for layer `l` in `SendNextLayer` (`H2hWriteDirectAsync`). Push `l` exists iff `l < h2hIssued` (pushes are issued in order). On success `wire[l] := prefillStaging[l]` |
| `recv e`                | the receive session's event `e`, no memory effect. `h2dBegin`, `h2dIssue` and `h2dReady` are disabled here. `pullReply ok` requires `ok = false ∨ send.pullStarted = true` and clears `pullWaiting`. `pollReady` is `CompleteReadWithDetails` finding `recv->IsReadyToComplete()` on a session that is not draining and calling `recv->Finish()`; `publish` is the same loop's `RetireSettledSessionLocked` (`if (!recv->Done()) return false;`) moving the outcome to `done_recving_` / `failed_recving_` and erasing the entry from `sessions_` |
| `h2dBegin l`            | `KVCacheManagerBase::OnLayerReceived(l, uuid)` → `GetSession(uuid)` → `TransferReceiveSession::OnDataReceived(l, {}, {})` → `ExecuteLayerH2d(l)` (the device-destination arm) up to its first unlock: once per layer (A1), only after layer `l` landed (`if (array_complete && !progress.on_layer_received_called)` in `BlockTransportServerPushSession::CompleteIncomingPush`) |
| `h2dIssue l ok`         | `ExecuteLayerH2d(l)` from the `if (done_ || draining_)` re-check through `h2d_futures_.push_back(future)` and the third `mu_` scope (`dispatched_layers_.insert(layer_idx)`): dispatches layer `l`'s H2D copy unless draining or failed; when the dispatch is the `num_layers()`-th and the receive is neither done nor draining it also sets `network_completed_` (`Recv.netIssued`) |
| `h2dReady l`            | the H2D copy of layer `l` finishes: `decodeHbm[l] := decodeStaging[l]` |
| `land l`                | the transport writes layer `l` from the wire into `decodeStaging[l]`, inside an accepted push (`BeginIncomingPush(delegate_, header.uuid)` .. `EndIncomingPush(session)` in `BlockTransportServerPushSession::StartStream`, reached from `BlockTransport::HandleIncomingPush`) |
| `reclaim`               | the engine frees prefill HBM once `poll_stats()` (`CompleteReadWithDetails`) has reported the send (`RetireSettledSessionLocked` in the `sessions_` loop) |
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
| `notifyForRead` | 1 producer-manager `mu_` scope in `NotifyForRead` (`sessions_.try_emplace(uuid, SessionEntry{...})`), then `cv_.SignalAll()` outside `mu_` | a `HandlePullStream` call that acquires `mu_` between the unlock and `SignalAll` already finds the entry in `sessions_` and skips the wait |
| `pullWait` | 1 producer-manager `mu_` scope in `HandlePullStream` up to `cv_.WaitWithTimeout(&mu_, left)`, which atomically releases `mu_` to sleep | while asleep (`pullWaiting = true`), either `sessions_` has no send entry for the uuid (`registered = false`) or, after `notifyForRead`, the thread has not yet re-acquired `mu_` to run `ValidateAndBeginPull` |
| `send .beginPull` | `ValidateAndBeginPull`'s session `mu_` scope, nested inside `HandlePullStream`'s producer-manager `mu_` scope once `FindSessionLocked<TransferSendSession>(req.uuid)` succeeds | no gap between waking from `WaitWithTimeout` with the entry present and `ValidateAndBeginPull`: both run in the same loop iteration under producer-manager `mu_` |
| `recv (.pullReply ok)` | 2 consumer-session `mu_` scopes (`Receive.lean`); `ok = true` additionally requires `send.pullStarted = true` | RPC causality: `HandlePullStream` returns `.status = 0` only after `ValidateAndBeginPull` (`send .beginPull`) succeeded; any exception, `stopping_` check, or timeout returns `.status = -1` (`ok = false`, clearing `pullWaiting`) |
| `land l` | unlocked payload write (`ForEachPayload` in `StartStream`) between `BeginIncomingPush` (`GetSession(uuid)` under manager `mu_`, then `TryBeginOp()` under session `mu_`: `recv .pushBegin`, which holds a push lease `pushes > 0`) and `CompleteIncomingPush` → `OnLayerReceived` (`h2dBegin l`) | the push lease keeps `in_flight_ > 0` on the receive session throughout `ForEachPayload`, so `decodeStaging` cannot be released (`reseatDecodeStaging` is disabled) while `land l` writes it, and `h2dBegin l` cannot run until `ForEachPayload` returns |
| `send .publish`, `recv .publish`, `recv .pollReady` | 1 manager `mu_` scope in `CompleteReadWithDetails` (the loop over `sessions_`), with the session's `IsDraining()` / `Finish()` / `Done()` each taking session `mu_` inside it (`RetireSettledSessionLocked`) | a session found `Done()` never changes again (`Lifecycle`), so publishing and erasing it under manager `mu_` sees the same outcome any later scope would; `pollReady`'s session scopes are §"The poll" in `Receive.lean` |
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
  `EndIncomingPush(session)` in `BlockTransportServerPushSession::StartStream`). This
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

None of these orders the block report (`recv .netAccount`,
`KVCacheManagerBase::OnBlockShardsReceived` → `OnDataReceived(0, block_ids,
shard_ids)`) against the layer completions. `pollReady` needs
`network_completed_`, which the receiver sets either from the block report
or from the last `h2dIssue l true` (`Recv.netIssued`); both routes require
every layer's copy to have been issued, and `IsReadyToComplete()` also needs
every issued copy ready, so a receive that `pollReady` finishes has
`decodeHbm = good n` by the per-layer chain alone, with no assumption about
when the transport's block accounting arrives.

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
* **Prefill HBM safety.** `send.published ≠ none → send.d2hPending = false ∧ send.d2hRetired = send.d2hIssued`:
  once the send is reported — which `reclaim` requires, so in particular when
  the engine frees prefill HBM — no D2H copy is dispatching or reading it,
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
`Transfer/Session.lean` — plus the per-layer guards that a push waits for its own layer's
copy and an H2D dispatch for its own layer's landing.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `send.h` | `TransferSendSession::ResolveStagingSlot` | `:158-160` |
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:200-214` |
| `send.cc` | `TransferSendSession::AcquireStagingSlot` | `:236-265` |
| `send.cc` | `TransferSendSession::LookupStagingSlot` | `:267-286` |
| `send.cc` | `TransferSendSession::StartPush` (`if (source_ == BlockLocation::kHost)`) | `:488-510` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Status acquire_status =` .. `Finish(acquire_status);`) | `:516-519` |
| `send.cc` | `TransferSendSession::StartPush` (`d2h_layer_futures_.emplace_back(std::move(future),`) | `:544-545` |
| `send.cc` | `TransferSendSession::StartPush` (`// 1. Arm the H2H chain on layer 0's future before any copy is issued, so` .. `SendNextLayer(0);`) | `:559-564` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Cleanup fail_unissued_layers =`) | `:575-587` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::StatusOr<size_t> host_layer_idx = AcquireStagingSlot(l);` .. `next_layer = l + 1;`) | `:589-638` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`layer_future = d2h_layer_futures_[l];` .. `layer_future->OnReady`) | `:656-660` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:706-734` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->SendNextLayer(l + 1);`) | `:738` |
| `recv.h` | `TransferReceiveSession::ResolveStagingSlot` | `:160-162` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:680-709` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:771-903` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`// The receive may have been finished` .. `h2d_futures_.push_back(future);`) | `:803-835` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`dispatched_layers_.insert(layer_idx);` .. `network_completed_ = true;`) | `:894-897` |
| `recv.cc` | `TransferReceiveSession::OnDataReceived` | `:721-740` |
| `recv.cc` | `TransferReceiveSession::SettleHostLayer` | `:742-769` |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` |
| `base.cc` | `KVCacheManagerBase::GetSession` | `:3081-3087` |
| `base.cc` | `KVCacheManagerBase::OnLayerReceived` | `:3345-3354` |
| `base.cc` | `KVCacheManagerBase::OnBlockShardsReceived` | `:3323-3341` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`size_t host_layer_idx = layer_idx;` .. `base_host_ptr = host_buffer->data();`) | `:3498-3512` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `bts.cc` | `ValidateChunks` | `:275-310` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`BeginIncomingPush(delegate_, header.uuid));` .. `FaultInjectSocket(hooks::kBlockTransportRecvSendAck, client_fd);`) | `:383-501` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`ShardsFromMask`) | `:420` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`ABSL_RETURN_IF_ERROR(ForEachPayload(`) | `:430-479` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`if (array_complete && !progress.on_layer_received_called)`) | `:852-879` |
| `bt.cc` | `BlockTransport::AsyncPush` | `:520-569` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(` .. `ValidateChunks(block_delegate_, l, sh, uuid, chunks));`) | `:836-846` |
| `base.h` | `KVCacheManagerBase::num_host_layers` | `:713` |
| `base.cc` | `*` (`size_t ResolveNumStagingLayers(size_t num_layers,` .. `return requested;`) | `:718-734` |
| `mgr.h` | `BlockLocation` | `:99` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`// ids name blocks of this manager's host pool, pushed as they are (no` .. `BlockLocation source = BlockLocation::kDevice);`) | `:187-199` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NotifyForRead` (`if (!sessions_` .. `cv_.SignalAll();`) | `:578-594` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CheckHostMemoryTransferSupported` | `:1402-1414` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = sessions_.begin();` .. `retired_uuids.push_back(uuid);`) | `:1084-1107` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` | `:1229-1358` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandlePullStream` (`const absl::Time give_up = absl::Now() + grace;` .. `session->ValidateAndBeginPull`) | `:1253-1278` |
<!-- source-map:end -->
-/

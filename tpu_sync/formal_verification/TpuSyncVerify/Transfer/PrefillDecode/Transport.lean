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
`tpu_sync/transport/block_transport_session.h`, `recv.h` =
`tpu_sync/kv_cache/transfer_receive_session.h`, `recv.cc` =
`tpu_sync/kv_cache/transfer_receive_session.cc`, `mgr.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`, `send.cc` =
`tpu_sync/kv_cache/transfer_send_session.cc`, `base.cc` =
`tpu_sync/kv_cache/kv_cache_manager_base.cc`, `coord.cc` =
`tpu_sync/kv_cache/reshard/reshard_coordinator.cc`, `planner.cc` =
`tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`, `block_transport_session_test.cc` =
`tpu_sync/transport/block_transport_session_test.cc`, `block_transport_test.cc` =
`tpu_sync/transport/block_transport_test.cc`. Exact line ranges for every
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
all-layers form `local_id == lib::kUnspecifiedLocalId` in `ResolveIncomingPushSpec` and
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
resharding, whose completion goes to the receive session directly via
`session->OnDataReceived(l, {}, {})`) is out of scope; see `Reshard/`. So is the
opt-in Peregrine contract (`else if (is_peregrine)`): when
`TPU_RAIDEN_ENABLE_PEREGRINE` selects `TransportBackend::kPeregrine` (default
off), `BlockTransport::AsyncPush` routes `kBlockPushExplicit` pushes to
`PeregrineTransportAdapter::PostExplicitPush` (`raddr = nullptr`, per-request
metadata) and `BlockTransport`'s constructor wires
`BlockTransport::ResolveRecvBuffer` and `BlockTransport::OnRecvComplete` into
`PeregrineTransportAdapter::RecvHooks` →
`BlockTransportServerPushSession::OnPeregrinePostComplete` (non-explicit
`kBlockPush` returns `Unimplemented` on Peregrine and `kBlockPull` is marked
`TODO` in `SyncPullInternal` and `PostPull`, and the default
socket path `StartStream` leaves `is_peregrine` false). On the receive session
`network_completed_` has two setters: the block report
(`RecordBlockShardsReceivedLocked`, `networkCompleted` here) and, once every
layer's copy has been issued, `TransferReceiveSession::ExecuteLayerH2d`
(`dispatched_layers_.insert(layer_idx)` reaching `num_layers()` on a receive
that is neither done nor draining), which `Receive.lean` models as `netIssued`.
The second setter is not state here: it fires only after every layer was
dispatched, so `NetworkSound` holds for it by construction, and in every state
where it would make `publish` enabled — all layers dispatched, no copy pending,
not draining — every callback has already run OK and `completedLayers =
numLayers` enables `publish` already.

The session side is `Transfer/Session.lean`'s `Lifecycle`, transcribed as
`Receive.lean` does: the push lease is `beginOp` (`TransferReceiveSession::TryBeginOp`,
called by `BeginIncomingPush` in `block_transport_session.cc` on the session
`delegate_->GetSession(uuid)` returns) and is returned inside the same event
(`TransferReceiveSession::EndOp` via `EndIncomingPush`, from `StartStream`);
the H2D callback and the poll finish through `finishLocked`/`endOpLocked`
(`TransferReceiveSession::FinishLocked`, `EndRecvOpLocked`). `inFlight` counts
only H2D copies whose callback has not run.

## State

| Field | C++ | Role |
|---|---|---|
| `Layer.entry` | `layer_progress_[l]` of the uuid's push session (`BlockTransportServerPushSession::LayerProgress`, `layer_progress_`) | `none` = no entry (never created, or the fresh session's table after retirement) |
| `Progress.completedChunks` | `LayerProgress::completed_chunks` | chunk-count contract |
| `Progress.landed` | `SenderStreams::landed` per sender | sender-declared contract |
| `Progress.sendersComplete` | `LayerProgress::senders_complete` | sender-declared contract |
| `Progress.called` | `LayerProgress::on_layer_received_called` | the once-latch |
| `Layer.sent` | ghost | streams accepted per sender for this layer |
| `Layer.dispatched` | ghost | `OnLayerReceived` → `KVCacheManagerBase::OnLayerReceived` → `OnDataReceived` → `ExecuteLayerH2d` entries for this layer |
| `Layer.pending` | subset of `h2d_futures_` (`ExecuteLayerH2d`: `h2d_futures_.push_back(future)`) | copies issued for this layer whose callback has not run |
| `blocks` | `blocks_received_per_shard_` collapsed to one shard (`RecordBlockShardsReceivedLocked`) | block units reported by `OnBlockShardsReceived` → `OnDataReceived` → `RecordBlockShards` |
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
| `stream l s` | `HandleIncomingPush` → `GetOrCreatePushSession` → `StartStream`: `BeginIncomingPush` → `delegate_->GetSession(uuid)` → `TryBeginOp` (refuses once `done_`/`draining_`, and always on a send session), payload, `CompleteIncomingPush`, on completion `OnLayerReceived` → `KVCacheManagerBase::OnLayerReceived` → `OnDataReceived(layer_idx, {}, {})` → `ExecuteLayerH2d` (`++in_flight_`), then `OnBlockShardsReceived` → `OnDataReceived(0, block_ids, shard_ids)` → `RecordBlockShards`, `EndIncomingPush` → `EndOp`. A rejection inside `CompleteIncomingPush` returns early and `absl::Cleanup end_incoming_push` ends the push with `InternalError("Incoming push failed")`, which `EndIncomingPush` turns into `Finish(status)` then `EndOp()`. A payload error before `CompleteIncomingPush` — a socket error, a size mismatch, or `ValidateChunks` finding a chunk outside the span `delegate_->GetBlockArrayHostSpan(l, sh, uuid)` returns (or that call failing) — takes the same cleanup with no table entry touched, i.e. `cancel` on a receive that is not yet draining. |
| `h2dDone l ok` | the `future.OnReady` callback of a layer's copy in `ExecuteLayerH2d`: on OK `num_completed_layers_++` and, at `num_layers()` while not draining, the success `FinishLocked()`; on error `FinishLocked(status_or.status(), kH2dPhase)`; then `absl::Cleanup end_op`'s `EndOp()` |
| `publish` | `CompleteReadWithDetails`: a receive that is not draining and `IsReadyToComplete()` (`network_completed_ || num_completed_layers_ == num_layers`, and every issued future ready) is `Finish()`ed. "Every future ready" is taken as "every callback has run"; the window between the two is `ReceivePoll.lean`'s subject (`pollReady`) and changes no property here. |
| `cancel` | the deadline sweep in `CompleteReadWithDetails` (`else if (session->deadline() <= now)`) or any other `Finish(error)` on a receive that is not yet draining |

`published` (`life.done && life.statusOk`) is what the poll reports in
`done_recving_` (`CompleteReadWithDetails`).

## Lock scopes

Four mutexes meet on the push thread: `BlockTransport::server_sessions_mu_`
(the uuid → push-session map `active_push_sessions_`), the manager's
`KVCacheManagerWithTransfer::mu_` (the uuid → `TransferSession` map `sessions_`
read by `GetSession`), the push session's `BlockTransportServerPushSession::mu_`
(its `layer_progress_` and `done_`), and the receive session's
`TransferReceiveSession::mu_`. `stream` is the one deliberately coarse event in
the prefill-to-decode models: it spans all four. In order on the push thread:
`GetOrCreatePushSession` (`server_sessions_mu_`, creating the session on the
uuid's first stream), `BeginIncomingPush` = `delegate_->GetSession(uuid)`
(manager `mu_`, returning the `shared_ptr` the rest of the stream holds) then
`TryBeginOp` (receive-session `mu_`), the payload write (unlocked; under
bounded host layer staging each chunk's `KVCacheManagerBase::GetBlockChunks`
and companion `ValidateChunks` → `KVCacheManagerBase::GetBlockArrayHostSpan` →
`KVCacheManagerBase::ResolveBlockArrayHostLayer` each look the session up under
manager `mu_` via `GetSession(uuid)` and run `session->ResolveStagingSlot` =
`TransferReceiveSession::AcquireStagingSlot`, audited in `LayerStaging.lean`), the
`expected_senders` lookup (push-session `mu_`), the unlocked
`ExpectedPushSenders` check (A2), the `layer_progress_` update with the
once-latch and retirement (push-session `mu_`, A1), then `OnLayerReceived` →
`KVCacheManagerBase::OnLayerReceived` (`GetSession`, manager `mu_`) →
`OnDataReceived` → `ExecuteLayerH2d` (receive-session `mu_` scopes, audited in
`Receive.lean`), `OnBlockShardsReceived` → `KVCacheManagerBase::OnBlockShardsReceived`
(`GetSession`, manager `mu_`) → `OnDataReceived` → `RecordBlockShards`
(receive-session `mu_`), and `EndIncomingPush` → `EndOp` (receive-session
`mu_`, plus `GetStatus()`); back in `HandleIncomingPush`, `DeletePushSession`
(`server_sessions_mu_`) if the push session `IsDone()`. If
`CompleteIncomingPush` rejects the push, the `absl::Cleanup end_incoming_push`
runs `EndIncomingPush(session, InternalError(...))`, which calls
`Finish(status)` before `EndOp()`. `h2dDone`, `publish` and `cancel` are
`Receive.lean`'s `h2dDone`, `pollReady` and `cancel`.

| Event | Scopes | Between them |
|---|---|---|
| `stream l s` | `server_sessions_mu_`, manager `mu_`, push-session `mu_` and receive-session `mu_` scopes above | concurrent streams of one uuid share one push session (`GetOrCreatePushSession` returns the existing entry) and serialize their `layer_progress_` updates on its `mu_` in the order of the model's `stream` events; streams of other uuids touch other sessions. Their receive-session calls are covered by `Receive.lean`; the `shared_ptr` from `GetSession` keeps the receive session alive for the stream even if the manager retires it from `sessions_` meanwhile. If `Finish(error)` lands after `TryBeginOp` and before the delegate calls finish, `ExecuteLayerH2d` and `RecordBlockShards` return at `if (done_ || draining_ || total_blocks_ == 0)` and `EndOp` releases the lease while `layer_progress_` keeps its latch; `Safe` still holds (`OnceDispatched` by the latch, `NetworkSound` vacuously because `blocks < c.target`, and `published = false`). Likewise, if an OK `Finish()` lands before `OnBlockShardsReceived`, every layer has already been dispatched and completed its H2D copy (`CompleteOnSuccess`), and `blocks < c.target` keeps `NetworkSound` vacuous |
| `h2dDone l ok` | 2 (`Receive.lean`) | an `EndOp` tail (`Receive.lean`) |
| `publish` | `IsDraining()`, `IsReadyToComplete()`, `Finish()` (`Receive.lean`, `pollReady`) | monotone readiness; an OK `Finish()` on a draining session is a no-op |
| `cancel` | 1: `Finish(error)` | guarded here by `!draining`; a stale sweep is a no-op |

**Retirement.** `if (all_layers_called)` in `CompleteIncomingPush` sets the push
session's `done_` in the same critical section as the latch and leaves
`layer_progress_` in place. A retired session is not handed out again:
`GetOrCreatePushSession` returns the map's session only while
`!session->IsDone() || session->HasAdmittedPosts()` (the second disjunct is the
Peregrine path's lease bookkeeping) and otherwise seats a fresh
`BlockTransportServerPushSession` under the uuid, and `HandleIncomingPush`
drops a session that `IsDone()` once `StartStream` has returned (after the
delegate callbacks, `EndIncomingPush` and the ack) through
`DeletePushSession(session)`, which erases the map entry only if it still holds
that session. Under both **A4** (`honest = true`) and `Cfg.WF.chunkSingle`
(`c.declared = false → c.numSenders = 1`), every sender's declared streams for
every layer have already landed when `all_layers_called` sets `done_`, so no
stream arrives at or after retirement. When `honest = false`, `retire`
(`Layer.entry := none`) models the fresh-lookup path: a stream arriving *after*
`active_push_sessions_` dropped or replaced the retired
`BlockTransportServerPushSession` enters through `GetOrCreatePushSession` on a
fresh table (`entry = none`). By contrast, an in-flight stream that already
took `shared_ptr<BlockTransportServerPushSession>` before `done_` was set and
reaches `CompleteIncomingPush` afterwards (`StartStream` and
`CompleteIncomingPush` never test `done_`) operates on the retained
pre-retirement `layer_progress_`: on the chunk-count contract it increments the
latched entry (`on_layer_received_called == true`) harmlessly without
dispatching again, whereas on the sender-declared contract (`streams.landed`
already at `streams.declared`) `++streams.landed` hits
`streams.landed > streams.declared` (`AlreadyExistsError`) and fails the
`TransferReceiveSession` through `absl::Cleanup end_incoming_push` →
`EndIncomingPush` (even when `numStreams ≥ 2`, where a fresh table accepts the
first post-retirement stream without error; `entry := none` is therefore a
fresh-lookup model rather than a pure over-approximation of `shared_ptr`
retention on sender-declared pushes).

**`ForgetPushProgress`.** `BlockTransport::ForgetPushProgress(uuid)` removes
`uuid`'s session from `active_push_sessions_` (under `server_sessions_mu_`) and
`Cancel()`s it (`ReleaseAdmittedPosts`: `done_`, `status_ = CancelledError`, and
the release of any Peregrine leases). `KVCacheManagerWithTransfer` calls it at
five sites, always *outside* `KVCacheManagerWithTransfer::mu_`:
- `CompleteReadWithDetails`, for each uuid `RetireSettledSessionLocked` retired
  with `retired_push_progress` that is not currently in `sessions_` or
  `pending_pool_reshard_recvs_` (a settled receive, a settled pool-reshard
  receive, or a failed send of either kind);
- `UnregisterActivePlan`, once `session->Done()`;
- `RegisterActivePlan`, when retiring a settled incumbent (after already seating
  the new `send_session` or `recv_session` in `sessions_` before releasing `mu_`);
- `PoolReshardPush`, after retiring a settled incumbent and before seating the
  new reshard send; and
- `PoolReshardRegisterRecv`, after retiring a settled incumbent under
  `pending_pool_reshard_recvs_` and before seating the new reshard receive.

By contrast, inline retirement in `StartRead(remote_endpoint)` and in
`RegisterRecv` (via `EmplaceRecvSessionLocked`; `RegisterActivePlan` already
retired `existing` with `&retired_existing` before calling
`EmplaceRecvSessionLocked`) invokes `RetireSettledSessionLocked` with
`retired_push_progress = nullptr`, leaving any unretired
`active_push_sessions_[uuid]` entry in place. Because `ForgetPushProgress(uuid)`
keys `active_push_sessions_` by integer `uuid` and runs unlocked, reusing a
`uuid` across generations could either inherit stale `active_push_sessions_[uuid]`
progress (after inline retirement of a partial receive) or have an unlocked
`ForgetPushProgress(uuid)` from the retired generation erase/cancel progress of
streams already admitted into the new live receive (e.g. in
`RegisterActivePlan`). Cross-generation `uuid` reuse is excluded by **A5**
below; within a single `uuid` generation, `ForgetPushProgress(uuid)` runs only
once the receive session is `Done()`, when `TryBeginOp` already refuses every
further stream, so it is not an event.

## Assumptions

* **A1 (one lock).** All progress-table mutation is under the push session's
  `mu_` in `CompleteIncomingPush`, and a session holds exactly one uuid's
  table; the latch is set in the same critical section that decides completion
  (`if (array_complete && !progress.on_layer_received_called)`).
  Encoded by making the table update one step.
* **A2 (stable expectation).** `ExpectedPushSenders(uuid)` is read outside
  the lock and latched on the layer's first stream (`if (!progress.expected_senders_resolved)`),
  so the contract a layer runs under is whatever the first stream saw. Encoded as
  the fixed `Cfg.declared`. `KVCacheManagerBase::ExpectedPushSenders` reads the
  plan of the uuid's session (`GetTransferPlan` → `GetSession(uuid)->plan()` →
  `TransferPlan::ExpectedPushSenders`) and returns `std::nullopt` when no
  receiver plan is registered (`StartRead`, `RegisterRecv`, `KVCacheStore` /
  `WorkerServiceImpl::TransferBuffers`, and `WeightSynchronizerBase`), when
  `pool_groups_size() > 0` (pool reshard via `ReshardCoordinator::RunSubPlan`,
  which takes the `GetPoolPushProgressSpec` branch in `CompleteIncomingPush`),
  and when `shard_push_schedules_size() == 0` (including primary
  `StageBroadcastGroup` hops in `BroadcastEngine.execute_slice_broadcast_pipeline`
  (`_dispatch_hop`) in
  `tpu_sync/weight_sync/manager/broadcast_engine.py`), so the sender-declared
  contract (`sender_streams`) is entered only on non-pool receiver
  `RegisterActivePlan` plans (`pool_groups_size() == 0` with
  `shard_push_schedules_size() > 0`). Inside
  `KVCacheManagerWithTransfer::RegisterActivePlan` the receive session carries
  its plan from construction (`TransferReceiveSession::CreateFromActivePlan`)
  and is seated in `sessions_` (`EmplaceRecvSessionLocked`) under the manager's
  `mu_`, which is the one `GetSession` takes, so the plan becomes visible to
  `ExpectedPushSenders` and to `TryBeginOp` in the same step, and `A2` holds
  whenever senders are dispatched only after `RegisterActivePlan` has returned
  (whereas a push arriving before or concurrently with `RegisterActivePlan`
  could latch chunk-count on a sender-declared plan, `trace_chunk_two_senders`). Every
  in-repo controller hop that populates receiver `shard_push_schedules` on a
  non-pool plan (`BroadcastEngine.execute_slice_broadcast_pipeline`
  (`_run_single_transfer`) in
  `tpu_sync/weight_sync/manager/broadcast_engine.py`, invoked from
  `RaidenController.start_transfer` in `tpu_sync/rpc/raiden_controller.py`, and
  remote destination preparation via
  `RaidenControllerClientFacade.register_transfer_schedule` in
  `tpu_sync/weight_sync/manager/controller_rpc_service.py`) awaits the
  destination `COMMAND_START_TRANSFER` RPC (`KVCacheListener::HandleControlRequest`
  runs `register_active_plan` synchronously before replying) before dispatching
  any sender; `A2` therefore remains an external ordering assumption only against
  callers that dispatch pushes before `RegisterActivePlan` returns.
* **A3 (uniform declaration, one node per schedule).** Every sender declares
  the same `numStreams`, and no sender outside the expected set appears; the
  `FailedPrecondition` and "more senders" rejections in `CompleteIncomingPush`
  (`else if (streams.declared != num_streams)` and
  `if (progress.sender_streams.size() > *expected_senders)`) are
  therefore never taken and are not modelled. Senders are told apart by
  `header.remote_id = node_id()` (`BuildBlockRequests`:
  `remote_id = static_cast<uint32_t>(block_delegate_->node_id())`), which also
  keys the plan's chunk lookup in `GetBlockChunks` →
  `TransferPlan::ResolvePlanBlockChunks` (`if (sender_node_id != -1)`), while
  `expected_senders` counts `shard_push_schedules` on non-pool plans
  (`StartTransferRequest.shard_push_schedules`). Pool reshard checks unique
  contiguous `transfer_rank`s in `BuildPoolReshardPlan` (`Source transfer_rank
  values must be unique`) and bypasses `ExpectedPushSenders`, and `KVCacheStore`
  startup checks dense `[0, N)` `node_id`s in
  `HostOffloadBackend::ComposeKVTransferSpec` when `expected_worker_count > 0`
  with a global registry (`WorkerRegistry::RegisterWorker` exempts
  `node_id == 0`, and store transfers bypass `RegisterActivePlan`). On
  legacy non-pool `RegisterActivePlan` plans (`KVCacheManagerWithTransfer` and
  its `KVCacheManager` bindings default `node_id` to `0`, while
  `KVCacheManagerBase::node_id()` defaults to `-1`, and
  `JobEntity.encode_start_transfer` in `tpu_sync/weight_sync/manager/job_entity.py`
  keys receiver `shard_push_schedules` per source shard when `num_src_shards > 1`),
  neither `KVCacheManagerBase` / `KVCacheManagerWithTransfer` nor
  `RegisterActivePlan` checks that pushing workers carry distinct `node_id`s
  matching one `shard_push_schedules` entry each.
* **A4 (honest senders) — `Cfg.honest`.** Every sender sends at most the
  streams it declared for each layer, for the life of the receive. Encoded as
  a guard on `stream`: a stream beyond the declaration is not enabled. This
  is the assumption the C++ does **not** enforce on the chunk-count contract,
  nor on any contract after the table entries are retired
  (`if (all_layers_called)` in `CompleteIncomingPush`); the traces at the end drop it.
* **A5 (single transfer generation / fresh `uuid`).** `init` (`Layer.entry =
  none` on every layer) models a fresh `uuid` with no prior
  `active_push_sessions_[uuid]` residue from an earlier unretired push session
  and no concurrent unlocked `BlockTransport::ForgetPushProgress(uuid)` still
  in flight from retiring an earlier plan or session on the same `uuid`.

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

`reachable_safe`: under A4 and `Cfg.WF` both contracts satisfy all three, for
any positive `numLayers`, `numSenders`, `numStreams`, `blocksPerStream`.
`Cfg.WF.chunkSingle` (one sender on the chunk-count contract) follows from the
C++ on the `StartRead` path and on plans with `shard_push_schedules`; for a
receive created by `KVCacheManagerWithTransfer::RegisterRecv` and for a
receiver plan with no `shard_push_schedules` it is an obligation on whoever
dispatches the pushes, which nothing in the C++ checks.

Without A4 (`Cfg.honest = false`) the traces at the end show what one extra
stream does: on the chunk-count contract before retirement the receive is
published with a layer never copied; on either contract after retirement the
layer is copied twice (the session's `in_flight_` still holds `done_` off
until both copies return, and a failing duplicate copy fails the receive).

Three transport unit tests replay on `step` (`Transport/Traces.lean`):
`BlockTransportTest.LayerCompletesOnlyAfterEveryDeclaredSender` (sender-declared,
either arrival order, then reuse after retirement),
`BlockTransportTest.SenderOverDeliveryIsRejected` (the `streams.landed >
streams.declared` rejection) and
`BlockTransportTest.UndeclaredUuidKeepsHeaderDeclaredCompletion` (the chunk-count
contract on two senders, `Cfg.WF.chunkSingle`'s excluded instance), alongside
the two-stream chunk-count retirement (`trace_session_two_streams_complete`).
Sibling push-progress tests that vary declarations per sender (outside A3),
call `Cancel` / `ForgetPushProgress`, or exercise the plan-declared pool or
Peregrine branches (`BlockTransportServerPushSessionTest.LifecycleAndPostCompletion`
drives `OnPeregrinePostComplete`) are documented alongside them.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `bts.h` | `BlockTransportServerPushSession` (`struct LayerProgress` .. `absl::flat_hash_map<int, LayerProgress> layer_progress_ ABSL_GUARDED_BY(mu_);`) | `:158-201` |
| `bt.h` | `BlockTransport` (`absl::Mutex server_sessions_mu_;` .. `active_push_sessions_ ABSL_GUARDED_BY(server_sessions_mu_);`) | `:310-313` |
| `bts.cc` | `ResolveIncomingPushSpec` | `:185-240` |
| `bt.cc` | `BlockTransport::GetOrCreatePushSession` | `:432-443` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` |
| `bt.cc` | `BlockTransport::DeletePushSession` | `:445-453` |
| `bt.cc` | `BlockTransport::OnRecvComplete` | `:485-497` |
| `bt.cc` | `BlockTransport::AsyncPush` (`if (config_.backend_for(op) == TransportBackend::kPeregrine)`) | `:530-543` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` |
| `bts.cc` | `BlockTransportServerPushSession::OnPeregrinePostComplete` | `:663-713` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`absl::Cleanup end_incoming_push`) | `:385-390` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`std::optional<size_t> expected_senders;` .. `expected_senders = delegate_->ExpectedPushSenders(uuid_);`) | `:742-754` |
| `bts.cc` | `BlockTransportServerPushSession::Cancel` | `:340` |
| `bts.cc` | `BlockTransportServerPushSession::ReleaseAdmittedPosts` | `:347-369` |
| `bts.cc` | `ValidateChunks` | `:275-310` |
| `bt.cc` | `BlockTransport::ResolveRecvBuffer` | `:474-483` |
| `bt.cc` | `BlockTransport::ForgetPushProgress` | `:1075-1089` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`transport->ForgetPushProgress(uuid);`) | `:1128` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` (`if (auto recv =` .. `transfer_details_[req_id] = recv->Details();`) | `:476-484` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`bool array_complete = false;` .. `array_complete = progress.completed_chunks == expected_chunks;`) | `:773-849` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`bool all_layers_called = true;` .. `done_ = true;`) | `:866-876` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`const uint32_t remote_id =` .. `static_cast<uint32_t>(header_layer_idx);`) | `:765-771` |
| `recv.h` | `TransferReceiveSession::Done` | `:104-107` |
| `recv.h` | `TransferReceiveSession::IsDraining` | `:121-124` |
| `recv.h` | `TransferReceiveSession::ResolveStagingSlot` | `:160-162` |
| `recv.cc` | `TransferReceiveSession::TryBeginOp` | `:456-467` |
| `recv.cc` | `TransferReceiveSession::AcquireStagingSlot` | `:680-709` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:390-405` |
| `recv.cc` | `TransferReceiveSession::Finish` | `:407-410` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:443-454` |
| `recv.cc` | `TransferReceiveSession::EndOp` | `:469-472` |
| `recv.cc` | `TransferReceiveSession::IsReadyToComplete` | `:481-487` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShardsReceivedLocked` | `:496-521` |
| `recv.cc` | `TransferReceiveSession::RecordBlockShards` | `:629-678` |
| `recv.cc` | `TransferReceiveSession::OnDataReceived` | `:721-740` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`dispatched_layers_.insert(layer_idx);`) | `:894` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:771-903` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`future.OnReady`) | `:839-888` |
| `block_transport_session_test.cc` | `MockSessionDelegate` | `:48-155` |
| `block_transport_session_test.cc` | `BlockTransportServerPushSessionTest.LifecycleAndPostCompletion` | `:157-195` |
| `block_transport_session_test.cc` | `BlockTransportServerPushSessionTest.Cancel` | `:197-216` |
| `block_transport_test.cc` | `MockDelegate` | `:218-420` |
| `block_transport_test.cc` | `MultiSenderReceiverDelegate` | `:464-483` |
| `block_transport_test.cc` | `BlockTransportTest.PushSplitBySourceNumaCompletesPlanDeclaredPool` | `:910-940` |
| `block_transport_test.cc` | `BlockTransportTest.LayerCompletesOnlyAfterEveryDeclaredSender` | `:1119-1165` |
| `block_transport_test.cc` | `BlockTransportTest.DeclaredSendersEachCompleteTheirOwnParallelism` | `:1167-1203` |
| `block_transport_test.cc` | `BlockTransportTest.SenderOverDeliveryIsRejected` | `:1205-1242` |
| `block_transport_test.cc` | `BlockTransportTest.SenderChangingDeclaredStreamCountIsRejected` | `:1244-1277` |
| `block_transport_test.cc` | `BlockTransportTest.UndeclaredUuidKeepsHeaderDeclaredCompletion` | `:1279-1313` |
| `block_transport_test.cc` | `BlockTransportTest.ForgetPushProgressAllowsUuidReuse` | `:1315-1347` |
| `block_transport_test.cc` | `BlockTransportTest.PoolProgressWaitsForEveryStreamOfEveryDeclaredPoolAndRetires` | `:1349-1388` |
| `block_transport_test.cc` | `BlockTransportTest.ForgetPushProgressResetsPartialPoolGeneration` | `:1390-1417` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.get_session =`) | `:307` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` |
| `base.cc` | `KVCacheManagerBase::ResolveBlockArrayHostLayer` | `:2139-2197` |
| `base.cc` | `KVCacheManagerBase::GetBlockArrayHostSpan` | `:2199-2268` |
| `base.cc` | `KVCacheManagerBase::GetSession` | `:3081-3087` |
| `base.cc` | `KVCacheManagerBase::GetTransferPlan` | `:3089-3099` |
| `base.cc` | `KVCacheManagerBase::OnLayerReceived` | `:3345-3354` |
| `base.cc` | `KVCacheManagerBase::OnBlockShardsReceived` | `:3323-3341` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`size_t host_layer_idx = layer_idx;` .. `base_host_ptr = host_buffer->data();`) | `:3498-3512` |
| `mgr.cc` | `KVCacheManagerWithTransfer::EmplaceRecvSessionLocked` | `:605-626` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterActivePlan` | `:628-707` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RegisterRecv` | `:709-731` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:733-775` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:828-874` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_descriptors)` | `:914-940` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:942-1074` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`if (auto recv =` .. `absl::DeadlineExceededError("Transfer session timed out"));`) | `:1088-1100` |
| `send.cc` | `TransferSendSession::SendNextLayer` (`self->base_->H2hWriteDirectAsync`) | `:706-734` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:3112-3120` |
| `tpu_sync/kv_cache/transfer_plan.cc` | `TransferPlan::ExpectedPushSenders` | `:273-279` |
| `tpu_sync/kv_cache/transfer_plan.cc` | `TransferPlan::ResolvePlanBlockChunks` (`if (sender_node_id != -1) {` .. `auto schedule_found_it = sender_node_id != -1`) | `:407-428` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`std::vector<absl::Status> arm_status` .. `for (std::thread& t : armers) t.join();`) | `:746-779` |
| `planner.cc` | `BuildPoolReshardPlan` (`std::map<int64_t, RaidenId> ranks;` .. `++expected_rank;`) | `:975-990` |
| `tpu_sync/core/controller/worker_registry.cc` | `WorkerRegistry::RegisterWorker` (`if (entry.node_id != 0)`) | `:63-73` |
| `tpu_sync/kv_cache/host_offload_backend.cc` | `HostOffloadBackend::ComposeKVTransferSpec` | `:1552-1595` |
| `tpu_sync/kv_cache/kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` (`absl::Status status = callbacks_.register_active_plan(` .. `(*resp->mutable_receiver_pool_addrs())[static_cast<int32_t>(l)] =`) | `:188-201` |
| `tpu_sync/rpc/raiden_service.proto` | `StartTransferRequest` (`map<int32, ShardPushScheduleProto> shard_push_schedules`) | `:198` |
<!-- source-map:end -->
-/

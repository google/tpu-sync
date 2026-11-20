import TpuSyncVerify.Transfer.PrefillDecode.Transport.Model
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Lemmas
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Traces

/-!
# Receiver-side stream accounting in `BlockTransport`

Models the per-uuid `BlockTransportServerPushSession::layer_progress_` table for
the **chunk-count** (`Cfg.declared = false`) and **sender-declared**
(`Cfg.declared = true`) completion contracts and the `TransferReceiveSession`
counters they feed, discharging `Receive.lean` **A1** (`OnceDispatched`) and
**A4** (`NetworkSound`). Pool-reshard `plan_declared` pushes are in `Reshard/`;
opt-in Peregrine (`OnPeregrinePostComplete`, `PostExplicitPush`) is in
`docs/transfer/prefill_decode.md` §Boundaries.

File aliases: `bt.{h,cc}` = `tpu_sync/transport/block_transport.{h,cc}`,
`bts.{h,cc}` = `tpu_sync/transport/block_transport_session.{h,cc}`,
`base.cc` = `tpu_sync/kv_cache/kv_cache_manager_base.cc`,
`block_transport_session_test.cc` = `tpu_sync/transport/block_transport_session_test.cc`,
`block_transport_test.cc` = `tpu_sync/transport/block_transport_test.cc`.

## State

| Field | C++ | Role |
|---|---|---|
| `Layer.entry` | `layer_progress_[l]` (`BlockTransportServerPushSession`) | `none` before first stream or on a fresh push session after retirement |
| `Progress.completedChunks` | `LayerProgress::completed_chunks` | chunk-count contract |
| `Progress.landed` | `SenderStreams::landed` per sender | sender-declared contract |
| `Progress.sendersComplete` | `LayerProgress::senders_complete` | sender-declared contract |
| `Progress.called` | `LayerProgress::on_layer_received_called` | per-layer once-latch |
| `Layer.sent` | ghost | streams accepted per sender for this layer |
| `Layer.dispatched` | ghost | `OnLayerReceived` → `ExecuteLayerH2d` calls for this layer |
| `Layer.pending` | subset of `h2d_futures_` | in-flight H2D copies for this layer |
| `blocks` | `blocks_received_per_shard_` (collapsed to one shard) | block units reported via `RecordBlockShards` |
| `completedLayers` | `num_completed_layers_` | H2D callbacks that ran OK |
| `life` | `Transfer.Lifecycle` | receive-session `in_flight_`, `draining_`, `done_`, `status_` |

Shards are collapsed into `blocks` against `numSenders * numStreams * blocksPerStream * numLayers`
because every shard shares the target `total_blocks_ * num_layers` in
`RecordBlockShardsReceivedLocked`. `ExecuteLayerH2d`'s `dispatched_layers_`
setter of `network_completed_` fires only once all layers are dispatched (`Receive.lean`).

## Events

| Event | C++ |
|---|---|
| `stream l s` | `HandleIncomingPush` → `GetOrCreatePushSession` → `StartStream`: `BeginIncomingPush`, `CompleteIncomingPush`, `OnLayerReceived`, `OnBlockShardsReceived`, `EndIncomingPush` (or `Finish(error)` + `EndOp` on rejection) |
| `h2dDone l ok` | `ExecuteLayerH2d` `future.OnReady` callback (`num_completed_layers_++` / `FinishLocked`, then `EndOp`) |
| `publish` | `CompleteReadWithDetails` calls `Finish()` on `!IsDraining() && IsReadyToComplete()` (`pending = 0`) |
| `cancel` | deadline sweep or external `Finish(error)` while `!draining` |

## Lock scopes

| Event | Scopes | Gap note |
|---|---|---|
| `stream l s` | `server_sessions_mu_`, manager `mu_`, push-session `mu_`, receive-session `mu_` | streams serialize `layer_progress_` on push-session `mu_`; receive-session gaps are covered in `Receive.lean` |
| `h2dDone l ok` | 2: callback `mu_`, then `EndOp` | `EndOp` tail (`Receive.lean`) |
| `publish` | `IsDraining()`, `IsReadyToComplete()`, `Finish()` | readiness is monotone; OK `Finish()` on draining is a no-op |
| `cancel` | 1: `Finish(error)` | guarded by `!draining` |

* **Retirement (`retire`).** Setting `done_ = true` on `all_layers_called` in
  `CompleteIncomingPush` causes `DeletePushSession` / `GetOrCreatePushSession` to
  seat a fresh `BlockTransportServerPushSession` (`entry := none`) for any
  post-retirement stream, while `honest = true` with `Cfg.WF.chunkSingle`
  ensures no stream arrives after `all_layers_called`.

## Assumptions

* **A1 (one lock).** `layer_progress_` and `on_layer_received_called` are mutated
  in one critical section of the push session's `mu_` in `CompleteIncomingPush`.
* **A2 (stable expectation).** `ExpectedPushSenders(uuid)` is latched on the
  layer's first stream (`Cfg.declared`). Holds whenever senders push after
  `RegisterActivePlan` returns (as on controller `COMMAND_START_TRANSFER` hops);
  always `nullopt` on `StartRead` and `RegisterRecv`.
* **A3 (uniform declaration, one node per schedule).** Every sender declares
  `numStreams` and carries a distinct `header.remote_id = node_id()` within the
  expected set (unchecked by `RegisterActivePlan` on legacy non-pool plans).
* **A4 (honest senders — `Cfg.honest`).** Each sender sends at most `numStreams`
  streams per layer over the receive's lifetime (not enforced by C++ on
  chunk-count or post-retirement; dropped in the dishonest-sender traces).
* **A5 (single transfer generation / fresh `uuid`).** `init` starts with
  `entry = none` and no concurrent unlocked `ForgetPushProgress(uuid)` runs from
  an earlier generation on the same `uuid`.

## Properties

Under `Cfg.honest = true` and `Cfg.WF` (`chunkSingle`: one sender on chunk-count,
enforced on `StartRead` and caller-obligated on `RegisterRecv`), every reachable
state satisfies `Safe` (`reachable_safe`):

* `OnceDispatched`: `ExecuteLayerH2d` runs at most once per layer (`Receive` A1).
* `NetworkSound`: `blocks ≥ target` implies every layer was dispatched (`Receive` A4).
* `CompleteOnSuccess`: `published` (`done ∧ statusOk`) implies every layer was dispatched once and `inFlight = 0`.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `bts.h` | `BlockTransportServerPushSession` (`struct LayerProgress` .. `absl::flat_hash_map<int, LayerProgress> layer_progress_ ABSL_GUARDED_BY(mu_);`) | `:158-201` |
| `bt.h` | `BlockTransport` (`absl::Mutex server_sessions_mu_;` .. `active_push_sessions_ ABSL_GUARDED_BY(server_sessions_mu_);`) | `:310-313` |
| `bts.cc` | `ResolveIncomingPushSpec` | `:185-240` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `bts.cc` | `ValidateChunks` | `:275-310` |
| `bts.cc` | `BlockTransportServerPushSession::Cancel` | `:340` |
| `bts.cc` | `BlockTransportServerPushSession::ReleaseAdmittedPosts` | `:347-369` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` |
| `bts.cc` | `BlockTransportServerPushSession::OnPeregrinePostComplete` | `:663-713` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` |
| `bt.cc` | `BlockTransport::HandleCustomRequest` | `:407-430` |
| `bt.cc` | `BlockTransport::GetOrCreatePushSession` | `:432-443` |
| `bt.cc` | `BlockTransport::DeletePushSession` | `:445-453` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` |
| `bt.cc` | `BlockTransport::ResolveRecvBuffer` | `:474-483` |
| `bt.cc` | `BlockTransport::OnRecvComplete` | `:485-497` |
| `bt.cc` | `BlockTransport::AsyncPush` | `:520-569` |
| `bt.cc` | `BlockTransport::BuildBlockRequests` (`const uint32_t remote_id =` .. `static_cast<uint32_t>(header_layer_idx);`) | `:765-771` |
| `bt.cc` | `BlockTransport::ForgetPushProgress` | `:1075-1089` |
| `base.cc` | `KVCacheManagerBase::GetSession` | `:3081-3087` |
| `base.cc` | `KVCacheManagerBase::GetTransferPlan` | `:3089-3099` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:3112-3120` |
| `base.cc` | `KVCacheManagerBase::OnBlockShardsReceived` | `:3323-3341` |
| `base.cc` | `KVCacheManagerBase::OnLayerReceived` | `:3345-3354` |
| `tpu_sync/kv_cache/transfer_plan.cc` | `TransferPlan::ExpectedPushSenders` | `:273-279` |
| `tpu_sync/kv_cache/transfer_plan.cc` | `TransferPlan::ResolvePlanBlockChunks` (`if (sender_node_id != -1) {` .. `auto schedule_found_it = sender_node_id != -1`) | `:407-428` |
| `tpu_sync/core/controller/worker_registry.cc` | `WorkerRegistry::RegisterWorker` (`if (entry.node_id != 0)`) | `:63-73` |
| `tpu_sync/kv_cache/host_offload_backend.cc` | `HostOffloadBackend::ComposeKVTransferSpec` | `:1552-1595` |
| `tpu_sync/kv_cache/kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` (`absl::Status status = callbacks_.register_active_plan(` .. `(*resp->mutable_receiver_pool_addrs())[static_cast<int32_t>(l)] =`) | `:188-201` |
| `tpu_sync/rpc/raiden_service.proto` | `StartTransferRequest` (`map<int32, ShardPushScheduleProto> shard_push_schedules`) | `:198` |
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
<!-- source-map:end -->
-/

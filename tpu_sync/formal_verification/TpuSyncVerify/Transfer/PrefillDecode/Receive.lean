import TpuSyncVerify.Transfer.PrefillDecode.Receive.Model
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Frame
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Traces

/-!
# Receive session

Consumer side of prefill-to-decode transfer: one `TransferReceiveSession` on a
device destination (`BlockLocation::kDevice`) and its manager poll/publication.
Layer data and the producer live in `Send.lean` and `Pipeline.lean`; bounded
`StagingLayerManager` leasing is in `LayerStaging.lean`; `sessions_` registration
is in `UuidTable.lean`; `BlockLocation::kHost` (`SettleHostLayer`) is in
`docs/transfer/prefill_decode.md` §Boundaries.

File aliases: `recv.{h,cc}` = `tpu_sync/kv_cache/transfer_receive_session.{h,cc}`,
`send_drain_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_send_drain_test.cc`,
`control_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc`,
`pool_reshard_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.

## State

Extends `Transfer.Lifecycle` (`Transfer/Session.lean`):

| Field | C++ | Role |
|---|---|---|
| `numLayers` | `base_->num_layers()` | fixed layer count |
| `issued` | `h2d_futures_.size()` / `dispatched_layers_.size()` (A1) | H2D copies handed to the device |
| `ready` | `IsReady()` futures in `h2d_futures_` | copies finished on the device |
| `completed` | `num_completed_layers_` | H2D callbacks that ran OK |
| `layersAccounted` | layers absorbed by `blocks_received_per_shard_` | layers whose blocks were reported |
| `netIssued` | `network_completed_` via `ExecuteLayerH2d` (`dispatched_layers_`) | every layer's copy has been issued |
| `published` | `done_recving_` / `failed_recving_` (`RetireSettledSessionLocked`) | outcome exposed by `poll_stats()` |
| `pushes` | ghost | open `BeginIncomingPush` .. `EndIncomingPush` leases |
| `pullPending` | `handshake_pending_` (`InitFromLoadPlan` .. `Start`) | `StartRead` pull handshake still holds its op |
| `pending` | ghost | `ExecuteLayerH2d` calls between their first and second `mu_` scope |
| `retired` | ghost | H2D callbacks that have run (OK or error) |

`network_completed_` has two setters (`networkCompleted := layersAccounted = numLayers ∨ netIssued`):
per-shard block accounting in `RecordBlockShardsReceivedLocked` (coarsened to
one `netAccount` per layer by A4) and `ExecuteLayerH2d`'s third `mu_` scope when
`dispatched_layers_` reaches `num_layers()` while neither `done_` nor `draining_`.

## Events

| Event | C++ |
|---|---|
| `pushBegin` | `BeginIncomingPush` → `GetSession(uuid)` + `TryBeginOp` (`StartStream`) |
| `pushEnd` | `EndIncomingPush` → `EndOp` (preceded by `cancel` when `!status.ok()`) |
| `pullReply ok` | `Start`'s `on_response` (`FinishLocked` on error while `!draining_`, clear `handshake_pending_`, `EndOp`) |
| `h2dBegin` | `OnLayerReceived` → `OnDataReceived` → `ExecuteLayerH2d` first `mu_` scope (`++in_flight_`) |
| `h2dIssue ok` | `ExecuteLayerH2d` `done_ \|\| draining_` re-check, `H2dSyncDispatch`, `h2d_futures_`, and `dispatched_layers_` |
| `h2dReady` | device completes an issued H2D copy (`future.IsReady()`) |
| `h2dDone ok` | `ExecuteLayerH2d`'s `future.OnReady` callback (`num_completed_layers_++` / `FinishLocked`, then `EndOp`) |
| `netAccount` | `OnBlockShardsReceived` → `RecordBlockShards` (`RecordBlockShardsReceivedLocked` + `RecordNetworkCompleteLocked`) |
| `pollReady` | `CompleteReadWithDetails` sees `!IsDraining() && IsReadyToComplete()` and calls `Finish()` |
| `cancel` | external `Finish(error)` (deadline sweep, `~KVCacheManagerWithTransfer`, `UnregisterActivePlan`, failed push) |
| `publish` | `RetireSettledSessionLocked` reports a `Done()` session under `SessionEntry::req_id` and erases it |

## Lock scopes

Mutable session fields are guarded by `TransferReceiveSession::mu_` (`cancelled_`
is an atomic mirror of `draining_`; `staging_layer_manager_` has its own mutex).

| Event | `mu_` scopes | Gap note |
|---|---|---|
| `pushBegin` | 1: `TryBeginOp` | preceded by `GetSession(uuid)` under manager `mu_`; retired or send sessions return `false` |
| `pushEnd` | 1: `EndOp` | failed pushes run `Finish(status)` (`cancel`) in an earlier scope |
| `pullReply` | 2: `on_response` under `mu_`, then `EndOp` | `EndOp` tail below; pre-`Start()` `StartRead` errors are `cancel` then `pullReply false` |
| `h2dBegin` | 1: first `ExecuteLayerH2d` scope | unlocked `LookupStagingSlot` precedes it under bounded staging (`LayerStaging.lean`) |
| `h2dIssue` | 1–3 in `ExecuteLayerH2d` | re-check, unlocked `H2dSyncDispatch` + `push_back`, then `dispatched_layers_` scope (see below) |
| `h2dReady` | 0 (device) | — |
| `h2dDone` | 2: callback `mu_`, then `EndOp` | `EndOp` tail below (`staging_layer_manager_.Release` runs in the gap) |
| `netAccount` | 1: `RecordBlockShards` | preceded by `GetSession` under manager `mu_` (empty `block_ids` returns early) |
| `pollReady` | `IsDraining()`, `IsReadyToComplete()`, `Finish()` | readiness is monotone; a concurrent `cancel` makes the OK `Finish()` a no-op |
| `cancel` | 1: `Finish(error)` | unguarded; records first error even after an OK settle (`FinishLocked`) |
| `publish` | manager `mu_` nesting `Done()`, `GetStatus()` | `Done()` is monotone; late external `Finish(error)` orders before or after `publish` |

* **`ExecuteLayerH2d` scopes (`h2dBegin` / `h2dIssue`).** During `H2dSyncDispatch`,
  A4 prevents `RecordBlockShards` on this layer and `completed < numLayers` keeps
  `IsReadyToComplete` false; a concurrent `Finish(error)` cannot settle while
  `h2dBegin`'s op is held. If `cancel` lands before the third scope skips setting
  `network_completed_`, no transition on a draining session reads that flag.
* **`EndOp` tails (`pullReply`, `h2dDone`).** Only `FinishLocked` and
  `EndRecvOpLocked` inspect `in_flight_`, so any step interleaved before `EndOp`
  reaches the same state as running the full event atomically.

## Assumptions

* **A1 (one dispatch per layer).** `CompleteIncomingPush` fires `OnLayerReceived`
  at most once per layer (`issued + pending < numLayers`). Discharged by
  `Transport.OnceDispatched` under honest senders (`Transport.lean`).
* **A2 (registration).** `total_blocks_ > 0`, seated in `sessions_`
  (`UuidTable.lean`), and `ReleaseStaging()` runs only in
  `~TransferReceiveSession` after any unshared candidate is dropped.
* **A3 (no double end).** Each `in_flight_` op ends at most once (`h2dDone`
  requires `retired < ready`, `pushEnd` requires `0 < pushes`).
* **A4 (transport ordering).** On the stream completing a layer,
  `ExecuteLayerH2d` pushes into `h2d_futures_` before `RecordBlockShards` runs on
  the same thread (`layersAccounted < issued`); multi-stream coarsening is
  discharged by `Transport.NetworkSound` (mutant: `netAccountUnordered`).

## Properties

All proved on every reachable state (`reachable_safe`), with finite drain to
settlement (`reachable_can_settle`):

* `SettleSafe`: `done → inFlight = 0` — settled sessions own no in-flight work.
* `NoRetiredCallback`: `done → retired = issued` — no H2D callback fires after settle.
* `StagingIntegrity`: `hasStaging = !done` — staging is held iff unsettled.
* `SettlesPromptly`: `draining → inFlight = 0 → done`.
* `ReadinessSound`: `IsReadyToComplete → ready = numLayers`.
* `Publication`: `published = some true → completed = numLayers`.
* `CountersOrdered`: `completed ≤ retired ≤ ready ≤ issued ≤ numLayers ∧ layersAccounted ≤ issued`.
* `PublishedQuiet`: `published ≠ none → pending = 0 ∧ retired = issued` — decode HBM is quiet once published.
* `NoOpLeak`: `0 < inFlight →` some `drainEvents` step is enabled.

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
| `recv.h` | `TransferReceiveSession::tracks_completion` | `:168-171` |
| `recv.h` | `TransferReceiveSession::deadline` | `:176-179` |
| `recv.h` | `TransferReceiveSession` (`mutable absl::Mutex mu_;` .. `std::atomic<bool> cancelled_{false};`) | `:309-362` |
| `recv.cc` | `TransferReceiveSession::Create(remote_block_ids)` | `:81-123` |
| `recv.cc` | `TransferReceiveSession::Create(total_blocks)` | `:125-137` |
| `recv.cc` | `TransferReceiveSession::InitFromActivePlan` | `:191-243` |
| `recv.cc` | `TransferReceiveSession::InitFromLoadPlan` | `:366-378` |
| `recv.cc` | `TransferReceiveSession::ReleaseStaging` | `:385-388` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:390-405` |
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
| `recv.cc` | `TransferReceiveSession::RecordBlockShards` | `:629-678` |
| `recv.cc` | `TransferReceiveSession::OnDataReceived` | `:721-740` |
| `recv.cc` | `TransferReceiveSession::SettleHostLayer` | `:742-769` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:771-903` |
| `send_drain_test.cc` | `RecvLifecycleTest.NetworkCompletionWaitsForH2d` | `:840-858` |
| `send_drain_test.cc` | `RecvLifecycleTest.EmptyBlockReportIssuesNoH2d` | `:860-878` |
| `send_drain_test.cc` | `RecvLifecycleTest.DetailsReportWhenTheLastH2dLanded` | `:880-895` |
| `send_drain_test.cc` | `RecvLifecycleTest.DetailsSayAReceiveTimedOutWaitingForData` | `:897-910` |
| `send_drain_test.cc` | `RecvLifecycleTest.DetailsSayAFailedUploadFailedInH2dWhileDataArrives` | `:912-926` |
| `send_drain_test.cc` | `RecvLifecycleTest.LateBlockAccountingAfterRetirementIsANoOp` | `:941-963` |
| `send_drain_test.cc` | `RecvLifecycleTest.OutOfOrderLayersSettleAfterEveryH2d` | `:965-984` |
| `send_drain_test.cc` | `RecvLifecycleTest.SingleFailedH2dReportsFailureAndReturnsStaging` | `:986-997` |
| `send_drain_test.cc` | `RecvLifecycleTest.ReceiveWithoutTrafficFailsAtItsDeadline` | `:999-1010` |
| `send_drain_test.cc` | `RecvDrainTest.ExpiredReceiveKeepsStagingUntilH2dEnds` | `:1012-1039` |
| `send_drain_test.cc` | `RecvDrainTest.FailedLayerWaitsForOtherH2dCopies` | `:1149-1175` |
| `send_drain_test.cc` | `RecvDrainTest.TimeoutDuringH2dDispatchKeepsStaging` | `:1177-1213` |
| `send_drain_test.cc` | `RecvLifecycleTest.SessionRemainsValidAcrossAsyncH2dCallbacksAfterCallerDropsPtr` | `:1237-1264` |
| `send_drain_test.cc` | `RecvLifecycleTest.IncomingPushLeasePinsStagingDuringWriteAndRejectsWhenDraining` | `:1266-1290` |
| `send_drain_test.cc` | `RecvLifecycleTest.FirstAdmittedPushRecordsFirstPacketOnce` | `:1292-1323` |
| `send_drain_test.cc` | `RecvLifecycleTest.AdmissionRejectedByDoneSessionRecordsNoFirstPacket` | `:1325-1357` |
| `send_drain_test.cc` | `RecvLifecycleTest.AdmissionRejectedByDrainingSessionRecordsNoFirstPacket` | `:1359-1400` |
| `send_drain_test.cc` | `RecvLifecycleTest.PeregrineLayerCompletionsRecordLastPacketOnceAndFinishAfterH2d` | `:1419-1459` |
| `send_drain_test.cc` | `RecvLifecycleTest.TcpLastLayerThenItsBlockReportRecordLastPacketOnceAndFinishOnce` | `:1461-1507` |
| `send_drain_test.cc` | `RecvLifecycleTest.TcpH2dFinishingBeforeLastBlockReportRecordsLastPacketAndFinishesOnce` | `:1509-1553` |
| `send_drain_test.cc` | `RecvLifecycleTest.RedeliveredLayerDoesNotCompleteNetworkPhase` | `:1555-1591` |
| `send_drain_test.cc` | `RecvLifecycleTest.IncomingPushLeaseSpansLayerH2dAndBlockAccountingBeforeReleasing` | `:1593-1621` |
| `send_drain_test.cc` | `RecvLifecycleTest.FailedIncomingPushImmediatelyFailsSessionAndReleasesStagingBeforeDeadline` | `:1623-1652` |
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

import TpuSyncVerify.Transfer.PrefillDecode.Send.Model
import TpuSyncVerify.Transfer.PrefillDecode.Send.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Send.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Send.Frame
import TpuSyncVerify.Transfer.PrefillDecode.Send.Traces

/-!
# Send session

Producer side of prefill-to-decode transfer: one `TransferSendSession`
(`BlockLocation::kDevice`) staging layers via D2H (`StartPush`) and chaining H2H
pushes (`SendNextLayer`) before releasing prefill HBM or host staging. Memory
contents live in `Pipeline.lean`, bounded layer slots in `LayerStaging.lean`,
and `BlockLocation::kHost` (`trace_host_source_prefix`) plus plan-registered
`CreateFromActivePlan` sends in `docs/transfer/prefill_decode.md` §Boundaries.

File aliases: `send.{h,cc}` = `tpu_sync/kv_cache/transfer_send_session.{h,cc}`,
`mgr.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`,
`bts.cc` = `tpu_sync/transport/block_transport_session.cc`.

## State

Extends `Transfer.Lifecycle` (`Transfer/Session.lean`):

| Field | C++ | Role |
|---|---|---|
| `numLayers` | `base_->num_layers()` | fixed layer count |
| `pullStarted` | `pull_started_` | `ValidateAndBeginPull` claimed the offer |
| `started` | `StartPush` passed its locked section | staging acquired and `d2h_layer_futures_` populated |
| `d2hPending` | ghost | D2H loop between `++in_flight_` and `D2hSyncDispatch` |
| `d2hIssued` | `next_layer` in `StartPush` | D2H copies handed to the device |
| `d2hReady` | `d2h_promises[l]` resolved (`IsReady()`) | D2H copies finished |
| `d2hRetired` | ghost | `layer_future.OnReady` callbacks that ended their op |
| `queued` | ghost | `SendNextLayer(l)` calls that took `++in_flight_` |
| `woken` | ghost | `SendNextLayer` `layer_future->OnReady` callbacks that ran |
| `pooled` | ghost | tasks scheduled on `push_pool()` not yet started |
| `h2hIssued` | ghost | pushes handed to `H2hWriteDirectAsync` |
| `chaining` | ghost | pool tasks between `H2hWriteDirectAsync` and `SendNextLayer(l + 1)` |
| `h2hRetired` | ghost | `H2hWriteDirectAsync` callbacks that ran |
| `h2hOk` | `num_layers - remaining_h2h_layers_` | pushes completed OK |
| `published` | `done_sending_` / `failed_recving_` (`RetireSettledSessionLocked`) | outcome exposed by `poll_stats()` |
| `underflow` | ghost | `EndSendOpLocked` ran with `in_flight_ == 0` |

Ghost counters partition `in_flight_` (`Accounted`); `underflow` records an
unguarded `--in_flight_` at zero instead of saturating so `NoUnderflow` is proved.

## Events

| Event | C++ |
|---|---|
| `beginPull` | `ValidateAndBeginPull`: checks `!draining` / `!done` and `!pull_started_`, sets `pull_started_ = true` |
| `start` | `StartPush` (`AcquireStagingWithRetry` + locked future setup + `SendNextLayer(0)`, or OK finish at `0` layers) |
| `d2hBegin` | `StartPush` D2H loop iteration up to `if (draining_) return; ++in_flight_;` |
| `d2hIssue ok` | `D2hSyncDispatch` (`!ok` calls `FinishLocked` + `EndOp` and triggers `fail_unissued_layers`) |
| `d2hReady` | `layer_future.OnReady` resolves `d2h_promises[l]` (also read by `GetDataReadyFuture`) |
| `d2hEnd` | `layer_future.OnReady` in `StartPush` calls `EndOp` |
| `wake ok` | `SendNextLayer` `layer_future->OnReady` (`!ok` finishes failed; non-draining schedules on `push_pool()`) |
| `h2hIssue` | `push_pool` task: drops if `draining_`, else takes `++in_flight_` and calls `H2hWriteDirectAsync` |
| `sendNext` | `push_pool` task calls `SendNextLayer(l + 1)` then releases the carried-over op (`end_op`) |
| `h2hDone ok` | `H2hWriteDirectAsync` callback (`!ok` fails; last OK push calls `Finish()`; ends op) |
| `cancel` | external `Finish(error)` (deadline sweep, shutdown, spawn/staging/argument failure; first `Finish` wins) |
| `publish` | `RetireSettledSessionLocked` reports a `Done()` send into `done_sending_` / `failed_recving_` |

## Lock scopes

Mutable fields are guarded by `TransferSendSession::mu_` (`remaining_h2h_layers_`
and `cancelled_` are atomics; `staging_layer_manager_` has its own mutex).

| Event | `mu_` scopes | Gap note |
|---|---|---|
| `beginPull` | 1: `ValidateAndBeginPull` | `deadline_` is `const` |
| `start` | `AcquireStagingWithRetry` + `StartPush` setup + `SendNextLayer(0)` | `Finish` before or in the A6 gap matches `cancel` before `start` |
| `d2hBegin` | 1: loop `session_lock` (`++in_flight_`) | preceded by `AcquireStagingSlot` (`LayerStaging.lean`); early exit runs `fail_unissued_layers` |
| `d2hIssue` | 0 on OK; `FinishLocked` + `EndOp` on error | `d2hBegin`'s op blocks settle during `D2hSyncDispatch` |
| `d2hReady` | 0 (`d2h_promises[l].Set`) | — |
| `d2hEnd` | 1: `EndOp` | — |
| `wake` | 1 on schedule; 2 (`FinishLocked`/check + `EndOp`) otherwise | `Finish` before the pool task runs is caught by `h2hIssue`'s `draining_` check |
| `h2hIssue` | 2 when draining; 1 (`++in_flight_`) before `H2hWriteDirectAsync` | `chaining` admits inline/early `h2hDone` before `sendNext` |
| `sendNext` | `SendNextLayer(l + 1)` scope (if `l + 1 < num_layers`) + `EndOp` | `EndOp` tail below |
| `h2hDone` | atomic `fetch_sub`, `Finish` (last OK or error), then `EndOp` | `finishOnceLocked` makes concurrent `cancel` commute with `fetch_sub` |
| `cancel` | 1: `Finish(error)` | ignored once `draining_ \|\| done_` (`finishOnceLocked`) |
| `publish` | manager `mu_` nesting `Done()`, `GetStatus()`, `Details()` | status is frozen once draining or done |

* **`EndOp` tails (`d2hIssue false`, `wake`, `h2hIssue`, `sendNext`, `h2hDone`).**
  Only `FinishLocked` and `EndSendOpLocked` inspect `in_flight_`, so steps in the
  pre-`EndOp` gap reach the same state as running the event atomically.

## Assumptions

* **A1 (at-most-once `StartPush`).** `ValidateAndBeginPull` (`beginPull`) sets `pull_started_ = true` and gates `start` (`PullClaimed`).
* **A2 (sequential D2H loop).** One `StartPush` thread issues copies sequentially (`d2hPending : Bool`).
* **A3 (layers interchangeable).** `Send.lean` counts copies/pushes without layer indices; `Pipeline.lean` indexes every layer's readiness and memory cell.
* **A4 (no double end).** Each op ends once, guarded by the ghost counter that opened it.
* **A5 (no consumer Ack).** `HandleAck → AckSend → Finish()` has only test callers and is excluded.
* **A6 (`SendNextLayer(0)` folded into `start`).** Only `BuildCoalescedCopySpec` and `remaining_h2h_layers_.store` run between `StartPush`'s setup lock and `SendNextLayer(0)` at `in_flight_ == 0`, so an intervening `Finish` matches `cancel` before `start`.
* **A7 (`per_transfer` connection-release window).** Under `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer`, the last `EndOp` drops `mu_` for `ReleaseDataConnectionsAfterSend` while `in_flight_ == 1 && draining_ && !done_`, when all other steps are disabled.
* **A8 (no incoming push on a send).** `TransferSendSession::TryBeginOp` returns `false` and `OnDataReceived` returns `FailedPreconditionError`.
* **A9 (plan-registered sends excluded).** `CreateFromActivePlan` sends (`RegisterActivePlan` `is_sender`, erased unreported by `UnregisterActivePlan`) never run `StartPush`; see `docs/transfer/prefill_decode.md` §Boundaries.

## Properties

All proved on every reachable state (`reachable_safe`), with finite drain (`reachable_can_settle`):

* `SettleSafe`: `done → inFlight = 0`.
* `Drained`: `done →` no D2H copy, `SendNextLayer` waiter, pool task, or H2H push is outstanding.
* `StagingIntegrity`: `hasStaging = !done`.
* `SettlesPromptly`: `draining → inFlight = 0 → done`.
* `NoUnderflow`: `underflow = false` (`EndSendOpLocked` never runs at `in_flight_ == 0`).
* `Publication`: `published = some true → h2hOk = numLayers`.
* `CountersOrdered`: `d2hRetired ≤ d2hReady ≤ d2hIssued ≤ numLayers`, `woken ≤ queued ≤ numLayers`, `pooled + h2hIssued ≤ woken`, `h2hOk ≤ h2hRetired ≤ h2hIssued`.
* `PullClaimed`: `(started → pullStarted) ∧ ((d2hPending ∨ 0 < d2hIssued) → started)`.
* `PublishedQuiet`: `published ≠ none → d2hPending = false ∧ d2hRetired = d2hIssued` — prefill HBM is quiet once published.
* `NoOpLeak`: `0 < inFlight →` some `drainEvents` step is enabled.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `send.h` | `TransferSendSession::Done` | `:87-90` |
| `send.h` | `TransferSendSession::Start` | `:94` |
| `send.h` | `TransferSendSession::TryBeginOp` | `:98` |
| `send.h` | `TransferSendSession::GetStatus` | `:106-109` |
| `send.h` | `TransferSendSession::IsDraining` | `:117-120` |
| `send.h` | `TransferSendSession::deadline` | `:167-169` |
| `send.h` | `TransferSendSession` (`mutable absl::Mutex mu_;` .. `const BlockLocation source_;`) | `:222-232` |
| `send.h` | `TransferSendSession` (`bool pull_started_` .. `StagingLayerManager staging_layer_manager_;`) | `:239-258` |
| `send.cc` | `*` (`constexpr absl::string_view kAwaitPullPhase` .. `constexpr absl::string_view kPushPhase`) | `:58-61` |
| `send.cc` | `TransferSendSession::Create` | `:65-91` |
| `send.cc` | `TransferSendSession::CreateFromActivePlan` | `:93-144` |
| `send.cc` | `TransferSendSession::ValidateAndBeginPull` | `:200-214` |
| `send.cc` | `TransferSendSession::GetDataReadyFuture` | `:216-223` |
| `send.cc` | `TransferSendSession::FinishLocked` | `:288-303` |
| `send.cc` | `TransferSendSession::Finish` | `:305-314` |
| `send.cc` | `TransferSendSession::OnDataReceived` | `:322-328` |
| `send.cc` | `TransferSendSession::Details` | `:330-350` |
| `send.cc` | `TransferSendSession::CurrentPhaseLocked` | `:352-362` |
| `send.cc` | `TransferSendSession::EndSendOpLocked` | `:364-371` |
| `send.cc` | `TransferSendSession::EndOp` | `:373-400` |
| `send.cc` | `TransferSendSession::AcquireStagingWithRetry` | `:433-467` |
| `send.cc` | `TransferSendSession::StartPush` | `:469-640` |
| `send.cc` | `TransferSendSession::SendNextLayer` | `:642-741` |
| `bts.cc` | `BlockTransportServerPullSession::TriggerNextSendStep` | `:1055-1151` |
| `mgr.h` | `*` (`enum class BlockLocation { kDevice, kHost };`) | `:99` |
| `mgr.cc` | `KVCacheManagerWithTransfer::HandleAck` | `:1360-1363` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CheckHostMemoryTransferSupported` | `:1402-1414` |
| `mgr.cc` | `KVCacheManagerWithTransfer::AckSend` | `:1459-1480` |
| `send_drain_test.cc` | `SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds` | `:518-542` |
| `send_drain_test.cc` | `SendDrainTest.FailedLayerWaitsForTheOtherLayersCopies` | `:544-562` |
| `send_drain_test.cc` | `SendDrainTest.BlockReadinessWaitsForTheSendsLayerCopy` | `:567-608` |
| `send_drain_test.cc` | `SendDrainTest.SendNobodyPulledFailsAtItsDeadline` | `:610-619` |
| `send_drain_test.cc` | `SendLifecycleTest.FailedSendWithoutWorkSettlesImmediately` | `:676-686` |
| `send_drain_test.cc` | `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess` | `:708-720` |
| `send_drain_test.cc` | `SendLifecycleTest.SuccessCannotOverrideAnEarlierFailure` | `:722-734` |
| `send_drain_test.cc` | `SendLifecycleTest.DetailsSayAnUnpulledSendTimedOutAwaitingItsPull` | `:754-775` |
| `send_drain_test.cc` | `SendDrainTest.DetailsSayAnExpiredSendWasStillCopying` | `:777-794` |
| `send_drain_test.cc` | `SendDrainTest.DetailsCarryTheErrorOfAFailedCopy` | `:796-809` |
| `send_drain_test.cc` | `SendDrainTest.DetailsSayAFailedPushFailedInPushWhileCopiesRun` | `:811-838` |
| `pool_reshard_test.cc` | `SendDeadlineTest.ExpiredSendSessionFailsInsteadOfReportingDone` | `:464-473` |
| `control_test.cc` | `ControlHandshakeTest.PullAfterRegistrationDeadlineIsRejected` | `:323-345` |
| `control_test.cc` | `ControlHandshakeTest.DuplicatePullIsRejectedBeforeAcknowledgement` | `:382-394` |
| `control_test.cc` | `ControlHandshakeTest.StartReadOverPeregrinePropagatesConsumerPortAndCompletesPush` | `:1289-1374` |
| `transfer_send_session_test.cc` | `FakeSendBase::CompleteH2h` | `:99-106` |
| `transfer_send_session_test.cc` | `CreateHostSendSession` | `:237-246` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeDone` | `:256-279` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnections` | `:281-301` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush` | `:303-335` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush` | `:337-363` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CreateRejectsUnknownDataConnReuseValue` | `:365-380` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.SendSessionImplementsTransferSessionInterface` | `:382-414` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator` | `:447-490` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.FinishBeforeStartPushDoesNotAcquireStagingOrAccessHbm` | `:492-518` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.DoneGuaranteesAllResourcesReleasedAndNoHbmOrTransportAccessAfterDone` | `:520-573` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.ZeroLayerSessionCompletesAndReleasesStagingWithoutHang` | `:612-630` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone` | `:652-682` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.D2hDispatchFailureFailsUnissuedLayersAndDrainsIssuedCopies` | `:717-755` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenFailedDispatchEndsSend` | `:761-815` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.FailedD2hDispatchOnFirstLayerSettlesAtOnce` | `:821-841` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CancelledSendWhosePushThenFailsKeepsCancelled` | `:847-869` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.HostSourcePushesRegisteredHostBlocksWithoutStagingOrD2h` | `:871-908` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.HostSourceFailedLayerPushFailsTheSendOnceInFlightLayersDrain` | `:910-932` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.HostSourceDeadlineWhilePushingIsAttributedToThePushPhase` | `:934-953` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.HostSourceWithNoLayersSettlesWithoutAPush` | `:955-968` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.HostSourceFinishedBeforeStartPushIssuesNothing` | `:970-984` |
| `kv_cache_manager_with_transfer_host_memory_test.cc` | `HostMemoryTransferTest.AReadNobodyPullsExpiresAwaitingThePull` | `:373-381` |
| `kv_cache_manager_with_transfer_host_memory_test.cc` | `HostMemoryTransferTest.AFailedPushFailsTheSendAndExpiresTheReceive` | `:400-423` |
<!-- source-map:end -->
-/

import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.Send.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Send.Frame

/-! # `Send`: replay of C++ tests, bounded searches, failure phase

Part of `TpuSyncVerify.Transfer.PrefillDecode.Send`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Replay and bounded search

Concrete traces, checked by `decide`, that document the behaviours the model
admits; and a bounded search confirming that no `Safe` violation is reachable
within a few events. The inductive proof above is the actual guarantee; the
search guards against a modelling slip making the proof vacuous.

Harness mapping for the replayed tests: a session test's
`ValidateAndBeginPull; StartPush` is
`beginPull, start`, and so is `TestManager::ServePull`, which calls
`StartPush` without `ValidateAndBeginPull` (the model's `start` needs the
claim, so `beginPull` is supplied; `pull_started_` stays false in those
tests); `start` includes the `SendNextLayer(0)` that `StartPush` arms before
its D2H loop. A D2H copy finishing (`FinishCopy` / `CompleteD2h`, or at
dispatch for `FakeSendBase` without `SetManualD2h`) runs the copy's `OnReady`
in `StartPush`, which first resolves the layer's promise (`d2hReady`) — and
the promise runs its waiter inline, so a `SendNextLayer(l)` already waiting
runs then (`wake ok`) — and then ends the op (`d2hEnd`); hence
`d2hReady, wake, d2hEnd` for layer 0, always armed first, and for any layer the
chain reached before its copy finished. A layer whose copy finished before
the chain reached it runs `wake` inline when `SendNextLayer(l)` registers
(`d2hReady, d2hEnd, …, sendNext, wake`). A push completing (`CompleteH2h`, or
`FakeSendBase`'s synchronous `H2hWriteDirectAsync`) is `h2hDone ok`;
`Finish(error)` from the deadline sweep, `Decide(failed)` or a direct call is
`cancel`, and so is a `StartPush` on a finished send
(`AcquireStagingWithRetry` returns `status_`, `StartPush` calls `Finish` with
it, `FinishLocked` ignores it); `CompleteReadRaw` / `CompleteReadWithDetails`
is `publish` once settled. A host-source send (`BlockLocation::kHost`,
`CreateHostSendSession` in `transfer_send_session_test.cc`, `NotifyForRead(..., kHost)`
in `kv_cache_manager_with_transfer_host_memory_test.cc`) seats an already-OK
future per layer in `StartPush` instead of staging and copying; its `StartPush`
is replayed as `start` followed by the completed-copy prefix
`(d2hBegin, d2hIssue true, d2hReady, d2hEnd)` per layer (`tr_host_source_prefix`,
`Send.lean` §"Host-resident source"), after which every layer's `wake` runs as
soon as its `SendNextLayer(l)` registers. -/

/-- Two layers, pushes succeed, published `done_sending`. -/
def tr_normal : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext,
   .h2hDone true, .h2hDone true, .publish]

/-- A two-layer send that completes and is published as `done_sending`. Note
the push chain: layer 1's push is only issued after layer 0's, and
`sendNext` after the last layer takes no op. -/
theorem trace_normal :
    ((sys 2).run tr_normal).map
      (fun s => (s.life.done, s.published, s.h2hOk, s.life.inFlight)) =
      some (true, some true, 2, 0) := by
  decide

/-- One layer, no manager: the copy completes at dispatch (`FakeSendBase`
without `SetManualD2h`), so `wake` runs inside `StartPush`'s loop iteration
and the push pool task runs concurrently with its tail; the push completes
before the pool task's tail (`FakeSendBase`'s synchronous `H2hWriteDirectAsync`). -/
def tr_one_layer_sync_push : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .h2hDone true, .sendNext]

/-- One layer, no manager, the copy completed by the test (`SetManualD2h`),
the push synchronous. -/
def tr_one_layer_manual_copy : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .h2hDone true, .sendNext]

/-- One layer, no manager, copy and push both completed by the test
(`SetManualD2h`, `SetManualH2h`): the task tail runs before `CompleteH2h`. -/
def tr_one_layer_async_push : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .sendNext, .h2hDone true]

/-- `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeDone`,
`TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnections`,
`TransferSendSessionTest.FinishingOneSendSessionDoesNotShutdownSharedStagingAllocator`
(each of its two sessions), `TransferSendSessionTest.ReleaseStagingBlocksClearsD2hLayerFuturesOnDone`
and `TransferSendSessionTest.AsyncCallbacksKeepSessionAliveAfterCallerDropsSharedPtr`:
a one-layer send whose only push succeeds settles OK with its staging
released, whichever of the push callback and the task tail runs first, and
also if the pool task's tail ran before the copy's own `EndOp`. -/
theorem trace_one_layer :
    ((sys 1).run tr_one_layer_sync_push).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) ∧
    ((sys 1).run tr_one_layer_manual_copy).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) ∧
    ((sys 1).run tr_one_layer_async_push).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) ∧
    ((sys 1).run [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true,
      .h2hIssue, .sendNext, .d2hEnd, .h2hDone true]).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) := by
  decide

/-- The deadline sweep finishes an idle send; the poll publishes it. -/
def tr_idle_cancel : List Ev := [.cancel, .publish]

/-- "One nobody pulled is reported now" (`CompleteReadWithDetails`;
`SendDrainTest.SendNobodyPulledFailsAtItsDeadline` and
`SendDeadlineTest.ExpiredSendSessionFailsInsteadOfReportingDone`): before the
deadline an unpulled send holds its `Lifecycle` staging token (`hasStaging = true`
by the `Session.lean` convention) and cannot be published; once the deadline
fires on the idle send it settles at once (`hasStaging = false`), publishes
failure (`some false`), and the model also checks that any late `.beginPull`
(`ControlHandshakeTest.PullAfterRegistrationDeadlineIsRejected`) or `.start` is
rejected. The `[.cancel, .publish]` conjunct is also
`SendLifecycleTest.FailedSendWithoutWorkSettlesImmediately`
(a synthetic send with `in_flight = 0` failed via `Decide(failed = true)` settles,
releases staging and is reported in `failed_recving` at once),
`SendLifecycleTest.DuplicateRegistrationCannotReplaceLiveOffer` (whose
duplicate `NotifyForRead` is `UuidTable`'s business) and
`SendLifecycleTest.DetailsSayAnUnpulledSendTimedOutAwaitingItsPull`. -/
theorem trace_never_pulled :
    ((sys 2).run []).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (false, true, none) ∧
    (sys 2).run [.publish] = none ∧
    (sys 2).run [.start] = none ∧
    ((sys 2).run tr_idle_cancel).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) ∧
    (sys 2).run [.cancel, .beginPull] = none ∧
    (sys 2).run [.cancel, .start] = none := by
  decide

/-- `ValidateAndBeginPull` alone. -/
def tr_pull : List Ev := [.beginPull]

/-- `ControlHandshakeTest.DuplicatePullIsRejectedBeforeAcknowledgement`: once
`ValidateAndBeginPull` (`.beginPull`) has claimed `pull_started_ = true`, any
duplicate pull is rejected both before and after `StartPush` (`.start`) begins
(the C++ test sets `pull_started_` through the `MarkPullStarted` hook and checks
only the pre-`StartPush` duplicate; the post-`StartPush` conjunct is
model-only). -/
theorem trace_duplicate_pull_rejected :
    ((sys 1).run tr_pull).map (fun s => (s.pullStarted, s.started)) = some (true, false) ∧
    (sys 1).run [.beginPull, .beginPull] = none ∧
    (sys 1).run [.beginPull, .start, .beginPull] = none := by
  decide

/-- `Finish(error)` between the pull claim and `StartPush`'s locked section. -/
def tr_cancel_before_start : List Ev := [.beginPull, .cancel]

/-- The same, followed by the late `StartPush`, whose refused acquisition is a
`Finish(error)` that `FinishLocked` ignores. -/
def tr_cancel_then_late_start : List Ev := [.beginPull, .cancel, .cancel]

/-- `HandlePullStream` worker spawn failure (`hooks::kKvCacheManagerPullSpawn`):
if `ValidateAndBeginPull` (`.beginPull`) succeeds but spawning the `StartPush`
thread fails, `counter_cleanup` calls `session->Finish(InternalError)`
(`.cancel`), which settles the session at once, releases staging, reports
failure, and leaves nothing for a late `StartPush` to do: the model refuses
`.start` (the post-acquisition `if (draining_ || done_) return;`), and in the
C++ the acquisition itself is refused and its `Finish(status_)` is ignored —
a second `.cancel` with no effect. The same shape, by other causes:
`TransferSendSessionTest.FinishBeforeStartPushDoesNotAcquireStagingOrAccessHbm`
(`Finish` before `StartPush` runs, then `StartPush`),
`TransferSendSessionTest.CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator`
(`Finish` while `AcquireStagingWithRetry` waits; `start` is atomic acquisition,
so the wait is the pre-`start` state, and the woken loop's refusal is the late
`cancel`) and
`TransferSendSessionTest.InvalidStartPushArgumentsFailSessionImmediatelyWithoutHang`
(`StartPush` rejects empty endpoints with `Finish(InvalidArgument)` before
acquiring anything). -/
theorem trace_pull_spawn_failure_cleanup :
    ((sys 1).run (tr_cancel_before_start ++ [.publish])).map
      (fun s => (s.pullStarted, s.started, s.life.done, s.life.hasStaging, s.published)) =
      some (true, false, true, false, some false) ∧
    (sys 1).run (tr_cancel_before_start ++ [.start]) = none ∧
    ((sys 1).run tr_cancel_then_late_start).map
      (fun s => (s.started, s.life.done, s.life.hasStaging)) = some (false, true, false) := by
  decide

/-- Two layers; the deadline fires after the first copy was issued. -/
def tr_deadline_during_copy : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .cancel, .d2hReady, .wake true, .d2hEnd, .publish]

/-- The deadline fires while a copy runs: the send drains, keeps its staging,
issues nothing more, and settles once the copy's `OnReady` and `SendNextLayer(0)`'s
callback have ended both ops. The one-layer form is `tr_expired_while_copying`
below. -/
theorem trace_deadline_during_copy :
    ((sys 2).run (tr_deadline_during_copy.take 5)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sys 2).run (tr_deadline_during_copy.take 5 ++ [.d2hBegin]) = none ∧
    ((sys 2).run tr_deadline_during_copy).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- A push fails with the next layer already queued. -/
def tr_push_fails : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .d2hEnd, .wake true, .h2hIssue, .sendNext,
   .d2hReady, .d2hEnd, .wake true, .h2hDone false, .h2hIssue, .publish]

/-- A push fails: the session drains, the layer still waiting in the chain is
dropped by `h2hIssue`'s re-check, and the send is published as failed. No C++
test exercises this "both copies done, next layer queued" shape:
`FakeSendBase::CompleteH2h(i, error)`
accepts an error status but every caller passes success; the one test that does
fail a push, `SendDrainTest.DetailsSayAFailedPushFailedInPushWhileCopiesRun`,
is the "copy still running" shape replayed by
`trace_details_failed_push_while_copies_run`. -/
theorem trace_push_fails :
    ((sys 2).run tr_push_fails).map
      (fun s => (s.life.done, s.h2hIssued, s.published)) = some (true, 1, some false) := by
  decide

/-- `Finish(error)` after the last push finished OK but before a D2H `OnReady`
ended its op. -/
def tr_cancel_after_ok_finish : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .h2hIssue, .sendNext,
   .h2hDone true, .cancel, .d2hEnd, .publish]

/-- Correspondence note (1): the OK finish happens in the last push callback
while a D2H `OnReady` may still be outstanding; a `Finish(error)` in that
window is ignored. No test drives it: `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess`
and `TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone` reach
the same frozen status through an external `Finish(OkStatus())`, which is A5's
excluded Ack path and not an event here. -/
theorem trace_cancel_after_ok_finish :
    ((sys 1).run tr_cancel_after_ok_finish).map
      (fun s => (s.life.statusOk, s.published)) = some (true, some true) := by
  decide

/-- `StartPush` on a zero-layer send. -/
def tr_zero_layers : List Ev := [.beginPull, .start]

/-- A zero-layer send finishes OK inside `StartPush`
(`TransferSendSessionTest.ZeroLayerSessionCompletesAndReleasesStagingWithoutHang`;
no manager there, so the `publish` is model-only). Also the producer side of
`ControlHandshakeTest.RegisteredPullIsAcknowledged`,
`ControlHandshakeTest.PullAheadOfRegistrationIsAcknowledgedOnceRegistered` and
`ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged`, whose `TestManager`
has zero layers: `HandlePullStream` claims the pull and spawns `StartPush` on a
detached thread, and the test returns on the ack, so the worker races the
manager's destructor; the pairing takes the order in which the worker wins
(`tr_zero_layers`), the other being `tr_cancel_then_late_start`. -/
theorem trace_zero_layers :
    ((sys 0).run tr_zero_layers).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) ∧
    ((sys 0).run (tr_zero_layers ++ [.publish])).map (fun s => s.published) = some (some true) := by
  decide

/-- Layer 0's push and layer 1's copy both in flight when the deadline fires;
copies completed by the test (`wake` before `d2hEnd`). -/
def tr_drain_h2h_and_d2h : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext,
   .cancel, .h2hDone true, .d2hReady, .wake true, .d2hEnd]

/-- `TransferSendSessionTest.DoneGuaranteesAllResourcesReleasedAndNoHbmOrTransportAccessAfterDone`:
two layers; layer 0's H2H push and layer 1's D2H copy are both in flight when
`Finish(DeadlineExceeded)` fires; layer 0's H2H completes first (`done` stays
`false`, `hasStaging` stays `true`); when layer 1's D2H completes, `wake` sees
`draining`, drops layer 1's H2H (`h2hIssued = 1`), and settles (`done = true`,
`hasStaging = false`), after which the test's trailing `StartPush` is a
`Finish` that changes nothing (`.cancel`; the model's `.start` is refused).
`WaitForH2hCalls` returns as soon as `H2hWriteDirectAsync` is entered, before
the pool task's tail, so the test's `Finish` may also land before `sendNext`;
that order refuses `SendNextLayer(1)` instead of `wake` and ends in the same
state. -/
theorem trace_drain_h2h_and_d2h :
    ((sys 2).run (tr_drain_h2h_and_d2h.take 13)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    ((sys 2).run tr_drain_h2h_and_d2h).map
      (fun s => (s.life.done, s.life.hasStaging, s.h2hIssued)) = some (true, false, 1) ∧
    ((sys 2).run (tr_drain_h2h_and_d2h ++ [.publish])).map (fun s => s.published) =
      some (some false) ∧
    (sys 2).run (tr_drain_h2h_and_d2h ++ [.start]) = none ∧
    ((sys 2).run (tr_drain_h2h_and_d2h ++ [.cancel])).map
      (fun s => (s.life.done, s.life.statusOk)) = some (true, false) := by
  decide

/-- Layer 0's copy fails while layer 1's still runs. -/
def tr_failed_d2h_waits_for_other_layer : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake false, .d2hEnd, .d2hReady, .d2hEnd, .publish]

/-- `SendDrainTest.FailedLayerWaitsForTheOtherLayersCopies`: two layers; layer 0's
D2H copy fails while layer 1's D2H copy is still running; the failure and
staging slot are held back until layer 1's copy finishes. -/
theorem trace_failed_d2h_waits_for_other_layer :
    let pre := tr_failed_d2h_waits_for_other_layer.take 9
    ((sys 2).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sys 2).run (pre ++ [.publish]) = none ∧
    ((sys 2).run tr_failed_d2h_waits_for_other_layer).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `Finish(error)` with the only push in flight; the push then completes OK. -/
def tr_ok_after_cancel_keeps_failure : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .d2hEnd, .wake true, .h2hIssue, .sendNext,
   .cancel, .h2hDone true, .publish]

/-- `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush`
and `TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush`
(no manager, so without the `publish`): dual of `trace_cancel_after_ok_finish`.
If `Finish(error)` runs while the last push is in flight, a subsequent OK push
completion (`h2hDone true` calling `finish true`) cannot override the earlier
failure. (`SendLifecycleTest.SuccessCannotOverrideAnEarlierFailure` reaches the
frozen failure without a push: `Decide(true)` then a no-op `Decide(false)`
with a D2H copy in flight — `tr_expired_while_copying`.) -/
theorem trace_ok_after_cancel_keeps_failure :
    ((sys 1).run (tr_ok_after_cancel_keeps_failure.take 11)).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, false, false) ∧
    ((sys 1).run tr_ok_after_cancel_keeps_failure).map
      (fun s => (s.life.done, s.life.statusOk, s.published)) = some (true, false, some false) := by
  decide

/-- Three layers; layer 0's copy is issued, layer 1's dispatch fails, and layer
0's copy then completes and drains without pushing. -/
def tr_d2h_dispatch_fail_drains_issued : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue false,
   .d2hReady, .wake true, .d2hEnd]

/-- `TransferSendSessionTest.D2hDispatchFailureFailsUnissuedLayersAndDrainsIssuedCopies`:
three layers; layer 0's D2H copy is issued, layer 1's `D2hSyncDispatch` fails
(`d2hIssue false`), and layer 2 is never tried (`fail_unissued_layers` resolves
layers 1 and 2 with that failure); layer 0's in-flight copy keeps staging
pinned until `CompleteD2h(0)` drains both its copy op and `SendNextLayer(0)`'s
callback without issuing any H2H push. -/
theorem trace_d2h_dispatch_fail_drains_issued :
    ((sys 3).run (tr_d2h_dispatch_fail_drains_issued.take 6)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging, s.d2hIssued)) =
      some (true, false, true, 1) ∧
    ((sys 3).run tr_d2h_dispatch_fail_drains_issued).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.h2hIssued)) =
      some (true, false, false, 0) := by
  decide

/-- Three layers; layer 2's dispatch is held back while layer 0 is copied and
pushed, the send is cancelled and layer 1's copy finishes, and layer 2's failed
dispatch ends the last op. -/
def tr_failed_dispatch_ends_send : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true, .d2hBegin,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext, .h2hDone true,
   .cancel, .d2hReady, .wake true, .d2hEnd, .d2hIssue false]

/-- `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenFailedDispatchEndsSend`:
three layers; layers 0 and 1 are copying while layer 2's `D2hSyncDispatch` is
held inside `SetD2hDispatchHook` (`d2hBegin`); layer 0 completes and is pushed,
`Finish(CancelledError)` cancels the send, layer 1's copy finishes (`Done()` is
still `false` because layer 2's pending dispatch holds the last op), and when
layer 2's dispatch returns an error (`d2hIssue false`), its `EndOp` settles
the session. (`WaitForH2hCalls` returns before the pool task's tail, so `Finish`
may also land before `sendNext`, refusing `SendNextLayer(1)` instead of `wake`.) -/
theorem trace_failed_dispatch_ends_send :
    ((sys 3).run (tr_failed_dispatch_ends_send.take 17)).map
      (fun s => (s.life.draining, s.life.done, s.life.inFlight)) = some (true, false, 1) ∧
    ((sys 3).run tr_failed_dispatch_ends_send).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.h2hIssued)) =
      some (true, false, false, 1) := by
  decide

/-- One layer; the first (only) dispatch fails and `fail_unissued_layers`
resolves layer 0's promise with the failure. -/
def tr_first_dispatch_fails : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue false, .wake false]

/-- `TransferSendSessionTest.FailedD2hDispatchOnFirstLayerSettlesAtOnce`: one
layer; `D2hSyncDispatch` fails for layer 0 (`d2hIssue false`), which fails the
send and ends the copy's op while `SendNextLayer(0)`'s op, armed in `start`,
keeps it unsettled; `fail_unissued_layers` then resolves layer 0's promise with
the dispatch error inside `StartPush`, the armed callback runs `wake false`
(`FinishLocked` ignores the repeat), and its `end_op` settles the send with
nothing in flight and no push issued. -/
theorem trace_first_dispatch_fails :
    ((sys 1).run (tr_first_dispatch_fails.take 4)).map
      (fun s => (s.life.draining, s.life.done, s.life.inFlight)) = some (true, false, 1) ∧
    ((sys 1).run tr_first_dispatch_fails).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.h2hIssued)) =
      some (true, false, false, 0) := by
  decide

/-- One layer; `Finish(error)` with the only push in flight, after which the
push fails. -/
def tr_cancel_then_push_fails : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .sendNext, .cancel, .h2hDone false]

/-- `TransferSendSessionTest.CancelledSendWhosePushThenFailsKeepsCancelled`:
the `h2hDone false` counterpart of `trace_ok_after_cancel_keeps_failure`. The
send is cancelled while its only push is in flight (`WaitForH2hCalls(1)`,
then `Finish(CancelledError)`); when the push then fails
(`CompleteH2h(0, InternalError)`), `FinishLocked(push_res.status(), kPushPhase)`
is ignored — the first `Finish` wins — and the push's `end_op` is what settles
the send. (`WaitForH2hCalls` returns before the pool task's tail, so the
`Finish` may also land before `sendNext`; the last layer's `SendNextLayer(1)`
returns at its bounds check either way.) -/
theorem trace_cancel_then_push_fails :
    ((sys 1).run (tr_cancel_then_push_fails.take 10)).map
      (fun s => (s.life.draining, s.life.done, s.life.inFlight)) = some (true, false, 1) ∧
    ((sys 1).run tr_cancel_then_push_fails).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, false, false) ∧
    ((sys 1).run [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
      .h2hIssue, .cancel, .sendNext, .h2hDone false]).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, false, false) := by
  decide

def events : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hIssue false, .d2hReady, .d2hEnd,
   .wake true, .wake false, .h2hIssue, .sendNext, .h2hDone true, .h2hDone false,
   .cancel, .publish]

instance (s : Send) : Decidable (Safe s) := by
  unfold Safe SettleSafe Drained StagingIntegrity SettlesPromptly NoUnderflow Publication CountersOrdered NoOpLeak PullClaimed PublishedQuiet
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorem check the same predicate. -/
def violates (s : Send) : Bool := !decide (Safe s)

theorem violates_iff (s : Send) : violates s = false ↔ Safe s := by
  simp [violates]

/-- Sanity check on the search itself: a `Finish` that settles at once, ignoring
in-flight work, must be caught. -/
def cancelEager (s : Send) : Option Send :=
  some { s with life := { s.life with draining := true, done := true, hasStaging := false } }

#guard (match ModelCheck.check ⟨init 1, fun s e => match e with
          | .cancel => cancelEager s
          | e => step s e⟩ events violates 8 with
        | .counterexample _ => true
        | _ => false)

/-- And the point of the op `SendNextLayer` takes (`++in_flight_`): without it a
send settles under a `Finish` while `SendNextLayer` is armed or waiting on the
pool, and the callback or pool task that then runs ends an op the session no
longer owns. -/
def trySendNextUncounted (s : Send) : Send :=
  if s.queued < s.numLayers ∧ s.life.draining = false then { s with queued := s.queued + 1 } else s

def startUncounted (s : Send) : Option Send :=
  if s.pullStarted = false ∨ s.started = true ∨ s.life.draining = true ∨ s.life.done = true then none
  else if s.numLayers = 0 then some (s.finish true)
  else some { s with started := true }.trySendNextUncounted

def sendNextUncounted (s : Send) : Option Send :=
  if s.chaining = 0 then none
  else some { s with chaining := s.chaining - 1 }.trySendNextUncounted.endOp

#guard (match ModelCheck.check ⟨init 1, fun s e => match e with
          | .start => startUncounted s
          | .sendNext => sendNextUncounted s
          | e => step s e⟩ events violates 9 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant for `NoOpLeak`: `SendNextLayer(l)` omits the
`l >= base_->num_layers()` / `l >= d2h_layer_futures_.size()`
bounds checks and takes an op unconditionally. After the last
layer's pool task calls `SendNextLayer(numLayers)`, that call claims
an op for a non-existent layer whose D2H future never becomes ready, leaking the
session and its staging forever (`SettleSafe` holds vacuously, `NoOpLeak`
catches it). -/
def trySendNextUnbounded (s : Send) : Send :=
  match s.life.beginOp with
  | some l => { s with life := l, queued := s.queued + 1 }
  | none => s

def sendNextUnbounded (s : Send) : Option Send :=
  if s.chaining = 0 then none
  else some { s with chaining := s.chaining - 1 }.trySendNextUnbounded.endOp

#guard (match ModelCheck.check ⟨init 1, fun s e => match e with
          | .sendNext => sendNextUnbounded s
          | e => step s e⟩ events violates 9 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant for `fail_unissued_layers` in `StartPush`: `wake` only fires on a
completed D2H copy (`woken < d2hReady`), omitting the cleanup that resolves
unissued layers' promises when the D2H loop exits early. If the send is
cancelled after `StartPush` arms `SendNextLayer(0)` and before layer 0's copy
is issued, `in_flight_ = 1` has no enabled `drainEvents` step under the mutant
and leaks forever. -/
def stepWithoutFailUnissued (s : Send) : Ev → Option Send
  | .wake ok =>
    if s.woken < s.queued ∧ s.woken < s.d2hReady then
      let s := { s with woken := s.woken + 1 }
      if ok = false then some (s.finish false).endOp
      else if s.life.draining = true then some s.endOp
      else some { s with pooled := s.pooled + 1 }
    else none
  | e => step s e

theorem trace_without_fail_unissued_leaks :
    let sysMut : System Send Ev := ⟨init 1, stepWithoutFailUnissued⟩
    (sysMut.run [.beginPull, .start, .cancel]).map
      (fun s => (s.life.inFlight, drainEvents.any fun e => (stepWithoutFailUnissued s e).isSome)) =
    some (1, false) := by
  decide

/-! ## Failure phase (`TransferDetails`)

A failed session records the phase it failed in:
`FinishLocked(status, phase)` latches `failed_phase_` from the
explicit `phase` argument when the failing step knows it (`kD2hPhase` in
`SendNextLayer`'s `layer_future->OnReady`, `kPushPhase` in
`H2hWriteDirectAsync`'s callback, and `kStagingPhase` in
`AcquireStagingSlot` when a layer-buffer lease fails under bounded layer
staging, which is modelled in `LayerStaging` and is not an event here) and
otherwise from `CurrentPhaseLocked`,
and `Details()` reports it to
`CompleteReadWithDetails` (`RetireSettledSessionLocked`:
`transfer_details_[req_id] = send->Details();`, under the `req_id` of the
session's `SessionEntry`).
Nothing in the lifecycle reads it, so
it is not state here; `phase` below is `CurrentPhaseLocked` evaluated on the
model, and `failedPhase s e` is the value `FinishLocked` latches when event `e`
fails in state `s`. Inside `StartPush` before its locked section the model
has one state (`pullStarted ∧ ¬started`) where C++ has three phases —
`await_pull` before the endpoints are recorded (`remote_data_endpoints_ = remote_data_endpoints`),
`staging` while `AcquireStagingWithRetry` runs (the shape of
`TransferSendSessionTest.CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator`),
`d2h` between `staging_ = *std::move(acquired)` and the `if (draining_ || done_)`
re-check — because `start` is atomic; `phase` reports
`awaitPull` for all three (a host-source send has only `await_pull` there:
`CurrentPhaseLocked` returns `kPushPhase` for `source_ == BlockLocation::kHost`
as soon as the endpoints are recorded). No replayed test evaluates a phase in that window;
the `trace_details_*` theorems only read `phase` in states where the two agree. -/

/-- The four phases of `TransferSendSession::CurrentPhaseLocked` (`kAwaitPullPhase` .. `kPushPhase`). -/
inductive Phase
  | awaitPull | staging | d2h | push
  deriving Repr, DecidableEq

/-- `CurrentPhaseLocked`: no endpoints yet (`remote_data_endpoints_.empty()`, set at
`StartPush` entry, i.e. `started = false`) → `awaitPull`; some
D2H future missing or not ready (`d2h_layer_futures_.size() < base_->num_layers()`
or `!future.IsReady()`) → `d2h`; otherwise `push`.
The `staging_.empty()` branch (`staging`) is folded into `awaitPull`
because `start` is atomic (see above); on a host-source send the
`source_ == BlockLocation::kHost` arm returns `push` before that test, which the
completed-copy prefix of `tr_host_source_prefix` reproduces
(`d2hIssued = d2hReady = numLayers` after `start`). -/
def phase (s : Send) : Phase :=
  if s.started = false then .awaitPull
  else if s.d2hIssued < s.numLayers ∨ s.d2hReady < s.d2hIssued then .d2h
  else .push

/-- The phase `FinishLocked` latches when `e` fails in `s`: the D2H completion
callback passes `kD2hPhase`, the push callback `kPushPhase`;
every other failure (`Finish(error)` from the deadline, shutdown,
spawn failure, or a copy that could not be issued in `StartPush`) takes the current
phase (`CurrentPhaseLocked()`). The explicit `kStagingPhase` of a failed
layer-buffer lease (`AcquireStagingSlot`, bounded layer staging only) is not
represented: that failure is a `cancel` here and `LayerStaging` models it. -/
def failedPhase (s : Send) : Ev → Phase
  | .wake false => .d2h
  | .h2hDone false => .push
  | _ => phase s

/-- `SendLifecycleTest.DetailsSayAnUnpulledSendTimedOutAwaitingItsPull`: a registered send
that nobody pulled expires in `await_pull` and is reported failed. -/
theorem trace_details_unpulled_send_awaits_pull :
    ((sys 1).run []).map (fun s => (phase s, failedPhase s .cancel)) =
      some (.awaitPull, .awaitPull) ∧
    ((sys 1).run tr_idle_cancel).map (fun s => s.published) = some (some false) := by
  decide

/-- One layer; the deadline fires with the copy in flight and `SendNextLayer(0)`'s
op taken; the copy's callbacks end both ops. -/
def tr_expired_while_copying : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .cancel, .d2hReady, .wake true, .d2hEnd, .publish]

/-- `SendDrainTest.DetailsSayAnExpiredSendWasStillCopying` and
`SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds`: the deadline fires
with a D2H copy in flight; the failure is latched in `d2h`, and the send still
waits for the copy before it settles and is reported. Also the session-side
shape of `TransferSendSessionTest.SendSessionImplementsTransferSessionInterface`
(`Finish(error)` by hand, no manager) and of
`SendLifecycleTest.SuccessCannotOverrideAnEarlierFailure` (`Decide(true)`, then
a `Decide(false)` that `FinishLocked` ignores — not an event). -/
theorem trace_details_expired_while_copying :
    let pre := tr_expired_while_copying.take 4
    ((sys 1).run pre).map (fun s => failedPhase s .cancel) = some .d2h ∧
    ((sys 1).run (pre ++ [.cancel])).map (fun s => (s.life.draining, s.life.done)) =
      some (true, false) ∧
    ((sys 1).run (tr_expired_while_copying.take 8)).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, false, false) ∧
    ((sys 1).run tr_expired_while_copying).map
      (fun s => (s.life.done, s.published)) = some (true, some false) := by
  decide

/-- One layer; the copy completes with an error. -/
def tr_failed_copy : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake false, .d2hEnd, .publish]

/-- `SendDrainTest.DetailsCarryTheErrorOfAFailedCopy`: a D2H copy
completes with an error; the completion callback latches `d2h` explicitly
(`self->FinishLocked(status_or.status(), kD2hPhase)`) and the send is reported failed. -/
theorem trace_details_failed_copy :
    let pre := tr_failed_copy.take 5
    ((sys 1).run pre).map (fun s => failedPhase s (.wake false)) = some .d2h ∧
    ((sys 1).run tr_failed_copy).map
      (fun s => (s.life.done, s.published)) = some (true, some false) := by
  decide

/-- Two layers; layer 0's push fails while layer 1's copy still runs. -/
def tr_failed_push_while_copies_run : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .h2hDone false, .sendNext,
   .d2hReady, .d2hEnd, .publish]

/-- `SendDrainTest.DetailsSayAFailedPushFailedInPushWhileCopiesRun`: two layers; layer
0's push fails while layer 1's copy is still running. `CurrentPhaseLocked` would
say `d2h` (a future is not ready), but the push callback passes `kPushPhase`
(`self->FinishLocked(push_res.status(), kPushPhase)`), so the failure is reported in `push`; the send still drains layer
1's copy before it settles. The injected failure
(`hooks::kRaidenManagerBaseH2hWrite`) is reported synchronously inside the
pool task, before its tail, so `SendNextLayer(1)` finds the send draining and
layer 1 gets no `wake`. -/
theorem trace_details_failed_push_while_copies_run :
    let pre := tr_failed_push_while_copies_run.take 10
    ((sys 2).run pre).map (fun s => (phase s, failedPhase s (.h2hDone false))) =
      some (.d2h, .push) ∧
    ((sys 2).run (pre ++ [.h2hDone false])).map
      (fun s => (s.life.draining, s.life.done)) = some (true, false) ∧
    ((sys 2).run tr_failed_push_while_copies_run).map
      (fun s => (s.life.done, s.published)) = some (true, some false) := by
  decide

/-! ### Host-resident source

`StartPush` on a `BlockLocation::kHost` send (`if (source_ == BlockLocation::kHost)`)
takes no staging, issues no D2H copy and seats an already-OK future per layer;
`step` has no branch for it, and `Send.lean` §"Host-resident source" argues that
the host send is the device send with its copies already complete. The prefix
below is that argument's witness; the traces after it replay the host-source
tests on it. -/

/-- `beginPull, start`, then every layer's copy issued, finished and retired
with nothing else happening. -/
def tr_host_source_prefix (n : Nat) : List Ev :=
  [.beginPull, .start] ++ (List.replicate n [.d2hBegin, .d2hIssue true, .d2hReady, .d2hEnd]).flatten

/-- The state a host-source `StartPush` leaves once `SendNextLayer(0)` has taken
its op: started, the push chain armed on layer 0 (`queued = 1`, `inFlight = 1`),
every layer's future ready and no copy pending or outstanding
(`d2hIssued = d2hReady = d2hRetired = numLayers`). From it `d2hBegin`,
`d2hReady` and `d2hEnd` are disabled, as on the host path, and `phase` is
`push`, as `CurrentPhaseLocked` returns for a host source. (`wake false` stays
enabled — a ready future carrying an error — which a host send, whose futures
are all seated OK, never produces: the model admits more than the host path.) -/
theorem trace_host_source_prefix :
    (∀ n ∈ [1, 2, 3],
      (sys n).run (tr_host_source_prefix n) =
        some { init n with pullStarted := true, started := true,
                           d2hIssued := n, d2hReady := n, d2hRetired := n,
                           queued := 1, life := { inFlight := 1 } }) ∧
    (∀ n ∈ [1, 2, 3],
      ((sys n).run (tr_host_source_prefix n)).map (fun s =>
        (phase s, [Ev.d2hBegin, .d2hReady, .d2hEnd].map fun e => (step s e).isSome)) =
        some (.push, [false, false, false])) := by
  decide

/-- Three layers, host source: the chain issues every push before any
completes, and the pushes complete in the order `0, 2, 1` (layers are
interchangeable here, A3). -/
def tr_host_source_three_pushes : List Ev :=
  tr_host_source_prefix 3 ++
    [.wake true, .h2hIssue, .sendNext, .wake true, .h2hIssue, .sendNext,
     .wake true, .h2hIssue, .sendNext, .h2hDone true, .h2hDone true, .h2hDone true]

/-- Two layers, host source: both pushes issued, layer 0's fails, layer 1's
then completes OK and is the last op. -/
def tr_host_source_push_fails_then_drains : List Ev :=
  tr_host_source_prefix 2 ++
    [.wake true, .h2hIssue, .sendNext, .wake true, .h2hIssue, .sendNext,
     .h2hDone false, .h2hDone true]

/-- One layer, host source: the deadline fires with the only push in flight. -/
def tr_host_source_deadline_while_pushing : List Ev :=
  tr_host_source_prefix 1 ++ [.wake true, .h2hIssue, .sendNext, .cancel, .h2hDone true]

/-- Two layers, host source, under a manager: both pushes succeed and the poll
publishes `done_sending`. -/
def tr_host_source_two_pushes_published : List Ev :=
  tr_host_source_prefix 2 ++
    [.wake true, .h2hIssue, .sendNext, .wake true, .h2hIssue, .sendNext,
     .h2hDone true, .h2hDone true, .publish]

/-- Two layers, host source, under a manager: the injected `H2hWriteDirectAsync`
failure is reported inside layer 0's pool task before its tail, so
`SendNextLayer(1)` finds the send draining, and the poll publishes the failure. -/
def tr_host_source_injected_push_failure : List Ev :=
  tr_host_source_prefix 2 ++ [.wake true, .h2hIssue, .h2hDone false, .sendNext, .publish]

/-- `TransferSendSessionTest.HostSourcePushesRegisteredHostBlocksWithoutStagingOrD2h`
(all three pushes issued, `d2h_calls() == 0`, the send settles OK with the last
completion whichever layer it is),
`TransferSendSessionTest.HostSourceFailedLayerPushFailsTheSendOnceInFlightLayersDrain`
(layer 0's push fails: draining but not done while layer 1's push runs; its OK
completion settles the send with the failure, reported in `push`),
`TransferSendSessionTest.HostSourceDeadlineWhilePushingIsAttributedToThePushPhase`
(`Finish(DeadlineExceeded)` with the push in flight latches `push`, since a host
send has no staging or D2H phase; the OK completion keeps the failure),
`TransferSendSessionTest.HostSourceWithNoLayersSettlesWithoutAPush`
(`tr_zero_layers`) and
`TransferSendSessionTest.HostSourceFinishedBeforeStartPushIssuesNothing`
(`tr_cancel_before_start`: the host branch returns at its
`if (draining_ || done_) return;` without calling `Finish`, and `.start` is
refused). The producer side of
`ControlHandshakeTest.StartReadOverPeregrinePropagatesConsumerPortAndCompletesPush`
is `tr_host_source_two_pushes_published`, of the socket-level
`HostMemoryTransferTest.AReadNobodyPullsExpiresAwaitingThePull` is
`tr_idle_cancel` with `await_pull` (`trace_details_unpulled_send_awaits_pull`),
and of `HostMemoryTransferTest.AFailedPushFailsTheSendAndExpiresTheReceive`
(`hooks::kRaidenManagerBaseH2hWrite` at probability 1) is
`tr_host_source_injected_push_failure`, published failed in `push`. -/
theorem trace_host_source :
    ((sys 3).run (tr_host_source_three_pushes.take 23)).map
      (fun s => (s.h2hIssued, s.d2hPending, s.life.done)) = some (3, false, false) ∧
    ((sys 3).run (tr_host_source_three_pushes.take 25)).map (fun s => s.life.done) = some false ∧
    ((sys 3).run tr_host_source_three_pushes).map
      (fun s => (s.life.done, s.life.statusOk, s.h2hOk)) = some (true, true, 3) ∧
    ((sys 2).run tr_host_source_two_pushes_published).map
      (fun s => (s.life.done, s.life.statusOk, s.published)) = some (true, true, some true) ∧
    ((sys 2).run (tr_host_source_push_fails_then_drains.take 17)).map
      (fun s => (s.life.draining, s.life.done, s.life.statusOk)) = some (true, false, false) ∧
    ((sys 2).run tr_host_source_push_fails_then_drains).map
      (fun s => (s.life.done, s.life.statusOk, failedPhase s (.h2hDone false))) =
      some (true, false, .push) ∧
    ((sys 1).run (tr_host_source_deadline_while_pushing.take 9)).map
      (fun s => failedPhase s .cancel) = some .push ∧
    ((sys 1).run tr_host_source_deadline_while_pushing).map
      (fun s => (s.life.done, s.life.statusOk)) = some (true, false) ∧
    (sys 2).run (tr_cancel_before_start ++ [.start]) = none ∧
    ((sys 2).run (tr_host_source_injected_push_failure.take 12)).map
      (fun s => failedPhase s (.h2hDone false)) = some .push ∧
    ((sys 2).run tr_host_source_injected_push_failure).map
      (fun s => (s.life.done, s.h2hIssued, s.published)) = some (true, 1, some false) := by
  decide

/-! The six data-connection-policy tests
(`TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeDone`,
`TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnections`,
`TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush`,
`TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush`,
`TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenFailedDispatchEndsSend`,
`TransferSendSessionTest.CreateRejectsUnknownDataConnReuseValue`) exercise
`TPU_RAIDEN_DATA_CONN_REUSE` parsing and the transport's idle-connection
counts, which are unmodelled: under `per_transfer` the last `EndOp` of a
draining send closes connections in a window where `in_flight_ == 1`,
`draining_` and `!done_` are all fixed (`ReleaseDataConnectionsAfterSend`, A7),
so the model's atomic `endOp` is unchanged. Their lifecycle content is
replayed: the two `…CancelledMidPush` tests are `tr_ok_after_cancel_keeps_failure`
(without the `publish`), `…WhenFailedDispatchEndsSend` is
`tr_failed_dispatch_ends_send`, and the two `…IdlePeerConnections` tests are
`tr_one_layer_sync_push`. -/


/-! Sibling send-lifecycle tests whose stimulus is outside this model's event set:
- `SendLifecycleTest.SuccessfulSendWithoutWorkSettlesImmediately`,
  `SendLifecycleTest.SuccessfulSendWaitsForEveryOperation`,
  `SendLifecycleTest.FailureCannotOverrideAnEarlierSuccess`,
  `SendLifecycleTest.DetailsReportWhenTheSendCompleted`,
  `SendDrainTest.WaitForPendingWorkWaitsForActiveSendSession`, and
  `TransferSendSessionTest.StatusIsFrozenOnceSessionIsDrainingOrDone` drive an
  external `Finish(OkStatus())` (`TestManager::Decide(failed = false)` /
  `session->Finish(absl::OkStatus())`), which in production is only the
  consumer-Ack path `AckSend` that A5 excludes as dead.
- `ControlHandshakeTest.PullOfUnregisteredBlockIsRejected`,
  `ControlHandshakeTest.PullWithDuplicateSourceBlockIsRejected`, and
  `ControlHandshakeTest.EmptyPullIsRejected` reject pulls in
  `ValidateRequestedBlocksLocked` before `pull_started_` is claimed (covered by
  `PrefillDecode/BlockOrdering.lean`).
- `SendLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` and
  `SendLifecycleTest.BoundedLayerStagingFailureModesSettleAndReleaseStagingBlock`
  exercise bounded `StagingLayerManager` (`num_staging_layers < num_layers`) and
  are replayed in `PrefillDecode/LayerStaging/Traces.lean`.
- `SendDrainTest.BlockReadinessWaitsForTheSendsLayerCopy` tests the transport's
  block-readiness gate (`KVCacheManagerBase::GetSession(uuid)` →
  `TransferSession::GetDataReadyFuture(l)`). -/

end Send

end TpuSyncVerify.Transfer.PrefillDecode

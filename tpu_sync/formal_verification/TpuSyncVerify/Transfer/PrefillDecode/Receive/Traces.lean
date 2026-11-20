import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Frame

/-! # `Recv`: replay of C++ tests, bounded searches, mutants

Part of `TpuSyncVerify.Transfer.PrefillDecode.Receive`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ## Replay and bounded search

Replay traces (`decide`) paired with `TransferReceiveSession` / manager GTests
plus bounded searches and mutants. -/

/-- Two-layer push receive with each stream lease spanning H2D dispatch and block accounting. -/
def tr_normal : List Ev :=
  [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd,
   .pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd,
   .h2dReady, .h2dReady, .h2dDone true, .h2dDone true, .publish]

/-- `RecvLifecycleTest.TcpLastLayerThenItsBlockReportRecordLastPacketOnceAndFinishOnce`:
two-layer push receive sets `netIssued` on the last dispatch, finishes in the last H2D callback, and publishes `done_recving`. -/
theorem trace_normal :
    ((sysPush 2).run (tr_normal.take 8)).map
      (fun s => (s.netIssued, s.layersAccounted, s.life.done)) = some (true, 1, false) ∧
    ((sysPush 2).run (tr_normal.take 12)).map (fun s => s.life.done) = some false ∧
    ((sysPush 2).run tr_normal).map
      (fun s => (s.life.done, s.published, s.completed)) = some (true, some true, 2) := by
  decide

/-- `RecvLifecycleTest.OutOfOrderLayersSettleAfterEveryH2d`: two layers complete
one at a time, refusing an intermediate `publish` until both H2D callbacks run. -/
def tr_layers_one_at_a_time : List Ev :=
  [.h2dBegin, .h2dIssue true, .netAccount, .h2dReady, .h2dDone true,
   .h2dBegin, .h2dIssue true, .netAccount, .h2dReady, .h2dDone true, .publish]

theorem trace_layers_one_at_a_time :
    (sysPush 2).run (tr_layers_one_at_a_time.take 5 ++ [.publish]) = none ∧
    ((sysPush 2).run tr_layers_one_at_a_time).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- `RecvLifecycleTest.DetailsReportWhenTheLastH2dLanded` (and `RecvLifecycleTest.EmptyBlockReportIssuesNoH2d`):
one-layer receive settles in its H2D callback and publishes `done_recving`. -/
def tr_one_layer : List Ev :=
  [.h2dBegin, .h2dIssue true, .netAccount, .h2dReady, .h2dDone true, .publish]

theorem trace_one_layer :
    ((sysPush 1).run tr_one_layer).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- `pollReady` fires before the H2D callback runs. -/
def tr_poll_before_callbacks : List Ev :=
  [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd, .h2dReady, .pollReady,
   .h2dDone true, .publish]

/-- Model-only: `pollReady` starts draining once futures are ready, but `publish`
waits until the H2D callback ends its op. -/
theorem trace_poll_before_callbacks :
    ((sysPush 1).run (tr_poll_before_callbacks.take 7)).map
      (fun s => (s.life.draining, s.life.done, s.completed)) = some (true, false, 0) ∧
    (sysPush 1).run (tr_poll_before_callbacks.take 7 ++ [.publish]) = none ∧
    ((sysPush 1).run tr_poll_before_callbacks).map
      (fun s => (s.published, s.completed)) = some (some true, 1) := by
  decide

/-- Deadline fires during an in-flight H2D copy on a pulled receive. -/
def tr_deadline_during_copy : List Ev :=
  [.pullReply true, .h2dBegin, .h2dIssue true, .cancel, .h2dReady, .h2dDone true, .publish]

/-- Deadline fires during an in-flight H2D copy on a push-plan receive. -/
def tr_deadline_during_copy_push : List Ev :=
  [.h2dBegin, .h2dIssue true, .cancel, .h2dReady, .h2dDone true, .publish]

/-- `RecvDrainTest.ExpiredReceiveKeepsStagingUntilH2dEnds` and
`RecvDrainTest.TimeoutDuringH2dDispatchKeepsStaging`: deadline during H2D
dispatch/copy keeps staging pinned until `h2dDone`, then publishes failure. -/
theorem trace_deadline_during_copy :
    ((sysLoad 1).run (tr_deadline_during_copy.take 4)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    ((sysLoad 1).run tr_deadline_during_copy).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) ∧
    ((sysPush 1).run (tr_deadline_during_copy_push.take 3)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    ((sysPush 1).run tr_deadline_during_copy_push).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `Finish` lands between the first and second `mu_` scopes of `ExecuteLayerH2d`. -/
def tr_finish_between_locks : List Ev := [.h2dBegin, .cancel, .h2dIssue true]

/-- Model-only: `ExecuteLayerH2d`'s `done_ || draining_` re-check skips dispatch
and settles via `EndRecvOpLocked()`. -/
theorem trace_finish_between_locks :
    ((sysPush 1).run tr_finish_between_locks).map
      (fun s => (s.issued, s.life.done)) = some (0, true) := by
  decide

/-- `pushBegin` is refused once the session is draining. -/
theorem trace_no_push_after_finish :
    (sysPush 1).run [.cancel, .pushBegin] = none := by
  decide

/-- `network_completed_` set while the H2D copy still runs. -/
def tr_net_completion_waits_for_h2d : List Ev :=
  [.h2dBegin, .h2dIssue true, .netAccount, .h2dReady, .h2dDone true, .publish]

/-- `RecvLifecycleTest.NetworkCompletionWaitsForH2d`: after `netAccount`, both
`pollReady` and `publish` remain disabled until the H2D copy finishes. -/
theorem trace_net_completion_waits_for_h2d :
    let pre := tr_net_completion_waits_for_h2d.take 3
    ((sysPush 1).run pre).map
      (fun s => (s.layersAccounted, s.life.done, s.life.hasStaging)) = some (1, false, true) ∧
    (sysPush 1).run (pre ++ [.pollReady]) = none ∧
    (sysPush 1).run (pre ++ [.publish]) = none ∧
    ((sysPush 1).run tr_net_completion_waits_for_h2d).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- Fast H2D callback retires the session before block accounting runs. -/
def tr_late_net_account_after_retire : List Ev :=
  [.h2dBegin, .h2dIssue true, .h2dReady, .h2dDone true, .publish]

/-- `RecvLifecycleTest.LateBlockAccountingAfterRetirementIsANoOp` and
`RecvLifecycleTest.SessionRemainsValidAcrossAsyncH2dCallbacksAfterCallerDropsPtr`:
the H2D callback settles and releases staging, and a late `netAccount` is ignored. -/
theorem trace_late_net_account_after_retire :
    ((sysPush 1).run (tr_late_net_account_after_retire.take 3)).map
      (fun s => (s.life.done, s.life.hasStaging)) = some (false, true) ∧
    ((sysPush 1).run (tr_late_net_account_after_retire.take 4)).map
      (fun s => (s.life.done, s.life.hasStaging)) = some (true, false) ∧
    ((sysPush 1).run tr_late_net_account_after_retire).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) ∧
    (sysPush 1).run (tr_late_net_account_after_retire ++ [.netAccount]) = none := by
  decide

/-- One of two issued H2D copies fails while the other is in flight. -/
def tr_failed_h2d_waits_for_other_layer : List Ev :=
  [.h2dBegin, .h2dIssue true, .h2dBegin, .h2dIssue true, .h2dReady, .h2dDone false,
   .h2dReady, .h2dDone true, .publish]

/-- `RecvDrainTest.FailedLayerWaitsForOtherH2dCopies`: when layer 0's H2D fails,
staging stays pinned until layer 1's H2D finishes, then failure is published. -/
theorem trace_failed_h2d_waits_for_other_layer :
    let pre := tr_failed_h2d_waits_for_other_layer.take 6
    ((sysPush 2).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysPush 2).run (pre ++ [.publish]) = none ∧
    ((sysPush 2).run tr_failed_h2d_waits_for_other_layer).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `RecvLifecycleTest.SingleFailedH2dReportsFailureAndReturnsStaging`: single
failing H2D copy settles in its callback and publishes failure. -/
def tr_single_failed_h2d : List Ev :=
  [.h2dBegin, .h2dIssue true, .h2dReady, .h2dDone false, .publish]

theorem trace_single_failed_h2d :
    ((sysPush 1).run tr_single_failed_h2d).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- Open push lease across `cancel`. -/
def tr_push_lease_pins_staging_on_cancel : List Ev := [.pushBegin, .cancel, .pushEnd, .publish]

/-- `RecvLifecycleTest.IncomingPushLeasePinsStagingDuringWriteAndRejectsWhenDraining`,
`DemandStagingTest.UnregisteringInFlightReceiverDefersUntilItSettles`, and
`RecvLifecycleTest.FailedIncomingPushImmediatelyFailsSessionAndReleasesStagingBeforeDeadline`:
an open push lease pins staging across `cancel` until `pushEnd`. -/
theorem trace_push_lease_pins_staging_on_cancel :
    ((sysPush 1).run (tr_push_lease_pins_staging_on_cancel.take 2)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysPush 1).run (tr_push_lease_pins_staging_on_cancel.take 2 ++ [.pushBegin]) = none ∧
    ((sysPush 1).run tr_push_lease_pins_staging_on_cancel).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- H2D copy and callback finish while the push lease is still open. -/
def tr_push_lease_outlives_h2d : List Ev :=
  [.pushBegin, .h2dBegin, .h2dIssue true, .h2dReady, .h2dDone true, .pushEnd, .publish]

/-- `RecvLifecycleTest.IncomingPushLeaseSpansLayerH2dAndBlockAccountingBeforeReleasing`:
if H2D finishes before `EndIncomingPush`, `pushes = 1` holds settlement until `pushEnd`. -/
theorem trace_push_lease_outlives_h2d :
    let pre := tr_push_lease_outlives_h2d.take 5
    ((sysPush 1).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysPush 1).run (pre ++ [.publish]) = none ∧
    ((sysPush 1).run tr_push_lease_outlives_h2d).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- Deadline during pending pull handshake, followed by `pullReply false`. -/
def tr_deadline_during_handshake : List Ev := [.cancel, .pullReply false, .publish]

/-- Idle push-plan receiver cancelled or timed out. -/
def tr_idle_cancel : List Ev := [.cancel, .publish]

/-- `ControlHandshakeTest.ExpiredReceiveKeepsStagingUntilHandshakeEnds`,
`RecvLifecycleTest.ReceiveWithoutTrafficFailsAtItsDeadline`, `RecvLifecycleTest.GetSessionResolvesActiveRecvAndReturnsNotFoundAfterRetirement`,
`DemandStagingTest.UnregisteringIdleReceiverReleasesPlanAtOnce`, and `DemandStagingTest.DemandStagedReceiverPlanUnregistersWhenItSettles`:
pending handshake pins staging across `cancel`, while an idle push receive settles at once. -/
theorem trace_deadline_during_handshake :
    ((sysLoad 1).run [.cancel]).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysLoad 1).run [.cancel, .publish] = none ∧
    ((sysLoad 1).run tr_deadline_during_handshake).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) ∧
    ((sysPush 1).run tr_idle_cancel).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

def events : List Ev :=
  [.pushBegin, .pushEnd, .pullReply true, .pullReply false, .h2dBegin,
   .h2dIssue true, .h2dIssue false, .h2dReady, .h2dDone true, .h2dDone false,
   .netAccount, .pollReady, .cancel, .publish]

instance (s : Recv) : Decidable (Safe s) := by
  unfold Safe SettleSafe NoRetiredCallback StagingIntegrity SettlesPromptly ReadinessSound Publication CountersOrdered NoOpLeak PublishedQuiet
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorem check the same predicate. -/
def violates (s : Recv) : Bool := !decide (Safe s)

theorem violates_iff (s : Recv) : violates s = false ↔ Safe s := by
  simp [violates]

/-- Sanity check on the search itself: a `Finish` that settles at once, ignoring
in-flight work, must be caught. -/
def cancelEager (s : Recv) : Option Recv :=
  some { s with life := { s.life with draining := true, done := true, hasStaging := false } }

#guard (match ModelCheck.check ⟨initPush 1, fun s e => match e with
          | .cancel => cancelEager s
          | e => step s e⟩ events violates 6 with
        | .counterexample _ => true
        | _ => false)

/-- And the point of A4: let the transport account a layer before its copy is
issued and `IsReadyToComplete` becomes unsound. -/
def netAccountUnordered (s : Recv) : Option Recv :=
  if s.life.done = true ∨ s.life.draining = true then none
  else if s.layersAccounted < s.numLayers then
    some { s with layersAccounted := s.layersAccounted + 1 }
  else none

#guard (match ModelCheck.check ⟨initPush 1, fun s e => match e with
          | .netAccount => netAccountUnordered s
          | e => step s e⟩ events violates 6 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant: returning on `ExecuteLayerH2d`'s `done_ || draining_` re-check
without `EndRecvOpLocked()` leaks `inFlight > 0` (`NoOpLeak`). -/
def h2dIssueLeak (ok : Bool) (s : Recv) : Option Recv :=
  if s.pending = 0 then none
  else
    let s := { s with pending := s.pending - 1 }
    if s.life.done = true ∨ s.life.draining = true then
      some s
    else if ok then
      some { s with issued := s.issued + 1,
                    netIssued := s.netIssued || decide (s.issued + 1 = s.numLayers) }
    else
      some { s with life := (s.life.finishLocked false).endOpLocked }

#guard (match ModelCheck.check ⟨initPush 1, fun s e => match e with
          | .h2dIssue ok => h2dIssueLeak ok s
          | e => step s e⟩ events violates 6 with
        | .counterexample _ => true
        | _ => false)

/-! ## Failure phase (`TransferDetails`)

Evaluates `TransferReceiveSession::CurrentPhaseLocked` (`phase`) and the
`failed_phase_` latched by `FinishLocked(status, phase)` (`failedPhase`) for
`Details()` reporting. -/

/-- The three phases of `TransferReceiveSession::CurrentPhaseLocked`. -/
inductive Phase
  | connect | receive | h2d
  deriving Repr, DecidableEq

/-- `CurrentPhaseLocked`. -/
def phase (s : Recv) : Phase :=
  if s.pullPending then .connect
  else if networkCompleted s then .h2d
  else .receive

/-- Phase latched by `FinishLocked` when `e` fails while `statusOk = true`
(explicit `kH2dPhase` on H2D dispatch/callback failure, else `CurrentPhaseLocked`). -/
def failedPhase (s : Recv) : Ev → Phase
  | .h2dIssue false => .h2d
  | .h2dDone false => .h2d
  | _ => phase s

/-- `RecvLifecycleTest.DetailsSayAReceiveTimedOutWaitingForData`: a push-plan
receive with no traffic expires in `receive` and publishes failure. -/
theorem trace_details_receive_timed_out_waiting_for_data :
    ((sysPush 1).run []).map (fun s => failedPhase s .cancel) = some .receive ∧
    ((sysPush 1).run tr_idle_cancel).map (fun s => (s.life.done, s.published)) =
      some (true, some false) := by
  decide

/-- Layer 0's H2D copy fails before block accounting. -/
def tr_details_failed_upload_in_h2d : List Ev :=
  [.h2dBegin, .h2dIssue true, .h2dReady, .h2dDone false, .publish]

/-- `RecvLifecycleTest.DetailsSayAFailedUploadFailedInH2dWhileDataArrives`:
the H2D callback's explicit `kH2dPhase` latches `.h2d` even when `phase = .receive`. -/
theorem trace_details_failed_upload_in_h2d :
    let pre := tr_details_failed_upload_in_h2d.take 3
    ((sysPush 1).run pre).map (fun s => (phase s, failedPhase s (.h2dDone false))) =
      some (.h2d, .h2d) ∧
    ((sysPush 2).run pre).map (fun s => (phase s, failedPhase s (.h2dDone false))) =
      some (.receive, .h2d) ∧
    ((sysPush 1).run tr_details_failed_upload_in_h2d).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `H2dSyncDispatch` fails on layer 0. -/
def tr_failed_h2d_dispatch : List Ev := [.h2dBegin, .h2dIssue false, .publish]

/-- `RecvLifecycleTest.FailedH2dDispatchFailsAndSettlesTheReceive`: failed
`H2dSyncDispatch` latches `.h2d`, releases its op, settles at once, and publishes failure. -/
theorem trace_failed_h2d_dispatch :
    ((sysPush 1).run (tr_failed_h2d_dispatch.take 1)).map
      (fun s => failedPhase s (.h2dIssue false)) = some .h2d ∧
    ((sysPush 1).run tr_failed_h2d_dispatch).map
      (fun s => (s.issued, s.life.done, s.life.hasStaging, s.published)) =
      some (0, true, false, some false) := by
  decide

/-- Second `Finish(error)` on an already-draining receive. -/
def tr_second_finish_noop : List Ev :=
  [.h2dBegin, .h2dIssue true, .cancel, .cancel, .h2dReady, .h2dDone true, .publish]

/-- `RecvLifecycleTest.SecondFinishOnAFailedReceiveIsANoOp`: a repeat `cancel`
leaves the draining session unchanged until the H2D callback settles it. -/
theorem trace_second_finish_noop :
    (sysPush 1).run (tr_second_finish_noop.take 4) = (sysPush 1).run (tr_second_finish_noop.take 3) ∧
    ((sysPush 1).run tr_second_finish_noop).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- Push lease closes while the H2D copy is still in flight. -/
def tr_push_lease_closes_before_copy : List Ev :=
  [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd, .h2dReady, .h2dDone true, .publish]

/-- `RecvLifecycleTest.PushLeaseClosingBeforeTheCopyEndsDoesNotSettle`: `pushEnd`
while H2D runs keeps `inFlight = 1` and `hasStaging = true` until `h2dDone`. -/
theorem trace_push_lease_closes_before_copy :
    let pre := tr_push_lease_closes_before_copy.take 5
    ((sysPush 1).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging, s.life.inFlight)) =
      some (false, false, true, 1) ∧
    (sysPush 1).run (pre ++ [.publish]) = none ∧
    ((sysPush 1).run tr_push_lease_closes_before_copy).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- Cancelled receive whose in-flight H2D copy subsequently fails. -/
def tr_cancelled_then_copy_fails : List Ev :=
  [.h2dBegin, .h2dIssue true, .cancel, .h2dReady, .h2dDone false, .publish]

/-- `RecvDrainTest.CancelledReceiveWhoseCopyThenFailsStaysCancelled`: first
error wins in `FinishLocked` and `h2dDone false` settles the draining receive. -/
theorem trace_cancelled_then_copy_fails :
    ((sysPush 1).run (tr_cancelled_then_copy_fails.take 3)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    ((sysPush 1).run tr_cancelled_then_copy_fails).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) ∧
    (sysPush 1).run (tr_cancelled_then_copy_fails ++ [.publish]) = none := by
  decide

/-- Two layers completed without block reports (Peregrine layer completions). -/
def tr_peregrine_layers : List Ev :=
  [.pushBegin, .h2dBegin, .h2dIssue true, .pushEnd,
   .pushBegin, .h2dBegin, .h2dIssue true, .pushEnd,
   .h2dReady, .h2dDone true, .h2dReady, .h2dDone true]

/-- `RecvLifecycleTest.PeregrineLayerCompletionsRecordLastPacketOnceAndFinishAfterH2d`:
without `netAccount`, the last layer's `h2dIssue` sets `netIssued` and the last `h2dDone` settles OK. -/
theorem trace_peregrine_layers :
    ((sysPush 2).run (tr_peregrine_layers.take 4)).map
      (fun s => (s.netIssued, decide (networkCompleted s))) = some (false, false) ∧
    ((sysPush 2).run (tr_peregrine_layers.take 8)).map
      (fun s => (s.netIssued, decide (networkCompleted s), s.layersAccounted, s.life.done)) =
      some (true, true, 0, false) ∧
    ((sysPush 2).run (tr_peregrine_layers.take 10)).map (fun s => s.life.done) = some false ∧
    ((sysPush 2).run tr_peregrine_layers).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) := by
  decide

/-- Both H2D callbacks run under the open second push lease before its block report. -/
def tr_tcp_h2d_before_last_report : List Ev :=
  [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd,
   .pushBegin, .h2dBegin, .h2dIssue true, .h2dReady, .h2dReady, .h2dDone true, .h2dDone true,
   .pushEnd, .publish]

/-- `RecvLifecycleTest.TcpH2dFinishingBeforeLastBlockReportRecordsLastPacketAndFinishesOnce`:
both callbacks finish under the open push lease (`draining`), refusing late `netAccount` before `pushEnd` settles. -/
theorem trace_tcp_h2d_before_last_report :
    let pre := tr_tcp_h2d_before_last_report.take 12
    ((sysPush 2).run pre).map
      (fun s => (s.life.draining, s.life.done, s.layersAccounted, s.netIssued)) =
      some (true, false, 1, true) ∧
    (sysPush 2).run (pre ++ [.netAccount]) = none ∧
    ((sysPush 2).run tr_tcp_h2d_before_last_report).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- Producer accepts the pull, then the receive fails in `receive`. -/
def tr_accepted_pull : List Ev := [.pullReply true, .cancel, .publish]

/-- `ControlHandshakeTest.AcceptedPullAdvancesReceivePhaseAndReleasesHandshakeOp`:
`pullReply true` clears `pullPending` and releases the handshake op (`phase = .receive`). -/
theorem trace_accepted_pull :
    ((sysLoad 1).run (tr_accepted_pull.take 1)).map
      (fun s => (s.pullPending, s.life.inFlight, s.life.draining, phase s)) =
      some (false, 0, false, .receive) ∧
    ((sysLoad 1).run (tr_accepted_pull.take 2)).map
      (fun s => (s.life.done, s.life.hasStaging)) = some (true, false) ∧
    ((sysLoad 1).run tr_accepted_pull).map (fun s => s.published) = some (some false) := by
  decide

/-- Deadline sweep fails the receive in `connect` before `pullReply true`. -/
def tr_accepted_pull_after_deadline : List Ev := [.cancel, .pullReply true, .publish]

/-- `ControlHandshakeTest.AcceptedPullAfterDeadlineSweepSettlesExpiredReceive`:
`cancel` latches `.connect`, and `pullReply true` ends the op to settle as failed. -/
theorem trace_accepted_pull_after_deadline :
    ((sysLoad 1).run []).map (fun s => failedPhase s .cancel) = some .connect ∧
    ((sysLoad 1).run (tr_accepted_pull_after_deadline.take 1)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    ((sysLoad 1).run tr_accepted_pull_after_deadline).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.published)) =
      some (true, false, false, some false) := by
  decide

/-- Producer rejects the pull (or connection is refused). -/
def tr_details_rejected_pull : List Ev := [.pullReply false, .publish]

/-- `ControlHandshakeTest.RejectedPullReportsWhyAndWhichProducerRejectedIt` and
`ControlHandshakeTest.RefusedConnectionKeepsItsStatusCode`: `pullReply false`
fails in `.connect` and settles at once. -/
theorem trace_details_rejected_pull :
    ((sysLoad 1).run []).map (fun s => (phase s, failedPhase s (.pullReply false))) =
      some (.connect, .connect) ∧
    ((sysLoad 1).run tr_details_rejected_pull).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- Data completes before a failing control reply arrives. -/
def tr_late_reply_failure_keeps_verdict : List Ev :=
  [.h2dBegin, .h2dIssue true, .h2dReady, .h2dDone true, .pullReply false, .publish]

/-- Model-only: when H2D finishes OK before `pullReply false`, `on_response`
sees `draining_` and skips `FinishLocked`, preserving the OK verdict. -/
theorem trace_late_reply_failure_keeps_verdict :
    ((sysLoad 1).run (tr_late_reply_failure_keeps_verdict.take 4)).map
      (fun s => (s.life.draining, s.life.done, s.life.statusOk, s.pullPending)) =
      some (true, false, true, true) ∧
    ((sysLoad 1).run tr_late_reply_failure_keeps_verdict).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.published)) =
      some (true, true, false, some true) := by
  decide


end Recv

end TpuSyncVerify.Transfer.PrefillDecode

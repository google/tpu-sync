import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Receive.Frame

/-! # `Recv`: replay of upstream tests, bounded searches, mutants

Part of `TpuSyncVerify.Transfer.PrefillDecode.Receive`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ## Replay and bounded search

Concrete traces, checked by `decide`, that document the behaviours the model
admits; and a bounded search confirming that no `Safe` violation is reachable
within a few events of either initial state. The inductive proof above is the
actual guarantee; the search guards against a modelling slip making the proof
vacuous. -/

/-- A two-layer push receive that completes through the last H2D callback and
is published as done. -/
theorem trace_normal :
    ((sysPush 2).run
      [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd,
       .pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd,
       .h2dReady, .h2dReady, .h2dDone true, .h2dDone true, .publish]).map
      (fun s => (s.life.done, s.published, s.completed)) = some (true, some true, 2) := by
  decide

/-- The poll can see `IsReadyToComplete` before the callbacks run; the session
drains but is not published until they have. -/
theorem trace_poll_before_callbacks :
    ((sysPush 1).run
      [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd, .h2dReady, .pollReady]).map
      (fun s => (s.life.draining, s.life.done, s.completed)) = some (true, false, 0) ∧
    (sysPush 1).run
      [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd, .h2dReady, .pollReady,
       .publish] = none ∧
    ((sysPush 1).run
      [.pushBegin, .h2dBegin, .h2dIssue true, .netAccount, .pushEnd, .h2dReady, .pollReady,
       .h2dDone true, .publish]).map
      (fun s => (s.published, s.completed)) = some (some true, 1) := by
  decide

/-- `RecvDrainTest.ExpiredReceiveKeepsStagingUntilH2dEnds` and
`RecvDrainTest.TimeoutDuringH2dDispatchKeepsStaging`: the deadline fires while
an H2D copy is in flight (or while the thread is inside `H2dSyncDispatch` in
`ExecuteLayerH2d`, after the `if (done_ || draining_)` re-check has passed —
since `ExecuteLayerH2d` never re-reads `draining_`/`done_` after the re-check,
that deadline linearises after `h2dIssue`). The session drains but does not
settle until the copy's callback ends the op, and is then published as failed.
Both tests seat the session with `RecvTestManager::AddRecv` (push-plan `Create`,
no pull), so their literal replay is `sysPush 1`; the `sysLoad 1` state after
`pullReply true` is lifecycle-identical and the trace keeps the `StartRead`
shape to show the deadline landing on a pulled receive. -/
theorem trace_deadline_during_copy :
    ((sysLoad 1).run [.pullReply true, .h2dBegin, .h2dIssue true, .cancel]).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    ((sysLoad 1).run
      [.pullReply true, .h2dBegin, .h2dIssue true, .cancel, .h2dReady, .h2dDone true, .publish]).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- The race the second `if (done_ || draining_)` check in `ExecuteLayerH2d`
closes: the session finishes between the two locks of `ExecuteLayerH2d`, so no
copy is issued and the op is released (`EndRecvOpLocked()`). -/
theorem trace_finish_between_locks :
    ((sysPush 1).run [.h2dBegin, .cancel, .h2dIssue true]).map
      (fun s => (s.issued, s.life.done)) = some (0, true) := by
  decide

/-- A push cannot start once the session is draining. -/
theorem trace_no_push_after_finish :
    (sysPush 1).run [.cancel, .pushBegin] = none := by
  decide

/-- `RecvLifecycleTest.NetworkCompletionWaitsForH2d`: `OnBlockShardsReceived`
(`netAccount`) sets `network_completed_` while the H2D copy is still running on
the device; neither the poll nor publication can complete the session until the
H2D copy finishes. -/
theorem trace_net_completion_waits_for_h2d :
    let pre := [.h2dBegin, .h2dIssue true, .netAccount]
    ((sysPush 1).run pre).map
      (fun s => (s.layersAccounted, s.life.done, s.life.hasStaging)) = some (1, false, true) ∧
    (sysPush 1).run (pre ++ [.pollReady]) = none ∧
    (sysPush 1).run (pre ++ [.publish]) = none ∧
    ((sysPush 1).run (pre ++ [.h2dReady, .h2dDone true, .publish])).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- `RecvLifecycleTest.LateBlockAccountingAfterRetirementIsANoOp`: a fast H2D
callback finishes and retires the session before `OnBlockShardsReceived` runs; the
late `netAccount` is ignored (`KVCacheManagerWithTransfer::OnBlockShardsReceived`
returns `OkStatus()` on `it == active_recv_sessions_.end()` once the poll has
dropped the session, and `TransferReceiveSession::OnBlockShardsReceived` returns
`OkStatus()` on `done_ || draining_` while it is still seated). The same copy
without the poll is
`RecvLifecycleTest.SessionRemainsValidAcrossAsyncH2dCallbacksAfterCallerDropsPtr`:
the caller drops its `shared_ptr` after `ExecuteLayerH2d`, the pending callback
keeps the session and its staging slot alive (`hasStaging` until `h2dDone`), and
the callback's `FinishLocked()` settles it and returns the slot. -/
theorem trace_late_net_account_after_retire :
    let pre := [.h2dBegin, .h2dIssue true, .h2dReady, .h2dDone true, .publish]
    ((sysPush 1).run (pre.take 3)).map
      (fun s => (s.life.done, s.life.hasStaging)) = some (false, true) ∧
    ((sysPush 1).run (pre.take 4)).map
      (fun s => (s.life.done, s.life.hasStaging)) = some (true, false) ∧
    ((sysPush 1).run pre).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) ∧
    (sysPush 1).run (pre ++ [.netAccount]) = none := by
  decide

/-- `RecvDrainTest.FailedLayerWaitsForOtherH2dCopies` (and the 1-layer case
`RecvLifecycleTest.SingleFailedH2dReportsFailureAndReturnsStaging`): two layers'
H2D copies are issued; one fails while the other is still in flight; the session
drains and keeps its staging until the remaining copy finishes, then publishes
failure. -/
theorem trace_failed_h2d_waits_for_other_layer :
    let pre := [.h2dBegin, .h2dIssue true, .h2dBegin, .h2dIssue true, .h2dReady, .h2dDone false]
    ((sysPush 2).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysPush 2).run (pre ++ [.publish]) = none ∧
    ((sysPush 2).run (pre ++ [.h2dReady, .h2dDone true, .publish])).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `RecvLifecycleTest.IncomingPushLeasePinsStagingDuringWriteAndRejectsWhenDraining`
and `DemandStagingTest.UnregisteringInFlightReceiverDefersUntilItSettles`:
an open incoming push lease (`pushBegin`) keeps staging pinned across `cancel`
while rejecting new pushes, and releases staging when `pushEnd` completes.
The `cancel, pushEnd` suffix is also what a failed push itself
does (`hooks.end_incoming_push` finishes the session with the push's status, then
ends the op):
`RecvLifecycleTest.FailedIncomingPushImmediatelyFailsSessionAndReleasesStagingBeforeDeadline`. -/
theorem trace_push_lease_pins_staging_on_cancel :
    ((sysPush 1).run [.pushBegin, .cancel]).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysPush 1).run [.pushBegin, .cancel, .pushBegin] = none ∧
    ((sysPush 1).run [.pushBegin, .cancel, .pushEnd, .publish]).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `RecvLifecycleTest.IncomingPushLeaseSpansLayerH2dAndBlockAccountingBeforeReleasing`:
even if the H2D copy and its callback finish inside the push stream
(`BlockTransportServerPushSession::StartStream`) before `EndIncomingPush`
runs, the open push lease (`pushes = 1`) keeps `done = false`
and `hasStaging = true` until `pushEnd`. -/
theorem trace_push_lease_outlives_h2d :
    let pre := [.pushBegin, .h2dBegin, .h2dIssue true, .h2dReady, .h2dDone true]
    ((sysPush 1).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysPush 1).run (pre ++ [.publish]) = none ∧
    ((sysPush 1).run (pre ++ [.pushEnd, .publish])).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some true) := by
  decide

/-- `ControlHandshakeTest.ExpiredReceiveKeepsStagingUntilHandshakeEnds`, plus the
idle push-plan cases `RecvLifecycleTest.ReceiveWithoutTrafficFailsAtItsDeadline`,
`DemandStagingTest.UnregisteringIdleReceiverReleasesPlanAtOnce`, and
`DemandStagingTest.DemandStagedReceiverPlanUnregistersWhenItSettles`: on
`sysLoad 1`, a deadline while the pull handshake is still pending keeps staging
pinned until `pullReply` ends the handshake op; on `sysPush 1` with no admitted
push, cancelling or timing out settles and releases staging at once. -/
theorem trace_deadline_during_handshake :
    ((sysLoad 1).run [.cancel]).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sysLoad 1).run [.cancel, .publish] = none ∧
    ((sysLoad 1).run [.cancel, .pullReply false, .publish]).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) ∧
    ((sysPush 1).run [.cancel, .publish]).map
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

/-- Mutant for `NoOpLeak`: `ExecuteLayerH2d` returns early on the second
`if (done_ || draining_)` check without calling `EndRecvOpLocked()`. A cancel
between the two locks leaks the op: `inFlight` stays positive with no enabled
drain event, so the session never settles (`SettleSafe` holds vacuously,
`NoOpLeak` catches it). -/
def h2dIssueLeak (ok : Bool) (s : Recv) : Option Recv :=
  if s.pending = 0 then none
  else
    let s := { s with pending := s.pending - 1 }
    if s.life.done = true ∨ s.life.draining = true then
      some s
    else if ok then
      some { s with issued := s.issued + 1 }
    else
      some { s with life := (s.life.finishLocked false).endOpLocked }

#guard (match ModelCheck.check ⟨initPush 1, fun s e => match e with
          | .h2dIssue ok => h2dIssueLeak ok s
          | e => step s e⟩ events violates 6 with
        | .counterexample _ => true
        | _ => false)

/-! ## Failure phase (`TransferDetails`)

As on the send side: `FinishLocked(status, phase)` latches `failed_phase_` from
the explicit `kH2dPhase` in the `ExecuteLayerH2d` paths (`FinishLocked(future_or.status(), kH2dPhase)`
and `self->FinishLocked(status_or.status(), kH2dPhase)`) and otherwise from
`CurrentPhaseLocked` (`handshake_pending_` → `connect`, `network_completed_` →
`h2d`, else `receive`); `Details()` reports it (`CompleteReadWithDetails`:
`transfer_details_[session->req_id()] = session->Details()`). Not state;
`phase` evaluates `CurrentPhaseLocked` on the model (`pullPending` is
`handshake_pending_`, `networkCompleted` is `network_completed_`). -/

/-- The three phases of `TransferReceiveSession::CurrentPhaseLocked`. -/
inductive Phase
  | connect | receive | h2d
  deriving Repr, DecidableEq

/-- `CurrentPhaseLocked`. -/
def phase (s : Recv) : Phase :=
  if s.pullPending then .connect
  else if s.layersAccounted = s.numLayers then .h2d
  else .receive

/-- The phase `FinishLocked` latches when `e` fails in `s` while
`s.life.statusOk = true` (first error wins: `if (!status.ok() && status_.ok())`):
the H2D dispatch failure and the H2D completion callback pass `kH2dPhase`; the
pull reply (`self->FinishLocked(pull_status)`) and every external `Finish(error)`
take the current phase. At `numLayers = 0`, `phase` says `h2d` where C++ says
`receive` (`network_completed_` is never set there, see `networkCompleted`). -/
def failedPhase (s : Recv) : Ev → Phase
  | .h2dIssue false => .h2d
  | .h2dDone false => .h2d
  | _ => phase s

/-- `RecvLifecycleTest.DetailsSayAReceiveTimedOutWaitingForData`: a push-plan
receive with no traffic expires in `receive` and is reported failed. -/
theorem trace_details_receive_timed_out_waiting_for_data :
    ((sysPush 1).run []).map (fun s => failedPhase s .cancel) = some .receive ∧
    ((sysPush 1).run [.cancel, .publish]).map (fun s => (s.life.done, s.published)) =
      some (true, some false) := by
  decide

/-- `RecvLifecycleTest.DetailsSayAFailedUploadFailedInH2dWhileDataArrives`:
layer 0's H2D copy fails before its blocks were accounted. `CurrentPhaseLocked`
would say `receive` (`network_completed_` is false), but the callback passes
`kH2dPhase` (`self->FinishLocked(status_or.status(), kH2dPhase)`), so the
failure is reported in `h2d`. -/
theorem trace_details_failed_upload_in_h2d :
    let pre := [.h2dBegin, .h2dIssue true, .h2dReady]
    ((sysPush 1).run pre).map (fun s => (phase s, failedPhase s (.h2dDone false))) =
      some (.receive, .h2d) ∧
    ((sysPush 1).run (pre ++ [.h2dDone false, .publish])).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `ControlHandshakeTest.RejectedPullReportsWhyAndWhichProducerRejectedIt` and
`ControlHandshakeTest.RefusedConnectionKeepsItsStatusCode`: a pull the producer
rejects, or whose connection is refused, fails in `connect`
(`handshake_pending_` still set when `on_response` runs
`self->FinishLocked(pull_status)`) and is reported failed. The two tests differ
only in the status code and message carried by `TransferDetails`
(`FailedPrecondition` with the producer's message on `response->status != 0` vs.
the transport's own `response.status().code()`), which the model does not
distinguish: both are `pullReply false`. -/
theorem trace_details_rejected_pull :
    ((sysLoad 1).run []).map (fun s => (phase s, failedPhase s (.pullReply false))) =
      some (.connect, .connect) ∧
    ((sysLoad 1).run [.pullReply false, .publish]).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide


end Recv

end TpuSyncVerify.Transfer.PrefillDecode

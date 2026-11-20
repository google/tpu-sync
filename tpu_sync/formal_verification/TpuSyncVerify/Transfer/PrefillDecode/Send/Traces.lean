import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.Send.Progress
import TpuSyncVerify.Transfer.PrefillDecode.Send.Frame

/-! # `Send`: replay of C++ tests, bounded searches, failure phase

Part of `TpuSyncVerify.Transfer.PrefillDecode.Send`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Replay and bounded search

Replay traces (`decide`) paired with `TransferSendSession` / manager GTests
plus bounded searches and mutants. Host-source (`BlockLocation::kHost`) sends
replay from `tr_host_source_prefix` (`docs/transfer/prefill_decode.md` §Boundaries). -/

/-- Two layers, pushes succeed, published `done_sending`. -/
def tr_normal : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext,
   .h2hDone true, .h2hDone true, .publish]

/-- Model-only: two-layer send chains pushes strictly in order and publishes `done_sending`. -/
theorem trace_normal :
    ((sys 2).run tr_normal).map
      (fun s => (s.life.done, s.published, s.h2hOk, s.life.inFlight)) =
      some (true, some true, 2, 0) := by
  decide

/-- One-layer session test with synchronous D2H and synchronous `H2hWriteDirectAsync`. -/
def tr_one_layer_sync_push : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .h2hDone true, .sendNext]

/-- One-layer session test with `SetManualD2h` and synchronous `H2hWriteDirectAsync`. -/
def tr_one_layer_manual_copy : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .h2hDone true, .sendNext]

/-- One-layer session test with both `SetManualD2h` and `SetManualH2h` (`sendNext` precedes `h2hDone`). -/
def tr_one_layer_async_push : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .sendNext, .h2hDone true]

/-- `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeDone`,
`TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnections`, `TransferSendSessionTest.FinishingOneSendSessionDoesNotShutdownSharedStagingAllocator`,
`TransferSendSessionTest.ReleaseStagingBlocksClearsD2hLayerFuturesOnDone`, and `TransferSendSessionTest.AsyncCallbacksKeepSessionAliveAfterCallerDropsSharedPtr`:
one-layer send settles OK and releases staging under either callback order. -/
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

/-- Deadline sweep finishes an idle send; poll publishes failure. -/
def tr_idle_cancel : List Ev := [.cancel, .publish]

/-- `SendDrainTest.SendNobodyPulledFailsAtItsDeadline`, `SendDeadlineTest.ExpiredSendSessionFailsInsteadOfReportingDone`,
`ControlHandshakeTest.PullAfterRegistrationDeadlineIsRejected`, `SendLifecycleTest.FailedSendWithoutWorkSettlesImmediately`,
and `SendLifecycleTest.DuplicateRegistrationCannotReplaceLiveOffer`: idle send settles on `cancel`, publishes failure, and rejects late pulls. -/
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

/-- `ValidateAndBeginPull` claim alone. -/
def tr_pull : List Ev := [.beginPull]

/-- `ControlHandshakeTest.DuplicatePullIsRejectedBeforeAcknowledgement`: once
`beginPull` claims `pull_started_ = true`, duplicate pulls are rejected. -/
theorem trace_duplicate_pull_rejected :
    ((sys 1).run tr_pull).map (fun s => (s.pullStarted, s.started)) = some (true, false) ∧
    (sys 1).run [.beginPull, .beginPull] = none ∧
    (sys 1).run [.beginPull, .start, .beginPull] = none := by
  decide

/-- `Finish(error)` between pull claim and `StartPush`. -/
def tr_cancel_before_start : List Ev := [.beginPull, .cancel]

/-- `Finish(error)` before `StartPush`, followed by late `StartPush` refusal (`Finish` no-op). -/
def tr_cancel_then_late_start : List Ev := [.beginPull, .cancel, .cancel]

/-- `TransferSendSessionTest.FinishBeforeStartPushDoesNotAcquireStagingOrAccessHbm`,
`TransferSendSessionTest.CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator`, and
`TransferSendSessionTest.InvalidStartPushArgumentsFailSessionImmediatelyWithoutHang` (plus `HandlePullStream` spawn failure):
`cancel` after `beginPull` settles immediately and refuses `.start`. -/
theorem trace_pull_spawn_failure_cleanup :
    ((sys 1).run (tr_cancel_before_start ++ [.publish])).map
      (fun s => (s.pullStarted, s.started, s.life.done, s.life.hasStaging, s.published)) =
      some (true, false, true, false, some false) ∧
    (sys 1).run (tr_cancel_before_start ++ [.start]) = none ∧
    ((sys 1).run tr_cancel_then_late_start).map
      (fun s => (s.started, s.life.done, s.life.hasStaging)) = some (false, true, false) := by
  decide

/-- Two layers; deadline fires after layer 0's copy is issued. -/
def tr_deadline_during_copy : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .cancel, .d2hReady, .wake true, .d2hEnd, .publish]

/-- Model-only (2-layer form of `tr_expired_while_copying`): deadline during D2H
keeps staging pinned and skips layer 1 until layer 0's ops drain. -/
theorem trace_deadline_during_copy :
    ((sys 2).run (tr_deadline_during_copy.take 5)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sys 2).run (tr_deadline_during_copy.take 5 ++ [.d2hBegin]) = none ∧
    ((sys 2).run tr_deadline_during_copy).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- Layer 0's push fails with layer 1 already queued on `push_pool`. -/
def tr_push_fails : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .d2hEnd, .wake true, .h2hIssue, .sendNext,
   .d2hReady, .d2hEnd, .wake true, .h2hDone false, .h2hIssue, .publish]

/-- Model-only: when layer 0's push fails after both D2H copies finished,
`h2hIssue`'s draining check drops layer 1 and the send publishes failed. -/
theorem trace_push_fails :
    ((sys 2).run tr_push_fails).map
      (fun s => (s.life.done, s.h2hIssued, s.published)) = some (true, 1, some false) := by
  decide

/-- `Finish(error)` after the last push finished OK while `d2hEnd` is still pending. -/
def tr_cancel_after_ok_finish : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .h2hIssue, .sendNext,
   .h2hDone true, .cancel, .d2hEnd, .publish]

/-- Model-only: `finishOnceLocked` ignores a `cancel` arriving between the last
OK `h2hDone` and a trailing `d2hEnd`. -/
theorem trace_cancel_after_ok_finish :
    ((sys 1).run tr_cancel_after_ok_finish).map
      (fun s => (s.life.statusOk, s.published)) = some (true, some true) := by
  decide

/-- `StartPush` on a zero-layer send. -/
def tr_zero_layers : List Ev := [.beginPull, .start]

/-- `TransferSendSessionTest.ZeroLayerSessionCompletesAndReleasesStagingWithoutHang`,
`ControlHandshakeTest.RegisteredPullIsAcknowledged`, `ControlHandshakeTest.PullAheadOfRegistrationIsAcknowledgedOnceRegistered`,
and `ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged`: zero-layer send settles OK inside `StartPush`. -/
theorem trace_zero_layers :
    ((sys 0).run tr_zero_layers).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, true, false) ∧
    ((sys 0).run (tr_zero_layers ++ [.publish])).map (fun s => s.published) = some (some true) := by
  decide

/-- Layer 0's H2H push and layer 1's D2H copy both in flight when deadline fires. -/
def tr_drain_h2h_and_d2h : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext,
   .cancel, .h2hDone true, .d2hReady, .wake true, .d2hEnd]

/-- `TransferSendSessionTest.DoneGuaranteesAllResourcesReleasedAndNoHbmOrTransportAccessAfterDone`:
with layer 0 pushing and layer 1 copying at `cancel`, staging stays pinned until
both drain without issuing layer 1's push. -/
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

/-- Layer 0's D2H copy fails while layer 1's D2H copy is still in flight. -/
def tr_failed_d2h_waits_for_other_layer : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake false, .d2hEnd, .d2hReady, .d2hEnd, .publish]

/-- `SendDrainTest.FailedLayerWaitsForTheOtherLayersCopies`: when layer 0's D2H
fails, staging stays pinned until layer 1's D2H finishes. -/
theorem trace_failed_d2h_waits_for_other_layer :
    let pre := tr_failed_d2h_waits_for_other_layer.take 9
    ((sys 2).run pre).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging)) = some (true, false, true) ∧
    (sys 2).run (pre ++ [.publish]) = none ∧
    ((sys 2).run tr_failed_d2h_waits_for_other_layer).map
      (fun s => (s.life.done, s.life.hasStaging, s.published)) = some (true, false, some false) := by
  decide

/-- `Finish(error)` while the last push is in flight, followed by `h2hDone true`. -/
def tr_ok_after_cancel_keeps_failure : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .d2hEnd, .wake true, .h2hIssue, .sendNext,
   .cancel, .h2hDone true, .publish]

/-- `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush` and
`TransferSendSessionTest.PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush`:
`cancel` mid-push freezes the failed status even when `h2hDone true` completes later. -/
theorem trace_ok_after_cancel_keeps_failure :
    ((sys 1).run (tr_ok_after_cancel_keeps_failure.take 11)).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging)) = some (true, false, false) ∧
    ((sys 1).run tr_ok_after_cancel_keeps_failure).map
      (fun s => (s.life.done, s.life.statusOk, s.published)) = some (true, false, some false) := by
  decide

/-- Three layers; layer 0's copy is issued, layer 1's dispatch fails, and layer 0 drains. -/
def tr_d2h_dispatch_fail_drains_issued : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue false,
   .d2hReady, .wake true, .d2hEnd]

/-- `TransferSendSessionTest.D2hDispatchFailureFailsUnissuedLayersAndDrainsIssuedCopies`:
failed `D2hSyncDispatch` on layer 1 skips layer 2 and pins staging until layer 0's D2H drains. -/
theorem trace_d2h_dispatch_fail_drains_issued :
    ((sys 3).run (tr_d2h_dispatch_fail_drains_issued.take 6)).map
      (fun s => (s.life.draining, s.life.done, s.life.hasStaging, s.d2hIssued)) =
      some (true, false, true, 1) ∧
    ((sys 3).run tr_d2h_dispatch_fail_drains_issued).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.h2hIssued)) =
      some (true, false, false, 0) := by
  decide

/-- Three layers; layer 2's failed dispatch ends the last op after `cancel`. -/
def tr_failed_dispatch_ends_send : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true, .d2hBegin,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .sendNext, .h2hDone true,
   .cancel, .d2hReady, .wake true, .d2hEnd, .d2hIssue false]

/-- `TransferSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenFailedDispatchEndsSend`:
when `cancel` and earlier layers drain while layer 2 is in `d2hBegin`, `d2hIssue false` settles the send. -/
theorem trace_failed_dispatch_ends_send :
    ((sys 3).run (tr_failed_dispatch_ends_send.take 17)).map
      (fun s => (s.life.draining, s.life.done, s.life.inFlight)) = some (true, false, 1) ∧
    ((sys 3).run tr_failed_dispatch_ends_send).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.h2hIssued)) =
      some (true, false, false, 1) := by
  decide

/-- One layer; `D2hSyncDispatch` fails on layer 0 and `fail_unissued_layers` wakes `SendNextLayer(0)`. -/
def tr_first_dispatch_fails : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue false, .wake false]

/-- `TransferSendSessionTest.FailedD2hDispatchOnFirstLayerSettlesAtOnce`: failed
first-layer dispatch resolves layer 0's promise so `wake false` settles the send. -/
theorem trace_first_dispatch_fails :
    ((sys 1).run (tr_first_dispatch_fails.take 4)).map
      (fun s => (s.life.draining, s.life.done, s.life.inFlight)) = some (true, false, 1) ∧
    ((sys 1).run tr_first_dispatch_fails).map
      (fun s => (s.life.done, s.life.statusOk, s.life.hasStaging, s.h2hIssued)) =
      some (true, false, false, 0) := by
  decide

/-- One layer; `cancel` while the push is in flight, followed by `h2hDone false`. -/
def tr_cancel_then_push_fails : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake true, .d2hEnd,
   .h2hIssue, .sendNext, .cancel, .h2hDone false]

/-- `TransferSendSessionTest.CancelledSendWhosePushThenFailsKeepsCancelled`:
after `cancel` mid-push, `h2hDone false` preserves the cancelled status and settles the send. -/
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

/-- Mutant: omitting `SendNextLayer`'s `l >= base_->num_layers()` bounds check
claims an op for `numLayers` whose future never resolves (`NoOpLeak`). -/
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

/-- Mutant: omitting `fail_unissued_layers` in `StartPush` strands `SendNextLayer(0)`
with `inFlight = 1` when `cancel` arrives before layer 0 is issued (`NoOpLeak`). -/
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

Evaluates `TransferSendSession::CurrentPhaseLocked` (`phase`) and the
`failed_phase_` latched by `FinishLocked(status, phase)` (`failedPhase`) for
`Details()` reporting. -/

/-- The four phases of `TransferSendSession::CurrentPhaseLocked` (`kAwaitPullPhase` .. `kPushPhase`). -/
inductive Phase
  | awaitPull | staging | d2h | push
  deriving Repr, DecidableEq

/-- `CurrentPhaseLocked` (`awaitPull` before `started`, `d2h` while any D2H
future is unissued or unready, else `push`). -/
def phase (s : Send) : Phase :=
  if s.started = false then .awaitPull
  else if s.d2hIssued < s.numLayers ∨ s.d2hReady < s.d2hIssued then .d2h
  else .push

/-- Phase latched by `FinishLocked` when `e` fails (explicit `kD2hPhase` on
`wake false`, `kPushPhase` on `h2hDone false`, else `CurrentPhaseLocked`). -/
def failedPhase (s : Send) : Ev → Phase
  | .wake false => .d2h
  | .h2hDone false => .push
  | _ => phase s

/-- `SendLifecycleTest.DetailsSayAnUnpulledSendTimedOutAwaitingItsPull`: an
unpulled send expires in `.awaitPull` and publishes failure. -/
theorem trace_details_unpulled_send_awaits_pull :
    ((sys 1).run []).map (fun s => (phase s, failedPhase s .cancel)) =
      some (.awaitPull, .awaitPull) ∧
    ((sys 1).run tr_idle_cancel).map (fun s => s.published) = some (some false) := by
  decide

/-- One layer; deadline fires with D2H copy and `SendNextLayer(0)` in flight. -/
def tr_expired_while_copying : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .cancel, .d2hReady, .wake true, .d2hEnd, .publish]

/-- `SendDrainTest.DetailsSayAnExpiredSendWasStillCopying`, `SendDrainTest.ExpiredSendKeepsItsStagingUntilTheCopyEnds`,
`TransferSendSessionTest.SendSessionImplementsTransferSessionInterface`, and `SendLifecycleTest.SuccessCannotOverrideAnEarlierFailure`:
deadline during D2H latches `.d2h` and waits for the copy before settling. -/
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

/-- One layer; D2H copy finishes with an error. -/
def tr_failed_copy : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hReady, .wake false, .d2hEnd, .publish]

/-- `SendDrainTest.DetailsCarryTheErrorOfAFailedCopy`: `wake false` latches `.d2h`
explicitly and publishes failure once `d2hEnd` runs. -/
theorem trace_details_failed_copy :
    let pre := tr_failed_copy.take 5
    ((sys 1).run pre).map (fun s => failedPhase s (.wake false)) = some .d2h ∧
    ((sys 1).run tr_failed_copy).map
      (fun s => (s.life.done, s.published)) = some (true, some false) := by
  decide

/-- Two layers; layer 0's push fails while layer 1's D2H copy still runs. -/
def tr_failed_push_while_copies_run : List Ev :=
  [.beginPull, .start, .d2hBegin, .d2hIssue true, .d2hBegin, .d2hIssue true,
   .d2hReady, .wake true, .d2hEnd, .h2hIssue, .h2hDone false, .sendNext,
   .d2hReady, .d2hEnd, .publish]

/-- `SendDrainTest.DetailsSayAFailedPushFailedInPushWhileCopiesRun`: `h2hDone false`
latches explicit `.push` while `phase = .d2h`, then drains layer 1's D2H copy. -/
theorem trace_details_failed_push_while_copies_run :
    let pre := tr_failed_push_while_copies_run.take 10
    ((sys 2).run pre).map (fun s => (phase s, failedPhase s (.h2hDone false))) =
      some (.d2h, .push) ∧
    ((sys 2).run (pre ++ [.h2hDone false])).map
      (fun s => (s.life.draining, s.life.done)) = some (true, false) ∧
    ((sys 2).run tr_failed_push_while_copies_run).map
      (fun s => (s.life.done, s.published)) = some (true, some false) := by
  decide

/-! ### Host-resident source (`BlockLocation::kHost`) -/

/-- Completed-D2H prefix matching the post-`StartPush` state of `BlockLocation::kHost`. -/
def tr_host_source_prefix (n : Nat) : List Ev :=
  [.beginPull, .start] ++ (List.replicate n [.d2hBegin, .d2hIssue true, .d2hReady, .d2hEnd]).flatten

/-- `tr_host_source_prefix` reaches `started`, `queued = 1`, `inFlight = 1`,
`d2hIssued = d2hReady = d2hRetired = n`, and `phase = .push` (`docs/transfer/prefill_decode.md` §Boundaries). -/
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

/-- Three-layer host source: issues all pushes before any completes. -/
def tr_host_source_three_pushes : List Ev :=
  tr_host_source_prefix 3 ++
    [.wake true, .h2hIssue, .sendNext, .wake true, .h2hIssue, .sendNext,
     .wake true, .h2hIssue, .sendNext, .h2hDone true, .h2hDone true, .h2hDone true]

/-- Two-layer host source: layer 0's push fails, then layer 1's push drains. -/
def tr_host_source_push_fails_then_drains : List Ev :=
  tr_host_source_prefix 2 ++
    [.wake true, .h2hIssue, .sendNext, .wake true, .h2hIssue, .sendNext,
     .h2hDone false, .h2hDone true]

/-- One-layer host source: deadline fires with the push in flight. -/
def tr_host_source_deadline_while_pushing : List Ev :=
  tr_host_source_prefix 1 ++ [.wake true, .h2hIssue, .sendNext, .cancel, .h2hDone true]

/-- Two-layer host source under a manager: both pushes succeed and publish `done_sending`. -/
def tr_host_source_two_pushes_published : List Ev :=
  tr_host_source_prefix 2 ++
    [.wake true, .h2hIssue, .sendNext, .wake true, .h2hIssue, .sendNext,
     .h2hDone true, .h2hDone true, .publish]

/-- Two-layer host source under a manager: synchronous `H2hWriteDirectAsync` failure on layer 0. -/
def tr_host_source_injected_push_failure : List Ev :=
  tr_host_source_prefix 2 ++ [.wake true, .h2hIssue, .h2hDone false, .sendNext, .publish]

/-- `TransferSendSessionTest.HostSourcePushesRegisteredHostBlocksWithoutStagingOrD2h`,
`TransferSendSessionTest.HostSourceFailedLayerPushFailsTheSendOnceInFlightLayersDrain`, `TransferSendSessionTest.HostSourceDeadlineWhilePushingIsAttributedToThePushPhase`,
`TransferSendSessionTest.HostSourceWithNoLayersSettlesWithoutAPush`, `TransferSendSessionTest.HostSourceFinishedBeforeStartPushIssuesNothing`,
`ControlHandshakeTest.StartReadOverPeregrinePropagatesConsumerPortAndCompletesPush`, `HostMemoryTransferTest.AReadNobodyPullsExpiresAwaitingThePull`, and `HostMemoryTransferTest.AFailedPushFailsTheSendAndExpiresTheReceive`. -/
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


end Send

end TpuSyncVerify.Transfer.PrefillDecode

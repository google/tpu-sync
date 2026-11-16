import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Progress

/-! # `PushWindow`: replay of upstream tests, searches, mutants

Part of `TpuSyncVerify.Transfer.PrefillDecode.PushWindow`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.PushWindow

open TpuSyncVerify TpuSyncVerify.Transfer

namespace S

/-- Every event, with both outcomes where it has one. -/
def allEvents : List Ev :=
  [.wake true, .wake false, .admit, .issue, .sendNext, .h2hDone true, .h2hDone false, .cancel,
   .publish]

/-- A unit's copy completes, and its push is admitted and issued. -/
def pushUnit : List Ev := [.wake true, .admit, .issue]

/-! ## Replay -/

/-- `TransferSendSessionTest.PushWindowCapsTheGroupsPushingAtOnce`: four units
under a window of two. The third unit parks while two pushes are in flight and
cannot be issued; the first completion resumes it, the fourth parks in turn,
and the send completes once every push has. -/
theorem trace_window_caps :
    let pre := pushUnit ++ [.sendNext] ++ pushUnit ++ [.sendNext, .wake true, .admit]
    ((sys 4 2).run pre).map (fun s => (s.h2hIssued, s.parked, s.pif)) = some (2, true, 2) ∧
    (sys 4 2).run (pre ++ [.issue]) = none ∧
    ((sys 4 2).run (pre ++ [.h2hDone true, .issue])).map
      (fun s => (s.h2hIssued, s.parked, s.pif)) = some (3, false, 2) ∧
    ((sys 4 2).run (pre ++ [.h2hDone true, .issue, .sendNext, .wake true, .admit, .h2hDone true,
        .issue, .sendNext, .h2hDone true, .h2hDone true, .publish])).map
      (fun s => (s.h2hIssued, s.h2hOk, s.published)) = some (4, 4, some true) := by
  decide

/-- `TransferSendSessionTest.CancelledSendDropsTheGroupWaitingForTheWindow`:
the second of two units parks under a window of one; the send is cancelled,
the push in flight completes without resuming the parked unit, and the send
settles failed with one push issued. -/
theorem trace_cancel_drops_parked :
    let pre := pushUnit ++ [.sendNext, .wake true, .admit]
    ((sys 2 1).run pre).map (fun s => (s.h2hIssued, s.parked)) = some (1, true) ∧
    ((sys 2 1).run (pre ++ [.cancel])).map (fun s => (s.life.draining, s.life.done)) =
      some (true, false) ∧
    ((sys 2 1).run (pre ++ [.cancel, .h2hDone true, .publish])).map
      (fun s => (s.h2hIssued, s.parked, s.published)) = some (1, true, some false) := by
  decide

/-- `TransferSendSessionTest.PushGroupsSplitALargeLayerIntoCopiesAndPushes` with
no window: three units are pushed back to back, each issued as soon as its
copy completes. -/
theorem trace_no_window :
    ((sys 3 0).run (pushUnit ++ [.sendNext] ++ pushUnit ++ [.sendNext] ++ pushUnit ++
        [.sendNext, .h2hDone true, .h2hDone true, .h2hDone true, .publish])).map
      (fun s => (s.h2hIssued, s.pif, s.published)) = some (3, 0, some true) := by
  decide

/-! ## Search

Every reachable state of these instances is visited (`.safe`); the theorems in
`Invariant.lean` and `Progress.lean` are the guarantee for every instance. -/

#guard ModelCheck.check (sys 1 0) allEvents violates 40 = .safe
#guard ModelCheck.check (sys 3 0) allEvents violates 60 = .safe
#guard ModelCheck.check (sys 3 1) allEvents violates 60 = .safe
#guard ModelCheck.check (sys 3 2) allEvents violates 60 = .safe
#guard ModelCheck.check (sys 4 2) allEvents violates 80 = .safe

/-! ## Mutants -/

/-- Mutant for `NoStall`: a full window parks the unit but `AdmitPush` still
chains into `SendNextUnit(u + 1)`. The next unit can park over the first,
which is never pushed, and the send stops with nothing in flight. -/
def stepParkKeepsChaining (s : S) : Ev → Option S
  | .admit =>
    if s.admitting = 0 then none
    else
      let s := { s with admitting := s.admitting - 1 }
      if s.life.draining = true then some s.endOp
      else if 0 < s.window ∧ s.window ≤ s.pif then
        some { s with parked := true, chaining := s.chaining + 1 }
      else some { s with pif := s.pif + 1, issuing := s.issuing + 1 }
  | e => s.step e

#guard match ModelCheck.check ⟨init 3 1, stepParkKeepsChaining⟩ allEvents violates 40 with
  | .counterexample _ => true
  | _ => false

/-- Mutant for `ParkedHasWaker`: the push callback never resumes the parked
unit (`if (self->parked_unit_.has_value() && !self->draining_)` removed). -/
def stepNoResume (s : S) : Ev → Option S
  | .h2hDone ok =>
    if s.h2hRetired < s.h2hIssued then
      let s := { s with h2hRetired := s.h2hRetired + 1, pif := s.pif - 1 }
      if ok then
        let s := { s with h2hOk := s.h2hOk + 1 }
        some (if s.h2hOk = s.numUnits then (s.finish true).endOp else s.endOp)
      else some (s.finish false).endOp
    else none
  | e => s.step e

#guard match ModelCheck.check ⟨init 2 1, stepNoResume⟩ allEvents violates 40 with
  | .counterexample _ => true
  | _ => false

/-- Mutant for `WindowBound`: the window check is off by one
(`push_window_ < pushes_in_flight_`). -/
def stepWindowOffByOne (s : S) : Ev → Option S
  | .admit =>
    if s.admitting = 0 then none
    else
      let s := { s with admitting := s.admitting - 1 }
      if s.life.draining = true then some s.endOp
      else if 0 < s.window ∧ s.window < s.pif then some { s with parked := true }.endOp
      else some { s with pif := s.pif + 1, issuing := s.issuing + 1 }
  | e => s.step e

#guard match ModelCheck.check ⟨init 3 1, stepWindowOffByOne⟩ allEvents violates 40 with
  | .counterexample _ => true
  | _ => false

end S

end TpuSyncVerify.Transfer.PrefillDecode.PushWindow

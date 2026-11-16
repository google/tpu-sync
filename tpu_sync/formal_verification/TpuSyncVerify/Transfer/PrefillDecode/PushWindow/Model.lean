import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `PushWindow`: state, events, `step`, properties

Part of `TpuSyncVerify.Transfer.PrefillDecode.PushWindow`; the module docstring
and source map live in `PushWindow.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.PushWindow

open TpuSyncVerify TpuSyncVerify.Transfer

/-- The push chain of one `TransferSendSession` over its units (`units_`),
from the moment `StartPush` has armed `SendNextUnit(0)`. -/
structure S where
  numUnits : Nat
  /-- `push_window_`; `0` is no cap. -/
  window : Nat
  life : Lifecycle := {}
  /-- `SendNextUnit(u)` calls that took their op (`++in_flight_`). -/
  queued : Nat := 0
  /-- `SendNextUnit`'s `unit_future->OnReady` callbacks that ran. -/
  woken : Nat := 0
  /-- `AdmitPush` tasks on `push_pool`, carrying the callback's op (ghost). -/
  admitting : Nat := 0
  /-- `parked_unit_.has_value()`. -/
  parked : Bool := false
  /-- Tasks about to run `IssuePush`, carrying an op: an `AdmitPush` past its
  admission, or the task a push callback scheduled to resume the parked unit
  (ghost). -/
  issuing : Nat := 0
  /-- Tasks past `IssuePush`, carrying their op into `SendNextUnit(u + 1)` and
  their `absl::Cleanup end_op` (ghost). -/
  chaining : Nat := 0
  /-- `pushes_in_flight_`. -/
  pif : Nat := 0
  /-- Pushes issued (`H2hWriteDirectAsync`) (ghost). -/
  h2hIssued : Nat := 0
  /-- Push callbacks that ran (ghost). -/
  h2hRetired : Nat := 0
  /-- Pushes that completed OK: `numUnits - remaining_h2h_units_`. -/
  h2hOk : Nat := 0
  published : Option Bool := none
  underflow : Bool := false
  deriving Repr, DecidableEq, Hashable

/-- The events, one per C++ callback or lock scope (see the events table). -/
inductive Ev where
  | wake (ok : Bool)
  | admit
  | issue
  | sendNext
  | h2hDone (ok : Bool)
  | cancel
  | publish
  deriving Repr, DecidableEq

namespace S

/-- `StartPush` has armed the chain: `SendNextUnit(0)` took its op and waits on
unit 0's D2H future (`Send.lean`'s `start`, its A6). -/
def init (numUnits window : Nat) : S :=
  { numUnits, window, life := { inFlight := 1 }, queued := 1 }

/-- `EndSendOpLocked`: `--in_flight_` and settle; an underflow is recorded, as in
`Send.lean`, so that its absence can be proved. -/
def endOp (s : S) : S :=
  { s with life := if s.life.inFlight = 0 then s.life else s.life.endOpLocked,
           underflow := s.underflow || decide (s.life.inFlight = 0) }

/-- `Finish(status)` / `FinishLocked(status)`: the first call decides. -/
def finish (ok : Bool) (s : S) : S := { s with life := s.life.finishOnceLocked ok }

/-- `SendNextUnit(u)` for `u = queued` up to its lock
(`if (draining_ || u >= d2h_unit_futures_.size()) return;` .. `++in_flight_;`):
takes an op unless `u` is past the last unit or the send is draining. -/
def trySendNext (s : S) : S :=
  if s.queued < s.numUnits ∧ s.life.draining = false then
    { s with queued := s.queued + 1, life := { s.life with inFlight := s.life.inFlight + 1 } }
  else s

/-- The resume decision in the push callback, under its lock
(`if (self->parked_unit_.has_value() && !self->draining_)`): the parked unit
takes a window slot (`++self->pushes_in_flight_`) and an op for the task that
will issue it (`++self->in_flight_`). -/
def resume (s : S) : S :=
  if s.parked = true ∧ s.life.draining = false then
    { s with parked := false, pif := s.pif + 1, issuing := s.issuing + 1,
             life := { s.life with inFlight := s.life.inFlight + 1 } }
  else s

/-! ## Events -/

/-- `SendNextUnit`'s `unit_future->OnReady` callback for unit `woken`. A failed
copy finishes the send; a draining send pushes nothing; otherwise the op is
carried into an `AdmitPush` task (`std::move(end_op).Cancel()` then
`push_pool()->Schedule`). The D2H copy itself, and whether its future resolved
from the copy or from `fail_unissued_units`, is `Send.lean`'s (A1). -/
def wake (ok : Bool) (s : S) : Option S :=
  if s.woken < s.queued then
    let s := { s with woken := s.woken + 1 }
    if ok = false then some (s.finish false).endOp
    else if s.life.draining = true then some s.endOp
    else some { s with admitting := s.admitting + 1 }
  else none

/-- `AdmitPush` up to its unlock: a draining send drops the unit; a full window
parks it (`parked_unit_ = u`) and the task ends; otherwise it takes a window
slot (`++pushes_in_flight_`) and goes on to `IssuePush`. -/
def admit (s : S) : Option S :=
  if s.admitting = 0 then none
  else
    let s := { s with admitting := s.admitting - 1 }
    if s.life.draining = true then some s.endOp
    else if 0 < s.window ∧ s.window ≤ s.pif then some { s with parked := true }.endOp
    else some { s with pif := s.pif + 1, issuing := s.issuing + 1 }

/-- `IssuePush`: a send that started draining since the unit was admitted
returns its window slot (`--pushes_in_flight_`) and pushes nothing; otherwise
the push takes its own op (`++in_flight_`) and is issued
(`H2hWriteDirectAsync`). Either way the task goes on to `SendNextUnit(u + 1)`. -/
def issue (s : S) : Option S :=
  if s.issuing = 0 then none
  else
    let s := { s with issuing := s.issuing - 1, chaining := s.chaining + 1 }
    if s.life.draining = true then some { s with pif := s.pif - 1 }
    else some { s with h2hIssued := s.h2hIssued + 1,
                       life := { s.life with inFlight := s.life.inFlight + 1 } }

/-- The task's tail: `SendNextUnit(u + 1)`, then `absl::Cleanup end_op`. -/
def sendNext (s : S) : Option S :=
  if s.chaining = 0 then none
  else some { s with chaining := s.chaining - 1 }.trySendNext.endOp

/-- The push callback in `IssuePush`: `--pushes_in_flight_`; a failed push
finishes the send; an OK one resumes the parked unit, if any, and the last OK
push (`remaining_h2h_units_.fetch_sub(1) == 1`) finishes the send OK; then
`absl::Cleanup end_op`. -/
def h2hDone (ok : Bool) (s : S) : Option S :=
  if s.h2hRetired < s.h2hIssued then
    let s := { s with h2hRetired := s.h2hRetired + 1, pif := s.pif - 1 }
    if ok then
      let s := { s with h2hOk := s.h2hOk + 1 }.resume
      some (if s.h2hOk = s.numUnits then (s.finish true).endOp else s.endOp)
    else some (s.finish false).endOp
  else none

/-- `Finish(error)` from outside the data path; enabled at any time. -/
def cancel (s : S) : Option S := some (s.finish false)

/-- `CompleteReadWithDetails` publishes a settled send. -/
def publish (s : S) : Option S :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk }
  else none

def step (s : S) : Ev → Option S
  | .wake ok => s.wake ok
  | .admit => s.admit
  | .issue => s.issue
  | .sendNext => s.sendNext
  | .h2hDone ok => s.h2hDone ok
  | .cancel => s.cancel
  | .publish => s.publish

/-- A send of `n` units under push window `w`. -/
def sys (n w : Nat) : System S Ev := ⟨init n w, step⟩

/-! ## Properties -/

def SettleSafe (s : S) : Prop := s.life.done = true → s.life.inFlight = 0

/-- A settled send has no callback, task or push outstanding. -/
def Drained (s : S) : Prop :=
  s.life.done = true →
    s.woken = s.queued ∧ s.admitting = 0 ∧ s.issuing = 0 ∧ s.chaining = 0 ∧
      s.h2hRetired = s.h2hIssued

def StagingIntegrity (s : S) : Prop := s.life.hasStaging = !s.life.done

def SettlesPromptly (s : S) : Prop :=
  s.life.draining = true → s.life.inFlight = 0 → s.life.done = true

def NoUnderflow (s : S) : Prop := s.underflow = false

/-- The push window: with a cap, at most `window` units hold a window slot, and
so at most `window` pushes are in flight. -/
def WindowBound (s : S) : Prop :=
  0 < s.window → s.pif ≤ s.window ∧ s.h2hIssued - s.h2hRetired ≤ s.window

/-- A parked unit is never stranded: while the send is not draining, some push
is in flight whose callback will resume it. -/
def ParkedHasWaker (s : S) : Prop :=
  s.parked = true → s.life.draining = false → s.h2hRetired < s.h2hIssued

/-- A send that is not draining always holds an op, so it cannot settle without
finishing. -/
def NoStall (s : S) : Prop := s.life.draining = false → 0 < s.life.inFlight

/-- A send reported done pushed every unit. -/
def Publication (s : S) : Prop := s.published = some true → s.h2hOk = s.numUnits

/-- Events that advance or retire an in-flight op. -/
def drainEvents : List Ev := [.wake true, .admit, .issue, .sendNext, .h2hDone true]

def NoOpLeak (s : S) : Prop :=
  0 < s.life.inFlight → ∃ e ∈ drainEvents, (step s e).isSome = true

def Safe (s : S) : Prop :=
  SettleSafe s ∧ Drained s ∧ StagingIntegrity s ∧ SettlesPromptly s ∧ NoUnderflow s ∧
    WindowBound s ∧ ParkedHasWaker s ∧ NoStall s ∧ Publication s ∧ NoOpLeak s

instance (s : S) : Decidable (Safe s) := by
  unfold Safe SettleSafe Drained StagingIntegrity SettlesPromptly NoUnderflow WindowBound
    ParkedHasWaker NoStall Publication NoOpLeak drainEvents
  infer_instance

/-- Executable negation of `Safe`. -/
def violates (s : S) : Bool := !decide (Safe s)

theorem violates_iff (s : S) : violates s = false ↔ Safe s := by
  simp [violates]

end S

end TpuSyncVerify.Transfer.PrefillDecode.PushWindow

import TpuSyncVerify.Transfer.PrefillDecode.Send.Invariant

/-! # `Send`: frame lemmas for the composing modules

Part of `TpuSyncVerify.Transfer.PrefillDecode.Send`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Frame lemmas

What each event leaves alone, and specs for the events the composed transfer
model (`Pipeline.lean`) attaches memory effects or layer guards to. -/

/-- Unfold `step` for a known event and split every branch. -/
macro "send_cases" hs:ident : tactic =>
  `(tactic| (simp only [step, beginPull, start, d2hBegin, d2hIssue, markD2hReady, d2hEnd, wake, h2hIssue,
      sendNext, h2hDone, cancel, publish] at $hs:ident <;> (repeat' split at $hs:ident)))

theorem step_numLayers {s s' : Send} {e : Ev} (hs : step s e = some s') :
    s'.numLayers = s.numLayers := by
  cases e <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> rfl)

theorem step_pullStarted_mono {s s' : Send} {e : Ev} (hs : step s e = some s')
    (hp : s.pullStarted = true) : s'.pullStarted = true := by
  cases e <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; simpa using hp)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> simp_all)

theorem step_pullStarted_of_ne_beginPull {s s' : Send} {e : Ev} (hs : step s e = some s')
    (he : e ≠ .beginPull) : s'.pullStarted = s.pullStarted := by
  cases e <;> (try exact absurd rfl he) <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> rfl)

theorem step_done_mono {s s' : Send} {e : Ev} (hs : step s e = some s') (hd : s.life.done = true) :
    s'.life.done = true := by
  cases e <;> simp only [step, beginPull, start, d2hBegin, d2hIssue, markD2hReady, d2hEnd, wake, h2hIssue,
      sendNext, h2hDone, cancel, publish, endOp, finish, trySendNext,
      Lifecycle.finishOnceLocked_inFlight, Lifecycle.beginOp_of_done hd, Option.map_none] at hs <;>
    (repeat' split at hs) <;> cases hs <;>
    simp [hd, Lifecycle.endOpLocked_done_mono, Lifecycle.finishOnceLocked_done_mono]

theorem step_published_mono {s s' : Send} {e : Ev} {b : Bool} (hs : step s e = some s')
    (hp : s.published = some b) : s'.published = some b := by
  cases e <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; simpa using hp)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> simp_all)

/-- Only `d2hReady` increments `d2hReady`. -/
theorem step_d2hReady {s s' : Send} {e : Ev} (hs : step s e = some s') (he : e ≠ .d2hReady) :
    s'.d2hReady = s.d2hReady := by
  cases e <;> (try exact absurd rfl he) <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> rfl)

theorem d2hReady_spec {s s' : Send} (hs : step s .d2hReady = some s') :
    s.d2hReady < s.d2hIssued ∧ s' = { s with d2hReady := s.d2hReady + 1 } := by
  simp only [step, markD2hReady] at hs
  split at hs
  · cases hs; exact ⟨‹_›, rfl⟩
  · cases hs

/-- Only `wake` consumes a layer's future. -/
theorem step_woken {s s' : Send} {e : Ev} (hs : step s e = some s') (he : ∀ ok, e ≠ .wake ok) :
    s'.woken = s.woken := by
  cases e <;> (try exact absurd rfl (he _)) <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> rfl)

/-- `wake` consumes exactly the next one. -/
theorem wake_woken {s s' : Send} {ok : Bool} (hs : step s (.wake ok) = some s') :
    s'.woken = s.woken + 1 := by
  simp only [step, wake] at hs
  split at hs
  · (repeat' split at hs) <;> cases hs <;>
       simp only [endOp, finish, Lifecycle.finishOnceLocked_inFlight] <;> (repeat' split) <;> rfl
  · cases hs

theorem h2hDone_guard {s s' : Send} {ok : Bool} (hs : step s (.h2hDone ok) = some s') :
    s.h2hRetired < s.h2hIssued := by
  simp only [step, h2hDone] at hs
  split at hs
  · assumption
  · cases hs

theorem step_d2hIssued_le {s s' : Send} {e : Ev} (hs : step s e = some s') :
    s.d2hIssued ≤ s'.d2hIssued := by
  cases e <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; exact Nat.le_refl _)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> simp)

/-- Only `d2hIssue` counts a copy as issued. -/
theorem step_d2hIssued_of_ne_d2hIssue {s s' : Send} {e : Ev} (hs : step s e = some s')
    (he : ∀ ok, e ≠ .d2hIssue ok) : s'.d2hIssued = s.d2hIssued := by
  cases e <;> (try exact absurd rfl (he _)) <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> rfl)

/-- `d2hIssue` consumes the pending dispatch and counts at most one copy. -/
theorem d2hIssue_spec {s s' : Send} {ok : Bool} (hs : step s (.d2hIssue ok) = some s') :
    s.d2hPending = true ∧ s'.d2hIssued ≤ s.d2hIssued + 1 := by
  simp only [step, d2hIssue] at hs
  split at hs
  · cases hs
  · rename_i hp
    refine ⟨by cases hq : s.d2hPending; exact absurd hq hp; rfl, ?_⟩
    split at hs <;> cases hs <;>
      simp only [endOp, finish, Lifecycle.finishOnceLocked_inFlight] <;> (repeat' split) <;> simp

theorem step_h2hRetired {s s' : Send} {e : Ev} (hs : step s e = some s')
    (he : ∀ ok, e ≠ .h2hDone ok) : s'.h2hRetired = s.h2hRetired := by
  cases e <;> (try exact absurd rfl (he _)) <;> send_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> simp only [endOp, finish, trySendNext, Lifecycle.finishOnceLocked_inFlight] <;>
        (repeat' split) <;> rfl)

theorem h2hDone_h2hRetired {s s' : Send} {ok : Bool} (hs : step s (.h2hDone ok) = some s') :
    s'.h2hRetired = s.h2hRetired + 1 := by
  simp only [step, h2hDone] at hs
  split at hs
  · (repeat' split at hs) <;> cases hs <;>
      simp only [endOp, finish, Lifecycle.finishOnceLocked_inFlight] <;> (repeat' split) <;> rfl
  · cases hs


end Send

end TpuSyncVerify.Transfer.PrefillDecode

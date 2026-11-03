import TpuSyncVerify.Transfer.PrefillDecode.Send.Invariant

/-! # `Send`: progress and eventual settlement

Part of `TpuSyncVerify.Transfer.PrefillDecode.Send`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ### Progress and eventual settlement -/

/-- Remaining steps needed to drain all in-flight operations once draining. -/
def drainRank (s : Send) : Nat :=
  3 * (if s.d2hPending then 1 else 0) + (s.d2hIssued - s.d2hReady) +
    (s.d2hIssued - s.d2hRetired) + (s.queued - s.woken) +
    s.pooled + s.chaining + (s.h2hIssued - s.h2hRetired)

theorem trySendNext_of_draining {s : Send} (hdr : s.life.draining = true) :
    s.trySendNext = s := by
  unfold trySendNext; split <;> simp [Lifecycle.beginOp_of_draining hdr]

@[simp] theorem endOp_draining (s : Send) :
    s.endOp.life.draining = s.life.draining := by
  unfold endOp; split <;> simp

theorem finish_of_draining (ok : Bool) {s : Send} (hdr : s.life.draining = true) :
    s.finish ok = s := by
  simp [finish, Lifecycle.finishOnceLocked_decided ok (Or.inl hdr)]

theorem drainRank_endOp (s : Send) : drainRank s.endOp = drainRank s := by
  unfold endOp; split <;> rfl

/-- While draining and not yet settled, some event in `drainEvents` is enabled,
preserves `draining`, and strictly decreases `drainRank`. -/
theorem drain_step {s : Send} (h : Inv s) (hdr : s.life.draining = true)
    (hnd : s.life.done = false) :
    ∃ e s', step s e = some s' ∧ s'.life.draining = true ∧ drainRank s' < drainRank s := by
  have hif : 0 < s.life.inFlight := by
    cases h0 : s.life.inFlight
    · have := h.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := h.accounted
  have hcnt := h.counters
  unfold Accounted at hacc
  unfold CountersOrdered at hcnt
  by_cases hp : s.d2hPending = true
  · have hs₁ : step s (.d2hIssue true) = some { s with d2hPending := false, d2hIssued := s.d2hIssued + 1 } := by
      simp [step, d2hIssue, hp]
    exact ⟨.d2hIssue true, _, hs₁, by simp [hdr], by simp [drainRank, hp]; omega⟩
  · have hp' : s.d2hPending = false := by
      cases hq : s.d2hPending
      · rfl
      · exact absurd hq hp
    by_cases hrd : s.d2hReady < s.d2hIssued
    · have hs₁ : step s .d2hReady = some { s with d2hReady := s.d2hReady + 1 } := by
        simp [step, markD2hReady, hrd]
      exact ⟨.d2hReady, _, hs₁, by simp [hdr], by simp [drainRank]; omega⟩
    · by_cases hret : s.d2hRetired < s.d2hReady
      · have hs₁ : step s .d2hEnd = some { s with d2hRetired := s.d2hRetired + 1 }.endOp := by
          simp [step, d2hEnd, hret]
        exact ⟨.d2hEnd, _, hs₁, by simp [hdr], by rw [drainRank_endOp]; simp [drainRank]; omega⟩
      · by_cases hwk : s.woken < s.queued
        · by_cases hwr : s.woken < s.d2hReady
          · have hs₁ : step s (.wake true) = some { s with woken := s.woken + 1 }.endOp := by
              simp [step, wake, hwk, hwr, hdr]
            exact ⟨.wake true, _, hs₁, by simp [hdr], by rw [drainRank_endOp]; simp [drainRank]; omega⟩
          · -- layer `woken` was never issued: `fail_unissued_layers` resolved its promise
            have hle : s.d2hIssued ≤ s.woken := by omega
            have hs₁ : step s (.wake false) = some ({ s with woken := s.woken + 1 }.finish false).endOp := by
              simp [step, wake, hwk, hwr, hdr, hp', hle]
            rw [finish_of_draining false (s := { s with woken := s.woken + 1 }) hdr] at hs₁
            exact ⟨.wake false, _, hs₁, by simp [hdr], by rw [drainRank_endOp]; simp [drainRank]; omega⟩
        · by_cases hpool : 0 < s.pooled
          · have hs₁ : step s .h2hIssue = some { s with pooled := s.pooled - 1 }.endOp := by
              simp [step, h2hIssue, Lifecycle.beginOp_of_draining hdr]; omega
            exact ⟨.h2hIssue, _, hs₁, by simp [hdr], by rw [drainRank_endOp]; simp [drainRank]; omega⟩
          · by_cases hch : 0 < s.chaining
            · have hs₁ : step s .sendNext = some { s with chaining := s.chaining - 1 }.endOp := by
                simp [step, sendNext, trySendNext_of_draining (s := { s with chaining := s.chaining - 1 }) hdr]
                omega
              exact ⟨.sendNext, _, hs₁, by simp [hdr], by rw [drainRank_endOp]; simp [drainRank]; omega⟩
            · have hh2h : s.h2hRetired < s.h2hIssued := by
                simp [hp'] at hacc; omega
              have hs₁ : step s (.h2hDone true) =
                  some { s with h2hRetired := s.h2hRetired + 1, h2hOk := s.h2hOk + 1 }.endOp := by
                simp [step, h2hDone, hh2h,
                  finish_of_draining true (s := { s with h2hRetired := s.h2hRetired + 1, h2hOk := s.h2hOk + 1 }) hdr]
              exact ⟨.h2hDone true, _, hs₁, by simp [hdr], by rw [drainRank_endOp]; simp [drainRank]; omega⟩

/-- Every reachable send session can settle and release its staging buffer in a
finite number of steps (`System.settles_of_rank` over `drainRank`, with
`drain_step` as the one-step argument). -/
theorem reachable_can_settle {n : Nat} {s : Send} (h : (sys n).Reachable s) :
    ∃ evs s', (sys n).runFrom s evs = some s' ∧
      s'.life.done = true ∧ s'.life.hasStaging = false := by
  have hinv := reachable_inv h
  have hcancel : step s .cancel = some (s.finish false) := rfl
  have hinv₁ := step_inv hinv hcancel
  have hdr₁ : (s.finish false).life.draining = true :=
    hinv.life.finishOnceLocked_draining false
  obtain ⟨evs, s', hrun, ⟨hinv', _⟩, hd'⟩ :=
    (sys n).settles_of_rank (Inv := fun s => Inv s ∧ s.life.draining = true)
      (Done := fun s => s.life.done = true) drainRank
      (fun s ⟨hi, hdr⟩ hnd => by
        obtain ⟨e, s₁, hs₁, hdr₁, hlt⟩ := drain_step hi hdr (by simpa using hnd)
        exact ⟨e, s₁, hs₁, ⟨step_inv hi hs₁, hdr₁⟩, hlt⟩)
      ⟨hinv₁, hdr₁⟩
  refine ⟨.cancel :: evs, s', ?_, hd', by rw [hinv'.life.staging, hd']; rfl⟩
  simp only [System.runFrom, sys, List.foldlM_cons, hcancel]
  exact hrun


end Send

end TpuSyncVerify.Transfer.PrefillDecode

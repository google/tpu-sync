import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Invariant

/-! # `PushWindow`: the drain measure and settlement

Part of `TpuSyncVerify.Transfer.PrefillDecode.PushWindow`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.PushWindow

open TpuSyncVerify TpuSyncVerify.Transfer

namespace S

/-- Work left before the send settles: units the chain has not reached, tasks
and the parked unit by how far each is from its push, and pushes in flight. -/
def drainRank (s : S) : Nat :=
  7 * (s.numUnits - s.woken) + 5 * s.admitting + 4 * s.issuing + 2 * s.chaining +
    (s.h2hIssued - s.h2hRetired) + 4 * (if s.parked then 1 else 0)

/-- One step of draining: some event that every send can take (its copies and
pushes completing OK) strictly lowers `drainRank`. -/
theorem drain_step {s : S} (h : Inv s) (hif : 0 < s.life.inFlight) :
    ∃ e s', s.step e = some s' ∧ drainRank s' < drainRank s := by
  have hacc := h.accounted
  have hb := h.baton_le
  have hwq := h.woken_le
  have hqn := h.queued_le
  unfold baton at hb
  by_cases hwk : s.woken < s.queued
  · refine ⟨.wake true, ?_⟩
    simp only [step, wake, hwk, ↓reduceIte, Bool.true_eq_false]
    split
    · exact ⟨_, rfl, by simp [drainRank, endOp]; omega⟩
    · exact ⟨_, rfl, by simp [drainRank]; omega⟩
  · by_cases ha : s.admitting = 0
    · by_cases his : s.issuing = 0
      · by_cases hc : s.chaining = 0
        · have hlt : s.h2hRetired < s.h2hIssued := by omega
          refine ⟨.h2hDone true, ?_⟩
          simp only [step, h2hDone, hlt, ↓reduceIte]
          unfold resume
          split
          · split
            · exact ⟨_, rfl, by simp_all [drainRank, endOp, finish]; omega⟩
            · exact ⟨_, rfl, by simp_all [drainRank, endOp]; omega⟩
          · split
            · exact ⟨_, rfl, by simp_all [drainRank, endOp, finish]; omega⟩
            · exact ⟨_, rfl, by simp_all [drainRank, endOp]; omega⟩
        · refine ⟨.sendNext, ?_⟩
          simp only [step, sendNext, hc, ↓reduceIte]
          unfold trySendNext
          split
          · exact ⟨_, rfl, by simp [drainRank, endOp]; omega⟩
          · exact ⟨_, rfl, by simp [drainRank, endOp]; omega⟩
      · refine ⟨.issue, ?_⟩
        simp only [step, issue, his, ↓reduceIte]
        split
        · exact ⟨_, rfl, by simp [drainRank]; omega⟩
        · exact ⟨_, rfl, by simp [drainRank]; omega⟩
    · have hp : s.parked = false := by
        cases hpk : s.parked
        · rfl
        · simp [hpk] at hb; omega
      refine ⟨.admit, ?_⟩
      simp only [step, admit, ha, ↓reduceIte]
      split
      · exact ⟨_, rfl, by simp [drainRank, endOp, hp]; omega⟩
      · split
        · exact ⟨_, rfl, by simp [drainRank, endOp, hp]; omega⟩
        · exact ⟨_, rfl, by simp [drainRank, hp]; omega⟩

/-- **The window never strands a send.** From every reachable state some run —
the remaining copies and pushes completing — settles it: parking a unit while
the window is full cannot deadlock the chain. -/
theorem settles {n w : Nat} (hn : 0 < n) {s : S} (hr : (sys n w).Reachable s) :
    ∃ evs s', (sys n w).runFrom s evs = some s' ∧ s'.life.done = true := by
  have hstep : ∀ s, Inv s → ¬ s.life.done = true →
      ∃ e s', (sys n w).step s e = some s' ∧ Inv s' ∧ drainRank s' < drainRank s := by
    intro s hinv hnd
    have hsafe := inv_safe hinv
    have hif : 0 < s.life.inFlight := by
      by_cases hdr : s.life.draining = true
      · by_cases hz : s.life.inFlight = 0
        · exact absurd (hinv.life.prompt hdr hz) hnd
        · omega
      · exact hsafe.2.2.2.2.2.2.2.1 (by simpa using hdr)
    obtain ⟨e, s', hs, hlt⟩ := drain_step hinv hif
    exact ⟨e, s', hs, step_inv hinv hs, hlt⟩
  obtain ⟨evs, s', hrun, _, hd⟩ :=
    (sys n w).settles_of_rank (Inv := Inv) (Done := fun s => s.life.done = true) drainRank hstep
      (reachable_inv hn hr)
  exact ⟨evs, s', hrun, hd⟩

end S

end TpuSyncVerify.Transfer.PrefillDecode.PushWindow

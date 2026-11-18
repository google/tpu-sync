import TpuSyncVerify.Transfer.Reshard.Receive.Invariant

/-! # `Recv`: progress and eventual settlement

Part of `TpuSyncVerify.Transfer.Reshard.Receive`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ### Progress and eventual settlement -/

def drainRank (s : Recv) : Nat :=
  s.pushes + poolDrainSteps s.pools

theorem poolDrainSteps_set_lt {ps : List Pool} {idx : Nat} {p p' : Pool}
    (hget : ps[idx]? = some p) (hlt : p'.stage.drainSteps < p.stage.drainSteps) :
    poolDrainSteps (ps.set idx p') < poolDrainSteps ps := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst hget
      simp only [List.set_cons_zero, poolDrainSteps]
      omega
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      have := ih hget
      simp only [List.set_cons_succ, poolDrainSteps]
      omega

theorem poolOps_le_poolDrainSteps : ∀ ps : List Pool, poolOps ps ≤ poolDrainSteps ps
  | [] => Nat.le_refl 0
  | q :: qs => by
    have ih := poolOps_le_poolDrainSteps qs
    cases hq : q.stage <;>
      simp only [poolOps, poolDrainSteps, hq, PoolStage.holdsOp, PoolStage.drainSteps,
        Bool.false_eq_true, ↓reduceIte] <;> omega

theorem drain_step (pol : PushEndPolicy) {s : Recv} (h : Inv s)
    (hdr : s.life.draining = true) (hnd : s.life.done = false) :
    ∃ e s', step pol s e = some s' ∧ s'.life.draining = true ∧ drainRank s' < drainRank s := by
  have hif : 0 < s.life.inFlight := by
    cases h0 : s.life.inFlight
    · have := h.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := h.accounted; unfold Accounted at hacc
  by_cases hp : 0 < s.pushes
  · have hne : s.pushes ≠ 0 := by omega
    have hs₁ : step pol s (.pushEnd true) =
        some { s with life := s.life.endOpLocked, pushes := s.pushes - 1 } := by
      simp [step, pushEnd, hne]
    exact ⟨.pushEnd true, _, hs₁, by simp [hdr], by simp [drainRank]; omega⟩
  · have hpo : 0 < poolOps s.pools := by omega
    obtain ⟨p, _, pool, hget, hh⟩ := exists_holdsOp_of_poolOps_pos hpo
    cases hst : pool.stage with
    | queued =>
      have hs₁ : step pol s (.h2dIssue p true) =
          some { s with pools := s.pools.set p { pool with stage := .aborted },
                        life := s.life.endOpLocked } := by
        simp [step, h2dIssue, hget, hst, hdr]
      have hlt := poolDrainSteps_set_lt hget (p' := { pool with stage := .aborted })
        (by simp [hst, PoolStage.drainSteps])
      exact ⟨.h2dIssue p true, _, hs₁, by simp [hdr], by simp only [drainRank]; omega⟩
    | issued =>
      have hs₁ : step pol s (.h2dReady p) =
          some { s with pools := s.pools.set p { pool with stage := .ready } } := by
        simp [step, h2dReady, hget, hst]
      have hlt := poolDrainSteps_set_lt hget (p' := { pool with stage := .ready })
        (by simp [hst, PoolStage.drainSteps])
      exact ⟨.h2dReady p, _, hs₁, hdr, by simp only [drainRank]; omega⟩
    | ready =>
      have hlt_ab := poolDrainSteps_set_lt hget (p' := { pool with stage := .aborted })
        (by simp [hst, PoolStage.drainSteps])
      have hlt_co := poolDrainSteps_set_lt hget (p' := { pool with stage := .completed })
        (by simp [hst, PoolStage.drainSteps])
      by_cases hfin : s.finalizing = true
      · have hs₁ : step pol s (.h2dDone p true) =
            some { s with
              pools := s.pools.set p { pool with stage := .aborted },
              life := s.life.endOpLocked } := by
          simp [step, h2dDone, hget, hst, hnd, hfin]
        exact ⟨.h2dDone p true, _, hs₁, by simp [hdr], by simp [drainRank]; omega⟩
      · simp only [Bool.not_eq_true] at hfin
        by_cases hall : allCompleted (s.pools.set p { pool with stage := .completed }) = true
        · have hs₁ : step pol s (.h2dDone p true) =
              some { s with
                pools := s.pools.set p { pool with stage := .completed },
                finalizing := true,
                durationRecorded := true,
                life := (s.life.finishLocked true).endOpLocked } := by
            simp [step, h2dDone, hget, hst, hnd, hfin, hall]
          exact ⟨.h2dDone p true, _, hs₁, by simp, by simp [drainRank]; omega⟩
        · simp only [Bool.not_eq_true] at hall
          have hs₁ : step pol s (.h2dDone p true) =
              some { s with
                pools := s.pools.set p { pool with stage := .completed },
                life := s.life.endOpLocked } := by
            simp [step, h2dDone, hget, hst, hnd, hfin, hall, launchEligible, hdr]
          exact ⟨.h2dDone p true, _, hs₁, by simp [hdr], by simp [drainRank]; omega⟩
    | waiting | received | completed | aborted =>
      simp [hst, PoolStage.holdsOp] at hh

/-- Every reachable reshard receive session can drain to `done = true` and
release its bounded `PoolStagingLease` in finitely many steps
(`System.settles_of_rank` over `drainRank`, with `drain_step` as the one-step
argument). -/
theorem reachable_can_settle {ranks : List Nat} {pol : PushEndPolicy} {s : Recv}
    (h : (sys ranks pol).Reachable s) :
    ∃ evs s', (sys ranks pol).runFrom s evs = some s' ∧
      s'.life.done = true ∧ s'.life.hasStaging = false := by
  have hinv := reachable_inv h
  have hcancel : step pol s .cancel = some { s with life := s.life.finishLocked false } := rfl
  have hinv₁ := step_inv pol hinv hcancel
  obtain ⟨evs, s', hrun, ⟨hinv', _⟩, hd'⟩ :=
    (sys ranks pol).settles_of_rank (Inv := fun s => Inv s ∧ s.life.draining = true)
      (Done := fun s => s.life.done = true) drainRank
      (fun s ⟨hi, hdr⟩ hnd => by
        obtain ⟨e, s₁, hs₁, hdr₁, hlt⟩ := drain_step pol hi hdr (by simpa using hnd)
        exact ⟨e, s₁, hs₁, ⟨step_inv pol hi hs₁, hdr₁⟩, hlt⟩)
      ⟨hinv₁, by simp⟩
  refine ⟨.cancel :: evs, s', ?_, hd', by rw [hinv'.life.staging, hd']; rfl⟩
  simp only [System.runFrom, sys, List.foldlM_cons, hcancel]
  exact hrun

end Recv

end TpuSyncVerify.Transfer.Reshard

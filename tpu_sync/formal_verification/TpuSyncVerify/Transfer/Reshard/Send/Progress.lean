import TpuSyncVerify.Transfer.Reshard.Send.Invariant

/-! # `Send`: progress and eventual settlement

Part of `TpuSyncVerify.Transfer.Reshard.Send`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ### Progress and eventual settlement -/

theorem drain_step {s : Send} (h : Inv s)
    (hdr : s.life.draining = true) (hnd : s.life.done = false) :
    ∃ e s', step s e = some s' ∧ s'.life.draining = true ∧ s'.life.inFlight < s.life.inFlight := by
  have hif : 0 < s.life.inFlight := by
    cases h0 : s.life.inFlight
    · have := h.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := h.accounted; unfold Accounted at hacc
  cases hex : s.execActive
  · simp only [hex, Bool.false_eq_true, ↓reduceIte, Nat.zero_add] at hacc
    obtain ⟨p, _, pool, hget, hop⟩ := exists_op_of_poolOps_pos (ps := s.pools) (by omega)
    cases hst : pool.stage with
    | pending | aborted => simp [hst, SendStage.opCount] at hop
    | d2h =>
      have hs₁ : step s (.d2hDone p true) =
          some (endOp { s with pools := s.pools.set p { pool with stage := .aborted } }) := by
        simp [step, d2hDone, hget, hst, hdr]
      exact ⟨.d2hDone p true, _, hs₁, by simp [endOp, hdr], by simp [endOp]; omega⟩
    | pushing k =>
      cases k with
      | zero => simp [hst, SendStage.opCount] at hop
      | succ k =>
        have hs₁ : step s (.pushDone p true) =
            some (endOp { s with pools := s.pools.set p { pool with stage := .pushing k } }) := by
          simp [step, pushDone, hget, hst, recordPushCompletion, hdr]
        exact ⟨.pushDone p true, _, hs₁, by simp [endOp, hdr], by simp [endOp]; omega⟩
  · cases hiss : allIssued s.pools
    · obtain ⟨p, _, pool, hget, hp⟩ := exists_pending_of_allIssued_false hiss
      have hs₁ : step s (.d2hIssue p false) =
          some (endOp { s with pools := s.pools.set p { pool with stage := .aborted },
                               execActive := false }) := by
        simp [step, d2hIssue, hex, hget, hp, recordPushCompletion, hdr]
      exact ⟨.d2hIssue p false, _, hs₁, by simp [endOp, hdr], by simp [endOp]; omega⟩
    · have hs₁ : step s .execFinish =
          some (endOp { s with execActive := false }) := by
        simp [step, execFinish, hex, hiss]
      exact ⟨.execFinish, _, hs₁, by simp [endOp, hdr], by simp [endOp]; omega⟩

/-- Every reachable reshard send session on a valid schedule can drain to
`done = true`, release its `PoolStagingLease`s (`hasStaging = false`), and
unregister its active plan (`planUnregistered = true`) in finitely many steps
(`System.settles_of_rank` over `in_flight_`, with `drain_step` as the one-step
argument). -/
theorem reachable_can_settle {ns : List Nat} {s : Send}
    (hvalid : validSchedule ns = true) (h : (sys ns).Reachable s) :
    ∃ evs s', (sys ns).runFrom s evs = some s' ∧
      s'.life.done = true ∧ s'.life.hasStaging = false ∧ s'.planUnregistered = true := by
  have hinv := reachable_inv hvalid h
  have hcancel : step s .cancel = some { s with life := s.life.finishLocked false } := rfl
  have hinv₁ := step_inv hinv hcancel
  obtain ⟨evs, s', hrun, ⟨hinv', _⟩, hd'⟩ :=
    (sys ns).settles_of_rank (Inv := fun s => Inv s ∧ s.life.draining = true)
      (Done := fun s => s.life.done = true) (fun s => s.life.inFlight)
      (fun s ⟨hi, hdr⟩ hnd => by
        obtain ⟨e, s₁, hs₁, hdr₁, hlt⟩ := drain_step hi hdr (by simpa using hnd)
        exact ⟨e, s₁, hs₁, ⟨step_inv hi hs₁, hdr₁⟩, hlt⟩)
      ⟨hinv₁, by simp⟩
  refine ⟨.cancel :: evs, s', ?_, hd', by rw [hinv'.life.staging, hd']; rfl, hinv'.done_plan hd'⟩
  simp only [System.runFrom, sys, List.foldlM_cons, hcancel]
  exact hrun

end Send

end TpuSyncVerify.Transfer.Reshard

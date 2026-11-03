import TpuSyncVerify.Transfer.Reshard.Send.Lemmas

/-! # `Send`: properties, inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.Reshard.Send`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Properties -/

def SettleSafe (s : Send) : Prop := s.life.done = true → s.life.inFlight = 0

def NoRetiredCallback (s : Send) : Prop :=
  s.life.done = true → s.execActive = false ∧ poolOps s.pools = 0

def StagingIntegrity (s : Send) : Prop := s.life.hasStaging = !s.life.done

def SettlesPromptly (s : Send) : Prop :=
  s.life.draining = true → s.life.inFlight = 0 → s.life.done = true

def ActiveOrDraining (s : Send) : Prop :=
  s.life.draining = false → 0 < s.life.inFlight

def PlanIntegrity (s : Send) : Prop :=
  (s.life.done = true → s.planUnregistered = true) ∧
  (s.planUnregistered = true → s.life.draining = true)

def Publication (s : Send) : Prop :=
  s.published = some true → s.remainingPushes = 0 ∧ allPushed s.pools = true

def drainEvents (n : Nat) : List Ev :=
  [.execFinish] ++
  (List.range n).flatMap fun p => [.d2hIssue p false, .d2hDone p true, .pushDone p true]

def NoOpLeak (s : Send) : Prop :=
  0 < s.life.inFlight → ∃ e ∈ drainEvents s.pools.length, (step s e).isSome = true

def Safe (s : Send) : Prop :=
  SettleSafe s ∧ NoRetiredCallback s ∧ StagingIntegrity s ∧ SettlesPromptly s ∧
    ActiveOrDraining s ∧ PlanIntegrity s ∧ Publication s ∧ NoOpLeak s

/-! ## Inductive invariant -/

def Accounted (s : Send) : Prop :=
  s.life.inFlight = (if s.execActive then 1 else 0) + poolOps s.pools

structure Inv (s : Send) : Prop where
  valid : validPools s.pools = true
  life : s.life.Consistent
  accounted : Accounted s
  exec_pending : s.execActive = false → s.life.draining = false → allIssued s.pools = true
  not_draining : s.life.draining = false →
    s.remainingPushes = sumRemainingSlots s.pools ∧ 0 < s.remainingPushes ∧ noAborted s.pools = true
  ok_draining : s.life.statusOk = true → s.life.draining = true →
    s.remainingPushes = 0 ∧ allPushed s.pools = true
  plan_draining : s.planUnregistered = true → s.life.draining = true
  done_plan : s.life.done = true → s.planUnregistered = true
  published_done : ∀ b, s.published = some b → s.life.done = true
  published_ok : s.published = some true → s.remainingPushes = 0 ∧ allPushed s.pools = true

theorem inv_init {ns : List Nat} (h : validSchedule ns = true) :
    Inv (init ns) := by
  refine ⟨validPools_init h, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · constructor <;> simp [init]
  · simp [init, Accounted, poolOps_init]
  · simp [init]
  · intro _
    exact ⟨(sumRemainingSlots_init ns).symm, sumPeers_pos_of_validSchedule h, noAborted_init ns⟩
  · simp [init]
  · simp [init]
  · simp [init]
  · simp [init]
  · simp [init]

theorem inv_activeOrDraining {s : Send} (h : Inv s) :
    ActiveOrDraining s := by
  intro hdr
  have hacc := h.accounted; unfold Accounted at hacc
  cases hex : s.execActive
  · have hiss := h.exec_pending hex hdr
    have ⟨hrem, hpos, _⟩ := h.not_draining hdr
    have hpo := poolOps_pos_of_sumRemainingSlots_pos hiss (by omega)
    simp [hex] at hacc
    omega
  · simp [hex] at hacc
    omega

theorem mem_drainEvents_d2hIssue {n p : Nat} (hp : p < n) :
    Ev.d2hIssue p false ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  exact Or.inr ⟨p, hp, by simp⟩

theorem mem_drainEvents_d2hDone {n p : Nat} (hp : p < n) :
    Ev.d2hDone p true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  exact Or.inr ⟨p, hp, by simp⟩

theorem mem_drainEvents_pushDone {n p : Nat} (hp : p < n) :
    Ev.pushDone p true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  exact Or.inr ⟨p, hp, by simp⟩

theorem inv_noOpLeak {s : Send} (h : Inv s) : NoOpLeak s := by
  intro hif
  have hacc := h.accounted; unfold Accounted at hacc
  cases hex : s.execActive
  · simp only [hex, Bool.false_eq_true, ↓reduceIte, Nat.zero_add] at hacc
    obtain ⟨p, hlt, pool, hget, hop⟩ := exists_op_of_poolOps_pos (ps := s.pools) (by omega)
    cases hst : pool.stage with
    | pending | aborted => simp [hst, SendStage.opCount] at hop
    | d2h =>
      refine ⟨.d2hDone p true, mem_drainEvents_d2hDone hlt, ?_⟩
      simp only [step, d2hDone, hget, hst, SendStage.bne_self, Bool.false_eq_true, ↓reduceIte,
        Bool.not_true]
      split <;> rfl
    | pushing k =>
      cases k with
      | zero => simp [hst, SendStage.opCount] at hop
      | succ k =>
        refine ⟨.pushDone p true, mem_drainEvents_pushDone hlt, ?_⟩
        simp [step, pushDone, hget, hst]
  · cases hiss : allIssued s.pools
    · obtain ⟨p, hlt, pool, hget, hp⟩ := exists_pending_of_allIssued_false hiss
      refine ⟨.d2hIssue p false, mem_drainEvents_d2hIssue hlt, ?_⟩
      simp [step, d2hIssue, hex, hget, hp]
    · refine ⟨.execFinish, by simp [drainEvents], ?_⟩
      simp [step, execFinish, hex, hiss]

theorem inv_safe {s : Send} (h : Inv s) : Safe s := by
  have hacc := h.accounted; unfold Accounted at hacc
  refine ⟨h.life.done_idle, ?_, h.life.staging, h.life.prompt,
    inv_activeOrDraining h, ⟨h.done_plan, h.plan_draining⟩,
    h.published_ok, inv_noOpLeak h⟩
  intro hd
  have h0 := h.life.done_idle hd
  cases hex : s.execActive
  · simp only [hex, Bool.false_eq_true, ↓reduceIte] at hacc
    exact ⟨rfl, by omega⟩
  · simp only [hex, ↓reduceIte] at hacc
    omega

theorem consistent_add_inFlight_of_pos {l : Lifecycle} (h : l.Consistent)
    (hpos : 0 < l.inFlight) (k : Nat) :
    Lifecycle.Consistent { l with inFlight := l.inFlight + k } := by
  have hnd : l.done = false := by
    cases hd : l.done
    · rfl
    · have := h.done_idle hd; omega
  constructor <;> simp [hnd, h.staging]
  intro _ h0; omega

theorem endOp_inv {s : Send}
    (hvalid : validPools s.pools = true)
    (hlife : s.life.Consistent)
    (hpos : 0 < s.life.inFlight)
    (hacc : s.life.inFlight = (if s.execActive then 1 else 0) + poolOps s.pools + 1)
    (hex : s.execActive = false → s.life.draining = false → allIssued s.pools = true)
    (hndrain : s.life.draining = false →
      s.remainingPushes = sumRemainingSlots s.pools ∧ 0 < s.remainingPushes ∧ noAborted s.pools = true)
    (hok : s.life.statusOk = true → s.life.draining = true →
      s.remainingPushes = 0 ∧ allPushed s.pools = true)
    (hplan_dr : s.planUnregistered = true → s.life.draining = true)
    (_hdone_pl : s.life.done = true → s.planUnregistered = true)
    (hpub_d : ∀ b, s.published = some b → s.life.done = true)
    (hpub_ok : s.published = some true → s.remainingPushes = 0 ∧ allPushed s.pools = true) :
    Inv (endOp s) := by
  have hnd : s.life.done = false := by
    cases hd : s.life.done
    · rfl
    · have := hlife.done_idle hd; omega
  unfold endOp
  refine ⟨hvalid, Lifecycle.endOpLocked_consistent hlife, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hpub_ok⟩
  · unfold Accounted
    simp only [Lifecycle.endOpLocked_inFlight]
    omega
  · dsimp only
    simp only [Lifecycle.endOpLocked_draining]
    exact hex
  · dsimp only
    simp only [Lifecycle.endOpLocked_draining]
    exact hndrain
  · dsimp only
    simp only [Lifecycle.endOpLocked_statusOk, Lifecycle.endOpLocked_draining]
    exact hok
  · dsimp only
    simp only [Lifecycle.endOpLocked_draining]
    split
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      intro _; exact hc.1.2
    · exact hplan_dr
  · dsimp only
    split
    · intro _; rfl
    · rename_i hc
      intro hd
      unfold Lifecycle.endOpLocked Lifecycle.settleLocked at hd
      split at hd
      · simp [hnd] at hd
      · split at hd
        · rename_i hne0 hc'
          dsimp only at hc'
          exfalso
          apply hc
          simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true']
          exact ⟨⟨by omega, hc'.1⟩, hnd⟩
        · simp [hnd] at hd
  · intro b hb
    exact Lifecycle.endOpLocked_done_mono (hpub_d b hb)

theorem step_inv {s s' : Send} {e : Ev}
    (h : Inv s) (hs : step s e = some s') : Inv s' := by
  have hacc := h.accounted; unfold Accounted at hacc
  have hallPos : allPosPeers s.pools = true := by
    have hv := h.valid; unfold validPools at hv
    simp only [Bool.and_eq_true] at hv; exact hv.2
  cases e with
  | d2hIssue p ok =>
    simp only [step, d2hIssue] at hs
    split at hs
    · cases hs
    · rename_i hex
      have hex_t : s.execActive = true := by revert hex; cases s.execActive <;> simp
      have hif_pos : 0 < s.life.inFlight := by simp [hex_t] at hacc; omega
      have hnd : s.life.done = false := by
        cases hd : s.life.done
        · rfl
        · have := h.life.done_idle hd; omega
      split at hs
      · rename_i pool hget
        split at hs
        · cases hs
        · rename_i hpend
          simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hpend
          have hne_push : pool.stage ≠ .pushing 0 := by simp [hpend]
          have hallPushed_f := allPushed_eq_false_of_ne hget hne_push
          have hvalid' : validPools (s.pools.set p { pool with stage := .d2h }) = true := by
            rw [validPools_set _ hget]; exact h.valid
          have hvalid_ab : validPools (s.pools.set p { pool with stage := .aborted }) = true := by
            rw [validPools_set _ hget]; exact h.valid
          have ⟨hpo_d2h, _⟩ := poolOps_set .d2h hget
          have ⟨hpo_ab, _⟩ := poolOps_set .aborted hget
          have ⟨hrem_d2h, _⟩ := sumRemainingSlots_set .d2h hget
          split at hs
          · cases hs
            refine ⟨hvalid', consistent_add_inFlight_of_pos h.life hif_pos 1,
              ?_, ?_, ?_, ?_, h.plan_draining, h.done_plan,
              h.published_done, ?_⟩
            · unfold Accounted
              simp only [hex_t, ↓reduceIte] at hacc ⊢
              simp only [hpend, SendStage.opCount] at hpo_d2h
              omega
            · simp [hex_t]
            · intro hdr
              have ⟨hr, hpos, hno⟩ := h.not_draining hdr
              simp only [SendPool.remainingSlots, hpend] at hrem_d2h
              dsimp only
              exact ⟨by omega, hpos, noAborted_set_of_ne_aborted hget (by decide) hno⟩
            · intro hok hdr
              have ⟨_, hpushed⟩ := h.ok_draining hok hdr
              simp [hallPushed_f] at hpushed
            · intro hpub
              have ⟨_, hpushed⟩ := h.published_ok hpub
              simp [hallPushed_f] at hpushed
          · cases hs
            unfold recordPushCompletion
            dsimp only
            split
            · rename_i hact
              rcases hact with hd | hdr
              · simp [hnd] at hd
              · refine endOp_inv hvalid_ab h.life hif_pos ?_ ?_ ?_ ?_
                  h.plan_draining h.done_plan h.published_done ?_
                · simp only [hex_t, ↓reduceIte] at hacc
                  simp only [hpend, SendStage.opCount] at hpo_ab
                  simp; omega
                · intro _ hdr'; dsimp only at hdr'; simp [hdr] at hdr'
                · intro hdr'; dsimp only at hdr'; simp [hdr] at hdr'
                · intro hok _
                  have ⟨_, hpushed⟩ := h.ok_draining hok hdr
                  simp [hallPushed_f] at hpushed
                · intro hpub
                  have ⟨_, hpushed⟩ := h.published_ok hpub
                  simp [hallPushed_f] at hpushed
            · rename_i hact
              simp only [not_or, Bool.not_eq_true] at hact
              obtain ⟨_, hdr⟩ := hact
              refine endOp_inv hvalid_ab (Lifecycle.finishLocked_consistent false h.life)
                ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
              · simp [Lifecycle.finishLocked_inFlight, hif_pos]
              · simp only [hex_t, ↓reduceIte] at hacc
                simp only [hpend, SendStage.opCount] at hpo_ab
                simp [Lifecycle.finishLocked_inFlight]; omega
              · simp
              · simp
              · simp
              · simp
              · intro hd
                simp only [Bool.not_false, ↓reduceIte] at hd
                rw [Lifecycle.finishLocked_done_of_inFlight _ (Nat.ne_of_gt hif_pos)] at hd
                simp [hnd] at hd
              · intro b hb
                have hd := h.published_done b hb
                simp [hnd] at hd
              · intro hpub
                have hd := h.published_done true hpub
                simp [hnd] at hd
      · cases hs
  | execFinish =>
    simp only [step, execFinish] at hs
    split at hs
    · rename_i hc
      obtain ⟨hex, hiss⟩ := hc
      cases hs
      have hif_pos : 0 < s.life.inFlight := by simp [hex] at hacc; omega
      refine endOp_inv h.valid h.life hif_pos ?_ ?_ h.not_draining h.ok_draining
        h.plan_draining h.done_plan h.published_done h.published_ok
      · simp [hex] at hacc; simp; omega
      · intro _ _; exact hiss
    · cases hs
  | d2hDone p ok =>
    simp only [step, d2hDone] at hs
    split at hs
    · rename_i pool hget
      split at hs
      · cases hs
      · rename_i hd2h
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hd2h
        have hne_push : pool.stage ≠ .pushing 0 := by simp [hd2h]
        have hallPushed_f := allPushed_eq_false_of_ne hget hne_push
        have ⟨hpo_ab, hle_po⟩ := poolOps_set .aborted hget
        have ⟨hpo_pu, _⟩ := poolOps_set (.pushing pool.peers) hget
        have ⟨hrem_pu, _⟩ := sumRemainingSlots_set (.pushing pool.peers) hget
        simp only [hd2h, SendStage.opCount] at hpo_ab hpo_pu hle_po
        have hif_pos : 0 < s.life.inFlight := by omega
        have hnd : s.life.done = false := by
          cases hd : s.life.done
          · rfl
          · have := h.life.done_idle hd; omega
        have hvalid_ab : validPools (s.pools.set p { pool with stage := .aborted }) = true := by
          rw [validPools_set _ hget]; exact h.valid
        have hvalid_pu : validPools (s.pools.set p { pool with stage := .pushing pool.peers }) = true := by
          rw [validPools_set _ hget]; exact h.valid
        split at hs
        · cases hs
          unfold recordPushCompletion
          dsimp only
          split
          · rename_i hact
            rcases hact with hd | hdr
            · simp [hnd] at hd
            · refine endOp_inv hvalid_ab h.life hif_pos (by dsimp only; omega) ?_ ?_ ?_
                h.plan_draining h.done_plan h.published_done ?_
              · intro _ hdr'; dsimp only at hdr'; simp [hdr] at hdr'
              · intro hdr'; dsimp only at hdr'; simp [hdr] at hdr'
              · intro hok _; have ⟨_, hp⟩ := h.ok_draining hok hdr; simp [hallPushed_f] at hp
              · intro hpub; have ⟨_, hp⟩ := h.published_ok hpub; simp [hallPushed_f] at hp
          · rename_i hact
            simp only [not_or, Bool.not_eq_true] at hact
            refine endOp_inv hvalid_ab (Lifecycle.finishLocked_consistent false h.life)
              ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
            · simp [Lifecycle.finishLocked_inFlight, hif_pos]
            · simp [Lifecycle.finishLocked_inFlight]; omega
            · simp
            · simp
            · simp
            · simp
            · intro hd
              simp only [Bool.not_false, ↓reduceIte] at hd
              rw [Lifecycle.finishLocked_done_of_inFlight _ (Nat.ne_of_gt hif_pos)] at hd
              simp [hnd] at hd
            · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
            · intro hpub; have hd := h.published_done true hpub; simp [hnd] at hd
        · split at hs
          · rename_i hact
            cases hs
            rcases hact with hd | hdr
            · simp [hnd] at hd
            · refine endOp_inv hvalid_ab h.life hif_pos (by dsimp only; omega) ?_ ?_ ?_
                h.plan_draining h.done_plan h.published_done ?_
              · intro _ hdr'; dsimp only at hdr'; simp [hdr] at hdr'
              · intro hdr'; dsimp only at hdr'; simp [hdr] at hdr'
              · intro hok _; have ⟨_, hp⟩ := h.ok_draining hok hdr; simp [hallPushed_f] at hp
              · intro hpub; have ⟨_, hp⟩ := h.published_ok hpub; simp [hallPushed_f] at hp
          · rename_i hact
            cases hs
            simp only [not_or, Bool.not_eq_true] at hact
            obtain ⟨_, hdr⟩ := hact
            have hp_pos := peers_pos_of_allPosPeers hallPos hget
            refine endOp_inv hvalid_pu (consistent_add_inFlight_of_pos h.life hif_pos pool.peers)
              ?_ ?_ ?_ ?_ ?_ h.plan_draining ?_ ?_ ?_
            · dsimp only; omega
            · dsimp only; omega
            · intro hex hdr'
              exact allIssued_set_of_ne_pending hget (by simp) (h.exec_pending hex hdr')
            · intro _
              have ⟨hr, hpos, hno⟩ := h.not_draining hdr
              simp only [SendPool.remainingSlots, hd2h] at hrem_pu
              dsimp only
              exact ⟨by omega, hpos, noAborted_set_of_ne_aborted hget (by simp) hno⟩
            · intro _ hdr'; dsimp only at hdr'; simp [hdr] at hdr'
            · intro hd; simp [hnd] at hd
            · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
            · intro hpub; have hd := h.published_done true hpub; simp [hnd] at hd
    · cases hs
  | pushDone p ok =>
    simp only [step, pushDone] at hs
    split at hs
    · rename_i pool hget
      split at hs
      · rename_i k hpush
        cases hs
        have hne_push : pool.stage ≠ .pushing 0 := by simp [hpush]
        have hallPushed_f := allPushed_eq_false_of_ne hget hne_push
        have ⟨hpo, hle_po⟩ := poolOps_set (.pushing k) hget
        have ⟨hrem, hle_rem⟩ := sumRemainingSlots_set (.pushing k) hget
        simp only [hpush, SendStage.opCount] at hpo hle_po
        simp only [SendPool.remainingSlots, hpush] at hrem hle_rem
        have hif_pos : 0 < s.life.inFlight := by omega
        have hnd : s.life.done = false := by
          cases hd : s.life.done
          · rfl
          · have := h.life.done_idle hd; omega
        have hvalid' : validPools (s.pools.set p { pool with stage := .pushing k }) = true := by
          rw [validPools_set _ hget]; exact h.valid
        have hallPos' : allPosPeers (s.pools.set p { pool with stage := .pushing k }) = true := by
          rw [allPosPeers_set _ hget]; exact hallPos
        unfold recordPushCompletion
        dsimp only
        split
        · rename_i hact
          rcases hact with hd | hdr
          · simp [hnd] at hd
          · refine endOp_inv hvalid' h.life hif_pos (by dsimp only; omega) ?_ ?_ ?_
              h.plan_draining h.done_plan h.published_done ?_
            · intro _ hdr'; dsimp only at hdr'; simp [hdr] at hdr'
            · intro hdr'; dsimp only at hdr'; simp [hdr] at hdr'
            · intro hok _; have ⟨_, hp⟩ := h.ok_draining hok hdr; simp [hallPushed_f] at hp
            · intro hpub; have ⟨_, hp⟩ := h.published_ok hpub; simp [hallPushed_f] at hp
        · rename_i hact
          simp only [not_or, Bool.not_eq_true] at hact
          obtain ⟨_, hdr⟩ := hact
          have ⟨hr_eq, hr_pos, hno⟩ := h.not_draining hdr
          have hno' := noAborted_set_of_ne_aborted hget (st := .pushing k) (by simp) hno
          split
          · cases ok
            · refine endOp_inv hvalid' (Lifecycle.finishLocked_consistent false h.life)
                ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
              · simp [Lifecycle.finishLocked_inFlight, hif_pos]
              · dsimp only; simp [Lifecycle.finishLocked_inFlight]; omega
              · simp
              · simp
              · simp
              · simp
              · intro _; rfl
              · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
              · intro hpub; have hd := h.published_done true hpub; simp [hnd] at hd
            · rename_i hfalse; simp at hfalse
          · split
            · rename_i hle1
              have hsum0 : sumRemainingSlots (s.pools.set p { pool with stage := .pushing k }) = 0 := by
                omega
              have hallPushed' := allPushed_of_sumRemainingSlots_zero hallPos' hno' hsum0
              refine endOp_inv hvalid' (Lifecycle.finishLocked_consistent true h.life)
                ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
              · simp [Lifecycle.finishLocked_inFlight, hif_pos]
              · dsimp only; simp [Lifecycle.finishLocked_inFlight]; omega
              · simp
              · simp
              · intro _ _; exact ⟨rfl, hallPushed'⟩
              · simp
              · intro _; rfl
              · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
              · intro hpub; have hd := h.published_done true hpub; simp [hnd] at hd
            · rename_i hgt1
              refine endOp_inv hvalid' h.life hif_pos ?_ ?_ ?_ ?_
                h.plan_draining h.done_plan h.published_done ?_
              · dsimp only; omega
              · intro hex _
                exact allIssued_set_of_ne_pending hget (by simp) (h.exec_pending hex hdr)
              · intro _
                dsimp only
                exact ⟨by omega, by omega, hno'⟩
              · intro _ hdr'; dsimp only at hdr'; simp [hdr] at hdr'
              · intro hpub; have hd := h.published_done true hpub; simp [hnd] at hd
      · cases hs
    · cases hs
  | cancel =>
    simp only [step, cancel] at hs
    cases hs
    refine ⟨h.valid, Lifecycle.finishLocked_consistent false h.life, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
      h.published_ok⟩
    · unfold Accounted; simpa using hacc
    · simp
    · simp
    · simp
    · simp
    · intro hd
      dsimp only at hd
      by_cases hdone : s.life.done = true
      · exact h.done_plan hdone
      · simp only [Bool.not_eq_true] at hdone
        have hif_pos : 0 < s.life.inFlight := by
          by_cases hdr : s.life.draining = true
          · cases h0 : s.life.inFlight
            · have := h.life.prompt hdr h0; simp [hdone] at this
            · omega
          · simp only [Bool.not_eq_true] at hdr
            exact inv_activeOrDraining h hdr
        rw [Lifecycle.finishLocked_done_of_inFlight _ (Nat.ne_of_gt hif_pos)] at hd
        simp [hdone] at hd
    · intro b hb
      exact Lifecycle.finishLocked_done_mono false (h.published_done b hb)
  | publish =>
    simp only [step, publish] at hs
    split at hs
    · rename_i hc
      obtain ⟨hdone, _⟩ := hc
      cases hs
      refine ⟨h.valid, h.life, h.accounted, h.exec_pending, h.not_draining, h.ok_draining,
        h.plan_draining, h.done_plan, fun _ _ => hdone, ?_⟩
      simp only [Option.some.injEq]
      intro hok
      exact h.ok_draining hok (h.life.done_draining hdone)
    · cases hs

theorem reachable_inv {ns : List Nat} {s : Send}
    (hvalid : validSchedule ns = true) (h : (sys ns).Reachable s) : Inv s :=
  (sys ns).reachable_induction (inv_init hvalid) (fun _ _ _ hi hs => step_inv hi hs) h

/-- Main result: every reachable reshard send session on a valid schedule
satisfies all safety properties. -/
theorem reachable_safe {ns : List Nat} {s : Send}
    (hvalid : validSchedule ns = true) (h : (sys ns).Reachable s) : Safe s :=
  inv_safe (reachable_inv hvalid h)

end Send

end TpuSyncVerify.Transfer.Reshard

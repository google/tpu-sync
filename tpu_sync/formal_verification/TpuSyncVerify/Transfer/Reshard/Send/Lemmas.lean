import TpuSyncVerify.Transfer.Reshard.Send.Model

/-! # `Send`: structural lemmas on `List SendPool`

Part of `TpuSyncVerify.Transfer.Reshard.Send`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Structural lemmas on `List SendPool` -/

theorem poolOps_init : ∀ ns : List Nat,
    poolOps (ns.map fun n => { peers := n, stage := .pending }) = 0
  | [] => rfl
  | _ :: ns => by simp [poolOps, SendStage.opCount, poolOps_init ns]

theorem sumRemainingSlots_init : ∀ ns : List Nat,
    sumRemainingSlots (ns.map fun n => { peers := n, stage := .pending }) = sumPeers ns
  | [] => rfl
  | n :: ns => by simp [sumRemainingSlots, SendPool.remainingSlots, sumPeers, sumRemainingSlots_init ns]

theorem noAborted_init : ∀ ns : List Nat,
    noAborted (ns.map fun n => { peers := n, stage := .pending }) = true
  | [] => rfl
  | _ :: _ => by simp [noAborted]

theorem validPools_init {ns : List Nat} (h : validSchedule ns = true) :
    validPools (ns.map fun n => { peers := n, stage := .pending }) = true := by
  unfold validSchedule at h
  unfold validPools allPosPeers
  rcases ns with _ | ⟨n, ns⟩
  · simp at h
  · simp only [List.isEmpty_cons, Bool.not_false, List.all_cons, Bool.true_and] at h
    simp only [List.map_cons, List.isEmpty_cons, Bool.not_false, List.all_cons, Bool.true_and,
      List.all_map]
    exact h

theorem sumPeers_pos_of_validSchedule {ns : List Nat} (h : validSchedule ns = true) :
    0 < sumPeers ns := by
  rcases ns with _ | ⟨n, ns⟩
  · simp [validSchedule] at h
  · simp only [validSchedule, List.isEmpty_cons, Bool.not_false, List.all_cons, Bool.true_and,
       Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [sumPeers]
    omega

theorem allPosPeers_set {ps : List SendPool} {idx : Nat} {p : SendPool} (st : SendStage)
    (hget : ps[idx]? = some p) :
    allPosPeers (ps.set idx { p with stage := st }) = allPosPeers ps := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      rfl
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      unfold allPosPeers at ih ⊢
      simp only [List.set_cons_succ, List.all_cons, ih hget]

theorem validPools_set {ps : List SendPool} {idx : Nat} {p : SendPool} (st : SendStage)
    (hget : ps[idx]? = some p) :
    validPools (ps.set idx { p with stage := st }) = validPools ps := by
  unfold validPools
  rw [allPosPeers_set st hget]
  cases ps <;> cases idx <;> rfl

theorem peers_pos_of_allPosPeers {ps : List SendPool} {idx : Nat} {p : SendPool}
    (hall : allPosPeers ps = true) (hget : ps[idx]? = some p) : 0 < p.peers := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    simp only [allPosPeers, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hall
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      exact hall.1
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      exact ih hall.2 hget

theorem poolOps_set {ps : List SendPool} {idx : Nat} {p : SendPool} (st : SendStage)
    (hget : ps[idx]? = some p) :
    poolOps (ps.set idx { p with stage := st }) + p.stage.opCount =
      poolOps ps + st.opCount ∧ p.stage.opCount ≤ poolOps ps := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      simp only [List.set_cons_zero, poolOps]
      omega
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      have ⟨heq, hle⟩ := ih hget
      simp only [List.set_cons_succ, poolOps]
      omega

theorem sumRemainingSlots_set {ps : List SendPool} {idx : Nat} {p : SendPool} (st : SendStage)
    (hget : ps[idx]? = some p) :
    sumRemainingSlots (ps.set idx { p with stage := st }) + p.remainingSlots =
      sumRemainingSlots ps + ({ p with stage := st } : SendPool).remainingSlots ∧
    p.remainingSlots ≤ sumRemainingSlots ps := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      simp only [List.set_cons_zero, sumRemainingSlots]
      omega
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      have ⟨heq, hle⟩ := ih hget
      simp only [List.set_cons_succ, sumRemainingSlots]
      omega

theorem allIssued_set_of_ne_pending {ps : List SendPool} {idx : Nat} {p : SendPool} {st : SendStage}
    (hget : ps[idx]? = some p) (hst : st ≠ .pending) (h : allIssued ps = true) :
    allIssued (ps.set idx { p with stage := st }) = true := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    simp only [allIssued, List.all_cons, Bool.and_eq_true] at h
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      simp [allIssued, hst, h.2]
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      simp only [allIssued, List.set_cons_succ, List.all_cons, Bool.and_eq_true]
      exact ⟨h.1, ih hget h.2⟩

theorem noAborted_set_of_ne_aborted {ps : List SendPool} {idx : Nat} {p : SendPool} {st : SendStage}
    (hget : ps[idx]? = some p) (hst : st ≠ .aborted) (h : noAborted ps = true) :
    noAborted (ps.set idx { p with stage := st }) = true := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    simp only [noAborted, List.all_cons, Bool.and_eq_true] at h
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      simp [noAborted, hst, h.2]
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      simp only [noAborted, List.set_cons_succ, List.all_cons, Bool.and_eq_true]
      exact ⟨h.1, ih hget h.2⟩

theorem allPushed_eq_false_of_ne {ps : List SendPool} {idx : Nat} {p : SendPool}
    (hget : ps[idx]? = some p) (hne : p.stage ≠ .pushing 0) :
    allPushed ps = false := by
  induction ps generalizing idx with
  | nil => simp at hget
  | cons q qs ih =>
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
      subst q
      have hbeq : (p.stage == .pushing 0) = false := beq_false_of_ne hne
      simp [allPushed, hbeq]
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hget
      unfold allPushed at ih ⊢
      simp [ih hget]

theorem poolOps_pos_of_sumRemainingSlots_pos :
    ∀ {ps : List SendPool}, allIssued ps = true → 0 < sumRemainingSlots ps → 0 < poolOps ps
  | [], _, h => by simp [sumRemainingSlots] at h
  | p :: ps, hiss, hsum => by
    simp only [allIssued, List.all_cons, Bool.and_eq_true, bne_iff_ne, ne_eq] at hiss
    cases hst : p.stage with
    | pending => exact absurd hst hiss.1
    | d2h => simp only [poolOps, hst, SendStage.opCount]; omega
    | pushing k =>
      cases k with
      | zero =>
        simp only [sumRemainingSlots, SendPool.remainingSlots, hst, Nat.zero_add] at hsum
        have ih := poolOps_pos_of_sumRemainingSlots_pos hiss.2 hsum
        simp only [poolOps]; omega
      | succ k =>
        simp only [poolOps, hst, SendStage.opCount]; omega
    | aborted =>
      simp only [sumRemainingSlots, SendPool.remainingSlots, hst, Nat.zero_add] at hsum
      have ih := poolOps_pos_of_sumRemainingSlots_pos hiss.2 hsum
      simp only [poolOps]; omega

theorem allPushed_of_sumRemainingSlots_zero :
    ∀ {ps : List SendPool},
      allPosPeers ps = true →
      noAborted ps = true →
      sumRemainingSlots ps = 0 →
      allPushed ps = true
  | [], _, _, _ => rfl
  | p :: ps, hpos, hno, hsum => by
    simp only [allPosPeers, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hpos
    simp only [noAborted, List.all_cons, Bool.and_eq_true, bne_iff_ne, ne_eq] at hno
    simp only [sumRemainingSlots] at hsum
    have hp0 : p.remainingSlots = 0 := by omega
    have hps0 : sumRemainingSlots ps = 0 := by omega
    have ih := allPushed_of_sumRemainingSlots_zero hpos.2 hno.2 hps0
    cases hst : p.stage with
    | pending =>
      simp only [SendPool.remainingSlots, hst] at hp0
      omega
    | d2h =>
      simp only [SendPool.remainingSlots, hst] at hp0
      omega
    | pushing k =>
      simp only [SendPool.remainingSlots, hst] at hp0
      subst hp0
      unfold allPushed at ih ⊢
      simp [hst, ih]
    | aborted =>
      exact absurd hst hno.1

theorem exists_pending_of_allIssued_false :
    ∀ {ps : List SendPool}, allIssued ps = false →
      ∃ p < ps.length, ∃ pool, ps[p]? = some pool ∧ pool.stage = .pending
  | [], h => by simp [allIssued] at h
  | pool :: ps, h => by
    by_cases hp : pool.stage = .pending
    · exact ⟨0, by simp, pool, rfl, hp⟩
    · have hne : (pool.stage != .pending) = true := bne_iff_ne.mpr hp
      simp only [allIssued, List.all_cons, hne, Bool.true_and] at h
      obtain ⟨p, hlt, pool', hget, hp'⟩ := exists_pending_of_allIssued_false h
      exact ⟨p + 1, by simp; omega, pool', hget, hp'⟩

theorem exists_op_of_poolOps_pos :
    ∀ {ps : List SendPool}, 0 < poolOps ps →
      ∃ p < ps.length, ∃ pool, ps[p]? = some pool ∧ 0 < pool.stage.opCount
  | [], h => by simp [poolOps] at h
  | pool :: ps, h => by
    by_cases hp : 0 < pool.stage.opCount
    · exact ⟨0, by simp, pool, rfl, hp⟩
    · simp only [poolOps] at h
      have hps : 0 < poolOps ps := by omega
      obtain ⟨p, hlt, pool', hget, hp'⟩ := exists_op_of_poolOps_pos hps
      exact ⟨p + 1, by simp; omega, pool', hget, hp'⟩

/-- Pools the pipeline has not reached hold no op. -/
theorem poolOps_eq_zero_of_allPending :
    ∀ {ps : List SendPool}, allPending ps = true → poolOps ps = 0
  | [], _ => rfl
  | p :: ps, h => by
    simp only [allPending, List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp only [poolOps, h.1, SendStage.opCount, Nat.zero_add]
    exact poolOps_eq_zero_of_allPending h.2

/-- A non-empty schedule none of whose pools was reached has pushed nothing. -/
theorem allPushed_eq_false_of_allPending {ps : List SendPool}
    (hne : ps.isEmpty = false) (h : allPending ps = true) : allPushed ps = false := by
  cases ps with
  | nil => simp at hne
  | cons p ps =>
    simp only [allPending, List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp [allPushed, h.1]

/-- Pools the pipeline has not reached have not aborted. -/
theorem noAborted_of_allPending :
    ∀ {ps : List SendPool}, allPending ps = true → noAborted ps = true
  | [], _ => rfl
  | p :: ps, h => by
    simp only [allPending, List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp only [noAborted, List.all_cons, Bool.and_eq_true, h.1, bne_iff_ne, ne_eq,
      reduceCtorEq, not_false_eq_true, true_and]
    exact noAborted_of_allPending h.2

end Send

end TpuSyncVerify.Transfer.Reshard

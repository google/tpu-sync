import TpuSyncVerify.Transfer.Reshard.Receive.Lemmas

/-! # `Recv`: properties, inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.Reshard.Receive`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ## Properties -/

def SettleSafe (s : Recv) : Prop := s.life.done = true → s.life.inFlight = 0

def NoRetiredCallback (s : Recv) : Prop := s.life.done = true → poolOps s.pools = 0

def StagingIntegrity (s : Recv) : Prop := s.life.hasStaging = !s.life.done

def SettlesPromptly (s : Recv) : Prop :=
  s.life.draining = true → s.life.inFlight = 0 → s.life.done = true

/-- Every launched pool has all strictly lower-rank pools completed
(`pool_order_ranks_` comment in `ReshardReceiveSession`,
`CollectEligiblePoolH2dsLocked`). -/
def RankWaveOrder (s : Recv) : Prop := rankWaveOrder s.pools = true

theorem rankWaveOrder_iff {s : Recv} :
    RankWaveOrder s ↔
      ∀ p ∈ s.pools, ∀ q ∈ s.pools,
        p.stage.launched = true → q.rank < p.rank → q.stage = .completed := by
  unfold RankWaveOrder rankWaveOrder prereqsCompleted
  simp only [List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    Nat.not_lt, beq_iff_eq]
  constructor
  · intro h p hp q hq hl hlt
    rcases h p hp with hnl | hpre
    · rw [hl] at hnl; cases hnl
    · rcases hpre q hq with hge | hcomp
      · omega
      · exact hcomp
  · intro h p hp
    cases hl : p.stage.launched
    · exact Or.inl rfl
    · right
      intro q hq
      by_cases hlt : q.rank < p.rank
      · exact Or.inr (h p hp q hq hl hlt)
      · exact Or.inl (by omega)

def Publication (s : Recv) : Prop :=
  s.published = some true → allCompleted s.pools = true ∧ s.durationRecorded = true

/-- Events that advance or retire an in-flight operation for a receiver with `n`
pools. -/
def drainEvents (n : Nat) : List Ev :=
  [.pushEnd true] ++
  (List.range n).flatMap fun p => [.h2dIssue p true, .h2dReady p, .h2dDone p true]

def NoOpLeak (pol : PushEndPolicy) (s : Recv) : Prop :=
  0 < s.life.inFlight → ∃ e ∈ drainEvents s.pools.length, (step pol s e).isSome = true

def Safe (pol : PushEndPolicy) (s : Recv) : Prop :=
  SettleSafe s ∧ NoRetiredCallback s ∧ StagingIntegrity s ∧
    SettlesPromptly s ∧ RankWaveOrder s ∧ Publication s ∧ NoOpLeak pol s

/-! ## Inductive invariant -/

/-- Every unit of `in_flight_` comes from an active incoming push or a pool in
`.queued`, `.issued`, or `.ready`. -/
def Accounted (s : Recv) : Prop :=
  s.life.inFlight = s.pushes + poolOps s.pools

structure Inv (s : Recv) : Prop where
  life : s.life.Consistent
  accounted : Accounted s
  rankWave : rankWaveOrder s.pools = true
  ok_draining : s.life.statusOk = true → s.life.draining = true →
    allCompleted s.pools = true ∧ s.durationRecorded = true
  published_done : ∀ b, s.published = some b → s.life.done = true
  published_ok : s.published = some true →
    allCompleted s.pools = true ∧ s.durationRecorded = true

theorem poolOps_init : ∀ ranks : List Nat,
    poolOps (ranks.map fun r => { rank := r, stage := .waiting }) = 0
  | [] => rfl
  | _ :: rs => by simp [poolOps, PoolStage.holdsOp, poolOps_init rs]

theorem rankWaveOrder_init (ranks : List Nat) :
    rankWaveOrder (ranks.map fun r => { rank := r, stage := .waiting }) = true := by
  unfold rankWaveOrder
  rw [List.all_eq_true]
  intro p hp
  rw [List.mem_map] at hp
  obtain ⟨r, _, rfl⟩ := hp
  rfl

theorem inv_init (ranks : List Nat) : Inv (init ranks) := by
  refine ⟨Lifecycle.consistent_init, ?_, rankWaveOrder_init ranks, ?_, ?_, ?_⟩ <;>
    simp [init, Accounted, poolOps_init]

theorem exists_holdsOp_of_poolOps_pos : ∀ {ps : List Pool}, 0 < poolOps ps →
    ∃ p < ps.length, ∃ pool, ps[p]? = some pool ∧ pool.stage.holdsOp = true
  | [], h => by simp [poolOps] at h
  | pool :: ps, h => by
    by_cases hh : pool.stage.holdsOp = true
    · exact ⟨0, by simp, pool, rfl, hh⟩
    · simp only [poolOps, hh, Bool.false_eq_true, ↓reduceIte, Nat.zero_add] at h
      obtain ⟨p, hp, pool', hget, hh'⟩ := exists_holdsOp_of_poolOps_pos h
      exact ⟨p + 1, by simp; omega, pool', hget, hh'⟩

theorem mem_drainEvents_issue {n p : Nat} (hp : p < n) :
    Ev.h2dIssue p true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  exact Or.inr ⟨p, hp, by simp⟩

theorem mem_drainEvents_ready {n p : Nat} (hp : p < n) :
    Ev.h2dReady p ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  exact Or.inr ⟨p, hp, by simp⟩

theorem mem_drainEvents_done {n p : Nat} (hp : p < n) :
    Ev.h2dDone p true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  exact Or.inr ⟨p, hp, by simp⟩

theorem inv_noOpLeak (pol : PushEndPolicy) {s : Recv} (h : Inv s) : NoOpLeak pol s := by
  intro hif
  have hacc := h.accounted
  unfold Accounted at hacc
  by_cases hp : 0 < s.pushes
  · refine ⟨.pushEnd true, by simp [drainEvents], ?_⟩
    have hne : s.pushes ≠ 0 := by omega
    simp [step, pushEnd, hne]
  · have hpo : 0 < poolOps s.pools := by omega
    obtain ⟨p, hlt, pool, hget, hh⟩ := exists_holdsOp_of_poolOps_pos hpo
    cases hst : pool.stage with
    | queued =>
      refine ⟨.h2dIssue p true, mem_drainEvents_issue hlt, ?_⟩
      simp only [step, h2dIssue, hget, hst, PoolStage.bne_self, Bool.false_eq_true, ↓reduceIte]
      split <;> rfl
    | issued =>
      refine ⟨.h2dReady p, mem_drainEvents_ready hlt, ?_⟩
      simp [step, h2dReady, hget, hst]
    | ready =>
      refine ⟨.h2dDone p true, mem_drainEvents_done hlt, ?_⟩
      simp only [step, h2dDone, hget, hst, PoolStage.bne_self, Bool.false_eq_true, ↓reduceIte,
        Bool.not_true]
      split <;> try rfl
      split <;> try rfl
      split <;> rfl
    | waiting | received | completed | aborted =>
      simp [hst, PoolStage.holdsOp] at hh

theorem inv_safe (pol : PushEndPolicy) {s : Recv} (h : Inv s) : Safe pol s := by
  have hnl := inv_noOpLeak pol h
  have hacc := h.accounted
  unfold Accounted at hacc
  refine ⟨h.life.done_idle, ?_, h.life.staging, h.life.prompt, h.rankWave, h.published_ok, hnl⟩
  intro hd
  have := h.life.done_idle hd
  omega

theorem consistent_add_inFlight {l : Lifecycle} (h : l.Consistent) (hdr : l.draining = false)
    (k : Nat) : Lifecycle.Consistent { l with inFlight := l.inFlight + k } := by
  have hnd := h.not_done hdr
  constructor <;> simp [hdr, hnd, h.staging]

theorem launchEligible_inv {s : Recv} (h : Inv s) : Inv (launchEligible s) := by
  unfold launchEligible
  split
  · exact h
  · rename_i hact
    simp only [not_or, Bool.not_eq_true] at hact
    obtain ⟨hnd, hdr, hnf⟩ := hact
    refine ⟨consistent_add_inFlight h.life hdr _, ?_,
      launchPass_preserves_rankWaveOrder s.pools h.rankWave, ?_, ?_, ?_⟩
    · unfold Accounted
      have hacc := h.accounted; unfold Accounted at hacc
      have hpo := launchPass_poolOps (prereqsCompleted s.pools) s.pools
      simp only; omega
    · intro _ hdr'
      simp [hdr] at hdr'
    · intro b hb
      have hd := h.published_done b hb
      simp [hnd] at hd
    · intro hpub
      have hd := h.published_done true hpub
      simp [hnd] at hd

theorem step_inv (pol : PushEndPolicy) {s s' : Recv} {e : Ev}
    (h : Inv s) (hs : step pol s e = some s') : Inv s' := by
  cases e with
  | pushBegin =>
    simp only [step, pushBegin, Option.map_eq_some_iff] at hs
    obtain ⟨l, hb, rfl⟩ := hs
    have hif := Lifecycle.beginOp_inFlight hb
    have hact := Lifecycle.beginOp_active hb
    have hok := Lifecycle.beginOp_statusOk hb
    have hdr := Lifecycle.beginOp_draining hb
    have hacc := h.accounted; unfold Accounted at hacc
    refine ⟨Lifecycle.beginOp_consistent h.life hb, ?_, h.rankWave, ?_, ?_, h.published_ok⟩
    · unfold Accounted; simp; omega
    · simp [hdr]
    · intro b hb'
      have hd := h.published_done b hb'
      simp [hact.1] at hd
  | pushEnd ok =>
    simp only [step, pushEnd] at hs
    split at hs
    · cases hs
    · rename_i hne
      cases hs
      have hacc := h.accounted; unfold Accounted at hacc
      refine ⟨?_, ?_, h.rankWave, ?_, ?_, h.published_ok⟩
      · apply Lifecycle.endOpLocked_consistent
        split
        · exact Lifecycle.finishLocked_consistent false h.life
        · exact h.life
      · unfold Accounted
        split <;> simp <;> omega
      · dsimp only
        split
        · simp
        · simp only [Lifecycle.endOpLocked_statusOk, Lifecycle.endOpLocked_draining]
          exact h.ok_draining
      · intro b hb
        have hd := h.published_done b hb
        apply Lifecycle.endOpLocked_done_mono
        split <;> simp [hd, Lifecycle.finishLocked_done_mono]
  | onPoolReceived p =>
    simp only [step, onPoolReceived] at hs
    split at hs
    · rename_i pool hget
      split at hs
      · cases hs
      · rename_i hact
        cases hs
        simp only [not_or, Bool.not_eq_true, bne_iff_ne, ne_eq, Decidable.not_not] at hact
        obtain ⟨hnd, hwait⟩ := hact
        apply launchEligible_inv
        have hne : pool.stage ≠ .completed := by simp [hwait]
        have hall_false := allCompleted_eq_false_of_ne hget hne
        refine ⟨h.life, ?_, ?_, ?_, h.published_done, ?_⟩
        · unfold Accounted
          have hacc := h.accounted; unfold Accounted at hacc
          have hpo := poolOps_set_eq hget (p' := { pool with stage := .received })
            (by rw [hwait]; rfl)
          simp [hpo, hacc]
        · dsimp only
          rw [rankWaveOrder_set_same hget (by rw [hwait]; rfl) (by rw [hwait]; rfl)]
          exact h.rankWave
        · intro hok hdr
          have ⟨hc, _⟩ := h.ok_draining hok hdr
          simp [hall_false] at hc
        · intro hpub
          have ⟨hc, _⟩ := h.published_ok hpub
          simp [hall_false] at hc
    · cases hs
  | h2dIssue p ok =>
    simp only [step, h2dIssue] at hs
    split at hs
    · rename_i pool hget
      split at hs
      · cases hs
      · rename_i hq
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hq
        have hne : pool.stage ≠ .completed := by simp [hq]
        have hall_false := allCompleted_eq_false_of_ne hget hne
        have hacc := h.accounted; unfold Accounted at hacc
        split at hs
        · cases hs
          have ⟨hpo, hpos⟩ := poolOps_set_drop hget (p' := { pool with stage := .aborted })
            (by rw [hq]; rfl) rfl
          refine ⟨Lifecycle.endOpLocked_consistent h.life, ?_, ?_, ?_, ?_, ?_⟩
          · unfold Accounted; simp; omega
          · dsimp only
            rw [rankWaveOrder_set_same hget (by rw [hq]; rfl) (by rw [hq]; rfl)]
            exact h.rankWave
          · dsimp only
            simp only [Lifecycle.endOpLocked_statusOk, Lifecycle.endOpLocked_draining]
            intro hok hdr; have ⟨hc, _⟩ := h.ok_draining hok hdr; simp [hall_false] at hc
          · exact fun b hb => Lifecycle.endOpLocked_done_mono (h.published_done b hb)
          · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
        · rename_i hact
          simp only [not_or, Bool.not_eq_true] at hact
          obtain ⟨hnd, hdr⟩ := hact
          split at hs
          · cases hs
            have hpo := poolOps_set_eq hget (p' := { pool with stage := .issued })
              (by rw [hq]; rfl)
            refine ⟨h.life, ?_, ?_, ?_, h.published_done, ?_⟩
            · unfold Accounted; simp [hpo, hacc]
            · dsimp only
              rw [rankWaveOrder_set_same hget (by rw [hq]; rfl) (by rw [hq]; rfl)]
              exact h.rankWave
            · intro _ hdr'; simp [hdr] at hdr'
            · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
          · cases hs
            have ⟨hpo, hpos⟩ := poolOps_set_drop hget (p' := { pool with stage := .aborted })
              (by rw [hq]; rfl) rfl
            refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
            · apply Lifecycle.endOpLocked_consistent
              split
              · exact h.life
              · exact Lifecycle.finishLocked_consistent false h.life
            · unfold Accounted; split <;> simp <;> omega
            · dsimp only
              rw [rankWaveOrder_set_same hget (by rw [hq]; rfl) (by rw [hq]; rfl)]
              exact h.rankWave
            · split
              · intro _ hdr'; simp [hdr] at hdr'
              · simp
            · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
            · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
    · cases hs
  | h2dReady p =>
    simp only [step, h2dReady] at hs
    split at hs
    · rename_i pool hget
      split at hs
      · rename_i his
        cases hs
        simp only [beq_iff_eq] at his
        have hne : pool.stage ≠ .completed := by simp [his]
        have hall_false := allCompleted_eq_false_of_ne hget hne
        have hacc := h.accounted; unfold Accounted at hacc
        have hpo := poolOps_set_eq hget (p' := { pool with stage := .ready })
          (by rw [his]; rfl)
        refine ⟨h.life, ?_, ?_, ?_, h.published_done, ?_⟩
        · unfold Accounted; simp [hpo, hacc]
        · dsimp only
          rw [rankWaveOrder_set_same hget (by rw [his]; rfl) (by rw [his]; rfl)]
          exact h.rankWave
        · intro hok hdr; have ⟨hc, _⟩ := h.ok_draining hok hdr; simp [hall_false] at hc
        · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
      · cases hs
    · cases hs
  | h2dDone p ok =>
    simp only [step, h2dDone] at hs
    split at hs
    · rename_i pool hget
      split at hs
      · cases hs
      · rename_i hrd
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hrd
        have hne : pool.stage ≠ .completed := by simp [hrd]
        have hall_false := allCompleted_eq_false_of_ne hget hne
        have hacc := h.accounted; unfold Accounted at hacc
        have ⟨hpo, hpos⟩ := poolOps_set_drop hget (p' := { pool with stage := .aborted })
          (by rw [hrd]; rfl) rfl
        split at hs
        · rename_i hd
          have h0 := h.life.done_idle hd
          omega
        · rename_i hnd
          simp only [Bool.not_eq_true] at hnd
          split at hs
          · cases hs
            refine ⟨Lifecycle.endOpLocked_consistent h.life, ?_, ?_, ?_, ?_, ?_⟩
            · unfold Accounted; simp; omega
            · dsimp only
              rw [rankWaveOrder_set_same hget (by rw [hrd]; rfl) (by rw [hrd]; rfl)]
              exact h.rankWave
            · dsimp only
              simp only [Lifecycle.endOpLocked_statusOk, Lifecycle.endOpLocked_draining]
              intro hok hdr; have ⟨hc, _⟩ := h.ok_draining hok hdr; simp [hall_false] at hc
            · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
            · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
          · rename_i hnf
            simp only [Bool.not_eq_true] at hnf
            split at hs
            · cases hs
              refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishLocked_consistent false h.life),
                ?_, ?_, ?_, ?_, ?_⟩
              · unfold Accounted; simp; omega
              · dsimp only
                rw [rankWaveOrder_set_same hget (by rw [hrd]; rfl) (by rw [hrd]; rfl)]
                exact h.rankWave
              · simp
              · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
              · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
            · have ⟨hpo_c, _⟩ := poolOps_set_drop hget (p' := { pool with stage := .completed })
                (by simp [hrd, PoolStage.holdsOp]) rfl
              have hrw_c := rankWaveOrder_set_completed hget (by simp [hrd, PoolStage.launched])
                h.rankWave
              split at hs
              · rename_i hall_c
                cases hs
                refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishLocked_consistent true h.life),
                  ?_, hrw_c, ?_, ?_, ?_⟩
                · unfold Accounted; simp; omega
                · intro _ _; exact ⟨hall_c, rfl⟩
                · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
                · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
              · rename_i hnot_all
                cases hs
                simp only [Bool.not_eq_true] at hnot_all
                unfold launchEligible
                split
                · rename_i hact
                  refine ⟨Lifecycle.endOpLocked_consistent h.life, ?_, hrw_c, ?_, ?_, ?_⟩
                  · unfold Accounted; simp; omega
                  · dsimp only
                    simp only [Lifecycle.endOpLocked_statusOk, Lifecycle.endOpLocked_draining]
                    intro hok hdr; have ⟨hc, _⟩ := h.ok_draining hok hdr; simp [hall_false] at hc
                  · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
                  · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
                · rename_i hact
                  simp only [not_or, Bool.not_eq_true] at hact
                  obtain ⟨_, hdr, _⟩ := hact
                  have hpo_l := launchPass_poolOps
                    (prereqsCompleted (s.pools.set p { pool with stage := .completed }))
                    (s.pools.set p { pool with stage := .completed })
                  have hrw_l := launchPass_preserves_rankWaveOrder
                    (s.pools.set p { pool with stage := .completed }) hrw_c
                  refine ⟨Lifecycle.endOpLocked_consistent
                      (consistent_add_inFlight h.life hdr _),
                    ?_, hrw_l, ?_, ?_, ?_⟩
                  · unfold Accounted; dsimp only; simp only [Lifecycle.endOpLocked_inFlight]; omega
                  · intro _ hdr'; simp [hdr] at hdr'
                  · intro b hb; have hd := h.published_done b hb; simp [hnd] at hd
                  · intro hpub; have ⟨hc, _⟩ := h.published_ok hpub; simp [hall_false] at hc
    · cases hs
  | cancel =>
    simp only [step, cancel] at hs
    cases hs
    have hacc := h.accounted; unfold Accounted at hacc
    refine ⟨Lifecycle.finishLocked_consistent false h.life, ?_, h.rankWave, ?_, ?_,
      h.published_ok⟩
    · unfold Accounted; simpa using hacc
    · simp
    · exact fun b hb => Lifecycle.finishLocked_done_mono false (h.published_done b hb)
  | publish =>
    simp only [step, publish] at hs
    split at hs
    · rename_i hd
      obtain ⟨hdone, _⟩ := hd
      cases hs
      refine ⟨h.life, h.accounted, h.rankWave, h.ok_draining, fun _ _ => hdone, ?_⟩
      simp only [Option.some.injEq]
      intro hok
      exact h.ok_draining hok (h.life.done_draining hdone)
    · cases hs

theorem reachable_inv {ranks : List Nat} {pol : PushEndPolicy} {s : Recv}
    (h : (sys ranks pol).Reachable s) : Inv s :=
  (sys ranks pol).reachable_induction (inv_init ranks) (fun _ _ _ hi hs => step_inv pol hi hs) h

/-- Main result: every reachable reshard receive session satisfies all safety
properties under both `PushEndPolicy.finishOnError` and `PushEndPolicy.endOpOnly`. -/
theorem reachable_safe {ranks : List Nat} {pol : PushEndPolicy} {s : Recv}
    (h : (sys ranks pol).Reachable s) : Safe pol s :=
  inv_safe pol (reachable_inv h)

end Recv

end TpuSyncVerify.Transfer.Reshard

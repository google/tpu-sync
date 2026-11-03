import TpuSyncVerify.Transfer.PrefillDecode.Send.Model

/-! # `Send`: inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.Send`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Inductive invariant -/

/-- Where every unit of `in_flight_` comes from: a dispatch in progress, a copy
whose `OnReady` has not ended it, a `SendNextLayer` waiting on its future, a
layer waiting for the pool, a pool task that has not released its op, or a
push whose callback has not run. -/
def Accounted (s : Send) : Prop :=
  s.life.inFlight =
    (if s.d2hPending then 1 else 0) + (s.d2hIssued - s.d2hRetired) + (s.queued - s.woken) +
      s.pooled + s.chaining + (s.h2hIssued - s.h2hRetired)

/-- Single-baton invariant of `TransferSendSession::SendNextLayer`'s push chain:
at most one layer is waiting on its future (`queued - woken`), queued on
`push_pool` (`pooled`), or chaining into the next layer (`chaining`). -/
structure ChainInv (s : Send) : Prop where
  chain_le : (s.queued - s.woken) + s.pooled + s.chaining ≤ 1
  woken_bounds : s.h2hIssued ≤ s.woken ∧ s.woken ≤ s.h2hIssued + 1
  pooled_state : s.pooled = 1 → s.woken = s.h2hIssued + 1 ∧ s.h2hIssued < s.d2hIssued
  chaining_state : s.chaining = 1 → s.woken = s.h2hIssued
  woken_ahead : s.woken = s.h2hIssued + 1 →
    s.queued = s.woken ∧ s.chaining = 0 ∧ (s.pooled = 1 ∨ s.life.draining = true)

structure Inv (s : Send) : Prop where
  life : s.life.Consistent
  accounted : Accounted s
  no_underflow : s.underflow = false
  counters : CountersOrdered s
  pull_claimed : PullClaimed s
  /-- `SendNextLayer(0)` is armed by `StartPush`: no op is queued before it started. -/
  queued_started : 0 < s.queued → s.started = true
  /-- The chain only runs past the issued copies once the D2H loop has returned
  and `fail_unissued_layers` resolved the rest: the send is draining and no
  dispatch is in progress. -/
  woken_issued : s.d2hIssued < s.woken → s.life.draining = true ∧ s.d2hPending = false
  /-- A send that started draining without an error had pushed every layer. -/
  ok_draining : s.life.statusOk = true → s.life.draining = true → s.h2hOk = s.numLayers
  /-- Only settled sessions are published. -/
  published_done : ∀ b, s.published = some b → s.life.done = true
  /-- Once published as done, every layer had been pushed (and still has). -/
  published_ok : s.published = some true → s.h2hOk = s.numLayers
  /-- Single-baton invariant of `SendNextLayer`'s sequential push chain. -/
  chain : ChainInv s

theorem inv_init (n : Nat) : Inv (init n) := by
  refine ⟨Lifecycle.consistent_init, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ⟨?_, ?_, ?_, ?_, ?_⟩⟩ <;>
    simp [init, Accounted, CountersOrdered, PullClaimed]

theorem inv_noOpLeak {s : Send} (h : Inv s) : NoOpLeak s := by
  intro hif
  have hacc := h.accounted
  have hcnt := h.counters
  unfold Accounted at hacc
  unfold CountersOrdered at hcnt
  by_cases hp : s.d2hPending = true
  · exact ⟨.d2hIssue true, by simp [drainEvents], by simp [step, d2hIssue, hp]⟩
  · have hp' : s.d2hPending = false := by
      cases hq : s.d2hPending
      · rfl
      · exact absurd hq hp
    by_cases hrd : s.d2hReady < s.d2hIssued
    · exact ⟨.d2hReady, by simp [drainEvents], by simp [step, markD2hReady, hrd]⟩
    · by_cases hret : s.d2hRetired < s.d2hReady
      · exact ⟨.d2hEnd, by simp [drainEvents], by simp [step, d2hEnd, hret]⟩
      · by_cases hwk : s.woken < s.queued
        · by_cases hwr : s.woken < s.d2hReady
          · refine ⟨.wake true, by simp [drainEvents], ?_⟩
            simp only [step, wake, hwk, hwr, true_or, and_self, ↓reduceIte, Bool.true_eq_false]
            split <;> rfl
          · -- layer `woken` has no copy yet: the loop issues it, or, draining,
            -- `fail_unissued_layers` resolves its promise
            have hle : s.d2hIssued ≤ s.woken := by omega
            cases hdr : s.life.draining
            · have hst : s.started = true := h.queued_started (by omega)
              have hnd : s.life.done = false := by
                cases hd : s.life.done
                · rfl
                · have := h.life.done_draining hd; rw [hdr] at this; cases this
              refine ⟨.d2hBegin, by simp [drainEvents], ?_⟩
              have hlt : s.d2hIssued < s.numLayers := by omega
              simp [step, d2hBegin, hst, hp', hlt, Lifecycle.beginOp, hdr, hnd]
            · refine ⟨.wake false, by simp [drainEvents], ?_⟩
              simp [step, wake, hwk, hdr, hp', hle]
        · by_cases hpool : 0 < s.pooled
          · have hne : s.pooled ≠ 0 := by omega
            refine ⟨.h2hIssue, by simp [drainEvents], ?_⟩
            simp only [step, h2hIssue, hne, ↓reduceIte]
            split <;> rfl
          · by_cases hch : 0 < s.chaining
            · exact ⟨.sendNext, by simp [drainEvents], by simp [step, sendNext]; omega⟩
            · have hh2h : s.h2hRetired < s.h2hIssued := by
                simp [hp'] at hacc; omega
              exact ⟨.h2hDone true, by simp [drainEvents], by simp [step, h2hDone, hh2h]⟩

theorem inv_safe {s : Send} (h : Inv s) : Safe s := by
  have hnl := inv_noOpLeak h
  obtain ⟨hl, hacc, hu, hcnt, hpc, _, _, _, hpd, hpub, _⟩ := h
  unfold Accounted at hacc
  unfold CountersOrdered at hcnt
  refine ⟨hl.done_idle, ?_, hl.staging, hl.prompt, hu, hpub, hcnt, hnl, hpc, ?_⟩
  · intro hd
    have h0 := hl.done_idle hd
    cases hp : s.d2hPending <;> simp [hp] at hacc ⊢ <;> omega
  · intro hp
    have hd : s.life.done = true := by
      cases hpb : s.published with
      | none => exact absurd hpb hp
      | some b => exact hpd b hpb
    have h0 := hl.done_idle hd
    cases hdp : s.d2hPending <;> simp [hdp] at hacc ⊢ <;> omega

/-! ### Helpers for the step proof -/

theorem endOp_eq {s : Send} (h : 0 < s.life.inFlight) :
    s.endOp = { s with life := s.life.endOpLocked } := by
  unfold endOp
  split
  · omega
  · rfl

theorem trySendNext_cases (s : Send) :
    s.trySendNext = s ∨
    (s.queued < s.numLayers ∧ s.life.draining = false ∧ s.life.done = false ∧
      s.trySendNext = { s with life := { s.life with inFlight := s.life.inFlight + 1 },
                               queued := s.queued + 1 }) := by
  unfold trySendNext
  split
  · rename_i hlt
    split
    · rename_i l hb
      right
      obtain ⟨hd, hdr⟩ := Lifecycle.beginOp_active hb
      refine ⟨hlt, hdr, hd, ?_⟩
      unfold Lifecycle.beginOp at hb
      split at hb
      · cases hb
      · cases hb; rfl
    · left; rfl
  · left; rfl

private theorem endOp_draining (s : Send) (hdr : s.life.draining = true) :
    s.endOp.life.draining = true := by
  unfold endOp Lifecycle.endOpLocked Lifecycle.settleLocked
  split
  · exact hdr
  · split <;> exact hdr

private theorem chainInv_of_same {s s' : Send} (hc : ChainInv s)
    (hQ : s'.queued = s.queued) (hW : s'.woken = s.woken) (hP : s'.pooled = s.pooled)
    (hC : s'.chaining = s.chaining) (hH : s'.h2hIssued = s.h2hIssued)
    (hD : s.d2hIssued ≤ s'.d2hIssued)
    (hDr : s.life.draining = true → s'.life.draining = true) :
    ChainInv s' := by
  refine ⟨by rw [hQ, hW, hP, hC]; exact hc.chain_le,
    by rw [hH, hW]; exact hc.woken_bounds,
    fun hp => by obtain ⟨hw, hlt⟩ := hc.pooled_state (hP ▸ hp); exact ⟨by omega, by omega⟩,
    fun hch => by have := hc.chaining_state (hC ▸ hch); omega,
    fun ha => ?_⟩
  obtain ⟨hq, hch, hpdr⟩ := hc.woken_ahead (by omega)
  exact ⟨by omega, by omega, hpdr.imp (fun hp => hP ▸ hp) hDr⟩

@[simp] private theorem finish_queued (ok : Bool) (s : Send) : (s.finish ok).queued = s.queued := rfl
@[simp] private theorem finish_woken (ok : Bool) (s : Send) : (s.finish ok).woken = s.woken := rfl
@[simp] private theorem finish_pooled (ok : Bool) (s : Send) : (s.finish ok).pooled = s.pooled := rfl
@[simp] private theorem finish_chaining (ok : Bool) (s : Send) : (s.finish ok).chaining = s.chaining := rfl
@[simp] private theorem finish_h2hIssued (ok : Bool) (s : Send) : (s.finish ok).h2hIssued = s.h2hIssued := rfl
@[simp] private theorem finish_d2hIssued (ok : Bool) (s : Send) : (s.finish ok).d2hIssued = s.d2hIssued := rfl
@[simp] private theorem endOp_queued (s : Send) : s.endOp.queued = s.queued := by unfold endOp; split <;> rfl
@[simp] private theorem endOp_woken (s : Send) : s.endOp.woken = s.woken := by unfold endOp; split <;> rfl
@[simp] private theorem endOp_pooled (s : Send) : s.endOp.pooled = s.pooled := by unfold endOp; split <;> rfl
@[simp] private theorem endOp_chaining (s : Send) : s.endOp.chaining = s.chaining := by unfold endOp; split <;> rfl
@[simp] private theorem endOp_h2hIssued (s : Send) : s.endOp.h2hIssued = s.h2hIssued := by unfold endOp; split <;> rfl
@[simp] private theorem endOp_d2hIssued (s : Send) : s.endOp.d2hIssued = s.d2hIssued := by unfold endOp; split <;> rfl

theorem step_chainInv {s s' : Send} {e : Ev}
    (hl : s.life.Consistent) (hcnt : CountersOrdered s) (hacc : Accounted s)
    (hqs : 0 < s.queued → s.started = true) (hc : ChainInv s)
    (hs : step s e = some s') : ChainInv s' := by
  unfold CountersOrdered at hcnt; unfold Accounted at hacc
  cases e with
  | beginPull =>
    simp only [step, beginPull] at hs; split at hs <;> cases hs
    exact chainInv_of_same hc rfl rfl rfl rfl rfl (Nat.le_refl _) id
  | start =>
    simp only [step, start] at hs; split at hs
    · cases hs
    · rename_i hng
      have hst : s.started = false := by cases h : s.started <;> simp_all
      have hq0 : s.queued = 0 := by
        cases hq : s.queued with
        | zero => rfl
        | succ _ => have := hqs (by omega); simp [hst] at this
      have hw0 : s.woken = 0 := by omega
      have hh0 : s.h2hIssued = 0 := by omega
      have hp0 : s.pooled = 0 := by omega
      have hch0 : s.chaining = 0 := by omega
      split at hs <;> simp only [Option.some.injEq] at hs <;> subst hs
      · refine ⟨by simp; omega, by simp; omega, by simp [hp0], by simp [hch0], fun ha => by simp [hw0, hh0] at ha⟩
      · rcases trySendNext_cases { s with started := true } with heq | ⟨_, _, _, heq⟩ <;>
          (rw [heq]; refine ⟨by simp [hq0, hw0, hp0, hch0], by simp [hw0, hh0],
            by simp [hp0], by simp [hch0], fun ha => by simp [hw0, hh0] at ha⟩)
  | d2hBegin =>
    simp only [step, d2hBegin] at hs; split at hs <;> try cases hs
    rcases Option.map_eq_some_iff.mp hs with ⟨l, hb, rfl⟩
    obtain ⟨_, hdr0⟩ := Lifecycle.beginOp_active hb
    exact chainInv_of_same hc rfl rfl rfl rfl rfl (Nat.le_refl _) (fun h => by simp [hdr0] at h)
  | d2hIssue ok =>
    simp only [step, d2hIssue] at hs; split at hs
    · cases hs
    · split at hs <;> simp only [Option.some.injEq] at hs <;> subst hs
      · exact chainInv_of_same hc rfl rfl rfl rfl rfl (Nat.le_succ _) id
      · exact chainInv_of_same hc (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
          (fun _ => endOp_draining _ (hl.finishOnceLocked_draining false))
  | d2hReady =>
    simp only [step, markD2hReady] at hs; split at hs <;> cases hs
    exact chainInv_of_same hc rfl rfl rfl rfl rfl (Nat.le_refl _) id
  | d2hEnd =>
    simp only [step, d2hEnd] at hs; split at hs <;> cases hs
    exact chainInv_of_same hc (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
      (fun hdr => endOp_draining _ hdr)
  | wake ok =>
    simp only [step, wake] at hs; split at hs
    · rename_i hg; obtain ⟨_, hbr⟩ := hg
      have hp0 : s.pooled = 0 := by have := hc.chain_le; omega
      have hch0 : s.chaining = 0 := by have := hc.chain_le; omega
      have hwh : s.woken = s.h2hIssued := by
        have := hc.woken_bounds
        by_cases heq : s.woken = s.h2hIssued + 1
        · have := (hc.woken_ahead heq).1; omega
        · omega
      have hqw : s.queued = s.woken + 1 := by have := hc.chain_le; omega
      split at hs
      · simp only [Option.some.injEq] at hs; subst hs
        refine ⟨by simp [hp0, hch0]; omega, by simp [hwh], by simp [hp0], by simp [hch0],
          fun _ => ⟨by simp [hqw], by simp [hch0],
            Or.inr (endOp_draining _ (hl.finishOnceLocked_draining false))⟩⟩
      · split at hs <;> rename_i hok hdr <;> simp only [Option.some.injEq] at hs <;> subst hs
        · refine ⟨by simp [hp0, hch0]; omega, by simp [hwh], by simp [hp0], by simp [hch0],
            fun _ => ⟨by simp [hqw], by simp [hch0], Or.inr (endOp_draining _ hdr)⟩⟩
        · have hdr_f : s.life.draining = false := by cases h : s.life.draining <;> simp_all
          have hlt : s.h2hIssued < s.d2hIssued := by
            rcases hbr with hwr | ⟨_, hdr_t, _, _⟩
            · omega
            · simp [hdr_f] at hdr_t
          exact ⟨by simp; omega, by simp; omega, fun _ => ⟨by simp [hwh], hlt⟩,
            by simp [hch0], fun _ => ⟨by simp [hqw], hch0, Or.inl (by simp [hp0])⟩⟩
    · cases hs
  | h2hIssue =>
    simp only [step, h2hIssue] at hs; split at hs
    · cases hs
    · have hp1 : s.pooled = 1 := by have := hc.chain_le; omega
      obtain ⟨hwh, _⟩ := hc.pooled_state hp1
      have hqw : s.queued = s.woken := by have := hc.chain_le; omega
      have hch0 : s.chaining = 0 := by have := hc.chain_le; omega
      split at hs <;> simp only [Option.some.injEq] at hs <;> subst hs
      · exact ⟨by simp [hp1, hqw, hch0], by simp [hwh], by simp [hp1],
          fun _ => by simp [hwh], fun ha => by simp [hwh] at ha⟩
      · rename_i hnone
        have hnd : s.life.done = false := by
          cases hd : s.life.done
          · rfl
          · have := hl.done_idle hd; omega
        have hdr : s.life.draining = true := by
          unfold Lifecycle.beginOp at hnone; split at hnone
          · rename_i hc0; exact hc0.elim (fun hd => by simp [hnd] at hd) id
          · cases hnone
        exact ⟨by simp [hp1, hqw, hch0], by simp [hwh], by simp [hp1], by simp [hch0],
          fun _ => ⟨by simp [hqw], by simp [hch0], Or.inr (endOp_draining _ hdr)⟩⟩
  | sendNext =>
    simp only [step, sendNext] at hs; split at hs
    · cases hs
    · simp only [Option.some.injEq] at hs; subst hs
      have hch1 : s.chaining = 1 := by have := hc.chain_le; omega
      have hwh : s.woken = s.h2hIssued := hc.chaining_state hch1
      have hqw : s.queued = s.woken := by have := hc.chain_le; omega
      have hp0 : s.pooled = 0 := by have := hc.chain_le; omega
      rcases trySendNext_cases { s with chaining := s.chaining - 1 } with heq | ⟨_, _, _, heq⟩ <;>
        (rw [heq]; exact ⟨by simp [hqw, hp0, hch1], by simp [hwh], by simp [hp0],
          by simp [hch1], fun ha => by simp [hwh] at ha⟩)
  | h2hDone ok =>
    simp only [step, h2hDone] at hs; split at hs <;> try cases hs
    split at hs <;> simp only [Option.some.injEq] at hs <;> subst hs
    · split
      · exact chainInv_of_same hc (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
          (fun _ => endOp_draining _ (hl.finishOnceLocked_draining true))
      · exact chainInv_of_same hc (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
          (fun hdr => endOp_draining _ hdr)
    · exact chainInv_of_same hc (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
        (fun _ => endOp_draining _ (hl.finishOnceLocked_draining false))
  | cancel =>
    cases hs
    exact chainInv_of_same hc (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
      (fun _ => hl.finishOnceLocked_draining false)
  | publish =>
    simp only [step, publish] at hs; split at hs <;> cases hs
    exact chainInv_of_same hc rfl rfl rfl rfl rfl (Nat.le_refl _) id

/-- Each event preserves the invariant. -/
theorem step_inv {s s' : Send} {e : Ev} (h : Inv s) (hs : step s e = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hcnt, hpc, hqs, hwi, hdr, hpd, hpub, hc⟩ := h
  have hc' : ChainInv s' := step_chainInv hl hcnt hacc hqs hc hs
  suffices hb :
      s'.life.Consistent ∧ Accounted s' ∧ s'.underflow = false ∧ CountersOrdered s' ∧
      PullClaimed s' ∧ (0 < s'.queued → s'.started = true) ∧
      (s'.d2hIssued < s'.woken → s'.life.draining = true ∧ s'.d2hPending = false) ∧
      (s'.life.statusOk = true → s'.life.draining = true → s'.h2hOk = s'.numLayers) ∧
      (∀ b, s'.published = some b → s'.life.done = true) ∧
      (s'.published = some true → s'.h2hOk = s'.numLayers) by
    obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := hb
    exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, hc'⟩
  unfold Accounted at hacc
  unfold CountersOrdered at hcnt
  cases e with
  | beginPull =>
    simp only [step, beginPull] at hs
    split at hs
    · cases hs
    · cases hs
      refine ⟨hl, by unfold Accounted; simpa using hacc, hu,
        by unfold CountersOrdered; simpa using hcnt,
        ⟨fun _ => rfl, hpc.2⟩, hqs, hwi, hdr, hpd, hpub⟩
  | start =>
    simp only [step, start] at hs
    split at hs
    · cases hs
    · rename_i hng
      have hpulled : s.pullStarted = true := by
        cases hps : s.pullStarted
        · exact absurd (Or.inl hps) hng
        · rfl
      have hdr0 : s.life.draining = false := by
        cases hq : s.life.draining
        · rfl
        · exact absurd (Or.inr (Or.inr (Or.inl hq))) hng
      split at hs
      · cases hs
        dsimp only [finish]
        refine ⟨Lifecycle.finishOnceLocked_consistent true hl,
          by unfold Accounted; simpa using hacc, hu, hcnt, hpc, hqs, ?_,
          by intro _ _; omega,
          fun b hp => Lifecycle.finishOnceLocked_done_mono true (hpd b hp), hpub⟩
        intro hlt
        obtain ⟨_, hp⟩ := hwi hlt
        exact ⟨hl.finishOnceLocked_draining true, hp⟩
      · cases hs
        rcases trySendNext_cases { s with started := true } with heq | ⟨hlt, hdr', hd', heq⟩ <;>
          rw [heq]
        · exact ⟨hl, by unfold Accounted; simpa using hacc, hu,
            by unfold CountersOrdered; simpa using hcnt,
            ⟨fun _ => hpulled, fun _ => rfl⟩, fun _ => rfl, hwi, hdr, hpd, hpub⟩
        · simp at hlt hdr' hd'
          refine ⟨Lifecycle.consistent_incr hl hdr', ?_, hu, ?_,
            ⟨fun _ => hpulled, fun _ => rfl⟩, fun _ => rfl, ?_, ?_, ?_, hpub⟩
          · unfold Accounted; simp; omega
          · unfold CountersOrdered; simp; omega
          · intro hlt'
            simp at hlt'
            have := (hwi (by omega)).1
            rw [hdr0] at this; cases this
          · simp [hdr']
          · intro b hq
            have := hpd b hq
            simp [hd'] at this
  | d2hBegin =>
    simp only [step, d2hBegin] at hs
    split at hs
    · rename_i hg
      obtain ⟨hst, hp, _⟩ := hg
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨l, hb, rfl⟩ := hs
      have := Lifecycle.beginOp_inFlight hb
      have hact := Lifecycle.beginOp_active hb
      have hdr' := Lifecycle.beginOp_draining hb
      refine ⟨Lifecycle.beginOp_consistent hl hb, ?_, hu, ?_, ⟨hpc.1, fun _ => hst⟩,
        fun _ => hst, ?_, ?_, ?_, hpub⟩
      · unfold Accounted; simp [hp] at hacc; simp; omega
      · unfold CountersOrdered; simp [hp] at hcnt; simp; omega
      · intro hlt
        simp at hlt
        have := (hwi hlt).1
        rw [hact.2] at this; cases this
      · simp [hdr']
      · intro b hq
        have := hpd b hq
        simp [hact.1] at this
    · cases hs
  | d2hIssue ok =>
    simp only [step, d2hIssue] at hs
    split at hs
    · cases hs
    · rename_i hp
      have hp' : s.d2hPending = true := by
        cases hq : s.d2hPending
        · exact absurd hq hp
        · rfl
      have hst : s.started = true := hpc.2 (Or.inl hp')
      have hwle : s.woken ≤ s.d2hIssued := by
        refine Nat.le_of_not_lt fun hlt => ?_
        have := (hwi hlt).2
        rw [hp'] at this; cases this
      simp [hp'] at hacc hcnt
      split at hs
      · cases hs
        refine ⟨hl, ?_, hu, ?_, ⟨hpc.1, fun _ => hst⟩, hqs, ?_, hdr, hpd, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · intro hlt; simp at hlt; omega
      · cases hs
        dsimp only [finish]
        rw [endOp_eq (by simp; omega)]
        refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishOnceLocked_consistent false hl),
          ?_, hu, ?_, ⟨hpc.1, fun _ => hst⟩, hqs, ?_, ?_, ?_, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · intro hlt; simp at hlt; omega
        · intro hok _
          obtain ⟨h1, h2⟩ := hl.finishOnceLocked_false_statusOk (by simpa using hok)
          exact hdr h1 h2
        · exact fun b hq =>
            Lifecycle.endOpLocked_done_mono (Lifecycle.finishOnceLocked_done_mono false (hpd b hq))
  | d2hReady =>
    simp only [step, markD2hReady] at hs
    split at hs
    · cases hs
      exact ⟨hl, by unfold Accounted; simpa using hacc, hu,
        by unfold CountersOrdered; simp; omega, hpc, hqs, hwi, hdr, hpd, hpub⟩
    · cases hs
  | d2hEnd =>
    simp only [step, d2hEnd] at hs
    split at hs
    · cases hs
      rw [endOp_eq (by simp; omega)]
      refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hu, ?_, hpc, hqs, by simpa using hwi,
        ?_, ?_, hpub⟩
      · unfold Accounted; simp; omega
      · unfold CountersOrdered; simp; omega
      · simpa using hdr
      · exact fun b hq => Lifecycle.endOpLocked_done_mono (hpd b hq)
    · cases hs
  | wake ok =>
    simp only [step, wake] at hs
    split at hs
    · rename_i hg
      obtain ⟨hwk, hres⟩ := hg
      -- after the callback, any layer past the issued copies was resolved by
      -- `fail_unissued_layers`, on a draining send with no dispatch pending
      have hwi' : s.d2hIssued < s.woken + 1 → s.life.draining = true ∧ s.d2hPending = false := by
        intro hlt
        rcases hres with hwr | ⟨_, hdr', hp', _⟩
        · exact hwi (by omega)
        · exact ⟨hdr', hp'⟩
      split at hs
      · -- the copy failed (or was never issued): Finish(error), then the cleanup ends the op
        cases hs
        dsimp only [finish]
        rw [endOp_eq (by simp; omega)]
        refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishOnceLocked_consistent false hl),
          ?_, hu, ?_, hpc, hqs, ?_, ?_, ?_, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · intro hlt
          simp at hlt
          exact ⟨by simpa using hl.finishOnceLocked_draining false, (hwi' hlt).2⟩
        · intro hok _
          obtain ⟨h1, h2⟩ := hl.finishOnceLocked_false_statusOk (by simpa using hok)
          exact hdr h1 h2
        · exact fun b hq =>
            Lifecycle.endOpLocked_done_mono (Lifecycle.finishOnceLocked_done_mono false (hpd b hq))
      · rename_i hok
        have hwr : s.woken < s.d2hReady := by
          rcases hres with hwr | ⟨h0, _⟩
          · exact hwr
          · exact absurd h0 hok
        split at hs
        · -- draining: the layer is dropped, the op ends
          cases hs
          rw [endOp_eq (by simp; omega)]
          refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hu, ?_, hpc, hqs,
            by simpa using hwi', ?_, ?_, hpub⟩
          · unfold Accounted; simp; omega
          · unfold CountersOrdered; simp; omega
          · simpa using hdr
          · exact fun b hq => Lifecycle.endOpLocked_done_mono (hpd b hq)
        · -- scheduled on the push pool; the op is carried over
          cases hs
          refine ⟨hl, ?_, hu, ?_, hpc, hqs, by simpa using hwi', hdr, hpd, hpub⟩
          · unfold Accounted; simp; omega
          · unfold CountersOrdered; simp; omega
    · cases hs
  | h2hIssue =>
    simp only [step, h2hIssue] at hs
    split at hs
    · cases hs
    · split at hs
      · rename_i l hb
        cases hs
        have := Lifecycle.beginOp_inFlight hb
        have hact := Lifecycle.beginOp_active hb
        have hdr' := Lifecycle.beginOp_draining hb
        refine ⟨Lifecycle.beginOp_consistent hl hb, ?_, hu, ?_, hpc, hqs, ?_,
          ?_, ?_, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · intro hlt; simp at hlt; have := (hwi hlt).1; rw [hact.2] at this; cases this
        · simp [hdr']
        · intro b hq
          have := hpd b hq
          simp [hact.1] at this
      · cases hs
        rw [endOp_eq (by simp; omega)]
        refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hu, ?_, hpc, hqs, by simpa using hwi,
          ?_, ?_, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · simpa using hdr
        · exact fun b hq => Lifecycle.endOpLocked_done_mono (hpd b hq)
  | sendNext =>
    simp only [step, sendNext] at hs
    split at hs
    · cases hs
    · cases hs
      rcases trySendNext_cases { s with chaining := s.chaining - 1 }
        with heq | ⟨hlt, hdr', hd', heq⟩ <;> rw [heq]
      · rw [endOp_eq (by simp; omega)]
        refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hu, ?_, hpc, hqs, by simpa using hwi,
          ?_, ?_, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · simpa using hdr
        · exact fun b hq => Lifecycle.endOpLocked_done_mono (hpd b hq)
      · simp at hlt hdr' hd'
        rw [endOp_eq (by simp)]
        refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.consistent_incr hl hdr'), ?_, hu, ?_,
          hpc, fun _ => hqs (by omega), by simpa using hwi, ?_, ?_, hpub⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · simp [hdr']
        · intro b hq
          have := hpd b hq
          simp [hd'] at this
  | h2hDone ok =>
    simp only [step, h2hDone] at hs
    split at hs
    · -- a push is outstanding, so the session is not settled, so not published
      have hnp : s.published = none := by
        cases hq : s.published with
        | none => rfl
        | some b =>
          have := hl.done_idle (hpd b hq)
          omega
      split at hs
      · cases hs
        split
        · -- the last push: Finish() then the cleanup ends the op
          dsimp only [finish]
          rw [endOp_eq (by simp; omega)]
          refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishOnceLocked_consistent true hl),
            ?_, hu, ?_, hpc, hqs, ?_, ?_, ?_, ?_⟩
          · unfold Accounted; simp; omega
          · unfold CountersOrdered; simp; omega
          · intro hlt
            simp at hlt
            exact ⟨by simpa using hl.finishOnceLocked_draining true, (hwi hlt).2⟩
          · intro _ _; simp; omega
          · simp [hnp]
          · simp [hnp]
        · rw [endOp_eq (by simp; omega)]
          refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hu, ?_, hpc, hqs, by simpa using hwi,
            ?_, ?_, ?_⟩
          · unfold Accounted; simp; omega
          · unfold CountersOrdered; simp; omega
          · intro h1 h2
            simp at h1 h2
            have := hdr h1 h2
            simp; omega
          · simp [hnp]
          · simp [hnp]
      · cases hs
        dsimp only [finish]
        rw [endOp_eq (by simp; omega)]
        refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishOnceLocked_consistent false hl),
          ?_, hu, ?_, hpc, hqs, ?_, ?_, ?_, ?_⟩
        · unfold Accounted; simp; omega
        · unfold CountersOrdered; simp; omega
        · intro hlt
          simp at hlt
          exact ⟨by simpa using hl.finishOnceLocked_draining false, (hwi hlt).2⟩
        · intro hok _
          obtain ⟨h1, h2⟩ := hl.finishOnceLocked_false_statusOk (by simpa using hok)
          exact hdr h1 h2
        · simp [hnp]
        · simp [hnp]
    · cases hs
  | cancel =>
    simp only [step, cancel] at hs
    cases hs
    dsimp only [finish]
    refine ⟨Lifecycle.finishOnceLocked_consistent false hl,
      by unfold Accounted; simpa using hacc, hu,
      by unfold CountersOrdered; simpa using hcnt, hpc, hqs, ?_, ?_,
      fun b hq => Lifecycle.finishOnceLocked_done_mono false (hpd b hq), hpub⟩
    · intro hlt
      exact ⟨hl.finishOnceLocked_draining false, (hwi hlt).2⟩
    · intro hok _
      obtain ⟨h1, h2⟩ := hl.finishOnceLocked_false_statusOk (by simpa using hok)
      exact hdr h1 h2
  | publish =>
    simp only [step, publish] at hs
    split at hs
    · rename_i hd
      obtain ⟨hdone, _⟩ := hd
      cases hs
      refine ⟨hl, by unfold Accounted; simpa using hacc, hu,
        by unfold CountersOrdered; simpa using hcnt, hpc, hqs, hwi, hdr, fun _ _ => hdone, ?_⟩
      simp
      intro hst
      exact hdr hst (hl.done_draining hdone)
    · cases hs

theorem reachable_inv {n : Nat} {s : Send} (h : (sys n).Reachable s) : Inv s :=
  (sys n).reachable_induction (inv_init n) (fun _ _ _ hi hs => step_inv hi hs) h

/-- Main result: every reachable send session satisfies all the properties. -/
theorem reachable_safe {n : Nat} {s : Send} (h : (sys n).Reachable s) : Safe s :=
  inv_safe (reachable_inv h)


end Send

end TpuSyncVerify.Transfer.PrefillDecode

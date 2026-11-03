import TpuSyncVerify.Transfer.PrefillDecode.Receive.Model

/-! # `Recv`: inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.Receive`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ## Inductive invariant -/

/-- Where every unit of `in_flight_` comes from: an open push, the pull
handshake, a dispatch between its two locks, or a copy whose callback has not
run. -/
def Accounted (s : Recv) : Prop :=
  s.life.inFlight =
    s.pushes + (if s.pullPending then 1 else 0) + s.pending + (s.issued - s.retired)

structure Inv (s : Recv) : Prop where
  life : s.life.Consistent
  accounted : Accounted s
  completed_le : s.completed ≤ s.retired
  retired_le : s.retired ≤ s.ready
  ready_le : s.ready ≤ s.issued
  issued_le : s.issued + s.pending ≤ s.numLayers
  accounted_le : s.layersAccounted ≤ s.issued
  /-- While no error has been recorded, every callback that ran succeeded. -/
  ok_completed : s.life.statusOk = true → s.completed = s.retired
  /-- A session that started draining without an error had issued every layer. -/
  ok_draining : s.life.statusOk = true → s.life.draining = true → s.issued = s.numLayers
  /-- Only settled sessions are published. -/
  published_done : ∀ b, s.published = some b → s.life.done = true
  /-- Once published as done, every layer had completed (and still has). -/
  published_ok : s.published = some true → s.completed = s.numLayers

theorem inv_initPush (n : Nat) : Inv (initPush n) := by
  refine ⟨Lifecycle.consistent_init, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [initPush, Accounted]

theorem inv_initLoad (n : Nat) : Inv (initLoad n) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [initLoad, Accounted]
  constructor <;> simp

theorem inv_noOpLeak {s : Recv} (h : Inv s) : NoOpLeak s := by
  intro hif
  have hacc := h.accounted
  unfold Accounted at hacc
  have hr := h.retired_le
  have hy := h.ready_le
  by_cases hp : 0 < s.pushes
  · exact ⟨.pushEnd, by simp [drainEvents], by simp [step, pushEnd]; omega⟩
  · by_cases hq : s.pullPending = true
    · exact ⟨.pullReply true, by simp [drainEvents], by simp [step, pullReply, hq]⟩
    · by_cases hpend : 0 < s.pending
      · have hne : s.pending ≠ 0 := by omega
        refine ⟨.h2dIssue true, by simp [drainEvents], ?_⟩
        simp only [step, h2dIssue, hne, ↓reduceIte]
        split <;> rfl
      · by_cases hrd : s.ready < s.issued
        · exact ⟨.h2dReady, by simp [drainEvents], by simp [step, h2dReady, hrd]⟩
        · have hret : s.retired < s.ready := by
            cases hqp : s.pullPending <;> simp_all <;> omega
          refine ⟨.h2dDone true, by simp [drainEvents], ?_⟩
          simp only [step, h2dDone, hret, ↓reduceIte]
          split <;> rfl

theorem inv_safe {s : Recv} (h : Inv s) : Safe s := by
  have hnl := inv_noOpLeak h
  obtain ⟨hl, hacc, hc, hr, hy, hi, ha, hok, hdr, hpd, hpub⟩ := h
  unfold Accounted at hacc
  refine ⟨hl.done_idle, ?_, hl.staging, hl.prompt, ?_, hpub, ⟨hc, hr, hy, by omega, ha⟩, hnl, ?_⟩
  · intro hd
    have h0 := hl.done_idle hd
    omega
  · rintro ⟨hnet | hcomp, hall⟩
    · unfold networkCompleted at hnet; unfold allH2dDone at hall; omega
    · unfold allH2dDone at hall; omega
  · intro hp
    have hd : s.life.done = true := by
      cases hpb : s.published with
      | none => exact absurd hpb hp
      | some b => exact hpd b hpb
    have h0 := hl.done_idle hd
    omega

/-- Each event preserves the invariant. One case per event; the `Lifecycle`
lemmas discharge the settle protocol and `omega` the arithmetic. -/
theorem step_inv {s s' : Recv} {e : Ev} (h : Inv s) (hs : step s e = some s') : Inv s' := by
  obtain ⟨hl, hacc, hc, hr, hy, hi, ha, hok, hdr, hpd, hpub⟩ := h
  unfold Accounted at hacc
  cases e with
  | pushBegin =>
    simp only [step, pushBegin, Option.map_eq_some_iff] at hs
    obtain ⟨l, hb, rfl⟩ := hs
    have := Lifecycle.beginOp_inFlight hb
    have hact := Lifecycle.beginOp_active hb
    have hok' := Lifecycle.beginOp_statusOk hb
    have hdr' := Lifecycle.beginOp_draining hb
    refine ⟨Lifecycle.beginOp_consistent hl hb, ?_, hc, hr, hy, hi, ha, ?_, ?_, ?_, hpub⟩
    · unfold Accounted; simp; omega
    · simpa [hok'] using hok
    · simp [hdr']
    · intro b hp
      have := hpd b hp
      simp [hact.1] at this
  | pushEnd =>
    simp only [step, pushEnd] at hs
    split at hs
    · cases hs
    · cases hs
      refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hc, hr, hy, hi, ha, ?_, ?_, ?_, hpub⟩
      · unfold Accounted; simp; omega
      · simpa using hok
      · simpa using hdr
      · exact fun b hp => Lifecycle.endOpLocked_done_mono (hpd b hp)
  | pullReply ok =>
    simp only [step, pullReply] at hs
    split at hs
    · cases hs
    · rename_i hp
      cases hs
      have hp' : s.pullPending = true := by
        cases hq : s.pullPending
        · exact absurd hq hp
        · rfl
      refine ⟨?_, ?_, hc, hr, hy, hi, ha, ?_, ?_, ?_, hpub⟩
      · apply Lifecycle.endOpLocked_consistent
        split
        · exact hl
        · exact Lifecycle.finishLocked_consistent _ hl
      · unfold Accounted
        cases ok <;> simp [hp'] at hacc ⊢ <;> omega
      · cases ok <;> simp <;> exact hok
      · cases ok <;> simp
        · exact hdr
      · intro b hp
        have hd := hpd b hp
        apply Lifecycle.endOpLocked_done_mono
        split
        · exact hd
        · exact Lifecycle.finishLocked_done_mono _ hd
  | h2dBegin =>
    simp only [step, h2dBegin] at hs
    split at hs
    · simp only [Option.map_eq_some_iff] at hs
      obtain ⟨l, hb, rfl⟩ := hs
      have := Lifecycle.beginOp_inFlight hb
      have hact := Lifecycle.beginOp_active hb
      have hok' := Lifecycle.beginOp_statusOk hb
      have hdr' := Lifecycle.beginOp_draining hb
      refine ⟨Lifecycle.beginOp_consistent hl hb, ?_, hc, hr, hy, by simp; omega, ha, ?_, ?_, ?_,
        hpub⟩
      · unfold Accounted; simp; omega
      · simpa [hok'] using hok
      · simp [hdr']
      · intro b hp
        have := hpd b hp
        simp [hact.1] at this
    · cases hs
  | h2dIssue ok =>
    simp only [step, h2dIssue] at hs
    split at hs
    · cases hs
    · rename_i hp
      split at hs
      · cases hs
        refine ⟨Lifecycle.endOpLocked_consistent hl, ?_, hc, hr, hy, by simp; omega, ha, ?_, ?_, ?_,
          hpub⟩
        · unfold Accounted; simp; omega
        · simpa using hok
        · simpa using hdr
        · exact fun b hp => Lifecycle.endOpLocked_done_mono (hpd b hp)
      · rename_i hact
        split at hs
        · cases hs
          refine ⟨hl, ?_, hc, hr, by simp; omega, by simp; omega, by simp; omega, hok, ?_, hpd, hpub⟩
          · unfold Accounted; simp; omega
          · intro _ hd
            exact absurd (Or.inr hd) hact
        · cases hs
          refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishLocked_consistent _ hl),
            ?_, hc, hr, hy, by simp; omega, ha, ?_, ?_, ?_, hpub⟩
          · unfold Accounted; simp; omega
          · simp
          · simp
          · exact fun b hp =>
              Lifecycle.endOpLocked_done_mono (Lifecycle.finishLocked_done_mono _ (hpd b hp))
  | h2dReady =>
    simp only [step, h2dReady] at hs
    split at hs
    · cases hs
      exact ⟨hl, by unfold Accounted; simpa using hacc, hc, by simp; omega, by simp; omega,
        hi, ha, hok, hdr, hpd, hpub⟩
    · cases hs
  | h2dDone ok =>
    simp only [step, h2dDone] at hs
    split at hs
    · rename_i hlt
      split at hs
      · rename_i hd
        -- the DFATAL branch: unreachable, since done forces retired = issued
        have h0 := hl.done_idle hd
        omega
      · rename_i hnd
        split at hs
        · cases hs
          refine ⟨?_, ?_, by simp; omega, by simp; omega, hy, hi, ha, ?_, ?_, ?_, ?_⟩
          · apply Lifecycle.endOpLocked_consistent
            split
            · exact Lifecycle.finishLocked_consistent _ hl
            · exact hl
          · unfold Accounted
            split <;> (try simp) <;> omega
          · split <;> simp <;> intro h1 <;> have := hok h1 <;> omega
          · split
            · rename_i hfin
              obtain ⟨hfin, _⟩ := hfin
              intro _ _
              simp
              omega
            · simp
              intro h1 h2
              exact hdr h1 h2
          · intro b hp
            exact absurd (hpd b hp) hnd
          · intro hp
            exact absurd (hpd true hp) hnd
        · cases hs
          refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishLocked_consistent _ hl),
            ?_, by simp; omega, by simp; omega, hy, hi, ha, ?_, ?_, ?_, hpub⟩
          · unfold Accounted; simp; omega
          · simp
          · simp
          · intro b hp
            exact absurd (hpd b hp) hnd
    · cases hs
  | netAccount =>
    simp only [step, netAccount] at hs
    split at hs
    · cases hs
    · rename_i hact
      split at hs
      · split at hs
        · rename_i hfin
          cases hs
          refine ⟨Lifecycle.finishLocked_consistent _ hl, ?_, hc, hr, hy, hi, by simp; omega,
            ?_, ?_, ?_, hpub⟩
          · unfold Accounted; simpa using hacc
          · simpa using hok
          · intro _ _
            simp
            omega
          · exact fun b hp => Lifecycle.finishLocked_done_mono _ (hpd b hp)
        · cases hs
          refine ⟨hl, ?_, hc, hr, hy, hi, by simp; omega, hok, ?_, hpd, hpub⟩
          · unfold Accounted; simpa using hacc
          · simpa using hdr
      · cases hs
  | pollReady =>
    simp only [step, pollReady] at hs
    split at hs
    · rename_i hrd
      obtain ⟨hndr, hnet | hcomp, hall⟩ := hrd
      · cases hs
        refine ⟨Lifecycle.finishLocked_consistent _ hl, ?_, hc, hr, hy, hi, ha, ?_, ?_, ?_, hpub⟩
        · unfold Accounted; simpa using hacc
        · simpa using hok
        · unfold networkCompleted at hnet; unfold allH2dDone at hall
          intro _ _; simp; omega
        · exact fun b hp => Lifecycle.finishLocked_done_mono _ (hpd b hp)
      · cases hs
        refine ⟨Lifecycle.finishLocked_consistent _ hl, ?_, hc, hr, hy, hi, ha, ?_, ?_, ?_, hpub⟩
        · unfold Accounted; simpa using hacc
        · simpa using hok
        · intro _ _; simp; omega
        · exact fun b hp => Lifecycle.finishLocked_done_mono _ (hpd b hp)
    · cases hs
  | cancel =>
    simp only [step, cancel] at hs
    cases hs
    refine ⟨Lifecycle.finishLocked_consistent _ hl, ?_, hc, hr, hy, hi, ha, ?_, ?_, ?_, hpub⟩
    · unfold Accounted; simpa using hacc
    · simp
    · simp
    · exact fun b hp => Lifecycle.finishLocked_done_mono _ (hpd b hp)
  | publish =>
    simp only [step, publish] at hs
    split at hs
    · rename_i hd
      obtain ⟨hdone, _⟩ := hd
      cases hs
      refine ⟨hl, by unfold Accounted; simpa using hacc, hc, hr, hy, hi, ha, hok, hdr,
        fun _ _ => hdone, ?_⟩
      simp
      intro hst
      have h0 := hl.done_idle hdone
      have h1 := hok hst
      have h2 := hdr hst (hl.done_draining hdone)
      omega
    · cases hs

theorem reachable_inv_push {n : Nat} {s : Recv} (h : (sysPush n).Reachable s) : Inv s :=
  (sysPush n).reachable_induction (inv_initPush n) (fun _ _ _ hi hs => step_inv hi hs) h

theorem reachable_inv_load {n : Nat} {s : Recv} (h : (sysLoad n).Reachable s) : Inv s :=
  (sysLoad n).reachable_induction (inv_initLoad n) (fun _ _ _ hi hs => step_inv hi hs) h

/-- Main result: every reachable receive session, on either creation path,
satisfies all the properties. -/
theorem reachable_safe {n : Nat} {s : Recv}
    (h : (sysPush n).Reachable s ∨ (sysLoad n).Reachable s) : Safe s :=
  inv_safe (h.elim reachable_inv_push reachable_inv_load)


end Recv

end TpuSyncVerify.Transfer.PrefillDecode

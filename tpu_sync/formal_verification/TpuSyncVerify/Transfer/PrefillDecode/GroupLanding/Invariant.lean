import TpuSyncVerify.Transfer.PrefillDecode.GroupLanding.Model

/-! # `GroupLanding`: inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.GroupLanding`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.GroupLanding

open TpuSyncVerify TpuSyncVerify.Transfer

namespace S

structure Inv (s : S) : Prop where
  life : s.life.Consistent
  /-- `in_flight_` is the streams' leases and the layer's outstanding copies. -/
  accounted : s.life.inFlight = s.streams + s.countedCopies + s.copyingCopies
  /-- Every block is in exactly one place. -/
  conserved : s.fresh + s.half + s.untaken + s.pending + s.counted + s.copying + s.copied +
    s.lost = s.numBlocks
  /-- Every copy has at least one block. -/
  counted_sizes : s.countedCopies ≤ s.counted ∧ (s.countedCopies = 0 → s.counted = 0)
  copying_sizes : s.copyingCopies ≤ s.copying ∧ (s.copyingCopies = 0 → s.copying = 0)
  pending_half : s.pendingHalf = 0
  no_bad_read : s.badRead = false
  eager : s.pending < s.groupBlocks ∨ (s.groupBlocks = 0 ∧ s.pending = 0)
  /-- A landed layer has nothing left to land or to take. -/
  landed_all : s.landed = true →
    s.executed = true ∧ s.fresh = 0 ∧ s.half = 0 ∧ s.pending = 0 ∧ s.untaken = 0
  completed_le : s.completed ≤ 1
  /-- A completed layer had landed, and none of its copies is outstanding. -/
  completed_landed : 0 < s.completed →
    s.landed = true ∧ s.countedCopies = 0 ∧ s.copyingCopies = 0
  /-- No copy failed or was dropped while the status is OK. -/
  ok_lost : s.life.statusOk = true → s.lost = 0
  /-- The only OK finish is the layer's completion. -/
  ok_draining : s.life.statusOk = true → s.life.draining = true → 0 < s.completed
  no_late : s.late = false

theorem inv_init {n g m : Nat} : Inv (init n g m) := by
  refine ⟨Lifecycle.consistent_init, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [init]
  rcases Nat.eq_zero_or_pos g with h | h <;> simp [h]

set_option hygiene false in
/-- Closes the per-field goals of an `Inv` proof once the event's effect on the
state has been unfolded. -/
macro "gl_auto" : tactic => `(tactic| (
  all_goals first
    | exact hc'
    | omega
    | (simp_all; done)
    | (simp_all; omega)))

theorem life_endOp {l : Lifecycle} (hl : l.Consistent) :
    (l.endOpLocked).Consistent ∧ (l.endOpLocked).inFlight = l.inFlight - 1 ∧
    (l.endOpLocked).draining = l.draining ∧ (l.endOpLocked).statusOk = l.statusOk :=
  ⟨Lifecycle.endOpLocked_consistent hl, by simp, by simp, by simp⟩

theorem life_fail_endOp {l : Lifecycle} (hl : l.Consistent) :
    ((l.finishLocked false).endOpLocked).Consistent ∧
    ((l.finishLocked false).endOpLocked).inFlight = l.inFlight - 1 ∧
    ((l.finishLocked false).endOpLocked).draining = true ∧
    ((l.finishLocked false).endOpLocked).statusOk = false :=
  ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishLocked_consistent false hl), by simp,
    by simp, by simp⟩

theorem life_incr_endOp {l : Lifecycle} (hl : l.Consistent) (hd : l.draining = false) :
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).Consistent ∧
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).inFlight = l.inFlight ∧
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).draining = l.draining ∧
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).statusOk = l.statusOk :=
  ⟨Lifecycle.endOpLocked_consistent (Lifecycle.consistent_incr hl hd), by simp, by simp, by simp⟩

theorem life_ok_endOp {l : Lifecycle} (hl : l.Consistent) :
    ((l.finishLocked true).endOpLocked).Consistent ∧
    ((l.finishLocked true).endOpLocked).inFlight = l.inFlight - 1 ∧
    ((l.finishLocked true).endOpLocked).draining = true ∧
    ((l.finishLocked true).endOpLocked).statusOk = l.statusOk :=
  ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishLocked_consistent true hl), by simp,
    by simp, by simp⟩

theorem streamBegin_inv {s s' : S} (h : Inv s) (hs : s.streamBegin = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold streamBegin at hs
  split at hs
  · cases hb : s.life.beginOp with
    | none => simp [hb] at hs
    | some l =>
      simp only [hb, Option.map_some, Option.some.injEq] at hs
      subst hs
      have hc' := Lifecycle.beginOp_consistent hl hb
      have hif := Lifecycle.beginOp_inFlight hb
      have hst := Lifecycle.beginOp_statusOk hb
      have hdr := Lifecycle.beginOp_draining hb
      constructor <;> simp only [hif, hst, hdr]
      gl_auto
  · cases hs

theorem landFull_inv {s s' : S} {k : Nat} (h : Inv s) (hs : s.landFull k = some s') :
    Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold landFull at hs
  dsimp only at hs
  split at hs
  · rename_i hg
    obtain ⟨hst, hk, hkf⟩ := hg
    have hnl : s.landed = false := by
      cases hl' : s.landed
      · rfl
      · have := (hla hl').2.1; omega
    split at hs
    · -- the report is ignored: the blocks land untaken
      cases hs
      have hc' := hl
      constructor <;> simp only [hnl] <;> gl_auto
    · rename_i hng
      have hg0 : s.groupBlocks ≠ 0 := fun h0 => hng (Or.inl h0)
      have hdr0 : s.life.draining = false := by
        cases hd : s.life.draining
        · rfl
        · exact absurd (Or.inr (Or.inr (Or.inl hd))) hng
      split at hs
      · -- a group's worth waits: its copy is counted
        cases hs
        have hc' := Lifecycle.consistent_incr hl hdr0
        constructor <;> simp only [hnl, hph, hbr, Bool.false_or, decide_eq_false_iff_not,
          Nat.lt_irrefl, not_false_eq_true] <;> gl_auto
      · cases hs
        have hc' := hl
        constructor <;> simp only [hnl] <;> gl_auto
  · cases hs

theorem landHalf_inv {s s' : S} (h : Inv s) (hs : s.landHalf = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold landHalf at hs
  split at hs
  · cases hs
    rename_i hg
    have hnl : s.landed = false := by
      cases hl' : s.landed
      · rfl
      · have := (hla hl').2.1; omega
    have hc' := hl
    constructor <;> simp only [hnl] <;> gl_auto
  · cases hs

theorem landOther_inv {s s' : S} (h : Inv s) (hs : s.landOther = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold landOther at hs
  split at hs
  · cases hs
    rename_i hg
    have hnl : s.landed = false := by
      cases hl' : s.landed
      · rfl
      · have := (hla hl').2.2.1; omega
    have hc' := hl
    constructor <;> simp only [hnl] <;> gl_auto
  · cases hs

theorem streamEnd_inv {s s' : S} (h : Inv s) (hs : s.streamEnd = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold streamEnd at hs
  split at hs
  · cases hs
    obtain ⟨hc', hif', hdr', hst'⟩ := life_endOp hl
    constructor <;> simp only [hif', hdr', hst'] <;> gl_auto
  · cases hs

theorem streamLast_inv {s s' : S} (h : Inv s) (hs : s.streamLast = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold streamLast at hs
  dsimp only at hs
  split at hs
  · rename_i hg
    obtain ⟨hst, hex, hf, hh⟩ := hg
    have hnl : s.landed = false := by
      cases hl' : s.landed
      · rfl
      · have := (hla hl').1; rw [hex] at this; cases this
    have hc0 : s.completed = 0 := by
      cases Nat.eq_zero_or_pos s.completed with
      | inl h0 => exact h0
      | inr hpos => have := (hcl hpos).1; rw [hnl] at this; cases this
    unfold executeLayer at hs
    dsimp only at hs
    split at hs
    · -- a draining receive returns from `ExecuteLayerH2d`
      cases hs
      obtain ⟨hc', hif', hdr', hst'⟩ := life_endOp hl
      constructor <;> simp only [hif', hdr', hst', hnl] <;> gl_auto
    · rename_i hnd
      have hdr0 : s.life.draining = false := by
        cases hd : s.life.draining
        · rfl
        · exact absurd (Or.inr hd) hnd
      split at hs
      · rename_i hrest
        split at hs
        · -- nothing left to copy and no copy outstanding: the layer completes
          cases hs
          rename_i hout
          obtain ⟨hc', hif', hdr', hst'⟩ := life_ok_endOp hl
          constructor <;> simp only [completeLayer, hdr0, if_true, hif', hdr', hst', hc0] <;>
            gl_auto
        · cases hs
          obtain ⟨hc', hif', hdr', hst'⟩ := life_endOp hl
          constructor <;> simp only [hif', hdr', hst', hc0] <;> gl_auto
      · -- the rest forms the layer's last copy
        cases hs
        rename_i hrest
        obtain ⟨hc', hif', hdr', hst'⟩ := life_incr_endOp hl hdr0
        constructor <;> simp only [hif', hdr', hst', hc0] <;> gl_auto
  · cases hs

theorem dispatch_inv {s s' : S} {k : Nat} {ok : Bool} (h : Inv s)
    (hs : s.dispatch k ok = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold dispatch at hs
  dsimp only at hs
  split at hs
  · rename_i hg
    obtain ⟨hcc, hk, hk1, hk2⟩ := hg
    have hc0 : s.completed = 0 := by
      cases Nat.eq_zero_or_pos s.completed with
      | inl h0 => exact h0
      | inr hpos => have := (hcl hpos).2.1; omega
    split at hs
    · -- the receive started draining: the copy is dropped
      cases hs
      rename_i hdd
      have hdr : s.life.draining = true := by
        rcases hdd with hd | hd
        · exact hl.done_draining hd
        · exact hd
      have hnok : s.life.statusOk = false := by
        cases hso : s.life.statusOk
        · rfl
        · have := hokd hso hdr; omega
      obtain ⟨hc', hif', hdr', hst'⟩ := life_endOp hl
      constructor <;> simp only [hif', hdr', hst', hnok] <;> gl_auto
    · split at hs
      · cases hs
        have hc' := hl
        constructor <;> simp only [hc0, hlate, Nat.lt_irrefl, decide_false, Bool.or_false] <;>
          gl_auto
      · cases hs
        obtain ⟨hc', hif', hdr', hst'⟩ := life_fail_endOp hl
        constructor <;> simp only [hif', hdr', hst'] <;> gl_auto
  · cases hs

theorem copyDone_inv {s s' : S} {k : Nat} {ok : Bool} (h : Inv s)
    (hs : s.copyDone k ok = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold copyDone at hs
  dsimp only at hs
  split at hs
  · rename_i hg
    obtain ⟨hcc, hk, hk1, hk2⟩ := hg
    have hc0 : s.completed = 0 := by
      cases Nat.eq_zero_or_pos s.completed with
      | inl h0 => exact h0
      | inr hpos => have := (hcl hpos).2.2; omega
    split at hs
    · split at hs
      · -- the layer's last outstanding copy: it completes
        cases hs
        rename_i hok hdone
        unfold completeLayer
        split
        · rename_i hdr0
          obtain ⟨hc', hif', hdr', hst'⟩ := life_ok_endOp hl
          constructor <;> simp only [hif', hdr', hst', hc0] <;> gl_auto
        · rename_i hdr0
          obtain ⟨hc', hif', hdr', hst'⟩ := life_endOp hl
          constructor <;> simp only [hif', hdr', hst', hc0] <;> gl_auto
      · cases hs
        obtain ⟨hc', hif', hdr', hst'⟩ := life_endOp hl
        constructor <;> simp only [hif', hdr', hst', hc0] <;> gl_auto
    · cases hs
      obtain ⟨hc', hif', hdr', hst'⟩ := life_fail_endOp hl
      constructor <;> simp only [hif', hdr', hst'] <;> gl_auto
  · cases hs

theorem cancel_inv {s s' : S} (h : Inv s) (hs : s.cancel = some s') : Inv s' := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  unfold cancel at hs
  cases hs
  have hc' := Lifecycle.finishLocked_consistent false hl
  constructor <;> simp only [Lifecycle.finishLocked_inFlight, Lifecycle.finishLocked_draining,
    Lifecycle.finishLocked_statusOk, Bool.false_and] <;> gl_auto

theorem step_inv {s s' : S} {e : Ev} (h : Inv s) (hs : s.step e = some s') : Inv s' := by
  cases e with
  | streamBegin => exact streamBegin_inv h hs
  | landFull k => exact landFull_inv h hs
  | landHalf => exact landHalf_inv h hs
  | landOther => exact landOther_inv h hs
  | streamEnd => exact streamEnd_inv h hs
  | streamLast => exact streamLast_inv h hs
  | dispatch k ok => exact dispatch_inv h hs
  | copyDone k ok => exact copyDone_inv h hs
  | cancel => exact cancel_inv h hs

theorem reachable_inv {n g m : Nat} {s : S} (hr : (sys n g m).Reachable s) : Inv s :=
  (sys n g m).reachable_induction inv_init (fun _ _ _ h hs => step_inv h hs) hr

theorem inv_safe {s : S} (h : Inv s) : Safe s := by
  obtain ⟨hl, hacc, hcons, hcs, hcps, hph, hbr, heag, hla, hcle, hcl, hol, hokd, hlate⟩ := h
  refine ⟨hcle, ?_, hlate, hbr, ?_, ?_⟩
  · intro hpos
    obtain ⟨hld, hcc, hcp⟩ := hcl hpos
    obtain ⟨_, hf, hh, hp, hu⟩ := hla hld
    refine ⟨hcc, hcp, fun hok => ?_⟩
    have := hol hok
    have := hcs.2 hcc
    have := hcps.2 hcp
    omega
  · intro hd
    have := hl.done_idle hd
    omega
  · rcases heag with h | ⟨h, _⟩
    · exact Or.inl h
    · exact Or.inr h

/-- **The grouped layer is published whole.** Every reachable state is `Safe`:
for any number of blocks and any group size, a layer completes once, only
after every one of its copies — the group copies issued as blocks landed and
the last copy of the rest — has finished; on an OK receive every block is then
on the device; no copy reaches the device after that; and every copy read
blocks whose shards had all landed. -/
theorem reachable_safe {n g m : Nat} {s : S} (hr : (sys n g m).Reachable s) : Safe s :=
  inv_safe (reachable_inv hr)

end S

end TpuSyncVerify.Transfer.PrefillDecode.GroupLanding

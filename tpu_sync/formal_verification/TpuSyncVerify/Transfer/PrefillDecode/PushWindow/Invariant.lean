import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Model

/-! # `PushWindow`: inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.PushWindow`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.PushWindow

open TpuSyncVerify TpuSyncVerify.Transfer

namespace S

/-- Where the push chain's baton is: armed on a unit's future, queued as an
`AdmitPush` task, about to `IssuePush`, chaining into the next unit, or parked. -/
def baton (s : S) : Nat :=
  (s.queued - s.woken) + s.admitting + s.issuing + s.chaining + (if s.parked then 1 else 0)

structure Inv (s : S) : Prop where
  life : s.life.Consistent
  /-- Where every unit of `in_flight_` comes from: the baton (armed, queued,
  issuing or chaining; a parked unit holds none) and the pushes in flight. -/
  accounted : s.life.inFlight =
    (s.queued - s.woken) + s.admitting + s.issuing + s.chaining + (s.h2hIssued - s.h2hRetired)
  no_underflow : s.underflow = false
  /-- Single-baton invariant: one unit at a time moves from its future to its
  push, so units are pushed in order and at most one is parked. -/
  baton_le : s.baton ≤ 1
  woken_le : s.woken ≤ s.queued
  queued_le : s.queued ≤ s.numUnits
  issued_le : s.h2hIssued + s.admitting + s.issuing + (if s.parked then 1 else 0) ≤ s.woken
  /-- Only a draining send drops a unit without pushing it. -/
  no_drop : s.life.draining = false →
    s.h2hIssued + s.admitting + s.issuing + (if s.parked then 1 else 0) = s.woken
  retired_le : s.h2hRetired ≤ s.h2hIssued
  ok_le : s.h2hOk ≤ s.h2hRetired
  /-- A window slot is held by a unit about to be issued or by a push in
  flight. -/
  pif_eq : s.pif = s.issuing + (s.h2hIssued - s.h2hRetired)
  window : 0 < s.window → s.pif ≤ s.window
  parked_waker : s.parked = true → s.life.draining = false → s.h2hRetired < s.h2hIssued
  /-- A send that started draining without an error pushed every unit. -/
  ok_draining : s.life.statusOk = true → s.life.draining = true → s.h2hOk = s.numUnits
  /-- A failed push finishes the send. -/
  failed_draining : s.h2hOk < s.h2hRetired → s.life.draining = true
  /-- The last OK push finishes the send. -/
  all_ok_draining : s.h2hOk = s.numUnits → s.life.draining = true
  /-- While not draining, the baton is somewhere, or the chain has issued
  every unit and some push is still in flight. -/
  live : s.life.draining = false →
    s.baton = 1 ∨ (s.h2hIssued = s.numUnits ∧ s.h2hRetired < s.h2hIssued)
  published_done : ∀ b, s.published = some b → s.life.done = true
  published_ok : s.published = some true → s.h2hOk = s.numUnits

theorem inv_init {n w : Nat} (hn : 0 < n) : Inv (init n w) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [init, baton] <;> try omega
  constructor <;> simp

/-! ### Helpers -/

theorem life_finish_endOp {l : Lifecycle} (ok : Bool) (hl : l.Consistent) :
    ((l.finishOnceLocked ok).endOpLocked).Consistent ∧
    ((l.finishOnceLocked ok).endOpLocked).inFlight = l.inFlight - 1 ∧
    ((l.finishOnceLocked ok).endOpLocked).draining = true ∧
    ((l.finishOnceLocked ok).endOpLocked).statusOk =
      (if l.draining = true ∨ l.done = true then l.statusOk else ok) ∧
    (l.done = true → ((l.finishOnceLocked ok).endOpLocked).done = true) := by
  refine ⟨Lifecycle.endOpLocked_consistent (Lifecycle.finishOnceLocked_consistent ok hl),
    ?_, ?_, ?_, ?_⟩
  · simp
  · cases hd : l.done
    · simp [hd]
    · simp [hl.done_draining hd]
  · simp
  · intro hd
    exact Lifecycle.endOpLocked_done_mono (Lifecycle.finishOnceLocked_done_mono ok hd)

theorem life_endOp {l : Lifecycle} (hl : l.Consistent) :
    (l.endOpLocked).Consistent ∧ (l.endOpLocked).inFlight = l.inFlight - 1 ∧
    (l.endOpLocked).draining = l.draining ∧ (l.endOpLocked).statusOk = l.statusOk ∧
    (l.done = true → (l.endOpLocked).done = true) :=
  ⟨Lifecycle.endOpLocked_consistent hl, by simp, by simp, by simp,
    fun hd => Lifecycle.endOpLocked_done_mono hd⟩

/-- A status that stays OK across a first `Finish` was already decided OK. -/
theorem ok_of_finish {l : Lifecycle} (hl : l.Consistent) {ok : Bool}
    (h : (if l.draining = true ∨ l.done = true then l.statusOk else ok) = true)
    (hok : ok = false) : l.statusOk = true ∧ l.draining = true := by
  split at h
  · rename_i hd
    refine ⟨h, ?_⟩
    rcases hd with hd | hd
    · exact hd
    · exact hl.done_draining hd
  · simp [hok] at h

set_option hygiene false in
/-- Closes the per-field goals of an `Inv` proof once the event's effect on the
state has been unfolded; `hc'`/`hdn'`/`hpd` name the new lifecycle's
consistency, its `done` monotonicity, and the old `published_done`. -/
macro "pw_auto" : tactic => `(tactic| (
  all_goals first
    | exact hc'
    | omega
    | (intro b hb'; exact hdn' (hpd b hb'))
    | (simp_all; done)
    | (simp_all; omega)))

theorem life_incr {l : Lifecycle} (hl : l.Consistent) (hd : l.draining = false) :
    ({ l with inFlight := l.inFlight + 1 } : Lifecycle).Consistent :=
  Lifecycle.consistent_incr hl hd

theorem life_incr_endOp {l : Lifecycle} (hl : l.Consistent) (hd : l.draining = false) :
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).Consistent ∧
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).inFlight = l.inFlight ∧
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).draining = l.draining ∧
    (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).statusOk = l.statusOk ∧
    (l.done = true →
      (({ l with inFlight := l.inFlight + 1 } : Lifecycle).endOpLocked).done = true) := by
  refine ⟨Lifecycle.endOpLocked_consistent (life_incr hl hd), by simp, by simp, by simp, ?_⟩
  intro hdn
  exact absurd hdn (by simp [hl.not_done hd])

theorem wake_inv {s s' : S} {ok : Bool} (h : Inv s) (hs : s.wake ok = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold wake at hs
  dsimp only at hs
  split at hs
  · rename_i hlt
    unfold baton at hb hlive
    have hpos : 0 < s.life.inFlight := by omega
    have ha : s.admitting = 0 := by omega
    have his : s.issuing = 0 := by omega
    have hc : s.chaining = 0 := by omega
    have hp : s.parked = false := by
      cases hpk : s.parked
      · rfl
      · simp [hpk] at hb; omega
    simp only [hp] at hil hnd hb hlive hpw
    have hne : s.life.inFlight ≠ 0 := by omega
    split at hs
    · -- a failed copy finishes the send
      cases hs
      obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_finish_endOp false hl
      have hne' : (s.life.finishOnceLocked false).inFlight ≠ 0 := by simp; omega
      constructor <;> simp only [endOp, finish, hne', if_false, baton, hp, ha, his, hc, hdr',
        hif', Bool.false_eq_true, false_implies, implies_true, decide_false, Bool.or_false, hu,
        Nat.add_zero, Nat.zero_add]
      all_goals first
        | (intro hok _; obtain ⟨hso, hsd⟩ := ok_of_finish hl (hst' ▸ hok) rfl; exact hokd hso hsd)
        | skip
      pw_auto
    · split at hs
      · -- a draining send pushes nothing
        cases hs
        rename_i _ hdr
        obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_endOp hl
        constructor <;> simp only [endOp, hne, if_false, baton, hp, ha, his, hc, hdr', hif', hst',
          hdr, Bool.false_eq_true, false_implies, implies_true, decide_false, Bool.or_false, hu,
          Nat.add_zero, Nat.zero_add]
        pw_auto
      · -- the op is carried into `AdmitPush`
        cases hs
        rename_i _ hdr
        have hdr0 : s.life.draining = false := by simpa using hdr
        have hc' := hl
        have hdn' : s.life.done = true → s.life.done = true := id
        constructor <;> simp only [baton, hp, ha, his, hc, hdr0, Bool.false_eq_true, false_implies,
          implies_true, Nat.add_zero, Nat.zero_add, hu]
        pw_auto
  · cases hs

theorem life_finish {l : Lifecycle} (ok : Bool) (hl : l.Consistent) :
    (l.finishOnceLocked ok).Consistent ∧ (l.finishOnceLocked ok).inFlight = l.inFlight ∧
    (l.finishOnceLocked ok).draining = true ∧
    (l.finishOnceLocked ok).statusOk =
      (if l.draining = true ∨ l.done = true then l.statusOk else ok) ∧
    (l.done = true → (l.finishOnceLocked ok).done = true) :=
  ⟨Lifecycle.finishOnceLocked_consistent ok hl, by simp, hl.finishOnceLocked_draining ok, by simp,
    fun hd => Lifecycle.finishOnceLocked_done_mono ok hd⟩

theorem admit_inv {s s' : S} (h : Inv s) (hs : s.admit = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold admit at hs
  dsimp only at hs
  split at hs
  · cases hs
  · rename_i hne0
    unfold baton at hb hlive
    have ha : s.admitting = 1 := by omega
    have hq : s.queued - s.woken = 0 := by omega
    have his : s.issuing = 0 := by omega
    have hc : s.chaining = 0 := by omega
    have hp : s.parked = false := by
      cases hpk : s.parked
      · rfl
      · simp [hpk] at hb; omega
    simp only [hp] at hil hnd hb hlive hpw
    have hne : s.life.inFlight ≠ 0 := by omega
    split at hs
    · -- a draining send drops the unit
      cases hs
      rename_i hdr
      obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_endOp hl
      constructor <;> simp only [endOp, hne, if_false, baton, hp, ha, his, hc, hdr', hif', hst',
        hdr, Bool.false_eq_true, false_implies, implies_true, decide_false, Bool.or_false, hu,
        Nat.add_zero, Nat.zero_add]
      pw_auto
    · split at hs
      · -- a full window parks the unit
        cases hs
        rename_i hdr hfull
        have hdr0 : s.life.draining = false := by simpa using hdr
        obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_endOp hl
        constructor <;> simp only [endOp, hne, if_false, baton, ha, his, hc, hdr', hif', hst',
          hdr0, Bool.false_eq_true, false_implies, implies_true, decide_false, Bool.or_false, hu,
          Nat.add_zero, Nat.zero_add]
        pw_auto
      · -- the unit takes a window slot and goes on to `IssuePush`
        cases hs
        rename_i hdr hfull
        have hdr0 : s.life.draining = false := by simpa using hdr
        have hc' := hl
        have hdn' : s.life.done = true → s.life.done = true := id
        constructor <;> simp only [baton, hp, ha, his, hc, hdr0, Bool.false_eq_true,
          false_implies, implies_true, Nat.add_zero, Nat.zero_add, hu]
        pw_auto

theorem issue_inv {s s' : S} (h : Inv s) (hs : s.issue = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold issue at hs
  dsimp only at hs
  split at hs
  · cases hs
  · rename_i hne0
    unfold baton at hb hlive
    have his : s.issuing = 1 := by omega
    have hq : s.queued - s.woken = 0 := by omega
    have ha : s.admitting = 0 := by omega
    have hc : s.chaining = 0 := by omega
    have hp : s.parked = false := by
      cases hpk : s.parked
      · rfl
      · simp [hpk] at hb; omega
    simp only [hp] at hil hnd hb hlive hpw
    split at hs
    · -- the send started draining since admission: the slot is returned
      cases hs
      rename_i hdr
      have hc' := hl
      have hdn' : s.life.done = true → s.life.done = true := id
      constructor <;> simp only [baton, hp, ha, his, hc, hdr, Bool.false_eq_true, false_implies,
        implies_true, Nat.add_zero, Nat.zero_add, hu]
      pw_auto
    · -- the push takes its op and is issued
      cases hs
      rename_i hdr
      have hdr0 : s.life.draining = false := by simpa using hdr
      have hc' := life_incr hl hdr0
      have hdn0 := hl.not_done hdr0
      have hdn' : s.life.done = true → s.life.done = true := id
      constructor <;> simp only [baton, hp, ha, his, hc, hdr0, hdn0, Bool.false_eq_true,
        false_implies, implies_true, Nat.add_zero, Nat.zero_add, hu]
      pw_auto

theorem sendNext_inv {s s' : S} (h : Inv s) (hs : s.sendNext = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold sendNext at hs
  split at hs
  · cases hs
  · rename_i hne0
    unfold baton at hb hlive
    have hc : s.chaining = 1 := by omega
    have hq : s.queued - s.woken = 0 := by omega
    have ha : s.admitting = 0 := by omega
    have his : s.issuing = 0 := by omega
    have hp : s.parked = false := by
      cases hpk : s.parked
      · rfl
      · simp [hpk] at hb; omega
    simp only [hp] at hil hnd hb hlive hpw
    have hne : s.life.inFlight ≠ 0 := by omega
    cases hs
    unfold trySendNext
    split
    · -- `SendNextUnit(u + 1)` arms the chain on the next unit
      rename_i hnext
      obtain ⟨hlt, hdr0⟩ := hnext
      obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_incr_endOp hl hdr0
      have hne' : ({ s.life with inFlight := s.life.inFlight + 1 } : Lifecycle).inFlight ≠ 0 := by
        simp
      constructor <;> simp only [endOp, hne', if_false, baton, hp, ha, his, hc, hdr', hif', hst',
        Bool.false_eq_true, false_implies, decide_false, Bool.or_false, hu,
        Nat.add_zero, Nat.zero_add]
      pw_auto
    · -- past the last unit, or draining: the chain ends
      rename_i hnext
      have hend : s.life.draining = false →
          s.h2hIssued = s.numUnits ∧ s.h2hRetired < s.h2hIssued := by
        intro hdr0
        have hqge : s.numUnits ≤ s.queued := by
          by_cases hlt' : s.queued < s.numUnits
          · exact absurd ⟨hlt', hdr0⟩ hnext
          · omega
        have hiss : s.h2hIssued = s.woken := by have := hnd hdr0; omega
        have hok : s.h2hOk = s.h2hRetired := by
          by_cases hlt2 : s.h2hOk < s.h2hRetired
          · have := hfd hlt2; rw [hdr0] at this; cases this
          · omega
        refine ⟨by omega, ?_⟩
        by_cases heq : s.h2hRetired = s.h2hIssued
        · have := hall (by omega); rw [hdr0] at this; cases this
        · omega
      obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_endOp hl
      constructor <;> simp only [endOp, hne, if_false, baton, hp, ha, his, hc, hdr', hif', hst',
        Bool.false_eq_true, false_implies, decide_false, Bool.or_false, hu,
        Nat.add_zero, Nat.zero_add]
      case live => intro hdr0; exact Or.inr (hend hdr0)
      pw_auto

theorem h2hDone_inv {s s' : S} {ok : Bool} (h : Inv s) (hs : s.h2hDone ok = some s') :
    Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold h2hDone at hs
  dsimp only at hs
  split at hs
  · rename_i hlt
    unfold baton at hb hlive
    have hne : s.life.inFlight ≠ 0 := by omega
    -- a push in flight: the send is not settled, so not published, and the
    -- status cannot have been decided OK yet
    have hnpub : ∀ b, s.published ≠ some b := fun b hb' => hne (hl.done_idle (hpd b hb'))
    have hiss_le : s.h2hIssued ≤ s.numUnits := by
      cases hpk : s.parked <;> simp [hpk] at hil <;> omega
    have hnokd : ¬(s.life.statusOk = true ∧ s.life.draining = true) := fun ⟨a, b⟩ => by
      have := hokd a b; omega
    have hokeq : s.life.draining = false → s.h2hOk = s.h2hRetired := by
      intro hdr0
      by_cases hlt2 : s.h2hOk < s.h2hRetired
      · have := hfd hlt2; rw [hdr0] at this; cases this
      · omega
    split at hs
    · rename_i hok
      unfold resume at hs
      split at hs
      · -- the parked unit resumes
        rename_i hres
        dsimp only at hres
        obtain ⟨hpk, hdr0⟩ := hres
        simp [hpk] at hil hnd hb hlive
        split at hs
        · -- last push: cannot happen with a parked unit (`issued_le`)
          cases hs
          rename_i hlast
          dsimp only at hlast
          omega
        · cases hs
          rename_i hlast
          dsimp only at hlast
          obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_incr_endOp hl hdr0
          have hne' : ({ s.life with inFlight := s.life.inFlight + 1 } : Lifecycle).inFlight ≠ 0 :=
            by simp
          constructor <;> simp only [endOp, hne', if_false, baton, hdr0,
            Bool.false_eq_true, false_implies, decide_false, Bool.or_false, hu,
            Nat.add_zero]
          case published_ok => intro hp; exact absurd hp (hnpub true)
          pw_auto
      · rename_i hres
        dsimp only at hres
        split at hs
        · -- the last OK push finishes the send
          cases hs
          rename_i hlast
          dsimp only at hlast
          obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_finish_endOp true hl
          have hne' : (s.life.finishOnceLocked true).inFlight ≠ 0 := by simp; omega
          constructor <;> simp only [endOp, finish, hne', if_false, baton, hdr', hif',
            implies_true, decide_false, Bool.or_false, hu,
            ]
          case published_ok => intro hp; exact absurd hp (hnpub true)
          pw_auto
        · cases hs
          rename_i hlast
          dsimp only at hlast
          obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_endOp hl
          constructor <;> simp only [endOp, hne, if_false, baton, hdr', hif', hst',
            decide_false, Bool.or_false, hu,
            ]
          case published_ok => intro hp; exact absurd hp (hnpub true)
          case ok_draining => intro a b; exact absurd ⟨a, b⟩ hnokd
          case live =>
            intro hdr0
            have hpk : s.parked = false := by
              cases hp' : s.parked
              · rfl
              · exact absurd ⟨hp', hdr0⟩ hres
            have hok' := hokeq hdr0
            rcases hlive hdr0 with hb1 | ⟨hin, _⟩
            · left; simp only [hpk] at hb1 ⊢; omega
            · right; refine ⟨hin, ?_⟩; omega
          pw_auto
    · -- a failed push finishes the send
      cases hs
      obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_finish_endOp false hl
      have hne' : (s.life.finishOnceLocked false).inFlight ≠ 0 := by simp; omega
      constructor <;> simp only [endOp, finish, hne', if_false, baton, hdr', hif',
        implies_true, decide_false, Bool.or_false, hu,
        ]
      case published_ok => intro hp; exact absurd hp (hnpub true)
      all_goals first
        | (intro hok _; obtain ⟨hso, hsd⟩ := ok_of_finish hl (hst' ▸ hok) rfl; exact hokd hso hsd)
        | skip
      pw_auto
  · cases hs

theorem cancel_inv {s s' : S} (h : Inv s) (hs : s.cancel = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold cancel at hs
  cases hs
  unfold baton at hb hlive
  obtain ⟨hc', hif', hdr', hst', hdn'⟩ := life_finish false hl
  constructor <;> simp only [finish, baton, hdr', hif',
    implies_true, hu]
  all_goals first
    | (intro hok _; obtain ⟨hso, hsd⟩ := ok_of_finish hl (hst' ▸ hok) rfl; exact hokd hso hsd)
    | skip
  pw_auto

theorem publish_inv {s s' : S} (h : Inv s) (hs : s.publish = some s') : Inv s' := by
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold publish at hs
  split at hs
  · cases hs
    rename_i hpub
    obtain ⟨hd, _⟩ := hpub
    exact ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
      hlive, fun _ _ => hd, fun hp => by simp at hp; exact hokd hp (hl.done_draining hd)⟩
  · cases hs

theorem step_inv {s s' : S} {e : Ev} (h : Inv s) (hs : s.step e = some s') : Inv s' := by
  cases e with
  | wake ok => exact wake_inv h hs
  | admit => exact admit_inv h hs
  | issue => exact issue_inv h hs
  | sendNext => exact sendNext_inv h hs
  | h2hDone ok => exact h2hDone_inv h hs
  | cancel => exact cancel_inv h hs
  | publish => exact publish_inv h hs

theorem reachable_inv {n w : Nat} (hn : 0 < n) {s : S} (hr : (sys n w).Reachable s) : Inv s :=
  (sys n w).reachable_induction (inv_init hn) (fun _ _ _ h hs => step_inv h hs) hr

theorem inv_noOpLeak {s : S} (h : Inv s) : NoOpLeak s := by
  intro hif
  have hacc := h.accounted
  by_cases hwk : s.woken < s.queued
  · refine ⟨.wake true, by simp [drainEvents], ?_⟩
    simp only [step, wake, hwk, ↓reduceIte, Bool.true_eq_false]
    split <;> rfl
  · by_cases ha : s.admitting = 0
    · by_cases his : s.issuing = 0
      · by_cases hc : s.chaining = 0
        · have hlt : s.h2hRetired < s.h2hIssued := by omega
          exact ⟨.h2hDone true, by simp [drainEvents], by simp [step, h2hDone, hlt]⟩
        · exact ⟨.sendNext, by simp [drainEvents], by simp [step, sendNext, hc]⟩
      · refine ⟨.issue, by simp [drainEvents], ?_⟩
        simp only [step, issue, his, ↓reduceIte]
        split <;> rfl
    · refine ⟨.admit, by simp [drainEvents], ?_⟩
      simp only [step, admit, ha, ↓reduceIte]
      split
      · rfl
      · split <;> rfl

theorem inv_safe {s : S} (h : Inv s) : Safe s := by
  have hnl := inv_noOpLeak h
  obtain ⟨hl, hacc, hu, hb, hwq, hqn, hil, hnd, hrl, hol, hpif, hwin, hpw, hokd, hfd, hall,
    hlive, hpd, hpo⟩ := h
  unfold baton at hb hlive
  refine ⟨hl.done_idle, ?_, hl.staging, hl.prompt, hu, ?_, hpw, ?_, hpo, hnl⟩
  · intro hd
    have := hl.done_idle hd
    omega
  · intro hw
    have := hwin hw
    omega
  · intro hdr
    rcases hlive hdr with hb1 | ⟨_, hlt⟩
    · cases hpk : s.parked
      · simp [hpk] at hb1; omega
      · have := hpw hpk hdr; omega
    · omega

theorem reachable_safe {n w : Nat} (hn : 0 < n) {s : S} (hr : (sys n w).Reachable s) : Safe s :=
  inv_safe (reachable_inv hn hr)

end S

end TpuSyncVerify.Transfer.PrefillDecode.PushWindow

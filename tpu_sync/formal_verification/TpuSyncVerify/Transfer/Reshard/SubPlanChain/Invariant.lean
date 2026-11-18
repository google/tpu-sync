import TpuSyncVerify.Transfer.Reshard.Coordinator.Lemmas
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Model

/-! # `SubPlanChain`: safety predicates, inductive invariants, `reachable_safe`

Part of `TpuSyncVerify.Transfer.Reshard.SubPlanChain`; the module docstring and
source map live in `SubPlanChain.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard.SubPlanChain

open TpuSyncVerify
open TpuSyncVerify.Transfer.Reshard.Registry
open TpuSyncVerify.Transfer.Reshard.Coordinator
  (CoordPhase abandonIfClaimed abandonIfClaimed_inv abandonIfClaimed_clears_single_owner
   register_preserves_idle_reg lookupAndClaim_from_idle_shape)

/-! ## Part I — `SubPlanUuid` and `PlanSubTransfers` -/

/-- `SubPlanUuid(uuid, 0)` is the identity (`coord.cc`). -/
theorem subPlanUuid_zero (uuid : UInt64) : subPlanUuid uuid 0 = uuid :=
  rfl

private theorem uint64_and_mask_le (x : UInt64) :
    (x &&& (0x7FFFFFFFFFFFFFFF : UInt64)) ≤ (0x7FFFFFFFFFFFFFFF : UInt64) := by
  rw [UInt64.le_iff_toNat_le, UInt64.toNat_and]
  exact Nat.and_le_right

/-- For every `index > 0`, `SubPlanUuid(uuid, index)` is strictly positive and
fits in a signed 64-bit positive integer (`[1, 0x7FFFFFFFFFFFFFFF]`). -/
theorem subPlanUuid_pos_of_pos (uuid : UInt64) {index : Nat} (h : 0 < index) :
    (0 : UInt64) < subPlanUuid uuid index ∧
    subPlanUuid uuid index ≤ (0x7FFFFFFFFFFFFFFF : UInt64) := by
  unfold subPlanUuid
  have hne : (index == 0) = false := beq_false_of_ne (Nat.ne_of_gt h)
  simp only [hne, Bool.false_eq_true, ↓reduceIte]
  split
  · rename_i hpos
    exact ⟨hpos, uint64_and_mask_le _⟩
  · decide

/-- **Aliased-page windowing guard (`planSubTransfers_no_aliased_window_split`,
`PlanSubTransfers` in `reshard_coordinator.cc`)**:
Whenever `planSubTransfers` produces a multi-window chain (`chain.count > 1`),
`hasWindowedTagSharingIds` is `false` (no windowed tag shares any destination
block ID with another tag, preserving intra-plan `RankWaveOrder` on aliased
device pages) and `chain.subPlans.length = chain.count`. -/
theorem planSubTransfers_no_aliased_window_split {args : PlanArgs} {chain : SubTransferChain}
    (hplan : planSubTransfers args = some chain) (hgt : 1 < chain.count) :
    chain.subPlans.length = chain.count ∧
    ∃ counts,
      let windows := args.dstWindowBlocks.map Int.toNat
      let winPerTag := (List.range args.numTags).map fun i =>
        windowsForTag (counts.getD i 0) (windows.getD i 0)
      hasWindowedTagSharingIds (idsByTag args.dstDeviceBlockIds counts) winPerTag = false := by
  unfold planSubTransfers at hplan
  split at hplan; · cases hplan
  split at hplan; · injection hplan with h; subst h; simp at hgt
  split at hplan; · cases hplan
  cases hcnt :
      (if args.dstBlockCounts.isEmpty then
        if args.numTags != 1 then none else some [args.dstDeviceBlockIds.length]
      else if args.dstBlockCounts.length != args.numTags then none
      else some args.dstBlockCounts) with
  | none => simp only [hcnt] at hplan; injection hplan with h; subst h; simp at hgt
  | some counts =>
    simp only [hcnt] at hplan
    split at hplan; · injection hplan with h; subst h; simp at hgt
    split at hplan; · injection hplan with h; subst h; simp at hgt
    rename_i hno_share
    split at hplan; · injection hplan with h; subst h; simp at hgt
    injection hplan with h; subst h
    exact ⟨by simp, counts, by simpa using hno_share⟩

/-! ## Part II — Worker chain folding -/

theorem mem_insertNodup {x y : Nat} {xs : List Nat} :
    x ∈ insertNodup y xs ↔ x = y ∨ x ∈ xs := by
  unfold insertNodup
  split
  · rename_i h
    have hm : y ∈ xs := List.contains_iff_mem.mp h
    constructor
    · intro hx; exact Or.inr hx
    · rintro (rfl | hx) <;> assumption
  · simp [List.mem_append, or_comm]

private theorem nodup_append_single {l : List Nat} {x : Nat} (h : l.Nodup) (hx : x ∉ l) :
    (l ++ [x]).Nodup := by
  induction l with
  | nil => simp
  | cons a t ih =>
    rw [List.nodup_cons] at h
    simp only [List.cons_append, List.nodup_cons, List.mem_append, List.mem_singleton]
    exact ⟨fun hm => hm.elim h.1 fun e => hx (e ▸ List.mem_cons_self ..),
      ih h.2 fun hm => hx (List.mem_cons_of_mem _ hm)⟩

theorem insertNodup_nodup {x : Nat} {xs : List Nat} (h : xs.Nodup) : (insertNodup x xs).Nodup := by
  unfold insertNodup
  split
  · exact h
  · rename_i hc
    exact nodup_append_single h (fun hm => hc (List.contains_iff_mem.mpr hm))

/-- **`done` only after every sub-plan settled (`DoneAfterAllSubPlans`,
`RecordPoolReshardTerminalLocked`)**: every insertion of `req_id` into
`done_recving_` / `done_sending_` was made with every sub-plan slot of the
request settled OK — `settled_count == progress.sub_sessions.size()` over a
`sub_sessions` vector at least `sub_plan_count` long — or, for a single-plan
request, with its one session settled OK. -/
def DoneAfterAllSubPlans (w : WorkerState) : Prop :=
  ∀ e ∈ w.doneLog, e.2.Nodup ∧ (∀ i ∈ e.2, i < e.1) ∧ e.1 ≤ e.2.length

structure WorkerSafe (w : WorkerState) : Prop where
  doneAfterAll : DoneAfterAllSubPlans w

/-- A chained session whose terminal can still be recorded as OK is accounted
for by the current entry: the entry exists and either holds the session in
`sub_sessions` or is anchored (so the session is a superseded chain's and will
be ignored). -/
def ChainedTracked (c : ChainProgress) (s : SubSession) : Prop :=
  1 < s.count → s.retired = false → s.ok = true →
    c.present = true ∧ (c.subSessions.contains (some s.uuid) = true ∨ c.anchored = true)

/-- Slot 0 is filled only by the anchoring registration. -/
def Slot0Anchored (c : ChainProgress) : Prop :=
  c.anchored = false → c.subSessions.getD 0 none = none

/-- Inductive invariant of `WorkerState.stepRestricted`. -/
structure WorkerInv (w : WorkerState) : Prop where
  safe : WorkerSafe w
  sessionIndex : ∀ s ∈ w.sessions, s.index < max 1 s.count
  slot0 : Slot0Anchored w.chain
  tracked : ∀ s ∈ w.sessions, ChainedTracked w.chain s

theorem init_workerInv : WorkerInv WorkerState.init where
  safe := ⟨fun _ h => by cases h⟩
  sessionIndex := fun _ h => by cases h
  slot0 := fun _ => rfl
  tracked := fun _ h => by cases h

private def GoodEntry (e : Nat × List Nat) : Prop :=
  e.2.Nodup ∧ (∀ i ∈ e.2, i < e.1) ∧ e.1 ≤ e.2.length

/-! ### Slot lemmas -/

private theorem getD_growTo (l : List (Option Nat)) (n j : Nat) :
    (growTo l n).getD j none = l.getD j none := by
  unfold growTo
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
  by_cases hj : j < l.length
  · rw [List.getElem?_append_left hj]
  · rw [List.getElem?_append_right (Nat.le_of_not_lt hj), List.getElem?_eq_none (Nat.le_of_not_lt hj)]
    rw [List.getElem?_replicate]
    split <;> rfl

private theorem length_growTo (l : List (Option Nat)) (n : Nat) :
    n ≤ (growTo l n).length := by
  unfold growTo
  simp only [List.length_append, List.length_replicate]
  omega

private theorem getD_set_ne (l : List (Option Nat)) {i j : Nat} (h : i ≠ j) (x : Option Nat) :
    (l.set i x).getD j none = l.getD j none := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_set_ne h]

private theorem getD_set_self (l : List (Option Nat)) {i : Nat} (h : i < l.length) (x : Option Nat) :
    (l.set i x).getD i none = x := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_set_self h]; rfl

private theorem mem_iff_getD {l : List (Option Nat)} {u : Nat} :
    some u ∈ l ↔ ∃ j, l.getD j none = some u := by
  rw [List.mem_iff_getElem?]
  constructor
  · rintro ⟨j, hj⟩
    exact ⟨j, by rw [List.getD_eq_getElem?_getD, hj]; rfl⟩
  · rintro ⟨j, hj⟩
    refine ⟨j, ?_⟩
    rw [List.getD_eq_getElem?_getD] at hj
    cases h : l[j]? with
    | none => rw [h] at hj; cases hj
    | some x => rw [h] at hj; simpa using hj

private theorem contains_iff_getD {l : List (Option Nat)} {u : Nat} :
    l.contains (some u) = true ↔ ∃ j, l.getD j none = some u := by
  rw [List.contains_iff_mem, mem_iff_getD]

/-- The slots after an `intoEntry` registration keep every filled slot of the
old entry when the written slot was empty. -/
private theorem contains_set_of_empty {l : List (Option Nat)} {i n u v : Nat}
    (hempty : l.getD i none = none) (hu : l.contains (some u) = true) :
    ((growTo l n).set i (some v)).contains (some u) = true := by
  obtain ⟨j, hj⟩ := contains_iff_getD.mp hu
  have hne : i ≠ j := fun h => by subst h; rw [hempty] at hj; cases hj
  exact contains_iff_getD.mpr ⟨j, by rw [getD_set_ne _ hne, getD_growTo, hj]⟩

private theorem settledSlots_nodup (ss : List SubSession) (slots : List (Option Nat)) :
    (settledSlots ss slots).Nodup :=
  List.filter_sublist.nodup List.nodup_range

private theorem settledSlots_lt (ss : List SubSession) (slots : List (Option Nat)) :
    ∀ i ∈ settledSlots ss slots, i < slots.length := by
  intro i hi
  exact List.mem_range.mp (List.mem_filter.mp hi).1

/-- When every slot settled, slot 0 is filled. -/
private theorem slot0_filled_of_all_settled {ss : List SubSession} {slots : List (Option Nat)}
    (hne : slots ≠ []) (hall : (settledSlots ss slots).length = slots.length) :
    slots.getD 0 none ≠ none := by
  unfold settledSlots at hall
  have hall' : (List.filter (fun i => slotSettledOk ss (slots.getD i none))
      (List.range slots.length)).length = (List.range slots.length).length := by
    rw [List.length_range]; exact hall
  have hpos : 0 < slots.length := by cases slots with | nil => exact absurd rfl hne | cons _ _ => simp
  have h0 : 0 ∈ List.range slots.length := List.mem_range.mpr hpos
  have hp := (List.length_filter_eq_length_iff.mp hall') 0 h0
  intro hnone
  rw [hnone] at hp
  cases hp

/-- The four arms of `NoteSubPlanRegisteredLocked`, as rewriting equations. -/
private theorem note_single (c : ChainProgress) (uuid index count : Nat) (h : count ≤ 1) :
    noteSubPlanRegistered c uuid index count = ({}, .single) := by
  unfold noteSubPlanRegistered; simp [h]

private theorem note_anchor (c : ChainProgress) (uuid count : Nat) (h : 1 < count) :
    noteSubPlanRegistered c uuid 0 count =
      ({ present := true, anchored := true,
         subSessions := (growTo [] (max 0 (max count 1))).set 0 (some uuid) }, .anchor) := by
  unfold noteSubPlanRegistered
  simp [Nat.not_le_of_lt h]

private theorem note_into (c : ChainProgress) (uuid index count : Nat) (h : 1 < count)
    (hi : index ≠ 0) (hp : c.present = true) :
    noteSubPlanRegistered c uuid index count =
      ({ c with subSessions := ((growTo c.subSessions (max c.subSessions.length (max count (index + 1)))).set index (some uuid)) }, .intoEntry) := by
  unfold noteSubPlanRegistered
  simp [Nat.not_le_of_lt h, hi, hp]

private theorem note_unanchored (c : ChainProgress) (uuid index count : Nat) (h : 1 < count)
    (hi : index ≠ 0) (hp : c.present = false) :
    noteSubPlanRegistered c uuid index count =
      ({ present := true,
         subSessions := (growTo [] (max 0 (max count (index + 1)))).set index (some uuid) },
       .unanchored) := by
  unfold noteSubPlanRegistered
  simp [Nat.not_le_of_lt h, hi, hp]

/-! ### `recordTerminal` -/

/-- Shape of one `RecordPoolReshardTerminalLocked` call: the entry's presence
and anchoring are kept, slot 0 stays anchored, every tracked session stays
tracked, and a `done` log entry is good. -/
private theorem recordTerminal_shape (c : ChainProgress) (ss : List SubSession) (s : SubSession)
    (hidx : s.index < max 1 s.count) (hret : s.retired = false)
    (hs : ChainedTracked c s) (h0 : Slot0Anchored c) :
    let r := recordTerminal c ss s
    r.1.present = c.present ∧ r.1.anchored = c.anchored ∧ Slot0Anchored r.1 ∧
    (∀ t, ChainedTracked c t → ChainedTracked r.1 t) ∧
    (∀ k idx, r.2 = .done k idx → GoodEntry (k, idx)) ∧
    (∀ ok, r.2 = .direct ok ∨ r.2 = .stray ok → ok = true → GoodEntry (max 1 s.count, [s.index])) := by
  -- A chained OK session is never `direct` or `stray`.
  have hsingle : ∀ ok, (recordTerminal c ss s).2 = .direct ok ∨ (recordTerminal c ss s).2 = .stray ok →
      ok = true → GoodEntry (max 1 s.count, [s.index]) := by
    intro ok hok htrue
    subst htrue
    have hok' : s.ok = true := by
      unfold recordTerminal at hok
      rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
    have hcount : s.count ≤ 1 := by
      rcases Nat.lt_or_ge 1 s.count with hgt | hle
      · exfalso
        obtain ⟨hpres, hin⟩ := hs hgt hret hok'
        unfold recordTerminal at hok
        rw [hpres] at hok
        simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hok
        rcases hin with hin | hin
        · rw [hin] at hok
          simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hok
          rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
        · rw [hin] at hok
          rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
      · exact hle
    have hidx0 : s.index = 0 := by
      have : max 1 s.count = 1 := by omega
      omega
    refine ⟨by simp, ?_, by simp only [List.length_singleton]; omega⟩
    intro i hi
    simp only [List.mem_singleton] at hi
    subst hi
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, hsingle⟩
  all_goals unfold recordTerminal
  all_goals (try dsimp only)
  · repeat' split
    all_goals rfl
  · repeat' split
    all_goals rfl
  · repeat' split
    all_goals first | exact h0 | (intro _; rfl)
  · intro t ht hc hr ho
    obtain ⟨hp, hin⟩ := ht hc hr ho
    repeat' split
    all_goals (try exact ⟨hp, hin⟩)
    -- the `done` arm: slots cleared, but the entry was anchored
    rename_i hpres hfound _ _ hall
    refine ⟨hp, Or.inr ?_⟩
    simp only [Bool.not_eq_true'] at hpres hfound
    have hne : c.subSessions ≠ [] := by
      intro h; rw [h] at hfound; simp at hfound
    have hfilled := slot0_filled_of_all_settled (ss := ss) hne (by simpa using hall)
    show c.anchored = true
    cases hanch : c.anchored with
    | false => exact absurd (h0 hanch) hfilled
    | true => rfl
  · intro k idx h
    repeat' split at h
    all_goals (try cases h)
    rename_i hall
    have heq : (settledSlots ss c.subSessions).length = c.subSessions.length := by simpa using hall
    exact ⟨settledSlots_nodup _ _, settledSlots_lt _ _, Nat.le_of_eq heq.symm⟩

/-! ### The poll fold -/

private theorem foldRecord_shape (ss : List SubSession) (settled : List SubSession) :
    ∀ (acc : WorkerState.SweepAcc),
      Slot0Anchored acc.chain →
      (∀ s ∈ settled, s.index < max 1 s.count ∧ s.retired = false) →
      (∀ s ∈ settled, ChainedTracked acc.chain s) →
      (∀ t ∈ ss, ChainedTracked acc.chain t) →
      (∀ e ∈ acc.log, GoodEntry e) →
      let r := settled.foldl (WorkerState.recordStep ss) acc
      r.chain.present = acc.chain.present ∧ Slot0Anchored r.chain ∧
      (∀ t ∈ ss, ChainedTracked r.chain t) ∧ ∀ e ∈ r.log, GoodEntry e := by
  induction settled with
  | nil =>
    intro acc h0 _ _ htr hlog
    exact ⟨rfl, h0, htr, hlog⟩
  | cons s rest ih =>
    intro acc h0 hs hsettled htr hlog
    obtain ⟨hidx, hret⟩ := hs s (List.mem_cons_self ..)
    have hrest : ∀ s' ∈ rest, s'.index < max 1 s'.count ∧ s'.retired = false :=
      fun s' h => hs s' (List.mem_cons_of_mem _ h)
    obtain ⟨hpres, _, h0', hkeep, hgood, hsingle⟩ :=
      recordTerminal_shape acc.chain ss s hidx hret (hsettled s (List.mem_cons_self ..)) h0
    simp only [List.foldl_cons]
    have hstep : ∀ acc' : WorkerState.SweepAcc, WorkerState.recordStep ss acc s = acc' →
        acc'.chain = (recordTerminal acc.chain ss s).1 ∧
        ∀ e ∈ acc'.log, GoodEntry e := by
      intro acc' h
      unfold WorkerState.recordStep at h
      rcases hrt : recordTerminal acc.chain ss s with ⟨c', t⟩
      rw [hrt] at h hgood hsingle
      have hlog' : ∀ (extra : List (Nat × List Nat)), (∀ e ∈ extra, GoodEntry e) →
          ∀ e ∈ acc.log ++ extra, GoodEntry e := by
        intro extra hextra e he
        rcases List.mem_append.mp he with he | he
        · exact hlog e he
        · exact hextra e he
      cases t with
      | done k idx =>
        subst h
        exact ⟨rfl, hlog' _ fun e he => by
          simp only [List.mem_singleton] at he; subst he; exact hgood k idx rfl⟩
      | direct ok =>
        cases ok
        · subst h; exact ⟨rfl, hlog⟩
        · subst h
          exact ⟨rfl, hlog' _ fun e he => by
            simp only [List.mem_singleton] at he; subst he; exact hsingle true (Or.inl rfl) rfl⟩
      | stray ok =>
        cases ok
        · subst h; exact ⟨rfl, hlog⟩
        · subst h
          exact ⟨rfl, hlog' _ fun e he => by
            simp only [List.mem_singleton] at he; subst he; exact hsingle true (Or.inr rfl) rfl⟩
      | _ => subst h; exact ⟨rfl, hlog⟩
    obtain ⟨hchain, hlog1⟩ := hstep _ rfl
    obtain ⟨hp', h0'', htr'', hlog''⟩ := ih _ (hchain ▸ h0') hrest
      (fun s' hs' => by rw [hchain]; exact hkeep s' (hsettled s' (List.mem_cons_of_mem _ hs')))
      (fun t ht => by rw [hchain]; exact hkeep t (htr t ht)) hlog1
    exact ⟨by rw [hp', hchain, hpres], h0'', htr'', hlog''⟩

private theorem recordAll_shape (c : ChainProgress) (ss settled : List SubSession)
    (h0 : Slot0Anchored c)
    (hs : ∀ s ∈ settled, s.index < max 1 s.count ∧ s.retired = false)
    (hsettled : ∀ s ∈ settled, ChainedTracked c s)
    (htr : ∀ t ∈ ss, ChainedTracked c t) :
    (WorkerState.recordAll c ss settled).chain.present = c.present ∧
    Slot0Anchored (WorkerState.recordAll c ss settled).chain ∧
    (∀ t ∈ ss, ChainedTracked (WorkerState.recordAll c ss settled).chain t) ∧
    ∀ e ∈ (WorkerState.recordAll c ss settled).log, GoodEntry e :=
  foldRecord_shape ss settled { chain := c } h0 hs hsettled htr (fun _ h => by cases h)

private theorem sweep_slot0 (c : ChainProgress) (b : Bool) (h0 : Slot0Anchored c) :
    Slot0Anchored (sweepStalledChain c b).1 := by
  unfold sweepStalledChain
  repeat' split
  all_goals first | exact h0 | (intro _; rfl)

private theorem applyDeadlineArm_shape (t0 : SubSession) :
    let t := applyDeadlineArm t0
    t.uuid = t0.uuid ∧ t.index = t0.index ∧ t.count = t0.count ∧ t.retired = t0.retired ∧
    (t.ok = true → t0.ok = true) ∧
    (t.retired = false → t.done = false → t.ok = true → t.draining = true) := by
  rcases t0 with ⟨_, _, _, dr, inf, dn, _, ret⟩
  cases ret <;> cases dn <;> cases dr <;> cases inf <;> simp [applyDeadlineArm]

private theorem anyOpenSlot_of_contains {ss : List SubSession} {slots : List (Option Nat)}
    {s : SubSession} (hs : s ∈ ss) (hopen : s.done = false)
    (hcont : slots.contains (some s.uuid) = true) :
    anyOpenSlot ss slots = true := by
  unfold anyOpenSlot
  rw [List.any_eq_true]
  refine ⟨some s.uuid, List.contains_iff_mem.mp hcont, ?_⟩
  rw [List.any_eq_true]
  exact ⟨s, hs, by simp [hopen]⟩

theorem WorkerInv.toSafe {w : WorkerState} (h : WorkerInv w) : WorkerSafe w := h.safe

/-! ### Per-event preservation -/

private theorem stepG_registerSubPlan {g : WorkerState.Guards} {w w' : WorkerState}
    {uuid index count : Nat}
    (hg6 : g.noSingleWhileSeated = true) (hg7 : g.oneSessionPerSlot = true)
    (hinv : WorkerInv w)
    (hstep : WorkerState.stepG g w (.registerSubPlan uuid index count) = some w') : WorkerInv w' := by
  simp only [WorkerState.stepG, hg6, hg7] at hstep
  split at hstep; · cases hstep
  split at hstep; · cases hstep
  rename_i hidx
  split at hstep; · cases hstep
  rename_i hA6
  split at hstep; · cases hstep
  rename_i hA7
  injection hstep with h; subst h
  simp only [Bool.true_and, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq, bne_iff_ne,
    ne_eq, Bool.not_eq_true, not_and, Option.isSome_eq_false_iff, Option.isNone_iff_eq_none,
    List.any_eq_false] at hA6 hA7
  have hidx' : index < max 1 count := Nat.lt_of_not_ge hidx
  have hsessIdx : ∀ s ∈ w.sessions ++ [{ uuid := uuid, index := index, count := count }],
      s.index < max 1 s.count := by
    intro s hs
    rcases List.mem_append.mp hs with hs | hs
    · exact hinv.sessionIndex s hs
    · simp only [List.mem_singleton] at hs; subst hs; exact hidx'
  rcases Nat.lt_or_ge 1 count with hgt | hle
  · -- a chained registration
    by_cases hi : index = 0
    · subst hi
      rw [note_anchor _ _ _ hgt]
      refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, (fun h => nomatch h), ?_⟩
      intro s hs hc hr ho
      refine ⟨rfl, ?_⟩
      rcases List.mem_append.mp hs with hs | hs
      · exact Or.inr rfl
      · simp only [List.mem_singleton] at hs; subst hs
        exact Or.inl (contains_iff_getD.mpr ⟨0, getD_set_self _
          (Nat.lt_of_lt_of_le (by omega) (length_growTo _ _)) _⟩)
    · by_cases hp : w.chain.present = true
      · rw [note_into _ _ _ _ hgt hi hp]
        have hempty : w.chain.subSessions.getD index none = none := hA7 ⟨⟨hgt, hi⟩, hp⟩
        refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, ?_, ?_⟩
        · intro hanch
          show ((growTo w.chain.subSessions _).set index (some uuid)).getD 0 none = none
          rw [getD_set_ne _ hi, getD_growTo]
          exact hinv.slot0 hanch
        · intro s hs hc hr ho
          rcases List.mem_append.mp hs with hs | hs
          · obtain ⟨_, hin⟩ := hinv.tracked s hs hc hr ho
            refine ⟨hp, ?_⟩
            rcases hin with hin | hin
            · exact Or.inl (contains_set_of_empty hempty hin)
            · exact Or.inr hin
          · simp only [List.mem_singleton] at hs; subst hs
            exact ⟨hp, Or.inl (contains_iff_getD.mpr ⟨index, getD_set_self _
              (Nat.lt_of_lt_of_le (by omega) (length_growTo _ _)) _⟩)⟩
      · have hp' : w.chain.present = false := by simpa using hp
        rw [note_unanchored _ _ _ _ hgt hi hp']
        refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, ?_, ?_⟩
        · intro _
          show ((growTo [] _).set index (some uuid)).getD 0 none = none
          rw [getD_set_ne _ hi, getD_growTo]; rfl
        · intro s hs hc hr ho
          rcases List.mem_append.mp hs with hs | hs
          · obtain ⟨hpres, _⟩ := hinv.tracked s hs hc hr ho
            rw [hpres] at hp'; cases hp'
          · simp only [List.mem_singleton] at hs; subst hs
            exact ⟨rfl, Or.inl (contains_iff_getD.mpr ⟨index, getD_set_self _
              (Nat.lt_of_lt_of_le (by omega) (length_growTo _ _)) _⟩)⟩
  · -- a single-plan registration erases the entry; A6: no chained session is seated
    rw [note_single _ _ _ _ hle]
    refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, fun _ => rfl, ?_⟩
    intro s hs hc hr _
    rcases List.mem_append.mp hs with hs | hs
    · exact absurd hr (hA6 hle s hs hc)
    · simp only [List.mem_singleton] at hs; subst hs
      simp at hc; omega

private theorem stepG_beginOp {g : WorkerState.Guards} {w w' : WorkerState} {uuid : Nat}
    (hinv : WorkerInv w)
    (hstep : WorkerState.stepG g w (.beginOp uuid) = some w') : WorkerInv w' := by
  simp only [WorkerState.stepG] at hstep
  split at hstep; · cases hstep
  injection hstep with h; subst h
  refine ⟨⟨hinv.safe.doneAfterAll⟩, ?_, hinv.slot0, ?_⟩
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have := hinv.sessionIndex s hs
    split <;> simpa using this
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have ht := hinv.tracked s hs
    split <;> exact ht

private theorem stepG_latchDrain {g : WorkerState.Guards} {w w' : WorkerState}
    {uuid : Nat} {ok : Bool}
    (hinv : WorkerInv w)
    (hstep : WorkerState.stepG g w (.latchDrain uuid ok) = some w') : WorkerInv w' := by
  simp only [WorkerState.stepG] at hstep
  split at hstep; · cases hstep
  injection hstep with h; subst h
  refine ⟨⟨hinv.safe.doneAfterAll⟩, ?_, hinv.slot0, ?_⟩
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have := hinv.sessionIndex s hs
    split <;> simpa using this
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have ht := hinv.tracked s hs
    split
    · intro hc hr ho
      simp only [Bool.and_eq_true] at ho
      exact ht hc hr ho.1
    · exact ht

private theorem stepG_settleSession {g : WorkerState.Guards} {w w' : WorkerState}
    {uuid : Nat} {ok : Bool}
    (hinv : WorkerInv w)
    (hstep : WorkerState.stepG g w (.settleSession uuid ok) = some w') : WorkerInv w' := by
  simp only [WorkerState.stepG] at hstep
  split at hstep; · cases hstep
  injection hstep with h; subst h
  refine ⟨⟨hinv.safe.doneAfterAll⟩, ?_, hinv.slot0, ?_⟩
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have := hinv.sessionIndex s hs
    split <;> simpa using this
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have ht := hinv.tracked s hs
    split
    · intro hc hr ho
      simp only [Bool.and_eq_true] at ho
      exact ht hc hr ho.1
    · exact ht

private theorem applyMidSettle_empty : WorkerState.applyMidSettle [] = id := by
  funext s; simp [WorkerState.applyMidSettle]

private theorem step_poll {w w' : WorkerState} {stalled : Bool}
    (hinv : WorkerInv w) (hstep : WorkerState.stepRestricted w (.poll stalled) = some w') :
    WorkerInv w' := by
  simp only [WorkerState.stepRestricted, WorkerState.stepG, WorkerState.pollCore,
    WorkerState.validMidSettle, List.isEmpty_nil, Bool.and_false, ↓reduceIte, List.all_nil,
    Bool.not_true, Bool.false_eq_true, applyMidSettle_empty, List.map_id] at hstep
  generalize hss : (if (true && stalled) = true then
      w.sessions.map applyDeadlineArm
    else w.sessions) = ss at hstep
  have hss_shape : ∀ t ∈ ss, ∃ t0 ∈ w.sessions,
      t.uuid = t0.uuid ∧ t.index = t0.index ∧ t.count = t0.count ∧ t.retired = t0.retired ∧
      (t.ok = true → t0.ok = true) := by
    intro t ht
    split at hss
    · subst hss
      obtain ⟨t0, ht0, rfl⟩ := List.mem_map.mp ht
      obtain ⟨hu, hi, hc, hr, ho, _⟩ := applyDeadlineArm_shape t0
      exact ⟨t0, ht0, hu, hi, hc, hr, ho⟩
    · subst hss; exact ⟨t, ht, rfl, rfl, rfl, rfl, fun h => h⟩
  have hidx : ∀ t ∈ ss, t.index < max 1 t.count := by
    intro t ht
    obtain ⟨t0, ht0, _, hi, hc, _, _⟩ := hss_shape t ht
    rw [hi, hc]; exact hinv.sessionIndex t0 ht0
  have htr : ∀ t ∈ ss, ChainedTracked w.chain t := by
    intro t ht hc hr ho
    obtain ⟨t0, ht0, hu, _, hcnt, hret, hok⟩ := hss_shape t ht
    rw [hu]
    exact hinv.tracked t0 ht0 (hcnt ▸ hc) (hret ▸ hr) (hok ho)
  have hsettled : ∀ s ∈ ss.filter (fun s => s.done && !s.retired),
      s.index < max 1 s.count ∧ s.retired = false := by
    intro s hs
    have hm := List.mem_filter.mp hs
    refine ⟨hidx s hm.1, ?_⟩
    have := hm.2
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at this
    exact this.2
  obtain ⟨hpres, h0, htr', hlog⟩ :=
    recordAll_shape w.chain ss (ss.filter (fun s => s.done && !s.retired))
      hinv.slot0 hsettled (fun s hs => htr s (List.mem_filter.mp hs).1) htr
  generalize hr : WorkerState.recordAll w.chain ss (ss.filter (fun s => s.done && !s.retired)) = r
    at hstep hpres h0 htr' hlog
  split at hstep; · cases hstep
  rename_i hnoDrainOk
  have hanyEq : anyOpenSlot (ss.map fun s => if s.done then { s with retired := true } else s)
      r.chain.subSessions = anyOpenSlot ss r.chain.subSessions := by
    unfold anyOpenSlot
    congr 1; funext opt
    cases opt with
    | none => rfl
    | some u =>
      simp only [List.any_map]
      congr 1; funext s
      cases hd : s.done <;> simp [hd]
  rw [hanyEq] at hstep
  generalize hr2 : (if stalled && !r.touched then
      sweepStalledChain r.chain (anyOpenSlot ss r.chain.subSessions)
    else (r.chain, .notSwept)) = r2 at hstep
  rcases r2 with ⟨c2, sweep⟩
  have h02 : Slot0Anchored c2 := by
    split at hr2
    · have h := congrArg Prod.fst hr2
      simp only at h
      rw [← h]; exact sweep_slot0 _ _ h0
    · injection hr2 with h _; subst h; exact h0
  injection hstep with h; subst h
  refine ⟨⟨fun e he => (List.mem_append.mp he).elim (hinv.safe.doneAfterAll e) (hlog e)⟩, ?_, h02, ?_⟩
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have := hidx s hs
    split <;> simpa using this
  · intro s' hs'
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    intro hc hret hok
    have hopen : s.done = false := by
      cases hd : s.done with
      | true => simp [hd] at hret
      | false => rfl
    have hmap : (if s.done = true then { s with retired := true } else s) = s := by simp [hopen]
    rw [hmap] at hc hret hok ⊢
    have hcount : 1 < s.count := hc
    have hok' : s.ok = true := hok
    have hret' : s.retired = false := hret
    obtain ⟨hrpres, hin⟩ := htr' s hs hcount hret' hok'
    cases hst : stalled with
    | false =>
      simp only [hst, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at hr2
      injection hr2 with h _; subst h
      exact ⟨hrpres, hin⟩
    | true =>
      have hcase : (true && true) = true := rfl
      rw [hst, if_pos hcase] at hss
      have hdraining : s.draining = true := by
        subst hss
        obtain ⟨t0, _, rfl⟩ := List.mem_map.mp hs
        exact (applyDeadlineArm_shape t0).2.2.2.2.2 hret' hopen hok'
      rw [hst] at hr2
      split at hr2
      · rename_i hquiet
        have hnotRep : (r.chain.failed || r.chain.completed) = false := by
          cases hfc : (r.chain.failed || r.chain.completed) with
          | false => rfl
          | true =>
            exfalso
            apply hnoDrainOk
            simp only [hst, Bool.true_and, Bool.and_eq_true] at hquiet ⊢
            refine ⟨⟨⟨hquiet, hrpres⟩, hfc⟩, ?_⟩
            rw [List.any_eq_true]
            exact ⟨s, hs, by simp [hcount, hret', hdraining, hopen, hok']⟩
        unfold sweepStalledChain at hr2
        rw [hrpres, hnotRep] at hr2
        simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hr2
        split at hr2
        · injection hr2 with h _; subst h
          exact ⟨hrpres, hin⟩
        · rename_i hnoOpen
          injection hr2 with h _; subst h
          rcases hin with hcont | hanch
          · exact absurd (anyOpenSlot_of_contains hs hopen hcont) hnoOpen
          · exact ⟨rfl, Or.inr hanch⟩
      · injection hr2 with h _; subst h
        exact ⟨hrpres, hin⟩

theorem step_preserves_workerInv {w w' : WorkerState} {e : WorkerEv}
    (hinv : WorkerInv w) (hstep : WorkerState.stepRestricted w e = some w') : WorkerInv w' := by
  cases e with
  | registerSubPlan uuid index count => exact stepG_registerSubPlan rfl rfl hinv hstep
  | beginOp uuid => exact stepG_beginOp hinv hstep
  | latchDrain uuid ok => exact stepG_latchDrain hinv hstep
  | settleSession uuid ok => exact stepG_settleSession hinv hstep
  | poll stalled => exact step_poll hinv hstep
  | pollMidSettle stalled midSettle =>
    simp only [WorkerState.stepRestricted, WorkerState.stepG, WorkerState.pollCore,
      Bool.true_and] at hstep
    cases hne : midSettle.isEmpty <;> simp [hne] at hstep

/-- **Main worker theorem (`reachable_worker_safe`)**: in every reachable state of
`WorkerState.stepRestricted`, `DoneAfterAllSubPlans` holds.
`WorkerState.stepRestricted` is `WorkerState.step` (A1, A2, A6, A7 as guards and
the A5a deadline arm) restricted to executions that satisfy A5b and A5c. The
theorem does not hold for `WorkerState.step`: A5b and A5c are not enforced by
anything in the C++, and `trace_draining_ok_at_sweep_reports_done_after_failure`
and `trace_mid_poll_endop_unanchored_reports_done_after_stall` (`Traces.lean`)
are executions of `WorkerState.step` that violate the property in those two
windows. A6 and A7 are caller assumptions the C++ does not check either
(`trace_single_plan_registration_erases_live_chain`,
`trace_slot_overwrite_reports_stray_done`). -/
theorem reachable_worker_safe {w : WorkerState} (hr : WorkerState.sysRestricted.Reachable w) :
    WorkerSafe w :=
  (WorkerState.sysRestricted.reachable_induction init_workerInv
    (fun _ _ _ hinv hstep => step_preserves_workerInv hinv hstep) hr).toSafe

/-! ### The alternative `stepSweepWaitsForOpen`, without A5b and without A5c -/

/-- Strengthened tracking invariant for `stepSweepWaitsForOpen`: an unretired
OK chained session is covered by a present entry through `sub_sessions`,
`anchored`, or a terminal flag `failed` / `completed` (checked before
`!anchored` in `recordTerminalWaitOpen` and retained by
`sweepStalledChainWaitOpen` while `sessions_` holds any unretired entry). -/
def ChainedTrackedWaitOpen (c : ChainProgress) (s : SubSession) : Prop :=
  1 < s.count → s.retired = false → s.ok = true →
    c.present = true ∧
    (c.subSessions.contains (some s.uuid) = true ∨ c.anchored = true ∨
      c.failed = true ∨ c.completed = true)

structure WorkerWaitOpenInv (w : WorkerState) : Prop where
  safe : WorkerSafe w
  sessionIndex : ∀ s ∈ w.sessions, s.index < max 1 s.count
  tracked : ∀ s ∈ w.sessions, ChainedTrackedWaitOpen w.chain s

theorem init_workerWaitOpenInv : WorkerWaitOpenInv WorkerState.init where
  safe := ⟨fun _ h => by cases h⟩
  sessionIndex := fun _ h => by cases h
  tracked := fun _ h => by cases h

private theorem recordTerminalWaitOpen_shape (c : ChainProgress) (ss : List SubSession)
    (s : SubSession) (hidx : s.index < max 1 s.count) (hret : s.retired = false)
    (hs : ChainedTrackedWaitOpen c s) :
    let r := recordTerminalWaitOpen c ss s
    r.1.present = c.present ∧
    (∀ t, ChainedTrackedWaitOpen c t → ChainedTrackedWaitOpen r.1 t) ∧
    (∀ k idx, r.2 = .done k idx → GoodEntry (k, idx)) ∧
    (∀ ok, r.2 = .direct ok ∨ r.2 = .stray ok → ok = true → GoodEntry (max 1 s.count, [s.index])) := by
  have hsingle : ∀ ok,
      (recordTerminalWaitOpen c ss s).2 = .direct ok ∨ (recordTerminalWaitOpen c ss s).2 = .stray ok →
      ok = true → GoodEntry (max 1 s.count, [s.index]) := by
    intro ok hok htrue
    subst htrue
    have hok' : s.ok = true := by
      unfold recordTerminalWaitOpen at hok
      rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
    have hcount : s.count ≤ 1 := by
      rcases Nat.lt_or_ge 1 s.count with hgt | hle
      · exfalso
        obtain ⟨hpres, hin⟩ := hs hgt hret hok'
        unfold recordTerminalWaitOpen at hok
        rw [hpres] at hok
        simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hok
        rcases hin with hin | hin | hfail | hcomp
        · rw [hin] at hok
          simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hok
          rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
        · rw [hin] at hok
          rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
        · rw [hfail] at hok
          rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
        · rw [hcomp] at hok
          rcases hok with hok | hok <;> (repeat' split at hok) <;> simp_all
      · exact hle
    have hidx0 : s.index = 0 := by
      have : max 1 s.count = 1 := by omega
      omega
    refine ⟨by simp, ?_, by simp only [List.length_singleton]; omega⟩
    intro i hi
    simp only [List.mem_singleton] at hi
    subst hi; omega
  refine ⟨?_, ?_, ?_, hsingle⟩
  all_goals unfold recordTerminalWaitOpen
  all_goals (try dsimp only)
  · repeat' split
    all_goals rfl
  · intro t ht hc hr ho
    obtain ⟨hp, hin⟩ := ht hc hr ho
    repeat' split
    all_goals first
      | exact ⟨hp, hin⟩
      | exact ⟨hp, Or.inr (Or.inr (Or.inl rfl))⟩
      | exact ⟨hp, Or.inr (Or.inr (Or.inr rfl))⟩
  · intro k idx h
    repeat' split at h
    all_goals (try cases h)
    rename_i hall
    have heq : (settledSlots ss c.subSessions).length = c.subSessions.length := by simpa using hall
    exact ⟨settledSlots_nodup _ _, settledSlots_lt _ _, Nat.le_of_eq heq.symm⟩

private theorem foldRecordWaitOpen_shape (ss : List SubSession) (settled : List SubSession) :
    ∀ (acc : WorkerState.SweepAcc),
      (∀ s ∈ settled, s.index < max 1 s.count ∧ s.retired = false) →
      (∀ s ∈ settled, ChainedTrackedWaitOpen acc.chain s) →
      (∀ t ∈ ss, ChainedTrackedWaitOpen acc.chain t) →
      (∀ e ∈ acc.log, GoodEntry e) →
      let r := settled.foldl (WorkerState.recordStepWaitOpen ss) acc
      r.chain.present = acc.chain.present ∧
      (∀ t ∈ ss, ChainedTrackedWaitOpen r.chain t) ∧ ∀ e ∈ r.log, GoodEntry e := by
  induction settled with
  | nil =>
    intro acc _ _ htr hlog
    exact ⟨rfl, htr, hlog⟩
  | cons s rest ih =>
    intro acc hs hsettled htr hlog
    obtain ⟨hidx, hret⟩ := hs s (List.mem_cons_self ..)
    have hrest : ∀ s' ∈ rest, s'.index < max 1 s'.count ∧ s'.retired = false :=
      fun s' h => hs s' (List.mem_cons_of_mem _ h)
    obtain ⟨hpres, hkeep, hgood, hsingle⟩ :=
      recordTerminalWaitOpen_shape acc.chain ss s hidx hret (hsettled s (List.mem_cons_self ..))
    simp only [List.foldl_cons]
    have hstep : ∀ acc' : WorkerState.SweepAcc, WorkerState.recordStepWaitOpen ss acc s = acc' →
        acc'.chain = (recordTerminalWaitOpen acc.chain ss s).1 ∧
        ∀ e ∈ acc'.log, GoodEntry e := by
      intro acc' h
      unfold WorkerState.recordStepWaitOpen at h
      rcases hrt : recordTerminalWaitOpen acc.chain ss s with ⟨c', t⟩
      rw [hrt] at h hgood hsingle
      have hlog' : ∀ (extra : List (Nat × List Nat)), (∀ e ∈ extra, GoodEntry e) →
          ∀ e ∈ acc.log ++ extra, GoodEntry e := by
        intro extra hextra e he
        rcases List.mem_append.mp he with he | he
        · exact hlog e he
        · exact hextra e he
      cases t with
      | done k idx =>
        subst h
        exact ⟨rfl, hlog' _ fun e he => by
          simp only [List.mem_singleton] at he; subst he; exact hgood k idx rfl⟩
      | direct ok =>
        cases ok
        · subst h; exact ⟨rfl, hlog⟩
        · subst h
          exact ⟨rfl, hlog' _ fun e he => by
            simp only [List.mem_singleton] at he; subst he; exact hsingle true (Or.inl rfl) rfl⟩
      | stray ok =>
        cases ok
        · subst h; exact ⟨rfl, hlog⟩
        · subst h
          exact ⟨rfl, hlog' _ fun e he => by
            simp only [List.mem_singleton] at he; subst he; exact hsingle true (Or.inr rfl) rfl⟩
      | _ => subst h; exact ⟨rfl, hlog⟩
    obtain ⟨hchain, hlog1⟩ := hstep _ rfl
    obtain ⟨hp', htr'', hlog''⟩ := ih _ hrest
      (fun s' hs' => by rw [hchain]; exact hkeep s' (hsettled s' (List.mem_cons_of_mem _ hs')))
      (fun t ht => by rw [hchain]; exact hkeep t (htr t ht)) hlog1
    exact ⟨by rw [hp', hchain, hpres], htr'', hlog''⟩

private theorem stepWaitOpen_registerSubPlan {g : WorkerState.Guards} {w w' : WorkerState}
    {uuid index count : Nat}
    (hg6 : g.noSingleWhileSeated = true) (hg7 : g.oneSessionPerSlot = true)
    (hinv : WorkerWaitOpenInv w)
    (hstep : WorkerState.stepG g w (.registerSubPlan uuid index count) = some w') :
    WorkerWaitOpenInv w' := by
  simp only [WorkerState.stepG, hg6, hg7] at hstep
  split at hstep; · cases hstep
  split at hstep; · cases hstep
  rename_i hidx
  split at hstep; · cases hstep
  rename_i hA6
  split at hstep; · cases hstep
  rename_i hA7
  injection hstep with h; subst h
  simp only [Bool.true_and, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq, bne_iff_ne,
    ne_eq, Bool.not_eq_true, not_and, Option.isSome_eq_false_iff, Option.isNone_iff_eq_none,
    List.any_eq_false] at hA6 hA7
  have hidx' : index < max 1 count := Nat.lt_of_not_ge hidx
  have hsessIdx : ∀ s ∈ w.sessions ++ [{ uuid := uuid, index := index, count := count }],
      s.index < max 1 s.count := by
    intro s hs
    rcases List.mem_append.mp hs with hs | hs
    · exact hinv.sessionIndex s hs
    · simp only [List.mem_singleton] at hs; subst hs; exact hidx'
  rcases Nat.lt_or_ge 1 count with hgt | hle
  · by_cases hi : index = 0
    · subst hi
      rw [note_anchor _ _ _ hgt]
      refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, ?_⟩
      intro s hs _ _ _
      refine ⟨rfl, ?_⟩
      rcases List.mem_append.mp hs with _ | hs
      · exact Or.inr (Or.inl rfl)
      · simp only [List.mem_singleton] at hs; subst hs
        exact Or.inl (contains_iff_getD.mpr ⟨0, getD_set_self _
          (Nat.lt_of_lt_of_le (by omega) (length_growTo _ _)) _⟩)
    · by_cases hp : w.chain.present = true
      · rw [note_into _ _ _ _ hgt hi hp]
        have hempty : w.chain.subSessions.getD index none = none := hA7 ⟨⟨hgt, hi⟩, hp⟩
        refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, ?_⟩
        intro s hs hc hr ho
        rcases List.mem_append.mp hs with hs | hs
        · obtain ⟨_, hin⟩ := hinv.tracked s hs hc hr ho
          refine ⟨hp, ?_⟩
          rcases hin with hin | hanch | hfail | hcomp
          · exact Or.inl (contains_set_of_empty hempty hin)
          · exact Or.inr (Or.inl hanch)
          · exact Or.inr (Or.inr (Or.inl hfail))
          · exact Or.inr (Or.inr (Or.inr hcomp))
        · simp only [List.mem_singleton] at hs; subst hs
          exact ⟨hp, Or.inl (contains_iff_getD.mpr ⟨index, getD_set_self _
            (Nat.lt_of_lt_of_le (by omega) (length_growTo _ _)) _⟩)⟩
      · have hp' : w.chain.present = false := by simpa using hp
        rw [note_unanchored _ _ _ _ hgt hi hp']
        refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, ?_⟩
        intro s hs hc hr ho
        rcases List.mem_append.mp hs with hs | hs
        · obtain ⟨hpres, _⟩ := hinv.tracked s hs hc hr ho
          rw [hpres] at hp'; cases hp'
        · simp only [List.mem_singleton] at hs; subst hs
          exact ⟨rfl, Or.inl (contains_iff_getD.mpr ⟨index, getD_set_self _
            (Nat.lt_of_lt_of_le (by omega) (length_growTo _ _)) _⟩)⟩
  · rw [note_single _ _ _ _ hle]
    refine ⟨⟨hinv.safe.doneAfterAll⟩, hsessIdx, ?_⟩
    intro s hs hc hr _
    rcases List.mem_append.mp hs with hs | hs
    · exact absurd hr (hA6 hle s hs hc)
    · simp only [List.mem_singleton] at hs; subst hs
      simp at hc; omega

private theorem pollWaitOpenCore_preserves {w w' : WorkerState} {stalled : Bool}
    {midSettle : List Nat}
    (hinv : WorkerWaitOpenInv w)
    (hstep : WorkerState.pollWaitOpenCore w stalled midSettle = some w') :
    WorkerWaitOpenInv w' := by
  simp only [WorkerState.pollWaitOpenCore] at hstep
  generalize hss : (if stalled = true then w.sessions.map applyDeadlineArm else w.sessions) = ss
    at hstep
  split at hstep; · cases hstep
  have hss_shape : ∀ t ∈ ss, ∃ t0 ∈ w.sessions,
      t.uuid = t0.uuid ∧ t.index = t0.index ∧ t.count = t0.count ∧ t.retired = t0.retired ∧
      (t.ok = true → t0.ok = true) := by
    intro t ht
    split at hss
    · subst hss
      obtain ⟨t0, ht0, rfl⟩ := List.mem_map.mp ht
      obtain ⟨hu, hi, hc, hr, ho, _⟩ := applyDeadlineArm_shape t0
      exact ⟨t0, ht0, hu, hi, hc, hr, ho⟩
    · subst hss; exact ⟨t, ht, rfl, rfl, rfl, rfl, fun h => h⟩
  have hidx : ∀ t ∈ ss, t.index < max 1 t.count := by
    intro t ht
    obtain ⟨t0, ht0, _, hi, hc, _, _⟩ := hss_shape t ht
    rw [hi, hc]; exact hinv.sessionIndex t0 ht0
  have htr : ∀ t ∈ ss, ChainedTrackedWaitOpen w.chain t := by
    intro t ht hc hr ho
    obtain ⟨t0, ht0, hu, _, hcnt, hret, hok⟩ := hss_shape t ht
    rw [hu]
    exact hinv.tracked t0 ht0 (hcnt ▸ hc) (hret ▸ hr) (hok ho)
  have hsettled : ∀ s ∈ ss.filter (fun s => s.done && !s.retired),
      s.index < max 1 s.count ∧ s.retired = false := by
    intro s hs
    have hm := List.mem_filter.mp hs
    refine ⟨hidx s hm.1, ?_⟩
    have := hm.2
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at this
    exact this.2
  have hrec : (WorkerState.recordAllWaitOpen w.chain ss
        (ss.filter (fun s => s.done && !s.retired))).chain.present = w.chain.present ∧
      (∀ t ∈ ss, ChainedTrackedWaitOpen
        (WorkerState.recordAllWaitOpen w.chain ss
          (ss.filter (fun s => s.done && !s.retired))).chain t) ∧
      ∀ e ∈ (WorkerState.recordAllWaitOpen w.chain ss
        (ss.filter (fun s => s.done && !s.retired))).log, GoodEntry e :=
    foldRecordWaitOpen_shape ss (ss.filter (fun s => s.done && !s.retired)) { chain := w.chain }
      hsettled (fun s hs => htr s (List.mem_filter.mp hs).1) htr (fun _ h => by cases h)
  obtain ⟨hpres, htr', hlog⟩ := hrec
  generalize hr : WorkerState.recordAllWaitOpen w.chain ss
      (ss.filter (fun s => s.done && !s.retired)) = r at hstep hpres htr' hlog
  generalize hssMid : ((ss.map fun s => if s.done then { s with retired := true } else s).map
      (WorkerState.applyMidSettle midSettle)) = ssMid at hstep
  generalize hr2 : (if stalled && !r.touched then
      sweepStalledChainWaitOpen r.chain (anyOpenSlot ssMid r.chain.subSessions)
        (ssMid.any fun s => !s.retired)
    else (r.chain, .notSwept)) = r2 at hstep
  rcases r2 with ⟨c2, sweep⟩
  injection hstep with h; subst h
  refine ⟨⟨fun e he => (List.mem_append.mp he).elim (hinv.safe.doneAfterAll e) (hlog e)⟩, ?_, ?_⟩
  · intro s'' hs''
    subst hssMid
    obtain ⟨s', hs', rfl⟩ := List.mem_map.mp hs''
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have := hidx s hs
    unfold WorkerState.applyMidSettle
    cases hd : s.done <;> cases hmc : midSettle.contains s.uuid <;> simp_all
  · intro s'' hs'' hc hret hok
    have hanyReq : (ssMid.any fun t => !t.retired) = true :=
      List.any_eq_true.mpr ⟨s'', hs'', by simp [hret]⟩
    have hm : s'' ∈ (ss.map fun s => if s.done then { s with retired := true } else s).map
        (WorkerState.applyMidSettle midSettle) := hssMid ▸ hs''
    obtain ⟨s', hs', hs''_eq⟩ := List.mem_map.mp hm
    obtain ⟨s, hs, hs'_eq⟩ := List.mem_map.mp hs'
    have hpreserve : s''.uuid = s.uuid ∧ s''.count = s.count ∧ s.retired = false ∧ s.ok = true := by
      subst hs''_eq; subst hs'_eq
      unfold WorkerState.applyMidSettle at hret hok ⊢
      cases hd : s.done <;> cases hmc : midSettle.contains s.uuid <;> simp_all
    obtain ⟨hu, hcnt, hsret, hsok⟩ := hpreserve
    obtain ⟨hrpres, hin⟩ := htr' s hs (hcnt ▸ hc) hsret hsok
    rw [hu]
    split at hr2
    · unfold sweepStalledChainWaitOpen at hr2
      rw [hrpres, hanyReq] at hr2
      simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hr2
      split at hr2
      · injection hr2 with h _; subst h; exact ⟨hrpres, hin⟩
      · split at hr2
        · injection hr2 with h _; subst h; exact ⟨hrpres, hin⟩
        · injection hr2 with h _; subst h
          exact ⟨rfl, Or.inr (Or.inr (Or.inl rfl))⟩
    · injection hr2 with h _; subst h
      exact ⟨hrpres, hin⟩

theorem stepSweepWaitsForOpen_preserves_inv {w w' : WorkerState} {e : WorkerEv}
    (hinv : WorkerWaitOpenInv w) (hstep : WorkerState.stepSweepWaitsForOpen w e = some w') :
    WorkerWaitOpenInv w' := by
  cases e with
  | registerSubPlan uuid index count =>
    dsimp only [WorkerState.stepSweepWaitsForOpen] at hstep
    exact stepWaitOpen_registerSubPlan rfl rfl hinv hstep
  | beginOp uuid =>
    simp only [WorkerState.stepSweepWaitsForOpen, WorkerState.stepG] at hstep
    split at hstep; · cases hstep
    injection hstep with h; subst h
    refine ⟨⟨hinv.safe.doneAfterAll⟩, ?_, ?_⟩
    · intro s' hs'; obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
      have := hinv.sessionIndex s hs; split <;> simpa using this
    · intro s' hs'; obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
      have ht := hinv.tracked s hs; split <;> exact ht
  | latchDrain uuid ok =>
    simp only [WorkerState.stepSweepWaitsForOpen, WorkerState.stepG] at hstep
    split at hstep; · cases hstep
    injection hstep with h; subst h
    refine ⟨⟨hinv.safe.doneAfterAll⟩, ?_, ?_⟩
    · intro s' hs'; obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
      have := hinv.sessionIndex s hs; split <;> simpa using this
    · intro s' hs'; obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
      have ht := hinv.tracked s hs
      split
      · intro hc hr ho; simp only [Bool.and_eq_true] at ho; exact ht hc hr ho.1
      · exact ht
  | settleSession uuid ok =>
    simp only [WorkerState.stepSweepWaitsForOpen, WorkerState.stepG] at hstep
    split at hstep; · cases hstep
    injection hstep with h; subst h
    refine ⟨⟨hinv.safe.doneAfterAll⟩, ?_, ?_⟩
    · intro s' hs'; obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
      have := hinv.sessionIndex s hs; split <;> simpa using this
    · intro s' hs'; obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
      have ht := hinv.tracked s hs
      split
      · intro hc hr ho; simp only [Bool.and_eq_true] at ho; exact ht hc hr ho.1
      · exact ht
  | poll stalled =>
    dsimp only [WorkerState.stepSweepWaitsForOpen] at hstep
    exact pollWaitOpenCore_preserves hinv hstep
  | pollMidSettle stalled midSettle =>
    dsimp only [WorkerState.stepSweepWaitsForOpen] at hstep
    split at hstep; · cases hstep
    exact pollWaitOpenCore_preserves hinv hstep

/-- **Alternative sweep theorem (`reachable_sweepWaitsForOpen_safe`)**: gating
tombstone erasure on `!anyReqUnretired` and checking `failed || completed` before
the `!anchored` stray arm (`stepSweepWaitsForOpen`) satisfies
`DoneAfterAllSubPlans` in every reachable state without assuming A5b
(`noDrainingOkAtSweep = false`) or A5c
(`noSettleBetweenRetireAndSweep = false`). -/
theorem reachable_sweepWaitsForOpen_safe {w : WorkerState}
    (hr : WorkerState.sysSweepWaitsForOpen.Reachable w) : WorkerSafe w :=
  (WorkerState.sysSweepWaitsForOpen.reachable_induction init_workerWaitOpenInv
    (fun _ _ _ hinv hstep => stepSweepWaitsForOpen_preserves_inv hinv hstep) hr).safe

/-! ## Part III — Coordinator multi-window chain -/

/-- Well-formed chain topology: at least one window, every window non-empty and
drawn from the claimed source snapshot (`BuildPoolReshardPlan` keeps a subset of
`request.src_units`, and a window with no spans is never planned). -/
structure ValidChainCfg (allSrcUnits : List Nat) (winSrcUnits : List (List Nat)) : Prop where
  winNonEmpty : winSrcUnits ≠ []
  winSub : ∀ ws ∈ winSrcUnits, ws ≠ [] ∧ ∀ u ∈ ws, u ∈ allSrcUnits

/-- **No dispatched sender without a claim (`NoSenderExposed`,
`ReshardCoordinator::RunSubPlan` `if (index == 0)`)**: a rank whose
`ReshardSendSession` was dispatched in some window and has not yet voted
`Complete` is never left without the generation's registry claim. -/
def NoSenderExposed (s : CoordChainState) : Prop := s.activeSenderExposed = false

/-- **No push before all armed (`NoPushBeforeAllArmed`, `RunSubPlan`)**: while a
window is arming, none of its senders has been dispatched. -/
def NoPushBeforeAllArmed (s : CoordChainState) : Prop :=
  s.winPhase = .arming → s.sendResults = []

/-- **All armed while dispatching (`AllArmedWhileDispatching`, `RunSubPlan`)**:
while a window dispatches senders, every destination's receiver for that
window is armed. -/
def AllArmedWhileDispatching (s : CoordChainState) : Prop :=
  s.winPhase = .dispatching → ∀ d ∈ s.dstStates, d.curWinArmed = true

/-- **One window staged per destination (`OneWindowStagedPerDestination`,
`ArmRefusedForBusyStaging`)**: a destination whose earlier window's receiver
still holds its staging lease never has the current window's receiver armed. -/
def OneWindowStagedPerDestination (s : CoordChainState) : Prop :=
  ∀ d ∈ s.dstStates, d.prevSessionActive = true → d.curWinArmed = false

/-- **`kCompleted` only at chain end (`CompletedOnlyAtChainEnd`, `StartChain`)**:
`transfer_status_` reads `kCompleted` only after the last window's Stage 2. -/
def CompletedOnlyAtChainEnd (s : CoordChainState) : Prop :=
  s.transferStatus = .completed → s.winIdx + 1 = s.winSrcUnits.length ∧ s.winPhase = .succeeded

/-- **Claim held across windows (`ClaimHeldAcrossWindows`, `RunSubPlan`
`LookupAndClaim`)**: from window 0's Stage 0 until the chain ends, the source
snapshot stays claimed by `ctx->claim_owner`, so `CancelIfUnclaimed` cannot
cancel it between windows. -/
def ClaimHeldAcrossWindows (s : CoordChainState) : Prop :=
  s.transferStatus = .inProgress → (s.winPhase ≠ .idle ∨ 0 < s.winIdx) →
    s.reg.claimedUnits = some s.allSrcUnits ∧ s.reg.claimOwners = [s.claimOwner]

structure CoordChainSafe (s : CoordChainState) : Prop where
  regInv : Registry.Inv s.reg
  noSenderExposed : NoSenderExposed s
  noPushBeforeAllArmed : NoPushBeforeAllArmed s
  allArmedWhileDispatching : AllArmedWhileDispatching s
  oneWindowStaged : OneWindowStagedPerDestination s
  completedOnlyAtChainEnd : CompletedOnlyAtChainEnd s
  claimHeldAcrossWindows : ClaimHeldAcrossWindows s

def RegIdle (s : CoordChainState) : Prop :=
  s.reg.claimedUnits = none ∧ s.reg.claimOwners = [] ∧ s.reg.completedUnits = []

def ClaimedBy (s : CoordChainState) : Prop :=
  s.reg.claimedUnits = some s.allSrcUnits ∧ s.reg.claimOwners = [s.claimOwner]

def ClaimReleased (s : CoordChainState) : Prop :=
  s.reg.claimedUnits = none ∧ s.reg.claimOwners = []

/-- No rank of the running or a later window has voted `Complete`. -/
def RemainingUnvoted (s : CoordChainState) : Prop :=
  ∀ j, s.winIdx ≤ j → ∀ w ∈ s.winSrcUnits.getD j [], w ∉ s.reg.completedUnits

/-- Per-`winPhase` shape of destination arming and Stage 2 sender results. -/
def WinPhaseInv (s : CoordChainState) : Prop :=
  match s.winPhase with
  | .idle | .succeeded => ∀ d ∈ s.dstStates, d.curWinArmed = false
  | .arming => s.sendResults = [] ∧ ∀ d ∈ s.dstStates, d.curWinArmed = armOk s.armResults d.unit
  | .dispatching => ∀ d ∈ s.dstStates, d.curWinArmed = true
  | .aborted => True

/-- Per-`transferStatus` shape of window progress and the registry claim. -/
def PhaseShape (s : CoordChainState) : Prop :=
  match s.transferStatus with
  | .notStarted =>
    s.winIdx = 0 ∧ s.winPhase = .idle ∧ s.activeSenders = [] ∧ RegIdle s
  | .inProgress =>
    (s.winIdx = 0 ∧ s.winPhase = .idle ∧ s.activeSenders = [] ∧ RegIdle s) ∨
    (s.winIdx < s.winSrcUnits.length ∧
      (s.winIdx = 0 → s.winPhase ≠ .idle ∧ (s.winPhase = .arming → s.activeSenders = [])) ∧
      ClaimedBy s ∧ RemainingUnvoted s)
  | .completed =>
    s.winIdx + 1 = s.winSrcUnits.length ∧ s.winPhase = .succeeded ∧
    (ClaimedBy s ∨ ClaimReleased s)
  | .failed =>
    ClaimedBy s ∨ ClaimReleased s

/-- Inductive invariant of `CoordChainState.step`. -/
structure CoordChainInv (s : CoordChainState) : Prop where
  cfg : ValidChainCfg s.allSrcUnits s.winSrcUnits
  dstShape : s.dstStates.map (·.unit) = s.dstUnits
  regInv : Registry.Inv s.reg
  noExposed : s.activeSenderExposed = false
  activeSrc : ∀ u ∈ s.activeSenders, u ∈ s.allSrcUnits ∧ u ∉ s.reg.completedUnits
  activeClaimed : s.activeSenders ≠ [] → ClaimedBy s
  oneLease : ∀ d ∈ s.dstStates, d.prevSessionActive = true → d.curWinArmed = false
  winShape : WinPhaseInv s
  phase : PhaseShape s

theorem CoordChainInv.toSafe {s : CoordChainState} (h : CoordChainInv s) : CoordChainSafe s where
  regInv := h.regInv
  noSenderExposed := h.noExposed
  noPushBeforeAllArmed := fun hp => by
    have hw := h.winShape; simp only [WinPhaseInv, hp] at hw; exact hw.1
  allArmedWhileDispatching := fun hp => by
    have hw := h.winShape; simp only [WinPhaseInv, hp] at hw; exact hw
  oneWindowStaged := h.oneLease
  completedOnlyAtChainEnd := fun hc => by
    have hp := h.phase; simp only [PhaseShape, hc] at hp; exact ⟨hp.1, hp.2.1⟩
  claimHeldAcrossWindows := by
    intro hst hrun
    have hp := h.phase; simp only [PhaseShape, hst] at hp
    rcases hp with ⟨hi, hidle, _⟩ | ⟨_, _, hcl, _⟩
    · rcases hrun with hne | hpos
      · exact absurd hidle hne
      · omega
    · exact hcl

/-! ### Registry, list, and abort helpers -/

private theorem lookupAndClaim_when_claimed {reg reg' : RegGen} {owner : Nat} {units : List Nat}
    (hinv : Registry.Inv reg) (hc : reg.claimedUnits = some units) (ho : reg.claimOwners = [owner])
    (hstep : RegGen.step reg (.lookupAndClaim owner units) = some reg') :
    reg'.claimedUnits = some units ∧ reg'.claimOwners = [owner] ∧
    reg'.completedUnits = reg.completedUnits := by
  dsimp only [RegGen.step] at hstep
  split at hstep; · cases hstep
  split at hstep; · cases hstep
  split at hstep; · cases hstep
  have hall : allVoted units reg.completedUnits = false := hinv.partialVotes units hc
  simp only [hall, Bool.false_eq_true, ↓reduceIte, Option.some.injEq] at hstep
  subst hstep
  simp [ho, insertNodup]

private theorem complete_claimed {reg : RegGen} {units : List Nat} {owner u : Nat}
    (hc : reg.claimedUnits = some units) (ho : reg.claimOwners = [owner]) (hu : u ∈ units) :
    RegGen.step reg (.complete u) =
      some (if allVoted units (insertNodup u reg.completedUnits) then reg.retireLocked
            else { reg with completedUnits := insertNodup u reg.completedUnits }) := by
  have hin : inClaimedUnits (some units) u = true := by simp [inClaimedUnits, hu]
  have _ := ho
  unfold RegGen.step
  simp only [hc, hin, Option.isSome_some, Bool.not_true, Bool.false_and, Bool.false_eq_true,
    ↓reduceIte, Bool.and_false]
  split <;> rfl

private theorem cancelIfUnclaimed_shape {reg reg' : RegGen}
    (hstep : RegGen.step reg .cancelIfUnclaimed = some reg') :
    reg' = reg ∨
      (reg.claimedUnits = none ∧
       reg'.claimedUnits = none ∧ reg'.claimOwners = [] ∧ reg'.completedUnits = []) := by
  dsimp only [RegGen.step] at hstep
  split at hstep; · injection hstep with h; exact Or.inl h.symm
  split at hstep; · injection hstep with h; exact Or.inl h.symm
  rename_i hns
  injection hstep with h; subst h
  refine Or.inr ⟨?_, rfl, rfl, rfl⟩
  cases hcl : reg.claimedUnits with
  | none => rfl
  | some _ => rw [hcl] at hns; simp at hns

private theorem armOk_append (rs : List (Nat × ArmOutcome)) (u : Nat) (o : ArmOutcome) (v : Nat) :
    armOk (rs ++ [(u, o)]) v = (armOk rs v || (u == v && o == .ok)) := by
  simp [armOk, List.any_append]

private theorem armOk_filter_ok (rs : List (Nat × ArmOutcome)) (v : Nat) :
    armOk (rs.filter (·.2 == .ok)) v = armOk rs v := by
  induction rs with
  | nil => rfl
  | cons r rest ih =>
    obtain ⟨ru, ro⟩ := r
    simp only [armOk, List.filter_cons] at ih ⊢
    split <;> simp_all [List.any_cons]

private theorem armOk_of_reported_no_fail {rs : List (Nat × ArmOutcome)} {v : Nat}
    (hrep : armReported rs v = true) (hok : anyArmFailed rs = false) : armOk rs v = true := by
  simp only [armReported, List.any_eq_true] at hrep
  simp only [anyArmFailed, List.any_eq_false] at hok
  obtain ⟨r, hr, hrv⟩ := hrep
  have := hok r hr
  simp only [armOk, List.any_eq_true]
  exact ⟨r, hr, by simp_all⟩

private theorem getD_mem_of_lt {l : List (List Nat)} {j : Nat} (h : j < l.length) :
    l.getD j [] ∈ l := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  exact List.getElem_mem h

private theorem getD_eq_nil_of_le {l : List (List Nat)} {j : Nat} (h : l.length ≤ j) :
    l.getD j [] = [] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none h]
  rfl

private theorem not_in_remaining {s : CoordChainState} {u : Nat}
    (hst : s.transferStatus = .inProgress) (h : s.appearsInRemainingWindows u = false) :
    ∀ j, s.winIdx ≤ j → u ∉ s.winSrcUnits.getD j [] := by
  intro j hj hmem
  unfold CoordChainState.appearsInRemainingWindows at h
  rw [hst] at h
  simp only [beq_self_eq_true, Bool.true_and, List.any_eq_false, List.mem_range] at h
  by_cases hlt : j < s.winSrcUnits.length
  · have := h j hlt
    simp only [hj, decide_true, Bool.true_and, List.contains_iff_mem] at this
    exact this hmem
  · rw [getD_eq_nil_of_le (Nat.le_of_not_lt hlt)] at hmem
    cases hmem

private theorem map_unit_preserved (ds : List ChainDstState) (f : ChainDstState → ChainDstState)
    (hf : ∀ d, (f d).unit = d.unit) : (ds.map f).map (·.unit) = ds.map (·.unit) := by
  rw [List.map_map]
  exact List.map_congr_left fun d _ => hf d

private theorem length_pos_of_ne_nil {l : List (List Nat)} (h : l ≠ []) : 0 < l.length := by
  cases l with
  | nil => exact absurd rfl h
  | cons _ _ => simp

/-- `syncSenderExposure` is the identity when no sender is active or the
generation is claimed. -/
private theorem syncSenderExposure_id {s : CoordChainState}
    (h : s.activeSenders = [] ∨ s.reg.claimedUnits.isSome = true) :
    CoordChainState.syncSenderExposure s = s := by
  unfold CoordChainState.syncSenderExposure
  rcases h with h | h <;> simp [h]

private theorem inv_syncSenderExposure {s : CoordChainState}
    (h : s.activeSenders = [] ∨ s.reg.claimedUnits.isSome = true) (hinv : CoordChainInv s) :
    CoordChainInv s.syncSenderExposure := by
  rw [syncSenderExposure_id h]; exact hinv

private theorem CoordChainInv.inProgress_active {s : CoordChainState} (h : CoordChainInv s)
    (hst : s.transferStatus = .inProgress) (hne : s.winPhase ≠ .idle) :
    s.winIdx < s.winSrcUnits.length ∧
    (s.winIdx = 0 → s.winPhase = .arming → s.activeSenders = []) ∧
    ClaimedBy s ∧ RemainingUnvoted s := by
  have hp := h.phase; simp only [PhaseShape, hst] at hp
  rcases hp with ⟨_, hidle, _⟩ | ⟨hl, hz, hc, hrem⟩
  · exact absurd hidle hne
  · exact ⟨hl, fun hw0 => (hz hw0).2, hc, hrem⟩

/-- Shared preservation for `abortWindow` across Stage 0, Stage 1, and Stage 2. -/
private theorem inv_abortWindow {stage : Nat} {s : CoordChainState}
    (hcfg : ValidChainCfg s.allSrcUnits s.winSrcUnits)
    (hdst : s.dstStates.map (·.unit) = s.dstUnits)
    (hreg : Registry.Inv s.reg)
    (hexp : s.activeSenderExposed = false)
    (hasrc : ∀ u ∈ s.activeSenders, u ∈ s.allSrcUnits ∧ u ∉ s.reg.completedUnits)
    (hacl : s.activeSenders ≠ [] → ClaimedBy s)
    (hlease : ∀ d ∈ s.dstStates, d.prevSessionActive = true → d.curWinArmed = false)
    (hab0 : stage ≤ 1 → s.winIdx = 0 → s.activeSenders = [] ∧ (ClaimedBy s ∨ ClaimReleased s))
    (hab1 : ¬(stage ≤ 1 ∧ s.winIdx = 0) → ClaimedBy s) :
    CoordChainInv (s.abortWindow stage) := by
  unfold CoordChainState.abortWindow
  by_cases hcond : stage ≤ 1 ∧ s.winIdx = 0
  · obtain ⟨hact, hsh⟩ := hab0 hcond.1 hcond.2
    have hcl := abandonIfClaimed_clears_single_owner s.reg s.allSrcUnits s.claimOwner hsh
    have hif : (decide (stage ≤ 1) && (s.winIdx == 0)) = true := by simp [hcond.1, hcond.2]
    simp only [hif, ↓reduceIte]
    exact inv_syncSenderExposure (Or.inl hact)
      ⟨hcfg, hdst, abandonIfClaimed_inv _ _ hreg, hexp,
       (fun _ hv => by simp [hact] at hv), (fun h => absurd hact h),
       hlease, trivial, Or.inr hcl⟩
  · have hcl : ClaimedBy s := hab1 hcond
    have hif : (decide (stage ≤ 1) && (s.winIdx == 0)) = false := by
      simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not, beq_eq_false_iff_ne]
      omega
    simp only [hif, Bool.false_eq_true, ↓reduceIte]
    exact inv_syncSenderExposure (Or.inr (by simp [hcl.1]))
      ⟨hcfg, hdst, hreg, hexp, hasrc, hacl, hlease, trivial, Or.inl hcl⟩

private theorem CoordChainInv.abort {stage : Nat} {s : CoordChainState} (h : CoordChainInv s)
    (hab0 : stage ≤ 1 → s.winIdx = 0 → s.activeSenders = [] ∧ (ClaimedBy s ∨ ClaimReleased s))
    (hab1 : ¬(stage ≤ 1 ∧ s.winIdx = 0) → ClaimedBy s) :
    CoordChainInv (s.abortWindow stage) :=
  inv_abortWindow h.cfg h.dstShape h.regInv h.noExposed h.activeSrc h.activeClaimed h.oneLease
    hab0 hab1

/-! ### Initial state and per-event preservation -/

theorem init_coordChainInv {allSrcs : List Nat} {winSrcs : List (List Nat)} {dsts : List Nat}
    {owner maxAtt : Nat} (hcfg : ValidChainCfg allSrcs winSrcs) :
    CoordChainInv (CoordChainState.init allSrcs winSrcs dsts owner maxAtt) where
  cfg := hcfg
  dstShape := by simp [CoordChainState.init, List.map_map, Function.comp_def]
  regInv := Registry.init_inv
  noExposed := rfl
  activeSrc := fun _ h => by cases h
  activeClaimed := fun h => absurd rfl h
  oneLease := by
    intro d hd
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp hd
    intro h; cases h
  winShape := by
    simp only [WinPhaseInv, CoordChainState.init]
    intro d hd
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp hd
    rfl
  phase := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

private theorem step_registerProducer {s s' : CoordChainState} {u : Nat}
    (h : CoordChainInv s) (hstep : CoordChainState.step s (.registerProducer u) = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true', not_or,
    Decidable.not_not] at hg
  cases hr : RegGen.step s.reg (.register u true) with
  | none => rw [hr] at hstep; cases hstep
  | some reg' =>
    rw [hr] at hstep
    simp only [Option.map_some, Option.some.injEq] at hstep
    subst hstep
    have hp := h.phase; simp only [PhaseShape, hg.1] at hp
    obtain ⟨hi, hidle, hact, hc, ho, hcomp⟩ := hp
    obtain ⟨hc', ho', hcomp'⟩ := register_preserves_idle_reg s.reg reg' u hc ho hcomp hr
    refine ⟨h.cfg, h.dstShape, Registry.step_preserves_inv h.regInv hr, h.noExposed,
      (fun _ hv => by simp [hact] at hv), (fun hne => absurd hact hne),
      h.oneLease, h.winShape, ?_⟩
    simp only [PhaseShape, hg.1]
    exact ⟨hi, hidle, hact, hc', ho', hcomp'⟩

private theorem step_startChainCoordination {s s' : CoordChainState}
    (h : CoordChainInv s) (hstep : CoordChainState.step s .startChainCoordination = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hg
  injection hstep with hs; subst hs
  have hp := h.phase; simp only [PhaseShape, hg] at hp
  exact ⟨h.cfg, h.dstShape, h.regInv, h.noExposed, h.activeSrc, h.activeClaimed,
    h.oneLease, h.winShape, Or.inl hp⟩

private theorem step_startWindowPlan {s s' : CoordChainState} {planOk : Bool}
    (h : CoordChainInv s) (hstep : CoordChainState.step s (.startWindowPlan planOk) = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hg
  obtain ⟨hst, hidle⟩ := hg
  have hp := h.phase; simp only [PhaseShape, hst] at hp
  have hw := h.winShape; simp only [WinPhaseInv, hidle] at hw
  cases hr : RegGen.step s.reg (.lookupAndClaim s.claimOwner s.allSrcUnits) with
  | none =>
    simp only [hr, Option.some.injEq] at hstep; subst hstep
    refine h.abort (fun _ hw0 => ?_) (fun hnot => ?_)
    · rcases hp with ⟨_, _, hact, hc, ho, _⟩ | ⟨_, hz, _⟩
      · exact ⟨hact, Or.inr ⟨hc, ho⟩⟩
      · exact absurd hidle (hz hw0).1
    · rcases hp with ⟨hi, _⟩ | ⟨_, _, hc, _⟩
      · exact absurd ⟨Nat.zero_le 1, hi⟩ hnot
      · exact hc
  | some regClaimed =>
    simp only [hr] at hstep
    have hreg' : Registry.Inv regClaimed := Registry.step_preserves_inv h.regInv hr
    have hclaimed : regClaimed.claimedUnits = some s.allSrcUnits ∧
        regClaimed.claimOwners = [s.claimOwner] ∧
        s.winIdx < s.winSrcUnits.length ∧
        (∀ j, s.winIdx ≤ j → ∀ w ∈ s.winSrcUnits.getD j [], w ∉ regClaimed.completedUnits) ∧
        (∀ u ∈ s.activeSenders, u ∈ s.allSrcUnits ∧ u ∉ regClaimed.completedUnits) ∧
        (s.winIdx = 0 → s.activeSenders = []) := by
      rcases hp with ⟨hi, _, hact, _, ho, hcomp⟩ | ⟨hl, hz, hc, hrem⟩
      · obtain ⟨hc', ho', hcomp'⟩ := lookupAndClaim_from_idle_shape s.reg regClaimed
          s.claimOwner s.allSrcUnits ho hcomp hr
        refine ⟨hc', ho', hi ▸ length_pos_of_ne_nil h.cfg.winNonEmpty, ?_, ?_, fun _ => hact⟩
        · intro _ _ _ _; rw [hcomp']; intro hm; cases hm
        · intro u hu; rw [hact] at hu; cases hu
      · obtain ⟨hc', ho', hcomp'⟩ := lookupAndClaim_when_claimed h.regInv hc.1 hc.2 hr
        refine ⟨hc', ho', hl, ?_, ?_, fun hw0 => absurd hidle (hz hw0).1⟩
        · intro j hj w hw; rw [hcomp']; exact hrem j hj w hw
        · intro u hu; rw [hcomp']; exact h.activeSrc u hu
    obtain ⟨hc', ho', hl, hrem', hasrc', hact0⟩ := hclaimed
    split at hstep
    · injection hstep with hs; subst hs
      refine inv_syncSenderExposure (Or.inr (by simp [hc'])) ?_
      refine ⟨h.cfg, h.dstShape, hreg', h.noExposed, hasrc', fun _ => ⟨hc', ho'⟩, h.oneLease,
        ⟨rfl, fun d hd => by rw [hw d hd]; rfl⟩, ?_⟩
      simp only [PhaseShape, hst]
      exact Or.inr ⟨hl, fun hw0 => ⟨by decide, fun _ => hact0 hw0⟩, ⟨hc', ho'⟩, hrem'⟩
    · injection hstep with hs; subst hs
      exact inv_abortWindow h.cfg h.dstShape hreg' h.noExposed hasrc' (fun _ => ⟨hc', ho'⟩)
        h.oneLease (fun _ hw0 => ⟨hact0 hw0, Or.inl ⟨hc', ho'⟩⟩) (fun _ => ⟨hc', ho'⟩)

private theorem step_armDstResult {s s' : CoordChainState} {dstUnit : Nat} {outcome : ArmOutcome}
    (h : CoordChainInv s)
    (hstep : CoordChainState.step s (.armDstResult dstUnit outcome) = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hg
  split at hstep; · cases hstep
  split at hstep; · cases hstep
  rename_i hadmit
  injection hstep with hs; subst hs
  have hw := h.winShape; simp only [WinPhaseInv, hg.2] at hw
  have hbusy_false : outcome = .ok →
      ∀ d ∈ s.dstStates, d.unit = dstUnit → d.prevSessionActive = false := by
    intro hok d hd hu
    simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', not_or,
      not_and] at hadmit
    have := hadmit.1 hok
    simp only [Bool.not_eq_true, List.any_eq_false, Bool.and_eq_true, beq_iff_eq, not_and] at this
    exact this d hd hu
  refine ⟨h.cfg, ?_, h.regInv, h.noExposed, h.activeSrc, h.activeClaimed, ?_, ?_, h.phase⟩
  · rw [map_unit_preserved _ _ (fun _ => by split <;> rfl)]; exact h.dstShape
  · intro d' hd'
    obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'
    split
    · rename_i hcond
      simp only [Bool.and_eq_true, beq_iff_eq] at hcond
      intro hprev
      exact absurd hprev (by rw [hbusy_false hcond.2 d hd hcond.1]; decide)
    · exact h.oneLease d hd
  · simp only [WinPhaseInv, hg.2]
    refine ⟨hw.1, fun d' hd' => ?_⟩
    obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'
    have hd_arm := hw.2 d hd
    split
    · rename_i hcond
      simp only [Bool.and_eq_true, beq_iff_eq] at hcond
      simp [armOk_append, hcond.1, hcond.2]
    · rename_i hcond
      simp only [Bool.and_eq_true, beq_iff_eq, not_and] at hcond
      simp only [armOk_append, hd_arm]
      by_cases hu : dstUnit = d.unit
      · simp [hu, beq_false_of_ne (fun hok => hcond hu.symm hok)]
      · simp [beq_false_of_ne hu]

private theorem step_retryBusyArm {s s' : CoordChainState}
    (h : CoordChainInv s) (hstep : CoordChainState.step s .retryBusyArm = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hg
  split at hstep; · cases hstep
  injection hstep with hs; subst hs
  have hw := h.winShape; simp only [WinPhaseInv, hg.2] at hw
  refine ⟨h.cfg, h.dstShape, h.regInv, h.noExposed, h.activeSrc, h.activeClaimed, h.oneLease,
    ?_, h.phase⟩
  simp only [WinPhaseInv, hg.2]
  exact ⟨hw.1, fun d hd => by rw [armOk_filter_ok]; exact hw.2 d hd⟩

private theorem step_finishWindowArm {s s' : CoordChainState}
    (h : CoordChainInv s) (hstep : CoordChainState.step s .finishWindowArm = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hg
  obtain ⟨hst, harming⟩ := hg
  split at hstep; · cases hstep
  rename_i hg2
  simp only [Bool.or_eq_true, Bool.not_eq_true', not_or] at hg2
  obtain ⟨hallrep, _⟩ := hg2
  simp only [Bool.not_eq_false] at hallrep
  obtain ⟨hl, hz, hc, hrem⟩ := h.inProgress_active hst (by simp [harming])
  have hw := h.winShape; simp only [WinPhaseInv, harming] at hw
  split at hstep
  · rename_i hnofail
    simp only [Bool.not_eq_true'] at hnofail
    injection hstep with hs; subst hs
    refine ⟨h.cfg, h.dstShape, h.regInv, h.noExposed, h.activeSrc, h.activeClaimed, h.oneLease,
      ?_, ?_⟩
    · show ∀ d ∈ s.dstStates, d.curWinArmed = true
      intro d hd
      rw [hw.2 d hd]
      simp only [allArmReported, List.all_eq_true] at hallrep
      exact armOk_of_reported_no_fail
        (hallrep d.unit (h.dstShape ▸ List.mem_map.mpr ⟨d, hd, rfl⟩)) hnofail
    · simp only [PhaseShape, hst]
      exact Or.inr ⟨hl, fun _ => ⟨by decide, fun hp => by cases hp⟩, hc, hrem⟩
  · injection hstep with hs; subst hs
    exact h.abort (fun _ hw0 => ⟨hz hw0 harming, Or.inl hc⟩) (fun _ => hc)

private theorem step_sendSrcResult {s s' : CoordChainState} {srcUnit : Nat} {ok : Bool}
    (h : CoordChainInv s) (hstep : CoordChainState.step s (.sendSrcResult srcUnit ok) = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hg
  obtain ⟨hst, hdisp⟩ := hg
  split at hstep; · cases hstep
  rename_i hg2
  simp only [Bool.or_eq_true, Bool.not_eq_true', not_or] at hg2
  have hmem : srcUnit ∈ s.winSrcUnits.getD s.winIdx [] := by
    simpa [CoordChainState.curWindowSrcs] using hg2.1
  injection hstep with hs; subst hs
  obtain ⟨hl, _, hc, hrem⟩ := h.inProgress_active hst (by simp [hdisp])
  have hw := h.winShape; simp only [WinPhaseInv, hdisp] at hw
  have hsrc : srcUnit ∈ s.allSrcUnits ∧ srcUnit ∉ s.reg.completedUnits :=
    ⟨(h.cfg.winSub _ (getD_mem_of_lt hl)).2 srcUnit hmem, hrem s.winIdx (Nat.le_refl _) srcUnit hmem⟩
  refine inv_syncSenderExposure (Or.inr (by simp [hc.1])) ?_
  refine ⟨h.cfg, h.dstShape, h.regInv, h.noExposed, ?_, fun _ => hc, h.oneLease, ?_, ?_⟩
  · intro u hu
    split at hu
    · rcases mem_insertNodup.mp hu with rfl | hu
      · exact hsrc
      · exact h.activeSrc u hu
    · exact h.activeSrc u hu
  · simp only [WinPhaseInv, hdisp]; exact hw
  · simp only [PhaseShape, hst]
    exact Or.inr ⟨hl, fun _ => ⟨by simp [hdisp], fun hp => by rw [hdisp] at hp; cases hp⟩, hc, hrem⟩

private theorem step_finishWindowSend {s s' : CoordChainState}
    (h : CoordChainInv s) (hstep : CoordChainState.step s .finishWindowSend = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hg
  obtain ⟨hst, hdisp⟩ := hg
  split at hstep; · cases hstep
  obtain ⟨hl, _, hc, hrem⟩ := h.inProgress_active hst (by simp [hdisp])
  split at hstep
  · injection hstep with hs; subst hs
    exact h.abort (fun hs _ => by omega) (fun _ => hc)
  · let advanceDst : ChainDstState → ChainDstState := fun d =>
      { d with prevSessionActive := d.prevSessionActive || d.curWinArmed, curWinArmed := false }
    have hdst' : (s.dstStates.map advanceDst).map (·.unit) = s.dstUnits := by
      rw [map_unit_preserved s.dstStates advanceDst (fun _ => rfl)]; exact h.dstShape
    have hlease' : ∀ d ∈ s.dstStates.map advanceDst,
        d.prevSessionActive = true → d.curWinArmed = false := by
      intro d' hd'; obtain ⟨_, _, rfl⟩ := List.mem_map.mp hd'; exact fun _ => rfl
    have hcur' : ∀ d ∈ s.dstStates.map advanceDst, d.curWinArmed = false := by
      intro d' hd'; obtain ⟨_, _, rfl⟩ := List.mem_map.mp hd'; rfl
    split at hstep
    · rename_i hmore
      injection hstep with hs; subst hs
      refine ⟨h.cfg, hdst', h.regInv, h.noExposed, h.activeSrc, h.activeClaimed, hlease', hcur', ?_⟩
      simp only [PhaseShape, hst]
      exact Or.inr ⟨hmore, fun hw => by omega, hc, fun j hj => hrem j (Nat.le_of_succ_le hj)⟩
    · rename_i hlast
      injection hstep with hs; subst hs
      refine ⟨h.cfg, hdst', h.regInv, h.noExposed, h.activeSrc, h.activeClaimed, hlease', hcur', ?_⟩
      show s.winIdx + 1 = s.winSrcUnits.length ∧ _ ∧ _
      exact ⟨by omega, rfl, Or.inl hc⟩

private theorem step_settleDispatchedRecv {s s' : CoordChainState} {dstUnit : Nat}
    (h : CoordChainInv s)
    (hstep : CoordChainState.step s (.settleDispatchedRecv dstUnit) = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  injection hstep with hs; subst hs
  have hunit : ∀ d : ChainDstState,
      (if d.unit == dstUnit then { d with prevSessionActive := false } else d).unit = d.unit :=
    fun _ => by split <;> rfl
  have hcur : ∀ d : ChainDstState,
      (if d.unit == dstUnit then { d with prevSessionActive := false } else d).curWinArmed =
        d.curWinArmed :=
    fun _ => by split <;> rfl
  refine ⟨h.cfg, map_unit_preserved s.dstStates _ hunit ▸ h.dstShape, h.regInv,
    h.noExposed, h.activeSrc, h.activeClaimed, ?_, ?_, h.phase⟩
  · intro d' hd'
    obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'
    split
    · intro hp; cases hp
    · exact h.oneLease d hd
  · have hw := h.winShape
    cases hp : s.winPhase <;> simp only [WinPhaseInv, hp] at hw ⊢
    · intro d' hd'; obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'; rw [hcur]; exact hw d hd
    · refine ⟨hw.1, fun d' hd' => ?_⟩
      obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'; rw [hcur, hunit]; exact hw.2 d hd
    · intro d' hd'; obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'; rw [hcur]; exact hw d hd
    · intro d' hd'; obtain ⟨d, hd, rfl⟩ := List.mem_map.mp hd'; rw [hcur]; exact hw d hd

private theorem step_senderVoteComplete {s s' : CoordChainState} {srcUnit : Nat}
    (h : CoordChainInv s) (hstep : CoordChainState.step s (.senderVoteComplete srcUnit) = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  split at hstep; · cases hstep
  rename_i hg
  simp only [Bool.or_eq_true, Bool.not_eq_true', not_or] at hg
  obtain ⟨hactive, hnorem⟩ := hg
  have hactive' : srcUnit ∈ s.activeSenders := List.contains_iff_mem.mp (by simpa using hactive)
  have hne : s.activeSenders ≠ [] := fun hn => by rw [hn] at hactive'; cases hactive'
  have hc := h.activeClaimed hne
  obtain ⟨hsrc_in, _⟩ := h.activeSrc srcUnit hactive'
  rw [complete_claimed hc.1 hc.2 hsrc_in] at hstep
  simp only [Option.map_some, Option.some.injEq] at hstep; subst hstep
  split
  · rename_i hall
    have hempty : s.activeSenders.filter (· != srcUnit) = [] := by
      refine List.eq_nil_iff_forall_not_mem.mpr fun v hv => ?_
      obtain ⟨hv_act, hv_ne⟩ := List.mem_filter.mp hv
      simp only [bne_iff_ne, ne_eq] at hv_ne
      obtain ⟨hv_in, hv_unvoted⟩ := h.activeSrc v hv_act
      simp only [allVoted, List.all_eq_true] at hall
      rcases mem_insertNodup.mp (List.contains_iff_mem.mp (hall v hv_in)) with rfl | hm
      · exact hv_ne rfl
      · exact hv_unvoted hm
    rw [hempty]
    refine inv_syncSenderExposure (Or.inl rfl)
      ⟨h.cfg, h.dstShape, Registry.retireLocked_inv s.reg, h.noExposed,
       (fun _ hm => by cases hm), (fun hn => absurd rfl hn), h.oneLease, h.winShape, ?_⟩
    have hp := h.phase
    cases hst : s.transferStatus <;> simp only [PhaseShape, hst] at hp ⊢
    · exact absurd hp.2.2.1 hne
    · rcases hp with ⟨_, _, hact, _⟩ | ⟨hl, _, _, hrem⟩
      · exact absurd hact hne
      · exfalso
        obtain ⟨hwne, hwsub⟩ := h.cfg.winSub _ (getD_mem_of_lt hl)
        obtain ⟨w, hw⟩ := List.exists_mem_of_ne_nil _ hwne
        simp only [allVoted, List.all_eq_true] at hall
        rcases mem_insertNodup.mp (List.contains_iff_mem.mp (hall w (hwsub w hw))) with rfl | hm
        · exact not_in_remaining hst (Bool.eq_false_iff.mpr hnorem) s.winIdx (Nat.le_refl _) hw
        · exact hrem s.winIdx (Nat.le_refl _) w hw hm
    · exact ⟨hp.1, hp.2.1, Or.inr ⟨rfl, rfl⟩⟩
    · exact Or.inr ⟨rfl, rfl⟩
  · rename_i hnall
    refine inv_syncSenderExposure (Or.inr (by simp [hc.1])) ?_
    have hreg' : Registry.Inv { s.reg with completedUnits := insertNodup srcUnit s.reg.completedUnits } := by
      simpa [hnall] using Registry.step_preserves_inv h.regInv (complete_claimed hc.1 hc.2 hsrc_in)
    have hasrc' : ∀ v ∈ s.activeSenders.filter (· != srcUnit),
        v ∈ s.allSrcUnits ∧ v ∉ insertNodup srcUnit s.reg.completedUnits := by
      intro v hv
      obtain ⟨hv_act, hv_ne⟩ := List.mem_filter.mp hv
      simp only [bne_iff_ne, ne_eq] at hv_ne
      obtain ⟨hv_in, hv_unvoted⟩ := h.activeSrc v hv_act
      exact ⟨hv_in, fun hm => (mem_insertNodup.mp hm).elim (fun e => hv_ne e) hv_unvoted⟩
    refine ⟨h.cfg, h.dstShape, hreg', h.noExposed, hasrc', fun _ => hc, h.oneLease, h.winShape, ?_⟩
    have hp := h.phase
    cases hst : s.transferStatus <;> simp only [PhaseShape, hst] at hp ⊢
    · exact absurd hp.2.2.1 hne
    · rcases hp with ⟨_, _, hact, _⟩ | ⟨hl, hz, _, hrem⟩
      · exact absurd hact hne
      · refine Or.inr ⟨hl, fun hw0 => ⟨(hz hw0).1, fun ha => absurd ((hz hw0).2 ha) hne⟩, hc, ?_⟩
        intro j hj w hw hm
        rcases mem_insertNodup.mp hm with rfl | hm
        · exact not_in_remaining hst (Bool.eq_false_iff.mpr hnorem) j hj hw
        · exact hrem j hj w hw hm
    · exact ⟨hp.1, hp.2.1, Or.inl hc⟩
    · exact Or.inl hc

private theorem step_cancelIfUnclaimed {s s' : CoordChainState}
    (h : CoordChainInv s) (hstep : CoordChainState.step s .cancelIfUnclaimed = some s') :
    CoordChainInv s' := by
  simp only [CoordChainState.step] at hstep
  cases hr : RegGen.step s.reg .cancelIfUnclaimed with
  | none => rw [hr] at hstep; cases hstep
  | some reg' =>
    rw [hr] at hstep
    simp only [Option.map_some, Option.some.injEq] at hstep; subst hstep
    have hreg' := Registry.step_preserves_inv h.regInv hr
    rcases cancelIfUnclaimed_shape hr with rfl | ⟨hnone, hc', ho', hcomp'⟩
    · refine inv_syncSenderExposure ?_
        ⟨h.cfg, h.dstShape, hreg', h.noExposed, h.activeSrc, h.activeClaimed,
         h.oneLease, h.winShape, h.phase⟩
      by_cases hact : s.activeSenders = []
      · exact Or.inl hact
      · exact Or.inr (by simp [(h.activeClaimed hact).1])
    · have hact : s.activeSenders = [] :=
        Classical.byContradiction fun hne => by
          have := (h.activeClaimed hne).1; rw [hnone] at this; cases this
      refine inv_syncSenderExposure (Or.inl hact)
        ⟨h.cfg, h.dstShape, hreg', h.noExposed, (fun _ hm => by simp [hact] at hm),
         (fun hne => absurd hact hne), h.oneLease, h.winShape, ?_⟩
      have hp := h.phase
      cases hst : s.transferStatus <;> simp only [PhaseShape, hst] at hp ⊢
      · exact ⟨hp.1, hp.2.1, hp.2.2.1, hc', ho', hcomp'⟩
      · rcases hp with ⟨hi, hidle, ha, _⟩ | ⟨_, _, hc, _⟩
        · exact Or.inl ⟨hi, hidle, ha, hc', ho', hcomp'⟩
        · exfalso; have := hc.1; rw [hnone] at this; cases this
      · exact ⟨hp.1, hp.2.1, Or.inr ⟨hc', ho'⟩⟩
      · exact Or.inr ⟨hc', ho'⟩

theorem step_preserves_coordChainInv {s s' : CoordChainState} {e : ChainEv}
    (hinv : CoordChainInv s) (hstep : CoordChainState.step s e = some s') :
    CoordChainInv s' := by
  cases e with
  | registerProducer _ => exact step_registerProducer hinv hstep
  | startChainCoordination => exact step_startChainCoordination hinv hstep
  | startWindowPlan _ => exact step_startWindowPlan hinv hstep
  | armDstResult _ _ => exact step_armDstResult hinv hstep
  | retryBusyArm => exact step_retryBusyArm hinv hstep
  | finishWindowArm => exact step_finishWindowArm hinv hstep
  | sendSrcResult _ _ => exact step_sendSrcResult hinv hstep
  | finishWindowSend => exact step_finishWindowSend hinv hstep
  | settleDispatchedRecv _ => exact step_settleDispatchedRecv hinv hstep
  | senderVoteComplete _ => exact step_senderVoteComplete hinv hstep
  | cancelIfUnclaimed => exact step_cancelIfUnclaimed hinv hstep

/-- **Main coordinator theorem (`reachable_coordChain_safe`)**: for every
well-formed chain topology, every reachable state of `CoordChainState.step`
satisfies `CoordChainSafe`. -/
theorem reachable_coordChain_safe {allSrcs : List Nat} {winSrcs : List (List Nat)}
    {dsts : List Nat} {owner maxAtt : Nat} (hcfg : ValidChainCfg allSrcs winSrcs)
    {s : CoordChainState}
    (hr : (CoordChainState.sys allSrcs winSrcs dsts owner maxAtt).Reachable s) :
    CoordChainSafe s :=
  ((CoordChainState.sys allSrcs winSrcs dsts owner maxAtt).reachable_induction
    (init_coordChainInv hcfg)
    (fun _ _ _ hinv hstep => step_preserves_coordChainInv hinv hstep) hr).toSafe

end TpuSyncVerify.Transfer.Reshard.SubPlanChain

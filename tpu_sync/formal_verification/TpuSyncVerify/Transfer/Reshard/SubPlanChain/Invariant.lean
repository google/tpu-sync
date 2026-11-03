import Std.Tactic.BVDecide
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
  change (x.toBitVec &&& (0x7FFFFFFFFFFFFFFF : BitVec 64)) ≤ (0x7FFFFFFFFFFFFFFF : BitVec 64)
  bv_decide

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
`done_recving_` / `done_sending_` for a `count`-plan request was made with
`count` distinct sub-plan indices in `[0, count)` settled OK
(`if (progress.settled.size() >= progress.count)`). -/
def DoneAfterAllSubPlans (w : WorkerState) : Prop :=
  ∀ e ∈ w.doneLog, e.2.Nodup ∧ (∀ i ∈ e.2, i < e.1) ∧ e.1 ≤ e.2.length

structure WorkerSafe (w : WorkerState) : Prop where
  doneAfterAll : DoneAfterAllSubPlans w

/-- Inductive invariant of `WorkerState.step`. -/
structure WorkerInv (w : WorkerState) : Prop where
  safe : WorkerSafe w
  settledNodup : w.chain.settled.Nodup
  settledBound : ∀ i ∈ w.chain.settled, i < w.chain.count
  activeIndex : ∀ s ∈ w.active, s.index < max 1 s.count

theorem init_workerInv : WorkerInv WorkerState.init where
  safe := ⟨fun _ h => by cases h⟩
  settledNodup := List.nodup_nil
  settledBound := fun _ h => by cases h
  activeIndex := fun _ h => by cases h

private def GoodEntry (e : Nat × List Nat) : Prop :=
  e.2.Nodup ∧ (∀ i ∈ e.2, i < e.1) ∧ e.1 ≤ e.2.length

private theorem noteSubPlanRegistered_shape (c : ChainProgress) (uuid index count : Nat)
    (hn : c.settled.Nodup) (hb : ∀ i ∈ c.settled, i < c.count) :
    (noteSubPlanRegistered c uuid index count).1.settled.Nodup ∧
    ∀ i ∈ (noteSubPlanRegistered c uuid index count).1.settled,
      i < (noteSubPlanRegistered c uuid index count).1.count := by
  unfold noteSubPlanRegistered
  split
  · exact ⟨hn, hb⟩
  · dsimp only
    split
    · exact ⟨List.nodup_nil, fun _ h => by cases h⟩
    · split
      · refine ⟨hn, ?_⟩
        intro i hi
        exact Nat.lt_of_lt_of_le (hb i hi) (Nat.le_max_left _ _)
      · exact ⟨List.nodup_nil, fun _ h => by cases h⟩

private theorem recordTerminal_shape (c : ChainProgress) (s : SubSession)
    (hn : c.settled.Nodup) (hb : ∀ i ∈ c.settled, i < c.count)
    (hs : s.index < max 1 s.count) :
    (recordTerminal c s).1.settled.Nodup ∧
    (∀ i ∈ (recordTerminal c s).1.settled, i < (recordTerminal c s).1.count) ∧
    (∀ k idx, (recordTerminal c s).2 = .done k idx → GoodEntry (k, idx)) := by
  unfold recordTerminal
  split
  · rename_i hle
    have hmax : max 1 s.count = 1 := by omega
    refine ⟨hn, hb, ?_⟩
    intro k idx h
    split at h
    · injection h with hk hidx
      subst hk; subst hidx
      refine ⟨by simp, ?_, by simp⟩
      intro i hi
      simp only [List.mem_singleton] at hi
      subst hi
      omega
    · cases h
  · rename_i hgt
    split
    · exact ⟨hn, hb, fun _ _ h => by cases h⟩
    · -- `c0` is `c` or a default-constructed entry.
      have hc0 : ∀ c0 : ChainProgress, (c0 = c ∨ c0 = { present := true }) →
          c0.settled.Nodup ∧ ∀ i ∈ c0.settled, i < c0.count := by
        rintro c0 (rfl | rfl)
        · exact ⟨hn, hb⟩
        · exact ⟨List.nodup_nil, fun _ h => by cases h⟩
      generalize hc0def : (if c.present then c else { present := true } : ChainProgress) = c0
      have hc0' : c0.settled.Nodup ∧ ∀ i ∈ c0.settled, i < c0.count := by
        split at hc0def
        · exact hc0 c0 (Or.inl hc0def.symm)
        · exact hc0 c0 (Or.inr hc0def.symm)
      obtain ⟨hn0, hb0⟩ := hc0'
      dsimp only
      split
      · exact ⟨hn0, hb0, fun _ _ h => by cases h⟩
      · split
        · refine ⟨hn0, ?_, fun _ _ h => by cases h⟩
          intro i hi
          exact Nat.lt_of_lt_of_le (hb0 i hi) (Nat.le_max_left _ _)
        · have hcount : max 1 s.count = s.count := by omega
          have hidx_bound : ∀ i ∈ insertNodup s.index c0.settled, i < max c0.count s.count := by
            intro i hi
            rcases mem_insertNodup.mp hi with rfl | hi
            · omega
            · exact Nat.lt_of_lt_of_le (hb0 i hi) (Nat.le_max_left _ _)
          split
          · rename_i hlen
            refine ⟨List.nodup_nil, fun _ h => by simp at h, ?_⟩
            intro k idx h
            injection h with hk hidx
            subst hk; subst hidx
            refine ⟨insertNodup_nodup hn0, hidx_bound, ?_⟩
            simpa using hlen
          · exact ⟨insertNodup_nodup hn0, hidx_bound, fun _ _ h => by cases h⟩

private theorem foldRecord_shape (settled : List SubSession) :
    ∀ (acc : WorkerState.SweepAcc),
      acc.chain.settled.Nodup → (∀ i ∈ acc.chain.settled, i < acc.chain.count) →
      (∀ s ∈ settled, s.index < max 1 s.count) →
      (∀ e ∈ acc.log, GoodEntry e) →
      let r := settled.foldl WorkerState.recordStep acc
      r.chain.settled.Nodup ∧ (∀ i ∈ r.chain.settled, i < r.chain.count) ∧ ∀ e ∈ r.log, GoodEntry e := by
  induction settled with
  | nil =>
    intro acc hn hb _ hlog
    exact ⟨hn, hb, hlog⟩
  | cons s rest ih =>
    intro acc hn hb hs hlog
    have hs0 := hs s (List.mem_cons_self ..)
    have hrest : ∀ s' ∈ rest, s'.index < max 1 s'.count :=
      fun s' h => hs s' (List.mem_cons_of_mem _ h)
    obtain ⟨hn', hb', hgood⟩ := recordTerminal_shape acc.chain s hn hb hs0
    simp only [List.foldl_cons]
    have hstep : ∀ acc' : WorkerState.SweepAcc, WorkerState.recordStep acc s = acc' →
        acc'.chain.settled.Nodup ∧ (∀ i ∈ acc'.chain.settled, i < acc'.chain.count) ∧
        ∀ e ∈ acc'.log, GoodEntry e := by
      intro acc' h
      unfold WorkerState.recordStep at h
      rcases hrt : recordTerminal acc.chain s with ⟨c', t⟩
      rw [hrt] at h hn' hb' hgood
      cases t with
      | done k idx =>
        subst h
        refine ⟨hn', hb', fun e he => ?_⟩
        rcases List.mem_append.mp he with he | he
        · exact hlog e he
        · simp only [List.mem_singleton] at he; subst he; exact hgood k idx rfl
      | _ => subst h; exact ⟨hn', hb', hlog⟩
    obtain ⟨hn1, hb1, hlog1⟩ := hstep _ rfl
    exact ih _ hn1 hb1 hrest hlog1

private theorem recordAll_shape (c : ChainProgress) (settled : List SubSession)
    (hn : c.settled.Nodup) (hb : ∀ i ∈ c.settled, i < c.count)
    (hs : ∀ s ∈ settled, s.index < max 1 s.count) :
    (WorkerState.recordAll c settled).chain.settled.Nodup ∧
    (∀ i ∈ (WorkerState.recordAll c settled).chain.settled,
      i < (WorkerState.recordAll c settled).chain.count) ∧
    ∀ e ∈ (WorkerState.recordAll c settled).log, GoodEntry e :=
  foldRecord_shape settled { chain := c } hn hb hs (fun _ h => by cases h)

theorem WorkerInv.toSafe {w : WorkerState} (h : WorkerInv w) : WorkerSafe w := h.safe

theorem step_preserves_workerInv {w w' : WorkerState} {e : WorkerEv}
    (hinv : WorkerInv w) (hstep : WorkerState.step w e = some w') : WorkerInv w' := by
  cases e with
  | registerSubPlan uuid index count =>
    simp only [WorkerState.step] at hstep
    split at hstep; · cases hstep
    split at hstep; · cases hstep
    rename_i hidx
    injection hstep with h; subst h
    obtain ⟨hn, hb⟩ := noteSubPlanRegistered_shape w.chain uuid index count
      hinv.settledNodup hinv.settledBound
    refine ⟨⟨hinv.safe.doneAfterAll⟩, hn, hb, fun s hs => ?_⟩
    rcases List.mem_append.mp hs with hs | hs
    · exact hinv.activeIndex s hs
    · simp only [List.mem_singleton] at hs; subst hs; exact Nat.lt_of_not_ge hidx
  | settleSession uuid ok =>
    simp only [WorkerState.step] at hstep
    split at hstep; · cases hstep
    injection hstep with h; subst h
    refine ⟨⟨hinv.safe.doneAfterAll⟩, hinv.settledNodup, hinv.settledBound, fun s' hs' => ?_⟩
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hs'
    have := hinv.activeIndex s hs
    split <;> simpa using this
  | poll stalled =>
    simp only [WorkerState.step] at hstep
    have hsettled : ∀ s ∈ w.active.filter (·.done), s.index < max 1 s.count :=
      fun s hs => hinv.activeIndex s (List.mem_filter.mp hs).1
    have hr := recordAll_shape w.chain (w.active.filter (·.done))
      hinv.settledNodup hinv.settledBound hsettled
    generalize WorkerState.recordAll w.chain (w.active.filter (·.done)) = r at hstep hr
    obtain ⟨hn1, hb1, hlog⟩ := hr
    have hsweep : ∀ (c2 : ChainProgress) (b : SweepOutcome),
        (if stalled && !r.touched then sweepStalledChain r.chain (w.active.filter (!·.done))
         else (r.chain, .notSwept)) = (c2, b) →
        c2.settled.Nodup ∧ ∀ i ∈ c2.settled, i < c2.count := by
      intro c2 b h
      split at h
      · unfold sweepStalledChain at h
        split at h; · injection h with h1 _; subst h1; exact ⟨hn1, hb1⟩
        split at h; · injection h with h1 _; subst h1; exact ⟨List.nodup_nil, fun _ h => by cases h⟩
        split at h <;> (injection h with h1 _; subst h1; exact ⟨hn1, hb1⟩)
      · injection h with h1 _; subst h1; exact ⟨hn1, hb1⟩
    generalize hr2 : (if stalled && !r.touched then sweepStalledChain r.chain (w.active.filter (!·.done))
        else (r.chain, .notSwept)) = r2 at hstep
    rcases r2 with ⟨c2, sweep⟩
    obtain ⟨hn2, hb2⟩ := hsweep c2 sweep hr2
    injection hstep with h; subst h
    refine ⟨⟨fun e he => (List.mem_append.mp he).elim (hinv.safe.doneAfterAll e) (hlog e)⟩,
      hn2, hb2, fun s hs => hinv.activeIndex s (List.mem_filter.mp hs).1⟩

/-- **Main worker theorem (`reachable_worker_safe`)**: in every reachable state of
`WorkerState.step`, `DoneAfterAllSubPlans` holds. -/
theorem reachable_worker_safe {w : WorkerState} (hr : WorkerState.sys.Reachable w) :
    WorkerSafe w :=
  (WorkerState.sys.reachable_induction init_workerInv
    (fun _ _ _ hinv hstep => step_preserves_workerInv hinv hstep) hr).toSafe

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

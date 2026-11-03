import TpuSyncVerify.Transfer.Reshard.Coordinator.Lemmas

/-! # `Coordinator`: universal protocol invariants (both policies)

Part of `TpuSyncVerify.Transfer.Reshard.Coordinator`. -/

namespace TpuSyncVerify.Transfer.Reshard.Coordinator

open TpuSyncVerify

open TpuSyncVerify.Transfer.Reshard.Registry

/-! ## Part III — Universal Protocol Invariants (Both Policies) -/

/-- In `.idle`, the registry has no active claim or completion votes, and all
destination units are unarmed. -/
def IdleShape (s : CoordState) : Prop :=
  s.phase = .idle →
    s.reg.claimedUnits = none ∧
    s.reg.claimOwners = [] ∧
    s.reg.completedUnits = [] ∧
    ∀ d ∈ s.dstStates, d.sessionArmed = false ∧ d.stagingLeaseHeld = false

/-- **Source Snapshot Claimed Throughout Stage 1 Arming (`ClaimedWhileArming`,
`ReshardCoordinator::RunSubPlan`)**:
While `s.phase = .arming`, the source snapshot is actively claimed in
`RequestBlockRegistry` (`s.reg.claimedUnits = some s.srcUnits` and
`s.reg.claimOwners = [s.claimOwner]`). -/
def ClaimedWhileArming (s : CoordState) : Prop :=
  s.phase = .arming →
    s.reg.claimedUnits = some s.srcUnits ∧
    s.reg.claimOwners = [s.claimOwner] ∧
    s.reg.completedUnits = []

/-- During `.dispatching`, the registry claim is either still held by
`s.claimOwner` or has been cleanly retired by `Complete` once all `srcUnits`
voted. -/
def DispatchingShape (s : CoordState) : Prop :=
  s.phase = .dispatching →
    (s.reg.claimedUnits = some s.srcUnits ∧ s.reg.claimOwners = [s.claimOwner]) ∨
    (s.reg.claimedUnits = none ∧ s.reg.claimOwners = [])

/-- **No Push Before All Armed (`NoPushBeforeAllArmed`,
`ReshardCoordinator::RunSubPlan`)**:
While the coordinator is in `.idle` or `.arming`, no sender has been dispatched
(`s.sendResults = []`) and no destination unit has received any push
(`∀ d ∈ s.dstStates, d.pushesReceived = []`). -/
def NoPushBeforeAllArmed (s : CoordState) : Prop :=
  (s.phase = .idle ∨ s.phase = .arming) →
    s.sendResults = [] ∧ ∀ d ∈ s.dstStates, d.pushesReceived = []

/-- **All Armed In Dispatching (`AllArmedInDispatching`,
`ReshardCoordinator::RunSubPlan`)**:
Whenever the coordinator is in `.dispatching` or `.succeeded`, the Stage 1
arming barrier succeeded (`s.armFailed = false` and `allReported s.dstUnits s.armResults = true`). -/
def AllArmedInDispatching (s : CoordState) : Prop :=
  (s.phase = .dispatching ∨ s.phase = .succeeded) →
    s.armFailed = false ∧ allReported s.dstUnits s.armResults = true

/-- **Staging Lease Matches Session Armed (`LeaseMatchesSession`,
`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`,
`ReshardReceiveSession::Create`, `ReshardReceiveSession::AcquireStagingLeases`,
`ReshardReceiveSession::ReleaseStaging`)**:
For every destination unit, its `PoolStagingLease` is held if and only if its
`ReshardReceiveSession` is currently armed. -/
def LeaseMatchesSession (s : CoordState) : Prop :=
  ∀ d ∈ s.dstStates, d.stagingLeaseHeld = d.sessionArmed

/-- Combined universal invariant of `CoordState` (holds for both `.shipping` and
`.disarmAndAbandonOnFailure`). -/
structure CoordInv (s : CoordState) : Prop where
  regInv : Registry.Inv s.reg
  idleShape : IdleShape s
  claimedArming : ClaimedWhileArming s
  dispatchingShape : DispatchingShape s
  noPushEarly : NoPushBeforeAllArmed s
  allArmedDispatch : AllArmedInDispatching s
  leaseMatches : LeaseMatchesSession s

theorem init_coordInv (srcUnits dstUnits : List Nat) (claimOwner : Nat) :
    CoordInv (CoordState.init srcUnits dstUnits claimOwner) := by
  refine ⟨Registry.init_inv, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro _
    refine ⟨rfl, rfl, rfl, ?_⟩
    intro d hd
    simp only [CoordState.init, List.mem_map] at hd
    obtain ⟨u, _, rfl⟩ := hd
    exact ⟨rfl, rfl⟩
  · intro h; contradiction
  · intro h; contradiction
  · intro _
    refine ⟨rfl, ?_⟩
    intro d hd
    simp only [CoordState.init, List.mem_map] at hd
    obtain ⟨u, _, rfl⟩ := hd
    rfl
  · intro h; cases h <;> contradiction
  · intro d hd
    simp only [CoordState.init, List.mem_map] at hd
    obtain ⟨u, _, rfl⟩ := hd
    rfl

theorem step_preserves_coordInv {policy : CleanupPolicy} {s s' : CoordState} {e : Ev}
    (hinv : CoordInv s) (hstep : CoordState.step policy s e = some s') :
    CoordInv s' := by
  cases e with
  | registerProducer srcUnit =>
    dsimp only [CoordState.step] at hstep
    split at hstep
    · cases hstep
    rename_i hphase
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hphase
    cases hreg : RegGen.step s.reg (.register srcUnit true) with
    | none => simp [hreg] at hstep
    | some reg' =>
      simp only [hreg, Option.some.injEq] at hstep
      subst hstep
      obtain ⟨hc, ho, hcomp, hdst⟩ := hinv.idleShape hphase
      obtain ⟨hc', ho', hcomp'⟩ := register_preserves_idle_reg s.reg reg' srcUnit hc ho hcomp hreg
      refine ⟨Registry.step_preserves_inv hinv.regInv hreg, ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
      · intro _; exact ⟨hc', ho', hcomp', hdst⟩
      · intro h; rw [hphase] at h; contradiction
      · intro h; rw [hphase] at h; contradiction
      · intro _; exact hinv.noPushEarly (Or.inl hphase)
      · intro h; cases h with
        | inl h => rw [hphase] at h; contradiction
        | inr h => rw [hphase] at h; contradiction

  | startPlan planOk =>
    dsimp only [CoordState.step] at hstep
    split at hstep
    · cases hstep
    rename_i hphase
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hphase
    cases hreg : RegGen.step s.reg (.lookupAndClaim s.claimOwner s.srcUnits) with
    | none => simp [hreg] at hstep
    | some regClaimed =>
      simp only [hreg] at hstep
      have hregClaimed_inv : Registry.Inv regClaimed :=
        Registry.step_preserves_inv hinv.regInv hreg
      obtain ⟨_, ho, hcomp, _⟩ := hinv.idleShape hphase
      have hclaimed := lookupAndClaim_from_idle_shape s.reg regClaimed s.claimOwner s.srcUnits ho hcomp hreg
      split at hstep
      · injection hstep with h; subst h
        refine ⟨hregClaimed_inv, ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
        · intro h; contradiction
        · intro _; exact hclaimed
        · intro h; contradiction
        · intro _; exact hinv.noPushEarly (Or.inl hphase)
        · intro h; cases h <;> contradiction
      · injection hstep with h; subst h
        refine ⟨abandonIfClaimed_inv regClaimed s.claimOwner hregClaimed_inv,
          ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
        · intro h; contradiction
        · intro h; contradiction
        · intro h; contradiction
        · intro h; cases h <;> contradiction
        · intro h; cases h <;> contradiction

  | armDstResult dstUnit ok =>
    dsimp only [CoordState.step] at hstep
    split at hstep
    · cases hstep
    rename_i hphase
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hphase
    split at hstep
    · cases hstep
    injection hstep with h; subst h
    refine ⟨hinv.regInv, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro h; rw [hphase] at h; contradiction
    · intro _; exact hinv.claimedArming hphase
    · intro h; rw [hphase] at h; contradiction
    · intro _
      obtain ⟨hsend, hpush⟩ := hinv.noPushEarly (Or.inr hphase)
      refine ⟨hsend, ?_⟩
      intro d hd
      split at hd
      · simp only [List.mem_map] at hd
        obtain ⟨d0, hd0, rfl⟩ := hd
        split
        · simp [DstUnitState.arm, hpush d0 hd0]
        · exact hpush d0 hd0
      · exact hpush d hd
    · intro h; cases h with
      | inl h => rw [hphase] at h; contradiction
      | inr h => rw [hphase] at h; contradiction
    · intro d hd
      split at hd
      · simp only [List.mem_map] at hd
        obtain ⟨d0, hd0, rfl⟩ := hd
        split
        · rfl
        · exact hinv.leaseMatches d0 hd0
      · exact hinv.leaseMatches d hd

  | finishArmBarrier =>
    dsimp only [CoordState.step] at hstep
    split at hstep
    · cases hstep
    rename_i h1
    simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not,
      Bool.not_eq_true, Bool.not_eq_false'] at h1
    obtain ⟨hphase, hall_rep⟩ := h1
    split at hstep
    · rename_i hno_fail
      simp only [Bool.not_eq_true'] at hno_fail
      injection hstep with h; subst h
      obtain ⟨hc, ho, _⟩ := hinv.claimedArming hphase
      refine ⟨hinv.regInv, ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
      · intro h; contradiction
      · intro h; contradiction
      · intro _; exact Or.inl ⟨hc, ho⟩
      · intro h; cases h <;> contradiction
      · intro _; exact ⟨hno_fail, hall_rep⟩
    · cases policy with
      | shipping =>
        injection hstep with h; subst h
        refine ⟨abandonIfClaimed_inv s.reg s.claimOwner hinv.regInv,
          ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
        · intro h; contradiction
        · intro h; contradiction
        · intro h; contradiction
        · intro h; cases h <;> contradiction
        · intro h; cases h <;> contradiction
      | disarmAndAbandonOnFailure =>
        injection hstep with h; subst h
        refine ⟨abandonIfClaimed_inv s.reg s.claimOwner hinv.regInv,
          ?_, ?_, ?_, ?_, ?_, ?_⟩
        · intro h; contradiction
        · intro h; contradiction
        · intro h; contradiction
        · intro h; cases h <;> contradiction
        · intro h; cases h <;> contradiction
        · intro d hd
          simp only [List.mem_map] at hd
          obtain ⟨d0, _, rfl⟩ := hd
          rfl

  | sendSrcResult srcUnit ok =>
    dsimp only [CoordState.step] at hstep
    split at hstep
    · cases hstep
    rename_i hphase
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hphase
    split at hstep
    · cases hstep
    split at hstep
    · injection hstep with h; subst h
      have hreg_inv : Registry.Inv (match RegGen.step s.reg (.complete srcUnit) with | some r => r | none => s.reg) := by
        cases h : RegGen.step s.reg (.complete srcUnit) with
        | none => exact hinv.regInv
        | some r => exact Registry.step_preserves_inv hinv.regInv h
      have hdisp_shape :
          (let r := match RegGen.step s.reg (.complete srcUnit) with | some r => r | none => s.reg
           (r.claimedUnits = some s.srcUnits ∧ r.claimOwners = [s.claimOwner]) ∨
           (r.claimedUnits = none ∧ r.claimOwners = [])) := by
        cases h : RegGen.step s.reg (.complete srcUnit) with
        | none => exact hinv.dispatchingShape hphase
        | some r =>
          exact complete_preserves_claim_shape s.reg r srcUnit s.srcUnits s.claimOwner
            (hinv.dispatchingShape hphase) h
      refine ⟨hreg_inv, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro h; rw [hphase] at h; contradiction
      · intro h; rw [hphase] at h; contradiction
      · intro _; exact hdisp_shape
      · intro h; cases h with
        | inl h => rw [hphase] at h; contradiction
        | inr h => rw [hphase] at h; contradiction
      · intro _; exact hinv.allArmedDispatch (Or.inl hphase)
      · intro d hd
        simp only [List.mem_map] at hd
        obtain ⟨d0, hd0, rfl⟩ := hd
        exact receivePush_leaseMatches d0 s.srcUnits srcUnit (hinv.leaseMatches d0 hd0)
    · injection hstep with h; subst h
      refine ⟨hinv.regInv, ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
      · intro h; rw [hphase] at h; contradiction
      · intro h; rw [hphase] at h; contradiction
      · intro _; exact hinv.dispatchingShape hphase
      · intro h; cases h with
        | inl h => rw [hphase] at h; contradiction
        | inr h => rw [hphase] at h; contradiction
      · intro _; exact hinv.allArmedDispatch (Or.inl hphase)

  | finishSendBarrier =>
    dsimp only [CoordState.step] at hstep
    split at hstep
    · cases hstep
    rename_i h1
    simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not] at h1
    obtain ⟨hphase, _⟩ := h1
    split at hstep
    · injection hstep with h; subst h
      refine ⟨hinv.regInv, ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
      · intro h; contradiction
      · intro h; contradiction
      · intro h; contradiction
      · intro h; cases h <;> contradiction
      · intro _; exact hinv.allArmedDispatch (Or.inl hphase)
    · cases policy with
      | shipping =>
        injection hstep with h; subst h
        refine ⟨hinv.regInv, ?_, ?_, ?_, ?_, ?_, hinv.leaseMatches⟩
        · intro h; contradiction
        · intro h; contradiction
        · intro h; contradiction
        · intro h; cases h <;> contradiction
        · intro h; cases h <;> contradiction
      | disarmAndAbandonOnFailure =>
        injection hstep with h; subst h
        refine ⟨abandonIfClaimed_inv s.reg s.claimOwner hinv.regInv,
          ?_, ?_, ?_, ?_, ?_, ?_⟩
        · intro h; contradiction
        · intro h; contradiction
        · intro h; contradiction
        · intro h; cases h <;> contradiction
        · intro h; cases h <;> contradiction
        · intro d hd
          simp only [List.mem_map] at hd
          obtain ⟨d0, _, rfl⟩ := hd
          rfl

/-- **Main Universal Reachability Theorem (`reachable_coord_inv`)**:
Every reachable state of `CoordState.sys` (under either `.shipping` or
`.disarmAndAbandonOnFailure`) satisfies `CoordInv`. -/
theorem reachable_coord_inv {policy : CleanupPolicy} {srcUnits dstUnits : List Nat}
    {claimOwner : Nat} {s : CoordState}
    (hr : (CoordState.sys policy srcUnits dstUnits claimOwner).Reachable s) :
    CoordInv s := by
  induction hr with
  | init => exact init_coordInv srcUnits dstUnits claimOwner
  | step _ hstep ih => exact step_preserves_coordInv ih hstep

/-- **Corollary (`no_push_before_all_armed`)**:
In any reachable state, while the coordinator is in `.idle` or `.arming`, no
sender has been dispatched and no destination unit has received any push. -/
theorem no_push_before_all_armed {policy : CleanupPolicy} {srcUnits dstUnits : List Nat}
    {claimOwner : Nat} {s : CoordState}
    (hr : (CoordState.sys policy srcUnits dstUnits claimOwner).Reachable s) :
    NoPushBeforeAllArmed s :=
  (reachable_coord_inv hr).noPushEarly

/-- **Corollary (`claimed_while_arming`)**:
In any reachable state, while the coordinator is in `.arming`, the source
snapshot is claimed in `RequestBlockRegistry` (`s.reg.claimedUnits = some s.srcUnits`),
and therefore `CancelIfUnclaimed` and late `Register` calls are blocked. -/
theorem claimed_while_arming {policy : CleanupPolicy} {srcUnits dstUnits : List Nat}
    {claimOwner : Nat} {s : CoordState}
    (hr : (CoordState.sys policy srcUnits dstUnits claimOwner).Reachable s)
    (harm : s.phase = .arming) :
    s.reg.claimedUnits = some s.srcUnits ∧
    RegGen.step s.reg .cancelIfUnclaimed = some s.reg := by
  have hinv := reachable_coord_inv hr
  obtain ⟨hclaim, _, _⟩ := hinv.claimedArming harm
  refine ⟨hclaim, ?_⟩
  exact (Registry.claimed_blocks_register_and_cancel hinv.regInv hclaim 0 true).2


end TpuSyncVerify.Transfer.Reshard.Coordinator

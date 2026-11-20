import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.Coordinator.Invariant

/-! # `Coordinator`: replay traces, searches, mutants, and disarm-and-abandon alternative

Part of `TpuSyncVerify.Transfer.Reshard.Coordinator`; see `Coordinator.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard.Coordinator

open TpuSyncVerify

open TpuSyncVerify.Transfer.Reshard.Registry

/-! ## Part IV — Traces, searches, and mutants -/

/-- Replays `ReshardStackTest.TwoDestinationsOneArmRefusalAbandonsClaimAndSkipsSenders`:
`dst_unit 10` arms and `dst_unit 11` refuses; `RunSubPlan` abandons the claim without disarming `10`. -/
def partialArmFailureTrace : List Ev :=
  [ .registerProducer 0
  , .registerProducer 1
  , .startPlan true
  , .armDstResult 10 true
  , .armDstResult 11 false
  , .finishArmBarrier ]

/-- `partialArmFailureTrace` aborts with `dst_unit 10` keeping `sessionArmed = true` and `stagingLeaseHeld = true`. -/
theorem trace_partial_arm_orphans_armed_receiver :
    ((CoordState.sys [0, 1] [10, 11] 101).run partialArmFailureTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.claimedUnits )) =
      some (.aborted, [(10, true, true), (11, false, false)], none) := by
  decide

/-- 2×2 counterpart of `ReshardStackTest.SenderDispatchFailurePropagatesErrorAfterReceiverArm`:
both destinations arm, `src_unit 0` succeeds, and `src_unit 1` fails Stage 2 dispatch. -/
def dispatchFailureTrace : List Ev :=
  [ .registerProducer 0
  , .registerProducer 1
  , .startPlan true
  , .armDstResult 10 true
  , .armDstResult 11 true
  , .finishArmBarrier
  , .sendSrcResult 0 true
  , .sendSrcResult 1 false
  , .finishSendBarrier ]

/-- `dispatchFailureTrace` aborts leaving both destinations armed/leased and the registry claim held (`CancelIfUnclaimed` returns `false`). -/
theorem trace_dispatch_failure_orphans_receivers_and_claim :
    ((CoordState.sys [0, 1] [10, 11] 101).run dispatchFailureTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.claimedUnits
        , s.reg.claimOwners
        , (RegGen.step s.reg .cancelIfUnclaimed).map (fun r => r.cancelled) )) =
      some (.aborted, [(10, true, true), (11, true, true)], some [0, 1], [101], some false) := by
  rfl

/-- Replays `ReshardStackTest.TwoDestinationsArmEachThenDispatchSendersOnce`:
both destinations arm before either sender pushes, and all leases and the registry generation retire cleanly. -/
def coordinatorHappyTrace : List Ev :=
  [ .registerProducer 0
  , .registerProducer 1
  , .startPlan true
  , .armDstResult 10 true
  , .armDstResult 11 true
  , .finishArmBarrier
  , .sendSrcResult 0 true
  , .sendSrcResult 1 true
  , .finishSendBarrier ]

theorem trace_coordinator_happy_path :
    ((CoordState.sys [0, 1] [10] 101).run
      [ .registerProducer 0
      , .registerProducer 1
      , .startPlan true
      , .armDstResult 10 true
      , .finishArmBarrier
      , .sendSrcResult 0 true
      , .sendSrcResult 1 true
      , .finishSendBarrier ]).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.completed, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.retired )) =
      some (.succeeded, [(10, true, false, false)], true) ∧
    ((CoordState.sys [0, 1] [10, 11] 101).run coordinatorHappyTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.completed, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.retired
        , s.reg.claimedUnits
        , s.reg.registeredUnits )) =
      some (.succeeded, [(10, true, false, false), (11, true, false, false)], true, none, []) := by
  exact ⟨rfl, rfl⟩

/-- Replays `ReshardStackTest.FullPoolReshardFlowEmitsArmThenDispatch`,
`ReshardStackTest.MissingRegistrationKeepsContractString`, and
`ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders`: missing registration is refused, and Stage 1 arm failure abandons the claim without dispatching senders. -/
theorem trace_arm_failure_abandons_claim_and_skips_senders :
    (CoordState.sys [0] [10] 101).run [.startPlan true] = none ∧
    (((CoordState.sys [0] [10] 101).run
      [ .registerProducer 0
      , .startPlan true
      , .armDstResult 10 false
      , .finishArmBarrier ]).map
      (fun s =>
        ( s.phase
        , s.sendResults
        , s.reg.claimedUnits
        , s.reg.status
        , (RegGen.step s.reg (.lookupAndClaim 102 [0])).map
            (fun r => r.claimedUnits) )) =
      some (.aborted, [], none, .registered, some (some [0]))) := by
  exact ⟨rfl, rfl⟩

/-- Mutant: dispatching Stage 2 without the Stage 1 `armers` join barrier drops a push arriving at unarmed `dst_unit 11`. -/
def stepSendWithoutArmBarrier (s : CoordState) (srcUnit : Nat) : CoordState :=
  { s with
    dstStates := s.dstStates.map (fun d => d.receivePush s.srcUnits srcUnit)
    sendResults := (srcUnit, true) :: s.sendResults }

theorem mutant_dispatch_without_arm_barrier_drops_push :
    ((CoordState.sys [0] [10, 11] 101).run
      [.registerProducer 0, .startPlan true, .armDstResult 10 true]).map
      (fun s =>
        let s_premature := stepSendWithoutArmBarrier s 0
        s_premature.dstStates.map (fun d => (d.unit, d.completed, d.pushesReceived))) =
      some [(10, true, [0]), (11, false, [])] := by
  decide

/-- Candidate coordinator events for `srcUnits = [0, 1]`, `dstUnits = [10, 11]`. -/
def checkEvents : List Ev :=
  [ .registerProducer 0, .registerProducer 1
  , .startPlan true, .startPlan false
  , .armDstResult 10 true, .armDstResult 10 false
  , .armDstResult 11 true, .armDstResult 11 false
  , .finishArmBarrier
  , .sendSrcResult 0 true, .sendSrcResult 0 false
  , .sendSrcResult 1 true, .sendSrcResult 1 false
  , .finishSendBarrier ]

instance (s : CoordState) : Decidable (IdleShape s) := by unfold IdleShape; infer_instance
instance (s : CoordState) : Decidable (ClaimedWhileArming s) := by unfold ClaimedWhileArming; infer_instance
instance (s : CoordState) : Decidable (DispatchingShape s) := by unfold DispatchingShape; infer_instance
instance (s : CoordState) : Decidable (NoPushBeforeAllArmed s) := by unfold NoPushBeforeAllArmed; infer_instance
instance (s : CoordState) : Decidable (AllArmedInDispatching s) := by unfold AllArmedInDispatching; infer_instance
instance (s : CoordState) : Decidable (LeaseMatchesSession s) := by unfold LeaseMatchesSession; infer_instance
instance (s : CoordState) : Decidable (CoordInv s) :=
  decidable_of_iff
    (Registry.Inv s.reg ∧ IdleShape s ∧ ClaimedWhileArming s ∧ DispatchingShape s ∧
      NoPushBeforeAllArmed s ∧ AllArmedInDispatching s ∧ LeaseMatchesSession s)
    ⟨fun ⟨a, b, c, d, e, f, g⟩ => ⟨a, b, c, d, e, f, g⟩,
     fun h => ⟨h.1, h.2, h.3, h.4, h.5, h.6, h.7⟩⟩

def violatesCoordInv (s : CoordState) : Bool := !decide (CoordInv s)

theorem violatesCoordInv_iff (s : CoordState) : violatesCoordInv s = false ↔ CoordInv s := by
  simp [violatesCoordInv]

#guard ModelCheck.check (CoordState.sys [0, 1] [10, 11] 101) checkEvents violatesCoordInv 12 = .safe

/-! ### Disarm-and-abandon alternative (not shipping C++) -/

/-- Disarm a destination, releasing its `ReshardReceiveSession` and `PoolStagingLease`. -/
def disarm (d : DstUnitState) : DstUnitState :=
  { d with sessionArmed := false, stagingLeaseHeld := false }

def stepDisarmAndAbandon (s : CoordState) (e : Ev) : Option CoordState :=
  match e with
  | .finishArmBarrier | .finishSendBarrier =>
    (CoordState.step s e).map fun s' =>
      if s'.phase == .aborted then
        { s' with dstStates := s'.dstStates.map disarm, reg := abandonIfClaimed s'.reg s'.claimOwner }
      else s'
  | e => CoordState.step s e

def sysDisarmAndAbandon (srcUnits dstUnits : List Nat) (claimOwner : Nat) : System CoordState Ev :=
  ⟨CoordState.init srcUnits dstUnits claimOwner, stepDisarmAndAbandon⟩

/-- No armed receiver, held lease, or registry claim survives an abort. -/
def NoOrphansWhenAborted (s : CoordState) : Prop :=
  s.phase = .aborted →
    (∀ d ∈ s.dstStates, d.sessionArmed = false ∧ d.stagingLeaseHeld = false) ∧
    s.reg.claimedUnits = none ∧ s.reg.claimOwners = []

instance (s : CoordState) : Decidable (NoOrphansWhenAborted s) := by
  unfold NoOrphansWhenAborted; infer_instance

def violatesNoOrphans (s : CoordState) : Bool := !decide (CoordInv s ∧ NoOrphansWhenAborted s)

/-- On both abort traces the alternative leaves nothing armed, leased, or claimed. -/
theorem trace_disarm_and_abandon_cleans_up_both_aborts :
    ((sysDisarmAndAbandon [0, 1] [10, 11] 101).run partialArmFailureTrace).map
      (fun s => (s.phase, s.dstStates.map (fun d => (d.sessionArmed, d.stagingLeaseHeld)),
        s.reg.claimedUnits)) =
      some (.aborted, [(false, false), (false, false)], none) ∧
    ((sysDisarmAndAbandon [0, 1] [10, 11] 101).run dispatchFailureTrace).map
      (fun s => (s.phase, s.dstStates.map (fun d => (d.sessionArmed, d.stagingLeaseHeld)),
        s.reg.claimedUnits, s.reg.claimOwners)) =
      some (.aborted, [(false, false), (false, false)], none, []) := by
  exact ⟨rfl, rfl⟩

#guard (ModelCheck.check (CoordState.sys [0, 1] [10, 11] 101) checkEvents violatesNoOrphans 12
  matches .counterexample _)
#guard ModelCheck.check (sysDisarmAndAbandon [0, 1] [10, 11] 101) checkEvents violatesNoOrphans 12 = .safe

end TpuSyncVerify.Transfer.Reshard.Coordinator

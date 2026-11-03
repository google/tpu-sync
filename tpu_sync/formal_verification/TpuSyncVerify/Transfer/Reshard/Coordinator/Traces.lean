import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.Coordinator.Hardened

/-! # `Coordinator`: failure-cleanup counterexamples, happy-path trace, and mutants

Part of `TpuSyncVerify.Transfer.Reshard.Coordinator`. -/

namespace TpuSyncVerify.Transfer.Reshard.Coordinator

open TpuSyncVerify

open TpuSyncVerify.Transfer.Reshard.Registry

/-! ## Part V — Failure-Cleanup Counterexamples, Happy-Path Trace, and Mutants -/

/-- **Partial-arm failure counterexample trace (`ReshardCoordinator::RunSubPlan`, `return arm_status[i];`)**:
Two source units `[0, 1]` and two destination units `[10, 11]`.
During Stage 1 (receiver arming), `dst_unit 10` succeeds (`armDstResult 10 true`),
creating an active `ReshardReceiveSession` holding a `PoolStagingLease`, while
`dst_unit 11` fails (`armDstResult 11 false`).
At `finishArmBarrier`, the shipping coordinator calls `AbandonClaim`
(`if (index == 0)`) and returns `arm_status[i]` (`.aborted`) **without disarming
`dst_unit 10`**. -/
def partialArmFailureTrace : List Ev :=
  [ .registerProducer 0
  , .registerProducer 1
  , .startPlan true
  , .armDstResult 10 true
  , .armDstResult 11 false
  , .finishArmBarrier ]

/-- Under `.shipping`, `partialArmFailureTrace` ends in `.aborted` with
`dst_unit 10` still holding `sessionArmed = true` and `stagingLeaseHeld = true`
(orphaned until `timeout_s_`). -/
theorem trace_partial_arm_orphans_armed_receiver :
    ((CoordState.sys .shipping [0, 1] [10, 11] 101).run partialArmFailureTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.claimedUnits )) =
      some (.aborted, [(10, true, true), (11, false, false)], none) := by
  decide

/-- **Sender-dispatch failure counterexample trace (`ReshardCoordinator::RunSubPlan`, `for (const absl::Status& status : sender_status)`)**:
Two source units `[0, 1]` and two destination units `[10, 11]`.
Both `dst_unit 10` and `dst_unit 11` arm in Stage 1 (`finishArmBarrier` transitions
to `.dispatching`). During Stage 2 (sender dispatch), `src_unit 0` succeeds
(`sendSrcResult 0 true`), pushing to `10` and `11` and voting `Complete(0)` in
`RequestBlockRegistry`, while `src_unit 1` fails (`sendSrcResult 1 false`).
At `finishSendBarrier`, the shipping coordinator returns `sender_status`
(`.aborted`) **without disarming `dst_units` AND without calling
`registry_->AbandonClaim(args.req_id, uuid, ctx->claim_owner)`** (for any
sub-plan index). -/
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

/-- Under `.shipping`, `dispatchFailureTrace` ends in `.aborted` with:
1. Both `dst_unit 10` and `dst_unit 11` orphaned with `sessionArmed = true` and
   `stagingLeaseHeld = true` (waiting for `src_unit 1`'s push until `timeout_s_`);
2. `s.reg.claimedUnits = some [0, 1]` and `s.reg.claimOwners = [101]` leaked in
   `RequestBlockRegistry` (blocking `CancelIfUnclaimed` until `ttl_s_`). -/
theorem trace_dispatch_failure_orphans_receivers_and_claim :
    ((CoordState.sys .shipping [0, 1] [10, 11] 101).run dispatchFailureTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.claimedUnits
        , s.reg.claimOwners
        , (RegGen.step s.reg .cancelIfUnclaimed).map (fun r => r.cancelled) )) =
      some (.aborted, [(10, true, true), (11, true, true)], some [0, 1], [101], some false) := by
  rfl

/-- Under `.disarmAndAbandonOnFailure`, both `partialArmFailureTrace` and
`dispatchFailureTrace` clean up all receiver sessions, staging leases, and
registry claims on abort. -/
theorem trace_hardened_policy_cleans_up_partial_arm_and_dispatch_failure :
    (((CoordState.sys .disarmAndAbandonOnFailure [0, 1] [10, 11] 101).run
        partialArmFailureTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.claimedUnits )) =
      some (.aborted, [(false, false), (false, false)], none)) ∧
    (((CoordState.sys .disarmAndAbandonOnFailure [0, 1] [10, 11] 101).run
        dispatchFailureTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.claimedUnits
        , s.reg.claimOwners )) =
      some (.aborted, [(false, false), (false, false)], none, [])) := by
  exact ⟨rfl, rfl⟩

/-- Happy-path 2x2 multicast reshard execution trace
(`ReshardCoordinator::RunSubPlan`,
`ReshardStackTest.TwoDestinationsArmEachThenDispatchSendersOnce`):
both producers (`[0, 1]`) register, coordinator claims the plan, both
destinations (`[10, 11]`) arm concurrently before either sender is dispatched,
both senders push and vote `Complete`, both receivers complete and release their
staging leases, and `RequestBlockRegistry` retires the generation. -/
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
    ((CoordState.sys .shipping [0, 1] [10] 101).run
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
    ((CoordState.sys .shipping [0, 1] [10, 11] 101).run coordinatorHappyTrace).map
      (fun s =>
        ( s.phase
        , s.dstStates.map (fun d => (d.unit, d.completed, d.sessionArmed, d.stagingLeaseHeld))
        , s.reg.retired
        , s.reg.claimedUnits
        , s.reg.registeredUnits )) =
      some (.succeeded, [(10, true, false, false), (11, true, false, false)], true, none, []) := by
  exact ⟨rfl, rfl⟩

/-- Replay of `ReshardStackTest.FullPoolReshardFlowEmitsArmThenDispatch`,
`ReshardStackTest.MissingRegistrationKeepsContractString`, and
`ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders`:
1. `ReshardStackTest.MissingRegistrationKeepsContractString`: starting a plan when
   `src_unit 0` has not registered in `RequestBlockRegistry` is rejected (`none`).
2. `ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders`: on a 1-source (`[0]`),
   1-destination (`[10]`) plan, when `dst_unit 10` refuses its arm
   (`armDstResult 10 false`, `finishArmBarrier`), the coordinator aborts without
   contacting any sender (`sendResults = []`), abandons its registry claim
   (`s.reg.claimedUnits = none`, `s.reg.status = .registered`), and a fresh
   coordination claim (`RegGen.step s.reg (.lookupAndClaim 102 [0])`) succeeds. -/
theorem trace_arm_failure_abandons_claim_and_skips_senders :
    (CoordState.sys .shipping [0] [10] 101).run [.startPlan true] = none ∧
    (((CoordState.sys .shipping [0] [10] 101).run
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

/-- Mutant: If Stage 2 dispatched senders without waiting for the Stage 1 join
barrier (`for (std::thread& t : armers) t.join()` in `ReshardCoordinator::RunSubPlan`), a sender push arriving before
`dst_unit 11` has armed would be dropped (`d.sessionArmed = false`), leaving
`dst_unit 11` incomplete even after all senders finish. -/
def stepSendWithoutArmBarrier (s : CoordState) (srcUnit : Nat) : CoordState :=
  { s with
    dstStates := s.dstStates.map (fun d => d.receivePush s.srcUnits srcUnit)
    sendResults := (srcUnit, true) :: s.sendResults }

theorem mutant_dispatch_without_arm_barrier_drops_push :
    ((CoordState.sys .shipping [0] [10, 11] 101).run
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

/-- Executable negation of universal `CoordInv` properties. -/
def violatesCoordInv (s : CoordState) : Bool :=
  Registry.violates s.reg ||
  ((s.phase == .idle || s.phase == .arming) &&
   (!s.sendResults.isEmpty || s.dstStates.any (fun d => !d.pushesReceived.isEmpty || d.completed))) ||
  (s.phase == .arming &&
   (s.reg.claimedUnits != some s.srcUnits || s.reg.claimOwners != [s.claimOwner])) ||
  ((s.phase == .dispatching || s.phase == .succeeded) &&
   (s.armFailed || !allReported s.dstUnits s.armResults)) ||
  s.dstStates.any (fun d => d.stagingLeaseHeld != d.sessionArmed)

/-- Executable check for orphan freedom in `.aborted`. -/
def violatesNoOrphans (s : CoordState) : Bool :=
  violatesCoordInv s ||
  (s.phase == .aborted &&
   (s.dstStates.any (fun d => d.sessionArmed || d.stagingLeaseHeld) ||
    s.reg.claimedUnits.isSome || !s.reg.claimOwners.isEmpty))

#guard ModelCheck.check (CoordState.sys .shipping [0, 1] [10, 11] 101) checkEvents violatesCoordInv 12 = .safe
#guard ModelCheck.check (CoordState.sys .disarmAndAbandonOnFailure [0, 1] [10, 11] 101) checkEvents violatesNoOrphans 12 = .safe
#guard (match ModelCheck.check (CoordState.sys .shipping [0, 1] [10, 11] 101) checkEvents violatesNoOrphans 12 with
  | .counterexample _ => true | _ => false)


end TpuSyncVerify.Transfer.Reshard.Coordinator

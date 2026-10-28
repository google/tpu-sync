import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Reshard.Registry

/-! # `Coordinator`: coordinator, receiver-arming, and cleanup policy state machine

Part of `TpuSyncVerify.Transfer.Reshard.Coordinator`; the module docstring and
source map live in `Coordinator.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard.Coordinator

open TpuSyncVerify
open TpuSyncVerify.Transfer.Reshard.Registry

/-! ## Part I — Coordinator, Receiver-Arming, and Cleanup Policy State Machine -/

/-- Coordinator failure cleanup policy comparing the shipping implementation
(`ReshardCoordinator::ExecutePoolReshard`'s `arm_status` and `sender_status` checks) against a hardened policy that
disarms armed receivers and abandons the registry claim on any barrier failure. -/
inductive CleanupPolicy where
  /-- Shipping C++ behavior:
      * Arm failure (`for (const absl::Status& status : arm_status)`): calls `AbandonClaim`,
        does not disarm already-armed `dst_units`.
      * Send failure (`for (const absl::Status& status : sender_status)`): neither disarms
        `dst_units` nor calls `AbandonClaim`. -/
  | shipping
  /-- Hardened cleanup behavior: on either arm-barrier failure or send-barrier
      failure, disarms all currently armed `dst_units` (releasing their
      `ReshardReceiveSession` and `PoolStagingLease`) and calls `AbandonClaim`
      on `RequestBlockRegistry`. -/
  | disarmAndAbandonOnFailure
  deriving DecidableEq, Repr

/-- Lifecycle phase of `ReshardCoordinator::ExecutePoolReshard`. -/
inductive CoordPhase where
  /-- Before `ExecutePoolReshard` claims the source snapshot. -/
  | idle
  /-- Stage 1 (`ReshardCoordinator::ExecutePoolReshard`, `armers` fan-out): `BuildPoolReshardPlan`
      claimed the source snapshot in `RequestBlockRegistry`; concurrent
      `PoolReshardRegisterRecv` RPCs to `dst_units` are in flight. -/
  | arming
  /-- Stage 2 (`ReshardCoordinator::ExecutePoolReshard`, `sender_status` fan-out): all `dst_units` armed with
      `arm_status.ok()`; concurrent `PoolReshardPush` RPCs to `src_units` are
      in flight. -/
  | dispatching
  /-- Both stages completed with `return absl::OkStatus()` (`ReshardCoordinator::ExecutePoolReshard`). -/
  | succeeded
  /-- `ExecutePoolReshard` returned a non-OK `absl::Status`
      (`if (!plan_or.ok())`, `for (const absl::Status& status : arm_status)`, or `for (const absl::Status& status : sender_status)`). -/
  | aborted
  deriving DecidableEq, Repr

/-- Per-destination-unit receiver state (`KVCacheListener::HandleControlRequest`,
`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`,
`ReshardReceiveSession::Create`, `ReshardReceiveSession::AcquireStagingLeases`,
`ReshardReceiveSession::ReleaseStaging`). -/
structure DstUnitState where
  /-- Destination unit ID (`UnitKey`). -/
  unit : Nat
  /-- True while `ReshardReceiveSession` is registered in
      `active_pool_reshard_recvs_` with an armed plan in `active_plans_`. -/
  sessionArmed : Bool
  /-- True while the `PoolStagingLease` acquired by `AcquireStagingLeases`
      (`ReshardReceiveSession::Create` and `ReshardReceiveSession::AcquireStagingLeases`) is held. -/
  stagingLeaseHeld : Bool
  /-- Set of `src_unit` IDs whose `PoolReshardPush` has arrived. -/
  pushesReceived : List Nat
  /-- True once all `src_units` have pushed and the session has settled and
      released its staging lease (`ReshardReceiveSession::ReleaseStagingLocked` / `ReshardReceiveSession::ReleaseStaging`). -/
  completed : Bool
  deriving DecidableEq, Repr

namespace DstUnitState

/-- Initial unarmed destination unit state. -/
def init (unit : Nat) : DstUnitState :=
  { unit := unit
    sessionArmed := false
    stagingLeaseHeld := false
    pushesReceived := []
    completed := false }

/-- Arm this destination unit via `PoolReshardRegisterRecv`
(`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`,
`ReshardReceiveSession::Create`). -/
def arm (d : DstUnitState) : DstUnitState :=
  { d with
    sessionArmed := true
    stagingLeaseHeld := true }

/-- Explicitly disarm this destination unit on coordinator abort, releasing its
active `ReshardReceiveSession` and `PoolStagingLease`. -/
def disarm (d : DstUnitState) : DstUnitState :=
  { d with
    sessionArmed := false
    stagingLeaseHeld := false }

/-- Deliver a sender push from `srcUnit` to an armed destination unit; if all
`srcUnits` have now contributed (`ReshardReceiveSession::RecordPoolH2dResultLocked` through `ReshardReceiveSession::FinishPoolH2d`), complete
the session and release its staging lease (`ReshardReceiveSession::ReleaseStagingLocked`). -/
def receivePush (d : DstUnitState) (srcUnits : List Nat) (srcUnit : Nat) : DstUnitState :=
  if !d.sessionArmed then
    d
  else
    let pushed := insertNodup srcUnit d.pushesReceived
    if allVoted srcUnits pushed then
      { d with
        sessionArmed := false
        stagingLeaseHeld := false
        pushesReceived := pushed
        completed := true }
    else
      { d with pushesReceived := pushed }

end DstUnitState

/-- Combined state of `ReshardCoordinator::ExecutePoolReshard`, the destination
units' `ReshardReceiveSession`s, and the shared `RequestBlockRegistry` generation. -/
structure CoordState where
  /-- Source unit IDs (`plan.src_units`). -/
  srcUnits : List Nat
  /-- Destination unit IDs (`plan.dst_units`). -/
  dstUnits : List Nat
  /-- Coordinator's unique `claim_owner` token (`claim_counter.fetch_add(1)` in
      `ReshardCoordinator::ExecutePoolReshard`). -/
  claimOwner : Nat
  /-- Current coordinator lifecycle phase. -/
  phase : CoordPhase
  /-- Per-destination-unit state. -/
  dstStates : List DstUnitState
  /-- Completed `PoolReshardRegisterRecv` RPC results `(dstUnit, ok)` collected
      during Stage 1 (`armers` in `ReshardCoordinator::ExecutePoolReshard`). -/
  armResults : List (Nat × Bool)
  /-- `true` if `!arm_status.ok()` during Stage 1 (`arm_status[i]` in `ReshardCoordinator::ExecutePoolReshard`). -/
  armFailed : Bool
  /-- Completed `PoolReshardPush` RPC results `(srcUnit, ok)` collected during
      Stage 2 (`futures` / `senders` in `ReshardCoordinator::ExecutePoolReshard`). -/
  sendResults : List (Nat × Bool)
  /-- `true` if `!sender_status.ok()` during Stage 2 (`sender_status[i]` in `ReshardCoordinator::ExecutePoolReshard`). -/
  sendFailed : Bool
  /-- Underlying `RequestBlockRegistry` state for `(args.req_id, args.uuid)`. -/
  reg : RegGen
  deriving DecidableEq, Repr

/-- Events of the two-stage coordinator + registry composition. -/
inductive Ev where
  /-- Producer unit registers its source blocks in `RequestBlockRegistry`
      before reshard planning (`RequestBlockRegistry::Register`). -/
  | registerProducer (srcUnit : Nat)
  /-- `BuildPoolReshardPlan(plan_request, registry_, claim_owner)` (`ReshardCoordinator::ExecutePoolReshard`):
      claims `srcUnits` in `RequestBlockRegistry` (`LookupAndClaim`), and then
      either succeeds (`planOk = true` -> `.arming`) or fails after claiming
      (`planOk = false` -> `AbandonClaim` + `.aborted`). -/
  | startPlan (planOk : Bool)
  /-- Stage 1 worker RPC `PoolReshardRegisterRecv` finishes for `dstUnit`
      (`armers.emplace_back` in `ReshardCoordinator::ExecutePoolReshard`). -/
  | armDstResult (dstUnit : Nat) (ok : Bool)
  /-- Stage 1 join barrier `for (std::thread& t : armers) t.join()` and
      `for (const absl::Status& status : arm_status)` check (`ReshardCoordinator::ExecutePoolReshard`). -/
  | finishArmBarrier
  /-- Stage 2 worker RPC `PoolReshardPush` finishes for `srcUnit`
      (`SubmitTransferProgram` / `senders.emplace_back` in `ReshardCoordinator::ExecutePoolReshard`). -/
  | sendSrcResult (srcUnit : Nat) (ok : Bool)
  /-- Stage 2 join barrier `for (std::thread& t : senders) t.join()` and
      `for (const absl::Status& status : sender_status)` check (`ReshardCoordinator::ExecutePoolReshard`). -/
  | finishSendBarrier
  deriving DecidableEq, Repr

/-- Helper: check if `unit` already has an RPC result in `results`. -/
def hasResult (results : List (Nat × Bool)) (unit : Nat) : Bool :=
  results.any (fun (u, _) => u == unit)

/-- Helper: check if every unit in `units` has reported an RPC result. -/
def allReported (units : List Nat) (results : List (Nat × Bool)) : Bool :=
  units.all (fun u => hasResult results u)

/-- Helper: abandon `claimOwner` in `reg` if currently claimed by `claimOwner`
(`RequestBlockRegistry::AbandonClaim`). -/
def abandonIfClaimed (reg : RegGen) (claimOwner : Nat) : RegGen :=
  match RegGen.step reg (.abandonClaim claimOwner) with
  | some reg' => reg'
  | none => reg

namespace CoordState

/-- Initial coordinator + registry state for a given `(srcUnits, dstUnits, claimOwner)`
configuration. -/
def init (srcUnits dstUnits : List Nat) (claimOwner : Nat) : CoordState :=
  { srcUnits := srcUnits
    dstUnits := dstUnits
    claimOwner := claimOwner
    phase := .idle
    dstStates := dstUnits.map DstUnitState.init
    armResults := []
    armFailed := false
    sendResults := []
    sendFailed := false
    reg := RegGen.init }

/-- Small-step transition relation of `ReshardCoordinator::ExecutePoolReshard`
parameterized by `CleanupPolicy`. -/
def step (policy : CleanupPolicy) (s : CoordState) (e : Ev) : Option CoordState :=
  match e with
  | .registerProducer srcUnit =>
    if s.phase != .idle then
      none
    else
      match RegGen.step s.reg (.register srcUnit true) with
      | some reg' => some { s with reg := reg' }
      | none => none

  | .startPlan planOk =>
    if s.phase != .idle then
      none
    else
      -- `BuildPoolReshardPlan` calls `LookupAndClaim(req_id, uuid, request.src_units, claim_owner)`
      -- (`ReshardCoordinator::ExecutePoolReshard`, `BuildPoolReshardPlan`).
      match RegGen.step s.reg (.lookupAndClaim s.claimOwner s.srcUnits) with
      | none => none
      | some regClaimed =>
        if planOk then
          some { s with phase := .arming, reg := regClaimed }
        else
          -- `if (!plan_or.ok())`: on plan failure, calls `registry_->AbandonClaim(args.req_id, uuid, claim_owner)`
          -- and returns error.
          some
            { s with
              phase := .aborted
              reg := abandonIfClaimed regClaimed s.claimOwner }

  | .armDstResult dstUnit ok =>
    if s.phase != .arming then
      none
    else if !s.dstUnits.contains dstUnit || hasResult s.armResults dstUnit then
      none
    else
      let dstStates' :=
        if ok then
          s.dstStates.map (fun d => if d.unit == dstUnit then d.arm else d)
        else
          s.dstStates
      some
        { s with
          dstStates := dstStates'
          armResults := (dstUnit, ok) :: s.armResults
          armFailed := s.armFailed || !ok }

  | .finishArmBarrier =>
    if s.phase != .arming || !allReported s.dstUnits s.armResults then
      none
    else if !s.armFailed then
      -- All `dst_units` armed with `arm_status.ok()` (`for (std::thread& t : armers) t.join()`).
      some { s with phase := .dispatching }
    else
      -- `if (!status.ok())` in `for (const absl::Status& status : arm_status)`:
      match policy with
      | .shipping =>
        -- Shipping code calls `registry_->AbandonClaim(args.req_id, uuid, claim_owner)` and returns `status`
        -- WITHOUT disarming `dst_units` that already armed!
        some
          { s with
            phase := .aborted
            reg := abandonIfClaimed s.reg s.claimOwner }
      | .disarmAndAbandonOnFailure =>
        some
          { s with
            phase := .aborted
            dstStates := s.dstStates.map DstUnitState.disarm
            reg := abandonIfClaimed s.reg s.claimOwner }

  | .sendSrcResult srcUnit ok =>
    if s.phase != .dispatching then
      none
    else if !s.srcUnits.contains srcUnit || hasResult s.sendResults srcUnit then
      none
    else if ok then
      -- `PoolReshardPush` succeeded (`ReshardCoordinator::ExecutePoolReshard`,
      -- `ReshardService::HandleControllerCommand` `COMMAND_COMPLETE_REQUEST_BLOCKS`, `RequestBlockRegistry::Complete`):
      -- pushes sub-blocks to armed `dst_units` and votes `registry_->Complete`.
      let dstStates' :=
        s.dstStates.map (fun d => d.receivePush s.srcUnits srcUnit)
      let reg' :=
        match RegGen.step s.reg (.complete srcUnit) with
        | some r => r
        | none => s.reg
      some
        { s with
          dstStates := dstStates'
          sendResults := (srcUnit, true) :: s.sendResults
          reg := reg' }
    else
      -- `!status.ok()` (`sender_status[i]` in `ReshardCoordinator::ExecutePoolReshard`):
      -- records `sender_status = status`; does NOT call `registry_->Complete`.
      some
        { s with
          sendResults := (srcUnit, false) :: s.sendResults
          sendFailed := true }

  | .finishSendBarrier =>
    if s.phase != .dispatching || !allReported s.srcUnits s.sendResults then
      none
    else if !s.sendFailed then
      -- All senders succeeded (`for (const absl::Status& status : sender_status)` through `return absl::OkStatus()`).
      some { s with phase := .succeeded }
    else
      -- `if (!status.ok()) return status;` in `for (const absl::Status& status : sender_status)`:
      match policy with
      | .shipping =>
        -- Shipping code returns `sender_status` directly WITHOUT disarming
        -- `dst_units` AND WITHOUT calling `registry_->AbandonClaim`!
        some { s with phase := .aborted }
      | .disarmAndAbandonOnFailure =>
        some
          { s with
            phase := .aborted
            dstStates := s.dstStates.map DstUnitState.disarm
            reg := abandonIfClaimed s.reg s.claimOwner }

/-- System wrapper for a given `(policy, srcUnits, dstUnits, claimOwner)`. -/
def sys (policy : CleanupPolicy) (srcUnits dstUnits : List Nat) (claimOwner : Nat) :
    System CoordState Ev where
  init := init srcUnits dstUnits claimOwner
  step := step policy

end CoordState


end TpuSyncVerify.Transfer.Reshard.Coordinator

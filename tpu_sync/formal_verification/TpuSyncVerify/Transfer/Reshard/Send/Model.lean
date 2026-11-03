import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `Send`: state, events, `step`

Part of `TpuSyncVerify.Transfer.Reshard.Send`; the module docstring and source map live in `Send.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

/-- Policy for when `ReshardSendSession` unregisters its active plan:
`shipping` unregisters immediately inside `RecordPushCompletion` when
`finalizing_` is latched and in `EndOp` when an externally finished session
drains its last op; `onSettle` always defers unregistration to `EndOp` when
`in_flight_ == 1 && finalizing_` (matching `ReshardReceiveSession::FinishPoolH2d`
and `ReshardReceiveSession::EndRecvOpLocked`). -/
inductive UnregisterPolicy where
  | shipping
  | onSettle
  deriving Repr, DecidableEq

inductive SendStage where
  | pending
  | d2h
  | pushing (active : Nat)
  | aborted
  deriving Repr, DecidableEq

namespace SendStage

/-- Number of `in_flight_` ops held by a pool in this stage. -/
def opCount : SendStage → Nat
  | .pending => 0
  | .d2h => 1
  | .pushing k => k
  | .aborted => 0

@[simp] theorem bne_self (s : SendStage) : (s != s) = false := by cases s <;> simp
@[simp] theorem beq_self (s : SendStage) : (s == s) = true := by cases s <;> simp

end SendStage

structure SendPool where
  peers : Nat
  stage : SendStage := .pending
  deriving Repr, DecidableEq

namespace SendPool

/-- Number of `(pool, peer)` completions still expected from this pool on a
non-failing run. -/
def remainingSlots (p : SendPool) : Nat :=
  match p.stage with
  | .pending | .d2h => p.peers
  | .pushing k => k
  | .aborted => 0

end SendPool

structure Send where
  pools : List SendPool
  life : Lifecycle := { inFlight := 1 }
  execActive : Bool := true
  remainingPushes : Nat
  planUnregistered : Bool := false
  published : Option Bool := none
  deriving Repr, DecidableEq

namespace Send

def sumPeers : List Nat → Nat
  | [] => 0
  | n :: ns => n + sumPeers ns

def poolOps : List SendPool → Nat
  | [] => 0
  | p :: ps => p.stage.opCount + poolOps ps

def sumRemainingSlots : List SendPool → Nat
  | [] => 0
  | p :: ps => p.remainingSlots + sumRemainingSlots ps

/-- Every pool in `pools` has positive peer count (`CountPoolReshardSendSlots`,
`ReshardSendSession::Create`, `ReshardSendSession::ExecutePush`). -/
def allPosPeers (pools : List SendPool) : Bool :=
  pools.all fun p => decide (0 < p.peers)

def validPools (pools : List SendPool) : Bool :=
  !pools.isEmpty && allPosPeers pools

def validSchedule (peerCounts : List Nat) : Bool :=
  !peerCounts.isEmpty && peerCounts.all fun n => decide (0 < n)

/-- Every pool has been issued by the `ExecutePush` loop. -/
def allIssued (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage != .pending

/-- No pool has aborted its D2H or peer push launch. -/
def noAborted (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage != .aborted

/-- Every pool launched its peer pushes and all of them completed (`stage == .pushing 0`). -/
def allPushed (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage == .pushing 0

/-- Initial state after `ReshardSendSession::Create` and the opening lock of
`ReshardSendSession::ExecutePush` (`++in_flight_; manager_ = &manager;`). -/
def init (peerCounts : List Nat) : Send :=
  { pools := peerCounts.map fun n => { peers := n, stage := .pending }
    life := { inFlight := 1 }
    execActive := true
    remainingPushes := sumPeers peerCounts }

/-- `ReshardSendSession::EndOp`: if this is the last op (`in_flight_ == 1`) of a
finalizing session, unregisters the active plan before decrementing `in_flight_`
and calling `SettleLocked()`. -/
def endOp (s : Send) : Send :=
  let unreg :=
    if s.life.inFlight == 1 && s.life.draining && !s.life.done then true
    else s.planUnregistered
  { s with
    life := s.life.endOpLocked
    planUnregistered := unreg }

/-- `ReshardSendSession::RecordPushCompletion`: on the first error or when
`--remaining_pool_peer_pushes_ == 0`, latches `finalizing_ = true` (and under
`UnregisterPolicy.shipping`, unregisters the active plan immediately). -/
def recordPushCompletion (pol : UnregisterPolicy) (ok : Bool) (s : Send) : Send :=
  if s.life.done = true ∨ s.life.draining = true then s
  else if !ok then
    { s with
      life := s.life.finishLocked false
      planUnregistered := if pol == .shipping then true else s.planUnregistered }
  else if s.remainingPushes <= 1 then
    { s with
      remainingPushes := 0
      life := s.life.finishLocked true
      planUnregistered := if pol == .shipping then true else s.planUnregistered }
  else
    { s with remainingPushes := s.remainingPushes - 1 }

inductive Ev where
  | d2hIssue (p : Nat) (ok : Bool)
  | execFinish
  | d2hDone (p : Nat) (ok : Bool)
  | pushDone (p : Nat) (ok : Bool)
  | cancel
  | publish
  deriving Repr, DecidableEq

/-- Per-pool body of `ReshardSendSession::ExecutePush`. -/
def d2hIssue (pol : UnregisterPolicy) (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  if !s.execActive then none
  else match s.pools[p]? with
  | some pool =>
    if pool.stage != .pending then none
    else if ok then
      some { s with
        pools := s.pools.set p { pool with stage := .d2h }
        life := { s.life with inFlight := s.life.inFlight + 1 } }
    else
      let s₁ := recordPushCompletion pol false
        { s with
          pools := s.pools.set p { pool with stage := .aborted }
          execActive := false }
      some (endOp s₁)
  | none => none

/-- Final `EndOp()` of `ReshardSendSession::ExecutePush` after the pool loop completes. -/
def execFinish (s : Send) : Option Send :=
  if s.execActive = true ∧ allIssued s.pools = true then
    some (endOp { s with execActive := false })
  else none

/-- `pool_future.OnReady` callback + `ReshardSendSession::StartPoolPush` + `EndOp()`. -/
def d2hDone (pol : UnregisterPolicy) (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .d2h then none
    else if !ok then
      let s₁ := recordPushCompletion pol false
        { s with pools := s.pools.set p { pool with stage := .aborted } }
      some (endOp s₁)
    else if s.life.done = true ∨ s.life.draining = true then
      some (endOp { s with pools := s.pools.set p { pool with stage := .aborted } })
    else
      some (endOp { s with
        pools := s.pools.set p { pool with stage := .pushing pool.peers }
        life := { s.life with inFlight := s.life.inFlight + pool.peers } })
  | none => none

/-- `AsyncPush(...).OnReady` callback for one peer of pool `p` in `ReshardSendSession::StartPoolPush`. -/
def pushDone (pol : UnregisterPolicy) (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  match s.pools[p]? with
  | some pool =>
    match pool.stage with
    | .pushing (k + 1) =>
      let s₁ := recordPushCompletion pol ok
        { s with pools := s.pools.set p { pool with stage := .pushing k } }
      some (endOp s₁)
    | _ => none
  | none => none

/-- External `ReshardSendSession::Finish` (`KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer`,
`KVCacheManagerWithTransfer::CompleteReadWithDetails`). -/
def cancel (s : Send) : Option Send :=
  some { s with life := s.life.finishLocked false }

/-- `KVCacheManagerWithTransfer::CompleteReadWithDetails` publishes a settled
session (reported through `RecordPoolReshardTerminalLocked`). -/
def publish (s : Send) : Option Send :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk }
  else none

def step (pol : UnregisterPolicy := .shipping) (s : Send) : Ev → Option Send
  | .d2hIssue p ok => s.d2hIssue pol p ok
  | .execFinish => s.execFinish
  | .d2hDone p ok => s.d2hDone pol p ok
  | .pushDone p ok => s.pushDone pol p ok
  | .cancel => s.cancel
  | .publish => s.publish

def sys (peerCounts : List Nat) (pol : UnregisterPolicy := .shipping) : System Send Ev :=
  ⟨init peerCounts, step pol⟩

end Send

end TpuSyncVerify.Transfer.Reshard

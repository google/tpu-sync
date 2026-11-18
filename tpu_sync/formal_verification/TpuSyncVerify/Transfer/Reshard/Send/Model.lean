import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `Send`: state, events, `step`

Part of `TpuSyncVerify.Transfer.Reshard.Send`; the module docstring and source map live in `Send.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

inductive SendStage where
  | pending
  | d2h
  | pushing (active : Nat)
  | aborted
  deriving Repr, DecidableEq, Hashable

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
  deriving Repr, DecidableEq, Hashable

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
  published : Option Bool := none
  deriving Repr, DecidableEq, Hashable

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
`CreateReshardSendSession`, `ReshardSendSession::ExecutePushPipeline`). -/
def allPosPeers (pools : List SendPool) : Bool :=
  pools.all fun p => decide (0 < p.peers)

def validPools (pools : List SendPool) : Bool :=
  !pools.isEmpty && allPosPeers pools

def validSchedule (peerCounts : List Nat) : Bool :=
  !peerCounts.isEmpty && peerCounts.all fun n => decide (0 < n)

/-- Every pool has been issued by the `ExecutePushPipeline` loop. -/
def allIssued (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage != .pending

/-- Every pool is still `.pending`: no pool has been reached by the
`ExecutePushPipeline` loop yet (`Start` before its pipeline thread runs). -/
def allPending (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage == .pending

/-- No pool has aborted its D2H or peer push launch. -/
def noAborted (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage != .aborted

/-- Every pool launched its peer pushes and all of them completed (`stage == .pushing 0`). -/
def allPushed (pools : List SendPool) : Bool :=
  pools.all fun p => p.stage == .pushing 0

/-- Initial state after `CreateReshardSendSession` and the opening lock of
`ReshardSendSession::Start` (`++in_flight_;`). -/
def init (peerCounts : List Nat) : Send :=
  { pools := peerCounts.map fun n => { peers := n, stage := .pending }
    life := { inFlight := 1 }
    execActive := true
    remainingPushes := sumPeers peerCounts }

/-- `ReshardSendSession::EndOp`: the `while (true)` scope decrements `in_flight_`
and, when the last op of a finalizing session ends, calls `SettleLocked()`
(`ReleaseLeasesLocked`, `done_ = true`). -/
def endOp (s : Send) : Send :=
  { s with life := s.life.endOpLocked }

/-- `ReshardSendSession::RecordPushCompletion`: on the first error or when
`--remaining_pool_peer_pushes_ == 0`, latches `finalizing_ = true` (and
`staging_layer_manager_.Shutdown()`), while sibling pools and peers may still
have ops in flight. -/
def recordPushCompletion (ok : Bool) (s : Send) : Send :=
  if s.life.done = true ∨ s.life.draining = true then s
  else if !ok then
    { s with life := s.life.finishLocked false }
  else if s.remainingPushes <= 1 then
    { s with
      remainingPushes := 0
      life := s.life.finishLocked true }
  else
    { s with remainingPushes := s.remainingPushes - 1 }

inductive Ev where
  | startFail
  | d2hIssue (p : Nat) (ok : Bool)
  | execFinish
  | d2hDone (p : Nat) (ok : Bool)
  | pushDone (p : Nat) (ok : Bool)
  | cancel
  | publish
  deriving Repr, DecidableEq

/-- `ReshardSendSession::Start` failing synchronously before its pipeline thread
runs: `ValidatePushPreconditions` returned an error, or `std::thread` threw
`std::system_error`. Either way `RecordPushCompletion(status)` latches
`finalizing_` and the `absl::Cleanup` `EndOp()` ends the caller op; no pool was
touched. -/
def startFail (s : Send) : Option Send :=
  if s.execActive = true ∧ allPending s.pools = true then
    some (endOp (recordPushCompletion false { s with execActive := false }))
  else none

/-- Per-pool body of `ReshardSendSession::ExecutePushPipeline`. -/
def d2hIssue (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  if !s.execActive then none
  else match s.pools[p]? with
  | some pool =>
    if pool.stage != .pending then none
    else if ok then
      some { s with
        pools := s.pools.set p { pool with stage := .d2h }
        life := { s.life with inFlight := s.life.inFlight + 1 } }
    else
      let s₁ := recordPushCompletion false
        { s with
          pools := s.pools.set p { pool with stage := .aborted }
          execActive := false }
      some (endOp s₁)
  | none => none

/-- The pipeline thread's `EndOp()` after `ReshardSendSession::ExecutePushPipeline`
returns from a completed pool loop. -/
def execFinish (s : Send) : Option Send :=
  if s.execActive = true ∧ allIssued s.pools = true then
    some (endOp { s with execActive := false })
  else none

/-- `pool_future.OnReady` callback + `ReshardSendSession::StartPoolPush` + `EndOp()`. -/
def d2hDone (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .d2h then none
    else if !ok then
      let s₁ := recordPushCompletion false
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
def pushDone (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  match s.pools[p]? with
  | some pool =>
    match pool.stage with
    | .pushing (k + 1) =>
      let s₁ := recordPushCompletion ok
        { s with pools := s.pools.set p { pool with stage := .pushing k } }
      some (endOp s₁)
    | _ => none
  | none => none

/-- External `ReshardSendSession::Finish` (`KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer`,
`KVCacheManagerWithTransfer::CompleteReadWithDetails`,
`KVCacheManagerWithTransfer::UnregisterActivePlan`): `if (done_ || finalizing_) return;`
comes first, so a send that is already finalizing keeps its status. -/
def cancel (s : Send) : Option Send :=
  some { s with life := s.life.finishOnceLocked false }

/-- `KVCacheManagerWithTransfer::CompleteReadWithDetails` publishes a settled
session (`RetireSettledSessionLocked`, reported through
`RecordPoolReshardTerminalLocked`). -/
def publish (s : Send) : Option Send :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk }
  else none

def step (s : Send) : Ev → Option Send
  | .startFail => s.startFail
  | .d2hIssue p ok => s.d2hIssue p ok
  | .execFinish => s.execFinish
  | .d2hDone p ok => s.d2hDone p ok
  | .pushDone p ok => s.pushDone p ok
  | .cancel => s.cancel
  | .publish => s.publish

def sys (peerCounts : List Nat) : System Send Ev :=
  ⟨init peerCounts, step⟩

end Send

end TpuSyncVerify.Transfer.Reshard

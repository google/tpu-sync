import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `Recv`: state, events, `step`

Part of `TpuSyncVerify.Transfer.Reshard.Receive`; the module docstring and source map live in `Receive.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

inductive PoolStage where
  | waiting
  | received
  | queued
  | issued
  | ready
  | completed
  | aborted
  deriving Repr, DecidableEq, Hashable

namespace PoolStage

/-- Stages that hold one unit of `in_flight_` (from
`in_flight_ += static_cast<int32_t>(to_launch.size())` in
`ExecuteEligiblePoolH2ds` until `EndRecvOpLocked()` or `EndOp()`). -/
def holdsOp : PoolStage → Bool
  | .queued | .issued | .ready => true
  | _ => false

/-- Stages in `h2d_launched_pools_` (`ReshardReceiveSession`, inserted in
`CollectEligiblePoolH2dsLocked`). -/
def launched : PoolStage → Bool
  | .waiting | .received => false
  | _ => true

/-- Remaining drain steps needed for a single pool to release its in-flight op. -/
def drainSteps : PoolStage → Nat
  | .queued => 1
  | .issued => 2
  | .ready => 1
  | _ => 0

@[simp] theorem bne_self (s : PoolStage) : (s != s) = false := by cases s <;> rfl
@[simp] theorem beq_self (s : PoolStage) : (s == s) = true := by cases s <;> rfl

end PoolStage

structure Pool where
  rank : Nat
  stage : PoolStage := .waiting
  deriving Repr, DecidableEq, Hashable

/-- Policy for `EndIncomingPush` (`block_transport_session.cc`) on a
`ReshardReceiveSession`: `finishOnError`, the shipping code, calls
`session->Finish(status)` before `session->EndOp()` when the incoming push
failed (`if (!status.ok())`), for every session type; `endOpOnly` is the mutant
that calls `EndOp()` unconditionally (`trace_failed_incoming_push` shows what
that leaks). -/
inductive PushEndPolicy where
  | finishOnError
  | endOpOnly
  deriving Repr, DecidableEq

structure Recv where
  pools : List Pool
  life : Lifecycle := {}
  finalizing : Bool := false
  durationRecorded : Bool := false
  published : Option Bool := none
  pushes : Nat := 0
  deriving Repr, DecidableEq, Hashable

namespace Recv

/-- Number of pools currently holding an `in_flight_` op. -/
def poolOps : List Pool → Nat
  | [] => 0
  | p :: ps => (if p.stage.holdsOp then 1 else 0) + poolOps ps

/-- Total remaining pool drain steps across `pools`. -/
def poolDrainSteps : List Pool → Nat
  | [] => 0
  | p :: ps => p.stage.drainSteps + poolDrainSteps ps

/-- `completed_pool_indices_ == expected_pool_indices_`
(`RecordPoolH2dResultLocked`). -/
def allCompleted (pools : List Pool) : Bool :=
  pools.all fun p => p.stage == .completed

/-- Every expected pool of order rank strictly below `r` has completed its H2D
upload (`CollectEligiblePoolH2dsLocked`). -/
def prereqsCompleted (pools : List Pool) (r : Nat) : Bool :=
  pools.all fun q => !(q.rank < r) || q.stage == .completed

/-- Executable form of `RankWaveOrder`: every launched pool has all strictly
lower-rank pools completed. -/
def rankWaveOrder (pools : List Pool) : Bool :=
  pools.all fun p => !p.stage.launched || prereqsCompleted pools p.rank

/-- `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` pass: transitions
every `.received` pool satisfying `canLaunch p.rank` to `.queued` and counts
how many were launched. -/
def launchPass (canLaunch : Nat → Bool) : List Pool → List Pool × Nat
  | [] => ([], 0)
  | p :: ps =>
    let (ps', n) := launchPass canLaunch ps
    if p.stage == .received && canLaunch p.rank then
      ({ p with stage := .queued } :: ps', n + 1)
    else
      (p :: ps', n)

/-- Initial receiver armed for a plan whose transferred pools have order ranks
`ranks` (`ReshardReceiveSession::ReshardReceiveSession`). -/
def init (ranks : List Nat) : Recv :=
  { pools := ranks.map fun r => { rank := r, stage := .waiting } }

/-- First critical section of `ReshardReceiveSession::ExecuteEligiblePoolH2ds`:
if not done, draining, or finalizing, launches all eligible `.received` pools
and adds `to_launch.size()` to `in_flight_`. -/
def launchEligible (s : Recv) : Recv :=
  if s.life.done = true ∨ s.life.draining = true ∨ s.finalizing = true then s
  else
    let (pools', k) := launchPass (prereqsCompleted s.pools) s.pools
    { s with pools := pools', life := { s.life with inFlight := s.life.inFlight + k } }

inductive Ev where
  | pushBegin
  | pushEnd (ok : Bool)
  | onPoolReceived (p : Nat)
  | h2dIssue (p : Nat) (ok : Bool)
  | h2dReady (p : Nat)
  | h2dDone (p : Nat) (ok : Bool)
  | cancel
  | publish
  deriving Repr, DecidableEq

/-- `ReshardReceiveSession::TryBeginOp` from `BeginIncomingPush`
(`block_transport_session.cc`, after `delegate->GetSession(uuid)`). -/
def pushBegin (s : Recv) : Option Recv :=
  s.life.beginOp.map fun l => { s with life := l, pushes := s.pushes + 1 }

/-- `EndIncomingPush` (`block_transport_session.cc`). Under `.finishOnError`
(shipping), a failed push calls `session->Finish(status)` before
`session->EndOp()`; under `.endOpOnly` (mutant), only `EndOp()` runs. -/
def pushEnd (pol : PushEndPolicy) (ok : Bool) (s : Recv) : Option Recv :=
  if s.pushes = 0 then none
  else
    let l := if pol == .finishOnError && !ok then s.life.finishLocked false else s.life
    some { s with life := l.endOpLocked, pushes := s.pushes - 1 }

/-- `ReshardReceiveSession::OnPoolReceived` (reached as
`OnDataReceived(pool_idx, {}, {})` from
`BlockTransportServerPushSession::CompleteIncomingPush` on the stream's session):
records `p` in `started_pool_indices_` (`RecordPoolReceivedLocked`) and calls
`ExecuteEligiblePoolH2ds`. -/
def onPoolReceived (p : Nat) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if s.life.done = true ∨ pool.stage != .waiting then none
    else
      let s₁ := { s with pools := s.pools.set p { pool with stage := .received } }
      some (launchEligible s₁)
  | none => none

/-- Per-pool body of `ReshardReceiveSession::ExecuteEligiblePoolH2ds`. Re-checks
`if (done_ || draining_)` under `mu_`; if set, calls `EndRecvOpLocked()` and
skips the upload. Otherwise resolves the pool's host buffer
(`LookupStagingSlot`) and dispatches `base_->H2dPoolBlocks`: on a synchronous
failure (empty destination list, no leased buffer, `H2dPoolBlocks` error) calls
`FinishPoolH2d(pool_idx, status)` and `EndOp()`; on success attaches the
completion to `future.OnReady`. -/
def h2dIssue (p : Nat) (ok : Bool) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .queued then none
    else if s.life.done = true ∨ s.life.draining = true then
      some { s with
        pools := s.pools.set p { pool with stage := .aborted }
        life := s.life.endOpLocked }
    else if ok then
      some { s with pools := s.pools.set p { pool with stage := .issued } }
    else
      let l := if s.finalizing then s.life else s.life.finishLocked false
      some { s with
        pools := s.pools.set p { pool with stage := .aborted }
        finalizing := true
        life := l.endOpLocked }
  | none => none

/-- Device finishes pool `p`'s H2D copy; the `future.OnReady` continuation has
not run yet. -/
def h2dReady (p : Nat) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage == .issued then
      some { s with pools := s.pools.set p { pool with stage := .ready } }
    else none
  | none => none

/-- `ExecuteEligiblePoolH2ds`'s `future.OnReady` callback calling
`FinishPoolH2d` and `EndOp`. -/
def h2dDone (p : Nat) (ok : Bool) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .ready then none
    else if s.life.done = true then
      some { s with
        pools := s.pools.set p { pool with stage := .aborted }
        life := s.life.endOpLocked }
    else if s.finalizing = true then
      some { s with
        pools := s.pools.set p { pool with stage := .aborted }
        life := s.life.endOpLocked }
    else if !ok then
      some { s with
        pools := s.pools.set p { pool with stage := .aborted }
        finalizing := true
        life := (s.life.finishLocked false).endOpLocked }
    else
      let pools₁ := s.pools.set p { pool with stage := .completed }
      if allCompleted pools₁ then
        some { s with
          pools := pools₁
          finalizing := true
          durationRecorded := true
          life := (s.life.finishLocked true).endOpLocked }
      else
        let s₁ := launchEligible { s with pools := pools₁ }
        some { s₁ with life := s₁.life.endOpLocked }
  | none => none

/-- External `Finish(error)` (`~KVCacheManagerWithTransfer`,
`UnregisterActivePlan`, `CompleteReadWithDetails`, a staging wait that timed
out in `AcquireStagingSlot`). -/
def cancel (s : Recv) : Option Recv :=
  some { s with life := s.life.finishLocked false }

/-- `CompleteReadWithDetails` retires a settled session
(`RetireSettledSessionLocked`: `session->Done()`, reported through
`RecordPoolReshardTerminalLocked`, erased from `sessions_`). -/
def publish (s : Recv) : Option Recv :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk }
  else none

def step (pol : PushEndPolicy := .finishOnError) (s : Recv) : Ev → Option Recv
  | .pushBegin => s.pushBegin
  | .pushEnd ok => s.pushEnd pol ok
  | .onPoolReceived p => s.onPoolReceived p
  | .h2dIssue p ok => s.h2dIssue p ok
  | .h2dReady p => s.h2dReady p
  | .h2dDone p ok => s.h2dDone p ok
  | .cancel => s.cancel
  | .publish => s.publish

/-- System for a reshard receiver with pool order ranks `ranks` under `pol`. -/
def sys (ranks : List Nat) (pol : PushEndPolicy := .finishOnError) : System Recv Ev :=
  ⟨init ranks, step pol⟩

end Recv

end TpuSyncVerify.Transfer.Reshard

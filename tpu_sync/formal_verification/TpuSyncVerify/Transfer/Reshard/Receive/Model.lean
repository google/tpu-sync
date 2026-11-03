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
  deriving Repr, DecidableEq

namespace PoolStage

/-- Stages that hold one unit of `in_flight_` (from
`in_flight_ += static_cast<int32_t>(to_launch.size())` in
`ExecuteEligiblePoolH2ds` until `EndRecvOpLocked()` or `EndRecvOp()`). -/
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
  deriving Repr, DecidableEq

/-- Policy for `hooks.end_incoming_push` on `ReshardReceiveSession`
(`KVCacheManagerWithTransfer::InitializeBaseHooks`): `finishOnError`, the
shipping code, mirrors the `TransferReceiveSession` branch by calling
`reshard_session->Finish(status)` before `reshard_session->EndRecvOp()` when
the incoming push failed; `endOpOnly` is the mutant that calls `EndRecvOp()`
unconditionally (`trace_failed_incoming_push` shows what that leaks). -/
inductive PushEndPolicy where
  | finishOnError
  | endOpOnly
  deriving Repr, DecidableEq

structure Recv where
  pools : List Pool
  life : Lifecycle := {}
  finalizing : Bool := false
  networkCompleted : Bool := false
  planUnregistered : Bool := false
  durationRecorded : Bool := false
  published : Option Bool := none
  pushes : Nat := 0
  deriving Repr, DecidableEq

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

/-- `ReshardReceiveSession::AllH2dDoneLocked`: every future pushed into
`h2d_futures_` (`h2d_futures_.push_back(future)` in `ExecuteEligiblePoolH2ds`,
stage `.issued`) has become `IsReady()`. -/
def allH2dDone (pools : List Pool) : Bool :=
  pools.all fun p => p.stage != .issued

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

/-- `ReshardReceiveSession::IsReadyToComplete`. -/
def isReadyToComplete (s : Recv) : Bool :=
  s.networkCompleted && allH2dDone s.pools

/-- `KVCacheManagerWithTransfer::MaybeUnregisterSettledReshardRecv`
(`ReshardReceiveSession::TakePendingUnregister`): once `done_` is set, claims
`unregister_on_settle_` and unregisters the plan. -/
def maybeUnregisterSettled (s : Recv) : Recv :=
  if s.life.done = true then { s with planUnregistered := true } else s

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
  | pollReady
  | cancel
  | publish
  deriving Repr, DecidableEq

/-- `ReshardReceiveSession::TryBeginRecvOp` from `hooks.begin_incoming_push`. -/
def pushBegin (s : Recv) : Option Recv :=
  s.life.beginOp.map fun l => { s with life := l, pushes := s.pushes + 1 }

/-- `hooks.end_incoming_push` (`KVCacheManagerWithTransfer::InitializeBaseHooks`).
Under `.finishOnError` (shipping), a failed push calls
`reshard_session->Finish(status)` before `reshard_session->EndRecvOp()`; under
`.endOpOnly` (mutant), only `EndRecvOp()` runs. If the session is now
settled, `reshard_session->TakePendingUnregister(&generation)` unregisters the
plan (`UnregisterSettledPlan`). -/
def pushEnd (pol : PushEndPolicy) (ok : Bool) (s : Recv) : Option Recv :=
  if s.pushes = 0 then none
  else
    let l := if pol == .finishOnError && !ok then s.life.finishLocked false else s.life
    some (maybeUnregisterSettled { s with life := l.endOpLocked, pushes := s.pushes - 1 })

/-- `ReshardReceiveSession::OnPoolReceived`: records `p` in
`started_pool_indices_` (`RecordPoolReceivedLocked`) and calls
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
skips the upload. Otherwise dispatches `base_->H2dPoolBlocks`: on synchronous
failure calls `FinishPoolH2d`, `EndRecvOp`, and
`MaybeUnregisterSettledReshardRecv`; on success pushes the future into
`h2d_futures_` (`h2d_futures_.push_back(future)`). -/
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
      some (maybeUnregisterSettled { s with
        pools := s.pools.set p { pool with stage := .aborted }
        finalizing := true
        life := l.endOpLocked })
  | none => none

/-- Device finishes pool `p`'s H2D copy; its future in `h2d_futures_` becomes
`IsReady()`. -/
def h2dReady (p : Nat) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage == .issued then
      some { s with pools := s.pools.set p { pool with stage := .ready } }
    else none
  | none => none

/-- `ExecuteEligiblePoolH2ds`'s `future.OnReady` callback calling
`FinishPoolH2d`, `EndRecvOp`, and `MaybeUnregisterSettledReshardRecv`. -/
def h2dDone (p : Nat) (ok : Bool) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .ready then none
    else if s.life.done = true then
      some (maybeUnregisterSettled { s with
        pools := s.pools.set p { pool with stage := .aborted }
        life := s.life.endOpLocked })
    else if s.finalizing = true then
      some (maybeUnregisterSettled { s with
        pools := s.pools.set p { pool with stage := .aborted }
        life := s.life.endOpLocked })
    else if !ok then
      some (maybeUnregisterSettled { s with
        pools := s.pools.set p { pool with stage := .aborted }
        finalizing := true
        life := (s.life.finishLocked false).endOpLocked })
    else
      let pools₁ := s.pools.set p { pool with stage := .completed }
      if allCompleted pools₁ then
        some (maybeUnregisterSettled { s with
          pools := pools₁
          finalizing := true
          networkCompleted := true
          durationRecorded := true
          life := (s.life.finishLocked true).endOpLocked })
      else
        let s₁ := launchEligible { s with pools := pools₁ }
        some (maybeUnregisterSettled { s₁ with life := s₁.life.endOpLocked })
  | none => none

/-- `CompleteReadWithDetails` polls a non-draining session and sees
`IsReadyToComplete()`. -/
def pollReady (s : Recv) : Option Recv :=
  if s.life.draining = false ∧ isReadyToComplete s = true then
    some { s with life := s.life.finishLocked true }
  else none

/-- External `Finish(error)` (`~KVCacheManagerWithTransfer`,
`UnregisterActivePlan`, `CompleteReadWithDetails`). -/
def cancel (s : Recv) : Option Recv :=
  some { s with life := s.life.finishLocked false }

/-- `CompleteReadWithDetails` publishes a settled session (`session->Done()`,
reported through `RecordPoolReshardTerminalLocked`). -/
def publish (s : Recv) : Option Recv :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk, planUnregistered := true }
  else none

def step (pol : PushEndPolicy := .finishOnError) (s : Recv) : Ev → Option Recv
  | .pushBegin => s.pushBegin
  | .pushEnd ok => s.pushEnd pol ok
  | .onPoolReceived p => s.onPoolReceived p
  | .h2dIssue p ok => s.h2dIssue p ok
  | .h2dReady p => s.h2dReady p
  | .h2dDone p ok => s.h2dDone p ok
  | .pollReady => s.pollReady
  | .cancel => s.cancel
  | .publish => s.publish

/-- System for a reshard receiver with pool order ranks `ranks` under `pol`. -/
def sys (ranks : List Nat) (pol : PushEndPolicy := .finishOnError) : System Recv Ev :=
  ⟨init ranks, step pol⟩

end Recv

end TpuSyncVerify.Transfer.Reshard

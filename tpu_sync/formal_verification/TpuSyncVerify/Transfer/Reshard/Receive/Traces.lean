import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.Receive.Progress

/-! # `Recv`: replay of the C++ tests, bounded searches, mutants

Part of `TpuSyncVerify.Transfer.Reshard.Receive`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ## Replay and bounded search -/


def tr_rank1_arrives_before_rank0 : List Ev :=
  [.pushBegin, .onPoolReceived 1, .pushEnd true,
   .pushBegin, .onPoolReceived 0, .pushEnd true,
   .h2dIssue 0 true, .h2dReady 0, .h2dDone 0 true,
   .h2dIssue 1 true, .h2dReady 1, .h2dDone 1 true, .publish]

/-- Ranks `[0, 1]`: pool 1 arrives first and waits in `.received` until pool 0 completes H2D and `FinishPoolH2d` queues Wave 1. -/
theorem trace_rank1_arrives_before_rank0 :
    let s := sys [0, 1]
    -- Pool 1 (rank 1) arrives first: recorded in `started_pool_indices_`, not launched.
    (s.run (tr_rank1_arrives_before_rank0.take 3)).map
      (fun r => (r.pools.map Pool.stage, r.life.inFlight)) =
      some ([.waiting, .received], 0) ∧
    -- Pool 1 cannot issue an H2D copy yet.
    s.run (tr_rank1_arrives_before_rank0.take 3 ++ [.h2dIssue 1 true]) = none ∧
    -- Pool 0 (rank 0) arrives: only pool 0 is queued (`inFlight = 1`).
    (s.run (tr_rank1_arrives_before_rank0.take 6)).map
      (fun r => (r.pools.map Pool.stage, r.life.inFlight)) =
      some ([.queued, .received], 1) ∧
    -- Pool 0's H2D completes: `FinishPoolH2d(0, OK)` queues pool 1 before releasing pool 0's op.
    (s.run (tr_rank1_arrives_before_rank0.take 9)).map
      (fun r => (r.pools.map Pool.stage, r.life.inFlight, r.life.done)) =
      some ([.completed, .queued], 1, false) ∧
    -- Pool 1 now issues, finishes, settles (its `plan()` goes with `done_`), and publishes `done_recving`.
    (s.run tr_rank1_arrives_before_rank0).map
      (fun r => (r.life.done, r.life.hasStaging, r.durationRecorded, r.published)) =
      some (true, false, true, some true) := by
  decide

def tr_batch_launch_on_rank_wave : List Ev :=
  [.onPoolReceived 0, .h2dIssue 0 true, .onPoolReceived 1, .onPoolReceived 2,
   .h2dReady 0, .h2dDone 0 true,
   .h2dIssue 1 true, .h2dIssue 2 true, .h2dReady 1, .h2dDone 1 true,
   .h2dReady 2, .h2dDone 2 true, .publish]

/-- Ranks `[0, 1, 1]`: completing rank 0 queues both waiting rank-1 pools in one batch (`ExecuteEligiblePoolH2ds`). -/
theorem trace_batch_launch_on_rank_wave :
    let s := sys [0, 1, 1]
    (s.run (tr_batch_launch_on_rank_wave.take 6)).map
      (fun r => (r.pools.map Pool.stage, r.life.inFlight, r.life.done)) =
      some ([.completed, .queued, .queued], 2, false) ∧
    (s.run tr_batch_launch_on_rank_wave).map
      (fun r => (r.life.done, r.published, r.durationRecorded)) =
      some (true, some true, true) := by
  decide

def tr_cancel_between_batch_queue_and_issue : List Ev :=
  [.onPoolReceived 0, .onPoolReceived 1, .h2dIssue 0 true, .cancel, .h2dIssue 1 true,
   .h2dReady 0, .h2dDone 0 true, .publish]

/-- Per-pool `if (done_ || draining_)` re-check in `ExecuteEligiblePoolH2ds`: `cancel` mid-loop aborts the second queued pool while the first drains. -/
theorem trace_cancel_between_batch_queue_and_issue :
    let s := sys [0, 0]
    (s.run (tr_cancel_between_batch_queue_and_issue.take 5)).map
      (fun r => (r.pools.map Pool.stage, r.life.inFlight, r.life.done, r.life.hasStaging)) =
      some ([.issued, .aborted], 1, false, true) ∧
    (s.run tr_cancel_between_batch_queue_and_issue).map
      (fun r => (r.life.done, r.life.hasStaging, r.durationRecorded, r.published)) =
      some (true, false, false, some false) := by
  decide

/-- Receiver plan/pool validation (`ReshardReceiveSession::ValidatePlan`, `ValidateCommonPoolReshardPlan`, `ValidatePoolBlockBounds`). -/
def validPlan (isSender : Bool) (ranks : List Nat) : Bool :=
  !isSender && !ranks.isEmpty

def validPlanPoolsAndBlocks (enableHostStaging : Bool) (numPools numBlocks : Nat)
    (poolIndices : List Nat) (dtypeMatches : Bool) (dstBlockIds : List Nat) : Bool :=
  enableHostStaging &&
  !poolIndices.isEmpty &&
  dtypeMatches &&
  decide poolIndices.Nodup &&
  poolIndices.all (fun p => decide (p < numPools)) &&
  !dstBlockIds.isEmpty &&
  dstBlockIds.all (fun b => decide (b < numBlocks))

def tr_parity_1pool : List Ev :=
  [.pushBegin, .onPoolReceived 0, .pushEnd true,
   .h2dIssue 0 true, .h2dReady 0, .h2dDone 0 true, .publish]

def tr_parity_2pool : List Ev :=
  [.pushBegin, .onPoolReceived 0, .pushEnd true,
   .h2dIssue 0 true, .h2dReady 0, .h2dDone 0 true,
   .pushBegin, .onPoolReceived 1, .pushEnd true,
   .h2dIssue 1 true, .h2dReady 1, .h2dDone 1 true, .publish]

/-- Replays `ReshardReceiveSessionTest.AcceptsCanonicalPlanOnExplicitPools`, `ReshardReceiveSessionTest.AcceptsImplicitPools`,
`ReshardReceiveSessionTest.HasNoTagPolicy`, `ReshardReceiveSessionTest.RejectsMissingIdentityAndPoolFields`,
`ReshardReceiveSessionTest.RejectsOutOfRangeDuplicateAndDtypeMismatch`, `ReshardReceiveSessionTest.RejectsBlockIdsOutsideDeclaredPool`,
`ReshardReceiveSessionTest.GroupScopesDestinationBlocks`, and `PoolReshardValidationTest.DeviceOnlyRejectionAtPublicEntryPoints`. -/
theorem trace_reshard_receive_session_test_parity :
    validPlanPoolsAndBlocks false 2 16 [0] true [1] = false ∧
    validPlan true [0] = false ∧
    validPlan false [] = false ∧
    validPlan false [0] = true ∧
    validPlanPoolsAndBlocks true 2 16 [99] true [1] = false ∧
    validPlanPoolsAndBlocks true 2 16 [0, 0] true [1] = false ∧
    validPlanPoolsAndBlocks true 2 16 [0] false [1] = false ∧
    validPlanPoolsAndBlocks true 2 16 [0] true [16] = false ∧
    validPlanPoolsAndBlocks true 2 16 [0] true [1, 2] = true ∧
    ((sys [0]).run tr_parity_1pool).map
      (fun r => (r.life.done, r.life.hasStaging, r.published)) =
      some (true, false, some true) ∧
    ((sys [0, 0]).run tr_parity_2pool).map
      (fun r => (r.pools.map Pool.stage, r.life.done, r.life.hasStaging, r.published)) =
      some ([.completed, .completed], true, false, some true) := by
  decide

def tr_duration_ok : List Ev :=
  [.onPoolReceived 0, .h2dIssue 0 true, .h2dReady 0, .h2dDone 0 true, .publish]

def tr_duration_fail : List Ev :=
  [.onPoolReceived 0, .h2dIssue 0 true, .h2dReady 0, .h2dDone 0 false, .publish]

/-- Replays `PoolReshardRecvTest.FinishPoolReshardRecvRecordsDurationMetric` and
`PoolReshardRecvTest.FinishPoolReshardRecvDoesNotRecordMetricOnFailure`: duration is recorded iff the final pool's H2D succeeds. -/
theorem trace_duration_metric_on_success_and_failure :
    let s := sys [0]
    ((s.run tr_duration_ok).map
      (fun r => (r.published, r.durationRecorded)) =
      some (some true, true)) ∧
    ((s.run tr_duration_fail).map
      (fun r => (r.published, r.durationRecorded)) =
      some (some false, false)) := by
  decide

def tr_failed_incoming_push : List Ev :=
  [.pushBegin, .pushEnd false, .publish]

/-- Replays `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline`: shipping `PushEndPolicy.finishOnError`
settles and releases staging on `pushEnd false`, whereas `.endOpOnly` leaves the session open until deadline. -/
theorem trace_failed_incoming_push :
    -- `.endOpOnly` mutant: after `[.pushBegin, .pushEnd false]`, session is still open and holds staging/plan.
    ((sys [0] .endOpOnly).run (tr_failed_incoming_push.take 2)).map
      (fun r => (r.life.draining, r.life.done, r.life.hasStaging)) =
      some (false, false, true) ∧
    (sys [0] .endOpOnly).run tr_failed_incoming_push = none ∧
    -- Shipping (`finishOnError`): `tr_failed_incoming_push` settles at once and publishes failure.
    ((sys [0] .finishOnError).run tr_failed_incoming_push).map
      (fun r => (r.life.done, r.life.hasStaging, r.published)) =
      some (true, false, some false) := by
  decide

def events (numPools : Nat) : List Ev :=
  [.pushBegin, .pushEnd true, .pushEnd false, .cancel, .publish] ++
  (List.range numPools).flatMap fun p =>
    [.onPoolReceived p, .h2dIssue p true, .h2dIssue p false,
     .h2dReady p, .h2dDone p true, .h2dDone p false]

instance (pol : PushEndPolicy) (s : Recv) : Decidable (Safe pol s) := by
  unfold Safe SettleSafe NoRetiredCallback StagingIntegrity SettlesPromptly RankWaveOrder Publication NoOpLeak
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorem check the same predicate. -/
def violates (pol : PushEndPolicy) (s : Recv) : Bool := !decide (Safe pol s)

theorem violates_iff (pol : PushEndPolicy) (s : Recv) : violates pol s = false ↔ Safe pol s := by
  simp [violates]


/-- Mutant 1 (`onPoolReceivedIgnoringRank`): omitting the `pool_order_ranks_` check in `CollectEligiblePoolH2dsLocked` violates `RankWaveOrder`. -/
def onPoolReceivedIgnoringRank (p : Nat) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if s.life.done = true ∨ pool.stage != .waiting then none
    else
      let pools₁ := s.pools.set p { pool with stage := .received }
      if s.life.done = true ∨ s.life.draining = true ∨ s.finalizing = true then
        some { s with pools := pools₁ }
      else
        let (pools', k) := launchPass (fun _ => true) pools₁
        some { s with pools := pools', life := { s.life with inFlight := s.life.inFlight + k } }
  | none => none

#guard (match ModelCheck.check ⟨init [0, 1], fun s e => match e with
          | .onPoolReceived p => onPoolReceivedIgnoringRank p s
          | e => step .finishOnError s e⟩ (events 2) (violates .finishOnError) 4 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant 2 (`h2dIssueLeak`): skipping a queued upload on `done_ || draining_` without `EndRecvOpLocked()` violates `NoOpLeak`. -/
def h2dIssueLeak (p : Nat) (ok : Bool) (s : Recv) : Option Recv :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .queued then none
    else if s.life.done = true ∨ s.life.draining = true then
      some { s with pools := s.pools.set p { pool with stage := .aborted } }
    else s.h2dIssue p ok
  | none => none

#guard (match ModelCheck.check ⟨init [0], fun s e => match e with
          | .h2dIssue p ok => h2dIssueLeak p ok s
          | e => step .finishOnError s e⟩ (events 1) (violates .finishOnError) 5 with
        | .counterexample _ => true
        | _ => false)

end Recv

end TpuSyncVerify.Transfer.Reshard

import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.Send.Progress

/-! # `Send`: replay of the C++ tests, bounded searches, mutants

Part of `TpuSyncVerify.Transfer.Reshard.Send`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Replay and bounded search -/

/-- Sender source-span validation (`ReshardSendSession::ValidatePlan`: `StridedSpanFitsBlock`, `StridedSpanFitsRegions`, `RegionCoversOffset`, `RegionsCoverRange`). -/
def validSenderSpan (blockStride liveBytes tokenStride tokenLive offset size : Nat) : Bool :=
  if size = 0 then false
  else if tokenStride = 0 ∨ tokenLive = 0 then
    decide (offset + size ≤ liveBytes)
  else
    decide (offset + size ≤ blockStride) &&
    decide (offset / tokenStride = (offset + size - 1) / tokenStride) &&
    decide (offset % tokenStride + size ≤ tokenLive)

def tr_parity_send : List Ev :=
  [.d2hIssue 0 true, .execFinish, .d2hDone 0 true, .pushDone 0 true, .publish]

/-- Replays `PoolReshardSendTest.SenderWithNoBytesForAnyTransferredPoolRefusedUpFront`,
`ReshardSendSessionTest.RejectsMissingIdentityAndPoolFields`, `ReshardSendSessionTest.RejectsOverflowingSenderSpan`,
`ReshardSendSessionTest.RejectsSenderSpanBetweenPackedTokenRegionsInAliasedStorage`, `ReshardSendSessionTest.AcceptsCanonicalPlanOnExplicitPools`,
`ReshardSendSessionTest.AcceptsImplicitPools`, and `ReshardSendSessionTest.IgnoresReceiverCoverageGap`. -/
theorem trace_reshard_send_session_test_parity :
    validSchedule [] = false ∧
    validSchedule [0] = false ∧
    validSenderSpan 128 96 0 0 64 64 = false ∧
    validSenderSpan 256 128 64 32 16 32 = false ∧
    validSenderSpan 256 128 64 32 64 32 = true ∧
    validSchedule [1] = true ∧
    ((sys [1]).run tr_parity_send).map
      (fun r => (r.life.done, r.life.hasStaging, r.published)) =
      some (true, false, some true) := by
  decide

def tr_multi_pool_multi_peer_send : List Ev :=
  [.d2hIssue 0 true, .d2hIssue 1 true, .execFinish, .d2hDone 0 true,
   .pushDone 0 true, .pushDone 0 true, .d2hDone 1 true, .pushDone 1 true, .publish]

/-- Peer counts `[2, 1]` (`CountPoolReshardSendSlots = 3`): settles and publishes `done_sending` only after all 3 `(pool, peer)` pushes complete. -/
theorem trace_multi_pool_multi_peer_send :
    let s := sys [2, 1]
    -- ExecutePushPipeline issues D2H for pool 0 and pool 1, then the thread calls EndOp (`inFlight = 2`).
    (s.run (tr_multi_pool_multi_peer_send.take 3)).map
      (fun r => (r.pools.map SendPool.stage, r.life.inFlight, r.remainingPushes, r.life.done)) =
      some ([.d2h, .d2h], 2, 3, false) ∧
    -- Pool 0's D2H completes (`StartPoolPush` adds 2 peer ops and releases 1 D2H op: `inFlight = 3`).
    (s.run (tr_multi_pool_multi_peer_send.take 4)).map
      (fun r => (r.pools.map SendPool.stage, r.life.inFlight, r.remainingPushes)) =
      some ([.pushing 2, .d2h], 3, 3) ∧
    -- Pool 0's two peer pushes and pool 1's D2H complete (`remainingPushes = 1`, still not done).
    (s.run (tr_multi_pool_multi_peer_send.take 7)).map
      (fun r => (r.pools.map SendPool.stage, r.life.inFlight, r.remainingPushes, r.life.done)) =
      some ([.pushing 0, .pushing 1], 1, 1, false) ∧
    -- Pool 1's peer push completes: `remainingPushes` hits 0, session settles and publishes `done_sending`.
    (s.run tr_multi_pool_multi_peer_send).map
      (fun r => (r.life.done, r.life.hasStaging, r.published)) =
      some (true, false, some true) := by
  decide

def tr_start_fails_fast : List Ev := [.startFail, .publish]

/-- Replays `ReshardSendSessionTest.StartFailsFastWithoutTransportServer`: synchronous `ValidatePushPreconditions` failure in `Start` settles immediately. -/
theorem trace_start_fails_fast :
    ((sys [1]).run (tr_start_fails_fast.take 1)).map
      (fun r => (r.life.done, r.life.statusOk, r.life.hasStaging, r.execActive,
                 r.pools.map SendPool.stage)) =
      some (true, false, false, false, [SendStage.pending]) ∧
    ((sys [1]).run tr_start_fails_fast).map (fun r => r.published) = some (some false) := by
  decide

def tr_cancel_settles_on_last_endOp : List Ev :=
  [.d2hIssue 0 true, .execFinish, .d2hDone 0 true, .cancel,
   .pushDone 0 true, .pushDone 0 true, .publish]

/-- External `Finish(error)` while ops are in flight (cf. `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidCopy`):
latches `finalizing_` and keeps staging/plan live until the last `EndOp` drains `in_flight_` to `0`. -/
theorem trace_cancel_settles_on_last_endOp :
    let s := sys [2]
    -- After cancel with 2 peer pushes in flight: draining, not done, staging held.
    (s.run (tr_cancel_settles_on_last_endOp.take 4)).map
      (fun r => (r.life.draining, r.life.done, r.life.inFlight, r.life.hasStaging)) =
      some (true, false, 2, true) ∧
    -- First peer push finishes (`inFlight = 1`): still not done.
    (s.run (tr_cancel_settles_on_last_endOp.take 5)).map
      (fun r => (r.life.done, r.life.inFlight, r.life.hasStaging)) =
      some (false, 1, true) ∧
    -- Second peer push finishes (`inFlight = 0`): `EndOp` settles.
    (s.run tr_cancel_settles_on_last_endOp).map
      (fun r => (r.life.done, r.life.hasStaging, r.published)) =
      some (true, false, some false) := by
  decide

/-! `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeSettling`,
`ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidCopy`, `ReshardSendSessionTest.PooledPolicyKeepsIdlePeerConnections`,
and `ReshardSendSessionTest.CreateRejectsUnknownDataConnReuseValue`: `kPerTransfer` release in `EndOp` folds into atomic `endOp` (`Send.lean` A4). -/

def tr_partial_failure_keeps_plan : List Ev :=
  [.d2hIssue 0 true, .d2hIssue 1 true, .execFinish,
   .d2hDone 0 true, .d2hDone 1 true, .pushDone 0 false]

/-- Partial push failure (`RecordPushCompletion`): pool 0's error latches `finalizing_` while pool 1's in-flight push keeps `plan()` live until `EndOp`. -/
theorem trace_partial_failure_keeps_plan :
    ((sys [1, 1]).run tr_partial_failure_keeps_plan).map
      (fun r => (r.life.done, r.life.draining, r.life.inFlight, r.life.statusOk)) =
      some (false, true, 1, false) ∧
    ((sys [1, 1]).run (tr_partial_failure_keeps_plan ++ [.pushDone 1 true])).map
      (fun r => (r.life.done, r.life.inFlight, r.life.hasStaging)) =
      some (true, 0, false) := by
  decide

def tr_finish_after_last_push : List Ev := [.pushDone 0 true, .cancel, .publish]

/-- `ReshardSendSession::Finish` after the last push (`if (done_ || finalizing_) return;`): a racing `cancel` is ignored and `published = some true`. -/
theorem trace_finish_after_last_push :
    ((sys [1]).runFrom (((sys [1]).run (tr_parity_send.take 3)).getD (init [1]))
        tr_finish_after_last_push).map
      (fun r => (r.life.done, r.life.statusOk, r.published)) =
      some (true, true, some true) := by
  decide

def events (numPools : Nat) : List Ev :=
  [.startFail, .execFinish, .cancel, .publish] ++
  (List.range numPools).flatMap fun p =>
    [.d2hIssue p true, .d2hIssue p false,
     .d2hDone p true, .d2hDone p false,
     .pushDone p true, .pushDone p false]

instance (s : Send) : Decidable (Safe s) := by
  unfold Safe SettleSafe NoRetiredCallback StagingIntegrity SettlesPromptly ActiveOrDraining Publication NoOpLeak
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorem check the same predicate. -/
def violates (s : Send) : Bool := !decide (Safe s)

theorem violates_iff (s : Send) : violates s = false ↔ Safe s := by
  simp [violates]

#guard ModelCheck.check (sys [2]) (events 1) violates 12 = .safe
#guard ModelCheck.check (sys [1, 1]) (events 2) violates 12 = .safe
#guard ModelCheck.check (sys [2, 1]) (events 2) violates 12 = .safe


/-- Mutant 1 (`initOvercountedSlots`): overcounting `remaining_pool_peer_pushes_` as `pools * peers = 4` instead of `3` violates `ActiveOrDraining`. -/
def initOvercountedSlots : Send :=
  { (init [2, 1]) with remainingPushes := 4 }

#guard (match ModelCheck.check ⟨initOvercountedSlots, step⟩
          (events 2) violates 9 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant 2 (`cancelRecordsStatusFirst`): recording `status_` in `ReshardSendSession::Finish` before `if (done_ || finalizing_) return;`
flips a completed send to `published = some false` on a racing `cancel`. -/
def cancelRecordsStatusFirst (s : Send) : Option Send :=
  some { s with life := s.life.finishLocked false }

def stepCancelRecordsStatusFirst (s : Send) : Ev → Option Send
  | .cancel => cancelRecordsStatusFirst s
  | e => step s e

theorem trace_finish_after_last_push_mutant :
    ((⟨init [1], stepCancelRecordsStatusFirst⟩ : System Send Ev).run
        (tr_parity_send.take 3 ++ tr_finish_after_last_push)).map
      (fun r => (r.life.done, r.life.statusOk, r.published)) =
      some (true, false, some false) := by
  decide

/-- Mutant 3 (`d2hDoneLeakOnDraining`): omitting `EndOp()` when `StartPoolPush` exits early on `done_ || finalizing_` violates `NoOpLeak`. -/
def d2hDoneLeakOnDraining (p : Nat) (ok : Bool) (s : Send) : Option Send :=
  match s.pools[p]? with
  | some pool =>
    if pool.stage != .d2h then none
    else if ok && (s.life.done || s.life.draining) then
      some { s with pools := s.pools.set p { pool with stage := .aborted } }
    else s.d2hDone p ok
  | none => none

#guard (match ModelCheck.check ⟨init [1], fun s e => match e with
          | .d2hDone p ok => d2hDoneLeakOnDraining p ok s
          | e => step s e⟩ (events 1) violates 5 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant 4 (`startFailWithoutRecord`): returning a `Start` error without `RecordPushCompletion` leaves `in_flight_ == 0` while `!finalizing_`, violating `ActiveOrDraining`. -/
def startFailWithoutRecord (s : Send) : Option Send :=
  if s.execActive = true ∧ allPending s.pools = true then
    some (endOp { s with execActive := false })
  else none

#guard (match ModelCheck.check ⟨init [1], fun s e => match e with
          | .startFail => startFailWithoutRecord s
          | e => step s e⟩ (events 1) violates 2 with
        | .counterexample _ => true
        | _ => false)

end Send

end TpuSyncVerify.Transfer.Reshard

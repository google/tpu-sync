import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.Send.Progress

/-! # `Send`: replay of the C++ tests, bounded searches, mutants

Part of `TpuSyncVerify.Transfer.Reshard.Send`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Replay and bounded search

Concrete traces, checked by `decide`, covering multi-pool multi-peer slot
counting (`CountPoolReshardSendSlots`), `ReshardSendSessionTest` and
`PoolReshardSendTest` parity, the synchronous failure of
`ReshardSendSession::Start`, external deadline or cancellation settling in
`ReshardSendSession::EndOp`, and a partial failure
(`ReshardSendSession::RecordPushCompletion` latching `finalizing_` while a
sibling pool is still in flight). -/

/-- Sender source-span validation (`ReshardSendSession::ValidatePlan`, invoked
by `CreateReshardSendSession`; `StridedSpanFitsBlock` and
`StridedSpanFitsRegions` in `utils.h`; `RegionCoversOffset` and
`RegionsCoverRange` in `pool_layout.cc`):
- If `tokenStride = 0` or `tokenLive = 0` (flat pool), requires `0 < size` and
  `offset + size ≤ liveBytes`.
- If `0 < tokenStride` and `0 < tokenLive` (packed token-strided pool in aliased
  storage, `RegionCoversOffset`), requires `0 < size`,
  `offset + size ≤ blockStride`,
  `offset / tokenStride = (offset + size - 1) / tokenStride` (cannot cross a
  token boundary), and `(offset % tokenStride) + size ≤ tokenLive` (must lie
  within the token's live sub-region). -/
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

/-- Replay of `ReshardSendSessionTest` and
`PoolReshardSendTest.SenderWithNoBytesForAnyTransferredPoolRefusedUpFront`:
1. `PoolReshardSendTest.SenderWithNoBytesForAnyTransferredPoolRefusedUpFront` and
   `ReshardSendSessionTest.RejectsMissingIdentityAndPoolFields`: when
   `CountPoolReshardSendSlots` is `0` (`ns = []` or `ns = [0]`, because the
   sender's only schedule entry targets a non-transferred pool group),
   `validSchedule` rejects the session up front (`validSchedule [0] = false`,
   `CreateReshardSendSession`).
2. `ReshardSendSessionTest.RejectsOverflowingSenderSpan`: an overflowing source
   span (modeled with `offset = 64, size = 64` on `liveBytes = 96`,
   `64 + 64 > 96`) is rejected (`false`).
3. `ReshardSendSessionTest.RejectsSenderSpanBetweenPackedTokenRegionsInAliasedStorage`:
   on a token-strided pool (modeled with `blockStride = 256, liveBytes = 128,
   tokenStride = 64, tokenLive = 32`), span `[16, 48)` (`offset = 16, size = 32`)
   straddles into the non-live gap `[32, 64)` and is rejected (`false`), whereas
   `[64, 96)` (`offset = 64, size = 32`, token 1's live region) is accepted
   (`true`).
4. `ReshardSendSessionTest.AcceptsCanonicalPlanOnExplicitPools`,
   `ReshardSendSessionTest.AcceptsImplicitPools`, and
   `ReshardSendSessionTest.IgnoresReceiverCoverageGap`: those tests accept a
   1-pool, 1-peer sender schedule (`ns = [1]`, `validSchedule [1] = true`);
   run through the session it completes D2H + push to settle with
   `published = some true`. The tests themselves only create the session
   (`CreateStatus` → `CreateReshardSendSession` → `ValidatePlan`) and never
   start it. -/
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

/-- Two active pools with peer counts `[2, 1]` (`CountPoolReshardSendSlots = 3`,
e.g. an attention group pushing to 2 peers and a state group pushing to 1 peer,
`CreateReshardSendSession`, `CountPoolReshardSendSlots`): `ExecutePushPipeline`
issues both D2H copies and the pipeline thread drops its caller op
(`execFinish`), each pool's `StartPoolPush` fans out to its group's peers, and
the session settles only when all 3 `(pool, peer)` pushes have completed. -/
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

/-- `ReshardSendSessionTest.StartFailsFastWithoutTransportServer`:
`ReshardSendSession::Start` on a manager whose transport server was stopped
fails `ValidatePushPreconditions` synchronously; `RecordPushCompletion`
latches `finalizing_` with the error and the `absl::Cleanup` `EndOp()` settles
the send at once (`Done()`, `GetStatus()` failed, `plan()` empty), before any
pool was staged. The poll then publishes it as failed. -/
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

/-- External deadline or cancellation (`cancel` in
`KVCacheManagerWithTransfer::CompleteReadWithDetails`, `ReshardSendSession::Finish`)
while peer pushes are in flight: `Finish(error)` sets `finalizing_ = true`
while `in_flight_ == 2`, so the session stays registered (its `plan()` is still
resolvable) while its pushes drain; the first peer push to finish decrements
`in_flight_` to `1`; the last peer push's `ReshardSendSession::EndOp` releases
the staging leases and settles the session, at which point `plan()` returns
`absl::nullopt`.
`ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidCopy`
drives the one-pool version with the op held by the D2H copy instead of a
push: `FakeD2hBase::HoldNextCopy` parks pool 0's copy, `Finish(Cancelled)`
starts finalizing without settling (`plan()` still set), and
`FakeD2hBase::CompleteCopy` runs the `pool_future.OnReady` callback, whose
`StartPoolPush` returns at `if (done_ || finalizing_)` and whose `EndOp()`
settles the send. -/
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
`ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidCopy`,
`ReshardSendSessionTest.PooledPolicyKeepsIdlePeerConnections`, and
`ReshardSendSessionTest.CreateRejectsUnknownDataConnReuseValue`: TCP
data-connection pooling (`TPU_RAIDEN_DATA_CONN_REUSE`,
`tpu_sync/kv_cache/data_connection_policy.{h,cc}`, `CreateReshardSendSession`,
`ReshardSendSession::EndOp`) is orthogonal to the session state machine;
Assumption A4 above justifies folding the per-transfer connection-release window
in `ReshardSendSession::EndOp` (`while (true)`) into the atomic `endOp`
transition. -/

def tr_partial_failure_keeps_plan : List Ev :=
  [.d2hIssue 0 true, .d2hIssue 1 true, .execFinish,
   .d2hDone 0 true, .d2hDone 1 true, .pushDone 0 false]

/-- Partial failure (`ReshardSendSession::RecordPushCompletion`): when pool 0's
push fails (`pushDone 0 false`) while pool 1's push is still in flight,
`RecordPushCompletion` latches `finalizing_` at once but the session is not
done (`inFlight = 1`), so `plan()` still returns the plan and pool 1's push
keeps resolving its chunks (`KVCacheManagerBase::GetBlockChunks`) and staging
slots (`ResolveStagingSlot`) against it until its own `EndOp` settles the send. -/
theorem trace_partial_failure_keeps_plan :
    ((sys [1, 1]).run tr_partial_failure_keeps_plan).map
      (fun r => (r.life.done, r.life.draining, r.life.inFlight, r.life.statusOk)) =
      some (false, true, 1, false) ∧
    ((sys [1, 1]).run (tr_partial_failure_keeps_plan ++ [.pushDone 1 true])).map
      (fun r => (r.life.done, r.life.inFlight, r.life.hasStaging)) =
      some (true, 0, false) := by
  decide

def tr_finish_after_last_push : List Ev := [.pushDone 0 true, .cancel, .publish]

/-- `Finish` racing the last push's `EndOp` (`ReshardSendSession::Finish`,
`ReshardSendSession::RecordPushCompletion`): after the last scheduled push
completed (`RecordPushCompletion` latched `finalizing_` with `status_` OK), a
`Finish(Cancelled)` is ignored (`if (done_ || finalizing_) return;` precedes
the status update), so the session settles OK. In the C++ the `Finish` can
land between `RecordPushCompletion` and `EndOp`; `pushDone` folds those two
into one event, and `cancel` on a finalizing session is the identity either
side of `EndOp`. No C++ test drives this window. -/
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


/-- Mutant 1 (`initOvercountedSlots`, cf. `CreateReshardSendSession`):
initializing `remaining_pool_peer_pushes_` to `pools.size() * total_peers` (4
instead of `2 + 1 = 3`) leaves the session open with `in_flight_ == 0` and
`finalizing_ == false` after all 3 scheduled pushes complete, violating
`ActiveOrDraining`. -/
def initOvercountedSlots : Send :=
  { (init [2, 1]) with remainingPushes := 4 }

#guard (match ModelCheck.check ⟨initOvercountedSlots, step⟩
          (events 2) violates 9 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant 2 (`cancelRecordsStatusFirst` / `stepCancelRecordsStatusFirst`): a
`ReshardSendSession::Finish` that records `status_` before
`if (done_ || finalizing_) return;` (`Lifecycle.finishLocked`) lets a
`Finish(Cancelled)` racing the last push's `EndOp` turn a completed send into a
failed one. Not a `Safe` violation (the session still settles cleanly), but
`trace_finish_after_last_push` distinguishes it: `published = some false`
instead of `some true`. -/
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

/-- Mutant 3 (`d2hDoneLeakOnDraining`): `pool_future.OnReady` in
`ReshardSendSession::ExecutePushPipeline` omits `EndOp()` when
`ReshardSendSession::StartPoolPush` returns early on `done_ || finalizing_`,
leaking the D2H op (caught by `NoOpLeak`). -/
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

/-- Mutant 4 (`startFailWithoutRecord`): `ReshardSendSession::Start` returning
its precondition error without `RecordPushCompletion` would end the caller op
on a session that is neither finalizing nor pushing, violating
`ActiveOrDraining` (`in_flight_ == 0` with `finalizing_ == false`), and the
send would sit until the deadline sweep. -/
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

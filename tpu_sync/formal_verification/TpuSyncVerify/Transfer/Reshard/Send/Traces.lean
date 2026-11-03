import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.Send.Progress

/-! # `Send`: replay of upstream tests, bounded searches, mutants

Part of `TpuSyncVerify.Transfer.Reshard.Send`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Send

/-! ## Replay and bounded search

Concrete traces, checked by `decide`, covering multi-pool multi-peer slot
counting (`CountPoolReshardSendSlots`), `ReshardSendSessionTest` and
`PoolReshardSendTest` parity, external deadline or cancellation unregistration
in `ReshardSendSession::EndOp`, and the early plan unregistration of
`ReshardSendSession::RecordPushCompletion` when one pool fails while a sibling
pool is still in flight (contrasted below with an `EndOp`-only alternative). -/

/-- Sender source-span validation (`ReshardSendSession::ValidatePlan`, invoked
by `ReshardSendSession::Create`; `StridedSpanFitsBlock` and
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
   `ReshardSendSession::Create`).
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
   `published = some true`. -/
theorem trace_reshard_send_session_test_parity :
    validSchedule [] = false ∧
    validSchedule [0] = false ∧
    validSenderSpan 128 96 0 0 64 64 = false ∧
    validSenderSpan 256 128 64 32 16 32 = false ∧
    validSenderSpan 256 128 64 32 64 32 = true ∧
    validSchedule [1] = true ∧
    ((sys [1]).run tr_parity_send).map
      (fun r => (r.life.done, r.life.hasStaging, r.planUnregistered, r.published)) =
      some (true, false, true, some true) := by
  decide

def tr_multi_pool_multi_peer_send : List Ev :=
  [.d2hIssue 0 true, .d2hIssue 1 true, .execFinish, .d2hDone 0 true,
   .pushDone 0 true, .pushDone 0 true, .d2hDone 1 true, .pushDone 1 true, .publish]

/-- Two active pools with peer counts `[2, 1]` (`CountPoolReshardSendSlots = 3`,
e.g. an attention group pushing to 2 peers and a state group pushing to 1 peer,
`ReshardSendSession::Create`, `CountPoolReshardSendSlots`): `ExecutePush` issues
both D2H copies and drops its caller op (`execFinish`), each pool's
`StartPoolPush` fans out to its group's peers, and the session settles only when
all 3 `(pool, peer)` pushes have completed. -/
theorem trace_multi_pool_multi_peer_send :
    let s := sys [2, 1]
    -- ExecutePush issues D2H for pool 0 and pool 1, then calls EndOp (`inFlight = 2`).
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
      (fun r => (r.life.done, r.life.hasStaging, r.planUnregistered, r.published)) =
      some (true, false, true, some true) := by
  decide

def tr_cancel_unregisters_on_last_endOp : List Ev :=
  [.d2hIssue 0 true, .execFinish, .d2hDone 0 true, .cancel,
   .pushDone 0 true, .pushDone 0 true, .publish]

/-- External deadline or cancellation (`cancel` in
`KVCacheManagerWithTransfer::CompleteReadWithDetails`, `ReshardSendSession::Finish`)
while peer pushes are in flight: `Finish(error)` sets `finalizing_ = true`
without unregistering the plan while
`in_flight_ == 2`; the first peer push to finish decrements `in_flight_` to `1`
without unregistering; the last peer push's `ReshardSendSession::EndOp`
unregisters the plan, releases the staging leases, and settles the session.
`ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush`
drives the one-peer version of the last two steps (`cancel`, then the push
completing) from a session whose push op `ReshardSendSessionTestPeer::BeginPush`
took. -/
theorem trace_cancel_unregisters_on_last_endOp :
    let s := sys [2]
    -- After cancel with 2 peer pushes in flight: draining, but plan still registered and staging held.
    (s.run (tr_cancel_unregisters_on_last_endOp.take 4)).map
      (fun r => (r.life.draining, r.life.done, r.life.inFlight, r.planUnregistered)) =
      some (true, false, 2, false) ∧
    -- First peer push finishes (`inFlight = 1`): still not done, plan still registered.
    (s.run (tr_cancel_unregisters_on_last_endOp.take 5)).map
      (fun r => (r.life.done, r.life.inFlight, r.planUnregistered, r.life.hasStaging)) =
      some (false, 1, false, true) ∧
    -- Second peer push finishes (`inFlight = 0`): `EndOp` unregisters the plan and settles.
    (s.run tr_cancel_unregisters_on_last_endOp).map
      (fun r => (r.life.done, r.planUnregistered, r.life.hasStaging, r.published)) =
      some (true, true, false, some false) := by
  decide

/-! `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeSettling`,
`ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush`,
`ReshardSendSessionTest.PooledPolicyKeepsIdlePeerConnections`, and
`ReshardSendSessionTest.CreateRejectsUnknownDataConnReuseValue`: TCP
data-connection pooling (`TPU_RAIDEN_DATA_CONN_REUSE`,
`tpu_sync/kv_cache/data_connection_policy.{h,cc}`, `ReshardSendSession::Create`,
`ReshardSendSession::EndOp`) is orthogonal to the session state machine;
Assumption A4 above justifies folding the per-transfer connection-release window
in `ReshardSendSession::EndOp` (`while (true)`) into the atomic `endOp`
transition. -/

def tr_early_unregister_on_error : List Ev :=
  [.d2hIssue 0 true, .d2hIssue 1 true, .execFinish,
   .d2hDone 0 true, .d2hDone 1 true, .pushDone 0 false]

/-- Early plan unregistration on partial failure
(`ReshardSendSession::RecordPushCompletion` vs. `ReshardSendSession::EndOp`):
when pool 0's push fails (`pushDone 0 false`) while pool 1's push is still in
flight, `RecordPushCompletion` unregisters the active plan at once
(`planUnregistered = true` while `done = false` and `inFlight = 1`), so pool 1's
`StartPoolPush` resolves its chunks against an unregistered plan. -/
theorem trace_early_unregister_on_error :
    ((sys [1, 1]).run tr_early_unregister_on_error).map
      (fun r => (r.life.done, r.life.inFlight, r.planUnregistered)) =
      some (false, 1, true) := by
  decide

def events (numPools : Nat) : List Ev :=
  [.execFinish, .cancel, .publish] ++
  (List.range numPools).flatMap fun p =>
    [.d2hIssue p true, .d2hIssue p false,
     .d2hDone p true, .d2hDone p false,
     .pushDone p true, .pushDone p false]

instance (s : Send) : Decidable (Safe s) := by
  unfold Safe SettleSafe NoRetiredCallback StagingIntegrity SettlesPromptly ActiveOrDraining PlanIntegrity Publication NoOpLeak
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorem check the same predicate. -/
def violates (s : Send) : Bool := !decide (Safe s)

theorem violates_iff (s : Send) : violates s = false ↔ Safe s := by
  simp [violates]

#guard ModelCheck.check (sys [2]) (events 1) violates 12 = .safe
#guard ModelCheck.check (sys [1, 1]) (events 2) violates 12 = .safe
#guard ModelCheck.check (sys [2, 1]) (events 2) violates 12 = .safe


/-- Mutant 1 (`initOvercountedSlots`, cf. `ReshardSendSession::Create`):
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

/-- Mutant 2 (`endOpWithoutUnregister` / `stepWithoutEndOpUnregister`): omitting
the settle-time plan unregistration in `ReshardSendSession::EndOp` leaves
`plan_unregistered_ == false` when an externally cancelled session (`cancel`)
finishes draining, violating `PlanIntegrity`. -/
def endOpWithoutUnregister (s : Send) : Send :=
  { s with life := s.life.endOpLocked }

def stepWithoutEndOpUnregister (s : Send) : Ev → Option Send
  | .execFinish =>
    if s.execActive = true ∧ allIssued s.pools = true then
      some (endOpWithoutUnregister { s with execActive := false })
    else none
  | e => step s e

#guard (match ModelCheck.check ⟨init [1], stepWithoutEndOpUnregister⟩
          (events 1) violates 4 with
        | .counterexample _ => true
        | _ => false)

/-- Mutant 3 (`d2hDoneLeakOnDraining`): `pool_future.OnReady` in
`ReshardSendSession::ExecutePush` omits `EndOp()` when
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

/-! ### The `EndOp`-only unregistration alternative

`ReshardReceiveSession` unregisters its plan only when its last op ends
(`FinishPoolH2d` / `EndRecvOpLocked`); `ReshardSendSession::RecordPushCompletion`
unregisters as soon as `finalizing_` is latched. `stepUnregisterOnSettle` is
`step` with the send session doing what the receive session does — the plan
stays registered until `EndOp` settles the session. It is not the shipping
code. -/

/-- `RecordPushCompletion` without `UnregisterActivePlan`. -/
def recordPushCompletionOnSettle (ok : Bool) (s : Send) : Send :=
  { recordPushCompletion ok s with planUnregistered := s.planUnregistered }

def stepUnregisterOnSettle (s : Send) : Ev → Option Send
  | .d2hIssue p ok =>
    if !s.execActive then none
    else match s.pools[p]? with
    | some pool =>
      if pool.stage != .pending then none
      else if ok then step s (.d2hIssue p ok)
      else some (endOp (recordPushCompletionOnSettle false
        { s with pools := s.pools.set p { pool with stage := .aborted }, execActive := false }))
    | none => none
  | .d2hDone p ok =>
    match s.pools[p]? with
    | some pool =>
      if pool.stage != .d2h then none
      else if !ok then
        some (endOp (recordPushCompletionOnSettle false
          { s with pools := s.pools.set p { pool with stage := .aborted } }))
      else step s (.d2hDone p ok)
    | none => none
  | .pushDone p ok =>
    match s.pools[p]? with
    | some pool =>
      match pool.stage with
      | .pushing (k + 1) =>
        some (endOp (recordPushCompletionOnSettle ok
          { s with pools := s.pools.set p { pool with stage := .pushing k } }))
      | _ => none
    | none => none
  | e => step s e

/-- Under the alternative the plan stays registered while pool 1 is in flight
and is unregistered when pool 1's op ends the session. -/
theorem trace_unregister_on_settle_alternative :
    ((⟨init [1, 1], stepUnregisterOnSettle⟩ : System Send Ev).run tr_early_unregister_on_error).map
      (fun r => (r.life.done, r.life.inFlight, r.planUnregistered)) =
      some (false, 1, false) ∧
    ((⟨init [1, 1], stepUnregisterOnSettle⟩ : System Send Ev).run
        (tr_early_unregister_on_error ++ [.pushDone 1 true])).map
      (fun r => (r.life.done, r.life.inFlight, r.planUnregistered)) =
      some (true, 0, true) := by
  decide

/-- `Safe` and `planUnregistered = done` in every state the alternative reaches
on the 1- and 2-pool instances (exhaustive); shipping `sys` violates the
biconditional on a multi-pool failure (`planUnregistered = true` while sibling
ops drain). -/
def violatesPlanIffDone (s : Send) : Bool :=
  !decide (Safe s ∧ s.planUnregistered = s.life.done)

#guard ModelCheck.check (⟨init [2], stepUnregisterOnSettle⟩ : System Send Ev) (events 1)
  violatesPlanIffDone 12 = .safe
#guard ModelCheck.check (⟨init [1, 1], stepUnregisterOnSettle⟩ : System Send Ev) (events 2)
  violatesPlanIffDone 12 = .safe
#guard ModelCheck.check (⟨init [2, 1], stepUnregisterOnSettle⟩ : System Send Ev) (events 2)
  violatesPlanIffDone 12 = .safe
#guard (ModelCheck.check (sys [1, 1]) (events 2) violatesPlanIffDone 12 matches .counterexample _)

end Send

end TpuSyncVerify.Transfer.Reshard

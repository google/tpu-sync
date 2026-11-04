import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Invariant

/-! # `SubPlanChain`: replay traces, searches, mutants

Part of `TpuSyncVerify.Transfer.Reshard.SubPlanChain`; the module docstring and
source map live in `SubPlanChain.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard.SubPlanChain

open TpuSyncVerify
open TpuSyncVerify.Transfer.Reshard.Registry
open TpuSyncVerify.Transfer.Reshard.Coordinator (abandonIfClaimed)

/-! ## Part I — `PlanSubTransfers` and `SubPlanUuid` -/

/-- Replays `PlanSubTransfersTest.NoWindowsPlanOneTransfer` (`reshard_service_test.cc`):
empty `dst_window_blocks`, `{0}`, and `{4}` on 4 destination blocks all return
`{ count := 1, subPlans := [] }`. -/
theorem trace_plan_no_windows_plan_one_transfer :
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8, 9, 10] } =
      some { count := 1, subPlans := [] } ∧
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8, 9, 10], dstWindowBlocks := [0] } =
      some { count := 1, subPlans := [] } ∧
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8, 9, 10], dstWindowBlocks := [4] } =
      some { count := 1, subPlans := [] } := by
  decide

/-- Replays `PlanSubTransfersTest.WindowSplitsPagedTagInDestinationOrder`
(`reshard_service_test.cc`): 5 destination blocks `{7, 8, 9, 10, 11}` with
window `2` produce 3 ordered sub-transfers of offsets `0, 2, 4` and counts
`2, 2, 1`. -/
theorem trace_plan_window_splits_paged_tag_in_dst_order :
    planSubTransfers
        { numTags := 1
          dstDeviceBlockIds := [7, 8, 9, 10, 11]
          dstWindowBlocks := [2] } =
      some
        { count := 3
          subPlans :=
            [ [{ tagIndex := 0, offset := 0, count := 2, total := 5 }],
              [{ tagIndex := 0, offset := 2, count := 2, total := 5 }],
              [{ tagIndex := 0, offset := 4, count := 1, total := 5 }] ] } := by
  decide

/-- Replays `PlanSubTransfersTest.UnwindowedTagsRideInTheLastSubTransfer`
(`reshard_service_test.cc`): 3 paged blocks windowed `1` per sub-transfer and
two unwindowed state tags sharing block `20` produce `count = 3`, with both
state tags riding whole (`offset = 0, count = 1, total = 0`) in the final
sub-transfer alongside the final paged slice. -/
theorem trace_plan_unwindowed_tags_ride_in_last_sub_transfer :
    planSubTransfers
        { numTags := 3
          dstDeviceBlockIds := [7, 8, 9, 20, 20]
          dstBlockCounts := [3, 1, 1]
          dstWindowBlocks := [1, 0, 0] } =
      some
        { count := 3
          subPlans :=
            [ [{ tagIndex := 0, offset := 0, count := 1, total := 3 }],
              [{ tagIndex := 0, offset := 1, count := 1, total := 3 }],
              [ { tagIndex := 0, offset := 2, count := 1, total := 3 },
                { tagIndex := 1, offset := 0, count := 1, total := 0 },
                { tagIndex := 2, offset := 0, count := 1, total := 0 } ] ] } := by
  decide

/-- Replays `PlanSubTransfersTest.WindowedTagSharingIdsDisablesWindows`
(`reshard_service_test.cc`): when windowed tag `0` (`{7, 8, 9}`) shares block `7`
with tag `1` (`{7}`), windowing collapses to `count = 1` so intra-plan aliased
page ordering (`RankWaveOrder`) is preserved; when only unwindowed tags `1` and
`2` share block `11`, windowing proceeds (`count = 2`). -/
theorem trace_plan_windowed_tag_sharing_ids_disables_windows :
    planSubTransfers
        { numTags := 2
          dstDeviceBlockIds := [7, 8, 9, 7]
          dstBlockCounts := [3, 1]
          dstWindowBlocks := [1, 0] } =
      some { count := 1, subPlans := [] } ∧
    (planSubTransfers
        { numTags := 3
          dstDeviceBlockIds := [7, 8, 9, 11, 11]
          dstBlockCounts := [3, 1, 1]
          dstWindowBlocks := [2, 0, 0] }).map (·.count) = some 2 := by
  decide

/-- Replays `PlanSubTransfersTest.MalformedWindowsFailClosed`
(`reshard_service_test.cc`): negative window (`{-1}`) or arity mismatch
(`{1, 1}` for 1 tag) returns `none` (`InvalidArgumentError`). -/
theorem trace_plan_malformed_windows_fail_closed :
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8], dstWindowBlocks := [-1] } = none ∧
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8], dstWindowBlocks := [1, 1] } = none := by
  decide

/-- Replays `SubPlanUuidTest.DerivedUuidsArePositiveAndDistinct`
(`reshard_service_test.cc`): `subPlanUuid uuid 0 = uuid`, all derived UUIDs for
`0 .. 63` are positive and pairwise distinct, and `subPlanUuid uuid 1 ≠ subPlanUuid (uuid + 1) 1`. -/
theorem trace_sub_plan_uuid_positive_and_distinct :
    let u : UInt64 := 0x7000123456789abc
    let derived := (List.range 64).map (subPlanUuid u)
    subPlanUuid u 0 = u ∧
    derived.all (fun x => x > 0 && x <= 0x7FFFFFFFFFFFFFFF) = true ∧
    derived.Nodup ∧
    subPlanUuid u 1 ≠ subPlanUuid (u + 1) 1 := by
  decide

/-! ## Part II — Worker chain folding -/

/-- Replays `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan`
(`pool_reshard_test.cc`): sub-plans `0`, `2`, `1` of `count = 3` settle out of
order; `done_recving` is inserted only once the third settles, and polling
again does not re-insert it. -/
theorem trace_recv_terminal_waits_for_last_sub_plan :
    let tr1 : List WorkerEv :=
      [ .registerSubPlan 5001 0 3, .settleSession 5001 true, .poll false,
        .registerSubPlan 5003 2 3, .settleSession 5003 true, .poll false ]
    let tr2 : List WorkerEv :=
      tr1 ++ [ .registerSubPlan 5002 1 3, .settleSession 5002 true, .poll false, .poll false ]
    (WorkerState.sys.run tr1).map (fun w => (w.doneReports, w.failReports)) = some (0, 0) ∧
    (WorkerState.sys.run tr2).map (fun w => (w.doneReports, w.failReports, w.chain.present)) =
      some (1, 0, false) := by
  decide

/-- Replays `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce`
(`pool_reshard_test.cc`): sub-plan `0` of `2` fails and `failed_recving` is
inserted once; sub-plan `1` is still seated afterwards and its clean settlement
is swallowed (`if (progress.failed) return;`). -/
theorem trace_first_sub_plan_failure_fails_once :
    let tr : List WorkerEv :=
      [ .registerSubPlan 5011 0 2, .settleSession 5011 false, .poll false,
        .registerSubPlan 5012 1 2, .settleSession 5012 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.failed)) =
      some (0, 1, true) := by
  decide

/-- Replays `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain`
(`pool_reshard_test.cc`): the first chain fails at sub-plan `0` (`6001`) while
sub-plan `1` (`6002`) is still active; the consumer retries under the same
`req_id` (`7001`, `7002`); sub-plan 0 of the retry re-anchors a fresh entry, the
stale `6002` settling is ignored (`!plan_uuids.contains(uuid)`), and the retry
inserts `done` once `7001` and `7002` settle. -/
theorem trace_retry_same_req_id_tracked_as_new_chain :
    let tr : List WorkerEv :=
      [ .registerSubPlan 6001 0 2, .settleSession 6001 false, .poll false,
        .registerSubPlan 6002 1 2,
        .registerSubPlan 7001 0 2,
        .settleSession 6002 true, .poll false,
        .settleSession 7001 true, .poll false,
        .registerSubPlan 7002 1 2, .settleSession 7002 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (1, 1) := by
  decide

/-- Replays `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout`
(`pool_reshard_test.cc`): sub-plan `0` of `2` settles OK (`5021`); when sub-plan
`1` never arrives and the entry is quiet for `timeout_s_` (`poll true`),
`SweepSubPlanChainsLocked` inserts `failed_recving`. -/
theorem trace_stalled_chain_fails_after_timeout :
    let tr : List WorkerEv :=
      [ .registerSubPlan 5021 0 2, .settleSession 5021 true, .poll false, .poll true ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (0, 1) := by
  decide

/-- Replays `SubPlanChainTest.SinglePlanRequestsReportAtOnce`
(`pool_reshard_test.cc`): `sub_plan_count <= 1` (`5031`) reports on settlement
without a `chains_` entry. -/
theorem trace_single_plan_reports_at_once :
    let tr : List WorkerEv :=
      [ .registerSubPlan 5031 0 1, .settleSession 5031 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.present)) =
      some (1, 0, false) := by
  decide

/-- A sender with spans in window `0` only of a `K = 2` request
(`ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries` dispatches rank
`0` once, stamped `sub_plan_index = 0, sub_plan_count = 2`): its single push
settles OK, `settled.size() = 1 < count = 2`, so no `done_sending` is inserted;
once the entry is quiet for `timeout_s_` the sweep inserts `failed_recving`. -/
theorem trace_sender_in_first_window_only :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9000 0 2, .settleSession 9000 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.settled)) =
      some (0, 0, [0]) ∧
    (WorkerState.sys.run (tr ++ [.poll true])).map (fun w => (w.doneReports, w.failReports)) =
      some (0, 1) := by
  decide

/-- A sender with spans in window `1` only (`sub_plan_index = 1`): its
registration default-constructs an unanchored entry (`anchored = false`), its
clean settlement counts `1 of 2`, and the sweep inserts `failed_recving`. -/
theorem trace_sender_in_last_window_only :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9001 1 2, .settleSession 9001 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.chain.anchored, w.chain.settled)) =
      some (0, false, [1]) ∧
    (WorkerState.sys.run (tr ++ [.poll true])).map (fun w => (w.doneReports, w.failReports)) =
      some (0, 1) := by
  decide

/-- Sub-plans `0` and `1` are both seated; sub-plan `0` fails and
`RecordPoolReshardTerminalLocked` inserts `failed_recving` while sub-plan `1`'s
session is still active (`active ≠ []`): the request's failure is published
before the chain's last session settles. -/
theorem trace_first_failure_reports_while_sibling_active :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7100 0 2, .registerSubPlan 7101 1 2,
        .settleSession 7100 false, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.failReports, w.active.map (·.uuid))) =
      some (1, [7101]) := by
  decide

/-- After the failed entry is erased by the sweep, a late sub-plan `1` of the
same request is seated into a fresh unanchored entry and swept into a second
`failed_recving` insertion (`failReports = 2`). -/
theorem trace_late_sub_plan_after_sweep_reports_failure_twice :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7300 0 2, .settleSession 7300 false, .poll false,
        .poll true,
        .registerSubPlan 7301 1 2, .settleSession 7301 true, .poll false, .poll true ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (0, 2) := by
  decide

/-! ### Search -/

instance (w : WorkerState) : Decidable (DoneAfterAllSubPlans w) := by
  unfold DoneAfterAllSubPlans; infer_instance

instance (w : WorkerState) : Decidable (WorkerSafe w) :=
  decidable_of_iff (DoneAfterAllSubPlans w) ⟨fun h => ⟨h⟩, fun h => h.1⟩

def workerViolates (w : WorkerState) : Bool := !decide (WorkerSafe w)

theorem workerViolates_iff (w : WorkerState) : workerViolates w = false ↔ WorkerSafe w := by
  simp [workerViolates]

/-- Two sub-plans of a `K = 2` request and a re-anchoring index-`0` sub-plan
(a retry under the same `req_id`), each settling either way; polls with and
without the stall sweep. -/
def workerCheckEvents : List WorkerEv :=
  [ .registerSubPlan 10 0 2, .registerSubPlan 11 1 2, .registerSubPlan 13 0 2,
    .settleSession 10 true, .settleSession 10 false,
    .settleSession 11 true, .settleSession 11 false,
    .settleSession 13 true, .settleSession 13 false,
    .poll false, .poll true ]

#guard ModelCheck.check WorkerState.sys workerCheckEvents workerViolates 32 = .safe

/-! ### Mutants -/

/-- Mutant: `RecordPoolReshardTerminalLocked` without the
`!plan_uuids.contains(uuid)` check counts a superseded chain's sub-plan into the
new chain. -/
def recordTerminalCountingStale (c : ChainProgress) (s : SubSession) : ChainProgress × Terminal :=
  recordTerminal { c with planUuids := insertNodup s.uuid c.planUuids } s

def stepCountStale (w : WorkerState) (e : WorkerEv) : Option WorkerState :=
  match e with
  | .poll stalled =>
    let settledNow := w.active.filter (·.done)
    let stillActive := w.active.filter (!·.done)
    let r := settledNow.foldl
      (fun (acc : WorkerState.SweepAcc) s =>
        let touched := acc.touched || s.count > 1
        match recordTerminalCountingStale acc.chain s with
        | (c', .done k idx) => { acc with chain := c', done := acc.done + 1, log := acc.log ++ [(k, idx)], touched }
        | (c', .failed) => { acc with chain := c', fail := acc.fail + 1, touched }
        | (c', _) => { acc with chain := c', touched })
      { chain := w.chain }
    let (c2, sweep) :=
      if stalled && !r.touched then sweepStalledChain r.chain stillActive else (r.chain, .notSwept)
    some
      { w with
        doneReports := w.doneReports + r.done
        failReports := w.failReports + r.fail + (if sweep == .stalled then 1 else 0)
        doneLog := w.doneLog ++ r.log
        active := stillActive
        chain := c2 }
  | e => WorkerState.step w e

/-- On the retry trace, the mutant inserts `done` for the retry as soon as the
stale `6002` and the new `7001` settle, while the retry's own sub-plan `1`
(`7002`) is still active; the shipping step waits for `7002`. -/
theorem mutant_counting_stale_sub_plan_reports_done_early :
    let tr : List WorkerEv :=
      [ .registerSubPlan 6001 0 2, .settleSession 6001 false, .poll false,
        .registerSubPlan 6002 1 2,
        .registerSubPlan 7001 0 2, .registerSubPlan 7002 1 2,
        .settleSession 6002 true, .settleSession 7001 true, .poll false ]
    ((⟨WorkerState.init, stepCountStale⟩ : System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.active.map (·.uuid))) = some (1, [7002]) ∧
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.active.map (·.uuid))) =
      some (0, [7002]) := by
  decide

/-- Mutant: `NoteSubPlanRegisteredLocked` without the fresh entry for
`sub_plan_index == 0`: a retry under the same `req_id` lands in the failed
entry and is never reported `done`. -/
def noteSubPlanRegisteredNoReanchor (c : ChainProgress) (uuid _index count : Nat) : ChainProgress :=
  if count <= 1 then c
  else
    let base : ChainProgress := if c.present then c else { present := true }
    { base with planUuids := insertNodup uuid base.planUuids, count := max base.count count }

def stepNoReanchor (w : WorkerState) (e : WorkerEv) : Option WorkerState :=
  match e with
  | .registerSubPlan uuid index count =>
    if w.seenUuids.contains uuid || index >= max 1 count then none
    else
      some
        { w with
          seenUuids := uuid :: w.seenUuids
          active := w.active ++ [{ uuid := uuid, index := index, count := count }]
          chain := noteSubPlanRegisteredNoReanchor w.chain uuid index count }
  | e => WorkerState.step w e

theorem mutant_no_reanchor_never_reports_retry_done :
    let tr : List WorkerEv :=
      [ .registerSubPlan 6001 0 2, .settleSession 6001 false, .poll false,
        .registerSubPlan 7001 0 2, .settleSession 7001 true, .poll false,
        .registerSubPlan 7002 1 2, .settleSession 7002 true, .poll false, .poll true ]
    ((⟨WorkerState.init, stepNoReanchor⟩ : System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.failReports)) = some (0, 1) ∧
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (1, 1) := by
  decide

/-! ## Part III — Coordinator multi-window chain -/

/-- Replays `ReshardStackTest.WindowedRequestChainsOneSubTransferPerWindow`
(`reshard_service_test.cc`): 2 source ranks `{0, 1}`, 4 single-page windows
`[[0], [0], [1], [1]]`, 1 destination `10`; each window arms `10` then dispatches
its owning rank under a single registry claim (`claimOwner = 1`), ending at
`transferStatus = .completed` with the registration still `.claimed` until both
ranks vote `Complete`. -/
theorem trace_windowed_request_chains_one_sub_transfer_per_window :
    let tr : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend, .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend, .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend, .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend ]
    ((CoordChainState.sys [0, 1] [[0], [0], [1], [1]] [10] 1).run tr).map
        (fun s => (s.transferStatus, s.reg.status)) = some (.completed, .claimed) := by
  decide

/-- Replays `ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries`
(`reshard_service_test.cc`): `K = 2` windows where window `0` dispatches both
ranks `[0, 1]` and window `1` dispatches only rank `[1]`. -/
theorem trace_window_clips_straddling_spans_at_window_boundaries :
    let tr : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .sendSrcResult 1 true, .finishWindowSend,
        .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend ]
    ((CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1).run tr).map
        (fun s => (s.transferStatus, s.activeSenders)) = some (.completed, [0, 1]) := by
  decide

/-- Replays `ReshardStackTest.WindowComposesWithPrefixClip`
(`reshard_service_test.cc`): prefix clip plus a 2-page window dispatches `[0, 1]`
in window `0` and `[1]` in window `1`; rank `0`, absent from window `1`, may
vote `Complete` as soon as its window-0 push settles, and the claim stays until
rank `1` votes after the chain ends. -/
theorem trace_window_composes_with_prefix_clip :
    let tr : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .sendSrcResult 1 true, .finishWindowSend,
        .settleDispatchedRecv 10, .senderVoteComplete 0,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend ]
    ((CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1).run tr).map
        (fun s => (s.transferStatus, s.reg.status, s.activeSenders)) =
      some (.completed, .claimed, [1]) ∧
    ((CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1).run (tr ++ [.senderVoteComplete 1])).map
        (fun s => (s.reg.status, s.reg.retired)) = some (.unknown, true) := by
  decide

/-- Replays `ReshardStackTest.ChainedArmRetriesWhileReceiverStagingIsBusy`
(`reshard_service_test.cc`): window `1`'s `PoolReshardRegisterRecv` is refused
twice (`ArmRefusedForBusyStaging`) while window `0`'s receiver still holds its
staging lease, `RunSubPlan` re-arms (`attempt + 1 < kChainArmAttempts`), and the
third arm succeeds once window `0` settles. -/
theorem trace_chained_arm_retries_while_receiver_staging_busy :
    let tr : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend,
        .startWindowPlan true,
        .armDstResult 10 .refusedBusy, .retryBusyArm,
        .armDstResult 10 .refusedBusy, .retryBusyArm,
        .settleDispatchedRecv 10,
        .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend ]
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1 (maxArmAttempts := 4)).run tr).map
        (fun s => (s.transferStatus, s.armAttempts)) = some (.completed, 2) := by
  decide

/-- With `kChainArmAttempts = 2`, the second busy refusal is final:
`finishWindowArm` aborts the chain (`.failed`) and the claim is kept
(`index == 1`). -/
theorem trace_busy_refusals_exhaust_arm_attempts :
    let tr : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend,
        .startWindowPlan true,
        .armDstResult 10 .refusedBusy, .retryBusyArm,
        .armDstResult 10 .refusedBusy ]
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run (tr ++ [.retryBusyArm])) = none ∧
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run (tr ++ [.finishWindowArm])).map
        (fun s => (s.transferStatus, s.reg.status)) = some (.failed, .claimed) := by
  decide

/-- Replays `ReshardStackTest.ChainedRequestStaysInProgressUntilTheChainEnds`
(`reshard_service_test.cc`): after window `0` dispatches, `transferStatus`
stays `.inProgress` while window `1` arms on `StartChain`'s thread, and turns
`.completed` only when window `1` finishes. -/
theorem trace_chained_request_stays_in_progress_until_chain_ends :
    let trMid : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend, .settleDispatchedRecv 10,
        .startWindowPlan true ]
    let trEnd : List ChainEv :=
      trMid ++ [ .armDstResult 10 .ok, .finishWindowArm, .sendSrcResult 1 true, .finishWindowSend ]
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run trMid).map
        (fun s => (s.transferStatus, s.winIdx)) = some (.inProgress, 1) ∧
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run trEnd).map (·.transferStatus) =
      some .completed := by
  decide

/-- Replays `ReshardStackTest.ChainedArmFailureMarksTheTransferFailed`
(`reshard_service_test.cc`) and continues it: window `1` (`index = 1`) fails
arming after window `0` dispatched only rank `0`; `transferStatus` becomes
`.failed` and the claim is kept (`if (index == 0)` does not fire). Rank `1` was
never dispatched and never votes, so after rank `0` votes `Complete` the
generation still reads `.claimed` and `CancelIfUnclaimed` returns `false`; the
claim lasts until `ttl_s_`. -/
theorem trace_chained_arm_failure_keeps_claim_past_last_vote :
    let trFail : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend, .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .failed, .finishWindowArm ]
    let trCancel : List ChainEv := trFail ++ [ .senderVoteComplete 0, .cancelIfUnclaimed ]
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run trFail).map
        (fun s => (s.transferStatus, s.reg.status)) = some (.failed, .claimed) ∧
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run trCancel).map
        (fun s => (s.activeSenders, s.reg.status, s.reg.cancelled)) =
      some ([], .claimed, false) := by
  decide

/-! ### Search -/

instance (s : CoordChainState) : Decidable (NoSenderExposed s) := by
  unfold NoSenderExposed; infer_instance
instance (s : CoordChainState) : Decidable (NoPushBeforeAllArmed s) := by
  unfold NoPushBeforeAllArmed; infer_instance
instance (s : CoordChainState) : Decidable (AllArmedWhileDispatching s) := by
  unfold AllArmedWhileDispatching; infer_instance
instance (s : CoordChainState) : Decidable (OneWindowStagedPerDestination s) := by
  unfold OneWindowStagedPerDestination; infer_instance
instance (s : CoordChainState) : Decidable (CompletedOnlyAtChainEnd s) := by
  unfold CompletedOnlyAtChainEnd; infer_instance
instance (s : CoordChainState) : Decidable (ClaimHeldAcrossWindows s) := by
  unfold ClaimHeldAcrossWindows; infer_instance

instance (s : CoordChainState) : Decidable (CoordChainSafe s) :=
  decidable_of_iff
    (Registry.Inv s.reg ∧ NoSenderExposed s ∧ NoPushBeforeAllArmed s ∧
      AllArmedWhileDispatching s ∧ OneWindowStagedPerDestination s ∧
      CompletedOnlyAtChainEnd s ∧ ClaimHeldAcrossWindows s)
    ⟨fun ⟨a, b, c, d, e, f, g⟩ => ⟨a, b, c, d, e, f, g⟩,
     fun h => ⟨h.1, h.2, h.3, h.4, h.5, h.6, h.7⟩⟩

def coordViolates (s : CoordChainState) : Bool := !decide (CoordChainSafe s)

theorem coordViolates_iff (s : CoordChainState) : coordViolates s = false ↔ CoordChainSafe s := by
  simp [coordViolates]

/-- Two ranks over two windows `[[0], [1]]` and one destination, with every
outcome of every RPC, busy refusals, votes, and cancellation attempts. -/
def coordCheckEvents : List ChainEv :=
  [ .registerProducer 0, .registerProducer 1,
    .startChainCoordination,
    .startWindowPlan true, .startWindowPlan false,
    .armDstResult 10 .ok, .armDstResult 10 .refusedBusy, .armDstResult 10 .failed,
    .retryBusyArm, .finishWindowArm,
    .sendSrcResult 0 true, .sendSrcResult 0 false,
    .sendSrcResult 1 true, .sendSrcResult 1 false,
    .finishWindowSend,
    .settleDispatchedRecv 10,
    .senderVoteComplete 0, .senderVoteComplete 1,
    .cancelIfUnclaimed ]

#guard ModelCheck.check (CoordChainState.sys [0, 1] [[0], [1]] [10] 1) coordCheckEvents
  coordViolates 32 = .safe
#guard ModelCheck.check (CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1) coordCheckEvents
  coordViolates 32 = .safe

/-! ### Mutant -/

/-- Mutant: `RunSubPlan` without `if (index == 0)` — `AbandonClaim` on every
window's arm failure. -/
def stepAbandonAtEveryIndex (s : CoordChainState) (e : ChainEv) : Option CoordChainState :=
  match e with
  | .finishWindowArm =>
    (CoordChainState.step s .finishWindowArm).map fun s' =>
      if s'.winPhase == .aborted then
        CoordChainState.syncSenderExposure
          { s' with reg := abandonIfClaimed s'.reg s'.claimOwner }
      else s'
  | e => CoordChainState.step s e

/-- Abandoning the claim when window `1` fails to arm releases it while window
`0`'s dispatched sender `0` is still reading its source HBM
(`activeSenderExposed = true`), and `CancelIfUnclaimed` then tombstones the
generation mid-transfer; the shipping step keeps the claim. -/
theorem mutant_abandon_at_every_index_exposes_active_sender :
    let tr : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .finishWindowSend, .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .failed, .finishWindowArm,
        .cancelIfUnclaimed ]
    ((⟨CoordChainState.init [0, 1] [[0], [1]] [10] 1, stepAbandonAtEveryIndex⟩ :
        System CoordChainState ChainEv).run tr).map
        (fun s => (s.activeSenderExposed, s.reg.cancelled)) = some (true, true) ∧
    ((CoordChainState.sys [0, 1] [[0], [1]] [10] 1).run tr).map
        (fun s => (s.activeSenderExposed, s.reg.cancelled)) = some (false, false) := by
  decide

#guard (ModelCheck.check (⟨CoordChainState.init [0, 1] [[0], [1]] [10] 1, stepAbandonAtEveryIndex⟩ :
    System CoordChainState ChainEv) coordCheckEvents coordViolates 32 matches .counterexample _)

end TpuSyncVerify.Transfer.Reshard.SubPlanChain

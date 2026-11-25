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

/-- Replays `PlanSubTransfersTest.NoWindowsPlanOneTransfer`: empty `dst_window_blocks`, `{0}`, and `{4}` on 4 blocks return `{ count := 1, subPlans := [] }`. -/
theorem trace_plan_no_windows_plan_one_transfer :
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8, 9, 10] } =
      some { count := 1, subPlans := [] } ∧
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8, 9, 10], dstWindowBlocks := [0] } =
      some { count := 1, subPlans := [] } ∧
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8, 9, 10], dstWindowBlocks := [4] } =
      some { count := 1, subPlans := [] } := by
  decide

/-- Replays `PlanSubTransfersTest.WindowSplitsPagedTagInDestinationOrder`: 5 blocks with window `2` split into offsets `0, 2, 4` of counts `2, 2, 1`. -/
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

/-- Replays `PlanSubTransfersTest.UnwindowedTagsRideInTheLastSubTransfer`: unwindowed state tags ride whole (`total = 0`) in the final sub-transfer. -/
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

/-- Replays `PlanSubTransfersTest.WindowedTagSharingIdsDisablesWindows`: a windowed tag sharing destination IDs collapses to `count = 1`; sharing only between unwindowed tags keeps `count = 2`. -/
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

/-- Replays `PlanSubTransfersTest.MalformedWindowsFailClosed`: negative window or arity mismatch returns `none`. -/
theorem trace_plan_malformed_windows_fail_closed :
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8], dstWindowBlocks := [-1] } = none ∧
    planSubTransfers { numTags := 1, dstDeviceBlockIds := [7, 8], dstWindowBlocks := [1, 1] } = none := by
  decide

/-- Replays `SubPlanUuidTest.DerivedUuidsArePositiveAndDistinct`: `subPlanUuid u 0 = u`, and `0 .. 63` are positive and pairwise distinct. -/
theorem trace_sub_plan_uuid_positive_and_distinct :
    let u : UInt64 := 0x7000123456789abc
    let derived := (List.range 64).map (subPlanUuid u)
    subPlanUuid u 0 = u ∧
    derived.all (fun x => x > 0 && x <= 0x7FFFFFFFFFFFFFFF) = true ∧
    derived.Nodup ∧
    subPlanUuid u 1 ≠ subPlanUuid (u + 1) 1 := by
  decide

/-! ## Part II — Worker chain folding -/

/-- `TestManager::FinishPoolReshardRecvPool`: starts upload (`beginOp`), latches drain (`FinishPoolH2d`), and settles (`EndOp`). -/
def finishRecvPool (uuid : Nat) (ok : Bool) : List WorkerEv :=
  [.beginOp uuid, .latchDrain uuid ok, .settleSession uuid true]

/-- Replays `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan`: out-of-order sub-plans `0, 2, 1` of `3` emit `done` once on the third settlement. -/
theorem trace_recv_terminal_waits_for_last_sub_plan :
    let tr1 : List WorkerEv :=
      [.registerSubPlan 5001 0 3] ++ finishRecvPool 5001 true ++ [.poll false,
       .registerSubPlan 5003 2 3] ++ finishRecvPool 5003 true ++ [.poll false]
    let tr2 : List WorkerEv :=
      tr1 ++ [.registerSubPlan 5002 1 3] ++ finishRecvPool 5002 true ++ [.poll false, .poll false]
    (WorkerState.sys.run tr1).map (fun w => (w.doneReports, w.failReports)) = some (0, 0) ∧
    (WorkerState.sys.run tr2).map
        (fun w => (w.doneReports, w.failReports, w.chain.present, w.chain.completed)) =
      some (1, 0, true, true) := by
  decide

/-- Replays `SubPlanChainTest.SendTerminalWaitsForTheLastSubPlan`: `done_sending` waits until both send sub-plans settle. -/
theorem trace_send_terminal_waits_for_last_sub_plan :
    let tr1 : List WorkerEv :=
      [ .registerSubPlan 5041 0 2, .settleSession 5041 true, .poll false ]
    let tr2 : List WorkerEv :=
      tr1 ++ [ .registerSubPlan 5042 1 2, .settleSession 5042 true, .poll false ]
    (WorkerState.sys.run tr1).map (fun w => (w.doneReports, w.failReports)) = some (0, 0) ∧
    (WorkerState.sys.run tr2).map (fun w => (w.doneReports, w.failReports)) = some (1, 0) := by
  decide

/-- Replays `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce`: sub-plan `0` failing reports `failed_recving` once and swallows sub-plan `1`'s clean settlement. -/
theorem trace_first_sub_plan_failure_fails_once :
    let tr : List WorkerEv :=
      [.registerSubPlan 5011 0 2] ++ finishRecvPool 5011 false ++ [.poll false,
       .registerSubPlan 5012 1 2] ++ finishRecvPool 5012 true ++ [.poll false]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.failed)) =
      some (0, 1, true) := by
  decide

/-- Replays `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain`: retry sub-plan `0` re-anchors a fresh entry, ignores stale `6002`, and reports `done` once `7001` and `7002` settle. -/
theorem trace_retry_same_req_id_tracked_as_new_chain :
    let tr : List WorkerEv :=
      [.registerSubPlan 6001 0 2] ++ finishRecvPool 6001 false ++ [.poll false,
       .registerSubPlan 6002 1 2,
       .registerSubPlan 7001 0 2] ++
      finishRecvPool 6002 true ++ [.poll false] ++
      finishRecvPool 7001 true ++ [.poll false,
       .registerSubPlan 7002 1 2] ++ finishRecvPool 7002 true ++ [.poll false]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (1, 1) := by
  decide

/-- Replays `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout`: when sub-plan `1` never arrives, `poll true` reports `failed_recving`. -/
theorem trace_stalled_chain_fails_after_timeout :
    let tr : List WorkerEv :=
      [.registerSubPlan 5021 0 2] ++ finishRecvPool 5021 true ++ [.poll false, .poll true]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (0, 1) := by
  decide

/-- Replays `SubPlanChainTest.SinglePlanRequestsReportAtOnce`: `sub_plan_count <= 1` reports directly without a `sub_plan_chains_` entry. -/
theorem trace_single_plan_reports_at_once :
    let tr : List WorkerEv :=
      [.registerSubPlan 5031 0 1] ++ finishRecvPool 5031 true ++ [.poll false]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.present)) =
      some (1, 0, false) := by
  decide

/-- Sparse sender in window `0` only (`ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries` rank `0`): fills `1` of `2` slots, so the stall sweep reports `failed_recving` rather than `done_sending`. -/
theorem trace_sender_in_first_window_only :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9000 0 2, .settleSession 9000 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.subSessions)) =
      some (0, 0, [some 9000, none]) ∧
    (WorkerState.sys.run (tr ++ [.poll true])).map (fun w => (w.doneReports, w.failReports)) =
      some (0, 1) := by
  decide

/-- Sparse sender in window `1` only: seats an unanchored entry (`[none, some 9001]`) that stall-sweeps to `failed_recving`. -/
theorem trace_sender_in_last_window_only :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9001 1 2, .settleSession 9001 true, .poll false ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.chain.anchored, w.chain.subSessions)) =
      some (0, false, [none, some 9001]) ∧
    (WorkerState.sys.run (tr ++ [.poll true])).map (fun w => (w.doneReports, w.failReports)) =
      some (0, 1) := by
  decide

/-- Sub-plan `0` failing publishes `failed_recving` while sibling sub-plan `1` is still active in `sessions_`. -/
theorem trace_first_failure_reports_while_sibling_active :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7100 0 2, .registerSubPlan 7101 1 2,
        .settleSession 7100 false, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.failReports, w.active.map (·.uuid))) =
      some (1, [7101]) := by
  decide

/-- A late sub-plan `1` arriving after the failed tombstone was swept seats a fresh unanchored entry and sweeps to a second `failed_recving`. -/
theorem trace_late_sub_plan_after_sweep_reports_failure_twice :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7300 0 2, .settleSession 7300 false, .poll false,
        .poll true,
        .registerSubPlan 7301 1 2, .settleSession 7301 true, .poll false, .poll true ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (0, 2) := by
  decide

/-- Under A5a, an in-flight `!IsDraining()` sub-plan open across tombstone erasure is latched with deadline error and later reports a second failure, never `done`. -/
theorem trace_sub_plan_seated_at_sweep_is_failed_not_done :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7500 0 2, .settleSession 7500 false, .poll false,
        .registerSubPlan 7501 1 2, .beginOp 7501, .poll true,
        .settleSession 7501 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.present)) =
      some (0, 2, false) := by
  decide

/-- An open (`in_flight_ > 0`) session keeps a live stalled entry (`activeKeeps`) while A5a latches error on it, failing on the next poll after `EndOp`. -/
theorem trace_stall_sweep_keeps_entry_with_open_session :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9100 0 2, .registerSubPlan 9101 1 2,
        .settleSession 9100 true, .poll false, .beginOp 9101, .poll true ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.failReports, w.chain.failed, w.active.map (·.ok))) =
      some (0, 0, false, [false]) ∧
    (WorkerState.sys.run (tr ++ [.settleSession 9101 true, .poll false])).map
        (fun w => (w.doneReports, w.failReports, w.chain.failed)) = some (0, 1, true) := by
  decide

/-- An idle (`in_flight_ == 0`) session past its deadline settles synchronously in `Finish(DeadlineExceeded)` and retires in the same poll before the sweep. -/
theorem trace_idle_seated_session_settles_at_deadline_poll :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9200 0 2, .registerSubPlan 9201 1 2,
        .settleSession 9200 true, .poll false, .poll true ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.failReports, w.chain.present, w.chain.failed, w.active)) =
      some (0, 1, true, true, []) := by
  decide

/-- On a live chain, a draining-OK session (`IsDraining() && !Done()`) is skipped by `!IsDraining()` and kept by `!sub_session->Done()`, completing cleanly once `EndOp` and sub-plan `1` finish. -/
theorem trace_draining_session_on_live_chain_survives_stalled_poll :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9300 0 2, .beginOp 9300, .latchDrain 9300 true, .poll true,
        .settleSession 9300 true, .poll false,
        .registerSubPlan 9301 1 2, .settleSession 9301 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (1, 0) := by
  decide

/-- A single-plan session seated before an unanchored sub-plan `1` of the same `req_id` hits the `!anchored` stray arm and still reports `done`. -/
theorem trace_single_plan_beside_unanchored_entry_reports :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7600 0 1, .registerSubPlan 7601 1 2,
        .settleSession 7600 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.anchored)) =
      some (1, 0, false) := by
  decide

/-! ### Search -/

instance (w : WorkerState) : Decidable (DoneAfterAllSubPlans w) := by
  unfold DoneAfterAllSubPlans; infer_instance

instance (w : WorkerState) : Decidable (WorkerSafe w) :=
  decidable_of_iff (DoneAfterAllSubPlans w) ⟨fun h => ⟨h⟩, fun h => h.1⟩

def workerViolates (w : WorkerState) : Bool := !decide (WorkerSafe w)

theorem workerViolates_iff (w : WorkerState) : workerViolates w = false ↔ WorkerSafe w := by
  simp [workerViolates]

/-- Candidate events for a `K = 2` chain with all session lifecycle transitions and both poll modes. -/
def chain2CheckEvents : List WorkerEv :=
  [ .registerSubPlan 10 0 2, .registerSubPlan 11 1 2,
    .beginOp 10, .beginOp 11,
    .latchDrain 10 true, .latchDrain 10 false,
    .latchDrain 11 true, .latchDrain 11 false,
    .settleSession 10 true, .settleSession 10 false,
    .settleSession 11 true, .settleSession 11 false,
    .poll false, .poll true ]

/-- Candidate events for a `K = 2` chain plus a same-`req_id` retry re-anchoring at index `0`. -/
def workerCheckEvents : List WorkerEv :=
  [ .registerSubPlan 10 0 2, .registerSubPlan 11 1 2, .registerSubPlan 13 0 2,
    .beginOp 11, .latchDrain 11 true, .latchDrain 11 false,
    .settleSession 10 true, .settleSession 10 false,
    .settleSession 11 true, .settleSession 11 false,
    .settleSession 13 true, .settleSession 13 false,
    .poll false, .poll true ]

/-- A single-plan request beside a `K = 2` chain under the same `req_id`. -/
def mixedCheckEvents : List WorkerEv :=
  [ .registerSubPlan 20 0 1, .registerSubPlan 10 0 2, .registerSubPlan 11 1 2,
    .beginOp 11, .latchDrain 11 true, .latchDrain 11 false,
    .settleSession 20 true, .settleSession 20 false,
    .settleSession 10 true, .settleSession 10 false,
    .settleSession 11 true, .settleSession 11 false,
    .poll false, .poll true ]

#guard ModelCheck.check WorkerState.sysRestricted chain2CheckEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysRestricted workerCheckEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysRestricted mixedCheckEvents workerViolates 32 = .safe

/-! ### Mutants -/

/-- Mutant: omitting `std::find` on `sub_sessions` counts a superseded chain's settled session into its index slot. -/
def stepCountStale (w : WorkerState) (e : WorkerEv) : Option WorkerState :=
  match e with
  | .poll stalled =>
    let c := w.sessions.foldl (fun (c : ChainProgress) s =>
      if s.done && !s.retired && c.present && !c.subSessions.contains (some s.uuid) then
        { c with subSessions := (growTo c.subSessions (s.index + 1)).set s.index (some s.uuid) }
      else c) w.chain
    WorkerState.step { w with chain := c } (.poll stalled)
  | e => WorkerState.step w e

/-- `stepCountStale` emits `done` for a retry while the retry's sub-plan `1` (`7002`) is still active. -/
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

/-- Mutant: `NoteSubPlanRegisteredLocked` without re-anchoring on `sub_plan_index == 0` traps same-`req_id` retries in the failed entry. -/
def noteSubPlanRegisteredNoReanchor (c : ChainProgress) (uuid index count : Nat) : ChainProgress :=
  if count <= 1 then {}
  else
    let base : ChainProgress := if c.present then c else { present := true }
    let needed := max base.subSessions.length (max count (index + 1))
    { base with subSessions := (growTo base.subSessions needed).set index (some uuid) }

def stepNoReanchor (w : WorkerState) (e : WorkerEv) : Option WorkerState :=
  match e with
  | .registerSubPlan uuid index count =>
    if w.seenUuids.contains uuid || index >= max 1 count then none
    else
      some
        { w with
          seenUuids := uuid :: w.seenUuids
          sessions := w.sessions ++ [{ uuid := uuid, index := index, count := count }]
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

/-- Mutant: omitting the A5a deadline arm lets an in-flight `!IsDraining()` session survive tombstone sweep and emit `done` (`doneLog = [(2, [1])]`). -/
theorem mutant_no_deadline_arm_reports_done_after_sweep :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7500 0 2, .settleSession 7500 false, .poll false,
        .registerSubPlan 7501 1 2, .beginOp 7501, .poll true,
        .settleSession 7501 true, .poll false ]
    ((⟨WorkerState.init, WorkerState.stepG
        { deadlineArm := false, noDrainingOkAtSweep := false,
          noSettleBetweenRetireAndSweep := false }⟩ :
        System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.failReports, w.doneLog)) = some (1, 1, [(2, [1])]) ∧
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.doneLog)) =
      some (0, 2, []) := by
  decide

#guard (ModelCheck.check (⟨WorkerState.init, WorkerState.stepG { deadlineArm := false }⟩ :
    System WorkerState WorkerEv) workerCheckEvents workerViolates 32 matches .counterexample _)

/-! ### Executions of `WorkerState.step` that violate `DoneAfterAllSubPlans` (A5b, A5c) -/

/-- A5b window (`WorkerState.sys`): a draining-OK sibling skipped by `!IsDraining()` outlives tombstone erasure and emits `done` after failure (`sysRestricted` refuses; `sysSweepWaitsForOpen` swallows it). -/
theorem trace_draining_ok_at_sweep_reports_done_after_failure :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7510 0 2, .settleSession 7510 false, .poll false,
        .registerSubPlan 7511 1 2, .beginOp 7511, .latchDrain 7511 true,
        .poll true,
        .settleSession 7511 true, .poll false ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.failReports, w.doneLog)) = some (1, 1, [(2, [1])]) ∧
    WorkerState.sysRestricted.run tr = none ∧
    (WorkerState.sysSweepWaitsForOpen.run tr).map
        (fun w => (w.doneReports, w.failReports, w.doneLog)) = some (0, 1, []) := by
  decide

/-- `chain2CheckEvents` plus intra-poll `EndOp` settlements between retire and sweep. -/
def chain2MidSettleEvents : List WorkerEv :=
  chain2CheckEvents ++
    [ .pollMidSettle true [10], .pollMidSettle true [11], .pollMidSettle true [10, 11],
      .pollMidSettle false [10], .pollMidSettle false [11], .pollMidSettle false [10, 11] ]

/-- `workerCheckEvents` plus intra-poll `EndOp` settlements on `11`. -/
def workerMidSettleEvents : List WorkerEv :=
  workerCheckEvents ++ [ .pollMidSettle true [11], .pollMidSettle false [11] ]

#guard (ModelCheck.check WorkerState.sys chain2CheckEvents workerViolates 32
  matches .counterexample _)
#guard (ModelCheck.check WorkerState.sys workerCheckEvents workerViolates 32
  matches .counterexample _)
#guard (ModelCheck.check WorkerState.sys mixedCheckEvents workerViolates 32
  matches .counterexample _)

/-- A5c window (`WorkerState.sys`): mid-poll `EndOp` stalls an unanchored chain and clears `sub_sessions`, so the next poll hits `!anchored` before `failed` and emits `done` (`sysRestricted` refuses; `sysSweepWaitsForOpen` suppresses it). -/
theorem trace_mid_poll_endop_unanchored_reports_done_after_stall :
    let tr : List WorkerEv :=
      [ .registerSubPlan 11 1 2, .beginOp 11, .latchDrain 11 true,
        .pollMidSettle true [11], .poll false ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.failReports, w.doneLog)) = some (1, 1, [(2, [1])]) ∧
    ((⟨WorkerState.init, WorkerState.stepG
        { noSettleBetweenRetireAndSweep := false }⟩ :
        System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.failReports, w.doneLog)) = some (1, 1, [(2, [1])]) ∧
    WorkerState.sysRestricted.run tr = none ∧
    (WorkerState.sysSweepWaitsForOpen.run tr).map
        (fun w => (w.doneReports, w.failReports, w.doneLog)) = some (0, 1, []) := by
  dsimp; refine ⟨by decide, by decide, by decide, by decide⟩

#guard (ModelCheck.check WorkerState.sys chain2MidSettleEvents workerViolates 32
  matches .counterexample _)
#guard (ModelCheck.check (⟨WorkerState.init,
    WorkerState.stepG { noSettleBetweenRetireAndSweep := false }⟩ :
    System WorkerState WorkerEv) chain2MidSettleEvents workerViolates 32 matches .counterexample _)
#guard ModelCheck.check WorkerState.sysSweepWaitsForOpen chain2MidSettleEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysSweepWaitsForOpen workerMidSettleEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysSweepWaitsForOpen mixedCheckEvents workerViolates 32 = .safe

/-! ### Registrations the C++ accepts outside A6 and A7 -/

/-- Outside A6 (`noSingleWhileSeated := false`): a single-plan registration erases `sub_plan_chains_[req_id]`, so the seated sub-plan's settlement emits premature `done`. -/
theorem trace_single_plan_registration_erases_live_chain :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7700 0 2, .registerSubPlan 7701 0 1,
        .settleSession 7700 true, .poll false ]
    ((⟨WorkerState.init, WorkerState.stepG
        { noDrainingOkAtSweep := false, noSettleBetweenRetireAndSweep := false,
          noSingleWhileSeated := false }⟩ :
        System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.doneLog, w.active.map (·.uuid))) =
      some (1, [(2, [0])], [7701]) ∧
    WorkerState.sys.run tr = none := by
  decide

#guard (ModelCheck.check (⟨WorkerState.init, WorkerState.stepG { noSingleWhileSeated := false }⟩ :
    System WorkerState WorkerEv) mixedCheckEvents workerViolates 32 matches .counterexample _)

/-- Outside A7 (`oneSessionPerSlot := false`): overwriting `sub_sessions[1]` on an unanchored chain sends the displaced session to the `!anchored` stray arm, emitting premature `done`. -/
theorem trace_slot_overwrite_reports_stray_done :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7800 1 2, .registerSubPlan 7801 1 2,
        .settleSession 7800 true, .poll false ]
    ((⟨WorkerState.init, WorkerState.stepG
        { noDrainingOkAtSweep := false, noSettleBetweenRetireAndSweep := false,
          oneSessionPerSlot := false }⟩ :
        System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.doneLog, w.chain.subSessions)) =
      some (1, [(2, [1])], [none, some 7801]) ∧
    ((⟨WorkerState.init, WorkerState.stepG { oneSessionPerSlot := false }⟩ :
        System WorkerState WorkerEv).run tr).map
        (fun w => (w.doneReports, w.doneLog, w.chain.subSessions)) =
      some (1, [(2, [1])], [none, some 7801]) ∧
    WorkerState.sys.run tr = none ∧
    WorkerState.sysRestricted.run tr = none := by
  dsimp; refine ⟨by decide, by decide, by decide, by decide⟩

/-! ## Part III — Coordinator multi-window chain -/

/-- Replays `ReshardStackTest.WindowedRequestChainsOneSubTransferPerWindow`: 4 single-page windows chain under one registry claim, ending `.completed` while refusing `CancelIfUnclaimed`. -/
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
        .sendSrcResult 1 true, .finishWindowSend,
        .cancelIfUnclaimed ]
    ((CoordChainState.sys [0, 1] [[0], [0], [1], [1]] [10] 1).run tr).map
        (fun s => (s.transferStatus, s.reg.status, s.reg.cancelled)) =
      some (.completed, .claimed, false) := by
  decide

/-- Replays `ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries`: window `0` dispatches `[0, 1]` and window `1` dispatches `[1]`. -/
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

/-- Replays `ReshardStackTest.WindowComposesWithPrefixClip` and extends it with A3 `Complete` votes: rank `0` (absent from window `1`) can vote after window `0` while keeping the claim until rank `1` votes. -/
theorem trace_window_composes_with_prefix_clip :
    let trReplay : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .sendSrcResult 1 true, .finishWindowSend,
        .settleDispatchedRecv 10,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend ]
    let trVote0 : List ChainEv :=
      [ .registerProducer 0, .registerProducer 1, .startChainCoordination,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 0 true, .sendSrcResult 1 true, .finishWindowSend,
        .settleDispatchedRecv 10, .senderVoteComplete 0,
        .startWindowPlan true, .armDstResult 10 .ok, .finishWindowArm,
        .sendSrcResult 1 true, .finishWindowSend ]
    ((CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1).run trReplay).map
        (fun s => (s.transferStatus, s.activeSenders)) = some (.completed, [0, 1]) ∧
    ((CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1).run trVote0).map
        (fun s => (s.transferStatus, s.reg.status, s.activeSenders)) =
      some (.completed, .claimed, [1]) ∧
    ((CoordChainState.sys [0, 1] [[0, 1], [1]] [10] 1).run (trVote0 ++ [.senderVoteComplete 1])).map
        (fun s => (s.reg.status, s.reg.retired)) = some (.unknown, true) := by
  decide

/-- Replays `ReshardStackTest.ChainedArmRetriesWhileReceiverStagingIsBusy`: window `1` retries two `ArmRefusedForBusyStaging` refusals and arms once window `0` settles. -/
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

/-- Exhausting `maxArmAttempts` (`2` here) on busy refusals aborts the chain (`.failed`) while keeping the claim (`index == 1`). -/
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

/-- Replays `ReshardStackTest.ChainedRequestStaysInProgressUntilTheChainEnds`: `transferStatus` stays `.inProgress` across window `0` until window `1` finishes. -/
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

/-- Replays `ReshardStackTest.ChainedArmFailureMarksTheTransferFailed` and continues it: window `1` arm failure marks `.failed` and keeps the claim (`index == 1`) past rank `0`'s vote until `ttl_s_`. -/
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

/-- Candidate chain events over two source ranks, two windows, and one destination. -/
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

/-- Mutant: `RunSubPlan` calling `AbandonClaim` without `if (index == 0)`. -/
def stepAbandonAtEveryIndex (s : CoordChainState) (e : ChainEv) : Option CoordChainState :=
  match e with
  | .finishWindowArm =>
    (CoordChainState.step s .finishWindowArm).map fun s' =>
      if s'.winPhase == .aborted then
        CoordChainState.syncSenderExposure
          { s' with reg := abandonIfClaimed s'.reg s'.claimOwner }
      else s'
  | e => CoordChainState.step s e

/-- `stepAbandonAtEveryIndex` drops the claim on window `1` arm failure while window `0`'s sender `0` is still active, letting `CancelIfUnclaimed` tombstone the generation. -/
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

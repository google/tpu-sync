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

/-- `TestManager::FinishPoolReshardRecvPool` (`pool_reshard_test.cc`):
`OnDataReceived` starts the pool upload (`++in_flight_`), `FinishPoolH2d` calls
`Finish(terminal_status)` latching `draining_` while the upload op is counted,
and `self->EndOp()` settles `done_ = true` preserving the latched status. -/
def finishRecvPool (uuid : Nat) (ok : Bool) : List WorkerEv :=
  [.beginOp uuid, .latchDrain uuid ok, .settleSession uuid true]

/-- Replays `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan`
(`pool_reshard_test.cc`): sub-plans `0`, `2`, `1` of `count = 3` settle out of
order via `FinishPoolReshardRecvPool`; `done_recving` is inserted only once the
third settles (`settled_count == progress.sub_sessions.size()`), polling again
does not re-insert it (`progress.completed`), and the entry stays, completed,
until the sweep expires it. -/
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

/-- Replays `SubPlanChainTest.SendTerminalWaitsForTheLastSubPlan`
(`pool_reshard_test.cc`): sub-plan `0` of `2` (`5041`, a send session seated by
`AddPoolReshardSend` with `SessionRole::kPoolReshardSend` and `in_flight_ == 0`)
settles OK on `Finish()` and `done_sending` stays empty; the request is inserted
once sub-plan `1` (`5042`) settles. -/
theorem trace_send_terminal_waits_for_last_sub_plan :
    let tr1 : List WorkerEv :=
      [ .registerSubPlan 5041 0 2, .settleSession 5041 true, .poll false ]
    let tr2 : List WorkerEv :=
      tr1 ++ [ .registerSubPlan 5042 1 2, .settleSession 5042 true, .poll false ]
    (WorkerState.sys.run tr1).map (fun w => (w.doneReports, w.failReports)) = some (0, 0) ∧
    (WorkerState.sys.run tr2).map (fun w => (w.doneReports, w.failReports)) = some (1, 0) := by
  decide

/-- Replays `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce`
(`pool_reshard_test.cc`): sub-plan `0` of `2` fails and `failed_recving` is
inserted once; sub-plan `1` is still seated afterwards and its clean settlement
is swallowed (`if (progress.failed || progress.completed) return;`). -/
theorem trace_first_sub_plan_failure_fails_once :
    let tr : List WorkerEv :=
      [.registerSubPlan 5011 0 2] ++ finishRecvPool 5011 false ++ [.poll false,
       .registerSubPlan 5012 1 2] ++ finishRecvPool 5012 true ++ [.poll false]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.failed)) =
      some (0, 1, true) := by
  decide

/-- Replays `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain`
(`pool_reshard_test.cc`): the first chain fails at sub-plan `0` (`6001`) while
sub-plan `1` (`6002`) is still active; the consumer retries under the same
`req_id` (`7001`, `7002`); sub-plan 0 of the retry re-anchors a fresh entry, the
stale `6002` settling is ignored (`std::find` on `sub_sessions` fails and the
entry is anchored), and the retry inserts `done` once `7001` and `7002`
settle. -/
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

/-- Replays `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout`
(`pool_reshard_test.cc`): sub-plan `0` of `2` settles OK (`5021`); when sub-plan
`1` never arrives and the entry is quiet for `timeout_s_` (`poll true`),
`SweepSubPlanChainsLocked` inserts `failed_recving`. -/
theorem trace_stalled_chain_fails_after_timeout :
    let tr : List WorkerEv :=
      [.registerSubPlan 5021 0 2] ++ finishRecvPool 5021 true ++ [.poll false, .poll true]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (0, 1) := by
  decide

/-- Replays `SubPlanChainTest.SinglePlanRequestsReportAtOnce`
(`pool_reshard_test.cc`): `sub_plan_count <= 1` (`5031`) reports on settlement
without a `sub_plan_chains_` entry. -/
theorem trace_single_plan_reports_at_once :
    let tr : List WorkerEv :=
      [.registerSubPlan 5031 0 1] ++ finishRecvPool 5031 true ++ [.poll false]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.present)) =
      some (1, 0, false) := by
  decide

/-- A sender with spans in window `0` only of a `K = 2` request
(`ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries` dispatches rank
`0` once, stamped `sub_plan_index = 0, sub_plan_count = 2`): its single push
settles OK, `sub_sessions = [session, nullptr]` has one settled slot of two, so
no `done_sending` is inserted; once the entry is quiet for `timeout_s_` the
sweep inserts `failed_recving`. The coordinator model's A3 guard still admits
this rank's `Complete` vote once window `0` is dispatched
(`trace_window_composes_with_prefix_clip`); whether the producer votes after a
`failed_recving` report is not determined by the cited C++. -/
theorem trace_sender_in_first_window_only :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9000 0 2, .settleSession 9000 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.subSessions)) =
      some (0, 0, [some 9000, none]) ∧
    (WorkerState.sys.run (tr ++ [.poll true])).map (fun w => (w.doneReports, w.failReports)) =
      some (0, 1) := by
  decide

/-- A sender with spans in window `1` only (`sub_plan_index = 1`): its
registration inserts an unanchored entry (`anchored = false`) whose slot 0 is
`nullptr`, its clean settlement counts `1 of 2`, and the sweep inserts
`failed_recving`. -/
theorem trace_sender_in_last_window_only :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9001 1 2, .settleSession 9001 true, .poll false ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.chain.anchored, w.chain.subSessions)) =
      some (0, false, [none, some 9001]) ∧
    (WorkerState.sys.run (tr ++ [.poll true])).map (fun w => (w.doneReports, w.failReports)) =
      some (0, 1) := by
  decide

/-- Sub-plans `0` and `1` are both seated; sub-plan `0` fails and
`RecordPoolReshardTerminalLocked` inserts `failed_recving` while sub-plan `1`'s
session is still in `sessions_` (`active ≠ []`): the request's failure is
published before the chain's last session settles. -/
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

/-- A sub-plan with an op in flight (`beginOp 7501`: `in_flight_ > 0`) when its
failed entry is erased by the sweep has passed its own deadline (A5a):
`CompleteReadWithDetails` latches `draining` with error while leaving `!Done()`
across `RetireSettledSessionLocked`, `SweepSubPlanChainsLocked` erases the
failed tombstone, and once `EndOp` settles `7501` the next poll reports it
through `it == sub_plan_chains_.end()` as a second `failed_recving`, never as
`done`. -/
theorem trace_sub_plan_seated_at_sweep_is_failed_not_done :
    let tr : List WorkerEv :=
      [ .registerSubPlan 7500 0 2, .settleSession 7500 false, .poll false,
        .registerSubPlan 7501 1 2, .beginOp 7501, .poll true,
        .settleSession 7501 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports, w.chain.present)) =
      some (0, 2, false) := by
  decide

/-- The stall sweep keeps a live entry while a slot's session still has an op
in flight (`beginOp 9101`: `in_flight_ > 0` stays `!Done()` across
`RetireSettledSessionLocked` into `SweepSubPlanChainsLocked`, which refreshes
`last_activity = now`); `CompleteReadWithDetails`'s deadline arm already latched
error on `9101` (A5a), so once `EndOp` settles it the next poll reports the
chain's first failure. -/
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

/-- An idle (`inFlight = false`, `in_flight_ == 0`) session past its deadline at
`poll true` settles synchronously inside `Finish(DeadlineExceeded)` (`"One
nobody pulled is reported now"`): `RetireSettledSessionLocked` records the
chain failure (`failReports = 1`), retires `9201` (`active = []`), and refreshes
`last_activity = now` (`touched = true`) before `SweepSubPlanChainsLocked`
runs. -/
theorem trace_idle_seated_session_settles_at_deadline_poll :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9200 0 2, .registerSubPlan 9201 1 2,
        .settleSession 9200 true, .poll false, .poll true ]
    (WorkerState.sys.run tr).map
        (fun w => (w.doneReports, w.failReports, w.chain.present, w.chain.failed, w.active)) =
      some (0, 1, true, true, []) := by
  decide

/-- On a live (`!failed && !completed`) chain, a sub-plan already in
`IsDraining() && !Done()` with `OkStatus()` (`beginOp 9300, latchDrain 9300 true`)
is skipped by `if (!session->IsDraining())` at `poll true` while
`SweepSubPlanChainsLocked` sees `!sub_session->Done()` and keeps the entry
(`activeKeeps`); `EndOp` and sub-plan `1` then complete the chain cleanly. -/
theorem trace_draining_session_on_live_chain_survives_stalled_poll :
    let tr : List WorkerEv :=
      [ .registerSubPlan 9300 0 2, .beginOp 9300, .latchDrain 9300 true, .poll true,
        .settleSession 9300 true, .poll false,
        .registerSubPlan 9301 1 2, .settleSession 9301 true, .poll false ]
    (WorkerState.sys.run tr).map (fun w => (w.doneReports, w.failReports)) = some (1, 0) := by
  decide

/-- A single-plan session seated before a chained sub-plan `1` of the same
`req_id` is not in the unanchored entry's `sub_sessions`, so its terminal is
the request's (the `stray` arm) — the single-plan request still reports. -/
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

/-- Two sub-plans of a `K = 2` request with every session lifecycle transition
(`beginOp`, `latchDrain`, `settleSession`) and both poll modes. -/
def chain2CheckEvents : List WorkerEv :=
  [ .registerSubPlan 10 0 2, .registerSubPlan 11 1 2,
    .beginOp 10, .beginOp 11,
    .latchDrain 10 true, .latchDrain 10 false,
    .latchDrain 11 true, .latchDrain 11 false,
    .settleSession 10 true, .settleSession 10 false,
    .settleSession 11 true, .settleSession 11 false,
    .poll false, .poll true ]

/-- Two sub-plans of a `K = 2` request and a re-anchoring index-`0` sub-plan
(a retry under the same `req_id`), with in-flight / draining transitions on the
sibling sub-plan `11` and polls with and without the stall sweep. -/
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

/-! `WorkerState.sysRestricted` (`WorkerState.step` restricted to executions
satisfying A5b and A5c, the system of `reachable_worker_safe`) is safe on all
three event lists. `WorkerState.sys` is not: see "Executions of
`WorkerState.step` that violate `DoneAfterAllSubPlans`" below. -/

#guard ModelCheck.check WorkerState.sysRestricted chain2CheckEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysRestricted workerCheckEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysRestricted mixedCheckEvents workerViolates 32 = .safe

/-! ### Mutants

`stepCountStale` and `stepNoReanchor` change one function of `WorkerState.step`
and defer to it for everything else. The deadline-arm mutant is
`WorkerState.stepG` with `deadlineArm` off. -/

/-- Mutant: `RecordPoolReshardTerminalLocked` without the `std::find` on
`sub_sessions` — a settled session missing from the entry is counted in its
index's slot (the slot a superseded chain's sub-plan was stamped with). -/
def stepCountStale (w : WorkerState) (e : WorkerEv) : Option WorkerState :=
  match e with
  | .poll stalled =>
    let c := w.sessions.foldl (fun (c : ChainProgress) s =>
      if s.done && !s.retired && c.present && !c.subSessions.contains (some s.uuid) then
        { c with subSessions := (growTo c.subSessions (s.index + 1)).set s.index (some s.uuid) }
      else c) w.chain
    WorkerState.step { w with chain := c } (.poll stalled)
  | e => WorkerState.step w e

/-- On the retry trace, the mutant inserts `done` for the retry as soon as the
stale `6002` and the new `7001` settle, while the retry's own sub-plan `1`
(`7002`) is still active; `WorkerState.step` waits for `7002`. -/
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

/-- Mutant: `WorkerState.step` without the A5a deadline arm (`stepG` with
`deadlineArm`, `noDrainingOkAtSweep` and `noSettleBetweenRetireAndSweep` off;
A6 and A7 on). An in-flight `!IsDraining()` sub-plan still seated when the
sweep erases its failed entry later settles OK and, finding no entry, is
reported as the request's `done` (`doneLog = [(2, [1])]`): a `done_recving` for
a two-plan request with one sub-plan settled. In `WorkerState.step` the
deadline arm latches error on that session at the sweep. -/
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

/-! The search below keeps the A5b and A5c guards on (`stepG { deadlineArm :=
false }` is `WorkerState.stepRestricted` without the deadline arm), because
`WorkerState.sys` already has a counterexample on these events through the A5b
window: the counterexample found here is one that A5b and A5c do not exclude,
so the deadline arm is needed for `reachable_worker_safe`. -/

#guard (ModelCheck.check (⟨WorkerState.init, WorkerState.stepG { deadlineArm := false }⟩ :
    System WorkerState WorkerEv) workerCheckEvents workerViolates 32 matches .counterexample _)

/-! ### Executions of `WorkerState.step` that violate `DoneAfterAllSubPlans`

A5b and A5c are not caller assumptions and not guards of `WorkerState.step`:
each names a timing window inside `CompleteReadWithDetails` /
`SweepSubPlanChainsLocked` that the C++ permits with every caller well behaved.
The two traces below are executions of `WorkerState.sys` in those windows that
end with a `done` logged for a two-sub-plan request with one slot settled;
`WorkerState.sysRestricted` refuses each, and the alternative
`WorkerState.sysSweepWaitsForOpen` (not the C++) runs each without the `done`. -/

/-- The A5b window (an execution of `WorkerState.sys`; a C++ interleaving
on a single non-retried 2-sub-plan chain): sub-plan `0` (`7510`)
fails on poll 1 (`failReports = 1`, `chain.failed = true`); sub-plan `1`
(`7511`) is then registered into the failed entry and latches clean drain
(`beginOp 7511, latchDrain 7511 true`,
e.g. `ReshardSendSession::RecordPushCompletion` setting `finalizing_ = true`
before `EndOp` finishes `ReleaseDataConnectionsAfterSend`, or
`ReshardReceiveSession::FinishPoolH2d` setting `draining_ = true` before
`self->EndOp()`). On poll 2 (`poll true`), `if (!session->IsDraining())` skips
`7511` and `SweepSubPlanChainsLocked` unconditionally erases the `failed`
tombstone; once `EndOp` sets `done_ = true`, poll 3 hits
`it == sub_plan_chains_.end()` and emits `done` (`doneLog = [(2, [1])]`) after
failure. `WorkerState.sysRestricted` (A5b) refuses poll 2; the
alternative `WorkerState.sysSweepWaitsForOpen` keeps the tombstone until
`7511` settles and swallows its terminal (`(0, 1, [])`). -/
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

/-- `chain2CheckEvents` plus intra-poll `session->mu_` `EndOp` settlements
between `RetireSettledSessionLocked` and `SweepSubPlanChainsLocked`. -/
def chain2MidSettleEvents : List WorkerEv :=
  chain2CheckEvents ++
    [ .pollMidSettle true [10], .pollMidSettle true [11], .pollMidSettle true [10, 11],
      .pollMidSettle false [10], .pollMidSettle false [11], .pollMidSettle false [10, 11] ]

/-- `workerCheckEvents` plus intra-poll `EndOp` settlements on `11`. -/
def workerMidSettleEvents : List WorkerEv :=
  workerCheckEvents ++ [ .pollMidSettle true [11], .pollMidSettle false [11] ]

/-! `WorkerState.sys` has a counterexample on each of `chain2CheckEvents`,
`workerCheckEvents` and `mixedCheckEvents`, none of which has a `pollMidSettle`
event: the A5b window alone. -/

#guard (ModelCheck.check WorkerState.sys chain2CheckEvents workerViolates 32
  matches .counterexample _)
#guard (ModelCheck.check WorkerState.sys workerCheckEvents workerViolates 32
  matches .counterexample _)
#guard (ModelCheck.check WorkerState.sys mixedCheckEvents workerViolates 32
  matches .counterexample _)

/-- The A5c window (an execution of `WorkerState.sys` in which no poll sweeps a
tombstone, so A5b holds throughout; a C++ interleaving): an unanchored sub-plan `1` (`11`, `index = 1,
count = 2`, `anchored = false`) latches clean drain (`beginOp 11,
latchDrain 11 true`) right as `timeout_s_` elapses. In `CompleteReadWithDetails`
(`pollMidSettle true [11]`), `!session->IsDraining()` skips `11` and
`RetireSettledSessionLocked` sees `Done() == false`; before
`SweepSubPlanChainsLocked` runs, `11->EndOp()` sets `done_ = true` under
`11->mu_`, so `!sub_session->Done()` is `false` and the sweep declares the live
chain stalled (`failed_recving_.insert(req_id); progress.failed = true;
progress.sub_sessions.clear()`) with `anchored` still `false`. On the next poll
(`poll false`), `RecordPoolReshardTerminalLocked` checks
`sub_it == progress.sub_sessions.end()` before
`if (progress.failed || progress.completed) return;`, hits `!progress.anchored`,
and emits `done` (`doneLog = [(2, [1])]`) after failure.
`WorkerState.sysRestricted` (A5c) refuses the mid-poll settle; the
alternative `WorkerState.sysSweepWaitsForOpen` checks `failed || completed`
before `!anchored` and keeps tombstones while `sessions_` holds any unretired
entry (`(0, 1, [])`). -/
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

/-! `WorkerState.sys` has a counterexample on `chain2MidSettleEvents`, and so
does `WorkerState.stepG { noSettleBetweenRetireAndSweep := false }` (A5b on,
A5c off) on the same events: the A5c window alone. The alternative is safe on
every event list. -/

#guard (ModelCheck.check WorkerState.sys chain2MidSettleEvents workerViolates 32
  matches .counterexample _)
#guard (ModelCheck.check (⟨WorkerState.init,
    WorkerState.stepG { noSettleBetweenRetireAndSweep := false }⟩ :
    System WorkerState WorkerEv) chain2MidSettleEvents workerViolates 32 matches .counterexample _)
#guard ModelCheck.check WorkerState.sysSweepWaitsForOpen chain2MidSettleEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysSweepWaitsForOpen workerMidSettleEvents workerViolates 32 = .safe
#guard ModelCheck.check WorkerState.sysSweepWaitsForOpen mixedCheckEvents workerViolates 32 = .safe

/-! ### Registrations the C++ accepts outside A6 and A7

A6 and A7 are assumptions about callers that nothing in the C++ checks.
`WorkerState.step` carries them as guards and refuses the two traces below;
`WorkerState.step` with the one guard off (`stepG` with `noDrainingOkAtSweep`,
`noSettleBetweenRetireAndSweep` and that guard off) runs them, and each is an
execution of the C++ as it is, not a mutant. -/

/-- Outside A6 (`WorkerState.step` without the `noSingleWhileSeated` guard; the
C++ does not refuse the registration): a single-plan registration under the `req_id` of a
chain in flight erases the chain's entry
(`NoteSubPlanRegisteredLocked` `sub_plan_chains_.erase(req_id)`),
and the chained sub-plan's clean settlement is then reported as the request's
`done` while the single-plan session is still open. `WorkerState.step` (A6)
refuses the registration. -/
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

/-! The search below keeps the A5b and A5c guards on (`stepG {
noSingleWhileSeated := false }` is `WorkerState.stepRestricted` without A6),
because `WorkerState.sys` already has a counterexample on these events through
the A5b window: the counterexample found here is one that A5b and A5c do not
exclude, so A6 is needed for `reachable_worker_safe`. -/

#guard (ModelCheck.check (⟨WorkerState.init, WorkerState.stepG { noSingleWhileSeated := false }⟩ :
    System WorkerState WorkerEv) mixedCheckEvents workerViolates 32 matches .counterexample _)

/-- Outside A7 (`WorkerState.step` and `WorkerState.stepRestricted` without the
`oneSessionPerSlot` guard; the C++ does not refuse the registration): a second
sub-plan `1` registered into an unanchored entry overwrites the slot
(`progress.sub_sessions[sub_plan_index] = session`); the first session, no
longer in `sub_sessions`, settles into the `stray` arm and is reported as the
request's `done`. `WorkerState.step` and `WorkerState.stepRestricted` (A7)
refuse the second registration; running with only `oneSessionPerSlot := false`
shows A7 is needed by `reachable_worker_safe`. -/
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

/-- Replays `ReshardStackTest.WindowedRequestChainsOneSubTransferPerWindow`
(`reshard_service_test.cc`): 2 source ranks `{0, 1}`, 4 single-page windows
`[[0], [0], [1], [1]]`, 1 destination `10`; each window arms `10` then dispatches
its owning rank under a single registry claim (`claimOwner = 1`), ending at
`transferStatus = .completed` with the registration still `.claimed` so
`CancelIfUnclaimed` returns `false` (`cancelled = false`). -/
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
(`reshard_service_test.cc`) and continues it with sender `Complete` votes:
prefix clip plus a 2-page window dispatches `[0, 1]` in window `0` and `[1]` in
window `1`; the A3 guard lets rank `0`, absent from window `1`, vote `Complete`
once window `0` has finished dispatching, and the claim stays until rank `1`
votes after the chain ends. The vote is what the coordinator model admits, not
what the test drives: on its worker, rank `0` fills one of two `sub_sessions`
slots and is reported `failed_recving`, not `done_sending`
(`trace_sender_in_first_window_only`), and whether the producer then votes is
not determined by the cited C++. -/
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

/-- With `maxArmAttempts = 2` (parameterizing `kChainArmAttempts = 12`), the
second busy refusal is final: `finishWindowArm` aborts the chain (`.failed`)
and the claim is kept (`index == 1`). -/
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

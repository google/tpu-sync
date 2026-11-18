import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Model
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Invariant
import TpuSyncVerify.Transfer.Reshard.SubPlanChain.Traces

/-!
# Reshard sub-plan chains (`PlanSubTransfers`, `StartChain`, `SubPlanChainProgress`)

A pool reshard whose destination has less staging than the request
(`dst_window_blocks`) runs as a chain of sub-plans, one per window. This module
models the three places that know about the chain:
1. **The planner** — `PlanSubTransfers` and `SubPlanUuid` in `coord.cc`: how
   a request's destination block lists are split by window, how shorter and
   unwindowed tags are right-aligned into the last sub-transfers, when a
   windowed tag that shares destination block IDs with another tag disables
   windowing, and how `index >= 1` sub-plans get their UUIDs.
2. **The worker** — `KVCacheManagerWithTransfer::SubPlanChainProgress`,
   `NoteSubPlanRegisteredLocked`, `RecordPoolReshardTerminalLocked`, and
   `SweepSubPlanChainsLocked` in `mgr.{h,cc}`: how one worker folds the
   terminals of a request's sub-plan sessions into the single `done_recving` /
   `done_sending` / `failed_recving` that `CompleteReadWithDetails` publishes.
3. **The coordinator** — `ReshardCoordinator::StartPoolReshard`,
   `ExecutePoolReshard`, `StartChain`, and `RunSubPlan(ctx, 0 .. K - 1)` in
   `coord.cc`, composed with the request's `RequestBlockRegistry` generation
   (`Reshard/Registry.lean`): one claim across the windows, each window's
   arm-then-dispatch over the per-window `plan.src_units`, the
   `ArmRefusedForBusyStaging` re-arm loop, and when `transfer_status_` reads
   `kCompleted` or `kFailed`.

One sub-plan on its own — its two-stage barrier, its receiver and sender
sessions — is `Reshard/Coordinator.lean`, `Reshard/Receive.lean`,
`Reshard/Send.lean`, and `Reshard/Pipeline.lean`; this module takes those as
given and models what the chain adds.

`coord.h`/`coord.cc` are `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`;
`mgr.h`/`mgr.cc` are `tpu_sync/kv_cache/kv_cache_manager_with_transfer.{h,cc}`;
`reshard_send.cc` is `tpu_sync/kv_cache/reshard_send_session.cc`;
`reshard_recv.cc` is `tpu_sync/kv_cache/reshard_receive_session.cc`;
`planner.cc` is `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`;
`reg.cc` is `tpu_sync/kv_cache/reshard/request_block_registry.cc`;
`reshard_client.cc` is `tpu_sync/kv_cache/reshard/reshard_client.cc`;
`reshard_service.h` is `tpu_sync/kv_cache/reshard/reshard_service.h`;
`pool_reshard_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`; and
`reshard_service_test.cc` is
`tpu_sync/kv_cache/reshard/reshard_service_test.cc`. Exact line ranges for
every symbol, sub-block, and test are in the generated `## Source map` table
below.

## State

### Worker (`WorkerState`, one worker, one `req_id`)

| Field | C++ | Role |
|---|---|---|
| `chain` | `sub_plan_chains_[req_id]` (`SubPlanChainProgress`: `sub_sessions`, `completed`, `failed`, `anchored`) | `present = false` is absence from the map; sender workers (`SessionRole::kPoolReshardSend`) and receiver workers (`SessionRole::kPoolReshardReceive`) use the same per-`req_id` map |
| `sessions` | the pool-reshard `SessionEntry`s (`SessionRole::kPoolReshardSend` / `kPoolReshardReceive`) registered for `req_id`, with `(sub_plan_index, sub_plan_count)`, `IsDraining()` (`draining_` / `finalizing_`), `in_flight_ > 0` (`inFlight`), `Done()`, and `GetStatus().ok()` | `retired` marks erasure from `sessions_` by `RetireSettledSessionLocked`; the chain entry keeps its `shared_ptr` (`active` is the unretired ones) |
| `doneReports`, `failReports` | insertions of `req_id` into `done_recving_` / `done_sending_` and into `failed_recving_` | what `CompleteReadWithDetails` publishes |
| `doneLog` (ghost) | — | for each `done` insertion, `sub_sessions.size()` and the settled slots at that moment (`DoneAfterAllSubPlans`) |
| `seenUuids` (ghost) | — | UUIDs registered so far (A2) |

### Coordinator (`CoordChainState`, one request)

| Field | C++ | Role |
|---|---|---|
| `allSrcUnits`, `winSrcUnits`, `dstUnits` | `args.src_units`; per-window `plan.src_units` (`BuildPoolReshardPlan` keeps the ranks with spans in the window); `args.dst_units` | the topology; `winSrcUnits[w] ⊆ allSrcUnits` |
| `transferStatus` | `transfer_status_[req_id]` | `notStarted`, `inProgress`, `completed`, `failed` |
| `winIdx`, `winPhase`, `armAttempts` | the running `RunSubPlan(ctx, index)`, its stage, and `attempt` of the re-arm loop | `RunSubPlan(ctx, 0)` runs inline, `1 .. K - 1` on `StartChain`'s thread |
| `maxArmAttempts` | `kChainArmAttempts` | `attempts = index == 0 ? 1 : kChainArmAttempts`; the constant is `12`, the model's default is `2` (a small bound for the searches) and a trace that needs more passes its own |
| `armResults`, `sendResults` | `arm_status[i]` / `armed[i]`, `sender_status[i]` of the running window | an arm result is `ok`, `refusedBusy` (`ArmRefusedForBusyStaging`), or `failed` |
| `dstStates` | per destination: an earlier window's `ReshardReceiveSession` still holding its `PoolStagingLease` (`prevSessionActive`), the running window's receiver armed (`curWinArmed`) | a busy destination refuses the arm |
| `activeSenders` | ranks with a dispatched `ReshardSendSession` that have not voted `Complete` | — |
| `activeSenderExposed` (ghost) | — | latched if a rank in `activeSenders` is left without a registry claim (`NoSenderExposed`) |
| `reg` | the request's `RequestBlockRegistry` generation (`Registry.RegGen`) | shared by all windows through `ctx->claim_owner` |

## Events

| System | Event | C++ |
|---|---|---|
| worker | `registerSubPlan uuid index count` | `PoolReshardRegisterRecv` / `PoolReshardPush` seating the session and calling `NoteSubPlanRegisteredLocked` |
| worker | `beginOp uuid` | `TryBeginOp` / `Start` / `ExecuteEligiblePoolH2ds` entering `in_flight_ > 0` |
| worker | `latchDrain uuid ok` | `FinishPoolH2d` / `RecordPushCompletion` / `Finish` latching `draining_` / `finalizing_` while `in_flight_ > 0` (`!Done()`) |
| worker | `settleSession uuid ok` | `EndOp` / `EndRecvOpLocked` settling a draining session (`ok = true` preserves `s.ok`; `ok = false` models a late `Finish(error)` before `EndOp` in `ReshardReceiveSession::FinishLocked`, over-approximating `ReshardSendSession::Finish`), or `Finish` settling an idle (`in_flight_ == 0`) session synchronously; `ok` is fixed at settlement (see the lock-scope row) |
| worker | `poll stalled`, `pollMidSettle stalled midSettle` | `CompleteReadWithDetails`: the `if (!session->IsDraining())` `session->deadline() <= now` arm, `RecordPoolReshardTerminalLocked` for each `Done()` session (from `RetireSettledSessionLocked`), optional `EndOp` settlements `midSettle` in the `session->mu_` gap before the sweep, then `SweepSubPlanChainsLocked(now)`; `stalled` is `now - progress.last_activity > stall` |
| coordinator | `registerProducer srcUnit` | `RequestBlockRegistry::Register` |
| coordinator | `startChainCoordination` | `StartPoolReshard` setting `kInProgress` |
| coordinator | `startWindowPlan planOk` | `RunSubPlan` Stage 0: `LookupAndClaim`, `BuildPoolReshardPlan` |
| coordinator | `armDstResult dstUnit outcome` | one Stage 1 `PoolReshardRegisterRecv` RPC returning |
| coordinator | `retryBusyArm` | the Stage 1 join setting `retry = true` and re-arming the unarmed destinations |
| coordinator | `finishWindowArm` | the Stage 1 join dispatching, or `return arm_status[i]` |
| coordinator | `sendSrcResult srcUnit ok` | one Stage 2 `PoolReshardPush` RPC returning |
| coordinator | `finishWindowSend` | the Stage 2 join: next window, final status, or `return status` |
| coordinator | `settleDispatchedRecv dstUnit` | a dispatched window's receiver settling and releasing its lease |
| coordinator | `senderVoteComplete srcUnit` | the producer calling `RequestBlockRegistry::Complete` for a dispatched rank after its last window (A3) |
| coordinator | `cancelIfUnclaimed` | the producer calling `RequestBlockRegistry::CancelIfUnclaimed` |

## Lock scopes

| Event | Lock scopes |
|---|---|
| `registerSubPlan` | the `mu_` scope of `PoolReshardRegisterRecv` / `PoolReshardPush` that inserts into `sessions_` and calls `NoteSubPlanRegisteredLocked`. Each function checks `sessions_` for the UUID in an earlier `mu_` scope and creates the session unlocked in between: `PoolReshardRegisterRecv` reserves the UUID in `pending_pool_reshard_recvs_` across that gap, `PoolReshardPush` reserves nothing (its check and its insert are separate scopes with no re-check); A2 covers the gap |
| `beginOp`, `latchDrain`, `settleSession` | one `mu_` scope each on the session (`Reshard/Send.lean`, `Reshard/Receive.lean`); `ReshardReceiveSession::FinishLocked` updates `status_` on error before `if (draining_) return;` (`settleSession uuid false` while `draining`), whereas `ReshardSendSession::Finish` returns early when `finalizing_`. That `status_` update has no `done_` test, so a `Finish(error)` on a receive session that is already `Done()` still turns an OK status into an error, and `RecordPoolReshardTerminalLocked` re-reads every slot's `GetStatus()` on each call; the model fixes `ok` at settlement. The change is OK-to-error only, so it can withhold a `done` but not produce one |
| `poll`, `pollMidSettle` | one `KVCacheManagerWithTransfer::mu_` scope of `CompleteReadWithDetails`, which acquires `session->mu_` separately in `!session->IsDraining()`, `RetireSettledSessionLocked` (`reshard->Done()`), and `SweepSubPlanChainsLocked` (`!sub_session->Done()`); `pollMidSettle` models `EndOp` completing under `session->mu_` between the second and third acquisitions (enabled in `WorkerState.step` and in the alternative `WorkerState.stepSweepWaitsForOpen`; refused by `WorkerState.stepRestricted`, A5c). The model applies the deadline arm to every session and then records the settled ones in registration order; the C++ loop arms and retires session by session in `sessions_` iteration order, and other orders are not enumerated |
| `registerProducer`, `senderVoteComplete`, `cancelIfUnclaimed` | `RequestBlockRegistry::mu_` |
| `startChainCoordination`, `finishWindowSend` (last window), `finishWindowArm` / `startWindowPlan` / `finishWindowSend` (abort) | `status_mu_` for `transfer_status_`; `RunSubPlan`'s registry calls under `RequestBlockRegistry::mu_` |
| `startWindowPlan`, `armDstResult`, `retryBusyArm`, `finishWindowArm`, `sendSrcResult` | `RunSubPlan`'s own thread joins (`armers`, `senders`); the RPCs run unlocked and report through per-index slots |
| `settleDispatchedRecv` | the receiver's `mu_` and `PoolStagingStorage`'s lock on release |

`RunSubPlan(ctx, 0)` returns before `StartChain` spawns `RunSubPlan(ctx, 1)`
(`ExecutePoolReshard`), so the windows of one request never overlap on the
coordinator; what overlaps is a dispatched window's data plane with the next
window's arming, which is what `prevSessionActive` and `retryBusyArm` model.

## Assumptions

The worker model has two transition functions, both instances of
`WorkerState.stepG (g : Guards)`:

* `WorkerState.step` (`WorkerState.sys`) is the C++ under A1, fresh UUIDs (A2),
  A6 and A7 (same-UUID re-registration after settlement is not modelled):
  A1, A2, A6 and A7 are guards of `registerSubPlan` (a registration that
  violates one returns `none`), and `poll` runs the A5a deadline arm, which is
  what the C++ does. A5b and A5c are **not** guards of `WorkerState.step`.
* `WorkerState.stepRestricted` (`WorkerState.sysRestricted`) is
  `WorkerState.step` further restricted to executions that satisfy A5b and A5c.
  `reachable_worker_safe` is proved for this system.

**A5b and A5c are not properties of the C++ and not assumptions about its
callers.** Each excludes a timing window inside `CompleteReadWithDetails` /
`SweepSubPlanChainsLocked` that the C++ permits with every caller well behaved,
and `WorkerState.step` violates `DoneAfterAllSubPlans` in each
(`trace_draining_ok_at_sweep_reports_done_after_failure`,
`trace_mid_poll_endop_unanchored_reports_done_after_stall`). The alternative
`WorkerState.stepSweepWaitsForOpen` (not the C++) is proved safe without them.

Who establishes the caller assumptions differs. A1 is established by the
coordinator. A2 is enforced only in part. **A6 and A7 are not enforced by
anything in the C++**: it accepts registrations outside them, and two theorems
run `WorkerState.step` without that one guard to show what the C++ does there
(`trace_single_plan_registration_erases_live_chain`,
`trace_slot_overwrite_reports_stray_done`).

`CoordChainState.step` carries A3, A4 and A8 as guards: an event that violates
one returns `none`, so every theorem about its reachable states is a statement
about executions that satisfy them. A3 is a caller contract outside the cited
C++; A4 and A8 are modelling restrictions.

* **A1 — `0 <= sub_plan_index < max(1, sub_plan_count)`.** `RunSubPlan` stamps
  every sub-plan with its `index` in `0 .. chain.count - 1`
  (`BuildStartTransferForTarget`). Guard: `registerSubPlan` excludes indices
  `>= max 1 count` (`NoteSubPlanRegisteredLocked` would grow `sub_sessions` to
  `sub_plan_index + 1` slots that no sub-plan fills).
* **A2 — sub-plan UUIDs are fresh (enforced only in part).** Guard:
  `registerSubPlan` excludes a UUID already in `seenUuids`, so a worker
  registers each UUID at most once. What the C++ provides: `SubPlanUuid(uuid,
  index)` is a pure function of the request UUID and the index (`index == 0` is
  the request UUID itself; `subPlanUuid_pos_of_pos` gives positivity, and
  `trace_sub_plan_uuid_positive_and_distinct` checks distinctness for indices
  `0 .. 63` of the one UUID in
  `SubPlanUuidTest.DerivedUuidsArePositiveAndDistinct`; distinctness is not
  proved in general), and `PoolReshardPush` / `PoolReshardRegisterRecv` refuse at entry,
  with `ALREADY_EXISTS`, a UUID whose session in `sessions_` has not settled.
  What it does not provide:
  - *Fresh request UUID per retry.* `ReshardCoordinator::ExecutePoolReshard`
    takes the request UUID from its caller (`int64_t uuid = args.uuid;`, drawing
    a random one only when the wire carries none), and a re-coordination of the
    same producer registration has to reuse it, because the registry generation
    is keyed `(req_id, uuid)`
    (`ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders` retries under the
    same UUID). Whether a retry reaches a worker under new sub-plan UUIDs is a
    property of the caller that nothing in this repository enforces.
  - *Same-UUID re-registration after settlement.* When the UUID's earlier
    session has settled, both worker entry points retire it inside the
    registration call (`RetireSettledSessionLocked`) and seat a new session
    under the same key. This is not modelled.
  - *Concurrent pushes of one UUID.* `PoolReshardPush` checks `sessions_` and
    inserts into it in separate `mu_` scopes and reserves nothing in between
    (only `PoolReshardRegisterRecv` reserves the UUID, in
    `pending_pool_reshard_recvs_`), so the C++ does not exclude two concurrent
    pushes of one UUID both creating and starting a session, the second replacing
    the first in `sessions_` and in its `sub_sessions` slot (leaving the first
    to run unpolled); the guard does.

  `PoolReshardRegisterRecv`'s duplicate-plan lease handling is modelled in
  `Reshard/PoolStaging.lean`.
* **A3 — a rank votes `Complete` only after its last window (caller
  contract).** The only C++ that sends `COMMAND_COMPLETE_REQUEST_BLOCKS` is
  `ReshardClient::CompleteRequestBlocks`; when it is called is decided by the
  engine driving the producer, outside the cited C++, so A3 cannot be checked
  in this repository. Guard: `senderVoteComplete` is enabled for a dispatched
  rank (in `activeSenders`) that is not in `plan.src_units` of the running or a
  later window of a chain in progress. The guard thus admits a vote at any time
  after the rank's last window was dispatched. It does not require the rank's
  worker to have reported `done_sending` (the worker model is not composed with
  the coordinator model), and a rank with spans in fewer than `K` windows never
  gets `done_sending` from its worker at all: it gets `failed_recving` after the
  stall timeout (`trace_sender_in_first_window_only`). What the producer does
  with that report, and whether it then votes, is not determined by the cited
  C++; `trace_window_composes_with_prefix_clip` shows the vote the guard admits
  for such a rank. Without the guard a rank's vote could retire the claim while
  a window that still dispatches that rank is being armed.
* **A4 — producers register before coordination starts
  (`ReshardCoordinator::StartPoolReshard`).** `registerProducer` is
  enabled only before `startChainCoordination`. The registry refuses `Register`
  once the claim is taken (`Reshard/Registry.lean`); a registration landing
  between `StartPoolReshard` and window 0's `LookupAndClaim` is the
  single-window case of `Reshard/Coordinator.lean`.
* **A5 — deadline arm, tombstone sweep, and intra-poll `EndOp` window
  (`Guards.deadlineArm`, `Guards.noDrainingOkAtSweep`,
  `Guards.noSettleBetweenRetireAndSweep`).** A5a is what the C++ does and is
  part of `WorkerState.step`. A5b and A5c are guards of
  `WorkerState.stepRestricted` only: nothing in the C++ enforces them and no
  caller behaviour establishes them.
  - *A5a (`Guards.deadlineArm`):* `DeadlineFromNow()` is `timeout_s_` from a
    session's creation and `NoteSubPlanRegisteredLocked` sets `last_activity` on
    every chained registration, so when an entry has been quiet for `timeout_s_`
    (`stalled`), every session in `sessions_` is past `session->deadline()`, and
    `CompleteReadWithDetails` executes
    `if (!session->IsDraining()) { ... session->Finish(DeadlineExceededError(...)); }`
    before `RetireSettledSessionLocked` and `SweepSubPlanChainsLocked`. An idle
    session (`in_flight_ == 0`) sets `done_ = true` synchronously inside
    `Finish` and is retired in that same poll
    (`trace_idle_seated_session_settles_at_deadline_poll`); an in-flight
    `!IsDraining()` session (`in_flight_ > 0`) latches `draining_` /
    `finalizing_` with error while staying `!Done()` until `EndOp`
    (`trace_sub_plan_seated_at_sweep_is_failed_not_done`,
    `trace_stall_sweep_keeps_entry_with_open_session`). Without A5a, an
    in-flight `!IsDraining()` session surviving a tombstone sweep settles OK
    into `it == sub_plan_chains_.end()` as the request's `done`
    (`mutant_no_deadline_arm_reports_done_after_sweep`).
  - *A5b (`Guards.noDrainingOkAtSweep`):* Because `CompleteReadWithDetails`
    gates its deadline arm on `if (!session->IsDraining())`
    (`ReshardReceiveSession::IsDraining` returns `draining_`;
    `ReshardSendSession::IsDraining` returns `finalizing_`, and
    `ReshardSendSession::Finish` returns early on `if (done_ || finalizing_) return;`),
    a sibling sub-plan already in `IsDraining() && !Done()` with `OkStatus()` —
    `ReshardSendSession::RecordPushCompletion` having set `finalizing_ = true`
    while `ReshardSendSession::EndOp` runs unlocked
    `ReleaseDataConnectionsAfterSend`, or
    `ReshardReceiveSession::FinishPoolH2d` having called `Finish(OkStatus())`
    before `self->EndOp()` — is skipped by the deadline arm. Meanwhile,
    `SweepSubPlanChainsLocked` erases `progress.failed || progress.completed`
    entries unconditionally (checking `!sub_session->Done()` only in the live
    `!failed && !completed` arm; cf.
    `trace_draining_session_on_live_chain_survives_stalled_poll`).
    `DoneAfterAllSubPlans` therefore requires that no unretired multi-sub-plan
    session be in `IsDraining() && !Done()` with `OkStatus()` when a
    `failed || completed` tombstone is swept (`Guards.noDrainingOkAtSweep`, a
    guard of `WorkerState.stepRestricted`). C++ does **not** enforce A5b: if
    `EndOp` / `FinishPoolH2d` straddles the tombstone sweep of a failed chain,
    that sibling sub-plan settles on a later
    poll into `it == sub_plan_chains_.end()` and emits `done_sending_` /
    `done_recving_` on a single non-retried failed request
    (`trace_draining_ok_at_sweep_reports_done_after_failure`).
  - *A5c (`Guards.noSettleBetweenRetireAndSweep`):* Because
    `CompleteReadWithDetails` holds `KVCacheManagerWithTransfer::mu_` without
    holding `session->mu_`, a draining session skipped by `!session->IsDraining()`
    and seen as `!Done()` by `RetireSettledSessionLocked` can finish `EndOp`
    (`done_ = true`) before `SweepSubPlanChainsLocked` checks
    `!sub_session->Done()` (`pollMidSettle`, which `WorkerState.stepRestricted`
    refuses). C++ does **not** enforce A5c: on an unanchored live
    chain (`progress.anchored == false`, `sub_plan_index > 0`), that mid-poll
    `EndOp` makes `!sub_session->Done()` read `false`, so
    `SweepSubPlanChainsLocked` marks the chain stalled (`progress.failed = true`)
    and executes `progress.sub_sessions.clear()` with `anchored` still `false`;
    on the next poll, `RecordPoolReshardTerminalLocked` checks
    `sub_it == progress.sub_sessions.end()` before
    `if (progress.failed || progress.completed) return;`, hits
    `!progress.anchored`, and emits `done_sending_` / `done_recving_` after
    `failed_recving_` (`trace_mid_poll_endop_unanchored_reports_done_after_stall`).
    Likewise on a `failed || completed` chain, a session finishing `EndOp` in
    that gap is `Done() == true` yet still unretired in `sessions_`. Checking
    `progress.failed || progress.completed` before `!progress.anchored`
    (`recordTerminalWaitOpen`) and keeping `failed || completed` tombstones in
    `SweepSubPlanChainsLocked` while `sessions_` holds any unretired
    `kPoolReshard*` entry for `req_id` (`sweepStalledChainWaitOpen`) is the
    alternative `WorkerState.stepSweepWaitsForOpen`, which is not the C++ and
    under which the property holds without A5b and A5c
    (`reachable_sweepWaitsForOpen_safe`).
* **A6 — no single-plan registration while a sub-plan is seated
  (`Guards.noSingleWhileSeated`; not enforced by the C++).** C++ worker RPCs (`PoolReshardPush`,
  `PoolReshardRegisterRecv`) and `NoteSubPlanRegisteredLocked` do **not** refuse
  a `sub_plan_count <= 1` arrival while multi-sub-plan sessions of the same
  `req_id` are seated in `sessions_`: `NoteSubPlanRegisteredLocked`
  unconditionally executes `sub_plan_chains_.erase(req_id)` when
  `sub_plan_count <= 1`. Guard: `WorkerState.step` excludes `registerSubPlan`
  with `count <= 1` while an unretired `count > 1` session is seated
  (`Guards.noSingleWhileSeated`). Callers can break A6 in two ways
  (`trace_single_plan_registration_erases_live_chain`): (i) two pipelined
  `ReshardCoordinator::StartPoolReshard` stages sharing one producer registration
  `req_id` (as across disjoint worker endpoints in
  `ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim`) if run on an
  overlapping worker where one stage plans `count > 1` and another plans
  `count = 1`, or (ii) a same-`req_id` retry whose `PlanSubTransfers`
  shrinks from `count > 1` to `count = 1` while an earlier sub-plan session is
  still seated.
* **A7 — one session per `sub_plan_index >= 1` slot
  (`Guards.oneSessionPerSlot`; not enforced by the C++).**
  `NoteSubPlanRegisteredLocked` unconditionally
  overwrites `progress.sub_sessions[sub_plan_index] = session` without checking
  whether that slot already holds an unretired session. Guard:
  `WorkerState.step` excludes `registerSubPlan` at `index >= 1` when
  `w.chain.present` and `subSessions[index]` is already filled
  (`Guards.oneSessionPerSlot`; `index == 0` re-anchors a fresh entry). Even when
  a retry carries a new request UUID (A2), a **sparse-window sender**
  whose spans appear only in window `index > 0` (`BuildPoolReshardPlan` in
  `planner.cc` skips empty-span senders per window) never receives `index = 0`,
  leaving `progress.anchored = false`; on a same-`req_id` retry while Attempt
  1's window-`index` session is still seated, Attempt 2 again dispatches only
  `index > 0` (never re-anchoring at `0`), overwriting `sub_sessions[index]` so
  Attempt 1's session misses `std::find` on `sub_sessions` and hits the
  `!progress.anchored` stray arm to emit premature `done_sending_`
  (`trace_slot_overwrite_reports_stray_done`).
* **A8 — single-window exclusive destination staging capacity
  (`CoordChainState`).** In `CoordChainState.step`, `armDstResult` equates
  staging contention (`busy`) with `prevSessionActive`, assuming each
  destination's bounded `PoolStagingStorage` is dedicated to this chain and fits
  at most one window's `PoolStagingLease` at a time. When `staging_leases == 0`
  (unbounded full-mirror storage) or spare arena capacity exists, two windows'
  receivers can stage concurrently; conversely, external transfers sharing the
  arena can trigger `ArmRefusedForBusyStaging` even when
  `prevSessionActive = false`.
* **Scope / not modelled:**
  - *Coordinator shutdown (`stopping_`):* `!stopping_.load()` in the re-arm
    condition (a coordinator shutting down aborts instead of re-arming;
    `retryBusyArm` assumes it is not stopping) and `StartChain`'s
    `if (stopping_.load())` before each `RunSubPlan(ctx, index >= 1)`, which
    ends the chain as `kFailed` with the claim kept.
  - *Registry clock and forced `Release`:* `ChainEv` has no TTL-expiry event
    for `PurgeExpiredLifecycleLocked` / `PurgeExpiredRequestBlocksLocked` and no
    `RequestBlockRegistry::Release` between a claim and the end of the chain.
    The claim is refreshed to `now + ttl_s_` only by each window's
    `LookupAndClaim`, while one window's arm loop can run `kChainArmAttempts`
    (12) rounds of worker RPCs each bounded by `kWorkerRpcTimeout` (60 s)
    against a default `request_registry_ttl_s` of 600 s, so `NoSenderExposed`
    and `ClaimHeldAcrossWindows` are statements about chains in which every
    window is claimed again within `ttl_s_` of the previous claim and nobody
    releases the generation.
  - *Same-UUID re-registration and pipelined stages:* same-UUID re-registration
    after settlement (A2), and pipelined destination stages sharing one `req_id`
    on an overlapping worker
    (`ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim`), which
    share `sub_plan_chains_[req_id]` and are outside the
    single-chain-per-`req_id` worker model.
  - *Delivery mode, details, and role routing (`is_recv`):* the `kController`
    delivery mode's null-controller error path, `transfer_details_`, and which
    of `done_recving_` / `done_sending_` a `done` terminal lands in.
    `sub_plan_chains_` is shared by sender workers
    (`SessionRole::kPoolReshardSend` → `done_sending_` / `failed_recving_`) and
    receiver workers (`SessionRole::kPoolReshardReceive` → `done_recving_` /
    `failed_recving_`), while a single worker cannot hold both roles on the same
    sub-plan (`BuildStartTransferForTarget` sets `is_sender = true` whenever
    sender schedules are non-empty, the worker entry points refuse at entry,
    with `ALREADY_EXISTS`, a `SubPlanUuid(uuid, index)` whose session in
    `sessions_` has not settled, and `sub_sessions[index]` holds a single
    pointer).

## Properties

* `subPlanUuid_zero`, `subPlanUuid_pos_of_pos`: `SubPlanUuid(uuid, 0) = uuid`
  and `0 < SubPlanUuid(uuid, index) <= 0x7FFFFFFFFFFFFFFF` for `index > 0`.
* `planSubTransfers_no_aliased_window_split`: a multi-window chain is planned
  only when no windowed tag shares a destination block ID with another tag, so
  the per-plan `RankWaveOrder` on aliased device pages (`Reshard/Receive.lean`)
  is never split across sub-plans.
* `reachable_worker_safe` (`DoneAfterAllSubPlans`): in every reachable state of
  `WorkerState.sysRestricted`, every `done_recving` / `done_sending` insertion
  was made with every slot of the request's `sub_sessions` (at least
  `sub_plan_count` of them) settled OK, or by a single-plan session settled OK.
  `WorkerState.stepRestricted` is `WorkerState.step` (A1, A2, A6 and A7 as
  guards, with the A5a deadline arm) restricted to executions that satisfy A5b
  and A5c. The property does **not** hold for `WorkerState.step`: A5b and A5c
  are not properties of the C++, and
  `trace_draining_ok_at_sweep_reports_done_after_failure` and
  `trace_mid_poll_endop_unanchored_reports_done_after_stall` are executions of
  `WorkerState.step` that violate it. It is therefore not a
  statement about every execution of `KVCacheManagerWithTransfer`, even with
  well-behaved callers.
* `reachable_sweepWaitsForOpen_safe`: the same property in every reachable
  state of the alternative `WorkerState.stepSweepWaitsForOpen` (not the C++),
  which needs neither A5b (`noDrainingOkAtSweep = false`) nor A5c
  (`noSettleBetweenRetireAndSweep = false`); A1, A2, A6 and A7 remain guards.
* `reachable_coordChain_safe` (`CoordChainSafe`), for every topology whose
  windows are non-empty subsets of the claimed snapshot (`ValidChainCfg`):
  `Registry.Inv`; no dispatched rank is ever left without the registry claim
  (`NoSenderExposed` — the `if (index == 0)` guard on `AbandonClaim`); no
  sender of a window is dispatched while the window arms
  (`NoPushBeforeAllArmed`); every destination is armed while a window
  dispatches (`AllArmedWhileDispatching`); under A8, a destination never has two
  windows' receivers staged at once (`OneWindowStagedPerDestination`);
  `transfer_status_` reads `kCompleted` only after the last window's Stage 2
  (`CompletedOnlyAtChainEnd`); and the claim is held from window 0's Stage 0
  to the end of the chain (`ClaimHeldAcrossWindows`).
* What the shipping code does beyond those guarantees is shown by traces
  (`Traces.lean`): a rank with spans in fewer than `K` windows never fills
  `K` slots and is failed by the stall sweep
  (`trace_sender_in_first_window_only`, `trace_sender_in_last_window_only`);
  the first failing sub-plan publishes `failed_recving` while sibling sessions
  are still active (`trace_first_failure_reports_while_sibling_active`); a
  sub-plan arriving after the failed entry was swept is failed a second time
  (`trace_late_sub_plan_after_sweep_reports_failure_twice`), as is an in-flight
  one still `!Done()` at the sweep
  (`trace_sub_plan_seated_at_sweep_is_failed_not_done`,
  `trace_stall_sweep_keeps_entry_with_open_session`), whereas an idle
  (`in_flight_ == 0`) one settles synchronously in the deadline poll before the
  sweep runs (`trace_idle_seated_session_settles_at_deadline_poll`) and a
  draining one on a live chain is kept by `!sub_session->Done()`
  (`trace_draining_session_on_live_chain_survives_stalled_poll`); a single-plan
  session beside an unanchored entry still reports
  (`trace_single_plan_beside_unanchored_entry_reports`); and a window
  `>= 1` failure keeps the claim until `ttl_s_` even after every dispatched rank
  has voted (`trace_chained_arm_failure_keeps_claim_past_last_vote`).
* Executions of `WorkerState.step` that violate `DoneAfterAllSubPlans`, each
  ending with a `done` logged for a two-sub-plan request with one slot settled:
  `trace_draining_ok_at_sweep_reports_done_after_failure` (the A5b window) and
  `trace_mid_poll_endop_unanchored_reports_done_after_stall` (the A5c window).
  `WorkerState.sysRestricted` refuses each trace, and
  `WorkerState.sysSweepWaitsForOpen` runs each without the `done`.
* Registrations the C++ accepts outside the caller assumptions A6 and A7
  (`WorkerState.step` without that one guard; `WorkerState.step` refuses each
  trace), ending the same way:
  `trace_single_plan_registration_erases_live_chain` (A6) and
  `trace_slot_overwrite_reports_stray_done` (A7).
* Mutants (variants that are not the C++): counting a superseded chain's
  sub-plan (`mutant_counting_stale_sub_plan_reports_done_early`), not
  re-anchoring on sub-plan 0 (`mutant_no_reanchor_never_reports_retry_done`),
  no deadline arm, A5a (`mutant_no_deadline_arm_reports_done_after_sweep`), and
  `AbandonClaim` at every index
  (`mutant_abandon_at_every_index_exposes_active_sender`).
* Searches (`workerViolates`, `coordViolates`, both `!decide (Safe s)`) exhaust
  a two-window worker (single chain, retry re-anchoring, intra-poll `EndOp`
  settlements, and alongside a single-plan request: safe under
  `WorkerState.sysRestricted` and `WorkerState.sysSweepWaitsForOpen`, with a
  counterexample under `WorkerState.sys` in the A5b and A5c windows) and two
  two-window coordinator topologies.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `SubTransferSlice` | `:97-102` |
| `coord.h` | `SubTransferChain` | `:108-111` |
| `coord.h` | `ReshardCoordinator` | `:138-207` |
| `coord.cc` | `BuildStartTransferForTarget` | `:111-274` |
| `coord.cc` | `*` (`struct ReshardCoordinator::ChainContext {` .. `std::atomic<bool> finished{false};`) | `:313-324` |
| `coord.cc` | `*` (`constexpr absl::Duration kWorkerRpcTimeout = absl::Seconds(60);`) | `:60` |
| `coord.cc` | `SubPlanUuid` | `:359-370` |
| `coord.cc` | `PlanSubTransfers` | `:372-475` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:477-529` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:531-604` |
| `coord.cc` | `ReshardCoordinator::ReapFinishedChainsLocked` | `:606-615` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:617-652` |
| `coord.cc` | `ReshardCoordinator::WaitForSubTransferChains` | `:654-663` |
| `coord.cc` | `ArmRefusedForBusyStaging` | `:670-673` |
| `coord.cc` | `*` (`constexpr int kChainArmAttempts = 12;`) | `:677` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:681-935` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`const int attempts = index == 0 ? 1 : kChainArmAttempts;` .. `if (!retry) break;`) | `:752-793` |
| `planner.cc` | `BuildPoolReshardPlan` (`PoolReshardPlan plan;` .. `plan.dst_units = request.dst_units;`) | `:1308-1314` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`enum class SessionRole {` .. `SessionRole role;`) | `:295-307` |
| `mgr.h` | `KVCacheManagerWithTransfer` (`struct SubPlanChainProgress {` .. `void SweepSubPlanChainsLocked(std::chrono::steady_clock::time_point now)`) | `:382-409` |
| `mgr.cc` | `KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked` | `:148-184` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:186-252` |
| `mgr.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:254-303` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:733-775` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1076-1139` |
| `reshard_send.cc` | `ReshardSendSession::IsDraining` | `:117-120` |
| `reshard_send.cc` | `ReshardSendSession::RecordPushCompletion` | `:699-718` |
| `reshard_send.cc` | `ReshardSendSession::Finish` | `:720-738` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:754-782` |
| `reshard_recv.cc` | `ReshardReceiveSession::IsDraining` | `:116-119` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishLocked` | `:657-669` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndRecvOpLocked` | `:735-745` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:890-926` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredRequestBlocksLocked` | `:61-69` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredLifecycleLocked` | `:71-96` |
| `reg.cc` | `RequestBlockRegistry::Release` | `:302-315` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
| `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` | `:373-405` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-501` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:503-519` |
| `reshard_client.cc` | `ReshardClient::CompleteRequestBlocks` | `:355-360` |
| `reshard_service.h` | `*` (`double request_registry_ttl_s = 600.0;`) | `:47` |
| `pool_reshard_test.cc` | `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan` | `:921-957` |
| `pool_reshard_test.cc` | `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce` | `:959-982` |
| `pool_reshard_test.cc` | `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain` | `:984-1027` |
| `pool_reshard_test.cc` | `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout` | `:1029-1051` |
| `pool_reshard_test.cc` | `SubPlanChainTest.SinglePlanRequestsReportAtOnce` | `:1053-1065` |
| `pool_reshard_test.cc` | `SubPlanChainTest.SendTerminalWaitsForTheLastSubPlan` | `:1067-1095` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.NoWindowsPlanOneTransfer` | `:1020-1038` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.WindowSplitsPagedTagInDestinationOrder` | `:1040-1057` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.UnwindowedTagsRideInTheLastSubTransfer` | `:1059-1087` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.WindowedTagSharingIdsDisablesWindows` | `:1089-1108` |
| `reshard_service_test.cc` | `PlanSubTransfersTest.MalformedWindowsFailClosed` | `:1110-1122` |
| `reshard_service_test.cc` | `SubPlanUuidTest.DerivedUuidsArePositiveAndDistinct` | `:1124-1134` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowedRequestChainsOneSubTransferPerWindow` | `:1136-1196` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowClipsStraddlingSpansAtWindowBoundaries` | `:1198-1283` |
| `reshard_service_test.cc` | `ReshardStackTest.WindowComposesWithPrefixClip` | `:1285-1322` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedArmRetriesWhileReceiverStagingIsBusy` | `:1324-1349` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedRequestStaysInProgressUntilTheChainEnds` | `:1351-1373` |
| `reshard_service_test.cc` | `ReshardStackTest.ChainedArmFailureMarksTheTransferFailed` | `:1375-1397` |
| `reshard_service_test.cc` | `ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders` | `:1460-1476` |
| `reshard_service_test.cc` | `ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim` | `:1823-1848` |
<!-- source-map:end -->
-/

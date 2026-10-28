import TpuSyncVerify.Transfer.Reshard.Receive.Model
import TpuSyncVerify.Transfer.Reshard.Receive.Lemmas
import TpuSyncVerify.Transfer.Reshard.Receive.Invariant
import TpuSyncVerify.Transfer.Reshard.Receive.Progress
import TpuSyncVerify.Transfer.Reshard.Receive.Traces

/-!
# Reshard receive session

One `ReshardReceiveSession` on the destination (consumer) side and the slice of
`KVCacheManagerWithTransfer` that drives it (`PoolReshardRegisterRecv`,
`OnPoolReceived`, `begin_incoming_push`, `end_incoming_push`,
`MaybeUnregisterSettledReshardRecv`, and the `CompleteReadWithDetails` poll).
Sender sessions, multi-sender stream counting, and aliased-page memory contents
are modelled in `Reshard/Send.lean` and `Reshard/Pipeline.lean`.

`reshard_recv.h`/`reshard_recv.cc` are
`tpu_sync/kv_cache/reshard_receive_session.{h,cc}`; `mgr.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`; `bt.cc` is
`tpu_sync/transport/block_transport.cc`; `utils.h` is
`tpu_sync/core/utils.h`; `reshard_recv_test.cc` is
`tpu_sync/kv_cache/reshard_receive_session_test.cc`; and `pool_reshard_test.cc`
is `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.
Exact line ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

## How a reshard receive runs

`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` creates a
`ReshardReceiveSession` (`ReshardReceiveSession::Create`,
`ReshardReceiveSession::ReshardReceiveSession`), claiming the active plan
(`base->RegisterActivePlanDirect(plan.uuid(), plan, /*is_sender=*/false)`),
acquiring bounded per-UUID `PoolStagingLease`s across the plan's destination
storages (`ReshardReceiveSession::AcquireStagingLeases`), and inserting the
session into `active_pool_reshard_recvs_`
(`active_pool_reshard_recvs_[plan.uuid()] = std::move(recv_session)`). Each
transferred pool in `expected_pool_indices_` has an order rank in
`pool_order_ranks_` (`pool_order_ranks_[static_cast<size_t>(pool_idx)] = group.order_rank()`
in `ReshardReceiveSession::ReshardReceiveSession`).

As incoming push streams arrive at `BlockTransport::HandleIncomingPush` /
`BlockTransport::CompleteIncomingPush`, each stream takes a receive op via
`hooks.begin_incoming_push` (`ReshardReceiveSession::TryBeginRecvOp`) and
releases it in `hooks.end_incoming_push` (`ReshardReceiveSession::EndRecvOp`).
When all `expected_pushes` streams for a pool have landed, the transport fires
`block_delegate_->OnPoolReceived(l, uuid)` (`hooks.on_pool_received` →
`ReshardReceiveSession::OnPoolReceived`), which records the pool in
`started_pool_indices_` (`ReshardReceiveSession::RecordPoolReceivedLocked`) and
calls `ReshardReceiveSession::ExecuteEligiblePoolH2ds`.

`ExecuteEligiblePoolH2ds` selects under `mu_` every started, unlaunched pool
whose strictly lower-rank prerequisite pools are all in
`completed_pool_indices_` (`ReshardReceiveSession::CollectEligiblePoolH2dsLocked`),
marks them in `h2d_launched_pools_`, and adds `to_launch.size()` to `in_flight_`
(`in_flight_ += static_cast<int32_t>(to_launch.size())`). Outside `mu_`, it
loops over `to_launch`: for each pool it re-checks `if (done_ || draining_)`
under `mu_` (calling `EndRecvOpLocked()` and skipping the upload if the session
finished in the window), dispatches `base_->H2dPoolBlocks`, pushes the future
into `h2d_futures_` (`h2d_futures_.push_back(future)`), and attaches
`FinishPoolH2d` + `EndRecvOp` + `MaybeUnregisterSettledReshardRecv` to
`future.OnReady`.

When a pool's H2D finishes, `ReshardReceiveSession::FinishPoolH2d` calls
`ReshardReceiveSession::RecordPoolH2dResultLocked`. On an intermediate OK
completion (`if (!finished && status.ok())`), it calls
`ExecuteEligiblePoolH2ds(manager)` to launch the next rank wave before the outer
`future.OnReady` callback's `self->EndRecvOp()` releases the completed pool's
op. On the first error or when `completed_pool_indices_ == expected_pool_indices_`,
`reshard_finalizing_` is set (`finished = true`), `network_completed_ = true`
and `should_record_duration = true` are latched on success, and
`Finish(terminal_status)` drains the session.

## State

The settle protocol itself is `Transfer.Lifecycle` (`in_flight_`, `draining_`,
`done_`, `status_.ok()`, and `!staging_released_`). On top of it, each expected
pool is a `Pool := ⟨rank, stage⟩` in `pools : List Pool`:

| Pool stage   | C++ membership (`ReshardReceiveSession`)             | Holds op in `in_flight_` |
|--------------|------------------------------------------------------|--------------------------|
| `.waiting`   | in `expected_pool_indices_`, not yet in `started_pool_indices_` | no |
| `.received`  | in `started_pool_indices_`, not yet in `h2d_launched_pools_` | no |
| `.queued`    | in `h2d_launched_pools_` after `in_flight_ += static_cast<int32_t>(to_launch.size())` (`CollectEligiblePoolH2dsLocked`, `ExecuteEligiblePoolH2ds`), before the per-pool re-check lock (`if (done_ || draining_)`) | yes |
| `.issued`    | `H2dPoolBlocks` succeeded; future in `h2d_futures_` (`h2d_futures_.push_back(future)`) with `!IsReady()` | yes |
| `.ready`     | future in `h2d_futures_` has `IsReady()`, `OnReady` callback not yet completed | yes |
| `.completed` | in `completed_pool_indices_` (`RecordPoolH2dResultLocked`) and `self->EndRecvOp()` ran (`ExecuteEligiblePoolH2ds`'s `future.OnReady`) | no |
| `.aborted`   | op ended without completing: skipped at re-check (`EndRecvOpLocked()`), or `H2dPoolBlocks`/`OnReady` failed or saw `reshard_finalizing_` | no |

Session-level fields:

| Field              | C++                                                                 | Role |
|--------------------|---------------------------------------------------------------------|------|
| `pools`            | `expected_pool_indices_`, `pool_order_ranks_`, `started_pool_indices_`, `h2d_launched_pools_`, `h2d_futures_`, `completed_pool_indices_` | per-pool state |
| `life`             | `in_flight_`, `draining_`, `done_`, `status_`, `!staging_released_` (`ReshardReceiveSession`) | `Transfer.Lifecycle` |
| `finalizing`       | `reshard_finalizing_` (`ReshardReceiveSession`)                     | stops further H2D launches and results after first error or full completion |
| `networkCompleted` | `network_completed_` (set in `FinishPoolH2d`)                       | latched on final pool H2D success |
| `planUnregistered` | `!unregister_on_settle_` (claimed in `ReshardReceiveSession::TakePendingUnregister`) | active plan unregistered on settle |
| `durationRecorded` | ghost (`should_record_duration` / `RecordTransferDuration` in `FinishPoolH2d`) | end-of-transfer latency histogram recorded |
| `published`        | membership in `done_recving_` / `failed_recving_` (`CompleteReadWithDetails`) | what `poll_stats()` shows the engine |
| `pushes`           | ghost                                                               | incoming pushes between `TryBeginRecvOp` (`hooks.begin_incoming_push`) and `EndRecvOp` (`hooks.end_incoming_push`) |

## Events

| Event               | C++ |
|---------------------|-----|
| `pushBegin`         | `ReshardReceiveSession::TryBeginRecvOp` from `hooks.begin_incoming_push`, called in `BlockTransport::HandleIncomingPush` (`block_delegate_->BeginIncomingPush(header.uuid)`) |
| `pushEnd ok`        | `hooks.end_incoming_push`, called in `BlockTransport::HandleIncomingPush` (`block_delegate_->EndIncomingPush`). Parameterized by `PushEndPolicy`: `.finishOnError` (shipping) calls `reshard_session->Finish(status)` before `reshard_session->EndRecvOp()` on `ok = false` (matching the `TransferReceiveSession` branch), whereas `.endOpOnly` (mutant) runs `EndRecvOp()` unconditionally |
| `onPoolReceived p`  | `ReshardReceiveSession::OnPoolReceived` + `ExecuteEligiblePoolH2ds` first lock (`to_launch = CollectEligiblePoolH2dsLocked()`): records `p` in `started_pool_indices_` and batch-queues every eligible started pool |
| `h2dIssue p ok`     | `ExecuteEligiblePoolH2ds` loop body (`for (auto& [pool_idx, dst_chip_block_ids] : to_launch)`): re-checks `if (done_ || draining_)` under `mu_`, otherwise dispatches `base_->H2dPoolBlocks` |
| `h2dReady p`        | device finishes pool `p`'s H2D copy: its future in `h2d_futures_` becomes `IsReady()` |
| `h2dDone p ok`      | `ExecuteEligiblePoolH2ds`'s `future.OnReady` callback calling `FinishPoolH2d`, `EndRecvOp`, and `MaybeUnregisterSettledReshardRecv` |
| `pollReady`         | `CompleteReadWithDetails` sees `!session->IsDraining()` and `session->IsReadyToComplete()` and calls `session->Finish()` |
| `cancel`            | external `Finish(error)`: deadline (`CompleteReadWithDetails`), shutdown (`~KVCacheManagerWithTransfer`), or `UnregisterActivePlan` |
| `publish`           | `CompleteReadWithDetails` moves a settled session (`session->Done()`) into `done_recving_` or `failed_recving_`, claims `session->TakePendingUnregister(&generation)`, and erases it from `active_pool_reshard_recvs_` |

## Lock scopes

Every mutable field of `ReshardReceiveSession` is `ABSL_GUARDED_BY(mu_)`;
`base_`, `staging_allocator_`, `uuid_`, `req_id_`, `deadline_`, and
`start_time_` are `const`.

| Event | `mu_` scopes | Between them |
|---|---|---|
| `pushBegin` | 1: `ReshardReceiveSession::TryBeginRecvOp` (`if (done_ || draining_) return false; ++in_flight_;`), called from `hooks.begin_incoming_push` after looking up `active_pool_reshard_recvs_` under manager `mu_` | — |
| `pushEnd ok` | 3–6 in `hooks.end_incoming_push`: on `ok = false` under `.finishOnError`, `reshard_session->Finish(status)`; then `reshard_session->EndRecvOp()`; then `reshard_session->Done()` and, if settled, `reshard_session->TakePendingUnregister(&generation)` (`2–4` scopes), followed by `1–2` read-only `GetStatus()` scopes for the hook's return value | `EndRecvOp` and `TakePendingUnregister` tail, below |
| `onPoolReceived p` | 2: `ReshardReceiveSession::OnPoolReceived` checks `if (done_)` and calls `RecordPoolReceivedLocked(pool_idx)`, then `ExecuteEligiblePoolH2ds` checks `if (done_ || draining_)` and runs `to_launch = CollectEligiblePoolH2dsLocked(); in_flight_ += static_cast<int32_t>(to_launch.size());` | folded into one event by A4 |
| `h2dIssue p ok` | 1–7 in `ExecuteEligiblePoolH2ds`: re-check lock (`if (done_ || draining_) { EndRecvOpLocked(); continue; }`, 1 scope when draining), then unlocked `base_->H2dPoolBlocks`; on `ok = true`, 1 scope (`h2d_futures_.push_back(future);`) before `future.OnReady` is attached; on `ok = false`, `FinishPoolH2d` (1 or 3 scopes), `EndRecvOp()` (1 scope), and `MaybeUnregisterSettledReshardRecv` (0–2 scopes) | pool `p`'s op in `in_flight_` is already counted before `H2dPoolBlocks`; on `ok = false`, the `reshard_finalizing_` latch and `EndRecvOp` tail below apply |
| `h2dReady p` | 0 (device) | — |
| `h2dDone p ok` | 2–6 in `ExecuteEligiblePoolH2ds`'s `future.OnReady` callback: `FinishPoolH2d` locks `mu_` once for `if (done_) return;` and `finished = RecordPoolH2dResultLocked(pool_idx, status);`; on `!finished && status.ok()` it runs `ExecuteEligiblePoolH2ds`'s opening lock (1 scope, folded by A4), on `finished` it latches `terminal_status` or `should_record_duration = true; network_completed_ = true;` and calls `Finish(terminal_status)` (2 scopes), and on `!finished && !status.ok()` (when `reshard_finalizing_` was already set) neither branch runs; then `self->EndRecvOp()` (1 scope) and `MaybeUnregisterSettledReshardRecv` (0–2 scopes) | see the `reshard_finalizing_` latch, wave launch, and `EndRecvOp` tail notes below |
| `pollReady` | 3 in `CompleteReadWithDetails` (under manager `mu_`): `session->IsDraining()`, `session->IsReadyToComplete()`, `session->Finish()` | disabled on every reachable state (`reachable_pollReady_disabled`) because `FinishPoolH2d` calls `Finish(terminal_status)` right after setting `network_completed_ = true` while the final pool's op still holds `in_flight_ >= 1` |
| `cancel` | 1–4: `ReshardReceiveSession::Finish(error)` (`FinishLocked`), called directly in `~KVCacheManagerWithTransfer` (1 scope), after `!session->IsDraining()`, `!session->IsReadyToComplete()`, and `session->deadline() <= now` in `CompleteReadWithDetails` (3 scopes), or after `!reshard_it->second->Done()` under manager `mu_` in `UnregisterActivePlan` followed by `Done()` and `TakePendingUnregister` if idle (3–4 scopes) | `cancel` has no `!draining` guard, admitting every interleaving; late errors, below |
| `publish` | 3–4 in `CompleteReadWithDetails` (under manager `mu_`): `session->Done()`, `session->GetStatus()` (twice on failure), and `session->TakePendingUnregister(&generation)` before `active_pool_reshard_recvs_.erase(it++)` | on a settled session only `status_` can still change; late errors, below |

**`reshard_finalizing_` latch in `FinishPoolH2d`.** When `RecordPoolH2dResultLocked`
returns `finished == true`, it has already set `reshard_finalizing_ = true`
under the first `mu_` scope. Any concurrent `FinishPoolH2d` or
`CollectEligiblePoolH2dsLocked` before `Finish(terminal_status)` sees
`reshard_finalizing_ == true` and is a no-op, while the caller's pool op keeps
`in_flight_ >= 1` until `self->EndRecvOp()`.

**Wave launch inside `FinishPoolH2d` before `self->EndRecvOp()`.** On an
intermediate OK completion (`!finished && status.ok()`), `FinishPoolH2d` calls
`ExecuteEligiblePoolH2ds(manager)` before returning to the outer callback's
`self->EndRecvOp()`. In `h2dDone p true`, `launchEligible` folds the opening
lock of `ExecuteEligiblePoolH2ds` (which adds `to_launch.size()` to `in_flight_`)
with pool `p`'s `self->EndRecvOp()`, leaving the per-pool `to_launch` loop to
subsequent `h2dIssue` steps: while that loop runs, pool `p`'s unreleased op only
keeps `in_flight_` higher by 1 until `self->EndRecvOp()`, commuting with
`h2dIssue` for the same reason as any `EndRecvOp` tail.

**`EndRecvOp` and `TakePendingUnregister` tail.** Before `EndRecvOp()`, the
caller's op keeps `in_flight_ >= 1` and `done_ = false`, so `EndRecvOp()`
commutes with any interleaved step. Between `EndRecvOp()` setting `done_ = true`
(releasing staging in the same scope) and `MaybeUnregisterSettledReshardRecv`
claiming `TakePendingUnregister` (`unregister_on_settle_ = false`, modelled as
`planUnregistered = true`), the intermediate state `(done = true,
planUnregistered = false, published = none)` already satisfies `Inv`.

**Late error after settle.** `ReshardReceiveSession::FinishLocked` updates
`if (!status.ok() && status_.ok()) status_ = status;` before
`if (draining_) return;`, so a late `Finish(error)` after an OK settle still
flips `status_` to the error (`cancel` has no `!draining` guard).

## Assumptions

* **A1 (at-most-once `OnPoolReceived` per declared pool).**
  `BlockTransport::CompleteIncomingPush` fires `block_delegate_->OnPoolReceived(l, uuid)`
  only when `plan_pushes == expected_chunks`, and
  `ReshardReceiveSession::RecordPoolReceivedLocked` rejects undeclared or
  duplicate pool indices. Encoded as the `onPoolReceived p` guard
  `pool.stage == .waiting`. (Discharged in `Reshard/Pipeline.lean` by explicit
  stream counting.)
* **A2 (registration before visibility).**
  `base->RegisterActivePlanDirect(plan.uuid(), plan, /*is_sender=*/false)` and
  `AcquireStagingLeases` in `ReshardReceiveSession::Create` succeed before the
  session is inserted into `active_pool_reshard_recvs_` in
  `PoolReshardRegisterRecv`. If `AcquireStagingLeases` fails, `Create` calls
  `session->ReleaseStaging()` and rolls back
  `base->UnregisterActivePlanDirect(plan.uuid())` before any hook can find the
  session.
* **A3 (no double end).** Each push and each H2D op ends at most once: `pushEnd`
  requires `pushes > 0`, `h2dIssue p` requires `stage == .queued`, and
  `h2dDone p` requires `stage == .ready`.
* **A4 (fold of `RecordPoolReceivedLocked` / `RecordPoolH2dResultLocked` with
  `ExecuteEligiblePoolH2ds` first lock).** Between unlocking `mu_` after
  recording a pool (`RecordPoolReceivedLocked(pool_idx)` in `OnPoolReceived` or
  `RecordPoolH2dResultLocked(pool_idx, status)` in `FinishPoolH2d`) and locking
  `mu_` in `ExecuteEligiblePoolH2ds`, an external `Finish` produces the same
  state as `cancel` arriving before the event (`draining_ = true` causes
  `if (done_ || draining_) return;` to skip `CollectEligiblePoolH2dsLocked`,
  leaving eligible pools in `.received`), whereas a `Finish` arriving after
  `in_flight_ += static_cast<int32_t>(to_launch.size())` is `cancel` after the
  event (where the per-pool re-check `if (done_ || draining_) { EndRecvOpLocked(); continue; }`
  aborts each queued pool and releases its op).

## Properties

All proved on every reachable state (`reachable_safe`), under both
`PushEndPolicy.finishOnError` (shipping) and `PushEndPolicy.endOpOnly`:

* **Settle safety (`SettleSafe`).** `done → inFlight = 0`: a settled session
  owns no in-flight push or H2D copy, as required by the comments in
  `ExecuteEligiblePoolH2ds` and `FinishPoolH2d`.
* **No retired callback (`NoRetiredCallback`).** `done → poolOps pools = 0`:
  no pool is `.queued`, `.issued`, or `.ready` once settled, so the early return
  `if (done_) return;` in `FinishPoolH2d` is unreachable.
* **Staging integrity (`StagingIntegrity`).** `hasStaging = !done`: the bounded
  `PoolStagingLease` is held exactly until the session settles.
* **Plan integrity (`PlanIntegrity`).** `(planUnregistered → done) ∧
  (published ≠ none → planUnregistered)`: the active transfer plan is never
  unregistered while any push or H2D copy can still resolve through it
  (`FinishPoolH2d`, `UnregisterActivePlan`, `MaybeUnregisterSettledReshardRecv`),
  and is always unregistered once published.
* **Prompt settle (`SettlesPromptly`).** `draining → inFlight = 0 → done`.
* **Rank-wave ordering (`RankWaveOrder`).** For every pair of pools `p, q ∈ pools`,
  if `p` has launched (`p.stage.launched = true`) and `q.rank < p.rank`, then
  `q.stage = .completed`: no higher-rank pool is ever queued, issued, or
  completed before every strictly lower-rank pool has completed its H2D upload
  (`pool_order_ranks_` comment in `ReshardReceiveSession`,
  `CollectEligiblePoolH2dsLocked`).
* **Readiness soundness (`ReadinessSound`).** `isReadyToComplete → allCompleted pools`:
  when `IsReadyToComplete()` holds, every expected pool has completed its H2D
  upload. Moreover, at callback granularity `pollReady` is never enabled on any
  reachable state (`reachable_pollReady_disabled`), because `FinishPoolH2d` sets
  `network_completed_ = true` immediately before calling
  `Finish(terminal_status)` (and after latching `should_record_duration = true`).
* **Publication (`Publication`).** `published = some true → allCompleted pools ∧
  durationRecorded`: when `poll_stats()` reports `done_recving`, every expected
  pool has completed its H2D upload and the transfer duration metric has been
  recorded.
* **No op leak & termination (`NoOpLeak`, `reachable_can_settle`).** While
  `0 < inFlight`, some event in `drainEvents` is enabled, and every reachable
  state can drain to `done = true ∧ hasStaging = false` in a finite number of
  steps.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `reshard_recv.h` | `ReshardReceiveSession::Done` | `:73-76` |
| `reshard_recv.h` | `ReshardReceiveSession::GetStatus` | `:80-83` |
| `reshard_recv.h` | `ReshardReceiveSession::IsDraining` | `:87-90` |
| `reshard_recv.h` | `ReshardReceiveSession::deadline` | `:94` |
| `reshard_recv.h` | `ReshardReceiveSession::TryBeginRecvOp` | `:100-105` |
| `reshard_recv.h` | `ReshardReceiveSession` (`bool staging_released_` .. `absl::flat_hash_set<size_t> h2d_launched_pools_`) | `:172-190` |
| `reshard_recv.cc` | `RecordTransferDuration` | `:60-63` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidatePlan` | `:67-144` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` | `:351-390` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReshardReceiveSession` | `:392-418` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:420-459` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:461-465` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:467-470` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishLocked` | `:472-482` |
| `reshard_recv.cc` | `ReshardReceiveSession::Finish` | `:484-487` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndRecvOpLocked` | `:495-505` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndRecvOp` | `:507-510` |
| `reshard_recv.cc` | `ReshardReceiveSession::TakePendingUnregister` | `:512-520` |
| `reshard_recv.cc` | `ReshardReceiveSession::AllH2dDoneLocked` | `:522-527` |
| `reshard_recv.cc` | `ReshardReceiveSession::IsReadyToComplete` | `:529-532` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolReceivedLocked` | `:541-552` |
| `reshard_recv.cc` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` | `:554-583` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolH2dResultLocked` | `:585-598` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnPoolReceived` | `:600-612` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` | `:614-666` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`to_launch = CollectEligiblePoolH2dsLocked();` .. `in_flight_ += static_cast<int32_t>(to_launch.size());`) | `:622-623` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`EndRecvOpLocked();`) | `:635` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`h2d_futures_.push_back(future);`) | `:652` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`future.OnReady`) | `:654-664` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:668-706` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` (`finished = RecordPoolH2dResultLocked(pool_idx, status);`) | `:677` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` (`should_record_duration = true;` .. `Finish(terminal_status);`) | `:696-700` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.on_pool_received =`) | `:180-202` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.begin_incoming_push =`) | `:224-249` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.end_incoming_push =`) | `:250-296` |
| `mgr.cc` | `KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer` (`for (const auto& [uuid, session] : active_pool_reshard_recvs_)`) | `:415-419` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:703-739` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:741-778` |
| `mgr.cc` | `KVCacheManagerWithTransfer::MaybeUnregisterSettledReshardRecv` | `:806-812` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_recvs_.begin();`) | `:1067-1097` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:496-639` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` | `:691-887` |
| `utils.h` | `ValidateCommonPoolReshardPlan` | `:353-469` |
| `utils.h` | `ValidatePoolBlockBounds` | `:473-489` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.AcceptsCanonicalPlanOnExplicitPools` | `:137-145` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.AcceptsImplicitPools` | `:147-154` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.HasNoTagPolicy` | `:156-164` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsMissingIdentityAndPoolFields` | `:166-198` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsOutOfRangeDuplicateAndDtypeMismatch` | `:200-225` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsBlockIdsOutsideDeclaredPool` | `:257-276` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.GroupScopesDestinationBlocks` | `:405-439` |
| `pool_reshard_test.cc` | `PoolReshardValidationTest.DeviceOnlyRejectionAtPublicEntryPoints` | `:271-287` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FinishPoolReshardRecvRecordsDurationMetric` | `:327-347` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FinishPoolReshardRecvDoesNotRecordMetricOnFailure` | `:349-367` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline` | `:766-802` |
<!-- source-map:end -->
-/

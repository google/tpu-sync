import TpuSyncVerify.Transfer.Reshard.Receive.Model
import TpuSyncVerify.Transfer.Reshard.Receive.Lemmas
import TpuSyncVerify.Transfer.Reshard.Receive.Invariant
import TpuSyncVerify.Transfer.Reshard.Receive.Progress
import TpuSyncVerify.Transfer.Reshard.Receive.Traces

/-!
# Reshard receive session

One `ReshardReceiveSession` on the destination (consumer) side and the slice of
`KVCacheManagerWithTransfer` and the transport that drives it
(`PoolReshardRegisterRecv`, `GetSession`, `BeginIncomingPush` /
`EndIncomingPush`, `BlockTransportServerPushSession::CompleteIncomingPush`, and
the `CompleteReadWithDetails` poll). Sender sessions, multi-sender stream
counting, and aliased-page memory contents are modelled in `Reshard/Send.lean`
and `Reshard/Pipeline.lean`.

`reshard_recv.cc` is `tpu_sync/kv_cache/reshard_receive_session.cc` (which
holds the class `ReshardReceiveSession` behind the factory
`CreateReshardReceiveSession` declared in
`tpu_sync/kv_cache/reshard_receive_session.h`); `staging.h` is
`tpu_sync/kv_cache/staging_manager.h` (`StagingLayerManager`) and
`tpu_sync/transport/transfer_session.h` is the `TransferSession` interface;
`mgr.cc` is `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`;
`base.h`/`base.cc` are `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`;
`raw_core.h` is `tpu_sync/core/raw_transfer_core.h`; `bt.cc` and `bts.cc` are
`tpu_sync/transport/block_transport.cc` and
`tpu_sync/transport/block_transport_session.cc`; `utils.h` is
`tpu_sync/core/utils.h`; `reshard_recv_test.cc` and `reshard_send_test.cc` are
`tpu_sync/kv_cache/reshard_receive_session_test.cc` and
`tpu_sync/kv_cache/reshard_send_session_test.cc`; and `pool_reshard_test.cc`
is `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.
Exact line ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

## How a reshard receive runs

`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` first looks the plan's
uuid up in the manager's single `sessions_` map — an incumbent that is `Done()`
is retired (`RetireSettledSessionLocked`), one that is not is refused — and
claims the uuid in `pending_pool_reshard_recvs_` (a second registration racing
the factory is refused with `AlreadyExistsError`, `Reshard/PoolStaging.lean`).
It then creates a `ReshardReceiveSession` (`CreateReshardReceiveSession`,
`ReshardReceiveSession::ReshardReceiveSession`), which builds the receiver's
`TransferPlan` and acquires bounded per-UUID `PoolStagingLease`s across the
plan's destination storages (`ReshardReceiveSession::AcquireStagingLeases`),
inserts the session into `sessions_` as a `SessionEntry` carrying the plan's
`req_id` and `SessionRole::kPoolReshardReceive`
(`sessions_[plan.uuid()] = SessionEntry{recv_session, plan.req_id(),`,
dropping the pending claim in the same lock) and calls `Start()`, which for a
receive does nothing. Each transferred pool in `expected_pool_indices_` has an
order rank in `pool_order_ranks_`
(`pool_order_ranks_[static_cast<size_t>(pool_idx)] = group.order_rank()` in
`ReshardReceiveSession::ReshardReceiveSession`).

As incoming push streams arrive at the destination's per-uuid
`BlockTransportServerPushSession` (`StartStream`, reached from
`BlockTransport::HandleIncomingPush`, then `CompleteIncomingPush`), each stream
takes a receive op through `BeginIncomingPush` (`delegate->GetSession(uuid)`
then `ReshardReceiveSession::TryBeginOp`, which also hands the stream its
`session`) and releases it through `EndIncomingPush`
(`ReshardReceiveSession::EndOp`, preceded by `session->Finish(status)` when the
stream failed). When all `expected_pushes` streams for a pool have landed, the
push session calls, on that same session and still inside the stream's op,
`session->OnDataReceived(l, /*minor_dimension_indices=*/{}, /*shard_ids=*/{})`
→ `ReshardReceiveSession::OnDataReceived` → `ReshardReceiveSession::OnPoolReceived`
(a report with block indices is accepted without effect; a completion on a
`Done()` session fails with `kNotFound`), which records the pool in
`started_pool_indices_` (`ReshardReceiveSession::RecordPoolReceivedLocked`) and
calls `ReshardReceiveSession::ExecuteEligiblePoolH2ds`.

`ExecuteEligiblePoolH2ds` selects under `mu_` every started, unlaunched pool
whose strictly lower-rank prerequisite pools are all in
`completed_pool_indices_` (`ReshardReceiveSession::CollectEligiblePoolH2dsLocked`),
marks them in `h2d_launched_pools_`, and adds `to_launch.size()` to `in_flight_`
(`in_flight_ += static_cast<int32_t>(to_launch.size())`). Outside `mu_`, it
loops over `to_launch`: for each pool it re-checks `if (done_ || draining_)`
under `mu_` (calling `EndRecvOpLocked()` and skipping the upload if the session
finished in the window), resolves the pool's host buffer
(`LookupStagingSlot`, §"Bounded host layer staging"), dispatches
`base_->H2dPoolBlocks` into it, and attaches `FinishPoolH2d` + `EndOp` to
`future.OnReady`; an empty destination list, a missing buffer or a synchronous
`H2dPoolBlocks` error runs `FinishPoolH2d(pool_idx, status)` and `EndOp()`
inline.

When a pool's H2D finishes, `ReshardReceiveSession::FinishPoolH2d` calls
`ReshardReceiveSession::RecordPoolH2dResultLocked`. On an intermediate OK
completion (`if (!finished && status.ok())`), it calls
`ExecuteEligiblePoolH2ds()` to launch the next rank wave before the outer
`future.OnReady` callback's `self->EndOp()` releases the completed pool's op.
On the first error or when `completed_pool_indices_ == expected_pool_indices_`,
`reshard_finalizing_` is set (`finished = true`), `should_record_duration = true`
is latched on success, and `Finish(terminal_status)` drains the session.

**Plan lifetime.** The plan is the session's own `transfer_plan_`:
`ReshardReceiveSession::plan()` returns it until `Done()` and `absl::nullopt`
afterwards, and the manager resolves it only through `GetSession(uuid)`
(`KVCacheManagerBase::GetTransferPlan`, `GetBlockChunks`). So the plan is
visible exactly while the session is in `sessions_` and not settled, and it
goes with the staging in the same `mu_` scope (`FinishLocked` /
`EndRecvOpLocked` setting `done_`); there is no separate unregistration step.
`KVCacheManagerWithTransfer::UnregisterActivePlan` on a reshard receive calls
`Finish(Cancelled)` if it is not `Done()` (`cancel`) and leaves it in
`sessions_` for the poll; `RetireSettledSessionLocked` erases a `Done()`
session when it reports it (`publish`). `~KVCacheManagerWithTransfer` calls
`Finish(Cancelled)` on every session under `mu_`, then `AwaitForDone()` on each.

## State

The settle protocol itself is `Transfer.Lifecycle` (`in_flight_`, `draining_`,
`done_`, `status_.ok()`, and `!staging_released_`). On top of it, each expected
pool is a `Pool := ⟨rank, stage⟩` in `pools : List Pool`:

| Pool stage   | C++ membership (`ReshardReceiveSession`)             | Holds op in `in_flight_` |
|--------------|------------------------------------------------------|--------------------------|
| `.waiting`   | in `expected_pool_indices_`, not yet in `started_pool_indices_` | no |
| `.received`  | in `started_pool_indices_`, not yet in `h2d_launched_pools_` | no |
| `.queued`    | in `h2d_launched_pools_` after `in_flight_ += static_cast<int32_t>(to_launch.size())` (`CollectEligiblePoolH2dsLocked`, `ExecuteEligiblePoolH2ds`), before the per-pool re-check lock (`if (done_ || draining_)`) | yes |
| `.issued`    | `H2dPoolBlocks` succeeded; the copy is running on the device (`future.OnReady` attached) | yes |
| `.ready`     | the copy finished; the `OnReady` continuation has not run yet | yes |
| `.completed` | in `completed_pool_indices_` (`RecordPoolH2dResultLocked`) and `self->EndOp()` ran (`ExecuteEligiblePoolH2ds`'s `future.OnReady`) | no |
| `.aborted`   | op ended without completing: skipped at re-check (`EndRecvOpLocked()`), or `LookupStagingSlot`/`H2dPoolBlocks`/`OnReady` failed or saw `reshard_finalizing_` | no |

Session-level fields:

| Field              | C++                                                                 | Role |
|--------------------|---------------------------------------------------------------------|------|
| `pools`            | `expected_pool_indices_`, `pool_order_ranks_`, `started_pool_indices_`, `h2d_launched_pools_`, `completed_pool_indices_` and the pending `OnReady` continuations | per-pool state |
| `life`             | `in_flight_`, `draining_`, `done_`, `status_`, `!staging_released_` (`ReshardReceiveSession`) | `Transfer.Lifecycle` |
| `finalizing`       | `reshard_finalizing_` (`ReshardReceiveSession`)                     | stops further H2D launches and results after first error or full completion |
| `durationRecorded` | ghost (`should_record_duration` / `RecordTransferDuration` in `FinishPoolH2d`) | end-of-transfer latency histogram recorded |
| `published`        | the session's terminal as `RetireSettledSessionLocked` hands it to `RecordPoolReshardTerminalLocked` (with the `req_id` of its `SessionEntry`): membership in `done_recving_` / `failed_recving_` for a single-plan request (`sub_plan_count <= 1`), or the sub-plan's entry in the request's `sub_plan_chains_` record | what `poll_stats()` shows the engine |
| `pushes`           | ghost                                                               | incoming pushes between `TryBeginOp` (`BeginIncomingPush`) and `EndOp` (`EndIncomingPush`) |

## Events

| Event               | C++ |
|---------------------|-----|
| `pushBegin`         | `ReshardReceiveSession::TryBeginOp` from `BeginIncomingPush` (`delegate->GetSession(uuid)`, then `session->TryBeginOp()`), called in `BlockTransportServerPushSession::StartStream` (`BeginIncomingPush(delegate_, header.uuid)`), `BlockTransportServerPushSession::ResolveRecvBuffer`, and `BlockTransportServerPushSession::OnPeregrinePostComplete` |
| `pushEnd ok`        | `EndIncomingPush(session, status)`, called in `BlockTransportServerPushSession::StartStream` (directly on success, or from the `end_incoming_push` cleanup with `InternalError("Incoming push failed")`). Parameterized by `PushEndPolicy`: `.finishOnError` (shipping) calls `session->Finish(status)` before `session->EndOp()` on `ok = false` (`if (!status.ok())`, the same for every session type), whereas `.endOpOnly` (mutant) runs `EndOp()` unconditionally |
| `onPoolReceived p`  | `ReshardReceiveSession::OnPoolReceived` (via `ReshardReceiveSession::OnDataReceived`, called by `BlockTransportServerPushSession::CompleteIncomingPush` on the stream's own `session`) + `ExecuteEligiblePoolH2ds` first lock (`to_launch = CollectEligiblePoolH2dsLocked()`): records `p` in `started_pool_indices_` and batch-queues every eligible started pool |
| `h2dIssue p ok`     | `ExecuteEligiblePoolH2ds` loop body (`for (auto& [pool_idx, dst_chip_block_ids] : to_launch)`): re-checks `if (done_ || draining_)` under `mu_`, otherwise resolves the host buffer and dispatches `base_->H2dPoolBlocks` |
| `h2dReady p`        | device finishes pool `p`'s H2D copy; its `OnReady` continuation is pending |
| `h2dDone p ok`      | `ExecuteEligiblePoolH2ds`'s `future.OnReady` callback calling `FinishPoolH2d` and `EndOp` (and returning the pool's host buffer, `staging_layer_manager_.Release(pool_idx)`) |
| `cancel`            | external `Finish(error)`: deadline (`CompleteReadWithDetails`), shutdown (`~KVCacheManagerWithTransfer`), `UnregisterActivePlan`, or a staging wait that timed out (`FinishLocked(slot.status())` in `AcquireStagingSlot`) |
| `publish`           | `CompleteReadWithDetails` retires a settled session (`RetireSettledSessionLocked`: `reshard->Done()`) through `RecordPoolReshardTerminalLocked` — into `done_recving_` or `failed_recving_` for a single-plan request, or into the request's `sub_plan_chains_` entry for a sub-plan of a chain (the request reaches `done_recving_` when its last sub-plan settles and `failed_recving_` at its first failed sub-plan) — and erases it from `sessions_`; after the manager lock the poll drops the uuid's push progress and any remaining pool staging leases (`ForgetPushProgress`, `ReleasePoolStagingLeases`) |

## Lock scopes

Every mutable field of `ReshardReceiveSession` is `ABSL_GUARDED_BY(mu_)`
except `staging_layer_manager_` (its own mutex) and the atomic `cancelled_`;
`base_`, `deadline_`, `start_time_`, and `transfer_plan_` (whose request
supplies the uuid and req_id) are `const`.

| Event | `mu_` scopes | Between them |
|---|---|---|
| `pushBegin` | 1: `ReshardReceiveSession::TryBeginOp` (`if (done_ || draining_) return false; ++in_flight_;`), after `GetSession` looked the uuid up in `sessions_` under manager `mu_` | — |
| `pushEnd ok` | 2–4 in `EndIncomingPush`: on `ok = false` under `.finishOnError`, `session->Finish(status)`; then `session->EndOp()`; followed by `0–1` read-only `GetStatus()` scopes for the return value | `EndOp` tail, below |
| `onPoolReceived p` | 2: `ReshardReceiveSession::OnPoolReceived` checks `if (done_)` and calls `RecordPoolReceivedLocked(pool_idx)`, then `ExecuteEligiblePoolH2ds` checks `if (done_ || draining_)` and runs `to_launch = CollectEligiblePoolH2dsLocked(); in_flight_ += static_cast<int32_t>(to_launch.size());`; from `CompleteIncomingPush` both run on the push thread while its stream op (`pushBegin`) is still counted, though the event needs no open push | folded into one event by A4 |
| `h2dIssue p ok` | 1–6 in `ExecuteEligiblePoolH2ds`: re-check lock (`if (done_ || draining_) { ... EndRecvOpLocked(); continue; }`, 1 scope when draining), then `LookupStagingSlot` (1 read-only scope) and the unlocked `base_->H2dPoolBlocks`; on `ok = true`, nothing more before `future.OnReady` is attached; on `ok = false`, `FinishPoolH2d` (1 or 3 scopes) and `EndOp()` (1 scope) | pool `p`'s op in `in_flight_` is already counted before `H2dPoolBlocks`; on `ok = false`, the `reshard_finalizing_` latch and `EndOp` tail below apply |
| `h2dReady p` | 0 (device) | — |
| `h2dDone p ok` | 2–5 in `ExecuteEligiblePoolH2ds`'s `future.OnReady` callback: `FinishPoolH2d` locks `mu_` once for `if (done_) return;` and `finished = RecordPoolH2dResultLocked(pool_idx, status);`; on `!finished && status.ok()` it runs `ExecuteEligiblePoolH2ds`'s opening lock (1 scope, folded by A4), on `finished` it latches `terminal_status` or `should_record_duration = true;` and calls `Finish(terminal_status)` (2 scopes), and on `!finished && !status.ok()` (when `reshard_finalizing_` was already set) neither branch runs; then `self->EndOp()` (1 scope) | see the `reshard_finalizing_` latch, wave launch, and `EndOp` tail notes below |
| `cancel` | 1–3: `ReshardReceiveSession::Finish(error)` (`FinishLocked`), called directly in `~KVCacheManagerWithTransfer` (1 scope, under manager `mu_`), after `!session->IsDraining()` and `session->deadline() <= now` in `CompleteReadWithDetails` (2 scopes), after an unlocked `session->Done()` in `UnregisterActivePlan` (2–3 scopes), or as `FinishLocked(slot.status())` inside `AcquireStagingSlot`'s closing scope | `cancel` has no `!draining` guard, admitting every interleaving; late errors, below |
| `publish` | 2 in `RetireSettledSessionLocked` (under manager `mu_`, on the entry's `SessionRole::kPoolReshardReceive`): `reshard->Done()` and `reshard->GetStatus()` (passed to `RecordPoolReshardTerminalLocked`) before `sessions_.erase(it)` | on a settled session only `status_` can still change; late errors, below |

**`reshard_finalizing_` latch in `FinishPoolH2d`.** When `RecordPoolH2dResultLocked`
returns `finished == true`, it has already set `reshard_finalizing_ = true`
under the first `mu_` scope. Any concurrent `FinishPoolH2d` or
`CollectEligiblePoolH2dsLocked` before `Finish(terminal_status)` sees
`reshard_finalizing_ == true` and is a no-op, while the caller's pool op keeps
`in_flight_ >= 1` until `self->EndOp()`.

**Wave launch inside `FinishPoolH2d` before `self->EndOp()`.** On an
intermediate OK completion (`!finished && status.ok()`), `FinishPoolH2d` calls
`ExecuteEligiblePoolH2ds()` before returning to the outer callback's
`self->EndOp()`. In `h2dDone p true`, `launchEligible` folds the opening
lock of `ExecuteEligiblePoolH2ds` (which adds `to_launch.size()` to `in_flight_`)
with pool `p`'s `self->EndOp()`, leaving the per-pool `to_launch` loop to
subsequent `h2dIssue` steps: while that loop runs, pool `p`'s unreleased op only
keeps `in_flight_` higher by 1 until `self->EndOp()`, commuting with
`h2dIssue` for the same reason as any `EndOp` tail.

**`EndOp` tail.** Before `EndOp()`, the caller's op keeps `in_flight_ >= 1` and
`done_ = false`, so `EndOp()` commutes with any interleaved step; when it is
the last op of a draining session, `EndRecvOpLocked` releases the staging and
sets `done_` (after which `plan()` is empty) in the same scope.

**Late error after settle.** `ReshardReceiveSession::FinishLocked` updates
`if (!status.ok() && status_.ok()) status_ = status;` before
`if (draining_) return;`, so a late `Finish(error)` after an OK settle still
flips `status_` to the error (`cancel` has no `!draining` guard).

### Bounded host layer staging

The model omits host layer buffer leases: each declared pool transitions
independently. On a full host mirror (`!HostLayersBounded()`, i.e.
`num_host_layers() == num_layers()`, the default when `num_staging_layers` is
unset), `ReshardReceiveSession` constructs its `StagingLayerManager`
(`staging.h`) with `0` slots and both `AcquireStagingSlot` and
`LookupStagingSlot` return `StaticHostLayer(pool_idx)`
(`base_->pool(pool_idx)->storage_index`, for explicit and implicit pools alike
since `KVCacheManagerBase::pool` materializes the implicit table; `NotFound`
only for a `pool_idx` outside the pool table, which `ValidatePlan` rejects),
staging every pool in place in its own storage's host buffer with no slot wait
(`ReshardReceiveSessionTest.FullHostMirrorAddressesPoolsInPlace`). When host
layers are bounded (`HostLayersBounded()`, `num_host_layers() < num_layers()`
under `num_staging_layers`), `ReshardReceiveSession` constructs
`staging_layer_manager_` with `num_host_layers()` slots and leases one buffer
per pool index: each incoming push chunk's `KVCacheManagerBase::GetBlockChunks`
(inside the push's op, from `StartStream`'s payload loop) calls
`session->ResolveStagingSlot(layer_idx)` →
`ReshardReceiveSession::AcquireStagingSlot`, which takes an op of its own
(`if (done_ || draining_)` refusal, `++in_flight_`, the pattern of `TryBeginOp`),
waits in `StagingLayerManager::Acquire` until a buffer is free, the pool
already holds one, `deadline_` passes or `cancelled_` is set (`FinishLocked`
sets `cancelled_` and calls `staging_layer_manager_.Shutdown()` to wake
waiters), then under `mu_` fails the receive on a timeout
(`FinishLocked(slot.status())`), returns a buffer won by a draining session,
and ends its op (`EndRecvOpLocked()`). `ExecuteEligiblePoolH2ds` reads the
leased buffer (`LookupStagingSlot`, failing with `FailedPrecondition` if the
pool holds none and `Cancelled` on a draining session) and dispatches with it
as `host_layer_idx`; the buffer is returned on each early return, by the H2D
callback (`staging_layer_manager_.Release(pool_idx)` around `FinishPoolH2d`),
and wholesale at settle (`ReleaseStagingLocked` → `ReleaseAll()`).

None of the safety invariants (`Safe`) changes under bounded host layer staging
when pools outnumber `num_host_layers()`: the extra op is nested inside the
push's op on the push thread (taken after `pushBegin`, ended before the
stream's `CompleteIncomingPush` and `EndIncomingPush`), so `in_flight_` is
transiently one higher while `pushes > 0` already holds and every settle
decision is the one the model makes; a timeout fails the receive, which is
`cancel` before `pushEnd` (the unguarded `cancel` admits it), and a cancelled
wait or a failed `Lookup` fails the stream or the upload on a session that is
already draining (`pushEnd false`, `h2dIssue p false` with
`reshard_finalizing_` or the re-check first).

Progress without hitting `deadline_` is narrower when `HostLayersBounded()` and
transferred pools exceed `num_host_layers()`: on a single `order_rank`, later
pools wait behind earlier pools' uploads, whereas across multiple
`order_rank`s, gated higher-rank pools whose first chunks arrive early can hold
every buffer while a lower-rank pool's push thread blocks in
`StagingLayerManager::Acquire` until `deadline_` fails the receive
(`FinishLocked(slot.status())`), which the slot-free model does not express
(`ReshardReceiveSessionTest.BoundedStagingSlotAcquisitionAndRelease` drives the
wait through `ResolveStagingSlot` on a second thread, releases it by completing
the first pool through `OnDataReceived` on a `FakeH2dBase` whose uploads finish
at once, then the cancel and, on a second session, the timeout).

## Assumptions

* **A1 (at-most-once `OnPoolReceived` per declared pool).**
  `BlockTransportServerPushSession::CompleteIncomingPush` calls
  `session->OnDataReceived(l, /*minor_dimension_indices=*/{}, /*shard_ids=*/{})`
  only when `plan_pushes == expected_chunks` (under the push session's own
  `mu_`, with `layer_progress_` per uuid) and only once per pool (the
  `on_layer_received_called` latch; a push beyond the plan's count is rejected
  by `if (plan_pushes > expected_chunks)`), and
  `ReshardReceiveSession::RecordPoolReceivedLocked` rejects undeclared or
  duplicate pool indices. Encoded as the `onPoolReceived p` guard
  `pool.stage == .waiting`. `Reshard/Pipeline.lean` encodes the count as a
  second guard on the same event (`landedStreams[p] = peerCounts[p]`); neither
  model derives the assumption from the transport.
* **A2 (registration before visibility).** `PoolReshardRegisterRecv` claims the
  uuid in `pending_pool_reshard_recvs_` before `CreateReshardReceiveSession`
  runs `AcquireStagingLeases`, and inserts the session into `sessions_` (where
  `GetSession` can find it) only after the factory succeeded. If
  `AcquireStagingLeases` fails, the factory calls `session->ReleaseStaging()`
  and the `absl::Cleanup` drops the pending claim before any push can find the
  uuid.
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
  event (where the per-pool re-check `if (done_ || draining_) { ... EndRecvOpLocked(); continue; }`
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
* **Prompt settle (`SettlesPromptly`).** `draining → inFlight = 0 → done`.
* **Rank-wave ordering (`RankWaveOrder`).** For every pair of pools `p, q ∈ pools`,
  if `p` has launched (`p.stage.launched = true`) and `q.rank < p.rank`, then
  `q.stage = .completed`: no higher-rank pool is ever queued, issued, or
  completed before every strictly lower-rank pool has completed its H2D upload
  (`pool_order_ranks_` comment in `ReshardReceiveSession`,
  `CollectEligiblePoolH2dsLocked`).
* **Publication (`Publication`).** `published = some true → allCompleted pools ∧
  durationRecorded`: when `poll_stats()` reports `done_recving`, every expected
  pool has completed its H2D upload and the transfer duration metric has been
  recorded.
* **No op leak & termination (`NoOpLeak`, `reachable_can_settle`).** While
  `0 < inFlight`, some event in `drainEvents` is enabled, and every reachable
  state can drain to `done = true ∧ hasStaging = false` in a finite number of
  steps.

Plan visibility follows from these without state of its own: `plan()` has a
value exactly while `!done`, so `SettleSafe` says no push or upload — nothing
that resolves chunks or staging slots through the plan — is in flight once the
plan is gone, and `publish` (which erases the session from `sessions_`)
requires `done`.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `tpu_sync/transport/transfer_session.h` | `TransferSession` | `:36-115` |
| `staging.h` | `StagingLayerManager` | `:39-74` |
| `reshard_recv.cc` | `ReshardReceiveSession::Done` | `:91-94` |
| `reshard_recv.cc` | `ReshardReceiveSession::Start` | `:97` |
| `reshard_recv.cc` | `ReshardReceiveSession::TryBeginOp` | `:101-106` |
| `reshard_recv.cc` | `ReshardReceiveSession::GetStatus` | `:109-112` |
| `reshard_recv.cc` | `ReshardReceiveSession::IsDraining` | `:116-119` |
| `reshard_recv.cc` | `ReshardReceiveSession::deadline` | `:121-123` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnDataReceived` | `:128-135` |
| `reshard_recv.cc` | `ReshardReceiveSession::ResolveStagingSlot` | `:144-146` |
| `reshard_recv.cc` | `ReshardReceiveSession::HostLayersBounded` | `:168-170` |
| `reshard_recv.cc` | `ReshardReceiveSession::StaticHostLayer` | `:178-186` |
| `reshard_recv.cc` | `ReshardReceiveSession::plan` | `:148-153` |
| `reshard_recv.cc` | `ReshardReceiveSession` (`bool staging_released_` .. `std::atomic<bool> cancelled_`) | `:221-242` |
| `reshard_recv.cc` | `RecordTransferDuration` | `:250-253` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidatePlan` | `:257-334` |
| `reshard_recv.cc` | `CreateReshardReceiveSession` | `:541-575` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReshardReceiveSession` | `:577-602` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:604-643` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:645-650` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:652-655` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishLocked` | `:657-669` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingSlot` | `:671-702` |
| `reshard_recv.cc` | `ReshardReceiveSession::LookupStagingSlot` | `:704-722` |
| `reshard_recv.cc` | `ReshardReceiveSession::Finish` | `:724-727` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndRecvOpLocked` | `:735-745` |
| `reshard_recv.cc` | `ReshardReceiveSession::EndOp` | `:747-750` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolReceivedLocked` | `:752-763` |
| `reshard_recv.cc` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` | `:765-794` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolH2dResultLocked` | `:796-809` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnPoolReceived` | `:811-822` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` | `:824-888` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`to_launch = CollectEligiblePoolH2dsLocked();` .. `in_flight_ += static_cast<int32_t>(to_launch.size());`) | `:831-832` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`EndRecvOpLocked();`) | `:845` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`absl::StatusOr<size_t> slot = LookupStagingSlot(pool_idx);`) | `:859` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` (`future.OnReady`) | `:876-886` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:890-926` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` (`finished = RecordPoolH2dResultLocked(pool_idx, status);`) | `:898` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` (`should_record_duration = true;` .. `Finish(terminal_status);`) | `:917-920` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` | `:305-309` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` |
| `mgr.cc` | `KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer` (`for (const auto& [uuid, entry] : sessions_)`) | `:399-403` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:828-874` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`} else if (session->deadline() <= now) {` .. `absl::DeadlineExceededError("Transfer session timed out"));`) | `:1094-1100` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (uint64_t uuid : retired_uuids)` .. `base_->ReleasePoolStagingLeases(uuid);`) | `:1126-1130` |
| `base.h` | `KVCacheManagerBase::AttachPlaceholderDeviceHoldForTest` | `:767` |
| `base.cc` | `KVCacheManagerBase::GetTransferPlan` | `:3089-3099` |
| `base.cc` | `KVCacheManagerBase::CopyPoolBlocks` | `:2917-3058` |
| `base.cc` | `KVCacheManagerBase::H2dPoolBlocks` | `:3068-3074` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`absl::StatusOr<size_t> slot = absl::NotFoundError("session not found");` .. `} else if (!absl::IsNotFound(slot.status()) ||`) | `:3461-3467` |
| `raw_core.h` | `PjRtCopyFuture::OnReady` | `:514-568` |
| `raw_core.h` | `IssueH2dShard` | `:687-715` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` |
| `bts.cc` | `BlockTransportServerPushSession::ResolveRecvBuffer` | `:510-628` |
| `bts.cc` | `BlockTransportServerPushSession::OnPeregrinePostComplete` | `:663-713` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`ABSL_RETURN_IF_ERROR(session->OnDataReceived(` .. `l, /*minor_dimension_indices=*/{}, /*shard_ids=*/{}));`) | `:886-887` |
| `utils.h` | `ValidateCommonPoolReshardPlan` | `:353-469` |
| `utils.h` | `ValidatePoolBlockBounds` | `:473-489` |
| `reshard_recv_test.cc` | `CreateStatus` | `:150-160` |
| `reshard_recv_test.cc` | `FakeH2dBase` | `:164-191` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.AcceptsCanonicalPlanOnExplicitPools` | `:193-199` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.AcceptsImplicitPools` | `:201-206` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.HasNoTagPolicy` | `:208-214` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsMissingIdentityAndPoolFields` | `:216-242` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsOutOfRangeDuplicateAndDtypeMismatch` | `:244-265` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsBlockIdsOutsideDeclaredPool` | `:294-309` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.GroupScopesDestinationBlocks` | `:428-459` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.BoundedStagingSlotAcquisitionAndRelease` | `:461-537` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.OutOfOrderPoolArrivalResolvesLeasedHostBufferForValidation` | `:543-637` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.FullHostMirrorAddressesPoolsInPlace` | `:659-725` |
| `reshard_send_test.cc` | `RecordingH2dReceiverBase` | `:846-904` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PoolOrderInvertedFromStorageOrderPassesTransportValidateChunks` | `:906-966` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PushesOverPeregrineWithScheduleEntryEndpoint` | `:971-1029` |
| `pool_reshard_test.cc` | `FakeH2dBase` | `:75-109` |
| `pool_reshard_test.cc` | `TestManager::FinishPoolReshardRecvPool` | `:127-137` |
| `pool_reshard_test.cc` | `PoolReshardValidationTest.DeviceOnlyRejectionAtPublicEntryPoints` | `:366-382` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FinishPoolReshardRecvRecordsDurationMetric` | `:422-442` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FinishPoolReshardRecvDoesNotRecordMetricOnFailure` | `:444-462` |
| `pool_reshard_test.cc` | `DeviceBackedTestBase` | `:692-705` |
| `pool_reshard_test.cc` | `BoundedPoolTestManager` | `:707-718` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.DuplicatePoolRecvRegistrationKeepsOriginalStagingLease` | `:720-770` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.ConcurrentDuplicatePoolRecvRegistrationKeepsWinnerStagingLease` | `:772-828` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.StagingAcquireFailureReleasesUuidClaim` | `:830-860` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline` | `:862-900` |
| `pool_reshard_test.cc` | `SubPlanChainTest.ReceiveTerminalWaitsForTheLastSubPlan` | `:921-957` |
| `pool_reshard_test.cc` | `SubPlanChainTest.FirstSubPlanFailureFailsTheRequestOnce` | `:959-982` |
| `pool_reshard_test.cc` | `SubPlanChainTest.RetryUnderTheSameReqIdIsTrackedAsANewChain` | `:984-1027` |
| `pool_reshard_test.cc` | `SubPlanChainTest.StalledChainFailsAfterTheTransferTimeout` | `:1029-1051` |
| `pool_reshard_test.cc` | `SubPlanChainTest.SinglePlanRequestsReportAtOnce` | `:1053-1065` |
| `pool_reshard_test.cc` | `PoolReshardSessionLookupTest.GetSessionResolvesActiveReshardRecvUntilRetired` | `:1097-1129` |
| `pool_reshard_test.cc` | `PoolReshardSessionLookupTest.PoolCompletionOnLegacyArmedReceiverFailsWithoutH2d` | `:1167-1191` |
<!-- source-map:end -->
-/

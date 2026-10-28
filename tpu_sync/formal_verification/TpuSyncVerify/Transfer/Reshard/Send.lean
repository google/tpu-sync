import TpuSyncVerify.Transfer.Reshard.Send.Model
import TpuSyncVerify.Transfer.Reshard.Send.Lemmas
import TpuSyncVerify.Transfer.Reshard.Send.Invariant
import TpuSyncVerify.Transfer.Reshard.Send.Progress
import TpuSyncVerify.Transfer.Reshard.Send.Traces

/-!
# Reshard send session

One `ReshardSendSession` on the producer side and the slice of
`KVCacheManagerWithTransfer` that drives it (`PoolReshardPush`,
`CountPoolReshardSendSlots`, and the `CompleteReadWithDetails` poll). Receiver
sessions live in `Reshard/Receive.lean`; multi-sender stream counting and
aliased-page contents compose both in `Reshard/Pipeline.lean`.

In the source-map table below, `reshard_send.h` and `reshard_send.cc` are
`tpu_sync/kv_cache/reshard_send_session.{h,cc}`; `reshard_recv.cc` is
`tpu_sync/kv_cache/reshard_receive_session.cc`; `slots.cc` is
`tpu_sync/core/pool_reshard_send_slots.cc`; `mgr.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`; `base.cc` is
`tpu_sync/kv_cache/kv_cache_manager_base.cc`; `utils.h` is
`tpu_sync/core/utils.h`; `pool_layout.cc` is `tpu_sync/kv_cache/pool_layout.cc`;
`reshard_send_test.cc` is `tpu_sync/kv_cache/reshard_send_session_test.cc`; and
`pool_reshard_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.

## How a reshard send runs

`KVCacheManagerWithTransfer::PoolReshardPush` creates a `ReshardSendSession`
(`ReshardSendSession::Create`), which validates the plan, computes
`remaining_pool_peer_pushes_ = CountPoolReshardSendSlots(plan, schedule)`,
registers the active plan (`RegisterActivePlanDirect`), and inserts the session
into `active_pool_reshard_sends_`. It then calls
`ExecutePush(*this, src_block_ids)`.

`ReshardSendSession::ExecutePush` takes its own caller op
(`++in_flight_; manager_ = &manager;`) and iterates over
`plan_.transfer_pool_indices()`:
* Pools whose group has no entries on this sender are skipped
  (`if (pool_src_block_ids.empty())`), matching `CountPoolReshardSendSlots`
  (`if (it == peers_by_group.end()) continue;`).
* For each active pool, `ExecutePush` acquires a per-UUID `PoolStagingLease`
  on the pool's storage (`AcquirePoolStagingLease`) and issues
  `base_->D2hPoolBlocks`. On synchronous failure of either step, it calls
  `RecordPushCompletion(manager, status)` and `EndOp()` and returns early.
* On success, it pushes the D2H future to `d2h_futures_`, increments
  `++in_flight_` under `mu_`, and attaches `StartPoolPush` + `EndOp` to
  `pool_future.OnReady`.
* After the loop, `ExecutePush` releases its caller op via `EndOp()`.

When a pool's D2H copy completes, `ReshardSendSession::StartPoolPush` checks
`done_ || finalizing_` under `mu_`, adds `transfers_by_peer.size()` (the number
of distinct destination peers for the pool's group) to `in_flight_`, and
dispatches one `BlockTransport::AsyncPush` per peer, whose `OnReady` callback
calls `RecordPushCompletion` and `EndOp`.

`ReshardSendSession::RecordPushCompletion` returns early if
`done_ || finalizing_`. Otherwise, on the first error (`!status.ok()`) or when
`--remaining_pool_peer_pushes_ == 0`, it latches `finalizing_ = true`.
In shipping C++, `RecordPushCompletion` also sets `plan_unregistered_ = true`
and calls `manager.UnregisterActivePlan(uuid_)` immediately, whereas an
external `ReshardSendSession::Finish` leaves unregistration to
`ReshardSendSession::EndOp` when
`in_flight_ == 1 && finalizing_ && !done_ && !plan_unregistered_`. When
`EndOp()` decrements `in_flight_` to `0` while `finalizing_`,
`ReshardSendSession::SettleLocked` releases the pool staging leases
(`ReleaseLeasesLocked`) and sets `done_ = true`.

## State

The settle protocol `(in_flight_, finalizing_, done_, status_.ok(), !leases_released_)`
is an exact instance of `Transfer.Lifecycle` (`ReshardSendSession::IsDraining`
returns `finalizing_`). Each active pool with `peers > 0` scheduled peer pushes
is a `SendPool := ⟨peers, stage⟩`:

| Pool stage    | C++ (`ReshardSendSession::ExecutePush`, `StartPoolPush`)             | Holds ops in `in_flight_` | Remaining push slots |
|---------------|----------------------------------------------------------------------|---------------------------|----------------------|
| `.pending`    | not yet reached by the `ExecutePush` loop (`for (int32_t encoded_pool_idx : plan_.transfer_pool_indices())`) | `0` | `peers` |
| `.d2h`        | `D2hPoolBlocks` succeeded (`++in_flight_`); `OnReady` not yet finished | `1`          | `peers`              |
| `.pushing k`  | `StartPoolPush` added `peers` ops (`in_flight_ += static_cast<int>(transfers_by_peer.size());`) and released the D2H op; `k` peer `AsyncPush` callbacks remain in flight | `k` | `k` |
| `.aborted`    | D2H failed or `StartPoolPush` returned early on `done_ || finalizing_` | `0`                   | `0`                  |

Session-level fields:

| Field              | C++ (`ReshardSendSession`, `KVCacheManagerWithTransfer`)            | Role |
|--------------------|---------------------------------------------------------------------|------|
| `pools`            | active pools in `plan_.transfer_pool_indices()` and their D2H/push state | per-pool state |
| `life`             | `in_flight_`, `finalizing_`, `done_`, `status_`, `!leases_released_` | `Transfer.Lifecycle` |
| `execActive`       | `ExecutePush` caller op between `++in_flight_;` and `EndOp()`       | caller op active |
| `remainingPushes`  | `remaining_pool_peer_pushes_`                                       | remaining `(pool, peer)` completions |
| `planUnregistered` | `plan_unregistered_`                                                | active plan unregistered |
| `published`        | membership in `done_sending_` / `failed_recving_` (`CompleteReadWithDetails`) | what `poll_stats()` shows the engine |

## Events

| Event              | C++ |
|--------------------|-----|
| `d2hIssue p ok`    | `ExecutePush` loop body for pool `p` (`AcquirePoolStagingLease` .. `pool_future.OnReady`): acquires lease and calls `D2hPoolBlocks`; on failure calls `RecordPushCompletion` + `EndOp` and returns early |
| `execFinish`       | `ExecutePush` finishes its pool loop and calls `EndOp()` |
| `d2hDone p ok`     | `pool_future.OnReady` callback + `StartPoolPush` + `EndOp()` |
| `pushDone p ok`    | one peer's `AsyncPush(...).OnReady` callback for pool `p`: calls `RecordPushCompletion` + `EndOp()` |
| `cancel`           | external `Finish(error)`: deadline (`CompleteReadWithDetails`) or shutdown (`~KVCacheManagerWithTransfer`) |
| `publish`          | `CompleteReadWithDetails` moves a settled session into `done_sending_` or `failed_recving_` and erases it |

## Lock scopes

Every mutable field of `ReshardSendSession` is `ABSL_GUARDED_BY(mu_)`;
`base_`, `staging_allocator_`, `req_id_`, `uuid_`, `parallelism_`, `deadline_`,
`plan_`, `dst_peers_`, and `data_connection_reuse_` are `const`.
`ReshardSendSession::EndOp` is always at least two `mu_` scopes: the
fallback-unregistration check
(`if (in_flight_ == 1 && finalizing_ && !done_ && !plan_unregistered_`), then
the `while (true)` decrement/settle scope — three under `kPerTransfer` when
`release_due` holds (A4).

| Event | `mu_` scopes | Between them |
|---|---|---|
| `d2hIssue p ok` | 1–5 in `ReshardSendSession::ExecutePush` (after unlocked `AcquirePoolStagingLease` and `D2hPoolBlocks`): on `ok = true`, 1 scope (`d2h_futures_.push_back(pool_future); ++in_flight_;`) before `pool_future.OnReady`; on `ok = false`, `RecordPushCompletion` (1 scope) and `EndOp()` (2–3 scopes) before returning early (`execActive = false`) | the caller op keeps `in_flight_ >= 1` throughout `ExecutePush`; no `done_ || finalizing_` re-check, below |
| `execFinish` | 2–3: final `EndOp()` at the end of `ReshardSendSession::ExecutePush` | `EndOp` tail, below |
| `d2hDone p ok` | 3–6 in `pool_future.OnReady`: on `ok = false` (or `transport_srv == nullptr`), `RecordPushCompletion` (1–2 scopes) + `absl::Cleanup` `EndOp()` (2–3 scopes); on `ok = true`, `StartPoolPush` checks `if (done_ || finalizing_) return;` on entry and again when adding `in_flight_ += static_cast<int>(transfers_by_peer.size());` (1–2 scopes, folded by A3), dispatches `AsyncPush` per peer outside `mu_`, and runs `EndOp()` (2–3 scopes) | the D2H op and (on `ok = true`) the added peer ops keep `in_flight_ >= 1` until `EndOp()`; `EndOp` tail, below |
| `pushDone p ok` | 3–5 in `StartPoolPush`'s `AsyncPush(...).OnReady` callback: `RecordPushCompletion` (1–2 scopes) followed by `absl::Cleanup` `EndOp()` (2–3 scopes) | the peer op keeps `in_flight_ >= 1` until `EndOp()`; `EndOp` tail, below |
| `cancel` | 1–2: `ReshardSendSession::Finish(error)`, called from `~KVCacheManagerWithTransfer` (1 scope) or after `!session->IsDraining() && session->deadline() <= now` in `CompleteReadWithDetails` (2 scopes) | `Finish` sets `status_ = status` if `!status.ok() && status_.ok()`, then returns if `done_ || finalizing_`, matching `cancel` |
| `publish` | 2–3 in `CompleteReadWithDetails` (under manager `mu_`): `session->Done()` and `session->GetStatus()` (twice on failure) before `active_pool_reshard_sends_.erase(it++)` | on a settled session only `status_` can change if it was still OK |

**Caller and callback ops hold `in_flight_ >= 1` until `EndOp()`.** Throughout
`ExecutePush`, `StartPoolPush`, and `RecordPushCompletion`, the running caller
or callback holds its own op in `in_flight_` (and `StartPoolPush` adds
`transfers_by_peer.size()` before dropping `mu_` to call `AsyncPush` and
`EndOp()`), so `done_` cannot become `true` before `EndOp()`'s decrement scope.

**No re-check in `ExecutePush`'s loop.** `ExecutePush` does not check
`done_ || finalizing_` before `AcquirePoolStagingLease`, `D2hPoolBlocks`, or
`++in_flight_`. If an external `Finish(error)` or an earlier pool's
`RecordPushCompletion` sets `finalizing_ = true` mid-loop, `ExecutePush`'s
caller op keeps `in_flight_ >= 1` so `SettleLocked()` cannot release leases
while `ExecutePush` is still acquiring them, and any D2H future queued after
`finalizing_ = true` aborts at `StartPoolPush`'s entry check (`d2hDone p true`
when draining).

**`EndOp()` tail.** When `EndOp()` is the last op (`in_flight_ == 1 &&
finalizing_ && !done_`), its pre-decrement scope sets `plan_unregistered_ = true`
(to call `settle_manager->UnregisterActivePlan(uuid_)` outside `mu_`) and, under
`kPerTransfer`, `connections_released_ = true` (A4). Because `finalizing_ == true`
and `in_flight_ == 1` (owned by this `EndOp()` call), `execActive = false` and
`poolOps = 0`: no other op can run, `publish` is disabled (`done_ == false`),
and an interleaved `Finish(error)` only updates `status_` if it was still OK
(matching `cancel` after `EndOp()`).

## Assumptions

* **A1 (non-empty schedule with positive peer counts per active pool).**
  `ReshardSendSession::Create` rejects plans with `remaining_pool_peer_pushes <= 0`,
  and `ExecutePush` skips any pool whose group has no entries on this sender
  (`if (pool_src_block_ids.empty())`), matching `CountPoolReshardSendSlots`
  (`if (it == peers_by_group.end()) continue;`).
  Modeled by `validPools (init peerCounts).pools = true` (`peerCounts ≠ []` and
  `∀ n ∈ peerCounts, 0 < n`).
* **A2 (`ExecutePush` starts before external `Finish`).** `PoolReshardPush`
  inserts the session into `active_pool_reshard_sends_` and immediately invokes
  `ExecutePush`, which increments `++in_flight_` and sets `manager_ = &manager`.
  `init` starts at the end of that opening lock with `execActive = true` and
  `inFlight = 1`.
* **A3 (single `StartPoolPush` lock step).** `StartPoolPush` checks
  `done_ || finalizing_` on entry and again when adding
  `transfers_by_peer.size()` to `in_flight_`. Since the intermediate lines only
  inspect immutable plan fields and `base_->transport_server()`, the two checks
  fold into `d2hDone`.
* **A4 (per-transfer connection-release window in `EndOp`).** Under
  `DataConnectionReuse::kPerTransfer`, the `while (true)` loop in `EndOp`
  checks `in_flight_ == 1 && finalizing_ && !done_ && !connections_released_ &&
  manager_ != nullptr && !manager_->IsShuttingDown()`, latches
  `connections_released_ = true`, drops `mu_` to call
  `ReleaseDataConnectionsAfterSend(base_->transport_server(), dst_peers_, uuid_)`,
  and re-locks `mu_` to run `--in_flight_` and `SettleLocked()`. Because
  `finalizing_` is already `true` and `in_flight_` is `1` (held by this caller),
  no other op is in flight (`execActive = false` and `poolOps = 0`), no new op
  can be started, and a concurrent `Finish(error)` on a session with
  `finalizing_ == true` is a no-op on `in_flight_` and `done_` (it can only flip
  `status_`, matching `cancel`), so the next loop iteration is guaranteed to
  reach `--in_flight_ == 0` and `SettleLocked()` with no observable change to
  any other modelled variable.

## Properties

All proved on every reachable state (`reachable_safe`) for any valid schedule
(`validSchedule peerCounts = true`):

* **Settle safety (`SettleSafe`).** `done → inFlight = 0`.
* **No retired callback (`NoRetiredCallback`).** `done → execActive = false ∧
  poolOps pools = 0`: once settled, `ExecutePush` has returned and no D2H or
  peer push callback remains in flight.
* **Staging integrity (`StagingIntegrity`).** `hasStaging = !done`: the
  sender's `PoolStagingLease`s are released (`ReleaseLeasesLocked`) if and only
  if the session has settled (`SettleLocked`).
* **Prompt settle (`SettlesPromptly`).** `draining → inFlight = 0 → done`.
* **No non-draining idle state (`ActiveOrDraining`).** `draining = false →
  0 < inFlight`: the session never drops to `in_flight_ == 0` before
  `finalizing_` is set.
* **Plan unregistration (`PlanIntegrity`).** `done → planUnregistered` and
  `planUnregistered → draining` under both policies; under
  `UnregisterPolicy.onSettle`, `planUnregistered = done` (`reachable_onSettle_plan_iff_done`),
  so the active plan is never torn down while sibling pools or peer pushes are
  still resolving chunks through `KVCacheManagerBase::GetBlockChunks`
  (`if (schedule_it != schedules.end())`).
* **Slot accounting & publication (`Publication`).** `published = some true →
  remainingPushes = 0 ∧ allPushed pools = true`: `done_sending` is published
  only when `CountPoolReshardSendSlots` has counted down to `0` and every active
  pool has completed its D2H copy and all of its scheduled peer pushes.
* **No op leak & termination (`NoOpLeak`, `reachable_can_settle`).** While
  `0 < inFlight`, some event in `drainEvents` is enabled, and every reachable
  state can drain to `done = true ∧ hasStaging = false ∧ planUnregistered = true`
  in finitely many steps.

## Source map

Line ranges are populated and checked by `tpu_sync/formal_verification/tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `reshard_send.h` | `ReshardSendSession::Done` | `:67-70` |
| `reshard_send.h` | `ReshardSendSession::GetStatus` | `:74-77` |
| `reshard_send.h` | `ReshardSendSession::IsDraining` | `:81-84` |
| `reshard_send.h` | `ReshardSendSession::deadline` | `:88` |
| `reshard_send.h` | `ReshardSendSession` (`int remaining_pool_peer_pushes_` .. `bool connections_released_`) | `:138-151` |
| `reshard_send.cc` | `ReshardSendSession::ValidatePlan` | `:52-90` |
| `reshard_send.cc` | `ReshardSendSession::Create` | `:92-151` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` | `:177-265` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` (`manager_ = &manager;`) | `:183` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` (`for (int32_t encoded_pool_idx : plan_.transfer_pool_indices())`) | `:204-262` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` (`d2h_futures_.push_back(pool_future);` .. `++in_flight_;`) | `:250-251` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` (`pool_future.OnReady`) | `:253-261` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:267-349` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` (`in_flight_ += static_cast<int>(transfers_by_peer.size());`) | `:318` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` (`for (const auto& [peer, transfers] : transfers_by_peer)`) | `:329-348` |
| `reshard_send.cc` | `ReshardSendSession::RecordPushCompletion` | `:351-392` |
| `reshard_send.cc` | `ReshardSendSession::Finish` | `:394-410` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:418-468` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` (`if (in_flight_ == 1 && finalizing_ && !done_ && !plan_unregistered_`) | `:428-432` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` (`while (true)`) | `:449-467` |
| `reshard_send.cc` | `ReshardSendSession::ReleaseLeasesLocked` | `:470-474` |
| `reshard_send.cc` | `ReshardSendSession::SettleLocked` | `:476-481` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:668-706` |
| `slots.cc` | `CountPoolReshardSendSlots` | `:27-51` |
| `mgr.cc` | `KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer` (`for (const auto& [uuid, session] : active_pool_reshard_sends_)`) | `:410-414` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:664-701` |
| `mgr.cc` | `KVCacheManagerWithTransfer::UnregisterActivePlan` | `:741-778` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_sends_.begin();`) | `:1011-1030` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` | `:3235-3519` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`if (schedule_it != schedules.end())`) | `:3414-3446` |
| `utils.h` | `StridedSpanFitsBlock` | `:322-333` |
| `utils.h` | `StridedSpanFitsRegions` | `:335-349` |
| `pool_layout.cc` | `RegionCoversOffset` | `:66-87` |
| `pool_layout.cc` | `RegionsCoverRange` | `:257-274` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.AcceptsCanonicalPlanOnExplicitPools` | `:151-159` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.AcceptsImplicitPools` | `:161-168` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.RejectsMissingIdentityAndPoolFields` | `:170-202` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.RejectsOverflowingSenderSpan` | `:204-218` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.RejectsSenderSpanBetweenPackedTokenRegionsInAliasedStorage` | `:220-254` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.IgnoresReceiverCoverageGap` | `:256-277` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeSettling` | `:322-362` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush` | `:364-404` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PooledPolicyKeepsIdlePeerConnections` | `:406-435` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.CreateRejectsUnknownDataConnReuseValue` | `:437-453` |
| `pool_reshard_test.cc` | `PoolReshardSendTest.SenderWithNoBytesForAnyTransferredPoolRefusedUpFront` | `:302-325` |
<!-- source-map:end -->
-/

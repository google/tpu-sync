import TpuSyncVerify.Transfer.Reshard.Receive.Model
import TpuSyncVerify.Transfer.Reshard.Receive.Lemmas
import TpuSyncVerify.Transfer.Reshard.Receive.Invariant
import TpuSyncVerify.Transfer.Reshard.Receive.Progress
import TpuSyncVerify.Transfer.Reshard.Receive.Traces

/-!
# Reshard receive session

One `ReshardReceiveSession` on the destination side, driven by incoming pushes
(`BeginIncomingPush` / `CompleteIncomingPush` / `EndIncomingPush`),
`PoolReshardRegisterRecv` (`Reshard/PoolStaging.lean`), and
`CompleteReadWithDetails` (`Transfer/Session.lean`, `Reshard/SubPlanChain.lean`).
Sender sessions and aliased-page contents live in `Reshard/Send.lean` and
`Reshard/Pipeline.lean`.

File aliases: `reshard_recv.{h,cc}` = `tpu_sync/kv_cache/reshard_receive_session.{h,cc}`,
`base.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`, `raw_core.h` =
`tpu_sync/core/raw_transfer_core.h`, `utils.h` = `tpu_sync/core/utils.h`,
`reshard_recv_test.cc` = `tpu_sync/kv_cache/reshard_receive_session_test.cc`,
`pool_reshard_test.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.

## State

Each expected pool is a `Pool := ⟨rank, stage⟩` in `pools : List Pool` over `Transfer.Lifecycle`:

| Pool stage / Field | C++ (`ReshardReceiveSession`) | Role |
|---|---|---|
| `.waiting` / `.received` | `expected_pool_indices_` / `started_pool_indices_` before `h2d_launched_pools_` | waiting on pushes / rank prerequisites (no op) |
| `.queued` | `h2d_launched_pools_` after `in_flight_ += to_launch.size()` | holds op before per-pool draining re-check |
| `.issued` / `.ready` | `H2dPoolBlocks` future in flight / finished before `OnReady` runs | holds op in `in_flight_` |
| `.completed` / `.aborted` | `completed_pool_indices_` after `EndOp()` / skipped or failed op ended | terminal pool stages (no op) |
| `life` | `in_flight_`, `draining_`, `done_`, `status_`, `!staging_released_` | `Transfer.Lifecycle` (`Transfer/Session.lean`) |
| `finalizing` | `reshard_finalizing_` | halts H2D launches/results after error or completion |
| `durationRecorded` | ghost (`should_record_duration` / `RecordTransferDuration`) | transfer latency metric recorded |
| `published` | `RetireSettledSessionLocked` → `RecordPoolReshardTerminalLocked` | `done_recving_` / `failed_recving_` or chain slot |
| `pushes` | ghost | open pushes between `TryBeginOp` and `EndOp` |

## Events

| Event | C++ |
|---|---|
| `pushBegin` | `BeginIncomingPush` (`GetSession(uuid)` + `ReshardReceiveSession::TryBeginOp`) |
| `pushEnd ok` | `EndIncomingPush`: `.finishOnError` (shipping) runs `Finish(status)` on `!ok` before `EndOp()`; `.endOpOnly` runs `EndOp()` only |
| `onPoolReceived p` | `OnDataReceived` → `OnPoolReceived` (`RecordPoolReceivedLocked`) + `ExecuteEligiblePoolH2ds` opening batch lock |
| `h2dIssue p ok` | `ExecuteEligiblePoolH2ds` loop body: `done_ \|\| draining_` re-check, `LookupStagingSlot`, `H2dPoolBlocks` |
| `h2dReady p` | device finishes pool `p`'s H2D copy (`future.OnReady` pending) |
| `h2dDone p ok` | `future.OnReady` callback: `FinishPoolH2d` (`RecordPoolH2dResultLocked`, next wave or `Finish`) + `EndOp()` |
| `cancel` | external `Finish(error)` (deadline sweep, `UnregisterActivePlan`, destructor, or staging slot timeout) |
| `publish` | `CompleteReadWithDetails` retires `Done()` session via `RetireSettledSessionLocked` and releases staging/push progress |

## Lock scopes

Mutable fields of `ReshardReceiveSession` are `ABSL_GUARDED_BY(mu_)` except
`staging_layer_manager_` (own mutex) and atomic `cancelled_`.

| Event | `mu_` scopes | Between them |
|---|---|---|
| `pushBegin` | 1 `mu_` scope (`TryBeginOp`) after `GetSession` under manager `mu_` | retired session is `done_` and refuses the op |
| `pushEnd ok` | 1–2 `mu_` scopes in `EndIncomingPush`: optional `Finish(status)`, then `EndOp()` | caller's push op holds `in_flight_ >= 1` (`EndOp` tail) |
| `onPoolReceived p` | 2 `mu_` scopes: `OnPoolReceived`, then `ExecuteEligiblePoolH2ds` opening lock | folded into one transition by A4 |
| `h2dIssue p ok` | 1–5 `mu_` scopes: re-check, `LookupStagingSlot`, unlocked `H2dPoolBlocks`, error cleanup | pool op counted before `H2dPoolBlocks`; `EndOp` tail on error |
| `h2dReady p` | 0 (device copy completion) | — |
| `h2dDone p ok` | 2–4 `mu_` scopes in `future.OnReady`: `FinishPoolH2d`, then `EndOp()` | `reshard_finalizing_` latch, wave fold (A4), and `EndOp` tail |
| `cancel` | 1 `mu_` scope (`FinishLocked`) after optional poll or unlocked `Done()` check | unguarded in model; late errors still flip `status_` |
| `publish` | 2 `mu_` scopes in `RetireSettledSessionLocked` (`Done()`, `GetStatus()`) under manager `mu_` | settled session only allows late error status flip |

* **`reshard_finalizing_` and wave launch (`FinishPoolH2d`).** `RecordPoolH2dResultLocked`
  latches `reshard_finalizing_` in the first `mu_` scope so concurrent completions/launches
  no-op before `Finish`; on intermediate OK, `ExecuteEligiblePoolH2ds` folds with `EndOp()` (A4).
* **`EndOp` tail and late errors.** Any unreleased op holds `in_flight_ >= 1`
  (`done_ = false`, `Transfer/Session.lean`); `FinishLocked` records `!status.ok()`
  before `if (draining_) return;`, so `cancel` needs no `!draining` guard.

### Bounded host layer staging

Per-pool `StagingLayerManager` slot waits under `HostLayersBounded()` (`AcquireStagingSlot`,
`Transfer/LayerStaging.lean`) nest inside the push op (`pushes > 0`) and map timeouts or
cancellations to `cancel` / `pushEnd false`, preserving `Safe` (`docs/transfer/reshard.md` §Boundaries).

## Assumptions

* **A1 (at-most-once `OnPoolReceived` per declared pool).** `BlockTransportServerPushSession::CompleteIncomingPush`
  delivers `session->OnDataReceived(l, {}, {})` once per pool at `plan_pushes == expected_chunks`,
  and `RecordPoolReceivedLocked` rejects undeclared/duplicate pools (`pool.stage == .waiting`).
* **A2 (registration before visibility).** `PoolReshardRegisterRecv` claims `uuid` in
  `pending_pool_reshard_recvs_` before `CreateReshardReceiveSession` runs `AcquireStagingLeases`,
  seating `sessions_[uuid]` only on success (`Reshard/PoolStaging.lean`).
* **A3 (no double end).** Each push and H2D op ends at most once: `pushEnd` requires
  `pushes > 0`, `h2dIssue p` requires `stage == .queued`, and `h2dDone p` requires `stage == .ready`.
* **A4 (fold of `RecordPool*Locked` with `ExecuteEligiblePoolH2ds` first lock).** An external
  `Finish` between recording a pool and locking `mu_` in `ExecuteEligiblePoolH2ds` sets `draining_`
  and skips queuing (`cancel` before), whereas one after `in_flight_ += to_launch.size()` aborts
  queued pools at the per-pool re-check (`cancel` after).

## Properties

Proved on every reachable state (`reachable_safe`) under both `PushEndPolicy` values:
* **`SettleSafe` & `NoRetiredCallback`**: `done → inFlight = 0` and `done → poolOps pools = 0`
  (`if (done_) return;` in `FinishPoolH2d` is unreachable; `plan()` is non-empty iff `!done`).
* **`StagingIntegrity` & `SettlesPromptly`**: `hasStaging = !done`, and `draining → inFlight = 0 → done`.
* **`RankWaveOrder`** (`rankWaveOrder_iff`): if `p.stage.launched` and `q.rank < p.rank`,
  then `q.stage = .completed` (`CollectEligiblePoolH2dsLocked`).
* **`Publication`**: `published = some true → allCompleted pools ∧ durationRecorded`.
* **`NoOpLeak` & `reachable_can_settle`**: `0 < inFlight` enables a `drainEvents` step, and
  every reachable state settles to `done ∧ ¬hasStaging` in finitely many steps.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
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
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:890-926` |
| `base.h` | `KVCacheManagerBase::AttachPlaceholderDeviceHoldForTest` | `:767` |
| `base.cc` | `KVCacheManagerBase::H2dPoolBlocks` | `:3068-3074` |
| `raw_core.h` | `PjRtCopyFuture::OnReady` | `:514-568` |
| `raw_core.h` | `IssueH2dShard` | `:687-715` |
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
| `pool_reshard_test.cc` | `FakeH2dBase` | `:75-109` |
| `pool_reshard_test.cc` | `TestManager::FinishPoolReshardRecvPool` | `:127-137` |
| `pool_reshard_test.cc` | `PoolReshardValidationTest.DeviceOnlyRejectionAtPublicEntryPoints` | `:366-382` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FinishPoolReshardRecvRecordsDurationMetric` | `:422-442` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FinishPoolReshardRecvDoesNotRecordMetricOnFailure` | `:444-462` |
| `pool_reshard_test.cc` | `DeviceBackedTestBase` | `:692-705` |
| `pool_reshard_test.cc` | `PoolReshardRecvTest.FailedIncomingPushReleasesStagingLeaseBeforeDeadline` | `:862-900` |
| `pool_reshard_test.cc` | `PoolReshardSessionLookupTest.GetSessionResolvesActiveReshardRecvUntilRetired` | `:1097-1129` |
| `pool_reshard_test.cc` | `PoolReshardSessionLookupTest.PoolCompletionOnLegacyArmedReceiverFailsWithoutH2d` | `:1167-1191` |
<!-- source-map:end -->
-/

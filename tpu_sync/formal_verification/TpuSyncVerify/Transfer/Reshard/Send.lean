import TpuSyncVerify.Transfer.Reshard.Send.Model
import TpuSyncVerify.Transfer.Reshard.Send.Lemmas
import TpuSyncVerify.Transfer.Reshard.Send.Invariant
import TpuSyncVerify.Transfer.Reshard.Send.Progress
import TpuSyncVerify.Transfer.Reshard.Send.Traces

/-!
# Reshard send session

One `ReshardSendSession` on the producer side and its `KVCacheManagerWithTransfer`
driver (`PoolReshardPush`, `CountPoolReshardSendSlots` in `Reshard/Coverage.lean`,
and `CompleteReadWithDetails` retirement in `Transfer/Session.lean` /
`Reshard/SubPlanChain.lean`). Receiver sessions and aliased-page contents live in
`Reshard/Receive.lean` and `Reshard/Pipeline.lean`.

File aliases: `reshard_send.{h,cc}` = `tpu_sync/kv_cache/reshard_send_session.{h,cc}`,
`mgr.cc` = `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`, `base.cc` =
`tpu_sync/kv_cache/kv_cache_manager_base.cc`, `raw_core.h` =
`tpu_sync/core/raw_transfer_core.h`, `utils.h` = `tpu_sync/core/utils.h`,
`pool_layout.cc` = `tpu_sync/kv_cache/pool_layout.cc`, `reshard_send_test.cc` =
`tpu_sync/kv_cache/reshard_send_session_test.cc`, `pool_reshard_test.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`.

## State

Each active pool with `peers > 0` scheduled peer pushes is a `SendPool := ⟨peers, stage⟩`
over `Transfer.Lifecycle` (`IsDraining` returns `finalizing_`):

| Pool stage / Field | C++ (`ReshardSendSession`) | Role |
|---|---|---|
| `.pending` | active pool not yet reached by `ExecutePushPipeline` | `0` ops in flight; `peers` remaining slots |
| `.d2h` | `D2hPoolBlocks` in flight (`++in_flight_`) | `1` op in flight; `peers` remaining slots |
| `.pushing k` | `StartPoolPush` added `peers` ops and ended D2H op | `k` peer `AsyncPush` ops and slots remaining |
| `.aborted` | D2H/lease/slot failed, or `StartPoolPush` exited on `done_ \|\| finalizing_` | `0` ops in flight; `0` remaining slots |
| `life` | `in_flight_`, `finalizing_`, `done_`, `status_`, `!leases_released_` | `Transfer.Lifecycle` (`Transfer/Session.lean`) |
| `execActive` | caller op between `Start`'s `++in_flight_` and pipeline thread `EndOp()` | keeps `in_flight_ >= 1` while issuing D2H |
| `remainingPushes` | `remaining_pool_peer_pushes_` | remaining `(pool, peer)` completions |
| `published` | `RetireSettledSessionLocked` → `RecordPoolReshardTerminalLocked` | `done_sending_` / `failed_recving_` or chain slot |

## Events

| Event | C++ |
|---|---|
| `startFail` | `Start` fails `ValidatePushPreconditions` or `std::thread`: `RecordPushCompletion` + `absl::Cleanup` `EndOp()` |
| `d2hIssue p ok` | `ExecutePushPipeline` pool step: `AcquireStagingSlot`, `AcquirePoolStagingLease`, `D2hPoolBlocks` (or error + thread `EndOp()`) |
| `execFinish` | pipeline thread's `EndOp()` after finishing its `ordered_pools` loop |
| `d2hDone p ok` | `pool_future.OnReady` callback: `StartPoolPush` (or `RecordPushCompletion` on error) + `EndOp()` |
| `pushDone p ok` | one peer's `AsyncPush(...).OnReady` callback: `RecordPushCompletion` + `EndOp()` |
| `cancel` | external `ReshardSendSession::Finish(error)` (deadline sweep, `UnregisterActivePlan`, destructor); no-op once `done_ \|\| finalizing_` |
| `publish` | `CompleteReadWithDetails` retires `Done()` session via `RetireSettledSessionLocked` and erases `sessions_[uuid]` |

## Lock scopes

Mutable fields of `ReshardSendSession` are `ABSL_GUARDED_BY(mu_)` except
`staging_layer_manager_` (own mutex) and atomic `cancelled_`.

| Event | `mu_` scopes | Between them |
|---|---|---|
| `startFail` | 2–3 `mu_` scopes in `Start`: `RecordPushCompletion`, then `EndOp()` | caller op keeps `in_flight_ >= 1` until `EndOp()` |
| `d2hIssue p ok` | 2–6 `mu_` scopes in `ExecutePushPipeline`: slot acquire, re-check, D2H or error cleanup | caller op holds `in_flight_ >= 1`; unlocked D2H gap below |
| `execFinish` | 1–2 `mu_` scopes: pipeline thread's `EndOp()` | `EndOp` tail (`Transfer/Session.lean`, A4) |
| `d2hDone p ok` | 2–5 `mu_` scopes in `pool_future.OnReady`: `StartPoolPush` / error record, then `EndOp()` | D2H and added peer ops hold `in_flight_ >= 1` (A3) |
| `pushDone p ok` | 2–3 `mu_` scopes in `AsyncPush` callback: `RecordPushCompletion`, then `EndOp()` | peer op holds `in_flight_ >= 1` until `EndOp()` |
| `cancel` | 1 `mu_` scope (`Finish`), after optional unlocked or poll checks | `if (done_ \|\| finalizing_) return;` makes late `Finish` a no-op |
| `publish` | 2 `mu_` scopes in `RetireSettledSessionLocked` (`Done()`, `GetStatus()`) under manager `mu_` | settled send state is immutable |

* **Caller op and pipeline re-check window.** `Start`'s caller op holds `in_flight_ >= 1`
  across `ExecutePushPipeline` so `SettleLocked()` cannot release leases mid-loop; if
  `finalizing_` is latched during unlocked `AcquirePoolStagingLease` / `D2hPoolBlocks`
  after the re-check, the queued D2H aborts at `StartPoolPush`'s entry check (`d2hDone p true`).

### Bounded host layer staging

Per-pool `StagingLayerManager` waits under `HostLayersBounded()` (`AcquireStagingSlot`,
`Transfer/LayerStaging.lean`) run inside `d2hIssue` while the caller op holds `in_flight_ >= 1`,
mapping timeouts/cancellations to `d2hIssue p false` (`docs/transfer/reshard.md` §Boundaries).

## Assumptions

* **A1 (non-empty schedule with positive peer counts, one endpoint per `dst_peer`).**
  `CreateReshardSendSession` rejects `remaining_pool_peer_pushes <= 0` and `ExecutePushPipeline`
  skips zero-entry pools (`validPools`). Entries of one group sharing `dst_peer` are assumed to
  carry one `dst_peregrine_peer` (`BuildPoolReshardPlan`), matching `CountPoolReshardSendSlots`
  to `StartPoolPush`'s distinct `PeerEndpoint` pushes.
* **A2 (`Start` runs before external `Finish`, fresh UUID per send).** `PoolReshardPush`
  seats `sessions_[uuid]` and immediately runs `Start`'s `++in_flight_` (`init` has
  `execActive = true, inFlight = 1`). Because `PoolReshardPush` has no pending-UUID claim
  (`Reshard/PoolStaging.lean`), concurrent sends sharing one `uuid` are assumed absent.
* **A3 (single `StartPoolPush` lock step).** `StartPoolPush` checks `done_ || finalizing_`
  on entry and again when adding `transfers_by_peer.size()` to `in_flight_`, reading only
  immutable plan fields and `transport_server()` in between; both fold into `d2hDone`.
* **A4 (`kPerTransfer` connection-release window in `EndOp`).** Under `DataConnectionReuse::kPerTransfer`,
  the last op (`in_flight_ == 1 && finalizing_ && !done_`) drops `mu_` in `EndOp` for
  `ReleaseDataConnectionsAfterSend` before `--in_flight_` and `SettleLocked()` (`Transfer/Session.lean`);
  with `finalizing_` latched and no other op in flight, nothing modelled changes in the gap.

## Properties

Proved on every reachable state (`reachable_safe`) for any `validSchedule peerCounts = true`:
* **`SettleSafe` & `NoRetiredCallback`**: `done → inFlight = 0` and `done → execActive = false ∧ poolOps pools = 0`
  (`plan()` is non-empty iff `!done`; `trace_partial_failure_keeps_plan`).
* **`StagingIntegrity` & `SettlesPromptly`**: `hasStaging = !done`, and `draining → inFlight = 0 → done`.
* **`ActiveOrDraining`**: `draining = false → 0 < inFlight` (`in_flight_` never hits `0` before `finalizing_`).
* **`Publication`**: `published = some true → remainingPushes = 0 ∧ allPushed pools = true`.
* **`NoOpLeak` & `reachable_can_settle`**: `0 < inFlight` enables a `drainEvents` step, and
  every reachable state drains to `done ∧ ¬hasStaging` in finitely many steps.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `reshard_send.cc` | `ReshardSendSession::Done` | `:96-99` |
| `reshard_send.cc` | `ReshardSendSession::TryBeginOp` | `:103` |
| `reshard_send.cc` | `ReshardSendSession::GetStatus` | `:110-113` |
| `reshard_send.cc` | `ReshardSendSession::IsDraining` | `:117-120` |
| `reshard_send.cc` | `ReshardSendSession::deadline` | `:122-124` |
| `reshard_send.cc` | `ReshardSendSession::ResolveStagingSlot` | `:135-137` |
| `reshard_send.cc` | `ReshardSendSession::StaticHostLayer` | `:171-179` |
| `reshard_send.cc` | `ReshardSendSession::HostLayersBounded` | `:183-185` |
| `reshard_send.cc` | `ReshardSendSession::plan` | `:150-155` |
| `reshard_send.cc` | `ReshardSendSession` (`int remaining_pool_peer_pushes_` .. `std::atomic<bool> cancelled_`) | `:219-229` |
| `reshard_send.cc` | `EntryPeer` | `:232-235` |
| `reshard_send.cc` | `ReshardSendSession::ValidatePlan` | `:239-277` |
| `reshard_send.cc` | `CreateReshardSendSession` | `:279-341` |
| `reshard_send.cc` | `ReshardSendSession::AcquireStagingSlot` | `:366-394` |
| `reshard_send.cc` | `ReshardSendSession::LookupStagingSlot` | `:396-417` |
| `reshard_send.cc` | `ReshardSendSession::ValidatePushPreconditions` | `:419-435` |
| `reshard_send.cc` | `ReshardSendSession::Start` | `:437-474` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePushPipeline` | `:476-590` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:592-697` |
| `reshard_send.cc` | `ReshardSendSession::RecordPushCompletion` | `:699-718` |
| `reshard_send.cc` | `ReshardSendSession::Finish` | `:720-738` |
| `reshard_send.cc` | `ReshardSendSession::OnDataReceived` | `:746-752` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:754-782` |
| `reshard_send.cc` | `ReshardSendSession::ReleaseLeasesLocked` | `:784-789` |
| `reshard_send.cc` | `ReshardSendSession::SettleLocked` | `:791-798` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:733-775` |
| `base.cc` | `KVCacheManagerBase::D2hPoolBlocks` | `:3060-3066` |
| `raw_core.h` | `IssueD2hShard` | `:657-685` |
| `utils.h` | `StridedSpanFitsBlock` | `:322-333` |
| `utils.h` | `StridedSpanFitsRegions` | `:335-349` |
| `pool_layout.cc` | `RegionCoversOffset` | `:66-87` |
| `pool_layout.cc` | `RegionsCoverRange` | `:257-274` |
| `reshard_send_test.cc` | `StubRecvSession` | `:68-95` |
| `reshard_send_test.cc` | `MakeTestBase` | `:97-103` |
| `reshard_send_test.cc` | `CreateStatus` | `:171-181` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.AcceptsCanonicalPlanOnExplicitPools` | `:183-189` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.AcceptsImplicitPools` | `:191-196` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.RejectsMissingIdentityAndPoolFields` | `:198-224` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.RejectsOverflowingSenderSpan` | `:226-239` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.RejectsSenderSpanBetweenPackedTokenRegionsInAliasedStorage` | `:241-272` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.IgnoresReceiverCoverageGap` | `:274-293` |
| `reshard_send_test.cc` | `FakeD2hBase` | `:341-427` |
| `reshard_send_test.cc` | `RouteSessionLookups` | `:431-441` |
| `reshard_send_test.cc` | `AcceptPushes` | `:444-453` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsBeforeSettling` | `:475-514` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidCopy` | `:516-560` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PooledPolicyKeepsIdlePeerConnections` | `:562-593` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.CreateRejectsUnknownDataConnReuseValue` | `:595-612` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.BoundedStagingSlotAcquisitionAndRelease` | `:614-708` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.StartFailsFastWithoutTransportServer` | `:713-738` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PipelineSlotWaitTimeoutSettlesSendOutOfBand` | `:744-793` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.StagingSlotReleasedAfterLastPeerPush` | `:798-844` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PushesOverPeregrineWithScheduleEntryEndpoint` | `:971-1029` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PushOverPeregrineWithoutScheduleEntryEndpointFailsClosed` | `:1034-1067` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.FullHostMirrorStagesPoolsInPlace` | `:1074-1120` |
| `pool_reshard_test.cc` | `PoolReshardSendTest.SenderWithNoBytesForAnyTransferredPoolRefusedUpFront` | `:397-420` |
| `pool_reshard_test.cc` | `TestManager::AddPoolReshardSend` | `:138-154` |
| `pool_reshard_test.cc` | `PoolReshardSessionLookupTest.GetSessionResolvesActiveReshardSendUntilRetired` | `:1131-1162` |
<!-- source-map:end -->
-/

import TpuSyncVerify.Transfer.Reshard.Pipeline.Model
import TpuSyncVerify.Transfer.Reshard.Pipeline.Lemmas
import TpuSyncVerify.Transfer.Reshard.Pipeline.Invariant
import TpuSyncVerify.Transfer.Reshard.Pipeline.Preservation
import TpuSyncVerify.Transfer.Reshard.Pipeline.Progress
import TpuSyncVerify.Transfer.Reshard.Pipeline.Traces

/-!
# Pool-reshard transfer pipeline

End-to-end single-plan pool-reshard model (`docs/transfer/reshard.md`; multi-window
sub-plan chaining is `Reshard/SubPlanChain.lean`), composing `Reshard.Send`,
`Reshard.Recv`, plan-declared stream progress in `BlockTransportServerPushSession`,
the `ReshardCoordinator::RunSubPlan` Arm-then-Dispatch barrier, and five
pool-indexed memories with aliased-page rank overwrites (`writePoolHbm`,
over-approximating `KVCacheManagerBase::CopyPoolBlocks`).

File aliases: `base.{h,cc}` = `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`,
`bts.cc` = `tpu_sync/transport/block_transport_session.cc`,
`reshard_send_test.cc` = `tpu_sync/kv_cache/reshard_send_session_test.cc`.

## State

Memories are `List Cell` (`Cell := .blank | .kv p | .junk`) indexed by pool `p < numPools`:

| Field | C++ | Role |
|---|---|---|
| `send`, `recv` | `ReshardSendSession`, `ReshardReceiveSession` | producer and consumer session states (`Send.lean`, `Receive.lean`) |
| `srcHbm`, `dstHbm` | producer and consumer device HBM (`CopyPoolBlocks`) | `writePoolHbm` writes `dstHbm[p]` and clobbers `dstHbm[q] := .junk` for `rank p < rank q` |
| `srcStaging`, `dstStaging` | per-pool host staging (`StagingLayerManager`, `PoolStagingStorage`, or mirror) | disjoint per-pool `Cell` lists (A1; `docs/transfer/reshard.md` §Boundaries) |
| `wire` | push payload from `ReshardSendSession::StartPoolPush` | written at `send (.pushDone p true)` |
| `sentStreams`, `landedStreams` | `AsyncPush` completions, `CompleteIncomingPush` `plan_pushes` | per-pool sent and landed counts against `peerCounts` (A2, A3) |
| `reclaimed`, `armed` | caller HBM free flag, `RunSubPlan` Stage 1 barrier | ghost flags (`init` starts armed, `initUnarmed` unarmed) |

## Events

| Event | C++ |
|---|---|
| `armRecv` | `ReshardCoordinator::RunSubPlan` Stage 1 (`PoolReshardRegisterRecv`) |
| `send e` | `Send.step` (`d2hIssue p true` requires `armed`; `d2hDone` stages `srcHbm`; `pushDone` writes `wire` and `sentStreams`) |
| `recv e` | `Recv.step` (`pushBegin` requires `armed`; `onPoolReceived p` requires `landedStreams[p] = peerCounts[p]`; `h2dDone` runs `writePoolHbm`) |
| `landStream p` | `StartStream` payload read into `dstStaging[p]` + `CompleteIncomingPush` `plan_pushes` increment while `recv.pushes > 0` |
| `reclaim` | caller engine reclaims `srcHbm` after `send.published ≠ none` |
| `reseatSrcStaging`, `reseatDstStaging` | host staging reuse after `!send.life.hasStaging` or `!recv.life.hasStaging` |

## Lock scopes

Sender and receiver transitions inherit `Send.lean` and `Receive.lean` scopes.

| Event | Scopes | Between them |
|---|---|---|
| `armRecv` | 2 manager `mu_` scopes in `PoolReshardRegisterRecv` around `CreateReshardReceiveSession` | `RunSubPlan` thread join blocks Stage 2 dispatch until `sessions_` seats receiver |
| `landStream p` | unlocked `StartStream` socket read, then 1 push-session `mu_` scope in `CompleteIncomingPush` | active `pushBegin` op keeps `recv.life.inFlight > 0`, disabling `reseatDstStaging` |
| `reclaim`, `reseatSrcStaging`, `reseatDstStaging` | 0 (caller engine or `staging_mu_`, outside session `mu_`) | enabled only after publication or staging release under session/manager locks |

## Assumptions

Session assumptions from `Send.lean` and `Receive.lean` carry over, plus:

* **A1 (disjoint host staging across pools; no same-rank device overlap).** `srcStaging` and
  `dstStaging` keep one cell per pool, and `clobberHigherRanks` clobbers only strictly higher
  `order_rank`s (`CollectEligiblePoolH2dsLocked`). Device-overlapping pools are covered only
  across distinct `order_rank`s under `HostLayersBounded()`; full-mirror shared-storage overlap
  (`StaticHostLayer`) and same-rank device overlap are outside the model (`docs/transfer/reshard.md` §Boundaries).
* **A2 (`landStream` ordered after `pushDone`).** On TCP sockets (`PostSocketPushInternal` /
  `PostSocketPush`), `StartStream` lands and acks before `AsyncPush(...).OnReady` (`pushDone`),
  whereas the model orders `landStream` after `pushDone`. For succeeding pushes, `srcStaging[p]`
  is constant while the push op is held (`StagingSafe`); mid-push landings and landed-then-failed
  pushes are not model states (`docs/transfer/reshard.md` §Assumptions).
* **A3 (one stream equals one plan push from one sender peer push).** `peerCounts[p]` is both
  `Send.init`'s peer count and `CompleteIncomingPush`'s `expected_pushes`. That `AsyncPushShards`
  per-NUMA stream fan-out and `AccumulateLandedShards` fold to `expected_pushes` is assumed
  (`Reshard/Coverage.lean` proves the validator check only).

## Properties

Proved for `sys` (`reachable_safe`, `reachable_can_settle`, `attention_safe`) and `sysUnarmed`
(`reachable_unarmed_safe`, `reachable_unarmed_can_settle`) within A1–A3:
* **`PublicationCorrect` & `attention_safe`**: `recv.published = some true → dstHbm = good numPools`,
  preserved across all subsequent transitions (`dstHbm_quiet`).
* **`AliasedPageCorrect`**: any completed pool `q` holds `dstHbm[q] = .kv q` across later H2D waves.
* **`DstHbmSafe`, `SrcHbmSafe`, `StagingSafe`**: no H2D/D2H/push op touches HBM after publication
  (`srcHbm_quiet`) or staging after lease release.
* **`StreamAccountingSafe` & `NoUnarmedPush`**: `OnPoolReceived(p)` fires only at
  `landedStreams[p] = peerCounts[p]`, and neither sender D2H nor receiver `pushBegin` runs while `!armed`.
* **`NoOpLeak`**: `Send.NoOpLeak` and `Recv.NoOpLeak`, draining every reachable state to settlement.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `base.cc` | `KVCacheManagerBase::CopyPoolBlocks` | `:2917-3058` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`progress.completed_chunks++;` .. `if (trigger_completion)`) | `:772-882` |
| `tpu_sync/transport/lib/adapter/socket_transport_adapter.cc` | `SocketTransportAdapter::PostSocketPush` (`if (remaining_workers->fetch_sub(1) == 1) {`) | `:474-495` |
| `tpu_sync/transport/lib/adapter/socket_transport_adapter.cc` | `SocketTransportAdapter::PostSocketPushInternal` (`ReadExact(fd, &ack, 1, config_.final_ack_read_timeout_ms);`) | `:706` |
| `base.h` | `KVCacheManagerBase` (`absl::StatusOr<raiden::PjRtCopyFuture> D2hPoolBlocks` .. `absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks`) | `:643-652` |
| `reshard_send_test.cc` | `RecordingH2dReceiverBase` | `:846-904` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PoolOrderInvertedFromStorageOrderPassesTransportValidateChunks` | `:906-966` |
<!-- source-map:end -->
-/

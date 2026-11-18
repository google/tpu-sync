import TpuSyncVerify.Transfer.Reshard.Pipeline.Model
import TpuSyncVerify.Transfer.Reshard.Pipeline.Lemmas
import TpuSyncVerify.Transfer.Reshard.Pipeline.Invariant
import TpuSyncVerify.Transfer.Reshard.Pipeline.Preservation
import TpuSyncVerify.Transfer.Reshard.Pipeline.Progress
import TpuSyncVerify.Transfer.Reshard.Pipeline.Traces

/-!
# Pool-reshard transfer pipeline

End-to-end pool-addressed resharding model (`docs/transfer/reshard.md`):
composes `Reshard.Send` (`ReshardSendSession`, `Send.lean`), `Reshard.Recv`
(`ReshardReceiveSession`, `Receive.lean`), the plan-declared stream progress in
the per-uuid `BlockTransportServerPushSession` (`StartStream`, reached from
`BlockTransport::HandleIncomingPush`, and `CompleteIncomingPush`), the
Arm-then-Dispatch coordination barrier in `ReshardCoordinator::RunSubPlan`, and
the five memories KV data moves
through between source and destination HBM — including **aliased device pages**
shared across pool ranks (`pool_order_ranks_` in `ReshardReceiveSession`,
`D2hPoolBlocks` / `H2dPoolBlocks` in `KVCacheManagerBase`).

File aliases used in the source map below:
* `reshard_send.cc` — `tpu_sync/kv_cache/reshard_send_session.cc`
* `slots.cc` — `tpu_sync/core/pool_reshard_send_slots.cc`
* `reshard_recv.cc` — `tpu_sync/kv_cache/reshard_receive_session.cc`
* `bt.cc` — `tpu_sync/transport/block_transport.cc`
* `bts.cc` — `tpu_sync/transport/block_transport_session.cc`
* `mgr.cc` — `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`
* `base.h` / `base.cc` — `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`
* `coord.cc` — `tpu_sync/kv_cache/reshard/reshard_coordinator.cc`
* `utils.h` — `tpu_sync/core/utils.h`
* `reshard_send_test.cc` — `tpu_sync/kv_cache/reshard_send_session_test.cc`

## Scope: one plan

The model is one transfer plan — one engine `uuid` with one
`ReshardSendSession` per source unit, one `ReshardReceiveSession` per
destination unit, and one `BlockTransportServerPushSession` per receiving
transport. A request whose destination lists are windowed
(`dst_window_blocks`) is coordinated as a chain of such plans
(`PlanSubTransfers`, `ReshardCoordinator::StartChain`):
each sub-plan carries its own plan uuid (`SubPlanUuid`), is armed and
dispatched by its own `ReshardCoordinator::RunSubPlan` call, and lands in its
own staging leases, so everything below applies to each sub-plan in turn. How
a worker folds the sub-plans' settlements into the request's single
`done_sending` / `done_recving` / `failed_recving` entry
(`KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked`,
`RecordPoolReshardTerminalLocked`, `SweepSubPlanChainsLocked`) is modelled in
`Reshard/SubPlanChain.lean`; `send.published` and `recv.published` here are the
session's own terminal as `CompleteReadWithDetails` hands it to that
bookkeeping.

## Memories and aliased device pages

A memory is a list indexed by pool `p < numPools`. A `Cell` is `.kv p` (pool
`p`'s resharded KV data), `.junk` (stale or reclaimed memory, or the bytes a
lower-rank pool's H2D copy leaves in a higher-rank pool's slot where their
declared-live regions overlap), or `.blank` (never written on the wire).

| Memory       | Role | Written by |
|--------------|------|------------|
| `srcHbm`     | producer KV cache in source HBM | `reclaim` (after `send.published ≠ none`) |
| `srcStaging` | sender host staging of pool `p`: a host layer from the session's `StagingLayerManager` under `HostLayersBounded()`, else its leased arena slots on a bounded `PoolStagingStorage` or the blocks' own bytes in the host mirror (`ReshardSendSession::ExecutePushPipeline`) | `send (.d2hDone p true)`, `reseatSrcStaging` |
| `wire`       | bytes delivered by sender peer pushes (`ReshardSendSession::StartPoolPush`) | `send (.pushDone p true)` |
| `dstStaging` | receiver host staging of pool `p`: a host layer from the session's `StagingLayerManager` under `HostLayersBounded()`, else its leased arena slots on a bounded `PoolStagingStorage` or the blocks' own bytes in the host mirror (`ReshardReceiveSession::AcquireStagingLeases`) | `landStream p` (`BlockTransportServerPushSession::StartStream`), `reseatDstStaging` |
| `dstHbm`     | consumer KV cache in destination HBM (`ReshardReceiveSession::ExecuteEligiblePoolH2ds` / `FinishPoolH2d`) | `recv (.h2dDone p true)` via `writePoolHbm` |

`srcStaging` and `dstStaging` hold one cell per pool `p`: the model treats the
host staging bytes of different pools as disjoint. Assumption A1 below says
when the C++ gives that and when it does not.

### Modelling aliased device pages (`writePoolHbm`)

On hybrid attention + state arenas (`D2hPoolBlocks` / `H2dPoolBlocks` in
`KVCacheManagerBase`, `pool_order_ranks_` in `ReshardReceiveSession`), pools of
different ranks can share physical HBM pages. A pool's H2D copy writes exactly
its declared-live regions of the transferred blocks
(`KVCacheManagerBase::CopyPoolBlocks` over `ComputePoolBlockCopyExtents`;
padding between regions is never copied), so it overwrites another pool's
device bytes only where the two pools share a `storage_index` and their
declared-live regions overlap on the same storage. Where a lower-rank
pool's regions overlap a higher-rank pool's, the higher-rank H2D waves are
meant to run afterwards and overwrite those bytes with the higher-rank pool's
data.

We model this pessimistically across ranks on `dstHbm`, as if every lower-rank pool
overlapped every strictly higher-rank pool: when pool `p`
completes its H2D copy (`recv (.h2dDone p true)`), `writePoolHbm` writes
`dstStaging[p]` to `dstHbm[p]` **and** clobbers `dstHbm[q] := .junk` for every
pool `q` of strictly higher rank (`pool_p.rank < pool_q.rank`). Pools of equal
`order_rank` upload concurrently in the C++ (`CollectEligiblePoolH2dsLocked`
waits only on `other_rank < rank`) and never clobber each other in
`writePoolHbm`; device overlap between same-rank pools is assumed absent and is
not checked by the C++. Because
`Recv.RankWaveOrder` guarantees that no higher-rank pool `q` has entered
`queued`, `issued`, `ready`, or `completed` when a lower-rank pool `p` is in
`ready`, this clobbering never overwrites a completed higher-rank pool
(`AliasedPageCorrect`), so `dstHbm = good numPools` holds when `done_recving`
is published (`PublicationCorrect`). Conversely, a mutant that omits the rank
gate allows pool 1 (rank 1) to finish H2D before pool 0 (rank 0), and pool 0's
subsequent H2D clobbers `dstHbm[1]` to `.junk`. These results cover two
device-overlapping pools only when they have distinct `order_rank`s and their
host staging is disjoint (A1).

## Lock scopes

Sender events (`send e`) and receiver events (`recv e`) inherit the
`ReshardSendSession::mu_` and `ReshardReceiveSession::mu_` scopes from
`Send.lean` and `Receive.lean`. The pipeline-level events and cross-component
guards map to C++ synchronization as follows:

| Event | Scopes | Between them |
|---|---|---|
| `armRecv` | 2 `KVCacheManagerWithTransfer::mu_` scopes in `PoolReshardRegisterRecv`: incumbent check and the `pending_pool_reshard_recvs_` claim, then `CreateReshardReceiveSession` (`AcquireStagingLeases`) outside manager `mu_`, then insertion into `sessions_` as a `SessionEntry` with `SessionRole::kPoolReshardReceive` (dropping the claim, and `NoteSubPlanRegisteredLocked`) | Stage 1's `for (std::thread& t : armers) t.join();` barrier in `RunSubPlan` completes before Stage 2 dispatches any sender (`send (.d2hIssue p true)`), so no push arrives before `sessions_` holds the receiver (`recv .pushBegin` requires `armed = true`; `BeginIncomingPush` fails `GetSession` otherwise) |
| `landStream p` | unlocked socket read into staging (`if (expected_size > 0)`, `ReadVExact`) in `BlockTransportServerPushSession::StartStream` while holding a `pushBegin` op (`pushes > 0`; the chunk addresses come from `GetBlockChunks` → `ResolveStagingSlot`, `Receive.lean` §"Bounded host layer staging"), followed by 1 `BlockTransportServerPushSession::mu_` scope in `CompleteIncomingPush` (`if (plan_declared)`; `layer_progress_` is per push session, i.e. per uuid, keyed by pool) | the `pushBegin` op keeps `in_flight_ > 0` on the receive session so `dstStaging` cannot be released (`reseatDstStaging` is disabled) during the read; when `plan_pushes` reaches `expected_chunks`, the same `mu_` scope latches `trigger_completion = true` before `session->OnDataReceived(l, {}, {})` on the session whose op the stream holds — `ReshardReceiveSession::OnDataReceived`, which forwards a pool completion to `OnPoolReceived(pool_idx)` (`recv (.onPoolReceived p)`) — and `EndIncomingPush` (`pushEnd`) run |
| `reclaim`, `reseatSrcStaging`, `reseatDstStaging` | 0 (caller engine or `KVCacheManagerBase::staging_mu_`, outside session `mu_`) | enabled only after `send.published ≠ none` or `hasStaging = false`, which `Send.lean` and `Receive.lean` establish under their session locks |

## Assumptions

The session-level assumptions of `Send.lean` and `Receive.lean` carry over.
The composition adds:

* **A1 (host staging bytes of different pools are disjoint, and same-rank pools
  do not overlap on the device).** `srcStaging` and `dstStaging` hold one cell
  per pool:
  - *Host staging disjointness.* With bounded host layers (`HostLayersBounded()`,
    `num_staging_layers < num_layers`) every concurrently staged pool within one
    session leases its own host layer buffer from that session's
    `StagingLayerManager` (`AcquireStagingSlot`), so the session's cells are
    disjoint whatever the pools' layouts (cross-session host-layer sharing is
    addressed in `MultiRequest.lean` A1). On a full host mirror
    (`!HostLayersBounded()`) `AcquireStagingSlot` returns
    `StaticHostLayer(pool_idx) = pool->storage_index`, and
    `KVCacheManagerBase::CopyPoolBlocks` uses the device offsets on the host on
    full-mirror storages (`{extent.offset_bytes, extent.offset_bytes,
    extent.size_bytes}`), or shifts by `(slot - block_id) * stride` within a
    bounded `PoolStagingStorage` lease (which one uuid's pools on that storage
    share). Two pools whose declared-live regions overlap on a shared storage
    therefore share host staging bytes on a full host mirror: a stream landing
    for one overwrites what landed for the other before either H2D copy runs,
    which per-pool cells cannot express.
  - *Same-rank device disjointness.* `clobberHigherRanks` only clobbers pools of
    strictly higher `order_rank` (`other_rank < rank` in
    `ReshardReceiveSession::CollectEligiblePoolH2dsLocked`), so equal-rank pools
    are assumed not to overlap on the device in any configuration.
  Neither disjointness property is enforced by the C++. Consequently
  `AliasedPageCorrect`, `PublicationCorrect`, `system_data_correct`, and their
  `MultiRequest` / `Unarmed` counterparts cover device-overlapping pools **of
  different `order_rank`s** only under bounded host layers (within a session,
  and in the multi-transfer setting assuming concurrent sessions do not collide
  in shared host layer buffers); on a full host mirror they hold for the
  disjoint-host configurations above (pools on different storages or with
  disjoint declared-live byte ranges), in which no pool's H2D copy touches
  another pool's device bytes and `clobberHigherRanks` only over-approximates.
* **A2 (a stream lands after its sender push completed).** `landStream p`
  requires `landedStreams[p] < sentStreams[p]`, and `sentStreams[p]` and
  `wire[p]` are written by `send (.pushDone p true)`. On the TCP socket backend
  the order on the wire is the reverse: in
  `BlockTransportServerPushSession::StartStream`, the receiver reads the
  payload, runs `CompleteIncomingPush` and `EndIncomingPush` and only then writes
  the one-byte ack; the sender's `SocketTransportAdapter::PostSocketPushInternal`
  returns only after reading that ack (`ReadExact(fd, &ack, 1,
  config_.final_ack_read_timeout_ms)`), and `PostSocketPush`'s
  `finish_stream` runs `AsyncPush(...).OnReady` (`pushDone`) after all streams
  of the group complete. (With multiple NUMA shard groups `AsyncPush` joins
  futures via `tsl::JoinFutures`, and the opt-in Peregrine backend completes in
  `OnPeregrinePostComplete` with no socket ack; neither external future join nor
  Peregrine wire ordering is examined here.) For a push that succeeds, the data
  results do not depend on the order: while a push is in flight the sender holds
  an op, so `srcStaging[p]` is neither released nor rewritten (`StagingSafe`;
  its only writers are `d2hDone p true`, which precedes the push, and
  `reseatSrcStaging`), and the value the model copies at `pushDone` is the value
  the receiver read; sender events touch only the sender-side fields and receiver
  events only the receiver-side ones, so a C++ run whose push succeeds is a
  model run with the receiver's landing and what follows it delayed past
  `pushDone`.

  What the model cannot express: a state in which a stream has landed (and its
  pool may have uploaded, or the receive published) while the sender's push op
  is still in flight, and a push that landed on the receiver and then fails on
  the sender (e.g. ack write or read failure). In the model a failed push
  (`pushDone p false`) never lands; on the C++ wire a receiver can still publish
  `done_recving` on a landed-then-failed push, so its data integrity rests on
  the same informal constancy of `srcStaging[p]` while the push op was held, not
  on `PublicationCorrect`. `StreamAccountingSafe`'s `landedStreams ≤ sentStreams`
  is a fact about this encoding, not about the C++ counters.
* **A3 (one stream is one plan push from one sender peer push).**
  `peerCounts[p]` is both the sender's number of peer pushes for pool `p`
  (`Send.init peerCounts`) and the number of landings the receiver waits for
  before `OnPoolReceived(p)` (`plan_pushes == expected_chunks` in
  `CompleteIncomingPush`), and the model has one `Send`. In the C++ the
  receiver's expectation is the plan's `expected_pushes`, one `AsyncPush` of a
  peer opens `min(parallelism, blocks)` TCP streams per NUMA shard group
  (`BlockTransport::AsyncPushShards`), and
  `BlockTransportServerPushSession::AccumulateLandedShards` folds the shard
  groups back into plan pushes, so one peer push can land several plan pushes
  where the model lands one. That this fan-out produces exactly
  `expected_pushes` plan pushes per pool is assumed, not modelled
  (`Reshard/Coverage.lean` proves the receiver-side validator's contract only).

## Source map

Line ranges are populated and checked by `tpu_sync/formal_verification/tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `base.cc` | `KVCacheManagerBase::CopyPoolBlocks` | `:2917-3058` |
| `reshard_recv.cc` | `ReshardReceiveSession::StaticHostLayer` | `:178-186` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingSlot` | `:671-702` |
| `bt.cc` | `BlockTransport::AsyncPushShards` | `:571-644` |
| `bts.cc` | `BlockTransportServerPushSession::AccumulateLandedShards` | `:715-722` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` | `:371-505` |
| `reshard_send.cc` | `ReshardSendSession::Start` | `:437-474` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePushPipeline` | `:476-590` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:592-697` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:754-782` |
| `reshard_send.cc` | `ReshardSendSession::SettleLocked` | `:791-798` |
| `slots.cc` | `CountPoolReshardSendSlots` | `:27-51` |
| `reshard_recv.cc` | `ReshardReceiveSession` (`absl::flat_hash_map<size_t, int> pool_order_ranks_`) | `:235` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnDataReceived` | `:128-135` |
| `reshard_recv.cc` | `CreateReshardReceiveSession` | `:541-575` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:604-643` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:645-650` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolReceivedLocked` | `:752-763` |
| `reshard_recv.cc` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` | `:765-794` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnPoolReceived` | `:811-822` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` | `:824-888` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:890-926` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` | `:460-469` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`if (expected_size > 0)`) | `:473-477` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:724-893` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` (`progress.completed_chunks++;` .. `if (trigger_completion)`) | `:772-882` |
| `tpu_sync/transport/lib/adapter/socket_transport_adapter.cc` | `SocketTransportAdapter::PostSocketPush` (`if (remaining_workers->fetch_sub(1) == 1) {`) | `:474-495` |
| `tpu_sync/transport/lib/adapter/socket_transport_adapter.cc` | `SocketTransportAdapter::PostSocketPushInternal` (`ReadExact(fd, &ack, 1, config_.final_ack_read_timeout_ms);`) | `:706` |
| `tpu_sync/kv_cache/staging_manager.h` | `StagingLayerManager` | `:39-74` |
| `bts.cc` | `BeginIncomingPush` | `:242-256` |
| `bts.cc` | `EndIncomingPush` | `:258-271` |
| `mgr.cc` | `KVCacheManagerWithTransfer::GetSession` | `:311-318` |
| `mgr.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:733-775` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = sessions_.begin(); it != sessions_.end();)` .. `SweepSubPlanChainsLocked(now);`) | `:1084-1110` |
| `base.h` | `KVCacheManagerBase` (`absl::StatusOr<raiden::PjRtCopyFuture> D2hPoolBlocks` .. `absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks`) | `:643-652` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`const int64_t receiver_arm_start_ns = MonotonicNs();` .. `if (!status.ok()) return status;`) | `:744-880` |
| `utils.h` | `ValidateCommonPoolReshardPlan` (`if (plan.transfer_pool_indices().empty())` .. `if (group.expected_pushes() <= 0)`) | `:367-377` |
| `reshard_send_test.cc` | `RecordingH2dReceiverBase` | `:846-904` |
| `reshard_send_test.cc` | `ReshardSendSessionTest.PoolOrderInvertedFromStorageOrderPassesTransportValidateChunks` | `:906-966` |
<!-- source-map:end -->
-/

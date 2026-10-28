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
`BlockTransport::HandleIncomingPush` / `BlockTransport::CompleteIncomingPush`,
the Arm-then-Dispatch coordination barrier in
`ReshardCoordinator::ExecutePoolReshard`, and the five memories KV data moves
through between source and destination HBM — including **aliased device pages**
shared across pool ranks (`pool_order_ranks_` in `ReshardReceiveSession`,
`D2hPoolBlocks` / `H2dPoolBlocks` in `KVCacheManagerBase`).

File aliases used in the source map below:
* `reshard_send.cc` — `tpu_sync/kv_cache/reshard_send_session.cc`
* `slots.cc` — `tpu_sync/core/pool_reshard_send_slots.cc`
* `reshard_recv.h` / `reshard_recv.cc` — `tpu_sync/kv_cache/reshard_receive_session.{h,cc}`
* `bt.cc` — `tpu_sync/transport/block_transport.cc`
* `mgr.cc` — `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`
* `base.h` — `tpu_sync/kv_cache/kv_cache_manager_base.h`
* `coord.cc` — `tpu_sync/kv_cache/reshard/reshard_coordinator.cc`
* `utils.h` — `tpu_sync/core/utils.h`

## Memories and aliased device pages

A memory is a list indexed by pool `p < numPools`. A `Cell` is `.kv p` (pool
`p`'s resharded KV data), `.junk` (stale/reclaimed memory or lower-rank coarse
padding on an aliased page), or `.blank` (never written on the wire).

| Memory       | Role | Written by |
|--------------|------|------------|
| `srcHbm`     | producer KV cache in source HBM | `reclaim` (after `send.published ≠ none`) |
| `srcStaging` | sender `PoolStagingStorage` (`ReshardSendSession::ExecutePush`) | `send (.d2hDone p true)`, `reseatSrcStaging` |
| `wire`       | bytes delivered by sender peer pushes (`ReshardSendSession::StartPoolPush`) | `send (.pushDone p true)` |
| `dstStaging` | receiver `PoolStagingStorage` (`ReshardReceiveSession::AcquireStagingLeases`) | `landStream p` (`BlockTransport::HandleIncomingPush`), `reseatDstStaging` |
| `dstHbm`     | consumer KV cache in destination HBM (`ReshardReceiveSession::ExecuteEligiblePoolH2ds` / `FinishPoolH2d`) | `recv (.h2dDone p true)` via `writePoolHbm` |

### Modelling aliased device pages (`writePoolHbm`)

On hybrid attention + state arenas (`D2hPoolBlocks` / `H2dPoolBlocks` in
`KVCacheManagerBase`, `pool_order_ranks_` in `ReshardReceiveSession`), pools of
different ranks can share physical HBM pages: a lower-rank pool's coarse H2D
copy writes across the page (clobbering bytes belonging to higher-rank pools),
and higher-rank H2D waves are meant to run afterwards to overwrite those bytes
with the true higher-rank pool data.

We model this pessimistic page aliasing directly on `dstHbm`: when pool `p`
completes its H2D copy (`recv (.h2dDone p true)`), `writePoolHbm` writes
`dstStaging[p]` to `dstHbm[p]` **and** clobbers `dstHbm[q] := .junk` for every
pool `q` of strictly higher rank (`pool_p.rank < pool_q.rank`). Because
`Recv.RankWaveOrder` guarantees that no higher-rank pool `q` has entered
`queued`, `issued`, `ready`, or `completed` when a lower-rank pool `p` is in
`ready`, this clobbering never overwrites a completed higher-rank pool
(`AliasedPageCorrect`), so `dstHbm = good numPools` holds when `done_recving`
is published (`PublicationCorrect`). Conversely, a mutant that omits the rank
gate allows pool 1 (rank 1) to finish H2D before pool 0 (rank 0), and pool 0's
subsequent H2D clobbers `dstHbm[1]` to `.junk`.

## Lock scopes

Sender events (`send e`) and receiver events (`recv e`) inherit the
`ReshardSendSession::mu_` and `ReshardReceiveSession::mu_` scopes from
`Send.lean` and `Receive.lean`. The pipeline-level events and cross-component
guards map to C++ synchronization as follows:

| Event | Scopes | Between them |
|---|---|---|
| `armRecv` | 2 `KVCacheManagerWithTransfer::mu_` scopes in `PoolReshardRegisterRecv`: incumbent check, then `ReshardReceiveSession::Create` (`RegisterActivePlanDirect` + `AcquireStagingLeases`) outside manager `mu_`, then insertion into `active_pool_reshard_recvs_` | Stage 1's `for (std::thread& t : armers) t.join();` barrier in `ExecutePoolReshard` completes before Stage 2 dispatches any sender (`send (.d2hIssue p true)`), so no push arrives before `active_pool_reshard_recvs_` is populated (`recv .pushBegin` requires `armed = true`) |
| `landStream p` | unlocked socket write (`if (expected_size > 0)`) in `HandleIncomingPush` while holding a `pushBegin` lease (`pushes > 0`), followed by 1 `BlockTransport::progress_mu_` scope in `CompleteIncomingPush` (`if (plan_declared)`) | the `pushBegin` lease keeps `in_flight_ > 0` on the receive session so `dstStaging` cannot be released (`reseatDstStaging` is disabled) during the write; when `plan_pushes` reaches `expected_chunks`, the same `progress_mu_` scope latches `trigger_completion = true` before `OnPoolReceived` (`recv (.onPoolReceived p)`) and `EndIncomingPush` (`pushEnd`) run |
| `reclaim`, `reseatSrcStaging`, `reseatDstStaging` | 0 (caller engine or `KVCacheManagerBase::staging_mu_`, outside session `mu_`) | enabled only after `send.published ≠ none` or `hasStaging = false`, which `Send.lean` and `Receive.lean` establish under their session locks |

## Source map

Line ranges are populated and checked by `tpu_sync/formal_verification/tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` | `:177-265` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:267-349` |
| `reshard_send.cc` | `ReshardSendSession::EndOp` | `:418-468` |
| `reshard_send.cc` | `ReshardSendSession::SettleLocked` | `:476-481` |
| `slots.cc` | `CountPoolReshardSendSlots` | `:27-51` |
| `reshard_recv.h` | `ReshardReceiveSession` (`absl::flat_hash_map<size_t, int> pool_order_ranks_`) | `:189` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` | `:351-390` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:420-459` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:461-465` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolReceivedLocked` | `:541-552` |
| `reshard_recv.cc` | `ReshardReceiveSession::CollectEligiblePoolH2dsLocked` | `:554-583` |
| `reshard_recv.cc` | `ReshardReceiveSession::OnPoolReceived` | `:600-612` |
| `reshard_recv.cc` | `ReshardReceiveSession::ExecuteEligiblePoolH2ds` | `:614-666` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:668-706` |
| `bt.cc` | `BlockTransport::HandleIncomingPush` (`if (expected_size > 0)`) | `:604-608` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` | `:691-887` |
| `bt.cc` | `BlockTransport::CompleteIncomingPush` (`progress.completed_chunks++;` .. `if (trigger_completion)`) | `:736-864` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.on_pool_received =`) | `:180-202` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.begin_incoming_push =`) | `:224-249` |
| `mgr.cc` | `KVCacheManagerWithTransfer::InitializeBaseHooks` (`hooks.end_incoming_push =`) | `:250-296` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:664-701` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:703-739` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_sends_.begin();`) | `:1011-1030` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_recvs_.begin();`) | `:1067-1097` |
| `base.h` | `KVCacheManagerBase` (`absl::StatusOr<raiden::PjRtCopyFuture> D2hPoolBlocks` .. `absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks`) | `:696-702` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`const int64_t receiver_arm_start_ns = MonotonicNs();` .. `if (!status.ok()) return status;`) | `:432-543` |
| `utils.h` | `ValidateCommonPoolReshardPlan` (`if (plan.transfer_pool_indices().empty())` .. `if (group.expected_pushes() <= 0)`) | `:367-377` |
<!-- source-map:end -->
-/

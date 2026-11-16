import TpuSyncVerify.Transfer.PrefillDecode.GroupLanding.Model
import TpuSyncVerify.Transfer.PrefillDecode.GroupLanding.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.GroupLanding.Traces

/-!
# Early H2D copies of a layer's landed block groups

Models one layer of a pulled `TransferReceiveSession` with push groups set
(`KVCacheManagerBase::SetPushGroups`) on a full host mirror: each push stream
reports the blocks it landed (`BlockTransportServerPushSession::StartStream` →
`TransferReceiveSession::OnBlocksLanded`) before it counts toward the layer,
and once a group's worth of landed blocks waits (`LayerH2d::pending`), the
receive copies them to the device without waiting for the layer. When the
layer's last stream counts, `ExecuteLayerH2d` copies the rest, and the layer
completes when all of its copies have (`TransferReceiveSession::OnLayerH2dDone`,
`LayerH2d::outstanding`).

Abbreviations: `recv.cc` = `tpu_sync/kv_cache/transfer_receive_session.cc`,
`recv.h` = `tpu_sync/kv_cache/transfer_receive_session.h`, `bts.cc` =
`tpu_sync/transport/block_transport_session.cc`, `send_drain_test.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_send_drain_test.cc`. Exact
line ranges for every symbol and sub-block are in the generated
`## Source map` table below.

## What is modelled

`Receive.lean` models the receive session with one H2D copy per layer, issued
when the layer lands (its A1, "one dispatch per layer"), which is what a
receive runs without push groups, under bounded host layer staging (a layer
has no host buffer of its own to copy from early), or when its load plan maps a
host block to several device blocks: `InitFromLoadPlan` then leaves
`h2d_group_bytes_` at `0`. With push groups a layer has several copies, each
holding an op (`++in_flight_`) until its callback, and
`num_completed_layers_` counts a layer once, when its last copy finishes. This
model is that per-layer state machine; toward `Receive.lean` the layer's copies
act as its one copy: `CompleteLayerLocked` runs once per layer, after every
copy of the layer has finished, and only once the layer has landed.

Blocks are counted by where they are, not named: which block waits in
`pending` or rides in which copy changes nothing the properties observe, and
`LayerH2d::taken` makes every block join at most one copy. Shards are reduced
to two groups (`landHalf`, `landOther`): a stream that carries only some shards
of its blocks, as when a push is split by source NUMA node, leaves them to the
layer's last copy (`shard_ids.size() != base_->num_shards()`).

## State

| Field | C++ | Role |
|---|---|---|
| `numBlocks` | `load_plan_.h2d_host_block_ids.size()` | the layer's blocks |
| `groupBlocks` | least `k` with `k * block_bytes * num_shards >= h2d_group_bytes_`; `0` when `h2d_group_bytes_ == 0` | group size |
| `maxStreams` | — | bounds the searched instances; not read by the proofs |
| `life` | `in_flight_`, `draining_`, `done_`, `status_`, staging (`Lifecycle`) | settle protocol |
| `streams` | ghost | push streams holding their lease (`BeginIncomingPush` .. `EndIncomingPush`) |
| `fresh`, `half` | ghost | blocks not landed; blocks with one shard group landed |
| `untaken` | ghost | landed blocks no copy took (partial-shard streams, or reports a draining receive ignored) |
| `pending` | `LayerH2d::pending.size()` | landed blocks waiting for a group |
| `pendingHalf` | ghost | always `0`; for the mutant that groups partially landed blocks |
| `countedCopies`, `counted` | ghost | copies counted in `LayerH2d::outstanding` and `in_flight_` that `DispatchLayerH2d` has not handed to the device, and their blocks |
| `copyingCopies`, `copying` | ghost | copies on the device, and their blocks; `outstanding = countedCopies + copyingCopies` |
| `copied`, `lost` | ghost | blocks whose copy finished OK; blocks of failed or dropped copies |
| `landed` | `LayerH2d::landed` | the layer's last stream counted |
| `executed` | the transport's `on_layer_received_called` | `OnLayerReceived` ran for the layer |
| `completed` | the layer's `num_completed_layers_` increments | |
| `late`, `badRead` | ghost | a copy reached the device after completion; a copy read a block not fully landed |

## Events

| Event | C++ |
|---|---|
| `streamBegin` | `BeginIncomingPush` → `TryBeginOp` (refused once draining or settled) |
| `landFull k` | a stream carrying every shard lands `k` blocks; `OnBlocksLanded`: `if (done_ || draining_ || h2d_group_bytes_ == 0 || ...)` and `if (layer.landed)` return; otherwise the blocks join `pending` (`layer.taken.insert(host).second`) and, at a group's worth, `BuildGroupCopySpecLocked`, `++layer.outstanding`, `++in_flight_` |
| `landHalf`, `landOther` | a stream carrying only some shards lands one shard group of a block; `OnBlocksLanded` returns (`shard_ids.size() != base_->num_shards()`) |
| `streamEnd` | a stream that does not complete the layer: `CompleteIncomingPush`, `OnBlockShardsReceived`, `EndIncomingPush` |
| `streamLast` | the layer's last stream: `CompleteIncomingPush` → `OnLayerReceived` → `ExecuteLayerH2d`'s first lock (`executeLayer`): `layer.landed = true`; the rest (pending, then every block no copy took) forms the last copy, or with none left and `layer.outstanding == 0`, `CompleteLayerLocked`; then the stream ends |
| `dispatch k ok` | `DispatchLayerH2d`: `if (done_ || draining_)` → `--layer_h2d_[layer_idx].outstanding`, `EndRecvOpLocked()`; `H2dSyncDispatch` failing → `FinishLocked(future_or.status(), kH2dPhase)`, `EndRecvOpLocked()`; otherwise the copy runs and its future joins `h2d_futures_` |
| `copyDone k ok` | `OnLayerH2dDone`: `--layer.outstanding`; OK and `layer.landed && layer.outstanding == 0` → `CompleteLayerLocked`; error → `FinishLocked(status, kH2dPhase)`; then `absl::Cleanup end_op` |
| `cancel` | `Finish(error)`: the deadline sweep, a consumer cancel |

`CompleteLayerLocked` is `++num_completed_layers_` and, on the receive's last
layer unless draining, the success `FinishLocked()`: the model's one layer is
the last (`completeLayer`).

## Lock scopes

| Event | Scopes | Between them |
|---|---|---|
| `landFull` | `OnBlocksLanded`'s scope (`mu_`), then `DispatchLayerH2d` for a group copy | the copy is counted (`++layer.outstanding`, `++in_flight_`) in the first scope, so the gap is the `dispatch` event's race window: a `Finish` in it makes `DispatchLayerH2d` drop the copy; the layer cannot complete in it, since its outstanding count is positive |
| `streamLast` | `ExecuteLayerH2d`'s first scope; for the last copy its `DispatchLayerH2d` and `RecordLayerDispatchedLocked` scopes, then the stream's `OnBlockShardsReceived` and `EndIncomingPush` | as for `landFull`; the rest copy is counted in the first scope |
| `dispatch` | the re-check, `H2dSyncDispatch` (unlocked), on failure a second scope, on success the scope registering the future | `Receive.lean`'s `h2dIssue` |
| `copyDone` | `OnLayerH2dDone`'s scope, then `Release(layer)` (a no-op on a full mirror), then `EndOp` | an `EndRecvOp` tail (`Receive.lean`) |
| others | one scope | — |

## Assumptions

* **A1 (once per layer).** The transport calls `OnLayerReceived` once per layer
  (`Transport.lean`'s `OnceDispatched`, the latch
  `on_layer_received_called`). Encoded as `streamLast` requiring
  `executed = false`.
* **A2 (land before count).** `StartStream` reports a stream's landed blocks
  (`session->OnBlocksLanded`) after its payload is in host staging and before
  the stream counts toward the layer (`CompleteIncomingPush`), and the layer
  completes on its last stream; so when the layer's last stream counts, every
  block of the layer has landed in full and every group copy was counted.
  Encoded as `streamLast` requiring `fresh = 0` and `half = 0`.
  `stepCountBeforeLand` shows the layer completing without blocks that land
  later.
* **A3 (network accounting).** A stream's block report follows its
  `OnBlocksLanded`, and the last stream's follows `ExecuteLayerH2d`, so
  `Receive.lean`'s A4 ("a layer's blocks are accounted once its copy is
  issued") holds for every copy of the layer.

## Properties

* **`CompleteSound`** — a layer that completed has no copy counted or
  running, and on an OK receive every one of its blocks has been copied to the
  device. This is what `num_completed_layers_` promises `Receive.lean`.
* **`CompletesOnce`** — `CompleteLayerLocked` runs at most once for the layer.
* **`NoLateCopy`** — no copy reaches the device after the layer completed.
* **`CopiesReadLanded`** — every copy reads blocks whose shards have all
  landed.
* **`SettledQuiet`** — a settled receive has no copy of the layer counted or
  running.
* **`GroupsEager`** — fewer than a group's worth of blocks ever waits.

`reachable_safe`: every reachable state is `Safe`, for any number of blocks and
any group size, `0` (no groups) included. `Traces.lean` replays the receive
tests below and shows counterexamples for four mutants: completing while
another copy is outstanding, completing before the layer landed, grouping the
blocks of a partial-shard stream, and counting the last stream before every
block landed (A2).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `recv.h` | `TransferReceiveSession` (`struct LayerH2d {` .. `absl::flat_hash_map<int64_t, int64_t> h2d_device_of_host_`) | `:381-397` |
| `recv.cc` | `TransferReceiveSession::InitFromLoadPlan` (`if (destination_ == BlockLocation::kDevice && group_bytes > 0 &&`) | `:382-398` |
| `recv.cc` | `TransferReceiveSession::BuildGroupCopySpecLocked` | `:793-810` |
| `recv.cc` | `TransferReceiveSession::OnBlocksLanded` | `:812-858` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` | `:860-938` |
| `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`if (rest.empty())`) | `:896-904` |
| `recv.cc` | `TransferReceiveSession::DispatchLayerH2d` | `:940-983` |
| `recv.cc` | `TransferReceiveSession::CompleteLayerLocked` | `:985-995` |
| `recv.cc` | `TransferReceiveSession::OnLayerH2dDone` | `:1021-1064` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`session->OnBlocksLanded(`) | `:492-493` |
| `send_drain_test.cc` | `RecvLifecycleTest.PushGroupsCopyLandedBlocksBeforeTheirLayerLands` | `:956-993` |
| `send_drain_test.cc` | `RecvLifecycleTest.PushGroupsCompleteALayerWhoseCopiesAlreadyFinished` | `:997-1021` |
| `send_drain_test.cc` | `RecvLifecycleTest.WithoutPushGroupsALayerIsCopiedWhenItLands` | `:1025-1039` |
| `send_drain_test.cc` | `RecvLifecycleTest.PushGroupsCopyWholeLayersUnderBoundedHostStaging` | `:1043-1061` |
| `send_drain_test.cc` | `RecvLifecycleTest.FailedPushGroupCopyFailsTheReceiveAfterTheOthersEnd` | `:1065-1081` |
<!-- source-map:end -->
-/

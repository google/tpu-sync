import TpuSyncVerify.Transfer.Reshard.Pipeline

/-!
# Sub-block byte coverage and multi-sender stream accounting

Horizontal data-plane refinement of `Reshard/Pipeline.lean` for sub-block
byte-interval coverage and multi-sender push-stream accounting:
* `tpu_sync/kv_cache/reshard_receive_session.cc` (`ReshardReceiveSession::ValidatePlan`,
  `ReshardReceiveSession::ValidateReceiverCoverage`)
* `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc` (`ValidateSpanCoverage`,
  `EmitSpanSchedule` `expected_pushes` computation, `DefaultTransferStreams`,
  and `BuildPoolReshardPlan`)
* `tpu_sync/kv_cache/reshard_send_session.cc` (`ReshardSendSession::ExecutePush` /
  `ReshardSendSession::StartPoolPush` per-peer `(src_block_id, dst_block_id)`
  deduplication)
* `tpu_sync/core/pool_reshard_send_slots.cc` (`CountPoolReshardSendSlots`)
* `tpu_sync/transport/block_transport.cc` (`BlockTransport::AsyncPushShards`
  stream clamping `P = min(parallelism, num_blocks)`) and
  `tpu_sync/transport/block_transport_session.cc` (the per-uuid
  `BlockTransportServerPushSession::StartStream` /
  `BlockTransportServerPushSession::CompleteIncomingPush` plan-declared
  progress counter)

The coverage check is per plan: for a sub-plan of a windowed request
(`ReshardCoordinator::StartChain`), `BuildPoolReshardPlan` clips each tag's
declared spans to the sub-plan's window (`dst_window_offset_blocks` /
`dst_window_total_blocks` in `PlanRequest`) and `ValidateReceiverCoverage`
checks that sub-plan's own destination blocks.

File aliases used in the source map below:
`reshard_recv.cc` (`tpu_sync/kv_cache/reshard_receive_session.cc`),
`planner.{h,cc}` (`tpu_sync/kv_cache/reshard/pool_reshard_planner.{h,cc}`),
`reshard_send.cc` (`tpu_sync/kv_cache/reshard_send_session.cc`),
`slots.cc` (`tpu_sync/core/pool_reshard_send_slots.cc`),
`bt.cc` (`tpu_sync/transport/block_transport.cc`),
`bts.cc` (`tpu_sync/transport/block_transport_session.cc`),
`base.cc` (`tpu_sync/kv_cache/kv_cache_manager_base.cc`),
`reshard_recv_test.cc` (`tpu_sync/kv_cache/reshard_receive_session_test.cc`),
`pool_reshard_test.cc` (`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`),
`pool_reshard_send_slots_test.cc` (`tpu_sync/core/pool_reshard_send_slots_test.cc`),
`reshard_service_test.cc` (`tpu_sync/kv_cache/reshard/reshard_service_test.cc`),
`pool_reshard_planner_streams_test.cc` (`tpu_sync/kv_cache/reshard/pool_reshard_planner_streams_test.cc`).

What this module proves:
1. **Multi-sender stream count conservation (`stream_count_matches`,
   `senderStreams_pos_iff`).** Each sender `s` deduplicates its scheduled
   `(src_block_id, dst_block_id)` pairs for a pool group and opens
   `min(parallelism, |pairs_s|)` TCP push streams
   (`BlockTransport::AsyncPushShards` in `block_transport.cc`,
   `ReshardSendSession::StartPoolPush` in `reshard_send_session.cc`).
   `ReshardReceiveSession::ValidateReceiverCoverage`
   (`int64_t calculated_pushes = 0;` .. `if (calculated_pushes != view.expected_pushes)`
   in `reshard_receive_session.cc`) checks that the receiver's `expected_pushes`
   equals `∑_s min(parallelism, |pairs_s|)`, matching the planner
   (`BuildPoolReshardPlan` in `pool_reshard_planner.cc`).
2. **Exact sub-block byte reconstruction under out-of-order stream arrival
   (`checkContiguous_covers`, `checkContiguous_head_disjoint`,
   `partition_reconstructs_block`, `partition_reconstructs_perm`).**
   `ReshardReceiveSession::ValidateReceiverCoverage` sorts the logical live-byte
   intervals `[start_bytes, end_bytes)` targeting a destination block and checks
   that they partition `[0, extent)` without gaps or overlaps
   (`for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal)` in
   `reshard_receive_session.cc`). We prove that any interval list accepted by
   this check is pairwise disjoint and covers every byte `b < extent`, so
   landing the intervals in *any* network arrival order reconstructs
   `good extent` (`[kv 0, …, kv (extent - 1)]`).
3. **Refinement coupling (`CoveragePipeline.step_pipe`,
   `CoveragePipeline.reachable_safe`).** `CoveragePipeline` refines
   `Reshard.Pipeline` by replacing each pool's scalar stream counter with a
   queue of sub-block intervals and a byte-addressed staging buffer. Every
   `CoveragePipeline` step projects to a `Pipeline.step`, lifting all
   pipeline invariants together with byte-level destination reconstruction.

## Source map

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|------|-----------------|-------|
| `reshard_recv.cc` | `ReshardReceiveSession::ValidatePlan` | `:67-144` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` | `:146-349` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` (`const int64_t parallelism = plan.parallelism();`) | `:188` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` (`for (size_t ordinal = 0; ordinal < view.extents.size(); ++ordinal)`) | `:245-258` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` (`for (int64_t repeat = 0; repeat < entry.count(); ++repeat)` .. `state.pairs_by_sender[source_rank].insert`) | `:286-305` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` (`int64_t calculated_pushes = 0;` .. `if (calculated_pushes != view.expected_pushes)`) | `:314-319` |
| `reshard_recv.cc` | `ReshardReceiveSession::ValidateReceiverCoverage` (`std::sort(intervals.begin(), intervals.end());` .. `if (covered_until != view.extents[ordinal])`) | `:328-339` |
| `planner.h` | `*` (`inline constexpr char kTransferStreamsEnvVar[]`) | `:163` |
| `planner.cc` | `ValidateSpanCoverage` | `:350-487` |
| `planner.cc` | `EmitSpanSchedule` | `:503-607` |
| `planner.cc` | `DefaultTransferStreams` | `:611-628` |
| `planner.cc` | `BuildPoolReshardPlan` (`int64_t requested_parallelism = 0;` .. `if (requested_parallelism > topology_parallelism)`) | `:984-994` |
| `planner.cc` | `BuildPoolReshardPlan` (`expected_pushes_by_dst;` .. `if (expected_pushes_by_dst.at(dst_unit_id) <= 0)`) | `:1234-1246` |
| `reshard_send.cc` | `ReshardSendSession::ExecutePush` (`for (int32_t encoded_pool_idx : plan_.transfer_pool_indices())` .. `if (pool_src_block_ids.empty())`) | `:204-217` |
| `reshard_send.cc` | `ReshardSendSession::StartPoolPush` | `:267-349` |
| `slots.cc` | `CountPoolReshardSendSlots` | `:27-51` |
| `bts.cc` | `BlockTransportServerPushSession::StartStream` (`if (expected_size > 0)`) | `:423-427` |
| `bts.cc` | `BlockTransportServerPushSession::CompleteIncomingPush` | `:506-702` |
| `bt.cc` | `BlockTransport::AsyncPushShards` (`size_t num_blocks = src_block_ids.size();` .. `auto requests = BuildBlockRequests(`) | `:568-598` |
| `base.cc` | `KVCacheManagerBase::GetPoolPushProgressSpec` | `:2700-2743` |
| `base.cc` | `KVCacheManagerBase::ExpectedPushSenders` | `:2745-2762` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.ChecksEveryPoolSpanAndDestinationZeroCover` | `:227-255` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsCoverageGap` | `:278-289` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsCoverageOverlap` | `:291-302` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsShortCoverage` | `:304-314` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsNonPrefixExtents` | `:316-330` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsMissingExtents` | `:332-341` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsMissingParallelism` | `:343-352` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsExpectedPushMismatch` | `:354-363` |
| `reshard_recv_test.cc` | `ReshardReceiveSessionTest.RejectsWriteBeyondDeclaredLiveTail` | `:365-403` |
| `pool_reshard_test.cc` | `ExpectedPushSendersTest.CountsReceiverPlanSchedules` | `:222-269` |
| `pool_reshard_send_slots_test.cc` | `PoolReshardSendSlots.SingleDestinationIsPoolsTimesOnePeer` | `:49-55` |
| `pool_reshard_send_slots_test.cc` | `PoolReshardSendSlots.ReplicatedDestinationsIsPoolsTimesPeers` | `:57-65` |
| `pool_reshard_send_slots_test.cc` | `PoolReshardSendSlots.ShardedGroupCountsOnlyItsPeers` | `:67-76` |
| `pool_reshard_send_slots_test.cc` | `PoolReshardSendSlots.GroupWithoutEntriesContributesNoSlot` | `:78-83` |
| `pool_reshard_send_slots_test.cc` | `PoolReshardSendSlots.EmptyScheduleHasNoSlots` | `:85-89` |
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsShardedSpansRouteBytesPerReceiver` | `:1982-2047` |
| `reshard_service_test.cc` | `ReshardStackTest.ShardedDestinationWithoutCoverageFailsClosed` | `:2168-2181` |
| `reshard_service_test.cc` | `ReshardStackTest.ReplicatedAndRoutedSpansCombine` | `:2210-2242` |
| `pool_reshard_planner_streams_test.cc` | `DefaultTransferStreamsTest.UnsetEmptyOrZeroKeepsOneStreamPerSourceRank` | `:59-72` |
| `pool_reshard_planner_streams_test.cc` | `DefaultTransferStreamsTest.PositiveValueCapsStreamsWithinAdmission` | `:74-87` |
| `pool_reshard_planner_streams_test.cc` | `DefaultTransferStreamsTest.RejectsUnparsableValueNamingVariableAndValue` | `:89-95` |
| `pool_reshard_planner_streams_test.cc` | `DefaultTransferStreamsTest.RejectsNegativeValueNamingVariableAndValue` | `:97-103` |
| `reshard_service_test.cc` | `ReshardStackTest.UnsetTransferStreamsPlansOneStreamPerSourceRank` | `:2284-2304` |
| `reshard_service_test.cc` | `ReshardStackTest.TransferStreamsEnvCapsPlanStreamsAndExpectedPushes` | `:2306-2326` |
| `reshard_service_test.cc` | `ReshardStackTest.ExplicitPlanParallelismOverridesTransferStreamsEnv` | `:2328-2345` |
| `reshard_service_test.cc` | `ReshardStackTest.InvalidTransferStreamsEnvFailsThePlan` | `:2347-2362` |
<!-- source-map:end -->
-/

namespace TpuSyncVerify.Transfer.Reshard.Coverage

open TpuSyncVerify
open TpuSyncVerify.Transfer
open TpuSyncVerify.Transfer.Reshard
open TpuSyncVerify.Transfer.Reshard.Pipeline

/-! ## Part 1: Multi-sender `(src_block, dst_block)` pair deduplication and stream counts -/

/-- A logical sub-block range `[startByte, endByte)` inside a destination block's
live byte extent (`ReshardReceiveSession::ValidateReceiverCoverage`
`PhysicalLiveRangeToLogical` in `reshard_receive_session.cc`). -/
structure Interval where
  startByte : Nat
  endByte : Nat
  deriving Repr, DecidableEq

/-- One entry of a pool group's push schedule
(`ReshardReceiveSession::ValidateReceiverCoverage` in `reshard_receive_session.cc`,
`ReshardSendSession::StartPoolPush` in `reshard_send_session.cc`), attributed to
`sender` (`source_rank`). -/
structure ScheduleEntry where
  sender : Nat
  srcBlock : Nat
  dstBlock : Nat
  interval : Interval
  deriving Repr, DecidableEq

/-- Distinct `(srcBlock, dstBlock)` pairs scheduled for `sender`
(`ReshardSendSession::StartPoolPush` `block_pairs` in `reshard_send_session.cc`,
`ReshardReceiveSession::ValidateReceiverCoverage` `state.pairs_by_sender[source_rank].insert`
in `reshard_receive_session.cc`). -/
def senderPairs (entries : List ScheduleEntry) (sender : Nat) : List (Nat × Nat) :=
  ((entries.filter fun e => e.sender == sender).map fun e => (e.srcBlock, e.dstBlock)).eraseDups

/-- Number of TCP push streams opened by `sender` under `parallelism`:
`BlockTransport::AsyncPushShards` clamps `P = min(parallelism, num_blocks)`
(`block_transport.cc`), and a sender with zero pairs opens no streams
(`ReshardSendSession::ExecutePush` `if (pool_src_block_ids.empty())` in
`reshard_send_session.cc`). -/
def senderStreams (parallelism : Nat) (entries : List ScheduleEntry) (sender : Nat) : Nat :=
  Nat.min parallelism (senderPairs entries sender).length

/-- Total push streams recomputed across `senders` (`BuildPoolReshardPlan` in
`pool_reshard_planner.cc`, `ReshardReceiveSession::ValidateReceiverCoverage`
`int64_t calculated_pushes = 0;` in `reshard_receive_session.cc`). -/
def recomputedExpectedPushes (parallelism : Nat) (senders : List Nat)
    (entries : List ScheduleEntry) : Nat :=
  match senders with
  | [] => 0
  | s :: ss => senderStreams parallelism entries s + recomputedExpectedPushes parallelism ss entries

/-- Receiver push-count validation (`ReshardReceiveSession::ValidateReceiverCoverage`
`const int64_t parallelism = plan.parallelism();` and
`if (calculated_pushes != view.expected_pushes)` in `reshard_receive_session.cc`,
and `BuildPoolReshardPlan` in `pool_reshard_planner.cc`). -/
def validateExpectedPushes (parallelism : Nat) (senders : List Nat)
    (entries : List ScheduleEntry) (expectedPushes : Nat) : Bool :=
  decide (0 < parallelism) &&
  decide (0 < expectedPushes) &&
  decide (recomputedExpectedPushes parallelism senders entries = expectedPushes)

theorem senderStreams_eq_zero_of_no_pairs (parallelism : Nat) (entries : List ScheduleEntry)
    (sender : Nat) (h : senderPairs entries sender = []) :
    senderStreams parallelism entries sender = 0 := by
  simp [senderStreams, h]

theorem senderStreams_pos_iff {parallelism : Nat} {entries : List ScheduleEntry} {sender : Nat} :
    0 < senderStreams parallelism entries sender ↔
      0 < parallelism ∧ 0 < (senderPairs entries sender).length := by
  simp [senderStreams, Nat.lt_min]

theorem senderStreams_le_parallelism (parallelism : Nat) (entries : List ScheduleEntry)
    (sender : Nat) :
    senderStreams parallelism entries sender ≤ parallelism :=
  Nat.min_le_left _ _

theorem senderStreams_le_pairs (parallelism : Nat) (entries : List ScheduleEntry)
    (sender : Nat) :
    senderStreams parallelism entries sender ≤ (senderPairs entries sender).length :=
  Nat.min_le_right _ _

/-- When `validateExpectedPushes` succeeds, the sum of streams emitted by all
senders in `senders` equals the receiver's `expectedPushes`, and `0 < expectedPushes`
(`ReshardReceiveSession::ValidateReceiverCoverage`
`if (calculated_pushes != view.expected_pushes)` in `reshard_receive_session.cc`,
`BuildPoolReshardPlan` in `pool_reshard_planner.cc`). -/
theorem stream_count_matches {parallelism : Nat} {senders : List Nat}
    {entries : List ScheduleEntry} {expectedPushes : Nat}
    (h : validateExpectedPushes parallelism senders entries expectedPushes = true) :
    0 < parallelism ∧ 0 < expectedPushes ∧
      recomputedExpectedPushes parallelism senders entries = expectedPushes := by
  simp only [validateExpectedPushes, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-! ### The sender's slot count (`CountPoolReshardSendSlots`)

The sender sizes its session by `CountPoolReshardSendSlots(plan, schedule)`
(`tpu_sync/core/pool_reshard_send_slots.cc`): for every transferred pool
(`plan.transfer_pool_indices()`), the number of distinct `dst_peer`s among the
schedule entries whose `pool_group` is that pool's group
(`peers_by_group[entry.pool_group()].insert(entry.dst_peer())`), summed over the
pools; a pool whose group has no entry contributes nothing
(`if (it == peers_by_group.end()) continue;`). The per-pool terms are the
`peerCounts` the `Reshard/Send` model is initialised with, and the sum is its
push budget (`Send.init`, `sumPeers`). -/

/-- Distinct peers with at least one entry for pool group `g`; `entries` are
`(pool_group, dst_peer)` pairs. -/
def groupPeers (entries : List (Nat × Nat)) (g : Nat) : Nat :=
  ((entries.filter fun e => e.1 == g).map (·.2)).eraseDups.length

/-- The group of `pool` in `poolGroups` (`plan.pool_groups(g).pool_indices()`),
if any. -/
def poolGroupOf (poolGroups : List (List Nat)) (pool : Nat) : Option Nat :=
  poolGroups.findIdx? (·.contains pool)

/-- `CountPoolReshardSendSlots`. -/
def countPoolReshardSendSlots (transferPools : List Nat) (poolGroups : List (List Nat))
    (entries : List (Nat × Nat)) : Nat :=
  (transferPools.map fun pool =>
    match poolGroupOf poolGroups pool with
    | some g => groupPeers entries g
    | none => 0).foldl (· + ·) 0

/-- Replay of `PoolReshardSendSlots` (`tpu_sync/core/pool_reshard_send_slots_test.cc`);
peers `0` and `1` stand for `10.0.0.2:9400` and `10.0.0.2:9401`:
1. `PoolReshardSendSlots.SingleDestinationIsPoolsTimesOnePeer`: groups
   `{0,1,2}`, `{3}`, one peer in each: `3 + 1 = 4`.
2. `PoolReshardSendSlots.ReplicatedDestinationsIsPoolsTimesPeers`: groups
   `{0,1}`, `{2}`, two peers in each: `2 + 2 + 2 = 6`.
3. `PoolReshardSendSlots.ShardedGroupCountsOnlyItsPeers`: group `{0,1}` reaches
   both peers, group `{2,3}` one: `2 + 2 + 1 + 1 = 6`, not `4 * 2 = 8` (the
   over-count `Reshard/Send`'s mutant `initOvercountedSlots` plants).
4. `PoolReshardSendSlots.GroupWithoutEntriesContributesNoSlot`: groups `{0}`,
   `{1}`, only group `0` scheduled: `1`.
5. `PoolReshardSendSlots.EmptyScheduleHasNoSlots`: no entries: `0`. -/
theorem trace_pool_reshard_send_slots_test_parity :
    countPoolReshardSendSlots [0, 1, 2, 3] [[0, 1, 2], [3]] [(0, 0), (1, 0)] = 4 ∧
    countPoolReshardSendSlots [0, 1, 2] [[0, 1], [2]] [(0, 0), (1, 0), (0, 1), (1, 1)] = 6 ∧
    countPoolReshardSendSlots [0, 1, 2, 3] [[0, 1], [2, 3]] [(0, 0), (0, 1), (1, 1)] = 6 ∧
    countPoolReshardSendSlots [0, 1] [[0], [1]] [(0, 0)] = 1 ∧
    countPoolReshardSendSlots [0] [[0]] [] = 0 := by
  decide

/-! ## Part 2: Sub-block interval partition and byte-level block reconstruction -/

/-- Contiguous interval scan from `ReshardReceiveSession::ValidateReceiverCoverage`
(`for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal)` in
`reshard_receive_session.cc`): starting at `coveredUntil`, each interval
`[startByte, endByte)` must start at `coveredUntil` (`start_bytes ==
covered_until`, rejecting both overlaps `start_bytes < covered_until` and gaps
`start_bytes > covered_until`) and be non-empty (`startByte < endByte`, from
`size_bytes > 0` in `PhysicalLiveRangeToLogical`). Returns `some coveredUntil`
after scanning all intervals. -/
def checkContiguousIntervals (coveredUntil : Nat) : List Interval → Option Nat
  | [] => some coveredUntil
  | iv :: rest =>
    if iv.startByte = coveredUntil ∧ iv.startByte < iv.endByte then
      checkContiguousIntervals iv.endByte rest
    else
      none

/-- Full block coverage check (`ReshardReceiveSession::ValidateReceiverCoverage`
`for (size_t ordinal = 0; ordinal < view.extents.size(); ++ordinal)` and
`for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal)` in
`reshard_receive_session.cc`):
`0 < extent` and the sorted intervals partition `[0, extent)`. -/
def validBlockCoverage (extent : Nat) (intervals : List Interval) : Bool :=
  decide (0 < extent) && decide (checkContiguousIntervals 0 intervals = some extent)

/-- Write one logical byte interval `[iv.startByte, iv.endByte)` into a
destination block byte buffer (`BlockTransportServerPushSession::StartStream`
`ReadVExact(client_fd, ToIovec(chunks),` in `block_transport_session.cc`). Byte
`b` in range receives `.kv b`. -/
def writeInterval (iv : Interval) (buf : List Cell) : List Cell :=
  buf.mapIdx fun b c => if iv.startByte ≤ b ∧ b < iv.endByte then .kv b else c

/-- Write a sequence of intervals into `buf` in arrival order. -/
def writeIntervals : List Interval → List Cell → List Cell
  | [], buf => buf
  | iv :: rest, buf => writeIntervals rest (writeInterval iv buf)

@[simp] theorem length_writeInterval (iv : Interval) (buf : List Cell) :
    (writeInterval iv buf).length = buf.length := by
  simp [writeInterval]

@[simp] theorem length_writeIntervals (ivs : List Interval) (buf : List Cell) :
    (writeIntervals ivs buf).length = buf.length := by
  induction ivs generalizing buf with
  | nil => rfl
  | cons iv rest ih => simp [writeIntervals, ih]

theorem getElem?_writeInterval {iv : Interval} {buf : List Cell} {b : Nat}
    (hb : b < buf.length) :
    (writeInterval iv buf)[b]? =
      if iv.startByte ≤ b ∧ b < iv.endByte then some (.kv b) else buf[b]? := by
  simp only [writeInterval, List.getElem?_mapIdx, List.getElem?_eq_getElem hb, Option.map_some]
  split <;> rfl

/-- Writing two intervals into a block buffer commutes, so network arrival order
across senders does not affect the resulting staging bytes. -/
theorem writeInterval_comm (iv₁ iv₂ : Interval) (buf : List Cell) :
    writeInterval iv₁ (writeInterval iv₂ buf) = writeInterval iv₂ (writeInterval iv₁ buf) := by
  apply List.ext_getElem?
  intro b
  by_cases hb : b < buf.length
  · have hb₁ : b < (writeInterval iv₂ buf).length := by simp [hb]
    have hb₂ : b < (writeInterval iv₁ buf).length := by simp [hb]
    rw [getElem?_writeInterval hb₁, getElem?_writeInterval hb,
        getElem?_writeInterval hb₂, getElem?_writeInterval hb]
    by_cases h1 : iv₁.startByte ≤ b ∧ b < iv₁.endByte <;>
      by_cases h2 : iv₂.startByte ≤ b ∧ b < iv₂.endByte <;>
      simp [h1, h2]
  · have h₁ : (writeInterval iv₁ (writeInterval iv₂ buf))[b]? = none :=
      List.getElem?_eq_none (by simp; omega)
    have h₂ : (writeInterval iv₂ (writeInterval iv₁ buf))[b]? = none :=
      List.getElem?_eq_none (by simp; omega)
    rw [h₁, h₂]

theorem getElem?_writeIntervals {ivs : List Interval} {buf : List Cell} {b : Nat}
    (hb : b < buf.length) :
    (writeIntervals ivs buf)[b]? =
      if ivs.any (fun iv => decide (iv.startByte ≤ b ∧ b < iv.endByte)) then
        some (.kv b)
      else
        buf[b]? := by
  induction ivs generalizing buf with
  | nil => simp [writeIntervals]
  | cons iv rest ih =>
    simp only [writeIntervals, List.any_cons]
    rw [ih (by simp [hb]), getElem?_writeInterval hb]
    by_cases h1 : iv.startByte ≤ b ∧ b < iv.endByte <;> simp [h1]

theorem checkContiguous_le {cur ext : Nat} {ivs : List Interval}
    (h : checkContiguousIntervals cur ivs = some ext) :
    cur ≤ ext := by
  induction ivs generalizing cur with
  | nil =>
    simp only [checkContiguousIntervals, Option.some.injEq] at h
    omega
  | cons iv rest ih =>
    simp only [checkContiguousIntervals] at h
    split at h <;> try cases h
    rename_i hcond
    have := ih h
    omega

/-- Key characterization of `checkContiguousIntervals`: if the scan from `cur`
succeeds with `some ext`, then the intervals in `ivs` cover `[cur, ext)`
exactly. -/
theorem checkContiguous_covers {cur ext : Nat} {ivs : List Interval}
    (h : checkContiguousIntervals cur ivs = some ext) (b : Nat) :
    ivs.any (fun iv => decide (iv.startByte ≤ b ∧ b < iv.endByte)) = true ↔
      cur ≤ b ∧ b < ext := by
  induction ivs generalizing cur with
  | nil =>
    simp only [checkContiguousIntervals, Option.some.injEq] at h
    subst h; simp
  | cons iv rest ih =>
    simp only [checkContiguousIntervals] at h
    split at h <;> try cases h
    rename_i hcond
    obtain ⟨hstart, hlt⟩ := hcond
    have hle := checkContiguous_le h
    simp only [List.any_cons, Bool.or_eq_true, decide_eq_true_eq, ih h]
    omega

/-- Any two intervals in a list accepted by `checkContiguousIntervals` are
disjoint: the head interval lies in `[cur, iv.endByte)` while every subsequent
interval lies in `[iv.endByte, ext)`. -/
theorem checkContiguous_head_disjoint {cur ext : Nat} {iv : Interval} {rest : List Interval}
    (h : checkContiguousIntervals cur (iv :: rest) = some ext)
    {iv' : Interval} (hmem : iv' ∈ rest) {b : Nat}
    (hb₁ : iv.startByte ≤ b ∧ b < iv.endByte)
    (hb₂ : iv'.startByte ≤ b ∧ b < iv'.endByte) : False := by
  simp only [checkContiguousIntervals] at h
  split at h <;> try cases h
  rename_i hcond
  have hany : rest.any (fun x => decide (x.startByte ≤ b ∧ b < x.endByte)) = true :=
    List.any_eq_true.mpr ⟨iv', hmem, decide_eq_true_eq.mpr hb₂⟩
  have ⟨hge, _⟩ := (checkContiguous_covers h b).mp hany
  omega

/-- Main block reconstruction theorem: if `validBlockCoverage extent ivs = true`
and `ivs'` is any arrival order containing the same intervals as `ivs`, then
writing `ivs'` into a blank buffer of length `extent` produces `good extent`
(`[kv 0, kv 1, …, kv (extent - 1)]`). -/
theorem partition_reconstructs_perm {extent : Nat} {ivs ivs' : List Interval}
    (hvalid : validBlockCoverage extent ivs = true)
    (hperm : ∀ iv, iv ∈ ivs' ↔ iv ∈ ivs) :
    writeIntervals ivs' (List.replicate extent .blank) = good extent := by
  simp only [validBlockCoverage, Bool.and_eq_true, decide_eq_true_eq] at hvalid
  obtain ⟨_, hcontig⟩ := hvalid
  apply eq_good_of_getElem?
  · simp
  · intro b hb
    have hlen : b < (List.replicate extent Cell.blank).length := by simp [hb]
    rw [getElem?_writeIntervals hlen]
    have hany_iff : ivs'.any (fun iv => decide (iv.startByte ≤ b ∧ b < iv.endByte)) = true ↔
        ivs.any (fun iv => decide (iv.startByte ≤ b ∧ b < iv.endByte)) = true := by
      simp only [List.any_eq_true, hperm]
    have hcov : ivs.any (fun iv => decide (iv.startByte ≤ b ∧ b < iv.endByte)) = true :=
      (checkContiguous_covers hcontig b).mpr ⟨Nat.zero_le b, hb⟩
    rw [hany_iff.mpr hcov]
    simp

/-- Special case when intervals arrive in their canonical sorted order. -/
theorem partition_reconstructs_block {extent : Nat} {ivs : List Interval}
    (hvalid : validBlockCoverage extent ivs = true) :
    writeIntervals ivs (List.replicate extent .blank) = good extent :=
  partition_reconstructs_perm hvalid (fun _ => Iff.rfl)

/-! ## Part 3: Refinement coupling (`CoveragePipeline.step_pipe`) over `Reshard.Pipeline` -/

/-- A pool's byte-coverage state in `CoveragePipeline`:
* `extent` is the destination byte extent for the pool;
* `planIntervals` is the validated sorted interval list of length `peerCounts[p]`;
* `landedIntervals` is the prefix/subset of intervals landed so far;
* `byteStaging` is the receiver's byte-level host staging buffer for this pool. -/
structure PoolCoverage where
  extent : Nat
  planIntervals : List Interval
  landedIntervals : List Interval := []
  byteStaging : List Cell
  deriving Repr, DecidableEq

/-- `CoveragePipeline` refines `Reshard.Pipeline` by tracking sub-block byte
intervals and byte-level destination staging alongside the abstract pipeline
state `pipe`. -/
structure CoveragePipeline where
  pipe : Pipeline
  pools : List PoolCoverage
  deriving Repr, DecidableEq

namespace CoveragePipeline

def validSchedule (ranks : List Nat) (specs : List (Nat × List Interval)) : Bool :=
  let peerCounts := specs.map fun (_, ivs) => ivs.length
  validConfig ranks peerCounts &&
  specs.all fun (ext, ivs) => validBlockCoverage ext ivs

def init (ranks : List Nat) (specs : List (Nat × List Interval)) : CoveragePipeline :=
  let peerCounts := specs.map fun (_, ivs) => ivs.length
  { pipe := Pipeline.init ranks peerCounts
    pools := specs.map fun (ext, ivs) =>
      { extent := ext
        planIntervals := ivs
        landedIntervals := []
        byteStaging := List.replicate ext .blank } }

def step (s : CoveragePipeline) (e : Ev) : Option CoveragePipeline :=
  match e with
  | .landStream p =>
    match s.pipe.landStream p, s.pools[p]? with
    | some pipe', some pc =>
      let k := pc.landedIntervals.length
      match pc.planIntervals[k]? with
      | some iv =>
        let pc' : PoolCoverage :=
          { pc with landedIntervals := pc.landedIntervals ++ [iv]
                    byteStaging := writeInterval iv pc.byteStaging }
        some { pipe := pipe', pools := s.pools.set p pc' }
      | none => none
    | _, _ => none
  | .reseatDstStaging =>
    (s.pipe.reseatDstStaging).map fun pipe' =>
      { pipe := pipe'
        pools := s.pools.map fun pc =>
          { pc with byteStaging := List.replicate pc.extent .junk } }
  | e =>
    (Pipeline.step s.pipe e).map fun pipe' => { s with pipe := pipe' }

def sys (ranks : List Nat) (specs : List (Nat × List Interval)) :
    System CoveragePipeline Ev :=
  ⟨init ranks specs, step⟩

/-- Simulation theorem (`step_pipe`): every `CoveragePipeline` transition
projects directly to a `Reshard.Pipeline` transition on `s.pipe`. -/
theorem step_pipe {s s' : CoveragePipeline} {e : Ev}
    (hs : step s e = some s') :
    Pipeline.step s.pipe e = some s'.pipe := by
  cases e with
  | landStream p =>
    simp only [step] at hs
    split at hs <;> try cases hs
    rename_i pipe' pc hland _
    split at hs <;> simp at hs; subst hs
    exact hland
  | reseatDstStaging =>
    simp only [step, Option.map_eq_some_iff] at hs
    obtain ⟨pipe', hres, rfl⟩ := hs
    exact hres
  | armRecv | send _ | recv _ | reclaim | reseatSrcStaging =>
    simp only [step, Option.map_eq_some_iff] at hs
    obtain ⟨pipe', hstep, rfl⟩ := hs
    exact hstep

theorem reachable_pipe {ranks : List Nat} {specs : List (Nat × List Interval)}
    {s : CoveragePipeline}
    (h : (sys ranks specs).Reachable s) :
    (Pipeline.sys ranks (specs.map fun (_, ivs) => ivs.length)).Reachable s.pipe := by
  induction h with
  | init => exact .init
  | step _ hs ih => exact .step ih (step_pipe hs)

/-- Every state reachable in `CoveragePipeline` satisfies all pipeline
safety properties (`PublicationCorrect`, `AliasedPageCorrect`, `DstHbmSafe`,
`SrcHbmSafe`, `StagingSafe`, `StreamAccountingSafe`, `NoUnarmedPush`,
`NoOpLeak`). -/
theorem reachable_safe {ranks : List Nat} {specs : List (Nat × List Interval)}
    {s : CoveragePipeline}
    (hvalid : validSchedule ranks specs = true)
    (h : (sys ranks specs).Reachable s) :
    Pipeline.Safe s.pipe := by
  simp only [validSchedule, Bool.and_eq_true] at hvalid
  exact Pipeline.reachable_safe hvalid.1 (reachable_pipe h)

end CoveragePipeline

/-! ## Part 4: Concrete witness traces and mutants -/

/-- TP=4 → TP=2 sub-block merge witness: 2 senders (`rank 0`, `rank 1`) push
sub-block slices `[0, 16)` and `[16, 32)` into a 32-byte destination block.
Even when `[16, 32)` arrives on the wire before `[0, 16)`, the reconstructed
staging buffer equals `good 32` (`[kv 0, …, kv 31]`). -/
theorem trace_tp4_to_tp2_subblock_merge :
    let iv0 : Interval := ⟨0, 16⟩
    let iv1 : Interval := ⟨16, 32⟩
    validBlockCoverage 32 [iv0, iv1] = true ∧
    writeIntervals [iv1, iv0] (List.replicate 32 .blank) = good 32 := by
  decide

/-- Multi-sender stream count witness: sender 0 has 3 `(src, dst)` pairs and
sender 1 has 1 pair under `parallelism = 2`, yielding `min(2, 3) + min(2, 1) = 3`
expected push streams (`BuildPoolReshardPlan` in `pool_reshard_planner.cc`). -/
theorem trace_multi_sender_stream_count :
    let entries : List ScheduleEntry :=
      [ ⟨0, 10, 0, ⟨0, 8⟩⟩
      , ⟨0, 11, 0, ⟨8, 16⟩⟩
      , ⟨0, 12, 1, ⟨0, 16⟩⟩
      , ⟨1, 20, 1, ⟨16, 32⟩⟩ ]
    validateExpectedPushes 2 [0, 1] entries 3 = true ∧
    senderStreams 2 entries 0 = 2 ∧
    senderStreams 2 entries 1 = 1 := by
  decide

/-- End-to-end `CoveragePipeline` witness: 2 pools on an aliased page
(`ranks = [0, 1]`), where pool 0 merges 2 sub-block intervals `[0, 4)` and
`[4, 8)` and pool 1 lands `[0, 8)` first. Both `s.pipe.dstHbm = good 2` and
every pool's byte staging buffer equals `good 8`. -/
def coverageTraceEvents : List Ev :=
  [ .send (.d2hIssue 0 true), .send (.d2hIssue 1 true), .send .execFinish
  , .send (.d2hDone 0 true), .send (.d2hDone 1 true)
  , .send (.pushDone 1 true), .send (.pushDone 0 true), .send (.pushDone 0 true)
  , .recv .pushBegin, .landStream 1, .recv (.onPoolReceived 1), .recv (.pushEnd true)
  , .recv .pushBegin, .landStream 0, .recv (.pushEnd true)
  , .recv .pushBegin, .landStream 0, .recv (.onPoolReceived 0), .recv (.pushEnd true)
  , .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true)
  , .recv (.h2dIssue 1 true), .recv (.h2dReady 1), .recv (.h2dDone 1 true)
  , .send .publish, .recv .publish ]

theorem trace_coverage_pipeline_end_to_end :
    let specs : List (Nat × List Interval) :=
      [(8, [⟨0, 4⟩, ⟨4, 8⟩]), (8, [⟨0, 8⟩])]
    ((CoveragePipeline.sys [0, 1] specs).run coverageTraceEvents).map
      (fun s => (s.pipe.recv.published, s.pipe.dstHbm,
                 s.pools.map (·.byteStaging))) =
      some (some true, good 2, [good 8, good 8]) := by
  decide

/-- Mutant 1: a gap in the sub-block schedule (`[0, 12)` then `[16, 32)`) is
rejected by `validBlockCoverage` and leaves bytes `12..15` as `.blank`. -/
theorem mutant_gap_in_coverage_leaves_unwritten_byte :
    let ivs : List Interval := [⟨0, 12⟩, ⟨16, 32⟩]
    validBlockCoverage 32 ivs = false ∧
    (writeIntervals ivs (List.replicate 32 .blank))[12]? = some .blank := by
  decide

/-- Mutant 2: overlapping sub-block intervals (`[0, 20)` and `[16, 32)`) are
rejected by `validBlockCoverage`
(`ReshardReceiveSession::ValidateReceiverCoverage`
`if (start_bytes != covered_until)` in `reshard_receive_session.cc`). -/
theorem mutant_overlap_in_coverage_rejected :
    validBlockCoverage 32 [⟨0, 20⟩, ⟨16, 32⟩] = false := by
  decide

/-- Mutant 3: declaring `expected_pushes = 3` (summing the 3 raw pairs from
sender `0` instead of `min(parallelism, |pairs_s|) = min(2, 3) = 2`) is
rejected by `validateExpectedPushes`
(`ReshardReceiveSession::ValidateReceiverCoverage`
`if (calculated_pushes != view.expected_pushes)` in
`reshard_receive_session.cc`). -/
theorem mutant_mismatched_expected_pushes_rejected :
    let entries : List ScheduleEntry :=
      [ ⟨0, 10, 0, ⟨0, 8⟩⟩
      , ⟨0, 11, 0, ⟨8, 16⟩⟩
      , ⟨0, 12, 1, ⟨0, 16⟩⟩ ]
    validateExpectedPushes 2 [0] entries 3 = false ∧
    validateExpectedPushes 2 [0] entries 2 = true := by
  decide

/-- Group destination-extent prefix validation from
`ReshardReceiveSession::ValidateReceiverCoverage`
(`for (size_t ordinal = 0; ordinal < view.extents.size(); ++ordinal)` in
`reshard_receive_session.cc`):
every non-last destination block must cover the full `liveBytes` (`extent == liveBytes`),
and the final destination block must satisfy `0 < extent ≤ liveBytes`. -/
def validGroupExtents (liveBytes : Nat) : List Nat → Bool
  | [] => false
  | [last] => decide (0 < last ∧ last ≤ liveBytes)
  | e :: rest => decide (e = liveBytes) && validGroupExtents liveBytes rest

/-- `KVCacheManagerBase::ExpectedPushSenders`
(`tpu_sync/kv_cache/kv_cache_manager_base.cc`):
returns `some schedules.length` when the active plan is a block-addressed
receiver plan (`!isSender` and `pool_groups_size() == 0`) with non-empty
`shard_push_schedules`, and `none` for sender plans, pool-grouped plans (which
use `KVCacheManagerBase::GetPoolPushProgressSpec`), bare plans, or after
`UnregisterActivePlan`. -/
def expectedPushSenders (isSender : Bool) (schedules : List (Nat × List ScheduleEntry)) :
    Option Nat :=
  if isSender || schedules.isEmpty then none
  else some schedules.length

/-- Replay of `ReshardReceiveSessionTest` coverage and push-count unit tests
(`tpu_sync/kv_cache/reshard_receive_session_test.cc`):
1. `ReshardReceiveSessionTest.ChecksEveryPoolSpanAndDestinationZeroCover`
   (zero intervals `[]` on extent `16` rejected).
2. `ReshardReceiveSessionTest.RejectsCoverageGap` (extent `48` with `[0, 16)`
   and `[32, 48)`).
3. `ReshardReceiveSessionTest.RejectsCoverageOverlap` (extent `24` with `[0, 16)`
   and `[8, 24)`).
4. `ReshardReceiveSessionTest.RejectsShortCoverage` (extent `24` with `[0, 16)`).
5. `ReshardReceiveSessionTest.RejectsNonPrefixExtents` (modeled with
   `liveBytes = 64`: `[32, 64]` has a partially filled non-last block; `[80]`
   exceeds `liveBytes`; `[64, 32]` is a valid prefix).
6. `ReshardReceiveSessionTest.RejectsMissingExtents` (modeled with extent `0`).
7. `ReshardReceiveSessionTest.RejectsMissingParallelism` (`parallelism = 0`).
8. `ReshardReceiveSessionTest.RejectsExpectedPushMismatch` (1 schedule entry
   with `expected_pushes = 2`).
9. `ReshardReceiveSessionTest.RejectsWriteBeyondDeclaredLiveTail` (declared
   `extent = 40` with logical intervals `[0, 32)` and `[32, 48)` rejected, and
   `[0, 32)` and `[32, 40)` accepted). -/
theorem trace_reshard_receive_coverage_test_parity :
    let e1 : List ScheduleEntry := [⟨0, 3, 1, ⟨0, 16⟩⟩]
    validBlockCoverage 16 [] = false ∧
    validBlockCoverage 48 [⟨0, 16⟩, ⟨32, 48⟩] = false ∧
    validBlockCoverage 24 [⟨0, 16⟩, ⟨8, 24⟩] = false ∧
    validBlockCoverage 24 [⟨0, 16⟩] = false ∧
    validGroupExtents 64 [32, 64] = false ∧
    validGroupExtents 64 [80] = false ∧
    validGroupExtents 64 [64, 32] = true ∧
    validBlockCoverage 0 [⟨0, 16⟩] = false ∧
    validateExpectedPushes 0 [0] e1 1 = false ∧
    validateExpectedPushes 2 [0] e1 2 = false ∧
    validateExpectedPushes 2 [0] e1 1 = true ∧
    validBlockCoverage 40 [⟨0, 32⟩, ⟨32, 48⟩] = false ∧
    validBlockCoverage 40 [⟨0, 32⟩, ⟨32, 40⟩] = true := by
  decide

/-- Replay of `ExpectedPushSendersTest.CountsReceiverPlanSchedules`
(`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`):
- Receiver plan (`isSender = false`) with 2 source schedules (`ranks 4 and 5`)
  reports `some 2`.
- Sender plan (`isSender = true`) reports `none`.
- Bare plan (`schedules = []`) reports `none`. -/
theorem trace_expected_push_senders_test_parity :
    let sched4 : List ScheduleEntry := [⟨4, 0, 1, ⟨0, 16⟩⟩]
    let sched5 : List ScheduleEntry := [⟨5, 0, 2, ⟨0, 16⟩⟩]
    expectedPushSenders false [(4, sched4), (5, sched5)] = some 2 ∧
    expectedPushSenders true [(0, sched4)] = none ∧
    expectedPushSenders false [] = none := by
  decide

/-- Replay of the per-destination routing tests of `ReshardStackTest`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`), as each armed receiver
sees them: two source ranks (`topology_parallelism = 2`), two destination units
with 512-byte blocks, and the planner's `ValidateSpanCoverage` /
`expected_pushes` reproduced by `validBlockCoverage` /
`validateExpectedPushes` on the receiver (`ReshardReceiveSession::ValidateReceiverCoverage`
is the same check on the plan it is handed):
1. `ReshardStackTest.TwoDestinationsShardedSpansRouteBytesPerReceiver`: rank 0
   owns source block 3 → destination block 0, rank 1 source block 5 →
   destination block 1, each destination taking its own 512-byte half. Each
   receiver's plan carries one entry per sender (`entries_size() == 1` in both
   `shard_push_schedules`), `expected_pushes == 2`, and
   `dst_expected_extent_bytes == {512, 512}`, every block fully covered.
2. `ReshardStackTest.ReplicatedAndRoutedSpansCombine`: rank 0's block is
   replicated to both destinations and rank 1's is sharded; each receiver again
   sees two entries and `expected_pushes == 2`.
3. `ReshardStackTest.ShardedDestinationWithoutCoverageFailsClosed`: everything
   is routed to destination 0, so destination 1 has no bytes ("has no declared
   coverage"): an empty interval list covers nothing and recomputes zero pushes,
   and the plan is refused before any receiver is armed
   (`transport_.calls_.empty()`). -/
theorem trace_reshard_stack_routing_test_parity :
    let sharded : List ScheduleEntry := [⟨0, 3, 0, ⟨0, 512⟩⟩, ⟨1, 5, 1, ⟨0, 512⟩⟩]
    let mixed : List ScheduleEntry := [⟨0, 0, 0, ⟨0, 512⟩⟩, ⟨1, 1, 1, ⟨0, 512⟩⟩]
    (senderPairs sharded 0).length = 1 ∧ (senderPairs sharded 1).length = 1 ∧
    validateExpectedPushes 2 [0, 1] sharded 2 = true ∧
    validGroupExtents 512 [512, 512] = true ∧
    validBlockCoverage 512 [⟨0, 512⟩] = true ∧
    validateExpectedPushes 2 [0, 1] mixed 2 = true ∧
    validBlockCoverage 512 [] = false ∧
    validateExpectedPushes 2 [0, 1] [] 0 = false := by
  decide

/-- Parsed `TPU_RAIDEN_TRANSFER_STREAMS` environment variable
(`kTransferStreamsEnvVar` in `tpu_sync/kv_cache/reshard/pool_reshard_planner.h`,
`DefaultTransferStreams` in `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`). -/
inductive TransferStreamsEnv where
  | unsetOrEmpty
  | intVal (n : Int)
  | unparsable
  deriving Repr, DecidableEq

/-- `DefaultTransferStreams(topology_parallelism)`
(`tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`):
- Unset, empty, or `0` returns `some topologyParallelism`
  (`if (raw == nullptr || raw[0] == '\0')` / `if (streams == 0)`).
- A positive integer `n` returns `some (min n topologyParallelism)`
  (`if (streams > topology_parallelism)`).
- Unparsable or negative values return `none` (`InvalidArgumentError`,
  `if (!absl::SimpleAtoi(raw, &streams) || streams < 0)`). -/
def defaultTransferStreams (topologyParallelism : Nat) : TransferStreamsEnv → Option Nat
  | .unsetOrEmpty => some topologyParallelism
  | .intVal n =>
    if n < 0 then none
    else if n == 0 then some topologyParallelism
    else some (Nat.min n.toNat topologyParallelism)
  | .unparsable => none

/-- Plan parallelism resolution in `BuildPoolReshardPlan`
(`int64_t requested_parallelism = 0;` .. `if (requested_parallelism > topology_parallelism)`
in `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`):
an explicit `request.parallelism` overrides `TPU_RAIDEN_TRANSFER_STREAMS` and must
lie in `(0, topologyParallelism]`; otherwise `DefaultTransferStreams` is used. -/
def resolvePlanParallelism (topologyParallelism : Nat) (explicit : Option Nat)
    (env : TransferStreamsEnv) : Option Nat :=
  let resolved :=
    match explicit with
    | some p => some p
    | none => defaultTransferStreams topologyParallelism env
  match resolved with
  | some p =>
    if 0 < p ∧ p ≤ topologyParallelism then some p else none
  | none => none

/-- Replay of `DefaultTransferStreamsTest`
(`tpu_sync/kv_cache/reshard/pool_reshard_planner_streams_test.cc`) and
`ReshardStackTest` stream-count tests
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`):
1. `DefaultTransferStreamsTest.UnsetEmptyOrZeroKeepsOneStreamPerSourceRank`:
   unset/empty or `0` on `topology_parallelism = 8` returns `some 8`.
2. `DefaultTransferStreamsTest.PositiveValueCapsStreamsWithinAdmission`:
   `"1"` → `some 1`, `"4"` → `some 4`, and `"16"` clamps to `some 8`.
3. `DefaultTransferStreamsTest.RejectsUnparsableValueNamingVariableAndValue` and
   `DefaultTransferStreamsTest.RejectsNegativeValueNamingVariableAndValue`:
   `"many"` and `"-2"` return `none`.
4. `ReshardStackTest.UnsetTransferStreamsPlansOneStreamPerSourceRank`,
   `ReshardStackTest.TransferStreamsEnvCapsPlanStreamsAndExpectedPushes`,
   `ReshardStackTest.ExplicitPlanParallelismOverridesTransferStreamsEnv`, and
   `ReshardStackTest.InvalidTransferStreamsEnvFailsThePlan`: with 2 source ranks
   (`topology_parallelism = 2`) and 2 block pairs per rank, unset env plans
   `parallelism = 2` and `expected_pushes = 4`; `TPU_RAIDEN_TRANSFER_STREAMS=1`
   caps `parallelism = 1` and `expected_pushes = 2`; explicit `request.parallelism = 2`
   overrides `TPU_RAIDEN_TRANSFER_STREAMS=1` back to `parallelism = 2` and
   `expected_pushes = 4`; and `"many"` fails the plan (`none`). -/
theorem trace_default_transfer_streams_test_parity :
    defaultTransferStreams 8 .unsetOrEmpty = some 8 ∧
    defaultTransferStreams 8 (.intVal 0) = some 8 ∧
    defaultTransferStreams 8 (.intVal 1) = some 1 ∧
    defaultTransferStreams 8 (.intVal 4) = some 4 ∧
    defaultTransferStreams 8 (.intVal 16) = some 8 ∧
    defaultTransferStreams 8 .unparsable = none ∧
    defaultTransferStreams 8 (.intVal (-2)) = none ∧
    let twoRanksTwoPairs : List ScheduleEntry :=
      [ ⟨0, 0, 7, ⟨0, 1024⟩⟩, ⟨0, 1, 8, ⟨0, 1024⟩⟩
      , ⟨1, 0, 9, ⟨0, 1024⟩⟩, ⟨1, 1, 10, ⟨0, 1024⟩⟩ ]
    (resolvePlanParallelism 2 none .unsetOrEmpty).map
      (fun p => (p, recomputedExpectedPushes p [0, 1] twoRanksTwoPairs)) = some (2, 4) ∧
    (resolvePlanParallelism 2 none (.intVal 1)).map
      (fun p => (p, recomputedExpectedPushes p [0, 1] twoRanksTwoPairs)) = some (1, 2) ∧
    (resolvePlanParallelism 2 (some 2) (.intVal 1)).map
      (fun p => (p, recomputedExpectedPushes p [0, 1] twoRanksTwoPairs)) = some (2, 4) ∧
    resolvePlanParallelism 2 none .unparsable = none := by
  decide

end TpuSyncVerify.Transfer.Reshard.Coverage

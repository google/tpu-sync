import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Reshard.PoolStaging.Invariant

/-! # `PoolStaging`: concrete `decide` traces and mutants

Part of `TpuSyncVerify.Transfer.Reshard.PoolStaging`. -/

namespace TpuSyncVerify.Transfer.Reshard.PoolStaging

open TpuSyncVerify

/-! ## Part V — Concrete `decide` Traces (`pool_staging_test.cc` Parity) and Mutants -/

structure BoundedScenarioSummary where
  freeAfterA2 : Nat
  slot5 : Option Nat
  slot9 : Option Nat
  slot11 : Option Nat
  slot4 : Option Nat
  uuid8WhileFull : Bool
  uuid9TooLarge : Bool
  freeAfterRelease : Nat
  freeAfterUuid8 : Nat
  uuid8Slot1 : Option Nat
  uuid8Slot2 : Option Nat
  deriving DecidableEq, Repr

/-- Helper executing the `PoolStagingTest.BoundedArenaLeasesAndAddressing` scenario. -/
def runBoundedArenaScenario : Option BoundedScenarioSummary :=
  match (Arena.init 4).acquireLease 7 [5, 9] with
  | none => none
  | some a1 =>
    match a1.acquireLease 7 [9, 11] with
    | none => none
    | some a2 =>
      let a3 := a2.releaseLease 7
      match a3.acquireLease 8 [1, 2] with
      | none => none
      | some a4 =>
        some { freeAfterA2 := a2.freeSlots.length
             , slot5 := a2.resolveBlockSlot 7 5
             , slot9 := a2.resolveBlockSlot 7 9
             , slot11 := a2.resolveBlockSlot 7 11
             , slot4 := a2.resolveBlockSlot 7 4
             , uuid8WhileFull := (a2.acquireLease 8 [1, 2]).isNone
             , uuid9TooLarge := (a2.acquireLease 9 [1, 2, 3, 4, 6]).isNone
             , freeAfterRelease := a3.freeSlots.length
             , freeAfterUuid8 := a4.freeSlots.length
             , uuid8Slot1 := a4.resolveBlockSlot 8 1
             , uuid8Slot2 := a4.resolveBlockSlot 8 2 }

/-- Replay of `PoolStagingTest.BoundedArenaLeasesAndAddressing`:
1. 4-slot bounded arena (`Arena.init 4`).
2. `uuid = 7` leases blocks `[5, 9]` → mapped to slots `0` and `1` (2 free slots).
3. `uuid = 7` re-leases `[9, 11]` → block `9` stays at slot `1` (idempotent),
   block `11` takes slot `2` (1 free slot left).
4. Unleased block `4` resolves to `none`.
5. `uuid = 8` requesting `[1, 2]` (2 slots > 1 free) fails (`none`,
   `ResourceExhausted`).
6. `uuid = 9` requesting 5 blocks `[1, 2, 3, 4, 6]` (> 4 total slots) fails
   (`none`, `ResourceExhausted`).
7. Releasing `uuid = 7` returns all 3 leased slots (`[2, 1, 0, 3]`, 4 free slots),
   after which `uuid = 8` requesting `[1, 2]` succeeds with 2 free slots remaining. -/
theorem trace_bounded_arena_leases_and_addressing :
    runBoundedArenaScenario =
      some ⟨1, some 0, some 1, some 2, none, true, true, 4, 2, some 2, some 1⟩ := by
  decide

/-- Replay of `PoolStagingTest.SmallPoolIdentitySharedStorageAndStrideMismatch`:
Two pools (`gdn.conv` and `gdn.ssm`) sharing storage 0 acquire leases under the
same `uuid = 42` for overlapping block sets `[10, 20]` and `[10, 20]`. Because
`AcquirePoolStagingLease` is idempotent per `uuid`, the second pool's call
consumes 0 additional slots and shares the exact same page-slot mapping. -/
theorem trace_shared_storage_pools_share_lease :
    ((Arena.init 3).acquireLease 42 [10, 20]).map
      (fun a1 =>
        ( a1.acquireLease 42 [10, 20] == some a1
        , a1.freeSlots.length
        , a1.resolveBlockSlot 42 10
        , a1.resolveBlockSlot 42 20 )) =
      some (true, 1, some 0, some 1) := by
  decide

/-- Bounded vs. full-mirror storage configuration
(`KVCacheManagerBase::ConfigurePoolStaging` in `base.cc`):
allocates a bounded arena of `blocksPerLease * leases` slots (`some numSlots`)
if and only if `0 < blocksPerLease`, `0 < leases`, all pools sharing the
storage have equal `block_stride_bytes` (`stridesEqual`), and
`blocksPerLease * leases < maxPoolBlocks`; otherwise falls back to full-mirror
identity addressing (`none`). -/
def configureStorageBounded (blocksPerLease leases maxPoolBlocks : Nat)
    (stridesEqual : Bool) : Option Nat :=
  let numSlots := blocksPerLease * leases
  if 0 < blocksPerLease && 0 < leases && stridesEqual && numSlots < maxPoolBlocks then
    some numSlots
  else
    none

/-- Reported staging capacity (`KVCacheManagerBase::PoolHostBaseAddrs`,
`PoolStagingTest.PoolHostBaseAddrs`):
returns `numSlots` when the pool's storage is bounded and `maxPoolBlocks` on a
full-mirror storage. -/
def poolStagingCapacity (maxPoolBlocks : Nat) (boundedSlots : Option Nat) : Nat :=
  boundedSlots.getD maxPoolBlocks

/-- Remote address (`raddr`) resolution on sender `KVCacheManagerBase::GetBlockChunks`
(`RemoteAddress`) vs. receiver `KVCacheManagerBase::CopyPoolBlocks`
(`PoolStagingTest.SenderRaddrMatchesReceiverPointer`,
`PoolStagingTest.SenderRaddrUsesReceiverLeaseSlot`):
- On full-mirror storage (`arena = none`), `dstBlock < numBlocks` resolves to
  `some (baseAddr + dstBlock * stride + dstOffset)` (identity slot).
- On bounded storage (`arena = some a`), `dstBlock` is translated through
  `a.resolveBlockSlot uuid dstBlock` (`slot * stride + dstOffset`), returning
  `none` if `uuid` has no lease for `dstBlock`. -/
def resolveRemoteAddr (baseAddr stride numBlocks : Nat) (arena : Option Arena)
    (uuid dstBlock dstOffset : Nat) : Option Nat :=
  if dstBlock ≥ numBlocks then none
  else
    match arena with
    | none => some (baseAddr + dstBlock * stride + dstOffset)
    | some a =>
      (a.resolveBlockSlot uuid dstBlock).map (fun slot => baseAddr + slot * stride + dstOffset)

/-- Replay of `PoolStagingTest.FallsBackToFullMirrorWithoutHintsOrLeases` and
`PoolStagingTest.SmallPoolIdentitySharedStorageAndStrideMismatch`:
1. Zero hints (`blocks = 0, leases = 0, maxBlocks = 16`) or zero
   leases (`blocks = 2, leases = 0, maxBlocks = 16`) falls back to
   full mirror (`none`).
2. Small pool (`blocks = 2, leases = 2, maxBlocks = 3`, `4 ≥ 3`)
   falls back to full mirror (`none`).
3. Shared storage with matching strides (`blocks = 3, leases = 1, maxBlocks = 64`)
   configures a bounded arena of `some 3` slots.
4. Shared storage with mismatched strides (`stridesEqual = false`)
   falls back to full mirror (`none`). -/
theorem trace_fallback_to_full_mirror_and_stride_mismatch :
    configureStorageBounded 0 0 16 true = none ∧
    configureStorageBounded 2 0 16 true = none ∧
    configureStorageBounded 2 2 3 true = none ∧
    configureStorageBounded 3 1 64 true = some 3 ∧
    configureStorageBounded 2 2 64 false = none := by
  decide

/-- Replay of `PoolStagingTest.SenderRaddrMatchesReceiverPointer`,
`PoolStagingTest.SenderRaddrUsesReceiverLeaseSlot`, and
`PoolStagingTest.PoolHostBaseAddrs`:
1. Full-mirror receiver (`baseAddr = 10000, stride = 64, numBlocks = 8`):
   `(dstBlock = 3, dstOffset = 0)` resolves to `10000 + 3 * 64 = 10192`,
   `(dstBlock = 3, dstOffset = 32)` to `10224`, and `(dstBlock = 5, dstOffset = 16)`
   to `10336`.
2. Bounded receiver (`baseAddr = 20000, stride = 64, numBlocks = 64, numSlots = 2`):
   `uuid = 99` leases `[7, 42]` (mapped to slots `0` and `1`), so
   `(dstBlock = 42, dstOffset = 32)` resolves to `20000 + 1 * 64 + 32 = 20096`
   (using slot `1`, not `42 * 64`), while unleased `dstBlock = 10`
   resolves to `none`.
3. `PoolStagingTest.PoolHostBaseAddrs`: full-mirror pool reports `num_blocks = 8`,
   while bounded pool (`2 * 2 = 4 < 64`) reports `num_blocks = 4`. -/
theorem trace_sender_raddr_and_pool_host_base_addrs :
    resolveRemoteAddr 10000 64 8 none 77 3 0 = some 10192 ∧
    resolveRemoteAddr 10000 64 8 none 77 3 32 = some 10224 ∧
    resolveRemoteAddr 10000 64 8 none 77 5 16 = some 10336 ∧
    (((Arena.init 2).acquireLease 99 [7, 42]).map
      (fun a =>
        ( resolveRemoteAddr 20000 64 64 (some a) 99 42 32
        , resolveRemoteAddr 20000 64 64 (some a) 99 10 0 )) =
      some (some 20096, none)) ∧
    poolStagingCapacity 8 (configureStorageBounded 0 0 8 true) = 8 ∧
    poolStagingCapacity 64 (configureStorageBounded 2 2 64 true) = 4 := by
  decide

/-- End-to-end trace under `RegistrationPolicy.reserveBeforeLease` showing two
sessions (`uuid = 7` and `uuid = 8`) arming, leasing disjoint slots, settling,
and recycling slots while preserving `ActiveSessionsHoldLeases`. -/
def reserveBeforeLeaseTrace : List RegEv :=
  [ .beginArm 1 7 [5, 9]
  , .acquireArmLease 1
  , .finishArm 1
  , .beginArm 2 8 [5, 12]
  , .acquireArmLease 2
  , .finishArm 2
  , .settleSession 7
  , .settleSession 8 ]

theorem trace_reserveBeforeLease_end_to_end :
    ((RegState.sys .reserveBeforeLease 4).run reserveBeforeLeaseTrace).map
      (fun s => (ActiveSessionsHoldLeases s, s.activeSessions, s.arena.freeSlots.length)) =
      some (true, [], 4) := by
  decide

/-- Replay of `PoolStagingRecvRegistrationTest.DuplicatePoolRecvRegistrationKeepsOriginalStagingLease`
and `PoolStagingRecvRegistrationTest.StagingAcquireFailureReleasesUuidClaim`:
1. `PoolStagingRecvRegistrationTest.DuplicatePoolRecvRegistrationKeepsOriginalStagingLease`: on a
   2-slot bounded arena, `uuid = 4001` arms `[0]` (taking slot `0`, 1 free slot
   left); a duplicate `beginArm 2 4001 [0]` is rejected (`none`,
   `AlreadyExistsError`) without touching `4001`'s lease, and `uuid = 4002`
   arming `[5]` receives slot `1` (`≠ slot 0`).
2. `PoolStagingRecvRegistrationTest.StagingAcquireFailureReleasesUuidClaim`: on a 2-slot bounded
   arena, an oversized registration for `uuid = 4001` (`blocks = [0, 1, 2]`)
   claims `4001` in Phase 1 (`beginArm 1 4001 [0, 1, 2]`), fails in Phase 2
   (`acquireArmLease 1`, `ResourceExhausted`), and rolls back the claim leaving
   `2` free slots and `inFlightArms = []`; a subsequent valid registration for
   the same `uuid = 4001` (`blocks = [0]`) then succeeds and takes slot `0`. -/
theorem trace_pool_staging_recv_registration_tests :
    let sys2 := RegState.sys .reserveBeforeLease 2
    let arm4001 := [.beginArm 1 4001 [0], .acquireArmLease 1, .finishArm 1]
    (sys2.run (arm4001 ++ [.beginArm 2 4001 [0]]) = none) ∧
    ((sys2.run (arm4001 ++ [.beginArm 3 4002 [5], .acquireArmLease 3, .finishArm 3])).map
      (fun s => (s.arena.resolveBlockSlot 4001 0, s.arena.resolveBlockSlot 4002 5,
                 ActiveSessionsHoldLeases s, s.arena.freeSlots.length)) =
      some (some 0, some 1, true, 0)) ∧
    ((sys2.run [.beginArm 1 4001 [0, 1, 2], .acquireArmLease 1]).map
      (fun s => (s.hasActiveUuid 4001, s.hasInFlightUuid 4001, s.arena.freeSlots.length)) =
      some (false, false, 2)) ∧
    ((sys2.run [.beginArm 1 4001 [0, 1, 2], .acquireArmLease 1,
                .beginArm 2 4001 [0], .acquireArmLease 2, .finishArm 2]).map
      (fun s => (s.hasActiveUuid 4001, s.arena.resolveBlockSlot 4001 0,
                 s.arena.freeSlots.length)) =
      some (true, some 0, 1)) := by
  decide

/-- Mutant 1: If `AcquirePoolStagingLease` did not deduplicate/skip blocks
already leased to the same `uuid` (`if (cur != nullptr && cur->slot_by_block.contains(block_id)) continue;` in `KVCacheManagerBase::AcquirePoolStagingLease`), two pools
sharing a 3-slot storage and each needing 2 blocks `[10, 20]` would require
`2 + 2 = 4 > 3` slots and fail with `ResourceExhausted`. -/
def acquireLeaseWithoutIdempotence (a : Arena) (uuid : Nat) (blockIds : List Nat) : Option Arena :=
  if blockIds.length > a.numSlots || a.freeSlots.length < blockIds.length then
    none
  else
    let baseLease := (a.lookupLease uuid).getD ⟨[], []⟩
    match allocateMissing blockIds a.freeSlots baseLease with
    | none => none
    | some (free', lease') =>
      some { a with
        freeSlots := free'
        leases := upsertLease a.leases uuid lease' }

theorem mutant_non_idempotent_acquire_exhausts_shared_storage :
    (acquireLeaseWithoutIdempotence (Arena.init 3) 42 [10, 20]).bind
      (fun a1 => acquireLeaseWithoutIdempotence a1 42 [10, 20]) = none := by
  decide

/-- Mutant 2: If `ReleasePoolStagingLeases` erased `uuid` from `leases` without
returning `lease.slots` to `free_slots` (`for (int32_t slot : it->second->slots) { st.free_slots.push_back(slot); }` in `KVCacheManagerBase::ReleasePoolStagingLeases`),
total slots would shrink and subsequent sessions would starve with
`ResourceExhausted`. -/
def releaseLeaseLeakingSlots (a : Arena) (uuid : Nat) : Arena :=
  { a with leases := a.leases.filter (fun p => p.1 != uuid) }

theorem mutant_leaking_release_breaks_conservation_and_starves :
    ((Arena.init 2).acquireLease 7 [5, 9]).map
      (fun a1 =>
        let a2 := releaseLeaseLeakingSlots a1 7
        (decide (a2.totalSlots < a2.numSlots), a2.acquireLease 8 [1])) =
      some (true, none) := by
  decide

/-- Candidate registration events over two call IDs (`1, 2`), two UUIDs (`7, 8`),
and block `[5]`. -/
def checkEvents : List RegEv :=
  [ .beginArm 1 7 [5], .acquireArmLease 1, .finishArm 1
  , .beginArm 2 7 [5], .acquireArmLease 2, .finishArm 2
  , .beginArm 2 8 [5]
  , .settleSession 7, .settleSession 8 ]

def violatesHoldLeases (s : RegState) : Bool :=
  !ActiveSessionsHoldLeases s || (s.arena.totalSlots != s.arena.numSlots)

#guard (match ModelCheck.check (RegState.sys .leaseBeforeReserve 2) checkEvents violatesHoldLeases 8 with
  | .counterexample _ => true | _ => false)


end TpuSyncVerify.Transfer.Reshard.PoolStaging

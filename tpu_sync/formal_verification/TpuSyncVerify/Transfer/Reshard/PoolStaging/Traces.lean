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

/-- Replays `PoolStagingTest.BoundedArenaLeasesAndAddressing`: idempotent re-lease, over-capacity refusal, release, and slot reuse on a 4-slot arena. -/
theorem trace_bounded_arena_leases_and_addressing :
    runBoundedArenaScenario =
      some ⟨1, some 0, some 1, some 2, none, true, true, 4, 2, some 2, some 1⟩ := by
  decide

/-- Replays `PoolStagingTest.SmallPoolIdentitySharedStorageAndStrideMismatch` (shared-storage half): two pools on one storage share a single per-UUID lease. -/
theorem trace_shared_storage_pools_share_lease :
    ((Arena.init 3).acquireLease 5 [20]).map
      (fun a1 =>
        ( a1.freeSlots.length
        , a1.resolveBlockSlot 5 20
        , a1.acquireLease 5 [20] == some a1 )) =
      some (2, some 0, true) := by
  decide

/-- Bounded vs. full-mirror storage configuration (`KVCacheManagerBase::ConfigurePoolStaging`). -/
def configureStorageBounded (blocksPerLease leases maxPoolBlocks : Nat)
    (stridesEqual : Bool) : Option Nat :=
  let numSlots := blocksPerLease * leases
  if 0 < blocksPerLease && 0 < leases && stridesEqual && numSlots < maxPoolBlocks then
    some numSlots
  else
    none

/-- Reported staging capacity (`KVCacheManagerBase::PoolHostBaseAddrs`, `PoolStagingTest.PoolHostBaseAddrs`). -/
def poolStagingCapacity (maxPoolBlocks : Nat) (boundedSlots : Option Nat) : Nat :=
  boundedSlots.getD maxPoolBlocks

/-- Sender `RemoteAddress` vs. receiver `block_host_base` (`KVCacheManagerBase::GetBlockChunks`;
`PoolStagingTest.SenderRaddrMatchesReceiverPointer`, `PoolStagingTest.SenderRaddrUsesReceiverLeaseSlot`). -/
def resolveRemoteAddr (baseAddr stride numBlocks : Nat) (arena : Option Arena)
    (uuid dstBlock dstOffset : Nat) : Option Nat :=
  if dstBlock ≥ numBlocks then none
  else
    match arena with
    | none => some (baseAddr + dstBlock * stride + dstOffset)
    | some a =>
      (a.resolveBlockSlot uuid dstBlock).map (fun slot => baseAddr + slot * stride + dstOffset)

/-- Replays `PoolStagingTest.FallsBackToFullMirrorWithoutHintsOrLeases` and
`PoolStagingTest.SmallPoolIdentitySharedStorageAndStrideMismatch`: zero hints/leases, full-size arenas, and stride mismatches fall back to full mirror. -/
theorem trace_fallback_to_full_mirror_and_stride_mismatch :
    configureStorageBounded 0 2 16 true = none ∧
    configureStorageBounded 2 0 16 true = none ∧
    configureStorageBounded 2 2 4 true = none ∧
    configureStorageBounded 1 3 32 true = some 3 ∧
    configureStorageBounded 1 3 32 false = none := by
  decide

/-- Replays `PoolStagingTest.SenderRaddrMatchesReceiverPointer`, `PoolStagingTest.SenderRaddrUsesReceiverLeaseSlot`,
and `PoolStagingTest.PoolHostBaseAddrs`: full-mirror identity vs. bounded lease-slot addressing and reported capacities. -/
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

/-- End-to-end `RegistrationPolicy.reserveBeforeLease` trace: two sessions arm, lease disjoint slots, settle, and recycle slots. -/
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

/-- Replays `PoolStagingRecvRegistrationTest.DuplicatePoolRecvRegistrationKeepsOriginalStagingLease` and
`PoolStagingRecvRegistrationTest.StagingAcquireFailureReleasesUuidClaim`: duplicate UUID rejection preserves active leases and failed Phase 2 rolls back the claim. -/
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

/-- Mutant 1: omitting same-UUID block deduplication in `AcquirePoolStagingLease` exhausts a shared 3-slot storage. -/
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

/-- Mutant 2: erasing `leases[uuid]` in `ReleasePoolStagingLeases` without returning `lease.slots` breaks conservation and starves later sessions. -/
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

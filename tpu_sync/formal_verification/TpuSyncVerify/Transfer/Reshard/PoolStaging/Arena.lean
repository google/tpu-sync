import TpuSyncVerify.Common.System
import TpuSyncVerify.Common.ListAux

/-! # `PoolStaging`: bounded staging arena (`PoolStagingStorage`)

Part of `TpuSyncVerify.Transfer.Reshard.PoolStaging`; the module docstring and
source map live in `PoolStaging.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard.PoolStaging

open TpuSyncVerify

/-! ## Part I — Bounded Staging Arena (`PoolStagingStorage`) -/

/-- One UUID's staging lease on a bounded storage
(`KVCacheManagerBase::PoolStagingLease` in `base.h`).
`slotByBlock` maps `blockId ↦ slot`; `slots` records the leased slot indices in
allocation order. -/
structure Lease where
  slotByBlock : List (Nat × Nat)
  slots : List Nat
  deriving DecidableEq, Repr

/-- Look up the arena slot assigned to `blockId` in `lease`
(`cur->slot_by_block.contains(block_id)` in `KVCacheManagerBase::AcquirePoolStagingLease`,
`lease->slot_by_block.find(block_id)` in `KVCacheManagerBase::CopyPoolBlocks`, and
`staging_lease->slot_by_block.find(block_id)` in
`KVCacheManagerBase::GetBlockChunks`). -/
def Lease.lookupSlot (l : Lease) (blockId : Nat) : Option Nat :=
  (l.slotByBlock.find? (fun p => p.1 == blockId)).map Prod.snd

/-- One physical storage's bounded staging arena
(`KVCacheManagerBase::PoolStagingStorage` in `base.h`).
`freeSlots` is modeled with its top-of-stack (`back()` in C++) at the head of
the list so popping one slot is `freeSlots.head?` / `freeSlots.tail`. -/
structure Arena where
  numSlots : Nat
  freeSlots : List Nat
  leases : List (Nat × Lease)
  deriving DecidableEq, Repr

/-- Initial arena with `numSlots` free slots `[0, 1, ..., numSlots - 1]` in pop
order (`KVCacheManagerBase::ConfigurePoolStaging`: C++ pushes
`num_slots - 1` down to `0` so `free_slots.back()` pops `0, 1, ...`). -/
def Arena.init (numSlots : Nat) : Arena :=
  { numSlots := numSlots
  , freeSlots := List.range numSlots
  , leases := [] }

/-- Look up `uuid`'s current lease on `a` (`leases.find(uuid)` in
`KVCacheManagerBase::PoolStagingLeaseSnapshot` and `st.leases.find(uuid)` in
`KVCacheManagerBase::AcquirePoolStagingLease`). -/
def Arena.lookupLease (a : Arena) (uuid : Nat) : Option Lease :=
  (a.leases.find? (fun p => p.1 == uuid)).map Prod.snd

/-- Look up the slot assigned to `(uuid, blockId)` on `a`
(`KVCacheManagerBase::CopyPoolBlocks` / `KVCacheManagerBase::GetBlockChunks`). -/
def Arena.resolveBlockSlot (a : Arena) (uuid blockId : Nat) : Option Nat :=
  match a.lookupLease uuid with
  | none => none
  | some l => l.lookupSlot blockId

/-- Whether `blockId` is already mapped in `cur` (`cur != nullptr &&
cur->slot_by_block.contains(block_id)` in
`KVCacheManagerBase::AcquirePoolStagingLease`). -/
def alreadyLeased (cur : Option Lease) (blockId : Nat) : Bool :=
  match cur with
  | none => false
  | some l => (l.lookupSlot blockId).isSome

/-- Deduplicate `blockIds` in first-seen order while filtering out blocks
already mapped in `cur` (`std::vector<int64_t> missing;` loop in `KVCacheManagerBase::AcquirePoolStagingLease`). -/
def missingBlocksAux (cur : Option Lease) : List Nat → List Nat → List Nat
  | [], _ => []
  | b :: rest, seen =>
    if alreadyLeased cur b || seen.contains b then
      missingBlocksAux cur rest seen
    else
      b :: missingBlocksAux cur rest (b :: seen)

def missingBlocks (cur : Option Lease) (blockIds : List Nat) : List Nat :=
  missingBlocksAux cur blockIds []

/-- Assign the next `missing.length` slots from `freeSlots` to `missing` block
IDs, extending `cur` (`for (int64_t block_id : missing)` in `KVCacheManagerBase::AcquirePoolStagingLease`). -/
def allocateMissing : List Nat → List Nat → Lease → Option (List Nat × Lease)
  | [], free, acc => some (free, acc)
  | b :: bs, s :: free, acc =>
    allocateMissing bs free
      { slotByBlock := acc.slotByBlock ++ [(b, s)]
      , slots := acc.slots ++ [s] }
  | _ :: _, [], _ => none

/-- Update or insert `(uuid, l)` in `leases`, preserving key uniqueness
(`st.leases[uuid] = std::move(lease)` in
`KVCacheManagerBase::AcquirePoolStagingLease`). -/
def upsertLease : List (Nat × Lease) → Nat → Lease → List (Nat × Lease)
  | [], uuid, l => [(uuid, l)]
  | (u, old) :: rest, uuid, l =>
    if u = uuid then
      (uuid, l) :: rest
    else
      (u, old) :: upsertLease rest uuid l

/-- `KVCacheManagerBase::AcquirePoolStagingLease`.
Returns `some a'` on `OkStatus` and `none` on `ResourceExhausted`. -/
def Arena.acquireLease (a : Arena) (uuid : Nat) (blockIds : List Nat) : Option Arena :=
  let cur := a.lookupLease uuid
  let missing := missingBlocks cur blockIds
  if missing.isEmpty then
    some a
  else if missing.length > a.numSlots || a.freeSlots.length < missing.length then
    none
  else
    let baseLease := cur.getD ⟨[], []⟩
    match allocateMissing missing a.freeSlots baseLease with
    | none => none
    | some (free', lease') =>
      some { a with
        freeSlots := free'
        leases := upsertLease a.leases uuid lease' }

/-- `KVCacheManagerBase::ReleasePoolStagingLeases`.
Pushes `uuid`'s leased slots back onto the top of `freeSlots` (`push_back` in
C++, where `back()` is the head of `freeSlots`) and removes `uuid` from
`leases`. -/
def Arena.releaseLease (a : Arena) (uuid : Nat) : Arena :=
  match a.lookupLease uuid with
  | none => a
  | some l =>
    { a with
      freeSlots := l.slots.reverse ++ a.freeSlots
      leases := a.leases.filter (fun p => p.1 != uuid) }


end TpuSyncVerify.Transfer.Reshard.PoolStaging

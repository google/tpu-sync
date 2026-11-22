import TpuSyncVerify.Common.System
import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Common.ListAux

/-!
# Request block registry generation lifecycle (`RequestBlockRegistry`)

Models the controller-side `(req_id, uuid)` producer block lifecycle in
`RequestBlockRegistry` (`reg.{h,cc}`): per-unit `Register`, multi-owner
`LookupAndClaim`, owner-scoped `AbandonClaim`, full-set `Complete` voting,
`CancelIfUnclaimed` tombstones, forced `Release`, and TTL purges
(`PurgeExpired*`); composed into coordinator plans in
`Reshard/{Coordinator,SubPlanChain}.lean` (`docs/transfer/reshard.md`).

File aliases: `reg.{h,cc}` = `tpu_sync/kv_cache/reshard/request_block_registry.{h,cc}`,
`reshard_service.cc` = `tpu_sync/kv_cache/reshard/reshard_service.cc`,
`reshard_client.cc` = `tpu_sync/kv_cache/reshard/reshard_client.cc`,
`reshard_service_test.cc` = `tpu_sync/kv_cache/reshard/reshard_service_test.cc`.

## State (`RegGen`)

| Field | C++ | Role |
|---|---|---|
| `registeredUnits` | `request_blocks_[(req_id, unit)]` | `(unit, nonEmptyBlocks)` rows under this `(req_id, uuid)` |
| `claimedUnits` | `claimed_`, `claimed_units_[(req_id, uuid)]` | claimed source unit set (`some units` while claimed) |
| `claimOwners` | `claimed_owners_[(req_id, uuid)]` | active planning `claim_owner` tokens sharing the snapshot |
| `completedUnits` | `completed_units_[(req_id, uuid)].units` | producer units that have voted `Complete` |
| `cancelled` | `cancelled_[(req_id, uuid)]` | unclaimed cancellation tombstone |
| `retired` (ghost) | — | latches once `RetireLocked` fires (A4) |

## Events (`Ev`)

| Event | C++ |
|---|---|
| `register unit nonEmpty` | `RequestBlockRegistry::Register` (rejects cancelled, claimed, or conflicting payload; refreshes identical row) |
| `lookupAndClaim owner units` | `RequestBlockRegistry::LookupAndClaim` (claims snapshot, refreshes row TTLs, or retires if all voted) |
| `abandonClaim owner` | `RequestBlockRegistry::AbandonClaim` (drops claim only when `claimed_owners_` becomes empty) |
| `complete unit` | `RequestBlockRegistry::Complete` (rejects pre-claim non-empty vote; retires once `all_voted`) |
| `cancelIfUnclaimed` | `RequestBlockRegistry::CancelIfUnclaimed` (no-op if claimed; else erases rows and sets tombstone) |
| `release` | `RequestBlockRegistry::Release` (`RetireLocked`) |

## Lock scopes

Every mutable map of `RequestBlockRegistry` is `ABSL_GUARDED_BY(mu_)` (shared
with `WorkUnitDirectory`):

| Event | Lock scopes |
|---|---|
| `register` | 1 deciding `RequestBlockRegistry::mu_` scope (purges, checks, insert); optional prior manifest-snapshot `mu_` scope touches no registry map |
| `lookupAndClaim`, `complete`, `release` | 1 `RequestBlockRegistry::mu_` scope each (running `PurgeExpiredRequestBlocksLocked` and `PurgeExpiredLifecycleLocked` on entry) |
| `cancelIfUnclaimed` | 1 `RequestBlockRegistry::mu_` scope (running `PurgeExpiredLifecycleLocked` on entry) |
| `abandonClaim` | 1 `RequestBlockRegistry::mu_` scope |

## Assumptions (Scope)

1. **A1 — Non-empty deduplicated `units` in `lookupAndClaim`.** Callers reject
   empty `src_units` (`StartPoolReshard`) and duplicate `src_units`
   (`BuildPoolReshardPlan`, `Coordinator.lean`), matching C++ `std::set` equality.
2. **A2 — Pre-claim `Complete` checks `block_ids`.** `nonEmpty : Bool` abstracts
   `!block_ids.empty()` in `RequestBlockRegistry::Complete` (callers populate
   `block_ids` alongside or instead of `pool_spans`).
3. **A3 — No worker re-registration (`InvalidateUnitLocked`).**
   `HandleRaidenCommand` (`COMMAND_REGISTER_WORK_UNIT`) erases a unit's
   `request_blocks_` rows while keeping `claimed_` active; excluded from this
   single-worker-generation model.
4. **A4 — No re-registration after `RetireLocked`.** `RetireLocked` keeps no
   tombstone in C++, so a late `Register` can recreate `(req_id, uuid)`; ghost
   `retired` latches retirement for one generation.
5. **A5 — Per-generation TTL purges.** `purgeExpiredRequestBlocks` and
   `purgeExpiredLifecycle` clear all rows or lifecycle maps of the generation at
   once, coarser than C++ per-row / independent map `expires_at` timestamps.

## Properties

* `reachable_safe` (`Inv`): `ClaimOwnersConsistent` (`claimedUnits.isSome ↔
  claimOwners ≠ []` and claimed units registered, under A3), `TombstoneClean` &
  `tombstone_blocks_register_and_claim`, `claimed_blocks_register_and_cancel`,
  `abandonClaim_preserves_when_other_owner_present`, `RetiredClean` (under A4),
  and `PartialVotesNotAllVoted` & `complete_retires_iff_all_claimed_voted`.
* TTL & traces: `purgeExpiredAll_inv`,
  `trace_single_stage_votes_retire_multi_owner_claim` (`Complete` keyed by
  `(req_id, uuid)` retires sibling owners' claims),
  `trace_claim_refreshes_registration_ttl_while_failed_claim_does_not`, and
  `mutant_split_ttl_purges_request_blocks_while_claim_active`.

## Source map

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|------|-----------------|-------|
| `reg.h` | `RequestBlockStatus` | `:82-87` |
| `reg.h` | `RequestBlockRegistry` (`struct Completions {`) | `:123-126` |
| `reg.h` | `RequestBlockRegistry` (`std::map<BlockKey, RequestBlockRegistration, BlockKeyLess> request_blocks_` .. `std::map<LifecycleKey, double> cancelled_`) | `:139-150` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredRequestBlocksLocked` | `:61-69` |
| `reg.cc` | `RequestBlockRegistry::PurgeExpiredLifecycleLocked` | `:71-96` |
| `reg.cc` | `RequestBlockRegistry::RetireLocked` | `:98-116` |
| `reg.cc` | `RequestBlockRegistry::Register` | `:118-300` |
| `reg.cc` | `RequestBlockRegistry::Release` | `:302-315` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
| `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` | `:373-405` |
| `reg.cc` | `RequestBlockRegistry::Status` | `:407-436` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-501` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:503-519` |
| `reg.cc` | `RequestBlockRegistry::InvalidateUnitLocked` | `:521-529` |
| `reshard_service.cc` | `ReshardService::HandleControllerCommand` | `:167-353` |
| `reshard_service.cc` | `ReshardService::HandleRaidenCommand` (`registry_->InvalidateUnitLocked(registration.unit);`) | `:417` |
| `reshard_client.cc` | `ReshardClient::BuildReleaseRequestBlocks` | `:182-192` |
| `reshard_service_test.cc` | `ReshardStackTest.CancelTombstoneBlocksLateRegistration` | `:1478-1506` |
| `reshard_service_test.cc` | `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle` | `:1508-1572` |
| `reshard_service_test.cc` | `ReshardStackTest.CompletionVotesRetireAfterClaim` | `:1574-1597` |
| `reshard_service_test.cc` | `ReshardStackTest.TtlPurgesExpiredRegistrations` | `:1599-1607` |
| `reshard_service_test.cc` | `ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim` | `:1823-1848` |
| `reshard_service_test.cc` | `ReshardStackTest.AbandonedStageLeavesTheSiblingClaimInPlace` | `:1850-1877` |
| `reshard_service_test.cc` | `ReshardStackTest.AbandoningTheLastStageFreesTheClaim` | `:1879-1906` |
| `reshard_service_test.cc` | `ReshardStackTest.SpansForAnUnregisteredTagAreRefusedAtRegistration` | `:1908-1933` |
| `reshard_service_test.cc` | `ReshardStackTest.StridedGlobalSpansClipByWholePages` | `:2185-2216` |
| `reshard_service_test.cc` | `ReshardStackTest.NegativeOrdinalRejectedAtRegistration` | `:2278-2284` |
| `reshard_service_test.cc` | `ReshardStackTest.LegacyCommandsFailClosed` | `:2323-2334` |
| `reshard_service_test.cc` | `ReshardStackTest.DuplicateRegistrationRefreshesTtlWhileConflictsAndPostClaimWritesFail` | `:2443-2474` |
| `reshard_service_test.cc` | `ReshardStackTest.IncompatibleClaimUnitSetAndNonOwnerAbandonPreserveClaim` | `:2476-2505` |
| `reshard_service_test.cc` | `ReshardStackTest.CompletionNoOpsAndPreClaimGuard` | `:2507-2569` |
| `reshard_service_test.cc` | `ReshardStackTest.EmptyRegistrationCompletesBeforeClaimAndRetiresOnClaim` | `:2571-2598` |
| `reshard_service_test.cc` | `ReshardStackTest.ForcedReleaseClearsEveryLifecycleState` | `:2600-2663` |
| `reshard_service_test.cc` | `ReshardStackTest.ClaimRefreshesRegistrationTtlForSiblingStageAndCompletion` | `:2743-2786` |
| `reshard_service_test.cc` | `ReshardStackTest.FailedClaimDoesNotRefreshPartialRegistrationTtl` | `:2788-2806` |
<!-- source-map:end -->
-/

namespace TpuSyncVerify.Transfer.Reshard.Registry

open TpuSyncVerify

/-! ## Part I — Registry Generation State Machine (`RequestBlockRegistry`) -/

/-- Read-only status returned by `RequestBlockRegistry::Status`
(`RequestBlockStatus` in `request_block_registry.h`,
`RequestBlockRegistry::Status` in `request_block_registry.cc`). -/
inductive BlockStatus where
  | unknown
  | registered
  | claimed
  | cancelled
  deriving DecidableEq, Repr

/-- Per-`(req_id, uuid)` lifecycle state in `RequestBlockRegistry`
(`request_blocks_`, `claimed_`, `claimed_units_`, `claimed_owners_`,
`completed_units_`, and `cancelled_` in `request_block_registry.h`). -/
structure RegGen where
  /-- Registered producer units `(unit, nonEmptyBlocks)` in `request_blocks_`
  for this `(req_id, uuid)` (`RequestBlockRegistry::Register`
  `request_blocks_[key] = std::move(registration);`). -/
  registeredUnits : List (Nat × Bool)
  /-- Claimed source unit set `claimed_units_[(req_id, uuid)]` when `claimed_`
  is present (`RequestBlockRegistry::LookupAndClaim`
  `claimed_[lifecycle_key] = now + ttl_s_;`). -/
  claimedUnits : Option (List Nat)
  /-- Active planning owners `claimed_owners_[(req_id, uuid)]`
  (`RequestBlockRegistry` `claimed_owners_` in `request_block_registry.h`,
  `RequestBlockRegistry::LookupAndClaim`
  `claimed_owners_[lifecycle_key].insert(claim_owner);`). -/
  claimOwners : List Nat
  /-- Producer units that have cast a terminal vote in
  `completed_units_[(req_id, uuid)]` (`RequestBlockRegistry::Complete`
  `completion.units.insert(unit);`). -/
  completedUnits : List Nat
  /-- Whether `cancelled_` holds a tombstone for `(req_id, uuid)`
  (`RequestBlockRegistry::CancelIfUnclaimed`
  `cancelled_[lifecycle_key] = now + ttl_s_;`). -/
  cancelled : Bool
  /-- Ghost latch recording whether `RequestBlockRegistry::RetireLocked` has
  fired for the generation. The C++ keeps no such record; `step` uses it to
  exclude re-registration after retirement (Scope item 4). -/
  retired : Bool
  deriving DecidableEq, Repr, Hashable

/-- Initial empty state for a `(req_id, uuid)` generation. -/
def RegGen.init : RegGen :=
  { registeredUnits := []
  , claimedUnits := none
  , claimOwners := []
  , completedUnits := []
  , cancelled := false
  , retired := false }

/-- Look up `unit`'s registration (`request_blocks_.find({req_id, unit})` in
`RequestBlockRegistry::Register`, `Complete`, and `LookupAndClaim`). -/
def RegGen.lookupUnit (s : RegGen) (unit : Nat) : Option Bool :=
  (s.registeredUnits.find? (fun p => p.1 == unit)).map Prod.snd

/-- `RequestBlockRegistry::Status` precedence:
`cancelled > claimed > registered > unknown`
(`RequestBlockRegistry::Status` in `request_block_registry.cc`). -/
def RegGen.status (s : RegGen) : BlockStatus :=
  if s.cancelled then .cancelled
  else if s.claimedUnits.isSome then .claimed
  else if !s.registeredUnits.isEmpty then .registered
  else .unknown

/-- `RequestBlockRegistry::RetireLocked` (`request_block_registry.cc`):
erases `request_blocks_`, `claimed_`, `claimed_units_`, `claimed_owners_`,
`completed_units_`, and `cancelled_`. -/
def RegGen.retireLocked (_s : RegGen) : RegGen :=
  { registeredUnits := []
  , claimedUnits := none
  , claimOwners := []
  , completedUnits := []
  , cancelled := false
  , retired := true }

/-- Whether every unit in `expected` has voted in `completed`
(`all_voted` in `RequestBlockRegistry::Complete` and `LookupAndClaim`). -/
def allVoted (expected completed : List Nat) : Bool :=
  expected.all (fun u => completed.contains u)

/-- Insert `x` into `xs` if not already present (set semantics). -/
def insertNodup (x : Nat) (xs : List Nat) : List Nat :=
  if xs.contains x then xs else xs ++ [x]

/-- Whether `cur` is either unclaimed or already claimed for the exact same
source unit set `units` (`RequestBlockRegistry::LookupAndClaim`
`if (existing_units_it != claimed_units_.end() &&`). -/
def compatibleClaim (cur : Option (List Nat)) (units : List Nat) : Bool :=
  match cur with
  | none => true
  | some existing => existing == units

/-- Whether `unit` is in the claimed unit set `cur`
(`RequestBlockRegistry::Complete`
`expected_it->second.count(unit) > 0`). -/
def inClaimedUnits (cur : Option (List Nat)) (unit : Nat) : Bool :=
  match cur with
  | none => false
  | some expected => expected.contains unit

/-- Events of `RequestBlockRegistry` for a `(req_id, uuid)` generation. -/
inductive Ev where
  /-- `RequestBlockRegistry::Register(req_id, uuid, unit, block_ids, pool_spans)`. -/
  | register (unit : Nat) (nonEmpty : Bool)
  /-- `RequestBlockRegistry::LookupAndClaim(req_id, uuid, units, claim_owner)`. -/
  | lookupAndClaim (owner : Nat) (units : List Nat)
  /-- `RequestBlockRegistry::AbandonClaim(req_id, uuid, claim_owner)`. -/
  | abandonClaim (owner : Nat)
  /-- `RequestBlockRegistry::Complete(req_id, uuid, unit)`. -/
  | complete (unit : Nat)
  /-- `RequestBlockRegistry::CancelIfUnclaimed(req_id, uuid)`. -/
  | cancelIfUnclaimed
  /-- `RequestBlockRegistry::Release(req_id, uuid)`. -/
  | release
  deriving DecidableEq, Repr

/-- Transition function of `RequestBlockRegistry`
(`RequestBlockRegistry::Register`, `LookupAndClaim`, `AbandonClaim`, `Complete`,
`CancelIfUnclaimed`, and `Release` in `request_block_registry.cc`). -/
def RegGen.step (s : RegGen) (e : Ev) : Option RegGen :=
  match e with
  | .register unit nonEmpty =>
    -- Rejects if cancelled (`cancelled_.find(lifecycle_key) != cancelled_.end()`)
    -- or already claimed (`claimed_.find(lifecycle_key) != claimed_.end()`).
    -- `s.retired` is a model-only guard (Scope item 4): the C++ accepts a
    -- `Register` after `RetireLocked`.
    if s.cancelled || s.claimedUnits.isSome || s.retired then
      none
    else
      match s.lookupUnit unit with
      | some curNonEmpty =>
        -- Conflicting payload is rejected (`if (current.uuid != uuid || current.block_ids != normalized || current.pool_spans != normalized_pool_spans)`);
        -- identical payload is an idempotent refresh (`registration.expires_at = now + ttl_s_;`, `request_blocks_[key] = std::move(registration);`).
        if curNonEmpty != nonEmpty then none else some s
      | none =>
        some { s with registeredUnits := s.registeredUnits ++ [(unit, nonEmpty)] }
  | .lookupAndClaim owner units =>
    if s.cancelled || units.isEmpty then
      none
    else if !compatibleClaim s.claimedUnits units then
      -- `existing_units_it != claimed_units_.end()`: already claimed for a different source unit set.
      none
    else if !units.all (fun u => (s.lookupUnit u).isSome) then
      -- `for (const RaidenId& unit : units)`: missing producer block registration for a requested unit.
      none
    else
      let owners' := insertNodup owner s.claimOwners
      if allVoted units s.completedUnits then
        -- `if (completion_it != completed_units_.end())`: all claimed units had already voted (e.g. empty blocks).
        some s.retireLocked
      else
        some { s with
          claimedUnits := some units
          claimOwners := owners' }
  | .abandonClaim owner =>
    if !s.claimOwners.contains owner then
      none
    else
      let owners' := s.claimOwners.filter (fun o => o != owner)
      if owners'.isEmpty then
        -- `if (owner_it->second.empty())`: last owner abandoned; drop claim.
        some { s with
          claimedUnits := none
          claimOwners := [] }
      else
        some { s with claimOwners := owners' }
  | .complete unit =>
    let regOpt := s.lookupUnit unit
    let hasExpected := s.claimedUnits.isSome
    -- `if (!has_expected && has_registration &&`: non-empty producer registration cannot complete before claim.
    if !hasExpected && regOpt == some true then
      none
    else if regOpt.isNone && !inClaimedUnits s.claimedUnits unit then
      -- `if (!registration_matches && !unit_expected)`: idempotent no-op after retirement or for unknown unit.
      some s
    else
      let completed' := insertNodup unit s.completedUnits
      match s.claimedUnits with
      | some expected =>
        if allVoted expected completed' then
          -- `if (has_expected)` / `if (all_voted)`: full claimed set has voted -> RetireLocked.
          some s.retireLocked
        else
          some { s with completedUnits := completed' }
      | none =>
        some { s with completedUnits := completed' }
  | .cancelIfUnclaimed =>
    if s.cancelled then
      -- `if (cancelled_.find(lifecycle_key) != cancelled_.end())`: already cancelled -> returns true (idempotent).
      some s
    else if s.claimedUnits.isSome then
      -- `if (claimed_.find(lifecycle_key) != claimed_.end())`: already claimed -> returns false without modifying state.
      some s
    else
      -- `cancelled_[lifecycle_key] = now + ttl_s_;`: record tombstone and erase unclaimed registrations.
      some { s with
        registeredUnits := []
        claimedUnits := none
        claimOwners := []
        completedUnits := []
        cancelled := true }
  | .release =>
    some s.retireLocked

def RegGen.sys : System RegGen Ev where
  init := RegGen.init
  step := RegGen.step

/-! ## Part II — Core Safety Predicates and Inductive Invariant -/

/-- **Safety Property 1 (`ClaimOwnersConsistent`,
`RequestBlockRegistry::LookupAndClaim` and `RequestBlockRegistry::AbandonClaim`)**:
`claimedUnits` is `some units` if and only if `claimOwners` is non-empty, and
every unit in `units` is registered in `registeredUnits`. -/
def ClaimOwnersConsistent (s : RegGen) : Prop :=
  (s.claimedUnits.isSome = !s.claimOwners.isEmpty) ∧
  ∀ units, s.claimedUnits = some units →
    units ≠ [] ∧ ∀ u ∈ units, (s.lookupUnit u).isSome = true

/-- **Safety Property 2 (`TombstoneClean`,
`RequestBlockRegistry::Register`, `CancelIfUnclaimed`, and `LookupAndClaim`)**:
Whenever `s.cancelled = true`, all registration and claim tables are empty
(`registeredUnits = []`, `claimedUnits = none`, `claimOwners = []`,
`completedUnits = []`). -/
def TombstoneClean (s : RegGen) : Prop :=
  s.cancelled = true →
    s.registeredUnits = [] ∧
    s.claimedUnits = none ∧
    s.claimOwners = [] ∧
    s.completedUnits = []

/-- **Safety Property 3 (`RetiredClean`,
`RequestBlockRegistry::RetireLocked`)**:
Whenever `s.retired = true`, `registeredUnits = []` and `claimedUnits = none`
and `claimOwners = []`. This relies on `step` refusing `.register` on a retired
generation (Scope item 4); the C++ has the same three maps empty immediately
after `RetireLocked` and until the next `Register` of the same `(req_id, uuid)`. -/
def RetiredClean (s : RegGen) : Prop :=
  s.retired = true →
    s.registeredUnits = [] ∧
    s.claimedUnits = none ∧
    s.claimOwners = []

/-- **Safety Property 4 (`PartialVotesNotAllVoted`,
`RequestBlockRegistry::Complete` and `RequestBlockRegistry::LookupAndClaim`
`all_voted` check)**:
While a claim `s.claimedUnits = some expected` is active (not yet retired),
`allVoted expected s.completedUnits = false`. -/
def PartialVotesNotAllVoted (s : RegGen) : Prop :=
  ∀ expected, s.claimedUnits = some expected →
    allVoted expected s.completedUnits = false

/-- Combined inductive invariant of `RegGen`. -/
structure Inv (s : RegGen) : Prop where
  claimConsistent : ClaimOwnersConsistent s
  tombstoneClean : TombstoneClean s
  retiredClean : RetiredClean s
  partialVotes : PartialVotesNotAllVoted s

theorem init_inv : Inv RegGen.init where
  claimConsistent := ⟨rfl, by intro units h; cases h⟩
  tombstoneClean := by intro h; cases h
  retiredClean := by intro h; cases h
  partialVotes := by intro expected h; cases h

theorem retireLocked_inv (s : RegGen) : Inv s.retireLocked where
  claimConsistent := ⟨rfl, by intro units h; cases h⟩
  tombstoneClean := by intro h; cases h
  retiredClean := by intro _; exact ⟨rfl, rfl, rfl⟩
  partialVotes := by intro expected h; cases h

theorem insertNodup_nonEmpty (x : Nat) (xs : List Nat) :
    (insertNodup x xs).isEmpty = false := by
  unfold insertNodup
  split
  · rename_i h
    cases xs with
    | nil => simp at h
    | cons _ _ => rfl
  · simp

theorem step_preserves_inv {s s' : RegGen} {e : Ev}
    (hinv : Inv s) (hstep : RegGen.step s e = some s') :
    Inv s' := by
  cases e with
  | register unit nonEmpty =>
    dsimp only [RegGen.step] at hstep
    split at hstep
    · cases hstep
    rename_i hflags
    simp only [Bool.or_eq_true, not_or, Bool.not_eq_true] at hflags
    obtain ⟨⟨hcanc, hclaim⟩, hret⟩ := hflags
    cases hlook : s.lookupUnit unit with
    | some cur =>
      simp only [hlook] at hstep
      split at hstep
      · cases hstep
      · injection hstep with h; subst h; exact hinv
    | none =>
      simp only [hlook, Option.some.injEq] at hstep
      subst hstep
      have hclaim_none : s.claimedUnits = none := by
        cases h : s.claimedUnits <;> simp_all
      have howners_empty : s.claimOwners.isEmpty = true := by
        have h := hinv.claimConsistent.1
        rw [hclaim_none] at h
        cases h1 : s.claimOwners.isEmpty with
        | true => rfl
        | false => simp [h1] at h
      refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_⟩
      · simp [hclaim_none, howners_empty]
      · intro units h; simp [hclaim_none] at h
      · intro h; simp [hcanc] at h
      · intro h; simp [hret] at h
      · intro expected h; simp [hclaim_none] at h
  | lookupAndClaim owner units =>
    dsimp only [RegGen.step] at hstep
    split at hstep
    · cases hstep
    rename_i h1
    simp only [Bool.or_eq_true, not_or, Bool.not_eq_true] at h1
    obtain ⟨hcanc, hunits_ne⟩ := h1
    split at hstep
    · cases hstep
    split at hstep
    · cases hstep
    rename_i hall_reg
    simp only [Bool.not_eq_true, Bool.not_eq_false'] at hall_reg
    split at hstep
    · injection hstep with h; subst h
      exact retireLocked_inv s
    · rename_i hnot_voted
      injection hstep with h; subst h
      simp only [Bool.not_eq_true] at hnot_voted
      refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_⟩
      · simp [insertNodup_nonEmpty]
      · intro u' hu'
        injection hu' with hu'; subst hu'
        refine ⟨List.isEmpty_eq_false_iff.mp hunits_ne, ?_⟩
        exact List.all_eq_true.mp hall_reg
      · intro h; simp [hcanc] at h
      · intro hret
        have hreg_empty := (hinv.retiredClean hret).1
        cases units with
        | nil => simp at hunits_ne
        | cons u0 _ =>
          have hu0 := List.all_eq_true.mp hall_reg u0 List.mem_cons_self
          unfold RegGen.lookupUnit at hu0
          simp [hreg_empty] at hu0
      · intro expected hexp
        injection hexp with hexp; subst hexp
        exact hnot_voted
  | abandonClaim owner =>
    dsimp only [RegGen.step] at hstep
    split at hstep
    · cases hstep
    split at hstep
    · rename_i hemp
      injection hstep with h; subst h
      refine ⟨⟨rfl, by intro u h; cases h⟩, ?_, ?_, by intro e h; cases h⟩
      · intro hc
        obtain ⟨h1, _, _, h4⟩ := hinv.tombstoneClean hc
        exact ⟨h1, rfl, rfl, h4⟩
      · intro hr
        obtain ⟨h1, _, _⟩ := hinv.retiredClean hr
        exact ⟨h1, rfl, rfl⟩
    · rename_i hnot_emp
      injection hstep with h; subst h
      simp only [Bool.not_eq_true] at hnot_emp
      have hclaim_some : s.claimedUnits.isSome = true := by
        cases h : s.claimOwners with
        | nil => simp [h] at hnot_emp
        | cons _ _ =>
          have hc := hinv.claimConsistent.1
          simp [h] at hc
          exact hc
      refine ⟨⟨?_, hinv.claimConsistent.2⟩, ?_, ?_, hinv.partialVotes⟩
      · simp [hclaim_some, hnot_emp]
      · intro hc
        obtain ⟨_, h2, _⟩ := hinv.tombstoneClean hc
        simp [h2] at hclaim_some
      · intro hr
        obtain ⟨_, h2, _⟩ := hinv.retiredClean hr
        simp [h2] at hclaim_some
  | complete unit =>
    dsimp only [RegGen.step] at hstep
    split at hstep
    · cases hstep
    split at hstep
    · injection hstep with h; subst h; exact hinv
    rename_i hrel
    cases hclaim : s.claimedUnits with
    | some expected =>
      simp only [hclaim] at hstep
      split at hstep
      · injection hstep with h; subst h
        exact retireLocked_inv s
      · rename_i hnot_all
        injection hstep with h; subst h
        simp only [Bool.not_eq_true] at hnot_all
        refine ⟨?_, ?_, ?_, ?_⟩
        · refine ⟨?_, ?_⟩
          · have hc := hinv.claimConsistent.1
            simpa [hclaim] using hc
          · intro u hu
            exact hinv.claimConsistent.2 u (by simpa [hclaim] using hu)
        · intro hc
          obtain ⟨_, h2, _⟩ := hinv.tombstoneClean hc
          simp [hclaim] at h2
        · intro hr
          obtain ⟨_, h2, _⟩ := hinv.retiredClean hr
          simp [hclaim] at h2
        · intro exp hexp
          injection hexp with hexp; subst hexp
          exact hnot_all
    | none =>
      simp only [hclaim] at hstep
      injection hstep with h; subst h
      have howners : s.claimOwners.isEmpty = true := by
        have hc := hinv.claimConsistent.1
        rw [hclaim] at hc
        cases h1 : s.claimOwners.isEmpty with
        | true => rfl
        | false => simp [h1] at hc
      refine ⟨⟨by simp [howners], by intro u h; cases h⟩, ?_, ?_, by intro e h; cases h⟩
      · intro hc
        obtain ⟨h1, _, _⟩ := hinv.tombstoneClean hc
        unfold RegGen.lookupUnit inClaimedUnits at hrel
        simp [hclaim, h1] at hrel
      · intro hr
        obtain ⟨h1, _, h3⟩ := hinv.retiredClean hr
        exact ⟨h1, rfl, h3⟩
  | cancelIfUnclaimed =>
    dsimp only [RegGen.step] at hstep
    split at hstep
    · injection hstep with h; subst h; exact hinv
    split at hstep
    · injection hstep with h; subst h; exact hinv
    injection hstep with h; subst h
    refine ⟨⟨rfl, by intro u h; cases h⟩, ?_, ?_, by intro e h; cases h⟩
    · intro _; exact ⟨rfl, rfl, rfl, rfl⟩
    · intro _; exact ⟨rfl, rfl, rfl⟩
  | release =>
    dsimp only [RegGen.step] at hstep
    injection hstep with h; subst h
    exact retireLocked_inv s

/-- **Main Reachability Theorem for `RequestBlockRegistry` (`reachable_safe`)**:
Every reachable state of `RegGen.sys` satisfies `Inv`. -/
theorem reachable_safe {s : RegGen} (hr : RegGen.sys.Reachable s) : Inv s := by
  induction hr with
  | init => exact init_inv
  | step _ hstep ih => exact step_preserves_inv ih hstep

/-! ## Part III — Named Protocol Guarantees -/

/-- **Tombstone Blocks Late Registration & Claim (`TombstoneBlocksLateRegister`,
`RequestBlockRegistry::Register` and `RequestBlockRegistry::LookupAndClaim`
`if (cancelled_.find(lifecycle_key) != cancelled_.end())`)**:
Once a generation is cancelled (`s.cancelled = true`), both `register` and
`lookupAndClaim` are unconditionally rejected (`none`). -/
theorem tombstone_blocks_register_and_claim (s : RegGen) (hc : s.cancelled = true)
    (unit : Nat) (nonEmpty : Bool) (owner : Nat) (units : List Nat) :
    RegGen.step s (.register unit nonEmpty) = none ∧
    RegGen.step s (.lookupAndClaim owner units) = none := by
  simp [RegGen.step, hc]

/-- **Active Claim Blocks Cancellation & Registration (`ClaimedNotCancelled`,
`RequestBlockRegistry::Register` and `RequestBlockRegistry::CancelIfUnclaimed`
`if (claimed_.find(lifecycle_key) != claimed_.end())`)**:
While `s.claimedUnits = some units`, `register` is rejected and
`cancelIfUnclaimed` is a no-op that leaves `s` (and all its registered blocks
and claims) unchanged. -/
theorem claimed_blocks_register_and_cancel {s : RegGen} {units : List Nat}
    (hinv : Inv s) (hclaim : s.claimedUnits = some units)
    (unit : Nat) (nonEmpty : Bool) :
    RegGen.step s (.register unit nonEmpty) = none ∧
    RegGen.step s .cancelIfUnclaimed = some s := by
  have hnc : s.cancelled = false := by
    cases hc : s.cancelled with
    | false => rfl
    | true =>
      have h2 := (hinv.tombstoneClean hc).2.1
      simp [hclaim] at h2
  simp [RegGen.step, hclaim, hnc]

/-- **Multi-Owner Claim Survival (`MultiOwnerSurvival`,
`RequestBlockRegistry` `claimed_owners_` in `request_block_registry.h`,
`RequestBlockRegistry::AbandonClaim` in `request_block_registry.cc`)**:
If two distinct planning owners `o₁ ≠ o₂` hold claims on `s`, then
`abandonClaim o₁` preserves `s'.claimedUnits = s.claimedUnits` (`o₂`'s claim
survives). -/
theorem abandonClaim_preserves_when_other_owner_present {s : RegGen}
    {o₁ o₂ : Nat} (hne : o₂ ≠ o₁)
    (ho₁ : s.claimOwners.contains o₁ = true)
    (ho₂ : s.claimOwners.contains o₂ = true) :
    ∃ s', RegGen.step s (.abandonClaim o₁) = some s' ∧
      s'.claimedUnits = s.claimedUnits ∧
      s'.claimOwners.contains o₂ = true := by
  dsimp only [RegGen.step]
  simp only [ho₁, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  have hmem₂ : o₂ ∈ s.claimOwners.filter (fun o => o != o₁) := by
    rw [List.mem_filter]
    exact ⟨List.contains_iff_mem.mp ho₂, bne_iff_ne.mpr hne⟩
  have hnot_emp : (s.claimOwners.filter (fun o => o != o₁)).isEmpty = false :=
    List.isEmpty_eq_false_iff.mpr (List.ne_nil_of_mem hmem₂)
  simp only [hnot_emp, Bool.false_eq_true, ↓reduceIte]
  refine ⟨_, rfl, rfl, List.contains_iff_mem.mpr hmem₂⟩

/-- **Full-Set Voting Required to Retire Claimed Generation
(`CompleteRequiresFullVotingSet`, `RequestBlockRegistry::Complete`
`if (has_expected)`)**:
When `s.claimedUnits = some expected` (and `s.retired = false`), a `complete`
step for an expected unit `unit ∈ expected` transitions to `s'.retired = true`
if and only if every unit `u ∈ expected` has voted in
`insertNodup unit s.completedUnits`. -/
theorem complete_retires_iff_all_claimed_voted {s s' : RegGen} {expected : List Nat}
    {unit : Nat}
    (hclaim : s.claimedUnits = some expected)
    (hmem : expected.contains unit = true)
    (hnot_ret : s.retired = false)
    (hstep : RegGen.step s (.complete unit) = some s') :
    s'.retired = true ↔
      allVoted expected (insertNodup unit s.completedUnits) = true := by
  dsimp only [RegGen.step] at hstep
  simp only [hclaim, inClaimedUnits, hmem, Option.isSome_some, Bool.not_true,
    Bool.false_and, Bool.and_false, Bool.false_eq_true, ↓reduceIte] at hstep
  split at hstep
  · rename_i hall
    injection hstep with h; subst h
    simp [RegGen.retireLocked, hall]
  · rename_i hnot_all
    injection hstep with h; subst h
    simp only [Bool.not_eq_true] at hnot_all
    simp [hnot_ret, hnot_all]

/-! ## Part IV — Replay traces and mutants -/

/-- Replays `ReshardStackTest.CompletionVotesRetireAfterClaim`,
`ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim`, and
`ReshardStackTest.AbandonedStageLeavesTheSiblingClaimInPlace`: multi-owner claim
survives one owner abandoning and retires only once both claimed units vote. -/
def registryHappyTrace : List Ev :=
  [ .register 0 true
  , .register 1 true
  , .lookupAndClaim 101 [0, 1]
  , .lookupAndClaim 102 [0, 1]
  , .abandonClaim 101
  , .complete 0
  , .complete 1 ]

theorem trace_registry_multi_owner_and_voting :
    (RegGen.sys.run registryHappyTrace).map
      (fun s => (s.status, s.claimedUnits, s.registeredUnits, s.retired)) =
      some (.unknown, none, [], true) := by
  decide

/-- Replays `ReshardStackTest.CompletionVotesRetireAfterClaim`: after the first
of two claimed units votes, the generation remains `.claimed` (`retired = false`). -/
theorem trace_partial_vote_does_not_retire :
    (RegGen.sys.run (registryHappyTrace.take 6)).map
      (fun s => (s.status, s.claimedUnits, s.completedUnits, s.retired)) =
      some (.claimed, some [0, 1], [0], false) := by
  decide

/-- Replays `ReshardStackTest.CancelTombstoneBlocksLateRegistration`: cancelling
an unclaimed generation records `.cancelled` and rejects late `register` or `lookupAndClaim`. -/
theorem trace_tombstone_blocks_late_register_and_claim :
    ((RegGen.sys.run [.cancelIfUnclaimed]).map
      (fun s => (s.status, RegGen.step s (.register 0 true))) =
      some (.cancelled, none)) ∧
    ((RegGen.sys.run [.register 0 true, .cancelIfUnclaimed]).map
      (fun s =>
        ( s.status
        , s.registeredUnits
        , RegGen.step s (.register 1 true)
        , RegGen.step s (.lookupAndClaim 101 [0]) )) =
      some (.cancelled, [], none, none)) := by
  decide

/-- Replays `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle`:
`.unknown` → `.registered` → `.cancelled`, while `.claimed` refuses `cancelIfUnclaimed`. -/
theorem trace_request_block_status_probe_lifecycle :
    RegGen.init.status = .unknown ∧
    (RegGen.sys.run [.register 0 true]).map RegGen.status = some .registered ∧
    (RegGen.sys.run [.register 0 true, .cancelIfUnclaimed]).map RegGen.status =
      some .cancelled ∧
    (RegGen.sys.run
      [.register 0 true, .register 1 true, .lookupAndClaim 101 [0, 1],
       .cancelIfUnclaimed]).map
      (fun s => (s.status, s.cancelled)) = some (.claimed, false) := by
  decide

/-- Replays `ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim`,
`ReshardStackTest.AbandonedStageLeavesTheSiblingClaimInPlace`, and
`ReshardStackTest.AbandoningTheLastStageFreesTheClaim`: sibling claims survive single-stage
abandonment, whereas the last stage leaving frees the claim for `cancelIfUnclaimed`. -/
theorem trace_pipelined_stages_claim_and_abandon :
    ((RegGen.sys.run
      [ .register 0 true, .register 1 true
      , .lookupAndClaim 100 [0, 1], .lookupAndClaim 101 [0, 1]
      , .abandonClaim 101, .cancelIfUnclaimed
      , .lookupAndClaim 101 [0, 1] ]).map
      (fun s => (s.status, s.cancelled, s.claimOwners)) =
      some (.claimed, false, [100, 101])) ∧
    ((RegGen.sys.run
      [ .register 0 true, .register 1 true
      , .lookupAndClaim 100 [0, 1], .abandonClaim 100
      , .lookupAndClaim 101 [0, 1], .abandonClaim 101 ]).map
      (fun s =>
        ( s.status
        , (RegGen.step s .cancelIfUnclaimed).map
            (fun s_canc => (s_canc.status, RegGen.step s_canc (.lookupAndClaim 100 [0, 1]))) )) =
      some (.registered, some (.cancelled, none))) := by
  decide

/-- Mutant: retiring on the first `Complete` vote without checking `all_voted`
erases peer producer blocks while another unit is still transferring. -/
def stepCompletePremature (s : RegGen) (unit : Nat) : RegGen :=
  let _ := unit
  s.retireLocked

theorem mutant_premature_complete_retires_while_peer_active :
    (RegGen.sys.run [.register 0 true, .register 1 true, .lookupAndClaim 101 [0, 1]]).map
      (fun s =>
        let s_good := RegGen.step s (.complete 0)
        let s_bad := stepCompletePremature s 0
        (s_good.map (fun x => x.retired), s_bad.retired, s_bad.registeredUnits)) =
      some (some false, true, []) := by
  decide

/-- Mutant: `AbandonClaim` clearing `claimed_` without checking `claimed_owners_.empty()`
exposes a concurrent sibling owner's blocks to `CancelIfUnclaimed`. -/
def stepAbandonUnconditional (s : RegGen) (owner : Nat) : RegGen :=
  { s with
    claimedUnits := none
    claimOwners := s.claimOwners.filter (fun o => o != owner) }

theorem mutant_unconditional_abandon_exposes_sibling_owner_to_cancel :
    (RegGen.sys.run
      [.register 0 true, .lookupAndClaim 101 [0], .lookupAndClaim 102 [0]]).bind
      (fun s =>
        let s_bad := stepAbandonUnconditional s 101
        (RegGen.step s_bad .cancelIfUnclaimed).map (fun x => x.cancelled)) =
      some true := by
  decide

/-- Because `completed_units_` is keyed by `(req_id, uuid)` rather than `claim_owner`,
every unit in `claimed_units_` voting once retires all sibling owners' claims. -/
theorem trace_single_stage_votes_retire_multi_owner_claim :
    (RegGen.sys.run
      [ .register 0 true
      , .register 1 true
      , .lookupAndClaim 101 [0, 1]
      , .lookupAndClaim 102 [0, 1]
      , .complete 0
      , .complete 1 ]).map
      (fun s => (s.retired, s.claimedUnits, s.claimOwners, s.registeredUnits)) =
      some (true, none, [], []) := by
  decide

/-! ## Part V — Wall-Clock TTL Expiration (`PurgeExpired*`) -/

/-- `RequestBlockRegistry::PurgeExpiredRequestBlocksLocked`
(`tpu_sync/kv_cache/reshard/request_block_registry.cc`):
erases `request_blocks_` entries whose `registration.expires_at <= now` (set to
`now + ttl_s_` at `RequestBlockRegistry::Register`
`registration.expires_at = now + ttl_s_;` and refreshed to `now + ttl_s_` on
claimed rows at `RequestBlockRegistry::LookupAndClaim`
`for (const auto& row : claimed_rows)`). Because every claim refreshes its
claimed rows and `claimed_[lifecycle_key]` to the same `now + ttl_s_`, and
`Register` refuses a claimed generation, the claimed rows of a claimed
generation expire together with its claim (`purgeExpiredAll`); for those rows
this rows-only purge applies on its own only to an unclaimed generation.

Coarser than the C++: this helper clears every row of the generation at once.
In the C++ each row has its own `expires_at`, and `LookupAndClaim` refreshes
only the rows of the units it is called with, so a row registered under the
same `(req_id, uuid)` for a unit outside the claimed set keeps its registration
expiry and can be purged alone while the generation is claimed. -/
def RegGen.purgeExpiredRequestBlocks (s : RegGen) : RegGen :=
  { s with registeredUnits := [] }

/-- `RequestBlockRegistry::PurgeExpiredLifecycleLocked`
(`tpu_sync/kv_cache/reshard/request_block_registry.cc`):
erases the `claimed_`, `cancelled_`, and `completed_units_` entries whose own
`expires_at <= now`, then the `claimed_units_` / `claimed_owners_` entries whose
`claimed_` entry is gone.

Coarser than the C++: this helper drops the claim, the tombstone, and the votes
together. In the C++ the three expiries are independent, and
`completed_units_[key].expires_at` is refreshed by every `Complete`, so votes
can outlive the claim they were cast under. -/
def RegGen.purgeExpiredLifecycle (s : RegGen) : RegGen :=
  { s with
    claimedUnits := none
    claimOwners := []
    completedUnits := []
    cancelled := false }

/-- Combined TTL purge when `now` advances past both `registration.expires_at`
and lifecycle `expires_at` (`RequestBlockRegistry::PurgeExpiredRequestBlocksLocked`
and `PurgeExpiredLifecycleLocked` in `request_block_registry.cc`,
`ReshardStackTest.RequestBlockStatusProbeTracksLifecycle`,
`ReshardStackTest.TtlPurgesExpiredRegistrations`, and
`ReshardStackTest.FailedClaimDoesNotRefreshPartialRegistrationTtl` in
`reshard_service_test.cc`). -/
def RegGen.purgeExpiredAll (s : RegGen) : RegGen :=
  s.purgeExpiredRequestBlocks.purgeExpiredLifecycle

theorem purgeExpiredAll_inv (s : RegGen) : Inv s.purgeExpiredAll where
  claimConsistent := ⟨rfl, by intro units h; cases h⟩
  tombstoneClean := by intro h; cases h
  retiredClean := by intro _; exact ⟨rfl, rfl, rfl⟩
  partialVotes := by intro expected h; cases h

/-- Replays `ReshardStackTest.TtlPurgesExpiredRegistrations` and the TTL
tombstone lapse in `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle`:
`purgeExpiredAll` clears expired unclaimed registrations and `.cancelled` tombstones. -/
theorem trace_ttl_purges_expired_registrations_and_tombstones :
    ((RegGen.sys.run [.register 0 true]).map
      (fun s =>
        let s_exp := s.purgeExpiredAll
        (s_exp.status, RegGen.step s_exp (.lookupAndClaim 101 [0]))) =
      some (.unknown, none)) ∧
    ((RegGen.sys.run [.register 0 true, .cancelIfUnclaimed]).map
      (fun s => (s.status, s.purgeExpiredAll.status)) =
      some (.cancelled, .unknown)) := by
  decide

/-- Replays `ReshardStackTest.ClaimRefreshesRegistrationTtlForSiblingStageAndCompletion`
and `ReshardStackTest.FailedClaimDoesNotRefreshPartialRegistrationTtl`: a valid claim
refreshes row TTLs for sibling stages and votes, whereas a failed claim leaves TTLs unrefreshed. -/
theorem trace_claim_refreshes_registration_ttl_while_failed_claim_does_not :
    (let pre := [.register 0 true, .register 1 true, .lookupAndClaim 100 [0, 1],
                 .cancelIfUnclaimed, .lookupAndClaim 101 [0, 1]]
     (RegGen.sys.run pre).map
       (fun s => (s.status, s.lookupUnit 1)) =
       some (.claimed, some true) ∧
     (RegGen.sys.run (pre ++ [.complete 0, .complete 1])).map
       (fun s => (s.retired, s.status)) = some (true, .unknown)) ∧
    ((RegGen.sys.run [.register 0 true]).map
      (fun s =>
        ( (RegGen.step s (.lookupAndClaim 100 [0, 1])).isNone
        , s.purgeExpiredAll.status )) =
      some (true, .unknown)) := by
  decide

/-- Mutant: omitting `LookupAndClaim`'s row TTL refresh lets `purgeExpiredRequestBlocks`
erase `request_blocks_` while `claimed_` stays active, breaking sibling claims and `ClaimOwnersConsistent`. -/
theorem mutant_split_ttl_purges_request_blocks_while_claim_active :
    ((RegGen.sys.run [.register 0 true, .lookupAndClaim 101 [0]]).map
      (fun s =>
        let s_split := s.purgeExpiredRequestBlocks
        ( s_split.status
        , s_split.claimedUnits
        , s_split.registeredUnits
        , (RegGen.step s_split (.lookupAndClaim 102 [0])).isNone )) =
      some (.claimed, some [0], [], true)) := by
  decide

/-- Candidate registry events over two producer units (`0, 1`) and two pipelined
claim owners (`100, 101`). -/
def checkEvents : List Ev :=
  [ .register 0 true, .register 1 true, .register 0 false
  , .lookupAndClaim 100 [0, 1], .lookupAndClaim 101 [0, 1], .lookupAndClaim 100 [0]
  , .abandonClaim 100, .abandonClaim 101
  , .complete 0, .complete 1
  , .cancelIfUnclaimed, .release ]

private theorem forall_some_iff_all {α : Type} (o : Option α) (P : α → Prop) [DecidablePred P] :
    (∀ x, o = some x → P x) ↔ o.all (fun x => decide (P x)) = true := by
  cases o <;> simp

instance (s : RegGen) : Decidable (ClaimOwnersConsistent s) :=
  decidable_of_iff
    ((s.claimedUnits.isSome = !s.claimOwners.isEmpty) ∧
      s.claimedUnits.all (fun units =>
        decide (units ≠ [] ∧ ∀ u ∈ units, (s.lookupUnit u).isSome = true)) = true)
    (by unfold ClaimOwnersConsistent; rw [forall_some_iff_all])

instance (s : RegGen) : Decidable (TombstoneClean s) := by
  unfold TombstoneClean; infer_instance

instance (s : RegGen) : Decidable (RetiredClean s) := by
  unfold RetiredClean; infer_instance

instance (s : RegGen) : Decidable (PartialVotesNotAllVoted s) :=
  decidable_of_iff
    (s.claimedUnits.all (fun expected =>
      decide (allVoted expected s.completedUnits = false)) = true)
    (by unfold PartialVotesNotAllVoted; rw [forall_some_iff_all])

instance (s : RegGen) : Decidable (Inv s) :=
  decidable_of_iff
    (ClaimOwnersConsistent s ∧ TombstoneClean s ∧ RetiredClean s ∧ PartialVotesNotAllVoted s)
    ⟨fun ⟨a, b, c, d⟩ => ⟨a, b, c, d⟩, fun h => ⟨h.1, h.2, h.3, h.4⟩⟩

/-- Executable negation of `Inv`. -/
def violates (s : RegGen) : Bool := !decide (Inv s)

theorem violates_iff (s : RegGen) : violates s = false ↔ Inv s := by
  simp [violates]

#guard ModelCheck.check RegGen.sys checkEvents violates 12 = .safe

end TpuSyncVerify.Transfer.Reshard.Registry


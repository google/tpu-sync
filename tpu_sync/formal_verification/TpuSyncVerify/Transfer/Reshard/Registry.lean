import TpuSyncVerify.Common.System
import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Common.ListAux

/-!
# Request block registry generation lifecycle

Formal model and proofs of `RequestBlockRegistry`
(`tpu_sync/kv_cache/reshard/request_block_registry.{h,cc}`), the controller-side
state machine that manages `(req_id, uuid)` producer block generations across
registration, multi-owner planning claims, claim abandonment, per-unit
completion voting, and unclaimed cancellation tombstones.

File aliases used in the source map below:
`reg.h` / `reg.cc` (`tpu_sync/kv_cache/reshard/request_block_registry.{h,cc}`)
and `reshard_service_test.cc`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`).

  ## C++ anchorage

  1. **Per-unit producer registration (`RequestBlockRegistry::Register` in
     `tpu_sync/kv_cache/reshard/request_block_registry.cc`):**
     Records `request_blocks_[(req_id, unit)]` with `uuid` and normalized
     `block_ids` / `pool_spans`. Rejects if:
     - `cancelled_` contains `(req_id, uuid)`
       (`if (cancelled_.find(lifecycle_key) != cancelled_.end())`),
     - `claimed_` contains `(req_id, uuid)`
       (`if (claimed_.find(lifecycle_key) != claimed_.end())`), or
     - `request_blocks_[(req_id, unit)]` already exists with a different payload
       (`auto existing = request_blocks_.find(key);`; identical re-registration
       is an idempotent TTL refresh, `existing->second.expires_at = now + ttl_s_;`).

  2. **Claim linearization (`RequestBlockRegistry::LookupAndClaim` in
     `tpu_sync/kv_cache/reshard/request_block_registry.cc`):**
     Rejects if `cancelled_` contains `(req_id, uuid)`
     (`if (cancelled_.find(lifecycle_key) != cancelled_.end())`), if
     `claimed_units_` already holds a different source unit set
     (`if (existing_units_it != claimed_units_.end() &&`), or if any requested
     `unit ∈ units` is missing from `request_blocks_` or has a different `uuid`
     (`for (const RaidenId& unit : units)`). Otherwise marks
     `claimed_[(req_id, uuid)]`, sets `claimed_units_[(req_id, uuid)] = units`,
     and inserts `claim_owner` into `claimed_owners_[(req_id, uuid)]`
     (`claimed_[lifecycle_key] = now + ttl_s_;` ..
     `claimed_owners_[lifecycle_key].insert(claim_owner);`). If all `units` have
     already voted in `completed_units_`, calls `RetireLocked`
     (`if (completion_it != completed_units_.end())`).

  3. **Owner-scoped claim rollback (`RequestBlockRegistry::AbandonClaim` in
     `tpu_sync/kv_cache/reshard/request_block_registry.cc`):**
     Removes `claim_owner` from `claimed_owners_[(req_id, uuid)]`; only when
     `claimed_owners_` becomes empty does it erase `claimed_`, `claimed_units_`,
     and `claimed_owners_` (`if (owner_it->second.empty())`), allowing pipelined
     destination stages to share one source snapshot claim
     (`RequestBlockRegistry` `claimed_owners_` in `request_block_registry.h`).

  4. **Full-set completion voting (`RequestBlockRegistry::Complete` in
     `tpu_sync/kv_cache/reshard/request_block_registry.cc`):**
     Rejects pre-claim completion of a non-empty registration
     (`if (!has_expected && has_registration &&`). Otherwise inserts `unit` into
     `completed_units_[(req_id, uuid)].units`
     (`completion.units.insert(unit);`), and when `claimed_units_` is present
     and every expected unit has voted (`if (has_expected)` / `all_voted`),
     calls `RequestBlockRegistry::RetireLocked` to erase all `request_blocks_`
     rows and lifecycle maps for `(req_id, uuid)`.

  5. **Cancellation tombstone (`RequestBlockRegistry::CancelIfUnclaimed` in
     `tpu_sync/kv_cache/reshard/request_block_registry.cc`):**
     Returns `true` immediately if already in `cancelled_`
     (`if (cancelled_.find(lifecycle_key) != cancelled_.end())`), returns
     `false` without modifying state if `claimed_` contains `(req_id, uuid)`
     (`if (claimed_.find(lifecycle_key) != claimed_.end())`), and otherwise
     records a tombstone in `cancelled_` while erasing any unclaimed
     `request_blocks_` and lifecycle entries
     (`cancelled_[lifecycle_key] = now + ttl_s_;` .. `return true;`).

  6. **TTL expiration (`RequestBlockRegistry::PurgeExpiredRequestBlocksLocked` /
     `RequestBlockRegistry::PurgeExpiredLifecycleLocked` in
     `tpu_sync/kv_cache/reshard/request_block_registry.cc`):**
     Sweeps expired registrations, claims, and tombstones (Part V,
     `purgeExpiredAll_inv`, `trace_single_stage_votes_retire_multi_owner_claim`,
     and `trace_split_ttl_purges_request_blocks_while_claim_active`).

  ## Lock scopes

  Every mutable map of `RequestBlockRegistry` (`request_blocks_`, `claimed_`,
  `claimed_units_`, `claimed_owners_`, `completed_units_`, and `cancelled_`) is
  `ABSL_GUARDED_BY(mu_)` (the controller mutex shared with `WorkUnitDirectory`).

  | Event | `mu_` scopes | Between them |
  |---|---|---|
  | `register` | 1 deciding `RequestBlockRegistry::mu_` scope (`PurgeExpiredRequestBlocksLocked`, `PurgeExpiredLifecycleLocked`, `cancelled_` / `claimed_` checks, `request_blocks_` insertion); when `pool_spans` is non-empty, `Register` takes `mu_` once beforehand only to snapshot `directory_->PoolManifestLocked(unit)` | the manifest snapshot touches none of the registry maps (`retired` in `RegGen` is a single-generation latch assuming no re-registration of the same `(req_id, uuid)` after `RetireLocked`) |
  | `lookupAndClaim`, `complete`, `release` | 1 `RequestBlockRegistry::mu_` scope (each runs `PurgeExpiredRequestBlocksLocked` and `PurgeExpiredLifecycleLocked` on entry) | — |
  | `cancelIfUnclaimed` | 1 `RequestBlockRegistry::mu_` scope (runs `PurgeExpiredLifecycleLocked` on entry) | — |
  | `abandonClaim` | 1 `RequestBlockRegistry::mu_` scope | — |

  ## Main results

  - `reachable_safe` (`Inv`): proves inductively across all reachable registry
    states that:
    1. `ClaimOwnersConsistent`: `claimedUnits.isSome ↔ claimOwners ≠ []`, and
       every claimed unit is registered in `registeredUnits`.
    2. `TombstoneClean` & `tombstone_blocks_register_and_claim`: when
       `cancelled = true`, `registeredUnits = []`, `claimedUnits = none`, and
       `claimOwners = []`, and any subsequent `register` or `lookupAndClaim`
       is rejected.
    3. `claimed_blocks_register_and_cancel`: while `claimedUnits = some units`,
       `register` is rejected and `cancelIfUnclaimed` cannot cancel or erase the
       registered producer blocks.
    4. `RetiredClean` & `PartialVotesNotAllVoted`: once `retired = true`,
       `registeredUnits = []`, `claimedUnits = none`, and `claimOwners = []`;
       while `claimedUnits = some expected`, `allVoted expected completedUnits = false`,
       and `Complete` retires the generation only when every unit in `claimedUnits`
       has voted (`complete_retires_iff_all_claimed_voted`).
  - `purgeExpiredAll_inv`, `trace_single_stage_votes_retire_multi_owner_claim`,
    and `trace_split_ttl_purges_request_blocks_while_claim_active`.

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
  | `reg.cc` | `RequestBlockRegistry::Register` (`manifest = directory_->PoolManifestLocked(unit);`) | `:149` |
  | `reg.cc` | `RequestBlockRegistry::Register` (`registration.expires_at = now + ttl_s_;`) | `:263` |
  | `reg.cc` | `RequestBlockRegistry::Register` (`if (cancelled_.find(lifecycle_key) != cancelled_.end())`) | `:276-280` |
  | `reg.cc` | `RequestBlockRegistry::Register` (`if (claimed_.find(lifecycle_key) != claimed_.end())`) | `:281-285` |
  | `reg.cc` | `RequestBlockRegistry::Register` (`auto existing = request_blocks_.find(key);` .. `request_blocks_[key] = std::move(registration);`) | `:286-298` |
  | `reg.cc` | `RequestBlockRegistry::Release` | `:302-315` |
  | `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
  | `reg.cc` | `RequestBlockRegistry::Complete` (`if (!has_expected && has_registration &&`) | `:335-341` |
  | `reg.cc` | `RequestBlockRegistry::Complete` (`if (!registration_matches && !unit_expected)`) | `:346-350` |
  | `reg.cc` | `RequestBlockRegistry::Complete` (`if (has_expected)`) | `:358-369` |
  | `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` | `:373-405` |
  | `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` (`if (cancelled_.find(lifecycle_key) != cancelled_.end())`) | `:385-387` |
  | `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` (`if (claimed_.find(lifecycle_key) != claimed_.end())`) | `:388-390` |
  | `reg.cc` | `RequestBlockRegistry::CancelIfUnclaimed` (`cancelled_[lifecycle_key] = now + ttl_s_;` .. `return true;`) | `:391-404` |
  | `reg.cc` | `RequestBlockRegistry::Status` | `:407-436` |
  | `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-491` |
  | `reg.cc` | `RequestBlockRegistry::LookupAndClaim` (`if (cancelled_.find(lifecycle_key) != cancelled_.end())`) | `:448-452` |
  | `reg.cc` | `RequestBlockRegistry::LookupAndClaim` (`if (existing_units_it != claimed_units_.end() &&`) | `:455-460` |
  | `reg.cc` | `RequestBlockRegistry::LookupAndClaim` (`for (const RaidenId& unit : units)`) | `:462-470` |
  | `reg.cc` | `RequestBlockRegistry::LookupAndClaim` (`claimed_[lifecycle_key] = now + ttl_s_;` .. `claimed_owners_[lifecycle_key].insert(claim_owner);`) | `:473-475` |
  | `reg.cc` | `RequestBlockRegistry::LookupAndClaim` (`if (completion_it != completed_units_.end())`) | `:477-489` |
  | `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:493-509` |
  | `reg.cc` | `RequestBlockRegistry::AbandonClaim` (`if (owner_it->second.empty())`) | `:503-507` |
  | `reshard_service_test.cc` | `ReshardStackTest.CancelTombstoneBlocksLateRegistration` | `:933-961` |
  | `reshard_service_test.cc` | `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle` | `:963-1027` |
  | `reshard_service_test.cc` | `ReshardStackTest.CompletionVotesRetireAfterClaim` | `:1029-1052` |
  | `reshard_service_test.cc` | `ReshardStackTest.TtlPurgesExpiredRegistrations` | `:1054-1062` |
  | `reshard_service_test.cc` | `ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim` | `:1278-1303` |
  | `reshard_service_test.cc` | `ReshardStackTest.AbandonedStageLeavesTheSiblingClaimInPlace` | `:1305-1332` |
  | `reshard_service_test.cc` | `ReshardStackTest.AbandoningTheLastStageFreesTheClaim` | `:1334-1361` |
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
  /-- Monotonic flag recording whether `RequestBlockRegistry::RetireLocked`
  has fired to retire the generation. -/
  retired : Bool
  deriving DecidableEq, Repr

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
`expected_it->second.find(unit) != expected_it->second.end()`). -/
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
    -- or already claimed (`claimed_.find(lifecycle_key) != claimed_.end()`), or
    -- if generation was already retired.
    if s.cancelled || s.claimedUnits.isSome || s.retired then
      none
    else
      match s.lookupUnit unit with
      | some curNonEmpty =>
        -- Conflicting payload is rejected (`ExistingRegistrationMatches`);
        -- identical payload is an idempotent refresh (`existing->second.expires_at = now + ttl_s_;`).
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
and `claimOwners = []`. -/
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

/-! ## Part IV — Concrete `decide` Traces (`reshard_service_test.cc` Parity) and Mutants -/

/-- Happy-path 2-unit producer registration, multi-owner pipelined claim,
single-owner abandon (claim survives), and 2-unit completion voting that retires
only on the second vote (`RequestBlockRegistry` in `request_block_registry.cc`,
`ReshardStackTest.CompletionVotesRetireAfterClaim`,
`ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim`, and
`ReshardStackTest.AbandonedStageLeavesTheSiblingClaimInPlace` in
`reshard_service_test.cc`). -/
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

/-- Replay of `ReshardStackTest.CompletionVotesRetireAfterClaim`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`):
After the first of two claimed units votes (`complete 0`), the generation is
still `.claimed` and `retired = false` (`"0"` rows retired); when the second
unit votes (`complete 1`), `RetireLocked` fires (`retired = true`,
`registeredUnits = []`, `"2"` rows retired). -/
theorem trace_partial_vote_does_not_retire :
    (RegGen.sys.run (registryHappyTrace.take 6)).map
      (fun s => (s.status, s.claimedUnits, s.completedUnits, s.retired)) =
      some (.claimed, some [0, 1], [0], false) := by
  decide

/-- Replay of `ReshardStackTest.CancelTombstoneBlocksLateRegistration`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`) and
`RequestBlockRegistry::Register`, `CancelIfUnclaimed`, and `LookupAndClaim`:
1. Cancelling an unclaimed generation before any producer registers or after
   partial registration records a `.cancelled` tombstone.
2. Any late producer `register` or `lookupAndClaim` against the cancelled
   `(req_id, uuid)` is rejected (`none`). -/
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

/-- Replay of `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`):
1. Never registered (`RegGen.init`) reads as `.unknown`.
2. Registered and unclaimed (`[.register 0 true]`) reads as `.registered`.
3. Release-only cancel (`[.register 0 true, .cancelIfUnclaimed]`) erases the row
   and reads as `.cancelled`.
4. Claimed registration (`[.register 0 true, .register 1 true,
   .lookupAndClaim 101 [0, 1]]`) reads as `.claimed` and refuses cancellation
   (staying `.claimed` with `cancelled = false`). -/
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

/-- Replay of `ReshardStackTest.PipelinedDestinationStagesShareOneRequestClaim`,
`ReshardStackTest.AbandonedStageLeavesTheSiblingClaimInPlace`, and
`ReshardStackTest.AbandoningTheLastStageFreesTheClaim`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`):
1. Stage 0 (`100`) and Stage 1 (`101`) share one request claim on `[0, 1]`; when
   Stage 1 abandons its claim (`abandonClaim 101`), Stage 0's claim remains in
   place (`status = .claimed`, `cancelIfUnclaimed` refused with
   `cancelled = false`), and Stage 1 can rejoin (`lookupAndClaim 101 [0, 1]`).
2. When both Stage 0 (`100`) and Stage 1 (`101`) abandon their claims, the last
   stage leaving frees the claim (`status = .registered`), allowing
   `cancelIfUnclaimed` to succeed (`status = .cancelled`) and blocking a late
   Stage 0 retry. -/
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

/-- Mutant 1: If `Complete` retired on the *first* unit vote instead of checking
`all_voted` across `claimed_units_` (`RequestBlockRegistry::Complete`
`if (has_expected)`), `unit 0` finishing its push would prematurely retire
`unit 1`'s registered producer blocks while `unit 1` is still transferring. -/
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

/-- Mutant 2: If `AbandonClaim` erased `claimed_` unconditionally without
checking `claimed_owners_.empty()` (`RequestBlockRegistry::AbandonClaim`
`if (owner_it->second.empty())`), one pipelined stage abandoning its claim would
strip the claim out from under a concurrent stage `owner = 102`, allowing
`CancelIfUnclaimed` to cancel the blocks mid-transfer. -/
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

/-- **Multi-owner completion voting (`RequestBlockRegistry::Completions` and `claimed_owners_`
in `request_block_registry.h`, `RequestBlockRegistry::Complete` in
`request_block_registry.cc`)**:
Although `claimed_owners_` tracks multiple pipelined `claim_owner` strings for a
shared source snapshot, `completed_units_[(req_id, uuid)].units` is keyed only
by `(req_id, uuid)` (not `(req_id, uuid, claim_owner)`), and `Complete` calls
`RetireLocked` as soon as every unit in `claimed_units_` votes once. Thus when
stage `101` and stage `102` both hold active claims (`claimOwners = [102, 101]`),
stage `101`'s source units voting `.complete 0` and `.complete 1` immediately
retires the generation and erases `102`'s claim and source blocks. -/
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
`t_reg + ttl_s_` at `RequestBlockRegistry::Register`
`registration.expires_at = now + ttl_s_;`). -/
def RegGen.purgeExpiredRequestBlocks (s : RegGen) : RegGen :=
  { s with registeredUnits := [] }

/-- `RequestBlockRegistry::PurgeExpiredLifecycleLocked`
(`tpu_sync/kv_cache/reshard/request_block_registry.cc`):
erases `claimed_`, `cancelled_`, `completed_units_`, `claimed_owners_`, and
`claimed_units_` entries whose lifecycle `expires_at <= now`. -/
def RegGen.purgeExpiredLifecycle (s : RegGen) : RegGen :=
  { s with
    claimedUnits := none
    claimOwners := []
    completedUnits := []
    cancelled := false }

/-- Combined TTL purge when `now` advances past both `registration.expires_at`
and lifecycle `expires_at` (`RequestBlockRegistry::PurgeExpiredRequestBlocksLocked`
and `PurgeExpiredLifecycleLocked` in `request_block_registry.cc`,
`ReshardStackTest.RequestBlockStatusProbeTracksLifecycle` and
`ReshardStackTest.TtlPurgesExpiredRegistrations` in `reshard_service_test.cc`). -/
def RegGen.purgeExpiredAll (s : RegGen) : RegGen :=
  s.purgeExpiredRequestBlocks.purgeExpiredLifecycle

theorem purgeExpiredAll_inv (s : RegGen) : Inv s.purgeExpiredAll where
  claimConsistent := ⟨rfl, by intro units h; cases h⟩
  tombstoneClean := by intro h; cases h
  retiredClean := by intro _; exact ⟨rfl, rfl, rfl⟩
  partialVotes := by intro expected h; cases h

/-- Replay of `ReshardStackTest.TtlPurgesExpiredRegistrations` and the TTL
tombstone lapse in `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle`
(`tpu_sync/kv_cache/reshard/reshard_service_test.cc`):
1. `ReshardStackTest.TtlPurgesExpiredRegistrations`: unit `0` registers, `now_`
   advances past the 600 s TTL (`purgeExpiredAll`), and a subsequent
   `lookupAndClaim 101 [0]` fails (`none`, `"Missing producer block registration"`).
2. `ReshardStackTest.RequestBlockStatusProbeTracksLifecycle`: after
   `cancelIfUnclaimed` records a `.cancelled` tombstone, advancing `now_` past
   the TTL (`purgeExpiredAll`) purges the tombstone so `status` reads as
   `.unknown`. -/
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

/-- **Split TTL expiration (`RequestBlockRegistry::PurgeExpiredRequestBlocksLocked`,
`RequestBlockRegistry::Register` `registration.expires_at = now + ttl_s_;`, and
`RequestBlockRegistry::LookupAndClaim` `claimed_[lifecycle_key] = now + ttl_s_;`
in `request_block_registry.cc`)**:
`Register` at time `t_0` sets `registration.expires_at = t_0 + ttl_s_` on
`request_blocks_[(req_id, unit)]`, whereas `LookupAndClaim` at time `t_1 > t_0`
sets `claimed_[(req_id, uuid)] = t_1 + ttl_s_` **without** refreshing
`registration.expires_at` on the claimed `request_blocks_` rows.
In the window `[t_0 + ttl_s_, t_1 + ttl_s_)`, `PurgeExpiredRequestBlocksLocked`
erases `request_blocks_[(req_id, unit)]` while `claimed_` is still active
(`s_split = s.purgeExpiredRequestBlocks`):
1. `s_split.status` still reports `.claimed` (`claimedUnits = some [0]`), but
   `s_split.registeredUnits = []` (violating `ClaimOwnersConsistent`), and
2. A second pipelined stage calling `lookupAndClaim 102 [0]` in that window fails
   (`none`) with `"Missing producer block registration"`
   (`RequestBlockRegistry::LookupAndClaim` `for (const RaidenId& unit : units)`). -/
theorem trace_split_ttl_purges_request_blocks_while_claim_active :
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

/-- Executable negation of `Inv`. -/
def violates (s : RegGen) : Bool :=
  (s.claimedUnits.isSome != !s.claimOwners.isEmpty) ||
  (match s.claimedUnits with
   | none => false
   | some units =>
     units.isEmpty ||
     units.any (fun u => !(s.lookupUnit u).isSome) ||
     allVoted units s.completedUnits) ||
  (s.cancelled && (!s.registeredUnits.isEmpty || s.claimedUnits.isSome ||
                   !s.claimOwners.isEmpty || !s.completedUnits.isEmpty)) ||
  (s.retired && (!s.registeredUnits.isEmpty || s.claimedUnits.isSome || !s.claimOwners.isEmpty))

#guard ModelCheck.check RegGen.sys checkEvents violates 12 = .safe

end TpuSyncVerify.Transfer.Reshard.Registry


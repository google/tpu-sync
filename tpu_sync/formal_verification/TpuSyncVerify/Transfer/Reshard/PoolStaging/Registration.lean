import TpuSyncVerify.Transfer.Reshard.PoolStaging.ArenaLemmas

/-! # `PoolStaging`: receive registration lifecycle (`RegistrationPolicy`)

Part of `TpuSyncVerify.Transfer.Reshard.PoolStaging`. -/

namespace TpuSyncVerify.Transfer.Reshard.PoolStaging

open TpuSyncVerify

/-! ## Part III — Receive Registration Lifecycle (`PoolReshardRegisterRecv`)

This part models the interaction between `KVCacheManagerWithTransfer::PoolReshardRegisterRecv`,
`ReshardReceiveSession` (`CreateReshardReceiveSession`,
`ReshardReceiveSession::AcquireStagingLeases`, `ReshardReceiveSession::ReleaseStaging`),
and `PoolStagingStorage` across two registration policies. Only receive
registration is modelled; `PoolReshardPush` (send registration) has no
pending-uuid claim and no arm here (`PoolStaging.lean` §"Not modelled").
The policies:
- `RegistrationPolicy.reserveBeforeLease` (shipping): `PoolReshardRegisterRecv`
  claims `uuid` in `pending_pool_reshard_recvs_` under its `mu_`
  (`if (!pending_pool_reshard_recvs_.insert(plan.uuid()).second)`) before
  `CreateReshardReceiveSession` constructs the session and calls
  `AcquirePoolStagingLease`, releasing the partial lease
  (`session->ReleaseStaging();`) and dropping the claim
  (`absl::Cleanup clear_pending`) on lease failure.
- `RegistrationPolicy.leaseBeforeReserve` (mutant):
  `AcquirePoolStagingLease` runs before the uuid is claimed, so a rejected
  duplicate registration's destructor releases the active session's staging
  lease. -/

inductive RegistrationPolicy where
  | reserveBeforeLease
  | leaseBeforeReserve
  deriving DecidableEq, Repr

/-- A pending `PoolReshardRegisterRecv` call that has passed Phase 1 (the initial
`sessions_` check and, under `reserveBeforeLease`, the
`pending_pool_reshard_recvs_` claim) and is currently executing Phase 2
(`AcquireStagingLeases`) or Phase 3 (`sessions_` insertion). -/
structure InFlightArm where
  callId : Nat
  uuid : Nat
  blocks : List Nat
  leasedInPhase2 : Bool
  deriving DecidableEq, Repr

/-- State of the receive-registration + bounded-staging subsystem. -/
structure RegState where
  arena : Arena
  /-- UUIDs whose `ReshardReceiveSession` is in `sessions_`, paired with their
  required destination block IDs. -/
  activeSessions : List (Nat × List Nat)
  /-- Concurrent `PoolReshardRegisterRecv` calls in flight between Phase 1 and
  Phase 3. -/
  inFlightArms : List InFlightArm
  deriving DecidableEq, Repr

inductive RegEv where
  /-- Phase 1 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`): check if
  `uuid` is already active in `sessions_`; under `.reserveBeforeLease`, also
  claim `uuid` in `pending_pool_reshard_recvs_` against concurrent arms
  immediately. -/
  | beginArm (callId uuid : Nat) (blocks : List Nat)
  /-- Phase 2 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`,
  `CreateReshardReceiveSession`, `ReshardReceiveSession::AcquireStagingLeases`): call
  `AcquirePoolStagingLease(uuid, blocks)` outside `mu_`. -/
  | acquireArmLease (callId : Nat)
  /-- Phase 3 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`): finish registration
  and insert into `sessions_`. Under `.leaseBeforeReserve`, if `uuid` is already
  in `activeSessions`, the registration is refused (`AlreadyExistsError`) and
  the temporary `ReshardReceiveSession` destructor runs
  `ReleasePoolStagingLeases(uuid)` (`ReshardReceiveSession::~ReshardReceiveSession`,
  `ReshardReceiveSession::ReleaseStagingLocked`). -/
  | finishArm (callId : Nat)
  /-- Normal session completion (`FinishPoolH2d` / `UnregisterActivePlan`, then
  `RetireSettledSessionLocked` and `CompleteReadWithDetails`'s unlocked
  post-poll `ReleasePoolStagingLeases(uuid)`): releases `uuid`'s staging lease
  and removes `uuid` from `activeSessions`. Assumes fresh UUIDs across poll
  retirement (because `RetireSettledSessionLocked` and the
  `pending_pool_reshard_recvs_` filter run under the same `mu_` hold while
  `ReleasePoolStagingLeases(uuid)` runs after releasing `mu_`, a `uuid` being
  retired by `CompleteReadWithDetails` must not be concurrently re-registered
  during the unlocked cleanup). -/
  | settleSession (uuid : Nat)
  deriving DecidableEq, Repr

def RegState.init (numSlots : Nat) : RegState :=
  { arena := Arena.init numSlots
  , activeSessions := []
  , inFlightArms := [] }

def RegState.hasActiveUuid (s : RegState) (uuid : Nat) : Bool :=
  s.activeSessions.any (fun p => p.1 == uuid)

def RegState.hasInFlightUuid (s : RegState) (uuid : Nat) : Bool :=
  s.inFlightArms.any (fun a => a.uuid == uuid)

def RegState.findArm (s : RegState) (callId : Nat) : Option InFlightArm :=
  s.inFlightArms.find? (fun a => a.callId == callId)

def RegState.removeArm (s : RegState) (callId : Nat) : List InFlightArm :=
  s.inFlightArms.filter (fun a => a.callId != callId)

def RegState.step (policy : RegistrationPolicy) (s : RegState) (e : RegEv) : Option RegState :=
  match e with
  | .beginArm callId uuid blocks =>
    if (s.findArm callId).isSome then none
    else if s.hasActiveUuid uuid then none
    else if policy == .reserveBeforeLease && s.hasInFlightUuid uuid then none
    else
      some { s with
        inFlightArms := s.inFlightArms ++ [⟨callId, uuid, blocks, false⟩] }
  | .acquireArmLease callId =>
    match s.findArm callId with
    | none => none
    | some arm =>
      if arm.leasedInPhase2 then none
      else
        match s.arena.acquireLease arm.uuid arm.blocks with
        | none =>
          -- Phase 2 `AcquireStagingLeases` failed (`ResourceExhausted`):
          -- `CreateReshardReceiveSession` releases any partial lease and
          -- `PoolReshardRegisterRecv`'s cleanup drops the pending claim
          -- (`CreateReshardReceiveSession`, `ReshardReceiveSession::AcquireStagingLeases`, `ReshardReceiveSession::ReleaseStagingLocked`).
          some { s with
            arena := s.arena.releaseLease arm.uuid
            inFlightArms := s.removeArm callId }
        | some arena' =>
          some { s with
            arena := arena'
            inFlightArms := s.inFlightArms.map (fun a =>
              if a.callId == callId then { a with leasedInPhase2 := true } else a) }
  | .finishArm callId =>
    match s.findArm callId with
    | none => none
    | some arm =>
      if !arm.leasedInPhase2 then none
      else if s.hasActiveUuid arm.uuid then
        -- Under `.leaseBeforeReserve`, the registration is refused with
        -- `AlreadyExistsError` after the factory already acquired the lease, and
        -- the temporary
        -- `ReshardReceiveSession` destructor calls
        -- `ReleasePoolStagingLeases(arm.uuid)`.
        some { s with
          arena := s.arena.releaseLease arm.uuid
          inFlightArms := s.removeArm callId }
      else
        some { s with
          activeSessions := s.activeSessions ++ [(arm.uuid, arm.blocks)]
          inFlightArms := s.removeArm callId }
  | .settleSession uuid =>
    if !s.hasActiveUuid uuid then none
    else
      some { s with
        arena := s.arena.releaseLease uuid
        activeSessions := s.activeSessions.filter (fun p => p.1 != uuid) }

def RegState.sys (policy : RegistrationPolicy) (numSlots : Nat) :
    System RegState RegEv where
  init := RegState.init numSlots
  step := RegState.step policy

/-- **Safety Predicate (`ActiveSessionsHoldLeases`)**:
Every registered active session `(uuid, blocks) ∈ s.activeSessions` holds a
live lease in `s.arena.leases` that maps every block `b ∈ blocks` to an
allocated staging slot (`(s.arena.resolveBlockSlot uuid b).isSome`). -/
def ActiveSessionsHoldLeases (s : RegState) : Bool :=
  s.activeSessions.all (fun (uuid, blocks) =>
    blocks.all (fun b => (s.arena.resolveBlockSlot uuid b).isSome))

/-- **Constructive counterexample trace for the `leaseBeforeReserve` mutant
(`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`, `CreateReshardReceiveSession`, `ReshardReceiveSession::ReleaseStagingLocked`)**:
Under the unreserved mutant `.leaseBeforeReserve`:
1. Call 1 (`uuid = 7`, `blocks = [5]`) and Call 2 (duplicate `uuid = 7`,
   `blocks = [5]`) both pass Phase 1 (`beginArm`) before either reaches Phase 3.
2. Call 1 acquires slot `0` for `(uuid = 7, block = 5)` in Phase 2 and completes
   Phase 3 (`finishArm 1`), becoming an active receive session.
3. Call 2 executes Phase 2 (`acquireArmLease 2`), which succeeds idempotently on
   `uuid = 7`'s existing lease (`KVCacheManagerBase::AcquirePoolStagingLease`), and then
   reaches Phase 3 (`finishArm 2`). The registration rejects Call 2
   with `AlreadyExistsError`, and Call 2's `~ReshardReceiveSession` runs
   `ReleasePoolStagingLeases(7)` — freeing Call 1's slot `0` back to `freeSlots`
   while Call 1 (`uuid = 7`) is still in `activeSessions`!
4. Call 3 (`uuid = 8`, `blocks = [12]`) then arms and acquires **slot 0**, the
   exact physical staging slot that Call 1 (`uuid = 7`) was using. -/
def duplicateUuidCounterexampleEvents : List RegEv :=
  [ .beginArm 1 7 [5]
  , .beginArm 2 7 [5]
  , .acquireArmLease 1
  , .finishArm 1
  , .acquireArmLease 2
  , .finishArm 2
  , .beginArm 3 8 [12]
  , .acquireArmLease 3
  , .finishArm 3 ]

theorem trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot :
    ((RegState.sys .leaseBeforeReserve 2).run duplicateUuidCounterexampleEvents).map
      (fun s =>
        ( s.hasActiveUuid 7
        , s.arena.lookupLease 7
        , ActiveSessionsHoldLeases s
        , s.arena.resolveBlockSlot 8 12 )) =
      some (true, none, false, some 0) := by
  decide

/-- Under `RegistrationPolicy.reserveBeforeLease`
(`PoolStagingRecvRegistrationTest.ConcurrentDuplicatePoolRecvRegistrationKeepsWinnerStagingLease`),
the duplicate-UUID race is rejected at step 2 (`.beginArm 2 7 [5]`) because
`uuid = 7` is already claimed by Call 1. -/
theorem duplicate_uuid_blocked_by_reserveBeforeLease :
    (RegState.sys .reserveBeforeLease 2).run duplicateUuidCounterexampleEvents = none := by
  decide


end TpuSyncVerify.Transfer.Reshard.PoolStaging

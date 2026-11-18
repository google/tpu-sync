import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.UuidTable.Invariant

/-! # `UuidTable`: replay of C++ tests, bounded searches, mutant

Part of `TpuSyncVerify.Transfer.PrefillDecode.UuidTable`; the module docstring and
source map live in `UuidTable.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.UuidTable

open TpuSyncVerify.Transfer (Lifecycle)
open TpuSyncVerify.Transfer.PrefillDecode (Recv Send)

/-! ## Concrete traces (`SendLifecycleTest`, `RecvDrainTest`, `RecvLifecycleTest`, `ControlHandshakeTest`) -/

structure RecvSummary where
  reqId : ReqId
  gen : Nat
  draining : Bool
  done : Bool
  deriving Repr, DecidableEq

def recvSummary (s : State) (uuid : Uuid) : Option RecvSummary :=
  match s.recvTable[uuid]? with
  | some (some e) =>
    some { reqId := e.reqId, gen := e.gen,
           draining := e.recv.life.draining, done := e.recv.life.done }
  | _ => none

/-- `RecvDrainTest.DuplicateUuidIsRejectedUntilExpiredReceiveDrains`
(the model's `numSlots = 4`, the test's `kSlots = 2`, so its `free_slots`
assertions read `kSlots - 1` and `kSlots` where this trace says 3 and 4):
1. Register `"old"` (`reqId = 0`) at `uuid = 0`, issue layer 0 H2D copy
   (`h2dBegin`, `h2dIssue true`), and expire its deadline (`.cancel`).
2. While `"old"` is draining (`draining = true, done = false`), `.sweepRecv 0`
   and `.registerRecv 1 0` (`"retry"`, `reqId = 1`) are both rejected (`none`),
   and `"old"` keeps `uuid = 0` (`gen = 0`) and its staging slot
   (`freeSlots = 3`).
3. Once `"old"`'s H2D copy finishes (`h2dReady`, `h2dDone true`) and
   `.sweepRecv 0` runs, `"old"` is reported in `failedRecving = [0]`,
   `freeSlots` returns to `4`, and `.registerRecv 1 0` (`"retry"`, `gen = 1`)
   succeeds; `"retry"` then expires and is swept (`.recvStep 0 .cancel`,
   `.sweepRecv 0`) into `failedRecving = [0, 1]` with `recvTable[0]? = some none`
   and `freeSlots = 4`. -/
theorem trace_duplicate_uuid_rejected_until_drained :
    let cfg : Config := { numLayers := 1, numSlots := 4, numUuids := 1 }
    let pre : List Ev :=
      [.registerRecv 0 0,
       .recvStep 0 .h2dBegin, .recvStep 0 (.h2dIssue true),
       .recvStep 0 .cancel]
    ((sys cfg).run pre).map (fun s => (s.freeSlots, recvSummary s 0)) =
      some (3, some { reqId := 0, gen := 0, draining := true, done := false }) ∧
    (sys cfg).run (pre ++ [.sweepRecv 0]) = none ∧
    (sys cfg).run (pre ++ [.registerRecv 1 0]) = none ∧
    ((sys cfg).run (pre ++
      [.recvStep 0 .h2dReady, .recvStep 0 (.h2dDone true),
       .sweepRecv 0,
       .registerRecv 1 0,
       .recvStep 0 .cancel, .sweepRecv 0])).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving, s.recvTable[0]?)) =
      some (4, [], [0, 1], some none) := by
  decide

/-- `ControlHandshakeTest.DuplicateReceiveDoesNotReplaceOrLeakFirstRead`:
1. `startRead` for `"first"` (`reqId = 0, uuid = 0`) allocates 1 slot
   (`freeSlots = 3`).
2. `startRead` for `"duplicate"` (`reqId = 1, uuid = 0`) sees the live incumbent
   with a different `reqId`: it leaves `"first"` at `recvTable[0]` (`reqId = 0`),
   allocates no extra slot (`freeSlots = 3`), and records `"duplicate"` in
   `failedRecving = [1]`.
3. When `"first"`'s handshake fails (`pullReply false`) and `.sweepRecv 0` runs,
   both `"first"` and `"duplicate"` are in `failedRecving = [1, 0]` and
   `freeSlots = 4`. -/
theorem trace_duplicate_receive_different_req_id :
    let cfg : Config := { numLayers := 1, numSlots := 4, numUuids := 1 }
    let pre : List Ev := [.startRead 0 0, .startRead 1 0]
    ((sys cfg).run pre).map
      (fun s => (s.freeSlots, s.failedRecving,
                 (s.recvTable[0]?).bind (fun o => o.map RecvEntry.reqId))) =
      some (3, [1], some 0) ∧
    ((sys cfg).run (pre ++ [.recvStep 0 (.pullReply false), .sweepRecv 0])).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving, s.recvTable[0]?)) =
      some (4, [], [1, 0], some none) := by
  decide

/-- `ControlHandshakeTest.RepeatedReceiveAnnouncementIsIdempotent`:
re-delivering `startRead 0 0` with the **same** `(reqId = 0, uuid = 0)` while
the first is still in flight is a no-op (`failedRecving = []`, `freeSlots = 3`),
and when the handshake eventually fails and is swept, `reqId = 0` is reported
once in `failedRecving = [0]`. -/
theorem trace_repeated_receive_same_req_id_idempotent :
    let cfg : Config := { numLayers := 1, numSlots := 4, numUuids := 1 }
    let pre : List Ev := [.startRead 0 0, .startRead 0 0]
    ((sys cfg).run pre).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving)) =
      some (3, [], []) ∧
    ((sys cfg).run (pre ++ [.recvStep 0 (.pullReply false), .sweepRecv 0])).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving, s.recvTable[0]?)) =
      some (4, [], [0], some none) := by
  decide

/-- `SendLifecycleTest.DuplicateRegistrationCannotReplaceLiveOffer`:
`notifyForRead 0 0` (`"first"`, `reqId = 0, uuid = 0`) registers a live send
offer. A duplicate `notifyForRead 1 0` (`"replacement"`, `reqId = 1, uuid = 0`)
is rejected (`none`; in C++ a `std::runtime_error`, the test's `EXPECT_THROW`). When `"first"` times out (`.cancel`) and is swept
(`.sweepSend 0`), `"first"` (`reqId = 0`) is reported in `failedRecving = [0]`
and `uuid = 0` becomes free for a subsequent `notifyForRead 1 0` (the
re-registration step is model-only; the C++ test stops after the sweep). -/
theorem trace_duplicate_send_cannot_replace_live_offer :
    let cfg : Config := { numLayers := 1, numSlots := 4, numUuids := 1 }
    ((sys cfg).run [.notifyForRead 0 0]).map
      (fun s => (s.sendTable[0]?).bind (fun o => o.map SendEntry.reqId)) =
      some (some 0) ∧
    ((sys cfg).run [.notifyForRead 0 0, .notifyForRead 1 0]).isNone = true ∧
    ((sys cfg).run
      [.notifyForRead 0 0, .sendStep 0 .cancel, .sweepSend 0, .notifyForRead 1 0]).map
      (fun s => (s.failedRecving,
                 (s.sendTable[0]?).bind (fun o => o.map (fun e => (e.reqId, e.gen))))) =
      some ([0], some (1, 1)) := by
  decide

/-- Inline retirement on `StartRead` (`KVCacheManagerWithTransfer::StartRead(remote_endpoint)`): when an incumbent at
`uuid = 0` has already settled (`done = true`), a new `startRead 1 0` retires
the settled incumbent into `doneRecving` / `failedRecving` inline without
needing an intervening `sweepRecv 0`, and seats the new session (`gen = 1`)
without leaking a staging slot. -/
theorem trace_start_read_retires_settled_incumbent_inline :
    let cfg : Config := { numLayers := 1, numSlots := 4, numUuids := 1 }
    let evs : List Ev :=
      [.startRead 0 0,
       .recvStep 0 (.pullReply false),
       .startRead 1 0]
    ((sys cfg).run evs).map
      (fun s => (s.freeSlots, s.failedRecving,
                 (s.recvTable[0]?).bind (fun o => o.map (fun e => (e.reqId, e.gen))))) =
      some (3, [0], some (1, 1)) := by
  decide

/-- One layer, four slots, one uuid. -/
def oneUuid : Config := { numLayers := 1, numSlots := 4, numUuids := 1 }

/-- One `sessions_` map for both directions. A `notifyForRead` under a uuid
that a receive holds — live, or settled but not yet retired — is refused
(`sessions_.try_emplace` fails, the thrown `std::runtime_error`), and
`NotifyForRead` retires nothing. A `startRead` under a uuid that a live send
offer holds is rejected into `failedRecving` (`RetireSettledSessionLocked`
returns false and `recv_incumbent == nullptr`); once the offer has settled,
`startRead` retires it inline into `doneSending` / `failedRecving` and seats
the read. -/
theorem trace_one_map_for_both_directions :
    (sys oneUuid).run [.startRead 0 0, .notifyForRead 1 0] = none ∧
    (sys oneUuid).run [.startRead 0 0, .recvStep 0 (.pullReply false), .notifyForRead 1 0] = none ∧
    ((sys oneUuid).run [.startRead 0 0, .recvStep 0 (.pullReply false), .sweepRecv 0,
                    .notifyForRead 1 0]).map
      (fun s => (s.sendTable[0]?).bind (fun o => o.map SendEntry.reqId)) = some (some 1) := by
  decide

/-- The other direction of `trace_one_map_for_both_directions`: a read under a
uuid a send offer holds. -/
theorem trace_start_read_under_send_offer :
    ((sys oneUuid).run [.notifyForRead 0 0, .startRead 1 0]).map
      (fun s => (s.freeSlots, s.failedRecving, (s.recvTable[0]?).map Option.isNone,
                 (s.sendTable[0]?).bind (fun o => o.map SendEntry.reqId))) =
      some (4, [1], some true, some 0) ∧
    (sys oneUuid).run [.notifyForRead 0 0, .registerRecv 1 0] = none ∧
    ((sys oneUuid).run [.notifyForRead 0 0, .sendStep 0 .cancel, .startRead 1 0]).map
      (fun s => (s.freeSlots, s.doneSending, s.failedRecving, (s.sendTable[0]?).map Option.isNone)) =
      some (3, [], [0], some true) ∧
    ((sys oneUuid).run [.notifyForRead 0 0, .sendStep 0 .cancel, .startRead 1 0]).map
      (fun s => (s.recvTable[0]?).bind (fun o => o.map (fun e => (e.reqId, e.gen)))) =
      some (some (1, 1)) := by
  decide

/-- No layers, four slots, one uuid: a receive that is ready to complete with
nothing in flight from the moment it is seated. -/
def zeroLayers : Config := { numLayers := 0, numSlots := 4, numUuids := 1 }

/-- `RecvLifecycleTest.RegisterRecvFinishesAReadyIncumbentBeforeReplacingIt` and
`RecvLifecycleTest.RedeliveredStartReadLeavesAReadyReceiveToCompleteRead`.
Both tests seat a one-layer incumbent with `RecvTestManager::AddRecv` and make
it ready to complete with nothing in flight by reporting its blocks before any
copy is issued (`ReceiveBlocks`). `Receive.lean`'s A4 excludes that order on a
layer, so the replay runs on `zeroLayers`, whose receive is in the same
session condition (ready, not draining, `inFlight = 0`) as soon as it is
seated.

1. `registerRecv 1 0` (`"retry"`) finishes the ready incumbent `"old"`
   (`recv->Finish()` in `EmplaceRecvSessionLocked`), which settles it on the
   spot, retires it into `doneRecving = [0]` and seats the replacement
   (`gen = 1`) in the same call: the test's `recv->Done()`,
   `replacement != incumbent`, and `"old"` in `completions.received`. The
   model's `freeSlots` reads `numSlots - 1` (`3` of `4`) where the test reads all
   of `kSlots` (`kSlots = 2`): `registerRecv` takes a slot for the replacement,
   production `RegisterRecv` takes none (`UuidTable.lean`, rule 2).
2. A re-delivered `startRead 0 0` of the incumbent's own request changes
   nothing (the incumbent is neither finished nor replaced and keeps its
   slot); the poll then finishes it (`pollReady`), the sweep reports it in
   `doneRecving = [0]` and returns the slot, and a second sweep finds no entry
   (the test's empty `repeated`). -/
theorem trace_register_recv_replaces_ready_incumbent :
    ((sys zeroLayers).run [.registerRecv 0 0]).map (fun s => (s.freeSlots, recvSummary s 0)) =
      some (3, some { reqId := 0, gen := 0, draining := false, done := false }) ∧
    ((sys zeroLayers).run [.registerRecv 0 0, .registerRecv 1 0]).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving, recvSummary s 0)) =
      some (3, [0], [], some { reqId := 1, gen := 1, draining := false, done := false }) ∧
    (sys zeroLayers).run [.registerRecv 0 0, .startRead 0 0] =
      (sys zeroLayers).run [.registerRecv 0 0] ∧
    ((sys zeroLayers).run [.registerRecv 0 0, .startRead 0 0, .recvStep 0 .pollReady, .sweepRecv 0]).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving, recvSummary s 0)) =
      some (4, [0], [], none) ∧
    (sys zeroLayers).run
      [.registerRecv 0 0, .startRead 0 0, .recvStep 0 .pollReady, .sweepRecv 0, .sweepRecv 0] = none := by
  decide

/-- Model-only: the same two calls against a ready incumbent that still has
an operation in flight. On one layer the model's ready, non-draining
incumbent is the poll window of `ReceivePoll.lean` — every copy's future
ready, the last callback not yet run (no C++ test holds a receive there).
`registerRecv` finishes the incumbent (`recv->Finish()` in
`EmplaceRecvSessionLocked`) but cannot retire or replace it, since the
callback's op keeps it from settling (`draining = true`, `done = false`, same
`(reqId, gen)`, slot kept), and the call is refused with the incumbent left
draining; a re-delivered `startRead` of the same request neither finishes nor
replaces it. The callback then settles the incumbent and the sweep reports it
done. -/
theorem trace_register_recv_finishes_ready_incumbent :
    let pre : List Ev :=
      [.registerRecv 0 0, .recvStep 0 .h2dBegin, .recvStep 0 (.h2dIssue true),
       .recvStep 0 .netAccount, .recvStep 0 .h2dReady]
    ((sys oneUuid).run pre).map (fun s => (s.freeSlots, recvSummary s 0)) =
      some (3, some { reqId := 0, gen := 0, draining := false, done := false }) ∧
    ((sys oneUuid).run (pre ++ [.registerRecv 1 0])).map (fun s => (s.freeSlots, recvSummary s 0)) =
      some (3, some { reqId := 0, gen := 0, draining := true, done := false }) ∧
    ((sys oneUuid).run (pre ++ [.registerRecv 1 0, .recvStep 0 (.h2dDone true), .sweepRecv 0])).map
      (fun s => (s.freeSlots, s.doneRecving, s.failedRecving, (s.recvTable[0]?).map Option.isNone)) =
      some (4, [0], [], some true) ∧
    (let pre' : List Ev :=
      [.startRead 0 0, .recvStep 0 (.pullReply true), .recvStep 0 .h2dBegin,
       .recvStep 0 (.h2dIssue true), .recvStep 0 .netAccount, .recvStep 0 .h2dReady]
     (sys oneUuid).run (pre' ++ [.startRead 0 0]) = (sys oneUuid).run pre' ∧
     ((sys oneUuid).run (pre' ++ [.startRead 0 0, .recvStep 0 (.h2dDone true), .sweepRecv 0])).map
       (fun s => (s.freeSlots, s.doneRecving, s.failedRecving)) = some (4, [0], [])) := by
  decide

/-! ## Bounded model checks and mutant -/

def checkEvents : List Ev :=
  [.startRead 0 0, .startRead 1 0,
   .registerRecv 0 0, .registerRecv 1 0,
   .notifyForRead 0 0, .sendStep 0 .cancel, .sweepSend 0,
   .recvStep 0 (.pullReply true), .recvStep 0 (.pullReply false),
   .recvStep 0 .h2dBegin, .recvStep 0 (.h2dIssue true),
   .recvStep 0 .h2dReady, .recvStep 0 (.h2dDone true),
   .recvStep 0 .cancel, .sweepRecv 0]

/-- Executable check for staging-slot conservation (`freeSlots + activeStaging = numSlots`). -/
def violatesSlotConservation (s : State) : Bool :=
  s.freeSlots + activeStaging s.recvTable != s.cfg.numSlots

/-- The two tables stand for the one `sessions_` map: no uuid is held in both. -/
def violatesExclusive (s : State) : Bool :=
  (List.range s.cfg.numUuids).any fun u =>
    match s.recvTable[u]?, s.sendTable[u]? with
    | some (some _), some (some _) => true
    | _, _ => false

-- Bounded only: no theorem states table exclusivity, and the slot search is
-- the baseline the mutant below is compared against (`reachable_inv` is the
-- proof of slot conservation). `outOfFuel` here means no violation within 5
-- events, not an exhausted state space.
#guard ModelCheck.check
  (sys { numLayers := 1, numSlots := 2, numUuids := 1 })
  checkEvents violatesSlotConservation 5 = .outOfFuel
#guard ModelCheck.check
  (sys { numLayers := 1, numSlots := 2, numUuids := 1 })
  checkEvents violatesExclusive 5 = .outOfFuel

/-- **Mutant (`stepOverwriteDraining`):** what if `registerRecv` allowed
replacing an incumbent as soon as it is `draining` (e.g., past its deadline)
instead of waiting for `done = true` (`inFlight = 0`)? Overwriting a draining
incumbent whose H2D copy is still in flight leaks its pinned staging slot and
breaks slot conservation (`freeSlots + activeStaging != numSlots`). -/
def stepOverwriteDraining (s : State) (ev : Ev) : Option State :=
  match ev with
  | .registerRecv reqId uuid =>
    match s.recvTable[uuid]? with
    | some (some inc) =>
      if inc.recv.life.draining && 0 < s.freeSlots then
        let entry : RecvEntry :=
          { reqId, gen := s.nextGen, recv := Recv.initPush s.cfg.numLayers }
        some { s with
          freeSlots := s.freeSlots - 1,
          nextGen := s.nextGen + 1,
          recvTable := s.recvTable.set uuid (some entry) }
      else step s ev
    | _ => step s ev
  | _ => step s ev

#guard (match ModelCheck.check
          ⟨init { numLayers := 1, numSlots := 2, numUuids := 1 }, stepOverwriteDraining⟩
          checkEvents violatesSlotConservation 5 with
        | .counterexample _ => true
        | _ => false)

end TpuSyncVerify.Transfer.PrefillDecode.UuidTable

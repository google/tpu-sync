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

/-- `RecvDrainTest.DuplicateUuidIsRejectedUntilExpiredReceiveDrains`: while an
expired receive has an H2D copy in flight (`draining = true, done = false`),
`.sweepRecv` and duplicate `.registerRecv` are rejected until the copy finishes. -/
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
a duplicate `startRead` with a different `reqId` leaves the live incumbent and
its slot intact and records the duplicate `reqId` in `failedRecving`. -/
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

/-- `ControlHandshakeTest.RepeatedReceiveAnnouncementIsIdempotent`: re-delivering
`startRead` with the same `(reqId, uuid)` while in flight is an idempotent no-op. -/
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

/-- `SendLifecycleTest.DuplicateRegistrationCannotReplaceLiveOffer`: duplicate
`notifyForRead` on a live offer is rejected until the offer settles and is swept. -/
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

/-- Model-only: `startRead` on a settled incumbent retires it inline and seats
the replacement (`gen = 1`) without leaking a staging slot. -/
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

/-- Model-only: `notifyForRead` refuses any live or unretired receive `uuid`,
until `.sweepRecv` erases the settled receive. -/
theorem trace_one_map_for_both_directions :
    (sys oneUuid).run [.startRead 0 0, .notifyForRead 1 0] = none ∧
    (sys oneUuid).run [.startRead 0 0, .recvStep 0 (.pullReply false), .notifyForRead 1 0] = none ∧
    ((sys oneUuid).run [.startRead 0 0, .recvStep 0 (.pullReply false), .sweepRecv 0,
                    .notifyForRead 1 0]).map
      (fun s => (s.sendTable[0]?).bind (fun o => o.map SendEntry.reqId)) = some (some 1) := by
  decide

/-- Model-only: `startRead` under a live send offer fails into `failedRecving`,
whereas once the offer settles `startRead` retires it inline and seats the read. -/
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

/-- Zero-layer config (`IsReadyToComplete` holds with `inFlight = 0` at seating). -/
def zeroLayers : Config := { numLayers := 0, numSlots := 4, numUuids := 1 }

/-- `RecvLifecycleTest.RegisterRecvFinishesAReadyIncumbentBeforeReplacingIt` and
`RecvLifecycleTest.RedeliveredStartReadLeavesAReadyReceiveToCompleteRead`:
`registerRecv` finishes and replaces an idle ready incumbent inline, while a
same-`reqId` `startRead` leaves it for `pollReady` + `sweepRecv`. -/
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

/-- Model-only: when a ready incumbent still has a callback in flight, `registerRecv`
starts it draining (`Finish()`) without replacing it until `h2dDone` settles it. -/
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

/-- Mutant (`stepOverwriteDraining`): replacing a `draining` incumbent before
`done = true` leaks its pinned staging slot and violates `violatesSlotConservation`. -/
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

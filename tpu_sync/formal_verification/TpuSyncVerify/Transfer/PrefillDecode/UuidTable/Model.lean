import TpuSyncVerify.Common.ListAux
import TpuSyncVerify.Transfer.PrefillDecode.Receive
import TpuSyncVerify.Transfer.PrefillDecode.Send

/-! # `UuidTable`: state, events, `step`

Part of `TpuSyncVerify.Transfer.PrefillDecode.UuidTable`; the module docstring and
source map live in `UuidTable.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.UuidTable

open TpuSyncVerify.Transfer (Lifecycle)
open TpuSyncVerify.Transfer.PrefillDecode (Recv Send)

abbrev Uuid := Nat
abbrev ReqId := Nat

instance : DecidableEq Uuid := inferInstanceAs (DecidableEq Nat)
instance : DecidableEq ReqId := inferInstanceAs (DecidableEq Nat)

/-- A receive session seated at `sessions_[uuid]`, tagged with its `reqId`
(`SessionEntry::req_id`) and a monotonic registration generation `gen`. -/
structure RecvEntry where
  reqId : ReqId
  gen : Nat
  recv : Recv
  deriving Repr, DecidableEq

/-- A send offer seated at `sessions_[uuid]`, tagged with its `reqId`
(`SessionEntry::req_id`). -/
structure SendEntry where
  reqId : ReqId
  gen : Nat
  send : Send
  deriving Repr, DecidableEq

def slotHasStaging : Option RecvEntry → Bool
  | none => false
  | some e => e.recv.life.hasStaging

/-- Number of entries in `recvTable` currently holding a host staging slot. -/
def activeStaging : List (Option RecvEntry) → Nat
  | [] => 0
  | o :: os => (if slotHasStaging o then 1 else 0) + activeStaging os

/-- Set-like insertion (`absl::flat_hash_set::insert`) into a report list. -/
def insertReq (rs : List ReqId) (r : ReqId) : List ReqId :=
  if r ∈ rs then rs else rs ++ [r]

/-- Record a settled receive session's outcome in `doneRecving` or `failedRecving`
(`KVCacheManagerWithTransfer::StartRead(remote_endpoint)`,
`KVCacheManagerWithTransfer::CompleteReadWithDetails`). -/
def recordRecvReport (doneRecving failedRecving : List ReqId)
    (reqId : ReqId) (statusOk : Bool) : List ReqId × List ReqId :=
  if statusOk then
    (insertReq doneRecving reqId, failedRecving)
  else
    (doneRecving, insertReq failedRecving reqId)

/-- Record a settled send session's outcome in `doneSending` or `failedRecving`
(`KVCacheManagerWithTransfer::CompleteReadWithDetails`). -/
def recordSendReport (doneSending failedRecving : List ReqId)
    (reqId : ReqId) (statusOk : Bool) : List ReqId × List ReqId :=
  if statusOk then
    (insertReq doneSending reqId, failedRecving)
  else
    (doneSending, insertReq failedRecving reqId)

structure Config where
  numLayers : Nat := 1
  numSlots : Nat := 4
  numUuids : Nat := 2
  deriving Repr, DecidableEq

structure State where
  cfg : Config
  freeSlots : Nat
  nextGen : Nat := 0
  recvTable : List (Option RecvEntry)
  sendTable : List (Option SendEntry)
  doneSending : List ReqId := []
  doneRecving : List ReqId := []
  failedRecving : List ReqId := []
  deriving Repr, DecidableEq

def init (cfg : Config) : State :=
  { cfg,
    freeSlots := cfg.numSlots,
    recvTable := List.replicate cfg.numUuids none,
    sendTable := List.replicate cfg.numUuids none }

/-- The staging slot a `Finish()` of incumbent `inc` returns when it settles the
session as `r` (`ReleaseStagingLocked` inside `FinishLocked`), else `0`. -/
def freedBy (inc : RecvEntry) (r : Recv) : Nat :=
  if inc.recv.life.hasStaging && !r.life.hasStaging then 1 else 0

inductive Ev where
  /-- `StartRead(req_id, uuid, ...)` (`KVCacheManagerWithTransfer::StartRead(remote_endpoint)`): creates a load-plan
  receive session (`Recv.initLoad`), with idempotent same-`reqId` handling,
  immediate `failed_recving_` reporting on a conflicting duplicate `reqId` or
  on a live send offer under the uuid, and inline retirement of a settled
  incumbent of either kind (`RetireSettledSessionLocked`). -/
  | startRead (reqId : ReqId) (uuid : Uuid)
  /-- `RegisterRecv` → `EmplaceRecvSessionLocked(uuid, req_id, session)`
  (`KVCacheManagerWithTransfer::RegisterRecv`,
  `KVCacheManagerWithTransfer::EmplaceRecvSessionLocked`; `RegisterActivePlan`
  seats its receive through the same call, after its own plan check):
  creates a push-plan receive session (`Recv.initPush`); finishes a ready
  incumbent (`recv->Finish()`), retires a settled one (receive or send offer,
  `RetireSettledSessionLocked`), and otherwise rejects the duplicate `uuid`
  without leaking staging. The slot taken here models `RegisterActivePlan`
  under dynamic staging and the white-box test fixture
  `RecvTestManager::AddRecv` (`acquire_staging = true`); production
  `RegisterRecv` passes `acquire_staging = false` and takes no slot. -/
  | registerRecv (reqId : ReqId) (uuid : Uuid)
  /-- Step the active receive session at `uuid` (`Receive.lean`). -/
  | recvStep (uuid : Uuid) (e : Recv.Ev)
  /-- `CompleteReadWithDetails` / `CompleteReadRaw` sweeps a settled (`Done()`)
  receive at `uuid` (`KVCacheManagerWithTransfer::CompleteReadWithDetails`,
  `KVCacheManagerWithTransfer::CompleteReadRaw`). -/
  | sweepRecv (uuid : Uuid)
  /-- `NotifyForRead(req_id, uuid, ...)` (`KVCacheManagerWithTransfer::NotifyForRead`): registers a send
  offer at `sessions_[uuid]`; rejects any `uuid` the map holds, send or receive,
  live or settled (a thrown `std::runtime_error` when
  `sessions_.try_emplace(uuid, SessionEntry{session, req_id, SessionRole::kTransfer})`
  fails). -/
  | notifyForRead (reqId : ReqId) (uuid : Uuid)
  /-- Step the active send session at `uuid` (`Send.lean`). -/
  | sendStep (uuid : Uuid) (e : Send.Ev)
  /-- `CompleteReadWithDetails` / `CompleteReadRaw` sweeps a settled (`Done()`)
  send at `uuid` (`KVCacheManagerWithTransfer::CompleteReadWithDetails`,
  `KVCacheManagerWithTransfer::CompleteReadRaw`). -/
  | sweepSend (uuid : Uuid)
  deriving Repr, DecidableEq

def step (s : State) : Ev → Option State
  | .startRead reqId uuid =>
    match s.recvTable[uuid]? with
    | none => none
    | some (some inc) =>
      if !inc.recv.life.done then
        -- `RetireSettledSessionLocked(incumbent)` returns false on a live or
        -- draining receive (`KVCacheManagerWithTransfer::StartRead(remote_endpoint)`).
        if inc.reqId == reqId then
          some s
        else
          some { s with failedRecving := insertReq s.failedRecving reqId }
      else
        -- Incumbent is already `Done()`: retired inline (`RetireSettledSessionLocked`).
        let (doneR, failR) :=
          recordRecvReport s.doneRecving s.failedRecving inc.reqId inc.recv.life.statusOk
        if 0 < s.freeSlots then
          let entry : RecvEntry :=
            { reqId, gen := s.nextGen, recv := Recv.initLoad s.cfg.numLayers }
          some { s with
            freeSlots := s.freeSlots - 1,
            nextGen := s.nextGen + 1,
            recvTable := s.recvTable.set uuid (some entry),
            doneRecving := doneR,
            failedRecving := failR }
        else
          some { s with
            recvTable := s.recvTable.set uuid none,
            doneRecving := doneR,
            failedRecving := insertReq failR reqId }
    | some none =>
      match s.sendTable[uuid]? with
      | none => none
      | some (some snd) =>
        if !snd.send.life.done then
          -- A live send offer holds `uuid` in `sessions_`: `RetireSettledSessionLocked`
          -- returns false, `recv_incumbent == nullptr`, and the read is rejected.
          some { s with failedRecving := insertReq s.failedRecving reqId }
        else
          -- A settled send offer is retired inline into `done_sending_` / `failed_recving_`.
          let (doneS, failR) :=
            recordSendReport s.doneSending s.failedRecving snd.reqId snd.send.life.statusOk
          if 0 < s.freeSlots then
            let entry : RecvEntry :=
              { reqId, gen := s.nextGen, recv := Recv.initLoad s.cfg.numLayers }
            some { s with
              freeSlots := s.freeSlots - 1,
              nextGen := s.nextGen + 1,
              recvTable := s.recvTable.set uuid (some entry),
              sendTable := s.sendTable.set uuid none,
              doneSending := doneS,
              failedRecving := failR }
          else
            some { s with
              sendTable := s.sendTable.set uuid none,
              doneSending := doneS,
              failedRecving := insertReq failR reqId }
      | some none =>
        if 0 < s.freeSlots then
          let entry : RecvEntry :=
            { reqId, gen := s.nextGen, recv := Recv.initLoad s.cfg.numLayers }
          some { s with
            freeSlots := s.freeSlots - 1,
            nextGen := s.nextGen + 1,
            recvTable := s.recvTable.set uuid (some entry) }
        else
          some { s with failedRecving := insertReq s.failedRecving reqId }
  | .registerRecv reqId uuid =>
    match s.recvTable[uuid]? with
    | none => none
    | some (some inc) =>
      match Recv.pollReady inc.recv with
      | some r =>
        -- `EmplaceRecvSessionLocked` first finishes a ready incumbent
        -- (`recv->Finish()` if `!recv->IsDraining() && recv->IsReadyToComplete()`).
        if !r.life.done then
          -- Still draining (a callback is outstanding): `try_emplace` fails,
          -- `AlreadyExistsError`; the finish stays.
          some { s with
            freeSlots := s.freeSlots + freedBy inc r,
            recvTable := s.recvTable.set uuid (some { inc with recv := r }) }
        else if 0 < s.freeSlots + freedBy inc r then
          -- The finish settled it (no work in flight): retired and replaced in
          -- the same call.
          let (doneR, failR) :=
            recordRecvReport s.doneRecving s.failedRecving inc.reqId r.life.statusOk
          let entry : RecvEntry :=
            { reqId, gen := s.nextGen, recv := Recv.initPush s.cfg.numLayers }
          some { s with
            freeSlots := s.freeSlots + freedBy inc r - 1,
            nextGen := s.nextGen + 1,
            recvTable := s.recvTable.set uuid (some entry),
            doneRecving := doneR,
            failedRecving := failR }
        else
          none
      | none =>
        if !inc.recv.life.done then
          -- `EmplaceRecvSessionLocked` returns `AlreadyExistsError` and the
          -- candidate's staging is released with it
          -- (`TransferReceiveSession::~TransferReceiveSession`).
          none
        else if 0 < s.freeSlots then
          let (doneR, failR) :=
            recordRecvReport s.doneRecving s.failedRecving inc.reqId inc.recv.life.statusOk
          let entry : RecvEntry :=
            { reqId, gen := s.nextGen, recv := Recv.initPush s.cfg.numLayers }
          some { s with
            freeSlots := s.freeSlots - 1,
            nextGen := s.nextGen + 1,
            recvTable := s.recvTable.set uuid (some entry),
            doneRecving := doneR,
            failedRecving := failR }
        else
          none
    | some none =>
      match s.sendTable[uuid]? with
      | none => none
      | some (some snd) =>
        if !snd.send.life.done then
          -- A live send offer holds `uuid`: `try_emplace` fails.
          none
        else if 0 < s.freeSlots then
          -- A settled send offer is retired inline (`RetireSettledSessionLocked`).
          let (doneS, failR) :=
            recordSendReport s.doneSending s.failedRecving snd.reqId snd.send.life.statusOk
          let entry : RecvEntry :=
            { reqId, gen := s.nextGen, recv := Recv.initPush s.cfg.numLayers }
          some { s with
            freeSlots := s.freeSlots - 1,
            nextGen := s.nextGen + 1,
            recvTable := s.recvTable.set uuid (some entry),
            sendTable := s.sendTable.set uuid none,
            doneSending := doneS,
            failedRecving := failR }
        else
          none
      | some none =>
        if 0 < s.freeSlots then
          let entry : RecvEntry :=
            { reqId, gen := s.nextGen, recv := Recv.initPush s.cfg.numLayers }
          some { s with
            freeSlots := s.freeSlots - 1,
            nextGen := s.nextGen + 1,
            recvTable := s.recvTable.set uuid (some entry) }
        else
          none
  | .recvStep uuid e =>
    match s.recvTable[uuid]? with
    | none => none
    | some none => none
    | some (some entry) =>
      match Recv.step entry.recv e with
      | none => none
      | some recv' =>
        let entry' : RecvEntry := { entry with recv := recv' }
        let freeSlots' :=
          if entry.recv.life.hasStaging && !recv'.life.hasStaging then
            s.freeSlots + 1
          else s.freeSlots
        some { s with
          freeSlots := freeSlots',
          recvTable := s.recvTable.set uuid (some entry') }
  | .sweepRecv uuid =>
    match s.recvTable[uuid]? with
    | none => none
    | some none => none
    | some (some entry) =>
      if entry.recv.life.done then
        let (doneR, failR) :=
          recordRecvReport s.doneRecving s.failedRecving entry.reqId entry.recv.life.statusOk
        some { s with
          recvTable := s.recvTable.set uuid none,
          doneRecving := doneR,
          failedRecving := failR }
      else
        none
  | .notifyForRead reqId uuid =>
    match s.sendTable[uuid]?, s.recvTable[uuid]? with
    | some none, some none =>
      let entry : SendEntry :=
        { reqId, gen := s.nextGen, send := Send.init s.cfg.numLayers }
      some { s with
        nextGen := s.nextGen + 1,
        sendTable := s.sendTable.set uuid (some entry) }
    | _, _ =>
      -- `sessions_.try_emplace(uuid, SessionEntry{...})` fails on any seated session,
      -- send or receive, live or settled (`KVCacheManagerWithTransfer::NotifyForRead`).
      none
  | .sendStep uuid e =>
    match s.sendTable[uuid]? with
    | none => none
    | some none => none
    | some (some entry) =>
      match Send.step entry.send e with
      | none => none
      | some send' =>
        some { s with
          sendTable := s.sendTable.set uuid (some { entry with send := send' }) }
  | .sweepSend uuid =>
    match s.sendTable[uuid]? with
    | none => none
    | some none => none
    | some (some entry) =>
      if entry.send.life.done then
        let (doneS, failR) :=
          recordSendReport s.doneSending s.failedRecving entry.reqId entry.send.life.statusOk
        some { s with
          sendTable := s.sendTable.set uuid none,
          doneSending := doneS,
          failedRecving := failR }
      else
        none

def sys (cfg : Config) : System State Ev where
  init := init cfg
  step := step

end TpuSyncVerify.Transfer.PrefillDecode.UuidTable

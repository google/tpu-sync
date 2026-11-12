import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Reshard.Registry
import TpuSyncVerify.Transfer.Reshard.Coordinator.Model

/-! # `SubPlanChain`: window planning, worker chain folding, and coordinator chain

Part of `TpuSyncVerify.Transfer.Reshard.SubPlanChain`; the module docstring and
source map live in `SubPlanChain.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard.SubPlanChain

open TpuSyncVerify
open TpuSyncVerify.Transfer.Reshard.Registry
open TpuSyncVerify.Transfer.Reshard.Coordinator (CoordPhase abandonIfClaimed)

/-! ## Part I — Sub-transfer window planning (`PlanSubTransfers`, `SubPlanUuid`) -/

/-- Exact 64-bit `splitmix64` sub-plan UUID derivation (`SubPlanUuid` in
`reshard_coordinator.cc`): returns `uuid` for `index == 0` and a positive
63-bit mix otherwise. -/
def subPlanUuid (uuid : UInt64) (index : Nat) : UInt64 :=
  if index == 0 then
    uuid
  else
    let z0 := uuid + (0x9E3779B97F4A7C15 : UInt64) * UInt64.ofNat index
    let z1 := (z0 ^^^ (z0 >>> 30)) * (0xBF58476D1CE4E5B9 : UInt64)
    let z2 := (z1 ^^^ (z1 >>> 27)) * (0x94D049BB133111EB : UInt64)
    let z3 := z2 ^^^ (z2 >>> 31)
    let mixed := z3 &&& (0x7FFFFFFFFFFFFFFF : UInt64)
    if mixed > 0 then mixed else 1

/-- One pool tag's slice in one sub-transfer (`SubTransferSlice` in
`reshard_coordinator.h`): destination blocks `[offset, offset + count)` of the
tag's destination list (`total = 0` when the tag is unwindowed and rides whole). -/
structure SubTransferSlice where
  tagIndex : Nat
  offset : Nat
  count : Nat
  total : Nat
  deriving DecidableEq, Repr

/-- Windowed sub-transfer chain produced by `PlanSubTransfers`
(`SubTransferChain` in `reshard_coordinator.h`). -/
structure SubTransferChain where
  count : Nat := 1
  subPlans : List (List SubTransferSlice) := []
  deriving DecidableEq, Repr

/-- Inputs to `PlanSubTransfers` (`PoolReshardArgs` fields inspected by
`PlanSubTransfers` in `reshard_coordinator.cc`). -/
structure PlanArgs where
  numTags : Nat
  dstDeviceBlockIds : List Nat
  dstBlockCounts : List Nat := []
  dstWindowBlocks : List Int := []
  deriving DecidableEq, Repr

/-- Extract per-tag destination block ID slices according to `counts`. -/
def idsByTag (ids : List Nat) : List Nat → List (List Nat)
  | [] => []
  | c :: cs => ids.take c :: idsByTag (ids.drop c) cs

/-- Ceiling division `(n + w - 1) / w` for positive `w` and `n > w`
(`(counts[i] + window - 1) / window` in `PlanSubTransfers`). -/
def windowsForTag (cnt win : Nat) : Nat :=
  if win > 0 && cnt > win then (cnt + win - 1) / win else 1

/-- True if any windowed tag (`windowsPerTag[i] > 1`) shares a destination
block ID with another tag `j ≠ i` (`ids_by_tag` check in `PlanSubTransfers`). -/
def hasWindowedTagSharingIds (tagIds : List (List Nat)) (winCounts : List Nat) : Bool :=
  (List.range tagIds.length).any fun i =>
    let ki := winCounts.getD i 1
    if ki <= 1 then false
    else
      let idsi := tagIds.getD i []
      (List.range tagIds.length).any fun j =>
        j != i && (tagIds.getD j []).any (fun id => idsi.contains id)

/-- Executable port of `PlanSubTransfers` (`reshard_coordinator.cc`).
Returns `none` when `PlanSubTransfers` fails closed with `InvalidArgumentError`
(negative window or arity mismatch), `{ count := 1, subPlans := [] }` when
unwindowed, deferred to single-plan validation, or collapsed because a windowed
tag shares destination block IDs with another tag, and the `longest`-step
`SubTransferChain` otherwise. -/
def planSubTransfers (args : PlanArgs) : Option SubTransferChain :=
  if args.dstWindowBlocks.any (fun w => w < 0) then
    none
  else if !args.dstWindowBlocks.any (fun w => w > 0) then
    some {}
  else if args.dstWindowBlocks.length != args.numTags then
    none
  else
    let countsOpt : Option (List Nat) :=
      if args.dstBlockCounts.isEmpty then
        if args.numTags != 1 then none else some [args.dstDeviceBlockIds.length]
      else if args.dstBlockCounts.length != args.numTags then
        none
      else
        some args.dstBlockCounts
    match countsOpt with
    | none => some {}
    | some counts =>
      if counts.any (fun c => c == 0) || counts.foldl (· + ·) 0 != args.dstDeviceBlockIds.length then
        some {}
      else
        let windows : List Nat := args.dstWindowBlocks.map Int.toNat
        let winPerTag : List Nat :=
          (List.range args.numTags).map fun i =>
            windowsForTag (counts.getD i 0) (windows.getD i 0)
        let longest : Nat := winPerTag.foldl max 1
        let tagIds := idsByTag args.dstDeviceBlockIds counts
        if hasWindowedTagSharingIds tagIds winPerTag then
          some {}
        else if longest <= 1 then
          some {}
        else
          let subPlans : List (List SubTransferSlice) :=
            (List.range longest).map fun step =>
              (List.range args.numTags).filterMap fun i =>
                let ki := winPerTag.getD i 1
                let first := longest - ki
                if step < first then
                  none
                else
                  let w := step - first
                  let cnt := counts.getD i 0
                  if ki == 1 then
                    some { tagIndex := i, offset := 0, count := cnt, total := 0 }
                  else
                    let win := windows.getD i 0
                    let off := w * win
                    some { tagIndex := i, offset := off, count := min win (cnt - off), total := cnt }
          some { count := longest, subPlans := subPlans }

/-! ## Part II — Worker sub-plan chain folding (`KVCacheManagerWithTransfer`) -/

/-- One worker's `recv_chains_[req_id]` or `send_chains_[req_id]` entry
(`KVCacheManagerWithTransfer::SubPlanChainProgress` in
`kv_cache_manager_with_transfer.h`); `present = false` is absence from the map. -/
structure ChainProgress where
  present : Bool := false
  /-- `count`: the largest `sub_plan_count` seen (`std::max(progress.count, sub_plan_count)`). -/
  count : Nat := 0
  /-- `settled`: sub-plan indices whose session settled OK (`std::set<int32_t>`). -/
  settled : List Nat := []
  /-- `failed`: the request's failure has been reported. -/
  failed : Bool := false
  /-- `anchored`: sub-plan 0 of this chain registered on this worker. -/
  anchored : Bool := false
  /-- `plan_uuids`: sub-plan UUIDs registered into this entry. -/
  planUuids : List Nat := []
  deriving DecidableEq, Repr, Hashable

/-- One session in `active_pool_reshard_recvs_` / `active_pool_reshard_sends_`
for `req_id`, with its `sub_plan_index()` / `sub_plan_count()` and, once
`Done()`, the status it settled with. -/
structure SubSession where
  uuid : Nat
  index : Nat
  count : Nat
  done : Bool := false
  ok : Bool := true
  deriving DecidableEq, Repr, Hashable

/-- One worker's view of one `req_id`: its chain entry, its active reshard
sessions, and the terminals it has published for the request. -/
structure WorkerState where
  /-- Insertions of `req_id` into `done_recving_` / `done_sending_`. -/
  doneReports : Nat := 0
  /-- Insertions of `req_id` into `failed_recving_`. -/
  failReports : Nat := 0
  /-- Ghost: for each `done` insertion, the chain `count` and `settled` set at
  that moment (`(1, [0])` for a single-plan request). -/
  doneLog : List (Nat × List Nat) := []
  /-- Ghost: UUIDs this worker has registered for `req_id` (A2). -/
  seenUuids : List Nat := []
  /-- `active_pool_reshard_*` entries for `req_id`, in registration order. -/
  active : List SubSession := []
  chain : ChainProgress := {}
  deriving DecidableEq, Repr, Hashable

/-- Events of the worker chain folding. -/
inductive WorkerEv where
  /-- `PoolReshardRegisterRecv` / `PoolReshardPush` seats session `uuid` stamped
  `(sub_plan_index, sub_plan_count) = (index, count)` and calls
  `NoteSubPlanRegisteredLocked`. -/
  | registerSubPlan (uuid index count : Nat)
  /-- Session `uuid` reaches `Done()` with an OK (`ok = true`) or failed status. -/
  | settleSession (uuid : Nat) (ok : Bool)
  /-- `CompleteReadWithDetails`: `RecordPoolReshardTerminalLocked` for every
  `Done()` session, then `SweepSubPlanChainsLocked(now)`; `stalled = true` is
  `now - progress.last_activity > stall`. -/
  | poll (stalled : Bool)
  deriving DecidableEq, Repr

/-- Which arm of `NoteSubPlanRegisteredLocked` a registration took. -/
inductive RegisterArm where
  /-- `if (sub_plan_count <= 1) return;` -/
  | single
  /-- `sub_plan_index == 0`: `chains[req_id] = SubPlanChainProgress{}; anchored = true`. -/
  | anchor
  /-- `chains[req_id]` found for a later index. -/
  | intoEntry
  /-- `chains[req_id]` default-constructed for a later index (`anchored = false`). -/
  | unanchored
  deriving DecidableEq, Repr

/-- `NoteSubPlanRegisteredLocked`: nothing for `sub_plan_count <= 1`; sub-plan 0
replaces the entry with a fresh anchored one; any other index finds or
default-constructs the entry; then `plan_uuids.insert(uuid)` and
`count = max(count, sub_plan_count)`. -/
def noteSubPlanRegistered (c : ChainProgress) (uuid index count : Nat) :
    ChainProgress × RegisterArm :=
  if count <= 1 then
    (c, .single)
  else
    let (base, arm) : ChainProgress × RegisterArm :=
      if index == 0 then ({ present := true, anchored := true }, .anchor)
      else if c.present then (c, .intoEntry)
      else ({ present := true }, .unanchored)
    ({ base with
        planUuids := insertNodup uuid base.planUuids
        count := max base.count count }, arm)

/-- What one `RecordPoolReshardTerminalLocked` call did. -/
inductive Terminal where
  /-- `anchored && !plan_uuids.contains(uuid)`: a superseded chain's sub-plan, ignored. -/
  | supersededIgnored
  /-- `if (progress.failed) return;`: the failure was already reported. -/
  | failedIgnored
  /-- `settled.insert(sub_plan_index)` short of `count`. -/
  | settledPartial
  /-- `done.insert(req_id)` with the chain's `count` and `settled` at that moment. -/
  | done (count : Nat) (settled : List Nat)
  /-- `failed_recving_.insert(req_id)`. -/
  | failed
  deriving DecidableEq, Repr

/-- `RecordPoolReshardTerminalLocked` for one `Done()` session `s`. -/
def recordTerminal (c : ChainProgress) (s : SubSession) : ChainProgress × Terminal :=
  if s.count <= 1 then
    -- A single-plan request: the session's terminal is the request's.
    (c, if s.ok then .done 1 [s.index] else .failed)
  else if c.present && c.anchored && !c.planUuids.contains s.uuid then
    (c, .supersededIgnored)
  else
    let c0 : ChainProgress := if c.present then c else { present := true }
    if c0.failed then
      (c0, .failedIgnored)
    else
      let c1 := { c0 with count := max c0.count s.count }
      if !s.ok then
        ({ c1 with failed := true }, .failed)
      else
        let settled' := insertNodup s.index c1.settled
        if settled'.length >= c1.count then
          ({}, .done c1.count settled')
        else
          ({ c1 with settled := settled' }, .settledPartial)

/-- What `SweepSubPlanChainsLocked` did to an entry quiet for a whole `timeout_s_`. -/
inductive SweepOutcome where
  /-- No sweep (`now - progress.last_activity <= stall`, or no entry). -/
  | notSwept
  /-- `if (progress.failed) chains.erase(it++)`. -/
  | erasedFailed
  /-- A session of `req_id` is still active: `last_activity = now`. -/
  | activeKeeps
  /-- `failed_recving_.insert(req_id); progress.failed = true`. -/
  | stalled
  deriving DecidableEq, Repr

/-- `SweepSubPlanChainsLocked` for an entry quiet for a whole `timeout_s_`:
a reported failure is erased; otherwise, unless a session of `req_id` is still
active, `failed_recving_.insert(req_id)` and `failed = true`. -/
def sweepStalledChain (c : ChainProgress) (stillActive : List SubSession) :
    ChainProgress × SweepOutcome :=
  if !c.present then
    (c, .notSwept)
  else if c.failed then
    ({}, .erasedFailed)
  else if !stillActive.isEmpty then
    (c, .activeKeeps)
  else
    ({ c with failed := true }, .stalled)

namespace WorkerState

def init : WorkerState := {}

/-- Accumulator of one `CompleteReadWithDetails` sweep. -/
structure SweepAcc where
  chain : ChainProgress
  /-- `done.insert(req_id)` calls so far. -/
  done : Nat := 0
  /-- `failed_recving_.insert(req_id)` calls so far. -/
  fail : Nat := 0
  /-- Ghost `doneLog` entries. -/
  log : List (Nat × List Nat) := []
  /-- A chained sub-plan reached `progress.last_activity = now` in this call
  (not a single-plan or superseded one), so the entry is not quiet. -/
  touched : Bool := false
  deriving DecidableEq, Repr

def recordStep (acc : SweepAcc) (s : SubSession) : SweepAcc :=
  let touched := acc.touched || (s.count > 1 && (recordTerminal acc.chain s).2 != .supersededIgnored)
  match recordTerminal acc.chain s with
  | (c', .done k idx) => { acc with chain := c', done := acc.done + 1, log := acc.log ++ [(k, idx)], touched }
  | (c', .failed) => { acc with chain := c', fail := acc.fail + 1, touched }
  | (c', _) => { acc with chain := c', touched }

/-- Fold `recordTerminal` over the settled sessions of one
`CompleteReadWithDetails` call. -/
def recordAll (c : ChainProgress) (settled : List SubSession) : SweepAcc :=
  settled.foldl recordStep { chain := c }

/-- Transition function of the worker chain folding. -/
def step (w : WorkerState) (e : WorkerEv) : Option WorkerState :=
  match e with
  | .registerSubPlan uuid index count =>
    if w.seenUuids.contains uuid then
      none  -- A2: sub-plan UUIDs are fresh
    else if index >= max 1 count then
      none  -- A1: `0 <= sub_plan_index < max(1, sub_plan_count)`
    else
      some
        { w with
          seenUuids := uuid :: w.seenUuids
          active := w.active ++ [{ uuid := uuid, index := index, count := count }]
          chain := (noteSubPlanRegistered w.chain uuid index count).1 }

  | .settleSession uuid ok =>
    if !w.active.any (fun s => s.uuid == uuid && !s.done) then
      none
    else
      some { w with active := w.active.map fun s =>
        if s.uuid == uuid then { s with done := true, ok := ok } else s }

  | .poll stalled =>
    let settledNow := w.active.filter (·.done)
    let stillActive := w.active.filter (!·.done)
    let r := recordAll w.chain settledNow
    -- `SweepSubPlanChainsLocked(now)` runs after the records of this call, which
    -- set `last_activity = now`: an entry touched here is not quiet.
    let (c2, sweep) :=
      if stalled && !r.touched then sweepStalledChain r.chain stillActive else (r.chain, .notSwept)
    some
      { w with
        doneReports := w.doneReports + r.done
        failReports := w.failReports + r.fail + (if sweep == .stalled then 1 else 0)
        doneLog := w.doneLog ++ r.log
        active := stillActive
        chain := c2 }

def sys : System WorkerState WorkerEv := ⟨init, step⟩

end WorkerState

/-! ## Part III — Coordinator multi-window chain (`ExecutePoolReshard`, `StartChain`, `RunSubPlan`) -/

/-- `ReshardCoordinator::transfer_status_[req_id]` (`ReshardCoordinator::Status`). -/
inductive TransferStatus where
  | notStarted
  | inProgress
  | completed
  | failed
  deriving DecidableEq, Repr, Hashable

/-- Outcome of one Stage 1 `PoolReshardRegisterRecv` RPC as `RunSubPlan` reads
it: armed, refused because the destination's staging arena is still held by an
earlier sub-plan (`ArmRefusedForBusyStaging(arm_status[i])`), or any other
error. -/
inductive ArmOutcome where
  | ok
  | refusedBusy
  | failed
  deriving DecidableEq, Repr, Hashable

/-- Per-destination receiver occupancy across windows. -/
structure ChainDstState where
  unit : Nat
  /-- An earlier window's `ReshardReceiveSession` still holds this
  destination's bounded `PoolStagingLease`. -/
  prevSessionActive : Bool := false
  /-- The current window's `ReshardReceiveSession` is armed. -/
  curWinArmed : Bool := false
  deriving DecidableEq, Repr, Hashable

/-- `ReshardCoordinator::StartPoolReshard` / `ExecutePoolReshard` / `StartChain` /
`RunSubPlan(ctx, 0 .. K - 1)` composed with the request's `RequestBlockRegistry`
generation. -/
structure CoordChainState where
  /-- `args.src_units`: the claimed source snapshot. -/
  allSrcUnits : List Nat
  /-- Per-window `plan.src_units` (`BuildPoolReshardPlan` keeps only the ranks
  with spans in the window). -/
  winSrcUnits : List (List Nat)
  dstUnits : List Nat
  claimOwner : Nat
  /-- `kChainArmAttempts`. -/
  maxArmAttempts : Nat := 2
  transferStatus : TransferStatus := .notStarted
  /-- `index` of the running `RunSubPlan(ctx, index)`. -/
  winIdx : Nat := 0
  winPhase : CoordPhase := .idle
  /-- `attempt` of the Stage 1 re-arm loop. -/
  armAttempts : Nat := 0
  dstStates : List ChainDstState
  /-- `arm_status[i]` / `armed[i]` for the current window. -/
  armResults : List (Nat × ArmOutcome) := []
  /-- `sender_status[i]` for the current window. -/
  sendResults : List (Nat × Bool) := []
  /-- Source ranks with a dispatched `ReshardSendSession` not yet voted `Complete`. -/
  activeSenders : List Nat := []
  /-- Ghost: latched when a rank in `activeSenders` is left without a registry claim. -/
  activeSenderExposed : Bool := false
  reg : RegGen := RegGen.init
  deriving DecidableEq, Repr, Hashable

/-- Events of the coordinator chain. -/
inductive ChainEv where
  /-- `RequestBlockRegistry::Register` by a producer, before `Coordinate`. -/
  | registerProducer (srcUnit : Nat)
  /-- `StartPoolReshard` sets `transfer_status_[req_id] = kInProgress`. -/
  | startChainCoordination
  /-- `RunSubPlan(ctx, index)` Stage 0: `LookupAndClaim`, then `BuildPoolReshardPlan`. -/
  | startWindowPlan (planOk : Bool)
  /-- One Stage 1 `PoolReshardRegisterRecv` RPC to `dstUnit` returns. -/
  | armDstResult (dstUnit : Nat) (outcome : ArmOutcome)
  /-- Stage 1 join with every refusal busy and attempts left: re-arm the
  unarmed destinations (`retry = true`). -/
  | retryBusyArm
  /-- Stage 1 join otherwise: dispatch, or `return arm_status[i]`. -/
  | finishWindowArm
  /-- One Stage 2 `PoolReshardPush` RPC to `srcUnit` returns. -/
  | sendSrcResult (srcUnit : Nat) (ok : Bool)
  /-- Stage 2 join: next window on `StartChain`, final status, or `return status`. -/
  | finishWindowSend
  /-- A dispatched window's receiver on `dstUnit` settles and releases its lease. -/
  | settleDispatchedRecv (dstUnit : Nat)
  /-- A dispatched rank's worker reports `done_sending` and the producer calls
  `RequestBlockRegistry::Complete` (A3). -/
  | senderVoteComplete (srcUnit : Nat)
  /-- `RequestBlockRegistry::CancelIfUnclaimed` by the producer. -/
  | cancelIfUnclaimed
  deriving DecidableEq, Repr

/-- Whether destination `u` has reported in `rs`. -/
def armReported (rs : List (Nat × ArmOutcome)) (u : Nat) : Bool :=
  rs.any (·.1 == u)

def allArmReported (units : List Nat) (rs : List (Nat × ArmOutcome)) : Bool :=
  units.all (armReported rs)

/-- `armed[i]` for destination `u`. -/
def armOk (rs : List (Nat × ArmOutcome)) (u : Nat) : Bool :=
  rs.any (fun r => r.1 == u && r.2 == .ok)

def anyArmFailed (rs : List (Nat × ArmOutcome)) : Bool :=
  rs.any (·.2 != .ok)

/-- Every unarmed destination was refused for busy staging. -/
def allFailuresBusy (rs : List (Nat × ArmOutcome)) : Bool :=
  rs.all (·.2 != .failed)

def hasSendResult (rs : List (Nat × Bool)) (u : Nat) : Bool :=
  rs.any (·.1 == u)

def allSendReported (units : List Nat) (rs : List (Nat × Bool)) : Bool :=
  units.all (hasSendResult rs)

def anySendFailed (rs : List (Nat × Bool)) : Bool :=
  rs.any (!·.2)

namespace CoordChainState

/-- `const int attempts = index == 0 ? 1 : kChainArmAttempts;` -/
def attemptsFor (s : CoordChainState) : Nat :=
  if s.winIdx == 0 then 1 else s.maxArmAttempts

/-- The Stage 1 join sets `retry = true`: every unarmed destination was refused
for busy staging and `attempt + 1 < attempts`. -/
def canRetryBusy (s : CoordChainState) : Bool :=
  anyArmFailed s.armResults && allFailuresBusy s.armResults &&
    s.armAttempts + 1 < s.attemptsFor

/-- Source units of the running window. -/
def curWindowSrcs (s : CoordChainState) : List Nat :=
  s.winSrcUnits.getD s.winIdx []

/-- `srcUnit` is still in `plan.src_units` of the running or a later window of
a chain in progress. -/
def appearsInRemainingWindows (s : CoordChainState) (srcUnit : Nat) : Bool :=
  s.transferStatus == .inProgress &&
    (List.range s.winSrcUnits.length).any fun j =>
      decide (s.winIdx ≤ j) && (s.winSrcUnits.getD j []).contains srcUnit

/-- Ghost monitor: latches `activeSenderExposed` when a dispatched rank is still
reading its source HBM while the generation holds no registry claim. -/
def syncSenderExposure (s : CoordChainState) : CoordChainState :=
  if !s.activeSenders.isEmpty && !s.reg.claimedUnits.isSome then
    { s with activeSenderExposed := true }
  else
    s

/-- `RunSubPlan` returning an error from Stage `stage` (`0` plan, `1` arm,
`2` send): `transfer_status_` becomes `kFailed`; the registry claim is abandoned
only `if (index == 0)` and only before Stage 2. -/
def abortWindow (stage : Nat) (s : CoordChainState) : CoordChainState :=
  let reg' :=
    if stage <= 1 && s.winIdx == 0 then abandonIfClaimed s.reg s.claimOwner else s.reg
  syncSenderExposure
    { s with transferStatus := .failed, winPhase := .aborted, reg := reg' }

def init (allSrcUnits : List Nat) (winSrcUnits : List (List Nat))
    (dstUnits : List Nat) (claimOwner : Nat) (maxArmAttempts : Nat := 2) :
    CoordChainState :=
  { allSrcUnits := allSrcUnits
    winSrcUnits := winSrcUnits
    dstUnits := dstUnits
    claimOwner := claimOwner
    maxArmAttempts := max 1 maxArmAttempts
    dstStates := dstUnits.map fun u => { unit := u } }

/-- Transition function of the coordinator chain. -/
def step (s : CoordChainState) (e : ChainEv) : Option CoordChainState :=
  match e with
  | .registerProducer srcUnit =>
    if s.transferStatus != .notStarted || !s.allSrcUnits.contains srcUnit then
      none
    else
      (RegGen.step s.reg (.register srcUnit true)).map fun reg' => { s with reg := reg' }

  | .startChainCoordination =>
    if s.transferStatus != .notStarted then
      none
    else
      some { s with transferStatus := .inProgress }

  | .startWindowPlan planOk =>
    if s.transferStatus != .inProgress || s.winPhase != .idle then
      none
    else
      match RegGen.step s.reg (.lookupAndClaim s.claimOwner s.allSrcUnits) with
      | none => some (abortWindow 0 s)
      | some regClaimed =>
        let sClaimed := { s with reg := regClaimed }
        if planOk then
          some (syncSenderExposure
            { sClaimed with
              winPhase := .arming
              armAttempts := 0
              armResults := []
              sendResults := [] })
        else
          some (abortWindow 0 sClaimed)

  | .armDstResult dstUnit outcome =>
    if s.transferStatus != .inProgress || s.winPhase != .arming then
      none
    else if !s.dstUnits.contains dstUnit || armReported s.armResults dstUnit then
      none
    else
      -- The bounded staging arena admits the arm iff no earlier window's
      -- session still holds this destination's lease (`PoolStaging`).
      let busy := s.dstStates.any fun d => d.unit == dstUnit && d.prevSessionActive
      if (outcome == .ok && busy) || (outcome == .refusedBusy && !busy) then
        none
      else
        some
          { s with
            dstStates := s.dstStates.map fun d =>
              if d.unit == dstUnit && outcome == .ok then { d with curWinArmed := true } else d
            armResults := s.armResults ++ [(dstUnit, outcome)] }

  | .retryBusyArm =>
    if s.transferStatus != .inProgress || s.winPhase != .arming then
      none
    else if !allArmReported s.dstUnits s.armResults || !s.canRetryBusy then
      none
    else
      -- `if (armed[i]) continue;`: only the unarmed destinations are re-armed.
      some
        { s with
          armAttempts := s.armAttempts + 1
          armResults := s.armResults.filter (·.2 == .ok) }

  | .finishWindowArm =>
    if s.transferStatus != .inProgress || s.winPhase != .arming then
      none
    else if !allArmReported s.dstUnits s.armResults || s.canRetryBusy then
      none
    else if !anyArmFailed s.armResults then
      some { s with winPhase := .dispatching }
    else
      some (abortWindow 1 s)

  | .sendSrcResult srcUnit ok =>
    if s.transferStatus != .inProgress || s.winPhase != .dispatching then
      none
    else if !(curWindowSrcs s).contains srcUnit || hasSendResult s.sendResults srcUnit then
      none
    else
      some (syncSenderExposure
        { s with
          sendResults := s.sendResults ++ [(srcUnit, ok)]
          activeSenders := if ok then insertNodup srcUnit s.activeSenders else s.activeSenders })

  | .finishWindowSend =>
    if s.transferStatus != .inProgress || s.winPhase != .dispatching then
      none
    else if !allSendReported (curWindowSrcs s) s.sendResults then
      none
    else if anySendFailed s.sendResults then
      some (abortWindow 2 s)
    else
      -- The dispatched window's receivers now hold their leases until their
      -- pushes land; the window's arm is spent.
      let dstStates' := s.dstStates.map fun d =>
        { d with prevSessionActive := d.prevSessionActive || d.curWinArmed, curWinArmed := false }
      if s.winIdx + 1 < s.winSrcUnits.length then
        some { s with winIdx := s.winIdx + 1, winPhase := .idle, dstStates := dstStates' }
      else
        some { s with transferStatus := .completed, winPhase := .succeeded, dstStates := dstStates' }

  | .settleDispatchedRecv dstUnit =>
    if !s.dstStates.any (fun d => d.unit == dstUnit && d.prevSessionActive) then
      none
    else
      some { s with dstStates := s.dstStates.map fun d =>
        if d.unit == dstUnit then { d with prevSessionActive := false } else d }

  | .senderVoteComplete srcUnit =>
    if !s.activeSenders.contains srcUnit || s.appearsInRemainingWindows srcUnit then
      none
    else
      (RegGen.step s.reg (.complete srcUnit)).map fun reg' =>
        syncSenderExposure
          { s with
            activeSenders := s.activeSenders.filter (· != srcUnit)
            reg := reg' }

  | .cancelIfUnclaimed =>
    (RegGen.step s.reg .cancelIfUnclaimed).map fun reg' =>
      syncSenderExposure { s with reg := reg' }

def sys (allSrcUnits : List Nat) (winSrcUnits : List (List Nat)) (dstUnits : List Nat)
    (claimOwner : Nat) (maxArmAttempts : Nat := 2) : System CoordChainState ChainEv :=
  ⟨init allSrcUnits winSrcUnits dstUnits claimOwner maxArmAttempts, step⟩

end CoordChainState

end TpuSyncVerify.Transfer.Reshard.SubPlanChain

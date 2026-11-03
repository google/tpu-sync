import TpuSyncVerify.Common.ListAux
import TpuSyncVerify.Transfer.Reshard.Receive
import TpuSyncVerify.Transfer.Reshard.Send

/-! # `Pipeline`: memory cells, state, events, `step`, systems

Part of `TpuSyncVerify.Transfer.Reshard.Pipeline`; the module docstring and
source map live in `Pipeline.lean`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

/-- Contents of one pool slot in memory. -/
inductive Cell where
  | blank
  | kv (pool : Nat)
  | junk
  deriving Repr, DecidableEq

namespace Pipeline

/-- A memory holding `.kv p` at every pool index `0 … n - 1`. -/
def good (n : Nat) : List Cell := (List.range n).map Cell.kv

/-- Coarse lower-rank H2D copy clobbers any strictly higher-rank pool slot on an
aliased arena page (`pool_order_ranks_` in `ReshardReceiveSession`,
`D2hPoolBlocks` / `H2dPoolBlocks` in `KVCacheManagerBase`). -/
def clobberHigherRanks (pools : List Pool) (r : Nat) (hbm : List Cell) : List Cell :=
  List.zipWith (fun pool c => if r < pool.rank then .junk else c) pools hbm

/-- Completing pool `p`'s H2D copy at rank `r` writes `val` to `hbm[p]` after
clobbering any strictly higher-rank pool slot. -/
def writePoolHbm (pools : List Pool) (p : Nat) (r : Nat) (val : Cell)
    (hbm : List Cell) : List Cell :=
  (clobberHigherRanks pools r hbm).set p val

/-- A reshard pipeline schedule is valid when there is at least one pool,
`ranks` and `peerCounts` have the same length, and every pool has at least one
expected push stream (`ValidateCommonPoolReshardPlan`, `CountPoolReshardSendSlots`). -/
def validConfig (ranks peerCounts : List Nat) : Bool :=
  !ranks.isEmpty && (ranks.length == peerCounts.length) && peerCounts.all (0 < ·)

end Pipeline

/-- Composed reshard pipeline state. -/
structure Pipeline where
  numPools : Nat
  peerCounts : List Nat
  send : Send
  recv : Recv
  srcHbm : List Cell
  srcStaging : List Cell
  wire : List Cell
  dstStaging : List Cell
  dstHbm : List Cell
  /-- Completed sender peer pushes per pool (`ReshardSendSession::StartPoolPush`). -/
  sentStreams : List Nat
  /-- Streams landed by `BlockTransportServerPushSession::CompleteIncomingPush` per pool (`progress.completed_chunks++;`). -/
  landedStreams : List Nat
  /-- Whether source HBM has been freed after `send.published ≠ none`. -/
  reclaimed : Bool := false
  /-- Whether `PoolReshardRegisterRecv` has armed the receiver (`ReshardCoordinator::RunSubPlan`). -/
  armed : Bool := true
  deriving Repr, DecidableEq

namespace Pipeline

/-- Initial state for a reshard transfer with pool ranks `ranks` and per-pool
expected stream counts `peerCounts`, after `PoolReshardRegisterRecv` has armed
the receiver (`armed = true`). -/
def init (ranks peerCounts : List Nat) : Pipeline :=
  let n := ranks.length
  { numPools := n
    peerCounts := peerCounts
    send := Send.init peerCounts
    recv := Recv.init ranks
    srcHbm := good n
    srcStaging := List.replicate n .junk
    wire := List.replicate n .blank
    dstStaging := List.replicate n .junk
    dstHbm := List.replicate n .junk
    sentStreams := List.replicate n 0
    landedStreams := List.replicate n 0
    reclaimed := false
    armed := true }

/-- Initial state before Phase 1 of `ReshardCoordinator::RunSubPlan`
(`PoolReshardRegisterRecv`) has armed the receiver. -/
def initUnarmed (ranks peerCounts : List Nat) : Pipeline :=
  { init ranks peerCounts with armed := false }

inductive Ev where
  | armRecv
  | send (e : Send.Ev)
  | recv (e : Recv.Ev)
  | landStream (p : Nat)
  | reclaim
  | reseatSrcStaging
  | reseatDstStaging
  deriving Repr, DecidableEq

/-- Phase 1 of `ReshardCoordinator::RunSubPlan`
(`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`): arms the destination
receiver. -/
def armRecv (s : Pipeline) : Option Pipeline :=
  if s.armed = false then
    some { s with armed := true }
  else none

/-- Sender transition (`ReshardSendSession::ExecutePush`, `StartPoolPush`,
`EndOp`). Launching a D2H copy (`d2hIssue p true`) requires the receiver to be
armed (`ReshardCoordinator::RunSubPlan`); `d2hDone p true` copies
`srcHbm[p]` into `srcStaging[p]`; `pushDone p true` places `srcStaging[p]` on
`wire[p]` and increments `sentStreams[p]`. -/
def sendStep (s : Pipeline) (e : Send.Ev) : Option Pipeline :=
  match e with
  | .d2hIssue p ok =>
    if ok = false ∨ s.armed = true then
      (Send.step .shipping s.send (.d2hIssue p ok)).map fun snd =>
        { s with send := snd }
    else none
  | .d2hDone p ok =>
    (Send.step .shipping s.send (.d2hDone p ok)).map fun snd =>
      if ok then
        { s with send := snd
                 srcStaging := s.srcStaging.set p (s.srcHbm.getD p .junk) }
      else { s with send := snd }
  | .pushDone p ok =>
    (Send.step .shipping s.send (.pushDone p ok)).map fun snd =>
      if ok then
        { s with send := snd
                 wire := s.wire.set p (s.srcStaging.getD p .junk)
                 sentStreams := s.sentStreams.set p (s.sentStreams.getD p 0 + 1) }
      else { s with send := snd }
  | e =>
    (Send.step .shipping s.send e).map fun snd => { s with send := snd }

/-- Receiver transition (`ReshardReceiveSession::OnPoolReceived`,
`ExecuteEligiblePoolH2ds`, `FinishPoolH2d`,
`BlockTransportServerPushSession::CompleteIncomingPush`). `pushBegin` requires the receiver to be
armed (`KVCacheManagerWithTransfer::InitializeBaseHooks`,
`hooks.begin_incoming_push =`); `onPoolReceived p` requires all `peerCounts[p]`
expected streams to have landed (`BlockTransportServerPushSession::CompleteIncomingPush`);
`h2dDone p true` writes `dstStaging[p]` to `dstHbm[p]` and clobbers any strictly
higher-rank pool slots on the aliased page (`writePoolHbm`). -/
def recvStep (s : Pipeline) (e : Recv.Ev) : Option Pipeline :=
  match e with
  | .pushBegin =>
    if s.armed = true then
      (Recv.step .finishOnError s.recv .pushBegin).map fun rcv => { s with recv := rcv }
    else none
  | .onPoolReceived p =>
    if s.landedStreams[p]? = some (s.peerCounts.getD p 0) then
      (Recv.step .finishOnError s.recv (.onPoolReceived p)).map fun rcv =>
        { s with recv := rcv }
    else none
  | .h2dDone p ok =>
    match s.recv.pools[p]? with
    | some pool =>
      (Recv.step .finishOnError s.recv (.h2dDone p ok)).map fun rcv =>
        if ok then
          { s with recv := rcv
                   dstHbm := writePoolHbm s.recv.pools p pool.rank
                               (s.dstStaging.getD p .junk) s.dstHbm }
        else { s with recv := rcv }
    | none => none
  | e =>
    (Recv.step .finishOnError s.recv e).map fun rcv => { s with recv := rcv }

/-- `BlockTransportServerPushSession::StartStream` (`if (expected_size > 0)`,
reached from `BlockTransport::HandleIncomingPush`) lands one push stream for
pool `p` into `dstStaging[p]` and `CompleteIncomingPush` increments
`completed_chunks`. -/
def landStream (s : Pipeline) (p : Nat) : Option Pipeline :=
  if 0 < s.recv.pushes then
    match s.landedStreams[p]?, s.sentStreams[p]?, s.wire[p]? with
    | some k, some m, some c =>
      if k < m ∧ c ≠ .blank then
        some { s with dstStaging := s.dstStaging.set p c
                      landedStreams := s.landedStreams.set p (k + 1) }
      else none
    | _, _, _ => none
  else none

/-- Source engine/registry reclaims producer HBM once
`KVCacheManagerWithTransfer::CompleteReadWithDetails` has published the send
session (`ReshardCoordinator::RunSubPlan`). -/
def reclaim (s : Pipeline) : Option Pipeline :=
  if s.send.published ≠ none ∧ s.reclaimed = false then
    some { s with srcHbm := List.replicate s.numPools .junk, reclaimed := true }
  else none

/-- Released sender `PoolStagingLease` slots may be reused by another transfer
(`KVCacheManagerWithTransfer::PoolReshardPush`, `ReshardSendSession::SettleLocked`). -/
def reseatSrcStaging (s : Pipeline) : Option Pipeline :=
  if s.send.life.hasStaging = false then
    some { s with srcStaging := List.replicate s.numPools .junk }
  else none

/-- Released receiver `PoolStagingLease` slots may be reused by another transfer
(`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`,
`ReshardReceiveSession::ReleaseStagingLocked`). -/
def reseatDstStaging (s : Pipeline) : Option Pipeline :=
  if s.recv.life.hasStaging = false then
    some { s with dstStaging := List.replicate s.numPools .junk }
  else none

def step (s : Pipeline) : Ev → Option Pipeline
  | .armRecv => s.armRecv
  | .send e => s.sendStep e
  | .recv e => s.recvStep e
  | .landStream p => s.landStream p
  | .reclaim => s.reclaim
  | .reseatSrcStaging => s.reseatSrcStaging
  | .reseatDstStaging => s.reseatDstStaging

def sys (ranks peerCounts : List Nat) : System Pipeline Ev :=
  ⟨init ranks peerCounts, step⟩

def sysUnarmed (ranks peerCounts : List Nat) : System Pipeline Ev :=
  ⟨initUnarmed ranks peerCounts, step⟩


end Pipeline

end TpuSyncVerify.Transfer.Reshard

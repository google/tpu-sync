import TpuSyncVerify.Transfer.LayerStaging
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline

/-! # `LayerStaging`: bounded host layer staging state, events, `step`, `Safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`; the module
docstring and source map live in `LayerStaging.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager)

/-- Control location of `TransferSendSession::StartPush`'s sequential D2H loop. -/
inductive SendLoopPhase where
  | nextIter | acquiring | leasedBeforeD2h | exited
  deriving Repr, DecidableEq, Hashable

/-- Per-layer bounded-staging lifecycle on `TransferReceiveSession`. -/
inductive RecvLayerPhase where
  | idle | waiting (hadLease : Bool) | leasedInAcquire | leasedForStream
  | h2dPending | h2dRunning | h2dRetired | aborted
  deriving Repr, DecidableEq, Hashable

/-- End-to-end prefill-to-decode transfer multiplexing `pipe.numLayers` layers over
`sendMgr.numSlots` producer host layer buffers and `recvMgr.numSlots` consumer host
layer buffers. -/
structure BoundedPipeline where
  pipe : Pipeline
  sendMgr : StagingLayerManager
  recvMgr : StagingLayerManager
  prefillSlots : List Cell
  wireB : List Cell
  decodeSlots : List Cell
  decodeHbmB : List Cell
  sendLoop : SendLoopPhase := .nextIter
  recvPhaseL : List RecvLayerPhase
  activeStreams : Nat := 0
  slotFault : Bool := false
  deriving Repr, DecidableEq, Hashable

namespace LayerStaging

/-- Apply `Shutdown()` when `draining` flips `false → true` (`FinishLocked`) and
`ReleaseAll()` when `done` flips `false → true` (`ReleaseStagingBlocksLocked` /
`ReleaseStagingLocked`). -/
abbrev syncMgr := StagingLayerManager.sync

/-- Release layer `l`'s slot and synchronize `Shutdown()` / `ReleaseAll()` with
the session lifecycle transition (`StagingLayerManager.doShutdown_release`). -/
abbrev releaseSync := StagingLayerManager.releaseSync

/-- Resolve `mgr.lookup l` against `slots.length`, setting `slotFault := true` on
missing or out-of-bounds lease (proved unreachable under `Inv`). -/
def withSlot {α : Type} (mgr : StagingLayerManager) (slots : List α) (l : Nat)
    (base : BoundedPipeline) (k : Nat → BoundedPipeline) : BoundedPipeline :=
  match mgr.lookup l with
  | some slot => if slot < slots.length then k slot else { base with slotFault := true }
  | none => { base with slotFault := true }

def withSlotOpt {α : Type} (mgr : StagingLayerManager) (slots : List α) (l : Nat)
    (base : BoundedPipeline) (k : Nat → Option BoundedPipeline) : Option BoundedPipeline :=
  match mgr.lookup l with
  | some slot => if slot < slots.length then k slot else some { base with slotFault := true }
  | none => some { base with slotFault := true }

/-- Lift a pass-through `Pipeline` post-state onto `BoundedPipeline`, syncing
`Shutdown()` and `ReleaseAll()` on whichever session changed lifecycle state. -/
def syncPipe (s : BoundedPipeline) (p : Pipeline) : BoundedPipeline :=
  { s with pipe := p, sendMgr := syncMgr s.pipe.send.life p.send.life s.sendMgr,
           recvMgr := syncMgr s.pipe.recv.life p.recv.life s.recvMgr }

def init (n numSendSlots numRecvSlots : Nat) : BoundedPipeline :=
  { pipe := Pipeline.init n, sendMgr := StagingLayerManager.init numSendSlots,
    recvMgr := StagingLayerManager.init numRecvSlots,
    prefillSlots := List.replicate numSendSlots .junk, wireB := List.replicate n .blank,
    decodeSlots := List.replicate numRecvSlots .junk, decodeHbmB := List.replicate n .junk,
    sendLoop := .nextIter, recvPhaseL := List.replicate n .idle,
    activeStreams := 0, slotFault := false }

def initUnregistered (n numSendSlots numRecvSlots : Nat) : BoundedPipeline :=
  { init n numSendSlots numRecvSlots with pipe := Pipeline.initUnregistered n }

/-- `TransferSendSession::LookupStagingSlot(layer_idx)` (`send.cc`). -/
def sendLookupStagingSlot (s : BoundedPipeline) (l : Nat) : Option Nat :=
  if s.pipe.send.life.done || s.pipe.send.life.draining then none else s.sendMgr.lookup l

/-- Bounded-mode `staging_layer_manager_.Lookup(layer_idx)` precheck in `ExecuteLayerH2d` (`recv.cc`). -/
def recvLookupStagingSlot (s : BoundedPipeline) (l : Nat) : Option Nat := s.recvMgr.lookup l

/-- Events of `BoundedPipeline`: wraps `Pipeline.Ev` and adds the unlocked
`AcquireStagingSlot` steps on both sides plus layer-indexed `h2dDone`. -/
inductive Ev where
  | pipe (e : Pipeline.Ev)
  | sendAcquireBegin
  | sendAcquireDone (timedOut : Bool)
  | recvAcquireBegin (l : Nat)
  | recvAcquireWait (l : Nat) (timedOut : Bool)
  | recvAcquireFinish (l : Nat)
  | h2dDone (l : Nat) (ok : Bool)
  deriving Repr, DecidableEq

namespace Ev
abbrev notifyForRead : Ev := .pipe .notifyForRead
abbrev pullWait : Ev := .pipe .pullWait
abbrev beginPull : Ev := .pipe (.send .beginPull)
abbrev sendStart : Ev := .pipe (.send .start)
abbrev sendD2hBegin : Ev := .pipe (.send .d2hBegin)
abbrev sendD2hIssue (ok : Bool) : Ev := .pipe (.send (.d2hIssue ok))
abbrev d2hReady (l : Nat) : Ev := .pipe (.d2hReady l)
abbrev sendD2hEnd : Ev := .pipe (.send .d2hEnd)
abbrev sendWake (ok : Bool) : Ev := .pipe (.send (.wake ok))
abbrev sendH2hIssue : Ev := .pipe (.send .h2hIssue)
abbrev sendNext : Ev := .pipe (.send .sendNext)
abbrev h2hDone (l : Nat) (ok : Bool) : Ev := .pipe (.h2hDone l ok)
abbrev sendCancel : Ev := .pipe (.send .cancel)
abbrev sendPublish : Ev := .pipe (.send .publish)
abbrev recvPullReply (ok : Bool) : Ev := .pipe (.recv (.pullReply ok))
abbrev recvPushBegin : Ev := .pipe (.recv .pushBegin)
abbrev recvPushEnd : Ev := .pipe (.recv .pushEnd)
abbrev land (l : Nat) : Ev := .pipe (.land l)
abbrev h2dBegin (l : Nat) : Ev := .pipe (.h2dBegin l)
abbrev h2dIssue (l : Nat) (ok : Bool) : Ev := .pipe (.h2dIssue l ok)
abbrev h2dReady (l : Nat) : Ev := .pipe (.h2dReady l)
abbrev recvNetAccount : Ev := .pipe (.recv .netAccount)
abbrev recvPollReady : Ev := .pipe (.recv .pollReady)
abbrev recvCancel : Ev := .pipe (.recv .cancel)
abbrev recvPublish : Ev := .pipe (.recv .publish)
abbrev reclaim : Ev := .pipe .reclaim
abbrev reseatPrefillStaging : Ev := .pipe .reseatPrefillStaging
abbrev reseatDecodeStaging : Ev := .pipe .reseatDecodeStaging
end Ev

/-- `TransferSendSession::AcquireStagingSlot(l)` initial `mu_` check (`send.cc`). -/
def sendAcquireBegin (s : BoundedPipeline) : Option BoundedPipeline :=
  if s.pipe.send.started = true ∧ s.pipe.send.d2hPending = false ∧
      s.pipe.send.d2hIssued < s.pipe.numLayers ∧ s.sendLoop = .nextIter then
    if s.pipe.send.life.done || s.pipe.send.life.draining then
      (Pipeline.step s.pipe (.send .cancel)).map fun p => { syncPipe s p with sendLoop := .exited }
    else some { s with sendLoop := .acquiring }
  else none

/-- `StagingLayerManager::Acquire(l, deadline_, &cancelled_)` in `AcquireStagingSlot`. -/
def sendAcquireDone (s : BoundedPipeline) (timedOut : Bool) : Option BoundedPipeline :=
  if s.sendLoop = .acquiring then
    match s.sendMgr.acquire s.pipe.send.d2hIssued s.pipe.send.life.draining timedOut with
    | none => none
    | some (.leased _ mgr') => some { s with sendMgr := mgr', sendLoop := .leasedBeforeD2h }
    | some .noSlots | some .timedOut | some .cancelled =>
      (Pipeline.step s.pipe (.send .cancel)).map fun p => { syncPipe s p with sendLoop := .exited }
  else none

/-- `AcquireStagingSlot(l)` post-`Acquire` `mu_` re-check + `StartPush`'s `++in_flight_`. -/
def sendD2hBegin (s : BoundedPipeline) : Option BoundedPipeline :=
  if s.sendLoop = .leasedBeforeD2h then
    let l := s.pipe.send.d2hIssued
    if s.pipe.send.life.done || s.pipe.send.life.draining then
      (Pipeline.step s.pipe (.send .cancel)).map fun p =>
        { s with pipe := p, sendMgr := releaseSync s.pipe.send.life p.send.life l s.sendMgr, sendLoop := .exited }
    else (Pipeline.step s.pipe (.send .d2hBegin)).map fun p => { s with pipe := p, sendLoop := .nextIter }
  else none

/-- `D2hSyncDispatch` outcome in `StartPush`. -/
def sendD2hIssue (s : BoundedPipeline) (ok : Bool) : Option BoundedPipeline :=
  let l := s.pipe.send.d2hIssued
  (Pipeline.step s.pipe (.send (.d2hIssue ok))).map fun p =>
    if ok then { s with pipe := p }
    else { s with pipe := p, sendMgr := releaseSync s.pipe.send.life p.send.life l s.sendMgr, sendLoop := .exited }

/-- Layer `l`'s D2H copy writes `prefillHbm[l]` into leased `prefillSlots[slot]`. -/
def d2hReady (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  (Pipeline.step s.pipe (.d2hReady l)).map fun p =>
    withSlot s.sendMgr s.prefillSlots l { s with pipe := p } fun slot =>
      { s with pipe := p, prefillSlots := s.prefillSlots.set slot (s.pipe.prefillHbm.getD l .junk) }

/-- `SendNextLayer(woken)`'s `layer_future->OnReady` callback (`send.cc`). -/
def sendWake (s : BoundedPipeline) (ok : Bool) : Option BoundedPipeline :=
  let l := s.pipe.send.woken
  if s.pipe.d2hReadyL[l]? = some true ∨
      (s.sendLoop = .exited ∧ ok = false ∧ s.pipe.send.life.draining = true ∧
        s.pipe.send.d2hPending = false ∧ s.pipe.send.d2hIssued ≤ l) then
    (Pipeline.step s.pipe (.send (.wake ok))).map fun p =>
      if ok = false ∨ s.pipe.send.life.draining = true then
        { s with pipe := p, sendMgr := releaseSync s.pipe.send.life p.send.life l s.sendMgr }
      else { s with pipe := p }
  else none

/-- `SendNextLayer`'s `push_pool` task up to `H2hWriteDirectAsync` (`send.cc`). -/
def sendH2hIssue (s : BoundedPipeline) : Option BoundedPipeline :=
  let l := s.pipe.send.h2hIssued
  (Pipeline.step s.pipe (.send .h2hIssue)).map fun p =>
    if s.pipe.send.life.draining = true then
      { s with pipe := p, sendMgr := releaseSync s.pipe.send.life p.send.life l s.sendMgr }
    else
      withSlot s.sendMgr s.prefillSlots l { s with pipe := p } fun _ => { s with pipe := p }

/-- `H2hWriteDirectAsync` push completion callback for layer `l` (`send.cc`). -/
def h2hDone (s : BoundedPipeline) (l : Nat) (ok : Bool) : Option BoundedPipeline :=
  (Pipeline.step s.pipe (.h2hDone l ok)).map fun p =>
    let base := { s with pipe := p, sendMgr := releaseSync s.pipe.send.life p.send.life l s.sendMgr }
    if ok then
      withSlot s.sendMgr s.prefillSlots l base fun slot =>
        { base with wireB := s.wireB.set l (s.prefillSlots.getD slot .junk) }
    else base

/-- `BeginIncomingPush` / `EndIncomingPush` on `TransferReceiveSession`. -/
def recvPushBegin (s : BoundedPipeline) : Option BoundedPipeline :=
  (Pipeline.step s.pipe (.recv .pushBegin)).map fun p => { s with pipe := p, activeStreams := s.activeStreams + 1 }

def recvPushEnd (s : BoundedPipeline) : Option BoundedPipeline :=
  if 0 < s.activeStreams then
    (Pipeline.step s.pipe (.recv .pushEnd)).map fun p => { syncPipe s p with activeStreams := s.activeStreams - 1 }
  else none

/-- `TransferReceiveSession::AcquireStagingSlot(l)` first `mu_` lock (`recv.cc`). -/
def recvAcquireBegin (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  match s.recvPhaseL[l]? with
  | some .idle =>
    (Pipeline.step s.pipe (.recv .pushBegin)).map fun p =>
      { s with pipe := p, recvPhaseL := s.recvPhaseL.set l (.waiting false) }
  | some .leasedForStream =>
    if s.pipe.landedL[l]? = some false then
      (Pipeline.step s.pipe (.recv .pushBegin)).map fun p =>
        { s with pipe := p, recvPhaseL := s.recvPhaseL.set l (.waiting true) }
    else none
  | _ => none

/-- `StagingLayerManager::Acquire(l, deadline, &cancelled_)` in `AcquireStagingSlot`. -/
def recvAcquireWait (s : BoundedPipeline) (l : Nat) (timedOut : Bool) : Option BoundedPipeline :=
  match s.recvPhaseL[l]? with
  | some (.waiting hadLease) =>
    let restP : RecvLayerPhase := if hadLease then .leasedForStream else .idle
    match s.recvMgr.acquire l s.pipe.recv.life.draining timedOut with
    | none => none
    | some (.leased _ mgr') => some { s with recvMgr := mgr', recvPhaseL := s.recvPhaseL.set l .leasedInAcquire }
    | some .cancelled =>
      (Pipeline.step s.pipe (.recv .pushEnd)).map fun p => { syncPipe s p with recvPhaseL := s.recvPhaseL.set l restP }
    | some .noSlots | some .timedOut =>
      ((Pipeline.step s.pipe (.recv .cancel)).bind fun p₁ => Pipeline.step p₁ (.recv .pushEnd)).map fun p₂ =>
        { syncPipe s p₂ with recvPhaseL := s.recvPhaseL.set l restP }
  | _ => none

/-- `TransferReceiveSession::AcquireStagingSlot(l)` second `mu_` lock (`recv.cc`). -/
def recvAcquireFinish (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  if s.recvPhaseL[l]? = some .leasedInAcquire then
    (Pipeline.step s.pipe (.recv .pushEnd)).map fun p =>
      if s.pipe.recv.life.done || s.pipe.recv.life.draining then
        { s with pipe := p, recvMgr := releaseSync s.pipe.recv.life p.recv.life l s.recvMgr,
                 recvPhaseL := s.recvPhaseL.set l .aborted }
      else { s with pipe := p, recvPhaseL := s.recvPhaseL.set l .leasedForStream }
  else none

/-- `ForEachPayload` lands `wireB[l]` into leased `decodeSlots[slot]`. -/
def land (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  if 0 < s.activeStreams ∧ s.recvPhaseL[l]? = some .leasedForStream ∧ s.pipe.landedL[l]? = some false then
    match s.wireB[l]? with
    | some c =>
      if c = .blank then none
      else (Pipeline.step s.pipe (.land l)).map fun p =>
        withSlot s.recvMgr s.decodeSlots l { s with pipe := p } fun slot =>
          { s with pipe := p, decodeSlots := s.decodeSlots.set slot c }
    | none => none
  else none

/-- `OnLayerReceived(l)` → `ExecuteLayerH2d(l)` `Lookup(l)` + first `mu_` lock (`recv.cc`). -/
def h2dBegin (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  if 0 < s.activeStreams ∧ s.pipe.landedL[l]? = some true ∧
      s.pipe.claimedL[l]? = some false ∧ s.recvPhaseL[l]? = some .leasedForStream then
    withSlotOpt s.recvMgr s.decodeSlots l s fun _ =>
      if s.pipe.recv.life.done || s.pipe.recv.life.draining then
        some { s with recvMgr := (s.recvMgr.release l).2, recvPhaseL := s.recvPhaseL.set l .aborted }
      else (Pipeline.step s.pipe (.h2dBegin l)).map fun p =>
        { s with pipe := p, recvPhaseL := s.recvPhaseL.set l .h2dPending }
  else none

/-- `ExecuteLayerH2d(l)` second `mu_` lock + `H2dSyncDispatch` (`recv.cc`). -/
def h2dIssue (s : BoundedPipeline) (l : Nat) (ok : Bool) : Option BoundedPipeline :=
  if s.recvPhaseL[l]? = some .h2dPending then
    (Pipeline.step s.pipe (.h2dIssue l ok)).map fun p =>
      if s.pipe.recv.life.done || s.pipe.recv.life.draining || !ok then
        { s with pipe := p, recvMgr := releaseSync s.pipe.recv.life p.recv.life l s.recvMgr,
                 recvPhaseL := s.recvPhaseL.set l .aborted }
      else { s with pipe := p, recvPhaseL := s.recvPhaseL.set l .h2dRunning }
  else none

/-- Layer `l`'s H2D copy writes `decodeSlots[slot]` into `decodeHbmB[l]`. -/
def h2dReady (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  if s.recvPhaseL[l]? = some .h2dRunning ∧ s.pipe.h2dReadyL[l]? = some false then
    (Pipeline.step s.pipe (.h2dReady l)).map fun p =>
      withSlot s.recvMgr s.decodeSlots l { s with pipe := p } fun slot =>
        { s with pipe := p, decodeHbmB := s.decodeHbmB.set l (s.decodeSlots.getD slot .junk) }
  else none

/-- `ExecuteLayerH2d(l)`'s `future.OnReady` callback (`recv.cc`). -/
def h2dDone (s : BoundedPipeline) (l : Nat) (ok : Bool) : Option BoundedPipeline :=
  if s.recvPhaseL[l]? = some .h2dRunning ∧ s.pipe.h2dReadyL[l]? = some true then
    (Pipeline.step s.pipe (.recv (.h2dDone ok))).map fun p =>
      { s with pipe := p, recvMgr := releaseSync s.pipe.recv.life p.recv.life l s.recvMgr,
               recvPhaseL := s.recvPhaseL.set l .h2dRetired }
  else none

/-- Step function for `Pipeline.Ev` events: refines slot-touching events and
forwards all remaining `Pipeline` events through `syncPipe`. -/
def pipeStep (s : BoundedPipeline) : Pipeline.Ev → Option BoundedPipeline
  | .send .d2hBegin => sendD2hBegin s
  | .send (.d2hIssue ok) => sendD2hIssue s ok
  | .send (.wake ok) => sendWake s ok
  | .send .h2hIssue => sendH2hIssue s
  | .d2hReady l => d2hReady s l
  | .h2hDone l ok => h2hDone s l ok
  | .recv .pushBegin => recvPushBegin s
  | .recv .pushEnd => recvPushEnd s
  | .recv (.h2dDone _) => none
  | .land l => land s l
  | .h2dBegin l => h2dBegin s l
  | .h2dIssue l ok => h2dIssue s l ok
  | .h2dReady l => h2dReady s l
  | .reseatPrefillStaging =>
    (Pipeline.step s.pipe .reseatPrefillStaging).map fun p =>
      { s with pipe := p, prefillSlots := List.replicate s.sendMgr.numSlots .junk }
  | .reseatDecodeStaging =>
    (Pipeline.step s.pipe .reseatDecodeStaging).map fun p =>
      { s with pipe := p, decodeSlots := List.replicate s.recvMgr.numSlots .junk }
  | pe => (Pipeline.step s.pipe pe).map (syncPipe s)

def step (s : BoundedPipeline) : Ev → Option BoundedPipeline
  | .pipe pe => pipeStep s pe
  | .sendAcquireBegin => sendAcquireBegin s
  | .sendAcquireDone t => sendAcquireDone s t
  | .recvAcquireBegin l => recvAcquireBegin s l
  | .recvAcquireWait l t => recvAcquireWait s l t
  | .recvAcquireFinish l => recvAcquireFinish s l
  | .h2dDone l ok => h2dDone s l ok

def sys (n numSendSlots numRecvSlots : Nat) : System BoundedPipeline Ev :=
  ⟨init n numSendSlots numRecvSlots, step⟩

def sysUnregistered (n numSendSlots numRecvSlots : Nat) : System BoundedPipeline Ev :=
  ⟨initUnregistered n numSendSlots numRecvSlots, step⟩

/-! ## Observable properties and `Safe` -/

def PublicationCorrect (s : BoundedPipeline) : Prop :=
  s.pipe.recv.published = some true → s.decodeHbmB = Pipeline.good s.pipe.numLayers

def NoSlotFault (s : BoundedPipeline) : Prop := s.slotFault = false

def SlotManagersValid (s : BoundedPipeline) : Prop :=
  s.sendMgr.free.length + s.sendMgr.leases.length = s.sendMgr.numSlots ∧ s.sendMgr.allSlots.Nodup ∧
  s.recvMgr.free.length + s.recvMgr.leases.length = s.recvMgr.numSlots ∧ s.recvMgr.allSlots.Nodup

def SettledSlotsFree (s : BoundedPipeline) : Prop :=
  (s.pipe.send.life.done = true → s.sendMgr.leases = [] ∧ s.sendMgr.numFree = s.sendMgr.numSlots) ∧
  (s.pipe.recv.life.done = true → s.recvMgr.leases = [] ∧ s.recvMgr.numFree = s.recvMgr.numSlots)

def DrainingShutsDown (s : BoundedPipeline) : Prop :=
  (s.pipe.send.life.draining = true → s.sendMgr.shutdown = true) ∧
  (s.pipe.recv.life.draining = true → s.recvMgr.shutdown = true)

def Safe (s : BoundedPipeline) : Prop :=
  Pipeline.Safe s.pipe ∧ PublicationCorrect s ∧ NoSlotFault s ∧
    SlotManagersValid s ∧ SettledSlotsFree s ∧ DrainingShutsDown s

private def decPipeSafe (p : Pipeline) : Decidable (Pipeline.Safe p) := by
  unfold Pipeline.Safe Pipeline.PublicationCorrect Pipeline.DecodeHbmSafe
    Pipeline.PrefillHbmSafe Pipeline.StagingSafe Pipeline.HandshakeSafe Pipeline.NoOpLeak
  infer_instance

instance instDecidableSafe (s : BoundedPipeline) : Decidable (Safe s) := by
  haveI : Decidable (Pipeline.Safe s.pipe) := decPipeSafe s.pipe
  unfold Safe PublicationCorrect NoSlotFault SlotManagersValid SettledSlotsFree DrainingShutsDown
  infer_instance

def violates (s : BoundedPipeline) : Bool := !decide (Safe s)

theorem violates_iff {s : BoundedPipeline} : violates s = false ↔ Safe s := by simp [violates]

end LayerStaging

end TpuSyncVerify.Transfer.PrefillDecode

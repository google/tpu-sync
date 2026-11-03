import TpuSyncVerify.Common.ListAux
import TpuSyncVerify.Transfer.PrefillDecode.Receive
import TpuSyncVerify.Transfer.PrefillDecode.Send

/-! # `Pipeline`: memories, state, events, `step`, properties, `Safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.Pipeline`; the module docstring and
source map live in `Pipeline.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

/-- What a layer slot of a memory holds. -/
inductive Cell where
  | blank
  | kv (layer : Nat)
  | junk
  deriving Repr, DecidableEq, Hashable

structure Pipeline where
  numLayers : Nat
  send : Send
  recv : Recv
  prefillHbm : List Cell
  prefillStaging : List Cell
  wire : List Cell
  decodeStaging : List Cell
  decodeHbm : List Cell
  /-- ghost: layers whose D2H copy has finished -/
  d2hReadyL : List Bool
  /-- ghost: layers whose push callback has run -/
  h2hRetiredL : List Bool
  /-- ghost: layers the transport has landed -/
  landedL : List Bool
  /-- ghost: layers `OnLayerReceived` has entered `ExecuteLayerH2d` for -/
  claimedL : List Bool
  /-- ghost: layers between the first and second lock of `ExecuteLayerH2d` -/
  h2dPendingL : List Bool
  /-- ghost: layers `ExecuteLayerH2d` has dispatched to the device -/
  h2dIssuedL : List Bool
  /-- ghost: layers whose H2D copy has finished -/
  h2dReadyL : List Bool
  reclaimed : Bool := false
  /-- Whether `NotifyForRead` has registered the offer in `send_sessions_[uuid]`. -/
  registered : Bool := true
  /-- Whether `HandlePullStream` arrived before `NotifyForRead` and is waiting in `cv_.WaitWithTimeout`. -/
  pullWaiting : Bool := false
  deriving Repr, DecidableEq, Hashable

namespace Pipeline

/-- A memory holding layers `0 … n-1` correctly. -/
def good (n : Nat) : List Cell := (List.range n).map Cell.kv

/-- A registered send and a `StartRead` receive (`Recv.initLoad`: the path
the Hybrid Bridge uses) for the same `n`-layer request. Prefill HBM holds
the data; every other buffer holds whatever its previous user left. -/
def init (n : Nat) : Pipeline :=
  { numLayers := n, send := Send.init n, recv := Recv.initLoad n,
    prefillHbm := good n, prefillStaging := List.replicate n .junk,
    wire := List.replicate n .blank, decodeStaging := List.replicate n .junk,
    decodeHbm := List.replicate n .junk,
    d2hReadyL := List.replicate n false, h2hRetiredL := List.replicate n false,
    landedL := List.replicate n false, claimedL := List.replicate n false,
    h2dPendingL := List.replicate n false, h2dIssuedL := List.replicate n false,
    h2dReadyL := List.replicate n false }

/-- An `n`-layer transfer before `NotifyForRead` has registered the offer in
`send_sessions_[uuid]`: models the `PullAheadOfRegistration` grace-wait and
unregistered-pull rejection paths in `KVCacheManagerWithTransfer::HandlePullStream`. -/
def initUnregistered (n : Nat) : Pipeline :=
  { init n with registered := false }

inductive Ev where
  | notifyForRead
  | pullWait
  | send (e : Send.Ev)
  | d2hReady (l : Nat)
  | h2hDone (l : Nat) (ok : Bool)
  | recv (e : Recv.Ev)
  | h2dBegin (l : Nat)
  | h2dIssue (l : Nat) (ok : Bool)
  | h2dReady (l : Nat)
  | land (l : Nat)
  | reclaim
  | reseatPrefillStaging
  | reseatDecodeStaging
  deriving Repr, DecidableEq

/-- `KVCacheManagerWithTransfer::NotifyForRead` registers the send session under
`transfer_uuid` in `send_sessions_` (`send_sessions_.try_emplace(uuid, session)`)
and calls `cv_.SignalAll()`. -/
def notifyForRead (s : Pipeline) : Option Pipeline :=
  if s.registered = false then
    some { s with registered := true }
  else none

/-- `KVCacheManagerWithTransfer::HandlePullStream` arrives before `NotifyForRead`
has registered the offer and enters the `cv_.WaitWithTimeout` grace wait. -/
def pullWait (s : Pipeline) : Option Pipeline :=
  if s.registered = false ∧ s.pullWaiting = false ∧ s.recv.pullPending = true then
    some { s with pullWaiting := true }
  else none

/-- A send event with no memory effect. The two that move data are the
layer-indexed `d2hReady l` / `h2hDone l ok` and are disabled here. `beginPull`
(`session->ValidateAndBeginPull` in `HandlePullStream`) requires the offer to be registered in
`send_sessions_` (`s.registered = true`) and clears any grace-wait state
(`pullWaiting := false`). `wake` gets the per-layer guard the counter model
cannot state: `SendNextLayer(woken)` waits on layer `woken`'s own future
(`d2hReadyL[woken]? = some true`), unless layer `woken` was never issued
(`d2hIssued ≤ woken`) and `fail_unissued_layers` resolves its promise on a
draining send with no dispatch in progress. -/
def sendStep (s : Pipeline) (e : Send.Ev) : Option Pipeline :=
  match e with
  | .d2hReady => none
  | .h2hDone _ => none
  | .beginPull =>
    if s.registered = true then
      (Send.step s.send .beginPull).map fun snd => { s with send := snd, pullWaiting := false }
    else none
  | .wake ok =>
    if s.d2hReadyL[s.send.woken]? = some true ∨
        (ok = false ∧ s.send.life.draining = true ∧
          s.send.d2hPending = false ∧ s.send.d2hIssued ≤ s.send.woken) then
      (Send.step s.send (.wake ok)).map fun snd => { s with send := snd }
    else none
  | _ => (Send.step s.send e).map fun snd => { s with send := snd }

/-- The D2H copy of layer `l` finishes: `prefillStaging[l] := prefillHbm[l]`.
Copies are issued in layer order (`for (size_t l = 0; l < total_layers; ++l)` in `StartPush`), so layer `l`'s exists iff
`l < d2hIssued`; which finishes first is up to the device. -/
def d2hReady (s : Pipeline) (l : Nat) : Option Pipeline :=
  if l < s.send.d2hIssued ∧ s.d2hReadyL[l]? = some false then
    (Send.step s.send .d2hReady).map fun snd =>
      { s with send := snd,
               prefillStaging := s.prefillStaging.set l (s.prefillHbm.getD l .junk),
               d2hReadyL := s.d2hReadyL.set l true }
  else none

/-- The push callback for layer `l` in `SendNextLayer` (`H2hWriteDirectAsync`). Pushes are issued in
layer order by the `SendNextLayer` chain, so push `l` exists iff
`l < h2hIssued`; pushes complete in any order. A successful push delivers
`prefillStaging[l]`. -/
def h2hDone (s : Pipeline) (l : Nat) (ok : Bool) : Option Pipeline :=
  if l < s.send.h2hIssued ∧ s.h2hRetiredL[l]? = some false then
    (Send.step s.send (.h2hDone ok)).map fun snd =>
      if ok then
        { s with send := snd, h2hRetiredL := s.h2hRetiredL.set l true,
                 wire := s.wire.set l (s.prefillStaging.getD l .junk) }
      else { s with send := snd, h2hRetiredL := s.h2hRetiredL.set l true }
  else none

/-- A receive event with no memory effect. `h2dBegin`, `h2dIssue` and `h2dReady`
are the layer-indexed events below and are disabled here. A successful pull
handshake reply (`pullReply true`) requires `ValidateAndBeginPull` to have
claimed the producer (`s.send.pullStarted = true`), and any pull reply ends
`HandlePullStream`'s grace wait (`pullWaiting := false`). -/
def recvStep (s : Pipeline) (e : Recv.Ev) : Option Pipeline :=
  match e with
  | .h2dBegin => none
  | .h2dIssue _ => none
  | .h2dReady => none
  | .pullReply ok =>
    if ok = false ∨ s.send.pullStarted = true then
      (Recv.step s.recv (.pullReply ok)).map fun rcv => { s with recv := rcv, pullWaiting := false }
    else none
  | _ => (Recv.step s.recv e).map fun rcv => { s with recv := rcv }

/-- `OnLayerReceived(l)` → `ExecuteLayerH2d(l)` up to its first unlock: fires
once per layer (A1), after layer `l` landed (`if (array_complete && !progress.on_layer_received_called)` in `CompleteIncomingPush`). -/
def h2dBegin (s : Pipeline) (l : Nat) : Option Pipeline :=
  if s.landedL[l]? = some true ∧ s.claimedL[l]? = some false then
    (Recv.step s.recv .h2dBegin).map fun rcv =>
      { s with recv := rcv,
               claimedL := s.claimedL.set l true,
               h2dPendingL := s.h2dPendingL.set l true }
  else none

/-- `ExecuteLayerH2d(l)` from the `if (done_ || draining_)` re-check through
`h2d_futures_.push_back(future)`: consumes
layer `l`'s pending claim from `h2dBegin l` and, if the session is still active
and dispatch succeeds (`ok = true`), marks layer `l`'s copy as issued. -/
def h2dIssue (s : Pipeline) (l : Nat) (ok : Bool) : Option Pipeline :=
  if s.h2dPendingL[l]? = some true then
    (Recv.step s.recv (.h2dIssue ok)).map fun rcv =>
      { s with recv := rcv,
               h2dPendingL := s.h2dPendingL.set l false,
               h2dIssuedL :=
                 if s.recv.life.done || s.recv.life.draining || !ok then s.h2dIssuedL
                 else s.h2dIssuedL.set l true }
  else none

/-- The H2D copy of layer `l` finishes: `decodeHbm[l] := decodeStaging[l]`. -/
def h2dReady (s : Pipeline) (l : Nat) : Option Pipeline :=
  if s.h2dIssuedL[l]? = some true ∧ s.h2dReadyL[l]? = some false then
    (Recv.step s.recv .h2dReady).map fun rcv =>
      { s with recv := rcv,
               decodeHbm := s.decodeHbm.set l (s.decodeStaging.getD l .junk),
               h2dReadyL := s.h2dReadyL.set l true }
  else none

/-- The transport lands layer `l`, inside an accepted push, once something has
been delivered for it. Layers land in any order. -/
def land (s : Pipeline) (l : Nat) : Option Pipeline :=
  if s.recv.pushes ≠ 0 ∧ s.landedL[l]? = some false then
    match s.wire[l]? with
    | some c =>
      if c = .blank then none
      else some { s with decodeStaging := s.decodeStaging.set l c, landedL := s.landedL.set l true }
    | none => none
  else none

/-- The engine frees prefill HBM after the send was reported (as done or as
failed: either way the send has settled, so no copy is reading it). -/
def reclaim (s : Pipeline) : Option Pipeline :=
  if s.send.published ≠ none ∧ s.reclaimed = false then
    some { s with prefillHbm := List.replicate s.numLayers .junk, reclaimed := true }
  else none

/-- Released staging is somebody else's buffer. -/
def reseatPrefillStaging (s : Pipeline) : Option Pipeline :=
  if s.send.life.hasStaging = false then
    some { s with prefillStaging := List.replicate s.numLayers .junk }
  else none

def reseatDecodeStaging (s : Pipeline) : Option Pipeline :=
  if s.recv.life.hasStaging = false then
    some { s with decodeStaging := List.replicate s.numLayers .junk }
  else none

def step (s : Pipeline) : Ev → Option Pipeline
  | .notifyForRead => s.notifyForRead
  | .pullWait => s.pullWait
  | .send e => s.sendStep e
  | .d2hReady l => s.d2hReady l
  | .h2hDone l ok => s.h2hDone l ok
  | .recv e => s.recvStep e
  | .h2dBegin l => s.h2dBegin l
  | .h2dIssue l ok => s.h2dIssue l ok
  | .h2dReady l => s.h2dReady l
  | .land l => s.land l
  | .reclaim => s.reclaim
  | .reseatPrefillStaging => s.reseatPrefillStaging
  | .reseatDecodeStaging => s.reseatDecodeStaging

/-- An `n`-layer transfer. -/
def sys (n : Nat) : System Pipeline Ev := ⟨init n, step⟩

/-- An `n`-layer transfer starting before `NotifyForRead` registers the offer. -/
def sysUnregistered (n : Nat) : System Pipeline Ev := ⟨initUnregistered n, step⟩

/-! ## Properties -/

/-- When the engine is told `done_recving`, decode HBM holds the KV cache. -/
def PublicationCorrect (s : Pipeline) : Prop :=
  s.recv.published = some true → s.decodeHbm = good s.numLayers

/-- Once `poll_stats()` reports `done_recving` or `failed_recving` to the
caller (handing decode HBM back to the engine), no H2D copy is still
dispatching or writing to decode HBM. -/
def DecodeHbmSafe (s : Pipeline) : Prop :=
  s.recv.published ≠ none → s.recv.pending = 0 ∧ s.recv.retired = s.recv.issued

/-- Once `poll_stats()` reports `done_sending` or `failed_recving` to the
caller (handing prefill HBM back to the engine so it may be reclaimed), no D2H
copy is still dispatching or reading from prefill HBM. -/
def PrefillHbmSafe (s : Pipeline) : Prop :=
  s.send.published ≠ none → s.send.d2hPending = false ∧ s.send.d2hRetired = s.send.d2hIssued

/-- Once a session settles and releases its host staging buffer back to TPU
Sync's `StagingBlockAllocator`, no copy or push is still reading or writing
it. -/
def StagingSafe (s : Pipeline) : Prop :=
  (s.send.life.hasStaging = false →
    s.send.d2hPending = false ∧ s.send.d2hRetired = s.send.d2hIssued ∧
    s.send.h2hRetired = s.send.h2hIssued) ∧
  (s.recv.life.hasStaging = false →
    s.recv.pushes = 0 ∧ s.recv.pending = 0 ∧ s.recv.retired = s.recv.issued)

/-- Handshake rendezvous invariant (`KVCacheManagerWithTransfer::NotifyForRead`,
`KVCacheManagerWithTransfer::HandlePullStream`,
`TransferSendSession::ValidateAndBeginPull`):
1. `StartPush` (`send.started = true`) and all D2H/H2H activity require the
   offer to have been registered (`registered = true`) and claimed by
   `ValidateAndBeginPull` (`send.pullStarted = true`).
2. While `HandlePullStream` is in its grace wait (`pullWaiting = true`), the
   consumer's `PullStream` RPC is still in flight (`recv.pullPending = true`)
   and `pull_started_` has not yet been claimed (`send.pullStarted = false`). -/
def HandshakeSafe (s : Pipeline) : Prop :=
  (s.send.started = true → s.send.pullStarted = true ∧ s.registered = true) ∧
  (s.send.pullStarted = true → s.registered = true) ∧
  ((s.send.d2hPending = true ∨ 0 < s.send.d2hIssued ∨ 0 < s.send.h2hIssued) →
    s.send.pullStarted = true ∧ s.registered = true) ∧
  (s.pullWaiting = true → s.recv.pullPending = true ∧ s.send.pullStarted = false)

/-- Events that advance or retire an in-flight operation on either side. -/
def drainEvents (n : Nat) : List Ev :=
  [.send .d2hBegin, .send (.d2hIssue true), .send .d2hEnd,
   .send (.wake true), .send (.wake false),
   .send .h2hIssue, .send .sendNext,
   .recv .pushEnd, .recv (.pullReply false), .recv (.h2dDone true)] ++
  (List.range n).flatMap fun l =>
    [.d2hReady l, .h2hDone l true, .h2dIssue l true, .h2dReady l]

/-- Every unit of `in_flight_` on either side has an owner that can advance or
retire it in the layer-indexed pipeline. -/
def NoOpLeak (s : Pipeline) : Prop :=
  (0 < s.send.life.inFlight ∨ 0 < s.recv.life.inFlight) →
    ∃ e ∈ drainEvents s.numLayers, (step s e).isSome = true

def Safe (s : Pipeline) : Prop :=
  PublicationCorrect s ∧ DecodeHbmSafe s ∧ PrefillHbmSafe s ∧ StagingSafe s ∧
    HandshakeSafe s ∧ NoOpLeak s


end Pipeline

end TpuSyncVerify.Transfer.PrefillDecode

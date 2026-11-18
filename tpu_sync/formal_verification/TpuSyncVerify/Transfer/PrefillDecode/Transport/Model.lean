import TpuSyncVerify.Common.System
import TpuSyncVerify.Common.ListAux
import TpuSyncVerify.Transfer.Session

/-! # `Transport`: parameters, state, events, `step`, properties

Part of `TpuSyncVerify.Transfer.PrefillDecode.Transport`; the module docstring and
source map live in `Transport.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.Transport

open TpuSyncVerify TpuSyncVerify.Transfer

/-! ## Parameters and state -/

structure Cfg where
  numLayers : Nat
  numSenders : Nat
  /-- Streams each sender declares per layer (`header.reserved`). -/
  numStreams : Nat
  /-- Block units a stream adds to the receiver's count. -/
  blocksPerStream : Nat
  /-- `true`: sender-declared contract (`expected_senders = some numSenders`);
  `false`: chunk-count contract (`expected_senders = nullopt`). -/
  declared : Bool
  /-- A4: refuse streams beyond a sender's declaration. -/
  honest : Bool
  deriving Repr, DecidableEq

/-- `BlockTransport::LayerProgress`. -/
structure Progress where
  completedChunks : Nat
  landed : List Nat
  sendersComplete : Nat
  called : Bool
  deriving Repr, DecidableEq

structure Layer where
  entry : Option Progress
  sent : List Nat
  dispatched : Nat
  pending : Nat
  deriving Repr, DecidableEq

structure S where
  layers : List Layer
  blocks : Nat
  completedLayers : Nat
  /-- `in_flight_` / `draining_` / `done_` / `status_` / staging of the receive
  session (`Transfer/Session.lean`). `inFlight` counts only H2D copies whose
  callback has not run: the push lease is taken and returned inside the one
  atomic `stream` event. -/
  life : Lifecycle
  deriving Repr, DecidableEq

def natSum : List Nat → Nat
  | [] => 0
  | n :: ns => n + natSum ns

/-- Blocks an honest run delivers per layer. -/
def Cfg.perLayer (c : Cfg) : Nat := c.numSenders * c.numStreams * c.blocksPerStream

/-- `total_blocks_ * num_layers` (`RecordBlockShardsReceivedLocked`). -/
def Cfg.target (c : Cfg) : Nat := c.perLayer * c.numLayers

/-- Preconditions of the configuration: at least one layer-array sender, a
positive declared stream count (`CompleteIncomingPush`: `if (num_streams == 0)`),
non-empty streams, and `chunkSingle`, which the C++ does not check and which is
a caller obligation on two of the receiver kinds it covers (see the field). -/
structure Cfg.WF (c : Cfg) : Prop where
  senders : 0 < c.numSenders
  streams : 0 < c.numStreams
  blocks : 0 < c.blocksPerStream
  /-- The chunk-count contract is single-sender. On the `StartRead` path no
  receiver plan is registered at all and the pull goes to one endpoint
  (`StartRead(remote_endpoint)` → `TransferReceiveSession::Start`), so there is
  one sender; on the plan path a plan with several `shard_push_schedules` makes
  `ExpectedPushSenders` return their count
  (`TransferPlan::ExpectedPushSenders`: `request_.shard_push_schedules_size()`),
  selecting the
  sender-declared contract. Two receiver kinds run the chunk-count contract
  with nothing in the C++ bounding the number of pushing senders: a receive
  created by `KVCacheManagerWithTransfer::RegisterRecv` (no plan, so
  `ExpectedPushSenders` is `nullopt`) and a receiver plan with
  `shard_push_schedules_size() == 0`. For those, one sender per uuid is an
  obligation on the caller that dispatches the pushes, not something this
  model derives. `trace_chunk_two_senders` shows what the excluded instance
  does. -/
  chunkSingle : c.declared = false → c.numSenders = 1

def freshLayer (c : Cfg) : Layer :=
  { entry := none, sent := List.replicate c.numSenders 0, dispatched := 0, pending := 0 }

/-- Initial state for a fresh `uuid` (**A5**): no prior `active_push_sessions_[uuid]`
entry (`entry = none` on every layer) and no concurrent `ForgetPushProgress(uuid)`
in flight from retiring an earlier session or plan on the same `uuid`. -/
def init (c : Cfg) : S :=
  { layers := List.replicate c.numLayers (freshLayer c),
    blocks := 0, completedLayers := 0, life := {} }

/-- `network_completed_` (`RecordBlockShardsReceivedLocked`), derived. -/
def networkCompleted (c : Cfg) (s : S) : Bool := decide (c.target ≤ s.blocks)

/-- What `CompleteReadWithDetails` reports in `done_recving_`. -/
def published (s : S) : Bool := s.life.done && s.life.statusOk

/-! ## Events -/

inductive Ev where
  | stream (layer sender : Nat)
  | h2dDone (layer : Nat) (ok : Bool)
  | publish
  | cancel
  deriving Repr, DecidableEq

def freshProgress (c : Cfg) : Progress :=
  { completedChunks := 0, landed := List.replicate c.numSenders 0,
    sendersComplete := 0, called := false }

/-- `CompleteIncomingPush` on one table entry (`auto& progress = layer_progress_[l]` ..
`trigger_completion = true;`). Returns the updated entry, whether this stream
completed the layer (`trigger_completion`), and whether the push was rejected.
In the sender-declared contract the `++streams.landed` increment persists even
when the over-count check (`else if (streams.landed > streams.declared)`) that
follows it rejects the push. -/
def completePush (c : Cfg) (p : Progress) (sender : Nat) : Progress × Bool × Bool :=
  if c.declared then
    let landed := p.landed.getD sender 0 + 1
    let sendersComplete :=
      if landed = c.numStreams then p.sendersComplete + 1 else p.sendersComplete
    let reject := decide (c.numStreams < landed)
    let complete := decide (sendersComplete = c.numSenders)
    let fire := complete && !reject && !p.called
    ({ p with completedChunks := p.completedChunks + 1,
              landed := p.landed.set sender landed,
              sendersComplete := sendersComplete,
              called := p.called || (complete && !reject) },
     fire, reject)
  else
    let completed := p.completedChunks + 1
    let complete := decide (completed = c.numStreams)
    ({ p with completedChunks := completed, called := p.called || complete },
     complete && !p.called, false)

def Layer.entryCalled (L : Layer) : Bool :=
  match L.entry with
  | some p => p.called
  | none => false

/-- Retirement (`if (all_layers_called)` in `CompleteIncomingPush`): once every
layer's latch is set the push session is done (`done_ = true`), and the next
stream of the uuid gets a fresh session with a fresh table
(`GetOrCreatePushSession`, `DeletePushSession`). -/
def retire (layers : List Layer) : List Layer :=
  if layers.all Layer.entryCalled then layers.map (fun L => { L with entry := none })
  else layers

/-- The layer after an accepted stream: the updated table entry, the ghost
stream count, and — when the stream completed the layer — the `ExecuteLayerH2d`
entry and its in-flight copy. -/
def nextLayer (L : Layer) (p' : Progress) (sent' : List Nat) (fire : Bool) : Layer :=
  { entry := some p', sent := sent',
    dispatched := if fire then L.dispatched + 1 else L.dispatched,
    pending := if fire then L.pending + 1 else L.pending }

@[simp] theorem nextLayer_entry (L : Layer) (p' : Progress) (s' : List Nat) (f : Bool) :
    (nextLayer L p' s' f).entry = some p' := rfl
@[simp] theorem nextLayer_sent (L : Layer) (p' : Progress) (s' : List Nat) (f : Bool) :
    (nextLayer L p' s' f).sent = s' := rfl
@[simp] theorem nextLayer_dispatched (L : Layer) (p' : Progress) (s' : List Nat) (f : Bool) :
    (nextLayer L p' s' f).dispatched = if f then L.dispatched + 1 else L.dispatched := rfl
@[simp] theorem nextLayer_pending (L : Layer) (p' : Progress) (s' : List Nat) (f : Bool) :
    (nextLayer L p' s' f).pending = if f then L.pending + 1 else L.pending := rfl

/-- The session side of one stream, given the transport's verdict. Push lease
`TransferReceiveSession::TryBeginOp` (`beginOp`, from `BeginIncomingPush`); on
completion `ExecuteLayerH2d` takes the copy's op (`++in_flight_`, `fire`); a
rejection runs the cleanup's `EndIncomingPush(session, InternalError)` →
`Finish(status)`; the lease is returned by `TransferReceiveSession::EndOp`
(`EndIncomingPush`, on either path). -/
def streamLife (l : Lifecycle) (fire reject : Bool) : Lifecycle :=
  let l := if fire then { l with inFlight := l.inFlight + 1 } else l
  let l := if reject then l.finishLocked false else l
  l.endOpLocked

/-- `HandleIncomingPush` for one stream of layer `l` from sender `sender`. -/
def streamStep (c : Cfg) (s : S) (l sender : Nat) : Option S :=
  match s.life.beginOp, s.layers[l]? with
  | some life₁, some L =>
    if c.numSenders ≤ sender then none
    else if c.honest && c.numStreams ≤ L.sent.getD sender 0 then none
    else
      match completePush c (L.entry.getD (freshProgress c)) sender with
      | (p', fire, reject) =>
        let sent' := L.sent.set sender (L.sent.getD sender 0 + 1)
        if reject then
          some { s with layers := s.layers.set l { L with entry := some p', sent := sent' },
                        life := streamLife life₁ false true }
        else
          some { s with layers := retire (s.layers.set l (nextLayer L p' sent' fire)),
                        life := streamLife life₁ fire false,
                        blocks := s.blocks + c.blocksPerStream }
  | _, _ => none

def Layer.copyDone (L : Layer) : Layer := { L with pending := L.pending - 1 }

@[simp] theorem Layer.copyDone_pending (L : Layer) : L.copyDone.pending = L.pending - 1 := rfl
@[simp] theorem Layer.copyDone_dispatched (L : Layer) : L.copyDone.dispatched = L.dispatched := rfl
@[simp] theorem Layer.copyDone_sent (L : Layer) : L.copyDone.sent = L.sent := rfl
@[simp] theorem Layer.copyDone_entry (L : Layer) : L.copyDone.entry = L.entry := rfl

/-- The H2D `future.OnReady` callback in `ExecuteLayerH2d`: on success
`num_completed_layers_++` and, at `num_layers()` while not draining, the
success `FinishLocked()`; on failure `FinishLocked(status_or.status(), kH2dPhase)`;
then `absl::Cleanup end_op`'s `EndOp()`. -/
def callbackLife (c : Cfg) (l : Lifecycle) (ok : Bool) (completed : Nat) : Lifecycle :=
  let l :=
    if ok then
      if completed = c.numLayers ∧ l.draining = false then l.finishLocked true else l
    else l.finishLocked false
  l.endOpLocked

def h2dDoneStep (c : Cfg) (s : S) (l : Nat) (ok : Bool) : Option S :=
  match s.layers[l]? with
  | none => none
  | some L =>
    if L.pending = 0 then none
    else
      let completed := if ok then s.completedLayers + 1 else s.completedLayers
      some { s with layers := s.layers.set l L.copyDone,
                    completedLayers := completed,
                    life := callbackLife c s.life ok completed }

/-- The poll (`CompleteReadWithDetails`): a receive that is not draining and
`IsReadyToComplete()` is finished with OK. `AllH2dDoneLocked`
(every issued future ready) is taken as "every callback has run"; the window
between the two is `ReceivePoll.lean`'s subject and changes nothing here. -/
def publishStep (c : Cfg) (s : S) : Option S :=
  if s.life.draining then none
  else if (networkCompleted c s || decide (s.completedLayers = c.numLayers)) &&
          s.layers.all (fun L => decide (L.pending = 0)) then
    some { s with life := s.life.finishLocked true }
  else none

/-- The deadline sweep in `CompleteReadWithDetails` (`else if (session->deadline() <= now)`)
or any other `Finish(error)`. -/
def cancelStep (s : S) : Option S :=
  if s.life.draining then none else some { s with life := s.life.finishLocked false }

def step (c : Cfg) (s : S) : Ev → Option S
  | .stream l sender => streamStep c s l sender
  | .h2dDone l ok => h2dDoneStep c s l ok
  | .publish => publishStep c s
  | .cancel => cancelStep s

def sys (c : Cfg) : System S Ev := ⟨init c, step c⟩

/-! ## Properties -/

def OnceDispatched (s : S) : Prop := ∀ L ∈ s.layers, L.dispatched ≤ 1

def NetworkSound (c : Cfg) (s : S) : Prop :=
  networkCompleted c s = true → ∀ L ∈ s.layers, 1 ≤ L.dispatched

def CompleteOnSuccess (s : S) : Prop :=
  published s = true → ∀ L ∈ s.layers, L.dispatched = 1 ∧ L.pending = 0

def Safe (c : Cfg) (s : S) : Prop := OnceDispatched s ∧ NetworkSound c s ∧ CompleteOnSuccess s

instance (c : Cfg) (s : S) : Decidable (Safe c s) := by
  unfold Safe OnceDispatched NetworkSound CompleteOnSuccess
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorem check the same predicate. -/
def violates (c : Cfg) (s : S) : Bool := !decide (Safe c s)

theorem violates_iff (c : Cfg) (s : S) : violates c s = false ↔ Safe c s := by
  simp [violates]


end TpuSyncVerify.Transfer.PrefillDecode.Transport

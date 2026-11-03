import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.Transport.Invariant

/-! # `Transport`: configurations, replay, bounded search, candidate hardening

Part of `TpuSyncVerify.Transfer.PrefillDecode.Transport`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.Transport

open TpuSyncVerify TpuSyncVerify.Transfer

/-! ## Configurations and event lists for replay and search -/

def cfgChunk (numLayers numSenders numStreams : Nat) (honest : Bool) : Cfg :=
  { numLayers, numSenders, numStreams, blocksPerStream := 1, declared := false, honest }

def cfgDeclared (numLayers numSenders numStreams : Nat) (honest : Bool) : Cfg :=
  { numLayers, numSenders, numStreams, blocksPerStream := 1, declared := true, honest }

def events (c : Cfg) : List Ev :=
  ((List.range c.numLayers).flatMap fun l =>
    (List.range c.numSenders).map fun s => Ev.stream l s) ++
  ((List.range c.numLayers).flatMap fun l => [Ev.h2dDone l true, Ev.h2dDone l false]) ++
  [.publish, .cancel]

/-- Bounded search for a `Safe` violation on configuration `c`. -/
def search (c : Cfg) (fuel : Nat) : ModelCheck.Result Ev :=
  ModelCheck.check (sys c) (events c) (violates c) fuel

def foundCounterexample (c : Cfg) (fuel : Nat) : Bool :=
  match search c fuel with
  | .counterexample _ => true
  | _ => false

/-- Projection used by the traces: the session outcome and the per-layer
dispatch and pending counts. -/
structure View where
  draining : Bool
  done : Bool
  statusOk : Bool
  dispatched : List Nat
  pending : List Nat
  deriving Repr, DecidableEq

def view (s : S) : View :=
  ⟨s.life.draining, s.life.done, s.life.statusOk,
   s.layers.map (·.dispatched), s.layers.map (·.pending)⟩

/-! ### Honest senders: exhaustive on small instances -/

#guard search (cfgChunk 2 1 2 true) 40 = .safe
#guard search (cfgChunk 3 1 1 true) 40 = .safe
#guard search (cfgChunk 2 1 3 true) 40 = .safe
#guard search (cfgDeclared 2 2 1 true) 40 = .safe
#guard search (cfgDeclared 2 1 2 true) 40 = .safe
#guard search (cfgDeclared 2 2 2 true) 60 = .safe

/-- Honest run, chunk-count contract, two layers of two streams: both layers
dispatched once, the network completes exactly when the last stream lands,
the table is retired, and the second callback settles the receive with OK. -/
theorem trace_honest_chunk :
    ((sys (cfgChunk 2 1 2 true)).run
      [.stream 0 0, .stream 0 0, .stream 1 0, .stream 1 0, .h2dDone 0 true, .h2dDone 1 true]).map
      (fun s => (view s, s.blocks, s.layers.map (·.entry))) =
    some (⟨true, true, true, [1, 1], [0, 0]⟩, 4, [none, none]) := by
  decide

/-- Honest run, sender-declared contract, two senders: a layer is dispatched
only when both senders' streams have landed. -/
theorem trace_honest_declared :
    ((sys (cfgDeclared 2 2 1 true)).run [.stream 0 0, .stream 1 0]).map
      (fun s => s.layers.map (·.dispatched)) = some [0, 0] ∧
    ((sys (cfgDeclared 2 2 1 true)).run
      [.stream 0 0, .stream 1 0, .stream 0 1, .stream 1 1, .h2dDone 0 true, .h2dDone 1 true]).map
      view = some ⟨true, true, true, [1, 1], [0, 0]⟩ := by
  decide

/-- A failed copy fails the receive once every op has returned; the poll never
publishes it. -/
theorem trace_failed_copy :
    ((sys (cfgChunk 2 1 1 true)).run
      [.stream 0 0, .stream 1 0, .h2dDone 0 false, .h2dDone 1 true]).map view =
    some ⟨true, true, false, [1, 1], [0, 0]⟩ ∧
    (sys (cfgChunk 2 1 1 true)).run
      [.stream 0 0, .stream 1 0, .h2dDone 0 false, .publish] = none := by
  decide

/-- **Out of contract: chunk-count with two senders.** `Cfg.WF.chunkSingle`
excludes this instance; the trace shows why. `expected_chunks` is the stream
count one sender declared (`header.reserved`), so sender 1's single stream
completes each layer on its own, the table is retired once sender 1 has
covered both layers, and sender 0's first (honest) stream for layer 0 then
starts a fresh entry and dispatches layer 0 a second time. -/
theorem trace_chunk_two_senders :
    ((sys (cfgChunk 2 2 1 true)).run [.stream 1 1, .stream 0 1, .stream 0 0]).map
      (fun s => s.layers.map (·.dispatched)) = some [2, 1] := by
  decide

/-! ### Without A4: one extra stream -/

/-- **Chunk-count contract, before retirement.** Layer 0 (one declared stream)
lands twice; layer 1 never arrives. The duplicate is absorbed
(`completed_chunks = 2 ≠ 1`, no second dispatch, no error) but still reported
through `OnBlockShardsReceived`, so `network_completed_` is reached with
layer 1's blocks never received. Once layer 0's copy lands, `IsReadyToComplete`
is true, the poll's `Finish()` settles at once (`in_flight_ == 0`) and the
receive is published in `done_recving_` with layer 1 never copied. -/
theorem trace_dup_chunk_publishes_missing_layer :
    ((sys (cfgChunk 2 1 1 false)).run
      [.stream 0 0, .stream 0 0, .h2dDone 0 true, .publish]).map
      (fun s => (published s, networkCompleted (cfgChunk 2 1 1 false) s,
                 s.layers.map (·.dispatched))) =
    some (true, true, [1, 0]) := by
  decide

/-- **Either contract, after retirement.** Both layers land and the table
entries are erased (`if (all_layers_called)` in `CompleteIncomingPush`). An
extra stream for layer 0 then creates a fresh entry whose first stream
completes it: `OnLayerReceived` fires again and a second copy of layer 0 is
issued while the first may still run (`dispatched = [2, 1]`). The session is
not fooled about settlement — `in_flight_` holds `done_` off until every copy
has returned — but `num_completed_layers_` reaches `num_layers()` after the two
layer-0 callbacks, so the receive starts draining with layer 1's copy still
pending, and settles with OK once it returns. -/
theorem trace_dup_after_retirement :
    ((sys (cfgChunk 2 1 1 false)).run
      [.stream 0 0, .stream 1 0, .stream 0 0, .h2dDone 0 true, .h2dDone 0 true]).map view =
    some ⟨true, false, true, [2, 1], [0, 1]⟩ ∧
    ((sys (cfgChunk 2 1 1 false)).run
      [.stream 0 0, .stream 1 0, .stream 0 0, .h2dDone 0 true, .h2dDone 0 true,
       .h2dDone 1 true]).map view =
    some ⟨true, true, true, [2, 1], [0, 0]⟩ ∧
    ((sys (cfgDeclared 2 1 1 false)).run
      [.stream 0 0, .stream 1 0, .stream 0 0, .h2dDone 0 true, .h2dDone 0 true,
       .h2dDone 1 true]).map view =
    some ⟨true, true, true, [2, 1], [0, 0]⟩ := by
  decide

/-- The redundant copy can also fail the receive: `FinishLocked` records the
first error even when already draining (`if (!status.ok() && status_.ok())`),
so a duplicate copy that fails after the honest copies succeeded turns a
complete transfer into `failed_recving`. -/
theorem trace_dup_copy_fails_late :
    ((sys (cfgChunk 2 1 1 false)).run
      [.stream 0 0, .stream 1 0, .stream 0 0, .h2dDone 0 true, .h2dDone 1 true,
       .h2dDone 0 false]).map view =
    some ⟨true, true, false, [2, 1], [0, 0]⟩ := by
  decide

/-- **Sender-declared contract, before retirement.** The same duplicate is
rejected (`else if (streams.landed > streams.declared)` in
`CompleteIncomingPush`), the push fails and the receive is finished with an
error: loud, not silent. -/
theorem trace_dup_declared_rejected :
    ((sys (cfgDeclared 2 1 1 false)).run [.stream 0 0, .stream 0 0]).map view =
      some ⟨true, false, false, [1, 0], [1, 0]⟩ ∧
    (sys (cfgDeclared 2 1 1 false)).run [.stream 0 0, .stream 0 0, .stream 1 0] = none := by
  decide

#guard foundCounterexample (cfgChunk 2 1 1 false) 6
#guard foundCounterexample (cfgDeclared 2 1 1 false) 6

/-! ### Candidate hardening

Two small changes close every trace above without touching the honest
behaviour: (1) the chunk-count contract rejects a stream beyond the declared
count, as the other two contracts already do (`if (plan_pushes > expected_chunks)`
and `else if (streams.landed > streams.declared)` in `CompleteIncomingPush`);
(2) `ExecuteLayerH2d` keeps a per-layer once-latch of its own, so a dispatch
after the transport retired its entries is a no-op. `stepHardened` is `step`
with exactly those two edits. The dishonest state space stays infinite (after
retirement a fresh entry's first stream is within the declared count and is
accepted, so the block count is unbounded), so the searches below are bounded:
no violation within the depth, where the unhardened step finds one at depth 4. -/

def completePushHardened (c : Cfg) (p : Progress) (sender : Nat) : Progress × Bool × Bool :=
  if c.declared then completePush c p sender
  else
    let completed := p.completedChunks + 1
    let complete := decide (completed = c.numStreams)
    ({ p with completedChunks := completed, called := p.called || complete },
     complete && !p.called, decide (c.numStreams < completed))

def streamStepHardened (c : Cfg) (s : S) (l sender : Nat) : Option S :=
  match s.life.beginOp, s.layers[l]? with
  | some life₁, some L =>
    if c.numSenders ≤ sender then none
    else if c.honest && c.numStreams ≤ L.sent.getD sender 0 then none
    else
      match completePushHardened c (L.entry.getD (freshProgress c)) sender with
      | (p', fire, reject) =>
        let sent' := L.sent.set sender (L.sent.getD sender 0 + 1)
        if reject then
          some { s with layers := s.layers.set l { L with entry := some p', sent := sent' },
                        life := streamLife life₁ false true }
        else
          -- session-side once-latch: a second `OnLayerReceived` issues no copy
          let fire := fire && decide (L.dispatched = 0)
          some { s with layers := retire (s.layers.set l (nextLayer L p' sent' fire)),
                        life := streamLife life₁ fire false,
                        blocks := s.blocks + c.blocksPerStream }
  | _, _ => none

def stepHardened (c : Cfg) (s : S) : Ev → Option S
  | .stream l sender => streamStepHardened c s l sender
  | e => step c s e

def searchHardened (c : Cfg) (fuel : Nat) : ModelCheck.Result Ev :=
  ModelCheck.check ⟨init c, stepHardened c⟩ (events c) (violates c) fuel

-- Bounded only: with dishonest senders the state space is unbounded (still
-- `outOfFuel` at fuel 40), so these say "no violation within 12 events", which
-- is the only evidence for the hardening; the honest configs are proved above.
#guard searchHardened (cfgChunk 2 1 1 false) 12 = .outOfFuel
#guard searchHardened (cfgChunk 2 1 2 false) 12 = .outOfFuel
#guard searchHardened (cfgDeclared 2 1 1 false) 12 = .outOfFuel
#guard searchHardened (cfgDeclared 2 2 1 false) 12 = .outOfFuel

/-- The hardened step still runs the honest traces to the same end states. -/
theorem trace_hardened_honest :
    ((⟨init (cfgChunk 2 1 2 true), stepHardened (cfgChunk 2 1 2 true)⟩ : System S Ev).run
      [.stream 0 0, .stream 0 0, .stream 1 0, .stream 1 0, .h2dDone 0 true, .h2dDone 1 true]).map
      view = some ⟨true, true, true, [1, 1], [0, 0]⟩ := by
  decide

/-- Under the hardening the pre-retirement duplicate fails the receive loudly
and the post-retirement duplicate issues no second copy. -/
theorem trace_hardened_dups :
    ((⟨init (cfgChunk 2 1 1 false), stepHardened (cfgChunk 2 1 1 false)⟩ : System S Ev).run
      [.stream 0 0, .stream 0 0]).map view = some ⟨true, false, false, [1, 0], [1, 0]⟩ ∧
    ((⟨init (cfgChunk 2 1 1 false), stepHardened (cfgChunk 2 1 1 false)⟩ : System S Ev).run
      [.stream 0 0, .stream 1 0, .stream 0 0, .h2dDone 0 true]).map view =
    some ⟨false, false, true, [1, 1], [0, 1]⟩ := by
  decide

/-! ## Replay of upstream transport tests

The transport unit tests drive `CompleteIncomingPush` through mock delegates
(`MockSessionDelegate` in `block_transport_session_test.cc`, `MockDelegate` and
`MultiSenderReceiverDelegate` in `block_transport_test.cc`) with no
`TransferReceiveSession`, so what they observe is the table: `OnLayerReceived`
calls (`dispatched`), retirement (`entry = none`), and rejected push streams
(where `CompleteIncomingPush` returns `AlreadyExistsError` or
`FailedPreconditionError` and `StartStream` drops the stream before the 1-byte
ack, recorded on `step` as `life.statusOk = false`). The traces below replay
them on `step` with those projections. -/

/-- `BlockTransportServerPushSessionTest.LifecycleAndStreamCompletion`: one
layer, one sender declaring two streams (`num_streams = 2`), chunk-count
contract. The first stream completes nothing; the second fires
`OnLayerReceived` once (`layers_received() == 1`) and, every layer now called,
retires the table (`IsDone()`, `on_completed_`). -/
def tr_session_two_streams_complete : List Ev := [.stream 0 0, .stream 0 0]

theorem trace_session_two_streams_complete :
    ((sys (cfgChunk 1 1 2 true)).run (tr_session_two_streams_complete.take 1)).map
      (fun s => (s.layers.map (·.dispatched), s.layers.map (·.entry.isSome))) =
      some ([0], [true]) ∧
    ((sys (cfgChunk 1 1 2 true)).run tr_session_two_streams_complete).map
      (fun s => (s.layers.map (·.dispatched), s.layers.map (·.entry.isSome))) =
      some ([1], [false]) := by
  decide

/-- `BlockTransportTest.LayerCompletesOnlyAfterEveryDeclaredSender`: two
layers, two declared senders (`ExpectSenders(kUuid, 2)`), one stream each. A
layer completes only when both senders' streams have landed, in either order
(`layer_completion_count()` 0, 1, 1, 2); once both layers are called the uuid's
progress is retired, and the same two senders complete layer 0 again on a fresh
entry (count 3). The reuse after retirement is a stream beyond each sender's
per-layer declaration, so the run is on `honest = false`. -/
def tr_declared_senders_then_reuse : List Ev :=
  [.stream 0 0, .stream 0 1, .stream 1 1, .stream 1 0, .stream 0 0, .stream 0 1]

theorem trace_declared_senders_then_reuse :
    let c := cfgDeclared 2 2 1 false
    ((sys c).run (tr_declared_senders_then_reuse.take 1)).map
      (fun s => s.layers.map (·.dispatched)) = some [0, 0] ∧
    ((sys c).run (tr_declared_senders_then_reuse.take 2)).map
      (fun s => s.layers.map (·.dispatched)) = some [1, 0] ∧
    ((sys c).run (tr_declared_senders_then_reuse.take 3)).map
      (fun s => s.layers.map (·.dispatched)) = some [1, 0] ∧
    ((sys c).run (tr_declared_senders_then_reuse.take 4)).map
      (fun s => (s.layers.map (·.dispatched), s.layers.map (·.entry.isSome))) =
      some ([1, 1], [false, false]) ∧
    ((sys c).run (tr_declared_senders_then_reuse.take 5)).map
      (fun s => s.layers.map (·.dispatched)) = some [1, 1] ∧
    ((sys c).run tr_declared_senders_then_reuse).map
      (fun s => s.layers.map (·.dispatched)) = some [2, 1] := by
  decide

/-- `BlockTransportTest.SenderOverDeliveryIsRejected`: two layers, two declared
senders, one stream each. Both senders complete layer 0; layer 1 is still open,
so the uuid's progress is live, and a second stream from sender 0 for layer 0
exceeds the one stream it declared: `CompleteIncomingPush` rejects the push
stream (`else if (streams.landed > streams.declared)`, recorded on `step` as
`life.statusOk = false`) and does not dispatch the layer again. -/
def tr_sender_over_delivery_rejected : List Ev := [.stream 0 0, .stream 0 1, .stream 0 0]

theorem trace_sender_over_delivery_rejected :
    let c := cfgDeclared 2 2 1 false
    ((sys c).run (tr_sender_over_delivery_rejected.take 2)).map
      (fun s => (s.layers.map (·.dispatched), s.life.statusOk)) = some ([1, 0], true) ∧
    ((sys c).run tr_sender_over_delivery_rejected).map
      (fun s => (s.layers.map (·.dispatched), s.life.statusOk)) = some ([1, 0], false) := by
  decide

/-- `BlockTransportTest.UndeclaredUuidKeepsHeaderDeclaredCompletion`: one
layer, two senders, but `ExpectSenders` was called for another uuid, so
`ExpectedPushSenders` is `nullopt` and the layer runs the chunk-count contract
with `expected_chunks = 1`: each sender's single stream completes the layer by
itself (`layer_completion_count()` 1 then 2), the first retiring the entry the
second recreates. This is `trace_chunk_two_senders`'s out-of-contract instance
(`Cfg.WF.chunkSingle`), produced here by a test that deliberately leaves the
uuid undeclared. -/
def tr_undeclared_uuid_chunk_count : List Ev := [.stream 0 0, .stream 0 1]

theorem trace_undeclared_uuid_chunk_count :
    ((sys (cfgChunk 1 2 1 true)).run (tr_undeclared_uuid_chunk_count.take 1)).map
      (fun s => s.layers.map (·.dispatched)) = some [1] ∧
    ((sys (cfgChunk 1 2 1 true)).run tr_undeclared_uuid_chunk_count).map
      (fun s => s.layers.map (·.dispatched)) = some [2] := by
  decide

/-! Sibling push-progress tests whose stimulus is outside this model's event set:
- `BlockTransportTest.DeclaredSendersEachCompleteTheirOwnParallelism` (sender 0
  declares two streams and sender 1 one for the same layer) and
  `BlockTransportTest.SenderChangingDeclaredStreamCountIsRejected` (one sender
  declares one stream and then two, `else if (streams.declared != num_streams)`):
  both vary per-sender declarations, which A3 fixes to one `numStreams`.
- `BlockTransportServerPushSessionTest.Cancel` (`BlockTransportServerPushSession::Cancel`
  on a push session with no receive session behind it) and
  `BlockTransportTest.ForgetPushProgressAllowsUuidReuse` (clearing a partial
  generation with `BlockTransport::ForgetPushProgress` before checking
  auto-retirement across a full generation): `cancel` here is the receive
  session's `Finish(error)`, while push-session cancellation / `ForgetPushProgress`
  is not an event (`Transport.lean` §"Lock scopes").
- `BlockTransportTest.PushSplitBySourceNumaCompletesPlanDeclaredPool`,
  `BlockTransportTest.PoolProgressWaitsForEveryStreamOfEveryDeclaredPoolAndRetires`,
  and `BlockTransportTest.ForgetPushProgressResetsPartialPoolGeneration` exercise
  the plan-declared pool contract (`Reshard/Pipeline.lean` `landStream`), and
  `PeregrineRecvCompleteTest` exercises `OnPeregrinePostComplete` and the
  `is_peregrine = true` shard-bitmask branch of `CompleteIncomingPush`. -/

end TpuSyncVerify.Transfer.PrefillDecode.Transport

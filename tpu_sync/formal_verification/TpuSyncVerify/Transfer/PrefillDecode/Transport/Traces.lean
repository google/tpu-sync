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

/-- Honest chunk-count run (`2` layers, `2` streams each): dispatches each layer
once, retires the progress table, and settles OK. -/
theorem trace_honest_chunk :
    ((sys (cfgChunk 2 1 2 true)).run
      [.stream 0 0, .stream 0 0, .stream 1 0, .stream 1 0, .h2dDone 0 true, .h2dDone 1 true]).map
      (fun s => (view s, s.blocks, s.layers.map (·.entry))) =
    some (⟨true, true, true, [1, 1], [0, 0]⟩, 4, [none, none]) := by
  decide

/-- Honest sender-declared run (`2` senders): dispatches each layer only after
both senders' streams have landed. -/
theorem trace_honest_declared :
    ((sys (cfgDeclared 2 2 1 true)).run [.stream 0 0, .stream 1 0]).map
      (fun s => s.layers.map (·.dispatched)) = some [0, 0] ∧
    ((sys (cfgDeclared 2 2 1 true)).run
      [.stream 0 0, .stream 1 0, .stream 0 1, .stream 1 1, .h2dDone 0 true, .h2dDone 1 true]).map
      view = some ⟨true, true, true, [1, 1], [0, 0]⟩ := by
  decide

/-- A failed H2D copy settles the receive failed once all ops end; `publish` is disabled. -/
theorem trace_failed_copy :
    ((sys (cfgChunk 2 1 1 true)).run
      [.stream 0 0, .stream 1 0, .h2dDone 0 false, .h2dDone 1 true]).map view =
    some ⟨true, true, false, [1, 1], [0, 0]⟩ ∧
    (sys (cfgChunk 2 1 1 true)).run
      [.stream 0 0, .stream 1 0, .h2dDone 0 false, .publish] = none := by
  decide

/-- Out of contract (`Cfg.WF.chunkSingle` violated): on chunk-count with two
senders, sender 1 retires the table alone and sender 0 re-dispatches layer 0. -/
theorem trace_chunk_two_senders :
    ((sys (cfgChunk 2 2 1 true)).run [.stream 1 1, .stream 0 1, .stream 0 0]).map
      (fun s => s.layers.map (·.dispatched)) = some [2, 1] := by
  decide

/-! ### Without A4: one extra stream -/

/-- Dishonest chunk-count before retirement: a duplicate layer-0 stream bumps
`RecordBlockShards` to `network_completed_`, publishing `done_recving` with layer 1 uncopied. -/
theorem trace_dup_chunk_publishes_missing_layer :
    ((sys (cfgChunk 2 1 1 false)).run
      [.stream 0 0, .stream 0 0, .h2dDone 0 true, .publish]).map
      (fun s => (published s, networkCompleted (cfgChunk 2 1 1 false) s,
                 s.layers.map (·.dispatched))) =
    some (true, true, [1, 0]) := by
  decide

/-- Dishonest post-retirement push (either contract): after `all_layers_called`
retires the push session, an extra stream seats a fresh table and dispatches layer 0 twice. -/
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

/-- Post-retirement duplicate whose redundant H2D copy fails flips an otherwise
complete transfer to `failed_recving`. -/
theorem trace_dup_copy_fails_late :
    ((sys (cfgChunk 2 1 1 false)).run
      [.stream 0 0, .stream 1 0, .stream 0 0, .h2dDone 0 true, .h2dDone 1 true,
       .h2dDone 0 false]).map view =
    some ⟨true, true, false, [2, 1], [0, 0]⟩ := by
  decide

/-- Sender-declared before retirement: `CompleteIncomingPush` rejects `streams.landed > streams.declared`
and fails the receive loudly. -/
theorem trace_dup_declared_rejected :
    ((sys (cfgDeclared 2 1 1 false)).run [.stream 0 0, .stream 0 0]).map view =
      some ⟨true, false, false, [1, 0], [1, 0]⟩ ∧
    (sys (cfgDeclared 2 1 1 false)).run [.stream 0 0, .stream 0 0, .stream 1 0] = none := by
  decide

#guard foundCounterexample (cfgChunk 2 1 1 false) 6
#guard foundCounterexample (cfgDeclared 2 1 1 false) 6

/-! ### Candidate hardening (`stepHardened`, non-shipping)

Adds chunk-count over-count rejection (`completed > numStreams`) and a
session-side per-layer once-latch (`L.dispatched = 0`) in `ExecuteLayerH2d`. -/

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

/-! ## Replay of C++ transport tests -/

/-- Model-only (`cfgChunk 1 1 2 true`): two streams complete and retire one layer. -/
def tr_session_two_streams_complete : List Ev := [.stream 0 0, .stream 0 0]

theorem trace_session_two_streams_complete :
    ((sys (cfgChunk 1 1 2 true)).run (tr_session_two_streams_complete.take 1)).map
      (fun s => (s.layers.map (·.dispatched), s.layers.map (·.entry.isSome))) =
      some ([0], [true]) ∧
    ((sys (cfgChunk 1 1 2 true)).run tr_session_two_streams_complete).map
      (fun s => (s.layers.map (·.dispatched), s.layers.map (·.entry.isSome))) =
      some ([1], [false]) := by
  decide

/-- `BlockTransportTest.LayerCompletesOnlyAfterEveryDeclaredSender`: two declared
senders complete both layers, retire the push session, and re-dispatch layer 0 on reuse. -/
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

/-- `BlockTransportTest.SenderOverDeliveryIsRejected`: while layer 1 is still
open, a second stream from sender 0 for layer 0 is rejected (`statusOk = false`). -/
def tr_sender_over_delivery_rejected : List Ev := [.stream 0 0, .stream 0 1, .stream 0 0]

theorem trace_sender_over_delivery_rejected :
    let c := cfgDeclared 2 2 1 false
    ((sys c).run (tr_sender_over_delivery_rejected.take 2)).map
      (fun s => (s.layers.map (·.dispatched), s.life.statusOk)) = some ([1, 0], true) ∧
    ((sys c).run tr_sender_over_delivery_rejected).map
      (fun s => (s.layers.map (·.dispatched), s.life.statusOk)) = some ([1, 0], false) := by
  decide

/-- `BlockTransportTest.UndeclaredUuidKeepsHeaderDeclaredCompletion`: an
undeclared uuid runs chunk-count (`expected_chunks = 1`), so two senders dispatch layer 0 twice. -/
def tr_undeclared_uuid_chunk_count : List Ev := [.stream 0 0, .stream 0 1]

theorem trace_undeclared_uuid_chunk_count :
    ((sys (cfgChunk 1 2 1 true)).run (tr_undeclared_uuid_chunk_count.take 1)).map
      (fun s => s.layers.map (·.dispatched)) = some [1] ∧
    ((sys (cfgChunk 1 2 1 true)).run tr_undeclared_uuid_chunk_count).map
      (fun s => s.layers.map (·.dispatched)) = some [2] := by
  decide


end TpuSyncVerify.Transfer.PrefillDecode.Transport

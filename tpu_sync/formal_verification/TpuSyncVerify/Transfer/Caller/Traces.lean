import TpuSyncVerify.Transfer.Caller.Invariant

/-! # `Caller`: single-worker bounded search and trace theorems

Part of `TpuSyncVerify.Transfer.Caller`. -/

namespace TpuSyncVerify.Transfer.Caller

/-! ## Replay and search -/

def allEvents : List Ev :=
  [.pOutcome true, .pOutcome false, .dOutcome true, .dOutcome false, .promote, .finish]

/-- The events of a run in which every transfer succeeds. -/
def okEvents : List Ev := [.pOutcome true, .dOutcome true, .promote, .finish]

def search (c : Cfg) (events : List Ev := allEvents) (fuel : Nat := 16) : ModelCheck.Result Ev :=
  ModelCheck.check (sys c) events violates fuel

def leaked (c : Cfg) (events : List Ev := allEvents) : Bool :=
  match search c events with
  | .counterexample _ => true
  | _ => false

/-- Projection used by the traces. -/
structure View where
  pHeld : Bool
  dStatus : DStatus
  dHeld : Bool
  dFinRecv : Bool
  dFailRecv : Bool
  deriving Repr, DecidableEq

def view (s : S) : View := ⟨s.pHeld, s.dStatus, s.dHeld, s.dFinRecv, s.dFailRecv⟩

def current (p : Policy) : Cfg := ⟨connectorCurrent, p⟩
def fixed (p : Policy) : Cfg := ⟨connectorFixed, p⟩
def naive (p : Policy) : Cfg := ⟨connectorNaive, p⟩
def loadErrorOnly (p : Policy) : Cfg := ⟨connectorLoadErrorOnly, p⟩

/-! ### The current connector

When every transfer succeeds, the current connector is fine: `done_sending`
frees P and `done_recving` promotes D (what the connector was tested
against). Exhaustive over `okEvents`. -/

#guard search (current .fail) okEvents = .safe
#guard search (current .recompute) okEvents = .safe

/-- Success path, both nodes: P freed on `done_sending`, D promoted to decode
on `done_recving` and freed when it finishes. -/
theorem trace_current_ok :
    ((sys (current .fail)).run [.pOutcome true, .dOutcome true, .promote, .finish]).map view =
    some ⟨false, .finished, false, false, false⟩ := by
  decide

/-- **Current connector, P side.** The send is published as failed — a push that failed, or
one nobody pulled within `timeout_s` — and dropped by `get_finished()`; P keeps
the blocks. `leakP_absorbing`: forever. -/
theorem trace_current_connector_p :
    ((sys (current .fail)).run [.pOutcome false]).map (fun s => (view s, leakP s)) =
    some (⟨true, .waitingRemoteKv, true, false, false⟩, true) := by
  decide

/-- **Current connector, D side, abort after.** The pull fails, the report is dropped, the
request stays parked; "the request times out at the API layer instead"
(`conn.py:644`) — and the abort defers the free to a `finished_recving` that
will never come (`sched.py:2064-2066`). -/
theorem trace_current_connector_d_abort_after :
    ((sys (current .fail)).run [.dOutcome false, .finish]).map (fun s => (view s, leakD s)) =
    some (⟨true, .finished, true, false, false⟩, true) := by
  decide

/-- **Current connector, D side, abort before.** The client gives up while the pull is still
running; the free is deferred; the pull then fails and the report is dropped. -/
theorem trace_current_connector_d_abort_before :
    ((sys (current .fail)).run [.finish, .dOutcome false]).map (fun s => (view s, leakD s)) =
    some (⟨true, .finished, true, false, false⟩, true) := by
  decide

/-- The same two D-side traces under the recompute policy: the policy never
gets to act, because nothing reaches the scheduler. -/
theorem trace_current_connector_d_recompute :
    ((sys (current .recompute)).run [.dOutcome false]).map (fun s => (view s, leakD s)) =
    some (⟨true, .waitingRemoteKv, true, false, false⟩, true) ∧
    ((sys (current .recompute)).run [.finish, .dOutcome false]).map (fun s => (view s, leakD s)) =
    some (⟨true, .finished, true, false, false⟩, true) := by
  decide

#guard leaked (current .fail) = true
#guard leaked (current .recompute) = true

/-! ### The fix -/

#guard search (fixed .fail) = .safe
#guard search (fixed .recompute) = .safe

/-- Fixed connector, failed pull, recompute policy: the request is marked
failed and complete in one step, the next `schedule()` frees the blocks and
sends it to prefill locally. -/
theorem trace_fixed_recompute :
    ((sys (fixed .recompute)).run [.dOutcome false]).map view =
      some ⟨true, .waitingRemoteKv, true, true, true⟩ ∧
    ((sys (fixed .recompute)).run [.dOutcome false, .promote]).map view =
      some ⟨true, .recomputing, false, false, false⟩ := by
  decide

/-- Fixed connector, failed pull, fail policy (the default): the load error
finishes the request with `FINISHED_ERROR` and defers the free; the
`finished_recving` in the same step releases it. -/
theorem trace_fixed_fail :
    ((sys (fixed .fail)).run [.dOutcome false]).map view =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-- Fixed connector, P side: a failed or never-pulled send frees P. -/
theorem trace_fixed_p :
    ((sys (fixed .fail)).run [.pOutcome false]).map view =
      some ⟨false, .waitingRemoteKv, true, false, false⟩ := by
  decide

/-- Fixed connector, abort before the failed pull is reported: the deferred
free is released by the completion report. -/
theorem trace_fixed_abort_before :
    ((sys (fixed .fail)).run [.finish, .dOutcome false]).map view =
      some ⟨true, .finished, false, false, false⟩ ∧
    ((sys (fixed .recompute)).run [.finish, .dOutcome false]).map view =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-- Stage-3 Reshard connector under `Policy.fail` with
`_stage3_fail_uncertain_loads_early`: an `"uncertain"` post-arm RPC error stages
`invalid_block_ids` on the first pass (`applyLoadError .fail init`), finishing
the request while holding `dHeld = true`, and the later native receiver failure
(`.dOutcome false`) releases `dHeld = false`. -/
theorem trace_fixed_early_uncertain_load_error :
    let s₁ := applyLoadError .fail init
    view s₁ = ⟨true, .finished, true, false, false⟩ ∧
    (step (fixed .fail) s₁ (.dOutcome false)).map view =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-! ### The two half-fixes -/

/-- Reporting a failed receive as `done_recving` alone: the scheduler promotes
the request with its full `num_computed_tokens` and decodes on KV that was
never written — the EngineCore-killing assertion the connector's comment
describes. Same under either policy: the policy only acts on
`invalid_block_ids`. -/
theorem trace_naive_decodes :
    ((sys (naive .fail)).run [.dOutcome false, .promote]).map (fun s => (view s, decodesUnwritten s)) =
      some (⟨true, .decoding, true, false, false⟩, true) ∧
    ((sys (naive .recompute)).run [.dOutcome false, .promote]).map (fun s => (view s, decodesUnwritten s)) =
      some (⟨true, .decoding, true, false, false⟩, true) := by
  decide

#guard leaked (naive .fail) = true
#guard leaked (naive .recompute) = true

/-- The load-failure channel without the completion, fail policy: the request
is errored out but its free waits for a `finished_recving` that never comes. -/
theorem trace_loadErrorOnly_fail :
    ((sys (loadErrorOnly .fail)).run [.dOutcome false]).map (fun s => (view s, leakD s)) =
      some (⟨true, .finished, true, false, false⟩, true) := by
  decide

/-- The same under the recompute policy: marked for recomputation but never
promotable. -/
theorem trace_loadErrorOnly_recompute :
    ((sys (loadErrorOnly .recompute)).run [.dOutcome false]).map (fun s => (view s, leakD s)) =
      some (⟨true, .waitingRemoteKv, true, false, true⟩, true) := by
  decide

#guard leaked (loadErrorOnly .fail) = true
#guard leaked (loadErrorOnly .recompute) = true

end TpuSyncVerify.Transfer.Caller

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

def twoChannel (p : Policy) : Cfg := ⟨connectorTwoChannel, p⟩
def dropFailed (p : Policy) : Cfg := ⟨connectorDropFailed, p⟩
def naive (p : Policy) : Cfg := ⟨connectorNaive, p⟩
def loadErrorOnly (p : Policy) : Cfg := ⟨connectorLoadErrorOnly, p⟩

/-! ### The shipping two-channel connector (`connectorTwoChannel`) -/

#guard search (twoChannel .fail) = .safe
#guard search (twoChannel .recompute) = .safe

/-- Success path, both nodes: P freed on `done_sending`, D promoted to decode
on `done_recving` and freed when it finishes. -/
theorem trace_twoChannel_ok :
    ((sys (twoChannel .fail)).run [.pOutcome true, .dOutcome true, .promote, .finish]).map view =
    some ⟨false, .finished, false, false, false⟩ := by
  decide

/-- Two-channel connector, failed pull, recompute policy: the request is marked
failed and complete in one step, the next `schedule()` frees the blocks and
sends it to prefill locally. -/
theorem trace_twoChannel_recompute :
    ((sys (twoChannel .recompute)).run [.dOutcome false]).map view =
      some ⟨true, .waitingRemoteKv, true, true, true⟩ ∧
    ((sys (twoChannel .recompute)).run [.dOutcome false, .promote]).map view =
      some ⟨true, .recomputing, false, false, false⟩ := by
  decide

/-- Two-channel connector, failed pull, fail policy (the default): the load
error finishes the request with `FINISHED_ERROR` and defers the free; the
`finished_recving` in the same step releases it. -/
theorem trace_twoChannel_fail :
    ((sys (twoChannel .fail)).run [.dOutcome false]).map view =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-- Two-channel connector, P side: a failed or never-pulled send frees P. -/
theorem trace_twoChannel_p :
    ((sys (twoChannel .fail)).run [.pOutcome false]).map view =
      some ⟨false, .waitingRemoteKv, true, false, false⟩ := by
  decide

/-- Two-channel connector, abort before the failed pull is reported: the
deferred free is released by the completion report. -/
theorem trace_twoChannel_abort_before :
    ((sys (twoChannel .fail)).run [.finish, .dOutcome false]).map view =
      some ⟨true, .finished, false, false, false⟩ ∧
    ((sys (twoChannel .recompute)).run [.finish, .dOutcome false]).map view =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-- Stage-3 Reshard connector under `Policy.fail` with
`_stage3_fail_uncertain_loads_early`: an `"uncertain"` post-arm RPC error stages
`invalid_block_ids` on the first pass (`applyLoadError .fail init`), finishing
the request while holding `dHeld = true`, and the later native receiver failure
(`.dOutcome false`) releases `dHeld = false`. -/
theorem trace_twoChannel_early_uncertain_load_error :
    let s₁ := applyLoadError .fail init
    view s₁ = ⟨true, .finished, true, false, false⟩ ∧
    (step (twoChannel .fail) s₁ (.dOutcome false)).map view =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-! ### Counterfactual connector mutants

Each mutant omits one part of `Contract`:

1. `connectorDropFailed` drops `failed_recving` on both P and D (a
   `get_finished`-only connector that overrides neither
   `get_block_ids_with_load_errors` nor `get_transfer_results`). When every
   transfer succeeds (`okEvents`), it is fine; on any failure or unpulled send,
   blocks leak permanently (`leakP_absorbing`, `leakD_absorbing`).
2. `connectorNaive` reports failed receives in `finished_recving` without the
   load-failure channel (also reached if a `get_finished()` wrapper returning
   `(finished_sending, finished_recving)` without
   `get_block_ids_with_load_errors` is called by a runner that does not use
   `get_transfer_results`). D decodes unwritten KV.
3. `connectorLoadErrorOnly` reports on the load-failure channel without
   `finished_recving`. D leaks its blocks under either policy. -/

#guard search (dropFailed .fail) okEvents = .safe
#guard search (dropFailed .recompute) okEvents = .safe

/-- **Mutant `connectorDropFailed`, P side.** The send is published as failed — a
push that failed, or one nobody pulled within `timeout_s` — and dropped by
`get_finished()`; P keeps the blocks forever (`leakP_absorbing`). -/
theorem trace_dropFailed_p :
    ((sys (dropFailed .fail)).run [.pOutcome false]).map (fun s => (view s, leakP s)) =
    some (⟨true, .waitingRemoteKv, true, false, false⟩, true) := by
  decide

/-- **Mutant `connectorDropFailed`, D side, abort after.** The pull fails, the
report is dropped, the request stays parked; when the API layer later aborts it,
`Scheduler.finish_requests` defers the block free to a `finished_recving` that
will never come. -/
theorem trace_dropFailed_d_abort_after :
    ((sys (dropFailed .fail)).run [.dOutcome false, .finish]).map (fun s => (view s, leakD s)) =
    some (⟨true, .finished, true, false, false⟩, true) := by
  decide

/-- **Mutant `connectorDropFailed`, D side, abort before.** The client gives up
while the pull is still running; the free is deferred; the pull then fails and
the report is dropped. -/
theorem trace_dropFailed_d_abort_before :
    ((sys (dropFailed .fail)).run [.finish, .dOutcome false]).map (fun s => (view s, leakD s)) =
    some (⟨true, .finished, true, false, false⟩, true) := by
  decide

/-- **Mutant `connectorDropFailed`, recompute policy:** the policy never gets to
act because nothing reaches the scheduler. -/
theorem trace_dropFailed_d_recompute :
    ((sys (dropFailed .recompute)).run [.dOutcome false]).map (fun s => (view s, leakD s)) =
    some (⟨true, .waitingRemoteKv, true, false, false⟩, true) ∧
    ((sys (dropFailed .recompute)).run [.finish, .dOutcome false]).map (fun s => (view s, leakD s)) =
    some (⟨true, .finished, true, false, false⟩, true) := by
  decide

#guard leaked (dropFailed .fail) = true
#guard leaked (dropFailed .recompute) = true

/-- **Mutant `connectorNaive`:** reporting a failed receive as `done_recving` /
`finished_recving` alone promotes the request with its full
`num_computed_tokens` and decodes on KV that was never written. Same under
either policy, because the policy only acts on the load-failure channel. -/
theorem trace_naive_decodes :
    ((sys (naive .fail)).run [.dOutcome false, .promote]).map (fun s => (view s, decodesUnwritten s)) =
      some (⟨true, .decoding, true, false, false⟩, true) ∧
    ((sys (naive .recompute)).run [.dOutcome false, .promote]).map (fun s => (view s, decodesUnwritten s)) =
      some (⟨true, .decoding, true, false, false⟩, true) := by
  decide

#guard leaked (naive .fail) = true
#guard leaked (naive .recompute) = true

/-- **Mutant `connectorLoadErrorOnly`, fail policy:** the request is errored out
but its free waits for a `finished_recving` that never comes. -/
theorem trace_loadErrorOnly_fail :
    ((sys (loadErrorOnly .fail)).run [.dOutcome false]).map (fun s => (view s, leakD s)) =
      some (⟨true, .finished, true, false, false⟩, true) := by
  decide

/-- **Mutant `connectorLoadErrorOnly`, recompute policy:** marked for
recomputation (`dFailRecv = true`) but never promotable (`dFinRecv = false`). -/
theorem trace_loadErrorOnly_recompute :
    ((sys (loadErrorOnly .recompute)).run [.dOutcome false]).map (fun s => (view s, leakD s)) =
      some (⟨true, .waitingRemoteKv, true, false, true⟩, true) := by
  decide

#guard leaked (loadErrorOnly .fail) = true
#guard leaked (loadErrorOnly .recompute) = true

end TpuSyncVerify.Transfer.Caller

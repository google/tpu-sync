import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Caller.Traces

/-!
# Multi-worker `KVOutputAggregator` model (`TpuSyncVerify.Transfer.Caller.MultiWorker`)

Discharges assumption **B1** of the single-worker caller model by placing vLLM's
`KVOutputAggregator` (`vllm/distributed/kv_transfer/kv_connector/utils.py`,
configured by `get_finished_count()` in `conn.py` and `tpu_connector.py`)
between two producer workers (`w ∈ {false, true}`)
and two consumer workers (`w ∈ {false, true}`) and the scheduler, across **both**
load-failure channels of `KVOutputAggregator.aggregate`:

* **Barrier-counted completions (`finished_sending`, `finished_recving`).**
  Each worker `w` whose connector emits `finishedSending = true` (resp.
  `finishedRecving = true`) increments the request's aggregator counter
  `pSentCount` (resp. `dRecvCount`). The aggregator forwards
  `finishedSending = true` (resp. `finishedRecving = true`) to the scheduler
  only on the step where the counter reaches `expectedP` (resp. `expectedD`).
* **Two load-failure channels (`AggChannel`).**
  1. `AggChannel.eagerBlock` (`KVConnectorOutput.invalid_block_ids`, driven by
     `vllm-torchtpu` Stage-3 `tpu_connector.py` via
     `get_block_ids_with_load_errors`): unioned across workers and passed to
     `applyLoadError policy` immediately on whichever step any worker reports
     `loadError = true`, without waiting for `dRecvCount = expectedD`.
  2. `AggChannel.latchedReq` (`KVConnectorOutput.failed_recving`, driven by
     `tpu-inference` `TPUConnectorWorker.get_transfer_results`): latched in
     `_failed_recving_pending |= kv_output.failed_recving` (`dFailPending`) and
     emitted as `failed_recving = _failed_recving_pending & finished_recving`
     (clearing `_failed_recving_pending -= failed_recving`) only on the step
     that reaches `dRecvCount = expectedD`.
  Because `applyLoadError policy` is idempotent (`applyLoadError_idem`), two
  workers reporting in separate steps produce the exact same post-completion
  scheduler state on either channel as a single batched
  `KVOutputAggregator.aggregate` pass (`applySend_staggered_eq_batch`,
  `applyRecv_staggered_eq_batch`, `applyRecv_eager_eq_latched`).

Main results:

1. `mwLeakP_absorbing` and `mwLeakD_absorbing`: once all participating producer
   (resp. consumer) workers have published (`allPSent`, `allDRecv`), a block
   leak on P or D is permanent across all subsequent steps on either channel.
2. `mw_contract_safe` / `mw_twoChannel_safe`: when `Contract c.conn` holds and
   the aggregator thresholds match the participating worker counts
   (`c.expectedP = 2`, `c.expectedD = 2`), **on both `AggChannel.eagerBlock` and
   `AggChannel.latchedReq`**, no reachable state ever frees P or D HBM while any
   worker on that node is still in flight (`mwEarlyFree = false`, discharging
   the pipeline models' **A4** assumption across workers), leaks blocks on P or
   D (`mwLeakP = false`, `mwLeakD = false`), or decodes unwritten KV
   (`mwDecodesUnwritten = false`), across all staggered or batched worker
   completion and failure orders under either failure policy.
   `mw_earlyLoadError_fail_eq_finish` and `mw_earlyLoadError_fail_holds_blocks`
   extend the single-worker uncertain-load theorems (`AggChannel.eagerBlock`) to
   a 2-worker consumer.
3. Concrete traces for staggered asymmetric outcomes on both channels
   (`trace_mw_staggered_fail_then_ok_fail`,
   `trace_mw_staggered_fail_then_ok_recompute`,
   `trace_mw_staggered_ok_then_fail_recompute`,
   `trace_mw_latched_staggered_fail_then_ok_fail`,
   `trace_mw_latched_staggered_fail_then_ok_recompute`) and for the multi-worker
   failure modes (`trace_mw_dropFailed_partial_failure_leaks`,
   `trace_mw_latched_loadErrorOnly_never_emits`,
   `trace_mw_overcounted_expected_leaks`).
-/

namespace TpuSyncVerify.Transfer.Caller

/-- Which load-failure channel of `KVOutputAggregator.aggregate` (`utils.py`)
the consumer connector drives:
* `eagerBlock`: `KVConnectorOutput.invalid_block_ids`
  (`get_block_ids_with_load_errors`, `vllm-torchtpu` Stage-3 `tpu_connector.py`),
  unioned across workers and forwarded on every step.
* `latchedReq`: `KVConnectorOutput.failed_recving`
  (`KVConnectorTransferResults.failed_recving`, `tpu-inference`
  `TPUConnectorWorker.get_transfer_results`), latched in
  `_failed_recving_pending` until `finished_recving` fires. -/
inductive AggChannel where
  | eagerBlock
  | latchedReq
  deriving Repr, DecidableEq

/-- Multi-worker configuration: connector, scheduler failure policy, aggregator
load-failure channel, and the `get_finished_count()` thresholds (`expectedP`,
`expectedD`) used by `KVOutputAggregator` (`TPUConnectorScheduler.get_finished_count`
in `conn.py`, `TPURaidenConnector.get_finished_count` in `tpu_connector.py`). -/
structure MWCfg where
  conn : Connector
  policy : Policy
  channel : AggChannel := .eagerBlock
  expectedP : Nat := 2
  expectedD : Nat := 2

/-- State of one request across a 2-worker producer node, a 2-worker consumer
node, `KVOutputAggregator`, and vLLM's scheduler. -/
structure MWS where
  /-- Producer worker 0's `send.published`. -/
  sendPub0 : Option Bool := none
  /-- Producer worker 1's `send.published`. -/
  sendPub1 : Option Bool := none
  /-- Consumer worker 0's `recv.published`. -/
  recvPub0 : Option Bool := none
  /-- Consumer worker 1's `recv.published`. -/
  recvPub1 : Option Bool := none
  /-- `KVOutputAggregator` count of producer workers that reported `finished_sending`. -/
  pSentCount : Nat := 0
  /-- `KVOutputAggregator` count of consumer workers that reported `finished_recving`. -/
  dRecvCount : Nat := 0
  /-- `request_id ∈ _failed_recving_pending` in `KVOutputAggregator` (`utils.py`;
  stays `false` under `AggChannel.eagerBlock`). -/
  dFailPending : Bool := false
  /-- P still holds the request's KV blocks. -/
  pHeld : Bool := true
  dStatus : DStatus := .waitingRemoteKv
  /-- D still holds the blocks allocated for the remote KV. -/
  dHeld : Bool := true
  /-- `request_id ∈ finished_recving_kv_req_ids` (`Scheduler`). -/
  dFinRecv : Bool := false
  /-- `request_id ∈ failed_recving_kv_req_ids` (`Scheduler`). -/
  dFailRecv : Bool := false
  deriving Repr, DecidableEq

def mwInit : MWS := {}

inductive MWEv where
  /-- Producer worker `w` (`false` = worker 0, `true` = worker 1) publishes its
  send outcome `ok`, which passes through `conn.onSend ok` and
  `KVOutputAggregator` before reaching P's scheduler. -/
  | pWorker (w : Bool) (ok : Bool)
  /-- Consumer worker `w` (`false` = worker 0, `true` = worker 1) publishes its
  receive outcome `ok`, which passes through `conn.onRecv ok` and
  `KVOutputAggregator` before reaching D's scheduler. -/
  | dWorker (w : Bool) (ok : Bool)
  /-- D's next `Scheduler.schedule` promotes the waiting request. -/
  | promote
  /-- D's request leaves the engine (`Scheduler.finish_requests`). -/
  | finish
  deriving Repr, DecidableEq

/-- `KVOutputAggregator` on the producer side (`utils.py`): increment the
request's `finished_sending` counter when `raw.finishedSending` holds, and emit
`finishedSending = true` to the scheduler iff the counter reaches `expectedP`. -/
def mwAggSend (expectedP : Nat) (raw : Signal) (cnt : Nat) : Nat × Signal :=
  let cnt' := if raw.finishedSending then cnt + 1 else cnt
  (cnt', { finishedSending := raw.finishedSending && decide (cnt' = expectedP) })

/-- `KVOutputAggregator` on the consumer side (`utils.py`): increment the
request's `finished_recving` counter when `raw.finishedRecving` holds, emit
`finishedRecving = true` iff `cnt' = expectedD`, and either forward
`raw.loadError` immediately (`AggChannel.eagerBlock`, `invalid_block_ids`) or
latch it in `_failed_recving_pending` until `finishedRecving` fires
(`AggChannel.latchedReq`, `failed_recving = _failed_recving_pending &
finished_recving`). -/
def mwAggRecv (ch : AggChannel) (expectedD : Nat) (raw : Signal)
    (cnt : Nat) (pend : Bool) : Nat × Bool × Signal :=
  let cnt' := if raw.finishedRecving then cnt + 1 else cnt
  let fin := raw.finishedRecving && decide (cnt' = expectedD)
  match ch with
  | .eagerBlock =>
    (cnt', pend, { finishedRecving := fin, loadError := raw.loadError })
  | .latchedReq =>
    let pend₁ := pend || raw.loadError
    (cnt', pend₁ && !fin, { finishedRecving := fin, loadError := pend₁ && fin })

/-- Project the scheduler-owned fields of `MWS` to `S`. -/
@[simp] def toS (s : MWS) : S :=
  { pHeld := s.pHeld, dStatus := s.dStatus, dHeld := s.dHeld,
    dFinRecv := s.dFinRecv, dFailRecv := s.dFailRecv }

/-- Write updated scheduler-owned fields from `S` back into `MWS`. -/
@[simp] def withS (s : MWS) (u : S) : MWS :=
  { s with pHeld := u.pHeld, dStatus := u.dStatus, dHeld := u.dHeld,
           dFinRecv := u.dFinRecv, dFailRecv := u.dFailRecv }

def mwPWorker (c : MWCfg) (w : Bool) (ok : Bool) (s : MWS) : Option MWS :=
  match w with
  | false =>
    if s.sendPub0 = none then
      let (cnt', sig) := mwAggSend c.expectedP (c.conn.onSend ok) s.pSentCount
      let s₁ := { s with sendPub0 := some ok, pSentCount := cnt' }
      some (withS s₁ (applySend sig (toS s₁)))
    else none
  | true =>
    if s.sendPub1 = none then
      let (cnt', sig) := mwAggSend c.expectedP (c.conn.onSend ok) s.pSentCount
      let s₁ := { s with sendPub1 := some ok, pSentCount := cnt' }
      some (withS s₁ (applySend sig (toS s₁)))
    else none

def mwDWorker (c : MWCfg) (w : Bool) (ok : Bool) (s : MWS) : Option MWS :=
  match w with
  | false =>
    if s.recvPub0 = none then
      let (cnt', pend', sig) :=
        mwAggRecv c.channel c.expectedD (c.conn.onRecv ok) s.dRecvCount s.dFailPending
      let s₁ := { s with recvPub0 := some ok, dRecvCount := cnt', dFailPending := pend' }
      some (withS s₁ (applyRecv c.policy sig (toS s₁)))
    else none
  | true =>
    if s.recvPub1 = none then
      let (cnt', pend', sig) :=
        mwAggRecv c.channel c.expectedD (c.conn.onRecv ok) s.dRecvCount s.dFailPending
      let s₁ := { s with recvPub1 := some ok, dRecvCount := cnt', dFailPending := pend' }
      some (withS s₁ (applyRecv c.policy sig (toS s₁)))
    else none

def mwPromote (s : MWS) : Option MWS :=
  (promote (toS s)).map (withS s)

def mwFinish (s : MWS) : Option MWS :=
  (finish (toS s)).map (withS s)

def mwStep (c : MWCfg) (s : MWS) : MWEv → Option MWS
  | .pWorker w ok => mwPWorker c w ok s
  | .dWorker w ok => mwDWorker c w ok s
  | .promote => mwPromote s
  | .finish => mwFinish s

def mwSys (c : MWCfg) : System MWS MWEv := ⟨mwInit, mwStep c⟩

/-! ## Multi-worker safety properties -/

/-- Both producer workers have published their send outcome. -/
def allPSent (s : MWS) : Bool := s.sendPub0.isSome && s.sendPub1.isSome

/-- Both consumer workers have published their receive outcome. -/
def allDRecv (s : MWS) : Bool := s.recvPub0.isSome && s.recvPub1.isSome

/-- Both consumer workers have published `done_recving` (`some true`). -/
def allDOk (s : MWS) : Bool :=
  (s.recvPub0 == some true) && (s.recvPub1 == some true)

/-- Every producer worker has published its send outcome and P still holds the
request's KV blocks. -/
def mwLeakP (s : MWS) : Bool := allPSent s && s.pHeld

/-- Every consumer worker has published its receive outcome and D either keeps
the request parked with its blocks and nothing to promote it, or has finished it
with the block free deferred. -/
def mwLeakD (s : MWS) : Bool :=
  allDRecv s &&
    ((s.dStatus == .waitingRemoteKv && !s.dFinRecv && s.dHeld) ||
     (s.dStatus == .finished && s.dHeld))

/-- D proceeds to decode when at least one consumer worker did not succeed. -/
def mwDecodesUnwritten (s : MWS) : Bool :=
  s.dStatus == .decoding && !allDOk s

/-- **Multi-worker A4 (`mwEarlyFree`):** P frees its HBM (`pHeld = false`)
while a P worker is still unpublished, or D frees its HBM (`dHeld = false`)
while a D worker is still unpublished. -/
def mwEarlyFree (s : MWS) : Bool :=
  (!s.pHeld && !allPSent s) || (!s.dHeld && !allDRecv s)

def mwViolates (s : MWS) : Bool :=
  mwLeakP s || mwLeakD s || mwDecodesUnwritten s || mwEarlyFree s

def MWSafe (s : MWS) : Prop := mwViolates s = false

/-! ## Multi-worker leaks are absorbing -/

theorem mwLeakP_absorbing {c : MWCfg} {s s' : MWS} {e : MWEv}
    (h : mwLeakP s = true) (hs : mwStep c s e = some s') : mwLeakP s' = true := by
  have hp : (s.sendPub0.isSome = true ∧ s.sendPub1.isSome = true) ∧ s.pHeld = true := by
    simpa [mwLeakP, allPSent] using h
  cases e with
  | pWorker w ok =>
    cases w <;> simp only [mwStep, mwPWorker] at hs
    · split at hs
      · rename_i hn; rw [hn] at hp; simp at hp
      · simp at hs
    · split at hs
      · rename_i hn; rw [hn] at hp; simp at hp
      · simp at hs
  | dWorker w ok =>
    cases w <;> simp only [mwStep, mwDWorker] at hs
    · split at hs
      · simp only [Option.some.injEq] at hs; subst hs
        simp [mwLeakP, allPSent, applyRecv_p, hp]
      · simp at hs
    · split at hs
      · simp only [Option.some.injEq] at hs; subst hs
        simp [mwLeakP, allPSent, applyRecv_p, hp]
      · simp at hs
  | promote =>
    simp only [mwStep, mwPromote, promote, Option.map_eq_some_iff] at hs
    rcases hs with ⟨u, hu, rfl⟩
    split at hu
    · split at hu <;> simp only [Option.some.injEq] at hu <;>
        subst hu <;> simpa [mwLeakP, allPSent] using hp
    · simp at hu
  | finish =>
    simp only [mwStep, mwFinish, finish, Option.map_eq_some_iff] at hs
    rcases hs with ⟨u, hu, rfl⟩
    split at hu
    · simp only [Option.some.injEq] at hu; subst hu; simpa [mwLeakP, allPSent] using hp
    · simp only [Option.some.injEq] at hu; subst hu; simpa [mwLeakP, allPSent] using hp
    · simp only [Option.some.injEq] at hu; subst hu; simpa [mwLeakP, allPSent] using hp
    · simp at hu

theorem mwLeakD_absorbing {c : MWCfg} {s s' : MWS} {e : MWEv}
    (h : mwLeakD s = true) (hs : mwStep c s e = some s') : mwLeakD s' = true := by
  cases e with
  | pWorker w ok =>
    cases w <;> simp only [mwStep, mwPWorker] at hs
    · split at hs
      · simp only [Option.some.injEq] at hs; subst hs
        simp only [applySend]
        split <;> simpa [mwLeakD, allDRecv] using h
      · simp at hs
    · split at hs
      · simp only [Option.some.injEq] at hs; subst hs
        simp only [applySend]
        split <;> simpa [mwLeakD, allDRecv] using h
      · simp at hs
  | dWorker w ok =>
    cases w <;> simp only [mwStep, mwDWorker] at hs
    · split at hs
      · rename_i hn; simp [mwLeakD, allDRecv, hn] at h
      · simp at hs
    · split at hs
      · rename_i hn; simp [mwLeakD, allDRecv, hn] at h
      · simp at hs
  | promote =>
    simp only [mwStep, mwPromote, promote, Option.map_eq_some_iff] at hs
    rcases hs with ⟨u, hu, rfl⟩
    split at hu
    · rename_i hw
      simp only [toS] at hw
      simp [mwLeakD, allDRecv, hw.1, hw.2] at h
    · simp at hu
  | finish =>
    simp only [mwStep, mwFinish, finish, Option.map_eq_some_iff] at hs
    rcases hs with ⟨u, hu, rfl⟩
    split at hu
    · rename_i hd; simp only [toS] at hd
      simp only [Option.some.injEq] at hu; subst hu; simp_all [mwLeakD, allDRecv]
    · rename_i hd; simp only [toS] at hd
      simp only [Option.some.injEq] at hu; subst hu; simp_all [mwLeakD, allDRecv]
    · rename_i hd; simp only [toS] at hd
      simp only [Option.some.injEq] at hu; subst hu; simp_all [mwLeakD, allDRecv]
    · simp at hu

/-! ## Inductive invariant when `expectedP = 2`, `expectedD = 2`, and `Contract c.conn` -/

@[simp] def optCount (o : Option Bool) : Nat :=
  match o with
  | none => 0
  | some _ => 1

/-- Inductive invariant for the 2-worker `KVOutputAggregator` system when both
workers participate (`expectedP = 2`, `expectedD = 2`) and the connector
satisfies `Contract`, parameterized by the aggregator load-failure channel `ch`. -/
structure MWInv (ch : AggChannel) (s : MWS) : Prop where
  /-- `pSentCount` tracks the number of producer workers that have published. -/
  p_cnt : s.pSentCount = optCount s.sendPub0 + optCount s.sendPub1
  /-- P holds the blocks iff at least one producer worker has not published. -/
  p_held : s.pHeld = true ↔ (s.sendPub0 = none ∨ s.sendPub1 = none)
  /-- `dRecvCount` tracks the number of consumer workers that have published. -/
  d_cnt : s.dRecvCount = optCount s.recvPub0 + optCount s.recvPub1
  /-- While at least one consumer worker has not published, D has not received
  an aggregated `finished_recving` (`dFinRecv = false`), still holds its HBM
  (`dHeld = true`), and is still waiting or finished. -/
  d_partial : (s.recvPub0 = none ∨ s.recvPub1 = none) →
    (s.dStatus = .waitingRemoteKv ∨ s.dStatus = .finished) ∧ s.dHeld = true ∧ s.dFinRecv = false
  /-- Once both consumer workers have published, a waiting request has
  `dFinRecv = true` and is promotable. -/
  d_wait : s.recvPub0 ≠ none → s.recvPub1 ≠ none → s.dStatus = .waitingRemoteKv → s.dFinRecv = true
  /-- On `AggChannel.eagerBlock`, any consumer worker publishing `some false`
  marks a waiting request `dFailRecv = true` immediately (preserved when the
  other worker later publishes `some true`). -/
  d_fail_eager : ch = .eagerBlock →
    (s.recvPub0 = some false ∨ s.recvPub1 = some false) →
    s.dStatus = .waitingRemoteKv → s.dFailRecv = true
  /-- On `AggChannel.latchedReq`, while one consumer worker is still
  unpublished, an earlier failed worker is latched in `dFailPending = true`. -/
  d_pend : ch = .latchedReq →
    (s.recvPub0 = none ∨ s.recvPub1 = none) →
    (s.recvPub0 = some false ∨ s.recvPub1 = some false) →
    s.dFailPending = true
  /-- Once both consumer workers have published, on either channel a waiting
  request with any failed worker has `dFailRecv = true`. -/
  d_fail : s.recvPub0 ≠ none → s.recvPub1 ≠ none →
    (s.recvPub0 = some false ∨ s.recvPub1 = some false) →
    s.dStatus = .waitingRemoteKv → s.dFailRecv = true
  /-- Once both consumer workers have published, a finished request holds no blocks. -/
  d_fin : s.recvPub0 ≠ none → s.recvPub1 ≠ none → s.dStatus = .finished → s.dHeld = false
  /-- A recomputing request holds no transfer blocks. -/
  d_rec : s.dStatus = .recomputing → s.dHeld = false
  /-- D decodes only when both consumer workers published `some true`. -/
  d_dec : s.dStatus = .decoding → s.recvPub0 = some true ∧ s.recvPub1 = some true

theorem mw_inv_init (ch : AggChannel) : MWInv ch mwInit := by
  constructor <;> simp [mwInit]

theorem mw_inv_safe {ch : AggChannel} {s : MWS} (h : MWInv ch s) : MWSafe s := by
  simp only [MWSafe, mwViolates, mwLeakP, mwLeakD, mwDecodesUnwritten, mwEarlyFree,
    allPSent, allDRecv, allDOk, Bool.or_eq_false_iff, Bool.and_eq_false_iff]
  refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩
  · cases hp0 : s.sendPub0 with
    | none => simp
    | some b0 =>
      cases hp1 : s.sendPub1 with
      | none => simp
      | some b1 =>
        right
        cases hh : s.pHeld
        · rfl
        · have hnone := h.p_held.mp hh
          simp [hp0, hp1] at hnone
  · cases hr0 : s.recvPub0 with
    | none => simp
    | some b0 =>
      cases hr1 : s.recvPub1 with
      | none => simp
      | some b1 =>
        right
        have hne0 : s.recvPub0 ≠ none := by simp [hr0]
        have hne1 : s.recvPub1 ≠ none := by simp [hr1]
        cases hst : s.dStatus <;> simp
        · left
          exact h.d_wait hne0 hne1 hst
        · exact h.d_fin hne0 hne1 hst
  · cases hst : s.dStatus <;> simp
    obtain ⟨hr0, hr1⟩ := h.d_dec hst
    simp [hr0, hr1]
  · refine ⟨?_, ?_⟩
    · cases hp0 : s.sendPub0 with
      | none =>
        left
        have := h.p_held.mpr (Or.inl hp0)
        simp [this]
      | some b0 =>
        cases hp1 : s.sendPub1 with
        | none =>
          left
          have := h.p_held.mpr (Or.inr hp1)
          simp [this]
        | some b1 => right; simp
    · cases hr0 : s.recvPub0 with
      | none =>
        left
        have := (h.d_partial (Or.inl hr0)).2.1
        simp [this]
      | some b0 =>
        cases hr1 : s.recvPub1 with
        | none =>
          left
          have := (h.d_partial (Or.inr hr1)).2.1
          simp [this]
        | some b1 => right; simp

/-! ## Same-step batch, staggered eager, and latched request-level equivalence -/

/-- `applyLoadError policy` is idempotent: once a request's load failure has
been handled (`dFailRecv := true` under `.recompute`, or `dStatus := .finished`
under `.fail`), a second worker's `loadError = true` on the same or a later pass
is a no-op. -/
theorem applyLoadError_idem (policy : Policy) (s : S) :
    applyLoadError policy (applyLoadError policy s) = applyLoadError policy s := by
  cases policy <;> simp only [applyLoadError] <;> split <;> simp

/-- Delivering a non-completing producer pass (`finishedSending = false`)
followed by the completing pass (`finishedSending = true`) yields the exact
same scheduler state `S` as a single batched `KVOutputAggregator.aggregate`
pass (`finishedSending = true`). -/
theorem applySend_staggered_eq_batch (s : S) :
    applySend { finishedSending := true }
      (applySend { finishedSending := false } s) =
    applySend { finishedSending := true } s := rfl

/-- Delivering worker 0's partial aggregator pass (`finishedRecving = false`,
`loadError = le₀`) followed by worker 1's completing pass
(`finishedRecving = true`, `loadError = le₁`) yields the exact same scheduler
state `S` as a single batched `KVOutputAggregator.aggregate` pass with
`finishedRecving = true` and `loadError = (le₀ || le₁)`. -/
theorem applyRecv_staggered_eq_batch (policy : Policy) (le₀ le₁ : Bool) (s : S) :
    applyRecv policy { finishedRecving := true, loadError := le₁ }
      (applyRecv policy { finishedRecving := false, loadError := le₀ } s) =
    applyRecv policy { finishedRecving := true, loadError := le₀ || le₁ } s := by
  cases le₀ <;> cases le₁ <;> simp [applyRecv, applyLoadError_idem]

/-- Across two back-to-back consumer worker reports (`le₀`, then `le₁`),
`AggChannel.eagerBlock` (`invalid_block_ids` forwarded immediately) and
`AggChannel.latchedReq` (`_failed_recving_pending` emitted with
`finished_recving`) arrive at the exact same scheduler state `S`. -/
theorem applyRecv_eager_eq_latched (policy : Policy) (le₀ le₁ : Bool) (s : S) :
    applyRecv policy { finishedRecving := true, loadError := le₁ }
      (applyRecv policy { finishedRecving := false, loadError := le₀ } s) =
    applyRecv policy { finishedRecving := true, loadError := le₀ || le₁ }
      (applyRecv policy { finishedRecving := false, loadError := false } s) := by
  simpa [applyRecv] using applyRecv_staggered_eq_batch policy le₀ le₁ s

theorem mw_inv_pWorker {c : MWCfg} (hc : Contract c.conn) (hp : c.expectedP = 2)
    {s s' : MWS} {w ok : Bool} (h : MWInv c.channel s)
    (hs : mwPWorker c w ok s = some s') : MWInv c.channel s' := by
  have hsend := hc.send_total ok
  have hpc := h.p_cnt
  have hph := h.p_held
  cases w with
  | false =>
    simp only [mwPWorker] at hs
    split at hs
    · rename_i hn0
      cases hp1 : s.sendPub1 with
      | none =>
        have hph_true : s.pHeld = true := hph.mpr (Or.inl hn0)
        simp only [mwAggSend, hsend, hp, hpc, hn0, hp1, optCount, ite_true,
          applySend, toS, withS, Option.some.injEq] at hs
        subst hs
        exact { p_cnt := by simp, p_held := by simp [hph_true],
                d_cnt := h.d_cnt, d_partial := h.d_partial, d_wait := h.d_wait,
                d_fail_eager := h.d_fail_eager, d_pend := h.d_pend, d_fail := h.d_fail,
                d_fin := h.d_fin, d_rec := h.d_rec, d_dec := h.d_dec }
      | some b1 =>
        simp only [mwAggSend, hsend, hp, hpc, hn0, hp1, optCount, ite_true,
          applySend, toS, withS, Option.some.injEq] at hs
        subst hs
        exact { p_cnt := by simp, p_held := by simp,
                d_cnt := h.d_cnt, d_partial := h.d_partial, d_wait := h.d_wait,
                d_fail_eager := h.d_fail_eager, d_pend := h.d_pend, d_fail := h.d_fail,
                d_fin := h.d_fin, d_rec := h.d_rec, d_dec := h.d_dec }
    · simp at hs
  | true =>
    simp only [mwPWorker] at hs
    split at hs
    · rename_i hn1
      cases hp0 : s.sendPub0 with
      | none =>
        have hph_true : s.pHeld = true := hph.mpr (Or.inr hn1)
        simp only [mwAggSend, hsend, hp, hpc, hn1, hp0, optCount, ite_true,
          applySend, toS, withS, Option.some.injEq] at hs
        subst hs
        exact { p_cnt := by simp, p_held := by simp [hph_true],
                d_cnt := h.d_cnt, d_partial := h.d_partial, d_wait := h.d_wait,
                d_fail_eager := h.d_fail_eager, d_pend := h.d_pend, d_fail := h.d_fail,
                d_fin := h.d_fin, d_rec := h.d_rec, d_dec := h.d_dec }
      | some b0 =>
        simp only [mwAggSend, hsend, hp, hpc, hn1, hp0, optCount, ite_true,
          applySend, toS, withS, Option.some.injEq] at hs
        subst hs
        exact { p_cnt := by simp, p_held := by simp,
                d_cnt := h.d_cnt, d_partial := h.d_partial, d_wait := h.d_wait,
                d_fail_eager := h.d_fail_eager, d_pend := h.d_pend, d_fail := h.d_fail,
                d_fin := h.d_fin, d_rec := h.d_rec, d_dec := h.d_dec }
    · simp at hs

theorem mw_inv_dWorker {c : MWCfg} (hc : Contract c.conn) (hd : c.expectedD = 2)
    {s s' : MWS} {w ok : Bool} (h : MWInv c.channel s)
    (hs : mwDWorker c w ok s = some s') : MWInv c.channel s' := by
  have hfr := hc.recv_total ok
  have hok_of_nle : (c.conn.onRecv ok).loadError = false → ok = true := by
    cases ok <;> simp [hc.recv_failure_reported]
  have hle_of_nok : ok = false → (c.conn.onRecv ok).loadError = true := by
    intro hnok; subst hnok; exact hc.recv_failure_reported
  have hdc := h.d_cnt
  have hpc := h.p_cnt
  have hph := h.p_held
  cases hch : c.channel with
  | eagerBlock =>
    cases w with
    | false =>
      simp only [mwDWorker, hch] at hs
      split at hs
      · rename_i hn0
        obtain ⟨hst, hdh, hfin⟩ := h.d_partial (Or.inl hn0)
        cases hr1 : s.recvPub1 with
        | none =>
          cases hst with
          | inl hw =>
            cases hl : (c.conn.onRecv ok).loadError with
            | false =>
              have hok := hok_of_nle hl; subst hok
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn0, hr1, optCount, ite_true,
                applyRecv, toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp [hw, hdh, hfin], d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp [hw], d_dec := by simp [hw] }
            | true =>
              cases hpol : c.policy <;>
                simp only [mwAggRecv, hfr, hl, hd, hdc, hn0, hr1, optCount, ite_true,
                  applyRecv, applyLoadError, hpol, hw, toS, withS, Option.some.injEq] at hs <;>
                subst hs <;>
                exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                        d_partial := by simp [hdh, hfin], d_wait := by simp,
                        d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                        d_fin := by simp, d_rec := by simp, d_dec := by simp }
          | inr hf =>
            cases hl : (c.conn.onRecv ok).loadError <;>
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn0, hr1, optCount, ite_true,
                applyRecv, applyLoadError, hf, toS, withS, Option.some.injEq] at hs <;>
              subst hs <;>
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp [hdh, hfin], d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
        | some b1 =>
          cases hst with
          | inl hw =>
            have hb1_fail : b1 = false → s.dFailRecv = true :=
              fun hb1 => h.d_fail_eager hch (Or.inr (by simp [hr1, hb1])) hw
            cases hl : (c.conn.onRecv ok).loadError with
            | false =>
              have hok := hok_of_nle hl; subst hok
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn0, hr1, optCount, ite_true,
                applyRecv, applyFinishedRecving, hw, toS, withS, Option.some.injEq] at hs
              subst hs
              have hdf : (some true = some false ∨ some b1 = some false) → s.dFailRecv = true := by
                intro hfail; rcases hfail with h0 | h1
                · simp at h0
                · simp only [Option.some.injEq] at h1; exact hb1_fail h1
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := fun _ hfail _ => hdf hfail,
                      d_pend := by simp,
                      d_fail := fun _ _ hfail _ => hdf hfail,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
            | true =>
              cases hpol : c.policy <;>
                simp only [mwAggRecv, hfr, hl, hd, hdc, hn0, hr1, optCount, ite_true,
                  applyRecv, applyLoadError, applyFinishedRecving, hpol, hw, toS, withS,
                  Option.some.injEq] at hs <;>
                subst hs <;>
                exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                        d_partial := by simp, d_wait := by simp,
                        d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                        d_fin := by simp, d_rec := by simp, d_dec := by simp }
          | inr hf =>
            cases hl : (c.conn.onRecv ok).loadError <;>
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn0, hr1, optCount, ite_true,
                applyRecv, applyLoadError, applyFinishedRecving, hf, toS, withS,
                Option.some.injEq] at hs <;>
              subst hs <;>
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
      · simp at hs
    | true =>
      simp only [mwDWorker, hch] at hs
      split at hs
      · rename_i hn1
        obtain ⟨hst, hdh, hfin⟩ := h.d_partial (Or.inr hn1)
        cases hr0 : s.recvPub0 with
        | none =>
          cases hst with
          | inl hw =>
            cases hl : (c.conn.onRecv ok).loadError with
            | false =>
              have hok := hok_of_nle hl; subst hok
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn1, hr0, optCount, ite_true,
                applyRecv, toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp [hw, hdh, hfin], d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp [hw], d_dec := by simp [hw] }
            | true =>
              cases hpol : c.policy <;>
                simp only [mwAggRecv, hfr, hl, hd, hdc, hn1, hr0, optCount, ite_true,
                  applyRecv, applyLoadError, hpol, hw, toS, withS, Option.some.injEq] at hs <;>
                subst hs <;>
                exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                        d_partial := by simp [hdh, hfin], d_wait := by simp,
                        d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                        d_fin := by simp, d_rec := by simp, d_dec := by simp }
          | inr hf =>
            cases hl : (c.conn.onRecv ok).loadError <;>
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn1, hr0, optCount, ite_true,
                applyRecv, applyLoadError, hf, toS, withS, Option.some.injEq] at hs <;>
              subst hs <;>
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp [hdh, hfin], d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
        | some b0 =>
          cases hst with
          | inl hw =>
            have hb0_fail : b0 = false → s.dFailRecv = true :=
              fun hb0 => h.d_fail_eager hch (Or.inl (by simp [hr0, hb0])) hw
            cases hl : (c.conn.onRecv ok).loadError with
            | false =>
              have hok := hok_of_nle hl; subst hok
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn1, hr0, optCount, ite_true,
                applyRecv, applyFinishedRecving, hw, toS, withS, Option.some.injEq] at hs
              subst hs
              have hdf : (some b0 = some false ∨ some true = some false) → s.dFailRecv = true := by
                intro hfail; rcases hfail with h0 | h1
                · simp only [Option.some.injEq] at h0; exact hb0_fail h0
                · simp at h1
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := fun _ hfail _ => hdf hfail,
                      d_pend := by simp,
                      d_fail := fun _ _ hfail _ => hdf hfail,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
            | true =>
              cases hpol : c.policy <;>
                simp only [mwAggRecv, hfr, hl, hd, hdc, hn1, hr0, optCount, ite_true,
                  applyRecv, applyLoadError, applyFinishedRecving, hpol, hw, toS, withS,
                  Option.some.injEq] at hs <;>
                subst hs <;>
                exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                        d_partial := by simp, d_wait := by simp,
                        d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                        d_fin := by simp, d_rec := by simp, d_dec := by simp }
          | inr hf =>
            cases hl : (c.conn.onRecv ok).loadError <;>
              simp only [mwAggRecv, hfr, hl, hd, hdc, hn1, hr0, optCount, ite_true,
                applyRecv, applyLoadError, applyFinishedRecving, hf, toS, withS,
                Option.some.injEq] at hs <;>
              subst hs <;>
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
      · simp at hs
  | latchedReq =>
    cases w with
    | false =>
      simp only [mwDWorker, hch] at hs
      split at hs
      · rename_i hn0
        obtain ⟨hst, hdh, hfin⟩ := h.d_partial (Or.inl hn0)
        cases hr1 : s.recvPub1 with
        | none =>
          -- First worker (`cnt' = 1 ≠ 2`, `fin = false`): scheduler untouched, `dFailPending` latches `loadError`.
          simp [mwAggRecv, hfr, hd, hdc, hn0, hr1, optCount, applyRecv, toS, withS] at hs
          subst hs
          exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                  d_partial := by simp [hst, hdh, hfin], d_wait := by simp,
                  d_fail_eager := by simp,
                  d_pend := by
                    intro _ _ hfail
                    rcases hfail with h0 | h1
                    · simp only [Option.some.injEq] at h0
                      simp [hle_of_nok h0]
                    · simp at h1
                  d_fail := by simp, d_fin := by simp,
                  d_rec := by intro hr; rcases hst with hw | hf; { simp [hw] at hr }; { simp [hf] at hr },
                  d_dec := by intro hd0; rcases hst with hw | hf; { simp [hw] at hd0 }; { simp [hf] at hd0 } }
        | some b1 =>
          -- Second worker (`cnt' = 2 = expectedD`, `fin = true`): emits latched `dFailPending || loadError`.
          have hb1_pend : b1 = false → s.dFailPending = true :=
            fun hb1 => h.d_pend hch (Or.inl hn0) (Or.inr (by simp [hr1, hb1]))
          cases hem : (s.dFailPending || (c.conn.onRecv ok).loadError) with
          | false =>
            have hnp : s.dFailPending = false ∧ (c.conn.onRecv ok).loadError = false := by
              simpa [Bool.or_eq_false_iff] using hem
            have hok : ok = true := hok_of_nle hnp.2
            have hb1 : b1 = true := by
              cases b1
              · exact absurd (hb1_pend rfl) (by simp [hnp.1])
              · rfl
            subst hok; subst hb1
            cases hst with
            | inl hw =>
              simp only [mwAggRecv, hfr, hd, hdc, hn0, hr1, optCount, ite_true,
                hem, applyRecv, applyFinishedRecving, hw, toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
            | inr hf =>
              simp only [mwAggRecv, hfr, hd, hdc, hn0, hr1, optCount, ite_true,
                hem, applyRecv, applyFinishedRecving, hf, toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
          | true =>
            cases hst with
            | inl hw =>
              cases hpol : c.policy <;>
                simp only [mwAggRecv, hfr, hd, hdc, hn0, hr1, optCount, ite_true,
                  hem, applyRecv, applyLoadError, applyFinishedRecving, hpol, hw,
                  toS, withS, Option.some.injEq] at hs <;>
                subst hs <;>
                exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                        d_partial := by simp, d_wait := by simp,
                        d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                        d_fin := by simp, d_rec := by simp, d_dec := by simp }
            | inr hf =>
              simp only [mwAggRecv, hfr, hd, hdc, hn0, hr1, optCount, ite_true,
                hem, applyRecv, applyLoadError, applyFinishedRecving, hf,
                toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
      · simp at hs
    | true =>
      simp only [mwDWorker, hch] at hs
      split at hs
      · rename_i hn1
        obtain ⟨hst, hdh, hfin⟩ := h.d_partial (Or.inr hn1)
        cases hr0 : s.recvPub0 with
        | none =>
          simp [mwAggRecv, hfr, hd, hdc, hn1, hr0, optCount, applyRecv, toS, withS] at hs
          subst hs
          exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                  d_partial := by simp [hst, hdh, hfin], d_wait := by simp,
                  d_fail_eager := by simp,
                  d_pend := by
                    intro _ _ hfail
                    rcases hfail with h0 | h1
                    · simp at h0
                    · simp only [Option.some.injEq] at h1
                      simp [hle_of_nok h1]
                  d_fail := by simp, d_fin := by simp,
                  d_rec := by intro hr; rcases hst with hw | hf; { simp [hw] at hr }; { simp [hf] at hr },
                  d_dec := by intro hd0; rcases hst with hw | hf; { simp [hw] at hd0 }; { simp [hf] at hd0 } }
        | some b0 =>
          have hb0_pend : b0 = false → s.dFailPending = true :=
            fun hb0 => h.d_pend hch (Or.inr hn1) (Or.inl (by simp [hr0, hb0]))
          cases hem : (s.dFailPending || (c.conn.onRecv ok).loadError) with
          | false =>
            have hnp : s.dFailPending = false ∧ (c.conn.onRecv ok).loadError = false := by
              simpa [Bool.or_eq_false_iff] using hem
            have hok : ok = true := hok_of_nle hnp.2
            have hb0 : b0 = true := by
              cases b0
              · exact absurd (hb0_pend rfl) (by simp [hnp.1])
              · rfl
            subst hok; subst hb0
            cases hst with
            | inl hw =>
              simp only [mwAggRecv, hfr, hd, hdc, hn1, hr0, optCount, ite_true,
                hem, applyRecv, applyFinishedRecving, hw, toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
            | inr hf =>
              simp only [mwAggRecv, hfr, hd, hdc, hn1, hr0, optCount, ite_true,
                hem, applyRecv, applyFinishedRecving, hf, toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
          | true =>
            cases hst with
            | inl hw =>
              cases hpol : c.policy <;>
                simp only [mwAggRecv, hfr, hd, hdc, hn1, hr0, optCount, ite_true,
                  hem, applyRecv, applyLoadError, applyFinishedRecving, hpol, hw,
                  toS, withS, Option.some.injEq] at hs <;>
                subst hs <;>
                exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                        d_partial := by simp, d_wait := by simp,
                        d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                        d_fin := by simp, d_rec := by simp, d_dec := by simp }
            | inr hf =>
              simp only [mwAggRecv, hfr, hd, hdc, hn1, hr0, optCount, ite_true,
                hem, applyRecv, applyLoadError, applyFinishedRecving, hf,
                toS, withS, Option.some.injEq] at hs
              subst hs
              exact { p_cnt := hpc, p_held := hph, d_cnt := by simp,
                      d_partial := by simp, d_wait := by simp,
                      d_fail_eager := by simp, d_pend := by simp, d_fail := by simp,
                      d_fin := by simp, d_rec := by simp, d_dec := by simp }
      · simp at hs

theorem mw_inv_promote {ch : AggChannel} {s s' : MWS}
    (h : MWInv ch s) (hs : mwPromote s = some s') : MWInv ch s' := by
  simp only [mwPromote, promote, Option.map_eq_some_iff] at hs
  rcases hs with ⟨u, hu, rfl⟩
  split at hu
  · rename_i hw
    simp only [toS] at hw
    obtain ⟨hw, hfin⟩ := hw
    have hboth : s.recvPub0 ≠ none ∧ s.recvPub1 ≠ none := by
      constructor <;> intro hn
      · have := (h.d_partial (Or.inl hn)).2.2; rw [hfin] at this; cases this
      · have := (h.d_partial (Or.inr hn)).2.2; rw [hfin] at this; cases this
    have hnp : ¬(s.recvPub0 = none ∨ s.recvPub1 = none) :=
      fun hp => hp.elim hboth.1 hboth.2
    split at hu
    · simp only [Option.some.injEq] at hu; subst hu
      exact { p_cnt := h.p_cnt, p_held := h.p_held, d_cnt := h.d_cnt,
              d_partial := fun hp => absurd hp hnp, d_wait := by simp,
              d_fail_eager := by simp, d_pend := fun _ hp => absurd hp hnp,
              d_fail := by simp, d_fin := by simp, d_rec := by simp, d_dec := by simp }
    · rename_i hfail
      simp only [toS] at hfail
      simp only [Option.some.injEq] at hu; subst hu
      have hok : s.recvPub0 = some true ∧ s.recvPub1 = some true := by
        cases hr0 : s.recvPub0 with
        | none => exact absurd hr0 hboth.1
        | some b0 =>
          cases hr1 : s.recvPub1 with
          | none => exact absurd hr1 hboth.2
          | some b1 =>
            cases b0
            · have := h.d_fail hboth.1 hboth.2 (Or.inl hr0) hw; simp [this] at hfail
            · cases b1
              · have := h.d_fail hboth.1 hboth.2 (Or.inr hr1) hw; simp [this] at hfail
              · exact ⟨rfl, rfl⟩
      exact { p_cnt := h.p_cnt, p_held := h.p_held, d_cnt := h.d_cnt,
              d_partial := fun hp => absurd hp hnp, d_wait := by simp,
              d_fail_eager := by simp, d_pend := fun _ hp => absurd hp hnp,
              d_fail := by simp, d_fin := by simp, d_rec := by simp,
              d_dec := by simpa using hok }
  · simp at hu

theorem mw_inv_finish {ch : AggChannel} {s s' : MWS}
    (h : MWInv ch s) (hs : mwFinish s = some s') : MWInv ch s' := by
  simp only [mwFinish, finish, Option.map_eq_some_iff] at hs
  rcases hs with ⟨u, hu, rfl⟩
  split at hu
  · rename_i hw; simp only [toS] at hw
    simp only [Option.some.injEq] at hu; subst hu
    exact { p_cnt := h.p_cnt, p_held := h.p_held, d_cnt := h.d_cnt,
            d_partial := fun hp => by
              obtain ⟨_, hdh, hfin⟩ := h.d_partial hp; simp [hdh, hfin]
            d_wait := by simp, d_fail_eager := by simp, d_pend := h.d_pend,
            d_fail := by simp,
            d_fin := fun h0 h1 _ => by have := h.d_wait h0 h1 hw; simp [this]
            d_rec := by simp, d_dec := by simp }
  · rename_i hd; simp only [toS] at hd
    simp only [Option.some.injEq] at hu; subst hu
    obtain ⟨hr0, hr1⟩ := h.d_dec hd
    exact { p_cnt := h.p_cnt, p_held := h.p_held, d_cnt := h.d_cnt,
            d_partial := by simp [hr0, hr1], d_wait := by simp,
            d_fail_eager := by simp, d_pend := h.d_pend, d_fail := by simp,
            d_fin := by simp, d_rec := by simp, d_dec := by simp }
  · rename_i hr; simp only [toS] at hr
    simp only [Option.some.injEq] at hu; subst hu
    have hheld := h.d_rec hr
    exact { p_cnt := h.p_cnt, p_held := h.p_held, d_cnt := h.d_cnt,
            d_partial := fun hp => by have := (h.d_partial hp).1; simp [hr] at this
            d_wait := by simp, d_fail_eager := by simp, d_pend := h.d_pend,
            d_fail := by simp,
            d_fin := fun _ _ _ => by simpa using hheld
            d_rec := by simp, d_dec := by simp }
  · simp at hu

theorem mw_step_inv {c : MWCfg} (hc : Contract c.conn)
    (hp : c.expectedP = 2) (hd : c.expectedD = 2)
    {s s' : MWS} {e : MWEv} (h : MWInv c.channel s)
    (hs : mwStep c s e = some s') : MWInv c.channel s' := by
  cases e with
  | pWorker w ok => exact mw_inv_pWorker hc hp h hs
  | dWorker w ok => exact mw_inv_dWorker hc hd h hs
  | promote => exact mw_inv_promote h hs
  | finish => exact mw_inv_finish h hs

/-- Any connector satisfying `Contract` is safe across multi-worker
`KVOutputAggregator` aggregation on **both** `AggChannel.eagerBlock` and
`AggChannel.latchedReq` when the aggregator thresholds match the participating
worker counts (`expectedP = 2`, `expectedD = 2`), under either failure policy
and across all staggered worker completion/failure interleavings. -/
theorem mw_contract_safe {c : MWCfg} (hc : Contract c.conn)
    (hp : c.expectedP = 2) (hd : c.expectedD = 2) {s : MWS}
    (hr : (mwSys c).Reachable s) : MWSafe s :=
  mw_inv_safe ((mwSys c).reachable_induction (mw_inv_init c.channel)
    (fun _ _ _ h hs => mw_step_inv hc hp hd h hs) hr)

theorem mw_twoChannel_safe {policy : Policy} {ch : AggChannel} {s : MWS}
    (hr : (mwSys ⟨connectorTwoChannel, policy, ch, 2, 2⟩).Reachable s) : MWSafe s :=
  mw_contract_safe twoChannel_contract rfl rfl hr

/-- **Multi-worker early uncertain-load failure under `Policy.fail`:**
In `vllm-torchtpu` (`tpu_connector.py` `_apply_stage3_submit_outcome` and
`_sync_stage3_tp_submit_states`), when `tp_rank == 0` gets an `"uncertain"`
post-arm `coordinate_transfer` error with `_stage3_fail_uncertain_loads_early`
enabled (`Policy.fail`, `AggChannel.eagerBlock`), `tp_rank == 0` stages
`_failed_block_ids` on that pass (`KVOutputAggregator.aggregate` forwards
`loadError = true` with `finishedRecving = false`) while
`_sync_stage3_tp_submit_states` broadcasts `"uncertain"` to all `tp_rank != 0`
so every rank withholds `done_recving` until its native receiver or 180 s
deadline settles. On any waiting `MWS` state, this early `applyLoadError .fail`
equals `mwFinish s`, preserves `MWInv`, and keeps `dHeld = true` while any
worker is still unpublished. -/
theorem mw_earlyLoadError_fail_eq_finish {s : MWS} (hw : s.dStatus = .waitingRemoteKv) :
    some (withS s (applyLoadError .fail (toS s))) = mwFinish s := by
  simp [mwFinish, finish, applyLoadError, toS, hw]

theorem mw_earlyLoadError_fail_holds_blocks {ch : AggChannel} {s : MWS}
    (h : MWInv ch s) (hw : s.dStatus = .waitingRemoteKv)
    (hpart : s.recvPub0 = none ∨ s.recvPub1 = none) :
    let s' := withS s (applyLoadError .fail (toS s))
    MWInv ch s' ∧ s'.dStatus = .finished ∧ s'.dHeld = true ∧ s'.dFinRecv = false := by
  have hfin : mwFinish s = some (withS s (applyLoadError .fail (toS s))) :=
    (mw_earlyLoadError_fail_eq_finish hw).symm
  have hinv' : MWInv ch (withS s (applyLoadError .fail (toS s))) :=
    mw_inv_finish h hfin
  obtain ⟨_, hdh, hfr⟩ := hinv'.d_partial hpart
  exact ⟨hinv', by simp [applyLoadError, toS, hw], hdh, hfr⟩

/-! ## Bounded search and concrete multi-worker traces -/

def mwAllEvents : List MWEv :=
  [ .pWorker false true, .pWorker false false,
    .pWorker true true, .pWorker true false,
    .dWorker false true, .dWorker false false,
    .dWorker true true, .dWorker true false,
    .promote, .finish ]

def mwSearch (c : MWCfg) (events : List MWEv := mwAllEvents) (fuel : Nat := 16) :
    ModelCheck.Result MWEv :=
  ModelCheck.check (mwSys c) events mwViolates fuel

def mwLeaked (c : MWCfg) (events : List MWEv := mwAllEvents) : Bool :=
  match mwSearch c events with
  | .counterexample _ => true
  | _ => false

def mwView (s : MWS) : View := view (toS s)

def mwTwoChannel (p : Policy) (ch : AggChannel := .eagerBlock) : MWCfg :=
  ⟨connectorTwoChannel, p, ch, 2, 2⟩
def mwDropFailed (p : Policy) (ch : AggChannel := .eagerBlock) : MWCfg :=
  ⟨connectorDropFailed, p, ch, 2, 2⟩

#guard mwSearch (mwTwoChannel .fail .eagerBlock) = .safe
#guard mwSearch (mwTwoChannel .recompute .eagerBlock) = .safe
#guard mwSearch (mwTwoChannel .fail .latchedReq) = .safe
#guard mwSearch (mwTwoChannel .recompute .latchedReq) = .safe
#guard mwLeaked (mwDropFailed .fail .eagerBlock) = true
#guard mwLeaked (mwDropFailed .recompute .eagerBlock) = true
#guard mwLeaked (mwDropFailed .fail .latchedReq) = true
#guard mwLeaked (mwDropFailed .recompute .latchedReq) = true

/-- **Staggered asymmetric D outcome on `AggChannel.eagerBlock` under `Policy.fail`
(fail-first, succeed-second).** Consumer worker 0 fails (`dWorker false false`):
`KVOutputAggregator` forwards `loadError = true` immediately while holding
`finishedRecving = false` (`dRecvCount = 1 < 2`), so the scheduler finishes the
request with `dHeld = true` (keeping destination HBM allocated while worker 1's
H2D is still in flight). When consumer worker 1 later succeeds
(`dRecvCount = 2`), the aggregator emits `finishedRecving = true` and the
scheduler frees `dHeld = false`. -/
theorem trace_mw_staggered_fail_then_ok_fail :
    ((mwSys (mwTwoChannel .fail .eagerBlock)).run [.dWorker false false]).map mwView =
      some ⟨true, .finished, true, false, false⟩ ∧
    ((mwSys (mwTwoChannel .fail .eagerBlock)).run
      [.dWorker false false, .dWorker true true]).map mwView =
      some ⟨true, .finished, false, false, false⟩ := by
  decide

/-- **Staggered asymmetric D outcome on `AggChannel.eagerBlock` under
`Policy.recompute` (fail-first, succeed-second).** Consumer worker 0 fails
(`dWorker false false`): `loadError = true` sets `dFailRecv = true` while
`dFinRecv` stays `false` (`dRecvCount = 1 < 2`), so the request cannot be
promoted while worker 1 is still in flight. When worker 1 later succeeds
(`dRecvCount = 2`), `dFinRecv` becomes `true` with `dFailRecv` still `true`, and
`promote` sends the request to `.recomputing` (`dHeld = false`). -/
theorem trace_mw_staggered_fail_then_ok_recompute :
    ((mwSys (mwTwoChannel .recompute .eagerBlock)).run [.dWorker false false]).map mwView =
      some ⟨true, .waitingRemoteKv, true, false, true⟩ ∧
    ((mwSys (mwTwoChannel .recompute .eagerBlock)).run
      [.dWorker false false, .dWorker true true, .promote]).map mwView =
      some ⟨true, .recomputing, false, false, false⟩ := by
  decide

/-- **Staggered asymmetric D outcome on `AggChannel.eagerBlock` under
`Policy.recompute` (succeed-first, fail-second).** Consumer worker 0 succeeds
(`dWorker false true`), `KVOutputAggregator` withholds `finishedRecving`
(`dRecvCount = 1 < 2`), preventing premature promotion on worker 0's shard
alone; when worker 1 fails (`dWorker true false`), the aggregator emits
`finishedRecving = true` together with `loadError = true`, and `promote`
transitions to `.recomputing` (`dHeld = false`). -/
theorem trace_mw_staggered_ok_then_fail_recompute :
    ((mwSys (mwTwoChannel .recompute .eagerBlock)).run [.dWorker false true]).map mwView =
      some ⟨true, .waitingRemoteKv, true, false, false⟩ ∧
    ((mwSys (mwTwoChannel .recompute .eagerBlock)).run
      [.dWorker false true, .dWorker true false, .promote]).map mwView =
      some ⟨true, .recomputing, false, false, false⟩ := by
  decide

/-- **Staggered asymmetric D outcome on `AggChannel.latchedReq` under
`Policy.fail` (`tpu-inference` `get_transfer_results` +
`KVOutputAggregator._failed_recving_pending`, fail-first, succeed-second).**
Consumer worker 0 fails (`dWorker false false`): `dFailPending` latches `true`
while the request stays in `WAITING_FOR_REMOTE_KVS` holding its blocks
(`dHeld = true`) until worker 1 finishes. When worker 1 succeeds
(`dRecvCount = 2`), the aggregator emits `finishedRecving = true` and
`loadError = true` in the same step (`dFailPending` cleared to `false`), so the
scheduler marks the request `.finished` and releases `dHeld = false`. -/
theorem trace_mw_latched_staggered_fail_then_ok_fail :
    ((mwSys (mwTwoChannel .fail .latchedReq)).run [.dWorker false false]).map
      (fun s => (s.dFailPending, mwView s)) =
      some (true, ⟨true, .waitingRemoteKv, true, false, false⟩) ∧
    ((mwSys (mwTwoChannel .fail .latchedReq)).run
      [.dWorker false false, .dWorker true true]).map
      (fun s => (s.dFailPending, mwView s)) =
      some (false, ⟨true, .finished, false, false, false⟩) := by
  decide

/-- **Staggered asymmetric D outcome on `AggChannel.latchedReq` under
`Policy.recompute` (fail-first, succeed-second).** Worker 0's failure stays
latched in `dFailPending = true` (`dFailRecv = false` in the scheduler) until
worker 1 succeeds (`dRecvCount = 2`), at which point both `dFinRecv` and
`dFailRecv` become `true` and `promote` transitions to `.recomputing`
(`dHeld = false`). -/
theorem trace_mw_latched_staggered_fail_then_ok_recompute :
    ((mwSys (mwTwoChannel .recompute .latchedReq)).run [.dWorker false false]).map
      (fun s => (s.dFailPending, mwView s)) =
      some (true, ⟨true, .waitingRemoteKv, true, false, false⟩) ∧
    ((mwSys (mwTwoChannel .recompute .latchedReq)).run
      [.dWorker false false, .dWorker true true, .promote]).map
      (fun s => (s.dFailPending, mwView s)) =
      some (false, ⟨true, .recomputing, false, false, false⟩) := by
  decide

/-- **Mutant `connectorDropFailed`: partial worker failure leaks despite one
worker succeeding.** Worker 0 succeeds (`ok = true`) and worker 1 fails
(`ok = false`, dropped by `connectorDropFailed`). Both workers have settled in
`tpu-sync` (`allPSent` / `allDRecv`), but `pSentCount` / `dRecvCount` stop at
`1 < 2`, leaking the request's blocks on P and D forever (`mwLeakP_absorbing`,
`mwLeakD_absorbing`). -/
theorem trace_mw_dropFailed_partial_failure_leaks :
    ((mwSys (mwDropFailed .fail)).run
      [.pWorker false true, .pWorker true false]).map
      (fun s => (s.pSentCount, mwLeakP s)) = some (1, true) ∧
    ((mwSys (mwDropFailed .fail)).run
      [.dWorker false true, .dWorker true false]).map
      (fun s => (s.dRecvCount, mwLeakD s)) = some (1, true) := by
  decide

/-- **Mutant `connectorLoadErrorOnly` on `AggChannel.latchedReq`: latched
`failed_recving` is never even emitted to the scheduler.** Both workers fail,
latching `dFailPending = true`, but neither emits `finishedRecving = true`
(`dRecvCount = 0 < 2`), so `_failed_recving_pending & finished_recving` stays
empty and the request stays parked in `WAITING_FOR_REMOTE_KVS` (`mwLeakD = true`). -/
theorem trace_mw_latched_loadErrorOnly_never_emits :
    let c : MWCfg := ⟨connectorLoadErrorOnly, .fail, .latchedReq, 2, 2⟩
    ((mwSys c).run [.dWorker false false, .dWorker true false]).map
      (fun s => (s.dRecvCount, s.dFailPending, mwView s, mwLeakD s)) =
      some (0, true, ⟨true, .waitingRemoteKv, true, false, false⟩, true) := by
  decide

/-- **Over-counted `get_finished_count()` leaks even under `connectorTwoChannel`.**
If `get_finished_count()` exceeds the number of workers that actually run a
transfer session for the request (`expectedP = 3` or `expectedD = 3` when only 2
workers participate), `pSentCount` and `dRecvCount` stop at `2 < 3` after all
participating workers complete, leaking the request's blocks on both P and D. -/
theorem trace_mw_overcounted_expected_leaks :
    let c : MWCfg := ⟨connectorTwoChannel, .fail, .latchedReq, 3, 3⟩
    ((mwSys c).run [.pWorker false true, .pWorker true true]).map
      (fun s => (s.pSentCount, mwLeakP s)) = some (2, true) ∧
    ((mwSys c).run [.dWorker false true, .dWorker true true]).map
      (fun s => (s.dRecvCount, mwLeakD s)) = some (2, true) := by
  decide

end TpuSyncVerify.Transfer.Caller

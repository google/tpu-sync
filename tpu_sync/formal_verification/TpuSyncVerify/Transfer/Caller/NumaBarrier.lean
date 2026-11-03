import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.Caller.Traces

/-!
# Intra-worker NUMA sub-manager barrier (`TpuSyncVerify.Transfer.Caller.NumaBarrier`)

Where `MultiWorker.lean` models `KVOutputAggregator` combining completions across
separate worker processes above `get_transfer_results()`, this module models the
barrier *inside* a single JAX worker (`NumaAwareKVCacheManager::CompleteReadRaw`
in `tpu_sync/frameworks/jax/kv_cache_manager.cc`), which aggregates
`(done_sending, done_recving, failed_recving)` across `sub_managers_`
(`num_subs = 1` under `ENABLE_MULTI_NUMA=0`, `num_subs = 2` under
`ENABLE_MULTI_NUMA=1`) before `TPUConnectorWorker.get_transfer_results` forwards
the worker's report to vLLM, across arbitrarily many sequential transfers reusing
the same `req_id` (`recycle`).

Two barrier policies (`NumaPolicy`) are compared over the same sub-manager,
connector, and scheduler transitions:

1. `NumaPolicy.eagerFailSet` — the `NumaAwareKVCacheManager::CompleteReadRaw`
   implementation in `tpu_sync/frameworks/jax/kv_cache_manager.cc`:
   * `done_sending_counts_[req]` emits `done_sending` and erases its entry at
     `num_subs`;
   * `done_recving_counts_[req]` emits `done_recving` (unless `req` is in
     `failed_recving_set_`) and erases `done_recving_counts_[req]`,
     `req_expected_counts_[req]`, and `failed_recving_set_[req]` at `expected`;
   * `if (failed_recving_set_.insert(req).second)` emits `failed_recving`
     immediately on the first failing sub-manager without waiting for sibling
     sub-managers or erasing any map entry
     (`KVCacheManagerWrapperTest.AnyFailureImmediatelyTriggersFailedRecvingAndSuppressesFutureAcks`).
   Witnesses show both coupled failure modes:
   * under `twoSubs = true` (`num_subs = 2`), a failing sub-manager causes the
     engine to free consumer or producer HBM while the sibling sub-manager's DMA
     is still in flight (`trace_eagerFailSet_recv_early_free`,
     `trace_eagerFailSet_send_early_free`);
   * across `recycle` (`req_id` reuse, **N3**), `failed_recving_set_` stays
     populated even at `twoSubs = false` (`num_subs = 1`), suppressing later
     failure and success reports into permanent block leaks
     (`trace_eagerFailSet_single_sub_reuse_leaks`), while at `num_subs = 2` a
     partial failure leaves `done_recving_counts_[req] = 1` and
     `failed_recving_set_` set, suppressing the second transfer's `done_recving`
     and causing the third transfer to emit `done_recving` after only one of two
     sub-managers finishes (`trace_eagerFailSet_partial_fail_desync_premature_doneRecving`).

2. `NumaPolicy.settledCounts` — the settled-count barrier (`failed_recving_counts_`
   replacing `failed_recving_set_`): every sub-manager terminal report increments
   its respective count map (`done_sending_counts_`, `done_recving_counts_`, or
   `failed_recving_counts_`), emits once
   `done_sending_counts_ + done_recving_counts_ + failed_recving_counts_ ≥ expected`
   (`failed_recving` if `failed_recving_counts_ > 0`, else `done_recving` or
   `done_sending`), and erases all four maps for `req` (no such policy exists in
   C++ at the baseline commit).
   * `numa_settledCounts_safe` proves inductively across unbounded `recycle`
     rounds, for both `num_subs = 1` and `num_subs = 2`, both failure policies,
     and any `Contract`-satisfying connector, that HBM is never freed while a
     sub-manager is still unpublished (`numaEarlyFree = false`, discharging
     `Pipeline` **A4** across NUMA sub-managers), no blocks leak on P or D
     (`numaLeakP = false`, `numaLeakD = false`), and D never decodes unwritten KV
     (`numaDecodesUnwritten = false`).
   * `numa_settledCounts_maps_clean` proves all four per-request map entries are
     erased whenever all participating sub-managers settle, so `recycle` returns
     to the exact initial state (which exhaustive `ModelCheck.check` confirms on
     the cyclic state graph).
   * `numaProcessSettled_expected_numSubs_eq_none` proves that pre-populating
     `req_expected_counts_[req] = num_subs` in `StartRead(remote_descriptors)`
     emits the identical report and post-settlement state as leaving
     `req_expected_counts_` unset (`StartRead(remote_endpoint)`).

## Assumptions

* **N1 (one report per participating sub-manager per round).** Each of the
  `num_subs` sub-managers publishes exactly one terminal outcome for the
  request per transfer (`pSub` / `dSub` fire once per round). This is `Caller`
  **B1** applied per sub-manager; the model assumes it and does not discharge
  it. `KVCacheManagerWithTransfer::StartRead` reports on every exit path
  reachable from `NumaAwareKVCacheManager::StartRead` (session creation and
  registration failures go to `failed_recving_`; a same-`req_id` re-delivery is
  answered by the incumbent). On P, `NumaAwareKVCacheManager::NotifyForRead`
  registers on every sub-manager in turn with the same `block_ids`; empty
  `block_ids` registers nothing on any sub-manager, and a throw on a later
  sub-manager after an earlier one registered would leave the barrier one
  report short under both policies (not modelled).
* **N2 (same `num_subs` on P and D).** `twoSubs` sets both sides; the two
  barriers are independent, so an asymmetric deployment changes nothing in the
  per-side argument, but it is not instantiated.
* **N3 (`req_id` reuse only after the previous transfer is released).**
  `recycle` fires only once every participating sub-manager on both nodes has
  settled and both schedulers have freed the blocks. Under
  `NumaPolicy.settledCounts` the sub-manager guard is implied by
  `pHeld = false ∧ dHeld = false` (nothing is freed before emission). Under
  `NumaPolicy.eagerFailSet` it also excludes a reuse racing a sibling's late
  report, so the `eagerFailSet` counterexamples are a lower bound. Reuse itself
  needs an engine that repeats request ids: vLLM appends a random suffix to
  every request id unless `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`; direct
  `jax::KVCacheManager` callers with deterministic ids reuse them freely. With
  unique ids the shipping code's per-request entries are never reused but never
  erased either (unbounded map growth), which a one-request model cannot show.
* **N4 (one sub-manager report per step).** One `CompleteReadRaw()` call polls
  every sub-manager and hands the whole batch to the connector; the model splits
  the batch into consecutive `pSub` / `dSub` steps, each followed by the
  connector and scheduler. This only adds interleavings, so it is sound for the
  safety results; the `nanobind` bindings `notify_for_read`, `start_read`, and
  `complete_read` in `tpu_sync/frameworks/jax/tpu_raiden_jax_module.cc` do not
  release the GIL, so the unguarded maps are never touched concurrently.
-/

namespace TpuSyncVerify.Transfer.Caller

/-- Which of the three `CompleteReadRaw()` vectors (`done_sending`,
`done_recving`, `failed_recving`) a `NumaAwareKVCacheManager` step emits for one
request. -/
inductive RawReport where
  | doneSending
  | doneRecving
  | failedRecving
  deriving Repr, DecidableEq

/-- Sub-manager aggregation policy in `NumaAwareKVCacheManager::CompleteReadRaw`. -/
inductive NumaPolicy where
  /-- Baseline `kv_cache_manager.cc`: eager `failed_recving` via `failed_recving_set_`,
  cleared only when `done_recving_counts_` reaches `expected`. -/
  | eagerFailSet
  /-- Settled-count barrier: `failed_recving_counts_` with emission and full map
  cleanup at `done_sending_counts_ + done_recving_counts_ + failed_recving_counts_ ≥ expected`. -/
  | settledCounts
  deriving Repr, DecidableEq

/-- Persistent per-`req_id` map state held inside one `NumaAwareKVCacheManager`
across `CompleteReadRaw()` polls and across transfers that reuse `req_id`. -/
structure NumaMgr where
  /-- `done_sending_counts_[req]` (`0` when absent). -/
  doneSendingCounts : Nat := 0
  /-- `done_recving_counts_[req]` (`0` when absent). -/
  doneRecvingCounts : Nat := 0
  /-- `failed_recving_counts_[req]` (`0` when absent; used by `settledCounts`). -/
  failedRecvingCounts : Nat := 0
  /-- `req ∈ failed_recving_set_` (used by `eagerFailSet`). -/
  failedRecvingSet : Bool := false
  /-- `req_expected_counts_[req]` (`none` when absent; set to `some numSubs` by
  `NumaAwareKVCacheManager::StartRead(remote_descriptors)`). -/
  reqExpected : Option Nat := none
  deriving Repr, DecidableEq

/-- Number of sub-managers (`1` under `ENABLE_MULTI_NUMA=0`, `2` under
`ENABLE_MULTI_NUMA=1`). -/
def numaCount (twoSubs : Bool) : Nat := if twoSubs then 2 else 1

/-- Settled-count completion check (`process_completion`): once total reports
reach `expected`, emit `failed_recving` if any sub-manager failed, else
`done_recving` / `done_sending`, and erase all four map entries for `req`. -/
def numaProcessSettled (numSubs : Nat) (m : NumaMgr) :
    NumaMgr × Option RawReport :=
  let expected := m.reqExpected.getD numSubs
  let total := m.doneSendingCounts + m.doneRecvingCounts + m.failedRecvingCounts
  if total ≥ expected then
    let out :=
      if m.failedRecvingCounts > 0 then RawReport.failedRecving
      else if m.doneRecvingCounts ≥ expected then RawReport.doneRecving
      else if m.doneSendingCounts ≥ expected then RawReport.doneSending
      else RawReport.failedRecving
    ({}, some out)
  else
    (m, none)

/-- Pre-setting `req_expected_counts_[req] = numSubs` in
`StartRead(remote_descriptors)` produces the exact same emitted report as leaving
`req_expected_counts_` unset (`none`), and resets to `{}` on emission. -/
theorem numaProcessSettled_expected_numSubs_eq_none (numSubs : Nat) (m : NumaMgr) :
    numaProcessSettled numSubs { m with reqExpected := some numSubs } =
      let (m', out) := numaProcessSettled numSubs { m with reqExpected := none }
      (if out.isSome then m' else { m' with reqExpected := some numSubs }, out) := by
  simp only [numaProcessSettled, Option.getD_some, Option.getD_none]
  split <;> simp

/-- One producer sub-manager report in `NumaAwareKVCacheManager::CompleteReadRaw`:
`ok = true` lands in `s` (`done_sending`), `ok = false` lands in `f`
(`failed_recving`, since `KVCacheManagerWithTransfer::CompleteReadWithDetails`
places failed or timed-out sends in `failed_recving_`). -/
def numaStepSend (pol : NumaPolicy) (numSubs : Nat) (ok : Bool) (m : NumaMgr) :
    NumaMgr × Option RawReport :=
  match pol with
  | .eagerFailSet =>
    if ok then
      let cnt := m.doneSendingCounts + 1
      if cnt = numSubs then
        ({ m with doneSendingCounts := 0 }, some .doneSending)
      else
        ({ m with doneSendingCounts := cnt }, none)
    else
      if !m.failedRecvingSet then
        ({ m with failedRecvingSet := true }, some .failedRecving)
      else
        (m, none)
  | .settledCounts =>
    if ok then
      numaProcessSettled numSubs { m with doneSendingCounts := m.doneSendingCounts + 1 }
    else
      numaProcessSettled numSubs { m with failedRecvingCounts := m.failedRecvingCounts + 1 }

/-- One consumer sub-manager report in `NumaAwareKVCacheManager::CompleteReadRaw`:
`ok = true` lands in `r` (`done_recving`), `ok = false` lands in `f`
(`failed_recving`). -/
def numaStepRecv (pol : NumaPolicy) (numSubs : Nat) (ok : Bool) (m : NumaMgr) :
    NumaMgr × Option RawReport :=
  match pol with
  | .eagerFailSet =>
    if ok then
      let cnt := m.doneRecvingCounts + 1
      let expected := m.reqExpected.getD numSubs
      if cnt ≥ expected then
        let emit := if !m.failedRecvingSet then some RawReport.doneRecving else none
        ({ m with doneRecvingCounts := 0, reqExpected := none, failedRecvingSet := false }, emit)
      else
        ({ m with doneRecvingCounts := cnt }, none)
    else
      if !m.failedRecvingSet then
        ({ m with failedRecvingSet := true }, some .failedRecving)
      else
        (m, none)
  | .settledCounts =>
    if ok then
      numaProcessSettled numSubs { m with doneRecvingCounts := m.doneRecvingCounts + 1 }
    else
      numaProcessSettled numSubs { m with failedRecvingCounts := m.failedRecvingCounts + 1 }

/-- Producer connector mapping from `CompleteReadRaw()` output for `req`. -/
def rawToSendSignal (conn : Connector) : Option RawReport → Signal
  | some .doneSending => conn.onSend true
  | some .failedRecving => conn.onSend false
  | _ => {}

/-- Consumer connector mapping from `CompleteReadRaw()` output for `req`. -/
def rawToRecvSignal (conn : Connector) : Option RawReport → Signal
  | some .doneRecving => conn.onRecv true
  | some .failedRecving => conn.onRecv false
  | _ => {}

/-- Configuration of the single-worker NUMA barrier + connector + scheduler model. -/
structure NumaCfg where
  conn : Connector
  policy : Policy
  numaPolicy : NumaPolicy := .settledCounts
  /-- `false` = 1 sub-manager (`ENABLE_MULTI_NUMA=0`);
  `true` = 2 sub-managers (`ENABLE_MULTI_NUMA=1`). -/
  twoSubs : Bool := true
  /-- `true` models `StartRead(remote_descriptors)` setting
  `req_expected_counts_[req_id] = sub_managers_.size()`; `false` models
  `StartRead(remote_endpoint)` leaving `req_expected_counts_` unset. -/
  setExpected : Bool := true

/-- State of one `req_id` across P's and D's `NumaAwareKVCacheManager`s, their
per-sub-manager session publications for the active round, and the engine
schedulers. -/
structure NumaS where
  /-- Producer sub-manager 0 publication in the current round. -/
  sendSub0 : Option Bool := none
  /-- Producer sub-manager 1 publication in the current round (unused when `!twoSubs`). -/
  sendSub1 : Option Bool := none
  /-- Consumer sub-manager 0 publication in the current round. -/
  recvSub0 : Option Bool := none
  /-- Consumer sub-manager 1 publication in the current round (unused when `!twoSubs`). -/
  recvSub1 : Option Bool := none
  /-- Producer `NumaAwareKVCacheManager` persistent map state for `req_id`. -/
  pMgr : NumaMgr := {}
  /-- Consumer `NumaAwareKVCacheManager` persistent map state for `req_id`. -/
  dMgr : NumaMgr := {}
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

def numaInitDExpected (c : NumaCfg) : Option Nat :=
  if c.setExpected then some (numaCount c.twoSubs) else none

def numaInit (c : NumaCfg) : NumaS :=
  { dMgr := { reqExpected := numaInitDExpected c } }

inductive NumaEv where
  /-- Producer sub-manager `sub` (`false` = sub 0, `true` = sub 1) settles with
  `ok`, polled by `NumaAwareKVCacheManager::CompleteReadRaw` on P. -/
  | pSub (sub : Bool) (ok : Bool)
  /-- Consumer sub-manager `sub` (`false` = sub 0, `true` = sub 1) settles with
  `ok`, polled by `NumaAwareKVCacheManager::CompleteReadRaw` on D. -/
  | dSub (sub : Bool) (ok : Bool)
  /-- D's next `Scheduler.schedule` promotes the waiting request. -/
  | promote
  /-- D's request leaves the engine (`Scheduler.finish_requests`). -/
  | finish
  /-- Once all participating sub-managers on P and D have settled and both
  schedulers have released the transfer's HBM, a new transfer reuses the same
  `req_id`, preserving `pMgr` and `dMgr` (with `StartRead` updating
  `dMgr.reqExpected` when `setExpected = true`). -/
  | recycle
  deriving Repr, DecidableEq

def allPSubsDone (twoSubs : Bool) (s : NumaS) : Bool :=
  s.sendSub0.isSome && (!twoSubs || s.sendSub1.isSome)

def allDSubsDone (twoSubs : Bool) (s : NumaS) : Bool :=
  s.recvSub0.isSome && (!twoSubs || s.recvSub1.isSome)

def allPSubsOk (twoSubs : Bool) (s : NumaS) : Bool :=
  (s.sendSub0 == some true) && (!twoSubs || (s.sendSub1 == some true))

def allDSubsOk (twoSubs : Bool) (s : NumaS) : Bool :=
  (s.recvSub0 == some true) && (!twoSubs || (s.recvSub1 == some true))

/-- Projection onto the single-worker `Caller.S` scheduler state, with aggregate
publication defined once every participating sub-manager has settled. -/
def numaToS (twoSubs : Bool) (s : NumaS) : S where
  sendPub := if allPSubsDone twoSubs s then some (allPSubsOk twoSubs s) else none
  recvPub := if allDSubsDone twoSubs s then some (allDSubsOk twoSubs s) else none
  pHeld := s.pHeld
  dStatus := s.dStatus
  dHeld := s.dHeld
  dFinRecv := s.dFinRecv
  dFailRecv := s.dFailRecv

def numaWithS (s : NumaS) (base : S) : NumaS :=
  { s with pHeld := base.pHeld, dStatus := base.dStatus, dHeld := base.dHeld,
           dFinRecv := base.dFinRecv, dFailRecv := base.dFailRecv }

def numaPSub (c : NumaCfg) (sub : Bool) (ok : Bool) (s : NumaS) : Option NumaS :=
  if sub && !c.twoSubs then none
  else
    let cur := if sub then s.sendSub1 else s.sendSub0
    if cur.isNone then
      let (pMgr', rep) := numaStepSend c.numaPolicy (numaCount c.twoSubs) ok s.pMgr
      let sig := rawToSendSignal c.conn rep
      let base' := applySend sig (numaToS c.twoSubs s)
      let s' := if sub then { s with sendSub1 := some ok, pMgr := pMgr' }
                else { s with sendSub0 := some ok, pMgr := pMgr' }
      some (numaWithS s' base')
    else none

def numaDSub (c : NumaCfg) (sub : Bool) (ok : Bool) (s : NumaS) : Option NumaS :=
  if sub && !c.twoSubs then none
  else
    let cur := if sub then s.recvSub1 else s.recvSub0
    if cur.isNone then
      let (dMgr', rep) := numaStepRecv c.numaPolicy (numaCount c.twoSubs) ok s.dMgr
      let sig := rawToRecvSignal c.conn rep
      let base' := applyRecv c.policy sig (numaToS c.twoSubs s)
      let s' := if sub then { s with recvSub1 := some ok, dMgr := dMgr' }
                else { s with recvSub0 := some ok, dMgr := dMgr' }
      some (numaWithS s' base')
    else none

def numaPromote (twoSubs : Bool) (s : NumaS) : Option NumaS :=
  (promote (numaToS twoSubs s)).map (numaWithS s)

def numaFinish (twoSubs : Bool) (s : NumaS) : Option NumaS :=
  (finish (numaToS twoSubs s)).map (numaWithS s)

def numaRecycle (c : NumaCfg) (s : NumaS) : Option NumaS :=
  if allPSubsDone c.twoSubs s && allDSubsDone c.twoSubs s && !s.pHeld && !s.dHeld then
    let dMgr' := if c.setExpected then
                   { s.dMgr with reqExpected := some (numaCount c.twoSubs) }
                 else s.dMgr
    some { sendSub0 := none, sendSub1 := none,
           recvSub0 := none, recvSub1 := none,
           pMgr := s.pMgr, dMgr := dMgr',
           pHeld := true, dStatus := .waitingRemoteKv, dHeld := true,
           dFinRecv := false, dFailRecv := false }
  else none

def numaStep (c : NumaCfg) (s : NumaS) : NumaEv → Option NumaS
  | .pSub sub ok => numaPSub c sub ok s
  | .dSub sub ok => numaDSub c sub ok s
  | .promote => numaPromote c.twoSubs s
  | .finish => numaFinish c.twoSubs s
  | .recycle => numaRecycle c s

def numaSys (c : NumaCfg) : System NumaS NumaEv := ⟨numaInit c, numaStep c⟩

/-! ## Properties -/

/-- P frees its HBM while a participating P sub-manager is still unpublished
(D2H still in flight), or D frees its HBM while a participating D sub-manager is
still unpublished (H2D still in flight). -/
def numaEarlyFree (twoSubs : Bool) (s : NumaS) : Bool :=
  (!allPSubsDone twoSubs s && !s.pHeld) || (!allDSubsDone twoSubs s && !s.dHeld)

/-- All participating P sub-managers have settled and P still holds the blocks. -/
def numaLeakP (twoSubs : Bool) (s : NumaS) : Bool :=
  leakP (numaToS twoSubs s)

/-- All participating D sub-managers have settled and D either keeps the request
parked with no completion signal or has finished it with the free deferred. -/
def numaLeakD (twoSubs : Bool) (s : NumaS) : Bool :=
  leakD (numaToS twoSubs s)

/-- D decodes while not every participating D sub-manager succeeded in the
current round. -/
def numaDecodesUnwritten (twoSubs : Bool) (s : NumaS) : Bool :=
  decodesUnwritten (numaToS twoSubs s)

def numaViolates (twoSubs : Bool) (s : NumaS) : Bool :=
  numaEarlyFree twoSubs s || numaLeakP twoSubs s ||
    numaLeakD twoSubs s || numaDecodesUnwritten twoSubs s

def NumaSafe (twoSubs : Bool) (s : NumaS) : Prop :=
  numaViolates twoSubs s = false

/-! ## Invariant and safety proof for `NumaPolicy.settledCounts` -/

private def b2n (b : Bool) : Nat := if b then 1 else 0

def pOkCount (twoSubs : Bool) (s : NumaS) : Nat :=
  b2n (s.sendSub0 == some true) + (if twoSubs then b2n (s.sendSub1 == some true) else 0)

def pFailCount (twoSubs : Bool) (s : NumaS) : Nat :=
  b2n (s.sendSub0 == some false) + (if twoSubs then b2n (s.sendSub1 == some false) else 0)

def dOkCount (twoSubs : Bool) (s : NumaS) : Nat :=
  b2n (s.recvSub0 == some true) + (if twoSubs then b2n (s.recvSub1 == some true) else 0)

def dFailCount (twoSubs : Bool) (s : NumaS) : Nat :=
  b2n (s.recvSub0 == some false) + (if twoSubs then b2n (s.recvSub1 == some false) else 0)

/-- Inductive invariant for `NumaPolicy.settledCounts` under any connector
meeting `Contract`:
1. Single-worker `Caller.Inv` holds on `numaToS c.twoSubs s` (stuttering
   refinement onto `Caller.sys`);
2. Sub-manager 1 stays `none` when `!c.twoSubs`;
3. `pHeld = true` before `allPSubsDone` and `dHeld = true` before `allDSubsDone`;
4. Before `allPSubsDone` (resp. `allDSubsDone`), `pMgr` (resp. `dMgr`) holds the
   exact count of `ok` and `fail` sub-managers in the current round; once all
   participating sub-managers settle, `pMgr = {}` (resp. `dMgr = {}`). -/
structure NumaInv (c : NumaCfg) (s : NumaS) : Prop where
  caller : Inv (numaToS c.twoSubs s)
  p_sub1 : c.twoSubs = false → s.sendSub1 = none
  d_sub1 : c.twoSubs = false → s.recvSub1 = none
  p_held : allPSubsDone c.twoSubs s = false → s.pHeld = true
  d_held : allDSubsDone c.twoSubs s = false → s.dHeld = true
  p_mgr_active : allPSubsDone c.twoSubs s = false →
    s.pMgr = { doneSendingCounts := pOkCount c.twoSubs s,
               failedRecvingCounts := pFailCount c.twoSubs s }
  p_mgr_done : allPSubsDone c.twoSubs s = true → s.pMgr = {}
  d_mgr_active : allDSubsDone c.twoSubs s = false →
    s.dMgr = { doneRecvingCounts := dOkCount c.twoSubs s,
               failedRecvingCounts := dFailCount c.twoSubs s,
               reqExpected := numaInitDExpected c }
  d_mgr_done : allDSubsDone c.twoSubs s = true → s.dMgr = {}

theorem numaInv_init (c : NumaCfg) : NumaInv c (numaInit c) where
  caller := by
    have h0 : numaToS c.twoSubs (numaInit c) = init := by
      cases c.twoSubs <;> rfl
    rw [h0]
    exact inv_init
  p_sub1 := by intro _; rfl
  d_sub1 := by intro _; rfl
  p_held := by intro _; rfl
  d_held := by intro _; rfl
  p_mgr_active := by cases c.twoSubs <;> intro _ <;> rfl
  p_mgr_done := by cases c.twoSubs <;> simp [numaInit, allPSubsDone]
  d_mgr_active := by cases c.twoSubs <;> intro _ <;> rfl
  d_mgr_done := by cases c.twoSubs <;> simp [numaInit, allDSubsDone]

theorem numaInv_safe {c : NumaCfg} {s : NumaS} (h : NumaInv c s) :
    NumaSafe c.twoSubs s := by
  have hcs : Safe (numaToS c.twoSubs s) := inv_safe h.caller
  simp only [Safe, violates, Bool.or_eq_false_iff] at hcs
  obtain ⟨⟨hlp, hld⟩, hdu⟩ := hcs
  simp only [NumaSafe, numaViolates, numaLeakP, numaLeakD, numaDecodesUnwritten,
    hlp, hld, hdu, Bool.or_false]
  simp only [numaEarlyFree, Bool.or_eq_false_iff, Bool.and_eq_false_iff,
    Bool.not_eq_false']
  refine ⟨?_, ?_⟩
  · cases hp : allPSubsDone c.twoSubs s
    · right; exact h.p_held hp
    · left; rfl
  · cases hd : allDSubsDone c.twoSubs s
    · right; exact h.d_held hd
    · left; rfl

private theorem stepSend_1 (ok : Bool) :
    numaStepSend .settledCounts 1 ok { doneSendingCounts := 0, failedRecvingCounts := 0 } =
      ({}, some (if ok then .doneSending else .failedRecving)) := by
  cases ok <;> rfl

private theorem stepSend_2_first (ok : Bool) :
    numaStepSend .settledCounts 2 ok { doneSendingCounts := 0, failedRecvingCounts := 0 } =
      ({ doneSendingCounts := b2n ok, failedRecvingCounts := b2n (!ok) }, none) := by
  cases ok <;> rfl

private theorem stepSend_2_second (prev ok : Bool) :
    numaStepSend .settledCounts 2 ok
        { doneSendingCounts := b2n prev, failedRecvingCounts := b2n (!prev) } =
      ({}, some (if prev && ok then .doneSending else .failedRecving)) := by
  cases prev <;> cases ok <;> rfl

private theorem rawSend_settled (conn : Connector) (resOk : Bool) :
    rawToSendSignal conn (some (if resOk then .doneSending else .failedRecving)) =
      conn.onSend resOk := by
  cases resOk <;> rfl

private theorem stepRecv_1 (setExp ok : Bool) :
    numaStepRecv .settledCounts 1 ok
        { doneRecvingCounts := 0, failedRecvingCounts := 0,
          reqExpected := if setExp then some 1 else none } =
      ({}, some (if ok then .doneRecving else .failedRecving)) := by
  cases setExp <;> cases ok <;> rfl

private theorem stepRecv_2_first (setExp ok : Bool) :
    numaStepRecv .settledCounts 2 ok
        { doneRecvingCounts := 0, failedRecvingCounts := 0,
          reqExpected := if setExp then some 2 else none } =
      ({ doneRecvingCounts := b2n ok, failedRecvingCounts := b2n (!ok),
         reqExpected := if setExp then some 2 else none }, none) := by
  cases setExp <;> cases ok <;> rfl

private theorem stepRecv_2_second (setExp prev ok : Bool) :
    numaStepRecv .settledCounts 2 ok
        { doneRecvingCounts := b2n prev, failedRecvingCounts := b2n (!prev),
          reqExpected := if setExp then some 2 else none } =
      ({}, some (if prev && ok then .doneRecving else .failedRecving)) := by
  cases setExp <;> cases prev <;> cases ok <;> rfl

private theorem rawRecv_settled (conn : Connector) (resOk : Bool) :
    rawToRecvSignal conn (some (if resOk then .doneRecving else .failedRecving)) =
      conn.onRecv resOk := by
  cases resOk <;> rfl

private theorem numaToS_withS (twoSubs : Bool) (s : NumaS) (b : S) :
    numaToS twoSubs (numaWithS s b) =
      { b with sendPub := (numaToS twoSubs s).sendPub,
               recvPub := (numaToS twoSubs s).recvPub } := rfl

private theorem applySend_with_sendPub (sig : Signal) (a : S) (v : Option Bool) :
    applySend sig { a with sendPub := v } = { applySend sig a with sendPub := v } := by
  unfold applySend
  split <;> rfl

private theorem applyRecv_with_recvPub (pol : Policy) (sig : Signal) (a : S) (v : Option Bool) :
    applyRecv pol sig { a with recvPub := v } = { applyRecv pol sig a with recvPub := v } := by
  rcases a with ⟨_, _, _, ds, _, _, _⟩
  rcases sig with ⟨_, fr, le⟩
  cases pol <;> cases le <;> cases fr <;> cases ds <;> rfl

private theorem applySend_d (sig : Signal) (s : S) :
    (applySend sig s).dStatus = s.dStatus ∧
    (applySend sig s).dHeld = s.dHeld ∧
    (applySend sig s).dFinRecv = s.dFinRecv ∧
    (applySend sig s).dFailRecv = s.dFailRecv ∧
    (applySend sig s).recvPub = s.recvPub := by
  unfold applySend
  split <;> simp

private theorem numaInv_pStutter_helper {c : NumaCfg} {s sMid : NumaS}
    (h : NumaInv c s)
    (hts : c.twoSubs = true)
    (hap0 : allPSubsDone c.twoSubs s = false)
    (hap1 : allPSubsDone c.twoSubs sMid = false)
    (hr0 : sMid.recvSub0 = s.recvSub0)
    (hr1 : sMid.recvSub1 = s.recvSub1)
    (hdm : sMid.dMgr = s.dMgr)
    (hpm : sMid.pMgr = { doneSendingCounts := pOkCount c.twoSubs sMid,
                         failedRecvingCounts := pFailCount c.twoSubs sMid })
    (hfields : sMid.pHeld = s.pHeld ∧ sMid.dStatus = s.dStatus ∧
               sMid.dHeld = s.dHeld ∧ sMid.dFinRecv = s.dFinRecv ∧
               sMid.dFailRecv = s.dFailRecv) :
    NumaInv c sMid := by
  have had_eq : allDSubsDone c.twoSubs sMid = allDSubsDone c.twoSubs s := by
    simp [allDSubsDone, hr0, hr1]
  have haok_d : allDSubsOk c.twoSubs sMid = allDSubsOk c.twoSubs s := by
    simp [allDSubsOk, hr0, hr1]
  have htoS : numaToS c.twoSubs sMid = numaToS c.twoSubs s := by
    rcases hfields with ⟨h1, h2, h3, h4, h5⟩
    simp [numaToS, hap0, hap1, had_eq, haok_d, h1, h2, h3, h4, h5]
  refine {
    caller := by rw [htoS]; exact h.caller
    p_sub1 := by simp [hts]
    d_sub1 := by simp [hts]
    p_held := by intro _; rw [hfields.1]; exact h.p_held hap0
    d_held := by intro hd; rw [hfields.2.2.1]; exact h.d_held (by rwa [← had_eq])
    p_mgr_active := by intro _; exact hpm
    p_mgr_done := by rw [hap1]; intro hc; cases hc
    d_mgr_active := by
      intro hd
      have hd0 : allDSubsDone c.twoSubs s = false := by rwa [← had_eq]
      simpa [hdm, dOkCount, dFailCount, hr0, hr1] using h.d_mgr_active hd0
    d_mgr_done := by
      intro hd
      have hd0 : allDSubsDone c.twoSubs s = true := by rwa [← had_eq]
      simpa [hdm] using h.d_mgr_done hd0 }

private theorem numaInv_pSettled_helper {c : NumaCfg} (hc : Contract c.conn)
    {s sMid : NumaS} {resOk : Bool}
    (h : NumaInv c s)
    (hap0 : allPSubsDone c.twoSubs s = false)
    (hap1 : allPSubsDone c.twoSubs sMid = true)
    (haok : allPSubsOk c.twoSubs sMid = resOk)
    (hp1 : c.twoSubs = false → sMid.sendSub1 = none)
    (hr0 : sMid.recvSub0 = s.recvSub0)
    (hr1 : sMid.recvSub1 = s.recvSub1)
    (hdm : sMid.dMgr = s.dMgr)
    (hpm : sMid.pMgr = {}) :
    NumaInv c (numaWithS sMid (applySend (c.conn.onSend resOk) (numaToS c.twoSubs s))) := by
  have had_eq : allDSubsDone c.twoSubs sMid = allDSubsDone c.twoSubs s := by
    simp [allDSubsDone, hr0, hr1]
  have haok_d : allDSubsOk c.twoSubs sMid = allDSubsOk c.twoSubs s := by
    simp [allDSubsOk, hr0, hr1]
  have hstep : pOutcome ⟨c.conn, c.policy⟩ resOk (numaToS c.twoSubs s) =
      some (numaToS c.twoSubs (numaWithS sMid
        (applySend (c.conn.onSend resOk) (numaToS c.twoSubs s)))) := by
    have hnone : (numaToS c.twoSubs s).sendPub = none := by simp [numaToS, hap0]
    have hsp_mid : (numaToS c.twoSubs sMid).sendPub = some resOk := by
      simp [numaToS, hap1, haok]
    have hrp_mid : (numaToS c.twoSubs sMid).recvPub = (numaToS c.twoSubs s).recvPub := by
      simp [numaToS, had_eq, haok_d]
    rw [pOutcome, if_pos hnone, numaToS_withS, hsp_mid, hrp_mid, applySend_with_sendPub]
    have hrp_apply : (applySend (c.conn.onSend resOk) (numaToS c.twoSubs s)).recvPub =
        (numaToS c.twoSubs s).recvPub :=
      (applySend_d (c.conn.onSend resOk) (numaToS c.twoSubs s)).2.2.2.2
    rw [← hrp_apply]
  have hci' := inv_pOutcome (c := ⟨c.conn, c.policy⟩) hc h.caller hstep
  refine {
    caller := hci'
    p_sub1 := by intro hts; simpa [numaWithS] using hp1 hts
    d_sub1 := by intro hts; rw [numaWithS, hr1]; exact h.d_sub1 hts
    p_held := by
      intro hp
      have hp' : allPSubsDone c.twoSubs sMid = false := by
        simpa [numaWithS, allPSubsDone] using hp
      rw [hap1] at hp'; cases hp'
    d_held := by
      intro hd
      have hd_s : allDSubsDone c.twoSubs s = false := by
        simpa [numaWithS, allDSubsDone, hr0, hr1] using hd
      have hdh := h.d_held hd_s
      have hds := (applySend_d (c.conn.onSend resOk) (numaToS c.twoSubs s)).2.1
      simpa [numaWithS, numaToS, hdh] using hds
    p_mgr_active := by
      intro hp
      have hp' : allPSubsDone c.twoSubs sMid = false := by
        simpa [numaWithS, allPSubsDone] using hp
      rw [hap1] at hp'; cases hp'
    p_mgr_done := by intro _; simpa [numaWithS] using hpm
    d_mgr_active := by
      intro hd
      have hd_s : allDSubsDone c.twoSubs s = false := by
        simpa [numaWithS, allDSubsDone, hr0, hr1] using hd
      simpa [numaWithS, hdm, dOkCount, dFailCount, hr0, hr1] using h.d_mgr_active hd_s
    d_mgr_done := by
      intro hd
      have hd_s : allDSubsDone c.twoSubs s = true := by
        simpa [numaWithS, allDSubsDone, hr0, hr1] using hd
      simpa [numaWithS, hdm] using h.d_mgr_done hd_s }

private theorem numaInv_pSub {c : NumaCfg} (hpol : c.numaPolicy = .settledCounts)
    (hc : Contract c.conn) {s s' : NumaS} {sub ok : Bool}
    (h : NumaInv c s) (hs : numaPSub c sub ok s = some s') : NumaInv c s' := by
  unfold numaPSub at hs
  rw [hpol] at hs
  cases hts : c.twoSubs <;> cases sub
  · -- twoSubs = false, sub = false
    cases hs0 : s.sendSub0 with
    | some _ => simp [hts, hs0] at hs
    | none =>
      have hap0 : allPSubsDone c.twoSubs s = false := by simp [hts, allPSubsDone, hs0]
      have hpm : s.pMgr = { doneSendingCounts := 0, failedRecvingCounts := 0 } := by
        simpa [hts, pOkCount, pFailCount, hs0, b2n] using h.p_mgr_active hap0
      have hs1 : s.sendSub1 = none := h.p_sub1 hts
      have heval : numaPSub c false ok s =
          some (numaWithS { s with sendSub0 := some ok, pMgr := {} }
            (applySend (c.conn.onSend ok) (numaToS c.twoSubs s))) := by
        unfold numaPSub
        simp [hpol, hts, hs0, hpm, numaCount, stepSend_1, rawSend_settled]
      unfold numaPSub at heval
      rw [hpol] at heval
      rw [heval] at hs
      injection hs with hs
      rw [← hs]
      exact numaInv_pSettled_helper hc h hap0
        (by simp [hts, allPSubsDone])
        (by cases ok <;> simp [hts, allPSubsOk])
        (by intro _; exact hs1) rfl rfl rfl rfl
  · -- twoSubs = false, sub = true (disabled)
    simp [hts] at hs
  · -- twoSubs = true, sub = false
    cases hs0 : s.sendSub0 with
    | some _ => simp [hts, hs0] at hs
    | none =>
      have hap0 : allPSubsDone c.twoSubs s = false := by simp [hts, allPSubsDone, hs0]
      cases hs1 : s.sendSub1 with
      | none =>
        have hpm : s.pMgr = { doneSendingCounts := 0, failedRecvingCounts := 0 } := by
          simpa [hts, pOkCount, pFailCount, hs0, hs1, b2n] using h.p_mgr_active hap0
        have heval : (if Bool.false && !c.twoSubs then none else
            let cur := if Bool.false then s.sendSub1 else s.sendSub0
            if cur.isNone then
              let x := numaStepSend .settledCounts (numaCount c.twoSubs) ok s.pMgr
              let sig := rawToSendSignal c.conn x.2
              let base' := applySend sig (numaToS c.twoSubs s)
              let s' := if Bool.false then { s with sendSub1 := some ok, pMgr := x.1 }
                        else { s with sendSub0 := some ok, pMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some { s with sendSub0 := some ok,
                          pMgr := { doneSendingCounts := b2n ok,
                                    failedRecvingCounts := b2n (!ok) } } := by
          simp [hts, hs0, hpm, numaCount, stepSend_2_first, rawToSendSignal, applySend,
            numaWithS, numaToS]
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_pStutter_helper h hts hap0
          (by simp [hts, allPSubsDone, hs1]) rfl rfl rfl
          (by cases ok <;> simp [hts, pOkCount, pFailCount, hs1, b2n])
          ⟨rfl, rfl, rfl, rfl, rfl⟩
      | some ok1 =>
        have hpm : s.pMgr = { doneSendingCounts := b2n ok1,
                              failedRecvingCounts := b2n (!ok1) } := by
          simpa [hts, pOkCount, pFailCount, hs0, hs1, b2n] using h.p_mgr_active hap0
        have heval : (if Bool.false && !c.twoSubs then none else
            let cur := if Bool.false then s.sendSub1 else s.sendSub0
            if cur.isNone then
              let x := numaStepSend .settledCounts (numaCount c.twoSubs) ok s.pMgr
              let sig := rawToSendSignal c.conn x.2
              let base' := applySend sig (numaToS c.twoSubs s)
              let s' := if Bool.false then { s with sendSub1 := some ok, pMgr := x.1 }
                        else { s with sendSub0 := some ok, pMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some (numaWithS { s with sendSub0 := some ok, pMgr := {} }
              (applySend (c.conn.onSend (ok1 && ok)) (numaToS c.twoSubs s))) := by
          simp only [hts, hs0, hpm, numaCount, Bool.false_and, ↓reduceIte]
          rw [stepSend_2_second, rawSend_settled]
          rfl
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_pSettled_helper hc h hap0
          (by simp [hts, allPSubsDone, hs1])
          (by cases ok <;> cases ok1 <;> simp [hts, allPSubsOk, hs1])
          (by simp [hts]) rfl rfl rfl rfl
  · -- twoSubs = true, sub = true
    cases hs1 : s.sendSub1 with
    | some _ => simp [hts, hs1] at hs
    | none =>
      have hap0 : allPSubsDone c.twoSubs s = false := by simp [hts, allPSubsDone, hs1]
      cases hs0 : s.sendSub0 with
      | none =>
        have hpm : s.pMgr = { doneSendingCounts := 0, failedRecvingCounts := 0 } := by
          simpa [hts, pOkCount, pFailCount, hs0, hs1, b2n] using h.p_mgr_active hap0
        have heval : (if Bool.true && !c.twoSubs then none else
            let cur := if Bool.true then s.sendSub1 else s.sendSub0
            if cur.isNone then
              let x := numaStepSend .settledCounts (numaCount c.twoSubs) ok s.pMgr
              let sig := rawToSendSignal c.conn x.2
              let base' := applySend sig (numaToS c.twoSubs s)
              let s' := if Bool.true then { s with sendSub1 := some ok, pMgr := x.1 }
                        else { s with sendSub0 := some ok, pMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some { s with sendSub1 := some ok,
                          pMgr := { doneSendingCounts := b2n ok,
                                    failedRecvingCounts := b2n (!ok) } } := by
          simp [hts, hs1, hpm, numaCount, stepSend_2_first, rawToSendSignal, applySend,
            numaWithS, numaToS]
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_pStutter_helper h hts hap0
          (by simp [hts, allPSubsDone, hs0]) rfl rfl rfl
          (by cases ok <;> simp [hts, pOkCount, pFailCount, hs0, b2n])
          ⟨rfl, rfl, rfl, rfl, rfl⟩
      | some ok0 =>
        have hpm : s.pMgr = { doneSendingCounts := b2n ok0,
                              failedRecvingCounts := b2n (!ok0) } := by
          simpa [hts, pOkCount, pFailCount, hs0, hs1, b2n] using h.p_mgr_active hap0
        have heval : (if Bool.true && !c.twoSubs then none else
            let cur := if Bool.true then s.sendSub1 else s.sendSub0
            if cur.isNone then
              let x := numaStepSend .settledCounts (numaCount c.twoSubs) ok s.pMgr
              let sig := rawToSendSignal c.conn x.2
              let base' := applySend sig (numaToS c.twoSubs s)
              let s' := if Bool.true then { s with sendSub1 := some ok, pMgr := x.1 }
                        else { s with sendSub0 := some ok, pMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some (numaWithS { s with sendSub1 := some ok, pMgr := {} }
              (applySend (c.conn.onSend (ok0 && ok)) (numaToS c.twoSubs s))) := by
          simp only [hts, hs1, hpm, numaCount, Bool.not_true, Bool.true_and, ↓reduceIte]
          rw [stepSend_2_second, rawSend_settled]
          rfl
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_pSettled_helper hc h hap0
          (by simp [hts, allPSubsDone, hs0])
          (by cases ok0 <;> cases ok <;> simp [hts, allPSubsOk, hs0])
          (by simp [hts]) rfl rfl rfl rfl

private theorem numaInv_dStutter_helper {c : NumaCfg} {s sMid : NumaS}
    (h : NumaInv c s)
    (hts : c.twoSubs = true)
    (had0 : allDSubsDone c.twoSubs s = false)
    (had1 : allDSubsDone c.twoSubs sMid = false)
    (hp0 : sMid.sendSub0 = s.sendSub0)
    (hp1 : sMid.sendSub1 = s.sendSub1)
    (hpm : sMid.pMgr = s.pMgr)
    (hdm : sMid.dMgr = { doneRecvingCounts := dOkCount c.twoSubs sMid,
                         failedRecvingCounts := dFailCount c.twoSubs sMid,
                         reqExpected := numaInitDExpected c })
    (hfields : sMid.pHeld = s.pHeld ∧ sMid.dStatus = s.dStatus ∧
               sMid.dHeld = s.dHeld ∧ sMid.dFinRecv = s.dFinRecv ∧
               sMid.dFailRecv = s.dFailRecv) :
    NumaInv c sMid := by
  have hap_eq : allPSubsDone c.twoSubs sMid = allPSubsDone c.twoSubs s := by
    simp [allPSubsDone, hp0, hp1]
  have haok_p : allPSubsOk c.twoSubs sMid = allPSubsOk c.twoSubs s := by
    simp [allPSubsOk, hp0, hp1]
  have htoS : numaToS c.twoSubs sMid = numaToS c.twoSubs s := by
    rcases hfields with ⟨h1, h2, h3, h4, h5⟩
    simp [numaToS, had0, had1, hap_eq, haok_p, h1, h2, h3, h4, h5]
  refine {
    caller := by rw [htoS]; exact h.caller
    p_sub1 := by simp [hts]
    d_sub1 := by simp [hts]
    p_held := by intro hp; rw [hfields.1]; exact h.p_held (by rwa [← hap_eq])
    d_held := by intro _; rw [hfields.2.2.1]; exact h.d_held had0
    p_mgr_active := by
      intro hp
      have hp_s : allPSubsDone c.twoSubs s = false := by rwa [← hap_eq]
      simpa [hpm, pOkCount, pFailCount, hp0, hp1] using h.p_mgr_active hp_s
    p_mgr_done := by
      intro hp
      have hp_s : allPSubsDone c.twoSubs s = true := by rwa [← hap_eq]
      simpa [hpm] using h.p_mgr_done hp_s
    d_mgr_active := by intro _; exact hdm
    d_mgr_done := by rw [had1]; intro hc; cases hc }

private theorem numaInv_dSettled_helper {c : NumaCfg} (hc : Contract c.conn)
    {s sMid : NumaS} {resOk : Bool}
    (h : NumaInv c s)
    (had0 : allDSubsDone c.twoSubs s = false)
    (had1 : allDSubsDone c.twoSubs sMid = true)
    (haok : allDSubsOk c.twoSubs sMid = resOk)
    (hd1 : c.twoSubs = false → sMid.recvSub1 = none)
    (hp0 : sMid.sendSub0 = s.sendSub0)
    (hp1 : sMid.sendSub1 = s.sendSub1)
    (hpm : sMid.pMgr = s.pMgr)
    (hdm : sMid.dMgr = {}) :
    NumaInv c (numaWithS sMid (applyRecv c.policy (c.conn.onRecv resOk) (numaToS c.twoSubs s))) := by
  have hap_eq : allPSubsDone c.twoSubs sMid = allPSubsDone c.twoSubs s := by
    simp [allPSubsDone, hp0, hp1]
  have haok_p : allPSubsOk c.twoSubs sMid = allPSubsOk c.twoSubs s := by
    simp [allPSubsOk, hp0, hp1]
  have hstep : dOutcome ⟨c.conn, c.policy⟩ resOk (numaToS c.twoSubs s) =
      some (numaToS c.twoSubs (numaWithS sMid
        (applyRecv c.policy (c.conn.onRecv resOk) (numaToS c.twoSubs s)))) := by
    have hnone : (numaToS c.twoSubs s).recvPub = none := by simp [numaToS, had0]
    have hrp_mid : (numaToS c.twoSubs sMid).recvPub = some resOk := by
      simp [numaToS, had1, haok]
    have hsp_mid : (numaToS c.twoSubs sMid).sendPub = (numaToS c.twoSubs s).sendPub := by
      simp [numaToS, hap_eq, haok_p]
    rw [dOutcome, if_pos hnone, numaToS_withS, hrp_mid, hsp_mid, applyRecv_with_recvPub]
    have hsp_apply : (applyRecv c.policy (c.conn.onRecv resOk) (numaToS c.twoSubs s)).sendPub =
        (numaToS c.twoSubs s).sendPub :=
      (applyRecv_p c.policy (c.conn.onRecv resOk) (numaToS c.twoSubs s)).2
    rw [← hsp_apply]
  have hci' := inv_dOutcome (c := ⟨c.conn, c.policy⟩) hc h.caller hstep
  refine {
    caller := hci'
    p_sub1 := by intro hts; rw [numaWithS, hp1]; exact h.p_sub1 hts
    d_sub1 := by intro hts; simpa [numaWithS] using hd1 hts
    p_held := by
      intro hp
      have hp_s : allPSubsDone c.twoSubs s = false := by
        simpa [numaWithS, allPSubsDone, hp0, hp1] using hp
      have hph := h.p_held hp_s
      have hpr := (applyRecv_p c.policy (c.conn.onRecv resOk) (numaToS c.twoSubs s)).1
      simpa [numaWithS, numaToS, hph] using hpr
    d_held := by
      intro hd
      have hd' : allDSubsDone c.twoSubs sMid = false := by
        simpa [numaWithS, allDSubsDone] using hd
      rw [had1] at hd'; cases hd'
    p_mgr_active := by
      intro hp
      have hp_s : allPSubsDone c.twoSubs s = false := by
        simpa [numaWithS, allPSubsDone, hp0, hp1] using hp
      simpa [numaWithS, hpm, pOkCount, pFailCount, hp0, hp1] using h.p_mgr_active hp_s
    p_mgr_done := by
      intro hp
      have hp_s : allPSubsDone c.twoSubs s = true := by
        simpa [numaWithS, allPSubsDone, hp0, hp1] using hp
      simpa [numaWithS, hpm] using h.p_mgr_done hp_s
    d_mgr_active := by
      intro hd
      have hd' : allDSubsDone c.twoSubs sMid = false := by
        simpa [numaWithS, allDSubsDone] using hd
      rw [had1] at hd'; cases hd'
    d_mgr_done := by intro _; simpa [numaWithS] using hdm }

private theorem numaInv_dSub {c : NumaCfg} (hpol : c.numaPolicy = .settledCounts)
    (hc : Contract c.conn) {s s' : NumaS} {sub ok : Bool}
    (h : NumaInv c s) (hs : numaDSub c sub ok s = some s') : NumaInv c s' := by
  unfold numaDSub at hs
  rw [hpol] at hs
  cases hts : c.twoSubs <;> cases sub
  · -- twoSubs = false, sub = false
    cases hr0 : s.recvSub0 with
    | some _ => simp [hts, hr0] at hs
    | none =>
      have had0 : allDSubsDone c.twoSubs s = false := by simp [hts, allDSubsDone, hr0]
      have hdm : s.dMgr = { doneRecvingCounts := 0, failedRecvingCounts := 0,
                            reqExpected := if c.setExpected then some 1 else none } := by
        simpa [hts, dOkCount, dFailCount, hr0, b2n, numaInitDExpected, numaCount]
          using h.d_mgr_active had0
      have hr1 : s.recvSub1 = none := h.d_sub1 hts
      have heval : (if Bool.false && !c.twoSubs then none else
          let cur := if Bool.false then s.recvSub1 else s.recvSub0
          if cur.isNone then
            let x := numaStepRecv .settledCounts (numaCount c.twoSubs) ok s.dMgr
            let sig := rawToRecvSignal c.conn x.2
            let base' := applyRecv c.policy sig (numaToS c.twoSubs s)
            let s' := if Bool.false then { s with recvSub1 := some ok, dMgr := x.1 }
                      else { s with recvSub0 := some ok, dMgr := x.1 }
            some (numaWithS s' base')
          else none) =
          some (numaWithS { s with recvSub0 := some ok, dMgr := {} }
            (applyRecv c.policy (c.conn.onRecv ok) (numaToS c.twoSubs s))) := by
        simp [hts, hr0, hdm, numaCount, stepRecv_1, rawRecv_settled]
      rw [heval] at hs
      injection hs with hs
      rw [← hs]
      exact numaInv_dSettled_helper hc h had0
        (by simp [hts, allDSubsDone])
        (by cases ok <;> simp [hts, allDSubsOk])
        (by intro _; exact hr1) rfl rfl rfl rfl
  · -- twoSubs = false, sub = true (disabled)
    simp [hts] at hs
  · -- twoSubs = true, sub = false
    cases hr0 : s.recvSub0 with
    | some _ => simp [hts, hr0] at hs
    | none =>
      have had0 : allDSubsDone c.twoSubs s = false := by simp [hts, allDSubsDone, hr0]
      cases hr1 : s.recvSub1 with
      | none =>
        have hdm : s.dMgr = { doneRecvingCounts := 0, failedRecvingCounts := 0,
                              reqExpected := if c.setExpected then some 2 else none } := by
          simpa [hts, dOkCount, dFailCount, hr0, hr1, b2n, numaInitDExpected, numaCount]
            using h.d_mgr_active had0
        have heval : (if Bool.false && !c.twoSubs then none else
            let cur := if Bool.false then s.recvSub1 else s.recvSub0
            if cur.isNone then
              let x := numaStepRecv .settledCounts (numaCount c.twoSubs) ok s.dMgr
              let sig := rawToRecvSignal c.conn x.2
              let base' := applyRecv c.policy sig (numaToS c.twoSubs s)
              let s' := if Bool.false then { s with recvSub1 := some ok, dMgr := x.1 }
                        else { s with recvSub0 := some ok, dMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some { s with recvSub0 := some ok,
                          dMgr := { doneRecvingCounts := b2n ok,
                                    failedRecvingCounts := b2n (!ok),
                                    reqExpected := if c.setExpected then some 2 else none } } := by
          simp [hts, hr0, hdm, numaCount, stepRecv_2_first, rawToRecvSignal, applyRecv,
            numaWithS, numaToS]
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_dStutter_helper h hts had0
          (by simp [hts, allDSubsDone, hr1]) rfl rfl rfl
          (by cases ok <;> simp [hts, dOkCount, dFailCount, hr1, b2n, numaInitDExpected, numaCount])
          ⟨rfl, rfl, rfl, rfl, rfl⟩
      | some ok1 =>
        have hdm : s.dMgr = { doneRecvingCounts := b2n ok1, failedRecvingCounts := b2n (!ok1),
                              reqExpected := if c.setExpected then some 2 else none } := by
          simpa [hts, dOkCount, dFailCount, hr0, hr1, b2n, numaInitDExpected, numaCount]
            using h.d_mgr_active had0
        have heval : (if Bool.false && !c.twoSubs then none else
            let cur := if Bool.false then s.recvSub1 else s.recvSub0
            if cur.isNone then
              let x := numaStepRecv .settledCounts (numaCount c.twoSubs) ok s.dMgr
              let sig := rawToRecvSignal c.conn x.2
              let base' := applyRecv c.policy sig (numaToS c.twoSubs s)
              let s' := if Bool.false then { s with recvSub1 := some ok, dMgr := x.1 }
                        else { s with recvSub0 := some ok, dMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some (numaWithS { s with recvSub0 := some ok, dMgr := {} }
              (applyRecv c.policy (c.conn.onRecv (ok1 && ok)) (numaToS c.twoSubs s))) := by
          simp only [hts, hr0, hdm, numaCount, Bool.false_and, ↓reduceIte]
          rw [stepRecv_2_second, rawRecv_settled]
          rfl
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_dSettled_helper hc h had0
          (by simp [hts, allDSubsDone, hr1])
          (by cases ok <;> cases ok1 <;> simp [hts, allDSubsOk, hr1])
          (by simp [hts]) rfl rfl rfl rfl
  · -- twoSubs = true, sub = true
    cases hr1 : s.recvSub1 with
    | some _ => simp [hts, hr1] at hs
    | none =>
      have had0 : allDSubsDone c.twoSubs s = false := by simp [hts, allDSubsDone, hr1]
      cases hr0 : s.recvSub0 with
      | none =>
        have hdm : s.dMgr = { doneRecvingCounts := 0, failedRecvingCounts := 0,
                              reqExpected := if c.setExpected then some 2 else none } := by
          simpa [hts, dOkCount, dFailCount, hr0, hr1, b2n, numaInitDExpected, numaCount]
            using h.d_mgr_active had0
        have heval : (if Bool.true && !c.twoSubs then none else
            let cur := if Bool.true then s.recvSub1 else s.recvSub0
            if cur.isNone then
              let x := numaStepRecv .settledCounts (numaCount c.twoSubs) ok s.dMgr
              let sig := rawToRecvSignal c.conn x.2
              let base' := applyRecv c.policy sig (numaToS c.twoSubs s)
              let s' := if Bool.true then { s with recvSub1 := some ok, dMgr := x.1 }
                        else { s with recvSub0 := some ok, dMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some { s with recvSub1 := some ok,
                          dMgr := { doneRecvingCounts := b2n ok,
                                    failedRecvingCounts := b2n (!ok),
                                    reqExpected := if c.setExpected then some 2 else none } } := by
          simp [hts, hr1, hdm, numaCount, stepRecv_2_first, rawToRecvSignal, applyRecv,
            numaWithS, numaToS]
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_dStutter_helper h hts had0
          (by simp [hts, allDSubsDone, hr0]) rfl rfl rfl
          (by cases ok <;> simp [hts, dOkCount, dFailCount, hr0, b2n, numaInitDExpected, numaCount])
          ⟨rfl, rfl, rfl, rfl, rfl⟩
      | some ok0 =>
        have hdm : s.dMgr = { doneRecvingCounts := b2n ok0, failedRecvingCounts := b2n (!ok0),
                              reqExpected := if c.setExpected then some 2 else none } := by
          simpa [hts, dOkCount, dFailCount, hr0, hr1, b2n, numaInitDExpected, numaCount]
            using h.d_mgr_active had0
        have heval : (if Bool.true && !c.twoSubs then none else
            let cur := if Bool.true then s.recvSub1 else s.recvSub0
            if cur.isNone then
              let x := numaStepRecv .settledCounts (numaCount c.twoSubs) ok s.dMgr
              let sig := rawToRecvSignal c.conn x.2
              let base' := applyRecv c.policy sig (numaToS c.twoSubs s)
              let s' := if Bool.true then { s with recvSub1 := some ok, dMgr := x.1 }
                        else { s with recvSub0 := some ok, dMgr := x.1 }
              some (numaWithS s' base')
            else none) =
            some (numaWithS { s with recvSub1 := some ok, dMgr := {} }
              (applyRecv c.policy (c.conn.onRecv (ok0 && ok)) (numaToS c.twoSubs s))) := by
          simp only [hts, hr1, hdm, numaCount, Bool.not_true, Bool.true_and, ↓reduceIte]
          rw [stepRecv_2_second, rawRecv_settled]
          rfl
        rw [heval] at hs
        injection hs with hs
        rw [← hs]
        exact numaInv_dSettled_helper hc h had0
          (by simp [hts, allDSubsDone, hr0])
          (by cases ok0 <;> cases ok <;> simp [hts, allDSubsOk, hr0])
          (by simp [hts]) rfl rfl rfl rfl

private theorem numaInv_promote {c : NumaCfg} {s s' : NumaS}
    (h : NumaInv c s) (hs : numaPromote c.twoSubs s = some s') : NumaInv c s' := by
  unfold numaPromote at hs
  cases hp : promote (numaToS c.twoSubs s) with
  | none => simp [hp] at hs
  | some b =>
    simp only [hp, Option.map_some, Option.some.injEq] at hs
    subst hs
    have hci' := inv_promote h.caller hp
    have had_true : allDSubsDone c.twoSubs s = true := by
      cases hd : allDSubsDone c.twoSubs s
      · have hnone : (numaToS c.twoSubs s).recvPub = none := by simp [numaToS, hd]
        have hfr : s.dFinRecv = false := (h.caller.d_none hnone).2.1
        simp [promote, numaToS, hfr] at hp
      · rfl
    have hpres : b.pHeld = s.pHeld ∧
        b.sendPub = (numaToS c.twoSubs s).sendPub ∧
        b.recvPub = (numaToS c.twoSubs s).recvPub := by
      unfold promote at hp
      split at hp
      · split at hp <;> simp only [Option.some.injEq] at hp <;> subst hp <;> exact ⟨rfl, rfl, rfl⟩
      · simp at hp
    have htoS : numaToS c.twoSubs (numaWithS s b) = b := by
      rw [numaToS_withS, ← hpres.2.1, ← hpres.2.2]
    refine {
      caller := by rw [htoS]; exact hci'
      p_sub1 := by intro hts; simpa [numaWithS] using h.p_sub1 hts
      d_sub1 := by intro hts; simpa [numaWithS] using h.d_sub1 hts
      p_held := by
        intro hpa
        have hpa0 : allPSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allPSubsDone] using hpa
        simpa [numaWithS, hpres.1] using h.p_held hpa0
      d_held := by
        intro hda
        have hda0 : allDSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allDSubsDone] using hda
        rw [had_true] at hda0; cases hda0
      p_mgr_active := by
        intro hpa
        have hpa0 : allPSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allPSubsDone] using hpa
        simpa [numaWithS, pOkCount, pFailCount] using h.p_mgr_active hpa0
      p_mgr_done := by
        intro hpa
        have hpa0 : allPSubsDone c.twoSubs s = true := by
          simpa [numaWithS, allPSubsDone] using hpa
        simpa [numaWithS] using h.p_mgr_done hpa0
      d_mgr_active := by
        intro hda
        have hda0 : allDSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allDSubsDone] using hda
        rw [had_true] at hda0; cases hda0
      d_mgr_done := by intro _; simpa [numaWithS] using h.d_mgr_done had_true }

private theorem numaInv_finish {c : NumaCfg} {s s' : NumaS}
    (h : NumaInv c s) (hs : numaFinish c.twoSubs s = some s') : NumaInv c s' := by
  unfold numaFinish at hs
  cases hf : finish (numaToS c.twoSubs s) with
  | none => simp [hf] at hs
  | some b =>
    simp only [hf, Option.map_some, Option.some.injEq] at hs
    subst hs
    have hci' := inv_finish h.caller hf
    have hpres : b.pHeld = s.pHeld ∧
        b.sendPub = (numaToS c.twoSubs s).sendPub ∧
        b.recvPub = (numaToS c.twoSubs s).recvPub := by
      unfold finish at hf
      split at hf
      · simp only [Option.some.injEq] at hf; subst hf; exact ⟨rfl, rfl, rfl⟩
      · simp only [Option.some.injEq] at hf; subst hf; exact ⟨rfl, rfl, rfl⟩
      · simp only [Option.some.injEq] at hf; subst hf; exact ⟨rfl, rfl, rfl⟩
      · simp at hf
    have htoS : numaToS c.twoSubs (numaWithS s b) = b := by
      rw [numaToS_withS, ← hpres.2.1, ← hpres.2.2]
    refine {
      caller := by rw [htoS]; exact hci'
      p_sub1 := by intro hts; simpa [numaWithS] using h.p_sub1 hts
      d_sub1 := by intro hts; simpa [numaWithS] using h.d_sub1 hts
      p_held := by
        intro hpa
        have hpa0 : allPSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allPSubsDone] using hpa
        simpa [numaWithS, hpres.1] using h.p_held hpa0
      d_held := by
        intro hda
        have hda0 : allDSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allDSubsDone] using hda
        have hnone : (numaToS c.twoSubs s).recvPub = none := by simp [numaToS, hda0]
        obtain ⟨hst, hfr, _⟩ := h.caller.d_none hnone
        have hdh0 := h.d_held hda0
        cases hst with
        | inl hw =>
          change s.dStatus = .waitingRemoteKv at hw
          change s.dFinRecv = false at hfr
          simp [finish, numaToS, hw, hdh0, hfr] at hf
          subst hf; rfl
        | inr hfin =>
          change s.dStatus = .finished at hfin
          simp [finish, numaToS, hfin] at hf
      p_mgr_active := by
        intro hpa
        have hpa0 : allPSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allPSubsDone] using hpa
        simpa [numaWithS, pOkCount, pFailCount] using h.p_mgr_active hpa0
      p_mgr_done := by
        intro hpa
        have hpa0 : allPSubsDone c.twoSubs s = true := by
          simpa [numaWithS, allPSubsDone] using hpa
        simpa [numaWithS] using h.p_mgr_done hpa0
      d_mgr_active := by
        intro hda
        have hda0 : allDSubsDone c.twoSubs s = false := by
          simpa [numaWithS, allDSubsDone] using hda
        simpa [numaWithS, dOkCount, dFailCount] using h.d_mgr_active hda0
      d_mgr_done := by
        intro hda
        have hda0 : allDSubsDone c.twoSubs s = true := by
          simpa [numaWithS, allDSubsDone] using hda
        simpa [numaWithS] using h.d_mgr_done hda0 }

private theorem numaInv_recycle {c : NumaCfg} {s s' : NumaS}
    (h : NumaInv c s) (hs : numaRecycle c s = some s') : NumaInv c s' := by
  unfold numaRecycle at hs
  split at hs
  · rename_i hcond
    simp only [Bool.and_eq_true] at hcond
    obtain ⟨⟨⟨hap, had⟩, _⟩, _⟩ := hcond
    have hpm : s.pMgr = {} := h.p_mgr_done hap
    have hdm : s.dMgr = {} := h.d_mgr_done had
    simp only [Option.some.injEq] at hs
    have heq : s' = numaInit c := by
      rw [← hs, hpm, hdm]
      cases hse : c.setExpected <;> simp [numaInit, numaInitDExpected, hse]
    rw [heq]
    exact numaInv_init c
  · simp at hs

theorem numaStep_inv {c : NumaCfg} (hpol : c.numaPolicy = .settledCounts)
    (hc : Contract c.conn) {s s' : NumaS} {e : NumaEv}
    (h : NumaInv c s) (hs : numaStep c s e = some s') : NumaInv c s' := by
  cases e with
  | pSub sub ok => exact numaInv_pSub hpol hc h hs
  | dSub sub ok => exact numaInv_dSub hpol hc h hs
  | promote => exact numaInv_promote h hs
  | finish => exact numaInv_finish h hs
  | recycle => exact numaInv_recycle h hs

theorem numa_settledCounts_inv {c : NumaCfg} (hpol : c.numaPolicy = .settledCounts)
    (hc : Contract c.conn) {s : NumaS} (hr : (numaSys c).Reachable s) :
    NumaInv c s :=
  (numaSys c).reachable_induction (numaInv_init c)
    (fun _ _ _ h hs => numaStep_inv hpol hc h hs) hr

/-- Under `NumaPolicy.settledCounts` and any connector meeting `Contract`, across
unbounded `recycle` (`req_id` reuse) rounds, for both 1 and 2 sub-managers, both
failure policies, and either `StartRead` overload (`setExpected`), no reachable
state ever frees HBM while a sub-manager is still unpublished, leaks blocks on P
or D, or decodes unwritten KV. -/
theorem numa_settledCounts_safe {c : NumaCfg} (hpol : c.numaPolicy = .settledCounts)
    (hc : Contract c.conn) {s : NumaS} (hr : (numaSys c).Reachable s) :
    NumaSafe c.twoSubs s :=
  numaInv_safe (numa_settledCounts_inv hpol hc hr)

/-- Under `NumaPolicy.settledCounts` and `Contract`, once all participating
sub-managers on P (resp. D) have settled, all per-request map entries in `pMgr`
(resp. `dMgr`) are erased (`{}`). -/
theorem numa_settledCounts_maps_clean {c : NumaCfg}
    (hpol : c.numaPolicy = .settledCounts) (hc : Contract c.conn) {s : NumaS}
    (hr : (numaSys c).Reachable s) :
    (allPSubsDone c.twoSubs s = true → s.pMgr = {}) ∧
    (allDSubsDone c.twoSubs s = true → s.dMgr = {}) := by
  have h := numa_settledCounts_inv hpol hc hr
  exact ⟨h.p_mgr_done, h.d_mgr_done⟩

/-! ## Exhaustive state-space searches and concrete traces -/

def numaAllEvents : List NumaEv :=
  [ .pSub false true, .pSub false false,
    .pSub true true, .pSub true false,
    .dSub false true, .dSub false false,
    .dSub true true, .dSub true false,
    .promote, .finish, .recycle ]

def numaSearch (c : NumaCfg) (events : List NumaEv := numaAllEvents)
    (fuel : Nat := 24) : ModelCheck.Result NumaEv :=
  ModelCheck.check (numaSys c) events (numaViolates c.twoSubs) fuel

def numaHasCounterexample (c : NumaCfg) : Bool :=
  match numaSearch c with
  | .counterexample _ => true
  | _ => false

def numaView (twoSubs : Bool) (s : NumaS) : View := view (numaToS twoSubs s)

-- `NumaPolicy.settledCounts` exhausts the cyclic multi-round state space (`.safe`)
-- across both 1 and 2 sub-managers, both failure policies, and both `StartRead` overloads:
#guard numaSearch ⟨connectorTwoChannel, .fail, .settledCounts, true, true⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .recompute, .settledCounts, true, true⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .fail, .settledCounts, true, false⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .recompute, .settledCounts, true, false⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .fail, .settledCounts, false, true⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .recompute, .settledCounts, false, true⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .fail, .settledCounts, false, false⟩ = .safe
#guard numaSearch ⟨connectorTwoChannel, .recompute, .settledCounts, false, false⟩ = .safe

-- `NumaPolicy.eagerFailSet` finds a counterexample both at `num_subs = 2`
-- (`ENABLE_MULTI_NUMA=1`) and — via `recycle` — at `num_subs = 1` (`ENABLE_MULTI_NUMA=0`),
-- under either `StartRead` overload:
#guard numaHasCounterexample ⟨connectorTwoChannel, .fail, .eagerFailSet, true, true⟩ = true
#guard numaHasCounterexample ⟨connectorTwoChannel, .recompute, .eagerFailSet, true, true⟩ = true
#guard numaHasCounterexample ⟨connectorTwoChannel, .fail, .eagerFailSet, false, true⟩ = true
#guard numaHasCounterexample ⟨connectorTwoChannel, .recompute, .eagerFailSet, false, true⟩ = true
#guard numaHasCounterexample ⟨connectorTwoChannel, .fail, .eagerFailSet, true, false⟩ = true
#guard numaHasCounterexample ⟨connectorTwoChannel, .fail, .eagerFailSet, false, false⟩ = true

/-! ### `NumaPolicy.settledCounts` traces -/

/-- **Settled-count barrier on D (`twoSubs = true`): sibling H2D in flight keeps
`dHeld = true`, and subsequent `req_id` reuse succeeds cleanly.**
Sub-manager 0 fails while sub-manager 1 is still running H2D (`dHeld` stays
`true`, `dMgr.failedRecvingCounts = 1`). When sub-manager 1 finishes (`ok = true`),
`CompleteReadRaw` emits `failed_recving`, D frees `dHeld = false`, and `dMgr` is
cleared to `{}`. After P settles and `recycle` reuses `req_id`, a healthy receive
across both sub-managers promotes to `.decoding`. -/
theorem trace_settledCounts_recv_fail_waits_for_sibling :
    let c : NumaCfg := ⟨connectorTwoChannel, .fail, .settledCounts, true, true⟩
    ((numaSys c).run [.dSub false false]).map
      (fun s => (s.dMgr.failedRecvingCounts, numaView true s, numaEarlyFree true s)) =
      some (1, ⟨true, .waitingRemoteKv, true, false, false⟩, false) ∧
    ((numaSys c).run [.dSub false false, .dSub true true]).map
      (fun s => (s.dMgr, numaView true s)) =
      some ({}, ⟨true, .finished, false, false, false⟩) ∧
    ((numaSys c).run
      [.dSub false false, .dSub true true, .pSub false true, .pSub true true,
       .recycle, .dSub false true, .dSub true true, .promote]).map (numaView true) =
      some ⟨true, .decoding, true, false, false⟩ := by
  decide

/-- **Settled-count barrier on P (`twoSubs = true`): sibling D2H in flight keeps
`pHeld = true` until both producer sub-managers settle.** -/
theorem trace_settledCounts_send_fail_waits_for_sibling :
    let c : NumaCfg := ⟨connectorTwoChannel, .fail, .settledCounts, true, true⟩
    ((numaSys c).run [.pSub false false]).map
      (fun s => (s.pMgr.failedRecvingCounts, s.pHeld, numaEarlyFree true s)) =
      some (1, true, false) ∧
    ((numaSys c).run [.pSub false false, .pSub true true]).map
      (fun s => (s.pMgr, s.pHeld)) =
      some ({}, false) := by
  decide

/-- **Settled-count barrier at `num_subs = 1` (`twoSubs = false`): repeated
failures and subsequent healthy reuse on the same `req_id` all clear `dMgr` and
complete cleanly.** -/
theorem trace_settledCounts_single_sub_fail_and_reuse :
    let c : NumaCfg := ⟨connectorTwoChannel, .fail, .settledCounts, false, true⟩
    ((numaSys c).run
      [.pSub false false, .dSub false false, .recycle,
       .pSub false false, .dSub false false, .recycle,
       .pSub false true, .dSub false true, .promote]).map
      (fun s => (s.pMgr, s.dMgr, numaView false s)) =
      some ({}, {}, ⟨false, .decoding, true, false, false⟩) := by
  decide

/-! ### `NumaPolicy.eagerFailSet` counterexample traces (baseline `kv_cache_manager.cc`) -/

/-- **Baseline `eagerFailSet`, consumer premature HBM free (`twoSubs = true`).**
Consumer sub-manager 0 fails while sub-manager 1 still has H2D in flight
(`recvSub1 = none`, as in step 1 of
`KVCacheManagerWrapperTest.AnyFailureImmediatelyTriggersFailedRecvingAndSuppressesFutureAcks`):
`CompleteReadRaw` immediately emits `failed_recving`, freeing `dHeld = false`
while `recvSub1 = none` (`numaEarlyFree = true`). -/
theorem trace_eagerFailSet_recv_early_free :
    let cFail : NumaCfg := ⟨connectorTwoChannel, .fail, .eagerFailSet, true, true⟩
    let cRec : NumaCfg := ⟨connectorTwoChannel, .recompute, .eagerFailSet, true, true⟩
    ((numaSys cFail).run [.dSub false false]).map
      (fun s => (s.recvSub1, s.dHeld, numaEarlyFree true s)) =
      some (none, false, true) ∧
    ((numaSys cRec).run [.dSub false false, .promote]).map
      (fun s => (s.recvSub1, s.dHeld, numaEarlyFree true s)) =
      some (none, false, true) := by
  decide

/-- **Baseline `eagerFailSet`, producer premature HBM free (`twoSubs = true`).**
Producer sub-manager 0 fails while sub-manager 1 still has D2H in flight
(`sendSub1 = none`): `CompleteReadRaw` immediately emits `failed_recving`, which
the producer connector merges into `finished_sending`, freeing `pHeld = false`
while `sendSub1 = none` (`numaEarlyFree = true`). -/
theorem trace_eagerFailSet_send_early_free :
    let c : NumaCfg := ⟨connectorTwoChannel, .fail, .eagerFailSet, true, true⟩
    ((numaSys c).run [.pSub false false]).map
      (fun s => (s.sendSub1, s.pHeld, numaEarlyFree true s)) =
      some (none, false, true) := by
  decide

/-- **Baseline `eagerFailSet` at `num_subs = 1` (`twoSubs = false`, default
`ENABLE_MULTI_NUMA=0`): `failed_recving_set_` is never erased on failure,
leaking blocks on `req_id` reuse (**N3**).**
Round 1 fails on both P and D (`pMgr.failedRecvingSet = true`,
`dMgr.failedRecvingSet = true`) and recycles `req_id`. In Round 2:
* if P and D fail again, `failed_recving_set_.insert(req).second` is `false` on
  both managers, dropping `failed_recving` and leaking both `pHeld` and `dHeld`;
* if D's retry succeeds (`dSub false true`), `failed_recving_set_` suppresses
  `done_recving` (`emit = none`), leaving D permanently parked (`numaLeakD = true`). -/
theorem trace_eagerFailSet_single_sub_reuse_leaks :
    let c : NumaCfg := ⟨connectorTwoChannel, .fail, .eagerFailSet, false, true⟩
    ((numaSys c).run
      [.pSub false false, .dSub false false, .recycle,
       .pSub false false, .dSub false false]).map
      (fun s => (numaLeakP false s, numaLeakD false s)) =
      some (true, true) ∧
    ((numaSys c).run
      [.pSub false false, .dSub false false, .recycle,
       .dSub false true]).map (numaLeakD false) =
      some true := by
  decide

/-- **Baseline `eagerFailSet` at `num_subs = 2`: partial failure desynchronizes
`done_recving_counts_`, suppressing Transfer 2's `done_recving` and emitting
premature `done_recving` on Transfer 3 after only 1 of 2 sub-managers finishes.**
1. Transfer 1: `sub0` fails (`some .failedRecving`) and `sub1` finishes H2D
   (`none`), leaving `doneRecvingCounts = 1` and `failedRecvingSet = true`.
2. Transfer 2 (healthy, reusing `req_id`): `sub0` finishes (`cnt = 2 ≥ 2`,
   suppressed by `failedRecvingSet = true`, resetting `doneRecvingCounts = 0` and
   `failedRecvingSet = false`), then `sub1` finishes (`doneRecvingCounts = 1`,
   `none`) — Transfer 2 never emits `done_recving` (`numaLeakD = true` in `numaSys`).
3. Transfer 3 (healthy, reusing `req_id`): `sub0` finishes H2D while `sub1` has
   not yet finished; `doneRecvingCounts` reaches `1 + 1 = 2` with
   `failedRecvingSet = false`, emitting `some .doneRecving` on a single
   sub-manager's completion. -/
theorem trace_eagerFailSet_partial_fail_desync_premature_doneRecving :
    -- In `numaSys`, Transfer 2 already leaks D's blocks (`numaLeakD = true`)
    -- with `dMgr.doneRecvingCounts = 1` left behind:
    let c : NumaCfg := ⟨connectorTwoChannel, .fail, .eagerFailSet, true, true⟩
    (((numaSys c).run
      [.dSub false false, .dSub true true, .pSub false true, .pSub true true,
       .recycle, .dSub false true, .dSub true true]).map
      (fun s => (s.dMgr.doneRecvingCounts, s.dMgr.failedRecvingSet, numaLeakD true s)) =
      some (1, false, true)) ∧
    -- And a 3rd `StartRead` on that `dMgr` emits `some .doneRecving` after ONLY
    -- `sub0` reports (`sub1` still in flight):
    (let m0 : NumaMgr := { reqExpected := some 2 }
     let (m1, r1a) := numaStepRecv .eagerFailSet 2 false m0
     let (m2, r1b) := numaStepRecv .eagerFailSet 2 true m1
     let (m3, r2a) := numaStepRecv .eagerFailSet 2 true { m2 with reqExpected := some 2 }
     let (m4, r2b) := numaStepRecv .eagerFailSet 2 true m3
     let (_,  r3a) := numaStepRecv .eagerFailSet 2 true { m4 with reqExpected := some 2 }
     (r1a, r1b, r2a, r2b, r3a) =
       (some .failedRecving, none, none, none, some .doneRecving)) := by
  decide

end TpuSyncVerify.Transfer.Caller

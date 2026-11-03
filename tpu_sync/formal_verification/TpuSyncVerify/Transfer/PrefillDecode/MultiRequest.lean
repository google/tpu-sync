import TpuSyncVerify.Transfer.PrefillDecode.Pipeline

/-!
# Multi-request system and top-level properties across requests

The C++ names below are those of `Pipeline.lean`, which this module composes.

Multiple requests $R_0, R_1, R_2, \dots$ (`reqs : List Pipeline`) run
concurrently and recycle the four memories (`prefillHbm`, `prefillStaging`,
`decodeStaging`, `decodeHbm`) across two ownership boundaries:
- **Host staging (`prefillStaging`, `decodeStaging`)** is owned by TPU Sync's
  `StagingBlockAllocator` and is returned to the pool by `Lifecycle.settleLocked`
  (`ReleaseStagingBlocksLocked` / `ReleaseStagingLocked` in `FinishLocked` and
  `EndSendOpLocked` / `EndRecvOpLocked`, which also return any host layer
  buffer leases a bounded-staging session still holds,
  `staging_layer_manager_.ReleaseAll()`) as soon as each session settles
  (`hasStaging = false`).
- **TPU HBM (`prefillHbm`, `decodeHbm`)** is owned by the serving engines and
  is handed back to the caller when `poll_stats()` publishes the session
  outcome (`send.published ≠ none`, `recv.published ≠ none`).

The multi-request transition system `multiSys` models both **read-after-release**
and **write-after-release** across requests on all four buffers, as well as
**overlapped execution** where a later request recycles prefill HBM and prefill
staging while an earlier request's receive side is still running:
- **Read-after-release:** Once a request releases a buffer, recycling overwrites
  that request's copy of the buffer with `.junk` (via `.reclaim`,
  `.reseatPrefillStaging`, `.reseatDecodeStaging`, `.recyclePrefill`, or
  `.nextRequest`), so any straggling read (`d2hReady`, `h2hDone`, `h2dReady`)
  would copy `.junk`.
- **Write-after-release:** On every transfer step (`reqStep idx e`),
  `wroteReleased` checks each of the four buffers individually against its own
  release condition (`send.published ≠ none`, `!send.life.hasStaging`,
  `!recv.life.hasStaging`, `recv.published ≠ none`). If a request ever modifies
  a buffer it has already released, `multiStep` corrupts all requests' buffers
  with `.junk`.

The top-level system theorems use all of the single-request obligations from
`Pipeline.lean` (`PublicationCorrect`, `DecodeHbmSafe`, `PrefillHbmSafe`,
`StagingSafe`, `NoOpLeak`, `inv_can_handoff`) to establish for **every**
request `idx`:
1. `system_data_correct` and `system_attention_safe`: across any reachable
   multi-request state and any number of concurrent or recycled requests,
   whenever request `idx` publishes `done_recving`, its `decodeHbm` holds
   `good n` and remains `good n` across all subsequent multi-request transitions.
2. `system_progress`: from any reachable multi-request state and any request
   `idx`, a finite trace drains all in-flight operations of `idx`, returns both
   staging buffers to `StagingBlockAllocator`, and publishes both HBM outcomes
   via `poll_stats()` (`HandedOff`), enabling both `.recyclePrefill idx` and
   `.nextRequest idx`.

## Source map

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `tpu_sync/core/staging_arena.cc` | `StagingBlockAllocator::Allocation::Reset` | `:387-398` |
| `tpu_sync/core/staging_arena.cc` | `StagingBlockAllocator::AcquireLocked` | `:477-512` |
| `tpu_sync/core/staging_arena.cc` | `StagingBlockAllocator::Acquire` | `:514-518` |
| `tpu_sync/core/staging_arena.cc` | `StagingBlockAllocator::ReleaseSlot` | `:571-577` |
| `tpu_sync/core/staging_arena.cc` | `StagingBlockAllocator::ReleaseDynamicBlocks` | `:579-587` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::ReleaseStagingBlocksLocked` | `:186-190` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::ReleaseStagingLocked` | `:439-442` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::NotifyForRead` | `:654-701` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:1045-1149` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1151-1292` |
<!-- source-map:end -->
-/

namespace TpuSyncVerify.Transfer.PrefillDecode.Pipeline

/-! ## Multi-request transition system -/

/-- Corrupt all four buffers of a pipeline with `.junk` (triggered if any
request commits a write-after-release on any buffer). -/
def corruptBuffers (s : Pipeline) : Pipeline :=
  { s with
    prefillHbm := List.replicate s.numLayers .junk,
    prefillStaging := List.replicate s.numLayers .junk,
    decodeStaging := List.replicate s.numLayers .junk,
    decodeHbm := List.replicate s.numLayers .junk }

/-- Multi-request system state: all started requests `reqs : List Pipeline`,
which may execute concurrently and recycle buffers across requests. -/
structure MultiState where
  numLayers : Nat
  reqs : List Pipeline
  deriving Repr, DecidableEq

inductive MultiEv where
  /-- Request `idx` takes a pipeline transition `e`. If `e` is a transfer step
  and `reqs[idx]` modifies any buffer it has already released (`wroteReleased`),
  all requests' buffers are corrupted with `.junk`. -/
  | reqStep (idx : Nat) (e : Ev)
  /-- Once request `idx` has released its prefill buffers (`PrefillReleased`),
  recycle its prefill HBM and prefill staging (overwriting `reqs[idx]`'s copies
  with `.junk`) and start a new request — even while `reqs[idx]`'s receive side
  is still running. -/
  | recyclePrefill (idx : Nat)
  /-- Once request `idx` has released all four buffers (`HandedOff`), recycle
  its buffers and start a new request. -/
  | nextRequest (idx : Nat)
  deriving Repr, DecidableEq

def multiInit (n : Nat) : MultiState :=
  { numLayers := n, reqs := [init n] }

/-- Replace `reqs[idx]` with `r'` and spawn a fresh request `init ms.numLayers`. -/
def MultiState.spawnAfterRecycle (ms : MultiState) (idx : Nat) (r' : Pipeline) : MultiState :=
  { ms with reqs := (ms.reqs.set idx r') ++ [init ms.numLayers] }

def multiStep (ms : MultiState) : MultiEv → Option MultiState
  | .reqStep idx e =>
    match ms.reqs[idx]? with
    | none => none
    | some r =>
      (step r e).map fun r' =>
        let reqs' := ms.reqs.set idx r'
        if !isRecycleEv e && wroteReleased r r' then
          { ms with reqs := reqs'.map corruptBuffers }
        else
          { ms with reqs := reqs' }
  | .recyclePrefill idx =>
    match ms.reqs[idx]? with
    | none => none
    | some r =>
      if PrefillReleased r then
        some (ms.spawnAfterRecycle idx (recyclePrefillBufs r))
      else none
  | .nextRequest idx =>
    match ms.reqs[idx]? with
    | none => none
    | some r =>
      if HandedOff r then
        some (ms.spawnAfterRecycle idx (recycleAllBufs r))
      else none

def multiSys (n : Nat) : System MultiState MultiEv := ⟨multiInit n, multiStep⟩

/-! ## Multi-request inductive invariant -/

structure MultiInv (n : Nat) (ms : MultiState) : Prop where
  n_eq : ms.numLayers = n
  reqs_inv : ∀ r ∈ ms.reqs, Inv r ∧ r.numLayers = n

theorem multiInv_init (n : Nat) : MultiInv n (multiInit n) := by
  refine ⟨rfl, ?_⟩
  intro r hr
  simp only [multiInit, List.mem_singleton] at hr
  subst hr
  exact ⟨inv_init n, rfl⟩

theorem MultiInv.spawnAfterRecycle {n idx : Nat} {ms : MultiState} {r' : Pipeline}
    (h : MultiInv n ms) (hr' : Inv r') (hn' : r'.numLayers = n) :
    MultiInv n (ms.spawnAfterRecycle idx r') := by
  refine ⟨h.n_eq, ?_⟩
  intro x hx
  simp only [MultiState.spawnAfterRecycle, List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | rfl
  · rcases mem_set_cases hx with rfl | hx
    · exact ⟨hr', hn'⟩
    · exact h.reqs_inv x hx
  · exact ⟨h.n_eq ▸ inv_init n, h.n_eq⟩

theorem multiStep_inv {n : Nat} {ms ms' : MultiState} {ev : MultiEv}
    (h : MultiInv n ms) (hs : multiStep ms ev = some ms') : MultiInv n ms' := by
  cases ev with
  | reqStep idx e =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[idx]? with
    | none => simp [hget] at hs
    | some r =>
      simp only [hget, Option.map_eq_some_iff] at hs
      obtain ⟨r', hr', rfl⟩ := hs
      have ⟨hr_inv, hr_n⟩ := h.reqs_inv r (getElem?_mem hget)
      have hwf := not_isRecycleEv_and_wroteReleased_eq_false hr_inv hr'
      simp only [hwf, Bool.false_eq_true, ↓reduceIte]
      refine ⟨h.n_eq, ?_⟩
      intro x hx
      rcases mem_set_cases hx with rfl | hx
      · exact ⟨step_inv hr_inv hr', (step_numLayers hr').trans hr_n⟩
      · exact h.reqs_inv x hx
  | recyclePrefill idx =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[idx]? with
    | none => simp [hget] at hs
    | some r =>
      simp only [hget] at hs
      split at hs
      · rename_i hrel
        cases hs
        have ⟨hr_inv, hr_n⟩ := h.reqs_inv r (getElem?_mem hget)
        exact h.spawnAfterRecycle (hr_inv.recyclePrefillBufs hrel) hr_n
      · cases hs
  | nextRequest idx =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[idx]? with
    | none => simp [hget] at hs
    | some r =>
      simp only [hget] at hs
      split at hs
      · rename_i hrel
        cases hs
        have ⟨hr_inv, hr_n⟩ := h.reqs_inv r (getElem?_mem hget)
        exact h.spawnAfterRecycle (hr_inv.recycleAllBufs hrel) hr_n
      · cases hs

theorem reachable_multiInv {n : Nat} {ms : MultiState}
    (h : (multiSys n).Reachable ms) : MultiInv n ms :=
  (multiSys n).reachable_induction (multiInv_init n) (fun _ _ _ hi hs => multiStep_inv hi hs) h

/-! ## Top-level system theorems across requests -/

/-- **Publication correctness across requests:**
In any reachable multi-request state — across any number of concurrent,
overlapped, succeeded, failed, or cancelled requests — whenever any request `r`
at index `idx` publishes `done_recving`, its `decodeHbm` holds `good n`. -/
theorem system_data_correct {n : Nat} {ms : MultiState} {idx : Nat} {r : Pipeline}
    (h : (multiSys n).Reachable ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true) :
    r.decodeHbm = good n := by
  have hinv := reachable_multiInv h
  have ⟨hr_inv, hr_n⟩ := hinv.reqs_inv r (getElem?_mem hreq)
  rw [← hr_n]
  exact (inv_safe hr_inv).1 hp

theorem spawnAfterRecycle_decodeHbm_quiet {ms : MultiState} {idx j : Nat} {r rj' : Pipeline}
    (hreq : ms.reqs[idx]? = some r) (hp : r.recv.published ≠ none)
    (hd : j = idx → rj'.decodeHbm = r.decodeHbm ∧ rj'.recv.published = r.recv.published) :
    ∃ r', (ms.spawnAfterRecycle j rj').reqs[idx]? = some r' ∧
      r'.decodeHbm = r.decodeHbm ∧ r'.recv.published ≠ none := by
  have hlt : idx < ms.reqs.length := lt_length_of_getElem?_eq hreq
  have hlen_set : idx < (ms.reqs.set j rj').length := by rw [List.length_set]; exact hlt
  by_cases hji : j = idx
  · subst hji
    obtain ⟨hd', hp'⟩ := hd rfl
    refine ⟨rj', ?_, hd', hp' ▸ hp⟩
    simp [MultiState.spawnAfterRecycle, List.getElem?_append_left hlen_set, List.getElem?_set_self hlt]
  · refine ⟨r, ?_, rfl, hp⟩
    simp [MultiState.spawnAfterRecycle, List.getElem?_append_left hlen_set, List.getElem?_set_ne hji, hreq]

theorem multiStep_decodeHbm_quiet {n : Nat} {ms ms' : MultiState} {idx : Nat} {r : Pipeline}
    {ev : MultiEv} (hinv : MultiInv n ms) (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published ≠ none) (hs : multiStep ms ev = some ms') :
    ∃ r', ms'.reqs[idx]? = some r' ∧ r'.decodeHbm = r.decodeHbm ∧ r'.recv.published ≠ none := by
  have hlt : idx < ms.reqs.length := lt_length_of_getElem?_eq hreq
  cases ev with
  | reqStep j e =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[j]? with
    | none => simp [hget] at hs
    | some rj =>
      simp only [hget, Option.map_eq_some_iff] at hs
      obtain ⟨rj', hrj', rfl⟩ := hs
      have ⟨hrj_inv, _⟩ := hinv.reqs_inv rj (getElem?_mem hget)
      have hwf := not_isRecycleEv_and_wroteReleased_eq_false hrj_inv hrj'
      simp only [hwf, Bool.false_eq_true, ↓reduceIte]
      by_cases hji : j = idx
      · subst hji
        rw [hreq, Option.some.injEq] at hget; subst hget
        refine ⟨rj', List.getElem?_set_self hlt,
          decodeHbm_quiet hrj_inv hp hrj',
          step_published_ne_none hrj' hp⟩
      · refine ⟨r, ?_, rfl, hp⟩
        rw [List.getElem?_set_ne hji, hreq]
  | recyclePrefill j =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[j]? with
    | none => simp [hget] at hs
    | some rj =>
      simp only [hget] at hs
      split at hs
      · cases hs
        refine spawnAfterRecycle_decodeHbm_quiet hreq hp fun hji => ?_
        subst hji; rw [hreq, Option.some.injEq] at hget; subst hget
        exact ⟨rfl, rfl⟩
      · cases hs
  | nextRequest j =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[j]? with
    | none => simp [hget] at hs
    | some rj =>
      simp only [hget] at hs
      split at hs
      · cases hs
        refine spawnAfterRecycle_decodeHbm_quiet hreq hp fun hji => ?_
        subst hji; rw [hreq, Option.some.injEq] at hget; subst hget
        exact ⟨rfl, rfl⟩
      · cases hs

theorem multiRunFrom_decodeHbm_quiet {n idx : Nat} :
    ∀ (evs : List MultiEv) {ms ms' : MultiState} {r : Pipeline},
      MultiInv n ms → ms.reqs[idx]? = some r → r.recv.published ≠ none →
      (multiSys n).runFrom ms evs = some ms' →
      ∃ r', ms'.reqs[idx]? = some r' ∧ r'.decodeHbm = r.decodeHbm
  | [], _, _, r, _, hreq, _, hr => by
    simp [System.runFrom] at hr; subst hr
    exact ⟨r, hreq, rfl⟩
  | ev :: evs, ms, ms', r, hinv, hreq, hp, hr => by
    simp only [System.runFrom, multiSys, List.foldlM_cons] at hr
    cases hse : multiStep ms ev with
    | none => simp [hse] at hr
    | some ms₁ =>
      simp only [hse] at hr
      obtain ⟨r₁, hreq₁, hd₁, hp₁⟩ := multiStep_decodeHbm_quiet hinv hreq hp hse
      obtain ⟨r', hreq', hd'⟩ :=
        @multiRunFrom_decodeHbm_quiet n idx evs _ _ r₁ (multiStep_inv hinv hse) hreq₁ hp₁ hr
      exact ⟨r', hreq', hd'.trans hd₁⟩

/-- **Decoding safety across requests:**
Once any request `r` at index `idx` publishes `done_recving`, its `decodeHbm`
stays equal to `good n` across any subsequent sequence of multi-request
transitions `evs` (including steps of `idx`, steps of earlier or later
concurrent requests, and buffer recycling via `recyclePrefill` or
`nextRequest`). -/
theorem system_attention_safe {n : Nat} {ms ms' : MultiState} {idx : Nat} {r : Pipeline}
    (h : (multiSys n).Reachable ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true)
    (evs : List MultiEv)
    (hr : (multiSys n).runFrom ms evs = some ms') :
    ∃ r', ms'.reqs[idx]? = some r' ∧ r'.decodeHbm = good n := by
  have hinv := reachable_multiInv h
  obtain ⟨r', hreq', hd⟩ := multiRunFrom_decodeHbm_quiet evs hinv hreq (by simp [hp]) hr
  exact ⟨r', hreq', hd.trans (system_data_correct h hreq hp)⟩

theorem runFrom_reqSteps {n idx : Nat} :
    ∀ (evs : List Ev) {ms : MultiState} {r r' : Pipeline},
      MultiInv n ms →
      ms.reqs[idx]? = some r →
      (sys n).runFrom r evs = some r' →
      ∃ ms', (multiSys n).runFrom ms (evs.map (.reqStep idx)) = some ms' ∧
        ms'.reqs[idx]? = some r'
  | [], ms, r, r', _, hreq, hr => by
    simp [System.runFrom] at hr; subst hr
    exact ⟨ms, rfl, hreq⟩
  | e :: evs, ms, r, r', hinv, hreq, hr => by
    simp only [System.runFrom, sys, multiSys, List.map_cons, List.foldlM_cons] at hr ⊢
    cases hse : step r e with
    | none => simp [hse] at hr
    | some r₁ =>
      simp only [hse] at hr
      have ⟨hr_inv, _⟩ := hinv.reqs_inv r (getElem?_mem hreq)
      have hwf := not_isRecycleEv_and_wroteReleased_eq_false hr_inv hse
      have hms : multiStep ms (.reqStep idx e) = some { ms with reqs := ms.reqs.set idx r₁ } := by
        simp only [multiStep, hreq, hse, Option.map_some, hwf, Bool.false_eq_true, ↓reduceIte]
      simp only [hms]
      have hinv₁ : MultiInv n { ms with reqs := ms.reqs.set idx r₁ } := multiStep_inv hinv hms
      have hreq₁ : ({ ms with reqs := ms.reqs.set idx r₁ } : MultiState).reqs[idx]? = some r₁ :=
        List.getElem?_set_self (lt_length_of_getElem?_eq hreq)
      exact @runFrom_reqSteps n idx evs _ _ _ hinv₁ hreq₁ hr

/-- **Progress and buffer release across requests:**
From any reachable multi-request state and any request `idx`
(`ms.reqs[idx]? = some r`), there exists a finite trace `evs` that drains all
in-flight operations of `idx`, returns both host staging buffers to
`StagingBlockAllocator`, and publishes both HBM outcomes (`HandedOff r'`),
enabling both `.recyclePrefill idx` and `.nextRequest idx`. -/
theorem system_progress {n : Nat} {ms : MultiState} {idx : Nat} {r : Pipeline}
    (h : (multiSys n).Reachable ms)
    (hreq : ms.reqs[idx]? = some r) :
    ∃ evs ms' r', (multiSys n).runFrom ms evs = some ms' ∧
      ms'.reqs[idx]? = some r' ∧
      HandedOff r' ∧
      (multiStep ms' (.recyclePrefill idx)).isSome = true ∧
      (multiStep ms' (.nextRequest idx)).isSome = true := by
  have hinv := reachable_multiInv h
  have ⟨hr_inv, _⟩ := hinv.reqs_inv r (getElem?_mem hreq)
  obtain ⟨evs, r', hr, hrel⟩ := inv_can_handoff n hr_inv
  obtain ⟨ms', hrun, hreq'⟩ := runFrom_reqSteps evs hinv hreq hr
  refine ⟨evs.map (.reqStep idx), ms', r', hrun, hreq', hrel, ?_, ?_⟩
  · simp [multiStep, hreq', hrel.prefillReleased]
  · simp [multiStep, hreq', hrel]

end TpuSyncVerify.Transfer.PrefillDecode.Pipeline

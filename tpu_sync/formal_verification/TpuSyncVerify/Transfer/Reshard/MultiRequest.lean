import TpuSyncVerify.Transfer.Reshard.Pipeline

/-!
# Multi-request reshard system and top-level properties across transfers

The C++ names and file aliases below are those of `Pipeline.lean`, which this
module composes.

Multiple reshard transfers $R_0, R_1, R_2, \dots$ (`reqs : List Pipeline`) run
concurrently and recycle the four caller- and arena-visible memories (`srcHbm`,
`srcStaging`, `dstStaging`, `dstHbm`) across two ownership boundaries:
- **Host staging (`srcStaging`, `dstStaging`)** is owned by `PoolStagingStorage`
  and is returned to the bounded staging arena by `Lifecycle.settleLocked`
  (`ReshardSendSession::SettleLocked` / `ReshardSendSession::ReleaseLeasesLocked`
  and `ReshardReceiveSession::ReleaseStagingLocked`) as soon as each session
  settles (`hasStaging = false`).
- **TPU HBM (`srcHbm`, `dstHbm`)** is owned by the serving engines and
  is handed back to the caller when `KVCacheManagerWithTransfer::CompleteReadWithDetails`
  publishes the session outcome (`send.published ≠ none`, `recv.published ≠ none`).

The multi-request transition systems `multiSys` (post-Phase-1 armed transfers)
and `multiSysUnarmed` (pre-Phase-1 unarmed transfers requiring
`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` before
`KVCacheManagerWithTransfer::PoolReshardPush`, as driven by
`ReshardCoordinator::ExecutePoolReshard`) model both **read-after-release** and
**write-after-release** across transfers on all four buffers, as well as
**overlapped execution** where a later transfer recycles source HBM and sender
staging while an earlier transfer's receive side is still running:
- **Read-after-release:** Once a transfer releases a buffer, recycling overwrites
  that transfer's copy of the buffer with `.junk` (via `.reclaim`,
  `.reseatSrcStaging`, `.reseatDstStaging`, `.recycleSrc`, or `.nextRequest`;
  `recycleAllBufs` retains `r.dstHbm` so post-publication quietness of `r` can
  be observed), so any straggling read (`d2hDone`, `landStream`, `h2dDone` in
  `ReshardReceiveSession::FinishPoolH2d`) would copy `.junk`.
- **Write-after-release:** On every transfer step (`reqStep idx e`),
  `wroteReleased` checks each of the four buffers individually against its own
  release condition (`send.published ≠ none`, `!send.life.hasStaging`,
  `!recv.life.hasStaging`, `recv.published ≠ none`). If any transfer ever
  modifies a buffer it has already released, `multiStep` corrupts every
  transfer's buffers in `ms.reqs` with `.junk` (`corruptBuffers`).

The top-level system theorems establish for **every** transfer `idx` (under both
`multiSys` and `multiSysUnarmed`):
1. `system_data_correct` and `system_attention_safe` (and their `multiSysUnarmed`
   counterparts): across any reachable multi-transfer state and any number of
   concurrent or recycled transfers, whenever transfer `idx` publishes
   `done_recving`, its `dstHbm` holds `good ranks.length` (no within-transfer
   aliased-page rank clobbering and no straggling reads from recycled buffers)
   and remains `good ranks.length` across all subsequent multi-transfer
   transitions (no post-publication writes by `idx` and no write-after-release
   by any concurrent or recycled transfer `j`).
2. `system_progress` (and `system_unarmed_progress`): from any reachable
   multi-transfer state and any transfer `idx`, a finite trace drains all
   in-flight operations of `idx`, returns both staging leases to
   `PoolStagingStorage`, and publishes both HBM outcomes via
   `CompleteReadWithDetails` (`HandedOff`), enabling both `.recycleSrc idx`
   and `.nextRequest idx`.

## Assumptions

* **A1 — Cross-transfer destination page disjointness (`dstHbm` per transfer):**
  Each transfer `r ∈ ms.reqs` owns its own `dstHbm` vector while in flight, and
  aliased-page rank ordering (`writePoolHbm`, `AliasedPageCorrect`) is enforced
  within each `ReshardReceiveSession` (`CollectEligiblePoolH2dsLocked`). Across
  concurrent transfers, destination physical pages are disjoint:
  - On any shared storage (`PoolSpec::storage_index`), all pools declared for
    bounded pool staging agree on `block_stride_bytes` and `num_blocks`
    (`KVCacheManagerBase::ConfigurePoolStaging`), and `ComputePoolBlockCopyExtents`
    places every region of block $b$ of pool $P$ at device offset
    $\text{base\_offset\_bytes} + b \cdot \text{block\_stride\_bytes} + \text{region\_offset} \in [b \cdot S, (b + 1) \cdot S)$
    when each pool's `base_offset_bytes + region.extent_end_bytes()` fits within
    one storage page $S = \text{block\_stride\_bytes}$ (`PoolSpec::Validate(storage_bytes)`
    with `storage_bytes = num_blocks * block_stride_bytes`, and the caller's
    manifest builder). Two pool blocks $b$ and $b'$ on the same storage
    therefore overlap in device HBM bytes only when $b = b'$.
  - On the host staging side, `KVCacheManagerBase::AcquirePoolStagingLease`
    allocates disjoint host arena slots to distinct concurrent `uuid`s
    (`PoolStaging.no_slot_aliasing_between_distinct_uuids`).
  - Above `tpu_sync`, the serving engine's unified KV-cache block allocator
    draws block IDs from a single `[0, num_blocks)` free pool shared across all
    KV-cache groups (full-attention and state/linear-attention groups alike) and
    excludes shared prefix-cached blocks from `dst_device_block_ids`, so two
    concurrent transfers never target the same uncached destination `block_id` $b$.

## Lock scopes

| Event | Scopes | Between them |
|---|---|---|
| `reqStep idx e` | `Pipeline.step` scopes on `ms.reqs[idx]` (`Pipeline.lean` §"Lock scopes") | `wroteReleased` checks `ms.reqs[idx]`'s post-step buffers against its pre-state release flags (`send.published`, `send.life.hasStaging`, `recv.life.hasStaging`, `recv.published`) |
| `recycleSrc idx`, `nextRequest idx` | 0 (caller engine and `PoolStagingStorage` after release, then `ReshardCoordinator::ExecutePoolReshard` to spawn a fresh transfer) | enabled only after `SettleLocked` / `ReleaseLeasesLocked` and `ReleaseStagingLocked` have released the `PoolStagingLease` under session `mu_` and `staging_mu_`, and after `CompleteReadWithDetails` has published the session outcome under manager `mu_` |

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `reshard_send.cc` | `ReshardSendSession::ReleaseLeasesLocked` | `:470-474` |
| `reshard_send.cc` | `ReshardSendSession::SettleLocked` | `:476-481` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:461-465` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:668-706` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:664-701` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:703-739` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:980-1121` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`const int64_t receiver_arm_start_ns = MonotonicNs();` .. `if (!status.ok()) return status;`) | `:432-543` |
<!-- source-map:end -->
-/

namespace TpuSyncVerify.Transfer.Reshard.Pipeline

/-! ## Multi-request transition system -/

/-- Corrupt all four buffers of a reshard pipeline with `.junk` (triggered if any
transfer commits a write-after-release on any buffer). -/
def corruptBuffers (s : Pipeline) : Pipeline :=
  { s with
    srcHbm := List.replicate s.numPools .junk,
    srcStaging := List.replicate s.numPools .junk,
    dstStaging := List.replicate s.numPools .junk,
    dstHbm := List.replicate s.numPools .junk }

/-- Multi-request reshard system state: all started transfers `reqs : List Pipeline`,
which may execute concurrently and recycle buffers across transfers. When
`startArmed = true` (`multiSys`), spawned transfers begin in `init` (post-Stage-1
armed); when `startArmed = false` (`multiSysUnarmed`), spawned transfers begin
in `initUnarmed` (requiring `.armRecv` before any sender D2H or receiver push). -/
structure MultiState where
  ranks : List Nat
  peerCounts : List Nat
  startArmed : Bool := true
  reqs : List Pipeline
  deriving Repr, DecidableEq

inductive MultiEv where
  /-- Transfer `idx` takes a pipeline transition `e`. If `e` is a transfer step
  and `reqs[idx]` modifies any buffer it has already released (`wroteReleased`),
  all transfers' buffers are corrupted with `.junk`. -/
  | reqStep (idx : Nat) (e : Ev)
  /-- Once transfer `idx` has released its source buffers (`SrcReleased`),
  recycle its source HBM and sender staging (overwriting `reqs[idx]`'s copies
  with `.junk`) and start a new transfer — even while `reqs[idx]`'s receive side
  is still running. -/
  | recycleSrc (idx : Nat)
  /-- Once transfer `idx` has released all four buffers (`HandedOff`), recycle
  its buffers and start a new transfer. -/
  | nextRequest (idx : Nat)
  deriving Repr, DecidableEq

/-- Fresh pipeline instance matching `ms.startArmed`. -/
def MultiState.freshPipeline (ms : MultiState) : Pipeline :=
  if ms.startArmed then init ms.ranks ms.peerCounts
  else initUnarmed ms.ranks ms.peerCounts

def multiInit (ranks peerCounts : List Nat) (startArmed : Bool := true) : MultiState :=
  { ranks := ranks
    peerCounts := peerCounts
    startArmed := startArmed
    reqs := [if startArmed then init ranks peerCounts else initUnarmed ranks peerCounts] }

/-- Replace `reqs[idx]` with `r'` and spawn a fresh transfer `ms.freshPipeline`. -/
def MultiState.spawnAfterRecycle (ms : MultiState) (idx : Nat) (r' : Pipeline) : MultiState :=
  { ms with reqs := (ms.reqs.set idx r') ++ [ms.freshPipeline] }

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
  | .recycleSrc idx =>
    match ms.reqs[idx]? with
    | none => none
    | some r =>
      if SrcReleased r then
        some (ms.spawnAfterRecycle idx (recycleSrcBufs r))
      else none
  | .nextRequest idx =>
    match ms.reqs[idx]? with
    | none => none
    | some r =>
      if HandedOff r then
        some (ms.spawnAfterRecycle idx (recycleAllBufs r))
      else none

def multiSys (ranks peerCounts : List Nat) : System MultiState MultiEv :=
  ⟨multiInit ranks peerCounts true, multiStep⟩

def multiSysUnarmed (ranks peerCounts : List Nat) : System MultiState MultiEv :=
  ⟨multiInit ranks peerCounts false, multiStep⟩

/-! ## Multi-request inductive invariant -/

structure MultiInv (ranks peerCounts : List Nat) (ms : MultiState) : Prop where
  ranks_eq : ms.ranks = ranks
  peers_eq : ms.peerCounts = peerCounts
  valid : validConfig ranks peerCounts = true
  reqs_inv : ∀ r ∈ ms.reqs, Inv r ∧ r.numPools = ranks.length

theorem inv_freshPipeline {ranks peerCounts : List Nat} (startArmed : Bool)
    (hvalid : validConfig ranks peerCounts = true) :
    Inv (if startArmed then init ranks peerCounts else initUnarmed ranks peerCounts) ∧
    (if startArmed then init ranks peerCounts else initUnarmed ranks peerCounts).numPools =
      ranks.length := by
  cases startArmed
  · exact ⟨inv_initUnarmed hvalid, rfl⟩
  · exact ⟨inv_init hvalid, rfl⟩

theorem multiInv_init {ranks peerCounts : List Nat} {startArmed : Bool}
    (hvalid : validConfig ranks peerCounts = true) :
    MultiInv ranks peerCounts (multiInit ranks peerCounts startArmed) := by
  refine ⟨rfl, rfl, hvalid, ?_⟩
  intro r hr
  simp only [multiInit, List.mem_singleton] at hr
  subst hr
  exact inv_freshPipeline startArmed hvalid

theorem MultiInv.spawnAfterRecycle {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r' : Pipeline}
    (h : MultiInv ranks peerCounts ms) (hr' : Inv r') (hn' : r'.numPools = ranks.length) :
    MultiInv ranks peerCounts (ms.spawnAfterRecycle idx r') := by
  refine ⟨h.ranks_eq, h.peers_eq, h.valid, ?_⟩
  intro x hx
  simp only [MultiState.spawnAfterRecycle, MultiState.freshPipeline,
    List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | rfl
  · rcases mem_set_cases hx with rfl | hx
    · exact ⟨hr', hn'⟩
    · exact h.reqs_inv x hx
  · rw [h.ranks_eq, h.peers_eq]
    exact inv_freshPipeline ms.startArmed h.valid

theorem multiStep_inv {ranks peerCounts : List Nat} {ms ms' : MultiState} {ev : MultiEv}
    (h : MultiInv ranks peerCounts ms) (hs : multiStep ms ev = some ms') :
    MultiInv ranks peerCounts ms' := by
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
      refine ⟨h.ranks_eq, h.peers_eq, h.valid, ?_⟩
      intro x hx
      rcases mem_set_cases hx with rfl | hx
      · exact ⟨step_inv hr_inv hr', (step_numPools hr').trans hr_n⟩
      · exact h.reqs_inv x hx
  | recycleSrc idx =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[idx]? with
    | none => simp [hget] at hs
    | some r =>
      simp only [hget] at hs
      split at hs
      · rename_i hrel
        cases hs
        have ⟨hr_inv, hr_n⟩ := h.reqs_inv r (getElem?_mem hget)
        exact h.spawnAfterRecycle (hr_inv.recycleSrcBufs hrel) hr_n
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

theorem reachable_multiInv {ranks peerCounts : List Nat} {ms : MultiState}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSys ranks peerCounts).Reachable ms) : MultiInv ranks peerCounts ms :=
  (multiSys ranks peerCounts).reachable_induction (multiInv_init hvalid)
    (fun _ _ _ hi hs => multiStep_inv hi hs) h

theorem reachable_unarmed_multiInv {ranks peerCounts : List Nat} {ms : MultiState}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSysUnarmed ranks peerCounts).Reachable ms) : MultiInv ranks peerCounts ms :=
  (multiSysUnarmed ranks peerCounts).reachable_induction (multiInv_init hvalid)
    (fun _ _ _ hi hs => multiStep_inv hi hs) h

/-! ## Top-level system theorems across transfers -/

theorem multiInv_data_correct {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r : Pipeline}
    (hinv : MultiInv ranks peerCounts ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true) :
    r.dstHbm = good ranks.length := by
  have ⟨hr_inv, hr_n⟩ := hinv.reqs_inv r (getElem?_mem hreq)
  rw [← hr_n]
  exact (inv_safe hr_inv).1 hp

/-- **Publication correctness across transfers (`multiSys`):**
In any reachable multi-transfer state — across any number of concurrent,
overlapped, succeeded, failed, or cancelled reshard transfers — whenever any
transfer `r` at index `idx` publishes `done_recving`, its `dstHbm` holds
`good ranks.length`. -/
theorem system_data_correct {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSys ranks peerCounts).Reachable ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true) :
    r.dstHbm = good ranks.length :=
  multiInv_data_correct (reachable_multiInv hvalid h) hreq hp

/-- **Publication correctness across unarmed transfers (`multiSysUnarmed`):**
Same guarantee when every initial and recycled transfer starts unarmed before
Phase 1 of `ReshardCoordinator::ExecutePoolReshard` (`initUnarmed`). -/
theorem system_unarmed_data_correct {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSysUnarmed ranks peerCounts).Reachable ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true) :
    r.dstHbm = good ranks.length :=
  multiInv_data_correct (reachable_unarmed_multiInv hvalid h) hreq hp

theorem spawnAfterRecycle_dstHbm_quiet {ms : MultiState} {idx j : Nat} {r rj' : Pipeline}
    (hreq : ms.reqs[idx]? = some r) (hp : r.recv.published ≠ none)
    (hd : j = idx → rj'.dstHbm = r.dstHbm ∧ rj'.recv.published = r.recv.published) :
    ∃ r', (ms.spawnAfterRecycle j rj').reqs[idx]? = some r' ∧
      r'.dstHbm = r.dstHbm ∧ r'.recv.published ≠ none := by
  have hlt : idx < ms.reqs.length := lt_length_of_getElem?_eq hreq
  have hlen_set : idx < (ms.reqs.set j rj').length := by rw [List.length_set]; exact hlt
  by_cases hji : j = idx
  · subst hji
    obtain ⟨hd', hp'⟩ := hd rfl
    refine ⟨rj', ?_, hd', hp' ▸ hp⟩
    simp [MultiState.spawnAfterRecycle, List.getElem?_append_left hlen_set, List.getElem?_set_self hlt]
  · refine ⟨r, ?_, rfl, hp⟩
    simp [MultiState.spawnAfterRecycle, List.getElem?_append_left hlen_set, List.getElem?_set_ne hji, hreq]

theorem multiStep_dstHbm_quiet {ranks peerCounts : List Nat} {ms ms' : MultiState}
    {idx : Nat} {r : Pipeline} {ev : MultiEv}
    (hinv : MultiInv ranks peerCounts ms) (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published ≠ none) (hs : multiStep ms ev = some ms') :
    ∃ r', ms'.reqs[idx]? = some r' ∧ r'.dstHbm = r.dstHbm ∧ r'.recv.published ≠ none := by
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
          (dstHbm_quiet hrj_inv hp hrj').1,
          by rw [(dstHbm_quiet hrj_inv hp hrj').2.1]; exact hp⟩
      · refine ⟨r, ?_, rfl, hp⟩
        rw [List.getElem?_set_ne hji, hreq]
  | recycleSrc j =>
    simp only [multiStep] at hs
    cases hget : ms.reqs[j]? with
    | none => simp [hget] at hs
    | some rj =>
      simp only [hget] at hs
      split at hs
      · cases hs
        refine spawnAfterRecycle_dstHbm_quiet hreq hp fun hji => ?_
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
        refine spawnAfterRecycle_dstHbm_quiet hreq hp fun hji => ?_
        subst hji; rw [hreq, Option.some.injEq] at hget; subst hget
        exact ⟨rfl, rfl⟩
      · cases hs

theorem multiRunFrom_dstHbm_quiet {ranks peerCounts : List Nat} {idx : Nat} :
    ∀ (evs : List MultiEv) {ms ms' : MultiState} {r : Pipeline},
      MultiInv ranks peerCounts ms → ms.reqs[idx]? = some r → r.recv.published ≠ none →
      (multiSys ranks peerCounts).runFrom ms evs = some ms' →
      ∃ r', ms'.reqs[idx]? = some r' ∧ r'.dstHbm = r.dstHbm
  | [], _, _, r, _, hreq, _, hr => by
    simp [System.runFrom] at hr; subst hr
    exact ⟨r, hreq, rfl⟩
  | ev :: evs, ms, ms', r, hinv, hreq, hp, hr => by
    simp only [System.runFrom, multiSys, List.foldlM_cons] at hr
    cases hse : multiStep ms ev with
    | none => simp [hse] at hr
    | some ms₁ =>
      simp only [hse] at hr
      obtain ⟨r₁, hreq₁, hd₁, hp₁⟩ := multiStep_dstHbm_quiet hinv hreq hp hse
      obtain ⟨r', hreq', hd'⟩ :=
        @multiRunFrom_dstHbm_quiet ranks peerCounts idx evs _ _ r₁
          (multiStep_inv hinv hse) hreq₁ hp₁ hr
      exact ⟨r', hreq', hd'.trans hd₁⟩

/-- **Post-publication HBM safety across transfers (`multiSys`):**
Once any transfer `r` at index `idx` publishes `done_recving`, its `dstHbm`
holds `good ranks.length` (from within-transfer aliased-page rank-wave ordering
and read-before-release) and stays equal to `good ranks.length` across any
subsequent sequence of multi-transfer transitions `evs` (including steps of
`idx`, steps of earlier or later concurrent transfers checked by
`wroteReleased`, and buffer recycling via `recycleSrc` or `nextRequest`). -/
theorem system_attention_safe {ranks peerCounts : List Nat} {ms ms' : MultiState}
    {idx : Nat} {r : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSys ranks peerCounts).Reachable ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true)
    (evs : List MultiEv)
    (hr : (multiSys ranks peerCounts).runFrom ms evs = some ms') :
    ∃ r', ms'.reqs[idx]? = some r' ∧ r'.dstHbm = good ranks.length := by
  have hinv := reachable_multiInv hvalid h
  obtain ⟨r', hreq', hd⟩ :=
    multiRunFrom_dstHbm_quiet evs hinv hreq (by simp [hp]) hr
  exact ⟨r', hreq', hd.trans (system_data_correct hvalid h hreq hp)⟩

/-- **Post-publication HBM safety across unarmed transfers (`multiSysUnarmed`):**
Same post-publication invariance when every transfer starts unarmed (`initUnarmed`). -/
theorem system_unarmed_attention_safe {ranks peerCounts : List Nat} {ms ms' : MultiState}
    {idx : Nat} {r : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSysUnarmed ranks peerCounts).Reachable ms)
    (hreq : ms.reqs[idx]? = some r)
    (hp : r.recv.published = some true)
    (evs : List MultiEv)
    (hr : (multiSysUnarmed ranks peerCounts).runFrom ms evs = some ms') :
    ∃ r', ms'.reqs[idx]? = some r' ∧ r'.dstHbm = good ranks.length := by
  have hinv := reachable_unarmed_multiInv hvalid h
  obtain ⟨r', hreq', hd⟩ :=
    multiRunFrom_dstHbm_quiet evs hinv hreq (by simp [hp]) hr
  exact ⟨r', hreq', hd.trans (system_unarmed_data_correct hvalid h hreq hp)⟩

theorem runFrom_reqSteps {ranks peerCounts : List Nat} {idx : Nat} :
    ∀ (evs : List Ev) {ms : MultiState} {r r' : Pipeline},
      MultiInv ranks peerCounts ms →
      ms.reqs[idx]? = some r →
      (sys ranks peerCounts).runFrom r evs = some r' →
      ∃ ms', (multiSys ranks peerCounts).runFrom ms (evs.map (.reqStep idx)) = some ms' ∧
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
      have hinv₁ : MultiInv ranks peerCounts { ms with reqs := ms.reqs.set idx r₁ } := multiStep_inv hinv hms
      have hreq₁ : ({ ms with reqs := ms.reqs.set idx r₁ } : MultiState).reqs[idx]? = some r₁ :=
        List.getElem?_set_self (lt_length_of_getElem?_eq hreq)
      exact @runFrom_reqSteps ranks peerCounts idx evs _ _ _ hinv₁ hreq₁ hr

theorem multiInv_progress {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r : Pipeline}
    (hinv : MultiInv ranks peerCounts ms)
    (hreq : ms.reqs[idx]? = some r) :
    ∃ evs ms' r', (multiSys ranks peerCounts).runFrom ms evs = some ms' ∧
      ms'.reqs[idx]? = some r' ∧
      HandedOff r' ∧
      (multiStep ms' (.recycleSrc idx)).isSome = true ∧
      (multiStep ms' (.nextRequest idx)).isSome = true := by
  have ⟨hr_inv, _⟩ := hinv.reqs_inv r (getElem?_mem hreq)
  obtain ⟨evs, r', hr, hrel⟩ := inv_can_handoff hr_inv
  obtain ⟨ms', hrun, hreq'⟩ := runFrom_reqSteps evs hinv hreq hr
  refine ⟨evs.map (.reqStep idx), ms', r', hrun, hreq', hrel, ?_, ?_⟩
  · simp [multiStep, hreq', hrel.srcReleased]
  · simp [multiStep, hreq', hrel]

/-- **Progress and buffer release across transfers (`multiSys`):**
From any reachable multi-transfer state and any transfer `idx`
(`ms.reqs[idx]? = some r`), there exists a finite trace `evs` that drains all
in-flight operations of `idx`, releases both host staging leases to
`PoolStagingStorage`, and publishes both HBM outcomes via
`CompleteReadWithDetails` (`HandedOff r'`), enabling both `.recycleSrc idx` and
`.nextRequest idx`. -/
theorem system_progress {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSys ranks peerCounts).Reachable ms)
    (hreq : ms.reqs[idx]? = some r) :
    ∃ evs ms' r', (multiSys ranks peerCounts).runFrom ms evs = some ms' ∧
      ms'.reqs[idx]? = some r' ∧
      HandedOff r' ∧
      (multiStep ms' (.recycleSrc idx)).isSome = true ∧
      (multiStep ms' (.nextRequest idx)).isSome = true :=
  multiInv_progress (reachable_multiInv hvalid h) hreq

/-- **Progress and buffer release across unarmed transfers (`multiSysUnarmed`):**
Same progress guarantee when every transfer starts unarmed (`initUnarmed`). -/
theorem system_unarmed_progress {ranks peerCounts : List Nat} {ms : MultiState}
    {idx : Nat} {r : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (multiSysUnarmed ranks peerCounts).Reachable ms)
    (hreq : ms.reqs[idx]? = some r) :
    ∃ evs ms' r', (multiSysUnarmed ranks peerCounts).runFrom ms evs = some ms' ∧
      ms'.reqs[idx]? = some r' ∧
      HandedOff r' ∧
      (multiStep ms' (.recycleSrc idx)).isSome = true ∧
      (multiStep ms' (.nextRequest idx)).isSome = true :=
  multiInv_progress (reachable_unarmed_multiInv hvalid h) hreq

/-! ## Concrete multi-transfer traces and write-after-release mutant -/

/-- Two-pool producer trace (`ranks = [0, 1]`, `peerCounts = [1, 2]`): issues D2H
for pool 0 (1 peer) and pool 1 (2 peers), finishes `ExecutePush`, completes all
3 peer pushes onto the wire, and publishes `done_sending`. -/
def sendProducer2 : List Ev :=
  [ .send (.d2hIssue 0 true), .send (.d2hIssue 1 true), .send .execFinish,
    .send (.d2hDone 0 true), .send (.d2hDone 1 true),
    .send (.pushDone 0 true), .send (.pushDone 1 true), .send (.pushDone 1 true),
    .send .publish ]

/-- Two-pool out-of-order aliased-page consumer trace (`ranks = [0, 1]`,
`peerCounts = [1, 2]`): pool 1 (rank 1, 2 streams) lands all streams and calls
`onPoolReceived 1` before pool 0 (rank 0, 1 stream) lands; Wave 0 H2D for pool 0
then runs before Wave 1 H2D for pool 1, and `publish` reports `done_recving`
with `dstHbm = [.kv 0, .kv 1]`. -/
def recvConsumer2OutOfOrder : List Ev :=
  [ .recv .pushBegin,
    .landStream 1, .landStream 1, .recv (.onPoolReceived 1),
    .landStream 0, .recv (.onPoolReceived 0),
    .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true),
    .recv (.h2dIssue 1 true), .recv (.h2dReady 1), .recv (.h2dDone 1 true),
    .recv (.pushEnd true), .recv .publish ]

/-- Multi-request trace: transfer $R_0$ starts on 2 aliased pools (`ranks = [0, 1]`,
`peerCounts = [1, 2]`), cancels mid-flight while pool 0 D2H is in flight, drains
its in-flight op, releases both staging leases, and publishes failed outcomes
(`HandedOff`); `.nextRequest 0` recycles the buffers to `.junk` and spawns
transfer $R_1$, which completes the out-of-order aliased-page transfer and
publishes `done_recving` with `dstHbm = [.kv 0, .kv 1]`. -/
theorem trace_multi_request :
    ((multiSys [0, 1] [1, 2]).run
      (([ .send (.d2hIssue 0 true), .send .cancel, .recv .cancel,
          .send (.d2hIssue 1 false), .send (.d2hDone 0 true),
          .send .publish, .recv .publish ].map (MultiEv.reqStep 0)) ++
       [.nextRequest 0] ++
       ((sendProducer2 ++ recvConsumer2OutOfOrder).map (MultiEv.reqStep 1)))).map
      (fun ms => ms.reqs.map (fun r => (r.recv.published, r.dstHbm))) =
    some [(some false, [.junk, .junk]), (some true, [.kv 0, .kv 1])] := by
  decide

/-- Overlapped multi-request trace: transfer $R_0$ finishes its send side
(`sendProducer2`), `.recycleSrc 0` immediately recycles $R_0$'s source HBM and
sender staging (overwriting them with `.junk`) and starts transfer $R_1$ while
$R_0$'s receive side has not even started yet; $R_1$'s send side
(`sendProducer2`) and $R_0$'s out-of-order aliased-page receive side
(`recvConsumer2OutOfOrder`) then run concurrently, followed by $R_1$'s receive
side (`recvConsumer2OutOfOrder`), and both $R_0$ and $R_1$ publish `done_recving`
with `dstHbm = [.kv 0, .kv 1]`. -/
theorem trace_overlapped_requests :
    ((multiSys [0, 1] [1, 2]).run
      ((sendProducer2.map (MultiEv.reqStep 0)) ++
       [.recycleSrc 0] ++
       (sendProducer2.map (MultiEv.reqStep 1)) ++
       (recvConsumer2OutOfOrder.map (MultiEv.reqStep 0)) ++
       (recvConsumer2OutOfOrder.map (MultiEv.reqStep 1)))).map
      (fun ms => ms.reqs.map (fun r => (r.reclaimed, r.srcStaging, r.recv.published, r.dstHbm))) =
    some [(true, [.junk, .junk], some true, [.kv 0, .kv 1]),
          (false, [.kv 0, .kv 1], some true, [.kv 0, .kv 1])] := by
  decide

/-- Unarmed multi-request trace (`multiSysUnarmed`, `ReshardCoordinator::ExecutePoolReshard`):
both the initial transfer $R_0$ and the recycled transfer $R_1$ spawned by
`.recycleSrc 0` start with `armed = false`, reject premature sender D2H before
`.armRecv`, and complete once `.armRecv` runs. -/
theorem trace_unarmed_multi_request :
    let sysU := multiSysUnarmed [0] [1]
    let send1 : List Ev :=
      [ .armRecv, .send (.d2hIssue 0 true), .send .execFinish,
        .send (.d2hDone 0 true), .send (.pushDone 0 true), .send .publish ]
    let recv1 : List Ev :=
      [ .recv .pushBegin, .landStream 0, .recv (.onPoolReceived 0),
        .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true),
        .recv (.pushEnd true), .recv .publish ]
    -- R_0 rejects D2H before .armRecv:
    sysU.run [.reqStep 0 (.send (.d2hIssue 0 true))] = none ∧
    -- After R_0 finishes its send side and .recycleSrc 0 spawns R_1, R_1 also
    -- starts unarmed and rejects D2H before .reqStep 1 .armRecv:
    sysU.run ((send1.map (MultiEv.reqStep 0)) ++
              [.recycleSrc 0, .reqStep 1 (.send (.d2hIssue 0 true))]) = none ∧
    -- Once R_1 runs .armRecv, both R_0 and R_1 complete and publish [.kv 0]:
    (sysU.run ((send1.map (MultiEv.reqStep 0)) ++
               [.recycleSrc 0] ++
               (send1.map (MultiEv.reqStep 1)) ++
               (recv1.map (MultiEv.reqStep 0)) ++
               (recv1.map (MultiEv.reqStep 1)))).map
      (fun ms => ms.reqs.map (fun r => (r.armed, r.recv.published, r.dstHbm))) =
    some [(true, some true, [.kv 0]), (true, some true, [.kv 0])] := by
  decide

/-- Mutant step: sender staging lease is released at `execFinish` while D2H is
still in flight (`hasStaging := false` before `d2hDone`), so the subsequent
`d2hDone 0 true` writes `srcStaging` after release (`wroteReleased = true`). -/
def stepReleaseStagingEarly (s : Pipeline) : Ev → Option Pipeline
  | .send .execFinish =>
    (step s (.send .execFinish)).map fun s' =>
      { s' with send := { s'.send with life := { s'.send.life with hasStaging := false } } }
  | e => step s e

def multiStepReleaseStagingEarly (ms : MultiState) : MultiEv → Option MultiState
  | .reqStep idx e =>
    match ms.reqs[idx]? with
    | none => none
    | some r =>
      (stepReleaseStagingEarly r e).map fun r' =>
        let reqs' := ms.reqs.set idx r'
        if !isRecycleEv e && wroteReleased r r' then
          { ms with reqs := reqs'.map corruptBuffers }
        else
          { ms with reqs := reqs' }
  | ev => multiStep ms ev

def multiSysReleaseStagingEarly (ranks peerCounts : List Nat) : System MultiState MultiEv :=
  ⟨multiInit ranks peerCounts true, multiStepReleaseStagingEarly⟩

/-- Mutant check: if a transfer releases `srcStaging` at `execFinish` while its
D2H copy is still in flight, the straggling `d2hDone 0 true` commits a
write-after-release (`wroteReleased = true`), triggering `corruptBuffers` in
`multiStep` and clobbering every transfer's buffers to `.junk`. -/
theorem trace_mutant_write_after_release_corrupts_all :
    ((multiSysReleaseStagingEarly [0] [1]).run
      [ .reqStep 0 (.send (.d2hIssue 0 true)),
        .reqStep 0 (.send .execFinish),
        .reqStep 0 (.send (.d2hDone 0 true)) ]).map
      (fun ms => ms.reqs.map (fun r => (r.srcHbm, r.srcStaging, r.dstStaging, r.dstHbm))) =
    some [([.junk], [.junk], [.junk], [.junk])] := by
  decide

end TpuSyncVerify.Transfer.Reshard.Pipeline

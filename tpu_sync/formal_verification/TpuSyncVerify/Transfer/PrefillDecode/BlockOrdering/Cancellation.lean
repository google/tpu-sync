import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.CopyPlan

/-! # `BlockOrdering`: algebraic dual-permutation cancellation

Part of `TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

namespace BlockOrdering

/-! ## 4. Algebraic Dual-Permutation Cancellation (`d2hStage` → `landStage` → `h2dStage`) -/

/-- A single KV block cell at `(layer, block)`. -/
inductive BlockCell where
  | blank
  | kv (layer : Nat) (block : Nat)
  | junk
  deriving Repr, DecidableEq

/-- Layer `l` of prefill HBM with `numBlocks` blocks `[.kv l 0, …, .kv l (numBlocks - 1)]`. -/
def goodBlockLayer (numBlocks l : Nat) : List BlockCell :=
  (List.range numBlocks).map (BlockCell.kv l)

def blankBlockLayer (n : Nat) : List BlockCell :=
  List.replicate n .blank

def junkBlockLayer (n : Nat) : List BlockCell :=
  List.replicate n .junk

theorem length_goodBlockLayer (numBlocks l : Nat) :
    (goodBlockLayer numBlocks l).length = numBlocks := by
  simp [goodBlockLayer]

theorem getD_goodBlockLayer {numBlocks l b : Nat} (hb : b < numBlocks) :
    (goodBlockLayer numBlocks l).getD b .junk = .kv l b := by
  simp [List.getD_eq_getElem?_getD, goodBlockLayer, hb]

theorem getElem?_blankBlockLayer {n b : Nat} (hb : b < n) :
    (blankBlockLayer n)[b]? = some .blank := by
  simp [blankBlockLayer, hb]

/-- Stage 1 (Producer D2H): coalesced gather from `goodBlockLayer numBlocks l`
into contiguous producer host staging `0 … k-1` (`TransferSendSession::StartPush`). -/
def d2hStage (l numBlocks : Nat) (plan : CopyPlan) : List BlockCell :=
  execCoalesced .junk (goodBlockLayer numBlocks l) (junkBlockLayer plan.numBlocks) plan.d2hSpec

/-- Stage 2 (Network H2H landing): `BlockTransport::HandleIncomingPush`
(`ForEachPayload`) lands producer host blocks `0 … k-1` into consumer
`transport_host_block_ids`. -/
def landStage (l numBlocks numHostBlocks : Nat) (plan : CopyPlan) : List BlockCell :=
  execElementwise .junk (d2hStage l numBlocks plan) (blankBlockLayer numHostBlocks) plan.h2hPairs

/-- Stage 3 (Consumer H2D): coalesced scatter from consumer `h2d_host_block_ids`
into consumer device `h2d_local_block_ids` (`BuildLoadCopyPlan` and
`base_->H2dSyncDispatch` in `TransferReceiveSession::DispatchLayerH2d`). -/
def h2dStage (l numBlocks numHostBlocks : Nat) (plan : CopyPlan) : List BlockCell :=
  execCoalesced .junk (landStage l numBlocks numHostBlocks plan) (blankBlockLayer numBlocks) plan.h2dSpec

/-- Well-formedness of `triples` for an `(numBlocks, numHostBlocks)` transfer:
1. All remote and local block IDs are in `< numBlocks`, and all host block IDs
   are in `< numHostBlocks`.
2. `triples.Nodup` (satisfied whenever remote block IDs are distinct, as checked
   by `TransferSendSession::ValidateRequestedBlocksLocked` at
   `if (!seen.insert(block_id).second)`).
3. Host block assignment is injective on distinct triples (`AllocateStagingForLoad`
   assigns distinct staged blocks to distinct local blocks, and requested pairs
   are distinct).
4. Duplicate local block IDs map to the same host block ID
   (`plan.h2d_host_block_ids.back() != host_bid` check in `BuildLoadCopyPlan`). -/
structure ValidTriples (numBlocks numHostBlocks : Nat) (triples : List BlockTriple) : Prop where
  remote_lt : ∀ t ∈ triples, t.remote < numBlocks
  local_lt : ∀ t ∈ triples, t.local_ < numBlocks
  host_lt : ∀ t ∈ triples, t.host < numHostBlocks
  nodup : triples.Nodup
  host_inj : ∀ t₁ ∈ triples, ∀ t₂ ∈ triples, t₁.host = t₂.host → t₁ = t₂
  local_consistent : ∀ t₁ ∈ triples, ∀ t₂ ∈ triples, t₁.local_ = t₂.local_ → t₁.host = t₂.host

/-- Stage 1 correctness: for every `(t, j) ∈ enumFrom 0 plan.remoteSorted`,
producer staging slot `j` receives `.kv l t.remote`. -/
theorem d2hStage_get_enum {l numBlocks : Nat} {triples : List BlockTriple}
    (hv : ValidTriples numBlocks numHostBlocks triples)
    {t : BlockTriple} {j : Nat}
    (hj : (t, j) ∈ enumFrom 0 (buildCopyPlanFromTriples triples).remoteSorted) :
    (d2hStage l numBlocks (buildCopyPlanFromTriples triples))[j]? =
      some (.kv l t.remote) := by
  let plan := buildCopyPlanFromTriples triples
  have hspec : plan.d2hSpec = buildCoalescedSpec plan.d2hPairs := rfl
  change (execCoalesced .junk (goodBlockLayer numBlocks l) (junkBlockLayer plan.numBlocks) plan.d2hSpec)[j]? = _
  rw [hspec, execCoalesced_buildCoalescedSpec]
  have ht_mem : t ∈ triples := by
    have := fst_mem_of_mem_enumFrom hj
    exact (mem_sortBy _ _ _).mp this
  have hj_lt : j < (junkBlockLayer plan.numBlocks).length := by
    have hlt := snd_lt_of_mem_enumFrom hj
    simp only [junkBlockLayer, List.length_replicate, CopyPlan.numBlocks, plan,
      buildCopyPlanFromTriples, length_sortBy] at hlt ⊢
    omega
  have hmem_pair : (t.remote, j) ∈ plan.d2hPairs := by
    simp only [plan, buildCopyPlanFromTriples, List.mem_map]
    exact ⟨(t, j), hj, rfl⟩
  rw [getElem?_execElementwise_of_mem .junk _ _ plan.d2hPairs t.remote j hmem_pair hj_lt]
  · rw [getD_goodBlockLayer (hv.remote_lt t ht_mem)]
  · intro p hp hp2
    simp only [plan, buildCopyPlanFromTriples, List.mem_map] at hp
    rcases hp with ⟨⟨t', j'⟩, hj', rfl⟩
    simp only at hp2
    subst hp2
    have heq := enumFrom_snd_inj hj' hj
    subst heq
    rfl

/-- Stage 2 correctness (requested host block): for every `t ∈ triples`,
consumer host block `t.host` receives `.kv l t.remote`. -/
theorem landStage_get_requested {l numBlocks numHostBlocks : Nat}
    {triples : List BlockTriple}
    (hv : ValidTriples numBlocks numHostBlocks triples)
    {t : BlockTriple} (ht : t ∈ triples) :
    (landStage l numBlocks numHostBlocks (buildCopyPlanFromTriples triples))[t.host]? =
      some (.kv l t.remote) := by
  let plan := buildCopyPlanFromTriples triples
  have ht_sorted : t ∈ plan.remoteSorted := by
    simp only [plan, buildCopyPlanFromTriples, mem_sortBy]
    exact ht
  obtain ⟨j, hj⟩ := exists_mem_enumFrom 0 ht_sorted
  have hmem_pair : (j, t.host) ∈ plan.h2hPairs := by
    simp only [plan, buildCopyPlanFromTriples, List.mem_map]
    exact ⟨(t, j), hj, rfl⟩
  have hhost_lt : t.host < (blankBlockLayer numHostBlocks).length := by
    simp only [blankBlockLayer, List.length_replicate]
    exact hv.host_lt t ht
  change (execElementwise .junk (d2hStage l numBlocks plan) (blankBlockLayer numHostBlocks) plan.h2hPairs)[t.host]? = _
  rw [getElem?_execElementwise_of_mem .junk _ _ plan.h2hPairs j t.host hmem_pair hhost_lt]
  · rw [List.getD_eq_getElem?_getD, d2hStage_get_enum hv hj]
    rfl
  · intro p hp hp2
    simp only [plan, buildCopyPlanFromTriples, List.mem_map] at hp
    rcases hp with ⟨⟨t', j'⟩, hj', rfl⟩
    simp only at hp2
    have ht'_mem : t' ∈ triples :=
      (mem_sortBy _ _ _).mp (fst_mem_of_mem_enumFrom hj')
    have hteq : t' = t := hv.host_inj t' ht'_mem t ht hp2
    subst hteq
    have hnd_sorted : plan.remoteSorted.Nodup := nodup_sortBy hv.nodup
    exact enumFrom_fst_inj hnd_sorted hj' hj

/-- Stage 2 frame (untouched host block): any consumer host block `h < numHostBlocks`
not in `triples.map (·.host)` remains `.blank`
(`KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock`). -/
theorem landStage_get_untouched {l numBlocks numHostBlocks : Nat}
    {triples : List BlockTriple} {h : Nat} (hh : h < numHostBlocks)
    (huntouched : ∀ t ∈ triples, t.host ≠ h) :
    (landStage l numBlocks numHostBlocks (buildCopyPlanFromTriples triples))[h]? =
      some .blank := by
  let plan := buildCopyPlanFromTriples triples
  change (execElementwise .junk (d2hStage l numBlocks plan) (blankBlockLayer numHostBlocks) plan.h2hPairs)[h]? = _
  rw [getElem?_execElementwise_of_not_mem]
  · exact getElem?_blankBlockLayer hh
  · intro p hp
    simp only [plan, buildCopyPlanFromTriples, List.mem_map] at hp
    rcases hp with ⟨⟨t', j'⟩, hj', rfl⟩
    have ht'_mem : t' ∈ triples :=
      (mem_sortBy _ _ _).mp (fst_mem_of_mem_enumFrom hj')
    exact huntouched t' ht'_mem

/-- **Stage 3 end-to-end correctness (requested local block):** for every
`t ∈ triples`, consumer device block `t.local_` receives `.kv l t.remote`,
cancelling the intermediate $\pi_{\text{remote}}$ and $\pi_{\text{local}}$
permutations and contiguous-run coalescing. -/
theorem h2dStage_get_requested {l numBlocks numHostBlocks : Nat}
    {triples : List BlockTriple}
    (hv : ValidTriples numBlocks numHostBlocks triples)
    {t : BlockTriple} (ht : t ∈ triples) :
    (h2dStage l numBlocks numHostBlocks (buildCopyPlanFromTriples triples))[t.local_]? =
      some (.kv l t.remote) := by
  let plan := buildCopyPlanFromTriples triples
  have hspec : plan.h2dSpec = buildCoalescedSpec plan.h2dPairs := rfl
  change (execCoalesced .junk (landStage l numBlocks numHostBlocks plan) (blankBlockLayer numBlocks) plan.h2dSpec)[t.local_]? = _
  rw [hspec, execCoalesced_buildCoalescedSpec]
  have hmem_pair : (t.host, t.local_) ∈ plan.h2dPairs := by
    simp only [plan, buildCopyPlanFromTriples, mem_dedupAdj, List.mem_map, mem_sortBy]
    exact ⟨t, ht, rfl⟩
  have hlocal_lt : t.local_ < (blankBlockLayer numBlocks).length := by
    simp only [blankBlockLayer, List.length_replicate]
    exact hv.local_lt t ht
  rw [getElem?_execElementwise_of_mem .junk _ _ plan.h2dPairs t.host t.local_ hmem_pair hlocal_lt]
  · rw [List.getD_eq_getElem?_getD, landStage_get_requested hv ht]
    rfl
  · intro p hp hp2
    simp only [plan, buildCopyPlanFromTriples, mem_dedupAdj, List.mem_map, mem_sortBy] at hp
    rcases hp with ⟨t', ht', rfl⟩
    exact hv.local_consistent t' ht' t ht hp2

/-- **Stage 3 frame (untouched local block):** any consumer device block
`b < numBlocks` not in `triples.map (·.local_)` remains `.blank`
(`KVCacheManagerWithTransferTest.LocalOrchestratedTransfer`,
`test_non_contiguous_blocks`, `test_large_complex_non_contiguous_and_reorder`). -/
theorem h2dStage_get_untouched {l numBlocks numHostBlocks : Nat}
    {triples : List BlockTriple} {b : Nat} (hb : b < numBlocks)
    (huntouched : ∀ t ∈ triples, t.local_ ≠ b) :
    (h2dStage l numBlocks numHostBlocks (buildCopyPlanFromTriples triples))[b]? =
      some .blank := by
  let plan := buildCopyPlanFromTriples triples
  have hspec : plan.h2dSpec = buildCoalescedSpec plan.h2dPairs := rfl
  change (execCoalesced .junk (landStage l numBlocks numHostBlocks plan) (blankBlockLayer numBlocks) plan.h2dSpec)[b]? = _
  rw [hspec, execCoalesced_buildCoalescedSpec, getElem?_execElementwise_of_not_mem]
  · exact getElem?_blankBlockLayer hb
  · intro p hp
    simp only [plan, buildCopyPlanFromTriples, mem_dedupAdj, List.mem_map, mem_sortBy] at hp
    rcases hp with ⟨t', ht', rfl⟩
    exact huntouched t' ht'


end BlockOrdering

end TpuSyncVerify.Transfer.PrefillDecode

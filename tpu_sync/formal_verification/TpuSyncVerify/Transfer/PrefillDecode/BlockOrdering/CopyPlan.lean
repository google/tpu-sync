import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Coalescing

/-! # `BlockOrdering`: consumer staging allocation and dual-permutation `CopyPlan`

Part of `TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

namespace BlockOrdering

/-! ## 3. Consumer Staging Allocation & Dual-Permutation `CopyPlan` (`AllocateStagingForLoad`, `BuildLoadCopyPlan`) -/

/-- How consumer host staging blocks are chosen in
`TransferReceiveSession::AllocateStagingForLoad`:
- `allocator stagedBlocks`: deduplicates `local_block_ids` and assigns blocks
  drawn from `staging_allocator_->Acquire(unique_local_bids.size())`.
- `customHost hostBlocks`: uses caller-specified `local_host_block_ids` directly
  without touching `staging_allocator_` (`if (local_host_block_ids.has_value())`). -/
inductive StagingMode where
  | allocator (stagedBlocks : List Nat)
  | customHost (hostBlocks : List Nat)
  deriving Repr, DecidableEq

/-- Look up or assign a host block for each local block ID, matching the
`local_to_host` loop in `TransferReceiveSession::AllocateStagingForLoad`. -/
def assignHostBlocksAux (stagedBlocks : List Nat) :
    List Nat → List (Nat × Nat) → Nat → Option (List Nat)
  | [], _, _ => some []
  | localBid :: rest, map, nextIdx =>
    match map.lookup localBid with
    | some hostBid =>
      (assignHostBlocksAux stagedBlocks rest map nextIdx).map (hostBid :: ·)
    | none =>
      match stagedBlocks[nextIdx]? with
      | some hostBid =>
        (assignHostBlocksAux stagedBlocks rest ((localBid, hostBid) :: map) (nextIdx + 1)).map
          (hostBid :: ·)
      | none => none

/-- `TransferReceiveSession::AllocateStagingForLoad`. -/
def allocateStagingForLoad (localBlocks : List Nat) : StagingMode → Option (List Nat)
  | .customHost hostBlocks =>
    if hostBlocks.length = localBlocks.length then some hostBlocks else none
  | .allocator stagedBlocks =>
    assignHostBlocksAux stagedBlocks localBlocks [] 0

/-- A requested `(remote_block_id, local_block_id, local_host_block_id)` triple
at one position in `StartRead`. -/
structure BlockTriple where
  remote : Nat
  local_ : Nat
  host : Nat
  deriving Repr, DecidableEq

/-- Stable insertion sort helper (`std::stable_sort` in
`TransferReceiveSession::BuildLoadCopyPlan`). -/
def insertBy (le : α → α → Bool) (x : α) : List α → List α
  | [] => [x]
  | y :: ys => if le x y then x :: y :: ys else y :: insertBy le x ys

def sortBy (le : α → α → Bool) : List α → List α
  | [] => []
  | x :: xs => insertBy le x (sortBy le xs)

theorem mem_insertBy (le : α → α → Bool) (x z : α) (ys : List α) :
    z ∈ insertBy le x ys ↔ z = x ∨ z ∈ ys := by
  induction ys with
  | nil => simp [insertBy]
  | cons y ys ih =>
    simp only [insertBy]
    split
    · simp [List.mem_cons]
    · simp only [List.mem_cons, ih]
      exact ⟨fun h => h.elim (fun h1 => Or.inr (Or.inl h1)) (fun h2 => h2.elim Or.inl (fun h3 => Or.inr (Or.inr h3))),
             fun h => h.elim (fun h1 => Or.inr (Or.inl h1)) (fun h2 => h2.elim Or.inl (fun h3 => Or.inr (Or.inr h3)))⟩

theorem mem_sortBy (le : α → α → Bool) (z : α) (xs : List α) :
    z ∈ sortBy le xs ↔ z ∈ xs := by
  induction xs with
  | nil => simp [sortBy]
  | cons x xs ih =>
    simp [sortBy, mem_insertBy, ih]

theorem length_insertBy (le : α → α → Bool) (x : α) (ys : List α) :
    (insertBy le x ys).length = ys.length + 1 := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    simp only [insertBy]
    split <;> simp [ih]

theorem length_sortBy (le : α → α → Bool) (xs : List α) :
    (sortBy le xs).length = xs.length := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp [sortBy, length_insertBy, ih]

theorem nodup_insertBy {le : α → α → Bool} {x : α} {ys : List α}
    (hx : x ∉ ys) (hnd : ys.Nodup) : (insertBy le x ys).Nodup := by
  induction ys with
  | nil => simp [insertBy]
  | cons y ys ih =>
    rw [List.nodup_cons] at hnd
    have hxy : x ≠ y := fun h => hx (by simp [h])
    have hxys : x ∉ ys := fun h => hx (by simp [h])
    simp only [insertBy]
    split
    · rw [List.nodup_cons, List.nodup_cons]
      exact ⟨by simp [hxy, hxys], hnd⟩
    · rw [List.nodup_cons, mem_insertBy]
      refine ⟨?_, ih hxys hnd.2⟩
      rintro (rfl | hy)
      · exact hxy rfl
      · exact hnd.1 hy

theorem nodup_sortBy {le : α → α → Bool} {xs : List α} (hnd : xs.Nodup) :
    (sortBy le xs).Nodup := by
  induction xs with
  | nil => simp [sortBy]
  | cons x xs ih =>
    rw [List.nodup_cons] at hnd
    apply nodup_insertBy _ (ih hnd.2)
    rw [mem_sortBy]
    exact hnd.1

/-- Enumerate a list starting from index `idx` (`(x₀, idx), (x₁, idx + 1), …`). -/
def enumFrom (idx : Nat) : List α → List (α × Nat)
  | [] => []
  | x :: xs => (x, idx) :: enumFrom (idx + 1) xs

theorem length_enumFrom (idx : Nat) (xs : List α) :
    (enumFrom idx xs).length = xs.length := by
  induction xs generalizing idx with
  | nil => rfl
  | cons x xs ih => simp [enumFrom, ih]

theorem snd_ge_of_mem_enumFrom {x : α} {j idx : Nat} {xs : List α}
    (h : (x, j) ∈ enumFrom idx xs) : idx ≤ j := by
  induction xs generalizing idx with
  | nil => cases h
  | cons y ys ih =>
    rcases List.mem_cons.mp h with heq | htail
    · cases heq; exact Nat.le_refl _
    · have := ih htail; omega

theorem snd_lt_of_mem_enumFrom {x : α} {j idx : Nat} {xs : List α}
    (h : (x, j) ∈ enumFrom idx xs) : j < idx + xs.length := by
  induction xs generalizing idx with
  | nil => cases h
  | cons y ys ih =>
    rcases List.mem_cons.mp h with heq | htail
    · cases heq; simp
    · have := ih htail; simp; omega

theorem fst_mem_of_mem_enumFrom {x : α} {j idx : Nat} {xs : List α}
    (h : (x, j) ∈ enumFrom idx xs) : x ∈ xs := by
  induction xs generalizing idx with
  | nil => cases h
  | cons y ys ih =>
    rcases List.mem_cons.mp h with heq | htail
    · cases heq; exact .head _
    · exact .tail _ (ih htail)

theorem exists_mem_enumFrom {x : α} {xs : List α} (idx : Nat) (hx : x ∈ xs) :
    ∃ j, (x, j) ∈ enumFrom idx xs := by
  induction xs generalizing idx with
  | nil => cases hx
  | cons y ys ih =>
    rcases List.mem_cons.mp hx with rfl | htail
    · exact ⟨idx, .head _⟩
    · obtain ⟨j, hj⟩ := ih (idx + 1) htail
      exact ⟨j, .tail _ hj⟩

theorem enumFrom_snd_inj {x₁ x₂ : α} {j idx : Nat} {xs : List α}
    (h₁ : (x₁, j) ∈ enumFrom idx xs) (h₂ : (x₂, j) ∈ enumFrom idx xs) :
    x₁ = x₂ := by
  induction xs generalizing idx with
  | nil => cases h₁
  | cons y ys ih =>
    rcases List.mem_cons.mp h₁ with heq₁ | htail₁ <;>
      rcases List.mem_cons.mp h₂ with heq₂ | htail₂
    · cases heq₁; cases heq₂; rfl
    · cases heq₁
      have := snd_ge_of_mem_enumFrom htail₂
      omega
    · cases heq₂
      have := snd_ge_of_mem_enumFrom htail₁
      omega
    · exact ih htail₁ htail₂

theorem enumFrom_fst_inj [DecidableEq α] {x : α} {j₁ j₂ idx : Nat} {xs : List α}
    (hnd : xs.Nodup)
    (h₁ : (x, j₁) ∈ enumFrom idx xs) (h₂ : (x, j₂) ∈ enumFrom idx xs) :
    j₁ = j₂ := by
  induction xs generalizing idx with
  | nil => cases h₁
  | cons y ys ih =>
    rw [List.nodup_cons] at hnd
    rcases List.mem_cons.mp h₁ with heq₁ | htail₁ <;>
      rcases List.mem_cons.mp h₂ with heq₂ | htail₂
    · cases heq₁; cases heq₂; rfl
    · cases heq₁
      exact (hnd.1 (fst_mem_of_mem_enumFrom htail₂)).elim
    · cases heq₂
      exact (hnd.1 (fst_mem_of_mem_enumFrom htail₁)).elim
    · exact ih hnd.2 htail₁ htail₂

/-- Deduplicate adjacent equal pairs: `TransferReceiveSession::BuildLoadCopyPlan`
appends a pair only
`if (plan.h2d_local_block_ids.empty() || plan.h2d_local_block_ids.back() != local_bid)`. -/
def dedupAdj [DecidableEq α] : List α → List α
  | [] => []
  | [x] => [x]
  | x :: y :: xs => if x = y then dedupAdj (y :: xs) else x :: dedupAdj (y :: xs)

theorem mem_dedupAdj [DecidableEq α] (z : α) (xs : List α) :
    z ∈ dedupAdj xs ↔ z ∈ xs := by
  induction xs with
  | nil => rfl
  | cons x rest ih =>
    cases rest with
    | nil => rfl
    | cons y ys =>
      simp only [dedupAdj]
      split
      · rename_i hxy
        subst hxy
        rw [ih]
        simp [List.mem_cons]
      · simp [List.mem_cons, ih]

/-- Zip three equal-length block lists into `List BlockTriple`. -/
def zipTriples : List Nat → List Nat → List Nat → List BlockTriple
  | r :: rs, d :: ds, h :: hs => ⟨r, d, h⟩ :: zipTriples rs ds hs
  | _, _, _ => []

/-- The compiled dual-permutation copy plan (`CopyPlan` in
`TransferReceiveSession::BuildLoadCopyPlan` coupled with producer
`TransferSendSession::StartPush` staging). -/
structure CopyPlan where
  triples : List BlockTriple
  /-- `remote_order` in `BuildLoadCopyPlan`: `triples` sorted by `remote_block_ids`. -/
  remoteSorted : List BlockTriple
  /-- `local_order` in `BuildLoadCopyPlan`: `triples` sorted by `local_block_ids`. -/
  localSorted : List BlockTriple
  /-- `producer_remote_block_ids` sent in `PullStream` (`plan.producer_remote_block_ids.push_back(remote_block_ids[idx])`). -/
  producerRemoteBlocks : List Nat
  /-- `transport_host_block_ids` where `BlockTransport` lands blocks (`plan.transport_host_block_ids.push_back(local_host_block_ids[idx])`). -/
  transportHostBlocks : List Nat
  /-- Producer D2H pairs `(producer_remote_block_ids[j], j)` in `StartPush`. -/
  d2hPairs : List (Nat × Nat)
  /-- Network H2H pairs `(j, transport_host_block_ids[j])` in `StartPush`. -/
  h2hPairs : List (Nat × Nat)
  /-- Consumer H2D pairs `(h2d_host_block_ids[j], h2d_local_block_ids[j])` in `BuildLoadCopyPlan`. -/
  h2dPairs : List (Nat × Nat)
  /-- Coalesced producer D2H `CopySpec` (`BuildCoalescedCopySpec(requested_block_ids, dst_block_ids)` in `StartPush`). -/
  d2hSpec : List CopyRun
  /-- Coalesced consumer H2D `CopySpec` (`TransferSendSession::BuildCoalescedCopySpec(plan.h2d_host_block_ids, plan.h2d_local_block_ids)` in `BuildLoadCopyPlan`). -/
  h2dSpec : List CopyRun
  deriving Repr, DecidableEq

namespace CopyPlan

def numBlocks (p : CopyPlan) : Nat := p.triples.length

end CopyPlan

/-- Build a `CopyPlan` from `triples`, executing the exact dual-permutation
sorting and run-coalescing of `TransferReceiveSession::BuildLoadCopyPlan`
and `TransferSendSession::StartPush`. -/
def buildCopyPlanFromTriples (triples : List BlockTriple) : CopyPlan :=
  let remoteSorted := sortBy (fun a b => a.remote ≤ b.remote) triples
  let localSorted := sortBy (fun a b => a.local_ ≤ b.local_) triples
  let enumRemote := enumFrom 0 remoteSorted
  let d2hPairs := enumRemote.map (fun (t, j) => (t.remote, j))
  let h2hPairs := enumRemote.map (fun (t, j) => (j, t.host))
  let h2dRaw := localSorted.map (fun t => (t.host, t.local_))
  let h2dPairs := dedupAdj h2dRaw
  { triples := triples,
    remoteSorted := remoteSorted,
    localSorted := localSorted,
    producerRemoteBlocks := remoteSorted.map (·.remote),
    transportHostBlocks := remoteSorted.map (·.host),
    d2hPairs := d2hPairs,
    h2hPairs := h2hPairs,
    h2dPairs := h2dPairs,
    d2hSpec := buildCoalescedSpec d2hPairs,
    h2dSpec := buildCoalescedSpec h2dPairs }

/-- `TransferReceiveSession::BuildLoadCopyPlan`. -/
def buildLoadCopyPlan (remoteBlocks localBlocks hostBlocks : List Nat) : CopyPlan :=
  buildCopyPlanFromTriples (zipTriples remoteBlocks localBlocks hostBlocks)


end BlockOrdering

end TpuSyncVerify.Transfer.PrefillDecode

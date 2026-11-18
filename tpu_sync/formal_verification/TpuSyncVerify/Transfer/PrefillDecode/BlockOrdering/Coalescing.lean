import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Validation

/-! # `BlockOrdering`: contiguous-run DMA coalescing

Part of `TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

namespace BlockOrdering

/-! ## 2. Contiguous-Run DMA Coalescing (`TransferSendSession::BuildCoalescedCopySpec`) -/

/-- One contiguous `(src_offset, dst_offset, size)` DMA copy chunk produced by
`TransferSendSession::BuildCoalescedCopySpec`. -/
structure CopyRun where
  srcOffset : Nat
  dstOffset : Nat
  size : Nat
  deriving Repr, DecidableEq

/-- Inner coalescing loop in `TransferSendSession::BuildCoalescedCopySpec`
(`while (end < n && src_block_ids[end] == src_block_ids[end - 1] + 1 && dst_block_ids[end] == dst_block_ids[end - 1] + 1)`):
extends `cur` by 1
whenever the next `(s, d)` pair continues both contiguous runs
(`s = cur.srcOffset + cur.size` and `d = cur.dstOffset + cur.size`), and starts
a new run otherwise. -/
def coalesceAux (cur : CopyRun) : List (Nat × Nat) → List CopyRun
  | [] => [cur]
  | (s, d) :: rest =>
    if s = cur.srcOffset + cur.size ∧ d = cur.dstOffset + cur.size then
      coalesceAux { cur with size := cur.size + 1 } rest
    else
      cur :: coalesceAux { srcOffset := s, dstOffset := d, size := 1 } rest

/-- `TransferSendSession::BuildCoalescedCopySpec`. -/
def buildCoalescedSpec : List (Nat × Nat) → List CopyRun
  | [] => []
  | (s, d) :: rest => coalesceAux { srcOffset := s, dstOffset := d, size := 1 } rest

/-- Execute a single contiguous DMA run of `sz` blocks from `src[srcOff ..]`
into `dst[dstOff ..]`. -/
def execRun (default : α) (src dst : List α) (srcOff dstOff : Nat) : Nat → List α
  | 0 => dst
  | sz + 1 =>
    execRun default src (dst.set dstOff (src.getD srcOff default))
      (srcOff + 1) (dstOff + 1) sz

/-- Execute a coalesced `CopySpec` (`List CopyRun`) sequentially. -/
def execCoalesced (default : α) (src dst : List α) : List CopyRun → List α
  | [] => dst
  | r :: rs =>
    execCoalesced default src (execRun default src dst r.srcOffset r.dstOffset r.size) rs

/-- Execute an uncoalesced list of `(src_block, dst_block)` pairs one block at a time. -/
def execElementwise (default : α) (src dst : List α) : List (Nat × Nat) → List α
  | [] => dst
  | (s, d) :: rest =>
    execElementwise default src (dst.set d (src.getD s default)) rest

/-- Extending a contiguous DMA run by 1 block at the end equals running the
`sz`-block DMA run followed by setting `(srcOff + sz, dstOff + sz)`. -/
theorem execRun_succ_right (default : α) (src dst : List α) (srcOff dstOff sz : Nat) :
    execRun default src dst srcOff dstOff (sz + 1) =
      (execRun default src dst srcOff dstOff sz).set (dstOff + sz)
        (src.getD (srcOff + sz) default) := by
  induction sz generalizing dst srcOff dstOff with
  | zero => simp [execRun]
  | succ n ih =>
    rw [execRun, ih]
    have h1 : dstOff + 1 + n = dstOff + (n + 1) := by omega
    have h2 : srcOff + 1 + n = srcOff + (n + 1) := by omega
    rw [h1, h2]
    rfl

theorem execCoalesced_coalesceAux (default : α) (src dst : List α)
    (cur : CopyRun) (rest : List (Nat × Nat)) :
    execCoalesced default src dst (coalesceAux cur rest) =
      execElementwise default src
        (execRun default src dst cur.srcOffset cur.dstOffset cur.size) rest := by
  induction rest generalizing dst cur with
  | nil => rfl
  | cons p tail ih =>
    rcases p with ⟨s, d⟩
    simp only [coalesceAux]
    split
    · rename_i hcontig
      rcases hcontig with ⟨hs, hd⟩
      rw [ih]
      simp only [execRun_succ_right, ← hs, ← hd, execElementwise]
    · simp only [execCoalesced, ih, execRun, execElementwise]

/-- **Coalescing equivalence (`TransferSendSession::BuildCoalescedCopySpec`):**
executing the coalesced DMA runs produced by `buildCoalescedSpec` is
extensionally identical on every source and destination buffer to executing
block-by-block elementwise copies. -/
theorem execCoalesced_buildCoalescedSpec (default : α) (src dst : List α)
    (pairs : List (Nat × Nat)) :
    execCoalesced default src dst (buildCoalescedSpec pairs) =
      execElementwise default src dst pairs := by
  cases pairs with
  | nil => rfl
  | cons p rest =>
    rcases p with ⟨s, d⟩
    simp only [buildCoalescedSpec, execCoalesced_coalesceAux, execRun, execElementwise]

/-! ### Fundamental lemmas for `execElementwise` -/

theorem length_execElementwise (default : α) (src dst : List α)
    (pairs : List (Nat × Nat)) :
    (execElementwise default src dst pairs).length = dst.length := by
  induction pairs generalizing dst with
  | nil => rfl
  | cons p rest ih =>
    rcases p with ⟨s, d⟩
    simp only [execElementwise, ih, List.length_set]

theorem length_execCoalesced (default : α) (src dst : List α)
    (pairs : List (Nat × Nat)) :
    (execCoalesced default src dst (buildCoalescedSpec pairs)).length = dst.length := by
  rw [execCoalesced_buildCoalescedSpec, length_execElementwise]

/-- Frame lemma: any destination block `b` not targeted by `pairs` retains its
initial value across `execElementwise`. -/
theorem getElem?_execElementwise_of_not_mem (default : α) (src dst : List α)
    (pairs : List (Nat × Nat)) (b : Nat) (hb : ∀ p ∈ pairs, p.2 ≠ b) :
    (execElementwise default src dst pairs)[b]? = dst[b]? := by
  induction pairs generalizing dst with
  | nil => rfl
  | cons p rest ih =>
    rcases p with ⟨s, d⟩
    simp only [execElementwise]
    rw [ih]
    · have hne : d ≠ b := hb (s, d) (.head _)
      exact List.getElem?_set_ne hne
    · intro q hq
      exact hb q (.tail _ hq)

/-- Target read-back lemma: if `(s, d) ∈ pairs`, `d < dst.length`, and every
pair in `pairs` targeting `d` has source `s`, then `execElementwise` writes
`src.getD s default` at destination block `d`. -/
theorem getElem?_execElementwise_of_mem (default : α) (src dst : List α)
    (pairs : List (Nat × Nat)) (s d : Nat)
    (hmem : (s, d) ∈ pairs) (hd : d < dst.length)
    (huniq : ∀ p ∈ pairs, p.2 = d → p.1 = s) :
    (execElementwise default src dst pairs)[d]? = some (src.getD s default) := by
  induction pairs generalizing dst with
  | nil => cases hmem
  | cons p rest ih =>
    rcases p with ⟨s₀, d₀⟩
    simp only [execElementwise]
    by_cases hrest : (s, d) ∈ rest
    · apply ih (dst.set d₀ (src.getD s₀ default)) hrest
      · rw [List.length_set]; exact hd
      · intro q hq hqd
        exact huniq q (.tail _ hq) hqd
    · rcases List.mem_cons.mp hmem with heq | hmem'
      · cases heq
        rw [getElem?_execElementwise_of_not_mem]
        · exact List.getElem?_set_self hd
        · intro q hq hqd
          have hqs : q.1 = s := huniq q (.tail _ hq) hqd
          rcases q with ⟨qs, qd⟩
          subst hqs hqd
          exact hrest hq
      · exact (hrest hmem').elim


end BlockOrdering

end TpuSyncVerify.Transfer.PrefillDecode

import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Model

/-! # `Pipeline`: facts about `good`, per-layer sets and `countTrue`

Part of `TpuSyncVerify.Transfer.PrefillDecode.Pipeline`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ## Facts about `good`, per-layer sets and `countTrue` -/

theorem good_length (n : Nat) : (good n).length = n := by simp [good]

theorem good_get {n k : Nat} (hk : k < n) : (good n)[k]? = some (.kv k) := by simp [good, hk]

theorem good_getD {n k : Nat} (hk : k < n) : (good n).getD k .junk = .kv k := by
  simp [List.getD_eq_getElem?_getD, good, hk]

theorem eq_good {l : List Cell} {n : Nat} (hlen : l.length = n)
    (hget : ∀ k < n, l[k]? = some (.kv k)) : l = good n := by
  apply List.ext_getElem?
  intro i
  by_cases hi : i < n
  · rw [hget i hi, good_get hi]
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by rw [good_length]; omega)]

/-- Reading a memory after `.kv k` was written at a valid index `k`. -/
theorem getElem?_set_kv {l : List Cell} {k j : Nat} (hk : k < l.length)
    (hj : j ≠ k → l[j]? = some (.kv j)) : (l.set k (.kv k))[j]? = some (.kv j) := by
  by_cases hjk : j = k
  · subst hjk; exact List.getElem?_set_self hk
  · rw [List.getElem?_set_ne (Ne.symm hjk)]; exact hj hjk

end Pipeline

end TpuSyncVerify.Transfer.PrefillDecode

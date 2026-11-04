/-!
# List and boolean-set helpers

Generic helper lemmas on `List` indexing, `List.set`, and `countTrue` used by
the layer-indexed transfer pipeline and multi-request models.
-/

namespace TpuSyncVerify

/-- Number of `true`s in a boolean list. -/
def countTrue : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + countTrue bs

theorem lt_length_of_getElem?_eq {α : Type} {l : List α} {i : Nat} {a : α}
    (h : l[i]? = some a) : i < l.length :=
  (List.getElem?_eq_some_iff.mp h).1

theorem getD_of_getElem?_eq_some {α : Type} {xs : List α} {i : Nat} {v d : α}
    (h : xs[i]? = some v) : xs.getD i d = v := by
  simp [List.getD, h]

theorem mem_set_cases {α : Type} {l : List α} {i : Nat} {x y : α}
    (h : y ∈ l.set i x) : y = x ∨ y ∈ l := by
  induction l generalizing i with
  | nil => simp at h
  | cons a as ih =>
    cases i with
    | zero =>
      simp only [List.set_cons_zero, List.mem_cons] at h
      rcases h with rfl | h
      · exact Or.inl rfl
      · exact Or.inr (List.mem_cons_of_mem a h)
    | succ i =>
      simp only [List.set_cons_succ, List.mem_cons] at h
      rcases h with rfl | h
      · exact Or.inr List.mem_cons_self
      · rcases ih h with rfl | h
        · exact Or.inl rfl
        · exact Or.inr (List.mem_cons_of_mem a h)

theorem getElem?_mem {α : Type} {l : List α} {i : Nat} {x : α}
    (h : l[i]? = some x) : x ∈ l := by
  induction l generalizing i with
  | nil => simp at h
  | cons a as ih =>
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h; exact List.mem_cons_self
    | succ i =>
      simp only [List.getElem?_cons_succ] at h
      exact List.mem_cons_of_mem a (ih h)

/-- Adding `i` to a set: membership afterwards is `i` or membership before. -/
theorem mem_of_set_true {L : List Bool} {i j : Nat} (h : (L.set i true)[j]? = some true) :
    j = i ∨ L[j]? = some true := by
  by_cases hji : j = i
  · exact Or.inl hji
  · right; rwa [List.getElem?_set_ne (Ne.symm hji)] at h

/-- Removing `i` from a set: membership afterwards implies membership before. -/
theorem mem_of_set_false {L : List Bool} {i j : Nat} (h : (L.set i false)[j]? = some true) :
    L[j]? = some true := by
  by_cases hji : j = i
  · subst hji
    by_cases hi : j < L.length
    · rw [List.getElem?_set_self hi] at h; cases h
    · rw [List.getElem?_eq_none (by rw [List.length_set]; omega)] at h; cases h
  · rwa [List.getElem?_set_ne (Ne.symm hji)] at h

theorem mem_of_set_false_ne {L : List Bool} {i j : Nat} (h : (L.set i false)[j]? = some true) :
    j ≠ i := by
  intro rfl
  by_cases hi : j < L.length
  · rw [List.getElem?_set_self hi] at h; cases h
  · rw [List.getElem?_eq_none (by rw [List.length_set]; omega)] at h; cases h

theorem set_true_self {L : List Bool} {i : Nat} (hi : L[i]? = some false) :
    (L.set i true)[i]? = some true :=
  List.getElem?_set_self (lt_length_of_getElem?_eq hi)

theorem set_true_of_mem {L : List Bool} {i j : Nat} (hi : L[i]? = some false)
    (hj : L[j]? = some true) : (L.set i true)[j]? = some true := by
  by_cases hji : j = i
  · subst hji; rw [hi] at hj; cases hj
  · rw [List.getElem?_set_ne (Ne.symm hji)]; exact hj

theorem false_of_not_true {l₁ l₂ : List Bool} {i : Nat} (hlen : l₁.length = l₂.length)
    (hi : l₁[i]? = some false) (himp : l₂[i]? = some true → False) : l₂[i]? = some false := by
  have hlt : i < l₂.length := by have := lt_length_of_getElem?_eq hi; omega
  have heq := List.getElem?_eq_getElem hlt
  cases hb : l₂[i]
  · rw [heq, hb]
  · rw [hb] at heq; exact (himp heq).elim

/-- No layer is in an empty set. -/
theorem not_mem_replicate_false {n i : Nat} (h : (List.replicate n false)[i]? = some true) :
    False := by
  simp only [List.getElem?_replicate] at h
  split at h <;> cases h

theorem countTrue_replicate_false : ∀ n, countTrue (List.replicate n false) = 0
  | 0 => rfl
  | n + 1 => by simp [List.replicate_succ, countTrue, countTrue_replicate_false n]

theorem countTrue_le_length : ∀ l : List Bool, countTrue l ≤ l.length
  | [] => Nat.le_refl _
  | b :: bs => by
    have := countTrue_le_length bs
    simp only [countTrue, List.length_cons]; split <;> omega

theorem countTrue_set_true : ∀ {l : List Bool} {i : Nat}, l[i]? = some false →
    countTrue (l.set i true) = countTrue l + 1
  | [], i, h => by simp at h
  | b :: bs, 0, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h; simp [countTrue]; omega
  | b :: bs, i + 1, h => by
    simp only [List.getElem?_cons_succ] at h
    simp only [List.set_cons_succ, countTrue, countTrue_set_true h]; omega

theorem countTrue_set_false : ∀ {l : List Bool} {i : Nat}, l[i]? = some true →
    countTrue (l.set i false) + 1 = countTrue l
  | [], i, h => by simp at h
  | b :: bs, 0, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h; simp [countTrue]; omega
  | b :: bs, i + 1, h => by
    simp only [List.getElem?_cons_succ] at h
    have ih := countTrue_set_false h
    simp only [List.set_cons_succ, countTrue]; omega

/-- A set with as many members as slots is full. -/
theorem all_true_of_countTrue_eq_length : ∀ {l : List Bool}, countTrue l = l.length →
    ∀ i, i < l.length → l[i]? = some true
  | [], _, i, hi => by simp at hi
  | b :: bs, h, i, hi => by
    have hle := countTrue_le_length bs
    simp only [countTrue, List.length_cons] at h
    cases b with
    | false => simp at h; omega
    | true =>
      cases i with
      | zero => simp
      | succ i =>
        simp only [List.getElem?_cons_succ]
        exact all_true_of_countTrue_eq_length (by simp at h; omega) i (by simp at hi; omega)

theorem exists_true_of_countTrue_pos : ∀ {l : List Bool}, 0 < countTrue l →
    ∃ i < l.length, l[i]? = some true
  | [], h => by simp [countTrue] at h
  | true :: bs, _ => ⟨0, by simp, rfl⟩
  | false :: bs, h => by
    simp [countTrue] at h
    obtain ⟨i, hi, hget⟩ := exists_true_of_countTrue_pos h
    exact ⟨i + 1, by simp; omega, hget⟩

theorem exists_false_lt : ∀ {l : List Bool} {k : Nat},
    k ≤ l.length → countTrue l < k → ∃ i < k, l[i]? = some false
  | _, 0, _, hcnt => by omega
  | [], k + 1, hk, _ => by simp at hk
  | false :: bs, k + 1, _, _ => ⟨0, by omega, rfl⟩
  | true :: bs, k + 1, hk, hcnt => by
    simp [countTrue] at hk hcnt
    obtain ⟨i, hi, hget⟩ := @exists_false_lt bs k (by omega) (by omega)
    exact ⟨i + 1, by omega, hget⟩

theorem exists_diff_of_countTrue_lt : ∀ {l₁ l₂ : List Bool},
    l₁.length = l₂.length → countTrue l₁ < countTrue l₂ →
    ∃ i < l₁.length, l₁[i]? = some false ∧ l₂[i]? = some true
  | [], [], _, hcnt => by simp [countTrue] at hcnt
  | [], _ :: _, hlen, _ => by simp at hlen
  | _ :: _, [], hlen, _ => by simp at hlen
  | b₁ :: bs₁, b₂ :: bs₂, hlen, hcnt => by
    by_cases hb : b₁ = false ∧ b₂ = true
    · obtain ⟨rfl, rfl⟩ := hb; exact ⟨0, by simp, rfl, rfl⟩
    · simp only [List.length_cons, countTrue] at hlen hcnt
      have hlt : countTrue bs₁ < countTrue bs₂ := by
        cases b₁ <;> cases b₂ <;> simp_all <;> omega
      obtain ⟨i, hi, h₁, h₂⟩ := exists_diff_of_countTrue_lt (by omega) hlt
      exact ⟨i + 1, by simp; omega, h₁, h₂⟩

theorem countTrue_le_of_bound : ∀ {l : List Bool} {k : Nat},
    (∀ i, l[i]? = some true → i < k) → countTrue l ≤ k
  | [], _, _ => by simp [countTrue]
  | false :: bs, k, hb => by
    have hb' : ∀ i, bs[i]? = some true → i < k := fun i hi =>
      Nat.lt_of_succ_lt (hb (i + 1) hi)
    simpa [countTrue] using countTrue_le_of_bound hb'
  | true :: bs, 0, hb => by have := hb 0 rfl; omega
  | true :: bs, k + 1, hb => by
    have hb' : ∀ i, bs[i]? = some true → i < k := fun i hi =>
      Nat.lt_of_succ_lt_succ (hb (i + 1) hi)
    have := countTrue_le_of_bound hb'
    simp only [countTrue, ↓reduceIte]; omega

theorem all_true_lt_of_countTrue : ∀ {l : List Bool} {k : Nat},
    (∀ i, l[i]? = some true → i < k) → k ≤ countTrue l →
    ∀ i < k, l[i]? = some true
  | _, 0, _, _, _, hi => by omega
  | [], k + 1, _, hcnt, _, _ => by simp [countTrue] at hcnt
  | false :: bs, k + 1, hb, hcnt, _, _ => by
    have hb' : ∀ i, bs[i]? = some true → i < k := fun i hi =>
      Nat.lt_of_succ_lt_succ (hb (i + 1) hi)
    have := countTrue_le_of_bound hb'
    simp [countTrue] at hcnt; omega
  | true :: bs, k + 1, _, _, 0, _ => rfl
  | true :: bs, k + 1, hb, hcnt, i + 1, hi => by
    have hb' : ∀ j, bs[j]? = some true → j < k := fun j hj =>
      Nat.lt_of_succ_lt_succ (hb (j + 1) hj)
    simp [countTrue] at hcnt
    exact all_true_lt_of_countTrue hb' (by omega) i (by omega)

theorem countTrue_le_of_imp : ∀ {a b : List Bool},
    a.length = b.length →
    (∀ i : Nat, a[i]? = some true → b[i]? = some true) →
    countTrue a ≤ countTrue b
  | [], [], _, _ => Nat.le_refl 0
  | _ :: _, [], hlen, _ => by simp at hlen
  | [], _ :: _, hlen, _ => by simp at hlen
  | x :: xs, y :: ys, hlen, himp => by
    have ih := countTrue_le_of_imp (by simpa using hlen)
      (fun i hi => himp (i + 1) (by simpa using hi))
    cases x with
    | false => cases y <;> simp [countTrue] <;> omega
    | true =>
      have hy : y = true := by simpa using himp 0 rfl
      subst hy; simp [countTrue, ih]

theorem countTrue_lt_of_imp : ∀ {a b : List Bool},
    a.length = b.length →
    (∀ i : Nat, a[i]? = some true → b[i]? = some true) →
    ∀ {k : Nat}, a[k]? = some false → b[k]? = some true →
    countTrue a < countTrue b
  | [], _, _, _, _, ha, _ => by simp at ha
  | _ :: _, [], hlen, _, _, _, _ => by simp at hlen
  | _ :: xs, _ :: ys, hlen, himp, 0, ha, hb => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at ha hb; subst ha hb
    have := countTrue_le_of_imp (by simpa using hlen)
      (fun i hi => himp (i + 1) (by simpa using hi))
    simp [countTrue]; omega
  | x :: xs, y :: ys, hlen, himp, k + 1, ha, hb => by
    have ih := countTrue_lt_of_imp (by simpa using hlen)
      (fun i hi => himp (i + 1) (by simpa using hi)) (by simpa using ha) (by simpa using hb)
    cases x with
    | false => cases y <;> simp [countTrue] <;> omega
    | true =>
      have hy : y = true := by simpa using himp 0 rfl
      subst hy; simp [countTrue, ih]

/-- Number of elements satisfying `p` in `xs`. -/
def countPred {α : Type} (p : α → Bool) (xs : List α) : Nat :=
  countTrue (xs.map p)

private theorem map_set_eq {α β : Type} (f : α → β) :
    ∀ (xs : List α) (i : Nat) (v : α), (xs.set i v).map f = (xs.map f).set i (f v)
  | [], _, _ => rfl
  | _ :: _, 0, _ => rfl
  | _ :: xs, i + 1, v => by simp [List.set_cons_succ, map_set_eq f xs i v]

theorem countPred_replicate_false {α : Type} {p : α → Bool} {v : α}
    (hp : p v = false) (n : Nat) : countPred p (List.replicate n v) = 0 := by
  simp [countPred, hp, countTrue_replicate_false]

theorem countPred_set_true {α : Type} {p : α → Bool} {xs : List α} {l : Nat}
    {oldV newV : α} (hl : xs[l]? = some oldV)
    (hold : p oldV = false) (hnew : p newV = true) :
    countPred p (xs.set l newV) = countPred p xs + 1 := by
  unfold countPred; rw [map_set_eq, hnew]
  exact countTrue_set_true (by simp [hl, hold])

theorem countPred_set_false {α : Type} {p : α → Bool} {xs : List α} {l : Nat}
    {oldV newV : α} (hl : xs[l]? = some oldV)
    (hold : p oldV = true) (hnew : p newV = false) :
    countPred p (xs.set l newV) + 1 = countPred p xs := by
  unfold countPred; rw [map_set_eq, hnew]
  exact countTrue_set_false (by simp [hl, hold])

theorem countPred_set_same {α : Type} {p : α → Bool} {xs : List α} {l : Nat}
    {oldV newV : α} (hl : xs[l]? = some oldV) (heq : p newV = p oldV) :
    countPred p (xs.set l newV) = countPred p xs := by
  have hlt : l < (xs.map p).length := by simp [lt_length_of_getElem?_eq hl]
  unfold countPred; rw [map_set_eq, heq]
  congr 1; apply List.ext_getElem?; intro i
  by_cases hi : i = l
  · subst hi; rw [List.getElem?_set_self hlt]; simp [hl]
  · exact List.getElem?_set_ne (Ne.symm hi)

theorem exists_of_countPred_pos : ∀ {α : Type} {p : α → Bool} {xs : List α},
    0 < countPred p xs → ∃ (l : Nat) (v : α), xs[l]? = some v ∧ p v = true
  | _, _, [], h => by simp [countPred, countTrue] at h
  | _, p, v :: rest, h => by
    cases hp : p v
    · obtain ⟨l, w, hl, hw⟩ :=
        exists_of_countPred_pos (by simpa [countPred, countTrue, hp] using h)
      exact ⟨l + 1, w, hl, hw⟩
    · exact ⟨0, v, rfl, hp⟩

/-- A boolean checklist `xs` of length `n` with `cnt` marked entries (`countTrue xs = cnt`). -/
structure Checklist (xs : List Bool) (n cnt : Nat) : Prop where
  len : xs.length = n
  cnt_eq : countTrue xs = cnt

namespace Checklist

theorem init (n : Nat) : Checklist (List.replicate n false) n 0 :=
  ⟨List.length_replicate, countTrue_replicate_false n⟩

theorem set_true {xs : List Bool} {n cnt l : Nat}
    (h : Checklist xs n cnt) (hf : xs[l]? = some false) :
    Checklist (xs.set l true) n (cnt + 1) :=
  ⟨(List.length_set ..).trans h.len, (countTrue_set_true hf).trans (by rw [h.cnt_eq])⟩

theorem set_false {xs : List Bool} {n cnt l : Nat}
    (h : Checklist xs n cnt) (ht : xs[l]? = some true) :
    Checklist (xs.set l false) n (cnt - 1) :=
  ⟨(List.length_set ..).trans h.len, by have := countTrue_set_false ht; rw [h.cnt_eq] at this; omega⟩

theorem cast_cnt {xs : List Bool} {n c₁ c₂ : Nat}
    (h : Checklist xs n c₁) (hc : c₁ = c₂) : Checklist xs n c₂ :=
  hc ▸ h

theorem all_true {xs : List Bool} {n : Nat}
    (h : Checklist xs n n) {k : Nat} (hk : k < n) : xs[k]? = some true :=
  all_true_of_countTrue_eq_length (h.cnt_eq.trans h.len.symm) k (h.len ▸ hk)

theorem exists_false {xs : List Bool} {n cnt k : Nat}
    (h : Checklist xs n cnt) (hk : k ≤ n) (hlt : cnt < k) :
    ∃ l < k, xs[l]? = some false :=
  exists_false_lt (h.len ▸ hk) (h.cnt_eq ▸ hlt)

theorem exists_true {xs : List Bool} {n cnt : Nat}
    (h : Checklist xs n cnt) (hpos : 0 < cnt) :
    ∃ l < n, xs[l]? = some true := by
  obtain ⟨l, hl, ht⟩ := exists_true_of_countTrue_pos (h.cnt_eq ▸ hpos)
  exact ⟨l, h.len ▸ hl, ht⟩

theorem exists_diff {xs₁ xs₂ : List Bool} {n c₁ c₂ : Nat}
    (h₁ : Checklist xs₁ n c₁) (h₂ : Checklist xs₂ n c₂) (hlt : c₁ < c₂) :
    ∃ l < n, xs₁[l]? = some false ∧ xs₂[l]? = some true := by
  obtain ⟨l, hl, hf, ht⟩ := exists_diff_of_countTrue_lt (h₁.len.trans h₂.len.symm)
    (by rw [h₁.cnt_eq, h₂.cnt_eq]; exact hlt)
  exact ⟨l, h₁.len ▸ hl, hf, ht⟩

end Checklist

end TpuSyncVerify

import TpuSyncVerify.Transfer.Reshard.PoolStaging.Registration

/-! # `PoolStaging`: inductive safety of `reserveBeforeLease` and cross-UUID frame lemmas

Part of `TpuSyncVerify.Transfer.Reshard.PoolStaging`. -/

namespace TpuSyncVerify.Transfer.Reshard.PoolStaging

open TpuSyncVerify

/-! ## Part IV — Inductive Safety of `reserveBeforeLease` and Cross-UUID Frame Lemmas

We prove that:
1. `releaseLease` and `acquireLease` on `uuid` leave `lookupLease uuid'` and
   `resolveBlockSlot uuid' b` unchanged for every `uuid' ≠ uuid`.
2. Whenever `a.acquireLease uuid blockIds = some a'`, every `b ∈ blockIds`
   resolves to `some slot` under `a'.resolveBlockSlot uuid b`.
3. Consequently, in every state where active sessions have no concurrent
   in-flight arm with the same `uuid`, any step for another `uuid'` preserves
   every active session's lease resolution. -/

/-- Disjointness of `releaseLease` from any other `uuid' ≠ uuid`: releasing
`uuid` preserves `lookupLease uuid'`. -/
theorem lookupLease_filter_ne (leases : List (Nat × Lease)) {uuid uuid' : Nat}
    (hne : uuid' ≠ uuid) :
    ((leases.filter (fun p => p.1 != uuid)).find? (fun p => p.1 == uuid')).map Prod.snd =
      (leases.find? (fun p => p.1 == uuid')).map Prod.snd := by
  induction leases with
  | nil => rfl
  | cons hd tl ih =>
    rcases hd with ⟨u, l⟩
    simp only [List.filter_cons]
    by_cases hu : u = uuid
    · subst hu
      have hne' : (u == uuid') = false := beq_eq_false_iff_ne.mpr (Ne.symm hne)
      simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, List.find?_cons, hne', ih]
    · have hne_u : (u != uuid) = true := bne_iff_ne.mpr hu
      simp only [hne_u, ↓reduceIte, List.find?_cons]
      by_cases hu' : u = uuid'
      · subst hu'; simp
      · have hne'' : (u == uuid') = false := beq_eq_false_iff_ne.mpr hu'
        simp only [hne'', ih]

/-- Releasing `uuid` leaves `lookupLease uuid'` and `resolveBlockSlot uuid' b`
unchanged for every `uuid' ≠ uuid`. -/
theorem releaseLease_preserves_other_uuid (a : Arena) {uuid uuid' : Nat}
    (hne : uuid' ≠ uuid) :
    (a.releaseLease uuid).lookupLease uuid' = a.lookupLease uuid' := by
  unfold Arena.releaseLease
  cases a.lookupLease uuid with
  | none => rfl
  | some l =>
    unfold Arena.lookupLease
    exact lookupLease_filter_ne a.leases hne

theorem releaseLease_preserves_other_resolve (a : Arena) {uuid uuid' b : Nat}
    (hne : uuid' ≠ uuid) :
    (a.releaseLease uuid).resolveBlockSlot uuid' b = a.resolveBlockSlot uuid' b := by
  unfold Arena.resolveBlockSlot
  rw [releaseLease_preserves_other_uuid a hne]

/-- Upserting `uuid`'s lease leaves `lookupLease uuid'` unchanged for every
`uuid' ≠ uuid`. -/
theorem lookupLease_upsert_ne (leases : List (Nat × Lease)) (uuid : Nat) (l : Lease)
    {uuid' : Nat} (hne : uuid' ≠ uuid) :
    ((upsertLease leases uuid l).find? (fun p => p.1 == uuid')).map Prod.snd =
      (leases.find? (fun p => p.1 == uuid')).map Prod.snd := by
  induction leases with
  | nil =>
    have hne' : (uuid == uuid') = false := beq_eq_false_iff_ne.mpr (Ne.symm hne)
    simp [upsertLease, hne']
  | cons hd tl ih =>
    rcases hd with ⟨u, old⟩
    simp only [upsertLease]
    by_cases hu : u = uuid
    · subst hu
      have hne' : (u == uuid') = false := beq_eq_false_iff_ne.mpr (Ne.symm hne)
      simp [hne']
    · simp only [hu, ↓reduceIte, List.find?_cons]
      by_cases hu' : u = uuid'
      · subst hu'; simp
      · have hne' : (u == uuid') = false := beq_eq_false_iff_ne.mpr hu'
        simp [hne', ih]

/-- Upserting `uuid`'s lease makes `lookupLease uuid` return `some l`. -/
theorem lookupLease_upsert_self (leases : List (Nat × Lease)) (uuid : Nat) (l : Lease) :
    ((upsertLease leases uuid l).find? (fun p => p.1 == uuid)).map Prod.snd = some l := by
  induction leases with
  | nil => simp [upsertLease]
  | cons hd tl ih =>
    rcases hd with ⟨u, old⟩
    simp only [upsertLease]
    by_cases hu : u = uuid
    · subst hu; simp
    · simp only [hu, ↓reduceIte, List.find?_cons]
      have hne : (u == uuid) = false := beq_eq_false_iff_ne.mpr hu
      simp [hne, ih]

/-- Acquiring a lease for `uuid` preserves `lookupLease uuid'` and
`resolveBlockSlot uuid' b` for every `uuid' ≠ uuid`. -/
theorem acquireLease_preserves_other_uuid {a a' : Arena} {uuid uuid' : Nat}
    {blockIds : List Nat}
    (hacq : a.acquireLease uuid blockIds = some a')
    (hne : uuid' ≠ uuid) :
    a'.lookupLease uuid' = a.lookupLease uuid' := by
  dsimp only [Arena.acquireLease] at hacq
  split at hacq
  · injection hacq with h; subst h; rfl
  · split at hacq
    · cases hacq
    · cases halloc : allocateMissing (missingBlocks (a.lookupLease uuid) blockIds)
          a.freeSlots ((a.lookupLease uuid).getD ⟨[], []⟩) with
      | none => simp [halloc] at hacq
      | some pair =>
        rcases pair with ⟨free', lease'⟩
        simp only [halloc, Option.some.injEq] at hacq
        subst hacq
        unfold Arena.lookupLease
        exact lookupLease_upsert_ne a.leases uuid lease' hne

theorem acquireLease_preserves_other_resolve {a a' : Arena} {uuid uuid' b : Nat}
    {blockIds : List Nat}
    (hacq : a.acquireLease uuid blockIds = some a')
    (hne : uuid' ≠ uuid) :
    a'.resolveBlockSlot uuid' b = a.resolveBlockSlot uuid' b := by
  unfold Arena.resolveBlockSlot
  rw [acquireLease_preserves_other_uuid hacq hne]

/-- Extending `slotByBlock` on the right preserves any existing `lookupSlot`
hit. -/
theorem lookupSlot_append_isSome_left (l : Lease) (extra : List (Nat × Nat))
    (slots' : List Nat) {b : Nat} (h : (l.lookupSlot b).isSome = true) :
    ((Lease.mk (l.slotByBlock ++ extra) slots').lookupSlot b).isSome = true := by
  unfold Lease.lookupSlot at h ⊢
  rw [List.find?_append]
  cases hfind : l.slotByBlock.find? (fun p => p.1 == b) with
  | none => simp [hfind] at h
  | some p => simp

/-- Appending `(b, s)` to `slotByBlock` ensures `lookupSlot b` is `some`. -/
theorem lookupSlot_snoc_self (xs : List (Nat × Nat)) (slots' : List Nat) (b s : Nat) :
    ((Lease.mk (xs ++ [(b, s)]) slots').lookupSlot b).isSome = true := by
  unfold Lease.lookupSlot
  rw [List.find?_append]
  cases xs.find? (fun p => p.1 == b) <;> simp

/-- `allocateMissing` preserves existing block mappings in `base` and adds a
mapping for every block in `missing`. -/
theorem allocateMissing_covers {missing free free' : List Nat} {base lease' : Lease}
    (halloc : allocateMissing missing free base = some (free', lease')) :
    (∀ b, (base.lookupSlot b).isSome = true → (lease'.lookupSlot b).isSome = true) ∧
    (∀ b ∈ missing, (lease'.lookupSlot b).isSome = true) := by
  induction missing generalizing free base with
  | nil =>
    simp only [allocateMissing, Option.some.injEq, Prod.mk.injEq] at halloc
    rcases halloc with ⟨_, rfl⟩
    exact ⟨fun _ h => h, fun _ h => by cases h⟩
  | cons m ms ih =>
    cases free with
    | nil => simp [allocateMissing] at halloc
    | cons s rest =>
      simp only [allocateMissing] at halloc
      obtain ⟨hpres, hmem⟩ := ih halloc
      refine ⟨?_, ?_⟩
      · intro b hb
        exact hpres b (lookupSlot_append_isSome_left base [(m, s)] (base.slots ++ [s]) hb)
      · intro b hb
        rcases List.mem_cons.mp hb with rfl | hb_ms
        · exact hpres b (lookupSlot_snoc_self base.slotByBlock (base.slots ++ [s]) b s)
        · exact hmem b hb_ms

/-- Every block `b ∈ blockIds` is either already mapped in `cur`, already in
`seen`, or included in `missingBlocksAux cur blockIds seen`. -/
theorem mem_missingBlocksAux (cur : Option Lease) (blockIds seen : List Nat)
    {b : Nat} (hb : b ∈ blockIds) :
    alreadyLeased cur b = true ∨
    seen.contains b = true ∨
    b ∈ missingBlocksAux cur blockIds seen := by
  induction blockIds generalizing seen with
  | nil => cases hb
  | cons hd tl ih =>
    simp only [missingBlocksAux]
    rcases List.mem_cons.mp hb with rfl | hb_tl
    · split
      · rename_i hcond
        rcases Bool.or_eq_true_iff.mp hcond with h1 | h2
        · exact Or.inl h1
        · exact Or.inr (Or.inl h2)
      · exact Or.inr (Or.inr List.mem_cons_self)
    · split
      · exact ih seen hb_tl
      · rcases ih (hd :: seen) hb_tl with h1 | h2 | h3
        · exact Or.inl h1
        · simp only [List.contains_cons, Bool.or_eq_true, beq_iff_eq] at h2
          rcases h2 with rfl | h2
          · exact Or.inr (Or.inr List.mem_cons_self)
          · exact Or.inr (Or.inl h2)
        · exact Or.inr (Or.inr (List.mem_cons_of_mem hd h3))

/-- **Postcondition of `acquireLease` (`KVCacheManagerBase::AcquirePoolStagingLease`)**:
Whenever `a.acquireLease uuid blockIds = some a'`, every `b ∈ blockIds`
resolves to `some slot` under `a'.resolveBlockSlot uuid b`. -/
theorem acquireLease_resolves_all {a a' : Arena} {uuid : Nat} {blockIds : List Nat}
    (hacq : a.acquireLease uuid blockIds = some a')
    {b : Nat} (hb : b ∈ blockIds) :
    (a'.resolveBlockSlot uuid b).isSome = true := by
  have hcov := mem_missingBlocksAux (a.lookupLease uuid) blockIds [] hb
  dsimp only [Arena.acquireLease] at hacq
  split at hacq
  · rename_i hempty
    injection hacq with h; subst h
    have hnil : missingBlocks (a.lookupLease uuid) blockIds = [] :=
      List.isEmpty_iff.mp hempty
    unfold missingBlocks at hnil
    rw [hnil] at hcov
    rcases hcov with hcur | hseen | hmem
    · unfold Arena.resolveBlockSlot alreadyLeased at *
      cases hopt : a.lookupLease uuid with
      | none => simp [hopt] at hcur
      | some l => simp [hopt] at hcur ⊢; exact hcur
    · simp at hseen
    · cases hmem
  · split at hacq
    · cases hacq
    · cases halloc : allocateMissing (missingBlocks (a.lookupLease uuid) blockIds)
          a.freeSlots ((a.lookupLease uuid).getD ⟨[], []⟩) with
      | none => simp [halloc] at hacq
      | some pair =>
        rcases pair with ⟨free', lease'⟩
        simp only [halloc, Option.some.injEq] at hacq
        subst hacq
        obtain ⟨hpres, hmem⟩ := allocateMissing_covers halloc
        unfold Arena.resolveBlockSlot Arena.lookupLease
        rw [lookupLease_upsert_self]
        rcases hcov with hcur | hseen | hmiss
        · apply hpres
          unfold alreadyLeased at hcur
          cases hopt : a.lookupLease uuid with
          | none => simp [hopt] at hcur
          | some l => simp [hopt] at hcur ⊢; exact hcur
        · simp at hseen
        · exact hmem b hmiss

/-- Inductive invariant for `RegState.sys .reserveBeforeLease numSlots`:
1. Every active session `(uuid, blocks) ∈ s.activeSessions` resolves every
   `b ∈ blocks` to an allocated staging slot.
2. Every in-flight arm that has completed Phase 2 (`leasedInPhase2 = true`)
   resolves every `b ∈ arm.blocks` to an allocated staging slot.
3. No in-flight arm shares a `uuid` with any active session.
4. Any two in-flight arms with the same `uuid` have the same `callId`.
5. Any two in-flight arms with the same `callId` have the same `uuid` and
   `blocks`. -/
structure ReserveBeforeLeaseInv (s : RegState) : Prop where
  activeValid :
    ∀ p ∈ s.activeSessions, ∀ b ∈ p.2,
      (s.arena.resolveBlockSlot p.1 b).isSome = true
  leasedArmsValid :
    ∀ arm ∈ s.inFlightArms, arm.leasedInPhase2 = true →
      ∀ b ∈ arm.blocks, (s.arena.resolveBlockSlot arm.uuid b).isSome = true
  activeDisjointArms :
    ∀ p ∈ s.activeSessions, ∀ arm ∈ s.inFlightArms, arm.uuid ≠ p.1
  armsSameUuidSameCall :
    ∀ a₁ ∈ s.inFlightArms, ∀ a₂ ∈ s.inFlightArms,
      a₁.uuid = a₂.uuid → a₁.callId = a₂.callId
  armsSameCallSameSpec :
    ∀ a₁ ∈ s.inFlightArms, ∀ a₂ ∈ s.inFlightArms,
      a₁.callId = a₂.callId → a₁.uuid = a₂.uuid ∧ a₁.blocks = a₂.blocks

theorem init_reserveBeforeLease_inv (numSlots : Nat) :
    ReserveBeforeLeaseInv (RegState.init numSlots) where
  activeValid := by intro p hp; cases hp
  leasedArmsValid := by intro arm harm; cases harm
  activeDisjointArms := by intro p hp; cases hp
  armsSameUuidSameCall := by intro a₁ ha₁; cases ha₁
  armsSameCallSameSpec := by intro a₁ ha₁; cases ha₁

theorem findArm_mem_and_callId {s : RegState} {callId : Nat} {arm : InFlightArm}
    (h : s.findArm callId = some arm) :
    arm ∈ s.inFlightArms ∧ arm.callId = callId := by
  unfold RegState.findArm at h
  refine ⟨List.mem_of_find?_eq_some h, ?_⟩
  have := List.find?_some h
  simpa using this

theorem not_findArm_isSome {s : RegState} {callId : Nat}
    (h : (s.findArm callId).isSome = false) :
    ∀ a ∈ s.inFlightArms, a.callId ≠ callId := by
  unfold RegState.findArm at h
  intro a ha heq
  cases hfind : s.inFlightArms.find? (fun x => x.callId == callId) with
  | none =>
    have := List.find?_eq_none.mp hfind a ha
    simp [heq] at this
  | some x =>
    simp [hfind] at h

theorem mem_removeArm_iff {s : RegState} {callId : Nat} {a : InFlightArm} :
    a ∈ s.removeArm callId ↔ a ∈ s.inFlightArms ∧ a.callId ≠ callId := by
  unfold RegState.removeArm
  simp [List.mem_filter, bne_iff_ne]

theorem not_hasInFlightUuid_of_false {s : RegState} {uuid : Nat}
    (h : s.hasInFlightUuid uuid = false) :
    ∀ a ∈ s.inFlightArms, a.uuid ≠ uuid := by
  unfold RegState.hasInFlightUuid at h
  intro a ha heq
  have : s.inFlightArms.any (fun x => x.uuid == uuid) = true :=
    List.any_eq_true.mpr ⟨a, ha, beq_iff_eq.mpr heq⟩
  rw [h] at this
  cases this

theorem step_reserveBeforeLease_inv {s s' : RegState} {e : RegEv}
    (hinv : ReserveBeforeLeaseInv s)
    (hstep : RegState.step .reserveBeforeLease s e = some s') :
    ReserveBeforeLeaseInv s' := by
  cases e with
  | beginArm callId uuid blocks =>
    dsimp only [RegState.step] at hstep
    split at hstep
    · cases hstep
    rename_i hno_call
    split at hstep
    · cases hstep
    rename_i hno_act
    split at hstep
    · cases hstep
    rename_i hno_flt
    injection hstep with hs'; subst hs'
    simp only [beq_self_eq_true, Bool.true_and, Bool.not_eq_true] at hno_call hno_act hno_flt
    have hcid_ne : ∀ a ∈ s.inFlightArms, a.callId ≠ callId :=
      not_findArm_isSome hno_call
    have hact_ne : ∀ p ∈ s.activeSessions, p.1 ≠ uuid := by
      intro p hp heq
      have : s.hasActiveUuid uuid = true :=
        List.any_eq_true.mpr ⟨p, hp, beq_iff_eq.mpr heq⟩
      rw [hno_act] at this; cases this
    have hflt_ne : ∀ a ∈ s.inFlightArms, a.uuid ≠ uuid :=
      not_hasInFlightUuid_of_false hno_flt
    refine ⟨hinv.activeValid, ?_, ?_, ?_, ?_⟩
    · intro arm harm hleased
      rcases List.mem_append.mp harm with harm | harm
      · exact hinv.leasedArmsValid arm harm hleased
      · rcases List.mem_singleton.mp harm with rfl
        cases hleased
    · intro p hp arm harm
      rcases List.mem_append.mp harm with harm | harm
      · exact hinv.activeDisjointArms p hp arm harm
      · rcases List.mem_singleton.mp harm with rfl
        exact Ne.symm (hact_ne p hp)
    · intro a₁ ha₁ a₂ ha₂ hu
      rcases List.mem_append.mp ha₁ with ha₁ | ha₁ <;>
        rcases List.mem_append.mp ha₂ with ha₂ | ha₂
      · exact hinv.armsSameUuidSameCall a₁ ha₁ a₂ ha₂ hu
      · rcases List.mem_singleton.mp ha₂ with rfl
        exact absurd hu (hflt_ne a₁ ha₁)
      · rcases List.mem_singleton.mp ha₁ with rfl
        exact absurd hu.symm (hflt_ne a₂ ha₂)
      · rcases List.mem_singleton.mp ha₁ with rfl
        rcases List.mem_singleton.mp ha₂ with rfl
        rfl
    · intro a₁ ha₁ a₂ ha₂ hcid
      rcases List.mem_append.mp ha₁ with ha₁ | ha₁ <;>
        rcases List.mem_append.mp ha₂ with ha₂ | ha₂
      · exact hinv.armsSameCallSameSpec a₁ ha₁ a₂ ha₂ hcid
      · rcases List.mem_singleton.mp ha₂ with rfl
        exact absurd hcid (hcid_ne a₁ ha₁)
      · rcases List.mem_singleton.mp ha₁ with rfl
        exact absurd hcid.symm (hcid_ne a₂ ha₂)
      · rcases List.mem_singleton.mp ha₁ with rfl
        rcases List.mem_singleton.mp ha₂ with rfl
        exact ⟨rfl, rfl⟩
  | acquireArmLease callId =>
    dsimp only [RegState.step] at hstep
    cases hfind : s.findArm callId with
    | none => simp [hfind] at hstep
    | some arm =>
      simp only [hfind] at hstep
      obtain ⟨harm_mem, harm_cid⟩ := findArm_mem_and_callId hfind
      split at hstep
      · cases hstep
      cases hacq : s.arena.acquireLease arm.uuid arm.blocks with
      | none =>
        simp only [hacq, Option.some.injEq] at hstep
        subst hstep
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · intro p hp b hb
          have hne : p.1 ≠ arm.uuid := Ne.symm (hinv.activeDisjointArms p hp arm harm_mem)
          rw [releaseLease_preserves_other_resolve s.arena hne]
          exact hinv.activeValid p hp b hb
        · intro a ha hleased b hb
          obtain ⟨ha_mem, ha_ne⟩ := mem_removeArm_iff.mp ha
          have hu_ne : a.uuid ≠ arm.uuid := by
            intro heq
            exact ha_ne (by rw [hinv.armsSameUuidSameCall a ha_mem arm harm_mem heq, harm_cid])
          rw [releaseLease_preserves_other_resolve s.arena hu_ne]
          exact hinv.leasedArmsValid a ha_mem hleased b hb
        · intro p hp a ha
          exact hinv.activeDisjointArms p hp a (mem_removeArm_iff.mp ha).1
        · intro a₁ ha₁ a₂ ha₂ hu
          exact hinv.armsSameUuidSameCall
            a₁ (mem_removeArm_iff.mp ha₁).1 a₂ (mem_removeArm_iff.mp ha₂).1 hu
        · intro a₁ ha₁ a₂ ha₂ hc
          exact hinv.armsSameCallSameSpec
            a₁ (mem_removeArm_iff.mp ha₁).1 a₂ (mem_removeArm_iff.mp ha₂).1 hc
      | some arena' =>
        simp only [hacq, Option.some.injEq] at hstep
        subst hstep
        have hmap_fields : ∀ a' ∈ s.inFlightArms.map (fun a =>
            if a.callId == callId then { a with leasedInPhase2 := true } else a),
            ∃ a ∈ s.inFlightArms,
              a'.callId = a.callId ∧ a'.uuid = a.uuid ∧ a'.blocks = a.blocks ∧
              (a.callId = callId ∨ a' = a) := by
          intro a' ha'
          rcases List.mem_map.mp ha' with ⟨a, ha_mem, ha_eq⟩
          refine ⟨a, ha_mem, ?_⟩
          by_cases hc : a.callId = callId
          · have hbeq : (a.callId == callId) = true := beq_iff_eq.mpr hc
            simp only [hbeq, ↓reduceIte] at ha_eq
            subst ha_eq
            exact ⟨rfl, rfl, rfl, Or.inl hc⟩
          · have hbeq : (a.callId == callId) = false := beq_eq_false_iff_ne.mpr hc
            simp only [hbeq, Bool.false_eq_true, ↓reduceIte] at ha_eq
            subst ha_eq
            exact ⟨rfl, rfl, rfl, Or.inr rfl⟩
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · intro p hp b hb
          have hne : p.1 ≠ arm.uuid := Ne.symm (hinv.activeDisjointArms p hp arm harm_mem)
          rw [acquireLease_preserves_other_resolve hacq hne]
          exact hinv.activeValid p hp b hb
        · intro a' ha' hleased b hb
          obtain ⟨a, ha_mem, hcid_eq, hu_eq, hblk_eq, hcases⟩ := hmap_fields a' ha'
          rcases hcases with hc | rfl
          · have ⟨hu_arm, hblk_arm⟩ :=
              hinv.armsSameCallSameSpec a ha_mem arm harm_mem (hc.trans harm_cid.symm)
            rw [hu_eq, hu_arm]
            rw [hblk_eq, hblk_arm] at hb
            exact acquireLease_resolves_all hacq hb
          · by_cases hu : a'.uuid = arm.uuid
            · have hc := hinv.armsSameUuidSameCall a' ha_mem arm harm_mem hu
              have ⟨hu_arm, hblk_arm⟩ :=
                hinv.armsSameCallSameSpec a' ha_mem arm harm_mem hc
              rw [hu_arm]
              rw [hblk_arm] at hb
              exact acquireLease_resolves_all hacq hb
            · rw [acquireLease_preserves_other_resolve hacq hu]
              exact hinv.leasedArmsValid a' ha_mem hleased b hb
        · intro p hp a' ha'
          obtain ⟨a, ha_mem, _, hu_eq, _, _⟩ := hmap_fields a' ha'
          rw [hu_eq]
          exact hinv.activeDisjointArms p hp a ha_mem
        · intro a₁ ha₁ a₂ ha₂ hu
          obtain ⟨x₁, hx₁, hc₁, hu₁, _, _⟩ := hmap_fields a₁ ha₁
          obtain ⟨x₂, hx₂, hc₂, hu₂, _, _⟩ := hmap_fields a₂ ha₂
          rw [hc₁, hc₂]
          exact hinv.armsSameUuidSameCall x₁ hx₁ x₂ hx₂ (by omega)
        · intro a₁ ha₁ a₂ ha₂ hc
          obtain ⟨x₁, hx₁, hc₁, hu₁, hb₁, _⟩ := hmap_fields a₁ ha₁
          obtain ⟨x₂, hx₂, hc₂, hu₂, hb₂, _⟩ := hmap_fields a₂ ha₂
          rw [hu₁, hu₂, hb₁, hb₂]
          exact hinv.armsSameCallSameSpec x₁ hx₁ x₂ hx₂ (by omega)
  | finishArm callId =>
    dsimp only [RegState.step] at hstep
    cases hfind : s.findArm callId with
    | none => simp [hfind] at hstep
    | some arm =>
      simp only [hfind] at hstep
      obtain ⟨harm_mem, harm_cid⟩ := findArm_mem_and_callId hfind
      split at hstep
      · cases hstep
      rename_i hnot_unleased
      have hleased : arm.leasedInPhase2 = true := by
        cases h : arm.leasedInPhase2 <;> simp [h] at hnot_unleased <;> rfl
      split at hstep
      · rename_i hact_true
        unfold RegState.hasActiveUuid at hact_true
        obtain ⟨p, hp, hp_eq⟩ := List.any_eq_true.mp hact_true
        have hu_eq : p.1 = arm.uuid := beq_iff_eq.mp hp_eq
        exact absurd hu_eq.symm (hinv.activeDisjointArms p hp arm harm_mem)
      · injection hstep with hs'; subst hs'
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · intro p hp b hb
          rcases List.mem_append.mp hp with hp | hp
          · exact hinv.activeValid p hp b hb
          · rcases List.mem_singleton.mp hp with rfl
            exact hinv.leasedArmsValid arm harm_mem hleased b hb
        · intro a ha hleased_a b hb
          exact hinv.leasedArmsValid a (mem_removeArm_iff.mp ha).1 hleased_a b hb
        · intro p hp a ha
          obtain ⟨ha_mem, ha_ne⟩ := mem_removeArm_iff.mp ha
          rcases List.mem_append.mp hp with hp | hp
          · exact hinv.activeDisjointArms p hp a ha_mem
          · rcases List.mem_singleton.mp hp with rfl
            intro heq
            exact ha_ne (by rw [hinv.armsSameUuidSameCall a ha_mem arm harm_mem heq, harm_cid])
        · intro a₁ ha₁ a₂ ha₂ hu
          exact hinv.armsSameUuidSameCall
            a₁ (mem_removeArm_iff.mp ha₁).1 a₂ (mem_removeArm_iff.mp ha₂).1 hu
        · intro a₁ ha₁ a₂ ha₂ hc
          exact hinv.armsSameCallSameSpec
            a₁ (mem_removeArm_iff.mp ha₁).1 a₂ (mem_removeArm_iff.mp ha₂).1 hc
  | settleSession uuid =>
    dsimp only [RegState.step] at hstep
    split at hstep
    · cases hstep
    rename_i hact
    injection hstep with hs'; subst hs'
    simp only [Bool.not_eq_true, Bool.not_eq_false'] at hact
    obtain ⟨p_act, hp_act, hp_uuid⟩ := List.any_eq_true.mp hact
    have hp_eq : p_act.1 = uuid := beq_iff_eq.mp hp_uuid
    have hflt_ne : ∀ a ∈ s.inFlightArms, a.uuid ≠ uuid := by
      intro a ha
      rw [← hp_eq]
      exact hinv.activeDisjointArms p_act hp_act a ha
    refine ⟨?_, ?_, ?_, hinv.armsSameUuidSameCall, hinv.armsSameCallSameSpec⟩
    · intro p hp b hb
      simp only [List.mem_filter, bne_iff_ne] at hp
      rw [releaseLease_preserves_other_resolve s.arena hp.2]
      exact hinv.activeValid p hp.1 b hb
    · intro a ha hleased b hb
      rw [releaseLease_preserves_other_resolve s.arena (hflt_ne a ha)]
      exact hinv.leasedArmsValid a ha hleased b hb
    · intro p hp a ha
      simp only [List.mem_filter, bne_iff_ne] at hp
      exact hinv.activeDisjointArms p hp.1 a ha

theorem reachable_reserveBeforeLease_inv {numSlots : Nat} {s : RegState}
    (hr : (RegState.sys .reserveBeforeLease numSlots).Reachable s) :
    ReserveBeforeLeaseInv s := by
  induction hr with
  | init => exact init_reserveBeforeLease_inv numSlots
  | step _ hstep ih => exact step_reserveBeforeLease_inv ih hstep

/-- **Inductive Safety Theorem for `reserveBeforeLease`**:
In every reachable state of `RegState.sys .reserveBeforeLease numSlots`,
`ActiveSessionsHoldLeases s = true`: every registered active receive session
holds a live lease in `PoolStagingStorage` covering all of its destination
blocks. -/
theorem reachable_reserveBeforeLease_safe {numSlots : Nat} {s : RegState}
    (hr : (RegState.sys .reserveBeforeLease numSlots).Reachable s) :
    ActiveSessionsHoldLeases s = true := by
  have hinv := reachable_reserveBeforeLease_inv hr
  unfold ActiveSessionsHoldLeases
  rw [List.all_eq_true]
  intro ⟨uuid, blocks⟩ hp
  rw [List.all_eq_true]
  intro b hb
  exact hinv.activeValid (uuid, blocks) hp b hb


end TpuSyncVerify.Transfer.Reshard.PoolStaging

import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.RecvInv

/-! # `LayerStaging`: combined inductive invariant and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. Combines
`Pipeline.Inv`, `SendSideInv`, and `RecvSideInv` with observational equivalence
(`wireB = pipe.wire`, `decodeHbmB = pipe.decodeHbm`) and `slotFault = false`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager)

/-- Full inductive invariant of `BoundedPipeline`. -/
structure Inv (s : BoundedPipeline) : Prop where
  pipeInv : Pipeline.Inv s.pipe
  sendSide : SendSideInv s
  recvSide : RecvSideInv s
  wireEq : s.wireB = s.pipe.wire
  dhbmEq : s.decodeHbmB = s.pipe.decodeHbm
  noFault : s.slotFault = false

theorem inv_init (n ks kr : Nat) : Inv (init n ks kr) :=
  ⟨Pipeline.inv_init n, sendSideInv_init n ks kr, recvSideInv_init n ks kr, rfl, rfl, rfl⟩

theorem inv_initUnregistered (n ks kr : Nat) : Inv (initUnregistered n ks kr) :=
  ⟨Pipeline.inv_initUnregistered n, sendSideInv_initUnregistered n ks kr,
   recvSideInv_initUnregistered n ks kr, rfl, rfl, rfl⟩

theorem inv_safe {s : BoundedPipeline} (h : Inv s) : Safe s := by
  have hpSafe := Pipeline.inv_safe h.pipeInv
  refine ⟨hpSafe, fun hpub => h.dhbmEq ▸ hpSafe.1 hpub, h.noFault,
    ⟨h.sendSide.mgrInv.conservation, h.sendSide.mgrInv.nodupSlots,
     h.recvSide.mgrInv.conservation, h.recvSide.mgrInv.nodupSlots⟩,
    ⟨fun hd => by simpa [StagingLayerManager.numFree, h.sendSide.doneLeases hd]
                    using h.sendSide.mgrInv.conservation,
     fun hd => by simpa [StagingLayerManager.numFree, h.recvSide.doneLeases hd]
                    using h.recvSide.mgrInv.conservation⟩,
    ⟨h.sendSide.drainingShutdown, h.recvSide.drainingShutdown⟩⟩

/-- Every `BoundedPipeline` step preserves `Inv`. -/
theorem step_inv {s s' : BoundedPipeline} {e : Ev}
    (h : Inv s) (hst : step s e = some s') : Inv s' := by
  have hp' := step_pipe_inv h.pipeInv hst
  have hs' := step_sendSideInv h.pipeInv h.sendSide hst
  have hwireGood : ∀ (k : Nat) (c : Cell), s.wireB[k]? = some c → c = .blank ∨ c = .kv k :=
    fun k c hk => h.pipeInv.wire_good k c (h.wireEq ▸ hk)
  have hr' := step_recvSideInv h.pipeInv h.recvSide hwireGood hst
  suffices hmem : s'.wireB = s'.pipe.wire ∧ s'.decodeHbmB = s'.pipe.decodeHbm ∧ s'.slotFault = false by
    exact ⟨hp', hs', hr', hmem.1, hmem.2.1, hmem.2.2⟩
  cases e with
  | pipe pe =>
    simp only [step] at hst
    cases pe with
    | d2hReady l =>
      simp only [pipeStep, d2hReady, Pipeline.step, Pipeline.d2hReady, Option.map_eq_some_iff] at hst
      rcases hst with ⟨p, hp0, hpost⟩; split at hp0 <;> try cases hp0
      rename_i hg; rcases Option.map_eq_some_iff.mp hp0 with ⟨_, _, rfl⟩
      obtain ⟨_, slot, hlk, hslt⟩ := send_d2hReady_slot h.pipeInv h.sendSide hg.1 hg.2
      rw [withSlot_eq hlk hslt] at hpost; subst hpost
      exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
    | send se =>
      cases se with
      | h2hIssue =>
        simp only [pipeStep, sendH2hIssue, Pipeline.step, Pipeline.sendStep, Send.step,
                   Send.h2hIssue, Option.map_eq_some_iff] at hst
        rcases hst with ⟨p, ⟨snd, hsnd, rfl⟩, hpost⟩; split at hsnd <;> try cases hsnd
        have hp1 : s.pipe.send.pooled = 1 := by have := h.pipeInv.send.chain.chain_le; omega
        obtain ⟨_, slot, hlk, hslt⟩ := send_h2hIssue_slot h.pipeInv h.sendSide hp1
        cases hdr : s.pipe.send.life.draining <;>
          (simp only [hdr, Bool.false_eq_true, ↓reduceIte, withSlot_eq hlk hslt] at hpost; subst hpost;
           exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩)
      | _ =>
        simp only [pipeStep, sendD2hBegin, sendD2hIssue, sendWake,
                   Pipeline.step, Pipeline.sendStep] at hst
        repeat' first
          | exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
          | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hst, hpost⟩
             (repeat' split at hpost) <;> subst hpost)
          | (rw [Option.some.injEq] at hst; subst hst)
          | split at hst | cases hst | contradiction
    | h2hDone l ok =>
      simp only [pipeStep, h2hDone, Pipeline.step, Pipeline.h2hDone, Option.map_eq_some_iff] at hst
      rcases hst with ⟨p, hp0, hpost⟩; split at hp0 <;> try cases hp0
      rename_i hg
      obtain ⟨hd_f, slot, hlk, hslt, hkv_slot⟩ :=
        send_h2hDone_slot h.pipeInv h.sendSide hg.1 hg.2
      rcases Option.map_eq_some_iff.mp hp0 with ⟨snd, _, hp_eq⟩
      cases ok with
      | true =>
        simp only [↓reduceIte, withSlot_eq hlk hslt] at hp_eq hpost; subst hp_eq hpost
        have hcnt := h.pipeInv.send.counters; unfold Send.CountersOrdered at hcnt
        have hwb := h.pipeInv.send.chain.woken_bounds
        have hrdL := h.pipeInv.woken_d2hReady l (by omega) (by omega)
        have hkv_pipe := h.pipeInv.pstaging_good hd_f l hrdL
        refine ⟨?_, h.dhbmEq, h.noFault⟩
        dsimp only
        rw [h.wireEq, getD_of_getElem?_eq_some hkv_slot, getD_of_getElem?_eq_some hkv_pipe]
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte] at hp_eq hpost; subst hp_eq hpost
        exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
    | land l =>
      simp only [pipeStep, land] at hst; split at hst <;> try cases hst
      rename_i hg; rcases hg with ⟨_, hph, _⟩
      have hacc := h.pipeInv.recv.accounted; unfold Recv.Accounted at hacc
      have hd_f := not_done_of_inFlight_pos h.pipeInv.recv.life (by have := h.recvSide.pushesEq; omega)
      obtain ⟨slot, hlk, hslt⟩ := recv_leasedForStream_slot h.pipeInv h.recvSide hd_f hph
      cases hw : s.wireB[l]? with
      | none => simp [hw] at hst
      | some c =>
        simp only [hw] at hst; split at hst <;> try cases hst
        rcases Option.map_eq_some_iff.mp hst with ⟨p, hp0, hpost⟩
        simp only [Pipeline.step, Pipeline.land, show s.pipe.wire[l]? = some c from h.wireEq ▸ hw,
                   ‹¬c = Cell.blank›, ↓reduceIte] at hp0
        split at hp0 <;> cases hp0
        rw [withSlot_eq hlk hslt] at hpost; subst hpost
        exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
    | h2dBegin l =>
      simp only [pipeStep, h2dBegin] at hst; split at hst <;> try cases hst
      rename_i hg; rcases hg with ⟨_, _, _, hph⟩
      have hacc := h.pipeInv.recv.accounted; unfold Recv.Accounted at hacc
      have hd_f := not_done_of_inFlight_pos h.pipeInv.recv.life (by have := h.recvSide.pushesEq; omega)
      obtain ⟨slot, hlk, hslt⟩ := recv_leasedForStream_slot h.pipeInv h.recvSide hd_f hph
      rw [withSlotOpt_eq hlk hslt] at hst
      split at hst
      · simp only [Option.some.injEq] at hst; subst hst; exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
      · rcases Option.map_eq_some_iff.mp hst with ⟨p, hp0, rfl⟩
        simp only [Pipeline.step, Pipeline.h2dBegin] at hp0
        split at hp0 <;> try cases hp0
        rcases Option.map_eq_some_iff.mp hp0 with ⟨_, _, rfl⟩
        exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
    | h2dReady l =>
      simp only [pipeStep, h2dReady] at hst; split at hst <;> try cases hst
      rename_i hg; rcases hg with ⟨hph, hrd_f⟩
      obtain ⟨hd_f, slot, hlk, hslt, hkv_slot⟩ := recv_h2dReady_slot h.pipeInv h.recvSide hph hrd_f
      have hiss_t := (h.recvSide.issuedPhase l).mpr (Or.inl hph)
      have hkv_pipe := h.pipeInv.dstaging_good hd_f l (h.pipeInv.issued_landed l hiss_t)
      simp only [Pipeline.step, Pipeline.h2dReady, hiss_t, hrd_f, and_self,
                 ↓reduceIte, Option.map_eq_some_iff] at hst
      rcases hst with ⟨p, ⟨_, _, rfl⟩, hpost⟩
      rw [withSlot_eq hlk hslt] at hpost; subst hpost
      refine ⟨h.wireEq, ?_, h.noFault⟩
      dsimp only
      rw [h.dhbmEq, getD_of_getElem?_eq_some hkv_slot, getD_of_getElem?_eq_some hkv_pipe]
    | _ =>
      simp only [pipeStep, recvPushBegin, recvPushEnd, h2dIssue, syncPipe,
                 Pipeline.step, Pipeline.notifyForRead, Pipeline.pullWait,
                 Pipeline.recvStep, Pipeline.h2dIssue, Pipeline.reclaim,
                 Pipeline.reseatPrefillStaging, Pipeline.reseatDecodeStaging] at hst
      repeat' first
        | exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
        | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hst, hpost⟩
           (repeat' split at hpost) <;> subst hpost)
        | (rw [Option.some.injEq] at hst; subst hst)
        | split at hst | cases hst | contradiction
  | _ =>
    simp only [step, sendAcquireBegin, sendAcquireDone, recvAcquireBegin,
               recvAcquireWait, recvAcquireFinish, h2dDone, syncPipe,
               Pipeline.step, Pipeline.sendStep, Pipeline.recvStep] at hst
    repeat' first
      | exact ⟨h.wireEq, h.dhbmEq, h.noFault⟩
      | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hst, hpost⟩
         (repeat' split at hpost) <;> subst hpost)
      | (rw [Option.bind_eq_some_iff] at hst; rcases hst with ⟨_, hst₁, hst⟩
         rw [Option.map_eq_some_iff] at hst₁; rcases hst₁ with ⟨_, _, rfl⟩)
      | (rw [Option.some.injEq] at hst; subst hst)
      | split at hst | cases hst

theorem reachable_inv {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sys n ks kr).Reachable s) : Inv s :=
  (sys n ks kr).reachable_induction (inv_init n ks kr)
    (fun _ _ _ hi hs => step_inv hi hs) h

/-- **Main safety theorem (registered offer)**: every reachable state of
`sys n ks kr` satisfies `Safe` — for all `n`, `ks`, and `kr`. -/
theorem reachable_safe {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sys n ks kr).Reachable s) : Safe s :=
  inv_safe (reachable_inv h)

theorem reachable_unregistered_inv {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sysUnregistered n ks kr).Reachable s) : Inv s :=
  (sysUnregistered n ks kr).reachable_induction (inv_initUnregistered n ks kr)
    (fun _ _ _ hi hs => step_inv hi hs) h

/-- **Main safety theorem (unregistered offer / handshake race)**: every
reachable state of `sysUnregistered n ks kr` satisfies `Safe`. -/
theorem reachable_unregistered_safe {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sysUnregistered n ks kr).Reachable s) : Safe s :=
  inv_safe (reachable_unregistered_inv h)

theorem runFrom_inv : ∀ {evs : List Ev} {s s' : BoundedPipeline},
    Inv s → evs.foldlM step s = some s' → Inv s'
  | [], _, _, hi, hr => Option.some.inj hr ▸ hi
  | e :: es, a, _, hi, hr => by
    simp only [List.foldlM_cons] at hr
    cases hse : step a e with
    | none => simp [hse] at hr
    | some m => simp only [hse] at hr; exact runFrom_inv (step_inv hi hse) hr

theorem sendLookupStagingSlot_active {s : BoundedPipeline} (h : Inv s) {l : Nat}
    (hdr : s.pipe.send.life.draining = false)
    (hact : (s.pipe.send.pooled = 1 ∧ l = s.pipe.send.h2hIssued) ∨
            (l < s.pipe.send.h2hIssued ∧ s.pipe.h2hRetiredL[l]? = some false)) :
    ∃ slot, sendLookupStagingSlot s l = some slot ∧
      slot < s.sendMgr.numSlots ∧ s.prefillSlots[slot]? = some (.kv l) := by
  rcases hact with ⟨hp1, rfl⟩ | ⟨hlt, hret⟩
  · obtain ⟨hd_f, slot, hlk, hslt⟩ := send_h2hIssue_slot h.pipeInv h.sendSide hp1
    obtain ⟨hwh, hd2h⟩ := h.pipeInv.send.chain.pooled_state hp1
    have hrdL := h.pipeInv.woken_d2hReady s.pipe.send.h2hIssued (by omega) hd2h
    exact ⟨slot, by simp [sendLookupStagingSlot, hd_f, hdr, hlk],
      h.sendSide.lenSlots ▸ hslt, h.sendSide.slotsGood hd_f _ slot hrdL hlk⟩
  · obtain ⟨hd_f, slot, hlk, hslt, hkv⟩ :=
      send_h2hDone_slot h.pipeInv h.sendSide hlt hret
    exact ⟨slot, by simp [sendLookupStagingSlot, hd_f, hdr, hlk],
      h.sendSide.lenSlots ▸ hslt, hkv⟩

theorem recvLookupStagingSlot_active {s : BoundedPipeline} (h : Inv s) {l : Nat}
    (hstr : 0 < s.activeStreams) (hland : s.pipe.landedL[l]? = some true)
    (hph : s.recvPhaseL[l]? = some .leasedForStream) :
    ∃ slot, recvLookupStagingSlot s l = some slot ∧
      slot < s.recvMgr.numSlots ∧ s.decodeSlots[slot]? = some (.kv l) := by
  have hacc := h.pipeInv.recv.accounted; unfold Recv.Accounted at hacc
  have hd_f := not_done_of_inFlight_pos h.pipeInv.recv.life (by have := h.recvSide.pushesEq; omega)
  obtain ⟨slot, hlk, hkv⟩ := h.recvSide.slotsGood hd_f l hland (Or.inl hph)
  exact ⟨slot, hlk, StagingLayerManager.lookup_lt h.recvSide.mgrInv hlk, hkv⟩

theorem sendLookupStagingSlot_injective {s : BoundedPipeline} (h : Inv s)
    {l₁ l₂ slot : Nat}
    (h₁ : sendLookupStagingSlot s l₁ = some slot)
    (h₂ : sendLookupStagingSlot s l₂ = some slot) :
    l₁ = l₂ ∧ slot < s.sendMgr.numSlots := by
  unfold sendLookupStagingSlot at h₁ h₂
  split at h₁
  · cases h₁
  · rename_i hnd; simp [hnd] at h₂
    exact ⟨StagingLayerManager.lookup_injective h.sendSide.mgrInv h₁ h₂,
           StagingLayerManager.lookup_lt h.sendSide.mgrInv h₁⟩

theorem recvLookupStagingSlot_injective {s : BoundedPipeline} (h : Inv s)
    {l₁ l₂ slot : Nat}
    (h₁ : recvLookupStagingSlot s l₁ = some slot)
    (h₂ : recvLookupStagingSlot s l₂ = some slot) :
    l₁ = l₂ ∧ slot < s.recvMgr.numSlots :=
  ⟨StagingLayerManager.lookup_injective h.recvSide.mgrInv h₁ h₂,
   StagingLayerManager.lookup_lt h.recvSide.mgrInv h₁⟩

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

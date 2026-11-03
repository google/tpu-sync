import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.SendInv

/-! # `LayerStaging`: consumer (`TransferReceiveSession`) bounded-slot invariant

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. Relates `recvMgr`
leases, per-layer `RecvLayerPhase` states, `Recv` counters, and `decodeSlots`
contents across all `BoundedPipeline` events. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager SlotPoolInv)

/-- Consumer-side bounded-slot invariant, built on `SlotPoolInv`. Depends only on
consumer state fields so producer events preserve it by frame. -/
structure RecvSideInv (s : BoundedPipeline) : Prop where
  pool : SlotPoolInv s.pipe.recv.life s.recvMgr s.decodeSlots (recvOwnsSlot s)
  lenPhase : s.recvPhaseL.length = s.pipe.numLayers
  pushesEq : s.pipe.recv.pushes = s.activeStreams + countAcquiring s.recvPhaseL
  cntRetired : countRetired s.recvPhaseL = s.pipe.recv.retired
  zeroSlots : s.recvMgr.numSlots = 0 → s.pipe.recv.pending = 0 ∧ s.pipe.recv.issued = 0
  retiredReady : ∀ l : Nat, s.recvPhaseL[l]? = some .h2dRetired →
    s.pipe.h2dReadyL[l]? = some true
  pendingPhase : ∀ l : Nat, s.pipe.h2dPendingL[l]? = some true ↔
    s.recvPhaseL[l]? = some .h2dPending
  issuedPhase : ∀ l : Nat, s.pipe.h2dIssuedL[l]? = some true ↔
    (s.recvPhaseL[l]? = some .h2dRunning ∨ s.recvPhaseL[l]? = some .h2dRetired)
  landedPhase : ∀ l : Nat, s.pipe.landedL[l]? = some true →
    s.recvPhaseL[l]? = some .leasedForStream ∨
      s.recvPhaseL[l]? = some .h2dPending ∨
      s.recvPhaseL[l]? = some .h2dRunning ∨
      s.recvPhaseL[l]? = some .h2dRetired ∨
      s.recvPhaseL[l]? = some .aborted
  slotsGood : s.pipe.recv.life.done = false →
    ∀ l : Nat, s.pipe.landedL[l]? = some true →
      (s.recvPhaseL[l]? = some .leasedForStream ∨
        s.recvPhaseL[l]? = some .h2dPending ∨
        s.recvPhaseL[l]? = some .h2dRunning) →
      ∃ slot, s.recvMgr.lookup l = some slot ∧ s.decodeSlots[slot]? = some (.kv l)

namespace RecvSideInv
theorem mgrInv {s : BoundedPipeline} (h : RecvSideInv s) : s.recvMgr.Inv := h.pool.mgrInv
theorem lenSlots {s : BoundedPipeline} (h : RecvSideInv s) : s.decodeSlots.length = s.recvMgr.numSlots := h.pool.lenSlots
theorem drainingShutdown {s : BoundedPipeline} (h : RecvSideInv s) :
    s.pipe.recv.life.draining = true → s.recvMgr.shutdown = true := h.pool.drainingShutdown
theorem doneLeases {s : BoundedPipeline} (h : RecvSideInv s) :
    s.pipe.recv.life.done = true → s.recvMgr.leases = [] := h.pool.doneLeases
theorem leasesIff {s : BoundedPipeline} (h : RecvSideInv s) :
    s.pipe.recv.life.done = false →
      ∀ l : Nat, (s.recvMgr.lookup l).isSome = true ↔ recvOwnsSlot s l := h.pool.leasesIff

theorem frame {s s' : BoundedPipeline} (h : RecvSideInv s)
    (hN : s'.pipe.numLayers = s.pipe.numLayers := by rfl)
    (hRecv : s'.pipe.recv = s.pipe.recv := by rfl)
    (hLandL : s'.pipe.landedL = s.pipe.landedL := by rfl)
    (hPendL : s'.pipe.h2dPendingL = s.pipe.h2dPendingL := by rfl)
    (hIssL : s'.pipe.h2dIssuedL = s.pipe.h2dIssuedL := by rfl)
    (hRdyL : s'.pipe.h2dReadyL = s.pipe.h2dReadyL := by rfl)
    (hMgr : s'.recvMgr = s.recvMgr := by rfl)
    (hSlots : s'.decodeSlots = s.decodeSlots := by rfl)
    (hPhase : s'.recvPhaseL = s.recvPhaseL := by rfl)
    (hAct : s'.activeStreams = s.activeStreams := by rfl) : RecvSideInv s' :=
  ⟨h.pool.frame (congrArg Recv.life hRecv) hMgr hSlots
     (fun l => by unfold recvOwnsSlot; rw [hPhase]),
   (by rw [hPhase, hN]; exact h.lenPhase),
   (by rw [hRecv, hAct, hPhase]; exact h.pushesEq),
   (by rw [hPhase, hRecv]; exact h.cntRetired),
   (by rw [hMgr, hRecv]; exact h.zeroSlots),
   (fun l hl => by rw [hRdyL]; exact h.retiredReady l (hPhase ▸ hl)),
   (fun l => by rw [hPendL, hPhase]; exact h.pendingPhase l),
   (fun l => by rw [hIssL, hPhase]; exact h.issuedPhase l),
   (fun l hl => by rw [hPhase]; exact h.landedPhase l (hLandL ▸ hl)),
   (fun hnd l hl hp => by
     rw [hMgr, hSlots]; rw [hPhase] at hp
     exact h.slotsGood (hRecv ▸ hnd) l (hLandL ▸ hl) hp)⟩
end RecvSideInv

theorem recvSideInv_init (n ks kr : Nat) : RecvSideInv (init n ks kr) :=
  ⟨SlotPoolInv.init kr Cell.junk rfl rfl
     (fun _ => by simp only [recvOwnsSlot, init, List.getElem?_replicate]; split <;> simp [ownsRecvSlot]),
   (by simp [init, Pipeline.init]),
   (by simp [init, Pipeline.init, Recv.initLoad,
             countPred_replicate_false (p := isAcquiringPhase) (v := .idle) rfl]),
   (by simp [init, Pipeline.init, Recv.initLoad,
             countPred_replicate_false (p := isRetiredPhase) (v := .idle) rfl]),
   (fun _ => ⟨rfl, rfl⟩),
   (fun _ h => by simp only [init, List.getElem?_replicate] at h; split at h <;> cases h),
   (fun _ => by simp only [init, Pipeline.init, List.getElem?_replicate]; split <;> simp),
   (fun _ => by simp only [init, Pipeline.init, List.getElem?_replicate]; split <;> simp),
   (fun _ h => (not_mem_replicate_false h).elim),
   (fun _ _ h _ => (not_mem_replicate_false h).elim)⟩

theorem recvSideInv_initUnregistered (n ks kr : Nat) : RecvSideInv (initUnregistered n ks kr) :=
  (recvSideInv_init n ks kr).frame

theorem recv_leasedForStream_slot {s : BoundedPipeline}
    (_hp : Pipeline.Inv s.pipe) (hr : RecvSideInv s) {l : Nat}
    (hnd : s.pipe.recv.life.done = false) (hph : s.recvPhaseL[l]? = some .leasedForStream) :
    ∃ slot, s.recvMgr.lookup l = some slot ∧ slot < s.decodeSlots.length :=
  hr.pool.lookup_of_owns hnd (by simp [recvOwnsSlot, hph, ownsRecvSlot])

theorem recv_running_active {s : BoundedPipeline}
    (hp : Pipeline.Inv s.pipe) (hr : RecvSideInv s) {l : Nat}
    (hph : s.recvPhaseL[l]? = some .h2dRunning) :
    s.pipe.h2dIssuedL[l]? = some true ∧ s.pipe.recv.retired < s.pipe.recv.issued ∧
      s.pipe.recv.life.done = false := by
  have hiss : s.pipe.h2dIssuedL[l]? = some true := (hr.issuedPhase l).mpr (Or.inl hph)
  have hacc := hp.recv.accounted; unfold Recv.Accounted at hacc
  have hlt_ret : s.pipe.recv.retired < s.pipe.recv.issued := by
    have hlt_cnt : countTrue (s.recvPhaseL.map isRetiredPhase) < countTrue s.pipe.h2dIssuedL :=
      countTrue_lt_of_imp (by simp [hr.lenPhase, hp.len_h2dIssuedL])
        (fun i hi => by
          simp only [List.getElem?_map] at hi
          cases hpi : s.recvPhaseL[i]? with
          | none => simp [hpi] at hi
          | some ph =>
            simp only [hpi, Option.map_some, Option.some.injEq] at hi
            cases ph <;> simp [isRetiredPhase] at hi
            exact (hr.issuedPhase i).mpr (Or.inr hpi))
        (by simp [List.getElem?_map, hph, isRetiredPhase]) hiss
    change countRetired s.recvPhaseL < countTrue s.pipe.h2dIssuedL at hlt_cnt
    rwa [hr.cntRetired, hp.cnt_h2dIssued] at hlt_cnt
  exact ⟨hiss, hlt_ret, not_done_of_inFlight_pos hp.recv.life (by omega)⟩

theorem recv_h2dReady_slot {s : BoundedPipeline}
    (hp : Pipeline.Inv s.pipe) (hr : RecvSideInv s) {l : Nat}
    (hph : s.recvPhaseL[l]? = some .h2dRunning) (_hrd_f : s.pipe.h2dReadyL[l]? = some false) :
    s.pipe.recv.life.done = false ∧ ∃ slot, s.recvMgr.lookup l = some slot ∧
      slot < s.decodeSlots.length ∧ s.decodeSlots[slot]? = some (.kv l) := by
  obtain ⟨hiss, _, hnd⟩ := recv_running_active hp hr hph
  obtain ⟨slot, hlk, hkv⟩ := hr.slotsGood hnd l (hp.issued_landed l hiss) (Or.inr (Or.inr hph))
  exact ⟨hnd, slot, hlk, hr.lenSlots ▸ StagingLayerManager.lookup_lt hr.mgrInv hlk, hkv⟩

private theorem syncMgr_preserves_recv {s s' : BoundedPipeline} (hr : RecvSideInv s)
    (hPushes : s'.pipe.recv.pushes = s'.activeStreams + countAcquiring s'.recvPhaseL)
    (hOldDone : s.pipe.recv.life.done = true → s'.pipe.recv.life.done = true)
    (hZero : s.recvMgr.numSlots = 0 → s'.pipe.recv.pending = 0 ∧ s'.pipe.recv.issued = 0 := by exact hr.zeroSlots)
    (hN : s'.pipe.numLayers = s.pipe.numLayers := by rfl)
    (hRet : s'.pipe.recv.retired = s.pipe.recv.retired := by rfl)
    (hLandL : s'.pipe.landedL = s.pipe.landedL := by rfl)
    (hPendL : s'.pipe.h2dPendingL = s.pipe.h2dPendingL := by rfl)
    (hIssL : s'.pipe.h2dIssuedL = s.pipe.h2dIssuedL := by rfl)
    (hRdyL : s'.pipe.h2dReadyL = s.pipe.h2dReadyL := by rfl)
    (hMgr : s'.recvMgr = syncMgr s.pipe.recv.life s'.pipe.recv.life s.recvMgr := by rfl)
    (hSlots : s'.decodeSlots = s.decodeSlots := by rfl)
    (hPhase : s'.recvPhaseL = s.recvPhaseL := by rfl) : RecvSideInv s' := by
  have hOldNotDone : s'.pipe.recv.life.done = false → s.pipe.recv.life.done = false := fun hnd => by
    cases hd : s.pipe.recv.life.done with
    | false => rfl
    | true => have := hOldDone hd; rw [hnd] at this; cases this
  exact ⟨hr.pool.sync hMgr hSlots hOldNotDone (fun _ l => by unfold recvOwnsSlot; rw [hPhase]),
    (by rw [hPhase, hN]; exact hr.lenPhase),
    hPushes, (by rw [hPhase, hRet]; exact hr.cntRetired),
    (fun hz => hZero (by simpa [hMgr] using hz)),
    (fun l hl => by rw [hRdyL]; exact hr.retiredReady l (hPhase ▸ hl)),
    (fun l => by rw [hPendL, hPhase]; exact hr.pendingPhase l),
    (fun l => by rw [hIssL, hPhase]; exact hr.issuedPhase l),
    (fun l hl => by rw [hPhase]; exact hr.landedPhase l (hLandL ▸ hl)),
    (fun hnd l hl hp => by
      rw [hMgr, StagingLayerManager.sync_lookup_of_not_done _ hnd, hSlots]; rw [hPhase] at hp
      exact hr.slotsGood (hOldNotDone hnd) l (hLandL ▸ hl) hp)⟩

private theorem passive_recv_preserves {s : BoundedPipeline} {p : Pipeline} {re : Recv.Ev}
    (hr : RecvSideInv s) (hp0 : Pipeline.step s.pipe (.recv re) = some p)
    (hpb : re ≠ .pushBegin) (hpe : re ≠ .pushEnd) (hhd : ∀ ok, re ≠ .h2dDone ok) :
    RecvSideInv (syncPipe s p) := by
  have ⟨rcv, hrcv, hN, hRcv, hL, hP, hI, hR, hbi⟩ : ∃ rcv, Recv.step s.pipe.recv re = some rcv ∧
      p.numLayers = s.pipe.numLayers ∧ p.recv = rcv ∧ p.landedL = s.pipe.landedL ∧
      p.h2dPendingL = s.pipe.h2dPendingL ∧ p.h2dIssuedL = s.pipe.h2dIssuedL ∧
      p.h2dReadyL = s.pipe.h2dReadyL ∧ (re ≠ .h2dBegin ∧ ∀ ok, re ≠ .h2dIssue ok) := by
    cases re <;> simp only [Pipeline.step, Pipeline.recvStep] at hp0
    case pullReply =>
      split at hp0 <;> simp only [Option.map_eq_some_iff, reduceCtorEq] at hp0
      rcases hp0 with ⟨rcv, hrcv, rfl⟩; exact ⟨rcv, hrcv, rfl, rfl, rfl, rfl, rfl, rfl, nofun, nofun⟩
    all_goals
      simp only [Option.map_eq_some_iff, reduceCtorEq] at hp0 <;>
        (rcases hp0 with ⟨rcv, hrcv, rfl⟩; exact ⟨rcv, hrcv, rfl, rfl, rfl, rfl, rfl, rfl, nofun, nofun⟩)
  obtain ⟨hpush, hret⟩ := Recv.step_pushes_retired hrcv hpb hpe hhd
  obtain ⟨hpend, hiss⟩ := Recv.step_pending_issued hrcv hbi.1 hbi.2
  exact syncMgr_preserves_recv hr
    (by dsimp only [syncPipe]; rw [hRcv, hpush]; exact hr.pushesEq)
    (fun hd => by dsimp only [syncPipe]; rw [hRcv]; exact Recv.step_done_mono hrcv hd)
    (fun hz => by dsimp only [syncPipe]; rw [hRcv, hpend, hiss]; exact hr.zeroSlots hz)
    hN (by dsimp only [syncPipe]; rw [hRcv, hret]) hL hP hI hR

private theorem setPhase_preserves_recv {s s' : BoundedPipeline} {l : Nat}
    {newP : RecvLayerPhase} (hr : RecvSideInv s) (hlt_l : l < s.recvPhaseL.length)
    (hPhase : s'.recvPhaseL = s.recvPhaseL.set l newP := by rfl)
    (hPool : SlotPoolInv s'.pipe.recv.life s'.recvMgr s'.decodeSlots (recvOwnsSlot s'))
    (hOldNotDone : s'.pipe.recv.life.done = false → s.pipe.recv.life.done = false)
    (hLookupNe : s'.pipe.recv.life.done = false →
      ∀ u : Nat, u ≠ l → s'.recvMgr.lookup u = s.recvMgr.lookup u)
    (hPushes : s'.pipe.recv.pushes = s'.activeStreams + countAcquiring s'.recvPhaseL)
    (hCntRet : countRetired s'.recvPhaseL = s'.pipe.recv.retired)
    (hZero : s'.recvMgr.numSlots = 0 → s'.pipe.recv.pending = 0 ∧ s'.pipe.recv.issued = 0)
    (hRetRdyAtL : newP = .h2dRetired → s.pipe.h2dReadyL[l]? = some true)
    (hPendL : s'.pipe.h2dPendingL[l]? = some true ↔ newP = .h2dPending)
    (hIssL : s'.pipe.h2dIssuedL[l]? = some true ↔ (newP = .h2dRunning ∨ newP = .h2dRetired))
    (hLandAtL : s.pipe.landedL[l]? = some true →
      newP = .leasedForStream ∨ newP = .h2dPending ∨ newP = .h2dRunning ∨
        newP = .h2dRetired ∨ newP = .aborted)
    (hSlotAtL : s'.pipe.recv.life.done = false → s.pipe.landedL[l]? = some true →
      (newP = .leasedForStream ∨ newP = .h2dPending ∨ newP = .h2dRunning) →
      ∃ slot, s'.recvMgr.lookup l = some slot ∧ s.decodeSlots[slot]? = some (.kv l))
    (hPendNe : ∀ u : Nat, u ≠ l → s'.pipe.h2dPendingL[u]? = s.pipe.h2dPendingL[u]? := by intro _ _; rfl)
    (hIssNe : ∀ u : Nat, u ≠ l → s'.pipe.h2dIssuedL[u]? = s.pipe.h2dIssuedL[u]? := by intro _ _; rfl)
    (hSlots : s'.decodeSlots = s.decodeSlots := by rfl)
    (hLandL : s'.pipe.landedL = s.pipe.landedL := by rfl)
    (hRdyL : s'.pipe.h2dReadyL = s.pipe.h2dReadyL := by rfl)
    (hN : s'.pipe.numLayers = s.pipe.numLayers := by rfl) : RecvSideInv s' := by
  refine ⟨hPool, (by rw [hPhase, List.length_set, hN]; exact hr.lenPhase),
    hPushes, hCntRet, hZero, ?_, ?_, ?_, ?_, ?_⟩
  · intro u hu_ret; rw [hRdyL]
    by_cases hu : u = l
    · subst hu; rw [hPhase, List.getElem?_set_self hlt_l, Option.some.injEq] at hu_ret
      exact hRetRdyAtL hu_ret
    · rw [hPhase, List.getElem?_set_ne (Ne.symm hu)] at hu_ret; exact hr.retiredReady u hu_ret
  · intro u; by_cases hu : u = l
    · subst hu; rw [hPhase, List.getElem?_set_self hlt_l]; simpa using hPendL
    · rw [hPendNe u hu, hPhase, List.getElem?_set_ne (Ne.symm hu)]; exact hr.pendingPhase u
  · intro u; by_cases hu : u = l
    · subst hu; rw [hPhase, List.getElem?_set_self hlt_l]; simpa using hIssL
    · rw [hIssNe u hu, hPhase, List.getElem?_set_ne (Ne.symm hu)]; exact hr.issuedPhase u
  · intro u hl_t; by_cases hu : u = l
    · subst hu; rw [hPhase, List.getElem?_set_self hlt_l]; simpa using hLandAtL (hLandL ▸ hl_t)
    · rw [hPhase, List.getElem?_set_ne (Ne.symm hu)]; exact hr.landedPhase u (hLandL ▸ hl_t)
  · intro hnd u hl_t hp_u; by_cases hu : u = l
    · subst hu; simp only [hPhase, List.getElem?_set_self hlt_l, Option.some.injEq] at hp_u
      rw [hSlots]; exact hSlotAtL hnd (hLandL ▸ hl_t) hp_u
    · rw [hPhase, List.getElem?_set_ne (Ne.symm hu)] at hp_u
      obtain ⟨slot', hlk', hkv⟩ := hr.slotsGood (hOldNotDone hnd) u (hLandL ▸ hl_t) hp_u
      exact ⟨slot', (hLookupNe hnd u hu).trans hlk', hSlots ▸ hkv⟩

private theorem releasePhase_preserves_recv {s s' : BoundedPipeline} {l : Nat}
    {oldP newP : RecvLayerPhase} (hr : RecvSideInv s)
    (hph : s.recvPhaseL[l]? = some oldP) (hd_f : s.pipe.recv.life.done = false)
    (hPhase : s'.recvPhaseL = s.recvPhaseL.set l newP := by rfl)
    (hOwnF : ownsRecvSlot newP = false := by rfl)
    (hPushes : s'.pipe.recv.pushes = s'.activeStreams + countAcquiring s'.recvPhaseL)
    (hCntRet : countRetired s'.recvPhaseL = s'.pipe.recv.retired)
    (hZero : s.recvMgr.numSlots = 0 → s'.pipe.recv.pending = 0 ∧ s'.pipe.recv.issued = 0 := by exact hr.zeroSlots)
    (hRetRdyAtL : newP = .h2dRetired → s.pipe.h2dReadyL[l]? = some true)
    (hPendL : s'.pipe.h2dPendingL[l]? = some true ↔ newP = .h2dPending)
    (hIssL : s'.pipe.h2dIssuedL[l]? = some true ↔ (newP = .h2dRunning ∨ newP = .h2dRetired))
    (hLandAtL : s.pipe.landedL[l]? = some true →
      newP = .leasedForStream ∨ newP = .h2dPending ∨ newP = .h2dRunning ∨
        newP = .h2dRetired ∨ newP = .aborted := by intro _; simp)
    (hPendNe : ∀ u : Nat, u ≠ l → s'.pipe.h2dPendingL[u]? = s.pipe.h2dPendingL[u]? := by intro _ _; rfl)
    (hIssNe : ∀ u : Nat, u ≠ l → s'.pipe.h2dIssuedL[u]? = s.pipe.h2dIssuedL[u]? := by intro _ _; rfl)
    (hMgr : s'.recvMgr = releaseSync s.pipe.recv.life s'.pipe.recv.life l s.recvMgr := by first | rfl | simp)
    (hSlots : s'.decodeSlots = s.decodeSlots := by rfl)
    (hLandL : s'.pipe.landedL = s.pipe.landedL := by rfl)
    (hRdyL : s'.pipe.h2dReadyL = s.pipe.h2dReadyL := by rfl)
    (hN : s'.pipe.numLayers = s.pipe.numLayers := by rfl) :
    RecvSideInv s' := by
  have hlt_l := lt_length_of_getElem?_eq hph
  obtain ⟨_, _, _, hrel_ne⟩ := s.recvMgr.release_lookup l hr.mgrInv
  exact setPhase_preserves_recv hr hlt_l hPhase
    (hr.pool.releaseSync hMgr hSlots (fun _ => hd_f)
      (fun _ => by simp [recvOwnsSlot, hPhase, List.getElem?_set_self hlt_l, hOwnF])
      (fun _ u hu => by simp [recvOwnsSlot, hPhase, List.getElem?_set_ne (Ne.symm hu)]))
    (fun _ => hd_f)
    (fun hnd u hu => by
      rw [hMgr, StagingLayerManager.releaseSync_lookup_of_not_done l _ hnd u, hrel_ne u hu])
    hPushes hCntRet (fun hz => hZero (by simpa [hMgr] using hz)) hRetRdyAtL hPendL hIssL hLandAtL
    (fun _ _ hp_l => by rcases hp_l with rfl | rfl | rfl <;> simp [ownsRecvSlot] at hOwnF)
    hPendNe hIssNe hSlots hLandL hRdyL hN

private theorem acquirePhase_preserves_recv {s s' : BoundedPipeline} {l : Nat}
    {oldP newP : RecvLayerPhase} (hr : RecvSideInv s) (hph : s.recvPhaseL[l]? = some oldP)
    (hPhase : s'.recvPhaseL = s.recvPhaseL.set l newP := by rfl)
    (hOldNP : oldP ≠ .h2dPending := by nofun) (hOldNR : oldP ≠ .h2dRunning := by nofun)
    (hOldNRet : oldP ≠ .h2dRetired := by nofun) (hNewNP : newP ≠ .h2dPending := by nofun)
    (hNewNR : newP ≠ .h2dRunning := by nofun) (hNewNRet : newP ≠ .h2dRetired := by nofun)
    (hNotLand : s.pipe.landedL[l]? ≠ some true)
    (hPool : SlotPoolInv s'.pipe.recv.life s'.recvMgr s'.decodeSlots (recvOwnsSlot s'))
    (hOldNotDone : s'.pipe.recv.life.done = false → s.pipe.recv.life.done = false)
    (hLookupNe : s'.pipe.recv.life.done = false →
      ∀ u, u ≠ l → s'.recvMgr.lookup u = s.recvMgr.lookup u)
    (hPushes : s'.pipe.recv.pushes = s'.activeStreams + countAcquiring s'.recvPhaseL)
    (hZero : s'.recvMgr.numSlots = 0 → s'.pipe.recv.pending = 0 ∧ s'.pipe.recv.issued = 0)
    (hPendEq : s'.pipe.h2dPendingL = s.pipe.h2dPendingL := by rfl)
    (hIssEq : s'.pipe.h2dIssuedL = s.pipe.h2dIssuedL := by rfl)
    (hRdyEq : s'.pipe.h2dReadyL = s.pipe.h2dReadyL := by rfl)
    (hRetEq : s'.pipe.recv.retired = s.pipe.recv.retired := by rfl)
    (hLandL : s'.pipe.landedL = s.pipe.landedL := by rfl)
    (hN : s'.pipe.numLayers = s.pipe.numLayers := by rfl)
    (hSlots : s'.decodeSlots = s.decodeSlots := by rfl) : RecvSideInv s' := by
  have hRetSame : isRetiredPhase newP = isRetiredPhase oldP := by
    cases oldP <;> cases newP <;> simp_all [isRetiredPhase]
  have hnp : s.pipe.h2dPendingL[l]? ≠ some true := fun ht =>
    hOldNP (Option.some.inj (hph.symm.trans ((hr.pendingPhase l).mp ht)))
  have hni : s.pipe.h2dIssuedL[l]? ≠ some true := fun ht =>
    ((hr.issuedPhase l).mp ht).elim
      (fun h => hOldNR (Option.some.inj (hph.symm.trans h)))
      (fun h => hOldNRet (Option.some.inj (hph.symm.trans h)))
  exact setPhase_preserves_recv hr (lt_length_of_getElem?_eq hph) hPhase
    hPool hOldNotDone hLookupNe hPushes
    (by rw [hPhase, countPred_set_same (p := isRetiredPhase) hph hRetSame, hRetEq]; exact hr.cntRetired)
    hZero (fun h => absurd h hNewNRet)
    (by rw [hPendEq]; exact ⟨fun ht => absurd ht hnp, fun hn => absurd hn hNewNP⟩)
    (by rw [hIssEq]; exact ⟨fun ht => absurd ht hni, fun hn => hn.elim (absurd · hNewNR) (absurd · hNewNRet)⟩)
    (fun hl => absurd hl hNotLand) (fun _ hl _ => absurd hl hNotLand)
    (fun _ _ => by rw [hPendEq]) (fun _ _ => by rw [hIssEq]) hSlots hLandL hRdyEq hN

/-- Every `BoundedPipeline` step preserves `RecvSideInv`. -/
theorem step_recvSideInv {s s' : BoundedPipeline} {e : Ev}
    (hp : Pipeline.Inv s.pipe) (hr : RecvSideInv s)
    (hwire : ∀ (k : Nat) (c : Cell), s.wireB[k]? = some c → c = .blank ∨ c = .kv k)
    (hst : step s e = some s') : RecvSideInv s' := by
  have hacc := hp.recv.accounted; unfold Recv.Accounted at hacc
  cases e with
  | recvAcquireBegin l =>
    simp only [step, recvAcquireBegin] at hst
    have hbegin : ∀ (hadLease : Bool) (oldP : RecvLayerPhase),
        s.recvPhaseL[l]? = some oldP → isAcquiringPhase oldP = false →
        ownsRecvSlot (.waiting hadLease) = ownsRecvSlot oldP →
        oldP ≠ .h2dPending → oldP ≠ .h2dRunning → oldP ≠ .h2dRetired →
        s.pipe.landedL[l]? ≠ some true →
        ((Pipeline.step s.pipe (.recv .pushBegin)).map fun p =>
          { s with pipe := p, recvPhaseL := s.recvPhaseL.set l (.waiting hadLease) }) = some s' →
        RecvSideInv s' := by
      intro hadLease oldP hph hold hown hnp hnr hnret hNotLand hmap
      simp only [Pipeline.step, Pipeline.recvStep, Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hmap
      rcases hmap with ⟨_, ⟨_, ⟨_, hlf, rfl⟩, rfl⟩, rfl⟩
      obtain ⟨hd_f, hdr_f, rfl⟩ := beginOp_eq hlf
      have hcnt := countPred_set_true (p := isAcquiringPhase) hph (newV := .waiting hadLease) hold rfl
      have hlt_l := lt_length_of_getElem?_eq hph
      exact acquirePhase_preserves_recv hr hph rfl hnp hnr hnret nofun nofun nofun hNotLand
        (hr.pool.activeOp hdr_f hd_f hd_f fun u => by
          by_cases hu : u = l
          · subst hu; simp [recvOwnsSlot, List.getElem?_set_self hlt_l, hph, hown]
          · simp [recvOwnsSlot, List.getElem?_set_ne (Ne.symm hu)])
        (fun _ => hd_f) (fun _ _ _ => rfl) (by simp [hr.pushesEq, hcnt]; omega) hr.zeroSlots
    cases hph : s.recvPhaseL[l]? with
    | none => simp [hph] at hst
    | some ph =>
      cases ph with
      | idle =>
        exact hbegin false .idle hph rfl rfl nofun nofun nofun
          (fun hl => by have := hr.landedPhase l hl; rw [hph] at this; simp at this)
          (by simpa [hph] using hst)
      | leasedForStream =>
        simp only [hph] at hst; split at hst <;> try cases hst
        rename_i hland_f
        exact hbegin true .leasedForStream hph rfl rfl nofun nofun nofun (by simp [hland_f]) hst
      | _ => simp [hph] at hst
  | recvAcquireWait l t =>
    simp only [step, recvAcquireWait] at hst
    cases hph : s.recvPhaseL[l]? with
    | none => simp [hph] at hst
    | some ph =>
      cases ph with
      | waiting hadLease =>
        simp only [hph] at hst
        let restoreP : RecvLayerPhase := if hadLease then .leasedForStream else .idle
        have hlt_l := lt_length_of_getElem?_eq hph
        have hcnt_dec := countPred_set_false (p := isAcquiringPhase) hph (newV := restoreP)
          rfl (by cases hadLease <;> rfl)
        have hd_f := not_done_of_inFlight_pos hp.recv.life (by have := hr.pushesEq; omega)
        have hNotLand : s.pipe.landedL[l]? ≠ some true := fun hl => by
          have := hr.landedPhase l hl; rw [hph] at this; simp at this
        have hrestore : ∀ {p : Pipeline},
            p.numLayers = s.pipe.numLayers → p.recv.pushes = s.pipe.recv.pushes - 1 →
            p.recv.retired = s.pipe.recv.retired →
            p.recv.pending = s.pipe.recv.pending → p.recv.issued = s.pipe.recv.issued →
            p.landedL = s.pipe.landedL → p.h2dPendingL = s.pipe.h2dPendingL →
            p.h2dIssuedL = s.pipe.h2dIssuedL → p.h2dReadyL = s.pipe.h2dReadyL →
            RecvSideInv { syncPipe s p with recvPhaseL := s.recvPhaseL.set l restoreP } := by
          intro p hN hPsh hRet hPendC hIssC hL hP hI hR
          exact acquirePhase_preserves_recv hr hph rfl nofun nofun nofun
            (by cases hadLease <;> decide) (by cases hadLease <;> decide) (by cases hadLease <;> decide)
            hNotLand
            (hr.pool.sync rfl rfl (fun _ => hd_f) fun _ u => by
              by_cases hu : u = l
              · subst hu
                cases hadLease <;> simp [recvOwnsSlot, restoreP, List.getElem?_set_self hlt_l, hph, ownsRecvSlot]
              · simp [recvOwnsSlot, List.getElem?_set_ne (Ne.symm hu)])
            (fun _ => hd_f)
            (fun hnd u _ => StagingLayerManager.sync_lookup_of_not_done _ hnd u)
            (by dsimp only [syncPipe]; rw [hPsh, hr.pushesEq]; omega)
            (fun hz => by dsimp only [syncPipe] at hz ⊢; rw [hPendC, hIssC]; exact hr.zeroSlots (by simpa using hz))
            hP hI hR hRet hL hN
        split at hst
        · cases hst
        · rename_i slot mgr' hacq; simp only [Option.some.injEq] at hst; subst hst
          obtain ⟨_, hdr_f, _⟩ := s.recvMgr.acquire_leased_cases hacq
          obtain ⟨_, hslt_m, hnum', _, hlk_ne⟩ := s.recvMgr.acquire_leased_lookup hr.mgrInv hacq
          exact acquirePhase_preserves_recv hr hph rfl nofun nofun nofun nofun nofun nofun hNotLand
            (hr.pool.acquire hacq hdr_f hd_f hd_f
              (by simp [recvOwnsSlot, List.getElem?_set_self hlt_l, ownsRecvSlot])
              fun u hu => by simp [recvOwnsSlot, List.getElem?_set_ne (Ne.symm hu)])
            (fun _ => hd_f) (fun _ => hlk_ne)
            (by dsimp only; rw [countPred_set_same (p := isAcquiringPhase) hph (newV := .leasedInAcquire) rfl]; exact hr.pushesEq)
            (fun hz => by dsimp only at hz; omega)
        · simp only [Pipeline.step, Pipeline.recvStep, Recv.step, Recv.pushEnd, Option.map_eq_some_iff] at hst
          rcases hst with ⟨_, ⟨_, hrcv, rfl⟩, rfl⟩; split at hrcv <;> cases hrcv
          exact hrestore rfl rfl rfl rfl rfl rfl rfl rfl rfl
        all_goals
          rcases Option.map_eq_some_iff.mp hst with ⟨_, hb, rfl⟩
          rcases Option.bind_eq_some_iff.mp hb with ⟨_, hp₁, hp₂⟩
          simp only [Pipeline.step, Pipeline.recvStep, Recv.step, Recv.cancel,
                     Recv.pushEnd, Option.map_eq_some_iff, Option.some.injEq] at hp₁ hp₂
          rcases hp₁ with ⟨_, rfl, rfl⟩; rcases hp₂ with ⟨_, hrcv₂, rfl⟩
          split at hrcv₂ <;> cases hrcv₂
          exact hrestore rfl rfl rfl rfl rfl rfl rfl rfl rfl
      | _ => simp [hph] at hst
  | recvAcquireFinish l =>
    simp only [step, recvAcquireFinish] at hst; split at hst <;> try cases hst
    rename_i hph
    have hlt_l := lt_length_of_getElem?_eq hph
    have hcnt_ab := countPred_set_false (p := isAcquiringPhase) hph (newV := .aborted) rfl rfl
    have hd_f := not_done_of_inFlight_pos hp.recv.life (by have := hr.pushesEq; omega)
    have hNotLand : s.pipe.landedL[l]? ≠ some true := fun hl => by
      have := hr.landedPhase l hl; rw [hph] at this; simp at this
    simp only [Pipeline.step, Pipeline.recvStep, Recv.step, Recv.pushEnd, Option.map_eq_some_iff] at hst
    rcases hst with ⟨_, ⟨_, hrcv, rfl⟩, hpost⟩; split at hrcv <;> cases hrcv
    split at hpost
    · subst hpost
      have hnp : s.pipe.h2dPendingL[l]? ≠ some true := fun ht => by
        have := (hr.pendingPhase l).mp ht; rw [hph] at this; cases this
      have hni : s.pipe.h2dIssuedL[l]? ≠ some true := fun ht => by
        rcases (hr.issuedPhase l).mp ht with h1 | h2 <;> (rw [hph] at *; cases ‹_›)
      exact releasePhase_preserves_recv hr hph hd_f rfl rfl
        (by dsimp only; rw [hr.pushesEq]; omega)
        (by dsimp only; rw [countPred_set_same (p := isRetiredPhase) hph (newV := .aborted) rfl]; exact hr.cntRetired)
        hr.zeroSlots nofun (by simp [hnp]) (by simp [hni])
    · rename_i hndd; subst hpost
      have hdr_f : s.pipe.recv.life.draining = false := by cases h : s.pipe.recv.life.draining <;> simp_all
      have hnd' : s.pipe.recv.life.endOpLocked.done = false := by
        unfold Lifecycle.endOpLocked Lifecycle.settleLocked; split <;> simp [hdr_f, hd_f]
      have hndr' : s.pipe.recv.life.endOpLocked.draining = false := by
        unfold Lifecycle.endOpLocked Lifecycle.settleLocked; split <;> simp [hdr_f]
      have hcnt_dec := countPred_set_false (p := isAcquiringPhase) hph (newV := .leasedForStream) rfl rfl
      exact acquirePhase_preserves_recv hr hph rfl nofun nofun nofun nofun nofun nofun hNotLand
        (hr.pool.activeOp hndr' hnd' hd_f fun u => by
          by_cases hu : u = l
          · subst hu; simp [recvOwnsSlot, List.getElem?_set_self hlt_l, hph, ownsRecvSlot]
          · simp [recvOwnsSlot, List.getElem?_set_ne (Ne.symm hu)])
        (fun _ => hd_f) (fun _ _ _ => rfl) (by dsimp only; rw [hr.pushesEq]; omega) hr.zeroSlots
  | h2dDone l ok =>
    simp only [step, h2dDone] at hst; split at hst <;> try cases hst
    rename_i hg; rcases hg with ⟨hph, hrd_t⟩
    obtain ⟨hiss_t, _, hd_f⟩ := recv_running_active hp hr hph
    have hnp : s.pipe.h2dPendingL[l]? ≠ some true := fun ht => by
      have := (hr.pendingPhase l).mp ht; rw [hph] at this; cases this
    simp only [Pipeline.step, Pipeline.recvStep, Recv.step, Recv.h2dDone, Option.map_eq_some_iff] at hst
    rcases hst with ⟨_, ⟨rcv, hrcv, rfl⟩, rfl⟩; split at hrcv <;> try cases hrcv
    simp only [hd_f, Bool.false_eq_true, ↓reduceIte] at hrcv
    obtain ⟨hpush', hret', hpend', hiss'⟩ :
        rcv.pushes = s.pipe.recv.pushes ∧ rcv.retired = s.pipe.recv.retired + 1 ∧
        rcv.pending = s.pipe.recv.pending ∧ rcv.issued = s.pipe.recv.issued := by
      split at hrcv <;> simp only [Option.some.injEq] at hrcv <;> subst hrcv <;> exact ⟨rfl, rfl, rfl, rfl⟩
    exact releasePhase_preserves_recv hr hph hd_f rfl rfl
      (by dsimp only; rw [hpush', countPred_set_same (p := isAcquiringPhase) hph (newV := .h2dRetired) rfl]; exact hr.pushesEq)
      (by dsimp only; rw [countPred_set_true (p := isRetiredPhase) hph (newV := .h2dRetired) rfl rfl, hr.cntRetired, hret'])
      (fun hz => by dsimp only; rw [hpend', hiss']; exact hr.zeroSlots hz)
      (fun _ => hrd_t) (by simp [hnp]) (by simp [hiss_t])
  | pipe pe =>
    simp only [step] at hst
    cases pe with
    | recv re =>
      cases re with
      | pushBegin =>
        simp only [pipeStep, recvPushBegin, Pipeline.step, Pipeline.recvStep,
                   Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hst
        rcases hst with ⟨_, ⟨_, ⟨_, hlf, rfl⟩, rfl⟩, rfl⟩
        obtain ⟨hd_f, hdr_f, rfl⟩ := beginOp_eq hlf
        exact ⟨hr.pool.activeOp hdr_f hd_f hd_f (fun _ => Iff.rfl), hr.lenPhase,
          (by simp [hr.pushesEq]; omega), hr.cntRetired, hr.zeroSlots, hr.retiredReady,
          hr.pendingPhase, hr.issuedPhase, hr.landedPhase, fun _ l hl hp => hr.slotsGood hd_f l hl hp⟩
      | pushEnd =>
        simp only [pipeStep, recvPushEnd] at hst; split at hst <;> try cases hst
        simp only [Pipeline.step, Pipeline.recvStep, Recv.step, Recv.pushEnd, Option.map_eq_some_iff] at hst
        rcases hst with ⟨_, ⟨_, hrcv, rfl⟩, rfl⟩; split at hrcv <;> cases hrcv
        exact syncMgr_preserves_recv hr (by simp [syncPipe, hr.pushesEq]; omega) Lifecycle.endOpLocked_done_mono
      | h2dBegin | h2dIssue _ | h2dReady => simp [pipeStep, Pipeline.step, Pipeline.recvStep] at hst
      | h2dDone _ => simp [pipeStep] at hst
      | _ =>
        simp only [pipeStep, Option.map_eq_some_iff] at hst
        rcases hst with ⟨p, hp0, rfl⟩
        exact passive_recv_preserves hr hp0 nofun nofun nofun
    | land l =>
      simp only [pipeStep, land] at hst; split at hst <;> try cases hst
      rename_i hg; rcases hg with ⟨_, hph, hland_f⟩
      cases hw : s.wireB[l]? with
      | none => simp [hw] at hst
      | some c =>
        simp only [hw] at hst; split at hst <;> try cases hst
        rename_i hc_nb
        rcases Option.map_eq_some_iff.mp hst with ⟨p, hp0, hpost⟩
        have hc_kv : c = .kv l := (hwire l c hw).resolve_left hc_nb; subst hc_kv
        have hd_f := not_done_of_inFlight_pos hp.recv.life (by have := hr.pushesEq; omega)
        obtain ⟨slot, hlk, hslt⟩ := recv_leasedForStream_slot hp hr hd_f hph
        simp only [Pipeline.step, Pipeline.land] at hp0; split at hp0 <;> try cases hp0
        cases hw_p : s.pipe.wire[l]? with
        | none => simp [hw_p] at hp0
        | some cp =>
          simp only [hw_p] at hp0; split at hp0 <;> cases hp0
          rw [withSlot_eq hlk hslt] at hpost; subst hpost
          refine ⟨hr.pool.setSlot _ (fun _ _ => Iff.rfl), hr.lenPhase,
            hr.pushesEq, hr.cntRetired, hr.zeroSlots,
            hr.retiredReady, hr.pendingPhase, hr.issuedPhase, ?_, ?_⟩
          · intro u hl'
            rcases mem_of_set_true hl' with rfl | hold
            · exact Or.inl hph
            · exact hr.landedPhase u hold
          · intro _ u hl' hp_u
            rcases mem_of_set_true hl' with rfl | hold
            · exact ⟨slot, hlk, List.getElem?_set_self hslt⟩
            · have hne_u : u ≠ l := fun heq => by subst heq; rw [hland_f] at hold; cases hold
              obtain ⟨slot', hlk', hkv⟩ := hr.slotsGood hd_f u hold hp_u
              exact ⟨slot', hlk', by simpa [List.getElem?_set_ne (Ne.symm fun heq : slot' = slot =>
                hne_u (StagingLayerManager.lookup_injective hr.mgrInv hlk' (heq ▸ hlk)))] using hkv⟩
    | h2dBegin l =>
      simp only [pipeStep, h2dBegin] at hst; split at hst <;> try cases hst
      rename_i hg; rcases hg with ⟨_, hland_t, hclm_f, hph⟩
      have hd_f := not_done_of_inFlight_pos hp.recv.life (by have := hr.pushesEq; omega)
      obtain ⟨slot, hlk, hslt⟩ := recv_leasedForStream_slot hp hr hd_f hph
      rw [withSlotOpt_eq hlk hslt] at hst
      have hpos_slots : 0 < s.recvMgr.numSlots := by rw [hr.lenSlots] at hslt; omega
      have hlt_l : l < s.recvPhaseL.length := lt_length_of_getElem?_eq hph
      have hnp : s.pipe.h2dPendingL[l]? ≠ some true := fun ht => by
        have := (hr.pendingPhase l).mp ht; rw [hph] at this; cases this
      have hni : s.pipe.h2dIssuedL[l]? ≠ some true := fun ht => by
        rcases (hr.issuedPhase l).mp ht with h1 | h2 <;> (rw [hph] at *; cases ‹_›)
      split at hst
      · simp only [Option.some.injEq] at hst; subst hst
        exact releasePhase_preserves_recv hr hph hd_f rfl rfl
          (by dsimp only; rw [countPred_set_same (p := isAcquiringPhase) hph (newV := .aborted) rfl]; exact hr.pushesEq)
          (by dsimp only; rw [countPred_set_same (p := isRetiredPhase) hph (newV := .aborted) rfl]; exact hr.cntRetired)
          hr.zeroSlots nofun (by simp [hnp]) (by simp [hni])
      · rename_i hndd
        have hdr_f : s.pipe.recv.life.draining = false := by cases h : s.pipe.recv.life.draining <;> simp_all
        simp only [Pipeline.step, Pipeline.h2dBegin, hland_t, hclm_f, and_self,
                   ↓reduceIte, Recv.step, Recv.h2dBegin, Option.map_eq_some_iff] at hst
        rcases hst with ⟨_, ⟨_, hrcv, rfl⟩, rfl⟩; split at hrcv <;> try cases hrcv
        rcases Option.map_eq_some_iff.mp hrcv with ⟨_, hlf, rfl⟩
        obtain ⟨_, _, rfl⟩ := beginOp_eq hlf
        have hlt_pend : l < s.pipe.h2dPendingL.length := by rw [hp.len_h2dPendingL, ← hr.lenPhase]; exact hlt_l
        exact setPhase_preserves_recv hr hlt_l rfl
          (hr.pool.activeOp hdr_f hd_f hd_f fun u => by
            by_cases hu : u = l
            · subst hu; simp [recvOwnsSlot, List.getElem?_set_self hlt_l, hph, ownsRecvSlot]
            · simp [recvOwnsSlot, List.getElem?_set_ne (Ne.symm hu)])
          (fun _ => hd_f) (fun _ _ _ => rfl)
          (by dsimp only; rw [countPred_set_same (p := isAcquiringPhase) hph (newV := .h2dPending) rfl]; exact hr.pushesEq)
          (by dsimp only; rw [countPred_set_same (p := isRetiredPhase) hph (newV := .h2dPending) rfl]; exact hr.cntRetired)
          (fun hz => by dsimp only at hz; omega)
          nofun (by rw [List.getElem?_set_self hlt_pend]; simp) (by simp [hni]) (fun _ => by simp)
          (fun _ hl_t _ => hr.slotsGood hd_f l hl_t (Or.inl hph))
          (fun u hu => List.getElem?_set_ne (Ne.symm hu))
    | h2dIssue l ok =>
      simp only [pipeStep, h2dIssue] at hst; split at hst <;> try cases hst
      rename_i hph
      have hpend_t : s.pipe.h2dPendingL[l]? = some true := (hr.pendingPhase l).mpr hph
      have hlt_l : l < s.recvPhaseL.length := lt_length_of_getElem?_eq hph
      have hlt_pend : l < s.pipe.h2dPendingL.length := lt_length_of_getElem?_eq hpend_t
      have hlt_iss : l < s.pipe.h2dIssuedL.length := by rw [hp.len_h2dIssuedL, ← hr.lenPhase]; exact hlt_l
      have hiss_f : s.pipe.h2dIssuedL[l]? = some false := by
        have heq := List.getElem?_eq_getElem hlt_iss
        cases hb : s.pipe.h2dIssuedL[l]
        · rw [heq, hb]
        · rw [hb] at heq; have := (hp.issued_claimed l heq).2; rw [hpend_t] at this; cases this
      have hpend_pos : 0 < s.pipe.recv.pending := by
        have hc := countTrue_set_false hpend_t; rw [hp.cnt_h2dPending] at hc; omega
      have hnoZero : ∀ {P : Prop}, s.recvMgr.numSlots = 0 → P := fun hz => by
        have := (hr.zeroSlots hz).1; omega
      have hd_f := not_done_of_inFlight_pos hp.recv.life (by omega)
      simp only [Pipeline.step, Pipeline.h2dIssue, hpend_t, ↓reduceIte,
                 Recv.step, Recv.h2dIssue, Option.map_eq_some_iff] at hst
      rcases hst with ⟨_, ⟨rcv, hrcv, rfl⟩, hpost⟩
      simp only [show s.pipe.recv.pending ≠ 0 by omega, ↓reduceIte] at hrcv
      cases hbr : (s.pipe.recv.life.done || s.pipe.recv.life.draining || !ok) with
      | true =>
        simp only [hbr, ↓reduceIte] at hpost; subst hpost
        obtain ⟨hpush', hret'⟩ : rcv.pushes = s.pipe.recv.pushes ∧ rcv.retired = s.pipe.recv.retired := by
          split at hrcv
          · simp only [Option.some.injEq] at hrcv; subst hrcv; exact ⟨rfl, rfl⟩
          · have hok_f : ok = false := by
              cases hd : s.pipe.recv.life.done <;> cases hdr : s.pipe.recv.life.draining <;>
                cases hok : ok <;> simp_all
            simp only [hok_f, Bool.false_eq_true, ↓reduceIte, Option.some.injEq] at hrcv
            subst hrcv; exact ⟨rfl, rfl⟩
        exact releasePhase_preserves_recv hr hph hd_f rfl rfl
          (by dsimp only; rw [hpush', countPred_set_same (p := isAcquiringPhase) hph (newV := .aborted) rfl]; exact hr.pushesEq)
          (by dsimp only; rw [countPred_set_same (p := isRetiredPhase) hph (newV := .aborted) rfl, hret']; exact hr.cntRetired)
          (fun hz => hnoZero hz)
          nofun (by rw [List.getElem?_set_self hlt_pend]; simp) (by simp [hiss_f])
          (fun _ => by simp) (fun u hu => List.getElem?_set_ne (Ne.symm hu))
      | false =>
        have hdr_f : s.pipe.recv.life.draining = false := by cases h : s.pipe.recv.life.draining <;> simp_all
        have hok_t : ok = true := by cases h : ok <;> simp_all
        simp only [hbr, Bool.false_eq_true, ↓reduceIte] at hpost; subst hpost
        simp [hd_f, hdr_f, hok_t] at hrcv; subst hrcv
        exact setPhase_preserves_recv hr hlt_l rfl
          (hr.pool.activeOp hdr_f hd_f hd_f fun u => by
            by_cases hu : u = l
            · subst hu; simp [recvOwnsSlot, List.getElem?_set_self hlt_l, hph, ownsRecvSlot]
            · simp [recvOwnsSlot, List.getElem?_set_ne (Ne.symm hu)])
          (fun _ => hd_f) (fun _ _ _ => rfl)
          (by dsimp only; rw [countPred_set_same (p := isAcquiringPhase) hph (newV := .h2dRunning) rfl]; exact hr.pushesEq)
          (by dsimp only; rw [countPred_set_same (p := isRetiredPhase) hph (newV := .h2dRunning) rfl]; exact hr.cntRetired)
          (fun hz => hnoZero hz)
          nofun (by rw [List.getElem?_set_self hlt_pend]; simp)
          (by rw [List.getElem?_set_self hlt_iss]; simp) (fun _ => by simp)
          (fun _ hl_t _ => hr.slotsGood hd_f l hl_t (Or.inr (Or.inl hph)))
          (fun u hu => List.getElem?_set_ne (Ne.symm hu))
          (fun u hu => List.getElem?_set_ne (Ne.symm hu))
    | h2dReady l =>
      simp only [pipeStep, h2dReady] at hst; split at hst <;> try cases hst
      rename_i hg; rcases hg with ⟨hph, hrd_f⟩
      obtain ⟨_, slot, hlk, hslt, _⟩ := recv_h2dReady_slot hp hr hph hrd_f
      have hiss_t : s.pipe.h2dIssuedL[l]? = some true := (hr.issuedPhase l).mpr (Or.inl hph)
      simp only [Pipeline.step, Pipeline.h2dReady, hiss_t, hrd_f, and_self,
                 ↓reduceIte, Recv.step, Recv.h2dReady, Option.map_eq_some_iff] at hst
      rcases hst with ⟨_, ⟨_, hrcv, rfl⟩, hpost⟩; split at hrcv <;> cases hrcv
      rw [withSlot_eq hlk hslt] at hpost; subst hpost
      exact ⟨hr.pool, hr.lenPhase, hr.pushesEq, hr.cntRetired, hr.zeroSlots,
        fun u hu => set_true_of_mem hrd_f (hr.retiredReady u hu),
        hr.pendingPhase, hr.issuedPhase, hr.landedPhase, hr.slotsGood⟩
    | reseatDecodeStaging =>
      simp only [pipeStep, Pipeline.step, Pipeline.reseatDecodeStaging, Option.map_eq_some_iff] at hst
      rcases hst with ⟨_, hp0, rfl⟩; split at hp0 <;> cases hp0
      have hd_t : s.pipe.recv.life.done = true := by
        have := hp.recv.life.staging; cases hd : s.pipe.recv.life.done <;> simp_all
      exact ⟨hr.pool.reseat Cell.junk hd_t, hr.lenPhase, hr.pushesEq, hr.cntRetired, hr.zeroSlots,
        hr.retiredReady, hr.pendingPhase, hr.issuedPhase, hr.landedPhase, fun hnd => by simp [hd_t] at hnd⟩
    | _ =>
      simp only [pipeStep, sendD2hBegin, sendD2hIssue, sendWake, sendH2hIssue, d2hReady, h2hDone,
                 withSlot] at hst
      repeat' first
        | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hp0, hpost⟩
           obtain ⟨hN, hRcv, hL, hP, hI, hR⟩ := pipe_recv_unchanged hp0 nofun nofun nofun nofun nofun
           (repeat' split at hpost) <;> subst hpost <;>
           exact hr.frame hN hRcv hL hP hI hR (by first | rfl | simp [syncPipe, hRcv]))
        | (rw [Option.some.injEq] at hst; subst hst; exact hr.frame)
        | split at hst | cases hst | contradiction
  | _ =>
    simp only [step, sendAcquireBegin, sendAcquireDone] at hst
    repeat' first
      | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hp0, rfl⟩
         obtain ⟨hN, hRcv, hL, hP, hI, hR⟩ := pipe_recv_unchanged hp0 nofun nofun nofun nofun nofun
         exact hr.frame hN hRcv hL hP hI hR (by first | rfl | simp [syncPipe, hRcv]))
      | (rw [Option.some.injEq] at hst; subst hst; exact hr.frame)
      | split at hst | cases hst

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

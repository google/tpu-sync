import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Lemmas

/-! # `LayerStaging`: producer (`TransferSendSession`) bounded-slot invariant

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. Establishes exact
correspondence between `sendMgr` leases, `StartPush` loop phases, `Send`
counters, and `prefillSlots` contents across all `BoundedPipeline` transitions. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager SlotPoolInv)

/-- `StartPush`'s D2H loop thread owns layer `l`'s slot lease between
`AcquireStagingSlot(l)` returning (`leasedBeforeD2h`) and `D2hSyncDispatch(l)`
completing (`d2hPending = true`). -/
abbrev sendLoopOwnsSlot (s : BoundedPipeline) (l : Nat) : Prop :=
  l = s.pipe.send.d2hIssued ∧ (s.sendLoop = .leasedBeforeD2h ∨ s.pipe.send.d2hPending = true)

/-- After `D2hSyncDispatch(l)` succeeds (`l < d2hIssued`), the async D2H →
`push_pool` → H2H pipeline owns layer `l`'s slot lease until released by an
aborted `SendNextLayer` callback (`woken ≤ l`), a draining `push_pool` task
(`l = h2hIssued ∧ pooled = 1`), or `H2hWriteDirectAsync`'s completion callback
(`l < h2hIssued ∧ h2hRetiredL[l] = false`). -/
abbrev sendPipeOwnsSlot (s : BoundedPipeline) (l : Nat) : Prop :=
  l < s.pipe.send.d2hIssued ∧
    ((l < s.pipe.send.h2hIssued ∧ s.pipe.h2hRetiredL[l]? = some false) ∨
     (l = s.pipe.send.h2hIssued ∧ s.pipe.send.pooled = 1) ∨
     s.pipe.send.woken ≤ l)

/-- Exact characterization of which logical layer `l` owns a live lease in
`sendMgr` while the send session has not settled (`done = false`). -/
def sendOwnsSlot (s : BoundedPipeline) (l : Nat) : Prop :=
  sendLoopOwnsSlot s l ∨ sendPipeOwnsSlot s l

/-- Producer-side bounded-slot invariant, built on `SlotPoolInv`. -/
structure SendSideInv (s : BoundedPipeline) : Prop where
  pool : SlotPoolInv s.pipe.send.life s.sendMgr s.prefillSlots (sendOwnsSlot s)
  loopBeforeStart : s.pipe.send.started = false → s.sendLoop = .nextIter
  loopActive :
    (s.sendLoop = .acquiring ∨ s.sendLoop = .leasedBeforeD2h) →
      s.pipe.send.started = true ∧ s.pipe.send.d2hPending = false ∧
        s.pipe.send.d2hIssued < s.pipe.numLayers
  pendingLoop : s.pipe.send.d2hPending = true → s.sendLoop = .nextIter
  wokenLoop : s.pipe.send.d2hIssued < s.pipe.send.woken → s.sendLoop = .exited
  zeroSlots : s.sendMgr.numSlots = 0 →
    s.sendLoop ≠ .leasedBeforeD2h ∧ s.pipe.send.d2hPending = false ∧ s.pipe.send.d2hIssued = 0
  slotsGood : s.pipe.send.life.done = false →
    ∀ (l slot : Nat), s.pipe.d2hReadyL[l]? = some true →
      s.sendMgr.lookup l = some slot → s.prefillSlots[slot]? = some (.kv l)

namespace SendSideInv
theorem mgrInv {s : BoundedPipeline} (h : SendSideInv s) : s.sendMgr.Inv := h.pool.mgrInv
theorem lenSlots {s : BoundedPipeline} (h : SendSideInv s) : s.prefillSlots.length = s.sendMgr.numSlots := h.pool.lenSlots
theorem drainingShutdown {s : BoundedPipeline} (h : SendSideInv s) :
    s.pipe.send.life.draining = true → s.sendMgr.shutdown = true := h.pool.drainingShutdown
theorem doneLeases {s : BoundedPipeline} (h : SendSideInv s) :
    s.pipe.send.life.done = true → s.sendMgr.leases = [] := h.pool.doneLeases
theorem leasesIff {s : BoundedPipeline} (h : SendSideInv s) :
    s.pipe.send.life.done = false → ∀ l : Nat, (s.sendMgr.lookup l).isSome = true ↔ sendOwnsSlot s l := h.pool.leasesIff

theorem frame {s s' : BoundedPipeline} (h : SendSideInv s)
    (hN : s'.pipe.numLayers = s.pipe.numLayers := by rfl)
    (hSnd : s'.pipe.send = s.pipe.send := by rfl)
    (hRdyL : s'.pipe.d2hReadyL = s.pipe.d2hReadyL := by rfl)
    (hRetL : s'.pipe.h2hRetiredL = s.pipe.h2hRetiredL := by rfl)
    (hMgr : s'.sendMgr = s.sendMgr := by rfl)
    (hSlots : s'.prefillSlots = s.prefillSlots := by rfl)
    (hLoop : s'.sendLoop = s.sendLoop := by rfl) : SendSideInv s' :=
  ⟨h.pool.frame (congrArg Send.life hSnd) hMgr hSlots
     (fun l => by unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; rw [hSnd, hLoop, hRetL]),
   (by rw [hSnd, hLoop]; exact h.loopBeforeStart),
   (by rw [hLoop, hSnd, hN]; exact h.loopActive),
   (by rw [hSnd, hLoop]; exact h.pendingLoop),
   (by rw [hSnd, hLoop]; exact h.wokenLoop),
   (by rw [hMgr, hLoop, hSnd]; exact h.zeroSlots),
   (fun hnd l slot hrd hl => by
     rw [hSlots]; exact h.slotsGood (hSnd ▸ hnd) l slot (hRdyL ▸ hrd) (hMgr ▸ hl))⟩
end SendSideInv

theorem sendSideInv_init (n ks kr : Nat) : SendSideInv (init n ks kr) :=
  ⟨SlotPoolInv.init ks Cell.junk rfl rfl
     (fun _ => by simp [init, Pipeline.init, Send.init, sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot]),
   (fun _ => rfl), (fun h => by rcases h with h | h <;> cases h),
   (by simp [init, Pipeline.init, Send.init]), (by simp [init, Pipeline.init, Send.init]),
   (fun _ => ⟨nofun, rfl, rfl⟩),
   (fun _ _ _ hrd _ => (not_mem_replicate_false hrd).elim)⟩

theorem sendSideInv_initUnregistered (n ks kr : Nat) : SendSideInv (initUnregistered n ks kr) :=
  (sendSideInv_init n ks kr).frame

theorem send_d2hReady_slot {s : BoundedPipeline} (hp : Pipeline.Inv s.pipe) (hs : SendSideInv s)
    {l : Nat} (hlt : l < s.pipe.send.d2hIssued) (hrd : s.pipe.d2hReadyL[l]? = some false) :
    s.pipe.send.life.done = false ∧
      ∃ slot, s.sendMgr.lookup l = some slot ∧ slot < s.prefillSlots.length := by
  have hcnt := hp.send.counters; unfold Send.CountersOrdered at hcnt
  have hacc := hp.send.accounted; unfold Send.Accounted at hacc
  have hrdyLt : s.pipe.send.d2hReady < s.pipe.send.d2hIssued := by
    if hge : s.pipe.send.d2hIssued ≤ s.pipe.send.d2hReady then
      have := all_true_lt_of_countTrue hp.d2hReady_lt (hp.cnt_d2hReady.symm ▸ hge) l hlt
      rw [hrd] at this; cases this
    else omega
  have hnd := not_done_of_inFlight_pos hp.send.life (by omega)
  have hwok : s.pipe.send.woken ≤ l := by
    if hltW : l < s.pipe.send.woken then
      have := hp.woken_d2hReady l hltW hlt; rw [hrd] at this; cases this
    else omega
  exact ⟨hnd, hs.pool.lookup_of_owns hnd (Or.inr ⟨hlt, Or.inr (Or.inr hwok)⟩)⟩

theorem send_h2hIssue_slot {s : BoundedPipeline} (hp : Pipeline.Inv s.pipe) (hs : SendSideInv s)
    (hp1 : s.pipe.send.pooled = 1) :
    s.pipe.send.life.done = false ∧
      ∃ slot, s.sendMgr.lookup s.pipe.send.h2hIssued = some slot ∧ slot < s.prefillSlots.length := by
  have hacc := hp.send.accounted; unfold Send.Accounted at hacc
  have hnd := not_done_of_inFlight_pos hp.send.life (by omega)
  obtain ⟨_, hlt⟩ := hp.send.chain.pooled_state hp1
  exact ⟨hnd, hs.pool.lookup_of_owns hnd (Or.inr ⟨hlt, Or.inr (Or.inl ⟨rfl, hp1⟩)⟩)⟩

theorem send_h2hDone_slot {s : BoundedPipeline} (hp : Pipeline.Inv s.pipe) (hs : SendSideInv s)
    {l : Nat} (hlt : l < s.pipe.send.h2hIssued) (hret : s.pipe.h2hRetiredL[l]? = some false) :
    s.pipe.send.life.done = false ∧ ∃ slot, s.sendMgr.lookup l = some slot ∧
      slot < s.prefillSlots.length ∧ s.prefillSlots[slot]? = some (.kv l) := by
  have hcnt := hp.send.counters; unfold Send.CountersOrdered at hcnt
  have hacc := hp.send.accounted; unfold Send.Accounted at hacc
  have hwb := hp.send.chain.woken_bounds
  have hrLt : s.pipe.send.h2hRetired < s.pipe.send.h2hIssued := by
    if hge : s.pipe.send.h2hIssued ≤ s.pipe.send.h2hRetired then
      have := all_true_lt_of_countTrue hp.h2hRetired_lt (hp.cnt_h2hRetired.symm ▸ hge) l hlt
      rw [hret] at this; cases this
    else omega
  have hnd := not_done_of_inFlight_pos hp.send.life (by omega)
  have hd2h : l < s.pipe.send.d2hIssued := by omega
  obtain ⟨slot, hlk, hslt⟩ := hs.pool.lookup_of_owns hnd (Or.inr ⟨hd2h, Or.inl ⟨hlt, hret⟩⟩)
  exact ⟨hnd, slot, hlk, hslt, hs.slotsGood hnd l slot (hp.woken_d2hReady l (by omega) hd2h) hlk⟩

private theorem syncMgr_preserves_send {s s' : BoundedPipeline} (hs : SendSideInv s)
    (hOldDone : s.pipe.send.life.done = true → s'.pipe.send.life.done = true)
    (hBefore : s'.pipe.send.started = false → s'.sendLoop = .nextIter)
    (hAct : (s'.sendLoop = .acquiring ∨ s'.sendLoop = .leasedBeforeD2h) →
      s'.pipe.send.started = true ∧ s'.pipe.send.d2hPending = false ∧
        s'.pipe.send.d2hIssued < s'.pipe.numLayers)
    (hPend : s'.pipe.send.d2hPending = true → s'.sendLoop = .nextIter)
    (hWok : s'.pipe.send.d2hIssued < s'.pipe.send.woken → s'.sendLoop = .exited)
    (hZero : s.sendMgr.numSlots = 0 →
      s'.sendLoop ≠ .leasedBeforeD2h ∧ s'.pipe.send.d2hPending = false ∧ s'.pipe.send.d2hIssued = 0)
    (hOwns : ∀ u, sendOwnsSlot s' u ↔ sendOwnsSlot s u)
    (hMgr : s'.sendMgr = syncMgr s.pipe.send.life s'.pipe.send.life s.sendMgr := by simp)
    (hSlots : s'.prefillSlots = s.prefillSlots := by rfl)
    (hRdyL : s'.pipe.d2hReadyL = s.pipe.d2hReadyL := by rfl) : SendSideInv s' := by
  have hOldNotDone : s'.pipe.send.life.done = false → s.pipe.send.life.done = false := fun hnd => by
    cases hd : s.pipe.send.life.done with
    | false => rfl
    | true => have := hOldDone hd; rw [hnd] at this; cases this
  exact ⟨hs.pool.sync hMgr hSlots hOldNotDone (fun _ => hOwns),
    hBefore, hAct, hPend, hWok,
    fun hz => hZero (by simpa [hMgr] using hz),
    fun hnd u slot hrd hlk => by
      rw [hMgr, StagingLayerManager.sync_lookup_of_not_done _ hnd] at hlk; rw [hSlots]
      exact hs.slotsGood (hOldNotDone hnd) u slot (hRdyL ▸ hrd) hlk⟩

private theorem releaseSync_preserves_send {s s' : BoundedPipeline} {l : Nat} (hs : SendSideInv s)
    (hMgr : s'.sendMgr = releaseSync s.pipe.send.life s'.pipe.send.life l s.sendMgr)
    (hOldNotDone : s'.pipe.send.life.done = false → s.pipe.send.life.done = false)
    (hBefore : s'.pipe.send.started = false → s'.sendLoop = .nextIter)
    (hAct : (s'.sendLoop = .acquiring ∨ s'.sendLoop = .leasedBeforeD2h) →
      s'.pipe.send.started = true ∧ s'.pipe.send.d2hPending = false ∧
        s'.pipe.send.d2hIssued < s'.pipe.numLayers)
    (hPend : s'.pipe.send.d2hPending = true → s'.sendLoop = .nextIter)
    (hWok : s'.pipe.send.d2hIssued < s'.pipe.send.woken → s'.sendLoop = .exited)
    (hZero : s.sendMgr.numSlots = 0 →
      s'.sendLoop ≠ .leasedBeforeD2h ∧ s'.pipe.send.d2hPending = false ∧ s'.pipe.send.d2hIssued = 0)
    (hNotL : ¬ sendOwnsSlot s' l)
    (hOther : ∀ u, u ≠ l → (sendOwnsSlot s' u ↔ sendOwnsSlot s u))
    (hSlots : s'.prefillSlots = s.prefillSlots := by rfl)
    (hRdyL : s'.pipe.d2hReadyL = s.pipe.d2hReadyL := by rfl) : SendSideInv s' := by
  obtain ⟨hrel_l, _, _, hrel_ne⟩ := s.sendMgr.release_lookup l hs.mgrInv
  exact ⟨hs.pool.releaseSync hMgr hSlots hOldNotDone (fun _ => hNotL) (fun _ => hOther),
    hBefore, hAct, hPend, hWok,
    fun hz => hZero (by simpa [hMgr] using hz),
    fun hnd u slot hrd hlk' => by
      rw [hMgr, StagingLayerManager.releaseSync_lookup_of_not_done l _ hnd u] at hlk'
      by_cases hu : u = l
      · subst hu; rw [hrel_l] at hlk'; cases hlk'
      · rw [hrel_ne u hu] at hlk'; rw [hSlots]
        exact hs.slotsGood (hOldNotDone hnd) u slot (hRdyL ▸ hrd) hlk'⟩

private theorem passive_send_preserves {s : BoundedPipeline} {p : Pipeline} {se : Send.Ev}
    (hs : SendSideInv s) (hp : s.pipe.sendStep se = some p)
    (hst : se ≠ .start) (hdb : se ≠ .d2hBegin) (hdi : ∀ ok, se ≠ .d2hIssue ok)
    (hwk : ∀ ok, se ≠ .wake ok) (hhi : se ≠ .h2hIssue) : SendSideInv (syncPipe s p) := by
  have ⟨snd, hsnd, hN, hSnd, hRdyL, hRetL⟩ : ∃ snd, Send.step s.pipe.send se = some snd ∧
      p.numLayers = s.pipe.numLayers ∧ p.send = snd ∧
      p.d2hReadyL = s.pipe.d2hReadyL ∧ p.h2hRetiredL = s.pipe.h2hRetiredL := by
    unfold Pipeline.sendStep at hp; (repeat' split at hp) <;> try cases hp
    all_goals rcases Option.map_eq_some_iff.mp hp with ⟨snd, hsnd, rfl⟩; exact ⟨snd, hsnd, rfl, rfl, rfl, rfl⟩
  obtain ⟨hf1, hf2, hf3, hf4, hf5, hf6⟩ := Send.step_lease_counters hsnd hst hdb hdi hwk hhi
  exact syncMgr_preserves_send hs (fun hd => hSnd ▸ Send.step_done_mono hsnd hd)
    (by dsimp [syncPipe]; rw [hSnd, hf1]; exact hs.loopBeforeStart)
    (by dsimp [syncPipe]; rw [hSnd, hf1, hf2, hf3, hN]; exact hs.loopActive)
    (by dsimp [syncPipe]; rw [hSnd, hf2]; exact hs.pendingLoop)
    (by dsimp [syncPipe]; rw [hSnd, hf3, hf6]; exact hs.wokenLoop)
    (by dsimp [syncPipe]; rw [hSnd, hf2, hf3]; exact hs.zeroSlots)
    (fun u => by unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; dsimp [syncPipe]; rw [hSnd, hf2, hf3, hf4, hf5, hf6, hRetL])
    rfl rfl hRdyL

private theorem cancel_exit_preserves {s : BoundedPipeline} {p : Pipeline} (hs : SendSideInv s)
    (hst_t : s.pipe.send.started = true) (hpend_f : s.pipe.send.d2hPending = false)
    (hn : s.sendLoop ≠ .leasedBeforeD2h) (hp0 : Pipeline.step s.pipe (.send .cancel) = some p) :
    SendSideInv { syncPipe s p with sendLoop := .exited } := by
  simp only [Pipeline.step, Pipeline.sendStep, Send.step, Send.cancel,
             Option.map_eq_some_iff, Option.some.injEq] at hp0
  rcases hp0 with ⟨_, rfl, rfl⟩
  exact syncMgr_preserves_send hs (Send.step_done_mono (e := .cancel) rfl)
    (fun h => by simp [syncPipe, Send.finish, hst_t] at h) (fun h => by rcases h with h | h <;> cases h)
    (fun h => by simp [syncPipe, Send.finish, hpend_f] at h) (fun _ => rfl)
    (fun hz => ⟨nofun, hpend_f, (hs.zeroSlots hz).2.2⟩)
    (fun u => by simp [syncPipe, sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, Send.finish, hn, hpend_f]) rfl

@[simp] private theorem endOp_started (s : Send) : s.endOp.started = s.started := by unfold Send.endOp; split <;> rfl
@[simp] private theorem endOp_d2hPending (s : Send) : s.endOp.d2hPending = s.d2hPending := by unfold Send.endOp; split <;> rfl
@[simp] private theorem endOp_d2hIssued (s : Send) : s.endOp.d2hIssued = s.d2hIssued := by unfold Send.endOp; split <;> rfl
@[simp] private theorem endOp_h2hIssued (s : Send) : s.endOp.h2hIssued = s.h2hIssued := by unfold Send.endOp; split <;> rfl
@[simp] private theorem endOp_pooled (s : Send) : s.endOp.pooled = s.pooled := by unfold Send.endOp; split <;> rfl
@[simp] private theorem endOp_woken (s : Send) : s.endOp.woken = s.woken := by unfold Send.endOp; split <;> rfl

/-- Every `BoundedPipeline` step preserves `SendSideInv`. -/
theorem step_sendSideInv {s s' : BoundedPipeline} {e : Ev}
    (hp : Pipeline.Inv s.pipe) (hs : SendSideInv s) (hst : step s e = some s') :
    SendSideInv s' := by
  have hc := hp.send.chain
  have hcnt := hp.send.counters; unfold Send.CountersOrdered at hcnt
  have hacc := hp.send.accounted; unfold Send.Accounted at hacc
  have hpc := hp.send.pull_claimed; unfold Send.PullClaimed at hpc
  cases e with
  | sendAcquireBegin =>
    simp only [step, sendAcquireBegin] at hst; split at hst <;> try cases hst
    rename_i hg; rcases hg with ⟨hst_t, hpend_f, hd2h_lt, hloop_n⟩
    split at hst
    · rcases Option.map_eq_some_iff.mp hst with ⟨p, hp0, rfl⟩
      exact cancel_exit_preserves hs hst_t hpend_f (by simp [hloop_n]) hp0
    · simp only [Option.some.injEq] at hst; subst hst
      exact syncMgr_preserves_send hs id
        (fun h => by simp [hst_t] at h) (fun _ => ⟨hst_t, hpend_f, hd2h_lt⟩)
        (fun h => by simp [hpend_f] at h) (fun hw => by have := hs.wokenLoop hw; simp [hloop_n] at this)
        (fun hz => ⟨nofun, hpend_f, (hs.zeroSlots hz).2.2⟩)
        (fun u => by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, hloop_n, hpend_f])
  | sendAcquireDone t =>
    simp only [step, sendAcquireDone] at hst; split at hst <;> try cases hst
    rename_i hloop_a; obtain ⟨hst_t, hpend_f, hd2h_lt⟩ := hs.loopActive (Or.inl hloop_a)
    split at hst
    · cases hst
    · rename_i slot mgr' hacq; simp only [Option.some.injEq] at hst; subst hst
      obtain ⟨_, hdr_f, _⟩ := s.sendMgr.acquire_leased_cases hacq
      have hd_f : s.pipe.send.life.done = false := by
        cases hd : s.pipe.send.life.done with
        | false => rfl
        | true => have := hp.send.life.done_draining hd; rw [hdr_f] at this; cases this
      obtain ⟨_, hslt_m, hnum', _, hlk_ne⟩ := s.sendMgr.acquire_leased_lookup hs.mgrInv hacq
      exact ⟨hs.pool.acquire hacq hdr_f hd_f hd_f (Or.inl ⟨rfl, Or.inl rfl⟩)
          (fun u hu => by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, hu, hloop_a, hpend_f]),
        fun h => by simp [hst_t] at h, fun _ => ⟨hst_t, hpend_f, hd2h_lt⟩,
        fun h => by simp [hpend_f] at h, fun hw => by have := hs.wokenLoop hw; simp [hloop_a] at this,
        fun hz => by dsimp only at hz; omega,
        fun _ u slot' hrd hlk' => by
          rw [hlk_ne u (by have := hp.d2hReady_lt u hrd; omega)] at hlk'
          exact hs.slotsGood hd_f u slot' hrd hlk'⟩
    all_goals
      rcases Option.map_eq_some_iff.mp hst with ⟨p, hp0, rfl⟩
      exact cancel_exit_preserves hs hst_t hpend_f (by simp [hloop_a]) hp0
  | pipe pe =>
    simp only [step] at hst
    cases pe with
    | send se =>
      cases se with
      | start =>
        simp only [pipeStep, syncPipe, Pipeline.step, Pipeline.sendStep, Send.step,
                   Send.start, Option.map_eq_some_iff] at hst
        rcases hst with ⟨_, ⟨snd, hsnd, rfl⟩, rfl⟩; split at hsnd <;> try cases hsnd
        have hst0 : s.pipe.send.started = false := by cases h : s.pipe.send.started <;> simp_all
        have hloop0 := hs.loopBeforeStart hst0
        have hd2h0 : s.pipe.send.d2hIssued = 0 := by
          cases k : s.pipe.send.d2hIssued
          · rfl
          · exact absurd (hpc.2 (Or.inr (by omega))) (by simp [hst0])
        have hpend0 : s.pipe.send.d2hPending = false := by
          cases h : s.pipe.send.d2hPending
          · rfl
          · exact absurd (hpc.2 (Or.inl h)) (by simp [hst0])
        have hq0 : s.pipe.send.queued = 0 := by
          cases h : s.pipe.send.queued
          · rfl
          · exact absurd (hp.send.queued_started (by omega)) (by simp [hst0])
        have hd0 : s.pipe.send.life.done = false := by cases h : s.pipe.send.life.done <;> simp_all
        have ⟨hf2, hf3, hf4, hf5, hf6⟩ : snd.d2hPending = false ∧ snd.d2hIssued = 0 ∧
            snd.h2hIssued = 0 ∧ snd.pooled = 0 ∧ snd.woken = 0 := by
          split at hsnd <;> simp only [Option.some.injEq] at hsnd <;> subst hsnd
          · dsimp [Send.finish]; exact ⟨hpend0, hd2h0, by omega, by omega, by omega⟩
          · rcases Send.trySendNext_cases { s.pipe.send with started := true } with h | ⟨_, _, _, h⟩ <;>
              (rw [h]; simp [hpend0, hd2h0, show s.pipe.send.h2hIssued = 0 by omega,
                show s.pipe.send.pooled = 0 by omega, show s.pipe.send.woken = 0 by omega])
        exact syncMgr_preserves_send hs (fun hd => by simp [hd] at hd0)
          (fun _ => hloop0) (fun h => by rw [hloop0] at h; rcases h with h | h <;> cases h)
          (fun _ => hloop0) (fun h => by dsimp [syncPipe] at h; omega)
          (fun _ => by dsimp [syncPipe]; simp [hloop0, hf2, hf3])
          (fun u => by
            unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; dsimp [syncPipe]
            simp [hf2, hf3, hf4, hf5, hf6, hd2h0, hloop0, hpend0,
              show s.pipe.send.h2hIssued = 0 by omega, show s.pipe.send.pooled = 0 by omega,
              show s.pipe.send.woken = 0 by omega]) rfl
      | d2hBegin =>
        simp only [pipeStep, sendD2hBegin] at hst; split at hst <;> try cases hst
        rename_i hloop_l; obtain ⟨hst_t, hpend_f, _⟩ := hs.loopActive (Or.inr hloop_l)
        split at hst
        · simp only [Pipeline.step, Pipeline.sendStep, Send.step, Send.cancel,
                     Option.map_eq_some_iff, Option.some.injEq] at hst
          rcases hst with ⟨_, ⟨_, rfl, rfl⟩, rfl⟩
          exact releaseSync_preserves_send hs rfl
            (fun hnd => by
              cases hd : s.pipe.send.life.done
              · rfl
              · have := Lifecycle.finishOnceLocked_done_mono false hd; simp [Send.finish, this] at hnd)
            (fun h => by simp [Send.finish, hst_t] at h) (fun h => by rcases h with h | h <;> cases h)
            (fun h => by simp [Send.finish, hpend_f] at h) (fun _ => rfl)
            (fun hz => ⟨nofun, hpend_f, (hs.zeroSlots hz).2.2⟩)
            (by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, Send.finish, hpend_f])
            (fun u hu => by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, Send.finish, hu, hloop_l, hpend_f])
        · simp only [Pipeline.step, Pipeline.sendStep, Send.step, Send.d2hBegin, Option.map_eq_some_iff] at hst
          rcases hst with ⟨_, ⟨_, hsnd, rfl⟩, rfl⟩; split at hsnd <;> try cases hsnd
          rcases Option.map_eq_some_iff.mp hsnd with ⟨_, hlf, rfl⟩
          obtain ⟨hd_f, hdr_f, rfl⟩ := beginOp_eq hlf
          exact syncMgr_preserves_send hs id
            (fun h => by simp [hst_t] at h) (fun h => by rcases h with h | h <;> cases h)
            (fun _ => rfl) (fun hw => by have := hs.wokenLoop hw; simp [hloop_l] at this)
            (fun hz => absurd hloop_l (hs.zeroSlots hz).1)
            (fun u => by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, hloop_l])
            (StagingLayerManager.sync_of_active _ hdr_f hd_f).symm
      | d2hIssue ok =>
        simp only [pipeStep, sendD2hIssue, Pipeline.step, Pipeline.sendStep,
                   Send.step, Send.d2hIssue, Option.map_eq_some_iff] at hst
        rcases hst with ⟨_, ⟨snd, hsnd, rfl⟩, hpost⟩; split at hsnd <;> try cases hsnd
        have hpend_t : s.pipe.send.d2hPending = true := by cases h : s.pipe.send.d2hPending <;> simp_all
        have hloop_n := hs.pendingLoop hpend_t
        have hst_t := hpc.2 (Or.inl hpend_t)
        have hd_f := not_done_of_inFlight_pos hp.send.life (by simp [hpend_t] at hacc; omega)
        have hwok_le : s.pipe.send.woken ≤ s.pipe.send.d2hIssued := by
          if hltW : s.pipe.send.d2hIssued < s.pipe.send.woken then
            have := hs.wokenLoop hltW; simp [hloop_n] at this
          else omega
        have hnoZero : ∀ {P : Prop}, s.sendMgr.numSlots = 0 → P := fun hz => by
          have := (hs.zeroSlots hz).2.1; simp [hpend_t] at this
        cases ok with
        | true =>
          simp only [↓reduceIte, Option.some.injEq] at hsnd hpost; subst hsnd hpost
          refine syncMgr_preserves_send hs id
            (fun h => by simp [hst_t] at h) (fun h => by rw [hloop_n] at h; rcases h with h | h <;> cases h)
            (fun h => by simp at h) (fun hw => by simp at hw; omega) (fun hz => hnoZero hz) (fun u => ?_)
          unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; simp only [hloop_n, hpend_t]
          by_cases hu : u = s.pipe.send.d2hIssued
          · subst hu; exact ⟨fun _ => Or.inl ⟨rfl, Or.inr trivial⟩,
              fun _ => Or.inr ⟨by omega, Or.inr (Or.inr hwok_le)⟩⟩
          · simp [hu, show u < s.pipe.send.d2hIssued + 1 ↔ u < s.pipe.send.d2hIssued by omega]
        | false =>
          simp only [Bool.false_eq_true, ↓reduceIte, Option.some.injEq] at hsnd hpost; subst hsnd hpost
          exact releaseSync_preserves_send hs rfl (fun _ => hd_f)
            (fun h => by simp [Send.finish, hst_t] at h) (fun h => by rcases h with h | h <;> cases h)
            (fun h => by simp [Send.finish] at h) (fun _ => rfl) (fun hz => hnoZero hz)
            (by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, Send.finish])
            (fun u hu => by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, Send.finish, hu, hloop_n, hpend_t])
      | wake ok =>
        simp only [pipeStep, sendWake] at hst; split at hst <;> try cases hst
        rename_i hgW; rcases Option.map_eq_some_iff.mp hst with ⟨p, hp0, hpost⟩
        simp only [Pipeline.step, Pipeline.sendStep] at hp0; split at hp0 <;> try cases hp0
        rcases Option.map_eq_some_iff.mp hp0 with ⟨snd, hsnd, rfl⟩
        simp only [Send.step, Send.wake] at hsnd; split at hsnd <;> try cases hsnd
        have hp0_eq : s.pipe.send.pooled = 0 := by have := hc.chain_le; omega
        have hwh : s.pipe.send.woken = s.pipe.send.h2hIssued := by
          have := hc.woken_bounds
          if heq : s.pipe.send.woken = s.pipe.send.h2hIssued + 1 then
            have := (hc.woken_ahead heq).1; omega
          else omega
        have hd_f := not_done_of_inFlight_pos hp.send.life (by omega)
        let l := s.pipe.send.woken
        have hwok_post : s.pipe.send.d2hIssued < l + 1 → s.sendLoop = .exited := fun _ => by
          rcases hgW with hrdL | ⟨hex, _, _, _, _⟩
          · have := hp.d2hReady_lt l hrdL; omega
          · exact hex
        by_cases hbr : ok = false ∨ s.pipe.send.life.draining = true
        · simp only [hbr, ↓reduceIte] at hpost; subst hpost
          have ⟨hf1, hf2, hf3, hf4, hf5, hf6⟩ :
              snd.started = s.pipe.send.started ∧ snd.d2hPending = s.pipe.send.d2hPending ∧
              snd.d2hIssued = s.pipe.send.d2hIssued ∧ snd.h2hIssued = s.pipe.send.h2hIssued ∧
              snd.pooled = 0 ∧ snd.woken = l + 1 := by
            rcases hbr with rfl | hdr_t
            · simp only [↓reduceIte, Option.some.injEq] at hsnd; subst hsnd; simp [Send.finish, hp0_eq, l]
            · cases ok <;> simp [hdr_t] at hsnd <;> subst hsnd <;> simp [Send.finish, hp0_eq, l]
          refine releaseSync_preserves_send hs rfl (fun _ => hd_f)
            (hf1 ▸ hs.loopBeforeStart) (by rw [hf1, hf2, hf3]; exact hs.loopActive)
            (hf2 ▸ hs.pendingLoop) (by rw [hf3, hf6]; exact hwok_post)
            (by dsimp only; rw [hf2, hf3]; exact hs.zeroSlots) ?_
            (fun u hu => by
              unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot
              rw [hf2, hf3, hf4, hf5, hf6, hp0_eq]
              simp [l, show s.pipe.send.woken + 1 ≤ u ↔ s.pipe.send.woken ≤ u by omega])
          unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; rw [hf2, hf3, hf4, hf5, hf6]
          rintro (⟨_, hlp⟩ | ⟨_, ⟨_, _⟩ | ⟨_, hp1⟩ | _⟩)
          · rcases hgW with hrdL | ⟨hex, _, _, hpf, _⟩
            · have := hp.d2hReady_lt l hrdL; omega
            · rw [hex, hpf] at hlp; rcases hlp with h | h <;> cases h
          · omega
          · cases hp1
          · omega
        · have hok_t : ok = true := by cases ok <;> simp_all
          have hdr_f : s.pipe.send.life.draining = false := by cases h : s.pipe.send.life.draining <;> simp_all
          simp [hok_t, hdr_f] at hsnd; subst hsnd; simp only [hbr, ↓reduceIte] at hpost; subst hpost
          have hl_lt_d2h : l < s.pipe.send.d2hIssued :=
            hgW.elim (hp.d2hReady_lt l) fun ⟨_, hok_f, _, _, _⟩ => by simp [hok_t] at hok_f
          refine syncMgr_preserves_send hs id
            hs.loopBeforeStart hs.loopActive hs.pendingLoop (by simpa using hwok_post) hs.zeroSlots (fun u => ?_)
          unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; simp only [hp0_eq, Nat.zero_add, hwh]
          by_cases hu : u = s.pipe.send.h2hIssued
          · subst hu; exact ⟨fun _ => Or.inr ⟨hwh ▸ hl_lt_d2h, Or.inr (Or.inr (Nat.le_refl _))⟩,
              fun _ => Or.inr ⟨hwh ▸ hl_lt_d2h, Or.inr (Or.inl ⟨rfl, trivial⟩)⟩⟩
          · simp [hu, show s.pipe.send.h2hIssued + 1 ≤ u ↔ s.pipe.send.h2hIssued ≤ u by omega]
      | h2hIssue =>
        simp only [pipeStep, sendH2hIssue, Pipeline.step, Pipeline.sendStep, Send.step,
                   Send.h2hIssue, Option.map_eq_some_iff] at hst
        rcases hst with ⟨_, ⟨snd, hsnd, rfl⟩, hpost⟩; split at hsnd <;> try cases hsnd
        have hp1 : s.pipe.send.pooled = 1 := by have := hc.chain_le; omega
        obtain ⟨hwh, hlt_d2h⟩ := hc.pooled_state hp1
        obtain ⟨hd_f, slot, hlk, hslt⟩ := send_h2hIssue_slot hp hs hp1
        let l := s.pipe.send.h2hIssued
        cases hdr_b : s.pipe.send.life.draining with
        | true =>
          simp only [show s.pipe.send.life.beginOp = none by simp [Lifecycle.beginOp, hdr_b],
                     Option.some.injEq] at hsnd; subst hsnd
          simp only [hdr_b, ↓reduceIte] at hpost; subst hpost
          exact releaseSync_preserves_send hs rfl (fun _ => hd_f)
            (by simpa using hs.loopBeforeStart) (by simpa using hs.loopActive)
            (by simpa using hs.pendingLoop) (by simpa using hs.wokenLoop) (by simpa using hs.zeroSlots)
            (by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, hp1, hwh]; omega)
            (fun u hu => by simp [sendOwnsSlot, sendLoopOwnsSlot, sendPipeOwnsSlot, hu, hp1])
        | false =>
          simp only [show s.pipe.send.life.beginOp =
              some { s.pipe.send.life with inFlight := s.pipe.send.life.inFlight + 1 } by
            simp [Lifecycle.beginOp, hd_f, hdr_b], Option.some.injEq] at hsnd; subst hsnd
          simp only [hdr_b, Bool.false_eq_true, ↓reduceIte, withSlot_eq hlk hslt] at hpost; subst hpost
          have hret_f : s.pipe.h2hRetiredL[l]? = some false := by
            have hlen : l < s.pipe.h2hRetiredL.length := by have := hp.len_h2hRetiredL; have := hp.n_send; omega
            have heq := List.getElem?_eq_getElem hlen
            cases hb : s.pipe.h2hRetiredL[l]
            · rw [heq, hb]
            · rw [hb] at heq; have := hp.h2hRetired_lt l heq; omega
          refine syncMgr_preserves_send hs id
            hs.loopBeforeStart hs.loopActive hs.pendingLoop hs.wokenLoop hs.zeroSlots (fun u => ?_)
            (by simp [StagingLayerManager.sync, hd_f])
          unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot; simp only [hp1, Nat.sub_self]
          by_cases hu : u = s.pipe.send.h2hIssued
          · subst hu; exact ⟨fun _ => Or.inr ⟨hlt_d2h, Or.inr (Or.inl ⟨rfl, trivial⟩)⟩,
              fun _ => Or.inr ⟨hlt_d2h, Or.inl ⟨by omega, hret_f⟩⟩⟩
          · simp [hu, show u < s.pipe.send.h2hIssued + 1 ↔ u < s.pipe.send.h2hIssued by omega]
      | d2hReady => simp [pipeStep, Pipeline.step, Pipeline.sendStep] at hst
      | h2hDone _ => simp [pipeStep, Pipeline.step, Pipeline.sendStep] at hst
      | _ =>
        simp only [pipeStep, Pipeline.step, Option.map_eq_some_iff] at hst
        rcases hst with ⟨p, hp0, rfl⟩
        exact passive_send_preserves hs hp0 nofun nofun nofun nofun nofun
    | d2hReady l =>
      simp only [pipeStep, d2hReady, Pipeline.step, Pipeline.d2hReady, Option.map_eq_some_iff] at hst
      rcases hst with ⟨p, hp0, hpost⟩; split at hp0 <;> try cases hp0
      rename_i hg; rcases hg with ⟨hlt, hrd_f⟩
      rcases Option.map_eq_some_iff.mp hp0 with ⟨snd, hsnd, rfl⟩
      obtain ⟨_, rfl⟩ := Send.d2hReady_spec hsnd
      obtain ⟨hd_f, slot, hlk, hslt⟩ := send_d2hReady_slot hp hs hlt hrd_f
      rw [withSlot_eq hlk hslt] at hpost; subst hpost
      have hrec_f : s.pipe.reclaimed = false := by
        cases hr : s.pipe.reclaimed with
        | false => rfl
        | true => have := hp.reclaimed_done hr; rw [hd_f] at this; cases this
      rw [show s.pipe.prefillHbm.getD l .junk = .kv l from by
        rw [hp.phbm_good hrec_f]; exact Pipeline.good_getD (by have := hp.n_send; omega)]
      refine ⟨hs.pool.setSlot _ (fun _ _ => Iff.rfl),
        hs.loopBeforeStart, hs.loopActive, hs.pendingLoop, hs.wokenLoop, hs.zeroSlots,
        fun _ u slot' hrd' hlk' => ?_⟩
      rcases mem_of_set_true hrd' with rfl | hold
      · rw [hlk] at hlk'; cases Option.some.inj hlk'; exact List.getElem?_set_self hslt
      · have hne_u : u ≠ l := fun heq => by subst heq; rw [hrd_f] at hold; cases hold
        rw [List.getElem?_set_ne (Ne.symm fun heq : slot' = slot =>
          hne_u (StagingLayerManager.lookup_injective hs.mgrInv hlk' (heq ▸ hlk)))]
        exact hs.slotsGood hd_f u slot' hold hlk'
    | h2hDone l ok =>
      simp only [pipeStep, h2hDone, Pipeline.step, Pipeline.h2hDone, Option.map_eq_some_iff] at hst
      rcases hst with ⟨p, hp0, hpost⟩; split at hp0 <;> try cases hp0
      rename_i hg; rcases hg with ⟨hlt_h2h, hret_f⟩
      obtain ⟨hd_f, slot, hlk, hslt, _⟩ := send_h2hDone_slot hp hs hlt_h2h hret_f
      rcases Option.map_eq_some_iff.mp hp0 with ⟨snd, hsnd, hp_eq⟩
      obtain ⟨hf1, hf2, hf3, hf4, hf5, hf6⟩ := Send.step_lease_counters hsnd nofun nofun nofun nofun nofun
      have ⟨hsp, hsm, hss, hsl, hps, hpn, hpd, hpr⟩ :
          s'.pipe = p ∧ s'.sendMgr = releaseSync s.pipe.send.life p.send.life l s.sendMgr ∧
          s'.prefillSlots = s.prefillSlots ∧ s'.sendLoop = s.sendLoop ∧
          p.send = snd ∧ p.numLayers = s.pipe.numLayers ∧
          p.d2hReadyL = s.pipe.d2hReadyL ∧ p.h2hRetiredL = s.pipe.h2hRetiredL.set l true := by
        cases ok <;> simp only [↓reduceIte, Bool.false_eq_true, withSlot_eq hlk hslt] at hp_eq hpost <;>
          subst hp_eq hpost <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
      have hwb := hc.woken_bounds
      have hlen_ret : l < s.pipe.h2hRetiredL.length := by have := hp.len_h2hRetiredL; have := hp.n_send; omega
      refine releaseSync_preserves_send hs (by rw [hsp, hsm]) (fun _ => hd_f)
        (by rw [hsp, hps, hf1, hsl]; exact hs.loopBeforeStart)
        (by rw [hsl, hsp, hps, hf1, hf2, hf3, hpn]; exact hs.loopActive)
        (by rw [hsp, hps, hf2, hsl]; exact hs.pendingLoop)
        (by rw [hsp, hps, hf3, hf6, hsl]; exact hs.wokenLoop)
        (by rw [hsl, hsp, hps, hf2, hf3]; exact hs.zeroSlots) ?_
        (fun u hu => by
          unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot
          rw [hsp, hps, hf2, hf3, hf4, hf5, hf6, hsl, hpr, List.getElem?_set_ne (Ne.symm hu)])
        hss (by rw [hsp, hpd])
      unfold sendOwnsSlot sendLoopOwnsSlot sendPipeOwnsSlot
      rw [hsp, hps, hf2, hf3, hf4, hf5, hf6, hsl, hpr, List.getElem?_set_self hlen_ret]
      rintro (⟨_, _⟩ | ⟨_, ⟨_, hf⟩ | ⟨_, _⟩ | _⟩) <;> first | omega | cases hf
    | reseatPrefillStaging =>
      simp only [pipeStep, Pipeline.step, Pipeline.reseatPrefillStaging, Option.map_eq_some_iff] at hst
      rcases hst with ⟨_, hp0, rfl⟩; split at hp0 <;> cases hp0
      have hd_t : s.pipe.send.life.done = true := by
        have := hp.send.life.staging; cases hd : s.pipe.send.life.done <;> simp_all
      exact ⟨hs.pool.reseat Cell.junk hd_t, hs.loopBeforeStart, hs.loopActive,
        hs.pendingLoop, hs.wokenLoop, hs.zeroSlots, fun hnd => by simp [hd_t] at hnd⟩
    | _ =>
      simp only [pipeStep, recvPushBegin, recvPushEnd, land, h2dBegin, h2dIssue, h2dReady,
                 withSlot, withSlotOpt, syncPipe] at hst
      repeat' first
        | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hp0, hpost⟩
           obtain ⟨hN, hSnd, hRdyL, hRetL⟩ := pipe_send_unchanged hp0 nofun nofun nofun
           (repeat' split at hpost) <;> subst hpost <;>
           exact hs.frame hN hSnd hRdyL hRetL (by first | rfl | simp [syncPipe, hSnd]))
        | (rw [Option.some.injEq] at hst; subst hst; exact hs.frame)
        | split at hst | cases hst | contradiction
  | _ =>
    simp only [step, recvAcquireBegin, recvAcquireWait, recvAcquireFinish, h2dDone] at hst
    repeat' first
      | (rw [Option.map_eq_some_iff] at hst; rcases hst with ⟨_, hst, hpost⟩
         (repeat' split at hpost) <;> subst hpost)
      | (rw [Option.bind_eq_some_iff] at hst; rcases hst with ⟨_, hst₁, hst⟩
         obtain ⟨hN₁, hS₁, hRd₁, hRt₁⟩ := pipe_send_unchanged hst₁ nofun nofun nofun
         obtain ⟨hN₂, hS₂, hRd₂, hRt₂⟩ := pipe_send_unchanged hst nofun nofun nofun
         exact hs.frame (hN₂.trans hN₁) (hS₂.trans hS₁) (hRd₂.trans hRd₁) (hRt₂.trans hRt₁)
           (by first | rfl | simp [syncPipe, hS₂.trans hS₁]))
      | (obtain ⟨hN, hSnd, hRdyL, hRetL⟩ := pipe_send_unchanged hst nofun nofun nofun
         exact hs.frame hN hSnd hRdyL hRetL (by first | rfl | simp [syncPipe, hSnd]))
      | (rw [Option.some.injEq] at hst; subst hst; exact hs.frame)
      | split at hst | cases hst

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

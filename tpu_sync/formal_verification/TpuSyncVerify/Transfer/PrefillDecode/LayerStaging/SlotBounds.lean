import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Progress

/-! # `LayerStaging`: necessary and sufficient host-layer slot bounds

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. Proves:
1. `no_slots_cannot_publish_ok`: for `0 < n`, `ks = 0 ∨ kr = 0` prevents
   `recv.published = some true`.
2. `exists_complete_trace`: whenever `n = 0 ∨ (0 < ks ∧ 0 < kr)`, `completeTrace n`
   cycles host layer buffers across all `n` layers and publishes
   `decodeHbmB = Pipeline.good n` with all host staging slots returned. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager)

/-! ## 1. Slot-count preservation and zero-slot impossibility -/

theorem step_numSlots {s s' : BoundedPipeline} {e : Ev} (hs : step s e = some s') :
    s'.sendMgr.numSlots = s.sendMgr.numSlots ∧ s'.recvMgr.numSlots = s.recvMgr.numSlots := by
  cases e with
  | sendAcquireDone t =>
    simp only [step, sendAcquireDone] at hs; split at hs <;> try cases hs
    split at hs
    · cases hs
    · rename_i _ _ hacq; cases hs; exact ⟨s.sendMgr.acquire_leased_numSlots hacq, rfl⟩
    all_goals rcases Option.map_eq_some_iff.mp hs with ⟨_, _, rfl⟩; simp [syncPipe]
  | recvAcquireWait l t =>
    simp only [step, recvAcquireWait] at hs; split at hs <;> try cases hs
    split at hs
    · cases hs
    · rename_i _ _ hacq; cases hs; exact ⟨rfl, s.recvMgr.acquire_leased_numSlots hacq⟩
    all_goals rcases Option.map_eq_some_iff.mp hs with ⟨_, _, rfl⟩; simp [syncPipe]
  | _ =>
    simp only [step, pipeStep, sendAcquireBegin, recvAcquireBegin, recvAcquireFinish, h2dDone,
      sendD2hBegin, sendD2hIssue, sendWake, sendH2hIssue, d2hReady, h2hDone,
      recvPushBegin, recvPushEnd, land, h2dBegin, h2dIssue, h2dReady,
      withSlot, withSlotOpt] at hs
    repeat' first
      | (rw [Option.map_eq_some_iff] at hs; rcases hs with ⟨_, _, hpost⟩;
         (repeat' split at hpost) <;> subst hpost <;>
         exact ⟨by simp only [syncPipe, StagingLayerManager.sync_numSlots, StagingLayerManager.releaseSync,
                              StagingLayerManager.release_numSlots],
                by simp only [syncPipe, StagingLayerManager.sync_numSlots, StagingLayerManager.releaseSync,
                              StagingLayerManager.release_numSlots]⟩)
      | (rw [Option.some.injEq] at hs; subst hs;
         exact ⟨rfl, by simp only [StagingLayerManager.release_numSlots]⟩)
      | split at hs | cases hs

theorem reachable_numSlots {n ks kr : Nat} {s : BoundedPipeline}
    (hr : (sys n ks kr).Reachable s) :
    s.sendMgr.numSlots = ks ∧ s.recvMgr.numSlots = kr :=
  (sys n ks kr).reachable_induction
    (P := fun s => s.sendMgr.numSlots = ks ∧ s.recvMgr.numSlots = kr) ⟨rfl, rfl⟩
    (fun _ _ _ ⟨hs, hr⟩ hst => ⟨(step_numSlots hst).1.trans hs, (step_numSlots hst).2.trans hr⟩) hr

/-- **Necessity of ≥ 1 slot per side**: for any `n > 0`, if either
`numSendSlots = 0` or `numRecvSlots = 0`, no reachable state of `sys n ks kr`
can ever publish `recv.published = some true`. -/
theorem no_slots_cannot_publish_ok {n ks kr : Nat} {s : BoundedPipeline}
    (hn : 0 < n) (hzero : ks = 0 ∨ kr = 0) (hr : (sys n ks kr).Reachable s) :
    s.pipe.recv.published ≠ some true := fun hpub => by
  have hinv := reachable_inv hr
  have hn_eq : s.pipe.numLayers = n :=
    (Pipeline.sys n).reachable_induction (P := fun p => p.numLayers = n) rfl
      (fun _ _ _ h0 h1 => (Pipeline.step_numLayers h1).trans h0) (reachable_pipe hr)
  obtain ⟨hs0, hr0⟩ := reachable_numSlots hr
  obtain ⟨hd2h_pos, hiss_pos⟩ :=
    Pipeline.inv_published_ok_issued_pos hinv.pipeInv (hn_eq ▸ hn) hpub
  rcases hzero with rfl | rfl
  · have := (hinv.sendSide.zeroSlots hs0).2.2; omega
  · have := (hinv.recvSide.zeroSlots hr0).2; omega

/-! ## 2. Sufficiency of 1 slot per side (`exists_complete_trace`) -/

def sendLayerEvs (l : Nat) : List Ev :=
  [.sendAcquireBegin, .sendAcquireDone false, .sendD2hBegin, .sendD2hIssue true,
   .d2hReady l, .sendD2hEnd, .sendWake true, .sendH2hIssue, .sendNext, .h2hDone l true]

def recvLayerEvs (l : Nat) : List Ev :=
  [.recvPushBegin, .recvAcquireBegin l, .recvAcquireWait l false, .recvAcquireFinish l,
   .land l, .h2dBegin l, .h2dIssue l true, .recvNetAccount, .recvPushEnd,
   .h2dReady l, .h2dDone l true]

def singleLayerEvs (l : Nat) : List Ev := sendLayerEvs l ++ recvLayerEvs l

def completeTrace (n : Nat) : List Ev :=
  if n == 0 then
    [.beginPull, .sendStart, .recvPullReply true, .recvPollReady, .sendPublish, .recvPublish]
  else
    [.beginPull, .sendStart, .recvPullReply true] ++
      ((List.range n).map singleLayerEvs).flatten ++ [.sendPublish, .recvPublish]

private structure PrefixDone (n ks kr m : Nat) (s : BoundedPipeline) : Prop where
  inv : Inv s
  n_eq : s.pipe.numLayers = n
  sendSlots : s.sendMgr.numSlots = ks
  recvSlots : s.recvMgr.numSlots = kr
  sendState :
    if m < n then
      s.pipe.send =
        { numLayers := n, life := { inFlight := 1, draining := false, done := false, statusOk := true, hasStaging := true },
          pullStarted := true, started := true, d2hPending := false, d2hIssued := m, d2hReady := m, d2hRetired := m,
          queued := m + 1, woken := m, pooled := 0, h2hIssued := m, chaining := 0, h2hRetired := m, h2hOk := m,
          published := none, underflow := false } ∧
      s.sendLoop = .nextIter ∧ s.sendMgr.leases = [] ∧ s.sendMgr.shutdown = false
    else
      s.pipe.send.life.done = true ∧ s.pipe.send.life.statusOk = true ∧
      s.pipe.send.life.hasStaging = false ∧ s.pipe.send.published = none
  recvState :
    if m < n then
      s.pipe.recv =
        { numLayers := n, life := { inFlight := 0, draining := false, done := false, statusOk := true, hasStaging := true },
          issued := m, ready := m, completed := m, layersAccounted := m, published := none,
          pushes := 0, pullPending := false, pending := 0, retired := m } ∧
      s.activeStreams = 0 ∧ s.recvMgr.leases = [] ∧ s.recvMgr.shutdown = false
    else
      s.pipe.recv.life.done = true ∧ s.pipe.recv.life.statusOk = true ∧
      s.pipe.recv.life.hasStaging = false ∧ s.pipe.recv.published = none
  checklists : ∀ i : Nat, i < n →
    s.pipe.d2hReadyL[i]? = some (decide (i < m)) ∧ s.pipe.h2hRetiredL[i]? = some (decide (i < m)) ∧
    s.pipe.landedL[i]? = some (decide (i < m)) ∧ s.pipe.claimedL[i]? = some (decide (i < m)) ∧
    s.pipe.h2dPendingL[i]? = some false ∧ s.pipe.h2dIssuedL[i]? = some (decide (i < m)) ∧
    s.pipe.h2dReadyL[i]? = some (decide (i < m)) ∧
    s.recvPhaseL[i]? = some (if i < m then .h2dRetired else .idle)

private theorem lease_cycle_of_nil {m : StagingLayerManager} {slots : List Cell}
    (hinv : m.Inv) (hlen : slots.length = m.numSlots)
    (hl : m.leases = []) (hsh : m.shutdown = false) (hpos : 0 < m.numSlots) (l : Nat) :
    ∃ (slot : Nat) (rest : List Nat),
      m.acquire l false false = some (.leased slot { m with free := rest, leases := [(l, slot)] }) ∧
      ({ m with free := rest, leases := [(l, slot)] } : StagingLayerManager).lookup l = some slot ∧
      slot < slots.length ∧
      (({ m with free := rest, leases := [(l, slot)] } : StagingLayerManager).release l).2 =
        { m with free := rest ++ [slot], leases := [] } := by
  have hc := hinv.conservation; rw [hl] at hc
  cases hf : m.free with
  | nil => simp [hf] at hc; omega
  | cons slot rest =>
    have hacq : m.acquire l false false =
        some (.leased slot { m with free := rest, leases := [(l, slot)] }) := by
      simp [StagingLayerManager.acquire, StagingLayerManager.ready, StagingLayerManager.lookup, hl, hsh, hf]
    have hlk : ({ m with free := rest, leases := [(l, slot)] } : StagingLayerManager).lookup l = some slot := by
      simp [StagingLayerManager.lookup]
    exact ⟨slot, rest, hacq, hlk, hlen ▸ StagingLayerManager.lookup_lt (m.acquire_inv hinv hacq) hlk,
      by simp [StagingLayerManager.release, StagingLayerManager.lookup]⟩

private theorem set_self_of_some {α : Type} {xs : List α} {l : Nat} {a : α}
    (h : xs[l]? = some a) : ∀ v, (xs.set l v)[l]? = some v :=
  fun _ => List.getElem?_set_self (lt_length_of_getElem?_eq h)

private theorem prefixDone_zero {n ks kr : Nat} (hn : 0 < n) :
    ∃ s₀, (sys n ks kr).runFrom (init n ks kr)
        [.beginPull, .sendStart, .recvPullReply true] = some s₀ ∧ PrefixDone n ks kr 0 s₀ := by
  have hne0 : n ≠ 0 := by omega
  have heval : (sys n ks kr).runFrom (init n ks kr) [.beginPull, .sendStart, .recvPullReply true] =
      (sys n ks kr).runFrom (init n ks kr) [.beginPull, .sendStart, .recvPullReply true] := rfl
  conv at heval =>
    rhs; simp [System.runFrom, sys, init, Pipeline.init, Send.init, Recv.initLoad,
      step, pipeStep, syncPipe, Pipeline.step, Pipeline.sendStep, Pipeline.recvStep,
      Send.step, Send.beginPull, Send.start, Send.trySendNext, Recv.step, Recv.pullReply,
      Lifecycle.beginOp, Lifecycle.endOpLocked, Lifecycle.settleLocked,
      StagingLayerManager.sync, StagingLayerManager.init, hn, hne0]
  exact ⟨_, heval, ⟨runFrom_inv (inv_init n ks kr) heval, rfl, rfl, rfl,
    by simp [hn], by simp [hn], fun i hi => by simp [hi]⟩⟩

local macro "simp_send_layer" "[" extra:Lean.Parser.Tactic.simpLemma,* "]" : conv =>
  `(conv| simp [sendLayerEvs, System.runFrom, sys, step, pipeStep, syncPipe, withSlot,
      sendAcquireBegin, sendAcquireDone, sendD2hBegin, sendD2hIssue, d2hReady, sendWake, sendH2hIssue, h2hDone,
      Pipeline.step, Pipeline.sendStep, Pipeline.d2hReady, Pipeline.h2hDone,
      Send.step, Send.d2hBegin, Send.d2hIssue, Send.markD2hReady, Send.d2hEnd, Send.wake, Send.h2hIssue,
      Send.sendNext, Send.trySendNext, Send.h2hDone, Send.finish, Send.endOp,
      Lifecycle.beginOp, Lifecycle.finishOnceLocked, Lifecycle.finishLocked, Lifecycle.endOpLocked, Lifecycle.settleLocked,
      StagingLayerManager.sync, StagingLayerManager.releaseSync, StagingLayerManager.doShutdown,
      StagingLayerManager.releaseAll, StagingLayerManager.release, StagingLayerManager.lookup, $extra,*])

local macro "simp_recv_layer" "[" extra:Lean.Parser.Tactic.simpLemma,* "]" : conv =>
  `(conv| simp [recvLayerEvs, System.runFrom, sys, step, pipeStep, syncPipe, withSlot, withSlotOpt,
      recvAcquireBegin, recvAcquireWait, recvAcquireFinish, recvPushBegin, land, h2dBegin, recvPushEnd, h2dIssue, h2dReady, h2dDone,
      Pipeline.step, Pipeline.recvStep, Pipeline.land, Pipeline.h2dBegin, Pipeline.h2dIssue, Pipeline.h2dReady,
      Recv.step, Recv.pushBegin, Recv.pushEnd, Recv.h2dBegin, Recv.h2dIssue, Recv.netAccount, Recv.networkCompleted,
      Recv.h2dReady, Recv.h2dDone, Lifecycle.beginOp, Lifecycle.finishOnceLocked, Lifecycle.finishLocked,
      Lifecycle.endOpLocked, Lifecycle.settleLocked, StagingLayerManager.sync, StagingLayerManager.releaseSync,
      StagingLayerManager.doShutdown, StagingLayerManager.releaseAll, StagingLayerManager.release, StagingLayerManager.lookup, $extra,*])

set_option hygiene false in
local macro "eval_single_layer" n_tm:term "," "[" extra:Lean.Parser.Tactic.simpLemma,* "]" : tactic =>
  `(tactic| (
    have hevalS : (sys $n_tm ks kr).runFrom s (sendLayerEvs l) = (sys $n_tm ks kr).runFrom s (sendLayerEvs l) := rfl
    conv at hevalS => rhs; simp_send_layer [hsend, hloop, hp.n_eq, hacqS, hlkS₁, hsltS, hrelS₁, hphbm_l,
      hchk_d2h, hchk_h2h, hset_d2hL, hset_pstag, hset_pslots, $extra,*]
    rcases (⟨_, hevalS, rfl⟩ : ∃ m, (sys $n_tm ks kr).runFrom s (sendLayerEvs l) = some m ∧ m = _) with ⟨s_mid, hevalS_k, hmid⟩
    clear hevalS
    have hevalR : (sys $n_tm ks kr).runFrom s_mid (recvLayerEvs l) = (sys $n_tm ks kr).runFrom s_mid (recvLayerEvs l) := rfl
    conv at hevalR => rhs; rw [hmid]; simp_recv_layer [hrecv, hact0, hacqR, hlkR₁, hsltR, hrelR₁,
      hchk_land, hchk_clm, hchk_pend, hchk_iss, hchk_rdy, hchk_ph,
      hset_landL, hset_pendL, hset_issL, hset_rdyL, hset_phL, hset_wireB, hset_wireP, hset_dstag, hset_dslots, $extra,*]
    have heval := (sys $n_tm ks kr).runFrom_append (sendLayerEvs l) (recvLayerEvs l) hevalS_k hevalR
    exact ⟨_, heval, ⟨runFrom_inv hinv heval, rfl, hp.sendSlots, hp.recvSlots,
      by simp [$extra,*], by simp [$extra,*], fun i hi => by simpa [List.set_set] using hchk_next i hi⟩⟩))

private theorem prefixDone_step {n ks kr l : Nat} {s : BoundedPipeline}
    (hlt : l < n) (hks : 0 < ks) (hkr : 0 < kr) (hp : PrefixDone n ks kr l s) :
    ∃ s', (sys n ks kr).runFrom s (singleLayerEvs l) = some s' ∧ PrefixDone n ks kr (l + 1) s' := by
  have hinv := hp.inv
  have hpS := hp.sendState; simp only [hlt, ↓reduceIte] at hpS
  rcases hpS with ⟨hsend, hloop, hsleases, hsshut⟩
  have hpR := hp.recvState; simp only [hlt, ↓reduceIte] at hpR
  rcases hpR with ⟨hrecv, hact0, hrleases, hrshut⟩
  obtain ⟨hchk_d2h, hchk_h2h, hchk_land, hchk_clm, hchk_pend, hchk_iss, hchk_rdy, hchk_ph⟩ := hp.checklists l hlt
  simp only [Nat.lt_irrefl, decide_false, ↓reduceIte] at hchk_d2h hchk_h2h hchk_land hchk_clm hchk_iss hchk_rdy hchk_ph
  obtain ⟨slotS, restS, hacqS, hlkS₁, hsltS, hrelS₁⟩ :=
    lease_cycle_of_nil hinv.sendSide.mgrInv hinv.sendSide.lenSlots hsleases hsshut (hp.sendSlots ▸ hks) l
  obtain ⟨slotR, restR, hacqR, hlkR₁, hsltR, hrelR₁⟩ :=
    lease_cycle_of_nil hinv.recvSide.mgrInv hinv.recvSide.lenSlots hrleases hrshut (hp.recvSlots ▸ hkr) l
  have hrec_f : s.pipe.reclaimed = false := by
    cases hr : s.pipe.reclaimed
    · rfl
    · have := hinv.pipeInv.reclaimed_done hr; simp [hsend] at this
  have hphbm_l : s.pipe.prefillHbm[l]? = some (.kv l) := by
    have hl : l < s.pipe.prefillHbm.length := by
      rw [hinv.pipeInv.phbm_good hrec_f, Pipeline.good, List.length_map, List.length_range, hp.n_eq]; exact hlt
    have hd := Pipeline.good_getD hlt
    rw [← hp.n_eq, ← hinv.pipeInv.phbm_good hrec_f, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hl] at hd
    rw [List.getElem?_eq_getElem hl]; simpa using hd
  have hset_d2hL := set_self_of_some hchk_d2h; have hset_h2hL := set_self_of_some hchk_h2h
  have hset_landL := set_self_of_some hchk_land; have hset_clmL := set_self_of_some hchk_clm
  have hset_pendL := set_self_of_some hchk_pend; have hset_issL := set_self_of_some hchk_iss
  have hset_rdyL := set_self_of_some hchk_rdy; have hset_phL := set_self_of_some hchk_ph
  have hset_wireB : ∀ v, (s.wireB.set l v)[l]? = some v :=
    fun _ => List.getElem?_set_self (by rw [hinv.wireEq, hinv.pipeInv.len_wire, hp.n_eq]; exact hlt)
  have hset_wireP : ∀ v, (s.pipe.wire.set l v)[l]? = some v :=
    fun _ => List.getElem?_set_self (by rw [hinv.pipeInv.len_wire, hp.n_eq]; exact hlt)
  have hset_pstag : ∀ v, (s.pipe.prefillStaging.set l v)[l]? = some v :=
    fun _ => List.getElem?_set_self (by rw [hinv.pipeInv.len_pstaging, hp.n_eq]; exact hlt)
  have hset_dstag : ∀ v, (s.pipe.decodeStaging.set l v)[l]? = some v :=
    fun _ => List.getElem?_set_self (by rw [hinv.pipeInv.len_dstaging, hp.n_eq]; exact hlt)
  have hset_pslots : ∀ v, (s.prefillSlots.set slotS v)[slotS]? = some v := fun _ => List.getElem?_set_self hsltS
  have hset_dslots : ∀ v, (s.decodeSlots.set slotR v)[slotR]? = some v := fun _ => List.getElem?_set_self hsltR
  have hchk_next : ∀ i : Nat, i < n →
      (s.pipe.d2hReadyL.set l true)[i]? = some (decide (i < l + 1)) ∧
      (s.pipe.h2hRetiredL.set l true)[i]? = some (decide (i < l + 1)) ∧
      (s.pipe.landedL.set l true)[i]? = some (decide (i < l + 1)) ∧
      (s.pipe.claimedL.set l true)[i]? = some (decide (i < l + 1)) ∧
      (s.pipe.h2dPendingL.set l false)[i]? = some false ∧
      (s.pipe.h2dIssuedL.set l true)[i]? = some (decide (i < l + 1)) ∧
      (s.pipe.h2dReadyL.set l true)[i]? = some (decide (i < l + 1)) ∧
      (s.recvPhaseL.set l .h2dRetired)[i]? = some (if i < l + 1 then .h2dRetired else .idle) := by
    intro i hi; by_cases hil : i = l
    · subst hil; simp [hset_d2hL, hset_h2hL, hset_landL, hset_clmL, hset_pendL, hset_issL, hset_rdyL, hset_phL]
    · obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8⟩ := hp.checklists i hi
      simp [List.getElem?_set_ne (Ne.symm hil), show (i < l + 1) ↔ (i < l) by omega, c1, c2, c3, c4, c5, c6, c7, c8]
  by_cases hlt1 : l + 1 < n
  · have hneq1 : l + 1 ≠ n := by omega
    eval_single_layer n, [hlt, hlt1, hneq1, hsshut, hrshut]
  · have heq1 : l + 1 = n := by omega
    subst heq1
    eval_single_layer (l + 1), []

private theorem prefixDone_all {n ks kr : Nat} (hn : 0 < n) (hks : 0 < ks) (hkr : 0 < kr) :
    ∀ m : Nat, m ≤ n →
      ∃ s_m, (sys n ks kr).runFrom (init n ks kr)
          ([.beginPull, .sendStart, .recvPullReply true] ++ ((List.range m).map singleLayerEvs).flatten) = some s_m ∧
        PrefixDone n ks kr m s_m
  | 0, _ => by simpa using prefixDone_zero hn
  | m + 1, hle => by
    obtain ⟨s_m, hrun_m, hp_m⟩ := prefixDone_all hn hks hkr m (by omega)
    obtain ⟨s', hrun_step, hp'⟩ := prefixDone_step (by omega) hks hkr hp_m
    rw [List.range_succ, List.map_append, List.flatten_append, List.map_singleton,
        List.flatten_singleton, ← List.append_assoc]
    exact ⟨s', (sys n ks kr).runFrom_append _ _ hrun_m hrun_step, hp'⟩

/-- **Sufficiency of 1 slot per side (`exists_complete_trace`)**: whenever
`n = 0` or both `0 < ks` and `0 < kr`, `completeTrace n` executes from `init n ks kr`
to a settled state with `send.published = some true`, `recv.published = some true`,
`decodeHbmB = Pipeline.good n`, and all bounded host staging slots returned. -/
theorem exists_complete_trace (n ks kr : Nat) (hslots : n = 0 ∨ (0 < ks ∧ 0 < kr)) :
    ∃ s, (sys n ks kr).run (completeTrace n) = some s ∧
      s.pipe.send.published = some true ∧ s.pipe.recv.published = some true ∧
      s.decodeHbmB = Pipeline.good n ∧ s.sendMgr.numFree = ks ∧ s.recvMgr.numFree = kr := by
  by_cases hn0 : n = 0
  · subst hn0
    have heval : (sys 0 ks kr).run (completeTrace 0) = (sys 0 ks kr).run (completeTrace 0) := rfl
    conv at heval =>
      rhs; simp [System.run, System.runFrom, completeTrace, sys, init, Pipeline.init, Send.init, Recv.initLoad,
        step, pipeStep, syncPipe, Pipeline.step, Pipeline.sendStep, Pipeline.recvStep,
        Send.step, Send.beginPull, Send.start, Send.finish, Send.publish,
        Recv.step, Recv.pullReply, Recv.pollReady, Recv.isReadyToComplete,
        Recv.allH2dDone, Recv.networkCompleted, Recv.publish,
        Lifecycle.finishOnceLocked, Lifecycle.finishLocked, Lifecycle.endOpLocked, Lifecycle.settleLocked,
        StagingLayerManager.sync, StagingLayerManager.doShutdown, StagingLayerManager.releaseAll]
    exact ⟨_, heval, rfl, rfl, by simp [Pipeline.good],
      by simp [StagingLayerManager.init, StagingLayerManager.numFree],
      by simp [StagingLayerManager.init, StagingLayerManager.numFree]⟩
  · have hn_pos : 0 < n := by omega
    have ⟨hks, hkr⟩ : 0 < ks ∧ 0 < kr := hslots.elim (absurd · hn0) id
    obtain ⟨s_n, hrun_n, hp_n⟩ := prefixDone_all hn_pos hks hkr n (Nat.le_refl n)
    have hpS := hp_n.sendState; simp only [Nat.lt_irrefl, ↓reduceIte] at hpS
    rcases hpS with ⟨hdS, hokS, _, hpubS⟩
    have hpR := hp_n.recvState; simp only [Nat.lt_irrefl, ↓reduceIte] at hpR
    rcases hpR with ⟨hdR, hokR, _, hpubR⟩
    let s_pub : BoundedPipeline :=
      { s_n with pipe := { s_n.pipe with
          send := { s_n.pipe.send with published := some true },
          recv := { s_n.pipe.recv with published := some true } } }
    have hrun_pub : (sys n ks kr).runFrom s_n [.sendPublish, .recvPublish] = some s_pub := by
      simp [System.runFrom, sys, step, pipeStep, syncPipe, Pipeline.step, Pipeline.sendStep, Pipeline.recvStep,
            Send.step, Send.publish, Recv.step, Recv.publish, hdS, hokS, hpubS, hdR, hokR, hpubR, s_pub]
    have hn_ne : (n == 0) = false := by cases n <;> simp_all
    have hrun_full : (sys n ks kr).run (completeTrace n) = some s_pub := by
      unfold System.run; simp only [completeTrace, hn_ne, Bool.false_eq_true, ↓reduceIte]
      exact (sys n ks kr).runFrom_append _ _ hrun_n hrun_pub
    have hsafe := reachable_safe ((sys n ks kr).run_reachable hrun_full)
    exact ⟨s_pub, hrun_full, rfl, rfl,
      by simpa [s_pub, hp_n.n_eq] using hsafe.2.1 (rfl : s_pub.pipe.recv.published = some true),
      by simpa [s_pub, hp_n.sendSlots] using (hsafe.2.2.2.2.1.1 (by simpa [s_pub] using hdS)).2,
      by simpa [s_pub, hp_n.recvSlots] using (hsafe.2.2.2.2.1.2 (by simpa [s_pub] using hdR)).2⟩

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

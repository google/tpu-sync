import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Progress

/-! # `LayerStaging`: eventual settlement and full slot reclamation

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. Proves
`reachable_can_settle` and `reachable_unregistered_can_settle`: from every
reachable state of `sys n ks kr` or `sysUnregistered n ks kr` (for any
`n, ks, kr`), a finite trace settles both sessions, publishes both outcomes,
reclaims prefill HBM, and returns all producer and consumer host staging slots. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager)

/-! ## 1. Draining ranks and one-step progress -/

def sendLoopRank : SendLoopPhase → Nat
  | .leasedBeforeD2h | .acquiring => 2 | .nextIter => 1 | .exited => 0

def sendDrainRank (s : BoundedPipeline) : Nat :=
  3 * Send.drainRank s.pipe.send + sendLoopRank s.sendLoop

def recvDrainRank (s : BoundedPipeline) : Nat :=
  2 * Recv.drainRank s.pipe.recv + countWaiting s.recvPhaseL

def drainRank (s : BoundedPipeline) : Nat := sendDrainRank s + recvDrainRank s

private theorem mk_send_drain {s : BoundedPipeline} (e : Ev) {s' : BoundedPipeline}
    (hst : step s e = some s') (hrk : sendDrainRank s' < sendDrainRank s)
    (hdr : s'.pipe.send.life.draining = true := by
      simp_all [syncPipe, Send.finish, Lifecycle.finishOnceLocked])
    (hrecv : s'.pipe.recv = s.pipe.recv := by rfl)
    (hph : s'.recvPhaseL = s.recvPhaseL := by rfl) :
    ∃ e s', step s e = some s' ∧ s'.pipe.send.life.draining = true ∧
      s'.pipe.recv = s.pipe.recv ∧ s'.recvPhaseL = s.recvPhaseL ∧
      sendDrainRank s' < sendDrainRank s :=
  ⟨e, s', hst, hdr, hrecv, hph, hrk⟩

private theorem mk_recv_drain {s : BoundedPipeline} (e : Ev) {s' : BoundedPipeline}
    (hst : step s e = some s') (hrk : recvDrainRank s' < recvDrainRank s)
    (hdr : s'.pipe.recv.life.draining = true := by
      simp_all [syncPipe, Lifecycle.finishLocked_draining])
    (hsend : s'.pipe.send = s.pipe.send := by rfl)
    (hlp : s'.sendLoop = s.sendLoop := by rfl) :
    ∃ e s', step s e = some s' ∧ s'.pipe.recv.life.draining = true ∧
      s'.pipe.send = s.pipe.send ∧ s'.sendLoop = s.sendLoop ∧
      recvDrainRank s' < recvDrainRank s :=
  ⟨e, s', hst, hdr, hsend, hlp, hrk⟩

theorem drain_send_step {s : BoundedPipeline} (h : Inv s)
    (hdr : s.pipe.send.life.draining = true) (hnd : s.pipe.send.life.done = false) :
    ∃ e s', step s e = some s' ∧ s'.pipe.send.life.draining = true ∧
      s'.pipe.recv = s.pipe.recv ∧ s'.recvPhaseL = s.recvPhaseL ∧
      sendDrainRank s' < sendDrainRank s := by
  have hp := h.pipeInv
  have hif : 0 < s.pipe.send.life.inFlight := by
    cases h0 : s.pipe.send.life.inFlight
    · have := hp.send.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := hp.send.accounted; unfold Send.Accounted at hacc
  have hcnt := hp.send.counters; unfold Send.CountersOrdered at hcnt
  have hpc := hp.send.pull_claimed; unfold Send.PullClaimed at hpc
  have hsn := hp.n_send
  by_cases hdp : s.pipe.send.d2hPending = true
  · have hloop := h.sendSide.pendingLoop hdp
    exact mk_send_drain (.sendD2hIssue false)
      (by simp [step, pipeStep, sendD2hIssue, hdp, Pipeline.step, Pipeline.sendStep, Send.step, Send.d2hIssue]; rfl)
      (by simp only [sendDrainRank, sendLoopRank, hloop,
            Send.finish_of_draining false (s := { s.pipe.send with d2hPending := false }) hdr,
            Send.drainRank_endOp]; simp [Send.drainRank, hdp]; omega)
      (by simp [Send.finish_of_draining false (s := { s.pipe.send with d2hPending := false }) hdr, hdr])
  · have hdp_f : s.pipe.send.d2hPending = false := by cases b : s.pipe.send.d2hPending <;> simp_all
    by_cases hlp_l : s.sendLoop = .leasedBeforeD2h
    · exact mk_send_drain .sendD2hBegin
        (by simp [step, pipeStep, sendD2hBegin, hlp_l, hdr, Pipeline.step, Pipeline.sendStep, Send.step, Send.cancel]; rfl)
        (by simp [sendDrainRank, sendLoopRank, hlp_l, Send.finish_of_draining false hdr])
    · by_cases hlp_a : s.sendLoop = .acquiring
      · have hacq : s.sendMgr.acquire s.pipe.send.d2hIssued s.pipe.send.life.draining false = some .noSlots ∨
            s.sendMgr.acquire s.pipe.send.d2hIssued s.pipe.send.life.draining false = some .cancelled := by
          cases hns : s.sendMgr.numSlots with
          | zero => exact Or.inl (s.sendMgr.no_slots_acquire h.sendSide.mgrInv hns _ _ _)
          | succ k =>
            exact Or.inr (s.sendMgr.shutdown_cancels h.sendSide.mgrInv
              (h.sendSide.drainingShutdown hdr) (hns ▸ Nat.succ_pos k) _ _ _)
        rcases hacq with hacq | hacq <;>
          exact mk_send_drain (.sendAcquireDone false)
            (by simp [step, sendAcquireDone, hlp_a, hacq, Pipeline.step, Pipeline.sendStep, Send.step, Send.cancel]; rfl)
            (by simp [syncPipe, sendDrainRank, sendLoopRank, hlp_a, Send.finish_of_draining false hdr])
      · by_cases hlp_ni : s.sendLoop = .nextIter ∧ s.pipe.send.d2hIssued < s.pipe.numLayers
        · rcases hlp_ni with ⟨hlp_n, hdlt⟩
          have hst_t : s.pipe.send.started = true := by
            cases hs0 : s.pipe.send.started
            · have hd0 : s.pipe.send.d2hIssued = 0 := by
                cases k : s.pipe.send.d2hIssued
                · rfl
                · exact absurd (hpc.2 (Or.inr (by omega))) (by simp [hs0])
              have hq0 : s.pipe.send.queued = 0 := by
                cases k : s.pipe.send.queued
                · rfl
                · exact absurd (hp.send.queued_started (by omega)) (by simp [hs0])
              simp [hdp_f, hd0, hq0] at hacc hcnt; omega
            · rfl
          exact mk_send_drain .sendAcquireBegin
            (by simp [step, sendAcquireBegin, hst_t, hdp_f, hdlt, hlp_n, hdr,
                      Pipeline.step, Pipeline.sendStep, Send.step, Send.cancel]; rfl)
            (by simp [syncPipe, sendDrainRank, sendLoopRank, hlp_n, Send.finish_of_draining false hdr])
        · have hex_of_lt : s.pipe.send.d2hIssued < s.pipe.numLayers → s.sendLoop = .exited := by
            intro hdlt; cases hlp : s.sendLoop <;> simp_all
          by_cases hrd : s.pipe.send.d2hReady < s.pipe.send.d2hIssued
          · obtain ⟨l, hl, hf⟩ := hp.chk_d2hReady.exists_false (by omega) hrd
            obtain ⟨_, slot, hlk, hslt⟩ := send_d2hReady_slot hp h.sendSide hl hf
            exact mk_send_drain (.d2hReady l)
              (by simp [step, pipeStep, d2hReady, Pipeline.step, Pipeline.d2hReady, hl, hf,
                        Send.step, Send.markD2hReady, hrd, withSlot_eq hlk hslt]; rfl)
              (by simp [sendDrainRank, Send.drainRank]; omega)
          · by_cases hret : s.pipe.send.d2hRetired < s.pipe.send.d2hReady
            · exact mk_send_drain .sendD2hEnd
                (by simp [step, pipeStep, Pipeline.step, Pipeline.sendStep, Send.step, Send.d2hEnd, hret]; rfl)
                (by simp only [syncPipe, sendDrainRank, Send.drainRank_endOp]; simp [Send.drainRank]; omega)
            · by_cases hwk : s.pipe.send.woken < s.pipe.send.queued
              · by_cases hwi : s.pipe.send.woken < s.pipe.send.d2hIssued
                · have hwr : s.pipe.send.woken < s.pipe.send.d2hReady := by omega
                  have hwl := all_true_lt_of_countTrue hp.d2hReady_lt
                    (by rw [hp.cnt_d2hReady]; omega) s.pipe.send.woken hwi
                  exact mk_send_drain (.sendWake true)
                    (by simp [step, pipeStep, sendWake, hwl, Pipeline.step, Pipeline.sendStep,
                              Send.step, Send.wake, hwk, hwr, hdr]; rfl)
                    (by simp only [sendDrainRank, Send.drainRank_endOp]; simp [Send.drainRank]; omega)
                · have hex := hex_of_lt (by omega)
                  have hwr : ¬ s.pipe.send.woken < s.pipe.send.d2hReady := by omega
                  have hle : s.pipe.send.d2hIssued ≤ s.pipe.send.woken := by omega
                  exact mk_send_drain (.sendWake false)
                    (by simp [step, pipeStep, sendWake, hex, hwk, hdp_f, hle, Pipeline.step,
                              Pipeline.sendStep, Send.step, Send.wake, hwr, hdr]; rfl)
                    (by simp only [sendDrainRank, Send.drainRank_endOp, Send.finish,
                                   Lifecycle.finishOnceLocked, hdr]
                        simp [Send.drainRank, hex]; omega)
              · by_cases hpool : 0 < s.pipe.send.pooled
                · exact mk_send_drain .sendH2hIssue
                    (by simp [step, pipeStep, sendH2hIssue, Pipeline.step, Pipeline.sendStep, Send.step,
                              Send.h2hIssue, show s.pipe.send.pooled ≠ 0 by omega,
                              Lifecycle.beginOp_of_draining hdr, hdr]; rfl)
                    (by simp only [sendDrainRank, Send.drainRank_endOp]; simp [Send.drainRank]; omega)
                · by_cases hch : 0 < s.pipe.send.chaining
                  · exact mk_send_drain .sendNext
                      (by simp [step, pipeStep, Pipeline.step, Pipeline.sendStep, Send.step, Send.sendNext,
                                show s.pipe.send.chaining ≠ 0 by omega,
                                Send.trySendNext_of_draining (s := { s.pipe.send with chaining := s.pipe.send.chaining - 1 }) hdr]; rfl)
                      (by simp only [syncPipe, sendDrainRank, Send.drainRank_endOp]; simp [Send.drainRank]; omega)
                  · have hh2h : s.pipe.send.h2hRetired < s.pipe.send.h2hIssued := by simp [hdp_f] at hacc; omega
                    obtain ⟨l, hl, hf⟩ := hp.chk_h2hRetired.exists_false (by omega) hh2h
                    exact mk_send_drain (.h2hDone l false)
                      (by simp [step, pipeStep, h2hDone, Pipeline.step, Pipeline.h2hDone, hl, hf,
                                Send.step, Send.h2hDone, hh2h]; rfl)
                      (by simp only [sendDrainRank,
                            Send.finish_of_draining false (s := { s.pipe.send with h2hRetired := s.pipe.send.h2hRetired + 1 }) hdr,
                            Send.drainRank_endOp]; simp [Send.drainRank]; omega)
                      (by simp [Send.finish_of_draining false (s := { s.pipe.send with h2hRetired := s.pipe.send.h2hRetired + 1 }) hdr, hdr])

theorem drain_recv_step {s : BoundedPipeline} (h : Inv s)
    (hdr : s.pipe.recv.life.draining = true) (hnd : s.pipe.recv.life.done = false) :
    ∃ e s', step s e = some s' ∧ s'.pipe.recv.life.draining = true ∧
      s'.pipe.send = s.pipe.send ∧ s'.sendLoop = s.sendLoop ∧
      recvDrainRank s' < recvDrainRank s := by
  have hp := h.pipeInv
  have hif : 0 < s.pipe.recv.life.inFlight := by
    cases h0 : s.pipe.recv.life.inFlight
    · have := hp.recv.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := hp.recv.accounted; unfold Recv.Accounted at hacc
  have hr_le := hp.recv.retired_le
  have hy_le := hp.recv.ready_le
  by_cases hact : 0 < s.activeStreams
  · have hpush : 0 < s.pipe.recv.pushes := by have := h.recvSide.pushesEq; omega
    exact mk_recv_drain .recvPushEnd
      (by simp [step, pipeStep, recvPushEnd, hact, Pipeline.step, Pipeline.recvStep,
                Recv.step, Recv.pushEnd, show s.pipe.recv.pushes ≠ 0 by omega]; rfl)
      (by simp [syncPipe, recvDrainRank, Recv.drainRank]; omega)
  · by_cases hpush : 0 < s.pipe.recv.pushes
    · have hpush_ne : s.pipe.recv.pushes ≠ 0 := by omega
      obtain ⟨l, ph, hph, hacq⟩ :=
        exists_of_countPred_pos (p := isAcquiringPhase) (xs := s.recvPhaseL)
          (by have := h.recvSide.pushesEq; omega)
      cases ph with
      | leasedInAcquire =>
        have hcw := countPred_set_same (p := isWaitingPhase) hph (newV := .aborted) rfl
        exact mk_recv_drain (.recvAcquireFinish l)
          (by simp [step, recvAcquireFinish, hph, Pipeline.step, Pipeline.recvStep,
                    Recv.step, Recv.pushEnd, hpush_ne, hdr]; rfl)
          (by simp [recvDrainRank, Recv.drainRank, hcw]; omega)
      | waiting hadLease =>
        let restoreP : RecvLayerPhase := if hadLease then .leasedForStream else .idle
        have hcw_dec := countPred_set_false (p := isWaitingPhase) hph rfl
          (newV := restoreP) (by cases hadLease <;> rfl)
        dsimp only [restoreP] at hcw_dec
        cases hns : s.recvMgr.numSlots with
        | zero =>
          have hacq_out := s.recvMgr.no_slots_acquire h.recvSide.mgrInv hns l s.pipe.recv.life.draining false
          exact mk_recv_drain (.recvAcquireWait l false)
            (by simp [step, recvAcquireWait, hph, hacq_out, Pipeline.step, Pipeline.recvStep,
                      Recv.step, Recv.cancel, Recv.pushEnd, hpush_ne]; rfl)
            (by simp [syncPipe, recvDrainRank, Recv.drainRank]; omega)
        | succ k =>
          have hacq_out := s.recvMgr.shutdown_cancels h.recvSide.mgrInv
            (h.recvSide.drainingShutdown hdr) (hns ▸ Nat.succ_pos k) l s.pipe.recv.life.draining false
          exact mk_recv_drain (.recvAcquireWait l false)
            (by simp [step, recvAcquireWait, hph, hacq_out, Pipeline.step, Pipeline.recvStep,
                      Recv.step, Recv.pushEnd, hpush_ne]; rfl)
            (by simp [syncPipe, recvDrainRank, Recv.drainRank]; omega)
      | _ => simp [isAcquiringPhase] at hacq
    · by_cases hq : s.pipe.recv.pullPending = true
      · exact mk_recv_drain (.recvPullReply false)
          (by simp [step, pipeStep, Pipeline.step, Pipeline.recvStep, Recv.step, Recv.pullReply, hq]; rfl)
          (by simp [syncPipe, recvDrainRank, Recv.drainRank, hq])
      · by_cases hpend : 0 < s.pipe.recv.pending
        · obtain ⟨l, _, hpl⟩ := hp.chk_h2dPending.exists_true hpend
          have hph := (h.recvSide.pendingPhase l).mp hpl
          have hcw := countPred_set_same (p := isWaitingPhase) hph (newV := .aborted) rfl
          exact mk_recv_drain (.h2dIssue l false)
            (by simp [step, pipeStep, h2dIssue, hph, Pipeline.step, Pipeline.h2dIssue, hpl,
                      Recv.step, Recv.h2dIssue, show s.pipe.recv.pending ≠ 0 by omega, hdr]; rfl)
            (by simp [recvDrainRank, Recv.drainRank, hcw]; omega)
        · by_cases hrd : s.pipe.recv.ready < s.pipe.recv.issued
          · obtain ⟨l, _, hrf, hit⟩ := hp.chk_h2dReady.exists_diff hp.chk_h2dIssued hrd
            have hph := ((h.recvSide.issuedPhase l).mp hit).resolve_right fun hret_t => by
              rw [h.recvSide.retiredReady l hret_t] at hrf; cases hrf
            obtain ⟨_, slot, hlk, hslt, _⟩ := recv_h2dReady_slot hp h.recvSide hph hrf
            exact mk_recv_drain (.h2dReady l)
              (by simp [step, pipeStep, h2dReady, hph, hrf, Pipeline.step, Pipeline.h2dReady, hit,
                        Recv.step, Recv.h2dReady, hrd, withSlot_eq hlk hslt]; rfl)
              (by simp [recvDrainRank, Recv.drainRank]; omega)
          · have hret : s.pipe.recv.retired < s.pipe.recv.ready := by
              cases hqp : s.pipe.recv.pullPending <;> simp_all <;> omega
            obtain ⟨l, _, hret_f, hrt⟩ := exists_diff_of_countTrue_lt
              (l₁ := s.recvPhaseL.map isRetiredPhase) (l₂ := s.pipe.h2dReadyL)
              (by simp [h.recvSide.lenPhase, hp.len_h2dReadyL])
              (by change countRetired s.recvPhaseL < countTrue s.pipe.h2dReadyL
                  rw [h.recvSide.cntRetired, hp.cnt_h2dReady]; exact hret)
            have hph := ((h.recvSide.issuedPhase l).mp (hp.ready_issued l hrt)).resolve_right fun hret_t => by
              simp [List.getElem?_map, hret_t, isRetiredPhase] at hret_f
            have hcw := countPred_set_same (p := isWaitingPhase) hph (newV := .h2dRetired) rfl
            exact mk_recv_drain (.h2dDone l false)
              (by simp [step, h2dDone, hph, hrt, Pipeline.step, Pipeline.recvStep,
                        Recv.step, Recv.h2dDone, hret, hnd]; rfl)
              (by simp [recvDrainRank, Recv.drainRank, hcw]; omega)

/-! ## 2. Eventual settlement (`reachable_can_settle`) -/

def finalizeEvs (s : BoundedPipeline) : List Ev :=
  (if s.pipe.send.published.isNone then [.sendPublish] else []) ++
  (if s.pipe.recv.published.isNone then [.recvPublish] else []) ++
  (if s.pipe.reclaimed then [] else [.reclaim])

theorem draining_can_settle (n ks kr : Nat) {s : BoundedPipeline} (h : Inv s)
    (hdrS : s.pipe.send.life.draining = true) (hdrR : s.pipe.recv.life.draining = true) :
    ∃ evs s', (sys n ks kr).runFrom s evs = some s' ∧ Inv s' ∧
      s'.pipe.send.life.done = true ∧ s'.pipe.send.life.hasStaging = false ∧
      s'.pipe.recv.life.done = true ∧ s'.pipe.recv.life.hasStaging = false := by
  obtain ⟨evs, s', hrun, ⟨hinv', _, _⟩, hdS', hdR'⟩ :=
    (sys n ks kr).settles_of_rank
      (Inv := fun s => Inv s ∧ s.pipe.send.life.draining = true ∧ s.pipe.recv.life.draining = true)
      (Done := fun s => s.pipe.send.life.done = true ∧ s.pipe.recv.life.done = true)
      drainRank
      (fun s ⟨hi, hdrS0, hdrR0⟩ hnd => by
        cases hndS : s.pipe.send.life.done
        · obtain ⟨e, s₁, hs₁, hdrS₁, hrecv_eq, hph_eq, hlt⟩ := drain_send_step hi hdrS0 hndS
          refine ⟨e, s₁, hs₁, ⟨step_inv hi hs₁, hdrS₁, hrecv_eq ▸ hdrR0⟩, ?_⟩
          unfold drainRank recvDrainRank; rw [hrecv_eq, hph_eq]; omega
        · cases hndR : s.pipe.recv.life.done
          · obtain ⟨e, s₁, hs₁, hdrR₁, hsend_eq, hlp_eq, hlt⟩ := drain_recv_step hi hdrR0 hndR
            refine ⟨e, s₁, hs₁, ⟨step_inv hi hs₁, hsend_eq ▸ hdrS0, hdrR₁⟩, ?_⟩
            unfold drainRank sendDrainRank; rw [hsend_eq, hlp_eq]; omega
          · exact absurd ⟨hndS, hndR⟩ hnd)
      ⟨h, hdrS, hdrR⟩
  exact ⟨evs, s', hrun, hinv', hdS',
    by rw [hinv'.pipeInv.send.life.staging, hdS']; rfl,
    hdR', by rw [hinv'.pipeInv.recv.life.staging, hdR']; rfl⟩

theorem inv_can_settle (n ks kr : Nat) {s : BoundedPipeline} (hinv : Inv s) :
    ∃ evs s', (sys n ks kr).runFrom s evs = some s' ∧
      s'.pipe.send.life.done = true ∧ s'.pipe.send.life.hasStaging = false ∧
      s'.pipe.recv.life.done = true ∧ s'.pipe.recv.life.hasStaging = false ∧
      s'.pipe.send.published ≠ none ∧ s'.pipe.recv.published ≠ none ∧
      s'.pipe.reclaimed = true ∧
      s'.sendMgr.numFree = s'.sendMgr.numSlots ∧
      s'.recvMgr.numFree = s'.recvMgr.numSlots := by
  let s₁ := syncPipe s { s.pipe with send := s.pipe.send.finish false }
  have hcS : step s .sendCancel = some s₁ := rfl
  let s₂ := syncPipe s₁ { s₁.pipe with recv := { s₁.pipe.recv with life := s₁.pipe.recv.life.finishLocked false } }
  have hcR : step s₁ .recvCancel = some s₂ := rfl
  obtain ⟨evs_d, s_d, hrd, hinv_d, hdS, hstS, hdR, hstR⟩ :=
    draining_can_settle n ks kr (step_inv (step_inv hinv hcS) hcR)
      (hinv.pipeInv.send.life.finishOnceLocked_draining false) (Lifecycle.finishLocked_draining false _)
  have ⟨s', hrf, hdS', hstS', hdR', hstR', hpubS', hpubR', hrec'⟩ :
      ∃ s', (sys n ks kr).runFrom s_d (finalizeEvs s_d) = some s' ∧
        s'.pipe.send.life.done = true ∧ s'.pipe.send.life.hasStaging = false ∧
        s'.pipe.recv.life.done = true ∧ s'.pipe.recv.life.hasStaging = false ∧
        s'.pipe.send.published ≠ none ∧ s'.pipe.recv.published ≠ none ∧
        s'.pipe.reclaimed = true := by
    cases hpS : s_d.pipe.send.published <;> cases hpR : s_d.pipe.recv.published <;>
      cases hrec : s_d.pipe.reclaimed <;>
      simp [finalizeEvs, System.runFrom, sys, step, pipeStep, syncPipe,
        Pipeline.step, Pipeline.sendStep, Pipeline.recvStep, Pipeline.reclaim,
        Send.step, Send.publish, Recv.step, Recv.publish, hdS, hstS, hdR, hstR, hpS, hpR, hrec]
  have hsafe' := inv_safe (runFrom_inv hinv_d hrf)
  exact ⟨[.sendCancel, .recvCancel] ++ evs_d ++ finalizeEvs s_d, s',
    (sys n ks kr).runFrom_append _ (finalizeEvs s_d)
      ((sys n ks kr).runFrom_append _ evs_d (by simp [System.runFrom, sys, hcS, hcR]) hrd) hrf,
    hdS', hstS', hdR', hstR', hpubS', hpubR', hrec',
    (hsafe'.2.2.2.2.1.1 hdS').2, (hsafe'.2.2.2.2.1.2 hdR').2⟩

/-- **Eventual settlement and full slot reclamation (registered offer)**: from
any reachable state of `sys n ks kr`, a finite trace settles both sessions,
publishes both outcomes, reclaims prefill HBM, and returns all bounded host
staging slots (`numFree = numSlots`). -/
theorem reachable_can_settle {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sys n ks kr).Reachable s) :
    ∃ evs s', (sys n ks kr).runFrom s evs = some s' ∧
      s'.pipe.send.life.done = true ∧ s'.pipe.send.life.hasStaging = false ∧
      s'.pipe.recv.life.done = true ∧ s'.pipe.recv.life.hasStaging = false ∧
      s'.pipe.send.published ≠ none ∧ s'.pipe.recv.published ≠ none ∧
      s'.pipe.reclaimed = true ∧
      s'.sendMgr.numFree = s'.sendMgr.numSlots ∧
      s'.recvMgr.numFree = s'.recvMgr.numSlots :=
  inv_can_settle n ks kr (reachable_inv h)

/-- **Eventual settlement and full slot reclamation (unregistered offer)**: from
any reachable state of `sysUnregistered n ks kr`, a finite trace settles both
sessions, publishes both outcomes, reclaims prefill HBM, and returns all bounded
host staging slots (`numFree = numSlots`). -/
theorem reachable_unregistered_can_settle {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sysUnregistered n ks kr).Reachable s) :
    ∃ evs s', (sysUnregistered n ks kr).runFrom s evs = some s' ∧
      s'.pipe.send.life.done = true ∧ s'.pipe.send.life.hasStaging = false ∧
      s'.pipe.recv.life.done = true ∧ s'.pipe.recv.life.hasStaging = false ∧
      s'.pipe.send.published ≠ none ∧ s'.pipe.recv.published ≠ none ∧
      s'.pipe.reclaimed = true ∧
      s'.sendMgr.numFree = s'.sendMgr.numSlots ∧
      s'.recvMgr.numFree = s'.recvMgr.numSlots :=
  inv_can_settle n ks kr (reachable_unregistered_inv h)

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

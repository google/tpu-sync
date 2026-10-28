import TpuSyncVerify.Transfer.Reshard.Pipeline.Preservation

/-! # `Pipeline`: quietness, `attention_safe`, progress and eventual settlement

Part of `TpuSyncVerify.Transfer.Reshard.Pipeline`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ## Quietness and temporal post-publication safety (`attention_safe`) -/

@[simp] theorem launchEligible_published (r : Recv) :
    (Recv.launchEligible r).published = r.published := by
  unfold Recv.launchEligible; split <;> rfl

@[simp] theorem maybeUnregisterSettled_published (r : Recv) :
    (Recv.maybeUnregisterSettled r).published = r.published := by
  unfold Recv.maybeUnregisterSettled; split <;> rfl

theorem recv_step_published_eq {pol : PushEndPolicy} {r r' : Recv} {e : Recv.Ev}
    (hs : Recv.step pol r e = some r') (hpub : r.published ≠ none) :
    r'.published = r.published := by
  cases e with
  | pushBegin =>
    simp only [Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hs
    obtain ⟨_, _, rfl⟩ := hs; rfl
  | pushEnd ok =>
    simp only [Recv.step, Recv.pushEnd] at hs
    split at hs <;> simp at hs; subst hs; simp
  | onPoolReceived p =>
    simp only [Recv.step, Recv.onPoolReceived] at hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; simp
  | h2dIssue p ok =>
    simp only [Recv.step, Recv.h2dIssue] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · simp at hs; subst hs; rfl
    · split at hs <;> { simp at hs; subst hs; simp }
  | h2dReady p =>
    simp only [Recv.step, Recv.h2dReady] at hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; rfl
  | h2dDone p ok =>
    simp only [Recv.step, Recv.h2dDone] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · simp at hs; subst hs; simp
    · split at hs
      · simp at hs; subst hs; simp
      · split at hs
        · simp at hs; subst hs; simp
        · split at hs <;> { simp at hs; subst hs; simp }
  | pollReady =>
    simp only [Recv.step, Recv.pollReady] at hs
    split at hs <;> simp at hs; subst hs; rfl
  | cancel =>
    simp only [Recv.step, Recv.cancel, Option.some.injEq] at hs
    subst hs; rfl
  | publish =>
    simp only [Recv.step, Recv.publish] at hs
    split at hs <;> try cases hs
    rename_i hc; exact absurd hc.2 hpub

/-- Once the receiver is published (`done_recving` or `failed_recving`), no
pipeline step can modify `dstHbm` or `recv.published`. -/
theorem dstHbm_quiet {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hpub : s.recv.published ≠ none) (hs : step s e = some s') :
    s'.dstHbm = s.dstHbm ∧ s'.recv.published = s.recv.published ∧ s'.numPools = s.numPools := by
  cases e with
  | armRecv =>
    simp only [step, armRecv] at hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl, rfl⟩
  | send e =>
    cases e <;> simp only [step, sendStep, Option.map_eq_some_iff] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl, rfl⟩
    · obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl, rfl⟩
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; exact ⟨rfl, rfl, rfl⟩ }
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; exact ⟨rfl, rfl, rfl⟩ }
    · obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl, rfl⟩
    · obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl, rfl⟩
  | recv e =>
    cases e with
    | pushBegin =>
      simp only [step, recvStep] at hs
      split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, rfl⟩ := hs
      exact ⟨rfl, recv_step_published_eq hrcv hpub, rfl⟩
    | onPoolReceived p =>
      simp only [step, recvStep] at hs
      split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, rfl⟩ := hs
      exact ⟨rfl, recv_step_published_eq hrcv hpub, rfl⟩
    | h2dDone p ok =>
      simp only [step, recvStep] at hs
      split at hs <;> try cases hs
      rename_i pool hget
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, _⟩ := hs
      obtain ⟨pool₀, _, hget₀, hrd, _, _⟩ := h2dDone_pools_shape hrcv
      rw [hget] at hget₀; injection hget₀ with heq; subst heq
      have hnd := recv_not_done_of_ready h.recvInv hget hrd
      cases hb : s.recv.published with
      | none => exact absurd hb hpub
      | some b =>
        have ⟨hd, _⟩ := h.recvInv.published_done b hb
        simp [hnd] at hd
    | pushEnd ok | h2dIssue p ok | h2dReady p | pollReady | cancel | publish =>
      simp only [step, recvStep, Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, rfl⟩ := hs
      exact ⟨rfl, recv_step_published_eq hrcv hpub, rfl⟩
  | landStream p =>
    simp only [step, landStream] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl, rfl⟩
  | reclaim =>
    simp only [step, reclaim] at hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl, rfl⟩
  | reseatSrcStaging =>
    simp only [step, reseatSrcStaging] at hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl, rfl⟩
  | reseatDstStaging =>
    simp only [step, reseatDstStaging] at hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl, rfl⟩

theorem runFrom_dstHbm_quiet {ranks peerCounts : List Nat} :
    ∀ (evs : List Ev) {s s' : Pipeline},
      Inv s → s.recv.published ≠ none →
      (sys ranks peerCounts).runFrom s evs = some s' →
      s'.dstHbm = s.dstHbm ∧ s'.recv.published = s.recv.published ∧ s'.numPools = s.numPools
  | [], _, _, _, _, hrun => by
    simp only [System.runFrom, List.foldlM_nil] at hrun
    injection hrun with hrun
    subst hrun; exact ⟨rfl, rfl, rfl⟩
  | e :: evs, s, s', h, hpub, hrun => by
    simp only [System.runFrom, List.foldlM_cons, Option.bind_eq_bind,
      Option.bind_eq_some_iff] at hrun
    obtain ⟨s₁, hs₁, hrun₁⟩ := hrun
    have ⟨hd₁, hp₁, hn₁⟩ := dstHbm_quiet h hpub hs₁
    have hinv₁ := step_inv h hs₁
    have hpub₁ : s₁.recv.published ≠ none := by rw [hp₁]; exact hpub
    have ⟨hd₂, hp₂, hn₂⟩ := runFrom_dstHbm_quiet evs hinv₁ hpub₁ hrun₁
    exact ⟨hd₂.trans hd₁, hp₂.trans hp₁, hn₂.trans hn₁⟩

/-- Temporal post-publication safety (`attention_safe`): once `poll_stats()`
reports `done_recving` (`s.recv.published = some true`), destination HBM holds
`good numPools` and remains equal to `good numPools` across any subsequent
sequence of pipeline events (including sender/receiver staging reuse, source HBM
reclamation, and late drain or cancellation steps). -/
theorem attention_safe {ranks peerCounts : List Nat} {s s' : Pipeline} {evs : List Ev}
    (hvalid : validConfig ranks peerCounts = true)
    (hr : (sys ranks peerCounts).Reachable s)
    (hpub : s.recv.published = some true)
    (hrun : (sys ranks peerCounts).runFrom s evs = some s') :
    s'.dstHbm = good s.numPools ∧ s'.recv.published = some true := by
  have hinv := reachable_inv hvalid hr
  have hgood : s.dstHbm = good s.numPools := (inv_safe hinv).1 hpub
  have hne : s.recv.published ≠ none := by simp [hpub]
  have ⟨hd, hp, _⟩ := runFrom_dstHbm_quiet evs hinv hne hrun
  exact ⟨hd.trans hgood, hp.trans hpub⟩

/-- Before the sender is published (`s.send.published = none`), no pipeline step
can modify `srcHbm`. -/
theorem srcHbm_quiet {s s' : Pipeline} {e : Ev}
    (hpub : s.send.published = none) (hs : step s e = some s') :
    s'.srcHbm = s.srcHbm := by
  cases e with
  | armRecv =>
    simp only [step, armRecv] at hs; split at hs <;> simp at hs; subst hs; rfl
  | send e =>
    cases e <;> simp only [step, sendStep, Option.map_eq_some_iff] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, rfl⟩ := hs; rfl
  | recv e =>
    cases e <;> simp only [step, recvStep] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
  | landStream p =>
    simp only [step, landStream] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; rfl
  | reclaim =>
    simp only [step, reclaim, hpub, ne_eq, not_true_eq_false, false_and, ↓reduceIte] at hs
    cases hs
  | reseatSrcStaging =>
    simp only [step, reseatSrcStaging] at hs; split at hs <;> simp at hs; subst hs; rfl
  | reseatDstStaging =>
    simp only [step, reseatDstStaging] at hs; split at hs <;> simp at hs; subst hs; rfl

/-- Once the sender releases its `PoolStagingLease` (`hasStaging = false`), no
sender event can read or write `srcStaging` or `wire`. -/
theorem srcStaging_quiet {s s' : Pipeline} {e : Send.Ev} (h : Inv s)
    (hst : s.send.life.hasStaging = false) (hs : step s (.send e) = some s') :
    s'.srcStaging = s.srcStaging ∧ s'.wire = s.wire := by
  have hd : s.send.life.done = true := by
    rw [h.sendInv.life.staging] at hst; simpa using hst
  cases e with
  | d2hIssue p ok =>
    simp only [step, sendStep] at hs
    split at hs <;> try cases hs
    simp only [Option.map_eq_some_iff] at hs
    obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
  | d2hDone p ok =>
    simp only [step, sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, _⟩ := hs
    simp only [Send.step, Send.d2hDone] at hsnd
    split at hsnd <;> try cases hsnd
    rename_i pool hget
    split at hsnd <;> try cases hsnd
    rename_i hd2h
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hd2h
    have hnd := send_not_done_of_stage h.sendInv hget (by simp [hd2h, SendStage.opCount])
    simp [hnd] at hd
  | pushDone p ok =>
    simp only [step, sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, _⟩ := hs
    simp only [Send.step, Send.pushDone] at hsnd
    cases hget : s.send.pools[p]? with
    | none => simp [hget] at hsnd
    | some pool =>
      simp only [hget] at hsnd
      cases hpush : pool.stage with
      | pending | d2h | aborted => simp [hpush] at hsnd
      | pushing n =>
        cases n with
        | zero => simp [hpush] at hsnd
        | succ k =>
          have hnd := send_not_done_of_stage h.sendInv hget (by simp [hpush, SendStage.opCount])
          simp [hnd] at hd
  | execFinish | cancel | publish =>
    simp only [step, sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩

/-- Once the receiver releases its `PoolStagingLease` (`hasStaging = false`),
neither `landStream` nor any receiver event can modify `dstStaging` or `dstHbm`. -/
theorem dstStaging_quiet {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hst : s.recv.life.hasStaging = false)
    (hne : e ≠ .reseatDstStaging) (hs : step s e = some s') :
    s'.dstStaging = s.dstStaging ∧ s'.dstHbm = s.dstHbm := by
  have hd : s.recv.life.done = true := by
    rw [h.recvInv.life.staging] at hst; simpa using hst
  have hif := h.recvInv.life.done_idle hd
  have hacc := h.recvInv.accounted; unfold Recv.Accounted at hacc
  have hp0 : s.recv.pushes = 0 := by omega
  cases e with
  | armRecv =>
    simp only [step, armRecv] at hs; split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩
  | send e =>
    cases e <;> simp only [step, sendStep, Option.map_eq_some_iff] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; exact ⟨rfl, rfl⟩ }
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; exact ⟨rfl, rfl⟩ }
    · obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
  | recv e =>
    cases e with
    | pushBegin =>
      simp only [step, recvStep] at hs
      split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    | onPoolReceived p =>
      simp only [step, recvStep] at hs
      split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    | h2dDone p ok =>
      simp only [step, recvStep] at hs
      split at hs <;> try cases hs
      rename_i pool hget
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, _⟩ := hs
      obtain ⟨pool₀, _, hget₀, hrd, _, _⟩ := h2dDone_pools_shape hrcv
      rw [hget] at hget₀; injection hget₀ with heq; subst heq
      have hnd := recv_not_done_of_ready h.recvInv hget hrd
      simp [hnd] at hd
    | pushEnd ok | h2dIssue p ok | h2dReady p | pollReady | cancel | publish =>
      simp only [step, recvStep, Option.map_eq_some_iff] at hs
      obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
  | landStream p =>
    simp only [step, landStream, hp0, Nat.lt_irrefl, ↓reduceIte] at hs; cases hs
  | reclaim =>
    simp only [step, reclaim] at hs; split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩
  | reseatSrcStaging =>
    simp only [step, reseatSrcStaging] at hs; split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩
  | reseatDstStaging => exact absurd rfl hne

/-! ## Progress and eventual settlement -/

def drainRank (s : Pipeline) : Nat :=
  s.send.life.inFlight + Recv.drainRank s.recv

theorem send_drain_step_lift {s : Pipeline} (h : Inv s)
    (hdr : s.send.life.draining = true) (hnd : s.send.life.done = false) :
    ∃ e s', step s (.send e) = some s' ∧
      s'.send.life.draining = true ∧ s'.recv = s.recv ∧
      s'.send.life.inFlight < s.send.life.inFlight := by
  have hif : 0 < s.send.life.inFlight := by
    cases h0 : s.send.life.inFlight
    · have := h.sendInv.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := h.sendInv.accounted; unfold Send.Accounted at hacc
  cases hex : s.send.execActive
  · simp only [hex, Bool.false_eq_true, ↓reduceIte, Nat.zero_add] at hacc
    obtain ⟨p, _, pool, hget, hop⟩ := Send.exists_op_of_poolOps_pos (ps := s.send.pools) (by omega)
    cases hst : pool.stage with
    | pending | aborted => simp [hst, SendStage.opCount] at hop
    | d2h =>
      let snd := Send.endOp { s.send with pools := s.send.pools.set p { pool with stage := .aborted } }
      have hs₁ : Send.step .shipping s.send (.d2hDone p true) = some snd := by
        simp [Send.step, Send.d2hDone, hget, hst, hdr, snd]
      let s' : Pipeline := { s with send := snd, srcStaging := s.srcStaging.set p (s.srcHbm.getD p .junk) }
      refine ⟨.d2hDone p true, s', ?_, by simp [s', snd, hdr], rfl, by simp [s', snd]; omega⟩
      simp [step, sendStep, hs₁, s']
    | pushing k =>
      cases k with
      | zero => simp [hst, SendStage.opCount] at hop
      | succ k =>
        let snd := Send.endOp { s.send with pools := s.send.pools.set p { pool with stage := .pushing k } }
        have hs₁ : Send.step .shipping s.send (.pushDone p true) = some snd := by
          simp [Send.step, Send.pushDone, hget, hst, Send.recordPushCompletion, hdr, snd]
        let s' : Pipeline :=
          { s with send := snd
                   wire := s.wire.set p (s.srcStaging.getD p .junk)
                   sentStreams := s.sentStreams.set p (s.sentStreams.getD p 0 + 1) }
        refine ⟨.pushDone p true, s', ?_, by simp [s', snd, hdr], rfl, by simp [s', snd]; omega⟩
        simp [step, sendStep, hs₁, s']
  · cases hiss : Send.allIssued s.send.pools
    · obtain ⟨p, _, pool, hget, hp⟩ := Send.exists_pending_of_allIssued_false hiss
      let snd := Send.endOp { s.send with pools := s.send.pools.set p { pool with stage := .aborted },
                                          execActive := false }
      have hs₁ : Send.step .shipping s.send (.d2hIssue p false) = some snd := by
        simp [Send.step, Send.d2hIssue, hex, hget, hp, Send.recordPushCompletion, hdr, snd]
      let s' : Pipeline := { s with send := snd }
      refine ⟨.d2hIssue p false, s', ?_, by simp [s', snd, hdr], rfl, by simp [s', snd]; omega⟩
      simp [step, sendStep, hs₁, s']
    · let snd := Send.endOp { s.send with execActive := false }
      have hs₁ : Send.step .shipping s.send .execFinish = some snd := by
        simp [Send.step, Send.execFinish, hex, hiss, snd]
      let s' : Pipeline := { s with send := snd }
      refine ⟨.execFinish, s', ?_, by simp [s', snd, hdr], rfl, by simp [s', snd]; omega⟩
      simp [step, sendStep, hs₁, s']

theorem recv_drain_step_lift {s : Pipeline} (h : Inv s)
    (hdr : s.recv.life.draining = true) (hnd : s.recv.life.done = false) :
    ∃ e s', step s (.recv e) = some s' ∧
      s'.recv.life.draining = true ∧ s'.send = s.send ∧
      Recv.drainRank s'.recv < Recv.drainRank s.recv := by
  obtain ⟨e, rcv, hrcv, hdr', hlt⟩ := Recv.drain_step .finishOnError h.recvInv hdr hnd
  have hlift : ∃ s', step s (.recv e) = some s' ∧ s'.recv = rcv ∧ s'.send = s.send := by
    cases e with
    | pushBegin =>
      simp only [Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hrcv
      obtain ⟨_, hb, _⟩ := hrcv
      have hact := Lifecycle.beginOp_active hb
      simp [hdr] at hact
    | onPoolReceived p =>
      simp only [Recv.step, Recv.onPoolReceived] at hrcv
      split at hrcv <;> try cases hrcv
      rename_i pool hget
      split at hrcv <;> try cases hrcv
      rename_i hact
      simp only [Recv.launchEligible, hdr, true_or, or_true, ↓reduceIte] at hlt
      -- launchEligible is identity when draining = true, and setting waiting -> received
      -- leaves drainSteps = 0 unchanged, contradicting hlt!
      exfalso
      have hwait : pool.stage = .waiting := by
        simp only [not_or, Bool.not_eq_true, bne_iff_ne, ne_eq, Decidable.not_not] at hact
        exact hact.2
      have hd_eq : Recv.poolDrainSteps (s.recv.pools.set p { pool with stage := .received }) =
          Recv.poolDrainSteps s.recv.pools := by
        clear hlt hdr'
        have key : ∀ (ps : List Pool) (idx : Nat), ps[idx]? = some pool →
            Recv.poolDrainSteps (ps.set idx { pool with stage := .received }) =
            Recv.poolDrainSteps ps := by
          intro ps idx hi
          induction ps generalizing idx with
          | nil => simp at hi
          | cons q qs ih =>
            cases idx with
            | zero =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at hi
              subst hi
              simp [List.set_cons_zero, Recv.poolDrainSteps, hwait, PoolStage.drainSteps]
            | succ idx =>
              simp only [List.getElem?_cons_succ] at hi
              simp [List.set_cons_succ, Recv.poolDrainSteps, ih idx hi]
        exact key s.recv.pools p hget
      unfold Recv.drainRank at hlt
      simp only [hd_eq] at hlt
      omega
    | h2dDone p ok =>
      have hrcv_step : Recv.h2dDone p ok s.recv = some rcv := hrcv
      obtain ⟨pool, _, hget, _, _, _⟩ := h2dDone_pools_shape hrcv_step
      let s' : Pipeline :=
        if ok then
          { s with recv := rcv
                   dstHbm := writePoolHbm s.recv.pools p pool.rank
                               (s.dstStaging.getD p .junk) s.dstHbm }
        else { s with recv := rcv }
      refine ⟨s', ?_, ?_⟩
      · simp only [step, recvStep, hget, hrcv, Option.map_some, s']
      · dsimp only [s']; split <;> exact ⟨rfl, rfl⟩
    | pushEnd ok | h2dIssue p ok | h2dReady p | pollReady | cancel | publish =>
      exact ⟨{ s with recv := rcv }, by simp [step, recvStep, hrcv], rfl, rfl⟩
  obtain ⟨s', hs', hrcv_eq, hsnd_eq⟩ := hlift
  exact ⟨e, s', hs', by rw [hrcv_eq]; exact hdr', hsnd_eq, by rw [hrcv_eq]; exact hlt⟩

theorem drain_step {s : Pipeline} (h : Inv s)
    (hsdr : s.send.life.draining = true) (hrdr : s.recv.life.draining = true)
    (hnot_both : ¬(s.send.life.done = true ∧ s.recv.life.done = true)) :
    ∃ e s', step s e = some s' ∧
      s'.send.life.draining = true ∧ s'.recv.life.draining = true ∧
      drainRank s' < drainRank s := by
  cases hsnd_d : s.send.life.done
  · obtain ⟨e, s', hs', hdr', hrcv_eq, hlt⟩ := send_drain_step_lift h hsdr hsnd_d
    refine ⟨.send e, s', hs', hdr', by rw [hrcv_eq]; exact hrdr, ?_⟩
    unfold drainRank
    rw [hrcv_eq]; omega
  · have hrcv_nd : s.recv.life.done = false := by
      cases hr_d : s.recv.life.done
      · rfl
      · exact absurd ⟨hsnd_d, hr_d⟩ hnot_both
    obtain ⟨e, s', hs', hdr', hsnd_eq, hlt⟩ := recv_drain_step_lift h hrdr hrcv_nd
    refine ⟨.recv e, s', hs', by rw [hsnd_eq]; exact hsdr, hdr', ?_⟩
    unfold drainRank
    rw [hsnd_eq]; omega

theorem both_done_settle (sys₀ : System Pipeline Ev) (hstep : sys₀.step = step)
    {s : Pipeline} (h : Inv s)
    (hs_done : s.send.life.done = true) (hr_done : s.recv.life.done = true) :
    ∃ evs s', sys₀.runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.recv.life.done = true ∧
      s'.send.life.hasStaging = false ∧ s'.recv.life.hasStaging = false ∧
      s'.send.planUnregistered = true ∧ s'.recv.planUnregistered = true := by
  have hs_stg : s.send.life.hasStaging = false := by rw [h.sendInv.life.staging, hs_done]; rfl
  have hr_stg : s.recv.life.hasStaging = false := by rw [h.recvInv.life.staging, hr_done]; rfl
  have hs_pl : s.send.planUnregistered = true := h.sendInv.done_plan hs_done
  by_cases hpl : s.recv.planUnregistered = true
  · exact ⟨[], s, rfl, hs_done, hr_done, hs_stg, hr_stg, hs_pl, hpl⟩
  · cases hpub : s.recv.published with
    | none =>
      let s₁ : Pipeline :=
        { s with recv := { s.recv with published := some s.recv.life.statusOk,
                                       planUnregistered := true } }
      have hstep_pub : step s (.recv .publish) = some s₁ := by
        simp [step, recvStep, Recv.step, Recv.publish, hr_done, hpub, s₁]
      refine ⟨[.recv .publish], s₁, ?_, hs_done, hr_done, hs_stg, hr_stg, hs_pl, rfl⟩
      simp [System.runFrom, hstep, hstep_pub]
    | some b =>
      exact absurd (h.recvInv.published_done b hpub).2 hpl

/-- Both sessions drain to `done`, then `both_done_settle` finishes the job
(`System.settles_of_rank` over `drainRank` with `drain_step` as the one-step
argument). -/
theorem draining_can_settle (sys₀ : System Pipeline Ev) (hstep : sys₀.step = step)
    {s : Pipeline} (h : Inv s)
    (hsdr : s.send.life.draining = true) (hrdr : s.recv.life.draining = true) :
    ∃ evs s', sys₀.runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.recv.life.done = true ∧
      s'.send.life.hasStaging = false ∧ s'.recv.life.hasStaging = false ∧
      s'.send.planUnregistered = true ∧ s'.recv.planUnregistered = true := by
  obtain ⟨evs₁, s₁, hrun₁, ⟨hinv₁, _, _⟩, hsd, hrd⟩ :=
    sys₀.settles_of_rank
      (Inv := fun s => Inv s ∧ s.send.life.draining = true ∧ s.recv.life.draining = true)
      (Done := fun s => s.send.life.done = true ∧ s.recv.life.done = true) drainRank
      (fun s ⟨hi, hsdr, hrdr⟩ hnd => by
        obtain ⟨e, s₁, hs₁, hsdr₁, hrdr₁, hlt⟩ := drain_step hi hsdr hrdr hnd
        exact ⟨e, s₁, hstep ▸ hs₁, ⟨step_inv hi hs₁, hsdr₁, hrdr₁⟩, hlt⟩)
      ⟨h, hsdr, hrdr⟩
  obtain ⟨evs₂, s', hrun₂, h1, h2, h3, h4, h5, h6⟩ := both_done_settle sys₀ hstep hinv₁ hsd hrd
  exact ⟨evs₁ ++ evs₂, s', sys₀.runFrom_append evs₁ evs₂ hrun₁ hrun₂, h1, h2, h3, h4, h5, h6⟩

theorem can_settle_of_inv (sys₀ : System Pipeline Ev) (hstep : sys₀.step = step)
    {s : Pipeline} (hinv : Inv s) :
    ∃ evs s', sys₀.runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.recv.life.done = true ∧
      s'.send.life.hasStaging = false ∧ s'.recv.life.hasStaging = false ∧
      s'.send.planUnregistered = true ∧ s'.recv.planUnregistered = true := by
  let s₁ : Pipeline := { s with send := { s.send with life := s.send.life.finishLocked false } }
  let s₂ : Pipeline := { s₁ with recv := { s₁.recv with life := s₁.recv.life.finishLocked false } }
  have hs₁ : step s (.send .cancel) = some s₁ := rfl
  have hs₂ : step s₁ (.recv .cancel) = some s₂ := rfl
  have hinv₁ := step_inv hinv hs₁
  have hinv₂ := step_inv hinv₁ hs₂
  have hsdr₂ : s₂.send.life.draining = true := by simp [s₂, s₁]
  have hrdr₂ : s₂.recv.life.draining = true := by simp [s₂]
  obtain ⟨evs, s', hrun, h1, h2, h3, h4, h5, h6⟩ :=
    draining_can_settle sys₀ hstep hinv₂ hsdr₂ hrdr₂
  have hstep₁ : sys₀.step s (.send .cancel) = some s₁ := hstep ▸ hs₁
  have hstep₂ : sys₀.step s₁ (.recv .cancel) = some s₂ := hstep ▸ hs₂
  refine ⟨.send .cancel :: .recv .cancel :: evs, s', ?_, h1, h2, h3, h4, h5, h6⟩
  simp only [System.runFrom, List.foldlM_cons, hstep₁]
  change (sys₀.step s₁ (.recv .cancel) >>= fun s₃ => List.foldlM sys₀.step s₃ evs) = some s'
  rw [hstep₂]
  exact hrun

/-- Every reachable state of the composed reshard pipeline on a valid schedule
can drain both sessions to `done = true`, release both `PoolStagingLease`s
(`hasStaging = false`), and unregister both active plans
(`planUnregistered = true`) in finitely many steps. -/
theorem reachable_can_settle {ranks peerCounts : List Nat} {s : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (sys ranks peerCounts).Reachable s) :
    ∃ evs s', (sys ranks peerCounts).runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.recv.life.done = true ∧
      s'.send.life.hasStaging = false ∧ s'.recv.life.hasStaging = false ∧
      s'.send.planUnregistered = true ∧ s'.recv.planUnregistered = true :=
  can_settle_of_inv (sys ranks peerCounts) rfl (reachable_inv hvalid h)

/-- Every state reachable from the pre-Phase-1 unarmed state (`initUnarmed`)
can also drain both sessions to `done = true` and release all staging leases
and active plans in finitely many steps. -/
theorem reachable_unarmed_can_settle {ranks peerCounts : List Nat} {s : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (sysUnarmed ranks peerCounts).Reachable s) :
    ∃ evs s', (sysUnarmed ranks peerCounts).runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.recv.life.done = true ∧
      s'.send.life.hasStaging = false ∧ s'.recv.life.hasStaging = false ∧
      s'.send.planUnregistered = true ∧ s'.recv.planUnregistered = true :=
  can_settle_of_inv (sysUnarmed ranks peerCounts) rfl (reachable_unarmed_inv hvalid h)

/-! ### Buffer release predicates, non-interference, and single-request handoff

The predicates `SrcReleased` and `HandedOff` capture when a reshard request's
source buffers or all four buffers have been released, `wroteReleased_eq_false`
and `Inv.recycleSrcBufs` / `Inv.recycleAllBufs` package the single-request
write-after-release and read-after-release guarantees, and `inv_can_handoff`
packages single-request eventual drain for `MultiRequest.lean`. -/

/-- For any step other than `.reclaim`, source HBM is unmodified. -/
theorem srcHbm_quiet_step {s s' : Pipeline} {e : Ev}
    (hne : e ≠ .reclaim) (hs : step s e = some s') :
    s'.srcHbm = s.srcHbm := by
  cases e with
  | armRecv =>
    simp only [step, armRecv] at hs; split at hs <;> simp at hs; subst hs; rfl
  | send e =>
    cases e <;> simp only [step, sendStep, Option.map_eq_some_iff] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, rfl⟩ := hs; rfl
  | recv e =>
    cases e <;> simp only [step, recvStep] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
  | landStream p =>
    simp only [step, landStream] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; rfl
  | reclaim =>
    exact absurd rfl hne
  | reseatSrcStaging =>
    simp only [step, reseatSrcStaging] at hs; split at hs <;> simp at hs; subst hs; rfl
  | reseatDstStaging =>
    simp only [step, reseatDstStaging] at hs; split at hs <;> simp at hs; subst hs; rfl

/-- Every pipeline step preserves `numPools`. -/
theorem step_numPools {s s' : Pipeline} {e : Ev} (hs : step s e = some s') :
    s'.numPools = s.numPools := by
  cases e with
  | armRecv =>
    simp only [step, armRecv] at hs; split at hs <;> simp at hs; subst hs; rfl
  | send e =>
    cases e <;> simp only [step, sendStep, Option.map_eq_some_iff] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · obtain ⟨_, _, rfl⟩ := hs; rfl
    · obtain ⟨_, _, rfl⟩ := hs; rfl
  | recv e =>
    cases e <;> simp only [step, recvStep] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; rfl }
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; rfl
  | landStream p =>
    simp only [step, landStream] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; rfl
  | reclaim =>
    simp only [step, reclaim] at hs
    split at hs <;> simp at hs; subst hs; rfl
  | reseatSrcStaging =>
    simp only [step, reseatSrcStaging] at hs; split at hs <;> simp at hs; subst hs; rfl
  | reseatDstStaging =>
    simp only [step, reseatDstStaging] at hs; split at hs <;> simp at hs; subst hs; rfl

/-- Once the sender releases its staging lease, no event other than `.reseatSrcStaging`
can modify `srcStaging` or `wire`. -/
theorem srcStaging_quiet_step {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hst : s.send.life.hasStaging = false) (hne : e ≠ .reseatSrcStaging)
    (hs : step s e = some s') :
    s'.srcStaging = s.srcStaging ∧ s'.wire = s.wire := by
  cases e with
  | armRecv =>
    simp only [step, armRecv] at hs; split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩
  | send e =>
    exact srcStaging_quiet h hst hs
  | recv e =>
    cases e <;> simp only [step, recvStep] at hs
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · split at hs <;> try cases hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨_, _, hs'⟩ := hs; split at hs' <;> { subst hs'; exact ⟨rfl, rfl⟩ }
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
    · simp only [Option.map_eq_some_iff] at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩
  | landStream p =>
    simp only [step, landStream] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩
  | reclaim =>
    simp only [step, reclaim] at hs
    split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩
  | reseatSrcStaging =>
    exact absurd rfl hne
  | reseatDstStaging =>
    simp only [step, reseatDstStaging] at hs; split at hs <;> simp at hs; subst hs; exact ⟨rfl, rfl⟩

/-- True if `e` is one of the environment buffer-overwrite events (`.reclaim`,
`.reseatSrcStaging`, `.reseatDstStaging`). -/
def isRecycleEv (e : Ev) : Bool :=
  e == .reclaim || e == .reseatSrcStaging || e == .reseatDstStaging

/-- Per-buffer write-after-release detector: checks whether a transition `r → r'`
modified any of the four buffers after `r` had already released that specific
buffer. -/
def wroteReleased (r r' : Pipeline) : Bool :=
  (r.send.published != none && r'.srcHbm != r.srcHbm) ||
  (!r.send.life.hasStaging && r'.srcStaging != r.srcStaging) ||
  (!r.recv.life.hasStaging && r'.dstStaging != r.dstStaging) ||
  (r.recv.published != none && r'.dstHbm != r.dstHbm)

/-- Key per-buffer non-interference lemma: for any reshard request `r` satisfying `Inv r`,
no transfer step ever modifies any of the four buffers after `r` has released
that specific buffer (`wroteReleased r r' = false`). -/
theorem wroteReleased_eq_false {r r' : Pipeline} {e : Ev} (h : Inv r)
    (hne₀ : e ≠ .reclaim)
    (hne₁ : e ≠ .reseatSrcStaging) (hne₂ : e ≠ .reseatDstStaging)
    (hs : step r e = some r') :
    wroteReleased r r' = false := by
  have hq₀ : r'.srcHbm = r.srcHbm := srcHbm_quiet_step hne₀ hs
  have hq₁ : (!r.send.life.hasStaging && r'.srcStaging != r.srcStaging) = false := by
    cases hst : r.send.life.hasStaging
    · simp [(srcStaging_quiet_step h hst hne₁ hs).1]
    · simp
  have hq₂ : (!r.recv.life.hasStaging && r'.dstStaging != r.dstStaging) = false := by
    cases hst : r.recv.life.hasStaging
    · simp [(dstStaging_quiet h hst hne₂ hs).1]
    · simp
  have hq₃ : (r.recv.published != none && r'.dstHbm != r.dstHbm) = false := by
    cases hp : r.recv.published with
    | none => simp
    | some b =>
      have hne : r.recv.published ≠ none := by simp [hp]
      simp [(dstHbm_quiet h hne hs).1]
  simp [wroteReleased, hq₀, hq₁, hq₂, hq₃]

theorem not_isRecycleEv_and_wroteReleased_eq_false {r r' : Pipeline} {e : Ev}
    (h : Inv r) (hs : step r e = some r') :
    (!isRecycleEv e && wroteReleased r r') = false := by
  by_cases he : isRecycleEv e = true
  · simp [he]
  · simp only [isRecycleEv, Bool.or_eq_true, beq_iff_eq, not_or] at he
    rw [wroteReleased_eq_false h he.1.1 he.1.2 he.2 hs]
    simp

/-- The source side of a reshard request has released both of its buffers: its host
staging buffer has been returned to `PoolStagingStorage`
(`send.life.hasStaging = false`) and `CompleteReadWithDetails` has published
`done_sending` or `failed_recving` (`send.published ≠ none`). -/
def SrcReleased (s : Pipeline) : Prop :=
  s.send.life.hasStaging = false ∧
  s.send.published ≠ none

instance (s : Pipeline) : Decidable (SrcReleased s) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- TPU Sync has finished with all four memories of this reshard transfer:
both host staging leases have been released back to `PoolStagingStorage`
(`hasStaging = false`), and both HBM outcomes have been published via
`CompleteReadWithDetails` (`send.published ≠ none`, `recv.published ≠ none`). -/
def HandedOff (s : Pipeline) : Prop :=
  s.send.life.hasStaging = false ∧
  s.recv.life.hasStaging = false ∧
  s.send.published ≠ none ∧
  s.recv.published ≠ none

instance (s : Pipeline) : Decidable (HandedOff s) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

theorem HandedOff.srcReleased {s : Pipeline} (h : HandedOff s) : SrcReleased s :=
  ⟨h.1, h.2.2.1⟩

/-- Overwrite a request's source buffers (`srcHbm` and `srcStaging`) with `.junk`
when a later request recycles them (`recycleSrc`). -/
def recycleSrcBufs (s : Pipeline) : Pipeline :=
  { s with
    srcHbm := List.replicate s.numPools .junk,
    reclaimed := true,
    srcStaging := List.replicate s.numPools .junk }

/-- Overwrite a request's `srcHbm`, `srcStaging`, and `dstStaging` with `.junk`
when a later request recycles all buffers (`nextRequest`). -/
def recycleAllBufs (s : Pipeline) : Pipeline :=
  { s with
    srcHbm := List.replicate s.numPools .junk,
    reclaimed := true,
    srcStaging := List.replicate s.numPools .junk,
    dstStaging := List.replicate s.numPools .junk }

theorem Inv.recycleSrcBufs {s : Pipeline} (h : Inv s) (hp : SrcReleased s) :
    Inv (Pipeline.recycleSrcBufs s) := by
  have hd : s.send.life.done = true := by
    have hst := hp.1
    rw [h.sendInv.life.staging] at hst; simpa using hst
  unfold Pipeline.recycleSrcBufs
  exact ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    by simp, by simp, h.len_wire, h.len_dstStg, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match,
    fun hr => by simp at hr, fun _ => hp.2,
    fun hnd => by simp [hd] at hnd,
    h.sent_bound, h.wire_ok, h.landed_le_sent,
    h.dstStg_ok, h.recv_stage_landed, h.done_intact,
    h.unarmed_send, h.unarmed_recv⟩

theorem Inv.recycleAllBufs {s : Pipeline} (h : Inv s) (hp : HandedOff s) :
    Inv (Pipeline.recycleAllBufs s) := by
  have hdS : s.send.life.done = true := by
    have hst := hp.1
    rw [h.sendInv.life.staging] at hst; simpa using hst
  have hdR : s.recv.life.done = true := by
    have hst := hp.2.1
    rw [h.recvInv.life.staging] at hst; simpa using hst
  unfold Pipeline.recycleAllBufs
  exact ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    by simp, by simp, h.len_wire, by simp, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match,
    fun hr => by simp at hr, fun _ => hp.2.2.1,
    fun hnd => by simp [hdS] at hnd,
    h.sent_bound, h.wire_ok, h.landed_le_sent,
    fun hnd => by simp [hdR] at hnd,
    h.recv_stage_landed, h.done_intact,
    h.unarmed_send, h.unarmed_recv⟩

/-- Once a reshard transfer has released both staging buffers and published both
HBM outcomes (`HandedOff`), no subsequent transfer transition can ever modify
any of the five memories (`srcHbm`, `srcStaging`, `wire`, `dstStaging`, `dstHbm`).
Proved by combining `srcHbm_quiet_step`, `srcStaging_quiet_step`,
`dstStaging_quiet`, and `dstHbm_quiet`. -/
theorem handedOff_quiet {r r' : Pipeline} {e : Ev} (h : Inv r) (hp : HandedOff r)
    (hne₀ : e ≠ .reclaim)
    (hne₁ : e ≠ .reseatSrcStaging) (hne₂ : e ≠ .reseatDstStaging)
    (hs : step r e = some r') :
    r'.srcHbm = r.srcHbm ∧
    r'.srcStaging = r.srcStaging ∧
    r'.wire = r.wire ∧
    r'.dstStaging = r.dstStaging ∧
    r'.dstHbm = r.dstHbm :=
  ⟨srcHbm_quiet_step hne₀ hs,
   (srcStaging_quiet_step h hp.1 hne₁ hs).1,
   (srcStaging_quiet_step h hp.1 hne₁ hs).2,
   (dstStaging_quiet h hp.2.1 hne₂ hs).1,
   (dstHbm_quiet h hp.2.2.2 hs).1⟩

/-- From any settled reshard pipeline state, publishing both sessions completes handoff. -/
theorem settled_can_handoff (sys₀ : System Pipeline Ev) (hstep : sys₀.step = step)
    {s : Pipeline}
    (hs_done : s.send.life.done = true) (hr_done : s.recv.life.done = true)
    (hs_stg : s.send.life.hasStaging = false) (hr_stg : s.recv.life.hasStaging = false) :
    ∃ evs s', sys₀.runFrom s evs = some s' ∧ HandedOff s' := by
  cases hpubS : s.send.published with
  | none =>
    let s₁ : Pipeline :=
      { s with send := { s.send with published := some s.send.life.statusOk } }
    have hstepS : step s (.send .publish) = some s₁ := by
      simp [step, sendStep, Send.step, Send.publish, hs_done, hpubS, s₁]
    cases hpubR : s.recv.published with
    | none =>
      let s₂ : Pipeline :=
        { s₁ with recv := { s₁.recv with published := some s₁.recv.life.statusOk,
                                         planUnregistered := true } }
      have hstepR : step s₁ (.recv .publish) = some s₂ := by
        simp [step, recvStep, Recv.step, Recv.publish, hr_done, hpubR, s₂, s₁]
      refine ⟨[.send .publish, .recv .publish], s₂, ?_, ⟨hs_stg, hr_stg, by simp [s₂, s₁], by simp [s₂]⟩⟩
      simp [System.runFrom, hstep, hstepS, hstepR]
    | some bR =>
      refine ⟨[.send .publish], s₁, ?_, ⟨hs_stg, hr_stg, by simp [s₁], by simp [s₁, hpubR]⟩⟩
      simp [System.runFrom, hstep, hstepS]
  | some bS =>
    cases hpubR : s.recv.published with
    | none =>
      let s₂ : Pipeline :=
        { s with recv := { s.recv with published := some s.recv.life.statusOk,
                                       planUnregistered := true } }
      have hstepR : step s (.recv .publish) = some s₂ := by
        simp [step, recvStep, Recv.step, Recv.publish, hr_done, hpubR, s₂]
      refine ⟨[.recv .publish], s₂, ?_, ⟨hs_stg, hr_stg, by simp [s₂, hpubS], by simp [s₂]⟩⟩
      simp [System.runFrom, hstep, hstepR]
    | some bR =>
      refine ⟨[], s, rfl, ⟨hs_stg, hr_stg, by simp [hpubS], by simp [hpubR]⟩⟩

/-- From any pipeline state satisfying `Inv`, a finite trace settles both
sessions, releases both host staging buffers to `PoolStagingStorage`, and
publishes both HBM outcomes (`HandedOff`). -/
theorem inv_can_handoff {ranks peerCounts : List Nat} {s : Pipeline}
    (hinv : Inv s) :
    ∃ evs s', (sys ranks peerCounts).runFrom s evs = some s' ∧ HandedOff s' := by
  obtain ⟨evs₁, s₁, hrun₁, hsd, hrd, hstS, hstR, _, _⟩ :=
    can_settle_of_inv (sys ranks peerCounts) rfl hinv
  obtain ⟨evs₂, s', hrun₂, hhand⟩ := settled_can_handoff (sys ranks peerCounts) rfl hsd hrd hstS hstR
  exact ⟨evs₁ ++ evs₂, s', (sys ranks peerCounts).runFrom_append evs₁ evs₂ hrun₁ hrun₂, hhand⟩


end Pipeline

end TpuSyncVerify.Transfer.Reshard

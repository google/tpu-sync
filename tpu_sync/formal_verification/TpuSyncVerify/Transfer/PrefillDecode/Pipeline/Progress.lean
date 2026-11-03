import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Invariant

/-! # `Pipeline`: progress, buffer quietness after release, and single-request handoff

Part of `TpuSyncVerify.Transfer.PrefillDecode.Pipeline`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ### Progress and eventual settlement -/

/-- Remaining steps needed to drain all in-flight operations once both sessions
are draining. -/
def drainRank (s : Pipeline) : Nat :=
  Send.drainRank s.send + Recv.drainRank s.recv

theorem drain_send_step {s : Pipeline} (h : Inv s)
    (hdr : s.send.life.draining = true) (hnd : s.send.life.done = false) :
    ∃ e s', step s e = some s' ∧ s'.send.life.draining = true ∧
      s'.recv = s.recv ∧ Send.drainRank s'.send < Send.drainRank s.send := by
  have hif : 0 < s.send.life.inFlight := by
    cases h0 : s.send.life.inFlight
    · have := h.send.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := h.send.accounted
  have hcnt := h.send.counters
  unfold Send.Accounted at hacc
  unfold Send.CountersOrdered at hcnt
  by_cases hp : s.send.d2hPending = true
  · have hs₁ : step s (.send (.d2hIssue true)) =
        some { s with send := { s.send with d2hPending := false, d2hIssued := s.send.d2hIssued + 1 } } := by
      simp [step, sendStep, Send.step, Send.d2hIssue, hp]
    exact ⟨.send (.d2hIssue true), _, hs₁, by simp [hdr], rfl, by simp [Send.drainRank, hp]; omega⟩
  · have hp' : s.send.d2hPending = false := by
      cases hq : s.send.d2hPending
      · rfl
      · exact absurd hq hp
    by_cases hrd : s.send.d2hReady < s.send.d2hIssued
    · have hk : s.send.d2hIssued ≤ s.d2hReadyL.length := by
        rw [h.len_d2hReadyL, ← h.n_send]; omega
      obtain ⟨l, hl, hf⟩ := exists_false_lt hk (by rw [h.cnt_d2hReady]; exact hrd)
      have hs₁ : step s (.d2hReady l) =
          some { s with send := { s.send with d2hReady := s.send.d2hReady + 1 },
                        prefillStaging := s.prefillStaging.set l (s.prefillHbm.getD l .junk),
                        d2hReadyL := s.d2hReadyL.set l true } := by
        simp [step, d2hReady, hl, hf, Send.step, Send.markD2hReady, hrd]
      exact ⟨.d2hReady l, _, hs₁, by simp [hdr], rfl, by simp [Send.drainRank]; omega⟩
    · by_cases hret : s.send.d2hRetired < s.send.d2hReady
      · have hs₁ : step s (.send .d2hEnd) =
            some { s with send := { s.send with d2hRetired := s.send.d2hRetired + 1 }.endOp } := by
          simp [step, sendStep, Send.step, Send.d2hEnd, hret]
        exact ⟨.send .d2hEnd, _, hs₁, by simp [hdr], rfl, by rw [Send.drainRank_endOp]; simp [Send.drainRank]; omega⟩
      · by_cases hwk : s.send.woken < s.send.queued
        · by_cases hwi : s.send.woken < s.send.d2hIssued
          · have hwr : s.send.woken < s.send.d2hReady := by omega
            have hready_eq : s.send.d2hIssued ≤ countTrue s.d2hReadyL := by
              rw [h.cnt_d2hReady]; omega
            have hwl : s.d2hReadyL[s.send.woken]? = some true :=
              all_true_lt_of_countTrue h.d2hReady_lt hready_eq s.send.woken hwi
            have hs₁ : step s (.send (.wake true)) =
                some { s with send := { s.send with woken := s.send.woken + 1 }.endOp } := by
              simp [step, sendStep, hwl, Send.step, Send.wake, hwk, hwr, hdr]
            exact ⟨.send (.wake true), _, hs₁, by simp [hdr], rfl, by rw [Send.drainRank_endOp]; simp [Send.drainRank]; omega⟩
          · have hwr : ¬ s.send.woken < s.send.d2hReady := by omega
            have hle : s.send.d2hIssued ≤ s.send.woken := by omega
            have hs₁ : step s (.send (.wake false)) =
                some { s with send := ({ s.send with woken := s.send.woken + 1 }.finish false).endOp } := by
              simp [step, sendStep, Send.step, Send.wake, hwk, hwr, hdr, hp', hle]
            rw [Send.finish_of_draining false (s := { s.send with woken := s.send.woken + 1 }) hdr] at hs₁
            exact ⟨.send (.wake false), _, hs₁, by simp [hdr], rfl, by rw [Send.drainRank_endOp]; simp [Send.drainRank]; omega⟩
        · by_cases hpool : 0 < s.send.pooled
          · have hs₁ : step s (.send .h2hIssue) =
                some { s with send := { s.send with pooled := s.send.pooled - 1 }.endOp } := by
              simp [step, sendStep, Send.step, Send.h2hIssue, Lifecycle.beginOp_of_draining hdr]; omega
            exact ⟨.send .h2hIssue, _, hs₁, by simp [hdr], rfl, by rw [Send.drainRank_endOp]; simp [Send.drainRank]; omega⟩
          · by_cases hch : 0 < s.send.chaining
            · have hs₁ : step s (.send .sendNext) =
                  some { s with send := { s.send with chaining := s.send.chaining - 1 }.endOp } := by
                simp [step, sendStep, Send.step, Send.sendNext,
                  Send.trySendNext_of_draining (s := { s.send with chaining := s.send.chaining - 1 }) hdr]
                omega
              exact ⟨.send .sendNext, _, hs₁, by simp [hdr], rfl, by rw [Send.drainRank_endOp]; simp [Send.drainRank]; omega⟩
            · have hh2h : s.send.h2hRetired < s.send.h2hIssued := by
                cases hdp : s.send.d2hPending <;> simp_all <;> omega
              have hk : s.send.h2hIssued ≤ s.h2hRetiredL.length := by
                rw [h.len_h2hRetiredL, ← h.n_send]; omega
              obtain ⟨l, hl, hf⟩ := exists_false_lt hk (by rw [h.cnt_h2hRetired]; exact hh2h)
              have hs₁ : step s (.h2hDone l true) =
                  some { s with send := { s.send with h2hRetired := s.send.h2hRetired + 1, h2hOk := s.send.h2hOk + 1 }.endOp,
                                wire := s.wire.set l (s.prefillStaging.getD l .junk),
                                h2hRetiredL := s.h2hRetiredL.set l true } := by
                simp [step, h2hDone, hl, hf, Send.step, Send.h2hDone, hh2h,
                  Send.finish_of_draining true (s := { s.send with h2hRetired := s.send.h2hRetired + 1, h2hOk := s.send.h2hOk + 1 }) hdr]
              exact ⟨.h2hDone l true, _, hs₁, by simp [hdr], rfl, by rw [Send.drainRank_endOp]; simp [Send.drainRank]; omega⟩

theorem drain_recv_step {s : Pipeline} (h : Inv s)
    (hdr : s.recv.life.draining = true) (hnd : s.recv.life.done = false) :
    ∃ e s', step s e = some s' ∧ s'.recv.life.draining = true ∧
      s'.send = s.send ∧ Recv.drainRank s'.recv < Recv.drainRank s.recv := by
  have hif : 0 < s.recv.life.inFlight := by
    cases h0 : s.recv.life.inFlight
    · have := h.recv.life.prompt hdr h0; simp [hnd] at this
    · omega
  have hacc := h.recv.accounted
  unfold Recv.Accounted at hacc
  have hr := h.recv.retired_le
  have hy := h.recv.ready_le
  by_cases hp : 0 < s.recv.pushes
  · have hs₁ : step s (.recv .pushEnd) =
        some { s with recv := { s.recv with life := s.recv.life.endOpLocked, pushes := s.recv.pushes - 1 } } := by
      simp [step, recvStep, Recv.step, Recv.pushEnd]; omega
    exact ⟨.recv .pushEnd, _, hs₁, by simp [hdr], rfl, by simp [Recv.drainRank]; omega⟩
  · by_cases hq : s.recv.pullPending = true
    · have hs₁ : step s (.recv (.pullReply false)) =
          some { s with recv := { s.recv with life := (s.recv.life.finishLocked false).endOpLocked, pullPending := false },
                        pullWaiting := false } := by
        simp [step, recvStep, Recv.step, Recv.pullReply, hq]
      exact ⟨.recv (.pullReply false), _, hs₁, by simp, rfl, by simp [Recv.drainRank, hq]⟩
    · by_cases hpend : 0 < s.recv.pending
      · obtain ⟨l, _, hpl⟩ := exists_true_of_countTrue_pos (by rw [h.cnt_h2dPending]; exact hpend)
        have hs₁ : step s (.h2dIssue l true) =
            some { s with recv := { s.recv with pending := s.recv.pending - 1, life := s.recv.life.endOpLocked },
                          h2dPendingL := s.h2dPendingL.set l false,
                          h2dIssuedL := s.h2dIssuedL } := by
          simp [step, h2dIssue, hpl, Recv.step, Recv.h2dIssue, hdr]; omega
        exact ⟨.h2dIssue l true, _, hs₁, by simp [hdr], rfl, by simp [Recv.drainRank]; omega⟩
      · by_cases hrd : s.recv.ready < s.recv.issued
        · have hlen : s.h2dReadyL.length = s.h2dIssuedL.length := by
            rw [h.len_h2dReadyL, h.len_h2dIssuedL]
          have hcnt : countTrue s.h2dReadyL < countTrue s.h2dIssuedL := by
            rw [h.cnt_h2dReady, h.cnt_h2dIssued]; exact hrd
          obtain ⟨l, _, hrf, hit⟩ := exists_diff_of_countTrue_lt hlen hcnt
          have hs₁ : step s (.h2dReady l) =
              some { s with recv := { s.recv with ready := s.recv.ready + 1 },
                            decodeHbm := s.decodeHbm.set l (s.decodeStaging.getD l .junk),
                            h2dReadyL := s.h2dReadyL.set l true } := by
            simp [step, h2dReady, hit, hrf, Recv.step, Recv.h2dReady, hrd]
          exact ⟨.h2dReady l, _, hs₁, by simp [hdr], rfl, by simp [Recv.drainRank]; omega⟩
        · have hret : s.recv.retired < s.recv.ready := by
            cases hqp : s.recv.pullPending <;> simp_all <;> omega
          have hs₁ : step s (.recv (.h2dDone true)) =
              some { s with recv := { s.recv with retired := s.recv.retired + 1, completed := s.recv.completed + 1, life := s.recv.life.endOpLocked } } := by
            simp [step, recvStep, Recv.step, Recv.h2dDone, hret, hnd, hdr]
          exact ⟨.recv (.h2dDone true), _, hs₁, by simp [hdr], rfl, by simp [Recv.drainRank]; omega⟩

/-- Both sessions drain to `done` with staging released
(`System.settles_of_rank` over `drainRank`; `drain_send_step` / `drain_recv_step`
supply the one-step argument, whichever session is still open). -/
theorem draining_can_settle (n : Nat) {s : Pipeline} (h : Inv s)
    (hdrS : s.send.life.draining = true) (hdrR : s.recv.life.draining = true) :
    ∃ evs s', (sys n).runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.send.life.hasStaging = false ∧
      s'.recv.life.done = true ∧ s'.recv.life.hasStaging = false := by
  obtain ⟨evs, s', hrun, ⟨hinv', _, _⟩, hdS', hdR'⟩ :=
    (sys n).settles_of_rank
      (Inv := fun s => Inv s ∧ s.send.life.draining = true ∧ s.recv.life.draining = true)
      (Done := fun s => s.send.life.done = true ∧ s.recv.life.done = true) drainRank
      (fun s ⟨hi, hdrS, hdrR⟩ hnd => by
        cases hndS : s.send.life.done
        · obtain ⟨e, s₁, hs₁, hdrS₁, hrecv_eq, hlt⟩ := drain_send_step hi hdrS hndS
          refine ⟨e, s₁, hs₁, ⟨step_inv hi hs₁, hdrS₁, by rw [hrecv_eq]; exact hdrR⟩, ?_⟩
          unfold drainRank; rw [hrecv_eq]; omega
        · cases hndR : s.recv.life.done
          · obtain ⟨e, s₁, hs₁, hdrR₁, hsend_eq, hlt⟩ := drain_recv_step hi hdrR hndR
            refine ⟨e, s₁, hs₁, ⟨step_inv hi hs₁, by rw [hsend_eq]; exact hdrS, hdrR₁⟩, ?_⟩
            unfold drainRank; rw [hsend_eq]; omega
          · exact absurd ⟨hndS, hndR⟩ hnd)
      ⟨h, hdrS, hdrR⟩
  exact ⟨evs, s', hrun, hdS', by rw [hinv'.send.life.staging, hdS']; rfl,
    hdR', by rw [hinv'.recv.life.staging, hdR']; rfl⟩

/-- The remaining finalization events once both sessions have settled: publish
any unpublished session outcome and reclaim prefill HBM if not yet reclaimed. -/
def finalizeEvs (s : Pipeline) : List Ev :=
  (if s.send.published.isNone then [.send .publish] else []) ++
  (if s.recv.published.isNone then [.recv .publish] else []) ++
  (if s.reclaimed then [] else [.reclaim])

/-- Once both sessions have settled (`done = true`), `finalizeEvs s` publishes
both outcomes and reclaims prefill HBM in at most three steps. -/
theorem settled_can_finalize (n : Nat) {s : Pipeline}
    (hdS : s.send.life.done = true) (hstS : s.send.life.hasStaging = false)
    (hdR : s.recv.life.done = true) (hstR : s.recv.life.hasStaging = false) :
    ∃ evs s', (sys n).runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.send.life.hasStaging = false ∧
      s'.recv.life.done = true ∧ s'.recv.life.hasStaging = false ∧
      s'.send.published ≠ none ∧ s'.recv.published ≠ none ∧ s'.reclaimed = true := by
  refine ⟨finalizeEvs s, ?_⟩
  cases hpS : s.send.published <;> cases hpR : s.recv.published <;> cases hrec : s.reclaimed <;>
    simp [finalizeEvs, System.runFrom, sys, step, sendStep, recvStep,
      Send.step, Send.publish, Recv.step, Recv.publish, reclaim,
      hdS, hstS, hdR, hstR, hpS, hpR, hrec]

/-- From any pipeline state satisfying `Inv`, a finite trace settles both
sessions, releases both staging buffers, publishes both outcomes, and reclaims
prefill HBM. -/
theorem inv_can_settle (n : Nat) {s : Pipeline} (hinv : Inv s) :
    ∃ evs s', (sys n).runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.send.life.hasStaging = false ∧
      s'.recv.life.done = true ∧ s'.recv.life.hasStaging = false ∧
      s'.send.published ≠ none ∧ s'.recv.published ≠ none ∧ s'.reclaimed = true := by
  have hcS : step s (.send .cancel) = some { s with send := s.send.finish false } := rfl
  let s₁ : Pipeline := { s with send := s.send.finish false }
  have hinv₁ : Inv s₁ := step_inv hinv hcS
  have hcR : step s₁ (.recv .cancel) =
      some { s₁ with recv := { s₁.recv with life := s₁.recv.life.finishLocked false } } := rfl
  let s₂ : Pipeline := { s₁ with recv := { s₁.recv with life := s₁.recv.life.finishLocked false } }
  have hinv₂ : Inv s₂ := step_inv hinv₁ hcR
  have hdrS₂ : s₂.send.life.draining = true :=
    hinv.send.life.finishOnceLocked_draining false
  have hdrR₂ : s₂.recv.life.draining = true := by simp [s₂]
  obtain ⟨evs_d, s_d, hrd, hdS, hstS, hdR, hstR⟩ := draining_can_settle n hinv₂ hdrS₂ hdrR₂
  obtain ⟨evs_f, s', hrf, hdS', hstS', hdR', hstR', hpubS', hpubR', hrec'⟩ :=
    settled_can_finalize n hdS hstS hdR hstR
  have hr_cancel : (sys n).runFrom s [.send .cancel, .recv .cancel] = some s₂ := by
    simp [System.runFrom, sys, hcS, hcR, s₁, s₂]
  exact ⟨[.send .cancel, .recv .cancel] ++ evs_d ++ evs_f, s',
    (sys n).runFrom_append _ evs_f ((sys n).runFrom_append _ evs_d hr_cancel hrd) hrf,
    hdS', hstS', hdR', hstR', hpubS', hpubR', hrec'⟩

/-- Every reachable pipeline state can settle both sessions, release both
staging buffers, publish both outcomes, and reclaim prefill HBM in a finite
number of steps. -/
theorem reachable_can_settle {n : Nat} {s : Pipeline} (h : (sys n).Reachable s) :
    ∃ evs s', (sys n).runFrom s evs = some s' ∧
      s'.send.life.done = true ∧ s'.send.life.hasStaging = false ∧
      s'.recv.life.done = true ∧ s'.recv.life.hasStaging = false ∧
      s'.send.published ≠ none ∧ s'.recv.published ≠ none ∧ s'.reclaimed = true :=
  inv_can_settle n (reachable_inv h)

/-! ### Buffer quietness and non-interference after release

`DecodeHbmSafe`, `PrefillHbmSafe`, and `StagingSafe` state that when a buffer is
released, the session's internal counters for that buffer are drained. The
theorems below turn those counter equalities into the memory-level
non-interference property needed across requests: once a transfer releases a
buffer, no subsequent transition of that transfer can ever read or write that
buffer, and overwriting released buffers with `.junk` preserves `Inv`. -/

/-- Unfold `step` for a known event and split every branch, leaving `hs` as
`some … = some s'`, as the `Option.map` form, or closed. -/
macro "pipe_cases" hs:ident : tactic =>
  `(tactic| (simp only [step, notifyForRead, pullWait, sendStep, recvStep, d2hReady, h2hDone,
      h2dBegin, h2dIssue, h2dReady, land, reclaim, reseatPrefillStaging, reseatDecodeStaging]
      at $hs:ident <;>
      (repeat' split at $hs:ident) <;>
      (try simp only [Option.map_eq_some_iff] at $hs:ident)))

theorem step_numLayers {s s' : Pipeline} {e : Ev} (hs : step s e = some s') :
    s'.numLayers = s.numLayers := by
  cases e <;> (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
  first
    | (obtain ⟨_, _, rfl⟩ := hs; rfl)
    | (cases hs <;> rfl)
    | cases hs

theorem step_send_published_mono {s s' : Pipeline} {e : Ev} {b : Bool} (hs : step s e = some s')
    (hp : s.send.published = some b) : s'.send.published = some b := by
  cases e <;> (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
  first
    | (obtain ⟨_, hr, rfl⟩ := hs; first | exact hp | exact Send.step_published_mono hr hp)
    | (cases hs <;> exact hp)
    | cases hs

theorem step_send_published_ne_none {s s' : Pipeline} {e : Ev} (hs : step s e = some s')
    (hp : s.send.published ≠ none) : s'.send.published ≠ none := by
  cases h : s.send.published with
  | none => exact absurd h hp
  | some b => rw [step_send_published_mono hs h]; simp

theorem step_published_mono {s s' : Pipeline} {e : Ev} {b : Bool} (hs : step s e = some s')
    (hp : s.recv.published = some b) : s'.recv.published = some b := by
  cases e <;> (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
  first
    | (obtain ⟨_, hr, rfl⟩ := hs; first | exact hp | exact Recv.step_published_mono hr hp)
    | (cases hs <;> exact hp)
    | cases hs

theorem step_published_ne_none {s s' : Pipeline} {e : Ev} (hs : step s e = some s')
    (hp : s.recv.published ≠ none) : s'.recv.published ≠ none := by
  cases h : s.recv.published with
  | none => exact absurd h hp
  | some b => rw [step_published_mono hs h]; simp

theorem step_send_done_mono {s s' : Pipeline} {e : Ev} (hs : step s e = some s')
    (hd : s.send.life.done = true) : s'.send.life.done = true := by
  cases e <;> (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
  first
    | (obtain ⟨_, hr, rfl⟩ := hs; first | exact hd | exact Send.step_done_mono hr hd)
    | (cases hs <;> exact hd)
    | cases hs

theorem step_recv_done_mono {s s' : Pipeline} {e : Ev} (hs : step s e = some s')
    (hd : s.recv.life.done = true) : s'.recv.life.done = true := by
  cases e <;> (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
  first
    | (obtain ⟨_, hr, rfl⟩ := hs; first | exact hd | exact Recv.step_done_mono hr hd)
    | (cases hs <;> exact hd)
    | cases hs

theorem numLayers_eq {n : Nat} {s : Pipeline} (h : (sys n).Reachable s) : s.numLayers = n :=
  (sys n).reachable_induction (P := fun s => s.numLayers = n) rfl
    (fun _ _ _ hi hs => (step_numLayers hs).trans hi) h

/-- Once `poll_stats()` reports `done_sending` or `failed_recving`
(`s.send.published ≠ none`), `d2hReady` is permanently disabled (`PrefillHbmSafe`
gives `d2hRetired = d2hIssued`, which with `d2hRetired ≤ d2hReady ≤ d2hIssued`
forces `d2hReady = d2hIssued`). -/
theorem prefillHbm_quiet {s : Pipeline} (h : Inv s) (hp : s.send.published ≠ none) (l : Nat) :
    step s (.d2hReady l) = none := by
  have ⟨_, hret⟩ := (inv_safe h).2.2.1 hp
  have hcnt := h.send.counters
  unfold Send.CountersOrdered at hcnt
  have hnr : ¬ (s.send.d2hReady < s.send.d2hIssued) := by omega
  simp [step, d2hReady, Send.step, Send.markD2hReady, hnr]

theorem prefillHbm_quiet_step {s s' : Pipeline} {e : Ev}
    (hne : e ≠ .reclaim) (hs : step s e = some s') :
    s'.prefillHbm = s.prefillHbm := by
  cases e with
  | reclaim => exact absurd rfl hne
  | _ =>
    (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
    first | (obtain ⟨_, _, rfl⟩ := hs; rfl) | (cases hs <;> rfl) | cases hs

/-- Once the send session releases its staging buffer (`hasStaging = false`), no
transfer event (anything other than the environment's `reseatPrefillStaging`)
can modify `prefillStaging` or `wire`. Proved from `StagingSafe`. -/
theorem prefillStaging_quiet {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hst : s.send.life.hasStaging = false) (hne : e ≠ .reseatPrefillStaging)
    (hs : step s e = some s') :
    s'.prefillStaging = s.prefillStaging ∧ s'.wire = s.wire := by
  have ⟨_, hd2h, hh2h⟩ := (inv_safe h).2.2.2.1.1 hst
  have hcnt := h.send.counters
  unfold Send.CountersOrdered at hcnt
  cases e with
  | d2hReady l =>
    have hnr : ¬ (s.send.d2hReady < s.send.d2hIssued) := by omega
    simp [step, d2hReady, Send.step, Send.markD2hReady, hnr] at hs
  | h2hDone l ok =>
    have hnr : ¬ (s.send.h2hRetired < s.send.h2hIssued) := by omega
    simp [step, h2hDone, Send.step, Send.h2hDone, hnr] at hs
  | reseatPrefillStaging => exact absurd rfl hne
  | _ =>
    (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
    first | (obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩) | (cases hs <;> exact ⟨rfl, rfl⟩) | cases hs

/-- Once the receive session releases its staging buffer (`hasStaging = false`),
no transfer event (anything other than the environment's `reseatDecodeStaging`)
can modify `decodeStaging` or `decodeHbm`. Proved from `StagingSafe`. -/
theorem decodeStaging_quiet {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hst : s.recv.life.hasStaging = false) (hne : e ≠ .reseatDecodeStaging)
    (hs : step s e = some s') :
    s'.decodeStaging = s.decodeStaging ∧ s'.decodeHbm = s.decodeHbm := by
  have ⟨hpush, _, hret⟩ := (inv_safe h).2.2.2.1.2 hst
  have hr := h.recv.retired_le
  have hy := h.recv.ready_le
  cases e with
  | land l =>
    simp [step, land, hpush] at hs
  | h2dReady l =>
    have hnr : ¬ (s.recv.ready < s.recv.issued) := by omega
    simp [step, h2dReady, Recv.step, Recv.h2dReady, hnr] at hs
  | reseatDecodeStaging => exact absurd rfl hne
  | _ =>
    (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
    first | (obtain ⟨_, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩) | (cases hs <;> exact ⟨rfl, rfl⟩) | cases hs

/-- Once the receive outcome has been published (`done_recving` or
`failed_recving`), no subsequent transition can modify `decodeHbm`. Proved from
`DecodeHbmSafe`: publication implies `retired = issued`, which forces
`ready = issued` and disables `h2dReady`. -/
theorem decodeHbm_quiet {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hp : s.recv.published ≠ none) (hs : step s e = some s') :
    s'.decodeHbm = s.decodeHbm := by
  have ⟨_, hret⟩ := (inv_safe h).2.1 hp
  have hr := h.recv.retired_le
  have hy := h.recv.ready_le
  cases e with
  | h2dReady l =>
    have hnr : ¬ (s.recv.ready < s.recv.issued) := by omega
    simp [step, h2dReady, Recv.step, Recv.h2dReady, hnr] at hs
  | _ =>
    (try (rename_i e; cases e)) <;> (try (rename_i ok; cases ok)) <;> pipe_cases hs <;>
    first | (obtain ⟨_, _, rfl⟩ := hs; rfl) | (cases hs <;> rfl) | cases hs

theorem runFrom_decodeHbm_quiet {n : Nat} :
    ∀ (evs : List Ev) {s s' : Pipeline}, Inv s → s.recv.published ≠ none →
      (sys n).runFrom s evs = some s' → s'.decodeHbm = s.decodeHbm
  | [], _, _, _, _, hr => by simp [System.runFrom] at hr; exact hr ▸ rfl
  | e :: evs, s, s', hinv, hp, hr => by
    simp only [System.runFrom, sys, List.foldlM_cons] at hr
    cases hse : step s e with
    | none => simp [hse] at hr
    | some s₁ =>
      simp only [hse] at hr
      rw [@runFrom_decodeHbm_quiet n evs _ _ (step_inv hinv hse) (step_published_ne_none hse hp) hr]
      exact decodeHbm_quiet hinv hp hse

/-- Once the engine has been told `done_recving`, decode HBM holds the KV
cache from then on: nothing in the rest of the run disturbs it. Proved by
combining `PublicationCorrect` with `runFrom_decodeHbm_quiet` (from
`DecodeHbmSafe`). -/
theorem attention_safe {n : Nat} {s s' : Pipeline} (h : (sys n).Reachable s)
    (hp : s.recv.published = some true) (evs : List Ev)
    (hr : (sys n).runFrom s evs = some s') : s'.decodeHbm = good n := by
  have hinv := reachable_inv h
  rw [runFrom_decodeHbm_quiet evs hinv (by simp [hp]) hr, ← numLayers_eq h]
  exact (inv_safe hinv).1 hp

/-! ### Buffer release predicates, non-interference, and single-request handoff

The predicates `PrefillReleased` and `HandedOff` capture when a request's
prefill buffers or all four buffers have been released, `wroteReleased_eq_false`
and `Inv.recyclePrefillBufs` / `Inv.recycleAllBufs` package the single-request
write-after-release and read-after-release guarantees, and `inv_can_handoff`
packages single-request eventual drain for `MultiRequest.lean`. -/

/-- True if `e` is one of the environment buffer-overwrite events (`.reclaim`,
`.reseatPrefillStaging`, `.reseatDecodeStaging`). -/
def isRecycleEv (e : Ev) : Bool :=
  e == .reclaim || e == .reseatPrefillStaging || e == .reseatDecodeStaging

/-- Per-buffer write-after-release detector: checks whether a transition `r → r'`
modified any of the four buffers after `r` had already released that specific
buffer. -/
def wroteReleased (r r' : Pipeline) : Bool :=
  (r.send.published != none && r'.prefillHbm != r.prefillHbm) ||
  (!r.send.life.hasStaging && r'.prefillStaging != r.prefillStaging) ||
  (!r.recv.life.hasStaging && r'.decodeStaging != r.decodeStaging) ||
  (r.recv.published != none && r'.decodeHbm != r.decodeHbm)

/-- Key per-buffer non-interference lemma: for any request `r` satisfying `Inv r`,
no transfer step ever modifies any of the four buffers after `r` has released
that specific buffer (`wroteReleased r r' = false`). Proved from
`prefillHbm_quiet_step`, `prefillStaging_quiet`, `decodeStaging_quiet`, and
`decodeHbm_quiet`. -/
theorem wroteReleased_eq_false {r r' : Pipeline} {e : Ev} (h : Inv r)
    (hne₀ : e ≠ .reclaim)
    (hne₁ : e ≠ .reseatPrefillStaging) (hne₂ : e ≠ .reseatDecodeStaging)
    (hs : step r e = some r') :
    wroteReleased r r' = false := by
  have hq₀ : r'.prefillHbm = r.prefillHbm := prefillHbm_quiet_step hne₀ hs
  have hq₁ : (!r.send.life.hasStaging && r'.prefillStaging != r.prefillStaging) = false := by
    cases hst : r.send.life.hasStaging
    · simp [(prefillStaging_quiet h hst hne₁ hs).1]
    · simp
  have hq₂ : (!r.recv.life.hasStaging && r'.decodeStaging != r.decodeStaging) = false := by
    cases hst : r.recv.life.hasStaging
    · simp [(decodeStaging_quiet h hst hne₂ hs).1]
    · simp
  have hq₃ : (r.recv.published != none && r'.decodeHbm != r.decodeHbm) = false := by
    cases hp : r.recv.published with
    | none => simp
    | some b =>
      have hne : r.recv.published ≠ none := by simp [hp]
      simp [decodeHbm_quiet h hne hs]
  simp [wroteReleased, hq₀, hq₁, hq₂, hq₃]

theorem not_isRecycleEv_and_wroteReleased_eq_false {r r' : Pipeline} {e : Ev}
    (h : Inv r) (hs : step r e = some r') :
    (!isRecycleEv e && wroteReleased r r') = false := by
  by_cases he : isRecycleEv e = true
  · simp [he]
  · simp only [isRecycleEv, Bool.or_eq_true, beq_iff_eq, not_or] at he
    rw [wroteReleased_eq_false h he.1.1 he.1.2 he.2 hs]
    simp

/-- The prefill side of a request has released both of its buffers: its host
staging buffer has been returned to `StagingBlockAllocator`
(`send.life.hasStaging = false`) and `poll_stats()` has published
`done_sending` or `failed_recving` (`send.published ≠ none`), so a later
request may recycle prefill HBM and prefill staging even while this request's
receive side is still running. -/
def PrefillReleased (s : Pipeline) : Prop :=
  s.send.life.hasStaging = false ∧
  s.send.published ≠ none

instance (s : Pipeline) : Decidable (PrefillReleased s) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- TPU Sync has finished with all four memories of this transfer:
both host staging buffers have been released back to TPU Sync's
`StagingBlockAllocator` (`hasStaging = false`), and both HBM buffers have been
handed back to the caller via `poll_stats()` (`send.published ≠ none`,
`recv.published ≠ none`). -/
def HandedOff (s : Pipeline) : Prop :=
  s.send.life.hasStaging = false ∧
  s.recv.life.hasStaging = false ∧
  s.send.published ≠ none ∧
  s.recv.published ≠ none

instance (s : Pipeline) : Decidable (HandedOff s) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

theorem HandedOff.prefillReleased {s : Pipeline} (h : HandedOff s) : PrefillReleased s :=
  ⟨h.1, h.2.2.1⟩

/-- Overwrite a request's prefill buffers (`prefillHbm` and `prefillStaging`)
with `.junk` when a later request recycles them (`recyclePrefill`). Any
straggling D2H or H2H read by `s` would observe `.junk`. -/
def recyclePrefillBufs (s : Pipeline) : Pipeline :=
  { s with
    prefillHbm := List.replicate s.numLayers .junk,
    reclaimed := true,
    prefillStaging := List.replicate s.numLayers .junk }

/-- Overwrite a request's `prefillHbm`, `prefillStaging`, and `decodeStaging`
with `.junk` when a later request recycles all buffers (`nextRequest`). -/
def recycleAllBufs (s : Pipeline) : Pipeline :=
  { s with
    prefillHbm := List.replicate s.numLayers .junk,
    reclaimed := true,
    prefillStaging := List.replicate s.numLayers .junk,
    decodeStaging := List.replicate s.numLayers .junk }

theorem Inv.recyclePrefillBufs {s : Pipeline} (h : Inv s) (hp : PrefillReleased s) :
    Inv (Pipeline.recyclePrefillBufs s) := by
  have hd : s.send.life.done = true := h.send.life.done_of_released hp.1
  exact h.env_frame _ true _ s.decodeStaging List.length_replicate h.len_dstaging
    nofun (fun _ => hd) (by rw [hd]; nofun) h.dstaging_good

theorem Inv.recycleAllBufs {s : Pipeline} (h : Inv s) (hp : HandedOff s) :
    Inv (Pipeline.recycleAllBufs s) := by
  have hdS : s.send.life.done = true := h.send.life.done_of_released hp.1
  have hdR : s.recv.life.done = true := h.recv.life.done_of_released hp.2.1
  exact h.env_frame _ true _ _ List.length_replicate List.length_replicate
    nofun (fun _ => hdS) (by rw [hdS]; nofun) (by rw [hdR]; nofun)

theorem step_handedOff {s s' : Pipeline} {e : Ev} (h : Inv s)
    (hp : HandedOff s) (hs : step s e = some s') : HandedOff s' := by
  have h' := step_inv h hs
  have hdS : s'.send.life.done = true :=
    step_send_done_mono hs (h.send.life.done_of_released hp.1)
  have hdR : s'.recv.life.done = true :=
    step_recv_done_mono hs (h.recv.life.done_of_released hp.2.1)
  exact ⟨by rw [h'.send.life.staging, hdS]; rfl,
    by rw [h'.recv.life.staging, hdR]; rfl,
    step_send_published_ne_none hs hp.2.2.1,
    step_published_ne_none hs hp.2.2.2⟩

/-- Once a transfer has released both staging buffers and published both HBM
outcomes (`HandedOff`), no subsequent transfer transition can ever modify any
memory. Proved by combining `prefillHbm_quiet_step`, `prefillStaging_quiet`,
`decodeStaging_quiet`, and `decodeHbm_quiet`. -/
theorem handedOff_quiet {r r' : Pipeline} {e : Ev} (h : Inv r) (hp : HandedOff r)
    (hne₀ : e ≠ .reclaim)
    (hne₁ : e ≠ .reseatPrefillStaging) (hne₂ : e ≠ .reseatDecodeStaging)
    (hs : step r e = some r') :
    r'.prefillHbm = r.prefillHbm ∧
    r'.prefillStaging = r.prefillStaging ∧
    r'.wire = r.wire ∧
    r'.decodeStaging = r.decodeStaging ∧
    r'.decodeHbm = r.decodeHbm :=
  ⟨prefillHbm_quiet_step hne₀ hs,
   (prefillStaging_quiet h hp.1 hne₁ hs).1,
   (prefillStaging_quiet h hp.1 hne₁ hs).2,
   (decodeStaging_quiet h hp.2.1 hne₂ hs).1,
   decodeHbm_quiet h hp.2.2.2 hs⟩

/-- From any pipeline state satisfying `Inv`, a finite trace settles both
sessions, releases both host staging buffers to `StagingBlockAllocator`, and
publishes both HBM outcomes (`HandedOff`). -/
theorem inv_can_handoff (n : Nat) {s : Pipeline} (hinv : Inv s) :
    ∃ evs s', (sys n).runFrom s evs = some s' ∧ HandedOff s' := by
  obtain ⟨evs, s', hr, _, hstS, _, hstR, hpubS, hpubR, _⟩ := inv_can_settle n hinv
  exact ⟨evs, s', hr, hstS, hstR, hpubS, hpubR⟩


end Pipeline

end TpuSyncVerify.Transfer.PrefillDecode

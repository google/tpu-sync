import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Lemmas

/-! # `Pipeline`: inductive invariant, preservation and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.Pipeline`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ## Inductive invariant

`Inv` combines the session invariants (`Send.Inv` and `Recv.Inv`, which enforce
the settle gate and `in_flight_` conservation) with the per-layer pipeline
invariants:
- **Per-layer stage ordering & data validity:** `phbm_good`, `reclaimed_done`,
  `d2hReady_lt`, `woken_d2hReady`, `pstaging_good`, `wire_good`,
  `pending_landed`, `issued_landed`, `pending_claimed`, `issued_claimed`,
  `dstaging_good`, and `dhbm_good`.
- **At-most-once layer counting:** `cnt_d2hReady`, `cnt_h2hRetired`,
  `cnt_h2dPending`, `cnt_h2dIssued`, and `cnt_h2dReady` equate each session's
  integer counter to the number of `true` entries in the corresponding
  per-layer boolean checklist.
- **Handshake rendezvous:** `pullStarted_registered` and `pullWaiting_state`. -/

structure Inv (s : Pipeline) : Prop where
  send : s.send.Inv
  recv : s.recv.Inv
  n_send : s.send.numLayers = s.numLayers
  n_recv : s.recv.numLayers = s.numLayers
  len_pstaging : s.prefillStaging.length = s.numLayers
  len_wire : s.wire.length = s.numLayers
  len_dstaging : s.decodeStaging.length = s.numLayers
  len_dhbm : s.decodeHbm.length = s.numLayers
  len_d2hReadyL : s.d2hReadyL.length = s.numLayers
  len_h2hRetiredL : s.h2hRetiredL.length = s.numLayers
  len_claimedL : s.claimedL.length = s.numLayers
  len_h2dPendingL : s.h2dPendingL.length = s.numLayers
  len_h2dIssuedL : s.h2dIssuedL.length = s.numLayers
  len_h2dReadyL : s.h2dReadyL.length = s.numLayers
  /-- Prefill HBM holds the data until the engine reclaims it. -/
  phbm_good : s.reclaimed = false → s.prefillHbm = good s.numLayers
  /-- Prefill HBM is reclaimed only once the send has settled. -/
  reclaimed_done : s.reclaimed = true → s.send.life.done = true
  /-- The per-layer record of finished D2H copies agrees with the send's counter. -/
  cnt_d2hReady : countTrue s.d2hReadyL = s.send.d2hReady
  /-- Only issued D2H copies have finished. -/
  d2hReady_lt : ∀ l : Nat, s.d2hReadyL[l]? = some true → l < s.send.d2hIssued
  /-- The per-layer record of retired H2H pushes agrees with the send's counter. -/
  cnt_h2hRetired : countTrue s.h2hRetiredL = s.send.h2hRetired
  /-- `SendNextLayer(l)`'s callback has run for an issued layer only after that
  layer's copy finished (unissued layers are resolved with an error by
  `fail_unissued_layers` once the D2H loop has exited). -/
  woken_d2hReady : ∀ l : Nat, l < s.send.woken → l < s.send.d2hIssued → s.d2hReadyL[l]? = some true
  /-- While the send holds its staging, every finished copy's layer is there. -/
  pstaging_good : s.send.life.done = false →
    ∀ l : Nat, s.d2hReadyL[l]? = some true → s.prefillStaging[l]? = some (.kv l)
  /-- Whatever was delivered for layer `k` is layer `k`. -/
  wire_good : ∀ (k : Nat) (c : Cell), s.wire[k]? = some c → c = .blank ∨ c = .kv k
  /-- A layer is pending in `ExecuteLayerH2d` only after it landed. -/
  pending_landed : ∀ l : Nat, s.h2dPendingL[l]? = some true → s.landedL[l]? = some true
  /-- A layer's H2D copy is issued to the device only after it landed. -/
  issued_landed : ∀ l : Nat, s.h2dIssuedL[l]? = some true → s.landedL[l]? = some true
  /-- A pending layer was claimed. -/
  pending_claimed : ∀ l : Nat, s.h2dPendingL[l]? = some true → s.claimedL[l]? = some true
  /-- An issued layer was claimed and is no longer pending. -/
  issued_claimed : ∀ l : Nat, s.h2dIssuedL[l]? = some true →
    s.claimedL[l]? = some true ∧ s.h2dPendingL[l]? = some false
  /-- The per-layer record of pending H2D dispatches agrees with the receive's counter. -/
  cnt_h2dPending : countTrue s.h2dPendingL = s.recv.pending
  /-- The per-layer record of issued H2D copies agrees with the receive's counter. -/
  cnt_h2dIssued : countTrue s.h2dIssuedL = s.recv.issued
  /-- While the receive holds its staging, every landed layer is there. -/
  dstaging_good : s.recv.life.done = false →
    ∀ l : Nat, s.landedL[l]? = some true → s.decodeStaging[l]? = some (.kv l)
  /-- The per-layer record of finished H2D copies agrees with the receive's counter. -/
  cnt_h2dReady : countTrue s.h2dReadyL = s.recv.ready
  /-- Every finished copy put its layer in decode HBM. -/
  dhbm_good : ∀ l : Nat, s.h2dReadyL[l]? = some true → s.decodeHbm[l]? = some (.kv l)
  /-- `ValidateAndBeginPull` can only claim the send session once `NotifyForRead`
  has registered the offer in `send_sessions_[uuid]`. -/
  pullStarted_registered : s.send.pullStarted = true → s.registered = true
  /-- While `HandlePullStream` is in its grace wait (`cv_.WaitWithTimeout`),
  the consumer's pull RPC is still pending and `pull_started_` is still false. -/
  pullWaiting_state : s.pullWaiting = true →
    s.recv.pullPending = true ∧ s.send.pullStarted = false

theorem inv_init (n : Nat) : Inv (init n) := by
  refine ⟨Send.inv_init n, Recv.inv_initLoad n, rfl, rfl, by simp [init], by simp [init],
    by simp [init], by simp [init], by simp [init], by simp [init], by simp [init],
    by simp [init], by simp [init], by simp [init], fun _ => rfl, by simp [init, Send.init],
    by simp [init, Send.init, countTrue_replicate_false],
    fun l h => (not_mem_replicate_false h).elim,
    by simp [init, Send.init, countTrue_replicate_false],
    by simp [init, Send.init],
    fun _ l h => (not_mem_replicate_false h).elim, ?_,
    fun l h => (not_mem_replicate_false h).elim,
    fun l h => (not_mem_replicate_false h).elim,
    fun l h => (not_mem_replicate_false h).elim,
    fun l h => (not_mem_replicate_false h).elim,
    by simp [init, Recv.initLoad, countTrue_replicate_false],
    by simp [init, Recv.initLoad, countTrue_replicate_false],
    fun _ l h => (not_mem_replicate_false h).elim,
    by simp [init, Recv.initLoad, countTrue_replicate_false],
    fun l h => (not_mem_replicate_false h).elim,
    fun _ => rfl, nofun⟩
  intro k c hk
  simp only [init, List.getElem?_replicate] at hk
  split at hk
  · cases hk; exact Or.inl rfl
  · cases hk

theorem inv_initUnregistered (n : Nat) : Inv (initUnregistered n) := by
  have h := inv_init n
  refine ⟨h.send, h.recv, h.n_send, h.n_recv, h.len_pstaging, h.len_wire, h.len_dstaging,
    h.len_dhbm, h.len_d2hReadyL, h.len_h2hRetiredL, h.len_claimedL, h.len_h2dPendingL,
    h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good, h.reclaimed_done, h.cnt_d2hReady,
    h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady, h.pstaging_good, h.wire_good,
    h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
    h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, h.cnt_h2dReady, h.dhbm_good,
    nofun, nofun⟩

theorem mem_drainEvents_d2hReady {n l : Nat} (hl : l < n) :
    Ev.d2hReady l ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  right; exact ⟨l, hl, by simp⟩

theorem mem_drainEvents_h2hDone {n l : Nat} (hl : l < n) :
    Ev.h2hDone l true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  right; exact ⟨l, hl, by simp⟩

theorem mem_drainEvents_h2dIssue {n l : Nat} (hl : l < n) :
    Ev.h2dIssue l true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  right; exact ⟨l, hl, by simp⟩

theorem mem_drainEvents_h2dReady {n l : Nat} (hl : l < n) :
    Ev.h2dReady l ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]
  right; exact ⟨l, hl, by simp⟩

theorem inv_noOpLeak {s : Pipeline} (h : Inv s) : NoOpLeak s := by
  rintro (hif | hif)
  · have hacc := h.send.accounted
    have hcnt := h.send.counters
    unfold Send.Accounted at hacc
    unfold Send.CountersOrdered at hcnt
    by_cases hp : s.send.d2hPending = true
    · refine ⟨.send (.d2hIssue true), by simp [drainEvents], ?_⟩
      simp [step, sendStep, Send.step, Send.d2hIssue, hp]
    · have hp' : s.send.d2hPending = false := by
        cases hq : s.send.d2hPending
        · rfl
        · exact absurd hq hp
      by_cases hrd : s.send.d2hReady < s.send.d2hIssued
      · have hk : s.send.d2hIssued ≤ s.d2hReadyL.length := by
          rw [h.len_d2hReadyL, ← h.n_send]; omega
        obtain ⟨l, hl, hf⟩ := exists_false_lt hk (by rw [h.cnt_d2hReady]; exact hrd)
        have hln : l < s.numLayers := by rw [← h.n_send]; omega
        refine ⟨.d2hReady l, mem_drainEvents_d2hReady hln, ?_⟩
        simp [step, d2hReady, hl, hf, Send.step, Send.markD2hReady, hrd]
      · by_cases hret : s.send.d2hRetired < s.send.d2hReady
        · refine ⟨.send .d2hEnd, by simp [drainEvents], ?_⟩
          simp [step, sendStep, Send.step, Send.d2hEnd, hret]
        · by_cases hwk : s.send.woken < s.send.queued
          · by_cases hwi : s.send.woken < s.send.d2hIssued
            · have hwr : s.send.woken < s.send.d2hReady := by omega
              have hready_eq : s.send.d2hIssued ≤ countTrue s.d2hReadyL := by
                rw [h.cnt_d2hReady]; omega
              have hwl : s.d2hReadyL[s.send.woken]? = some true :=
                all_true_lt_of_countTrue h.d2hReady_lt hready_eq s.send.woken hwi
              refine ⟨.send (.wake true), by simp [drainEvents], ?_⟩
              simp only [step, sendStep, hwl, true_or, ↓reduceIte, Send.step, Send.wake, hwk, hwr,
                and_self, Bool.true_eq_false]
              split <;> rfl
            · have hle : s.send.d2hIssued ≤ s.send.woken := by omega
              cases hdr : s.send.life.draining
              · have hst : s.send.started = true := h.send.queued_started (by omega)
                have hnd : s.send.life.done = false := by
                  cases hd : s.send.life.done
                  · rfl
                  · have := h.send.life.done_draining hd; rw [hdr] at this; cases this
                have hlt : s.send.d2hIssued < s.send.numLayers := by omega
                refine ⟨.send .d2hBegin, by simp [drainEvents], ?_⟩
                simp [step, sendStep, Send.step, Send.d2hBegin, hst, hp', hlt, Lifecycle.beginOp, hdr, hnd]
              · refine ⟨.send (.wake false), by simp [drainEvents], ?_⟩
                simp [step, sendStep, Send.step, Send.wake, hwk, hdr, hp', hle]
          · by_cases hpool : 0 < s.send.pooled
            · have hne : s.send.pooled ≠ 0 := by omega
              refine ⟨.send .h2hIssue, by simp [drainEvents], ?_⟩
              simp only [step, sendStep, Send.step, Send.h2hIssue, hne, ↓reduceIte]
              split <;> rfl
            · by_cases hch : 0 < s.send.chaining
              · refine ⟨.send .sendNext, by simp [drainEvents], ?_⟩
                simp [step, sendStep, Send.step, Send.sendNext]; omega
              · have hh2h : s.send.h2hRetired < s.send.h2hIssued := by
                  cases hdp : s.send.d2hPending <;> simp_all <;> omega
                have hk : s.send.h2hIssued ≤ s.h2hRetiredL.length := by
                  rw [h.len_h2hRetiredL, ← h.n_send]; omega
                obtain ⟨l, hl, hf⟩ := exists_false_lt hk (by rw [h.cnt_h2hRetired]; exact hh2h)
                have hln : l < s.numLayers := by rw [← h.n_send]; omega
                refine ⟨.h2hDone l true, mem_drainEvents_h2hDone hln, ?_⟩
                simp [step, h2hDone, hl, hf, Send.step, Send.h2hDone, hh2h]
  · have hacc := h.recv.accounted
    unfold Recv.Accounted at hacc
    have hr := h.recv.retired_le
    have hy := h.recv.ready_le
    by_cases hp : 0 < s.recv.pushes
    · refine ⟨.recv .pushEnd, by simp [drainEvents], ?_⟩
      simp [step, recvStep, Recv.step, Recv.pushEnd]; omega
    · by_cases hq : s.recv.pullPending = true
      · refine ⟨.recv (.pullReply false), by simp [drainEvents], ?_⟩
        simp [step, recvStep, Recv.step, Recv.pullReply, hq]
      · by_cases hpend : 0 < s.recv.pending
        · have hne : s.recv.pending ≠ 0 := by omega
          obtain ⟨l, hl, hpl⟩ := exists_true_of_countTrue_pos (by rw [h.cnt_h2dPending]; exact hpend)
          rw [h.len_h2dPendingL] at hl
          refine ⟨.h2dIssue l true, mem_drainEvents_h2dIssue hl, ?_⟩
          simp only [step, h2dIssue, hpl, ↓reduceIte, Recv.step, Recv.h2dIssue, hne]
          (repeat' split) <;> rfl
        · by_cases hrd : s.recv.ready < s.recv.issued
          · have hlen : s.h2dReadyL.length = s.h2dIssuedL.length := by
              rw [h.len_h2dReadyL, h.len_h2dIssuedL]
            have hcnt : countTrue s.h2dReadyL < countTrue s.h2dIssuedL := by
              rw [h.cnt_h2dReady, h.cnt_h2dIssued]; exact hrd
            obtain ⟨l, hl, hrf, hit⟩ := exists_diff_of_countTrue_lt hlen hcnt
            rw [h.len_h2dReadyL] at hl
            refine ⟨.h2dReady l, mem_drainEvents_h2dReady hl, ?_⟩
            simp [step, h2dReady, hit, hrf, Recv.step, Recv.h2dReady, hrd]
          · have hret : s.recv.retired < s.recv.ready := by
              cases hqp : s.recv.pullPending <;> simp_all <;> omega
            refine ⟨.recv (.h2dDone true), by simp [drainEvents], ?_⟩
            simp only [step, recvStep, Recv.step, Recv.h2dDone, hret, ↓reduceIte]
            split <;> rfl

theorem inv_handshakeSafe {s : Pipeline} (h : Inv s) : HandshakeSafe s := by
  have hcnt := h.send.counters
  have ⟨hpc₁, hpc₂⟩ := h.send.pull_claimed
  unfold Send.CountersOrdered at hcnt
  refine ⟨fun hs => ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩,
    h.pullStarted_registered, ?_, h.pullWaiting_state⟩
  rintro (hdp | hd2h | hh2h)
  · have hs := hpc₂ (Or.inl hdp)
    exact ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩
  · have hs := hpc₂ (Or.inr hd2h)
    exact ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩
  · have hs := hpc₂ (Or.inr (by omega))
    exact ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩

theorem inv_safe {s : Pipeline} (h : Inv s) : Safe s := by
  have hsend := h.send
  have hrecv := h.recv
  have hsS := Send.inv_safe hsend
  have hcnt := hsend.counters
  unfold Send.CountersOrdered at hcnt
  refine ⟨?_, ?_, ?_, ⟨?_, ?_⟩, inv_handshakeSafe h, inv_noOpLeak h⟩
  · intro hp
    apply eq_good h.len_dhbm
    intro k hk
    apply h.dhbm_good
    have hready : s.recv.ready = s.numLayers := by
      have := hrecv.published_ok hp
      have := hrecv.completed_le
      have := hrecv.retired_le
      have := hrecv.ready_le
      have := hrecv.issued_le
      have := h.n_recv
      omega
    apply all_true_of_countTrue_eq_length
    · rw [h.cnt_h2dReady, hready, h.len_h2dReadyL]
    · rw [h.len_h2dReadyL]; exact hk
  · intro hp
    cases hpub : s.recv.published with
    | none => exact (hp hpub).elim
    | some b =>
      have hd := hrecv.published_done b hpub
      have h0 := hrecv.life.done_idle hd
      have hacc := hrecv.accounted
      unfold Recv.Accounted at hacc
      have := hrecv.retired_le
      have := hrecv.ready_le
      exact ⟨by omega, by omega⟩
  · intro hp
    cases hpub : s.send.published with
    | none => exact (hp hpub).elim
    | some b =>
      obtain ⟨h0, h1, -⟩ := hsS.2.1 (hsend.published_done b hpub)
      exact ⟨h0, h1⟩
  · intro hst
    obtain ⟨h0, h1, -, -, -, h2⟩ := hsS.2.1 (hsend.life.done_of_released hst)
    exact ⟨h0, h1, h2⟩
  · intro hst
    have hd := hrecv.life.done_of_released hst
    have h0 := hrecv.life.done_idle hd
    have hacc := hrecv.accounted
    unfold Recv.Accounted at hacc
    have := hrecv.retired_le
    have := hrecv.ready_le
    refine ⟨by omega, by omega, by omega⟩

/-! ### Preservation, one lemma per kind of effect -/

/-- A send event that updates `send`, `h2hRetiredL`, and `wire`, given that the
issued layers `SendNextLayer` has consumed afterwards all have their copy finished. -/
theorem Inv.send_wire_frame {s : Pipeline} {snd : Send} {e : Send.Ev} (h : Inv s)
    (hs : Send.step s.send e = some snd) (he : e ≠ .d2hReady) (he_bp : e ≠ .beginPull)
    (hw : ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true) (x : List Bool)
    (hxlen : x.length = s.numLayers) (hxcnt : countTrue x = snd.h2hRetired)
    (w : List Cell) (hwlen : w.length = s.numLayers)
    (hwgood : ∀ (k : Nat) (c : Cell), w[k]? = some c → c = .blank ∨ c = .kv k) :
    Inv { s with send := snd, h2hRetiredL := x, wire := w } := by
  refine ⟨Send.step_inv h.send hs, h.recv, (Send.step_numLayers hs).trans h.n_send, h.n_recv,
    h.len_pstaging, hwlen, h.len_dstaging, h.len_dhbm, h.len_d2hReadyL, hxlen,
    h.len_claimedL, h.len_h2dPendingL, h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good,
    fun hr => Send.step_done_mono hs (h.reclaimed_done hr), ?_, ?_, hxcnt, hw, ?_, hwgood,
    h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
    h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, h.cnt_h2dReady, h.dhbm_good,
    by rw [Send.step_pullStarted_of_ne_beginPull hs he_bp]; exact h.pullStarted_registered,
    by rw [Send.step_pullStarted_of_ne_beginPull hs he_bp]; exact h.pullWaiting_state⟩
  · show countTrue s.d2hReadyL = snd.d2hReady
    rw [Send.step_d2hReady hs he]; exact h.cnt_d2hReady
  · show ∀ l : Nat, s.d2hReadyL[l]? = some true → l < snd.d2hIssued
    intro l hl
    exact Nat.lt_of_lt_of_le (h.d2hReady_lt l hl) (Send.step_d2hIssued_le hs)
  · show snd.life.done = false → ∀ l : Nat, s.d2hReadyL[l]? = some true → s.prefillStaging[l]? = some (.kv l)
    intro hd
    apply h.pstaging_good
    cases hd0 : s.send.life.done
    · rfl
    · rw [Send.step_done_mono hs hd0] at hd; cases hd

/-- A send event that moves no memory. -/
theorem Inv.send_frame {s : Pipeline} {snd : Send} {e : Send.Ev} (h : Inv s)
    (hs : Send.step s.send e = some snd) (he : e ≠ .d2hReady) (he_bp : e ≠ .beginPull)
    (hw : ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true) (x : List Bool)
    (hxlen : x.length = s.numLayers) (hxcnt : countTrue x = snd.h2hRetired) :
    Inv { s with send := snd, h2hRetiredL := x } :=
  h.send_wire_frame hs he he_bp hw x hxlen hxcnt s.wire h.len_wire h.wire_good

/-- A send that still has a copy or push outstanding has not settled. -/
theorem send_not_done_of_outstanding {t : Send} (h : t.Inv)
    (ho : t.d2hReady < t.d2hIssued ∨ t.h2hRetired < t.h2hIssued) : t.life.done = false := by
  cases hd : t.life.done
  · rfl
  · have hdr := (Send.inv_safe h).2.1 hd
    have hcnt := h.counters
    unfold Send.CountersOrdered at hcnt
    omega

/-- The D2H copy of layer `l` lands in staging. -/
theorem Inv.send_d2hReady {s : Pipeline} {snd : Send} {l : Nat} (h : Inv s)
    (hl : l < s.send.d2hIssued) (hf : s.d2hReadyL[l]? = some false)
    (hs : Send.step s.send .d2hReady = some snd) :
    Inv { s with send := snd,
                 prefillStaging := s.prefillStaging.set l (s.prefillHbm.getD l .junk),
                 d2hReadyL := s.d2hReadyL.set l true } := by
  obtain ⟨hlt, rfl⟩ := Send.d2hReady_spec hs
  have hcnt := h.send.counters
  unfold Send.CountersOrdered at hcnt
  have hd : s.send.life.done = false := send_not_done_of_outstanding h.send (Or.inl hlt)
  have hnr : s.reclaimed = false := by
    cases hr : s.reclaimed
    · rfl
    · have := h.reclaimed_done hr; rw [hd] at this; cases this
  have hk : l < s.numLayers := by have := h.n_send; omega
  have hsrc : s.prefillHbm.getD l .junk = .kv l := by
    rw [h.phbm_good hnr]; exact good_getD hk
  rw [hsrc]
  refine ⟨Send.step_inv h.send hs, h.recv, (Send.step_numLayers hs).trans h.n_send, h.n_recv,
    ?_, h.len_wire, h.len_dstaging, h.len_dhbm, ?_, h.len_h2hRetiredL,
    h.len_claimedL, h.len_h2dPendingL, h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good,
    fun hr => Send.step_done_mono hs (h.reclaimed_done hr), ?_, ?_, h.cnt_h2hRetired, ?_, ?_, h.wire_good,
    h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
    h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, h.cnt_h2dReady, h.dhbm_good,
    h.pullStarted_registered, h.pullWaiting_state⟩
  · show (s.prefillStaging.set _ _).length = _
    rw [List.length_set]; exact h.len_pstaging
  · show (s.d2hReadyL.set _ _).length = _
    rw [List.length_set]; exact h.len_d2hReadyL
  · show countTrue (s.d2hReadyL.set l true) = s.send.d2hReady + 1
    rw [countTrue_set_true hf, h.cnt_d2hReady]
  · show ∀ j : Nat, (s.d2hReadyL.set l true)[j]? = some true → j < s.send.d2hIssued
    intro j hj
    rcases mem_of_set_true hj with rfl | hj
    · exact hl
    · exact h.d2hReady_lt j hj
  · show ∀ j < s.send.woken, j < s.send.d2hIssued → (s.d2hReadyL.set l true)[j]? = some true
    intro j hj hji
    exact set_true_of_mem hf (h.woken_d2hReady j hj hji)
  · show _ → ∀ j : Nat, (s.d2hReadyL.set l true)[j]? = some true →
      (s.prefillStaging.set l (.kv l))[j]? = some (.kv j)
    intro _ j hj
    apply getElem?_set_kv (by rw [h.len_pstaging]; exact hk)
    intro hjl
    exact h.pstaging_good hd j ((mem_of_set_true hj).resolve_left hjl)

/-- A successful push of layer `l` delivers it. -/
theorem Inv.send_h2hDone {s : Pipeline} {snd : Send} {l : Nat} (h : Inv s)
    (hl : l < s.send.h2hIssued) (hf : s.h2hRetiredL[l]? = some false)
    (hs : Send.step s.send (.h2hDone true) = some snd)
    (hw : ∀ l' < snd.woken, l' < snd.d2hIssued → s.d2hReadyL[l']? = some true) :
    Inv { s with send := snd, h2hRetiredL := s.h2hRetiredL.set l true,
                 wire := s.wire.set l (s.prefillStaging.getD l .junk) } := by
  have hlt := Send.h2hDone_guard hs
  have hcnt := h.send.counters
  unfold Send.CountersOrdered at hcnt
  have hd : s.send.life.done = false := send_not_done_of_outstanding h.send (Or.inr hlt)
  have hk : l < s.numLayers := by have := h.n_send; omega
  have hdl : s.d2hReadyL[l]? = some true := h.woken_d2hReady l (by omega) (by omega)
  have hsrc : s.prefillStaging.getD l .junk = .kv l := by
    rw [List.getD_eq_getElem?_getD, h.pstaging_good hd l hdl]; rfl
  rw [hsrc]
  apply h.send_wire_frame hs nofun nofun hw (s.h2hRetiredL.set l true)
    (by rw [List.length_set]; exact h.len_h2hRetiredL)
    (by rw [countTrue_set_true hf, h.cnt_h2hRetired, Send.h2hDone_h2hRetired hs])
    (s.wire.set l (.kv l)) (by rw [List.length_set]; exact h.len_wire)
  intro j c hj
  by_cases hjl : j = l
  · subst hjl
    rw [List.getElem?_set_self (by rw [h.len_wire]; exact hk)] at hj
    cases hj; exact Or.inr rfl
  · rw [List.getElem?_set_ne (Ne.symm hjl)] at hj
    exact h.wire_good j c hj

/-- A receive event that moves no memory. `c`, `p` and `i` let the same lemma
serve `h2dBegin l` and `h2dIssue l ok`, which only update the per-layer dispatch
sets. -/
theorem Inv.recv_frame {s : Pipeline} {rcv : Recv} {e : Recv.Ev} (h : Inv s)
    (hs : Recv.step s.recv e = some rcv) (he : e ≠ .h2dReady)
    (he_pr : ∀ ok, e ≠ .pullReply ok) (c p i : List Bool)
    (hclen : c.length = s.numLayers) (hplen : p.length = s.numLayers)
    (hilen : i.length = s.numLayers)
    (hp : ∀ l : Nat, p[l]? = some true → s.landedL[l]? = some true)
    (hi : ∀ l : Nat, i[l]? = some true → s.landedL[l]? = some true)
    (hpc : ∀ l : Nat, p[l]? = some true → c[l]? = some true)
    (hic : ∀ l : Nat, i[l]? = some true → c[l]? = some true ∧ p[l]? = some false)
    (hpcnt : countTrue p = rcv.pending)
    (hicnt : countTrue i = rcv.issued) :
    Inv { s with recv := rcv, claimedL := c, h2dPendingL := p, h2dIssuedL := i } := by
  refine ⟨h.send, Recv.step_inv h.recv hs, h.n_send, (Recv.step_numLayers hs).trans h.n_recv,
    h.len_pstaging, h.len_wire, h.len_dstaging, h.len_dhbm, h.len_d2hReadyL, h.len_h2hRetiredL,
    hclen, hplen, hilen, h.len_h2dReadyL, h.phbm_good, h.reclaimed_done, h.cnt_d2hReady,
    h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady, h.pstaging_good, h.wire_good, hp, hi,
    hpc, hic, hpcnt, hicnt, ?_, ?_, h.dhbm_good, h.pullStarted_registered,
    by rw [Recv.step_pullPending_of_ne_pullReply hs he_pr]; exact h.pullWaiting_state⟩
  · show rcv.life.done = false → ∀ l : Nat, s.landedL[l]? = some true → s.decodeStaging[l]? = some (.kv l)
    intro hd
    apply h.dstaging_good
    cases hd0 : s.recv.life.done
    · rfl
    · rw [Recv.step_done_mono hs hd0] at hd; cases hd
  · show countTrue s.h2dReadyL = rcv.ready
    rw [Recv.step_ready hs he]; exact h.cnt_h2dReady

/-- The H2D copy of layer `l` lands in decode HBM. -/
theorem Inv.recv_h2dReady {s : Pipeline} {rcv : Recv} {l : Nat} (h : Inv s)
    (hc : s.h2dIssuedL[l]? = some true) (hf : s.h2dReadyL[l]? = some false)
    (hs : Recv.step s.recv .h2dReady = some rcv) :
    Inv { s with recv := rcv,
                 decodeHbm := s.decodeHbm.set l (s.decodeStaging.getD l .junk),
                 h2dReadyL := s.h2dReadyL.set l true } := by
  obtain ⟨hlt, rfl⟩ := Recv.h2dReady_spec hs
  have hr := h.recv
  have hd : s.recv.life.done = false := by
    cases hd : s.recv.life.done
    · rfl
    · have := (Recv.inv_safe hr).2.1 hd
      have := hr.retired_le
      omega
  have hk : l < s.numLayers := by
    have := lt_length_of_getElem?_eq hf; rw [h.len_h2dReadyL] at this; exact this
  have hsrc : s.decodeStaging.getD l .junk = .kv l := by
    rw [List.getD_eq_getElem?_getD, h.dstaging_good hd l (h.issued_landed l hc)]
    rfl
  rw [hsrc]
  refine ⟨h.send, Recv.step_inv hr hs, h.n_send, (Recv.step_numLayers hs).trans h.n_recv,
    h.len_pstaging, h.len_wire, h.len_dstaging, ?_, h.len_d2hReadyL, h.len_h2hRetiredL,
    h.len_claimedL, h.len_h2dPendingL, h.len_h2dIssuedL, ?_, h.phbm_good, h.reclaimed_done,
    h.cnt_d2hReady, h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady, h.pstaging_good,
    h.wire_good, h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
    h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, ?_, ?_,
    h.pullStarted_registered, h.pullWaiting_state⟩
  · show (s.decodeHbm.set _ _).length = _
    rw [List.length_set]; exact h.len_dhbm
  · show (s.h2dReadyL.set _ _).length = _
    rw [List.length_set]; exact h.len_h2dReadyL
  · show countTrue (s.h2dReadyL.set l true) = s.recv.ready + 1
    rw [countTrue_set_true hf, h.cnt_h2dReady]
  · show ∀ j : Nat, (s.h2dReadyL.set l true)[j]? = some true → (s.decodeHbm.set l (.kv l))[j]? = some (.kv j)
    intro j hj
    apply getElem?_set_kv (by rw [h.len_dhbm]; exact hk)
    intro hjl
    exact h.dhbm_good j ((mem_of_set_true hj).resolve_left hjl)

/-- Landing a delivered layer `l` into decode staging. -/
theorem Inv.land_layer {s : Pipeline} {l : Nat} {c : Cell} (h : Inv s)
    (hf : s.landedL[l]? = some false) (hc : s.wire[l]? = some c) (hcb : c ≠ .blank) :
    Inv { s with decodeStaging := s.decodeStaging.set l c,
                 landedL := s.landedL.set l true } := by
  have hkv : c = .kv l := (h.wire_good _ _ hc).resolve_left hcb
  have hlen : l < s.numLayers := by
    have := lt_length_of_getElem?_eq hc
    rw [h.len_wire] at this; exact this
  subst hkv
  refine ⟨h.send, h.recv, h.n_send, h.n_recv, h.len_pstaging, h.len_wire, ?_, h.len_dhbm,
    h.len_d2hReadyL, h.len_h2hRetiredL, h.len_claimedL, h.len_h2dPendingL,
    h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good, h.reclaimed_done, h.cnt_d2hReady,
    h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady, h.pstaging_good, h.wire_good,
    ?_, ?_, h.pending_claimed, h.issued_claimed, h.cnt_h2dPending, h.cnt_h2dIssued,
    ?_, h.cnt_h2dReady, h.dhbm_good, h.pullStarted_registered, h.pullWaiting_state⟩
  · rw [List.length_set]; exact h.len_dstaging
  · exact fun j hj => set_true_of_mem hf (h.pending_landed j hj)
  · exact fun j hj => set_true_of_mem hf (h.issued_landed j hj)
  · intro hd j hj
    apply getElem?_set_kv (by rw [h.len_dstaging]; exact hlen)
    intro hjl
    exact h.dstaging_good hd j ((mem_of_set_true hj).resolve_left hjl)

/-- Frame lemma for the environment's buffer-reclaim and staging-reseat
operations (`reclaim`, `reseatPrefillStaging`, `reseatDecodeStaging`,
`recyclePrefillBufs`, `recycleAllBufs`). -/
theorem Inv.env_frame {s : Pipeline} (h : Inv s)
    (phbm : List Cell) (rec : Bool) (pstg dstg : List Cell)
    (hp_len : pstg.length = s.numLayers) (hd_len : dstg.length = s.numLayers)
    (hphbm : rec = false → phbm = good s.numLayers)
    (hrec : rec = true → s.send.life.done = true)
    (hpstg : s.send.life.done = false →
      ∀ l : Nat, s.d2hReadyL[l]? = some true → pstg[l]? = some (.kv l))
    (hdstg : s.recv.life.done = false →
      ∀ l : Nat, s.landedL[l]? = some true → dstg[l]? = some (.kv l)) :
    Inv { s with prefillHbm := phbm, reclaimed := rec,
                 prefillStaging := pstg, decodeStaging := dstg } :=
  ⟨h.send, h.recv, h.n_send, h.n_recv, hp_len, h.len_wire, hd_len, h.len_dhbm,
   h.len_d2hReadyL, h.len_h2hRetiredL, h.len_claimedL, h.len_h2dPendingL,
   h.len_h2dIssuedL, h.len_h2dReadyL, hphbm, hrec, h.cnt_d2hReady, h.d2hReady_lt,
   h.cnt_h2hRetired, h.woken_d2hReady, hpstg, h.wire_good, h.pending_landed,
   h.issued_landed, h.pending_claimed, h.issued_claimed, h.cnt_h2dPending,
   h.cnt_h2dIssued, hdstg, h.cnt_h2dReady, h.dhbm_good,
   h.pullStarted_registered, h.pullWaiting_state⟩

/-- Issued layers `SendNextLayer` has consumed after a `wake`: the ones before,
plus layer `woken` when it was issued (an unissued layer has `d2hIssued ≤ woken`). -/
theorem woken_after_wake {s : Pipeline} {snd : Send} {ok : Bool} (h : Inv s)
    (hs : Send.step s.send (.wake ok) = some snd)
    (hg : s.d2hReadyL[s.send.woken]? = some true ∨
      (ok = false ∧ s.send.life.draining = true ∧
        s.send.d2hPending = false ∧ s.send.d2hIssued ≤ s.send.woken)) :
    ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true := by
  rw [Send.wake_woken hs, Send.step_d2hIssued_of_ne_d2hIssue hs nofun]
  intro l hl hli
  rcases Nat.lt_succ_iff_lt_or_eq.mp hl with hl | rfl
  · exact h.woken_d2hReady l hl hli
  · rcases hg with hd2h | ⟨_, _, _, hle⟩
    · exact hd2h
    · omega

/-- Issued layers `SendNextLayer` has consumed after any other send event: unchanged
(`d2hIssue` only runs with `d2hPending = true`, when `woken ≤ d2hIssued`). -/
theorem woken_after_other {s : Pipeline} {snd : Send} {e : Send.Ev} (h : Inv s)
    (hs : Send.step s.send e = some snd) (he : ∀ ok, e ≠ .wake ok) :
    ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true := by
  rw [Send.step_woken hs he]
  intro l hl hli
  match e with
  | .d2hIssue ok =>
    obtain ⟨hp, _⟩ := Send.d2hIssue_spec hs
    have hwle : s.send.woken ≤ s.send.d2hIssued := by
      refine Nat.le_of_not_lt fun hlt => ?_
      have := (h.send.woken_issued hlt).2
      rw [hp] at this; cases this
    exact h.woken_d2hReady l hl (Nat.lt_of_lt_of_le hl hwle)
  | .beginPull | .start | .d2hBegin | .d2hReady | .d2hEnd | .h2hIssue
  | .sendNext | .h2hDone _ | .cancel | .publish =>
    rw [Send.step_d2hIssued_of_ne_d2hIssue hs nofun] at hli
    exact h.woken_d2hReady l hl hli
  | .wake ok => exact absurd rfl (he ok)

theorem step_inv {s s' : Pipeline} {e : Ev} (h : Inv s) (hs : step s e = some s') : Inv s' := by
  cases e with
  | notifyForRead =>
    simp only [step, notifyForRead] at hs
    split at hs
    · cases hs
      exact ⟨h.send, h.recv, h.n_send, h.n_recv, h.len_pstaging, h.len_wire, h.len_dstaging,
        h.len_dhbm, h.len_d2hReadyL, h.len_h2hRetiredL, h.len_claimedL, h.len_h2dPendingL,
        h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good, h.reclaimed_done, h.cnt_d2hReady,
        h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady, h.pstaging_good, h.wire_good,
        h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
        h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, h.cnt_h2dReady, h.dhbm_good,
        fun _ => rfl, h.pullWaiting_state⟩
    · cases hs
  | pullWait =>
    simp only [step, pullWait] at hs
    split at hs
    · rename_i hg
      cases hs
      refine ⟨h.send, h.recv, h.n_send, h.n_recv, h.len_pstaging, h.len_wire, h.len_dstaging,
        h.len_dhbm, h.len_d2hReadyL, h.len_h2hRetiredL, h.len_claimedL, h.len_h2dPendingL,
        h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good, h.reclaimed_done, h.cnt_d2hReady,
        h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady, h.pstaging_good, h.wire_good,
        h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
        h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, h.cnt_h2dReady, h.dhbm_good,
        h.pullStarted_registered, ?_⟩
      intro _
      refine ⟨hg.2.2, ?_⟩
      cases hps : s.send.pullStarted
      · rfl
      · have := h.pullStarted_registered hps; rw [hg.1] at this; cases this
    · cases hs
  | send e =>
    cases e
    case d2hReady => simp only [step, sendStep] at hs; cases hs
    case h2hDone ok => simp only [step, sendStep] at hs; cases hs
    case beginPull =>
      simp only [step, sendStep] at hs
      split at hs
      · rename_i hg
        simp only [Option.map_eq_some_iff] at hs
        obtain ⟨snd, hsnd, rfl⟩ := hs
        have hw := woken_after_other h hsnd nofun
        refine ⟨Send.step_inv h.send hsnd, h.recv, (Send.step_numLayers hsnd).trans h.n_send, h.n_recv,
          h.len_pstaging, h.len_wire, h.len_dstaging, h.len_dhbm, h.len_d2hReadyL, h.len_h2hRetiredL,
          h.len_claimedL, h.len_h2dPendingL, h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good,
          fun hr => Send.step_done_mono hsnd (h.reclaimed_done hr),
          by rw [Send.step_d2hReady hsnd nofun]; exact h.cnt_d2hReady,
          fun l hl => Nat.lt_of_lt_of_le (h.d2hReady_lt l hl) (Send.step_d2hIssued_le hsnd),
          by rw [Send.step_h2hRetired hsnd nofun]; exact h.cnt_h2hRetired,
          hw,
          fun hd => h.pstaging_good (by cases hd0 : s.send.life.done; rfl; rw [Send.step_done_mono hsnd hd0] at hd; cases hd),
          h.wire_good, h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
          h.cnt_h2dPending, h.cnt_h2dIssued, h.dstaging_good, h.cnt_h2dReady, h.dhbm_good,
          fun _ => hg, nofun⟩
      · cases hs
    case wake ok =>
      simp only [step, sendStep] at hs
      split at hs
      · rename_i hg
        simp only [Option.map_eq_some_iff] at hs
        obtain ⟨snd, hsnd, rfl⟩ := hs
        exact h.send_frame hsnd nofun nofun (woken_after_wake h hsnd hg) _ h.len_h2hRetiredL
          (by rw [Send.step_h2hRetired hsnd nofun]; exact h.cnt_h2hRetired)
      · cases hs
    all_goals
      (try (rename_i ok; cases ok))
      all_goals
        simp only [step, sendStep, Option.map_eq_some_iff] at hs
        obtain ⟨snd, hsnd, rfl⟩ := hs
        exact h.send_frame hsnd nofun nofun (woken_after_other h hsnd nofun) _ h.len_h2hRetiredL
          (by rw [Send.step_h2hRetired hsnd nofun]; exact h.cnt_h2hRetired)
  | d2hReady l =>
    simp only [step, d2hReady] at hs
    split at hs
    · rename_i hg
      obtain ⟨hl, hf⟩ := hg
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨snd, hsnd, rfl⟩ := hs
      exact h.send_d2hReady hl hf hsnd
    · cases hs
  | h2hDone l ok =>
    simp only [step, h2hDone] at hs
    split at hs
    · rename_i hg
      obtain ⟨hl, hf⟩ := hg
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨snd, hsnd, rfl⟩ := hs
      have hw := woken_after_other h hsnd nofun
      cases ok
      · exact h.send_frame hsnd nofun nofun hw _
          (by rw [List.length_set]; exact h.len_h2hRetiredL)
          (by rw [countTrue_set_true hf, h.cnt_h2hRetired, Send.h2hDone_h2hRetired hsnd])
      · exact h.send_h2hDone hl hf hsnd hw
    · cases hs
  | recv e =>
    cases e
    case h2dBegin => simp only [step, recvStep] at hs; cases hs
    case h2dIssue ok => simp only [step, recvStep] at hs; cases hs
    case h2dReady => simp only [step, recvStep] at hs; cases hs
    case pullReply ok =>
      simp only [step, recvStep] at hs
      split at hs
      · simp only [Option.map_eq_some_iff] at hs
        obtain ⟨rcv, hrcv, rfl⟩ := hs
        have hpi := Recv.step_pending_issued hrcv nofun nofun
        refine ⟨h.send, Recv.step_inv h.recv hrcv, h.n_send, (Recv.step_numLayers hrcv).trans h.n_recv,
          h.len_pstaging, h.len_wire, h.len_dstaging, h.len_dhbm, h.len_d2hReadyL, h.len_h2hRetiredL,
          h.len_claimedL, h.len_h2dPendingL, h.len_h2dIssuedL, h.len_h2dReadyL, h.phbm_good,
          h.reclaimed_done, h.cnt_d2hReady, h.d2hReady_lt, h.cnt_h2hRetired, h.woken_d2hReady,
          h.pstaging_good, h.wire_good, h.pending_landed, h.issued_landed, h.pending_claimed,
          h.issued_claimed, by rw [hpi.1]; exact h.cnt_h2dPending,
          by rw [hpi.2]; exact h.cnt_h2dIssued,
          fun hd => h.dstaging_good (by cases hd0 : s.recv.life.done; rfl; rw [Recv.step_done_mono hrcv hd0] at hd; cases hd),
          by rw [Recv.step_ready hrcv nofun]; exact h.cnt_h2dReady,
          h.dhbm_good, h.pullStarted_registered, nofun⟩
      · cases hs
    all_goals
      (try (rename_i ok; cases ok))
      all_goals
        simp only [step, recvStep, Option.map_eq_some_iff] at hs
        obtain ⟨rcv, hrcv, rfl⟩ := hs
        have hpi := Recv.step_pending_issued hrcv nofun nofun
        exact h.recv_frame hrcv nofun nofun _ _ _ h.len_claimedL h.len_h2dPendingL h.len_h2dIssuedL
          h.pending_landed h.issued_landed h.pending_claimed h.issued_claimed
          (by rw [hpi.1]; exact h.cnt_h2dPending) (by rw [hpi.2]; exact h.cnt_h2dIssued)
  | h2dBegin l =>
    simp only [step, h2dBegin] at hs
    split at hs
    · rename_i hg
      obtain ⟨hld, hf⟩ := hg
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, rfl⟩ := hs
      obtain ⟨_, hpend, hiss⟩ := Recv.h2dBegin_spec hrcv
      have hpf : s.h2dPendingL[l]? = some false :=
        false_of_not_true (h.len_claimedL.trans h.len_h2dPendingL.symm) hf
          (fun hp => by rw [h.pending_claimed l hp] at hf; cases hf)
      have hif : s.h2dIssuedL[l]? = some false :=
        false_of_not_true (h.len_claimedL.trans h.len_h2dIssuedL.symm) hf
          (fun hi => by rw [(h.issued_claimed l hi).1] at hf; cases hf)
      apply h.recv_frame hrcv nofun nofun _ _ _
        (by rw [List.length_set]; exact h.len_claimedL)
        (by rw [List.length_set]; exact h.len_h2dPendingL)
        h.len_h2dIssuedL
      · intro j hj
        rcases mem_of_set_true hj with rfl | hj
        · exact hld
        · exact h.pending_landed j hj
      · exact h.issued_landed
      · intro j hj
        rcases mem_of_set_true hj with rfl | hj
        · exact set_true_self hf
        · exact set_true_of_mem hf (h.pending_claimed j hj)
      · intro j hj
        have hjl : j ≠ l := by intro rfl; rw [hif] at hj; cases hj
        refine ⟨set_true_of_mem hf (h.issued_claimed j hj).1, ?_⟩
        rw [List.getElem?_set_ne (Ne.symm hjl)]
        exact (h.issued_claimed j hj).2
      · rw [countTrue_set_true hpf, h.cnt_h2dPending, hpend]
      · rw [h.cnt_h2dIssued, hiss]
    · cases hs
  | h2dIssue l ok =>
    simp only [step, h2dIssue] at hs
    split at hs
    · rename_i hp
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, rfl⟩ := hs
      obtain ⟨_, hpend, hiss⟩ := Recv.h2dIssue_spec hrcv
      have hlt_p : l < s.h2dPendingL.length := lt_length_of_getElem?_eq hp
      have hif : s.h2dIssuedL[l]? = some false := by
        have hlt_i : l < s.h2dIssuedL.length := by
          rw [h.len_h2dIssuedL, ← h.len_h2dPendingL]; exact hlt_p
        have heq := List.getElem?_eq_getElem hlt_i
        cases hb : s.h2dIssuedL[l]
        · rw [heq, hb]
        · rw [hb] at heq
          rw [(h.issued_claimed l heq).2] at hp; cases hp
      apply h.recv_frame hrcv nofun nofun _ _ _
        h.len_claimedL
        (by rw [List.length_set]; exact h.len_h2dPendingL)
        (by split <;> simp [h.len_h2dIssuedL])
      · intro j hj
        exact h.pending_landed j (mem_of_set_false hj)
      · intro j hj
        split at hj
        · exact h.issued_landed j hj
        · rcases mem_of_set_true hj with rfl | hj'
          · exact h.pending_landed j hp
          · exact h.issued_landed j hj'
      · intro j hj
        exact h.pending_claimed j (mem_of_set_false hj)
      · intro j hj
        split at hj
        · refine ⟨(h.issued_claimed j hj).1, ?_⟩
          by_cases hjl : j = l
          · subst hjl; exact List.getElem?_set_self hlt_p
          · rw [List.getElem?_set_ne (Ne.symm hjl)]; exact (h.issued_claimed j hj).2
        · rcases mem_of_set_true hj with rfl | hj'
          · exact ⟨h.pending_claimed j hp, List.getElem?_set_self hlt_p⟩
          · have hjl : j ≠ l := by intro rfl; rw [hif] at hj'; cases hj'
            refine ⟨(h.issued_claimed j hj').1, ?_⟩
            rw [List.getElem?_set_ne (Ne.symm hjl)]; exact (h.issued_claimed j hj').2
      · have := countTrue_set_false hp
        rw [h.cnt_h2dPending] at this
        omega
      · rw [hiss]
        split
        · exact h.cnt_h2dIssued
        · rw [countTrue_set_true hif, h.cnt_h2dIssued]
    · cases hs
  | h2dReady l =>
    simp only [step, h2dReady] at hs
    split at hs
    · rename_i hg
      obtain ⟨hc, hf⟩ := hg
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨rcv, hrcv, rfl⟩ := hs
      exact h.recv_h2dReady hc hf hrcv
    · cases hs
  | land l =>
    simp only [step, land] at hs
    split at hs
    · rename_i hg
      obtain ⟨-, hf⟩ := hg
      split at hs
      · rename_i c hc
        split at hs
        · cases hs
        · rename_i hcb
          cases hs
          exact h.land_layer hf hc hcb
      · cases hs
    · cases hs
  | reclaim =>
    simp only [step, reclaim] at hs
    split at hs
    · rename_i hg
      obtain ⟨hp, -⟩ := hg
      cases hs
      obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.mp hp
      exact h.env_frame _ true s.prefillStaging s.decodeStaging h.len_pstaging h.len_dstaging
        nofun (fun _ => h.send.published_done b hb) h.pstaging_good h.dstaging_good
    · cases hs
  | reseatPrefillStaging =>
    simp only [step, reseatPrefillStaging] at hs
    split at hs
    · rename_i hst
      cases hs
      have hd := h.send.life.done_of_released hst
      exact h.env_frame s.prefillHbm s.reclaimed _ s.decodeStaging
        List.length_replicate h.len_dstaging h.phbm_good h.reclaimed_done
        (by rw [hd]; nofun) h.dstaging_good
    · cases hs
  | reseatDecodeStaging =>
    simp only [step, reseatDecodeStaging] at hs
    split at hs
    · rename_i hst
      cases hs
      have hd := h.recv.life.done_of_released hst
      exact h.env_frame s.prefillHbm s.reclaimed s.prefillStaging _
        h.len_pstaging List.length_replicate h.phbm_good h.reclaimed_done
        h.pstaging_good (by rw [hd]; nofun)
    · cases hs

theorem reachable_inv {n : Nat} {s : Pipeline} (h : (sys n).Reachable s) : Inv s :=
  (sys n).reachable_induction (inv_init n) (fun _ _ _ hi hs => step_inv hi hs) h

theorem reachable_unregistered_inv {n : Nat} {s : Pipeline}
    (h : (sysUnregistered n).Reachable s) : Inv s :=
  (sysUnregistered n).reachable_induction (inv_initUnregistered n)
    (fun _ _ _ hi hs => step_inv hi hs) h

/-- Main result: every reachable state of the pipeline satisfies all the
properties. -/
theorem reachable_safe {n : Nat} {s : Pipeline} (h : (sys n).Reachable s) : Safe s :=
  inv_safe (reachable_inv h)

/-- Every state reachable from an initially unregistered pipeline
(`initUnregistered n`) also satisfies all safety and handshake properties. -/
theorem reachable_unregistered_safe {n : Nat} {s : Pipeline}
    (h : (sysUnregistered n).Reachable s) : Safe s :=
  inv_safe (reachable_unregistered_inv h)


end Pipeline

end TpuSyncVerify.Transfer.PrefillDecode

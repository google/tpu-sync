import TpuSyncVerify.Transfer.PrefillDecode.Pipeline.Lemmas

/-! # `Pipeline`: inductive invariant, preservation and `reachable_safe`

Part of `TpuSyncVerify.Transfer.PrefillDecode.Pipeline`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ## Inductive invariant

`Inv` combines the session invariants (`Send.Inv` and `Recv.Inv`, which enforce
the settle gate and `in_flight_` conservation) with:
- **At-most-once layer checklists:** `chk_d2hReady`, `chk_h2hRetired`,
  `chk_h2dPending`, `chk_h2dIssued`, and `chk_h2dReady` package each per-layer
  `List Bool` length and `countTrue` equality with the session counter.
- **Per-layer stage ordering & data provenance:** `phbm_good`, `reclaimed_done`,
  `d2hReady_lt`, `woken_d2hReady`, `pstaging_good`, `wire_good`,
  `wire_h2hRetired`, `landed_wire`, `pending_landed`, `issued_landed`,
  `pending_claimed`, `issued_claimed`, `dstaging_good`, and `dhbm_good`.
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
  len_claimedL : s.claimedL.length = s.numLayers
  chk_d2hReady : Checklist s.d2hReadyL s.numLayers s.send.d2hReady
  chk_h2hRetired : Checklist s.h2hRetiredL s.numLayers s.send.h2hRetired
  chk_h2dPending : Checklist s.h2dPendingL s.numLayers s.recv.pending
  chk_h2dIssued : Checklist s.h2dIssuedL s.numLayers s.recv.issued
  chk_h2dReady : Checklist s.h2dReadyL s.numLayers s.recv.ready
  /-- Prefill HBM holds the data until the engine reclaims it. -/
  phbm_good : s.reclaimed = false → s.prefillHbm = good s.numLayers
  /-- Prefill HBM is reclaimed only once the send has settled. -/
  reclaimed_done : s.reclaimed = true → s.send.life.done = true
  /-- Only issued D2H copies have finished. -/
  d2hReady_lt : ∀ l : Nat, s.d2hReadyL[l]? = some true → l < s.send.d2hIssued
  /-- Only issued H2H pushes have retired. -/
  h2hRetired_lt : ∀ l : Nat, s.h2hRetiredL[l]? = some true → l < s.send.h2hIssued
  /-- `SendNextLayer(l)`'s callback has run for an issued layer only after that
  layer's copy finished. -/
  woken_d2hReady : ∀ l : Nat, l < s.send.woken → l < s.send.d2hIssued → s.d2hReadyL[l]? = some true
  /-- While the send holds its staging, every finished copy's layer is there. -/
  pstaging_good : s.send.life.done = false →
    ∀ l : Nat, s.d2hReadyL[l]? = some true → s.prefillStaging[l]? = some (.kv l)
  /-- Whatever was delivered for layer `k` is layer `k`. -/
  wire_good : ∀ (k : Nat) (c : Cell), s.wire[k]? = some c → c = .blank ∨ c = .kv k
  /-- Every delivered layer on the wire comes from a retired H2H push. -/
  wire_h2hRetired : ∀ l : Nat, s.wire[l]? = some (.kv l) → s.h2hRetiredL[l]? = some true
  /-- Every landed layer was read from a delivered wire cell. -/
  landed_wire : ∀ l : Nat, s.landedL[l]? = some true → s.wire[l]? = some (.kv l)
  /-- A layer is pending in `ExecuteLayerH2d` only after it landed. -/
  pending_landed : ∀ l : Nat, s.h2dPendingL[l]? = some true → s.landedL[l]? = some true
  /-- A layer's H2D copy is issued to the device only after it landed. -/
  issued_landed : ∀ l : Nat, s.h2dIssuedL[l]? = some true → s.landedL[l]? = some true
  /-- A pending layer was claimed. -/
  pending_claimed : ∀ l : Nat, s.h2dPendingL[l]? = some true → s.claimedL[l]? = some true
  /-- An issued layer was claimed and is no longer pending. -/
  issued_claimed : ∀ l : Nat, s.h2dIssuedL[l]? = some true →
    s.claimedL[l]? = some true ∧ s.h2dPendingL[l]? = some false
  /-- While the receive holds its staging, every landed layer is there. -/
  dstaging_good : s.recv.life.done = false →
    ∀ l : Nat, s.landedL[l]? = some true → s.decodeStaging[l]? = some (.kv l)
  /-- Only issued H2D copies have finished. -/
  ready_issued : ∀ l : Nat, s.h2dReadyL[l]? = some true → s.h2dIssuedL[l]? = some true
  /-- Every finished copy put its layer in decode HBM. -/
  dhbm_good : ∀ l : Nat, s.h2dReadyL[l]? = some true → s.decodeHbm[l]? = some (.kv l)
  /-- `ValidateAndBeginPull` can only claim the send session once `NotifyForRead`
  has registered the offer in `send_sessions_[uuid]`. -/
  pullStarted_registered : s.send.pullStarted = true → s.registered = true
  /-- While `HandlePullStream` is in its grace wait (`cv_.WaitWithTimeout`),
  the consumer's pull RPC is still pending and `pull_started_` is still false. -/
  pullWaiting_state : s.pullWaiting = true →
    s.recv.pullPending = true ∧ s.send.pullStarted = false

namespace Inv
theorem len_d2hReadyL {s : Pipeline} (h : Inv s) : s.d2hReadyL.length = s.numLayers := h.chk_d2hReady.len
theorem cnt_d2hReady {s : Pipeline} (h : Inv s) : countTrue s.d2hReadyL = s.send.d2hReady := h.chk_d2hReady.cnt_eq
theorem len_h2hRetiredL {s : Pipeline} (h : Inv s) : s.h2hRetiredL.length = s.numLayers := h.chk_h2hRetired.len
theorem cnt_h2hRetired {s : Pipeline} (h : Inv s) : countTrue s.h2hRetiredL = s.send.h2hRetired := h.chk_h2hRetired.cnt_eq
theorem len_h2dPendingL {s : Pipeline} (h : Inv s) : s.h2dPendingL.length = s.numLayers := h.chk_h2dPending.len
theorem cnt_h2dPending {s : Pipeline} (h : Inv s) : countTrue s.h2dPendingL = s.recv.pending := h.chk_h2dPending.cnt_eq
theorem len_h2dIssuedL {s : Pipeline} (h : Inv s) : s.h2dIssuedL.length = s.numLayers := h.chk_h2dIssued.len
theorem cnt_h2dIssued {s : Pipeline} (h : Inv s) : countTrue s.h2dIssuedL = s.recv.issued := h.chk_h2dIssued.cnt_eq
theorem len_h2dReadyL {s : Pipeline} (h : Inv s) : s.h2dReadyL.length = s.numLayers := h.chk_h2dReady.len
theorem cnt_h2dReady {s : Pipeline} (h : Inv s) : countTrue s.h2dReadyL = s.recv.ready := h.chk_h2dReady.cnt_eq

/-- Frame lemma for handshake-only updates (`registered`, `pullWaiting`). -/
theorem handshake_frame {s : Pipeline} (h : Inv s) (reg pw : Bool)
    (hreg : s.send.pullStarted = true → reg = true)
    (hpw : pw = true → s.recv.pullPending = true ∧ s.send.pullStarted = false) :
    Inv { s with registered := reg, pullWaiting := pw } :=
  ⟨h.send, h.recv, h.n_send, h.n_recv, h.len_pstaging, h.len_wire, h.len_dstaging, h.len_dhbm,
   h.len_claimedL, h.chk_d2hReady, h.chk_h2hRetired, h.chk_h2dPending, h.chk_h2dIssued,
   h.chk_h2dReady, h.phbm_good, h.reclaimed_done, h.d2hReady_lt, h.h2hRetired_lt,
   h.woken_d2hReady, h.pstaging_good, h.wire_good, h.wire_h2hRetired, h.landed_wire,
   h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
   h.dstaging_good, h.ready_issued, h.dhbm_good, hreg, hpw⟩
end Inv

theorem inv_init (n : Nat) : Inv (init n) := by
  refine ⟨Send.inv_init n, Recv.inv_initLoad n, rfl, rfl, by simp [init], by simp [init],
    by simp [init], by simp [init], by simp [init],
    Checklist.init n, Checklist.init n, Checklist.init n, Checklist.init n, Checklist.init n,
    fun _ => rfl, by simp [init, Send.init],
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    by simp [init, Send.init],
    fun _ _ h => (not_mem_replicate_false h).elim,
    (fun k c hk => by simp only [init, List.getElem?_replicate] at hk; split at hk <;> simp_all),
    (fun l hk => by simp only [init, List.getElem?_replicate] at hk; split at hk <;> cases hk),
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ h => (not_mem_replicate_false h).elim,
    fun _ => rfl, nofun⟩

theorem inv_initUnregistered (n : Nat) : Inv (initUnregistered n) :=
  (inv_init n).handshake_frame false false nofun nofun

theorem mem_drainEvents_d2hReady {n l : Nat} (hl : l < n) :
    Ev.d2hReady l ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]; exact Or.inr ⟨l, hl, by simp⟩

theorem mem_drainEvents_h2hDone {n l : Nat} (hl : l < n) :
    Ev.h2hDone l true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]; exact Or.inr ⟨l, hl, by simp⟩

theorem mem_drainEvents_h2dIssue {n l : Nat} (hl : l < n) :
    Ev.h2dIssue l true ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]; exact Or.inr ⟨l, hl, by simp⟩

theorem mem_drainEvents_h2dReady {n l : Nat} (hl : l < n) :
    Ev.h2dReady l ∈ drainEvents n := by
  simp only [drainEvents, List.mem_append, List.mem_flatMap, List.mem_range]; exact Or.inr ⟨l, hl, by simp⟩

theorem inv_noOpLeak {s : Pipeline} (h : Inv s) : NoOpLeak s := by
  rintro (hif | hif)
  · have hacc := h.send.accounted; unfold Send.Accounted at hacc
    have hcnt := h.send.counters; unfold Send.CountersOrdered at hcnt
    by_cases hp : s.send.d2hPending = true
    · exact ⟨.send (.d2hIssue true), by simp [drainEvents],
        by simp [step, sendStep, Send.step, Send.d2hIssue, hp]⟩
    · have hp' : s.send.d2hPending = false := by cases b : s.send.d2hPending <;> simp_all
      by_cases hrd : s.send.d2hReady < s.send.d2hIssued
      · obtain ⟨l, hl, hf⟩ := h.chk_d2hReady.exists_false (by rw [← h.n_send]; omega) hrd
        exact ⟨.d2hReady l, mem_drainEvents_d2hReady (by rw [← h.n_send]; omega),
          by simp [step, d2hReady, hl, hf, Send.step, Send.markD2hReady, hrd]⟩
      · by_cases hret : s.send.d2hRetired < s.send.d2hReady
        · exact ⟨.send .d2hEnd, by simp [drainEvents],
            by simp [step, sendStep, Send.step, Send.d2hEnd, hret]⟩
        · by_cases hwk : s.send.woken < s.send.queued
          · by_cases hwi : s.send.woken < s.send.d2hIssued
            · have hwr : s.send.woken < s.send.d2hReady := by omega
              have hwl := all_true_lt_of_countTrue h.d2hReady_lt
                (by rw [h.cnt_d2hReady]; omega) s.send.woken hwi
              refine ⟨.send (.wake true), by simp [drainEvents], ?_⟩
              simp only [step, sendStep, hwl, true_or, ↓reduceIte, Send.step, Send.wake,
                hwk, hwr, and_self, Bool.true_eq_false]; split <;> rfl
            · cases hdr : s.send.life.draining
              · have hst := h.send.queued_started (by omega)
                have hnd : s.send.life.done = false := by
                  cases hd : s.send.life.done
                  · rfl
                  · have := h.send.life.done_draining hd; rw [hdr] at this; cases this
                exact ⟨.send .d2hBegin, by simp [drainEvents],
                  by simp [step, sendStep, Send.step, Send.d2hBegin, hst, hp',
                       show s.send.d2hIssued < s.send.numLayers by omega, Lifecycle.beginOp, hdr, hnd]⟩
              · exact ⟨.send (.wake false), by simp [drainEvents],
                  by simp [step, sendStep, Send.step, Send.wake, hwk, hdr, hp',
                       show s.send.d2hIssued ≤ s.send.woken by omega]⟩
          · by_cases hpool : 0 < s.send.pooled
            · refine ⟨.send .h2hIssue, by simp [drainEvents], ?_⟩
              simp only [step, sendStep, Send.step, Send.h2hIssue, show s.send.pooled ≠ 0 by omega, ↓reduceIte]
              split <;> rfl
            · by_cases hch : 0 < s.send.chaining
              · exact ⟨.send .sendNext, by simp [drainEvents],
                  by simp [step, sendStep, Send.step, Send.sendNext]; omega⟩
              · have hh2h : s.send.h2hRetired < s.send.h2hIssued := by simp [hp'] at hacc; omega
                obtain ⟨l, hl, hf⟩ := h.chk_h2hRetired.exists_false (by rw [← h.n_send]; omega) hh2h
                exact ⟨.h2hDone l true, mem_drainEvents_h2hDone (by rw [← h.n_send]; omega),
                  by simp [step, h2hDone, hl, hf, Send.step, Send.h2hDone, hh2h]⟩
  · have hacc := h.recv.accounted; unfold Recv.Accounted at hacc
    have := h.recv.retired_le; have := h.recv.ready_le
    by_cases hp : 0 < s.recv.pushes
    · exact ⟨.recv .pushEnd, by simp [drainEvents],
        by simp [step, recvStep, Recv.step, Recv.pushEnd]; omega⟩
    · by_cases hq : s.recv.pullPending = true
      · exact ⟨.recv (.pullReply false), by simp [drainEvents],
          by simp [step, recvStep, Recv.step, Recv.pullReply, hq]⟩
      · by_cases hpend : 0 < s.recv.pending
        · obtain ⟨l, hl, hpl⟩ := h.chk_h2dPending.exists_true hpend
          refine ⟨.h2dIssue l true, mem_drainEvents_h2dIssue hl, ?_⟩
          simp only [step, h2dIssue, hpl, ↓reduceIte, Recv.step, Recv.h2dIssue,
            show s.recv.pending ≠ 0 by omega]; (repeat' split) <;> rfl
        · by_cases hrd : s.recv.ready < s.recv.issued
          · obtain ⟨l, hl, hrf, hit⟩ := h.chk_h2dReady.exists_diff h.chk_h2dIssued hrd
            exact ⟨.h2dReady l, mem_drainEvents_h2dReady hl,
              by simp [step, h2dReady, hit, hrf, Recv.step, Recv.h2dReady, hrd]⟩
          · have hret : s.recv.retired < s.recv.ready := by
              cases hqp : s.recv.pullPending <;> simp_all <;> omega
            refine ⟨.recv (.h2dDone true), by simp [drainEvents], ?_⟩
            simp only [step, recvStep, Recv.step, Recv.h2dDone, hret, ↓reduceIte]; split <;> rfl

theorem inv_handshakeSafe {s : Pipeline} (h : Inv s) : HandshakeSafe s := by
  have hcnt := h.send.counters; unfold Send.CountersOrdered at hcnt
  have ⟨hpc₁, hpc₂⟩ := h.send.pull_claimed
  refine ⟨fun hs => ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩,
    h.pullStarted_registered, ?_, h.pullWaiting_state⟩
  rintro (hdp | hd2h | hh2h)
  · have hs := hpc₂ (Or.inl hdp); exact ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩
  · have hs := hpc₂ (Or.inr hd2h); exact ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩
  · have hs := hpc₂ (Or.inr (by omega)); exact ⟨hpc₁ hs, h.pullStarted_registered (hpc₁ hs)⟩

theorem inv_recv_ready_eq_of_published_ok {s : Pipeline} (h : Inv s)
    (hp : s.recv.published = some true) : s.recv.ready = s.numLayers := by
  have := h.recv.published_ok hp; have := h.recv.completed_le; have := h.recv.retired_le
  have := h.recv.ready_le; have := h.recv.issued_le; have := h.n_recv; omega

/-- Publishing `done_recving` for `n > 0` layers implies at least layer `0` was
dispatched on both the producer (`0 < send.d2hIssued`) and consumer (`0 < recv.issued`). -/
theorem inv_published_ok_issued_pos {s : Pipeline} (h : Inv s)
    (hpos : 0 < s.numLayers) (hp : s.recv.published = some true) :
    0 < s.send.d2hIssued ∧ 0 < s.recv.issued := by
  have hready := inv_recv_ready_eq_of_published_ok h hp
  have hr0 := (h.chk_h2dReady.cast_cnt hready).all_true hpos
  have hret0 := h.wire_h2hRetired 0 (h.landed_wire 0 (h.issued_landed 0 (h.ready_issued 0 hr0)))
  have hh0 := h.h2hRetired_lt 0 hret0
  have hcnt := h.send.counters; unfold Send.CountersOrdered at hcnt
  have := h.recv.ready_le
  omega

theorem inv_safe {s : Pipeline} (h : Inv s) : Safe s := by
  have hsend := h.send; have hrecv := h.recv; have hsS := Send.inv_safe hsend
  have hcnt := hsend.counters; unfold Send.CountersOrdered at hcnt
  refine ⟨fun hp => eq_good h.len_dhbm fun k hk =>
      h.dhbm_good k ((h.chk_h2dReady.cast_cnt (inv_recv_ready_eq_of_published_ok h hp)).all_true hk),
    ?_, ?_, ⟨?_, ?_⟩, inv_handshakeSafe h, inv_noOpLeak h⟩
  · intro hp
    cases hpub : s.recv.published with
    | none => exact (hp hpub).elim
    | some b =>
      have hd := hrecv.published_done b hpub
      have := hrecv.life.done_idle hd; have hacc := hrecv.accounted; unfold Recv.Accounted at hacc
      have := hrecv.retired_le; have := hrecv.ready_le; exact ⟨by omega, by omega⟩
  · intro hp
    cases hpub : s.send.published with
    | none => exact (hp hpub).elim
    | some b =>
      obtain ⟨h0, h1, -⟩ := hsS.2.1 (hsend.published_done b hpub); exact ⟨h0, h1⟩
  · intro hst
    obtain ⟨h0, h1, -, -, -, h2⟩ := hsS.2.1 (hsend.life.done_of_released hst); exact ⟨h0, h1, h2⟩
  · intro hst
    have hd := hrecv.life.done_of_released hst
    have := hrecv.life.done_idle hd; have hacc := hrecv.accounted; unfold Recv.Accounted at hacc
    have := hrecv.retired_le; have := hrecv.ready_le; exact ⟨by omega, by omega, by omega⟩

/-! ### Preservation, one lemma per kind of effect -/

theorem Inv.send_wire_frame {s : Pipeline} {snd : Send} {e : Send.Ev} (h : Inv s)
    (hs : Send.step s.send e = some snd) (he : e ≠ .d2hReady)
    (hw : ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true)
    (x : List Bool) (hxchk : Checklist x s.numLayers snd.h2hRetired)
    (hxlt : ∀ l : Nat, x[l]? = some true → l < snd.h2hIssued)
    (w : List Cell) (hwlen : w.length = s.numLayers)
    (hwgood : ∀ (k : Nat) (c : Cell), w[k]? = some c → c = .blank ∨ c = .kv k)
    (hwret : ∀ l : Nat, w[l]? = some (.kv l) → x[l]? = some true)
    (hlw : ∀ l : Nat, s.landedL[l]? = some true → w[l]? = some (.kv l))
    (pw : Bool := s.pullWaiting)
    (hreg : snd.pullStarted = true → s.registered = true)
    (hpw : pw = true → s.recv.pullPending = true ∧ snd.pullStarted = false) :
    Inv { s with send := snd, h2hRetiredL := x, wire := w, pullWaiting := pw } :=
  ⟨Send.step_inv h.send hs, h.recv, (Send.step_numLayers hs).trans h.n_send, h.n_recv,
   h.len_pstaging, hwlen, h.len_dstaging, h.len_dhbm, h.len_claimedL,
   h.chk_d2hReady.cast_cnt (Send.step_d2hReady hs he).symm, hxchk,
   h.chk_h2dPending, h.chk_h2dIssued, h.chk_h2dReady, h.phbm_good,
   fun hr => Send.step_done_mono hs (h.reclaimed_done hr),
   fun l hl => Nat.lt_of_lt_of_le (h.d2hReady_lt l hl) (Send.step_d2hIssued_le hs),
   hxlt, hw,
   fun hd => h.pstaging_good (by
     cases hd0 : s.send.life.done with
     | false => rfl
     | true => rw [Send.step_done_mono hs hd0] at hd; cases hd),
   hwgood, hwret, hlw, h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
   h.dstaging_good, h.ready_issued, h.dhbm_good, hreg, hpw⟩

theorem Inv.send_frame {s : Pipeline} {snd : Send} {e : Send.Ev} (h : Inv s)
    (hs : Send.step s.send e = some snd) (he : e ≠ .d2hReady) (he_bp : e ≠ .beginPull)
    (hw : ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true)
    (x : List Bool) (hxchk : Checklist x s.numLayers snd.h2hRetired)
    (hxlt : ∀ l : Nat, x[l]? = some true → l < snd.h2hIssued)
    (hxmono : ∀ l : Nat, s.h2hRetiredL[l]? = some true → x[l]? = some true) :
    Inv { s with send := snd, h2hRetiredL := x } :=
  h.send_wire_frame hs he hw x hxchk hxlt s.wire h.len_wire h.wire_good
    (fun l hl => hxmono l (h.wire_h2hRetired l hl)) h.landed_wire s.pullWaiting
    (by rw [Send.step_pullStarted_of_ne_beginPull hs he_bp]; exact h.pullStarted_registered)
    (by rw [Send.step_pullStarted_of_ne_beginPull hs he_bp]; exact h.pullWaiting_state)

theorem send_not_done_of_outstanding {t : Send} (h : t.Inv)
    (ho : t.d2hReady < t.d2hIssued ∨ t.h2hRetired < t.h2hIssued) : t.life.done = false := by
  cases hd : t.life.done
  · rfl
  · have := (Send.inv_safe h).2.1 hd; have hcnt := h.counters; unfold Send.CountersOrdered at hcnt; omega

theorem Inv.send_d2hReady {s : Pipeline} {snd : Send} {l : Nat} (h : Inv s)
    (hl : l < s.send.d2hIssued) (hf : s.d2hReadyL[l]? = some false)
    (hs : Send.step s.send .d2hReady = some snd) :
    Inv { s with send := snd,
                 prefillStaging := s.prefillStaging.set l (s.prefillHbm.getD l .junk),
                 d2hReadyL := s.d2hReadyL.set l true } := by
  obtain ⟨hlt, rfl⟩ := Send.d2hReady_spec hs
  have hd : s.send.life.done = false := send_not_done_of_outstanding h.send (Or.inl hlt)
  have hnr : s.reclaimed = false := by
    cases hr : s.reclaimed with
    | false => rfl
    | true => have := h.reclaimed_done hr; rw [hd] at this; cases this
  have hcnt := h.send.counters; unfold Send.CountersOrdered at hcnt
  have hk : l < s.numLayers := by have := h.n_send; omega
  rw [show s.prefillHbm.getD l .junk = .kv l from h.phbm_good hnr ▸ good_getD hk]
  exact ⟨Send.step_inv h.send hs, h.recv, (Send.step_numLayers hs).trans h.n_send, h.n_recv,
    (List.length_set ..).trans h.len_pstaging, h.len_wire, h.len_dstaging, h.len_dhbm, h.len_claimedL,
    h.chk_d2hReady.set_true hf, h.chk_h2hRetired, h.chk_h2dPending, h.chk_h2dIssued, h.chk_h2dReady,
    h.phbm_good, fun hr => Send.step_done_mono hs (h.reclaimed_done hr),
    fun j hj => (mem_of_set_true hj).elim (· ▸ hl) (h.d2hReady_lt j),
    fun l' hl' => Nat.lt_of_lt_of_le (h.h2hRetired_lt l' hl') (Send.step_h2hIssued_le hs),
    fun j hj hji => set_true_of_mem hf (h.woken_d2hReady j hj hji),
    fun _ j hj => getElem?_set_kv (h.len_pstaging ▸ hk) fun hjl =>
      h.pstaging_good hd j ((mem_of_set_true hj).resolve_left hjl),
    h.wire_good, h.wire_h2hRetired, h.landed_wire,
    h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed,
    h.dstaging_good, h.ready_issued, h.dhbm_good, h.pullStarted_registered, h.pullWaiting_state⟩

theorem Inv.send_h2hDone {s : Pipeline} {snd : Send} {l : Nat} (h : Inv s)
    (hl : l < s.send.h2hIssued) (hf : s.h2hRetiredL[l]? = some false)
    (hs : Send.step s.send (.h2hDone true) = some snd)
    (hw : ∀ l' < snd.woken, l' < snd.d2hIssued → s.d2hReadyL[l']? = some true) :
    Inv { s with send := snd, h2hRetiredL := s.h2hRetiredL.set l true,
                 wire := s.wire.set l (s.prefillStaging.getD l .junk) } := by
  have hlt := Send.h2hDone_guard hs
  have hcnt := h.send.counters; unfold Send.CountersOrdered at hcnt
  have hd : s.send.life.done = false := send_not_done_of_outstanding h.send (Or.inr hlt)
  have hk : l < s.numLayers := by have := h.n_send; omega
  have hdl : s.d2hReadyL[l]? = some true := h.woken_d2hReady l (by omega) (by omega)
  rw [show s.prefillStaging.getD l .junk = .kv l from getD_of_getElem?_eq_some (h.pstaging_good hd l hdl)]
  refine h.send_wire_frame hs nofun hw (s.h2hRetiredL.set l true)
    ((h.chk_h2hRetired.set_true hf).cast_cnt (Send.h2hDone_h2hRetired hs).symm)
    (fun j hj => (mem_of_set_true hj).elim
      (fun heq => heq ▸ Nat.lt_of_lt_of_le hl (Send.step_h2hIssued_le hs))
      (fun hj => Nat.lt_of_lt_of_le (h.h2hRetired_lt j hj) (Send.step_h2hIssued_le hs)))
    (s.wire.set l (.kv l)) ((List.length_set ..).trans h.len_wire)
    (fun j c hj => by
      by_cases hjl : j = l
      · subst hjl; rw [List.getElem?_set_self (h.len_wire ▸ hk)] at hj; cases hj; exact Or.inr rfl
      · rw [List.getElem?_set_ne (Ne.symm hjl)] at hj; exact h.wire_good j c hj)
    (fun j hj => by
      by_cases hjl : j = l
      · subst hjl; exact set_true_self hf
      · rw [List.getElem?_set_ne (Ne.symm hjl)] at hj; exact set_true_of_mem hf (h.wire_h2hRetired j hj))
    (fun j hj => getElem?_set_kv (h.len_wire ▸ hk) fun _ => h.landed_wire j hj) s.pullWaiting
    (by rw [Send.step_pullStarted_of_ne_beginPull hs nofun]; exact h.pullStarted_registered)
    (by rw [Send.step_pullStarted_of_ne_beginPull hs nofun]; exact h.pullWaiting_state)

theorem Inv.recv_frame {s : Pipeline} {rcv : Recv} {e : Recv.Ev} (h : Inv s)
    (hs : Recv.step s.recv e = some rcv) (he : e ≠ .h2dReady)
    (c p i : List Bool) (hclen : c.length = s.numLayers)
    (hpchk : Checklist p s.numLayers rcv.pending)
    (hichk : Checklist i s.numLayers rcv.issued)
    (hp : ∀ l : Nat, p[l]? = some true → s.landedL[l]? = some true)
    (hi : ∀ l : Nat, i[l]? = some true → s.landedL[l]? = some true)
    (hpc : ∀ l : Nat, p[l]? = some true → c[l]? = some true)
    (hic : ∀ l : Nat, i[l]? = some true → c[l]? = some true ∧ p[l]? = some false)
    (hri : ∀ l : Nat, s.h2dReadyL[l]? = some true → i[l]? = some true)
    (pw : Bool := s.pullWaiting)
    (hpw : pw = true → rcv.pullPending = true ∧ s.send.pullStarted = false) :
    Inv { s with recv := rcv, claimedL := c, h2dPendingL := p, h2dIssuedL := i, pullWaiting := pw } :=
  ⟨h.send, Recv.step_inv h.recv hs, h.n_send, (Recv.step_numLayers hs).trans h.n_recv,
   h.len_pstaging, h.len_wire, h.len_dstaging, h.len_dhbm, hclen,
   h.chk_d2hReady, h.chk_h2hRetired, hpchk, hichk,
   h.chk_h2dReady.cast_cnt (Recv.step_ready hs he).symm,
   h.phbm_good, h.reclaimed_done, h.d2hReady_lt, h.h2hRetired_lt, h.woken_d2hReady,
   h.pstaging_good, h.wire_good, h.wire_h2hRetired, h.landed_wire, hp, hi, hpc, hic,
   fun hd => h.dstaging_good (by
     cases hd0 : s.recv.life.done with
     | false => rfl
     | true => rw [Recv.step_done_mono hs hd0] at hd; cases hd),
   hri, h.dhbm_good, h.pullStarted_registered, hpw⟩

theorem Inv.recv_h2dReady {s : Pipeline} {rcv : Recv} {l : Nat} (h : Inv s)
    (hc : s.h2dIssuedL[l]? = some true) (hf : s.h2dReadyL[l]? = some false)
    (hs : Recv.step s.recv .h2dReady = some rcv) :
    Inv { s with recv := rcv,
                 decodeHbm := s.decodeHbm.set l (s.decodeStaging.getD l .junk),
                 h2dReadyL := s.h2dReadyL.set l true } := by
  obtain ⟨hlt, rfl⟩ := Recv.h2dReady_spec hs
  have hd : s.recv.life.done = false := by
    cases hd : s.recv.life.done with
    | false => rfl
    | true => have := (Recv.inv_safe h.recv).2.1 hd; have := h.recv.retired_le; omega
  have hk : l < s.numLayers := h.len_h2dReadyL ▸ lt_length_of_getElem?_eq hf
  rw [show s.decodeStaging.getD l .junk = .kv l from
    getD_of_getElem?_eq_some (h.dstaging_good hd l (h.issued_landed l hc))]
  exact ⟨h.send, Recv.step_inv h.recv hs, h.n_send, (Recv.step_numLayers hs).trans h.n_recv,
    h.len_pstaging, h.len_wire, h.len_dstaging, (List.length_set ..).trans h.len_dhbm, h.len_claimedL,
    h.chk_d2hReady, h.chk_h2hRetired, h.chk_h2dPending, h.chk_h2dIssued, h.chk_h2dReady.set_true hf,
    h.phbm_good, h.reclaimed_done, h.d2hReady_lt, h.h2hRetired_lt, h.woken_d2hReady,
    h.pstaging_good, h.wire_good, h.wire_h2hRetired, h.landed_wire,
    h.pending_landed, h.issued_landed, h.pending_claimed, h.issued_claimed, h.dstaging_good,
    fun j hj => (mem_of_set_true hj).elim (· ▸ hc) (h.ready_issued j),
    fun j hj => getElem?_set_kv (h.len_dhbm ▸ hk) fun hjl =>
      h.dhbm_good j ((mem_of_set_true hj).resolve_left hjl),
    h.pullStarted_registered, h.pullWaiting_state⟩

theorem Inv.land_layer {s : Pipeline} {l : Nat} {c : Cell} (h : Inv s)
    (hf : s.landedL[l]? = some false) (hc : s.wire[l]? = some c) (hcb : c ≠ .blank) :
    Inv { s with decodeStaging := s.decodeStaging.set l c, landedL := s.landedL.set l true } := by
  have hkv : c = .kv l := (h.wire_good _ _ hc).resolve_left hcb
  have hlen : l < s.numLayers := h.len_wire ▸ lt_length_of_getElem?_eq hc
  subst hkv
  exact ⟨h.send, h.recv, h.n_send, h.n_recv, h.len_pstaging, h.len_wire,
    (List.length_set ..).trans h.len_dstaging, h.len_dhbm, h.len_claimedL,
    h.chk_d2hReady, h.chk_h2hRetired, h.chk_h2dPending, h.chk_h2dIssued, h.chk_h2dReady,
    h.phbm_good, h.reclaimed_done, h.d2hReady_lt, h.h2hRetired_lt, h.woken_d2hReady,
    h.pstaging_good, h.wire_good, h.wire_h2hRetired,
    fun j hj => (mem_of_set_true hj).elim (· ▸ hc) (h.landed_wire j),
    fun j hj => set_true_of_mem hf (h.pending_landed j hj),
    fun j hj => set_true_of_mem hf (h.issued_landed j hj),
    h.pending_claimed, h.issued_claimed,
    fun hd j hj => getElem?_set_kv (h.len_dstaging ▸ hlen) fun hjl =>
      h.dstaging_good hd j ((mem_of_set_true hj).resolve_left hjl),
    h.ready_issued, h.dhbm_good, h.pullStarted_registered, h.pullWaiting_state⟩

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
   h.len_claimedL, h.chk_d2hReady, h.chk_h2hRetired, h.chk_h2dPending, h.chk_h2dIssued,
   h.chk_h2dReady, hphbm, hrec, h.d2hReady_lt, h.h2hRetired_lt, h.woken_d2hReady,
   hpstg, h.wire_good, h.wire_h2hRetired, h.landed_wire, h.pending_landed, h.issued_landed,
   h.pending_claimed, h.issued_claimed, hdstg, h.ready_issued, h.dhbm_good,
   h.pullStarted_registered, h.pullWaiting_state⟩

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
  · exact hg.elim id fun ⟨_, _, _, hle⟩ => by omega

theorem woken_after_other {s : Pipeline} {snd : Send} {e : Send.Ev} (h : Inv s)
    (hs : Send.step s.send e = some snd) (he : ∀ ok, e ≠ .wake ok) :
    ∀ l < snd.woken, l < snd.d2hIssued → s.d2hReadyL[l]? = some true := by
  rw [Send.step_woken hs he]
  intro l hl hli
  match e with
  | .d2hIssue ok =>
    obtain ⟨hp, _⟩ := Send.d2hIssue_spec hs
    have hwle : s.send.woken ≤ s.send.d2hIssued := Nat.le_of_not_lt fun hlt => by
      have := (h.send.woken_issued hlt).2; rw [hp] at this; cases this
    exact h.woken_d2hReady l hl (Nat.lt_of_lt_of_le hl hwle)
  | .beginPull | .start | .d2hBegin | .d2hReady | .d2hEnd | .h2hIssue
  | .sendNext | .h2hDone _ | .cancel | .publish =>
    rw [Send.step_d2hIssued_of_ne_d2hIssue hs nofun] at hli
    exact h.woken_d2hReady l hl hli
  | .wake ok => exact absurd rfl (he ok)

theorem step_inv {s s' : Pipeline} {e : Ev} (h : Inv s) (hs : step s e = some s') : Inv s' := by
  cases e with
  | notifyForRead =>
    simp only [step, notifyForRead] at hs; split at hs <;> cases hs
    exact h.handshake_frame true s.pullWaiting (fun _ => rfl) h.pullWaiting_state
  | pullWait =>
    simp only [step, pullWait] at hs; split at hs <;> cases hs
    rename_i hg
    refine h.handshake_frame s.registered true h.pullStarted_registered fun _ => ⟨hg.2.2, ?_⟩
    cases hps : s.send.pullStarted with
    | false => rfl
    | true => have := h.pullStarted_registered hps; rw [hg.1] at this; cases this
  | send e =>
    cases e
    case d2hReady => simp [step, sendStep] at hs
    case h2hDone => simp [step, sendStep] at hs
    case beginPull =>
      simp only [step, sendStep] at hs; split at hs <;> try cases hs
      rename_i hg; rcases Option.map_eq_some_iff.mp hs with ⟨snd, hsnd, rfl⟩
      exact h.send_wire_frame hsnd nofun (woken_after_other h hsnd nofun)
        s.h2hRetiredL (h.chk_h2hRetired.cast_cnt (Send.step_h2hRetired hsnd nofun).symm)
        (fun l hl => Nat.lt_of_lt_of_le (h.h2hRetired_lt l hl) (Send.step_h2hIssued_le hsnd))
        s.wire h.len_wire h.wire_good h.wire_h2hRetired h.landed_wire false (fun _ => hg) nofun
    case wake ok =>
      simp only [step, sendStep] at hs; split at hs <;> try cases hs
      rename_i hg; rcases Option.map_eq_some_iff.mp hs with ⟨snd, hsnd, rfl⟩
      exact h.send_frame hsnd nofun nofun (woken_after_wake h hsnd hg) s.h2hRetiredL
        (h.chk_h2hRetired.cast_cnt (Send.step_h2hRetired hsnd nofun).symm)
        (fun l hl => Nat.lt_of_lt_of_le (h.h2hRetired_lt l hl) (Send.step_h2hIssued_le hsnd)) (fun _ => id)
    all_goals
      (try (rename_i ok; cases ok))
      all_goals
        simp only [step, sendStep, Option.map_eq_some_iff] at hs
        rcases hs with ⟨snd, hsnd, rfl⟩
        exact h.send_frame hsnd nofun nofun (woken_after_other h hsnd nofun) s.h2hRetiredL
          (h.chk_h2hRetired.cast_cnt (Send.step_h2hRetired hsnd nofun).symm)
          (fun l hl => Nat.lt_of_lt_of_le (h.h2hRetired_lt l hl) (Send.step_h2hIssued_le hsnd)) (fun _ => id)
  | d2hReady l =>
    simp only [step, d2hReady] at hs; split at hs <;> try cases hs
    rename_i hg; rcases Option.map_eq_some_iff.mp hs with ⟨snd, hsnd, rfl⟩
    exact h.send_d2hReady hg.1 hg.2 hsnd
  | h2hDone l ok =>
    simp only [step, h2hDone] at hs; split at hs <;> try cases hs
    rename_i hg; rcases Option.map_eq_some_iff.mp hs with ⟨snd, hsnd, rfl⟩
    have hw := woken_after_other h hsnd nofun
    cases ok
    · exact h.send_frame hsnd nofun nofun hw (s.h2hRetiredL.set l true)
        ((h.chk_h2hRetired.set_true hg.2).cast_cnt (Send.h2hDone_h2hRetired hsnd).symm)
        (fun j hj => (mem_of_set_true hj).elim
          (fun heq => heq ▸ Nat.lt_of_lt_of_le hg.1 (Send.step_h2hIssued_le hsnd))
          (fun hj => Nat.lt_of_lt_of_le (h.h2hRetired_lt j hj) (Send.step_h2hIssued_le hsnd)))
        (fun _ => set_true_of_mem hg.2)
    · exact h.send_h2hDone hg.1 hg.2 hsnd hw
  | recv e =>
    cases e
    case h2dBegin | h2dIssue | h2dReady => simp [step, recvStep] at hs
    case pullReply ok =>
      simp only [step, recvStep] at hs; split at hs <;> try cases hs
      rcases Option.map_eq_some_iff.mp hs with ⟨rcv, hrcv, rfl⟩
      have hpi := Recv.step_pending_issued hrcv nofun nofun
      exact h.recv_frame hrcv nofun s.claimedL s.h2dPendingL s.h2dIssuedL h.len_claimedL
        (h.chk_h2dPending.cast_cnt hpi.1.symm) (h.chk_h2dIssued.cast_cnt hpi.2.symm)
        h.pending_landed h.issued_landed h.pending_claimed h.issued_claimed h.ready_issued false nofun
    all_goals
      (try (rename_i ok; cases ok))
      all_goals
        simp only [step, recvStep, Option.map_eq_some_iff] at hs
        rcases hs with ⟨rcv, hrcv, rfl⟩
        have hpi := Recv.step_pending_issued hrcv nofun nofun
        exact h.recv_frame hrcv nofun s.claimedL s.h2dPendingL s.h2dIssuedL h.len_claimedL
          (h.chk_h2dPending.cast_cnt hpi.1.symm) (h.chk_h2dIssued.cast_cnt hpi.2.symm)
          h.pending_landed h.issued_landed h.pending_claimed h.issued_claimed h.ready_issued
          s.pullWaiting (by rw [Recv.step_pullPending_of_ne_pullReply hrcv nofun]; exact h.pullWaiting_state)
  | h2dBegin l =>
    simp only [step, h2dBegin] at hs; split at hs <;> try cases hs
    rename_i hg; obtain ⟨hld, hf⟩ := hg
    rcases Option.map_eq_some_iff.mp hs with ⟨rcv, hrcv, rfl⟩
    obtain ⟨_, hpend, hiss⟩ := Recv.h2dBegin_spec hrcv
    have hpf := false_of_not_true (h.len_claimedL.trans h.len_h2dPendingL.symm) hf
      fun hp => by rw [h.pending_claimed l hp] at hf; cases hf
    have hif := false_of_not_true (h.len_claimedL.trans h.len_h2dIssuedL.symm) hf
      fun hi => by rw [(h.issued_claimed l hi).1] at hf; cases hf
    refine h.recv_frame hrcv nofun (s.claimedL.set l true) (s.h2dPendingL.set l true) s.h2dIssuedL
      ((List.length_set ..).trans h.len_claimedL)
      ((h.chk_h2dPending.set_true hpf).cast_cnt hpend.symm) (h.chk_h2dIssued.cast_cnt hiss.symm)
      (fun j hj => (mem_of_set_true hj).elim (· ▸ hld) (h.pending_landed j))
      h.issued_landed
      (fun j hj => by
        rcases mem_of_set_true hj with rfl | hj
        · exact set_true_self hf
        · exact set_true_of_mem hf (h.pending_claimed j hj))
      (fun j hj => by
        have hjl : j ≠ l := by rintro rfl; rw [hif] at hj; cases hj
        exact ⟨set_true_of_mem hf (h.issued_claimed j hj).1,
          (List.getElem?_set_ne (Ne.symm hjl)).trans (h.issued_claimed j hj).2⟩)
      h.ready_issued s.pullWaiting
      (by rw [Recv.step_pullPending_of_ne_pullReply hrcv nofun]; exact h.pullWaiting_state)
  | h2dIssue l ok =>
    simp only [step, h2dIssue] at hs; split at hs <;> try cases hs
    rename_i hp; rcases Option.map_eq_some_iff.mp hs with ⟨rcv, hrcv, rfl⟩
    obtain ⟨_, hpend, hiss⟩ := Recv.h2dIssue_spec hrcv
    have hlt_p := lt_length_of_getElem?_eq hp
    have hif : s.h2dIssuedL[l]? = some false :=
      false_of_not_true (rfl : s.h2dIssuedL.length = s.h2dIssuedL.length)
        (by
          have hlt_i : l < s.h2dIssuedL.length := by rw [h.len_h2dIssuedL, ← h.len_h2dPendingL]; exact hlt_p
          have heq := List.getElem?_eq_getElem hlt_i
          cases hb : s.h2dIssuedL[l]
          · exact heq.trans (congrArg some hb)
          · rw [hb] at heq; rw [(h.issued_claimed l heq).2] at hp; cases hp)
        fun hi => by rw [(h.issued_claimed l hi).2] at hp; cases hp
    refine h.recv_frame hrcv nofun s.claimedL (s.h2dPendingL.set l false)
      (if s.recv.life.done || s.recv.life.draining || !ok then s.h2dIssuedL else s.h2dIssuedL.set l true)
      h.len_claimedL
      ((h.chk_h2dPending.set_false hp).cast_cnt hpend.symm)
      (by
        split <;> rename_i hbr
        · exact h.chk_h2dIssued.cast_cnt (by simp [hbr] at hiss; exact hiss.symm)
        · exact (h.chk_h2dIssued.set_true hif).cast_cnt (by simp [hbr] at hiss; exact hiss.symm))
      (fun j hj => h.pending_landed j (mem_of_set_false hj))
      (fun j hj => by
        split at hj
        · exact h.issued_landed j hj
        · rcases mem_of_set_true hj with rfl | hj'
          · exact h.pending_landed j hp
          · exact h.issued_landed j hj')
      (fun j hj => h.pending_claimed j (mem_of_set_false hj))
      (fun j hj => by
        split at hj
        · refine ⟨(h.issued_claimed j hj).1, ?_⟩
          by_cases hjl : j = l
          · subst hjl; exact List.getElem?_set_self hlt_p
          · exact (List.getElem?_set_ne (Ne.symm hjl)).trans (h.issued_claimed j hj).2
        · rcases mem_of_set_true hj with rfl | hj'
          · exact ⟨h.pending_claimed j hp, List.getElem?_set_self hlt_p⟩
          · have hjl : j ≠ l := by rintro rfl; rw [hif] at hj'; cases hj'
            exact ⟨(h.issued_claimed j hj').1,
              (List.getElem?_set_ne (Ne.symm hjl)).trans (h.issued_claimed j hj').2⟩)
      (fun j hj => by
        have hji := h.ready_issued j hj
        split
        · exact hji
        · by_cases hjl : j = l
          · subst hjl; exact set_true_self hif
          · exact (List.getElem?_set_ne (Ne.symm hjl)).trans hji)
      s.pullWaiting (by rw [Recv.step_pullPending_of_ne_pullReply hrcv nofun]; exact h.pullWaiting_state)
  | h2dReady l =>
    simp only [step, h2dReady] at hs; split at hs <;> try cases hs
    rename_i hg; rcases Option.map_eq_some_iff.mp hs with ⟨rcv, hrcv, rfl⟩
    exact h.recv_h2dReady hg.1 hg.2 hrcv
  | land l =>
    simp only [step, land] at hs; split at hs <;> try cases hs
    rename_i hg; split at hs <;> try cases hs
    rename_i c hc; split at hs <;> cases hs
    rename_i hcb; exact h.land_layer hg.2 hc hcb
  | reclaim =>
    simp only [step, reclaim] at hs; split at hs <;> cases hs
    rename_i hg; obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.mp hg.1
    exact h.env_frame _ true s.prefillStaging s.decodeStaging h.len_pstaging h.len_dstaging
      nofun (fun _ => h.send.published_done b hb) h.pstaging_good h.dstaging_good
  | reseatPrefillStaging =>
    simp only [step, reseatPrefillStaging] at hs; split at hs <;> cases hs
    rename_i hst; have hd := h.send.life.done_of_released hst
    exact h.env_frame s.prefillHbm s.reclaimed _ s.decodeStaging
      List.length_replicate h.len_dstaging h.phbm_good h.reclaimed_done
      (by rw [hd]; nofun) h.dstaging_good
  | reseatDecodeStaging =>
    simp only [step, reseatDecodeStaging] at hs; split at hs <;> cases hs
    rename_i hst; have hd := h.recv.life.done_of_released hst
    exact h.env_frame s.prefillHbm s.reclaimed s.prefillStaging _
      h.len_pstaging List.length_replicate h.phbm_good h.reclaimed_done
      h.pstaging_good (by rw [hd]; nofun)

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

import TpuSyncVerify.Transfer.Caller.Model

/-! # `Caller`: single-worker invariant, absorbing-leak lemmas, safety theorem

Part of `TpuSyncVerify.Transfer.Caller`. -/

namespace TpuSyncVerify.Transfer.Caller

/-! ## Leaks are absorbing

Whatever the connector and policy: once `leakP` or `leakD` holds it holds in
every later state. The publications are the only events that free a held
block without the engine's own progress, and each happens once. -/

/-- The D-side update leaves the P-side fields alone. -/
theorem applyLoadError_p (p : Policy) (s : S) :
    (applyLoadError p s).pHeld = s.pHeld ∧ (applyLoadError p s).sendPub = s.sendPub := by
  unfold applyLoadError
  split <;> cases p <;> simp

theorem applyFinishedRecving_p (s : S) :
    (applyFinishedRecving s).pHeld = s.pHeld ∧ (applyFinishedRecving s).sendPub = s.sendPub := by
  unfold applyFinishedRecving
  split <;> simp

theorem applyRecv_p (p : Policy) (sig : Signal) (s : S) :
    (applyRecv p sig s).pHeld = s.pHeld ∧ (applyRecv p sig s).sendPub = s.sendPub := by
  unfold applyRecv
  split <;> split <;> simp [applyFinishedRecving_p, applyLoadError_p]

theorem leakP_absorbing {c : Cfg} {s s' : S} {e : Ev}
    (h : leakP s = true) (hs : step c s e = some s') : leakP s' = true := by
  have hp : s.sendPub.isSome = true ∧ s.pHeld = true := by
    simpa [leakP] using h
  cases e with
  | pOutcome ok =>
    simp only [step, pOutcome] at hs
    split at hs
    · rename_i hn
      rw [hn] at hp
      simp at hp
    · simp at hs
  | dOutcome ok =>
    simp only [step, dOutcome] at hs
    split at hs
    · simp only [Option.some.injEq] at hs
      subst hs
      simp [leakP, applyRecv_p, hp]
    · simp at hs
  | promote =>
    simp only [step, promote] at hs
    split at hs
    · split at hs <;> simp only [Option.some.injEq] at hs <;> subst hs <;> simpa [leakP] using hp
    · simp at hs
  | finish =>
    simp only [step, finish] at hs
    split at hs
    · simp only [Option.some.injEq] at hs; subst hs; simpa [leakP] using hp
    · simp only [Option.some.injEq] at hs; subst hs; simpa [leakP] using hp
    · simp only [Option.some.injEq] at hs; subst hs; simpa [leakP] using hp
    · simp at hs

theorem leakD_absorbing {c : Cfg} {s s' : S} {e : Ev}
    (h : leakD s = true) (hs : step c s e = some s') : leakD s' = true := by
  cases e with
  | pOutcome ok =>
    simp only [step, pOutcome] at hs
    split at hs
    · simp only [Option.some.injEq] at hs
      subst hs
      simp only [applySend]
      split <;> simpa [leakD] using h
    · simp at hs
  | dOutcome ok =>
    simp only [step, dOutcome] at hs
    split at hs
    · rename_i hn
      simp [leakD, hn] at h
    · simp at hs
  | promote =>
    simp only [step, promote] at hs
    split at hs
    · rename_i hw
      simp [leakD, hw.1, hw.2] at h
    · simp at hs
  | finish =>
    simp only [step, finish] at hs
    split at hs
    · simp only [Option.some.injEq] at hs; subst hs; simp_all [leakD]
    · simp only [Option.some.injEq] at hs; subst hs; simp_all [leakD]
    · simp only [Option.some.injEq] at hs; subst hs; simp_all [leakD]
    · simp at hs

/-! ## Any connector meeting the contract is safe -/

/-- The inductive invariant for a contract-abiding connector. -/
structure Inv (s : S) : Prop where
  /-- P holds only while the send is unpublished. -/
  p : s.pHeld = true → s.sendPub = none
  /-- Before the receive is published nothing has happened on D. -/
  d_none : s.recvPub = none →
    (s.dStatus = .waitingRemoteKv ∨ s.dStatus = .finished) ∧ s.dFinRecv = false ∧ s.dFailRecv = false
  /-- After publication a parked request is promotable. -/
  d_wait : s.recvPub ≠ none → s.dStatus = .waitingRemoteKv → s.dFinRecv = true
  /-- After a failed publication a parked request is marked for recomputation. -/
  d_fail : s.recvPub = some false → s.dStatus = .waitingRemoteKv → s.dFailRecv = true
  /-- After publication a finished request holds nothing. -/
  d_fin : s.recvPub ≠ none → s.dStatus = .finished → s.dHeld = false
  /-- A recomputing request holds nothing. -/
  d_rec : s.dStatus = .recomputing → s.dHeld = false
  /-- Only a receive published as done decodes. -/
  d_dec : s.dStatus = .decoding → s.recvPub = some true

theorem inv_init : Inv init := by
  constructor <;> simp [init]

theorem inv_safe {s : S} (h : Inv s) : Safe s := by
  simp only [Safe, violates, leakP, leakD, decodesUnwritten, Bool.or_eq_false_iff,
    Bool.and_eq_false_iff]
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · cases hp : s.sendPub with
    | none => simp
    | some b =>
      right
      cases hh : s.pHeld
      · rfl
      · have := h.p hh
        rw [hp] at this
        cases this
  · cases hr : s.recvPub with
    | none => simp
    | some b =>
      right
      have hne : s.recvPub ≠ none := by simp [hr]
      cases hst : s.dStatus <;> simp
      · left
        exact h.d_wait hne hst
      · exact h.d_fin hne hst
  · cases hst : s.dStatus <;> simp
    have := h.d_dec hst
    simp [this]

theorem inv_pOutcome {c : Cfg} (hc : Contract c.conn) {s s' : S} {ok : Bool}
    (h : Inv s) (hs : pOutcome c ok s = some s') : Inv s' := by
  unfold pOutcome at hs
  split at hs
  · rename_i hn
    have hsend := hc.send_total ok
    simp only [applySend, hsend, ite_true, Option.some.injEq] at hs
    subst hs
    exact {
      p := by simp
      d_none := by simpa using h.d_none
      d_wait := by simpa using h.d_wait
      d_fail := by simpa using h.d_fail
      d_fin := by simpa using h.d_fin
      d_rec := by simpa using h.d_rec
      d_dec := by simpa using h.d_dec }
  · simp at hs

theorem inv_dOutcome {c : Cfg} (hc : Contract c.conn) {s s' : S} {ok : Bool}
    (h : Inv s) (hs : dOutcome c ok s = some s') : Inv s' := by
  unfold dOutcome at hs
  split at hs
  · rename_i hn
    obtain ⟨hst, hfin, hfail⟩ := h.d_none hn
    have hfr := hc.recv_total ok
    have hpn : s.pHeld = true → s.sendPub = none := h.p
    simp only [Option.some.injEq] at hs
    subst hs
    -- the P field is untouched by every branch below
    have hle : ok = false → (c.conn.onRecv ok).loadError = true := by
      intro hok; subst hok; exact hc.recv_failure_reported
    cases hst with
    | inl hw =>
      cases hl : (c.conn.onRecv ok).loadError with
      | false =>
        have hok : ok = true := by
          cases ok
          · exact absurd (hle rfl) (by simp [hl])
          · rfl
        subst hok
        simp only [applyRecv, hl, hfr, applyFinishedRecving, hw, ite_true]
        exact {
          p := by simpa using hpn
          d_none := by simp
          d_wait := by simp
          d_fail := by simp
          d_fin := by simp
          d_rec := by simp
          d_dec := by simp }
      | true =>
        cases hp : c.policy with
        | recompute =>
          simp only [applyRecv, hl, hfr, applyLoadError, applyFinishedRecving, hw, ite_true]
          exact {
            p := by simpa using hpn
            d_none := by simp
            d_wait := by simp
            d_fail := by simp
            d_fin := by simp
            d_rec := by simp
            d_dec := by simp }
        | fail =>
          simp only [applyRecv, hl, hfr, applyLoadError, applyFinishedRecving, hw, ite_true]
          exact {
            p := by simpa using hpn
            d_none := by simp
            d_wait := by simp
            d_fail := by simp
            d_fin := by simp
            d_rec := by simp
            d_dec := by simp }
    | inr hf =>
      -- aborted before the outcome: the load error is ignored, the completion frees
      cases hl : (c.conn.onRecv ok).loadError
      · simp only [applyRecv, hl, hfr, applyFinishedRecving, hf, ite_true]
        exact {
          p := by simpa using hpn
          d_none := by simp
          d_wait := by simp
          d_fail := by simp
          d_fin := by simp
          d_rec := by simp
          d_dec := by simp }
      · simp only [applyRecv, hl, hfr, applyLoadError, applyFinishedRecving, hf, ite_true]
        exact {
          p := by simpa using hpn
          d_none := by simp
          d_wait := by simp
          d_fail := by simp
          d_fin := by simp
          d_rec := by simp
          d_dec := by simp }
  · simp at hs

theorem inv_promote {s s' : S} (h : Inv s) (hs : promote s = some s') : Inv s' := by
  unfold promote at hs
  split at hs
  · rename_i hw
    obtain ⟨hw, hfin⟩ := hw
    -- the publication has happened: otherwise `dFinRecv = false`
    have hpub : s.recvPub ≠ none := by
      intro hn
      have := (h.d_none hn).2.1
      rw [hfin] at this
      cases this
    split at hs
    · rename_i hfail
      simp only [Option.some.injEq] at hs
      subst hs
      exact {
        p := by simpa using h.p
        d_none := by intro hn; exact absurd hn hpub
        d_wait := by simp
        d_fail := by simp
        d_fin := by simp
        d_rec := by simp
        d_dec := by simp }
    · rename_i hfail
      simp only [Option.some.injEq] at hs
      subst hs
      have hok : s.recvPub = some true := by
        cases hr : s.recvPub with
        | none => exact absurd hr hpub
        | some b =>
          cases b
          · have := h.d_fail hr hw
            simp [this] at hfail
          · rfl
      exact {
        p := by simpa using h.p
        d_none := by intro hn; exact absurd hn hpub
        d_wait := by simp
        d_fail := by simp
        d_fin := by simp
        d_rec := by simp
        d_dec := by simpa using hok }
  · simp at hs

theorem inv_finish {s s' : S} (h : Inv s) (hs : finish s = some s') : Inv s' := by
  unfold finish at hs
  split at hs
  · rename_i hw
    simp only [Option.some.injEq] at hs
    subst hs
    exact {
      p := by simpa using h.p
      d_none := by intro _; simp
      d_wait := by simp
      d_fail := by simp
      d_fin := by
        intro hpub _
        have := h.d_wait hpub hw
        simp [this]
      d_rec := by simp
      d_dec := by simp }
  · rename_i hd
    simp only [Option.some.injEq] at hs
    subst hs
    exact {
      p := by simpa using h.p
      d_none := by
        intro hn
        have := h.d_dec hd
        rw [hn] at this
        cases this
      d_wait := by simp
      d_fail := by simp
      d_fin := by simp
      d_rec := by simp
      d_dec := by simp }
  · rename_i hr
    simp only [Option.some.injEq] at hs
    subst hs
    have hheld := h.d_rec hr
    exact {
      p := by simpa using h.p
      d_none := by
        intro hn
        have := (h.d_none hn).1
        simp [hr] at this
      d_wait := by simp
      d_fail := by simp
      d_fin := by intro _ _; simpa using hheld
      d_rec := by simp
      d_dec := by simp }
  · simp at hs

theorem step_inv {c : Cfg} (hc : Contract c.conn) {s s' : S} {e : Ev}
    (h : Inv s) (hs : step c s e = some s') : Inv s' := by
  cases e with
  | pOutcome ok => exact inv_pOutcome hc h hs
  | dOutcome ok => exact inv_dOutcome hc h hs
  | promote => exact inv_promote h hs
  | finish => exact inv_finish h hs

/-- A connector that forwards every outcome and reports a failed receive on
the load-failure channel never leaks a transfer-held block on either node and
never lets D decode on unwritten KV, under either failure policy. -/
theorem contract_safe {c : Cfg} (hc : Contract c.conn) {s : S}
    (hr : (sys c).Reachable s) : Safe s :=
  inv_safe ((sys c).reachable_induction inv_init (fun _ _ _ h hs => step_inv hc h hs) hr)

theorem twoChannel_safe {policy : Policy} {s : S}
    (hr : (sys ⟨connectorTwoChannel, policy⟩).Reachable s) : Safe s :=
  contract_safe twoChannel_contract hr

/-- On a waiting request, staging an early `invalid_block_ids` under
`Policy.fail` (`vllm-torchtpu`'s `_stage3_fail_uncertain_loads_early` in
`tpu_connector.py` `_apply_stage3_submit_outcome`) produces the exact same
scheduler state as an API-layer abort (`finish`): the request is marked
`.finished` with its block free deferred (`dHeld := s.dHeld && !s.dFinRecv`)
until `finished_recving` arrives. -/
theorem earlyLoadError_fail_eq_finish {s : S} (hw : s.dStatus = .waitingRemoteKv) :
    some (applyLoadError .fail s) = finish s := by
  simp [applyLoadError, finish, hw]

/-- While the receive is unpublished (`s.recvPub = none`), an early
`applyLoadError .fail` on a waiting request preserves `Inv` and keeps
`dHeld = s.dHeld` (so destination HBM is never freed while the armed
`ReshardReceiveSession` is still live). -/
theorem earlyLoadError_fail_holds_blocks {s : S}
    (h : Inv s) (hn : s.recvPub = none) (hw : s.dStatus = .waitingRemoteKv) :
    Inv (applyLoadError .fail s) ∧ (applyLoadError .fail s).dHeld = s.dHeld := by
  have hfin : s.dFinRecv = false := (h.d_none hn).2.1
  have hstep : finish s = some (applyLoadError .fail s) :=
    (earlyLoadError_fail_eq_finish hw).symm
  exact ⟨inv_finish h hstep, by simp [applyLoadError, hw, hfin]⟩

end TpuSyncVerify.Transfer.Caller

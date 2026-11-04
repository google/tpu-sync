import TpuSyncVerify.Transfer.PrefillDecode.Receive.Invariant

/-! # `Recv`: frame lemmas for the composing modules

Part of `TpuSyncVerify.Transfer.PrefillDecode.Receive`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

namespace Recv

/-! ## Frame lemmas

What each event leaves alone, and the exact spec for `h2dReady`, which the
composed transfer model (`Pipeline.lean`) attaches a memory effect to. Each is
proved by unfolding `step` for every event and splitting every branch. -/

/-- Unfold `step` for a known event and split every branch, leaving `hs` as
`some … = some s'` or, for `beginOp` branches, the `Option.map` form. -/
macro "recv_cases" hs:ident : tactic =>
  `(tactic| (simp only [step, pushBegin, pushEnd, pullReply, h2dBegin, h2dIssue, h2dReady, h2dDone,
      netAccount, pollReady, cancel, publish] at $hs:ident <;> (repeat' split at $hs:ident)))

theorem step_numLayers {s s' : Recv} {e : Ev} (hs : step s e = some s') :
    s'.numLayers = s.numLayers := by
  cases e <;> recv_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> (repeat' split) <;> rfl)

theorem step_done_mono {s s' : Recv} {e : Ev} (hs : step s e = some s') (hd : s.life.done = true) :
    s'.life.done = true := by
  cases e <;> simp only [step, pushBegin, pushEnd, pullReply, h2dBegin, h2dIssue, h2dReady, h2dDone,
      netAccount, pollReady, cancel, publish, Lifecycle.beginOp_of_done hd, Option.map_none] at hs <;>
    (repeat' split at hs) <;> cases hs <;>
    simp [hd, Lifecycle.endOpLocked_done_mono, Lifecycle.finishLocked_done_mono]

theorem step_published_mono {s s' : Recv} {e : Ev} {b : Bool} (hs : step s e = some s')
    (hp : s.published = some b) : s'.published = some b := by
  cases e <;> recv_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; simpa using hp)
    | (cases hs <;> (repeat' split) <;> simp_all)

/-- Only `h2dReady` increments `ready`. -/
theorem step_ready {s s' : Recv} {e : Ev} (hs : step s e = some s') (he : e ≠ .h2dReady) :
    s'.ready = s.ready := by
  cases e <;> (try exact absurd rfl he) <;> recv_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> (repeat' split) <;> rfl)

theorem h2dReady_spec {s s' : Recv} (hs : step s .h2dReady = some s') :
    s.ready < s.issued ∧ s' = { s with ready := s.ready + 1 } := by
  simp only [step, h2dReady] at hs
  split at hs
  · cases hs; exact ⟨‹_›, rfl⟩
  · cases hs

theorem step_pending_issued {s s' : Recv} {e : Ev} (hs : step s e = some s')
    (hb : e ≠ .h2dBegin) (hi : ∀ ok, e ≠ .h2dIssue ok) :
    s'.pending = s.pending ∧ s'.issued = s.issued := by
  cases e <;> (try exact absurd rfl hb) <;> (try exact absurd rfl (hi _)) <;> recv_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩)
    | (cases hs <;> (repeat' split) <;> exact ⟨rfl, rfl⟩)

theorem h2dBegin_spec {s s' : Recv} (hs : step s .h2dBegin = some s') :
    s.issued + s.pending < s.numLayers ∧ s'.pending = s.pending + 1 ∧ s'.issued = s.issued := by
  simp only [step, h2dBegin] at hs
  split at hs
  · simp only [Option.map_eq_some_iff] at hs
    obtain ⟨l, _, rfl⟩ := hs
    exact ⟨‹_›, rfl, rfl⟩
  · cases hs

theorem h2dIssue_spec {s s' : Recv} {ok : Bool} (hs : step s (.h2dIssue ok) = some s') :
    s.pending ≠ 0 ∧ s'.pending = s.pending - 1 ∧
    s'.issued = (if s.life.done || s.life.draining || !ok then s.issued else s.issued + 1) := by
  simp only [step, h2dIssue] at hs
  split at hs
  · cases hs
  · rename_i hp
    split at hs
    · rename_i hc
      cases hs; exact ⟨hp, rfl, by simp [hc]⟩
    · rename_i hc
      cases ok with
      | false => cases hs; exact ⟨hp, rfl, by simp⟩
      | true =>
        cases hs
        exact ⟨hp, rfl, by simp_all⟩

theorem step_pullPending_of_ne_pullReply {s s' : Recv} {e : Ev} (hs : step s e = some s')
    (he : ∀ ok, e ≠ .pullReply ok) : s'.pullPending = s.pullPending := by
  cases e <;> (try exact absurd rfl (he _)) <;> recv_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; rfl)
    | (cases hs <;> (repeat' split) <;> rfl)

/-- Events outside `{pushBegin, pushEnd, h2dDone}` leave `pushes` and
`retired` unchanged. -/
theorem step_pushes_retired {s s' : Recv} {e : Ev} (hs : step s e = some s')
    (hpb : e ≠ .pushBegin) (hpe : e ≠ .pushEnd) (hhd : ∀ ok, e ≠ .h2dDone ok) :
    s'.pushes = s.pushes ∧ s'.retired = s.retired := by
  cases e <;>
    (try exact absurd rfl hpb) <;> (try exact absurd rfl hpe) <;>
    (try exact absurd rfl (hhd _)) <;> recv_cases hs <;>
  first
    | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; exact ⟨rfl, rfl⟩)
    | (cases hs <;> (repeat' split) <;> exact ⟨rfl, rfl⟩)


end Recv

end TpuSyncVerify.Transfer.PrefillDecode

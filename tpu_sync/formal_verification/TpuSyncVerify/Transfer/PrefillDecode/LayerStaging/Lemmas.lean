import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Model

/-! # `LayerStaging`: `withSlot` helpers, stuttering projection to `Pipeline`, and phase counts

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (Lifecycle StagingLayerManager)

/-! ## 1. `withSlot` / `withSlotOpt` combinator lemmas -/

theorem withSlot_eq {α : Type} {mgr : StagingLayerManager} {slots : List α} {l slot : Nat}
    {base : BoundedPipeline} {k : Nat → BoundedPipeline}
    (hlk : mgr.lookup l = some slot) (hslt : slot < slots.length) :
    withSlot mgr slots l base k = k slot := by simp [withSlot, hlk, hslt]

theorem withSlotOpt_eq {α : Type} {mgr : StagingLayerManager} {slots : List α} {l slot : Nat}
    {base : BoundedPipeline} {k : Nat → Option BoundedPipeline}
    (hlk : mgr.lookup l = some slot) (hslt : slot < slots.length) :
    withSlotOpt mgr slots l base k = k slot := by simp [withSlotOpt, hlk, hslt]

theorem withSlot_cases {α : Type} {mgr : StagingLayerManager} {slots : List α} {l : Nat}
    {base : BoundedPipeline} {k : Nat → BoundedPipeline} {P : BoundedPipeline → Prop}
    (hk : ∀ slot, mgr.lookup l = some slot → slot < slots.length → P (k slot))
    (hf : P { base with slotFault := true }) :
    P (withSlot mgr slots l base k) := by
  unfold withSlot
  cases hlk : mgr.lookup l with
  | none => exact hf
  | some slot =>
    if hslt : slot < slots.length then simpa [hslt] using hk slot hlk hslt
    else simpa [hslt] using hf

/-! ## 2. Stuttering simulation from `BoundedPipeline` to `Pipeline` -/

/-- Every `LayerStaging.step` advances `s.pipe` by a `StepsOf` chain of `Pipeline.step`s. -/
theorem step_pipe_steps {s s' : BoundedPipeline} {e : Ev} (hs : step s e = some s') :
    System.StepsOf Pipeline.step s.pipe s'.pipe := by
  unfold step at hs; split at hs
  · simp only [pipeStep, sendD2hBegin, sendD2hIssue, sendWake, sendH2hIssue, d2hReady, h2hDone,
      recvPushBegin, recvPushEnd, land, h2dBegin, h2dIssue, h2dReady,
      withSlot, withSlotOpt, syncPipe] at hs
    repeat' first
      | exact .single hs
      | (rw [Option.map_eq_some_iff] at hs; rcases hs with ⟨_, hs, hpost⟩;
         (repeat' split at hpost); all_goals subst hpost)
      | (rw [Option.some.injEq] at hs; subst hs; exact .refl _)
      | split at hs | cases hs
  · simp only [sendAcquireBegin, syncPipe] at hs; split at hs <;> try cases hs
    split at hs
    · rcases Option.map_eq_some_iff.mp hs with ⟨_, hp, rfl⟩; exact .single hp
    · cases hs; exact .refl _
  · simp only [sendAcquireDone, syncPipe] at hs; split at hs <;> try cases hs
    split at hs
    · cases hs
    · cases hs; exact .refl _
    all_goals rcases Option.map_eq_some_iff.mp hs with ⟨_, hp, rfl⟩; exact .single hp
  · simp only [recvAcquireBegin] at hs; (repeat' split at hs) <;> try cases hs
    all_goals rcases Option.map_eq_some_iff.mp hs with ⟨_, hp, rfl⟩; exact .single hp
  · simp only [recvAcquireWait, syncPipe] at hs; split at hs <;> try cases hs
    split at hs
    · cases hs
    · cases hs; exact .refl _
    · rcases Option.map_eq_some_iff.mp hs with ⟨_, hp, rfl⟩; exact .single hp
    all_goals
      rcases Option.map_eq_some_iff.mp hs with ⟨_, hb, rfl⟩
      rcases Option.bind_eq_some_iff.mp hb with ⟨_, hp₁, hp₂⟩
      exact .two hp₁ hp₂
  · simp only [recvAcquireFinish] at hs; split at hs <;> try cases hs
    rcases Option.map_eq_some_iff.mp hs with ⟨_, hp, hpost⟩
    split at hpost <;> subst hpost <;> exact .single hp
  · simp only [h2dDone] at hs; split at hs <;> try cases hs
    rcases Option.map_eq_some_iff.mp hs with ⟨_, hp, rfl⟩; exact .single hp

theorem step_pipe_inv {s s' : BoundedPipeline} {e : Ev}
    (hp : Pipeline.Inv s.pipe) (hs : step s e = some s') : Pipeline.Inv s'.pipe :=
  (step_pipe_steps hs).preserve (fun _ _ _ hi h1 => Pipeline.step_inv hi h1) hp

theorem reachable_pipe {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sys n ks kr).Reachable s) : (Pipeline.sys n).Reachable s.pipe :=
  (sys n ks kr).simulates_reachable (Pipeline.sys n) BoundedPipeline.pipe .init
    (fun _ _ _ => step_pipe_steps) h

theorem reachable_unregistered_pipe {n ks kr : Nat} {s : BoundedPipeline}
    (h : (sysUnregistered n ks kr).Reachable s) :
    (Pipeline.sysUnregistered n).Reachable s.pipe :=
  (sysUnregistered n ks kr).simulates_reachable (Pipeline.sysUnregistered n) BoundedPipeline.pipe .init
    (fun _ _ _ => step_pipe_steps) h

/-! ## 3. Lifecycle and cross-side frame helpers -/

theorem not_done_of_inFlight_pos {l : Lifecycle} (hc : l.Consistent) (hpos : 0 < l.inFlight) :
    l.done = false := hc.not_done_of_inFlight (by omega)

theorem beginOp_eq {l l' : Lifecycle} (h : l.beginOp = some l') :
    l.done = false ∧ l.draining = false ∧ l' = { l with inFlight := l.inFlight + 1 } := by
  unfold Lifecycle.beginOp at h; split at h <;> simp_all

theorem pipe_send_unchanged {p p' : Pipeline} {pe : Pipeline.Ev}
    (hp : Pipeline.step p pe = some p')
    (hns : ∀ se, pe ≠ .send se) (hnd : ∀ l, pe ≠ .d2hReady l) (hnh : ∀ l ok, pe ≠ .h2hDone l ok) :
    p'.numLayers = p.numLayers ∧ p'.send = p.send ∧
      p'.d2hReadyL = p.d2hReadyL ∧ p'.h2hRetiredL = p.h2hRetiredL := by
  cases pe <;>
    (try exact absurd rfl (hns _)) <;> (try exact absurd rfl (hnd _)) <;>
    (try exact absurd rfl (hnh _ _)) <;>
    simp only [Pipeline.step, Pipeline.notifyForRead, Pipeline.pullWait, Pipeline.recvStep,
      Pipeline.h2dBegin, Pipeline.h2dIssue, Pipeline.h2dReady, Pipeline.land,
      Pipeline.reclaim, Pipeline.reseatPrefillStaging, Pipeline.reseatDecodeStaging] at hp <;>
    (repeat' split at hp) <;>
    first
      | (rcases Option.map_eq_some_iff.mp hp with ⟨_, _, rfl⟩; exact ⟨rfl, rfl, rfl, rfl⟩)
      | (cases hp <;> exact ⟨rfl, rfl, rfl, rfl⟩)

theorem pipe_recv_unchanged {p p' : Pipeline} {pe : Pipeline.Ev}
    (hp : Pipeline.step p pe = some p')
    (hnr : ∀ re, pe ≠ .recv re) (hnl : ∀ l, pe ≠ .land l)
    (hnb : ∀ l, pe ≠ .h2dBegin l) (hni : ∀ l ok, pe ≠ .h2dIssue l ok)
    (hnd : ∀ l, pe ≠ .h2dReady l) :
    p'.numLayers = p.numLayers ∧ p'.recv = p.recv ∧ p'.landedL = p.landedL ∧
      p'.h2dPendingL = p.h2dPendingL ∧ p'.h2dIssuedL = p.h2dIssuedL ∧ p'.h2dReadyL = p.h2dReadyL := by
  cases pe <;>
    (try exact absurd rfl (hnr _)) <;> (try exact absurd rfl (hnl _)) <;>
    (try exact absurd rfl (hnb _)) <;> (try exact absurd rfl (hni _ _)) <;>
    (try exact absurd rfl (hnd _)) <;>
    simp only [Pipeline.step, Pipeline.notifyForRead, Pipeline.pullWait, Pipeline.sendStep,
      Pipeline.d2hReady, Pipeline.h2hDone, Pipeline.reclaim,
      Pipeline.reseatPrefillStaging, Pipeline.reseatDecodeStaging] at hp <;>
    (repeat' split at hp) <;>
    first
      | (rcases Option.map_eq_some_iff.mp hp with ⟨_, _, hpost⟩
         (repeat' split at hpost) <;> subst hpost <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩)
      | (cases hp <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩)

/-! ## 4. Consumer phase predicates and counts -/

def isAcquiringPhase : RecvLayerPhase → Bool
  | .waiting _ | .leasedInAcquire => true | _ => false

def isRetiredPhase : RecvLayerPhase → Bool
  | .h2dRetired => true | _ => false

def isWaitingPhase : RecvLayerPhase → Bool
  | .waiting _ => true | _ => false

def ownsRecvSlot : RecvLayerPhase → Bool
  | .waiting true | .leasedInAcquire | .leasedForStream | .h2dPending | .h2dRunning => true
  | _ => false

abbrev recvOwnsSlot (s : BoundedPipeline) (l : Nat) : Prop :=
  (s.recvPhaseL[l]?).map ownsRecvSlot = some true

notation "countAcquiring" => countPred isAcquiringPhase
notation "countRetired" => countPred isRetiredPhase
notation "countWaiting" => countPred isWaitingPhase

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

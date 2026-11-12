/-!
# Transition systems

A model in this library is a *guarded, executable* step function: `step s e`
is `some s'` when event `e` is enabled in state `s`, and `none` otherwise. One
definition then serves three purposes:

* **replay** — `run` checks a concrete trace, so a scenario can be confirmed
  or refuted by `decide`;
* **search** — `TpuSyncVerify.ModelCheck` explores small instances exhaustively;
* **proof** — `Reachable` and `reachable_induction` give the induction
  principle for proving an invariant on every reachable state.

Keeping the relation definitional in the step function means the guards used
by proofs are exactly the guards used by replay and search; they cannot drift
apart.
-/

namespace TpuSyncVerify

/-- A transition system: an initial state and a guarded step function. -/
structure System (S Ev : Type) where
  init : S
  step : S → Ev → Option S

namespace System

variable {S Ev : Type} (sys : System S Ev)

/-- Replay `evs` from `s`; `none` if some event along the way is disabled. -/
def runFrom (s : S) (evs : List Ev) : Option S :=
  evs.foldlM sys.step s

/-- Replay `evs` from the initial state. -/
def run (evs : List Ev) : Option S :=
  sys.runFrom sys.init evs

/-- States reachable from the initial state through enabled events. -/
inductive Reachable : S → Prop
  | init : Reachable sys.init
  | step {s s' : S} {e : Ev} : Reachable s → sys.step s e = some s' → Reachable s'

/-- An invariant holds on every reachable state once it holds initially and is
preserved by every enabled event. -/
theorem reachable_induction {P : S → Prop}
    (hinit : P sys.init)
    (hstep : ∀ s e s', P s → sys.step s e = some s' → P s')
    {s : S} (h : sys.Reachable s) : P s := by
  induction h with
  | init => exact hinit
  | step _ hse ih => exact hstep _ _ _ ih hse

/-- Replay never leaves the reachable states. -/
theorem runFrom_reachable {s : S} (hs : sys.Reachable s) :
    ∀ (evs : List Ev) {s' : S}, sys.runFrom s evs = some s' → sys.Reachable s' := by
  intro evs
  induction evs generalizing s with
  | nil =>
    intro s' h
    simp [runFrom] at h
    exact h ▸ hs
  | cons e evs ih =>
    intro s' h
    simp only [runFrom, List.foldlM_cons] at h
    cases hse : sys.step s e with
    | none => simp [hse] at h
    | some s₁ =>
      simp only [hse] at h
      exact ih (Reachable.step hs hse) h

/-- The end state of a successful replay from the initial state is reachable. -/
theorem run_reachable {evs : List Ev} {s : S} (h : sys.run evs = some s) :
    sys.Reachable s :=
  sys.runFrom_reachable Reachable.init evs h

/-- Replaying `evs₁ ++ evs₂` is the same as replaying `evs₁` then `evs₂`. -/
theorem runFrom_append :
    ∀ (evs₁ evs₂ : List Ev) {s s₁ s₂ : S},
      sys.runFrom s evs₁ = some s₁ →
      sys.runFrom s₁ evs₂ = some s₂ →
      sys.runFrom s (evs₁ ++ evs₂) = some s₂
  | [], evs₂, _, _, _, h₁, h₂ => by
    simp [runFrom] at h₁; subst h₁; exact h₂
  | e :: evs₁, evs₂, s, _, _, h₁, h₂ => by
    simp only [runFrom, List.cons_append, List.foldlM_cons] at h₁ ⊢
    cases hse : sys.step s e with
    | none => simp [hse] at h₁
    | some s' =>
      simp only [hse] at h₁ ⊢
      exact runFrom_append evs₁ evs₂ h₁ h₂

/-- Ranking-function argument for eventual settlement. If every state that
satisfies `Inv` but not `Done` has an enabled event that preserves `Inv` and
strictly decreases `rank`, then from any `Inv` state a finite replay reaches a
`Done` state (which still satisfies `Inv`). The models instantiate `Inv` with
their inductive invariant plus `draining`, `Done` with `done`, and `rank` with
a count of the operations still in flight; the per-model content is the
one-step lemma `hstep`. -/
theorem settles_of_rank {Inv Done : S → Prop} (rank : S → Nat)
    (hstep : ∀ s, Inv s → ¬ Done s →
      ∃ e s', sys.step s e = some s' ∧ Inv s' ∧ rank s' < rank s) :
    ∀ {s : S}, Inv s → ∃ evs s', sys.runFrom s evs = some s' ∧ Inv s' ∧ Done s' := by
  suffices aux : ∀ (k : Nat) {s : S}, rank s ≤ k → Inv s →
      ∃ evs s', sys.runFrom s evs = some s' ∧ Inv s' ∧ Done s' from
    fun {s} hs => aux (rank s) (Nat.le_refl _) hs
  intro k
  induction k with
  | zero =>
    intro s hk hs
    by_cases hd : Done s
    · exact ⟨[], s, rfl, hs, hd⟩
    · obtain ⟨_, _, _, _, hlt⟩ := hstep s hs hd
      omega
  | succ k ih =>
    intro s hk hs
    by_cases hd : Done s
    · exact ⟨[], s, rfl, hs, hd⟩
    · obtain ⟨e, s₁, hse, hs₁, hlt⟩ := hstep s hs hd
      obtain ⟨evs, s', hrun, hs', hd'⟩ := ih (by omega) hs₁
      refine ⟨e :: evs, s', ?_, hs', hd'⟩
      simp only [runFrom, List.foldlM_cons, hse]
      exact hrun

/-- Multi-step transition relation (reflexive-transitive closure of `step`,
independent of `init` so systems sharing `step` share `StepsOf`). -/
inductive StepsOf {S Ev : Type} (step : S → Ev → Option S) : S → S → Prop
  | refl (s : S) : StepsOf step s s
  | step {s₁ s₂ s₃ : S} {e : Ev} :
      step s₁ e = some s₂ → StepsOf step s₂ s₃ → StepsOf step s₁ s₃

theorem StepsOf.single {S Ev : Type} {step : S → Ev → Option S} {s₁ s₂ : S} {e : Ev}
    (h : step s₁ e = some s₂) : StepsOf step s₁ s₂ :=
  .step h (.refl _)

theorem StepsOf.two {S Ev : Type} {step : S → Ev → Option S} {s₁ s₂ s₃ : S} {e₁ e₂ : Ev}
    (h₁ : step s₁ e₁ = some s₂) (h₂ : step s₂ e₂ = some s₃) : StepsOf step s₁ s₃ :=
  .step h₁ (.single h₂)

theorem StepsOf.preserve {S Ev : Type} {step : S → Ev → Option S} {P : S → Prop}
    (hstep : ∀ s e s', P s → step s e = some s' → P s')
    {s s' : S} (hp : P s) (hs : StepsOf step s s') : P s' := by
  induction hs with
  | refl => exact hp
  | step h1 _ ih => exact ih (hstep _ _ _ hp h1)

theorem Reachable.steps {s s' : S} (hr : sys.Reachable s) (hs : StepsOf sys.step s s') :
    sys.Reachable s' := by
  induction hs with
  | refl => exact hr
  | step h1 _ ih => exact ih (hr.step h1)

/-- Stuttering simulation: if `proj` maps `sub.init` into an `abs`-reachable
state and maps every `sub.step` into zero or more `abs.step`s (`StepsOf abs.step`),
then every `sub`-reachable state projects to an `abs`-reachable state. -/
theorem simulates_reachable {S₁ Ev₁ S₂ Ev₂ : Type}
    (sub : System S₁ Ev₁) (abs : System S₂ Ev₂) (proj : S₁ → S₂)
    (hinit : abs.Reachable (proj sub.init))
    (hstep : ∀ s e s', sub.step s e = some s' → StepsOf abs.step (proj s) (proj s'))
    {s : S₁} (hr : sub.Reachable s) : abs.Reachable (proj s) :=
  sub.reachable_induction (P := fun s => abs.Reachable (proj s)) hinit
    (fun _ _ _ h hs => Reachable.steps abs h (hstep _ _ _ hs)) hr

end System

end TpuSyncVerify

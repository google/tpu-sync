import TpuSyncVerify.Common.System

/-!
# Bounded model checking

Breadth-first search over a `System` for a state satisfying `bad`, keeping one
witness trace per state. Used as a cheap sanity check on a model (does the
scenario we expect to be bad actually show up? does an alternative remove it?)
before investing in an inductive proof, and as an exhaustive check for models
whose state space is small enough to exhaust.

Closure (`safe`) is relative to the supplied `events` list, so it speaks for
the whole system only when `events` lists every enabled event instance;
`outOfFuel` checks only up to depth `fuel`.

Results are evaluated (`#guard`, `#eval`), not proved: no theorem here relates
`check` to `System.Reachable`. Where a property is also proved by induction
(`reachable_safe`), that theorem is the proof and the search is a cross-check.
-/

namespace TpuSyncVerify.ModelCheck

/-- Outcome of a bounded search. -/
inductive Result (Ev : Type) where
  /-- The search closed: every state reachable from `init` through the supplied
  `events` was visited, within the depth bound, and none is bad. -/
  | safe
  /-- No bad state within the depth bound; the states reachable through the
  supplied `events` were not exhausted. -/
  | outOfFuel
  /-- A trace from the initial state to a bad state. -/
  | counterexample (trace : List Ev)
  deriving Repr, DecidableEq

variable {S Ev : Type} [DecidableEq S]

private def dedupe (seen : List S) (fresh : List (S × List Ev)) : List (S × List Ev) :=
  (fresh.foldl (fun (acc : List S × List (S × List Ev)) p =>
    if acc.1.contains p.1 then acc else (p.1 :: acc.1, p :: acc.2)) (seen, [])).2

private def search (step : S → Ev → Option S) (events : List Ev) (bad : S → Bool) :
    Nat → List (S × List Ev) → List S → Result Ev
  | fuel, frontier, seen =>
    match frontier.find? (fun p => bad p.1) with
    | some (_, tr) => .counterexample tr.reverse
    | none =>
      match fuel with
      | 0 => .outOfFuel
      | fuel + 1 =>
        let next := frontier.flatMap fun (s, tr) =>
          events.filterMap fun e => (step s e).map fun s' => (s', e :: tr)
        let fresh := dedupe seen next
        if fresh.isEmpty then .safe
        else search step events bad fuel fresh (seen ++ fresh.map (·.1))

/-- Search `sys` for a state satisfying `bad`, trying every event in `events`
(and no other) at every state, to a depth of `fuel` events. Every state at
depth at most `fuel` is tested for `bad` before the fuel is consulted. -/
def check (sys : System S Ev) (events : List Ev) (bad : S → Bool) (fuel : Nat := 32) :
    Result Ev :=
  search sys.step events bad fuel [(sys.init, [])] [sys.init]

end TpuSyncVerify.ModelCheck

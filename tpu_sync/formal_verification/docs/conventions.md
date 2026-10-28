# Modelling conventions

Every model under `TpuSyncVerify/` is built the same way and laid out in the
same order. This page is the template for writing a new one and the key for
reading an existing one.

* Writing a model: [The shape of a model](#the-shape-of-a-model),
  [How a module file is laid out](#how-a-module-file-is-laid-out),
  [Staying faithful to the C++](#staying-faithful-to-the-c).
* Proving: [Properties and proofs](#properties-and-proofs),
  [Evidence that the proof means something](#evidence-that-the-proof-means-something).
* Changing C++ that a model refers to: [Citing the C++](#citing-the-c).

## The shape of a model

A model is a transition system: an initial state and one function that says
what each event does to a state.

```lean
System S Ev := ⟨init, step⟩      -- Common/System.lean
step : S → Ev → Option S         -- none means "this event is not enabled here"
```

Nothing else defines the behaviour. `Reachable` is the closure of `step` from
`init`; `run` replays a list of events through `step`; `ModelCheck.check`
searches the same `step`. There is no separate inductive `Step` relation whose
guards could drift from the function. Three things follow:

* a concrete scenario is a theorem about `run`, proved by `decide`;
* a bounded search and an inductive proof talk about the same system;
* a **mutant** — a variant of the model with one guard removed or one
  ordering changed — is just a different `step`:
  `⟨init, fun s e => match e with | .x => mutated s | e => step s e⟩`.

Events are named in the voice of the C++ (`h2dBegin`, `pullReply`, `cancel`),
not of the property, and carry their outcome as an argument (`h2dDone (ok :
Bool)`, `acquireReply ok`) instead of being split into a success and a
failure constructor. Event lists stay short, and one C++ callback maps to one
row of the events table.

## How a module file is laid out

Modules are grouped by subsystem (`Transfer/…`, `Controller/…`), with shared
machinery in `Common/`. Each model file has these sections, in this order:

1. **Module docstring** — what is modelled; abbreviations for cited files; the
   **State** table (field → C++ → role); the **Events** table (event → C++
   symbol or code fragment); the **Lock scopes** table; the **Source map** (see
   [Citing the C++](#citing-the-c)); **Assumptions**; **Properties**, each
   stated in words with the C++ promise it corresponds to.
2. `structure S` and `init`.
3. `inductive Ev`; one `def` per event (named so it does not collide with a
   field of `S`, or dot notation breaks); `step`; `sys`.
4. Property definitions and `Safe`.
5. `Inv`, `inv_init`, `inv_safe`, `step_inv`, `reachable_safe`; progress.
6. Frame lemmas other modules need.
7. Replay traces, `violates`, `#guard` searches, mutants.

The next three chapters say how to write them: sections 1–4 under
[Staying faithful to the C++](#staying-faithful-to-the-c), section 5 under
[Properties and proofs](#properties-and-proofs), section 7 under
[Evidence that the proof means something](#evidence-that-the-proof-means-something).

A module over roughly a thousand lines is split along these sections into an
umbrella file and a directory of parts:

* `X.lean` — the module docstring and source map; re-exports the parts, so
  `import …X` still brings in the whole model.
* `X/Model.lean` — state, events, `step`, properties.
* `X/Lemmas.lean` — structural lemmas, where needed.
* `X/Invariant.lean` — `Inv` and `reachable_safe`.
* `X/Progress.lean` — the drain measure and settlement.
* `X/Frame.lean` — frame lemmas for other modules.
* `X/Traces.lean` — traces, searches, mutants.

Modules with their own internal numbering (`BlockOrdering`, `PoolStaging`)
name their parts after their sections instead.

## Staying faithful to the C++

**Transcribe, do not paraphrase.** The settle logic the proofs rest on
(`FinishLocked`, `EndRecvOpLocked`, `TryBeginRecvOp`) is transcribed into small
Lean functions in `Transfer/Session.lean`, and every event is composed from
them. No event re-derives `done` or `draining` by hand.

**One event per race window.** Where the C++ takes a value under one lock and
uses it under another, the model has two events (`h2dBegin` / `h2dIssue`), and
the state between them is a ghost counter.

**Lock scopes.** A Lean `step` runs as a single atomic transition, whereas the
C++ only holds locks over individual critical sections. Each module's **Lock
scopes** table (right after the events table) shows which C++ lock acquisitions
each event represents. Most events map to a single `absl::MutexLock(&mu_)` or a
single atomic read/write. When an event covers two lock scopes—or checks an
unlocked atomic before taking `mu_`—the table explains why nothing another
thread can do in the gap leads to a state outside the model (or cites the
assumption that prevents it). Where an interleaving between two lock scopes
matters, the model splits them into two events (as in **One event per race
window** above).

**Ghost fields.** A ghost field has no single C++ variable behind it. It
records where each unit of a counter came from (`pending`, `retired`,
`chaining`, …) or a fact about the environment (`reused`). Ghosts are what let
an invariant say "`in_flight_` equals the number of things that currently hold
an op". Every ghost is listed in the module's state table and marked `ghost`.

Prefer counters to indexed maps when identity does not matter: `Nat` keeps
`DecidableEq`, avoids `funext`, and lets the same model be both proved and
model-checked. Use indexed structure only where a property needs it
(`Pipeline` memories are a `List Cell` indexed by layer). When identity matters
but the C++ keeps only a counter, add a per-item `List Bool` ghost next to the
counter and tie them with an invariant (`countTrue set = counter`): the counter
is what the C++ checks, the set is what the proof reasons about.

**Assumptions.** Anything the proof trusts about code outside the model is an
assumption with a number (A1, A2, …), a C++ citation, and the guard that
encodes it. If violating the assumption would be a real bug, say so and, when
cheap, show the consequence as a mutant (`netAccountUnordered` for the receive
model's A4).

## Properties and proofs

**State observable outcomes.** Phrase each property the way its consumer would
read it — the caller of the C++ or the next Lean module — not as a fact about
the model's bookkeeping. For example, the receive model's promise to the
engine is "once `poll_stats()` reports `done_recving`, nothing writes that
request's decode HBM again", so its property is

```lean
DecodeHbmSafe s := s.published = some true → <decode HBM no longer changes>
```

and not `published ≠ none → pending = 0 ∧ retired = issued`. The second
statement is true and is *why* the first holds (every issued H2D op has
retired), but `pending`, `retired` and `issued` are ghost counters a C++
reader cannot map to anything. Counter equations like that belong in `Inv`,
the strengthening the induction needs; `Safe` is the conjunction of the
observable properties, and `reachable_safe` is the theorem to cite.

**Proof shape.** Proofs are `induction` on `Reachable` and `cases` on the
event, with a lemma per event where a case is long. Progress is a `drainRank`
measure plus a one-step `drain_step` lemma, discharged through
`System.settles_of_rank`.

**No `sorry`, no axioms, no Mathlib.**

## Evidence that the proof means something

Every module ends with checks that the proved system is the intended one:

* **Replay traces** — `trace_*` theorems that reach the interesting states
  (a publication, a deadline during a copy, the interleaving a re-check
  guards against) and traces that are correctly refused.
* **Search** — where the state space is finite, an exhaustive search for
  `violates` (`#guard ModelCheck.check … = .safe`). A depth-bounded search
  that ends in `.outOfFuel` is *not* a verification result; it is kept only
  where no theorem covers the property, marked `-- Bounded only`, and never
  placed next to a proved `reachable_safe` for the same system.
* **Mutants** — at least one per load-bearing guard, each shown to produce a
  counterexample, and named for what it breaks (`cancelEager`,
  `dispatchEarly`, `reseatAtFinish`).

**The search tests the proved predicate.** The theorems are about `Safe`; the
model checker needs a `Bool`. Give `Safe` a `Decidable` instance and define

```lean
def violates (s : S) : Bool := !decide (Safe s)
```

so a `#guard` search and `reachable_safe` are statements about one and the
same predicate (`violates_iff` records the equivalence). Never hand-write a
second boolean version of `Safe` for the checker — the two drift apart.

**Keep checks fast.** `decide` on a 20-event trace is instant and a mutant
search at fuel 6–8 takes under a second. Keep every `#guard` under about 5 s;
they all run on every `lake build`.

## Citing the C++

**Symbols in prose, line numbers in tables.** Prose, state/event tables, and
`def`/`theorem` docstrings cite C++ symbols (`Class::Method`), quoted code
fragments (`if (done_ || draining_)`), and GTest names (`Suite.TestName`).
Each module and subsystem doc carries one source-map table,

```
<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
...
<!-- source-map:end -->
```

whose `Lines` column is resolved from the C++ tree by `tools/resolve_anchors.py`:

* `--check` verifies every `Lines` cell against the working tree and reports
  stray raw citations outside source-map blocks;
* `--apply` rewrites the `Lines` cells from the current tree;
* `--diff <old> <new>` classifies each anchor between two commits as `same`,
  `shift` (moved, body identical), `modified` (body changed — reread the
  model) or `FAIL` (renamed or removed).

**One last-resolved commit.** The commit at which the spans were last resolved
is stated once, in `README.md`; modules and docs do not repeat it.

**Describe the code as it is.** Models and docs state what the current C++
does and what is proved about it. An old behaviour worth remembering is kept
only as a mutant.

# Modelling conventions

The models under `TpuSyncVerify/` follow a common structure and layout. This
page is the template for writing a new model and the key for reading an
existing one.

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
guards could drift from the function. Two things follow:

* a concrete scenario is a theorem about `run`, proved by `decide`;
* a bounded search and an inductive proof talk about the same system.

Events are named in the voice of the C++ (`h2dBegin`, `pullReply`, `cancel`),
not of the property, and carry their outcome as an argument (`h2dDone (ok :
Bool)`, `pullReply ok`) instead of being split into a success and a
failure constructor. Event lists stay short, and one C++ callback maps to one
row of the events table.

### Mutants and alternatives

A **mutant** drops or reorders a C++ guard to show that it matters; an
**alternative** adds a check the C++ lacks to show what it would buy.

Keep variants outside `step` as wrapper systems next to the traces so `step`,
`Safe`, and the theorems describe shipping C++ only. Parameterize `step` only
when theorems quantify over multiple variants (e.g. `ReceivePoll`,
`PoolStaging`), and name the shipping value in the module docstring.

Never encode an unverified assumption silently in `step`: number it under
**Assumptions**, name executions the C++ still permits `trace_*` rather than
`mutant_*`, and if a theorem excludes timing windows the C++ permits, state it
on an explicitly restricted system (e.g. `sysRestricted`).

## How a module file is laid out

Modules are grouped by transfer path (`Transfer/PrefillDecode/…`,
`Transfer/Reshard/…`), with shared machinery in `Common/` and `Transfer/`. Each
model file has these sections, in this order:

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

Include a **Lock scopes** table whenever the module's events correspond to C++
critical sections (pure-function helpers and thin compositions of another
module's events omit it).

A large module (roughly over a thousand lines) is split along these sections
into an umbrella file and a directory of parts (single-proof parts may still
exceed that):

* `X.lean` — the module docstring and source map; re-exports the parts, so
  `import …X` still brings in the whole model.
* `X/Model.lean` — state, events, `step`, properties.
* `X/Lemmas.lean` — structural lemmas, where needed.
* `X/Invariant.lean` — `Inv` and `reachable_safe`.
* `X/Progress.lean` — the drain measure and settlement.
* `X/Frame.lean` — frame lemmas for other modules.
* `X/Traces.lean` — traces, searches, mutants.
Where these names do not fit, name each part for what it holds (e.g., `Arena`,
`Registration`, `Preservation`).

### Docstring and source-map budget

* **One home per mechanism:** explain each C++ mechanism or boundary once in
  its owning module or subsystem doc; cross-reference elsewhere in one line.
* **Compact umbrella docstrings:** aim for ~60–100 lines (excluding
  `## Source map`), with one-line State/Events rows and concise Lock scopes
  cells.
* **Short trace/mutant docstrings:** 1–3 lines (`Suite.TestName` plus what is
  asserted), without narrating event lists step by step.
* **Direct source maps:** anchor only directly modelled symbols and directly
  replayed tests in `.lean` files; leave cross-module routing to the subsystem
  docs.

## Staying faithful to the C++

**Transcribe, do not paraphrase.** The settle logic the proofs rest on
(`FinishLocked`, `EndRecvOpLocked`, `TryBeginOp`) is transcribed into small
Lean functions in `Transfer/Session.lean`, and every event is composed from
them. No event re-derives `done` or `draining` by hand.

**One event per race window.** Where the C++ takes a value under one lock and
uses it under another, the model has two events (`h2dBegin` / `h2dIssue`), and
the state between them is a ghost counter.

**Lock scopes.** A Lean `step` runs as one atomic transition, whereas the C++
only holds locks over individual critical sections. A module's **Lock scopes**
table (right after the events table) shows which C++ lock acquisitions each
event represents (usually one `absl::MutexLock(&mu_)` or atomic access). When
an event spans multiple lock scopes (or checks an unlocked atomic first), the
table either explains why interleavings in the gap are harmless or cites the
assumption that excludes them; where an interleaving matters, split it into two
events (**One event per race window**).

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
encodes it. Say who establishes it: another Lean module, C++ control flow, or
a caller or configuration the C++ does not check. If violating the assumption
would be a real bug, say so and, when cheap, show the consequence as a mutant
(`netAccountUnordered` for the receive model's A4). If the C++ itself permits
executions outside the assumption, show one as a `trace_*` theorem. A mechanism
the model leaves out is named, with its C++ symbol, in the module's assumptions
or "not modelled" paragraph.

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

**No `sorry`, no added axioms, no Mathlib.** The corpus declares no `axiom`
and uses neither `native_decide` nor `bv_decide`, so every theorem depends on
at most Lean's three standard axioms (`propext`, `Classical.choice`,
`Quot.sound`); `#print axioms <theorem>` shows it. The only dependency is the
Lean core that ships with the toolchain.

A `theorem` is checked by Lean's kernel. A `#guard` is not a theorem: it
evaluates a `Bool` during the build and fails the build if the result is
`false`. Model-checker searches are `#guard`s.

## Evidence that the proof means something

Every module ends with checks that the proved system is the intended one:

* **Replay traces** — `trace_*` theorems that reach the interesting states
  (a publication, a deadline during a copy, the interleaving a re-check
  guards against) and traces that are correctly refused. A trace that replays a
  C++ or Python test names it in its docstring.
* **Search** — where the state space is finite, `#guard ModelCheck.check … = .safe`
  checks every state reachable from `init` through the supplied event list (an
  evaluated cross-check, not a kernel theorem; `reachable_safe` is the proof).
  A depth-bounded `.outOfFuel` search is not a verification result — mark it
  `-- Bounded only` and keep it only where no theorem applies or as a baseline
  for a mutant search.
* **Mutants** — for a guard the proof depends on, a variant without it, shown
  to produce a counterexample and named for what it breaks (`cancelEager`,
  `dispatchEarly`, `reseatAtFinish`).

**The search tests the proved predicate.** The theorems are about `Safe`; the
model checker needs a `Bool`. Give `Safe` a `Decidable` instance and define

```lean
def violates (s : S) : Bool := !decide (Safe s)
```

so a `#guard` search and `reachable_safe` test the same predicate
(`violates_iff` records the equivalence). Never hand-write a second boolean
version of `Safe` — the two drift apart. A search targeting a single conjunct or
mutant-only property uses its own boolean named for that property (e.g.
`violatesExclusive`).

**Keep checks fast.** Every `#guard` runs on every `lake build`; keep each
under about 5 s.

## Citing the C++

**Symbols in prose, line numbers in tables.** Prose, state/event tables, and
`def`/`theorem` docstrings cite C++ symbols (`Class::Method`), quoted code
fragments (`if (done_ || draining_)`), and GTest names (`Suite.TestName`).
Line numbers appear only in source-map tables,

```
<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
...
<!-- source-map:end -->
```

A split module keeps its table in the umbrella file `X.lean` and its parts
carry none; a single-file module carries its own; each subsystem doc carries
one table per section of its C++ → Lean routing index. The `Lines` column is
resolved from the C++ tree by `tools/resolve_anchors.py`:

* `--check` verifies every `Lines` cell against the working tree, flags stray
  raw line citations outside source-map blocks, writes nothing, and exits
  non-zero on any mismatch;
* `--apply` rewrites `Lines` cells from the working tree (combine with `--check`
  to fail only on unresolved anchors or stray citations);
* `--diff <old> <new>` classifies each anchor as `same`, `shift` (moved,
  unchanged), `MODIFIED` (body changed — reread the model), `NEW`, or `FAIL`
  (renamed or removed).

A quoted fragment in prose is not checked by the tool; when the C++ it quotes
changes, `--diff` flags the enclosing symbol's row as `MODIFIED`, and the prose
must be reread against it.

**One last-resolved commit.** The commit at which the spans were last resolved
is stated once, in `README.md`; modules and docs do not repeat it.

**Describe the code as it is.** Models and docs state what the current C++
does and what is proved about it. An old behaviour worth remembering is kept
only as a mutant. Tests are cited as C++ tests (GTest `Suite.TestName`) or by
the Python test's name.

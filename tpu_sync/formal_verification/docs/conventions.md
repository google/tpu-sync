# Modelling conventions

The models under `TpuSyncVerify/` are built the same way and laid out in the
same order; where a module departs from a rule below, the rule names it. This
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

A **mutant** is a variant of the model with one C++ check removed or one
ordering changed, kept to show that the check matters. An **alternative** is a
variant that adds a check or changes a policy the C++ does not have, kept to
show what the change would and would not buy. Neither is the C++.

Most models keep them outside `step`, as a wrapper system next to the traces:

```lean
⟨init, fun s e => match e with | .x => mutated s | e => step s e⟩
```

so `step`, `Safe` and the inductive theorems describe the C++ only
(`Recv.cancelEager`, `Pipeline.dispatchEarly`, `UuidTable.stepOverwriteDraining`;
alternatives `PeerIsolation.stepPerPeerQuota`, `Transport.stepHardened`,
`SubPlanChain.WorkerState.stepSweepWaitsForOpen`).

Five models instead take the variant as a parameter of `step`, because their
theorems are stated for more than one value of it. In each, the value that is
the C++ is named in the module docstring and in the subsystem doc's
"Shipping C++ vs. modelled alternatives" table or section:

| Model | Parameter | The C++ | Other values |
|---|---|---|---|
| `PrefillDecode/ReceivePoll.lean` | `stepM (poll : Bool)` | `stepM true`: the poll in `CompleteReadWithDetails` finishes a ready receive | `stepM false` (alternative): only the last H2D callback finishes it; safety and settlement are proved for it too (`noPoll_safe`, `noPoll_can_settle`) |
| `Reshard/Receive` | `Recv.step (pol : PushEndPolicy := .finishOnError)` | `.finishOnError`, the default: `EndIncomingPush` calls `Finish(status)` on a failed push before `EndOp()` | `.endOpOnly` (mutant): `EndOp()` only. `Recv.reachable_safe` is proved for both values; `trace_failed_incoming_push` shows the difference (the mutant keeps staging until the deadline) |
| `Reshard/PoolStaging/Registration.lean` | `RegState.step (policy : RegistrationPolicy)` | `.reserveBeforeLease`: `PoolReshardRegisterRecv` claims the uuid before leasing staging (`reachable_reserveBeforeLease_safe`) | `.leaseBeforeReserve` (mutant): a counterexample trace, no theorem |
| `Caller/NumaBarrier.lean` | `numaStep (c : NumaCfg)`, field `numaPolicy : NumaPolicy := .settledCounts` | `.settledCounts`, the default: `process_completion` in `NumaAwareKVCacheManager::CompleteReadRaw` (`numa_settledCounts_safe`) | `.eagerFailSet` (mutant): counterexample traces (`trace_eagerFailSet_*`) |
| `Reshard/SubPlanChain` (worker) | `WorkerState.stepG (g : Guards)` | `WorkerState.step`: `stepG` with `noDrainingOkAtSweep` and `noSettleBetweenRetireAndSweep` off (see below) | `WorkerState.stepRestricted := stepG {}` (every guard on; `reachable_worker_safe`); `deadlineArm` off (mutant); see below |

`SubPlanChain`'s `Guards` is different in kind from the other four, and its
theorems must be read with that in mind. Its five fields are of three kinds:

* `deadlineArm` (A5a) is a check the C++ performs; turning it off is a mutant
  (`mutant_no_deadline_arm_reports_done_after_sweep`).
* `noSingleWhileSeated` (A6) and `oneSessionPerSlot` (A7) are assumptions about
  callers that nothing in the C++ checks. `WorkerState.step` carries them as
  guards, like A1 and A2. A registration outside one of them is still an
  execution of the C++ as it is, so the theorems that exhibit them are named
  `trace_*`, not `mutant_*` (`trace_single_plan_registration_erases_live_chain`,
  `trace_slot_overwrite_reports_stray_done`).
* `noDrainingOkAtSweep` (A5b) and `noSettleBetweenRetireAndSweep` (A5c) are not
  assumptions about callers: each excludes a timing window inside
  `CompleteReadWithDetails` / `SweepSubPlanChainsLocked` that the C++ permits
  with every caller well behaved. They are **not** guards of
  `WorkerState.step`.

`WorkerState.step` is therefore the C++ under its caller assumptions, and it
does not satisfy `DoneAfterAllSubPlans`
(`trace_draining_ok_at_sweep_reports_done_after_failure` and
`trace_mid_poll_endop_unanchored_reports_done_after_stall` are executions of
`WorkerState.sys`). `reachable_worker_safe` is stated on the explicitly
restricted system `WorkerState.sysRestricted` (`stepRestricted := stepG {}`,
A5b and A5c in addition), and is a statement about those executions only.

The `Caller` and `Transport` models also take a configuration record
(`Caller.Cfg` / `NumaCfg` with the connector under test, `Transport.Cfg` with
`honest`); those fields choose the environment, and the theorems state which
values they cover.

A new model should keep variants outside `step` unless a theorem has to
quantify over them, and should never encode an assumption the C++ does not
enforce as a silent guard: give it a number, say so in the Assumptions section,
and show an execution outside it (except where the state representation itself
cannot express the violation — e.g. per-pool cells in `Reshard/Pipeline`, A1
and A3 — in which case document the gap in the Assumptions section).

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

A Lock scopes table is present where the module's own events stand for C++
critical sections. It is absent from `Transfer/Session.lean` (the lifecycle
functions the events are composed from), from `PrefillDecode/BlockOrdering` and
`Reshard/Coverage` (block plans and byte coverage as pure functions, composed
with another model's events), from modules that extend or compose another
module's events (`PrefillDecode/ReceivePoll`, `PrefillDecode/MultiRequest`,
`PrefillDecode/PipelineChecks`), and from `Reshard/PoolStaging`, which
describes its `mu_` scopes in prose.

The next three chapters say how to write them: sections 1–4 under
[Staying faithful to the C++](#staying-faithful-to-the-c), section 5 under
[Properties and proofs](#properties-and-proofs), section 7 under
[Evidence that the proof means something](#evidence-that-the-proof-means-something).

A large module is split along these sections into an umbrella file and a
directory of parts. Most modules over about a thousand lines are split
(`Reshard/Registry.lean` is a single file of that size), and a part can itself be longer than that where
it is one proof (`Reshard/SubPlanChain/Invariant.lean`,
`Caller/NumaBarrier.lean`, `Caller/MultiWorker.lean`).

* `X.lean` — the module docstring and source map; re-exports the parts, so
  `import …X` still brings in the whole model.
* `X/Model.lean` — state, events, `step`, properties.
* `X/Lemmas.lean` — structural lemmas, where needed.
* `X/Invariant.lean` — `Inv` and `reachable_safe`.
* `X/Progress.lean` — the drain measure and settlement.
* `X/Frame.lean` — frame lemmas for other modules.
* `X/Traces.lean` — traces, searches, mutants.
Where these names do not fit, a part is named for what it holds: the sections
of `PrefillDecode/BlockOrdering` (`Validation`, `Coalescing`, `CopyPlan`,
`Cancellation`, `BlockPipeline`) and `Reshard/PoolStaging` (`Arena`,
`ArenaLemmas`, `Registration`), the split invariants and slot bounds of
`PrefillDecode/LayerStaging` (`SendInv`, `RecvInv`, `SlotBounds`),
`Reshard/Pipeline/Preservation`, and the two further caller models
`Caller/MultiWorker` and `Caller/NumaBarrier`.

## Staying faithful to the C++

**Transcribe, do not paraphrase.** The settle logic the proofs rest on
(`FinishLocked`, `EndRecvOpLocked`, `TryBeginOp`) is transcribed into small
Lean functions in `Transfer/Session.lean`, and every event is composed from
them. No event re-derives `done` or `draining` by hand.

**One event per race window.** Where the C++ takes a value under one lock and
uses it under another, the model has two events (`h2dBegin` / `h2dIssue`), and
the state between them is a ghost counter.

**Lock scopes.** A Lean `step` runs as a single atomic transition, whereas the
C++ only holds locks over individual critical sections. A module's **Lock
scopes** table (right after the events table) shows which C++ lock acquisitions
each event represents. Most events map to a single `absl::MutexLock(&mu_)` or a
single atomic read/write. When an event covers two lock scopes—or checks an
unlocked atomic before taking `mu_`—the table explains why nothing another
thread can do in the gap leads to a state outside the model (or cites the
assumption that prevents it). Where an interleaving between two lock scopes
matters, the model splits them into two events (as in **One event per race
window** above). Where an event folds two scopes whose intermediate state is
not a model state and the argument that this is harmless is made from the C++
and not proved, the module says so in those words (`PrefillDecode/LayerStaging`
LS2).

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
* **Search** — where the state space is finite, a search for `violates` that
  closes (`#guard ModelCheck.check … = .safe`). `.safe` means every state
  reachable from `init` *through the event list passed to the search* was
  visited and none violates; it speaks for the whole system only when that list
  enumerates every event instance, and it is an evaluated result, not a theorem
  (no theorem relates `ModelCheck.check` to `Reachable`). Where
  `reachable_safe` is proved, it is the proof and the search is a cross-check.
  A depth-bounded search that ends in `.outOfFuel` says only "no violation
  within that many events"; it is *not* a verification result. It is marked
  `-- Bounded only` and kept where no theorem covers the property (an
  alternative such as `Transport.stepHardened`) or as the baseline a mutant's
  counterexample search is compared against (`UuidTable`).
* **Mutants** — for a guard the proof depends on, a variant without it, shown
  to produce a counterexample and named for what it breaks (`cancelEager`,
  `dispatchEarly`, `reseatAtFinish`).

**The search tests the proved predicate.** The theorems are about `Safe`; the
model checker needs a `Bool`. Give `Safe` a `Decidable` instance and define

```lean
def violates (s : S) : Bool := !decide (Safe s)
```

so a `#guard` search and `reachable_safe` are statements about one and the
same predicate (`violates_iff` records the equivalence). Never hand-write a
second boolean version of `Safe` for the checker — the two drift apart. (The
`Caller` model goes the other way: its `Safe` is defined from the boolean.) A
search for one property that is not the module's `Safe` — a single conjunct, or
a property only a mutant or alternative is tested against — has its own
boolean, named for the property (`violatesSlotConservation`,
`violatesExclusive`, `violatesHoldLeases`, `violatesMetrics`, `violatesQuota`).

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

* `--check` verifies every `Lines` cell against the working tree, reports
  stray raw citations outside source-map blocks, and exits non-zero on any
  mismatch (or if it found no file to process); it writes nothing;
* `--apply` rewrites the `Lines` cells from the current tree; with `--check` as
  well, the cells are rewritten first and the run fails only on an anchor that
  does not resolve or a stray citation;
* `--diff <old> <new>` classifies each anchor between two commits as `same`,
  `shift` (moved, body identical), `MODIFIED` (body changed — reread the
  model), `NEW` (resolves at `<new>` but not at `<old>`) or `FAIL` (does not
  resolve at `<new>`: renamed or removed); in `--diff` mode, table `Lines`
  cells are not compared for `drift`.

A quoted fragment in prose is not checked by the tool; when the C++ it quotes
changes, `--diff` flags the enclosing symbol's row as `MODIFIED`, and the prose
must be reread against it.

**One last-resolved commit.** The commit at which the spans were last resolved
is stated once, in `README.md`; modules and docs do not repeat it.

**Describe the code as it is.** Models and docs state what the current C++
does and what is proved about it. An old behaviour worth remembering is kept
only as a mutant. Tests are cited as C++ tests (GTest `Suite.TestName`) or by
the Python test's name.

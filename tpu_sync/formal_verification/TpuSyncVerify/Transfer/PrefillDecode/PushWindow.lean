import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Model
import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Invariant
import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Progress
import TpuSyncVerify.Transfer.PrefillDecode.PushWindow.Traces

/-!
# A send's push chain over block groups, under a push window

Models the push chain of one `TransferSendSession` with push groups set
(`KVCacheManagerBase::SetPushGroups`): the send's units — one per layer, or one
per block group of a layer whose transfer exceeds the group size
(`TransferSendSession::PlanUnitsLocked`, `TransferSendSession::BlockGroupBounds`)
— are pushed in unit order by the chain `SendNextUnit` → `AdmitPush` →
`IssuePush`, with at most `push_window_` pushes in flight (`0`: no cap). A unit
whose copy completes while the window is full is parked (`parked_unit_`) and
issued by the push callback that frees a slot.

Abbreviations: `send.cc` = `tpu_sync/kv_cache/transfer_send_session.cc`,
`send.h` = `tpu_sync/kv_cache/transfer_send_session.h`, `base.h` =
`tpu_sync/kv_cache/kv_cache_manager_base.h`. Exact line ranges for every
symbol and sub-block are in the generated `## Source map` table below.

## What is modelled

`Send.lean` models the same session with one unit per layer and no window,
which is what a send runs without push groups (`push_groups()` returns
`group_bytes == 0` and `window == 0`): every unit is a layer and `AdmitPush` never parks,
so `Send.lean`'s `wake`, `h2hIssue`, `sendNext` and `h2hDone` are this model's
`wake`, `admit` then `issue`, `sendNext` and `h2hDone`. This model starts where
`Send.lean`'s `start` leaves the chain, `SendNextUnit(0)` armed, and takes the
D2H side as given (A1). What it adds is the window and the hand-off of a parked
unit from the chain to a push callback.

The host layer buffer that a grouped layer's units share under bounded host
layer staging is `GroupStaging.lean`'s; the receiving side of a grouped layer
is `Transport.lean`'s (stream accounting) and `GroupLanding.lean`'s (early H2D
copies).

## State

| Field | C++ | Role |
|---|---|---|
| `numUnits` | `units_.size()` (`PlanUnitsLocked`) | units in push order |
| `window` | `push_window_` (`push_window_ = push_groups.window;`, read once in `PlanUnitsLocked`) | `0` is no cap |
| `life` | `in_flight_`, `draining_`, `done_`, `status_`, staging (`Lifecycle`, the sender's `finishOnceLocked`) | settle protocol |
| `queued` | ghost | `SendNextUnit(u)` calls that took their op |
| `woken` | ghost | `unit_future->OnReady` callbacks that ran |
| `admitting` | ghost | `AdmitPush` tasks on `push_pool`, carrying the callback's op |
| `parked` | `parked_unit_.has_value()` | a unit waits for a window slot |
| `issuing` | ghost | tasks about to run `IssuePush`, carrying an op: an `AdmitPush` past admission or a resume task |
| `chaining` | ghost | tasks past `IssuePush`, carrying their op into `SendNextUnit(u + 1)` |
| `pif` | `pushes_in_flight_` | window slots held |
| `h2hIssued`, `h2hRetired` | ghost | pushes issued, push callbacks that ran |
| `h2hOk` | `units_.size() - remaining_h2h_units_` | pushes completed OK |
| `published` | what `CompleteReadWithDetails` reports | |
| `underflow` | ghost | an `EndSendOpLocked` with nothing in flight |

The chain is a baton: one unit at a time is armed, queued for admission,
issuing, chaining or parked (`Inv.baton_le`), which is why units are pushed in
order and at most one is parked.

## Events

| Event | C++ |
|---|---|
| `wake ok` | `SendNextUnit`'s `unit_future->OnReady` callback: on error `FinishLocked(status_or.status(), kD2hPhase)`, `Release(layer)`, `EndOp()`; on a draining send `Release(layer)`, `EndOp()`; otherwise `std::move(end_op).Cancel()` and `push_pool()->Schedule` of `AdmitPush(u)` |
| `admit` | `AdmitPush`'s locked section: `if (draining_)` → `Release(layer)` and return; `if (push_window_ > 0 && pushes_in_flight_ >= push_window_)` → `parked_unit_ = u` and return; otherwise `++pushes_in_flight_`. Returning ends the op (`absl::Cleanup end_op`). |
| `issue` | `IssuePush`'s locked section and the push: `if (draining_)` → `--pushes_in_flight_`, `Release(layer)`, return; otherwise `++in_flight_` and `H2hWriteDirectAsync` |
| `sendNext` | `SendNextUnit(u + 1)` (`if (draining_ || u >= d2h_unit_futures_.size())` .. `++in_flight_;`), then the task's `absl::Cleanup end_op` |
| `h2hDone ok` | the push callback in `IssuePush`: `--self->pushes_in_flight_`; on error `FinishLocked(push_res.status(), kPushPhase)`; on success `--unpushed_units_[unit.layer]` and, `if (self->parked_unit_.has_value() && !self->draining_)`, `++pushes_in_flight_` and `++in_flight_` for the task that issues the parked unit; then `remaining_h2h_units_.fetch_sub(1) == 1` → `Finish()`, the resume task scheduled (`IssuePush(next)`, `SendNextUnit(next + 1)`), and `absl::Cleanup end_op` |
| `cancel` | `Finish(error)` from outside the data path (deadline, consumer cancel) |
| `publish` | `CompleteReadWithDetails` reporting a settled send |

## Lock scopes

Each event is one critical section of the session's `mu_` except the
following; `Release(layer)` takes only `StagingLayerManager::mu_`.

| Event | Scopes | Between them |
|---|---|---|
| `wake ok` | on error 2 (`FinishLocked`, then `EndOp`); otherwise 1 (`if (self->draining_)`), then `EndOp` or the `Schedule` | a `Finish` after the check is seen by `AdmitPush`'s own check |
| `admit`, `issue` | `AdmitPush`'s scope, then `IssuePush`'s | two events, because `Finish` can land between them: `IssuePush` then returns the slot (`--pushes_in_flight_`) and issues nothing (`trace_cancel_drops_parked` shows the parked case). No other unit is admitted meanwhile (the task holds the baton) and push callbacks only free slots, so the slot taken in `AdmitPush` is still within the window when `IssuePush` uses it |
| `sendNext` | `SendNextUnit`'s scope, then `EndOp` | folded: a `Finish` in between makes the next `SendNextUnit` refuse, as if it had come first |
| `h2hDone ok` | the callback's scope (slot, status, unpushed count, resume decision), then `Release(layer)`, `Finish()` on the last push, the resume `Schedule`, and `EndOp` | folded: a `cancel` landing after the scope keeps its error (the first `FinishLocked` wins), as if it came before the event; a resume decided before it issues nothing, because `IssuePush` sees `draining_` |

## Assumptions

* **A1 (D2H side).** Every unit's D2H future resolves: by its copy, or with the
  send's error once the D2H loop has returned (`fail_unissued_units`). The
  copies, their ops and that loop are `Send.lean`'s (and `LayerStaging.lean`'s
  under bounded host layer staging). Encoded as `wake` being enabled whenever
  the chain is armed, with either outcome.
* **A2 (armed start).** `StartPush` arms `SendNextUnit(0)` with its op before
  the D2H loop (`Send.lean` A6), and a send has at least one unit (a send with
  no layer finishes in `StartPush`, `Send.lean`'s `start`). Encoded as `init`
  and `0 < n`.

## Properties

* **`WindowBound`** — with a cap, at most `window` units hold a slot, so at
  most `window` pushes are in flight.
* **`ParkedHasWaker`** — a parked unit on a send that is not draining always
  has a push in flight whose callback will issue it.
* **`NoStall`** — a send that is not draining holds an op, so it cannot settle
  without finishing.
* **`Publication`** — a send reported done pushed every unit OK.
* `SettleSafe`, `Drained`, `StagingIntegrity`, `SettlesPromptly`,
  `NoUnderflow`, `NoOpLeak` — as in `Send.lean`.

`reachable_safe`: every reachable state is `Safe`, for any number of units and
any window. `settles` (`Progress.lean`): from every reachable state, the
remaining callbacks and pushes completing settle the send, so parking never
deadlocks the chain.

`Traces.lean` replays `TransferSendSessionTest.PushWindowCapsTheGroupsPushingAtOnce`,
`TransferSendSessionTest.CancelledSendDropsTheGroupWaitingForTheWindow` and
`TransferSendSessionTest.PushGroupsSplitALargeLayerIntoCopiesAndPushes`, and
shows a counterexample for each of three mutants: a park that keeps chaining
(`NoStall`), a callback that never resumes (`ParkedHasWaker`), and an
off-by-one window check (`WindowBound`).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `base.h` | `KVCacheManagerBase::SetPushGroups` | `:782-785` |
| `send.h` | `TransferSendSession` (`struct PushUnit {` .. `void IssuePush(size_t u);`) | `:233-249` |
| `send.h` | `TransferSendSession` (`std::vector<PushUnit> units_ ABSL_GUARDED_BY(mu_);` .. `std::optional<size_t> parked_unit_ ABSL_GUARDED_BY(mu_);`) | `:280-296` |
| `send.cc` | `TransferSendSession::BlockGroupBounds` | `:434-453` |
| `send.cc` | `TransferSendSession::PlanUnitsLocked` | `:455-483` |
| `send.cc` | `TransferSendSession::StartPush` (`absl::Cleanup fail_unissued_units =`) | `:650-661` |
| `send.cc` | `TransferSendSession::SendNextUnit` | `:730-776` |
| `send.cc` | `TransferSendSession::SendNextUnit` (`unit_future->OnReady`) | `:746-775` |
| `send.cc` | `TransferSendSession::AdmitPush` | `:778-797` |
| `send.cc` | `TransferSendSession::AdmitPush` (`if (push_window_ > 0 && pushes_in_flight_ >= push_window_)`) | `:788-791` |
| `send.cc` | `TransferSendSession::IssuePush` | `:799-878` |
| `send.cc` | `TransferSendSession::IssuePush` (`--pushes_in_flight_;  // admitted, never issued`) | `:809` |
| `send.cc` | `TransferSendSession::IssuePush` (`--self->pushes_in_flight_;` .. `++self->in_flight_;  // held by the resume task below`) | `:834-845` |
| `send.cc` | `TransferSendSession::IssuePush` (`const bool last = self->remaining_h2h_units_.fetch_sub(1) == 1;`) | `:865` |
| `send.cc` | `TransferSendSession::IssuePush` (`self->base_->push_pool()->Schedule([self, next = *resume]()`) | `:871-875` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PushGroupsSplitALargeLayerIntoCopiesAndPushes` | `:1047-1088` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.PushWindowCapsTheGroupsPushingAtOnce` | `:1121-1153` |
| `transfer_send_session_test.cc` | `TransferSendSessionTest.CancelledSendDropsTheGroupWaitingForTheWindow` | `:1157-1179` |
<!-- source-map:end -->
-/

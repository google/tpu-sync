import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `Send`: state, events, `step`, properties

Part of `TpuSyncVerify.Transfer.PrefillDecode.Send`; the module docstring and
source map live in `Send.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

structure Send where
  numLayers : Nat
  life : Lifecycle := {}
  pullStarted : Bool := false
  started : Bool := false
  d2hPending : Bool := false
  d2hIssued : Nat := 0
  d2hReady : Nat := 0
  d2hRetired : Nat := 0
  queued : Nat := 0
  woken : Nat := 0
  pooled : Nat := 0
  h2hIssued : Nat := 0
  chaining : Nat := 0
  h2hRetired : Nat := 0
  h2hOk : Nat := 0
  published : Option Bool := none
  underflow : Bool := false
  deriving Repr, DecidableEq, Hashable

namespace Send

/-- A registered send (`NotifyForRead`) that nobody has
pulled yet. -/
def init (numLayers : Nat) : Send := { numLayers }

/-- `EndSendOpLocked`: `--in_flight_` and settle. The C++ has
no underflow check; the model records one instead of ignoring it. -/
def endOp (s : Send) : Send :=
  if s.life.inFlight = 0 then { s with underflow := true }
  else { s with life := s.life.endOpLocked }

/-- `Finish(status)` / `FinishLocked(status)`. -/
def finish (ok : Bool) (s : Send) : Send := { s with life := s.life.finishOnceLocked ok }

/-- `SendNextLayer(l)` up to its lock (`if (l >= base_->num_layers())` .. `++in_flight_;`)
for `l = queued`: takes an op unless `l` is past the last layer or the session
is draining. Every layer's future exists from `StartPush`'s locked section on
(`d2h_layer_futures_.emplace_back`), so `l >= d2h_layer_futures_.size()` only
refuses once the settle has cleared them (`ReleaseStagingBlocksLocked`), when
`draining_` already does. -/
def trySendNext (s : Send) : Send :=
  if s.queued < s.numLayers then
    match s.life.beginOp with
    | some l => { s with life := l, queued := s.queued + 1 }
    | none => s
  else s

inductive Ev where
  | beginPull
  | start
  | d2hBegin
  | d2hIssue (ok : Bool)
  | d2hReady
  | d2hEnd
  | wake (ok : Bool)
  | h2hIssue
  | sendNext
  | h2hDone (ok : Bool)
  | cancel
  | publish
  deriving Repr, DecidableEq

/-- `ValidateAndBeginPull`: rejects an expired offer
(`deadline_ <= now`; modelled together with pre-pull
cancellation via `draining`/`done`; the deadline is the only lifecycle check
`ValidateAndBeginPull` itself makes — `draining_`/`done_` are not read there,
but under A5 an unpulled send only drains by deadline or by shutdown, which
`HandlePullStream` rejects first via `stopping_.load()`) or a
duplicate pull (`if (pull_started_)`), and otherwise claims `pull_started_ = true`
before `HandlePullStream` spawns the `StartPush` worker thread. -/
def beginPull (s : Send) : Option Send :=
  if s.pullStarted = true ∨ s.life.draining = true ∨ s.life.done = true then none
  else some { s with pullStarted := true }

/-- `StartPush` with staging acquired, spawned after
`ValidateAndBeginPull` claimed `pull_started_` in `HandlePullStream`. A failed
acquisition (`!acquire_status.ok()`) or worker spawn failure (`HandlePullStream`'s
`counter_cleanup`) is `Finish(error)` with nothing in flight, i.e. `cancel`. On
a session that finished meanwhile the call returns (`if (draining_ || done_) return;`)
and nothing changes. Otherwise the locked section creates every layer's
future (`tsl::MakePromise<>()` per layer into `d2h_layer_futures_`) and, before
the D2H loop, `SendNextLayer(0)` arms the push chain on layer 0's future,
taking its op (A6). -/
def start (s : Send) : Option Send :=
  if s.pullStarted = false ∨ s.started = true ∨ s.life.draining = true ∨ s.life.done = true then none
  else if s.numLayers = 0 then some (s.finish true)
  else some { s with started := true }.trySendNext

/-- One iteration of `StartPush`'s D2H loop up to `++in_flight_`
(`AcquireStagingSlot(l)`, then `if (draining_) { ...; return; } ++in_flight_;`). -/
def d2hBegin (s : Send) : Option Send :=
  if s.started = true ∧ s.d2hPending = false ∧ s.d2hIssued < s.numLayers then
    s.life.beginOp.map fun l => { s with life := l, d2hPending := true }
  else none

/-- The dispatch's outcome in `StartPush`. Failure: `FinishLocked(future.status())`
then `EndSendOp()`, and the loop returns. Success: the copy's `OnReady` is
registered to resolve the layer's promise and end the op. -/
def d2hIssue (ok : Bool) (s : Send) : Option Send :=
  if s.d2hPending = false then none
  else
    let s := { s with d2hPending := false }
    if ok then some { s with d2hIssued := s.d2hIssued + 1 }
    else some (s.finish false).endOp

/-- A dispatched copy finishes on the device: its `OnReady` in `StartPush`
resolves the layer's promise (`promise.Set(status_or.status())`), so
`d2h_layer_futures_[l]` becomes ready. -/
def markD2hReady (s : Send) : Option Send :=
  if s.d2hReady < s.d2hIssued then some { s with d2hReady := s.d2hReady + 1 } else none

/-- The same callback then ends the copy's op (`self->EndSendOp()`). -/
def d2hEnd (s : Send) : Option Send :=
  if s.d2hRetired < s.d2hReady then some { s with d2hRetired := s.d2hRetired + 1 }.endOp
  else none

/-- `SendNextLayer`'s `layer_future->OnReady` callback for layer `woken`, which
runs once that layer's promise is resolved: by its copy's `OnReady` in
`StartPush` (the copy is ready, `woken < d2hReady`), or — for a layer whose
copy was never issued (`d2hIssued ≤ woken`) — by `fail_unissued_layers` with
the send's status once the D2H loop has returned, which it only does on a
draining send with no dispatch in progress; that resolution is always an
error (`ok = false`). -/
def wake (ok : Bool) (s : Send) : Option Send :=
  if s.woken < s.queued ∧
      (s.woken < s.d2hReady ∨
        (ok = false ∧ s.life.draining = true ∧ s.d2hPending = false ∧
          s.d2hIssued ≤ s.woken)) then
    let s := { s with woken := s.woken + 1 }
    if ok = false then some (s.finish false).endOp
    else if s.life.draining = true then some s.endOp
    else some { s with pooled := s.pooled + 1 }
  else none

/-- The `push_pool()->Schedule` task in `SendNextLayer` up to and including `H2hWriteDirectAsync`. -/
def h2hIssue (s : Send) : Option Send :=
  if s.pooled = 0 then none
  else
    let s := { s with pooled := s.pooled - 1 }
    match s.life.beginOp with
    | some l => some { s with life := l, h2hIssued := s.h2hIssued + 1, chaining := s.chaining + 1 }
    | none => some s.endOp

/-- The pool task's tail in `SendNextLayer`: `self->SendNextLayer(l + 1)`, then
`absl::Cleanup end_op` ends the op it carried. -/
def sendNext (s : Send) : Option Send :=
  if s.chaining = 0 then none
  else some { s with chaining := s.chaining - 1 }.trySendNext.endOp

/-- The `H2hWriteDirectAsync` push callback in `SendNextLayer`. -/
def h2hDone (ok : Bool) (s : Send) : Option Send :=
  if s.h2hRetired < s.h2hIssued then
    let s := { s with h2hRetired := s.h2hRetired + 1 }
    if ok then
      let s := { s with h2hOk := s.h2hOk + 1 }
      some (if s.h2hOk = s.numLayers then (s.finish true).endOp else s.endOp)
    else some (s.finish false).endOp
  else none

/-- `Finish(error)` from outside the data path; enabled at any time. -/
def cancel (s : Send) : Option Send := some (s.finish false)

/-- `CompleteReadWithDetails` / `CompleteReadRaw` publishes a settled session
(`if (session->Done())`). -/
def publish (s : Send) : Option Send :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk }
  else none

def step (s : Send) : Ev → Option Send
  | .beginPull => s.beginPull
  | .start => s.start
  | .d2hBegin => s.d2hBegin
  | .d2hIssue ok => s.d2hIssue ok
  | .d2hReady => s.markD2hReady
  | .d2hEnd => s.d2hEnd
  | .wake ok => s.wake ok
  | .h2hIssue => s.h2hIssue
  | .sendNext => s.sendNext
  | .h2hDone ok => s.h2hDone ok
  | .cancel => s.cancel
  | .publish => s.publish

/-- A registered send with `n` layers. -/
def sys (n : Nat) : System Send Ev := ⟨init n, step⟩

/-! ## Properties -/

def SettleSafe (s : Send) : Prop := s.life.done = true → s.life.inFlight = 0

/-- A settled send has no copy, callback, pool task or push outstanding. -/
def Drained (s : Send) : Prop :=
  s.life.done = true →
    s.d2hPending = false ∧ s.d2hRetired = s.d2hIssued ∧ s.woken = s.queued ∧
      s.pooled = 0 ∧ s.chaining = 0 ∧ s.h2hRetired = s.h2hIssued

def StagingIntegrity (s : Send) : Prop := s.life.hasStaging = !s.life.done

def SettlesPromptly (s : Send) : Prop :=
  s.life.draining = true → s.life.inFlight = 0 → s.life.done = true

def NoUnderflow (s : Send) : Prop := s.underflow = false

def Publication (s : Send) : Prop := s.published = some true → s.h2hOk = s.numLayers

def CountersOrdered (s : Send) : Prop :=
  s.d2hRetired ≤ s.d2hReady ∧ s.d2hReady ≤ s.d2hIssued ∧
    s.d2hIssued + (if s.d2hPending then 1 else 0) ≤ s.numLayers ∧
    s.woken ≤ s.queued ∧ s.queued ≤ s.numLayers ∧
    s.pooled + s.h2hIssued ≤ s.woken ∧ s.pooled + s.h2hIssued ≤ s.d2hReady ∧
    s.chaining ≤ s.h2hIssued ∧ s.h2hRetired ≤ s.h2hIssued ∧ s.h2hOk ≤ s.h2hRetired

/-- `StartPush` (`started = true`) only runs after `ValidateAndBeginPull`
claimed `pull_started_` (`pullStarted = true`), and no D2H copy is pending or
issued before `StartPush` starts. -/
def PullClaimed (s : Send) : Prop :=
  (s.started = true → s.pullStarted = true) ∧
  ((s.d2hPending = true ∨ 0 < s.d2hIssued) → s.started = true)

/-- Events that advance or retire an in-flight operation: a loop iteration
issuing the copy an armed `SendNextLayer` waits for, a dispatch outcome, a
copy or push completing, a callback or pool task running — including the
`SendNextLayer` callback that `fail_unissued_layers` resolves with the send's
failure (`wake false`). -/
def drainEvents : List Ev :=
  [.d2hBegin, .d2hIssue true, .d2hReady, .d2hEnd, .wake true, .wake false, .h2hIssue, .sendNext,
   .h2hDone true]

/-- Every unit of `in_flight_` has an owner that can advance or retire it:
while `inFlight > 0`, some event in `drainEvents` is enabled. -/
def NoOpLeak (s : Send) : Prop :=
  0 < s.life.inFlight → ∃ e ∈ drainEvents, (step s e).isSome = true

/-- The sender's promise to the engine: once `poll_stats()` has reported the
send (`done_sending` or `failed_recving`), no D2H copy is dispatching or
still reading prefill HBM, so the caller may reclaim it. -/
def PublishedQuiet (s : Send) : Prop :=
  s.published ≠ none → s.d2hPending = false ∧ s.d2hRetired = s.d2hIssued

/-- Everything we want to know about a reachable send session. -/
def Safe (s : Send) : Prop :=
  SettleSafe s ∧ Drained s ∧ StagingIntegrity s ∧ SettlesPromptly s ∧ NoUnderflow s ∧
    Publication s ∧ CountersOrdered s ∧ NoOpLeak s ∧ PullClaimed s ∧ PublishedQuiet s


end Send

end TpuSyncVerify.Transfer.PrefillDecode

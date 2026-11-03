import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `Recv`: state, events, `step`, properties

Part of `TpuSyncVerify.Transfer.PrefillDecode.Receive`; the module docstring and
source map live in `Receive.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

open TpuSyncVerify.Transfer (Lifecycle)

structure Recv where
  numLayers : Nat
  life : Lifecycle := {}
  issued : Nat := 0
  ready : Nat := 0
  completed : Nat := 0
  layersAccounted : Nat := 0
  published : Option Bool := none
  pushes : Nat := 0
  pullPending : Bool := false
  pending : Nat := 0
  retired : Nat := 0
  deriving Repr, DecidableEq, Hashable

namespace Recv

/-- A receiver registered from a push plan (`InitFromActivePlan`): nothing in
flight until the transport starts pushing. -/
def initPush (numLayers : Nat) : Recv := { numLayers }

/-- A receiver created by `StartRead` (`InitFromLoadPlan`):
`in_flight_ = load_plan.num_blocks > 0 ? 1 : 0` for the pull handshake
(zero-block reads get `in_flight_ = 0` and are finished by `StartRead` at
`if (num_blocks == 0) { session->Finish(); return; }` without a pull, excluded
by A2). -/
def initLoad (numLayers : Nat) : Recv :=
  { numLayers, life := { inFlight := 1 }, pullPending := true }

/-- `network_completed_` (`RecordBlockShardsReceivedLocked`): every shard's
blocks accounted (agrees with `TransferReceiveSession::network_completed_` for
`numLayers > 0`; at `numLayers = 0`, C++'s `IsReadyToComplete` holds via its
`num_completed_layers_ == static_cast<int32_t>(total_layers)` disjunct while
`network_completed_` stays `false`). -/
def networkCompleted (s : Recv) : Prop := s.layersAccounted = s.numLayers

/-- `AllH2dDoneLocked`: every future in `h2d_futures_` is ready. -/
def allH2dDone (s : Recv) : Prop := s.ready = s.issued

/-- `IsReadyToComplete`. -/
def isReadyToComplete (s : Recv) : Prop :=
  (networkCompleted s ∨ s.completed = s.numLayers) ∧ allH2dDone s

instance : DecidablePred networkCompleted :=
  fun s => inferInstanceAs (Decidable (s.layersAccounted = s.numLayers))
instance : DecidablePred allH2dDone :=
  fun s => inferInstanceAs (Decidable (s.ready = s.issued))
instance : DecidablePred isReadyToComplete :=
  fun s => inferInstanceAs (Decidable ((networkCompleted s ∨ s.completed = s.numLayers) ∧ allH2dDone s))

inductive Ev where
  | pushBegin
  | pushEnd
  | pullReply (ok : Bool)
  | h2dBegin
  | h2dIssue (ok : Bool)
  | h2dReady
  | h2dDone (ok : Bool)
  | netAccount
  | pollReady
  | cancel
  | publish
  deriving Repr, DecidableEq

/-- `TryBeginRecvOp`. When it returns `false` the transport drops the push and
the session is unchanged, so only the accepting case is a transition. -/
def pushBegin (s : Recv) : Option Recv :=
  s.life.beginOp.map fun l => { s with life := l, pushes := s.pushes + 1 }

/-- `EndRecvOp` for a push that was accepted. -/
def pushEnd (s : Recv) : Option Recv :=
  if s.pushes = 0 then none
  else some { s with life := s.life.endOpLocked, pushes := s.pushes - 1 }

/-- The pull handshake resolves (`ExecutePullRequest`'s `on_response`):
`self->FinishLocked(pull_status)` if `!pull_status.ok()` and
`self->handshake_pending_ = false`, then `absl::Cleanup end_op` calls
`EndRecvOp()`. -/
def pullReply (ok : Bool) (s : Recv) : Option Recv :=
  if s.pullPending = false then none
  else
    let l := if ok then s.life else s.life.finishLocked false
    some { s with life := l.endOpLocked, pullPending := false }

/-- `ExecuteLayerH2d` up to the first unlock: refused once settled or draining
(`if (done_ || draining_) return absl::OkStatus();`), and only while
`issued + pending < numLayers` (A1); otherwise `++in_flight_`. -/
def h2dBegin (s : Recv) : Option Recv :=
  if s.issued + s.pending < s.numLayers then
    s.life.beginOp.map fun l => { s with life := l, pending := s.pending + 1 }
  else none

/-- `ExecuteLayerH2d` from the re-check on. If the session finished in the
window between the two locks (`if (done_ || draining_) { EndRecvOpLocked(); return absl::OkStatus(); }`),
the op is released and no copy is issued. Otherwise the copy is dispatched via
`H2dSyncDispatch`: on failure `FinishLocked(future_or.status(), kH2dPhase); EndRecvOpLocked()`;
on success its future joins `h2d_futures_` (`h2d_futures_.push_back(future)`)
and the op stays in flight until the callback. -/
def h2dIssue (ok : Bool) (s : Recv) : Option Recv :=
  if s.pending = 0 then none
  else
    let s := { s with pending := s.pending - 1 }
    if s.life.done = true ∨ s.life.draining = true then
      some { s with life := s.life.endOpLocked }
    else if ok then
      some { s with issued := s.issued + 1 }
    else
      some { s with life := (s.life.finishLocked false).endOpLocked }

/-- A dispatched copy finishes on the device; its future becomes ready. -/
def h2dReady (s : Recv) : Option Recv :=
  if s.ready < s.issued then some { s with ready := s.ready + 1 } else none

/-- H2D completion callback (`ExecuteLayerH2d`'s `future.OnReady`), run once per
ready future. `absl::Cleanup end_op` calls `EndRecvOp()` on every path. On a
retired session (`if (self->done_)`) the body returns early (`LOG(DFATAL)`). On
success `self->num_completed_layers_++`; the last layer calls
`self->FinishLocked()` unless `!self->draining_` fails. On failure
`self->FinishLocked(status_or.status(), kH2dPhase)`. -/
def h2dDone (ok : Bool) (s : Recv) : Option Recv :=
  if s.retired < s.ready then
    let s := { s with retired := s.retired + 1 }
    if s.life.done = true then
      some { s with life := s.life.endOpLocked }
    else if ok then
      let s := { s with completed := s.completed + 1 }
      let l := if s.completed = s.numLayers ∧ s.life.draining = false
               then s.life.finishLocked true else s.life
      some { s with life := l.endOpLocked }
    else
      some { s with life := (s.life.finishLocked false).endOpLocked }
  else none

/-- `OnBlockShardsReceived` for the stream that completes a layer (A4). Ignored
once settled or draining (`if (done_ || draining_) return absl::OkStatus();`).
Otherwise `RecordBlockShardsReceivedLocked` accounts the stream's blocks per
shard; if every shard reaches `per_shard_target` and
`num_completed_layers_ == static_cast<int32_t>(total_layers)`,
`RecordNetworkCompleteLocked` calls `FinishLocked()`. -/
def netAccount (s : Recv) : Option Recv :=
  if s.life.done = true ∨ s.life.draining = true then none
  else if s.layersAccounted < s.issued then
    let s := { s with layersAccounted := s.layersAccounted + 1 }
    if networkCompleted s ∧ s.completed = s.numLayers then
      some { s with life := s.life.finishLocked true }
    else some s
  else none

/-- `CompleteReadWithDetails` / `CompleteReadRaw` polls a session that is not
draining (`!session->IsDraining()`) and finds `session->IsReadyToComplete()`,
calling `session->Finish()`. -/
def pollReady (s : Recv) : Option Recv :=
  if s.life.draining = false ∧ isReadyToComplete s then
    some { s with life := s.life.finishLocked true }
  else none

/-- `Finish(error)` from outside the session; enabled at any time. -/
def cancel (s : Recv) : Option Recv :=
  some { s with life := s.life.finishLocked false }

/-- `CompleteReadWithDetails` / `CompleteReadRaw` publishes a settled session
(`if (session->Done())`). -/
def publish (s : Recv) : Option Recv :=
  if s.life.done = true ∧ s.published = none then
    some { s with published := some s.life.statusOk }
  else none

def step (s : Recv) : Ev → Option Recv
  | .pushBegin => s.pushBegin
  | .pushEnd => s.pushEnd
  | .pullReply ok => s.pullReply ok
  | .h2dBegin => s.h2dBegin
  | .h2dIssue ok => s.h2dIssue ok
  | .h2dReady => s.h2dReady
  | .h2dDone ok => s.h2dDone ok
  | .netAccount => s.netAccount
  | .pollReady => s.pollReady
  | .cancel => s.cancel
  | .publish => s.publish

/-- A push-plan receiver with `n` layers. -/
def sysPush (n : Nat) : System Recv Ev := ⟨initPush n, step⟩

/-- A StartRead receiver with `n` layers. -/
def sysLoad (n : Nat) : System Recv Ev := ⟨initLoad n, step⟩

/-! ## Properties -/

def SettleSafe (s : Recv) : Prop := s.life.done = true → s.life.inFlight = 0

def NoRetiredCallback (s : Recv) : Prop := s.life.done = true → s.retired = s.issued

def StagingIntegrity (s : Recv) : Prop := s.life.hasStaging = !s.life.done

def SettlesPromptly (s : Recv) : Prop :=
  s.life.draining = true → s.life.inFlight = 0 → s.life.done = true

def ReadinessSound (s : Recv) : Prop := isReadyToComplete s → s.ready = s.numLayers

def Publication (s : Recv) : Prop := s.published = some true → s.completed = s.numLayers

def CountersOrdered (s : Recv) : Prop :=
  s.completed ≤ s.retired ∧ s.retired ≤ s.ready ∧ s.ready ≤ s.issued ∧
    s.issued ≤ s.numLayers ∧ s.layersAccounted ≤ s.issued

/-- Events that advance or retire an in-flight operation. -/
def drainEvents : List Ev :=
  [.pushEnd, .pullReply true, .h2dIssue true, .h2dReady, .h2dDone true]

/-- Every unit of `in_flight_` has an owner that can advance or retire it:
while `inFlight > 0`, some event in `drainEvents` is enabled. -/
def NoOpLeak (s : Recv) : Prop :=
  0 < s.life.inFlight → ∃ e ∈ drainEvents, (step s e).isSome = true

/-- The receiver's promise to the engine: once `poll_stats()` has reported the
receive, as `done_recving` or as `failed_recving`, no H2D copy is dispatching
or still writing decode HBM, so the caller may reuse or refill it. -/
def PublishedQuiet (s : Recv) : Prop :=
  s.published ≠ none → s.pending = 0 ∧ s.retired = s.issued

/-- Everything we want to know about a reachable receive session. -/
def Safe (s : Recv) : Prop :=
  SettleSafe s ∧ NoRetiredCallback s ∧ StagingIntegrity s ∧ SettlesPromptly s ∧
    ReadinessSound s ∧ Publication s ∧ CountersOrdered s ∧ NoOpLeak s ∧ PublishedQuiet s


end Recv

end TpuSyncVerify.Transfer.PrefillDecode

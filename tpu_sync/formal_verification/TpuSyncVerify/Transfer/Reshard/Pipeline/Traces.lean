import TpuSyncVerify.Transfer.Reshard.Pipeline.Progress

/-! # `Pipeline`: concrete traces, bounded verification, mutants

Part of `TpuSyncVerify.Transfer.Reshard.Pipeline`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ## Concrete traces and bounded verification

### Out-of-order network arrival on aliased device pages

Two pools with `ranks = [0, 1]` and `peerCounts = [1, 2]` share an aliased
device page (`pool_order_ranks_` in `ReshardReceiveSession`, `D2hPoolBlocks` /
`H2dPoolBlocks` in `KVCacheManagerBase`). Even when pool 1 (rank 1, 2 peer
streams) finishes landing all of its network streams and calls
`onPoolReceived 1` **before** pool 0 (rank 0, 1 peer stream) has landed its
stream:
1. `onPoolReceived 1` records pool 1 as `.received` without launching its H2D
   copy (`dstHbm = [.junk, .junk]`).
2. When pool 0 subsequently lands its stream and calls `onPoolReceived 0`,
   pool 0 enters `.queued`, completes its H2D copy (`h2dDone 0 true`), writes
   `.kv 0` to `dstHbm[0]` (while clobbering `dstHbm[1]` to `.junk`), and
   `FinishPoolH2d` automatically unblocks and queues pool 1 (`stage = .queued`).
3. Pool 1's second-wave H2D copy (`h2dDone 1 true`) then overwrites `dstHbm[1]`
   with `.kv 1`, so `publish` reports `done_recving` with `dstHbm = good 2`.
-/

def outOfOrderPrefix : List Ev :=
  [ -- Sender issues D2H for pool 0 (1 peer) and pool 1 (2 peers), finishes ExecutePush,
    -- and completes all 3 peer pushes onto the wire.
    .send (.d2hIssue 0 true), .send (.d2hIssue 1 true), .send .execFinish,
    .send (.d2hDone 0 true), .send (.d2hDone 1 true),
    .send (.pushDone 0 true), .send (.pushDone 1 true), .send (.pushDone 1 true),
    .send .publish,
    -- Receiver begins incoming pushes; pool 1 (rank 1) lands BOTH streams and calls
    -- OnPoolReceived(1) BEFORE pool 0 (rank 0) has landed its stream.
    .recv .pushBegin,
    .landStream 1, .landStream 1,
    .recv (.onPoolReceived 1) ]

def outOfOrderSuffix : List Ev :=
  [ -- Pool 0 (rank 0) now lands its stream, calls OnPoolReceived(0), and runs Wave 0 H2D.
    .landStream 0,
    .recv (.onPoolReceived 0),
    .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true),
    -- Finishing pool 0's H2D automatically queues pool 1 (rank 1); Wave 1 H2D now runs.
    .recv (.h2dIssue 1 true), .recv (.h2dReady 1), .recv (.h2dDone 1 true),
    .recv (.pushEnd true),
    .recv .publish,
    -- Post-settlement cleanup: reclaim producer HBM and reseat both staging leases.
    .reclaim, .reseatSrcStaging, .reseatDstStaging ]

theorem trace_aliased_page_out_of_order_arrival :
    let s := sys [0, 1] [1, 2]
    -- After pool 1 (rank 1) lands first, it waits at `.received` while pool 0 is `.waiting`.
    (s.run outOfOrderPrefix).map
      (fun p => (p.recv.pools.map Pool.stage, p.dstHbm)) =
      some ([.waiting, .received], [.junk, .junk]) ∧
    -- After pool 0 (rank 0) completes Wave 0 H2D, pool 1 (rank 1) completes Wave 1 H2D
    -- and `dstHbm = [.kv 0, .kv 1] = good 2` survives post-publication cleanup.
    (s.run (outOfOrderPrefix ++ outOfOrderSuffix)).map
      (fun p => (p.send.published, p.recv.published, p.dstHbm, p.reclaimed)) =
      some (some true, some true, [.kv 0, .kv 1], true) := by decide

/-- Two-phase Arm-then-Dispatch trace (`ReshardCoordinator::RunSubPlan`):
from `initUnarmed`, neither sender D2H (`d2hIssue 0 true`) nor receiver
`pushBegin` can fire before `armRecv`; once `armRecv` runs, the transfer
completes cleanly. -/
theorem trace_arm_before_dispatch :
    let s := sysUnarmed [0] [1]
    s.run [.send (.d2hIssue 0 true)] = none ∧
    s.run [.recv .pushBegin] = none ∧
    (s.run [
      .armRecv,
      .send (.d2hIssue 0 true), .send .execFinish,
      .send (.d2hDone 0 true), .send (.pushDone 0 true), .send .publish,
      .recv .pushBegin, .landStream 0, .recv (.onPoolReceived 0),
      .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true),
      .recv (.pushEnd true), .recv .publish
    ]).map (fun p => (p.armed, p.recv.published, p.dstHbm)) =
      some (true, some true, [.kv 0]) := by decide

/-! ### Executable safety checker and `#guard` bounded searches -/

def allEvents (n : Nat) : List Ev :=
  [.armRecv, .reclaim, .reseatSrcStaging, .reseatDstStaging] ++
  (Send.events n).map Ev.send ++
  (Recv.events n).map Ev.recv ++
  (List.range n).map Ev.landStream

/-- `∀ x, o = some x → P x` is decided by inspecting `o`. -/
private instance decForallSome {α : Type} (o : Option α) (P : α → Prop) [DecidablePred P] :
    Decidable (∀ x, o = some x → P x) :=
  match o with
  | none => isTrue (fun _ h => by cases h)
  | some a =>
    if h : P a then isTrue (fun _ hx => by cases hx; exact h)
    else isFalse (fun hall => h (hall a rfl))

instance (s : Pipeline) : Decidable (Safe s) := by
  unfold Safe PublicationCorrect AliasedPageCorrect DstHbmSafe SrcHbmSafe StagingSafe
    StreamAccountingSafe NoUnarmedPush NoOpLeak
  infer_instance

/-- Executable negation of `Safe`, by construction: the bounded searches and the
`reachable_safe` theorems check the same predicate. -/
def violates (s : Pipeline) : Bool := !decide (Safe s)

theorem violates_iff {s : Pipeline} : violates s = false ↔ Safe s := by
  simp [violates]


/-! ## Mutants (`docs/transfer/reshard.md`)

1. **Rank-blind H2D launch (`stepRankBlind`)**: if `CollectEligiblePoolH2dsLocked`
   ignores pool rank (`prereqsCompleted` replaced by `fun _ => true`, as in
   `Recv.onPoolReceivedIgnoringRank`), then when pool 1 (rank 1) lands before
   pool 0 (rank 0), pool 1 completes H2D first and pool 0's subsequent coarse
   H2D copy clobbers `dstHbm[1]` to `.junk`, violating `PublicationCorrect` and
   `AliasedPageCorrect`.
2. **Premature `OnPoolReceived` (`stepPrematurePoolReceived`)**: if
   `BlockTransportServerPushSession::CompleteIncomingPush` fires `OnPoolReceived(p)` on the first
   arriving stream (`0 < landedStreams[p]`) instead of waiting for all
   `peerCounts[p]` streams (`progress.completed_chunks >= progress.expected_chunks`),
   `StreamAccountingSafe` is violated.
3. **Premature producer reclamation (`stepReclaimEarly`)**: if producer HBM is
   reclaimed before `send.published ≠ none`, D2H copies `.junk` into destination
   HBM and violates `PublicationCorrect`.
-/

def recvStepRankBlind (s : Pipeline) (e : Recv.Ev) : Option Pipeline :=
  match e with
  | .onPoolReceived p =>
    if s.landedStreams[p]? = some (s.peerCounts.getD p 0) then
      (Recv.onPoolReceivedIgnoringRank p s.recv).map fun rcv =>
        { s with recv := rcv }
    else none
  | e => recvStep s e

def stepRankBlind (s : Pipeline) : Ev → Option Pipeline
  | .recv e => recvStepRankBlind s e
  | e => step s e

def sysRankBlind (ranks peerCounts : List Nat) : System Pipeline Ev :=
  ⟨init ranks peerCounts, stepRankBlind⟩

/-- Mutant 1: without the rank gate in `CollectEligiblePoolH2dsLocked`,
out-of-order network arrival lets pool 1 (rank 1) finish H2D before pool 0
(rank 0), and pool 0's subsequent H2D copy clobbers the shared page slot
`dstHbm[1]` to `.junk`, publishing `done_recving` with corrupted HBM
(`[.kv 0, .junk] ≠ good 2`). -/
theorem mutant_ignoring_rank_corrupts_aliased_page :
    ∃ s, (sysRankBlind [0, 1] [1, 2]).Reachable s ∧
      s.recv.published = some true ∧ s.dstHbm = [.kv 0, .junk] ∧
      ¬PublicationCorrect s ∧ ¬AliasedPageCorrect s := by
  let evs : List Ev :=
    outOfOrderPrefix ++
    [ -- Without the rank gate, pool 1 (rank 1) was already queued by onPoolReceived 1
      -- and completes its H2D copy BEFORE pool 0 (rank 0) even lands!
      .recv (.h2dIssue 1 true), .recv (.h2dReady 1), .recv (.h2dDone 1 true),
      -- Now pool 0 (rank 0) lands and runs its coarse H2D copy, clobbering dstHbm[1]!
      .landStream 0, .recv (.onPoolReceived 0),
      .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true),
      .recv (.pushEnd true), .recv .publish ]
  let s := ((sysRankBlind [0, 1] [1, 2]).run evs).getD (init [0, 1] [1, 2])
  have hrun : (sysRankBlind [0, 1] [1, 2]).run evs = some s := by decide
  refine ⟨s, (sysRankBlind [0, 1] [1, 2]).run_reachable hrun, by decide, by decide, ?_, ?_⟩
  · intro hpub
    have := hpub (by decide)
    revert this; decide
  · intro hal
    have := hal 1 (by decide) { rank := 1, stage := .completed } (by decide) rfl
    revert this; decide

def stepPrematurePoolReceived (s : Pipeline) : Ev → Option Pipeline
  | .recv (.onPoolReceived p) =>
    if 0 < s.landedStreams.getD p 0 then
      (Recv.step .finishOnError s.recv (.onPoolReceived p)).map fun rcv =>
        { s with recv := rcv }
    else none
  | e => step s e

def sysPrematurePoolReceived (ranks peerCounts : List Nat) : System Pipeline Ev :=
  ⟨init ranks peerCounts, stepPrematurePoolReceived⟩

/-- Mutant 2: firing `OnPoolReceived(0)` after 1 stream when `peerCounts = [2]`
violates `StreamAccountingSafe`. -/
theorem mutant_onPoolReceived_before_all_streams :
    ∃ s, (sysPrematurePoolReceived [0] [2]).Reachable s ∧ ¬StreamAccountingSafe s := by
  let evs : List Ev :=
    [ .send (.d2hIssue 0 true), .send .execFinish,
      .send (.d2hDone 0 true), .send (.pushDone 0 true),
      .recv .pushBegin, .landStream 0, .recv (.onPoolReceived 0) ]
  let s := ((sysPrematurePoolReceived [0] [2]).run evs).getD (init [0] [2])
  have hrun : (sysPrematurePoolReceived [0] [2]).run evs = some s := by decide
  refine ⟨s, (sysPrematurePoolReceived [0] [2]).run_reachable hrun, ?_⟩
  intro hstr
  have ⟨_, _, hstage⟩ := hstr 0 (by decide)
  have := hstage { rank := 0, stage := .queued } (by decide) (by decide)
  revert this; decide

def stepReclaimEarly (s : Pipeline) : Ev → Option Pipeline
  | .reclaim =>
    if s.reclaimed = false then
      some { s with srcHbm := List.replicate s.numPools .junk, reclaimed := true }
    else none
  | e => step s e

def sysReclaimEarly (ranks peerCounts : List Nat) : System Pipeline Ev :=
  ⟨init ranks peerCounts, stepReclaimEarly⟩

/-- Mutant 3: reclaiming producer HBM before `send.published ≠ none` causes D2H
to stage `.junk` and publish `done_recving` with `dstHbm = [.junk] ≠ good 1`. -/
theorem mutant_reclaim_before_publish_corrupts_dstHbm :
    ∃ s, (sysReclaimEarly [0] [1]).Reachable s ∧
      s.recv.published = some true ∧ s.dstHbm = [.junk] ∧ ¬PublicationCorrect s := by
  let evs : List Ev :=
    [ .reclaim,
      .send (.d2hIssue 0 true), .send .execFinish,
      .send (.d2hDone 0 true), .send (.pushDone 0 true),
      .recv .pushBegin, .landStream 0, .recv (.onPoolReceived 0),
      .recv (.h2dIssue 0 true), .recv (.h2dReady 0), .recv (.h2dDone 0 true),
      .recv (.pushEnd true), .recv .publish ]
  let s := ((sysReclaimEarly [0] [1]).run evs).getD (init [0] [1])
  have hrun : (sysReclaimEarly [0] [1]).run evs = some s := by decide
  refine ⟨s, (sysReclaimEarly [0] [1]).run_reachable hrun, by decide, by decide, ?_⟩
  intro hpub
  have := hpub (by decide)
  revert this; decide


end Pipeline

end TpuSyncVerify.Transfer.Reshard

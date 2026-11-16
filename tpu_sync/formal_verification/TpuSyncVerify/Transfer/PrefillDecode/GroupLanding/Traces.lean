import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.GroupLanding.Invariant

/-! # `GroupLanding`: replay of upstream tests, searches, mutants

Part of `TpuSyncVerify.Transfer.PrefillDecode.GroupLanding`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.GroupLanding

open TpuSyncVerify TpuSyncVerify.Transfer

namespace S

/-- Every event of a layer of up to `n` blocks, with both outcomes where it
has one. -/
def allEvents (n : Nat) : List Ev :=
  [.streamBegin, .landHalf, .landOther, .streamEnd, .streamLast, .cancel] ++
    (List.range n).flatMap fun k =>
      [.landFull (k + 1), .dispatch (k + 1) true, .dispatch (k + 1) false,
       .copyDone (k + 1) true, .copyDone (k + 1) false]

/-- One stream carrying every shard lands `k` blocks and ends. -/
def stream (k : Nat) : List Ev := [.streamBegin, .landFull k, .streamEnd]

/-! ## Replay

The receive tests seat a pulled receive (`RecvTestManager::AddLoadRecv`) and
land streams as `StartStream` reports them (`RecvTestManager::LandStream`:
the stream's op, `OnBlocksLanded`, for the last stream the layer, the block
report, then `EndOp`). A copy is dispatched inside the call that counted it,
so its `dispatch` precedes the stream's end. -/

/-- `RecvLifecycleTest.PushGroupsCopyLandedBlocksBeforeTheirLayerLands`: five
blocks in groups of two over streams of one, one, two and one block. The
second and third streams each issue a group copy; the last stream's block is
copied with the layer. Copies finish out of order, and the layer completes
with the last of them. -/
theorem trace_groups_copy_before_layer :
    let pre := stream 1 ++ [.streamBegin, .landFull 1, .dispatch 2 true, .streamEnd,
      .streamBegin, .landFull 2, .dispatch 2 true, .streamEnd,
      .streamBegin, .landFull 1, .streamLast, .dispatch 1 true]
    ((sys 5 2 1).run pre).map (fun s => (s.copyingCopies, s.landed, s.completed)) =
      some (3, true, 0) ∧
    ((sys 5 2 1).run (pre ++ [.copyDone 1 true, .copyDone 2 true])).map
      (fun s => (s.completed, s.life.done)) = some (0, false) ∧
    ((sys 5 2 1).run (pre ++ [.copyDone 1 true, .copyDone 2 true, .copyDone 2 true])).map
      (fun s => (s.completed, s.copied, s.life.done, s.life.statusOk)) =
      some (1, 5, true, true) := by
  decide

/-- `RecvLifecycleTest.PushGroupsCompleteALayerWhoseCopiesAlreadyFinished`: two
blocks in groups of one. Both group copies finish before the last stream
counts; that stream finds nothing left to copy and completes the layer. -/
theorem trace_copies_finish_first :
    ((sys 2 1 1).run [.streamBegin, .landFull 1, .dispatch 1 true, .streamEnd,
        .streamBegin, .landFull 1, .dispatch 1 true, .copyDone 1 true, .copyDone 1 true]).map
      (fun s => (s.completed, s.life.done)) = some (0, false) ∧
    ((sys 2 1 1).run [.streamBegin, .landFull 1, .dispatch 1 true, .streamEnd,
        .streamBegin, .landFull 1, .dispatch 1 true, .copyDone 1 true, .copyDone 1 true,
        .streamLast]).map
      (fun s => (s.completed, s.copied, s.life.done, s.life.statusOk)) =
      some (1, 2, true, true) := by
  decide

/-- `RecvLifecycleTest.WithoutPushGroupsALayerIsCopiedWhenItLands` (and, since
a receive under bounded host layer staging copies whole layers,
`RecvLifecycleTest.PushGroupsCopyWholeLayersUnderBoundedHostStaging`): without
groups the reports copy nothing, and the layer is copied whole when it lands. -/
theorem trace_no_groups :
    ((sys 3 0 1).run (stream 2 ++ [.streamBegin, .landFull 1, .streamLast])).map
      (fun s => (s.countedCopies, s.counted)) = some (1, 3) ∧
    ((sys 3 0 1).run (stream 2 ++ [.streamBegin, .landFull 1, .streamLast, .dispatch 3 true,
        .copyDone 3 true])).map
      (fun s => (s.completed, s.copied, s.life.done)) = some (1, 3, true) := by
  decide

/-- `RecvLifecycleTest.FailedPushGroupCopyFailsTheReceiveAfterTheOthersEnd`: a
failed group copy fails the receive, which settles only once the other copy
has finished. -/
theorem trace_failed_group_copy :
    let pre := [.streamBegin, .landFull 1, .dispatch 1 true, .streamEnd,
      .streamBegin, .landFull 1, .dispatch 1 true, .streamEnd]
    ((sys 2 1 1).run (pre ++ [.copyDone 1 false])).map
      (fun s => (s.life.draining, s.life.done)) = some (true, false) ∧
    ((sys 2 1 1).run (pre ++ [.copyDone 1 false, .copyDone 1 true])).map
      (fun s => (s.life.done, s.life.statusOk, s.completed)) = some (true, false, 0) := by
  decide

/-- A stream that carries only some shards issues no group copy: the block is
copied with the layer, once its other shards have landed. -/
theorem trace_partial_shards_wait_for_the_layer :
    ((sys 1 1 1).run [.streamBegin, .landHalf, .landOther]).map
      (fun s => (s.countedCopies, s.untaken)) = some (0, 1) ∧
    ((sys 1 1 1).run [.streamBegin, .landHalf, .landOther, .streamLast, .dispatch 1 true,
        .copyDone 1 true]).map
      (fun s => (s.completed, s.copied)) = some (1, 1) := by
  decide

/-! ## Search

Every reachable state of these instances is visited (`.safe`); `reachable_safe`
is the guarantee for every instance. -/

#guard ModelCheck.check (sys 2 1 2) (allEvents 2) violates 60 = .safe
#guard ModelCheck.check (sys 3 2 2) (allEvents 3) violates 60 = .safe
#guard ModelCheck.check (sys 3 0 2) (allEvents 3) violates 60 = .safe

/-! ## Mutants -/

/-- Mutant for `CompleteSound`: the copy callback completes the layer once it
has landed, whether or not another of its copies is still outstanding
(`layer.outstanding == 0` dropped). -/
def stepIgnoreOutstanding (s : S) : Ev → Option S
  | .copyDone k ok =>
    if 0 < s.copyingCopies ∧ fits k s.copying s.copyingCopies then
      let s := { s with copying := s.copying - k, copyingCopies := s.copyingCopies - 1 }
      if ok then
        let s := { s with copied := s.copied + k }
        let s := if s.landed = true ∧ s.completed = 0 then s.completeLayer else s
        some { s with life := s.life.endOpLocked }
      else some { s with lost := s.lost + k, life := (s.life.finishLocked false).endOpLocked }
    else none
  | e => s.step e

#guard match ModelCheck.check ⟨init 2 1 1, stepIgnoreOutstanding⟩ (allEvents 2) violates 40 with
  | .counterexample _ => true
  | _ => false

/-- Mutant for `CompleteSound`: the copy callback completes a layer whose
copies have all finished before its last stream counted (`layer.landed`
dropped), so the blocks that stream leaves for the last copy are never
copied. -/
def stepCompleteBeforeLanded (s : S) : Ev → Option S
  | .copyDone k ok =>
    if 0 < s.copyingCopies ∧ fits k s.copying s.copyingCopies then
      let s := { s with copying := s.copying - k, copyingCopies := s.copyingCopies - 1 }
      if ok then
        let s := { s with copied := s.copied + k }
        let s :=
          if s.completed = 0 ∧ s.countedCopies + s.copyingCopies = 0 then s.completeLayer else s
        some { s with life := s.life.endOpLocked }
      else some { s with lost := s.lost + k, life := (s.life.finishLocked false).endOpLocked }
    else none
  | e => s.step e

#guard match ModelCheck.check ⟨init 2 1 1, stepCompleteBeforeLanded⟩ (allEvents 2) violates 40 with
  | .counterexample _ => true
  | _ => false

/-- Mutant for `CopiesReadLanded`: `OnBlocksLanded` takes the blocks of a
stream that carries only some shards (`shard_ids.size() != num_shards()`
dropped), so a group copy can read a block whose other shards are still in
flight. -/
def stepGroupPartialShards (s : S) : Ev → Option S
  | .landHalf =>
    if 0 < s.streams ∧ 0 < s.fresh ∧ s.groupBlocks ≠ 0 ∧ s.life.draining = false then
      let s := { s with fresh := s.fresh - 1, pending := s.pending + 1,
                        pendingHalf := s.pendingHalf + 1 }
      if s.groupBlocks ≤ s.pending then
        some { s with counted := s.counted + s.pending, countedCopies := s.countedCopies + 1,
                      badRead := s.badRead || decide (0 < s.pendingHalf),
                      pending := 0, pendingHalf := 0,
                      life := { s.life with inFlight := s.life.inFlight + 1 } }
      else some s
    else s.step .landHalf
  | e => s.step e

#guard match ModelCheck.check ⟨init 1 1 1, stepGroupPartialShards⟩ (allEvents 1) violates 20 with
  | .counterexample _ => true
  | _ => false

/-- Mutant for A2: the layer's last stream counts before every block has
landed (`fresh = 0 ∧ half = 0` dropped from `streamLast`). The blocks that land
afterwards find the layer landed, are never copied, and the layer completes
without them. -/
def stepCountBeforeLand (s : S) : Ev → Option S
  | .streamLast =>
    if 0 < s.streams ∧ s.executed = false then
      let s := { s with executed := true }.executeLayer
      some { s with streams := s.streams - 1, life := s.life.endOpLocked }
    else none
  | e => s.step e

#guard match ModelCheck.check ⟨init 2 1 1, stepCountBeforeLand⟩ (allEvents 2) violates 40 with
  | .counterexample _ => true
  | _ => false

end S

end TpuSyncVerify.Transfer.PrefillDecode.GroupLanding

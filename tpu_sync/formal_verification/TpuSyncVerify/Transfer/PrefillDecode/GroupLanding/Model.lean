import TpuSyncVerify.Common.System
import TpuSyncVerify.Transfer.Session

/-! # `GroupLanding`: state, events, `step`, properties

Part of `TpuSyncVerify.Transfer.PrefillDecode.GroupLanding`; the module
docstring and source map live in `GroupLanding.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.GroupLanding

open TpuSyncVerify TpuSyncVerify.Transfer

/-- One layer of a pulled receive with push groups, as its blocks land and are
copied to the device. Blocks are counted by where they are, not named: the
C++ keeps them in `LayerH2d::pending` and `LayerH2d::taken`, and which block
is where changes nothing the properties observe. -/
structure S where
  /-- Host blocks of the layer in the load plan (`h2d_host_block_ids`). -/
  numBlocks : Nat
  /-- Blocks in a group: the least `k` with `k * block_bytes * num_shards`
  reaching `h2d_group_bytes_`. -/
  groupBlocks : Nat
  /-- Streams that may hold their op at once; bounds the instances the
  searches explore and plays no part in the proofs. -/
  maxStreams : Nat
  life : Lifecycle := {}
  /-- Push streams of the receive holding their op (`BeginIncomingPush` ..
  `EndIncomingPush`) (ghost). -/
  streams : Nat := 0
  /-- Blocks no stream has landed (ghost). -/
  fresh : Nat
  /-- Blocks one of whose two shard groups a stream that carries only some
  shards has landed (ghost). -/
  half : Nat := 0
  /-- Landed blocks that are neither waiting for a group nor taken by a copy:
  the streams that landed them carried only some shards, or the receive was
  already draining (ghost). -/
  untaken : Nat := 0
  /-- `LayerH2d::pending.size()`. -/
  pending : Nat := 0
  /-- Blocks of `pending` whose other shards are still in flight; always `0`,
  kept for the mutant that groups such blocks (ghost). -/
  pendingHalf : Nat := 0
  /-- Copies counted in `LayerH2d::outstanding` and `in_flight_` that
  `DispatchLayerH2d` has not handed to the device yet, and their blocks
  (ghost). -/
  countedCopies : Nat := 0
  counted : Nat := 0
  /-- Copies on the device whose `OnLayerH2dDone` has not run, and their blocks
  (ghost). `LayerH2d::outstanding` is `countedCopies + copyingCopies`. -/
  copyingCopies : Nat := 0
  copying : Nat := 0
  /-- Blocks whose copy finished OK (ghost). -/
  copied : Nat := 0
  /-- Blocks of copies that failed or were dropped (ghost). -/
  lost : Nat := 0
  /-- `LayerH2d::landed`: the layer's last stream counted and
  `ExecuteLayerH2d` took the rest. -/
  landed : Bool := false
  /-- The transport ran `OnLayerReceived` for the layer (its once-latch
  `on_layer_received_called`), whether or not `ExecuteLayerH2d` got past its
  `if (done_ || draining_)` (ghost). -/
  executed : Bool := false
  /-- `CompleteLayerLocked` calls for the layer (`num_completed_layers_`). -/
  completed : Nat := 0
  /-- A copy was handed to the device after the layer completed (ghost). -/
  late : Bool := false
  /-- A copy read a block whose shards had not all landed (ghost). -/
  badRead : Bool := false
  deriving Repr, DecidableEq, Hashable

/-- The events, one per C++ callback or lock scope (see the events table). -/
inductive Ev where
  | streamBegin
  | landFull (k : Nat)
  | landHalf
  | landOther
  | streamEnd
  | streamLast
  | dispatch (k : Nat) (ok : Bool)
  | copyDone (k : Nat) (ok : Bool)
  | cancel
  deriving Repr, DecidableEq

namespace S

/-- A receive whose pull was accepted, before any push. -/
def init (numBlocks groupBlocks maxStreams : Nat) : S :=
  { numBlocks, groupBlocks, maxStreams, fresh := numBlocks }

/-- `CompleteLayerLocked` for the receive's one layer: `++num_completed_layers_`
and, unless draining, the success `FinishLocked()` (the layer is the receive's
last). -/
def completeLayer (s : S) : S :=
  { s with completed := s.completed + 1,
           life := if s.life.draining = false then s.life.finishLocked true else s.life }

/-- A copy of `k` blocks may leave a pool of `blocks` blocks in `copies`
copies: every copy has at least one block. -/
def fits (k blocks copies : Nat) : Prop :=
  0 < k ∧ (copies = 1 → k = blocks) ∧ k + (copies - 1) ≤ blocks

instance (k blocks copies : Nat) : Decidable (fits k blocks copies) := by
  unfold fits; infer_instance

/-- `BeginIncomingPush`: the stream's lease (`TryBeginOp`), refused once the
receive is draining or settled. -/
def streamBegin (s : S) : Option S :=
  if s.streams < s.maxStreams then
    s.life.beginOp.map fun l => { s with life := l, streams := s.streams + 1 }
  else none

/-- A stream that carries every shard lands `k` blocks, then
`TransferReceiveSession::OnBlocksLanded`: a receive without push groups
(`groupBlocks = 0`), or that is draining, or whose layer has landed, ignores
the report; otherwise the blocks join
`LayerH2d::pending` and, once a group's worth waits, they form a copy counted
in `outstanding` and `in_flight_`. -/
def landFull (k : Nat) (s : S) : Option S :=
  if 0 < s.streams ∧ 0 < k ∧ k ≤ s.fresh then
    let s := { s with fresh := s.fresh - k }
    if s.groupBlocks = 0 ∨ s.life.done = true ∨ s.life.draining = true ∨ s.landed = true then
      some { s with untaken := s.untaken + k }
    else
      let s := { s with pending := s.pending + k }
      if s.groupBlocks ≤ s.pending then
        some { s with counted := s.counted + s.pending, countedCopies := s.countedCopies + 1,
                      badRead := s.badRead || decide (0 < s.pendingHalf),
                      pending := 0, pendingHalf := 0,
                      life := { s.life with inFlight := s.life.inFlight + 1 } }
      else some s
  else none

/-- A stream that carries only some shards lands one shard group of a block:
`OnBlocksLanded` ignores it (`shard_ids.size() != base_->num_shards()`). -/
def landHalf (s : S) : Option S :=
  if 0 < s.streams ∧ 0 < s.fresh then
    some { s with fresh := s.fresh - 1, half := s.half + 1 }
  else none

/-- The stream carrying the block's other shards lands them; `OnBlocksLanded`
ignores it too. The block has landed, untaken. -/
def landOther (s : S) : Option S :=
  if 0 < s.streams ∧ 0 < s.half then
    some { s with half := s.half - 1, untaken := s.untaken + 1 }
  else none

/-- A stream that does not complete the layer counts toward it
(`CompleteIncomingPush`), reports its blocks, and returns its lease
(`EndIncomingPush`). -/
def streamEnd (s : S) : Option S :=
  if 0 < s.streams then
    some { s with streams := s.streams - 1, life := s.life.endOpLocked }
  else none

/-- `ExecuteLayerH2d`'s first lock: a draining receive returns; otherwise the
layer has landed, and the blocks no copy took — waiting for a group, or never
reported in full — form one last copy counted in `outstanding` and
`in_flight_`; with none left, a layer whose copies have all finished
completes. -/
def executeLayer (s : S) : S :=
  if s.life.done = true ∨ s.life.draining = true then s
  else
    let rest := s.pending + s.untaken
    let s := { s with landed := true, pending := 0, untaken := 0, pendingHalf := 0 }
    if rest = 0 then
      if s.countedCopies + s.copyingCopies = 0 then s.completeLayer else s
    else
      { s with counted := s.counted + rest, countedCopies := s.countedCopies + 1,
               life := { s.life with inFlight := s.life.inFlight + 1 } }

/-- The layer's last stream counts once every block has landed (A1, A2):
`OnLayerReceived` → `executeLayer`; then the stream returns its lease. -/
def streamLast (s : S) : Option S :=
  if 0 < s.streams ∧ s.executed = false ∧ s.fresh = 0 ∧ s.half = 0 then
    let s := { s with executed := true }.executeLayer
    some { s with streams := s.streams - 1, life := s.life.endOpLocked }
  else none

/-- `DispatchLayerH2d` for a counted copy of `k` blocks: a receive that started
draining drops it (`--outstanding`, `EndRecvOpLocked()`); a dispatch failure
fails the receive; otherwise the copy is on the device. -/
def dispatch (k : Nat) (ok : Bool) (s : S) : Option S :=
  if 0 < s.countedCopies ∧ fits k s.counted s.countedCopies then
    let s := { s with counted := s.counted - k, countedCopies := s.countedCopies - 1 }
    if s.life.done = true ∨ s.life.draining = true then
      some { s with lost := s.lost + k, life := s.life.endOpLocked }
    else if ok then
      some { s with copying := s.copying + k, copyingCopies := s.copyingCopies + 1,
                    late := s.late || decide (0 < s.completed) }
    else
      some { s with lost := s.lost + k, life := (s.life.finishLocked false).endOpLocked }
  else none

/-- `OnLayerH2dDone` for a copy of `k` blocks: `--outstanding`; an OK copy that
leaves a landed layer with none outstanding completes it; a failed one fails
the receive; then `absl::Cleanup end_op`. -/
def copyDone (k : Nat) (ok : Bool) (s : S) : Option S :=
  if 0 < s.copyingCopies ∧ fits k s.copying s.copyingCopies then
    let s := { s with copying := s.copying - k, copyingCopies := s.copyingCopies - 1 }
    if ok then
      let s := { s with copied := s.copied + k }
      let s :=
        if s.landed = true ∧ s.countedCopies + s.copyingCopies = 0 then s.completeLayer else s
      some { s with life := s.life.endOpLocked }
    else
      some { s with lost := s.lost + k, life := (s.life.finishLocked false).endOpLocked }
  else none

/-- `Finish(error)`: the deadline sweep or any other failure. -/
def cancel (s : S) : Option S := some { s with life := s.life.finishLocked false }

def step (s : S) : Ev → Option S
  | .streamBegin => s.streamBegin
  | .landFull k => s.landFull k
  | .landHalf => s.landHalf
  | .landOther => s.landOther
  | .streamEnd => s.streamEnd
  | .streamLast => s.streamLast
  | .dispatch k ok => s.dispatch k ok
  | .copyDone k ok => s.copyDone k ok
  | .cancel => s.cancel

/-- A layer of `n` blocks with groups of `g` blocks, at most `m` streams at once. -/
def sys (n g m : Nat) : System S Ev := ⟨init n g m, step⟩

/-! ## Properties -/

/-- The layer completes once. -/
def CompletesOnce (s : S) : Prop := s.completed ≤ 1

/-- **Data on publication.** A layer that completed on a receive whose status
is OK has every block on the device, and no copy of it is counted or running. -/
def CompleteSound (s : S) : Prop :=
  0 < s.completed →
    s.countedCopies = 0 ∧ s.copyingCopies = 0 ∧ (s.life.statusOk = true → s.copied = s.numBlocks)

/-- No copy reaches the device after the layer completed. -/
def NoLateCopy (s : S) : Prop := s.late = false

/-- Every copy reads blocks whose shards have all landed. -/
def CopiesReadLanded (s : S) : Prop := s.badRead = false

/-- A settled receive has no copy counted or running. -/
def SettledQuiet (s : S) : Prop :=
  s.life.done = true → s.countedCopies = 0 ∧ s.copyingCopies = 0

/-- Groups are eager: fewer than a group's worth of blocks ever waits. -/
def GroupsEager (s : S) : Prop := s.pending < s.groupBlocks ∨ s.groupBlocks = 0

def Safe (s : S) : Prop :=
  CompletesOnce s ∧ CompleteSound s ∧ NoLateCopy s ∧ CopiesReadLanded s ∧ SettledQuiet s ∧
    GroupsEager s

instance (s : S) : Decidable (Safe s) := by
  unfold Safe CompletesOnce CompleteSound NoLateCopy CopiesReadLanded SettledQuiet GroupsEager
  infer_instance

/-- Executable negation of `Safe`. -/
def violates (s : S) : Bool := !decide (Safe s)

theorem violates_iff (s : S) : violates s = false ↔ Safe s := by
  simp [violates]

end S

end TpuSyncVerify.Transfer.PrefillDecode.GroupLanding

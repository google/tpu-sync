import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.BlockPipeline

/-! # `BlockOrdering`: concrete traces and regression checks

Part of `TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

namespace BlockOrdering

/-! ## 6. Concrete Traces & Regression Checks (`decide`) -/

section Checks

open Pipeline (Ev)

/-- 1-layer end-to-end trace helper (producer D2H → H2H → consumer land → H2D → publish). -/
def trace1Layer : List Ev :=
  [.send .beginPull, .send .start, .send .d2hBegin, .send (.d2hIssue true),
   .d2hReady 0, .send .d2hEnd, .send (.wake true), .send .h2hIssue,
   .send .sendNext, .h2hDone 0 true, .send .publish,
   .recv (.pullReply true), .recv .pushBegin, .land 0, .h2dBegin 0,
   .h2dIssue 0 true, .recv .netAccount, .recv .pushEnd, .h2dReady 0,
   .recv (.h2dDone true), .recv .publish]

/-- 2-layer out-of-order end-to-end trace helper (D2H finishes 1 then 0, pushes
complete 1 then 0, layer 1 lands and finishes H2D before layer 0). -/
def trace2LayersOutOfOrder : List Ev :=
  [.send .beginPull, .send .start, .send .d2hBegin, .send (.d2hIssue true),
   .send .d2hBegin, .send (.d2hIssue true),
   .d2hReady 1, .d2hReady 0, .send .d2hEnd, .send .d2hEnd,
   .send (.wake true), .send .h2hIssue, .send .sendNext,
   .send (.wake true), .send .h2hIssue, .send .sendNext,
   .h2hDone 1 true, .h2hDone 0 true, .send .publish,
   .recv (.pullReply true),
   .recv .pushBegin, .land 1, .h2dBegin 1, .h2dIssue 1 true, .recv .netAccount, .recv .pushEnd,
   .recv .pushBegin, .land 0, .h2dBegin 0, .h2dIssue 0 true, .recv .netAccount, .recv .pushEnd,
   .h2dReady 1, .h2dReady 0, .recv (.h2dDone true), .recv (.h2dDone true), .recv .publish]

/-- **Trace 1 (`SendLifecycleTest.DuplicateBlocksAreRejectedAtRegistration` in
`kv_cache_manager_with_transfer_send_drain_test.cc`):**
`PopulateRegisteredBlocks` rejects the registration `{5, 9, 5}` at `NotifyForRead`
(the test expects `std::invalid_argument` naming block 5 at positions 0 and 2
of 3, `has_send(15) == false`, nothing reported); the
duplicate-free `{0, 1, 2}` conjunct is the registration of
`ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged`. -/
theorem trace_duplicate_registration_rejected :
    validateRegistration [5, 9, 5] = false ∧
    validateRegistration [0, 1, 2] = true := by decide

/-- **Trace 2 (`ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged` in
`kv_cache_manager_with_transfer_control_test.cc`):**
`NotifyForRead` registers `{0, 1, 2}`; `PullStream` requests the reordered
subset `source_blocks = {2, 0}` into `destination_blocks = {6, 7}`.
`ValidateAndBeginPull` accepts the subset (the test asserts the handshake
`status == 0`); model-only beyond the test, the transfer delivers
`decodeHbm[0][6] = .kv 0 2` and `decodeHbm[0][7] = .kv 0 0`. -/
theorem trace_subset_pull_acknowledged :
    ((BlockPipeline.sys 1 8 2 [0, 1, 2] [⟨2, 6, 0⟩, ⟨0, 7, 1⟩]).run trace1Layer).map
      (fun s => (s.pipe.recv.published,
                 (s.decodeHbmB.getD 0 [])[6]?,
                 (s.decodeHbmB.getD 0 [])[7]?,
                 (s.decodeHbmB.getD 0 [])[1]?)) =
    some (some true, some (BlockCell.kv 0 2), some (BlockCell.kv 0 0), some BlockCell.blank) := by
  decide

/-- **Trace 3 (`ControlHandshakeTest.PullOfUnregisteredBlockIsRejected` in
`kv_cache_manager_with_transfer_control_test.cc`):**
`NotifyForRead` registers `{0, 1}`; `PullStream` requests `{0, 2}`.
`ValidateAndBeginPull` (`.send .beginPull`) is rejected (`none`) — the test
checks the producer's response — and, model-only, the consumer's pull fails
(`.recv (.pullReply false)`) with `pullStarted = false`. -/
theorem trace_unregistered_block_rejected :
    let sys := BlockPipeline.sys 1 4 2 [0, 1] [⟨0, 0, 0⟩, ⟨2, 1, 1⟩]
    sys.run [.send .beginPull] = none ∧
    (sys.run [.recv (.pullReply false), .recv .publish]).map
      (fun s => (s.pipe.send.pullStarted, s.pipe.recv.published)) =
    some (false, some false) := by
  decide

/-- **Trace 4 (`ControlHandshakeTest.PullWithDuplicateSourceBlockIsRejected` in
`kv_cache_manager_with_transfer_control_test.cc`):**
`NotifyForRead` registers `{0, 1}`; `PullStream` requests duplicate source
blocks `{0, 0}`. `ValidateAndBeginPull` (`.send .beginPull`) is rejected. -/
theorem trace_duplicate_source_block_rejected :
    (BlockPipeline.sys 1 4 2 [0, 1] [⟨0, 0, 0⟩, ⟨0, 1, 1⟩]).run [.send .beginPull] = none := by
  decide

/-- **Trace 5 (`ControlHandshakeTest.EmptyPullIsRejected` in
`kv_cache_manager_with_transfer_control_test.cc`):**
`NotifyForRead` registers `{0}`; `PullStream` requests `{}`.
`ValidateAndBeginPull` (`.send .beginPull`) is rejected. -/
theorem trace_empty_pull_rejected :
    (BlockPipeline.sys 1 4 2 [0] []).run [.send .beginPull] = none := by
  decide

/-- **Trace 6 (`KVCacheManagerWithTransferTest.LocalOrchestratedTransfer` in
`kv_cache_manager_with_transfer_test.cc`):**
Pull remote block `0` into local device block `1`: `decodeHbmB[0][1] = .kv 0 0`
while local device block `0` remains untouched (`.blank`). -/
theorem trace_local_orchestrated_transfer :
    ((BlockPipeline.sys 1 2 2 [0] [⟨0, 1, 0⟩]).run trace1Layer).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true, [[BlockCell.blank, BlockCell.kv 0 0]]) := by
  decide

/-- **Trace 7 (`KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock`
in `kv_cache_manager_with_transfer_test.cc`):**
Pull remote block `0` into local device block `1` using custom host block `4`
out of `6` host blocks (`allocateStagingForLoad [1] (.customHost [4]) = some [4]`):
- Device block `1` receives `.kv 0 0` while device block `0` stays `.blank`.
- Custom host block `4` receives `.kv 0 0` while host block `5` stays `.blank`. -/
theorem trace_custom_host_block_transfer :
    allocateStagingForLoad [1] (.customHost [4]) = some [4] ∧
    ((BlockPipeline.sys 1 2 6 [0] [⟨0, 1, 4⟩]).run trace1Layer).map
      (fun s => (s.pipe.recv.published,
                 s.decodeHbmB,
                 (s.decodeStagingB.getD 0 [])[4]?,
                 (s.decodeStagingB.getD 0 [])[5]?)) =
    some (some true, [[BlockCell.blank, BlockCell.kv 0 0]],
          some (BlockCell.kv 0 0), some BlockCell.blank) := by
  decide

/-- **Trace 8 (`def test_non_contiguous_blocks` in
`tpu_sync/api/jax/kv_cache_manager_transfer_test.py` and
`tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py`):**
2 layers completing out of order; `registered = [0, 2]`, `remote = [0, 2]`,
`local = [0, 1]` out of `3` blocks. For both layers `l ∈ {0, 1}`,
`decodeHbmB[l] = [.kv l 0, .kv l 2, .blank]` (local block `2` stays untouched). -/
theorem trace_non_contiguous_blocks :
    ((BlockPipeline.sys 2 3 2 [0, 2] [⟨0, 0, 0⟩, ⟨2, 1, 1⟩]).run trace2LayersOutOfOrder).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true, [[BlockCell.kv 0 0, BlockCell.kv 0 2, BlockCell.blank],
                      [BlockCell.kv 1 0, BlockCell.kv 1 2, BlockCell.blank]]) := by
  decide

/-- **Trace 9 (`def test_host_reordering` in
`tpu_sync/api/jax/kv_cache_manager_transfer_test.py` and
`tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py`):**
2 layers completing out of order; `registered = [0, 1]`, `remote = [1, 0]`
(reversed), `local = [0, 1]`. Both layers deliver `[.kv l 1, .kv l 0]` in
`decodeHbmB[l]` (`local[0] ← remote[1]`, `local[1] ← remote[0]`). -/
theorem trace_host_reordering :
    ((BlockPipeline.sys 2 2 2 [0, 1] [⟨1, 0, 0⟩, ⟨0, 1, 1⟩]).run trace2LayersOutOfOrder).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true, [[BlockCell.kv 0 1, BlockCell.kv 0 0],
                      [BlockCell.kv 1 1, BlockCell.kv 1 0]]) := by
  decide

/-- The 10-block non-contiguous reversed transfer configuration from
`def test_large_complex_non_contiguous_and_reorder` in
`tpu_sync/api/jax/kv_cache_manager_transfer_test.py` and
`tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py`. -/
def largeComplexRegistered : List Nat := [0, 2, 3, 5, 6, 7, 9, 11, 12, 14]
def largeComplexRemote : List Nat := largeComplexRegistered.reverse
def largeComplexLocal : List Nat := List.range 10
def largeComplexHost : List Nat := List.range 10

/-- **Trace 10 (`def test_large_complex_non_contiguous_and_reorder` in
`tpu_sync/api/jax/kv_cache_manager_transfer_test.py` and
`tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py`):**
16 blocks per layer, 2 layers completing out of order; `registered = [0, 2, 3, 5, 6, 7, 9, 11, 12, 14]`,
`remote = reversed(registered)`, `local = [0..9]`.
- `BuildCoalescedCopySpec` compresses the 10 sorted producer D2H blocks into 6
  contiguous DMA runs (`[0]`, `[2, 3]`, `[5, 6, 7]`, `[9]`, `[11, 12]`, `[14]`).
- For every layer `l ∈ {0, 1}`, local blocks `0..9` receive `.kv l remote[i]`
  and untouched local blocks `10..15` remain `.blank`. -/
theorem trace_large_complex_non_contiguous_and_reorder :
    let plan := buildLoadCopyPlan largeComplexRemote largeComplexLocal largeComplexHost
    let triples := zipTriples largeComplexRemote largeComplexLocal largeComplexHost
    plan.d2hSpec.length = 6 ∧
    ((BlockPipeline.sys 2 16 10 largeComplexRegistered triples).run trace2LayersOutOfOrder).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true,
          [(largeComplexRemote.map (BlockCell.kv 0)) ++ List.replicate 6 BlockCell.blank,
           (largeComplexRemote.map (BlockCell.kv 1)) ++ List.replicate 6 BlockCell.blank]) := by
  decide

/-! ### Mutant: Mismatched Transport Order (`BuildLoadCopyPlan`)

If `TransferReceiveSession::BuildLoadCopyPlan` sorted `producer_remote_block_ids`
by `remote_order` (`plan.producer_remote_block_ids.push_back(remote_block_ids[idx])`)
without applying the same `remote_order` permutation to
`transport_host_block_ids` (`plan.transport_host_block_ids.push_back(local_host_block_ids[idx])`),
then on `test_host_reordering` (`remote = [1, 0], local = [0, 1]`), `decodeHbm`
silently receives the un-permuted blocks `[.kv 0 0, .kv 0 1]` instead of
`[.kv 0 1, .kv 0 0]`. -/

def buildBuggyCopyPlan (triples : List BlockTriple) : CopyPlan :=
  let base := buildCopyPlanFromTriples triples
  -- Bug: `h2hPairs` uses unsorted `triples` instead of `remoteSorted`.
  let buggyH2hPairs := (enumFrom 0 triples).map (fun (t, j) => (j, t.host))
  { base with h2hPairs := buggyH2hPairs }

theorem trace_mutant_mismatched_transport_order :
    let triples : List BlockTriple := [⟨1, 0, 0⟩, ⟨0, 1, 1⟩]
    let goodHbm := h2dStage 0 2 2 (buildCopyPlanFromTriples triples)
    let buggyHbm := h2dStage 0 2 2 (buildBuggyCopyPlan triples)
    goodHbm = [BlockCell.kv 0 1, BlockCell.kv 0 0] ∧
    buggyHbm = [BlockCell.kv 0 0, BlockCell.kv 0 1] ∧
    buggyHbm ≠ goodHbm := by decide

end Checks


end BlockOrdering

end TpuSyncVerify.Transfer.PrefillDecode

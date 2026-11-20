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

/-- `SendLifecycleTest.DuplicateBlocksAreRejectedAtRegistration` and
`ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged`: `PopulateRegisteredBlocks`
rejects `{5, 9, 5}` and accepts `{0, 1, 2}`. -/
theorem trace_duplicate_registration_rejected :
    validateRegistration [5, 9, 5] = false ∧
    validateRegistration [0, 1, 2] = true := by decide

/-- `ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged`: reordered subset
`{2, 0}` of registered `{0, 1, 2}` into local `{6, 7}` delivers `.kv 0 2` and `.kv 0 0`. -/
theorem trace_subset_pull_acknowledged :
    ((BlockPipeline.sys 1 8 2 [0, 1, 2] [⟨2, 6, 0⟩, ⟨0, 7, 1⟩]).run trace1Layer).map
      (fun s => (s.pipe.recv.published,
                 (s.decodeHbmB.getD 0 [])[6]?,
                 (s.decodeHbmB.getD 0 [])[7]?,
                 (s.decodeHbmB.getD 0 [])[1]?)) =
    some (some true, some (BlockCell.kv 0 2), some (BlockCell.kv 0 0), some BlockCell.blank) := by
  decide

/-- `ControlHandshakeTest.PullOfUnregisteredBlockIsRejected`: requesting `{0, 2}`
against registered `{0, 1}` is rejected by `ValidateAndBeginPull`. -/
theorem trace_unregistered_block_rejected :
    let sys := BlockPipeline.sys 1 4 2 [0, 1] [⟨0, 0, 0⟩, ⟨2, 1, 1⟩]
    sys.run [.send .beginPull] = none ∧
    (sys.run [.recv (.pullReply false), .recv .publish]).map
      (fun s => (s.pipe.send.pullStarted, s.pipe.recv.published)) =
    some (false, some false) := by
  decide

/-- `ControlHandshakeTest.PullWithDuplicateSourceBlockIsRejected`: duplicate
requested source blocks `{0, 0}` are rejected by `ValidateAndBeginPull`. -/
theorem trace_duplicate_source_block_rejected :
    (BlockPipeline.sys 1 4 2 [0, 1] [⟨0, 0, 0⟩, ⟨0, 1, 1⟩]).run [.send .beginPull] = none := by
  decide

/-- `ControlHandshakeTest.EmptyPullIsRejected`: an empty block request `{}` is
rejected by `ValidateAndBeginPull`. -/
theorem trace_empty_pull_rejected :
    (BlockPipeline.sys 1 4 2 [0] []).run [.send .beginPull] = none := by
  decide

/-- `KVCacheManagerWithTransferTest.LocalOrchestratedTransfer`: pulling remote
block `0` into local block `1` leaves local block `0` `.blank`. -/
theorem trace_local_orchestrated_transfer :
    ((BlockPipeline.sys 1 2 2 [0] [⟨0, 1, 0⟩]).run trace1Layer).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true, [[BlockCell.blank, BlockCell.kv 0 0]]) := by
  decide

/-- `KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock`:
lands remote block `0` in custom host block `4` (leaving `5` `.blank`) and local block `1`. -/
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

/-- `test_non_contiguous_blocks` (JAX/Torch): 2 out-of-order layers gather
`remote = [0, 2]` into `local = [0, 1]`, leaving block `2` `.blank`. -/
theorem trace_non_contiguous_blocks :
    ((BlockPipeline.sys 2 3 2 [0, 2] [⟨0, 0, 0⟩, ⟨2, 1, 1⟩]).run trace2LayersOutOfOrder).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true, [[BlockCell.kv 0 0, BlockCell.kv 0 2, BlockCell.blank],
                      [BlockCell.kv 1 0, BlockCell.kv 1 2, BlockCell.blank]]) := by
  decide

/-- `test_host_reordering` (JAX/Torch): 2 out-of-order layers with reversed
`remote = [1, 0]` into `local = [0, 1]` deliver `[.kv l 1, .kv l 0]`. -/
theorem trace_host_reordering :
    ((BlockPipeline.sys 2 2 2 [0, 1] [⟨1, 0, 0⟩, ⟨0, 1, 1⟩]).run trace2LayersOutOfOrder).map
      (fun s => (s.pipe.recv.published, s.decodeHbmB)) =
    some (some true, [[BlockCell.kv 0 1, BlockCell.kv 0 0],
                      [BlockCell.kv 1 1, BlockCell.kv 1 0]]) := by
  decide

/-- Configuration for `test_large_complex_non_contiguous_and_reorder` (JAX/Torch). -/
def largeComplexRegistered : List Nat := [0, 2, 3, 5, 6, 7, 9, 11, 12, 14]
def largeComplexRemote : List Nat := largeComplexRegistered.reverse
def largeComplexLocal : List Nat := List.range 10
def largeComplexHost : List Nat := List.range 10

/-- `test_large_complex_non_contiguous_and_reorder` (JAX/Torch): coalesces 10
blocks into 6 DMA runs and delivers reversed `remote` into `local = [0..9]`. -/
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

/-! ### Mutant: mismatched transport order (`buildBuggyCopyPlan`)

Omitting `remote_order` on `transport_host_block_ids` in `BuildLoadCopyPlan`
delivers `[.kv 0 0, .kv 0 1]` instead of `[.kv 0 1, .kv 0 0]` on `test_host_reordering`. -/

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

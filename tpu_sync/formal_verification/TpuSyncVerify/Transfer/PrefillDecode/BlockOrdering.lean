import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Validation
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Coalescing
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.CopyPlan
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Cancellation
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.BlockPipeline
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Traces

/-!
# Within-Layer Block-Index Gather, Reordering, Subset Validation & Custom Host Staging

`send.cc`, `recv.cc`, `bt.cc` and `mgr.cc` abbreviate as in `Pipeline.lean`.

## Source map

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|------|-----------------|-------|
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::Create` | `:64-89` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::Create` (`std::optional<DuplicateBlock> duplicate_block =` .. `" of ", block_ids.size(), " blocks"));`) | `:79-86` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::PopulateRegisteredBlocks` | `:106-124` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::ValidateRequestedBlocksLocked` | `:126-143` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::ValidateRequestedBlocksLocked` (`if (requested_block_ids.empty())`) | `:128-131` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::ValidateRequestedBlocksLocked` (`if (registered_block_set_.find(block_id) == registered_block_set_.end())`) | `:134-137` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::ValidateRequestedBlocksLocked` (`if (!seen.insert(block_id).second)`) | `:138-141` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::ValidateAndBeginPull` | `:145-159` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::BuildCoalescedCopySpec` | `:345-374` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::StartPush` | `:412-559` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::StartPush` (`absl::Span<const int> blocks = staging_.block_ids();` .. `CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);`) | `:453-475` |
| `tpu_sync/kv_cache/transfer_send_session.cc` | `TransferSendSession::StartPush` (`CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);` .. `: absl::StatusOr<raiden::PjRtCopyFuture>(injected);`) | `:475-532` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::NotifyForRead` (`if (!created.ok())` .. `throw std::invalid_argument(std::string(created.status().message()));`) | `:674-675` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::AllocateStagingForLoad` | `:299-345` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::AllocateStagingForLoad` (`if (local_host_block_ids.has_value())`) | `:304-307` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::AllocateStagingForLoad` (`absl::flat_hash_set<int64_t> unique_local_bids` .. `host_block_ids->push_back(it->second);`) | `:312-341` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::BuildLoadCopyPlan` | `:347-415` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::BuildLoadCopyPlan` (`std::vector<size_t> remote_order(remote_block_ids.size());` .. `plan.transport_host_block_ids.push_back(local_host_block_ids[original_idx]);`) | `:366-380` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::BuildLoadCopyPlan` (`std::vector<size_t> local_order(local_block_ids.size());` .. `plan.h2d_host_block_ids, plan.h2d_local_block_ids);`) | `:384-412` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::BuildLoadCopyPlan` (`if (plan.h2d_local_block_ids.empty() ||` .. `"Duplicate local block IDs must map to the same host block ID");`) | `:399-406` |
| `tpu_sync/kv_cache/transfer_receive_session.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`base_->H2dSyncDispatch(copy_spec.src_offsets, copy_spec.dst_offsets,`) | `:842-846` |
| `tpu_sync/transport/block_transport_session.cc` | `BlockTransportServerPushSession::StartStream` (`ABSL_RETURN_IF_ERROR(ForEachPayload(` .. `total_received_bytes += expected_size;`) | `:381-426` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_send_drain_test.cc` | `SendLifecycleTest.DuplicateBlocksAreRejectedAtRegistration` | `:586-606` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc` | `ControlHandshakeTest.UniqueRegisteredSubsetIsAcknowledged` | `:356-368` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc` | `ControlHandshakeTest.PullOfUnregisteredBlockIsRejected` | `:384-397` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc` | `ControlHandshakeTest.PullWithDuplicateSourceBlockIsRejected` | `:399-412` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc` | `ControlHandshakeTest.EmptyPullIsRejected` | `:414-426` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_test.cc` | `KVCacheManagerWithTransferTest.LocalOrchestratedTransfer` | `:111-290` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_test.cc` | `KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock` | `:324-438` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer_test.cc` | `KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock` (`// Host Block 5 should still be 0` .. `EXPECT_EQ(host_block_5_float[i], 0.0f);`) | `:434-436` |
| `tpu_sync/api/jax/kv_cache_manager_transfer_test.py` | `test_non_contiguous_blocks` | `:187-271` |
| `tpu_sync/api/jax/kv_cache_manager_transfer_test.py` | `test_host_reordering` | `:274-356` |
| `tpu_sync/api/jax/kv_cache_manager_transfer_test.py` | `test_large_complex_non_contiguous_and_reorder` | `:359-448` |
| `tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py` | `test_non_contiguous_blocks` | `:176-206` |
| `tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py` | `test_host_reordering` | `:209-237` |
| `tpu_sync/api/torch/tests/kv_cache_manager_transfer_test.py` | `test_large_complex_non_contiguous_and_reorder` | `:240-282` |
<!-- source-map:end -->

Stage 4 (spatial block refinement) of the prefill-to-decode model: refines
`Pipeline`'s one-`Cell`-per-layer abstraction (`Pipeline` Assumption A1) down to
per-block arrays `Layer → BlockId → BlockCell`, modelling:

1. **Producer block registration & pull validation**
   (`TransferSendSession::PopulateRegisteredBlocks` and
   `TransferSendSession::ValidateRequestedBlocksLocked`):
   - `PopulateRegisteredBlocks`: `NotifyForRead` rejects duplicate block IDs
     while populating `registered_block_set_` (reported with both positions by
     `TransferSendSession::Create` and thrown as `std::invalid_argument` by
     `KVCacheManagerWithTransfer::NotifyForRead`).
   - `ValidateRequestedBlocksLocked`: `ValidateAndBeginPull` rejects empty
     requests, rejects any block ID not in `registered_block_set_` (allowing
     strict subset pulls), and rejects duplicate producer block IDs.
2. **Contiguous-run DMA coalescing** (`TransferSendSession::BuildCoalescedCopySpec`):
   - Merges adjacent `(src_block_ids[k], dst_block_ids[k])` pairs with
     `src[k] == src[k-1] + 1 && dst[k] == dst[k-1] + 1` into contiguous
     `(src_offset, dst_offset, size)` DMA runs (`CopyRun`).
   - Proves `execCoalesced_buildCoalescedSpec`: executing the coalesced DMA runs
     is extensionally identical on every memory state to executing elementwise
     block-by-block copies (`execElementwise`).
3. **Consumer staging allocation & custom host staging blocks**
   (`TransferReceiveSession::AllocateStagingForLoad`):
   - Supports both caller-supplied custom host block IDs
     (`if (local_host_block_ids.has_value())`, tested in
     `KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock`)
     and allocator-backed host staging
     (`staging_allocator_->Acquire(unique_local_bids.size())`).
4. **Dual-permutation `BuildLoadCopyPlan`**
   (`TransferReceiveSession::BuildLoadCopyPlan` and `TransferSendSession::StartPush`):
   - **Transport order (`remote_order` in `BuildLoadCopyPlan`):** stable-sorts
     request indices by `remote_block_ids` so both `producer_remote_block_ids`
     (staged into producer host slots `0 … k-1` by `StartPush`) and
     `transport_host_block_ids` (where `BlockTransport` lands incoming blocks on
     the consumer host) are permuted by the same permutation $\pi_{\text{remote}}$.
   - **H2D order (`local_order` in `BuildLoadCopyPlan`):** stable-sorts request
     indices by `local_block_ids`, deduplicates identical `(local_bid, host_bid)`
     pairs, and coalesces contiguous `(h2d_host_block_ids, h2d_local_block_ids)`
     runs via `BuildCoalescedCopySpec`.
   - Proves that $\pi_{\text{remote}}$ and $\pi_{\text{local}}$ cancel across the
     3-stage copy chain (`d2hStage` → `landStage` → `h2dStage`), routing every
     requested `remote_block_ids[i]` into `decodeStaging[local_host_block_ids[i]]`
     and `decodeHbm[local_block_ids[i]]` while leaving all non-targeted host and
     device blocks untouched.
5. **Multi-layer asynchronous `BlockPipeline` integration**:
   - Couples the dual-permutation `CopyPlan` and coalesced DMA specs with
     `Pipeline`'s asynchronous layer-indexed state machine (`d2hReady l`,
     `h2hDone l ok`, `land l`, `h2dReady l`, `reclaim`, `reseatPrefillStaging`,
     `reseatDecodeStaging`).
   - Proves `BlockPipeline.reachable_safe`: on every reachable state, all
     temporal session/buffer/handshake properties (`Pipeline.Safe`) hold together
     with `BlockValidationSafe`, `BlockPublicationCorrect`, and
     `CustomHostStagingCorrect`.
-/

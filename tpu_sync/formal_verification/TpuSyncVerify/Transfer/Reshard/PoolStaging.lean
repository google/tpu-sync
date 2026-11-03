import TpuSyncVerify.Transfer.Reshard.PoolStaging.Arena
import TpuSyncVerify.Transfer.Reshard.PoolStaging.ArenaLemmas
import TpuSyncVerify.Transfer.Reshard.PoolStaging.Registration
import TpuSyncVerify.Transfer.Reshard.PoolStaging.Invariant
import TpuSyncVerify.Transfer.Reshard.PoolStaging.Traces

/-!
# Bounded pool host staging and per-UUID lease lifecycle

Formal model and proofs of bounded host staging (`PoolStagingStorage`) and
per-UUID page-lease lifecycle in `KVCacheManagerBase` and
`KVCacheManagerWithTransfer`.

`base.h`/`base.cc` are `tpu_sync/kv_cache/kv_cache_manager_base.{h,cc}`;
`mgr.cc` is `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`;
`reshard_recv.h`/`reshard_recv.cc` are
`tpu_sync/kv_cache/reshard_receive_session.{h,cc}`; `pool_reshard_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`. Exact
line ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

  ## C++ anchorage

  1. **Bounded staging arena (`PoolStagingStorage` in `base.h`,
     `KVCacheManagerBase::ConfigurePoolStaging` in `base.cc`):**
     When a storage's pools declare `staging_blocks_per_request > 0` and
     `staging_leases > 0` such that `num_slots = staging_leases *
     blocks_per_lease < max_pool_blocks`, `ConfigurePoolStaging` allocates a
     bounded host arena of `num_slots` page slots (`free_slots = [num_slots-1,
     ..., 0]`, so `.back()` pops `0, 1, ...` in order).

  2. **Idempotent per-UUID lease acquisition (`KVCacheManagerBase::AcquirePoolStagingLease` in `base.cc`):**
     Filters `block_ids` down to `missing` — deduplicated block IDs not already
     present in `st.leases[uuid]->slot_by_block` (`if (cur != nullptr && cur->slot_by_block.contains(block_id)) continue;`). Multiple pools
     sharing a physical storage (`storage_index`) therefore share one page lease
     under the same `uuid` without double-allocating slots. If
     `missing.size() > st.num_slots` or `st.free_slots.size() < missing.size()`,
     acquisition fails with `ResourceExhausted` (`if (static_cast<int64_t>(missing.size()) > st.num_slots)` and `while (st.free_slots.size() < missing.size())`). Otherwise it
     pops `missing.size()` slots from `free_slots` and records `block_id ↦ slot`
     in `st.leases[uuid]` (`for (int64_t block_id : missing)` through `st.leases[uuid] = std::move(lease)`).

  3. **Lease release (`KVCacheManagerBase::ReleasePoolStagingLeases` in `base.cc`):**
     Looks up `st.leases.find(uuid)`, pushes every `slot ∈ lease->slots` back
     onto `st.free_slots`, and erases `uuid` from `st.leases`.

  4. **Receive registration lifecycle (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` in `mgr.cc`,
     `ReshardReceiveSession::Create`, `ReshardReceiveSession::AcquireStagingLeases`, `ReshardReceiveSession::ReleaseStaging` in `reshard_recv.cc`):**
     `PoolReshardRegisterRecv` executes in three steps:
     - Phase 1 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` checking `active_pool_reshard_recvs_`,
       `ReshardReceiveSession::Create` claiming `plan.uuid()`): checks `active_pool_reshard_recvs_`
       under `mu_` and claims `plan.uuid()` via
       `base->RegisterActivePlanDirect(plan.uuid(), plan, /*is_sender=*/false)`
       before constructing any `ReshardReceiveSession` object.
     - Phase 2 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` calling `ReshardReceiveSession::Create` and `ReshardReceiveSession::AcquireStagingLeases`): constructs `ReshardReceiveSession`
       and calls `AcquireStagingLeases(plan)`,
       which sets `staging_released_ = false` and calls
       `base_->AcquirePoolStagingLease(uuid_, ...)`. On failure,
       `Create` calls `session->ReleaseStaging()` (`ReleaseStagingLocked`) while
       `uuid` is still claimed and rolls back
       `base->UnregisterActivePlanDirect(plan.uuid())`.
     - Phase 3 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`, `active_pool_reshard_recvs_[plan.uuid()] = std::move(recv_session);` under `mu_`):
       inserts the session into `active_pool_reshard_recvs_` (and, for a
       sub-plan of a chained request, records it in the request's
       `recv_chains_` entry, `NoteSubPlanRegisteredLocked`, in the same scope).
     Under the unreserved mutant (`RegistrationPolicy.leaseBeforeReserve`),
     if Phase 2 ran before `RegisterActivePlanDirect`
     claimed `uuid`, two concurrent or duplicate `PoolReshardRegisterRecv`
     calls with the same `uuid` would share `st.leases[uuid]` in Phase 2; when
     `RegisterActivePlanDirect` rejected the second call (`AlreadyExistsError`),
     destroying its `ReshardReceiveSession` would run
     `~ReshardReceiveSession() -> ReleaseStaging() -> ReleasePoolStagingLeases(uuid)`
     (`ReshardReceiveSession::~ReshardReceiveSession` in `reshard_recv.h`, `ReshardReceiveSession::ReleaseStagingLocked` and `ReshardReceiveSession::ReleaseStaging` in `reshard_recv.cc`),
     erasing `st.leases[uuid]` and returning the surviving session's slots to
     `free_slots` while the surviving session is still active.

  ## Main results

  - `missingBlocksAux_empty_of_covered` & `acquireLease_idempotent`: re-acquiring
    a subset of already-leased block IDs for the same `uuid` is a no-op
    (`KVCacheManagerBase::AcquirePoolStagingLease`: `if (cur != nullptr && cur->slot_by_block.contains(block_id)) continue;` and `if (missing.empty()) return absl::OkStatus();`).
  - `ArenaInv.conservation` & `allocateMissing_slot_sum`:
    total slots across `freeSlots` and all active leases equals `numSlots`.
  - `no_slot_aliasing_between_distinct_uuids`: when active leases hold disjoint
    slot lists (`ArenaInv.nodupSlots` / `leases_disjoint_of_nodup`), distinct
    `uuid`s never resolve any block ID to the same host staging slot.
  - `reachable_reserveBeforeLease_safe`: under the shipping `reserveBeforeLease`
    protocol (claiming `uuid` before acquiring staging leases), every reachable
    state keeps active session leases live and pairwise slot-disjoint.
  - `trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`: constructive
    counterexample trace under the `leaseBeforeReserve` mutant showing a
    rejected duplicate-UUID arm freeing the survivor's lease
    and a third session (`uuid = 8`) receiving the same physical slot `0`.

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `base.h` | `PoolStagingLease` | `:1006-1009` |
| `base.h` | `PoolStagingStorage` | `:1010-1018` |
| `base.cc` | `RemoteAddress` | `:154-175` |
| `base.cc` | `KVCacheManagerBase::ConfigurePoolStaging` | `:2012-2115` |
| `base.cc` | `KVCacheManagerBase::PoolStagingLeaseSnapshot` | `:2123-2134` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` | `:2136-2193` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` (`std::vector<int64_t> missing;` .. `if (missing.empty())`) | `:2147-2161` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` (`if (static_cast<int64_t>(missing.size()) > st.num_slots)` .. `while (st.free_slots.size() < missing.size())`) | `:2164-2172` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` (`auto lease = std::make_shared<PoolStagingLease>();` .. `st.leases[uuid] = std::move(lease);`) | `:2181-2191` |
| `base.cc` | `KVCacheManagerBase::ReleasePoolStagingLeases` | `:2195-2211` |
| `base.cc` | `KVCacheManagerBase::PoolBlockHostBase` | `:2235-2280` |
| `base.cc` | `KVCacheManagerBase::PoolHostBaseAddrs` | `:2409-2456` |
| `base.cc` | `KVCacheManagerBase::CopyPoolBlocks` (`std::shared_ptr<const PoolStagingLease> lease;` .. `auto slot_it = lease->slot_by_block.find(block_id);`) | `:2546-2577` |
| `base.cc` | `KVCacheManagerBase::RegisterActivePlan(host_block_of)` (`if (auto [it, inserted] = active_plans_.try_emplace(`) | `:3210-3217` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`std::shared_ptr<const PoolStagingLease> staging_lease;` .. `const auto block_host_base =`) | `:3303-3358` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`.raddr = RemoteAddress(dst_addrs, entry.dst_shard_idx(),`) | `:3503-3505` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:873-910` |
| `reshard_recv.h` | `ReshardReceiveSession` (`~ReshardReceiveSession() override`) | `:71` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` | `:351-390` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` (`ABSL_RETURN_IF_ERROR(` .. `return session;`) | `:376-389` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:422-461` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:463-467` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:469-472` |
| `pool_staging_test.cc` | `PoolStagingTest.BoundedArenaLeasesAndAddressing` | `:115-208` |
| `pool_staging_test.cc` | `PoolStagingTest.FallsBackToFullMirrorWithoutHintsOrLeases` | `:212-235` |
| `pool_staging_test.cc` | `PoolStagingTest.SmallPoolIdentitySharedStorageAndStrideMismatch` | `:240-301` |
| `pool_staging_test.cc` | `PoolStagingTest.SenderRaddrMatchesReceiverPointer` | `:306-416` |
| `pool_staging_test.cc` | `PoolStagingTest.SenderRaddrUsesReceiverLeaseSlot` | `:419-484` |
| `pool_staging_test.cc` | `PoolStagingTest.PoolHostBaseAddrs` | `:489-521` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.DuplicatePoolRecvRegistrationKeepsOriginalStagingLease` | `:624-674` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.ConcurrentDuplicatePoolRecvRegistrationKeepsWinnerStagingLease` | `:676-732` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.StagingAcquireFailureReleasesUuidClaim` | `:734-764` |
<!-- source-map:end -->
-/

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
`reshard_recv.cc` is `tpu_sync/kv_cache/reshard_receive_session.cc` (which
holds the class `ReshardReceiveSession` behind the factory
`CreateReshardReceiveSession`); `pool_reshard_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc`. Exact
line ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

  ## C++ anchorage

  1. **Bounded staging arena (`PoolStagingStorage` in `base.h`,
     `KVCacheManagerBase::ConfigurePoolStaging` in `base.cc`):**
     A storage is bounded when it is device-backed, `staging_leases > 0`,
     every pool on it declares `staging_blocks_per_request > 0`
     (`agg.hint_min > 0`), its pools agree on a positive `block_stride_bytes`
     and a positive `num_blocks`, and `num_slots = staging_leases *
     blocks_per_lease` (`blocks_per_lease = agg.hint_max`) is smaller than that
     `num_blocks` (`if (slots >= agg.num_blocks) bounded = false;`).
     `ConfigurePoolStaging` then allocates a bounded host arena of `num_slots`
     page slots (`free_slots = [num_slots-1, ..., 0]`, so `.back()` pops
     `0, 1, ...` in order). Any other device-backed storage with pools, including one whose
     pools disagree on stride or block count, is accepted and keeps a full
     host mirror with identity addressing, on which
     `AcquirePoolStagingLease` returns OK without leasing anything. On a full host mirror
     (`layers_.size() == num_layers_`), each storage keeps its own arena or
     extent size; when host layers are bounded (`layers_.size() < num_layers_`),
     each allocated host layer is grown to `max_arena_bytes` across all storages
     (`PoolStagingTest.FullHostMirrorKeepsHeterogeneousStorageSizes`).

  2. **Idempotent per-UUID lease acquisition (`KVCacheManagerBase::AcquirePoolStagingLease` in `base.cc`):**
     Filters `block_ids` down to `missing` — deduplicated block IDs not already
     present in `st.leases[uuid]->slot_by_block` (`if (cur != nullptr && cur->slot_by_block.contains(block_id)) continue;`). Multiple pools
     sharing a physical storage (`storage_index`) therefore share one page lease
     under the same `uuid` without double-allocating slots. If
     `missing.size() > st.num_slots`, acquisition fails at once with
     `ResourceExhausted` (`if (static_cast<int64_t>(missing.size()) > st.num_slots)`).
     While `st.free_slots.size() < missing.size()` the caller waits on
     `staging_cv_` (releasing `staging_mu_`) until enough slots are free or
     `timeout` (the callers pass `pool_staging_lease_timeout()`) elapses, and
     only then fails with `ResourceExhausted`
     (`while (st.free_slots.size() < missing.size())`). Otherwise it
     pops `missing.size()` slots from `free_slots` and records `block_id ↦ slot`
     in `st.leases[uuid]` (`for (int64_t block_id : missing)` through `st.leases[uuid] = std::move(lease)`).
     `Arena.acquireLease = none` stands for the first failure or for a wait
     that timed out; the wait itself is not modelled (see "Not modelled").

  3. **Lease release (`KVCacheManagerBase::ReleasePoolStagingLeases` in `base.cc`):**
     Looks up `st.leases.find(uuid)`, pushes every `slot ∈ lease->slots` back
     onto `st.free_slots`, and erases `uuid` from `st.leases`.

  4. **Receive registration lifecycle (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` in `mgr.cc`,
     `CreateReshardReceiveSession`, `ReshardReceiveSession::AcquireStagingLeases`, `ReshardReceiveSession::ReleaseStaging` in `reshard_recv.cc`):**
     `PoolReshardRegisterRecv` executes in three steps:
     - Phase 1 (one `mu_` scope of `KVCacheManagerWithTransfer::PoolReshardRegisterRecv`):
       looks `plan.uuid()` up in `sessions_` — a settled incumbent is retired
       (`RetireSettledSessionLocked`), a live one is refused
       (`AlreadyExistsError`) — and claims the uuid in
       `pending_pool_reshard_recvs_`
       (`if (!pending_pool_reshard_recvs_.insert(plan.uuid()).second)` refuses
       a concurrent duplicate) before constructing any `ReshardReceiveSession`
       object; the retired incumbent's leases are released outside the lock
       (`ReleasePoolStagingLeases`) while the claim already blocks a third arm.
     - Phase 2 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv` calling `CreateReshardReceiveSession` and `ReshardReceiveSession::AcquireStagingLeases`): constructs `ReshardReceiveSession`
       and calls `AcquireStagingLeases(plan)`,
       which sets `staging_released_ = false` and calls
       `base_->AcquirePoolStagingLease(uuid(), ...)`. On failure,
       the factory calls `session->ReleaseStaging()` (`ReleaseStagingLocked`) while
       `uuid` is still claimed, and the `absl::Cleanup clear_pending` of
       `PoolReshardRegisterRecv` drops the claim afterwards.
     - Phase 3 (`KVCacheManagerWithTransfer::PoolReshardRegisterRecv`, `sessions_[plan.uuid()] = SessionEntry{recv_session, plan.req_id(),` under `mu_`):
       drops the pending claim and inserts the session into `sessions_` (as a
       `SessionEntry` with `SessionRole::kPoolReshardReceive`) in one scope
       (and, for a sub-plan of a chained request, records it in the request's
       `sub_plan_chains_` entry, `NoteSubPlanRegisteredLocked`).

     *Post-poll lease cleanup.* In `CompleteReadWithDetails`,
     `RetireSettledSessionLocked` and the `std::erase_if(retired_uuids, ...)`
     filter (`sessions_.contains(uuid) || pending_pool_reshard_recvs_.contains(uuid)`)
     execute inside the same `mu_` hold, while `base_->ReleasePoolStagingLeases(uuid)`
     runs unlocked after `mu_` is released. Because `PoolReshardRegisterRecv`
     cannot hold a pending claim in `pending_pool_reshard_recvs_` for a `uuid`
     that was just seated in `sessions_`, that filter does not prevent a
     concurrent same-UUID `PoolReshardRegisterRecv` arriving during the unlocked
     post-poll cleanup from seating a fresh `PoolStagingLease` and having
     `CompleteReadWithDetails` release it; single-step `settleSession` therefore
     assumes **fresh UUIDs across poll retirement** (assumed of the caller: a
     `uuid` being retired by `CompleteReadWithDetails` is not concurrently
     re-registered during its unlocked post-poll cleanup; `ReshardCoordinator`
     uses a caller-supplied `args.uuid` when `args.uuid > 0` and draws a fresh
     UUID only when `args.uuid <= 0`, `Reshard/Coordinator.lean`).

     *Unreserved mutant (`RegistrationPolicy.leaseBeforeReserve`).* If Phase 2
     ran before the uuid was claimed, two concurrent or duplicate
     `PoolReshardRegisterRecv` calls with the same `uuid` would share
     `st.leases[uuid]` in Phase 2; when the second call was then refused
     (`AlreadyExistsError`), destroying its `ReshardReceiveSession` would run
     `~ReshardReceiveSession() -> ReleaseStaging() -> ReleasePoolStagingLeases(uuid)`
     (`ReshardReceiveSession::~ReshardReceiveSession`,
     `ReshardReceiveSession::ReleaseStagingLocked` and
     `ReshardReceiveSession::ReleaseStaging` in `reshard_recv.cc`), erasing
     `st.leases[uuid]` and returning the surviving session's slots to
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
    protocol of `PoolReshardRegisterRecv` (claiming `uuid` before acquiring
    staging leases), every reachable state keeps active receive sessions'
    leases live and pairwise slot-disjoint. Send registration is not part of
    this model (see "Not modelled").
  - `trace_duplicate_uuid_frees_survivor_lease_and_aliases_slot`: constructive
    counterexample trace under the `leaseBeforeReserve` mutant showing a
    rejected duplicate-UUID arm freeing the survivor's lease
    and a third session (`uuid = 8`) receiving the same physical slot `0`.

  ## Not modelled

  - **The timed wait in `AcquirePoolStagingLease`.** A caller short of free
    slots blocks for up to `pool_staging_lease_timeout()` while holding what it
    already has: the receive factory inside `PoolReshardRegisterRecv` holds the
    pending uuid claim and the storages it leased earlier
    (`ReshardReceiveSession::AcquireStagingLeases` leases storage by storage);
    `ReshardSendSession::ExecutePushPipeline` holds its caller op, earlier
    pools' leases and (under `HostLayersBounded()`) a host layer slot. The model has no waiting state: an
    acquisition either succeeds or returns `none`.
  - **Send registration.** `KVCacheManagerWithTransfer::PoolReshardPush` checks
    `sessions_` for the uuid in one `mu_` scope and assigns
    `sessions_[plan.uuid()]` in a later one, with no counterpart of the
    `pending_pool_reshard_recvs_` claim in between, and the send session leases
    under the same uuid. `RegState` has only the `PoolReshardRegisterRecv`
    arms; concurrent sends with one uuid are assumed away (assumed of the
    caller as a fresh uuid per attempt, as for the poll's unlocked cleanup
    above).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `base.h` | `PoolStagingLease` | `:948-951` |
| `base.h` | `PoolStagingStorage` | `:952-960` |
| `base.cc` | `RemoteAddress` | `:146-167` |
| `base.cc` | `KVCacheManagerBase::ConfigurePoolStaging` | `:2437-2551` |
| `base.cc` | `KVCacheManagerBase::PoolStagingLeaseSnapshot` | `:2559-2570` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` | `:2572-2629` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` (`std::vector<int64_t> missing;` .. `if (missing.empty())`) | `:2583-2597` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` (`if (static_cast<int64_t>(missing.size()) > st.num_slots)` .. `while (st.free_slots.size() < missing.size())`) | `:2600-2608` |
| `base.cc` | `KVCacheManagerBase::AcquirePoolStagingLease` (`auto lease = std::make_shared<PoolStagingLease>();` .. `st.leases[uuid] = std::move(lease);`) | `:2617-2627` |
| `base.cc` | `KVCacheManagerBase::ReleasePoolStagingLeases` | `:2631-2647` |
| `base.cc` | `KVCacheManagerBase::PoolHostBaseAddrs` | `:2801-2851` |
| `base.cc` | `KVCacheManagerBase::CopyPoolBlocks` (`std::shared_ptr<const PoolStagingLease> lease;` .. `auto slot_it = lease->slot_by_block.find(block_id);`) | `:2950-2981` |
| `base.cc` | `KVCacheManagerBase::GetBlockChunks` (`std::shared_ptr<const PoolStagingLease> staging_lease;` .. `const auto block_host_base =`) | `:3446-3525` |
| `tpu_sync/kv_cache/transfer_plan.cc` | `TransferPlan::ResolvePlanBlockChunks` (`.raddr = RemoteAddress(dst_addrs, entry.dst_shard_idx(),`) | `:398-400` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:777-826` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` (`if (!pending_pool_reshard_recvs_.insert(plan.uuid()).second)`) | `:795-798` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` (`absl::Cleanup clear_pending = [this, uuid = plan.uuid()]() {`) | `:808-811` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`std::erase_if(retired_uuids,` .. `pending_pool_reshard_recvs_.contains(uuid);`) | `:1111-1114` |
| `mgr.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (uint64_t uuid : retired_uuids)` .. `base_->ReleasePoolStagingLeases(uuid);`) | `:1126-1130` |
| `reshard_recv.cc` | `ReshardReceiveSession` (`~ReshardReceiveSession() override { ReleaseStaging(); }`) | `:89` |
| `reshard_recv.cc` | `CreateReshardReceiveSession` | `:541-575` |
| `reshard_recv.cc` | `CreateReshardReceiveSession` (`auto session =` .. `return session;`) | `:567-574` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:604-643` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:645-650` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:652-655` |
| `pool_staging_test.cc` | `PoolStagingTest.BoundedArenaLeasesAndAddressing` | `:278-378` |
| `pool_staging_test.cc` | `PoolStagingTest.FallsBackToFullMirrorWithoutHintsOrLeases` | `:382-405` |
| `pool_staging_test.cc` | `PoolStagingTest.SmallPoolIdentitySharedStorageAndStrideMismatch` | `:410-475` |
| `pool_staging_test.cc` | `PoolStagingTest.SenderRaddrMatchesReceiverPointer` | `:480-594` |
| `pool_staging_test.cc` | `PoolStagingTest.SenderRaddrUsesReceiverLeaseSlot` | `:597-666` |
| `pool_staging_test.cc` | `PoolStagingTest.PoolHostBaseAddrs` | `:671-705` |
| `pool_staging_test.cc` | `PoolStagingTest.FullHostMirrorKeepsHeterogeneousStorageSizes` | `:1006-1046` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.DuplicatePoolRecvRegistrationKeepsOriginalStagingLease` | `:720-770` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.ConcurrentDuplicatePoolRecvRegistrationKeepsWinnerStagingLease` | `:772-828` |
| `pool_reshard_test.cc` | `PoolStagingRecvRegistrationTest.StagingAcquireFailureReleasesUuidClaim` | `:830-860` |
<!-- source-map:end -->
-/

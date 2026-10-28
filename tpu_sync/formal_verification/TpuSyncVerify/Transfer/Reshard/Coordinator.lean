import TpuSyncVerify.Transfer.Reshard.Coordinator.Model
import TpuSyncVerify.Transfer.Reshard.Coordinator.Lemmas
import TpuSyncVerify.Transfer.Reshard.Coordinator.Invariant
import TpuSyncVerify.Transfer.Reshard.Coordinator.Hardened
import TpuSyncVerify.Transfer.Reshard.Coordinator.Traces

/-!
# Two-stage reshard coordinator (`ReshardCoordinator::ExecutePoolReshard`)

Formal model of two-stage `ReshardCoordinator::ExecutePoolReshard`
orchestration (`tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`)
composed with `RequestBlockRegistry` (`TpuSyncVerify.Transfer.Reshard.Registry`).

`coord.h`/`coord.cc` are `tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`;
`mgr.cc` is `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`;
`reshard_recv.cc` is `tpu_sync/kv_cache/reshard_receive_session.cc`;
`reg.cc` is `tpu_sync/kv_cache/reshard/request_block_registry.cc`;
`planner.cc` is `tpu_sync/kv_cache/reshard/pool_reshard_planner.cc`;
`kv_cache_listener.cc` is `tpu_sync/kv_cache/kv_cache_listener.cc`;
`reshard_service.cc` is `tpu_sync/kv_cache/reshard/reshard_service.cc`; and
`reshard_service_test.cc` is
`tpu_sync/kv_cache/reshard/reshard_service_test.cc`. Exact line
ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

  C++ source under analysis:
  * `PoolReshardArgs` and `ReshardCoordinator` (`coord.h`)
  * `ReshardCoordinator::ExecutePoolReshard` (`coord.cc`)
  * `KVCacheListener::HandleControlRequest` (`tpu_sync/kv_cache/kv_cache_listener.cc`, `PoolReshardRegisterRecv` control-pipe handler)
  * `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` (`mgr.cc`: `ReshardReceiveSession::Create` + `RegisterActivePlanDirect`)
  * `ReshardReceiveSession::Create`, `ReshardReceiveSession::AcquireStagingLeases`, and `ReshardReceiveSession::ReleaseStaging` / `ReshardReceiveSession::ReleaseStagingLocked` (`reshard_recv.cc`)

  What this file proves:
  1. **Universal Protocol Guarantees (`CoordInv`, `reachable_coord_inv`)**:
     Across all reachable states (under both shipping and hardened cleanup
     policies):
     * `Registry.Inv s.reg` holds at every step.
     * `NoPushBeforeAllArmed` (`ReshardCoordinator::ExecutePoolReshard`, `for (std::thread& t : armers) t.join();` before `sender_dispatch_ns`): No sender
       is dispatched (`s.sendResults = []`) and no receiver receives any push
       while `s.phase = .idle ∨ s.phase = .arming`.
     * `AllArmedInDispatching` (`ReshardCoordinator::ExecutePoolReshard`, `for (std::thread& t : armers) t.join();` and `for (const absl::Status& status : arm_status)`): Entering
       `.dispatching` requires `s.armFailed = false` and all `dstUnits`
       reported in `s.armResults`.
     * `ClaimedWhileArming` (`ReshardCoordinator::ExecutePoolReshard`, `BuildPoolReshardPlan(plan_request, registry_, claim_owner)` through the `arm_status` check): Throughout
       `.arming`, the source snapshot remains claimed in `RequestBlockRegistry`
       (`s.reg.claimedUnits = some s.srcUnits` and `s.reg.claimOwners = [s.claimOwner]`),
       preventing `CancelIfUnclaimed` and conflicting `Register` calls.
     * `LeaseMatchesSession`: Every destination unit holds a `PoolStagingLease`
       if and only if its `ReshardReceiveSession` is currently armed.
  2. **Failure-Cleanup Counterexample Traces under `.shipping`**:
     * `trace_partial_arm_orphans_armed_receiver`
       (`ReshardCoordinator::ExecutePoolReshard`, `for (const absl::Status& status : arm_status)`): When `dst_unit 10` succeeds in arming
       and `dst_unit 11` fails, `ExecutePoolReshard` calls `AbandonClaim` and
       aborts without disarming `dst_unit 10`, leaving `dst_unit 10`'s
       `ReshardReceiveSession` and `PoolStagingLease` orphaned until
       `timeout_s_`.
     * `trace_dispatch_failure_orphans_receivers_and_claim`
       (`ReshardCoordinator::ExecutePoolReshard`, `for (const absl::Status& status : sender_status)`): When all `dst_units` arm and
       `src_unit 0` succeeds but `src_unit 1` fails during Stage 2 dispatch,
       `ExecutePoolReshard` returns `sender_status` without disarming
       `dst_units` AND without calling `registry_->AbandonClaim(...)`. Both
       `dst_units` remain orphaned with active sessions/leases AND
       `claimed_[(req_id, uuid)]` remains pinned in `RequestBlockRegistry`
       until `ttl_s_`.
  3. **Hardened Cleanup Policy Proof (`reachable_disarmAndAbandon_no_orphans`)**:
     Under `CleanupPolicy.disarmAndAbandonOnFailure`, every reachable `.aborted`
     state has zero armed receiver sessions, zero held staging leases, and
     `s.reg.claimedUnits = none ∧ s.reg.claimOwners = []`.

  ## Lock scopes

  `ReshardCoordinator::StartPoolReshard` brackets `ExecutePoolReshard` with two
  `ReshardCoordinator::status_mu_` scopes (`transfer_status_[args.req_id]`), and
  `ExecutePoolReshard` reads `directory_->LocalMetadata` under `*directory_->mu_`
  (shared with `RequestBlockRegistry::mu_`) on entry. Stage-result vectors
  (`arm_replies` / `arm_status`, `sender_status`) are synchronized across worker
  threads by the two `std::thread::join` barriers, while shared registry and
  receiver state are guarded by `RequestBlockRegistry::mu_` (`Registry.lean`) and
  each destination's `KVCacheManagerWithTransfer::mu_`:

  | Event | Scopes | Between them |
  |---|---|---|
  | `registerProducer srcUnit` | 1 `RequestBlockRegistry::mu_` scope (`RequestBlockRegistry::Register`) | — |
  | `startPlan planOk` | 1 `RequestBlockRegistry::mu_` scope in `BuildPoolReshardPlan` (`LookupAndClaim`), plus a second scope (`AbandonClaim`) if `planOk = false` after claiming | — |
  | `armDstResult dstUnit ok` | 1 `armers` worker thread calling `PoolReshardRegisterRecv` on `dstUnit` (2 `KVCacheManagerWithTransfer::mu_` scopes, `Pipeline.lean`) and writing its own `arm_replies[i]` / `arm_status[i]` slot | worker threads write disjoint slots of `arm_replies` / `arm_status`, joined at `finishArmBarrier` |
  | `finishArmBarrier` | `for (std::thread& t : armers) t.join();`, followed on `!status.ok()` by 1 `RequestBlockRegistry::mu_` scope (`registry_->AbandonClaim`) | `join()` orders all `arm_status[i]` writes before the loop that inspects them |
  | `sendSrcResult srcUnit ok` | 1 `senders` worker thread (or `SubmitTransferProgram` future) calling `PoolReshardPush` on `srcUnit` and writing `sender_status[i]` on failure; on `ok = true`, folds the sender RPC, receiver settlement (`Receive.lean`), and the producer's `registry_->Complete` vote (1 `RequestBlockRegistry::mu_` scope) | folding `Complete` into `ok = true` is sound for `CoordInv` because whenever any `srcUnit` fails (`sendFailed = true`), that unit never calls `Complete`, keeping `allVoted = false` |
  | `finishSendBarrier` | `for (std::thread& t : senders) t.join();` (or the final `futures[i].Await()`) before `for (const absl::Status& status : sender_status)` | `join()` / `Await()` orders all `sender_status[i]` writes before the status check |

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `WorkerDelivery` | `:57-62` |
| `coord.h` | `PoolReshardArgs` | `:65-84` |
| `coord.h` | `ReshardCoordinator` | `:91-129` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:318-367` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:369-592` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`static std::atomic<uintptr_t> claim_counter` .. `PoolReshardPlan& plan = *plan_or;`) | `:417-425` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`std::vector<absl::Status> arm_status` .. `for (std::thread& t : armers) t.join();`) | `:434-457` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`for (const absl::Status& status : arm_status)`) | `:458-463` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`std::vector<absl::Status> sender_status` .. `for (std::thread& t : senders) t.join();`) | `:481-539` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`for (const absl::Status& status : sender_status)`) | `:542-544` |
| `kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` (`} else if (is_pool_reshard) {` .. `(*resp->mutable_receiver_pool_addrs())[pool_idx] =`) | `:138-169` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:664-701` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:703-739` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` | `:351-390` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:420-459` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:461-465` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:467-470` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolH2dResultLocked` | `:585-598` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:668-706` |
| `planner.cc` | `BuildPoolReshardPlan` (`auto registrations_or =` .. `if (!registrations_or.ok()) return registrations_or.status();`) | `:1024-1026` |
| `reg.cc` | `RequestBlockRegistry::Register` | `:118-300` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-491` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:493-509` |
| `reshard_service.cc` | `ReshardService::HandleControllerCommand` (`case tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS:`) | `:291-303` |
| `reshard_service_test.cc` | `ReshardStackTest.FullPoolReshardFlowEmitsArmThenDispatch` | `:677-751` |
| `reshard_service_test.cc` | `ReshardStackTest.MissingRegistrationKeepsContractString` | `:903-913` |
| `reshard_service_test.cc` | `ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders` | `:915-931` |
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsArmEachThenDispatchSendersOnce` | `:1065-1135` |
<!-- source-map:end -->
-/

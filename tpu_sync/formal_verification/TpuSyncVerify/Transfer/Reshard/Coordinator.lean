import TpuSyncVerify.Transfer.Reshard.Coordinator.Model
import TpuSyncVerify.Transfer.Reshard.Coordinator.Lemmas
import TpuSyncVerify.Transfer.Reshard.Coordinator.Invariant
import TpuSyncVerify.Transfer.Reshard.Coordinator.Hardened
import TpuSyncVerify.Transfer.Reshard.Coordinator.Traces

/-!
# Two-stage reshard coordinator (`ReshardCoordinator::RunSubPlan`)

Formal model of one sub-transfer of `ReshardCoordinator::ExecutePoolReshard`
(`tpu_sync/kv_cache/reshard/reshard_coordinator.{h,cc}`) — the plan, arm and
dispatch stages of `ReshardCoordinator::RunSubPlan` — composed with
`RequestBlockRegistry` (`TpuSyncVerify.Transfer.Reshard.Registry`).

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
  * `ReshardCoordinator::ExecutePoolReshard`, `ReshardCoordinator::RunSubPlan`
    and `ReshardCoordinator::StartChain` (`coord.cc`)
  * `KVCacheListener::HandleControlRequest` (`tpu_sync/kv_cache/kv_cache_listener.cc`, `PoolReshardRegisterRecv` control-pipe handler)
  * `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` (`mgr.cc`: `ReshardReceiveSession::Create` + `RegisterActivePlanDirect`)
  * `ReshardReceiveSession::Create`, `ReshardReceiveSession::AcquireStagingLeases`, and `ReshardReceiveSession::ReleaseStaging` / `ReshardReceiveSession::ReleaseStagingLocked` (`reshard_recv.cc`)

  ## Scope: one sub-plan of a request

  `ReshardCoordinator::ExecutePoolReshard` fetches the source and destination
  metadata once, picks the request's `uuid`, plans the request's sub-transfer
  chain (`ReshardCoordinator::PlanSubTransfers`: one sub-plan per
  `dst_window_blocks` window of the destination block lists, `chain.count == 1`
  when no tag is windowed), draws one `claim_owner` for the whole chain
  (`claim_counter.fetch_add(1)`, so the registry sees one claim per request),
  stores all of it in a `ReshardCoordinator::ChainContext`, and runs sub-plan 0
  inline (`RunSubPlan(ctx, 0)`). When `ctx->chain.count > 1`,
  `ReshardCoordinator::StartChain` runs sub-plans `1..count-1` one after another
  on a coordinator thread (`chains_`), which records the request's final
  `transfer_status_` (`kCompleted` once every sub-plan has been dispatched,
  `kFailed` at the first failure or on shutdown); `StartPoolReshard` records
  that status itself only `if (!chain_started)`.

  Each `RunSubPlan(ctx, index)` call is one plan → arm → dispatch sequence:
  `BuildPoolReshardPlan(plan_request, registry_, ctx->claim_owner)` claims the
  source snapshot in `RequestBlockRegistry` (`LookupAndClaim`); the `armers`
  threads arm every destination (`PoolReshardRegisterRecv`) and are joined; the
  senders are dispatched (`PoolReshardPush`) and joined. This module models one
  such call with `index == 0`, which is the whole of a single-plan request's
  coordination. Later sub-plans of a chain differ in two places the model does
  not cover: a planning or arm failure leaves the registry claim to the
  dispatched senders' terminals (`if (index == 0) registry_->AbandonClaim(args.req_id, uuid, ctx->claim_owner);`),
  and an arm refused because the destination's staging arena is still held by
  earlier sub-plans (`ArmRefusedForBusyStaging`) is re-tried up to
  `kChainArmAttempts` times while `!stopping_.load()`. The sender-dispatch
  failure path is the same for every index: `for (const absl::Status& status : sender_status)`
  returns the first error without disarming receivers and without
  `AbandonClaim`. The per-request terminal bookkeeping a chained request gets
  on each worker (`KVCacheManagerWithTransfer::NoteSubPlanRegisteredLocked`,
  `RecordPoolReshardTerminalLocked`, `SweepSubPlanChainsLocked`) is outside this
  model.

  What this file proves:
  1. **Universal Protocol Guarantees (`CoordInv`, `reachable_coord_inv`)**:
     Across all reachable states (under both shipping and hardened cleanup
     policies):
     * `Registry.Inv s.reg` holds at every step.
     * `NoPushBeforeAllArmed` (`ReshardCoordinator::RunSubPlan`, `for (std::thread& t : armers) t.join();` before `sender_dispatch_ns`): No sender
       is dispatched (`s.sendResults = []`) and no receiver receives any push
       while `s.phase = .idle ∨ s.phase = .arming`.
     * `AllArmedInDispatching` (`ReshardCoordinator::RunSubPlan`, `for (std::thread& t : armers) t.join();` and the `if (armed[i]) continue;` scan that ends in `return arm_status[i];`): Entering
       `.dispatching` requires `s.armFailed = false` and all `dstUnits`
       reported in `s.armResults`.
     * `ClaimedWhileArming` (`ReshardCoordinator::RunSubPlan`, `BuildPoolReshardPlan(plan_request, registry_, ctx->claim_owner)` through the `armed[i]` scan): Throughout
       `.arming`, the source snapshot remains claimed in `RequestBlockRegistry`
       (`s.reg.claimedUnits = some s.srcUnits` and `s.reg.claimOwners = [s.claimOwner]`),
       preventing `CancelIfUnclaimed` and conflicting `Register` calls.
     * `LeaseMatchesSession`: Every destination unit holds a `PoolStagingLease`
       if and only if its `ReshardReceiveSession` is currently armed.
  2. **Failure-Cleanup Counterexample Traces under `.shipping`**:
     * `trace_partial_arm_orphans_armed_receiver`
       (`ReshardCoordinator::RunSubPlan`, `return arm_status[i];`): When `dst_unit 10` succeeds in arming
       and `dst_unit 11` fails, `RunSubPlan` calls `AbandonClaim` (`if (index == 0)`) and
       aborts without disarming `dst_unit 10`, leaving `dst_unit 10`'s
       `ReshardReceiveSession` and `PoolStagingLease` orphaned until
       `timeout_s_`.
     * `trace_dispatch_failure_orphans_receivers_and_claim`
       (`ReshardCoordinator::RunSubPlan`, `for (const absl::Status& status : sender_status)`): When all `dst_units` arm and
       `src_unit 0` succeeds but `src_unit 1` fails during Stage 2 dispatch,
       `RunSubPlan` returns `sender_status` without disarming
       `dst_units` AND without calling `registry_->AbandonClaim(...)`. Both
       `dst_units` remain orphaned with active sessions/leases AND
       `claimed_[(req_id, uuid)]` remains pinned in `RequestBlockRegistry`
       until `ttl_s_`.
  3. **Hardened Cleanup Policy Proof (`reachable_disarmAndAbandon_no_orphans`)**:
     Under `CleanupPolicy.disarmAndAbandonOnFailure`, every reachable `.aborted`
     state has zero armed receiver sessions, zero held staging leases, and
     `s.reg.claimedUnits = none ∧ s.reg.claimOwners = []`.

  ## Lock scopes

  `ReshardCoordinator::StartPoolReshard` writes `transfer_status_[args.req_id]`
  under `ReshardCoordinator::status_mu_` before `ExecutePoolReshard` and, for a
  request that did not continue as a chain (`if (!chain_started)`), again after
  it; a chained request's final status is written by `StartChain`'s thread under
  the same `status_mu_`. `ExecutePoolReshard` reads `directory_->LocalMetadata`
  under `*directory_->mu_` (shared with `RequestBlockRegistry::mu_`) on entry.
  `StartChain`, `ReapFinishedChainsLocked`, `WaitForSubTransferChains` and
  `~ReshardCoordinator` hold `ReshardCoordinator::chains_mu_` only to spawn,
  reap and join chain threads; `RunSubPlan` itself takes neither `status_mu_`
  nor `chains_mu_` (it reads the `stopping_` atomic only when deciding whether
  to re-arm). Stage-result vectors (`arm_replies` / `arm_status` / `armed`,
  `sender_status`) are synchronized across worker threads by the
  `std::thread::join` barriers, while shared registry and receiver state are
  guarded by `RequestBlockRegistry::mu_` (`Registry.lean`) and each
  destination's `KVCacheManagerWithTransfer::mu_`:

  | Event | Scopes | Between them |
  |---|---|---|
  | `registerProducer srcUnit` | 1 `RequestBlockRegistry::mu_` scope (`RequestBlockRegistry::Register`) | — |
  | `startPlan planOk` | 1 `RequestBlockRegistry::mu_` scope in `BuildPoolReshardPlan` (`LookupAndClaim`), plus a second scope (`AbandonClaim`, `if (index == 0)`) if `planOk = false` after claiming | — |
  | `armDstResult dstUnit ok` | 1 `armers` worker thread calling `PoolReshardRegisterRecv` on `dstUnit` (2 `KVCacheManagerWithTransfer::mu_` scopes, `Pipeline.lean`) and writing its own `arm_replies[i]` / `arm_status[i]` / `armed[i]` slot | worker threads write disjoint slots of `arm_replies` / `arm_status` / `armed`, joined at `finishArmBarrier` |
  | `finishArmBarrier` | `for (std::thread& t : armers) t.join();`, followed for an unarmed destination (`if (armed[i]) continue;`) by 1 `RequestBlockRegistry::mu_` scope (`registry_->AbandonClaim`, `if (index == 0)`) before `return arm_status[i];` | `join()` orders all `arm_status[i]` / `armed[i]` writes before the scan that inspects them; with `index == 0` there is a single arm attempt (`attempts = 1`), so the re-arm loop is not entered |
  | `sendSrcResult srcUnit ok` | 1 `senders` worker thread (or `SubmitTransferProgram` future) calling `PoolReshardPush` on `srcUnit` and writing `sender_status[i]` on failure; on `ok = true`, folds the sender RPC, receiver settlement (`Receive.lean`), and the producer's `registry_->Complete` vote (1 `RequestBlockRegistry::mu_` scope) | folding `Complete` into `ok = true` is sound for `CoordInv` because whenever any `srcUnit` fails (`sendFailed = true`), that unit never calls `Complete`, keeping `allVoted = false` |
  | `finishSendBarrier` | `for (std::thread& t : senders) t.join();` (or the final `futures[i].Await()`) before `for (const absl::Status& status : sender_status)` | `join()` / `Await()` orders all `sender_status[i]` writes before the status check |

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `coord.h` | `WorkerDelivery` | `:62-67` |
| `coord.h` | `PoolReshardArgs` | `:70-92` |
| `coord.h` | `ReshardCoordinator` | `:138-207` |
| `coord.cc` | `*` (`struct ReshardCoordinator::ChainContext {` .. `std::atomic<bool> finished{false};`) | `:310-321` |
| `coord.cc` | `ReshardCoordinator::StartPoolReshard` | `:474-526` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` | `:528-601` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`static std::atomic<uintptr_t> claim_counter` .. `reinterpret_cast<const void*>(claim_counter.fetch_add(1));`) | `:566-568` |
| `coord.cc` | `ReshardCoordinator::ExecutePoolReshard` (`absl::Status first = RunSubPlan(ctx, 0);` .. `return absl::OkStatus();`) | `:592-600` |
| `coord.cc` | `ReshardCoordinator::StartChain` | `:614-649` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` | `:678-932` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`auto plan_or = BuildPoolReshardPlan(plan_request, registry_, ctx->claim_owner);` .. `PoolReshardPlan& plan = *plan_or;`) | `:723-728` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`std::vector<absl::Status> arm_status` .. `for (std::thread& t : armers) t.join();`) | `:743-776` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`bool retry = false;` .. `return arm_status[i];`) | `:777-788` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`std::vector<absl::Status> sender_status` .. `for (std::thread& t : senders) t.join();`) | `:813-873` |
| `coord.cc` | `ReshardCoordinator::RunSubPlan` (`for (const absl::Status& status : sender_status)`) | `:876-878` |
| `kv_cache_listener.cc` | `KVCacheListener::HandleControlRequest` (`} else if (is_pool_reshard) {` .. `(*resp->mutable_receiver_pool_addrs())[pool_idx] =`) | `:138-169` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardPush` | `:833-871` |
| `mgr.cc` | `KVCacheManagerWithTransfer::PoolReshardRegisterRecv` | `:873-910` |
| `reshard_recv.cc` | `ReshardReceiveSession::Create` | `:351-390` |
| `reshard_recv.cc` | `ReshardReceiveSession::AcquireStagingLeases` | `:422-461` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStagingLocked` | `:463-467` |
| `reshard_recv.cc` | `ReshardReceiveSession::ReleaseStaging` | `:469-472` |
| `reshard_recv.cc` | `ReshardReceiveSession::RecordPoolH2dResultLocked` | `:587-600` |
| `reshard_recv.cc` | `ReshardReceiveSession::FinishPoolH2d` | `:670-708` |
| `planner.cc` | `BuildPoolReshardPlan` (`auto registrations_or =` .. `if (!registrations_or.ok()) return registrations_or.status();`) | `:1119-1121` |
| `reg.cc` | `RequestBlockRegistry::Register` | `:118-300` |
| `reg.cc` | `RequestBlockRegistry::Complete` | `:317-371` |
| `reg.cc` | `RequestBlockRegistry::LookupAndClaim` | `:439-491` |
| `reg.cc` | `RequestBlockRegistry::AbandonClaim` | `:493-509` |
| `reshard_service.cc` | `ReshardService::HandleControllerCommand` (`case tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS:`) | `:293-305` |
| `reshard_service_test.cc` | `ReshardStackTest.FullPoolReshardFlowEmitsArmThenDispatch` | `:760-834` |
| `reshard_service_test.cc` | `ReshardStackTest.MissingRegistrationKeepsContractString` | `:1369-1379` |
| `reshard_service_test.cc` | `ReshardStackTest.ArmFailureAbandonsClaimAndSkipsSenders` | `:1381-1397` |
| `reshard_service_test.cc` | `ReshardStackTest.TwoDestinationsArmEachThenDispatchSendersOnce` | `:1531-1601` |
<!-- source-map:end -->
-/

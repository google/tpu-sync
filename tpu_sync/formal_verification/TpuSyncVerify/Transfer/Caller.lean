import TpuSyncVerify.Transfer.Caller.Model
import TpuSyncVerify.Transfer.Caller.Invariant
import TpuSyncVerify.Transfer.Caller.Traces
import TpuSyncVerify.Transfer.Caller.MultiWorker
import TpuSyncVerify.Transfer.Caller.NumaBarrier

/-!
# The engine above `poll_stats()`: NUMA sub-managers, connector, aggregator, and scheduler

Both transfer pipelines (`PrefillDecode/Pipeline.lean` and
`Reshard/Pipeline.lean`, together with their `MultiRequest.lean` lifts) end at
the `poll_stats()` boundary (`KVCacheManagerWithTransfer::CompleteReadWithDetails`):
once tpu-sync publishes a request's send or receive outcome (`send.published`,
`recv.published : Option Bool` — `some true` for `done_sending` /
`done_recving`, `some false` for `failed_recving`) no copy touches the
request's HBM any more (`PrefillHbmSafe` / `DecodeHbmSafe` and `SrcHbmSafe` /
`DstHbmSafe`), and the pipeline models *assume* (**A4**) that the engine frees
that HBM only after publication. This module models the other direction of the
same contract — what the caller stack must do *with* a publication, across a
single worker (`Model.lean`, `Invariant.lean`, `Traces.lean`), across multiple
workers aggregated by `KVOutputAggregator` (`MultiWorker.lean`, covering both
its eager block-level `invalid_block_ids` channel and its latched request-level
`_failed_recving_pending` channel), and across intra-worker NUMA sub-managers
inside `NumaAwareKVCacheManager::CompleteReadRaw` (`NumaBarrier.lean`, proving
the settled-count barrier `NumaPolicy.settledCounts` of `process_completion`
safe across `req_id` reuse and refuting the eager-failure-set mutant
`NumaPolicy.eagerFailSet`) — for the two production vLLM connectors that drive
`tpu-sync`:

1. **Prefill-to-decode (`tpu-inference`):** vLLM's scheduler driving `tpu-sync`
   through `tpu_raiden_connector.py` (`TPUConnectorWorker.get_transfer_results`,
   instantiating `connectorTwoChannel` over `AggChannel.latchedReq`: producer
   merges `failed_recving` into `finished_sending`, consumer reports failed
   receives in both `finished_recving` and `failed_recving`).
2. **Pool reshard (`vllm-torchtpu` Stage-3):** vLLM's scheduler driving
   `tpu-sync` through `tpu_connector.py` (`TPURaidenConnectorWorker`,
   instantiating `connectorTwoChannel` over `AggChannel.eagerBlock`: accepted
   steady state forwards `done_sending | sender_failures | cancelled_sends` and
   `done_recving | failed_recving` while staging failed receive blocks into
   `_failed_block_ids` for `get_block_ids_with_load_errors`, and its
   `"fail"`-policy early uncertain-load path `_stage3_fail_uncertain_loads_early`
   stages `invalid_block_ids` on the RPC error pass while deferring
   `done_recving` — and the block free — until the native receiver settles:
   `earlyLoadError_fail_eq_finish`, `earlyLoadError_fail_holds_blocks`,
   `mw_earlyLoadError_fail_eq_finish`, `mw_earlyLoadError_fail_holds_blocks`).

Alongside `connectorTwoChannel`, three counterfactual connector mutants omit one
piece of `Contract` (`connectorDropFailed`, `connectorNaive`,
`connectorLoadErrorOnly`), witnessing why dropping `failed_recving` leaks blocks
on both nodes (`trace_dropFailed_*`, `trace_mw_dropFailed_partial_failure_leaks`),
why reporting `finished_recving` without the load-failure channel decodes
unwritten KV (`trace_naive_decodes`), and why reporting on the load-failure
channel without `finished_recving` still leaks D's blocks
(`trace_loadErrorOnly_*`, `trace_mw_latched_loadErrorOnly_never_emits`).

Citations: `conn.py` = `tpu_inference/distributed/tpu_raiden_connector.py`;
`tpu_connector.py` = `src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py`;
`sched.py` = `vllm/v1/core/sched/scheduler.py`,
`base.py` = `vllm/distributed/kv_transfer/kv_connector/v1/base.py`,
`utils.py` = `vllm/distributed/kv_transfer/kv_connector/utils.py`,
`mixin.py` = `vllm/v1/worker/kv_connector_model_runner_mixin.py`,
`kvcfg.py` = `vllm/config/kv_transfer.py`; `mgr.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` (pinned commits in
`README.md`).

## Source map

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|------|-----------------|-------|
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1076-1139` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`absl::MutexLock lock(mu_);`) | `:1082` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = sessions_.begin(); it != sessions_.end();)` .. `failed_recving_.clear();`) | `:1084-1122` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` | `:471-530` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` (`(!recv->GetStatus().ok() ? failed_recving_ : done_recving_)`) | `:482-483` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::RetireSettledSessionLocked` (`(failed ? failed_recving_ : done_sending_).insert(req_id);`) | `:520` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.h` | `KVCacheManagerWithTransfer` (`struct SessionEntry {` .. `SessionRole role;`) | `:304-307` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:186-252` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:254-303` |
| `tpu_sync/frameworks/jax/kv_cache_manager.h` | `NumaAwareKVCacheManager` (`std::map<std::string, int> done_sending_counts_;` .. `std::map<std::string, int> req_expected_counts_;`) | `:295-298` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::NotifyForRead` | `:532-540` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::StartRead(remote_descriptors)` | `:580-624` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::StartRead(remote_endpoint)` | `:626-641` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::CompleteReadRaw` | `:644-706` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::CompleteReadRaw` (`auto process_completion = [&](const std::string& req) {` .. `req_expected_counts_.erase(req);`) | `:650-685` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.VotingConsensusPromotesDoneRecvingOnlyWhenAllAgree` | `:194-221` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.FailureWaitsForSiblingSubManagersBeforeEmittingFailedRecving` | `:223-263` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.FailureWhenAllSubManagersFailEmitsFailedRecvingOnce` | `:265-298` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.RecvFirstThenFailureWaitsForSubManagersAndEmitsFailure` | `:300-327` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.SendFailureWaitsForSiblingSubManagersBeforeEmittingFailedRecving` | `:329-357` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.SingleSubManagerFailureEmitsImmediatelyAndCleansUp` | `:359-384` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.VotingConsensusPromotesDoneSendingOnlyWhenAllAgree` | `:386-406` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.StartReadUnifiedMultiEndpointMultiSubManager` | `:600-660` |
| `tpu_sync/frameworks/jax/kv_cache_manager_wrapper_test.cc` | `KVCacheManagerWrapperTest.StartReadInputVectorValidation` | `:662-679` |
| `tpu_sync/frameworks/jax/kv_cache_manager.cc` | `NumaAwareKVCacheManager::InitSubManagers` | `:305-471` |
<!-- source-map:end -->

## Lock scopes

| Event | Scopes | Between them |
|---|---|---|
| `pOutcome ok`, `dOutcome ok`, `pWorker w ok`, `dWorker w ok`, `pSub sub ok`, `dSub sub ok` | 1 `KVCacheManagerWithTransfer::mu_` scope per polled sub-manager in `CompleteReadWithDetails` (`absl::MutexLock lock(mu_);`); `NumaAwareKVCacheManager::CompleteReadRaw` aggregates the sub-managers' lists with no lock of its own, serialized with `StartRead` and `NotifyForRead` by the GIL (the `complete_read`, `start_read`, and `notify_for_read` bindings do not release it); then, on the single-threaded Python scheduler / model-runner loop, `get_transfer_results` (or `get_finished` + `get_block_ids_with_load_errors`), `KVOutputAggregator.aggregate`, and `update_from_output` | nothing engine-visible happens between `CompleteReadWithDetails` draining `done_sending_`, `done_recving_`, and `failed_recving_` and the scheduler step that consumes them; by `applySend_staggered_eq_batch`, `applyRecv_staggered_eq_batch`, and `applyRecv_eager_eq_latched`, two workers reporting in one `KVOutputAggregator.aggregate` pass produce the same state as two back-to-back `pWorker` / `dWorker` steps on either `AggChannel`; one `CompleteReadRaw` batch is split into back-to-back `pSub` / `dSub` steps (`NumaBarrier.lean` **N4**) |
| `promote`, `finish`, `recycle` | 0 (single-threaded Python scheduler `schedule()` and `finish_requests()`) | serialized on the Python scheduler loop |

## What is modelled

One request, both nodes. On the prefill/producer node (P) the request has
finished its one output token and the connector has asked vLLM to keep its KV
blocks until the send is reported (`TPUConnectorScheduler.request_finished`
returns `delay_free_blocks = True` in `conn.py`); on the decode/consumer node
(D) the request has been allocated blocks and parked in `WAITING_FOR_REMOTE_KVS`
with the transfer issued (`TPUConnectorScheduler.update_state_after_alloc` and
`TPUConnectorWorker._start_load_kv` in `conn.py`;
`tpu_connector.py` `_apply_stage3_submit_outcome`). Nothing before that point
touches a transfer-held block, so the model starts there (`init`, `mwInit`).

Below the engine sits tpu-sync, abstracted to the publications of
`PrefillDecode/Pipeline.lean` and `Reshard/Pipeline.lean` (`sendPub` =
`send.published`, `recvPub` = `recv.published`):

* **Poll sweep (`KVCacheManagerWithTransfer::CompleteReadWithDetails`, `mgr.cc`):**
  one loop over `sessions_` (`for (auto it = sessions_.begin(); it != sessions_.end();)`)
  sweeps deadlines (`session->deadline() <= now` across every session kind) and
  passes each settled session to
  `KVCacheManagerWithTransfer::RetireSettledSessionLocked`, which erases it and
  inserts its `SessionEntry`'s `req_id` into `done_sending_`, `done_recving_`,
  or `failed_recving_` across three arms: `TransferReceiveSession` on D,
  pool-reshard sessions distinguished by `SessionEntry::role`
  (`kPoolReshardSend` on P, `kPoolReshardReceive` on D), and
  `TransferSendSession` on P. A receive with `tracks_completion()` false is
  erased unreported.
* **Pool-reshard chains (`RecordPoolReshardTerminalLocked` / `SweepSubPlanChainsLocked`):**
  a single-plan request (no `sub_plan_chains_` entry for `req_id`) writes
  `(status.ok() ? done : failed_recving_).insert(req_id)` directly; a
  multi-window sub-plan chain (`SubPlanChainProgress::sub_sessions`) reports one
  terminal per request when its last sub-plan settles, at its first failure, or
  when `SweepSubPlanChainsLocked` fails a stalled chain (see
  `Reshard/SubPlanChain.lean` for the chain model and its caller assumptions).
* **Other writers under `mu_`:** `RetireSettledSessionLocked` also runs inline
  when `StartRead`, `EmplaceRecvSessionLocked`, `RegisterActivePlan` (sender or
  receiver), `PoolReshardPush`, or `PoolReshardRegisterRecv` sees a settled
  incumbent; `NotifyForRead` inserts into `done_sending_` if the ack for its
  uuid already arrived; `StartRead` inserts into `failed_recving_` on duplicate
  uuid, session-creation failure, or registration failure.
* **Model abstraction:** all of these collapse to one nondeterministic
  publication per side, set once by `pOutcome ok` / `dOutcome ok` (or per worker
  by `pWorker w ok` / `dWorker w ok`). `ok = false` covers failed pushes/pulls
  and unpulled sends expiring at `timeout_s` (both
  `(failed ? failed_recving_ : done_sending_)` in `RetireSettledSessionLocked`
  and timed-out sends land in `failed_recving_`; connectors pass
  `TPU_P2P_WAIT_PULL_TIMEOUT`, 180 s, via `TPUConnectorWorker.register_kv_caches`
  and `tpu_connector.py` `get_p2p_wait_pull_timeout()`). Which outcome occurs is
  left to the `Pipeline.lean` models.

Between tpu-sync and the scheduler sits the **connector**. Per request and per
step it sees at most one of the three `poll_stats()` lists and produces a
`Signal`: `finishedSending` / `finishedRecving` are membership in
`KVConnectorTransferResults.finished_sending` / `finished_recving` (or the pair
`KVConnectorBase_V1.get_finished` returns in `base.py`), and `loadError` is
vLLM's load-failure channel for the request — either "the request is in
`KVConnectorTransferResults.failed_recving`" (`KVConnectorOutput.failed_recving`,
copied by `KVConnectorModelRunnerMixin._get_kv_connector_output`) or "the
request's local blocks are in `KVConnectorBase_V1.get_block_ids_with_load_errors`"
(`KVConnectorOutput.invalid_block_ids` in `base.py`). The two channels
reach the single-worker scheduler with the same effect (step 1 below), and
differ in multi-worker aggregation (`AggChannel.latchedReq` vs.
`AggChannel.eagerBlock` in `MultiWorker.lean`). A `Connector` is the pair of
functions `onSend`, `onRecv : (ok : Bool) → Signal`:

| Connector | `onSend ok` | `onRecv ok` | Role |
|-----------|-------------|-------------|------|
| `connectorTwoChannel` | `finishedSending := true` | `finishedRecving := true, loadError := !ok` | **shipping** (`Contract`): `TPUConnectorWorker.get_transfer_results` in `tpu_raiden_connector.py` (producer merges `failed_recving` into `finished_sending`; consumer reports failed receives in `finished_recving` and `failed_recving`, over `AggChannel.latchedReq`) and Stage-3 `TPURaidenConnectorWorker` in `tpu_connector.py` (`get_finished` + `_failed_block_ids` → `get_block_ids_with_load_errors`, over `AggChannel.eagerBlock`) |
| `connectorDropFailed` | `finishedSending := ok` | `finishedRecving := ok` | **mutant**: `failed_recving` dropped on both nodes (a `get_finished`-only connector that overrides neither `get_block_ids_with_load_errors` nor `get_transfer_results`) |
| `connectorNaive` | `finishedSending := true` | `finishedRecving := true` | **mutant**: failed receive reported in `finished_recving` (`done_recving`) without the load-failure channel (also reached if a `get_finished()` wrapper returning `(finished_sending, finished_recving)` without `get_block_ids_with_load_errors` is called by a runner that does not use `get_transfer_results`) |
| `connectorLoadErrorOnly` | `finishedSending := true` | `finishedRecving := ok, loadError := !ok` | **mutant**: load-failure channel without the `finished_recving` completion |

The **scheduler** is `sched.py` restricted to what touches a transfer-held
request, in the order `Scheduler.update_from_output` runs it within one step:

1. Load-failure handlers (`Scheduler._handle_failed_recving` on
   `KVConnectorOutput.failed_recving` and `Scheduler._handle_invalid_blocks` on
   `KVConnectorOutput.invalid_block_ids` in `sched.py`): only a request still in
   `WAITING_FOR_REMOTE_KVS` is affected. Under
   `kv_load_failure_policy = "recompute"` (`recompute_kv_load_failures`) its
   `num_computed_tokens` is reset to 0 and it is added to
   `failed_recving_kv_req_ids`. Under `"fail"` — the default in
   `KVTransferConfig.kv_load_failure_policy` (`kvcfg.py`) — it is finished with
   `FINISHED_ERROR` through `Scheduler.finish_requests`, which for a
   `WAITING_FOR_REMOTE_KVS` request defers the block free until the id is in
   `finished_recving_kv_req_ids`. Since a tpu-sync failure invalidates the whole
   transfer (**B2**), `applyLoadError` is the scheduler effect of either
   handler.
2. `finished_recving` (`Scheduler._update_from_kv_xfer_finished`): a request
   still waiting is added to `finished_recving_kv_req_ids`; a request already
   finished (aborted or errored) has its blocks freed on the spot
   (`Scheduler._free_blocks`, which also deletes it from `scheduler.requests`).
3. `finished_sending` (`Scheduler._update_from_kv_xfer_finished`): blocks freed
   on the spot.

Across steps: the next `Scheduler.schedule` promotes a waiting request whose id
is in `finished_recving_kv_req_ids`
(`Scheduler._try_promote_blocked_waiting_request` →
`Scheduler._update_waiting_for_remote_kv`): if the id is also in
`failed_recving_kv_req_ids` and nothing valid was computed, the blocks are
freed and the request goes back to `WAITING` to prefill locally
(`recomputing`); otherwise the blocks are kept as the request's KV and it
proceeds to decode (`decoding`). And at any time the API layer may abort the
request (`Scheduler.finish_requests`): a waiting request's free is deferred as
in 1, a decoding one is freed; a P request is already finished and is skipped —
there is no API-layer action that reaches a P-side hold.

The P side keeps only `pHeld`; its connector signal is read for
`finishedSending` alone (a P scheduler has no request in
`WAITING_FOR_REMOTE_KVS`, and `TPUConnectorScheduler.request_finished` on D
returns `delay_free_blocks = False` in `conn.py`).

In a multi-host or multi-rank deployment
(`TPUConnectorScheduler.get_finished_count` in `conn.py`,
`TPURaidenConnector.get_finished_count` in `tpu_connector.py`),
`KVOutputAggregator` (`utils.py`) sits between the worker-rank connectors and
`update_from_output` (`MultiWorker.lean`): it counts
`finished_sending` / `finished_recving` reports per request across workers and
withholds completion until all workers have reported (`pSentCount = expectedP`,
`dRecvCount = expectedD`), while handling the load-failure channel according to
`AggChannel`:
* `AggChannel.eagerBlock`: unions and forwards `invalid_block_ids` immediately
  on whichever step any worker reports a failure;
* `AggChannel.latchedReq`: latches `_failed_recving_pending |=
  kv_output.failed_recving` (`dFailPending`) and emits `failed_recving =
  _failed_recving_pending & finished_recving` (clearing `_failed_recving_pending`)
  on the completing step (`dRecvCount = expectedD`).

Inside a single JAX worker, `NumaAwareKVCacheManager::CompleteReadRaw`
(`tpu_sync/frameworks/jax/kv_cache_manager.cc`) sits between the per-NUMA
`KVCacheManagerWithTransfer` sub-managers and
`TPUConnectorWorker.get_transfer_results` (`NumaBarrier.lean`).
`NumaAwareKVCacheManager::InitSubManagers` creates one sub-manager by default
and, under `ENABLE_MULTI_NUMA=1`/`true`, one per device NUMA node (or
`min(num_ext_nics, total_num_shards_)` when multiple shards share one node on a
multi-NIC host); the model and its theorems instantiate 1 and 2 sub-managers
(`twoSubs`). Two sub-manager
barrier policies are compared across sequential transfers that reuse the same
`req_id` (`recycle`):
* `NumaPolicy.settledCounts` (`process_completion` in `kv_cache_manager.cc`):
  counts `done_sending_counts_`, `done_recving_counts_`, and
  `failed_recving_counts_`, withholds emission until
  `s_count + r_count + f_count >= expected`, emits `failed_recving` if
  `f_count > 0` (else `done_recving` or `done_sending`), and erases all four
  per-request map entries
  (`KVCacheManagerWrapperTest.FailureWaitsForSiblingSubManagersBeforeEmittingFailedRecving`,
  its six sibling barrier tests and
  `KVCacheManagerWrapperTest.StartReadUnifiedMultiEndpointMultiSubManager`,
  replayed in `NumaBarrier.lean`).
* `NumaPolicy.eagerFailSet` (mutant): emits `done_sending` at
  `done_sending_counts_[req] == num_subs`, emits `done_recving` (unless
  suppressed by a per-request failure set) and erases `done_recving_counts_[req]`,
  `req_expected_counts_[req]`, and the set entry at
  `done_recving_counts_[req] >= expected`, and emits `failed_recving`
  immediately on the first failing sub-manager without waiting for sibling
  sub-managers or erasing any map entry.

## Properties

* `leakP` (`mwLeakP`, `numaLeakP`): the send is published (on all workers /
  sub-managers) and P still holds the blocks.
* `leakD` (`mwLeakD`, `numaLeakD`): the receive is published (on all workers /
  sub-managers) and D either keeps the request parked with its blocks and
  nothing to promote it, or has finished it with the free deferred. Both are
  *absorbing* for every connector, aggregator channel, and policy (`leakP_absorbing`,
  `leakD_absorbing`, `mwLeakP_absorbing`, `mwLeakD_absorbing`): nothing in the
  engine can free them afterwards, because the only events that could are the
  publications, and each happens once.
* `decodesUnwritten` (`mwDecodesUnwritten`, `numaDecodesUnwritten`): D's request
  proceeds to decode while the receive is not published as done (on every worker
  and sub-manager).
* `mwEarlyFree` (`numaEarlyFree`): P frees its HBM (`pHeld = false`) while a P
  worker or sub-manager is still unpublished, or D frees its HBM
  (`dHeld = false`) while a D worker or sub-manager is still unpublished —
  discharging the pipeline models' **A4** assumption across workers and NUMA
  sub-managers when one shard fails early while another shard's DMA is still in
  flight.

`Contract conn` is the obligation on a connector that the theorems need:
every P outcome yields `finishedSending`, every D outcome yields
`finishedRecving`, and a failed D outcome also yields `loadError` — the
`base.py` contracts on both channels
(`KVConnectorBase_V1.get_block_ids_with_load_errors` and
`KVConnectorTransferResults`). `contract_safe` (single
worker) and `mw_contract_safe` (2-worker `KVOutputAggregator` with
`expectedP = 2`, `expectedD = 2`, on both `AggChannel.eagerBlock` and
`AggChannel.latchedReq`) prove that any connector satisfying `Contract`, under
either failure policy and any mix of worker outcomes or arrival orders, never
frees HBM while a worker is still in flight, never leaks, and never decodes
unwritten KV; `connectorTwoChannel` satisfies it (`twoChannel_contract`,
`twoChannel_safe`, `mw_twoChannel_safe`). The three mutants fail `Contract` and
each has refutation traces: `trace_dropFailed_p`,
`trace_dropFailed_d_abort_after`, `trace_dropFailed_d_abort_before`,
`trace_dropFailed_d_recompute`, and `trace_mw_dropFailed_partial_failure_leaks`
(`connectorDropFailed`), `trace_naive_decodes` (`connectorNaive`),
`trace_loadErrorOnly_fail`, `trace_loadErrorOnly_recompute`, and
`trace_mw_latched_loadErrorOnly_never_emits` (`connectorLoadErrorOnly`), plus
`trace_mw_overcounted_expected_leaks` (why `get_finished_count()` must match the
participating worker count). The searches `search` and `mwSearch`, run over
every event instance (`allEvents`, `mwAllEvents`), close with `.safe` for the
`connectorTwoChannel` configurations; they are evaluated by `#guard` and
cross-check `contract_safe` / `mw_contract_safe`, which are the proofs.

On the Reshard path under `Policy.fail`, `vllm-torchtpu`'s
`_stage3_fail_uncertain_loads_early` (`tpu_connector.py`
`_apply_stage3_submit_outcome`) stages `_failed_block_ids` as soon as a
post-arm `coordinate_transfer` RPC fails ("uncertain"), while withholding
`done_recving` until the native receiver settles (`tpu_connector.py`
`get_finished` / `_poll_finished`). `earlyLoadError_fail_eq_finish` and
`earlyLoadError_fail_holds_blocks` (and their 2-worker counterparts
`mw_earlyLoadError_fail_eq_finish` and `mw_earlyLoadError_fail_holds_blocks`)
show that this early `applyLoadError .fail` produces the exact same state as
`finish` / `mwFinish` (marking the request `.finished` with `dHeld = true` so
destination HBM is not freed while any receiver rank is still armed), and the
subsequent terminal `dOutcome false` / `dWorker` publications on
`connectorTwoChannel` free `dHeld = false`.

On the intra-worker JAX wrapper (`NumaBarrier.lean`), the mutant
`NumaPolicy.eagerFailSet` frees consumer or producer HBM while a sibling
sub-manager's DMA is still in flight at `num_subs = 2`
(`trace_eagerFailSet_recv_early_free`, `trace_eagerFailSet_send_early_free`)
and, across `recycle` (`req_id` reuse), leaks blocks at `num_subs = 1`
(`trace_eagerFailSet_single_sub_reuse_leaks`) and desynchronizes
`done_recving_counts_` at `num_subs = 2` into premature `done_recving` on a
later transfer (`trace_eagerFailSet_partial_fail_desync_premature_doneRecving`).
Under `NumaPolicy.settledCounts` (`process_completion`), `numa_settledCounts_safe` and
`numa_settledCounts_maps_clean` prove safety (`NumaSafe`) and full map cleanup
across unbounded `recycle` rounds for both `num_subs = 1` and `num_subs = 2`,
both failure policies, and both `StartRead` overloads (`numaSearch`, over
every event instance, closes on the cyclic multi-round state space with
`.safe` as a cross-check), while
`numaProcessSettled_expected_numSubs_eq_none` shows that pre-setting
`req_expected_counts_[req] = num_subs` in `StartRead(remote_descriptors)` is
observationally redundant.

## Assumptions

* **B1 (every participating worker sub-manager eventually publishes once).**
  Each worker rank that was armed for the transfer eventually produces one
  terminal publication (`some true` or `some false`) after its local DMAs have
  drained, and `get_finished_count()` equals the number of participating workers
  (`expectedP = 2`, `expectedD = 2`). On both paths this is enforced for
  single-round, single-plan transfers (no `sub_plan_chains_` entry) on
  single-sub-manager workers (`ENABLE_MULTI_NUMA` unset or `0`) by the 180 s native session
  deadline (`TPU_P2P_WAIT_PULL_TIMEOUT` in
  `TPUConnectorWorker.register_kv_caches` and `tpu_connector.py`
  `get_p2p_wait_pull_timeout()`) together with `_sync_stage3_tp_submit_states`
  in `tpu_connector.py` (which broadcasts `"failed"`, `"uncertain"`, or
  `"accepted"` across TP ranks so no rank is left unarmed while another waits).
  `MultiWorker.lean` models the 2-worker `KVOutputAggregator` combining step
  directly on both `AggChannel.eagerBlock` and `AggChannel.latchedReq`;
  `NumaBarrier.lean` models the intra-worker `NumaAwareKVCacheManager` barrier
  below it, assuming one report per NUMA sub-manager per transfer (its **N1**,
  this assumption applied per sub-manager) and proving that
  `NumaPolicy.settledCounts` turns those reports into exactly one worker-level
  publication, after every sub-manager has settled, also with two
  sub-managers and under `req_id` reuse.
* **B2 (whole-transfer failure).** A failed pull invalidates every local
  block of the request, so recomputation starts from token 0 and the
  `num_computed_tokens ≠ 0` branch of `Scheduler._update_waiting_for_remote_kv`
  (cache the valid prefix) is not reached. tpu-sync
  reports a receive as one outcome (`(!recv->GetStatus().ok() ? failed_recving_ : done_recving_)`
  in `KVCacheManagerWithTransfer::RetireSettledSessionLocked`), so a connector has
  no finer information to report.
* **B3 (no preemption, prefix caching or chunking).** None of them changes
  which event frees a transfer-held block.
-/

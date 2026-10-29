import TpuSyncVerify.Transfer.Caller.Model
import TpuSyncVerify.Transfer.Caller.Invariant
import TpuSyncVerify.Transfer.Caller.Traces
import TpuSyncVerify.Transfer.Caller.MultiWorker

/-!
# The engine above `poll_stats()`: connector and scheduler block lifecycle

Both transfer pipelines (`PrefillDecode/Pipeline.lean` and
`Reshard/Pipeline.lean`, together with their `MultiRequest.lean` lifts) end at
the `poll_stats()` boundary (`KVCacheManagerWithTransfer::CompleteReadWithDetails`):
once tpu-sync publishes a request's send or receive outcome (`send.published`,
`recv.published : Option Bool` — `some true` for `done_sending` /
`done_recving`, `some false` for `failed_recving`) no copy touches the
request's HBM any more (`PrefillHbmSafe` / `DecodeHbmSafe` and `SrcHbmSafe` /
`DstHbmSafe`), and the pipeline models *assume* (**A4**) that the engine frees
that HBM only after publication. This module models the other direction of the
same contract — what the engine must do *with* a publication, both for a
single worker (`Model.lean`, `Invariant.lean`, `Traces.lean`) and across
multiple workers aggregated by `KVOutputAggregator` (`MultiWorker.lean`,
covering both its eager block-level `invalid_block_ids` channel and its latched
request-level `_failed_recving_pending` channel) — for the two production vLLM
connectors that drive `tpu-sync`:

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
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:1151-1292` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`absl::MutexLock lock(mu_);`) | `:1157` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = send_sessions_.begin();` .. `send_sessions_.erase(it++);`) | `:1159-1177` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_sends_.begin();` .. `active_pool_reshard_sends_.erase(it++);`) | `:1182-1198` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_recv_sessions_.begin();` .. `active_recv_sessions_.erase(it++);`) | `:1207-1234` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_recvs_.begin();` .. `active_pool_reshard_recvs_.erase(it++);`) | `:1239-1263` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked` | `:177-227` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::SweepSubPlanChainsLocked` | `:229-288` |
<!-- source-map:end -->

## Lock scopes

| Event | Scopes | Between them |
|---|---|---|
| `pOutcome ok`, `dOutcome ok`, `pWorker w ok`, `dWorker w ok` | 1 `KVCacheManagerWithTransfer::mu_` scope in `CompleteReadWithDetails` (`absl::MutexLock lock(mu_);`), followed on the single-threaded Python scheduler / model-runner loop by `get_transfer_results` (or `get_finished` + `get_block_ids_with_load_errors`), `KVOutputAggregator.aggregate`, and `update_from_output` | nothing engine-visible happens between `CompleteReadWithDetails` draining `done_sending_`, `done_recving_`, and `failed_recving_` and the scheduler step that consumes them; by `applySend_staggered_eq_batch`, `applyRecv_staggered_eq_batch`, and `applyRecv_eager_eq_latched`, two workers reporting in one `KVOutputAggregator.aggregate` pass produce the same state as two back-to-back `pWorker` / `dWorker` steps on either `AggChannel` |
| `promote`, `finish` | 0 (single-threaded Python scheduler `schedule()` and `finish_requests()`) | serialized on the Python scheduler loop |

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
`PrefillDecode/Pipeline.lean` and `Reshard/Pipeline.lean`: `sendPub` is
`send.published` and `recvPub` is `recv.published`. In
`KVCacheManagerWithTransfer::CompleteReadWithDetails` (`mgr.cc`), all four
session loops — `for (auto it = send_sessions_.begin();` ..
`send_sessions_.erase(it++);` and
`for (auto it = active_pool_reshard_sends_.begin();` ..
`active_pool_reshard_sends_.erase(it++);` on P, and
`for (auto it = active_recv_sessions_.begin();` ..
`active_recv_sessions_.erase(it++);` and
`for (auto it = active_pool_reshard_recvs_.begin();` ..
`active_pool_reshard_recvs_.erase(it++);` on D — write into the same three
lists (`done_sending_`, `done_recving_`, `failed_recving_`); the two pool-reshard
loops write through `KVCacheManagerWithTransfer::RecordPoolReshardTerminalLocked`,
which for a single-plan request (`sub_plan_count <= 1`) is the same
`(status.ok() ? done : failed_recving_).insert(req_id)` and for a request
coordinated as a chain of sub-plans reports one terminal per request (when its
last sub-plan settles, or at its first failed sub-plan — `SweepSubPlanChainsLocked`
fails a chain whose next sub-plan never arrives). Each publication is
set once, by `pOutcome ok` / `dOutcome ok` (or per worker by `pWorker w ok` /
`dWorker w ok`), with `ok = false` covering a failed push or pull *and* a send
nobody pulled before `timeout_s` (both `(failed ? failed_recving_ : done_sending_)`
and timed-out sends go to `failed_recving_`; the connectors pass
`TPU_P2P_WAIT_PULL_TIMEOUT`, 180 s, as `timeout_s` in
`TPUConnectorWorker.register_kv_caches` and `tpu_connector.py`
`get_p2p_wait_pull_timeout()`). Which value is published is
nondeterministic here; the `Pipeline.lean` modules say when each is possible.

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

## Properties

* `leakP` (`mwLeakP`): the send is published (on all workers) and P still
  holds the blocks.
* `leakD` (`mwLeakD`): the receive is published (on all workers) and D
  either keeps the request parked with its blocks and nothing to promote it, or
  has finished it with the free deferred. Both are *absorbing* for every
  connector, aggregator channel, and policy (`leakP_absorbing`,
  `leakD_absorbing`, `mwLeakP_absorbing`, `mwLeakD_absorbing`): nothing in the
  engine can free them afterwards, because the only events that could are the
  publications, and each happens once.
* `decodesUnwritten` (`mwDecodesUnwritten`): D's request proceeds to decode
  while the receive is not published as done (on every worker).
* `mwEarlyFree`: on a multi-worker node, P frees its HBM (`pHeld = false`)
  while a P worker is still unpublished, or D frees its HBM (`dHeld = false`)
  while a D worker is still unpublished — discharging the pipeline models' **A4**
  assumption across workers when one worker fails early while another worker's
  transfer is still in flight.

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
participating worker count). Bounded search (`search`, `mwSearch`) exhausts the
state space for each configuration.

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

## Assumptions

* **B1 (every participating worker sub-manager eventually publishes once).**
  Each worker rank that was armed for the transfer eventually produces one
  terminal publication (`some true` or `some false`) after its local DMAs have
  drained, and `get_finished_count()` equals the number of participating workers
  (`expectedP = 2`, `expectedD = 2`). On both paths this is enforced for
  single-plan transfers (`sub_plan_count <= 1`) on single-sub-manager workers
  (`ENABLE_MULTI_NUMA=0`) by the 180 s native session
  deadline (`TPU_P2P_WAIT_PULL_TIMEOUT` in
  `TPUConnectorWorker.register_kv_caches` and `tpu_connector.py`
  `get_p2p_wait_pull_timeout()`) together with `_sync_stage3_tp_submit_states`
  in `tpu_connector.py` (which broadcasts `"failed"`, `"uncertain"`, or
  `"accepted"` across TP ranks so no rank is left unarmed while another waits).
  `MultiWorker.lean` models the 2-worker `KVOutputAggregator` combining step
  directly on both `AggChannel.eagerBlock` and `AggChannel.latchedReq`.
* **B2 (whole-transfer failure).** A failed pull invalidates every local
  block of the request, so recomputation starts from token 0 and the
  `num_computed_tokens ≠ 0` branch of `Scheduler._update_waiting_for_remote_kv`
  (cache the valid prefix) is not reached. tpu-sync
  reports a receive as one outcome (`(!session->GetStatus().ok() ? failed_recving_ : done_recving_)`
  in `KVCacheManagerWithTransfer::CompleteReadWithDetails`), so a connector has
  no finer information to report.
* **B3 (no preemption, prefix caching or chunking).** None of them changes
  which event frees a transfer-held block.
-/

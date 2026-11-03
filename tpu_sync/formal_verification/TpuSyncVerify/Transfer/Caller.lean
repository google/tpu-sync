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
multiple workers aggregated by `KVOutputAggregator` (`MultiWorker.lean`) — for
the two production vLLM connectors that drive `tpu-sync`:

1. **Prefill-to-decode (`tpu-inference`):** vLLM's scheduler driving `tpu-sync`
   through `tpu_raiden_connector.py` (`connectorCurrent`). The
   `trace_current_connector_*` and `trace_mw_current_partial_failure_leaks`
   theorems are the traces where, with the connector as it is, a publication
   never becomes a free.
2. **Pool reshard (`vllm-torchtpu` Stage-3):** vLLM's scheduler driving
   `tpu-sync` through `tpu_connector.py`, whose accepted-transfer steady state
   forwards both `done_sending | sender_failures | cancelled_sends` and
   `done_recving | failed_recving` and stages failed receive blocks into
   `_failed_block_ids` (`connectorFixed`), and whose `"fail"`-policy early
   uncertain-load path (`_stage3_fail_uncertain_loads_early`) stages
   `invalid_block_ids` on the RPC error pass while deferring `done_recving`
   (and the block free) until the native receiver settles
   (`earlyLoadError_fail_eq_finish`, `earlyLoadError_fail_holds_blocks`,
   `mw_earlyLoadError_fail_eq_finish`, `mw_earlyLoadError_fail_holds_blocks`).

Citations: `conn.py` = `tpu_inference/distributed/tpu_raiden_connector.py` at
tpu-inference `7ed46a6`; `tpu_connector.py` =
`src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py` at vllm-torchtpu
`9725110`; `sched.py` = `vllm/v1/core/sched/scheduler.py`,
`base.py` = `vllm/distributed/kv_transfer/kv_connector/v1/base.py`,
`utils.py` = `vllm/distributed/kv_transfer/kv_connector/utils.py`,
`mixin.py` = `vllm/v1/worker/kv_connector_model_runner_mixin.py`,
`kvcfg.py` = `vllm/config/kv_transfer.py`, all at vLLM `6185d73` (the pair
`examples/*_disagg/setup.sh` pins); `mgr.cc` =
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`.

## Source map

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|------|-----------------|-------|
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` | `:980-1121` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`absl::MutexLock lock(mu_);`) | `:986` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = send_sessions_.begin();` .. `send_sessions_.erase(it++);`) | `:988-1006` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_sends_.begin();` .. `active_pool_reshard_sends_.erase(it++);`) | `:1011-1026` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_recv_sessions_.begin();` .. `active_recv_sessions_.erase(it++);`) | `:1035-1062` |
| `tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc` | `KVCacheManagerWithTransfer::CompleteReadWithDetails` (`for (auto it = active_pool_reshard_recvs_.begin();` .. `active_pool_reshard_recvs_.erase(it++);`) | `:1067-1093` |
<!-- source-map:end -->

## Lock scopes

| Event | Scopes | Between them |
|---|---|---|
| `pOutcome ok`, `dOutcome ok`, `pWorker w ok`, `dWorker w ok` | 1 `KVCacheManagerWithTransfer::mu_` scope in `CompleteReadWithDetails` (`absl::MutexLock lock(mu_);`), followed on the single-threaded Python scheduler / model-runner loop by `get_finished`, `get_block_ids_with_load_errors`, `KVOutputAggregator.aggregate`, and `update_from_output` | nothing engine-visible happens between `CompleteReadWithDetails` draining `done_sending_`, `done_recving_`, and `failed_recving_` and the scheduler step that consumes them; by `applySend_staggered_eq_batch` / `applyRecv_staggered_eq_batch`, two workers reporting in one `KVOutputAggregator.aggregate` pass produce the same state as two back-to-back `pWorker` / `dWorker` steps |
| `promote`, `finish` | 0 (single-threaded Python scheduler `schedule()` and `finish_requests()`) | serialized on the Python scheduler loop |

## What is modelled

One request, both nodes. On the prefill/producer node (P) the request has
finished its one output token and the connector has asked vLLM to keep its KV
blocks until the send is reported (`request_finished` returns
`delay_free_blocks = True`, `conn.py:438`); on the decode/consumer node (D) the
request has been allocated blocks and parked in `WAITING_FOR_REMOTE_KVS` with
the transfer issued (`conn.py:322-378`, `conn.py:606`;
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
lists (`done_sending_`, `done_recving_`, `failed_recving_`). Each publication is
set once, by `pOutcome ok` / `dOutcome ok` (or per worker by `pWorker w ok` /
`dWorker w ok`), with `ok = false` covering a failed push or pull *and* a send
nobody pulled before `timeout_s` (both `(failed ? failed_recving_ : done_sending_)`
and timed-out sends go to `failed_recving_`; the connectors pass
`TPU_P2P_WAIT_PULL_TIMEOUT`, 180 s, as `timeout_s`, `conn.py:529`,
`tpu_connector.py` `get_p2p_wait_pull_timeout()`). Which value is published is
nondeterministic here; the `Pipeline.lean` modules say when each is possible.

Between tpu-sync and the scheduler sits the **connector**, the object under
test. Per request and per step it sees at most one of the three
`poll_stats()` lists and produces a `Signal`: `finishedSending` /
`finishedRecving` are membership in the pair `get_finished()` returns
(`base.py:357-373`, read by the worker at `mixin.py:102-104`), `loadError` is
"the request's local blocks are in `get_block_ids_with_load_errors()`"
(`base.py:375-393`, `mixin.py:105`). A `Connector` is the pair of functions
`onSend`, `onRecv : (ok : Bool) → Signal`; the publication and the step that
forwards it are one event, which is sound because nothing engine-visible
happens between tpu-sync inserting into a list and the next `poll_stats()`
draining it. Four connectors are instantiated:

| Connector | `onSend ok` | `onRecv ok` | Is |
|-----------|-------------|-------------|----|
| `connectorCurrent` | `finishedSending := ok` | `finishedRecving := ok` | `conn.py:632-656`: the third list is logged and dropped (`conn.py:640-647`), `get_block_ids_with_load_errors` is not overridden |
| `connectorFixed` | `finishedSending := true` | `finishedRecving := true, loadError := !ok` | the proposed fix for `conn.py`, and the shipping Stage-3 Reshard connector in `tpu_connector.py` (`get_finished` / `_poll_finished`): forward every outcome, failed receives through vLLM's load-failure channel |
| `connectorNaive` | `finishedSending := true` | `finishedRecving := true` | the fix the connector's comment warns against: a failed receive reported as done |
| `connectorLoadErrorOnly` | `finishedSending := true` | `finishedRecving := ok, loadError := !ok` | the other half-fix: the failure channel without the completion |

The **scheduler** is `sched.py` restricted to what touches a transfer-held
request, in the order `update_from_output` runs it within one step:

1. `invalid_block_ids` (`sched.py:1516-1523` → `_handle_invalid_blocks`,
   `sched.py:2579-2648`): only a request still in `WAITING_FOR_REMOTE_KVS` is
   affected (`sched.py:2591-2594`). Under `kv_load_failure_policy = "recompute"`
   its `num_computed_tokens` is cut back to the first failed block
   (`sched.py:2553`; the connector would report every local block, so to 0) and
   it is added to `failed_recving_kv_req_ids` (`sched.py:2646`). Under `"fail"`
   — the default, `kvcfg.py:69` — it is finished with `FINISHED_ERROR` through
   `finish_requests` (`sched.py:1742-1744`), which for a
   `WAITING_FOR_REMOTE_KVS` request defers the block free until the id is in
   `finished_recving_kv_req_ids` (`sched.py:2061-2070`).
2. `finished_recving` (`sched.py:1758` → `sched.py:2462-2470`): a request still
   waiting is added to `finished_recving_kv_req_ids`; a request already finished
   (aborted or errored) has its blocks freed on the spot (`_free_blocks`,
   `sched.py:2094-2097`, which also deletes it from `scheduler.requests`).
3. `finished_sending` (`sched.py:2471-2474`): blocks freed on the spot.

Across steps: the next `schedule()` promotes a waiting request whose id is in
`finished_recving_kv_req_ids` (`sched.py:643-652`, `sched.py:2414-2429` →
`_update_waiting_for_remote_kv`, `sched.py:2380-2412`): if the id is also in
`failed_recving_kv_req_ids` and nothing valid was computed, the blocks are
freed and the request goes back to `WAITING` to prefill locally
(`recomputing`, `sched.py:2399`); otherwise the blocks are kept as the
request's KV and it proceeds to decode (`decoding`, `sched.py:2405`). And at any
time the API layer may abort the request (`finish_requests`,
`sched.py:2012-2073`): a waiting request's free is deferred as in 1, a decoding
one is freed; a P request is already finished and is skipped
(`sched.py:2041-2043`) — there is no API-layer action that reaches a P-side
hold.

The P side keeps only `pHeld`; its connector signal is read for
`finishedSending` alone (a P scheduler has no request in
`WAITING_FOR_REMOTE_KVS`, and `request_finished` on D returns
`delay_free_blocks = False`, `conn.py:404-455`).

In a multi-host or multi-rank deployment (`conn.py:398-402`
`get_finished_count = len(kv_ip)`, `tpu_connector.py:2566-2583`
`get_finished_count`), `KVOutputAggregator` (`utils.py`) sits between the
worker-rank connectors and `update_from_output`: it counts `finished_sending` /
`finished_recving` reports per request across workers and withholds the
completion from the scheduler until all workers have reported
(`pSentCount = expectedP`, `dRecvCount = expectedD`), while unioning and
forwarding `invalid_block_ids` (`loadError`) immediately on whichever step any
worker reports a failure (`MultiWorker.lean`).

## Properties

* `leakP` (`mwLeakP`): the send is published (on all workers) and P still
  holds the blocks.
* `leakD` (`mwLeakD`): the receive is published (on all workers) and D
  either keeps the request parked with its blocks and nothing to promote it, or
  has finished it with the free deferred. Both are *absorbing* for every
  connector and policy (`leakP_absorbing`, `leakD_absorbing`,
  `mwLeakP_absorbing`, `mwLeakD_absorbing`): nothing in the engine can
  free them afterwards, because the only events that could are the
  publications, and each happens once. That is what makes them leaks rather than
  delays.
* `decodesUnwritten` (`mwDecodesUnwritten`): D's request proceeds to decode
  while the receive is not published as done (on every worker). In the real
  system this is the `AssertionError` the connector's comment says kills the
  EngineCore (`conn.py:641-644`).
* `mwEarlyFree`: on a multi-worker node, P frees its HBM (`pHeld = false`)
  while a P worker is still unpublished, or D frees its HBM (`dHeld = false`)
  while a D worker is still unpublished — discharging the pipeline models' **A4**
  assumption across workers when one worker fails early while another worker's
  transfer is still in flight.

`Contract conn` is the obligation on a connector that the theorems need:
every P outcome yields `finishedSending`, every D outcome yields
`finishedRecving`, and a failed D outcome also yields `loadError` — the two
`base.py` contracts (`base.py:375-393`: "the request must still be reported via
`get_finished()`, and the failed block IDs must appear here no later than that
same pass"). `contract_safe` (single worker) and `mw_contract_safe` (2-worker
`KVOutputAggregator` with `expectedP = 2`, `expectedD = 2`) prove that any
connector satisfying `Contract`, under either failure policy and any mix of
worker outcomes or arrival orders, never frees HBM while a worker is still in
flight, never leaks, and never decodes unwritten KV; `connectorFixed` satisfies
it (`fixed_contract`, `fixed_safe`, `mw_fixed_safe`). The other three do not,
and each has traces: `trace_current_connector_p`,
`trace_current_connector_d_abort_after`,
`trace_current_connector_d_abort_before`, and
`trace_mw_current_partial_failure_leaks` (the current connector),
`trace_naive_decodes` (why the comment's fear was justified),
`trace_loadErrorOnly_fail`, `trace_loadErrorOnly_recompute` (why the failure
channel alone is not enough), and `trace_mw_overcounted_expected_leaks` (why
`get_finished_count()` must match the participating worker count). Bounded
search (`search`, `mwSearch`) exhausts the state space for each configuration.

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
`connectorFixed` free `dHeld = false`.

## Assumptions

* **B1 (every participating worker eventually publishes).** Each worker rank
  that was armed for the transfer eventually produces a terminal publication
  (`some true` or `some false`), and `get_finished_count()` equals the number
  of participating workers (`expectedP = 2`, `expectedD = 2`). On both paths
  this is enforced by the 180 s native session deadline
  (`TPU_P2P_WAIT_PULL_TIMEOUT`, `conn.py:529`, `tpu_connector.py`
  `get_p2p_wait_pull_timeout()`) together with `_sync_stage3_tp_submit_states`
  in `tpu_connector.py` (which broadcasts `"failed"`, `"uncertain"`, or
  `"accepted"` across TP ranks so no rank is left unarmed while another waits).
  `MultiWorker.lean` models the 2-worker `KVOutputAggregator` combining step
  directly.
* **B2 (whole-transfer failure).** A failed pull invalidates every local
  block of the request, so recomputation starts from token 0 and the
  `num_computed_tokens ≠ 0` branch of `_update_waiting_for_remote_kv`
  (`sched.py:2393-2395`, cache the valid prefix) is not reached. tpu-sync
  reports a receive as one outcome (`(!session->GetStatus().ok() ? failed_recving_ : done_recving_)`
  in `KVCacheManagerWithTransfer::CompleteReadWithDetails`), so a connector has
  no finer information to report.
* **B3 (no preemption, prefix caching or chunking).** None of them changes
  which event frees a transfer-held block.
-/

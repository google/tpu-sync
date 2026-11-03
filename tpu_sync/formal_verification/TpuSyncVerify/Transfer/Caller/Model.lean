import TpuSyncVerify.Common.System
import TpuSyncVerify.Common.ModelCheck

/-! # `Caller`: single-worker state, events, `step`, properties

Part of `TpuSyncVerify.Transfer.Caller`; the module docstring and source map
live in `Caller.lean`. -/

namespace TpuSyncVerify.Transfer.Caller

/-- What the worker-side connector tells the scheduler about one request in
one step: membership in the two sets `get_finished()` returns
(`KVConnectorTransferResults.finished_sending` / `finished_recving` with
`get_transfer_results`) and whether the request is on vLLM's load-failure
channel — its local blocks in `get_block_ids_with_load_errors()`
(`invalid_block_ids`) or its id in `KVConnectorTransferResults.failed_recving`
(`Caller.lean` §"What is modelled", step 1). -/
structure Signal where
  finishedSending : Bool := false
  finishedRecving : Bool := false
  loadError : Bool := false
  deriving Repr, DecidableEq

/-- A worker-side connector, as the function it applies to tpu-sync's report
for one request: `onSend ok` on the producer node when the request is in
`done_sending` (`ok`) or `failed_recving` (`!ok`); `onRecv ok` on the consumer
node when it is in `done_recving` (`ok`) or `failed_recving` (`!ok`). -/
structure Connector where
  onSend : Bool → Signal
  onRecv : Bool → Signal

/-- Shipping two-channel connector meeting `Contract`: every outcome is
forwarded on `finished_sending` / `finished_recving` and a failed receive also
goes on vLLM's load-failure channel. Instantiated by both production connectors:
* `tpu-inference` `TPUConnectorWorker.get_transfer_results` (`conn.py`, paired
  with vLLM's `KVConnectorModelRunnerMixin._get_kv_connector_output` and
  `Scheduler._handle_failed_recving`: producer merges `failed_recving` into
  `finished_sending`; consumer drops the id from `_submitted` and reports it in
  both `finished_recving` and `failed_recving`, the request-level channel), and
* `vllm-torchtpu` Stage-3 `TPURaidenConnectorWorker` (`tpu_connector.py`:
  `get_finished` / `_poll_finished`, staging failed receive blocks into
  `_failed_block_ids` for `get_block_ids_with_load_errors`, the block-level
  channel). -/
def connectorTwoChannel : Connector where
  onSend _ := { finishedSending := true }
  onRecv ok := { finishedRecving := true, loadError := !ok }

/-- Mutant: `failed_recving` is logged and dropped on both nodes
(`finishedSending := ok`, `finishedRecving := ok`, `loadError := false` — a
`get_finished`-only connector where neither `get_block_ids_with_load_errors` nor
`get_transfer_results` is overridden). -/
def connectorDropFailed : Connector where
  onSend ok := { finishedSending := ok }
  onRecv ok := { finishedRecving := ok }

/-- Mutant: a failed receive is reported in `finished_recving` (`done_recving`)
with no signal on the load-failure channel — a one-channel shortcut, and also
what happens if a `get_finished()` wrapper (`return results.finished_sending,
results.finished_recving`, without `get_block_ids_with_load_errors`) is invoked
by a runner that does not call `get_transfer_results()`. -/
def connectorNaive : Connector where
  onSend _ := { finishedSending := true }
  onRecv _ := { finishedRecving := true }

/-- Mutant: failed receives reported on the load-failure channel without the
`finished_recving` completion report. -/
def connectorLoadErrorOnly : Connector where
  onSend _ := { finishedSending := true }
  onRecv ok := { finishedRecving := ok, loadError := !ok }

/-- vLLM's `KVTransferConfig.kv_load_failure_policy` (`kvcfg.py`; `"fail"` is
the default). -/
inductive Policy where
  | recompute
  | fail
  deriving Repr, DecidableEq

structure Cfg where
  conn : Connector
  policy : Policy

/-- The decode-node request, as far as its transfer-held blocks are concerned. -/
inductive DStatus where
  /-- `WAITING_FOR_REMOTE_KVS`, blocks allocated, pull issued. -/
  | waitingRemoteKv
  /-- promoted with its blocks as its KV cache (`Scheduler._update_waiting_for_remote_kv`). -/
  | decoding
  /-- promoted after a load failure: blocks freed, back to `WAITING` to prefill locally (`Scheduler._update_waiting_for_remote_kv`). -/
  | recomputing
  /-- finished (aborted by the API layer, or `FINISHED_ERROR`); blocks freed unless deferred. -/
  | finished
  deriving Repr, DecidableEq

structure S where
  /-- `send.published`: what `poll_stats()` on P has reported for the send. -/
  sendPub : Option Bool := none
  /-- `recv.published`: what `poll_stats()` on D has reported for the receive. -/
  recvPub : Option Bool := none
  /-- P still holds the request's KV blocks (and the request in `scheduler.requests`). -/
  pHeld : Bool := true
  dStatus : DStatus := .waitingRemoteKv
  /-- D still holds the blocks allocated for the remote KV. -/
  dHeld : Bool := true
  /-- `request_id ∈ finished_recving_kv_req_ids` (`Scheduler`). -/
  dFinRecv : Bool := false
  /-- `request_id ∈ failed_recving_kv_req_ids` (`Scheduler`). -/
  dFailRecv : Bool := false
  deriving Repr, DecidableEq

/-- P finished and holding (`TPUConnectorScheduler.request_finished`), D
allocated and waiting (`TPUConnectorScheduler.update_state_after_alloc`), pull
issued, nothing published. -/
def init : S := {}

inductive Ev where
  /-- tpu-sync publishes the send (`done_sending` if `ok`, else `failed_recving`,
  `(failed ? failed_recving_ : done_sending_).insert(session->req_id())` in
  `KVCacheManagerWithTransfer::CompleteReadWithDetails`) and P's next
  `get_finished()` / `get_transfer_results()` forwards it. -/
  | pOutcome (ok : Bool)
  /-- tpu-sync publishes the receive (`done_recving` if `ok`, else
  `failed_recving`, `(!session->GetStatus().ok() ? failed_recving_ : done_recving_).insert(session->req_id())`
  in `KVCacheManagerWithTransfer::CompleteReadWithDetails`) and D's next
  `get_finished()` / `get_block_ids_with_load_errors()` (or
  `get_transfer_results()`) forward it. -/
  | dOutcome (ok : Bool)
  /-- D's next `Scheduler.schedule` promotes the waiting request. -/
  | promote
  /-- D's request leaves the engine: an API-layer abort
  (`Scheduler.finish_requests`) at any time, or normal completion once decoding. -/
  | finish
  deriving Repr, DecidableEq

/-- `Scheduler._update_from_kv_xfer_finished`: `finished_sending` frees the blocks. -/
def applySend (sig : Signal) (s : S) : S :=
  if sig.finishedSending then { s with pHeld := false } else s

/-- `Scheduler._handle_invalid_blocks` and `Scheduler._handle_failed_recving`
for a request still in `WAITING_FOR_REMOTE_KVS` (`Caller.lean` step 1). -/
def applyLoadError (policy : Policy) (s : S) : S :=
  if s.dStatus = .waitingRemoteKv then
    match policy with
    | .recompute => { s with dFailRecv := true }
    | .fail =>
      -- `Scheduler.finish_requests(FINISHED_ERROR)`
      { s with dStatus := .finished, dHeld := s.dHeld && !s.dFinRecv,
               dFinRecv := false, dFailRecv := false }
  else s

/-- `Scheduler._update_from_kv_xfer_finished`: `finished_recving` for a waiting
request records it; for a finished request frees its blocks. -/
def applyFinishedRecving (s : S) : S :=
  match s.dStatus with
  | .waitingRemoteKv => { s with dFinRecv := true }
  | .finished => { s with dHeld := false }
  -- `decoding` / `recomputing` need a promotion, which needs this very
  -- report, which happens once: unreachable (the real code would assert).
  | _ => s

/-- One `Scheduler.update_from_output` on D, in its order: load errors first,
then transfer completions. -/
def applyRecv (policy : Policy) (sig : Signal) (s : S) : S :=
  let s := if sig.loadError then applyLoadError policy s else s
  if sig.finishedRecving then applyFinishedRecving s else s

def pOutcome (c : Cfg) (ok : Bool) (s : S) : Option S :=
  if s.sendPub = none then
    some (applySend (c.conn.onSend ok) { s with sendPub := some ok })
  else none

def dOutcome (c : Cfg) (ok : Bool) (s : S) : Option S :=
  if s.recvPub = none then
    some (applyRecv c.policy (c.conn.onRecv ok) { s with recvPub := some ok })
  else none

/-- `Scheduler._try_promote_blocked_waiting_request` →
`Scheduler._update_waiting_for_remote_kv`. -/
def promote (s : S) : Option S :=
  if s.dStatus = .waitingRemoteKv ∧ s.dFinRecv = true then
    if s.dFailRecv then
      some { s with dStatus := .recomputing, dHeld := false, dFinRecv := false, dFailRecv := false }
    else
      some { s with dStatus := .decoding, dFinRecv := false }
  else none

/-- `Scheduler.finish_requests` on D. -/
def finish (s : S) : Option S :=
  match s.dStatus with
  | .waitingRemoteKv =>
    some { s with dStatus := .finished, dHeld := s.dHeld && !s.dFinRecv,
                  dFinRecv := false, dFailRecv := false }
  | .decoding => some { s with dStatus := .finished, dHeld := false }
  | .recomputing => some { s with dStatus := .finished }
  | .finished => none

def step (c : Cfg) (s : S) : Ev → Option S
  | .pOutcome ok => pOutcome c ok s
  | .dOutcome ok => dOutcome c ok s
  | .promote => promote s
  | .finish => finish s

def sys (c : Cfg) : System S Ev := ⟨init, step c⟩

/-! ## Properties -/

/-- The send is published and P still holds the blocks. -/
def leakP (s : S) : Bool := s.sendPub.isSome && s.pHeld

/-- The receive is published and D either keeps the request parked with its
blocks and nothing to promote it, or has finished it with the free deferred. -/
def leakD (s : S) : Bool :=
  s.recvPub.isSome &&
    ((s.dStatus == .waitingRemoteKv && !s.dFinRecv && s.dHeld) ||
     (s.dStatus == .finished && s.dHeld))

/-- D proceeds to decode on KV that was never written. -/
def decodesUnwritten (s : S) : Bool :=
  s.dStatus == .decoding && s.recvPub != some true

def violates (s : S) : Bool := leakP s || leakD s || decodesUnwritten s

def Safe (s : S) : Prop := violates s = false

/-- What a connector owes the scheduler (`KVConnectorBase_V1.get_finished`,
`KVConnectorBase_V1.get_block_ids_with_load_errors`, and the
`KVConnectorTransferResults` contract that a failed receive also appears in
`finished_recving`). -/
structure Contract (conn : Connector) : Prop where
  /-- Every send outcome is reported as finished sending. -/
  send_total : ∀ ok, (conn.onSend ok).finishedSending = true
  /-- Every receive outcome is reported as finished receiving — including a
  failed one, "so the scheduler can release the request". -/
  recv_total : ∀ ok, (conn.onRecv ok).finishedRecving = true
  /-- A failed receive is also reported on the load-failure channel, no later
  than its completion. -/
  recv_failure_reported : (conn.onRecv false).loadError = true
  /-- No cross-talk: a producer-side report never names the request in
  `finished_recving` or the load-failure set, and a consumer-side report never
  names it in `finished_sending`. The model does not read these fields; the
  real scheduler would assert on them (`Scheduler._update_from_kv_xfer_finished` →
  `Scheduler._free_blocks`), so the contract forbids them. -/
  send_no_recv : ∀ ok, (conn.onSend ok).finishedRecving = false ∧ (conn.onSend ok).loadError = false
  recv_no_send : ∀ ok, (conn.onRecv ok).finishedSending = false

theorem twoChannel_contract : Contract connectorTwoChannel where
  send_total := by intro ok; rfl
  recv_total := by intro ok; rfl
  recv_failure_reported := rfl
  send_no_recv := by intro ok; exact ⟨rfl, rfl⟩
  recv_no_send := by intro ok; rfl

end TpuSyncVerify.Transfer.Caller

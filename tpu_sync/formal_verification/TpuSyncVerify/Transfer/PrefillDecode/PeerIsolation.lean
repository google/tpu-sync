import TpuSyncVerify.Common.ListAux
import TpuSyncVerify.Transfer.PrefillDecode.Receive

/-!
# Cross-peer fault isolation and staging-slot starvation on `StartRead`

Models a decode consumer (`KVCacheManagerWithTransfer`) pulling KV caches from
multiple prefill producers concurrently (`Peer.sick` vs. `Peer.healthy`) across
the two finite consumer resources shared by `StartRead`. `mgr.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc`, `recv.cc` is
`tpu_sync/kv_cache/transfer_receive_session.cc`, `arena.cc` is
`tpu_sync/core/staging_arena.cc`, and `control_test.cc` is
`tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc`. Exact line
ranges for every symbol, sub-block, and test are in the generated
`## Source map` table below.

1. **Host staging slots (`StagingBlockAllocator`, capacity `numSlots`):**
   `KVCacheManagerWithTransfer::StartRead` calls
   `TransferReceiveSession::Create` → `AllocateStagingForLoad` **before**
   contacting the producer, because `PullStreamRequestSpec` sends the allocated
   host block IDs (`dst_block_ids = load_plan_.transport_host_block_ids` in
   `ExecutePullRequest`) to the producer. The session starts at
   `Recv.initLoad numLayers` (`inFlight = 1, pullPending = true, hasStaging = true`)
   and holds its staging slot until `Lifecycle.settleLocked`
   (`FinishLocked` / `EndRecvOpLocked` → `ReleaseStagingLocked` →
   `StagingBlockAllocator::ReleaseSlot`) sets `hasStaging = false` when
   `draining = true ∧ inFlight = 0`. If no staging slot can
   be acquired, `StartRead` immediately records the request in
   `failed_recving_` (`if (!created.ok()) { failed_recving_.insert(req_id); ... }`).

2. **Outbound handshake worker pool (`push_pool_`, capacity `poolSize`):**
   `TransferReceiveSession::ExecutePullRequest` (`base_->push_pool()->Schedule`)
   schedules a task on `push_pool_` to call `control_backend_->SendPullRequest`:
   - On `Backend.tcpBlocking` (`TcpControlPlaneBackend::SendPullRequestBlocking`,
     `SendPullRequest` → `SendPullRequestBlocking` in
     `tcp_control_plane_backend.cc`), the worker thread blocks waiting
     for the producer's reply (`pullReply`), holding one worker from
     `dispatchPull` until `pullReply`.
   - On `Backend.grpcAsync` (`GrpcControlPlaneBackend::SendPullRequest` in
     `grpc_control_plane_backend.cc`), the worker issues an async RPC
     and returns immediately, holding no worker while waiting on the wire.

## Lock scopes

| Event | Scopes | Between them |
|---|---|---|
| `startRead peer` | 1 outer `KVCacheManagerWithTransfer::mu_` scope in `StartRead(remote_endpoint)` (calling `TransferReceiveSession::Create` — which acquires session `mu_` and `StagingBlockAllocator::mu_` in `AllocateStagingForLoad` and `InitFromLoadPlan` — and `EmplaceRecvSessionLocked`), followed outside manager `mu_` by `ExecutePullRequest`'s initial session `mu_` scope before `push_pool()->Schedule` | `in_flight_ = 1` (`handshake_pending_ = true`) is set during `InitFromLoadPlan` before the session enters `active_recv_sessions_`, so any `cancel` before `ExecutePullRequest` still sees `in_flight_ = 1` and cannot release the staging slot early |
| `dispatchPull idx` | 1 `push_pool_` worker thread picking up the scheduled task from `ExecutePullRequest` | on `tcpBlocking`, the worker blocks inside `SendPullRequestBlocking` until `pullReply`; on `grpcAsync`, `SendPullRequest` starts an async RPC and returns immediately |
| `sessStep idx e` | `TransferReceiveSession::mu_` scopes of `e : Recv.Ev` (`Receive.lean`) | when `sessStep` settles a session, `ReleaseStagingLocked` → `StagingBlockAllocator::ReleaseSlot` runs inside that settling `mu_` scope, returning the slot to `freeSlots` under `StagingBlockAllocator::mu_` |

## What this module proves

1. **Resource conservation (`reachable_inv`):**
   Across all reachable states for any backend and slot policy:
   - `s.freeSlots + activeStaging s.sessions = s.cfg.numSlots` (no staging slot
     is ever leaked or double-freed);
   - `s.freeWorkers + activeWorkers s.cfg.backend s.sessions = s.cfg.poolSize`
     (worker threads are conserved);
   - every active session satisfies `Recv.Inv` and `Recv.Safe` (`Receive.lean`).

2. **TCP head-of-line blocking vs. gRPC cross-peer progress (Issue #888):**
   - On `Backend.tcpBlocking`, once `poolSize` handshakes to `Peer.sick` are
     dispatched, `freeWorkers = 0` and an admitted `Peer.healthy` session cannot
     dispatch its handshake (`tcp_healthy_blocked_when_pool_full`,
     `trace_tcp_sick_peer_blocks_healthy`).
   - On `Backend.grpcAsync` (`0 < poolSize`), `freeWorkers = poolSize` is an
     invariant (`grpc_freeWorkers_eq_poolSize`), and **any** newly admitted
     `Peer.healthy` session completes its handshake, H2D copy, settlement, slot
     release, and `done_recving` publication in finitely many steps while every
     `Peer.sick` session stays wedged (`grpc_healthy_can_complete`,
     `trace_grpc_sick_peer_does_not_delay_healthy`,
     `trace_grpc_healthy_progresses_under_backlog`).

3. **Staging-slot starvation counterexample & per-peer quota fix
   (`ControlHandshakeTest.DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`):**
   - Under the shipping `SlotPolicy.unboundedPerPeer`, `numSlots` wedged reads
     to `Peer.sick` exhaust `freeSlots = 0` (even if their session deadlines
     expire via `.cancel`, since `inFlight = 1` keeps their staging pinned until
     `pullReply false`). A subsequent `StartRead` to `Peer.healthy` fails slot
     allocation immediately (`trace_sick_peer_starves_staging_slots`, plus the
     bounded model check counterexample).
   - Under `SlotPolicy.perPeerQuota maxPerPeer` (the per-peer admission the
     comment above `DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer` calls
     for; no such policy exists in C++ at the baseline commit), `Peer.sick` can
     hold at most `maxPerPeer` slots (`reachable_sick_staging_le_quota`). Thus
     whenever `maxPerPeer < numSlots` and `Peer.healthy` has fewer than
     `min maxPerPeer (numSlots - maxPerPeer)` sessions holding staging (in particular,
     when the first healthy read arrives), `canAdmit s .healthy = true` holds on
     **every** reachable state (`reachable_quota_admits_healthy`,
     `trace_per_peer_quota_admits_healthy`).

## Source map

Generated by `tools/resolve_anchors.py`.

<!-- source-map:begin -->
| File | Symbol / anchor | Lines |
|---|---|---|
| `mgr.cc` | `KVCacheManagerWithTransfer::EmplaceRecvSessionLocked` | `:703-719` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` | `:1045-1149` |
| `mgr.cc` | `KVCacheManagerWithTransfer::StartRead(remote_endpoint)` (`absl::StatusOr<std::shared_ptr<TransferReceiveSession>> created =` .. `transfer_details_[req_id] = DetailsFromStatus(created.status());`) | `:1104-1111` |
| `recv.cc` | `TransferReceiveSession::Create(remote_block_ids)` | `:164-190` |
| `recv.cc` | `TransferReceiveSession::AllocateStagingForLoad` | `:299-345` |
| `recv.cc` | `TransferReceiveSession::AllocateStagingForLoad` (`absl::StatusOr<StagingAllocation> acquired =` .. `staging_ = *std::move(acquired);`) | `:314-326` |
| `recv.cc` | `TransferReceiveSession::InitFromLoadPlan` | `:417-437` |
| `recv.cc` | `TransferReceiveSession::ReleaseStagingLocked` | `:439-442` |
| `recv.cc` | `TransferReceiveSession::FinishLocked` | `:449-464` |
| `recv.cc` | `TransferReceiveSession::EndRecvOpLocked` | `:500-511` |
| `recv.cc` | `TransferReceiveSession::ExecutePullRequest` | `:607-683` |
| `recv.cc` | `TransferReceiveSession::ExecutePullRequest` (`base_->push_pool()->Schedule`) | `:669-682` |
| `arena.cc` | `StagingBlockAllocator::Acquire` | `:514-518` |
| `arena.cc` | `StagingBlockAllocator::ReleaseSlot` | `:571-577` |
| `tcp_control_plane_backend.cc` | `TcpControlPlaneBackend::SendPullRequest` | `:585-590` |
| `tcp_control_plane_backend.cc` | `TcpControlPlaneBackend::SendPullRequestBlocking` | `:592-660` |
| `grpc_control_plane_backend.cc` | `GrpcControlPlaneBackend::SendPullRequest` | `:306-334` |
| `control_test.cc` | `ControlHandshakeTest.ConsumerGivesUpOnProducerThatNeverAnswers` | `:687-719` |
| `control_test.cc` | `*` (`// Cross-peer fault isolation on the consumer handshake path` .. `// on push_pool_ and is being retired rather than fixed.`) | `:897-908` |
| `control_test.cc` | `ControlHandshakeTest.GrpcSickPeerDoesNotDelayHandshakeToHealthyPeer` | `:986-1026` |
| `control_test.cc` | `ControlHandshakeTest.GrpcHealthyPeerProgressesWhileSickPeerBacklogDrains` | `:1033-1090` |
| `control_test.cc` | `*` (`// A receive session holds its staging slot for the whole handshake` .. `// StartRead, which is a separate change.`) | `:1092-1102` |
| `control_test.cc` | `ControlHandshakeTest.SickPeerStarvesStagingSlotsForHealthyPeer` | `:1103-1151` |
<!-- source-map:end -->
-/

namespace TpuSyncVerify.Transfer.PrefillDecode.PeerIsolation

open TpuSyncVerify.Transfer (Lifecycle)
open TpuSyncVerify.Transfer.PrefillDecode (Recv)

/-- Target prefill producer peer for a consumer `StartRead` request:
- `sick`: a wedged / unresponsive producer (`SilentProducer` / `StalledGrpcProducer`)
- `healthy`: a producer whose handshake is answered promptly; in the model it
  also pushes layers to completion (the gRPC tests' "healthy" peer is a
  `StalledGrpcProducer` and they assert only prompt contact, see the traces) -/
inductive Peer where
  | sick
  | healthy
  deriving Repr, DecidableEq

/-- Control-plane handshake backend (`TransferReceiveSession::ExecutePullRequest`). -/
inductive Backend where
  | tcpBlocking
  | grpcAsync
  deriving Repr, DecidableEq

/-- Host staging slot admission policy at `KVCacheManagerWithTransfer::StartRead`
(`TransferReceiveSession::AllocateStagingForLoad`). -/
inductive SlotPolicy where
  | unboundedPerPeer
  | perPeerQuota (maxPerPeer : Nat)
  deriving Repr, DecidableEq

/-- One receive session in `active_recv_sessions_` together with its target
prefill `peer` and whether `ExecutePullRequest` has dispatched `SendPullRequest`
via `push_pool_` (`pullDispatched`). -/
structure Entry where
  peer : Peer
  pullDispatched : Bool := false
  recv : Recv
  deriving Repr, DecidableEq

/-- Whether `e` currently holds a `push_pool_` worker thread: on `tcpBlocking`,
a dispatched handshake holds its worker until `pullReply` clears `pullPending`;
on `grpcAsync`, waiting for `pullReply` holds no worker. -/
def Entry.holdsWorker (b : Backend) (e : Entry) : Bool :=
  match b with
  | .tcpBlocking => e.pullDispatched && e.recv.pullPending
  | .grpcAsync => false

/-- Number of sessions currently holding a host staging slot (`hasStaging`). -/
def activeStaging : List Entry → Nat
  | [] => 0
  | e :: es => (if e.recv.life.hasStaging then 1 else 0) + activeStaging es

/-- Number of sessions for peer `p` currently holding a host staging slot. -/
def peerStaging (p : Peer) : List Entry → Nat
  | [] => 0
  | e :: es => (if e.peer == p && e.recv.life.hasStaging then 1 else 0) + peerStaging p es

/-- Number of `push_pool_` workers currently held by in-flight handshakes. -/
def activeWorkers (b : Backend) : List Entry → Nat
  | [] => 0
  | e :: es => (if e.holdsWorker b then 1 else 0) + activeWorkers b es

structure Config where
  numLayers : Nat := 1
  numSlots : Nat := 8
  poolSize : Nat := 4
  backend : Backend := .grpcAsync
  policy : SlotPolicy := .unboundedPerPeer
  deriving Repr, DecidableEq

structure State where
  cfg : Config
  freeSlots : Nat
  freeWorkers : Nat
  sessions : List Entry := []
  /-- Whether a `StartRead` to `Peer.healthy` failed staging allocation up front
  and was dropped into `failed_recving_` (`failed_recving_.insert(req_id)` in `StartRead`). -/
  rejectedHealthy : Bool := false
  /-- Whether a `StartRead` to `Peer.sick` failed staging allocation up front. -/
  rejectedSick : Bool := false
  deriving Repr, DecidableEq

def init (cfg : Config) : State :=
  { cfg, freeSlots := cfg.numSlots, freeWorkers := cfg.poolSize }

/-- Whether `StartRead` for `peer` can acquire a host staging slot. -/
def canAdmit (s : State) (p : Peer) : Bool :=
  0 < s.freeSlots &&
  match s.cfg.policy with
  | .unboundedPerPeer => true
  | .perPeerQuota maxPerPeer => peerStaging p s.sessions < maxPerPeer

/-- Events that require `SendPullRequest` to have been dispatched (`pullDispatched`)
before the producer can reply or push data. -/
def requiresDispatch : Recv.Ev → Bool
  | .pullReply _ | .pushBegin | .h2dBegin => true
  | _ => false

/-- Events that a wedged (`Peer.sick`) producer never sends (it never replies OK
to `PullStream` and never pushes layers). -/
def allowedForPeer (p : Peer) (e : Recv.Ev) : Bool :=
  match p with
  | .healthy => true
  | .sick =>
    match e with
    | .pullReply true | .pushBegin | .h2dBegin => false
    | _ => true

inductive Ev where
  /-- `KVCacheManagerWithTransfer::StartRead` for target `peer`. -/
  | startRead (peer : Peer)
  /-- `push_pool_` worker dispatches `SendPullRequest` for session `idx`
  (`TransferReceiveSession::ExecutePullRequest`). -/
  | dispatchPull (idx : Nat)
  /-- Session `idx` takes a `Recv.Ev` step `e` (`Receive.lean`). -/
  | sessStep (idx : Nat) (e : Recv.Ev)
  deriving Repr, DecidableEq

def step (s : State) : Ev → Option State
  | .startRead peer =>
    if canAdmit s peer then
      some { s with
        freeSlots := s.freeSlots - 1,
        sessions := s.sessions ++
          [{ peer, pullDispatched := false, recv := Recv.initLoad s.cfg.numLayers }] }
    else
      match peer with
      | .healthy =>
        if s.rejectedHealthy then none else some { s with rejectedHealthy := true }
      | .sick =>
        if s.rejectedSick then none else some { s with rejectedSick := true }
  | .dispatchPull idx =>
    match s.sessions[idx]? with
    | none => none
    | some entry =>
      if entry.pullDispatched || s.freeWorkers == 0 then none
      else
        let entry' : Entry := { entry with pullDispatched := true }
        let freeWorkers' :=
          if entry'.holdsWorker s.cfg.backend then s.freeWorkers - 1 else s.freeWorkers
        some { s with
          freeWorkers := freeWorkers',
          sessions := s.sessions.set idx entry' }
  | .sessStep idx e =>
    match s.sessions[idx]? with
    | none => none
    | some entry =>
      if (requiresDispatch e && !entry.pullDispatched) || !allowedForPeer entry.peer e then
        none
      else
        match Recv.step entry.recv e with
        | none => none
        | some recv' =>
          let entry' : Entry := { entry with recv := recv' }
          let freeSlots' :=
            if entry.recv.life.hasStaging && !recv'.life.hasStaging then
              s.freeSlots + 1
            else s.freeSlots
          let freeWorkers' :=
            if entry.holdsWorker s.cfg.backend && !(entry'.holdsWorker s.cfg.backend) then
              s.freeWorkers + 1
            else s.freeWorkers
          some { s with
            freeSlots := freeSlots',
            freeWorkers := freeWorkers',
            sessions := s.sessions.set idx entry' }

def sys (cfg : Config) : System State Ev where
  init := init cfg
  step := step

/-! ## Monotonicity of `hasStaging` and `pullPending` across `Recv.step` -/

theorem settleLocked_hasStaging_false {l : Lifecycle} (h : l.hasStaging = false) :
    (Lifecycle.settleLocked l).hasStaging = false := by
  unfold Lifecycle.settleLocked; split <;> simp [h]

theorem finishLocked_hasStaging_false {l : Lifecycle} (ok : Bool) (h : l.hasStaging = false) :
    (Lifecycle.finishLocked ok l).hasStaging = false := by
  simp only [Lifecycle.finishLocked]
  split <;> split
  · exact h
  · exact settleLocked_hasStaging_false h
  · exact h
  · exact settleLocked_hasStaging_false h

theorem endOpLocked_hasStaging_false {l : Lifecycle} (h : l.hasStaging = false) :
    (Lifecycle.endOpLocked l).hasStaging = false := by
  unfold Lifecycle.endOpLocked
  split
  · exact h
  · exact settleLocked_hasStaging_false h

/-- `Recv.step` never re-acquires staging once released (`hasStaging` only goes
`true → false`). -/
theorem recv_step_hasStaging_false {r r' : Recv} {e : Recv.Ev}
    (hs : Recv.step r e = some r') (h : r.life.hasStaging = false) :
    r'.life.hasStaging = false := by
  cases e <;>
    simp only [Recv.step, Recv.pushBegin, Recv.pushEnd, Recv.pullReply, Recv.h2dBegin,
      Recv.h2dIssue, Recv.h2dReady, Recv.h2dDone, Recv.netAccount, Recv.pollReady,
      Recv.cancel, Recv.publish] at hs <;>
    (repeat' split at hs)
  all_goals
    first
      | (simp only [Option.map_eq_some_iff, Lifecycle.beginOp] at hs
         obtain ⟨l, hl, rfl⟩ := hs
         split at hl <;> cases hl; exact h)
      | (cases hs <;> (repeat' split) <;>
         simp [h, finishLocked_hasStaging_false, endOpLocked_hasStaging_false])

/-- `Recv.step` never sets `pullPending` back to `true`. -/
theorem recv_step_pullPending_false {r r' : Recv} {e : Recv.Ev}
    (hs : Recv.step r e = some r') (h : r.pullPending = false) :
    r'.pullPending = false := by
  cases e <;>
    simp only [Recv.step, Recv.pushBegin, Recv.pushEnd, Recv.pullReply, Recv.h2dBegin,
      Recv.h2dIssue, Recv.h2dReady, Recv.h2dDone, Recv.netAccount, Recv.pollReady,
      Recv.cancel, Recv.publish] at hs <;>
    (repeat' split at hs)
  all_goals
    first
      | (simp only [Option.map_eq_some_iff] at hs; obtain ⟨l, _, rfl⟩ := hs; exact h)
      | (cases hs <;> exact h)

theorem entry_holdsWorker_mono {b : Backend} {e : Entry} {r' : Recv} {ev : Recv.Ev}
    (hs : Recv.step e.recv ev = some r') (h : e.holdsWorker b = false) :
    (Entry.holdsWorker b { e with recv := r' }) = false := by
  cases b with
  | grpcAsync => rfl
  | tcpBlocking =>
    simp only [Entry.holdsWorker, Bool.and_eq_false_iff] at h ⊢
    rcases h with hd | hp
    · exact Or.inl hd
    · exact Or.inr (recv_step_pullPending_false hs hp)

/-! ## List accounting lemmas -/

@[simp] theorem activeStaging_append_initLoad (es : List Entry) (p : Peer) (n : Nat) :
    activeStaging (es ++ [{ peer := p, pullDispatched := false, recv := Recv.initLoad n }]) =
      activeStaging es + 1 := by
  induction es with
  | nil => rfl
  | cons e es ih => simp [activeStaging, ih]; omega

@[simp] theorem activeWorkers_append_undispatched (b : Backend) (es : List Entry) (p : Peer) (n : Nat) :
    activeWorkers b (es ++ [{ peer := p, pullDispatched := false, recv := Recv.initLoad n }]) =
      activeWorkers b es := by
  induction es with
  | nil => cases b <;> rfl
  | cons e es ih => simp [activeWorkers, ih]

@[simp] theorem peerStaging_append_initLoad (p q : Peer) (es : List Entry) (n : Nat) :
    peerStaging p (es ++ [{ peer := q, pullDispatched := false, recv := Recv.initLoad n }]) =
      peerStaging p es + (if q == p then 1 else 0) := by
  induction es with
  | nil => simp [peerStaging, Recv.initLoad]
  | cons e es ih => simp [peerStaging, ih]; omega

theorem peerStaging_sum (es : List Entry) :
    peerStaging .sick es + peerStaging .healthy es = activeStaging es := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    cases hp : e.peer <;> cases hs : e.recv.life.hasStaging <;>
      simp [peerStaging, activeStaging, hp, hs, ← ih] <;> omega

theorem activeStaging_set : ∀ {es : List Entry} {idx : Nat} {e e' : Entry},
    es[idx]? = some e →
    (e.recv.life.hasStaging = false → e'.recv.life.hasStaging = false) →
    activeStaging (es.set idx e') +
      (if e.recv.life.hasStaging && !e'.recv.life.hasStaging then 1 else 0) =
      activeStaging es
  | [], _, _, _, h, _ => by simp at h
  | a :: as, 0, e, e', h, hmono => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    cases h1 : a.recv.life.hasStaging <;> cases h2 : e'.recv.life.hasStaging
    · simp [activeStaging, h1, h2]
    · have := hmono h1; simp [h2] at this
    · simp [activeStaging, h1, h2]; omega
    · simp [activeStaging, h1, h2]
  | a :: as, idx + 1, e, e', h, hmono => by
    simp only [List.getElem?_cons_succ] at h
    have ih := activeStaging_set h hmono
    simp only [List.set_cons_succ, activeStaging]
    omega

theorem peerStaging_set_le : ∀ {es : List Entry} {idx : Nat} {e e' : Entry} (p : Peer),
    es[idx]? = some e →
    e'.peer = e.peer →
    (e.recv.life.hasStaging = false → e'.recv.life.hasStaging = false) →
    peerStaging p (es.set idx e') ≤ peerStaging p es
  | [], _, _, _, _, h, _, _ => by simp at h
  | a :: as, 0, e, e', p, h, hpeer, hmono => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    cases h1 : a.recv.life.hasStaging <;> cases h2 : e'.recv.life.hasStaging
    · simp [peerStaging, hpeer, h1, h2]
    · have := hmono h1; simp [h2] at this
    · simp [peerStaging, hpeer, h1, h2]
    · simp [peerStaging, hpeer, h1, h2]
  | a :: as, idx + 1, e, e', p, h, hpeer, hmono => by
    simp only [List.getElem?_cons_succ] at h
    have ih := peerStaging_set_le p h hpeer hmono
    simp only [List.set_cons_succ, peerStaging]
    omega

theorem peerStaging_set_eq_of_hasStaging : ∀ {es : List Entry} {idx : Nat} {e e' : Entry} (p : Peer),
    es[idx]? = some e →
    e'.peer = e.peer →
    e'.recv.life.hasStaging = e.recv.life.hasStaging →
    peerStaging p (es.set idx e') = peerStaging p es
  | [], _, _, _, _, h, _, _ => by simp at h
  | a :: as, 0, e, e', p, h, hpeer, hst => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h; simp [peerStaging, hpeer, hst]
  | a :: as, idx + 1, e, e', p, h, hpeer, hst => by
    simp only [List.getElem?_cons_succ] at h
    simp [List.set_cons_succ, peerStaging, peerStaging_set_eq_of_hasStaging p h hpeer hst]

theorem activeWorkers_set_dispatch : ∀ {b : Backend} {es : List Entry} {idx : Nat} {e : Entry},
    es[idx]? = some e →
    e.pullDispatched = false →
    activeWorkers b (es.set idx { e with pullDispatched := true }) =
      activeWorkers b es +
        (if Entry.holdsWorker b { e with pullDispatched := true } then 1 else 0)
  | _, [], _, _, h, _ => by simp at h
  | b, a :: as, 0, e, h, hdisp => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    have h0 : a.holdsWorker b = false := by cases b <;> simp [Entry.holdsWorker, hdisp]
    simp [activeWorkers, h0]; omega
  | b, a :: as, idx + 1, e, h, hdisp => by
    simp only [List.getElem?_cons_succ] at h
    have ih := @activeWorkers_set_dispatch b as idx e h hdisp
    simp only [List.set_cons_succ, activeWorkers, ih]; omega

theorem activeWorkers_set_step : ∀ {b : Backend} {es : List Entry} {idx : Nat} {e e' : Entry},
    es[idx]? = some e →
    (e.holdsWorker b = false → e'.holdsWorker b = false) →
    activeWorkers b (es.set idx e') +
      (if e.holdsWorker b && !(e'.holdsWorker b) then 1 else 0) =
      activeWorkers b es
  | _, [], _, _, _, h, _ => by simp at h
  | b, a :: as, 0, e, e', h, hmono => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    cases h1 : a.holdsWorker b <;> cases h2 : e'.holdsWorker b
    · simp [activeWorkers, h1, h2]
    · have := hmono h1; simp [h2] at this
    · simp [activeWorkers, h1, h2]; omega
    · simp [activeWorkers, h1, h2]
  | b, a :: as, idx + 1, e, e', h, hmono => by
    simp only [List.getElem?_cons_succ] at h
    have ih := activeWorkers_set_step h hmono
    simp only [List.set_cons_succ, activeWorkers]; omega

@[simp] theorem activeWorkers_grpcAsync (es : List Entry) :
    activeWorkers .grpcAsync es = 0 := by
  induction es with
  | nil => rfl
  | cons e es ih => simp [activeWorkers, Entry.holdsWorker, ih]

/-! ## Inductive invariant and resource conservation -/

structure Inv (s : State) : Prop where
  slots : s.freeSlots + activeStaging s.sessions = s.cfg.numSlots
  workers : s.freeWorkers + activeWorkers s.cfg.backend s.sessions = s.cfg.poolSize
  sessions_inv : ∀ e ∈ s.sessions, Recv.Inv e.recv ∧ e.recv.numLayers = s.cfg.numLayers
  quota_sick : ∀ maxPerPeer, s.cfg.policy = .perPeerQuota maxPerPeer →
    peerStaging .sick s.sessions ≤ maxPerPeer

theorem inv_init (cfg : Config) : Inv (init cfg) := by
  refine ⟨by simp [init, activeStaging], by simp [init, activeWorkers], ?_, ?_⟩
  · intro e he; simp [init] at he
  · intro m _; simp [init, peerStaging]

theorem step_cfg {s s' : State} {ev : Ev} (hs : step s ev = some s') :
    s'.cfg = s.cfg := by
  cases ev with
  | startRead peer =>
    simp only [step] at hs
    split at hs
    · cases hs; rfl
    · cases peer <;> dsimp only at hs <;> split at hs <;> cases hs <;> rfl
  | dispatchPull idx =>
    simp only [step] at hs
    split at hs
    · cases hs
    · split at hs <;> cases hs; rfl
  | sessStep idx e =>
    simp only [step] at hs
    split at hs
    · cases hs
    · split at hs
      · cases hs
      · split at hs
        · cases hs
        · cases hs; rfl

theorem step_inv {s s' : State} {ev : Ev} (h : Inv s) (hs : step s ev = some s') :
    Inv s' := by
  rcases h with ⟨hslots, hworkers, hsess, hquota⟩
  cases ev with
  | startRead peer =>
    simp only [step] at hs
    split at hs
    · rename_i hadm
      cases hs
      simp only [canAdmit, Bool.and_eq_true, decide_eq_true_eq] at hadm
      refine ⟨?_, ?_, ?_, ?_⟩
      · simp only [activeStaging_append_initLoad]; omega
      · simp only [activeWorkers_append_undispatched]; exact hworkers
      · intro e he
        simp only [List.mem_append, List.mem_singleton] at he
        rcases he with he | rfl
        · exact hsess e he
        · exact ⟨Recv.inv_initLoad s.cfg.numLayers, rfl⟩
      · intro maxPerPeer hpol
        have hq := hquota maxPerPeer hpol
        simp only [peerStaging_append_initLoad]
        cases peer with
        | sick =>
          have hlt : peerStaging .sick s.sessions < maxPerPeer := by
            have h2 := hadm.2
            rw [hpol] at h2
            simpa using h2
          simp; omega
        | healthy =>
          simp; omega
    · cases peer <;> dsimp only at hs <;> split at hs <;> cases hs <;>
      exact ⟨hslots, hworkers, hsess, hquota⟩
  | dispatchPull idx =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i entry hget
      split at hs
      · cases hs
      · rename_i hguard
        cases hs
        simp only [Bool.or_eq_false_iff, Bool.not_eq_true, beq_eq_false_iff_ne] at hguard
        have hst : activeStaging (s.sessions.set idx { entry with pullDispatched := true }) =
            activeStaging s.sessions := by
          have := activeStaging_set (e' := { entry with pullDispatched := true }) hget (fun h => h)
          simpa using this
        have hw := activeWorkers_set_dispatch (b := s.cfg.backend) hget hguard.1
        refine ⟨by simp only [hst, hslots], ?_, ?_, ?_⟩
        · dsimp only
          split <;> rename_i hhold <;> simp [hhold] at hw <;> omega
        · intro e he
          rcases mem_set_cases he with rfl | he
          · exact hsess entry (getElem?_mem hget)
          · exact hsess e he
        · intro maxPerPeer hpol
          dsimp only at hpol ⊢
          rw [peerStaging_set_eq_of_hasStaging .sick (e' := { entry with pullDispatched := true }) hget rfl rfl]
          exact hquota maxPerPeer hpol
  | sessStep idx e =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i entry hget
      split at hs
      · cases hs
      · split at hs
        · cases hs
        · rename_i recv' hstep
          cases hs
          have hst := activeStaging_set (e' := { entry with recv := recv' }) hget
            (recv_step_hasStaging_false hstep)
          have hw := activeWorkers_set_step (b := s.cfg.backend)
            (e' := { entry with recv := recv' }) hget (entry_holdsWorker_mono hstep)
          refine ⟨?_, ?_, ?_, ?_⟩
          · dsimp only
            split <;> rename_i hrel <;> simp [hrel] at hst <;> omega
          · dsimp only
            split <;> rename_i hrel <;> simp [hrel] at hw <;> omega
          · intro e' he'
            rcases mem_set_cases he' with rfl | he'
            · obtain ⟨hinv_e, hlen_e⟩ := hsess entry (getElem?_mem hget)
              exact ⟨Recv.step_inv hinv_e hstep, (Recv.step_numLayers hstep).trans hlen_e⟩
            · exact hsess e' he'
          · intro maxPerPeer hpol
            dsimp only at hpol ⊢
            exact Nat.le_trans
              (peerStaging_set_le .sick hget rfl (recv_step_hasStaging_false hstep))
              (hquota maxPerPeer hpol)

theorem reachable_inv {cfg : Config} {s : State} (h : (sys cfg).Reachable s) : Inv s :=
  (sys cfg).reachable_induction (inv_init cfg) (fun _ _ _ ih hs => step_inv ih hs) h

theorem reachable_cfg {cfg : Config} {s : State} (h : (sys cfg).Reachable s) : s.cfg = cfg :=
  (sys cfg).reachable_induction (P := fun s => s.cfg = cfg) rfl
    (fun _ _ _ ih hs => (step_cfg hs).trans ih) h

/-- Every active session in any reachable state satisfies `Recv.Safe`. -/
theorem reachable_sessions_safe {cfg : Config} {s : State}
    (h : (sys cfg).Reachable s) {e : Entry} (he : e ∈ s.sessions) : Recv.Safe e.recv :=
  Recv.inv_safe ((reachable_inv h).sessions_inv e he).1

/-! ## General theorems: TCP blocking vs. gRPC non-blocking & per-peer quota -/

/-- On `Backend.tcpBlocking`, when all `poolSize` workers are held by in-flight
handshakes (`activeWorkers = poolSize`), `freeWorkers = 0` and no queued
session (including a healthy peer's session) can dispatch its handshake. -/
theorem tcp_healthy_blocked_when_pool_full {cfg : Config} {s : State}
    (h : (sys cfg).Reachable s)
    (hfull : activeWorkers cfg.backend s.sessions = cfg.poolSize) (idx : Nat) :
    step s (.dispatchPull idx) = none := by
  have hinv := (reachable_inv h).workers
  have hcfg := reachable_cfg h
  rw [hcfg, hfull] at hinv
  have hfw : s.freeWorkers = 0 := by omega
  simp only [step, hfw]
  split <;> simp

/-- On `Backend.grpcAsync`, `freeWorkers = cfg.poolSize` on every reachable
state: waiting handshakes to a wedged peer never consume a worker. -/
theorem grpc_freeWorkers_eq_poolSize {cfg : Config} {s : State}
    (h : (sys cfg).Reachable s) (hgrpc : cfg.backend = .grpcAsync) :
    s.freeWorkers = cfg.poolSize := by
  have hinv := (reachable_inv h).workers
  have hcfg := reachable_cfg h
  rw [hcfg, hgrpc, activeWorkers_grpcAsync] at hinv
  omega

/-- Completed `Entry` state after a 1-layer `Peer.healthy` session finishes its
handshake, H2D copy, settlement, and `done_recving` publication. -/
def completedHealthyEntry : Entry :=
  { peer := .healthy, pullDispatched := true,
    recv := { numLayers := 1,
              life := { inFlight := 0, draining := true, done := true,
                        statusOk := true, hasStaging := false },
              issued := 1, ready := 1, completed := 1,
              layersAccounted := 0, published := some true,
              pushes := 0, pullPending := false,
              pending := 0, retired := 1 } }

/-- On `Backend.grpcAsync` with `0 < poolSize` and `numLayers = 1`, any newly
admitted `Peer.healthy` session at index `idx` completes its handshake, H2D
copy, settlement, staging-slot release (`freeSlots` increases by 1), and
`done_recving` publication (`published = some true`) in 7 steps without any
`Peer.sick` session taking a step. -/
theorem grpc_healthy_can_complete {s : State} {idx : Nat}
    (hgrpc : s.cfg.backend = .grpcAsync)
    (hfw : 0 < s.freeWorkers)
    (hget : s.sessions[idx]? =
      some { peer := .healthy, pullDispatched := false, recv := Recv.initLoad 1 }) :
    let evs : List Ev :=
      [.dispatchPull idx,
       .sessStep idx (.pullReply true),
       .sessStep idx .h2dBegin,
       .sessStep idx (.h2dIssue true),
       .sessStep idx .h2dReady,
       .sessStep idx (.h2dDone true),
       .sessStep idx .publish]
    let s' : State :=
      { s with
        freeSlots := s.freeSlots + 1,
        sessions := s.sessions.set idx completedHealthyEntry }
    (sys s.cfg).runFrom s evs = some s' ∧
      s'.freeSlots = s.freeSlots + 1 ∧
      s'.sessions[idx]? = some completedHealthyEntry ∧
      (∀ j, j ≠ idx → s'.sessions[j]? = s.sessions[j]?) := by
  intro evs s'
  have hlt : idx < s.sessions.length := lt_length_of_getElem?_eq hget
  have hfw_ne : (s.freeWorkers == 0) = false := by
    simp only [beq_eq_false_iff_ne]; omega
  refine ⟨?_, rfl, List.getElem?_set_self hlt, fun j hj => List.getElem?_set_ne (Ne.symm hj)⟩
  simp [evs, s', completedHealthyEntry, sys, System.runFrom, step, hget, hgrpc,
    Entry.holdsWorker, hfw_ne, List.getElem?_set_self hlt, List.set_set,
    requiresDispatch, allowedForPeer, Recv.step, Recv.initLoad, Recv.pullReply,
    Recv.h2dBegin, Recv.h2dIssue, Recv.h2dReady, Recv.h2dDone, Recv.publish,
    Lifecycle.beginOp, Lifecycle.endOpLocked, Lifecycle.finishLocked,
    Lifecycle.settleLocked]

/-- Under `SlotPolicy.perPeerQuota maxPerPeer`, `Peer.sick` holds at most
`maxPerPeer` staging slots on every reachable state. -/
theorem reachable_sick_staging_le_quota {cfg : Config} {s : State} {maxPerPeer : Nat}
    (h : (sys cfg).Reachable s) (hpol : cfg.policy = .perPeerQuota maxPerPeer) :
    peerStaging .sick s.sessions ≤ maxPerPeer := by
  have hinv := reachable_inv h
  have hcfg := reachable_cfg h
  exact hinv.quota_sick maxPerPeer (by rw [hcfg, hpol])

/-- Under `SlotPolicy.perPeerQuota maxPerPeer`, whenever `Peer.healthy` holds
fewer than `maxPerPeer` slots and `maxPerPeer + peerStaging .healthy < numSlots`
(in particular, when `maxPerPeer < numSlots` and `peerStaging .healthy = 0`),
at least one staging slot is guaranteed to be free (`0 < s.freeSlots`) and
`canAdmit s .healthy = true` on **every** reachable state — regardless of how
many `StartRead` requests were sent to `Peer.sick`. -/
theorem reachable_quota_admits_healthy {cfg : Config} {s : State} {maxPerPeer : Nat}
    (h : (sys cfg).Reachable s)
    (hpol : cfg.policy = .perPeerQuota maxPerPeer)
    (hhealthy_lt : peerStaging .healthy s.sessions < maxPerPeer)
    (hsum_lt : maxPerPeer + peerStaging .healthy s.sessions < cfg.numSlots) :
    0 < s.freeSlots ∧ canAdmit s .healthy = true := by
  have hinv := reachable_inv h
  have hcfg := reachable_cfg h
  have hsick := reachable_sick_staging_le_quota h hpol
  have hsum := peerStaging_sum s.sessions
  have hslots := hinv.slots
  rw [hcfg] at hslots
  have hfree : 0 < s.freeSlots := by omega
  refine ⟨hfree, ?_⟩
  simp [canAdmit, hcfg, hpol, hfree, hhealthy_lt]

/-! ## Concrete traces and bounded model checks (`ControlHandshakeTest`) -/

/-- Issue #888 background (the cross-peer fault isolation design comment in
`control_test.cc`, not a test: the TCP backend still runs the blocking call on
`push_pool_` and is being retired rather than fixed): on `Backend.tcpBlocking` (`poolSize = 2, numSlots = 4`), two `StartRead` calls
to `Peer.sick` dispatch and hold both worker threads (`freeWorkers = 0`). A
third `StartRead` to `Peer.healthy` acquires a staging slot (`freeSlots = 1`),
but `.dispatchPull 2` is blocked (`none`) until one of the `.sick` handshakes
times out (`pullReply false`). -/
theorem trace_tcp_sick_peer_blocks_healthy :
    let cfg : Config :=
      { numLayers := 1, numSlots := 4, poolSize := 2,
        backend := .tcpBlocking, policy := .unboundedPerPeer }
    let pre : List Ev :=
      [.startRead .sick, .dispatchPull 0,
       .startRead .sick, .dispatchPull 1,
       .startRead .healthy]
    ((sys cfg).run pre).map (fun s => (s.freeSlots, s.freeWorkers)) = some (1, 0) ∧
    (sys cfg).run (pre ++ [.dispatchPull 2]) = none ∧
    ((sys cfg).run (pre ++ [.sessStep 0 (.pullReply false), .dispatchPull 2])).map
      (fun s => (s.freeSlots, s.freeWorkers,
                 (s.sessions[2]?).map Entry.pullDispatched)) =
      some (2, 0, some true) := by
  decide

/-- `ControlHandshakeTest.GrpcSickPeerDoesNotDelayHandshakeToHealthyPeer`:
on `Backend.grpcAsync` (`poolSize = 4, numSlots = 8`), `kPoolSize = 4`
handshakes to `Peer.sick` are outstanding on the wire (`pullDispatched = true,
pullPending = true`). A 5th `StartRead` to `Peer.healthy` immediately dispatches
its handshake — the test's assertion (`healthy.WaitUntilAccepted`, `waited <
0.5`; its "healthy" peer is a `StalledGrpcProducer` that never completes a
read) — and, model-only beyond the test, receives its layer, finishes H2D,
returns its staging slot, and publishes `done_recving` (`published = some true`)
while all 4 `.sick` reads are
still waiting on the wedged producer. -/
theorem trace_grpc_sick_peer_does_not_delay_healthy :
    let cfg : Config :=
      { numLayers := 1, numSlots := 8, poolSize := 4,
        backend := .grpcAsync, policy := .unboundedPerPeer }
    let sick4 : List Ev :=
      [.startRead .sick, .dispatchPull 0,
       .startRead .sick, .dispatchPull 1,
       .startRead .sick, .dispatchPull 2,
       .startRead .sick, .dispatchPull 3]
    let healthy4 : List Ev :=
      [.startRead .healthy, .dispatchPull 4,
       .sessStep 4 (.pullReply true),
       .sessStep 4 .h2dBegin, .sessStep 4 (.h2dIssue true),
       .sessStep 4 .h2dReady, .sessStep 4 (.h2dDone true),
       .sessStep 4 .publish]
    ((sys cfg).run (sick4 ++ healthy4)).map
      (fun s => (s.freeSlots, s.freeWorkers,
                 (s.sessions[4]?).bind (fun e => e.recv.published),
                 (s.sessions[0]?).map (fun e => e.recv.pullPending))) =
      some (4, 4, some true, some true) := by
  decide

/-- `ControlHandshakeTest.GrpcHealthyPeerProgressesWhileSickPeerBacklogDrains`:
`kSickReads = 5 > kPoolSize = 4` wedged reads to `Peer.sick` plus
`kHealthyReads = 3` reads to `Peer.healthy` within `numSlots = 8`: all 3 healthy
reads dispatch their handshakes at once (the test asserts prompt contact only;
its healthy peer is a `StalledGrpcProducer`) and, model-only beyond the test,
complete H2D and `done_recving` publication while all 5 sick reads remain
stalled on the wire. -/
theorem trace_grpc_healthy_progresses_under_backlog :
    let cfg : Config :=
      { numLayers := 1, numSlots := 8, poolSize := 4,
        backend := .grpcAsync, policy := .unboundedPerPeer }
    let sick5 : List Ev :=
      [.startRead .sick, .dispatchPull 0,
       .startRead .sick, .dispatchPull 1,
       .startRead .sick, .dispatchPull 2,
       .startRead .sick, .dispatchPull 3,
       .startRead .sick, .dispatchPull 4]
    let finishHealthy (idx : Nat) : List Ev :=
      [.startRead .healthy, .dispatchPull idx,
       .sessStep idx (.pullReply true),
       .sessStep idx .h2dBegin, .sessStep idx (.h2dIssue true),
       .sessStep idx .h2dReady, .sessStep idx (.h2dDone true),
       .sessStep idx .publish]
    ((sys cfg).run (sick5 ++ finishHealthy 5 ++ finishHealthy 6 ++ finishHealthy 7)).map
      (fun s => (s.freeSlots,
                 (s.sessions[5]?).bind (fun e => e.recv.published),
                 (s.sessions[6]?).bind (fun e => e.recv.published),
                 (s.sessions[7]?).bind (fun e => e.recv.published),
                 peerStaging .sick s.sessions)) =
      some (3, some true, some true, some true, 5) := by
  decide

/-- `ControlHandshakeTest.ConsumerGivesUpOnProducerThatNeverAnswers` (which runs
on the TCP backend with `kPoolSize + 1` reads and `num_layers = 0`; this trace models
only its second half on `grpcAsync` with one layer): stalled reads to
`Peer.sick` give up when their handshake timeout (`pullReply false`) fires,
settle, publish failure (`published = some false`), and return every staging
slot back to `freeSlots = numSlots`. -/
theorem trace_consumer_gives_up_and_drains :
    let cfg : Config :=
      { numLayers := 1, numSlots := 2, poolSize := 2,
        backend := .grpcAsync, policy := .unboundedPerPeer }
    let evs : List Ev :=
      [.startRead .sick, .dispatchPull 0,
       .startRead .sick, .dispatchPull 1,
       .sessStep 0 (.pullReply false), .sessStep 0 .publish,
       .sessStep 1 (.pullReply false), .sessStep 1 .publish]
    ((sys cfg).run evs).map
      (fun s => (s.freeSlots,
                 (s.sessions[0]?).bind (fun e => e.recv.published),
                 (s.sessions[1]?).bind (fun e => e.recv.published))) =
      some (2, some false, some false) := by
  decide

/-- `ControlHandshakeTest.DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`:
under the shipping `SlotPolicy.unboundedPerPeer`, once `numSlots` reads to
`Peer.sick` are outstanding (even if their session deadlines `.cancel` have
already fired, since `inFlight = 1` keeps their staging slots pinned until
`pullReply false`), `freeSlots = 0`. A subsequent `StartRead` to `Peer.healthy`
is rejected outright (`rejectedHealthy = true`). -/
theorem trace_sick_peer_starves_staging_slots :
    let cfg : Config :=
      { numLayers := 1, numSlots := 2, poolSize := 2,
        backend := .grpcAsync, policy := .unboundedPerPeer }
    let evs : List Ev :=
      [.startRead .sick, .dispatchPull 0, .sessStep 0 .cancel,
       .startRead .sick, .dispatchPull 1, .sessStep 1 .cancel,
       .startRead .healthy]
    ((sys cfg).run evs).map
      (fun s => (s.freeSlots, s.rejectedHealthy, peerStaging .sick s.sessions)) =
      some (0, true, 2) := by
  decide

/-- Under `SlotPolicy.perPeerQuota 1` (`maxPerPeer = 1 < numSlots = 2`; the
per-peer admission that the comment above
`ControlHandshakeTest.DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`
calls for, not present in C++ at the baseline commit), the second
`StartRead` to `Peer.sick` is refused by the per-peer
quota, leaving a staging slot free so `Peer.healthy` is admitted and completes
its transfer to `published = some true` with `rejectedHealthy = false`. -/
theorem trace_per_peer_quota_admits_healthy :
    let cfg : Config :=
      { numLayers := 1, numSlots := 2, poolSize := 2,
        backend := .grpcAsync, policy := .perPeerQuota 1 }
    let evs : List Ev :=
      [.startRead .sick, .dispatchPull 0, .sessStep 0 .cancel,
       .startRead .sick,
       .startRead .healthy, .dispatchPull 1,
       .sessStep 1 (.pullReply true),
       .sessStep 1 .h2dBegin, .sessStep 1 (.h2dIssue true),
       .sessStep 1 .h2dReady, .sessStep 1 (.h2dDone true),
       .sessStep 1 .publish]
    ((sys cfg).run evs).map
      (fun s => (s.rejectedSick, s.rejectedHealthy, s.freeSlots,
                 (s.sessions[1]?).bind (fun e => e.recv.published))) =
      some (true, false, 1, some true) := by
  decide

/-! ## Bounded model checks -/

def checkEvents : List Ev :=
  [.startRead .sick, .startRead .healthy,
   .dispatchPull 0, .dispatchPull 1,
   .sessStep 0 (.pullReply false), .sessStep 0 .cancel, .sessStep 0 .publish,
   .sessStep 1 (.pullReply true), .sessStep 1 (.pullReply false),
   .sessStep 1 .h2dBegin, .sessStep 1 (.h2dIssue true),
   .sessStep 1 .h2dReady, .sessStep 1 (.h2dDone true), .sessStep 1 .publish]

/-- Violation predicate for `DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer`:
the first `StartRead` to `Peer.healthy` (`peerStaging .healthy s.sessions = 0` and
no prior healthy session) is rejected at admission (`rejectedHealthy = true`). -/
def starvesFirstHealthy (s : State) : Bool :=
  s.rejectedHealthy &&
  (s.sessions.filter (fun e => e.peer == .healthy)).isEmpty

-- Shipping `unboundedPerPeer` (`numSlots = 2`): bounded search finds the
-- starvation counterexample in 3 steps (`startRead .sick`, `startRead .sick`,
-- `startRead .healthy`).
#guard (match ModelCheck.check
          (sys { numLayers := 1, numSlots := 2, poolSize := 2,
                 backend := .grpcAsync, policy := .unboundedPerPeer })
          checkEvents starvesFirstHealthy 4 with
        | .counterexample _ => true
        | _ => false)

end TpuSyncVerify.Transfer.PrefillDecode.PeerIsolation

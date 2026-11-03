import TpuSyncVerify.Transfer.Reshard.Pipeline.Lemmas

/-! # `Pipeline`: safety properties, inductive invariant, helper and frame lemmas

Part of `TpuSyncVerify.Transfer.Reshard.Pipeline`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ## Safety properties -/

/-- When `poll_stats()` reports `done_recving`, destination HBM holds every
transferred pool's resharded KV data (`dstHbm = good numPools`). -/
def PublicationCorrect (s : Pipeline) : Prop :=
  s.recv.published = some true → s.dstHbm = good s.numPools

/-- Aliased-page rank-wave integrity: once any pool `q` completes its H2D copy
(`pool.stage = .completed`), its destination slot `dstHbm[q]` holds `.kv q` and
is never clobbered by any subsequent H2D wave. -/
def AliasedPageCorrect (s : Pipeline) : Prop :=
  ∀ q < s.numPools, ∀ pool,
    s.recv.pools[q]? = some pool → pool.stage = .completed → s.dstHbm[q]? = some (.kv q)

abbrev DonePoolsIntact (s : Pipeline) : Prop := AliasedPageCorrect s

/-- Once the receiver is published (`done_recving` or `failed_recving`), no pool
has an H2D copy queued or in flight. -/
def DstHbmSafe (s : Pipeline) : Prop :=
  s.recv.published ≠ none → Recv.poolOps s.recv.pools = 0

/-- Once the sender is published (`done_sending_` or `failed_recving_`) or source
HBM is reclaimed, neither `ExecutePush` nor any D2H/push op is active. -/
def SrcHbmSafe (s : Pipeline) : Prop :=
  (s.send.published ≠ none →
    s.send.execActive = false ∧ Send.poolOps s.send.pools = 0) ∧
  (s.reclaimed = true →
    s.send.life.done = true ∧ s.send.execActive = false ∧ Send.poolOps s.send.pools = 0)

/-- Once either session releases its `PoolStagingLease`, no DMA copy or network
push is reading or writing its staging buffer. -/
def StagingSafe (s : Pipeline) : Prop :=
  (s.send.life.hasStaging = false →
    s.send.execActive = false ∧ Send.poolOps s.send.pools = 0) ∧
  (s.recv.life.hasStaging = false →
    s.recv.pushes = 0 ∧ Recv.poolOps s.recv.pools = 0)

/-- Plan-declared transport progress accounting
(`BlockTransportServerPushSession::CompleteIncomingPush`): landed streams never exceed sent
streams, sent streams never exceed `peerCounts`, and any pool that has passed
`OnPoolReceived` (`stage ≠ .waiting`) has landed all `peerCounts[p]` expected
streams. -/
def StreamAccountingSafe (s : Pipeline) : Prop :=
  ∀ p < s.numPools,
    s.landedStreams.getD p 0 ≤ s.sentStreams.getD p 0 ∧
    s.sentStreams.getD p 0 ≤ s.peerCounts.getD p 0 ∧
    (∀ pool, s.recv.pools[p]? = some pool → pool.stage ≠ .waiting →
      s.landedStreams.getD p 0 = s.peerCounts.getD p 0)

/-- Two-phase Arm-then-Dispatch barrier
(`ReshardCoordinator::RunSubPlan`): before `PoolReshardRegisterRecv`
arms the receiver (`armed = false`), no sender pool has launched a D2H copy or
push and no receiver push or H2D is active. -/
def NoUnarmedPush (s : Pipeline) : Prop :=
  s.armed = false →
    (∀ pool ∈ s.send.pools, pool.stage = .pending ∨ pool.stage = .aborted) ∧
    s.recv.pushes = 0 ∧
    (∀ pool ∈ s.recv.pools, pool.stage = .waiting)

/-- Events that drain in-flight operations on either session. -/
def drainEvents (n : Nat) : List Ev :=
  (Send.drainEvents n).map Ev.send ++ (Recv.drainEvents n).map Ev.recv

/-- Every in-flight operation on either session has an enabled drain step in the
composed pipeline. -/
def NoOpLeak (s : Pipeline) : Prop :=
  (0 < s.send.life.inFlight ∨ 0 < s.recv.life.inFlight) →
    ∃ e ∈ drainEvents s.numPools, (step s e).isSome = true

def Safe (s : Pipeline) : Prop :=
  PublicationCorrect s ∧ AliasedPageCorrect s ∧ DstHbmSafe s ∧ SrcHbmSafe s ∧
    StagingSafe s ∧ StreamAccountingSafe s ∧ NoUnarmedPush s ∧ NoOpLeak s

/-! ## Inductive invariant -/

structure Inv (s : Pipeline) : Prop where
  sendInv : Send.Inv .shipping s.send
  recvInv : Recv.Inv s.recv
  num_pos : 0 < s.numPools
  len_peers : s.peerCounts.length = s.numPools
  len_send : s.send.pools.length = s.numPools
  len_recv : s.recv.pools.length = s.numPools
  len_srcHbm : s.srcHbm.length = s.numPools
  len_srcStg : s.srcStaging.length = s.numPools
  len_wire : s.wire.length = s.numPools
  len_dstStg : s.dstStaging.length = s.numPools
  len_dstHbm : s.dstHbm.length = s.numPools
  len_sent : s.sentStreams.length = s.numPools
  len_landed : s.landedStreams.length = s.numPools
  peers_match : ∀ p < s.numPools, ∀ pool,
    s.send.pools[p]? = some pool → pool.peers = s.peerCounts.getD p 0
  srcHbm_ok : s.reclaimed = false → s.srcHbm = good s.numPools
  reclaimed_pub : s.reclaimed = true → s.send.published ≠ none
  srcStg_ok : s.send.life.done = false →
    ∀ p < s.numPools, ∀ pool, s.send.pools[p]? = some pool →
      (∃ k, pool.stage = .pushing k) → s.srcStaging[p]? = some (.kv p)
  sent_bound : ∀ p < s.numPools, ∀ pool,
    s.send.pools[p]? = some pool →
      s.sentStreams.getD p 0 + pool.remainingSlots ≤ pool.peers
  wire_ok : ∀ p < s.numPools,
    0 < s.sentStreams.getD p 0 → s.wire[p]? = some (.kv p)
  landed_le_sent : ∀ p < s.numPools,
    s.landedStreams.getD p 0 ≤ s.sentStreams.getD p 0
  dstStg_ok : s.recv.life.done = false →
    ∀ p < s.numPools, 0 < s.landedStreams.getD p 0 → s.dstStaging[p]? = some (.kv p)
  recv_stage_landed : ∀ p < s.numPools, ∀ pool,
    s.recv.pools[p]? = some pool → pool.stage ≠ .waiting →
      s.landedStreams.getD p 0 = s.peerCounts.getD p 0
  done_intact : AliasedPageCorrect s
  unarmed_send : s.armed = false →
    ∀ p < s.numPools, ∀ pool, s.send.pools[p]? = some pool →
      pool.stage = .pending ∨ pool.stage = .aborted
  unarmed_recv : s.armed = false →
    s.recv.pushes = 0 ∧
    (∀ p < s.numPools, s.landedStreams.getD p 0 = 0) ∧
    (∀ p < s.numPools, ∀ pool, s.recv.pools[p]? = some pool → pool.stage = .waiting)

theorem validConfig_parts {ranks peerCounts : List Nat}
    (h : validConfig ranks peerCounts = true) :
    Send.validSchedule peerCounts = true ∧
    0 < ranks.length ∧
    peerCounts.length = ranks.length := by
  unfold validConfig at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨hne, hlen⟩, hall⟩ := h
  have hpos : 0 < ranks.length := by
    cases ranks with
    | nil => simp at hne
    | cons _ _ => simp
  have hpe_ne : (!peerCounts.isEmpty) = true := by
    cases peerCounts with
    | nil =>
      have : ranks.length = 0 := hlen
      omega
    | cons _ _ => rfl
  exact ⟨by simp [Send.validSchedule, hpe_ne, hall], hpos, hlen.symm⟩

theorem getElem?_map_init_send {peerCounts : List Nat} {p : Nat} {pool : SendPool}
    (h : (peerCounts.map fun n => ({ peers := n, stage := .pending } : SendPool))[p]? = some pool) :
    pool.peers = peerCounts.getD p 0 ∧ pool.stage = .pending := by
  rw [List.getElem?_map] at h
  cases hpe : peerCounts[p]? with
  | none => simp [hpe] at h
  | some n =>
    simp only [hpe, Option.map_some, Option.some.injEq] at h
    subst h
    simp [List.getD_eq_getElem?_getD, hpe]

theorem getElem?_map_init_recv {ranks : List Nat} {p : Nat} {pool : Pool}
    (h : (ranks.map fun r => ({ rank := r, stage := .waiting } : Pool))[p]? = some pool) :
    pool.rank = ranks.getD p 0 ∧ pool.stage = .waiting := by
  rw [List.getElem?_map] at h
  cases hr : ranks[p]? with
  | none => simp [hr] at h
  | some r =>
    simp only [hr, Option.map_some, Option.some.injEq] at h
    subst h
    simp [List.getD_eq_getElem?_getD, hr]

theorem inv_init {ranks peerCounts : List Nat} (h : validConfig ranks peerCounts = true) :
    Inv (init ranks peerCounts) := by
  obtain ⟨hsend, hpos, hlen⟩ := validConfig_parts h
  refine ⟨Send.inv_init .shipping hsend, Recv.inv_init ranks, hpos,
    hlen, by simp [init, Send.init, hlen], by simp [init, Recv.init],
    by simp [init], by simp [init], by simp [init], by simp [init],
    by simp [init], by simp [init], by simp [init],
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro p _ pool hget
    exact (getElem?_map_init_send hget).1
  · intro _; rfl
  · intro hr; simp [init] at hr
  · intro _ p _ pool hget ⟨k, hk⟩
    have hst := (getElem?_map_init_send hget).2
    simp [hst] at hk
  · intro p _ pool hget
    have hst := (getElem?_map_init_send hget).2
    simp [init, SendPool.remainingSlots, hst]
  · intro p _ hgt
    simp [init] at hgt
  · intro p _
    simp [init]
  · intro _ p _ hgt
    simp [init] at hgt
  · intro p _ pool hget hne
    have hst := (getElem?_map_init_recv hget).2
    exact absurd hst hne
  · intro q _ pool hget hdone
    have hst := (getElem?_map_init_recv hget).2
    simp [hst] at hdone
  · intro ha; simp [init] at ha
  · intro ha; simp [init] at ha

theorem inv_initUnarmed {ranks peerCounts : List Nat} (h : validConfig ranks peerCounts = true) :
    Inv (initUnarmed ranks peerCounts) := by
  have hi := inv_init h
  refine ⟨hi.sendInv, hi.recvInv, hi.num_pos, hi.len_peers, hi.len_send, hi.len_recv,
    hi.len_srcHbm, hi.len_srcStg, hi.len_wire, hi.len_dstStg, hi.len_dstHbm,
    hi.len_sent, hi.len_landed, hi.peers_match, hi.srcHbm_ok, hi.reclaimed_pub,
    hi.srcStg_ok, hi.sent_bound, hi.wire_ok, hi.landed_le_sent,
    hi.dstStg_ok, hi.recv_stage_landed, hi.done_intact, ?_, ?_⟩
  · intro _ p _ pool hget
    exact Or.inl (getElem?_map_init_send hget).2
  · refine fun _ => ⟨rfl, fun p _ => ?_, fun p _ pool hget => (getElem?_map_init_recv hget).2⟩
    simp [initUnarmed, init]

/-! ### Helper lemmas for `Safe` and `step_inv` -/

theorem peerCount_pos {s : Pipeline} (h : Inv s) {p : Nat} (hp : p < s.numPools) :
    0 < s.peerCounts.getD p 0 := by
  have hlt : p < s.send.pools.length := by rw [h.len_send]; exact hp
  have hget : s.send.pools[p]? = some s.send.pools[p] := List.getElem?_eq_getElem hlt
  have hallPos : Send.allPosPeers s.send.pools = true := by
    have hv := h.sendInv.valid; unfold Send.validPools at hv
    simp only [Bool.and_eq_true] at hv; exact hv.2
  have hp_pos := Send.peers_pos_of_allPosPeers hallPos hget
  rw [← h.peers_match p hp s.send.pools[p] hget]
  exact hp_pos

theorem step_send_isSome {s : Pipeline} {e : Send.Ev} (he : e ∈ Send.drainEvents s.numPools) :
    (step s (.send e)).isSome = (Send.step .shipping s.send e).isSome := by
  simp only [Send.drainEvents, List.mem_append, List.mem_singleton, List.mem_flatMap,
    List.mem_range] at he
  rcases he with rfl | ⟨p, _, hp⟩
  · simp [step, sendStep]
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl <;> simp [step, sendStep]

theorem step_recv_isSome {s : Pipeline} {e : Recv.Ev} (he : e ∈ Recv.drainEvents s.numPools) :
    (step s (.recv e)).isSome = (Recv.step .finishOnError s.recv e).isSome := by
  simp only [Recv.drainEvents, List.mem_append, List.mem_singleton, List.mem_flatMap,
    List.mem_range] at he
  rcases he with rfl | ⟨p, _, hp⟩
  · simp [step, recvStep]
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl
    · simp [step, recvStep]
    · simp [step, recvStep]
    · simp only [step, recvStep, Recv.step, Recv.h2dDone]
      cases s.recv.pools[p]? <;> simp

theorem inv_noOpLeak {s : Pipeline} (h : Inv s) : NoOpLeak s := by
  intro hif
  rcases hif with hsnd | hrcv
  · obtain ⟨e, he, hsome⟩ := Send.inv_noOpLeak h.sendInv hsnd
    rw [h.len_send] at he
    refine ⟨.send e, ?_, ?_⟩
    · simp only [drainEvents, List.mem_append, List.mem_map]
      exact Or.inl ⟨e, he, rfl⟩
    · rw [step_send_isSome he]; exact hsome
  · obtain ⟨e, he, hsome⟩ := Recv.inv_noOpLeak .finishOnError h.recvInv hrcv
    rw [h.len_recv] at he
    refine ⟨.recv e, ?_, ?_⟩
    · simp only [drainEvents, List.mem_append, List.mem_map]
      exact Or.inr ⟨e, he, rfl⟩
    · rw [step_recv_isSome he]; exact hsome

theorem allCompleted_getElem? {ps : List Pool} (h : Recv.allCompleted ps = true)
    {p : Nat} {pool : Pool} (hget : ps[p]? = some pool) :
    pool.stage = .completed := by
  unfold Recv.allCompleted at h
  rw [List.all_eq_true] at h
  have hmem := List.mem_of_getElem? hget
  exact beq_iff_eq.mp (h pool hmem)

theorem inv_safe {s : Pipeline} (h : Inv s) : Safe s := by
  have hsendSafe := Send.inv_safe h.sendInv
  have hrecvSafe := Recv.inv_safe .finishOnError h.recvInv
  refine ⟨?_, h.done_intact, ?_, ?_, ?_, ?_, ?_, inv_noOpLeak h⟩
  · -- PublicationCorrect
    intro hpub
    have ⟨hallComp, _⟩ := h.recvInv.published_ok hpub
    apply eq_good_of_getElem? h.len_dstHbm
    intro p hp
    have hlt : p < s.recv.pools.length := by rw [h.len_recv]; exact hp
    have hget : s.recv.pools[p]? = some s.recv.pools[p] := List.getElem?_eq_getElem hlt
    exact h.done_intact p hp s.recv.pools[p] hget (allCompleted_getElem? hallComp hget)
  · -- DstHbmSafe
    intro hpub
    cases hb : s.recv.published with
    | none => exact absurd hb hpub
    | some b =>
      have ⟨hd, _⟩ := h.recvInv.published_done b hb
      exact hrecvSafe.2.1 hd
  · -- SrcHbmSafe
    refine ⟨?_, ?_⟩
    · intro hpub
      cases hb : s.send.published with
      | none => exact absurd hb hpub
      | some b =>
        have hd := h.sendInv.published_done b hb
        exact hsendSafe.2.1 hd
    · intro hrec
      have hpub := h.reclaimed_pub hrec
      cases hb : s.send.published with
      | none => exact absurd hb hpub
      | some b =>
        have hd := h.sendInv.published_done b hb
        have ⟨hex, hpo⟩ := hsendSafe.2.1 hd
        exact ⟨hd, hex, hpo⟩
  · -- StagingSafe
    refine ⟨?_, ?_⟩
    · intro hst
      have hd : s.send.life.done = true := by
        rw [h.sendInv.life.staging] at hst; simpa using hst
      exact hsendSafe.2.1 hd
    · intro hst
      have hd : s.recv.life.done = true := by
        rw [h.recvInv.life.staging] at hst; simpa using hst
      have hif := h.recvInv.life.done_idle hd
      have hacc := h.recvInv.accounted; unfold Recv.Accounted at hacc
      omega
  · -- StreamAccountingSafe
    intro p hp
    have hlt : p < s.send.pools.length := by rw [h.len_send]; exact hp
    have hget : s.send.pools[p]? = some s.send.pools[p] := List.getElem?_eq_getElem hlt
    have hpe := h.peers_match p hp s.send.pools[p] hget
    have hbnd := h.sent_bound p hp s.send.pools[p] hget
    exact ⟨h.landed_le_sent p hp, by omega, h.recv_stage_landed p hp⟩
  · -- NoUnarmedPush
    intro ha
    have hlen_s := h.len_send
    have hlen_r := h.len_recv
    refine ⟨?_, (h.unarmed_recv ha).1, ?_⟩
    · intro pool hmem
      obtain ⟨p, hp, hget⟩ := exists_getElem?_of_mem hmem
      exact h.unarmed_send ha p (by omega) pool hget
    · intro pool hmem
      obtain ⟨p, hp, hget⟩ := exists_getElem?_of_mem hmem
      exact (h.unarmed_recv ha).2.2 p (by omega) pool hget

/-! ### Frame lemmas on `Send.step` and `Recv.step` -/

@[simp] theorem endOp_pools (r : Send) : (Send.endOp r).pools = r.pools := rfl
@[simp] theorem endOp_published (r : Send) : (Send.endOp r).published = r.published := rfl
@[simp] theorem endOp_life (r : Send) : (Send.endOp r).life = r.life.endOpLocked := rfl

@[simp] theorem recordPushCompletion_pools (pol : UnregisterPolicy) (ok : Bool) (r : Send) :
    (Send.recordPushCompletion pol ok r).pools = r.pools := by
  unfold Send.recordPushCompletion
  split <;> try rfl
  split <;> try rfl
  split <;> rfl

@[simp] theorem recordPushCompletion_published (pol : UnregisterPolicy) (ok : Bool) (r : Send) :
    (Send.recordPushCompletion pol ok r).published = r.published := by
  unfold Send.recordPushCompletion
  split <;> try rfl
  split <;> try rfl
  split <;> rfl

theorem recordPushCompletion_done_mono (pol : UnregisterPolicy) (ok : Bool) (r : Send)
    (hd : r.life.done = true) :
    (Send.recordPushCompletion pol ok r).life.done = true := by
  unfold Send.recordPushCompletion
  simp [hd]

theorem send_step_published_ne_none {pol : UnregisterPolicy} {r r' : Send} {e : Send.Ev}
    (hs : Send.step pol r e = some r') (hpub : r.published ≠ none) :
    r'.published ≠ none := by
  cases e with
  | d2hIssue p ok =>
    simp only [Send.step, Send.d2hIssue] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> { cases hs; simpa using hpub }
  | execFinish =>
    simp only [Send.step, Send.execFinish] at hs
    split at hs <;> try cases hs
    simpa using hpub
  | d2hDone p ok =>
    simp only [Send.step, Send.d2hDone] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · cases hs; simpa using hpub
    · split at hs <;> { cases hs; simpa using hpub }
  | pushDone p ok =>
    simp only [Send.step, Send.pushDone] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    simpa using hpub
  | cancel =>
    simp only [Send.step, Send.cancel, Option.some.injEq] at hs
    subst hs; exact hpub
  | publish =>
    simp only [Send.step, Send.publish] at hs
    split at hs <;> try cases hs
    simp

theorem send_step_done_mono {pol : UnregisterPolicy} {r r' : Send} {e : Send.Ev}
    (hs : Send.step pol r e = some r') (hd : r.life.done = true) :
    r'.life.done = true := by
  cases e with
  | d2hIssue p ok =>
    simp only [Send.step, Send.d2hIssue] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · cases hs; exact hd
    · cases hs
      exact Lifecycle.endOpLocked_done_mono (recordPushCompletion_done_mono pol false _ hd)
  | execFinish =>
    simp only [Send.step, Send.execFinish] at hs
    split at hs <;> try cases hs
    exact Lifecycle.endOpLocked_done_mono hd
  | d2hDone p ok =>
    simp only [Send.step, Send.d2hDone] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · cases hs
      exact Lifecycle.endOpLocked_done_mono (recordPushCompletion_done_mono pol false _ hd)
    · split at hs <;> { cases hs; exact Lifecycle.endOpLocked_done_mono hd }
  | pushDone p ok =>
    simp only [Send.step, Send.pushDone] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    exact Lifecycle.endOpLocked_done_mono (recordPushCompletion_done_mono pol ok _ hd)
  | cancel =>
    simp only [Send.step, Send.cancel, Option.some.injEq] at hs
    subst hs; exact Lifecycle.finishLocked_done_mono false hd
  | publish =>
    simp only [Send.step, Send.publish] at hs
    split at hs <;> try cases hs
    exact hd

theorem send_poolOps_pos_of_stage {ps : List SendPool} {p : Nat} {pool : SendPool}
    (hget : ps[p]? = some pool) (hop : 0 < pool.stage.opCount) :
    0 < Send.poolOps ps := by
  have ⟨_, hle⟩ := Send.poolOps_set pool.stage hget
  omega

theorem send_not_done_of_stage {r : Send} (h : Send.Inv .shipping r) {p : Nat} {pool : SendPool}
    (hget : r.pools[p]? = some pool) (hop : 0 < pool.stage.opCount) :
    r.life.done = false := by
  have hpo := send_poolOps_pos_of_stage hget hop
  have hacc := h.accounted; unfold Send.Accounted at hacc
  cases hd : r.life.done
  · rfl
  · have := h.life.done_idle hd; omega

theorem launchPass_length (canLaunch : Nat → Bool) :
    ∀ ps : List Pool, (Recv.launchPass canLaunch ps).1.length = ps.length
  | [] => rfl
  | _ :: ps => by
    simp only [Recv.launchPass]
    split <;> simp [launchPass_length canLaunch ps]

theorem launchPass_getElem? (canLaunch : Nat → Bool) :
    ∀ (ps : List Pool) (q : Nat) (pool' : Pool),
      (Recv.launchPass canLaunch ps).1[q]? = some pool' →
        ∃ pool, ps[q]? = some pool ∧ pool'.rank = pool.rank ∧
          (pool'.stage = .waiting ↔ pool.stage = .waiting) ∧
          (pool'.stage = .completed ↔ pool.stage = .completed)
  | [], _, _, h => by simp [Recv.launchPass] at h
  | p :: _, 0, pool', h => by
    simp only [Recv.launchPass] at h
    split at h
    · rename_i hc
      have hst : p.stage = .received := by
        simp only [Bool.and_eq_true, beq_iff_eq] at hc; exact hc.1
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h
      exact ⟨p, rfl, rfl, by simp [hst], by simp [hst]⟩
    · simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h
      exact ⟨p, rfl, rfl, Iff.rfl, Iff.rfl⟩
  | _ :: ps, q + 1, pool', h => by
    simp only [Recv.launchPass] at h
    split at h <;> {
      simp only [List.getElem?_cons_succ] at h
      exact launchPass_getElem? canLaunch ps q pool' h
    }

@[simp] theorem launchEligible_pools_length (r : Recv) :
    (Recv.launchEligible r).pools.length = r.pools.length := by
  unfold Recv.launchEligible
  split
  · rfl
  · exact launchPass_length _ r.pools

@[simp] theorem launchEligible_done (r : Recv) :
    (Recv.launchEligible r).life.done = r.life.done := by
  unfold Recv.launchEligible
  split <;> rfl

@[simp] theorem launchEligible_pushes (r : Recv) :
    (Recv.launchEligible r).pushes = r.pushes := by
  unfold Recv.launchEligible
  split <;> rfl

theorem launchEligible_getElem? {r : Recv} {q : Nat} {pool' : Pool}
    (h : (Recv.launchEligible r).pools[q]? = some pool') :
    ∃ pool, r.pools[q]? = some pool ∧ pool'.rank = pool.rank ∧
      (pool'.stage = .waiting ↔ pool.stage = .waiting) ∧
      (pool'.stage = .completed ↔ pool.stage = .completed) := by
  unfold Recv.launchEligible at h
  split at h
  · exact ⟨pool', h, rfl, Iff.rfl, Iff.rfl⟩
  · exact launchPass_getElem? _ r.pools q pool' h

theorem recv_step_pools_length {pol : PushEndPolicy} {r r' : Recv} {e : Recv.Ev}
    (hs : Recv.step pol r e = some r') : r'.pools.length = r.pools.length := by
  cases e with
  | pushBegin =>
    simp only [Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hs
    obtain ⟨_, _, rfl⟩ := hs; rfl
  | pushEnd ok =>
    simp only [Recv.step, Recv.pushEnd] at hs
    split at hs <;> try cases hs
    simp
  | onPoolReceived p =>
    simp only [Recv.step, Recv.onPoolReceived] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    simp
  | h2dIssue p ok =>
    simp only [Recv.step, Recv.h2dIssue] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · cases hs; simp
    · split at hs <;> { cases hs; simp }
  | h2dReady p =>
    simp only [Recv.step, Recv.h2dReady] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    simp
  | h2dDone p ok =>
    simp only [Recv.step, Recv.h2dDone] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · cases hs; simp
    · split at hs
      · cases hs; simp
      · split at hs
        · cases hs; simp
        · split at hs <;> { cases hs; simp }
  | pollReady =>
    simp only [Recv.step, Recv.pollReady] at hs
    split at hs <;> try cases hs
    rfl
  | cancel =>
    simp only [Recv.step, Recv.cancel, Option.some.injEq] at hs
    subst hs; rfl
  | publish =>
    simp only [Recv.step, Recv.publish] at hs
    split at hs <;> try cases hs
    rfl

theorem recv_step_done_mono {pol : PushEndPolicy} {r r' : Recv} {e : Recv.Ev}
    (hs : Recv.step pol r e = some r') (hd : r.life.done = true) :
    r'.life.done = true := by
  cases e with
  | pushBegin =>
    simp only [Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hs
    obtain ⟨l, hb, _⟩ := hs
    have hact := Lifecycle.beginOp_active hb
    simp [hd] at hact
  | pushEnd ok =>
    simp only [Recv.step, Recv.pushEnd] at hs
    split at hs <;> try cases hs
    simp only [Recv.maybeUnregisterSettled_life]
    apply Lifecycle.endOpLocked_done_mono
    split <;> simp [hd, Lifecycle.finishLocked_done_mono]
  | onPoolReceived p =>
    simp only [Recv.step, Recv.onPoolReceived] at hs
    split at hs <;> try cases hs
    split at hs
    · cases hs
    · rename_i hact; exfalso; exact hact (Or.inl hd)
  | h2dIssue p ok =>
    simp only [Recv.step, Recv.h2dIssue] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    split at hs
    · cases hs; exact Lifecycle.endOpLocked_done_mono hd
    · rename_i hact; exfalso; exact hact (Or.inl hd)
  | h2dReady p =>
    simp only [Recv.step, Recv.h2dReady] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    exact hd
  | h2dDone p ok =>
    simp only [Recv.step, Recv.h2dDone, hd, ↓reduceIte] at hs
    split at hs <;> try cases hs
    split at hs <;> try cases hs
    simp [Lifecycle.endOpLocked_done_mono hd]
  | pollReady =>
    simp only [Recv.step, Recv.pollReady] at hs
    split at hs <;> try cases hs
    exact Lifecycle.finishLocked_done_mono true hd
  | cancel =>
    simp only [Recv.step, Recv.cancel, Option.some.injEq] at hs
    subst hs; exact Lifecycle.finishLocked_done_mono false hd
  | publish =>
    simp only [Recv.step, Recv.publish] at hs
    split at hs <;> try cases hs
    exact hd

theorem recv_not_done_of_ready {r : Recv} (h : Recv.Inv r) {p : Nat} {pool : Pool}
    (hget : r.pools[p]? = some pool) (hrd : pool.stage = .ready) :
    r.life.done = false := by
  have ⟨_, hpos⟩ := Recv.poolOps_set_drop hget (p' := { pool with stage := .aborted })
    (by simp [hrd, PoolStage.holdsOp]) rfl
  have hacc := h.accounted; unfold Recv.Accounted at hacc
  cases hd : r.life.done
  · rfl
  · have := h.life.done_idle hd; omega


end Pipeline

end TpuSyncVerify.Transfer.Reshard

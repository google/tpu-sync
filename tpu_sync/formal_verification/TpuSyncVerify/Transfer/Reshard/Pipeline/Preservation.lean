import TpuSyncVerify.Transfer.Reshard.Pipeline.Invariant

/-! # `Pipeline`: per-event invariant preservation and `reachable_safe`

Part of `TpuSyncVerify.Transfer.Reshard.Pipeline`. -/

namespace TpuSyncVerify.Transfer.Reshard

open TpuSyncVerify.Transfer (Lifecycle)

namespace Pipeline

/-! ### Per-event invariant preservation -/

theorem step_armRecv_inv {s s' : Pipeline} (h : Inv s) (hs : s.armRecv = some s') :
    Inv s' := by
  simp only [armRecv] at hs
  split at hs <;> try cases hs
  exact ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
    h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
    h.dstStg_ok, h.recv_stage_landed, h.done_intact,
    fun ha => by simp at ha, fun ha => by simp at ha⟩

theorem step_reclaim_inv {s s' : Pipeline} (h : Inv s) (hs : s.reclaim = some s') :
    Inv s' := by
  simp only [reclaim] at hs
  split at hs <;> try cases hs
  rename_i hc
  exact ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    by simp, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match,
    fun hr => by simp at hr, fun _ => hc.1,
    h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
    h.dstStg_ok, h.recv_stage_landed, h.done_intact,
    h.unarmed_send, h.unarmed_recv⟩

theorem step_reseatSrcStaging_inv {s s' : Pipeline} (h : Inv s)
    (hs : s.reseatSrcStaging = some s') : Inv s' := by
  simp only [reseatSrcStaging] at hs
  split at hs <;> try cases hs
  rename_i hst
  have hd : s.send.life.done = true := by
    rw [h.sendInv.life.staging] at hst; simpa using hst
  exact ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    h.len_srcHbm, by simp, h.len_wire, h.len_dstStg, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
    fun hnd => by simp [hd] at hnd, h.sent_bound, h.wire_ok, h.landed_le_sent,
    h.dstStg_ok, h.recv_stage_landed, h.done_intact,
    h.unarmed_send, h.unarmed_recv⟩

theorem step_reseatDstStaging_inv {s s' : Pipeline} (h : Inv s)
    (hs : s.reseatDstStaging = some s') : Inv s' := by
  simp only [reseatDstStaging] at hs
  split at hs <;> try cases hs
  rename_i hst
  have hd : s.recv.life.done = true := by
    rw [h.recvInv.life.staging] at hst; simpa using hst
  exact ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    h.len_srcHbm, h.len_srcStg, h.len_wire, by simp, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
    h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
    fun hnd => by simp [hd] at hnd, h.recv_stage_landed, h.done_intact,
    h.unarmed_send, h.unarmed_recv⟩

theorem step_landStream_inv {s s' : Pipeline} {p : Nat} (h : Inv s)
    (hs : s.landStream p = some s') : Inv s' := by
  simp only [landStream] at hs
  split at hs <;> try cases hs
  rename_i hpush
  split at hs <;> try cases hs
  rename_i k m c hlk hsm hwc
  split at hs <;> try cases hs
  rename_i hkm
  obtain ⟨hlt_km, _⟩ := hkm
  have hlen_l := h.len_landed
  have hlen_dst := h.len_dstStg
  have hp : p < s.numPools := by
    have := lt_length_of_getElem?_eq hlk
    omega
  have hsm_d : s.sentStreams.getD p 0 = m := by
    rw [List.getD_eq_getElem?_getD, hsm]; rfl
  have hlk_d : s.landedStreams.getD p 0 = k := by
    rw [List.getD_eq_getElem?_getD, hlk]; rfl
  have hc_kv : c = .kv p := by
    have hw := h.wire_ok p hp (by omega)
    rw [hwc] at hw; exact Option.some.inj hw
  subst hc_kv
  refine ⟨h.sendInv, h.recvInv, h.num_pos, h.len_peers, h.len_send, h.len_recv,
    h.len_srcHbm, h.len_srcStg, h.len_wire, by simp [h.len_dstStg], h.len_dstHbm,
    h.len_sent, by simp [h.len_landed], h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
    h.srcStg_ok, h.sent_bound, h.wire_ok, ?_, ?_, ?_, h.done_intact,
    h.unarmed_send, fun ha => by have := (h.unarmed_recv ha).1; omega⟩
  · intro q hq
    by_cases hqp : p = q
    · subst hqp
      rw [getD_set_self (by omega), hsm_d]
      omega
    · rw [getD_set_ne (by omega)]
      exact h.landed_le_sent q hq
  · intro hnd q hq hgt
    by_cases hqp : p = q
    · subst hqp
      exact List.getElem?_set_self (by omega)
    · rw [List.getElem?_set_ne hqp]
      rw [getD_set_ne (by omega)] at hgt
      exact h.dstStg_ok hnd q hq hgt
  · intro q hq pool hget hne
    have hprev := h.recv_stage_landed q hq pool hget hne
    by_cases hqp : p = q
    · subst hqp
      have hlt_s : p < s.send.pools.length := by rw [h.len_send]; exact hp
      have hsget : s.send.pools[p]? = some s.send.pools[p] := List.getElem?_eq_getElem hlt_s
      have hpe := h.peers_match p hp s.send.pools[p] hsget
      have hbnd := h.sent_bound p hp s.send.pools[p] hsget
      omega
    · rw [getD_set_ne (by omega)]
      exact hprev

theorem step_send_inv {s s' : Pipeline} {e : Send.Ev} (h : Inv s)
    (hs : s.sendStep e = some s') : Inv s' := by
  have hlen_s := h.len_send
  have hlen_srcStg := h.len_srcStg
  have hlen_w := h.len_wire
  have hlen_sent := h.len_sent
  rcases e with ⟨p, ok⟩ | _ | ⟨p, ok⟩ | ⟨p, ok⟩ | _ | _
  · simp only [sendStep] at hs
    split at hs <;> try cases hs
    rename_i hguard
    simp only [Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, rfl⟩ := hs
    have hsndInv := Send.step_inv h.sendInv hsnd
    have hpub_m : s.send.published ≠ none → snd.published ≠ none :=
      send_step_published_ne_none hsnd
    have hdone_m : snd.life.done = false → s.send.life.done = false := by
      intro hnd; cases hd : s.send.life.done
      · rfl
      · have := send_step_done_mono hsnd hd; simp [hnd] at this
    simp only [Send.step, Send.d2hIssue] at hsnd
    split at hsnd <;> try cases hsnd
    split at hsnd <;> try cases hsnd
    rename_i pool hget
    split at hsnd <;> try cases hsnd
    rename_i hpend
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hpend
    have hp_lt : p < s.send.pools.length := lt_length_of_getElem?_eq hget
    have hlen' : snd.pools.length = s.numPools := by
      split at hsnd <;> { cases hsnd; simp [h.len_send] }
    have hset : snd.pools = s.send.pools.set p { pool with stage := if ok then .d2h else .aborted } := by
      split at hsnd <;> { rename_i hok; cases hsnd; simp [hok] }
    refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, hlen', h.len_recv,
      h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
      h.len_sent, h.len_landed, ?_, h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
      ?_, ?_, h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
      h.done_intact, ?_, h.unarmed_recv⟩
    · intro q hq pool' hget'
      rw [hset] at hget'
      by_cases hpq : p = q
      · subst hpq
        rw [List.getElem?_set_self hp_lt] at hget'
        cases hget'; exact h.peers_match p hq pool hget
      · rw [List.getElem?_set_ne hpq] at hget'
        exact h.peers_match q hq pool' hget'
    · intro hnd q hq pool' hget' ⟨k, hk⟩
      rw [hset] at hget'
      by_cases hpq : p = q
      · subst hpq
        rw [List.getElem?_set_self hp_lt] at hget'
        cases hget'
        cases ok <;> simp at hk
      · rw [List.getElem?_set_ne hpq] at hget'
        exact h.srcStg_ok (hdone_m hnd) q hq pool' hget' ⟨k, hk⟩
    · intro q hq pool' hget'
      rw [hset] at hget'
      by_cases hpq : p = q
      · subst hpq
        rw [List.getElem?_set_self hp_lt] at hget'
        cases hget'
        have hb := h.sent_bound p hq pool hget
        cases ok <;> simp only [SendPool.remainingSlots, hpend, ↓reduceIte, Bool.false_eq_true] at hb ⊢ <;> omega
      · rw [List.getElem?_set_ne hpq] at hget'
        exact h.sent_bound q hq pool' hget'
    · intro ha q hq pool' hget'
      rw [hset] at hget'
      by_cases hpq : p = q
      · subst hpq
        rw [List.getElem?_set_self hp_lt] at hget'
        cases hget'
        rcases hguard with hok | harm
        · simp [hok]
        · rw [ha] at harm; cases harm
      · rw [List.getElem?_set_ne hpq] at hget'
        exact h.unarmed_send ha q hq pool' hget'
  · simp only [sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, rfl⟩ := hs
    have hsndInv := Send.step_inv h.sendInv hsnd
    have hpub_m : s.send.published ≠ none → snd.published ≠ none :=
      send_step_published_ne_none hsnd
    have hdone_m : snd.life.done = false → s.send.life.done = false := by
      intro hnd; cases hd : s.send.life.done
      · rfl
      · have := send_step_done_mono hsnd hd; simp [hnd] at this
    have hpools_eq : snd.pools = s.send.pools := by
      simp only [Send.step, Send.execFinish] at hsnd
      split at hsnd <;> simp at hsnd; subst hsnd; rfl
    refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by rw [hpools_eq]; exact h.len_send,
      h.len_recv, h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
      h.len_sent, h.len_landed, by rw [hpools_eq]; exact h.peers_match,
      h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
      fun hnd => by rw [hpools_eq]; exact h.srcStg_ok (hdone_m hnd),
      by rw [hpools_eq]; exact h.sent_bound,
      h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
      h.done_intact, fun ha => by rw [hpools_eq]; exact h.unarmed_send ha, h.unarmed_recv⟩
  · simp only [sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, hs'⟩ := hs
    have hsndInv := Send.step_inv h.sendInv hsnd
    have hpub_m : s.send.published ≠ none → snd.published ≠ none :=
      send_step_published_ne_none hsnd
    have hdone_m : snd.life.done = false → s.send.life.done = false := by
      intro hnd; cases hd : s.send.life.done
      · rfl
      · have := send_step_done_mono hsnd hd; simp [hnd] at this
    have hsnd_step := hsnd
    simp only [Send.step, Send.d2hDone] at hsnd_step
    split at hsnd_step <;> try cases hsnd_step
    rename_i pool hget
    split at hsnd_step <;> try cases hsnd_step
    rename_i hd2h
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hd2h
    have hp_lt : p < s.send.pools.length := lt_length_of_getElem?_eq hget
    have hp : p < s.numPools := by omega
    have hnd_s : s.send.life.done = false :=
      send_not_done_of_stage h.sendInv hget (by simp [hd2h, SendStage.opCount])
    have hrec_f : s.reclaimed = false := by
      cases hr : s.reclaimed
      · rfl
      · have hpub := h.reclaimed_pub hr
        cases hb : s.send.published with
        | none => exact absurd hb hpub
        | some b =>
          have hd := h.sendInv.published_done b hb
          simp [hnd_s] at hd
    have hsrc_good : s.srcHbm = good s.numPools := h.srcHbm_ok hrec_f
    have harm_t : s.armed = true := by
      cases ha : s.armed
      · rcases h.unarmed_send ha p hp pool hget with hst | hst <;> simp [hd2h] at hst
      · rfl
    have hpools : ∃ st', snd.pools = s.send.pools.set p { pool with stage := st' } ∧
        (ok = false → st' = .aborted) ∧
        (ok = true → st' = .aborted ∨ st' = .pushing pool.peers) := by
      split at hsnd_step
      · rename_i hnok
        cases hsnd_step
        exact ⟨.aborted, by simp, fun _ => rfl, fun hok => by subst hok; cases hnok⟩
      · rename_i hnok
        split at hsnd_step
        · cases hsnd_step
          exact ⟨.aborted, by simp, fun _ => rfl, fun _ => Or.inl rfl⟩
        · cases hsnd_step
          exact ⟨.pushing pool.peers, by simp, fun hok_f => by subst hok_f; simp at hnok, fun _ => Or.inr rfl⟩
    obtain ⟨st', hset, hst_f, hst_t⟩ := hpools
    cases ok with
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at hs'
      subst hs'
      have hst_ab : st' = .aborted := hst_f rfl
      subst hst_ab
      refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by simp [hset, h.len_send], h.len_recv,
        h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
        h.len_sent, h.len_landed, ?_, h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
        ?_, ?_, h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
        h.done_intact, fun ha => by simp [harm_t] at ha, h.unarmed_recv⟩
      · intro q hq pool' hget'
        rw [hset] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'; exact h.peers_match p hq pool hget
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.peers_match q hq pool' hget'
      · intro hnd q hq pool' hget' ⟨k, hk⟩
        rw [hset] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'; simp at hk
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.srcStg_ok (hdone_m hnd) q hq pool' hget' ⟨k, hk⟩
      · intro q hq pool' hget'
        rw [hset] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'
          have hb := h.sent_bound p hq pool hget
          simp only [SendPool.remainingSlots, hd2h] at hb ⊢; omega
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.sent_bound q hq pool' hget'
    | true =>
      simp only [↓reduceIte] at hs'
      subst hs'
      refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by simp [hset, h.len_send], h.len_recv,
        h.len_srcHbm, by simp [h.len_srcStg], h.len_wire, h.len_dstStg, h.len_dstHbm,
        h.len_sent, h.len_landed, ?_, h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
        ?_, ?_, h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
        h.done_intact, fun ha => by simp [harm_t] at ha, h.unarmed_recv⟩
      · intro q hq pool' hget'
        rw [hset] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'; exact h.peers_match p hq pool hget
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.peers_match q hq pool' hget'
      · intro hnd q hq pool' hget' ⟨k, hk⟩
        rw [hset] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self (by omega), hsrc_good, getD_good hp]
        · rw [List.getElem?_set_ne hpq] at hget' ⊢
          exact h.srcStg_ok (hdone_m hnd) q hq pool' hget' ⟨k, hk⟩
      · intro q hq pool' hget'
        rw [hset] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'
          have hb := h.sent_bound p hq pool hget
          rcases hst_t rfl with rfl | rfl <;>
            simp only [SendPool.remainingSlots, hd2h] at hb ⊢ <;> omega
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.sent_bound q hq pool' hget'
  · simp only [sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, hs'⟩ := hs
    have hsndInv := Send.step_inv h.sendInv hsnd
    have hpub_m : s.send.published ≠ none → snd.published ≠ none :=
      send_step_published_ne_none hsnd
    have hdone_m : snd.life.done = false → s.send.life.done = false := by
      intro hnd; cases hd : s.send.life.done
      · rfl
      · have := send_step_done_mono hsnd hd; simp [hnd] at this
    have hsnd_step := hsnd
    simp only [Send.step, Send.pushDone] at hsnd_step
    split at hsnd_step <;> try cases hsnd_step
    rename_i pool hget
    split at hsnd_step <;> try cases hsnd_step
    rename_i k hpush
    have hp_lt : p < s.send.pools.length := lt_length_of_getElem?_eq hget
    have hp : p < s.numPools := by omega
    have hnd_s : s.send.life.done = false :=
      send_not_done_of_stage h.sendInv hget (by simp [hpush, SendStage.opCount])
    have harm_t : s.armed = true := by
      cases ha : s.armed
      · rcases h.unarmed_send ha p hp pool hget with hst | hst <;> simp [hpush] at hst
      · rfl
    have hstg_kv : s.srcStaging.getD p .junk = .kv p := by
      have h1 := h.srcStg_ok hnd_s p hp pool hget ⟨k + 1, hpush⟩
      rw [List.getD_eq_getElem?_getD, h1]; rfl
    cases ok with
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at hs'
      subst hs'
      refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by simp [h.len_send], h.len_recv,
        h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
        h.len_sent, h.len_landed, ?_, h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
        ?_, ?_, h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
        h.done_intact, fun ha => by simp [harm_t] at ha, h.unarmed_recv⟩
      · intro q hq pool' hget'
        simp only [endOp_pools, recordPushCompletion_pools] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'; exact h.peers_match p hq pool hget
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.peers_match q hq pool' hget'
      · intro hnd q hq pool' hget' ⟨m, hm⟩
        simp only [endOp_pools, recordPushCompletion_pools] at hget'
        by_cases hpq : p = q
        · subst hpq
          exact h.srcStg_ok hnd_s p hq pool hget ⟨k + 1, hpush⟩
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.srcStg_ok hnd_s q hq pool' hget' ⟨m, hm⟩
      · intro q hq pool' hget'
        simp only [endOp_pools, recordPushCompletion_pools] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'
          have hb := h.sent_bound p hq pool hget
          simp only [SendPool.remainingSlots, hpush] at hb ⊢; omega
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.sent_bound q hq pool' hget'
    | true =>
      simp only [↓reduceIte] at hs'
      subst hs'
      refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by simp [h.len_send], h.len_recv,
        h.len_srcHbm, h.len_srcStg, by simp [h.len_wire], h.len_dstStg, h.len_dstHbm,
        by simp [h.len_sent], h.len_landed, ?_, h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
        ?_, ?_, ?_, ?_, h.dstStg_ok, h.recv_stage_landed,
        h.done_intact, fun ha => by simp [harm_t] at ha, h.unarmed_recv⟩
      · intro q hq pool' hget'
        simp only [endOp_pools, recordPushCompletion_pools] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'; exact h.peers_match p hq pool hget
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.peers_match q hq pool' hget'
      · intro hnd q hq pool' hget' ⟨m, hm⟩
        simp only [endOp_pools, recordPushCompletion_pools] at hget'
        by_cases hpq : p = q
        · subst hpq
          exact h.srcStg_ok hnd_s p hq pool hget ⟨k + 1, hpush⟩
        · rw [List.getElem?_set_ne hpq] at hget'
          exact h.srcStg_ok hnd_s q hq pool' hget' ⟨m, hm⟩
      · intro q hq pool' hget'
        simp only [endOp_pools, recordPushCompletion_pools] at hget'
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self hp_lt] at hget'
          cases hget'
          have hb := h.sent_bound p hq pool hget
          rw [getD_set_self (by omega)]
          simp only [SendPool.remainingSlots, hpush] at hb ⊢; omega
        · rw [List.getElem?_set_ne hpq] at hget'
          rw [getD_set_ne (by omega)]
          exact h.sent_bound q hq pool' hget'
      · intro q hq hgt
        by_cases hpq : p = q
        · subst hpq
          rw [List.getElem?_set_self (by omega), hstg_kv]
        · rw [List.getElem?_set_ne hpq]
          rw [getD_set_ne (by omega)] at hgt
          exact h.wire_ok q hq hgt
      · intro q hq
        have hle := h.landed_le_sent q hq
        dsimp only
        by_cases hpq : p = q
        · subst hpq
          rw [getD_set_self (by omega)]; omega
        · rw [getD_set_ne (by omega)]; exact hle
  · simp only [sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, rfl⟩ := hs
    have hsndInv := Send.step_inv h.sendInv hsnd
    have hpub_m : s.send.published ≠ none → snd.published ≠ none :=
      send_step_published_ne_none hsnd
    have hdone_m : snd.life.done = false → s.send.life.done = false := by
      intro hnd; cases hd : s.send.life.done
      · rfl
      · have := send_step_done_mono hsnd hd; simp [hnd] at this
    have hpools_eq : snd.pools = s.send.pools := by
      simp only [Send.step, Send.cancel] at hsnd
      cases hsnd; rfl
    refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by rw [hpools_eq]; exact h.len_send,
      h.len_recv, h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
      h.len_sent, h.len_landed, by rw [hpools_eq]; exact h.peers_match,
      h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
      fun hnd => by rw [hpools_eq]; exact h.srcStg_ok (hdone_m hnd),
      by rw [hpools_eq]; exact h.sent_bound,
      h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
      h.done_intact, fun ha => by rw [hpools_eq]; exact h.unarmed_send ha, h.unarmed_recv⟩
  · simp only [sendStep, Option.map_eq_some_iff] at hs
    obtain ⟨snd, hsnd, rfl⟩ := hs
    have hsndInv := Send.step_inv h.sendInv hsnd
    have hpub_m : s.send.published ≠ none → snd.published ≠ none :=
      send_step_published_ne_none hsnd
    have hdone_m : snd.life.done = false → s.send.life.done = false := by
      intro hnd; cases hd : s.send.life.done
      · rfl
      · have := send_step_done_mono hsnd hd; simp [hnd] at this
    have hpools_eq : snd.pools = s.send.pools := by
      simp only [Send.step, Send.publish] at hsnd
      split at hsnd <;> simp at hsnd; subst hsnd; rfl
    refine ⟨hsndInv, h.recvInv, h.num_pos, h.len_peers, by rw [hpools_eq]; exact h.len_send,
      h.len_recv, h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
      h.len_sent, h.len_landed, by rw [hpools_eq]; exact h.peers_match,
      h.srcHbm_ok, fun hr => hpub_m (h.reclaimed_pub hr),
      fun hnd => by rw [hpools_eq]; exact h.srcStg_ok (hdone_m hnd),
      by rw [hpools_eq]; exact h.sent_bound,
      h.wire_ok, h.landed_le_sent, h.dstStg_ok, h.recv_stage_landed,
      h.done_intact, fun ha => by rw [hpools_eq]; exact h.unarmed_send ha, h.unarmed_recv⟩

theorem h2dDone_pools_shape {r r' : Recv} {p : Nat} {ok : Bool}
    (hs : Recv.h2dDone p ok r = some r') :
    ∃ pool st, r.pools[p]? = some pool ∧ pool.stage = .ready ∧
      (st = .aborted ∨ (ok = true ∧ st = .completed)) ∧
      (r'.pools = r.pools.set p { pool with stage := st } ∨
       r'.pools = (Recv.launchEligible { r with pools := r.pools.set p { pool with stage := st } }).pools) := by
  simp only [Recv.h2dDone] at hs
  split at hs <;> try cases hs
  rename_i pool hget
  split at hs <;> try cases hs
  rename_i hrd
  simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hrd
  split at hs
  · cases hs
    exact ⟨pool, .aborted, hget, hrd, Or.inl rfl, Or.inl (by simp)⟩
  · split at hs
    · cases hs
      exact ⟨pool, .aborted, hget, hrd, Or.inl rfl, Or.inl (by simp)⟩
    · split at hs
      · cases hs
        exact ⟨pool, .aborted, hget, hrd, Or.inl rfl, Or.inl (by simp)⟩
      · rename_i hnot_nok
        have hok : ok = true := by cases ok <;> simp at hnot_nok ⊢
        split at hs
        · cases hs
          exact ⟨pool, .completed, hget, hrd, Or.inr ⟨hok, rfl⟩, Or.inl (by simp)⟩
        · cases hs
          exact ⟨pool, .completed, hget, hrd, Or.inr ⟨hok, rfl⟩, Or.inr (by simp)⟩

theorem h2dDone_getElem? {r r' : Recv} {p q : Nat} {ok : Bool} {pool' : Pool}
    (hs : Recv.h2dDone p ok r = some r') (hget' : r'.pools[q]? = some pool') :
    (∃ pool, r.pools[p]? = some pool ∧ pool.stage = .ready ∧ p = q ∧
      (pool'.stage = .completed → ok = true)) ∨
    (p ≠ q ∧ ∃ pool_q, r.pools[q]? = some pool_q ∧
      (pool'.stage = .waiting ↔ pool_q.stage = .waiting) ∧
      (pool'.stage = .completed ↔ pool_q.stage = .completed)) := by
  obtain ⟨pool, st, hget, hrd, hst, hshape⟩ := h2dDone_pools_shape hs
  have hp_lt : p < r.pools.length := lt_length_of_getElem?_eq hget
  have hmid : ∃ pool₁, (r.pools.set p { pool with stage := st })[q]? = some pool₁ ∧
      (pool'.stage = .waiting ↔ pool₁.stage = .waiting) ∧
      (pool'.stage = .completed ↔ pool₁.stage = .completed) := by
    rcases hshape with h1 | h2
    · rw [h1] at hget'
      exact ⟨pool', hget', Iff.rfl, Iff.rfl⟩
    · rw [h2] at hget'
      obtain ⟨pool₁, hget₁, _, hw, hc⟩ := launchEligible_getElem? hget'
      exact ⟨pool₁, hget₁, hw, hc⟩
  obtain ⟨pool₁, hget₁, hw, hc⟩ := hmid
  by_cases hpq : p = q
  · subst hpq
    rw [List.getElem?_set_self hp_lt, Option.some.injEq] at hget₁
    subst hget₁
    refine Or.inl ⟨pool, hget, hrd, rfl, fun hcomp => ?_⟩
    have hst_c : st = .completed := hc.mp hcomp
    rcases hst with rfl | ⟨hok, _⟩
    · cases hst_c
    · exact hok
  · rw [List.getElem?_set_ne hpq] at hget₁
    exact Or.inr ⟨hpq, pool₁, hget₁, hw, hc⟩

theorem recv_other_stage_pres {r rcv : Recv} {e : Recv.Ev}
    (hrcv : Recv.step .finishOnError r e = some rcv)
    (hne1 : e ≠ .pushBegin)
    (hne2 : ∀ p, e ≠ .onPoolReceived p)
    (hne3 : ∀ p ok, e ≠ .h2dDone p ok)
    (q : Nat) (pool' : Pool) (hget' : rcv.pools[q]? = some pool') :
    ∃ pool_q : Pool, r.pools[q]? = some pool_q ∧
      (pool'.stage = .waiting → pool_q.stage = .waiting) ∧
      (pool_q.stage = .waiting → pool'.stage = .waiting) ∧
      (pool'.stage = .completed → pool_q.stage = .completed) := by
  cases e with
  | pushBegin => exact absurd rfl hne1
  | onPoolReceived p => exact absurd rfl (hne2 p)
  | h2dDone p ok => exact absurd rfl (hne3 p ok)
  | pushEnd ok =>
    simp only [Recv.step, Recv.pushEnd] at hrcv
    split at hrcv <;> try cases hrcv
    simp only [Recv.maybeUnregisterSettled_pools] at hget'
    exact ⟨pool', hget', id, id, id⟩
  | h2dIssue p ok =>
    simp only [Recv.step, Recv.h2dIssue] at hrcv
    split at hrcv <;> try cases hrcv
    rename_i pool hget
    split at hrcv <;> try cases hrcv
    rename_i hq_st
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hq_st
    have hp_lt : p < r.pools.length := lt_length_of_getElem?_eq hget
    have hset : ∃ st', (st' = .aborted ∨ st' = .issued) ∧
        rcv.pools = r.pools.set p { pool with stage := st' } := by
      split at hrcv
      · cases hrcv; exact ⟨.aborted, Or.inl rfl, rfl⟩
      · split at hrcv
        · cases hrcv; exact ⟨.issued, Or.inr rfl, rfl⟩
        · cases hrcv; exact ⟨.aborted, Or.inl rfl, by simp⟩
    obtain ⟨st', hst', hpools'⟩ := hset
    rw [hpools'] at hget'
    by_cases hpq : p = q
    · subst hpq
      rw [List.getElem?_set_self hp_lt, Option.some.injEq] at hget'
      subst hget'
      refine ⟨pool, hget, ?_, fun hw => by simp [hq_st] at hw, ?_⟩ <;>
        { rcases hst' with rfl | rfl <;> simp }
    · rw [List.getElem?_set_ne hpq] at hget'
      exact ⟨pool', hget', id, id, id⟩
  | h2dReady p =>
    simp only [Recv.step, Recv.h2dReady] at hrcv
    split at hrcv <;> try cases hrcv
    rename_i pool hget
    split at hrcv <;> try cases hrcv
    rename_i his
    simp only [beq_iff_eq] at his
    have hp_lt : p < r.pools.length := lt_length_of_getElem?_eq hget
    by_cases hpq : p = q
    · subst hpq
      rw [List.getElem?_set_self hp_lt, Option.some.injEq] at hget'
      subst hget'
      exact ⟨pool, hget, by simp, fun hw => by simp [his] at hw, by simp⟩
    · rw [List.getElem?_set_ne hpq] at hget'
      exact ⟨pool', hget', id, id, id⟩
  | pollReady =>
    simp only [Recv.step, Recv.pollReady] at hrcv
    split at hrcv <;> try cases hrcv
    exact ⟨pool', hget', id, id, id⟩
  | cancel =>
    simp only [Recv.step, Recv.cancel, Option.some.injEq] at hrcv
    subst hrcv; exact ⟨pool', hget', id, id, id⟩
  | publish =>
    simp only [Recv.step, Recv.publish] at hrcv
    split at hrcv <;> try cases hrcv
    exact ⟨pool', hget', id, id, id⟩

theorem recv_other_pushes_zero {r rcv : Recv} {e : Recv.Ev}
    (hrcv : Recv.step .finishOnError r e = some rcv)
    (hne1 : e ≠ .pushBegin)
    (hne2 : ∀ p, e ≠ .onPoolReceived p)
    (hne3 : ∀ p ok, e ≠ .h2dDone p ok)
    (hp0 : r.pushes = 0) : rcv.pushes = 0 := by
  cases e with
  | pushBegin => exact absurd rfl hne1
  | onPoolReceived p => exact absurd rfl (hne2 p)
  | h2dDone p ok => exact absurd rfl (hne3 p ok)
  | pushEnd ok =>
    simp only [Recv.step, Recv.pushEnd, hp0, ↓reduceIte] at hrcv; cases hrcv
  | h2dIssue p ok =>
    simp only [Recv.step, Recv.h2dIssue] at hrcv
    split at hrcv <;> try cases hrcv
    split at hrcv <;> try cases hrcv
    split at hrcv
    · cases hrcv; exact hp0
    · split at hrcv <;> { cases hrcv; simpa using hp0 }
  | h2dReady p =>
    simp only [Recv.step, Recv.h2dReady] at hrcv
    split at hrcv <;> try cases hrcv
    split at hrcv <;> try cases hrcv
    exact hp0
  | pollReady =>
    simp only [Recv.step, Recv.pollReady] at hrcv
    split at hrcv <;> try cases hrcv
    exact hp0
  | cancel =>
    simp only [Recv.step, Recv.cancel, Option.some.injEq] at hrcv
    subst hrcv; exact hp0
  | publish =>
    simp only [Recv.step, Recv.publish] at hrcv
    split at hrcv <;> try cases hrcv
    exact hp0

theorem step_recv_other_inv {s : Pipeline} {rcv : Recv} {e : Recv.Ev} (h : Inv s)
    (hrcv : Recv.step .finishOnError s.recv e = some rcv)
    (hne1 : e ≠ .pushBegin)
    (hne2 : ∀ p, e ≠ .onPoolReceived p)
    (hne3 : ∀ p ok, e ≠ .h2dDone p ok) :
    Inv { s with recv := rcv } := by
  have hrcvInv := Recv.step_inv .finishOnError h.recvInv hrcv
  have hdone_m : rcv.life.done = false → s.recv.life.done = false := by
    intro hnd; cases hd : s.recv.life.done
    · rfl
    · have := recv_step_done_mono hrcv hd; simp [hnd] at this
  have hlen' : rcv.pools.length = s.numPools := by
    rw [recv_step_pools_length hrcv, h.len_recv]
  refine ⟨h.sendInv, hrcvInv, h.num_pos, h.len_peers, h.len_send, hlen',
    h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
    h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
    h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
    fun hnd => h.dstStg_ok (hdone_m hnd), ?_, ?_, h.unarmed_send, ?_⟩
  · intro q hq pool' hget' hne
    obtain ⟨pool_q, hget_q, _, hw_to, _⟩ := recv_other_stage_pres hrcv hne1 hne2 hne3 q pool' hget'
    exact h.recv_stage_landed q hq pool_q hget_q (fun hw_q => hne (hw_to hw_q))
  · intro q hq pool' hget' hcomp'
    obtain ⟨pool_q, hget_q, _, _, hc⟩ := recv_other_stage_pres hrcv hne1 hne2 hne3 q pool' hget'
    exact h.done_intact q hq pool_q hget_q (hc hcomp')
  · intro ha
    have ⟨hp0, hl0, hw_all⟩ := h.unarmed_recv ha
    refine ⟨recv_other_pushes_zero hrcv hne1 hne2 hne3 hp0, hl0, fun q hq pool' hget' => ?_⟩
    obtain ⟨pool_q, hget_q, _, hw_to, _⟩ := recv_other_stage_pres hrcv hne1 hne2 hne3 q pool' hget'
    exact hw_to (hw_all q hq pool_q hget_q)

theorem step_recv_inv {s s' : Pipeline} {e : Recv.Ev} (h : Inv s)
    (hs : s.recvStep e = some s') : Inv s' := by
  have hlen_r := h.len_recv
  cases e with
  | pushBegin =>
    simp only [recvStep] at hs
    split at hs <;> try cases hs
    rename_i harm
    simp only [Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    have hrcvInv := Recv.step_inv .finishOnError h.recvInv hrcv
    have hdone_m : rcv.life.done = false → s.recv.life.done = false := by
      intro hnd; cases hd : s.recv.life.done
      · rfl
      · have := recv_step_done_mono hrcv hd; simp [hnd] at this
    have hpools_eq : rcv.pools = s.recv.pools := by
      simp only [Recv.step, Recv.pushBegin, Option.map_eq_some_iff] at hrcv
      obtain ⟨_, _, rfl⟩ := hrcv; rfl
    exact ⟨h.sendInv, hrcvInv, h.num_pos, h.len_peers, h.len_send,
      by rw [hpools_eq]; exact h.len_recv,
      h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
      h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
      h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
      fun hnd => h.dstStg_ok (hdone_m hnd),
      by rw [hpools_eq]; exact h.recv_stage_landed,
      fun q hq pool hget => by rw [hpools_eq] at hget; exact h.done_intact q hq pool hget,
      h.unarmed_send, fun ha => by simp [harm] at ha⟩
  | onPoolReceived p =>
    simp only [recvStep] at hs
    split at hs <;> try cases hs
    rename_i hland_p
    simp only [Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    have hrcvInv := Recv.step_inv .finishOnError h.recvInv hrcv
    have hdone_m : rcv.life.done = false → s.recv.life.done = false := by
      intro hnd; cases hd : s.recv.life.done
      · rfl
      · have := recv_step_done_mono hrcv hd; simp [hnd] at this
    have hlen' : rcv.pools.length = s.numPools := by
      rw [recv_step_pools_length hrcv, h.len_recv]
    simp only [Recv.step, Recv.onPoolReceived] at hrcv
    split at hrcv <;> try cases hrcv
    rename_i pool hget
    split at hrcv <;> try cases hrcv
    rename_i hact
    simp only [not_or, Bool.not_eq_true, bne_iff_ne, ne_eq, Decidable.not_not] at hact
    obtain ⟨_, hwait⟩ := hact
    have hp_lt : p < s.recv.pools.length := lt_length_of_getElem?_eq hget
    have hp : p < s.numPools := by omega
    have hland_d : s.landedStreams.getD p 0 = s.peerCounts.getD p 0 := by
      rw [List.getD_eq_getElem?_getD, hland_p]; rfl
    have harm_t : s.armed = true := by
      have hpe_pos := peerCount_pos h hp
      cases ha : s.armed
      · have hl0 := (h.unarmed_recv ha).2.1 p hp
        omega
      · rfl
    refine ⟨h.sendInv, hrcvInv, h.num_pos, h.len_peers, h.len_send, hlen',
      h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
      h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
      h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
      fun hnd => h.dstStg_ok (hdone_m hnd), ?_, ?_,
      h.unarmed_send, ?_⟩
    · intro q hq pool' hget' hne
      obtain ⟨pool₁, hget₁, _, hw, _⟩ := launchEligible_getElem? hget'
      by_cases hpq : p = q
      · subst hpq; exact hland_d
      · rw [List.getElem?_set_ne hpq] at hget₁
        exact h.recv_stage_landed q hq pool₁ hget₁ (fun hw₁ => hne (hw.mpr hw₁))
    · intro q hq pool' hget' hcomp'
      obtain ⟨pool₁, hget₁, _, _, hc⟩ := launchEligible_getElem? hget'
      have hcomp₁ : pool₁.stage = .completed := hc.mp hcomp'
      by_cases hpq : p = q
      · subst hpq
        rw [List.getElem?_set_self hp_lt, Option.some.injEq] at hget₁
        subst hget₁; cases hcomp₁
      · rw [List.getElem?_set_ne hpq] at hget₁
        exact h.done_intact q hq pool₁ hget₁ hcomp₁
    · intro ha; simp [harm_t] at ha
  | h2dDone p ok =>
    simp only [recvStep] at hs
    split at hs <;> try cases hs
    rename_i pool hget
    simp only [Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, hs'⟩ := hs
    have hrcvInv := Recv.step_inv .finishOnError h.recvInv hrcv
    have hdone_m : rcv.life.done = false → s.recv.life.done = false := by
      intro hnd; cases hd : s.recv.life.done
      · rfl
      · have := recv_step_done_mono hrcv hd; simp [hnd] at this
    have hlen' : rcv.pools.length = s.numPools := by
      rw [recv_step_pools_length hrcv, h.len_recv]
    have hrcv_step : Recv.h2dDone p ok s.recv = some rcv := hrcv
    obtain ⟨pool₀, st, hget₀, hrd, _, _⟩ := h2dDone_pools_shape hrcv_step
    rw [hget] at hget₀; injection hget₀ with hpool_eq; subst hpool_eq
    have hp_lt : p < s.recv.pools.length := lt_length_of_getElem?_eq hget
    have hp : p < s.numPools := by omega
    have hnd_r : s.recv.life.done = false := recv_not_done_of_ready h.recvInv hget hrd
    have harm_t : s.armed = true := by
      cases ha : s.armed
      · have hw := (h.unarmed_recv ha).2.2 p hp pool hget
        simp [hrd] at hw
      · rfl
    cases ok with
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at hs'
      subst hs'
      refine ⟨h.sendInv, hrcvInv, h.num_pos, h.len_peers, h.len_send, hlen',
        h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg, h.len_dstHbm,
        h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
        h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
        fun hnd => h.dstStg_ok (hdone_m hnd), ?_, ?_,
        h.unarmed_send, fun ha => by simp [harm_t] at ha⟩
      · intro q hq pool' hget' hne
        rcases h2dDone_getElem? hrcv_step hget' with ⟨_, _, _, rfl, _⟩ | ⟨_, pool_q, hget_q, hw, _⟩
        · exact h.recv_stage_landed p hq pool hget (by simp [hrd])
        · exact h.recv_stage_landed q hq pool_q hget_q (fun hw_q => hne (hw.mpr hw_q))
      · intro q hq pool' hget' hcomp'
        rcases h2dDone_getElem? hrcv_step hget' with ⟨_, _, _, rfl, hok⟩ | ⟨_, pool_q, hget_q, _, hc⟩
        · cases hok hcomp'
        · exact h.done_intact q hq pool_q hget_q (hc.mp hcomp')
    | true =>
      simp only [↓reduceIte] at hs'
      subst hs'
      have hland_p : s.landedStreams.getD p 0 = s.peerCounts.getD p 0 :=
        h.recv_stage_landed p hp pool hget (by simp [hrd])
      have hpe_pos : 0 < s.peerCounts.getD p 0 := peerCount_pos h hp
      have hstg_kv : s.dstStaging.getD p .junk = .kv p := by
        have h1 := h.dstStg_ok hnd_r p hp (by omega)
        rw [List.getD_eq_getElem?_getD, h1]; rfl
      refine ⟨h.sendInv, hrcvInv, h.num_pos, h.len_peers, h.len_send, hlen',
        h.len_srcHbm, h.len_srcStg, h.len_wire, h.len_dstStg,
        by simp [h.len_recv, h.len_dstHbm],
        h.len_sent, h.len_landed, h.peers_match, h.srcHbm_ok, h.reclaimed_pub,
        h.srcStg_ok, h.sent_bound, h.wire_ok, h.landed_le_sent,
        fun hnd => h.dstStg_ok (hdone_m hnd), ?_, ?_,
        h.unarmed_send, fun ha => by simp [harm_t] at ha⟩
      · intro q hq pool' hget' hne
        rcases h2dDone_getElem? hrcv_step hget' with ⟨_, _, _, rfl, _⟩ | ⟨_, pool_q, hget_q, hw, _⟩
        · exact hland_p
        · exact h.recv_stage_landed q hq pool_q hget_q (fun hw_q => hne (hw.mpr hw_q))
      · intro q hq pool' hget' hcomp'
        rcases h2dDone_getElem? hrcv_step hget' with ⟨_, _, _, rfl, _⟩ | ⟨hpq, pool_q, hget_q, _, hc⟩
        · rw [getElem?_writePoolHbm_self (by simp [h.len_recv, h.len_dstHbm, hp]), hstg_kv]
        · have hcomp_q : pool_q.stage = .completed := hc.mp hcomp'
          have hnlt : ¬(pool.rank < pool_q.rank) := by
            intro hlt_r
            have hrw := Recv.rankWaveOrder_iff.mp h.recvInv.rankWave
              pool_q (List.mem_of_getElem? hget_q) pool (List.mem_of_getElem? hget)
              (by simp [hcomp_q, PoolStage.launched]) hlt_r
            simp [hrd] at hrw
          rw [getElem?_writePoolHbm_of_ne_of_not_lt hpq hget_q hnlt]
          exact h.done_intact q hq pool_q hget_q hcomp_q
  | pushEnd ok =>
    simp only [recvStep, Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    exact step_recv_other_inv h hrcv (fun h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)
  | h2dIssue p ok =>
    simp only [recvStep, Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    exact step_recv_other_inv h hrcv (fun h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)
  | h2dReady p =>
    simp only [recvStep, Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    exact step_recv_other_inv h hrcv (fun h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)
  | pollReady =>
    simp only [recvStep, Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    exact step_recv_other_inv h hrcv (fun h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)
  | cancel =>
    simp only [recvStep, Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    exact step_recv_other_inv h hrcv (fun h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)
  | publish =>
    simp only [recvStep, Option.map_eq_some_iff] at hs
    obtain ⟨rcv, hrcv, rfl⟩ := hs
    exact step_recv_other_inv h hrcv (fun h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)

theorem step_inv {s s' : Pipeline} {e : Ev} (h : Inv s) (hs : step s e = some s') :
    Inv s' := by
  cases e with
  | armRecv => exact step_armRecv_inv h hs
  | send e => exact step_send_inv h hs
  | recv e => exact step_recv_inv h hs
  | landStream p => exact step_landStream_inv h hs
  | reclaim => exact step_reclaim_inv h hs
  | reseatSrcStaging => exact step_reseatSrcStaging_inv h hs
  | reseatDstStaging => exact step_reseatDstStaging_inv h hs

theorem reachable_inv {ranks peerCounts : List Nat} {s : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (sys ranks peerCounts).Reachable s) : Inv s :=
  (sys ranks peerCounts).reachable_induction (inv_init hvalid) (fun _ _ _ => step_inv) h

theorem reachable_unarmed_inv {ranks peerCounts : List Nat} {s : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (sysUnarmed ranks peerCounts).Reachable s) : Inv s :=
  (sysUnarmed ranks peerCounts).reachable_induction (inv_initUnarmed hvalid)
    (fun _ _ _ => step_inv) h

/-- Main safety theorem: every reachable state of the composed reshard pipeline
on a valid schedule satisfies `PublicationCorrect`, `AliasedPageCorrect`,
`DstHbmSafe`, `SrcHbmSafe`, `StagingSafe`, `StreamAccountingSafe`,
`NoUnarmedPush`, and `NoOpLeak`. -/
theorem reachable_safe {ranks peerCounts : List Nat} {s : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (sys ranks peerCounts).Reachable s) : Safe s :=
  inv_safe (reachable_inv hvalid h)

/-- Every state reachable from the pre-Phase-1 unarmed state (`initUnarmed`)
also satisfies all pipeline safety properties. -/
theorem reachable_unarmed_safe {ranks peerCounts : List Nat} {s : Pipeline}
    (hvalid : validConfig ranks peerCounts = true)
    (h : (sysUnarmed ranks peerCounts).Reachable s) : Safe s :=
  inv_safe (reachable_unarmed_inv hvalid h)


end Pipeline

end TpuSyncVerify.Transfer.Reshard

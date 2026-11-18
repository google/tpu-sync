import TpuSyncVerify.Transfer.PrefillDecode.UuidTable.Model

/-! # `UuidTable`: inductive invariant and drain-before-reuse theorems

Part of `TpuSyncVerify.Transfer.PrefillDecode.UuidTable`; the module docstring and
source map live in `UuidTable.lean`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.UuidTable

open TpuSyncVerify.Transfer (Lifecycle)
open TpuSyncVerify.Transfer.PrefillDecode (Recv Send)

/-! ## Helper lemmas on `activeStaging` and `Recv.step` -/

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

theorem activeStaging_replicate_none : ∀ n, activeStaging (List.replicate n none) = 0
  | 0 => rfl
  | n + 1 => by simp [List.replicate_succ, activeStaging, slotHasStaging, activeStaging_replicate_none n]

theorem activeStaging_set_from_false :
    ∀ {os : List (Option RecvEntry)} {uuid : Nat} {o o' : Option RecvEntry},
      os[uuid]? = some o →
      slotHasStaging o = false →
      activeStaging (os.set uuid o') =
        activeStaging os + (if slotHasStaging o' then 1 else 0)
  | [], _, _, _, h, _ => by simp at h
  | a :: as, 0, o, o', h, hfalse => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp [activeStaging, hfalse]; omega
  | a :: as, uuid + 1, o, o', h, hfalse => by
    simp only [List.getElem?_cons_succ] at h
    have ih := activeStaging_set_from_false (o' := o') h hfalse
    simp only [List.set_cons_succ, activeStaging, ih]; omega

theorem activeStaging_set_step :
    ∀ {os : List (Option RecvEntry)} {uuid : Nat} {e e' : RecvEntry},
      os[uuid]? = some (some e) →
      (e.recv.life.hasStaging = false → e'.recv.life.hasStaging = false) →
      activeStaging (os.set uuid (some e')) +
        (if e.recv.life.hasStaging && !e'.recv.life.hasStaging then 1 else 0) =
        activeStaging os
  | [], _, _, _, h, _ => by simp at h
  | a :: as, 0, e, e', h, hmono => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    cases h1 : e.recv.life.hasStaging <;> cases h2 : e'.recv.life.hasStaging
    · simp [activeStaging, slotHasStaging, h1, h2]
    · have := hmono h1; simp [h2] at this
    · simp [activeStaging, slotHasStaging, h1, h2]; omega
    · simp [activeStaging, slotHasStaging, h1, h2]
  | a :: as, uuid + 1, e, e', h, hmono => by
    simp only [List.getElem?_cons_succ] at h
    have ih := activeStaging_set_step h hmono
    simp only [List.set_cons_succ, activeStaging]; omega

/-! ## Inductive invariant and resource conservation -/

structure Inv (s : State) : Prop where
  slots : s.freeSlots + activeStaging s.recvTable = s.cfg.numSlots
  recv_inv : ∀ e : RecvEntry, some e ∈ s.recvTable → Recv.Inv e.recv
  send_inv : ∀ e : SendEntry, some e ∈ s.sendTable → Send.Inv e.send

theorem inv_init (cfg : Config) : Inv (init cfg) := by
  refine ⟨by simp [init, activeStaging_replicate_none], ?_, ?_⟩
  · intro e he
    simp [init] at he
  · intro e he
    simp [init] at he

theorem step_inv {s s' : State} {ev : Ev} (h : Inv s) (hs : step s ev = some s') :
    Inv s' := by
  rcases h with ⟨hslots, hrecv, hsend⟩
  cases ev with
  | startRead reqId uuid =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i inc hget
      split at hs
      · split at hs <;> cases hs <;> exact ⟨hslots, hrecv, hsend⟩
      · rename_i hnd
        have hdone : inc.recv.life.done = true := by
          cases hd : inc.recv.life.done <;> simp_all
        have hinc_inv := hrecv inc (getElem?_mem hget)
        have hfalse : slotHasStaging (some inc) = false := by
          simp [slotHasStaging, hinc_inv.life.staging, hdone]
        split at hs
        · rename_i hfree
          cases hs
          refine ⟨?_, ?_, hsend⟩
          · dsimp only
            rw [activeStaging_set_from_false hget hfalse]
            simp [slotHasStaging, Recv.initLoad]
            omega
          · intro e he
            rcases mem_set_cases he with h_eq | he
            · cases h_eq
              exact Recv.inv_initLoad s.cfg.numLayers
            · exact hrecv e he
        · cases hs
          refine ⟨?_, ?_, hsend⟩
          · dsimp only
            rw [activeStaging_set_from_false hget hfalse]
            simp [slotHasStaging]
            omega
          · intro e he
            rcases mem_set_cases he with h_eq | he
            · cases h_eq
            · exact hrecv e he
    · rename_i hget
      -- `sessions_[uuid]` holds no receive: a send offer may hold it.
      split at hs
      · cases hs
      · rename_i snd hsget
        split at hs
        · cases hs
          exact ⟨hslots, hrecv, hsend⟩
        · split at hs
          · rename_i hfree
            cases hs
            refine ⟨?_, ?_, ?_⟩
            · dsimp only
              rw [activeStaging_set_from_false hget rfl]
              simp [slotHasStaging, Recv.initLoad]
              omega
            · intro e he
              rcases mem_set_cases he with h_eq | he
              · cases h_eq
                exact Recv.inv_initLoad s.cfg.numLayers
              · exact hrecv e he
            · intro e he
              rcases mem_set_cases he with h_eq | he
              · cases h_eq
              · exact hsend e he
          · cases hs
            refine ⟨hslots, hrecv, ?_⟩
            intro e he
            rcases mem_set_cases he with h_eq | he
            · cases h_eq
            · exact hsend e he
      · split at hs
        · rename_i hfree
          cases hs
          refine ⟨?_, ?_, hsend⟩
          · dsimp only
            rw [activeStaging_set_from_false hget rfl]
            simp [slotHasStaging, Recv.initLoad]
            omega
          · intro e he
            rcases mem_set_cases he with h_eq | he
            · cases h_eq
              exact Recv.inv_initLoad s.cfg.numLayers
            · exact hrecv e he
        · cases hs
          exact ⟨hslots, hrecv, hsend⟩
  | registerRecv reqId uuid =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i inc hget
      have hinc_inv := hrecv inc (getElem?_mem hget)
      have hlt : uuid < s.recvTable.length := lt_length_of_getElem?_eq hget
      split at hs
      · -- `recv->Finish()` on a ready incumbent.
        rename_i r hfin
        have hfin' : Recv.step inc.recv .pollReady = some r := hfin
        have hr_inv : Recv.Inv r := Recv.step_inv hinc_inv hfin'
        have hmono := recv_step_hasStaging_false hfin'
        have hst : activeStaging (s.recvTable.set uuid (some { inc with recv := r })) + freedBy inc r =
            activeStaging s.recvTable :=
          activeStaging_set_step (e' := { inc with recv := r }) hget hmono
        split at hs
        · cases hs
          refine ⟨?_, ?_, hsend⟩
          · dsimp only
            omega
          · intro e he
            rcases mem_set_cases he with h_eq | he
            · cases h_eq
              exact hr_inv
            · exact hrecv e he
        · split at hs
          · rename_i hndone hfree
            cases hs
            have hrdone : r.life.done = true := by
              cases hd : r.life.done <;> simp_all
            have hfalse : slotHasStaging (some { inc with recv := r }) = false := by
              simp [slotHasStaging, hr_inv.life.staging, hrdone]
            have hst2 := activeStaging_set_from_false
              (os := s.recvTable.set uuid (some { inc with recv := r }))
              (o' := some { reqId, gen := s.nextGen, recv := Recv.initPush s.cfg.numLayers })
              (List.getElem?_set_self hlt) hfalse
            refine ⟨?_, ?_, hsend⟩
            · dsimp only
              rw [List.set_set] at hst2
              rw [hst2]
              simp [slotHasStaging, Recv.initPush]
              omega
            · intro e he
              rcases mem_set_cases he with h_eq | he
              · cases h_eq
                exact Recv.inv_initPush s.cfg.numLayers
              · exact hrecv e he
          · cases hs
      · split at hs
        · cases hs
        · rename_i hnd
          have hdone : inc.recv.life.done = true := by
            cases hd : inc.recv.life.done <;> simp_all
          have hfalse : slotHasStaging (some inc) = false := by
            simp [slotHasStaging, hinc_inv.life.staging, hdone]
          split at hs
          · rename_i hfree
            cases hs
            refine ⟨?_, ?_, hsend⟩
            · dsimp only
              rw [activeStaging_set_from_false hget hfalse]
              simp [slotHasStaging, Recv.initPush]
              omega
            · intro e he
              rcases mem_set_cases he with h_eq | he
              · cases h_eq
                exact Recv.inv_initPush s.cfg.numLayers
              · exact hrecv e he
          · cases hs
    · rename_i hget
      split at hs
      · cases hs
      · rename_i snd hsget
        split at hs
        · cases hs
        · split at hs
          · rename_i hfree
            cases hs
            refine ⟨?_, ?_, ?_⟩
            · dsimp only
              rw [activeStaging_set_from_false hget rfl]
              simp [slotHasStaging, Recv.initPush]
              omega
            · intro e he
              rcases mem_set_cases he with h_eq | he
              · cases h_eq
                exact Recv.inv_initPush s.cfg.numLayers
              · exact hrecv e he
            · intro e he
              rcases mem_set_cases he with h_eq | he
              · cases h_eq
              · exact hsend e he
          · cases hs
      · split at hs
        · rename_i hfree
          cases hs
          refine ⟨?_, ?_, hsend⟩
          · dsimp only
            rw [activeStaging_set_from_false hget rfl]
            simp [slotHasStaging, Recv.initPush]
            omega
          · intro e he
            rcases mem_set_cases he with h_eq | he
            · cases h_eq
              exact Recv.inv_initPush s.cfg.numLayers
            · exact hrecv e he
        · cases hs
  | recvStep uuid e =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · rename_i entry hget
      split at hs
      · cases hs
      · rename_i recv' hstep
        cases hs
        have hst := activeStaging_set_step (e' := { entry with recv := recv' }) hget
          (recv_step_hasStaging_false hstep)
        refine ⟨?_, ?_, hsend⟩
        · dsimp only
          split <;> rename_i hrel <;> simp [hrel] at hst <;> omega
        · intro e' he'
          rcases mem_set_cases he' with h_eq | he'
          · cases h_eq
            exact Recv.step_inv (hrecv entry (getElem?_mem hget)) hstep
          · exact hrecv e' he'
  | sweepRecv uuid =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · rename_i entry hget
      split at hs
      · rename_i hdone
        cases hs
        have hentry_inv := hrecv entry (getElem?_mem hget)
        have hfalse : slotHasStaging (some entry) = false := by
          simp [slotHasStaging, hentry_inv.life.staging, hdone]
        refine ⟨?_, ?_, hsend⟩
        · dsimp only
          rw [activeStaging_set_from_false hget hfalse]
          simp [slotHasStaging]
          omega
        · intro e he
          rcases mem_set_cases he with h_eq | he
          · cases h_eq
          · exact hrecv e he
      · cases hs
  | notifyForRead reqId uuid =>
    simp only [step] at hs
    split at hs
    · rename_i hsget hrget
      cases hs
      refine ⟨hslots, hrecv, ?_⟩
      intro e he
      rcases mem_set_cases he with h_eq | he
      · cases h_eq
        exact Send.inv_init s.cfg.numLayers
      · exact hsend e he
    · cases hs
  | sendStep uuid e =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · rename_i entry hget
      split at hs
      · cases hs
      · rename_i send' hstep
        cases hs
        refine ⟨hslots, hrecv, ?_⟩
        intro e' he'
        rcases mem_set_cases he' with h_eq | he'
        · cases h_eq
          exact Send.step_inv (hsend entry (getElem?_mem hget)) hstep
        · exact hsend e' he'
  | sweepSend uuid =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · rename_i entry hget
      split at hs
      · cases hs
        refine ⟨hslots, hrecv, ?_⟩
        intro e he
        rcases mem_set_cases he with h_eq | he
        · cases h_eq
        · exact hsend e he
      · cases hs

theorem reachable_inv {cfg : Config} {s : State} (h : (sys cfg).Reachable s) : Inv s :=
  (sys cfg).reachable_induction (inv_init cfg) (fun _ _ _ ih hs => step_inv ih hs) h

/-- Every active receive entry in any reachable state satisfies `Recv.Safe`. -/
theorem reachable_recv_safe {cfg : Config} {s : State}
    (h : (sys cfg).Reachable s) {e : RecvEntry} (he : some e ∈ s.recvTable) :
    Recv.Safe e.recv :=
  Recv.inv_safe ((reachable_inv h).recv_inv e he)

/-- Every active send entry in any reachable state satisfies `Send.Safe`. -/
theorem reachable_send_safe {cfg : Config} {s : State}
    (h : (sys cfg).Reachable s) {e : SendEntry} (he : some e ∈ s.sendTable) :
    Send.Safe e.send :=
  Send.inv_safe ((reachable_inv h).send_inv e he)

/-! ## Drain-before-reuse persistence theorems -/

/-- **Consumer drain-before-reuse:** while an entry `inc` at `s.recvTable[uuid]`
has `inc.recv.life.done = false` and either an operation in flight
(`0 < inc.recv.life.inFlight`, including after a deadline has marked
`draining = true`) or nothing ready for the poll/registration finish check
(`¬ Recv.isReadyToComplete inc.recv`), **every** enabled system step preserves
the incumbent's ownership of `uuid` with the exact same `(reqId, gen)`. Neither
`startRead`, `registerRecv`, nor `sweepRecv` can overwrite or erase `inc` until
it has settled (`done = true`). -/
theorem active_recv_preserved {s s' : State} {ev : Ev} {uuid : Uuid} {inc : RecvEntry}
    (hget : s.recvTable[uuid]? = some (some inc))
    (hnd : inc.recv.life.done = false)
    (hkeep : 0 < inc.recv.life.inFlight ∨ ¬ Recv.isReadyToComplete inc.recv)
    (hs : step s ev = some s') :
    ∃ recv', s'.recvTable[uuid]? =
      some (some { reqId := inc.reqId, gen := inc.gen, recv := recv' }) := by
  have hlt : uuid < s.recvTable.length := lt_length_of_getElem?_eq hget
  cases ev with
  | startRead reqId u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      simp only [hget, hnd, Bool.not_false, ↓reduceIte] at hs
      split at hs <;> cases hs <;> exact ⟨inc.recv, hget⟩
    · repeat' split at hs
      all_goals ((cases hs) <;> first
        | exact ⟨inc.recv, hget⟩
        | exact ⟨inc.recv, by rw [List.getElem?_set_ne hu]; exact hget⟩)
  | registerRecv reqId u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      simp only [hget] at hs
      split at hs
      · rename_i r hfin
        split at hs
        · cases hs
          exact ⟨r, List.getElem?_set_self hlt⟩
        · -- The finish cannot settle an incumbent with work in flight, and a
          -- not-ready incumbent is not finished at all.
          exfalso
          rename_i hrdone
          simp only [Recv.pollReady] at hfin
          split at hfin
          · rename_i hcond
            cases hfin
            rcases hkeep with hif | hnr
            · have := Lifecycle.finishLocked_done_of_inFlight (l := inc.recv.life) true
                (by omega)
              simp_all
            · exact hnr hcond.2
          · cases hfin
      · split at hs
        · cases hs
        · simp [hnd] at *
    · repeat' split at hs
      all_goals ((cases hs) <;> first
        | exact ⟨inc.recv, hget⟩
        | exact ⟨inc.recv, by rw [List.getElem?_set_ne hu]; exact hget⟩)
  | recvStep u e =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      simp only [hget] at hs
      split at hs
      · cases hs
      · rename_i recv' _
        cases hs
        exact ⟨recv', List.getElem?_set_self hlt⟩
    · split at hs
      · cases hs
      · cases hs
      · split at hs
        · cases hs
        · cases hs
          exact ⟨inc.recv, by rw [List.getElem?_set_ne hu]; exact hget⟩
  | sweepRecv u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      simp [hget, hnd] at hs
    · split at hs
      · cases hs
      · cases hs
      · split at hs <;> cases hs
        exact ⟨inc.recv, by rw [List.getElem?_set_ne hu]; exact hget⟩
  | notifyForRead _ _ =>
    simp only [step] at hs
    split at hs <;> cases hs
    exact ⟨inc.recv, hget⟩
  | sendStep _ _ =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · split at hs <;> cases hs
      exact ⟨inc.recv, hget⟩
  | sweepSend _ =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · split at hs <;> cases hs
      exact ⟨inc.recv, hget⟩

/-- **Producer drain-before-reuse:** while a send entry `inc` at
`s.sendTable[uuid]` has `inc.send.life.done = false`, **every** enabled system
step preserves `inc`'s ownership of `uuid` with the exact same `(reqId, gen)`:
a `startRead` or `registerRecv` under that uuid is rejected (`RetireSettledSessionLocked`
returns false, `try_emplace` fails) and leaves it in place. -/
theorem active_send_preserved {s s' : State} {ev : Ev} {uuid : Uuid} {inc : SendEntry}
    (hget : s.sendTable[uuid]? = some (some inc))
    (hnd : inc.send.life.done = false)
    (hs : step s ev = some s') :
    ∃ send', s'.sendTable[uuid]? =
      some (some { reqId := inc.reqId, gen := inc.gen, send := send' }) := by
  have hlt : uuid < s.sendTable.length := lt_length_of_getElem?_eq hget
  cases ev with
  | startRead _ u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      split at hs
      · cases hs
      · split at hs <;> split at hs <;> cases hs <;> exact ⟨inc.send, hget⟩
      · simp only [hget, hnd, Bool.not_false, ↓reduceIte] at hs
        cases hs
        exact ⟨inc.send, hget⟩
    · repeat' split at hs
      all_goals ((cases hs) <;> first
        | exact ⟨inc.send, hget⟩
        | exact ⟨inc.send, by rw [List.getElem?_set_ne hu]; exact hget⟩)
  | registerRecv _ u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      split at hs
      · cases hs
      · repeat' split at hs
        all_goals ((cases hs) <;> exact ⟨inc.send, hget⟩)
      · simp only [hget, hnd, Bool.not_false, ↓reduceIte] at hs
        cases hs
    · repeat' split at hs
      all_goals ((cases hs) <;> first
        | exact ⟨inc.send, hget⟩
        | exact ⟨inc.send, by rw [List.getElem?_set_ne hu]; exact hget⟩)
  | recvStep _ _ =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · split at hs <;> cases hs <;> exact ⟨inc.send, hget⟩
  | sweepRecv _ =>
    simp only [step] at hs
    split at hs
    · cases hs
    · cases hs
    · split at hs <;> cases hs <;> exact ⟨inc.send, hget⟩
  | notifyForRead reqId u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu; simp [hget] at hs
    · split at hs <;> cases hs
      exact ⟨inc.send, by rw [List.getElem?_set_ne hu]; exact hget⟩
  | sendStep u e =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu
      simp only [hget] at hs
      split at hs
      · cases hs
      · rename_i send' _
        cases hs
        exact ⟨send', List.getElem?_set_self hlt⟩
    · split at hs
      · cases hs
      · cases hs
      · split at hs <;> cases hs
        exact ⟨inc.send, by rw [List.getElem?_set_ne hu]; exact hget⟩
  | sweepSend u =>
    simp only [step] at hs
    by_cases hu : u = uuid
    · subst hu; simp [hget, hnd] at hs
    · split at hs
      · cases hs
      · cases hs
      · split at hs <;> cases hs
        exact ⟨inc.send, by rw [List.getElem?_set_ne hu]; exact hget⟩

end TpuSyncVerify.Transfer.PrefillDecode.UuidTable

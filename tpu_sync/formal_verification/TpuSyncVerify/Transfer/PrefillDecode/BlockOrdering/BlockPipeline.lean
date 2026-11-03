import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering.Cancellation

/-! # `BlockOrdering`: multi-layer asynchronous `BlockPipeline` integration

Part of `TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering`. -/

namespace TpuSyncVerify.Transfer.PrefillDecode

namespace BlockOrdering

/-! ## 5. Multi-Layer Asynchronous `BlockPipeline` Integration -/

/-- A multi-layer prefill-to-decode transfer pipeline refined to 2D
`(layer, block)` memories (`List (List BlockCell)`), coupling `Pipeline` with
`validateRegistration`, `validateRequestedBlocks`, `CopyPlan`, and coalesced
D2H / H2H / H2D block execution. -/
structure BlockPipeline where
  pipe : Pipeline
  numBlocks : Nat
  numHostBlocks : Nat
  registeredBlocks : List Nat
  plan : CopyPlan
  prefillHbmB : List (List BlockCell)
  prefillStagingB : List (List BlockCell)
  wireB : List (Option (List BlockCell))
  decodeStagingB : List (List BlockCell)
  decodeHbmB : List (List BlockCell)
  prefillStagingReseated : Bool := false
  decodeStagingReseated : Bool := false
  deriving Repr, DecidableEq

namespace BlockPipeline

/-- Initial 2D prefill HBM (`numLayers × numBlocks`), where layer `l`, block `b`
holds `.kv l b`. -/
def initPrefillHbmB (numLayers numBlocks : Nat) : List (List BlockCell) :=
  (List.range numLayers).map (goodBlockLayer numBlocks)

/-- Construct an `n`-layer `BlockPipeline` for a given `registeredBlocks` and
`triples` specification. -/
def init (numLayers numBlocks numHostBlocks : Nat)
    (registeredBlocks : List Nat) (triples : List BlockTriple) : BlockPipeline :=
  let plan := buildCopyPlanFromTriples triples
  { pipe := Pipeline.init numLayers,
    numBlocks := numBlocks,
    numHostBlocks := numHostBlocks,
    registeredBlocks := registeredBlocks,
    plan := plan,
    prefillHbmB := initPrefillHbmB numLayers numBlocks,
    prefillStagingB := List.replicate numLayers (junkBlockLayer plan.numBlocks),
    wireB := List.replicate numLayers none,
    decodeStagingB := List.replicate numLayers (blankBlockLayer numHostBlocks),
    decodeHbmB := List.replicate numLayers (blankBlockLayer numBlocks) }

/-- Step function of `BlockPipeline`:
- `.send .beginPull` (`TransferSendSession::ValidateAndBeginPull`) additionally
  checks `validateRequestedBlocks s.registeredBlocks s.plan.producerRemoteBlocks = true`.
- `.d2hReady l` executes the coalesced D2H `CopySpec` (`s.plan.d2hSpec`) on layer `l`.
- `.h2hDone l true` places layer `l`'s staged producer blocks on the wire.
- `.land l` lands the wire blocks into `decodeStagingB[l]` using `s.plan.h2hPairs`.
- `.h2dReady l` executes the coalesced H2D `CopySpec` (`s.plan.h2dSpec`) on layer `l`.
- `.reclaim`, `.reseatPrefillStaging`, and `.reseatDecodeStaging` overwrite the
  corresponding 2D block buffers with `.junk`. -/
def step (s : BlockPipeline) (e : Pipeline.Ev) : Option BlockPipeline :=
  match e with
  | .send .beginPull =>
    if validateRequestedBlocks s.registeredBlocks s.plan.producerRemoteBlocks = true then
      (Pipeline.step s.pipe (.send .beginPull)).map fun p => { s with pipe := p }
    else none
  | .d2hReady l =>
    (Pipeline.step s.pipe (.d2hReady l)).map fun p =>
      let srcLayer := s.prefillHbmB.getD l []
      let dstLayer := s.prefillStagingB.getD l []
      let staged := execCoalesced .junk srcLayer dstLayer s.plan.d2hSpec
      { s with pipe := p, prefillStagingB := s.prefillStagingB.set l staged }
  | .h2hDone l ok =>
    (Pipeline.step s.pipe (.h2hDone l ok)).map fun p =>
      if ok then
        { s with pipe := p, wireB := s.wireB.set l (some (s.prefillStagingB.getD l [])) }
      else
        { s with pipe := p }
  | .land l =>
    (Pipeline.step s.pipe (.land l)).map fun p =>
      let wireLayer := (s.wireB.getD l none).getD []
      let dstLayer := s.decodeStagingB.getD l []
      let landed := execElementwise .junk wireLayer dstLayer s.plan.h2hPairs
      { s with pipe := p, decodeStagingB := s.decodeStagingB.set l landed }
  | .h2dReady l =>
    (Pipeline.step s.pipe (.h2dReady l)).map fun p =>
      let srcLayer := s.decodeStagingB.getD l []
      let dstLayer := s.decodeHbmB.getD l []
      let copied := execCoalesced .junk srcLayer dstLayer s.plan.h2dSpec
      { s with pipe := p, decodeHbmB := s.decodeHbmB.set l copied }
  | .reclaim =>
    (Pipeline.step s.pipe .reclaim).map fun p =>
      { s with pipe := p,
               prefillHbmB := List.replicate s.pipe.numLayers (junkBlockLayer s.numBlocks) }
  | .reseatPrefillStaging =>
    (Pipeline.step s.pipe .reseatPrefillStaging).map fun p =>
      { s with pipe := p,
               prefillStagingB := List.replicate s.pipe.numLayers (junkBlockLayer s.plan.numBlocks),
               prefillStagingReseated := true }
  | .reseatDecodeStaging =>
    (Pipeline.step s.pipe .reseatDecodeStaging).map fun p =>
      { s with pipe := p,
               decodeStagingB := List.replicate s.pipe.numLayers (junkBlockLayer s.numHostBlocks),
               decodeStagingReseated := true }
  | e =>
    (Pipeline.step s.pipe e).map fun p => { s with pipe := p }

def sys (numLayers numBlocks numHostBlocks : Nat)
    (registeredBlocks : List Nat) (triples : List BlockTriple) :
    System BlockPipeline Pipeline.Ev :=
  ⟨init numLayers numBlocks numHostBlocks registeredBlocks triples, step⟩

/-- Projection lemma: every `BlockPipeline` step refines a `Pipeline` step. -/
theorem step_pipe {s s' : BlockPipeline} {e : Pipeline.Ev}
    (hs : step s e = some s') : Pipeline.step s.pipe e = some s'.pipe := by
  cases e with
  | send se =>
    cases se <;> simp only [step] at hs
    case beginPull =>
      split at hs
      · simp only [Option.map_eq_some_iff] at hs
        rcases hs with ⟨p, hp, rfl⟩
        exact hp
      · cases hs
    all_goals
      simp only [Option.map_eq_some_iff] at hs
      rcases hs with ⟨p, hp, rfl⟩
      exact hp
  | h2hDone l ok =>
    simp only [step, Option.map_eq_some_iff] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split <;> exact hp
  | _ =>
    simp only [step, Option.map_eq_some_iff] at hs
    rcases hs with ⟨p, hp, rfl⟩
    exact hp

/-! ### Block-level Safety Properties -/

/-- `TransferSendSession::ValidateRequestedBlocksLocked` safety: whenever
`pull_started_` is claimed (`s.pipe.send.pullStarted = true`), the requested
`producerRemoteBlocks` passed non-empty, subset, and no-duplicate validation
against `s.registeredBlocks`. -/
def BlockValidationSafe (s : BlockPipeline) : Prop :=
  s.pipe.send.pullStarted = true →
    s.plan.producerRemoteBlocks ≠ [] ∧
    (∀ b ∈ s.plan.producerRemoteBlocks, b ∈ s.registeredBlocks) ∧
    s.plan.producerRemoteBlocks.Nodup

/-- **Within-layer & across-layer publication correctness:** when `poll_stats()`
reports `done_recving` (`s.pipe.recv.published = some true`), for every layer
`l < s.pipe.numLayers`, `s.decodeHbmB[l]` equals `h2dStage l s.numBlocks s.numHostBlocks s.plan`
— which by `h2dStage_get_requested` and `h2dStage_get_untouched` holds
`some (.kv l t.remote)` at every `t.local_` and `some .blank` at every untouched
block `b < s.numBlocks`. -/
def BlockPublicationCorrect (s : BlockPipeline) : Prop :=
  s.pipe.recv.published = some true →
    ∀ l < s.pipe.numLayers,
      s.decodeHbmB[l]? = some (h2dStage l s.numBlocks s.numHostBlocks s.plan)

/-- **Custom host staging block correctness
(`KVCacheManagerWithTransferTest.LocalOrchestratedTransferToCustomHostBlock`):**
as long as consumer host staging has not been reseated
(`s.decodeStagingReseated = false`), every landed layer `l` in
`s.decodeStagingB[l]` equals `landStage l s.numBlocks s.numHostBlocks s.plan`
— which by `landStage_get_requested` and `landStage_get_untouched` holds
`some (.kv l t.remote)` at every `t.host` and `some .blank` at every untouched
host block `h < s.numHostBlocks`. -/
def CustomHostStagingCorrect (s : BlockPipeline) : Prop :=
  s.decodeStagingReseated = false →
    ∀ l : Nat, s.pipe.landedL[l]? = some true →
      s.decodeStagingB[l]? = some (landStage l s.numBlocks s.numHostBlocks s.plan)

def Safe (s : BlockPipeline) : Prop :=
  s.pipe.Safe ∧ BlockValidationSafe s ∧ BlockPublicationCorrect s ∧
    CustomHostStagingCorrect s

/-! ### Inductive Invariant of `BlockPipeline` -/

structure Inv (s : BlockPipeline) : Prop where
  pipe : s.pipe.Inv
  len_phbmB : s.prefillHbmB.length = s.pipe.numLayers
  len_pstagingB : s.prefillStagingB.length = s.pipe.numLayers
  len_wireB : s.wireB.length = s.pipe.numLayers
  len_dstagingB : s.decodeStagingB.length = s.pipe.numLayers
  len_dhbmB : s.decodeHbmB.length = s.pipe.numLayers
  pull_validated : s.pipe.send.pullStarted = true →
    validateRequestedBlocks s.registeredBlocks s.plan.producerRemoteBlocks = true
  pstg_reseated_done : s.prefillStagingReseated = true → s.pipe.send.life.done = true
  dstg_reseated_done : s.decodeStagingReseated = true → s.pipe.recv.life.done = true
  phbmB_good : s.pipe.reclaimed = false →
    ∀ l < s.pipe.numLayers, s.prefillHbmB[l]? = some (goodBlockLayer s.numBlocks l)
  pstagingB_unwritten : s.prefillStagingReseated = false →
    ∀ l < s.pipe.numLayers, s.pipe.d2hReadyL[l]? = some false →
      s.prefillStagingB[l]? = some (junkBlockLayer s.plan.numBlocks)
  pstagingB_good : s.prefillStagingReseated = false →
    ∀ l : Nat, s.pipe.d2hReadyL[l]? = some true →
      s.prefillStagingB[l]? = some (d2hStage l s.numBlocks s.plan)
  wireB_some_iff : ∀ l < s.pipe.numLayers,
    s.pipe.wire[l]? = some (.kv l) →
      s.wireB[l]? = some (some (d2hStage l s.numBlocks s.plan))
  dstagingB_unwritten : s.decodeStagingReseated = false →
    ∀ l < s.pipe.numLayers, s.pipe.landedL[l]? = some false →
      s.decodeStagingB[l]? = some (blankBlockLayer s.numHostBlocks)
  dstagingB_good : s.decodeStagingReseated = false →
    ∀ l : Nat, s.pipe.landedL[l]? = some true →
      s.decodeStagingB[l]? = some (landStage l s.numBlocks s.numHostBlocks s.plan)
  dhbmB_unwritten : ∀ l < s.pipe.numLayers, s.pipe.h2dReadyL[l]? = some false →
    s.decodeHbmB[l]? = some (blankBlockLayer s.numBlocks)
  dhbmB_good : ∀ l : Nat, s.pipe.h2dReadyL[l]? = some true →
    s.decodeHbmB[l]? = some (h2dStage l s.numBlocks s.numHostBlocks s.plan)

theorem inv_init (numLayers numBlocks numHostBlocks : Nat)
    (registeredBlocks : List Nat) (triples : List BlockTriple) :
    Inv (init numLayers numBlocks numHostBlocks registeredBlocks triples) := by
  refine ⟨Pipeline.inv_init numLayers,
    by simp [init, initPrefillHbmB, Pipeline.init],
    by simp [init, Pipeline.init],
    by simp [init, Pipeline.init],
    by simp [init, Pipeline.init],
    by simp [init, Pipeline.init],
    nofun, nofun, nofun,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro _ l hl
    simp only [init, initPrefillHbmB, Pipeline.init] at hl ⊢
    simp [hl]
  · intro _ l hl _
    simp only [init, Pipeline.init] at hl ⊢
    simp [hl]
  · intro _ l h
    simp only [init, Pipeline.init] at h
    exact (not_mem_replicate_false h).elim
  · intro l _ h
    simp only [init, Pipeline.init, List.getElem?_replicate] at h
    split at h <;> cases h
  · intro _ l hl _
    simp only [init, Pipeline.init] at hl ⊢
    simp [hl]
  · intro _ l h
    simp only [init, Pipeline.init] at h
    exact (not_mem_replicate_false h).elim
  · intro l hl _
    simp only [init, Pipeline.init] at hl ⊢
    simp [hl]
  · intro l h
    simp only [init, Pipeline.init] at h
    exact (not_mem_replicate_false h).elim

theorem inv_safe {s : BlockPipeline} (h : Inv s) : Safe s := by
  refine ⟨Pipeline.inv_safe h.pipe, ?_, ?_, h.dstagingB_good⟩
  · intro hps
    exact (validateRequestedBlocks_iff _ _).mp (h.pull_validated hps)
  · intro hpub l hl
    apply h.dhbmB_good
    have hready : s.pipe.recv.ready = s.pipe.numLayers := by
      have := h.pipe.recv.published_ok hpub
      have := h.pipe.recv.completed_le
      have := h.pipe.recv.retired_le
      have := h.pipe.recv.ready_le
      have := h.pipe.recv.issued_le
      have := h.pipe.n_recv
      omega
    apply all_true_of_countTrue_eq_length
    · rw [h.pipe.cnt_h2dReady, hready, h.pipe.len_h2dReadyL]
    · rw [h.pipe.len_h2dReadyL]; exact hl

/-- Combined spatial + temporal theorem at `done_recving`: when `s` is reachable
with `ValidTriples s.numBlocks s.numHostBlocks triples` and `s.pipe.recv.published = some true`,
every layer `l < s.pipe.numLayers` has `.kv l t.remote` at `t.local_` for every
`t ∈ triples` and `.blank` at every untouched block `b < s.numBlocks`. -/
theorem inv_decodeHbm_requested {s : BlockPipeline} {triples : List BlockTriple}
    (h : Inv s) (hplan : s.plan = buildCopyPlanFromTriples triples)
    (hv : ValidTriples s.numBlocks s.numHostBlocks triples)
    (hpub : s.pipe.recv.published = some true)
    {l : Nat} (hl : l < s.pipe.numLayers)
    {t : BlockTriple} (ht : t ∈ triples) :
    (s.decodeHbmB.getD l [])[t.local_]? = some (.kv l t.remote) := by
  have hlayer := (inv_safe h).2.2.1 hpub l hl
  rw [List.getD_eq_getElem?_getD, hlayer, Option.getD_some, hplan]
  exact h2dStage_get_requested hv ht

theorem inv_decodeHbm_untouched {s : BlockPipeline} {triples : List BlockTriple}
    (h : Inv s) (hplan : s.plan = buildCopyPlanFromTriples triples)
    (hpub : s.pipe.recv.published = some true)
    {l : Nat} (hl : l < s.pipe.numLayers)
    {b : Nat} (hb : b < s.numBlocks) (huntouched : ∀ t ∈ triples, t.local_ ≠ b) :
    (s.decodeHbmB.getD l [])[b]? = some .blank := by
  have hlayer := (inv_safe h).2.2.1 hpub l hl
  rw [List.getD_eq_getElem?_getD, hlayer, Option.getD_some, hplan]
  exact h2dStage_get_untouched hb huntouched

/-- Frame lemma for `BlockPipeline` transitions that only update `pipe` without
modifying `numLayers`, `reclaimed`, `d2hReadyL`, `wire`, `landedL`, `h2dReadyL`,
or the 2D block memories. -/
theorem Inv.frame_no_mem {s : BlockPipeline} {p : Pipeline} (h : Inv s)
    (hp : p.Inv)
    (hn : p.numLayers = s.pipe.numLayers)
    (hrec : p.reclaimed = s.pipe.reclaimed)
    (hd2h : p.d2hReadyL = s.pipe.d2hReadyL)
    (hwire : p.wire = s.pipe.wire)
    (hland : p.landedL = s.pipe.landedL)
    (hh2d : p.h2dReadyL = s.pipe.h2dReadyL)
    (hs_done : s.pipe.send.life.done = true → p.send.life.done = true)
    (hr_done : s.pipe.recv.life.done = true → p.recv.life.done = true)
    (hpull : p.send.pullStarted = true →
      validateRequestedBlocks s.registeredBlocks s.plan.producerRemoteBlocks = true) :
    Inv { s with pipe := p } := by
  refine ⟨hp,
    by rw [hn]; exact h.len_phbmB,
    by rw [hn]; exact h.len_pstagingB,
    by rw [hn]; exact h.len_wireB,
    by rw [hn]; exact h.len_dstagingB,
    by rw [hn]; exact h.len_dhbmB,
    hpull,
    fun hpr => hs_done (h.pstg_reseated_done hpr),
    fun hdr => hr_done (h.dstg_reseated_done hdr),
    by rw [hrec, hn]; exact h.phbmB_good,
    by rw [hn, hd2h]; exact h.pstagingB_unwritten,
    by rw [hd2h]; exact h.pstagingB_good,
    by rw [hn, hwire]; exact h.wireB_some_iff,
    by rw [hn, hland]; exact h.dstagingB_unwritten,
    by rw [hland]; exact h.dstagingB_good,
    by rw [hn, hh2d]; exact h.dhbmB_unwritten,
    by rw [hh2d]; exact h.dhbmB_good⟩

theorem step_inv {s s' : BlockPipeline} {e : Pipeline.Ev}
    (h : Inv s) (hs : step s e = some s') : Inv s' := by
  have hp_inv : s'.pipe.Inv := Pipeline.step_inv h.pipe (step_pipe hs)
  cases e with
  | notifyForRead =>
    simp only [step, Pipeline.step, Pipeline.notifyForRead] at hs
    split at hs
    · simp only [Option.map_eq_some_iff, Option.some.injEq] at hs
      rcases hs with ⟨_, rfl, rfl⟩
      exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl id id h.pull_validated
    · cases hs
  | pullWait =>
    simp only [step, Pipeline.step, Pipeline.pullWait] at hs
    split at hs
    · simp only [Option.map_eq_some_iff, Option.some.injEq] at hs
      rcases hs with ⟨_, rfl, rfl⟩
      exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl id id h.pull_validated
    · cases hs
  | send se =>
    cases se with
    | d2hReady =>
      simp only [step, Pipeline.step, Pipeline.sendStep] at hs; cases hs
    | h2hDone ok =>
      simp only [step, Pipeline.step, Pipeline.sendStep] at hs; cases hs
    | beginPull =>
      simp only [step] at hs
      split at hs
      · rename_i hval
        simp only [Option.map_eq_some_iff, Pipeline.step, Pipeline.sendStep] at hs
        rcases hs with ⟨p, hp, rfl⟩
        split at hp
        · simp only [Option.map_eq_some_iff] at hp
          rcases hp with ⟨snd, hsnd, rfl⟩
          exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
            (Send.step_done_mono hsnd) id (fun _ => hval)
        · cases hp
      · cases hs
    | wake ok =>
      simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.sendStep] at hs
      rcases hs with ⟨p, hp, rfl⟩
      split at hp
      · simp only [Option.map_eq_some_iff] at hp
        rcases hp with ⟨snd, hsnd, rfl⟩
        exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
          (Send.step_done_mono hsnd) id
          (by rw [Send.step_pullStarted_of_ne_beginPull hsnd nofun]; exact h.pull_validated)
      · cases hp
    | start | d2hBegin | d2hIssue _ | d2hEnd | h2hIssue | sendNext | cancel | publish =>
      simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.sendStep] at hs
      rcases hs with ⟨p, ⟨snd, hsnd, rfl⟩, rfl⟩
      exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
        (Send.step_done_mono hsnd) id
        (by rw [Send.step_pullStarted_of_ne_beginPull hsnd nofun]; exact h.pull_validated)
  | recv re =>
    cases re with
    | h2dBegin | h2dIssue _ | h2dReady =>
      simp only [step, Pipeline.step, Pipeline.recvStep] at hs; cases hs
    | pullReply ok =>
      simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.recvStep] at hs
      rcases hs with ⟨p, hp, rfl⟩
      split at hp
      · simp only [Option.map_eq_some_iff] at hp
        rcases hp with ⟨rcv, hrcv, rfl⟩
        exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
          id (Recv.step_done_mono hrcv) h.pull_validated
      · cases hp
    | pushBegin | pushEnd | h2dDone _ | netAccount | pollReady | cancel | publish =>
      simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.recvStep] at hs
      rcases hs with ⟨p, ⟨rcv, hrcv, rfl⟩, rfl⟩
      exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
        id (Recv.step_done_mono hrcv) h.pull_validated
  | h2dBegin l =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.h2dBegin] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · simp only [Option.map_eq_some_iff] at hp
      rcases hp with ⟨rcv, hrcv, rfl⟩
      exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
        id (Recv.step_done_mono hrcv) h.pull_validated
    · cases hp
  | h2dIssue l ok =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.h2dIssue] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · simp only [Option.map_eq_some_iff] at hp
      rcases hp with ⟨rcv, hrcv, rfl⟩
      exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
        id (Recv.step_done_mono hrcv) h.pull_validated
    · cases hp
  | d2hReady l =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.d2hReady] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · rename_i hg
      rcases hg with ⟨hl_iss, hf⟩
      simp only [Option.map_eq_some_iff] at hp
      rcases hp with ⟨snd, hsnd, rfl⟩
      obtain ⟨hlt, rfl⟩ := Send.d2hReady_spec hsnd
      have hd : s.pipe.send.life.done = false :=
        Pipeline.send_not_done_of_outstanding h.pipe.send (Or.inl hlt)
      have hnr : s.pipe.reclaimed = false := by
        cases hr : s.pipe.reclaimed
        · rfl
        · have := h.pipe.reclaimed_done hr; rw [hd] at this; cases this
      have hnpr : s.prefillStagingReseated = false := by
        cases hpr : s.prefillStagingReseated
        · rfl
        · have := h.pstg_reseated_done hpr; rw [hd] at this; cases this
      have hk : l < s.pipe.numLayers := by
        have := lt_length_of_getElem?_eq hf
        rw [h.pipe.len_d2hReadyL] at this
        exact this
      have hsrc : s.prefillHbmB.getD l [] = goodBlockLayer s.numBlocks l := by
        rw [List.getD_eq_getElem?_getD, h.phbmB_good hnr l hk]; rfl
      have hdst : s.prefillStagingB.getD l [] = junkBlockLayer s.plan.numBlocks := by
        rw [List.getD_eq_getElem?_getD, h.pstagingB_unwritten hnpr l hk hf]; rfl
      rw [hsrc, hdst]
      refine ⟨hp_inv, h.len_phbmB, ?_, h.len_wireB, h.len_dstagingB, h.len_dhbmB,
        h.pull_validated, h.pstg_reseated_done, h.dstg_reseated_done,
        h.phbmB_good, ?_, ?_, h.wireB_some_iff, h.dstagingB_unwritten,
        h.dstagingB_good, h.dhbmB_unwritten, h.dhbmB_good⟩
      · rw [List.length_set]; exact h.len_pstagingB
      · intro hpr j hj hjf
        have hjl : j ≠ l := by
          intro heq; subst heq
          rw [List.getElem?_set_self (by rw [h.pipe.len_d2hReadyL]; exact hk)] at hjf
          cases hjf
        rw [List.getElem?_set_ne (Ne.symm hjl)] at hjf ⊢
        exact h.pstagingB_unwritten hpr j hj hjf
      · intro hpr j hj
        by_cases hjl : j = l
        · subst hjl
          exact List.getElem?_set_self (by rw [h.len_pstagingB]; exact hk)
        · rw [List.getElem?_set_ne (Ne.symm hjl)] at hj ⊢
          exact h.pstagingB_good hpr j hj
    · cases hp
  | h2hDone l ok =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.h2hDone] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · rename_i hg
      rcases hg with ⟨hl_iss, hf⟩
      simp only [Option.map_eq_some_iff] at hp
      rcases hp with ⟨snd, hsnd, rfl⟩
      cases ok with
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte]
        exact h.frame_no_mem hp_inv rfl rfl rfl rfl rfl rfl
          (Send.step_done_mono hsnd) id
          (by rw [Send.step_pullStarted_of_ne_beginPull hsnd nofun]; exact h.pull_validated)
      | true =>
        simp only [↓reduceIte]
        have hlt := Send.h2hDone_guard hsnd
        have hcnt := h.pipe.send.counters
        unfold Send.CountersOrdered at hcnt
        have hd : s.pipe.send.life.done = false :=
          Pipeline.send_not_done_of_outstanding h.pipe.send (Or.inr hlt)
        have hnpr : s.prefillStagingReseated = false := by
          cases hpr : s.prefillStagingReseated
          · rfl
          · have := h.pstg_reseated_done hpr; rw [hd] at this; cases this
        have hk : l < s.pipe.numLayers := by have := h.pipe.n_send; omega
        have hdl : s.pipe.d2hReadyL[l]? = some true := h.pipe.woken_d2hReady l (by omega) (by omega)
        have hsrc : s.prefillStagingB.getD l [] = d2hStage l s.numBlocks s.plan := by
          rw [List.getD_eq_getElem?_getD, h.pstagingB_good hnpr l hdl]; rfl
        rw [hsrc]
        refine ⟨hp_inv, h.len_phbmB, h.len_pstagingB, ?_, h.len_dstagingB, h.len_dhbmB,
          by rw [Send.step_pullStarted_of_ne_beginPull hsnd nofun]; exact h.pull_validated,
          fun hpr => Send.step_done_mono hsnd (h.pstg_reseated_done hpr),
          h.dstg_reseated_done, h.phbmB_good, h.pstagingB_unwritten, h.pstagingB_good,
          ?_, h.dstagingB_unwritten, h.dstagingB_good, h.dhbmB_unwritten, h.dhbmB_good⟩
        · rw [List.length_set]; exact h.len_wireB
        · intro j hj hjw
          by_cases hjl : j = l
          · subst hjl
            exact List.getElem?_set_self (by rw [h.len_wireB]; exact hk)
          · rw [List.getElem?_set_ne (Ne.symm hjl)] at hjw ⊢
            exact h.wireB_some_iff j hj hjw
    · cases hp
  | land l =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.land] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · rename_i hg
      rcases hg with ⟨hpush, hf⟩
      split at hp
      · rename_i c hc
        split at hp
        · cases hp
        · rename_i hcb
          cases hp
          have hkv : c = .kv l := (h.pipe.wire_good l c hc).resolve_left hcb
          subst hkv
          have hk : l < s.pipe.numLayers := by
            have := lt_length_of_getElem?_eq hc
            rw [h.pipe.len_wire] at this
            exact this
          have hd : s.pipe.recv.life.done = false := by
            cases hd : s.pipe.recv.life.done
            · rfl
            · have := h.pipe.recv.life.done_idle hd
              have := h.pipe.recv.accounted
              unfold Recv.Accounted at this
              omega
          have hndr : s.decodeStagingReseated = false := by
            cases hdr : s.decodeStagingReseated
            · rfl
            · have := h.dstg_reseated_done hdr; rw [hd] at this; cases this
          have hwire : (s.wireB.getD l none).getD [] = d2hStage l s.numBlocks s.plan := by
            rw [List.getD_eq_getElem?_getD, h.wireB_some_iff l hk hc]; rfl
          have hdst : s.decodeStagingB.getD l [] = blankBlockLayer s.numHostBlocks := by
            rw [List.getD_eq_getElem?_getD, h.dstagingB_unwritten hndr l hk hf]; rfl
          rw [hwire, hdst]
          refine ⟨hp_inv, h.len_phbmB, h.len_pstagingB, h.len_wireB, ?_, h.len_dhbmB,
            h.pull_validated, h.pstg_reseated_done, h.dstg_reseated_done,
            h.phbmB_good, h.pstagingB_unwritten, h.pstagingB_good, h.wireB_some_iff,
            ?_, ?_, h.dhbmB_unwritten, h.dhbmB_good⟩
          · rw [List.length_set]; exact h.len_dstagingB
          · intro hdr j hj hjf
            have hjl : j ≠ l := by
              intro heq; subst heq
              rw [List.getElem?_set_self (lt_length_of_getElem?_eq hf)] at hjf
              cases hjf
            rw [List.getElem?_set_ne (Ne.symm hjl)] at hjf ⊢
            exact h.dstagingB_unwritten hdr j hj hjf
          · intro hdr j hj
            by_cases hjl : j = l
            · subst hjl
              exact List.getElem?_set_self (by rw [h.len_dstagingB]; exact hk)
            · rw [List.getElem?_set_ne (Ne.symm hjl)] at hj ⊢
              exact h.dstagingB_good hdr j hj
      · cases hp
    · cases hp
  | h2dReady l =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.h2dReady] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · rename_i hg
      rcases hg with ⟨hiss, hf⟩
      simp only [Option.map_eq_some_iff] at hp
      rcases hp with ⟨rcv, hrcv, rfl⟩
      obtain ⟨hlt, rfl⟩ := Recv.h2dReady_spec hrcv
      have hd : s.pipe.recv.life.done = false := by
        cases hd : s.pipe.recv.life.done
        · rfl
        · have := (Recv.inv_safe h.pipe.recv).2.1 hd
          have := h.pipe.recv.retired_le
          omega
      have hndr : s.decodeStagingReseated = false := by
        cases hdr : s.decodeStagingReseated
        · rfl
        · have := h.dstg_reseated_done hdr; rw [hd] at this; cases this
      have hk : l < s.pipe.numLayers := by
        have := lt_length_of_getElem?_eq hf
        rw [h.pipe.len_h2dReadyL] at this
        exact this
      have hland : s.pipe.landedL[l]? = some true := h.pipe.issued_landed l hiss
      have hsrc : s.decodeStagingB.getD l [] = landStage l s.numBlocks s.numHostBlocks s.plan := by
        rw [List.getD_eq_getElem?_getD, h.dstagingB_good hndr l hland]; rfl
      have hdst : s.decodeHbmB.getD l [] = blankBlockLayer s.numBlocks := by
        rw [List.getD_eq_getElem?_getD, h.dhbmB_unwritten l hk hf]; rfl
      rw [hsrc, hdst]
      refine ⟨hp_inv, h.len_phbmB, h.len_pstagingB, h.len_wireB, h.len_dstagingB, ?_,
        h.pull_validated, h.pstg_reseated_done, h.dstg_reseated_done,
        h.phbmB_good, h.pstagingB_unwritten, h.pstagingB_good, h.wireB_some_iff,
        h.dstagingB_unwritten, h.dstagingB_good, ?_, ?_⟩
      · rw [List.length_set]; exact h.len_dhbmB
      · intro j hj hjf
        have hjl : j ≠ l := by
          intro heq; subst heq
          rw [List.getElem?_set_self (by rw [h.pipe.len_h2dReadyL]; exact hk)] at hjf
          cases hjf
        rw [List.getElem?_set_ne (Ne.symm hjl)] at hjf ⊢
        exact h.dhbmB_unwritten j hj hjf
      · intro j hj
        by_cases hjl : j = l
        · subst hjl
          exact List.getElem?_set_self (by rw [h.len_dhbmB]; exact hk)
        · rw [List.getElem?_set_ne (Ne.symm hjl)] at hj ⊢
          exact h.dhbmB_good j hj
    · cases hp
  | reclaim =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.reclaim] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · cases hp
      refine ⟨hp_inv, by simp, h.len_pstagingB, h.len_wireB, h.len_dstagingB, h.len_dhbmB,
        h.pull_validated, h.pstg_reseated_done, h.dstg_reseated_done,
        nofun, h.pstagingB_unwritten, h.pstagingB_good, h.wireB_some_iff,
        h.dstagingB_unwritten, h.dstagingB_good, h.dhbmB_unwritten, h.dhbmB_good⟩
    · cases hp
  | reseatPrefillStaging =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.reseatPrefillStaging] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · rename_i hrel
      cases hp
      have hd : s.pipe.send.life.done = true := h.pipe.send.life.done_of_released hrel
      refine ⟨hp_inv, h.len_phbmB, by simp, h.len_wireB, h.len_dstagingB, h.len_dhbmB,
        h.pull_validated, fun _ => hd, h.dstg_reseated_done,
        h.phbmB_good, nofun, nofun, h.wireB_some_iff,
        h.dstagingB_unwritten, h.dstagingB_good, h.dhbmB_unwritten, h.dhbmB_good⟩
    · cases hp
  | reseatDecodeStaging =>
    simp only [step, Option.map_eq_some_iff, Pipeline.step, Pipeline.reseatDecodeStaging] at hs
    rcases hs with ⟨p, hp, rfl⟩
    split at hp
    · rename_i hrel
      cases hp
      have hd : s.pipe.recv.life.done = true := h.pipe.recv.life.done_of_released hrel
      refine ⟨hp_inv, h.len_phbmB, h.len_pstagingB, h.len_wireB, by simp, h.len_dhbmB,
        h.pull_validated, h.pstg_reseated_done, fun _ => hd,
        h.phbmB_good, h.pstagingB_unwritten, h.pstagingB_good, h.wireB_some_iff,
        nofun, nofun, h.dhbmB_unwritten, h.dhbmB_good⟩
    · cases hp

theorem reachable_inv {numLayers numBlocks numHostBlocks : Nat}
    {registeredBlocks : List Nat} {triples : List BlockTriple} {s : BlockPipeline}
    (hr : (sys numLayers numBlocks numHostBlocks registeredBlocks triples).Reachable s) :
    Inv s :=
  (sys numLayers numBlocks numHostBlocks registeredBlocks triples).reachable_induction
    (inv_init numLayers numBlocks numHostBlocks registeredBlocks triples)
    (fun _ _ _ h hs => step_inv h hs) hr

/-- **Main theorem of `BlockPipeline`:** every reachable state satisfies
`Pipeline.Safe`, `BlockValidationSafe`, `BlockPublicationCorrect`, and
`CustomHostStagingCorrect`. -/
theorem reachable_safe {numLayers numBlocks numHostBlocks : Nat}
    {registeredBlocks : List Nat} {triples : List BlockTriple} {s : BlockPipeline}
    (hr : (sys numLayers numBlocks numHostBlocks registeredBlocks triples).Reachable s) :
    Safe s :=
  inv_safe (reachable_inv hr)

end BlockPipeline


end BlockOrdering

end TpuSyncVerify.Transfer.PrefillDecode

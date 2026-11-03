import TpuSyncVerify.Transfer.PrefillDecode.LayerStaging.Invariant
import TpuSyncVerify.Common.ModelCheck

/-! # `LayerStaging`: concrete replay traces, bounded search, and mutants

Part of `TpuSyncVerify.Transfer.PrefillDecode.LayerStaging`. Replays all four
bounded-staging GTests in `kv_cache_manager_with_transfer_send_drain_test.cc`
together with lock-gap interleavings, zero-slot rejection, unregistered
handshake flows, and three guard/cleanup mutants. -/

namespace TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

open TpuSyncVerify.Transfer (StagingLayerManager)

/-! ## Event enumeration and step sequences -/

def events (n : Nat) : List Ev :=
  [.notifyForRead, .pullWait, .beginPull, .sendStart,
   .sendAcquireBegin, .sendAcquireDone false, .sendAcquireDone true,
   .sendD2hBegin, .sendD2hIssue true, .sendD2hIssue false,
   .sendD2hEnd, .sendWake true, .sendWake false,
   .sendH2hIssue, .sendNext, .sendCancel, .sendPublish,
   .recvPullReply true, .recvPullReply false, .recvPushBegin, .recvPushEnd,
   .recvNetAccount, .recvPollReady, .recvCancel, .recvPublish,
   .reclaim, .reseatPrefillStaging, .reseatDecodeStaging] ++
  (List.range n).flatMap fun l =>
    [.d2hReady l, .h2hDone l true, .h2hDone l false,
     .recvAcquireBegin l, .recvAcquireWait l false, .recvAcquireWait l true,
     .recvAcquireFinish l, .land l, .h2dBegin l, .h2dIssue l true, .h2dIssue l false,
     .h2dReady l, .h2dDone l true, .h2dDone l false]

def sendIssueD2h : List Ev :=
  [.sendAcquireBegin, .sendAcquireDone false, .sendD2hBegin, .sendD2hIssue true]

def sendFinishD2hStartH2h (l : Nat) : List Ev :=
  [.d2hReady l, .sendD2hEnd, .sendWake true, .sendH2hIssue, .sendNext]

def sendOneLayer (l : Nat) : List Ev := sendIssueD2h ++ sendFinishD2hStartH2h l ++ [.h2hDone l true]

def streamProducer (n : Nat) : List Ev :=
  [.beginPull, .sendStart] ++ (List.range n).flatMap sendOneLayer ++ [.sendPublish]

def recvAcquireOk (l : Nat) : List Ev := [.recvAcquireBegin l, .recvAcquireWait l false, .recvAcquireFinish l]

def recvIssueH2d (l : Nat) : List Ev :=
  [.recvPushBegin] ++ recvAcquireOk l ++ [.land l, .h2dBegin l, .h2dIssue l true, .recvNetAccount, .recvPushEnd]

def recvFinishH2d (l : Nat) : List Ev := [.h2dReady l, .h2dDone l true]

/-! ## GTest replay traces (`send_drain_test.cc`) -/

/-- `SendLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` (`sys 4 2 2`). -/
def traceSendCyclesEvs : List Ev :=
  [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++ [.sendAcquireBegin] ++
  sendFinishD2hStartH2h 0 ++ sendFinishD2hStartH2h 1 ++
  [.h2hDone 1 true, .sendAcquireDone false, .sendD2hBegin, .sendD2hIssue true] ++
  sendFinishD2hStartH2h 2 ++
  [.sendAcquireBegin, .h2hDone 0 true, .sendAcquireDone false, .sendD2hBegin, .sendD2hIssue true] ++
  sendFinishD2hStartH2h 3 ++ [.h2hDone 2 true, .h2hDone 3 true, .sendPublish]

theorem trace_send_cycles_host_layer_buffers :
    (let pre := [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++
                  [.sendAcquireBegin] ++ sendFinishD2hStartH2h 0 ++ sendFinishD2hStartH2h 1
     ((sys 4 2 2).run pre).map (fun s =>
       (s.sendMgr.numFree, sendLookupStagingSlot s 0, sendLookupStagingSlot s 1)) = some (0, some 0, some 1) ∧
     (sys 4 2 2).run (pre ++ [.sendAcquireDone false]) = none ∧
     ((sys 4 2 2).run (pre ++ [.h2hDone 1 true, .sendAcquireDone false])).map (fun s =>
       (s.sendMgr.lookup 1, sendLookupStagingSlot s 2)) = some (none, some 1)) ∧
    ((sys 4 2 2).run traceSendCyclesEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.life.hasStaging, s.pipe.send.published, s.wireB)) =
      some (2, false, some true, [.kv 0, .kv 1, .kv 2, .kv 3]) := by decide

/-- `RecvLifecycleTest.StagingLayerManagerCyclesHostLayerBuffers` (`sys 4 2 2`). -/
def traceRecvCyclesEvs : List Ev :=
  traceSendCyclesEvs ++ [.recvPullReply true] ++ recvAcquireOk 0 ++ recvIssueH2d 0 ++
  recvIssueH2d 1 ++ [.recvPushBegin, .recvAcquireBegin 2] ++ recvFinishH2d 1 ++
  [.recvAcquireWait 2 false, .recvAcquireFinish 2,
   .land 2, .h2dBegin 2, .h2dIssue 2 true, .recvNetAccount, .recvPushEnd] ++
  recvFinishH2d 0 ++ recvIssueH2d 3 ++ recvFinishH2d 2 ++ recvFinishH2d 3 ++ [.recvPublish]

theorem trace_recv_cycles_host_layer_buffers :
    (let pre0 := traceSendCyclesEvs ++ [.recvPullReply true] ++ recvAcquireOk 0
     ((sys 4 2 2).run pre0).map (fun s => (s.recvMgr.numFree, recvLookupStagingSlot s 0)) = some (1, some 0) ∧
     ((sys 4 2 2).run (pre0 ++ recvIssueH2d 0)).map (fun s =>
       (s.recvMgr.numFree, recvLookupStagingSlot s 0)) = some (1, some 0)) ∧
    (let blocked := traceSendCyclesEvs ++ [.recvPullReply true] ++ recvAcquireOk 0 ++
                      recvIssueH2d 0 ++ recvIssueH2d 1 ++ [.recvPushBegin, .recvAcquireBegin 2]
     (sys 4 2 2).run (blocked ++ [.recvAcquireWait 2 false]) = none ∧
     ((sys 4 2 2).run (blocked ++ recvFinishH2d 1 ++ [.recvAcquireWait 2 false, .recvAcquireFinish 2])).map (fun s =>
       (recvLookupStagingSlot s 1, recvLookupStagingSlot s 2)) = some (none, some 1)) ∧
    ((sys 4 2 2).run traceRecvCyclesEvs).map (fun s =>
      (s.recvMgr.numFree, s.pipe.recv.life.hasStaging, s.pipe.recv.published, s.decodeHbmB)) =
      some (2, false, some true, [.kv 0, .kv 1, .kv 2, .kv 3]) := by decide

/-- `RecvLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 1 (`uuid = 201`). -/
def traceRecvFailH2dWaiterEvs : List Ev :=
  streamProducer 3 ++ [.recvPullReply true] ++ recvIssueH2d 0 ++ recvIssueH2d 1 ++
  [.recvPushBegin, .recvAcquireBegin 2, .h2dReady 0, .h2dDone 0 false, .recvAcquireWait 2 false, .recvPushEnd] ++
  recvFinishH2d 1 ++ [.recvPublish]

theorem trace_recv_fail_h2d_with_waiter :
    (sys 3 2 2).run (streamProducer 3 ++ [.recvPullReply true, .h2dBegin 0]) = none ∧
    (let mid := streamProducer 3 ++ [.recvPullReply true] ++ recvIssueH2d 0 ++ recvIssueH2d 1 ++
                  [.recvPushBegin, .recvAcquireBegin 2, .h2dReady 0, .h2dDone 0 false,
                   .recvAcquireWait 2 false, .recvPushEnd]
     ((sys 3 2 2).run mid).map (fun s =>
       (s.recvMgr.shutdown, s.pipe.recv.issued, s.pipe.recv.life.hasStaging)) = some (true, 2, true)) ∧
    ((sys 3 2 2).run traceRecvFailH2dWaiterEvs).map (fun s =>
      (s.recvMgr.numFree, s.pipe.recv.life.hasStaging, s.pipe.recv.published)) =
      some (2, false, some false) := by decide

/-- `RecvLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 2 (`uuid = 202`). -/
def traceRecvFailPushAbortEvs : List Ev :=
  [.beginPull, .recvPullReply true, .recvPushBegin] ++ recvAcquireOk 0 ++ [.recvCancel, .recvPushEnd, .recvPublish]

theorem trace_recv_fail_push_abort_after_chunk :
    ((sys 3 2 2).run ([.beginPull, .recvPullReply true, .recvPushBegin] ++
        recvAcquireOk 0 ++ [.recvCancel])).map (fun s =>
      (s.recvMgr.numFree, s.pipe.recv.life.draining, s.pipe.recv.life.hasStaging)) = some (1, true, true) ∧
    ((sys 3 2 2).run traceRecvFailPushAbortEvs).map (fun s =>
      (s.recvMgr.numFree, s.recvMgr.leases, s.pipe.recv.life.hasStaging, s.pipe.recv.published)) =
      some (2, [], false, some false) := by decide

/-- `RecvLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 3 (`uuid = 203`). -/
def traceRecvFailTimeoutEvs : List Ev :=
  streamProducer 3 ++ [.recvPullReply true] ++ recvIssueH2d 0 ++ recvIssueH2d 1 ++
  [.recvPushBegin, .recvAcquireBegin 2, .recvAcquireWait 2 true, .recvPushEnd] ++
  recvFinishH2d 0 ++ recvFinishH2d 1 ++ [.recvPublish]

theorem trace_recv_fail_staging_timeout :
    (let mid := streamProducer 3 ++ [.recvPullReply true] ++ recvIssueH2d 0 ++ recvIssueH2d 1 ++
                  [.recvPushBegin, .recvAcquireBegin 2, .recvAcquireWait 2 true, .recvPushEnd]
     ((sys 3 2 2).run mid).map (fun s =>
       (s.pipe.recv.life.draining, s.pipe.recv.life.hasStaging, s.recvMgr.numFree)) = some (true, true, 0)) ∧
    ((sys 3 2 2).run traceRecvFailTimeoutEvs).map (fun s =>
      (s.recvMgr.numFree, s.pipe.recv.life.hasStaging, s.pipe.recv.published)) =
      some (2, false, some false) := by decide

/-- `SendLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 1 (`uuid = 401`). -/
def traceSendFailD2hWaiterEvs : List Ev :=
  [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++
  [.sendAcquireBegin, .d2hReady 0, .sendD2hEnd, .sendWake false,
   .sendAcquireDone false, .d2hReady 1, .sendD2hEnd, .sendPublish]

theorem trace_send_fail_d2h_with_waiter :
    sendLookupStagingSlot (init 3 2 2) 0 = none ∧
    (let mid := [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++
                  [.sendAcquireBegin, .d2hReady 0, .sendD2hEnd, .sendWake false, .sendAcquireDone false]
     ((sys 3 2 2).run mid).map (fun s =>
       (s.sendLoop, s.pipe.send.d2hIssued, s.pipe.send.h2hIssued, s.pipe.send.life.hasStaging)) =
       some (.exited, 2, 0, true)) ∧
    ((sys 3 2 2).run traceSendFailD2hWaiterEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.life.hasStaging, s.pipe.send.published)) =
      some (2, false, some false) := by decide

/-- `SendLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 2 (`uuid = 402`). -/
def traceSendFailH2hWaiterEvs : List Ev :=
  [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++ [.sendAcquireBegin] ++
  sendFinishD2hStartH2h 0 ++ sendFinishD2hStartH2h 1 ++
  [.h2hDone 0 false, .sendAcquireDone false, .sendWake false, .h2hDone 1 true, .sendPublish]

theorem trace_send_fail_h2h_with_waiter :
    (let mid := [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++ [.sendAcquireBegin] ++
                  sendFinishD2hStartH2h 0 ++ sendFinishD2hStartH2h 1 ++
                  [.h2hDone 0 false, .sendAcquireDone false, .sendWake false]
     ((sys 3 2 2).run mid).map (fun s =>
       (s.sendLoop, s.pipe.send.d2hIssued, s.pipe.send.life.hasStaging)) = some (.exited, 2, true)) ∧
    ((sys 3 2 2).run traceSendFailH2hWaiterEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.life.hasStaging, s.pipe.send.published)) =
      some (2, false, some false) := by decide

/-- `SendLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 3 (`uuid = 403`). -/
def traceSendFailTimeoutEvs : List Ev :=
  [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++
  [.sendAcquireBegin, .sendAcquireDone true, .d2hReady 0, .sendD2hEnd, .sendWake true,
   .d2hReady 1, .sendD2hEnd, .sendPublish]

theorem trace_send_fail_staging_timeout :
    (let mid := [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++ [.sendAcquireBegin, .sendAcquireDone true]
     ((sys 3 2 2).run mid).map (fun s =>
       (s.pipe.send.life.draining, s.pipe.send.life.hasStaging, s.sendMgr.numFree)) = some (true, true, 0)) ∧
    ((sys 3 2 2).run traceSendFailTimeoutEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.life.hasStaging, s.pipe.send.published)) =
      some (2, false, some false) := by decide

/-- `SendLifecycleTest.StagingLayerManagerReleasesAllResourcesAcrossFailureModes` subcase 4 (`uuid = 404`). -/
def traceSendFailCancelWaiterEvs : List Ev :=
  [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++
  [.sendAcquireBegin, .sendCancel, .sendAcquireDone false, .d2hReady 0, .sendD2hEnd, .sendWake true,
   .d2hReady 1, .sendD2hEnd, .sendPublish]

theorem trace_send_fail_cancel_with_waiter :
    (let mid := [.beginPull, .sendStart] ++ sendIssueD2h ++ sendIssueD2h ++
                  [.sendAcquireBegin, .sendCancel, .sendAcquireDone false]
     ((sys 3 2 2).run mid).map (fun s =>
       (s.sendLoop, s.pipe.send.d2hIssued, s.pipe.send.life.hasStaging)) = some (.exited, 2, true)) ∧
    ((sys 3 2 2).run traceSendFailCancelWaiterEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.life.hasStaging, s.pipe.send.published)) =
      some (2, false, some false) := by decide

/-! ## Lock-gap interleavings, zero-slot rejection, and recycle -/

def traceSendPostAcquireCancelEvs : List Ev :=
  [.beginPull, .sendStart, .sendAcquireBegin, .sendAcquireDone false,
   .sendCancel, .sendD2hBegin, .sendWake false, .sendPublish]

theorem trace_send_post_acquire_cancel :
    ((sys 1 1 1).run traceSendPostAcquireCancelEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.d2hIssued, s.pipe.send.published)) = some (1, 0, some false) := by decide

def traceRecvPostAcquireCancelEvs : List Ev :=
  [.beginPull, .recvPullReply true, .recvAcquireBegin 0, .recvAcquireWait 0 false,
   .recvCancel, .recvAcquireFinish 0, .recvPublish]

theorem trace_recv_post_acquire_cancel :
    ((sys 1 1 1).run traceRecvPostAcquireCancelEvs).map (fun s =>
      (s.recvMgr.numFree, s.recvPhaseL, s.pipe.recv.published)) = some (1, [.aborted], some false) := by decide

def traceRecvDrainingH2dBeginEvs : List Ev :=
  streamProducer 1 ++ [.recvPullReply true, .recvPushBegin] ++ recvAcquireOk 0 ++
  [.land 0, .recvCancel, .h2dBegin 0, .recvPushEnd, .recvPublish]

theorem trace_recv_draining_h2d_begin :
    ((sys 1 1 1).run traceRecvDrainingH2dBeginEvs).map (fun s =>
      (s.recvMgr.numFree, s.recvPhaseL, s.pipe.recv.issued, s.pipe.recv.published)) =
      some (1, [.aborted], 0, some false) := by decide

def traceSendD2hDispatchFailEvs : List Ev :=
  [.beginPull, .sendStart, .sendAcquireBegin, .sendAcquireDone false,
   .sendD2hBegin, .sendD2hIssue false, .sendWake false, .sendPublish]

def traceRecvH2dDispatchFailEvs : List Ev :=
  streamProducer 1 ++ [.recvPullReply true, .recvPushBegin] ++ recvAcquireOk 0 ++
  [.land 0, .h2dBegin 0, .h2dIssue 0 false, .recvPushEnd, .recvPublish]

theorem trace_sync_dispatch_failures_release_slots :
    ((sys 1 1 1).run traceSendD2hDispatchFailEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.published)) = some (1, some false) ∧
    ((sys 1 1 1).run traceRecvH2dDispatchFailEvs).map (fun s =>
      (s.recvMgr.numFree, s.pipe.recv.published)) = some (1, some false) := by decide

def traceZeroSendSlotsEvs : List Ev :=
  [.beginPull, .sendStart, .sendAcquireBegin, .sendAcquireDone false, .sendWake false, .sendPublish]

def traceZeroRecvSlotsEvs : List Ev :=
  [.beginPull, .recvPullReply true, .recvAcquireBegin 0, .recvAcquireWait 0 false, .recvPublish]

theorem trace_zero_slots_fail_cleanly :
    ((sys 1 0 1).run traceZeroSendSlotsEvs).map (fun s =>
      (s.sendMgr.numFree, s.pipe.send.published)) = some (0, some false) ∧
    ((sys 1 1 0).run traceZeroRecvSlotsEvs).map (fun s =>
      (s.recvMgr.numFree, s.pipe.recv.published)) = some (0, some false) := by decide

def traceUnregisteredPollAndRecycleEvs : List Ev :=
  [.pullWait, .notifyForRead] ++ streamProducer 1 ++
  [.reclaim, .reseatPrefillStaging, .recvPullReply true] ++ recvIssueH2d 0 ++
  [.h2dReady 0, .recvPollReady, .h2dDone 0 true, .recvPublish, .reseatDecodeStaging]

theorem trace_unregistered_poll_and_recycle :
    ((sysUnregistered 1 1 1).run traceUnregisteredPollAndRecycleEvs).map (fun s =>
      (s.pipe.reclaimed, s.prefillSlots, s.decodeSlots, s.pipe.recv.published, s.decodeHbmB)) =
      some (true, [.junk], [.junk], some true, [.kv 0]) := by decide

/-! ## Mutants and counterexample traces -/

/-- **Mutant 1 (early producer slot release):** releasing layer `l`'s host slot at
`sendH2hIssue` instead of `h2hDone` lets layer 1 overwrite slot 0 before layer 0's
H2H push reads it, corrupting `decodeHbmB` to `[.kv 1, .kv 1]`. -/
def sendH2hIssueEarlyRelease (s : BoundedPipeline) : Option BoundedPipeline :=
  let l := s.pipe.send.h2hIssued
  (Pipeline.step s.pipe (.send .h2hIssue)).map fun p =>
    if s.pipe.send.life.draining = true then
      { s with pipe := p, sendMgr := releaseSync s.pipe.send.life p.send.life l s.sendMgr }
    else
      match s.sendMgr.lookup l with
      | some slot =>
        if slot < s.prefillSlots.length then { s with pipe := p, sendMgr := (s.sendMgr.release l).2 }
        else { s with pipe := p, slotFault := true }
      | none => { s with pipe := p, slotFault := true }

def h2hDoneEarlyRelease (s : BoundedPipeline) (l : Nat) (ok : Bool) : Option BoundedPipeline :=
  (Pipeline.step s.pipe (.h2hDone l ok)).map fun p =>
    if ok then
      { s with pipe := p, wireB := s.wireB.set l (s.prefillSlots.getD 0 .junk),
               sendMgr := syncMgr s.pipe.send.life p.send.life s.sendMgr }
    else { s with pipe := p, sendMgr := syncMgr s.pipe.send.life p.send.life s.sendMgr }

def sysEarlySendRelease : System BoundedPipeline Ev :=
  ⟨init 2 1 1, fun s e => match e with
    | .pipe (.send .h2hIssue) => sendH2hIssueEarlyRelease s
    | .pipe (.h2hDone l ok) => h2hDoneEarlyRelease s l ok
    | e => step s e⟩

theorem mutant_early_send_release_corrupts_hbm :
    let tr :=
      [.beginPull, .sendStart] ++ sendIssueD2h ++ sendFinishD2hStartH2h 0 ++
      sendIssueD2h ++ [.d2hReady 1, .sendD2hEnd] ++
      [.h2hDone 0 true, .sendWake true, .sendH2hIssue, .sendNext, .h2hDone 1 true, .sendPublish] ++
      [.recvPullReply true] ++ recvIssueH2d 0 ++ recvFinishH2d 0 ++
      recvIssueH2d 1 ++ recvFinishH2d 1 ++ [.recvPublish]
    (sysEarlySendRelease.run tr).map (fun s => (s.pipe.recv.published, s.decodeHbmB, violates s)) =
      some (some true, [.kv 1, .kv 1], true) := by decide

/-- **Mutant 2 (early consumer slot release):** releasing layer `l`'s host slot at
`h2dIssue` instead of `h2dDone` lets layer 1 land `.kv 1` into slot 0 before
layer 0's H2D copy reads it, corrupting `decodeHbmB`. -/
def h2dIssueEarlyRelease (s : BoundedPipeline) (l : Nat) (ok : Bool) : Option BoundedPipeline :=
  if s.recvPhaseL[l]? = some .h2dPending then
    (Pipeline.step s.pipe (.h2dIssue l ok)).map fun p =>
      if s.pipe.recv.life.done || s.pipe.recv.life.draining || !ok then
        { s with pipe := p, recvMgr := releaseSync s.pipe.recv.life p.recv.life l s.recvMgr,
                 recvPhaseL := s.recvPhaseL.set l .aborted }
      else { s with pipe := p, recvMgr := (s.recvMgr.release l).2, recvPhaseL := s.recvPhaseL.set l .h2dRunning }
  else none

def h2dReadyEarlyRelease (s : BoundedPipeline) (l : Nat) : Option BoundedPipeline :=
  if s.recvPhaseL[l]? = some .h2dRunning ∧ s.pipe.h2dReadyL[l]? = some false then
    (Pipeline.step s.pipe (.h2dReady l)).map fun p =>
      { s with pipe := p, decodeHbmB := s.decodeHbmB.set l (s.decodeSlots.getD 0 .junk) }
  else none

def sysEarlyRecvRelease : System BoundedPipeline Ev :=
  ⟨init 2 2 1, fun s e => match e with
    | .pipe (.h2dIssue l ok) => h2dIssueEarlyRelease s l ok
    | .pipe (.h2dReady l) => h2dReadyEarlyRelease s l
    | e => step s e⟩

theorem mutant_early_recv_release_corrupts_hbm :
    let tr :=
      streamProducer 2 ++ [.recvPullReply true] ++ recvIssueH2d 0 ++
      [.recvPushBegin] ++ recvAcquireOk 1 ++ [.land 1] ++ recvFinishH2d 0 ++
      [.h2dBegin 1, .h2dIssue 1 true, .recvNetAccount, .recvPushEnd] ++ recvFinishH2d 1 ++ [.recvPublish]
    (sysEarlyRecvRelease.run tr).map (fun s => (s.pipe.recv.published, s.decodeHbmB, violates s)) =
      some (some true, [.kv 1, .kv 1], true) := by decide

/-- **Mutant 3 (missing `ReleaseAll()` on settlement):** omitting `ReleaseAll()`
when `done` flips `false → true` permanently leaks an uncompleted push's slot. -/
def syncMgrNoReleaseAll (oldLife newLife : TpuSyncVerify.Transfer.Lifecycle)
    (m : StagingLayerManager) : StagingLayerManager :=
  if !oldLife.draining && newLife.draining then m.doShutdown else m

def sysNoReleaseAll : System BoundedPipeline Ev :=
  ⟨init 1 1 1, fun s e => match e with
    | .pipe (.recv .pushEnd) =>
      if 0 < s.activeStreams then
        (Pipeline.step s.pipe (.recv .pushEnd)).map fun p =>
          { s with pipe := p, activeStreams := s.activeStreams - 1,
                   recvMgr := syncMgrNoReleaseAll s.pipe.recv.life p.recv.life s.recvMgr }
      else none
    | e => step s e⟩

theorem mutant_missing_release_all_leaks_slot :
    let tr := [.beginPull, .recvPullReply true, .recvPushBegin] ++ recvAcquireOk 0 ++ [.recvCancel, .recvPushEnd]
    (sysNoReleaseAll.run tr).map (fun s =>
      (s.pipe.recv.life.done, s.recvMgr.numFree, s.recvMgr.leases, violates s)) =
      some (true, 0, [(0, 0)], true) := by decide

#guard (match ModelCheck.check sysNoReleaseAll (events 1) violates 10 with
        | .counterexample _ => true | _ => false)

end TpuSyncVerify.Transfer.PrefillDecode.LayerStaging

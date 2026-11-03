import TpuSyncVerify.Common.System
import TpuSyncVerify.Common.ModelCheck
import TpuSyncVerify.Common.ListAux
import TpuSyncVerify.Transfer.Session
import TpuSyncVerify.Transfer.Caller
import TpuSyncVerify.Transfer.PrefillDecode.Receive
import TpuSyncVerify.Transfer.PrefillDecode.ReceivePoll
import TpuSyncVerify.Transfer.PrefillDecode.Send
import TpuSyncVerify.Transfer.PrefillDecode.Pipeline
import TpuSyncVerify.Transfer.PrefillDecode.MultiRequest
import TpuSyncVerify.Transfer.PrefillDecode.BlockOrdering
import TpuSyncVerify.Transfer.PrefillDecode.PipelineChecks
import TpuSyncVerify.Transfer.PrefillDecode.PeerIsolation
import TpuSyncVerify.Transfer.PrefillDecode.UuidTable
import TpuSyncVerify.Transfer.PrefillDecode.Transport
import TpuSyncVerify.Transfer.Reshard.Receive
import TpuSyncVerify.Transfer.Reshard.Send
import TpuSyncVerify.Transfer.Reshard.Pipeline
import TpuSyncVerify.Transfer.Reshard.MultiRequest
import TpuSyncVerify.Transfer.Reshard.Coverage
import TpuSyncVerify.Transfer.Reshard.PoolStaging
import TpuSyncVerify.Transfer.Reshard.Registry
import TpuSyncVerify.Transfer.Reshard.Coordinator

/-!
# TpuSyncVerify

Lean 4 models of TPU Sync's KV-cache data paths and machine-checked proofs
about them. This module imports every model; see `README.md` for what is
modelled, the directory layout, and how the C++ is cited.

* `Common` — transition-system core, list lemmas, bounded model checker.
* `Transfer` — `Session` (settle protocol shared by every transfer session),
  `Caller` (engine caller contract above `poll_stats()`), `PrefillDecode/`,
  `Reshard/`.
-/

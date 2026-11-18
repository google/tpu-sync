# Formal Verification of TPU Sync (Lean 4)

TPU Sync is a KV-cache orchestration, host-memory offload and cross-node
transfer library for TPUs. Its transfer data paths run on asynchronous
callbacks and lock-scoped state machines across TPU HBM, host DRAM staging
pools, and network peers. This directory holds Lean 4 models of those transfer
data paths and machine-checked proofs about them.

Each result below is a theorem about a model, and holds under the numbered
assumptions the model's module lists. The two subsystem documents tabulate the
assumptions and say which are discharged by another Lean module, which follow
from C++ control flow, and which are left to the caller or the configuration;
they also list what is not modelled.

## What is modelled

**Prefill-to-decode transfer** — [prefill_decode.md](docs/transfer/prefill_decode.md)
- C++: `TransferSendSession`, `TransferReceiveSession`, `StagingLayerManager`,
  `KVCacheManagerWithTransfer`, the receiver side of `BlockTransport`
  (`BlockTransportServerPushSession`).
- Lean: [`Transfer/Session.lean`](TpuSyncVerify/Transfer/Session.lean),
  [`Transfer/LayerStaging.lean`](TpuSyncVerify/Transfer/LayerStaging.lean),
  [`Transfer/PrefillDecode/`](TpuSyncVerify/Transfer/PrefillDecode/).
- Proved: data correctness on publication (both full host mirror and bounded
  host layer staging under `num_staging_layers`), HBM and staging
  safety, host-slot conservation/exclusivity and `1`-slot sufficiency,
  drain-to-settle termination, cross-peer isolation.

**Pool resharding** — [reshard.md](docs/transfer/reshard.md)
- C++: `ReshardSendSession`, `ReshardReceiveSession`, `PoolStagingStorage`,
  `RequestBlockRegistry`, `ReshardCoordinator`, and multi-window sub-plan
  chains (`PlanSubTransfers`, `ReshardCoordinator::StartChain`,
  `KVCacheManagerWithTransfer::SubPlanChainProgress`).
- Lean: [`Transfer/Reshard/`](TpuSyncVerify/Transfer/Reshard/).
- Proved: data correctness including sub-block coverage and aliased-page
  ordering, staging slot conservation, registry claim safety,
  arm-before-dispatch, termination, and one worker-level outcome per
  sub-plan chain (the last under caller assumptions the C++ does not check
  and only for executions outside two timing windows the C++ permits;
  [reshard.md](docs/transfer/reshard.md) lists them).

**Engine caller contract** — [prefill_decode.md](docs/transfer/prefill_decode.md) §Boundaries, [reshard.md](docs/transfer/reshard.md) §Boundaries
- C++ / Python: `NumaAwareKVCacheManager` (`tpu_sync/frameworks/jax/`), and
  vLLM's scheduler and `KVOutputAggregator` through tpu-inference's and
  vllm-torchtpu's connectors, above `poll_stats()`.
- Lean: [`Transfer/Caller.lean`](TpuSyncVerify/Transfer/Caller.lean).
- Proved: a connector and sub-manager barrier that forward one settled outcome
  per transfer leak no held block, never free HBM while a worker or sub-manager
  is still in flight, and never decode on unwritten KV.

## Layout

```text
TpuSyncVerify.lean            imports every model
TpuSyncVerify/
  Common/                     transition-system core, list lemmas, bounded model checker
  Transfer/Session.lean       settle protocol shared by every transfer session
  Transfer/LayerStaging.lean  per-session bounded host layer buffer manager (StagingLayerManager)
  Transfer/Caller.lean        engine caller contract above poll_stats() shared by both transfer paths
  Transfer/PrefillDecode/     prefill-to-decode sessions, pipeline, bounded layer staging, block
                              ordering, uuid table, peer isolation, transport accounting
  Transfer/Reshard/           pool-resharding sessions, pipeline, staging arena, registry,
                              coordinator, sub-plan chains
docs/
  conventions.md              how the models are written (one executable step, ghost
                              fields, replay traces, mutants) and how the C++ is cited
  transfer/                   per-subsystem C++ <-> Lean docs: states, events,
                              assumptions, proved theorems
tools/
  resolve_anchors.py          checks (--check) or rewrites (--apply) the line spans in the
                              source-map tables, and compares them between two commits
                              (--diff)
lakefile.toml, lake-manifest.json, lean-toolchain
```

## How the C++ is cited

Prose and docstrings cite C++ symbols (`Class::Method`), code fragments, and
GTest names (`Suite.TestName`); exact line spans live in per-file
`<!-- source-map:begin --> ... <!-- source-map:end -->` tables. Those spans were
last resolved at commit **`521871ae`**;
[`tools/resolve_anchors.py`](tools/resolve_anchors.py) `--check` re-verifies
them against the working tree. The external callers that `Caller.lean`
cites are pinned separately: tpu-inference `f366ef52`, vllm-torchtpu
`9725110`, and vLLM `e7bb7e2f`.

## Building & Verifying

Requires [elan](https://github.com/leanprover/elan); the toolchain pinned in
[`lean-toolchain`](lean-toolchain) (Lean 4.33.0) is fetched automatically.

```sh
cd tpu_sync/formal_verification
lake build
python3 tools/resolve_anchors.py --check
```

`lake build` checks every `theorem` with Lean's kernel and evaluates every
`#guard` (the model-checker searches; a `#guard`
is an evaluated check, not a theorem). The corpus has no `sorry`, declares no
axiom and has no dependency beyond the Lean toolchain.

`resolve_anchors.py --check` re-resolves every source-map table against the
C++ in the working tree and reports stray raw line citations outside the
tables. It writes nothing and exits non-zero on any mismatch, or if it finds no
file to check. It locates the repository from the current directory and
otherwise from its own path, so it needs the C++ sources under `tpu_sync/`
alongside this directory. `--apply` rewrites the line spans after a C++ change;
`python3 tools/resolve_anchors.py --help` describes every mode.


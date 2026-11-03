# Formal Verification of TPU Sync (Lean 4)

TPU Sync is a KV-cache orchestration, host-memory offload and cross-node
transfer library for TPUs. Its data paths run on asynchronous callbacks and
lock-scoped state machines across TPU HBM, host DRAM staging pools, and network
peers. This directory holds Lean 4 models of those data paths and
machine-checked proofs about them.

## What is modelled

**Prefill-to-decode transfer** — [prefill_decode.md](docs/transfer/prefill_decode.md)
- C++: `TransferSendSession`, `TransferReceiveSession`,
  `KVCacheManagerWithTransfer`, the receiver side of `BlockTransport`.
- Lean: [`Transfer/Session.lean`](TpuSyncVerify/Transfer/Session.lean),
  [`Transfer/PrefillDecode/`](TpuSyncVerify/Transfer/PrefillDecode/).
- Proved: data correctness on publication, HBM and staging safety,
  drain-to-settle termination, cross-peer isolation.

**Engine caller contract** — [prefill_decode.md](docs/transfer/prefill_decode.md) §Boundaries, [reshard.md](docs/transfer/reshard.md) §Boundaries
- Python: vLLM's scheduler through tpu-inference's and vllm-torchtpu's
  connectors, above `poll_stats()`.
- Lean: [`Transfer/Caller.lean`](TpuSyncVerify/Transfer/Caller.lean).
- Proved: a connector that forwards every publication leaks no held block and
  never decodes on unwritten KV.

**Pool resharding** — [reshard.md](docs/transfer/reshard.md)
- C++: `ReshardSendSession`, `ReshardReceiveSession`, `PoolStagingStorage`,
  `RequestBlockRegistry`, `ReshardCoordinator`.
- Lean: [`Transfer/Reshard/`](TpuSyncVerify/Transfer/Reshard/).
- Proved: data correctness including sub-block coverage and aliased-page
  ordering, staging slot conservation, registry claim safety,
  arm-before-dispatch, termination.

## Layout

```text
TpuSyncVerify.lean            imports every model
TpuSyncVerify/
  Common/                     transition-system core, list lemmas, bounded model checker
  Transfer/Session.lean       settle protocol shared by every transfer session
  Transfer/Caller.lean        engine caller contract above poll_stats() shared by both transfer paths
  Transfer/PrefillDecode/     prefill-to-decode sessions, pipeline, transport
  Transfer/Reshard/           pool-resharding sessions, staging arena, registry, coordinator
docs/
  conventions.md              how the models are written (one executable step, ghost
                              fields, replay traces, mutants) and how the C++ is cited
  transfer/                   per-subsystem C++ <-> Lean docs: states, events,
                              assumptions, proved theorems
tools/
  resolve_anchors.py          resolves and checks the source-map tables
lakefile.toml, lake-manifest.json, lean-toolchain
```

## How the C++ is cited

Prose and docstrings cite C++ symbols (`Class::Method`), code fragments, and
GTest names (`Suite.TestName`); exact line spans live in per-file
`<!-- source-map:begin --> ... <!-- source-map:end -->` tables. Those spans were
last resolved at commit **`c9bc124`**;
[`tools/resolve_anchors.py`](tools/resolve_anchors.py) `--check` re-verifies
them against the working tree. The external callers that `Caller.lean`
cites are pinned separately: tpu-inference `7ed46a6`, vllm-torchtpu `9725110`,
and vLLM `6185d73`.

## Building & Verifying

Requires [elan](https://github.com/leanprover/elan); the toolchain pinned in
[`lean-toolchain`](lean-toolchain) (Lean 4.33.0) is fetched automatically.

```sh
cd tpu_sync/formal_verification
lake build
python3 tools/resolve_anchors.py --check
```

A clean build takes about half a minute; every `theorem` is machine-checked,
every `#guard` bounded search is executed, and `resolve_anchors.py --check`
verifies all C++ source-map tables and checks for stray raw line citations.

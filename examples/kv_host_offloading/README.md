# Single-host KV host-offloading (tpu-raiden)

Offload KV-cache prefix blocks from TPU HBM to a **host-RAM pool** on a single TPU
VM, using tpu-raiden as the device↔host DMA engine. One vLLM engine runs locally
and acts as both producer and consumer (`kv_role=kv_both`):

- On a prefix-cache **miss**, the engine computes the prefix and **saves** its full
  KV blocks to the host pool (device→host, `d2h_auto_allocate`).
- On a later **hit**, the engine **loads** those blocks back from host RAM
  (host→device, `h2d`) instead of recomputing them.

This extends vLLM's on-device prefix cache with a much larger second tier in host
RAM. The connector lives in tpu-inference
(`tpu_inference.offload.raiden_offload_connector`); the physical transfers and the
host block pool are provided by tpu-raiden.

```
              ┌──────────────────────── single TPU VM ───────────────────────┐
  client ───► │  vLLM engine :8000  (RaidenOffloadConnector, kv_both)         │
 (benchmark)  │        │  ▲                                                    │
              │   d2h  │  │  h2d           ┌───────────────────────────────┐  │
              │  (save)│  │ (load)  ◄────► │   host-RAM KV pool (tpu-raiden)│  │
              │        ▼  │                │   TPU_OFFLOAD_NUM_CPU_CHUNKS    │  │
              └───────────┴────────────────┴───────────────────────────────┘  │
              └──────────────────────────────────────────────────────────────┘
```

## Prerequisites

- A single TPU VM with enough free chips for `TENSOR_PARALLEL_SIZE` (default 8) and a
  **python3.12 venv**.
- The venv must have **`tpu_raiden`**, **`vllm`**, and **`tpu_inference`** importable
  (see Step 1). These scripts do **not** activate any environment — activate your
  venv yourself before running them.
- The benchmark (`benchmark.sh`) uses tpu-inference's `benchmark_serving.py`, which
  imports **`evaluate`** and **`nltk`** (installed by `setup.sh`).
- Hugging Face access for the model (e.g. `export HF_TOKEN=...`) if it is gated.

---

## Step 1 — Install

In an **activated** python3.12 venv:

1. Install **tpu_raiden** via one of:

   ```bash
   # Build from source (generally available) -- from the repo root:
   ./build.sh jax

   # Or install the pre-built wheel from PyPI (`--pre` until the first stable release):
   pip install --pre tpu-sync-jax
   ```

   The run scripts pick up whichever install you did (see `raiden_env.sh`).

2. Install **vllm** + **tpu_inference** (the offload connector ships in
   tpu_inference), plus the benchmark client's `evaluate`/`nltk` deps, via this
   example's own installer:

   ```bash
   bash setup.sh
   ```

   `setup.sh` clones vLLM + tpu-inference at pinned commits (which include the
   RaidenOffloadConnector) into a hidden in-tree `.src/` and installs them editable.
   Override `VLLM_COMMIT` / `TPU_INFERENCE_COMMIT` to use different versions.

## Step 2 — Run the demo

```bash
bash run_offload.sh
```

This starts the offload server, waits for it to be healthy, runs the benchmark
against it (`:8000`), prints the results and an offload-activity summary (the
**External prefix cache hit rate** is the fraction served from the host pool), then
tears the server down.

Logs land in `tmp/<timestamp>/` (`server.log`, `benchmark.log`).

> Assumes the local TPU is free — there is no pre-run TPU cleanup. The script
> stops only the server it launched.

---

## Configuration

All knobs are environment variables (with defaults); set them before `run_offload.sh`.

**Model / serving** (`server.sh`):

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL` | `Qwen/Qwen3-32B` | Served model (must match `benchmark.sh`). |
| `TENSOR_PARALLEL_SIZE` | `8` | TPU chips to shard across. |
| `GPU_MEMORY_UTILIZATION` | `0.8` | Device HBM fraction for the KV cache. |
| `PORT` | `8000` | Server port. |

**Offloading** (`server.sh`):

| Var | Default | Meaning |
|-----|---------|---------|
| `TPU_OFFLOAD_NUM_CPU_CHUNKS` | `8192` | Host KV pool size, in blocks. |
| `TPU_OFFLOAD_DECODE_SAVE` | `0` | Also save KV produced during decode. |

**Benchmark** (`benchmark.sh`): `RANDOM_PREFIX_LEN` (32000), `RANDOM_INPUT_LEN` (128),
`RANDOM_OUTPUT_LEN` (128), `NUM_PROMPTS` (75), `REQUEST_RATE` (4). The shared prefix
is what gets offloaded and re-loaded across requests. The benchmark runs
tpu-inference's `benchmark_serving.py` (found via the installed `tpu_inference`;
override the path with `BENCH_SCRIPT`).

Example — smaller model on a single chip:

```bash
MODEL=Qwen/Qwen3-8B TENSOR_PARALLEL_SIZE=1 GPU_MEMORY_UTILIZATION=0.5 bash run_offload.sh
```

---

## Notes

- **Pool sizing.** The logical store capacity and the physical host pool are both
  sized from `TPU_OFFLOAD_NUM_CPU_CHUNKS`. Keep it comfortably **above the
  benchmark's distinct-block working set** so the pool serves hits without needing
  to evict. A larger pool also means a larger host allocation at startup (which can
  add a minute), so size it to your host RAM.
- **Files.** `server.sh` (engine + offload connector), `benchmark.sh` (shared-prefix
  benchmark), `run_offload.sh` (one-shot orchestrator), `raiden_env.sh` (makes
  `tpu_raiden` importable for both wheel and build-from-source installs).
</content>

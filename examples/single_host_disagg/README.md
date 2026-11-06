# Single-host disaggregated serving (tpu-raiden KV transfer)

Disaggregated LLM serving on a **single TPU VM**, using tpu-raiden as the
high-bandwidth KV-cache DMA engine. Both engines run locally on **distinct TPU
chips**:

- **Prefill** (KV producer) — vLLM server on chip 0, port `:8400`.
- **Decode** (KV consumer) — vLLM server on chip 1, port `:9400`.
- **Router/proxy** on `:8000` fans each request to prefill then decode; the decode
  pulls the prefill's KV-cache via tpu-raiden.

Because both engines share `localhost`, each binds a **distinct** KV control /
side-channel port (prefill `9100/9600`, decode `9101/9601`); the decode pulls from
the prefill's advertised port, threaded into each request by the proxy.

```
                 ┌──────────────────────────── single TPU VM ───────────────────────────┐
  client ──────► │  router/proxy :8000 ──► prefill vLLM :8400 (KV producer, chip 0) ┐   │
 (bm.sh)         │                                ▲                                 │   │
                 │                                │ KV pull (localhost, 9100/9600)  │   │
                 │         decode vLLM :9400 (KV consumer, chip 1) ─────────────────┘   │
                 └──────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- A single TPU VM with at least 2 free chips, and a python3.12 venv.
- `tpu_raiden` installed into that venv via **either** path (see Step 1).

---

## Step 1 — Install

In an activated python3.12 venv, install tpu-raiden via one of:

```bash
# Build from source (generally available) -- from the repo root:
./build.sh jax

# Or install the pre-built wheel from PyPI (`--pre` until the first stable release):
pip install --pre tpu-sync-jax
```

Then, from this directory, with the venv active:

```bash
bash setup.sh
```

`setup.sh` clones vLLM + tpu-inference (at pinned commits) into a hidden in-tree
`.src/` and installs them editable into the venv. (It does not install tpu-raiden;
the run scripts pick up whichever install you did above — see `raiden_env.sh`.)

## Step 2 — Run the demo

```bash
bash run_all.sh
```

This starts the router + prefill (chip 0) + decode (chip 1), waits for both
engines to be healthy, runs the benchmark against the router (`:8000`), prints the
results, then tears down everything it started.

Logs land in `tmp/<timestamp>/` (`router.log`, `prefill.log`, `decode.log`,
`bm.log`).

> Assumes the local TPUs are free — there is no pre-run TPU cleanup. The script
> stops only the servers it launched.

---

## Notes

- Model defaults to `Qwen/Qwen3-8B` (set `MODEL` consistently across
  `prefill.sh` / `decode.sh` / `bm.sh` to change it).
- For a multi-machine version (one server per VM, two chips), see
  [`../multihost_disagg/`](../multihost_disagg/).
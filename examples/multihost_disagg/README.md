# Multi-host disaggregated serving (tpu-raiden KV transfer)

Disaggregated LLM serving across **two TPU VMs**, using tpu-raiden as the
high-bandwidth KV-cache DMA engine. Each VM uses **all 4 chips** (`tpu7x-standard-4t`,
`--tensor-parallel-size 8`) and serves `Qwen/Qwen3-32B`:

- **Prefill VM** (KV producer) — runs the prefill vLLM server, the proxy/router,
  and the benchmark client. This is where you run `run_all.sh`.
- **Decode VM** (KV consumer) — runs the decode vLLM server. Started over SSH by
  `run_all.sh`; it pulls KV from the prefill VM over the network.

The prefill connector advertises its own IP(`dist_utils.get_host_ip()`),
and the proxy threads it into each request, so the
decode VM learns where to pull from. The only IP you provide is the decode VM's,
so the router can route HTTP requests and `run_all.sh` can SSH in.

```
          ┌──────────────── Prefill VM (4 chips, TP=8) ─────────────────┐
 client ─►│  router/proxy :8000 ──► prefill vLLM :8400 (KV producer)    │
 (bm.sh)  └────────┬──────────────────────────────▲─────────────────────┘
                   │ HTTP route to decode         │ KV pull over network (:9100)
          ┌────────▼──────────────────────────────┴─────────────────────────┐
          │  decode vLLM :9400 (KV consumer) Decode VM (4 chips, TP=8)      │
          └─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Two TPU VMs (e.g. `tpu7x-standard-4t`, 4 chips each) that can reach each other
  on the internal network (on GCP the default `allow-internal` firewall rule
  covers this; otherwise allow traffic between the two VMs on ports **9400**
  (decode HTTP) and **9100** (KV transfer)).
- The two VMs are assumed to **mirror each other**: same user, same repo path,
  same venv path. (All overridable — see [Step 4](#step-4--run-the-demo).)

---

## Step 1 — Install on BOTH VMs

On **each** VM, in a python3.12 venv, install tpu-raiden, then run `setup.sh`.

Install tpu-raiden via one of:

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

## Step 2 — Find the decode VM's IP

**On the decode VM**, print its internal IP:

```bash
hostname -I | awk '{print $1}'
```

(On GCP you can also get it from the prefill VM without logging in:
`gcloud compute instances describe <decode-vm-name> --zone <zone> \
  --format='get(networkInterfaces[0].networkIP)'`.)

Call this `<DECODE_IP>`.

## Step 3 — Verify passwordless SSH (prefill VM → decode VM)

`run_all.sh` starts the decode server over SSH, so **from the prefill VM** this
must work without a password prompt:

```bash
ssh <DECODE_IP> hostname
```

It should print the decode VM's hostname. If it fails with `Permission denied
(publickey)`, install the key manually. NOTE: `ssh-copy-id` does **not** work on
GCP — VMs disallow password auth, so it can't log in to install the key.

```bash
# 1. On the PREFILL VM: create a key if needed, then print the public key.
ssh-keygen -t ed25519 -N ''          # skip if ~/.ssh/id_ed25519 already exists
cat ~/.ssh/id_ed25519.pub
```

```bash
# 2. On the DECODE VM (get in via `gcloud compute ssh` or the Cloud Console SSH),
#    append that public key to authorized_keys:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo '<paste the prefill VM public key line here>' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Then re-run `ssh <DECODE_IP> hostname` from the prefill VM — it should now
succeed without a prompt. (The `~/.ssh` 700 / `authorized_keys` 600 permissions
matter; sshd silently rejects the key otherwise. If your project enforces **OS
Login**, manual `authorized_keys` edits get overwritten — instead run
`gcloud compute instances add-metadata <decode-vm> --zone <zone> \
  --metadata ssh-keys="$USER:$(cat ~/.ssh/id_ed25519.pub)"`.)

## Step 4 — Run the demo

From the **prefill VM** (venv active), pass the decode VM's IP:

```bash
bash run_all.sh <DECODE_IP>
```

This starts the router + prefill (local) and the decode server (remote, over
SSH), waits for both to be healthy, runs the benchmark against the router
(`:8000`), prints the results, and tears down everything it started.

Logs land in `tmp/<timestamp>/` (`decode.log` is streamed from the decode VM).

### Overrides (if the two VMs are not identical)

`run_all.sh` reuses the prefill VM's active venv path and this directory's path on
the decode VM. Override via env vars if they differ:

| Variable | Default | Meaning |
|---|---|---|
| `RAIDEN_VENV` | `$VIRTUAL_ENV` | venv path to activate on the decode VM |
| `REMOTE_DIR` | this script's dir | path to `multihost_disagg/` on the decode VM |
| `SSH_USER` | current user | SSH user for the decode VM |
| `SSH_OPTS` | `-o BatchMode=yes …` | extra `ssh` options |

Example:

```bash
RAIDEN_VENV=/home/me/.venv312 REMOTE_DIR=/home/me/tpu-raiden/examples/multihost_disagg \
  bash run_all.sh <DECODE_IP>
```

---

## Notes

- Model defaults to `Qwen/Qwen3-32B` and `--tensor-parallel-size 8` (all 4 chips
  per VM). Set `MODEL` consistently across `prefill.sh` / `decode.sh` / `bm.sh`
  (and adjust `--tensor-parallel-size`) to change it.
- For a single-machine version (both engines on one VM, two chips), see
  [`../single_host_disagg/`](../single_host_disagg/).
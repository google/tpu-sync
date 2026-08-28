# TPU Sync

> [!IMPORTANT]
> TPU Raiden codebase has graduated into TPU Sync.

> [!IMPORTANT]
> TPU Sync is currently under active development and is not yet recommended for general use.
> If you are interested in adopting this library, please reach out to the owners first to discuss compatibility, or proceed at your own risk.

## 🚀 What is TPU Sync?

TPU Sync is a high-performance Key-Value (KV) cache orchestration, multi-tiered memory management, and cross-node data transfer library purpose-built to accelerate large-scale Large Language Model (LLM) inference and Reinforcement Learning (RL) training loops on Cloud TPU infrastructure.

As LLMs scale toward massive context windows and complex, multi-turn AI agents, the bottleneck in the underlying AI infrastructure has shifted from raw compute power to memory orchestration. Managing the key-value (KV) cache has become one of the most prominent bottlenecks driving up latency and leaving expensive compute cycles underutilized.

To address these bottlenecks, we are introducing TPU Sync — an open-source, native C++ framework for high-performance data transfer and KV cache management on TPUs. TPU Sync functions as the low-level, zero-copy memory and DMA foundation executing directly beneath Python-based ML orchestrators. By mapping directly onto hardware-accelerated memory layers, TPU Sync bridges HBM, Local System Host DRAM, and Remote Disaggregated TPU Nodes, unlocking bare-metal throughput and context capacity with minimal latency overhead.

## 💡 Key Architectural Features

*   Zero-Copy PJRT Memory Aliasing: TPU Sync accesses internal JAX arrays and PyTorch tensors by extracting native `PjRtBuffer` hardware descriptors. This bypasses intermediary CPU bounce-allocation, establishing true zero-copy H2D, D2H, and H2H (reshard) tensor transmission.
*   GIL-Independent Asynchronous Pipelines: Dispatches high-volume tensor transfers, networking routines, and multi-tier memory allocations entirely onto native C++ background thread pools. By eliminating the Python Global Interpreter Lock (GIL) and Python-level object serialization, memory routing executes in parallel with active TPU Prefill and Decode computation.
*   Near-Saturated Hardware DMA Throughput: Extractor-aligned transfer pipelines reach approximately ~200 GB/s bidirectional PCIe bandwidth utilizing real-world, production-scale LLM KV-cache block dimensions.
*   Multi-Tiered System DRAM Host Offloading: Establishes host system CPU RAM as a massive, ultra-fast "Tier 2" prefix cache. When accelerator HBM reaches capacity limits, LRU-eviction blocks stream continuously to Host DRAM and are retrieved via page-locked, asynchronous DMA on-demand, maximizing RadixAttention and Prefix-Cache hit rates across thousands of concurrent prompt contexts.
*   Warm-Boot Crash Persistence via `/dev/shm`: Backs Host-DRAM KV caches and LRU metadata using OS-level POSIX Shared Memory. If an inference server container restarts, upgrades, or crashes, terabytes of active, offloaded KV-cache remain fully intact in host DRAM. On reboot, the system instantly re-attaches to the warm cache, avoiding catastrophic cold-start re-prefill computation.
*   Accelerated RL Training Support: Facilitates zero-copy, highly parallelized tensor weight and activation migrations between Reinforcement Learning trainers and samplers, maximizing TPU cluster compute utilization.


## 🛠️ Get Started

### Latest Known Good (LKG) Revision
Due to fast-paced active development, the `main` branch may occasionally contain temporary unstable changes. We publish the latest verified stable commit hash in the [`lkg.version`](lkg.version) file at the root of the repository.

To check out the latest verified stable revision that passes all E2E functional and performance test criteria:

```bash
git checkout $(cat lkg.version)
```

### Supported Hardware

TPU Sync is actively validated and optimized for the following accelerator generations:

- Cloud TPU v7x
- Cloud TPU v6e

### Prerequisites

You will need a python environment to run the JAX or torch code. Our codebase is validated against Python 3.12. Execute the following sequence to provision a compatible local environment:

```bash
cd
python3.12 -m venv .venv312
source .venv312/bin/activate
```

#### Installing Bazel
To compile the `tpu_sync` C++ extension binaries, you will need Bazel 8.6.0.

Option 1: Install Bazel 8.6.0 directly (Linux amd64)

```bash
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/8.6.0/bazel-8.6.0-linux-x86_64
sudo chmod +x /usr/local/bin/bazel
```

Option 2: Install via Bazelisk (npm)
Bazelisk is a wrapper that will automatically read the `.bazelversion` file in the project and download the correct version (8.6.0).

```bash
npm install -g @bazel/bazelisk
```

Verify the installation:

```bash
bazel --version
```

#### Installing Patchelf (Required for PyTorch)
To compile and link the PyTorch C++ extension (`_tpu_raiden_torch.so`), you MUST install `patchelf`:

```bash
sudo apt-get install -y patchelf
```
*Why this is necessary:* PyTorch's compiled extension requires `patchelf` to inject a `NEEDED` link on `libpywrap_torch_tpu_common.so` at build time. This ensures TPU backend symbols resolve locally during import without triggering fatal duplicate XLA allocator registration crashes.

#### TPUVM Development Notes
* Disk Space: Remote Bazel builds on standard TPUVMs can exhaust disk space in `/tmp`. Always point Bazel output to a directory that has enough disk space:

  ```bash
  export BAZEL_OUTPUT_BASE=$YOUR_TMP_DIR_WITH_ENOUGH_SPACE
  ```
* PyTorch Wheel Compatibility: Ensure your environment aligns with `torch_tpu`'s pinned C++ ABI expectations (e.g., `torch==2.11.0+cpu`).

###  Installing or Building `tpu_sync`

#### Option 1: Direct installation from Google Artifact Registry

> [!NOTE]
> The pre-built `tpu_sync` wheel will be available on PyPI to the public shortly.

You can install the pre-built `tpu_sync` wheel directly from our Google Artifact Registry:

1. Install the Artifact Registry keyring helper to enable authenticated pip downloads:
   ```bash
   pip install keyrings.google-artifactregistry-auth
   ```
2. Install the framework-specific wheel:

   * For JAX version:

     ```bash
     pip install tpu-raiden-jax --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/
     ```
   * For PyTorch version:

     ```bash
     pip install tpu-raiden-torch --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/
     ```
     > [!IMPORTANT]
     > Unlike the JAX wheel, the torch wheel does not pull `torch` or `torch_tpu`
     > (they are not on a public index). Install a matching `torch_tpu` and its exact
     > `torch` pin (e.g. `torch==2.11.0`) first — the raiden torch extension is
     > ABI-locked to the `torch`/`torch_tpu` build it was compiled against.

     > [!NOTE]
     > The torch wheel's release process is not yet gated on end-to-end
     > performance tests (the torch path is still maturing).

#### Option 2: Building from source

We provide a script to handle the build process and compile extension binaries locally. You can scope compilation to specific frameworks:

```bash
./build.sh [jax|torch|both]
```

What this script does:

1. Navigates to the workspace directory.
2. Compiles the selected extension modules (`_tpu_raiden_jax.so` and/or `_tpu_raiden_torch.so`) using Bazel.
3. For PyTorch builds, executes `patchelf --add-needed` on the generated shared library.
4. Installs necessary Python dependencies listed in `requirements.txt`.
5. Copies compiled `.so` extension binaries directly into their respective framework source packages.

### Testing `tpu_sync`

These are the core functional unit tests designed to verify the correctness of the foundational components and APIs. Once the build is complete, you can run the test suite across JAX and PyTorch:

```bash
./run_tests.sh [jax|torch|both]
```

What this script does:

1. Sets up `PYTHONPATH` so Python can locate the compiled `bazel-bin` and framework wrapper modules.
2. Executes the selected unit test suites across JAX and/or PyTorch directly via `python`.

### Legacy UUID Control-Plane Tuning

The legacy UUID transfer path reads the following optional environment
variables when each KV cache manager is constructed. Values must be positive
integers; invalid values retain the listed default.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `RAIDEN_SEND_TOMBSTONE_TTL_S` | `300` | Retention time for terminal UUID tombstones |
| `RAIDEN_PENDING_ACK_TTL_S` | `30` | Retention time for acknowledgements that arrive before registration |
| `RAIDEN_MAX_SEND_TOMBSTONES` | `4096` | Maximum retained terminal UUID tombstones |
| `RAIDEN_MAX_PENDING_ACKS` | `4096` | Maximum retained pre-registration acknowledgements |
| `RAIDEN_CONTROL_IO_TIMEOUT_S` | `30` | Absolute timeout for one control-plane socket operation |
| `RAIDEN_MAX_LEASE_BATCH_SIZE` | `4096` | UUIDs per renew/cancel request; capped at the protocol safety limit of `8192` |

Configure `RAIDEN_MAX_LEASE_BATCH_SIZE` consistently on communicating peers:
the consumer uses it to chunk requests and the producer uses it as its accepted
request limit.

### Playing with TPU Sync

If you'd like to try out TPU Sync and see it in action, please refer to the [`examples/`](examples/) directory. This folder contains a collection of hands-on scripts designed for users to interact with the library, including:

- [Single-Host Disaggregated Serving](examples/single_host_disagg/README.md): End-to-end prefill-to-decode context migration between co-located local TPU chips.
- [Multi-Host Disaggregated Serving](examples/multihost_disagg/README.md): Multi-VM, distributed KV-cache orchestration and retrieval spanning Data Center network topologies.
- [Host KV-Offloading](examples/kv_host_offloading/README.md): Executing dynamic HBM-to-DRAM prefix caching on a live serving topology.
- [Performance Microbenchmarks](examples/microbenchmarks/README.md): Raw, framework-overhead-free Host-to-Device (H2D) and Device-to-Host (D2H) DMA throughput evaluation.

For detailed execution instructions, dependencies, and shell scripts, consult the root [Examples README](examples/README.md).

### Persistent Shared Memory Cache (TPU/Host Buffer Persistence)

TPU Sync supports allocating host memory staging buffers in POSIX Shared Memory (`/dev/shm`). This allows preserving the KV cache in DRAM when the model serving process terminates (e.g., during serving binary updates), preventing cold starts on process restarts.

#### 1. Enabling Shared Memory
To enable shared memory, set the following environment variables before starting the model serving process:

```bash
# Enable shared memory by specifying a base namespace key
export RAIDEN_SHM_KEY="raiden_cache"

# Specify a unique identifier of the current model config for validation safety
export RAIDEN_SHM_MODEL_UID="model_v1_architecture_config_hash"

# [Optional] Set a server name if running multiple serving instances on the same host
export RAIDEN_SHM_SERVER_NAME="server_8000"
```

When these variables are active, TPU Sync will automatically check for compatible shared memory segments:
- Cold Boot (First run): TPU Sync creates `/dev/shm/raiden_cache_<server_name>_dev_<local_dev_id>` files, initializes layout validation metadata headers, and sets up mappings.
- Warm Boot (Restarts): TPU Sync automatically re-attaches to the existing shared memory files, verifies that the model configuration (`RAIDEN_SHM_MODEL_UID` and caching dimensions) matches, and re-registers the pages with the TPU DMA engine without re-allocation.

#### 2. Running Multiple Servers on the Same Host
If you are running multiple model servers on the same TPU VM, you can avoid namespace collisions by specifying a unique `RAIDEN_SHM_SERVER_NAME` for each server instance (e.g. `server_8000` and `server_8008`). If specified, TPU Sync automatically namespaces the file paths as `/dev/shm/<base_key>_<server_name>_dev_<dev_id>`.

#### 3. Disabling Shared Memory
To disable shared memory and fall back to standard anonymous private memory allocations, simply unset the environment variables:

```bash
unset RAIDEN_SHM_KEY
unset RAIDEN_SHM_MODEL_UID
unset RAIDEN_SHM_SERVER_NAME
```

#### 4. Manual Cleanup & Memory Reclamation
Because POSIX shared memory files survive process termination, you may need to clean them up manually to free up host DRAM on the TPUVM.

To see currently allocated shared memory files:

```bash
ls -la /dev/shm/ | grep raiden_cache
```

To reclaim memory, unlink/delete the shared memory files:

```bash
rm -f /dev/shm/raiden_cache_*
```
*(Note: unlinking deletes the filenames immediately, and the physical host DRAM pages are freed by the kernel as soon as all active serving processes detach or exit).*

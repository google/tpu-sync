# High-Performance KVCacheManager for Disaggregated Serving & Offloading

## 1. Project Vision & Overview

**`KVCacheManager`** is an advanced multi-chip layout synchronization and
memory orchestration framework designed to serve as the foundational storage
back-end for high-performance Large Language Model (LLM) inference
architectures.

Our ultimate roadmap aims to enable **disaggregated serving topologies** and
high-throughput **Key-Value (KV) cache offloading** to support ultra-long
context handling inside state-of-the-art production serving engines such as
**vLLM** and **SGLang** with exceptional bare-metal hardware utilization.

By pre-extracting physical accelerator buffer handles (`PjRtBuffer`), hardware
layout major-stride dimensions (`slice_byte_size_`), and zero-copy storage
memory holds (`raiden::BufferHoldAndAlias`), `KVCacheManager` completely
bypasses Python iteration overheads during transfer execution loops. It maps
fine-grained sub-region chunk operations directly to highly parallelized
background Direct Memory Access (DMA) device streams natively exposed to
Python orchestration scripts via non-blocking tuple lambda futures.

---

## 2. Completed Milestones & Functionality (Done)

### Core Layout Abstraction & Centralization

* **Zero-Overhead Extraction Engine**: Unified extraction of
  `PjRtCApiBuffer` and `CommonPjRtBuffer` descriptor structs, physical byte
  sizes, and C-API dynamic downcasting abstractions.
* **Loop Dispatch Minimization**: Refactored linear partial array boundary
  bounds checks and chunk aggregation math into centralized private dispatch
  macros (`GetMajorSliceByteSize`, `DispatchD2hChunks`), eliminating
  duplicate buffer lifetime handling code entirely across core modules.

### Dynamic Host Memory Column Block Allocation

* **`LogicalBlockManager` Integration**: Precomputes total available host
  memory slice capacities based on customizable runtime `block_size`
  configuration parameters.
* **Strided Transfer Execution (`d2h_auto_allocate`)**: Allocates free
  host memory block indices dynamically upon transfer execution requests,
  computes absolute physical byte offsets locally (`assigned_block_id *
  block_size_ * slice_byte_size_`), and automatically locks contiguous
  sub-regions without requiring manual destination layout strides. Binds
  tuple arrays containing assigned block IDs and future handles directly via
  nanobind.

### Host Memory Internal Buffer Allocation

* **Internal Host Memory Allocation**: Added an optional
  `host_blocks_to_allocate` parameter to `KVCacheManager` constructors. When
  specified, `KVCacheManager` bypasses Python host array extraction entirely
  and dynamically allocates 64-byte cache-line aligned host memory buffers
  (`posix_memalign`) internally for each shard. This eliminates the
  significant latency (often >20 minutes) associated with allocating large
  host JAX arrays in Python during engine initialization.

### Unsafe Buffer Lock Skipping

* **Exclusive Lock Skipping (`unsafe_skip_buffer_lock`)**: Added an optional
  `unsafe_skip_buffer_lock` parameter to `KVCacheManager` constructors. When
  set to `True`, it bypasses exclusive hold/lock acquisition on destination
  buffers during copies. This is highly optimized for disaggregated serving
  scenarios where the caller guarantees that no other concurrent threads will
  modify the buffers during the transfer loop, eliminating synchronization
  lock overhead entirely.
* **Concurrent Pipeline Verification**: Implemented a multi-threaded unit test
  (`test_concurrent_transfer_skip_buffer_lock`) simulating concurrent LLM
  serving. Thread 1 repeatedly executes `h2d`/`d2h` on blocks 0:4, while
  Thread 2 concurrently copies unique data via `h2d` to blocks 4:8. This
  verifies that skipping locks allows flawless parallel transfers on different
  blocks without deadlocks, data races, or corruption.

### POSIX TCP Transport Engine Framework

* **Standalone Binary Transport Prototyping**: Designed custom socket transfer
  pipelines (`tpu_raiden::transport::SocketTransport`) inheriting from virtual
  `mlcl::Transport` adapter baselines. Implemented robust server listening
  loops wrapped within bounding 50ms non-blocking `poll` constraints to
  guarantee immediate listener thread termination joins upon sandbox container
  destruction.

### Block-Level Distributed Server Integration (Active Stage)

* **Embedded Socket Routing Management**: Extends `KVCacheManager` optional
  constructors to spin up dedicated background TCP socket server loops
  encapsulated cleanly inside private nested structs (`BlockTransportServer`).
* **Dynamic Ephemeral Ports**: Binds peer servers to dynamic kernel ephemeral
  ports automatically (passing port `0`), extracting actual assigned ports via
  `getsockname` to resolve strict network isolation limitations inside shared
  verification sandboxes gracefully. Exposed directly via native
  `local_port()` accessors.
* **Pure-Client Decoupled Streams**: Refactored `h2h_write` and
  `h2h_read` to build lightweight independent connection
  streams on the fly via standalone thread-safe `getaddrinfo` host resolution
  loops (`ConnectToPeer`), allowing clients to push or pull data transparently
  without starting background listening server threads.
* **Transparent Symmetric Tensor Exchanges**: Uses custom block pointer chunk
  structures (`lib::ChunkHeader`). Receiving worker streams
  inspect pre-extracted internal attention layers (`layers_`) to compute target
  byte offset locations locally on the fly, bypassing raw virtual base addresses
  entirely. Implements automated column assignments during push streams and
  local allocations during pulling symmetrically.

### NUMA and Network Performance Tuning

* **Multi-NUMA Network Offloading (`ENABLE_MULTI_NUMA`)**: For single-host
  tasks (e.g. local KV cache offloading), leaving this environment variable
  unset (Single NUMA) provides maximum local DMA throughput (up to 195 GB/s).
  For multi-host tasks (e.g. Prefill/Decode resharding), users should opt-in
  by setting `ENABLE_MULTI_NUMA=1` in their environment. This activates
  multiple NICs across NUMA nodes, unlocking 50 GB/s H2H network transfer
  speeds, at the necessary cost of double-dispatch CPU overhead and slightly
  reduced local DMA speed due to JAX thread pinning restrictions.

### Crash-Persistent KV Cache in Shared Memory (`RAIDEN_SHM_KEY`)

* **Both planes persist in shm**: With `RAIDEN_SHM_KEY` set, the KV cache
  data (each worker's host pool, one `<key>[_<server>]_dev_<id>` segment per
  device) and the `KVCacheMetadata` table (one `<key>_metadata[_<server>]`
  segment per store, a crash-persistent mirror of the LRU cache) live in
  POSIX shared memory. Segment lifetime belongs to the `/dev/shm` tmpfs
  mount, not to any process: an engine crash — even SIGKILL — leaves both
  planes intact, and nothing in the serving stack ever unlinks them. On
  restart, construction re-attaches and self-validates each segment
  (magic/model_uid/geometry); a surviving table warm-starts the store via
  `RecoverFromLocalManifest`, any mismatch reformats to a clean cold start.
* **Expected lifecycle management**: Recovery therefore reduces to one
  deployment requirement — the same `/dev/shm` mount and `RAIDEN_SHM_*`
  environment must be there after the restart, preserved or wiped as a
  whole (partial deletion of segments under one key is not supported).
  * *Cloud VM / bare metal*: `/dev/shm` is host-wide, so any process
    restart warm-starts for free; a reboot clears it (tmpfs). Decommission
    is an explicit operator `shm_unlink`/`rm` of the deployment's
    `/dev/shm/<key>*` segments.
  * *Docker*: a container's private `/dev/shm` dies with the container.
    To survive an engine container restart, place the segments on an IPC
    mount that outlives the engine — share the IPC namespace with a
    longer-lived container (`--ipc=container:<sidecar>`) or mount a shared
    tmpfs volume — and size it for the KV pool (default `--shm-size` is
    64MB).
  * *Kubernetes*: mount a pod-level `emptyDir{medium: Memory}` volume at
    `/dev/shm` in the engine container; a container crash/restart within
    the pod keeps the volume, so the engine warm-starts. Deleting or
    rescheduling the pod discards it — that is a clean cold start, not a
    correctness issue.

### Bounded-Window Host Staging

Bounded-window host staging throttles host memory footprint during large KV-cache
chunk transfers by streaming blocks through a sliding ring buffer of windowed
chunks, avoiding the memory spikes of unbounded full-transfer buffering.

#### Configuration Surface

The staging pipeline can be configured via constructor kwargs or environment
variables:

* `staging_window_chunks` (`TPU_RAIDEN_STAGING_WINDOW_CHUNKS`): Number of active
  chunks in the bounded sliding window ($W$, default: 0 / disabled).
* `staging_chunk_blocks` (`TPU_RAIDEN_STAGING_CHUNK_BLOCKS`): Number of KV blocks
  per window chunk ($C$, default: 0 / auto-derived from byte target).
* `staging_chunk_bytes` (`TPU_RAIDEN_STAGING_CHUNK_BYTES`): Soft chunk size
  target in bytes (default: 8MB).
* `staging_stream_bytes` (`TPU_RAIDEN_STAGING_STREAM_BYTES`): Stream transfer
  target in bytes (default: 4MB).
* `staging_admission` (`TPU_RAIDEN_STAGING_ADMISSION`): Admission control
  behavior under memory pressure: `"fail_fast"` (default) or `"wait"`.
* `staging_admission_timeout_ms` (`TPU_RAIDEN_STAGING_ADMISSION_TIMEOUT_MS`):
  Timeout in milliseconds when `staging_admission="wait"` (default: 0 / no wait).
* `staging_assume_peer_support` (`TPU_RAIDEN_STAGING_WINDOW_ASSUME_PEER`):
  Assume peer support for windowed transfers without handshake probe (default:
  false).

#### Budget = Pool Contract

Host staging enforces a strict equivalence between the staging budget and the
allocated pool geometry:
$$\text{budget\_blocks} = \text{num\_slots} \times \text{max\_blocks}$$

Recommended sizing aligns `max_blocks` with the full window capacity and
`num_slots` with concurrent stream concurrency:
$$\text{max\_blocks} = W \times C, \quad \text{num\_slots} = B$$
where $B$ is the maximum number of concurrent request streams.

#### Dynamic-Mode LRU Co-Tenancy Warning

When host staging buffers share an underlying host memory pool with an LRU cache
or secondary KV store, bounded-window staging pins active host blocks while
chunks are in flight. If the total host pool size is smaller than
$\text{num\_slots} \times \text{max\_blocks}$, unpinned cache blocks may be
evicted prematurely under concurrency spikes, or staging admission may block.
Always provision the underlying pool to exceed the sum of pinned staging demand
and cached working sets. The manager logs a one-time WARNING at construction
when the staging window is enabled and the host pool already has other users
(`num_free_blocks() < total_blocks()`).

#### Construction-Time Selection vs. Runtime Mutation

* **Construction-Time**: `window_chunks` ($W$) and `serve_windowed_pulls` define
  the immutable memory layouts and protocol capabilities of the worker.
  Mutating these fields via `set_staging_window()` will raise `ValueError` /
  `kInvalidArgument`.
* **Runtime-Mutable**: `chunk_blocks`, `chunk_bytes_target`,
  `stream_bytes_target`, `admission`, `admission_timeout_ms`, and
  `assume_peer_support` can be updated dynamically at runtime via
  `manager.set_staging_window(...)`.

#### Window-Off Semantics

With `window_chunks = 0`, the Decode role is behaviour-identical to the
pre-series binary (only log levels change). However, a new binary's Prefill role
still answers capability probes (`kOpCapabilities`), pre-validates incoming
pull requests, relaxes the TCP body pre-check to the absolute capacity cap, and
serves windowed pulls from remote peers. Setting the kill switch
`TPU_RAIDEN_STAGING_WINDOW_SERVE=0` explicitly disables serving windowed pulls
on this worker: capability probe replies return empty capabilities (`caps=`),
and windowed pulls are rejected or downgraded to unchunked exact-$k$ transfers.

#### Admission Policies

* `"fail_fast"`: Immediately rejects new streaming transfers if the current
  active block reservation exceeds available budget blocks.
* `"wait"`: Queues incoming transfer streams up to `staging_admission_timeout_ms`
  until prior in-flight chunks retire and free up staging blocks.

#### Telemetry & Metrics (`staging_stats()`)

`manager.staging_stats()` returns a dictionary with 9 telemetry counters and
watermarks:

* `budget_blocks`: Total host staging block budget ($\text{num\_slots} \times
  \text{max\_blocks}$).
* `blocks_in_use`: Current number of pinned staging blocks across active chunks.
* `peak_blocks_in_use`: High-water mark of pinned staging blocks since launch.
* `gate_waits_over_1ms`: Number of staging chunk acquisitions delayed >1ms by
  flow control gates.
* `rejected_streams`: Number of incoming streams rejected by admission control.
* `admission_rejects`: Cumulative streams rejected due to insufficient budget.
* `admission_waits`: Streams that successfully waited for budget before
  proceeding.
* `admission_requeues`: Streams requeued during admission waits.
* `admission_timeouts`: Streams that reached timeout during admission wait.

---

## 3. Immediate Execution Roadmap (TODO List)

* **Official MLCL Engine Migration**: Replace temporary POSIX TCP socket
  streaming loops with the official production-ready MLCL transport library
  adapter layer once upstream infrastructure matures completely.
* **Overlapped Execution Stream Polling**: Refactor asynchronous memory future
  polling logic to exploit customized parallel CUDA/TPU stream synchronization
  triggers, ensuring background tensor transfers overlap computational
  attention mechanisms flawlessly.
* **Granular Host Memory Pinning**: Implement highly optimized asynchronous
  page-locked (pinned) host memory buffer pooling allocations to maximize
  cross-host multi-lane PCIe/Network memory copy bandwidth efficiency.

---

## 4. Long-Term Strategic Horizon (Stretch Goals)

* **Serving Engine Integration**: Expose native end-to-end API bindings
  supporting continuous background Key-Value cache offloading and elastic
  block swapping inside production runtime engines (**vLLM** and **SGLang**).
* **Cross-Cluster Auto-Scaling Storage**: Enable automated cross-host
  distributed inference topologies capable of live tensor mapping and
  transparent cache slice migration across remote disaggregated memory
  accelerator nodes.

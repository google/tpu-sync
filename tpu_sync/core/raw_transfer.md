# Raw Transfer Library Technical Documentation

## Overview

The `raw_transfer` library is a custom C++ extension for python MLframeworks (e.g., JAX), 
designed to perform high-performance, asynchronous device-to-host (D2H) and host-to-device
(H2D) data transfers for sharded arrays on TPU. It bypasses standard XLA buffer
management to achieve lower overhead and higher throughput.

## API Reference

### `transfer_d2h_async`

Asynchronously copies data from device to host.

```python
def transfer_d2h_async(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> list[PjRtCopyFuture]:
```

- **`src_arr`**: Source JAX array on device.
- **`dst_arr`**: Destination JAX array on host.
- **`src_offsets_major_dim`**: (Optional) List of offsets along the major dimension for partial copies.
- **`dst_offsets_major_dim`**: (Optional) List of offsets along the major dimension for partial copies.
- **`copy_sizes_major_dim`**: (Optional) List of sizes to copy along the major dimension.
- **Returns**: A list of `PjRtCopyFuture` objects that can be awaited.

### `transfer_h2d_async`

Asynchronously copies data from host to device.

```python
def transfer_h2d_async(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> list[PjRtCopyFuture]:
```

-   **`src_arr`**: Source JAX array on host.
-   **`dst_arr`**: Destination JAX array on device.
-   **Arguments** are similar to `transfer_d2h_async`.

### `transfer_d2h`

Synchronously copies data from device to host.

```python
def transfer_d2h(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None:
```

-   Blocks until the transfer is complete.
-   Arguments are identical to `transfer_d2h_async`.

### `transfer_h2d`

Synchronously copies data from host to device.

```python
def transfer_h2d(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None:
```

-   Blocks until the transfer is complete.
-   Arguments are identical to `transfer_h2d_async`.

### `transfer_d2h_batch`

Synchronously copies a batch of arrays from device to host.

```python
def transfer_d2h_batch(
    src_arrs: list,
    dst_arrs: list,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None:
```

- Blocks until all transfers in the batch are complete.
- Arguments are lists of arrays and optional offsets.

### `transfer_h2d_batch`

Synchronously copies a batch of arrays from host to device.

```python
def transfer_h2d_batch(
    src_arrs: list,
    dst_arrs: list,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None:
```

- Blocks until all transfers in the batch are complete.
- Arguments are similar to `transfer_d2h_batch`.

## Requirements & Constraints
- **Rank**: Array rank must be >= 3 for **partial copies**. Full array copies are supported for lower ranks (e.g., 1D and 2D arrays).
- **Alignment**: For **partial copies**, the product of all non-major dimensions multiplied by the element size in bytes must be a multiple of the device tile size (typically 4KB). This constraint is relaxed for full array copies.
- **Sharding**: Partial copy is NOT supported if the array is sharded on the major dimension.

## WARNING: Data Layout

-   **D2H**: The data layout on the host destination array may NOT be standard
    row-major. It preserves the physical layout of the TPU, which may include
    padding and tiling. Do NOT use the host array for any computations or assume
    standard NumPy layout.
-   **H2D**: The source host array must have the specific physical layout
    expected by the TPU (matching what was produced by `transfer_d2h_async`) to
    avoid data corruption.

## Internal Design

The `raw_transfer` library achieves high performance by bypassing standard XLA
buffer management and interacting directly with PjRt buffers and raw device
memory.

### Core Components:
1. **Buffer Extraction**: It extracts the underlying `PjRtBuffer` from JAX arrays by accessing internal IFRT (`xla::ifrt::Array`) structures. It supports both standard `CommonPjRtBuffer` and C-API based `PjRtCApiBuffer`.
2. **Raw Memory Access**: To prevent garbage collection during asynchronous transfers, it acquires a scoped hold on `CommonPjRtBuffer` or creates a raw alias for `PjRtCApiBuffer`.
3. **Tiling-Aware Addressing**: For partial copies, it assumes that tiling and padding only affect the minor dimensions. This allows it to calculate linear physical offsets by multiplying the major dimension offset by the total size of the minor dimensions (including padding), enabling simple pointer arithmetic for slicing.
4. **Asynchronous DMA**: It calls `CopyRawToHost` or `PjRtCApiRawBuffer_CopyRawDeviceToHost` to trigger direct DMA transfers between TPU HBM and host memory.
5. **Multi-Device Parallelism**: The library iterates over all addressable shards of a distributed JAX array and issues concurrent transfer commands for each shard. It returns a list of custom `PjRtCopyFuture` objects wrapping the underlying XLA futures, which are then awaited in Python.

### DMA Pacing (torch bindings)

The TPU runtime executes a TensorCore's DMAs and program launches through one
in-order queue, and a PJRT raw-buffer copy reaches it as a single request for
the whole range. Issuing a large raw copy therefore holds every kernel launched
after it until the copy has crossed the host link: on TPU7x a kernel launched
behind N MiB of queued raw H2D starts only when the DMA finishes (16 MiB →
0.5 ms, 1 GiB → 19 ms), and the sibling core of the same chip is delayed by
about half of that. With compute issued first, libtpu's completion reads
(`ReadSyncFlag`) wait behind the same DMA, its launch semaphore stays held, and
the TensorCore idles between kernels ("kernel bubbles"; 5–22 ms stalls with
eight ranks copying 1.4 GB each). Tensors copied with `.to()`
(`BufferFromHostBuffer`) are also queued as one whole-buffer DMA each and delay
kernels the same way.

`RawDmaPacer` (`raw_dma_pacer.h`) covers only tpu_raiden's torch raw transfers,
not `.to()`: every torch raw transfer (`transfer_*`, `transfer_*_batch*`,
`PreparedTorchRawTransfer`) is split into pieces of at most `chunk_bytes`
(default 16 MiB) and at most `max_inflight_bytes` (default 64 MiB) are kept
outstanding per process, first-in first-out across callers. A piece is issued
on the calling thread when it fits; otherwise a pacer thread issues it as
earlier pieces complete. The copy's future completes after its last piece and
the pacer keeps the tensors and buffers alive until then.

Configure with `raw_transfer.set_dma_pacing(max_inflight_bytes, chunk_bytes)`
or the `TPU_RAIDEN_RAW_DMA_MAX_INFLIGHT_BYTES` /
`TPU_RAIDEN_RAW_DMA_CHUNK_BYTES` environment variables; a limit <= 0 restores
the unpaced behaviour. If either variable is set to a value that is not an
integer (e.g. `16M`), raw transfers and the pacing calls raise `RuntimeError`.
`raw_transfer.get_dma_pacing()` reports the limits and the in-flight and peak
byte counts.


## TODO

- [ ] **Support Multi-Host Transfers**: Fix the shard mismatch error on multi-host setups. Experiments attempted on multi-host setups (8 cores) failed with `RuntimeError: Number of shards in source and destination must match` because the TPU array has 4 addressable shards on the local host, while the host array was created with 8 addressable shards, causing a mismatch in the library's validation check. We need to handle cases where the host array is sharded across all devices but the TPU array only has local shards addressable by the current process.


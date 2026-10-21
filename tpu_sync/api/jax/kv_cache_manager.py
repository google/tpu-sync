# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-performance JAX KV Cache Manager (repurposed as TransferEngine).

Architecture and HBM staging overview:
  Each TPU worker host runs a C++ `KVCacheManager` (`_tpu_raiden_jax`) that
  transfers KV-cache blocks between TPU HBM and pinned Host DRAM via PJRT DMA.

  In PagedAttention serving (e.g. vLLM on JAX TPU), each transformer layer owns
  a separate device `jax.Array` (`kv_caches[layer]` shaped `[num_blocks, ...]`),
  and the blocks evicted to or recalled from Host DRAM are typically scattered
  across non-contiguous indices. On the un-staged path (`_impl.d2h` / `h2d`),
  copying `B` scattered blocks across `L` layers issues `B * L` small,
  fragmented PCIe PJRT DMA slices per TPU shard, incurring high driver and
  descriptor overhead and low PCIe bandwidth utilization.

  When `enable_hbm_staging=True` (or `RAIDEN_ENABLE_HBM_STAGING=1`), this
  wrapper uses TPU HBM's ~TB/s internal bandwidth to coalesce transfers first:
    * D2H (`d2h`, `d2h_auto_allocate`): A power-of-two bucketed `jax.jit`
      kernel (`_jit_gather`) gathers scattered blocks across all `L` layers in
      HBM into one contiguous `[bucket_blocks, num_layers, ...]` tensor, and
      `_impl.copy_packed` streams it to Host DRAM in a single bulk PJRT DMA per
      TPU shard (scattering into per-layer Host DRAM slots via a pooled C++
      bounce buffer).
    * H2D (`h2d`): `_impl.copy_packed` gathers Host DRAM slots across all layers
      into one bulk PJRT H2D DMA into a reusable HBM staging buffer, and a
      donated power-of-two bucketed `jax.jit` kernel (`_jit_scatter`) scatters
      the blocks into each layer's `kv_caches` array in place in HBM.

  Serving loops that donate and replace `kv_caches` on every forward step pass
  the live arrays via `kv_caches=...` and read updated arrays from
  `manager.kv_caches`.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax

from tpu_sync.frameworks.jax import _tpu_raiden_jax as _impl

# Default cap on KV blocks staged in HBM per bulk DMA window, bounding peak
# temporary HBM usage while keeping each PCIe DMA large enough to saturate link
# bandwidth.
_DEFAULT_MAX_STAGING_BLOCKS = 64


def _next_pow2(n: int) -> int:
  """Rounds `n` up to a power of two (`>= 1`) to bound XLA JIT bucket shapes."""
  return 1 if n <= 1 else 1 << (n - 1).bit_length()


class _StagedCopyFuture:
  """Wraps RaidenFutures while keeping staged device arrays alive until done.

  C++ `copy_packed` executes PJRT D2H DMA asynchronously against the underlying
  TPU buffer of a temporary `packed` `jax.Array`. Retaining `keep_alive` until
  `Await()` or `IsReady()` observes completion prevents Python GC from freeing
  or recycling that HBM buffer mid-DMA.
  """

  def __init__(self, futures: Sequence[Any], keep_alive: Any = None):
    self._futures = list(futures)
    self._keep_alive = keep_alive

  def Await(self) -> None:  # pylint: disable=invalid-name
    try:
      for fut in self._futures:
        fut.Await()
    finally:
      self._keep_alive = None

  def IsReady(self) -> bool:  # pylint: disable=invalid-name
    if all(fut.IsReady() for fut in self._futures):
      self._keep_alive = None
      return True
    return False


def _jit_gather(packed_sharding: Any) -> Callable[..., Any]:
  """Builds a bucketed JIT kernel gathering blocks across all layers in HBM.

  Given `caches` (a list of `L` arrays of shape `[num_blocks, *tail]`) and `idx`
  of static bucket length `_b`, gathers `idx` along axis 0 of every layer and
  stacks them into a contiguous `[bucket_size, L, *tail]` HBM buffer. Keying
  `static_argnums=(0,)` on power-of-two `_b` limits XLA compilation to
  `O(log2(max_staging_blocks))` executables.
  """
  return jax.jit(
      lambda _b, caches, idx: jax.numpy.stack(
          [jax.numpy.take(c, idx, axis=0) for c in caches], axis=1
      ),
      static_argnums=(0,),
      out_shardings=packed_sharding,
  )


def _jit_scatter(caches_shardings: Any) -> Callable[..., List[Any]]:
  """Builds a bucketed JIT kernel scattering staged H2D blocks into `caches`.

  Slices `bsize` blocks starting at `start` from `buf` (`[max_staging_blocks,
  L, *tail]`) and scatters `staged[:, i]` into `dst_ids` of each `caches[i]`.
  `donate_argnums=(1,)` lets XLA update the full KV-cache arrays in place in HBM
  without allocating duplicate full-cache buffers.
  """
  def _fn(bsize: int, caches: List[Any], buf: Any, start: Any, dst_ids: Any) -> List[Any]:
    staged = jax.lax.dynamic_slice_in_dim(buf, start, bsize, axis=0)
    return [c.at[dst_ids].set(staged[:, i]) for i, c in enumerate(caches)]

  kw = {} if caches_shardings is None else {"out_shardings": caches_shardings}
  return jax.jit(_fn, static_argnums=(0,), donate_argnums=(1,), **kw)


class KVCacheManager:
  """JAX Python frontend around Raiden's C++ `KVCacheManager` engine."""

  def __init__(
      self,
      kv_caches: List[Any],
      local_control_port: int = -1,
      max_blocks: Optional[int] = None,
      num_slots: Optional[int] = None,
      timeout_s: int = 30,
      unsafe_skip_buffer_lock: bool = False,
      host_blocks_to_allocate: Optional[int] = None,
      parallelism: int = 1,
      node_id: int = 0,
      raiden_worker_port: Optional[int] = None,
      raiden_controller_address: Optional[str] = None,
      worker_id: Optional[str] = None,
      enable_shm: bool = False,
      backend_configs: Optional[Sequence[Any]] = None,
      enable_hbm_staging: Optional[bool] = None,
      max_staging_blocks: Optional[int] = None,
  ):
    """Initializes the JAX KV-cache transfer manager and optional HBM staging.

    Args:
      kv_caches: Per-layer device-placed `jax.Array` KV caches, each shaped
        `[num_blocks, *block_tail_shape]`.
      local_control_port: Legacy TCP control-plane port (`<= 0` disables it when
        `host_blocks_to_allocate` is used).
      max_blocks: Maximum blocks per staging slot in slot-based host allocation.
      num_slots: Number of host staging slots when not using
        `host_blocks_to_allocate`.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: If True, skips PJRT dynamic buffer locking when
        extracting raw device pointers.
      host_blocks_to_allocate: Total blocks to allocate in the unified Host DRAM
        block pool (preferred over `max_blocks`/`num_slots`).
      parallelism: Number of parallel transport streams per layer.
      node_id: Dense worker shard index `[0, expected_worker_count)` within this
        engine's `(TP x PCP x PP)` topology, used to pair corresponding KV
        shards across peer stores.
      raiden_worker_port: Port for the `WorkerService` gRPC server (`None` or
        `0` selects an ephemeral port when `raiden_controller_address` is set).
      raiden_controller_address: Address of the central `RaidenController`
        coordinating this worker.
      worker_id: Optional human-readable identifier for this worker.
      enable_shm: If True, backs Host DRAM staging arenas with `/dev/shm`
        segments under `RAIDEN_SHM_KEY`.
      backend_configs: Optional secondary storage `BackendConfig` entries (e.g.
        POSIX or TDS offload tiers) to register on this worker.
      enable_hbm_staging: Enables on-device HBM gather/scatter coalescing so
        scattered multi-layer D2H/H2D transfers issue one bulk PJRT DMA per
        shard. Defaults to `RAIDEN_ENABLE_HBM_STAGING == "1"` when `None`.
      max_staging_blocks: Maximum KV blocks packed per HBM staging window
        (rounded up to a power of two; defaults to `min(64, num_blocks)`).
    """
    self._kv_caches = list(kv_caches)
    self._unsafe_skip_buffer_lock = unsafe_skip_buffer_lock
    if enable_hbm_staging is None:
      enable_hbm_staging = os.environ.get("RAIDEN_ENABLE_HBM_STAGING") == "1"
    self._enable_hbm_staging = enable_hbm_staging
    self._h2d_staging_buf: Any = None

    if self._enable_hbm_staging:
      first = self._kv_caches[0]
      self._block_tail_shape = first.shape[1:]
      self._layer_dtype = first.dtype
      self._num_device_blocks = first.shape[0]
      self._max_staging_blocks = _next_pow2(
          max_staging_blocks or min(_DEFAULT_MAX_STAGING_BLOCKS, self._num_device_blocks)
      )
      sharding = first.sharding
      if isinstance(sharding, jax.sharding.NamedSharding):
        # Keep axis 0 (`bucket_blocks`) and axis 1 (`num_layers`) replicated
        # (`None, None`) while preserving per-chip tensor/head sharding on the
        # trailing dimensions (`*sharding.spec[1:]`). Every TPU chip therefore
        # packs its own local shard of all staged blocks and layers contiguously
        # in HBM with zero cross-device collectives, matching the per-shard
        # layout expected by C++ `CopyPackedDeviceBuffer`.
        mk, pspec = sharding.memory_kind, jax.sharding.PartitionSpec
        self._packed_sharding = jax.sharding.NamedSharding(
            sharding.mesh, pspec(None, None, *sharding.spec[1:]), memory_kind=mk
        )
        # Replicate int32 block-index vectors across all mesh devices.
        self._index_sharding = jax.sharding.NamedSharding(sharding.mesh, pspec(), memory_kind=mk)
        caches_shardings = [a.sharding for a in self._kv_caches]
      else:
        self._packed_sharding = self._index_sharding = sharding
        caches_shardings = None
      self._gather_bucket_fn = _jit_gather(self._packed_sharding)
      self._scatter_bucket_fn = _jit_scatter(caches_shardings)

    if host_blocks_to_allocate is not None:
      self._impl = _impl.KVCacheManager(
          self._kv_caches,
          local_control_port if local_control_port > 0 else None,
          host_blocks_to_allocate,
          unsafe_skip_buffer_lock,
          parallelism,
          raiden_worker_port or 0,
          raiden_controller_address,
          worker_id,
          node_id=node_id,
          enable_shm=enable_shm,
      )
    else:
      self._impl = _impl.KVCacheManager(
          kv_caches=self._kv_caches,
          node_id=node_id,
          local_control_port=local_control_port,
          max_blocks=max_blocks,
          num_slots=num_slots,
          timeout_s=timeout_s,
          unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
          raiden_worker_port=raiden_worker_port or 0,
          raiden_controller_address=raiden_controller_address,
          worker_id=worker_id,
          enable_shm=enable_shm,
      )

    if backend_configs:
      self._impl.register_kv_backends(list(backend_configs))

  @property
  def kv_caches(self) -> List[Any]:
    """Latest per-layer device KV-cache arrays (updated after donated `h2d`)."""
    return self._kv_caches

  def _idx(self, values: Any) -> Any:
    """Places block indices on device as a replicated `int32` JAX array."""
    return jax.numpy.asarray(values, dtype=jax.numpy.int32, device=self._index_sharding)

  def _prep_copy(
      self, src_offsets: List[int], copy_sizes: Optional[List[int]], kv_caches: Optional[List[Any]]
  ) -> Tuple[List[int], List[Any]]:
    """Normalizes `copy_sizes` and updates live `kv_caches` when provided."""
    if kv_caches is not None:
      self._kv_caches = kv_caches
    return ([1] * len(src_offsets) if copy_sizes is None else list(copy_sizes)), self._kv_caches

  def _expand_blocks(
      self, offsets: Sequence[int], sizes: Sequence[int], limit: Optional[int] = None
  ) -> List[int]:
    """Flattens `(offset, size)` segments into block IDs and checks bounds.

    Explicitly checking `limit` is essential on the HBM-staged path because XLA
    `jnp.take` and `.at[idx].set(...)` silently clamp or drop out-of-bounds
    indices instead of raising an error.
    """
    blocks: List[int] = []
    for off, sz in zip(offsets, sizes):
      if off < 0 or sz < 0 or (limit is not None and off + sz > limit):
        raise ValueError(f"Block range [{off}, {off + sz}) out of bounds (limit={limit})")
      blocks.extend(range(off, off + sz))
    return blocks

  def _ensure_h2d_staging_buf(self) -> Any:
    """Lazily allocates the reusable HBM staging buffer for bulk H2D DMAs."""
    if self._h2d_staging_buf is None:
      shape = (self._max_staging_blocks, len(self._kv_caches), *self._block_tail_shape)
      self._h2d_staging_buf = jax.jit(
          lambda: jax.numpy.zeros(shape, dtype=self._layer_dtype), out_shardings=self._packed_sharding
      )()
      jax.block_until_ready(self._h2d_staging_buf)
    return self._h2d_staging_buf

  def _staged_d2h_blocks(
      self, src_blocks: List[int], dst_blocks: List[int], active_caches: List[Any]
  ) -> _StagedCopyFuture:
    """Gathers `src_blocks` in HBM by window and issues bulk packed D2H DMAs."""
    futures, keep_alives, step = [], [], self._max_staging_blocks
    for s in range(0, len(src_blocks), step):
      win_src, win_dst = src_blocks[s : s + step], dst_blocks[s : s + step]
      # Pad `win_src` up to the next power-of-two bucket size by repeating the
      # last valid index. C++ `copy_packed` transfers only `len(win_dst)` blocks
      # and ignores the padded tail.
      b_size = _next_pow2(len(win_src))
      padded = win_src + [win_src[-1]] * (b_size - len(win_src))
      packed = self._gather_bucket_fn(b_size, active_caches, self._idx(padded))
      # Wait for the XLA gather kernel to finish populating `packed` before raw
      # C++ PJRT DMA reads directly from its underlying HBM buffer.
      jax.block_until_ready(packed)
      futures.append(self._impl.copy_packed(packed, win_dst, True, self._unsafe_skip_buffer_lock))
      keep_alives.append(packed)
    return _StagedCopyFuture(futures, keep_alives)

  def get_raiden_worker_port(self) -> int:
    """Returns the listening `WorkerService` gRPC port, or `0` if inactive."""
    return self._impl.get_raiden_worker_port()

  @property
  def is_listener_active(self) -> bool:
    """Whether the worker's `WorkerService` gRPC listener is running."""
    return self.get_raiden_worker_port() > 0

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns transport endpoint descriptors for this worker's local shards."""
    return self._impl.get_local_endpoints()

  def register_read(self, req_id: str, uuid: int, block_ids: List[int]) -> bool:
    """Publishes `block_ids` on the producer so a remote peer can pull them."""
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: Dict[str, Any],
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> None:
    """Starts an asynchronous pull of `remote_block_ids` from a peer worker."""
    self._impl.start_read(req_id, uuid, remote_endpoint, remote_block_ids, local_block_ids, parallelism)

  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]:
    """Returns `(done_sending, done_recving, failed_recving)` request IDs."""
    return self._impl.complete_read()

  def d2h(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
      *,
      kv_caches: Optional[List[Any]] = None,
  ) -> Any:
    """Copies KV blocks from TPU HBM (`src_offsets`) to Host DRAM.

    Args:
      src_offsets: Starting device block indices for each segment.
      dst_offsets: Starting Host DRAM block indices for each segment.
      copy_sizes: Contiguous block count per segment (`None` means `1` each).
      kv_caches: Optional live per-layer `jax.Array` caches (required on the
        HBM-staged path whenever a JIT step has donated/replaced prior arrays).

    Returns:
      A future (`RaidenFuture` or `_StagedCopyFuture`) supporting `.Await()` and
      `.IsReady()`.
    """
    sizes, active = self._prep_copy(src_offsets, copy_sizes, kv_caches)
    if self._enable_hbm_staging and src_offsets:
      src_blocks = self._expand_blocks(src_offsets, sizes, self._num_device_blocks)
      return self._staged_d2h_blocks(src_blocks, self._expand_blocks(dst_offsets, sizes), active)
    return self._impl.d2h(src_offsets, dst_offsets, sizes)

  def h2d(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
      *,
      kv_caches: Optional[List[Any]] = None,
  ) -> Any:
    """Copies KV blocks from Host DRAM (`src_offsets`) into TPU HBM.

    When HBM staging is enabled, `_scatter_bucket_fn` donates and replaces the
    per-layer `jax.Array` handles in place; both `active[:]` (if a mutable list
    was passed via `kv_caches`) and `self.kv_caches` are updated to the live
    post-scatter arrays before returning.

    Args:
      src_offsets: Starting Host DRAM block indices for each segment.
      dst_offsets: Starting device block indices for each segment.
      copy_sizes: Contiguous block count per segment (`None` means `1` each).
      kv_caches: Optional live per-layer `jax.Array` caches to scatter into.

    Returns:
      A future (`RaidenFuture` or `_StagedCopyFuture`) supporting `.Await()` and
      `.IsReady()`.
    """
    sizes, active = self._prep_copy(src_offsets, copy_sizes, kv_caches)
    if self._enable_hbm_staging and src_offsets:
      dst_blocks = self._expand_blocks(dst_offsets, sizes, self._num_device_blocks)
      if not dst_blocks:
        return _StagedCopyFuture([])
      src_blocks = self._expand_blocks(src_offsets, sizes)
      staging_buf = self._ensure_h2d_staging_buf()
      cur_caches, last_fut, step = list(active), None, self._max_staging_blocks
      for s in range(0, len(dst_blocks), step):
        win_src, win_dst = src_blocks[s : s + step], dst_blocks[s : s + step]
        # Pull all layers of `win_src` from Host DRAM into `staging_buf` prefix
        # via a single bulk H2D PJRT DMA per TPU shard, and wait before XLA
        # reads `staging_buf` (raw PJRT DMA is outside XLA's stream ordering).
        last_fut = self._impl.copy_packed(staging_buf, win_src, False, self._unsafe_skip_buffer_lock)
        last_fut.Await()
        # Decompose `win_len` into descending powers of two (e.g. 5 = 4 + 1) so
        # `_scatter_bucket_fn` writes exact disjoint slices without duplicate
        # padding indices (duplicate indices in `.at[dst_ids].set(...)` have
        # non-deterministic overwrite semantics) while reusing power-of-two JIT
        # bucket executables.
        offset, win_len = 0, len(win_dst)
        for bit in range(win_len.bit_length() - 1, -1, -1):
          if win_len & (1 << bit):
            b_size = 1 << bit
            cur_caches = self._scatter_bucket_fn(
                b_size, cur_caches, staging_buf, self._idx(offset), self._idx(win_dst[offset : offset + b_size])
            )
            offset += b_size
        # Ensure XLA finishes reading `staging_buf` before the next loop window
        # overwrites `staging_buf` with another raw H2D PJRT DMA.
        jax.block_until_ready(cur_caches)
      # Write back newly donated `jax.Array` handles so both `self.kv_caches`
      # and the caller's `kv_caches` list hold the live post-scatter arrays.
      active[:] = cur_caches
      return _StagedCopyFuture([last_fut])
    return self._impl.h2d(src_offsets, dst_offsets, sizes)

  def d2h_auto_allocate(
      self,
      src_offsets: List[int],
      copy_sizes: Optional[List[int]] = None,
      *,
      kv_caches: Optional[List[Any]] = None,
  ) -> Tuple[List[int], Any]:
    """Allocates locked Host DRAM slots and copies `src_offsets` from TPU HBM.

    Args:
      src_offsets: Starting device block indices for each segment.
      copy_sizes: Contiguous block count per segment (`None` means `1` each).
      kv_caches: Optional live per-layer `jax.Array` caches.

    Returns:
      `(allocated_host_block_ids, copy_future)`. Call `unlock_blocks` on the
      returned IDs once the offloaded Host DRAM blocks may be evicted/reused.
    """
    sizes, active = self._prep_copy(src_offsets, copy_sizes, kv_caches)
    if self._enable_hbm_staging:
      src_blocks = self._expand_blocks(src_offsets, sizes, self._num_device_blocks)
      if not src_blocks:
        return [], _StagedCopyFuture([])
      chunks = list(self._impl.allocate_blocks(len(src_blocks)))
      return chunks, self._staged_d2h_blocks(src_blocks, chunks, active)
    return self._impl.d2h_auto_allocate(src_offsets, sizes)

  def unlock_blocks(self, block_ids: List[int]) -> None:
    """Unlocks Host DRAM staging `block_ids` so LRU eviction may reuse them."""
    self._impl.unlock_blocks(block_ids)

  def dump_metrics_to_string(self) -> str:
    """Returns collected transfer telemetry metrics serialized as JSON."""
    return self._impl.dump_metrics_to_string()

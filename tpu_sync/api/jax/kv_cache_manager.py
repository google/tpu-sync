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

"""High-performance JAX KV Cache Manager (repurposed as TransferEngine)."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from tpu_sync.frameworks.jax import _tpu_raiden_jax as _impl


class KVCacheManager:
  """Wrapper around compiled C++ TransferEngine.

  This class has been repurposed to expose the TransferEngine API interface
  for distributed disaggregated KV-cache movement on JAX TPUs.
  """

  def __init__(
      self,
      kv_caches: List[Any],
      local_control_port: int,
      max_blocks: Optional[int] = None,
      num_slots: Optional[int] = None,
      timeout_s: float = 120.0,
      unsafe_skip_buffer_lock: bool = True,
      host_blocks_to_allocate: Optional[int] = None,
      parallelism: int = 4,
      node_id: int = 0,
      raiden_worker_port: int = 0,
      raiden_controller_address: Optional[str] = None,
      worker_id: Optional[str] = None,
      enable_shm: bool = False,
      backend_configs: Optional[Sequence[Any]] = None,
      staging_window_chunks: Optional[int] = None,
      staging_chunk_blocks: Optional[int] = None,
      staging_chunk_bytes: Optional[int] = None,
      staging_stream_bytes: Optional[int] = None,
      staging_admission: Optional[str] = None,
      staging_admission_timeout_ms: Optional[int] = None,
      staging_assume_peer_support: Optional[bool] = None,
  ):
    """Instantiates the TransferEngine-based KVCacheManager.

    Bounded-window host staging enforces that the host staging block budget
    matches the staging pool size: `budget = num_slots * max_blocks`.
    Recommended configuration sets `max_blocks = W * C` (where W is
    `staging_window_chunks` and C is `staging_chunk_blocks`) and `num_slots = B`
    (concurrency limit / batch slots).

    WARNING: In dynamic mode where host buffers are shared with an LRU cache or
    KV store, bounded-window staging pins host blocks while chunks are in
    flight.
    If the total allocated host pool is smaller than `num_slots * max_blocks`,
    unpinned blocks may be evicted prematurely or admission may stall.

    Args:
      kv_caches: List of device-placed contiguous Tensors representing the
        sharded KV caches.
      local_control_port: TCP socket server port for control plane coordination.
      max_blocks: Maximum number of blocks per staging slot.
      num_slots: Number of transfer slots to allocate.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
      host_blocks_to_allocate: Legacy/unified total blocks to allocate in host
        pool.
      parallelism: Number of parallel network copies per layer.
      node_id: Unique identifier for this host/node in the distributed mesh.
      raiden_worker_port: Optional port for WorkerService gRPC server.
      raiden_controller_address: Optional address of central RaidenController.
        If provided, the WorkerService gRPC server is enabled.
      worker_id: Optional identifier for this worker.
      enable_shm: Opt this manager's host buffers into the shared-memory
        segments named by RAIDEN_SHM_KEY. The env var supplies the segment
        namespace; this flag supplies the per-manager decision. Managers whose
        host buffers are transient staging must leave it off.
      backend_configs: Optional backend configurations (e.g. persistent
        storage). Pass the same BackendConfig objects given to the store's
        secondary_backend_configs. The posix backend derives its storage
        topology from `parallelism`: declare at least one axis (`tp`, `pcp`,
        `pp`) with `<axis>_size`, using the same axes and sizes as the store,
        and set `<axis>_rank` to this worker's rank in [0, size). Duplicate or
        missing ranks cause wrong or missed recalls. An invalid config aborts
        the process (LOG(FATAL)) at registration.
      staging_window_chunks: Number of active chunks in the bounded sliding
        window (W).
      staging_chunk_blocks: Number of KV blocks per window chunk (C).
      staging_chunk_bytes: Target chunk size in bytes (default 8MB).
      staging_stream_bytes: Target stream transfer chunk size in bytes (default
        4MB).
      staging_admission: Admission policy when budget is constrained:
        "fail_fast" or "wait".
      staging_admission_timeout_ms: Timeout in ms when staging_admission="wait".
      staging_assume_peer_support: Whether to assume peer support for windowed
        streaming. Staging options default to None. If all are None,
        configuration falls back to environment variables
        (TPU_RAIDEN_STAGING_*). Explicit kwargs always override environment
        variables and omitted kwargs keep env-resolved values or defaults.
    """
    has_staging_args = any(
        arg is not None
        for arg in (
            staging_window_chunks,
            staging_chunk_blocks,
            staging_chunk_bytes,
            staging_stream_bytes,
            staging_admission,
            staging_admission_timeout_ms,
            staging_assume_peer_support,
        )
    )
    if host_blocks_to_allocate is not None:
      if has_staging_args:
        raise ValueError(
            "Bounded-window host staging options cannot be combined with"
            " host_blocks_to_allocate. Use (max_blocks, num_slots) instead."
        )
      self._impl = _impl.KVCacheManager(
          kv_caches,
          local_control_port if local_control_port > 0 else None,
          host_blocks_to_allocate,
          unsafe_skip_buffer_lock,
          parallelism,
          raiden_worker_port,
          raiden_controller_address,
          worker_id,
          # Pass node_id so ReadRemote can always match src<->dst workers by
          # node_id (see the non-host_blocks branch, which already forwards it).
          node_id=node_id,
          enable_shm=enable_shm,
      )
    else:
      if max_blocks is None or num_slots is None:
        raise ValueError(
            "Must specify either (max_blocks, num_slots) or"
            " host_blocks_to_allocate."
        )
      self._impl = _impl.KVCacheManager(
          kv_caches=kv_caches,
          node_id=node_id,
          local_control_port=local_control_port,
          max_blocks=max_blocks,
          num_slots=num_slots,
          timeout_s=timeout_s,
          unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
          parallelism=parallelism,
          raiden_worker_port=raiden_worker_port,
          raiden_controller_address=raiden_controller_address,
          worker_id=worker_id,
          enable_shm=enable_shm,
          staging_window_chunks=staging_window_chunks,
          staging_chunk_blocks=staging_chunk_blocks,
          staging_chunk_bytes=staging_chunk_bytes,
          staging_stream_bytes=staging_stream_bytes,
          staging_admission=staging_admission,
          staging_admission_timeout_ms=staging_admission_timeout_ms,
          staging_assume_peer_support=staging_assume_peer_support,
      )
    if backend_configs:
      self._impl.register_kv_backends(list(backend_configs))

  def get_raiden_worker_port(self) -> int:
    """Returns the gRPC server port if running, or 0."""
    return self._impl.get_raiden_worker_port()

  @property
  def is_listener_active(self) -> bool:
    """Returns True if the worker gRPC service listener is active."""
    return self.get_raiden_worker_port() > 0

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the active Raiden endpoint descriptors."""
    return self._impl.get_local_endpoints()

  def register_read(self, req_id: str, uuid: int, block_ids: List[int]) -> bool:
    """Producer node notifies the registry/peer that blocks are ready for read.

    Args:
      req_id: The request ID of the transfer operation.
      uuid: The UUID of the request.
      block_ids: The list of block IDs to be read.

    Returns:
      True if a transfer is indeed needed; False if there is nothing to be
      transferred.

    Raises:
      ValueError: If `block_ids` contains duplicate block IDs.
      RuntimeError: If `uuid` already has an active registered read.
    """
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: Union[str, List[Dict[str, Any]]],
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> None:
    """Consumer node initiates an asynchronous pull of blocks from a remote peer."""
    self._impl.start_read(
        req_id,
        uuid,
        remote_endpoint,
        remote_block_ids,
        local_block_ids,
        parallelism,
    )

  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]:
    """Polls the status of all active background transfer operations.

    Returns:
      A tuple of (done_sending, done_recving, failed_recving) lists of request
      IDs.
    """
    return self._impl.complete_read()

  def d2h(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
  ) -> Any:
    """Device-to-Host (D2H) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.d2h(src_offsets, dst_offsets, copy_sizes)

  def h2d(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
  ) -> Any:
    """Host-to-Device (H2D) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.h2d(src_offsets, dst_offsets, copy_sizes)

  # =========================================================================
  # EXPERIMENTAL PHYSICAL CACHE MANAGEMENT APIs
  # The following APIs are experimental, expose physical cache internals,
  # and are subject to change in future releases.
  # =========================================================================

  def d2h_auto_allocate(
      self,
      src_offsets: List[int],
      copy_sizes: Optional[List[int]] = None,
  ) -> Tuple[List[int], Any]:
    """[EXPERIMENTAL] Device-to-Host (D2H) copy transfer with automatic host block allocation.

    WARNING: This API is experimental and subject to change in future releases.

    Args:
      src_offsets: Source block offsets on device.
      copy_sizes: Optional number of contiguous blocks to copy per segment.

    Returns:
      A tuple of (allocated_physical_chunk_ids, copy_future).
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.d2h_auto_allocate(src_offsets, copy_sizes)

  def unlock_blocks(self, block_ids: List[int]) -> None:
    """[EXPERIMENTAL] Unlocks the specified physical staging blocks on host.

    WARNING: This API is experimental and subject to change in future releases.

    Args:
      block_ids: List of physical chunk/block IDs to unlock.
    """
    self._impl.unlock_blocks(block_ids)

  def dump_metrics_to_string(self) -> str:
    """[EXPERIMENTAL] Dumps the metrics collector telemetry as a JSON string.

    WARNING: This API is experimental and subject to change in future releases.

    Returns:
      A JSON string representing the collected telemetry metrics.
    """
    return self._impl.dump_metrics_to_string()

  def set_staging_window(
      self,
      *,
      chunk_blocks: Optional[int] = None,
      chunk_bytes_target: Optional[int] = None,
      stream_bytes_target: Optional[int] = None,
      admission: Optional[str] = None,
      admission_timeout_ms: Optional[int] = None,
      assume_peer_support: Optional[bool] = None,
      window_chunks: Optional[int] = None,
      serve_windowed_pulls: Optional[bool] = None,
  ) -> None:
    """Updates runtime-mutable configuration for bounded-window host staging.

    Args:
      chunk_blocks: Number of blocks per chunk.
      chunk_bytes_target: Soft target chunk size in bytes.
      stream_bytes_target: Soft target stream size in bytes.
      admission: Admission policy string ("fail_fast" or "wait").
      admission_timeout_ms: Timeout in milliseconds when admission is "wait".
      assume_peer_support: Whether peer is assumed to support windowing.
      window_chunks: Construction-time field; mutating raises ValueError.
      serve_windowed_pulls: Construction-time field; mutating raises ValueError.

    Raises:
      ValueError: If a construction-time field (window_chunks or
        serve_windowed_pulls) is provided, or if an argument is invalid.
    """
    kwargs = {}
    if chunk_blocks is not None:
      kwargs["chunk_blocks"] = chunk_blocks
    if chunk_bytes_target is not None:
      kwargs["chunk_bytes_target"] = chunk_bytes_target
    if stream_bytes_target is not None:
      kwargs["stream_bytes_target"] = stream_bytes_target
    if admission is not None:
      kwargs["admission"] = admission
    if admission_timeout_ms is not None:
      kwargs["admission_timeout_ms"] = admission_timeout_ms
    if assume_peer_support is not None:
      kwargs["assume_peer_support"] = assume_peer_support
    if window_chunks is not None:
      kwargs["window_chunks"] = window_chunks
    if serve_windowed_pulls is not None:
      kwargs["serve_windowed_pulls"] = serve_windowed_pulls
    self._impl.set_staging_window(**kwargs)

  def staging_stats(self) -> Dict[str, int]:
    """Returns telemetry counters and watermark metrics for host staging.

    Returns:
      A dictionary containing 9 fields:
        - budget_blocks: Host staging block pool budget (num_slots *
        max_blocks).
        - blocks_in_use: Number of staging blocks currently pinned.
        - peak_blocks_in_use: High-water mark of pinned staging blocks.
        - gate_waits_over_1ms: Number of times staging gate waited >1ms.
        - rejected_streams: Number of streams rejected by admission control.
        - admission_rejects: Streams rejected due to insufficient budget.
        - admission_waits: Streams that waited for budget before admission.
        - admission_requeues: Streams requeued during admission.
        - admission_timeouts: Streams that timed out waiting for budget.
    """
    return self._impl.staging_stats()

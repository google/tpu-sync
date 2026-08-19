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

from typing import Any, Dict, List, Optional, Tuple, Union
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
  ):
    """Instantiates the TransferEngine-based KVCacheManager.

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
    """
    if host_blocks_to_allocate is not None:
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
      )

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
    """
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def renew_leases(self, remote_endpoint: str, uuids: List[int]) -> List[int]:
    """Renews queued producer sends without extending them indefinitely.

    Results align positionally with ``uuids``: 1 means applied, 0 is not yet
    registered, -1 is terminal/expired, -2 is already transferring, and -3
    reached the producer's absolute retention cap.
    """
    return self._impl.renew_leases(remote_endpoint, uuids)

  def cancel_leases(self, remote_endpoint: str, uuids: List[int]) -> List[int]:
    """Releases queued producer sends that will no longer be pulled.

    Status values match :meth:`renew_leases`. A claimed transfer is never
    interrupted and returns -2.
    """
    return self._impl.cancel_leases(remote_endpoint, uuids)

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

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
# pool_layout is framework-neutral (pure dataclasses, no torch import, and
# tpu_raiden ships as a namespace package so importing it pulls nothing else
# from the torch API). It is shared rather than duplicated so the JAX and
# torch pool descriptors cannot drift.
from tpu_sync.api.torch import pool_layout
from tpu_sync.frameworks.jax import _tpu_raiden_jax as _impl

PoolSpec = pool_layout.PoolSpec
RegionSpec = pool_layout.RegionSpec


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
      listener_port: Optional[int] = None,
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
      listener_port: Optional port for the control-plane KVCacheListener. A
        RaidenController talks to this socket to arm a receiver
        (PoolReshardRegisterRecv) and to fire a sender (PoolReshardPush) for a
        pool-addressed reshard plan. Distinct from raiden_worker_port, which
        is the buffer-oriented WorkerService gRPC server. Requires a manager
        with exactly one NUMA sub-manager.
    """
    self._admission_summary: Optional[Dict[str, Any]] = None
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
          listener_port=listener_port,
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
          listener_port=listener_port,
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

  # =========================================================================
  # POOL-ADDRESSED RESHARD APIs
  # Mirrors api/torch/kv_cache_manager.py so a caller (e.g. the TPU vLLM
  # connector) can drive either framework through the same names and shapes.
  # =========================================================================

  @property
  def listener_port(self) -> Optional[int]:
    """Returns the control-plane KVCacheListener port, or None if disabled."""
    return self._impl.listener_port

  @property
  def is_control_listener_active(self) -> bool:
    """Returns True if the control-plane KVCacheListener is running.

    Distinct from ``is_listener_active``, which reports on the buffer-oriented
    WorkerService gRPC server. The two are separate sockets and either can be
    up without the other.
    """
    return self._impl.is_listener_active

  @property
  def listener_address(self) -> str:
    """Returns the formatted control listener endpoint string (host:port)."""
    return self._impl.listener_address

  @property
  def transfer_address(self) -> str:
    """Returns the formatted data transfer endpoint string (host:port)."""
    return self._impl.transfer_address

  def register_pools(self, pools: Sequence[Any]) -> Dict[str, Any]:
    """Registers explicit block pools over the wrapped storages.

    Args:
      pools: Sequence of ``pool_layout.PoolSpec`` (or equivalent mappings) in
        the caller's canonical order. Pool indices travel on the wire, so both
        transfer peers must agree on this order.

    Returns:
      A generic admission summary (also served by ``admission_summary``).
    """
    coerced = tuple(pool_layout.coerce_pool_spec(pool) for pool in pools)
    if not coerced:
      raise ValueError("pool table must be non-empty")
    num_storages = int(self._impl.num_layers)
    for pool_idx, pool in enumerate(coerced):
      try:
        pool.validate()
      except Exception as exc:
        raise ValueError(f"invalid pool {pool_idx}: {exc}") from exc
      if pool.storage_index >= num_storages:
        raise ValueError(
            f"pool {pool_idx} ({pool.tag}) storage_index "
            f"{pool.storage_index} out of range: manager wraps "
            f"{num_storages} storages"
        )
    self._impl.register_pools_native(
        [pool.to_native_tuple() for pool in coerced]
    )
    tags: Dict[str, int] = {}
    for pool in coerced:
      tags[pool.tag] = tags.get(pool.tag, 0) + 1
    storages = len({pool.storage_index for pool in coerced})
    summary = {
        "admitted": True,
        "pools": len(coerced),
        "storages": storages,
        "tags": tags,
    }
    self._admission_summary = dict(summary)
    return dict(summary)

  def get_block_ref(
      self, pool_idx: int, block_id: int, shard_idx: int = 0
  ) -> Dict[str, Any]:
    """Returns a reference descriptor for one host-side pool block."""
    return dict(
        self._impl.get_pool_block_ref_native(
            pool_idx=pool_idx, shard_idx=shard_idx, block_id=block_id
        )
    )

  def pool_ids_with_tag(self, tag: str) -> List[int]:
    """Returns the pool indices registered with the given opaque tag."""
    return [
        int(pool_idx)
        for pool_idx in self._impl.pool_indices_with_tag_native(tag)
    ]

  def num_pools(self) -> int:
    """Returns the number of pools (implicit or explicit)."""
    return int(self._impl.num_pools())

  def has_explicit_pools(self) -> bool:
    """Returns True if an explicit pool table was admitted."""
    return bool(self._impl.has_explicit_pools())

  def pool_spec(self, pool_idx: int) -> Dict[str, Any]:
    """Returns one pool's descriptor as a dict."""
    return dict(self._impl.pool_spec_native(pool_idx))

  def d2h_pool_blocks(
      self,
      pool_idx: int,
      block_ids: Sequence[int],
      shard_idx: Optional[int] = None,
  ) -> Any:
    """Partial D2H of whole pool blocks into the host mirror."""
    return self._impl.d2h_pool_blocks(pool_idx, list(block_ids), shard_idx)

  def h2d_pool_blocks(
      self,
      pool_idx: int,
      block_ids: Sequence[int],
      shard_idx: Optional[int] = None,
  ) -> Any:
    """Partial H2D of whole pool blocks from the host mirror."""
    return self._impl.h2d_pool_blocks(pool_idx, list(block_ids), shard_idx)

  def admission_summary(self) -> Dict[str, Any]:
    """Returns the last successful pool admission summary."""
    if self._admission_summary is None:
      return {"admitted": False}
    return dict(self._admission_summary)

  def register_active_plan(
      self, uuid: int, request: Union[bytes, Any], is_sender: bool
  ) -> None:
    """Registers a serialized StartTransferRequest for strided push."""
    if hasattr(request, "SerializeToString"):
      request = request.SerializeToString()
    if not isinstance(request, (bytes, bytearray)):
      raise TypeError("request must be bytes or a protobuf message")
    self._impl.register_active_plan(uuid, bytes(request), is_sender)

  def unregister_active_plan(self, uuid: int) -> None:
    """Removes a previously registered strided push plan."""
    self._impl.unregister_active_plan(uuid)

  def register_recv(
      self, uuid: int, req_id: str, expected_block_count: int
  ) -> None:
    """[EXPERIMENTAL] Registers expected incoming blocks for push resharding.

    Allocates staging slots in the C++ receiver engine and sets the barrier for
    the expected physical block-pushes; the engine triggers H2D into TPU HBM
    once the count is reached.

    Args:
      uuid: Unique identifier for the transfer transaction.
      req_id: Request ID associated with the transfer.
      expected_block_count: Total number of physical block-pushes expected from
        all contributing source ranks.
    """
    self._impl.register_recv(uuid, req_id, expected_block_count)

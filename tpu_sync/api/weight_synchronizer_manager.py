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

"""High-performance Weight Synchronizer Manager for distributed RL Trainer-Inference Pipelines.

This module provides the `WeightSynchronizerManager` control plane service,
primarily instantiated by Reinforcement Learning (RL) orchestrator / coordinator
jobs to orchestrate zero-copy weight synchronization and resharding between
trainer jobs and sampler/inference rollouts.

Overview
--------
In large-scale distributed RL pipelines (e.g. actor-critic, PPO, GRPO), training
and inference frequently execute in separate jobs or clusters with distinct TPU
slice topologies and sharding strategies (e.g., trainer on Megacore with Tensor
Parallelism TP=8 vs. samplers across multiple hosts with TP=2 / DP=4).

The `WeightSynchronizerManager`:
1. Runs as a centralized control-plane coordinator service in the RL
   orchestrator job.
2. Tracks registered work units and physical transfer endpoints published by
   source (trainer) and destination (sampler) workers.
3. Automatically computes optimal multi-dimensional tensor resharding and
   point-to-point network transfer schedules across disparate TPU meshes.
4. Triggers asynchronous push/pull transfers directly between worker nodes
   over zero-copy TCP streams without passing data through the orchestrator.

Typical RL Orchestrator Usage
-----------------------------
```python
from tpu_sync.api.weight_synchronizer_manager import WeightSynchronizerManager
from tpu_sync.rpc.raiden_controller import RaidenId

# 1. Instantiate and start the manager in the RL orchestrator job
manager = WeightSynchronizerManager(port=10000, auto_start_server=True)

# 2. Define source (trainer) and destination (sampler) unit IDs
src_unit = RaidenId(
    job_name="trainer",
    job_replica_id="0",
    data_name="qwen_weights",
)
dst_units = [
    RaidenId(
        job_name="sampler",
        job_replica_id=str(i),
        data_name="qwen_weights",
    )
    for i in range(num_sampler_hosts)
]

# 3. In the training loop: trigger weight sync from trainer to samplers
for step in range(num_training_steps):
  # Trainer completes optimization step...
  # Orchestrator triggers weight distribution to all rollout samplers:
  manager.start_transfer(
      src_units=[src_unit],
      dst_units=dst_units,
  )
  # Samplers continue rollout with newly updated weights...

manager.close()
```
"""

from typing import Any, Optional, Sequence

from tpu_sync.rpc import raiden_controller


class WeightSynchronizerManager:
  """Centralized Weight Synchronizer Manager orchestrating distributed RL weight transfers.

  The manager is typically instantiated by a Reinforcement Learning (RL)
  orchestrator or coordinator job to manage weight synchronization between
  trainer instances (sources) and sampler/inference instances (destinations).

  Key Responsibilities:
  - Control-Plane Service: Hosts the RPC server for worker registration and
    transfer dispatch.
  - Resharding Planning: Automatically calculates sub-tensor slice mappings
    when trainer and sampler jobs use different TPU mesh configurations or
    partition specifications.
  - Multi-Host Fan-out: Orchestrates direct peer-to-peer or tree-based
    broadcast transfers from the trainer to multiple rollout sampler hosts.
  """

  def __init__(
      self,
      port: int = 0,
      worker_rpc_client: Optional[raiden_controller.WorkerRpcClient] = None,
      request_registry_ttl_s: float = 600.0,
      broadcast_k: Optional[int] = None,
      enable_plan_cache: bool = True,
      auto_start_server: bool = False,
  ):
    """Initializes the WeightSynchronizerManager.

    Args:
      port: Port number the manager / controller service runs on. A port of 0
        selects an ephemeral available port upon starting the server.
      worker_rpc_client: Optional worker RPC client facade for dispatching RPCs.
      request_registry_ttl_s: TTL in seconds for request registry entries.
      broadcast_k: Fan-out factor K for tree-based broadcast transfers across
        multiple sampler nodes.
      enable_plan_cache: Whether to cache transfer planning and resharding
        schedules across transfer invocations with identical topologies.
      auto_start_server: Whether to automatically spawn the background TCP
        servicer loop on initialization.
    """
    self._controller = raiden_controller.RaidenController(
        port=port,
        worker_rpc_client=worker_rpc_client,
        request_registry_ttl_s=request_registry_ttl_s,
        broadcast_k=broadcast_k,
        enable_plan_cache=enable_plan_cache,
    )
    self._server: Optional[raiden_controller.RaidenControllerServer] = None
    if auto_start_server:
      self.start_server()

  @property
  def port(self) -> int:
    """Returns the active TCP listener port coordinate."""
    return self._controller.port

  @port.setter
  def port(self, value: int) -> None:
    self._controller.port = value

  def register_work_unit(
      self,
      unit: raiden_controller.RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      mesh_shape: Optional[Sequence[int]] = None,
      layout: Optional[Sequence[int]] = None,
      global_shape: Optional[Sequence[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[Sequence[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
      variables: Optional[Sequence[Any]] = None,
      mesh_axes: Optional[Sequence[str]] = None,
  ) -> None:
    """Registers physical worker shard Data addresses and metadata.

    Args:
      unit: Work unit identifier owning the data shards.
      shards: list of physical Data TCP addresses (e.g. 'IP:Port').
      control_plane_rpc_address: Optional worker Control-Plane RPC endpoint.
      mesh_shape: Optional logical mesh shape for reshard planning.
      layout: Optional minor-to-major mapping layout.
      global_shape: Optional global array shape.
      itemsize: Optional item size in bytes.
      pool_manifest: Optional manifest of memory pools.
      layout_fingerprint: Optional layout fingerprint string.
      page_tokens: Optional page token size.
      transfer_parallelism: Optional transfer parallelism factor.
      transfer_rank: Optional transfer rank.
      variables: Optional list of registered variables metadata.
      mesh_axes: Optional list of mesh axis names (e.g. ['fsdp', 'tp']).
    """
    return self._controller.register_work_unit(
        unit=unit,
        shards=shards,
        control_plane_rpc_address=control_plane_rpc_address,
        mesh_shape=mesh_shape,
        layout=layout,
        global_shape=global_shape,
        itemsize=itemsize,
        pool_manifest=pool_manifest,
        layout_fingerprint=layout_fingerprint,
        page_tokens=page_tokens,
        transfer_parallelism=transfer_parallelism,
        transfer_rank=transfer_rank,
        variables=variables,
        mesh_axes=mesh_axes,
    )

  def clear_plan_cache(self) -> None:
    """Clears all cached transfer schedules."""
    return self._controller.clear_plan_cache()

  def get_plan_cache_size(self) -> int:
    """Returns the number of cached transfer schedules."""
    return self._controller.get_plan_cache_size()

  def get_all_metadata(self) -> list[Any]:
    """Returns replacement-safe metadata for all registered work units."""
    return self._controller.get_all_metadata()

  def get_plan(self, req_id: str) -> Optional[raiden_controller.TransferPlan]:
    """Returns the generated TransferPlan for a given transfer request ID."""
    return self._controller.get_plan(req_id)

  def start_transfer(
      self,
      src_units: list[raiden_controller.RaidenId],
      dst_units: list[raiden_controller.RaidenId],
      req_id: Optional[str] = None,
      src_block_ids: Optional[list[int]] = None,
      dst_device_block_ids: Optional[list[int]] = None,
      dst_mem_type: Optional[int] = None,
      use_block_chunks: bool = False,
      src_controller_address: Optional[str] = None,
      dst_controller_address: Optional[str] = None,
      uuid: Optional[str] = None,
      is_sender: bool = False,
      expected_block_count: int = 0,
      shard_push_schedules: Optional[dict[Any, Any]] = None,
      num_tokens: Optional[int] = None,
      transfer_pool_tags: Optional[list[int]] = None,
      parallelism: int = 1,
      skip_d2h: bool = False,
      dst_block_counts: Optional[dict[Any, int]] = None,
      skip_tiling: Optional[dict[int, bool]] = None,
      group_size: int = 1,
      use_cached_plan: bool = False,
  ) -> str:
    """Initiates an asynchronous distributed transfer.

    Args:
      src_units: list of source RaidenId work units.
      dst_units: list of destination RaidenId work units.
      req_id: Optional request identifier.
      src_block_ids: Source device block IDs.
      dst_device_block_ids: Destination device block IDs.
      dst_mem_type: Destination memory type enum.
      use_block_chunks: Whether block transfers are divided into chunks.
      src_controller_address: Source-side controller BNS endpoint.
      dst_controller_address: Destination-side controller BNS endpoint.
      uuid: Optional unique transfer session UUID.
      is_sender: True if this controller is acting on behalf of sender nodes.
      expected_block_count: Expected total blocks transferred.
      shard_push_schedules: Pre-calculated shard push schedules.
      num_tokens: Optional number of tokens transferred.
      transfer_pool_tags: Memory pool tags.
      parallelism: Parallel transfer stream count.
      skip_d2h: Whether to skip D2H staging.
      dst_block_counts: Optional block counts per destination unit.
      skip_tiling: Optional per-device tiling bypass settings.
      group_size: Variable group size for broadcast pipeline.
      use_cached_plan: Whether to reuse precomputed plans.

    Returns:
      Request ID string of the started transfer task.
    """
    return self._controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        req_id=req_id,
        src_block_ids=src_block_ids,
        dst_device_block_ids=dst_device_block_ids,
        dst_mem_type=dst_mem_type,
        use_block_chunks=use_block_chunks,
        src_controller_address=src_controller_address,
        dst_controller_address=dst_controller_address,
        uuid=uuid,
        is_sender=is_sender,
        expected_block_count=expected_block_count,
        shard_push_schedules=shard_push_schedules,
        num_tokens=num_tokens,
        transfer_pool_tags=transfer_pool_tags,
        parallelism=parallelism,
        skip_d2h=skip_d2h,
        dst_block_counts=dst_block_counts,
        skip_tiling=skip_tiling,
        group_size=group_size,
        use_cached_plan=use_cached_plan,
    )

  def get_transfer_status(self, req_id: str) -> int:
    """Returns the current execution status enum for a transfer request ID."""
    return self._controller.get_transfer_status(req_id)

  def start_server(self) -> int:
    """Starts the background RPC server for this manager if not already running.

    Returns:
      The active TCP listener port coordinate.
    """
    if self._server is None:
      self._server = raiden_controller.RaidenControllerServer(self._controller)
      self.port = self._server.start()
    return self.port

  def stop_server(self) -> None:
    """Stops the background RPC server if running."""
    if self._server is not None:
      self._server.stop()
      self._server = None

  def close(self) -> None:
    """Closes the manager and stops any background server threads."""
    self.stop_server()

  def __enter__(self) -> "WeightSynchronizerManager":
    self.start_server()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.stop_server()


__all__ = ["WeightSynchronizerManager"]

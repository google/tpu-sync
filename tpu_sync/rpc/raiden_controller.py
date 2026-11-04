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

"""High-level RaidenController orchestrating JobEntities and ReshardPlanner."""

import asyncio
import os
import random
import threading
import time
from typing import Any, Optional, Sequence

from absl import logging

from tpu_sync.api import common
from tpu_sync.api.common import RaidenId
from tpu_sync.common.control_pipe import control_pipe_client  # pylint: disable=unused-import
from tpu_sync.kv_cache import nd_slice_math  # pylint: disable=unused-import
from tpu_sync.rpc import controller_service_pb2
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import broadcast_engine
from tpu_sync.weight_sync.manager import controller_rpc_service
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity
from tpu_sync.weight_sync.manager import reshard_planner

# Re-export symbols across all layers for backward compatibility.
BroadcastEngine = broadcast_engine.BroadcastEngine

NDSlice = controller_types.NDSlice
NameResolver = controller_types.NameResolver
RaidenFuture = controller_types.RaidenFuture
RaidenMemoryType = controller_types.RaidenMemoryType
TransferPlan = controller_types.TransferPlan
_CachedTransferSchedule = controller_types.CachedTransferSchedule
_VariableMetadata = controller_types.VariableMetadata
_coerce_pool_spec_proto = controller_types.coerce_pool_spec_proto
_coerce_variable_proto = controller_types.coerce_variable_proto
_extract_host_ip = controller_types.extract_host_ip
_format_unit = controller_types.format_unit
_format_units = controller_types.format_units
_is_variable_spec_identical = controller_types.is_variable_spec_identical
_proto_to_nd_slice = controller_types.proto_to_nd_slice
_raiden_id_from_proto = controller_types.raiden_id_from_proto
_raiden_id_to_proto = controller_types.raiden_id_to_proto

HostDescriptor = job_entity.HostDescriptor
HostGroup = job_entity.HostGroup
JobEntity = job_entity.JobEntity
WeightSyncWorkerRpcClient = job_entity.WeightSyncWorkerRpcClient
WorkerRpcClient = job_entity.WorkerRpcClient

ReshardPlanner = reshard_planner.ReshardPlanner
_get_global_indices = reshard_planner.get_global_indices
generate_strided_copy_chunks = reshard_planner.generate_strided_copy_chunks
generate_strided_copy_chunks_tile_aware = (
    reshard_planner.generate_strided_copy_chunks_tile_aware
)
intersect_nd_slices = reshard_planner.intersect_nd_slices
is_nd_slice_tile_aligned = reshard_planner.is_nd_slice_tile_aligned
to_physical = reshard_planner.to_physical

RaidenControllerClientFacade = (
    controller_rpc_service.RaidenControllerClientFacade
)
RaidenControllerServer = controller_rpc_service.RaidenControllerServer


class _EntityBroadcastDispatcher:
  """Adapter allowing BroadcastEngine to dispatch transfers directly to JobEntities."""

  def __init__(self, controller: "RaidenController") -> None:
    self._controller = controller

  @property
  def name_resolver(self) -> Optional[NameResolver]:
    return self._controller.name_resolver

  @property
  def executor(self) -> Any:
    return self._controller.executor

  def include_receiver_push_schedules(
      self, transfer_plan: Optional[TransferPlan] = None
  ) -> bool:
    return self._controller.include_receiver_push_schedules(transfer_plan)

  async def start_transfer(
      self, unit: RaidenId, transfer_plan: TransferPlan
  ) -> None:
    await self._controller._dispatch_entity_transfer(unit, transfer_plan)


class RaidenController:
  """High-level transfer controller managing JobEntities and generating resharding plans."""

  def __init__(
      self,
      port: int,
      worker_rpc_client: Optional[WorkerRpcClient] = None,
      request_registry_ttl_s: float = 600.0,
      broadcast_host_ratio: Optional[float] = None,
      broadcast_pipeline_stages: Optional[int] = None,
      enable_plan_cache: bool = True,
  ):
    """Initializes the RaidenController.

    Args:
      port: Port number the controller service runs on.
      worker_rpc_client: Optional legacy worker RPC client override wrapped into
        managed JobEntity instances for backward compatibility.
      request_registry_ttl_s: TTL in seconds for request registry entries.
      broadcast_host_ratio: Ratio of trainer TX host bandwidth to sampler RX
        host bandwidth (K0 = B_train_TX / B_sample_RX). Values <= 0.0 or None
        are normalized to 1.0.
      broadcast_pipeline_stages: Target number of pipelined broadcast stages
        (default 4 or RAIDEN_BROADCAST_PIPELINE_STAGES).
      enable_plan_cache: Whether to cache transfer planning and resharding
        schedules across transfer invocations with identical topologies.
    """
    self.port = port
    raw_ratio = (
        broadcast_host_ratio
        if broadcast_host_ratio is not None
        else float(os.environ.get("RAIDEN_BROADCAST_HOST_RATIO", "1.0"))
    )
    self.broadcast_host_ratio = raw_ratio
    raw_stages = (
        broadcast_pipeline_stages
        if broadcast_pipeline_stages is not None
        else int(os.environ.get("RAIDEN_BROADCAST_PIPELINE_STAGES", "4"))
    )
    if raw_stages < 1:
      raise ValueError(
          f"broadcast_pipeline_stages must be >= 1, got {raw_stages}"
      )
    self.broadcast_pipeline_stages = int(raw_stages)
    self.enable_plan_cache = enable_plan_cache
    self._plan_cache: dict[Any, _CachedTransferSchedule] = {}
    self._active_transfers: dict[str, TransferPlan] = {}
    self._active_tasks: dict[str, RaidenFuture] = {}
    self._task_units: dict[str, list[RaidenId]] = {}
    self._src_replica_counts: dict[str, int] = {}
    self._dst_replica_counts: dict[str, int] = {}
    self._next_session_id: int = 0
    self._entities: dict[RaidenId, JobEntity] = {}
    self._registered_shards: dict[RaidenId, list[str]] = {}
    self._registered_mesh_shapes: dict[RaidenId, list[int]] = {}
    self._registered_mesh_axes: dict[RaidenId, list[str]] = {}
    self._registered_layouts: dict[RaidenId, list[int]] = {}
    self._registered_global_shapes: dict[RaidenId, list[int]] = {}
    self._registered_itemsizes: dict[RaidenId, int] = {}
    self._computed_phys_meshes: dict[RaidenId, list[int]] = {}
    self._lock = threading.RLock()
    self._registered_pool_manifests: dict[RaidenId, list[Any]] = {}
    self._registered_layout_fingerprints: dict[RaidenId, str] = {}
    self._registered_page_tokens: dict[RaidenId, int] = {}
    self._registered_transfer_parallelism: dict[RaidenId, int] = {}
    self._registered_transfer_ranks: dict[RaidenId, int] = {}
    if request_registry_ttl_s <= 0:
      raise ValueError("request_registry_ttl_s must be positive")
    self._request_registry_ttl_s = request_registry_ttl_s
    self._legacy_rpc_client = worker_rpc_client or WorkerRpcClient()
    if hasattr(self._legacy_rpc_client, "bind_entities"):
      self._legacy_rpc_client.bind_entities(self._entities)
      for ent in self._entities.values():
        ent.set_on_update(self._sync_from_entity)
        self._sync_from_entity(ent)
    self._broadcast_engine = BroadcastEngine(
        _EntityBroadcastDispatcher(self),
        remote_controller_client_factory=RaidenControllerClientFacade,
    )
    self._registered_variables: dict[RaidenId, list[Any]] = {}
    self._registered_control_plane_endpoints: dict[RaidenId, list[str]] = {}
    self._planner = ReshardPlanner()

  @property
  def broadcast_host_ratio(self) -> float:
    """Ratio of trainer TX host bandwidth to sampler RX host bandwidth."""
    return self._broadcast_host_ratio

  @broadcast_host_ratio.setter
  def broadcast_host_ratio(self, value: Optional[float]) -> None:
    if value is None or value <= 0.0:
      self._broadcast_host_ratio = 1.0
    else:
      self._broadcast_host_ratio = float(value)

  @property
  def worker_rpc_client(self) -> WorkerRpcClient:
    """Compatibility property returning the entity-bound WorkerRpcClient view."""
    return self._legacy_rpc_client

  @worker_rpc_client.setter
  def worker_rpc_client(self, client: WorkerRpcClient) -> None:
    self._legacy_rpc_client = client
    if hasattr(client, "bind_entities"):
      client.bind_entities(self._entities)

  @property
  def name_resolver(self) -> Optional[NameResolver]:
    """Returns the NameResolver configured across managed JobEntities."""
    return getattr(self._legacy_rpc_client, "name_resolver", None)

  @property
  def executor(self) -> Any:
    """Returns the RPC executor configured across managed JobEntities."""
    return getattr(self._legacy_rpc_client, "executor", None)

  def include_receiver_push_schedules(
      self, transfer_plan: Optional[TransferPlan] = None
  ) -> bool:
    """Returns whether receiver StartTransferRequests need shard_push_schedules."""
    if hasattr(self._legacy_rpc_client, "include_receiver_push_schedules"):
      return self._legacy_rpc_client.include_receiver_push_schedules(
          transfer_plan
      )
    if transfer_plan is not None and getattr(
        transfer_plan, "is_weight_sync", False
    ):
      return False
    return True

  def get_entity_rpc_addresses(self) -> dict[RaidenId, str]:
    """Returns control-plane endpoint strings across all managed JobEntities."""
    with self._lock:
      res: dict[RaidenId, str] = {}
      for u, ent in self._entities.items():
        for unit_key, unit_eps in ent._endpoints_by_unit.items():  # pylint: disable=protected-access
          if unit_eps:
            res[unit_key] = ",".join(unit_eps)
        if ent._endpoints_by_replica:  # pylint: disable=protected-access
          for rep_id, eps in ent._endpoints_by_replica.items():  # pylint: disable=protected-access
            if eps:
              ep_str = ",".join(eps)
              units = ent._units_by_replica.get(rep_id) or [  # pylint: disable=protected-access
                  RaidenId(
                      job_name=ent.unit.job_name,
                      job_replica_id=rep_id,
                      data_name=ent._default_data_name,  # pylint: disable=protected-access
                      data_replica_idx=ent._default_data_replica_idx,  # pylint: disable=protected-access
                  )
              ]
              for host_unit in units:
                unit_eps = ent._endpoints_by_unit.get(host_unit)  # pylint: disable=protected-access
                res[host_unit] = ",".join(unit_eps) if unit_eps else ep_str
        elif ent.get_registered_endpoints():
          res[u] = ent.get_worker_endpoint_str()
      for (
          unit_key,
          unit_eps,
      ) in self._registered_control_plane_endpoints.items():
        if unit_key not in res and unit_eps:
          res[unit_key] = ",".join(unit_eps)
      return res

  @property
  def worker_endpoints(self) -> dict[RaidenId, str]:
    """Returns control-plane endpoint strings keyed by RaidenId."""
    return self.get_entity_rpc_addresses()

  def _sync_from_entity(self, entity: JobEntity) -> None:
    """Synchronizes legacy registry mirrors when a JobEntity is modified."""
    with self._lock:
      if entity._endpoints_by_replica:  # pylint: disable=protected-access
        for rep_id, eps in entity._endpoints_by_replica.items():  # pylint: disable=protected-access
          units = entity._units_by_replica.get(rep_id) or [  # pylint: disable=protected-access
              RaidenId(
                  job_name=entity.unit.job_name,
                  job_replica_id=rep_id,
                  data_name=entity._default_data_name,  # pylint: disable=protected-access
                  data_replica_idx=entity._default_data_replica_idx,  # pylint: disable=protected-access
              )
          ]
          rep_shards = entity._shards_by_replica.get(rep_id)  # pylint: disable=protected-access
          for host_unit in units:
            unit_eps = entity._endpoints_by_unit.get(host_unit, eps)  # pylint: disable=protected-access
            unit_shards = entity._shards_by_unit.get(host_unit, rep_shards)  # pylint: disable=protected-access
            if unit_eps:
              self._registered_control_plane_endpoints[host_unit] = list(
                  unit_eps
              )
            if unit_shards:
              self._registered_shards[host_unit] = list(unit_shards)
      else:
        unit = entity.unit
        if entity.shards:
          self._registered_shards[unit] = list(entity.shards)
        if entity.host_endpoints:
          self._registered_control_plane_endpoints[unit] = list(
              entity.host_endpoints
          )

  @property
  def entities(self) -> dict[RaidenId, JobEntity]:
    """Returns a read-only snapshot of all managed JobEntity instances."""
    with self._lock:
      return dict(self._entities)

  def get_entity(self, unit: RaidenId) -> Optional[JobEntity]:
    """Returns the JobEntity managed for `unit`, or None if unregistered."""
    key = controller_types.entity_key_from_unit(unit)
    with self._lock:
      return self._entities.get(key)

  def get_or_create_entity(self, unit: RaidenId) -> JobEntity:
    """Returns the existing JobEntity for `unit` (by entity key) or creates a new one."""
    key = controller_types.entity_key_from_unit(unit)
    with self._lock:
      ent = self._entities.get(key)
      if ent is None:
        if hasattr(self._legacy_rpc_client, "get_or_create_entity"):
          ent = self._legacy_rpc_client.get_or_create_entity(unit)
        else:
          ent = JobEntity(unit=key)
        ent.set_on_update(self._sync_from_entity)
        self._entities[key] = ent
      if getattr(unit, "data_name", "") and not ent._default_data_name:  # pylint: disable=protected-access
        ent._default_data_name = unit.data_name  # pylint: disable=protected-access
      if (
          getattr(unit, "data_replica_idx", 0)
          and not ent._default_data_replica_idx  # pylint: disable=protected-access
      ):
        ent._default_data_replica_idx = unit.data_replica_idx  # pylint: disable=protected-access
      rep_id = getattr(unit, "job_replica_id", "") or ""
      if rep_id and isinstance(unit, RaidenId):
        rep_units = ent._units_by_replica.setdefault(rep_id, [])  # pylint: disable=protected-access
        if unit not in rep_units:
          rep_units.append(unit)
      return ent

  def attach_host(
      self,
      unit: RaidenId,
      control_address: str,
      shards: Optional[Sequence[str]] = None,
  ) -> JobEntity:
    """Attaches a host endpoint (`unit.job_replica_id`) and local shards to `unit`'s JobEntity.

    Args:
      unit: Host work unit identifier (e.g. trainer or sampler host).
      control_address: Host Control-Plane RPC endpoint address.
      shards: Optional data-plane shard addresses served by this host.

    Returns:
      The updated JobEntity instance managing `unit`.
    """
    with self._lock:
      ent = self.get_or_create_entity(unit)
      ent.attach_host(control_address=control_address, shards=shards, unit=unit)
      rep_id = getattr(unit, "job_replica_id", "") or ""
      rep_shards = ent.get_shards_for_unit(unit)
      if rep_shards:
        self._registered_shards[unit] = list(rep_shards)
      self._registered_control_plane_endpoints[unit] = list(
          ent.get_registered_endpoints(rep_id)
      )
      return ent

  async def _dispatch_entity_transfer(
      self,
      unit: RaidenId,
      transfer_plan: TransferPlan,
  ) -> None:
    """Delegates transfer command dispatch directly to `unit`'s JobEntity."""
    ent = self.get_or_create_entity(unit)
    await ent.start_transfer(transfer_plan, unit=unit)

  async def shutdown_entities(self, timeout: float = 10.0) -> None:
    """Dispatches remote shutdown commands across all managed JobEntities."""
    if self._legacy_rpc_client is not None and (
        "shutdown_workers" in getattr(self._legacy_rpc_client, "__dict__", {})
        or type(self._legacy_rpc_client).shutdown_workers
        not in (
            WorkerRpcClient.shutdown_workers,
            WeightSyncWorkerRpcClient.shutdown_workers,
        )
    ):
      await self._legacy_rpc_client.shutdown_workers()
      return
    with self._lock:
      entities = list(self._entities.values())
    if entities:
      await asyncio.gather(
          *[ent.shutdown_hosts(timeout=timeout) for ent in entities],
          return_exceptions=True,
      )

  def _prune_completed_transfers_locked(self, max_completed: int = 16) -> None:
    """Evicts oldest completed transfer records to prevent unbounded memory growth."""
    completed_reqs = [
        r
        for r in self._active_transfers
        if r not in self._active_tasks or self._active_tasks[r].done()
    ]
    if len(completed_reqs) > max_completed:
      for r in completed_reqs[:-max_completed]:
        self._active_transfers.pop(r, None)
        self._active_tasks.pop(r, None)
        self._task_units.pop(r, None)

  def register_work_unit(
      self,
      unit: RaidenId,
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
    """Registers physical worker shard Data addresses and metadata on `unit`'s JobEntity."""
    has_metadata = (
        mesh_shape is not None or layout is not None or global_shape is not None
    )
    if has_metadata and not variables:
      if mesh_shape is None or layout is None or global_shape is None:
        raise ValueError(
            "If any of mesh_shape, layout, or global_shape is provided, "
            "all of them must be provided to enable centralized slice planning."
        )
      if itemsize is None or itemsize <= 0:
        raise ValueError(
            "itemsize must be provided and must be greater than 0 if resharding"
            " metadata is provided."
        )

    has_reshard_metadata = any(
        value is not None
        for value in (
            pool_manifest,
            layout_fingerprint,
            page_tokens,
            transfer_parallelism,
            transfer_rank,
        )
    )
    if has_reshard_metadata:
      if (
          pool_manifest is None
          or layout_fingerprint is None
          or page_tokens is None
          or transfer_parallelism is None
          or transfer_rank is None
      ):
        raise ValueError(
            "pool_manifest, layout_fingerprint, page_tokens, "
            "transfer_parallelism, and transfer_rank must be provided "
            "together for pool resharding"
        )
      if not pool_manifest:
        raise ValueError("pool_manifest must not be empty")
      if not layout_fingerprint:
        raise ValueError("layout_fingerprint must not be empty")
      if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
      if transfer_parallelism <= 0:
        raise ValueError("transfer_parallelism must be positive")
      if transfer_rank < 0:
        raise ValueError("transfer_rank must be non-negative")
      if transfer_rank >= transfer_parallelism:
        raise ValueError("transfer_rank must be less than transfer_parallelism")
      normalized_pools = [_coerce_pool_spec_proto(p) for p in pool_manifest]
    else:
      normalized_pools = []

    if not shards or any(not shard for shard in shards):
      raise ValueError("shards must contain at least one non-empty endpoint")

    endpoints: list[str] = []
    if control_plane_rpc_address:
      if isinstance(control_plane_rpc_address, (list, tuple)):
        endpoints = [
            str(a).strip() for a in control_plane_rpc_address if str(a).strip()
        ]
      else:
        endpoints = [
            a.strip()
            for a in str(control_plane_rpc_address).split(",")
            if a.strip()
        ]

    normalized_shards = list(shards)
    new_mesh_shape = list(mesh_shape) if mesh_shape is not None else None
    new_mesh_axes = list(mesh_axes) if mesh_axes is not None else None
    new_layout = list(layout) if layout is not None else None
    new_global_shape = list(global_shape) if global_shape is not None else None
    new_itemsize = itemsize
    new_pools = normalized_pools if has_reshard_metadata else None
    new_layout_fingerprint = (
        layout_fingerprint if has_reshard_metadata else None
    )
    new_page_tokens = page_tokens if has_reshard_metadata else None
    new_transfer_parallelism = (
        transfer_parallelism if has_reshard_metadata else None
    )
    new_transfer_rank = transfer_rank if has_reshard_metadata else None
    new_variables = list(variables) if variables is not None else None

    with self._lock:
      old_shards = self._registered_shards.get(unit)
      old_endpoints = self._registered_control_plane_endpoints.get(unit, [])
      is_topology_identical = (
          old_shards is not None
          and len(old_shards) == len(normalized_shards)
          and self._registered_mesh_shapes.get(unit) == new_mesh_shape
          and self._registered_mesh_axes.get(unit) == new_mesh_axes
          and self._registered_layouts.get(unit) == new_layout
          and self._registered_global_shapes.get(unit) == new_global_shape
          and self._registered_itemsizes.get(unit) == new_itemsize
          and self._registered_pool_manifests.get(unit) == new_pools
          and self._registered_layout_fingerprints.get(unit)
          == new_layout_fingerprint
          and self._registered_page_tokens.get(unit) == new_page_tokens
          and self._registered_transfer_parallelism.get(unit)
          == new_transfer_parallelism
          and self._registered_transfer_ranks.get(unit) == new_transfer_rank
          and self._registered_variables.get(unit) == new_variables
      )
      is_idempotent = (
          is_topology_identical
          and old_shards == normalized_shards
          and old_endpoints == endpoints
      )

      if is_idempotent:
        ent = self.get_or_create_entity(unit)
        for addr in endpoints:
          ent.register_host_endpoint(
              addr, job_replica_id=getattr(unit, "job_replica_id", "")
          )
        return

      if unit in self._registered_shards:
        tasks_to_clear = []
        for req_id, units in self._task_units.items():
          if unit in units:
            tasks_to_clear.append(req_id)
        for req_id in tasks_to_clear:
          self._active_tasks.pop(req_id, None)
          self._task_units.pop(req_id, None)
          self._active_transfers.pop(req_id, None)

      ent = self.get_or_create_entity(unit)
      ent.update_registration(
          shards=normalized_shards,
          control_endpoints=endpoints,
          mesh_shape=new_mesh_shape,
          mesh_axes=new_mesh_axes,
          layout=new_layout,
          global_shape=new_global_shape,
          itemsize=new_itemsize,
          pool_manifest=new_pools,
          layout_fingerprint=new_layout_fingerprint,
          page_tokens=new_page_tokens,
          transfer_parallelism=new_transfer_parallelism,
          transfer_rank=new_transfer_rank,
          variables=new_variables,
          unit=unit,
      )

      self._registered_shards[unit] = normalized_shards
      self._registered_control_plane_endpoints[unit] = endpoints
      for registry in (
          self._registered_mesh_shapes,
          self._registered_mesh_axes,
          self._registered_layouts,
          self._registered_global_shapes,
          self._registered_itemsizes,
          self._registered_pool_manifests,
          self._registered_layout_fingerprints,
          self._registered_page_tokens,
          self._registered_transfer_parallelism,
          self._registered_transfer_ranks,
          self._registered_variables,
      ):
        registry.pop(unit, None)
      if new_mesh_shape is not None:
        self._registered_mesh_shapes[unit] = new_mesh_shape
      if new_mesh_axes is not None:
        self._registered_mesh_axes[unit] = new_mesh_axes
      if new_layout is not None:
        self._registered_layouts[unit] = new_layout
      if new_global_shape is not None:
        self._registered_global_shapes[unit] = new_global_shape
      if new_itemsize is not None:
        self._registered_itemsizes[unit] = new_itemsize
      if has_reshard_metadata:
        self._registered_pool_manifests[unit] = new_pools
        self._registered_layout_fingerprints[unit] = new_layout_fingerprint
        self._registered_page_tokens[unit] = new_page_tokens
        self._registered_transfer_parallelism[unit] = new_transfer_parallelism
        self._registered_transfer_ranks[unit] = new_transfer_rank
      if new_variables is not None:
        self._registered_variables[unit] = new_variables

      keys_to_clear = []
      for k, cached_sched in self._plan_cache.items():
        if not (isinstance(k, tuple) and len(k) >= 2):
          continue
        in_src = unit in k[0]
        in_dst = unit in k[1]
        if not (in_src or in_dst):
          continue
        if not is_topology_identical or (
            in_dst and old_shards != normalized_shards
        ):
          keys_to_clear.append(k)
        else:
          if endpoints:
            cached_sched.rpc_addresses[unit] = ",".join(endpoints)
            cached_sched.cached_serialized_payloads.clear()
            cached_sched.sender_push_schedule_protos.clear()
          if unit in cached_sched.data_addresses:
            cached_sched.data_addresses[unit] = list(normalized_shards)
            cached_sched.cached_serialized_payloads.clear()
            cached_sched.sender_push_schedule_protos.clear()
      for k in keys_to_clear:
        self._plan_cache.pop(k, None)

  def clear_plan_cache(self) -> None:
    """Clears all cached transfer schedules."""
    with self._lock:
      self._plan_cache.clear()

  def get_plan_cache_size(self) -> int:
    """Returns the number of cached transfer schedules."""
    with self._lock:
      return len(self._plan_cache)

  def _make_plan_cache_key(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      broadcast_host_ratio: Optional[float] = None,
      broadcast_pipeline_stages: Optional[int] = None,
  ) -> tuple[Any, ...]:
    """Builds a hashable plan cache key from transfer arguments."""
    ratio = (
        broadcast_host_ratio
        if broadcast_host_ratio is not None
        else self.broadcast_host_ratio
    )
    stages = (
        broadcast_pipeline_stages
        if broadcast_pipeline_stages is not None
        else self.broadcast_pipeline_stages
    )
    return ReshardPlanner.make_plan_cache_key(
        src_units=src_units,
        dst_units=dst_units,
        group_size=group_size,
        skip_tiling=skip_tiling,
        dst_controller_address=dst_controller_address,
        src_controller_address=src_controller_address,
        broadcast_host_ratio=ratio,
        broadcast_pipeline_stages=stages,
    )

  def _prepare_cached_schedule(self, schedule: _CachedTransferSchedule) -> None:
    """Precomputes per-plan artifacts reused by every transfer of `schedule`.

    Coalesces the planner's per-(eq_class, layer_group) StageBroadcastGroups
    into pipeline stages (memoized on the groups themselves so
    BroadcastEngine.execute_slice_broadcast_pipeline reuses the result), builds
    the single-stage 1-hop trainer push schedules, and serializes the sender
    push schedule protos. Doing this once here keeps all of it off the
    per-transfer critical path.

    Args:
      schedule: Freshly computed transfer schedule to prepare in place.
    """
    if not schedule.computed_schedules and schedule.broadcast_groups:
      groups_list = list(schedule.broadcast_groups.values())
      if groups_list and isinstance(
          groups_list[0], controller_types.StageBroadcastGroup
      ):
        effective_stages, coalesced_groups = (
            BroadcastEngine.coalesce_pipeline_groups_cached(
                groups_list,
                self.broadcast_pipeline_stages,
                schedule.n_seed,
            )
        )
        # Only a single-stage 1-hop plan reuses precomputed trainer push
        # schedules (`is_single_stage_1hop` in BroadcastEngine._dispatch_hop);
        # multi-hop pipelines build per-hop schedules at dispatch time.
        if effective_stages == 1:
          for stage_group in coalesced_groups:
            sg_scheds = BroadcastEngine.build_stage_trainer_push_schedules(
                stage_group,
                stage_group.dst_units,
                registered_shards=self._registered_shards,
            )
            for s_u, sched_map in sg_scheds.items():
              target_map = schedule.computed_schedules.setdefault(s_u, {})
              target_map.update(sched_map)
    raw_schedules = schedule.direct_schedules or schedule.computed_schedules
    if raw_schedules:
      for src_u, push_schedules in raw_schedules.items():
        if src_u not in schedule.sender_push_schedule_protos and push_schedules:
          src_ent = self.get_or_create_entity(src_u)
          schedule.sender_push_schedule_protos[src_u] = (
              src_ent.build_sender_push_schedule_protos(push_schedules)
          )

  async def warmup_transfer_plan(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
  ) -> _CachedTransferSchedule:
    """Precomputes and caches transfer schedule outside of the critical path."""
    if group_size <= 0:
      raise ValueError("group_size must be positive")
    cache_key = self._make_plan_cache_key(
        src_units=src_units,
        dst_units=dst_units,
        group_size=group_size,
        skip_tiling=skip_tiling,
        dst_controller_address=dst_controller_address,
        src_controller_address=src_controller_address,
    )
    with self._lock:
      if cache_key in self._plan_cache:
        return self._plan_cache[cache_key]

    schedule = await self._compute_transfer_schedule(
        src_units=src_units,
        dst_units=dst_units,
        dst_controller_address=dst_controller_address,
        group_size=group_size,
        skip_tiling=skip_tiling,
        req_id="warmup",
        uuid=str(random.randint(1, 2**63 - 1)),
    )
    self._prepare_cached_schedule(schedule)
    with self._lock:
      self._plan_cache[cache_key] = schedule
    return schedule

  async def _compute_transfer_schedule(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      dst_controller_address: Optional[str] = None,
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      shard_push_schedules: Optional[
          dict[RaidenId, dict[int, list[Any]]]
      ] = None,
      req_id: str = "warmup",
      uuid: Any = "",
  ) -> _CachedTransferSchedule:
    """Computes transfer schedule math via ReshardPlanner."""
    t_start = time.perf_counter()
    if group_size <= 0:
      raise ValueError("group_size must be positive")
    if dst_controller_address:
      logging.info(
          "Querying remote destination controller %s for metadata",
          dst_controller_address,
      )
      dst_metadata = await self._query_remote_metadata(dst_controller_address)
    else:
      logging.info("Using local registration for destination metadata")
      dst_metadata = self._get_local_metadata(dst_units)

    worker_endpoints = self.get_entity_rpc_addresses()

    schedule = self._planner.compute_transfer_schedule_from_metadata(
        src_units=src_units,
        dst_units=dst_units,
        dst_metadata=dst_metadata,
        entities=self._entities,
        registered_variables=self._registered_variables,
        registered_global_shapes=self._registered_global_shapes,
        registered_mesh_shapes=self._registered_mesh_shapes,
        registered_layouts=self._registered_layouts,
        registered_itemsizes=self._registered_itemsizes,
        registered_shards=self._registered_shards,
        computed_phys_meshes=self._computed_phys_meshes,
        worker_endpoints=worker_endpoints,
        broadcast_host_ratio=self.broadcast_host_ratio,
        lock=self._lock,
        group_size=group_size,
        skip_tiling=skip_tiling,
        shard_push_schedules=shard_push_schedules,
        req_id=req_id,
        uuid=uuid,
        pipeline_target_stages=self.broadcast_pipeline_stages,
    )
    common.record_histogram(
        "weight_sync_schedule_generation_time_ms",
        (time.perf_counter() - t_start) * 1000.0,
    )
    return schedule

  def _metadata_proto_locked(self, unit: RaidenId) -> Any:
    """Builds an owned registration proto while `_lock` is held."""
    reg_req = raiden_service_pb2.RegisterWorkUnitRequest(
        unit=raiden_service_pb2.RaidenIdProto(
            job_name=unit.job_name,
            job_replica_id=unit.job_replica_id,
            data_name=unit.data_name,
            data_replica_idx=unit.data_replica_idx,
        ),
        shards=self._registered_shards[unit],
        control_plane_rpc_address=self.worker_endpoints.get(unit, ""),
        itemsize=self._registered_itemsizes.get(unit, 0),
        layout_fingerprint=self._registered_layout_fingerprints.get(unit, ""),
        page_tokens=self._registered_page_tokens.get(unit, 0),
        transfer_parallelism=self._registered_transfer_parallelism.get(unit, 0),
        transfer_rank=self._registered_transfer_ranks.get(unit, 0),
    )
    reg_req.mesh_shape.extend(self._registered_mesh_shapes.get(unit, ()))
    reg_req.mesh_axes.extend(self._registered_mesh_axes.get(unit, ()))
    reg_req.layout.extend(self._registered_layouts.get(unit, ()))
    reg_req.global_shape.extend(self._registered_global_shapes.get(unit, ()))
    for pool in self._registered_pool_manifests.get(unit, ()):
      reg_req.pools.add().CopyFrom(pool)
    if unit in self._registered_variables:
      for var in self._registered_variables[unit]:
        reg_req.variables.add().CopyFrom(
            _coerce_variable_proto(var, raiden_service_pb2)
        )
    return reg_req

  def get_all_metadata(self) -> list[Any]:
    """Returns replacement-safe metadata for all registered work units."""
    with self._lock:
      return [
          self._metadata_proto_locked(unit) for unit in self._registered_shards
      ]

  def get_registered_units(self) -> set[RaidenId]:
    """Returns the set of currently registered work unit IDs."""
    with self._lock:
      return set(self._registered_shards.keys())

  def _resolve_shards(self, unit: RaidenId) -> list[str]:
    with self._lock:
      ent = self._entities.get(unit)
      shards = (
          ent.shards
          if (ent and ent.shards)
          else self._registered_shards.get(unit)
      )
      if not shards:
        raise ValueError(f"Work unit is not registered: {unit}")
      return list(shards)

  async def _query_remote_metadata(self, addr: str) -> list[Any]:
    """Queries registered work unit metadata from a remote controller endpoint."""
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA
    )
    meta_ent = self.get_or_create_entity(RaidenId())
    resp_bytes = await meta_ent._send_rpc(  # pylint: disable=protected-access
        addr, req.SerializeToString()
    )
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(f"Failed to query remote metadata: {resp.message}")
    return list(resp.get_metadata_response.metadata)

  def _get_local_metadata(self, units: list[RaidenId]) -> list[Any]:
    with self._lock:
      missing = [unit for unit in units if unit not in self._registered_shards]
      if missing:
        raise ValueError(f"Work units are not registered: {missing}")
      return [self._metadata_proto_locked(unit) for unit in units]

  @classmethod
  def _metadata_by_unit(
      cls, metadata: Sequence[Any], units: Sequence[RaidenId]
  ) -> dict[RaidenId, Any]:
    """Selects exact requested metadata and rejects duplicate identities."""
    requested = set(units)
    result = {}
    for item in metadata:
      unit = _raiden_id_from_proto(item.unit)
      if unit not in requested:
        continue
      if unit in result:
        raise ValueError(f"Duplicate registration metadata for {unit}")
      result[unit] = item
    missing = [unit for unit in units if unit not in result]
    if missing:
      raise ValueError(f"Missing registration metadata for {missing}")
    return result

  def get_plan(self, req_id: str) -> Optional[TransferPlan]:
    """Returns the generated TransferPlan for a given transfer request ID."""
    with self._lock:
      return self._active_transfers.get(req_id)

  async def _execute_slice_broadcast(
      self,
      keys_and_targets: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]],
      final_plan: TransferPlan,
      n_seed: int,
      req_id: str,
      dst_mem_type: int,
      dst_controller_address: Optional[str],
      src_controller_address: Optional[str] = None,
  ) -> None:
    """Executes a pipelined tree broadcast for a group of variables."""
    await self._broadcast_engine.execute_slice_broadcast(
        keys_and_targets=keys_and_targets,
        final_plan=final_plan,
        n_seed=n_seed,
        req_id=req_id,
        dst_mem_type=dst_mem_type,
        registered_shards=self._registered_shards,
        dst_controller_address=dst_controller_address,
        src_controller_address=src_controller_address,
        pipeline_target_stages=self.broadcast_pipeline_stages,
    )

  async def _execute_slice_broadcast_pipeline(
      self,
      groups_list: list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]],
      final_plan: TransferPlan,
      n_seed: int,
      req_id: str,
      dst_mem_type: int,
      dst_controller_address: Optional[str],
      src_controller_address: Optional[str] = None,
  ) -> None:
    """Executes a pipelined multi-group tree broadcast across groups.

    Args:
      groups_list: List of transfer groups, where each group is a list of (key,
        targets) slice tuples.
      final_plan: TransferPlan containing metadata and worker addresses.
      n_seed: Number of seed samplers for bandwidth-matched tree broadcast.
      req_id: Identifier for the transfer request.
      dst_mem_type: Destination memory type enum or integer.
      dst_controller_address: Optional address of remote destination controller.
      src_controller_address: Optional address of source controller.
    """
    await self._broadcast_engine.execute_slice_broadcast_pipeline(
        groups_list=groups_list,
        final_plan=final_plan,
        n_seed=n_seed,
        req_id=req_id,
        dst_mem_type=dst_mem_type,
        registered_shards=self._registered_shards,
        dst_controller_address=dst_controller_address,
        src_controller_address=src_controller_address,
        pipeline_target_stages=self.broadcast_pipeline_stages,
    )

  def _start_pool_reshard_transfer(self, *args, **kwargs):
    """REMOVED: the Python pool-reshard implementation is retired."""
    raise RuntimeError(
        "RaidenController._start_pool_reshard_transfer was removed: the "
        "pool-reshard path is served by the C++ reshard store"
    )

  def start_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      src_block_ids: Optional[list[int]] = None,
      dst_device_block_ids: Optional[list[int]] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      use_block_chunks: bool = False,
      src_controller_address: Optional[str] = None,
      dst_controller_address: Optional[str] = None,
      uuid: Optional[int] = None,
      is_sender: bool = True,
      expected_block_count: int = 0,
      shard_push_schedules: Optional[dict[Any, Any]] = None,
      num_tokens: Optional[int] = None,
      transfer_pool_tags: Optional[Sequence[str]] = None,
      parallelism: Optional[int] = None,
      skip_d2h: bool = False,
      dst_block_counts: Optional[Sequence[int]] = None,
      skip_tiling: Optional[dict[int, bool]] = None,
      group_size: int = 1,
      use_cached_plan: bool = True,
      expected_layer_chunk_counts: Optional[dict[int, int]] = None,
      host_tiling_mode: int = raiden_service_pb2.HOST_TILING_MODE_UNSPECIFIED,
  ) -> RaidenFuture:
    """Generates a transfer plan for the requested entities and dispatches it."""
    if group_size <= 0:
      raise ValueError("group_size must be positive")

    if (
        num_tokens is not None
        or dst_device_block_ids is not None
        or transfer_pool_tags is not None
    ):
      if shard_push_schedules:
        raise ValueError(
            "Pool reshard requests never carry prepared schedules; the "
            "planning controller builds and arms them itself"
        )
      return self._start_pool_reshard_transfer(
          src_units=src_units,
          dst_units=dst_units,
          req_id=req_id,
          src_block_ids=src_block_ids,
          dst_device_block_ids=dst_device_block_ids,
          dst_mem_type=dst_mem_type,
          use_block_chunks=use_block_chunks,
          dst_controller_address=dst_controller_address,
          uuid=uuid,
          is_sender=is_sender,
          num_tokens=num_tokens,
          transfer_pool_tags=transfer_pool_tags,
          parallelism=parallelism,
          skip_d2h=skip_d2h,
          dst_block_counts=dst_block_counts,
      )

    if not src_units:
      raise ValueError("src_units must not be empty.")
    if not dst_units:
      raise ValueError("dst_units must not be empty.")
    if src_block_ids:
      raise NotImplementedError("src_block_ids are not supported yet.")
    if dst_device_block_ids:
      raise NotImplementedError("dst_device_block_ids are not supported yet.")

    with self._lock:
      if req_id and req_id in self._active_tasks:
        logging.info(
            "start_transfer req_id %s already exists, returning existing"
            " future.",
            req_id,
        )
        return self._active_tasks[req_id]

      self._prune_completed_transfers_locked()

      selected_src = min(
          src_units,
          key=lambda s: self._src_replica_counts.get(s.job_replica_id, 0),
      )

      if uuid is None:
        uuid = random.randint(1, 2**63 - 1)

      session_id = self._next_session_id
      self._next_session_id += 1
      if not req_id:
        req_id = f"req_{session_id}"

      recorded_srcs = [selected_src] if not use_block_chunks else src_units
      for s in recorded_srcs:
        self._src_replica_counts[s.job_replica_id] = (
            self._src_replica_counts.get(s.job_replica_id, 0) + 1
        )
      for t in dst_units:
        self._dst_replica_counts[t.job_replica_id] = (
            self._dst_replica_counts.get(t.job_replica_id, 0) + 1
        )

    if not use_block_chunks:
      num_src = len(self._resolve_shards(selected_src))
      dst_shard_counts = [
          (dst_unit, len(self._resolve_shards(dst_unit)))
          for dst_unit in dst_units
      ]
      default_plan_dict = self._planner.build_default_1d_plan(
          selected_src, num_src, dst_shard_counts
      )

      rpc_addresses = self.get_entity_rpc_addresses()
      data_addresses = {unit: self._resolve_shards(unit) for unit in dst_units}

      plan = TransferPlan(
          src_units=[selected_src],
          dst_units=dst_units,
          plan=default_plan_dict,
          shard_push_schedules={},
          worker_rpc_addresses=rpc_addresses,
          worker_data_addresses=data_addresses,
          uuid=uuid,
          dst_mem_type=dst_mem_type,
          use_block_chunks=False,
          skip_d2h=skip_d2h,
      )
      with self._lock:
        self._active_transfers[req_id] = plan
    else:
      plan = TransferPlan(
          src_units=src_units,
          dst_units=dst_units,
          plan=None,
          shard_push_schedules=shard_push_schedules or {},
          worker_rpc_addresses=self.get_entity_rpc_addresses(),
          worker_data_addresses=dict(self._registered_shards),
          uuid=uuid,
          dst_mem_type=dst_mem_type,
          use_block_chunks=True,
          is_sender=is_sender,
          expected_block_count=expected_block_count,
          req_id=req_id,
          skip_d2h=skip_d2h,
          parallelism=parallelism or 1,
      )
      with self._lock:
        self._active_transfers[req_id] = plan

    t_transfer_start = time.monotonic()

    async def _execute_transfer_inner() -> None:
      nonlocal skip_d2h, expected_block_count, t_transfer_start
      if use_block_chunks:
        if not is_sender:
          local_dst_units = [
              u for u in dst_units if u in self._registered_shards
          ]
          logging.info(
              "RaidenController acting as DESTINATION COORDINATOR"
              " (is_sender=False) for req_id %s (uuid=%s): destination=%s"
              " (source=%s)",
              req_id,
              uuid,
              _format_units(local_dst_units),
              _format_units(src_units),
          )
          if not local_dst_units:
            logging.warning("No local destination units found to prepare!")
            return

          rpc_addresses = self.get_entity_rpc_addresses()

          receiver_plan = TransferPlan(
              src_units=src_units,
              dst_units=dst_units,
              plan=None,
              shard_push_schedules=shard_push_schedules or {},
              worker_rpc_addresses=rpc_addresses,
              worker_data_addresses={
                  u: self._registered_shards[u] for u in local_dst_units
              },
              uuid=uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=True,
              is_sender=False,
              expected_block_count=expected_block_count,
              expected_layer_chunk_counts=dict(
                  expected_layer_chunk_counts or {}
              ),
              req_id=req_id,
              skip_d2h=skip_d2h,
              skip_tiling=skip_tiling or {},
              parallelism=parallelism or 1,
              is_weight_sync=bool(
                  skip_tiling
                  or any(
                      u in self._registered_variables
                      for u in (*src_units, *dst_units)
                  )
              ),
              host_tiling_mode=host_tiling_mode,
          )

          if expected_block_count > 0:
            logging.info(
                "Triggering preparation RPCs on local destination workers: %s,"
                " expected blocks: %d",
                _format_units(local_dst_units),
                expected_block_count,
            )
            await asyncio.gather(*[
                self._dispatch_entity_transfer(unit, receiver_plan)
                for unit in local_dst_units
            ])
          else:
            logging.info(
                "Skipping preparation RPCs on local destination workers"
                " because expected_block_count is 0"
            )
          logging.info(
              "Symmetric preparation complete on all local destination workers."
          )

        else:
          t_transfer_start = time.monotonic()
          logging.info(
              "RaidenController acting as SENDER COORDINATOR (is_sender=True)"
              " for req_id %s (uuid=%s): %s -> %s",
              req_id,
              uuid,
              _format_units(src_units),
              _format_units(dst_units),
          )

          cache_key = self._make_plan_cache_key(
              src_units=src_units,
              dst_units=dst_units,
              group_size=group_size,
              skip_tiling=skip_tiling,
              dst_controller_address=dst_controller_address,
              src_controller_address=src_controller_address,
          )

          cached_schedule = None
          if (
              self.enable_plan_cache
              and use_cached_plan
              and not shard_push_schedules
          ):
            with self._lock:
              cached_schedule = self._plan_cache.get(cache_key)

          logging.vlog(
              1,
              "RAIDEN_DIAG plan_compute_start req_id=%s uuid=%s"
              " use_cached_plan=%s cache_hit=%s",
              req_id,
              uuid,
              use_cached_plan,
              cached_schedule is not None,
          )

          newly_computed = False
          if cached_schedule is None:
            newly_computed = True
            cached_schedule = await self._compute_transfer_schedule(
                src_units=src_units,
                dst_units=dst_units,
                dst_controller_address=dst_controller_address,
                group_size=group_size,
                skip_tiling=skip_tiling,
                shard_push_schedules=shard_push_schedules,
                req_id=req_id,
                uuid=uuid,
            )
            self._prepare_cached_schedule(cached_schedule)
            if self.enable_plan_cache and not shard_push_schedules:
              with self._lock:
                self._plan_cache[cache_key] = cached_schedule
          else:
            logging.info(
                "Transfer %s (uuid=%s): reusing cached schedule for %s -> %s"
                " (%d expected blocks)",
                req_id,
                uuid,
                _format_units(src_units),
                _format_units(dst_units),
                cached_schedule.expected_block_count,
            )

          computed_schedules = cached_schedule.computed_schedules
          direct_schedules = cached_schedule.direct_schedules
          broadcast_groups = cached_schedule.broadcast_groups
          logging.vlog(
              1,
              "RAIDEN_DIAG plan_ready req_id=%s uuid=%s compute_ms=%.2f"
              " is_cached=%s direct_schedules=%d broadcast_groups=%d",
              req_id,
              uuid,
              (time.monotonic() - t_transfer_start) * 1000.0,
              cached_schedule is not None and not newly_computed,
              len(direct_schedules),
              len(broadcast_groups),
          )
          local_skip_tiling = cached_schedule.local_skip_tiling
          if expected_block_count == 0:
            expected_block_count = cached_schedule.expected_block_count
          dst_unit_layer_counts = cached_schedule.dst_unit_layer_counts
          direct_dsts = cached_schedule.direct_dsts
          rpc_addresses = dict(cached_schedule.rpc_addresses)
          rpc_addresses.update(self.get_entity_rpc_addresses())
          data_addresses = cached_schedule.data_addresses
          dst_unit_counts = cached_schedule.dst_unit_counts
          endpoint_to_shards: dict[Any, set[int]] = {}
          for u in (*src_units, *dst_units):
            u_rpc = rpc_addresses.get(u, "")
            u_shards = data_addresses.get(u) or self._registered_shards.get(
                u, ()
            )
            if u_rpc and u_shards:
              endpoint_to_shards.update(
                  controller_types.compute_endpoint_to_shards(
                      u, [u_rpc], u_shards
                  )
              )

          if cached_schedule.is_weight_sync:
            for src_u in src_units:
              src_ent = self.get_or_create_entity(src_u)
              all_shards = self._registered_shards.get(src_u, []) or list(
                  src_ent.shards
              )
              if not all_shards:
                continue

              has_host_shards = bool(
                  src_ent.hosts
                  and any(h.control_address and h.shards for h in src_ent.hosts)
                  and (
                      not hasattr(src_ent, "_hosts_by_replica")
                      or (
                          bool(src_ent._hosts_by_replica)
                          and not (
                              len(src_ent.hosts) == 1
                              and len(
                                  str(src_ent.hosts[0].control_address).split(
                                      ","
                                  )
                              )
                              > 1
                          )
                      )
                  )
              )
              if has_host_shards:
                for host_desc in src_ent.hosts:
                  if host_desc.control_address:
                    host_shards_set = set(host_desc.shards)
                    owned_indices = {
                        idx
                        for idx, s in enumerate(all_shards)
                        if s in host_shards_set
                    }
                    if owned_indices:
                      for c_addr in str(host_desc.control_address).split(","):
                        c_addr = c_addr.strip()
                        if c_addr:
                          endpoint_to_shards[(src_u, c_addr)] = owned_indices
                          endpoint_to_shards[c_addr] = owned_indices
              else:
                control_addrs: list[str] = []
                for host_desc in src_ent.hosts:
                  if host_desc.control_address:
                    for a in str(host_desc.control_address).split(","):
                      c_addr = a.strip()
                      if c_addr and c_addr not in control_addrs:
                        control_addrs.append(c_addr)

                if not control_addrs:
                  for a in self._registered_control_plane_endpoints.get(
                      src_u, []
                  ):
                    c_addr = str(a).strip()
                    if c_addr and c_addr not in control_addrs:
                      control_addrs.append(c_addr)

                if not control_addrs:
                  continue

                if len(control_addrs) == 1:
                  c_addr = control_addrs[0]
                  owned = set(range(len(all_shards)))
                  if owned:
                    endpoint_to_shards[(src_u, c_addr)] = owned
                    endpoint_to_shards[c_addr] = owned
                elif len(control_addrs) > 1:
                  distinct_data_addrs: list[str] = []
                  for s in all_shards:
                    if s not in distinct_data_addrs:
                      distinct_data_addrs.append(s)

                  if len(distinct_data_addrs) == len(control_addrs):
                    for i, c_addr in enumerate(control_addrs):
                      target_data_addr = distinct_data_addrs[i]
                      owned = {
                          idx
                          for idx, s in enumerate(all_shards)
                          if s == target_data_addr
                      }
                      if owned:
                        endpoint_to_shards[(src_u, c_addr)] = owned
                        endpoint_to_shards[c_addr] = owned
                  else:
                    control_ips: list[str] = []
                    c_by_ip: dict[str, list[str]] = {}
                    for c in control_addrs:
                      ip = c.rsplit(":", 1)[0]
                      if ip not in control_ips:
                        control_ips.append(ip)
                      c_by_ip.setdefault(ip, []).append(c)

                    d_by_ip: dict[str, list[str]] = {}
                    for d in distinct_data_addrs:
                      ip = d.rsplit(":", 1)[0]
                      d_by_ip.setdefault(ip, []).append(d)

                    can_match_by_ip = set(control_ips) == set(
                        d_by_ip.keys()
                    ) and all(
                        len(c_by_ip[ip]) == len(d_by_ip.get(ip, []))
                        or len(c_by_ip[ip]) == 1
                        for ip in control_ips
                    )

                    if can_match_by_ip:
                      for ip in control_ips:
                        c_addrs_on_ip = c_by_ip[ip]
                        data_addrs_on_ip = d_by_ip.get(ip, [])
                        if len(c_addrs_on_ip) == len(data_addrs_on_ip):
                          for j, c_addr in enumerate(c_addrs_on_ip):
                            target_data_addr = data_addrs_on_ip[j]
                            owned = {
                                idx
                                for idx, s in enumerate(all_shards)
                                if s == target_data_addr
                            }
                            if owned:
                              endpoint_to_shards[(src_u, c_addr)] = owned
                              endpoint_to_shards[c_addr] = owned
                        elif len(c_addrs_on_ip) == 1:
                          owned = {
                              idx
                              for idx, s in enumerate(all_shards)
                              if s.rsplit(":", 1)[0] == ip
                          }
                          if owned:
                            endpoint_to_shards[(src_u, c_addrs_on_ip[0])] = (
                                owned
                            )
                            endpoint_to_shards[c_addrs_on_ip[0]] = owned

          all_push_schedules = dict(computed_schedules)

          final_plan = TransferPlan(
              src_units=list(all_push_schedules.keys())
              if all_push_schedules
              else src_units,
              dst_units=dst_units,
              plan=None,
              shard_push_schedules=all_push_schedules,
              worker_rpc_addresses=rpc_addresses,
              worker_data_addresses=data_addresses,
              uuid=uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=use_block_chunks,
              is_sender=True,
              expected_block_count=expected_block_count,
              dst_expected_layer_chunk_counts=dst_unit_layer_counts,
              dst_expected_block_counts=dst_unit_counts,
              dst_endpoint_counts=cached_schedule.dst_endpoint_counts,
              dst_endpoint_layer_counts=cached_schedule.dst_endpoint_layer_counts,
              req_id=req_id,
              skip_d2h=skip_d2h,
              skip_tiling=local_skip_tiling,
              parallelism=parallelism or 1,
              is_weight_sync=cached_schedule.is_weight_sync,
              sender_push_schedule_protos=(
                  cached_schedule.sender_push_schedule_protos
              ),
              cached_serialized_payloads=(
                  cached_schedule.cached_serialized_payloads
              ),
              endpoint_to_shards=endpoint_to_shards,
              variable_plans=cached_schedule.variable_plans,
              variable_to_plan_id=cached_schedule.variable_to_plan_id,
              has_explicit_shard_push_schedules=bool(shard_push_schedules),
          )
          with self._lock:
            self._active_transfers[req_id] = final_plan

          # Arm destination controller for tree broadcast top-level req_id
          if broadcast_groups:
            if dst_controller_address:
              logging.vlog(
                  1,
                  "Registering top-level req_id %s on destination controller"
                  " %s",
                  req_id,
                  dst_controller_address,
              )
              dst_facade = RaidenControllerClientFacade(
                  dst_controller_address,
                  name_resolver=self.name_resolver,
              )
              loop = asyncio.get_running_loop()
              rpc_executor = self.executor
              success = await loop.run_in_executor(
                  rpc_executor,
                  dst_facade.register_transfer_schedule,
                  src_units,
                  dst_units,
                  req_id,
                  True,
                  False,
                  0,
                  uuid,
                  dst_controller_address,
                  src_controller_address,
                  None,
                  dst_mem_type,
              )
              if not success:
                logging.warning(
                    "Failed to register top-level req_id %s on destination"
                    " controller",
                    req_id,
                )

          push_tasks = []

          senders_start = time.monotonic()
          logging.vlog(
              1,
              "RAIDEN_DIAG senders_dispatch_start req_id=%s uuid=%s"
              " broadcast_groups=%d elapsed_since_transfer_start_ms=%.2f",
              req_id,
              uuid,
              len(broadcast_groups),
              (senders_start - t_transfer_start) * 1000.0,
          )

          if broadcast_groups:
            groups_list = list(broadcast_groups.values())
            push_tasks.append(
                self._execute_slice_broadcast_pipeline(
                    groups_list=groups_list,
                    final_plan=final_plan,
                    n_seed=cached_schedule.n_seed,
                    req_id=req_id,
                    dst_mem_type=dst_mem_type,
                    dst_controller_address=dst_controller_address,
                    src_controller_address=src_controller_address,
                )
            )

          if push_tasks:
            await asyncio.gather(*push_tasks)

          logging.vlog(
              1,
              "RAIDEN_DIAG senders_dispatch_complete req_id=%s uuid=%s"
              " push_tasks_ms=%.2f total_transfer_ms=%.2f",
              req_id,
              uuid,
              (time.monotonic() - senders_start) * 1000.0,
              (time.monotonic() - t_transfer_start) * 1000.0,
          )

      else:
        with self._lock:
          old_plan = self._active_transfers[req_id]

        for unit in dst_units:
          await self._dispatch_entity_transfer(unit, old_plan)

        await asyncio.gather(*[
            self._dispatch_entity_transfer(unit, old_plan)
            for unit in old_plan.src_units
        ])

    async def _execute_transfer() -> None:
      try:
        await _execute_transfer_inner()
        if is_sender:
          common.record_histogram(
              "weight_sync_e2e_broadcast_duration_ms",
              (time.monotonic() - t_transfer_start) * 1000.0,
          )
      except Exception:
        common.record_counter(
            "weight_sync_transfer_failures_total",
            1,
            {"error_code": "INTERNAL", "direction": "push"},
        )
        raise
      finally:
        common.flush_cloud_logging(
            phase="controller_transfer",
            uuid=uuid,
            req_id=req_id or "",
        )

    def _on_transfer_done():
      with self._lock:
        self._prune_completed_transfers_locked()

    transfer_task = _execute_transfer()
    future = RaidenFuture(
        session_id=session_id,
        transfer_task=transfer_task,
        on_complete=_on_transfer_done,
    )
    with self._lock:
      self._active_tasks[req_id] = future
      self._task_units[req_id] = list(src_units) + list(dst_units)
    return future

  def get_transfer_status(self, req_id: str) -> int:
    """Returns the status of the transfer for req_id."""
    with self._lock:
      future = self._active_tasks.get(req_id)
    if not future:
      return controller_service_pb2.GetTransferStatusResponse.STATUS_NOT_STARTED

    if future.done():
      if future.exception():
        return controller_service_pb2.GetTransferStatusResponse.STATUS_FAILED
      return controller_service_pb2.GetTransferStatusResponse.STATUS_COMPLETED
    return controller_service_pb2.GetTransferStatusResponse.STATUS_IN_PROGRESS

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

"""Multi-host JobEntity (HostGroup) abstraction owning ControlPipeClient and host dispatch."""

import asyncio
from collections import abc
import concurrent.futures
import dataclasses
import os
from typing import Any, Callable, Optional, Sequence

from absl import logging

from tpu_sync.api.common import RaidenId
from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types

NameResolver = controller_types.NameResolver
TransferPlan = controller_types.TransferPlan
_coerce_variable_proto = controller_types.coerce_variable_proto
_extract_host_ip = controller_types.extract_host_ip


def _default_max_workers(max_workers: Optional[int] = None) -> int:
  """Computes default thread pool concurrency for control-plane RPC dispatch."""
  if max_workers is None:
    env_concurrency = os.environ.get("RAIDEN_RPC_CONCURRENCY")
    if env_concurrency:
      try:
        max_workers = int(env_concurrency)
      except ValueError:
        max_workers = None
  if max_workers is None:
    max_workers = max(128, (os.cpu_count() or 1) * 16)
  return max_workers


@dataclasses.dataclass
class HostDescriptor:
  """Represents a single physical host attached to a multi-host JobEntity."""

  control_address: str
  shards: list[str] = dataclasses.field(default_factory=list)
  job_replica_id: str = ""
  unit: Optional[RaidenId] = None


class JobEntity:
  """Abstraction for a potentially multi-host trainer or sampler job entity.

  A JobEntity represents a single logical job replica (e.g., 1 trainer job with
  10 hosts, or 1 sampler replica with 4 hosts) managed by RaidenController.
  Its entity key is `RaidenId(job_name, "", data_name, data_replica_idx)`, while
  individual hosts attached to this entity are indexed by `job_replica_id`
  (e.g., `"0".."9"` for 10 trainer hosts, `"0".."3"` for 4 sampler hosts).

  It owns:
    1. The WorkerRpcClient (which in turn owns the ControlPipeClient) used to
       communicate with its attached hosts.
    2. All attached host control-plane RPC endpoints and data-plane shard
       endpoints, indexed by `job_replica_id`.
    3. Per-host schedule slicing, payload encoding, and parallel transfer
       command dispatch across all hosts attached to this entity.
  """

  def __init__(
      self,
      unit: RaidenId,
      shards: Optional[Sequence[str]] = None,
      control_endpoints: Optional[Sequence[str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
      proto_module: Optional[Any] = None,
      use_legacy_tcp_framing: bool = False,
      control_pipe: Optional[control_pipe_client.ControlPipeClient] = None,
      executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
      weight_sync_mode: bool = False,
      on_update: Optional[Callable[["JobEntity"], None]] = None,
  ):
    self.unit = controller_types.entity_key_from_unit(unit)
    self._shards: list[str] = list(shards) if shards else []
    self._endpoints: list[str] = []
    self._hosts: list[HostDescriptor] = []
    self._hosts_by_replica: dict[str, HostDescriptor] = {}
    self._units_by_replica: dict[str, list[RaidenId]] = {}
    self._default_data_name: str = getattr(unit, "data_name", "") or ""
    self._default_data_replica_idx: int = (
        getattr(unit, "data_replica_idx", 0) or 0
    )
    self._endpoints_by_replica: dict[str, list[str]] = {}
    self._shards_by_replica: dict[str, list[str]] = {}
    self._endpoints_by_unit: dict[RaidenId, list[str]] = {}
    self._shards_by_unit: dict[RaidenId, list[str]] = {}
    self._pending_endpoints: Optional[asyncio.Future[list[str]]] = None
    self._pending_endpoints_by_replica: dict[str, asyncio.Future[list[str]]] = (
        {}
    )
    self._resolve_timeout = resolve_timeout
    self._name_resolver = name_resolver
    self._proto_module = proto_module or raiden_service_pb2
    self._weight_sync_mode = weight_sync_mode
    self._on_update = on_update

    self._owns_rpc_client = True
    client_cls = (
        WeightSyncWorkerRpcClient if weight_sync_mode else WorkerRpcClient
    )
    self._worker_rpc_client = client_cls(
        resolve_timeout=resolve_timeout,
        name_resolver=name_resolver,
        proto_module=self._proto_module,
        use_legacy_tcp_framing=use_legacy_tcp_framing,
        control_pipe=control_pipe,
        executor=executor,
        bind_entity=self,
    )

    # Registration metadata owned by this entity.
    self.mesh_shape: Optional[list[int]] = None
    self.mesh_axes: Optional[list[str]] = None
    self.host_subgrid: Optional[list[int]] = None
    self.layout: Optional[list[int]] = None
    self.global_shape: Optional[list[int]] = None
    self.itemsize: Optional[int] = None
    self.pool_manifest: Optional[list[Any]] = None
    self.layout_fingerprint: Optional[str] = None
    self.page_tokens: Optional[int] = None
    self.transfer_parallelism: Optional[int] = None
    self.transfer_rank: Optional[int] = None
    self.variables: Optional[list[Any]] = None
    self.computed_phys_mesh: Optional[list[int]] = None

    initial_rep = getattr(unit, "job_replica_id", "") or ""
    if shards and initial_rep:
      self._shards_by_replica[initial_rep] = list(shards)
    if control_endpoints:
      for ep in control_endpoints:
        self.register_host_endpoint(ep, job_replica_id=initial_rep)
      self._rebuild_host_descriptors()

  def set_on_update(
      self, callback: Optional[Callable[["JobEntity"], None]]
  ) -> None:
    """Registers a callback invoked when entity hosts or shards are updated."""
    self._on_update = callback

  @property
  def worker_rpc_client(self) -> "WorkerRpcClient":
    """Returns the WorkerRpcClient used by this JobEntity."""
    return self._worker_rpc_client

  @property
  def control_pipe_client(self) -> control_pipe_client.ControlPipeClient:
    """Returns the worker ControlPipeClient owned via this entity's WorkerRpcClient."""
    return self._worker_rpc_client.control_pipe_client

  @property
  def executor(self) -> concurrent.futures.ThreadPoolExecutor:
    """Returns the ThreadPoolExecutor used for dispatching host RPCs."""
    return self._worker_rpc_client.executor

  @property
  def name_resolver(self) -> Optional[NameResolver]:
    return self._name_resolver

  @property
  def shards(self) -> list[str]:
    """Returns data-plane shard addresses across all hosts in this entity."""
    return list(self._shards)

  @shards.setter
  def shards(self, value: Sequence[str]) -> None:
    self._shards = list(value)
    self._rebuild_host_descriptors()

  def get_shards_for_unit(self, unit: Optional[RaidenId] = None) -> list[str]:
    """Returns shards for a specific host `unit` (by unit or job_replica_id) or all shards."""
    if unit is not None and unit in self._shards_by_unit:
      return list(self._shards_by_unit[unit])
    if unit is not None and unit.job_replica_id:
      rep_shards = self._shards_by_replica.get(unit.job_replica_id)
      if rep_shards is not None:
        return list(rep_shards)
    return list(self._shards)

  @property
  def host_endpoints(self) -> list[str]:
    """Returns the control-plane RPC addresses for all attached hosts."""
    return list(self._endpoints)

  @property
  def hosts(self) -> list[HostDescriptor]:
    """Returns structured descriptors for all hosts attached to this entity."""
    return list(self._hosts)

  @property
  def num_hosts(self) -> int:
    """Returns the number of hosts attached to this entity."""
    if self._hosts:
      return len(self._hosts)
    if self._endpoints:
      return len(self._endpoints)
    return 1 if self._shards else 0

  @classmethod
  def _replica_sort_key(cls, rep_id: str) -> tuple[int, Any]:
    del cls
    try:
      return (0, int(rep_id))
    except ValueError:
      return (1, rep_id)

  def _rebuild_host_descriptors(self) -> None:
    """Associates data-plane shards with attached host control endpoints."""
    if self._hosts_by_replica:
      ordered_reps = sorted(
          self._hosts_by_replica.keys(), key=self._replica_sort_key
      )
      self._hosts = [self._hosts_by_replica[r] for r in ordered_reps]
      agg_shards: list[str] = []
      agg_eps: list[str] = []
      for r in ordered_reps:
        for s in self._shards_by_replica.get(r, ()):
          if s:
            agg_shards.append(s)
        for ep in self._endpoints_by_replica.get(r, ()):
          if ep and ep not in agg_eps:
            agg_eps.append(ep)
      if agg_shards:
        self._shards = agg_shards
      if agg_eps:
        self._endpoints = agg_eps
      return

    if not self._endpoints:
      self._hosts = (
          [HostDescriptor(control_address="", shards=list(self._shards))]
          if self._shards
          else []
      )
      return
    num_eps = len(self._endpoints)
    num_shards = len(self._shards)
    hosts: list[HostDescriptor] = []
    for idx, ep in enumerate(self._endpoints):
      if num_shards >= num_eps and num_eps > 0:
        start = (idx * num_shards) // num_eps
        end = ((idx + 1) * num_shards) // num_eps
        host_shards = self._shards[start:end]
      elif idx < num_shards:
        host_shards = [self._shards[idx]]
      else:
        host_shards = []
      hosts.append(
          HostDescriptor(
              control_address=ep,
              shards=host_shards,
              job_replica_id=str(idx),
          )
      )
    self._hosts = hosts

  def attach_host(
      self,
      control_address: str = "",
      shards: Optional[Sequence[str]] = None,
      job_replica_id: Optional[str] = None,
      host_index: Optional[int] = None,
      unit: Optional[RaidenId] = None,
  ) -> None:
    """Attaches or updates a host (identified by `job_replica_id` / `host_index`)."""
    rep_id = job_replica_id
    if rep_id is None and unit is not None and unit.job_replica_id:
      rep_id = unit.job_replica_id
    if rep_id is None and host_index is not None:
      rep_id = str(host_index)

    clean_eps = [
        a.strip() for a in str(control_address).split(",") if a.strip()
    ]
    clean_shards = [s for s in shards or () if s]

    if unit is not None:
      if unit.data_name and not self._default_data_name:
        self._default_data_name = unit.data_name
      if unit.data_replica_idx and not self._default_data_replica_idx:
        self._default_data_replica_idx = unit.data_replica_idx
      if clean_shards:
        self._shards_by_unit[unit] = list(clean_shards)
      if clean_eps:
        self._endpoints_by_unit[unit] = list(clean_eps)

    if rep_id:
      host_unit = unit or RaidenId(
          job_name=self.unit.job_name,
          job_replica_id=rep_id,
          data_name=self._default_data_name,
          data_replica_idx=self._default_data_replica_idx,
      )
      rep_units = self._units_by_replica.setdefault(rep_id, [])
      if host_unit not in rep_units:
        rep_units.append(host_unit)
      if clean_shards:
        self._shards_by_replica[rep_id] = list(clean_shards)
        self._shards_by_unit[host_unit] = list(clean_shards)
      if clean_eps:
        self._endpoints_by_replica[rep_id] = list(clean_eps)
        self._endpoints_by_unit[host_unit] = list(clean_eps)
      self._hosts_by_replica[rep_id] = HostDescriptor(
          control_address=",".join(
              self._endpoints_by_replica.get(rep_id, clean_eps)
          ),
          shards=list(self._shards_by_replica.get(rep_id, clean_shards)),
          job_replica_id=rep_id,
          unit=host_unit,
      )
      for ep in clean_eps:
        self.register_host_endpoint(ep, job_replica_id=rep_id)
      self._rebuild_host_descriptors()
      if self._on_update is not None:
        self._on_update(self)
      return

    if clean_shards:
      for s in clean_shards:
        self._shards.append(s)
    for ep in clean_eps:
      self.register_host_endpoint(ep)
    self._rebuild_host_descriptors()
    if self._on_update is not None:
      self._on_update(self)

  def register_host_endpoint(
      self, rpc_address: str, job_replica_id: str = ""
  ) -> None:
    """Registers a host control-plane RPC endpoint on this entity."""
    clean = rpc_address.strip()
    if not clean:
      return
    if job_replica_id:
      rep_eps = self._endpoints_by_replica.setdefault(job_replica_id, [])
      if clean not in rep_eps:
        rep_eps.append(clean)
      default_unit = RaidenId(
          job_name=self.unit.job_name,
          job_replica_id=job_replica_id,
          data_name=self._default_data_name,
          data_replica_idx=self._default_data_replica_idx,
      )
      rep_units = self._units_by_replica.setdefault(job_replica_id, [])
      if default_unit not in rep_units:
        rep_units.append(default_unit)
      if job_replica_id in self._hosts_by_replica:
        self._hosts_by_replica[job_replica_id].control_address = ",".join(
            rep_eps
        )
      else:
        self._hosts_by_replica[job_replica_id] = HostDescriptor(
            control_address=",".join(rep_eps),
            shards=list(self._shards_by_replica.get(job_replica_id, ())),
            job_replica_id=job_replica_id,
            unit=default_unit,
        )
      fut = self._pending_endpoints_by_replica.get(job_replica_id)
      if fut is not None and not fut.done():
        fut.set_result(list(rep_eps))
        self._pending_endpoints_by_replica.pop(job_replica_id, None)
    if clean not in self._endpoints:
      self._endpoints.append(clean)
    self._rebuild_host_descriptors()
    if self._pending_endpoints and not self._pending_endpoints.done():
      self._pending_endpoints.set_result(list(self._endpoints))
      self._pending_endpoints = None
    if self._on_update is not None:
      self._on_update(self)

  def unregister_host_endpoints(self, job_replica_id: str = "") -> None:
    """Clears registered host control-plane RPC endpoints on this entity."""
    if job_replica_id:
      removed = self._endpoints_by_replica.pop(job_replica_id, [])
      self._hosts_by_replica.pop(job_replica_id, None)
      for u in list(self._endpoints_by_unit):
        if getattr(u, "job_replica_id", "") == job_replica_id:
          self._endpoints_by_unit.pop(u, None)
      for ep in removed:
        still_used = any(
            ep in eps for eps in self._endpoints_by_replica.values()
        )
        if not still_used and ep in self._endpoints:
          self._endpoints.remove(ep)
    else:
      self._endpoints.clear()
      self._endpoints_by_replica.clear()
      self._endpoints_by_unit.clear()
      self._hosts_by_replica.clear()
    self._rebuild_host_descriptors()
    if self._on_update is not None:
      self._on_update(self)

  def get_registered_endpoints(self, job_replica_id: str = "") -> list[str]:
    """Returns the list of registered host control-plane RPC endpoints."""
    if job_replica_id and job_replica_id in self._endpoints_by_replica:
      return list(self._endpoints_by_replica[job_replica_id])
    return list(self._endpoints)

  def get_worker_endpoint_str(self, job_replica_id: str = "") -> str:
    """Returns comma-joined control-plane RPC endpoints for this entity."""
    return ",".join(self.get_registered_endpoints(job_replica_id))

  async def resolve_endpoints(self, job_replica_id: str = "") -> list[str]:
    """Resolves host RPC endpoints or suspends until at least one host registers."""
    if job_replica_id:
      rep_eps = self._endpoints_by_replica.get(job_replica_id)
      if rep_eps:
        return list(rep_eps)
      if self._endpoints and not self._endpoints_by_replica:
        return list(self._endpoints)
      fut = self._pending_endpoints_by_replica.get(job_replica_id)
      if fut is None:
        fut = asyncio.Future()
        self._pending_endpoints_by_replica[job_replica_id] = fut
      try:
        res = await asyncio.wait_for(fut, timeout=self._resolve_timeout)
        return list(res)
      except asyncio.TimeoutError as e:
        target = RaidenId(
            job_name=self.unit.job_name,
            job_replica_id=job_replica_id,
            data_name=self.unit.data_name,
            data_replica_idx=self.unit.data_replica_idx,
        )
        raise RuntimeError(
            f"Timeout ({self._resolve_timeout}s) waiting for remote RPC"
            f" endpoint {target} to self-register"
        ) from e
    if self._endpoints:
      return list(self._endpoints)
    if self._pending_endpoints is None:
      self._pending_endpoints = asyncio.Future()
    try:
      res = await asyncio.wait_for(
          self._pending_endpoints, timeout=self._resolve_timeout
      )
      return list(res)
    except asyncio.TimeoutError as e:
      raise RuntimeError(
          f"Timeout ({self._resolve_timeout}s) waiting for remote RPC"
          f" endpoint {self.unit} to self-register"
      ) from e

  def matches_topology(
      self,
      shards_len: int,
      mesh_shape: Optional[list[int]],
      mesh_axes: Optional[list[str]],
      layout: Optional[list[int]],
      global_shape: Optional[list[int]],
      itemsize: Optional[int],
      pool_manifest: Optional[list[Any]],
      layout_fingerprint: Optional[str],
      page_tokens: Optional[int],
      transfer_parallelism: Optional[int],
      transfer_rank: Optional[int],
      variables: Optional[list[Any]],
      host_subgrid: Optional[list[int]],
  ) -> bool:
    """Returns True if the given registration metadata matches this entity's topology."""
    return (
        len(self._shards) == shards_len
        and self.mesh_shape == mesh_shape
        and self.mesh_axes == mesh_axes
        and self.layout == layout
        and self.global_shape == global_shape
        and self.itemsize == itemsize
        and self.pool_manifest == pool_manifest
        and self.layout_fingerprint == layout_fingerprint
        and self.page_tokens == page_tokens
        and self.transfer_parallelism == transfer_parallelism
        and self.transfer_rank == transfer_rank
        and self.variables == variables
        and self.host_subgrid == host_subgrid
    )

  def update_registration(
      self,
      shards: list[str],
      control_endpoints: list[str],
      mesh_shape: Optional[list[int]] = None,
      mesh_axes: Optional[list[str]] = None,
      layout: Optional[list[int]] = None,
      global_shape: Optional[list[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[list[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
      variables: Optional[list[Any]] = None,
      host_subgrid: Optional[list[int]] = None,
      unit: Optional[RaidenId] = None,
  ) -> None:
    """Updates host registration (`unit.job_replica_id`), shards, and endpoints."""
    if unit is not None:
      self._shards_by_unit[unit] = list(shards)
      self._endpoints_by_unit[unit] = list(control_endpoints)
    rep_id = (unit.job_replica_id if unit is not None else "") or ""
    if rep_id:
      self.unregister_host_endpoints(rep_id)
      if unit is not None:
        self._endpoints_by_unit[unit] = list(control_endpoints)
      self._shards_by_replica[rep_id] = list(shards)
      self._endpoints_by_replica[rep_id] = list(control_endpoints)
      self._hosts_by_replica[rep_id] = HostDescriptor(
          control_address=",".join(control_endpoints),
          shards=list(shards),
          job_replica_id=rep_id,
          unit=unit,
      )
      for ep in control_endpoints:
        self.register_host_endpoint(ep, job_replica_id=rep_id)
      self._rebuild_host_descriptors()
    else:
      self._shards = list(shards)
      self.unregister_host_endpoints()
      if unit is not None:
        self._endpoints_by_unit[unit] = list(control_endpoints)
      for ep in control_endpoints:
        self.register_host_endpoint(ep)
      self._rebuild_host_descriptors()

    self.mesh_shape = mesh_shape
    self.mesh_axes = mesh_axes
    self.layout = layout
    self.global_shape = global_shape
    self.itemsize = itemsize
    self.pool_manifest = pool_manifest
    self.layout_fingerprint = layout_fingerprint
    self.page_tokens = page_tokens
    self.transfer_parallelism = transfer_parallelism
    self.transfer_rank = transfer_rank
    self.variables = variables
    self.host_subgrid = host_subgrid
    if self._on_update is not None:
      self._on_update(self)

  def to_metadata_proto(self) -> Any:
    """Builds an owned RegisterWorkUnitRequest metadata proto for this entity."""
    reg_req = raiden_service_pb2.RegisterWorkUnitRequest(
        unit=raiden_service_pb2.RaidenIdProto(
            job_name=self.unit.job_name,
            job_replica_id=self.unit.job_replica_id,
            data_name=self.unit.data_name,
            data_replica_idx=self.unit.data_replica_idx,
        ),
        shards=self._shards,
        control_plane_rpc_address=self.get_worker_endpoint_str(),
        itemsize=self.itemsize or 0,
        layout_fingerprint=self.layout_fingerprint or "",
        page_tokens=self.page_tokens or 0,
        transfer_parallelism=self.transfer_parallelism or 0,
        transfer_rank=self.transfer_rank or 0,
    )
    reg_req.mesh_shape.extend(self.mesh_shape or ())
    reg_req.mesh_axes.extend(self.mesh_axes or ())
    reg_req.host_subgrid.extend(self.host_subgrid or ())
    reg_req.layout.extend(self.layout or ())
    reg_req.global_shape.extend(self.global_shape or ())
    for pool in self.pool_manifest or ():
      reg_req.pools.add().CopyFrom(pool)
    if self.variables is not None:
      for var in self.variables:
        reg_req.variables.add().CopyFrom(
            _coerce_variable_proto(var, raiden_service_pb2)
        )
    return reg_req

  def include_receiver_push_schedules(
      self, transfer_plan: Optional[TransferPlan] = None
  ) -> bool:
    """Returns whether receiver StartTransferRequests need shard_push_schedules."""
    return self._worker_rpc_client.include_receiver_push_schedules(
        transfer_plan
    )

  def _resolve_target_unit(
      self,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
      unit: Optional[RaidenId] = None,
  ) -> RaidenId:
    """Resolves the target RaidenId in `transfer_plan` for `unit` or `address`."""
    plan_units = list(getattr(transfer_plan, "src_units", ())) + list(
        getattr(transfer_plan, "dst_units", ())
    )
    if unit is not None and unit in plan_units:
      return unit
    if address:
      addr_clean = address.strip()
      for h in self._hosts_by_replica.values():
        eps = [a.strip() for a in h.control_address.split(",") if a.strip()]
        if addr_clean in eps and h.unit is not None and h.unit in plan_units:
          return h.unit
    if self.unit in plan_units:
      return self.unit
    for u in plan_units:
      if controller_types.entity_key_from_unit(u) == self.unit:
        return u
    return unit if unit is not None else self.unit

  def get_host_owned_shards(
      self,
      transfer_plan: TransferPlan,
      address: Optional[str],
      unit: Optional[RaidenId] = None,
  ) -> Optional[set[int]]:
    """Determines shard indices owned by a specific host endpoint attached to this entity."""
    if not address:
      return None

    target_id = self._resolve_target_unit(
        transfer_plan, address=address, unit=unit
    )
    addr_clean = address.strip()
    if (
        hasattr(transfer_plan, "endpoint_to_shards")
        and transfer_plan.endpoint_to_shards
    ):
      if (target_id, addr_clean) in transfer_plan.endpoint_to_shards:
        return set(transfer_plan.endpoint_to_shards[(target_id, addr_clean)])
      if (self.unit, addr_clean) in transfer_plan.endpoint_to_shards:
        return set(transfer_plan.endpoint_to_shards[(self.unit, addr_clean)])
      if addr_clean in transfer_plan.endpoint_to_shards:
        return set(transfer_plan.endpoint_to_shards[addr_clean])

    rep_id = getattr(target_id, "job_replica_id", "") or ""
    endpoints = self.get_registered_endpoints(rep_id)
    if not endpoints and hasattr(transfer_plan, "worker_rpc_addresses"):
      rpc_addr = transfer_plan.worker_rpc_addresses.get(
          target_id, transfer_plan.worker_rpc_addresses.get(self.unit, "")
      )
      if rpc_addr:
        if isinstance(rpc_addr, (list, tuple)):
          endpoints = [str(a).strip() for a in rpc_addr if str(a).strip()]
        else:
          endpoints = [a.strip() for a in str(rpc_addr).split(",") if a.strip()]

    if not endpoints or len(endpoints) <= 1:
      return None

    norm_endpoints = []
    for e in endpoints:
      clean_e = e.strip()
      if clean_e and clean_e not in norm_endpoints:
        norm_endpoints.append(clean_e)

    if addr_clean not in norm_endpoints:
      return None
    worker_idx = norm_endpoints.index(addr_clean)
    num_workers = len(norm_endpoints)

    num_shards = 0
    data_shards = getattr(transfer_plan, "worker_data_addresses", {}).get(
        target_id, []
    )
    if data_shards:
      num_shards = len(data_shards)
    if num_shards == 0:
      cached_protos = getattr(
          transfer_plan, "sender_push_schedule_protos", None
      )
      if (
          cached_protos
          and target_id in cached_protos
          and cached_protos[target_id]
      ):
        num_shards = max(
            len(cached_protos[target_id]),
            max(cached_protos[target_id].keys()) + 1,
        )
    if num_shards == 0:
      push_schedules = getattr(transfer_plan, "shard_push_schedules", {}).get(
          target_id, {}
      )
      if push_schedules:
        num_shards = max(
            len(push_schedules),
            max(push_schedules.keys()) + 1,
        )

    if num_shards <= 1:
      return None

    if num_shards < num_workers:
      return {worker_idx} if worker_idx < num_shards else set()

    start_shard = (worker_idx * num_shards) // num_workers
    end_shard = ((worker_idx + 1) * num_shards) // num_workers
    return set(range(start_shard, end_shard))

  def is_payload_invariant_across_hosts(
      self,
      transfer_plan: TransferPlan,
      addrs: list[str],
      unit: Optional[RaidenId] = None,
  ) -> bool:
    """Returns True if the encoded StartTransferRequest payload is identical across all host addresses."""
    if len(addrs) <= 1:
      return True

    target_id = self._resolve_target_unit(
        transfer_plan, address=addrs[0] if addrs else None, unit=unit
    )
    is_sender = target_id in getattr(
        transfer_plan, "src_units", []
    ) and getattr(transfer_plan, "is_sender", False)
    if is_sender:
      cached_protos = getattr(
          transfer_plan, "sender_push_schedule_protos", None
      )
      has_protos = bool(cached_protos and target_id in cached_protos)
      push_schedules = getattr(transfer_plan, "shard_push_schedules", {}).get(
          target_id
      )
      if not has_protos and not push_schedules:
        return True

      first_owned = self.get_host_owned_shards(
          transfer_plan, addrs[0], unit=target_id
      )
      for addr in addrs[1:]:
        if (
            self.get_host_owned_shards(transfer_plan, addr, unit=target_id)
            != first_owned
        ):
          return False
      return True

    dst_counts = getattr(transfer_plan, "dst_endpoint_counts", None)
    dst_layer_counts = getattr(transfer_plan, "dst_endpoint_layer_counts", None)
    if dst_counts or dst_layer_counts:
      return False
    if self.include_receiver_push_schedules(transfer_plan) and getattr(
        transfer_plan, "shard_push_schedules", None
    ):
      return False
    return True

  def build_sender_push_schedule_protos(
      self, push_schedules: dict[int, list[Any]]
  ) -> dict[int, Any]:
    """Builds ShardPushScheduleProto objects for shards from schedule tuples."""
    target_protos = {}
    for shard_idx, entries in push_schedules.items():
      schedule_proto = self._proto_module.ShardPushScheduleProto()
      groups = {}
      for entry_item in entries:
        if hasattr(entry_item, "dst_peers") or hasattr(entry_item, "dst_peer"):
          if entry_item.dst_peers:
            raw_peers = list(entry_item.dst_peers)
          elif entry_item.dst_peer:
            raw_peers = [entry_item.dst_peer]
          else:
            raw_peers = []
          dst_shard_idx = entry_item.dst_shard_idx
          dst_offset = entry_item.dst_offset_bytes
          src_offset = entry_item.src_offset_bytes
          size = entry_item.size_bytes
          src_block_id = entry_item.src_block_id
          dst_block_id = entry_item.dst_block_id
          src_stride = entry_item.src_stride_bytes
          dst_stride = entry_item.dst_stride_bytes
          count = entry_item.count
          layer_idx = (
              entry_item.layer_idx if entry_item.HasField("layer_idx") else 0
          )
          pool_group = (
              entry_item.pool_group if entry_item.HasField("pool_group") else 0
          )
        else:
          (
              raw_peers,
              dst_shard_idx,
              dst_offset,
              src_offset,
              size,
              src_block_id,
              dst_block_id,
              src_stride,
              dst_stride,
              count,
              *extra,
          ) = entry_item
          layer_idx = extra[0] if extra else 0
          pool_group = extra[1] if len(extra) > 1 else 0
          if not isinstance(raw_peers, (list, tuple, set)):
            raw_peers = [raw_peers]

        key = (
            dst_shard_idx,
            dst_offset,
            src_offset,
            size,
            src_block_id,
            dst_block_id,
            src_stride,
            dst_stride,
            count,
            layer_idx,
            pool_group,
        )
        if key not in groups:
          groups[key] = []
        for p in raw_peers:
          if p and p not in groups[key]:
            groups[key].append(p)

      for key, peers in groups.items():
        (
            dst_shard_idx,
            dst_offset,
            src_offset,
            size,
            src_block_id,
            dst_block_id,
            src_stride,
            dst_stride,
            count,
            layer_idx,
            pool_group,
        ) = key
        entry_proto = schedule_proto.entries.add()
        if peers:
          entry_proto.dst_peer = peers[0]
          entry_proto.dst_peers.extend(peers)
        entry_proto.dst_shard_idx = dst_shard_idx
        entry_proto.dst_offset_bytes = dst_offset
        entry_proto.src_offset_bytes = src_offset
        entry_proto.size_bytes = size
        entry_proto.src_block_id = src_block_id
        entry_proto.dst_block_id = dst_block_id
        entry_proto.src_stride_bytes = src_stride
        entry_proto.dst_stride_bytes = dst_stride
        entry_proto.count = count
        entry_proto.layer_idx = layer_idx
        entry_proto.pool_group = pool_group
      target_protos[shard_idx] = schedule_proto
    return target_protos

  def _raiden_id_to_proto(self, unit: RaidenId) -> Any:
    return self._proto_module.RaidenIdProto(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

  def encode_start_transfer(
      self,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
      unit: Optional[RaidenId] = None,
  ) -> Optional[bytes]:
    """Serializes Protobuf command for collective transfer kickoff on a host of this entity."""
    target_id = self._resolve_target_unit(
        transfer_plan, address=address, unit=unit
    )
    if (
        target_id not in transfer_plan.src_units
        and target_id not in transfer_plan.dst_units
    ):
      return None

    rep_id = getattr(target_id, "job_replica_id", "") or ""
    target_eps = self.get_registered_endpoints(rep_id)
    payload_cache = getattr(transfer_plan, "cached_serialized_payloads", None)
    uuid_val = getattr(transfer_plan, "uuid", None)
    req_id_val = getattr(transfer_plan, "req_id", None)
    skip_d2h_val = bool(getattr(transfer_plan, "skip_d2h", False))
    is_sender = target_id in transfer_plan.src_units and transfer_plan.is_sender
    is_ws = getattr(transfer_plan, "is_weight_sync", False)
    ep_count = len(target_eps)
    include_recv_sched = self.include_receiver_push_schedules(transfer_plan)
    cache_key = (
        target_id,
        address,
        uuid_val,
        req_id_val,
        skip_d2h_val,
        is_sender,
        is_ws,
        ep_count,
        include_recv_sched,
    )
    steady_key = (
        target_id,
        address,
        uuid_val,
        skip_d2h_val,
        is_sender,
        is_ws,
        ep_count,
        include_recv_sched,
    )
    template_key = (
        "__template__",
        target_id,
        address,
        is_sender,
        is_ws,
        int(transfer_plan.dst_mem_type),
        bool(transfer_plan.use_block_chunks),
        int(transfer_plan.parallelism or 0),
        ep_count,
        include_recv_sched,
    )

    if payload_cache is not None:
      if cache_key in payload_cache:
        return payload_cache[cache_key]
      if is_sender and is_ws and steady_key in payload_cache:
        return payload_cache[steady_key]
      if is_sender and is_ws and template_key in payload_cache:
        cached_req = payload_cache[template_key]
        cached_req.start_transfer_request.uuid = int(uuid_val or 0)
        cached_req.start_transfer_request.req_id = str(req_id_val or "")
        cached_req.start_transfer_request.skip_d2h = skip_d2h_val
        serialized_bytes = cached_req.SerializeToString()
        payload_cache[cache_key] = serialized_bytes
        payload_cache[steady_key] = serialized_bytes
        return serialized_bytes

    peers = []
    for dst in transfer_plan.dst_units:
      dst_coords = transfer_plan.worker_data_addresses.get(dst)
      if not dst_coords:
        raise ValueError(f"No data-plane endpoint registered for {dst}")
      peers.extend(dst_coords)

    req = self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_START_TRANSFER,
        peers=peers,
    )

    expected_block_count = transfer_plan.dst_expected_block_counts.get(
        target_id, transfer_plan.expected_block_count
    )
    layer_counts = transfer_plan.dst_expected_layer_chunk_counts.get(
        target_id, transfer_plan.expected_layer_chunk_counts
    )
    if not is_sender and address and len(target_eps) != 1:
      host_ip = _extract_host_ip(address)
      if host_ip in getattr(transfer_plan, "dst_endpoint_counts", {}):
        expected_block_count = transfer_plan.dst_endpoint_counts[host_ip]
        logging.info(
            "Target %s: customized expected_block_count=%d for receiver"
            " endpoint %s (host=%s)",
            target_id,
            expected_block_count,
            address,
            host_ip,
        )
      if host_ip in getattr(transfer_plan, "dst_endpoint_layer_counts", {}):
        layer_counts = transfer_plan.dst_endpoint_layer_counts[host_ip]

    start_req = self._proto_module.StartTransferRequest(
        src_units=[
            self._raiden_id_to_proto(u) for u in transfer_plan.src_units
        ],
        dst_units=[
            self._raiden_id_to_proto(u) for u in transfer_plan.dst_units
        ],
        is_sender=is_sender,
        dst_mem_type=int(transfer_plan.dst_mem_type),
        use_block_chunks=transfer_plan.use_block_chunks,
        expected_block_count=expected_block_count,
        transfer_pool_indices=transfer_plan.transfer_pool_indices,
        pool_dtype_tags=transfer_plan.pool_dtype_tags,
        parallelism=transfer_plan.parallelism,
    )
    for layer_idx, skip in transfer_plan.skip_tiling.items():
      start_req.skip_tiling[layer_idx] = skip

    for layer_idx, count in layer_counts.items():
      start_req.expected_layer_chunk_counts[layer_idx] = count

    for group in transfer_plan.pool_groups:
      group_proto = start_req.pool_groups.add()
      group_proto.pool_indices.extend(int(idx) for idx in group["pool_indices"])
      group_proto.dst_device_block_ids.extend(
          int(bid) for bid in group["dst_device_block_ids"]
      )
      group_proto.expected_pushes = int(group["expected_pushes"])
      group_proto.dst_expected_extent_bytes.extend(
          int(e) for e in group["dst_expected_extent_bytes"]
      )
      group_proto.order_rank = int(group.get("order_rank", 0))

    cached_protos = getattr(transfer_plan, "sender_push_schedule_protos", None)
    if transfer_plan.shard_push_schedules or cached_protos:
      if not is_sender:
        if (
            transfer_plan.shard_push_schedules
            and self.include_receiver_push_schedules(transfer_plan)
        ):
          target_endpoints = transfer_plan.worker_data_addresses.get(
              target_id, []
          )
          for (
              src_unit,
              push_schedules,
          ) in transfer_plan.shard_push_schedules.items():
            num_src_shards = len(push_schedules)
            for shard_idx, schedule in push_schedules.items():
              if num_src_shards == 1:
                key_idx = transfer_plan.src_schedule_keys.get(src_unit)
                if key_idx is None:
                  if src_unit in transfer_plan.src_units:
                    key_idx = transfer_plan.src_units.index(src_unit)
                  else:
                    key_idx = 0
              else:
                src_base = (
                    transfer_plan.src_units.index(src_unit)
                    if src_unit in transfer_plan.src_units
                    else 0
                )
                key_idx = src_base * num_src_shards + shard_idx
              schedule_proto = self._proto_module.ShardPushScheduleProto()
              raw_entries = (
                  schedule.entries if hasattr(schedule, "entries") else schedule
              )
              target_endpoints_set = set(target_endpoints)
              for entry_item in raw_entries:
                if hasattr(entry_item, "dst_peers") or hasattr(
                    entry_item, "dst_peer"
                ):
                  if entry_item.dst_peers:
                    item_peers = list(entry_item.dst_peers)
                  elif entry_item.dst_peer:
                    item_peers = [entry_item.dst_peer]
                  else:
                    item_peers = []
                  matching_peers = [
                      p for p in item_peers if p in target_endpoints_set
                  ]
                  if matching_peers:
                    entry_proto = schedule_proto.entries.add()
                    entry_proto.CopyFrom(entry_item)
                    entry_proto.dst_peer = matching_peers[0]
                    del entry_proto.dst_peers[:]
                    entry_proto.dst_peers.extend(matching_peers)
                else:
                  (
                      dst_peer,
                      dst_shard_idx,
                      dst_offset,
                      src_offset,
                      size,
                      src_block_id,
                      dst_block_id,
                      src_stride,
                      dst_stride,
                      count,
                      *extra,
                  ) = entry_item
                  layer_idx = extra[0] if extra else 0
                  pool_group = extra[1] if len(extra) > 1 else 0
                  item_peers = (
                      list(dst_peer)
                      if isinstance(dst_peer, (list, tuple, set))
                      else ([dst_peer] if dst_peer else [])
                  )
                  matching_peers = [
                      p for p in item_peers if p in target_endpoints_set
                  ]
                  if matching_peers:
                    entry_proto = schedule_proto.entries.add()
                    entry_proto.dst_peer = matching_peers[0]
                    entry_proto.dst_peers.extend(matching_peers)
                    entry_proto.dst_shard_idx = dst_shard_idx
                    entry_proto.dst_offset_bytes = dst_offset
                    entry_proto.src_offset_bytes = src_offset
                    entry_proto.size_bytes = size
                    entry_proto.src_block_id = src_block_id
                    entry_proto.dst_block_id = dst_block_id
                    entry_proto.src_stride_bytes = src_stride
                    entry_proto.dst_stride_bytes = dst_stride
                    entry_proto.count = count
                    entry_proto.layer_idx = layer_idx
                    entry_proto.pool_group = pool_group
              if len(schedule_proto.entries) > 0:
                start_req.shard_push_schedules[key_idx].CopyFrom(schedule_proto)
      else:
        owned_shards = None
        if address:
          owned_shards = self.get_host_owned_shards(
              transfer_plan, address, unit=target_id
          )

        cached_protos = getattr(
            transfer_plan, "sender_push_schedule_protos", None
        )
        if cached_protos is not None and target_id in cached_protos:
          for shard_idx, schedule_proto in cached_protos[target_id].items():
            if owned_shards is None or shard_idx in owned_shards:
              start_req.shard_push_schedules[shard_idx].CopyFrom(schedule_proto)
        else:
          push_schedules = transfer_plan.shard_push_schedules.get(target_id)
          if push_schedules:
            target_protos = self.build_sender_push_schedule_protos(
                push_schedules
            )
            for shard_idx, schedule_proto in target_protos.items():
              if owned_shards is None or shard_idx in owned_shards:
                start_req.shard_push_schedules[shard_idx].CopyFrom(
                    schedule_proto
                )
            if cached_protos is not None:
              cached_protos[target_id] = target_protos

    start_req.uuid = int(uuid_val or 0)
    start_req.req_id = str(req_id_val or "")
    start_req.skip_d2h = skip_d2h_val
    req.start_transfer_request.CopyFrom(start_req)
    serialized_bytes = req.SerializeToString()
    if payload_cache is not None:
      payload_cache[cache_key] = serialized_bytes
      if is_sender and is_ws:
        payload_cache[steady_key] = serialized_bytes
        payload_cache[template_key] = req
    return serialized_bytes

  # pylint: disable=protected-access
  async def _send_rpc(
      self, addr: str, payload: bytes, timeout: float = 600.0
  ) -> bytes:
    """Dispatches RPC payload via this entity's WorkerRpcClient."""
    return await self._worker_rpc_client._send_rpc(
        addr, payload, timeout=timeout
    )

  async def _send_and_verify(self, addr: str, payload: bytes) -> None:
    """Sends RPC payload to `addr` and verifies the response status via WorkerRpcClient."""
    await self._worker_rpc_client._send_and_verify(addr, payload)

  def _encode_for_host(
      self,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
      unit: Optional[RaidenId] = None,
  ) -> Optional[bytes]:
    """Encodes StartTransferRequest for `address` via WorkerRpcClient override or entity."""
    target_id = self._resolve_target_unit(
        transfer_plan, address=address, unit=unit
    )
    if (
        type(self._worker_rpc_client)._encode_start_transfer
        is not WorkerRpcClient._encode_start_transfer
        or "_encode_start_transfer" in self._worker_rpc_client.__dict__
    ):
      try:
        return self._worker_rpc_client._encode_start_transfer(
            target_id, transfer_plan, address=address
        )
      except TypeError:
        return self._worker_rpc_client._encode_start_transfer(
            target_id, transfer_plan
        )
    return self.encode_start_transfer(
        transfer_plan, address=address, unit=target_id
    )

  async def _execute_start_transfer(
      self,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
      unit: Optional[RaidenId] = None,
  ) -> None:
    """Encodes and dispatches StartTransferRequests to hosts attached to this JobEntity."""
    target_id = self._resolve_target_unit(
        transfer_plan, address=address, unit=unit
    )
    rep_id = getattr(target_id, "job_replica_id", "") or ""
    if address:
      addrs = [a.strip() for a in address.split(",") if a.strip()]
    else:
      addrs = await self.resolve_endpoints(rep_id)

    coros = []
    if self.is_payload_invariant_across_hosts(
        transfer_plan, addrs, unit=target_id
    ):
      try:
        spec_addr = addrs[0] if addrs else None
        payload = self._encode_for_host(
            transfer_plan, address=spec_addr, unit=target_id
        )
      except NotImplementedError:
        payload = None
      if payload:
        for addr in addrs:
          coros.append(self._send_and_verify(addr, payload))
    else:
      for addr in addrs:
        try:
          payload = self._encode_for_host(
              transfer_plan, address=addr, unit=target_id
          )
          if not payload:
            continue
        except NotImplementedError:
          continue
        coros.append(self._send_and_verify(addr, payload))

    if coros:
      await asyncio.gather(*coros)

  async def start_transfer(
      self,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
      unit: Optional[RaidenId] = None,
  ) -> None:
    """Dispatches transfer commands to hosts attached to this JobEntity."""
    target_id = self._resolve_target_unit(
        transfer_plan, address=address, unit=unit
    )
    if (
        type(self._worker_rpc_client).start_transfer
        not in (
            WorkerRpcClient.start_transfer,
            WeightSyncWorkerRpcClient.start_transfer,
        )
        or "start_transfer" in self._worker_rpc_client.__dict__
    ):
      await self._worker_rpc_client.start_transfer(target_id, transfer_plan)
      return
    await self._execute_start_transfer(
        transfer_plan, address=address, unit=target_id
    )

  # pylint: enable=protected-access

  def _encode_shutdown(self) -> bytes:
    req = self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_SHUTDOWN
    )
    return req.SerializeToString()

  async def shutdown_hosts(self, timeout: float = 10.0) -> None:
    """Dispatches remote shutdown signaling payloads to all hosts attached to this entity."""
    payload = self._encode_shutdown()
    if self._endpoints:
      await asyncio.gather(
          *[
              self._send_rpc(addr, payload, timeout=timeout)
              for addr in self._endpoints
          ],
          return_exceptions=True,
      )

  def close(self) -> None:
    """Closes the owned WorkerRpcClient when created by this entity."""
    if self._owns_rpc_client and self._worker_rpc_client is not None:
      self._worker_rpc_client.close()


# Alias for callers referencing HostGroup
HostGroup = JobEntity


class _EndpointsView(abc.MutableMapping):
  """MutableMapping view over JobEntity endpoints keyed by RaidenId."""

  def __init__(self, client: "WorkerRpcClient"):
    self._client = client

  def __getitem__(self, key: RaidenId) -> list[str]:
    entity = self._client.get_entity(key)
    if entity is None:
      raise KeyError(key)
    if key in entity._endpoints_by_unit:  # pylint: disable=protected-access
      return entity._endpoints_by_unit[key]  # pylint: disable=protected-access
    rep_id = getattr(key, "job_replica_id", "") or ""
    if rep_id and rep_id in entity._endpoints_by_replica:  # pylint: disable=protected-access
      return entity._endpoints_by_replica[rep_id]  # pylint: disable=protected-access
    if not rep_id:
      return entity._endpoints  # pylint: disable=protected-access
    raise KeyError(key)

  def __setitem__(self, key: RaidenId, value: Sequence[str]) -> None:
    entity = self._client.get_or_create_entity(key)
    rep_id = getattr(key, "job_replica_id", "") or ""
    entity.unregister_host_endpoints(rep_id)
    entity._endpoints_by_unit[key] = [str(a).strip() for a in value if str(a).strip()]  # pylint: disable=protected-access
    for addr in value:
      entity.register_host_endpoint(addr, job_replica_id=rep_id)

  def __delitem__(self, key: RaidenId) -> None:
    entity = self._client.get_entity(key)
    if entity is None:
      raise KeyError(key)
    rep_id = getattr(key, "job_replica_id", "") or ""
    if rep_id and rep_id not in entity._endpoints_by_replica:  # pylint: disable=protected-access
      raise KeyError(key)
    entity.unregister_host_endpoints(rep_id)

  def __iter__(self):
    for k, entity in self._client._entities.items():  # pylint: disable=protected-access
      if entity._endpoints_by_replica:  # pylint: disable=protected-access
        for rep_id, eps in entity._endpoints_by_replica.items():  # pylint: disable=protected-access
          if eps:
            units = entity._units_by_replica.get(rep_id) or [  # pylint: disable=protected-access
                RaidenId(
                    job_name=entity.unit.job_name,
                    job_replica_id=rep_id,
                    data_name=entity._default_data_name,  # pylint: disable=protected-access
                    data_replica_idx=entity._default_data_replica_idx,  # pylint: disable=protected-access
                )
            ]
            for u in units:
              yield u
      elif entity._endpoints:  # pylint: disable=protected-access
        yield k

  def __len__(self) -> int:
    return sum(1 for _ in self.__iter__())


class WorkerRpcClient:
  """Distributed RPC Client handling ControlPipeClient transport and worker RPCs."""

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
      proto_module: Optional[Any] = None,
      max_workers: Optional[int] = None,
      use_legacy_tcp_framing: bool = False,
      control_pipe: Optional[control_pipe_client.ControlPipeClient] = None,
      executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
      bind_entity: Optional[JobEntity] = None,
  ):
    self._resolve_timeout = resolve_timeout
    self._name_resolver = name_resolver
    self._proto_module = proto_module or raiden_service_pb2
    self._use_legacy_tcp_framing = use_legacy_tcp_framing
    self._owns_control_pipe = control_pipe is None
    self._owns_executor = executor is None
    if control_pipe is not None:
      self._control_pipe_client = control_pipe
    else:
      self._control_pipe_client = control_pipe_client.ControlPipeClient(
          name_resolver=name_resolver,
          use_legacy_tcp_framing=use_legacy_tcp_framing,
      )
    if executor is not None:
      self._executor = executor
    else:
      self._executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=_default_max_workers(max_workers),
          thread_name_prefix="WorkerRpcClient",
      )
    self._entities: dict[RaidenId, JobEntity] = {}
    if bind_entity is not None:
      self._entities[bind_entity.unit] = bind_entity
    self._endpoints = _EndpointsView(self)
    if endpoint_addresses:
      for k, v in endpoint_addresses.items():
        addrs = [v] if isinstance(v, str) else list(v)
        for addr in addrs:
          self.register_worker_endpoint(k, addr)

  @property
  def control_pipe_client(self) -> control_pipe_client.ControlPipeClient:
    """Returns the ControlPipeClient owned by this WorkerRpcClient."""
    return self._control_pipe_client

  @property
  def entities(self) -> dict[RaidenId, JobEntity]:
    """Returns the dictionary of managed JobEntity instances."""
    return self._entities

  def get_entity(self, unit: RaidenId) -> Optional[JobEntity]:
    key = controller_types.entity_key_from_unit(unit)
    return self._entities.get(key)

  def get_or_create_entity(self, unit: RaidenId) -> JobEntity:
    """Returns existing JobEntity for `unit` (by entity key) or creates a new one."""
    key = controller_types.entity_key_from_unit(unit)
    entity = self._entities.get(key)
    if entity is None:
      entity = JobEntity(
          unit=key,
          resolve_timeout=self._resolve_timeout,
          name_resolver=self._name_resolver,
          proto_module=self._proto_module,
          use_legacy_tcp_framing=self._use_legacy_tcp_framing,
          control_pipe=self._control_pipe_client,
          executor=self._executor,
          weight_sync_mode=isinstance(self, WeightSyncWorkerRpcClient),
      )
      entity._worker_rpc_client = self  # pylint: disable=protected-access
      entity._owns_rpc_client = False  # pylint: disable=protected-access
      self._entities[key] = entity
    if getattr(unit, "data_name", "") and not entity._default_data_name:  # pylint: disable=protected-access
      entity._default_data_name = unit.data_name  # pylint: disable=protected-access
    if (
        getattr(unit, "data_replica_idx", 0)
        and not entity._default_data_replica_idx  # pylint: disable=protected-access
    ):
      entity._default_data_replica_idx = unit.data_replica_idx  # pylint: disable=protected-access
    rep_id = getattr(unit, "job_replica_id", "") or ""
    if rep_id and isinstance(unit, RaidenId):
      rep_units = entity._units_by_replica.setdefault(rep_id, [])  # pylint: disable=protected-access
      if unit not in rep_units:
        rep_units.append(unit)
    return entity

  def bind_entities(
      self,
      entities: dict[RaidenId, JobEntity],
      override_entity_client: bool = True,
  ) -> None:
    """Shares the controller's JobEntity registry with this client."""
    for unit, existing in self._entities.items():
      key = controller_types.entity_key_from_unit(unit)
      if key not in entities:
        if override_entity_client:
          existing._worker_rpc_client = self  # pylint: disable=protected-access
          existing._owns_rpc_client = False  # pylint: disable=protected-access
        entities[key] = existing
      else:
        for rep_id, eps in existing._endpoints_by_replica.items():  # pylint: disable=protected-access
          for ep in eps:
            entities[key].register_host_endpoint(ep, job_replica_id=rep_id)
        for ep in existing.get_registered_endpoints():
          entities[key].register_host_endpoint(ep)
    self._entities = entities
    if override_entity_client:
      for entity in self._entities.values():
        entity._worker_rpc_client = self  # pylint: disable=protected-access
        entity._owns_rpc_client = False  # pylint: disable=protected-access

  @property
  def executor(self) -> concurrent.futures.ThreadPoolExecutor:
    return self._executor

  def close(self) -> None:
    """Closes ControlPipeClient and shuts down the internal ThreadPoolExecutor."""
    for entity in list(self._entities.values()):
      if entity.worker_rpc_client is not self:
        entity.close()
    if (
        self._owns_control_pipe
        and hasattr(self, "_control_pipe_client")
        and self._control_pipe_client
    ):
      self._control_pipe_client.close()
    if self._owns_executor and hasattr(self, "_executor") and self._executor:
      self._executor.shutdown(wait=False)

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  @property
  def name_resolver(self) -> Optional[NameResolver]:
    return self._name_resolver

  def register_worker_endpoint(
      self, worker_name: RaidenId, rpc_address: str
  ) -> None:
    """Registers a host control-plane RPC address onto the JobEntity for `worker_name`."""
    entity = self.get_or_create_entity(worker_name)
    entity.register_host_endpoint(
        rpc_address, job_replica_id=getattr(worker_name, "job_replica_id", "")
    )

  def unregister_worker_endpoint(self, worker_name: RaidenId) -> None:
    """Removes registered host endpoints from the JobEntity for `worker_name`."""
    entity = self.get_entity(worker_name)
    if entity is not None:
      entity.unregister_host_endpoints(
          job_replica_id=getattr(worker_name, "job_replica_id", "")
      )

  async def _resolve_endpoint(self, target_id: RaidenId) -> str:
    addrs = await self._resolve_endpoints(target_id)
    return addrs[0] if addrs else ""

  async def _resolve_endpoints(self, target_id: RaidenId) -> list[str]:
    entity = self.get_or_create_entity(target_id)
    return await entity.resolve_endpoints(
        job_replica_id=getattr(target_id, "job_replica_id", "")
    )

  async def _send_rpc(
      self, addr: str, payload: bytes, timeout: float = 600.0
  ) -> bytes:
    """Connects to remote address, sends payload, and returns the response bytes."""
    if (
        "_send_rpc_sync" not in self.__dict__
        and "send_raw_bytes_sync"
        not in getattr(self._control_pipe_client, "__dict__", {})
        and self._control_pipe_client.backend
        != control_pipe_client.ControlPipeBackendType.TCP
    ):
      return await self._control_pipe_client.send_raw_bytes(
          addr,
          payload,
          timeout=timeout,
          message_type=self._proto_module.ControlRequest.DESCRIPTOR.full_name,
      )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        self._executor, self._send_rpc_sync, addr, payload, timeout
    )

  def _send_rpc_sync(
      self, addr: str, payload: bytes, timeout: float = 600.0
  ) -> bytes:
    """Dispatches payload synchronously via ControlPipeClient and returns response bytes."""
    return self._control_pipe_client.send_raw_bytes_sync(
        addr,
        payload,
        timeout=timeout,
        message_type=self._proto_module.ControlRequest.DESCRIPTOR.full_name,
    )

  def _get_worker_owned_shards(
      self,
      target_id: RaidenId,
      transfer_plan: TransferPlan,
      address: Optional[str],
  ) -> Optional[set[int]]:
    """Delegates shard ownership calculation to the JobEntity for `target_id`."""
    return self.get_or_create_entity(target_id).get_host_owned_shards(
        transfer_plan, address, unit=target_id
    )

  def _is_payload_invariant_across_addrs(
      self,
      target_id: RaidenId,
      transfer_plan: TransferPlan,
      addrs: list[str],
  ) -> bool:
    """Delegates payload invariance check to the JobEntity for `target_id`."""
    return self.get_or_create_entity(
        target_id
    ).is_payload_invariant_across_hosts(transfer_plan, addrs, unit=target_id)

  async def start_transfer(
      self,
      target_id: RaidenId,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
  ) -> None:
    """Delegates transfer kickoff to the JobEntity for `target_id`."""
    entity = self.get_or_create_entity(target_id)
    await entity._execute_start_transfer(  # pylint: disable=protected-access
        transfer_plan, address=address, unit=target_id
    )

  async def _send_and_verify(self, addr: str, payload: bytes) -> None:
    resp_bytes = await self._send_rpc(addr, payload, timeout=1800.0)
    self._verify_response(resp_bytes)

  def _raiden_id_to_proto(self, unit: RaidenId) -> Any:
    return self._proto_module.RaidenIdProto(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

  def _encode_start_transfer(
      self,
      target_id: RaidenId,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
  ) -> Optional[bytes]:
    """Delegates start_transfer encoding to the JobEntity for `target_id`."""
    return self.get_or_create_entity(target_id).encode_start_transfer(
        transfer_plan, address=address, unit=target_id
    )

  def build_sender_push_schedule_protos(
      self, push_schedules: dict[int, list[Any]]
  ) -> dict[int, Any]:
    """Builds ShardPushScheduleProto objects for shards from schedule tuples."""
    dummy_entity = self.get_or_create_entity(RaidenId())
    return dummy_entity.build_sender_push_schedule_protos(push_schedules)

  def include_receiver_push_schedules(
      self, transfer_plan: Optional[TransferPlan] = None
  ) -> bool:
    """Returns whether receiver StartTransferRequests need shard_push_schedules."""
    if transfer_plan is not None and getattr(
        transfer_plan, "is_weight_sync", False
    ):
      return False
    return True

  def _verify_response(self, resp_bytes: bytes) -> None:
    resp = self._proto_module.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(
          f"Raiden remote native execution failed: {resp.message}"
      )

  def get_worker_endpoints(self) -> dict[RaidenId, str]:
    """Returns active read-only snapshot of known registered Worker RPC endpoints."""
    res: dict[RaidenId, str] = {}
    for k, entity in self._entities.items():
      if entity._endpoints_by_replica:  # pylint: disable=protected-access
        for rep_id, eps in entity._endpoints_by_replica.items():  # pylint: disable=protected-access
          if eps:
            ep_str = ",".join(eps)
            units = entity._units_by_replica.get(rep_id) or [  # pylint: disable=protected-access
                RaidenId(
                    job_name=entity.unit.job_name,
                    job_replica_id=rep_id,
                    data_name=entity._default_data_name,  # pylint: disable=protected-access
                    data_replica_idx=entity._default_data_replica_idx,  # pylint: disable=protected-access
                )
            ]
            for host_unit in units:
              res[host_unit] = ep_str
      elif entity.get_registered_endpoints():
        res[k] = entity.get_worker_endpoint_str()
    return res

  def get_registered_endpoints(self, worker_name: RaidenId) -> list[str]:
    """Returns list of registered RPC endpoints for the given worker."""
    entity = self.get_entity(worker_name)
    if not entity:
      return []
    return entity.get_registered_endpoints(
        getattr(worker_name, "job_replica_id", "")
    )

  async def shutdown_workers(self, timeout: float = 10.0) -> None:
    """Dispatches remote shutdown signaling payloads to all registered JobEntities."""
    payload = self._encode_shutdown()
    all_addrs = set()
    for entity in self._entities.values():
      all_addrs.update(entity.get_registered_endpoints())
    if all_addrs:
      await asyncio.gather(
          *[
              self._send_rpc(addr, payload, timeout=timeout)
              for addr in all_addrs
          ],
          return_exceptions=True,
      )

  def _encode_shutdown(self) -> bytes:
    req = self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_SHUTDOWN
    )
    return req.SerializeToString()


class WeightSyncWorkerRpcClient(WorkerRpcClient):
  """Concrete domain subclass for Weight Synchronizer Protobuf serialization."""

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
      proto_module: Optional[Any] = None,
      max_workers: Optional[int] = None,
      use_legacy_tcp_framing: bool = False,
      control_pipe: Optional[control_pipe_client.ControlPipeClient] = None,
      executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
      bind_entity: Optional[JobEntity] = None,
  ):
    super().__init__(
        endpoint_addresses=endpoint_addresses,
        resolve_timeout=resolve_timeout,
        name_resolver=name_resolver,
        proto_module=proto_module or raiden_service_pb2,
        max_workers=max_workers,
        use_legacy_tcp_framing=use_legacy_tcp_framing,
        control_pipe=control_pipe,
        executor=executor,
        bind_entity=bind_entity,
    )

  def include_receiver_push_schedules(
      self, transfer_plan: Optional[TransferPlan] = None
  ) -> bool:
    """Weight sync C++ receivers only consume expected block/layer counts."""
    return False

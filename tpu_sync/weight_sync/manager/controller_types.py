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

"""Shared types, dataclasses, enums, and protobuf helpers for Raiden Controller."""

from collections import abc
import dataclasses
import enum
import threading
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import raiden_service_pb2

NDSlice = list[tuple[int, int]]


class NameResolver(Protocol):
  """Interface for resolving remote network coordinates (e.g.

  BNS) to raw IP addresses.
  """

  def resolve(self, address_str: str) -> str:
    ...


class RaidenMemoryType(enum.IntEnum):
  """Raiden memory type constants."""

  DRAM = 1
  HBM = 2


@dataclasses.dataclass
class _VariableMetadata:
  """Metadata for a variable registered on a worker.

  When global_shard_indices is provided, each entry
  corresponds to the global shard index owned by the local shard at that index.
  In this mode:
    - mesh_axes is no longer necessary.
    - sharding_spec is no longer necessary.
    - top-level mesh_shape is no longer necessary.
  """

  name: str
  shape: list[int]
  mesh_shape: list[int]
  layout: list[int]
  item_size: int
  layer_idx: int
  sharding_spec: list[str] = dataclasses.field(default_factory=list)
  global_shard_indices: list[int] = dataclasses.field(default_factory=list)


def _coerce_variable_proto(var: Any, proto_module=raiden_service_pb2) -> Any:
  """Coerces _VariableMetadata or proto into a VariableMetadataProto."""
  if isinstance(var, proto_module.VariableMetadataProto):
    return var
  return proto_module.VariableMetadataProto(
      name=var.name,
      shape=var.shape,
      mesh_shape=var.mesh_shape,
      layout=var.layout,
      item_size=var.item_size,
      layer_idx=var.layer_idx,
      sharding_spec=getattr(var, "sharding_spec", []),
      global_shard_indices=getattr(var, "global_shard_indices", []),
  )


def _is_variable_spec_identical(
    src_var: _VariableMetadata, dst_var: _VariableMetadata
) -> bool:
  """Returns True if the shape, layout, and mesh_shape match between variables."""
  return (
      list(src_var.shape) == list(dst_var.shape)
      and list(src_var.layout) == list(dst_var.layout)
      and list(src_var.mesh_shape) == list(dst_var.mesh_shape)
  )


class _PlanReferencedShardSchedule(abc.Sequence):
  """Shard push schedule backed by a plan_id dictionary and variable->plan_id map.

  Instead of duplicating schedule entry tuples for every variable that shares
  identical shape, source sharding, destination sharding, and layout, each
  unique variable plan is stored once in `plans_by_id[plan_id]`, and each
  variable (`layer_idx`) references its `plan_id` via `variable_to_plan_id`.
  """

  def __init__(
      self,
      plans_by_id: dict[int, list[tuple[Any, ...]]],
      variable_to_plan_id: dict[int, int],
      ordered_vars_and_plans: Optional[list[tuple[int, int]]] = None,
      pool_group: int = 0,
  ) -> None:
    self.plans_by_id = plans_by_id
    self.variable_to_plan_id = variable_to_plan_id
    self._ordered_vars = (
        ordered_vars_and_plans
        if ordered_vars_and_plans is not None
        else list(variable_to_plan_id.items())
    )
    self.pool_group = pool_group
    self._total_len: Optional[int] = None
    self._materialized: Optional[list[tuple[Any, ...]]] = None

  def get_plan_id(self, layer_idx: int) -> Optional[int]:
    """Returns the unique plan_id referenced by `layer_idx`."""
    return self.variable_to_plan_id.get(layer_idx)

  def get_plan(self, layer_idx: int) -> list[tuple[Any, ...]]:
    """Returns the stored template entries for `layer_idx` via its `plan_id`."""
    plan_id = self.variable_to_plan_id.get(layer_idx)
    if plan_id is None:
      return []
    return self.plans_by_id.get(plan_id, [])

  @property
  def unique_entry_count(self) -> int:
    """Returns the number of unique schedule entries stored across all plan_ids."""
    return sum(len(entries) for entries in self.plans_by_id.values())

  def __len__(self) -> int:
    if self._total_len is None:
      plans = self.plans_by_id
      self._total_len = sum(
          len(plans.get(pid, ())) for _, pid in self._ordered_vars
      )
    return self._total_len

  def __bool__(self) -> bool:
    return any(bool(entries) for entries in self.plans_by_id.values())

  def __iter__(self):
    plans = self.plans_by_id
    pg = self.pool_group
    for layer_idx, pid in self._ordered_vars:
      entries = plans.get(pid)
      if entries:
        for p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 in entries:
          yield (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, layer_idx, pg)

  def _ensure_materialized(self) -> list[tuple[Any, ...]]:
    if self._materialized is None:
      self._materialized = list(iter(self))
    return self._materialized

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self._ensure_materialized()[index]
    n = len(self)
    if index < 0:
      index += n
    if index < 0 or index >= n:
      raise IndexError("schedule index out of range")
    plans = self.plans_by_id
    offset = 0
    for layer_idx, pid in self._ordered_vars:
      entries = plans.get(pid)
      if not entries:
        continue
      elen = len(entries)
      if index < offset + elen:
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = entries[index - offset]
        return (
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
            p9,
            layer_idx,
            self.pool_group,
        )
      offset += elen
    raise IndexError("schedule index out of range")

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, _PlanReferencedShardSchedule):
      if (
          self.plans_by_id == other.plans_by_id
          and self._ordered_vars == other._ordered_vars
          and self.pool_group == other.pool_group
      ):
        return True
    if isinstance(other, abc.Sequence):
      return len(self) == len(other) and self._ensure_materialized() == list(
          other
      )
    return False


@dataclasses.dataclass
class _CachedTransferSchedule:
  """Cached pre-computed transfer schedules and metadata for resharding plans."""

  computed_schedules: dict[Any, Any] = dataclasses.field(default_factory=dict)
  direct_schedules: dict[Any, Any] = dataclasses.field(default_factory=dict)
  broadcast_groups: dict[Any, Any] = dataclasses.field(default_factory=dict)
  local_skip_tiling: dict[int, bool] = dataclasses.field(default_factory=dict)
  expected_block_count: int = 0
  dst_unit_layer_counts: dict[Any, dict[int, int]] = dataclasses.field(
      default_factory=dict
  )
  data_address_to_unit: dict[str, Any] = dataclasses.field(default_factory=dict)
  direct_dsts: list[Any] = dataclasses.field(default_factory=list)
  rpc_addresses: dict[Any, str] = dataclasses.field(default_factory=dict)
  data_addresses: dict[Any, list[str]] = dataclasses.field(default_factory=dict)
  peer_control_endpoints: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  dst_unit_counts: dict[Any, int] = dataclasses.field(default_factory=dict)
  dst_endpoint_counts: dict[str, int] = dataclasses.field(default_factory=dict)
  dst_endpoint_layer_counts: dict[str, dict[int, int]] = dataclasses.field(
      default_factory=dict
  )
  is_weight_sync: bool = False
  sender_push_schedule_protos: dict[Any, dict[int, Any]] = dataclasses.field(
      default_factory=dict
  )
  cached_serialized_payloads: dict[Any, bytes] = dataclasses.field(
      default_factory=dict
  )
  # Dictionary of unique calculated variable plans per source unit:
  # {src_unit: {plan_id: {shard_idx: [entry_tuples]}}}
  variable_plans: dict[Any, dict[int, dict[int, list[Any]]]] = (
      dataclasses.field(default_factory=dict)
  )
  # Mapping from each variable/layer index to its deduplicated plan_id:
  # {src_unit: {layer_idx: plan_id}}
  variable_to_plan_id: dict[Any, dict[int, int]] = dataclasses.field(
      default_factory=dict
  )
  # Deduplicated canonical relay plans mapping plan_id to local destination
  # shard index to list of whole-block specifications:
  # {plan_id: {local_dst_idx: [(min_offset, block_size, dst_block_id)]}}
  canonical_relay_plans: dict[int, dict[int, list[tuple[int, int, int]]]] = (
      dataclasses.field(default_factory=dict)
  )
  n_seed: int = 1


@dataclasses.dataclass
class BroadcastRoundDestinations:
  """Destinations grouped by broadcast round for audit and native logging."""

  round_idx: int
  dst_units: list[str] = dataclasses.field(default_factory=list)
  dst_peers: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class StageBroadcastGroup:
  """A stage-level broadcast group for pipelined multi-hop tree broadcast.

  Instead of unpacking and duplicating millions of micro-slice tuples across
  all destination units and trainer shards, a StageBroadcastGroup retains the
  address-free canonical resharding plan and canonical relay plan for all
  variables in a pipeline stage group.
  """

  pool_group: int
  layer_group_idx: int
  src_units: list[RaidenId]
  dst_units: list[RaidenId]
  stage_ordered_vars_by_unit: dict[RaidenId, list[tuple[int, int]]]
  canonical_variable_plans: dict[RaidenId, dict[int, dict[int, list[Any]]]]
  canonical_relay_plans: dict[int, dict[int, list[tuple[int, int, int]]]]
  data_addresses: dict[RaidenId, list[str]]
  cached_hop_schedules: dict[Any, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )
  eq_class_groups: list["StageBroadcastGroup"] = dataclasses.field(
      default_factory=list, repr=False, compare=False
  )


@dataclasses.dataclass
class TransferPlan:
  """A detailed plan for data transfer with resharding if needed."""

  src_units: list[RaidenId]
  dst_units: list[RaidenId]

  # For push model, maps each source's `RaidenId` to its specific shard push
  # schedule, i.e. shard index to a list of destination's `RaidenId`, shard
  # index, and the n-dimensional slice offsets for the shard index.
  plan: dict[RaidenId, list[list[tuple[RaidenId, int, list[NDSlice]]]]]

  shard_push_schedules: dict[RaidenId, dict[int, Any]] = dataclasses.field(
      default_factory=dict
  )

  # Maps every RaidenId in the plan to its physical Control-Plane RPC
  # address
  worker_rpc_addresses: dict[RaidenId, str] = dataclasses.field(
      default_factory=dict
  )

  # Maps every RaidenId in the plan to its physical Data TCP socket
  # endpoints
  worker_data_addresses: dict[RaidenId, list[str]] = dataclasses.field(
      default_factory=dict
  )
  # Maps peer Data TCP endpoint ("ip:port") to its Peregrine control-plane
  # listener address ("ip:port"). Treated as immutable once attached to a plan.
  peer_control_endpoints: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  # Precomputed sorted-items key for |peer_control_endpoints| built once per
  # transfer for payload-cache lookup; None falls back to sorting on encode.
  peer_control_endpoints_key: Optional[tuple[tuple[str, str], ...]] = None
  uuid: int = 0
  dst_mem_type: int = RaidenMemoryType.DRAM
  use_block_chunks: bool = False
  is_sender: bool = True
  expected_block_count: int = 0
  req_id: str = ""
  expected_pushes_per_pool: int = 0
  transfer_pool_indices: list[int] = dataclasses.field(default_factory=list)
  pool_dtype_tags: list[str] = dataclasses.field(default_factory=list)
  src_block_ids: dict[RaidenId, list[int]] = dataclasses.field(
      default_factory=dict
  )
  dst_device_block_ids: list[int] = dataclasses.field(default_factory=list)
  src_schedule_keys: dict[RaidenId, int] = dataclasses.field(
      default_factory=dict
  )
  parallelism: int = 1
  num_tokens: int = 0
  skipped_pool_counts: dict[str, int] = dataclasses.field(default_factory=dict)
  pool_groups: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  dst_expected_extent_bytes: list[int] = dataclasses.field(default_factory=list)
  request_block_claim_owner: Any = dataclasses.field(
      default=None, repr=False, compare=False
  )
  skip_d2h: bool = False
  skip_tiling: dict[int, bool] = dataclasses.field(default_factory=dict)
  expected_layer_chunk_counts: dict[int, int] = dataclasses.field(
      default_factory=dict
  )
  dst_expected_layer_chunk_counts: dict[RaidenId, dict[int, int]] = (
      dataclasses.field(default_factory=dict)
  )
  dst_expected_block_counts: dict[RaidenId, int] = dataclasses.field(
      default_factory=dict
  )
  dst_endpoint_counts: dict[str, int] = dataclasses.field(default_factory=dict)
  dst_endpoint_layer_counts: dict[str, dict[int, int]] = dataclasses.field(
      default_factory=dict
  )
  is_weight_sync: bool = False
  sender_push_schedule_protos: dict[RaidenId, dict[int, Any]] = (
      dataclasses.field(default_factory=dict, repr=False, compare=False)
  )
  cached_serialized_payloads: dict[Any, bytes] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )
  endpoint_to_shards: dict[Any, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )
  variable_plans: dict[RaidenId, dict[int, dict[int, list[Any]]]] = (
      dataclasses.field(default_factory=dict, repr=False, compare=False)
  )
  variable_to_plan_id: dict[RaidenId, dict[int, int]] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )
  broadcast_round: Optional[int] = None
  broadcast_round_destinations: list[Any] = dataclasses.field(
      default_factory=list
  )
  has_explicit_shard_push_schedules: bool = False
  # raiden_service_pb2.HostTilingMode of the host buffers moved by this plan.
  host_tiling_mode: int = raiden_service_pb2.HOST_TILING_MODE_UNSPECIFIED


class RaidenFuture:
  """Future representing an asynchronous transfer execution."""

  session_id: int

  def __init__(
      self,
      session_id: int = 0,
      transfer_task=None,
      on_complete: Optional[Callable[[], None]] = None,
  ):
    self.session_id = session_id
    self._transfer_task = transfer_task
    self._on_complete = on_complete
    self._completed_event = threading.Event()
    self._completed = False
    self._exception = None
    self._lock = threading.Lock()
    self._started = False

  def try_start(self) -> bool:
    """Attempts to mark the future as started.

    Returns:
      True if this call successfully started it, False otherwise.
    """
    with self._lock:
      if self._started:
        return False
      self._started = True
      return True

  async def wait(self) -> None:
    """Waits asynchronously for the transfer operation to complete."""
    with self._lock:
      if not self._started:
        self._started = True
    if self._transfer_task:
      try:
        await self._transfer_task
      except Exception as e:
        self._exception = e
        raise e
      finally:
        self._transfer_task = None
        self._completed = True
        self._completed_event.set()
        if self._on_complete is not None:
          try:
            self._on_complete()
          except Exception:  # pylint: disable=broad-exception-caught
            pass
    else:
      self._completed = True
      self._completed_event.set()
      if self._on_complete is not None:
        try:
          self._on_complete()
        except Exception:  # pylint: disable=broad-exception-caught
          pass

  def wait_threadsafe(self, timeout=None) -> None:
    """Blocks the calling thread until the transfer is complete."""
    self._completed_event.wait(timeout)

  def done(self) -> bool:
    """Returns True if the transfer operation has completed."""
    return self._completed

  def exception(self) -> Optional[Exception]:
    """Returns the exception raised by the transfer operation, if any."""
    return self._exception


def _extract_host_ip(addr: str) -> str:
  """Extracts the IP address or host string from an endpoint (host:port)."""
  if not addr:
    return ""
  addr = addr.strip()
  if addr.startswith("[") and "]" in addr:
    return addr[1 : addr.index("]")]
  if ":" in addr:
    return addr.rsplit(":", 1)[0]
  return addr


def compute_shard_host_ranks(
    src_shards: Sequence[str], num_src_units: int, unit_idx: int
) -> dict[int, int]:
  """Computes deterministic host rank per source shard index.

  Args:
    src_shards: Sequence of shard network addresses (e.g. 'ip:port') for the
      source unit.
    num_src_units: Total number of source units in the stage broadcast group.
    unit_idx: Index of the current source unit within the source units sequence.

  Returns:
    A dictionary mapping each local shard index in `src_shards` to its
    deterministic host rank integer.
  """
  shard_ips = [_extract_host_ip(s) for s in src_shards]
  unique_ips = list(dict.fromkeys(ip for ip in shard_ips if ip))
  if len(unique_ips) > 1:
    ip_to_rank = {ip: idx for idx, ip in enumerate(unique_ips)}
    return {
        idx: unit_idx * len(unique_ips) + ip_to_rank[shard_ips[idx]]
        for idx in range(len(src_shards))
    }
  else:
    unique_addrs = list(dict.fromkeys(src_shards))
    if num_src_units == 1 and len(unique_addrs) > 1:
      addr_to_rank = {addr: idx for idx, addr in enumerate(unique_addrs)}
      return {
          idx: addr_to_rank[src_shards[idx]] for idx in range(len(src_shards))
      }
    else:
      return {idx: unit_idx for idx in range(len(src_shards))}


def compute_endpoint_to_shards(
    unit: RaidenId,
    control_endpoints: Sequence[str],
    data_shards: Sequence[str],
) -> dict[Any, set[int]]:
  """Maps control-plane RPC endpoints of `unit` to their owned global shard indices.

  Args:
    unit: The work unit `RaidenId`.
    control_endpoints: Sequence of control-plane RPC endpoints (or
      comma-separated endpoint strings) registered for `unit`.
    data_shards: Sequence of data-plane `ip:port` strings for each global shard
      index `0..N-1` of `unit`.

  Returns:
    A dictionary mapping `(unit, ep)` and `(_entity_key_from_unit(unit), ep)` to
    the `set[int]` of global shard indices owned by `ep`, or `{}` when there is
    at most one control endpoint or no host mapping can be inferred.
  """
  ctrl_eps: list[str] = []
  for raw_ep in control_endpoints:
    if not raw_ep:
      continue
    for part in str(raw_ep).split(","):
      clean = part.strip()
      if clean and clean not in ctrl_eps:
        ctrl_eps.append(clean)
  if len(ctrl_eps) <= 1 or not data_shards:
    return {}

  ctrl_by_host: dict[str, list[str]] = {}
  for ep in ctrl_eps:
    h_ip = _extract_host_ip(ep)
    ctrl_by_host.setdefault(h_ip, []).append(ep)

  data_eps_by_host: dict[str, dict[str, set[int]]] = {}
  all_shards_by_host: dict[str, set[int]] = {}
  for shard_idx, shard_addr in enumerate(data_shards):
    clean_shard = shard_addr.strip() if shard_addr else ""
    if not clean_shard:
      continue
    h_ip = _extract_host_ip(clean_shard)
    data_eps_by_host.setdefault(h_ip, {}).setdefault(clean_shard, set()).add(
        shard_idx
    )
    all_shards_by_host.setdefault(h_ip, set()).add(shard_idx)

  if any(h_ip not in data_eps_by_host for h_ip in ctrl_by_host):
    return {}

  entity_key = _entity_key_from_unit(unit)
  result: dict[Any, set[int]] = {}
  num_hosts = len(ctrl_by_host)
  for h_ip, h_ctrl_eps in ctrl_by_host.items():
    unique_data_sets = list(data_eps_by_host[h_ip].values())
    n_ctrl = len(h_ctrl_eps)
    n_data = len(unique_data_sets)
    if n_ctrl == n_data:
      for ep, shard_set in zip(h_ctrl_eps, unique_data_sets):
        owned = set(shard_set)
        result[(unit, ep)] = owned
        result[(entity_key, ep)] = owned
    elif n_ctrl == 1:
      owned = set(all_shards_by_host[h_ip])
      ep = h_ctrl_eps[0]
      result[(unit, ep)] = owned
      result[(entity_key, ep)] = owned
    elif num_hosts > 1:
      owned = set(all_shards_by_host[h_ip])
      for ep in h_ctrl_eps:
        result[(unit, ep)] = owned
        result[(entity_key, ep)] = owned
    else:
      return {}

  return result


def _raiden_id_from_proto(unit: Any) -> RaidenId:
  return RaidenId(
      job_name=unit.job_name,
      job_replica_id=unit.job_replica_id,
      data_name=unit.data_name,
      data_replica_idx=unit.data_replica_idx,
  )


def _raiden_id_to_proto(unit: RaidenId, proto_module=raiden_service_pb2) -> Any:
  return proto_module.RaidenIdProto(
      job_name=unit.job_name,
      job_replica_id=unit.job_replica_id,
      data_name=unit.data_name,
      data_replica_idx=unit.data_replica_idx,
  )


def _proto_to_nd_slice(proto_slice: Any) -> list[tuple[int, int]]:
  """Converts an NDSliceProto message to a Python list of (start, end) tuples."""
  return [(dim.start, dim.end) for dim in proto_slice.dimensions]


def _coerce_pool_spec_proto(pool: Any) -> Any:
  """Returns an owned PoolSpecProto from a proto, mapping, or dataclass."""
  result = raiden_service_pb2.PoolSpecProto()
  if isinstance(pool, raiden_service_pb2.PoolSpecProto):
    result.CopyFrom(pool)
    return result

  def value(name: str, default: Any = None) -> Any:
    if isinstance(pool, Mapping):
      return pool.get(name, default)
    return getattr(pool, name, default)

  result.tag = str(value("tag", ""))
  result.storage_index = int(value("storage_index", 0))
  result.base_offset_bytes = int(value("base_offset_bytes", 0))
  result.block_stride_bytes = int(value("block_stride_bytes", 0))
  result.num_blocks = int(value("num_blocks", 0))
  result.dtype_tag = str(value("dtype_tag", ""))
  for region in value("regions", ()):
    if isinstance(region, Mapping):
      region_value = region.get
    else:
      region_value = lambda name, default=None, r=region: getattr(
          r, name, default
      )
    region_proto = result.regions.add()
    region_proto.name = str(region_value("name", ""))
    region_proto.offset_bytes = int(region_value("offset_bytes", 0))
    region_proto.stride_bytes = int(region_value("stride_bytes", 0))
    region_proto.unit_bytes = int(region_value("unit_bytes", 0))
    region_proto.num_units = int(region_value("num_units", 0))
    region_proto.units_per_stride = int(region_value("units_per_stride", 1))
  return result


def _format_unit(unit: Any) -> str:
  """Formats a RaidenId or work unit into a concise identifier string."""
  if hasattr(unit, "job_name"):
    job = unit.job_name or "unknown"
    rep = f":{unit.job_replica_id}" if unit.job_replica_id else ""
    data_rep = (
        f"#{unit.data_replica_idx}"
        if getattr(unit, "data_replica_idx", 0)
        else ""
    )
    data = f"[{unit.data_name}{data_rep}]" if unit.data_name else ""
    return f"{job}{rep}{data}"
  return str(unit)


def _format_units(units: Any) -> str:
  """Formats a collection of units into a concise comma-separated list string."""
  if isinstance(units, abc.Iterable) and not isinstance(units, (str, bytes)):
    return f"[{', '.join(_format_unit(u) for u in units)}]"
  return _format_unit(units)


def _entity_key_from_unit(unit: RaidenId) -> RaidenId:
  """Constructs the JobEntity key using only job_name in RaidenId.

  For multi-replica jobs (such as samplers), the replica index is encoded in
  `job_name` (e.g. "sampler_0", "sampler_1", ..., "sampler_9", and "trainer"
  for the trainer job). Thus, `RaidenId(job_name=...)` uniquely indexes each
  `JobEntity`, while `unit.job_replica_id` indexes the host within that entity.

  Args:
    unit: The per-host or entity-level RaidenId.

  Returns:
    The entity-level RaidenId with only `job_name` populated.
  """
  if not isinstance(unit, RaidenId):
    return unit
  job_name = unit.job_name
  if (
      unit.data_replica_idx > 0
      and job_name
      and not job_name.endswith(f"_{unit.data_replica_idx}")
  ):
    job_name = f"{job_name}_{unit.data_replica_idx}"
  return RaidenId(job_name=job_name)


VariableMetadata = _VariableMetadata
CachedTransferSchedule = _CachedTransferSchedule
PlanReferencedShardSchedule = _PlanReferencedShardSchedule
coerce_variable_proto = _coerce_variable_proto
is_variable_spec_identical = _is_variable_spec_identical
extract_host_ip = _extract_host_ip
raiden_id_from_proto = _raiden_id_from_proto
raiden_id_to_proto = _raiden_id_to_proto
entity_key_from_unit = _entity_key_from_unit
proto_to_nd_slice = _proto_to_nd_slice
coerce_pool_spec_proto = _coerce_pool_spec_proto
format_unit = _format_unit
format_units = _format_units

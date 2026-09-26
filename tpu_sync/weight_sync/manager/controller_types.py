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
    - host_subgrid is no longer necessary.
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
  ) -> None:
    self.plans_by_id = plans_by_id
    self.variable_to_plan_id = variable_to_plan_id
    self._ordered_vars = (
        ordered_vars_and_plans
        if ordered_vars_and_plans is not None
        else list(variable_to_plan_id.items())
    )
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
    for layer_idx, pid in self._ordered_vars:
      entries = plans.get(pid)
      if entries:
        for p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 in entries:
          yield (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, layer_idx, 0)

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
        return (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, layer_idx, 0)
      offset += elen
    raise IndexError("schedule index out of range")

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, _PlanReferencedShardSchedule):
      if (
          self.plans_by_id == other.plans_by_id
          and self._ordered_vars == other._ordered_vars
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

  computed_schedules: dict[Any, Any]
  direct_schedules: dict[Any, Any]
  broadcast_groups: dict[Any, Any]
  local_skip_tiling: dict[int, bool]
  expected_block_count: int
  dst_unit_layer_counts: dict[Any, dict[int, int]]
  data_address_to_unit: dict[str, Any]
  direct_dsts: list[Any]
  rpc_addresses: dict[Any, str]
  data_addresses: dict[Any, list[str]]
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


SYMBOLIC_ENDPOINT_PREFIX = "raiden_symbolic://"


def _make_symbolic_endpoint(unit: RaidenId, shard_idx: int) -> str:
  """Builds a deterministic symbolic endpoint string for an offline shard."""
  return (
      f"{SYMBOLIC_ENDPOINT_PREFIX}{unit.job_name}/{unit.job_replica_id}/"
      f"{unit.data_name}/{unit.data_replica_idx}:{int(shard_idx)}"
  )


def _make_symbolic_shards(unit: RaidenId, num_shards: int) -> list[str]:
  """Builds a list of symbolic shard endpoints for `unit`."""
  return [_make_symbolic_endpoint(unit, i) for i in range(max(1, num_shards))]


def _is_symbolic_endpoint(endpoint: str) -> bool:
  """Returns True if `endpoint` is an offline symbolic shard placeholder."""
  return isinstance(endpoint, str) and endpoint.startswith(
      SYMBOLIC_ENDPOINT_PREFIX
  )


def _parse_symbolic_endpoint(endpoint: str) -> Optional[tuple[RaidenId, int]]:
  """Parses a symbolic endpoint into (RaidenId, shard_idx), or None if not symbolic."""
  if not _is_symbolic_endpoint(endpoint):
    return None
  body = endpoint[len(SYMBOLIC_ENDPOINT_PREFIX) :]
  if ":" not in body:
    return None
  unit_part, shard_str = body.rsplit(":", 1)
  parts = unit_part.split("/")
  if len(parts) != 4:
    return None
  try:
    shard_idx = int(shard_str)
    data_rep_idx = int(parts[3])
  except ValueError:
    return None
  return (
      RaidenId(
          job_name=parts[0],
          job_replica_id=parts[1],
          data_name=parts[2],
          data_replica_idx=data_rep_idx,
      ),
      shard_idx,
  )


def _resolve_symbolic_endpoint(
    endpoint: str,
    live_data_addresses: Mapping[RaidenId, Sequence[str]],
) -> str:
  """Resolves a symbolic endpoint string against `live_data_addresses`."""
  parsed = _parse_symbolic_endpoint(endpoint)
  if parsed is None:
    return endpoint
  unit, shard_idx = parsed
  shards = live_data_addresses.get(unit)
  if not shards:
    return endpoint
  if 0 <= shard_idx < len(shards):
    return shards[shard_idx]
  return shards[0]


def _bind_symbolic_endpoints_in_proto(
    req: Any,
    live_data_addresses: Mapping[RaidenId, Sequence[str]],
) -> Any:
  """Binds symbolic endpoints in a ControlRequest or StartTransferRequest proto in-place."""
  if not live_data_addresses:
    return req
  if hasattr(req, "peers") and req.peers:
    resolved_peers = [
        _resolve_symbolic_endpoint(p, live_data_addresses) for p in req.peers
    ]
    del req.peers[:]
    req.peers.extend(resolved_peers)

  start_req = (
      req.start_transfer_request
      if hasattr(req, "start_transfer_request")
      else req
  )
  if hasattr(start_req, "shard_push_schedules"):
    for _, schedule_proto in start_req.shard_push_schedules.items():
      for entry in schedule_proto.entries:
        if entry.dst_peers:
          resolved_list = []
          for p in entry.dst_peers:
            rp = _resolve_symbolic_endpoint(p, live_data_addresses)
            if rp not in resolved_list:
              resolved_list.append(rp)
          del entry.dst_peers[:]
          entry.dst_peers.extend(resolved_list)
          if resolved_list:
            entry.dst_peer = resolved_list[0]
        elif entry.dst_peer:
          entry.dst_peer = _resolve_symbolic_endpoint(
              entry.dst_peer, live_data_addresses
          )
  return req


def _unit_filename_stem(unit: RaidenId) -> str:
  """Returns a filesystem-safe filename stem for a work unit's offline plan."""

  def _clean(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

  return (
      f"plan_{_clean(unit.job_name)}_{_clean(unit.job_replica_id)}_"
      f"{_clean(unit.data_name)}_{int(unit.data_replica_idx)}"
  )


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
make_symbolic_endpoint = _make_symbolic_endpoint
make_symbolic_shards = _make_symbolic_shards
is_symbolic_endpoint = _is_symbolic_endpoint
parse_symbolic_endpoint = _parse_symbolic_endpoint
resolve_symbolic_endpoint = _resolve_symbolic_endpoint
bind_symbolic_endpoints_in_proto = _bind_symbolic_endpoints_in_proto
unit_filename_stem = _unit_filename_stem

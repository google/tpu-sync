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
from typing import Any, Callable, Mapping, Optional, Protocol

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


@dataclasses.dataclass
class TransferPlan:
  """A detailed plan for data transfer with resharding if needed."""

  src_units: list[RaidenId]
  dst_units: list[RaidenId]

  # For push model, maps each source's `RaidenId` to its specific shard push
  # schedule, i.e. shard index to a list of destination's `RaidenId`, shard
  # index, and the n-dimensional slice offsets for the shard index.
  plan: dict[RaidenId, list[list[tuple[RaidenId, int, list[NDSlice]]]]]

  shard_push_schedules: dict[
      RaidenId, dict[int, list[tuple[str, int, int, int, int, int, int]]]
  ] = dataclasses.field(default_factory=dict)

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


VariableMetadata = _VariableMetadata
CachedTransferSchedule = _CachedTransferSchedule
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

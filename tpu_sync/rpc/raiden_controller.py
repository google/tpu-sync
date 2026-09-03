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

"""Raiden Controller providing high-level transfer API and resharding plans."""

import asyncio
from collections import abc
import dataclasses
import enum
import functools
import math
import os
import random
import socket
import threading
import time
import typing
from typing import Any, Optional

from absl import logging

from tpu_sync.api.common import RaidenId
from tpu_sync.kv_cache import nd_slice_math
from tpu_sync.rpc import controller_service_pb2
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync import weight_synchronization_worker_service_client


@dataclasses.dataclass
class _VariableMetadata:
  name: str
  shape: list[int]
  mesh_shape: list[int]
  layout: list[int]
  item_size: int
  layer_idx: int
  sharding_spec: list[str] = dataclasses.field(default_factory=list)


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


def to_physical(logical_shape, logical_mesh_shape, minor_to_major):
  """Maps logical tensor and mesh shapes to physical memory layout."""
  logical_shape = list(logical_shape)
  logical_mesh_shape = list(logical_mesh_shape)
  minor_to_major = list(minor_to_major)
  major_to_minor = list(reversed(minor_to_major))
  rank = len(logical_shape)
  if sorted(minor_to_major) == list(range(rank)):
    physical_shape = tuple(logical_shape[d] for d in major_to_minor)
    physical_mesh_shape = tuple(logical_mesh_shape[d] for d in major_to_minor)
  else:
    physical_shape = tuple(
        logical_shape[minor_to_major.index(d)] for d in major_to_minor
    )
    physical_mesh_shape = tuple(logical_mesh_shape[d] for d in major_to_minor)
  return physical_shape, physical_mesh_shape


class NameResolver(typing.Protocol):
  """Interface for resolving remote network coordinates (e.g.

  BNS) to raw IP addresses.
  """

  def resolve(self, address_str: str) -> str:
    ...


class RaidenMemoryType(enum.IntEnum):
  """Raiden memory type constants."""

  DRAM = 1
  HBM = 2


def _raiden_id_from_proto(unit: Any) -> RaidenId:
  return RaidenId(
      job_name=unit.job_name,
      job_replica_id=unit.job_replica_id,
      data_name=unit.data_name,
      data_replica_idx=unit.data_replica_idx,
  )


NDSlice = list[tuple[int, int]]


def _get_global_indices(
    unit: RaidenId,
    shards: list[str],
    logical_mesh_shape: list[int],
    layout: list[int],
    num_physical_hosts: int,
    sharding_spec: Optional[list[str]] = None,
    mesh_axes: Optional[list[str]] = None,
    physical_mesh_shape: Optional[list[int]] = None,
) -> list[tuple[int, int]]:
  """Maps local shard indices to global slice indices, handling replication."""
  num_shards = len(shards)
  if num_shards == 0:
    return []

  try:
    replica_id = int(unit.job_replica_id)
  except ValueError:
    replica_id = 0

  if not logical_mesh_shape:
    return [(i, i) for i in range(num_shards)]

  if all(d == 1 for d in logical_mesh_shape):
    return [(i, 0) for i in range(num_shards)]

  major_to_minor = list(reversed(layout))
  phys_mesh = [logical_mesh_shape[d] for d in major_to_minor]

  host_axis_logical = None
  for d, size in enumerate(logical_mesh_shape):
    if size == num_physical_hosts:
      host_axis_logical = d
      break

  use_spec_mapping = bool(sharding_spec and mesh_axes and physical_mesh_shape)

  if use_spec_mapping:
    # Use physical mesh mapping when host_axis_logical is not found
    devices_per_host = num_shards
    indices = []
    for j in range(num_shards):
      global_device_id = replica_id * devices_per_host + j

      # Reconstruct physical mesh coordinates from global_device_id (row-major)
      phys_coords = []
      temp = global_device_id
      for size in reversed(physical_mesh_shape):
        phys_coords.append(temp % size)
        temp //= size
      phys_coords.reverse()

      # Map physical coordinates to tensor dimensions using sharding_spec
      tensor_coords = []
      for axis_name in sharding_spec:
        if not axis_name:
          tensor_coords.append(0)
        elif "," in axis_name:
          sub_axes = [a.strip() for a in axis_name.split(",") if a.strip()]
          coord = 0
          for sub_a in sub_axes:
            try:
              phys_axis_idx = mesh_axes.index(sub_a)
              sub_size = physical_mesh_shape[phys_axis_idx]
              coord = coord * sub_size + phys_coords[phys_axis_idx]
            except ValueError:
              logging.warning(
                  "Sub-spec axis %s not found in mesh axes %s", sub_a, mesh_axes
              )
          tensor_coords.append(coord)
        else:
          try:
            phys_axis_idx = mesh_axes.index(axis_name)
            tensor_coords.append(phys_coords[phys_axis_idx])
          except ValueError:
            logging.warning(
                "Spec axis %s not found in mesh axes %s", axis_name, mesh_axes
            )
            tensor_coords.append(0)

      # Compute flat index in logical_mesh_shape (row-major)
      global_idx = 0
      stride = 1
      for val, size in zip(
          reversed(tensor_coords), reversed(logical_mesh_shape)
      ):
        global_idx += val * stride
        stride *= size
      indices.append((j, global_idx))

    return indices

  else:
    if host_axis_logical is None:
      logging.warning(
          "host_axis_logical is None and sharding_spec, mesh_axes, or"
          " physical_mesh_shape is missing for %s. Falling back to legacy"
          " mapping.",
          unit,
      )

    non_host_axes = [
        d for d in range(len(logical_mesh_shape)) if d != host_axis_logical
    ]

    indices = []
    for j in range(num_shards):
      local_coords = {}
      temp = j
      for d in reversed(non_host_axes):
        size = logical_mesh_shape[d]
        local_coords[d] = temp % size
        temp = temp // size

      full_coords = [0] * len(logical_mesh_shape)
      for d in range(len(logical_mesh_shape)):
        if d == host_axis_logical:
          full_coords[d] = replica_id
        else:
          full_coords[d] = local_coords[d]

      tensor_coords = [full_coords[m_axis] for m_axis in major_to_minor]

      global_idx = 0
      stride = 1
      for val, size in zip(reversed(tensor_coords), reversed(phys_mesh)):
        global_idx += val * stride
        stride *= size
      indices.append((j, global_idx))
    return indices


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
  # Multi-tag transfers: one dict per requested tag (pool_indices,
  # dst_device_block_ids, expected_pushes, dst_expected_extent_bytes,
  # order_rank), scoping schedule entries via their 11th tuple element.
  # Empty for single-tag plans (legacy scalar fields apply).
  pool_groups: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  # Byte-span plans: the expected destination coverage end per registered
  # destination block, in dst_device_block_ids order. Empty on legacy
  # token plans; non-empty selects the byte-level receiver
  # validation with no token arithmetic.
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


def _coerce_pool_spec_proto(pool: Any) -> Any:
  """Returns an owned PoolSpecProto from a proto, mapping, or dataclass."""
  result = raiden_service_pb2.PoolSpecProto()
  if isinstance(pool, raiden_service_pb2.PoolSpecProto):
    result.CopyFrom(pool)
    return result

  def value(name: str, default: Any = None) -> Any:
    if isinstance(pool, typing.Mapping):
      return pool.get(name, default)
    return getattr(pool, name, default)

  result.tag = str(value("tag", ""))
  result.storage_index = int(value("storage_index", 0))
  result.base_offset_bytes = int(value("base_offset_bytes", 0))
  result.block_stride_bytes = int(value("block_stride_bytes", 0))
  result.num_blocks = int(value("num_blocks", 0))
  result.dtype_tag = str(value("dtype_tag", ""))
  for region in value("regions", ()):
    if isinstance(region, typing.Mapping):
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


def create_server_socket(port: int) -> socket.socket:
  """Creates an IPv6 socket (supporting IPv4 dual-stack on Linux) or falls back to IPv4."""
  try:
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", port))
    sock.listen(128)
    return sock
  except Exception:  # pylint: disable=broad-except
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(128)
    return sock


def connect_socket(
    address_str: str,
    timeout: float = 60.0,
    resolver: Optional[NameResolver] = None,
) -> socket.socket:
  """Connects to an IPv4 or IPv6 endpoint robustly with optional coordinate name resolution."""
  start_time = time.time()

  while True:
    resolved_addr = address_str
    if resolver:
      try:
        resolved_addr = resolver.resolve(address_str)
      except Exception:  # pylint: disable=broad-except
        pass

    rindex = resolved_addr.rfind(":")
    if rindex != -1:
      host = resolved_addr[:rindex]
      try:
        port = int(resolved_addr[rindex + 1 :])
        if host.startswith("[") and host.endswith("]"):
          host = host[1:-1]

        for res in socket.getaddrinfo(
            host, port, socket.AF_UNSPEC, socket.SOCK_STREAM
        ):
          af, socktype, proto, _, sa = res
          sock = None
          try:
            sock = socket.socket(af, socktype, proto)
            sock.settimeout(min(10.0, timeout))
            sock.connect(sa)
            sock.settimeout(timeout)
            return sock
          except OSError:
            if sock:
              sock.close()
      except (ValueError, OSError):
        pass

    if time.time() - start_time > timeout:
      raise RuntimeError(
          f"Timeout ({timeout}s) failed to connect to robust endpoint"
          f" {address_str}"
      )
    time.sleep(2.0)


async def _await_grpc_future(fut: Any) -> Any:
  """Adapts a grpc.Future to an asyncio awaitable without blocking the event loop."""
  loop = asyncio.get_running_loop()
  async_fut = loop.create_future()

  def _done_callback(f):
    try:
      res = f.result()
      loop.call_soon_threadsafe(
          lambda: not async_fut.done() and async_fut.set_result(res)
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      loop.call_soon_threadsafe(
          lambda: not async_fut.done() and async_fut.set_exception(e)
      )

  fut.add_done_callback(_done_callback)
  return await async_fut


class WorkerRpcClient:
  """Distributed RPC Client connecting to Native C++ Control Daemons with Event-Driven resolution.

  Maintains an asynchronous endpoint catalog that resolves worker network
  coordinates instantaneously when participating worker tasks self-register,
  completely eliminating hardcoded active polling loops or arbitrary delays.
  Supports both raw TCP socket streams and gRPC
  WeightSynchronizationWorkerService.
  """

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
      proto_module: Optional[Any] = None,
      use_grpc: bool = False,
  ):
    """Instantiates RPC Client with an optional initial endpoint mapping.

    Args:
      endpoint_addresses: Initial catalog of known Worker RPC addresses.
      resolve_timeout: Maximum duration in seconds to wait for a pending worker
        task to self-register before raising a Timeout RuntimeError.
      name_resolver: Interface for resolving remote coordinates (e.g. BNS).
      proto_module: Optional protobuf module to use for ControlRequest/Response.
      use_grpc: If True, uses gRPC WeightSynchronizationWorkerServiceClient
        instead of raw TCP socket streams.
    """
    self._endpoints = {}
    if endpoint_addresses:
      for k, v in endpoint_addresses.items():
        self._endpoints[k] = [v] if isinstance(v, str) else list(v)
    self._pending_endpoints: dict[RaidenId, asyncio.Future[list[str]]] = {}
    self._resolve_timeout = resolve_timeout
    self._name_resolver = name_resolver
    self._proto_module = proto_module or raiden_service_pb2
    self._use_grpc = use_grpc
    self._grpc_clients: dict[
        str,
        weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient,
    ] = {}
    self._grpc_lock = threading.Lock()

  @property
  def name_resolver(self) -> Optional[NameResolver]:
    return self._name_resolver

  @property
  def use_grpc(self) -> bool:
    return self._use_grpc

  def get_grpc_client(
      self, addr: str
  ) -> (
      weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient
  ):
    """Returns or creates a cached WeightSynchronizationWorkerServiceClient for addr."""
    if not self._use_grpc:
      raise ValueError("WorkerRpcClient is not configured to use gRPC")
    resolved_addr = addr
    if self._name_resolver:
      try:
        resolved_addr = self._name_resolver.resolve(addr)
      except Exception:  # pylint: disable=broad-except
        pass
    with self._grpc_lock:
      client = self._grpc_clients.get(resolved_addr)
      if client is None:
        client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
            target=resolved_addr
        )
        self._grpc_clients[resolved_addr] = client
      return client

  def register_worker_endpoint(
      self, worker_name: RaidenId, rpc_address: str
  ) -> None:
    """Registers remote Worker Control-Plane RPC TCP listener address.

    If any active coroutine is currently suspended awaiting this specific worker
    coordinate, its asyncio Future is immediately resolved.

    Args:
      worker_name: Participating worker RaidenId coordinate.
      rpc_address: TCP server address in 'IP:Port' or Google BNS format.
    """
    if worker_name not in self._endpoints:
      self._endpoints[worker_name] = []
    if rpc_address not in self._endpoints[worker_name]:
      self._endpoints[worker_name].append(rpc_address)
    future = self._pending_endpoints.pop(worker_name, None)
    if future and not future.done():
      future.set_result(self._endpoints[worker_name])

  def unregister_worker_endpoint(self, worker_name: RaidenId) -> None:
    """Removes a stale worker endpoint during work-unit replacement."""
    self._endpoints.pop(worker_name, None)

  async def _resolve_endpoint(self, target_id: RaidenId) -> str:
    addrs = await self._resolve_endpoints(target_id)
    return addrs[0] if addrs else ""

  async def _resolve_endpoints(self, target_id: RaidenId) -> list[str]:
    """Resolves worker RPC addresses asynchronously or suspends execution until registered.

    Args:
      target_id: Target worker RaidenId coordinate to resolve.

    Returns:
      List of resolved remote RPC TCP server address strings.

    Raises:
      RuntimeError: If the remote endpoint fails to self-register within
        `resolve_timeout`.
    """
    addrs = self._endpoints.get(target_id)
    if addrs:
      return list(addrs)

    future = self._pending_endpoints.setdefault(target_id, asyncio.Future())
    try:
      res = await asyncio.wait_for(future, timeout=self._resolve_timeout)
      return list(res)
    except asyncio.TimeoutError as e:
      raise RuntimeError(
          f"Timeout ({self._resolve_timeout}s) waiting for remote RPC"
          f" endpoint {target_id} to self-register"
      ) from e

  async def _send_rpc(
      self, addr: str, payload: bytes, timeout: float = 600.0
  ) -> bytes:
    """Connects to remote address, sends payload, and returns the response bytes."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, self._send_rpc_sync, addr, payload, timeout
    )

  def _send_rpc_sync(
      self, addr: str, payload: bytes, timeout: float = 600.0
  ) -> bytes:
    """Connects synchronously, sends payload, and returns response bytes."""
    sock = connect_socket(addr, timeout=timeout, resolver=self._name_resolver)
    try:
      sock.sendall(len(payload).to_bytes(4, "big") + payload)

      resp_len_bytes = b""
      while len(resp_len_bytes) < 4:
        chunk = sock.recv(4 - len(resp_len_bytes))
        if not chunk:
          raise RuntimeError(
              "Remote servicer closed connection while reading response length"
          )
        resp_len_bytes += chunk
      resp_len = int.from_bytes(resp_len_bytes, "big")

      resp_bytes = b""
      while len(resp_bytes) < resp_len:
        chunk = sock.recv(resp_len - len(resp_bytes))
        if not chunk:
          raise RuntimeError(
              "Remote servicer closed connection while reading response data"
          )
        resp_bytes += chunk

      return resp_bytes
    finally:
      sock.close()

  async def _send_control_request(
      self, addr: str, req: Any, timeout: float = 600.0
  ) -> Any:
    """Sends a ControlRequest via gRPC or raw TCP socket, verifying success."""
    if self._use_grpc:
      client = self.get_grpc_client(addr)
      fut = client.handle_control(req, timeout=timeout)
      resp = await _await_grpc_future(fut)
    else:
      resp_bytes = await self._send_rpc(
          addr, req.SerializeToString(), timeout=timeout
      )
      resp = self._proto_module.ControlResponse()
      resp.ParseFromString(resp_bytes)

    if not resp.success:
      raise RuntimeError(
          f"Raiden remote native execution failed: {resp.message}"
      )
    return resp

  async def start_transfer(
      self,
      target_id: RaidenId,
      transfer_plan: TransferPlan,
      address: Optional[str] = None,
  ) -> None:
    """Connects to remote Worker servicer and dispatches collective transfer commands.

    Args:
      target_id: Target participating worker RaidenId.
      transfer_plan: Distributed transfer execution plan mapping source and
        destination topology.
      address: Explicit worker control-plane address. When given, the local
        endpoint registry is bypassed — a remote planning controller learned the
        address from the worker's registered metadata rather than from local
        self-registration.

    Raises:
      RuntimeError: If remote servicer socket connection fails, or if remote
        native execution reports failure status.
    """
    try:
      req = self._build_start_transfer_request(target_id, transfer_plan)
      if req is None:
        return
    except NotImplementedError:
      return
    addrs = [address] if address else await self._resolve_endpoints(target_id)
    await asyncio.gather(
        *[self._send_control_request(addr, req) for addr in addrs]
    )

  def _raiden_id_to_proto(self, unit: RaidenId) -> Any:
    return self._proto_module.RaidenIdProto(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

  def _build_start_transfer_request(
      self, target_id: RaidenId, transfer_plan: TransferPlan
  ) -> Optional[Any]:
    """Constructs domain-specific Protobuf ControlRequest for collective transfer kickoff."""
    if (
        target_id not in transfer_plan.src_units
        and target_id not in transfer_plan.dst_units
    ):
      return None

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

    is_sender = target_id in transfer_plan.src_units
    start_req = self._proto_module.StartTransferRequest(
        src_units=[
            self._raiden_id_to_proto(u) for u in transfer_plan.src_units
        ],
        dst_units=[
            self._raiden_id_to_proto(u) for u in transfer_plan.dst_units
        ],
        uuid=transfer_plan.uuid,
        is_sender=is_sender,
        dst_mem_type=int(transfer_plan.dst_mem_type),
        use_block_chunks=transfer_plan.use_block_chunks,
        expected_block_count=transfer_plan.dst_expected_block_counts.get(
            target_id, transfer_plan.expected_block_count
        ),
        req_id=transfer_plan.req_id,
        transfer_pool_indices=transfer_plan.transfer_pool_indices,
        pool_dtype_tags=transfer_plan.pool_dtype_tags,
        parallelism=transfer_plan.parallelism,
        skip_d2h=transfer_plan.skip_d2h,
    )
    for layer_idx, skip in transfer_plan.skip_tiling.items():
      start_req.skip_tiling[layer_idx] = skip

    layer_counts = transfer_plan.dst_expected_layer_chunk_counts.get(
        target_id, transfer_plan.expected_layer_chunk_counts
    )
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

    if transfer_plan.shard_push_schedules:
      if target_id in transfer_plan.dst_units:
        # Receiver path: send FILTERED plan, only containing entries for this
        # receiver
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
            for entry_tuple in schedule:
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
              ) = entry_tuple
              layer_idx = extra[0] if extra else 0
              pool_group = extra[1] if len(extra) > 1 else 0
              if dst_peer in target_endpoints:
                entry_proto = schedule_proto.entries.add()
                entry_proto.dst_peer = dst_peer
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
        # Sender path: only send local schedule
        push_schedules = transfer_plan.shard_push_schedules.get(target_id)
        if push_schedules:
          for shard_idx, entries in push_schedules.items():
            schedule_proto = self._proto_module.ShardPushScheduleProto()
            for entry_tuple in entries:
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
              ) = entry_tuple
              layer_idx = extra[0] if extra else 0
              pool_group = extra[1] if len(extra) > 1 else 0
              entry_proto = schedule_proto.entries.add()
              entry_proto.dst_peer = dst_peer
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
            start_req.shard_push_schedules[shard_idx].CopyFrom(schedule_proto)

    req.start_transfer_request.CopyFrom(start_req)
    return req

  def _encode_start_transfer(
      self, target_id: RaidenId, transfer_plan: TransferPlan
  ) -> Optional[bytes]:
    """Serializes domain-specific binary Protobuf command for collective transfer kickoff."""
    req = self._build_start_transfer_request(target_id, transfer_plan)
    return req.SerializeToString() if req is not None else None

  def _verify_response(self, resp_bytes: bytes) -> None:
    """Validates demarshaled remote response bytes returned from C++ workers."""
    resp = self._proto_module.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(
          f"Raiden remote native execution failed: {resp.message}"
      )

  def get_worker_endpoints(self) -> dict[RaidenId, str]:
    """Returns active read-only snapshot of known registered Worker RPC endpoints."""
    return {k: v[0] for k, v in self._endpoints.items() if v}

  def get_registered_endpoints(self, worker_name: RaidenId) -> list[str]:
    """Returns list of registered RPC endpoints for the given worker."""
    return list(self._endpoints.get(worker_name, []))

  async def shutdown_workers(self, timeout: float = 10.0) -> None:
    """Dispatches remote shutdown signaling payloads to all registered worker daemons."""
    all_addrs = set()
    for addrs in self._endpoints.values():
      all_addrs.update(addrs)
    if not all_addrs:
      return

    req = self._build_shutdown_request()
    await asyncio.gather(
        *[
            self._send_control_request(addr, req, timeout=timeout)
            for addr in all_addrs
        ],
        return_exceptions=True,
    )

  def _build_shutdown_request(self) -> Any:
    """Constructs domain-specific binary command for remote shutdown signaling."""
    return self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_SHUTDOWN
    )

  def _encode_shutdown(self) -> bytes:
    """Serializes domain-specific binary command for remote shutdown signaling."""
    req = self._build_shutdown_request()
    return req.SerializeToString()

  async def query_metadata(
      self, addr: str, timeout: float = 600.0
  ) -> list[Any]:
    """Queries metadata from a remote worker endpoint."""
    req = self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_GET_METADATA
    )
    resp = await self._send_control_request(addr, req, timeout=timeout)
    return list(resp.get_metadata_response.metadata)

  def close(self) -> None:
    """Closes all cached gRPC client channels."""
    with self._grpc_lock:
      for client in self._grpc_clients.values():
        client.close()
      self._grpc_clients.clear()

  def __enter__(self) -> "WorkerRpcClient":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()


class WeightSyncWorkerRpcClient(WorkerRpcClient):
  """Concrete domain subclass for state-of-the-art Weight Synchronizer Protobuf serialization."""

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
  ):
    super().__init__(
        endpoint_addresses=endpoint_addresses,
        resolve_timeout=resolve_timeout,
        name_resolver=name_resolver,
        proto_module=raiden_service_pb2,
    )


class RaidenFuture:
  """Future representing an asynchronous transfer execution."""

  session_id: int

  def __init__(self, session_id: int = 0, transfer_task=None):
    self.session_id = session_id
    self._transfer_task = transfer_task
    self._completed_event = threading.Event()
    self._completed = False
    self._exception = None
    self._lock = threading.Lock()
    self._started = False

  def try_start(self) -> bool:
    """Attempts to mark the future as started.

    Returns True if this call successfully started it (first time).
    Returns False if it was already started.
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
        self._completed = True
        self._completed_event.set()
    else:
      self._completed = True
      self._completed_event.set()

  def wait_threadsafe(self, timeout=None) -> None:
    """Blocks the calling thread until the transfer is complete."""
    self._completed_event.wait(timeout)
    if self._exception:
      raise self._exception

  def done(self) -> bool:
    """Returns True if the transfer operation has completed."""
    return self._completed

  def exception(self) -> Optional[Exception]:
    """Returns the exception raised by the transfer operation, if any."""
    return self._exception


def _proto_to_nd_slice(proto_slice: Any) -> list[tuple[int, int]]:
  """Converts an NDSliceProto message to a Python list of (start, end) tuples."""
  return [(dim.start, dim.end) for dim in proto_slice.dimensions]


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


def intersect_nd_slices(
    slice1: list[tuple[int, int]], slice2: list[tuple[int, int]]
) -> Optional[list[tuple[int, int]]]:
  """Computes the precise N-dimensional intersection bounding box between two multi-dimensional slices.

  Each slice is represented as a list of coordinate bounds (start, end) for
  each dimension.

  Args:
    slice1: First N-dimensional slice bounding box.
    slice2: Second N-dimensional slice bounding box.

  Returns:
    A list of (start, end) coordinate bounds representing the intersecting
    subgrid, or None if the slices do not overlap in any dimension.
  """
  result = []
  for (s1, e1), (s2, e2) in zip(slice1, slice2):
    start = max(s1, s2)
    end = min(e1, e2)
    if start >= end:
      return None
    result.append((start, end))
  return result


def generate_strided_copy_chunks(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    itemsize: int,
) -> list[tuple[int, int, int, int, int, int]]:
  """Translates an N-dimensional grid intersection into strided memory copy chunks.

  Instead of returning flat 1D chunks, this function groups contiguous dimension
  runs and returns strided chunk descriptors that enable hardware-accelerated
  2D/3D
  transfers without host-side scatter/gather loops.

  Args:
    src_shard_slice: Bounding box slice of the source shard across all N
      dimensions.
    dst_shard_slice: Bounding box slice of the destination shard across all N
      dimensions.
    intersection_slice: The overlapping region between source and destination
      shards.
    itemsize: Byte size of a single element (e.g., 4 for float32, 2 for
      bfloat16).

  Returns:
    A list of strided chunk descriptors where each tuple contains:
      (src_offset, dst_offset, size_bytes, src_stride_bytes, dst_stride_bytes,
      count)
  """
  rank = len(src_shard_slice)
  if rank == 0:
    return [(0, 0, itemsize, 0, 0, 1)]
  if rank == 1:
    s_s, _ = src_shard_slice[0]
    d_s, _ = dst_shard_slice[0]
    i_s, i_e = intersection_slice[0]
    size = (i_e - i_s) * itemsize
    return [((i_s - s_s) * itemsize, (i_s - d_s) * itemsize, size, 0, 0, 1)]
  src_shape = [e - s for s, e in src_shard_slice]
  dst_shape = [e - s for s, e in dst_shard_slice]
  int_shape = [e - s for s, e in intersection_slice]

  # Calculate how many inner dimensions can be merged into a contiguous chunk
  split_dim = -1
  for d in range(rank - 1, -1, -1):
    dim_size = int_shape[d]
    src_full = dim_size == src_shape[d]
    dst_full = dim_size == dst_shape[d]
    if not (src_full and dst_full):
      split_dim = d
      break

  if split_dim != -1:
    contiguous_elements = math.prod(int_shape[max(1, split_dim) :])
    stride_dim = max(0, split_dim - 1)
  else:
    contiguous_elements = math.prod(int_shape)
    stride_dim = -1

  contiguous_bytes = contiguous_elements * itemsize

  src_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1]

  dst_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1]

  if stride_dim >= 0:
    count = int_shape[stride_dim]
    src_stride = src_strides[stride_dim] * itemsize
    dst_stride = dst_strides[stride_dim] * itemsize
    outer_shape = int_shape[:stride_dim]
  else:
    count = 1
    src_stride = 0
    dst_stride = 0
    outer_shape = []

  num_outer_elements = math.prod(outer_shape) if outer_shape else 1

  src_local_int_slice = [
      (int_s - src_s, int_e - src_s)
      for (src_s, _), (int_s, int_e) in zip(src_shard_slice, intersection_slice)
  ]
  dst_local_int_slice = [
      (int_s - dst_s, int_e - dst_s)
      for (dst_s, _), (int_s, int_e) in zip(dst_shard_slice, intersection_slice)
  ]

  chunks = []
  for i in range(num_outer_elements):
    multi_index = []
    temp = i
    for dim_size in reversed(outer_shape):
      multi_index.append(temp % dim_size)
      temp //= dim_size
    multi_index.reverse()

    src_offset_items = 0
    dst_offset_items = 0

    # Calculate offset for outer dimensions
    for d in range(len(outer_shape)):
      src_idx = src_local_int_slice[d][0] + multi_index[d]
      src_offset_items += src_idx * src_strides[d]

      dst_idx = dst_local_int_slice[d][0] + multi_index[d]
      dst_offset_items += dst_idx * dst_strides[d]

    # For merged dimensions (and the stride dim), we use the start of the
    # intersection as the base offset for this chunk.
    start_d = len(outer_shape)
    for d in range(start_d, rank):
      src_offset_items += src_local_int_slice[d][0] * src_strides[d]
      dst_offset_items += dst_local_int_slice[d][0] * dst_strides[d]

    chunks.append((
        src_offset_items * itemsize,
        dst_offset_items * itemsize,
        contiguous_bytes,
        src_stride,
        dst_stride,
        count,
    ))

  return chunks


def is_nd_slice_tile_aligned(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    tile_shape: tuple[int, int] = (8, 128),
) -> bool:
  """Checks if slices and their intersection align with hardware tile boundaries."""
  rank = len(src_shard_slice)
  if rank < 2:
    return False
  t_row, t_col = tile_shape
  s_row_s, s_row_e = src_shard_slice[-2]
  s_col_s, s_col_e = src_shard_slice[-1]
  d_row_s, d_row_e = dst_shard_slice[-2]
  d_col_s, d_col_e = dst_shard_slice[-1]
  i_row_s, i_row_e = intersection_slice[-2]
  i_col_s, i_col_e = intersection_slice[-1]

  if (i_row_s - s_row_s) % t_row != 0 or (i_row_s - d_row_s) % t_row != 0:
    return False
  if (i_col_s - s_col_s) % t_col != 0 or (i_col_s - d_col_s) % t_col != 0:
    return False
  if (i_row_e - i_row_s) % t_row != 0:
    return False
  if (i_col_e - i_col_s) % t_col != 0:
    return False
  if (s_col_e - s_col_s) % t_col != 0 or (d_col_e - d_col_s) % t_col != 0:
    return False
  if (s_row_e - s_row_s) % t_row != 0 or (d_row_e - d_row_s) % t_row != 0:
    return False
  return True


def generate_strided_copy_chunks_tile_aware(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    itemsize: int,
    tile_shape: tuple[int, int] = (8, 128),
) -> list[tuple[int, int, int, int, int, int]]:
  """Translates an N-dimensional grid intersection into tile-aware physical strided copy chunks."""
  rank = len(src_shard_slice)
  if rank == 0:
    return [(0, 0, itemsize, 0, 0, 1)]
  if rank == 1:
    s_s, _ = src_shard_slice[0]
    d_s, _ = dst_shard_slice[0]
    i_s, i_e = intersection_slice[0]
    size = (i_e - i_s) * itemsize
    return [((i_s - s_s) * itemsize, (i_s - d_s) * itemsize, size, 0, 0, 1)]

  t_row, _ = tile_shape
  s_row_s, s_row_e = src_shard_slice[-2]
  s_col_s, s_col_e = src_shard_slice[-1]
  d_row_s, d_row_e = dst_shard_slice[-2]
  d_col_s, d_col_e = dst_shard_slice[-1]
  i_row_s, i_row_e = intersection_slice[-2]
  i_col_s, i_col_e = intersection_slice[-1]

  w_src = s_col_e - s_col_s
  w_dst = d_col_e - d_col_s
  w_int = i_col_e - i_col_s
  h_int = i_row_e - i_row_s

  local_src_row = i_row_s - s_row_s
  local_src_col = i_col_s - s_col_s
  local_dst_row = i_row_s - d_row_s
  local_dst_col = i_col_s - d_col_s

  # Physical tile parameters
  size_bytes = w_int * t_row * itemsize
  src_stride = w_src * t_row * itemsize
  dst_stride = w_dst * t_row * itemsize
  count = h_int // t_row

  src_offset = (
      local_src_row * w_src * itemsize + local_src_col * t_row * itemsize
  )
  dst_offset = (
      local_dst_row * w_dst * itemsize + local_dst_col * t_row * itemsize
  )

  if rank > 2:
    num_outer_dims = rank - 2
    outer_shape = [
        intersection_slice[d][1] - intersection_slice[d][0]
        for d in range(num_outer_dims)
    ]
    src_outer_strides = [1] * num_outer_dims
    src_outer_strides[-1] = (s_row_e - s_row_s) * (s_col_e - s_col_s)
    for d in range(num_outer_dims - 2, -1, -1):
      dim_len = src_shard_slice[d + 1][1] - src_shard_slice[d + 1][0]
      src_outer_strides[d] = src_outer_strides[d + 1] * dim_len

    dst_outer_strides = [1] * num_outer_dims
    dst_outer_strides[-1] = (d_row_e - d_row_s) * (d_col_e - d_col_s)
    for d in range(num_outer_dims - 2, -1, -1):
      dim_len = dst_shard_slice[d + 1][1] - dst_shard_slice[d + 1][0]
      dst_outer_strides[d] = dst_outer_strides[d + 1] * dim_len

    src_local_outer_start = [
        intersection_slice[d][0] - src_shard_slice[d][0]
        for d in range(num_outer_dims)
    ]
    dst_local_outer_start = [
        intersection_slice[d][0] - dst_shard_slice[d][0]
        for d in range(num_outer_dims)
    ]

    num_outer = math.prod(outer_shape) if outer_shape else 1
    if (
        (count == 1 or (size_bytes == src_stride and size_bytes == dst_stride))
        and num_outer_dims == 1
    ):
      inner_size = size_bytes * count
      s_stride_outer = src_outer_strides[-1] * itemsize
      d_stride_outer = dst_outer_strides[-1] * itemsize
      base_src = src_offset + src_local_outer_start[0] * s_stride_outer
      base_dst = dst_offset + dst_local_outer_start[0] * d_stride_outer
      if inner_size == s_stride_outer and inner_size == d_stride_outer:
        return [(base_src, base_dst, inner_size * num_outer, 0, 0, 1)]
      elif inner_size == d_stride_outer:
        return [(
            base_src,
            base_dst,
            inner_size,
            s_stride_outer,
            d_stride_outer,
            num_outer,
        )]

    chunks = []
    for i in range(num_outer):
      multi_idx = []
      temp = i
      for dim_size in reversed(outer_shape):
        multi_idx.append(temp % dim_size)
        temp //= dim_size
      multi_idx.reverse()

      outer_src_bytes = sum(
          (src_local_outer_start[d] + multi_idx[d])
          * src_outer_strides[d]
          * itemsize
          for d in range(num_outer_dims)
      )
      outer_dst_bytes = sum(
          (dst_local_outer_start[d] + multi_idx[d])
          * dst_outer_strides[d]
          * itemsize
          for d in range(num_outer_dims)
      )

      curr_src_offset = src_offset + outer_src_bytes
      curr_dst_offset = dst_offset + outer_dst_bytes
      chunks.append((
          curr_src_offset,
          curr_dst_offset,
          size_bytes,
          src_stride,
          dst_stride,
          count,
      ))
    return chunks

  if size_bytes == src_stride and size_bytes == dst_stride:
    return [(src_offset, dst_offset, size_bytes * count, 0, 0, 1)]

  return [(src_offset, dst_offset, size_bytes, src_stride, dst_stride, count)]


class RaidenController:
  """High-level transfer controller managing active transfers and generating transfer plans."""

  def __init__(
      self,
      port: int,
      worker_rpc_client: Optional[WorkerRpcClient] = None,
      request_registry_ttl_s: float = 600.0,
      broadcast_k: Optional[int] = None,
      enable_plan_cache: bool = True,
      use_grpc: bool = False,
  ):
    """Initializes the RaidenController.

    Args:
      port: Port number the controller service runs on.
      worker_rpc_client: Optional worker RPC client facade for dispatching RPCs.
      request_registry_ttl_s: TTL in seconds for request registry entries.
      broadcast_k: Fan-out factor K for tree-based broadcast transfers.
      enable_plan_cache: Whether to cache transfer planning and resharding
        schedules across transfer invocations with identical topologies.
      use_grpc: Whether to default to WorkerRpcClient in gRPC mode if
        worker_rpc_client is not specified.
    """
    self.port = port
    self.broadcast_k = (
        broadcast_k
        if broadcast_k is not None
        else int(os.environ.get("RAIDEN_BROADCAST_K", "64"))
    )
    self.enable_plan_cache = enable_plan_cache
    self._plan_cache: dict[Any, _CachedTransferSchedule] = {}
    self._active_transfers: dict[str, TransferPlan] = {}
    self._active_tasks: dict[str, RaidenFuture] = {}
    self._task_units: dict[str, list[RaidenId]] = {}
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
    # The C++ reshard store owns the request-block registry lifecycle end to
    # end; the Python controller does not allocate or expire these entries.
    if request_registry_ttl_s <= 0:
      raise ValueError("request_registry_ttl_s must be positive")
    self._request_registry_ttl_s = request_registry_ttl_s
    use_grpc_effective = use_grpc or (
        os.environ.get("RAIDEN_WEIGHT_SYNC_USE_GRPC", "").lower()
        in ("1", "true")
    )
    self.worker_rpc_client = worker_rpc_client or WorkerRpcClient(
        use_grpc=use_grpc_effective
    )
    self._registered_variables = {}

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      mesh_shape: Optional[typing.Sequence[int]] = None,
      layout: Optional[typing.Sequence[int]] = None,
      global_shape: Optional[typing.Sequence[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[typing.Sequence[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
      variables: Optional[typing.Sequence[Any]] = None,
      mesh_axes: Optional[typing.Sequence[str]] = None,
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
    # Duplicate endpoints are legal: a single process serving several local
    # devices (JAX/Pathways) shares one transfer port across its shards, so
    # the shard list carries the shard COUNT while the addresses coincide.
    # Delivery is per-unit (one RPC with the shard-indexed plan), so repeats
    # never double-send. Pool/state reshard planning enforces its stricter
    # one-endpoint-per-unit contract at plan time.

    with self._lock:
      if unit in self._registered_shards:
        tasks_to_clear = []
        for req_id, units in self._task_units.items():
          if unit in units:
            tasks_to_clear.append(req_id)
        for req_id in tasks_to_clear:
          self._active_tasks.pop(req_id, None)
          self._task_units.pop(req_id, None)

      self._registered_shards[unit] = list(shards)
      # Registration is replacement, not a patch: stale optional metadata
      # must disappear when a unit restarts with a different payload.
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
      if mesh_shape is not None:
        self._registered_mesh_shapes[unit] = list(mesh_shape)
      if mesh_axes is not None:
        self._registered_mesh_axes[unit] = list(mesh_axes)
      if layout is not None:
        self._registered_layouts[unit] = list(layout)
      if global_shape is not None:
        self._registered_global_shapes[unit] = list(global_shape)
      if itemsize is not None:
        self._registered_itemsizes[unit] = itemsize
      if has_reshard_metadata:
        self._registered_pool_manifests[unit] = normalized_pools
        self._registered_layout_fingerprints[unit] = layout_fingerprint
        self._registered_page_tokens[unit] = page_tokens
        self._registered_transfer_parallelism[unit] = transfer_parallelism
        self._registered_transfer_ranks[unit] = transfer_rank
      if variables is not None:
        self._registered_variables[unit] = list(variables)
      if control_plane_rpc_address and hasattr(
          self.worker_rpc_client, "register_worker_endpoint"
      ):
        self.worker_rpc_client.register_worker_endpoint(
            unit, control_plane_rpc_address
        )
      elif hasattr(self.worker_rpc_client, "unregister_worker_endpoint"):
        self.worker_rpc_client.unregister_worker_endpoint(unit)
      self._plan_cache.clear()

  def clear_plan_cache(self) -> None:
    """Clears all cached transfer schedules."""
    with self._lock:
      self._plan_cache.clear()

  def get_plan_cache_size(self) -> int:
    """Returns the number of cached transfer schedules."""
    with self._lock:
      return len(self._plan_cache)

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
        control_plane_rpc_address=(
            self.worker_rpc_client.get_worker_endpoints().get(unit, "")
        ),
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
      reg_req.variables.extend(self._registered_variables[unit])
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
      shards = self._registered_shards.get(unit)
      if not shards:
        raise ValueError(f"Work unit is not registered: {unit}")
      return list(shards)

  async def _query_remote_metadata(self, addr: str) -> list[Any]:
    """Queries metadata from a remote controller or worker endpoint."""
    return await self.worker_rpc_client.query_metadata(addr)

  def _get_local_metadata(self, units: list[RaidenId]) -> list[Any]:
    with self._lock:
      missing = [unit for unit in units if unit not in self._registered_shards]
      if missing:
        raise ValueError(f"Work units are not registered: {missing}")
      return [self._metadata_proto_locked(unit) for unit in units]

  @staticmethod
  def _metadata_by_unit(
      metadata: typing.Sequence[Any], units: typing.Sequence[RaidenId]
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
      fanout_k: int,
      req_id: str,
      dst_mem_type: int,
      dst_controller_address: Optional[str],
      src_controller_address: Optional[str] = None,
  ) -> None:
    """Executes a pipelined tree broadcast for a group of variables.

    This method builds and executes a broadcast tree to distribute a group of
    variables (represented by keys_and_targets) from a single source unit to
    multiple destination units. It uses a greedy tree-building algorithm where
    nodes that have received the data (destination units) are promoted to act as
    new source nodes (available_sources) for subsequent hops, limited by the
    maximum fanout (fanout_k).

    The broadcast is pipelined such that the transfer for all variables in the
    group targeting a specific destination unit is bundled into a single
    TransferPlan (sub-schedule) per hop, minimizing coordination overhead.

    Args:
      keys_and_targets: A list of tuples, where each tuple contains: - key: A
        tuple describing the source slice metadata: (src_unit, shard_idx,
        src_block_id, src_block_offset, size, src_stride, count, layer_idx,
        pool_group) - targets: A list of tuples, where each tuple describes a
        target device: (dst_unit, dst_peer, dst_shard_idx, dst_block_id,
        dst_block_offset, dst_stride)
      final_plan: The parent TransferPlan containing global configurations like
        worker RPC/data addresses, skip_d2h, skip_tiling, etc.
      fanout_k: The maximum number of simultaneous active pushes allowed per
        source node (the fanout factor of the broadcast tree).
      req_id: The base request ID for this transfer session.
      dst_mem_type: The destination memory type (e.g. HBM, DRAM).
      dst_controller_address: Optional BNS address of the destination-side
        controller (used for remote coordination).
      src_controller_address: Optional BNS address of the source-side
        controller.
    """
    # Sort targets for each key to ensure consistent indexing
    keys_and_sorted_targets = []
    for key, targets in keys_and_targets:
      sorted_t = sorted(targets, key=lambda t: (t[1], t[2]))
      keys_and_sorted_targets.append((key, sorted_t))

    ref_key, ref_targets = keys_and_sorted_targets[0]
    (
        src_unit,
        shard_idx,
        _,  # src_block_id
        _,  # src_block_offset
        _,  # size
        _,  # src_stride
        _,  # count
        _,  # layer_idx
        _,  # pool_group
    ) = ref_key

    available_sources = [src_unit]
    node_slice_offsets = {
        src_unit: {
            key: (shard_idx, key[2], key[3], key[5])
            for key, _ in keys_and_sorted_targets
        }
    }

    pending_indices = list(range(len(ref_targets)))
    active_pushes = {u: 0 for u in [src_unit] + [t[0] for t in ref_targets]}
    transfers_in_progress = {}

    while pending_indices or transfers_in_progress:
      # 1. Greedy assignment step
      scheduled_any = False
      for s in list(available_sources):
        while active_pushes[s] < fanout_k and pending_indices:
          # Pick the first pending target index
          first_idx = pending_indices[0]
          first_target = ref_targets[first_idx]
          dst_unit = first_target[0]

          # Find all pending indices that target the same dst_unit
          dst_indices = [
              idx for idx in pending_indices if ref_targets[idx][0] == dst_unit
          ]
          # Remove them from pending_indices
          pending_indices = [
              idx for idx in pending_indices if idx not in dst_indices
          ]

          active_pushes[s] += 1
          scheduled_any = True

          ref_offset = node_slice_offsets[s][ref_key]
          s_shard_idx = ref_offset[0]

          # Build sub-schedule containing entries for all keys in the group
          # and all target devices on dst_unit
          sub_schedule = {s: {s_shard_idx: []}}
          hop_expected_block_count = 0
          for key, k_targets in keys_and_sorted_targets:
            _, k_s_block_id, k_s_block_offset, k_s_stride = node_slice_offsets[
                s
            ][key]
            k_size = key[4]
            k_count = key[6]
            k_layer_idx = key[7]
            k_pool_group = key[8]

            for idx in dst_indices:
              k_target = k_targets[idx]
              (
                  _,
                  k_dst_peer,
                  k_dst_shard_idx,
                  k_dst_block_id,
                  k_dst_block_offset,
                  k_dst_stride,
              ) = k_target

              entry = (
                  k_dst_peer,
                  k_dst_shard_idx,
                  k_dst_block_offset,
                  k_s_block_offset,
                  k_size,
                  k_s_block_id,
                  k_dst_block_id,
                  k_s_stride,
                  k_dst_stride,
                  k_count,
                  k_layer_idx,
                  k_pool_group,
              )
              sub_schedule[s][s_shard_idx].append(entry)

              is_contiguous = (k_count == 1) or (
                  k_s_stride == k_size and k_dst_stride == k_size
              )
              push_count = 1 if is_contiguous else k_count
              hop_expected_block_count += push_count

          hop_uuid = random.randint(1, 2**63 - 1)
          hop_req_id = f"{req_id}_{hop_uuid}"

          sub_plan = TransferPlan(
              src_units=[s],
              dst_units=[dst_unit],
              plan=None,
              shard_push_schedules=sub_schedule,
              worker_rpc_addresses=dict(final_plan.worker_rpc_addresses),
              worker_data_addresses=dict(final_plan.worker_data_addresses),
              uuid=hop_uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=True,
              is_sender=True,
              expected_block_count=hop_expected_block_count,
              req_id=hop_req_id,
              skip_d2h=final_plan.skip_d2h or (s != src_unit),
              skip_tiling=final_plan.skip_tiling,
          )

          async def _run_single_transfer(s_node, d_node, plan):
            if dst_controller_address:
              dst_facade = RaidenControllerClientFacade(
                  dst_controller_address,
                  name_resolver=self.worker_rpc_client.name_resolver,
              )
              loop = asyncio.get_running_loop()
              success = await loop.run_in_executor(
                  None,
                  functools.partial(
                      dst_facade.register_transfer_schedule,
                      [s_node],
                      [d_node],
                      plan.req_id,
                      True,
                      s_node not in self._registered_shards,
                      plan.expected_block_count,
                      plan.uuid,
                      dst_controller_address,
                      src_controller_address,
                      plan.shard_push_schedules,
                      dst_mem_type,
                      skip_d2h=plan.skip_d2h,
                  ),
              )
              if not success:
                raise RuntimeError(
                    "Failed remote prepare in slice tree broadcast"
                )
            else:
              await self.worker_rpc_client.start_transfer(d_node, plan)

            if s_node in self._registered_shards:
              await self.worker_rpc_client.start_transfer(s_node, plan)

          task = asyncio.create_task(
              _run_single_transfer(s, dst_unit, sub_plan)
          )
          transfers_in_progress[dst_unit] = (s, dst_unit, dst_indices, task)

      # 2. Wait step
      if transfers_in_progress:
        futures_to_dsts = {
            info[3]: dst for dst, info in transfers_in_progress.items()
        }
        done, _ = await asyncio.wait(
            futures_to_dsts.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        # 3. Promotion step
        for fut in done:
          if fut.exception():
            logging.error(
                "Slice transfer failed in broadcast tree: %s", fut.exception()
            )
            for info in transfers_in_progress.values():
              info[3].cancel()
            raise fut.exception()
          dst_unit = futures_to_dsts[fut]
          s, _, dst_indices, _ = transfers_in_progress.pop(dst_unit)
          active_pushes[s] -= 1
          available_sources.append(dst_unit)

          # Add to node_slice_offsets for the new source for all keys.
          # We use the first dst_index to populate the offsets for dst_unit.
          # Since all dst_indices received the same shard, any of them is fine.
          ref_idx = dst_indices[0]
          node_slice_offsets[dst_unit] = {}
          for key, k_targets in keys_and_sorted_targets:
            k_target = k_targets[ref_idx]
            (
                _,
                _,
                k_dst_shard_idx,
                k_dst_block_id,
                k_dst_block_offset,
                k_dst_stride,
            ) = k_target
            node_slice_offsets[dst_unit][key] = (
                k_dst_shard_idx,
                k_dst_block_id,
                k_dst_block_offset,
                k_dst_stride,
            )
      elif not scheduled_any:
        break

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
      transfer_pool_tags: Optional[typing.Sequence[str]] = None,
      parallelism: Optional[int] = None,
      skip_d2h: bool = False,
      dst_block_counts: Optional[typing.Sequence[int]] = None,
      skip_tiling: Optional[dict[int, bool]] = None,
      group_size: int = 1,
      use_cached_plan: bool = True,
  ) -> RaidenFuture:
    """For a requested data transfer, generates a transfer plan for the work units to carry out and start it.

    Args:
      src_units: list of source work units containing the data.
      dst_units: All destination work units that need the data.
      req_id: Unique identifier for the active transfer entity.
      src_block_ids: list of source block IDs to be transferred.
      dst_device_block_ids: list of destination device block IDs to receive the
        data. This is only needed when the destination memory type is HBM.
      dst_mem_type: The dst memory type of the data written to.
      use_block_chunks: Whether to use chunked transport.
      src_controller_address: Optional address of the source controller.
      dst_controller_address: Optional address of the destination controller.
      uuid: Optional pre-determined UUID for the transfer.
      is_sender: If True, this controller acts as the Sender Coordinator,
        querying destination metadata and triggering the active push on source
        workers. If False, this controller acts as the Destination Coordinator,
        preparing local receiver workers and setting up their expected block
        count.
      expected_block_count: The total number of physical block-pushes expected
        per destination rank (only applicable when is_sender=False).
      shard_push_schedules: Optional dictionary containing pre-computed
        schedules.
      num_tokens: Optional number of tokens for resharding.
      transfer_pool_tags: Optional sequence of tags for pool-based transfers.
      parallelism: Optional parallelism factor.
      skip_d2h: If True, skip the Device-to-Host copy.
      dst_block_counts: Optional sequence of expected block counts per
        destination.
      skip_tiling: Optional dictionary specifying which layers to skip tiling
        for.
      group_size: Number of weights to group together for synchronization
        (default 1).
      use_cached_plan: If True and enable_plan_cache is enabled on the
        controller, attempts to reuse pre-computed transfer schedules and slice
        partitioning across identical transfer configurations.

    Returns:
      A Future for the call site to wait for the transfer to complete.
    """

    # The block-granular reshard mode dispatches on exactly the arguments the
    # legacy body rejects with NotImplementedError below — provably disjoint
    # argument spaces behind one public entry point.
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

    # Select only one source unit for now.
    with self._lock:
      active_src_counts = {}
      active_dst_counts = {}
      for existing_plan in self._active_transfers.values():
        for s in existing_plan.src_units:
          active_src_counts[s.job_replica_id] = (
              active_src_counts.get(s.job_replica_id, 0) + 1
          )

        for t in existing_plan.dst_units:
          active_dst_counts[t.job_replica_id] = (
              active_dst_counts.get(t.job_replica_id, 0) + 1
          )

    selected_src = min(
        src_units, key=lambda s: active_src_counts.get(s.job_replica_id, 0)
    )

    if uuid is None:
      uuid = random.randint(1, 2**63 - 1)

    # Determine session_id and req_id synchronously
    with self._lock:
      session_id = len(self._active_transfers)
      if not req_id:
        req_id = f"req_{session_id}"

    if not use_block_chunks:
      # === OLD WORKFLOW: Fully build and store plan SYNCHRONOUSLY ===
      # Generate the default plan. Only this synchronous workflow consumes it
      # (the block-chunks branch stores plan=None and builds its schedules
      # asynchronously), and only this workflow may resolve destination
      # shards eagerly: on a source controller the destination units of a
      # cross-controller transfer are not locally registered.
      num_src = len(self._resolve_shards(selected_src))
      default_plan_dict = {}
      src_plan = [[] for _ in range(num_src)]
      for dst_unit in dst_units:
        num_dst = len(self._resolve_shards(dst_unit))
        for i in range(num_src):
          src_start = i * num_dst
          src_end = (i + 1) * num_dst
          for j in range(num_dst):
            dst_start = j * num_src
            dst_end = (j + 1) * num_src
            intersect_start = max(src_start, dst_start)
            intersect_end = min(src_end, dst_end)
            if intersect_start < intersect_end:
              local_start = intersect_start - src_start
              local_end = intersect_end - src_start
              nd_slice = [(local_start, local_end)]
              src_plan[i].append((dst_unit, j, [nd_slice]))
      default_plan_dict[selected_src] = src_plan

      rpc_addresses = {}
      if hasattr(self.worker_rpc_client, "get_worker_endpoints"):
        rpc_addresses = self.worker_rpc_client.get_worker_endpoints()
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
      # === NEW WORKFLOW: Store partial plan SYNCHRONOUSLY ===
      plan = TransferPlan(
          src_units=src_units,
          dst_units=dst_units,
          plan=None,
          shard_push_schedules=shard_push_schedules or {},
          worker_rpc_addresses=dict(
              self.worker_rpc_client.get_worker_endpoints()
          ),
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

    async def _execute_transfer() -> None:
      nonlocal skip_d2h, expected_block_count
      if use_block_chunks:
        # === NEW SYMMETRIC DECENTRALIZED WORKFLOW ===

        if not is_sender:
          # --- ROLE: DESTINATION CONTROLLER (RECEIVER COORDINATOR) ---
          # 1. Discover local destination units
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

          # 2. Build rpc_addresses for local destination workers
          rpc_addresses = self.worker_rpc_client.get_worker_endpoints()

          # 3. Construct a lightweight TransferPlan containing receiver
          # parameters
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
              req_id=req_id,
              skip_d2h=skip_d2h,
              skip_tiling=skip_tiling or {},
              parallelism=parallelism or 1,
          )

          # 4. Trigger COMMAND_START_TRANSFER (is_sender=False) on local workers
          if expected_block_count > 0:
            logging.info(
                "Triggering preparation RPCs on local destination workers: %s,"
                " expected blocks: %d",
                _format_units(local_dst_units),
                expected_block_count,
            )
            await asyncio.gather(*[
                self.worker_rpc_client.start_transfer(unit, receiver_plan)
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
          # --- ROLE: SENDER CONTROLLER (SENDER COORDINATOR) ---
          logging.info(
              "RaidenController acting as SENDER COORDINATOR (is_sender=True)"
              " for req_id %s (uuid=%s): %s -> %s",
              req_id,
              uuid,
              _format_units(src_units),
              _format_units(dst_units),
          )

          cache_key = (
              tuple(src_units),
              tuple(dst_units),
              dst_mem_type,
              skip_d2h,
              parallelism or 1,
              group_size,
              tuple(sorted(skip_tiling.items()))
              if skip_tiling is not None
              else None,
              dst_controller_address,
              src_controller_address,
          )

          cached_schedule = None
          dst_unit_counts = {}
          dst_unit_layer_counts = {}
          if (
              self.enable_plan_cache
              and use_cached_plan
              and not shard_push_schedules
          ):
            with self._lock:
              cached_schedule = self._plan_cache.get(cache_key)

          if cached_schedule is not None:
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
            local_skip_tiling = cached_schedule.local_skip_tiling
            if expected_block_count == 0:
              expected_block_count = cached_schedule.expected_block_count
            dst_unit_layer_counts = cached_schedule.dst_unit_layer_counts
            data_address_to_unit = cached_schedule.data_address_to_unit
            direct_dsts = cached_schedule.direct_dsts
            rpc_addresses = dict(self.worker_rpc_client.get_worker_endpoints())
            rpc_addresses.update(cached_schedule.rpc_addresses)
            data_addresses = cached_schedule.data_addresses
            dst_unit_counts = cached_schedule.dst_unit_counts

          else:
            # 1. Retrieve destination metadata (either from remote dst_controller
            # or local)
            dst_metadata = []
            if dst_controller_address:
              logging.info(
                  "Querying remote destination controller %s for metadata",
                  dst_controller_address,
              )
              dst_metadata = await self._query_remote_metadata(
                  dst_controller_address
              )
            else:
              logging.info("Using local registration for destination metadata")
              dst_metadata = self._get_local_metadata(dst_units)

            # 2. Compute slices or use pre-computed schedules centrally
            computed_schedules = {}
            computed_slices = {}
            data_address_to_unit = {}
            num_vars = 0
            if shard_push_schedules:
              logging.info("Using pre-computed shard_push_schedules")
              computed_schedules = shard_push_schedules
              for meta in dst_metadata:
                unit = _raiden_id_from_proto(meta.unit)
                for shard in meta.shards:
                  data_address_to_unit[shard] = unit
            else:
              is_legacy_by_unit = {}
              # Source slices (always local to sender controller)
              for unit in src_units:
                with self._lock:
                  variables = self._registered_variables.get(unit)
                if variables:
                  src_vars = variables
                  is_legacy_by_unit[unit] = False
                else:
                  with self._lock:
                    global_shape = self._registered_global_shapes.get(unit)
                    mesh_shape = self._registered_mesh_shapes.get(unit)
                    layout = self._registered_layouts.get(unit)
                    itemsize = self._registered_itemsizes.get(unit) or 4
                  if global_shape and mesh_shape and layout:
                    src_vars = [
                        _VariableMetadata(
                            name=unit.data_name,
                            shape=global_shape,
                            mesh_shape=mesh_shape,
                            layout=layout,
                            item_size=itemsize,
                            layer_idx=0,
                        )
                    ]
                  else:
                    src_vars = []
                  is_legacy_by_unit[unit] = True

                num_vars = max(num_vars, len(src_vars))
                computed_slices[unit] = {}
                for var in src_vars:
                  phys_shape, phys_mesh = to_physical(
                      var.shape, var.mesh_shape, var.layout
                  )
                  self._computed_phys_meshes[unit] = phys_mesh
                  slices = nd_slice_math.compute_nd_shard_slices(
                      phys_shape, phys_mesh
                  )
                  computed_slices[unit][var.name] = slices
                  logging.debug(
                      "Computed source slices for %s var %s: %s",
                      unit,
                      var.name,
                      nd_slice_math.format_nd_slices(slices),
                  )

              # Destination slices
              dst_vars_by_unit = {}
              for meta in dst_metadata:
                unit = _raiden_id_from_proto(meta.unit)
                for shard in meta.shards:
                  data_address_to_unit[shard] = unit
                if meta.variables:
                  dst_vars = meta.variables
                  is_legacy_by_unit[unit] = False
                else:
                  global_shape = (
                      list(meta.global_shape) if meta.global_shape else []
                  )
                  mesh_shape = list(meta.mesh_shape) if meta.mesh_shape else []
                  layout = list(meta.layout) if meta.layout else []
                  itemsize = meta.itemsize if meta.itemsize else 4
                  if global_shape and mesh_shape and layout:
                    dst_vars = [
                        _VariableMetadata(
                            name=unit.data_name,
                            shape=global_shape,
                            mesh_shape=mesh_shape,
                            layout=layout,
                            item_size=itemsize,
                            layer_idx=0,
                        )
                    ]
                  else:
                    dst_vars = []
                  is_legacy_by_unit[unit] = True

                dst_vars_by_unit[unit] = dst_vars
                computed_slices[unit] = {}
                for var in dst_vars:
                  phys_shape, phys_mesh = to_physical(
                      list(var.shape),
                      list(var.mesh_shape),
                      list(var.layout),
                  )
                  self._computed_phys_meshes[unit] = phys_mesh
                  slices = nd_slice_math.compute_nd_shard_slices(
                      phys_shape, phys_mesh
                  )
                  computed_slices[unit][var.name] = slices
                  logging.debug(
                      "Computed destination slices for %s var %s: %s",
                      unit,
                      var.name,
                      nd_slice_math.format_nd_slices(slices),
                  )

              # Compute skip_tiling if not provided
              local_skip_tiling = skip_tiling
              if local_skip_tiling is None:
                local_skip_tiling = {}
                if src_units and dst_units:
                  reference_src_unit = src_units[0]
                  with self._lock:
                    reference_src_vars = self._registered_variables.get(
                        reference_src_unit
                    )
                  if not reference_src_vars:
                    with self._lock:
                      global_shape = self._registered_global_shapes.get(
                          reference_src_unit
                      )
                      mesh_shape = self._registered_mesh_shapes.get(
                          reference_src_unit
                      )
                      layout = self._registered_layouts.get(reference_src_unit)
                      itemsize = (
                          self._registered_itemsizes.get(reference_src_unit)
                          or 4
                      )
                    if global_shape and mesh_shape and layout:
                      reference_src_vars = [
                          _VariableMetadata(
                              name=reference_src_unit.data_name,
                              shape=global_shape,
                              mesh_shape=mesh_shape,
                              layout=layout,
                              item_size=itemsize,
                              layer_idx=0,
                          )
                      ]
                    else:
                      reference_src_vars = []

                  reference_dst_unit = dst_units[0]
                  reference_dst_vars = dst_vars_by_unit.get(
                      reference_dst_unit, []
                  )

                  for src_var in reference_src_vars:
                    layer_idx = src_var.layer_idx
                    dst_var = next(
                        (
                            v
                            for v in reference_dst_vars
                            if v.layer_idx == layer_idx
                        ),
                        None,
                    )
                    if dst_var:
                      is_identical = _is_variable_spec_identical(
                          src_var, dst_var
                      )
                      s_slices = computed_slices.get(
                          reference_src_unit, {}
                      ).get(src_var.name, [])
                      d_slices = computed_slices.get(
                          reference_dst_unit, {}
                      ).get(dst_var.name, [])
                      all_aligned = True
                      for s_proto in s_slices:
                        s_sl = _proto_to_nd_slice(s_proto)
                        for d_proto in d_slices:
                          d_sl = _proto_to_nd_slice(d_proto)
                          inter = intersect_nd_slices(s_sl, d_sl)
                          if inter:
                            if not is_nd_slice_tile_aligned(
                                s_sl, d_sl, inter, tile_shape=(8, 128)
                            ):
                              all_aligned = False
                              break
                        if not all_aligned:
                          break
                      is_2d_identical = is_identical and len(src_var.shape) >= 2
                      local_skip_tiling[layer_idx] = (
                          is_2d_identical or all_aligned
                      )

              # Pre-index source slice holders to deduplicate and load-balance
              # across replicated source shards.
              src_slice_holders = {}
              for s_unit in src_units:
                with self._lock:
                  variables = self._registered_variables.get(s_unit)
                s_vars = variables if variables else []
                s_shards = self._resolve_shards(s_unit)
                with self._lock:
                  s_job_reps = {
                      u.job_replica_id
                      for u in self._registered_shards
                      if u.job_name == s_unit.job_name
                  }
                  s_phys_mesh = self._registered_mesh_shapes.get(s_unit)
                  s_mesh_axes = self._registered_mesh_axes.get(s_unit)
                num_src_hosts = max(1, len(s_job_reps))
                for s_var in s_vars:
                  s_slices_list = computed_slices.get(s_unit, {}).get(
                      s_var.name
                  )
                  if not s_slices_list:
                    continue
                  s_indices_list = _get_global_indices(
                      s_unit,
                      s_shards,
                      list(s_var.mesh_shape),
                      list(s_var.layout),
                      num_src_hosts,
                      sharding_spec=list(s_var.sharding_spec),
                      mesh_axes=s_mesh_axes,
                      physical_mesh_shape=s_phys_mesh,
                  )
                  for l_s_idx, g_s_idx in s_indices_list:
                    if g_s_idx < len(s_slices_list):
                      s_proto = s_slices_list[g_s_idx]
                      sl = tuple(_proto_to_nd_slice(s_proto))
                      k = (s_var.name, sl)
                      src_slice_holders.setdefault(k, []).append(
                          (s_unit, l_s_idx)
                      )

              # 3. Generate plan (Intersection)
              for src_unit in src_units:
                with self._lock:
                  variables = self._registered_variables.get(src_unit)
                if variables:
                  src_vars = variables
                else:
                  with self._lock:
                    global_shape = self._registered_global_shapes.get(src_unit)
                    mesh_shape = self._registered_mesh_shapes.get(src_unit)
                    layout = self._registered_layouts.get(src_unit)
                    itemsize = self._registered_itemsizes.get(src_unit) or 4
                  if global_shape and mesh_shape and layout:
                    src_vars = [
                        _VariableMetadata(
                            name=src_unit.data_name,
                            shape=global_shape,
                            mesh_shape=mesh_shape,
                            layout=layout,
                            item_size=itemsize,
                            layer_idx=0,
                        )
                    ]
                  else:
                    src_vars = []

                src_shards = self._resolve_shards(src_unit)
                unit_schedules = {}

                for src_var in src_vars:
                  itemsize = src_var.item_size
                  layer_idx = src_var.layer_idx
                  var_name = src_var.name

                  src_slices = computed_slices.get(src_unit, {}).get(var_name)
                  if not src_slices:
                    continue

                  with self._lock:
                    src_job_replicas = {
                        u.job_replica_id
                        for u in self._registered_shards
                        if u.job_name == src_unit.job_name
                    }
                    src_phys_mesh_shape = self._registered_mesh_shapes.get(
                        src_unit
                    )
                    src_mesh_axes = self._registered_mesh_axes.get(src_unit)
                  num_src_physical_hosts = max(1, len(src_job_replicas))
                  src_logical_mesh = list(src_var.mesh_shape)
                  src_layout = list(src_var.layout)

                  src_indices = _get_global_indices(
                      src_unit,
                      src_shards,
                      src_logical_mesh,
                      src_layout,
                      num_src_physical_hosts,
                      sharding_spec=list(src_var.sharding_spec),
                      mesh_axes=src_mesh_axes,
                      physical_mesh_shape=src_phys_mesh_shape,
                  )

                  for local_src_idx, global_src_idx in src_indices:
                    if global_src_idx >= len(src_slices):
                      continue

                    src_slice_proto = src_slices[global_src_idx]
                    src_slice = _proto_to_nd_slice(src_slice_proto)
                    shard_entries = unit_schedules.setdefault(local_src_idx, [])

                    for dst_unit in dst_units:
                      dst_vars = dst_vars_by_unit.get(dst_unit, [])
                      dst_var = next(
                          (v for v in dst_vars if v.name == var_name), None
                      )
                      if not dst_var:
                        continue

                      d_slices = computed_slices.get(dst_unit, {}).get(var_name)
                      if not d_slices:
                        continue

                      dst_shards = []
                      dst_phys_mesh_shape = None
                      dst_mesh_axes = None
                      for meta in dst_metadata:
                        meta_unit = _raiden_id_from_proto(meta.unit)
                        if meta_unit == dst_unit:
                          dst_shards = list(meta.shards)
                          dst_phys_mesh_shape = (
                              list(meta.mesh_shape) if meta.mesh_shape else None
                          )
                          dst_mesh_axes = (
                              list(meta.mesh_axes) if meta.mesh_axes else None
                          )
                          break
                      if not dst_shards:
                        dst_shards = ["127.0.0.1:8000"]  # fallback

                      with self._lock:
                        dst_job_replicas = {
                            m.unit.job_replica_id
                            for m in dst_metadata
                            if m.unit.job_name == dst_unit.job_name
                        }
                      num_dst_physical_hosts = max(1, len(dst_job_replicas))

                      dst_logical_mesh = list(dst_var.mesh_shape)
                      dst_layout = list(dst_var.layout)

                      dst_indices = _get_global_indices(
                          dst_unit,
                          dst_shards,
                          dst_logical_mesh,
                          dst_layout,
                          num_dst_physical_hosts,
                          sharding_spec=list(dst_var.sharding_spec),
                          mesh_axes=dst_mesh_axes,
                          physical_mesh_shape=dst_phys_mesh_shape,
                      )

                      for local_dst_idx, global_dst_idx in dst_indices:
                        if global_dst_idx >= len(d_slices):
                          continue

                        dst_slice_proto = d_slices[global_dst_idx]
                        dst_slice = _proto_to_nd_slice(dst_slice_proto)

                        dst_peer = (
                            dst_shards[local_dst_idx]
                            if local_dst_idx < len(dst_shards)
                            else dst_shards[0]
                        )

                        intersection = intersect_nd_slices(src_slice, dst_slice)
                        if intersection:
                          s_key = (var_name, tuple(src_slice))
                          candidates = src_slice_holders.get(
                              s_key, [(src_unit, local_src_idx)]
                          )
                          if len(candidates) > 1:
                            dst_global_idx = (
                                dst_units.index(dst_unit)
                                if dst_unit in dst_units
                                else 0
                            ) * max(1, len(dst_shards)) + local_dst_idx
                            chosen_src = candidates[
                                dst_global_idx % len(candidates)
                            ]
                            if (src_unit, local_src_idx) != chosen_src:
                              continue

                          is_tile_aware = (
                              local_skip_tiling.get(layer_idx, False)
                              if local_skip_tiling
                              else False
                          ) and is_nd_slice_tile_aligned(
                              src_slice,
                              dst_slice,
                              intersection,
                              tile_shape=(8, 128),
                          )
                          if is_tile_aware:
                            chunks = generate_strided_copy_chunks_tile_aware(
                                src_slice,
                                dst_slice,
                                intersection,
                                itemsize,
                                tile_shape=(8, 128),
                            )
                          else:
                            chunks = generate_strided_copy_chunks(
                                src_slice, dst_slice, intersection, itemsize
                            )
                          for (
                              src_offset,
                              dst_offset,
                              size,
                              src_stride,
                              dst_stride,
                              count,
                          ) in chunks:
                            is_legacy = is_legacy_by_unit.get(
                                src_unit, True
                            ) or is_legacy_by_unit.get(dst_unit, True)

                            if len(src_slice) > 1:
                              src_block_bytes = (
                                  math.prod([e - s for s, e in src_slice[1:]])
                                  * itemsize
                              )
                              src_block_id = src_offset // src_block_bytes
                              src_block_offset = (
                                  src_offset % src_block_bytes
                                  if is_legacy
                                  else src_offset
                              )
                            else:
                              src_block_bytes = (
                                  src_slice[0][1] - src_slice[0][0]
                              ) * itemsize
                              src_block_id = 0
                              src_block_offset = src_offset

                            if len(dst_slice) > 1:
                              dst_block_bytes = (
                                  math.prod([e - s for s, e in dst_slice[1:]])
                                  * itemsize
                              )
                              dst_block_id = dst_offset // dst_block_bytes
                              dst_block_offset = (
                                  dst_offset % dst_block_bytes
                                  if is_legacy
                                  else dst_offset
                              )
                            else:
                              dst_block_bytes = (
                                  dst_slice[0][1] - dst_slice[0][0]
                              ) * itemsize
                              dst_block_id = 0
                              dst_block_offset = dst_offset

                            shard_entries.append((
                                dst_peer,
                                local_dst_idx,
                                dst_block_offset,
                                src_block_offset,
                                size,
                                src_block_id,
                                dst_block_id,
                                src_stride,
                                dst_stride,
                                count,
                                layer_idx,
                                0,
                            ))

                if unit_schedules:
                  computed_schedules[src_unit] = unit_schedules

            # Build rpc_addresses for local source workers
            rpc_addresses = self.worker_rpc_client.get_worker_endpoints()
            # Merge destination rpc addresses from metadata
            for meta in dst_metadata:
              unit = _raiden_id_from_proto(meta.unit)
              if meta.control_plane_rpc_address:
                rpc_addresses[unit] = meta.control_plane_rpc_address

            data_addresses = {unit: [] for unit in dst_units}
            for meta in dst_metadata:
              unit = _raiden_id_from_proto(meta.unit)
              if unit in data_addresses:
                data_addresses[unit] = list(meta.shards)

            # Group flat entries into slices for broadcast
            groups = {}
            for src_unit, schedules in computed_schedules.items():
              for shard_idx, entries in schedules.items():
                for entry in entries:
                  (
                      dst_peer,
                      dst_shard_idx,
                      dst_block_offset,
                      src_block_offset,
                      size,
                      src_block_id,
                      dst_block_id,
                      src_stride,
                      dst_stride,
                      count,
                      layer_idx,
                      pool_group,
                  ) = entry
                  dst_unit = data_address_to_unit.get(dst_peer)
                  if not dst_unit:
                    continue
                  key = (
                      src_unit,
                      shard_idx,
                      src_block_id,
                      src_block_offset,
                      size,
                      src_stride,
                      count,
                      layer_idx,
                      pool_group,
                  )
                  val = (
                      dst_unit,
                      dst_peer,
                      dst_shard_idx,
                      dst_block_id,
                      dst_block_offset,
                      dst_stride,
                  )
                  groups.setdefault(key, []).append(val)

            # Partition slices into direct transfers and tree-broadcast
            # transfers.
            direct_schedules = {}
            broadcast_groups = {}

            # Group broadcast tasks by routing compatibility and layer_group_idx
            for key, targets in groups.items():
              unique_dst_units = set(t[0] for t in targets)
              is_tree_broadcast = (
                  len(unique_dst_units) > 1
                  and len(unique_dst_units) > self.broadcast_k
              )
              if not is_tree_broadcast:
                # Re-assemble entry for flat schedule (direct transfer)
                (
                    src_unit,
                    shard_idx,
                    src_block_id,
                    src_block_offset,
                    size,
                    src_stride,
                    count,
                    layer_idx,
                    pool_group,
                ) = key
                for (
                    dst_unit,
                    dst_peer,
                    dst_shard_idx,
                    dst_block_id,
                    dst_block_offset,
                    dst_stride,
                ) in targets:
                  entry = (
                      dst_peer,
                      dst_shard_idx,
                      dst_block_offset,
                      src_block_offset,
                      size,
                      src_block_id,
                      dst_block_id,
                      src_stride,
                      dst_stride,
                      count,
                      layer_idx,
                      pool_group,
                  )
                  direct_schedules.setdefault(src_unit, {}).setdefault(
                      shard_idx, []
                  ).append(entry)
              else:
                (
                    src_unit,
                    shard_idx,
                    src_block_id,
                    src_block_offset,
                    size,
                    src_stride,
                    count,
                    layer_idx,
                    pool_group,
                ) = key

                layer_group_idx = (
                    layer_idx // group_size if group_size > 1 else layer_idx
                )

                # Routing key: same src_unit, shard_idx, pool_group,
                # layer_group_idx and same set of destination units
                # Sort targets by dst_peer to ensure consistent order.
                # Compatibility key includes dst_unit, dst_peer, and
                # dst_shard_idx to ensure identical routing and shard mapping
                # at all hops.
                sorted_targets = sorted(targets, key=lambda t: t[1])
                targets_routing_key = tuple(
                    (t[0], t[1], t[2]) for t in sorted_targets
                )

                group_key = (
                    src_unit,
                    shard_idx,
                    pool_group,
                    layer_group_idx,
                    targets_routing_key,
                )
                broadcast_groups.setdefault(group_key, []).append(
                    (key, targets)
                )

            if direct_schedules:
              dst_unit_counts = {}
              dst_unit_layer_counts = {}
              for src_unit, schedules in direct_schedules.items():
                for shard_idx, entries in schedules.items():
                  for entry in entries:
                    dst_peer = entry[0]
                    dst_unit = data_address_to_unit.get(dst_peer)
                    if dst_unit:
                      size = entry[4]
                      src_stride = entry[7]
                      dst_stride = entry[8]
                      count = entry[9]
                      layer_idx = entry[10] if len(entry) > 10 else 0
                      is_contiguous = (count == 1) or (
                          src_stride == size and dst_stride == size
                      )
                      tasks_count = 1 if is_contiguous else count
                      dst_unit_counts[dst_unit] = (
                          dst_unit_counts.get(dst_unit, 0) + tasks_count
                      )
                      dst_unit_layer_counts.setdefault(dst_unit, {})
                      dst_unit_layer_counts[dst_unit][layer_idx] = (
                          dst_unit_layer_counts[dst_unit].get(layer_idx, 0)
                          + tasks_count
                      )
              if expected_block_count == 0 and dst_unit_counts:
                expected_block_count = max(dst_unit_counts.values())
              vars_info = f"{num_vars} variable(s), " if num_vars > 0 else ""
              logging.info(
                  "Transfer %s (uuid=%s): generated schedule for %s -> %s"
                  " (%s%d expected blocks)",
                  req_id,
                  uuid,
                  _format_units(src_units),
                  _format_units(dst_units),
                  vars_info,
                  expected_block_count,
              )
            else:
              dst_unit_counts = {}
              dst_unit_layer_counts = {}

            direct_dsts = []
            for scheds in direct_schedules.values():
              for entries in scheds.values():
                for entry in entries:
                  dst_peer = entry[0]
                  d_node = data_address_to_unit.get(dst_peer)
                  if d_node and d_node not in direct_dsts:
                    direct_dsts.append(d_node)

            if (
                self.enable_plan_cache
                and use_cached_plan
                and not shard_push_schedules
            ):
              with self._lock:
                self._plan_cache[cache_key] = _CachedTransferSchedule(
                    computed_schedules=computed_schedules,
                    direct_schedules=direct_schedules,
                    broadcast_groups=broadcast_groups,
                    local_skip_tiling=dict(local_skip_tiling)
                    if local_skip_tiling
                    else {},
                    expected_block_count=expected_block_count,
                    dst_unit_layer_counts=dst_unit_layer_counts,
                    data_address_to_unit=dict(data_address_to_unit),
                    direct_dsts=list(direct_dsts),
                    rpc_addresses=dict(rpc_addresses),
                    data_addresses=data_addresses,
                    dst_unit_counts=dst_unit_counts,
                )

          # Build final plan and replace the partial plan
          final_plan = TransferPlan(
              src_units=list(computed_schedules.keys())
              if computed_schedules
              else src_units,
              dst_units=dst_units,
              plan=None,
              shard_push_schedules=computed_schedules,
              worker_rpc_addresses=rpc_addresses,
              worker_data_addresses=data_addresses,
              uuid=uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=use_block_chunks,
              is_sender=True,
              expected_block_count=expected_block_count,
              dst_expected_layer_chunk_counts=dst_unit_layer_counts,
              dst_expected_block_counts=dst_unit_counts,
              req_id=req_id,
              skip_d2h=skip_d2h,
              skip_tiling=local_skip_tiling,
              parallelism=parallelism or 1,
          )
          with self._lock:
            self._active_transfers[req_id] = final_plan

          # Construct direct_plan upfront if direct_schedules exist
          direct_plan = None
          if direct_schedules:
            direct_plan = TransferPlan(
                src_units=list(direct_schedules.keys()),
                dst_units=dst_units,
                plan=None,
                shard_push_schedules=direct_schedules,
                worker_rpc_addresses=dict(final_plan.worker_rpc_addresses),
                worker_data_addresses=dict(final_plan.worker_data_addresses),
                uuid=uuid,
                dst_mem_type=dst_mem_type,
                use_block_chunks=True,
                is_sender=True,
                expected_block_count=expected_block_count,
                dst_expected_layer_chunk_counts=dst_unit_layer_counts,
                dst_expected_block_counts=dst_unit_counts,
                src_schedule_keys={
                    u: i for i, u in enumerate(direct_schedules.keys())
                },
                req_id=req_id,
                skip_d2h=skip_d2h,
                skip_tiling=local_skip_tiling,
                parallelism=final_plan.parallelism,
            )

          # 1. Arm direct schedule receivers
          if direct_schedules:
            if dst_controller_address:
              dst_facade = RaidenControllerClientFacade(
                  dst_controller_address,
                  name_resolver=self.worker_rpc_client.name_resolver,
              )
              loop = asyncio.get_running_loop()
              success = await loop.run_in_executor(
                  None,
                  dst_facade.register_transfer_schedule,
                  list(direct_schedules.keys()),
                  direct_dsts,
                  req_id,
                  True,
                  False,
                  expected_block_count,
                  uuid,
                  dst_controller_address,
                  src_controller_address,
                  direct_schedules,
                  dst_mem_type,
                  skip_d2h,
                  local_skip_tiling,
              )
              if not success:
                raise RuntimeError("Failed remote prepare for direct schedules")
            else:
              local_direct_dsts = [
                  u for u in direct_dsts if u in self._registered_shards
              ]
              if local_direct_dsts:
                await asyncio.gather(*[
                    self.worker_rpc_client.start_transfer(unit, direct_plan)
                    for unit in local_direct_dsts
                ])

          # 2. Arm destination controller for tree broadcast top-level req_id
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
                  name_resolver=self.worker_rpc_client.name_resolver,
              )
              loop = asyncio.get_running_loop()
              success = await loop.run_in_executor(
                  None,
                  dst_facade.register_transfer_schedule,
                  src_units,
                  dst_units,
                  req_id,
                  True,
                  False,
                  0,  # expected_block_count = 0 to complete immediately
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

          if direct_schedules:
            local_direct_srcs = [
                u
                for u in direct_schedules.keys()
                if u in self._registered_shards
            ]
            if local_direct_srcs:
              push_tasks.append(
                  asyncio.gather(*[
                      self.worker_rpc_client.start_transfer(unit, direct_plan)
                      for unit in local_direct_srcs
                  ])
              )

          if broadcast_groups:
            for group_key, keys_and_targets in broadcast_groups.items():
              task = self._execute_slice_broadcast(
                  keys_and_targets=keys_and_targets,
                  final_plan=final_plan,
                  fanout_k=self.broadcast_k,
                  req_id=req_id,
                  dst_mem_type=dst_mem_type,
                  dst_controller_address=dst_controller_address,
                  src_controller_address=src_controller_address,
              )
              push_tasks.append(task)

          if push_tasks:
            await asyncio.gather(*push_tasks)

      else:
        # === OLD PLAN-BASED WORKFLOW (Backward Compatibility) ===
        # Retrieve the plan that was stored synchronously
        with self._lock:
          old_plan = self._active_transfers[req_id]

        # 1. Send start_transfer to Destination workers first to register the
        # plan.
        # This is REQUIRED in the old workflow because they need the plan to
        # unpack!
        for unit in dst_units:
          await self.worker_rpc_client.start_transfer(unit, old_plan)

        # 2. Send start_transfer to Source workers to trigger the actual push.
        await asyncio.gather(*[
            self.worker_rpc_client.start_transfer(unit, old_plan)
            for unit in old_plan.src_units
        ])

    transfer_task = _execute_transfer()
    future = RaidenFuture(session_id=session_id, transfer_task=transfer_task)
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


class RaidenControllerServer:
  """Centralized Control-Plane network servicer hosting a highly secure JSON/Pickle TCP Controller server."""

  def __init__(
      self,
      controller: "RaidenController",
      proto_module: Optional[Any] = None,
      raiden_proto_module: Optional[Any] = None,
  ):
    """Instantiates RaidenControllerServer on an active RaidenController instance.

    Args:
      controller: High-level RaidenController instance managing transfer plans.
      proto_module: Optional protobuf module to use for
        ControllerRequest/Response. Defaults to controller_service_pb2.
      raiden_proto_module: Optional protobuf module for raiden service
        primitives.
    """
    self._controller = controller
    self._proto_module = proto_module or controller_service_pb2
    self._raiden_proto_module = raiden_proto_module or raiden_service_pb2
    self._sock = create_server_socket(controller.port)
    # Port 0 is useful for atomic ephemeral-port selection in tests and local
    # harnesses. Publish the kernel-selected port before start()/stop() use it.
    if controller.port == 0:
      controller.port = int(self._sock.getsockname()[1])
    self._stopped = False
    self._thread = None

  @property
  def port(self) -> int:
    """Returns the bound listener port, including for a requested port 0."""
    return self._controller.port

  def start(self) -> int:
    """Spawns background server acceptance thread listening for incoming Controller RPCs.

    Returns:
      Active TCP listener port coordinate.
    """
    self._thread = threading.Thread(target=self._server_loop, daemon=True)
    self._thread.start()
    return self._controller.port

  def stop(self) -> None:
    """Signals servicer loop shutdown and unblocks pending accept state."""
    self._stopped = True
    for host in ("[::1]", "127.0.0.1"):
      try:
        wake_socket = connect_socket(
            f"{host}:{self._controller.port}", timeout=0.5
        )
        wake_socket.close()
        break
      except Exception:  # pylint: disable=broad-except
        pass

    try:
      self._sock.shutdown(socket.SHUT_RDWR)
    except Exception:
      pass
    try:
      self._sock.close()
    except Exception:
      pass

  def _server_loop(self) -> None:
    """Accepts connections; every handler owns its asyncio event loop."""
    while not self._stopped:
      try:
        conn, _ = self._sock.accept()
        if self._stopped:
          conn.close()
          break
        threading.Thread(
            target=self._handle_conn, args=(conn,), daemon=True
        ).start()
      except OSError:
        break

  def _handle_conn(self, conn: socket.socket) -> None:
    """Internal connection processing handler executing deserialized ControllerRequest Protobuf RPC payloads.

    Args:
      conn: Accepted incoming TCP socket client handle.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

      len_bytes = b""
      while len(len_bytes) < 4:
        chunk = conn.recv(4 - len(len_bytes))
        if not chunk:
          return
        len_bytes += chunk
      req_len = int.from_bytes(len_bytes, "big")

      req_bytes = b""
      while len(req_bytes) < req_len:
        req_bytes += conn.recv(req_len - len(req_bytes))

      req = self._proto_module.ControllerRequest()
      try:
        req.ParseFromString(req_bytes)
      except Exception:
        req.command = self._proto_module.ControllerRequest.COMMAND_UNSPECIFIED

      if (
          req.command
          == self._proto_module.ControllerRequest.COMMAND_COORDINATE_TRANSFER
          and req.HasField("coordinate_transfer_request")
      ):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        try:
          if (
              req.command
              == self._proto_module.ControllerRequest.COMMAND_COORDINATE_TRANSFER
          ):
            coord_req = req.coordinate_transfer_request
            srcs = [_raiden_id_from_proto(u) for u in coord_req.src_units]
            dsts = [_raiden_id_from_proto(u) for u in coord_req.dst_units]
            dst_mem_type = RaidenMemoryType.DRAM
            if (
                coord_req.dst_mem_type
                == self._raiden_proto_module.MEMORY_TYPE_HBM
            ):
              dst_mem_type = RaidenMemoryType.HBM

            # Reshard requests are recognized by their destination block
            # list; num_tokens was retired from the wire (planning is fully
            # byte-derived).
            num_tokens = 0 if coord_req.dst_device_block_ids else None

            future = self._controller.start_transfer(
                src_units=srcs,
                dst_units=dsts,
                req_id=coord_req.req_id if coord_req.req_id else None,
                dst_mem_type=dst_mem_type,
                use_block_chunks=coord_req.use_block_chunks,
                src_controller_address=coord_req.src_controller_address
                if coord_req.src_controller_address
                else None,
                dst_controller_address=coord_req.dst_controller_address
                if coord_req.dst_controller_address
                else None,
                uuid=coord_req.uuid if coord_req.uuid > 0 else None,
                is_sender=coord_req.is_sender,
                expected_block_count=coord_req.expected_block_count,
                shard_push_schedules=None,
                dst_device_block_ids=(
                    list(coord_req.dst_device_block_ids)
                    if coord_req.dst_device_block_ids
                    else None
                ),
                num_tokens=num_tokens,
                transfer_pool_tags=(
                    list(coord_req.transfer_pool_tags)
                    if coord_req.transfer_pool_tags
                    else None
                ),
                dst_block_counts=(
                    list(coord_req.dst_block_counts)
                    if coord_req.dst_block_counts
                    else None
                ),
            )
            if future.try_start():
              loop.run_until_complete(future.wait())
            else:
              future.wait_threadsafe()
            resp.success = True
        except Exception as e:
          resp.message = str(e)
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      elif (
          req.command
          == self._proto_module.ControllerRequest.COMMAND_GET_TRANSFER_STATUS
          and req.HasField("get_transfer_status_request")
      ):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        try:
          status_req = req.get_transfer_status_request
          status = self._controller.get_transfer_status(status_req.req_id)
          resp.get_transfer_status_response.status = status
          resp.success = True
        except Exception as e:  # pylint: disable=broad-except
          resp.message = str(e)
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      elif (
          req.command
          == self._proto_module.ControllerRequest.COMMAND_REGISTER_REQUEST_BLOCKS
          and req.HasField("register_request_blocks_request")
      ):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        resp.message = (
            "the Python request-block registration surface is unavailable; "
            "use the C++ reshard store"
        )
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      elif (
          req.command
          == self._proto_module.ControllerRequest.COMMAND_RELEASE_REQUEST_BLOCKS
          and req.HasField("release_request_blocks_request")
      ):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        resp.message = (
            "the Python request-block release surface is unavailable; use "
            "the C++ reshard store"
        )
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      elif (
          req.command
          == self._proto_module.ControllerRequest.COMMAND_COMPLETE_REQUEST_BLOCKS
          and req.HasField("complete_request_blocks_request")
      ):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        resp.message = (
            "the Python request-block completion surface is unavailable; "
            "use the C++ reshard store"
        )
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      elif req.command == (
          self._proto_module.ControllerRequest.COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED
      ) and req.HasField("cancel_request_blocks_if_unclaimed_request"):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        resp.message = (
            "the Python request-block cancellation surface is unavailable; "
            "use the C++ reshard store"
        )
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      else:
        raiden_req = self._raiden_proto_module.ControlRequest()
        raiden_req.ParseFromString(req_bytes)
        raiden_resp = self._raiden_proto_module.ControlResponse()
        raiden_resp.success = False
        try:
          if (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_WORK_UNIT
          ):
            reg = raiden_req.register_work_unit_request
            unit = _raiden_id_from_proto(reg.unit)
            shards = list(reg.shards)
            ctrl_addr = (
                reg.control_plane_rpc_address
                if reg.control_plane_rpc_address
                else None
            )
            mesh_shape = list(reg.mesh_shape) if reg.mesh_shape else None
            mesh_axes = list(reg.mesh_axes) if reg.mesh_axes else None
            layout = list(reg.layout) if reg.layout else None
            global_shape = list(reg.global_shape) if reg.global_shape else None
            itemsize = reg.itemsize if reg.itemsize > 0 else None
            pool_manifest = list(reg.pools) if reg.pools else None
            layout_fingerprint = (
                reg.layout_fingerprint if reg.layout_fingerprint else None
            )
            page_tokens = reg.page_tokens if reg.page_tokens > 0 else None
            transfer_parallelism = (
                reg.transfer_parallelism
                if reg.transfer_parallelism > 0
                else None
            )
            transfer_rank = (
                reg.transfer_rank if pool_manifest is not None else None
            )
            variables = list(reg.variables) if reg.variables else None

            self._controller.register_work_unit(
                unit,
                shards,
                control_plane_rpc_address=ctrl_addr,
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
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_GET_METADATA
          ):
            metadata_protos = self._controller.get_all_metadata()
            raiden_resp.get_metadata_response.metadata.extend(metadata_protos)
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_TRANSFER_SCHEDULE
          ):
            start_req = raiden_req.start_transfer_request
            srcs = [_raiden_id_from_proto(u) for u in start_req.src_units]
            dsts = [_raiden_id_from_proto(u) for u in start_req.dst_units]
            dst_mem_type = RaidenMemoryType.DRAM
            if (
                start_req.dst_mem_type
                == self._raiden_proto_module.MEMORY_TYPE_HBM
            ):
              dst_mem_type = RaidenMemoryType.HBM

            def decode_entries(schedule_proto):
              return [
                  (
                      entry.dst_peer,
                      entry.dst_shard_idx,
                      entry.dst_offset_bytes,
                      entry.src_offset_bytes,
                      entry.size_bytes,
                      entry.src_block_id,
                      entry.dst_block_id,
                      entry.src_stride_bytes,
                      entry.dst_stride_bytes,
                      entry.count,
                      entry.layer_idx,
                      entry.pool_group,
                  )
                  for entry in schedule_proto.entries
              ]

            if start_req.transfer_pool_indices or start_req.pool_groups:
              # Fail closed for pre-P2 peers: the planning controller now
              # arms destination workers directly and the worker's executor
              # owns receiver-side validation; there is no inter-controller
              # receiver-plan registration to accept. (Same pool-plan
              # discriminator as the worker listener's HasPoolReshardFields.)
              raise ValueError(
                  "Inter-controller reshard schedule registration is "
                  "retired: the planning controller arms destination "
                  "workers directly"
              )
            else:
              shard_push_schedules = {}
              if len(srcs) == 1:
                unit_schedules = {}
                for (
                    key_idx,
                    schedule_proto,
                ) in start_req.shard_push_schedules.items():
                  entries = decode_entries(schedule_proto)
                  if entries:
                    unit_schedules[key_idx] = entries
                if unit_schedules:
                  shard_push_schedules[srcs[0]] = unit_schedules
              else:
                # Legacy transfer IDs historically used job_replica_id as a
                # schedule key. This compatibility branch is intentionally
                # outside the Stage-3 fail-closed path.
                for src_unit in srcs:
                  src_replica_idx = int(src_unit.job_replica_id)
                  if src_replica_idx in start_req.shard_push_schedules:
                    entries = decode_entries(
                        start_req.shard_push_schedules[src_replica_idx]
                    )
                    if entries:
                      shard_push_schedules[src_unit] = {0: entries}

              skip_tiling = dict(start_req.skip_tiling)
              future = self._controller.start_transfer(
                  src_units=srcs,
                  dst_units=dsts,
                  req_id=start_req.req_id if start_req.req_id else None,
                  dst_mem_type=dst_mem_type,
                  use_block_chunks=start_req.use_block_chunks,
                  src_controller_address=None,
                  dst_controller_address=None,
                  uuid=start_req.uuid if start_req.uuid > 0 else None,
                  is_sender=start_req.is_sender,
                  expected_block_count=start_req.expected_block_count,
                  shard_push_schedules=shard_push_schedules,
                  skip_d2h=start_req.skip_d2h,
                  skip_tiling=skip_tiling,
              )
            if future.try_start():
              loop.run_until_complete(future.wait())
            else:
              future.wait_threadsafe()
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_SHUTDOWN
          ):
            if hasattr(self._controller.worker_rpc_client, "shutdown_workers"):
              asyncio.run(self._controller.worker_rpc_client.shutdown_workers())
            self.stop()
            raiden_resp.success = True
        except Exception as e:
          raiden_resp.message = str(e)
        resp_bytes = raiden_resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
    except Exception:  # pylint: disable=broad-except
      pass

    finally:
      loop.close()
      conn.close()


class RaidenControllerClientFacade:
  """Client-side stub encapsulating real remote Network RPCs to a centralized RaidenControllerServer."""

  def __init__(
      self,
      controller_address: str,
      name_resolver: Optional[NameResolver] = None,
      proto_module: Optional[Any] = None,
      raiden_proto_module: Optional[Any] = None,
  ):
    """Accepts Controller server coordinate 'ip:port'."""
    self._address = controller_address
    self._name_resolver = name_resolver
    self._proto_module = proto_module or controller_service_pb2
    self._raiden_proto_module = raiden_proto_module or raiden_service_pb2

  def _raiden_id_to_proto(
      self,
      unit: RaidenId,
  ) -> Any:
    return self._raiden_proto_module.RaidenIdProto(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

  def _send_protobuf_rpc(self, req: Any) -> Any:
    """Helper method to serialize and send an RPC Protobuf over robust persistent TCP sockets."""
    sock = connect_socket(
        self._address, timeout=300.0, resolver=self._name_resolver
    )

    try:
      payload = req.SerializeToString()
      sock.sendall(len(payload).to_bytes(4, "big") + payload)

      resp_len_bytes = b""
      while len(resp_len_bytes) < 4:
        chunk = sock.recv(4 - len(resp_len_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response length")
        resp_len_bytes += chunk
      resp_len = int.from_bytes(resp_len_bytes, "big")

      resp_bytes = b""
      while len(resp_bytes) < resp_len:
        chunk = sock.recv(resp_len - len(resp_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response data")
        resp_bytes += chunk

      resp = self._proto_module.ControllerResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return resp
    finally:
      sock.close()

  def _send_raiden_protobuf_rpc_response(self, req: Any) -> Any:
    """Sends a Raiden protobuf RPC response."""
    sock = connect_socket(
        self._address, timeout=300.0, resolver=self._name_resolver
    )

    try:
      payload = req.SerializeToString()
      sock.sendall(len(payload).to_bytes(4, "big") + payload)

      resp_len_bytes = b""
      while len(resp_len_bytes) < 4:
        chunk = sock.recv(4 - len(resp_len_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response length")
        resp_len_bytes += chunk
      resp_len = int.from_bytes(resp_len_bytes, "big")

      resp_bytes = b""
      while len(resp_bytes) < resp_len:
        chunk = sock.recv(resp_len - len(resp_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response data")
        resp_bytes += chunk

      resp = self._raiden_proto_module.ControlResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return resp
    finally:
      sock.close()

  def _send_raiden_protobuf_rpc(self, req: Any) -> bool:
    self._send_raiden_protobuf_rpc_response(req)
    return True

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      mesh_shape: Optional[typing.Sequence[int]] = None,
      layout: Optional[typing.Sequence[int]] = None,
      global_shape: Optional[typing.Sequence[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[typing.Sequence[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
      variables: Optional[typing.Sequence[Any]] = None,
      mesh_axes: Optional[typing.Sequence[str]] = None,
  ) -> None:
    """Sends remote RPC to register a physical worker entity with the central RaidenControllerServer.

    Args:
      unit: Work unit identifier owning the data shards.
      shards: list of physical Data TCP addresses (e.g. 'IP:Port').
      control_plane_rpc_address: Optional worker Control-Plane RPC servicer
        endpoint coordinate.
      mesh_shape: Optional logical mesh shape.
      layout: Optional minor_to_major mapping layout.
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
    reg_req = self._raiden_proto_module.RegisterWorkUnitRequest(
        unit=self._raiden_id_to_proto(unit),
        shards=shards,
        control_plane_rpc_address=(
            control_plane_rpc_address if control_plane_rpc_address else ""
        ),
    )
    if mesh_shape:
      reg_req.mesh_shape.extend(mesh_shape)
    if layout:
      reg_req.layout.extend(layout)
    if global_shape:
      reg_req.global_shape.extend(global_shape)
    if itemsize:
      reg_req.itemsize = itemsize
    if pool_manifest is not None:
      for pool in pool_manifest:
        reg_req.pools.add().CopyFrom(_coerce_pool_spec_proto(pool))
    if layout_fingerprint is not None:
      reg_req.layout_fingerprint = layout_fingerprint
    if page_tokens is not None:
      reg_req.page_tokens = page_tokens
    if transfer_parallelism is not None:
      reg_req.transfer_parallelism = transfer_parallelism
    if transfer_rank is not None:
      reg_req.transfer_rank = transfer_rank
    if variables is not None:
      reg_req.variables.extend(variables)
    if mesh_axes is not None:
      reg_req.mesh_axes.extend(mesh_axes)

    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_WORK_UNIT,
        register_work_unit_request=reg_req,
    )
    self._send_raiden_protobuf_rpc(req)

  def coordinate_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      use_block_chunks: bool = False,
      is_sender: bool = True,
      expected_block_count: int = 0,
      uuid: int = 0,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      shard_push_schedules: Optional[dict] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      dst_device_block_ids: Optional[typing.Sequence[int]] = None,
      num_tokens: Optional[int] = None,
      transfer_pool_tags: Optional[typing.Sequence[str]] = None,
      dst_block_counts: Optional[typing.Sequence[int]] = None,
  ) -> bool:
    """Sends remote RPC to coordinate global least-loaded transfer and blocks until fully complete."""
    coord_req = self._proto_module.CoordinateTransferRequest(
        src_units=[self._raiden_id_to_proto(u) for u in src_units],
        dst_units=[self._raiden_id_to_proto(u) for u in dst_units],
        use_block_chunks=use_block_chunks,
        is_sender=is_sender,
        expected_block_count=expected_block_count,
        uuid=uuid,
        req_id=req_id if req_id else "",
        dst_controller_address=dst_controller_address
        if dst_controller_address
        else "",
        src_controller_address=src_controller_address
        if src_controller_address
        else "",
        dst_mem_type=int(dst_mem_type),
    )
    if dst_device_block_ids is not None:
      coord_req.dst_device_block_ids.extend(dst_device_block_ids)
    # num_tokens is accepted for caller compatibility but retired from the
    # wire: planning and receiver validation are fully byte-derived.
    del num_tokens
    if transfer_pool_tags is not None:
      coord_req.transfer_pool_tags.extend(transfer_pool_tags)
    if dst_block_counts:
      coord_req.dst_block_counts.extend(int(c) for c in dst_block_counts)

    req = self._proto_module.ControllerRequest(
        command=self._proto_module.ControllerRequest.COMMAND_COORDINATE_TRANSFER,
        coordinate_transfer_request=coord_req,
    )
    self._send_protobuf_rpc(req)
    return True

  def get_transfer_status(self, req_id: str, uuid: int = 0) -> int:
    """Queries the controller for transfer status."""
    status_req = self._proto_module.GetTransferStatusRequest(
        req_id=req_id,
        uuid=uuid,
    )
    req = self._proto_module.ControllerRequest(
        command=self._proto_module.ControllerRequest.COMMAND_GET_TRANSFER_STATUS,
        get_transfer_status_request=status_req,
    )
    resp = self._send_protobuf_rpc(req)
    return resp.get_transfer_status_response.status

  def register_transfer_schedule(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      use_block_chunks: bool = False,
      is_sender: bool = False,
      expected_block_count: int = 0,
      uuid: int = 0,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      shard_push_schedules: Optional[dict] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      skip_d2h: bool = False,
      skip_tiling: Optional[dict[int, bool]] = None,
  ) -> bool:
    """Inter-controller RPC to register computed push schedules and prepare receivers."""
    start_req = self._raiden_proto_module.StartTransferRequest(
        src_units=[self._raiden_id_to_proto(u) for u in src_units],
        dst_units=[self._raiden_id_to_proto(u) for u in dst_units],
        use_block_chunks=use_block_chunks,
        is_sender=is_sender,
        expected_block_count=expected_block_count,
        uuid=uuid,
        req_id=req_id if req_id else "",
        dst_mem_type=int(dst_mem_type),
        skip_d2h=skip_d2h,
    )
    if skip_tiling:
      for layer_idx, skip in skip_tiling.items():
        start_req.skip_tiling[layer_idx] = skip

    if shard_push_schedules:
      for src_unit, push_schedules in shard_push_schedules.items():
        num_src_shards = len(push_schedules)
        for shard_idx, schedule in push_schedules.items():
          key_idx = (
              int(src_unit.job_replica_id)
              if (len(src_units) > 1 and num_src_shards == 1)
              else shard_idx
          )
          schedule_proto = self._raiden_proto_module.ShardPushScheduleProto()
          for entry_tuple in schedule:
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
            ) = entry_tuple
            layer_idx = extra[0] if extra else 0
            pool_group = extra[1] if len(extra) > 1 else 0
            entry_proto = schedule_proto.entries.add()
            entry_proto.dst_peer = dst_peer
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

    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_TRANSFER_SCHEDULE,
        start_transfer_request=start_req,
    )
    return self._send_raiden_protobuf_rpc(req)

  def start_transfer(self, *args, **kwargs) -> bool:
    """Alias for coordinate_transfer for backward compatibility."""
    return self.coordinate_transfer(*args, **kwargs)

  def shutdown(self) -> bool:
    """Sends remote RPC to trigger global cluster shutdown across all cooperating jobs."""
    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_SHUTDOWN
    )
    return self._send_raiden_protobuf_rpc(req)

  def get_metadata(self) -> list[Any]:
    """Queries the controller for all registered work units' metadata."""
    sock = connect_socket(
        self._address, timeout=300.0, resolver=self._name_resolver
    )
    try:
      req = self._raiden_proto_module.ControlRequest(
          command=self._raiden_proto_module.ControlRequest.COMMAND_GET_METADATA
      )
      payload = req.SerializeToString()
      sock.sendall(len(payload).to_bytes(4, "big") + payload)

      resp_len_bytes = b""
      while len(resp_len_bytes) < 4:
        chunk = sock.recv(4 - len(resp_len_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response length")
        resp_len_bytes += chunk
      resp_len = int.from_bytes(resp_len_bytes, "big")

      resp_bytes = b""
      while len(resp_bytes) < resp_len:
        chunk = sock.recv(resp_len - len(resp_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response data")
        resp_bytes += chunk

      resp = self._raiden_proto_module.ControlResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return list(resp.get_metadata_response.metadata)
    finally:
      sock.close()

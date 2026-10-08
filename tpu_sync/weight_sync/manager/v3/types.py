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

"""Typed configuration, options, futures, and plan views for the V3 controller.

Shared proto/ID helpers (``NameResolver``, ``RaidenMemoryType``,
``VariableMetadata``, ``coerce_variable_proto``, ``raiden_id_from_proto``,
...) are imported from the parent ``controller_types`` module rather than
re-implemented here; this module only defines V3-specific types.
"""

import asyncio
import concurrent.futures
import copy
import dataclasses
import enum
import itertools
import math
import os
import socket
from typing import Any, Mapping, Optional, Protocol, Sequence

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import controller_service_pb2
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager.v3 import _controller_v3

NameResolver = controller_types.NameResolver
RaidenMemoryType = controller_types.RaidenMemoryType
VariableMetadata = controller_types.VariableMetadata
coerce_variable_proto = controller_types.coerce_variable_proto
extract_host_ip = controller_types.extract_host_ip
raiden_id_from_proto = controller_types.raiden_id_from_proto
raiden_id_to_proto = controller_types.raiden_id_to_proto

_ENV_BROADCAST_HOST_RATIO = "RAIDEN_BROADCAST_HOST_RATIO"
_ENV_NUM_BUNDLE_GROUPS = "RAIDEN_NUM_BUNDLE_GROUPS"
_ENV_MAX_CONCURRENT_UPLOADS = "RAIDEN_MAX_CONCURRENT_UPLOADS_PER_SOURCE"
_ENV_NUM_STRIPES = "RAIDEN_NUM_STRIPES"
_ENV_SEED_REPLICATION = "RAIDEN_SEED_REPLICATION"
_ENV_GRANT_BATCH_SIZE = "RAIDEN_GRANT_BATCH_SIZE"
_ENV_LEASE_TIMEOUT_S = "RAIDEN_LEASE_TIMEOUT_S"
_ENV_LONG_POLL_TIMEOUT_S = "RAIDEN_LONG_POLL_TIMEOUT_S"
_ENV_TRANSFER_TIMEOUT_S = "RAIDEN_TRANSFER_TIMEOUT_S"


class TransferStatus(enum.IntEnum):
  """Transfer lifecycle status; values match ``GetTransferStatusResponse``."""

  UNSPECIFIED = (
      controller_service_pb2.GetTransferStatusResponse.STATUS_UNSPECIFIED
  )
  NOT_STARTED = (
      controller_service_pb2.GetTransferStatusResponse.STATUS_NOT_STARTED
  )
  IN_PROGRESS = (
      controller_service_pb2.GetTransferStatusResponse.STATUS_IN_PROGRESS
  )
  COMPLETED = controller_service_pb2.GetTransferStatusResponse.STATUS_COMPLETED
  FAILED = controller_service_pb2.GetTransferStatusResponse.STATUS_FAILED

  @property
  def is_terminal(self) -> bool:
    return self in (TransferStatus.COMPLETED, TransferStatus.FAILED)


def seconds_to_ms(seconds: float) -> int:
  """Converts a duration in seconds to whole milliseconds (rounded)."""
  return int(round(float(seconds) * 1000))


def check_timeout_s(name: str, seconds: float) -> None:
  """Raises ``ValueError`` unless ``seconds`` is positive and at least 1ms."""
  if not float(seconds) > 0 or seconds_to_ms(seconds) < 1:
    raise ValueError(f"{name} must be > 0 (at least 1ms), got {seconds}")


def check_grant_batch_size(
    grant_batch_size: int, max_concurrent_uploads_per_source: int
) -> None:
  """Raises ``ValueError`` unless ``1 <= grant_batch_size <= c``."""
  if int(grant_batch_size) < 1:
    raise ValueError(f"grant_batch_size must be >= 1, got {grant_batch_size}")
  if int(grant_batch_size) > int(max_concurrent_uploads_per_source):
    raise ValueError(
        f"grant_batch_size ({grant_batch_size}) must not exceed"
        " max_concurrent_uploads_per_source"
        f" ({max_concurrent_uploads_per_source})"
    )


@dataclasses.dataclass(frozen=True)
class WeightSyncConfig:
  """Immutable construction-time configuration for ``RaidenControllerV3``.

  All fields have concrete defaults; nothing is read from the environment
  implicitly. Use ``WeightSyncConfig.from_env()`` to layer environment
  overrides on top of the defaults.

  Attributes:
    port: TCP port for ``start_server()``; ``0`` binds an ephemeral port.
    request_registry_ttl_s: Seconds a finished (COMPLETED/FAILED) transfer
      record, its plan and its pull scheduler are retained after its result has
      been observed (returned by ``sync_weights``/``execute_plan``,
      ``wait_for_transfer`` or a terminal ``get_transfer_status``, including
      remote polling). Results nobody has observed yet, and materialized but not
      yet executed plans, are retained for at least 60s regardless. With ``0`` a
      record is evicted by the next eviction pass after its result was observed.
    broadcast_host_ratio: Trainer-egress/sampler-ingress bandwidth ratio.
    enable_plan_cache: Whether logical reshard schedules are cached across
      transfers with identical topology.
    num_bundle_groups: Number of coarse variable bundles per transfer.
    max_concurrent_uploads_per_source: Pulls a source host serves at once (c).
    num_stripes: Number of stripes (contiguous bundle ranges) the Trainer seeds;
      ``0`` picks ``min(#bundles, #dst_replicas // seed_replication)``.
    seed_replication: Number of destination replicas every stripe is pushed to
      by the Trainer (>= 1).
    grant_batch_size: Pull leases a sampler host holds at once (k_b, >= 1, at
      most ``max_concurrent_uploads_per_source``: more leases than a source
      serves at once would only queue up behind its uploads).
    lease_timeout_s: How long a pull lease stays valid without being reported (>
      0). An expired lease is granted again, possibly from another source.
    long_poll_timeout_s: How long the controller holds a pull request it cannot
      grant anything to yet (> 0).
    transfer_timeout_s: Deadline of a whole transfer (Trainer push and pull
      phase), counted from the start of its execution (> 0). A transfer still
      running at its deadline fails with ``DEADLINE_EXCEEDED``.
      ``TransferOptions.timeout_s`` overrides it per transfer.
  """

  port: int = 0
  request_registry_ttl_s: float = 600.0
  broadcast_host_ratio: float = 1.0
  enable_plan_cache: bool = True
  num_bundle_groups: int = 8
  max_concurrent_uploads_per_source: int = 8
  num_stripes: int = 0
  seed_replication: int = 2
  grant_batch_size: int = 8
  lease_timeout_s: float = 30.0
  long_poll_timeout_s: float = 5.0
  transfer_timeout_s: float = 600.0

  def __post_init__(self) -> None:
    if self.port < 0:
      raise ValueError(f"port must be >= 0, got {self.port}")
    if self.request_registry_ttl_s < 0:
      raise ValueError("request_registry_ttl_s must be >= 0")
    if self.broadcast_host_ratio <= 0:
      raise ValueError("broadcast_host_ratio must be > 0")
    if self.num_bundle_groups < 1:
      raise ValueError("num_bundle_groups must be >= 1")
    if self.max_concurrent_uploads_per_source < 1:
      raise ValueError("max_concurrent_uploads_per_source must be >= 1")
    if self.num_stripes < 0:
      raise ValueError("num_stripes must be >= 0")
    if self.seed_replication < 1:
      raise ValueError("seed_replication must be >= 1")
    check_grant_batch_size(
        self.grant_batch_size, self.max_concurrent_uploads_per_source
    )
    check_timeout_s("lease_timeout_s", self.lease_timeout_s)
    check_timeout_s("long_poll_timeout_s", self.long_poll_timeout_s)
    check_timeout_s("transfer_timeout_s", self.transfer_timeout_s)

  @classmethod
  def from_env(
      cls, env: Optional[Mapping[str, str]] = None, **overrides: Any
  ) -> "WeightSyncConfig":
    """Builds a config from defaults, then environment, then ``overrides``.

    Recognized environment variables: ``RAIDEN_BROADCAST_HOST_RATIO``,
    ``RAIDEN_NUM_BUNDLE_GROUPS``, ``RAIDEN_MAX_CONCURRENT_UPLOADS_PER_SOURCE``,
    ``RAIDEN_NUM_STRIPES``, ``RAIDEN_SEED_REPLICATION``,
    ``RAIDEN_GRANT_BATCH_SIZE``, ``RAIDEN_LEASE_TIMEOUT_S``,
    ``RAIDEN_LONG_POLL_TIMEOUT_S``, ``RAIDEN_TRANSFER_TIMEOUT_S``.

    Args:
      env: Mapping to read from (defaults to ``os.environ``).
      **overrides: Explicit field values that take precedence over ``env``.

    Returns:
      A validated ``WeightSyncConfig``.
    """
    src = os.environ if env is None else env
    values: dict[str, Any] = {}
    if _ENV_BROADCAST_HOST_RATIO in src:
      values["broadcast_host_ratio"] = float(src[_ENV_BROADCAST_HOST_RATIO])
    if _ENV_NUM_BUNDLE_GROUPS in src:
      values["num_bundle_groups"] = int(src[_ENV_NUM_BUNDLE_GROUPS])
    if _ENV_MAX_CONCURRENT_UPLOADS in src:
      values["max_concurrent_uploads_per_source"] = int(
          src[_ENV_MAX_CONCURRENT_UPLOADS]
      )
    if _ENV_NUM_STRIPES in src:
      values["num_stripes"] = int(src[_ENV_NUM_STRIPES])
    if _ENV_SEED_REPLICATION in src:
      values["seed_replication"] = int(src[_ENV_SEED_REPLICATION])
    if _ENV_GRANT_BATCH_SIZE in src:
      values["grant_batch_size"] = int(src[_ENV_GRANT_BATCH_SIZE])
    if _ENV_LEASE_TIMEOUT_S in src:
      values["lease_timeout_s"] = float(src[_ENV_LEASE_TIMEOUT_S])
    if _ENV_LONG_POLL_TIMEOUT_S in src:
      values["long_poll_timeout_s"] = float(src[_ENV_LONG_POLL_TIMEOUT_S])
    if _ENV_TRANSFER_TIMEOUT_S in src:
      values["transfer_timeout_s"] = float(src[_ENV_TRANSFER_TIMEOUT_S])
    values.update(overrides)
    return cls(**values)


@dataclasses.dataclass(frozen=True)
class TransferOptions:
  """Immutable per-transfer options.

  Attributes:
    dst_mem_type: Destination memory type.
    skip_d2h: Skip the device-to-host staging copy on senders.
    skip_tiling: Per-layer ``{layer_idx: skip}`` tiling overrides.
    parallelism: Sender parallelism hint (>= 1).
    use_cached_plan: Whether a cached logical schedule may be reused.
    timeout_s: Deadline of this transfer in seconds (> 0), overriding the
      controller's ``transfer_timeout_s``; ``None`` uses the controller's. An
      execution option: it does not affect the plan.
  """

  dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM
  skip_d2h: bool = False
  skip_tiling: Mapping[int, bool] = dataclasses.field(default_factory=dict)
  parallelism: int = 1
  use_cached_plan: bool = True
  timeout_s: Optional[float] = None

  def __post_init__(self) -> None:
    if self.parallelism < 1:
      raise ValueError("parallelism must be >= 1")
    if self.timeout_s is not None:
      check_timeout_s("timeout_s", self.timeout_s)
    object.__setattr__(
        self,
        "skip_tiling",
        {int(k): bool(v) for k, v in dict(self.skip_tiling or {}).items()},
    )


@dataclasses.dataclass(frozen=True)
class NDSlice:
  """N-dimensional half-open index interval ``[(start, end), ...]``."""

  dims: tuple[tuple[int, int], ...]

  @classmethod
  def from_bounds(cls, bounds: Sequence[tuple[int, int]]) -> "NDSlice":
    return cls(dims=tuple((int(s), int(e)) for s, e in bounds))

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(max(0, e - s) for s, e in self.dims)

  def __len__(self) -> int:
    return len(self.dims)

  def __iter__(self):
    return iter(self.dims)

  def __getitem__(self, idx: int) -> tuple[int, int]:
    return self.dims[idx]


def bounds_to_cpp_nd_slice(slice_like: Any) -> _controller_v3.NdSlice:
  """Converts an ``NDSlice`` or bounds sequence into a C++ ``NdSlice``."""
  if isinstance(slice_like, _controller_v3.NdSlice):
    return slice_like
  bounds = list(
      slice_like.dims if isinstance(slice_like, NDSlice) else slice_like
  )
  offsets = [int(s) for s, _ in bounds]
  sizes = [int(e) - int(s) for s, e in bounds]
  return _controller_v3.NdSlice(offsets, sizes)


@dataclasses.dataclass(frozen=True)
class VariableBundleSpec:
  """Contiguous bundle of variables tracked and pulled as an atomic unit."""

  bundle_id: int
  layer_indices: tuple[int, ...] = ()
  layer_byte_sizes: tuple[int, ...] = ()
  total_bytes: int = 0

  @classmethod
  def from_cpp(cls, cpp_spec: Any) -> "VariableBundleSpec":
    if isinstance(cpp_spec, cls):
      return cpp_spec
    return cls(
        bundle_id=int(cpp_spec.bundle_id),
        layer_indices=tuple(int(x) for x in cpp_spec.layer_indices),
        layer_byte_sizes=tuple(int(x) for x in cpp_spec.layer_byte_sizes),
        total_bytes=int(cpp_spec.total_bytes),
    )

  def to_cpp(self) -> _controller_v3.VariableBundleSpec:
    spec = _controller_v3.VariableBundleSpec()
    spec.bundle_id = int(self.bundle_id)
    spec.layer_indices = [int(x) for x in self.layer_indices]
    spec.layer_byte_sizes = [int(x) for x in self.layer_byte_sizes]
    spec.total_bytes = int(self.total_bytes)
    return spec


@dataclasses.dataclass(frozen=True)
class HostCommand:
  """Python view of one materialized per-host ``StartTransferRequest``."""

  unit: RaidenId
  host_idx: int
  control_endpoint: str
  request_bytes: bytes

  @classmethod
  def from_cpp(cls, cmd: Any) -> "HostCommand":
    proto = raiden_service_pb2.RaidenIdProto()
    proto.ParseFromString(bytes(cmd.unit_bytes))
    return cls(
        unit=raiden_id_from_proto(proto),
        host_idx=int(cmd.host_idx),
        control_endpoint=str(cmd.control_endpoint),
        request_bytes=bytes(cmd.request_bytes()),
    )

  def request(self) -> raiden_service_pb2.StartTransferRequest:
    req = raiden_service_pb2.StartTransferRequest()
    req.ParseFromString(self.request_bytes)
    return req


@dataclasses.dataclass(frozen=True)
class SamplerCommand:
  """Everything one sampler host receives for a transfer.

  A single receiver ``command`` covers the layers the host gets from the
  Trainer (``seeded_bundles``) and the layers it pulls from other samplers as
  the controller grants them.
  """

  replica_idx: int
  command: HostCommand
  seeded_bundles: tuple[int, ...]
  seeded_layers: tuple[int, ...]

  @classmethod
  def from_cpp(cls, cmd: Any) -> "SamplerCommand":
    return cls(
        replica_idx=int(cmd.replica_idx),
        command=HostCommand.from_cpp(cmd.command),
        seeded_bundles=tuple(int(x) for x in cmd.seeded_bundles),
        seeded_layers=tuple(int(x) for x in cmd.seeded_layers),
    )

  @property
  def unit(self) -> RaidenId:
    return self.command.unit

  @property
  def control_endpoint(self) -> str:
    return self.command.control_endpoint

  def request(self) -> raiden_service_pb2.StartTransferRequest:
    return self.command.request()


@dataclasses.dataclass(frozen=True)
class TransferPlan:
  """Immutable Python view of a materialized V3 transfer plan.

  Built from the C++ ``MaterializedTransferPlan``; the authoritative plan
  lives in the C++ controller's transfer record keyed by ``req_id``.

  The Trainer splits the variable bundles into ``num_stripes`` contiguous
  stripes and pushes every stripe to ``seed_replication`` destination
  replicas (``stripe_seeds``), in ``trainer_waves``; the samplers then complete
  each other with pulls the controller schedules while the transfer runs.

  Attributes:
    req_id: Transfer identifier.
    uuid: Transfer UUID.
    src_units: Source (trainer) work units.
    dst_units: Destination (sampler) work units; replica ``i`` is
      ``dst_units[i]``.
    options: Per-transfer options the plan was built with; executing the plan
      uses their ``timeout_s``.
    num_stripes: Number of stripes (G).
    seed_replication: Number of replicas every stripe is pushed to (R).
    stripe_bundles: ``stripe_bundles[g]``: bundle ids of stripe ``g``.
    stripe_seeds: ``stripe_seeds[g]``: replica indices stripe ``g`` is pushed
      to.
    trainer_waves: ``trainer_waves[w]``: ``(replica_idx, stripe)`` pushes that
      run concurrently in wave ``w``.
    seed_units: Destination units seeded by the Trainer, in replica order.
    variable_bundles: Bundles the model is split into.
    sampler_commands: One command per sampler host of every destination replica,
      in replica order.
    trainer_wave_commands: ``trainer_wave_commands[w]``: one sender command per
      Trainer host for wave ``w``.
  """

  req_id: str
  uuid: int
  src_units: tuple[RaidenId, ...]
  dst_units: tuple[RaidenId, ...]
  options: TransferOptions
  num_stripes: int
  seed_replication: int
  stripe_bundles: tuple[tuple[int, ...], ...]
  stripe_seeds: tuple[tuple[int, ...], ...]
  trainer_waves: tuple[tuple[tuple[int, int], ...], ...]
  seed_units: tuple[RaidenId, ...]
  variable_bundles: tuple[VariableBundleSpec, ...]
  sampler_commands: tuple[SamplerCommand, ...]
  trainer_wave_commands: tuple[tuple[HostCommand, ...], ...]
  _description: Mapping[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @classmethod
  def from_cpp(
      cls,
      cpp_plan: Any,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      options: TransferOptions,
  ) -> "TransferPlan":
    """Builds a ``TransferPlan`` from a C++ ``MaterializedTransferPlan``."""
    description = cpp_plan.describe_plan()
    dst_list = tuple(dst_units)
    stripe_seeds = tuple(
        tuple(int(r) for r in seeds) for seeds in description["stripe_seeds"]
    )
    seed_replicas = sorted(set(itertools.chain.from_iterable(stripe_seeds)))
    return cls(
        req_id=str(cpp_plan.req_id),
        uuid=int(cpp_plan.uuid),
        src_units=tuple(src_units),
        dst_units=dst_list,
        options=options,
        num_stripes=int(cpp_plan.num_stripes),
        seed_replication=int(cpp_plan.seed_replication),
        stripe_bundles=tuple(
            tuple(int(b) for b in bundles)
            for bundles in description["stripe_bundles"]
        ),
        stripe_seeds=stripe_seeds,
        trainer_waves=tuple(
            tuple((int(r), int(g)) for r, g in wave)
            for wave in description["trainer_waves"]
        ),
        seed_units=tuple(dst_list[r] for r in seed_replicas),
        variable_bundles=tuple(
            VariableBundleSpec.from_cpp(b) for b in cpp_plan.variable_bundles
        ),
        sampler_commands=tuple(
            SamplerCommand.from_cpp(c) for c in cpp_plan.sampler_commands
        ),
        trainer_wave_commands=tuple(
            tuple(HostCommand.from_cpp(c) for c in wave)
            for wave in cpp_plan.trainer_wave_commands
        ),
        _description=description,
    )

  def describe_plan(self) -> dict[str, Any]:
    """Returns the seed layout as plain Python data.

    Same layout as the C++ ``MaterializedTransferPlan.describe_plan()``:
    ``num_stripes``, ``seed_replication``, ``stripe_bundles``,
    ``stripe_seeds`` and ``trainer_waves``. The result is a copy and may be
    mutated freely.
    """
    return copy.deepcopy(dict(self._description))


class RaidenFuture:
  """Eager handle for an in-flight transfer.

  The work is already running when a ``RaidenFuture`` is handed out; waiting
  is optional. ``cancel()`` only succeeds if the work has not started yet (the
  underlying C++ transfer cannot be interrupted once dispatched); a timed-out
  ``wait_threadsafe``/``result`` does *not* cancel the transfer. The transfer
  itself is bounded by its transfer timeout (``TransferOptions.timeout_s`` or
  the controller's ``transfer_timeout_s``), after which it fails with
  ``DEADLINE_EXCEEDED``.
  """

  def __init__(self, future: concurrent.futures.Future[Any], req_id: str):
    self._future = future
    self.req_id = req_id

  async def wait(self) -> Any:
    """Awaits completion from an asyncio event loop."""
    return await asyncio.wrap_future(self._future)

  def wait_threadsafe(self, timeout: Optional[float] = None) -> Any:
    """Blocks the calling thread until completion; raises on failure/timeout."""
    try:
      return self._future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as e:
      raise TimeoutError(
          f"Timed out waiting for transfer {self.req_id!r} after {timeout}s; "
          "the transfer is still running."
      ) from e

  def result(self, timeout: Optional[float] = None) -> Any:
    return self.wait_threadsafe(timeout=timeout)

  def done(self) -> bool:
    return self._future.done()

  def running(self) -> bool:
    return self._future.running()

  def cancel(self) -> bool:
    """Best-effort cancel; returns False once the transfer has started."""
    return self._future.cancel()

  def cancelled(self) -> bool:
    return self._future.cancelled()

  def exception(
      self, timeout: Optional[float] = None
  ) -> Optional[BaseException]:
    return self._future.exception(timeout=timeout)

  def add_done_callback(self, fn: Any) -> None:
    self._future.add_done_callback(lambda _: fn(self))


class SynchronizerLike(Protocol):
  """Structural interface accepted by ``register_synchronizer``.

  Only ``num_shards`` and ``get_local_endpoints`` are strictly required for
  endpoint extraction; the remaining attributes let the controller synthesize
  ``VariableMetadataProto`` descriptors when the caller does not pass
  ``variables`` explicitly.
  """

  num_shards: int
  num_layers: int
  local_port: int
  listener_port: int

  def get_local_endpoints(self) -> Sequence[Any]:
    ...

  def get_host_buffer(self, layer_idx: int, shard_idx: int) -> Any:
    ...


def unit_to_bytes(unit_or_replica_id: Any) -> bytes:
  """Serializes a ``RaidenId`` (or bare replica id string) to proto bytes."""
  if isinstance(unit_or_replica_id, RaidenId):
    return raiden_id_to_proto(unit_or_replica_id).SerializeToString()
  return raiden_service_pb2.RaidenIdProto(
      job_replica_id=str(unit_or_replica_id)
  ).SerializeToString()


def _copy_variable_proto(var: Any) -> raiden_service_pb2.VariableMetadataProto:
  """Returns a private ``VariableMetadataProto`` copy of ``var``."""
  out = raiden_service_pb2.VariableMetadataProto()
  out.CopyFrom(coerce_variable_proto(var))
  return out


def routable_local_ip() -> str:
  """Returns a routable local IP, or ``127.0.0.1`` when none is found.

  Only used as a fallback by ``extract_synchronizer_endpoints`` when the caller
  provides neither ``bind_ip`` nor explicit endpoints.
  """
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect(("10.255.255.255", 1))
      ip = s.getsockname()[0]
      if ip and not ip.startswith("127.") and ip != "0.0.0.0":
        return ip
  except OSError:
    pass
  try:
    for info in socket.getaddrinfo(
        socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
    ):
      ip = str(info[4][0])
      if ip and not ip.startswith("127.") and ip != "0.0.0.0":
        return ip
  except OSError:
    pass
  return "127.0.0.1"


def extract_synchronizer_endpoints(
    synchronizer: SynchronizerLike,
    bind_ip: Optional[str] = None,
    shards: Optional[Sequence[str]] = None,
    control_plane_rpc_address: Optional[str] = None,
) -> tuple[list[str], Optional[str]]:
  """Extracts data shard endpoints and control-plane address from a synchronizer."""
  num_shards = int(getattr(synchronizer, "num_shards", 0) or 0)
  resolved_shards: list[str] = list(shards) if shards is not None else []

  if not resolved_shards and hasattr(synchronizer, "get_local_endpoints"):
    raw_eps = synchronizer.get_local_endpoints() or []
    shards_by_idx: dict[int, str] = {}
    ep_list: list[str] = []
    for ep_info in raw_eps:
      if isinstance(ep_info, dict):
        ep_str = str(ep_info.get("endpoint", ""))
        ep_shards = ep_info.get("shards") or []
      else:
        ep_str = str(getattr(ep_info, "endpoint", ""))
        ep_shards = getattr(ep_info, "shards", ()) or ()
      if bind_ip and ":" in ep_str:
        ep_str = f"{bind_ip}:{ep_str.rsplit(':', 1)[-1]}"
      if ep_str:
        ep_list.append(ep_str)
        for s_idx in ep_shards:
          shards_by_idx[int(s_idx)] = ep_str
    if shards_by_idx:
      if num_shards > 0 and all(i in shards_by_idx for i in range(num_shards)):
        resolved_shards = [shards_by_idx[i] for i in range(num_shards)]
      else:
        resolved_shards = [shards_by_idx[k] for k in sorted(shards_by_idx)]
    elif ep_list:
      count = max(1, num_shards)
      resolved_shards = [
          ep_list[min(i * len(ep_list) // count, len(ep_list) - 1)]
          for i in range(count)
      ]

  if not resolved_shards:
    local_port = getattr(synchronizer, "local_port", None)
    if local_port is not None:
      host_ip = bind_ip or routable_local_ip()
      resolved_shards = [f"{host_ip}:{int(local_port)}"] * max(1, num_shards)

  resolved_ctrl = control_plane_rpc_address
  if not resolved_ctrl:
    listener_port = getattr(synchronizer, "listener_port", None)
    if listener_port is not None and int(listener_port) > 0:
      host_ip = (
          bind_ip
          or (extract_host_ip(resolved_shards[0]) if resolved_shards else "")
          or routable_local_ip()
      )
      resolved_ctrl = f"{host_ip}:{int(listener_port)}"

  return resolved_shards, resolved_ctrl


def _global_shard_indices_error(
    var_name: str, num_shards: int, got: int
) -> str:
  """Returns the error for a variable without one global index per shard."""
  return (
      f"Variable {var_name!r} has {got} global_shard_indices for"
      f" {num_shards} shards. global_shard_indices is required on every"
      " variable: entry i is the global shard index, in [0, prod(mesh_shape)),"
      " held by shards[i]. The deprecated mesh_axes, host_subgrid, unit-level"
      " mesh_shape and sharding_spec are not used to derive it."
  )


def extract_synchronizer_variables(
    synchronizer: SynchronizerLike,
    resolved_shards: Sequence[str],
    *,
    variables: Optional[Sequence[Any]] = None,
    global_shard_indices: Optional[Sequence[int]] = None,
    mesh_shape: Optional[Sequence[int]] = None,
    layout: Optional[Sequence[int]] = None,
    itemsize: Optional[int] = None,
) -> tuple[Optional[list[raiden_service_pb2.VariableMetadataProto]], bool]:
  """Extracts or synthesizes ``VariableMetadataProto`` descriptors.

  The global shard indices come from the synchronizer's actual placement:
  ``global_shard_indices`` if given, else the synchronizer's public
  ``global_shard_indices`` attribute. They fill in descriptors that carry none
  and are required to synthesize descriptors.

  Args:
    synchronizer: Local ``WeightSynchronizer``-like object.
    resolved_shards: Data shard endpoints already resolved for this host.
    variables: Explicit variable descriptors; take precedence over anything read
      from ``synchronizer``.
    global_shard_indices: Global shard index held by each local shard, in the
      order of ``resolved_shards``.
    mesh_shape: Mesh shape used when synthesizing descriptors.
    layout: Layout used when synthesizing descriptors.
    itemsize: Element size in bytes used when synthesizing descriptors.

  Returns:
    ``(variables, was_auto_1d)`` where ``was_auto_1d`` is True when a 1-D mesh
    was synthesized from the shard placement (so multi-host merging may widen
    it).

  Raises:
    ValueError: If descriptors must be synthesized but no global shard indices
      are available, or if their count differs from the number of local shards.
  """
  num_local_shards = max(
      1,
      len(resolved_shards) or int(getattr(synchronizer, "num_shards", 0) or 1),
  )
  raw_gsi = global_shard_indices
  if raw_gsi is None:
    raw_gsi = getattr(synchronizer, "global_shard_indices", None)
  gsi = [int(x) for x in raw_gsi] if raw_gsi is not None else None
  if gsi is not None and len(gsi) != num_local_shards:
    raise ValueError(
        f"global_shard_indices has {len(gsi)} entries for"
        f" {num_local_shards} local shards; pass one global shard index per"
        " local shard."
    )

  raw_vars = variables
  if raw_vars is None:
    raw_vars = getattr(synchronizer, "variables", None)
  if raw_vars is None and callable(
      getattr(synchronizer, "get_variable_metadata", None)
  ):
    raw_vars = synchronizer.get_variable_metadata()
  if raw_vars is None:
    raw_vars = getattr(synchronizer, "variable_metadata", None)

  if raw_vars is not None:
    out_vars = []
    for v in raw_vars:
      vp = _copy_variable_proto(v)
      if not vp.global_shard_indices and gsi is not None:
        vp.global_shard_indices.extend(gsi)
      out_vars.append(vp)
    return out_vars, False

  num_layers = int(getattr(synchronizer, "num_layers", 0) or 0)
  if num_layers <= 0:
    return None, False
  if gsi is None:
    raise ValueError(
        "Cannot describe the synchronizer's weights without their global"
        " shard indices: pass global_shard_indices (the global shard index"
        " held by each local shard) or expose a public `global_shard_indices`"
        " attribute on the synchronizer."
    )

  explicit_mesh = mesh_shape or getattr(synchronizer, "mesh_shape", None)
  eff_mesh_shape = (
      [int(x) for x in explicit_mesh]
      if explicit_mesh
      else [max(num_local_shards, max(gsi) + 1)]
  )
  raw_layout = layout
  if raw_layout is None:
    raw_layout = getattr(synchronizer, "layout", None) or list(
        range(len(eff_mesh_shape) - 1, -1, -1)
    )
  eff_layout = [int(x) for x in raw_layout]
  layer_byte_sizes = getattr(synchronizer, "layer_byte_sizes", None)
  get_host_buffer = getattr(synchronizer, "get_host_buffer", None)
  slice_byte_size = int(getattr(synchronizer, "slice_byte_size", 0) or 0)

  synthesized = []
  for layer_idx in range(num_layers):
    shard_bytes = 0
    buf_itemsize = 0
    if layer_byte_sizes is not None and layer_idx < len(layer_byte_sizes):
      shard_bytes = int(layer_byte_sizes[layer_idx])
    if callable(get_host_buffer):
      try:
        buf = get_host_buffer(layer_idx, 0)
      except Exception:  # pylint: disable=broad-except
        buf = None
      if buf is not None:
        if shard_bytes <= 0 and hasattr(buf, "nbytes"):
          shard_bytes = int(buf.nbytes)
        dtype = getattr(buf, "dtype", None)
        if dtype is not None and hasattr(dtype, "itemsize"):
          buf_itemsize = int(dtype.itemsize)
    if shard_bytes <= 0 and slice_byte_size > 0:
      shard_bytes = slice_byte_size
    if shard_bytes <= 0:
      return None, False

    eff_itemsize = int(
        itemsize or getattr(synchronizer, "itemsize", 0) or buf_itemsize or 1
    )
    if eff_itemsize <= 0 or shard_bytes % eff_itemsize != 0:
      eff_itemsize = math.gcd(shard_bytes, max(1, eff_itemsize)) or 1
    elems_per_shard = shard_bytes // eff_itemsize
    var_shape = [elems_per_shard * eff_mesh_shape[0]] + [
        eff_mesh_shape[d] for d in range(1, len(eff_mesh_shape))
    ]
    synthesized.append(
        raiden_service_pb2.VariableMetadataProto(
            name=f"var_{layer_idx}",
            shape=var_shape,
            mesh_shape=eff_mesh_shape,
            layout=eff_layout,
            item_size=eff_itemsize,
            layer_idx=layer_idx,
            global_shard_indices=gsi,
        )
    )

  return synthesized, not explicit_mesh


HostRegistration = tuple[
    list[str],
    Optional[str],
    Optional[list[raiden_service_pb2.VariableMetadataProto]],
    bool,
]


def merge_host_synchronizer_entries(
    host_entries: Mapping[int, HostRegistration],
) -> tuple[
    list[str],
    Optional[str],
    Optional[list[raiden_service_pb2.VariableMetadataProto]],
]:
  """Merges per-host synchronizer registrations ordered by ``host_idx``."""
  sorted_items = sorted(host_entries.items(), key=lambda kv: kv[0])
  if len(sorted_items) == 1:
    _, (h_shards, h_ctrl, h_vars, _) = sorted_items[0]
    return (
        list(h_shards),
        h_ctrl,
        (list(h_vars) if h_vars is not None else None),
    )

  merged_shards: list[str] = []
  merged_ctrls: list[str] = []
  for _, (h_shards, h_ctrl, _, _) in sorted_items:
    merged_shards.extend(h_shards)
    if h_ctrl:
      for addr in str(h_ctrl).split(","):
        addr_s = addr.strip()
        if addr_s and addr_s not in merged_ctrls:
          merged_ctrls.append(addr_s)

  first_vars = sorted_items[0][1][2]
  was_auto_1d = all(item[1][3] for item in sorted_items)
  merged_vars = None
  if first_vars is not None:
    merged_vars = []
    for v_idx, base_v in enumerate(first_vars):
      vp = _copy_variable_proto(base_v)
      combined_gsi: list[int] = []
      for h_idx, (h_shards, _, h_vars, _) in sorted_items:
        hv = (
            coerce_variable_proto(h_vars[v_idx])
            if h_vars is not None and v_idx < len(h_vars)
            else None
        )
        got = len(hv.global_shard_indices) if hv is not None else 0
        if got != len(h_shards):
          raise ValueError(
              f"Host {h_idx}: "
              + _global_shard_indices_error(vp.name, len(h_shards), got)
          )
        combined_gsi.extend(int(g) for g in hv.global_shard_indices)
      del vp.global_shard_indices[:]
      vp.global_shard_indices.extend(combined_gsi)
      if (
          was_auto_1d
          and len(vp.mesh_shape) == 1
          and len(vp.shape) == 1
          and vp.mesh_shape[0] > 0
      ):
        elems_per_shard = vp.shape[0] // vp.mesh_shape[0]
        total_shards = max(len(merged_shards), max(combined_gsi) + 1)
        vp.mesh_shape[0] = total_shards
        vp.shape[0] = elems_per_shard * total_shards
      merged_vars.append(vp)

  return (
      merged_shards,
      ",".join(merged_ctrls) if merged_ctrls else None,
      merged_vars,
  )


def build_register_work_unit_request(
    unit: RaidenId,
    shards: Sequence[str],
    *,
    control_plane_rpc_address: Optional[str | Sequence[str]] = None,
    variables: Optional[Sequence[Any]] = None,
    itemsize: Optional[int] = None,
) -> raiden_service_pb2.RegisterWorkUnitRequest:
  """Validates registration arguments and builds a ``RegisterWorkUnitRequest``.

  Shared by the local controller and the remote facade so that both paths
  apply identical validation and produce identical wire payloads.

  Args:
    unit: Work unit being registered.
    shards: Data shard endpoints (one per shard, all non-empty).
    control_plane_rpc_address: Worker control endpoint(s), either a sequence or
      a comma-separated string.
    variables: ``VariableMetadataProto``-like descriptors, one per weight, each
      with ``shape``, ``mesh_shape``, ``layout``, ``item_size`` and the required
      ``global_shard_indices``: one entry per shard, where entry i is the global
      shard index, in ``[0, prod(mesh_shape))``, held by ``shards[i]``.
    itemsize: Element size in bytes for descriptors whose ``item_size`` is 0.

  Returns:
    The populated ``RegisterWorkUnitRequest``.

  Raises:
    ValueError: On empty shards, or a descriptor without one global shard index
      per shard.
  """
  if not shards or any(not s for s in shards):
    raise ValueError("shards must contain at least one non-empty endpoint")

  endpoints: list[str] = []
  if control_plane_rpc_address:
    if isinstance(control_plane_rpc_address, (list, tuple)):
      raw = [str(a) for a in control_plane_rpc_address]
    else:
      raw = str(control_plane_rpc_address).split(",")
    endpoints = [a.strip() for a in raw if a.strip()]

  req = raiden_service_pb2.RegisterWorkUnitRequest(
      unit=raiden_id_to_proto(unit),
      shards=list(shards),
      control_plane_rpc_address=",".join(endpoints),
      itemsize=int(itemsize) if itemsize is not None else 0,
  )
  for v in variables or ():
    vp = req.variables.add()
    vp.CopyFrom(coerce_variable_proto(v))
    if len(vp.global_shard_indices) != len(shards):
      raise ValueError(
          _global_shard_indices_error(
              vp.name, len(shards), len(vp.global_shard_indices)
          )
      )
  return req


def build_synchronizer_registration(
    host_registrations: dict[RaidenId, dict[int, HostRegistration]],
    unit: RaidenId,
    synchronizer: SynchronizerLike,
    *,
    variables: Optional[Sequence[Any]] = None,
    global_shard_indices: Optional[Sequence[int]] = None,
    mesh_shape: Optional[Sequence[int]] = None,
    layout: Optional[Sequence[int]] = None,
    itemsize: Optional[int] = None,
    bind_ip: Optional[str] = None,
    shards: Optional[Sequence[str]] = None,
    control_plane_rpc_address: Optional[str] = None,
    host_idx: Optional[int] = None,
) -> raiden_service_pb2.RegisterWorkUnitRequest:
  """Extracts endpoints/variables from ``synchronizer`` and builds a request.

  When ``host_idx`` is given, the per-host registration is recorded in
  ``host_registrations`` and merged with previously seen hosts of ``unit`` so
  that multi-host work units accumulate into one composite registration.

  Args:
    host_registrations: Per-unit, per-host registration state; updated in place
      (callers must serialize access).
    unit: Work unit being registered.
    synchronizer: Local ``WeightSynchronizer``-like object.
    variables: Explicit variable descriptors.
    global_shard_indices: Global shard index held by each local shard, from the
      synchronizer's actual placement. Defaults to the synchronizer's public
      ``global_shard_indices`` attribute; required when descriptors are
      synthesized or lack ``global_shard_indices``.
    mesh_shape: Mesh shape used to synthesize descriptors.
    layout: Layout used to synthesize descriptors.
    itemsize: Element size in bytes.
    bind_ip: IP substituted into the synchronizer's local endpoints.
    shards: Explicit data shard endpoints (skip extraction).
    control_plane_rpc_address: Explicit worker control endpoint.
    host_idx: Host index for multi-host units; ``None`` registers the unit as a
      single host and drops any accumulated per-host state.

  Returns:
    The (possibly merged) ``RegisterWorkUnitRequest`` for ``unit``.

  Raises:
    ValueError: If the global shard indices are missing or do not match the
      number of shards.
  """
  resolved_shards, resolved_ctrl = extract_synchronizer_endpoints(
      synchronizer,
      bind_ip=bind_ip,
      shards=shards,
      control_plane_rpc_address=control_plane_rpc_address,
  )
  resolved_vars, was_auto_1d = extract_synchronizer_variables(
      synchronizer,
      resolved_shards,
      variables=variables,
      global_shard_indices=global_shard_indices,
      mesh_shape=mesh_shape,
      layout=layout,
      itemsize=itemsize,
  )
  if host_idx is not None:
    host_map = host_registrations.setdefault(unit, {})
    host_map[int(host_idx)] = (
        list(resolved_shards),
        resolved_ctrl,
        resolved_vars,
        was_auto_1d,
    )
    resolved_shards, resolved_ctrl, resolved_vars = (
        merge_host_synchronizer_entries(host_map)
    )
  else:
    host_registrations.pop(unit, None)

  return build_register_work_unit_request(
      unit,
      resolved_shards,
      control_plane_rpc_address=resolved_ctrl,
      variables=resolved_vars,
      itemsize=itemsize,
  )

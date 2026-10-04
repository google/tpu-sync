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

"""Thin Python driver for the C++ V3 weight-sync controller.

All planning (striped Trainer seeding), caching, the per-transfer pull
schedulers, transfer records, worker RPC transport and the control-plane server
live in C++ (``_controller_v3.RaidenControllerV3``).
This module only:

* validates and marshals registration/transfer arguments into protos,
* exposes an eager ``sync_weights_async`` backed by a thread pool, and
* caches Python ``TransferPlan`` views for plans the C++ controller still
  retains (retention is decided solely by the C++ TTL policy).

Networking is explicit: nothing is bound or probed until ``start_server()`` is
called, and remote metadata is only fetched via ``import_remote_metadata``.
"""

import concurrent.futures
import dataclasses
import threading
from typing import Any, Callable, Optional, Sequence

from tpu_sync.api.common import RaidenId
from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager.v3 import _controller_v3
from tpu_sync.weight_sync.manager.v3 import types

NameResolver = types.NameResolver
RaidenFuture = types.RaidenFuture
TransferOptions = types.TransferOptions
TransferPlan = types.TransferPlan
TransferStatus = types.TransferStatus
WeightSyncConfig = types.WeightSyncConfig

RpcSender = Callable[[str, bytes], Optional[bytes]]


class RaidenControllerV3:
  """V3 weight-sync controller (Trainer -> striped seeds + planned pulls)."""

  def __init__(
      self,
      config: Optional[WeightSyncConfig] = None,
      *,
      name_resolver: Optional[NameResolver] = None,
      rpc_sender: Optional[RpcSender] = None,
      max_workers: int = 8,
  ):
    """Creates the controller; no network activity until ``start_server()``.

    Args:
      config: Construction-time configuration. Defaults to
        ``WeightSyncConfig()``; nothing is read from the environment unless the
        caller passes ``WeightSyncConfig.from_env()``. The property setters
        below keep ``config`` in sync with the live values.
      name_resolver: Optional ``NameResolver`` (object with ``resolve(str) ->
        str``) applied to every worker control endpoint before it is dialed,
        both by the C++ transport and by the Python-side metadata import.
      rpc_sender: Optional custom worker transport ``fn(endpoint, request_bytes)
        -> Optional[bytes]``. Replaces the built-in C++ transport for all
        worker-bound RPCs (tests inject no-op or recording senders here). When
        both ``rpc_sender`` and ``name_resolver`` are given, ``rpc_sender``
        receives the *unresolved* endpoint and is responsible for resolution.
      max_workers: Size of the thread pool executing transfers.

    Raises:
      TypeError: If ``name_resolver`` or ``rpc_sender`` has the wrong shape.
    """
    if name_resolver is not None and not callable(
        getattr(name_resolver, "resolve", None)
    ):
      raise TypeError(
          "name_resolver must implement NameResolver.resolve(str) -> str"
      )
    if rpc_sender is not None and not callable(rpc_sender):
      raise TypeError("rpc_sender must be callable")
    self._config = config or WeightSyncConfig()
    self._name_resolver = name_resolver
    self._rpc_sender = rpc_sender
    self._cpp = _controller_v3.RaidenControllerV3(
        port=self._config.port,
        broadcast_host_ratio=self._config.broadcast_host_ratio,
        num_bundle_groups=self._config.num_bundle_groups,
        max_concurrent_uploads_per_source=(
            self._config.max_concurrent_uploads_per_source
        ),
        enable_plan_cache=self._config.enable_plan_cache,
        num_stripes=self._config.num_stripes,
        seed_replication=self._config.seed_replication,
        grant_batch_size=self._config.grant_batch_size,
        lease_timeout_ms=types.seconds_to_ms(self._config.lease_timeout_s),
        long_poll_timeout_ms=types.seconds_to_ms(
            self._config.long_poll_timeout_s
        ),
        request_registry_ttl_s=self._config.request_registry_ttl_s,
        transfer_timeout_ms=types.seconds_to_ms(
            self._config.transfer_timeout_s
        ),
        py_rpc_sender=rpc_sender,
        py_endpoint_resolver=(
            name_resolver.resolve if name_resolver is not None else None
        ),
    )
    self._pipe_client: Optional[control_pipe_client.ControlPipeClient] = None
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(max_workers)),
        thread_name_prefix="raiden-v3-transfer",
    )
    self._lock = threading.Lock()
    # Python views of materialized plans, keyed by req_id. An entry is only
    # served while the C++ controller retains a plan with the same uuid.
    self._plans: dict[str, TransferPlan] = {}
    self._host_registrations: dict[
        RaidenId, dict[int, types.HostRegistration]
    ] = {}
    self._server_running = False
    self._address_override = ""
    self._closed = False

  # ---------------------------------------------------------------- lifecycle

  def start_server(self) -> int:
    """Starts the embedded C++ control-plane server; returns the bound port."""
    self._check_open()
    port = int(self._cpp.start_server())
    self._server_running = True
    return port

  def stop_server(self) -> None:
    """Stops the embedded server and waits for server-spawned transfers."""
    if self._server_running:
      self._cpp.stop_server()
      self._server_running = False

  def close(self, wait: bool = True) -> None:
    """Stops the server and the transfer executor. Idempotent.

    Does NOT send shutdown to workers; call ``shutdown_workers()`` first if
    that is desired.

    Args:
      wait: Whether to block until running transfers on the executor finish.
    """
    if self._closed:
      return
    self._closed = True
    self.stop_server()
    self._executor.shutdown(wait=wait, cancel_futures=True)
    if self._pipe_client is not None:
      self._pipe_client.close()
      self._pipe_client = None

  def __enter__(self) -> "RaidenControllerV3":
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()

  def __del__(self) -> None:
    # Safety net for callers that drop the controller without ``close()``.
    # The embedded server must be stopped *with the GIL released* (which
    # ``stop_server`` does) before the C++ object is destroyed: the C++
    # destructor runs under the GIL and joins server-spawned transfer threads,
    # which may be blocked waiting for the GIL inside a Python ``rpc_sender`` /
    # ``name_resolver`` callback.
    try:
      if getattr(self, "_cpp", None) is not None:
        self.close(wait=False)
    except Exception:  # pylint: disable=broad-except
      pass

  def _check_open(self) -> None:
    if self._closed:
      raise RuntimeError("RaidenControllerV3 is closed")

  # --------------------------------------------------------------- properties

  @property
  def config(self) -> WeightSyncConfig:
    return self._config

  @property
  def name_resolver(self) -> Optional[NameResolver]:
    return self._name_resolver

  @property
  def port(self) -> int:
    return int(self._cpp.port)

  @property
  def controller_address(self) -> str:
    """Address advertised to workers for controller callbacks.

    Returns the explicitly configured override if one was set, otherwise the
    ``host:port`` of the running server, or ``""`` if the server is not
    running.
    """
    if self._address_override:
      return self._address_override
    return str(self._cpp.controller_address) if self._server_running else ""

  @controller_address.setter
  def controller_address(self, address: str) -> None:
    """Overrides the advertised address (e.g. an externally routable VIP)."""
    self._address_override = str(address)
    self._cpp.controller_address = self._address_override

  @property
  def broadcast_host_ratio(self) -> float:
    return float(self._cpp.broadcast_host_ratio)

  @broadcast_host_ratio.setter
  def broadcast_host_ratio(self, val: float) -> None:
    if val <= 0:
      raise ValueError("broadcast_host_ratio must be > 0")
    self._cpp.broadcast_host_ratio = float(val)
    self._update_config(broadcast_host_ratio=float(val))

  @property
  def num_bundle_groups(self) -> int:
    return int(self._cpp.num_bundle_groups)

  @num_bundle_groups.setter
  def num_bundle_groups(self, val: int) -> None:
    if int(val) < 1:
      raise ValueError("num_bundle_groups must be >= 1")
    self._cpp.num_bundle_groups = int(val)
    self._update_config(num_bundle_groups=int(val))

  @property
  def max_concurrent_uploads_per_source(self) -> int:
    return int(self._cpp.max_concurrent_uploads_per_source)

  @max_concurrent_uploads_per_source.setter
  def max_concurrent_uploads_per_source(self, val: int) -> None:
    if int(val) < 1:
      raise ValueError("max_concurrent_uploads_per_source must be >= 1")
    types.check_grant_batch_size(self.grant_batch_size, val)
    self._cpp.max_concurrent_uploads_per_source = int(val)
    self._update_config(max_concurrent_uploads_per_source=int(val))

  @property
  def grant_batch_size(self) -> int:
    """Pull leases a sampler host holds at once (k_b)."""
    return int(self._cpp.grant_batch_size)

  @grant_batch_size.setter
  def grant_batch_size(self, val: int) -> None:
    types.check_grant_batch_size(val, self.max_concurrent_uploads_per_source)
    self._cpp.grant_batch_size = int(val)
    self._update_config(grant_batch_size=int(val))

  @property
  def num_stripes(self) -> int:
    """Number of Trainer seed stripes; ``0`` picks it per transfer."""
    return int(self._cpp.num_stripes)

  @num_stripes.setter
  def num_stripes(self, val: int) -> None:
    if int(val) < 0:
      raise ValueError("num_stripes must be >= 0")
    self._cpp.num_stripes = int(val)
    self._update_config(num_stripes=int(val))

  @property
  def seed_replication(self) -> int:
    """Number of destination replicas every stripe is pushed to."""
    return int(self._cpp.seed_replication)

  @seed_replication.setter
  def seed_replication(self, val: int) -> None:
    if int(val) < 1:
      raise ValueError("seed_replication must be >= 1")
    self._cpp.seed_replication = int(val)
    self._update_config(seed_replication=int(val))

  @property
  def lease_timeout_s(self) -> float:
    """How long a pull lease stays valid without being reported."""
    return int(self._cpp.lease_timeout_ms) / 1000.0

  @lease_timeout_s.setter
  def lease_timeout_s(self, val: float) -> None:
    types.check_timeout_s("lease_timeout_s", val)
    self._cpp.lease_timeout_ms = types.seconds_to_ms(val)
    self._update_config(lease_timeout_s=float(val))

  @property
  def long_poll_timeout_s(self) -> float:
    """How long a pull request that cannot be granted anything yet is held."""
    return int(self._cpp.long_poll_timeout_ms) / 1000.0

  @long_poll_timeout_s.setter
  def long_poll_timeout_s(self, val: float) -> None:
    types.check_timeout_s("long_poll_timeout_s", val)
    self._cpp.long_poll_timeout_ms = types.seconds_to_ms(val)
    self._update_config(long_poll_timeout_s=float(val))

  @property
  def transfer_timeout_s(self) -> float:
    """Default deadline of a transfer, from the start of its execution.

    Applies to transfers started afterwards without ``TransferOptions.
    timeout_s``. Changing it never invalidates plans.
    """
    return int(self._cpp.transfer_timeout_ms) / 1000.0

  @transfer_timeout_s.setter
  def transfer_timeout_s(self, val: float) -> None:
    types.check_timeout_s("transfer_timeout_s", val)
    self._cpp.transfer_timeout_ms = types.seconds_to_ms(val)
    self._update_config(transfer_timeout_s=float(val))

  @property
  def request_registry_ttl_s(self) -> float:
    return float(self._cpp.request_registry_ttl_s)

  @request_registry_ttl_s.setter
  def request_registry_ttl_s(self, val: float) -> None:
    if val < 0:
      raise ValueError("request_registry_ttl_s must be >= 0")
    self._cpp.request_registry_ttl_s = float(val)
    self._update_config(request_registry_ttl_s=float(val))
    self._prune_plans()

  def _update_config(self, **changes: Any) -> None:
    with self._lock:
      self._config = dataclasses.replace(self._config, **changes)

  # ------------------------------------------------------------- registration

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: Sequence[str],
      control_plane_rpc_address: Optional[str | Sequence[str]] = None,
      *,
      variables: Optional[Sequence[Any]] = None,
      itemsize: Optional[int] = None,
  ) -> None:
    """Registers (or updates) a work unit in the C++ ``EntityRegistry``.

    Replaces any per-host state accumulated for ``unit`` by
    ``register_synchronizer``.

    Args:
      unit: Work unit being registered.
      shards: Data shard endpoints (one per shard, all non-empty).
      control_plane_rpc_address: Worker control endpoint(s), either a sequence
        or a comma-separated string.
      variables: ``VariableMetadataProto``-like descriptors, one per weight,
        each with ``shape``, ``mesh_shape``, ``layout``, ``item_size`` and the
        required ``global_shard_indices``: one entry per shard, where entry i is
        the global shard index, in ``[0, prod(mesh_shape))``, held by
        ``shards[i]``.
      itemsize: Element size in bytes for descriptors whose ``item_size`` is 0.

    Raises:
      ValueError: On empty shards, or a descriptor without one global shard
        index per shard.
    """
    self._check_open()
    req = types.build_register_work_unit_request(
        unit,
        shards,
        control_plane_rpc_address=control_plane_rpc_address,
        variables=variables,
        itemsize=itemsize,
    )
    with self._lock:
      self._host_registrations.pop(unit, None)
      self._cpp.register_work_unit_bytes(req.SerializeToString())

  def register_synchronizer(
      self,
      unit: RaidenId,
      synchronizer: types.SynchronizerLike,
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
  ) -> None:
    """Registers ``unit`` from a local ``WeightSynchronizer``-like object.

    Pass ``host_idx`` for multi-host units; successive calls for the same unit
    are merged into one composite registration.

    Args:
      unit: Work unit being registered.
      synchronizer: Local ``WeightSynchronizer``-like object.
      variables: Explicit variable descriptors; derived from the synchronizer
        when omitted.
      global_shard_indices: Global shard index held by each local shard, from
        the synchronizer's actual placement. Defaults to the synchronizer's
        public ``global_shard_indices`` attribute; required when descriptors are
        derived or lack ``global_shard_indices``.
      mesh_shape: Mesh shape used to derive descriptors.
      layout: Layout used to derive descriptors.
      itemsize: Element size in bytes.
      bind_ip: IP substituted into the synchronizer's local endpoints.
      shards: Explicit data shard endpoints (skips extraction).
      control_plane_rpc_address: Explicit worker control endpoint.
      host_idx: Host index for multi-host units; ``None`` registers the unit as
        a single host and drops any accumulated per-host state.

    Raises:
      ValueError: If the global shard indices are missing or do not match the
        number of shards.
    """
    self._check_open()
    with self._lock:
      req = types.build_synchronizer_registration(
          self._host_registrations,
          unit,
          synchronizer,
          variables=variables,
          global_shard_indices=global_shard_indices,
          mesh_shape=mesh_shape,
          layout=layout,
          itemsize=itemsize,
          bind_ip=bind_ip,
          shards=shards,
          control_plane_rpc_address=control_plane_rpc_address,
          host_idx=host_idx,
      )
      self._cpp.register_work_unit_bytes(req.SerializeToString())

  def attach_host(
      self,
      unit: RaidenId,
      control_address: str,
      shards: Optional[Sequence[str]] = None,
  ) -> None:
    """Attaches a host control endpoint (and optional shards) to ``unit``."""
    self._check_open()
    self._cpp.attach_host_bytes(
        types.unit_to_bytes(unit),
        str(control_address),
        list(shards) if shards is not None else None,
    )

  def has_unit(self, unit: RaidenId) -> bool:
    return bool(self._cpp.has_unit_bytes(types.unit_to_bytes(unit)))

  def get_registered_units(self) -> list[RaidenId]:
    out: list[RaidenId] = []
    for raw in self._cpp.get_registered_units_bytes():
      proto = raiden_service_pb2.RaidenIdProto()
      proto.ParseFromString(bytes(raw))
      out.append(types.raiden_id_from_proto(proto))
    return out

  def get_all_metadata(
      self,
  ) -> list[raiden_service_pb2.RegisterWorkUnitRequest]:
    out: list[raiden_service_pb2.RegisterWorkUnitRequest] = []
    for raw in self._cpp.get_all_metadata_bytes():
      msg = raiden_service_pb2.RegisterWorkUnitRequest()
      msg.ParseFromString(bytes(raw))
      out.append(msg)
    return out

  def import_remote_metadata(
      self,
      controller_address: str,
      units: Optional[Sequence[RaidenId]] = None,
      timeout: float = 30.0,
  ) -> list[RaidenId]:
    """Copies work-unit registrations from another controller.

    Explicit replacement for the old implicit ``dst_controller_address``
    behaviour.

    Args:
      controller_address: Control endpoint of the controller to import from.
      units: If given, only these units are imported.
      timeout: RPC timeout in seconds.

    Returns:
      The units that were imported.

    Raises:
      RuntimeError: If the remote ``GET_METADATA`` request fails.
    """
    self._check_open()
    if self._pipe_client is None:
      self._pipe_client = control_pipe_client.ControlPipeClient(
          name_resolver=self._name_resolver
      )
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA
    )
    resp = self._pipe_client.call_sync(
        str(controller_address),
        req,
        raiden_service_pb2.ControlResponse,
        timeout=float(timeout),
    )
    if not resp.success:
      raise RuntimeError(
          f"GET_METADATA from {controller_address} failed: {resp.message}"
      )
    wanted = set(units) if units is not None else None
    imported: list[RaidenId] = []
    for meta in resp.get_metadata_response.metadata:
      unit = types.raiden_id_from_proto(meta.unit)
      if wanted is not None and unit not in wanted:
        continue
      self._cpp.register_work_unit_bytes(meta.SerializeToString())
      imported.append(unit)
    return imported

  # ---------------------------------------------------------------- transfers

  def plan_transfer(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      *,
      req_id: Optional[str] = None,
      uuid: int = 0,
      options: Optional[TransferOptions] = None,
  ) -> TransferPlan:
    """Materializes and stores a plan for later execution (no worker I/O).

    The plan uses the controller's current striped-seeding settings
    (``num_stripes``, ``seed_replication``); the pull settings
    (``grant_batch_size``, ``max_concurrent_uploads_per_source``,
    ``lease_timeout_s``, ``long_poll_timeout_s``) apply when it is executed.
    The C++ controller keeps the plan (and the pull scheduler of its last
    execution) under ``req_id`` until it is executed and then for
    ``request_registry_ttl_s``; an unexecuted plan is kept for at least 60s.

    Args:
      src_units: Source (trainer) work units.
      dst_units: Destination (sampler) work units.
      req_id: Transfer identifier; generated from ``uuid`` when omitted.
      uuid: Transfer UUID; allocated when ``<= 0``.
      options: Per-transfer options; defaults to ``TransferOptions()``.

    Returns:
      An immutable ``TransferPlan``; execute it with ``execute_plan`` or let
      ``sync_weights_async`` do both.

    Raises:
      ValueError: On empty unit lists.
      RuntimeError: If a unit is unregistered or materialization fails.
    """
    self._check_open()
    srcs, dsts = list(src_units), list(dst_units)
    if not srcs or not dsts:
      raise ValueError("src_units and dst_units must not be empty")
    opts = options or TransferOptions()
    # Missing ids are allocated by the C++ controller from the same counter it
    # uses for server-initiated transfers, so they can never collide.
    cpp_plan = self._cpp.build_materialized_plan_bytes(
        str(req_id) if req_id else "",
        int(uuid) if uuid and int(uuid) > 0 else 0,
        [types.unit_to_bytes(u) for u in srcs],
        [types.unit_to_bytes(u) for u in dsts],
        dst_mem_type=int(opts.dst_mem_type),
        skip_d2h=bool(opts.skip_d2h),
        parallelism=int(opts.parallelism),
        skip_tiling=dict(opts.skip_tiling),
        use_cached_plan=bool(opts.use_cached_plan),
    )
    plan = TransferPlan.from_cpp(cpp_plan, srcs, dsts, opts)
    with self._lock:
      self._plans[plan.req_id] = plan
    self._prune_plans()
    return plan

  def execute_plan(self, plan: TransferPlan) -> TransferPlan:
    """Synchronously executes a plan produced by ``plan_transfer``.

    The transfer must finish within ``plan.options.timeout_s`` (or the
    controller's ``transfer_timeout_s``) from the start of the execution.

    Args:
      plan: The plan to execute. The C++ controller must still retain it (same
        ``req_id`` and ``uuid``).

    Returns:
      ``plan``.

    Raises:
      RuntimeError: If the plan is no longer retained (evicted or replaced by
        a newer plan for the same ``req_id``), or if the transfer fails,
        including with ``DEADLINE_EXCEEDED`` at its transfer timeout.
    """
    self._check_open()
    timeout_s = plan.options.timeout_s
    self._cpp.execute_materialized_transfer_sync_bytes(
        plan.req_id,
        expected_uuid=plan.uuid,
        transfer_timeout_ms=(
            types.seconds_to_ms(timeout_s) if timeout_s is not None else None
        ),
    )
    return plan

  def sync_weights_async(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      *,
      req_id: Optional[str] = None,
      uuid: int = 0,
      options: Optional[TransferOptions] = None,
  ) -> RaidenFuture:
    """Plans and immediately starts a transfer on the controller executor.

    The returned ``RaidenFuture`` is eager: the transfer runs to completion
    whether or not the caller waits on it. Planning errors are raised
    synchronously; execution errors surface from ``wait``/``result``.

    Args:
      src_units: Source (trainer) work units.
      dst_units: Destination (sampler) work units.
      req_id: Transfer identifier; generated from ``uuid`` when omitted.
      uuid: Transfer UUID; allocated when ``<= 0``.
      options: Per-transfer options; defaults to ``TransferOptions()``.

    Returns:
      A ``RaidenFuture`` resolving to the executed ``TransferPlan``.
    """
    plan = self.plan_transfer(
        src_units, dst_units, req_id=req_id, uuid=uuid, options=options
    )
    fut = self._executor.submit(self.execute_plan, plan)
    return RaidenFuture(fut, plan.req_id)

  def sync_weights(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      *,
      req_id: Optional[str] = None,
      uuid: int = 0,
      options: Optional[TransferOptions] = None,
      timeout: Optional[float] = None,
  ) -> TransferPlan:
    """Blocking ``sync_weights_async``.

    Args:
      src_units: Source (trainer) work units.
      dst_units: Destination (sampler) work units.
      req_id: Transfer identifier; generated from ``uuid`` when omitted.
      uuid: Transfer UUID; allocated when ``<= 0``.
      options: Per-transfer options; defaults to ``TransferOptions()``.
      timeout: How long to wait in seconds; ``None`` waits until the transfer
        finishes. A timeout only stops waiting (``TimeoutError``); it does not
        cancel the transfer, which is bounded by its transfer timeout
        (``options.timeout_s`` or ``transfer_timeout_s``) instead.

    Returns:
      The executed ``TransferPlan``.
    """
    return self.sync_weights_async(
        src_units, dst_units, req_id=req_id, uuid=uuid, options=options
    ).wait_threadsafe(timeout=timeout)

  def get_plan(self, req_id: str) -> Optional[TransferPlan]:
    """Returns the plan for ``req_id`` while the C++ controller retains it."""
    self._prune_plans()
    with self._lock:
      return self._plans.get(str(req_id))

  def get_transfer_status(self, req_id: str) -> TransferStatus:
    return TransferStatus(int(self._cpp.get_transfer_status(str(req_id))))

  def wait_for_transfer(
      self, req_id: str, timeout: Optional[float] = None
  ) -> None:
    """Waits for a (possibly server-initiated) transfer to reach a terminal state.

    Args:
      req_id: Transfer identifier.
      timeout: How long to wait in seconds. ``None`` follows the transfer's
        deadline: it waits until shortly after it, by which time a transfer that
          did not complete has failed with ``DEADLINE_EXCEEDED``. A timeout does
          not cancel the transfer.

    Raises:
      RuntimeError: If the transfer failed, is unknown, or the wait timed out.
    """
    self._cpp.wait_for_transfer(
        str(req_id), float(timeout) if timeout is not None else None
    )

  def _prune_plans(self) -> None:
    """Drops plan views whose C++ plan was evicted or replaced."""
    retained = self._cpp.get_retained_plan_uuids()
    with self._lock:
      for key in [
          k for k, p in self._plans.items() if retained.get(k) != p.uuid
      ]:
        del self._plans[key]

  # --------------------------------------------------------- plan cache stats

  def clear_plan_cache(self) -> None:
    self._cpp.clear_plan_cache()

  def get_plan_cache_size(self) -> int:
    return int(self._cpp.get_plan_cache_size())

  def get_plan_materialization_count(self) -> int:
    return int(self._cpp.get_plan_materialization_count())

  def get_transfer_record_count(self) -> int:
    return int(self._cpp.get_transfer_record_count())

  # ------------------------------------------------------------ worker control

  def shutdown_workers(self) -> None:
    """Sends ``COMMAND_SHUTDOWN`` to every registered worker control endpoint.

    Routed through the C++ controller so the configured ``rpc_sender`` /
    ``name_resolver`` transport is honored.

    Raises:
      RuntimeError: If the C++ controller reports a failure.
    """
    self._check_open()
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_SHUTDOWN
    )
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(
        bytes(self._cpp.handle_control_request_bytes(req.SerializeToString()))
    )
    if not resp.success:
      raise RuntimeError(f"shutdown_workers failed: {resp.message}")

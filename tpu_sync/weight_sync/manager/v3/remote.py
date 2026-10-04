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

"""Remote client facade for a running V3 controller server.

Wire contract (HEAD protos only; no fields are invented):

* Registration: ``ControlRequest.COMMAND_REGISTER_WORK_UNIT``.
* Kick-off: ``ControlRequest.COMMAND_START_TRANSFER`` carrying a
  ``StartTransferRequest`` (``req_id``, ``uuid``, ``dst_mem_type``,
  ``skip_d2h``, ``skip_tiling``, ``parallelism``). The server starts the
  transfer **asynchronously** and returns immediately. ``req_id``/``uuid`` are
  generated client-side when omitted so the client can always poll status.
* Progress: ``ControllerRequest.COMMAND_GET_TRANSFER_STATUS`` polled until
  ``COMPLETED``/``FAILED``. The HEAD status proto carries no error message, so a
  remote failure surfaces as ``RuntimeError`` without a cause string.

Options the HEAD protos cannot carry (``use_cached_plan=False``, ``timeout_s``)
are rejected with ``ValueError`` instead of being silently dropped. Striped-
seeding settings (``num_stripes``, ``seed_replication``), the pull settings
(``grant_batch_size``, ``max_concurrent_uploads_per_source``,
``lease_timeout_s``, ``long_poll_timeout_s``) and the transfer timeout
(``transfer_timeout_s``) are controller-wide and configured on the serving
controller, whose transfer timeout bounds every transfer started through this
facade.
"""

import concurrent.futures
import threading
import time
from typing import Any, Optional, Sequence
import uuid as uuid_lib

from tpu_sync.api.common import RaidenId
from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.rpc import controller_service_pb2
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager.v3 import types

NameResolver = types.NameResolver
RaidenFuture = types.RaidenFuture
TransferOptions = types.TransferOptions
TransferStatus = types.TransferStatus


def _resolve_transfer_ids(
    req_id: Optional[str], uuid: Optional[int]
) -> tuple[str, int]:
  """Returns ``(req_id, uuid)``, generating whichever is unset.

  Args:
    req_id: Transfer id; ``None``/empty generates ``f"req_{uuid}"``.
    uuid: Transfer uuid; ``None``/``0`` generates a random positive 62-bit id.

  Returns:
    The effective ``(req_id, uuid)``.

  Raises:
    ValueError: If ``uuid`` is negative.
  """
  eff_uuid = int(uuid or 0)
  if eff_uuid < 0:
    raise ValueError(f"uuid must be >= 0, got {uuid}")
  if eff_uuid == 0:
    eff_uuid = (uuid_lib.uuid4().int >> 66) + 1
  eff_req_id = str(req_id) if req_id else f"req_{eff_uuid}"
  return eff_req_id, eff_uuid


def _check_carryable(options: TransferOptions) -> None:
  """Raises ``ValueError`` for options the HEAD wire proto cannot carry."""
  if not options.use_cached_plan:
    raise ValueError(
        "use_cached_plan=False cannot be sent over the HEAD"
        " StartTransferRequest proto; call clear_plan_cache() on the"
        " controller instead."
    )
  if options.timeout_s is not None:
    raise ValueError(
        "timeout_s cannot be sent over the HEAD StartTransferRequest proto;"
        " the serving controller's transfer_timeout_s applies. Set it on that"
        " controller, or pass `timeout` to bound only the wait."
    )


class RemoteTransferResult:
  """Outcome of a facade-driven transfer (``req_id``, ``uuid``, final status)."""

  __slots__ = ("req_id", "uuid", "status")

  def __init__(self, req_id: str, uuid: int, status: TransferStatus):
    self.req_id = req_id
    self.uuid = uuid
    self.status = status

  def __repr__(self) -> str:
    return (
        f"RemoteTransferResult(req_id={self.req_id!r}, uuid={self.uuid},"
        f" status={self.status.name})"
    )


class RaidenControllerClientFacade:
  """Client stub for a remote ``RaidenControllerV3`` server."""

  def __init__(
      self,
      controller_address: str,
      name_resolver: Optional[NameResolver] = None,
      *,
      rpc_timeout: float = 300.0,
      poll_interval: float = 0.05,
  ):
    """Creates the facade; no RPC is issued until a method is called.

    Args:
      controller_address: ``host:port`` of the controller server.
      name_resolver: Optional ``NameResolver`` applied to
        ``controller_address``.
      rpc_timeout: Per-RPC deadline in seconds.
      poll_interval: Status polling period used by ``sync_weights``.

    Raises:
      TypeError: If ``name_resolver`` does not implement ``resolve``.
    """
    if name_resolver is not None and not callable(
        getattr(name_resolver, "resolve", None)
    ):
      raise TypeError(
          "name_resolver must implement NameResolver.resolve(str) -> str"
      )
    self._address = str(controller_address)
    self._rpc_timeout = float(rpc_timeout)
    self._poll_interval = max(0.001, float(poll_interval))
    self._client = control_pipe_client.ControlPipeClient(
        name_resolver=name_resolver
    )
    self._registration_lock = threading.Lock()
    self._host_registrations: dict[
        RaidenId, dict[int, types.HostRegistration]
    ] = {}
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="raiden-v3-facade"
    )

  @property
  def controller_address(self) -> str:
    return self._address

  def close(self) -> None:
    self._executor.shutdown(wait=False, cancel_futures=True)
    close_fn = getattr(self._client, "close", None)
    if callable(close_fn):
      close_fn()

  def __enter__(self) -> "RaidenControllerClientFacade":
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()

  # ------------------------------------------------------------------- wire

  def _control(
      self, req: raiden_service_pb2.ControlRequest
  ) -> raiden_service_pb2.ControlResponse:
    """Sends a ``ControlRequest`` and raises ``RuntimeError`` on rejection."""
    resp = self._client.call_sync(
        self._address,
        req,
        raiden_service_pb2.ControlResponse,
        timeout=self._rpc_timeout,
    )
    if not resp.success:
      raise RuntimeError(
          f"Controller {self._address} rejected"
          f" {raiden_service_pb2.ControlRequest.Command.Name(req.command)}:"
          f" {resp.message}"
      )
    return resp

  def _controller(
      self, req: controller_service_pb2.ControllerRequest
  ) -> controller_service_pb2.ControllerResponse:
    """Sends a ``ControllerRequest`` and raises ``RuntimeError`` on rejection."""
    resp = self._client.call_sync(
        self._address,
        req,
        controller_service_pb2.ControllerResponse,
        timeout=self._rpc_timeout,
    )
    if not resp.success:
      raise RuntimeError(
          f"Controller {self._address} rejected"
          f" {controller_service_pb2.ControllerRequest.Command.Name(req.command)}:"
          f" {resp.message}"
      )
    return resp

  # ----------------------------------------------------------- registration

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: Sequence[str],
      control_plane_rpc_address: Optional[str | Sequence[str]] = None,
      *,
      variables: Optional[Sequence[Any]] = None,
      itemsize: Optional[int] = None,
  ) -> None:
    """Registers a work unit on the remote controller.

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
      RuntimeError: If the controller rejects the registration.
    """
    reg = types.build_register_work_unit_request(
        unit,
        shards,
        control_plane_rpc_address=control_plane_rpc_address,
        variables=variables,
        itemsize=itemsize,
    )
    with self._registration_lock:
      self._host_registrations.pop(unit, None)
    self._control(
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_REGISTER_WORK_UNIT,
            register_work_unit_request=reg,
        )
    )

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
    """Registers ``unit`` from a local synchronizer on the remote controller.

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
      RuntimeError: If the controller rejects the registration.
    """
    with self._registration_lock:
      reg = types.build_synchronizer_registration(
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
    self._control(
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_REGISTER_WORK_UNIT,
            register_work_unit_request=reg,
        )
    )

  def get_metadata(self) -> list[raiden_service_pb2.RegisterWorkUnitRequest]:
    resp = self._control(
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA
        )
    )
    return list(resp.get_metadata_response.metadata)

  # -------------------------------------------------------------- transfers

  def start_transfer_async(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      *,
      req_id: Optional[str] = None,
      uuid: Optional[int] = None,
      options: Optional[TransferOptions] = None,
  ) -> tuple[str, int]:
    """Asks the remote controller to start a transfer and returns immediately.

    Args:
      src_units: Source (trainer) work units.
      dst_units: Destination (sampler) work units.
      req_id: Transfer id; generated as ``f"req_{uuid}"`` when omitted.
      uuid: Transfer uuid; a random positive id is generated when omitted/0.
      options: Per-transfer options; defaults to ``TransferOptions()``.

    Returns:
      The effective ``(req_id, uuid)`` to poll with ``wait_for_transfer``.

    Raises:
      ValueError: If ``options`` contains fields the wire proto cannot carry,
        ``uuid`` is negative, or a unit list is empty.
      RuntimeError: If the controller rejects the request.
    """
    opts = options or TransferOptions()
    _check_carryable(opts)
    eff_req_id, eff_uuid = _resolve_transfer_ids(req_id, uuid)
    if not src_units or not dst_units:
      raise ValueError("src_units and dst_units must not be empty")
    start = raiden_service_pb2.StartTransferRequest(
        src_units=[types.raiden_id_to_proto(u) for u in src_units],
        dst_units=[types.raiden_id_to_proto(u) for u in dst_units],
        uuid=eff_uuid,
        req_id=eff_req_id,
        is_sender=True,
        dst_mem_type=int(opts.dst_mem_type),
        skip_d2h=bool(opts.skip_d2h),
        parallelism=int(opts.parallelism),
    )
    for layer_idx, skip in opts.skip_tiling.items():
      start.skip_tiling[int(layer_idx)] = bool(skip)
    self._control(
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=start,
        )
    )
    return eff_req_id, eff_uuid

  def get_transfer_status(self, req_id: str, uuid: int = 0) -> TransferStatus:
    """Returns the controller-side status of ``req_id``."""
    resp = self._controller(
        controller_service_pb2.ControllerRequest(
            command=controller_service_pb2.ControllerRequest.COMMAND_GET_TRANSFER_STATUS,
            get_transfer_status_request=(
                controller_service_pb2.GetTransferStatusRequest(
                    req_id=str(req_id), uuid=int(uuid)
                )
            ),
        )
    )
    return TransferStatus(int(resp.get_transfer_status_response.status))

  def wait_for_transfer(
      self, req_id: str, uuid: int = 0, timeout: Optional[float] = None
  ) -> RemoteTransferResult:
    """Polls status until terminal.

    Args:
      req_id: Transfer id.
      uuid: Transfer uuid (informational on the HEAD wire).
      timeout: Overall deadline in seconds; ``None`` waits until the transfer is
        terminal, which the serving controller's transfer timeout bounds.

    Returns:
      The ``RemoteTransferResult`` of the completed transfer.

    Raises:
      RuntimeError: If the remote transfer reports ``FAILED`` (the HEAD proto
        carries no failure message) or its record was evicted before
        completion (``NOT_STARTED`` after having been started).
      TimeoutError: If ``timeout`` elapses; the remote transfer keeps running.
    """
    return self._poll_until_terminal(req_id, uuid, timeout, started=False)

  def _poll_until_terminal(
      self,
      req_id: str,
      uuid: int,
      timeout: Optional[float],
      *,
      started: bool,
  ) -> RemoteTransferResult:
    """Polls ``req_id`` to a terminal state.

    Args:
      req_id: Transfer id.
      uuid: Transfer uuid (informational on the HEAD wire).
      timeout: Overall deadline in seconds; ``None`` waits forever.
      started: True when this client has already received a successful
        ``START_TRANSFER`` ack. The controller marks the record ``IN_PROGRESS``
        before acking, so any later ``NOT_STARTED`` can only mean the record was
        evicted (``request_registry_ttl_s`` too short).

    Returns:
      The ``RemoteTransferResult`` of the completed transfer.

    Raises:
      RuntimeError: If the transfer fails or its record disappears.
      TimeoutError: If ``timeout`` elapses.
    """
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    seen_in_progress = started
    while True:
      status = self.get_transfer_status(req_id, uuid)
      if status == TransferStatus.COMPLETED:
        return RemoteTransferResult(str(req_id), int(uuid), status)
      if status == TransferStatus.FAILED:
        raise RuntimeError(
            f"Remote transfer {req_id!r} failed on controller"
            f" {self._address} (no failure detail is carried by the HEAD"
            " GetTransferStatusResponse proto; check the controller log)."
        )
      if status == TransferStatus.IN_PROGRESS:
        seen_in_progress = True
      elif seen_in_progress:
        raise RuntimeError(
            f"Remote transfer {req_id!r} record disappeared before its"
            " completion could be observed (controller"
            " request_registry_ttl_s too short for the poll interval?)"
        )
      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for remote transfer"
            f" {req_id!r}; it is still running on the controller."
        )
      time.sleep(self._poll_interval)

  def sync_weights(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      *,
      req_id: Optional[str] = None,
      uuid: Optional[int] = None,
      options: Optional[TransferOptions] = None,
      timeout: Optional[float] = None,
  ) -> RemoteTransferResult:
    """Starts a remote transfer and blocks until it completes.

    Args:
      src_units: Source (trainer) work units.
      dst_units: Destination (sampler) work units.
      req_id: Transfer id; generated as ``f"req_{uuid}"`` when omitted.
      uuid: Transfer uuid; a random positive id is generated when omitted/0.
      options: Per-transfer options; defaults to ``TransferOptions()``.
      timeout: Bound on status polling in seconds; the remote transfer keeps
        running after a timeout. ``None`` waits until the transfer is terminal,
        which the serving controller's transfer timeout bounds.

    Returns:
      The ``RemoteTransferResult`` with the effective ``req_id``/``uuid``.

    Raises:
      ValueError: As for ``start_transfer_async``.
      RuntimeError: If the transfer is rejected, fails, or its record is
        evicted before completion is observed.
      TimeoutError: If ``timeout`` elapses.
    """
    eff_req_id, eff_uuid = self.start_transfer_async(
        src_units, dst_units, req_id=req_id, uuid=uuid, options=options
    )
    return self._poll_until_terminal(
        eff_req_id, eff_uuid, timeout, started=True
    )

  def sync_weights_async(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      *,
      req_id: Optional[str] = None,
      uuid: Optional[int] = None,
      options: Optional[TransferOptions] = None,
      timeout: Optional[float] = None,
  ) -> RaidenFuture:
    """Eagerly starts a remote transfer; the future polls it to completion.

    Args:
      src_units: Source (trainer) work units.
      dst_units: Destination (sampler) work units.
      req_id: Transfer id; generated as ``f"req_{uuid}"`` when omitted.
      uuid: Transfer uuid; a random positive id is generated when omitted/0.
      options: Per-transfer options; defaults to ``TransferOptions()``.
      timeout: Bound on the future's status polling in seconds.

    Returns:
      A ``RaidenFuture`` resolving to the ``RemoteTransferResult``; its
      ``req_id`` is the effective (possibly generated) transfer id. Start-up
      errors (``ValueError``/``RuntimeError``) are raised synchronously.
    """
    eff_req_id, eff_uuid = self.start_transfer_async(
        src_units, dst_units, req_id=req_id, uuid=uuid, options=options
    )
    fut = self._executor.submit(
        self._poll_until_terminal, eff_req_id, eff_uuid, timeout, started=True
    )
    return RaidenFuture(fut, eff_req_id)

  # ----------------------------------------------------------------- control

  def shutdown_workers(self) -> None:
    """Asks the controller to send ``COMMAND_SHUTDOWN`` to all workers."""
    self._control(
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_SHUTDOWN
        )
    )

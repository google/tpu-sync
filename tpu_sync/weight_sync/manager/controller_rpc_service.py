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

"""Controller RPC server and client facade for RaidenController."""

import asyncio
import threading
from typing import Any, Optional, Sequence

from tpu_sync.api.common import RaidenId
from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.rpc import controller_service_pb2
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types

NameResolver = controller_types.NameResolver
RaidenMemoryType = controller_types.RaidenMemoryType
_coerce_pool_spec_proto = controller_types.coerce_pool_spec_proto
_coerce_variable_proto = controller_types.coerce_variable_proto
_raiden_id_from_proto = controller_types.raiden_id_from_proto


class RaidenControllerServer:
  """Centralized Control-Plane network servicer backed by ControlPipeServer."""

  def __init__(
      self,
      controller: Any,
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
    self._server = control_pipe_client.ControlPipeServer(
        controller.port, self._handle_request
    )
    if controller.port == 0:
      controller.port = self._server.port

  @property
  def port(self) -> int:
    """Returns the bound listener port, including for a requested port 0."""
    return self._controller.port

  @property
  def _thread(self) -> Optional[threading.Thread]:
    return self._server.thread

  def start(self) -> int:
    """Spawns background server acceptance thread listening for incoming Controller RPCs.

    Returns:
      Active TCP listener port coordinate.
    """
    return self._server.start()

  def stop(self) -> None:
    """Signals servicer loop shutdown and unblocks pending accept state."""
    self._server.stop()

  def _handle_request(self, req_bytes: bytes) -> bytes:
    """Executes deserialized ControllerRequest or ControlRequest Protobuf RPC payloads."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

      req = self._proto_module.ControllerRequest()
      try:
        req.ParseFromString(req_bytes)
      except Exception:  # pylint: disable=broad-except
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
        except Exception as e:  # pylint: disable=broad-except
          resp.message = str(e)
        return resp.SerializeToString()
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
        return resp.SerializeToString()
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
        return resp.SerializeToString()
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
        return resp.SerializeToString()
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
        return resp.SerializeToString()
      elif req.command == (
          self._proto_module.ControllerRequest.COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED
      ) and req.HasField("cancel_request_blocks_if_unclaimed_request"):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        resp.message = (
            "the Python request-block cancellation surface is unavailable; "
            "use the C++ reshard store"
        )
        return resp.SerializeToString()
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
            host_subgrid = list(reg.host_subgrid) if reg.host_subgrid else None

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
                host_subgrid=host_subgrid,
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
              entries = []
              for entry in schedule_proto.entries:
                if entry.dst_peers:
                  peers = list(entry.dst_peers)
                elif entry.dst_peer:
                  peers = [entry.dst_peer]
                else:
                  peers = []
                for peer in peers:
                  entries.append((
                      peer,
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
                  ))
              return entries

            if start_req.transfer_pool_indices or start_req.pool_groups:
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
            if hasattr(self._controller, "shutdown_entities"):
              asyncio.run(self._controller.shutdown_entities())
            self.stop()
            raiden_resp.success = True
        except Exception as e:  # pylint: disable=broad-except
          raiden_resp.message = str(e)
        return raiden_resp.SerializeToString()
    except Exception:  # pylint: disable=broad-except
      return b""
    finally:
      loop.close()


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
    self._control_pipe_client = control_pipe_client.ControlPipeClient(
        name_resolver=name_resolver
    )

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
    """Helper method to serialize and send an RPC Protobuf via ControlPipeClient."""
    resp = self._control_pipe_client.call_sync(
        self._address, req, self._proto_module.ControllerResponse, timeout=300.0
    )
    if not resp.success:
      raise RuntimeError(
          f"Remote Controller Server execution failed: {resp.message}"
      )
    return resp

  def _send_raiden_protobuf_rpc_response(self, req: Any) -> Any:
    """Sends a Raiden protobuf RPC response via ControlPipeClient."""
    resp = self._control_pipe_client.call_sync(
        self._address,
        req,
        self._raiden_proto_module.ControlResponse,
        timeout=300.0,
    )
    if not resp.success:
      raise RuntimeError(
          f"Remote Controller Server execution failed: {resp.message}"
      )
    return resp

  def _send_raiden_protobuf_rpc(self, req: Any) -> bool:
    self._send_raiden_protobuf_rpc_response(req)
    return True

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
      host_subgrid: Optional[Sequence[int]] = None,
  ) -> None:
    """Sends remote RPC to register a physical worker entity with the central RaidenControllerServer."""
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
      for v in variables:
        reg_req.variables.add().CopyFrom(
            _coerce_variable_proto(v, self._raiden_proto_module)
        )
    if mesh_axes is not None:
      reg_req.mesh_axes.extend(mesh_axes)
    if host_subgrid is not None:
      reg_req.host_subgrid.extend(host_subgrid)

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
      shard_push_schedules: Optional[dict[Any, Any]] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      dst_device_block_ids: Optional[Sequence[int]] = None,
      num_tokens: Optional[int] = None,
      transfer_pool_tags: Optional[Sequence[str]] = None,
      dst_block_counts: Optional[Sequence[int]] = None,
  ) -> bool:
    """Sends remote RPC to coordinate global least-loaded transfer and blocks until fully complete."""
    del shard_push_schedules
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
      shard_push_schedules: Optional[dict[Any, Any]] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      skip_d2h: bool = False,
      skip_tiling: Optional[dict[int, bool]] = None,
  ) -> bool:
    """Inter-controller RPC to register computed push schedules and prepare receivers."""
    del dst_controller_address, src_controller_address
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
            entry_proto.dst_peers.append(dst_peer)
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
    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_GET_METADATA
    )
    resp = self._send_raiden_protobuf_rpc_response(req)
    return list(resp.get_metadata_response.metadata)

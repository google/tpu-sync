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

"""Unit tests for tree broadcast weight sync with relay hop verification.

Validates All-Source Binomial Tree broadcast parity across multi-stage
pipelined weight transfers from a TP=2 Pathways-proxy trainer to TP=1
samplers with replicated dimensions, testing 1-hop and multi-hop relays.
"""

import asyncio
import dataclasses
import math
import os
import socket
import threading
import time
from typing import Any, Sequence

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np

from tpu_sync.api.common import RaidenId
from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

_TRANSFER_TIMEOUT_SECS = flags.DEFINE_float(
    "transfer_timeout_secs",
    20.0,
    "Maximum seconds to wait for controller future and destination completion.",
)


@dataclasses.dataclass(frozen=True)
class Scaled35BVarSpec:
  """Specification for a scaled-down 35B model parameter."""

  name: str
  shape: tuple[int, ...]
  trainer_sharding: list[str]
  sampler_sharding: list[str]
  item_size: int = 2


def _make_scaled_35b_specs() -> list[Scaled35BVarSpec]:
  """Generates scaled-down 35B model specs matching MLPerf recipe sharding."""
  return [
      # 1. Attn Q projection (emb, heads, head_dim)
      Scaled35BVarSpec(
          name="decoder.layers.0.attention.query.kernel",
          shape=(64, 4, 32),
          trainer_sharding=["fsdp,context", "tensor", ""],
          sampler_sharding=["", "expert", ""],
          item_size=2,
      ),
      # 2. Attn K projection
      Scaled35BVarSpec(
          name="decoder.layers.0.attention.key.kernel",
          shape=(64, 2, 32),
          trainer_sharding=["fsdp,context", "tensor", ""],
          sampler_sharding=["", "expert", ""],
          item_size=2,
      ),
      # 3. Attn V projection
      Scaled35BVarSpec(
          name="decoder.layers.0.attention.value.kernel",
          shape=(64, 2, 32),
          trainer_sharding=["fsdp,context", "tensor", ""],
          sampler_sharding=["", "expert", ""],
          item_size=2,
      ),
      # 4. Attn Out projection
      Scaled35BVarSpec(
          name="decoder.layers.0.attention.out.kernel",
          shape=(4, 32, 64),
          trainer_sharding=["tensor", "", "fsdp,context"],
          sampler_sharding=["expert", "", ""],
          item_size=2,
      ),
      # 5. GDN in_proj_qkvz
      Scaled35BVarSpec(
          name="decoder.layers.0.linear_attention.in_proj_qkvz.kernel",
          shape=(64, 4, 32),
          trainer_sharding=["fsdp,context", "tensor", ""],
          sampler_sharding=["", "expert", ""],
          item_size=2,
      ),
      # 6. GDN in_proj_ba
      Scaled35BVarSpec(
          name="decoder.layers.0.linear_attention.in_proj_ba.kernel",
          shape=(64, 2, 32),
          trainer_sharding=["fsdp,context", "", "tensor"],
          sampler_sharding=["", "", "expert"],
          item_size=2,
      ),
      # 7. GDN out_proj
      Scaled35BVarSpec(
          name="decoder.layers.0.linear_attention.out_proj.kernel",
          shape=(4, 32, 64),
          trainer_sharding=["tensor", "", "fsdp,context"],
          sampler_sharding=["expert", "", ""],
          item_size=2,
      ),
      # 8. MoE Router Gate
      Scaled35BVarSpec(
          name="decoder.layers.0.router.gate.kernel",
          shape=(64, 8),
          trainer_sharding=["fsdp,context", ""],
          sampler_sharding=["", ""],
          item_size=2,
      ),
      # 9. Fused Experts wi (num_experts, emb, 2*hidden)
      Scaled35BVarSpec(
          name="decoder.layers.0.moe.experts.wi.kernel",
          shape=(8, 64, 128),
          trainer_sharding=["", "fsdp,context", "tensor"],
          sampler_sharding=["expert", "", ""],
          item_size=2,
      ),
      # 10. Fused Experts wo (num_experts, hidden, emb)
      Scaled35BVarSpec(
          name="decoder.layers.0.moe.experts.wo.kernel",
          shape=(8, 64, 64),
          trainer_sharding=["", "tensor", "fsdp,context"],
          sampler_sharding=["expert", "", ""],
          item_size=2,
      ),
      # 11. Shared Expert wi_0 (emb, hidden)
      Scaled35BVarSpec(
          name="decoder.layers.0.shared_expert.wi_0.kernel",
          shape=(64, 64),
          trainer_sharding=["fsdp,context", "tensor"],
          sampler_sharding=["", "expert"],
          item_size=2,
      ),
      # 12. Shared Expert wi_1 (emb, hidden)
      Scaled35BVarSpec(
          name="decoder.layers.0.shared_expert.wi_1.kernel",
          shape=(64, 64),
          trainer_sharding=["fsdp,context", "tensor"],
          sampler_sharding=["", "expert"],
          item_size=2,
      ),
      # 13. Shared Expert wo (hidden, emb)
      Scaled35BVarSpec(
          name="decoder.layers.0.shared_expert.wo.kernel",
          shape=(64, 64),
          trainer_sharding=["tensor", "fsdp,context"],
          sampler_sharding=["expert", ""],
          item_size=2,
      ),
      # 14. Shared Expert Gate (emb, 1) - float32
      Scaled35BVarSpec(
          name="decoder.layers.0.shared_expert_gate.kernel",
          shape=(64, 1),
          trainer_sharding=["fsdp,context", ""],
          sampler_sharding=["", ""],
          item_size=4,
      ),
  ]


def _build_35b_variable_protos(
    specs: list[Scaled35BVarSpec],
    mesh_shape_dict: dict[str, int],
    is_trainer: bool,
) -> list[raiden_service_pb2.VariableMetadataProto]:
  """Constructs VariableMetadataProto descriptors with explicit sharding specs."""
  mesh_items = list(mesh_shape_dict.items())
  total_shards = math.prod(size for _, size in mesh_items)
  phys_coords_per_shard = []
  for j in range(total_shards):
    rem = j
    phys_coords = {}
    for axis, size in reversed(mesh_items):
      phys_coords[axis] = rem % size
      rem //= size
    phys_coords_per_shard.append(phys_coords)

  protos = []
  for l, item in enumerate(specs):
    sharding = item.trainer_sharding if is_trainer else item.sampler_sharding
    sharding_shape = []
    for axis in sharding:
      if not axis:
        sharding_shape.append(1)
      elif "," in axis:
        sub_axes = [a.strip() for a in axis.split(",") if a.strip()]
        sharding_shape.append(math.prod(mesh_shape_dict[a] for a in sub_axes))
      else:
        sharding_shape.append(mesh_shape_dict[axis])
    layout = list(range(len(item.shape) - 1, -1, -1))
    global_shard_indices = []
    for phys_coords in phys_coords_per_shard:
      slice_idx = 0
      for dim_mesh_size, axis_spec in zip(sharding_shape, sharding):
        coord = 0
        if axis_spec:
          for sub_axis in axis_spec.split(","):
            axis_name = sub_axis.strip()
            if not axis_name:
              continue
            coord = coord * mesh_shape_dict[axis_name] + phys_coords[axis_name]
        slice_idx = slice_idx * dim_mesh_size + coord
      global_shard_indices.append(slice_idx)
    protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=item.name,
            shape=list(item.shape),
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=item.item_size,
            layer_idx=l,
            sharding_spec=sharding,
            global_shard_indices=global_shard_indices,
        )
    )
  return protos


def _calculate_shard_byte_sizes(
    specs: list[Scaled35BVarSpec],
    mesh_shape_dict: dict[str, int],
    is_trainer: bool,
) -> list[int]:
  """Computes per-layer host buffer capacity for one shard."""
  sizes = []
  for item in specs:
    sharding = item.trainer_sharding if is_trainer else item.sampler_sharding
    sharding_shape = []
    for axis in sharding:
      if not axis:
        sharding_shape.append(1)
      elif "," in axis:
        sub_axes = [a.strip() for a in axis.split(",") if a.strip()]
        sharding_shape.append(math.prod(mesh_shape_dict[a] for a in sub_axes))
      else:
        sharding_shape.append(mesh_shape_dict[axis])
    shard_elements = int(np.prod(item.shape) // np.prod(sharding_shape))
    sizes.append(shard_elements * item.item_size)
  return sizes


def _slice_global_array(
    global_arr: np.ndarray,
    mesh_axes: list[str],
    mesh_shape: tuple[int, ...],
    sharding_spec: list[str],
    coords: dict[str, int],
) -> np.ndarray:
  """Extracts the slice of global_arr belonging to a worker with coords."""
  slices = []
  for dim_idx, axis in enumerate(sharding_spec):
    dim_len = global_arr.shape[dim_idx]
    if not axis:
      slices.append(slice(0, dim_len))
    elif "," in axis:
      sub_axes = [a.strip() for a in axis.split(",") if a.strip()]
      sub_sizes = [mesh_shape[mesh_axes.index(a)] for a in sub_axes]
      total_parts = math.prod(sub_sizes)
      part_idx = 0
      for sa in sub_axes:
        sa_size = mesh_shape[mesh_axes.index(sa)]
        part_idx = part_idx * sa_size + coords[sa]
      chunk_len = dim_len // total_parts
      start = part_idx * chunk_len
      slices.append(slice(start, start + chunk_len))
    else:
      num_parts = mesh_shape[mesh_axes.index(axis)]
      chunk_len = dim_len // num_parts
      part_idx = coords[axis]
      start = part_idx * chunk_len
      slices.append(slice(start, start + chunk_len))
  return np.ascontiguousarray(global_arr[tuple(slices)])


def _fill_trainer_and_compute_expected_35b(
    specs: list[Scaled35BVarSpec],
    trainer_mesh_axes: list[str],
    trainer_mesh_shape: tuple[int, ...],
    trainer_host_shards: list[list[int]],
    ws_src_list: list[weight_synchronizer.WeightSynchronizer],
    sampler_mesh_axes: list[str],
    sampler_mesh_shape: tuple[int, ...],
    seed: int = 42,
) -> list[list[np.ndarray]]:
  """Fills trainer host buffers and computes expected arrays per sampler shard."""
  total_trainer_shards = int(np.prod(trainer_mesh_shape))
  expected_by_layer: list[list[np.ndarray]] = []

  for l, item in enumerate(specs):
    num_elem = int(np.prod(item.shape))
    if item.item_size == 4:
      global_arr = (
          (np.arange(num_elem, dtype=np.float32) + 1.0) * 0.125 * (l + 1)
      ).reshape(item.shape)
    else:
      seed_u16 = np.uint16((seed + l * 7) & 0xFF) << np.uint16(8)
      layer_tag = np.uint16((l + 1) & 0x0F) << np.uint16(4)
      offsets = np.arange(num_elem, dtype=np.uint16) & np.uint16(0x000F)
      global_arr = (seed_u16 | layer_tag | offsets).reshape(item.shape)

    for j in range(total_trainer_shards):
      rem = j
      phys_coords = {}
      for axis, size in reversed(
          list(zip(trainer_mesh_axes, trainer_mesh_shape))
      ):
        phys_coords[axis] = rem % size
        rem //= size

      shard_arr = _slice_global_array(
          global_arr,
          trainer_mesh_axes,
          trainer_mesh_shape,
          item.trainer_sharding,
          phys_coords,
      )

      shard_assigned = False
      for ws, shards in zip(ws_src_list, trainer_host_shards):
        if j in shards:
          local_slot = shards.index(j)
          buf = ws.get_host_buffer(layer_idx=l, shard_idx=local_slot)
          words = buf.view(shard_arr.dtype)
          words[: shard_arr.size] = shard_arr.ravel()
          shard_assigned = True
          break
      if not shard_assigned:
        raise ValueError(
            f"Global shard {j} is not mapped to any host synchronizer"
        )

    layer_expected = []
    total_sampler_shards = int(np.prod(sampler_mesh_shape))
    for k in range(total_sampler_shards):
      rem = k
      sampler_phys_coords = {}
      for axis, size in reversed(
          list(zip(sampler_mesh_axes, sampler_mesh_shape))
      ):
        sampler_phys_coords[axis] = rem % size
        rem //= size
      expected_shard = _slice_global_array(
          global_arr,
          sampler_mesh_axes,
          sampler_mesh_shape,
          item.sampler_sharding,
          sampler_phys_coords,
      )
      layer_expected.append(expected_shard)
    expected_by_layer.append(layer_expected)

  return expected_by_layer


@dataclasses.dataclass
class RecordedHop:
  """Metadata of a hop recorded from the broadcast execution engine."""

  sender: RaidenId
  receiver: RaidenId
  round_idx: int
  uuid: int
  req_id: str
  is_sender: bool
  expected_block_count: int = 0
  dst_endpoint_counts: dict[str, int] | None = None


@dataclasses.dataclass
class SamplerVerificationResult:
  """Detailed verification report for a single destination sampler unit."""

  unit: RaidenId
  is_seeded: bool
  hop_depth: int
  parent_sender: RaidenId
  matched: bool
  mismatched_vars: int
  mismatched_bytes: int
  total_bytes: int
  mismatch_details: list[str] = dataclasses.field(default_factory=list)
  recheck_matched: bool = False


@dataclasses.dataclass(frozen=True)
class SamplerHostTarget:
  """Metadata and synchronizer handle for a single host in a sampler unit."""

  unit: RaidenId
  sampler_idx: int
  host_idx: int
  ws: weight_synchronizer.WeightSynchronizer
  global_shard_indices: list[int]


@dataclasses.dataclass
class HostCompletionRecord:
  """Internal thread completion record for a single destination host."""

  unblock_time: float = 0.0
  controller_done: bool = False
  snapshot: list[list[np.ndarray]] = dataclasses.field(default_factory=list)
  metrics: Any = None
  exception: Exception | None = None
  trigger_fired: bool = False
  trigger_count: int = 0
  timed_out: bool = False


@dataclasses.dataclass
class CompletionSnapshotResult:
  """Results from observing host staging memory at the OnDataReceived/auto-H2D trigger."""

  unit: RaidenId
  sampler_idx: int
  host_idx: int
  global_shards: list[int]
  is_seeded: bool
  hop_depth: int
  parent_sender: RaidenId
  unblock_timestamp: float
  controller_done_at_unblock: bool
  trigger_fired: bool
  trigger_count: int
  bytes_at_trigger: int
  snapshot_matched: bool
  snapshot_mismatched_vars: int
  snapshot_mismatched_bytes: int
  snapshot_total_bytes: int
  snapshot_details: list[str] = dataclasses.field(default_factory=list)
  final_matched: bool = False
  final_mismatched_bytes: int = 0
  final_bytes_present: int = 0
  timed_out: bool = False
  expected_blocks_from_engine: int = 0
  total_h2d_time_ms: float = 0.0
  last_h2d_time_ms: float = 0.0


class WeightSyncTreeRelayChecksumTest(absltest.TestCase):
  """Comprehensive unit tests for All-Source Binomial Tree weight sync parity."""

  def setUp(self):
    super().setUp()
    weight_synchronizer.configure_telemetry(["buffered"])
    self.addCleanup(lambda: weight_synchronizer.configure_telemetry([]))
    weight_synchronizer.get_and_reset_metric_samples()
    self.last_telemetry_samples: dict[str, list[float]] = {}
    self.timeout_secs = _TRANSFER_TIMEOUT_SECS.value
    self.controller_network_client = (
        raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
    )
    self.addCleanup(self.controller_network_client.close)

    self.controller = raiden_controller.RaidenController(
        port=0,
        worker_rpc_client=self.controller_network_client,
    )
    self.controller_server = raiden_controller.RaidenControllerServer(
        self.controller
    )
    self.controller_server.start()
    self.addCleanup(self.controller_server.stop)

    self.ctrl_client = raiden_controller.RaidenControllerClientFacade(
        f"127.0.0.1:{self.controller_server.port}",
        name_resolver=None,
    )
    self.recorded_hops: list[RecordedHop] = []
    self.registered_expected_blocks: dict[str, int] = {}

    pipe_client = self.controller_network_client._control_pipe_client
    original_send_raw_bytes = pipe_client.send_raw_bytes
    original_send_raw_bytes_sync = pipe_client.send_raw_bytes_sync

    def _record_start_transfer_payload(endpoint: str, payload: bytes) -> None:
      try:
        req = raiden_service_pb2.ControlRequest()
        req.ParseFromString(payload)
        if (
            req.command
            == raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
            and req.HasField("start_transfer_request")
            and not req.start_transfer_request.is_sender
            and req.start_transfer_request.expected_block_count > 0
        ):
          self.registered_expected_blocks[endpoint] = (
              req.start_transfer_request.expected_block_count
          )
      except Exception:  # pylint: disable=broad-except
        pass

    async def _recording_send_raw_bytes(
        endpoint: str, payload: bytes, *args, **kwargs
    ):
      _record_start_transfer_payload(endpoint, payload)
      return await original_send_raw_bytes(endpoint, payload, *args, **kwargs)

    def _recording_send_raw_bytes_sync(
        endpoint: str, payload: bytes, *args, **kwargs
    ):
      _record_start_transfer_payload(endpoint, payload)
      return original_send_raw_bytes_sync(endpoint, payload, *args, **kwargs)

    pipe_client.send_raw_bytes = _recording_send_raw_bytes
    pipe_client.send_raw_bytes_sync = _recording_send_raw_bytes_sync

    # Wrap worker_rpc_client.start_transfer to record actual tree hops
    original_start_transfer = (
        self.controller._broadcast_engine._worker_rpc_client.start_transfer
    )

    async def _recording_start_transfer(
        target_id, transfer_plan, *args, **kwargs
    ):
      is_sender = (
          target_id in transfer_plan.src_units and transfer_plan.is_sender
      )
      src_u = (
          transfer_plan.src_units[0] if transfer_plan.src_units else RaidenId()
      )
      dst_u = (
          transfer_plan.dst_units[0] if transfer_plan.dst_units else RaidenId()
      )
      r_idx = (
          transfer_plan.broadcast_round
          if transfer_plan.broadcast_round is not None
          else 0
      )
      u_val = transfer_plan.uuid
      req_val = transfer_plan.req_id
      exp_count = transfer_plan.expected_block_count
      dst_counts = (
          dict(transfer_plan.dst_endpoint_counts)
          if transfer_plan.dst_endpoint_counts
          else None
      )
      self.recorded_hops.append(
          RecordedHop(
              sender=src_u,
              receiver=dst_u,
              round_idx=r_idx,
              uuid=u_val,
              req_id=req_val,
              is_sender=is_sender,
              expected_block_count=exp_count,
              dst_endpoint_counts=dst_counts,
          )
      )
      return await original_start_transfer(
          target_id, transfer_plan, *args, **kwargs
      )

    self.controller._broadcast_engine._worker_rpc_client.start_transfer = (
        _recording_start_transfer
    )

  def _unblock_waiter(
      self, ws: weight_synchronizer.WeightSynchronizer, uuid: int
  ) -> None:
    """Unblocks any waiter thread stuck inside WaitForTransferCompletion(uuid).

    Justification for protocol message injection in test failure teardown:
    In C++, WeightSynchronizerBase::WaitForTransferCompletion unconditionally
    awaits on completed_transfers_mu_ without a timeout parameter,
    cancellation mechanism, or interruption signal (confirmed at
    weight_synchronizer_base.cc:1463-1475). During test teardown, any waiter
    thread still blocked inside C++ would remain blocked indefinitely. If the
    test runner proceeds to fixture teardown and destroys the C++ synchronizer,
    the blocked daemon thread accesses deallocated synchronization primitives,
    triggering SIGSEGV during process exit.

    Because C++ does not expose a thread interruption API, injecting a
    synthetic COMMAND_START_TRANSFER with expected_block_count=0 is the only
    safe mechanism to satisfy completed_chunks >= expected_chunks in
    RawBufferTransport, triggering OnDataReceived and waking the condition
    variable so the waiter thread joins cleanly before teardown.

    Args:
      ws: The synchronizer instance whose waiter should be unblocked.
      uuid: Transfer UUID to unblock.
    """
    try:
      req = raiden_service_pb2.ControlRequest(
          command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
          start_transfer_request=raiden_service_pb2.StartTransferRequest(
              uuid=uuid,
              expected_block_count=0,
          ),
      )
      bind_ip = ws.bind_ip
      self.controller_network_client._control_pipe_client.send_raw_bytes_sync(
          f"{bind_ip}:{ws.listener_port}",
          req.SerializeToString(),
          timeout=5.0,
          message_type=raiden_service_pb2.ControlRequest.DESCRIPTOR.full_name,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to unblock waiter on port %s: %s", ws.listener_port, e
      )

  def _build_tree_topology(
      self,
      primary_src_unit: RaidenId,
      dst_units: Sequence[RaidenId],
  ) -> tuple[dict[RaidenId, int], dict[RaidenId, RaidenId]]:
    """Determines hop depth and direct parent for each destination unit."""
    hop_depths: dict[RaidenId, int] = {primary_src_unit: 0}
    parents: dict[RaidenId, RaidenId] = {}

    for hop in self.recorded_hops:
      if not hop.is_sender and hop.receiver in dst_units:
        s = hop.sender
        d = hop.receiver
        if d not in parents:
          parents[d] = s
          if s in hop_depths:
            hop_depths[d] = hop_depths[s] + 1

    for d in dst_units:
      if d not in hop_depths:
        hop_depths[d] = -1
        parents[d] = RaidenId("unknown", "", "")

    return hop_depths, parents

  def _execute_tree_transfer_and_verify(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      dst_ws_list: list[weight_synchronizer.WeightSynchronizer],
      specs: list[Scaled35BVarSpec],
      expected_by_layer: list[list[np.ndarray]],
      uuid: int,
      req_id: str,
      test_label: str,
      broadcast_host_ratio: float,
      pipeline_stages: int,
      num_sampler_shards: int,
  ) -> list[SamplerVerificationResult]:
    """Executes tree weight sync and asserts byte-exact parity across all units."""
    self.controller.broadcast_host_ratio = broadcast_host_ratio
    self.controller.broadcast_pipeline_stages = pipeline_stages
    self.controller._plan_cache.clear()
    self.recorded_hops.clear()

    num_layers = len(specs)
    future = self.controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=uuid,
        req_id=req_id,
        skip_d2h=True,
        skip_tiling={l: False for l in range(num_layers)},
    )

    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(
          asyncio.wait_for(future.wait(), timeout=self.timeout_secs)
      )
    except asyncio.TimeoutError as err:
      raise AssertionError(
          f"{test_label}: Controller start_transfer future timed out after"
          f" {self.timeout_secs}s at stage controller_wait"
      ) from err
    finally:
      loop.close()

    primary_src = src_units[0]
    hop_depths, parents = self._build_tree_topology(primary_src, dst_units)

    waiter_records = []
    for idx, (dst_unit, ws_dst) in enumerate(zip(dst_units, dst_ws_list)):
      depth = hop_depths[dst_unit]
      parent = parents[dst_unit]
      role = (
          "SEED"
          if (depth == 1 and parent == primary_src)
          else f"RELAY(depth={depth}, parent={parent})"
      )
      done_event = threading.Event()
      err_holder: list[Exception] = []

      def _target(w, events, errors):
        try:
          w.wait_for_transfer_completion(uuid=uuid)
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)
        finally:
          events.set()

      t = threading.Thread(
          target=_target, args=(ws_dst, done_event, err_holder), daemon=True
      )
      t.start()
      waiter_records.append(
          (idx, dst_unit, ws_dst, role, t, done_event, err_holder)
      )

    wait_start = time.time()
    for (
        idx,
        dst_unit,
        ws_dst,
        role,
        t,
        done_event,
        err_holder,
    ) in waiter_records:
      elapsed = time.time() - wait_start
      remaining = max(0.1, self.timeout_secs - elapsed)
      if not done_event.wait(timeout=remaining):
        for _, _, other_ws, _, other_t, other_event, _ in waiter_records:
          if not other_event.is_set():
            self._unblock_waiter(other_ws, uuid)
            other_t.join(timeout=2.0)
        raise AssertionError(
            f"{test_label}: Destination {dst_unit} (role={role}) timed out"
            f" awaiting transfer completion for uuid={uuid} after"
            f" {self.timeout_secs}s at stage destination_completion_wait"
        )
      if err_holder:
        t.join(timeout=2.0)
        raise AssertionError(
            f"{test_label}: Destination {dst_unit} (role={role}) failed"
            f" awaiting transfer completion for uuid={uuid} at stage"
            f" destination_completion_wait: {err_holder[0]}"
        )
      t.join(timeout=2.0)

    results: list[SamplerVerificationResult] = []
    any_mismatch = False

    for dst_unit, ws_dst in zip(dst_units, dst_ws_list):
      depth = hop_depths[dst_unit]
      parent = parents[dst_unit]
      is_seeded = depth == 1 and parent == primary_src

      total_sampler_bytes = 0
      mismatched_bytes = 0
      mismatched_vars = 0
      details: list[str] = []

      for l, item in enumerate(specs):
        for k in range(num_sampler_shards):
          buf = ws_dst.get_host_buffer(layer_idx=l, shard_idx=k)
          expected_arr = expected_by_layer[l][k]
          actual_arr = buf.view(expected_arr.dtype)[
              : expected_arr.size
          ].reshape(expected_arr.shape)

          byte_count = expected_arr.nbytes
          total_sampler_bytes += byte_count

          if not np.array_equal(actual_arr, expected_arr):
            diff_mask = actual_arr != expected_arr
            diff_count = int(np.sum(diff_mask))
            mismatched_vars += 1
            mismatched_bytes += diff_count * expected_arr.itemsize
            if len(details) < 5:
              mismatch_indices = np.where(diff_mask)
              first_idx = tuple(axis_idx[0] for axis_idx in mismatch_indices)
              details.append(
                  f"Layer {l} ({item.name}) shard {k}: {diff_count} elems"
                  f" differ, first at {first_idx}: expected"
                  f" {expected_arr[first_idx]!r} vs actual"
                  f" {actual_arr[first_idx]!r}"
              )

      matched = mismatched_bytes == 0
      if not matched:
        any_mismatch = True

      results.append(
          SamplerVerificationResult(
              unit=dst_unit,
              is_seeded=is_seeded,
              hop_depth=depth,
              parent_sender=parent,
              matched=matched,
              mismatched_vars=mismatched_vars,
              mismatched_bytes=mismatched_bytes,
              total_bytes=total_sampler_bytes,
              mismatch_details=details,
          )
      )

    # Re-check parity after a short delay to distinguish 'late' vs 'lost' data
    time.sleep(0.3)
    for res_idx, (dst_unit, ws_dst) in enumerate(zip(dst_units, dst_ws_list)):
      recheck_matched = True
      for l in range(num_layers):
        for k in range(num_sampler_shards):
          buf = ws_dst.get_host_buffer(layer_idx=l, shard_idx=k)
          expected_arr = expected_by_layer[l][k]
          actual_arr = buf.view(expected_arr.dtype)[
              : expected_arr.size
          ].reshape(expected_arr.shape)
          if not np.array_equal(actual_arr, expected_arr):
            recheck_matched = False
            break
        if not recheck_matched:
          break
      results[res_idx].recheck_matched = recheck_matched

    # Format structured diagnostic report
    report_lines = [
        f"=== {test_label} Parity Report (stages={pipeline_stages},"
        f" ratio={broadcast_host_ratio}) ==="
    ]
    for res in results:
      role = (
          "SEED"
          if res.is_seeded
          else f"RELAY(depth={res.hop_depth}, parent={res.parent_sender})"
      )
      status = "MATCH" if res.matched else "MISMATCH"
      recheck_status = (
          "LATE_RECOVERED"
          if (not res.matched and res.recheck_matched)
          else ("LOST_PERMANENT" if not res.matched else "STABLE")
      )
      report_lines.append(
          f"  [{status}] Unit={res.unit} Role={role}: {res.mismatched_vars}"
          f" vars ({res.mismatched_bytes}/{res.total_bytes} bytes mismatch)"
          f" [{recheck_status}]"
      )
      for d in res.mismatch_details:
        report_lines.append(f"      - {d}")

    logging.info("\n".join(report_lines))

    if any_mismatch:
      raise AssertionError(
          f"{test_label}: Checksum parity failure detected across tree"
          " samplers!\n"
          + "\n".join(report_lines)
      )

    return results

  def test_control_1host_trainer_1shard_sampler_1hop_relay(self):
    """Control: Single-host trainer and 1-shard samplers with 1-hop relay."""
    req_id = "ctrl_1host_1shard_1hop"
    uuid = 4004
    specs = _make_scaled_35b_specs()[:4]
    num_layers = len(specs)

    trainer_mesh_shape = (1,)
    trainer_mesh_axes = ["shard"]
    trainer_mesh_dict = {"shard": 1}
    trainer_host_shards = [[0]]

    sampler_mesh_shape = (1,)
    sampler_mesh_axes = ["shard"]
    sampler_mesh_dict = {"shard": 1}
    total_sampler_shards = 1

    ctrl_specs = [
        Scaled35BVarSpec(
            name=s.name,
            shape=s.shape,
            trainer_sharding=["" for _ in s.shape],
            sampler_sharding=["" for _ in s.shape],
            item_size=s.item_size,
        )
        for s in specs
    ]

    src_slice_sizes = _calculate_shard_byte_sizes(
        ctrl_specs, trainer_mesh_dict, is_trainer=True
    )
    dst_slice_sizes = _calculate_shard_byte_sizes(
        ctrl_specs, sampler_mesh_dict, is_trainer=False
    )

    ws_src = (
        weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
            num_layers=num_layers,
            num_shards=1,
            slice_byte_size=src_slice_sizes,
            local_port=0,
            listener_port=0,
            bind_ip="127.0.0.1",
            global_shard_indices=[0],
        )
    )
    ws_src.bind_ip = "127.0.0.1"
    self.addCleanup(ws_src.shutdown)
    ws_src_list = [ws_src]

    num_samplers = 4
    ws_dst_list = []
    dst_units = []
    dst_protos = _build_35b_variable_protos(
        ctrl_specs, sampler_mesh_dict, is_trainer=False
    )

    for i in range(num_samplers):
      ws_dst = (
          weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
              num_layers=num_layers,
              num_shards=1,
              slice_byte_size=dst_slice_sizes,
              local_port=0,
              listener_port=0,
              bind_ip="127.0.0.1",
              auto_h2d=True,
              global_shard_indices=[0],
          )
      )
      ws_dst.bind_ip = "127.0.0.1"
      self.addCleanup(ws_dst.shutdown)
      ws_dst_list.append(ws_dst)
      d_u = RaidenId(f"sampler_{req_id}_{i}", "", "weights")
      dst_units.append(d_u)
      self.ctrl_client.register_work_unit(
          d_u,
          [f"127.0.0.1:{ws_dst.local_port}"],
          control_plane_rpc_address=f"127.0.0.1:{ws_dst.listener_port}",
          mesh_shape=[1],
          variables=dst_protos,
          mesh_axes=sampler_mesh_axes,
      )

    src_unit = RaidenId(f"trainer_{req_id}", "", "weights")
    src_protos = _build_35b_variable_protos(
        ctrl_specs, trainer_mesh_dict, is_trainer=True
    )
    self.ctrl_client.register_work_unit(
        src_unit,
        [f"127.0.0.1:{ws_src.local_port}"],
        control_plane_rpc_address=f"127.0.0.1:{ws_src.listener_port}",
        mesh_shape=[1],
        variables=src_protos,
        mesh_axes=trainer_mesh_axes,
    )

    expected_by_layer = _fill_trainer_and_compute_expected_35b(
        specs=ctrl_specs,
        trainer_mesh_axes=trainer_mesh_axes,
        trainer_mesh_shape=trainer_mesh_shape,
        trainer_host_shards=trainer_host_shards,
        ws_src_list=ws_src_list,
        sampler_mesh_axes=sampler_mesh_axes,
        sampler_mesh_shape=sampler_mesh_shape,
    )

    for ws_dst in ws_dst_list:
      for l in range(num_layers):
        ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)[:] = 0x00

    self._execute_tree_transfer_and_verify(
        src_units=[src_unit],
        dst_units=dst_units,
        dst_ws_list=ws_dst_list,
        specs=ctrl_specs,
        expected_by_layer=expected_by_layer,
        uuid=uuid,
        req_id=req_id,
        test_label="Control 1-Host Trainer -> 1-Shard Samplers (1-hop relay)",
        broadcast_host_ratio=2.0,
        pipeline_stages=4,
        num_sampler_shards=total_sampler_shards,
    )

  def test_control_1host_trainer_1shard_sampler_multihop_relay(self):
    """Control: Single-host trainer and 1-shard samplers with multi-hop relay."""
    req_id = "ctrl_1host_1shard_multihop"
    uuid = 4005
    specs = _make_scaled_35b_specs()[:4]
    num_layers = len(specs)

    trainer_mesh_shape = (1,)
    trainer_mesh_axes = ["shard"]
    trainer_mesh_dict = {"shard": 1}
    trainer_host_shards = [[0]]

    sampler_mesh_shape = (1,)
    sampler_mesh_axes = ["shard"]
    sampler_mesh_dict = {"shard": 1}
    total_sampler_shards = 1

    ctrl_specs = [
        Scaled35BVarSpec(
            name=s.name,
            shape=s.shape,
            trainer_sharding=["" for _ in s.shape],
            sampler_sharding=["" for _ in s.shape],
            item_size=s.item_size,
        )
        for s in specs
    ]

    src_slice_sizes = _calculate_shard_byte_sizes(
        ctrl_specs, trainer_mesh_dict, is_trainer=True
    )
    dst_slice_sizes = _calculate_shard_byte_sizes(
        ctrl_specs, sampler_mesh_dict, is_trainer=False
    )

    ws_src = (
        weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
            num_layers=num_layers,
            num_shards=1,
            slice_byte_size=src_slice_sizes,
            local_port=0,
            listener_port=0,
            bind_ip="127.0.0.1",
            global_shard_indices=[0],
        )
    )
    ws_src.bind_ip = "127.0.0.1"
    self.addCleanup(ws_src.shutdown)
    ws_src_list = [ws_src]

    num_samplers = 5
    ws_dst_list = []
    dst_units = []
    dst_protos = _build_35b_variable_protos(
        ctrl_specs, sampler_mesh_dict, is_trainer=False
    )

    for i in range(num_samplers):
      ws_dst = (
          weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
              num_layers=num_layers,
              num_shards=1,
              slice_byte_size=dst_slice_sizes,
              local_port=0,
              listener_port=0,
              bind_ip="127.0.0.1",
              auto_h2d=True,
              global_shard_indices=[0],
          )
      )
      ws_dst.bind_ip = "127.0.0.1"
      self.addCleanup(ws_dst.shutdown)
      ws_dst_list.append(ws_dst)
      d_u = RaidenId(f"sampler_{req_id}_{i}", "", "weights")
      dst_units.append(d_u)
      self.ctrl_client.register_work_unit(
          d_u,
          [f"127.0.0.1:{ws_dst.local_port}"],
          control_plane_rpc_address=f"127.0.0.1:{ws_dst.listener_port}",
          mesh_shape=[1],
          variables=dst_protos,
          mesh_axes=sampler_mesh_axes,
      )

    src_unit = RaidenId(f"trainer_{req_id}", "", "weights")
    src_protos = _build_35b_variable_protos(
        ctrl_specs, trainer_mesh_dict, is_trainer=True
    )
    self.ctrl_client.register_work_unit(
        src_unit,
        [f"127.0.0.1:{ws_src.local_port}"],
        control_plane_rpc_address=f"127.0.0.1:{ws_src.listener_port}",
        mesh_shape=[1],
        variables=src_protos,
        mesh_axes=trainer_mesh_axes,
    )

    expected_by_layer = _fill_trainer_and_compute_expected_35b(
        specs=ctrl_specs,
        trainer_mesh_axes=trainer_mesh_axes,
        trainer_mesh_shape=trainer_mesh_shape,
        trainer_host_shards=trainer_host_shards,
        ws_src_list=ws_src_list,
        sampler_mesh_axes=sampler_mesh_axes,
        sampler_mesh_shape=sampler_mesh_shape,
    )

    for ws_dst in ws_dst_list:
      for l in range(num_layers):
        ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)[:] = 0x00

    self._execute_tree_transfer_and_verify(
        src_units=[src_unit],
        dst_units=dst_units,
        dst_ws_list=ws_dst_list,
        specs=ctrl_specs,
        expected_by_layer=expected_by_layer,
        uuid=uuid,
        req_id=req_id,
        test_label=(
            "Control 1-Host Trainer -> 1-Shard Samplers (Multi-hop relay)"
        ),
        broadcast_host_ratio=1.0,
        pipeline_stages=4,
        num_sampler_shards=total_sampler_shards,
    )

  def _execute_tree_transfer_with_completion_snapshots(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      dst_host_targets: list[SamplerHostTarget],
      specs: list[Scaled35BVarSpec],
      expected_by_layer: list[list[np.ndarray]],
      uuid: int,
      req_id: str,
      test_label: str,
      broadcast_host_ratio: float,
      pipeline_stages: int,
      expect_telemetry: bool = False,
  ) -> list[CompletionSnapshotResult]:
    """Executes tree weight sync and observes OnDataReceived per destination host.

    NOTE: WaitForTransferCompletion is used exclusively as an existing
    observable
    of the native C++ OnDataReceived / auto-H2D trigger (which sets
    completed_transfers_ only upon finishing OnDataReceived in
    weight_synchronizer_base.cc:1396), not because production workers call it.
    Production workers run with auto_h2d=True and invoke raiden_h2d solely to
    wait for hardware DMA to settle (raiden_weight_sync_delegate.py:63-68).

    Args:
      src_units: Source trainer work unit IDs.
      dst_units: Destination sampler work unit IDs.
      dst_host_targets: Per-host sampler targets to observe.
      specs: Scaled model parameter specifications.
      expected_by_layer: Ground-truth reference arrays by layer and shard.
      uuid: Transfer generation UUID.
      req_id: Unique request identifier.
      test_label: Descriptive label for reporting.
      broadcast_host_ratio: Fan-out ratio for tree broadcast scheduling.
      pipeline_stages: Number of pipeline stage groups.
      expect_telemetry: Whether canonical H2D telemetry is asserted.

    Returns:
      List of per-host completion and snapshot results.

    Raises:
      AssertionError: If controller future times out.
      KeyError: If canonical telemetry metric is missing when expected.
    """
    self.controller.broadcast_host_ratio = broadcast_host_ratio
    self.controller.broadcast_pipeline_stages = pipeline_stages
    self.controller._plan_cache.clear()
    self.recorded_hops.clear()
    self.registered_expected_blocks.clear()

    num_layers = len(specs)
    future_holder: list[Any] = [None]
    completion_records = [HostCompletionRecord() for _ in dst_host_targets]
    threads = []

    for idx, target in enumerate(dst_host_targets):

      def _wait_and_snapshot(t_target, t_idx, fut_holder, records):
        try:
          # Observe native OnDataReceived / auto-H2D trigger return
          t_target.ws.wait_for_transfer_completion(uuid=uuid)
          t_unblock = time.time()
          rec = records[t_idx]
          if rec.timed_out:
            # Woken up by teardown unblocker, not natural transfer completion
            return
          fut = fut_holder[0]
          is_ctrl_done = fut.done() if fut is not None else False
          snap = []
          for l in range(num_layers):
            layer_shards = []
            for local_k in range(len(t_target.global_shard_indices)):
              layer_shards.append(
                  np.copy(
                      t_target.ws.get_host_buffer(
                          layer_idx=l, shard_idx=local_k
                      )
                  )
              )
            snap.append(layer_shards)
          m = t_target.ws._impl.get_metrics()
          rec.unblock_time = t_unblock
          rec.controller_done = is_ctrl_done
          rec.snapshot = snap
          rec.metrics = m
          rec.exception = None
          rec.trigger_fired = True
          rec.trigger_count += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
          rec = records[t_idx]
          rec.unblock_time = time.time()
          rec.controller_done = False
          rec.snapshot = []
          rec.metrics = None
          rec.exception = e
          rec.trigger_fired = False

      t = threading.Thread(
          target=_wait_and_snapshot,
          args=(target, idx, future_holder, completion_records),
          daemon=True,
      )
      threads.append(t)
      t.start()

    time.sleep(0.05)

    future = self.controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=uuid,
        req_id=req_id,
        skip_d2h=True,
        skip_tiling={l: False for l in range(num_layers)},
    )
    future_holder[0] = future

    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(
          asyncio.wait_for(future.wait(), timeout=self.timeout_secs)
      )
    except asyncio.TimeoutError as err:
      raise AssertionError(
          f"{test_label}: Controller start_transfer future timed out after"
          f" {self.timeout_secs}s at stage controller_wait"
      ) from err
    finally:
      loop.close()

    primary_src = src_units[0]
    hop_depths, parents = self._build_tree_topology(primary_src, dst_units)

    # Wait up to 5.0s for destination completion triggers to register
    completion_wait_timeout = min(self.timeout_secs, 5.0)
    wait_start = time.time()
    for idx, (t, target) in enumerate(zip(threads, dst_host_targets)):
      elapsed = time.time() - wait_start
      remaining = max(0.05, completion_wait_timeout - elapsed)
      t.join(timeout=remaining)

    # Teardown unblocker: unblock any thread still waiting so it joins cleanly
    # before destruction. We explicitly mark timed_out beforehand so that
    # teardown synthetic triggers cannot falsely record completion or pass
    # tests.
    for other_idx, (other_t, other_target) in enumerate(
        zip(threads, dst_host_targets)
    ):
      if other_t.is_alive():
        completion_records[other_idx].timed_out = True
        self._unblock_waiter(other_target.ws, uuid)
        other_t.join(timeout=2.0)

    self.last_telemetry_samples = (
        weight_synchronizer.get_and_reset_metric_samples()
    )
    h2d_metric_keys = [
        "tpu_raiden_weight_sync_h2d_transfer_time_ms",
        "weight_sync_h2d_transfer_time_ms",
    ]
    h2d_samples = []
    for k in h2d_metric_keys:
      if k in self.last_telemetry_samples:
        h2d_samples.extend(self.last_telemetry_samples[k])

    if expect_telemetry and not h2d_samples:
      raise KeyError(
          "Expected canonical telemetry metric"
          " 'tpu_raiden_weight_sync_h2d_transfer_time_ms' but it was not"
          " recorded. Observed metrics:"
          f" {list(self.last_telemetry_samples.keys())}"
      )

    logging.info(
        "%s: Telemetry observed %d H2D sample(s): %s",
        test_label,
        len(h2d_samples),
        h2d_samples,
    )

    results: list[CompletionSnapshotResult] = []

    for d_idx, target in enumerate(dst_host_targets):
      depth = hop_depths[target.unit]
      parent = parents[target.unit]
      is_seeded = depth == 1 and parent == primary_src

      rec = completion_records[d_idx]
      unblock_t = rec.unblock_time
      ctrl_done = rec.controller_done
      snap = rec.snapshot
      trigger_fired = rec.trigger_fired

      total_host_bytes = 0
      mismatched_bytes = 0
      mismatched_vars = 0
      bytes_at_trigger = 0
      details: list[str] = []

      # Total expected bytes across all layers for this host
      for l in range(num_layers):
        for global_k in target.global_shard_indices:
          total_host_bytes += expected_by_layer[l][global_k].nbytes

      if not snap or not trigger_fired:
        if rec.exception is not None:
          exc = f"Exception in waiter thread: {rec.exception}"
        elif rec.timed_out:
          exc = "Never unblocked within timeout"
        else:
          exc = "Trigger never fired"
        details.append(f"Trigger missing: {exc}")
        mismatched_vars = num_layers
        mismatched_bytes = total_host_bytes
        matched = False
      else:
        for l, item in enumerate(specs):
          for local_k, global_k in enumerate(target.global_shard_indices):
            buf = snap[l][local_k]
            expected_arr = expected_by_layer[l][global_k]
            actual_arr = buf.view(expected_arr.dtype)[
                : expected_arr.size
            ].reshape(expected_arr.shape)

            non_zeros = int(np.sum(actual_arr != 0)) * expected_arr.itemsize
            bytes_at_trigger += non_zeros

            if not np.array_equal(actual_arr, expected_arr):
              diff_mask = actual_arr != expected_arr
              diff_count = int(np.sum(diff_mask))
              mismatched_vars += 1
              mismatched_bytes += diff_count * expected_arr.itemsize
              if len(details) < 5:
                all_zeros = np.all(actual_arr == 0)
                zero_tag = (
                    " [UNRECEIVED STAGE - ALL ZEROS]" if all_zeros else ""
                )
                details.append(
                    f"Layer {l} ({item.name}) shard {global_k}{zero_tag}:"
                    f" {diff_count} elements differ: expected"
                    f" {expected_arr.flat[0]!r}... vs actual"
                    f" {actual_arr.flat[0]!r}..."
                )

        matched = mismatched_bytes == 0

      # Final DRAM parity check
      final_mismatched_bytes = 0
      final_bytes_present = 0
      for l in range(num_layers):
        for local_k, global_k in enumerate(target.global_shard_indices):
          buf = target.ws.get_host_buffer(layer_idx=l, shard_idx=local_k)
          expected_arr = expected_by_layer[l][global_k]
          actual_arr = buf.view(expected_arr.dtype)[
              : expected_arr.size
          ].reshape(expected_arr.shape)
          final_bytes_present += (
              int(np.sum(actual_arr != 0)) * expected_arr.itemsize
          )
          if not np.array_equal(actual_arr, expected_arr):
            diff_mask = actual_arr != expected_arr
            final_mismatched_bytes += (
                int(np.sum(diff_mask)) * expected_arr.itemsize
            )

      final_matched = final_mismatched_bytes == 0
      final_m = target.ws._impl.get_metrics()
      total_h2d_ms = final_m.total_h2d_time_ms
      last_h2d_ms = final_m.last_h2d_time_ms

      # Determine per-host expected block count registered by this receiver
      host_endpoint = f"{target.ws.bind_ip}:{target.ws.listener_port}"
      exp_blocks = self.registered_expected_blocks[host_endpoint]

      results.append(
          CompletionSnapshotResult(
              unit=target.unit,
              sampler_idx=target.sampler_idx,
              host_idx=target.host_idx,
              global_shards=target.global_shard_indices,
              is_seeded=is_seeded,
              hop_depth=depth,
              parent_sender=parent,
              unblock_timestamp=unblock_t,
              controller_done_at_unblock=ctrl_done,
              trigger_fired=trigger_fired,
              trigger_count=rec.trigger_count,
              bytes_at_trigger=bytes_at_trigger,
              snapshot_matched=matched,
              snapshot_mismatched_vars=mismatched_vars,
              snapshot_mismatched_bytes=mismatched_bytes,
              snapshot_total_bytes=total_host_bytes,
              snapshot_details=details,
              final_matched=final_matched,
              final_mismatched_bytes=final_mismatched_bytes,
              final_bytes_present=final_bytes_present,
              timed_out=rec.timed_out,
              expected_blocks_from_engine=exp_blocks,
              total_h2d_time_ms=total_h2d_ms,
              last_h2d_time_ms=last_h2d_ms,
          )
      )

    report_lines = [
        f"=== {test_label} Per-Host Trigger & Parity Diagnostic Table ==="
    ]
    for res in results:
      role = (
          "SEED"
          if res.is_seeded
          else f"RELAY(depth={res.hop_depth}, parent={res.parent_sender})"
      )
      if res.trigger_fired:
        trig_status = (
            f"FIRED ({res.bytes_at_trigger}/{res.snapshot_total_bytes} bytes"
            " present)"
        )
      else:
        trig_status = "TIMED_OUT (never fired)"
      parity_status = "MATCH" if res.final_matched else "MISMATCH"
      report_lines.append(
          f"  [{role}] Sampler {res.sampler_idx} Host {res.host_idx}"
          f" (shards={res.global_shards}, unit={res.unit}):"
          f" trigger={trig_status} | final DRAM={parity_status}"
          f" ({res.final_bytes_present}/{res.snapshot_total_bytes} bytes)"
          f" | chunk blocks: engine_expected={res.expected_blocks_from_engine}"
      )
      for d in res.snapshot_details:
        report_lines.append(f"      - {d}")

    logging.info("\n".join(report_lines))
    return results

  def _setup_2host_trainer_and_2host_samplers(
      self,
      req_id: str,
      specs: list[Scaled35BVarSpec],
      rate_limit_gbps: float = 0.0,
  ) -> tuple[
      RaidenId,
      list[RaidenId],
      list[SamplerHostTarget],
      list[list[np.ndarray]],
  ]:
    """Sets up a production-faithful 2-host trainer and 2-host samplers with auto_h2d=True."""
    num_layers = len(specs)

    trainer_mesh_shape = (2, 2, 2)
    trainer_mesh_axes = ["fsdp", "context", "tensor"]
    trainer_mesh_dict = {"fsdp": 2, "context": 2, "tensor": 2}
    trainer_host_shards = [[0, 1, 4, 5], [2, 3, 6, 7]]
    trainer_host_ips = ["127.0.0.1", "127.0.0.2"]

    sampler_mesh_shape = (4, 2)
    sampler_mesh_axes = ["attn_dp_expert", "expert"]
    sampler_mesh_dict = {"attn_dp_expert": 4, "expert": 2}
    total_sampler_shards = int(np.prod(sampler_mesh_shape))
    sampler_host_shards = [[0, 1, 2, 3], [4, 5, 6, 7]]

    src_slice_sizes = _calculate_shard_byte_sizes(
        specs, trainer_mesh_dict, is_trainer=True
    )
    dst_slice_sizes = _calculate_shard_byte_sizes(
        specs, sampler_mesh_dict, is_trainer=False
    )

    ws_src_list = []
    for host_shards, host_ip in zip(trainer_host_shards, trainer_host_ips):
      ws_src = (
          weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
              num_layers=num_layers,
              num_shards=len(host_shards),
              slice_byte_size=src_slice_sizes,
              local_port=0,
              listener_port=0,
              bind_ip=host_ip,
              global_shard_indices=host_shards,
              test_only_simulated_egress_gbps=rate_limit_gbps,
              test_only_simulated_ingress_gbps=rate_limit_gbps,
          )
      )
      ws_src.bind_ip = host_ip
      self.addCleanup(ws_src.shutdown)
      ws_src_list.append(ws_src)

    src_unit = RaidenId(f"trainer_{req_id}", "", "weights")
    src_protos = _build_35b_variable_protos(
        specs, trainer_mesh_dict, is_trainer=True
    )
    total_trainer_shards = int(np.prod(trainer_mesh_shape))
    trainer_shard_addresses: list[str | None] = [None] * total_trainer_shards
    for ws_src, host_shards in zip(ws_src_list, trainer_host_shards):
      for shard_idx in host_shards:
        trainer_shard_addresses[shard_idx] = (
            f"{ws_src.bind_ip}:{ws_src.local_port}"
        )
    trainer_rpc_endpoints = [
        f"{ws_src.bind_ip}:{ws_src.listener_port}" for ws_src in ws_src_list
    ]
    self.ctrl_client.register_work_unit(
        src_unit,
        trainer_shard_addresses,  # type: ignore[arg-type]
        control_plane_rpc_address=",".join(trainer_rpc_endpoints),
        mesh_shape=list(trainer_mesh_shape),
        variables=src_protos,
        mesh_axes=trainer_mesh_axes,
    )

    num_samplers = 4
    dst_units = []
    dst_host_targets: list[SamplerHostTarget] = []
    dst_protos = _build_35b_variable_protos(
        specs, sampler_mesh_dict, is_trainer=False
    )

    for i in range(num_samplers):
      d_u = RaidenId(f"sampler_{req_id}_{i}", "", "weights")
      dst_units.append(d_u)
      unit_shard_addresses: list[str | None] = [None] * total_sampler_shards
      unit_rpc_endpoints = []

      for h_idx, h_shards in enumerate(sampler_host_shards):
        host_ip = f"127.1.{i}.{h_idx + 1}"
        ws_dst = weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
            num_layers=num_layers,
            num_shards=len(h_shards),
            slice_byte_size=dst_slice_sizes,
            local_port=0,
            listener_port=0,
            bind_ip=host_ip,
            auto_h2d=True,
            global_shard_indices=h_shards,
            test_only_simulated_egress_gbps=rate_limit_gbps,
            test_only_simulated_ingress_gbps=rate_limit_gbps,
        )
        ws_dst.bind_ip = host_ip
        self.addCleanup(ws_dst.shutdown)
        unit_rpc_endpoints.append(f"{host_ip}:{ws_dst.listener_port}")
        for s_idx in h_shards:
          unit_shard_addresses[s_idx] = f"{host_ip}:{ws_dst.local_port}"

        dst_host_targets.append(
            SamplerHostTarget(
                unit=d_u,
                sampler_idx=i,
                host_idx=h_idx,
                ws=ws_dst,
                global_shard_indices=h_shards,
            )
        )

      self.ctrl_client.register_work_unit(
          d_u,
          unit_shard_addresses,  # type: ignore[arg-type]
          control_plane_rpc_address=",".join(unit_rpc_endpoints),
          mesh_shape=list(sampler_mesh_shape),
          variables=dst_protos,
          mesh_axes=sampler_mesh_axes,
      )

    expected_by_layer = _fill_trainer_and_compute_expected_35b(
        specs=specs,
        trainer_mesh_axes=trainer_mesh_axes,
        trainer_mesh_shape=trainer_mesh_shape,
        trainer_host_shards=trainer_host_shards,
        ws_src_list=ws_src_list,
        sampler_mesh_axes=sampler_mesh_axes,
        sampler_mesh_shape=sampler_mesh_shape,
    )

    for target in dst_host_targets:
      for l in range(num_layers):
        for local_k in range(len(target.global_shard_indices)):
          target.ws.get_host_buffer(layer_idx=l, shard_idx=local_k)[:] = 0x00

    return src_unit, dst_units, dst_host_targets, expected_by_layer

  def test_contract_trainer_seeded_sampler_triggers_and_matches(self):
    """Contract 3a: Trainer-seeded hosts must fire OnDataReceived/auto-H2D trigger and match trainer."""
    req_id = "contract_seeded"
    uuid = 4006
    specs = _make_scaled_35b_specs()
    src_unit, dst_units, dst_host_targets, expected_by_layer = (
        self._setup_2host_trainer_and_2host_samplers(
            req_id=req_id, specs=specs, rate_limit_gbps=0.0
        )
    )
    results = self._execute_tree_transfer_with_completion_snapshots(
        src_units=[src_unit],
        dst_units=dst_units,
        dst_host_targets=dst_host_targets,
        specs=specs,
        expected_by_layer=expected_by_layer,
        uuid=uuid,
        req_id=req_id,
        test_label="Contract 3a (Trainer-Seeded Hosts)",
        broadcast_host_ratio=2.0,
        pipeline_stages=4,
    )
    seed_results = [res for res in results if res.is_seeded]
    self.assertNotEmpty(seed_results, "No seed sampler hosts found!")

    for res in seed_results:
      # Contract 3a: Trigger must fire within timeout
      self.assertFalse(
          res.timed_out,
          f"Contract 3a violation: Trainer-seeded host {res.unit} Host"
          f" {res.host_idx} timed out awaiting OnDataReceived/auto-H2D trigger"
          " (never fired)! Expected blocks:"
          f" {res.expected_blocks_from_engine}.",
      )
      self.assertEqual(
          res.trigger_count,
          1,
          f"Contract 3a violation: Trainer-seeded host {res.unit} Host"
          f" {res.host_idx} must trigger exactly once, got"
          f" {res.trigger_count}!",
      )
      # Staging must match trainer
      self.assertTrue(
          res.final_matched,
          f"Contract 3a violation: Trainer-seeded host {res.unit} Host"
          f" {res.host_idx} staging buffer did not match trainer at last"
          " trigger!",
      )

  def test_contract_relayed_sampler_no_premature_trigger_and_matches(self):
    """Contract 3b: Relayed hosts must not trigger before all data lands, and must match trainer."""
    req_id = "contract_relayed"
    uuid = 4007
    specs = _make_scaled_35b_specs()
    src_unit, dst_units, dst_host_targets, expected_by_layer = (
        self._setup_2host_trainer_and_2host_samplers(
            req_id=req_id, specs=specs, rate_limit_gbps=0.02
        )
    )
    results = self._execute_tree_transfer_with_completion_snapshots(
        src_units=[src_unit],
        dst_units=dst_units,
        dst_host_targets=dst_host_targets,
        specs=specs,
        expected_by_layer=expected_by_layer,
        uuid=uuid,
        req_id=req_id,
        test_label="Contract 3b (Relayed Hosts)",
        broadcast_host_ratio=2.0,
        pipeline_stages=4,
    )
    relay_results = [res for res in results if not res.is_seeded]
    self.assertNotEmpty(relay_results, "No relay sampler hosts found!")

    for res in relay_results:
      self.assertFalse(
          res.timed_out,
          f"Contract 3b violation: Relayed host {res.unit} Host {res.host_idx}"
          " timed out awaiting OnDataReceived/auto-H2D trigger!",
      )
      self.assertEqual(
          res.trigger_count,
          1,
          f"Contract 3b violation: Relayed host {res.unit} Host {res.host_idx}"
          " must trigger exactly once, got"
          f" {res.trigger_count}!",
      )
      # Contract 3b: No trigger may fire before all data has landed
      self.assertEqual(
          res.bytes_at_trigger,
          res.snapshot_total_bytes,
          f"Contract 3b violation: Relayed host {res.unit} Host {res.host_idx}"
          " prematurely fired trigger before all data landed! Only"
          f" {res.bytes_at_trigger}/{res.snapshot_total_bytes} bytes were"
          " present at trigger instant. (Settles Open Question 1: Stage 0"
          " prematurely fires auto-H2D with partial weights).",
      )
      self.assertTrue(
          res.snapshot_matched,
          f"Contract 3b violation: Relayed host {res.unit} Host {res.host_idx}"
          " staging buffer did not match trainer at trigger instant!",
      )
      self.assertTrue(
          res.final_matched,
          f"Contract 3b violation: Relayed host {res.unit} Host {res.host_idx}"
          " staging buffer did not match trainer after all stages completed!",
      )

  def test_connection_storm_reproduces_timeout_and_recovers_with_retry(self):
    """Proves that a Round-1 accept-queue overflow reproduces the exact connect timeout error without retry and succeeds with retry."""

    old_env = {
        k: os.environ.get(k)
        for k in (
            "TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS",
            "TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS",
            "TPU_RAIDEN_TCP_CONNECT_INITIAL_BACKOFF_MS",
        )
    }

    def _restore_env():
      for k, v in old_env.items():
        if v is None:
          os.environ.pop(k, None)
        else:
          os.environ[k] = v

    self.addCleanup(_restore_env)

    os.environ["TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS"] = "500"
    os.environ["TPU_RAIDEN_TCP_CONNECT_INITIAL_BACKOFF_MS"] = "150"

    req_id = "conn_storm"
    specs = _make_scaled_35b_specs()[:4]
    num_layers = len(specs)

    trainer_mesh_shape = (1,)
    trainer_mesh_axes = ["shard"]
    trainer_mesh_dict = {"shard": 1}
    trainer_host_shards = [[0]]

    sampler_mesh_shape = (1,)
    sampler_mesh_axes = ["shard"]
    sampler_mesh_dict = {"shard": 1}
    total_sampler_shards = 1

    ctrl_specs = [
        Scaled35BVarSpec(
            name=s.name,
            shape=s.shape,
            trainer_sharding=["" for _ in s.shape],
            sampler_sharding=["" for _ in s.shape],
            item_size=s.item_size,
        )
        for s in specs
    ]

    src_slice_sizes = _calculate_shard_byte_sizes(
        ctrl_specs, trainer_mesh_dict, is_trainer=True
    )
    dst_slice_sizes = _calculate_shard_byte_sizes(
        ctrl_specs, sampler_mesh_dict, is_trainer=False
    )

    ws_src = (
        weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
            num_layers=num_layers,
            num_shards=1,
            slice_byte_size=src_slice_sizes,
            local_port=0,
            listener_port=0,
            bind_ip="127.0.0.1",
            global_shard_indices=[0],
        )
    )
    ws_src.bind_ip = "127.0.0.1"
    self.addCleanup(ws_src.shutdown)
    ws_src_list = [ws_src]

    num_samplers = 4
    ws_dst_list = []
    dst_units = []
    dst_protos = _build_35b_variable_protos(
        ctrl_specs, sampler_mesh_dict, is_trainer=False
    )

    # Create a gate socket with backlog=0 in front of sampler_0's data port so
    # we can deterministically fill its kernel accept queue for the first 650ms
    # (causing the kernel to drop incoming SYNs during Attempt 1) and then drain
    # it and forward accepted connections to sampler_0.
    seed_ip = "127.0.2.1"
    gate_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    gate_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    gate_sock.bind((seed_ip, 0))
    gate_sock.listen(0)
    gate_port = gate_sock.getsockname()[1]
    self.addCleanup(gate_sock.close)

    for i in range(num_samplers):
      host_ip = f"127.0.2.{i + 1}"
      ws_dst = (
          weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
              num_layers=num_layers,
              num_shards=1,
              slice_byte_size=dst_slice_sizes,
              local_port=0,
              listener_port=0,
              bind_ip=host_ip,
              auto_h2d=True,
              global_shard_indices=[0],
          )
      )
      ws_dst.bind_ip = host_ip
      self.addCleanup(ws_dst.shutdown)
      ws_dst_list.append(ws_dst)
      d_u = RaidenId(f"sampler_{req_id}_{i}", "", "weights")
      dst_units.append(d_u)
      data_port = gate_port if i == 0 else ws_dst.local_port
      self.ctrl_client.register_work_unit(
          d_u,
          [f"{host_ip}:{data_port}"],
          control_plane_rpc_address=f"{host_ip}:{ws_dst.listener_port}",
          mesh_shape=[1],
          variables=dst_protos,
          mesh_axes=sampler_mesh_axes,
      )

    src_unit = RaidenId(f"trainer_{req_id}", "", "weights")
    src_protos = _build_35b_variable_protos(
        ctrl_specs, trainer_mesh_dict, is_trainer=True
    )
    self.ctrl_client.register_work_unit(
        src_unit,
        [f"127.0.0.1:{ws_src.local_port}"],
        control_plane_rpc_address=f"127.0.0.1:{ws_src.listener_port}",
        mesh_shape=[1],
        variables=src_protos,
        mesh_axes=trainer_mesh_axes,
    )

    expected_by_layer = _fill_trainer_and_compute_expected_35b(
        specs=ctrl_specs,
        trainer_mesh_axes=trainer_mesh_axes,
        trainer_mesh_shape=trainer_mesh_shape,
        trainer_host_shards=trainer_host_shards,
        ws_src_list=ws_src_list,
        sampler_mesh_axes=sampler_mesh_axes,
        sampler_mesh_shape=sampler_mesh_shape,
    )

    stop_proxy = threading.Event()
    self.addCleanup(stop_proxy.set)

    def _fill_gate_and_drain_after(congest_secs: float, forward_after: bool):
      """Fills gate_sock's backlog=0 accept queue so the kernel drops SYNs for congest_secs, then drains and optionally proxies to ws_dst_list[0]."""
      fillers = []
      for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        s.connect_ex((seed_ip, gate_port))
        fillers.append(s)
      time.sleep(0.05)

      def _worker():
        time.sleep(congest_secs)
        for s in fillers:
          s.close()
        gate_sock.setblocking(False)
        while True:
          try:
            conn, _ = gate_sock.accept()
            conn.close()
          except BlockingIOError:
            break
        if not forward_after:
          return
        gate_sock.settimeout(0.05)
        while not stop_proxy.is_set():
          try:
            client_conn, _ = gate_sock.accept()
          except socket.timeout:
            continue
          except OSError:
            break
          backend_conn = socket.create_connection(
              (seed_ip, ws_dst_list[0].local_port)
          )

          def _pump(src_s, dst_s):
            try:
              while True:
                data = src_s.recv(65536)
                if not data:
                  break
                dst_s.sendall(data)
            except OSError:
              pass
            finally:
              try:
                dst_s.shutdown(socket.SHUT_WR)
              except OSError:
                pass
              try:
                src_s.close()
              except OSError:
                pass

          threading.Thread(
              target=_pump, args=(client_conn, backend_conn), daemon=True
          ).start()
          threading.Thread(
              target=_pump, args=(backend_conn, client_conn), daemon=True
          ).start()

      t = threading.Thread(target=_worker, daemon=True)
      t.start()
      return t

    # 1. With MAX_ATTEMPTS=1 (no retries), an accept-queue overflow on the
    # Round-1 seed receiver reproduces the exact production error:
    # RuntimeError: Raiden remote native execution failed: Failed to connect to
    # peer 127.0.2.1:...: connect timed out after 500ms
    os.environ["TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS"] = "1"
    self.controller.broadcast_host_ratio = 2.0
    self.controller.broadcast_pipeline_stages = 2
    self.controller._plan_cache.clear()

    # Keep the gate congested well past the first data-plane connect attempt.
    # The first transfer also pays for control-plane channel setup (LOAS2
    # handshake, CRL loading), which can exceed 700ms under sanitizers; if the
    # gate drains before the trainer connects, the connect succeeds and the
    # transfer hangs instead of hitting the 500ms connect timeout.
    congest_thread = _fill_gate_and_drain_after(5.0, forward_after=False)
    future_fail = self.controller.start_transfer(
        src_units=[src_unit],
        dst_units=dst_units,
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=9001,
        req_id=f"{req_id}_no_retry",
        skip_d2h=True,
        skip_tiling={l: False for l in range(num_layers)},
    )
    loop = asyncio.new_event_loop()
    try:
      with self.assertRaises(RuntimeError) as ctx:
        loop.run_until_complete(
            asyncio.wait_for(future_fail.wait(), timeout=self.timeout_secs)
        )
    finally:
      loop.close()
      congest_thread.join()

    logging.info("Reproduced expected error without retry: %s", ctx.exception)
    self.assertIn("Raiden remote native execution failed", str(ctx.exception))
    self.assertIn(
        f"Failed to connect to peer {seed_ip}:{gate_port}: connect timed out"
        " after 500ms",
        str(ctx.exception),
    )

    # 2. With MAX_ATTEMPTS=4 (retry with exponential backoff + jitter), the
    # exact same 700ms accept-queue overflow on Round-1 seed receiver is
    # absorbed by retry and passes byte-for-byte parity verification.
    os.environ["TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS"] = "4"
    for ws_dst in ws_dst_list:
      for l in range(num_layers):
        ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)[:] = 0x00

    proxy_thread = _fill_gate_and_drain_after(0.7, forward_after=True)
    try:
      self._execute_tree_transfer_and_verify(
          src_units=[src_unit],
          dst_units=dst_units,
          dst_ws_list=ws_dst_list,
          specs=ctrl_specs,
          expected_by_layer=expected_by_layer,
          uuid=9002,
          req_id=f"{req_id}_with_retry",
          test_label="Connection Storm Recovery With Exponential Backoff Retry",
          broadcast_host_ratio=2.0,
          pipeline_stages=2,
          num_sampler_shards=total_sampler_shards,
      )
    finally:
      stop_proxy.set()
      proxy_thread.join()


if __name__ == "__main__":
  absltest.main()

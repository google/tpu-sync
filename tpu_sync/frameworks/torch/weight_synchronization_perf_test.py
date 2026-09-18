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

"""Distributed multi-process perf test for PyTorch weight synchronization."""

import asyncio
import os
import pathlib
import socket
import time
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
import torch_tpu  # pylint: disable=unused-import

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from tpu_sync.api.torch import weight_synchronizer
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

_NUM_LAYERS = flags.DEFINE_integer(
    "num_decoder_layers",
    4,
    "Number of Qwen 3.5 35B decoder layers to benchmark (default 4 = 1 full"
    " cycle of 3 GDN + 1 Full Attn).",
)
_BENCHMARK_ITERATIONS = flags.DEFINE_integer(
    "benchmark_iterations",
    3,
    "Number of timed benchmark iterations.",
)
_GROUP_SIZE = flags.DEFINE_integer(
    "group_size",
    70,
    "Number of weights to group per transfer request.",
)
_PARALLELISM = flags.DEFINE_integer(
    "parallelism",
    16,
    "Number of parallel TCP stream worker threads for H2H.",
)
_DTYPE = flags.DEFINE_string(
    "dtype",
    "bfloat16",
    "Data type for synthetic weight tensors (bfloat16 or float32).",
)

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v4
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5p
    "0x0063": {1: "1,1,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5e
    "0x006f": {1: "1,1,1", 4: "2,2,1", 8: "2,4,1"},  # TPU v6e
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v7
}


def _scan_pci_tpus() -> tuple[int, dict[int, str] | None]:
  """Scans sysfs to count physical TPU chips and locate device topology map."""
  count = 0
  topology_map = None
  pci_devices = pathlib.Path("/sys/bus/pci/devices")
  if not pci_devices.exists():
    return 0, None
  for device_path in pci_devices.iterdir():
    try:
      vendor_id = (device_path / "vendor").read_text().strip()
      if vendor_id != _GOOGLE_PCI_VENDOR_ID:
        continue
      device_id = (device_path / "device").read_text().strip()
      if device_id in _TOPOLOGY_BY_TPU_PCI_DEVICE_ID:
        try:
          group_id = (device_path / "iommu_group").readlink().name
          (pathlib.Path("/dev/vfio") / group_id).stat()
        except OSError:
          continue
        count += 1
        if topology_map is None:
          topology_map = _TOPOLOGY_BY_TPU_PCI_DEVICE_ID[device_id]
    except OSError:
      continue
  return count, topology_map


def get_tpu_device_count() -> int:
  count, _ = _scan_pci_tpus()
  return count


def get_tpu_topology(world_size: int) -> str:
  _, topology_map = _scan_pci_tpus()
  if topology_map and world_size in topology_map:
    return topology_map[world_size]
  if world_size == 8:
    return "2,2,2"
  raise ValueError(f"No TPU topology found for count: {world_size}")


def pick_unused_ports(count: int) -> list[int]:
  """Picks ephemeral local ports that are currently unallocated."""
  ports = []
  sockets = []
  for _ in range(count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    ports.append(s.getsockname()[1])
    sockets.append(s)
  for s in sockets:
    s.close()
  return ports


def prepare_tpu_environment(world_size: int) -> None:
  """Prepares torch_tpu runtime environment variables for multi-process SPMD."""
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    ports = pick_unused_ports(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        [f"localhost:{p}" for p in ports]
    )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    os.environ["TORCH_TPU_TOPOLOGY"] = get_tpu_topology(world_size)


def _resolve_torch_dtype(dtype_str: str) -> tuple[torch.dtype, int]:
  """Maps flag string to (torch.dtype, element_byte_size)."""
  if dtype_str == "bfloat16":
    return torch.bfloat16, 2
  elif dtype_str == "float32":
    return torch.float32, 4
  raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_qwen3_5_35b_specs(
    num_layers: int,
    role: str = "source",
    tp_size: int = 4,
) -> list[tuple[tuple[int, ...], list[str], str]]:
  """Generates parameter specs for Qwen 3.5 35B in standard PyTree leaf order."""
  dim = 2048
  num_routed_experts = 256
  routed_mlp_dim = 512
  shared_mlp_dim = 512
  full_q_dim = 8192
  linear_qkvz_dim = 12288
  linear_ba_dim = 64
  linear_conv_dim = 8192
  linear_a_log_dim = 32
  linear_rms_dim = 128
  linear_out_dim = 4096

  specs = []
  is_dest = role == "destination"

  # 1. Final decoder layer norm
  specs.append((
      (dim,),
      [""] if is_dest else ["fsdp"],
      "decoder.decoder_norm.scale",
  ))

  # 2. Decoder layers (hybrid cycle: 3 GDN linear attn + 1 full GQA attn)
  for l in range(num_layers):
    is_full_attn = (l + 1) % 4 == 0

    if is_full_attn:
      kv_sharding = (
          ["", "tp", ""]
          if (is_dest and 2 % tp_size == 0)
          else (["", "", ""] if is_dest else ["fsdp", "", ""])
      )
      specs.append((
          (dim, 2, 256),
          kv_sharding,
          f"decoder.layers.{l}.attention.attention.key.kernel",
      ))
      specs.append((
          (256,),
          [""],
          f"decoder.layers.{l}.attention.attention.key_norm.scale",
      ))
      specs.append((
          (dim * 2, dim),
          ["tp", ""] if is_dest else ["tp", "fsdp"],
          f"decoder.layers.{l}.attention.attention.out.kernel",
      ))
      specs.append((
          (dim, full_q_dim),
          ["", "tp"] if is_dest else ["fsdp", "tp"],
          f"decoder.layers.{l}.attention.attention.query.kernel",
      ))
      specs.append((
          (256,),
          [""],
          f"decoder.layers.{l}.attention.attention.query_norm.scale",
      ))
      specs.append((
          (dim, 2, 256),
          kv_sharding,
          f"decoder.layers.{l}.attention.attention.value.kernel",
      ))
    else:
      specs.append((
          (linear_a_log_dim,),
          [""],
          f"decoder.layers.{l}.attention.linear_attn.a_log",
      ))
      specs.append((
          (linear_ba_dim, linear_conv_dim),
          ["", ""] if is_dest else ["fsdp", ""],
          f"decoder.layers.{l}.attention.linear_attn.b_kernel",
      ))
      specs.append((
          (dim, linear_ba_dim),
          ["", ""] if is_dest else ["fsdp", ""],
          f"decoder.layers.{l}.attention.linear_attn.ba_kernel",
      ))
      specs.append((
          (4, 1, linear_conv_dim),
          ["", "", "tp"] if is_dest else ["", "", ""],
          f"decoder.layers.{l}.attention.linear_attn.conv1d.kernel",
      ))
      specs.append((
          (linear_rms_dim,),
          [""],
          f"decoder.layers.{l}.attention.linear_attn.decay_norm.scale",
      ))
      specs.append((
          (linear_rms_dim,),
          [""],
          f"decoder.layers.{l}.attention.linear_attn.dt_bias",
      ))
      specs.append((
          (dim, linear_rms_dim),
          ["", ""] if is_dest else ["fsdp", ""],
          f"decoder.layers.{l}.attention.linear_attn.dt_kernel",
      ))
      specs.append((
          (dim, linear_out_dim),
          ["", "tp"] if is_dest else ["fsdp", "tp"],
          f"decoder.layers.{l}.attention.linear_attn.g_kernel",
      ))
      specs.append((
          (linear_out_dim, dim),
          ["tp", ""] if is_dest else ["tp", "fsdp"],
          f"decoder.layers.{l}.attention.linear_attn.out_kernel",
      ))
      specs.append((
          (dim, linear_qkvz_dim),
          ["", "tp"] if is_dest else ["fsdp", "tp"],
          f"decoder.layers.{l}.attention.linear_attn.qkvz_kernel",
      ))

    # Pre-attention layernorm
    specs.append((
        (dim,),
        [""] if is_dest else ["fsdp"],
        f"decoder.layers.{l}.pre_attention_layernorm.scale",
    ))

    # MLP / MoE Feedforward
    specs.append((
        (num_routed_experts, routed_mlp_dim, dim),
        ["", "tp", ""] if is_dest else ["fsdp", "", ""],
        f"decoder.layers.{l}.mlp.experts.down_proj.kernel",
    ))
    specs.append((
        (num_routed_experts, dim, routed_mlp_dim),
        ["", "", "tp"] if is_dest else ["fsdp", "", ""],
        f"decoder.layers.{l}.mlp.experts.gate_proj.kernel",
    ))
    specs.append((
        (num_routed_experts, dim, routed_mlp_dim),
        ["", "", "tp"] if is_dest else ["fsdp", "", ""],
        f"decoder.layers.{l}.mlp.experts.up_proj.kernel",
    ))
    specs.append((
        (dim, num_routed_experts),
        ["", ""] if is_dest else ["fsdp", ""],
        f"decoder.layers.{l}.mlp.router.kernel",
    ))
    specs.append((
        (shared_mlp_dim, dim),
        ["tp", ""] if is_dest else ["tp", "fsdp"],
        f"decoder.layers.{l}.mlp.shared_expert.down_proj.kernel",
    ))
    specs.append((
        (dim, shared_mlp_dim),
        ["", "tp"] if is_dest else ["fsdp", "tp"],
        f"decoder.layers.{l}.mlp.shared_expert.gate_proj.kernel",
    ))
    specs.append((
        (dim, shared_mlp_dim),
        ["", "tp"] if is_dest else ["fsdp", "tp"],
        f"decoder.layers.{l}.mlp.shared_expert.up_proj.kernel",
    ))
    specs.append((
        (dim, 1),
        ["", ""] if is_dest else ["fsdp", ""],
        f"decoder.layers.{l}.mlp.shared_expert_gate.kernel",
    ))

    # Post-attention layernorm
    specs.append((
        (dim,),
        [""] if is_dest else ["fsdp"],
        f"decoder.layers.{l}.post_attention_layernorm.scale",
    ))

  # 3. Output vocab projection and token embedder
  specs.append((
      (dim, 248320),
      ["", "fsdp"] if is_dest else ["fsdp", "tp"],
      "decoder.logits_dense.kernel",
  ))
  specs.append((
      (248320, dim),
      ["fsdp", ""] if is_dest else ["tp", "fsdp"],
      "token_embedder.embedding",
  ))
  return specs


def generate_local_shard(
    global_shape: tuple[int, ...],
    spec_axes: list[str],
    mesh_shape_dict: dict[str, int],
    rank: int,
    layer_idx: int,
    dtype: Any = np.float32,
) -> np.ndarray:
  """Generates local shard directly from coordinates without global arrays.

  Every element is a deterministic function of its global coordinates, allowing
  source and destination ranks to verify bit-level identical values after
  arbitrary resharded transfers.

  Args:
    global_shape: Global tensor shape tuple.
    spec_axes: Sharding axis names for each dimension (e.g. "fsdp", "tp", "").
    mesh_shape_dict: Mapping of mesh axis name to size.
    rank: Mesh rank index of this worker.
    layer_idx: Unique index of the tensor/layer.
    dtype: Numpy dtype for the array.

  Returns:
    Numpy ndarray containing the local tensor shard.
  """
  fsdp_size = mesh_shape_dict.get("fsdp", 1)
  tp_size = mesh_shape_dict.get("tp", 1)
  fsdp_coord = rank // tp_size
  tp_coord = rank % tp_size

  ranges = []
  local_dims = []
  for axis_name, dim_size in zip(spec_axes, global_shape):
    if axis_name == "fsdp":
      chunk = dim_size // fsdp_size
      st = fsdp_coord * chunk
      ranges.append((st, st + chunk))
      local_dims.append(chunk)
    elif axis_name == "tp":
      chunk = dim_size // tp_size
      st = tp_coord * chunk
      ranges.append((st, st + chunk))
      local_dims.append(chunk)
    else:
      ranges.append((0, dim_size))
      local_dims.append(dim_size)

  val = (layer_idx % 50 + 1) * 0.01
  if len(local_dims) == 1:
    idx = np.arange(ranges[0][0], ranges[0][1], dtype=np.int32)
    return (((idx % 256).astype(dtype) * 0.001) + val).astype(dtype)
  elif len(local_dims) == 2:
    r = (
        (np.arange(ranges[0][0], ranges[0][1], dtype=np.int32) * 17) % 256
    ).astype(np.int16)
    c = (np.arange(ranges[1][0], ranges[1][1], dtype=np.int32) % 256).astype(
        np.int16
    )
    grid = r[:, None] + c[None, :]
    np.remainder(grid, 256, out=grid)
    out = grid.astype(dtype)
    del grid
    out *= 0.001
    out += val
    return out
  elif len(local_dims) == 3:
    d0 = (
        (np.arange(ranges[0][0], ranges[0][1], dtype=np.int32) * 31) % 256
    ).astype(np.int16)
    d1 = (
        (np.arange(ranges[1][0], ranges[1][1], dtype=np.int32) * 17) % 256
    ).astype(np.int16)
    d2 = (np.arange(ranges[2][0], ranges[2][1], dtype=np.int32) % 256).astype(
        np.int16
    )
    grid = d0[:, None, None] + d1[None, :, None] + d2[None, None, :]
    np.remainder(grid, 256, out=grid)
    out = grid.astype(dtype)
    del grid
    out *= 0.001
    out += val
    return out
  else:
    return np.zeros(tuple(local_dims), dtype=dtype)


def _build_variable_protos(
    specs: list[tuple[tuple[int, ...], list[str], str]],
    mesh_shape_dict: dict[str, int],
    item_size: int,
) -> tuple[list[raiden_service_pb2.VariableMetadataProto], int]:
  """Builds controller metadata protos and computes total payload bytes."""
  variable_protos = []
  total_bytes = 0

  for idx, (global_shape, spec_axes, name) in enumerate(specs):
    sharding_shape = [mesh_shape_dict.get(axis, 1) for axis in spec_axes]
    num_elements = int(np.prod(global_shape))
    total_bytes += num_elements * item_size

    layout = list(range(len(global_shape) - 1, -1, -1))
    variable_protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=name,
            shape=global_shape,
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=item_size,
            layer_idx=idx,
            sharding_spec=spec_axes,
        )
    )

  return variable_protos, total_bytes


def _allocate_worker_tensors(
    specs: list[tuple[tuple[int, ...], list[str], str]],
    role: str,
    mesh_shape_dict: dict[str, int],
    local_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
  """Allocates single-device local shard tensors on TPU."""
  tensors = []
  for idx, (global_shape, spec_axes, _) in enumerate(specs):
    if role == "source":
      shard = generate_local_shard(
          global_shape=global_shape,
          spec_axes=spec_axes,
          mesh_shape_dict=mesh_shape_dict,
          rank=local_rank,
          layer_idx=idx,
          dtype=np.float32,
      )
      t = torch.from_numpy(np.ascontiguousarray(shard)).to(
          device=device, dtype=dtype
      )
    else:
      sharding_shape = [mesh_shape_dict.get(axis, 1) for axis in spec_axes]
      local_shape = tuple(g // s for g, s in zip(global_shape, sharding_shape))
      t = torch.zeros(local_shape, dtype=dtype, device=device)
    tensors.append(t)
  return tensors


def _compute_skip_tiling_map(
    specs: list[tuple[tuple[int, ...], list[str], str]],
) -> dict[int, bool]:
  """Returns per-layer skip_tiling flags based on TPU (8, 128) tile alignment."""
  skip_tiling = {}
  for idx, (shape, _, _) in enumerate(specs):
    if len(shape) >= 2 and shape[-2] % 8 == 0 and shape[-1] % 128 == 0:
      skip_tiling[idx] = True
    else:
      skip_tiling[idx] = False
  return skip_tiling


def _run_controller_transfer(
    controller: raiden_controller.RaidenController,
    src_units: list[raiden_controller.RaidenId],
    dst_units: list[raiden_controller.RaidenId],
    uuid: int,
    req_id: str,
    group_size: int,
    parallelism: int,
    skip_tiling: dict[int, bool],
    skip_d2h: bool = True,
) -> None:
  """Executes a synchronous controller-orchestrated H2H transfer."""
  future = controller.start_transfer(
      src_units=src_units,
      dst_units=dst_units,
      dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
      use_block_chunks=True,
      is_sender=True,
      expected_block_count=0,
      uuid=uuid,
      req_id=req_id,
      group_size=group_size,
      parallelism=parallelism,
      skip_tiling=skip_tiling,
      skip_d2h=skip_d2h,
  )
  loop = asyncio.new_event_loop()
  try:
    loop.run_until_complete(future.wait())
  finally:
    loop.close()


def _run_distributed_worker(
    rank: int,
    world_size: int,
    controller_port: int,
    num_layers: int,
    num_iters: int,
    group_size: int,
    parallelism: int,
    dtype_str: str,
) -> None:
  """Main SPMD worker function executed in each TPU subprocess."""
  del world_size
  dist.init_process_group(backend="tpu_dist")
  device = torch.device("tpu")
  torch_dtype, item_size = _resolve_torch_dtype(dtype_str)

  is_source = rank < 4
  role = "source" if is_source else "destination"
  local_rank = rank if is_source else rank - 4

  src_mesh_dict = {"fsdp": 4, "tp": 1}
  dst_mesh_dict = {"fsdp": 1, "tp": 4}
  active_mesh_dict = src_mesh_dict if is_source else dst_mesh_dict

  specs = get_qwen3_5_35b_specs(
      num_layers=num_layers, role=role, tp_size=dst_mesh_dict["tp"]
  )
  protos, total_bytes = _build_variable_protos(
      specs, mesh_shape_dict=active_mesh_dict, item_size=item_size
  )

  # Start in-process controller on rank 0
  controller = None
  controller_server = None
  if rank == 0:
    controller_network_client = raiden_controller.WeightSyncWorkerRpcClient(
        name_resolver=None
    )
    controller = raiden_controller.RaidenController(
        port=controller_port,
        worker_rpc_client=controller_network_client,
    )
    controller_server = raiden_controller.RaidenControllerServer(controller)
    controller_server.start()

  # Ensure controller server socket is open before any worker connects
  dist.barrier()

  ws = None
  try:
    # Allocate local shard tensors on TPU
    local_tensors = _allocate_worker_tensors(
        specs=specs,
        role=role,
        mesh_shape_dict=active_mesh_dict,
        local_rank=local_rank,
        device=device,
        dtype=torch_dtype,
    )
    torch.tpu.synchronize()

    # Wrap in PyTorch WeightSynchronizer (each rank manages 1 local shard)
    device_tensors = [[t] for t in local_tensors]
    ws = weight_synchronizer.WeightSynchronizer(
        device_tensors,
        local_port=0,
        listener_port=0,
        parallelism=parallelism,
        bind_ip="127.0.0.1",
        auto_h2d=False,
    )

    skip_tiling_map = _compute_skip_tiling_map(specs)
    skip_tiling_list = [skip_tiling_map[i] for i in range(len(specs))]
    ws.test_only_set_skip_tiling(skip_tiling_list)

    # Register local work unit with controller
    unit_id = raiden_controller.RaidenId(
        "trainer" if is_source else "sampler",
        str(local_rank),
        "weights",
    )
    ctrl_client = raiden_controller.RaidenControllerClientFacade(
        f"127.0.0.1:{controller_port}",
        name_resolver=None,
    )
    ctrl_client.register_work_unit(
        unit=unit_id,
        shards=[f"127.0.0.1:{ws.local_port}"],
        control_plane_rpc_address=f"127.0.0.1:{ws.listener_port}",
        mesh_shape=[active_mesh_dict.get(ax, 1) for ax in ["fsdp", "tp"]],
        variables=protos,
        mesh_axes=["fsdp", "tp"],
        host_subgrid=[1, 1],
    )

    # Rank 0 verifies all 8 work units have self-registered
    if rank == 0:
      start_wait = time.time()
      while True:
        with controller._lock:
          src_reg = [
              u
              for u in controller._registered_shards
              if u.job_name == "trainer"
          ]
          dst_reg = [
              u
              for u in controller._registered_shards
              if u.job_name == "sampler"
          ]
        if len(src_reg) == 4 and len(dst_reg) == 4:
          break
        if time.time() - start_wait > 30.0:
          raise TimeoutError(
              f"Timeout waiting for worker registration. src={len(src_reg)},"
              f" dst={len(dst_reg)}"
          )
        time.sleep(0.01)

    dist.barrier()

    src_units = [
        raiden_controller.RaidenId("trainer", str(i), "weights")
        for i in range(4)
    ]
    dst_units = [
        raiden_controller.RaidenId("sampler", str(i), "weights")
        for i in range(4)
    ]

    # Warmup transfer
    uuid_warmup = 999
    req_id_warmup = "warmup"
    if is_source:
      ws.d2h()
    dist.barrier()

    if rank == 0:
      msg = (
          f"Executing warmup transfer (group_size={group_size},"
          f" parallelism={parallelism})..."
      )
      print(msg, flush=True)
      logging.info(msg)
      _run_controller_transfer(
          controller,
          src_units=src_units,
          dst_units=dst_units,
          uuid=uuid_warmup,
          req_id=req_id_warmup,
          group_size=group_size,
          parallelism=parallelism,
          skip_tiling=skip_tiling_map,
          skip_d2h=True,
      )

    if not is_source:
      ws.wait_for_transfer_completion(uuid_warmup)
      ws.h2d()
      torch.tpu.synchronize()

    dist.barrier()

    # Numerical parity verification after warmup
    if not is_source:
      dst_specs = specs
      for idx, (global_shape, spec_axes, name) in enumerate(dst_specs):
        expected_shard = generate_local_shard(
            global_shape=global_shape,
            spec_axes=spec_axes,
            mesh_shape_dict=dst_mesh_dict,
            rank=local_rank,
            layer_idx=idx,
            dtype=np.float32,
        )
        local_cpu = local_tensors[idx].cpu()
        expected_cpu = torch.from_numpy(
            np.ascontiguousarray(expected_shard)
        ).to(dtype=local_cpu.dtype)
        if not torch.equal(local_cpu, expected_cpu):
          diff = torch.abs(local_cpu.float() - expected_cpu.float())
          max_diff = torch.max(diff).item()
          if max_diff > 1e-2:
            raise AssertionError(
                f"[Rank {rank}] Warmup parity failed at tensor {idx} ({name}):"
                f" max_diff={max_diff}"
            )

    dist.barrier()
    if rank == 0:
      msg = "Warmup numerical parity check passed!"
      print(msg, flush=True)
      logging.info(msg)

    # Multi-iteration timed benchmark run
    d2h_latencies_ms = []
    h2h_latencies_ms = []
    h2d_latencies_ms = []
    e2e_latencies_ms = []

    if rank == 0:
      msg = f"Executing {num_iters} benchmark iterations..."
      print(msg, flush=True)
      logging.info(msg)

    for it in range(num_iters):
      uuid_bench = 1000 + it
      req_id_bench = f"bench_{it}"

      # Stage 1: D2H
      dist.barrier()
      t0 = time.perf_counter()
      if is_source:
        ws.d2h()
      dist.barrier()
      d2h_ms = (time.perf_counter() - t0) * 1000.0

      # Stage 2: H2H
      dist.barrier()
      t1 = time.perf_counter()
      if rank == 0:
        _run_controller_transfer(
            controller,
            src_units=src_units,
            dst_units=dst_units,
            uuid=uuid_bench,
            req_id=req_id_bench,
            group_size=group_size,
            parallelism=parallelism,
            skip_tiling=skip_tiling_map,
            skip_d2h=True,
        )
      if not is_source:
        ws.wait_for_transfer_completion(uuid_bench)
      dist.barrier()
      h2h_ms = (time.perf_counter() - t1) * 1000.0

      # Stage 3: H2D
      dist.barrier()
      t2 = time.perf_counter()
      if not is_source:
        ws.h2d()
        torch.tpu.synchronize()
      dist.barrier()
      h2d_ms = (time.perf_counter() - t2) * 1000.0

      e2e_ms = d2h_ms + h2h_ms + h2d_ms

      if rank == 0:
        d2h_latencies_ms.append(d2h_ms)
        h2h_latencies_ms.append(h2h_ms)
        h2d_latencies_ms.append(h2d_ms)
        e2e_latencies_ms.append(e2e_ms)
        msg = (
            f"Iteration {it + 1}/{num_iters}: D2H={d2h_ms:.2f} ms,"
            f" H2H={h2h_ms:.2f} ms, H2D={h2d_ms:.2f} ms, E2E={e2e_ms:.2f} ms"
        )
        print(msg, flush=True)
        logging.info(msg)

    # Post-benchmark parity verification
    if not is_source:
      dst_specs = specs
      for idx, (global_shape, spec_axes, name) in enumerate(dst_specs):
        expected_shard = generate_local_shard(
            global_shape=global_shape,
            spec_axes=spec_axes,
            mesh_shape_dict=dst_mesh_dict,
            rank=local_rank,
            layer_idx=idx,
            dtype=np.float32,
        )
        local_cpu = local_tensors[idx].cpu()
        expected_cpu = torch.from_numpy(
            np.ascontiguousarray(expected_shard)
        ).to(dtype=local_cpu.dtype)
        if not torch.equal(local_cpu, expected_cpu):
          diff = torch.abs(local_cpu.float() - expected_cpu.float())
          max_diff = torch.max(diff).item()
          if max_diff > 1e-2:
            raise AssertionError(
                f"[Rank {rank}] Post-benchmark parity failed at tensor {idx}"
                f" ({name}): max_diff={max_diff}"
            )

    dist.barrier()

    if rank == 0:
      total_gb = total_bytes / 1e9
      med_d2h_ms = float(np.median(d2h_latencies_ms))
      med_h2h_ms = float(np.median(h2h_latencies_ms))
      med_h2d_ms = float(np.median(h2d_latencies_ms))
      med_e2e_ms = float(np.median(e2e_latencies_ms))

      d2h_bw_gbs = total_gb / (med_d2h_ms / 1000.0) if med_d2h_ms > 0 else 0.0
      h2h_bw_gbs = total_gb / (med_h2h_ms / 1000.0) if med_h2h_ms > 0 else 0.0
      h2d_bw_gbs = total_gb / (med_h2d_ms / 1000.0) if med_h2d_ms > 0 else 0.0
      e2e_bw_gbs = total_gb / (med_e2e_ms / 1000.0) if med_e2e_ms > 0 else 0.0

      summary_lines = [
          "=" * 80,
          (
              "QWEN 3.5 35B WEIGHT SYNCHRONIZATION BENCHMARK RESULTS (PyTorch"
              " MPMD)"
          ),
          "=" * 80,
          (
              "Topology                : 4 Trainer Ranks (FSDP=4, TP=1) -> 4"
              " Sampler Ranks (FSDP=1, TP=4)"
          ),
          (
              f"Model Parameters Payload: {total_gb:.2f} GB ({total_bytes}"
              f" bytes, {num_layers} layers, dtype={dtype_str})"
          ),
          (
              f"Transfer Configuration  : group_size={group_size},"
              f" parallelism={parallelism}, iters={num_iters}"
          ),
          "-" * 80,
          (
              f"Device-to-Host (D2H)    : {med_d2h_ms:8.2f} ms | Throughput:"
              f" {d2h_bw_gbs:6.2f} GB/s"
          ),
          (
              f"Host-to-Host (H2H)      : {med_h2h_ms:8.2f} ms | Throughput:"
              f" {h2h_bw_gbs:6.2f} GB/s"
          ),
          (
              f"Host-to-Device (H2D)    : {med_h2d_ms:8.2f} ms | Throughput:"
              f" {h2d_bw_gbs:6.2f} GB/s"
          ),
          (
              f"Total Pipeline E2E Time : {med_e2e_ms:8.2f} ms | Aggregate :"
              f" {e2e_bw_gbs:6.2f} GB/s"
          ),
          "=" * 80,
          (
              "Post-benchmark numerical parity verified across all"
              f" {len(specs)} tensors on all sampler ranks."
          ),
      ]
      for line in summary_lines:
        print(line, flush=True)
        logging.info(line)

  finally:
    if controller_server is not None:
      controller_server.stop()
    if ws is not None:
      del ws
    dist.destroy_process_group()


def _worker_fn(
    rank: int,
    world_size: int,
    master_port: int,
    controller_port: int,
    fn,
    args,
    kwargs,
) -> None:
  """Worker trampoline setting process-level rank and master address env vars."""
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
  fn(rank, world_size, controller_port, *args, **kwargs)


def dist_run(world_size: int, fn, *args, **kwargs) -> None:
  """Initializes TPU environment and spawns world_size worker subprocesses."""
  os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
  os.environ.pop("TORCH_TPU_XPROF_SESSION_ID", None)
  prepare_tpu_environment(world_size)
  master_port = pick_unused_ports(1)[0]
  controller_port = pick_unused_ports(1)[0]
  mp.spawn(
      _worker_fn,
      args=(world_size, master_port, controller_port, fn, args, kwargs),
      nprocs=world_size,
      join=True,
  )


class WeightSynchronizationPerfTest(parameterized.TestCase):

  def tearDown(self):
    super().tearDown()
    os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
    os.environ.pop("TORCH_TPU_XPROF_SESSION_ID", None)

  def test_model_specs(self):
    specs = get_qwen3_5_35b_specs(num_layers=4, tp_size=4)
    # 3 GDN layers * 19 tensors + 1 Full Attn layer * 16 tensors + 3 global = 79
    self.assertLen(specs, 79)

  def test_weight_synchronization_perf(self):
    world_size = 8
    device_count = get_tpu_device_count()
    if device_count < world_size:
      self.skipTest(
          f"Test requires {world_size} TPU chips, but found {device_count}"
      )
    num_layers = _NUM_LAYERS.value
    num_iters = _BENCHMARK_ITERATIONS.value
    group_size = _GROUP_SIZE.value
    parallelism = _PARALLELISM.value
    dtype_str = _DTYPE.value

    dist_run(
        world_size,
        _run_distributed_worker,
        num_layers,
        num_iters,
        group_size,
        parallelism,
        dtype_str,
    )


if __name__ == "__main__":
  try:
    mp.set_start_method("spawn")
  except RuntimeError:
    pass
  g3_multiprocessing.handle_test_main(absltest.main)

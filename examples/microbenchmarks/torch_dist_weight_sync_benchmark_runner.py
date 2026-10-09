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

"""Distributed multi-host PyTorch WeightSynchronizer TP to TP benchmark runner.

Benchmarks cross-node weight synchronization from a PyTorch training cluster
(source) to a PyTorch sampling/inference cluster (destination) via TPU Raiden.
Both the trainer and sampler conduct 2D Tensor Parallelism.
"""

import asyncio
from collections.abc import Sequence
from functools import partial
import os
import socket
import sys
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch import distributed as dist
import torch.multiprocessing as mp

from tpu_sync.api.torch import torch_tpu_common_loader
from tpu_sync.api.torch import weight_synchronizer
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

# Register the TPU device backend with PyTorch c10 dispatcher.
torch_tpu_common_loader.load_torch_tpu_common()

FLAGS = flags.FLAGS

_DIST_COORDINATOR_ADDRESS = flags.DEFINE_string(
    "dist_coordinator_address",
    "",
    "Host:port of the coordinator (Host 0) for multi-host PyTorch"
    " initialization.",
)
_DIST_BACKEND = flags.DEFINE_string(
    "dist_backend",
    "gloo",
    "Distributed backend for multi-rank synchronization (e.g. gloo, tpu_dist).",
)
_NUM_HOSTS = flags.DEFINE_integer(
    "num_hosts",
    1,
    "Number of host machines in this cluster (e.g. 2).",
)
_HOST_ID = flags.DEFINE_integer(
    "host_id",
    0,
    "Index of this host machine in the cluster (e.g. 0 on Machine 1, 1 on"
    " Machine 2).",
)
_DEVICES_PER_HOST = flags.DEFINE_integer(
    "devices_per_host",
    8,
    "Number of TPU devices/chips on this host machine.",
)

_ROLE = flags.DEFINE_string("role", None, "The role of the current process.")
_CONTROLLER_ADDRESS = flags.DEFINE_string(
    "controller_address",
    "localhost:50051",
    "The address of the raidencontroller process.",
)
_NUM_VARIABLES = flags.DEFINE_integer(
    "num_variables",
    1,
    "Number of weight variables to synchronize.",
)
_VARIABLE_SHAPE = flags.DEFINE_list(
    "variable_shape",
    ["8192", "8192"],
    "Global shape of each weight variable.",
)
_DTYPE = flags.DEFINE_enum(
    "dtype",
    "bfloat16",
    ["float32", "bfloat16"],
    "Data type of the weight variables.",
)
_NUM_ITERATIONS = flags.DEFINE_integer(
    "num_iterations",
    10,
    "Number of weight synchronization iterations to run (after warmup).",
)
_PARALLELISM = flags.DEFINE_integer(
    "parallelism",
    8,
    "Number of parallel transport streams for weight synchronization.",
)
_MESH_SHAPE = flags.DEFINE_list(
    "mesh_shape",
    ["1", "1"],
    "Mesh shape for sharding (e.g. ['4', '4'] for 16 devices, or ['1', '1'] for"
    " 1 device).",
)


def _resolve_dtype(dtype_str: str) -> tuple[torch.dtype, int]:
  if dtype_str == "float32":
    return torch.float32, 4
  elif dtype_str == "bfloat16":
    return torch.bfloat16, 2
  else:
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _allocate_weights(
    num_variables: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    is_source: bool = True,
) -> list[torch.Tensor]:
  """Allocates weight tensors on TPU."""
  tensors = []
  for i in range(num_variables):
    if is_source:
      val = float(i + 1) * 0.125
      t = torch.full(shape, val, dtype=dtype, device=device)
    else:
      t = torch.zeros(shape, dtype=dtype, device=device)
    tensors.append(t)
  torch.tpu.synchronize()
  return tensors


def _build_variable_protos(
    num_variables: int, shape: list[int], item_size: int, mesh_shape: list[int]
) -> list[raiden_service_pb2.VariableMetadataProto]:
  """Builds controller metadata protos for weight tensors."""
  protos = []
  layout = list(range(len(shape) - 1, -1, -1))
  for i in range(num_variables):
    protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=f"var_{i}",
            shape=shape,
            mesh_shape=mesh_shape,
            layout=layout,
            item_size=item_size,
            layer_idx=i,
        )
    )
  return protos


def _resolve_local_ip() -> str:
  """Resolves the primary routable local IP address."""
  ip = "127.0.0.1"
  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      sock.connect(("10.255.255.255", 1))
      ip = sock.getsockname()[0]
    finally:
      sock.close()
  except OSError:
    pass
  if ":" in ip and not ip.startswith("["):
    return f"[{ip}]"
  return ip


def _source_worker_fn(local_rank: int) -> None:
  """Worker process running on a single TPU device."""
  os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
  os.environ["TPU_VISIBLE_DEVICES"] = str(local_rank)
  os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
  os.environ["LIBTPU_INIT_ARGS"] = (
      "--deepsea_hal_test_skip_slicebuilder=true"
      " --deepsea_hal_test_allow_multichip_skip_slicebuilder=true "
      + os.environ.get("LIBTPU_INIT_ARGS", "")
  )
  for k in (
      "MEGASCALE_COORDINATOR_ADDRESS",
      "MEGASCALE_NUM_SLICES",
      "MEGASCALE_SLICE_ID",
      "MEGASCALE_PORT",
  ):
    os.environ.pop(k, None)
  if not FLAGS.is_parsed():
    FLAGS(sys.argv, known_only=True)

  # 1. Calculate global rank across all machines
  devices_per_host = _DEVICES_PER_HOST.value
  host_id = _HOST_ID.value
  num_hosts = _NUM_HOSTS.value
  global_rank = (host_id * devices_per_host) + local_rank
  world_size = num_hosts * devices_per_host

  # 2. Initialize PyTorch distributed if multi-device / multi-host
  coordinator_addr = _DIST_COORDINATOR_ADDRESS.value
  if coordinator_addr and world_size > 1:
    dist.init_process_group(
        backend=_DIST_BACKEND.value,
        init_method=f"tcp://{coordinator_addr}",
        rank=global_rank,
        world_size=world_size,
    )

  # 3. Bind to local TPU device
  torch_tpu_common_loader.load_torch_tpu_common()
  try:
    torch.tpu.set_device(local_rank)
  except Exception:
    pass
  device = torch.device("tpu")
  torch_dtype, item_size = _resolve_dtype(_DTYPE.value)
  var_shape = [int(x) for x in _VARIABLE_SHAPE.value]
  mesh_shape = [int(x) for x in _MESH_SHAPE.value]
  local_shape = [d // m for d, m in zip(var_shape, mesh_shape)]

  # 4. Allocate local shard on TPU
  src_tensors = _allocate_weights(
      num_variables=_NUM_VARIABLES.value,
      shape=local_shape,
      dtype=torch_dtype,
      device=device,
      is_source=True,
  )

  device_tensors = [[t] for t in src_tensors]
  ws = weight_synchronizer.WeightSynchronizer(
      device_tensors=device_tensors,
      local_port=0,
      listener_port=0,
      parallelism=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True,
      auto_h2d=False,
  )

  # 6. Benchmark D2H
  # Warmup
  ws.d2h()
  torch.tpu.synchronize()
  if dist.is_initialized():
    dist.barrier()

  d2h_times = []
  num_iters = _NUM_ITERATIONS.value
  for _ in range(num_iters):
    t0 = time.perf_counter()
    ws.d2h()
    torch.tpu.synchronize()
    if dist.is_initialized():
      dist.barrier()
    d2h_times.append(time.perf_counter() - t0)

  # Only rank 0 prints the benchmark summary
  if global_rank == 0:
    total_bytes = (
        sum(int(np.prod(t.shape)) * item_size for t in src_tensors) * world_size
    )
    total_mb = total_bytes / (1024.0 * 1024.0)
    avg_d2h_time = float(np.mean(d2h_times))
    avg_d2h_bw_gbs = (total_bytes / 1e9) / max(avg_d2h_time, 1e-9)
    avg_d2h_bw_gbps = avg_d2h_bw_gbs * 8.0

    print("==================================================")
    print("SOURCE (TRAINER) BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Size:       {total_mb:.2f} MB across {world_size} ranks")
    print(f"Parallelism:      {_PARALLELISM.value}")
    print(f"Avg D2H Time:     {avg_d2h_time:.6f} s")
    print(
        f"Avg D2H BW:       {avg_d2h_bw_gbs:.2f} GB/s ({avg_d2h_bw_gbps:.2f}"
        " Gbps)"
    )
    print("==================================================", flush=True)

  # 7. Register this rank's shard with Controller
  self_ip = _resolve_local_ip()
  shards = [f"{self_ip}:{ws.local_port}"]
  protos = _build_variable_protos(
      num_variables=_NUM_VARIABLES.value,
      shape=var_shape,
      item_size=item_size,
      mesh_shape=mesh_shape,
  )

  ctrl_client = raiden_controller.RaidenControllerClientFacade(
      _CONTROLLER_ADDRESS.value,
      name_resolver=None,
  )
  unit_id = raiden_controller.RaidenId(
      "trainer", str(global_rank), "benchmark_weights"
  )
  logging.info("Rank %d registering with controller...", global_rank)
  ctrl_client.register_work_unit(
      unit=unit_id,
      shards=shards,
      control_plane_rpc_address=f"{self_ip}:{ws.listener_port}",
      variables=protos,
  )

  # 8. Wait for Sampler to complete benchmark
  while ws.is_listener_active:
    time.sleep(0.5)


def _destination_worker_fn(local_rank: int) -> None:
  """Worker process running on a single Sampler TPU device."""
  os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
  os.environ["TPU_VISIBLE_DEVICES"] = str(local_rank)
  os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
  os.environ["LIBTPU_INIT_ARGS"] = (
      "--deepsea_hal_test_skip_slicebuilder=true"
      " --deepsea_hal_test_allow_multichip_skip_slicebuilder=true "
      + os.environ.get("LIBTPU_INIT_ARGS", "")
  )
  for k in (
      "MEGASCALE_COORDINATOR_ADDRESS",
      "MEGASCALE_NUM_SLICES",
      "MEGASCALE_SLICE_ID",
      "MEGASCALE_PORT",
  ):
    os.environ.pop(k, None)
  if not FLAGS.is_parsed():
    FLAGS(sys.argv, known_only=True)

  # 1. Calculate global rank across all machines
  devices_per_host = _DEVICES_PER_HOST.value
  host_id = _HOST_ID.value
  num_hosts = _NUM_HOSTS.value
  global_rank = (host_id * devices_per_host) + local_rank
  world_size = num_hosts * devices_per_host

  # 2. Initialize PyTorch distributed if multi-device / multi-host
  coordinator_addr = _DIST_COORDINATOR_ADDRESS.value
  if coordinator_addr and world_size > 1:
    dist.init_process_group(
        backend=_DIST_BACKEND.value,
        init_method=f"tcp://{coordinator_addr}",
        rank=global_rank,
        world_size=world_size,
    )

  # 3. Bind to local TPU device and allocate blank target tensors (zeros)
  torch_tpu_common_loader.load_torch_tpu_common()
  try:
    torch.tpu.set_device(local_rank)
  except Exception:
    pass
  device = torch.device("tpu")
  torch_dtype, item_size = _resolve_dtype(_DTYPE.value)
  var_shape = [int(x) for x in _VARIABLE_SHAPE.value]
  mesh_shape = [int(x) for x in _MESH_SHAPE.value]
  local_shape = [d // m for d, m in zip(var_shape, mesh_shape)]

  dst_tensors = _allocate_weights(
      num_variables=_NUM_VARIABLES.value,
      shape=local_shape,
      dtype=torch_dtype,
      device=device,
      is_source=False,
  )

  device_tensors = [[t] for t in dst_tensors]
  ws = weight_synchronizer.WeightSynchronizer(
      device_tensors=device_tensors,
      local_port=0,
      listener_port=0,
      parallelism=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True,
      auto_h2d=False,
  )

  # 5. Register with Controller
  self_ip = _resolve_local_ip()
  shards = [f"{self_ip}:{ws.local_port}"]
  protos = _build_variable_protos(
      num_variables=_NUM_VARIABLES.value,
      shape=var_shape,
      item_size=item_size,
      mesh_shape=mesh_shape,
  )

  ctrl_client = raiden_controller.RaidenControllerClientFacade(
      _CONTROLLER_ADDRESS.value,
      name_resolver=None,
  )
  unit_id = raiden_controller.RaidenId(
      "sampler",
      str(global_rank),
      "benchmark_weights",
  )
  logging.info("Sampler Rank %d registering with controller...", global_rank)
  ctrl_client.register_work_unit(
      unit=unit_id,
      shards=shards,
      control_plane_rpc_address=f"{self_ip}:{ws.listener_port}",
      variables=protos,
  )

  # 6. Wait until all Trainer and Sampler units are registered
  if global_rank == 0:
    logging.info(
        "Waiting for all Trainer and Sampler work units to register..."
    )
  wait_start = time.time()
  src_units: list[raiden_controller.RaidenId] = []
  dst_units: list[raiden_controller.RaidenId] = []
  while True:
    metadata_list = ctrl_client.get_metadata()
    src_meta = [m for m in metadata_list if m.unit.job_name == "trainer"]
    dst_meta = [m for m in metadata_list if m.unit.job_name == "sampler"]
    if len(src_meta) >= world_size and len(dst_meta) >= world_size:
      src_units = sorted(
          [
              raiden_controller.RaidenId(
                  m.unit.job_name, m.unit.job_replica_id, m.unit.data_name
              )
              for m in src_meta
          ],
          key=lambda u: int(u.job_replica_id),
      )
      dst_units = sorted(
          [
              raiden_controller.RaidenId(
                  m.unit.job_name, m.unit.job_replica_id, m.unit.data_name
              )
              for m in dst_meta
          ],
          key=lambda u: int(u.job_replica_id),
      )
      break
    if time.time() - wait_start > 1800.0:
      raise RuntimeError(
          "Timed out waiting for Trainer and Sampler registrations"
      )
    time.sleep(0.5)

  if dist.is_initialized():
    dist.barrier()

  # 7. Timed Benchmark Loop (Iteration 0 is warmup + accuracy check)
  h2h_times: list[float] = []
  h2d_times: list[float] = []
  num_iters = _NUM_ITERATIONS.value

  for it in range(num_iters + 1):
    sync_uuid = 1000 + it

    if dist.is_initialized():
      dist.barrier()

    # A. Timed Network Transfer (H2H)
    t_h2h_start = time.perf_counter()
    if global_rank == 0:
      ctrl_client.coordinate_transfer(
          src_units=src_units,
          dst_units=dst_units,
          req_id=f"sync_{it}",
          use_block_chunks=True,
          uuid=sync_uuid,
      )
    ws.wait_for_transfer_completion(sync_uuid)
    if dist.is_initialized():
      dist.barrier()
    h2h_duration_s = time.perf_counter() - t_h2h_start

    # B. Timed Host-to-Device Copy (H2D)
    t_h2d_start = time.perf_counter()
    ws.h2d()
    torch.tpu.synchronize()
    if dist.is_initialized():
      dist.barrier()
    h2d_duration_s = time.perf_counter() - t_h2d_start

    # C. Verification on iteration 0 (Warmup)
    if it == 0:
      for idx, t in enumerate(dst_tensors):
        expected_val = float(idx + 1) * 0.125
        t_cpu = t.cpu()
        expected = torch.full_like(t_cpu, expected_val)
        if not torch.allclose(t_cpu, expected, atol=1e-3):
          raise AssertionError(f"Weight verification failed on variable {idx}!")
      if global_rank == 0:
        logging.info(
            "Iteration 0 warmup & accuracy check PASSED across all ranks!"
        )
    else:
      h2h_times.append(h2h_duration_s)
      h2d_times.append(h2d_duration_s)

  # 8. Report Results (Rank 0 only) & Shutdown
  if global_rank == 0:
    total_bytes = (
        sum(int(np.prod(t.shape)) * item_size for t in dst_tensors) * world_size
    )
    total_mb = total_bytes / (1024.0 * 1024.0)
    avg_h2h_time = float(np.mean(h2h_times))
    avg_h2h_bw_gbs = (total_bytes / 1e9) / max(avg_h2h_time, 1e-9)
    avg_h2h_bw_gbps = avg_h2h_bw_gbs * 8.0
    avg_h2d_time = float(np.mean(h2d_times))
    avg_h2d_bw_gbs = (total_bytes / 1e9) / max(avg_h2d_time, 1e-9)
    avg_h2d_bw_gbps = avg_h2d_bw_gbs * 8.0

    print("==================================================")
    print("DESTINATION (SAMPLER) BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Size:            {total_mb:.2f} MB across {world_size} ranks")
    print(f"Parallelism:           {_PARALLELISM.value}")
    print(f"Avg Net Transfer Time: {avg_h2h_time:.6f} s")
    print(
        f"Avg Net Transfer BW:   {avg_h2h_bw_gbs:.2f} GB/s"
        f" ({avg_h2h_bw_gbps:.2f} Gbps)"
    )
    print(f"Avg H2D Time:          {avg_h2d_time:.6f} s")
    print(
        f"Avg H2D BW:            {avg_h2d_bw_gbs:.2f} GB/s"
        f" ({avg_h2d_bw_gbps:.2f} Gbps)"
    )
    print("==================================================", flush=True)

    ctrl_client.shutdown()
    logging.info("Destination benchmark finished cleanly.")


def run_controller() -> None:
  """Runs the centralized RaidenController server."""
  port = int(_CONTROLLER_ADDRESS.value.split(":")[-1])
  logging.info(
      "Starting RaidenControllerServer on port %d (parallelism=%d)...",
      port,
      _PARALLELISM.value,
  )
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(
      name_resolver=None
  )
  controller = raiden_controller.RaidenController(
      port=port,
      worker_rpc_client=worker_rpc_client,
  )

  # Configure transfer defaults: skip_d2h=True because the source benchmarks
  # D2H independently before the network transfer begins.

  orig_start_transfer = controller.start_transfer
  controller.start_transfer = partial(
      orig_start_transfer,
      parallelism=_PARALLELISM.value,
      skip_d2h=True,
  )

  server = raiden_controller.RaidenControllerServer(controller)
  server.start()

  # Keep running until the destination finishes and calls ctrl_client.shutdown()
  start_time = time.time()
  while not server._server._stopped:
    if time.time() - start_time > 1800.0:
      raise RuntimeError(
          "Controller timed out waiting for benchmark completion"
      )
    time.sleep(0.5)

  # Clean shutdown of any connected workers and server
  loop = asyncio.new_event_loop()
  try:
    loop.run_until_complete(worker_rpc_client.shutdown_workers())
  finally:
    loop.close()
  server.stop()
  logging.info("RaidenControllerServer finished cleanly.")


def run_source() -> None:
  """Runs the PyTorch source (Trainer) benchmark worker."""
  if os.path.exists("/tmp/libtpu_lockfile"):
    try:
      os.remove("/tmp/libtpu_lockfile")
    except OSError:
      pass
  num_devices = _DEVICES_PER_HOST.value
  logging.info(
      "Spawning %d Trainer device workers on host %d...",
      num_devices,
      _HOST_ID.value,
  )
  mp.spawn(_source_worker_fn, nprocs=num_devices)


def run_destination() -> None:
  """Runs the PyTorch destination (Sampler) benchmark workers."""
  if os.path.exists("/tmp/libtpu_lockfile"):
    try:
      os.remove("/tmp/libtpu_lockfile")
    except OSError:
      pass
  num_devices = _DEVICES_PER_HOST.value
  logging.info(
      "Spawning %d Sampler workers on host %d...",
      num_devices,
      _HOST_ID.value,
  )
  mp.spawn(_destination_worker_fn, nprocs=num_devices)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _ROLE.value == "controller":
    run_controller()
  elif _ROLE.value == "source":
    run_source()
  elif _ROLE.value == "destination":
    run_destination()
  else:
    raise ValueError(f"Unknown role: {_ROLE.value}")


if __name__ == "__main__":
  app.run(main)

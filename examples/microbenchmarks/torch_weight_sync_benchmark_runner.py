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

r"""Cross-node PyTorch WeightSynchronizer benchmark (trainer -> sampler).

Measures controller-coordinated weight synchronization between two TPU hosts
running the PyTorch `WeightSynchronizer` on both ends. Each host runs one
process per local TPU chip (`--num_processes`, default = all local chips, i.e.
4 on a `2x2x1` node): the SOURCE host holds row-sharded ("FSDP") shards of
every weight and the DESTINATION host holds column-sharded ("TP") shards, so
every transfer is a genuine reshard across the NIC rather than a 1:1 copy.

The payload matches `jax_pathways_mcjax_weight_sync_benchmark_runner.py`
(8 x 8192x8192 bfloat16 = 1024 MB by default) so the two frameworks' numbers
are directly comparable. Per-stage metrics mirror the JAX runner too:
D2H on the source, Net transfer and H2D on the destination.

Start the SOURCE first; with --parallelism_list it hosts an in-process
`RaidenControllerServer` on port (29550 + p) for each rung and waits for the
destination to drive and then shut each rung down:

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \
    examples/microbenchmarks/torch_weight_sync_benchmark_runner.py \
      --role=source \
      --parallelism_list=1,4,8 \
      --src_log_dir=/tmp/raiden_ws

Then, for each rung P, the DESTINATION pointed at the source host:

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \
    examples/microbenchmarks/torch_weight_sync_benchmark_runner.py \
      --role=destination \
      --controller_address=10.128.0.241:$((29550 + P)) \
      --parallelism=$P

The payload flags must match on both sides.
"""

import asyncio
from collections import abc
import functools
import multiprocessing
import os
import pathlib
import socket
import time
from typing import Any, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
import torch_tpu  # pylint: disable=unused-import

from tpu_sync.api.torch import weight_synchronizer
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "role",
    None,
    ["source", "destination"],
    "Role of this host in the weight synchronization benchmark.",
    required=True,
)
flags.DEFINE_string(
    "controller_address",
    "localhost:29551",
    "Host:port of the RaidenController (destination only). The source hosts"
    " the controller itself on port (29550 + parallelism).",
)
flags.DEFINE_integer(
    "num_processes",
    0,
    "Processes (one per TPU chip) to spawn on this host. 0 = all local chips.",
)
flags.DEFINE_integer(
    "num_iterations",
    10,
    "Number of benchmarked synchronization iterations (after 1 warmup).",
)
flags.DEFINE_integer(
    "num_variables",
    8,
    "Number of weight variables to synchronize.",
)
flags.DEFINE_list(
    "variable_shape",
    ["8192", "8192"],
    "Global 2D shape of each weight variable.",
)
flags.DEFINE_enum(
    "dtype",
    "bfloat16",
    ["float32", "bfloat16"],
    "Data type of the weight variables.",
)
flags.DEFINE_integer(
    "parallelism",
    8,
    "Number of parallel transport streams for weight synchronization.",
)
flags.DEFINE_list(
    "parallelism_list",
    [],
    "Source only: comma-separated parallelism values to sweep. For each"
    " value p the source hosts a RaidenControllerServer on port (29550 + p).",
)
flags.DEFINE_integer(
    "group_size",
    0,
    "Number of variables grouped per transfer request. 0 = all variables.",
)
flags.DEFINE_string(
    "src_log_dir",
    "",
    "Optional directory to write torch_ws_src_p{p}.log files on the source.",
)
flags.DEFINE_string(
    "bind_ip",
    "",
    "IP to advertise for this host's data/control endpoints. Defaults to the"
    " primary routable local IP.",
)

_UNIT_NAME = "benchmark_weights"
_SRC_JOB = "torch_trainer"
_DST_JOB = "torch_sampler"
# Keep below Linux ip_local_port_range (32768..60999) so outbound connect()
# sockets in TIME_WAIT cannot collide with even controller ports.
_CONTROLLER_BASE_PORT = 29550
_REGISTRATION_TIMEOUT_S = 1800.0

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v4
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5p
    "0x0063": {1: "1,1,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5e
    "0x006f": {1: "1,1,1", 4: "2,2,1", 8: "2,4,1"},  # TPU v6e
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v7
}


# ----------------------------------------------------------------------------
# Host / TPU environment helpers (mirrors weight_synchronization_perf_test.py)
# ----------------------------------------------------------------------------


def _scan_pci_tpus() -> tuple[int, Optional[dict[int, str]]]:
  """Counts physical TPU chips via sysfs and returns their topology table."""
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


def _pick_unused_ports(count: int) -> list[int]:
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


def _prepare_tpu_environment(world_size: int) -> None:
  """Sets torch_tpu SPMD env vars for a multi-process single-host slice."""
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    ports = _pick_unused_ports(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"localhost:{p}" for p in ports
    )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    _, topology_map = _scan_pci_tpus()
    if topology_map and world_size in topology_map:
      os.environ["TORCH_TPU_TOPOLOGY"] = topology_map[world_size]
    elif world_size == 8:
      os.environ["TORCH_TPU_TOPOLOGY"] = "2,2,2"
    else:
      raise ValueError(
          f"No torch_tpu topology known for {world_size} local chips; set"
          " TORCH_TPU_TOPOLOGY explicitly."
      )


def _resolve_local_ip() -> str:
  """Resolves the primary routable local IP address."""
  if FLAGS.bind_ip:
    return FLAGS.bind_ip
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


def _barrier() -> None:
  if dist.is_available() and dist.is_initialized():
    dist.barrier()


def _resolve_torch_dtype(dtype_str: str) -> tuple[torch.dtype, int]:
  if dtype_str == "bfloat16":
    return torch.bfloat16, 2
  return torch.float32, 4


# ----------------------------------------------------------------------------
# Payload helpers
# ----------------------------------------------------------------------------


def _expected_block(
    rows: tuple[int, int], cols: tuple[int, int], var_idx: int
) -> np.ndarray:
  """Deterministic value grid for global coords [rows) x [cols) of var_idx.

  Every element is a function of its global coordinates so the destination can
  verify the resharded result without ever materializing a global array.
  """
  r = ((np.arange(rows[0], rows[1], dtype=np.int32) * 17) % 256).astype(
      np.int16
  )
  c = (np.arange(cols[0], cols[1], dtype=np.int32) % 256).astype(np.int16)
  grid = r[:, None] + c[None, :]
  np.remainder(grid, 256, out=grid)
  out = grid.astype(np.float32)
  out *= 0.001
  out += (var_idx % 50 + 1) * 0.01
  return out


def _shard_ranges(
    role: str, rank: int, world_size: int, var_shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
  """Row/col ranges of `rank`'s shard: source splits rows, destination cols."""
  n_rows, n_cols = var_shape
  if role == "source":
    chunk = n_rows // world_size
    return (rank * chunk, (rank + 1) * chunk), (0, n_cols)
  chunk = n_cols // world_size
  return (0, n_rows), (rank * chunk, (rank + 1) * chunk)


def _sharding_shape(role: str, world_size: int) -> list[int]:
  return [world_size, 1] if role == "source" else [1, world_size]


def _build_variable_protos(
    role: str,
    rank: int,
    world_size: int,
    var_shape: tuple[int, int],
    item_size: int,
    num_variables: int,
) -> list[raiden_service_pb2.VariableMetadataProto]:
  """Controller metadata for this rank's single shard of every variable.

  With a [N,1] (source) or [1,N] (destination) sharding shape the row-major
  global shard index of rank r's shard is simply r in both cases.
  """
  mesh_shape = _sharding_shape(role, world_size)
  layout = [1, 0]
  return [
      raiden_service_pb2.VariableMetadataProto(
          name=f"var_{i}",
          shape=list(var_shape),
          mesh_shape=mesh_shape,
          layout=layout,
          item_size=item_size,
          layer_idx=i,
          global_shard_indices=[rank],
      )
      for i in range(num_variables)
  ]


def _skip_tiling_list(var_shape: tuple[int, int], n: int) -> list[bool]:
  aligned = var_shape[-2] % 8 == 0 and var_shape[-1] % 128 == 0
  return [aligned] * n


def _allocate_tensors(
    role: str,
    rank: int,
    world_size: int,
    var_shape: tuple[int, int],
    num_variables: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
  rows, cols = _shard_ranges(role, rank, world_size, var_shape)
  tensors = []
  for i in range(num_variables):
    if role == "source":
      block = _expected_block(rows, cols, i)
      t = torch.from_numpy(np.ascontiguousarray(block)).to(
          device=device, dtype=dtype
      )
    else:
      t = torch.zeros(
          (rows[1] - rows[0], cols[1] - cols[0]), dtype=dtype, device=device
      )
    tensors.append(t)
  torch.accelerator.synchronize()
  return tensors


def _verify_destination(
    tensors: abc.Sequence[torch.Tensor],
    rank: int,
    world_size: int,
    var_shape: tuple[int, int],
) -> None:
  rows, cols = _shard_ranges("destination", rank, world_size, var_shape)
  for i, t in enumerate(tensors):
    local_cpu = t.cpu()
    expected = torch.from_numpy(_expected_block(rows, cols, i)).to(
        dtype=local_cpu.dtype
    )
    if torch.equal(local_cpu, expected):
      continue
    max_diff = torch.max(torch.abs(local_cpu.float() - expected.float()))
    if float(max_diff) > 1e-2:
      raise AssertionError(
          f"[dst rank {rank}] weight verification failed on var_{i}:"
          f" max_diff={float(max_diff)}"
      )


def _unit(job: str, rank: int) -> raiden_controller.RaidenId:
  return raiden_controller.RaidenId(job, str(rank), _UNIT_NAME)


def _ping_port(addr: str) -> bool:
  ip, port_str = addr.rsplit(":", 1)
  if ip.startswith("[") and ip.endswith("]"):
    ip = ip[1:-1]
  try:
    sock = socket.create_connection((ip, int(port_str)), timeout=1.0)
    sock.close()
    return True
  except OSError:
    return False


# ----------------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------------


def _report_source(
    d2h_times: abc.Sequence[float],
    total_bytes: int,
    parallelism: int,
    src_log_dir: str,
) -> None:
  total_mb = total_bytes / (1024.0 * 1024.0)
  avg_d2h_time = float(np.mean(d2h_times))
  avg_d2h_bw_gbs = (total_bytes / 1e9) / max(avg_d2h_time, 1e-9)
  report = (
      "\n".join([
          "==================================================",
          "SOURCE BENCHMARK RESULTS",
          "==================================================",
          f"Total Size:       {total_mb:.2f} MB",
          f"Parallelism:      {parallelism}",
          f"Avg D2H Time:     {avg_d2h_time:.6f} s",
          (
              f"Avg D2H BW:       {avg_d2h_bw_gbs:.2f} GB/s"
              f" ({avg_d2h_bw_gbs * 8.0:.2f} Gbps)"
          ),
          "==================================================",
      ])
      + "\n"
  )
  print(report, end="", flush=True)
  if src_log_dir:
    os.makedirs(src_log_dir, exist_ok=True)
    with open(
        os.path.join(src_log_dir, f"torch_ws_src_p{parallelism}.log"), "w"
    ) as f:
      f.write(report)


def _run_source_worker(rank: int, world_size: int, cfg: dict[str, Any]) -> None:
  """Source process: holds row shards, serves pushes, hosts the controller."""
  if world_size > 1:
    dist.init_process_group(backend="tpu_dist")
  device = torch.device("tpu")
  torch_dtype, item_size = _resolve_torch_dtype(cfg["dtype"])
  var_shape: tuple[int, int] = cfg["var_shape"]
  num_variables: int = cfg["num_variables"]
  p_values: list[int] = cfg["p_values"]
  self_ip: str = cfg["self_ip"]
  group_size = cfg["group_size"] or num_variables

  tensors = _allocate_tensors(
      "source", rank, world_size, var_shape, num_variables, device, torch_dtype
  )
  total_bytes = int(np.prod(var_shape)) * item_size * num_variables
  skip_tiling = _skip_tiling_list(var_shape, num_variables)
  skip_tiling_map = dict(enumerate(skip_tiling))

  # One synchronizer for all rungs, sized for the largest parallelism so the
  # push pool has enough workers for every rung (same as the JAX runner).
  ws = weight_synchronizer.WeightSynchronizer(
      [[t] for t in tensors],
      local_port=0,
      listener_port=0,
      parallelism=max(p_values),
      auto_h2d=False,
      global_shard_indices=[rank],
  )
  ws.test_only_set_skip_tiling(skip_tiling)
  protos = _build_variable_protos(
      "source", rank, world_size, var_shape, item_size, num_variables
  )
  data_addr = f"{self_ip}:{ws.local_port}"
  ctrl_addr = f"{self_ip}:{ws.listener_port}"
  while not _ping_port(ctrl_addr):
    time.sleep(0.2)
  _barrier()

  worker_rpc_client = None
  try:
    for p in p_values:
      ctrl_port = _CONTROLLER_BASE_PORT + p
      server = None
      if rank == 0:
        worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(
            name_resolver=None
        )
        if p != p_values[-1]:
          # Keep worker listeners alive between rungs: the destination's
          # shutdown RPC would otherwise tear down the shared synchronizer.
          async def _noop_shutdown(*unused_args: Any, **unused_kw: Any):
            return None

          worker_rpc_client.shutdown_workers = _noop_shutdown
        controller = raiden_controller.RaidenController(
            port=ctrl_port, worker_rpc_client=worker_rpc_client
        )
        controller.start_transfer = functools.partial(
            controller.start_transfer,
            parallelism=p,
            skip_d2h=True,
            skip_tiling=skip_tiling_map,
            group_size=group_size,
        )
        server = raiden_controller.RaidenControllerServer(controller)
        server.start()
        logging.info("Controller for parallelism=%d on port %d", p, ctrl_port)
      _barrier()

      # Stage 1: D2H, timed across all local ranks (slowest rank bounds it).
      d2h_times = []
      for _ in range(cfg["num_iterations"]):
        _barrier()
        t0 = time.perf_counter()
        ws.d2h().wait()
        _barrier()
        d2h_times.append(time.perf_counter() - t0)
      if rank == 0:
        _report_source(d2h_times, total_bytes, p, cfg["src_log_dir"])

      # Register this rank's shard with the rung's controller.
      ctrl_client = raiden_controller.RaidenControllerClientFacade(
          f"{self_ip}:{ctrl_port}", name_resolver=None
      )
      ctrl_client.register_work_unit(
          _unit(_SRC_JOB, rank),
          [data_addr],
          ctrl_addr,
          variables=protos,
      )

      # The destination drives the transfers and shuts the controller down.
      if server is not None:
        start = time.time()
        while not server._server._stopped:  # pylint: disable=protected-access
          if time.time() - start > _REGISTRATION_TIMEOUT_S:
            raise RuntimeError(
                f"Timed out waiting for parallelism={p} rung to complete"
            )
          time.sleep(0.5)
        server.stop()
        time.sleep(2.0)
      _barrier()
  finally:
    if rank == 0 and worker_rpc_client is not None:
      loop = asyncio.new_event_loop()
      try:
        loop.run_until_complete(worker_rpc_client.shutdown_workers())
      except Exception:  # pylint: disable=broad-except
        pass
      finally:
        loop.close()
    del ws
    if world_size > 1:
      dist.destroy_process_group()


# ----------------------------------------------------------------------------
# Destination
# ----------------------------------------------------------------------------


def _run_destination_worker(
    rank: int, world_size: int, cfg: dict[str, Any]
) -> None:
  """Destination process: holds column shards, drives one rung, verifies."""
  if world_size > 1:
    dist.init_process_group(backend="tpu_dist")
  device = torch.device("tpu")
  torch_dtype, item_size = _resolve_torch_dtype(cfg["dtype"])
  var_shape: tuple[int, int] = cfg["var_shape"]
  num_variables: int = cfg["num_variables"]
  parallelism: int = cfg["parallelism"]
  self_ip: str = cfg["self_ip"]

  tensors = _allocate_tensors(
      "destination",
      rank,
      world_size,
      var_shape,
      num_variables,
      device,
      torch_dtype,
  )
  total_bytes = int(np.prod(var_shape)) * item_size * num_variables

  ws = weight_synchronizer.WeightSynchronizer(
      [[t] for t in tensors],
      local_port=0,
      listener_port=0,
      parallelism=parallelism,
      unsafe_skip_buffer_lock=False,
      auto_h2d=False,
      global_shard_indices=[rank],
  )
  ws.test_only_set_skip_tiling(_skip_tiling_list(var_shape, num_variables))
  protos = _build_variable_protos(
      "destination", rank, world_size, var_shape, item_size, num_variables
  )
  ctrl_addr = f"{self_ip}:{ws.listener_port}"
  while not _ping_port(ctrl_addr):
    time.sleep(0.2)

  ctrl_client = raiden_controller.RaidenControllerClientFacade(
      cfg["controller_address"], name_resolver=None
  )
  ctrl_client.register_work_unit(
      _unit(_DST_JOB, rank),
      [f"{self_ip}:{ws.local_port}"],
      ctrl_addr,
      variables=protos,
  )

  # Wait until every source and destination rank has registered.
  def _to_id(m: Any) -> raiden_controller.RaidenId:
    return raiden_controller.RaidenId(
        m.unit.job_name,
        m.unit.job_replica_id,
        m.unit.data_name,
        m.unit.data_replica_idx,
    )

  src_units: list[raiden_controller.RaidenId] = []
  dst_units: list[raiden_controller.RaidenId] = []
  wait_start = time.time()
  while True:
    metadata = ctrl_client.get_metadata()
    src_meta = [m for m in metadata if m.unit.job_name == _SRC_JOB]
    dst_meta = [m for m in metadata if m.unit.job_name == _DST_JOB]
    if len(src_meta) >= cfg["num_src_shards"] and len(dst_meta) >= world_size:
      src_units = sorted(
          (_to_id(m) for m in src_meta), key=lambda u: int(u.job_replica_id)
      )
      dst_units = sorted(
          (_to_id(m) for m in dst_meta), key=lambda u: int(u.job_replica_id)
      )
      break
    if time.time() - wait_start > _REGISTRATION_TIMEOUT_S:
      raise RuntimeError(
          f"Timed out waiting for work units: src={len(src_meta)},"
          f" dst={len(dst_meta)}"
      )
    time.sleep(0.5)
  _barrier()

  net_times: list[float] = []
  h2d_times: list[float] = []
  try:
    for it in range(cfg["num_iterations"] + 1):
      sync_uuid = 1000 + it
      _barrier()
      t_net = time.perf_counter()
      if rank == 0:
        ctrl_client.coordinate_transfer(
            src_units=src_units,
            dst_units=dst_units,
            req_id=f"sync_{it}",
            use_block_chunks=True,
            uuid=sync_uuid,
        )
      ws.wait_for_transfer_completion(sync_uuid)
      _barrier()
      net_s = time.perf_counter() - t_net

      t_h2d = time.perf_counter()
      ws.h2d()
      torch.accelerator.synchronize()
      _barrier()
      h2d_s = time.perf_counter() - t_h2d

      if it == 0:
        _verify_destination(tensors, rank, world_size, var_shape)
        _barrier()
        if rank == 0:
          print("Warmup weight verification passed.", flush=True)
      else:
        net_times.append(net_s)
        h2d_times.append(h2d_s)

    if rank == 0:
      total_mb = total_bytes / (1024.0 * 1024.0)
      avg_net = float(np.mean(net_times))
      avg_h2d = float(np.mean(h2d_times))
      net_gbs = (total_bytes / 1e9) / max(avg_net, 1e-9)
      h2d_gbs = (total_bytes / 1e9) / max(avg_h2d, 1e-9)
      print("==================================================")
      print("DESTINATION BENCHMARK RESULTS")
      print("==================================================")
      print(f"Total Size:           {total_mb:.2f} MB")
      print(f"Parallelism:          {parallelism}")
      print(f"Avg Net Transfer Time:{avg_net:.6f} s")
      print(
          f"Avg Net Transfer BW:  {net_gbs:.2f} GB/s ({net_gbs * 8.0:.2f} Gbps)"
      )
      print(f"Avg H2D Time:         {avg_h2d:.6f} s")
      print(
          f"Avg H2D BW:           {h2d_gbs:.2f} GB/s ({h2d_gbs * 8.0:.2f} Gbps)"
      )
      print("==================================================", flush=True)
      ctrl_client.shutdown()
    _barrier()
  finally:
    del ws
    if world_size > 1:
      dist.destroy_process_group()


# ----------------------------------------------------------------------------
# Process launch
# ----------------------------------------------------------------------------


def _worker_entry(
    rank: int,
    world_size: int,
    master_port: int,
    fn: abc.Callable[[int, int, dict[str, Any]], None],
    cfg: dict[str, Any],
) -> None:
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
  try:
    fn(rank, world_size, cfg)
  except Exception:
    logging.exception("[%s rank %d] worker failed", cfg["role"], rank)
    raise


def _local_world_size() -> int:
  world_size = FLAGS.num_processes
  if world_size <= 0:
    world_size, _ = _scan_pci_tpus()
    if world_size <= 0:
      raise RuntimeError(
          "No TPU chips found via sysfs; pass --num_processes explicitly."
      )
  return world_size


def _launch(
    fn: abc.Callable[[int, int, dict[str, Any]], None],
    cfg: dict[str, Any],
    world_size: int,
) -> None:
  logging.info("%s: launching %d worker processes", cfg["role"], world_size)
  if world_size == 1:
    fn(0, 1, cfg)
    return
  _prepare_tpu_environment(world_size)
  master_port = _pick_unused_ports(1)[0]
  mp.spawn(
      _worker_entry,
      args=(world_size, master_port, fn, cfg),
      nprocs=world_size,
      join=True,
  )


def main(argv: abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  var_shape = tuple(int(x) for x in FLAGS.variable_shape)
  if len(var_shape) != 2:
    raise app.UsageError("--variable_shape must be 2D (rows,cols).")
  world_size = _local_world_size()
  cfg: dict[str, Any] = {
      "role": FLAGS.role,
      "var_shape": var_shape,
      "num_variables": FLAGS.num_variables,
      "dtype": FLAGS.dtype,
      "num_iterations": FLAGS.num_iterations,
      "group_size": FLAGS.group_size,
      "self_ip": _resolve_local_ip(),
      "src_log_dir": FLAGS.src_log_dir,
      "controller_address": FLAGS.controller_address,
      "parallelism": FLAGS.parallelism,
      "p_values": (
          [int(x) for x in FLAGS.parallelism_list]
          if FLAGS.parallelism_list
          else [FLAGS.parallelism]
      ),
      # Source and destination hosts are assumed symmetric (same chip count);
      # the destination only needs a lower bound on registered source shards.
      "num_src_shards": world_size,
  }
  if FLAGS.role == "source":
    _launch(_run_source_worker, cfg, world_size)
  else:
    _launch(_run_destination_worker, cfg, world_size)


if __name__ == "__main__":
  try:
    multiprocessing.set_start_method("spawn")
  except RuntimeError:
    pass
  app.run(main)

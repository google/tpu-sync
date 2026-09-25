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

import gc
import json
import os
import pathlib
import socket
import sys
import time
from typing import Sequence

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_tpu

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from tpu_sync.api.torch import kv_cache_manager

SUPPORTED_DTYPES = {
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float8_e4m3fn: "fp8",
    torch.int32: "int32",
}

FLAGS = flags.FLAGS
flags.DEFINE_string("locality", "default", "Locality under benchmark")
flags.DEFINE_string(
    "telemetry_log_path",
    "/tmp/torch_kv_cache_manager_perf_performance.jsonl",
    "Path to record benchmark telemetry",
)
flags.DEFINE_integer("benchmark_runs", 10, "Number of benchmark runs")


def log_telemetry(
    test_name: str,
    dtype: torch.dtype,
    num_layers: int,
    shape: Sequence[int],
    d2h_times: list[float],
    h2d_times: list[float],
) -> None:
  """Records benchmark telemetry stats to a JSONL file."""
  if not d2h_times or not h2d_times:
    return

  d2h_ms = np.array(d2h_times) * 1000.0
  h2d_ms = np.array(h2d_times) * 1000.0

  record = {
      "test_name": test_name,
      "locality": FLAGS.locality,
      "dtype": SUPPORTED_DTYPES.get(dtype, str(dtype)),
      "num_layers": int(num_layers),
      "shape": [int(s) for s in shape],
      "d2h_latency_mean_ms": float(np.mean(d2h_ms)),
      "d2h_latency_median_ms": float(np.median(d2h_ms)),
      "d2h_latency_stddev_ms": (
          float(np.std(d2h_ms, ddof=1)) if len(d2h_ms) > 1 else 0.0
      ),
      "h2d_latency_mean_ms": float(np.mean(h2d_ms)),
      "h2d_latency_median_ms": float(np.median(h2d_ms)),
      "h2d_latency_stddev_ms": (
          float(np.std(h2d_ms, ddof=1)) if len(h2d_ms) > 1 else 0.0
      ),
      "timestamp": float(time.time()),
  }

  log_dir = os.path.dirname(FLAGS.telemetry_log_path)
  if log_dir:
    os.makedirs(log_dir, exist_ok=True)

  with open(FLAGS.telemetry_log_path, "a") as f:
    f.write(json.dumps(record) + "\n")


_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v4
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5p
    "0x0063": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5e
    "0x006f": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,4,1"},  # TPU v6e
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v7
}


def _scan_pci_tpus() -> tuple[int, dict[int, str] | None]:
  """Scans PCI bus to identify local physical TPU device IDs and topology."""
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


def get_tpu_topology(world_size: int) -> str:
  """Resolves the TPU topology string for the current slice."""
  _, topology_map = _scan_pci_tpus()
  if topology_map and world_size in topology_map:
    return topology_map[world_size]
  return (
      "2x4"
      if world_size == 8
      else "2x2"
      if world_size == 4
      else f"1x{world_size}"
  )


def pick_unused_ports(count: int = 1) -> list[int]:
  """Finds available ephemeral TCP ports."""
  sockets = []
  ports = []
  for _ in range(count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    ports.append(s.getsockname()[1])
    sockets.append(s)
  for s in sockets:
    s.close()
  return ports


def prepare_tpu_environment(world_size: int) -> None:
  """Configures necessary TPU log dirs, rendezvous ports, and mesh topology."""
  log_dir = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
  os.environ["TPU_LOG_DIR"] = log_dir
  os.environ["GLOG_log_dir"] = log_dir
  os.environ["GOOGLE_LOG_DIR"] = log_dir
  os.environ["TMPDIR"] = log_dir
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    ports = pick_unused_ports(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        [f"localhost:{p}" for p in ports]
    )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    os.environ["TORCH_TPU_TOPOLOGY"] = get_tpu_topology(world_size)


def create_tpu_tensors(
    num_layers: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
  """Allocates KV cache tensors on TPU HBM."""
  tensors = [
      torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
  ]
  torch.tpu.synchronize()
  return tensors


def mutate_tpu_tensor(tensor: torch.Tensor) -> None:
  """Mutates tensor in-place on TPU to bypass PJRT shadow host buffers."""
  tensor.add_(1 if tensor.dtype == torch.int32 else 0.01)


def _sync_max_time(local_time: float) -> float:
  tensor = torch.tensor([local_time], dtype=torch.float64)
  dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
  return float(tensor.item())


def _benchmark_raiden(
    src_tensors: list[torch.Tensor], num_blocks: int, benchmark_runs: int
) -> tuple[list[float], list[float]]:
  """Benchmarks raiden's KVCacheManager."""
  manager = kv_cache_manager.KVCacheManager(
      kv_caches=[[t] for t in src_tensors],
      local_control_port=0,
      host_blocks_to_allocate=num_blocks,
      unsafe_skip_buffer_lock=False,
  )

  offsets = [0]
  sizes = [num_blocks]
  d2h_times = []
  h2d_times = []

  for _ in range(benchmark_runs):
    for arr in src_tensors:
      mutate_tpu_tensor(arr)
    torch.tpu.synchronize()
    dist.barrier()

    # D2H: Transfer TPU -> Host DMA pool
    gc.disable()
    start = time.perf_counter()
    future = manager.d2h(
        src_offsets=offsets,
        dst_offsets=offsets,
        copy_sizes=sizes,
    )
    future.Await()
    torch.tpu.synchronize()
    local_time = time.perf_counter() - start
    gc.enable()
    gc.collect()
    d2h_times.append(_sync_max_time(local_time))

    # H2D: Transfer Host DMA pool -> TPU
    gc.disable()
    start = time.perf_counter()
    future = manager.h2d(
        src_offsets=offsets,
        dst_offsets=offsets,
        copy_sizes=sizes,
    )
    future.Await()
    torch.tpu.synchronize()
    local_time = time.perf_counter() - start
    gc.enable()
    gc.collect()
    h2d_times.append(_sync_max_time(local_time))

  return d2h_times, h2d_times


def _benchmark_torch_copy(
    src_tensors: list[torch.Tensor],
    shard_shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    benchmark_runs: int,
    pinned: bool,
) -> tuple[list[float], list[float]]:
  """Benchmarks native PyTorch .copy_() (pinned or standard unpinned host memory)."""
  num_layers = len(src_tensors)
  host_tensors = [
      torch.zeros(shard_shape, dtype=dtype, device="cpu", pin_memory=pinned)
      for _ in range(num_layers)
  ]
  tpu_dst_tensor = [
      torch.zeros(shard_shape, dtype=dtype, device=device)
      for _ in range(num_layers)
  ]
  torch.tpu.synchronize()

  d2h_times = []
  h2d_times = []

  for _ in range(benchmark_runs):
    for arr in src_tensors:
      mutate_tpu_tensor(arr)
    torch.tpu.synchronize()
    dist.barrier()

    # D2H: Copy TPU -> Host
    gc.disable()
    start = time.perf_counter()
    for j in range(num_layers):
      host_tensors[j].copy_(src_tensors[j], non_blocking=pinned)
    torch.tpu.synchronize()
    local_time = time.perf_counter() - start
    gc.enable()
    gc.collect()
    d2h_times.append(_sync_max_time(local_time))

    # H2D: Copy Host -> TPU
    gc.disable()
    start = time.perf_counter()
    for j in range(num_layers):
      tpu_dst_tensor[j].copy_(host_tensors[j], non_blocking=pinned)
    torch.tpu.synchronize()
    local_time = time.perf_counter() - start
    gc.enable()
    gc.collect()
    h2d_times.append(_sync_max_time(local_time))

  return d2h_times, h2d_times


def _report_results(
    leg_name: str,
    dtype: torch.dtype,
    num_layers: int,
    shape: Sequence[int],
    d2h_times: list[float],
    h2d_times: list[float],
    test_name: str,
    use_telemetry: bool = False,
) -> None:
  """Calculates aggregate cluster bandwidth and prints/logs benchmark results."""
  element_size = torch.tensor([], dtype=dtype).element_size()
  total_bytes = np.prod(shape) * element_size * num_layers

  d2h_median = float(np.median(d2h_times))
  h2d_median = float(np.median(h2d_times))
  d2h_bw = total_bytes / d2h_median / (1024**3)
  h2d_bw = total_bytes / h2d_median / (1024**3)

  lines = [
      (
          f"[{dtype}, {num_layers} layers, shape={shape}] {leg_name}"
          f" d2h avg time: {d2h_median:.6f} s (median)"
      ),
      (
          f"[{dtype}, {num_layers} layers, shape={shape}] {leg_name}"
          f" h2d avg time: {h2d_median:.6f} s (median)"
      ),
      (
          f"[{dtype}, {num_layers} layers, shape={shape}] {leg_name}"
          f" D2H bandwidth: {d2h_bw:.3f} GB/s"
      ),
      (
          f"[{dtype}, {num_layers} layers, shape={shape}] {leg_name}"
          f" H2D bandwidth: {h2d_bw:.3f} GB/s"
      ),
  ]
  for line in lines:
    print(line, flush=True)
    sys.stderr.write(line + "\n")
  sys.stderr.flush()

  if use_telemetry:
    log_telemetry(test_name, dtype, num_layers, shape, d2h_times, h2d_times)


def _mpmd_benchmark_worker(
    rank: int,
    world_size: int,
    master_port: int,
    test_name: str,
    dtype: torch.dtype,
    num_layers: int,
    shape: Sequence[int],
    benchmark_runs: int,
    locality: str,
    telemetry_log_path: str,
) -> None:
  """Runs the benchmark worker for one distributed rank."""
  os.environ.pop("TPU_PREMAPPED_BUFFER_SIZE", None)
  os.environ.pop("stairways_ifrt_tpu_premapped_buffer_size", None)
  os.environ.pop("pathways_tpu_premapped_buffer_size", None)
  os.environ["TPU_VISIBLE_DEVICES"] = str(rank)
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

  FLAGS.locality = locality
  FLAGS.telemetry_log_path = telemetry_log_path

  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{master_port}",
      rank=rank,
      world_size=world_size,
  )

  try:
    device = torch.device("tpu")
    shard_shape = list(shape)
    if len(shape) > 2:
      # shard axis 2 (num_kv_heads) across TPU ranks.
      shard_shape[2] //= world_size

    src_tensors = create_tpu_tensors(num_layers, shard_shape, dtype, device)
    num_blocks = shard_shape[0]

    # Branch 1: Raiden Zero-Copy DMA
    mgr_d2h, mgr_h2d = _benchmark_raiden(
        src_tensors, num_blocks, benchmark_runs
    )
    if rank == 0:
      _report_results(
          "KVCacheManager",
          dtype,
          num_layers,
          shape,
          mgr_d2h,
          mgr_h2d,
          test_name,
          use_telemetry=True,
      )

    # Branch 2: PyTorch Pinned Host Baseline
    pin_d2h, pin_h2d = _benchmark_torch_copy(
        src_tensors, shard_shape, dtype, device, benchmark_runs, pinned=True
    )
    if rank == 0:
      _report_results(
          "PyTorch Pinned",
          dtype,
          num_layers,
          shape,
          pin_d2h,
          pin_h2d,
          test_name,
      )

    # Branch 3: PyTorch Standard Host Baseline
    std_d2h, std_h2d = _benchmark_torch_copy(
        src_tensors, shard_shape, dtype, device, benchmark_runs, pinned=False
    )
    if rank == 0:
      _report_results(
          "PyTorch Standard",
          dtype,
          num_layers,
          shape,
          std_d2h,
          std_h2d,
          test_name,
      )

  finally:
    dist.destroy_process_group()


class KVCacheManagerPerfTest(parameterized.TestCase):
  device_count: int = 0

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    os.environ.pop("TPU_PREMAPPED_BUFFER_SIZE", None)
    os.environ.pop("stairways_ifrt_tpu_premapped_buffer_size", None)
    os.environ.pop("pathways_tpu_premapped_buffer_size", None)
    cls.device_count, _ = _scan_pci_tpus()

  def setUp(self):
    super().setUp()
    if self.device_count == 0:
      self.skipTest("No TPU devices found")

  def tearDown(self):
    gc.collect()
    time.sleep(0.5)
    super().tearDown()

  def _run_benchmark(
      self,
      test_name: str,
      dtype: torch.dtype,
      num_layers: int,
      shape: Sequence[
          int
      ],  # (num_blocks, block_size, num_kv_head, k_or_v, head_dim)
  ) -> None:
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")

    world_size = self.device_count
    if len(shape) > 2 and shape[2] % world_size != 0:
      self.skipTest(
          f"Shape axis 2 ({shape[2]}) is not divisible by world_size"
          f" ({world_size})"
      )

    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]

    mp.spawn(
        _mpmd_benchmark_worker,
        args=(
            world_size,
            master_port,
            test_name,
            dtype,
            num_layers,
            shape,
            FLAGS.benchmark_runs,
            FLAGS.locality,
            FLAGS.telemetry_log_path,
        ),
        nprocs=world_size,
        join=True,
    )

  @parameterized.named_parameters(
      ("bf16", torch.bfloat16, 64, 16),
      ("f32", torch.float32, 64, 16),
      ("f8", torch.float8_e4m3fn, 64, 16),
      ("int32_1_layer", torch.int32, 1, 16),
      ("int32_64_layers", torch.int32, 64, 16),
      ("int32_128_layers", torch.int32, 128, 16),
  )
  def test_kv_cache_perf_compare(
      self, dtype: torch.dtype, num_layers: int, num_blocks: int
  ) -> None:
    shape = (num_blocks, 128, 8, 2, 128)
    self._run_benchmark("kv_cache_manager_perf_test", dtype, num_layers, shape)

  @parameterized.named_parameters(
      ("1_layers_bf16", 1, (8, 128, 1024, 128), torch.bfloat16),
      ("1_layers_fp32", 1, (8, 128, 1024, 128), torch.float32),
      ("1_layers_fp8", 1, (8, 128, 1024, 128), torch.float8_e4m3fn),
      ("1_layers_int32", 1, (8, 128, 1024, 128), torch.int32),
      ("2_layer_int32", 2, (8, 128, 1024, 128), torch.int32),
      ("4_layer_int32", 4, (8, 128, 1024, 128), torch.int32),
      ("8_layer_int32", 8, (8, 128, 1024, 128), torch.int32),
  )
  def test_large_shape_perf_compare(
      self, num_layers: int, shape: Sequence[int], dtype: torch.dtype
  ) -> None:
    self._run_benchmark(
        "kv_cache_manager_perf_test_large_shape", dtype, num_layers, shape
    )


if __name__ == "__main__":
  mp.set_start_method("spawn", force=True)
  absltest.main()

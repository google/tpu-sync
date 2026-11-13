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

"""Multiprocess distributed (MPMD) PyTorch KVCacheManager D2H/H2D test."""

import os
import pathlib
import socket
import time

from absl.testing import absltest
import numpy as np

# Set log directories BEFORE importing torch so spawned workers don't fail on
# /tmp/tpu_logs
_LOG_DIR = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("TPU_LOG_DIR", _LOG_DIR)
os.environ.setdefault("GLOG_log_dir", _LOG_DIR)
os.environ.setdefault("GOOGLE_LOG_DIR", _LOG_DIR)
os.environ.setdefault("TMPDIR", _LOG_DIR)

import torch
import torch_tpu
import torch.distributed as dist
import torch.multiprocessing as mp

from tpu_sync.frameworks.torch import _tpu_raiden_torch as _kv_cache_manager

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v4
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5p
    "0x0063": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5e
    "0x006f": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,4,1"},  # TPU v6e
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v7
}


def _scan_pci_tpus():
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


def pick_unused_ports(count: int) -> list[int]:
  ports = []
  for _ in range(count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    ports.append(port)
    s.close()
  return ports


def prepare_tpu_environment(world_size: int) -> None:
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


def _mpmd_worker_fn(rank: int, world_size: int, master_port: int) -> None:
  # Ensure zero environment variable dependency on premapped DMA pool
  os.environ.pop("TPU_PREMAPPED_BUFFER_SIZE", None)
  os.environ.pop("stairways_ifrt_tpu_premapped_buffer_size", None)
  os.environ.pop("pathways_tpu_premapped_buffer_size", None)
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

  # Initialize PyTorch Gloo distributed process group across all 8 ranks
  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{master_port}",
      rank=rank,
      world_size=world_size,
  )

  try:
    device = torch.device("tpu")

    block_size = 65536  # 65536 blocks = 256MB transfer payload
    shape = (block_size, 128, 8)
    bytes_per_rank = block_size * 128 * 8 * 4  # 268,435,456 bytes (256MB)

    # 1. Allocate eager tensor on physical TPU core with random float values
    # to avoid constant page compression / zero skipping
    torch.manual_seed(42 + rank)
    cpu_initial_data = torch.randn(shape, dtype=torch.float32)
    t = cpu_initial_data.to(device)
    expected_ref = cpu_initial_data.numpy()

    # 2. Instantiate KVCacheManager per rank with an extreme 32GB host memory
    # buffer pool (8388608 blocks = 32GB per rank -> 256GB total pinned DMA
    # memory across 8 ranks)
    start_alloc = time.perf_counter()
    manager = _kv_cache_manager.KVCacheManager(
        [[t]],
        local_port=0,
        host_blocks_to_allocate=8388608,
        parallelism=1,
    )
    alloc_latency = time.perf_counter() - start_alloc

    # Synchronize all 8 worker ranks before concurrently stressing OS mmap,
    # PjRtClient::DmaMap, and copy engines
    dist.barrier()

    # Warmup transfer
    warmup_future = manager.D2h(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[16],
    )
    warmup_future.Await()
    dist.barrier()

    # 3. Benchmark local D2H copies from TPU HBM offset 0 into the local
    # pinned host DMA pool
    num_trials = 3
    d2h_rank_times = []
    d2h_wall_times = []
    for _ in range(num_trials):
      dist.barrier()
      t_start = time.perf_counter()
      d2h_future = manager.D2h(
          src_offsets_major_dim=[0],
          dst_offsets_major_dim=[0],
          copy_sizes_major_dim=[block_size],
      )
      d2h_future.Await()
      t_end = time.perf_counter()
      d2h_rank_times.append(t_end - t_start)

      # Measure global concurrent wall-clock across all ranks
      min_start = torch.tensor([t_start], dtype=torch.float64)
      dist.all_reduce(min_start, op=dist.ReduceOp.MIN)
      max_end = torch.tensor([t_end], dtype=torch.float64)
      dist.all_reduce(max_end, op=dist.ReduceOp.MAX)
      d2h_wall_times.append((max_end - min_start).item())
      dist.barrier()

    d2h_rank_time = np.median(d2h_rank_times)
    d2h_wall_time = np.median(d2h_wall_times)
    d2h_rank_bw = (
        bytes_per_rank / (d2h_rank_time * 1e9) if d2h_rank_time > 0 else 0.0
    )
    d2h_aggregate_bw = (
        (world_size * bytes_per_rank) / (d2h_wall_time * 1e9)
        if d2h_wall_time > 0
        else 0.0
    )

    # 4. Benchmark local H2D copies from the local pinned host DMA pool back
    # to TPU HBM offset 0
    h2d_rank_times = []
    h2d_wall_times = []
    for _ in range(num_trials):
      dist.barrier()
      t_start = time.perf_counter()
      h2d_future = manager.H2d(
          src_offsets_major_dim=[0],
          dst_offsets_major_dim=[0],
          copy_sizes_major_dim=[block_size],
      )
      h2d_future.Await()
      t_end = time.perf_counter()
      h2d_rank_times.append(t_end - t_start)

      # Measure global concurrent wall-clock across all ranks
      min_start = torch.tensor([t_start], dtype=torch.float64)
      dist.all_reduce(min_start, op=dist.ReduceOp.MIN)
      max_end = torch.tensor([t_end], dtype=torch.float64)
      dist.all_reduce(max_end, op=dist.ReduceOp.MAX)
      h2d_wall_times.append((max_end - min_start).item())
      dist.barrier()

    h2d_rank_time = np.median(h2d_rank_times)
    h2d_wall_time = np.median(h2d_wall_times)
    h2d_rank_bw = (
        bytes_per_rank / (h2d_rank_time * 1e9) if h2d_rank_time > 0 else 0.0
    )
    h2d_aggregate_bw = (
        (world_size * bytes_per_rank) / (h2d_wall_time * 1e9)
        if h2d_wall_time > 0
        else 0.0
    )

    # 5. Verify complete local roundtrip numerical restoration on hardware
    actual_data = t.cpu().numpy()
    np.testing.assert_allclose(actual_data, expected_ref, atol=1e-5)

    print(
        f"[Rank {rank}] 32GB Alloc: {alloc_latency:.4f}s | "
        f"D2H Rank Time: {d2h_rank_time*1000:.2f}ms (Rank BW:"
        f" {d2h_rank_bw:.2f} GB/s) | "
        f"H2D Rank Time: {h2d_rank_time*1000:.2f}ms (Rank BW:"
        f" {h2d_rank_bw:.2f} GB/s)",
        flush=True,
    )
    if rank == 0:
      print(
          f"[CONCURRENT MPMD SUMMARY] Total Payload: {world_size * 256} MB | "
          f"D2H Concurrent Wall Time: {d2h_wall_time*1000:.2f}ms (True"
          f" Aggregate BW: {d2h_aggregate_bw:.2f} GB/s) | "
          f"H2D Concurrent Wall Time: {h2d_wall_time*1000:.2f}ms (True"
          f" Aggregate BW: {h2d_aggregate_bw:.2f} GB/s)",
          flush=True,
      )

  finally:
    dist.barrier()
    dist.destroy_process_group()


class KVCacheManagerMpmdTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    os.environ.pop("TPU_PREMAPPED_BUFFER_SIZE", None)
    os.environ.pop("stairways_ifrt_tpu_premapped_buffer_size", None)
    os.environ.pop("pathways_tpu_premapped_buffer_size", None)

  def test_mpmd_8rank_local_d2h_and_h2d_stress_roundtrip(self):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    mp.spawn(
        _mpmd_worker_fn,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn", force=True)
  absltest.main()

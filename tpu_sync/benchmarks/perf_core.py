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

"""Shared measurement core for the h2d/d2h benchmark and its perf gate.

Deliberately flag-free: both the benchmark binary (multi_host_perf_test_oss)
and the gating binary (h2d_d2h_benchmark_gating) import measure() from here, so
neither leaks absl flags into the other and the transfer code is reused, not
duplicated.
"""

import ctypes
import gc
import os
import pathlib
import socket
import sys
import time
from typing import Optional

import numpy as np

ITEMSIZE = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int32": 4,
    "float8_e4m3fn": 1,
}

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {4: "2,2,1", 8: "2,4,1"},  # TPU v5e
    "0x0063": {4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v5p
    "0x006f": {4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v6e
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v7
}


# ---------------- PCI / TPU Hardware Helpers ----------------
def _scan_pci_tpus() -> tuple[int, Optional[dict[int, str]]]:
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


def get_tpu_device_count() -> int:
  count, _ = _scan_pci_tpus()
  return count


def get_tpu_topology(world_size: int) -> str:
  _, topology_map = _scan_pci_tpus()
  if topology_map and world_size in topology_map:
    return topology_map[world_size]
  return "2x4" if world_size == 8 else f"1x{world_size}"


def pick_unused_ports(count: int = 1) -> list[int]:
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
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    ports = pick_unused_ports(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        [f"localhost:{p}" for p in ports]
    )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    os.environ["TORCH_TPU_TOPOLOGY"] = get_tpu_topology(world_size)


# ---------------- NUMA (optional experiment knob) ----------------
def _cpus_of_node(node):
  try:
    spec = open(f'/sys/devices/system/node/node{node}/cpulist').read().strip()
  except OSError:
    return set()
  cpus = set()
  for part in spec.split(','):
    if '-' in part:
      a, b = part.split('-'); cpus.update(range(int(a), int(b) + 1))
    elif part:
      cpus.add(int(part))
  return cpus


def bind_to_numa(node):
  want = _cpus_of_node(node)
  allowed = os.sched_getaffinity(0)
  cpus = want & allowed
  if cpus:
    try:
      os.sched_setaffinity(0, cpus)
    except OSError as e:
      print(f'WARNING: sched_setaffinity failed: {e}', file=sys.stderr)
  else:
    print(f'WARNING: no allowed cores on NUMA node {node}.', file=sys.stderr)
  try:
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    nodemask = ctypes.c_ulong(1 << node)
    if libc.syscall(238, 2, ctypes.byref(nodemask), ctypes.c_ulong(64)) == 0:  # set_mempolicy(MPOL_BIND)
      print(f'[numa] pinned to node {node} (mem; cpus={len(cpus)}).')
    else:
      print(f'WARNING: set_mempolicy errno={ctypes.get_errno()}', file=sys.stderr)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: NUMA membind unavailable: {e}', file=sys.stderr)


def summarize(values):
  a = np.array(values, dtype=float)
  return {'min': float(a.min()), 'p50': float(np.median(a)), 'mean': float(a.mean()),
          'p90': float(np.percentile(a, 90)), 'p99': float(np.percentile(a, 99)),
          'max': float(a.max()), 'stddev': float(a.std(ddof=1)) if len(a) > 1 else 0.0}


# ---------------- JAX Backend Implementation ----------------
def _create_sharded_array_jax(
    shape, sharding, dtype, is_host=False, is_random=False
):
  """Places each shard on its own device (pinned_host if is_host) -> NUMA-local per chip."""
  import jax
  import jax.numpy as jnp

  mesh, spec = sharding.mesh, sharding.spec
  devices = list(mesh.devices.flat)
  shard_shape = list(shape)
  shard_axis = None
  for i, axis in enumerate(spec):
    if axis is not None:
      shard_axis = i
      break
  if shard_axis is not None:
    shard_shape[shard_axis] = shape[shard_axis] // len(devices)

  shards = []
  for idx, device in enumerate(devices):
    sd = (
        jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
        if is_host
        else jax.sharding.SingleDeviceSharding(device)
    )
    if is_random:
      shard_np = np.random.uniform(0, 1, shard_shape).astype(np.float32)
    elif dtype == jnp.int32:
      start = idx * int(np.prod(shard_shape))
      shard_np = (
          np.arange(np.prod(shard_shape), dtype=np.int32) + start
      ).reshape(shard_shape)
    else:
      shard_np = np.zeros(shard_shape, dtype=np.float32)
    shards.append(jax.device_put(shard_np, sd).astype(dtype))
  return jax.make_array_from_single_device_arrays(shape, sharding, shards)


def _verify_roundtrip_jax(manager, dev_arrs, num_blocks):
  """One-shot d2h->h2d data-integrity check (ported from V2 verify_device_cache).

  Uses the SAME manager + d2h/h2d as the benchmark, but with disjoint offsets
  and OUTSIDE the timed loop, so it changes neither the measured Gbps nor the
  recorded baselines. It pulls the source-half blocks [0:half] down to host,
  pushes them back up to the *distinct* destination half [half:2*half] on device,
  then asserts the destination now equals the source. The destination half
  starts with different values (see create_sharded_array init), so a no-op,
  misaligned, or corrupting transfer leaves the two halves unequal and fails.
  Raises AssertionError on mismatch -> the gate exits non-zero.
  """
  half = num_blocks // 2
  if half == 0:
    return  # major dim too small to split into src/dst halves; skip check
  # ONE bulk descriptor (copy_sizes=[half]) -- matches once()'s
  # offsets=[0], sizes=[num_blocks], so verify exercises the SAME bulk-copy path
  # that the timed/recorded copy uses, not a per-block scatter/gather.
  # d2h: device blocks [0:half] -> host [0:half]; h2d: host [0:half] -> device [half:2*half].
  manager.d2h(src_offsets_major_dim=[0], dst_offsets_major_dim=[0],
              copy_sizes_major_dim=[half]).Await()
  manager.h2d(src_offsets_major_dim=[0], dst_offsets_major_dim=[half],
              copy_sizes_major_dim=[half]).Await()
  # Read the raw shard buffers (like V2), not an XLA slice, so we observe the
  # bytes the manager actually wrote and never a cached/stale view.
  for li, a in enumerate(dev_arrs):
    for s in a.addressable_shards:
      d = np.asarray(s.data)
      np.testing.assert_array_equal(
          d[half : 2 * half],
          d[0:half],
          err_msg=f"CORRUPTION: layer {li} d2h/h2d round-trip mismatch",
      )


def _measure_jax(
    shape,
    num_layers,
    dtype,
    shard_axis=2,
    iters=20,
    warmup=3,
    lock_buffers=True,
    verify=True,
):
  """Executes the JAX transfer benchmark."""
  import jax
  import jax.numpy as jnp
  from tpu_sync.frameworks.jax import _tpu_raiden_jax as kv_cache_manager

  dtype_map = {
      "float32": jnp.float32,
      "bfloat16": jnp.bfloat16,
      "float16": jnp.float16,
      "int32": jnp.int32,
      "float8_e4m3fn": jnp.float8_e4m3fn,
  }

  devices = jax.devices("tpu")
  if not devices:
    raise RuntimeError("No TPU devices found.")
  num_devices = len(devices)
  num_blocks = shape[0]
  dt = dtype_map.get(dtype, jnp.float32)
  itemsize = ITEMSIZE.get(dtype, 4)

  mesh = jax.sharding.Mesh(
      np.array(devices).reshape(1, num_devices), ("data", "model")
  )
  spec = jax.sharding.PartitionSpec(
      *[("model" if i == shard_axis else None) for i in range(len(shape))]
  )
  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)

  src_arrs = [
      _create_sharded_array_jax(
          shape, tpu_sharding, dt, is_host=False, is_random=(dt != jnp.int32)
      )
      for _ in range(num_layers)
  ]
  jax.block_until_ready(src_arrs)

  mutate = jax.jit(
      lambda x: x
      + jnp.array(1 if x.dtype == jnp.int32 else 0.01, dtype=x.dtype)
  )

  manager = kv_cache_manager.KVCacheManager(
      device_arrays=src_arrs,
      host_blocks_to_allocate=num_blocks,
      unsafe_skip_buffer_lock=not lock_buffers,
  )

  if verify:
    _verify_roundtrip_jax(manager, src_arrs, num_blocks)

  offsets, sizes = [0], [num_blocks]
  total_bytes = num_layers * int(np.prod(shape)) * itemsize

  def once():
    nonlocal src_arrs
    src_arrs = [mutate(a) for a in src_arrs]
    jax.block_until_ready(src_arrs)
    gc.disable()
    t0 = time.perf_counter()
    manager.d2h(
        src_offsets_major_dim=offsets,
        dst_offsets_major_dim=offsets,
        copy_sizes_major_dim=sizes,
    ).Await()
    d2h = time.perf_counter() - t0
    gc.enable()
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    manager.h2d(
        src_offsets_major_dim=offsets,
        dst_offsets_major_dim=offsets,
        copy_sizes_major_dim=sizes,
    ).Await()
    h2d = time.perf_counter() - t0
    gc.enable()
    gc.collect()
    return d2h, h2d

  for _ in range(warmup):
    once()
  d2h_times, h2d_times = [], []
  for _ in range(iters):
    d, h = once()
    d2h_times.append(d)
    h2d_times.append(h)

  d2h_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in d2h_times]
  h2d_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in h2d_times]
  d2h_med_t, h2d_med_t = float(np.median(d2h_times)), float(
      np.median(h2d_times)
  )
  return {
      "shape": list(shape),
      "num_layers": num_layers,
      "dtype": dtype,
      "total_bytes": total_bytes,
      "d2h_times_sec": d2h_times,
      "h2d_times_sec": h2d_times,
      "d2h_gbps_all": d2h_gbps_all,
      "h2d_gbps_all": h2d_gbps_all,
      "d2h_med_t": d2h_med_t,
      "h2d_med_t": h2d_med_t,
      "d2h_gbps": (total_bytes * 8) / (d2h_med_t * 1e9),
      "h2d_gbps": (total_bytes * 8) / (h2d_med_t * 1e9),
      "d2h_gbps_summary": summarize(d2h_gbps_all),
      "h2d_gbps_summary": summarize(h2d_gbps_all),
  }


# ---------------- PyTorch Backend Implementation ----------------
def _torch_worker_fn(
    rank: int,
    world_size: int,
    master_port: int,
    shape: tuple[int, ...],
    num_layers: int,
    dtype: str,
    shard_axis: int,
    iters: int,
    warmup: int,
    lock_buffers: bool,
    verify: bool,
    result_queue,
):
  """Single-process worker driving local TPU shard transfers in PyTorch."""
  # pylint: disable=g-import-not-at-top
  import torch
  import torch.distributed as dist
  import torch_tpu
  from tpu_sync.api.torch import torch_tpu_common_loader
  from tpu_sync.frameworks.torch import _tpu_raiden_torch as kv_cache_manager_torch
  # pylint: enable=g-import-not-at-top

  if world_size > 1:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
    os.environ["GROUP_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )

  try:
    torch_tpu_common_loader.load_torch_tpu_common()
    device = torch.device("tpu")

    shard_shape = list(shape)
    if world_size > 1:
      shard_shape[shard_axis] = shape[shard_axis] // world_size
    shard_shape = tuple(shard_shape)
    num_blocks = shape[0]

    torch_dt_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "int32": torch.int32,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }
    dt = torch_dt_map.get(dtype, torch.float32)
    itemsize = ITEMSIZE.get(dtype, 4)
    total_bytes = num_layers * int(np.prod(shape)) * itemsize

    torch.manual_seed(42 + rank)
    src_tensors = []
    for _ in range(num_layers):
      if dtype == "int32":
        start = rank * int(np.prod(shard_shape))
        np_data = (
            np.arange(np.prod(shard_shape), dtype=np.int32) + start
        ).reshape(shard_shape)
        t = torch.from_numpy(np_data).to(device)
      else:
        t = torch.randn(shard_shape, dtype=dt, device=device)
      src_tensors.append([t])
    torch.tpu.synchronize()

    manager = kv_cache_manager_torch.KVCacheManager(
        device_tensors=src_tensors,
        local_port=0,
        host_blocks_to_allocate=num_blocks,
        unsafe_skip_buffer_lock=not lock_buffers,
    )

    if verify:
      half = num_blocks // 2
      if half > 0:
        manager.D2h(
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[0],
            copy_sizes_major_dim=[half],
        ).Await()
        manager.H2d(
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[half],
            copy_sizes_major_dim=[half],
        ).Await()
        torch.tpu.synchronize()
        for li, shards in enumerate(src_tensors):
          d = shards[0].cpu().numpy()
          np.testing.assert_array_equal(
              d[half : 2 * half],
              d[0:half],
              err_msg=(
                  f"CORRUPTION: rank {rank} layer {li} d2h/h2d round-trip"
                  " mismatch"
              ),
          )
        if world_size > 1:
          dist.barrier()

    offsets, sizes = [0], [num_blocks]

    def once():
      for shards in src_tensors:
        shards[0].add_(1 if shards[0].dtype == torch.int32 else 0.01)
      torch.tpu.synchronize()
      if world_size > 1:
        dist.barrier()
      gc.disable()
      t0 = time.perf_counter()
      manager.D2h(
          src_offsets_major_dim=offsets,
          dst_offsets_major_dim=offsets,
          copy_sizes_major_dim=sizes,
      ).Await()
      torch.tpu.synchronize()
      t1 = time.perf_counter()
      gc.enable()
      gc.collect()

      if world_size > 1:
        dist.barrier()
      gc.disable()
      t2 = time.perf_counter()
      manager.H2d(
          src_offsets_major_dim=offsets,
          dst_offsets_major_dim=offsets,
          copy_sizes_major_dim=sizes,
      ).Await()
      torch.tpu.synchronize()
      t3 = time.perf_counter()
      gc.enable()
      gc.collect()

      if world_size > 1:
        t0_t = torch.tensor([t0], dtype=torch.float64)
        t1_t = torch.tensor([t1], dtype=torch.float64)
        t2_t = torch.tensor([t2], dtype=torch.float64)
        t3_t = torch.tensor([t3], dtype=torch.float64)
        dist.all_reduce(t0_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(t1_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(t2_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(t3_t, op=dist.ReduceOp.MAX)
        return (t1_t - t0_t).item(), (t3_t - t2_t).item()
      return (t1 - t0), (t3 - t2)

    for _ in range(warmup):
      once()

    d2h_times, h2d_times = [], []
    for _ in range(iters):
      d, h = once()
      d2h_times.append(d)
      h2d_times.append(h)

    if rank == 0:
      d2h_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in d2h_times]
      h2d_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in h2d_times]
      d2h_med_t, h2d_med_t = float(np.median(d2h_times)), float(
          np.median(h2d_times)
      )
      res = {
          "shape": list(shape),
          "num_layers": num_layers,
          "dtype": dtype,
          "total_bytes": total_bytes,
          "d2h_times_sec": d2h_times,
          "h2d_times_sec": h2d_times,
          "d2h_gbps_all": d2h_gbps_all,
          "h2d_gbps_all": h2d_gbps_all,
          "d2h_med_t": d2h_med_t,
          "h2d_med_t": h2d_med_t,
          "d2h_gbps": (total_bytes * 8) / (d2h_med_t * 1e9),
          "h2d_gbps": (total_bytes * 8) / (h2d_med_t * 1e9),
          "d2h_gbps_summary": summarize(d2h_gbps_all),
          "h2d_gbps_summary": summarize(h2d_gbps_all),
      }
      if result_queue is not None:
        result_queue.put(res)
      return res
  finally:
    if world_size > 1 and dist.is_initialized():
      dist.destroy_process_group()


def _measure_torch(
    shape,
    num_layers,
    dtype,
    shard_axis=2,
    iters=20,
    warmup=3,
    lock_buffers=True,
    verify=True,
):
  """Executes the PyTorch transfer benchmark across local TPU chips."""
  import torch.distributed as dist
  import torch.multiprocessing as mp

  # If already inside a distributed worker process, run worker directly:
  if dist.is_available() and dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return _torch_worker_fn(
        rank=rank,
        world_size=world_size,
        master_port=0,
        shape=shape,
        num_layers=num_layers,
        dtype=dtype,
        shard_axis=shard_axis,
        iters=iters,
        warmup=warmup,
        lock_buffers=lock_buffers,
        verify=verify,
        result_queue=None,
    )

  count = get_tpu_device_count()
  world_size = count if count > 0 else 1

  if world_size <= 1:
    return _torch_worker_fn(
        rank=0,
        world_size=1,
        master_port=0,
        shape=shape,
        num_layers=num_layers,
        dtype=dtype,
        shard_axis=shard_axis,
        iters=iters,
        warmup=warmup,
        lock_buffers=lock_buffers,
        verify=verify,
        result_queue=None,
    )

  # Multi-device SPMD execution on host
  prepare_tpu_environment(world_size)
  master_port = pick_unused_ports(1)[0]
  ctx = mp.get_context("spawn")
  result_queue = ctx.SimpleQueue()
  mp.spawn(
      _torch_worker_fn,
      args=(
          world_size,
          master_port,
          shape,
          num_layers,
          dtype,
          shard_axis,
          iters,
          warmup,
          lock_buffers,
          verify,
          result_queue,
      ),
      nprocs=world_size,
      join=True,
  )
  return result_queue.get()


# ---------------- Unified Public Measurement Interface ----------------
def measure(
    shape,
    num_layers,
    dtype,
    shard_axis=2,
    iters=20,
    warmup=3,
    lock_buffers=True,
    verify=True,
    framework="jax",
):
  """Runs the d2h/h2d transfer benchmark for one config and returns a result dict.

  Args:
    shape: Per-layer shape string ('a,b,c') or tuple/list.
    num_layers: Number of cache arrays (layers).
    dtype: Element data type ('float32', 'bfloat16', 'int32', etc.).
    shard_axis: Axis to partition across device chips.
    iters: Number of timed iterations.
    warmup: Number of untimed warmup iterations.
    lock_buffers: Whether to lock host buffers.
    verify: Run one round-trip byte integrity assertion before timing.
    framework: 'jax' or 'torch'.

  Returns:
    Result dictionary containing throughput and latency statistics.
  """
  if isinstance(shape, str):
    shape = shape.split(",")
  shape = tuple(int(x) for x in shape)

  fw = framework.lower()
  if fw == "torch":
    return _measure_torch(
        shape,
        num_layers,
        dtype,
        shard_axis=shard_axis,
        iters=iters,
        warmup=warmup,
        lock_buffers=lock_buffers,
        verify=verify,
    )
  elif fw == "jax":
    return _measure_jax(
        shape,
        num_layers,
        dtype,
        shard_axis=shard_axis,
        iters=iters,
        warmup=warmup,
        lock_buffers=lock_buffers,
        verify=verify,
    )
  else:
    raise ValueError(
        f'Unsupported framework: {framework}. Expected "jax" or "torch".'
    )


# Backward-compatibility aliases
create_sharded_array = _create_sharded_array_jax
verify_roundtrip = _verify_roundtrip_jax

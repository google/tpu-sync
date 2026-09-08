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

import os
import pathlib
import socket
import subprocess
import sys
import time

_LOG_DIR = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("TPU_LOG_DIR", _LOG_DIR)
os.environ.setdefault("GLOG_log_dir", _LOG_DIR)
os.environ.setdefault("GOOGLE_LOG_DIR", _LOG_DIR)
os.environ.setdefault("TMPDIR", _LOG_DIR)

from absl import app
from absl import flags
from absl.testing import absltest
import numpy as np

import torch
import torch_tpu
import torch.distributed as dist

try:
  from google3.pyglib import resources
except ImportError:
  resources = None

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import kv_cache_store

FLAGS = flags.FLAGS
flags.DEFINE_boolean("run_worker", False, "")
flags.DEFINE_string("worker_mode", "save_load", "")
flags.DEFINE_boolean("use_slices", False, "")
flags.DEFINE_boolean("enable_shm", False, "")
flags.DEFINE_integer("rank", 0, "")
flags.DEFINE_integer("world_size", 0, "")
flags.DEFINE_integer("master_port", 0, "")
flags.DEFINE_integer("controller_port", 0, "")
flags.DEFINE_integer("controller_port_b", 0, "")
flags.DEFINE_integer("registry_port", 0, "")

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},
    "0x0063": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},
    "0x006f": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,4,1"},
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},
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
  os.environ["GLOG_alsologtostderr"] = "1"
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  ports = pick_unused_ports(world_size)
  os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
      [f"localhost:{p}" for p in ports]
  )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    os.environ["TORCH_TPU_TOPOLOGY"] = get_tpu_topology(world_size)

def worker_launch_cmd() -> list[str]:
  """The command prefix that re-enters this file as a worker rank.

  Under the build system argv[0] is an executable wrapper, so re-running it is
  all that is needed. Run straight from the source tree and argv[0] is a plain
  .py file that cannot be exec'd, so name the interpreter explicitly.
  """
  if os.access(sys.argv[0], os.X_OK):
    return [sys.argv[0]]
  return [sys.executable, os.path.abspath(__file__)]

_registry_process = None
_registry_port = None


def start_servers():
  global _registry_process
  global _registry_port
  _registry_port = pick_unused_ports(1)[0]
  if resources:
    registry_binary = resources.GetResourceFilename(
        "google3/third_party/tpu_raiden/tpu_sync/kv_cache/global_registry/global_registry_server"
    )
    extra_flags = ["--alsologtostderr"]
  else:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    registry_binary = os.path.abspath(
        os.path.join(
            this_dir,
            "..",
            "..",
            "kv_cache",
            "global_registry",
            "global_registry_server",
        )
    )
    extra_flags = []

  print(f"Starting Registry on port {_registry_port}")
  reg_log = open("/tmp/raiden_registry_mpmd.log", "w")
  _registry_process = subprocess.Popen(
      [registry_binary, f"--port={_registry_port}"] + extra_flags,
      stdout=reg_log,
      stderr=subprocess.STDOUT,
  )
  time.sleep(2)


def stop_servers():
  global _registry_process
  if _registry_process:
    code = _registry_process.poll()
    if code is not None and code != 0:
      print(f"--- Registry exited with {code} ---")
      try:
        with open("/tmp/raiden_registry_mpmd.log", "r") as f:
          print(f.read())
      except OSError as e:
        print(f"Failed to read registry log: {e}")
    _registry_process.terminate()
    _registry_process.wait()
    _registry_process = None


def setUpModule():
  os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"
  os.environ["GLOG_alsologtostderr"] = "1"


def tearDownModule():
  pass


def _worker_save_load_main(argv):
  rank = FLAGS.rank
  world_size = FLAGS.world_size
  master_port = FLAGS.master_port
  controller_port = FLAGS.controller_port
  registry_port = FLAGS.registry_port

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
  os.environ["GLOG_alsologtostderr"] = "1"

  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{master_port}",
      rank=rank,
      world_size=world_size,
  )

  try:
    device = torch.device("tpu")
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
        rank * 1000.0
    )
    tpu_cache = torch.tensor(host_data, device=device)

    # Expected reference after loading saved blocks 0 and 1 into blocks 2 and 3: [a, b, a, b]
    expected_ref = host_data.copy()
    expected_ref[2] = host_data[0]
    expected_ref[3] = host_data[1]

    store = None
    hashes = [b"hash_mpmd_0", b"hash_mpmd_1"]

    if rank == 0:
      block_elements = 128 * 8 * 8 * 128
      shard_size_bytes = block_elements * 4

      rid = kv_cache_store.RaidenId("mpmd_e2e_job", "0", "mpmd_cache", 0)
      # Init store first on rank 0, this binds the local controller server!
      store = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=f"localhost:{registry_port}",
          raiden_id=rid,
          num_shards=world_size,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="127.0.0.1",
          raiden_controller_port=controller_port,
      )

      slices = [
          kv_cache_store.RaidenBlockId(
              rid,
              host_block_id=-1,
              device_block_id=0,
              status=kv_cache_store.BlockStatus.HBM,
          ),
          kv_cache_store.RaidenBlockId(
              rid,
              host_block_id=-1,
              device_block_id=1,
              status=kv_cache_store.BlockStatus.HBM,
          ),
      ]
      assert store.insert(
          hashes, slices, on_host=False
      ), "Failed to insert blocks to store"

    dist.barrier()

    # Store guarantees controller is bound; now initialize managers
    # Store guarantees controller is bound; now initialize managers in rank order
    manager = None
    for r in range(world_size):
      if rank == r:
        manager = kv_cache_manager.KVCacheManager(
            kv_caches=[[tpu_cache]],
            local_control_port=0,
            max_blocks=num_blocks,
            num_slots=2,
            unsafe_skip_buffer_lock=True,
            raiden_worker_port=0,
            raiden_controller_address=f"localhost:{controller_port}",
            worker_id=f"worker_{rank}",
            host_blocks_to_allocate=4,
            node_id=rank,
            enable_shm=FLAGS.enable_shm,
        )
      dist.barrier()

    if rank == 0:
      store.save(hashes)
      done = False
      while not done:
        save_done, save_failed, _, _, _ = store.poll_save_status()
        if save_failed:
          raise RuntimeError(f"Async Save failed: {save_failed}")
        if save_done:
          done = True
        if not done:
          time.sleep(0.01)
      # A successful save consumed the pin; nothing to release.

    if rank == 0:
      print(
          "=== [Rank 0] Loading checkpoint from Host DRAM into TPU HBM blocks"
          " [2, 3] (store.load) ==="
      )
      if FLAGS.use_slices:
        # lookup() pins the returned entries; load(..., slices=...) consumes the pin on success.
        load_slices = [entry for _, entry in store.lookup(hashes)]
        assert len(load_slices) == len(
            hashes
        ), f"expected {len(hashes)} entries, got {load_slices}"
        for entry in load_slices:
          assert (
              entry.status == kv_cache_store.BlockStatus.HOST_AND_HBM
          ), f"entry is {entry.status}, not HOST_AND_HBM"
        assert store.load(
            hashes, [2, 3], slices=load_slices
        ), "load with slices failed"
      else:
        # lookup() pins the returned entries; load() consumes the pin on success.
        assert len(store.lookup(hashes)) == len(hashes)
        assert store.load(hashes, [2, 3]), "load failed"
      done = False
      while not done:
        load_done, load_failed, _ = store.poll_load_status()
        if load_failed:
          raise RuntimeError(f"Async Load failed: {load_failed}")
        if load_done:
          done = True
        if not done:
          time.sleep(0.01)

    dist.barrier()
    try:
      torch.tpu.synchronize()
    except (AttributeError, RuntimeError):
      pass
    print(
        f"=== [Rank {rank}] Verifying TPU memory blocks [2, 3] match saved"
        " blocks [0, 1] ==="
    )
    np.testing.assert_array_equal(tpu_cache.cpu().numpy(), expected_ref)
    print(
        f"=== [Rank {rank}] SUCCESS: E2E MPMD Save/Load [0, 1] -> [2, 3]"
        " roundtrip verified on physical TPU! ==="
    )

  finally:
    dist.barrier()
    dist.destroy_process_group()


def _worker_read_remote_main(argv):
  use_slices = FLAGS.use_slices
  rank = FLAGS.rank
  world_size = FLAGS.world_size
  master_port = FLAGS.master_port
  controller_port_a = FLAGS.controller_port
  controller_port_b = FLAGS.controller_port_b
  registry_port = FLAGS.registry_port

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
  os.environ["GLOG_alsologtostderr"] = "1"

  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{master_port}",
      rank=rank,
      world_size=world_size,
  )

  try:
    device = torch.device("tpu")
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
        rank * 1000.0
    )
    tpu_cache_a = torch.tensor(host_data_a, device=device)
    tpu_cache_b = torch.zeros(shape, dtype=torch.float32, device=device)

    # Expected reference after read_remote pulling blocks 0 and 1 into blocks 0 and 1 of tpu_cache_b
    expected_ref = host_data_a.copy()

    store_a = None
    store_b = None
    hashes = [b"hash_mpmd_rr_0", b"hash_mpmd_rr_1"]

    if rank == 0:
      block_elements = 128 * 8 * 8 * 128
      shard_size_bytes = block_elements * 4

      rid_a = kv_cache_store.RaidenId(
          "mpmd_rr_job_a", "0", "mpmd_rr_cache_a", 0
      )
      store_a = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=f"localhost:{registry_port}",
          raiden_id=rid_a,
          num_shards=world_size,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="127.0.0.1",
          raiden_controller_port=controller_port_a,
      )

      rid_b = kv_cache_store.RaidenId(
          "mpmd_rr_job_b", "0", "mpmd_rr_cache_b", 0
      )
      store_b = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=f"localhost:{registry_port}",
          raiden_id=rid_b,
          num_shards=world_size,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="127.0.0.1",
          raiden_controller_port=controller_port_b,
      )

      slices_a = [
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=0,
              status=kv_cache_store.BlockStatus.HBM,
          ),
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=1,
              status=kv_cache_store.BlockStatus.HBM,
          ),
      ]
      assert store_a.insert(
          hashes, slices_a, on_host=False
      ), "Failed to insert blocks to store_a"

    dist.barrier()

    # Initialize KVCacheManager A and B across all 8 ranks in Unified Host Pool mode (host_blocks_to_allocate=4)
    manager_a = None
    for r in range(world_size):
      if rank == r:
        manager_a = kv_cache_manager.KVCacheManager(
            kv_caches=[[tpu_cache_a]],
            local_control_port=0,
            max_blocks=num_blocks,
            num_slots=2,
            unsafe_skip_buffer_lock=True,
            raiden_worker_port=0,
            raiden_controller_address=f"localhost:{controller_port_a}",
            worker_id=f"worker_a_{rank}",
            host_blocks_to_allocate=4,
            node_id=rank,
        )
      dist.barrier()

    manager_b = None
    for r in range(world_size):
      if rank == r:
        manager_b = kv_cache_manager.KVCacheManager(
            kv_caches=[[tpu_cache_b]],
            local_control_port=0,
            max_blocks=num_blocks,
            num_slots=2,
            unsafe_skip_buffer_lock=True,
            raiden_worker_port=0,
            raiden_controller_address=f"localhost:{controller_port_b}",
            worker_id=f"worker_b_{rank}",
            host_blocks_to_allocate=4,
            node_id=rank,
        )
      dist.barrier()

    if rank == 0:
      store_a.save(hashes)
      done = False
      while not done:
        save_done, save_failed, _, _, _ = store_a.poll_save_status()
        if save_failed:
          raise RuntimeError(f"Job A Async Save failed: {save_failed}")
        if len(save_done) == 2:
          done = True
        if not done:
          time.sleep(0.01)
      # A successful save consumed the pin; nothing to release.

      # Wait for global registry propagation
      deadline = time.time() + 10.0
      lookup_res_b = []
      while time.time() < deadline:
        lookup_res_b = store_b.lookup(hashes, enable_global=True)
        if len(lookup_res_b) == 2:
          break
        time.sleep(0.5)
      assert (
          len(lookup_res_b) == 2
      ), f"Expected 2 remote blocks, got {len(lookup_res_b)}"

      slices_b = [lookup_res_b[0][1], lookup_res_b[1][1]]
      if use_slices:
        print(
            "=== [Rank 0] Launching peer-fetch Load from Job A to Job B with"
            " slices ==="
        )
        assert store_b.load(
            hashes, [0, 1], slices=slices_b
        ), "load failed on store_b"
        done = False
        while not done:
          load_done, load_failed, _ = store_b.poll_load_status()
          if load_failed:
            raise RuntimeError(f"Job B Load failed: {load_failed}")
          if len(load_done) == 2:
            done = True
          if not done:
            time.sleep(0.01)
      else:
        # The read goes straight into TPU blocks [0, 1]; the slices from the
        # lookup are the source coordinate, so there is nothing to insert and
        # no second Load step afterwards.
        print("=== [Rank 0] Launching ReadRemote from Job A to Job B ===")
        assert store_b.read_remote(
            hashes, slices_b, [0, 1]
        ), "read_remote launch failed on store_b"

        done = False
        while not done:
          read_done, read_failed, _ = store_b.poll_remote_read_status()
          if read_failed:
            raise RuntimeError(f"Job B ReadRemote failed: {read_failed}")
          if len(read_done) == 2:
            done = True
          if not done:
            time.sleep(0.01)

        # The read left no local entry, so there is nothing to release here.
        assert not store_b.lookup(hashes), "read_remote must record nothing"

      if use_slices:
        # A load from a peer records nothing locally: no host copy was kept, so
        # the cache is a miss for these hashes.
        assert not store_b.lookup(
            hashes
        ), "peer load must record nothing locally"

    dist.barrier()
    try:
      torch.tpu.synchronize()
    except (AttributeError, RuntimeError):
      pass
    print(
        f"=== [Rank {rank}] Verifying TPU memory blocks [0, 1] match remote"
        " producer blocks ==="
    )
    np.testing.assert_array_equal(
        tpu_cache_b.cpu().numpy()[:2], expected_ref[:2]
    )
    print(
        f"=== [Rank {rank}] SUCCESS: E2E MPMD ReadRemote [0, 1] roundtrip"
        " verified on physical TPU! ==="
    )

  finally:
    dist.barrier()
    dist.destroy_process_group()


def _worker_write_remote_main(argv):
  rank = FLAGS.rank
  world_size = FLAGS.world_size
  master_port = FLAGS.master_port
  controller_port_a = FLAGS.controller_port
  controller_port_b = FLAGS.controller_port_b
  registry_port = FLAGS.registry_port

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
  os.environ["GLOG_alsologtostderr"] = "1"

  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{master_port}",
      rank=rank,
      world_size=world_size,
  )

  try:
    device = torch.device("tpu")
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
        rank * 1000.0
    )
    tpu_cache_a = torch.tensor(host_data_a, device=device)
    tpu_cache_b = torch.zeros(shape, dtype=torch.float32, device=device)

    expected_ref = host_data_a.copy()

    store_a = None
    store_b = None
    hashes = [b"hash_mpmd_wr_0", b"hash_mpmd_wr_1"]

    if rank == 0:
      block_elements = 128 * 8 * 8 * 128
      shard_size_bytes = block_elements * 4

      rid_a = kv_cache_store.RaidenId(
          "mpmd_wr_job_a", "0", "mpmd_wr_cache_a", 0
      )
      store_a = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=f"localhost:{registry_port}",
          raiden_id=rid_a,
          num_shards=world_size,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="127.0.0.1",
          raiden_controller_port=controller_port_a,
      )

      rid_b = kv_cache_store.RaidenId(
          "mpmd_wr_job_b", "0", "mpmd_wr_cache_b", 0
      )
      store_b = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=f"localhost:{registry_port}",
          raiden_id=rid_b,
          num_shards=world_size,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="127.0.0.1",
          raiden_controller_port=controller_port_b,
      )

      slices_a = [
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=0,
              status=kv_cache_store.BlockStatus.HBM,
          ),
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=1,
              status=kv_cache_store.BlockStatus.HBM,
          ),
      ]
      assert store_a.insert(
          hashes, slices_a, on_host=False
      ), "Failed to insert blocks to store_a"

    dist.barrier()

    # Initialize KVCacheManager A and B across all 8 ranks in Unified Host Pool mode
    manager_a = None
    for r in range(world_size):
      if rank == r:
        manager_a = kv_cache_manager.KVCacheManager(
            kv_caches=[[tpu_cache_a]],
            local_control_port=0,
            max_blocks=num_blocks,
            num_slots=2,
            unsafe_skip_buffer_lock=True,
            raiden_worker_port=0,
            raiden_controller_address=f"localhost:{controller_port_a}",
            worker_id=f"worker_a_{rank}",
            host_blocks_to_allocate=4,
            node_id=rank,
        )
      dist.barrier()

    manager_b = None
    for r in range(world_size):
      if rank == r:
        manager_b = kv_cache_manager.KVCacheManager(
            kv_caches=[[tpu_cache_b]],
            local_control_port=0,
            max_blocks=num_blocks,
            num_slots=2,
            unsafe_skip_buffer_lock=True,
            raiden_worker_port=0,
            raiden_controller_address=f"localhost:{controller_port_b}",
            worker_id=f"worker_b_{rank}",
            host_blocks_to_allocate=4,
            node_id=rank,
        )
      dist.barrier()

    if rank == 0:
      store_a.save(hashes)
      deadline = time.time() + 120
      done = False
      while time.time() < deadline:
        save_done, save_failed, _, _, _ = store_a.poll_save_status()
        if save_failed:
          raise RuntimeError(f"Job A Async Save failed: {save_failed}")
        if len(save_done) == 2:
          done = True
          break
        time.sleep(0.01)
      assert done, "Job A Async Save timed out"

      print("=== [Rank 0] Launching WriteRemote from Job A to Job B ===")
      # The local save above consumed the pin, so the offer needs its own.
      # lookup() answers "host-resident here" AND grants the pin that
      # save(dst) spends -- the documented remote-save flow.
      assert len(store_a.lookup(hashes)) == len(hashes), "hashes not resident"
      assert store_a.save(hashes, rid_b), "remote save launch failed on store_a"

      deadline = time.time() + 120
      done = False
      while time.time() < deadline:
        wr_done, wr_failed, wr_pending, _, _ = store_a.poll_save_status()
        if wr_failed:
          raise RuntimeError(f"Job A WriteRemote failed: {wr_failed}")
        if len(wr_done) == 2:
          done = True
          break
        time.sleep(0.01)
      assert done, "Job A WriteRemote timed out"
      # A successful save consumed the pin; nothing to release.

      # Destination holds blocks locally on host DRAM. lookup() pins the landed entries.
      lookup_res_b = store_b.lookup(hashes, enable_global=False)
      assert (
          len(lookup_res_b) == 2
      ), f"Expected 2 blocks on store_b, got {len(lookup_res_b)}"

      # Load blocks [0, 1] from Job B's host pool into TPU HBM (consumes the pin)
      assert store_b.load(hashes, [0, 1]), "load failed on store_b"
      deadline = time.time() + 120
      done = False
      while time.time() < deadline:
        load_done, load_failed, _ = store_b.poll_load_status()
        if load_failed:
          raise RuntimeError(f"Job B Load failed: {load_failed}")
        if len(load_done) == 2:
          done = True
          break
        time.sleep(0.01)
      assert done, "Job B Load timed out"

    dist.barrier()
    try:
      torch.tpu.synchronize()
    except (AttributeError, RuntimeError):
      pass
    print(
        f"=== [Rank {rank}] Verifying TPU memory blocks [0, 1] match offered"
        " producer blocks ==="
    )
    np.testing.assert_array_equal(
        tpu_cache_b.cpu().numpy()[:2], expected_ref[:2]
    )
    print(
        f"=== [Rank {rank}] SUCCESS: E2E MPMD WriteRemote [0, 1] roundtrip"
        " verified on physical TPU! ==="
    )

  finally:
    dist.barrier()
    dist.destroy_process_group()


class KVCacheStoreMpmdE2ETest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    start_servers()

  @classmethod
  def tearDownClass(cls):
    stop_servers()
    super().tearDownClass()

  def test_mpmd_8rank_e2e_save_and_load(self):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    controller_port = pick_unused_ports(1)[0]

    procs = []
    for rank in range(world_size):
      env = os.environ.copy()
      cmd = worker_launch_cmd() + [
          "--run_worker",
          f"--rank={rank}",
          f"--world_size={world_size}",
          f"--master_port={master_port}",
          f"--controller_port={controller_port}",
          f"--registry_port={_registry_port}",
      ]
      procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for p in procs:
      p.wait()
      if p.returncode != 0:
        failed = True

    if failed:
      self.fail("One or more workers failed!")

  # The same 8-rank save/load/compare run, with rank 0's load driven through
  # the slices form. Worth running under MPMD rather than only single-process:
  # every rank has its own store and its own shard of each block, so a slices
  # path that resolved entries against the wrong rank's index would land the
  # wrong shards here and nowhere else.
  def test_mpmd_8rank_e2e_save_and_load_with_slices(self):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    controller_port = pick_unused_ports(1)[0]

    procs = []
    for rank in range(world_size):
      env = os.environ.copy()
      cmd = worker_launch_cmd() + [
          "--run_worker",
          "--use_slices",
          f"--rank={rank}",
          f"--world_size={world_size}",
          f"--master_port={master_port}",
          f"--controller_port={controller_port}",
          f"--registry_port={_registry_port}",
      ]
      procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for p in procs:
      p.wait()
      if p.returncode != 0:
        failed = True

    if failed:
      self.fail("One or more workers failed!")

  # The same 8-rank save/load/compare run with the KV pools (and metadata
  # tables) placed in shared memory. Every rank runs in its own process on
  # the same host, so this exercises per-rank segment names, one region per
  # allocation within a segment, and true multi-process DMA mapping.
  def test_mpmd_8rank_e2e_save_and_load_in_shared_memory(self):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    controller_port = pick_unused_ports(1)[0]
    shm_key = f"raiden_mpmd_e2e_{os.getpid()}"

    procs = []
    try:
      for rank in range(world_size):
        env = os.environ.copy()
        env["RAIDEN_SHM_KEY"] = shm_key
        env["RAIDEN_SHM_MODEL_UID"] = "mpmd_e2e_model"
        cmd = worker_launch_cmd() + [
            "--run_worker",
            "--enable_shm",
            f"--rank={rank}",
            f"--world_size={world_size}",
            f"--master_port={master_port}",
            f"--controller_port={controller_port}",
            f"--registry_port={_registry_port}",
        ]
        procs.append(subprocess.Popen(cmd, env=env))

      failed = False
      for p in procs:
        p.wait()
        if p.returncode != 0:
          failed = True

      if failed:
        self.fail("One or more workers failed!")
    finally:
      # Nothing in the serving stack ever unlinks the segments (they must
      # outlive any crash); decommissioning is the operator's explicit
      # shm_unlink — here, the test.
      for name in os.listdir("/dev/shm"):
        if name.startswith(shm_key):
          try:
            os.unlink(os.path.join("/dev/shm", name))
          except OSError:
            pass

  def test_mpmd_8rank_e2e_write_remote(self):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    controller_port_a = pick_unused_ports(1)[0]
    controller_port_b = pick_unused_ports(1)[0]

    procs = []
    for rank in range(world_size):
      env = os.environ.copy()
      cmd = worker_launch_cmd() + [
          "--run_worker",
          "--worker_mode=write_remote",
          f"--rank={rank}",
          f"--world_size={world_size}",
          f"--master_port={master_port}",
          f"--controller_port={controller_port_a}",
          f"--controller_port_b={controller_port_b}",
          f"--registry_port={_registry_port}",
      ]
      procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for p in procs:
      p.wait()
      if p.returncode != 0:
        failed = True

    if failed:
      self.fail("One or more workers failed in write_remote MPMD test!")

  def _drive_read_remote(self, use_slices: bool):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    controller_port_a = pick_unused_ports(1)[0]
    controller_port_b = pick_unused_ports(1)[0]

    procs = []
    for rank in range(world_size):
      env = os.environ.copy()
      cmd = worker_launch_cmd() + [
          "--run_worker",
          "--worker_mode=read_remote",
          f"--rank={rank}",
          f"--world_size={world_size}",
          f"--master_port={master_port}",
          f"--controller_port={controller_port_a}",
          f"--controller_port_b={controller_port_b}",
          f"--registry_port={_registry_port}",
      ]
      if use_slices:
        cmd.append("--use_slices")
      procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for p in procs:
      p.wait()
      if p.returncode != 0:
        failed = True

    if failed:
      self.fail("One or more workers failed in read_remote MPMD test!")

  def test_mpmd_8rank_e2e_read_remote(self):
    self._drive_read_remote(use_slices=False)

  def test_mpmd_8rank_e2e_read_remote_with_slices(self):
    # The peer load through load(slices=REMOTE): the worker body always
    # asserted the records-nothing contract, but no driver dispatched it.
    self._drive_read_remote(use_slices=True)

  # The expected_worker_count barrier tests live in kv_cache_store_test.py;
  # they spawn no MPMD workers, so duplicating them here added nothing.


def main(argv):
  if FLAGS.run_worker:
    if FLAGS.worker_mode == "read_remote":
      _worker_read_remote_main(argv)
    elif FLAGS.worker_mode == "write_remote":
      _worker_write_remote_main(argv)
    else:
      _worker_save_load_main(argv)
  else:
    absltest.main()


if __name__ == "__main__":
  app.run(main)

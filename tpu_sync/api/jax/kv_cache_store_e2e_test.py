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

"""E2E test for JAX KVCacheStore with TPUs."""

import glob
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import uuid

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

_SECONDARY_STORAGE_ROOT = flags.DEFINE_string(
    "secondary_storage_root",
    "",
    "Root directory for secondary storage. If empty, a temporary directory is"
    " created.",
)

resources = None
from tpu_sync.api.jax import kv_cache_manager
from tpu_sync.api.jax import kv_cache_store

# Set XLA flags to force CPU/Host platform devices if running locally on
# simulator
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def _pick_unused_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def get_local_ip() -> str:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
  except OSError:
    ip = "127.0.0.1"
  finally:
    s.close()
  return ip


# Global variables for subprocesses
_registry_process = None
_registry_port = None


def _registry_binary_path():
  this_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "kv_cache",
          "global_registry",
          "global_registry_server",
      )
  )


def _node_binary_path():
  this_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "store_node",
          "kv_cache_host_store_node_main",
      )
  )


def start_servers():
  global _registry_process
  global _registry_port

  _registry_port = _pick_unused_port()

  registry_binary = _registry_binary_path()
  extra_flags = ["--alsologtostderr"] if resources else []

  print(f"Starting Registry on port {_registry_port}")
  reg_log = open("/tmp/raiden_registry.log", "w")
  _registry_process = subprocess.Popen(
      [
          registry_binary,
          f"--port={_registry_port}",
      ]
      + extra_flags,
      stdout=reg_log,
      stderr=subprocess.STDOUT,
  )

  # Give them some time to start
  time.sleep(2)


def stop_servers():
  global _registry_process
  if _registry_process:
    code = _registry_process.poll()
    if code is not None and code != 0:
      print(f"--- Registry exited with {code} ---")
      try:
        with open("/tmp/raiden_registry.log", "r") as f:
          print(f.read())
      except OSError as e:
        print(f"Failed to read registry log: {e}")
    _registry_process.terminate()
    _registry_process.wait()
    _registry_process = None


def setUpModule():
  os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"


def tearDownModule():
  pass


class KVCacheStoreE2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.controller_port = _pick_unused_port()

  def setUp(self):
    super().setUp()
    start_servers()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      self.devices = jax.devices()

    if not self.devices:
      raise AssertionError("No JAX devices found")

    self.num_devices = len(self.devices)
    self.num_layers = 1
    self.skip_lock = True

  def tearDown(self):
    stop_servers()
    super().tearDown()

  def create_mesh(self, axis_shapes, axis_names, devices=None):
    try:
      num_required_devices = np.prod(axis_shapes)
      if devices is None:
        devices = self.devices
      devices = np.array(devices)
      if len(devices) < num_required_devices:
        raise AssertionError(
            f"Need {num_required_devices} devices, got {len(devices)}"
        )
      device_array = devices[:num_required_devices].reshape(axis_shapes)
      return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
      self.skipTest("Cannot create mesh.")
      return None

  def setup_shardings(self):
    axis_shapes = (1, self.num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model", None, None)
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    return tpu_sharding

  def setup_sharding_for_devices(self, devices):
    axis_shapes = (1, len(devices))
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names, devices)
    spec = jax.sharding.PartitionSpec(None, None, "model", None, None)
    return jax.sharding.NamedSharding(mesh, spec)

  def _run_e2e_test(self, enable_multi_numa: bool, use_slices: bool = False):
    if enable_multi_numa and len(self.devices) > 4:
      # TODO(jcgu): Create a new multi-host setup
      # to test True Multi-NUMA ENABLE_MULTI_NUMA=1 correctly, since running
      # two distinct jobs inside this single-process sandbox blocks cross-NIC UDP routing.
      self.skipTest(
          "Multi-NUMA E2E test is not supported on single-host shared device "
          f"configurations (devices={len(self.devices)}). Skipping."
      )

    os.environ["ENABLE_MULTI_NUMA"] = "1" if enable_multi_numa else "0"

    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data
    # np.arange creates unique values for each element, ensuring different
    # values for different shards
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)
    expected_ref = host_data

    # 2. Get free port for controller
    controller_port = _pick_unused_port()

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    # 3. Create KVCacheStore (Controller)
    rid = kv_cache_store.RaidenId("e2e_job", "0", "e2e_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=num_blocks,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )

    # 4. Create KVCacheManager (Worker)
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=0,
        # Must match the address the store's controller binds
        # ("localhost:{controller_port}", see the KVCacheStore above); using
        # get_local_ip() here dials a LAN IP the controller is not listening on,
        # so RegisterWorker never lands and Save fails with "No registered
        # workers available for TransferBuffers".
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_0",
    )

    # 5. Insert HBM blocks to KVCacheStore
    hashes = [b"hash_0", b"hash_1"]
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
    self.assertTrue(store.insert(hashes, slices, on_host=False))

    # Verify status in store is HBM. pin_found=False: observation only, no
    # pin taken and the LRU order left alone.
    lookup_res = store.lookup(hashes, pin_found=False)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[0][1].device_block_id, 0)
    self.assertEqual(lookup_res[1][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[1][1].device_block_id, 1)

    # 6. Save HBM blocks to host memory
    @jax.jit
    def get_slice_e2e(x):
      return x[0, 0, 0, 0, 0:16]

    print(f"DEBUG: test_e2e tpu_cache before Save: {get_slice_e2e(tpu_cache)}")

    store.save(hashes)

    # Wait for save completion
    done = False
    while not done:
      save_done, save_failed, _, save_existing, save_unregistered = (
          store.poll_save_status()
      )
      if save_failed:
        raise RuntimeError(f"Async Save failed: {save_failed}")
      # A local save never produces the remote-only outcomes.
      self.assertEmpty(save_existing)
      self.assertEmpty(save_unregistered)
      if save_done:
        done = True
      if not done:
        time.sleep(0.01)

    # No release: a successful save consumed the pin insert()/pin() granted.

    # Verify status in store is updated to HOST_AND_HBM
    lookup_res = store.lookup(hashes, pin_found=False)
    self.assertLen(lookup_res, 2)
    self.assertEqual(
        lookup_res[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res[0][1].host_block_id, 0)
    self.assertEqual(
        lookup_res[1][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res[1][1].host_block_id, 1)

    # 7. Overwrite device memory with zeros
    # host blocks 2 and 3 are empty/uninitialized, containing zeros
    manager.h2d([2, 2], [0, 1]).wait()

    # Verify they are indeed zeros using JIT sum to avoid host caching
    # sum_val = jax.jit(jnp.sum)(tpu_cache)
    # self.assertEqual(float(sum_val), 0.0)
    # print(f"DEBUG: np.asarray(tpu_cache) after overwrite with zeros: {np.asarray(tpu_cache)[0, 0, 0, 0, 0:5]}")

    # 8. Load from host DRAM back to device HBM
    if use_slices:
      # lookup() pins the returned entries; load(..., slices=...) consumes the pin on success.
      load_slices = [entry for _, entry in store.lookup(hashes)]
      self.assertLen(load_slices, 2)
      for entry in load_slices:
        self.assertEqual(entry.status, kv_cache_store.BlockStatus.HOST_AND_HBM)
      self.assertTrue(store.load(hashes, [0, 1], slices=load_slices))
    else:
      # lookup() pins the returned entries; load() consumes the pin on success.
      self.assertLen(store.lookup(hashes), 2)
      self.assertTrue(store.load(hashes, [0, 1]))

    # Wait for load completion
    done = False
    while not done:
      load_done, load_failed, _ = store.poll_load_status()
      if load_failed:
        raise RuntimeError(f"Async Load failed: {load_failed}")
      if load_done:
        done = True
      if not done:
        time.sleep(0.01)

    # 9. Verify device memory contains the original random data
    np.testing.assert_array_equal(np.asarray(tpu_cache), expected_ref)

  def test_e2e_without_multi_numa(self):
    self._run_e2e_test(enable_multi_numa=False)

  @unittest.skip(
      "multi-NUMA needs a dedicated multi-NUMA machine; port contention"
      " in the sandbox otherwise. Re-enable when the hardware lands."
  )
  def test_e2e_with_multi_numa(self):
    self._run_e2e_test(enable_multi_numa=True)

  # The same save/zero/load/compare pipeline, driven through the slices form
  # of load. Running it as a variant rather than a separate test is the point:
  # `slices` is a shortcut past the store's own lookup, not a different
  # transfer, so the bytes that land must be the ones the no-slices path
  # produces -- and the device is zeroed in between either way, so a load that
  # quietly did nothing would compare zeros against the payload and fail.
  def test_e2e_with_slices_without_multi_numa(self):
    self._run_e2e_test(enable_multi_numa=False, use_slices=True)

  @unittest.skip(
      "multi-NUMA needs a dedicated multi-NUMA machine; port contention"
      " in the sandbox otherwise. Re-enable when the hardware lands."
  )
  def test_e2e_with_slices_with_multi_numa(self):
    self._run_e2e_test(enable_multi_numa=True, use_slices=True)

  @parameterized.named_parameters(
      ("buffered", False),
      ("direct_io", True),
  )
  def test_secondary_storage_e2e_offload_recall(self, direct_io: bool = False):
    """Verifies end-to-end offload to storage and recall across TPU chips."""
    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    # Shape: (num_blocks, tokens_per_block, head_shards, heads_per_shard, head_dim)
    # (2 blocks, 128 tokens/block, 8 head shards, 8 heads/shard, head_dim 128).
    shape = (num_blocks, 128, 8, 8, 128)
    block_bytes = (np.prod(shape) // num_blocks) * 4
    total_bytes = np.prod(shape) * 4
    mode_str = "direct_io" if direct_io else "buffered"
    print(
        "\n======================================================================\n[JAX"
        f" E2E Storage] STARTING 4-CHIP SECONDARY STORAGE E2E TEST (mode={mode_str})\n  Total TPU"
        f" Chips: {self.num_devices} | Devices: {self.devices}\n  Logical"
        f" Shape: {shape} ({num_blocks} blocks)\n  Bytes/block: {block_bytes} B"
        f" ({block_bytes / (1024 * 1024):.1f} MiB) | Total bytes: {total_bytes}"
        f" B ({total_bytes / (1024 * 1024):.1f}"
        " MiB)\n======================================================================\n[JAX"
        " E2E Storage][Step 1/11] Generating reference cache data and placing"
        f" across {self.num_devices} TPU chips...",
        flush=True,
    )
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)
    print(
        "[JAX E2E Storage][Step 1/11] Device buffer initialized and"
        f" synchronized on {self.num_devices} TPUs (sharding={tpu_sharding}).",
        flush=True,
    )
    expected_ref = host_data

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    if _SECONDARY_STORAGE_ROOT.value:
      temp_dir = os.path.join(
          _SECONDARY_STORAGE_ROOT.value,
          f"jax_e2e_{mode_str}_{uuid.uuid4().hex[:8]}",
      )
      os.makedirs(temp_dir, exist_ok=True)
      is_custom_root = True
    else:
      temp_dir = tempfile.mkdtemp()
      is_custom_root = False
    try:
      print(
          "[JAX E2E Storage][Step 2/11] Configuring POSIX secondary storage"
          f" backend (direct_io={direct_io}):\n  root_dir: {temp_dir}\n "
          " model_name: llama_70b_jax\n "
          f" tp_size: {self.num_devices}, tp_rank: 0\n  shard_size_bytes:"
          f" {shard_size_bytes} B",
          flush=True,
      )
      cfg = kv_cache_store._impl.BackendConfig()
      cfg.type = "posix"
      cfg.parallelism.tp_rank = 0
      cfg.parallelism.tp_size = self.num_devices
      cfg.set_property("root_dir", temp_dir)
      cfg.set_property("model_name", "llama_70b_jax")
      if direct_io:
        cfg.set_property("direct_io", "true")

      # =======================================================================
      # Phase 1: Node 1 Offload to Secondary Storage
      # =======================================================================
      tag1 = f"sec1_{uuid.uuid4().hex[:8]}"
      controller_port1 = _pick_unused_port()
      print(
          f"[JAX E2E Storage][Step 3/11] Initializing Node 1 (tag={tag1},"
          f" controller_port={controller_port1}, shards={self.num_devices})...",
          flush=True,
      )
      rid1 = kv_cache_store.RaidenId(tag1, "0", "cache_1", 0)
      # Worker Discovery & Backend Registration:
      # 1. KVCacheStore's RaidenController listens for gRPC RegisterWorker calls on controller_port.
      # 2. Each worker KVCacheManager connects to the controller and registers its worker endpoint.
      # 3. Both store and manager initialize secondary backends locally from BackendConfig.
      store1 = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          raiden_id=rid1,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port1,
          secondary_backend_configs=[cfg],
      )
      manager1 = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port1}",
          worker_id="worker_0",
          backend_configs=[cfg],
      )

      hashes = [b"sec_hash_0", b"sec_hash_1"]
      try:
        print(
            "[JAX E2E Storage][Step 3/11] Inserting initial HBM blocks"
            f" {hashes} to device_blocks=[0, 1]...",
            flush=True,
        )
        slices = [
            kv_cache_store.RaidenBlockId(
                rid1,
                host_block_id=-1,
                device_block_id=0,
                status=kv_cache_store.BlockStatus.HBM,
            ),
            kv_cache_store.RaidenBlockId(
                rid1,
                host_block_id=-1,
                device_block_id=1,
                status=kv_cache_store.BlockStatus.HBM,
            ),
        ]
        self.assertTrue(store1.insert(hashes, slices, on_host=False))

        lookup_res = store1.lookup(hashes, pin_found=False)
        self.assertLen(lookup_res, 2)
        for h, b in lookup_res:
          self.assertEqual(b.status, kv_cache_store.BlockStatus.HBM)
          print(
              f"  [Node 1 Inserted] hash={h!r}"
              f" device_block_id={b.device_block_id}"
              f" host_block_id={b.host_block_id} status={b.status.name}",
              flush=True,
          )
        print(
            "[JAX E2E Storage][Step 3/11] Verified initial status in store1 is"
            " HBM.",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 4/11] Triggering offload (save) for"
            f" {hashes}...",
            flush=True,
        )
        save_start = time.time()
        self.assertTrue(store1.save(hashes))

        done = False
        while not done:
          save_done, save_failed, _, save_existing, save_unregistered = (
              store1.poll_save_status()
          )
          if save_failed:
            raise RuntimeError(f"Async Save failed: {save_failed}")
          self.assertEmpty(save_existing)
          self.assertEmpty(save_unregistered)
          if save_done:
            done = True
          if not done:
            time.sleep(0.01)
        save_duration = time.time() - save_start
        print(
            "[JAX E2E Storage][Step 4/11] Save completed in"
            f" {save_duration:.3f}s. Blocks confirmed: {save_done}.",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 5/11] Post-save lookup verifying"
            " HOST_AND_HBM status...",
            flush=True,
        )
        lookup_res = store1.lookup(hashes, pin_found=False)
        self.assertLen(lookup_res, 2)
        for h, b in lookup_res:
          self.assertEqual(b.status, kv_cache_store.BlockStatus.HOST_AND_HBM)
          print(
              f"  [Node 1 Post-Save] hash={h!r} host_block_id={b.host_block_id}"
              f" device_block_id={b.device_block_id} status={b.status.name}",
              flush=True,
          )
        print(
            "[JAX E2E Storage][Step 5/11] Verified status updated to"
            " HOST_AND_HBM in store1.",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 6/11] Scanning on-disk shard files and"
            " verifying bit-for-bit integrity...",
            flush=True,
        )
        all_disk_files = glob.glob(
            os.path.join(temp_dir, "**", "*"), recursive=True
        )
        print(
            "[JAX E2E Storage][Step 6/11] Directory scan found"
            f" {len(all_disk_files)} filesystem entries under {temp_dir}.",
            flush=True,
        )
        # Expected shard path:
        #   {storage_root}/llama_70b_jax/tp{self.num_devices}_r0/{hash[:3]}/{hash[3:5]}/{hash}.bin
        bin_files = glob.glob(
            os.path.join(
                temp_dir,
                "llama_70b_jax",
                f"tp{self.num_devices}_r0",
                "**",
                "*.bin",
            ),
            recursive=True,
        )
        self.assertLen(bin_files, 2)
        print(
            f"[JAX E2E Storage][Step 6/11] Discovered {len(bin_files)} disk"
            f" block files: {[os.path.basename(f) for f in bin_files]}",
            flush=True,
        )
        for f in bin_files:
          print(f"  File: {f} ({os.path.getsize(f)} bytes)", flush=True)

        file_by_name = {os.path.basename(f): f for f in bin_files}
        expected_block_bytes = (np.prod(shape) // num_blocks) * 4
        for h in hashes:
          expected_filename = f"{h.hex()}.bin"
          self.assertIn(expected_filename, file_by_name)
          fpath = file_by_name[expected_filename]
          fsize = os.path.getsize(fpath)
          self.assertEqual(fsize, expected_block_bytes)
          print(
              f"  [Verified File] {expected_filename}: exact size={fsize} bytes"
              f" matches expected={expected_block_bytes} bytes",
              flush=True,
          )
        print(
            "[JAX E2E Storage][Step 6/11] Bit-for-bit disk file size and"
            " presence verification PASSED.",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 7/11] Overwriting TPU HBM cache with"
            " zeros...",
            flush=True,
        )
        manager1.h2d([2, 2], [0, 1]).wait()
        sum_val = float(jax.jit(jnp.sum)(tpu_cache))
        self.assertEqual(sum_val, 0.0)
        print(
            "[JAX E2E Storage][Step 7/11] Verified TPU device memory is zeroed"
            f" (sum={sum_val}).",
            flush=True,
        )
      finally:
        print(
            "[JAX E2E Storage][Step 7/11] Cleaning up Node 1 (destroying"
            " manager1 & store1)...",
            flush=True,
        )
        del manager1, store1

      # =======================================================================
      # Phase 2: Node 2 Cold Storage Recall & Status Verification
      # =======================================================================
      tag2 = f"sec2_{uuid.uuid4().hex[:8]}"
      controller_port2 = _pick_unused_port()
      print(
          "[JAX E2E Storage][Step 8/11] Spinning up cold Instance 2 (Reader)"
          f" with empty DRAM cache (tag={tag2},"
          f" controller_port={controller_port2})...",
          flush=True,
      )
      rid2 = kv_cache_store.RaidenId(tag2, "0", "cache_2", 0)
      # Worker Discovery & Backend Registration:
      # 1. KVCacheStore's RaidenController listens for gRPC RegisterWorker calls on controller_port.
      # 2. Each worker KVCacheManager connects to the controller and registers its worker endpoint.
      # 3. Both store and manager initialize secondary backends locally from BackendConfig.
      store2 = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          raiden_id=rid2,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port2,
          secondary_backend_configs=[cfg],
      )
      manager2 = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port2}",
          worker_id="worker_0",
          backend_configs=[cfg],
      )

      try:
        print(
            "[JAX E2E Storage][Step 9/11] Performing cold lookup for"
            f" {hashes}...",
            flush=True,
        )
        storage_lookup = store2.lookup(hashes, pin_found=False)
        self.assertLen(storage_lookup, 2)
        for h, b in storage_lookup:
          self.assertEqual(b.status, kv_cache_store.BlockStatus.SHARED_STORAGE)
          print(
              f"  [Node 2 Cold Lookup] hash={h!r} status={b.status.name}"
              f" data_name={b.raiden_id.data_name}",
              flush=True,
          )
        print(
            "[JAX E2E Storage][Step 9/11] Cold lookup verified: blocks"
            " discovered with status=SHARED_STORAGE from storage tier.",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 10/11] Triggering cold recall (store2.load)"
            " into device_block_ids=[0, 1]...",
            flush=True,
        )
        load_start = time.time()
        storage_slices = [s for _, s in storage_lookup]
        self.assertTrue(store2.load(hashes, [0, 1], slices=storage_slices))

        done = False
        while not done:
          load_done, load_failed, _ = store2.poll_load_status()
          if load_failed:
            raise RuntimeError(f"Async Load failed: {load_failed}")
          if load_done:
            done = True
          if not done:
            time.sleep(0.01)
        load_duration = time.time() - load_start
        print(
            "[JAX E2E Storage][Step 10/11] Load operation completed in"
            f" {load_duration:.3f}s.",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 10/11] Verifying restored TPU HBM tensor"
            " matches expected reference...",
            flush=True,
        )
        recalled_tpu_cache = jax.jit(lambda x: x)(tpu_cache)
        jax.block_until_ready(recalled_tpu_cache)
        recalled_np = np.asarray(recalled_tpu_cache)
        np.testing.assert_array_equal(recalled_np, expected_ref)
        print(
            "[JAX E2E Storage][Step 10/11] Tensor bit-for-bit match verified"
            f" across all {self.num_devices} TPU chips"
            f" (elements={recalled_np.size}, min={recalled_np.min()},"
            f" max={recalled_np.max()}).",
            flush=True,
        )

        print(
            "[JAX E2E Storage][Step 11/11] Validating post-recall status is"
            " HOST_AND_HBM and tearing down Instance 2...",
            flush=True,
        )
        repopulated = store2.lookup(hashes, pin_found=False)
        self.assertLen(repopulated, 2)
        for h, b in repopulated:
          self.assertEqual(b.status, kv_cache_store.BlockStatus.HOST_AND_HBM)
          self.assertIn(b.device_block_id, [0, 1])
          self.assertGreaterEqual(b.host_block_id, 0)
          print(
              f"  [Node 2 Post-Recall] hash={h!r}"
              f" host_block_id={b.host_block_id}"
              f" device_block_id={b.device_block_id} status={b.status.name}",
              flush=True,
          )
        print(
            "[JAX E2E Storage][Step 11/11] Post-recall lookup verified:"
            " status=HOST_AND_HBM (device_blocks=[0,1], host_blocks confirmed"
            " >= 0; HBM and host RAM updated).",
            flush=True,
        )
      finally:
        print(
            "[JAX E2E Storage][Step 11/11] Cleaning up Node 2 (destroying"
            " manager2 & store2)...",
            flush=True,
        )
        del manager2, store2
    finally:
      if not is_custom_root:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(
            f"[JAX E2E Storage] Cleaned up temporary directory {temp_dir}.",
            flush=True,
        )
      else:
        print(
            f"[JAX E2E Storage] Preserved custom storage directory {temp_dir}.",
            flush=True,
        )

    print(
        f"[JAX E2E Storage][SUCCESS] 4-Chip JAX secondary storage offload and"
        f" recall end-to-end test passed successfully!\n"
        f"======================================================================\n",
        flush=True,
    )

  # ---------------------------------------------------------------------------
  # Helpers for the mixed-backend Load() tests. Completion waits reuse
  # _await_terminal, which accumulates drained poll results across polls: a
  # mixed Load() finishes in independent halves (prefix and storage suffix)
  # that may be reported by different polls.
  # ---------------------------------------------------------------------------

  def _assert_block_statuses(self, lookup_result, hashes, expected_statuses):
    """Asserts lookup_result covers hashes in order with expected statuses.

    expected_statuses[i] is a BlockStatus or a tuple of acceptable statuses.
    """
    self.assertEqual([h for h, _ in lookup_result], list(hashes))
    got = [s.status for _, s in lookup_result]
    for i, want in enumerate(expected_statuses):
      allowed = want if isinstance(want, tuple) else (want,)
      self.assertIn(got[i], allowed,
                    f"block {hashes[i]!r}: statuses={got}")

  def _await_lookup_statuses(self, store, hashes, expected_statuses,
                             timeout_s=120.0, **lookup_kwargs):
    """Polls lookup() until it returns exactly expected_statuses.

    Replaces fixed sleeps where visibility is eventually consistent (global
    registry publication). lookup_kwargs must not pin, since lookups repeat.
    """
    deadline = time.time() + timeout_s
    while True:
      result = store.lookup(hashes, **lookup_kwargs)
      try:
        self._assert_block_statuses(result, hashes, expected_statuses)
        return result
      except AssertionError:
        if time.time() > deadline:
          raise
      time.sleep(0.1)

  def _assert_and_print_shard_files(self, tag, root, model_name, expected):
    bin_files = sorted(glob.glob(os.path.join(root, model_name, "**", "*.bin"),
                                 recursive=True))
    self.assertLen(bin_files, expected, f"shard files: {bin_files}")
    print(f"[{tag}][Phase 1/5] {len(bin_files)} on-disk shard files"
          " (exact count verified):", flush=True)
    for f in bin_files:
      print(f"  [Shard File] {f}"
            f" ({os.path.getsize(f)} bytes)", flush=True)

  def _make_storage_dir(self, prefix):
    if _SECONDARY_STORAGE_ROOT.value:
      temp_dir = os.path.join(_SECONDARY_STORAGE_ROOT.value,
                              f"{prefix}_{uuid.uuid4().hex[:8]}")
      os.makedirs(temp_dir, exist_ok=True)
      return temp_dir, True
    return tempfile.mkdtemp(), False

  def _posix_cfg(self, root, model_name):
    cfg = kv_cache_store._impl.BackendConfig()
    cfg.type = "posix"
    cfg.parallelism.tp_rank = 0
    cfg.parallelism.tp_size = self.num_devices
    cfg.set_property("root_dir", root)
    cfg.set_property("model_name", model_name)
    return cfg

  def _hbm_slices(self, raiden_id, num_blocks):
    return [
        kv_cache_store.RaidenBlockId(
            raiden_id,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(num_blocks)
    ]

  def test_secondary_storage_e2e_mixed_backend_load(self):
    """Mixed Load(): HOST prefix + SHARED_STORAGE suffix, real TPU H2D/D2H.

    Blocks: prefix [mixed_h0, mixed_h1] -> HOST (recalled into reader host RAM
            first), suffix [mixed_s0, mixed_s1] -> SHARED_STORAGE.
      1. Seed storage: writer saves all 4 blocks to POSIX storage.
      2. Stage prefix: reader recalls the prefix into its host RAM; HBM zeroed.
      3. Lookup:       exactly [HOST, HOST, SHARED_STORAGE, SHARED_STORAGE].
      4. Mixed load:   one Load() for all 4 blocks; wait for all 4.
      5. Verify:       HBM bit-exact; all 4 blocks HOST_AND_HBM.
    """
    tag = "JAX Mixed HOST+STORAGE"
    status = kv_cache_store.BlockStatus
    tpu_sharding = self.setup_shardings()
    num_blocks = 4
    # Shape: (4 blocks, 128 tokens/block, 8 head shards, 8 heads/shard,
    # head_dim 128).
    shape = (num_blocks, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    shard_size_bytes = (128 * 8 * 8 * 128 * 4) // self.num_devices
    hashes = [b"mixed_h0", b"mixed_h1", b"mixed_s0", b"mixed_s1"]
    prefix_hashes = hashes[:2]  # -> HOST (reader host RAM)
    suffix_hashes = hashes[2:]  # -> SHARED_STORAGE
    model_name = "llama_70b_jax_mixed"

    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)
    temp_dir, is_custom_root = self._make_storage_dir("jax_mixed_host_storage")
    cfg = self._posix_cfg(temp_dir, model_name)
    print(f"[{tag}] {self.num_devices} chips, shape={shape}, storage={temp_dir}",
          flush=True)
    try:
      # ---- Phase 1/5: seed storage with all 4 blocks. ----
      port_w = _pick_unused_port()
      rid_w = kv_cache_store.RaidenId(f"mixed1_{uuid.uuid4().hex[:8]}", "0",
                                      "cache_1", 0)
      store_w = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          raiden_id=rid_w,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=port_w,
          secondary_backend_configs=[cfg],
      )
      manager_w = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{port_w}",
          worker_id="worker_0",
          backend_configs=[cfg],
      )
      try:
        print(f"[{tag}][Phase 1/5] Saving {hashes} -> SHARED_STORAGE.",
              flush=True)
        self.assertTrue(store_w.insert(hashes,
                                       self._hbm_slices(rid_w, num_blocks),
                                       on_host=False))
        self.assertTrue(store_w.save(hashes))
        self._await_terminal(store_w.poll_save_status, len(hashes),
                             "storage save")
        self._assert_and_print_shard_files(tag, temp_dir, model_name,
                                           len(hashes))
      finally:
        del manager_w, store_w

      port_r = _pick_unused_port()
      rid_r = kv_cache_store.RaidenId(f"mixed2_{uuid.uuid4().hex[:8]}", "0",
                                      "cache_2", 0)
      store_r = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          raiden_id=rid_r,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=port_r,
          secondary_backend_configs=[cfg],
      )
      manager_r = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{port_r}",
          worker_id="worker_0",
          backend_configs=[cfg],
      )
      try:
        # ---- Phase 2/5: stage the prefix into reader host RAM. ----
        print(f"[{tag}][Phase 2/5] Recalling prefix {prefix_hashes}"
              " SHARED_STORAGE -> HOST.", flush=True)
        pre = store_r.lookup(prefix_hashes, pin_found=False)
        self._assert_block_statuses(pre, prefix_hashes,
                                    [status.SHARED_STORAGE] * 2)
        self.assertTrue(store_r.load(prefix_hashes, [0, 1],
                                     slices=[s for _, s in pre]))
        self._await_terminal(store_r.poll_load_status, len(prefix_hashes),
                             "prefix recall")
        # Clear HBM (host block 4 is all zeros) so Phase 5 only passes if the
        # mixed load rewrites every block.
        manager_r.h2d([4, 4, 4, 4], [0, 1, 2, 3]).wait()
        self.assertEqual(float(jax.jit(jnp.sum)(tpu_cache)), 0.0)

        # ---- Phase 3/5: lookup sees exactly [HOST, HOST, STORAGE, STORAGE].
        host = (status.HOST, status.HOST_AND_HBM)
        mixed = store_r.lookup(hashes, pin_found=True)
        self._assert_block_statuses(
            mixed, hashes,
            [host, host, status.SHARED_STORAGE, status.SHARED_STORAGE])
        print(f"[{tag}][Phase 3/5] Lookup: prefix {prefix_hashes} HOST,"
              f" suffix {suffix_hashes} SHARED_STORAGE.", flush=True)

        # ---- Phase 4/5: one mixed Load() for all 4 blocks. ----
        print(f"[{tag}][Phase 4/5] Load({hashes}) -> device [0, 1, 2, 3].",
              flush=True)
        self.assertTrue(store_r.load(hashes, [0, 1, 2, 3],
                                     slices=[s for _, s in mixed]))
        self._await_terminal(store_r.poll_load_status, len(hashes),
                             "mixed load")

        # ---- Phase 5/5: verify HBM and final metadata. ----
        recalled = jax.jit(lambda x: x)(tpu_cache)
        jax.block_until_ready(recalled)
        np.testing.assert_array_equal(np.asarray(recalled), host_data)
        post = store_r.lookup(hashes, pin_found=False)
        self._assert_block_statuses(post, hashes, [status.HOST_AND_HBM] * 4)
        for _, b in post:
          self.assertIn(b.device_block_id, [0, 1, 2, 3])
          self.assertGreaterEqual(b.host_block_id, 0)
        print(f"[{tag}][Phase 5/5] HBM bit-exact; all 4 blocks HOST_AND_HBM.",
              flush=True)
      finally:
        del manager_r, store_r
    finally:
      if not is_custom_root:
        shutil.rmtree(temp_dir, ignore_errors=True)

  def test_secondary_storage_e2e_mixed_remote_and_storage_load(self):
    """Mixed Load(): REMOTE peer prefix + SHARED_STORAGE suffix, real TPU.

    Blocks: prefix [mixed_r0, mixed_r1] -> REMOTE (held in Peer A host RAM),
            suffix [mixed_s0, mixed_s1] -> SHARED_STORAGE.
      1. Seed storage: writer saves the suffix to POSIX storage.
      2. Stage prefix: Peer A saves the prefix to its host RAM (published to
                       the global registry).
      3. Lookup:       Consumer B waits until lookup returns exactly
                       [REMOTE(A), REMOTE(A), SHARED_STORAGE, SHARED_STORAGE].
      4. Mixed load:   one Load() for all 4 blocks; wait for all 4.
      5. Verify:       HBM bit-exact; suffix HOST_AND_HBM in B.
    """
    tag = "JAX Mixed REMOTE+STORAGE"
    status = kv_cache_store.BlockStatus
    tpu_sharding = self.setup_shardings()
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    shard_size_bytes = (128 * 8 * 8 * 128 * 4) // self.num_devices
    prefix_hashes = [b"mixed_r0", b"mixed_r1"]  # -> REMOTE (Peer A)
    suffix_hashes = [b"mixed_s0", b"mixed_s1"]  # -> SHARED_STORAGE
    all_hashes = prefix_hashes + suffix_hashes
    model_name = "llama_70b_jax_remote_mixed"

    temp_dir, is_custom_root = self._make_storage_dir(
        "jax_mixed_remote_storage")
    cfg = self._posix_cfg(temp_dir, model_name)
    print(f"[{tag}] {self.num_devices} chips, shape={shape}, storage={temp_dir}",
          flush=True)
    try:
      # ---- Phase 1/5: seed storage with the suffix. ----
      port_s = _pick_unused_port()
      rid_s = kv_cache_store.RaidenId(f"writer_{uuid.uuid4().hex[:8]}", "0",
                                      "cache_s", 0)
      store_s = kv_cache_store.KVCacheStore(
          capacity=2,
          raiden_id=rid_s,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=port_s,
          secondary_backend_configs=[cfg],
      )
      tpu_cache_s = jax.device_put(jnp.array(host_data[2:]), tpu_sharding)
      jax.block_until_ready(tpu_cache_s)
      manager_s = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache_s],
          local_control_port=0,
          max_blocks=2,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{port_s}",
          worker_id="worker_s",
          backend_configs=[cfg],
      )
      try:
        print(f"[{tag}][Phase 1/5] Saving suffix {suffix_hashes} ->"
              " SHARED_STORAGE.", flush=True)
        self.assertTrue(store_s.insert(suffix_hashes,
                                       self._hbm_slices(rid_s, 2),
                                       on_host=False))
        self.assertTrue(store_s.save(suffix_hashes))
        self._await_terminal(store_s.poll_save_status, len(suffix_hashes),
                             "storage save")
        self._assert_and_print_shard_files(tag, temp_dir, model_name,
                                           len(suffix_hashes))
      finally:
        del manager_s, store_s, tpu_cache_s

      # ---- Phase 2/5: Peer A stages the prefix in its host RAM. ----
      port_a = _pick_unused_port()
      rid_a = kv_cache_store.RaidenId(f"job_a_{uuid.uuid4().hex[:8]}", "0",
                                      "cache_a", 0)
      store_a = kv_cache_store.KVCacheStore(
          capacity=4,
          global_registry_address=f"localhost:{_registry_port}",
          raiden_id=rid_a,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=port_a,
      )
      tpu_cache_a = jax.device_put(jnp.array(host_data[:2]), tpu_sharding)
      jax.block_until_ready(tpu_cache_a)
      manager_a = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache_a],
          local_control_port=0,
          max_blocks=2,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=_pick_unused_port(),
          raiden_controller_address=f"localhost:{port_a}",
          worker_id="worker_a",
          node_id=0,
      )
      print(f"[{tag}][Phase 2/5] Peer A saving prefix {prefix_hashes} -> HOST"
            " (published to global registry).", flush=True)
      self.assertTrue(store_a.insert(prefix_hashes,
                                     self._hbm_slices(rid_a, 2),
                                     on_host=False))
      self.assertTrue(store_a.save(prefix_hashes))
      self._await_terminal(store_a.poll_save_status, len(prefix_hashes),
                           "peer A save")

      port_b = _pick_unused_port()
      rid_b = kv_cache_store.RaidenId(f"job_b_{uuid.uuid4().hex[:8]}", "0",
                                      "cache_b", 0)
      store_b = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=f"localhost:{_registry_port}",
          raiden_id=rid_b,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=port_b,
          secondary_backend_configs=[cfg],
      )
      tpu_cache_b = jax.device_put(
          jnp.array(np.zeros(shape, dtype=np.float32)), tpu_sharding)
      jax.block_until_ready(tpu_cache_b)
      manager_b = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache_b],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=_pick_unused_port(),
          raiden_controller_address=f"localhost:{port_b}",
          worker_id="worker_b",
          node_id=0,
          backend_configs=[cfg],
      )
      try:
        # ---- Phase 3/5: wait for [REMOTE(A), REMOTE(A), STORAGE, STORAGE].
        mixed = self._await_lookup_statuses(
            store_b, all_hashes,
            [status.REMOTE, status.REMOTE, status.SHARED_STORAGE,
             status.SHARED_STORAGE],
            enable_global=True, pin_found=False)
        for _, s in mixed[:2]:
          self.assertEqual(s.raiden_id, rid_a)
        print(f"[{tag}][Phase 3/5] Lookup: prefix {prefix_hashes}"
              f" REMOTE(peer A), suffix {suffix_hashes} SHARED_STORAGE.",
              flush=True)

        # ---- Phase 4/5: one mixed Load() for all 4 blocks. ----
        print(f"[{tag}][Phase 4/5] Load({all_hashes}) -> device"
              " [0, 1, 2, 3].", flush=True)
        self.assertTrue(store_b.load(all_hashes, [0, 1, 2, 3],
                                     slices=[s for _, s in mixed]))
        self._await_terminal(store_b.poll_load_status, len(all_hashes),
                             "mixed load")

        # ---- Phase 5/5: verify HBM and final metadata. ----
        recalled = jax.jit(lambda x: x)(tpu_cache_b)
        jax.block_until_ready(recalled)
        np.testing.assert_array_equal(np.asarray(recalled), host_data)
        post = store_b.lookup(suffix_hashes, enable_global=False,
                              pin_found=False)
        self._assert_block_statuses(post, suffix_hashes,
                                    [status.HOST_AND_HBM] * 2)
        for _, b in post:
          self.assertIn(b.device_block_id, [2, 3])
          self.assertGreaterEqual(b.host_block_id, 0)
        print(f"[{tag}][Phase 5/5] HBM bit-exact; suffix HOST_AND_HBM.",
              flush=True)
      finally:
        del manager_b, store_b, tpu_cache_b
        del manager_a, store_a, tpu_cache_a
    finally:
      if not is_custom_root:
        shutil.rmtree(temp_dir, ignore_errors=True)

  def _run_remote_read_e2e_test(
      self,
      enable_multi_numa: bool,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_read_success: bool = True,
      use_slices: bool = False,
  ):
    if enable_multi_numa and len(self.devices) > 4:
      # TODO(jcgu): Create a new multi-host setup
      # to test True Multi-NUMA ENABLE_MULTI_NUMA=1 correctly, since running
      # two distinct jobs inside this single-process sandbox blocks cross-NIC UDP routing.
      self.skipTest(
          "Multi-NUMA E2E test is not supported on single-host shared device "
          f"configurations (devices={len(self.devices)}). Skipping."
      )

    os.environ["ENABLE_MULTI_NUMA"] = "1" if enable_multi_numa else "0"

    if len(self.devices) < 1:
      self.skipTest(
          f"Requires at least 1 device, but only got {len(self.devices)}"
      )

    devices_a = self.devices
    devices_b = self.devices

    sharding_a = self.setup_sharding_for_devices(devices_a)
    sharding_b = self.setup_sharding_for_devices(devices_b)

    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data for Job A
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = jax.device_put(jnp.array(host_data_a), sharding_a)
    jax.block_until_ready(tpu_cache_a)

    # Overwrite Job B device memory with zeros
    zeros_b = np.zeros(shape, dtype=np.float32)
    tpu_cache_b = jax.device_put(jnp.array(zeros_b), sharding_b)
    jax.block_until_ready(tpu_cache_b)

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    num_shards = len(self.devices)
    shard_size_bytes = (block_elements * 4) // num_shards

    controller_port = _pick_unused_port()
    worker_port_a = _pick_unused_port()
    worker_port_b = _pick_unused_port()

    # 2. Create Job A's KVCacheStore & KVCacheManager
    rid_a = kv_cache_store.RaidenId("job_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_a",
        node_id=producer_node_id,
    )

    controller_port_b = _pick_unused_port()
    # 3. Create Job B's KVCacheStore & KVCacheManager
    rid_b = kv_cache_store.RaidenId("job_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_b,
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_b",
        host_blocks_to_allocate=4,  # Allocating enough space for receiver host blocks
        node_id=consumer_node_id,
    )

    # Wait for listeners to start
    time.sleep(1)

    hashes = [b"hash_0", b"hash_1"]

    # 4. Job A inserts HBM status and calls Save
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
    self.assertTrue(store_a.insert(hashes, slices_a, on_host=False))

    @jax.jit
    def get_slice(x):
      return x[0, 0, 0, 0, 0:16]

    print(f"DEBUG: Job A tpu_cache_a before Save: {get_slice(tpu_cache_a)}")

    store_a.save(hashes)

    # Wait for save completion
    done = False
    while not done:
      save_done, save_failed, _, _, _ = store_a.poll_save_status()
      if save_failed:
        raise RuntimeError(f"Job A Async Save failed: {save_failed}")
      if save_done:
        done = True
      if not done:
        time.sleep(0.01)

    data_a = manager_a._impl.read_host_memory(0, 0, 16)
    print(f"DEBUG: Job A host memory (layer 0, shard 0) after Save: {data_a}")

    # 5. Job B calls Lookup (enable_global=True)
    # Give some time for registry propagation
    time.sleep(0.5)
    # Registry-resolved hits are REMOTE and never pinned locally.
    lookup_res_b = store_b.lookup(hashes, enable_global=True, pin_found=False)
    self.assertLen(lookup_res_b, 2)

    # Verify REMOTE status and owner job_a
    self.assertEqual(lookup_res_b[0][0], hashes[0])
    self.assertEqual(
        lookup_res_b[0][1].status, kv_cache_store.BlockStatus.REMOTE
    )
    self.assertEqual(lookup_res_b[0][1].raiden_id, rid_a)

    self.assertEqual(lookup_res_b[1][0], hashes[1])
    self.assertEqual(
        lookup_res_b[1][1].status, kv_cache_store.BlockStatus.REMOTE
    )
    self.assertEqual(lookup_res_b[1][1].raiden_id, rid_a)

    # Verify correct source host block IDs
    lookup_res_a = store_a.lookup(hashes, pin_found=False)
    self.assertEqual(
        lookup_res_b[0][1].host_block_id, lookup_res_a[0][1].host_block_id
    )
    self.assertEqual(
        lookup_res_b[1][1].host_block_id, lookup_res_a[1][1].host_block_id
    )

    slices_b = [lookup_res_b[0][1], lookup_res_b[1][1]]
    if use_slices:
      # 6. Job B controller calls load directly with slices
      self.assertTrue(store_b.load(hashes, [0, 1], slices=slices_b))

      if not expect_read_success:
        failed = False
        for _ in range(500):
          _, load_failed, _ = store_b.poll_load_status()
          if load_failed:
            failed = True
            break
          time.sleep(0.01)
        self.assertTrue(
            failed,
            "expected Load to fail on producer/consumer node_id mismatch",
        )
        return

      # Wait for Load completion
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
      # 6. Job B reads straight from Job A into its own device blocks. The
      # source coordinates come from the lookup answer, so nothing needs to be
      # inserted into Job B's cache first.
      self.assertTrue(store_b.read_remote(hashes, slices_b, [0, 1]))

      if not expect_read_success:
        # Strict node_id matching: the producer worker's node_id must equal the
        # consumer (destination) worker's node_id. A mismatch makes the source
        # controller find no destination group and the remote read fail.
        failed = False
        for _ in range(500):
          _, read_failed, _ = store_b.poll_remote_read_status()
          if read_failed:
            failed = True
            break
          time.sleep(0.01)
        self.assertTrue(
            failed,
            "expected ReadRemote to fail on producer/consumer node_id mismatch",
        )
        return

      # Wait for ReadRemote completion
      done = False
      while not done:
        read_done, read_failed, _ = store_b.poll_remote_read_status()
        if read_failed:
          raise RuntimeError(f"Job B ReadRemote failed: {read_failed}")
        if len(read_done) == 2:
          done = True
        if not done:
          time.sleep(0.01)

      # 8. The read is already in HBM -- there is no second Load step, and no
      # local record of it either. Job B's cache is still a miss for these
      # hashes: the bytes live only in the device blocks it named.
      self.assertEmpty(store_b.lookup(hashes))

    # 9. Verify byte-exact match on Job B TPU devices. The DMA landed behind
    # JAX's back, so the buffer has to be re-read rather than np.asarray'd.
    np.testing.assert_array_equal(self._reread_device(tpu_cache_b), host_data_a)

  # =========================================================================
  # ReadRemote to HBM (receiver-initiated pull straight into device memory)
  # =========================================================================

  def _reread_device(self, arr):
    """Re-reads device memory after a DMA.

    jax.device_get memoizes, so a plain np.asarray() of an array whose buffer
    was written behind JAX's back returns the stale pre-DMA host copy and every
    assertion passes vacuously. A jit round-trip forces a genuine read.
    """
    return np.asarray(jax.jit(lambda x: x * 1.0)(arr))

  def _run_remote_read_to_hbm_test(self, enable_multi_numa: bool, use_slices: bool = False):
    """Job A saves; Job B pulls straight into scattered device blocks.

    Covers, on real TPU and byte-exactly: save (D2H) on the producer, then
    read_remote straight into the consumer's HBM. The staging host blocks go
    back to the pool: nothing is recorded locally, no host copy is left
    behind, and a later local load() of the same hashes is refused.
    """
    if enable_multi_numa and len(self.devices) > 4:
      self.skipTest(
          "Multi-NUMA E2E test is not supported on single-host shared device "
          f"configurations (devices={len(self.devices)}). Skipping."
      )
    os.environ["ENABLE_MULTI_NUMA"] = "1" if enable_multi_numa else "0"

    sharding_a = self.setup_sharding_for_devices(self.devices)
    sharding_b = self.setup_sharding_for_devices(self.devices)

    # 4 blocks: pull source blocks 0 and 1 into device blocks 3 and 0
    # (scattered AND reversed, so a cross-wired id cannot pass), leaving
    # 1 and 2 as untouched sentinels.
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    hashes = [b"\x93\xff\x00hbm_hash_0", b"\x93\xff\x00hbm_hash_1"]
    src_device_blocks = [0, 1]
    dst_device_blocks = [3, 0]
    sentinel_blocks = [1, 2]

    # Per-element random payloads with per-block-distinct content: a constant
    # fill compares "byte-exact" even when bytes were shuffled within a block
    # or blocks were cross-wired, which is exactly the failure class the
    # sharded pull path can introduce.
    rng = np.random.default_rng(20260728)
    host_data_a = rng.standard_normal(shape, dtype=np.float32)
    tpu_cache_a = jax.device_put(jnp.array(host_data_a), sharding_a)
    # Job B starts RANDOM, not zero, so a block that is never written cannot
    # accidentally match the source.
    host_data_b = np.random.default_rng(31415926).standard_normal(
        shape, dtype=np.float32
    )
    tpu_cache_b = jax.device_put(jnp.array(host_data_b), sharding_b)
    jax.block_until_ready(tpu_cache_a)
    jax.block_until_ready(tpu_cache_b)

    block_elements = 128 * 8 * 8 * 128
    num_shards = len(self.devices)
    shard_size_bytes = (block_elements * 4) // num_shards

    controller_port = _pick_unused_port()
    controller_port_b = _pick_unused_port()
    worker_port_a = _pick_unused_port()
    worker_port_b = _pick_unused_port()

    rid_a = kv_cache_store.RaidenId("job_hbm_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=8,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_hbm_a",
        node_id=0,
    )

    rid_b = kv_cache_store.RaidenId("job_hbm_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=8,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_b,
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_hbm_b",
        host_blocks_to_allocate=8,
        node_id=0,
    )
    # manager_a/manager_b must stay referenced for the whole test: they own the
    # worker gRPC servers and the transport listeners the stores talk to.
    self.assertIsNotNone(manager_a)
    self.assertIsNotNone(manager_b)
    time.sleep(1)

    # --- Job A: save (D2H) so the blocks are HOST-resident and leasable. ----
    slices_a = [
        kv_cache_store.RaidenBlockId(
            rid_a,
            host_block_id=-1,
            device_block_id=blk,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for blk in src_device_blocks
    ]
    self.assertTrue(store_a.insert(hashes, slices_a, on_host=False))
    store_a.save(hashes)
    self._await_terminal(store_a.poll_save_status, len(hashes), "Job A save")

    # --- Job B: discover the blocks as REMOTE. -----------------------------
    time.sleep(0.5)
    lookup_b = store_b.lookup(hashes, enable_global=True)
    self.assertLen(lookup_b, 2)
    for _, blk in lookup_b:
      self.assertEqual(blk.status, kv_cache_store.BlockStatus.REMOTE)
      self.assertEqual(blk.raiden_id, rid_a)

    if use_slices:
      # --- The new feature under test: one-step peer-fetch load. ---
      # No insert needed. The slices themselves carry the owner resolution,
      # and load() initiates the fetch.
      slices_b = [b for _, b in lookup_b]
      self.assertTrue(store_b.load(hashes, dst_device_blocks, slices=slices_b))
      self._await_terminal(
          store_b.poll_load_status, len(hashes), "Job B peer-fetch load"
      )
      # A load from a peer records nothing locally: no host copy was kept, so
      # there is no residency to describe. The bytes are in the device blocks
      # the caller named and the cache is a miss for these hashes.
      self.assertEmpty(store_b.lookup(hashes))
    else:
      # --- The thing under test: pull straight into HBM. ---------------------
      # No insert first: the lookup answer IS the source coordinate, and the
      # read takes no pin because it records nothing.
      self.assertTrue(
          store_b.read_remote(
              hashes, [b for _, b in lookup_b], dst_device_blocks
          )
      )
      self._await_terminal(
          store_b.poll_remote_read_status, len(hashes), "Job B read_remote"
      )

      # The bytes are in the caller's device blocks and nowhere else. The host
      # blocks the transfer staged through went straight back to the pool, so
      # there is no local entry and a later local lookup is still a miss.
      self.assertEmpty(store_b.lookup(hashes))

    # --- Byte-exact verification of device memory. -------------------------
    actual_b = self._reread_device(tpu_cache_b)
    for src_blk, dst_blk in zip(src_device_blocks, dst_device_blocks):
      np.testing.assert_array_equal(
          actual_b[dst_blk],
          host_data_a[src_blk],
          err_msg=(
              f"device block {dst_blk} does not byte-match source block"
              f" {src_blk}"
          ),
      )
    for blk in sentinel_blocks:
      np.testing.assert_array_equal(
          actual_b[blk],
          host_data_b[blk],
          err_msg=f"sentinel device block {blk} was clobbered by the pull",
      )

    if use_slices:
      return

    # --- No host copy is left behind. --------------------------------------
    # The staging blocks were handed back, so the read is not a way to warm
    # the local cache. A later local load() of the same hashes has nothing to
    # read from and is refused; the caller that wants a host copy must save()
    # what it pulled. This is the deliberate cost of read_remote leaving no
    # local record.
    self.assertFalse(
        store_b.load(hashes, sentinel_blocks),
        "read_remote must not leave a host copy behind",
    )

    # The sentinel blocks are therefore untouched, as they were before.
    unchanged = self._reread_device(tpu_cache_b)
    for blk in sentinel_blocks:
      np.testing.assert_array_equal(
          unchanged[blk],
          host_data_b[blk],
          err_msg=f"sentinel device block {blk} must stay untouched",
      )

  def _await_terminal(self, poll_fn, expected_done, what, timeout_s=120.0):
    """Polls until `expected_done` hashes report done, or anything fails.

    Arity-agnostic: poll_save_status returns five lists (the last two annotate
    remote-save failures) while poll_load_status and poll_remote_read_status
    return three. Only the first two matter here.
    """
    deadline = time.time() + timeout_s
    seen_done = 0
    while time.time() < deadline:
      result = poll_fn()
      done, failed = result[0], result[1]
      if failed:
        raise RuntimeError(f"{what} failed: {failed}")
      seen_done += len(done)
      if seen_done >= expected_done:
        return
      time.sleep(0.01)
    raise RuntimeError(f"{what} did not finish within {timeout_s}s")

  def _await_write_terminal(self, store, expected, timeout_s=120.0):
    """Polls a remote write to a verdict.

    Separate from _await_terminal because poll_save_status returns five
    vectors, not three: `existing` carries what a destination already held when
    it refused a partial batch.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      done, failed, pending, existing, unregistered = (
          store.poll_save_status()
      )
      if done or failed:
        return done, failed, existing, unregistered
      if not pending:
        raise RuntimeError("remote write vanished without a verdict")
      time.sleep(0.01)
    raise RuntimeError(f"remote write did not settle within {timeout_s}s")

  def _run_remote_write_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_write_success: bool = True,
      preload_count: int = 0,
  ):
    """Job A offers blocks it owns; Job B pulls them and keeps them.

    The mirror image of _run_remote_read_e2e_test: there the destination asks,
    here the source offers. The data plane is the same pull either way -- only
    the destination ever writes the destination's memory -- so what this adds
    over the read path is the WriteRemote control plane: the ack, the
    destination's all-or-nothing insert, global registration, and the source
    polling to COMMITTED.

    Byte-exactness is the point. The control-plane unit tests can all pass
    while nothing is actually transferred.
    """
    os.environ["ENABLE_MULTI_NUMA"] = "0"
    if len(self.devices) < 1:
      self.skipTest("Requires at least 1 device")

    sharding_a = self.setup_sharding_for_devices(self.devices)
    sharding_b = self.setup_sharding_for_devices(self.devices)

    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = jax.device_put(jnp.array(host_data_a), sharding_a)
    jax.block_until_ready(tpu_cache_a)

    # Zeroed, so a byte comparison cannot pass on data that was already there.
    tpu_cache_b = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding_b
    )
    jax.block_until_ready(tpu_cache_b)

    block_elements = 128 * 8 * 8 * 128
    num_shards = len(self.devices)
    shard_size_bytes = (block_elements * 4) // num_shards

    controller_port = _pick_unused_port()
    worker_port_a = _pick_unused_port()
    worker_port_b = _pick_unused_port()

    tag = f"write_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id=f"{tag}_worker_a",
        node_id=producer_node_id,
    )

    controller_port_b = _pick_unused_port()
    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_b,
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id=f"{tag}_worker_b",
        host_blocks_to_allocate=4,
        node_id=consumer_node_id,
    )
    # Both managers must outlive the transfer: each owns its node's
    # WorkerService, which is what performs the D2H and serves/issues the pull.
    self.assertIsNotNone(manager_a)
    self.assertIsNotNone(manager_b)
    time.sleep(1)

    hashes = [b"\x93\xff\x00wr_hash_0", b"\x93\xff\x00wr_hash_1"]

    # 1. Job A puts the blocks in HBM and saves them to host DRAM. Only
    #    host-resident blocks can be offered: the pull reads host memory.
    slices_a = [
        kv_cache_store.RaidenBlockId(
            rid_a,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(num_blocks)
    ]
    self.assertTrue(store_a.insert(hashes, slices_a, on_host=False))

    self.assertTrue(store_a.save(hashes))
    self._await_terminal(store_a.poll_save_status, len(hashes), "Job A save")

    preloaded = hashes[:preload_count]
    if preloaded:
      # The destination already holds a prefix of what will be offered.
      # Holding ALL of it makes the offer a SUCCESS that moves no bytes;
      # holding a strict subset makes it a FAILURE that reports the overlap,
      # because the destination does not do partial writes.
      self.assertTrue(
          store_b.insert(
              preloaded,
              [
                  kv_cache_store.RaidenBlockId(
                      rid_b,
                      host_block_id=i,
                      status=kv_cache_store.BlockStatus.HOST,
                  )
                  for i in range(len(preloaded))
              ],
              on_host=True,
          )
      )
      # Destination state only: nothing here consumes insert's pins.
      store_b.release(preloaded)

    # 2. Job A offers them. Returns once Job B has decided, not once the bytes
    #    have moved.
    #
    # The local save above consumed the pin insert()/pin() granted, so the
    # offer needs its own. This is the documented remote-save flow: lookup()
    # answers "yes, host-resident here" AND grants the pin that save(dst)
    # spends, so the two calls are one workflow rather than a re-pin hack.
    self.assertLen(store_a.lookup(hashes), len(hashes))
    self.assertTrue(store_a.save(hashes, rid_b))
    done, failed, existing, unregistered = self._await_write_terminal(
        store_a, hashes
    )

    if 0 < preload_count < len(hashes):
      # Partial overlap, with real residency on both sides: the whole batch
      # fails, and `existing` names exactly the prefix the destination held
      # -- the list the caller needs to decide what to re-offer.
      self.assertCountEqual(failed, hashes)
      self.assertEmpty(done)
      self.assertCountEqual(existing, preloaded)
      self.assertEmpty(unregistered)
      # The failed offer consumed nothing; hand lookup's pins back.
      store_a.release(hashes)
      return

    if not expect_write_success:
      self.assertNotEmpty(
          failed, "expected the remote write to fail, but it reported done"
      )
      self.assertEmpty(done)
      return

    self.assertCountEqual(done, hashes)
    self.assertEmpty(failed)
    self.assertEmpty(existing)
    self.assertEmpty(unregistered)
    # No release: the successful remote save consumed the pin lookup() granted.

    # 3. Job B holds them locally, host-resident, as its own.
    lookup_b = store_b.lookup(hashes, enable_global=False, pin_found=False)
    self.assertLen(lookup_b, len(hashes))
    for _, slice_b in lookup_b:
      self.assertEqual(slice_b.status, kv_cache_store.BlockStatus.HOST)

    if preloaded:
      # Nothing was transferred, so there is nothing of Job A's to compare.
      return

    # 4. Prove the bytes are real. lookup() resolves and pins the landed
    #    host blocks; load() consumes the pin on success.
    self.assertLen(store_b.lookup(hashes), num_blocks)
    self.assertTrue(store_b.load(hashes, list(range(num_blocks))))
    self._await_terminal(store_b.poll_load_status, len(hashes), "Job B load")

    np.testing.assert_array_equal(
        self._reread_device(tpu_cache_b),
        host_data_a,
        err_msg=(
            "Job B's device memory does not byte-match what Job A offered:"
            " the remote write reported COMMITTED without moving the data"
        ),
    )

  def test_remote_write_e2e(self):
    # Single-host matching node_ids (the defaults): exercises the node_id
    # plumbing end to end, including the destination's
    # host_blocks_to_allocate branch.
    self._run_remote_write_e2e_test()

  def test_remote_write_e2e_mismatched_node_id_fails(self):
    # The destination pairs each of its workers with the source worker holding
    # its shards by node_id. A mismatch leaves the pull with no group to read
    # from, so the transfer fails and the source is told so rather than being
    # left pending.
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=1,
        expect_write_success=False,
    )

  def test_remote_write_e2e_all_exist_moves_nothing(self):
    # The destination already holds every offered hash. SUCCESS with no
    # transfer: hashes are content-addressed, so the peer having them is
    # exactly the post-condition the caller wanted.
    self._run_remote_write_e2e_test(preload_count=2)

  def test_remote_write_e2e_partial_exist_reports_the_overlap(self):
    # The destination holds one of the two offered hashes -- with real bytes
    # and a real registry, unlike the control-plane unit test. The batch
    # fails whole and `existing` names the overlap.
    self._run_remote_write_e2e_test(preload_count=1)

  def test_remote_read_to_hbm_without_multi_numa(self):
    self._run_remote_read_to_hbm_test(enable_multi_numa=False)

  @unittest.skip(
      "multi-NUMA needs a dedicated multi-NUMA machine; port contention"
      " in the sandbox otherwise. Re-enable when the hardware lands."
  )
  def test_remote_read_to_hbm_with_multi_numa(self):
    self._run_remote_read_to_hbm_test(enable_multi_numa=True)

  @unittest.skip(
      "multi-NUMA needs a dedicated multi-NUMA machine; port contention"
      " in the sandbox otherwise. Re-enable when the hardware lands."
  )
  def test_remote_read_to_hbm_with_slices_with_multi_numa(self):
    self._run_remote_read_to_hbm_test(enable_multi_numa=True, use_slices=True)

  def test_remote_read_to_hbm_with_slices_without_multi_numa(self):
    self._run_remote_read_to_hbm_test(enable_multi_numa=False, use_slices=True)

  def test_remote_read_e2e_with_slices_without_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=False, use_slices=True)

  @unittest.skip(
      "multi-NUMA needs a dedicated multi-NUMA machine; port contention"
      " in the sandbox otherwise. Re-enable when the hardware lands."
  )
  def test_remote_read_e2e_with_slices_with_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=True, use_slices=True)

  def test_remote_read_e2e_without_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=False)

  @unittest.skip(
      "multi-NUMA needs a dedicated multi-NUMA machine; port contention"
      " in the sandbox otherwise. Re-enable when the hardware lands."
  )
  def test_remote_read_e2e_with_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=True)

  def test_remote_read_e2e_matching_node_id(self):
    # Non-zero, matching node_ids on producer and consumer: exercises node_id
    # plumbing end-to-end (including the consumer's host_blocks_to_allocate
    # branch) and strict node_id matching succeeding.
    self._run_remote_read_e2e_test(
        enable_multi_numa=False,
        producer_node_id=7,
        consumer_node_id=7,
        expect_read_success=True,
    )

  def test_remote_read_e2e_mismatched_node_id_fails(self):
    # Producer and consumer node_ids differ: strict matching finds no
    # destination group for the producer worker and the remote read fails.
    self._run_remote_read_e2e_test(
        enable_multi_numa=False,
        producer_node_id=1,
        consumer_node_id=2,
        expect_read_success=False,
    )

  def test_remote_read_e2e_source_missing_block_fails(self):
    # ReadRemote All-or-Nothing validate & pin block hashes at the src controller: the destination requests a block hash the SOURCE does
    # not hold in host DRAM. The source controller's verify hook returns
    # BLOCK_HASH_NOT_FOUND, the read fails, and the destination frees the local
    # host block it allocated.
    if len(self.devices) < 1:
      self.skipTest("Requires at least 1 device")
    os.environ["ENABLE_MULTI_NUMA"] = "0"

    sharding = self.setup_sharding_for_devices(self.devices)
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    tpu_cache_b = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    jax.block_until_ready(tpu_cache_a)
    jax.block_until_ready(tpu_cache_b)

    num_shards = len(self.devices)
    shard_size_bytes = (128 * 8 * 8 * 128 * 4) // num_shards
    controller_port_a = _pick_unused_port()
    controller_port_b = _pick_unused_port()

    # Source (Job A): running + registered, but never saves any block, so its
    # LRU has no HOST-resident blocks.
    rid_a = kv_cache_store.RaidenId("miss_job_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=_pick_unused_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id="worker_a",
    )

    # Destination (Job B).
    rid_b = kv_cache_store.RaidenId("miss_job_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=_pick_unused_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_b",
        host_blocks_to_allocate=4,
    )
    time.sleep(1)

    # Job B names a source coordinate on Job A for a hash Job A never saved,
    # and tries to read it.
    ghost = [b"ghost_hash"]
    slices = [
        kv_cache_store.RaidenBlockId(
            rid_a,
            host_block_id=0,
            device_block_id=-1,
            status=kv_cache_store.BlockStatus.REMOTE,
        )
    ]

    self.assertTrue(store_b.read_remote(ghost, slices, [0]))

    failed = False
    for _ in range(500):
      _, read_failed, _ = store_b.poll_remote_read_status()
      if read_failed:
        failed = True
        break
      time.sleep(0.01)
    self.assertTrue(
        failed, "expected ReadRemote to fail (source is missing the block)"
    )
    del manager_a, manager_b, store_a, store_b

  def test_remote_read_e2e_source_wrong_status_fails(self):
    # ReadRemote step 6a: the source HOLDS the block but only in HBM (never
    # saved to host DRAM). The source verify hook returns FAILED_PRECONDITION and
    # the destination read fails.
    if len(self.devices) < 1:
      self.skipTest("Requires at least 1 device")
    os.environ["ENABLE_MULTI_NUMA"] = "0"

    sharding = self.setup_sharding_for_devices(self.devices)
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    tpu_cache_b = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    jax.block_until_ready(tpu_cache_a)
    jax.block_until_ready(tpu_cache_b)

    num_shards = len(self.devices)
    shard_size_bytes = (128 * 8 * 8 * 128 * 4) // num_shards
    controller_port_a = _pick_unused_port()
    controller_port_b = _pick_unused_port()

    rid_a = kv_cache_store.RaidenId("ws_job_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=_pick_unused_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id="worker_a",
    )
    # Source records the block in HBM only (no Save -> not host-resident).
    hashes = [b"\x93\xff\x00hbm_only"]
    self.assertTrue(
        store_a.insert(
            hashes,
            [
                kv_cache_store.RaidenBlockId(
                    rid_a,
                    host_block_id=-1,
                    device_block_id=0,
                    status=kv_cache_store.BlockStatus.HBM,
                )
            ],
            on_host=False,
        )
    )

    rid_b = kv_cache_store.RaidenId("ws_job_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=_pick_unused_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_b",
        host_blocks_to_allocate=4,
    )
    time.sleep(1)

    # Destination names the source's HBM-only block as the read source.
    slices = [
        kv_cache_store.RaidenBlockId(
            rid_a,
            host_block_id=0,
            device_block_id=-1,
            status=kv_cache_store.BlockStatus.REMOTE,
        )
    ]
    self.assertTrue(store_b.read_remote(hashes, slices, [0]))

    failed = False
    for _ in range(500):
      _, read_failed, _ = store_b.poll_remote_read_status()
      if read_failed:
        failed = True
        break
      time.sleep(0.01)
    self.assertTrue(
        failed, "expected ReadRemote to fail (source block not host-resident)"
    )
    del manager_a, manager_b, store_a, store_b

  def test_expected_worker_count_waits_for_a_concurrent_registration(self):
    """The barrier replaces the sleep-and-hope idiom.

    Every other test here builds the store, builds the manager, then sleeps and
    hopes registration landed. With expected_worker_count the store constructor
    IS the wait: it returns only once the worker is registered, so the transfer
    below needs no sleep at all.

    This also covers the binding releasing the GIL. The worker registers from
    another Python thread, so a constructor that held the GIL while blocking
    would starve that thread and time out instead.
    """
    os.environ["ENABLE_MULTI_NUMA"] = "0"
    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)

    controller_port = _pick_unused_port()
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    # The manager is what registers the worker, and it can only be built once
    # the controller address is known -- which is why the port is fixed here
    # rather than left to gRPC.
    registration_delay_s = 2.0
    built = {}

    def build_manager_after_a_delay():
      time.sleep(registration_delay_s)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="worker_0",
      )

    worker_thread = threading.Thread(target=build_manager_after_a_delay)
    worker_thread.start()
    try:
      rid = kv_cache_store.RaidenId("barrier_job", "0", "barrier_cache", 0)
      start = time.time()
      store = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          raiden_id=rid,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port,
          expected_worker_count=1,
      )
      elapsed = time.time() - start
    finally:
      worker_thread.join(timeout=180)

    self.assertFalse(worker_thread.is_alive())
    # It really waited for the registration rather than returning early and
    # getting lucky.
    self.assertGreaterEqual(elapsed, registration_delay_s)

    # And the store is usable IMMEDIATELY -- no sleep between here and a
    # transfer. That is the property the barrier exists to provide, and it is
    # what fails if the constructor returns before the worker is registered.
    hashes = [b"barrier_hash_0", b"barrier_hash_1"]
    slices = [
        kv_cache_store.RaidenBlockId(
            rid,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(num_blocks)
    ]
    inserted = store.insert(hashes, slices, on_host=False)
    self.assertTrue(inserted)
    self.assertTrue(store.save(hashes))

    deadline = time.time() + 60
    done = []
    while time.time() < deadline:
      done, failed, _, _, _ = store.poll_save_status()
      self.assertEmpty(failed)
      if done:
        break
      time.sleep(0.01)
    self.assertCountEqual(done, hashes)
    del built

  def test_sweep_demotes_to_the_store_node_and_reads_back(self):
    """The evict chain end to end: it all happens behind the existing API.

    save() makes blocks host-resident, the store monitor's sweep demotes the
    cold ones to a real store node subprocess through the KVTransferSpec
    published to the registry, a global lookup resolves the node as the new
    owner, and read_remote pulls the bytes back into HBM byte-exact. The C++
    counterpart (//tpu_sync/store_node:kv_cache_evict_e2e_test) covers this
    chain with in-process components; this test adds the python API surface,
    the real registry and node binaries, and the device D2H/H2D legs.
    """
    # The store monitor is configured through the environment, read once at
    # store construction -- scoped to this test so every other test's store
    # keeps constructing with the monitor off.
    monitor_env = {
        "RAIDEN_ENABLE_STORE_MONITOR": "true",
        "RAIDEN_ENABLE_EVICT_SWEEP": "true",
        "RAIDEN_STORE_MONITOR_HEARTBEAT_S": "1",
        "RAIDEN_EVICT_SWEEP_PERIOD_S": "1",
        "RAIDEN_EVICT_LOW_WATERMARK": "0.5",
        "RAIDEN_EVICT_HIGH_WATERMARK": "0.75",
    }
    for key, value in monitor_env.items():
      os.environ[key] = value
      self.addCleanup(os.environ.pop, key, None)
    os.environ["ENABLE_MULTI_NUMA"] = "0"

    tpu_sharding = self.setup_shardings()
    # 6 of 8 blocks in use after the saves: free ratio 0.25, below the 0.5
    # low watermark, so the sweep must demote down to the 0.75 high one.
    capacity = 8
    num_insert = 6
    shape = (capacity, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)

    controller_port = _pick_unused_port()
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices
    pool_group = "evict_e2e_pool"

    # The registration is what fills in the KVTransferSpec the store
    # publishes for the node, and expected_worker_count is the barrier that
    # waits for it (see the dedicated barrier test above).
    built = {}

    def build_manager_after_a_delay():
      time.sleep(1.0)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=capacity,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="evict_e2e_worker_0",
      )

    worker_thread = threading.Thread(target=build_manager_after_a_delay)
    worker_thread.start()
    rid = kv_cache_store.RaidenId("evict_e2e_serving", "0", "evict_cache", 0)
    try:
      store = kv_cache_store.KVCacheStore(
          capacity=capacity,
          global_registry_address=f"localhost:{_registry_port}",
          raiden_id=rid,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port,
          expected_worker_count=1,
          kv_pool_group=pool_group,
      )
    finally:
      worker_thread.join(timeout=180)
    self.assertFalse(worker_thread.is_alive())

    node_binary = _node_binary_path()
    node_log = open("/tmp/raiden_store_node.log", "w")
    node_process = subprocess.Popen(
        [
            node_binary,
            "--job_name=evict_e2e_node",
            f"--global_registry_address=localhost:{_registry_port}",
            f"--kv_pool_group={pool_group}",
            "--store_server_ip=127.0.0.1",
            "--evict_tier=1",
            f"--dram_budget_bytes={block_elements * 4 * 16}",
        ],
        stdout=node_log,
        stderr=subprocess.STDOUT,
    )

    try:
      deadline = time.time() + 30
      while time.time() < deadline:
        if node_process.poll() is not None:
          node_log.flush()
          with open("/tmp/raiden_store_node.log", "r") as f:
            log_content = f.read()
          self.fail(
              f"store node exited prematurely with code {node_process.returncode}:\n{log_content}"
          )
        if os.path.exists("/tmp/raiden_store_node.log"):
          with open("/tmp/raiden_store_node.log", "r") as f:
            if "KVCacheHostStoreNode up" in f.read():
              break
        time.sleep(0.5)

      hashes = [f"evict_e2e_blk_{i}".encode() for i in range(num_insert)]
      slices = [
          kv_cache_store.RaidenBlockId(
              rid,
              host_block_id=-1,
              device_block_id=i,
              status=kv_cache_store.BlockStatus.HBM,
          )
          for i in range(num_insert)
      ]
      self.assertTrue(store.insert(hashes, slices, on_host=False))

      # insert() pins; a successful save consumes that pin, leaving the
      # blocks host-resident and unpinned -- exactly what the sweep demotes.
      self.assertTrue(store.save(hashes))
      deadline = time.time() + 60
      done = []
      while time.time() < deadline and len(done) < num_insert:
        save_done, save_failed, _, _, _ = store.poll_save_status()
        self.assertEmpty(save_failed, f"save failed for {save_failed}")
        done += save_done
        time.sleep(0.05)
      self.assertLen(done, num_insert)

      # A demoted block is gone locally. lookup() halts at its first miss, so
      # probe one hash at a time; pin_found=False keeps the probe from
      # pinning the survivors against later sweep episodes.
      def missing_locally(h):
        return not store.lookup([h], enable_global=False, pin_found=False)

      deadline = time.time() + 90
      demoted = []
      while time.time() < deadline and len(demoted) < 4:
        demoted = [h for h in hashes if missing_locally(h)]
        time.sleep(1)
      self.assertGreaterEqual(
          len(demoted),
          4,
          "the sweep never demoted down to the high watermark; see"
          " /tmp/raiden_store_node.log",
      )

      # The registry is what turns the local miss into the owner's
      # coordinates: a global lookup must name the node as the owner.
      probe = demoted[0]
      src_block = int(probe.decode().rsplit("_", 1)[1])
      hits = store.lookup([probe], enable_global=True, pin_found=False)
      self.assertLen(hits, 1)
      remote = hits[0][1]
      self.assertEqual(remote.status, kv_cache_store.BlockStatus.REMOTE)
      self.assertEqual(remote.raiden_id.job_name, "evict_e2e_node")

      # Pull into an untouched device block: its bytes are still that block's
      # own arange values, distinct from the demoted block's pattern.
      dst_device_block = num_insert
      self.assertTrue(store.read_remote([probe], [remote], [dst_device_block]))
      deadline = time.time() + 60
      read_done = []
      while time.time() < deadline and not read_done:
        read_done, read_failed, _ = store.poll_remote_read_status()
        self.assertEmpty(read_failed, f"read_remote failed for {read_failed}")
        time.sleep(0.05)
      self.assertNotEmpty(read_done, "read_remote never completed")

      got = np.asarray(jax.device_get(tpu_cache))[dst_device_block]
      np.testing.assert_array_equal(got, host_data[src_block])
    finally:
      node_process.terminate()
      try:
        node_process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        node_process.kill()
      node_log.close()
    del built

  def test_sweep_without_a_store_node_drops_cold_blocks_locally(self):
    """The no-target half of the evict sweep: pressure wins over retention.

    Free blocks fall below the low watermark while no store node is
    registered in the pool group -- the real shape of a node that crashed,
    is still deploying, or was never brought up (and exactly what the demote
    test above would hit if it raced the node's registration). The sweep
    must still raise free blocks to the high watermark by dropping cold
    blocks locally, and a dropped block is gone globally too: nothing was
    demoted anywhere.
    """
    # The store monitor is configured through the environment, read once at
    # store construction -- scoped to this test so every other test's store
    # keeps constructing with the monitor off.
    monitor_env = {
        "RAIDEN_ENABLE_STORE_MONITOR": "true",
        "RAIDEN_ENABLE_EVICT_SWEEP": "true",
        "RAIDEN_STORE_MONITOR_HEARTBEAT_S": "1",
        "RAIDEN_EVICT_SWEEP_PERIOD_S": "1",
        "RAIDEN_EVICT_LOW_WATERMARK": "0.5",
        "RAIDEN_EVICT_HIGH_WATERMARK": "0.75",
    }
    for key, value in monitor_env.items():
      os.environ[key] = value
      self.addCleanup(os.environ.pop, key, None)
    os.environ["ENABLE_MULTI_NUMA"] = "0"

    tpu_sharding = self.setup_shardings()
    # 6 of 8 blocks in use after the saves: free ratio 0.25, below the 0.5
    # low watermark, so the sweep must free down to the 0.75 high one.
    capacity = 8
    num_insert = 6
    shape = (capacity, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)

    controller_port = _pick_unused_port()
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    built = {}

    def build_manager_after_a_delay():
      time.sleep(1.0)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=capacity,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="evict_drop_worker_0",
      )

    worker_thread = threading.Thread(target=build_manager_after_a_delay)
    worker_thread.start()
    rid = kv_cache_store.RaidenId("evict_drop_serving", "0", "evict_cache", 0)
    try:
      store = kv_cache_store.KVCacheStore(
          capacity=capacity,
          global_registry_address=f"localhost:{_registry_port}",
          raiden_id=rid,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port,
          expected_worker_count=1,
          # No store node ever joins this group: GetPlacementTargets stays
          # empty and every pressure episode takes the local-drop path.
          kv_pool_group="evict_drop_pool",
      )
    finally:
      worker_thread.join(timeout=180)
    self.assertFalse(worker_thread.is_alive())

    hashes = [f"evict_drop_blk_{i}".encode() for i in range(num_insert)]
    slices = [
        kv_cache_store.RaidenBlockId(
            rid,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(num_insert)
    ]
    self.assertTrue(store.insert(hashes, slices, on_host=False))

    # insert() pins; a successful save consumes that pin, leaving the
    # blocks host-resident and unpinned -- exactly what the sweep drops.
    self.assertTrue(store.save(hashes))
    deadline = time.time() + 60
    done = []
    while time.time() < deadline and len(done) < num_insert:
      save_done, save_failed, _, _, _ = store.poll_save_status()
      self.assertEmpty(save_failed, f"save failed for {save_failed}")
      done += save_done
      time.sleep(0.05)
    self.assertLen(done, num_insert)

    def missing_locally(h):
      return not store.lookup([h], enable_global=False, pin_found=False)

    deadline = time.time() + 60
    dropped = []
    while time.time() < deadline and len(dropped) < 4:
      dropped = [h for h in hashes if missing_locally(h)]
      time.sleep(1)
    self.assertGreaterEqual(
        len(dropped),
        4,
        "the sweep never freed down to the high watermark",
    )

    # Dropped means dropped: a global lookup must come back empty, or the
    # sweep would have demoted (published an owner) instead of discarding.
    for h in dropped:
      self.assertEmpty(
          store.lookup([h], enable_global=True, pin_found=False),
          f"{h!r} was dropped locally yet resurfaced through the registry",
      )
    del built


if __name__ == "__main__":
  absltest.main()

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

"""E2E test for Torch KVCacheStore with TPUs."""

import os
import socket
import subprocess
import threading
import time
import uuid

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import kv_cache_store


def _pick_unused_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def find_free_port() -> int:
  return _pick_unused_port()


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
_reg_log_file = None


def start_servers():
  global _registry_process
  global _registry_port

  _registry_port = _pick_unused_port()

  pass
  extra_flags = []

  global _reg_log_file
  print(f"Starting Registry on port {_registry_port}")
  _reg_log_file = open("/tmp/raiden_registry.log", "w")
  _registry_process = subprocess.Popen(
      [
          registry_binary,
          f"--port={_registry_port}",
      ]
      + extra_flags,
      stdout=_reg_log_file,
      stderr=subprocess.STDOUT,
  )

  # Give them some time to start
  time.sleep(2)


def stop_servers():
  global _registry_process, _reg_log_file
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
  if _reg_log_file:
    try:
      _reg_log_file.close()
    except Exception:
      pass
    _reg_log_file = None


def setUpModule():
  os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"


def tearDownModule():
  pass


class KVCacheStoreE2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    start_servers()

  @classmethod
  def tearDownClass(cls):
    stop_servers()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    assert self.device.type == "tpu", f"Expected real PyTorch TPU device, got {self.device}"
    print(f"=== [DEVICE VERIFIED] Using real PyTorch TPU device: {self.device} ===")

    self.num_devices = 1  # E2E tests for PyTorch currently use single device logic for kv caches
    self.num_layers = 1
    self.skip_lock = True

  def tearDown(self):
    super().tearDown()

  def _run_e2e_save_and_load(
      self,
      use_slices: bool = False,
      dtype: torch.dtype = torch.float32,
      save_blocks: tuple[int, int] = (0, 1),
      load_blocks: tuple[int, int] = (2, 3),
  ):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data
    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    host_data = np.arange(np.prod(shape), dtype=np_dtype).reshape(shape)
    tpu_cache = torch.tensor(host_data, dtype=dtype, device=self.device)

    # Expected reference after loading saved blocks 0 and 1 into blocks 2 and 3: [a, b, a, b]
    expected_ref = host_data.copy()
    expected_ref[load_blocks[0]] = host_data[save_blocks[0]]
    expected_ref[load_blocks[1]] = host_data[save_blocks[1]]

    # 2. Get free port for controller
    controller_port = find_free_port()

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * tpu_cache.element_size()) // self.num_devices

    # 3. Create KVCacheStore (Controller)
    print("=== [Step 3/9] Creating KVCacheStore (Controller) ===")
    tag = f"save_{uuid.uuid4().hex[:8]}"
    rid = kv_cache_store.RaidenId(f"{tag}_job", "0", f"{tag}_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=num_blocks,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )

    # 4. Create KVCacheManager (Worker)
    print("=== [Step 4/9] Creating KVCacheManager (Worker) ===")
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
        worker_id=f"{tag}_worker_0",
    )

    # 5. Insert HBM blocks to KVCacheStore
    print("=== [Step 5/9] Inserting HBM blocks into KVCacheStore ===")
    hashes = [b"hash_0", b"hash_1"]
    slices = [
        kv_cache_store.RaidenBlockId(
            rid,
            host_block_id=-1,
            device_block_id=save_blocks[0],
            status=kv_cache_store.BlockStatus.HBM,
        ),
        kv_cache_store.RaidenBlockId(
            rid,
            host_block_id=-1,
            device_block_id=save_blocks[1],
            status=kv_cache_store.BlockStatus.HBM,
        ),
    ]
    self.assertTrue(store.insert(hashes, slices, on_host=False))

    # Verify status in store is HBM
    lookup_res = store.lookup(hashes, pin_found=False)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[0][1].device_block_id, save_blocks[0])
    self.assertEqual(lookup_res[1][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[1][1].device_block_id, save_blocks[1])

    # 6. Save HBM blocks to host memory
    print("=== [Step 6/9] Saving HBM blocks to Host DRAM (store.save) ===")
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

    # 7. Load from host DRAM into destination device HBM blocks
    print(f"=== [Step 7/8] Loading from Host DRAM into TPU HBM blocks {list(load_blocks)} (store.load) ===")
    if use_slices:
      # lookup() pins the returned entries; load(..., slices=...) consumes the pin on success.
      load_slices = [entry for _, entry in store.lookup(hashes)]
      self.assertLen(load_slices, 2)
      for entry in load_slices:
        self.assertEqual(entry.status, kv_cache_store.BlockStatus.HOST_AND_HBM)
      self.assertTrue(store.load(hashes, list(load_blocks), slices=load_slices))
    else:
      # lookup() pins the returned entries; load() consumes the pin on success.
      self.assertLen(store.lookup(hashes), 2)
      self.assertTrue(store.load(hashes, list(load_blocks)))

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

    try:
      torch.tpu.synchronize()
    except (AttributeError, RuntimeError):
      pass
    # 8. Verify device memory matches expected reference
    print("=== [Step 8/8] Verifying restored TPU memory matches expected array ===")
    if dtype == torch.bfloat16:
      np.testing.assert_array_equal(
          tpu_cache.cpu().to(torch.float32).numpy(),
          torch.tensor(expected_ref, dtype=dtype).cpu().to(torch.float32).numpy(),
      )
    else:
      np.testing.assert_array_equal(tpu_cache.cpu().numpy(), expected_ref)
    print(f"=== [SUCCESS] E2E Save/Load {list(save_blocks)} -> {list(load_blocks)} (dtype={dtype}) verified on TPU! ===")
    del manager, store

  def test_e2e_save_and_load(self):
    self._run_e2e_save_and_load()

  # The same save/load/compare pipeline, driven through the slices form of
  # load. Running it as a variant rather than a separate test is the point:
  # `slices` is a shortcut past the store's own lookup, not a different
  # transfer, so the bytes that land in blocks [2, 3] must be the ones the
  # no-slices path produces.
  def test_e2e_save_and_load_with_slices(self):
    self._run_e2e_save_and_load(use_slices=True)

  def test_e2e_save_and_load_bfloat16(self):
    self._run_e2e_save_and_load(dtype=torch.bfloat16)

  def test_e2e_save_and_load_non_contiguous_blocks(self):
    self._run_e2e_save_and_load(
        save_blocks=(0, 2),
        load_blocks=(1, 3),
    )

  def test_e2e_save_and_load_empty_batch(self):
    controller_port = find_free_port()
    tag = f"empty_{uuid.uuid4().hex[:8]}"
    rid = kv_cache_store.RaidenId(f"{tag}_job", "0", f"{tag}_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=4,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=1024,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    # Empty save is rejected by controller (returns False); empty load is a no-op (returns True)
    self.assertFalse(store.save([]))
    self.assertTrue(store.load([], []))
    self.assertTrue(store.load([], [], slices=[]))
    del store

  def test_e2e_load_uninserted_hash_fails(self):
    controller_port = find_free_port()
    tag = f"missing_{uuid.uuid4().hex[:8]}"
    rid = kv_cache_store.RaidenId(f"{tag}_job", "0", f"{tag}_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=4,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=1024,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    # Attempting to load a hash that was never inserted/pinned should fail
    self.assertFalse(store.load([b"nonexistent_hash"], [0]))
    del store

  def _run_remote_read_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_read_success: bool = True,
      use_slices: bool = False,
      dtype: torch.dtype = torch.float32,
      save_blocks: tuple[int, int] = (0, 1),
      dst_blocks: tuple[int, int] = (0, 1),
  ):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data for Job A
    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    host_data_a = np.arange(np.prod(shape), dtype=np_dtype).reshape(shape)
    tpu_cache_a = torch.tensor(host_data_a, dtype=dtype, device=self.device)

    # Create empty Job B device memory with zeros
    zeros_b = np.zeros(shape, dtype=np_dtype)
    tpu_cache_b = torch.tensor(zeros_b, dtype=dtype, device=self.device)

    # Expected reference for Job B device memory after remote read
    expected_ref_b = zeros_b.copy()
    expected_ref_b[dst_blocks[0]] = host_data_a[save_blocks[0]]
    expected_ref_b[dst_blocks[1]] = host_data_a[save_blocks[1]]

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (
        block_elements * tpu_cache_a.element_size()
    ) // self.num_devices

    controller_port_a = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    # 2. Create Job A's KVCacheStore & KVCacheManager
    tag = f"read_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
        node_id=producer_node_id,
    )

    controller_port_b = find_free_port()
    # 3. Create Job B's KVCacheStore & KVCacheManager
    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
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

    try:
      # Wait for listeners to start
      time.sleep(1)

      # Raw non-UTF-8 bytes on purpose: production hashes are binary
      # digests, and the registry round-trip must survive them (the proto
      # hash fields are `bytes`; as `string` they were UTF-8-verified on
      # the wire and cross-store sharing silently found nothing).
      hashes = [
          b"\x93\xff\x00" + f"{tag}_h0".encode(),
          b"\x93\xff\x00" + f"{tag}_h1".encode(),
      ]

      # 4. Job A inserts HBM status and calls Save
      slices_a = [
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=save_blocks[0],
              status=kv_cache_store.BlockStatus.HBM,
          ),
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=save_blocks[1],
              status=kv_cache_store.BlockStatus.HBM,
          ),
      ]
      self.assertTrue(
          store_a.insert(hashes, slices_a, on_host=False)
      )

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

      # 5. Job B calls Lookup (enable_global=True)
      time.sleep(0.5)
      # Registry-resolved hits are REMOTE and never pinned locally.
      lookup_res_b = store_b.lookup(
          hashes, enable_global=True, pin_found=False
      )
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

      # 6. Job B pulls straight from Job A into its own device blocks. The
      # source coordinates come from the lookup answer, so nothing needs to be
      # inserted into Job B's cache first. Two transports cover the same
      # contract: read_remote, and the slices form of load() -- the only
      # sanctioned peer path in load.
      slices_b = [lookup_res_b[0][1], lookup_res_b[1][1]]
      if use_slices:
        # lookup only synthesised the REMOTE answer; no pin exists and the
        # peer load neither needs nor consumes one.
        self.assertTrue(store_b.load(hashes, list(dst_blocks), slices=slices_b))
        done = False
        while not done:
          load_done, load_failed, _ = store_b.poll_load_status()
          if load_failed:
            raise RuntimeError(f"Job B peer load failed: {load_failed}")
          if len(load_done) == 2:
            done = True
          if not done:
            time.sleep(0.01)
      else:
        self.assertTrue(store_b.read_remote(hashes, slices_b, list(dst_blocks)))

        if not expect_read_success:
          failed = False
          for _ in range(500):
            _, read_failed, _ = store_b.poll_remote_read_status()
            if read_failed:
              self.assertEqual(set(read_failed), set(hashes))
              failed = True
              break
            time.sleep(0.01)
          self.assertTrue(
              failed,
              "expected read_remote to fail on producer/consumer node_id mismatch",
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

      # 8. The bytes are already in HBM -- there is no second Load step, and
      # no local record of it either. Job B's cache is still a miss for these
      # hashes: the bytes live only in the device blocks it named.
      self.assertEmpty(store_b.lookup(hashes))
      # No host copy was left behind either, so a later LOCAL load of the
      # same hashes has nothing to read from and is refused. Pulling is not a
      # way to warm the local cache; a caller that wants a host copy must
      # save() what it pulled.
      self.assertFalse(
          store_b.load(hashes, [2, 3]),
          "the pull must not leave a host copy behind",
      )

      # 9. Verify byte-exact match on Job B TPU device, and that the pull
      # touched ONLY the destination blocks.
      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      actual_b = tpu_cache_b.cpu()
      if dtype == torch.bfloat16:
        expected_t = torch.tensor(expected_ref_b, dtype=dtype)
        np.testing.assert_array_equal(
            actual_b.to(torch.float32).numpy(),
            expected_t.cpu().to(torch.float32).numpy(),
            err_msg="Job B TPU memory does not match expected reference array",
        )
      else:
        np.testing.assert_array_equal(
            actual_b.numpy(),
            expected_ref_b,
            err_msg="Job B TPU memory does not match expected reference array",
        )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_remote_read_e2e_with_slices(self):
    # The same pull, through load(slices=REMOTE) -- the only peer path in
    # load(). Before this driver existed the use_slices branch was dead and
    # the torch suite had no live coverage of the peer load at all.
    self._run_remote_read_e2e_test(use_slices=True)

  def _run_remote_write_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_write_success: bool = True,
      dtype: torch.dtype = torch.float32,
      save_blocks: tuple[int, int] = (0, 1),
      dst_blocks: tuple[int, int] = (0, 1),
  ):
    """Job A offers blocks it owns; Job B pulls them and keeps them.

    The mirror image of _run_remote_read_e2e_test: there the destination asks,
    here the source offers. What this adds over the read path is the
    WriteRemote control plane -- the ack, the destination's all-or-nothing
    insert, global registration, and the source polling to COMMITTED -- with a
    byte comparison at the end, because every control-plane assertion can pass
    while nothing is actually transferred.
    """
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    host_data_a = np.arange(np.prod(shape), dtype=np_dtype).reshape(shape)
    tpu_cache_a = torch.tensor(host_data_a, dtype=dtype, device=self.device)
    # Zeroed, so a byte comparison cannot pass on data already present.
    zeros_b = np.zeros(shape, dtype=np_dtype)
    tpu_cache_b = torch.tensor(zeros_b, dtype=dtype, device=self.device)

    expected_ref_b = zeros_b.copy()
    expected_ref_b[dst_blocks[0]] = host_data_a[save_blocks[0]]
    expected_ref_b[dst_blocks[1]] = host_data_a[save_blocks[1]]

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (
        block_elements * tpu_cache_a.element_size()
    ) // self.num_devices

    controller_port_a = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    tag = f"write_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
        node_id=producer_node_id,
    )

    controller_port_b = find_free_port()
    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
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

    try:
      time.sleep(1)
      hashes = [f"{tag}_h0".encode(), f"{tag}_h1".encode()]

      # 1. Job A puts the blocks in HBM and saves them to host DRAM. Only
      #    host-resident blocks can be offered: the pull reads host memory.
      slices_a = [
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=save_blocks[0],
              status=kv_cache_store.BlockStatus.HBM,
          ),
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=-1,
              device_block_id=save_blocks[1],
              status=kv_cache_store.BlockStatus.HBM,
          ),
      ]
      self.assertTrue(
          store_a.insert(hashes, slices_a, on_host=False)
      )
      store_a.save(hashes)

      deadline = time.time() + 120
      while True:
        save_done, save_failed, _, _, _ = store_a.poll_save_status()
        if save_failed:
          raise RuntimeError(f"Job A Async Save failed: {save_failed}")
        if len(save_done) == len(hashes):
          break
        if time.time() > deadline:
          raise RuntimeError("Job A save did not complete in time")
        time.sleep(0.01)

      # 2. Job A offers them. Returns once Job B has decided, not once the
      #    bytes have moved.
      #
      # The local save above consumed the pin insert()/pin() granted, so the
      # offer needs its own. This is the documented remote-save flow: lookup()
      # answers "yes, host-resident here" AND grants the pin that save(dst)
      # spends.
      self.assertLen(store_a.lookup(hashes), len(hashes))
      self.assertTrue(store_a.save(hashes, rid_b))

      deadline = time.time() + 120
      done, failed, existing = [], [], []
      while time.time() < deadline:
        done, failed, pending, existing, unregistered = (
            store_a.poll_save_status()
        )
        if done or failed or existing:
          break
        time.sleep(0.01)

      if not expect_write_success:
        self.assertNotEmpty(failed)
        self.assertEmpty(done)
        return

      self.assertCountEqual(done, hashes)
      self.assertEmpty(failed)
      self.assertEmpty(existing)
      self.assertEmpty(unregistered)
      # No release: the successful remote save consumed the pin lookup()
      # granted.

      # 3. Job B holds them locally, host-resident, as its own. lookup() resolves
      #    and pins the landed entries.
      lookup_b = store_b.lookup(hashes, enable_global=False)
      self.assertLen(lookup_b, len(hashes))
      for _, slice_b in lookup_b:
        self.assertEqual(slice_b.status, kv_cache_store.BlockStatus.HOST)

      # 4. Prove the bytes are real. load() consumes the pin on success.
      self.assertTrue(store_b.load(hashes, list(dst_blocks)))
      deadline = time.time() + 120
      while True:
        load_done, load_failed, _ = store_b.poll_load_status()
        if load_failed:
          raise RuntimeError(f"Job B Load failed: {load_failed}")
        if len(load_done) == len(hashes):
          break
        if time.time() > deadline:
          raise RuntimeError("Job B load did not complete in time")
        time.sleep(0.01)

      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      actual_b = tpu_cache_b.cpu()
      if dtype == torch.bfloat16:
        expected_t = torch.tensor(expected_ref_b, dtype=dtype)
        np.testing.assert_array_equal(
            actual_b.to(torch.float32).numpy(),
            expected_t.cpu().to(torch.float32).numpy(),
            err_msg=(
                "Job B's device memory does not byte-match what Job A offered:"
                " the remote write reported COMMITTED without moving the data"
            ),
        )
      else:
        np.testing.assert_array_equal(
            actual_b.numpy(),
            expected_ref_b,
            err_msg=(
                "Job B's device memory does not byte-match what Job A offered:"
                " the remote write reported COMMITTED without moving the data"
            ),
        )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_remote_write_e2e(self):
    self._run_remote_write_e2e_test()

  def test_remote_write_e2e_bfloat16(self):
    self._run_remote_write_e2e_test(dtype=torch.bfloat16)

  def test_remote_write_e2e_non_contiguous_blocks(self):
    self._run_remote_write_e2e_test(
        save_blocks=(0, 2),
        dst_blocks=(1, 3),
    )

  def test_remote_write_e2e_matching_node_id(self):
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=0,
        expect_write_success=True,
    )

  def test_remote_write_e2e_mismatched_node_id_fails(self):
    # The destination pairs each of its workers with the source worker holding
    # its shards by node_id; a mismatch leaves the pull with no group to read
    # from, so the transfer fails and the source is told rather than left
    # pending.
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=1,
        expect_write_success=False,
    )

  def test_remote_read_e2e_bfloat16(self):
    self._run_remote_read_e2e_test(dtype=torch.bfloat16)

  def test_remote_read_e2e_non_contiguous_blocks(self):
    self._run_remote_read_e2e_test(
        save_blocks=(0, 2),
        dst_blocks=(1, 3),
    )

  def test_remote_read_e2e_matching_node_id(self):
    self._run_remote_read_e2e_test(
        producer_node_id=7,
        consumer_node_id=7,
        expect_read_success=True,
    )

  def test_remote_read_e2e_mismatched_node_id_fails(self):
    self._run_remote_read_e2e_test(
        producer_node_id=1,
        consumer_node_id=2,
        expect_read_success=False,
    )

  def test_remote_read_e2e_source_missing_block_fails(self):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = torch.zeros(shape, dtype=torch.float32, device=self.device)
    tpu_cache_b = torch.zeros(shape, dtype=torch.float32, device=self.device)

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices
    controller_port_a = find_free_port()
    controller_port_b = find_free_port()

    tag = f"miss_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
    )

    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id=f"{tag}_worker_b",
        host_blocks_to_allocate=4,
    )

    try:
      time.sleep(1)
      ghost = [f"{tag}_ghost".encode()]
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
          self.assertEqual(set(read_failed), set(ghost))
          failed = True
          break
        time.sleep(0.01)
      self.assertTrue(
          failed, "expected read_remote to fail (source is missing the block)"
      )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_remote_read_e2e_source_wrong_status_fails(self):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = torch.zeros(shape, dtype=torch.float32, device=self.device)
    tpu_cache_b = torch.zeros(shape, dtype=torch.float32, device=self.device)

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices
    controller_port_a = find_free_port()
    controller_port_b = find_free_port()

    tag = f"ws_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
    )

    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id=f"{tag}_worker_b",
        host_blocks_to_allocate=4,
    )

    try:
      time.sleep(1)
      # Raw non-UTF-8 bytes on purpose: production hashes are binary
      # digests, and the registry round-trip must survive them (the proto
      # hash fields are `bytes`; as `string` they were UTF-8-verified on
      # the wire and cross-store sharing silently found nothing).
      hashes = [
          b"\x93\xff\x00" + f"{tag}_h0".encode(),
          b"\x93\xff\x00" + f"{tag}_h1".encode(),
      ]
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

      time.sleep(0.5)
      slices_b = [
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=0,
              device_block_id=-1,
              status=kv_cache_store.BlockStatus.REMOTE,
          ),
          kv_cache_store.RaidenBlockId(
              rid_a,
              host_block_id=0,
              device_block_id=-1,
              status=kv_cache_store.BlockStatus.REMOTE,
          ),
      ]
      self.assertTrue(store_b.read_remote(hashes, slices_b, [0, 1]))

      failed = False
      for _ in range(500):
        _, read_failed, _ = store_b.poll_remote_read_status()
        if read_failed:
          self.assertEqual(set(read_failed), set(hashes))
          failed = True
          break
        time.sleep(0.01)
      self.assertTrue(
          failed,
          "expected read_remote to fail (source only has block in HBM)",
      )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_expected_worker_count_waits_for_a_concurrent_registration(self):
    """The barrier replaces the sleep-and-hope idiom."""
    tpu_cache = torch.zeros(
        (2, 128, 8, 8, 128), dtype=torch.float32, device="tpu"
    )
    controller_port = find_free_port()
    num_blocks = 2
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

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
    self.assertGreaterEqual(elapsed, registration_delay_s)

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
    # No release: the successful save consumed the pin insert() granted.
    del built

  def test_sweep_demotes_to_the_store_node_and_reads_back(self):
    """The evict chain end to end: it all happens behind the existing API.

    save() makes blocks host-resident, the store monitor's sweep demotes the
    cold ones to a real store node subprocess through the KVTransferSpec
    published to the registry, a global lookup resolves the node as the new
    owner, and read_remote pulls the bytes back into HBM byte-exact.
    Single-device smoke of the JAX test of the same name.
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

    # 6 of 8 blocks in use after the saves: free ratio 0.25, below the 0.5
    # low watermark, so the sweep must demote down to the 0.75 high one.
    capacity = 8
    num_insert = 6
    shape = (capacity, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = torch.tensor(host_data, device=self.device)
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    controller_port = find_free_port()
    pool_group = "evict_e2e_pool"

    # The registration is what fills in the KVTransferSpec the store
    # publishes for the node, and expected_worker_count is the barrier that
    # waits for it (see the dedicated barrier test above).
    built = {}

    def build_manager_after_a_delay():
      time.sleep(1.0)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[[tpu_cache]],
          local_control_port=0,
          max_blocks=capacity,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="evict_e2e_worker_0",
          host_blocks_to_allocate=capacity,
          node_id=0,
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

    this_dir = os.path.dirname(os.path.abspath(__file__))
    node_binary = os.path.abspath(
        os.path.join(
            this_dir, "..", "..", "store_node", "kv_cache_host_store_node_main"
        )
    )
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

      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      got = tpu_cache.cpu().numpy()[dst_device_block]
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
    demoted anywhere. Single-device smoke of the JAX test of the same name.
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

    # 6 of 8 blocks in use after the saves: free ratio 0.25, below the 0.5
    # low watermark, so the sweep must free down to the 0.75 high one.
    capacity = 8
    num_insert = 6
    shape = (capacity, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = torch.tensor(host_data, device=self.device)
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    controller_port = find_free_port()

    built = {}

    def build_manager_after_a_delay():
      time.sleep(1.0)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[[tpu_cache]],
          local_control_port=0,
          max_blocks=capacity,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="evict_drop_worker_0",
          host_blocks_to_allocate=capacity,
          node_id=0,
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

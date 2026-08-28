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

resources = None
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

  def _wait_for_save(
      self, store: kv_cache_store.KVCacheStore, timeout_s: float = 5.0
  ) -> list[bytes]:
    """Polls until save finishes, asserting no failure or unexpected remote states."""
    start = time.time()
    while time.time() - start < timeout_s:
      save_done, save_failed, _, save_existing, save_unregistered = (
          store.poll_save_status()
      )
      if save_failed:
        raise RuntimeError(f"Async Save failed: {save_failed}")
      self.assertEmpty(save_existing)
      self.assertEmpty(save_unregistered)
      if save_done:
        return save_done
      time.sleep(0.01)
    raise TimeoutError("Timed out waiting for save to complete")

  def _wait_for_save_verdict(
      self, store: kv_cache_store.KVCacheStore, timeout_s: float = 5.0
  ) -> tuple[list[bytes], list[bytes], list[bytes], list[bytes]]:
    """Polls save status until settled, returning (done, failed, existing, unregistered)."""
    start = time.time()
    while time.time() - start < timeout_s:
      done, failed, pending, existing, unregistered = store.poll_save_status()
      if done or failed:
        return done, failed, existing, unregistered
      if not pending:
        raise RuntimeError("Save vanished without a verdict")
      time.sleep(0.01)
    raise TimeoutError(
        f"Timed out waiting for save to settle within {timeout_s}s"
    )

  def _wait_for_load(
      self, store: kv_cache_store.KVCacheStore, timeout_s: float = 5.0
  ) -> list[bytes]:
    """Polls until load finishes, asserting no failures."""
    start = time.time()
    while time.time() - start < timeout_s:
      load_done, load_failed, _ = store.poll_load_status()
      if load_failed:
        raise RuntimeError(f"Async Load failed: {load_failed}")
      if load_done:
        return load_done
      time.sleep(0.01)
    raise TimeoutError("Timed out waiting for load to complete")

  def _wait_for_remote_read(
      self,
      store: kv_cache_store.KVCacheStore,
      expected_count: int = 2,
      timeout_s: float = 5.0,
  ) -> list[bytes]:
    """Polls until remote read finishes, asserting no failures."""
    start = time.time()
    while time.time() - start < timeout_s:
      read_done, read_failed, _ = store.poll_remote_read_status()
      if read_failed:
        raise RuntimeError(f"Job B ReadRemote failed: {read_failed}")
      if len(read_done) == expected_count:
        return read_done
      time.sleep(0.01)
    raise TimeoutError("Timed out waiting for remote read to complete")

  def _wait_for_remote_read_failure(
      self, store: kv_cache_store.KVCacheStore, timeout_s: float = 5.0
  ) -> list[bytes]:
    """Polls until remote read reports a failure."""
    start = time.time()
    while time.time() - start < timeout_s:
      _, read_failed, _ = store.poll_remote_read_status()
      if read_failed:
        return read_failed
      time.sleep(0.01)
    raise TimeoutError("Timed out waiting for remote read failure")

  def _create_node(
      self,
      tag: str,
      job_name: str,
      tpu_cache: torch.Tensor,
      node_id: int = 0,
      enable_global_registry: bool = False,
      num_blocks: int = 4,
      expected_worker_count: int = 0,
      kv_pool_group: str = "",
  ) -> tuple[
      kv_cache_store.KVCacheStore,
      kv_cache_manager.KVCacheManager,
      kv_cache_store.RaidenId,
  ]:
    """Creates and binds a paired KVCacheStore and KVCacheManager."""
    controller_port = find_free_port()
    worker_port = find_free_port()
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    rid = kv_cache_store.RaidenId(
        f"{tag}_{job_name}", "0", f"{tag}_cache_{job_name}", 0
    )

    if expected_worker_count > 0:
      built = {}
      thread_error = None

      def build_manager():
        nonlocal thread_error
        try:
          built["manager"] = kv_cache_manager.KVCacheManager(
              kv_caches=[[tpu_cache]],
              local_control_port=0,
              max_blocks=num_blocks,
              num_slots=2,
              unsafe_skip_buffer_lock=self.skip_lock,
              raiden_worker_port=worker_port,
              raiden_controller_address=f"localhost:{controller_port}",
              worker_id=f"{tag}_worker_{job_name}",
              host_blocks_to_allocate=num_blocks,
              node_id=node_id,
          )
        except Exception as e:
          thread_error = e

      worker_thread = threading.Thread(target=build_manager)
      worker_thread.start()
      store = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=(
              f"localhost:{_registry_port}" if enable_global_registry else ""
          ),
          raiden_id=rid,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port,
          expected_worker_count=expected_worker_count,
          kv_pool_group=kv_pool_group,
      )
      worker_thread.join(timeout=30)
      if worker_thread.is_alive():
        raise RuntimeError(
            "worker_thread timed out initializing KVCacheManager"
        )
      if thread_error is not None:
        raise RuntimeError(
            f"worker_thread failed: {thread_error}"
        ) from thread_error
      manager = built["manager"]
    else:
      store = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          global_registry_address=(
              f"localhost:{_registry_port}" if enable_global_registry else ""
          ),
          raiden_id=rid,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port,
          kv_pool_group=kv_pool_group,
      )
      manager = kv_cache_manager.KVCacheManager(
          kv_caches=[[tpu_cache]],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=worker_port,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id=f"{tag}_worker_{job_name}",
          host_blocks_to_allocate=num_blocks,
          node_id=node_id,
      )
    return store, manager, rid

  def _insert_hbm_blocks(
      self,
      store: kv_cache_store.KVCacheStore,
      rid: kv_cache_store.RaidenId,
      hashes: list[bytes],
      device_blocks: list[int],
  ):
    """Registers HBM block descriptors in the store."""
    slices = [
        kv_cache_store.RaidenBlockId(
            rid,
            host_block_id=-1,
            device_block_id=d,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for d in device_blocks
    ]
    self.assertTrue(store.insert(hashes, slices, on_host=False))

  def _run_e2e_save_and_load(self, use_slices: bool):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = torch.tensor(host_data, device=self.device)

    # Expected reference after loading saved blocks 0 and 1 into blocks 2 and 3: [a, b, a, b]
    expected_ref = host_data.copy()
    expected_ref[2] = host_data[0]
    expected_ref[3] = host_data[1]

    # 2. Setup Node (Store & Manager)
    tag = f"save_{uuid.uuid4().hex[:8]}"
    store, manager, rid = self._create_node(
        tag=tag,
        job_name="job",
        tpu_cache=tpu_cache,
        enable_global_registry=False,
    )

    # 3. Insert initial HBM blocks to KVCacheStore
    hashes = [b"hash_0", b"hash_1"]
    self._insert_hbm_blocks(store, rid, hashes, device_blocks=[0, 1])

    # Verify initial status in store is HBM
    lookup_res = store.lookup(hashes, pin_found=False)
    self.assertEqual([h for h, _ in lookup_res], hashes)
    self.assertEqual([b.device_block_id for _, b in lookup_res], [0, 1])
    for _, b in lookup_res:
      self.assertEqual(b.status, kv_cache_store.BlockStatus.HBM)

    # 4. Save HBM blocks to Host DRAM
    store.save(hashes)
    self._wait_for_save(store)

    # Verify status in store is updated to HOST_AND_HBM with assigned host_block_ids
    lookup_res = store.lookup(hashes, pin_found=False)
    self.assertEqual([h for h, _ in lookup_res], hashes)
    self.assertEqual([b.host_block_id for _, b in lookup_res], [0, 1])
    for _, b in lookup_res:
      self.assertEqual(b.status, kv_cache_store.BlockStatus.HOST_AND_HBM)

    # 5. Load from Host DRAM into TPU HBM blocks [2, 3]
    if use_slices:
      load_slices = [entry for _, entry in store.lookup(hashes)]
      self.assertEqual(
          [s.status for s in load_slices],
          [kv_cache_store.BlockStatus.HOST_AND_HBM] * 2,
      )
      self.assertTrue(store.load(hashes, [2, 3], slices=load_slices))
    else:
      self.assertLen(store.lookup(hashes), 2)
      self.assertTrue(store.load(hashes, [2, 3]))

    self._wait_for_load(store)

    # 6. Verify restored TPU device memory matches expected array [a, b, a, b]
    try:
      torch.tpu.synchronize()
    except (AttributeError, RuntimeError):
      pass

    np.testing.assert_array_equal(tpu_cache.cpu().numpy(), expected_ref)
    del manager, store

  @parameterized.named_parameters(
      ("direct", False),
      ("with_slices", True),
  )
  def test_e2e_save_and_load(self, use_slices: bool):
    self._run_e2e_save_and_load(use_slices=use_slices)

  def _run_remote_read_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_read_success: bool = True,
      use_slices: bool = False,
  ):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    src_device_blocks = [0, 2]
    dst_device_blocks = [1, 3]
    sentinel_blocks = [0, 2]

    # 1. Generate sequential distinct cache data for Job A
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = torch.tensor(host_data_a, device=self.device)

    # Create empty Job B device memory with zeros
    zeros_b = np.zeros(shape, dtype=np.float32)
    tpu_cache_b = torch.tensor(zeros_b, device=self.device)

    # 2. Create Job A & Job B nodes
    tag = f"read_{uuid.uuid4().hex[:8]}"
    store_a, manager_a, rid_a = self._create_node(
        tag=tag,
        job_name="job_a",
        tpu_cache=tpu_cache_a,
        node_id=producer_node_id,
        enable_global_registry=True,
    )
    store_b, manager_b, _ = self._create_node(
        tag=tag,
        job_name="job_b",
        tpu_cache=tpu_cache_b,
        node_id=consumer_node_id,
        enable_global_registry=True,
    )

    try:
      # Wait for listeners to start
      time.sleep(1)

      hashes = [
          b"\x93\xff\x00" + f"{tag}_h0".encode(),
          b"\x93\xff\x00" + f"{tag}_h1".encode(),
      ]

      # 3. Job A inserts HBM status and calls Save
      self._insert_hbm_blocks(
          store_a, rid_a, hashes, device_blocks=src_device_blocks
      )
      self.assertTrue(store_a.save(hashes))
      done = self._wait_for_save(store_a)
      self.assertCountEqual(done, hashes)

      # 4. Job B calls Lookup (enable_global=True)
      time.sleep(0.5)
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

      # 5. Job B pulls straight from Job A into its own device blocks.
      slices_b = [lookup_res_b[0][1], lookup_res_b[1][1]]
      if use_slices:
        self.assertTrue(
            store_b.load(hashes, dst_device_blocks, slices=slices_b)
        )
        self._wait_for_load(store_b)
      else:
        self.assertTrue(
            store_b.read_remote(hashes, slices_b, dst_device_blocks)
        )

        if not expect_read_success:
          failed = self._wait_for_remote_read_failure(store_b)
          self.assertEqual(set(failed), set(hashes))
          return

        self._wait_for_remote_read(store_b)

      # 8. The bytes are already in HBM
      self.assertEmpty(store_b.lookup(hashes))
      self.assertFalse(
          store_b.load(hashes, sentinel_blocks),
          "the pull must not leave a host copy behind",
      )

      # 9. Verify byte-exact match on Job B TPU device, sentinels stay as created.
      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      actual_b = tpu_cache_b.cpu().numpy()
      for src_blk, dst_blk in zip(src_device_blocks, dst_device_blocks):
        np.testing.assert_array_equal(
            actual_b[dst_blk],
            host_data_a[src_blk],
            err_msg=(
                f"device block {dst_blk} does not match source block {src_blk}"
            ),
        )
      for blk in sentinel_blocks:
        np.testing.assert_array_equal(
            actual_b[blk],
            zeros_b[blk],
            err_msg=f"sentinel device block {blk} was clobbered by the pull",
        )
    finally:
      del manager_a, manager_b, store_a, store_b

  @parameterized.named_parameters(
      ("direct", False, 0, 0, True),
      ("with_slices", True, 0, 0, True),
      ("matching_node_id", False, 7, 7, True),
      ("mismatched_node_id_fails", False, 1, 2, False),
  )
  def test_remote_read_e2e(
      self,
      use_slices: bool,
      producer_node_id: int,
      consumer_node_id: int,
      expect_read_success: bool,
  ):
    self._run_remote_read_e2e_test(
        producer_node_id=producer_node_id,
        consumer_node_id=consumer_node_id,
        expect_read_success=expect_read_success,
        use_slices=use_slices,
    )

  def _run_remote_read_to_hbm_test(self, use_slices: bool = False):
    """Direct peer-to-HBM load bypassing local host DRAM allocation."""
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    src_device_blocks = [0, 2]
    dst_device_blocks = [1, 3]
    sentinel_blocks = [0, 2]

    # Generate random payloads
    rng_a = np.random.default_rng(20260728)
    host_data_a = rng_a.standard_normal(shape).astype(np.float32)
    tpu_cache_a = torch.tensor(host_data_a, device=self.device)

    # Job B starts with distinct random sentinel data
    rng_b = np.random.default_rng(31415926)
    host_data_b = rng_b.standard_normal(shape).astype(np.float32)
    tpu_cache_b = torch.tensor(host_data_b, device=self.device)

    tag = f"read_hbm_{uuid.uuid4().hex[:8]}"
    store_a, manager_a, rid_a = self._create_node(
        tag=tag,
        job_name="job_a",
        tpu_cache=tpu_cache_a,
        enable_global_registry=True,
    )
    store_b, manager_b, _ = self._create_node(
        tag=tag,
        job_name="job_b",
        tpu_cache=tpu_cache_b,
        enable_global_registry=True,
    )

    try:
      time.sleep(1)
      hashes = [f"{tag}_h0".encode(), f"{tag}_h1".encode()]

      # Job A saves to Host DRAM so blocks are leasable
      self._insert_hbm_blocks(
          store_a, rid_a, hashes, device_blocks=src_device_blocks
      )
      self.assertTrue(store_a.save(hashes))
      done = self._wait_for_save(store_a)
      self.assertCountEqual(done, hashes)

      # Job B discovers blocks as REMOTE
      time.sleep(0.5)
      lookup_b = store_b.lookup(hashes, enable_global=True, pin_found=False)
      self.assertLen(lookup_b, len(hashes))
      for _, blk in lookup_b:
        self.assertEqual(blk.status, kv_cache_store.BlockStatus.REMOTE)
        self.assertEqual(blk.raiden_id, rid_a)

      slices_b = [b for _, b in lookup_b]
      if use_slices:
        # One-step peer-fetch load straight into TPU HBM
        self.assertTrue(
            store_b.load(hashes, dst_device_blocks, slices=slices_b)
        )
        self._wait_for_load(store_b)
      else:
        # Pull straight into HBM via read_remote
        self.assertTrue(
            store_b.read_remote(hashes, slices_b, dst_device_blocks)
        )
        self._wait_for_remote_read(store_b)

      # Remote read to HBM leaves NO local host copy behind
      self.assertEmpty(store_b.lookup(hashes))

      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      actual_b = tpu_cache_b.cpu().numpy()
      for src_blk, dst_blk in zip(src_device_blocks, dst_device_blocks):
        np.testing.assert_array_equal(
            actual_b[dst_blk],
            host_data_a[src_blk],
            err_msg=(
                f"device block {dst_blk} does not match source block {src_blk}"
            ),
        )
      for blk in sentinel_blocks:
        np.testing.assert_array_equal(
            actual_b[blk],
            host_data_b[blk],
            err_msg=f"sentinel device block {blk} was clobbered by the pull",
        )

      if not use_slices:
        self.assertFalse(
            store_b.load(hashes, sentinel_blocks),
            "read_remote must not leave a host copy behind",
        )
    finally:
      del manager_a, manager_b, store_a, store_b

  @parameterized.named_parameters(
      ("direct", False),
      ("with_slices", True),
  )
  def test_remote_read_to_hbm(self, use_slices: bool):
    self._run_remote_read_to_hbm_test(use_slices=use_slices)

  def _run_remote_write_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_write_success: bool = True,
      preload_count: int = 0,
  ):
    """Job A offers blocks it owns; Job B pulls them and keeps them."""
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    src_device_blocks = [0, 2]
    dst_device_blocks = [1, 3]

    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = torch.tensor(host_data_a, device=self.device)
    # Zeroed, so a byte comparison cannot pass on data already present.
    zeros_b = np.zeros(shape, dtype=np.float32)
    tpu_cache_b = torch.tensor(zeros_b, device=self.device)

    tag = f"write_{uuid.uuid4().hex[:8]}"
    store_a, manager_a, rid_a = self._create_node(
        tag=tag,
        job_name="job_a",
        tpu_cache=tpu_cache_a,
        node_id=producer_node_id,
        enable_global_registry=True,
    )
    store_b, manager_b, rid_b = self._create_node(
        tag=tag,
        job_name="job_b",
        tpu_cache=tpu_cache_b,
        node_id=consumer_node_id,
        enable_global_registry=True,
    )

    try:
      time.sleep(1)
      hashes = [f"{tag}_h0".encode(), f"{tag}_h1".encode()]

      # 1. Job A puts the blocks in HBM and saves them to host DRAM. Only
      #    host-resident blocks can be offered: the pull reads host memory.
      self._insert_hbm_blocks(
          store_a, rid_a, hashes, device_blocks=src_device_blocks
      )
      self.assertTrue(store_a.save(hashes))
      done = self._wait_for_save(store_a)
      self.assertCountEqual(done, hashes)

      preloaded = hashes[:preload_count]
      if preloaded:
        # Pre-seed Job B directly into host memory
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
        store_b.release(preloaded)

      # 2. Job A offers them.
      self.assertLen(store_a.lookup(hashes), len(hashes))
      self.assertTrue(store_a.save(hashes, rid_b))

      done, failed, existing, unregistered = self._wait_for_save_verdict(
          store_a
      )

      if 0 < preload_count < len(hashes):
        self.assertCountEqual(failed, hashes)
        self.assertEmpty(done)
        self.assertCountEqual(existing, preloaded)
        self.assertEmpty(unregistered)
        store_a.release(hashes)
        return

      if not expect_write_success:
        self.assertNotEmpty(failed)
        self.assertEmpty(done)
        return

      self.assertCountEqual(done, hashes)
      self.assertEmpty(failed)
      self.assertEmpty(existing)
      self.assertEmpty(unregistered)

      # 3. Job B holds them locally, host-resident, as its own.
      lookup_b = store_b.lookup(hashes, enable_global=False, pin_found=False)
      self.assertLen(lookup_b, len(hashes))
      for _, slice_b in lookup_b:
        self.assertEqual(slice_b.status, kv_cache_store.BlockStatus.HOST)

      if preloaded:
        return

      # 4. Prove the bytes are real. load() consumes the pin on success.
      self.assertLen(store_b.lookup(hashes), len(hashes))
      self.assertTrue(store_b.load(hashes, dst_device_blocks))
      self._wait_for_load(store_b)

      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      actual_b = tpu_cache_b.cpu().numpy()
      for src_blk, dst_blk in zip(src_device_blocks, dst_device_blocks):
        np.testing.assert_array_equal(
            actual_b[dst_blk],
            host_data_a[src_blk],
            err_msg=(
                f"device block {dst_blk} does not match source block {src_blk}"
            ),
        )
    finally:
      del manager_a, manager_b, store_a, store_b

  @parameterized.named_parameters(
      ("matching_node_id", 0, 0, True, 0),
      ("mismatched_node_id_fails", 0, 1, False, 0),
      ("all_exist_moves_nothing", 0, 0, True, 2),
      ("partial_exist_reports_the_overlap", 0, 0, False, 1),
  )
  def test_remote_write_e2e(
      self,
      producer_node_id: int,
      consumer_node_id: int,
      expect_write_success: bool,
      preload_count: int,
  ):
    self._run_remote_write_e2e_test(
        producer_node_id=producer_node_id,
        consumer_node_id=consumer_node_id,
        expect_write_success=expect_write_success,
        preload_count=preload_count,
    )

  def test_remote_read_e2e_source_missing_block_fails(self):
    tag = f"miss_{uuid.uuid4().hex[:8]}"
    zeros = torch.zeros(
        (4, 128, 8, 8, 128), dtype=torch.float32, device=self.device
    )
    store_a, manager_a, rid_a = self._create_node(
        tag=tag,
        job_name="job_a",
        tpu_cache=zeros,
        enable_global_registry=True,
    )
    store_b, manager_b, _ = self._create_node(
        tag=tag,
        job_name="job_b",
        tpu_cache=zeros,
        enable_global_registry=True,
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
      failed = self._wait_for_remote_read_failure(store_b)
      self.assertEqual(set(failed), set(ghost))
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_remote_read_e2e_source_wrong_status_fails(self):
    tag = f"ws_{uuid.uuid4().hex[:8]}"
    zeros = torch.zeros(
        (4, 128, 8, 8, 128), dtype=torch.float32, device=self.device
    )
    store_a, manager_a, rid_a = self._create_node(
        tag=tag,
        job_name="job_a",
        tpu_cache=zeros,
        enable_global_registry=True,
    )
    store_b, manager_b, _ = self._create_node(
        tag=tag,
        job_name="job_b",
        tpu_cache=zeros,
        enable_global_registry=True,
    )

    try:
      time.sleep(1)
      hashes = [
          b"\x93\xff\x00" + f"{tag}_h0".encode(),
          b"\x93\xff\x00" + f"{tag}_h1".encode(),
      ]
      # Source inserts as HBM, but does not save to Host DRAM.
      self._insert_hbm_blocks(store_a, rid_a, hashes, device_blocks=[0, 1])

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

      failed = self._wait_for_remote_read_failure(store_b)
      self.assertEqual(set(failed), set(hashes))
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_expected_worker_count_waits_for_a_concurrent_registration(self):
    """The barrier replaces the sleep-and-hope idiom."""
    controller_port = find_free_port()
    worker_port = find_free_port()
    num_blocks = 2
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices
    tpu_cache = torch.zeros(
        (num_blocks, 128, 8, 8, 128), dtype=torch.float32, device=self.device
    )

    registration_delay_s = 2.0
    built = {}

    def build_manager_after_a_delay():
      time.sleep(registration_delay_s)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[[tpu_cache]],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=worker_port,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="worker_0",
          host_blocks_to_allocate=num_blocks,
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

    # Verify that the store and late-registered worker function normally
    hashes = [b"barrier_hash_0", b"barrier_hash_1"]
    self._insert_hbm_blocks(store, rid, hashes, device_blocks=[0, 1])
    self.assertTrue(store.save(hashes))
    done = self._wait_for_save(store)
    self.assertCountEqual(done, hashes)
    del built, store

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

    capacity = 8
    num_insert = 6
    shape = (capacity, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = torch.tensor(host_data, device=self.device)
    block_elements = 128 * 8 * 8 * 128
    pool_group = "evict_e2e_pool"

    store, manager, rid = self._create_node(
        tag="evict_e2e",
        job_name="serving",
        tpu_cache=tpu_cache,
        enable_global_registry=True,
        num_blocks=capacity,
        expected_worker_count=1,
        kv_pool_group=pool_group,
    )

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
      self._insert_hbm_blocks(
          store, rid, hashes, device_blocks=list(range(num_insert))
      )

      # insert() pins; a successful save consumes that pin, leaving the
      # blocks host-resident and unpinned -- exactly what the sweep demotes.
      self.assertTrue(store.save(hashes))
      done = self._wait_for_save(store)
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
      read_done = self._wait_for_remote_read(store, expected_count=1)
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
      del manager, store

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

    store, manager, rid = self._create_node(
        tag="evict_drop",
        job_name="serving",
        tpu_cache=tpu_cache,
        enable_global_registry=True,
        num_blocks=capacity,
        expected_worker_count=1,
        kv_pool_group="evict_drop_pool",
    )

    try:
      hashes = [f"evict_drop_blk_{i}".encode() for i in range(num_insert)]
      self._insert_hbm_blocks(
          store, rid, hashes, device_blocks=list(range(num_insert))
      )

      # insert() pins; a successful save consumes that pin, leaving the
      # blocks host-resident and unpinned -- exactly what the sweep drops.
      self.assertTrue(store.save(hashes))
      done = self._wait_for_save(store)
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
    finally:
      del manager, store


if __name__ == "__main__":
  absltest.main()

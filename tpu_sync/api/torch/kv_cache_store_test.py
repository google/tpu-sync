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
import socket
import subprocess
import time
import unittest

from absl.testing import absltest

resources = None
from tpu_sync.api.torch import kv_cache_store


def _pick_unused_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


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


def setUpModule():
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


def tearDownModule():
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


class KVCacheStoreTest(absltest.TestCase):

  def test_raiden_block_id_creation_and_equality(self):
    id_ = kv_cache_store.RaidenId("test_job", "0", "test_cache", 0)
    block_1 = kv_cache_store.RaidenBlockId(
        id_,
        host_block_id=10,
        device_block_id=20,
        status=kv_cache_store.BlockStatus.HBM,
    )
    self.assertEqual(block_1.raiden_id.job_name, id_.job_name)
    self.assertEqual(block_1.raiden_id.job_replica_id, id_.job_replica_id)
    self.assertEqual(block_1.raiden_id.data_name, id_.data_name)
    self.assertEqual(block_1.raiden_id.data_replica_idx, id_.data_replica_idx)
    self.assertEqual(block_1.host_block_id, 10)
    self.assertEqual(block_1.device_block_id, 20)
    self.assertEqual(block_1.status, kv_cache_store.BlockStatus.HBM)

    # Test constructor with position arguments (backwards compatibility)
    block_pos = kv_cache_store.RaidenBlockId(
        id_, 10, kv_cache_store.BlockStatus.HBM
    )
    self.assertEqual(block_pos.raiden_id.job_name, id_.job_name)
    self.assertEqual(block_pos.raiden_id.job_replica_id, id_.job_replica_id)
    self.assertEqual(block_pos.raiden_id.data_name, id_.data_name)
    self.assertEqual(block_pos.raiden_id.data_replica_idx, id_.data_replica_idx)
    self.assertEqual(block_pos.host_block_id, 10)
    self.assertEqual(block_pos.device_block_id, -1)
    self.assertEqual(block_pos.status, kv_cache_store.BlockStatus.HBM)

    block_2 = kv_cache_store.RaidenBlockId(
        id_,
        host_block_id=10,
        device_block_id=20,
        status=kv_cache_store.BlockStatus.HBM,
    )
    self.assertEqual(block_1.host_block_id, block_2.host_block_id)
    self.assertEqual(block_1.device_block_id, block_2.device_block_id)
    self.assertEqual(block_1.status, block_2.status)

    block_3 = kv_cache_store.RaidenBlockId(
        id_,
        host_block_id=10,
        device_block_id=21,
        status=kv_cache_store.BlockStatus.HBM,
    )
    self.assertNotEqual(block_1.device_block_id, block_3.device_block_id)

  def test_basic_tests(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    self.assertEqual(controller.capacity(), 20)

    hashes = [b"6001", b"6002"]
    slices = [
        kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0),
        kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0),
    ]

    # 1. Insert. Re-inserting hashes that are already present succeeds:
    # insert pins what is there and inserts what is not.
    self.assertTrue(controller.insert(hashes, slices, True))
    self.assertTrue(
        controller.insert(hashes, slices, True)
    )  # Already exists
    # Two inserts granted two pins each; hand them all back -- these cases
    # only need the blocks resident.
    controller.release(hashes)
    controller.release(hashes)

    # 2. Lookup with a partial miss at the end. pin_found=False: observation
    # only, no pin taken.
    hashes_with_miss = [b"6001", b"6002", b"6003"]
    lookup_res = controller.lookup(hashes_with_miss, pin_found=False)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], b"6001")
    self.assertEqual(lookup_res[0][1].raiden_id.job_name, "inference_server")
    self.assertEqual(lookup_res[0][1].raiden_id.job_replica_id, "0")

    # Lookup with an early miss
    hashes_early_miss = [b"6001", b"6003", b"6002"]
    lookup_res_early = controller.lookup(hashes_early_miss, pin_found=False)
    self.assertLen(lookup_res_early, 1)
    self.assertEqual(lookup_res_early[0][0], b"6001")

    # 3. Re-insert once more: still succeeds, and still pins.
    self.assertTrue(controller.insert(hashes, slices, True))
    controller.release(hashes)



  def test_large_and_arbitrary_length_hashes(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=5, num_shards=1, store_server_ip="127.0.0.1"
    )

    # Test both high-bit 8-byte hash and a very long arbitrary length hash
    large_hash = b"\xff" * 8
    long_hash = b"a" * 100
    hashes = [large_hash, long_hash]
    slices = [
        kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0),
        kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0),
    ]

    self.assertTrue(controller.insert(hashes, slices, True))
    controller.release(hashes)  # insert pins; these cases only need residency

    lookup_res = controller.lookup(hashes, pin_found=False)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], large_hash)
    self.assertEqual(lookup_res[1][0], long_hash)

  def test_global_lookup_case1_local_hit(self):
    # Case 1: Full local hit, no global hit.
    # We don't need a registry server for this because it shouldn't be queried.
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    hashes = [b"local_only"]
    slices = [
        kv_cache_store.RaidenId("local_job", "0", "kv_cache", 0),
    ]
    self.assertTrue(controller.insert(hashes, slices, True))
    controller.release(hashes)  # insert pins; this case only needs residency

    res = controller.lookup(hashes, enable_global=True, pin_found=False)
    self.assertLen(res, 1)
    self.assertEqual(res[0][0], b"local_only")
    self.assertEqual(res[0][1].raiden_id.job_name, "local_job")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 0)

  def test_global_lookup_case2_and_3_mocked(self):
    # We mock _impl to simulate Case 2 and Case 3 because we don't
    # have a running registry server in Python tests.
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )

    # Create a mock for the C++ impl
    mock_impl = unittest.mock.MagicMock()
    controller._impl = mock_impl

    # Case 2: Both local and global have the same hit, but we return local.
    local_id = kv_cache_store._impl.RaidenBlockId(
        kv_cache_store.RaidenId("local_job", "0", "kv_cache", 1)
    )
    mock_impl.lookup.return_value = [(b"shared_hash", local_id)]

    res = controller.lookup([b"shared_hash"], enable_global=True)
    self.assertLen(res, 1)
    self.assertEqual(res[0][0], b"shared_hash")
    self.assertEqual(res[0][1].raiden_id.job_name, "local_job")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 1)
    mock_impl.lookup.assert_called_with([b"shared_hash"], True, True)

    # Case 3: No local hit, only global hits.
    remote_id1 = kv_cache_store._impl.RaidenBlockId(
        kv_cache_store.RaidenId("10.0.0.1:1234", "0", "kv_cache", 42)
    )
    remote_id2 = kv_cache_store._impl.RaidenBlockId(
        kv_cache_store.RaidenId("10.0.0.2:1234", "0", "kv_cache", 43)
    )
    mock_impl.lookup.return_value = [
        (b"global_1", remote_id1),
        (b"global_2", remote_id2),
    ]

    res = controller.lookup([b"global_1", b"global_2"], enable_global=True)
    self.assertLen(res, 2)
    self.assertEqual(res[0][0], b"global_1")
    self.assertEqual(res[0][1].raiden_id.job_name, "10.0.0.1:1234")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 42)
    self.assertEqual(res[1][0], b"global_2")
    self.assertEqual(res[1][1].raiden_id.job_name, "10.0.0.2:1234")
    self.assertEqual(res[1][1].raiden_id.data_replica_idx, 43)
    mock_impl.lookup.assert_called_with([b"global_1", b"global_2"], True, True)

  def test_global_lookup_error_ignored(self):
    # Construction requires a registry that is actually reachable
    # (RegisterStore failure now fails construction), so an unreachable address
    # can no longer model "registry down" at
    # construction time. Instead: start a real, throwaway registry, construct
    # against it, then kill it before the Lookup this test cares about --
    # genuinely down for that RPC, not mocked.
    throwaway_port = _pick_unused_port()
    throwaway_registry = subprocess.Popen(
        [_registry_binary_path(), f"--port={throwaway_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    try:
      controller = kv_cache_store.KVCacheStore(
          capacity=20,
          global_registry_address=f"localhost:{throwaway_port}",
          raiden_id=kv_cache_store.RaidenId(
              "lookup_error_job", "0", "kv_cache", 0
          ),
          num_shards=1,
          store_server_ip="127.0.0.1",
      )
    finally:
      throwaway_registry.terminate()
      throwaway_registry.wait()

    hashes = [b"9001"]
    # Should not fail, just return empty because the registry is now down.
    res = controller.lookup(hashes, enable_global=True)
    self.assertEmpty(res)


  def test_insert_rejects_remote_slices(self):
    # The LRU cache holds LOCAL blocks only; a REMOTE slice names a block on
    # another node. The whole batch is refused before anything is touched, so
    # not even the local member is inserted or pinned.
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    local_id = kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockId(
            local_id, 1, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockId(
            kv_cache_store.RaidenId("peer_job", "0", "kv_cache", 0),
            42,
            kv_cache_store.BlockStatus.REMOTE,
        ),
    ]
    with self.assertRaises(ValueError):
      controller.insert([b"local_h", b"remote_h"], slices, True)

    # Nothing was inserted, the local member included.
    self.assertEmpty(controller.lookup([b"local_h"], pin_found=False))
    self.assertEmpty(controller.lookup([b"remote_h"], pin_found=False))

  def test_lookup_pin_found_false_does_not_pin(self):
    # pin_found=False is the observation mode: same answer, no pin taken.
    # Pins are not directly visible from Python, so use what they gate: an
    # insert refuses when pinned entries leave it no reclaimable space.
    def make_full_store():
      store = kv_cache_store.KVCacheStore(
          capacity=2, num_shards=1, store_server_ip="127.0.0.1"
      )
      rid = kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)
      slices = [
          kv_cache_store.RaidenBlockId(
              rid, 1, kv_cache_store.BlockStatus.HOST
          ),
          kv_cache_store.RaidenBlockId(
              rid, 2, kv_cache_store.BlockStatus.HOST
          ),
      ]
      self.assertTrue(store.insert([b"h1", b"h2"], slices, True))
      store.release([b"h1", b"h2"])  # resident but unpinned
      return store, rid

    # Observation: no pin taken, so both entries stay evictable and a new
    # insert succeeds by evicting one of them.
    store, rid = make_full_store()
    res = store.lookup([b"h1", b"h2"], pin_found=False)
    self.assertLen(res, 2)
    new_slice = [
        kv_cache_store.RaidenBlockId(rid, 3, kv_cache_store.BlockStatus.HOST)
    ]
    self.assertTrue(store.insert([b"h3"], new_slice, True))

    # Control: the default lookup pins both, and the same insert is refused
    # because nothing evictable is left.
    store, rid = make_full_store()
    res = store.lookup([b"h1", b"h2"])
    self.assertLen(res, 2)
    new_slice = [
        kv_cache_store.RaidenBlockId(rid, 3, kv_cache_store.BlockStatus.HOST)
    ]
    self.assertFalse(store.insert([b"h3"], new_slice, True))
    store.release([b"h1", b"h2"])

  def test_refused_operations_retain_the_pin(self):
    # A FAILED or refused operation never spends the pin: giving up is the
    # caller's decision. Pins are not directly visible from Python, so use
    # what they gate -- at capacity 2 with both entries pinned, an insert of
    # a third hash is refused; once the pins are handed back it goes through.
    store = kv_cache_store.KVCacheStore(
        capacity=2, num_shards=1, store_server_ip="127.0.0.1"
    )
    rid = kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)
    hashes = [b"h1", b"h2"]
    slices = [
        kv_cache_store.RaidenBlockId(
            rid,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(2)
    ]
    # insert() pins; the local save below is REFUSED (no workers are
    # registered in this unit environment, and h1/h2 name device blocks no
    # manager owns), so nothing consumes them.
    self.assertTrue(store.insert(hashes, slices, False))
    if store.save(hashes):
      # Submission succeeded; the transfer itself must then fail.
      deadline = time.time() + 30
      while time.time() < deadline:
        _, failed, pending, _, _ = store.poll_save_status()
        if failed:
          break
        self.assertNotEmpty(pending)
        time.sleep(0.01)
      self.assertNotEmpty(failed)

    probe = [
        kv_cache_store.RaidenBlockId(rid, 3, kv_cache_store.BlockStatus.HOST)
    ]
    self.assertFalse(
        store.insert([b"h3"], probe, True),
        "the refused save must have left h1/h2 pinned",
    )
    store.release(hashes)
    self.assertTrue(
        store.insert([b"h3"], probe, True),
        "after the hand-back the entries must be evictable again",
    )

  def test_unpinned_load_is_refused(self):
    # Resident but unpinned is refused: the pin a successful load consumes
    # has to exist on entry. Only lookup() (or insert()) grants it.
    store = kv_cache_store.KVCacheStore(
        capacity=4, num_shards=1, store_server_ip="127.0.0.1"
    )
    rid = kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockId(rid, 0, kv_cache_store.BlockStatus.HOST)
    ]
    self.assertTrue(store.insert([b"h1"], slices, True))
    store.release([b"h1"])  # resident, unpinned

    self.assertFalse(store.load([b"h1"], [0]))

  def test_local_save_requires_hbm_residency(self):
    # A local save reads the bytes out of HBM; a host-only entry has nothing
    # there to save. Refused, and the refusal spends no pin.
    store = kv_cache_store.KVCacheStore(
        capacity=4, num_shards=1, store_server_ip="127.0.0.1"
    )
    rid = kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockId(rid, 0, kv_cache_store.BlockStatus.HOST)
    ]
    self.assertTrue(store.insert([b"h1"], slices, True))

    self.assertFalse(store.save([b"h1"]))
    # insert's pin survives the refusal; hand it back.
    store.release([b"h1"])

  def test_remote_save_refuses_an_hbm_only_block(self):
    # No two-hop: a remote save pulls the bytes out of host DRAM. A block
    # living only in HBM must be saved locally first.
    src = self._make_registry_store("wr_src_hbm_only")
    dst = self._make_registry_store("wr_dst_hbm_only")
    slices = [
        kv_cache_store.RaidenBlockId(
            src.raiden_id,
            host_block_id=-1,
            device_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(src.insert([b"h1"], slices, False))

    self.assertFalse(src.save([b"h1"], dst.raiden_id))
    src.release([b"h1"])

  def _make_registry_store(self, job_name, capacity=20):
    return kv_cache_store.KVCacheStore(
        capacity=capacity,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=kv_cache_store.RaidenId(job_name, "0", "kv_cache", 0),
        num_shards=1,
        store_server_ip="127.0.0.1",
    )

  def test_remote_save_requires_a_registry(self):
    # No global registry address: nothing to resolve the destination through,
    # and nothing to make the blocks reachable once they landed.
    store = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    src_id = kv_cache_store.RaidenId("wr_noreg", "0", "kv_cache", 0)
    self.assertTrue(
        store.insert(
            [b"a"],
            [
                kv_cache_store.RaidenBlockId(
                    src_id, 0, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )
    # The documented remote-save flow: insert only fabricates residency and
    # hands its pin back; lookup() grants the pin save(dst) would consume.
    store.release([b"a"])
    self.assertLen(store.lookup([b"a"]), 1)
    self.assertFalse(
        store.save(
            [b"a"], kv_cache_store.RaidenId("wr_dst", "0", "kv_cache", 0)
        )
    )
    # The refused save consumed nothing; give the lookup's pin back.
    store.release([b"a"])

  def test_remote_save_all_exist_settles_done(self):
    src = self._make_registry_store("wr_src_allexist")
    dst = self._make_registry_store("wr_dst_allexist")
    dst_id = dst.raiden_id

    hashes = [b"wr_a", b"wr_b"]
    src_slices = [
        kv_cache_store.RaidenBlockId(
            src.raiden_id, i, kv_cache_store.BlockStatus.HOST
        )
        for i in range(len(hashes))
    ]
    dst_slices = [
        kv_cache_store.RaidenBlockId(
            dst_id, 5 + i, kv_cache_store.BlockStatus.HOST
        )
        for i in range(len(hashes))
    ]
    self.assertTrue(src.insert(hashes, src_slices, True))
    src.release(hashes)  # insert fabricates residency; lookup grants the pin
    self.assertTrue(dst.insert(hashes, dst_slices, True))
    dst.release(hashes)  # destination state only; nothing consumes these

    # The documented remote-save flow: lookup() answers "host-resident here"
    # AND grants the pin the successful save(dst) below consumes.
    self.assertLen(src.lookup(hashes), len(hashes))

    # The destination already holds every offered hash. That is a SUCCESS:
    # hashes are content-addressed, so the peer having them is exactly the
    # post-condition the caller wanted, and no bytes move.
    self.assertTrue(src.save(hashes, dst_id))
    done, failed, pending, existing, unregistered = (
        src.poll_save_status()
    )
    self.assertCountEqual(done, hashes)
    self.assertEmpty(failed)
    self.assertEmpty(pending)
    self.assertEmpty(existing)
    self.assertEmpty(unregistered)

  def test_remote_save_partial_exist_reports_the_overlap(self):
    src = self._make_registry_store("wr_src_partial")
    dst = self._make_registry_store("wr_dst_partial")
    dst_id = dst.raiden_id

    hashes = [b"wr_p_a", b"wr_p_b"]
    self.assertTrue(
        src.insert(
            hashes,
            [
                kv_cache_store.RaidenBlockId(
                    src.raiden_id, i, kv_cache_store.BlockStatus.HOST
                )
                for i in range(len(hashes))
            ],
            True,
        )
    )
    src.release(hashes)  # insert fabricates residency; lookup grants the pin
    # Only one of the two.
    self.assertTrue(
        dst.insert(
            [hashes[0]],
            [
                kv_cache_store.RaidenBlockId(
                    dst_id, 5, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )
    dst.release([hashes[0]])  # destination state only

    self.assertLen(src.lookup(hashes), len(hashes))
    self.assertTrue(src.save(hashes, dst_id))
    done, failed, pending, existing, unregistered = (
        src.poll_save_status()
    )
    self.assertEmpty(done)
    self.assertCountEqual(failed, hashes)
    self.assertEmpty(pending)
    # The fourth element is the whole reason this poller returns five: the
    # caller decides whether to reissue with the remainder, and needs to know
    # what the remainder is. The store does not retry on its own.
    self.assertCountEqual(existing, [hashes[0]])
    self.assertEmpty(unregistered)

  def test_remote_save_carries_a_non_utf8_hash(self):
    # Real block hashes are raw digests, essentially never valid UTF-8. These
    # travel through the bindings as bytes and over the wire as proto `bytes`;
    # while those fields were declared `string` the sender serialized happily
    # and the receiver rejected the whole message.
    binary_hash = os.urandom(32)
    src = self._make_registry_store("wr_src_binary")
    dst = self._make_registry_store("wr_dst_binary")
    dst_id = dst.raiden_id

    self.assertTrue(
        src.insert(
            [binary_hash],
            [
                kv_cache_store.RaidenBlockId(
                    src.raiden_id, 0, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )
    src.release([binary_hash])  # insert fabricates residency
    self.assertTrue(
        dst.insert(
            [binary_hash],
            [
                kv_cache_store.RaidenBlockId(
                    dst_id, 5, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )
    dst.release([binary_hash])  # destination state only

    # lookup() grants the pin the successful save(dst) consumes.
    self.assertLen(src.lookup([binary_hash]), 1)
    self.assertTrue(src.save([binary_hash], dst_id))
    done, failed, _, existing, unregistered = src.poll_save_status()
    self.assertCountEqual(done, [binary_hash])
    self.assertEmpty(failed)
    self.assertEmpty(existing)
    self.assertEmpty(unregistered)

  def test_save_and_load_mocked(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    mock_impl = unittest.mock.MagicMock()
    controller._impl = mock_impl

    hashes = [b"hash_1", b"hash_2"]
    mock_impl.save.return_value = True
    self.assertTrue(controller.save(hashes))
    # The wrapper always passes the destination through; None is a local save.
    mock_impl.save.assert_called_with(hashes, None)
    mock_impl.save.return_value = False
    self.assertFalse(controller.save(hashes))

    device_block_ids = [2, 3]
    mock_impl.load.return_value = True
    self.assertTrue(controller.load(hashes, device_block_ids))
    mock_impl.load.assert_called_with(hashes, device_block_ids)
    mock_impl.load.return_value = False
    self.assertFalse(controller.load(hashes, device_block_ids))

    mock_impl.poll_save_status.return_value = (
        [b"hash_1"],
        [],
        [b"hash_2"],
        [],
        [],
    )
    self.assertEqual(
        controller.poll_save_status(), ([b"hash_1"], [], [b"hash_2"], [], [])
    )
    mock_impl.poll_save_status.assert_called_once()

    mock_impl.poll_load_status.return_value = ([], [b"hash_1"], [b"hash_2"])
    self.assertEqual(
        controller.poll_load_status(), ([], [b"hash_1"], [b"hash_2"])
    )
    mock_impl.poll_load_status.assert_called_once()

  def test_expected_worker_count_zero_does_not_block(self):
    """The default must stay non-blocking."""
    start = time.time()
    store = kv_cache_store.KVCacheStore(
        capacity=4,
        num_shards=1,
        store_server_ip="127.0.0.1",
        expected_worker_count=0,
        kv_pool_group="test_group",
    )
    self.assertLess(time.time() - start, 30.0)
    self.assertTrue(store.raiden_controller_address)
    self.assertEqual(store.capacity(), 4)

  def test_expected_worker_count_times_out_when_no_worker_registers(self):
    """Workers that never arrive must fail AT CONSTRUCTION with timeout."""
    orig = os.environ.get("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S")
    os.environ["RAIDEN_EXPECTED_WORKERS_TIMEOUT_S"] = "2"
    try:
      start = time.time()
      with self.assertRaises(RuntimeError) as cm:
        kv_cache_store.KVCacheStore(
            capacity=4,
            num_shards=1,
            store_server_ip="127.0.0.1",
            expected_worker_count=1,
        )
      elapsed = time.time() - start
    finally:
      if orig is not None:
        os.environ["RAIDEN_EXPECTED_WORKERS_TIMEOUT_S"] = orig
      else:
        os.environ.pop("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S", None)

    self.assertIn("timed out waiting for", str(cm.exception).lower())
    self.assertIn("worker", str(cm.exception).lower())
    self.assertGreaterEqual(elapsed, 2.0)
    self.assertLess(elapsed, 60.0)


class KVCacheStoreLoadWithSlicesTest(absltest.TestCase):
  """Covers load(..., slices=...), the pre-resolved-entry form.

  The cases below need no TPU: every one of them is rejected before any
  transfer is issued, which is the whole point of asserting on them.
  """

  def _make_store(self):
    return kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )

  def test_rejects_slices_count_mismatch(self):
    store = self._make_store()
    slices = [
        kv_cache_store.RaidenBlockId(
            store.raiden_id, 0, kv_cache_store.BlockStatus.HOST
        )
    ]
    # Two hashes, one slice: the store must not guess which hash it belongs to.
    self.assertFalse(store.load([b"h1", b"h2"], [0, 1], slices=slices))

  def test_rejects_device_block_ids_count_mismatch(self):
    store = self._make_store()
    store_id = store.raiden_id
    slices = [
        kv_cache_store.RaidenBlockId(
            store_id, 0, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockId(
            store_id, 1, kv_cache_store.BlockStatus.HOST
        ),
    ]
    # Every block needs a destination; there is no load-to-host mode.
    self.assertFalse(store.load([b"h1", b"h2"], [0], slices=slices))

  def test_rejects_mixed_local_and_remote_slices(self):
    store = self._make_store()
    peer_id = kv_cache_store.RaidenId("peer_job", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockId(
            store.raiden_id, 0, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockId(
            peer_id, 7, kv_cache_store.BlockStatus.REMOTE
        ),
    ]
    # One call is one source. A partially cached prefix looks exactly like
    # this, and the caller has to split it rather than hand it over whole.
    self.assertFalse(store.load([b"h1", b"h2"], [0, 1], slices=slices))

  def test_rejects_two_different_peers_in_one_call(self):
    store = self._make_store()
    peer_a = kv_cache_store.RaidenId("peer_a", "0", "kv_cache", 0)
    peer_b = kv_cache_store.RaidenId("peer_b", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockId(
            peer_a, 7, kv_cache_store.BlockStatus.REMOTE
        ),
        kv_cache_store.RaidenBlockId(
            peer_b, 9, kv_cache_store.BlockStatus.REMOTE
        ),
    ]
    self.assertFalse(store.load([b"h1", b"h2"], [0, 1], slices=slices))

  def test_rejects_slice_without_a_host_block(self):
    store = self._make_store()
    # host_block_id -1 means the entry names no host block, so there is
    # nothing to copy up even though the status claims HOST.
    slices = [
        kv_cache_store.RaidenBlockId(
            store.raiden_id, -1, kv_cache_store.BlockStatus.HOST
        )
    ]
    self.assertFalse(store.load([b"h1"], [0], slices=slices))

  def test_empty_request_is_a_no_op(self):
    store = self._make_store()
    self.assertTrue(store.load([], [], slices=[]))
    done, failed, pending = store.poll_load_status()
    self.assertEmpty(done)
    self.assertEmpty(failed)
    self.assertEmpty(pending)

  def test_forwards_slices_in_the_order_the_impl_expects(self):
    """Guards the one transposition this wrapper can make.

    Python takes (hashes, device_block_ids, slices) because slices is the
    optional one; the bound C++ overload takes (hashes, slices,
    device_block_ids). Swapping the last two is a type error the binding would
    reject, but only once something actually calls it -- and there is no
    end-to-end Torch test to do that. So assert the order here, where it costs
    nothing and needs no hardware.
    """
    store = self._make_store()
    store_id = store.raiden_id
    slices = [
        kv_cache_store.RaidenBlockId(
            store_id, 0, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockId(
            store_id, 1, kv_cache_store.BlockStatus.HOST
        ),
    ]
    hashes = [b"h1", b"h2"]
    device_block_ids = [4, 5]

    mock_impl = unittest.mock.MagicMock()
    store._impl = mock_impl
    mock_impl.load.return_value = True

    self.assertTrue(store.load(hashes, device_block_ids, slices=slices))
    mock_impl.load.assert_called_with(
        hashes,
        [s._impl for s in slices],  # pylint: disable=protected-access
        device_block_ids,
    )

    # Omitting slices must still reach the two-argument overload, not the
    # three-argument one with an empty list.
    mock_impl.reset_mock()
    self.assertTrue(store.load(hashes, device_block_ids))
    mock_impl.load.assert_called_with(hashes, device_block_ids)


if __name__ == "__main__":
  absltest.main()

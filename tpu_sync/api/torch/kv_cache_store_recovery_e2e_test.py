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

"""E2E test for the PyTorch KVCacheStore crash recovery from shared memory.

The torch twin of tpu_raiden/api/jax/kv_cache_store_recovery_e2e_test.py.
Simulates an engine crash and restart on one host:

  Phase A (subprocess): builds a KVCacheStore (with a raiden controller) and a
  KVCacheManager with the KV pool in shared memory (RAIDEN_SHM_KEY), saves
  blocks of distinct bytes to the host pool, then SIGKILLs itself so no
  destructor runs.

  Phase B (fresh subprocess, same environment): constructs the same stack.
  The store recovers its LRU cache from the KVCacheMetadata table that
  survived in shared memory, the manager re-attaches the surviving KV data
  segment, and the test loads the recovered blocks into a zeroed device cache
  and verifies byte equality with the pre-crash data.

The phases run as subprocesses of the test process (re-executing this file),
because a real crash must cross a process boundary: recovery has to start
from nothing but the environment and the surviving shared memory.

The test drives one TPU device through torch_tpu. Production deployments run
one worker process per device, each with its own shared-memory data segment;
a single device mirrors one such worker. The skipped test below pins one
intended contract: a KV pool of several backing arrays must keep its own
shared-memory mirror per array.
"""

import os
import socket
import subprocess
import sys
import time
import unittest

from absl import app
from absl import flags
from absl.testing import absltest

_PHASE = flags.DEFINE_string(
    "phase", None, "The e2e test phase to run in this subprocess."
)

_SIGKILL = 9

_NUM_BLOCKS = 2
_SHAPE = (_NUM_BLOCKS, 128, 8, 8, 128)  # float32
_HASHES = [b"hash_0", b"hash_1"]

_PENDING_SEGMENT_FIX = unittest.skip(
    "The shared-memory allocator returns the same single segment for every"
    " allocation in a process, so the host mirrors of a second KV array alias"
    " (and corrupt) the first; enable once each allocation gets its own"
    " segment."
)

# The phases run in subprocesses, so their detailed assertions are invisible
# to the test process. Each phase prints one of these markers only after all
# of its assertions passed; the test process then checks the subprocess's
# stdout for the marker as the positive proof that the phase reached that
# checkpoint (an exit code alone cannot distinguish which step died).
_PHASE_A_MARKER = "PHASE_A_SAVED"
_PHASE_B_RECOVERED_MARKER = "PHASE_B_RECOVERED"
_PHASE_B_COLD_MARKER = "PHASE_B_COLD"
_PHASE_B_BYTES_MARKER = "PHASE_B_BYTES_MATCH"
_PHASE_GATE_MARKER = "PHASE_GATE_OK"


def _find_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def _array_fill(shape, index):
  """A distinct, exactly float32-representable pattern per KV array."""
  import numpy as np  # pylint: disable=g-import-not-at-top

  size = int(np.prod(shape))
  return (np.arange(size, dtype=np.float32) + index * 3_000_000.0).reshape(
      shape
  )


def _make_caches(host_arrays):
  """Puts each host array on the single TPU device."""
  import torch  # pylint: disable=g-import-not-at-top

  return [torch.tensor(host_data, device="tpu") for host_data in host_arrays]


def _build_stack(tpu_caches):
  """Builds a store (with controller) and a manager wired to it."""
  # pylint: disable=g-import-not-at-top
  from tpu_sync.api.torch import kv_cache_manager
  from tpu_sync.api.torch import kv_cache_store
  # pylint: enable=g-import-not-at-top

  controller_port = _find_free_port()
  block_elements = _SHAPE[1] * _SHAPE[2] * _SHAPE[3] * _SHAPE[4]
  rid = kv_cache_store.RaidenId("recovery_e2e_job", "0", "recovery_cache", 0)
  store = kv_cache_store.KVCacheStore(
      capacity=_NUM_BLOCKS,
      raiden_id=rid,
      num_shards=1,
      shard_size_bytes=block_elements * 4,
      store_server_ip="localhost",
      raiden_controller_port=controller_port,
  )
  manager = kv_cache_manager.KVCacheManager(
      kv_caches=tpu_caches,
      local_control_port=0,
      max_blocks=_NUM_BLOCKS,
      num_slots=2,
      unsafe_skip_buffer_lock=True,
      raiden_worker_port=0,
      raiden_controller_address=f"localhost:{controller_port}",
      worker_id="worker_0",
      enable_shm=True,
  )
  return store, manager, rid


def _poll(poll_fn, want: int, what: str):
  for _ in range(3000):
    result = poll_fn()
    done, failed = result[0], result[1]
    if failed:
      raise RuntimeError(f"{what} failed: {failed}")
    if len(done) >= want:
      return
    time.sleep(0.01)
  raise RuntimeError(f"{what} timed out")


def _phase_a(shapes=(_SHAPE,)):
  """Saves distinct bytes to the shared-memory host pool, then crashes."""
  # pylint: disable=g-import-not-at-top
  from tpu_sync.api.torch import kv_cache_store
  # pylint: enable=g-import-not-at-top

  host_arrays = [_array_fill(shape, i) for i, shape in enumerate(shapes)]
  tpu_caches = _make_caches(host_arrays)
  # The manager must stay referenced: it is the worker the store drives via
  # RPC for save/load.
  store, manager, rid = _build_stack(tpu_caches)

  slices = [
      kv_cache_store.RaidenBlockId(
          rid,
          host_block_id=-1,
          device_block_id=i,
          status=kv_cache_store.BlockStatus.HBM,
      )
      for i in range(_NUM_BLOCKS)
  ]
  assert store.insert(_HASHES, slices, on_host=False)
  store.save(_HASHES)
  _poll(store.poll_save_status, _NUM_BLOCKS, "save")

  # The blocks are host-resident now; their bytes and the metadata table both
  # live in shared memory and must survive the crash below.
  lookup_res = store.lookup(_HASHES, pin_found=False)
  assert len(lookup_res) == _NUM_BLOCKS
  for i, (_, blk) in enumerate(lookup_res):
    assert blk.status == kv_cache_store.BlockStatus.HOST_AND_HBM, blk.status
    assert blk.host_block_id == i, (i, blk.host_block_id)

  print(_PHASE_A_MARKER, flush=True)
  # A real crash: SIGKILL runs no destructors and closes nothing cleanly.
  os.kill(os.getpid(), _SIGKILL)


def _phase_b(expect_recovery: bool, shapes=(_SHAPE,)):
  """Restarts the stack and serves the recovered blocks from host memory."""
  # pylint: disable=g-import-not-at-top
  import numpy as np
  from tpu_sync.api.torch import kv_cache_store
  # pylint: enable=g-import-not-at-top

  zeros = [np.zeros(shape, dtype=np.float32) for shape in shapes]
  tpu_caches = _make_caches(zeros)
  # Recovery happens inside the store construction: it attaches to the
  # surviving metadata table and rebuilds the LRU cache from it.
  store, manager, rid = _build_stack(tpu_caches)
  del rid

  lookup_res = store.lookup(_HASHES)
  if not expect_recovery:
    store.release(_HASHES)
    assert not lookup_res, f"expected a cold start, got hits: {lookup_res}"
    print(_PHASE_B_COLD_MARKER, flush=True)
    return

  assert (
      len(lookup_res) == _NUM_BLOCKS
  ), f"expected {_NUM_BLOCKS} recovered blocks, got {len(lookup_res)}"
  for i, (block_hash, blk) in enumerate(lookup_res):
    assert block_hash == _HASHES[i], (block_hash, _HASHES[i])
    # HBM did not survive the crash, so HOST_AND_HBM comes back as HOST, at
    # the same host block id as before the crash.
    assert blk.status == kv_cache_store.BlockStatus.HOST, blk.status
    assert blk.host_block_id == i, (i, blk.host_block_id)
  print(_PHASE_B_RECOVERED_MARKER, flush=True)

  # lookup() already resolved and pinned the recovered blocks; load() consumes the pin on success.
  store.load(_HASHES, list(range(_NUM_BLOCKS)))
  _poll(store.poll_load_status, _NUM_BLOCKS, "load")

  for i, (tpu_cache, shape) in enumerate(zip(tpu_caches, shapes)):
    np.testing.assert_array_equal(
        tpu_cache.cpu().numpy(), _array_fill(shape, i)
    )
  print(_PHASE_B_BYTES_MARKER, flush=True)


def _phase_gate(enable_shm: bool):
  """Constructs a bare manager and asserts it created no shm segments.

  Covers both gate legs: with the env key present but enable_shm off, the
  manager must stay out of shared memory; with enable_shm on but no env key,
  construction must fall back to private host memory rather than fail.
  """
  # pylint: disable=g-import-not-at-top
  import numpy as np
  from tpu_sync.api.torch import kv_cache_manager
  # pylint: enable=g-import-not-at-top

  tpu_caches = _make_caches([np.zeros(_SHAPE, dtype=np.float32)])
  manager = kv_cache_manager.KVCacheManager(
      kv_caches=tpu_caches,
      local_control_port=0,
      max_blocks=_NUM_BLOCKS,
      num_slots=2,
      unsafe_skip_buffer_lock=True,
      enable_shm=enable_shm,
  )
  key = os.environ.get("RAIDEN_SHM_KEY", "")
  if key:
    segments = [n for n in os.listdir("/dev/shm") if n.startswith(key)]
    assert not segments, f"shared-memory segments appeared: {segments}"
  del manager
  print(_PHASE_GATE_MARKER, flush=True)


class KVCacheStoreRecoveryE2ETest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.shm_key = f"raiden_torch_recovery_e2e_{os.getpid()}"

  def tearDown(self):
    # Mirrors the standard reclamation flow for the KV segments on a bare
    # metal VM: nothing in the serving stack ever unlinks them (they must
    # outlive any crash), so decommissioning is an explicit shm_unlink of the
    # deployment's keys by the operator or deployment tooling — here, the
    # test.
    for name in os.listdir("/dev/shm"):
      if name.startswith(self.shm_key):
        try:
          os.unlink(os.path.join("/dev/shm", name))
        except OSError:
          pass
    super().tearDown()

  def _phase_env(self, model_uid: str, with_shm_key: bool = True) -> dict:
    env = dict(os.environ)
    env.pop("RAIDEN_SHM_KEY", None)
    env.update({
        "RAIDEN_SHM_MODEL_UID": model_uid,
        "RAIDEN_DISABLE_SINGLETON_WORKER": "1",
        # Keep the child's import path identical to this process's, whether
        # running under bazel runfiles or directly.
        "PYTHONPATH": os.pathsep.join(sys.path),
    })
    if with_shm_key:
      # The KV pool and the metadata table are shm-backed under this single
      # deployment-level key; recovery is exercised through the public
      # construction path only, with no recovery-specific configuration.
      env["RAIDEN_SHM_KEY"] = self.shm_key
    return env

  def _run_phase(
      self, phase: str, model_uid: str, with_shm_key: bool = True
  ) -> subprocess.CompletedProcess:
    if sys.executable:
      cmd = [sys.executable, os.path.abspath(__file__), f"--phase={phase}"]
    else:
      cmd = [os.path.abspath(sys.argv[0]), f"--phase={phase}"]
    return subprocess.run(
        cmd,
        env=self._phase_env(model_uid, with_shm_key),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

  def _crash_phase_a(self, model_uid: str = "recovery_model", phase: str = "a"):
    result = self._run_phase(phase, model_uid)
    self.assertEqual(
        result.returncode,
        -_SIGKILL,
        f"phase A should die by SIGKILL:\n{result.stderr[-4000:]}",
    )
    self.assertIn(_PHASE_A_MARKER, result.stdout)

  def test_recovers_saved_blocks_after_crash(self):
    self._crash_phase_a()

    result = self._run_phase("b", "recovery_model")
    self.assertEqual(
        result.returncode, 0, f"phase B failed:\n{result.stderr[-4000:]}"
    )
    self.assertIn(_PHASE_B_RECOVERED_MARKER, result.stdout)
    self.assertIn(_PHASE_B_BYTES_MARKER, result.stdout)

  def test_model_uid_mismatch_cold_starts(self):
    self._crash_phase_a(model_uid="recovery_model")

    # A restart under a different model must not resurrect the surviving
    # table: both the data segment and the metadata table fail validation and
    # are re-created empty.
    result = self._run_phase("b_cold", "another_model")
    self.assertEqual(
        result.returncode, 0, f"phase B failed:\n{result.stderr[-4000:]}"
    )
    self.assertIn(_PHASE_B_COLD_MARKER, result.stdout)

  @_PENDING_SEGMENT_FIX
  def test_recovers_hybrid_model_arrays_after_crash(self):
    # A hybrid model's KV pool reaches the manager as several backing arrays,
    # all reshaped to one uniform kernel geometry (the connectors pad every
    # group's pages to the same size). The host mirror of each array must be
    # its own memory and survive the crash unclobbered by the others'.
    self._crash_phase_a(phase="a_hybrid")

    result = self._run_phase("b_hybrid", "recovery_model")
    self.assertEqual(
        result.returncode, 0, f"phase B failed:\n{result.stderr[-4000:]}"
    )
    self.assertIn(_PHASE_B_RECOVERED_MARKER, result.stdout)
    self.assertIn(_PHASE_B_BYTES_MARKER, result.stdout)

  def test_env_key_alone_does_not_pull_manager_into_shm(self):
    # RAIDEN_SHM_KEY is a deployment-level setting shared by every manager in
    # the process, including those whose host buffers are transient staging
    # with nothing to persist. A manager that did not opt in via enable_shm
    # must stay out of shared memory even with the key present.
    result = self._run_phase("gate_default_off", "recovery_model")
    self.assertEqual(
        result.returncode, 0, f"gate phase failed:\n{result.stderr[-4000:]}"
    )
    self.assertIn(_PHASE_GATE_MARKER, result.stdout)

  def test_enable_shm_without_env_key_falls_back(self):
    # Opting in without a deployment key degrades to private host memory; it
    # must not fail construction.
    result = self._run_phase(
        "gate_no_env", "recovery_model", with_shm_key=False
    )
    self.assertEqual(
        result.returncode, 0, f"gate phase failed:\n{result.stderr[-4000:]}"
    )
    self.assertIn(_PHASE_GATE_MARKER, result.stdout)


def main(argv):
  del argv  # Unused.
  if _PHASE.value == "a":
    _phase_a()
  elif _PHASE.value == "b":
    _phase_b(expect_recovery=True)
  elif _PHASE.value == "b_cold":
    _phase_b(expect_recovery=False)
  elif _PHASE.value == "a_hybrid":
    _phase_a(shapes=(_SHAPE, _SHAPE))
  elif _PHASE.value == "b_hybrid":
    _phase_b(expect_recovery=True, shapes=(_SHAPE, _SHAPE))
  elif _PHASE.value == "gate_default_off":
    _phase_gate(enable_shm=False)
  elif _PHASE.value == "gate_no_env":
    _phase_gate(enable_shm=True)


if __name__ == "__main__":
  if any(arg.startswith("--phase=") for arg in sys.argv):
    app.run(main)
  else:
    absltest.main()

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

"""Multiprocess distributed (MCJAX MPMD) JAX KVCacheManager D2H/H2D test."""

import multiprocessing as mp
import os
import socket
import time
from absl.testing import absltest
import numpy as np

# Set log directories BEFORE importing jax so spawned workers don't fail on
# /tmp/tpu_logs
_LOG_DIR = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("TPU_LOG_DIR", _LOG_DIR)
os.environ.setdefault("GLOG_log_dir", _LOG_DIR)
os.environ.setdefault("GOOGLE_LOG_DIR", _LOG_DIR)
os.environ.setdefault("TMPDIR", _LOG_DIR)


def pick_unused_ports(count: int) -> list[int]:
  ports = []
  for _ in range(count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    ports.append(port)
    s.close()
  return ports


def _mcjax_worker_fn(rank: int, world_size: int, coordinator_port: int) -> None:
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["JAX_PROCESS_INDEX"] = str(rank)
  os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "true"

  # On 4-rank MPMD on bare-metal TPUv7 VM, assign chip r to rank r
  os.environ["TPU_VISIBLE_DEVICES"] = str(rank)

  # Stagger JAX driver initialization so concurrent libtpu instances don't
  # collide on lockfiles
  time.sleep(rank * 1.0)

  import jax
  import jax.numpy as jnp
  from tpu_sync.api.jax import kv_cache_manager

  coord_address = f"127.0.0.1:{coordinator_port}"

  jax.distributed.initialize(
      coordinator_address=coord_address,
      num_processes=world_size,
      process_id=rank,
  )

  device = jax.local_devices()[0]
  print(f"[Rank {rank}] Initialized JAX on device {device}")

  block_size = 1024
  shape = (block_size, 128, 8)
  expected_val = float((rank + 1) * 10.0)

  arr = jax.device_put(jnp.full(shape, expected_val, dtype=jnp.float32), device)

  manager = kv_cache_manager.KVCacheManager(
      kv_caches=[arr],
      local_control_port=0,
      host_blocks_to_allocate=8388608,
      parallelism=1,
  )

  # Test D2H transfer
  d2h_future = manager.d2h(
      src_offsets=[0],
      dst_offsets=[0],
      copy_sizes=[block_size],
  )
  d2h_future.Await()

  # Test H2D transfer
  h2d_future = manager.h2d(
      src_offsets=[0],
      dst_offsets=[0],
      copy_sizes=[block_size],
  )
  h2d_future.Await()

  actual_data = np.array(arr)
  np.testing.assert_allclose(actual_data, expected_val, atol=1e-5)
  print(f"[Rank {rank}] MCJAX MPMD roundtrip succeeded!")


class KVCacheManagerMcjaxMpmdTest(absltest.TestCase):

  def test_mcjax_mpmd_stress(self):
    world_size = 4
    coord_port = pick_unused_ports(1)[0]
    processes = []
    ctx = mp.get_context("spawn")
    for r in range(world_size):
      p = ctx.Process(target=_mcjax_worker_fn, args=(r, world_size, coord_port))
      p.start()
      processes.append(p)
    for p in processes:
      p.join()
      self.assertEqual(p.exitcode, 0)


if __name__ == "__main__":
  absltest.main()

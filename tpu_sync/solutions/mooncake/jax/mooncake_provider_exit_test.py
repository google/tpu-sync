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

"""TPU test of the Mooncake provider's DMA mappings at interpreter exit.

Mooncake unmaps its host staging pools when its TransferEngine is destroyed. An
engine destroyed during interpreter teardown does that after JAX's own atexit
handler has cleared the backends, which can destroy the PJRT client. This test
leaves a pool mapped and unmaps it from an atexit handler that runs after
JAX's, as such an engine would. The handler checks that the client is gone,
then that dma_unmap succeeds and dma_map fails. It prints a line when all
holds, and otherwise makes the process exit with status 1.
"""

import atexit
import gc
import os
import sys
from typing import NoReturn
import weakref

# What test_mapping_left_for_interpreter_exit leaves to the handler below: the
# provider table, the pool still mapped through it, and a weak reference to the
# PJRT client of the exported array.
_left_mapped = []


def _fail(message: str) -> NoReturn:
  print(f"mooncake_provider_exit_test FAILED: {message}", file=sys.stderr)
  os._exit(1)


def _unmap_after_jax_clean_up() -> None:
  """Unmaps like a TransferEngine destroyed after JAX's atexit handler."""
  for table, pool, client in _left_mapped:
    gc.collect()
    if client() is not None:
      _fail("JAX's atexit handler left the PJRT client alive.")
    unmapped = table.dma_unmap(pool.ctypes.data)
    mapped_again = table.dma_map(pool.ctypes.data, pool.nbytes)
    if unmapped != 0 or mapped_again == 0:
      _fail(
          f"With the PJRT client destroyed, dma_unmap returned {unmapped}"
          f" (want 0) and dma_map returned {mapped_again} (want non-zero)."
      )
    print(
        "mooncake_provider_exit_test: with the PJRT client destroyed,"
        " dma_unmap returned 0 and dma_map failed.",
        file=sys.stderr,
    )


# atexit runs handlers in reverse order of registration, so this one, which is
# registered before jax is imported, runs after JAX's.
_JAX_WAS_IMPORTED_FIRST = "jax" in sys.modules
atexit.register(_unmap_after_jax_clean_up)

# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position
from absl.testing import absltest
import jax
import numpy as np

from tpu_sync.solutions.mooncake.jax import _mooncake_provider
from tpu_sync.solutions.mooncake.jax import mooncake_provider
from tpu_sync.solutions.mooncake.jax import mooncake_provider_test_util as test_util

# pylint: enable=g-import-not-at-top,g-bad-import-order,wrong-import-position


class MooncakeProviderExitTest(absltest.TestCase):

  def test_mapping_left_for_interpreter_exit(self):
    # Otherwise the handler would run before JAX's, not after it.
    self.assertFalse(_JAX_WAS_IMPORTED_FIRST)
    table = test_util.load_table()
    device = jax.local_devices(backend="tpu")[0]
    array = jax.device_put(np.zeros((8, 128), np.float32), device)
    # The export only gives the provider a client to map with. No array or
    # handle outlives this test, so JAX's handler can destroy the client.
    for handle in mooncake_provider.export_shards(array):
      table.release(handle)

    pool = test_util.staging_pool(4 << 20)
    before = _mooncake_provider.live_dma_mapping_count()
    self.assertEqual(table.dma_map(pool.ctypes.data, pool.nbytes), 0)
    self.assertEqual(_mooncake_provider.live_dma_mapping_count(), before + 1)
    _left_mapped.append((table, pool, weakref.ref(device.client)))


if __name__ == "__main__":
  absltest.main()

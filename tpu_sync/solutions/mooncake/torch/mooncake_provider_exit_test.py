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

"""TPU test of the torch Mooncake provider's DMA mappings at interpreter exit.

Mooncake unmaps its host staging pools when its TransferEngine is destroyed. An
engine destroyed during interpreter teardown does that after TorchTPU's own
atexit handler has shut the PJRT runtime down. This test leaves a pool mapped
and checks two atexit handlers:
  * One registered after TorchTPU's, so it runs before it: with the runtime
    still up, dma_map must already fail, i.e. the provider's exit hook ran
    before TorchTPU's handler.
  * One registered before TorchTPU's, so it runs after it, as such an engine's
    teardown would: with the runtime gone, dma_unmap must succeed and dma_map
    fail.
Each prints a line when its checks hold, and otherwise makes the process exit
with status 1.
"""

import atexit
import os
import sys
from typing import NoReturn

# What test_mapping_left_for_interpreter_exit leaves to the handlers below: the
# provider table, the pool still mapped through it, a pool that is not mapped,
# and TorchTPU's check of whether its PJRT runtime is up.
_left_mapped = []


def _fail(message: str) -> NoReturn:
  print(f"mooncake_provider_exit_test FAILED: {message}", file=sys.stderr)
  os._exit(1)


def _unmap_after_torch_tpu_shutdown() -> None:
  """Unmaps like a TransferEngine destroyed after TorchTPU's atexit handler."""
  for table, pool, _, runtime_is_initialized in _left_mapped:
    if runtime_is_initialized():
      _fail("TorchTPU's atexit handler left its PJRT runtime initialized.")
    unmapped = table.dma_unmap(pool.ctypes.data)
    mapped_again = table.dma_map(pool.ctypes.data, pool.nbytes)
    if unmapped != 0 or mapped_again == 0:
      _fail(
          f"With the PJRT runtime shut down, dma_unmap returned {unmapped}"
          f" (want 0) and dma_map returned {mapped_again} (want non-zero)."
      )
    print(
        "mooncake_provider_exit_test: with the PJRT runtime shut down,"
        " dma_unmap returned 0 and dma_map failed.",
        file=sys.stderr,
    )


# atexit runs handlers in reverse order of registration, so this one, which is
# registered before torch (and with it TorchTPU) is imported, runs after
# TorchTPU's.
_TORCH_WAS_IMPORTED_FIRST = "torch" in sys.modules
atexit.register(_unmap_after_torch_tpu_shutdown)

# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position
from absl.testing import absltest
import torch

from tpu_sync.frameworks.torch import _tpu_raiden_torch
from tpu_sync.solutions.mooncake import mooncake_provider_test_util as test_util
from tpu_sync.solutions.mooncake.torch import mooncake_provider

# pylint: enable=g-import-not-at-top,g-bad-import-order,wrong-import-position


def _check_before_torch_tpu_shutdown() -> None:
  """Checks that the provider's exit hook ran before TorchTPU's handler."""
  for table, _, unmapped_pool, runtime_is_initialized in _left_mapped:
    if not runtime_is_initialized():
      _fail("TorchTPU's atexit handler ran before this one, registered later.")
    if table.dma_map(unmapped_pool.ctypes.data, unmapped_pool.nbytes) == 0:
      _fail(
          "With the PJRT runtime still up at exit, dma_map succeeded: the"
          " provider's exit hook did not run before TorchTPU's atexit handler."
      )
    print(
        "mooncake_provider_exit_test: before TorchTPU's atexit handler,"
        " dma_map already failed.",
        file=sys.stderr,
    )


# Importing the provider loaded TorchTPU, which registered its atexit handler,
# so this one runs before it. The provider registers its own exit hook at the
# first export, after this, so that hook runs before this one.
atexit.register(_check_before_torch_tpu_shutdown)


class MooncakeProviderExitTest(absltest.TestCase):

  def test_mapping_left_for_interpreter_exit(self):
    # Otherwise the handler would run before TorchTPU's, not after it.
    self.assertFalse(_TORCH_WAS_IMPORTED_FIRST)
    table = test_util.load_table(mooncake_provider.library_path())
    tensor = torch.zeros((8, 128), device="tpu:0")
    # The export gives the provider a client to map with and registers the
    # provider's own exit hook. No handle outlives this test.
    for handle in mooncake_provider.export_shards(tensor):
      table.release(handle)

    pool = test_util.staging_pool(4 << 20)
    # Mapped and unmapped once here, so that only the exiting process can make
    # mapping it fail at exit.
    unmapped_pool = test_util.staging_pool(4 << 20)
    impl = _tpu_raiden_torch.mooncake_provider
    before = impl.live_dma_mapping_count()
    unmapped_ptr = unmapped_pool.ctypes.data
    self.assertEqual(table.dma_map(unmapped_ptr, unmapped_pool.nbytes), 0)
    self.assertEqual(table.dma_unmap(unmapped_ptr), 0)
    self.assertEqual(table.dma_map(pool.ctypes.data, pool.nbytes), 0)
    self.assertEqual(impl.live_dma_mapping_count(), before + 1)
    _left_mapped.append((table, pool, unmapped_pool, torch.tpu.is_initialized))


if __name__ == "__main__":
  absltest.main()

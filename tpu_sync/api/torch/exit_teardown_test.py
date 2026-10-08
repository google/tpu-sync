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

"""Teardown tests: tpu_sync objects alive at interpreter exit must not crash it.

At exit torch_tpu drops its own reference to the PJRT client. The device
buffers still hold it, so objects that outlive their tensors can be the
client's last owners while the interpreter destroys them. Each case runs in a
fresh interpreter; this process never opens the TPU, which only one process
may hold.
"""

import os
import subprocess
import sys
import textwrap

from absl.testing import absltest
from absl.testing import parameterized

# Copies a block out and back through a manager, drops the tensor, and exits
# with either the manager or a finished transfer future still referenced.
_SCRIPT = textwrap.dedent("""
    import sys, time
    import torch
    import torch_tpu  # noqa: F401
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    def wait(future):
      while not future.is_ready():
        time.sleep(0.001)
      return future

    buf = torch.ones((8, 128, 2, 2, 128), dtype=torch.bfloat16).to("tpu")
    manager = KVCacheManager([[buf]], local_control_port=0,
                             host_blocks_to_allocate=8,
                             unsafe_skip_buffer_lock=True)
    wait(manager.d2h([1], [0], [1]))
    future = wait(manager.h2d([0], [4], [1]))
    torch.accelerator.synchronize()
    del buf
    if sys.argv[1] == "manager":
      del future
    else:
      del manager
""")


class ExitTeardownTest(parameterized.TestCase):

  @parameterized.parameters("manager", "future")
  def test_exits_cleanly(self, alive):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    try:
      result = subprocess.run(
          [sys.executable, "-c", _SCRIPT, alive],
          env=env,
          capture_output=True,
          text=True,
          timeout=300,
      )
    except subprocess.TimeoutExpired as e:
      stderr = (e.stderr or b"").decode(errors="replace")[-4000:]
      self.fail(f"hung at exit with {alive} alive:\n{stderr}")
    self.assertEqual(result.returncode, 0, result.stderr[-4000:])


if __name__ == "__main__":
  absltest.main()

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

"""Tests for framework-neutral N-D shard slice math."""

import os
import subprocess
import sys

from absl.testing import absltest

from tpu_sync.kv_cache import nd_slice_math


def _bounds(slices):
  return [
      tuple((dimension.start, dimension.end) for dimension in shard.dimensions)
      for shard in slices
  ]


class NDSliceMathTest(absltest.TestCase):

  def test_controller_imports_with_jax_masked(self):
    if not sys.executable:
      self.skipTest("sys.executable is None under hermetic python")
    python_path = os.pathsep.join(
        path for path in sys.path if isinstance(path, str)
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = python_path
    script = """
import sys

sys.modules["jax"] = None
from tpu_sync.kv_cache import nd_slice_math
from tpu_sync.rpc import raiden_controller

slices = nd_slice_math.compute_nd_shard_slices((8, 12), (2, 3))
assert len(slices) == 6
assert raiden_controller.nd_slice_math is nd_slice_math
assert sys.modules["jax"] is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    self.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )

  def test_matches_legacy_jax_module_goldens(self):
    cases = (
        (
            (5, 10),
            (2, 3),
            [
                ((0, 2), (0, 3)),
                ((0, 2), (3, 6)),
                ((0, 2), (6, 10)),
                ((2, 5), (0, 3)),
                ((2, 5), (3, 6)),
                ((2, 5), (6, 10)),
            ],
        ),
        (
            (16, 32, 64),
            (2, 1, 4),
            [
                ((0, 8), (0, 32), (0, 16)),
                ((0, 8), (0, 32), (16, 32)),
                ((0, 8), (0, 32), (32, 48)),
                ((0, 8), (0, 32), (48, 64)),
                ((8, 16), (0, 32), (0, 16)),
                ((8, 16), (0, 32), (16, 32)),
                ((8, 16), (0, 32), (32, 48)),
                ((8, 16), (0, 32), (48, 64)),
            ],
        ),
    )

    for global_shape, mesh_shape, golden in cases:
      with self.subTest(global_shape=global_shape, mesh_shape=mesh_shape):
        neutral_output = _bounds(
            nd_slice_math.compute_nd_shard_slices(global_shape, mesh_shape)
        )
        self.assertEqual(neutral_output, golden)

  def test_format_nd_slices(self):
    slices_1d = nd_slice_math.compute_nd_shard_slices((2048,), (4,))
    self.assertEqual(
        nd_slice_math.format_nd_slices(slices_1d),
        "[[0:512], [512:1024], [1024:1536], [1536:2048]]",
    )

    slices_2d = nd_slice_math.compute_nd_shard_slices((6144, 2048), (4, 1))
    self.assertEqual(
        nd_slice_math.format_nd_slices(slices_2d),
        "[[0:1536, 0:2048], [1536:3072, 0:2048], [3072:4608, 0:2048],"
        " [4608:6144, 0:2048]]",
    )

    slices_3d = nd_slice_math.compute_nd_shard_slices((2048, 8, 128), (1, 4, 1))
    self.assertEqual(
        nd_slice_math.format_nd_slices(slices_3d),
        "[[0:2048, 0:2, 0:128], [0:2048, 2:4, 0:128], [0:2048, 4:6, 0:128],"
        " [0:2048, 6:8, 0:128]]",
    )

    slices_3d_multi_axis = nd_slice_math.compute_nd_shard_slices(
        (2048, 8, 128), (2, 2, 2)
    )
    self.assertEqual(
        nd_slice_math.format_nd_slices(slices_3d_multi_axis),
        "[[0:1024, 0:4, 0:64], [0:1024, 0:4, 64:128], [0:1024, 4:8, 0:64],"
        " [0:1024, 4:8, 64:128], [1024:2048, 0:4, 0:64], [1024:2048, 0:4,"
        " 64:128], [1024:2048, 4:8, 0:64], [1024:2048, 4:8, 64:128]]",
    )

    slices_4d_multi_axis = nd_slice_math.compute_nd_shard_slices(
        (32, 16, 128, 64), (2, 2, 2, 1)
    )
    self.assertEqual(
        nd_slice_math.format_nd_slices(slices_4d_multi_axis),
        "[[0:16, 0:8, 0:64, 0:64], [0:16, 0:8, 64:128, 0:64], [0:16, 8:16,"
        " 0:64, 0:64], [0:16, 8:16, 64:128, 0:64], [16:32, 0:8, 0:64, 0:64],"
        " [16:32, 0:8, 64:128, 0:64], [16:32, 8:16, 0:64, 0:64], [16:32, 8:16,"
        " 64:128, 0:64]]",
    )


if __name__ == "__main__":
  absltest.main()

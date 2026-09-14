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

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_tpu  # pylint: disable=unused-import

from tpu_sync.api.torch.weight_synchronizer import (
    WeightSynchronizer,
)


class WeightSynchronizerTorchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 2
    self.num_shards = 1
    self.block_size = 2

  def _run_push_sync(
      self,
      src_tensors: list[list[torch.Tensor]],
      dst_tensors: list[list[torch.Tensor]],
  ):
    ws_source = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    ws_dest = WeightSynchronizer(
        dst_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.assertIsNotNone(ws_source.local_port)
    self.assertIsNotNone(ws_dest.local_port)

    peer_dest = f"127.0.0.1:{ws_dest.local_port}"
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()

    for l in range(len(src_tensors)):
      for sh in range(len(src_tensors[l])):
        self.assertTrue(
            torch.equal(dst_tensors[l][sh].cpu(), src_tensors[l][sh].cpu())
        )

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("bf16", torch.bfloat16),
      ("int32", torch.int32),
  )
  def test_e2e_3node_distributed_weight_push(self, dtype):
    shape = (self.block_size, 128, 8)

    src_tensors = [
        [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_shards)
        ]
        for _ in range(self.num_layers)
    ]
    dst1_tensors = [
        [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_shards)
        ]
        for _ in range(self.num_layers)
    ]
    dst2_tensors = [
        [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_shards)
        ]
        for _ in range(self.num_layers)
    ]

    ws_dest1 = WeightSynchronizer(
        dst1_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    ws_dest2 = WeightSynchronizer(
        dst2_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.assertIsNotNone(ws_dest1.local_port)
    self.assertIsNotNone(ws_dest2.local_port)

    peer_dest1 = f"127.0.0.1:{ws_dest1.local_port}"
    peer_dest2 = f"127.0.0.1:{ws_dest2.local_port}"

    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        val = int(l + 10) if dtype == torch.int32 else float(l + 10.0)
        src_tensors[l][sh].fill_(val)

    ws_source = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.assertIsNotNone(ws_source.local_port)

    ws_source.push_weights([peer_dest1, peer_dest2])
    ws_dest1.h2d()
    ws_dest2.h2d()

    # Assert both destinations have received the trainer's weights on TPU HBM!
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        self.assertTrue(
            torch.equal(dst1_tensors[l][sh].cpu(), src_tensors[l][sh].cpu())
        )
        self.assertTrue(
            torch.equal(dst2_tensors[l][sh].cpu(), src_tensors[l][sh].cpu())
        )

  def test_wait_for_transfer_completion_api_exists(self):
    shape = (self.block_size, 128, 8)
    tensors = [
        [torch.zeros(shape, dtype=torch.float32, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws = WeightSynchronizer(tensors, local_port=0, bind_ip="127.0.0.1")
    self.assertTrue(hasattr(ws, "wait_for_transfer_completion"))
    self.assertTrue(callable(ws.wait_for_transfer_completion))

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("bf16", torch.bfloat16),
  )
  def test_bind_weights(self, dtype):
    shape = (self.block_size, 128, 8)

    src_tensors = [
        [torch.full(shape, fill_value=5.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    dst_tensors = [
        [torch.zeros(shape, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]

    ws_source = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    ws_dest = WeightSynchronizer(
        dst_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    peer_dest = f"127.0.0.1:{ws_dest.local_port}"

    # --- Sync 1 (V1: 5.0 -> 0.0) ---
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()

    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(dst_tensors[l][0].cpu(), src_tensors[l][0].cpu())
      )

    # --- Bind weights to V2 ---
    new_src_tensors = [
        [torch.full(shape, fill_value=10.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws_source.bind_weights(new_src_tensors)
    ws_source.d2h()

    new_dst_tensors = [
        [torch.full(shape, fill_value=-1.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws_dest.bind_weights(new_dst_tensors)

    # --- Sync 2 (V2: 10.0 -> -1.0) ---
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()

    # Verify Sync 2 updated new_dst_tensors to 10.0
    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(new_dst_tensors[l][0].cpu(), new_src_tensors[l][0].cpu())
      )

    # Verify original V1 dst_tensors were NOT overwritten (still 5.0)
    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(dst_tensors[l][0].cpu(), src_tensors[l][0].cpu())
      )

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("bf16", torch.bfloat16),
  )
  def test_heterogeneous_layers_small_first(self, dtype):
    shapes = [(1024,), (1024, 3072), (2048, 2048)]
    src_tensors = [
        [
            torch.full(
                shape,
                fill_value=float(i + 1.0),
                dtype=dtype,
                device=self.device,
            )
        ]
        for i, shape in enumerate(shapes)
    ]
    dst_tensors = [
        [torch.zeros(shape, dtype=dtype, device=self.device)]
        for shape in shapes
    ]
    self._run_push_sync(src_tensors, dst_tensors)

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("bf16", torch.bfloat16),
  )
  def test_heterogeneous_layers_large_first(self, dtype):
    shapes = [(1024, 3072), (1024,), (128,)]
    src_tensors = [
        [
            torch.full(
                shape,
                fill_value=float(i + 1.0),
                dtype=dtype,
                device=self.device,
            )
        ]
        for i, shape in enumerate(shapes)
    ]
    dst_tensors = [
        [torch.zeros(shape, dtype=dtype, device=self.device)]
        for shape in shapes
    ]
    self._run_push_sync(src_tensors, dst_tensors)

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("bf16", torch.bfloat16),
  )
  def test_heterogeneous_layers_local_roundtrip(self, dtype):
    shapes = [(1024,), (1024, 3072), (2048, 2048)]
    src_tensors = [
        [
            torch.full(
                shape,
                fill_value=float(i + 10.0),
                dtype=dtype,
                device=self.device,
            )
        ]
        for i, shape in enumerate(shapes)
    ]
    ws = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    ws.d2h()

    # Zero out new destination tensors and bind them
    zero_tensors = [
        [torch.zeros(shape, dtype=dtype, device=self.device)]
        for shape in shapes
    ]
    ws.bind_weights(zero_tensors)
    ws.h2d()

    for i in range(len(shapes)):
      self.assertTrue(
          torch.equal(zero_tensors[i][0].cpu(), src_tensors[i][0].cpu())
      )

  def _make_tensors(
      self, num_layers: int, num_shards: int
  ) -> list[list[torch.Tensor]]:
    shape = (self.block_size, 128, 8)
    return [
        [
            torch.zeros(shape, dtype=torch.float32, device=self.device)
            for _ in range(num_shards)
        ]
        for _ in range(num_layers)
    ]

  def test_multi_numa_endpoints_and_metrics(self):
    os.environ["ENABLE_MULTI_NUMA"] = "1"
    try:
      tensors = self._make_tensors(num_layers=2, num_shards=2)
      ws = WeightSynchronizer(
          tensors,
          local_port=0,
          parallelism=2,
          auto_h2d=True,
      )
      self.assertEqual(ws.num_layers, 2)
      self.assertEqual(ws.num_shards, 2)
      self.assertIsNotNone(ws.local_port)

      eps = ws.get_local_endpoints()
      self.assertNotEmpty(eps)
      all_shards = []
      for ep in eps:
        self.assertIn("endpoint", ep)
        self.assertIn("shards", ep)
        all_shards.extend(ep["shards"])
      self.assertEqual(sorted(all_shards), [0, 1])

      ws.test_only_set_skip_tiling(True)
      ws.test_only_set_skip_tiling([True, False])

      metrics = ws.get_metrics()
      self.assertIn("last_d2h_time_ms", metrics)
      self.assertIn("total_d2h_time_ms", metrics)
      ws.reset_metrics()
    finally:
      os.environ["ENABLE_MULTI_NUMA"] = "0"

  def test_multi_numa_push_weights_e2e(self):
    os.environ["ENABLE_MULTI_NUMA"] = "1"
    try:
      dst_tensors = self._make_tensors(num_layers=1, num_shards=2)
      src_tensors = self._make_tensors(num_layers=1, num_shards=2)
      ws_dst = WeightSynchronizer(
          dst_tensors,
          local_port=0,
          parallelism=4,
          auto_h2d=True,
      )
      ws_src = WeightSynchronizer(
          src_tensors,
          local_port=0,
          parallelism=4,
      )
      eps = ws_dst.get_local_endpoints()
      peers = (
          [ep["endpoint"] for ep in eps]
          if eps
          else [f"127.0.0.1:{ws_dst.local_port}"]
      )
      ws_src.push_weights(peers)
      ws_dst.h2d()
    finally:
      os.environ["ENABLE_MULTI_NUMA"] = "0"


if __name__ == "__main__":
  absltest.main()

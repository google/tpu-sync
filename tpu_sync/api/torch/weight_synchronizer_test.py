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

"""E2E physical integration tests for PyTorch WeightSynchronizer on XLA TPUs."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
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
    self.slice_byte_size = 16384 // 4  # float32 capacity

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("int32", torch.int32),
  )
  def test_e2e_3node_distributed_weight_push(self, dtype):
    shape = (self.block_size, 128, 8)  # 16384 bytes capacity per layer shard

    # 1. Allocate source (Trainer) weights on Device TPU
    src_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      src_tensors.append(shards)

    # Allocate destination 1 (Inference Peer 1) weights
    dst1_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      dst1_tensors.append(shards)

    # Allocate destination 2 (Inference Peer 2) weights
    dst2_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      dst2_tensors.append(shards)

    # 2. Instantiate destination WeightSynchronizers on ephemeral ports!
    ws_dest1 = WeightSynchronizer(
        dst1_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    ws_dest2 = WeightSynchronizer(
        dst2_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )

    self.assertIsNotNone(ws_dest1.local_port)
    self.assertIsNotNone(ws_dest2.local_port)

    peer_dest1 = f"localhost:{ws_dest1.local_port}"
    peer_dest2 = f"localhost:{ws_dest2.local_port}"

    # ==========================================================================
    # Scenario A: Test the Push API E2E (1 Source pushes to 2 Destinations!)
    # ==========================================================================
    # Trainer fills source weights with distinct values per layer
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        val = float(l + 10.0)  # Layer 0=10.0, Layer 1=11.0
        src_tensors[l][sh].fill_(val)

    # Force execution of fill_ on source tensors to ensure TPU memory is updated
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        _ = src_tensors[l][sh].cpu()

    # Recreate/Instantiate ws_source to capture filled buffers!
    ws_source = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.assertIsNotNone(ws_source.local_port)

    # Source pushes weights to both dest1 and dest2 socket servers E2E!
    ws_source.push_weights([peer_dest1, peer_dest2])
    ws_dest1.h2d()
    ws_dest2.h2d()

    # Assert both destinations have received the trainer's weights on TPU HBM!
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        expected_val = float(l + 10.0)
        np.testing.assert_allclose(
            dst1_tensors[l][sh].cpu().numpy(), expected_val, atol=1e-5
        )
        np.testing.assert_allclose(
            dst2_tensors[l][sh].cpu().numpy(), expected_val, atol=1e-5
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

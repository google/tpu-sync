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

"""E2E physical unit tests for PyTorch KVCacheManager on XLA TPUs."""

import faulthandler

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu  # pylint: disable=unused-import  # registers torch.tpu backend

from tpu_sync.frameworks.torch import _tpu_raiden_torch as _kv_cache_manager

# Upper bound on how long constructing a manager over lazy KV caches may take.
# The constructor holds the GIL, so a regressed deadlock freezes the whole
# process and no Python-thread watchdog can fire; faulthandler dumps every
# thread stack and hard-exits the process past this bound so a regression fails
# loudly instead of hanging the test runner.
_CONSTRUCTION_DEADLOCK_TIMEOUT_S = 60.0


class KVCacheManagerTorchTest(parameterized.TestCase):

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
  def test_e2e_distributed_staging_offloads_and_socket_transceives(self, dtype):
    num_blocks = 2
    shape = (num_blocks * self.block_size, 128, 8)  # 2 blocks capacity

    # 1. Allocate sharded eager tensors directly on the physical XLA TPU devices!
    # Layer 0 and Layer 1 for Source (Trainer)
    src_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        # Distinct layer val patterns: Layer 0=1.0, Layer 1=2.0
        val = float(l + 1.0)
        t = torch.full(shape, fill_value=val, dtype=dtype, device=self.device)
        shards.append(t)
      src_tensors.append(shards)

    # Layer 0 and Layer 1 for Destination (Inference, initially zeroed)
    dst_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      dst_tensors.append(shards)

    # 2. Instantiate two real managers locally on ephemeral loopback ports!
    ws_source = _kv_cache_manager.KVCacheManager(
        src_tensors,
        local_port=0,
        host_blocks_to_allocate=num_blocks * self.block_size,
        parallelism=1,
    )
    ws_dest = _kv_cache_manager.KVCacheManager(
        dst_tensors,
        local_port=0,
        host_blocks_to_allocate=num_blocks * self.block_size,
        parallelism=1,
    )

    self.assertIsNotNone(ws_source.local_port)
    self.assertIsNotNone(ws_dest.local_port)

    peer_source = ws_source.get_local_endpoints()[0]["endpoint"]
    peer_dest = ws_dest.get_local_endpoints()[0]["endpoint"]

    # Validate baseline state (destination tensors on TPU are zeroed)
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        np.testing.assert_allclose(
            dst_tensors[l][sh].cpu().numpy(), 0.0, atol=1e-5
        )

    # ==========================================================================
    # Step 1 (Source): Offload caches from Device TPU to Host CPU staging buffer E2E!
    # ==========================================================================
    d2h_future = ws_source.D2h(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[self.block_size],
    )
    d2h_future.Await()

    # ==========================================================================
    # Step 2 (Network): Push and Pull E2E!
    # ==========================================================================
    # Source pushes block page 0 to destination peer server!
    # This will allocate block 0 on ws_dest.
    write_ids, write_future = ws_source.H2hWrite(
        peer_dest, src_block_ids=[0, 1]
    )
    write_future.Await()
    self.assertEqual(write_ids, [0, 1])

    # Destination pulls block page 0 from source peer!
    # This will allocate block 1 on ws_dest (since block 0 is locked).
    read_ids, read_future = ws_dest.H2hRead(peer_source, src_block_ids=[0, 1])
    read_future.Await()
    self.assertEqual(read_ids, [2, 3])

    # ==========================================================================
    # Step 3 (Destination): Stage received weights from Host onto Device TPU HBM E2E!
    # ==========================================================================
    # Copy pushed data (host block 0, offset 0) to device block 0 (offset 0)
    h2d_future_push = ws_dest.H2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[self.block_size],
    )
    h2d_future_push.Await()

    # Copy pulled data (host block 1, offset 2) to device block 1 (offset 2)
    h2d_future_pull = ws_dest.H2d(
        src_offsets_major_dim=[2],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[self.block_size],
    )
    h2d_future_pull.Await()

    # ==========================================================================
    # Step 4: Verify complete numerical parity on the destination TPU devices!
    # ==========================================================================
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        expected_val = float(l + 1.0)
        actual_data = dst_tensors[l][sh].cpu().numpy()
        # Verify both blocks have the expected values
        np.testing.assert_allclose(actual_data[0:2], expected_val, atol=1e-5)
        np.testing.assert_allclose(actual_data[2:4], expected_val, atol=1e-5)

  def test_construction_over_lazy_kv_caches_does_not_deadlock(self):
    """Constructing a manager over still-deferred KV caches must not hang.

    Under DEFER_AND_FUSE (vLLM's execution mode) a KV cache registered right
    after allocation still carries a pending deferred op. UnpackTorchTensor
    resolves each tensor's base buffer by awaiting it, and the await only
    completes once the deferred graph has been executed. The unpack forces that
    execution before awaiting; without it the constructor waits on a graph that
    only this (now-blocked) thread could flush.
    """
    num_blocks = 2
    shape = (num_blocks * self.block_size, 128, 8)

    with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
      lazy_tensors = [
          [torch.ones(shape, dtype=torch.float32, device=self.device)]
          for _ in range(self.num_layers)
      ]
      faulthandler.dump_traceback_later(
          _CONSTRUCTION_DEADLOCK_TIMEOUT_S, exit=True
      )
      try:
        manager = _kv_cache_manager.KVCacheManager(
            lazy_tensors,
            local_port=0,
            host_blocks_to_allocate=num_blocks * self.block_size,
            parallelism=1,
        )
      finally:
        faulthandler.cancel_dump_traceback_later()

    self.assertIsNotNone(manager.local_port)

  def test_heterogeneous_layer_block_sizes_transfer(self):
    """Layers with different per-block byte sizes transfer byte-exactly."""

    num_host_blocks = 4
    # Layer 0 rows are 4x larger than layer 1 rows (128*8 vs 128*2 fp32).
    shapes = [(num_host_blocks, 128, 8), (num_host_blocks, 128, 2)]

    src_tensors = []
    dst_tensors = []
    for layer_idx, shape in enumerate(shapes):
      numel = int(np.prod(shape))
      pattern = torch.arange(numel, dtype=torch.float32).reshape(
          shape
      ) + 1000.0 * (layer_idx + 1)
      src_tensors.append([pattern.to(self.device)])
      dst_tensors.append(
          [torch.zeros(shape, dtype=torch.float32, device=self.device)]
      )

    ws_source = _kv_cache_manager.KVCacheManager(
        src_tensors,
        local_port=0,
        host_blocks_to_allocate=num_host_blocks,
        parallelism=1,
    )
    ws_dest = _kv_cache_manager.KVCacheManager(
        dst_tensors,
        local_port=0,
        host_blocks_to_allocate=num_host_blocks,
        parallelism=1,
    )
    peer_dest = ws_dest.get_local_endpoints()[0]["endpoint"]

    # Stage rows 2..3 at host blocks 2..3 -- a NONZERO host base. The
    # regression only manifests when the wire's layer-0-derived offsets and
    # the DMA's per-layer offsets address disjoint regions, which a
    # contiguous-from-zero staging accidentally masks.
    d2h_future = ws_source.D2h(
        src_offsets_major_dim=[2],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[2],
    )
    d2h_future.Await()

    write_ids, write_future = ws_source.H2hWrite(
        peer_dest, src_block_ids=[2, 3]
    )
    write_future.Await()
    self.assertLen(write_ids, 2)

    h2d_future = ws_dest.H2d(
        src_offsets_major_dim=[write_ids[0]],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[2],
    )
    h2d_future.Await()

    for layer_idx in range(len(shapes)):
      actual = dst_tensors[layer_idx][0].cpu().numpy()
      expected = src_tensors[layer_idx][0].cpu().numpy()
      np.testing.assert_array_equal(
          actual[2:4],
          expected[2:4],
          err_msg=f"layer {layer_idx} bytes corrupted in heterogeneous transfer",
      )
      np.testing.assert_array_equal(
          actual[0:2],
          np.zeros_like(expected[0:2]),
          err_msg=f"layer {layer_idx} untouched rows unexpectedly written",
      )


if __name__ == "__main__":
  absltest.main()

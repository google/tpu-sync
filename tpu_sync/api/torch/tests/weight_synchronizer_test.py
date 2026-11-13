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

import asyncio
import gc
import os
import socket
import threading
import time

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_tpu  # pylint: disable=unused-import

from tpu_sync.api.torch import weight_synchronizer as ws_lib
from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer
from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

# Slice size for CPU-only synchronizers whose host buffer contents are unused.
_SMALL_SLICE_BYTES = 1024 * 1024

# Seconds to wait for a TCP handshake with a listener on loopback. A closed port
# is refused immediately, so this only bounds a hung handshake.
_CONNECT_TIMEOUT_SECS = 5.0

# 2D float32 weight shapes totaling 10 MiB (4 MiB + 4 MiB + 2 MiB), all
# (8, 128)-tile-aligned, for controller-driven pipelined D2H + H2H verification.
_PIPELINED_CONTROLLER_SHAPES = ((1024, 1024), (512, 2048), (512, 1024))

# 0.010 Gbps == 1.25 MB/s, so 10 MiB takes ~8.4 s on H2H while local TPU D2H
# finishes in tens of milliseconds (leaving >5 s margin beyond HBM poll window).
_PIPELINED_EGRESS_GBPS = 0.010
_CONTROLLER_TRANSFER_TIMEOUT_SECS = 60.0
_HBM_POLL_TIMEOUT_SECS = 3.0
_HBM_POLL_INTERVAL_SECS = 0.02
_MIN_HBM_RELEASE_FRACTION = 0.85

# Layer shapes for the bit-exact tests: one multiple of the TPU tile, and others
# of several ranks that are not, so device layouts carry padding.
_CORRECTNESS_SHAPES = ((256, 1024), (3, 100, 7), (1000,), (2, 3, 17, 130), (1,))

# Integer dtype of the same width as each tested dtype, for bitwise comparison.
_BITS_DTYPE = {
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.int32: torch.int32,
    torch.int8: torch.int8,
}


def _random_weights(
    dtype: torch.dtype, num_shards: int, step: int
) -> list[list[torch.Tensor]]:
  """Returns CPU [layer][shard] tensors of random values for one sync step.

  Every (step, layer, shard) gets its own seed, so a misplaced element, a
  swapped shard or layer, or a stale step all show up as mismatches, unlike a
  constant fill.

  Args:
    dtype: Element type of every tensor.
    num_shards: Number of shards per layer.
    step: Sync step, so that consecutive steps carry different values.
  """
  weights = []
  for layer, shape in enumerate(_CORRECTNESS_SHAPES):
    shards = []
    for shard in range(num_shards):
      generator = torch.Generator().manual_seed(
          step * 1_000_003 + layer * 1_009 + shard
      )
      if dtype.is_floating_point:
        shards.append(torch.randn(shape, generator=generator).to(dtype))
      else:
        info = torch.iinfo(dtype)
        shards.append(
            torch.randint(
                info.min, info.max, shape, generator=generator, dtype=dtype
            )
        )
    weights.append(shards)
  return weights


def _listener_accepts_connections(port: int) -> bool:
  """Returns whether a TCP listener accepts connections on 127.0.0.1:port."""
  try:
    with socket.create_connection(
        ("127.0.0.1", port), timeout=_CONNECT_TIMEOUT_SECS
    ):
      return True
  except ConnectionRefusedError:
    return False


def _synced_hbm_bytes() -> int:
  gc.collect()
  torch.accelerator.synchronize()
  return torch.accelerator.memory_allocated()


def _run_step_k_plus_1_on_tpu(
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
  """Allocates step-(k+1) TPU tensors and runs a bit-exact affine TPU op."""
  device_tensors: list[list[torch.Tensor]] = []
  cpu_expected: list[list[torch.Tensor]] = []
  for layer_idx, shape in enumerate(_PIPELINED_CONTROLLER_SHAPES):
    gen = torch.Generator().manual_seed(80_001 + layer_idx * 1_009)
    cpu_base = torch.randint(
        -1024, 1024, shape, generator=gen, dtype=torch.int32
    ).to(torch.float32)
    cpu_expected.append([cpu_base * 2.0 + 3.0])
    device_tensors.append([cpu_base.to(device) * 2.0 + 3.0])
  torch.accelerator.synchronize()
  return device_tensors, cpu_expected


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

  def test_init_without_bind_ip_and_empty_data_nics(self):
    """Verifies that WeightSynchronizer initialization does not crash (SIGFPE)

    when bind_ip is None and data_nics is empty (e.g. on standard GKE pods).
    """
    old_data_nics = os.environ.get("TPU_RAIDEN_DATA_NICS")
    borg_keys = ["BORG_TASK_HANDLE", "BORG_ALLOC_DIR", "BORG_CELL"]
    saved_borg = {k: os.environ.pop(k, None) for k in borg_keys}
    os.environ["TPU_RAIDEN_DATA_NICS"] = "nonexistent_nic_test"
    try:
      shape = (self.block_size, 128, 8)
      tensors = [
          [torch.zeros(shape, dtype=torch.float32, device=self.device)]
          for _ in range(self.num_layers)
      ]
      ws = WeightSynchronizer(
          tensors, local_port=0, parallelism=1, bind_ip=None
      )
      self.assertIsNotNone(ws.local_port)
    finally:
      for k, v in saved_borg.items():
        if v is not None:
          os.environ[k] = v
      if old_data_nics is not None:
        os.environ["TPU_RAIDEN_DATA_NICS"] = old_data_nics
      else:
        os.environ.pop("TPU_RAIDEN_DATA_NICS", None)

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
    ws_source.d2h().wait()

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
  def test_unbind_weights(self, dtype):
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

    # --- Explicit Unbind ---
    ws_source.unbind_weights()
    ws_dest.unbind_weights()

    # --- Re-bind weights to V2 on existing synchronizers ---
    new_src_tensors = [
        [torch.full(shape, fill_value=10.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws_source.bind_weights(new_src_tensors)
    ws_source.d2h().wait()

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

    # Unbind again
    ws_source.unbind_weights()
    ws_dest.unbind_weights()

  def test_transfer_after_unbind_raises(self):
    """Unbound synchronizers must fail loudly instead of shipping stale data."""
    shape = (self.block_size, 128, 8)
    dtype = torch.float32

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

    # Baseline sync works.
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()
    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(dst_tensors[l][0].cpu(), src_tensors[l][0].cpu())
      )

    ws_source.unbind_weights()
    ws_dest.unbind_weights()

    # Every device-touching entry point must raise while unbound.
    with self.assertRaisesRegex(RuntimeError, "unbound"):
      ws_source.d2h()
    with self.assertRaisesRegex(RuntimeError, "unbound"):
      ws_source.push_weights([peer_dest])
    with self.assertRaisesRegex(RuntimeError, "unbound"):
      ws_dest.h2d()

    # Re-binding on the same instances fully recovers.
    new_src_tensors = [
        [torch.full(shape, fill_value=7.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    new_dst_tensors = [
        [torch.zeros(shape, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws_source.bind_weights(new_src_tensors)
    ws_dest.bind_weights(new_dst_tensors)
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()
    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(new_dst_tensors[l][0].cpu(), new_src_tensors[l][0].cpu())
      )

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("bf16", torch.bfloat16),
  )
  def test_bind_weights_empty_list_unbinds(self, dtype):
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

    # --- Sync 1 ---
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()

    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(dst_tensors[l][0].cpu(), src_tensors[l][0].cpu())
      )

    # --- bind_weights([]) as alias for unbind ---
    ws_source.bind_weights([])
    ws_dest.bind_weights([])

    # --- Re-bind weights to V2 ---
    new_src_tensors = [
        [torch.full(shape, fill_value=12.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws_source.bind_weights(new_src_tensors)
    ws_source.d2h().wait()

    new_dst_tensors = [
        [torch.full(shape, fill_value=-2.0, dtype=dtype, device=self.device)]
        for _ in range(self.num_layers)
    ]
    ws_dest.bind_weights(new_dst_tensors)

    # --- Sync 2 ---
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()

    for l in range(self.num_layers):
      self.assertTrue(
          torch.equal(new_dst_tensors[l][0].cpu(), new_src_tensors[l][0].cpu())
      )

    ws_source.bind_weights([])
    ws_dest.bind_weights([])

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
    ws.d2h().wait()

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
    self.assertIn("total_h2h_bandwidth_gbps", metrics)
    ws.reset_metrics()

  def test_multi_numa_push_weights_e2e(self):
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

  def test_explicit_global_shard_indices(self):
    tensors = self._make_tensors(num_layers=2, num_shards=2)
    ws = WeightSynchronizer(
        tensors,
        local_port=0,
        parallelism=1,
        bind_ip="127.0.0.1",
        global_shard_indices=[4, 7],
    )
    eps = ws.get_local_endpoints()
    self.assertLen(eps, 1)
    self.assertEqual(eps[0]["shards"], [4, 7])

  def test_default_global_shard_indices_normalization(self):
    tensors = self._make_tensors(num_layers=2, num_shards=2)
    ws = WeightSynchronizer(
        tensors,
        local_port=0,
        parallelism=1,
        bind_ip="127.0.0.1",
    )
    eps = ws.get_local_endpoints()
    self.assertLen(eps, 1)
    self.assertEqual(eps[0]["shards"], [0, 1])
    self.assertEqual(eps[0]["global_shards"], [0, 1])

  def test_get_host_buffer_heterogeneous_layers(self):
    # Layer 0 is small (4 KB), Layer 1 is large (4 MB > Layer 0 + 256 KB)
    shapes = [(1024,), (1024, 1024)]
    tensors = [
        [torch.zeros(shape, dtype=torch.float32, device=self.device)]
        for shape in shapes
    ]
    ws = WeightSynchronizer(
        tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    buf0 = ws.get_host_buffer(layer_idx=0, shard_idx=0)
    buf1 = ws.get_host_buffer(layer_idx=1, shard_idx=0)
    self.assertGreaterEqual(buf0.numel(), 1024 * 4)
    self.assertLess(buf0.numel(), 1024 * 1024 * 4)
    self.assertGreaterEqual(buf1.numel(), 1024 * 1024 * 4)

  def test_host_buffer_release_without_gil_tears_down_synchronizer(self):
    """Dropping the last view's storage without the GIL frees the synchronizer.

    Tensor.set_() swaps out the view's storage inside ATen dispatch with the
    GIL released (gil_scoped_release in THPVariable_set_), so the deleter of
    the get_host_buffer() storage has to take the GIL itself before it drops
    the storage's reference to the synchronizer. The synchronizer's listener,
    which its destructor closes, shows whether it is still alive.
    """
    ws = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=1,
        num_shards=1,
        slice_byte_size=_SMALL_SLICE_BYTES,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    listener_port = ws.listener_port
    self.assertTrue(
        _listener_accepts_connections(listener_port),
        "Positive control failed: the listener of a live synchronizer on"
        f" 127.0.0.1:{listener_port} refused a connection.",
    )
    view = ws.get_host_buffer(layer_idx=0, shard_idx=0)
    ws.shutdown()
    del ws
    gc.collect()
    self.assertTrue(
        _listener_accepts_connections(listener_port),
        "Synchronizer destroyed while a get_host_buffer() view is alive: its"
        f" listener on 127.0.0.1:{listener_port} refused connections after"
        " shutdown().",
    )

    view.set_()

    self.assertEqual(
        view.numel(),
        0,
        "set_() did not replace the storage of the get_host_buffer() view,"
        " so the release without the GIL was not exercised.",
    )
    self.assertFalse(
        _listener_accepts_connections(listener_port),
        "Releasing the last get_host_buffer() storage inside set_(), with the"
        " GIL released, did not destroy the synchronizer: its listener on"
        f" 127.0.0.1:{listener_port} still accepts connections.",
    )

  def test_metrics_accumulation_bandwidth_and_reset(self):
    ws_source = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=1,
        num_shards=1,
        slice_byte_size=1024,
        local_port=0,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=1,
        num_shards=1,
        slice_byte_size=1024,
        local_port=0,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_source.shutdown)
    self.addCleanup(ws_dest.shutdown)

    m0 = ws_source.get_metrics()
    self.assertEqual(m0["total_h2h_bytes"], 0)
    self.assertEqual(m0["total_h2h_time_ms"], 0.0)
    self.assertEqual(m0["total_h2h_bandwidth_gbps"], 0.0)

    def _send_ctrl_req(port: int, req: raiden_service_pb2.ControlRequest):
      payload = req.SerializeToString()
      pipe_client = control_pipe_client.ControlPipeClient(
          backend=control_pipe_client.ControlPipeBackendType.GRPC
      )
      try:
        resp_bytes = pipe_client.send_raw_bytes_sync(
            f"localhost:{port}",
            payload,
            timeout=_CONNECT_TIMEOUT_SECS,
            message_type="tpu_sync.rpc.ControlRequest",
        )
      finally:
        pipe_client.close()
      resp = raiden_service_pb2.ControlResponse()
      resp.ParseFromString(resp_bytes)
      self.assertTrue(resp.success, resp.message)

    for step in range(2):
      uuid = 70001 + step
      dst_req = raiden_service_pb2.ControlRequest(
          command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
          start_transfer_request=raiden_service_pb2.StartTransferRequest(
              is_sender=False,
              uuid=uuid,
              expected_block_count=1,
              expected_layer_chunk_counts={0: 1},
          ),
      )
      _send_ctrl_req(ws_dest.listener_port, dst_req)

      src_transfer_req = raiden_service_pb2.StartTransferRequest(
          is_sender=True,
          uuid=uuid,
          skip_d2h=True,
      )
      sched = src_transfer_req.shard_push_schedules[0]
      entry = sched.entries.add()
      entry.dst_peer = f"127.0.0.1:{ws_dest.local_port}"
      entry.dst_shard_idx = 0
      entry.src_offset_bytes = 0
      entry.dst_offset_bytes = 0
      entry.size_bytes = 1024
      entry.count = 1
      entry.layer_idx = 0

      src_req = raiden_service_pb2.ControlRequest(
          command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
          start_transfer_request=src_transfer_req,
      )
      _send_ctrl_req(ws_source.listener_port, src_req)
      ws_dest.wait_for_transfer_completion(uuid)

    m2 = ws_source.get_metrics()
    self.assertEqual(m2["total_h2h_bytes"], 2048)
    self.assertGreater(m2["total_h2h_time_ms"], 0.0)
    self.assertGreater(m2["total_h2h_bandwidth_gbps"], 0.0)
    self.assertEqual(m2["push_resharded_call_count"], 2)

    ws_source.reset_metrics()
    m_reset = ws_source.get_metrics()
    self.assertEqual(m_reset["total_h2h_bytes"], 0)
    self.assertEqual(m_reset["total_h2h_time_ms"], 0.0)
    self.assertEqual(m_reset["total_h2h_bandwidth_gbps"], 0.0)
    self.assertEqual(m_reset["push_resharded_call_count"], 0)

  def test_local_d2h_and_h2d_record_metrics(self):
    """Explicit d2h()/h2d() calls show up in get_metrics()."""
    # Tile-aligned shapes, so the on-device size has no layout padding.
    shapes = [(256, 1024), (512, 2048)]
    tensors = [
        [torch.full(shape, 3.0, dtype=torch.float32, device=self.device)]
        for shape in shapes
    ]
    logical_bytes = sum(t[0].numel() * t[0].element_size() for t in tensors)
    ws = WeightSynchronizer(
        tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )

    m0 = ws.get_metrics()
    self.assertEqual(m0["d2h_call_count"], 0)
    self.assertEqual(m0["last_d2h_bytes"], 0)
    self.assertEqual(m0["last_h2d_bytes"], 0)
    self.assertEqual(m0["h2d_bandwidth_gbps"], 0.0)

    ws.d2h().wait()
    m1 = ws.get_metrics()
    self.assertEqual(m1["d2h_call_count"], 1)
    self.assertGreaterEqual(m1["last_d2h_bytes"], logical_bytes)
    self.assertEqual(m1["total_d2h_bytes"], m1["last_d2h_bytes"])
    self.assertGreater(m1["last_d2h_time_ms"], 0.0)
    self.assertGreater(m1["d2h_bandwidth_gbps"], 0.0)

    ws.h2d()
    m2 = ws.get_metrics()
    # H2D writes back exactly the bytes D2H read.
    self.assertEqual(m2["last_h2d_bytes"], m1["last_d2h_bytes"])
    self.assertEqual(m2["total_h2d_bytes"], m2["last_h2d_bytes"])
    self.assertGreater(m2["last_h2d_time_ms"], 0.0)
    self.assertEqual(m2["total_h2d_time_ms"], m2["last_h2d_time_ms"])
    self.assertGreater(m2["h2d_bandwidth_gbps"], 0.0)

    ws.d2h().wait()
    ws.h2d()
    m3 = ws.get_metrics()
    self.assertEqual(m3["d2h_call_count"], 2)
    self.assertEqual(m3["total_d2h_bytes"], 2 * m1["last_d2h_bytes"])
    self.assertEqual(m3["total_h2d_bytes"], 2 * m2["last_h2d_bytes"])

    ws.reset_metrics()
    m_reset = ws.get_metrics()
    self.assertEqual(m_reset["d2h_call_count"], 0)
    self.assertEqual(m_reset["total_h2d_bytes"], 0)
    self.assertEqual(m_reset["total_h2d_time_ms"], 0.0)

  def _to_device(
      self, weights: list[list[torch.Tensor]]
  ) -> list[list[torch.Tensor]]:
    return [[t.to(self.device) for t in shards] for shards in weights]

  def _zeros_on_device(
      self, weights: list[list[torch.Tensor]]
  ) -> list[list[torch.Tensor]]:
    return [
        [torch.zeros_like(t, device=self.device) for t in shards]
        for shards in weights
    ]

  def _copy_in_place(
      self,
      tensors: list[list[torch.Tensor]],
      values: list[list[torch.Tensor]],
  ) -> None:
    """Overwrites device `tensors` in place, as an optimizer step does."""
    for shards, new_shards in zip(tensors, values, strict=True):
      for tensor, new_tensor in zip(shards, new_shards, strict=True):
        tensor.copy_(new_tensor.to(self.device))

  def _assert_bit_exact(
      self,
      actual: list[list[torch.Tensor]],
      expected: list[list[torch.Tensor]],
      label: str,
  ) -> None:
    """Asserts that device tensors hold exactly the bits of CPU tensors."""
    for layer, (actual_shards, expected_shards) in enumerate(
        zip(actual, expected, strict=True)
    ):
      for shard, (got, want) in enumerate(
          zip(actual_shards, expected_shards, strict=True)
      ):
        got = got.cpu()
        where = f"{label}: layer {layer} {tuple(want.shape)} shard {shard}"
        self.assertEqual(got.shape, want.shape, where)
        self.assertEqual(got.dtype, want.dtype, where)
        bits = _BITS_DTYPE[want.dtype]
        mismatches = (got.view(bits) != want.view(bits)).nonzero()
        if mismatches.numel():
          self.fail(
              f"{where}: {mismatches.shape[0]} of {want.numel()} elements"
              f" differ, first at index {tuple(mismatches[0].tolist())}"
          )

  def _send_control_request(
      self, port: int, request: raiden_service_pb2.ControlRequest
  ) -> None:
    payload = request.SerializeToString()
    pipe_client = control_pipe_client.ControlPipeClient(
        backend=control_pipe_client.ControlPipeBackendType.GRPC
    )
    try:
      response_bytes = pipe_client.send_raw_bytes_sync(
          f"localhost:{port}",
          payload,
          timeout=_CONNECT_TIMEOUT_SECS,
          message_type="tpu_sync.rpc.ControlRequest",
      )
    finally:
      pipe_client.close()
    response = raiden_service_pb2.ControlResponse()
    response.ParseFromString(response_bytes)
    self.assertTrue(response.success, response.message)

  def _push_resharded(
      self,
      sender: WeightSynchronizer,
      receiver: WeightSynchronizer,
      weights: list[list[torch.Tensor]],
      uuid: int,
      host_tiling_mode: raiden_service_pb2.HostTilingMode,
  ) -> None:
    """Pushes every layer of every shard to the same shard of `receiver`.

    Sends the requests the controller sends when sender and receiver have the
    same sharding: the receiver is armed for one chunk per layer and shard, and
    the sender gets one whole-layer entry per layer and shard.

    Args:
      sender: Synchronizer that pushes.
      receiver: Synchronizer that receives.
      weights: The [layer][shard] tensors being synchronized, for their sizes.
      uuid: Transfer id.
      host_tiling_mode: Layout the receiver keeps its host buffers in.
    """
    num_shards = len(weights[0])
    self._send_control_request(
        receiver.listener_port,
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=raiden_service_pb2.StartTransferRequest(
                is_sender=False,
                uuid=uuid,
                expected_block_count=len(weights) * num_shards,
                expected_layer_chunk_counts={
                    layer: num_shards for layer in range(len(weights))
                },
                host_tiling_mode=host_tiling_mode,
            ),
        ),
    )
    request = raiden_service_pb2.StartTransferRequest(is_sender=True, uuid=uuid)
    for shard in range(num_shards):
      schedule = request.shard_push_schedules[shard]
      for layer, shards in enumerate(weights):
        entry = schedule.entries.add()
        entry.dst_peer = f"127.0.0.1:{receiver.local_port}"
        entry.dst_shard_idx = shard
        entry.size_bytes = shards[shard].numel() * shards[shard].element_size()
        entry.layer_idx = layer
    self._send_control_request(
        sender.listener_port,
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=request,
        ),
    )

  def test_push_weights_sends_in_place_updates(self):
    """Pushes send the values of in-place updates made after construction."""
    dtype = torch.bfloat16
    src_tensors = self._to_device(_random_weights(dtype, num_shards=2, step=0))
    dst_tensors = self._zeros_on_device(src_tensors)
    ws_source = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        dst_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_dest.shutdown)

    for step in range(1, 4):
      expected = _random_weights(dtype, num_shards=2, step=step)
      self._copy_in_place(src_tensors, expected)

      ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
      ws_dest.h2d()

      self._assert_bit_exact(dst_tensors, expected, f"step {step}")

  def test_h2d_reaches_receiver_tensors_updated_in_place(self):
    """H2D lands in receiver tensors that were updated in place after binding."""
    dtype = torch.bfloat16
    expected = _random_weights(dtype, num_shards=2, step=1)
    src_tensors = self._to_device(expected)
    dst_tensors = self._to_device(_random_weights(dtype, num_shards=2, step=0))
    ws_source = WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        dst_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_dest.shutdown)
    for shards in dst_tensors:
      for tensor in shards:
        tensor.zero_()

    ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
    ws_dest.h2d()

    self._assert_bit_exact(dst_tensors, expected, "receiver")

  @parameterized.named_parameters(
      ("deferred_tiling", raiden_service_pb2.HOST_TILING_MODE_DEFERRED),
      ("tiling_on_arrival", raiden_service_pb2.HOST_TILING_MODE_ON_ARRIVAL),
  )
  def test_resharded_push_with_auto_h2d_sends_in_place_updates(
      self, host_tiling_mode
  ):
    """Controller-driven pushes with auto-H2D, as RL trainers and samplers run."""
    dtype = torch.bfloat16
    src_tensors = self._to_device(_random_weights(dtype, num_shards=2, step=0))
    dst_tensors = self._zeros_on_device(src_tensors)
    ws_source = WeightSynchronizer(
        src_tensors,
        local_port=0,
        parallelism=2,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        dst_tensors,
        local_port=0,
        parallelism=2,
        listener_port=0,
        bind_ip="127.0.0.1",
        auto_h2d=True,
    )
    self.addCleanup(ws_dest.shutdown)

    for step in range(1, 3):
      uuid = 81000 + step
      expected = _random_weights(dtype, num_shards=2, step=step)
      self._copy_in_place(src_tensors, expected)

      self._push_resharded(ws_source, ws_dest, expected, uuid, host_tiling_mode)
      ws_dest.wait_for_transfer_completion(uuid)

      self._assert_bit_exact(dst_tensors, expected, f"step {step}")

  @parameterized.named_parameters(
      ("view_tiled", "view", False),
      ("view_skip_tiling", "view", True),
      ("view_clone_tiled", "view_clone", False),
      ("view_clone_skip_tiling", "view_clone", True),
      ("empty_copy_tiled", "empty_copy", False),
      ("empty_copy_skip_tiling", "empty_copy", True),
      ("times_one_tiled", "times_one", False),
      ("times_one_skip_tiling", "times_one", True),
      ("host_round_trip_tiled", "host_round_trip", False),
      ("host_round_trip_skip_tiling", "host_round_trip", True),
  )
  def test_push_folded_expert_stack(self, variant, skip_tiling):
    """Pushes an [E, F, D] expert stack folded to [E * F, D] each way.

    verl sends MoE expert stacks folded this way. torch_tpu backs the view, its
    clone and a copy into an empty tensor with the stack's storage buffer; the
    product and the host round trip get buffers of their own.

    Args:
      variant: How the folded tensor is made from the stack.
      skip_tiling: Whether both synchronizers copy raw device bytes instead of
        detiling on D2H and tiling on H2D.
    """
    generator = torch.Generator().manual_seed(101)
    stack = torch.randn(8, 768, 2048, generator=generator).to(torch.bfloat16)
    view = stack.to(self.device).reshape(8 * 768, 2048)
    if variant == "view":
      src = view
    elif variant == "view_clone":
      src = view.clone()
    elif variant == "empty_copy":
      src = torch.empty_like(view).copy_(view)
    elif variant == "times_one":
      src = view * 1
    else:
      src = view.cpu().to(self.device)
    dst = torch.zeros(8 * 768, 2048, dtype=torch.bfloat16, device=self.device)
    ws_source = WeightSynchronizer(
        [[src]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        [[dst]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_dest.shutdown)
    ws_source.test_only_set_skip_tiling(skip_tiling)
    ws_dest.test_only_set_skip_tiling(skip_tiling)

    ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
    ws_dest.h2d()

    self._assert_bit_exact(
        [[dst]], [[stack.reshape(8 * 768, 2048)]], "receiver"
    )

  @parameterized.named_parameters(
      ("fp32_tiled", torch.float32, False),
      ("fp32_skip_tiling", torch.float32, True),
      ("bf16_tiled", torch.bfloat16, False),
      ("bf16_skip_tiling", torch.bfloat16, True),
  )
  def test_push_from_clone_of_folded_tensor(self, dtype, skip_tiling):
    """Pushes a clone of an [E, F, D] expert stack folded to [E * F, D].

    torch_tpu backs the clone with the stack's [E, F, D] storage buffer.
    Skipping tiling sends that buffer's raw device bytes into the receiver's
    [E * F, D] buffer.

    Args:
      dtype: Element type of the stack.
      skip_tiling: Whether both synchronizers copy raw device bytes instead of
        detiling on D2H and tiling on H2D.
    """
    generator = torch.Generator().manual_seed(102)
    stack = torch.randn(8, 64, 256, generator=generator).to(dtype)
    src = stack.to(self.device).reshape(8 * 64, 256).clone()
    dst = torch.zeros(8 * 64, 256, dtype=dtype, device=self.device)
    ws_source = WeightSynchronizer(
        [[src]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        [[dst]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_dest.shutdown)
    ws_source.test_only_set_skip_tiling(skip_tiling)
    ws_dest.test_only_set_skip_tiling(skip_tiling)

    ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
    ws_dest.h2d()

    self._assert_bit_exact([[dst]], [[stack.reshape(8 * 64, 256)]], "receiver")
    # Raw copies neither detile on the sender nor tile on the receiver.
    self.assertEqual(
        ws_source.get_metrics()["total_detiled_bytes"] == 0, skip_tiling
    )
    self.assertEqual(
        ws_dest.get_metrics()["total_tiled_bytes"] == 0, skip_tiling
    )

  @parameterized.named_parameters(
      ("tiled", False),
      ("skip_tiling", True),
  )
  def test_push_from_folded_view_of_another_dtype(self, skip_tiling):
    """Pushes an int32 [E * F, D] view of an f32 [E, F, D] stack.

    The view is backed by the stack's f32 storage buffer, whose element type is
    not the view's.

    Args:
      skip_tiling: Whether both synchronizers copy raw device bytes instead of
        detiling on D2H and tiling on H2D.
    """
    generator = torch.Generator().manual_seed(103)
    stack = torch.randn(8, 64, 256, generator=generator)
    src = stack.to(self.device).view(torch.int32).view(8 * 64, 256)
    dst = torch.zeros(8 * 64, 256, dtype=torch.int32, device=self.device)
    ws_source = WeightSynchronizer(
        [[src]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        [[dst]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_dest.shutdown)
    ws_source.test_only_set_skip_tiling(skip_tiling)
    ws_dest.test_only_set_skip_tiling(skip_tiling)

    ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
    ws_dest.h2d()

    self._assert_bit_exact(
        [[dst]], [[stack.view(torch.int32).reshape(8 * 64, 256)]], "receiver"
    )
    self.assertEqual(
        ws_source.get_metrics()["total_detiled_bytes"] == 0, skip_tiling
    )
    self.assertEqual(
        ws_dest.get_metrics()["total_tiled_bytes"] == 0, skip_tiling
    )

  def test_bind_weights_to_view_folding_leading_dims(self):
    """Pushes from a folded view bound in place of an unfolded tensor.

    The view is backed by an [E, F, D] storage buffer, and every later transfer
    resolves the device buffers of the bound tensors again.
    """
    generator = torch.Generator().manual_seed(104)
    stack = torch.randn(8, 64, 256, generator=generator)
    dst = torch.zeros(8 * 64, 256, dtype=torch.float32, device=self.device)
    ws_source = WeightSynchronizer(
        [[torch.zeros(8 * 64, 256, dtype=torch.float32, device=self.device)]],
        local_port=0,
        parallelism=1,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        [[dst]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_dest.shutdown)

    ws_source.bind_weights([[stack.to(self.device).view(8 * 64, 256)]])
    ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
    ws_dest.h2d()

    self._assert_bit_exact([[dst]], [[stack.reshape(8 * 64, 256)]], "receiver")

  @parameterized.named_parameters(
      ("tiled", False),
      ("skip_tiling", True),
  )
  def test_h2d_into_view_folding_leading_dims_updates_base(self, skip_tiling):
    """H2D into an [E * F, D] view of an [E, F, D] tensor writes the tensor.

    The view is backed by the [E, F, D] storage buffer.

    Args:
      skip_tiling: Whether both synchronizers copy raw device bytes instead of
        detiling on D2H and tiling on H2D.
    """
    generator = torch.Generator().manual_seed(105)
    expected = torch.randn(8 * 64, 256, generator=generator)
    base = torch.zeros(8, 64, 256, dtype=torch.float32, device=self.device)
    ws_source = WeightSynchronizer(
        [[expected.to(self.device)]],
        local_port=0,
        parallelism=1,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_source.shutdown)
    ws_dest = WeightSynchronizer(
        [[base.view(8 * 64, 256)]],
        local_port=0,
        parallelism=1,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_dest.shutdown)
    ws_source.test_only_set_skip_tiling(skip_tiling)
    ws_dest.test_only_set_skip_tiling(skip_tiling)

    ws_source.push_weights([f"127.0.0.1:{ws_dest.local_port}"])
    ws_dest.h2d()

    self._assert_bit_exact([[base]], [[expected.reshape(8, 64, 256)]], "base")
    self.assertEqual(
        ws_dest.get_metrics()["total_tiled_bytes"] == 0, skip_tiling
    )

  def test_rejects_folded_view_with_different_device_layout(self):
    """Rejects a fold whose device bytes differ from a buffer of its shape's.

    An f32 [4, 5, 256] buffer cannot hold its 5-row dimension the way a
    [20, 256] buffer does, since 5 rows are not a whole (8, 128) tile, so raw
    copies of the [20, 256] view would misplace rows.
    """
    base = torch.randn(4, 5, 256).to(self.device)
    # The base itself is accepted, so the rejection below is about the view.
    ws_base = WeightSynchronizer(
        [[base]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_base.shutdown)

    with self.assertRaises(RuntimeError) as raised:
      WeightSynchronizer(
          [[base.view(20, 256)]],
          local_port=0,
          parallelism=1,
          bind_ip="127.0.0.1",
      )

    message = str(raised.exception)
    self.assertIn("folds the leading dimensions", message)
    self.assertIn(
        "does not hold its bytes as a [20, 256] buffer would", message
    )
    self.assertIn("[20, 256]", message)
    self.assertIn("[4, 5, 256]", message)

  def test_rejects_view_that_does_not_fold_leading_dims(self):
    """Rejects a view that regroups more than the leading dimensions."""
    base = torch.randn(4, 8, 256).to(self.device)
    # The base itself is accepted, so the rejection below is about the view.
    ws_base = WeightSynchronizer(
        [[base]], local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    self.addCleanup(ws_base.shutdown)

    with self.assertRaises(RuntimeError) as raised:
      WeightSynchronizer(
          [[base.view(64, 128)]],
          local_port=0,
          parallelism=1,
          bind_ip="127.0.0.1",
      )

    message = str(raised.exception)
    self.assertIn("leading (block) dimension does not match", message)
    self.assertIn("[64, 128]", message)
    self.assertIn("[4, 8, 256]", message)

  def _create_ws(
      self,
      tensors: list[list[torch.Tensor]],
      listener_port: int | None = None,
  ) -> WeightSynchronizer:
    ws = WeightSynchronizer(
        tensors,
        local_port=0,
        parallelism=1,
        listener_port=listener_port,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws.shutdown)
    return ws

  def test_d2h_returns_raiden_future_and_syncs_with_host_buffer_and_h2d(
      self,
  ) -> None:
    """Verifies non-blocking d2h() future API and per-layer consumer waits."""
    expected = _random_weights(torch.float32, num_shards=1, step=31)
    ws = self._create_ws(self._to_device(expected))

    for layer_idx, shards in enumerate(expected):
      nbytes = shards[0].numel() * shards[0].element_size()
      ws.get_host_buffer(layer_idx, 0)[:nbytes].zero_()

    d2h_fut = ws.d2h()
    bound_future_cls = ws_lib.RaidenFuture
    self.assertIsInstance(d2h_fut, bound_future_cls)
    self.assertEqual(type(d2h_fut).__name__, "RaidenFuture")

    def _assert_staged_matches(
        ref_weights: list[list[torch.Tensor]], tag: str
    ) -> None:
      for layer_idx, shards in enumerate(ref_weights):
        cpu_ref = shards[0]
        nbytes = cpu_ref.numel() * cpu_ref.element_size()
        staged = (
            ws.get_host_buffer(layer_idx, 0)[:nbytes]
            .clone()
            .view(torch.float32)
            .reshape(cpu_ref.shape)
        )
        self._assert_bit_exact([[staged]], [[cpu_ref]], f"{tag}_{layer_idx}")

    # Calling get_host_buffer() without d2h_fut.wait() must wait per layer.
    _assert_staged_matches(expected, "get_host_buffer_layer")
    d2h_fut.wait()
    self.assertTrue(d2h_fut.is_ready())
    self.assertTrue(d2h_fut.ok())
    self.assertEqual(d2h_fut.error_message(), "")

    with self.assertRaisesRegex(
        RuntimeError,
        f"layer_idx={ws.num_layers}.*num_layers={ws.num_layers}",
    ):
      ws.get_host_buffer(ws.num_layers, 0)

    # Calling h2d() right after d2h() without .wait() must also wait on D2H.
    updated_expected = _random_weights(torch.float32, num_shards=1, step=32)
    ws.bind_weights(self._to_device(updated_expected))
    d2h_fut_2 = ws.d2h()
    ws.h2d()
    d2h_fut_2.wait()
    self.assertTrue(d2h_fut_2.ok())
    _assert_staged_matches(updated_expected, "h2d_after_async_d2h_layer")

  def _make_pipelined_weights(self, seed_base: int) -> list[list[torch.Tensor]]:
    out: list[list[torch.Tensor]] = []
    for layer_idx, shape in enumerate(_PIPELINED_CONTROLLER_SHAPES):
      gen = torch.Generator().manual_seed(seed_base + layer_idx * 1_009)
      out.append([torch.randn(shape, generator=gen, dtype=torch.float32)])
    return out

  def _register_pipelined_controller_units(
      self,
      ws_src: WeightSynchronizer,
      ws_dst_list: list[WeightSynchronizer],
  ) -> tuple[
      raiden_controller.RaidenController,
      raiden_controller.RaidenId,
      list[raiden_controller.RaidenId],
  ]:
    """Creates a RaidenController and registers 1 trainer + N sampler units."""
    var_protos = [
        raiden_service_pb2.VariableMetadataProto(
            name=f"layer.{layer_idx}.weight",
            shape=list(shape),
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=4,
            layer_idx=layer_idx,
            sharding_spec=["", ""],
            global_shard_indices=[0],
        )
        for layer_idx, shape in enumerate(_PIPELINED_CONTROLLER_SHAPES)
    ]
    controller = raiden_controller.RaidenController(port=0)
    self.addCleanup(controller.worker_rpc_client.close)

    def _reg(
        role: str, idx: int, ws: WeightSynchronizer
    ) -> raiden_controller.RaidenId:
      unit = raiden_controller.RaidenId(role, str(idx), "weights", 0)
      controller.register_work_unit(
          unit,
          shards=[f"127.0.0.1:{ws.local_port}"],
          control_plane_rpc_address=f"127.0.0.1:{ws.listener_port}",
          mesh_shape=[1, 1],
          mesh_axes=["dp", "tp"],
          variables=var_protos,
      )
      return unit

    src_unit = _reg("trainer", 0, ws_src)
    dst_units = [_reg("sampler", i, ws) for i, ws in enumerate(ws_dst_list)]
    return controller, src_unit, dst_units

  def _drive_skip_d2h_transfer(
      self,
      controller: raiden_controller.RaidenController,
      src_unit: raiden_controller.RaidenId,
      dst_units: list[raiden_controller.RaidenId],
      req_id: str,
  ) -> tuple[
      threading.Thread, threading.Event, threading.Event, list[BaseException]
  ]:
    """Starts `start_transfer(..., skip_d2h=True)` on a background loop."""
    started, done = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def _drive() -> None:
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)

      async def _run() -> None:
        fut = controller.start_transfer(
            src_units=[src_unit],
            dst_units=dst_units,
            use_block_chunks=True,
            is_sender=True,
            skip_d2h=True,
            req_id=req_id,
        )
        started.set()
        await asyncio.wait_for(
            fut.wait(), timeout=_CONTROLLER_TRANSFER_TIMEOUT_SECS
        )

      try:
        loop.run_until_complete(_run())
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        errors.append(exc)
      finally:
        started.set()
        done.set()
        asyncio.set_event_loop(None)
        loop.close()

    thread = threading.Thread(target=_drive, daemon=True)
    thread.start()
    self.addCleanup(thread.join, _CONTROLLER_TRANSFER_TIMEOUT_SECS)
    return thread, started, done, errors

  def _join_driver(
      self,
      driver: threading.Thread,
      errors: list[BaseException],
      reraise: bool = True,
  ) -> None:
    driver.join(timeout=_CONTROLLER_TRANSFER_TIMEOUT_SECS)
    self.assertFalse(driver.is_alive(), "H2H transfer hung.")
    if reraise and errors:
      raise errors[0]

  def _assert_samplers_bit_exact(
      self,
      ws_dst_list: list[WeightSynchronizer],
      dst_tensors_by_sampler: list[list[list[torch.Tensor]]],
      expected: list[list[torch.Tensor]],
      tag_prefix: str,
  ) -> None:
    for idx, ws_dst in enumerate(ws_dst_list):
      ws_dst.h2d()
      self._assert_bit_exact(
          dst_tensors_by_sampler[idx], expected, f"{tag_prefix}_{idx}"
      )

  @parameterized.named_parameters(
      ("one_sampler", 1),
      ("two_samplers", 2),
  )
  def test_controller_pipelined_sync_unblocks_after_d2h_and_frees_hbm(
      self, num_samplers: int
  ) -> None:
    """Verifies uuid-free verl flow: async d2h + skip_d2h=True frees HBM."""
    _synced_hbm_bytes()
    expected_step_0 = self._make_pipelined_weights(70_001)
    expected_step_k = self._make_pipelined_weights(75_001)
    src_bytes = sum(s[0].numel() * s[0].element_size() for s in expected_step_0)
    min_expected_release = int(src_bytes * _MIN_HBM_RELEASE_FRACTION)

    dst_tensors_by_sampler = [
        self._zeros_on_device(expected_step_0) for _ in range(num_samplers)
    ]
    ws_dst_list = [
        self._create_ws(t, listener_port=0) for t in dst_tensors_by_sampler
    ]
    mem_before_src_alloc = _synced_hbm_bytes()

    src_tensors_0 = self._to_device(expected_step_0)
    ws_src = self._create_ws(src_tensors_0, listener_port=0)
    ws_src.test_only_set_bandwidth_limit(
        test_only_simulated_egress_gbps=_PIPELINED_EGRESS_GBPS
    )
    baseline_mem = _synced_hbm_bytes()
    self.assertGreaterEqual(baseline_mem - mem_before_src_alloc, src_bytes)

    controller, src_unit, dst_units = self._register_pipelined_controller_units(
        ws_src, ws_dst_list
    )

    # Sync 1 (fresh synchronizer, skip_tiling plan unknown): re-stages bound.
    t0_sync1 = time.monotonic()
    d2h_fut_0 = ws_src.d2h()
    driver_0, started_0, _, errors_0 = self._drive_skip_d2h_transfer(
        controller, src_unit, dst_units, f"verl_sync_1_s{num_samplers}"
    )
    self.assertTrue(started_0.wait(timeout=10.0))
    d2h_fut_0.wait()
    d2h_elapsed_0_s = time.monotonic() - t0_sync1
    self.assertTrue(d2h_fut_0.ok(), d2h_fut_0.error_message())
    self._join_driver(driver_0, errors_0)
    total_transfer_0_s = time.monotonic() - t0_sync1

    metrics_after_0 = ws_src.get_metrics()
    self.assertGreater(int(metrics_after_0["last_detiled_bytes"]), 0)

    del src_tensors_0
    ws_src.unbind_weights()
    mem_after_unbind_0 = _synced_hbm_bytes()
    self.assertGreaterEqual(
        baseline_mem - mem_after_unbind_0, min_expected_release
    )
    self._assert_samplers_bit_exact(
        ws_dst_list, dst_tensors_by_sampler, expected_step_0, "sync_1_sampler"
    )

    # Sync 2 (steady-state verl flow): unbind after D2H while H2H streams.
    step_k_tensors = self._to_device(expected_step_k)
    ws_src.bind_weights(step_k_tensors)
    mem_after_bind_k = _synced_hbm_bytes()
    self.assertGreaterEqual(mem_after_bind_k - mem_after_unbind_0, src_bytes)

    t0_k = time.monotonic()
    d2h_fut_k = ws_src.d2h()
    driver_k, started_k, done_k, errors_k = self._drive_skip_d2h_transfer(
        controller, src_unit, dst_units, f"verl_sync_2_s{num_samplers}"
    )
    self.assertTrue(started_k.wait(timeout=10.0))
    d2h_fut_k.wait()
    d2h_elapsed_k_s = time.monotonic() - t0_k
    self.assertTrue(d2h_fut_k.is_ready())
    self.assertTrue(d2h_fut_k.ok(), d2h_fut_k.error_message())
    self.assertFalse(done_k.is_set(), f"d2h_elapsed_k_s={d2h_elapsed_k_s:.4f}")

    # Negative control: del alone must not free HBM while ws_src stays bound.
    del step_k_tensors
    mem_after_del_before_unbind = _synced_hbm_bytes()
    self.assertLess(
        mem_after_bind_k - mem_after_del_before_unbind, src_bytes // 4
    )

    # Keep `d2h_fut_k` alive across unbind_weights() to verify no held HBM.
    ws_src.unbind_weights()
    poll_deadline = time.monotonic() + _HBM_POLL_TIMEOUT_SECS
    mem_after_unbind_k = _synced_hbm_bytes()
    while (
        mem_after_bind_k - mem_after_unbind_k < min_expected_release
        and time.monotonic() < poll_deadline
    ):
      time.sleep(_HBM_POLL_INTERVAL_SECS)
      mem_after_unbind_k = _synced_hbm_bytes()

    self.assertGreaterEqual(
        mem_after_bind_k - mem_after_unbind_k, min_expected_release
    )
    self.assertFalse(done_k.is_set(), "HBM release must precede H2H end.")
    self.assertTrue(d2h_fut_k.ok())

    step_k_plus_1_tensors, expected_step_k_plus_1 = _run_step_k_plus_1_on_tpu(
        self.device
    )
    self._join_driver(driver_k, errors_k)
    total_transfer_k_s = time.monotonic() - t0_k
    self.assertGreater(total_transfer_k_s, d2h_elapsed_k_s)

    metrics_after_k = ws_src.get_metrics()
    self.assertEqual(int(metrics_after_k["last_detiled_bytes"]), 0)
    self.assertEqual(
        int(metrics_after_k["d2h_call_count"])
        - int(metrics_after_0["d2h_call_count"]),
        1,
    )
    self._assert_samplers_bit_exact(
        ws_dst_list, dst_tensors_by_sampler, expected_step_k, "sync_2_sampler"
    )
    self._assert_bit_exact(
        step_k_plus_1_tensors, expected_step_k_plus_1, "step_k_plus_1_compute"
    )

    timing_str = (
        f"sync1_d2h_s={d2h_elapsed_0_s:.4f}"
        f" sync1_total_s={total_transfer_0_s:.4f}"
        f" sync2_d2h_s={d2h_elapsed_k_s:.4f}"
        f" sync2_total_s={total_transfer_k_s:.4f}"
    )
    logging.info(
        "CONTROLLER_PIPELINED_SYNC_METRICS num_samplers=%d src_bytes=%d %s"
        " mem_after_bind_k=%d mem_after_del=%d mem_after_unbind_k=%d",
        num_samplers,
        src_bytes,
        timing_str,
        mem_after_bind_k,
        mem_after_del_before_unbind,
        mem_after_unbind_k,
    )

  def test_first_sync_unbound_skip_tiling_mismatch_raises(self) -> None:
    """Unbinding before the first transfer fails loudly on layout mismatch."""
    expected = self._make_pipelined_weights(90_001)
    src_tensors = self._to_device(expected)
    dst_tensors = self._zeros_on_device(expected)
    ws_src = self._create_ws(src_tensors, listener_port=0)
    ws_dst = self._create_ws(dst_tensors, listener_port=0)
    controller, src_unit, dst_units = self._register_pipelined_controller_units(
        ws_src, [ws_dst]
    )

    ws_src.d2h().wait()
    ws_src.unbind_weights()

    bad_req_id = "first_sync_unbound_mismatch"
    driver, _, _, errors = self._drive_skip_d2h_transfer(
        controller, src_unit, dst_units, bad_req_id
    )
    self._join_driver(driver, errors, reraise=False)

    plan = controller.get_plan(bad_req_id)
    self.assertIsNotNone(plan)
    assert plan is not None
    self.assertTrue(any(plan.skip_tiling.values()), f"{plan.skip_tiling}")

    self.assertLen(errors, 1)
    err_msg = str(errors[0])
    for token in (
        "weights_0",
        "skip_tiling=false",
        "skip_tiling=true",
        "skip_d2h=false",
        "unbound",
    ):
      self.assertIn(token, err_msg)

    # Positive recovery: rebinding and staging succeeds now that plan is cached.
    ws_src.bind_weights(src_tensors)
    d2h_fut = ws_src.d2h()
    driver_ok, _, _, errors_ok = self._drive_skip_d2h_transfer(
        controller, src_unit, dst_units, "first_sync_rebound_recovery"
    )
    d2h_fut.wait()
    self._join_driver(driver_ok, errors_ok)
    ws_dst.h2d()
    self._assert_bit_exact(dst_tensors, expected, "rebound_after_mismatch")


if __name__ == "__main__":
  absltest.main()

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

"""Single-machine perf regression test for PyTorch weight synchronization."""

import time

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu  # pylint: disable=unused-import

from tpu_sync.api.torch import weight_synchronizer

_NUM_LAYERS = flags.DEFINE_integer(
    "num_decoder_layers",
    26,
    "Number of transformer decoder layers.",
)
_BENCHMARK_ITERATIONS = flags.DEFINE_integer(
    "benchmark_iterations",
    3,
    "Number of benchmark iterations to run.",
)


def _allocate_model_tensors(
    specs: list[tuple[tuple[int, ...], str]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator_fn=torch.zeros,
) -> list[list[torch.Tensor]]:
  """Allocates TPU tensors matching the model specs."""
  return [
      [generator_fn(shape, dtype=dtype, device=device)] for shape, _ in specs
  ]


def get_synthetic_transformer_specs(
    num_layers: int,
) -> list[tuple[tuple[int, ...], str]]:
  """Generates representative transformer parameter specifications."""
  specs = []

  # 1. Embedding
  specs.append(((256128, 2304), "embedder.input_embedding"))

  # 2. Transformer layers (10 tensors per layer)
  for l in range(num_layers):
    specs.append(((2304, 2048), f"layer_{l}.attn.q_proj"))
    specs.append(((2304, 1024), f"layer_{l}.attn.k_proj"))
    specs.append(((2304, 1024), f"layer_{l}.attn.v_proj"))
    specs.append(((2048, 2304), f"layer_{l}.attn.o_proj"))
    specs.append(((2304, 9216), f"layer_{l}.mlp.gate_proj"))
    specs.append(((2304, 9216), f"layer_{l}.mlp.up_proj"))
    specs.append(((9216, 2304), f"layer_{l}.mlp.down_proj"))
    specs.append(((2304,), f"layer_{l}.pre_attn_norm"))
    specs.append(((2304,), f"layer_{l}.post_attn_norm"))
    specs.append(((2304,), f"layer_{l}.pre_ffw_norm"))

  # 3. Final layer norm
  specs.append(((2304,), "final_norm"))
  return specs


class WeightSynchronizationPerfTest(parameterized.TestCase):

  def _verify_tensor_parity(
      self,
      src_tensors: list[list[torch.Tensor]],
      dst_tensors: list[list[torch.Tensor]],
  ):
    """Verifies bit-level numerical equality between source and destination."""
    for l in range(len(src_tensors)):
      for sh in range(len(src_tensors[l])):
        self.assertTrue(
            torch.equal(dst_tensors[l][sh].cpu(), src_tensors[l][sh].cpu()),
            f"Data mismatch at layer {l}, shard {sh}",
        )

  def test_model_specs(self):
    specs = get_synthetic_transformer_specs(num_layers=2)
    # 1 embedder + 2 layers * 10 tensors + 1 final norm = 22 tensors
    self.assertLen(specs, 2 * 10 + 2)

  def test_weight_synchronization_perf(self):
    device = torch.device("tpu")
    num_layers = _NUM_LAYERS.value
    num_iters = _BENCHMARK_ITERATIONS.value

    # 1. Allocate tensors for source (pseudo-random) and destination (zeros)
    torch.manual_seed(42)
    specs = get_synthetic_transformer_specs(num_layers=num_layers)
    src_tensors = _allocate_model_tensors(
        specs, device=device, dtype=torch.float32, generator_fn=torch.randn
    )
    dst_tensors = _allocate_model_tensors(
        specs, device=device, dtype=torch.float32, generator_fn=torch.zeros
    )

    # 2. Instantiate synchronizers
    ws_dest = weight_synchronizer.WeightSynchronizer(
        dst_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    ws_source = weight_synchronizer.WeightSynchronizer(
        src_tensors, local_port=0, parallelism=1, bind_ip="127.0.0.1"
    )
    peer_dest = f"127.0.0.1:{ws_dest.local_port}"

    # 3. Warmup transfer (establishes TCP connections and memory staging)
    logging.info("Executing warmup transfer...")
    ws_source.push_weights([peer_dest])
    ws_dest.h2d()

    # 4. Correctness gate (verify round-trip before measuring any numbers)
    self._verify_tensor_parity(src_tensors, dst_tensors)
    logging.info("Warmup correctness check passed!")

    # 5. Multi-iteration Timed Benchmark Run
    d2h_latencies_ms = []
    h2h_latencies_ms = []
    h2d_latencies_ms = []
    e2e_latencies_ms = []

    logging.info("Executing %d benchmark iterations...", num_iters)
    for it in range(num_iters):
      # Stage 1: Measure standalone D2H staging latency
      t0 = time.perf_counter()
      ws_source.d2h()
      d2h_ms = (time.perf_counter() - t0) * 1000.0

      # Stage 2: Time push_weights (fused D2H + network H2H)
      t1 = time.perf_counter()
      ws_source.push_weights([peer_dest])
      push_ms = (time.perf_counter() - t1) * 1000.0
      h2h_ms = max(push_ms - d2h_ms, 0.0)

      # Stage 3: Time standalone H2D on the destination
      t2 = time.perf_counter()
      ws_dest.h2d()
      h2d_ms = (time.perf_counter() - t2) * 1000.0

      # Total End-to-End time for the pipeline (push_weights + h2d)
      e2e_ms = push_ms + h2d_ms

      d2h_latencies_ms.append(d2h_ms)
      h2h_latencies_ms.append(h2h_ms)
      h2d_latencies_ms.append(h2d_ms)
      e2e_latencies_ms.append(e2e_ms)

      logging.info(
          "Iteration %d/%d: D2H=%.2f ms, H2H=%.2f ms, H2D=%.2f ms, E2E=%.2f ms",
          it + 1,
          num_iters,
          d2h_ms,
          h2h_ms,
          h2d_ms,
          e2e_ms,
      )

    # 6. Summary metrics calculation and reporting
    total_bytes = 0
    for layer in src_tensors:
      for shard in layer:
        total_bytes += shard.numel() * shard.element_size()
    total_gb = total_bytes / 1e9

    med_d2h_ms = float(np.median(d2h_latencies_ms))
    med_h2h_ms = float(np.median(h2h_latencies_ms))
    med_h2d_ms = float(np.median(h2d_latencies_ms))
    med_e2e_ms = float(np.median(e2e_latencies_ms))

    d2h_bw_gbs = total_gb / (med_d2h_ms / 1000.0) if med_d2h_ms > 0 else 0
    h2h_bw_gbs = total_gb / (med_h2h_ms / 1000.0) if med_h2h_ms > 0 else 0
    h2d_bw_gbs = total_gb / (med_h2d_ms / 1000.0) if med_h2d_ms > 0 else 0
    e2e_bw_gbs = total_gb / (med_e2e_ms / 1000.0) if med_e2e_ms > 0 else 0

    logging.info("=" * 80)
    logging.info("END-TO-END WEIGHT SYNCHRONIZATION BENCHMARK RESULTS")
    logging.info("=" * 80)
    logging.info(
        "Model Parameters Payload: %.2f GB (%d bytes)", total_gb, total_bytes
    )
    logging.info("Benchmark Iterations    : %d", num_iters)
    logging.info("-" * 80)
    logging.info(
        "Device-to-Host (D2H)    : %8.2f ms | Throughput: %6.2f GB/s",
        med_d2h_ms,
        d2h_bw_gbs,
    )
    logging.info(
        "Host-to-Host (H2H)      : %8.2f ms | Throughput: %6.2f GB/s",
        med_h2h_ms,
        h2h_bw_gbs,
    )
    logging.info(
        "Host-to-Device (H2D)    : %8.2f ms | Throughput: %6.2f GB/s",
        med_h2d_ms,
        h2d_bw_gbs,
    )
    logging.info(
        "Total Pipeline E2E Time : %8.2f ms | Aggregate : %6.2f GB/s",
        med_e2e_ms,
        e2e_bw_gbs,
    )
    logging.info("=" * 80)

    # 7. Post-benchmark parity verification across all tensors
    self._verify_tensor_parity(src_tensors, dst_tensors)
    logging.info(
        "Post-benchmark numerical parity verified across all %d tensors.",
        len(specs),
    )


if __name__ == "__main__":
  absltest.main()

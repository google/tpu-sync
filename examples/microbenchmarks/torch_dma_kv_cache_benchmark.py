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

"""Single-host raw PCIe DMA microbenchmark for PyTorch KVCacheManager.

Measures D2H and H2D throughput and latency across variable layers, block
counts, and dtypes (bf16, fp32, int32, fp8), comparing against native PyTorch
pinned and standard host baselines, and logs telemetry for golden baseline
comparisons.
"""

import gc
import json
import os
import time

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu

from tpu_sync.api.torch import kv_cache_manager

SUPPORTED_DTYPES = {
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.int32: "int32",
}
if hasattr(torch, "float8_e4m3fn"):
  SUPPORTED_DTYPES[torch.float8_e4m3fn] = "fp8"

FLAGS = flags.FLAGS
flags.DEFINE_string("locality", "default", "Locality under benchmark")
flags.DEFINE_string(
    "telemetry_log_path",
    "/tmp/torch_kv_cache_manager_perf_performance.jsonl",
    "Path to record benchmark telemetry",
)
flags.DEFINE_integer("benchmark_runs", None, "Number of benchmark runs")


def calculate_stats_and_ci(times):
  """Calculates sample mean, sample stddev, and 95% confidence interval."""
  times = np.array(times)
  mean_val = np.mean(times)
  std_val = np.std(times, ddof=1)  # sample standard deviation
  n = len(times)
  if n <= 1:
    return mean_val, 0.0, mean_val, mean_val
  t_table = {
      1: 12.706,
      2: 4.303,
      3: 3.182,
      4: 2.776,
      5: 2.571,
      6: 2.447,
      7: 2.365,
      8: 2.306,
      9: 2.262,
      10: 2.228,
      11: 2.201,
      12: 2.179,
      13: 2.160,
      14: 2.145,
      15: 2.131,
      16: 2.120,
      17: 2.110,
      18: 2.101,
      19: 2.093,
      20: 2.086,
      21: 2.080,
      22: 2.074,
      23: 2.069,
      24: 2.064,
      25: 2.060,
      26: 2.056,
      27: 2.052,
      28: 2.048,
      29: 2.045,
      30: 2.042,
  }
  t_val = t_table.get(n - 1, 1.960) if n - 1 <= 30 else 1.960
  margin_of_error = t_val * (std_val / np.sqrt(n))
  return (
      mean_val,
      std_val,
      mean_val - margin_of_error,
      mean_val + margin_of_error,
  )


def log_telemetry(test_name, dtype, num_layers, shape, d2h_times, h2d_times):
  """Logs telemetry metrics for regression tracking and baseline comparison."""
  if not d2h_times or not h2d_times:
    print(
        "Warning: Telemetry times lists are empty! Skipping log_telemetry"
        " calculations."
    )
    return

  def sample_stddev_ms(times):
    if len(times) <= 1:
      return 0.0
    times_ms = [t * 1000.0 for t in times]
    mean = sum(times_ms) / len(times_ms)
    variance = sum((x - mean) ** 2 for x in times_ms) / (len(times_ms) - 1)
    return float(np.sqrt(variance))

  d2h_mean_ms = sum(d2h_times) * 1000.0 / len(d2h_times)
  h2d_mean_ms = sum(h2d_times) * 1000.0 / len(h2d_times)
  d2h_stddev_ms = sample_stddev_ms(d2h_times)
  h2d_stddev_ms = sample_stddev_ms(h2d_times)

  d2h_median_ms = float(np.median(d2h_times)) * 1000.0
  h2d_median_ms = float(np.median(h2d_times)) * 1000.0

  dtype_str = SUPPORTED_DTYPES.get(dtype, str(dtype))

  record = {
      "test_name": test_name,
      "locality": FLAGS.locality,
      "dtype": dtype_str,
      "num_layers": int(num_layers),
      "shape": [int(s) for s in shape],
      "d2h_latency_mean_ms": float(d2h_mean_ms),
      "d2h_latency_median_ms": float(d2h_median_ms),
      "d2h_latency_stddev_ms": float(d2h_stddev_ms),
      "h2d_latency_mean_ms": float(h2d_mean_ms),
      "h2d_latency_median_ms": float(h2d_median_ms),
      "h2d_latency_stddev_ms": float(h2d_stddev_ms),
      "timestamp": float(time.time()),
  }

  log_dir = os.path.dirname(FLAGS.telemetry_log_path)
  if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

  with open(FLAGS.telemetry_log_path, "a") as f:
    f.write(json.dumps(record) + "\n")


def mutate_tpu_tensor(tensor: torch.Tensor):
  """Mutates tensor in-place to invalidate shadow cache pages."""
  if tensor.dtype == torch.int32:
    tensor.add_(1)
  else:
    tensor.add_(0.01)


class TorchKVCacheManagerPerfTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._require_tpu()
    self.device = torch.device("tpu")

  def tearDown(self):
    for attr in [
        "src_arrs",
        "pinned_host_dst_arrs",
        "pinned_tpu_dst_arrs",
        "std_host_cpu_arrs",
        "std_tpu_dst_arrs",
    ]:
      if hasattr(self, attr):
        delattr(self, attr)
    gc.collect()
    try:
      torch.tpu.synchronize()
    except Exception:  # pylint: disable=broad-except
      pass
    super().tearDown()

  def _require_tpu(self):
    try:
      _ = torch_tpu
      _ = torch.zeros(1, device="tpu")
    except Exception as e:
      self.skipTest(f"This test requires TPU hardware: {e}")

  def _create_tpu_tensors(self, shape, dtype, num_layers):
    """Initializes per-layer contiguous TPU tensors."""
    tensors = []
    for _ in range(num_layers):
      if dtype == torch.int32:
        t = torch.randint(0, 1000, shape, dtype=dtype, device=self.device)
      else:
        t = torch.randn(shape, dtype=dtype, device=self.device)
      tensors.append(t)
    torch.tpu.synchronize()
    return tensors

  @parameterized.named_parameters(
      ("bf16", torch.bfloat16, 64, 16),
      ("f32", torch.float32, 64, 16),
      ("int32_1_layer", torch.int32, 1, 16),
      ("int32_64_layers", torch.int32, 64, 16),
      ("int32_128_layers", torch.int32, 128, 16),
  )
  def test_kv_cache_perf_compare(self, dtype, num_layers, num_blocks):
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")

    # (num_blocks, 128, 8, 2, 128) matching standard model geometry
    shape = (num_blocks, 128, 8, 2, 128)
    self.src_arrs = self._create_tpu_tensors(shape, dtype, num_layers)

    num_iterations = (
        FLAGS.benchmark_runs if FLAGS.benchmark_runs is not None else 10
    )
    print(f"Running PyTorch benchmark with {num_iterations} iterations")

    # 1. Benchmark KVCacheManager (Raiden DMA)
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking"
        " KVCacheManager..."
    )
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=self.src_arrs,
        local_control_port=0,
        host_blocks_to_allocate=num_blocks,
        unsafe_skip_buffer_lock=False,
    )

    d2h_times = []
    h2d_times = []
    offsets = [0]
    sizes = [num_blocks]

    for _ in range(num_iterations):
      for arr in self.src_arrs:
        mutate_tpu_tensor(arr)
      torch.tpu.synchronize()

      gc.disable()
      start = time.time()
      future = manager.d2h(
          src_offsets=offsets,
          dst_offsets=offsets,
          copy_sizes=sizes,
      )
      future.Await()
      torch.tpu.synchronize()
      d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      future = manager.h2d(
          src_offsets=offsets,
          dst_offsets=offsets,
          copy_sizes=sizes,
      )
      future.Await()
      torch.tpu.synchronize()
      h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" d2h avg time: {np.median(d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" h2d avg time: {np.median(h2d_times):.6f} s (median)"
    )

    # 2. Benchmark PyTorch Pinned Host Baseline
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking PyTorch"
        " Pinned Host Baseline..."
    )
    self.pinned_host_dst_arrs = [
        torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)
        for _ in range(num_layers)
    ]
    self.pinned_tpu_dst_arrs = [
        torch.zeros(shape, dtype=dtype, device=self.device)
        for _ in range(num_layers)
    ]
    torch_pinned_d2h_times = []
    torch_pinned_h2d_times = []

    for _ in range(num_iterations):
      for arr in self.src_arrs:
        mutate_tpu_tensor(arr)
      torch.tpu.synchronize()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.pinned_host_dst_arrs[j].copy_(self.src_arrs[j], non_blocking=True)
      torch.tpu.synchronize()
      torch_pinned_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.pinned_tpu_dst_arrs[j].copy_(
            self.pinned_host_dst_arrs[j], non_blocking=True
        )
      torch.tpu.synchronize()
      torch_pinned_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned D2H avg"
        f" time: {np.median(torch_pinned_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned H2D avg"
        f" time: {np.median(torch_pinned_h2d_times):.6f} s (median)"
    )

    # 3. Benchmark PyTorch Standard Baseline (non-pinned CPU)
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking PyTorch"
        " Standard Baseline..."
    )
    self.std_host_cpu_arrs = [
        torch.zeros(shape, dtype=dtype, device="cpu") for _ in range(num_layers)
    ]
    self.std_tpu_dst_arrs = [
        torch.zeros(shape, dtype=dtype, device=self.device)
        for _ in range(num_layers)
    ]
    torch_std_d2h_times = []
    torch_std_h2d_times = []

    for _ in range(num_iterations):
      for arr in self.src_arrs:
        mutate_tpu_tensor(arr)
      torch.tpu.synchronize()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.std_host_cpu_arrs[j].copy_(self.src_arrs[j], non_blocking=False)
      torch.tpu.synchronize()
      torch_std_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.std_tpu_dst_arrs[j].copy_(
            self.std_host_cpu_arrs[j], non_blocking=False
        )
      torch.tpu.synchronize()
      torch_std_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Standard D2H"
        f" avg time: {np.median(torch_std_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Standard H2D"
        f" avg time: {np.median(torch_std_h2d_times):.6f} s (median)"
    )

    # 4. Calculate bandwidth in GB/s
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = np.prod(shape) * element_size * num_layers

    mgr_d2h_bw = total_bytes / np.median(d2h_times) / (1024 * 1024 * 1024)
    mgr_h2d_bw = total_bytes / np.median(h2d_times) / (1024 * 1024 * 1024)

    torch_pinned_d2h_bw = (
        total_bytes / np.median(torch_pinned_d2h_times) / (1024 * 1024 * 1024)
    )
    torch_pinned_h2d_bw = (
        total_bytes / np.median(torch_pinned_h2d_times) / (1024 * 1024 * 1024)
    )
    torch_std_d2h_bw = (
        total_bytes / np.median(torch_std_d2h_times) / (1024 * 1024 * 1024)
    )
    torch_std_h2d_bw = (
        total_bytes / np.median(torch_std_h2d_times) / (1024 * 1024 * 1024)
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" D2H bandwidth: {mgr_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" H2D bandwidth: {mgr_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned D2H"
        f" bandwidth: {torch_pinned_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned H2D"
        f" bandwidth: {torch_pinned_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Standard D2H"
        f" bandwidth: {torch_std_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Standard H2D"
        f" bandwidth: {torch_std_h2d_bw:.3f} GB/s"
    )

    log_telemetry(
        "torch_kv_cache_manager_perf_test",
        dtype,
        num_layers,
        shape,
        d2h_times,
        h2d_times,
    )

  @parameterized.named_parameters(
      ("1_layers_bf16", 1, (8, 128, 1024, 128), torch.bfloat16),
      ("1_layers_fp32", 1, (8, 128, 1024, 128), torch.float32),
      ("1_layers_int32", 1, (8, 128, 1024, 128), torch.int32),
      ("2_layer_int32", 2, (8, 128, 1024, 128), torch.int32),
      ("4_layer_int32", 4, (8, 128, 1024, 128), torch.int32),
      ("8_layer_int32", 8, (8, 128, 1024, 128), torch.int32),
  )
  def test_large_shape_perf_compare(self, num_layers, shape, dtype):
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")

    self.src_arrs = self._create_tpu_tensors(shape, dtype, num_layers)

    num_iterations = (
        FLAGS.benchmark_runs if FLAGS.benchmark_runs is not None else 10
    )
    print(f"Running PyTorch benchmark with {num_iterations} iterations")

    num_blocks = shape[0]
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=self.src_arrs,
        local_control_port=0,
        host_blocks_to_allocate=num_blocks,
        unsafe_skip_buffer_lock=False,
    )

    d2h_times = []
    h2d_times = []
    offsets = [0]
    sizes = [num_blocks]

    for _ in range(num_iterations):
      for arr in self.src_arrs:
        mutate_tpu_tensor(arr)
      torch.tpu.synchronize()

      gc.disable()
      start = time.time()
      future = manager.d2h(
          src_offsets=offsets,
          dst_offsets=offsets,
          copy_sizes=sizes,
      )
      future.Await()
      torch.tpu.synchronize()
      d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      future = manager.h2d(
          src_offsets=offsets,
          dst_offsets=offsets,
          copy_sizes=sizes,
      )
      future.Await()
      torch.tpu.synchronize()
      h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" d2h avg time: {np.median(d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" h2d avg time: {np.median(h2d_times):.6f} s (median)"
    )

    # PyTorch Pinned Host Baseline
    self.pinned_host_dst_arrs = [
        torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)
        for _ in range(num_layers)
    ]
    self.pinned_tpu_dst_arrs = [
        torch.zeros(shape, dtype=dtype, device=self.device)
        for _ in range(num_layers)
    ]
    torch_pinned_d2h_times = []
    torch_pinned_h2d_times = []

    for _ in range(num_iterations):
      for arr in self.src_arrs:
        mutate_tpu_tensor(arr)
      torch.tpu.synchronize()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.pinned_host_dst_arrs[j].copy_(self.src_arrs[j], non_blocking=True)
      torch.tpu.synchronize()
      torch_pinned_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.pinned_tpu_dst_arrs[j].copy_(
            self.pinned_host_dst_arrs[j], non_blocking=True
        )
      torch.tpu.synchronize()
      torch_pinned_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned D2H avg"
        f" time: {np.median(torch_pinned_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned H2D avg"
        f" time: {np.median(torch_pinned_h2d_times):.6f} s (median)"
    )

    # PyTorch Standard Baseline
    self.std_host_cpu_arrs = [
        torch.zeros(shape, dtype=dtype, device="cpu") for _ in range(num_layers)
    ]
    self.std_tpu_dst_arrs = [
        torch.zeros(shape, dtype=dtype, device=self.device)
        for _ in range(num_layers)
    ]
    torch_std_d2h_times = []
    torch_std_h2d_times = []

    for _ in range(num_iterations):
      for arr in self.src_arrs:
        mutate_tpu_tensor(arr)
      torch.tpu.synchronize()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.std_host_cpu_arrs[j].copy_(self.src_arrs[j], non_blocking=False)
      torch.tpu.synchronize()
      torch_std_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      for j in range(num_layers):
        self.std_tpu_dst_arrs[j].copy_(
            self.std_host_cpu_arrs[j], non_blocking=False
        )
      torch.tpu.synchronize()
      torch_std_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    # Bandwidth
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = np.prod(shape) * element_size * num_layers

    mgr_d2h_bw = total_bytes / np.median(d2h_times) / (1024 * 1024 * 1024)
    mgr_h2d_bw = total_bytes / np.median(h2d_times) / (1024 * 1024 * 1024)

    torch_pinned_d2h_bw = (
        total_bytes / np.median(torch_pinned_d2h_times) / (1024 * 1024 * 1024)
    )
    torch_pinned_h2d_bw = (
        total_bytes / np.median(torch_pinned_h2d_times) / (1024 * 1024 * 1024)
    )
    torch_std_d2h_bw = (
        total_bytes / np.median(torch_std_d2h_times) / (1024 * 1024 * 1024)
    )
    torch_std_h2d_bw = (
        total_bytes / np.median(torch_std_h2d_times) / (1024 * 1024 * 1024)
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" D2H bandwidth: {mgr_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" H2D bandwidth: {mgr_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned D2H"
        f" bandwidth: {torch_pinned_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Pinned H2D"
        f" bandwidth: {torch_pinned_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Standard D2H"
        f" bandwidth: {torch_std_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] PyTorch Standard H2D"
        f" bandwidth: {torch_std_h2d_bw:.3f} GB/s"
    )

    log_telemetry(
        "torch_kv_cache_manager_perf_test_large_shape",
        dtype,
        num_layers,
        shape,
        d2h_times,
        h2d_times,
    )


if __name__ == "__main__":
  absltest.main()

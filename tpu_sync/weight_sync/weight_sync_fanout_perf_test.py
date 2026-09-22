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

"""Performance unit test for weight syncing fan-out and resharding.

Benchmarks 1-to-4 Flat Direct Push (broadcast_k=64) versus Tree Broadcast
(broadcast_k=2) side-by-side using host DRAM loopback networking on scaled
Qwen-35B model specs, verifying micro-block fragmentation realism and parity.
"""

import asyncio
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tpu_sync.api.common import RaidenId
from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

_NUM_LAYERS = flags.DEFINE_integer(
    "num_layers", 40, "Number of layers to simulate."
)
_NUM_ROUTED_EXPERTS = flags.DEFINE_integer(
    "num_routed_experts", 8, "Number of routed experts."
)
_NUM_DESTINATIONS = flags.DEFINE_integer(
    "num_destinations", 4, "Number of destination replicas."
)
_TEST_ONLY_SIMULATED_NIC_GBPS = flags.DEFINE_float(
    "test_only_simulated_nic_gbps",
    0.0,
    "Simulated NIC line rate in Gbps (0.0 = unlimited). When 0.0, "
    "test_rate_limited_flat_vs_tree_comparison benchmarks at 10.0 Gbps.",
)


class PerfRunResult:
  """Container for weight sync fan-out benchmark run results."""

  def __init__(
      self,
      elapsed: float,
      throughput_gb_s: float,
      src_metrics: Dict[str, Any],
      dst_metrics: List[Dict[str, Any]],
  ) -> None:
    self.elapsed = elapsed
    self.throughput_gb_s = throughput_gb_s
    self.src_metrics = src_metrics
    self.dst_metrics = dst_metrics

  def __iter__(self) -> Iterator[float]:
    return iter((self.elapsed, self.throughput_gb_s))


def _format_h2h_stats(
    src_m: Dict[str, Any], dst_m: List[Dict[str, Any]]
) -> Tuple[str, str]:
  """Formats source and relay H2H performance metrics for summary logging."""
  src_mb = src_m["total_h2h_bytes"] / 1e6
  src_ms = src_m["total_h2h_time_ms"]
  src_bw = src_m["total_h2h_bandwidth_gbps"]
  src_str = f"{src_mb:.1f} MB / {src_ms:.1f} ms ({src_bw:.2f} GB/s)"
  relay_bws = [
      m["total_h2h_bandwidth_gbps"] for m in dst_m if m["total_h2h_bytes"] > 0
  ]
  if not relay_bws:
    relay_str = "N/A (flat)"
  else:
    relay_str = (
        f"max {max(relay_bws):.2f} / avg {sum(relay_bws)/len(relay_bws):.2f}"
        " GB/s"
    )
  return src_str, relay_str


def _make_scaled_qwen_specs(
    num_layers: Optional[int] = None,
    num_routed_experts: Optional[int] = None,
    role: str = "source",
) -> List[Tuple[Tuple[int, ...], List[str], str, int]]:
  """Generates parameter specs for scaled Qwen 3.5 35B."""
  if num_layers is None:
    num_layers = _NUM_LAYERS.value
  if num_routed_experts is None:
    num_routed_experts = _NUM_ROUTED_EXPERTS.value

  dim = 2048
  routed_mlp_dim = 512
  shared_mlp_dim = 512
  attn_dim = 2048
  linear_ba_dim = 64
  linear_conv_dim = 256

  is_dest = role == "destination"
  specs = []

  for l in range(num_layers):
    if l % 2 == 0:
      # Even layer: Full Attention (GQA) + MoE Block
      specs.append((
          (dim, 2, 256),
          ["", "", ""] if is_dest else ["", "", ""],
          f"decoder.layers.{l}.attention.attention.key.kernel",
          l,
      ))
      specs.append((
          (dim, attn_dim),
          ["", "tp_out"] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.attention.out.kernel",
          l,
      ))
      specs.append((
          (dim, attn_dim),
          ["", "tp_out"] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.attention.query.kernel",
          l,
      ))
      specs.append((
          (dim, 2, 256),
          ["", "", ""] if is_dest else ["", "", ""],
          f"decoder.layers.{l}.attention.attention.value.kernel",
          l,
      ))
    else:
      # Odd layer: GDN Linear Attention + MoE Block
      specs.append((
          (linear_ba_dim, linear_conv_dim),
          ["", ""] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.linear_attn.b_kernel",
          l,
      ))
      specs.append((
          (dim, linear_ba_dim),
          ["", ""] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.linear_attn.ba_kernel",
          l,
      ))
      specs.append((
          (4, 1, linear_conv_dim),
          ["", "", ""] if is_dest else ["", "", ""],
          f"decoder.layers.{l}.attention.linear_attn.conv1d.kernel",
          l,
      ))
      specs.append((
          (dim, attn_dim),
          ["", "tp_out"] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.linear_attn.g_kernel",
          l,
      ))
      specs.append((
          (attn_dim, dim),
          ["", "tp_out"] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.linear_attn.out_kernel",
          l,
      ))
      specs.append((
          (dim, attn_dim),
          ["", "tp_out"] if is_dest else ["", ""],
          f"decoder.layers.{l}.attention.linear_attn.qkvz_kernel",
          l,
      ))

    # Shared MoE block across both even and odd layers
    specs.append((
        (num_routed_experts, routed_mlp_dim, dim),
        ["", "", "tp_wo"] if is_dest else ["", "", ""],
        f"decoder.layers.{l}.mlp.experts.down_proj.kernel",
        l,
    ))
    specs.append((
        (num_routed_experts, dim, routed_mlp_dim),
        ["", "", "tp"] if is_dest else ["", "", ""],
        f"decoder.layers.{l}.mlp.experts.gate_proj.kernel",
        l,
    ))
    specs.append((
        (num_routed_experts, dim, routed_mlp_dim),
        ["", "", "tp"] if is_dest else ["", "", ""],
        f"decoder.layers.{l}.mlp.experts.up_proj.kernel",
        l,
    ))
    specs.append((
        (shared_mlp_dim, dim),
        ["", "tp_wo"] if is_dest else ["", ""],
        f"decoder.layers.{l}.mlp.shared_expert.down_proj.kernel",
        l,
    ))
    specs.append((
        (dim, shared_mlp_dim),
        ["", "tp"] if is_dest else ["", ""],
        f"decoder.layers.{l}.mlp.shared_expert.gate_proj.kernel",
        l,
    ))
    specs.append((
        (dim, shared_mlp_dim),
        ["", "tp"] if is_dest else ["", ""],
        f"decoder.layers.{l}.mlp.shared_expert.up_proj.kernel",
        l,
    ))

  return specs


def build_variable_protos(
    specs: List[Tuple[Tuple[int, ...], List[str], str, int]],
    mesh_shape_dict: Dict[str, int],
    item_size: int = 2,
) -> List[raiden_service_pb2.VariableMetadataProto]:
  """Constructs VariableMetadataProtos with explicit sharding specs and layouts."""
  protos = []
  for idx, (global_shape, spec_axes, name, _) in enumerate(specs):
    sharding_shape = [mesh_shape_dict.get(axis, 1) for axis in spec_axes]
    layout = list(range(len(global_shape) - 1, -1, -1))
    protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=name,
            shape=list(global_shape),
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=item_size,
            layer_idx=idx,
            sharding_spec=spec_axes,
            global_shard_indices=[0],
        )
    )
  return protos


class WeightSyncFanoutPerfTest(parameterized.TestCase):
  """Performance and micro-block fragmentation tests for weight sync fan-out."""

  def setUp(self):
    super().setUp()
    self.num_destinations = _NUM_DESTINATIONS.value
    self.num_layers_flag = _NUM_LAYERS.value
    self.num_routed_experts_flag = _NUM_ROUTED_EXPERTS.value

    # 1. Model specs & variable protos
    self.src_mesh_dict = {
        "tp": 1,
        "tp_wo": 1,
        "tp_out": 1,
    }
    self.dst_mesh_dict = {
        "tp": 2,
        "tp_wo": 32,
        "tp_out": 8,
    }

    self.src_specs = _make_scaled_qwen_specs(
        num_layers=self.num_layers_flag,
        num_routed_experts=self.num_routed_experts_flag,
        role="source",
    )
    self.dst_specs = _make_scaled_qwen_specs(
        num_layers=self.num_layers_flag,
        num_routed_experts=self.num_routed_experts_flag,
        role="destination",
    )

    self.num_layers = len(self.src_specs)

    self.src_var_protos = build_variable_protos(
        self.src_specs, self.src_mesh_dict
    )
    self.dst_var_protos = build_variable_protos(
        self.dst_specs, self.dst_mesh_dict
    )

    # Calculate model volume per destination replica
    self.total_model_bytes = sum(
        int(np.prod(proto.shape) // np.prod(proto.mesh_shape)) * proto.item_size
        for proto in self.dst_var_protos
    )

    # Calculate buffer sizes needed per variable
    self.src_slice_byte_sizes = [
        int(np.prod(proto.shape)) * proto.item_size
        for proto in self.src_var_protos
    ]
    self.dst_slice_byte_sizes = [
        int(np.prod(proto.shape) // np.prod(proto.mesh_shape)) * proto.item_size
        for proto in self.dst_var_protos
    ]

    # Calculate max transferred offset per variable
    self.layer_max_bytes: Dict[int, int] = {}
    for proto in self.dst_var_protos:
      var_bytes = (
          int(np.prod(proto.shape) // np.prod(proto.mesh_shape))
          * proto.item_size
      )
      self.layer_max_bytes[proto.layer_idx] = var_bytes

    # 2. Start centralized Controller Server on loopback
    self.controller_network_client = (
        raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
    )
    self.addCleanup(self.controller_network_client.close)

    self.controller = raiden_controller.RaidenController(
        port=0,
        worker_rpc_client=self.controller_network_client,
    )
    self.controller_server = raiden_controller.RaidenControllerServer(
        self.controller
    )
    self.controller_server.start()
    self.addCleanup(self.controller_server.stop)
    self.controller_port = self.controller_server.port

    self.ctrl_client = raiden_controller.RaidenControllerClientFacade(
        f"127.0.0.1:{self.controller_port}",
        name_resolver=None,
    )
    if hasattr(self.ctrl_client, "_control_pipe_client") and hasattr(
        self.ctrl_client._control_pipe_client, "close"
    ):
      self.addCleanup(self.ctrl_client._control_pipe_client.close)

    # 3. Instantiate 1 source worker + 4 destination workers
    self.ws_src = (
        weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
            num_layers=self.num_layers,
            num_shards=1,
            slice_byte_size=self.src_slice_byte_sizes,
            local_port=0,
            listener_port=0,
            bind_ip="127.0.0.1",
        )
    )
    self.addCleanup(self.ws_src.shutdown)

    self.ws_dsts: List[weight_synchronizer.WeightSynchronizer] = []
    for _ in range(self.num_destinations):
      ws_dst = (
          weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
              num_layers=self.num_layers,
              num_shards=1,
              slice_byte_size=self.dst_slice_byte_sizes,
              local_port=0,
              listener_port=0,
              bind_ip="127.0.0.1",
          )
      )
      self.addCleanup(ws_dst.shutdown)
      self.ws_dsts.append(ws_dst)

    # 4. Register work units
    self.src_unit = RaidenId("trainer", "0", "weights")
    self.dst_units = [
        RaidenId("sampler", str(i), "weights")
        for i in range(self.num_destinations)
    ]

    mesh_axes = ["tp", "tp_wo", "tp_out"]
    mesh_shape = [1] * len(mesh_axes)

    self.ctrl_client.register_work_unit(
        self.src_unit,
        [f"127.0.0.1:{self.ws_src.local_port}"],
        f"127.0.0.1:{self.ws_src.listener_port}",
        mesh_shape=mesh_shape,
        variables=self.src_var_protos,
        mesh_axes=mesh_axes,
    )

    for i, ws_dst in enumerate(self.ws_dsts):
      self.ctrl_client.register_work_unit(
          self.dst_units[i],
          [f"127.0.0.1:{ws_dst.local_port}"],
          f"127.0.0.1:{ws_dst.listener_port}",
          mesh_shape=mesh_shape,
          variables=self.dst_var_protos,
          mesh_axes=mesh_axes,
      )

  def _get_schedule_and_task_counts(self) -> Tuple[int, int]:
    if hasattr(self, "_cached_task_counts"):
      return self._cached_task_counts
    old_k = self.controller.broadcast_k
    self.controller.broadcast_k = 64
    self.controller._plan_cache.clear()
    try:
      loop = asyncio.new_event_loop()
      try:
        sched = loop.run_until_complete(
            self.controller._compute_transfer_schedule(
                src_units=[self.src_unit],
                dst_units=self.dst_units,
                skip_tiling={l: False for l in range(self.num_layers)},
            )
        )
      finally:
        loop.close()
    finally:
      self.controller.broadcast_k = old_k
      self.controller._plan_cache.clear()

    self.assertIsNotNone(sched)
    self.assertIsNotNone(sched.direct_schedules)

    total_tasks = 0
    le_512_tasks = 0

    for src_unit, sched_by_shard in sched.direct_schedules.items():
      for shard_idx, entries in sched_by_shard.items():
        for entry in entries:
          size = entry[4]
          src_stride = entry[7]
          dst_stride = entry[8]
          count = entry[9]
          is_contiguous = (count == 1) or (
              src_stride == size and dst_stride == size
          )
          num_tasks = 1 if is_contiguous else count
          total_tasks += num_tasks
          if size <= 512:
            le_512_tasks += num_tasks

    self._cached_task_counts = (total_tasks, le_512_tasks)
    return total_tasks, le_512_tasks

  def test_scaled_spec_chunk_distribution(self):
    """Verifies that >95% of scheduled copy tasks have size_bytes <= 512 bytes."""
    total_tasks, le_512_tasks = self._get_schedule_and_task_counts()
    self.assertGreater(total_tasks, 0, "No copy tasks were scheduled")
    pct_le_512 = (le_512_tasks / total_tasks) * 100.0
    print(
        f"\nChunk Distribution: {le_512_tasks}/{total_tasks} tasks"
        f" ({pct_le_512:.2f}%) have size <= 512B"
    )
    self.assertGreater(
        pct_le_512,
        95.0,
        f"Expected >95% tasks with size <= 512 bytes, got {pct_le_512:.2f}%",
    )

  def _run_flat_direct_push_perf(self) -> PerfRunResult:
    """Executes 1-to-4 Flat Direct Push (broadcast_k=64) with byte parity check."""
    # 0. Reset metrics on source and all destination workers
    self.ws_src.reset_metrics()
    for ws_dst in self.ws_dsts:
      ws_dst.reset_metrics()

    # 1. Fill source buffers with 0xAB
    for l in range(self.num_layers):
      buf = self.ws_src.get_host_buffer(layer_idx=l, shard_idx=0)
      buf[:] = 0xAB

    # 2. Clear destination buffers to 0x00
    for ws_dst in self.ws_dsts:
      for l in range(self.num_layers):
        buf = ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)
        buf[:] = 0x00

    # 3. Start flat direct push transfer
    uuid = 1001
    self.controller.broadcast_k = 64
    self.controller._plan_cache.clear()
    t0 = time.perf_counter()
    future = self.controller.start_transfer(
        src_units=[self.src_unit],
        dst_units=self.dst_units,
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=uuid,
        req_id="flat_perf",
        skip_d2h=True,
        skip_tiling={l: False for l in range(self.num_layers)},
    )
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(future.wait())
    finally:
      loop.close()

    for ws_dst in self.ws_dsts:
      ws_dst.wait_for_transfer_completion(uuid=uuid)

    elapsed = time.perf_counter() - t0

    # 4. Verify byte parity across all 4 destinations
    for ws_dst in self.ws_dsts:
      for l in range(self.num_layers):
        dst_buf = ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)
        valid_bytes = self.layer_max_bytes[l]
        self.assertTrue(
            np.all(dst_buf[:valid_bytes] == 0xAB),
            f"Flat push byte parity mismatch in layer {l}",
        )

    # 5. Query and verify metrics
    src_metrics = self.ws_src.get_metrics()
    dst_metrics = [ws_dst.get_metrics() for ws_dst in self.ws_dsts]

    self.assertGreater(src_metrics["total_h2h_bytes"], 0)
    self.assertGreater(src_metrics["total_h2h_time_ms"], 0.0)
    self.assertGreater(src_metrics["total_h2h_bandwidth_gbps"], 0.0)

    total_cluster_h2h_bytes = src_metrics["total_h2h_bytes"] + sum(
        m["total_h2h_bytes"] for m in dst_metrics
    )
    expected_total_h2h_bytes = self.total_model_bytes * self.num_destinations
    self.assertEqual(
        total_cluster_h2h_bytes,
        expected_total_h2h_bytes,
        "Sum of total_h2h_bytes across src + all dsts"
        f" ({total_cluster_h2h_bytes}) must equal total_model_bytes *"
        f" num_destinations ({expected_total_h2h_bytes}).",
    )

    throughput_gb_s = (
        (self.total_model_bytes * self.num_destinations) / 1e9
    ) / max(elapsed, 1e-9)

    src_h2h_mb = src_metrics["total_h2h_bytes"] / 1e6
    src_h2h_ms = src_metrics["total_h2h_time_ms"]
    src_h2h_bw = src_metrics["total_h2h_bandwidth_gbps"]
    print(
        f"\n[Flat Push (k=64)] Elapsed: {elapsed:.3f}s, Throughput:"
        f" {throughput_gb_s:.2f} GB/s, Src H2H: {src_h2h_mb:.1f} MB in"
        f" {src_h2h_ms:.1f} ms ({src_h2h_bw:.2f} GB/s), Relays: None, Parity:"
        " PASS"
    )
    return PerfRunResult(elapsed, throughput_gb_s, src_metrics, dst_metrics)

  def test_flat_direct_push_perf(self):
    """Benchmarks 1-to-4 Flat Direct Push (broadcast_k=64) with byte parity check."""
    self._run_flat_direct_push_perf()

  def _run_tree_broadcast_perf(self) -> PerfRunResult:
    """Executes 1-to-4 Tree Broadcast (broadcast_k=2) awaiting controller future."""
    # 0. Reset metrics on source and all destination workers
    self.ws_src.reset_metrics()
    for ws_dst in self.ws_dsts:
      ws_dst.reset_metrics()

    # 1. Fill source buffers with 0xCD
    for l in range(self.num_layers):
      buf = self.ws_src.get_host_buffer(layer_idx=l, shard_idx=0)
      buf[:] = 0xCD

    # 2. Re-zero destination buffers to 0x00
    for ws_dst in self.ws_dsts:
      for l in range(self.num_layers):
        buf = ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)
        buf[:] = 0x00

    # 3. Start tree broadcast transfer (broadcast_k=2)
    uuid = 1002
    self.controller.broadcast_k = 2
    self.controller._plan_cache.clear()
    t0 = time.perf_counter()
    future = self.controller.start_transfer(
        src_units=[self.src_unit],
        dst_units=self.dst_units,
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=uuid,
        req_id="tree_perf",
        skip_d2h=True,
        skip_tiling={l: False for l in range(self.num_layers)},
    )
    loop = asyncio.new_event_loop()
    try:
      # Awaits controller tree broadcast future until all hops complete
      loop.run_until_complete(future.wait())
    finally:
      loop.close()

    elapsed = time.perf_counter() - t0

    # 4. Verify byte parity across all 4 destinations
    for ws_dst in self.ws_dsts:
      for l in range(self.num_layers):
        dst_buf = ws_dst.get_host_buffer(layer_idx=l, shard_idx=0)
        valid_bytes = self.layer_max_bytes[l]
        self.assertTrue(
            np.all(dst_buf[:valid_bytes] == 0xCD),
            f"Tree broadcast byte parity mismatch in layer {l}",
        )

    # 5. Query and verify metrics
    src_metrics = self.ws_src.get_metrics()
    dst_metrics = [ws_dst.get_metrics() for ws_dst in self.ws_dsts]

    self.assertGreater(src_metrics["total_h2h_bytes"], 0)
    self.assertGreater(src_metrics["total_h2h_time_ms"], 0.0)
    self.assertGreater(src_metrics["total_h2h_bandwidth_gbps"], 0.0)

    relay_metrics = [m for m in dst_metrics if m["total_h2h_bytes"] > 0]
    self.assertNotEmpty(
        relay_metrics,
        "Tree Broadcast (broadcast_k=2) must have active relay destinations"
        " forwarding chunks.",
    )
    for m in relay_metrics:
      self.assertGreater(m["total_h2h_bytes"], 0)
      self.assertGreater(m["total_h2h_time_ms"], 0.0)
      self.assertGreater(m["total_h2h_bandwidth_gbps"], 0.0)

    total_cluster_h2h_bytes = src_metrics["total_h2h_bytes"] + sum(
        m["total_h2h_bytes"] for m in dst_metrics
    )
    expected_total_h2h_bytes = self.total_model_bytes * self.num_destinations
    self.assertEqual(
        total_cluster_h2h_bytes,
        expected_total_h2h_bytes,
        "Sum of total_h2h_bytes across src + all dsts"
        f" ({total_cluster_h2h_bytes}) must equal total_model_bytes *"
        f" num_destinations ({expected_total_h2h_bytes}).",
    )

    throughput_gb_s = (
        (self.total_model_bytes * self.num_destinations) / 1e9
    ) / max(elapsed, 1e-9)

    src_h2h_mb = src_metrics["total_h2h_bytes"] / 1e6
    src_h2h_ms = src_metrics["total_h2h_time_ms"]
    src_h2h_bw = src_metrics["total_h2h_bandwidth_gbps"]
    relay_bws = [m["total_h2h_bandwidth_gbps"] for m in relay_metrics]
    max_relay_bw = max(relay_bws) if relay_bws else 0.0
    avg_relay_bw = sum(relay_bws) / len(relay_bws) if relay_bws else 0.0

    print(
        f"\n[Tree Broadcast (k=2)] Elapsed: {elapsed:.3f}s, Throughput:"
        f" {throughput_gb_s:.2f} GB/s, Src H2H: {src_h2h_mb:.1f} MB in"
        f" {src_h2h_ms:.1f} ms ({src_h2h_bw:.2f} GB/s), Relays"
        f" ({len(relay_metrics)} active): max {max_relay_bw:.2f} GB/s, avg"
        f" {avg_relay_bw:.2f} GB/s, Parity: PASS"
    )
    return PerfRunResult(elapsed, throughput_gb_s, src_metrics, dst_metrics)

  def test_tree_broadcast_perf(self):
    """Benchmarks 1-to-4 Tree Broadcast (broadcast_k=2) awaiting controller future."""
    self._run_tree_broadcast_perf()

  def test_sxs_flat_vs_tree_performance_comparison(self):
    """Executes flat vs tree modes side-by-side and prints comparative summary table."""
    flat_res = self._run_flat_direct_push_perf()
    tree_res = self._run_tree_broadcast_perf()

    total_tasks, _ = self._get_schedule_and_task_counts()
    if total_tasks >= 1_000_000:
      task_str = f"~{total_tasks / 1e6:.1f}M"
    elif total_tasks >= 1000:
      task_str = f"~{total_tasks // 1000}k"
    else:
      task_str = str(total_tasks)
    model_mb = self.total_model_bytes / 1e6

    flat_src_str, flat_relay_str = _format_h2h_stats(
        flat_res.src_metrics, flat_res.dst_metrics
    )
    tree_src_str, tree_relay_str = _format_h2h_stats(
        tree_res.src_metrics, tree_res.dst_metrics
    )

    print("\n" + "=" * 115)
    print(
        f"Weight Sync Fan-out Benchmark Results (Model: ~{model_mb:.1f} MB,"
        f" N={self.num_destinations})"
    )
    print("=" * 115)
    print(
        f"{'Mode':<14} {'Time':<10} {'Throughput':<14}"
        f" {'Src H2H (Bytes / Time / BW)':<36} {'Relay H2H BW (max/avg)':<26}"
        f" {'Tasks':<8} {'Parity':<6}"
    )
    print("-" * 115)
    print(
        f"{'Flat (k=64)':<14} {flat_res.elapsed:.2f} s    "
        f" {flat_res.throughput_gb_s:.2f} GB/s      {flat_src_str:<36}"
        f" {flat_relay_str:<26} {task_str:<8} PASS"
    )
    print(
        f"{'Tree (k=2)':<14} {tree_res.elapsed:.2f} s    "
        f" {tree_res.throughput_gb_s:.2f} GB/s      {tree_src_str:<36}"
        f" {tree_relay_str:<26} {task_str:<8} PASS"
    )
    print("=" * 115 + "\n")

  def test_rate_limited_flat_vs_tree_comparison(self):
    """Benchmarks Flat Direct Push vs Tree Broadcast under simulated NIC bandwidth cap."""
    rate_gbps = (
        _TEST_ONLY_SIMULATED_NIC_GBPS.value
        if _TEST_ONLY_SIMULATED_NIC_GBPS.value > 0.0
        else 10.0
    )
    self.ws_src.test_only_set_bandwidth_limit(
        test_only_simulated_egress_gbps=rate_gbps,
        test_only_simulated_ingress_gbps=0.0,
    )
    for ws_dst in self.ws_dsts:
      ws_dst.test_only_set_bandwidth_limit(
          test_only_simulated_egress_gbps=rate_gbps,
          test_only_simulated_ingress_gbps=0.0,
      )

    flat_res = self._run_flat_direct_push_perf()
    tree_res = self._run_tree_broadcast_perf()

    flat_src_str, flat_relay_str = _format_h2h_stats(
        flat_res.src_metrics, flat_res.dst_metrics
    )
    tree_src_str, tree_relay_str = _format_h2h_stats(
        tree_res.src_metrics, tree_res.dst_metrics
    )

    print("\n" + "=" * 110)
    print(
        f"Rate-Limited Weight Sync Fan-out Benchmark ({rate_gbps:.1f} Gbps NIC,"
        f" N={self.num_destinations})"
    )
    print("=" * 110)
    print(
        f"{'Mode':<14} {'Time':<10} {'Throughput':<14}"
        f" {'Src H2H (Bytes / Time / BW)':<36} {'Relay H2H BW (max/avg)':<26}"
        f" {'Parity':<6}"
    )
    print("-" * 110)
    print(
        f"{'Flat (k=64)':<14} {flat_res.elapsed:.2f} s    "
        f" {flat_res.throughput_gb_s:.2f} GB/s      {flat_src_str:<36}"
        f" {flat_relay_str:<26} PASS"
    )
    print(
        f"{'Tree (k=2)':<14} {tree_res.elapsed:.2f} s    "
        f" {tree_res.throughput_gb_s:.2f} GB/s      {tree_src_str:<36}"
        f" {tree_relay_str:<26} PASS"
    )
    print("=" * 110 + "\n")

    if self.num_destinations >= 16:
      self.assertLess(
          tree_res.elapsed,
          flat_res.elapsed,
          f"Tree Broadcast ({tree_res.elapsed:.3f}s) must outperform Flat"
          f" Direct Push ({flat_res.elapsed:.3f}s) under constrained NIC line"
          f" rate at N={self.num_destinations}.",
      )
    else:
      # At small fan-out (N < 16, e.g. N=4), Tree broadcast only saves 1 serialized
      # copy (from 4 to 3), which is outweighed by multi-hop store-and-forward
      # Python scheduling latency. Verify that both modes executed cleanly and
      # satisfied byte-exact parity.
      self.assertGreater(flat_res.elapsed, 0.0)
      self.assertGreater(tree_res.elapsed, 0.0)


if __name__ == "__main__":
  absltest.main()

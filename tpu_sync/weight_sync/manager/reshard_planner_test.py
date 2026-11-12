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

"""Tests and benchmarks for ReshardPlanner schedule computation and deduplication."""

import collections
import threading
import timeit
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity
from tpu_sync.weight_sync.manager import reshard_planner


def _build_qwen3_397b_variables(
    num_layers: int,
    src_fsdp: int,
    is_src: bool,
) -> list[controller_types.VariableMetadata]:
  """Builds Qwen3.5-397B-A17B variable metadata list for |num_layers| layers."""
  variables = []
  layer_idx = 0

  def add_var(
      name: str,
      shape: list[int],
      src_mesh: list[int],
      dst_mesh: list[int],
      layout: list[int],
      src_spec: list[str],
      dst_spec: list[str],
  ):
    nonlocal layer_idx
    mesh = src_mesh if is_src else dst_mesh
    spec = src_spec if is_src else dst_spec
    variables.append(
        controller_types.VariableMetadata(
            name=name,
            shape=shape,
            mesh_shape=mesh,
            layout=layout,
            item_size=2,
            layer_idx=layer_idx,
            sharding_spec=spec,
        )
    )
    layer_idx += 1

  # Embeddings + final norm + lm_head (3 vars)
  add_var(
      "embed_tokens",
      [151936, 4096],
      [src_fsdp, 8],
      [1, 1],
      [1, 0],
      ["fsdp", "context,expert"],
      ["", ""],
  )
  add_var("final_norm", [4096], [1], [1], [0], [""], [""])
  add_var(
      "lm_head",
      [4096, 151936],
      [8, src_fsdp],
      [1, 1],
      [1, 0],
      ["context,expert", "fsdp"],
      ["", ""],
  )

  for l in range(num_layers):
    # 8 2D projections per layer
    for j in range(8):
      add_var(
          f"layer_{l}_2d_{j}",
          [4096, 4096],
          [src_fsdp, 8],
          [1, 1],
          [1, 0],
          ["fsdp", "context,expert"],
          ["", ""],
      )
    # 3D MoE weights (gate_up_proj + down_proj)
    add_var(
        f"layer_{l}_moe_gate_up",
        [512, 4096, 2048],
        [8, src_fsdp, 1],
        [16, 1, 1],
        [2, 1, 0],
        ["context,expert", "fsdp", ""],
        ["x,y", "", ""],
    )
    add_var(
        f"layer_{l}_moe_down",
        [512, 1024, 4096],
        [8, 1, src_fsdp],
        [16, 1, 1],
        [2, 1, 0],
        ["context,expert", "", "fsdp"],
        ["x,y", "", ""],
    )
    # 1D norms/gates per layer (6 for GatedDeltaNet layers 0..44,
    # 5 for Attention layers 45..59)
    num_1d = 6 if l < 45 else 5
    for j in range(num_1d):
      add_var(f"layer_{l}_1d_{j}", [4096], [1], [1], [0], [""], [""])

  return variables


class ReshardPlannerTest(absltest.TestCase):

  def _build_planner_inputs(
      self,
      src_vars_by_unit: dict[RaidenId, list[controller_types.VariableMetadata]],
      dst_vars_by_unit: dict[RaidenId, list[controller_types.VariableMetadata]],
      src_phys_mesh: list[int],
      dst_phys_mesh: list[int],
      dst_mesh_axes: list[str],
      broadcast_host_ratio: float = 0.0,
  ) -> dict[str, Any]:
    lock = threading.Lock()
    src_units = list(src_vars_by_unit.keys())
    dst_units = list(dst_vars_by_unit.keys())

    entities = {}
    registered_shards = {}
    registered_mesh_shapes = {}
    worker_endpoints = {}

    for i, u in enumerate(src_units):
      shards = [f"10.0.0.{i + 1}:{8000 + d}" for d in range(8)]
      registered_shards[u] = shards
      entities[u] = job_entity.JobEntity(unit=u, shards=shards)
      registered_mesh_shapes[u] = src_phys_mesh
      worker_endpoints[u] = f"10.0.0.{i + 1}:9000"

    dst_metadata = []
    proto_vars_cache = {}
    for j, u in enumerate(dst_units):
      shards = [f"10.1.0.{j + 1}:{8000 + d}" for d in range(8)]
      meta = raiden_service_pb2.RegisterWorkUnitRequest(
          unit=raiden_service_pb2.RaidenIdProto(
              job_name=u.job_name,
              job_replica_id=str(u.job_replica_id),
              data_name=u.data_name,
              data_replica_idx=u.data_replica_idx,
          ),
          control_plane_rpc_address=f"10.1.0.{j + 1}:9000",
      )
      meta.shards.extend(shards)
      meta.mesh_shape.extend(dst_phys_mesh)
      meta.mesh_axes.extend(dst_mesh_axes)
      u_vars = dst_vars_by_unit[u]
      proto_vars = proto_vars_cache.get(id(u_vars))
      if proto_vars is None:
        tmpl = raiden_service_pb2.RegisterWorkUnitRequest()
        for v in u_vars:
          vp = tmpl.variables.add()
          vp.name = v.name
          vp.shape.extend(v.shape)
          vp.mesh_shape.extend(v.mesh_shape)
          vp.layout.extend(v.layout)
          vp.item_size = v.item_size
          vp.layer_idx = v.layer_idx
          vp.sharding_spec.extend(v.sharding_spec)
          if getattr(v, "global_shard_indices", None):
            vp.global_shard_indices.extend(v.global_shard_indices)
        proto_vars = list(tmpl.variables)
        proto_vars_cache[id(u_vars)] = proto_vars
      meta.variables.extend(proto_vars)
      dst_metadata.append(meta)

    return dict(
        src_units=src_units,
        dst_units=dst_units,
        dst_metadata=dst_metadata,
        entities=entities,
        registered_variables=src_vars_by_unit,
        registered_global_shapes={},
        registered_mesh_shapes=registered_mesh_shapes,
        registered_layouts={},
        registered_itemsizes={},
        registered_shards=registered_shards,
        computed_phys_meshes={},
        worker_endpoints=worker_endpoints,
        broadcast_host_ratio=broadcast_host_ratio,
        lock=lock,
    )

  def _run_planner(
      self,
      src_vars_by_unit: dict[RaidenId, list[controller_types.VariableMetadata]],
      dst_vars_by_unit: dict[RaidenId, list[controller_types.VariableMetadata]],
      src_phys_mesh: list[int],
      dst_phys_mesh: list[int],
      dst_mesh_axes: list[str],
  ) -> controller_types.CachedTransferSchedule:
    kwargs = self._build_planner_inputs(
        src_vars_by_unit=src_vars_by_unit,
        dst_vars_by_unit=dst_vars_by_unit,
        src_phys_mesh=src_phys_mesh,
        dst_phys_mesh=dst_phys_mesh,
        dst_mesh_axes=dst_mesh_axes,
    )
    return (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **kwargs
        )
    )

  def test_variable_signature_deduplication_exact_equivalence(self):
    """Verifies multi-layer deduplicated schedule matches single-layer per-var plans."""
    src_units = [RaidenId("trainer", str(i), "weights", 0) for i in range(4)]
    dst_units = [
        RaidenId("rollout_0", "0", "weights", 0),
        RaidenId("rollout_0", "1", "weights", 0),
        RaidenId("rollout_1", "0", "weights", 0),
        RaidenId("rollout_1", "1", "weights", 0),
    ]
    src_vars = _build_qwen3_397b_variables(
        num_layers=4, src_fsdp=4, is_src=True
    )
    dst_vars = _build_qwen3_397b_variables(
        num_layers=4, src_fsdp=4, is_src=False
    )

    # Compute multi-layer schedule in one call (uses variable deduplication)
    batched_schedule = self._run_planner(
        src_vars_by_unit={u: src_vars for u in src_units},
        dst_vars_by_unit={u: dst_vars for u in dst_units},
        src_phys_mesh=[1, 1, 4, 4, 2],
        dst_phys_mesh=[2, 8],
        dst_mesh_axes=["x", "y"],
    )

    # Verify that computing variables individually in isolation produces identical
    # canonical variable plans as the batched multi-layer run.
    for v_idx, (s_var, d_var) in enumerate(zip(src_vars, dst_vars)):
      single_schedule = self._run_planner(
          src_vars_by_unit={u: [s_var] for u in src_units},
          dst_vars_by_unit={u: [d_var] for u in dst_units},
          src_phys_mesh=[1, 1, 4, 4, 2],
          dst_phys_mesh=[2, 8],
          dst_mesh_axes=["x", "y"],
      )
      for u in src_units:
        pid = batched_schedule.variable_to_plan_id[u][v_idx]
        self.assertEqual(
            batched_schedule.variable_plans[u][pid],
            single_schedule.variable_plans[u][0],
        )

    # Verify that repeated variables across layers reference the same plan_id
    # in variable_plans instead of duplicating calculations or plans.
    for u in src_units:
      self.assertLen(batched_schedule.variable_plans[u], 6)
      self.assertLen(batched_schedule.variable_to_plan_id[u], len(src_vars))
      # Layer 0's first 2D weight (layer_idx=3) and Layer 1's first 2D weight
      # (layer_idx=19) must refer to the exact same plan_id.
      self.assertEqual(
          batched_schedule.variable_to_plan_id[u][3],
          batched_schedule.variable_to_plan_id[u][19],
      )

    # Verify JobEntity.build_sender_push_schedule_protos produces valid
    # ShardPushScheduleProto messages from _PlanReferencedShardSchedule.
    for i, u in enumerate(src_units):
      entity = job_entity.JobEntity(
          unit=u, shards=[f"10.0.0.{i + 1}:{8000 + d}" for d in range(8)]
      )
      all_protos = {}
      for stage_group in batched_schedule.broadcast_groups.values():
        stage_schedules = (
            reshard_planner.BroadcastEngine.build_stage_trainer_push_schedules(
                stage_group, stage_group.dst_units
            )
        )
        protos = entity.build_sender_push_schedule_protos(stage_schedules[u])
        all_protos.update(protos)
      self.assertNotEmpty(all_protos)
      # Layer 0's first 2D weight (layer_idx=3) and Layer 1's first 2D weight
      # (layer_idx=19) must refer to the exact same plan_id.
      self.assertEqual(
          batched_schedule.variable_to_plan_id[u][3],
          batched_schedule.variable_to_plan_id[u][19],
      )

  def test_benchmark_qwen3_397b_exact_job_schedule(self):
    """Benchmarks 948-variable Qwen3.5-397B-A17B ReshardPlanner schedule computation via timeit."""

    def _make_plan_runner(kwargs, out):
      def _calc_plan():
        out[0] = (
            reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
                **kwargs
            )
        )

      return _calc_plan

    for num_src_units, src_fsdp in ((32, 32), (64, 64)):
      src_units = [
          RaidenId("trainer", str(i), "weights", 0)
          for i in range(num_src_units)
      ]
      src_vars = _build_qwen3_397b_variables(
          num_layers=60, src_fsdp=src_fsdp, is_src=True
      )
      dst_vars = _build_qwen3_397b_variables(
          num_layers=60, src_fsdp=src_fsdp, is_src=False
      )
      self.assertLen(src_vars, 948)

      for num_reps in (1, 8, 16, 32):
        dst_units = []
        for r in range(num_reps):
          for h in range(2):
            dst_units.append(RaidenId(f"rollout_{r}", str(h), "weights", 0))
        planner_kwargs = self._build_planner_inputs(
            src_vars_by_unit={u: src_vars for u in src_units},
            dst_vars_by_unit={u: dst_vars for u in dst_units},
            src_phys_mesh=[1, 1, src_fsdp, 4, 2],
            dst_phys_mesh=[2, 8],
            dst_mesh_axes=["x", "y"],
        )
        last_sched = [None]
        timer = timeit.Timer(_make_plan_runner(planner_kwargs, last_sched))
        runs = timer.repeat(repeat=1, number=1)
        avg_elapsed = min(runs)
        sched = last_sched[0]
        unique_entries = 0
        logical_entries = 0
        for stage_group in sched.broadcast_groups.values():
          for u_plans in stage_group.canonical_variable_plans.values():
            for shard_dict in u_plans.values():
              for tuples_9 in shard_dict.values():
                unique_entries += len(tuples_9)
        logging.info(
            "TIMEIT BENCHMARK Qwen3.5-397B (948 vars) %dd -> %d replicas"
            " (%du): %.4fs (unique_plans=%d,"
            " unique_entries=%d, logical_entries=%d)",
            num_src_units * 8,
            num_reps,
            len(dst_units),
            avg_elapsed,
            len(sched.variable_plans[src_units[0]]),
            unique_entries,
            logical_entries,
        )
        self.assertNotEmpty(sched.broadcast_groups)
        self.assertLen(sched.variable_plans[src_units[0]], 6)
        self.assertLen(sched.variable_to_plan_id[src_units[0]], 948)

  def test_deterministic_circular_shift_latin_square_scheduling(self):
    """Verifies that direct push schedules have rotated destination peer sequences across src_units."""
    src_units = [RaidenId("trainer", str(i), "weights", 0) for i in range(4)]
    dst_units = [
        RaidenId("rollout_0", "0", "weights", 0),
        RaidenId("rollout_1", "0", "weights", 0),
        RaidenId("rollout_2", "0", "weights", 0),
        RaidenId("rollout_3", "0", "weights", 0),
    ]
    src_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=4, is_src=True
    )
    dst_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=4, is_src=False
    )

    sched = self._run_planner(
        src_vars_by_unit={u: src_vars for u in src_units},
        dst_vars_by_unit={u: dst_vars for u in dst_units},
        src_phys_mesh=[1, 1, 4, 4, 2],
        dst_phys_mesh=[2, 8],
        dst_mesh_axes=["x", "y"],
    )

    # Latin Square property: across multiple src_units, each trainer has a destination
    # sequence circularly shifted by its index, so Trainer 0 starts with Dest 0, Trainer 1 with Dest 1.
    # We verify this on all variables where each trainer targets multiple destination peers.
    stage_group = sched.broadcast_groups[(0, 0)]
    s_u_schedules = (
        reshard_planner.BroadcastEngine.build_stage_trainer_push_schedules(
            stage_group, dst_units
        )
    )
    all_var_indices = sorted(
        list(set(e[-2] for e in s_u_schedules[src_units[0]][0]))
    )
    self.assertNotEmpty(all_var_indices)

    multi_dest_vars_tested = 0
    for var_idx in all_var_indices:
      peers_by_src = []
      for src_u in src_units:
        entries = s_u_schedules[src_u][0]
        var_entries = [e for e in entries if e[-2] == var_idx]
        peers = [e[0] for e in var_entries]
        peers_by_src.append(peers)

      # For multi-destination variables, verify that each trainer starts with a distinct peer
      # (balanced scheduling: Trainer 0 starts with Dest 0, Trainer 1 with Dest 1, etc.)
      num_dest_peers = len(set(peers_by_src[0]))
      if num_dest_peers > 1:
        multi_dest_vars_tested += 1
        first_dest_peers = [peers[0] for peers in peers_by_src]
        self.assertEqual(
            len(set(first_dest_peers)),
            min(len(src_units), num_dest_peers),
            f"Var {var_idx} trainers should start with distinct destination"
            " peers",
        )

    self.assertGreater(multi_dest_vars_tested, 0)

  def test_canonical_address_free_resharding_plan_invariant_to_n_samplers(self):
    """Verifies that variable_plans stores address-free 9-tuples invariant to N samplers."""
    num_src_units = 4
    src_units = [
        RaidenId("trainer", str(i), "weights", 0) for i in range(num_src_units)
    ]
    src_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=4, is_src=True
    )
    dst_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=4, is_src=False
    )

    plans_by_n = {}
    relay_plans_by_n = {}
    for num_samplers in (1, 4, 16, 64):
      dst_units = [
          RaidenId(f"rollout_{i}", "0", "weights", 0)
          for i in range(num_samplers)
      ]
      planner_inputs = self._build_planner_inputs(
          src_vars_by_unit={u: src_vars for u in src_units},
          dst_vars_by_unit={u: dst_vars for u in dst_units},
          src_phys_mesh=[1, 1, 4, 4, 2],
          dst_phys_mesh=[2, 8],
          dst_mesh_axes=["x", "y"],
      )
      sched = reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
          **planner_inputs
      )

      # 1. Verify 9-tuple shape and absence of concrete IP string
      for u in src_units:
        for pid, shard_dict in sched.variable_plans[u].items():
          for s_idx, tuples_9 in shard_dict.items():
            for t9 in tuples_9:
              self.assertLen(t9, 9)
              # Verify no address/string fields
              for item in t9:
                self.assertIsInstance(item, int)
                self.assertNotIn(":", str(item))

      # 2. Verify canonical relay plans structure
      self.assertNotEmpty(sched.canonical_relay_plans)
      for pid, local_shards in sched.canonical_relay_plans.items():
        for local_dst_idx, blocks in local_shards.items():
          for spec in blocks:
            self.assertLen(spec, 3)
            min_off, block_size, dst_block_id = spec
            self.assertGreaterEqual(min_off, 0)
            self.assertGreater(block_size, 0)
            self.assertGreaterEqual(dst_block_id, 0)

      plans_by_n[num_samplers] = sched.variable_plans
      relay_plans_by_n[num_samplers] = sched.canonical_relay_plans

    # 3. Exact equality of variable_plans and canonical_relay_plans across N=1, 4, 16, 64
    base_plans = plans_by_n[1]
    base_relay = relay_plans_by_n[1]
    for n in (4, 16, 64):
      self.assertEqual(plans_by_n[n], base_plans)
      self.assertEqual(relay_plans_by_n[n], base_relay)

  def test_compute_bandwidth_matched_seed(self):
    # <= 0.0 normalizes to 1.0 (Bandwidth matching: H_train=32, H_sample=2 -> 32/2 = 16)
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=0.0,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        16,
    )
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=-1.0,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        16,
    )
    # None normalizes to 1.0
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=None,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        16,
    )
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(None, 32, 2, 16),
        reshard_planner.compute_bandwidth_matched_seed(1.0, 32, 2, 16),
    )
    # Bandwidth matching: H_train=32, H_sample=2 -> 32/2 = 16
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=1.0,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        16,
    )
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=0.5,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        8,
    )
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=2.0,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        32,
    )
    # Floor behavior with minimum 1
    self.assertEqual(
        reshard_planner.compute_bandwidth_matched_seed(
            broadcast_host_ratio=0.01,
            num_src_hosts=32,
            num_dst_hosts_per_unit=2,
            num_dst_units=16,
        ),
        1,
    )
    # Validation errors
    with self.assertRaisesRegex(ValueError, "num_src_hosts must be positive"):
      reshard_planner.compute_bandwidth_matched_seed(
          broadcast_host_ratio=1.0,
          num_src_hosts=0,
          num_dst_hosts_per_unit=2,
          num_dst_units=16,
      )
    with self.assertRaisesRegex(
        ValueError, "num_dst_hosts_per_unit must be positive"
    ):
      reshard_planner.compute_bandwidth_matched_seed(
          broadcast_host_ratio=1.0,
          num_src_hosts=32,
          num_dst_hosts_per_unit=0,
          num_dst_units=16,
      )
    with self.assertRaisesRegex(
        ValueError, "num_dst_units must be non-negative"
    ):
      reshard_planner.compute_bandwidth_matched_seed(
          broadcast_host_ratio=1.0,
          num_src_hosts=32,
          num_dst_hosts_per_unit=2,
          num_dst_units=-1,
      )

  def test_make_plan_cache_key_includes_broadcast_host_ratio(self):
    src = [RaidenId("t", "0", "v", 0)]
    dst = [RaidenId("s", "0", "v", 0)]
    key1 = reshard_planner.ReshardPlanner.make_plan_cache_key(
        src_units=src, dst_units=dst, broadcast_host_ratio=1.0
    )
    key2 = reshard_planner.ReshardPlanner.make_plan_cache_key(
        src_units=src, dst_units=dst, broadcast_host_ratio=2.0
    )
    key0 = reshard_planner.ReshardPlanner.make_plan_cache_key(
        src_units=src, dst_units=dst, broadcast_host_ratio=0.0
    )
    key_none = reshard_planner.ReshardPlanner.make_plan_cache_key(
        src_units=src, dst_units=dst, broadcast_host_ratio=None
    )
    self.assertNotEqual(key1, key2)
    self.assertEqual(key1[-2], 1.0)
    self.assertEqual(key2[-2], 2.0)
    self.assertEqual(key0, key1)
    self.assertEqual(key_none, key1)
    self.assertEqual(key1[-1], 4)

  def test_make_plan_cache_key_includes_broadcast_pipeline_stages(self):
    src = [RaidenId("t", "0", "v", 0)]
    dst = [RaidenId("s", "0", "v", 0)]
    key1 = reshard_planner.ReshardPlanner.make_plan_cache_key(
        src_units=src, dst_units=dst, broadcast_pipeline_stages=4
    )
    key2 = reshard_planner.ReshardPlanner.make_plan_cache_key(
        src_units=src, dst_units=dst, broadcast_pipeline_stages=8
    )
    self.assertNotEqual(key1, key2)
    self.assertEqual(key1[-1], 4)
    self.assertEqual(key2[-1], 8)
    with self.assertRaisesRegex(
        ValueError, "broadcast_pipeline_stages must be >= 1"
    ):
      reshard_planner.ReshardPlanner.make_plan_cache_key(
          src_units=src, dst_units=dst, broadcast_pipeline_stages=0
      )

  def test_compute_transfer_schedule_pipeline_target_stages_validation(self):
    with self.assertRaisesRegex(
        ValueError, "pipeline_target_stages must be >= 1"
    ):
      reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
          src_units=[],
          dst_units=[],
          dst_metadata=[],
          entities={},
          registered_variables={},
          registered_global_shapes={},
          registered_mesh_shapes={},
          registered_layouts={},
          registered_itemsizes={},
          registered_shards={},
          computed_phys_meshes={},
          worker_endpoints={},
          broadcast_host_ratio=0.0,
          lock=threading.Lock(),
          pipeline_target_stages=0,
      )

  def test_compute_transfer_schedule_empty_src_units_raises(self):
    with self.assertRaisesRegex(ValueError, "src_units must not be empty"):
      reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
          src_units=[],
          dst_units=[RaidenId("s", "0", "v", 0)],
          dst_metadata=[],
          entities={},
          registered_variables={},
          registered_global_shapes={},
          registered_mesh_shapes={},
          registered_layouts={},
          registered_itemsizes={},
          registered_shards={RaidenId("s", "0", "v", 0): ["10.0.0.1:8000"]},
          computed_phys_meshes={},
          worker_endpoints={},
          broadcast_host_ratio=0.0,
          lock=threading.Lock(),
      )

  def test_compute_transfer_schedule_empty_dst_raises(self):
    with self.assertRaisesRegex(
        ValueError, "dst_metadata or dst_units must not be empty"
    ):
      reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
          src_units=[RaidenId("t", "0", "v", 0)],
          dst_units=[],
          dst_metadata=[],
          entities={},
          registered_variables={},
          registered_global_shapes={},
          registered_mesh_shapes={},
          registered_layouts={},
          registered_itemsizes={},
          registered_shards={RaidenId("t", "0", "v", 0): ["10.0.0.1:8000"]},
          computed_phys_meshes={},
          worker_endpoints={},
          broadcast_host_ratio=0.0,
          lock=threading.Lock(),
      )

  def test_compute_transfer_schedule_multi_unit_fsdp_seed_count(self):
    """Verifies that total src hosts across multiple 1-shard units scale N_seed."""
    src_units = [
        RaidenId("fsdp_trainer", str(i), "weights", 0) for i in range(8)
    ]
    dst_units = [
        RaidenId("fsdp_sampler", str(i), "weights", 0) for i in range(4)
    ]
    src_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=8, is_src=True
    )
    dst_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=8, is_src=False
    )
    src_vars_by_unit = {u: src_vars for u in src_units}
    dst_vars_by_unit = {u: dst_vars for u in dst_units}

    inputs_r0125 = self._build_planner_inputs(
        src_vars_by_unit=src_vars_by_unit,
        dst_vars_by_unit=dst_vars_by_unit,
        src_phys_mesh=[1, 1, 1],
        dst_phys_mesh=[1, 1, 1],
        dst_mesh_axes=["tp", "tp_wo", "tp_out"],
        broadcast_host_ratio=0.125,
    )
    for u in src_units:
      inputs_r0125["registered_shards"][u] = [f"10.0.0.{u.job_replica_id}:8000"]
      inputs_r0125["entities"][u] = job_entity.JobEntity(
          unit=u, shards=inputs_r0125["registered_shards"][u]
      )
    for j, _ in enumerate(dst_units):
      inputs_r0125["dst_metadata"][j].shards[:] = [f"10.1.0.{j}:8000"]

    sched_r0125 = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **inputs_r0125
        )
    )
    self.assertEqual(sched_r0125.n_seed, 1)

    inputs_r025 = dict(inputs_r0125)
    inputs_r025["broadcast_host_ratio"] = 0.25
    sched_r025 = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **inputs_r025
        )
    )
    self.assertEqual(sched_r025.n_seed, 2)

  def test_multi_host_destination_equivalence_class_broadcast(self):
    """Verifies disjoint broadcast trees for multi-host (:0 and :1) rollout replicas."""
    src_units = [RaidenId("train", str(i), "w", 0) for i in range(16)]
    dst_units = []
    for r in range(8):
      dst_units.append(RaidenId(f"roll-{r}", "0", "w", 0))
      dst_units.append(RaidenId(f"roll-{r}", "1", "w", 0))

    src_vars_by_unit = {}
    for i, u in enumerate(src_units):
      src_vars_by_unit[u] = [
          reshard_planner._VariableMetadata(
              name="layer0.weight",
              shape=[128, 256],
              mesh_shape=[16, 8],
              layout=[1, 0],
              item_size=2,
              layer_idx=0,
              sharding_spec=["fsdp", "tp"],
              global_shard_indices=list(range(i * 8, (i + 1) * 8)),
          )
      ]

    dst_vars_by_unit = {}
    for d_u in dst_units:
      host_idx = int(d_u.job_replica_id)
      dst_vars_by_unit[d_u] = [
          reshard_planner._VariableMetadata(
              name="layer0.weight",
              shape=[128, 256],
              mesh_shape=[1, 16],
              layout=[1, 0],
              item_size=2,
              layer_idx=0,
              sharding_spec=["", "tp"],
              global_shard_indices=list(
                  range(host_idx * 8, (host_idx + 1) * 8)
              ),
          )
      ]

    inputs = self._build_planner_inputs(
        src_vars_by_unit=src_vars_by_unit,
        dst_vars_by_unit=dst_vars_by_unit,
        src_phys_mesh=[4, 4, 8],
        dst_phys_mesh=[2, 2, 4],
        dst_mesh_axes=["x", "y", "z"],
        broadcast_host_ratio=0.25,
    )
    sched = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **inputs
        )
    )
    self.assertEqual(sched.n_seed, 2)
    self.assertEmpty(sched.direct_schedules)
    self.assertLen(sched.broadcast_groups, 2)
    sg0 = sched.broadcast_groups[(0, 0)]
    sg1 = sched.broadcast_groups[(1, 0)]
    self.assertEqual([u.job_replica_id for u in sg0.dst_units], ["0"] * 8)
    self.assertEqual([u.job_replica_id for u in sg1.dst_units], ["1"] * 8)
    for sg in (sg0, sg1):
      seed_block_ids_for_shard0 = set()
      for u_plans in sg.canonical_variable_plans.values():
        for shard_dict in u_plans.values():
          for tuples_9 in shard_dict.values():
            for t9 in tuples_9:
              if t9[0] == 0:
                seed_block_ids_for_shard0.add(t9[5])
      # Multiple FSDP source shards produce distinct sub-row t9[5] block IDs,
      # which coalesce into a single whole-buffer relay block with
      # dst_block_id=0.
      self.assertGreater(len(seed_block_ids_for_shard0), 1)
      for pid_relays in sg.canonical_relay_plans.values():
        self.assertLen(pid_relays, 8)
        for blocks in pid_relays.values():
          self.assertLen(blocks, 1)
          self.assertEqual(blocks[0][1], 4096)
          self.assertEqual(blocks[0][2], 0)

    # With broadcast_host_ratio=0.5, 16 source hosts / 2 dst hosts per replica
    # * 0.5 = 4 seeds per equivalence class.
    inputs_r05 = dict(inputs, broadcast_host_ratio=0.5)
    sched_r05 = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **inputs_r05
        )
    )
    self.assertEqual(sched_r05.n_seed, 4)

  def test_destination_equivalence_class_missing_metadata_raises(self):
    src_units = [RaidenId("train", "0", "w", 0)]
    dst_units = [
        RaidenId("roll-0", "0", "w", 0),
        RaidenId("roll-0", "1", "w", 0),
    ]
    v_meta = [
        reshard_planner._VariableMetadata(
            name="layer0.weight",
            shape=[16, 16],
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=2,
            layer_idx=0,
        )
    ]
    inputs = self._build_planner_inputs(
        src_vars_by_unit={src_units[0]: v_meta},
        dst_vars_by_unit={u: v_meta for u in dst_units},
        src_phys_mesh=[1, 1],
        dst_phys_mesh=[1, 1],
        dst_mesh_axes=["x", "y"],
    )
    inputs["dst_metadata"] = inputs["dst_metadata"][:1]
    with self.assertRaisesRegex(ValueError, "missing from dst_metadata"):
      reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
          **inputs
      )

  def test_pathways_single_unit_multi_host_flat_direct_push_latin_square_shift(
      self,
  ):
    """Verifies deterministic Latin Square circular shift for single-unit multi-host flat direct push."""
    src_unit = RaidenId("trainer", "", "weights", 0)
    src_units = [src_unit]
    dst_units = [RaidenId(f"rollout_{j}", "0", "weights", 0) for j in range(4)]

    src_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=4, is_src=True
    )
    dst_vars = _build_qwen3_397b_variables(
        num_layers=1, src_fsdp=4, is_src=False
    )

    cases = [
        ("contiguous_host_shards", lambda s: s // 8),
        ("torus_interleaved_host_shards", lambda s: (s // 2) % 4),
    ]

    for case_name, host_fn in cases:
      with self.subTest(placement=case_name):
        src_shards = [f"10.0.0.{host_fn(s) + 1}:8000" for s in range(32)]
        registered_shards = {src_unit: src_shards}
        entities = {
            src_unit: job_entity.JobEntity(unit=src_unit, shards=src_shards)
        }
        dst_metadata = []
        for j, u in enumerate(dst_units):
          shards = [f"10.1.{j + 1}.1:8000"]
          registered_shards[u] = shards
          meta = raiden_service_pb2.RegisterWorkUnitRequest(
              unit=raiden_service_pb2.RaidenIdProto(
                  job_name=u.job_name,
                  job_replica_id=str(u.job_replica_id),
                  data_name=u.data_name,
                  data_replica_idx=u.data_replica_idx,
              ),
              control_plane_rpc_address=f"10.1.{j + 1}.1:9000",
          )
          meta.shards.extend(shards)
          meta.mesh_shape.extend([1, 1])
          meta.mesh_axes.extend(["tp", "tp_wo"])
          for v in dst_vars:
            vp = meta.variables.add()
            vp.name = v.name
            vp.shape.extend(v.shape)
            vp.mesh_shape.extend(v.mesh_shape)
            vp.layout.extend(v.layout)
            vp.item_size = v.item_size
            vp.layer_idx = v.layer_idx
            vp.sharding_spec.extend(v.sharding_spec)
          dst_metadata.append(meta)

        sched = reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            src_units=src_units,
            dst_units=dst_units,
            dst_metadata=dst_metadata,
            entities=entities,
            registered_variables={src_unit: src_vars},
            registered_global_shapes={},
            registered_mesh_shapes={src_unit: [1, 1, 4, 4, 2]},
            registered_layouts={},
            registered_itemsizes={},
            registered_shards=registered_shards,
            computed_phys_meshes={},
            worker_endpoints={
                src_unit: (
                    "10.0.0.1:9000,10.0.0.2:9000,10.0.0.3:9000,10.0.0.4:9000"
                )
            },
            broadcast_host_ratio=0.0,
            lock=threading.Lock(),
        )

        stage_group = sched.broadcast_groups[(0, 0)]
        s_u_schedules = (
            reshard_planner.BroadcastEngine.build_stage_trainer_push_schedules(
                stage_group, dst_units, registered_shards=registered_shards
            )
        )
        schedule_protos = entities[src_unit].build_sender_push_schedule_protos(
            s_u_schedules[src_unit]
        )

        shards_by_host = collections.defaultdict(list)
        for s in range(32):
          h = host_fn(s)
          shards_by_host[h].append(s)

        peers_by_host_sched = {}
        peers_by_host_proto = {}
        first_dest_units_by_host = {}

        for h in range(4):
          host_shards = shards_by_host[h]
          for s_idx in host_shards:
            shard_sched = s_u_schedules[src_unit][s_idx]
            layer0_sched_peers = []
            for entry in shard_sched:
              if entry[10] == 0:  # layer_idx
                p = entry[0]
                if p not in layer0_sched_peers:
                  layer0_sched_peers.append(p)
            self.assertLen(layer0_sched_peers, 4)
            if h not in peers_by_host_sched:
              peers_by_host_sched[h] = layer0_sched_peers
            else:
              self.assertEqual(
                  peers_by_host_sched[h],
                  layer0_sched_peers,
                  f"All shards on host {h} must share destination sequence in"
                  " sched",
              )

            proto = schedule_protos[s_idx]
            layer0_entries = [e for e in proto.entries if e.layer_idx == 0]
            self.assertNotEmpty(layer0_entries)
            layer0_proto_peers = list(layer0_entries[0].dst_peers)
            self.assertLen(layer0_proto_peers, 4)
            if h not in peers_by_host_proto:
              peers_by_host_proto[h] = layer0_proto_peers
            else:
              self.assertEqual(
                  peers_by_host_proto[h],
                  layer0_proto_peers,
                  f"All shards on host {h} must share destination sequence in"
                  " proto",
              )

          first_peer = peers_by_host_proto[h][0]
          first_unit = sched.data_address_to_unit[first_peer]
          first_dest_units_by_host[h] = first_unit
          self.assertEqual(
              first_unit,
              dst_units[h],
              f"Host {h} must start with {dst_units[h]}",
          )

        for h in range(4):
          expected_shift_sched = (
              peers_by_host_sched[0][h:] + peers_by_host_sched[0][:h]
          )
          self.assertEqual(
              peers_by_host_sched[h],
              expected_shift_sched,
              f"Host {h} sched peers must be circularly shifted by {h}",
          )
          expected_shift_proto = (
              peers_by_host_proto[0][h:] + peers_by_host_proto[0][:h]
          )
          self.assertEqual(
              peers_by_host_proto[h],
              expected_shift_proto,
              f"Host {h} proto peers must be circularly shifted by {h}",
          )

        self.assertLen(
            set(first_dest_units_by_host.values()),
            4,
            "All 4 hosts must have distinct first destination units",
        )

  def test_skip_tiling_identical_specs_requires_tile_alignment(self):
    """Verifies identical src/dst variable specs only skip tiling when local shards are (8, 128)-aligned."""
    src_units = [RaidenId("trainer", str(i), "weights", 0) for i in range(4)]
    dst_units = [RaidenId("sampler", str(i), "weights", 0) for i in range(4)]

    identical_vars = [
        # Layer 0: 1D tensor -> rank < 2 -> False
        controller_types.VariableMetadata(
            name="layer_0.norm",
            shape=[5120],
            mesh_shape=[1],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=[""],
        ),
        # Layer 1: 2D local shard [256, 5120] -> (8, 128)-aligned -> True
        controller_types.VariableMetadata(
            name="layer_0.q_proj",
            shape=[8192, 5120],
            mesh_shape=[32, 1],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["tp", ""],
        ),
        # Layer 2: 2D local shard [4, 5120] -> row 4 % 8 != 0 -> False
        controller_types.VariableMetadata(
            name="layer_0.k_proj",
            shape=[128, 5120],
            mesh_shape=[32, 1],
            layout=[1, 0],
            item_size=2,
            layer_idx=2,
            sharding_spec=["tp", ""],
        ),
        # Layer 3: 2D local shard [5120, 64] -> col 64 % 128 != 0 -> False
        controller_types.VariableMetadata(
            name="layer_0.o_proj",
            shape=[5120, 2048],
            mesh_shape=[1, 32],
            layout=[1, 0],
            item_size=2,
            layer_idx=3,
            sharding_spec=["", "tp"],
        ),
        # Layer 4: 2D replicated tensor [2048, 1] -> col 1 % 128 != 0 -> False
        controller_types.VariableMetadata(
            name="layer_0.shared_expert_gate",
            shape=[2048, 1],
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=2,
            layer_idx=4,
            sharding_spec=["", ""],
        ),
        # Layer 5: 3D local shard [1, 512, 2048] -> (8, 128)-aligned -> True
        controller_types.VariableMetadata(
            name="layer_0.moe_aligned",
            shape=[32, 512, 2048],
            mesh_shape=[32, 1, 1],
            layout=[2, 1, 0],
            item_size=2,
            layer_idx=5,
            sharding_spec=["tp", "", ""],
        ),
        # Layer 6: 3D local shard [32, 512, 64] -> col 64 % 128 != 0 -> False
        controller_types.VariableMetadata(
            name="layer_0.moe_unaligned_col",
            shape=[32, 512, 2048],
            mesh_shape=[1, 1, 32],
            layout=[2, 1, 0],
            item_size=2,
            layer_idx=6,
            sharding_spec=["", "", "tp"],
        ),
    ]

    for v in identical_vars:
      self.assertTrue(controller_types.is_variable_spec_identical(v, v))

    sched = self._run_planner(
        src_vars_by_unit={u: identical_vars for u in src_units},
        dst_vars_by_unit={u: identical_vars for u in dst_units},
        src_phys_mesh=[1, 32],
        dst_phys_mesh=[1, 32],
        dst_mesh_axes=["data", "tp"],
    )

    self.assertEqual(
        sched.local_skip_tiling,
        {
            0: False,
            1: True,
            2: False,
            3: False,
            4: False,
            5: True,
            6: False,
        },
    )

  def _pair_warning_messages(self, warn_mock: mock.MagicMock) -> list[str]:
    return [
        call.args[0] % call.args[1:]
        for call in warn_mock.call_args_list
        if "peer_control_endpoints" in (call.args[0] if call.args else "")
    ]

  def test_pair_host_endpoints_cases_and_warning_dedup(self) -> None:
    """Verifies pairing cases and per-warned_units set warning dedup."""
    host_proto = raiden_service_pb2.HostEndpointsProto
    pair_fn = reshard_planner.pair_host_endpoints
    valid_host = host_proto(
        control_address="10.0.0.1:9000",
        shards=["10.0.0.1:8000", "10.0.0.1:8001"],
    )
    self.assertEqual(
        pair_fn(RaidenId("sampler", "0", "weights"), [valid_host]),
        {"10.0.0.1:8000": "10.0.0.1:9000", "10.0.0.1:8001": "10.0.0.1:9000"},
    )
    bad_cases = [
        ("empty_ctrl", [("", ["10.0.0.2:8000"])], "10.0.0.2:8000"),
        (
            "multi_ctrl",
            [("10.0.0.3:9000,10.0.0.4:9000", ["10.0.0.3:8000"])],
            "10.0.0.3:9000",
        ),
        ("no_hosts", [], "no HostEndpointsProto"),
        (
            "multi_hosts",
            [
                ("10.0.0.5:9000", ["10.0.0.5:8000"]),
                ("10.0.0.6:9000", ["10.0.0.6:8000"]),
            ],
            "10.0.0.5:9000",
        ),
    ]
    warned_a, warned_b = set(), set()
    for name, host_specs, expected_token in bad_cases:
      unit = RaidenId("s", name, "w")
      hosts = [host_proto(control_address=c, shards=s) for c, s in host_specs]
      with self.subTest(case=name), mock.patch.object(
          logging, "warning", wraps=logging.warning
      ) as warn_mock:
        for warned, count in ((warned_a, 1), (warned_a, 1), (warned_b, 2)):
          self.assertEqual(pair_fn(unit, hosts, warned), {})
          messages = self._pair_warning_messages(warn_mock)
          self.assertLen(messages, count)
          self.assertIn(str(unit), messages[-1])
          self.assertIn(expected_token, messages[-1])

  def test_planner_builds_peer_control_endpoints_from_metadata(self) -> None:
    """Verifies ReshardPlanner pairs metadata.hosts per destination unit."""
    src_unit = RaidenId("trainer", "0", "weights")
    dst_units = [RaidenId("sampler", str(i), "weights") for i in (0, 1)]
    var_metadata = [
        controller_types.VariableMetadata(
            name="w",
            shape=[128, 128],
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=4,
            layer_idx=0,
            sharding_spec=["", ""],
            global_shard_indices=[0] * 8,
        )
    ]
    planner_kwargs = self._build_planner_inputs(
        src_vars_by_unit={src_unit: var_metadata},
        dst_vars_by_unit={unit: var_metadata for unit in dst_units},
        src_phys_mesh=[1, 1],
        dst_phys_mesh=[1, 1],
        dst_mesh_axes=["x", "y"],
    )
    for meta in planner_kwargs["dst_metadata"]:
      meta.hosts.add(
          control_address=meta.control_plane_rpc_address, shards=meta.shards
      )
    compute_schedule = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata
    )
    schedule = compute_schedule(**planner_kwargs)
    self.assertEqual(
        schedule.peer_control_endpoints,
        {
            **{f"10.1.0.1:{8000 + d}": "10.1.0.1:9000" for d in range(8)},
            **{f"10.1.0.2:{8000 + d}": "10.1.0.2:9000" for d in range(8)},
        },
    )
    for meta in planner_kwargs["dst_metadata"]:
      meta.ClearField("hosts")
    with mock.patch.object(
        logging, "warning", wraps=logging.warning
    ) as warn_mock:
      no_hosts = compute_schedule(**planner_kwargs)
    self.assertEqual(no_hosts.peer_control_endpoints, {})
    self.assertLen(self._pair_warning_messages(warn_mock), 2)


if __name__ == "__main__":
  absltest.main()

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

import threading
import timeit
from typing import Any

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
      src_mesh_axes: list[str],
      src_host_subgrid: list[int],
      dst_phys_mesh: list[int],
      dst_mesh_axes: list[str],
      dst_host_subgrid: list[int],
  ) -> dict[str, Any]:
    lock = threading.Lock()
    src_units = list(src_vars_by_unit.keys())
    dst_units = list(dst_vars_by_unit.keys())

    entities = {}
    registered_shards = {}
    registered_mesh_shapes = {}
    registered_mesh_axes = {}
    registered_host_subgrids = {}
    worker_endpoints = {}

    for i, u in enumerate(src_units):
      shards = [f"10.0.0.{i + 1}:{8000 + d}" for d in range(8)]
      registered_shards[u] = shards
      entities[u] = job_entity.JobEntity(unit=u, shards=shards)
      registered_mesh_shapes[u] = src_phys_mesh
      registered_mesh_axes[u] = src_mesh_axes
      registered_host_subgrids[u] = src_host_subgrid
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
      meta.host_subgrid.extend(dst_host_subgrid)
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
        registered_mesh_axes=registered_mesh_axes,
        registered_host_subgrids=registered_host_subgrids,
        registered_layouts={},
        registered_itemsizes={},
        registered_shards=registered_shards,
        computed_phys_meshes={},
        worker_endpoints=worker_endpoints,
        broadcast_k=64,
        lock=lock,
    )

  def _run_planner(
      self,
      src_vars_by_unit: dict[RaidenId, list[controller_types.VariableMetadata]],
      dst_vars_by_unit: dict[RaidenId, list[controller_types.VariableMetadata]],
      src_phys_mesh: list[int],
      src_mesh_axes: list[str],
      src_host_subgrid: list[int],
      dst_phys_mesh: list[int],
      dst_mesh_axes: list[str],
      dst_host_subgrid: list[int],
  ) -> controller_types.CachedTransferSchedule:
    kwargs = self._build_planner_inputs(
        src_vars_by_unit=src_vars_by_unit,
        dst_vars_by_unit=dst_vars_by_unit,
        src_phys_mesh=src_phys_mesh,
        src_mesh_axes=src_mesh_axes,
        src_host_subgrid=src_host_subgrid,
        dst_phys_mesh=dst_phys_mesh,
        dst_mesh_axes=dst_mesh_axes,
        dst_host_subgrid=dst_host_subgrid,
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
        src_mesh_axes=["data", "stage", "fsdp", "context", "expert"],
        src_host_subgrid=[1, 1, 1, 4, 2],
        dst_phys_mesh=[2, 8],
        dst_mesh_axes=["x", "y"],
        dst_host_subgrid=[1, 8],
    )

    # Compute each variable individually in isolation (no variable dedup)
    expected_schedules = {u: {} for u in src_units}
    for s_var, d_var in zip(src_vars, dst_vars):
      single_schedule = self._run_planner(
          src_vars_by_unit={u: [s_var] for u in src_units},
          dst_vars_by_unit={u: [d_var] for u in dst_units},
          src_phys_mesh=[1, 1, 4, 4, 2],
          src_mesh_axes=["data", "stage", "fsdp", "context", "expert"],
          src_host_subgrid=[1, 1, 1, 4, 2],
          dst_phys_mesh=[2, 8],
          dst_mesh_axes=["x", "y"],
          dst_host_subgrid=[1, 8],
      )
      for u, unit_sched in single_schedule.computed_schedules.items():
        for local_idx, entries in unit_sched.items():
          expected_schedules[u].setdefault(local_idx, []).extend(entries)

    self.assertEqual(batched_schedule.computed_schedules, expected_schedules)

    # Reconstruct exact HEAD direct_schedules and counts via BroadcastEngine
    data_address_to_unit = {}
    for d_idx, d_unit in enumerate(dst_units):
      for s in range(8):
        data_address_to_unit[f"10.1.0.{d_idx + 1}:{8000 + s}"] = d_unit
    head_groups = {}
    for s_unit, schedules in expected_schedules.items():
      for shard_idx, entries in schedules.items():
        for entry in entries:
          (
              dst_peer,
              dst_shard_idx,
              dst_block_offset,
              src_block_offset,
              size,
              src_block_id,
              dst_block_id,
              src_stride,
              dst_stride,
              count,
              layer_idx,
              pool_group,
          ) = entry
          dst_unit = data_address_to_unit.get(dst_peer)
          if not dst_unit:
            continue
          key = (
              s_unit,
              shard_idx,
              src_block_id,
              src_block_offset,
              size,
              src_stride,
              count,
              layer_idx,
              pool_group,
          )
          val = (
              dst_unit,
              dst_peer,
              dst_shard_idx,
              dst_block_id,
              dst_block_offset,
              dst_stride,
          )
          head_groups.setdefault(key, []).append(val)
    head_direct_schedules, head_broadcast_groups = (
        reshard_planner.BroadcastEngine.partition_direct_and_broadcast_groups(
            head_groups, broadcast_k=64, group_size=1
        )
    )
    self.assertEqual(
        set(batched_schedule.direct_schedules.keys()),
        set(head_direct_schedules.keys()),
    )
    for u in batched_schedule.direct_schedules:
      self.assertEqual(
          set(batched_schedule.direct_schedules[u].keys()),
          set(head_direct_schedules[u].keys()),
      )
      for s_idx in batched_schedule.direct_schedules[u]:
        self.assertEqual(
            sorted(batched_schedule.direct_schedules[u][s_idx]),
            sorted(head_direct_schedules[u][s_idx]),
        )
    self.assertEqual(batched_schedule.broadcast_groups, head_broadcast_groups)

    head_direct_dsts = []
    for scheds in head_direct_schedules.values():
      for entries in scheds.values():
        for entry in entries:
          d_node = data_address_to_unit.get(entry[0])
          if d_node and d_node not in head_direct_dsts:
            head_direct_dsts.append(d_node)
    self.assertEqual(batched_schedule.direct_dsts, head_direct_dsts)

    # Verify JobEntity.build_sender_push_schedule_protos produces identical
    # ShardPushScheduleProto messages from _PlanReferencedShardSchedule vs HEAD.
    for i, u in enumerate(src_units):
      entity = job_entity.JobEntity(
          unit=u, shards=[f"10.0.0.{i + 1}:{8000 + d}" for d in range(8)]
      )
      dedup_protos = entity.build_sender_push_schedule_protos(
          batched_schedule.direct_schedules[u]
      )
      head_protos = entity.build_sender_push_schedule_protos(
          head_direct_schedules[u]
      )
      self.assertEqual(set(dedup_protos.keys()), set(head_protos.keys()))
      for s_idx in dedup_protos:
        self.assertEqual(
            dedup_protos[s_idx].SerializeToString(deterministic=True),
            head_protos[s_idx].SerializeToString(deterministic=True),
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
            src_mesh_axes=["data", "stage", "fsdp", "context", "expert"],
            src_host_subgrid=[1, 1, 1, 4, 2],
            dst_phys_mesh=[2, 8],
            dst_mesh_axes=["x", "y"],
            dst_host_subgrid=[1, 8],
        )
        last_sched = [None]
        timer = timeit.Timer(_make_plan_runner(planner_kwargs, last_sched))
        runs = timer.repeat(repeat=1, number=1)
        avg_elapsed = min(runs)
        sched = last_sched[0]
        unique_entries = 0
        logical_entries = 0
        for u_sched in sched.computed_schedules.values():
          for s in u_sched.values():
            unique_entries += s.unique_entry_count
            logical_entries += len(s)
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
        self.assertLen(sched.computed_schedules, num_src_units)
        self.assertLen(sched.variable_plans[src_units[0]], 6)
        self.assertLen(sched.variable_to_plan_id[src_units[0]], 948)


if __name__ == "__main__":
  absltest.main()

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

"""Unit tests for TPU_SYNC_PER_HOST_DEDUP (per-host broadcast + loopback relay)."""

import asyncio
import os
import threading
from typing import Any, Optional

from absl.testing import absltest

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import broadcast_engine
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity
from tpu_sync.weight_sync.manager import reshard_planner


class _RecordingWorkerRpcClient(raiden_controller.WeightSyncWorkerRpcClient):
  """Records all start_transfer invocations in dispatch order."""

  def __init__(self) -> None:
    super().__init__()
    self.invocations: list[tuple[RaidenId, Any]] = []

  async def start_transfer(
      self,
      target_id: RaidenId,
      transfer_plan: Any,
      address: Optional[str] = None,
  ) -> None:
    del address
    self.invocations.append((target_id, transfer_plan))


def _build_32chip_topology(
    replicas_per_host: int,
    tp_size: int,
    num_layers: int = 8,
    multi_nic_data_ips: bool = False,
):
  """Builds synthetic 4-trainer-host (32 rank) -> 4-rollout-host metadata."""
  trainer_ips = [f"10.0.1.{10 + h}" for h in range(4)]
  rollout_ctrl_ips = [f"10.0.2.{10 + h}" for h in range(4)]

  src_units: list[RaidenId] = []
  entities: dict[RaidenId, Any] = {}
  registered_shards: dict[RaidenId, list[str]] = {}
  registered_mesh_shapes: dict[RaidenId, list[int]] = {}
  src_vars_by_unit: dict[RaidenId, list[Any]] = {}
  worker_endpoints: dict[RaidenId, str] = {}

  for r in range(32):
    u = RaidenId(job_name="trainer", job_replica_id=str(r), data_name="weights")
    ip = trainer_ips[r // 8]
    port = 19000 + r
    shards = [f"{ip}:{port}"]
    src_units.append(u)
    registered_shards[u] = shards
    entities[u] = job_entity.JobEntity(unit=u, shards=shards)
    registered_mesh_shapes[u] = [32, 1]
    worker_endpoints[u] = f"{ip}:{port + 1000}"
    vars_list = []
    for l_idx in range(num_layers):
      vars_list.append(
          controller_types.VariableMetadata(
              name=f"layers.{l_idx}.w",
              shape=[256, 128],
              mesh_shape=[32, 1],
              layout=[1, 0],
              item_size=2,
              sharding_spec=["fsdp", ""],
              global_shard_indices=[r],
              layer_idx=l_idx,
          )
      )
    src_vars_by_unit[u] = vars_list

  dst_units: list[RaidenId] = []
  dst_metadata: list[Any] = []
  num_replicas = 4 * replicas_per_host
  for rep in range(num_replicas):
    host_idx = rep // replicas_per_host
    intra_host_rep = rep % replicas_per_host
    ctrl_ip = rollout_ctrl_ips[host_idx]
    for tp_rank in range(tp_size):
      idx = rep * tp_size + tp_rank
      u = RaidenId(
          job_name=f"sampler_{rep}",
          job_replica_id=str(tp_rank),
          data_name="weights",
      )
      port = 21000 + idx
      if multi_nic_data_ips:
        # Distinct data-plane NIC subnet per intra-host replica, while sharing
        # the physical node's control_plane_rpc_address IP.
        data_ip = f"10.100.{intra_host_rep + 1}.{10 + host_idx}"
      else:
        data_ip = ctrl_ip
      shard_addr = f"{data_ip}:{port}"
      rpc_addr = f"{ctrl_ip}:{port + 2000}"
      dst_units.append(u)
      registered_shards[u] = [shard_addr]
      worker_endpoints[u] = rpc_addr

      meta = raiden_service_pb2.RegisterWorkUnitRequest(
          unit=raiden_service_pb2.RaidenIdProto(
              job_name=u.job_name,
              job_replica_id=str(u.job_replica_id),
              data_name=u.data_name,
              data_replica_idx=u.data_replica_idx,
          ),
          control_plane_rpc_address=rpc_addr,
      )
      meta.shards.append(shard_addr)
      meta.mesh_shape.extend([1, tp_size])
      meta.mesh_axes.extend(["", "tp"])
      for l_idx in range(num_layers):
        vp = meta.variables.add()
        vp.name = f"layers.{l_idx}.w"
        vp.shape.extend([256, 128])
        vp.mesh_shape.extend([1, tp_size])
        vp.layout.extend([1, 0])
        vp.item_size = 2
        vp.layer_idx = l_idx
        vp.sharding_spec.extend(["", "tp"])
        vp.global_shard_indices.extend([tp_rank])
      dst_metadata.append(meta)

  return (
      src_units,
      dst_units,
      dst_metadata,
      entities,
      registered_shards,
      registered_mesh_shapes,
      src_vars_by_unit,
      worker_endpoints,
  )


class PerHostDedupTest(absltest.TestCase):

  def tearDown(self) -> None:
    os.environ.pop("TPU_SYNC_PER_HOST_DEDUP", None)
    os.environ.pop("TPU_SYNC_DEDUP_PIPELINE_STAGES", None)
    super().tearDown()

  def _compute_schedule(
      self,
      replicas_per_host: int,
      tp_size: int,
      dedup: bool,
      multi_nic_data_ips: bool = False,
  ):
    if dedup:
      os.environ["TPU_SYNC_PER_HOST_DEDUP"] = "1"
    else:
      os.environ.pop("TPU_SYNC_PER_HOST_DEDUP", None)

    (
        src_units,
        dst_units,
        dst_metadata,
        entities,
        registered_shards,
        registered_mesh_shapes,
        src_vars_by_unit,
        worker_endpoints,
    ) = _build_32chip_topology(
        replicas_per_host,
        tp_size,
        num_layers=8,
        multi_nic_data_ips=multi_nic_data_ips,
    )

    sched = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
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
            broadcast_host_ratio=1.0,
            group_size=1,
            req_id="test_req",
            uuid=12345,
            lock=threading.Lock(),
        )
    )
    return sched, src_units, dst_units, registered_shards, worker_endpoints

  def test_stock_vs_dedup_n_seed_and_stages(self) -> None:
    sched_stock, _, _, _, _ = self._compute_schedule(2, 4, dedup=False)
    self.assertEqual(sched_stock.n_seed, 8)
    eff_stock, groups_stock = (
        broadcast_engine.BroadcastEngine.coalesce_pipeline_groups_cached(
            list(sched_stock.broadcast_groups.values()), 4, sched_stock.n_seed
        )
    )
    self.assertEqual(eff_stock, 1)
    self.assertEqual(len(groups_stock), 1)

    sched_dedup, _, _, _, _ = self._compute_schedule(2, 4, dedup=True)
    self.assertEqual(sched_dedup.n_seed, 4)
    eff_dedup, groups_dedup = (
        broadcast_engine.BroadcastEngine.coalesce_pipeline_groups_cached(
            list(sched_dedup.broadcast_groups.values()), 4, sched_dedup.n_seed
        )
    )
    self.assertEqual(eff_dedup, 4)
    self.assertEqual(len(groups_dedup), 16)

    sched_tp2, _, _, _, _ = self._compute_schedule(4, 2, dedup=True)
    self.assertEqual(sched_tp2.n_seed, 4)

    sched_tp8, _, _, _, _ = self._compute_schedule(1, 8, dedup=True)
    self.assertEqual(sched_tp8.n_seed, 4)
    eff_tp8, _ = (
        broadcast_engine.BroadcastEngine.coalesce_pipeline_groups_cached(
            list(sched_tp8.broadcast_groups.values()), 4, sched_tp8.n_seed
        )
    )
    self.assertEqual(eff_tp8, 1)

  def test_execute_pipeline_8x_tp4_and_16x_tp2_strictly_intra_host_relays(
      self,
  ) -> None:
    for multi_nic in (False, True):
      for replicas_per_host, tp_size in ((2, 4), (4, 2)):
        with self.subTest(
            multi_nic=multi_nic,
            replicas_per_host=replicas_per_host,
            tp_size=tp_size,
        ):
          sched, src_units, dst_units, registered_shards, worker_endpoints = (
              self._compute_schedule(
                  replicas_per_host,
                  tp_size,
                  dedup=True,
                  multi_nic_data_ips=multi_nic,
              )
          )
          self.assertEqual(sched.n_seed, 4)
          rpc_client = _RecordingWorkerRpcClient()
          self.addCleanup(rpc_client.close)
          engine = broadcast_engine.BroadcastEngine(rpc_client)

          final_plan = raiden_controller.TransferPlan(
              src_units=src_units,
              dst_units=dst_units,
              plan=None,
              worker_rpc_addresses=worker_endpoints,
              worker_data_addresses=registered_shards,
              uuid=9999,
              is_weight_sync=True,
          )
          asyncio.run(
              engine.execute_slice_broadcast_pipeline(
                  groups_list=list(sched.broadcast_groups.values()),
                  final_plan=final_plan,
                  n_seed=sched.n_seed,
                  req_id="dedup_e2e",
                  dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                  registered_shards=registered_shards,
                  pipeline_target_stages=4,
              )
          )

          def phys_host_ip(
              u: RaidenId, eps: dict[RaidenId, str] = worker_endpoints
          ) -> str:
            return controller_types.extract_host_ip(eps[u])

          seed_receivers: set[RaidenId] = set()
          relay_receivers: set[RaidenId] = set()
          inter_host_relay_hops = 0
          intra_host_relay_hops = 0

          for target_id, plan in rpc_client.invocations:
            if not plan.is_sender and target_id in dst_units:
              self.assertEqual(
                  plan.host_tiling_mode,
                  raiden_service_pb2.HOST_TILING_MODE_ON_ARRIVAL,
              )
              seed_receivers.add(target_id)
              self.assertGreater(plan.expected_block_count, 0)
              self.assertEqual(len(plan.expected_layer_chunk_counts), 8)
            elif (
                plan.is_sender
                and target_id in dst_units
                and target_id in plan.src_units
            ):
              self.assertEqual(
                  plan.host_tiling_mode,
                  raiden_service_pb2.HOST_TILING_MODE_PRE_TILED,
              )
              self.assertTrue(plan.skip_d2h)
              for d_u in plan.dst_units:
                relay_receivers.add(d_u)
                if phys_host_ip(target_id) == phys_host_ip(d_u):
                  intra_host_relay_hops += 1
                else:
                  inter_host_relay_hops += 1

          self.assertEqual(seed_receivers & relay_receivers, set())
          self.assertEqual(seed_receivers | relay_receivers, set(dst_units))
          self.assertEqual(len(seed_receivers), 4 * tp_size)
          self.assertEqual(
              len(relay_receivers), (replicas_per_host - 1) * 4 * tp_size
          )
          self.assertEqual(inter_host_relay_hops, 0)
          self.assertEqual(
              intra_host_relay_hops,
              4 * tp_size * (replicas_per_host - 1) * 4,
          )

  def test_missing_host_id_fails_closed_per_unit_and_isolates_cache_key(
      self,
  ) -> None:
    u0 = RaidenId(job_name="sampler_0", job_replica_id="0", data_name="weights")
    u1 = RaidenId(job_name="sampler_1", job_replica_id="0", data_name="weights")
    leaders, followers = broadcast_engine.partition_dst_units_by_host(
        [u0, u1], lambda _: ""
    )
    self.assertEqual(leaders, [u0, u1])
    self.assertEqual(followers, {u0: [], u1: []})

    os.environ.pop("TPU_SYNC_PER_HOST_DEDUP", None)
    key_off = reshard_planner.ReshardPlanner.make_plan_cache_key([u0], [u1])
    os.environ["TPU_SYNC_PER_HOST_DEDUP"] = "1"
    key_on = reshard_planner.ReshardPlanner.make_plan_cache_key([u0], [u1])
    self.assertNotEqual(key_off, key_on)


if __name__ == "__main__":
  absltest.main()

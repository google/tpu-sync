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

"""Tests for the local V3 controller: planning, execution, lifecycle, E2E."""

import asyncio
import gc
import itertools
import sys
import threading
import time
import warnings

from absl.testing import absltest
import numpy as np

from tpu_sync.api.common import RaidenId
from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import v3


def _vars(
    num_layers,
    shape=(32, 64),
    mesh_shape=(1, 1),
    prefix="w",
    global_shard_indices=(0,),
):
  return [
      raiden_service_pb2.VariableMetadataProto(
          name=f"{prefix}_{l}",
          shape=list(shape),
          mesh_shape=list(mesh_shape),
          layout=[1, 0],
          item_size=2,
          layer_idx=l,
          global_shard_indices=list(global_shard_indices),
      )
      for l in range(num_layers)
  ]


def _register_fake_topology(ctrl, trainer, samplers, num_layers, **kw):
  vars_proto = _vars(num_layers, **kw)
  ctrl.register_work_unit(
      trainer,
      ["127.0.0.1:8000"],
      control_plane_rpc_address="127.0.0.1:9000",
      variables=vars_proto,
  )
  for idx, s_unit in enumerate(samplers):
    ctrl.register_work_unit(
        s_unit,
        [f"127.0.0.1:{8100 + idx}"],
        control_plane_rpc_address=f"127.0.0.1:{9100 + idx}",
        variables=vars_proto,
    )


def _cpu_ws(num_layers, num_shards, slice_bytes):
  return weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
      num_layers=num_layers,
      num_shards=num_shards,
      slice_byte_size=slice_bytes,
      local_port=0,
      listener_port=0,
      bind_ip="127.0.0.1",
  )


class _RecordingSender:
  """rpc_sender that records (endpoint, StartTransferRequest) and succeeds."""

  def __init__(self, fail=False, block_event=None):
    self.calls: list[tuple[str, raiden_service_pb2.ControlRequest]] = []
    self.fail = fail
    self.block_event = block_event
    self._lock = threading.Lock()

  def __call__(self, endpoint: str, req_bytes: bytes):
    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(req_bytes)
    with self._lock:
      self.calls.append((endpoint, req))
    if self.block_event is not None:
      self.block_event.wait()
    if self.fail:
      raise RuntimeError("injected worker failure")
    return None

  def start_transfer_requests(self):
    return [
        r.start_transfer_request
        for _, r in self.calls
        if r.command == raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
    ]


class ControllerV3PyTest(absltest.TestCase):

  def test_public_exports_do_not_import_v1_or_v2(self):
    self.assertIsNotNone(v3.RaidenControllerV3)
    self.assertIsNotNone(v3.VariableBundleSpec)
    # The lease directory is gone (pulls are granted by a per-transfer pull
    # scheduler in C++); its old Python surface must stay gone, and so must
    # the isolation-group knobs and the up-front pull plan views.
    for removed in (
        "PlannedPull",
        "BundleSourceDirectory",
        "AcquireBundlePullLeaseResponse",
        "RegisterBundleAvailabilityResponse",
    ):
      self.assertNotIn(removed, v3.__all__)
      self.assertFalse(hasattr(v3, removed), removed)
    for removed in (
        "has_bundle",
        "register_bundle_availability",
        "acquire_bundle_pull_lease",
        "get_active_uploads",
        "is_replica_complete",
        "are_all_replicas_complete",
        "isolation_group_size",
        "isolation_group_sizes",
    ):
      self.assertFalse(hasattr(v3.RaidenControllerV3, removed), removed)
    self.assertNotIn("RaidenController", v3.__all__)
    self.assertFalse([n for n in v3.__all__ if n.startswith("_")])
    # Depending on the parent `controller_types` module must NOT drag in the
    # v1 broadcast engine or the v2 package.
    self.assertIn(
        "tpu_sync.weight_sync.manager.controller_types",
        sys.modules,
    )
    for mod_name in sys.modules:
      self.assertNotIn(
          "tpu_sync.weight_sync.manager.broadcast_engine", mod_name
      )
      self.assertNotIn("tpu_sync.weight_sync.manager.v2", mod_name)
    self.assertIs(v3.RaidenMemoryType, controller_types.RaidenMemoryType)
    self.assertIs(v3.NameResolver, controller_types.NameResolver)
    self.assertEqual(
        int(v3.RaidenMemoryType.DRAM), int(raiden_service_pb2.MEMORY_TYPE_DRAM)
    )
    self.assertEqual(
        int(v3.RaidenMemoryType.HBM), int(raiden_service_pb2.MEMORY_TYPE_HBM)
    )

  def test_seed_sampler_count_by_host_ratio(self):
    self.assertEqual(
        v3.compute_seed_sampler_count(
            num_samplers=16,
            broadcast_host_ratio=1.0,
            trainer_hosts=8,
            sampler_hosts=2,
        ),
        4,
    )
    self.assertEqual(
        v3.compute_seed_sampler_count(
            num_samplers=16,
            broadcast_host_ratio=0.1,
            trainer_hosts=1,
            sampler_hosts=4,
        ),
        1,
    )
    self.assertEqual(
        v3.compute_seed_sampler_count(
            num_samplers=3,
            broadcast_host_ratio=2.0,
            trainer_hosts=8,
            sampler_hosts=1,
        ),
        3,
    )

  def test_constructor_rejects_bad_resolver_and_sender(self):
    with self.assertRaises(TypeError):
      v3.RaidenControllerV3(name_resolver=lambda ep: ep)  # bare callable
    with self.assertRaises(TypeError):
      v3.RaidenControllerV3(rpc_sender="not callable")

  def test_configurable_bundle_group_count_defaults_to_8(self):
    ctrl = v3.RaidenControllerV3()
    self.addCleanup(ctrl.close)
    self.assertEqual(ctrl.num_bundle_groups, 8)
    self.assertEqual(ctrl.controller_address, "")  # no implicit server

    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(4)]
    _register_fake_topology(ctrl, trainer, samplers, 24, shape=(64, 128))

    plan = ctrl.plan_transfer(
        [trainer], samplers, req_id="bundle_cfg_8", uuid=101
    )
    self.assertIs(ctrl.get_plan("bundle_cfg_8"), plan)
    self.assertLen(plan.variable_bundles, 8)
    # Default R=2 over D=4 replicas: G = min(8 bundles, 4 // 2) = 2 stripes,
    # so every sampler is seeded with one stripe.
    self.assertEqual(plan.num_stripes, 2)
    self.assertEqual(plan.seed_replication, 2)
    self.assertEqual(plan.stripe_seeds, ((0, 1), (2, 3)))
    self.assertEqual(plan.seed_units, tuple(samplers))
    self.assertEqual(plan.src_units, (trainer,))

    ctrl.num_bundle_groups = 4
    ctrl.clear_plan_cache()
    plan4 = ctrl.plan_transfer(
        [trainer], samplers, req_id="bundle_cfg_4", uuid=102
    )
    self.assertLen(plan4.variable_bundles, 4)
    with self.assertRaises(ValueError):
      ctrl.num_bundle_groups = 0

  def test_plan_transfer_retains_plan_without_io_and_executes_later(self):
    sender = _RecordingSender()
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=4), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    _register_fake_topology(ctrl, trainer, samplers, 8)

    plan = ctrl.plan_transfer([trainer], samplers, req_id="plan_only", uuid=7)
    self.assertEmpty(sender.calls)  # planning never touches workers
    self.assertEqual(
        ctrl.get_transfer_status("plan_only"), v3.TransferStatus.NOT_STARTED
    )
    self.assertIs(ctrl.get_plan("plan_only"), plan)
    self.assertEqual(ctrl.get_transfer_record_count(), 1)
    self.assertIs(ctrl.execute_plan(plan), plan)
    self.assertNotEmpty(sender.start_transfer_requests())
    self.assertEqual(
        ctrl.get_transfer_status("plan_only"), v3.TransferStatus.COMPLETED
    )
    self.assertEqual(ctrl.get_plan_materialization_count(), 1)
    with self.assertRaises(ValueError):
      ctrl.plan_transfer([], samplers, req_id="x")
    with self.assertRaises(RuntimeError):
      ctrl.plan_transfer(
          [RaidenId("ghost", "0", "weights")], samplers, req_id="ghost"
      )

  def test_execute_plan_rejects_replaced_plan(self):
    sender = _RecordingSender()
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    _register_fake_topology(ctrl, trainer, samplers, 2)

    old = ctrl.plan_transfer([trainer], samplers, req_id="replaced", uuid=1)
    new = ctrl.plan_transfer([trainer], samplers, req_id="replaced", uuid=2)
    self.assertIs(ctrl.get_plan("replaced"), new)
    with self.assertRaisesRegex(RuntimeError, "no longer retained"):
      ctrl.execute_plan(old)
    self.assertEmpty(sender.calls)
    ctrl.execute_plan(new)
    self.assertEqual(
        ctrl.get_transfer_status("replaced"), v3.TransferStatus.COMPLETED
    )
    # Re-executing a retained plan runs the transfer again.
    num_calls = len(sender.calls)
    ctrl.execute_plan(new)
    self.assertLen(sender.calls, 2 * num_calls)

  def test_changed_options_never_reuse_a_plan_built_for_old_options(self):
    sender = _RecordingSender()
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    _register_fake_topology(ctrl, trainer, samplers, 2)

    def receiver_requests(plan):
      return [c.request() for c in plan.sampler_commands]

    base = ctrl.plan_transfer(
        [trainer],
        samplers,
        req_id="opts",
        uuid=1,
        options=v3.TransferOptions(),
    )
    for req in receiver_requests(base):
      self.assertEqual(req.dst_mem_type, raiden_service_pb2.MEMORY_TYPE_DRAM)
      self.assertEqual(req.parallelism, 1)
      self.assertFalse(req.skip_d2h)
      self.assertFalse(req.skip_tiling[0])

    # Memory type / parallelism / skip_d2h reuse the cached logical schedule
    # but the plan must carry the new values.
    changed = ctrl.plan_transfer(
        [trainer],
        samplers,
        req_id="opts",
        uuid=1,
        options=v3.TransferOptions(
            dst_mem_type=v3.RaidenMemoryType.HBM, parallelism=3, skip_d2h=True
        ),
    )
    self.assertEqual(ctrl.get_plan_cache_size(), 1)
    self.assertIs(ctrl.get_plan("opts"), changed)
    for req in receiver_requests(changed):
      self.assertEqual(req.dst_mem_type, raiden_service_pb2.MEMORY_TYPE_HBM)
      self.assertEqual(req.parallelism, 3)
      self.assertTrue(req.skip_d2h)
    # skip_tiling changes the logical schedule itself.
    tiled = ctrl.plan_transfer(
        [trainer],
        samplers,
        req_id="opts",
        uuid=1,
        options=v3.TransferOptions(skip_tiling={0: True}),
    )
    self.assertEqual(ctrl.get_plan_cache_size(), 2)
    for req in receiver_requests(tiled):
      self.assertTrue(req.skip_tiling[0])
      self.assertFalse(req.skip_tiling[1])
    ctrl.execute_plan(tiled)
    for req in sender.start_transfer_requests():
      self.assertTrue(req.skip_tiling[0])

    # A server-initiated START_TRANSFER for the stored req_id/uuid with other
    # options must not run the stored plan.
    ctrl.plan_transfer([trainer], samplers, req_id="srv", uuid=5)
    materialized = ctrl.get_plan_materialization_count()
    sender.calls.clear()
    start = raiden_service_pb2.StartTransferRequest(
        req_id="srv",
        uuid=5,
        src_units=[v3.types.raiden_id_to_proto(trainer)],
        dst_units=[v3.types.raiden_id_to_proto(u) for u in samplers],
        dst_mem_type=raiden_service_pb2.MEMORY_TYPE_HBM,
        parallelism=2,
    )
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(
        ctrl._cpp.handle_control_request_bytes(
            raiden_service_pb2.ControlRequest(
                command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
                start_transfer_request=start,
            ).SerializeToString()
        )
    )
    self.assertTrue(resp.success, resp.message)
    ctrl.wait_for_transfer("srv", timeout=10.0)
    self.assertEqual(ctrl.get_plan_materialization_count(), materialized + 1)
    receivers = [r for r in sender.start_transfer_requests() if not r.is_sender]
    self.assertNotEmpty(receivers)
    for req in receivers:
      self.assertEqual(req.dst_mem_type, raiden_service_pb2.MEMORY_TYPE_HBM)
      self.assertEqual(req.parallelism, 2)

  def test_generated_ids_do_not_collide_with_server_initiated_transfers(self):
    sender = _RecordingSender()
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    _register_fake_topology(ctrl, trainer, samplers, 2)

    plan = ctrl.plan_transfer([trainer], samplers)  # ids generated
    self.assertEqual(plan.req_id, f"req_{plan.uuid}")
    # A server-side START_TRANSFER without ids (e.g. from a remote client
    # talking to the embedded server) to a different destination set.
    start = raiden_service_pb2.StartTransferRequest(
        src_units=[v3.types.raiden_id_to_proto(trainer)],
        dst_units=[v3.types.raiden_id_to_proto(samplers[0])],
        skip_d2h=True,
    )
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(
        ctrl._cpp.handle_control_request_bytes(
            raiden_service_pb2.ControlRequest(
                command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
                start_transfer_request=start,
            ).SerializeToString()
        )
    )
    self.assertTrue(resp.success, resp.message)
    deadline = time.monotonic() + 10.0
    while not sender.calls and time.monotonic() < deadline:
      time.sleep(0.01)
    server_req = sender.start_transfer_requests()[0]
    ctrl.wait_for_transfer(server_req.req_id, timeout=10.0)
    self.assertNotEqual(server_req.req_id, plan.req_id)
    self.assertNotEqual(server_req.uuid, plan.uuid)

    # The Python plan is untouched and still targets both samplers.
    self.assertIs(ctrl.get_plan(plan.req_id), plan)
    sender.calls.clear()
    ctrl.execute_plan(plan)
    receivers = {
        ep for ep, r in sender.calls if not r.start_transfer_request.is_sender
    }
    self.assertEqual(receivers, {"127.0.0.1:9100", "127.0.0.1:9101"})
    self.assertEqual(
        {r.uuid for r in sender.start_transfer_requests()}, {plan.uuid}
    )

  def test_setters_keep_config_in_sync(self):
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(
            num_bundle_groups=2,
            num_stripes=3,
            seed_replication=1,
            grant_batch_size=6,
            lease_timeout_s=4.0,
            long_poll_timeout_s=0.5,
            transfer_timeout_s=12.0,
        )
    )
    self.addCleanup(ctrl.close)
    original = ctrl.config
    # Construction-time values reach the C++ controller.
    self.assertEqual(ctrl.num_stripes, 3)
    self.assertEqual(ctrl.seed_replication, 1)
    self.assertEqual(ctrl.grant_batch_size, 6)
    self.assertEqual(ctrl._cpp.grant_batch_size, 6)
    self.assertEqual(ctrl.max_concurrent_uploads_per_source, 8)
    self.assertEqual(ctrl.lease_timeout_s, 4.0)
    self.assertEqual(ctrl._cpp.lease_timeout_ms, 4000)
    self.assertEqual(ctrl.long_poll_timeout_s, 0.5)
    self.assertEqual(ctrl._cpp.long_poll_timeout_ms, 500)
    self.assertEqual(ctrl.transfer_timeout_s, 12.0)
    self.assertEqual(ctrl._cpp.transfer_timeout_ms, 12000)
    ctrl.num_bundle_groups = 5
    ctrl.broadcast_host_ratio = 0.5
    ctrl.grant_batch_size = 2
    ctrl.max_concurrent_uploads_per_source = 3
    ctrl.num_stripes = 0
    ctrl.seed_replication = 3
    ctrl.lease_timeout_s = 2.5
    ctrl.long_poll_timeout_s = 1.25
    ctrl.transfer_timeout_s = 3.5
    ctrl.request_registry_ttl_s = 7.0
    cfg = ctrl.config
    self.assertEqual(cfg.num_bundle_groups, 5)
    self.assertEqual(cfg.broadcast_host_ratio, 0.5)
    self.assertEqual(cfg.max_concurrent_uploads_per_source, 3)
    self.assertEqual(ctrl.max_concurrent_uploads_per_source, 3)
    self.assertEqual((cfg.num_stripes, ctrl.num_stripes), (0, 0))
    self.assertEqual((cfg.seed_replication, ctrl.seed_replication), (3, 3))
    self.assertEqual((cfg.grant_batch_size, ctrl.grant_batch_size), (2, 2))
    self.assertEqual((cfg.lease_timeout_s, ctrl.lease_timeout_s), (2.5, 2.5))
    self.assertEqual(ctrl._cpp.lease_timeout_ms, 2500)
    self.assertEqual(
        (cfg.long_poll_timeout_s, ctrl.long_poll_timeout_s), (1.25, 1.25)
    )
    self.assertEqual(ctrl._cpp.long_poll_timeout_ms, 1250)
    self.assertEqual(
        (cfg.transfer_timeout_s, ctrl.transfer_timeout_s), (3.5, 3.5)
    )
    self.assertEqual(ctrl._cpp.transfer_timeout_ms, 3500)
    self.assertEqual(cfg.request_registry_ttl_s, 7.0)
    self.assertEqual(original.num_bundle_groups, 2)  # snapshot not mutated
    self.assertEqual(original.seed_replication, 1)
    # Invalid values are rejected without touching the C++ controller.
    with self.assertRaises(ValueError):
      ctrl.num_stripes = -1
    with self.assertRaises(ValueError):
      ctrl.seed_replication = 0
    with self.assertRaises(ValueError):
      ctrl.lease_timeout_s = 0
    with self.assertRaises(ValueError):
      ctrl.long_poll_timeout_s = 0
    # k_b <= c holds across both setters.
    with self.assertRaisesRegex(ValueError, "grant_batch_size"):
      ctrl.grant_batch_size = 4
    with self.assertRaisesRegex(ValueError, "grant_batch_size"):
      ctrl.max_concurrent_uploads_per_source = 1
    with self.assertRaises(ValueError):
      ctrl.grant_batch_size = 0
    for bad_timeout in (0, -1.0, 0.0004):
      with self.assertRaises(ValueError):
        ctrl.transfer_timeout_s = bad_timeout
    self.assertEqual(
        (
            ctrl.num_stripes,
            ctrl.seed_replication,
            ctrl.grant_batch_size,
            ctrl.max_concurrent_uploads_per_source,
            ctrl.lease_timeout_s,
            ctrl.long_poll_timeout_s,
            ctrl.transfer_timeout_s,
        ),
        (0, 3, 2, 3, 2.5, 1.25, 3.5),
    )
    self.assertEqual(ctrl.config, cfg)

  def test_transfer_timeout_fails_late_transfers_and_bounds_waits(self):
    def slow_trainer(endpoint, unused_req_bytes):
      if endpoint == "127.0.0.1:9000":  # The Trainer's control endpoint.
        time.sleep(0.5)
      return None

    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2, transfer_timeout_s=30.0),
        rpc_sender=slow_trainer,
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", "0", "weights")]
    _register_fake_topology(ctrl, trainer, samplers, 2)

    # `execute_plan` honors the timeout of the options the plan was built with.
    plan = ctrl.plan_transfer(
        [trainer],
        samplers,
        req_id="short",
        uuid=1,
        options=v3.TransferOptions(timeout_s=0.1),
    )
    with self.assertRaisesRegex(RuntimeError, "passed its deadline"):
      ctrl.execute_plan(plan)
    self.assertEqual(
        ctrl.get_transfer_status("short"), v3.TransferStatus.FAILED
    )
    # Changing the timeout never invalidates the plan.
    ctrl.transfer_timeout_s = 0.2
    self.assertIs(ctrl.get_plan("short"), plan)

    # The controller-wide timeout, and a per-transfer override that wins.
    with self.assertRaisesRegex(RuntimeError, "passed its deadline"):
      ctrl.sync_weights([trainer], samplers, req_id="late", uuid=2, timeout=10)
    self.assertEqual(ctrl.get_transfer_status("late"), v3.TransferStatus.FAILED)
    ctrl.sync_weights(
        [trainer],
        samplers,
        req_id="override",
        uuid=3,
        options=v3.TransferOptions(timeout_s=30.0),
        timeout=10,
    )
    self.assertEqual(
        ctrl.get_transfer_status("override"), v3.TransferStatus.COMPLETED
    )

    # `sync_weights(timeout=...)` only stops waiting.
    ctrl.transfer_timeout_s = 30.0
    with self.assertRaises(TimeoutError):
      ctrl.sync_weights(
          [trainer], samplers, req_id="waiter", uuid=4, timeout=0.1
      )
    self.assertEqual(
        ctrl.get_transfer_status("waiter"), v3.TransferStatus.IN_PROGRESS
    )
    ctrl.wait_for_transfer("waiter")
    self.assertEqual(
        ctrl.get_transfer_status("waiter"), v3.TransferStatus.COMPLETED
    )

    # Server-initiated transfers use the controller's timeout, and
    # `wait_for_transfer` follows their deadline by default.
    def start_on_server(req_id, uuid):
      start = raiden_service_pb2.StartTransferRequest(
          req_id=req_id,
          uuid=uuid,
          src_units=[v3.types.raiden_id_to_proto(trainer)],
          dst_units=[v3.types.raiden_id_to_proto(u) for u in samplers],
      )
      resp = raiden_service_pb2.ControlResponse()
      resp.ParseFromString(
          ctrl._cpp.handle_control_request_bytes(
              raiden_service_pb2.ControlRequest(
                  command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
                  start_transfer_request=start,
              ).SerializeToString()
          )
      )
      self.assertTrue(resp.success, resp.message)

    start_on_server("srv_slow", 5)
    with self.assertRaisesRegex(RuntimeError, "Timed out waiting"):
      ctrl.wait_for_transfer("srv_slow", timeout=0.05)
    self.assertEqual(
        ctrl.get_transfer_status("srv_slow"), v3.TransferStatus.IN_PROGRESS
    )
    ctrl.wait_for_transfer("srv_slow")
    ctrl.transfer_timeout_s = 0.2
    start_on_server("srv_late", 6)
    with self.assertRaisesRegex(RuntimeError, "passed its deadline"):
      ctrl.wait_for_transfer("srv_late")
    self.assertEqual(
        ctrl.get_transfer_status("srv_late"), v3.TransferStatus.FAILED
    )

  def test_sync_weights_async_is_eager_and_completes_without_wait(self):
    sender = _RecordingSender()
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(3)]
    _register_fake_topology(ctrl, trainer, samplers, 4)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      future = ctrl.sync_weights_async(
          [trainer], samplers, req_id="eager", uuid=11
      )
      self.assertEqual(future.req_id, "eager")
      # Never call wait()/result(): the transfer must still run to completion.
      deadline = time.monotonic() + 10.0
      while (
          ctrl.get_transfer_status("eager") != v3.TransferStatus.COMPLETED
          and time.monotonic() < deadline
      ):
        time.sleep(0.01)
      del future
      gc.collect()
    self.assertEqual(
        ctrl.get_transfer_status("eager"), v3.TransferStatus.COMPLETED
    )
    self.assertFalse(
        [w for w in caught if issubclass(w.category, RuntimeWarning)],
        f"unexpected RuntimeWarnings: {[str(w.message) for w in caught]}",
    )
    # Every destination host and the trainer host received a START_TRANSFER.
    endpoints = {ep for ep, _ in sender.calls}
    self.assertIn("127.0.0.1:9000", endpoints)
    for i in range(3):
      self.assertIn(f"127.0.0.1:{9100 + i}", endpoints)

  def test_sync_weights_propagates_worker_failure_and_marks_failed(self):
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2),
        rpc_sender=_RecordingSender(fail=True),
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", "0", "weights")]
    _register_fake_topology(ctrl, trainer, samplers, 2)
    with self.assertRaisesRegex(RuntimeError, "injected worker failure"):
      ctrl.sync_weights([trainer], samplers, req_id="boom", uuid=12, timeout=10)
    self.assertEqual(ctrl.get_transfer_status("boom"), v3.TransferStatus.FAILED)
    fut = ctrl.sync_weights_async([trainer], samplers, req_id="boom2", uuid=13)
    self.assertIsInstance(fut.exception(timeout=10), RuntimeError)
    self.assertTrue(fut.done())

  def test_future_timeout_does_not_cancel_running_transfer(self):
    release = threading.Event()
    sender = _RecordingSender(block_event=release)
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    self.addCleanup(release.set)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", "0", "weights")]
    _register_fake_topology(ctrl, trainer, samplers, 2)

    fut = ctrl.sync_weights_async([trainer], samplers, req_id="slow", uuid=14)
    with self.assertRaises(TimeoutError):
      fut.wait_threadsafe(timeout=0.2)
    self.assertFalse(fut.done())
    self.assertFalse(fut.cancel())  # already running
    self.assertEqual(
        ctrl.get_transfer_status("slow"), v3.TransferStatus.IN_PROGRESS
    )
    release.set()
    plan = fut.result(timeout=10)
    self.assertEqual(plan.req_id, "slow")
    self.assertEqual(
        ctrl.get_transfer_status("slow"), v3.TransferStatus.COMPLETED
    )

  def test_future_wait_from_asyncio_loop(self):
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=_RecordingSender()
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", "0", "weights")]
    _register_fake_topology(ctrl, trainer, samplers, 2)
    fut = ctrl.sync_weights_async([trainer], samplers, req_id="aio", uuid=15)

    async def _go():
      return await asyncio.wait_for(fut.wait(), timeout=10.0)

    plan = asyncio.run(_go())
    self.assertEqual(plan.req_id, "aio")

  def test_close_is_idempotent_and_context_manager(self):
    with v3.RaidenControllerV3() as ctrl:
      port = ctrl.start_server()
      self.assertGreater(port, 0)
      self.assertEqual(ctrl.port, port)
      self.assertTrue(ctrl.controller_address.endswith(f":{port}"))
    self.assertEqual(ctrl.controller_address, "")
    ctrl.close()  # idempotent
    with self.assertRaises(RuntimeError):
      ctrl.start_server()
    with self.assertRaises(RuntimeError):
      ctrl.plan_transfer(
          [RaidenId("t", "0", "w")], [RaidenId("s", "0", "w")], req_id="x"
      )

  def test_drop_without_close_does_not_deadlock_with_python_sender(self):
    """Regression: C++ dtor joins transfer threads while holding the GIL."""
    release = threading.Event()
    entered = threading.Event()

    def sender(endpoint, req_bytes):
      del endpoint, req_bytes
      entered.set()
      release.wait()
      return None

    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2), rpc_sender=sender
    )
    ctrl.start_server()
    facade = v3.RaidenControllerClientFacade(f"127.0.0.1:{ctrl.port}")
    self.addCleanup(facade.close)
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    _register_fake_topology(ctrl, trainer, [sampler], 2)
    # Server-spawned transfer: its C++ thread is now blocked inside `sender`.
    facade.start_transfer_async([trainer], [sampler], req_id="drop", uuid=1)
    self.assertTrue(entered.wait(10))
    threading.Timer(0.5, release.set).start()
    start = time.monotonic()
    del ctrl  # no close(): must stop the server with the GIL released.
    gc.collect()
    self.assertLess(time.monotonic() - start, 10.0)

  def test_controller_address_override_is_visible_before_start_server(self):
    ctrl = v3.RaidenControllerV3()
    self.addCleanup(ctrl.close)
    self.assertEqual(ctrl.controller_address, "")
    ctrl.controller_address = "10.1.2.3:4444"
    self.assertEqual(ctrl.controller_address, "10.1.2.3:4444")
    ctrl.start_server()
    self.assertEqual(ctrl.controller_address, "10.1.2.3:4444")
    ctrl.stop_server()
    self.assertEqual(ctrl.controller_address, "10.1.2.3:4444")

  def test_concurrent_async_transfers_exceed_max_workers_mixed_outcomes(self):
    failing = {f"c{i}" for i in range(0, 12, 3)}

    class _Sender(_RecordingSender):

      def __call__(self, endpoint, req_bytes):
        req = raiden_service_pb2.ControlRequest()
        req.ParseFromString(req_bytes)
        if (
            req.command
            == raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
            and req.start_transfer_request.req_id in failing
        ):
          raise RuntimeError("injected " + req.start_transfer_request.req_id)
        return super().__call__(endpoint, req_bytes)

    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2),
        rpc_sender=_Sender(),
        max_workers=3,
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    _register_fake_topology(ctrl, trainer, samplers, 2)
    futs = {
        f"c{i}": ctrl.sync_weights_async(
            [trainer], samplers, req_id=f"c{i}", uuid=i + 1
        )
        for i in range(12)
    }
    for req_id, fut in futs.items():
      if req_id in failing:
        self.assertIsInstance(fut.exception(timeout=20), RuntimeError)
        self.assertEqual(
            ctrl.get_transfer_status(req_id), v3.TransferStatus.FAILED
        )
      else:
        self.assertEqual(fut.result(timeout=20).req_id, req_id)
        self.assertEqual(
            ctrl.get_transfer_status(req_id), v3.TransferStatus.COMPLETED
        )

  def test_import_remote_metadata_copies_registrations(self):
    src = v3.RaidenControllerV3()
    self.addCleanup(src.close)
    src.start_server()
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    _register_fake_topology(src, trainer, [sampler], 2)

    dst = v3.RaidenControllerV3()
    self.addCleanup(dst.close)
    self.assertFalse(dst.has_unit(trainer))
    imported = dst.import_remote_metadata(
        f"127.0.0.1:{src.port}", units=[trainer]
    )
    self.assertEqual(imported, [trainer])
    self.assertTrue(dst.has_unit(trainer))
    self.assertFalse(dst.has_unit(sampler))
    self.assertEqual(dst.get_registered_units(), [trainer])
    self.assertEqual(
        dst.get_all_metadata()[0].control_plane_rpc_address, "127.0.0.1:9000"
    )

  def test_slice_math_and_bundle_spec_helpers(self):
    s1 = v3.NDSlice.from_bounds([(0, 64), (0, 256)])
    s2 = v3.NDSlice.from_bounds([(32, 64), (128, 256)])
    self.assertEqual(s1.shape, (64, 256))
    self.assertLen(s1, 2)
    inter = v3.intersect_nd_slices(s1, s2)
    self.assertEqual(inter, [(32, 64), (128, 256)])
    self.assertTrue(v3.is_nd_slice_tile_aligned(s1, s1, inter))
    self.assertEqual(v3.get_global_indices(3, [2, 2]), [1, 1])
    phys_shape, phys_mesh = v3.to_physical([64, 128], [2, 4], [0, 1])
    self.assertEqual(phys_shape, (128, 64))
    self.assertEqual(phys_mesh, (4, 2))
    self.assertEqual(v3.to_physical([64, 128], [2, 4]), ((64, 128), (2, 4)))
    self.assertNotEmpty(
        v3.generate_strided_copy_chunks(s1, s1, inter, itemsize=2)
    )
    self.assertNotEmpty(
        v3.generate_strided_copy_chunks_tile_aware(s1, s1, inter, itemsize=2)
    )

    bundle = v3.VariableBundleSpec(
        bundle_id=3,
        layer_indices=(0, 1),
        layer_byte_sizes=(1024, 2048),
        total_bytes=3072,
    )
    bundle_rt = v3.VariableBundleSpec.from_cpp(bundle.to_cpp())
    self.assertEqual(bundle_rt, bundle)
    bundles = v3.partition_variable_bundles(4, [10, 10, 10, 10], 2)
    self.assertLen(bundles, 2)

  def test_striped_seed_layout(self):
    sender = _RecordingSender()
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=8), rpc_sender=sender
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(4)]
    _register_fake_topology(ctrl, trainer, samplers, 24, shape=(64, 128))
    all_bundles = list(range(8))

    plan = ctrl.plan_transfer([trainer], samplers, req_id="striped", uuid=201)
    desc = plan.describe_plan()
    self.assertEqual(desc["num_stripes"], 2)
    self.assertEqual(desc["seed_replication"], 2)
    self.assertEqual(
        [tuple(s) for s in desc["stripe_seeds"]], list(plan.stripe_seeds)
    )
    # Stripes are contiguous bundle ranges covering every bundle once.
    self.assertEqual(
        list(itertools.chain.from_iterable(plan.stripe_bundles)), all_bundles
    )
    stripe_of_replica = {}
    for g, seeds in enumerate(plan.stripe_seeds):
      for r in seeds:
        stripe_of_replica[r] = g
    seeds_of_bundle = {}
    for g, bundles in enumerate(plan.stripe_bundles):
      for b in bundles:
        seeds_of_bundle[b] = plan.stripe_seeds[g]
    self.assertCountEqual(seeds_of_bundle, all_bundles)
    self.assertNotIn("replicas", desc)  # Pulls are scheduled at run time.

    # One sampler command per sampler host, bound to physical endpoints; every
    # replica is seeded with its own stripe.
    self.assertEqual(
        [c.replica_idx for c in plan.sampler_commands], [0, 1, 2, 3]
    )
    for cmd in plan.sampler_commands:
      self.assertEqual(cmd.unit, samplers[cmd.replica_idx])
      self.assertEqual(
          cmd.control_endpoint, f"127.0.0.1:{9100 + cmd.replica_idx}"
      )
      self.assertFalse(cmd.request().is_sender)
      self.assertEqual(
          list(cmd.seeded_bundles),
          list(plan.stripe_bundles[stripe_of_replica[cmd.replica_idx]]),
      )
      self.assertNotEmpty(cmd.seeded_layers)
      self.assertFalse(hasattr(cmd, "pulls"))

    # Trainer waves push every (seed, stripe) pair exactly once.
    pushes = list(itertools.chain.from_iterable(plan.trainer_waves))
    expected_pushes = []
    for g, seeds in enumerate(plan.stripe_seeds):
      expected_pushes.extend((r, g) for r in seeds)
    self.assertCountEqual(pushes, expected_pushes)
    self.assertLen(plan.trainer_wave_commands, len(plan.trainer_waves))
    for wave in plan.trainer_wave_commands:
      self.assertLen(wave, 1)  # one Trainer host
      self.assertEqual(wave[0].control_endpoint, "127.0.0.1:9000")
      self.assertTrue(wave[0].request().is_sender)
    ctrl.execute_plan(plan)
    trainer_pushes = [
        r
        for ep, r in sender.calls
        if ep == "127.0.0.1:9000" and r.start_transfer_request.is_sender
    ]
    self.assertLen(trainer_pushes, len(plan.trainer_waves))

    # seed_replication=1 gives G = min(#bundles, D) single-seed stripes.
    ctrl.seed_replication = 1
    plan_r1 = ctrl.plan_transfer([trainer], samplers, req_id="r1", uuid=202)
    self.assertEqual((plan_r1.num_stripes, plan_r1.seed_replication), (4, 1))
    self.assertEqual(plan_r1.stripe_seeds, ((0,), (1,), (2,), (3,)))
    self.assertEqual(plan_r1.describe_plan()["num_stripes"], 4)

    # One full stripe on two seeds spread over the replica order; the other
    # replicas are not seeded and pull everything.
    ctrl.seed_replication = 2
    ctrl.num_stripes = 1
    plan_g1 = ctrl.plan_transfer([trainer], samplers, req_id="g1", uuid=203)
    self.assertEqual(plan_g1.num_stripes, 1)
    self.assertEqual(plan_g1.stripe_seeds, ((0, 2),))
    self.assertEqual(plan_g1.seed_units, (samplers[0], samplers[2]))
    for r in (1, 3):
      self.assertEmpty(plan_g1.sampler_commands[r].seeded_bundles)

    # Requested stripes are capped at D / R; replication is capped at D.
    ctrl.num_stripes = 8
    self.assertEqual(
        ctrl.plan_transfer(
            [trainer], samplers, req_id="cap", uuid=204
        ).num_stripes,
        2,
    )
    ctrl.num_stripes = 0
    ctrl.seed_replication = 8
    plan_full = ctrl.plan_transfer([trainer], samplers, req_id="full", uuid=205)
    self.assertEqual(
        (plan_full.num_stripes, plan_full.seed_replication), (1, 4)
    )
    self.assertEqual(plan_full.seed_units, tuple(samplers))

  def test_attach_host_synchronizes_with_cpp_and_invalidates_unit_cache(self):
    ctrl = v3.RaidenControllerV3()
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    dst0 = RaidenId("sampler", "0", "weights")
    dst1 = RaidenId("sampler", "1", "weights")
    _register_fake_topology(ctrl, trainer, [dst0, dst1], 1)

    ctrl.plan_transfer(
        [trainer], [dst0, dst1], req_id="before_attach", uuid=301
    )
    self.assertEqual(ctrl.get_plan_cache_size(), 1)
    ctrl.attach_host(dst1, "127.0.0.1:9991", shards=["127.0.0.1:8991"])
    self.assertEqual(ctrl.get_plan_cache_size(), 0)
    plan = ctrl.plan_transfer(
        [trainer], [dst0, dst1], req_id="after_attach", uuid=302
    )
    # One stripe seeded on both replicas; dst1 now spans two hosts (original +
    # attached), each with its own receiver command.
    self.assertEqual(plan.seed_units, (dst0, dst1))
    self.assertEqual(
        [
            (c.replica_idx, c.unit, c.control_endpoint)
            for c in plan.sampler_commands
        ],
        [
            (0, dst0, "127.0.0.1:9100"),
            (1, dst1, "127.0.0.1:9101"),
            (1, dst1, "127.0.0.1:9991"),
        ],
    )
    # attach_host with shards=None must be accepted, invalidate the cache and
    # append (not replace) the control endpoint on the existing unit.
    ctrl.attach_host(dst0, "127.0.0.1:9992")
    self.assertEqual(ctrl.get_plan_cache_size(), 0)
    plan2 = ctrl.plan_transfer(
        [trainer], [dst0, dst1], req_id="attach2", uuid=303
    )
    eps_by_replica = {}
    for c in plan2.sampler_commands:
      eps_by_replica.setdefault(c.replica_idx, []).append(c.control_endpoint)
    self.assertEqual(eps_by_replica[0][0], "127.0.0.1:9100")
    self.assertIn("127.0.0.1:9992", eps_by_replica[0])
    self.assertEqual(eps_by_replica[1][0], "127.0.0.1:9101")

  def test_isolation_options_rejected_and_stripe_setters_drop_plan_cache(self):
    with self.assertRaises(TypeError):
      v3.TransferOptions(**{"isolation_group_size": 2})
    with self.assertRaises(TypeError):
      v3.TransferOptions(**{"isolation_group_sizes": [1, 3]})
    with self.assertRaises(TypeError):
      v3.WeightSyncConfig(**{"isolation_group_size": 2})
    ctrl = v3.RaidenControllerV3(v3.WeightSyncConfig(num_bundle_groups=8))
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(4)]
    _register_fake_topology(ctrl, trainer, samplers, 8)

    plan1 = ctrl.plan_transfer([trainer], samplers, req_id="k1", uuid=401)
    self.assertEqual(plan1.num_stripes, 2)
    self.assertEqual(ctrl.get_plan_cache_size(), 1)

    # Changing the seed layout settings drops the cached schedules, so a new
    # setting never reuses a schedule built for the old one.
    ctrl.seed_replication = 1
    self.assertEqual(ctrl.get_plan_cache_size(), 0)
    plan2 = ctrl.plan_transfer([trainer], samplers, req_id="k2", uuid=402)
    self.assertEqual(plan2.num_stripes, 4)
    self.assertEqual(ctrl.get_plan_cache_size(), 1)
    ctrl.seed_replication = 2
    plan3 = ctrl.plan_transfer([trainer], samplers, req_id="k3", uuid=403)
    self.assertEqual(plan3.stripe_seeds, plan1.stripe_seeds)
    ctrl.num_stripes = 1
    self.assertEqual(ctrl.get_plan_cache_size(), 0)
    plan_g1 = ctrl.plan_transfer([trainer], samplers, req_id="k5", uuid=405)
    self.assertEqual(plan_g1.num_stripes, 1)
    ctrl.num_stripes = 0

    # The pull settings are execution options: they keep the cached schedules
    # and the retained plans.
    materializations = ctrl._cpp.get_plan_materialization_count()
    cache_size = ctrl.get_plan_cache_size()
    ctrl.grant_batch_size = 2
    ctrl.max_concurrent_uploads_per_source = 3
    ctrl.lease_timeout_s = 1.5
    ctrl.long_poll_timeout_s = 0.5
    self.assertEqual(ctrl.get_plan_cache_size(), cache_size)
    self.assertEqual(
        ctrl._cpp.get_plan_materialization_count(), materializations
    )
    self.assertIsNotNone(ctrl.get_plan("k1"))

  def test_ttl_setter_evicts_completed_plans(self):
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(request_registry_ttl_s=60.0, num_bundle_groups=2),
        rpc_sender=_RecordingSender(),
    )
    self.addCleanup(ctrl.close)
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    _register_fake_topology(ctrl, trainer, [sampler], 1)
    ctrl.sync_weights(
        [trainer], [sampler], req_id="ttl_req", uuid=501, timeout=10
    )
    self.assertIsNotNone(ctrl.get_plan("ttl_req"))
    time.sleep(0.02)
    ctrl.request_registry_ttl_s = 0.005
    self.assertIsNone(ctrl.get_plan("ttl_req"))
    self.assertEqual(
        ctrl.get_transfer_status("ttl_req"), v3.TransferStatus.NOT_STARTED
    )

  def test_e2e_trainer_seed_push_and_dynamic_bundle_pull_swarm(self):
    """1 trainer (TP=2) -> 4 samplers (TP=1); 2 stripes, each on 2 seeds."""
    num_layers, shape, item_size, num_samplers = 16, (32, 64), 2, 4
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(
            broadcast_host_ratio=1.0,
            num_bundle_groups=8,
            grant_batch_size=2,
            max_concurrent_uploads_per_source=2,
        )
    )
    self.addCleanup(ctrl.close)
    ctrl.start_server()
    facade = v3.RaidenControllerClientFacade(f"127.0.0.1:{ctrl.port}")
    self.addCleanup(facade.close)

    ws_trainer = _cpu_ws(num_layers, 2, [32 * 32 * item_size] * num_layers)
    self.addCleanup(ws_trainer.shutdown)
    ws_samplers = []
    for _ in range(num_samplers):
      ws_s = _cpu_ws(num_layers, 1, [32 * 64 * item_size] * num_layers)
      self.addCleanup(ws_s.shutdown)
      ws_samplers.append(ws_s)

    trainer_unit = RaidenId("trainer", "0", "weights")
    sampler_units = [
        RaidenId("sampler", str(i), "weights") for i in range(num_samplers)
    ]
    src_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name=f"layer_{l}",
            shape=list(shape),
            mesh_shape=[1, 2],
            layout=[1, 0],
            item_size=item_size,
            layer_idx=l,
            global_shard_indices=[0, 1],
        )
        for l in range(num_layers)
    ]
    dst_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name=f"layer_{l}",
            shape=list(shape),
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=item_size,
            layer_idx=l,
            global_shard_indices=[0],
        )
        for l in range(num_layers)
    ]
    facade.register_work_unit(
        trainer_unit,
        [f"127.0.0.1:{ws_trainer.local_port}"] * 2,
        control_plane_rpc_address=f"127.0.0.1:{ws_trainer.listener_port}",
        variables=src_vars,
    )
    for s_unit, ws_s in zip(sampler_units, ws_samplers):
      facade.register_work_unit(
          s_unit,
          [f"127.0.0.1:{ws_s.local_port}"],
          control_plane_rpc_address=f"127.0.0.1:{ws_s.listener_port}",
          variables=dst_vars,
      )
    self.assertTrue(ctrl.has_unit(trainer_unit))

    expected_layers: list[np.ndarray] = []
    for l in range(num_layers):
      shard_arrays = []
      for s_idx in range(2):
        words = ws_trainer.get_host_buffer(layer_idx=l, shard_idx=s_idx).view(
            np.uint16
        )
        base_tag = np.uint16(((l + 1) << 8) | ((s_idx + 1) << 4))
        pattern = base_tag | (
            np.arange(32 * 32, dtype=np.uint16) & np.uint16(0x0F)
        )
        words[: 32 * 32] = pattern
        shard_arrays.append(pattern.reshape((32, 32)))
      expected_layers.append(np.concatenate(shard_arrays, axis=1))
    for ws_s in ws_samplers:
      for l in range(num_layers):
        ws_s.get_host_buffer(layer_idx=l, shard_idx=0)[:] = 0

    uuid, req_id = 9001, "v3_e2e_swarm_step"
    future = ctrl.sync_weights_async(
        [trainer_unit],
        sampler_units,
        req_id=req_id,
        uuid=uuid,
        options=v3.TransferOptions(
            dst_mem_type=v3.RaidenMemoryType.DRAM,
            skip_d2h=True,
            skip_tiling={l: False for l in range(num_layers)},
        ),
    )
    plan = ctrl.get_plan(req_id)
    self.assertEqual(plan.num_stripes, 2)
    self.assertEqual(plan.stripe_seeds, ((0, 1), (2, 3)))
    self.assertEqual(plan.seed_units, tuple(sampler_units))
    for cmd in plan.sampler_commands:  # every sampler pulls the other stripe
      self.assertLen(cmd.seeded_bundles, 4)
    self.assertLen(plan.variable_bundles, 8)
    self.assertTrue(plan.options.skip_d2h)

    async def _await():
      return await asyncio.wait_for(future.wait(), timeout=15.0)

    self.assertIs(asyncio.run(_await()), plan)

    def _wait_fn(ws, errs):
      try:
        ws.wait_for_transfer_completion(uuid=uuid)
      except Exception as e:  # pylint: disable=broad-except
        errs.append(e)

    for s_idx, ws_s in enumerate(ws_samplers):
      err_list: list[Exception] = []
      t = threading.Thread(target=_wait_fn, args=(ws_s, err_list), daemon=True)
      t.start()
      t.join(timeout=10.0)
      self.assertFalse(t.is_alive(), f"Sampler {s_idx} timed out waiting")
      self.assertEmpty(err_list)

    self.assertEqual(
        ctrl.get_transfer_status(req_id), v3.TransferStatus.COMPLETED
    )
    # The scheduler granted every sampler the 4 bundles of the other stripe.
    stats = ctrl._cpp.get_pull_stats(req_id)
    self.assertLen(stats, 1)  # one host index
    self.assertEqual(stats[0]["completions"], 4 * 4)
    self.assertEqual(stats[0]["failures"], 0)
    self.assertEqual(stats[0]["done_hosts"], 4)
    for s_idx, ws_s in enumerate(ws_samplers):
      for l, expected in enumerate(expected_layers):
        buf = ws_s.get_host_buffer(layer_idx=l, shard_idx=0)
        actual = buf.view(np.uint16)[: expected.size].reshape(expected.shape)
        self.assertTrue(
            np.array_equal(actual, expected),
            f"Parity mismatch on sampler {s_idx} layer {l}",
        )

  def test_server_port_zero_init_and_single_plan_materialization_and_ttl(self):
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(request_registry_ttl_s=0.03)
    )
    self.addCleanup(ctrl.close)
    self.assertEqual(ctrl.controller_address, "")
    bound_port = ctrl.start_server()
    self.assertGreater(bound_port, 0)
    self.assertTrue(ctrl.controller_address.endswith(f":{bound_port}"))
    self.assertEqual(ctrl.start_server(), bound_port)  # idempotent

    ws_trainer = _cpu_ws(2, 1, [128, 128])
    self.addCleanup(ws_trainer.shutdown)
    ws_sampler = _cpu_ws(2, 1, [128, 128])
    self.addCleanup(ws_sampler.shutdown)
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    vars_proto = _vars(2, shape=(8, 8))
    ctrl.register_synchronizer(
        trainer, ws_trainer, variables=vars_proto, bind_ip="127.0.0.1"
    )
    ctrl.register_synchronizer(
        sampler, ws_sampler, variables=vars_proto, bind_ip="127.0.0.1"
    )
    self.assertEqual(ctrl.get_plan_materialization_count(), 0)
    plan = ctrl.sync_weights(
        [trainer],
        [sampler],
        req_id="single_mat_req",
        uuid=777,
        options=v3.TransferOptions(skip_d2h=True),
        timeout=10.0,
    )
    self.assertEqual(plan.req_id, "single_mat_req")
    self.assertEqual(ctrl.get_plan_materialization_count(), 1)
    ws_sampler.wait_for_transfer_completion(uuid=777)
    time.sleep(0.05)
    self.assertIsNone(ctrl.get_plan("single_mat_req"))

  def test_register_synchronizer_auto_extracts_variables_and_multi_host_idx(
      self,
  ):
    ctrl = v3.RaidenControllerV3()
    self.addCleanup(ctrl.close)
    ws_t0, ws_t1, ws_s0, ws_s1 = (_cpu_ws(2, 1, [128, 128]) for _ in range(4))
    for ws in (ws_t0, ws_t1, ws_s0, ws_s1):
      self.addCleanup(ws.shutdown)
    for l in range(2):
      np.frombuffer(ws_t0.get_host_buffer(l, 0), dtype=np.uint8)[:] = 10 + l
      np.frombuffer(ws_t1.get_host_buffer(l, 0), dtype=np.uint8)[:] = 20 + l
      np.frombuffer(ws_s0.get_host_buffer(l, 0), dtype=np.uint8)[:] = 0
      np.frombuffer(ws_s1.get_host_buffer(l, 0), dtype=np.uint8)[:] = 0

    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    # The CPU synchronizers keep their placement private, so each host passes
    # the global shard index of its one local shard.
    for unit, host_idx, ws in (
        (trainer, 0, ws_t0),
        (trainer, 1, ws_t1),
        (sampler, 0, ws_s0),
        (sampler, 1, ws_s1),
    ):
      ctrl.register_synchronizer(
          unit,
          ws,
          global_shard_indices=[host_idx],
          host_idx=host_idx,
          bind_ip="127.0.0.1",
      )

    plan = ctrl.sync_weights(
        [trainer],
        [sampler],
        req_id="auto_vars_multi_host",
        uuid=991,
        options=v3.TransferOptions(skip_d2h=True),
        timeout=10.0,
    )
    self.assertEqual(plan.req_id, "auto_vars_multi_host")
    ws_s0.wait_for_transfer_completion(uuid=991)
    ws_s1.wait_for_transfer_completion(uuid=991)
    for l in range(2):
      self.assertTrue(
          np.all(
              np.frombuffer(ws_s0.get_host_buffer(l, 0), dtype=np.uint8)
              == 10 + l
          )
      )
      self.assertTrue(
          np.all(
              np.frombuffer(ws_s1.get_host_buffer(l, 0), dtype=np.uint8)
              == 20 + l
          )
      )

  def test_name_resolver_executes_full_two_stage_dynamic_pull(self):
    endpoint_map: dict[str, str] = {}
    resolved_calls: list[str] = []

    class _MapResolver:

      def resolve(self, ep: str) -> str:
        resolved_calls.append(ep)
        return endpoint_map.get(ep, ep)

    # seed_replication=1: each sampler is seeded with one stripe and pulls the
    # other one from its peer (two stages).
    ctrl = v3.RaidenControllerV3(
        v3.WeightSyncConfig(num_bundle_groups=2, seed_replication=1),
        name_resolver=_MapResolver(),
    )
    self.addCleanup(ctrl.close)
    ws_trainer, ws_s0, ws_s1 = (_cpu_ws(2, 1, [128, 128]) for _ in range(3))
    for ws in (ws_trainer, ws_s0, ws_s1):
      self.addCleanup(ws.shutdown)
    for l in range(2):
      np.frombuffer(ws_trainer.get_host_buffer(l, 0), dtype=np.uint8)[:] = (
          77 + l
      )
      np.frombuffer(ws_s0.get_host_buffer(l, 0), dtype=np.uint8)[:] = 0
      np.frombuffer(ws_s1.get_host_buffer(l, 0), dtype=np.uint8)[:] = 0
    endpoint_map["bns://trainer/0"] = f"127.0.0.1:{ws_trainer.listener_port}"
    endpoint_map["bns://sampler/0"] = f"127.0.0.1:{ws_s0.listener_port}"
    endpoint_map["bns://sampler/1"] = f"127.0.0.1:{ws_s1.listener_port}"

    trainer = RaidenId("trainer", "0", "weights")
    s0 = RaidenId("sampler", "0", "weights")
    s1 = RaidenId("sampler", "1", "weights")
    vars_proto = _vars(2, shape=(8, 8))
    ctrl.register_work_unit(
        trainer,
        [f"127.0.0.1:{ws_trainer.local_port}"],
        control_plane_rpc_address="bns://trainer/0",
        variables=vars_proto,
    )
    ctrl.register_work_unit(
        s0,
        [f"127.0.0.1:{ws_s0.local_port}"],
        control_plane_rpc_address="bns://sampler/0",
        variables=vars_proto,
    )
    ctrl.register_work_unit(
        s1,
        [f"127.0.0.1:{ws_s1.local_port}"],
        control_plane_rpc_address="bns://sampler/1",
        variables=vars_proto,
    )
    plan = ctrl.sync_weights(
        [trainer],
        [s0, s1],
        req_id="name_resolver_two_stage",
        uuid=995,
        options=v3.TransferOptions(skip_d2h=True),
        timeout=10.0,
    )
    self.assertEqual(plan.num_stripes, 2)
    self.assertEqual(plan.stripe_seeds, ((0,), (1,)))
    self.assertEqual(
        ctrl._cpp.get_pull_stats("name_resolver_two_stage")[0]["completions"],
        2,
    )
    ws_s0.wait_for_transfer_completion(uuid=995)
    ws_s1.wait_for_transfer_completion(uuid=995)
    for ep in ("bns://trainer/0", "bns://sampler/0", "bns://sampler/1"):
      self.assertIn(ep, resolved_calls)
    for l in range(2):
      for ws in (ws_s0, ws_s1):
        self.assertTrue(
            np.all(
                np.frombuffer(ws.get_host_buffer(l, 0), dtype=np.uint8)[:128]
                == 77 + l
            )
        )


if __name__ == "__main__":
  absltest.main()

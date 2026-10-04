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

"""Tests for `RaidenControllerClientFacade` against a real C++ server."""

import threading
import time

from absl.testing import absltest
import numpy as np

from tpu_sync.api.common import RaidenId
from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import v3


def _vars(num_layers, shape=(32, 64)):
  return [
      raiden_service_pb2.VariableMetadataProto(
          name=f"w_{l}",
          shape=list(shape),
          mesh_shape=[1, 1],
          layout=[1, 0],
          item_size=2,
          layer_idx=l,
          global_shard_indices=[0],
      )
      for l in range(num_layers)
  ]


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

  def __init__(self, block_event=None):
    self.calls: list[tuple[str, raiden_service_pb2.ControlRequest]] = []
    self.block_event = block_event
    self._lock = threading.Lock()

  def __call__(self, endpoint, req_bytes):
    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(req_bytes)
    with self._lock:
      self.calls.append((endpoint, req))
    if self.block_event is not None:
      self.block_event.wait()
    return None

  def start_transfer_requests(self):
    with self._lock:
      return [
          r.start_transfer_request
          for _, r in self.calls
          if r.command
          == raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
      ]


class RemoteFacadePyTest(absltest.TestCase):

  def _serving_controller(self, sender=None, **cfg):
    ctrl = v3.RaidenControllerV3(v3.WeightSyncConfig(**cfg), rpc_sender=sender)
    self.addCleanup(ctrl.close)
    ctrl.start_server()
    facade = v3.RaidenControllerClientFacade(f"127.0.0.1:{ctrl.port}")
    self.addCleanup(facade.close)
    return ctrl, facade

  def _register_fake(self, facade, trainer, samplers, num_layers):
    vars_proto = _vars(num_layers)
    facade.register_work_unit(
        trainer,
        ["127.0.0.1:7000"],
        control_plane_rpc_address="127.0.0.1:7001",
        variables=vars_proto,
    )
    for i, s in enumerate(samplers):
      facade.register_work_unit(
          s,
          [f"127.0.0.1:{7100 + i}"],
          control_plane_rpc_address=f"127.0.0.1:{7200 + i}",
          variables=vars_proto,
      )

  def test_register_and_get_metadata_round_trip(self):
    ctrl, facade = self._serving_controller()
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    self._register_fake(facade, trainer, [sampler], 2)
    self.assertTrue(ctrl.has_unit(trainer))
    self.assertTrue(ctrl.has_unit(sampler))
    meta = facade.get_metadata()
    self.assertLen(meta, 2)
    self.assertEqual(meta[0].control_plane_rpc_address, "127.0.0.1:7001")
    with self.assertRaises(ValueError):
      facade.register_work_unit(trainer, [], control_plane_rpc_address="x")

  def test_options_round_trip_to_worker_requests(self):
    sender = _RecordingSender()
    ctrl, facade = self._serving_controller(sender, num_bundle_groups=2)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    self._register_fake(facade, trainer, samplers, 4)

    result = facade.sync_weights(
        [trainer],
        samplers,
        req_id="facade_opts",
        uuid=321,
        options=v3.TransferOptions(
            dst_mem_type=v3.RaidenMemoryType.HBM,
            skip_d2h=True,
            skip_tiling={0: True, 1: False},
            parallelism=3,
        ),
        timeout=10.0,
    )
    self.assertEqual(result.req_id, "facade_opts")
    self.assertEqual(result.uuid, 321)
    self.assertEqual(result.status, v3.TransferStatus.COMPLETED)
    self.assertEqual(
        facade.get_transfer_status("facade_opts"), v3.TransferStatus.COMPLETED
    )
    self.assertEqual(
        ctrl.get_transfer_status("facade_opts"), v3.TransferStatus.COMPLETED
    )
    reqs = sender.start_transfer_requests()
    self.assertNotEmpty(reqs)
    self.assertTrue(all(r.req_id == "facade_opts" for r in reqs))
    self.assertTrue(all(r.uuid == 321 for r in reqs))
    self.assertTrue(
        all(r.dst_mem_type == raiden_service_pb2.MEMORY_TYPE_HBM for r in reqs)
    )
    self.assertTrue(any(r.skip_d2h for r in reqs))
    self.assertTrue(any(r.parallelism == 3 for r in reqs))
    tiled = [r for r in reqs if r.skip_tiling]
    self.assertNotEmpty(tiled)
    self.assertTrue(tiled[0].skip_tiling[0])
    self.assertFalse(tiled[0].skip_tiling[1])

  def test_raises_on_options_the_wire_cannot_carry(self):
    sender = _RecordingSender()
    _, facade = self._serving_controller(sender, num_bundle_groups=2)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    self._register_fake(facade, trainer, samplers, 2)
    # Per-transfer isolation overrides no longer exist at all.
    with self.assertRaises(TypeError):
      v3.TransferOptions(**{"isolation_group_size": 1})
    with self.assertRaises(TypeError):
      v3.TransferOptions(**{"isolation_group_sizes": [1, 1]})
    with self.assertRaises(ValueError):
      facade.start_transfer_async(
          [trainer],
          samplers,
          req_id="never",
          uuid=1,
          options=v3.TransferOptions(use_cached_plan=False),
      )
    # The serving controller's transfer timeout applies; a per-transfer one
    # cannot be carried and is never dropped silently.
    with self.assertRaisesRegex(ValueError, "transfer_timeout_s applies"):
      facade.start_transfer_async(
          [trainer],
          samplers,
          req_id="never",
          uuid=1,
          options=v3.TransferOptions(timeout_s=5.0),
      )
    with self.assertRaisesRegex(ValueError, "timeout_s"):
      facade.sync_weights(
          [trainer],
          samplers,
          req_id="never",
          uuid=1,
          options=v3.TransferOptions(timeout_s=5.0),
      )
    with self.assertRaisesRegex(ValueError, "timeout_s"):
      facade.sync_weights_async(
          [trainer],
          samplers,
          req_id="never",
          uuid=1,
          options=v3.TransferOptions(timeout_s=5.0),
      )
    with self.assertRaises(ValueError):
      facade.start_transfer_async([trainer], samplers, req_id="r", uuid=-1)
    with self.assertRaises(ValueError):
      facade.start_transfer_async([], samplers, req_id="r", uuid=1)
    self.assertEmpty(sender.calls)
    self.assertEqual(
        facade.get_transfer_status("never"), v3.TransferStatus.NOT_STARTED
    )

  def test_generates_missing_transfer_ids(self):
    sender = _RecordingSender()
    _, facade = self._serving_controller(sender, num_bundle_groups=2)
    trainer = RaidenId("trainer", "0", "weights")
    samplers = [RaidenId("sampler", str(i), "weights") for i in range(2)]
    self._register_fake(facade, trainer, samplers, 2)

    started = [
        facade.start_transfer_async([trainer], samplers),
        facade.start_transfer_async([trainer], samplers, req_id="", uuid=0),
        facade.start_transfer_async([trainer], samplers, req_id="named"),
        facade.start_transfer_async([trainer], samplers, uuid=42),
    ]
    for req_id, uuid in started:
      self.assertGreater(uuid, 0)
      result = facade.wait_for_transfer(req_id, uuid, timeout=10.0)
      self.assertEqual(result.status, v3.TransferStatus.COMPLETED)
    (gen0, uuid0), (gen1, uuid1), (named, _), (explicit, uuid42) = started
    self.assertEqual(gen0, f"req_{uuid0}")
    self.assertEqual(gen1, f"req_{uuid1}")
    self.assertNotEqual(uuid0, uuid1)
    self.assertEqual(named, "named")
    self.assertEqual((explicit, uuid42), ("req_42", 42))
    sent = {(r.req_id, r.uuid) for r in sender.start_transfer_requests()}
    self.assertContainsSubset(set(started), sent)

    result = facade.sync_weights([trainer], samplers, timeout=10.0)
    self.assertGreater(result.uuid, 0)
    self.assertEqual(result.req_id, f"req_{result.uuid}")
    fut = facade.sync_weights_async([trainer], samplers, req_id="async_named")
    self.assertEqual(fut.req_id, "async_named")
    self.assertEqual(
        fut.result(timeout=10.0).status, v3.TransferStatus.COMPLETED
    )

  def test_register_work_unit_drops_stale_synchronizer_hosts(self):
    ctrl, facade = self._serving_controller()
    ws0, ws1 = (_cpu_ws(2, 1, [128, 128]) for _ in range(2))
    for ws in (ws0, ws1):
      self.addCleanup(ws.shutdown)
    trainer = RaidenId("trainer", "0", "weights")
    vars_proto = _vars(2, shape=(8, 8))
    for host_idx, ws in enumerate((ws0, ws1)):
      facade.register_synchronizer(
          trainer,
          ws,
          variables=vars_proto,
          bind_ip="127.0.0.1",
          host_idx=host_idx,
      )
    self.assertLen(facade.get_metadata()[0].shards, 2)
    facade.register_work_unit(
        trainer,
        ["127.0.0.1:7000"],
        control_plane_rpc_address="127.0.0.1:7001",
        variables=vars_proto,
    )
    self.assertLen(facade.get_metadata()[0].shards, 1)
    # A later synchronizer registration starts from scratch instead of
    # resurrecting host 1 from before register_work_unit.
    facade.register_synchronizer(
        trainer, ws0, variables=vars_proto, bind_ip="127.0.0.1", host_idx=0
    )
    self.assertLen(facade.get_metadata()[0].shards, 1)
    self.assertTrue(ctrl.has_unit(trainer))

  def test_remote_failure_surfaces_as_runtime_error(self):
    _, facade = self._serving_controller(
        _RecordingSender(), num_bundle_groups=2
    )
    trainer = RaidenId("trainer", "0", "weights")
    registered = RaidenId("sampler", "0", "weights")
    ghost = RaidenId("sampler", "ghost", "weights")
    self._register_fake(facade, trainer, [registered], 2)
    with self.assertRaisesRegex(RuntimeError, "failed"):
      facade.sync_weights(
          [trainer],
          [registered, ghost],
          req_id="facade_fail",
          uuid=55,
          timeout=10.0,
      )
    self.assertEqual(
        facade.get_transfer_status("facade_fail"), v3.TransferStatus.FAILED
    )
    # Empty unit lists are rejected server-side with a message.
    with self.assertRaisesRegex(RuntimeError, "rejected"):
      facade._control(  # pylint: disable=protected-access
          raiden_service_pb2.ControlRequest(
              command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
              start_transfer_request=raiden_service_pb2.StartTransferRequest(
                  req_id="empty", uuid=1
              ),
          )
      )

  def test_timeout_does_not_cancel_remote_transfer(self):
    release = threading.Event()
    sender = _RecordingSender(block_event=release)
    _, facade = self._serving_controller(sender, num_bundle_groups=2)
    self.addCleanup(release.set)
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    self._register_fake(facade, trainer, [sampler], 2)
    with self.assertRaises(TimeoutError):
      facade.sync_weights(
          [trainer], [sampler], req_id="facade_slow", uuid=66, timeout=0.3
      )
    self.assertEqual(
        facade.get_transfer_status("facade_slow"), v3.TransferStatus.IN_PROGRESS
    )
    release.set()
    result = facade.wait_for_transfer("facade_slow", 66, timeout=10.0)
    self.assertEqual(result.status, v3.TransferStatus.COMPLETED)

  def test_serving_controller_transfer_timeout_bounds_remote_transfers(self):
    def slow_trainer(endpoint, unused_req_bytes):
      if endpoint == "127.0.0.1:7001":  # The Trainer's control endpoint.
        time.sleep(0.5)
      return None

    ctrl, facade = self._serving_controller(
        slow_trainer, num_bundle_groups=2, transfer_timeout_s=0.2
    )
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    self._register_fake(facade, trainer, [sampler], 2)
    with self.assertRaisesRegex(RuntimeError, "failed"):
      facade.sync_weights(
          [trainer], [sampler], req_id="facade_late", uuid=67, timeout=10.0
      )
    self.assertEqual(
        facade.get_transfer_status("facade_late"), v3.TransferStatus.FAILED
    )
    ctrl.transfer_timeout_s = 30.0
    result = facade.sync_weights(
        [trainer], [sampler], req_id="facade_ok", uuid=68, timeout=10.0
    )
    self.assertEqual(result.status, v3.TransferStatus.COMPLETED)

  def test_zero_ttl_still_reports_completion_to_remote_pollers(self):
    # With request_registry_ttl_s=0 the controller keeps a terminal record
    # until its result has been observed once, so a remote poller still sees
    # COMPLETED; the record is evicted at the next pass after that.
    ctrl, facade = self._serving_controller(
        _RecordingSender(), num_bundle_groups=2, request_registry_ttl_s=0.0
    )
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    self._register_fake(facade, trainer, [sampler], 2)
    result = facade.sync_weights(
        [trainer], [sampler], req_id="ttl0", uuid=5, timeout=10.0
    )
    self.assertEqual(result.status, v3.TransferStatus.COMPLETED)
    self.assertEqual(
        facade.get_transfer_status("ttl0"), v3.TransferStatus.NOT_STARTED
    )
    self.assertEqual(ctrl.get_transfer_record_count(), 0)

  def test_sync_weights_async_is_eager(self):
    sender = _RecordingSender()
    _, facade = self._serving_controller(sender, num_bundle_groups=2)
    trainer = RaidenId("trainer", "0", "weights")
    sampler = RaidenId("sampler", "0", "weights")
    self._register_fake(facade, trainer, [sampler], 2)
    fut = facade.sync_weights_async(
        [trainer], [sampler], req_id="facade_async", uuid=77
    )
    deadline = time.monotonic() + 10.0
    while not fut.done() and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertTrue(fut.done())
    self.assertEqual(fut.result().status, v3.TransferStatus.COMPLETED)
    self.assertNotEmpty(sender.start_transfer_requests())

  def test_register_synchronizer_and_sync_weights_e2e(self):
    ctrl, facade = self._serving_controller(
        num_bundle_groups=2, seed_replication=1
    )
    ws_trainer, ws_s0, ws_s1 = (_cpu_ws(2, 1, [128, 128]) for _ in range(3))
    for ws in (ws_trainer, ws_s0, ws_s1):
      self.addCleanup(ws.shutdown)
    for l in range(2):
      np.frombuffer(ws_trainer.get_host_buffer(l, 0), dtype=np.uint8)[:] = (
          40 + l
      )
      np.frombuffer(ws_s0.get_host_buffer(l, 0), dtype=np.uint8)[:] = 0
      np.frombuffer(ws_s1.get_host_buffer(l, 0), dtype=np.uint8)[:] = 0

    trainer = RaidenId("trainer", "0", "weights")
    s0 = RaidenId("sampler", "0", "weights")
    s1 = RaidenId("sampler", "1", "weights")
    vars_proto = _vars(2, shape=(8, 8))
    facade.register_synchronizer(
        trainer, ws_trainer, variables=vars_proto, bind_ip="127.0.0.1"
    )
    facade.register_synchronizer(
        s0, ws_s0, variables=vars_proto, bind_ip="127.0.0.1"
    )
    facade.register_synchronizer(
        s1, ws_s1, variables=vars_proto, bind_ip="127.0.0.1"
    )
    self.assertLen(ctrl.get_registered_units(), 3)

    result = facade.sync_weights(
        [trainer],
        [s0, s1],
        req_id="facade_sync_v3",
        uuid=888,
        options=v3.TransferOptions(
            skip_d2h=True, skip_tiling={0: False, 1: False}, parallelism=1
        ),
        timeout=10.0,
    )
    self.assertEqual(result.status, v3.TransferStatus.COMPLETED)
    ws_s0.wait_for_transfer_completion(uuid=888)
    ws_s1.wait_for_transfer_completion(uuid=888)
    self.assertEqual(
        ctrl.get_transfer_status("facade_sync_v3"), v3.TransferStatus.COMPLETED
    )
    for l in range(2):
      for ws in (ws_s0, ws_s1):
        self.assertTrue(
            np.all(
                np.frombuffer(ws.get_host_buffer(l, 0), dtype=np.uint8)[:128]
                == 40 + l
            )
        )


if __name__ == "__main__":
  absltest.main()

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

"""Unit tests for the pure-Python types in ``manager.v3.types``."""

import concurrent.futures
import dataclasses
import socket
import threading
from unittest import mock

from absl.testing import absltest

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import controller_service_pb2
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager.v3 import _controller_v3
from tpu_sync.weight_sync.manager.v3 import types


class _Obj:
  """Attribute bag standing in for nanobind views of C++ plan objects."""

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class ParentSymbolsTest(absltest.TestCase):

  def test_shared_symbols_are_parent_objects_not_forks(self):
    self.assertIs(types.NameResolver, controller_types.NameResolver)
    self.assertIs(types.RaidenMemoryType, controller_types.RaidenMemoryType)
    self.assertIs(types.VariableMetadata, controller_types.VariableMetadata)
    self.assertIs(
        types.coerce_variable_proto, controller_types.coerce_variable_proto
    )
    self.assertIs(types.extract_host_ip, controller_types.extract_host_ip)
    self.assertIs(
        types.raiden_id_from_proto, controller_types.raiden_id_from_proto
    )
    self.assertIs(types.raiden_id_to_proto, controller_types.raiden_id_to_proto)


class TransferStatusTest(absltest.TestCase):

  def test_values_match_proto_enum(self):
    resp = controller_service_pb2.GetTransferStatusResponse
    self.assertEqual(types.TransferStatus.NOT_STARTED, resp.STATUS_NOT_STARTED)
    self.assertEqual(types.TransferStatus.IN_PROGRESS, resp.STATUS_IN_PROGRESS)
    self.assertEqual(types.TransferStatus.COMPLETED, resp.STATUS_COMPLETED)
    self.assertEqual(types.TransferStatus.FAILED, resp.STATUS_FAILED)

  def test_is_terminal(self):
    self.assertTrue(types.TransferStatus.COMPLETED.is_terminal)
    self.assertTrue(types.TransferStatus.FAILED.is_terminal)
    self.assertFalse(types.TransferStatus.IN_PROGRESS.is_terminal)
    self.assertFalse(types.TransferStatus.NOT_STARTED.is_terminal)
    self.assertFalse(types.TransferStatus.UNSPECIFIED.is_terminal)


class SecondsToMsTest(absltest.TestCase):

  def test_rounds_to_whole_milliseconds(self):
    self.assertEqual(types.seconds_to_ms(30.0), 30000)
    self.assertEqual(types.seconds_to_ms(0.0016), 2)
    self.assertEqual(types.seconds_to_ms(2), 2000)
    self.assertIsInstance(types.seconds_to_ms(1.25), int)


class WeightSyncConfigTest(absltest.TestCase):

  def test_defaults(self):
    cfg = types.WeightSyncConfig()
    self.assertEqual(cfg.port, 0)
    self.assertEqual(cfg.request_registry_ttl_s, 600.0)
    self.assertEqual(cfg.broadcast_host_ratio, 1.0)
    self.assertTrue(cfg.enable_plan_cache)
    self.assertEqual(cfg.num_bundle_groups, 8)
    self.assertEqual(cfg.max_concurrent_uploads_per_source, 8)
    self.assertEqual(cfg.num_stripes, 0)
    self.assertEqual(cfg.seed_replication, 2)
    self.assertEqual(cfg.grant_batch_size, 8)
    self.assertEqual(cfg.lease_timeout_s, 30.0)
    self.assertEqual(cfg.long_poll_timeout_s, 5.0)
    self.assertEqual(cfg.transfer_timeout_s, 600.0)

  def test_frozen(self):
    cfg = types.WeightSyncConfig()
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(cfg, "port", 5)

  def test_validation(self):
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(port=-1)
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(request_registry_ttl_s=-0.1)
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(broadcast_host_ratio=0)
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(num_bundle_groups=0)
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(max_concurrent_uploads_per_source=0)
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(num_stripes=-1)
    with self.assertRaises(ValueError):
      types.WeightSyncConfig(seed_replication=0)
    with self.assertRaisesRegex(ValueError, "grant_batch_size"):
      types.WeightSyncConfig(grant_batch_size=0)
    # A host must not hold more leases than a source serves at once.
    with self.assertRaisesRegex(
        ValueError, "must not exceed max_concurrent_uploads_per_source"
    ):
      types.WeightSyncConfig(max_concurrent_uploads_per_source=4)
    cfg = types.WeightSyncConfig(
        grant_batch_size=4, max_concurrent_uploads_per_source=4
    )
    self.assertEqual(
        (cfg.grant_batch_size, cfg.max_concurrent_uploads_per_source), (4, 4)
    )
    for bad_timeout in (0.0, -1.0, 0.0004, float("nan")):
      with self.assertRaisesRegex(ValueError, "lease_timeout_s"):
        types.WeightSyncConfig(lease_timeout_s=bad_timeout)
      with self.assertRaisesRegex(ValueError, "long_poll_timeout_s"):
        types.WeightSyncConfig(long_poll_timeout_s=bad_timeout)
      with self.assertRaisesRegex(ValueError, "transfer_timeout_s"):
        types.WeightSyncConfig(transfer_timeout_s=bad_timeout)
    self.assertEqual(
        types.WeightSyncConfig(transfer_timeout_s=0.001).transfer_timeout_s,
        0.001,
    )
    cfg = types.WeightSyncConfig(
        num_stripes=4, seed_replication=1, lease_timeout_s=0.001
    )
    self.assertEqual(
        (cfg.num_stripes, cfg.seed_replication, cfg.lease_timeout_s),
        (4, 1, 0.001),
    )

  def test_isolation_fields_are_rejected(self):
    for removed in ("isolation_group_size", "isolation_group_sizes"):
      self.assertNotIn(
          removed, {f.name for f in dataclasses.fields(types.WeightSyncConfig)}
      )
    with self.assertRaises(TypeError):
      types.WeightSyncConfig(**{"isolation_group_size": 2})
    with self.assertRaises(TypeError):
      types.WeightSyncConfig(**{"isolation_group_sizes": (2, 2)})
    self.assertFalse(hasattr(types, "parse_group_sizes"))

  def test_from_env_empty_matches_defaults(self):
    self.assertEqual(
        types.WeightSyncConfig.from_env(env={}), types.WeightSyncConfig()
    )

  def test_from_env_reads_known_variables(self):
    env = {
        "RAIDEN_BROADCAST_HOST_RATIO": "2.5",
        "RAIDEN_NUM_BUNDLE_GROUPS": "16",
        "RAIDEN_MAX_CONCURRENT_UPLOADS_PER_SOURCE": "3",
        "RAIDEN_NUM_STRIPES": "4",
        "RAIDEN_SEED_REPLICATION": "3",
        "RAIDEN_GRANT_BATCH_SIZE": "2",
        "RAIDEN_LEASE_TIMEOUT_S": "1.5",
        "RAIDEN_LONG_POLL_TIMEOUT_S": "0.25",
        "RAIDEN_TRANSFER_TIMEOUT_S": "42.5",
        "RAIDEN_ISOLATION_GROUP_SIZE": "4",  # Removed; must be ignored.
        "RAIDEN_SOURCE_WAIT_TIMEOUT_S": "1.5",  # Removed; must be ignored.
        "UNRELATED": "ignored",
    }
    cfg = types.WeightSyncConfig.from_env(env=env)
    self.assertEqual(cfg.broadcast_host_ratio, 2.5)
    self.assertEqual(cfg.num_bundle_groups, 16)
    self.assertEqual(cfg.max_concurrent_uploads_per_source, 3)
    self.assertEqual(cfg.num_stripes, 4)
    self.assertEqual(cfg.seed_replication, 3)
    self.assertEqual(cfg.grant_batch_size, 2)
    self.assertEqual(cfg.lease_timeout_s, 1.5)
    self.assertEqual(cfg.long_poll_timeout_s, 0.25)
    self.assertEqual(cfg.transfer_timeout_s, 42.5)
    # Fields with no env var keep defaults.
    self.assertEqual(cfg.port, 0)
    self.assertTrue(cfg.enable_plan_cache)

  def test_from_env_explicit_overrides_win_over_env(self):
    env = {
        "RAIDEN_NUM_BUNDLE_GROUPS": "16",
        "RAIDEN_BROADCAST_HOST_RATIO": "2.0",
        "RAIDEN_SEED_REPLICATION": "3",
        "RAIDEN_TRANSFER_TIMEOUT_S": "30",
    }
    cfg = types.WeightSyncConfig.from_env(
        env=env,
        num_bundle_groups=2,
        port=1234,
        seed_replication=1,
        transfer_timeout_s=5.0,
    )
    self.assertEqual(cfg.num_bundle_groups, 2)
    self.assertEqual(cfg.broadcast_host_ratio, 2.0)
    self.assertEqual(cfg.port, 1234)
    self.assertEqual(cfg.seed_replication, 1)
    self.assertEqual(cfg.transfer_timeout_s, 5.0)

  def test_environment_is_only_read_by_from_env(self):
    with mock.patch.dict("os.environ", {"RAIDEN_TRANSFER_TIMEOUT_S": "7"}):
      self.assertEqual(types.WeightSyncConfig().transfer_timeout_s, 600.0)
      self.assertEqual(
          types.WeightSyncConfig.from_env().transfer_timeout_s, 7.0
      )

  def test_from_env_invalid_env_value_raises(self):
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_NUM_BUNDLE_GROUPS": "zero"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_NUM_BUNDLE_GROUPS": "0"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_SEED_REPLICATION": "0"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_NUM_STRIPES": "-1"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_LEASE_TIMEOUT_S": "0"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_LONG_POLL_TIMEOUT_S": "0"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_GRANT_BATCH_SIZE": "9"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_TRANSFER_TIMEOUT_S": "0"})
    with self.assertRaises(ValueError):
      types.WeightSyncConfig.from_env(env={"RAIDEN_TRANSFER_TIMEOUT_S": "soon"})


class TransferOptionsTest(absltest.TestCase):

  def test_defaults(self):
    opts = types.TransferOptions()
    self.assertEqual(opts.dst_mem_type, types.RaidenMemoryType.DRAM)
    self.assertFalse(opts.skip_d2h)
    self.assertEqual(opts.skip_tiling, {})
    self.assertEqual(opts.parallelism, 1)
    self.assertTrue(opts.use_cached_plan)
    self.assertIsNone(opts.timeout_s)

  def test_frozen(self):
    opts = types.TransferOptions()
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(opts, "parallelism", 2)

  def test_validation(self):
    with self.assertRaises(ValueError):
      types.TransferOptions(parallelism=0)
    for bad_timeout in (0.0, -1.0, 0.0004, float("nan")):
      with self.assertRaisesRegex(
          ValueError, "timeout_s", msg=str(bad_timeout)
      ):
        types.TransferOptions(timeout_s=bad_timeout)
    self.assertEqual(types.TransferOptions(timeout_s=0.001).timeout_s, 0.001)
    # Per-transfer isolation overrides no longer exist.
    with self.assertRaises(TypeError):
      types.TransferOptions(**{"isolation_group_size": 1})
    with self.assertRaises(TypeError):
      types.TransferOptions(**{"isolation_group_sizes": (1, 1)})
    self.assertFalse(hasattr(types.TransferOptions, "has_isolation_override"))

  def test_normalization(self):
    opts = types.TransferOptions(skip_tiling={"3": 1, 4: False})
    self.assertEqual(opts.skip_tiling, {3: True, 4: False})


class BindingTransferTimeoutTest(absltest.TestCase):

  def test_transfer_timeout_property_and_per_transfer_override(self):
    self.assertEqual(
        _controller_v3.RaidenControllerV3().transfer_timeout_ms, 600000
    )
    ctrl = _controller_v3.RaidenControllerV3(transfer_timeout_ms=1234)
    self.assertEqual(ctrl.transfer_timeout_ms, 1234)
    ctrl.transfer_timeout_ms = 0  # Clamped to 1 ms.
    self.assertEqual(ctrl.transfer_timeout_ms, 1)
    # The per-transfer override is validated before the plan is looked up.
    with self.assertRaisesRegex(ValueError, "must be positive"):
      ctrl.execute_materialized_transfer_sync_bytes(
          "missing", transfer_timeout_ms=0
      )
    with self.assertRaisesRegex(RuntimeError, "No materialized plan"):
      ctrl.execute_materialized_transfer_sync_bytes(
          "missing", transfer_timeout_ms=1000
      )
    # `wait_for_transfer` takes an explicit timeout or follows the deadline.
    with self.assertRaisesRegex(RuntimeError, "not found"):
      ctrl.wait_for_transfer("missing")
    with self.assertRaisesRegex(RuntimeError, "not found"):
      ctrl.wait_for_transfer("missing", 0.1)


class NDSliceAndBundleTest(absltest.TestCase):

  def test_nd_slice(self):
    s = types.NDSlice.from_bounds([(0, 4), (2, 8)])
    self.assertEqual(s.shape, (4, 6))
    self.assertLen(s, 2)
    self.assertEqual(s[1], (2, 8))
    self.assertEqual(list(s), [(0, 4), (2, 8)])
    self.assertEqual(types.NDSlice.from_bounds([(5, 3)]).shape, (0,))

  def test_bounds_to_cpp_nd_slice_roundtrip(self):
    cpp = types.bounds_to_cpp_nd_slice([(1, 4), (0, 2)])
    self.assertEqual(list(cpp.offsets), [1, 0])
    self.assertEqual(list(cpp.sizes), [3, 2])
    self.assertIs(types.bounds_to_cpp_nd_slice(cpp), cpp)
    cpp2 = types.bounds_to_cpp_nd_slice(types.NDSlice.from_bounds([(2, 3)]))
    self.assertEqual(list(cpp2.offsets), [2])
    self.assertEqual(list(cpp2.sizes), [1])

  def test_variable_bundle_spec_roundtrip(self):
    spec = types.VariableBundleSpec(
        bundle_id=3,
        layer_indices=(0, 1),
        layer_byte_sizes=(10, 20),
        total_bytes=30,
    )
    cpp = spec.to_cpp()
    back = types.VariableBundleSpec.from_cpp(cpp)
    self.assertEqual(back, spec)
    self.assertIs(types.VariableBundleSpec.from_cpp(spec), spec)
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(spec, "bundle_id", 1)


class HostCommandAndIdTest(absltest.TestCase):

  def test_unit_to_bytes(self):
    unit = RaidenId(
        job_name="s", job_replica_id="s0", data_name="w", data_replica_idx=2
    )
    proto = raiden_service_pb2.RaidenIdProto()
    proto.ParseFromString(types.unit_to_bytes(unit))
    self.assertEqual(types.raiden_id_from_proto(proto), unit)
    proto2 = raiden_service_pb2.RaidenIdProto()
    proto2.ParseFromString(types.unit_to_bytes("bare"))
    self.assertEqual(proto2.job_replica_id, "bare")

  def test_host_command_request_parses(self):
    req = raiden_service_pb2.StartTransferRequest(req_id="r1", uuid=7)
    cmd = types.HostCommand(
        unit=RaidenId(job_replica_id="x"),
        host_idx=1,
        control_endpoint="1.2.3.4:5",
        request_bytes=req.SerializeToString(),
    )
    parsed = cmd.request()
    self.assertEqual(parsed.req_id, "r1")
    self.assertEqual(parsed.uuid, 7)

  def test_copy_variable_proto_is_independent(self):
    src = raiden_service_pb2.VariableMetadataProto(name="v")
    out = types._copy_variable_proto(src)  # pylint: disable=protected-access
    self.assertIsNot(out, src)
    out.name = "changed"
    self.assertEqual(src.name, "v")

  def test_transfer_plan_from_cpp_striped_layout(self):
    units = [RaidenId(job_replica_id=f"s{i}") for i in range(3)]
    trainer = RaidenId(job_replica_id="t")

    def _cmd(unit, host_idx, endpoint, is_sender):
      req = raiden_service_pb2.StartTransferRequest(
          req_id="r", uuid=9, is_sender=is_sender
      )
      return _Obj(
          unit_bytes=types.unit_to_bytes(unit),
          host_idx=host_idx,
          control_endpoint=endpoint,
          request_bytes=req.SerializeToString,
      )

    description = {
        "num_stripes": 1,
        "seed_replication": 2,
        "stripe_bundles": [[0, 1]],
        "stripe_seeds": [[2, 0]],
        "trainer_waves": [[(2, 0)], [(0, 0)]],
    }
    sampler_cmds = [
        _Obj(
            replica_idx=r,
            command=_cmd(units[r], 0, f"ep{r}", False),
            seeded_bundles=[0, 1] if r != 1 else [],
            seeded_layers=[0, 1] if r != 1 else [],
        )
        for r in range(3)
    ]
    cpp_plan = _Obj(
        req_id="r",
        uuid=9,
        num_stripes=1,
        seed_replication=2,
        variable_bundles=[],
        sampler_commands=sampler_cmds,
        trainer_wave_commands=[
            [_cmd(trainer, 0, "tep", True)],
            [_cmd(trainer, 0, "tep", True)],
        ],
        describe_plan=lambda: description,
    )
    plan = types.TransferPlan.from_cpp(
        cpp_plan, [trainer], units, types.TransferOptions()
    )
    self.assertEqual((plan.num_stripes, plan.seed_replication), (1, 2))
    self.assertEqual(plan.stripe_bundles, ((0, 1),))
    self.assertEqual(plan.stripe_seeds, ((2, 0),))
    self.assertEqual(plan.trainer_waves, (((2, 0),), ((0, 0),)))
    self.assertEqual(plan.seed_units, (units[0], units[2]))  # replica order
    self.assertLen(plan.sampler_commands, 3)
    puller = plan.sampler_commands[1]
    self.assertEqual(puller.unit, units[1])
    self.assertEqual(puller.control_endpoint, "ep1")
    self.assertFalse(puller.request().is_sender)
    self.assertEmpty(puller.seeded_bundles)
    self.assertEqual(plan.sampler_commands[0].seeded_layers, (0, 1))
    self.assertLen(plan.trainer_wave_commands, 2)
    self.assertTrue(plan.trainer_wave_commands[0][0].request().is_sender)
    # describe_plan() returns an independent copy of the C++ description.
    described = plan.describe_plan()
    self.assertEqual(described, description)
    described["stripe_seeds"].clear()
    self.assertLen(plan.describe_plan()["stripe_seeds"], 1)


class RaidenFutureTest(absltest.TestCase):

  def test_result_of_completed_future(self):
    cf = concurrent.futures.Future()
    cf.set_result("ok")
    fut = types.RaidenFuture(cf, "req")
    self.assertTrue(fut.done())
    self.assertFalse(fut.running())
    self.assertEqual(fut.result(), "ok")
    self.assertEqual(fut.wait_threadsafe(timeout=0), "ok")
    self.assertIsNone(fut.exception())
    self.assertEqual(fut.req_id, "req")

  def test_exception_propagates(self):
    cf = concurrent.futures.Future()
    cf.set_exception(RuntimeError("boom"))
    fut = types.RaidenFuture(cf, "req")
    with self.assertRaisesRegex(RuntimeError, "boom"):
      fut.result()
    self.assertIsInstance(fut.exception(), RuntimeError)

  def test_timeout_raises_builtin_timeout_error_without_cancelling(self):
    release = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
      cf = ex.submit(release.wait)
      fut = types.RaidenFuture(cf, "slow")
      with self.assertRaisesRegex(TimeoutError, "still running"):
        fut.wait_threadsafe(timeout=0.05)
      self.assertFalse(fut.done())
      self.assertFalse(fut.cancelled())
      # cancel() must fail once running.
      self.assertFalse(fut.cancel())
      release.set()
      self.assertTrue(fut.result(timeout=5))

  def test_cancel_before_start(self):
    cf = concurrent.futures.Future()
    fut = types.RaidenFuture(cf, "pending")
    self.assertTrue(fut.cancel())
    self.assertTrue(fut.cancelled())
    self.assertTrue(fut.done())

  def test_add_done_callback_receives_wrapper(self):
    cf = concurrent.futures.Future()
    fut = types.RaidenFuture(cf, "cb")
    seen = []
    fut.add_done_callback(seen.append)
    cf.set_result(1)
    self.assertEqual(seen, [fut])

  def test_asyncio_wait(self):
    import asyncio  # pylint: disable=g-import-not-at-top

    cf = concurrent.futures.Future()
    fut = types.RaidenFuture(cf, "aio")

    async def _go():
      loop = asyncio.get_running_loop()
      loop.call_later(0.01, cf.set_result, 42)
      return await fut.wait()

    self.assertEqual(asyncio.run(_go()), 42)


class SynchronizerHelpersTest(absltest.TestCase):

  def test_extract_endpoints_from_shard_map_with_bind_ip(self):
    class _Sync:
      num_shards = 2
      listener_port = 9000
      local_port = 8000

      def get_local_endpoints(self):
        return [
            {"endpoint": "0.0.0.0:8001", "shards": [1]},
            {"endpoint": "0.0.0.0:8000", "shards": [0]},
        ]

    shards, ctrl = types.extract_synchronizer_endpoints(
        _Sync(), bind_ip="10.0.0.1"
    )
    self.assertEqual(shards, ["10.0.0.1:8000", "10.0.0.1:8001"])
    self.assertEqual(ctrl, "10.0.0.1:9000")

  def test_explicit_shards_and_control_address_win(self):
    class _Sync:
      num_shards = 1

      def get_local_endpoints(self):
        raise AssertionError("must not be called when shards are explicit")

    shards, ctrl = types.extract_synchronizer_endpoints(
        _Sync(), shards=["1.1.1.1:1"], control_plane_rpc_address="1.1.1.1:2"
    )
    self.assertEqual(shards, ["1.1.1.1:1"])
    self.assertEqual(ctrl, "1.1.1.1:2")

  def test_synchronizer_like_is_runtime_protocol_free(self):
    # SynchronizerLike is a typing.Protocol used for static typing only; the
    # extraction helpers must accept duck-typed objects without inheritance.
    class _Duck:
      num_shards = 1
      listener_port = 0

      def get_local_endpoints(self):
        return [{"endpoint": "9.9.9.9:7", "shards": [0]}]

    shards, ctrl = types.extract_synchronizer_endpoints(_Duck())
    self.assertEqual(shards, ["9.9.9.9:7"])
    self.assertIsNone(ctrl)

  def test_routable_local_ip_falls_back_to_getaddrinfo(self):
    def _addr(ip):
      return (socket.AF_INET, socket.SOCK_DGRAM, 0, "", (ip, 0))

    with mock.patch.object(
        socket, "socket", side_effect=OSError("no route")
    ), mock.patch.object(
        socket,
        "getaddrinfo",
        return_value=[_addr("127.0.1.1"), _addr("10.1.2.3")],
    ) as gai:
      self.assertEqual(types.routable_local_ip(), "10.1.2.3")
    self.assertEqual(gai.call_args.args[2], socket.AF_INET)
    with mock.patch.object(
        socket, "socket", side_effect=OSError("no route")
    ), mock.patch.object(
        socket, "getaddrinfo", side_effect=socket.gaierror("unresolvable")
    ):
      self.assertEqual(types.routable_local_ip(), "127.0.0.1")


class _CpuSynchronizer:
  """Synchronizer stand-in with one 64-byte layer per local shard."""

  num_layers = 1
  slice_byte_size = 64
  itemsize = 2

  def __init__(self, num_shards, **attrs):
    self.num_shards = num_shards
    self.__dict__.update(attrs)


class GlobalShardIndicesTest(absltest.TestCase):

  def test_build_request_requires_one_global_index_per_shard(self):
    unit = RaidenId("trainer", "0", "weights")
    var = raiden_service_pb2.VariableMetadataProto(
        name="w0", shape=[8], mesh_shape=[2], layout=[0], item_size=2
    )
    with self.assertRaisesRegex(
        ValueError, r"'w0' has 0 global_shard_indices for 2 shards.*required"
    ):
      types.build_register_work_unit_request(
          unit, ["a:1", "a:2"], variables=[var]
      )
    var.global_shard_indices.extend([1, 0])
    req = types.build_register_work_unit_request(
        unit, ["a:1", "a:2"], variables=[var]
    )
    self.assertEqual(list(req.variables[0].global_shard_indices), [1, 0])
    self.assertEmpty(req.mesh_shape)
    self.assertEmpty(req.global_shape)

  def test_synthesis_requires_the_synchronizer_placement(self):
    with self.assertRaisesRegex(ValueError, "global_shard_indices"):
      types.extract_synchronizer_variables(_CpuSynchronizer(2), ["a:1", "a:2"])
    with self.assertRaisesRegex(ValueError, "3 entries for 2 local shards"):
      types.extract_synchronizer_variables(
          _CpuSynchronizer(2), ["a:1", "a:2"], global_shard_indices=[0, 1, 2]
      )

  def test_synthesis_uses_the_given_or_public_global_indices(self):
    variables, auto_1d = types.extract_synchronizer_variables(
        _CpuSynchronizer(2), ["a:1", "a:2"], global_shard_indices=[2, 3]
    )
    self.assertTrue(auto_1d)
    self.assertEqual(list(variables[0].global_shard_indices), [2, 3])
    self.assertEqual(list(variables[0].mesh_shape), [4])
    self.assertEqual(list(variables[0].shape), [4 * 32])

    variables, _ = types.extract_synchronizer_variables(
        _CpuSynchronizer(2, global_shard_indices=[1, 0]), ["a:1", "a:2"]
    )
    self.assertEqual(list(variables[0].global_shard_indices), [1, 0])
    self.assertEqual(list(variables[0].mesh_shape), [2])

  def test_merge_requires_global_indices_from_every_host(self):
    with_gsi = raiden_service_pb2.VariableMetadataProto(
        name="w0", shape=[32], mesh_shape=[1], global_shard_indices=[0]
    )
    without_gsi = raiden_service_pb2.VariableMetadataProto(
        name="w0", shape=[32], mesh_shape=[1]
    )
    with self.assertRaisesRegex(ValueError, "Host 1: Variable 'w0' has 0"):
      types.merge_host_synchronizer_entries({
          0: (["a:1"], "a:9", [with_gsi], True),
          1: (["b:1"], "b:9", [without_gsi], True),
      })
    host1 = raiden_service_pb2.VariableMetadataProto()
    host1.CopyFrom(with_gsi)
    host1.global_shard_indices[:] = [1]
    shards, ctrl, variables = types.merge_host_synchronizer_entries({
        0: (["a:1"], "a:9", [with_gsi], True),
        1: (["b:1"], "b:9", [host1], True),
    })
    self.assertEqual(shards, ["a:1", "b:1"])
    self.assertEqual(ctrl, "a:9,b:9")
    self.assertEqual(list(variables[0].global_shard_indices), [0, 1])
    self.assertEqual(list(variables[0].mesh_shape), [2])
    self.assertEqual(list(variables[0].shape), [64])


if __name__ == "__main__":
  absltest.main()

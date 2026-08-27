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

"""Tests for WeightSynchronizerManager public API under api/."""

import time
from absl.testing import absltest
from tpu_sync.api import weight_synchronizer_manager
from tpu_sync.rpc import raiden_controller


class DummyWorkerRpcClient(raiden_controller.WorkerRpcClient):

  async def start_transfer(self, target_id, transfer_plan, address=None) -> None:
    pass


class WeightSynchronizerManagerTest(absltest.TestCase):

  def test_instantiation_and_methods(self):
    dummy_client = DummyWorkerRpcClient()
    manager = weight_synchronizer_manager.WeightSynchronizerManager(
        port=0,
        worker_rpc_client=dummy_client,
        broadcast_k=32,
        enable_plan_cache=True,
    )

    unit = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="host0",
        data_name="qwen_weights",
        data_replica_idx=0,
    )
    endpoints = ["10.0.0.1:8000", "10.0.0.1:8001"]
    manager.register_work_unit(unit, endpoints)
    self.assertEqual(manager.get_plan_cache_size(), 0)
    manager.clear_plan_cache()
    self.assertEqual(len(manager.get_all_metadata()), 1)
    self.assertIsNone(manager.get_plan("nonexistent_req"))
    manager.close()

  def test_server_lifecycle_and_context_manager(self):
    with weight_synchronizer_manager.WeightSynchronizerManager(
        port=0,
        worker_rpc_client=DummyWorkerRpcClient(),
        auto_start_server=True,
    ) as mgr:
      self.assertGreater(mgr.port, 0)
      client = raiden_controller.RaidenControllerClientFacade(
          f"127.0.0.1:{mgr.port}"
      )
      unit = raiden_controller.RaidenId(
          job_name="inference",
          job_replica_id="sampler0",
          data_name="qwen_weights",
          data_replica_idx=0,
      )
      client.register_work_unit(
          unit=unit,
          shards=["127.0.0.1:9000"],
      )
      time.sleep(0.2)
      metadata = mgr.get_all_metadata()
      self.assertEqual(len(metadata), 1)

  def test_public_api_exports_and_method_surface(self):
    self.assertEqual(
        weight_synchronizer_manager.__all__, ["WeightSynchronizerManager"]
    )
    expected_controller_methods = {
        "register_work_unit",
        "clear_plan_cache",
        "get_plan_cache_size",
        "get_all_metadata",
        "get_plan",
        "start_transfer",
        "get_transfer_status",
    }
    manager_methods = {
        m
        for m in dir(weight_synchronizer_manager.WeightSynchronizerManager)
        if not m.startswith("_")
    }
    for method in expected_controller_methods:
      self.assertIn(
          method,
          manager_methods,
          f"Missing expected method {method} on WeightSynchronizerManager",
      )

    # Ensure internal controller instance is not exposed to users
    self.assertNotIn("controller", manager_methods)


if __name__ == "__main__":
  absltest.main()

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

"""Tests for ReshardStore."""

from absl.testing import absltest
from tpu_sync.api.torch import reshard_store


class ReshardStoreTest(absltest.TestCase):

  def test_init_and_properties(self):
    raiden_id = reshard_store.RaidenId("test_job", "0", "test_cache", 0)
    store = reshard_store.ReshardStore(
        raiden_id=raiden_id,
        store_server_ip="127.0.0.1",
        raiden_controller_port=0,
        reshard_service_port=0,
    )
    self.assertEqual(store.raiden_id, raiden_id)
    self.assertEqual(store.raiden_id.job_name, "test_job")
    self.assertEqual(store.raiden_id.job_replica_id, "0")
    self.assertEqual(store.raiden_id.data_name, "test_cache")
    self.assertEqual(store.raiden_id.data_replica_idx, 0)
    self.assertIn("127.0.0.1:", store.raiden_controller_address)
    self.assertGreater(store.reshard_service_port, 0)

  def test_invalid_raiden_id_type_raises_type_error(self):
    with self.assertRaises(TypeError):
      reshard_store.ReshardStore(
          raiden_id="invalid_raiden_id",
          store_server_ip="127.0.0.1",
          raiden_controller_port=0,
          reshard_service_port=0,
      )

  def test_multiple_instances_dynamic_ports(self):
    raiden_id_1 = reshard_store.RaidenId("test_job_1", "0", "test_cache_1", 0)
    store_1 = reshard_store.ReshardStore(
        raiden_id=raiden_id_1,
        store_server_ip="127.0.0.1",
        raiden_controller_port=0,
        reshard_service_port=0,
    )
    raiden_id_2 = reshard_store.RaidenId("test_job_2", "0", "test_cache_2", 0)
    store_2 = reshard_store.ReshardStore(
        raiden_id=raiden_id_2,
        store_server_ip="127.0.0.1",
        raiden_controller_port=0,
        reshard_service_port=0,
    )
    self.assertGreater(store_1.reshard_service_port, 0)
    self.assertGreater(store_2.reshard_service_port, 0)
    self.assertNotEqual(
        store_1.reshard_service_port, store_2.reshard_service_port
    )

  def test_invalid_ip_raises_runtime_error(self):
    raiden_id = reshard_store.RaidenId("test_job", "0", "test_cache", 0)
    with self.assertRaises(RuntimeError):
      reshard_store.ReshardStore(
          raiden_id=raiden_id,
          store_server_ip="",
          raiden_controller_port=0,
          reshard_service_port=0,
      )


if __name__ == "__main__":
  absltest.main()

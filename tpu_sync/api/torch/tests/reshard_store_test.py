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

import time
import types

from absl.testing import absltest

from tpu_sync.api.torch import reshard_client
from tpu_sync.api.torch import reshard_store


def _register_one_block(store, req_id="req", uuid=7):
  """Registers one source unit and one single-block request with `store`."""
  client = reshard_client.ReshardClient(
      f"127.0.0.1:{store.reshard_service_port}"
  )
  unit = reshard_store.RaidenId("prefill", "0", "kv.fa", 0)
  pool = {
      "tag": "fa",
      "block_stride_bytes": 1024,
      "num_blocks": 16,
      "regions": [{
          "name": "kv",
          "stride_bytes": 1024,
          "unit_bytes": 1024,
          "num_units": 1,
      }],
  }
  client.register_work_unit(
      unit,
      shards=["127.0.0.1:9000"],
      control_plane_rpc_address="127.0.0.1:9100",
      pool_manifest=[pool],
      layout_fingerprint="fp1",
      page_tokens=512,
      transfer_parallelism=1,
      transfer_rank=0,
  )
  span = types.SimpleNamespace(
      src_block_ordinal=0,
      src_offset_bytes=0,
      dst_block_index=0,
      dst_offset_bytes=0,
      size_bytes=1024,
      src_stride_bytes=0,
      dst_stride_bytes=0,
      count=1,
  )
  entry = types.SimpleNamespace(
      tag="fa",
      block_ids=[3],
      declared_bytes=1024,
      dst_space_version=0,
      spans=[span],
  )
  client.register_request_blocks(req_id, uuid, unit, [3], pool_spans=[entry])
  return client


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

  def test_request_registry_ttl_defaults_and_property(self):
    raiden_id = reshard_store.RaidenId("test_job", "0", "test_cache", 0)
    default_store = reshard_store.ReshardStore(
        raiden_id=raiden_id, store_server_ip="127.0.0.1"
    )
    self.assertEqual(default_store.request_registry_ttl_s, 600.0)
    custom_store = reshard_store.ReshardStore(
        raiden_id=reshard_store.RaidenId("test_job", "1", "test_cache", 0),
        store_server_ip="127.0.0.1",
        request_registry_ttl_s=42.5,
    )
    self.assertEqual(custom_store.request_registry_ttl_s, 42.5)
    self.assertTrue(reshard_store.SUPPORTS_REQUEST_REGISTRY_TTL)

  def test_request_registry_ttl_rejects_non_positive(self):
    raiden_id = reshard_store.RaidenId("test_job", "0", "test_cache", 0)
    for ttl in (0, -1.0):
      with self.assertRaises(ValueError):
        reshard_store.ReshardStore(
            raiden_id=raiden_id,
            store_server_ip="127.0.0.1",
            request_registry_ttl_s=ttl,
        )

  def test_registration_expires_after_configured_ttl(self):
    short = reshard_store.ReshardStore(
        raiden_id=reshard_store.RaidenId("ttl_job", "short", "kv", 0),
        store_server_ip="127.0.0.1",
        request_registry_ttl_s=0.5,
    )
    long = reshard_store.ReshardStore(
        raiden_id=reshard_store.RaidenId("ttl_job", "long", "kv", 0),
        store_server_ip="127.0.0.1",
        request_registry_ttl_s=600.0,
    )
    short_client = _register_one_block(short)
    long_client = _register_one_block(long)
    key = [("req", 7)]
    self.assertEqual(
        short_client.get_request_block_status(key),
        [reshard_client.REQUEST_BLOCK_STATUS_REGISTERED],
    )
    self.assertEqual(
        long_client.get_request_block_status(key),
        [reshard_client.REQUEST_BLOCK_STATUS_REGISTERED],
    )
    time.sleep(0.8)
    # The short-TTL registry purged the row; the long-TTL one kept it.
    self.assertEqual(
        short_client.get_request_block_status(key),
        [reshard_client.REQUEST_BLOCK_STATUS_UNKNOWN],
    )
    self.assertEqual(
        long_client.get_request_block_status(key),
        [reshard_client.REQUEST_BLOCK_STATUS_REGISTERED],
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

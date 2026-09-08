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

"""Tests that ReshardStore accepts the native RaidenId binding."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from tpu_sync.api.torch import kv_cache_store
from tpu_sync.api.torch import reshard_store


class ReshardStoreTest(absltest.TestCase):

  def test_accepts_native_raiden_id(self):
    # kv_cache_store.RaidenId is the nanobind class itself, not a Python
    # wrapper, so it carries no `_impl` attribute to unwrap.
    native_id = kv_cache_store.RaidenId(
        job_name="test",
        job_replica_id="engine",
        data_name="reshard_store",
        data_replica_idx=0,
    )
    self.assertFalse(hasattr(native_id, "_impl"))

    with mock.patch.object(
        reshard_store._impl, "create_reshard_store"  # pylint: disable=protected-access
    ) as factory:
      store = reshard_store.ReshardStore(native_id, "127.0.0.1", 1234, 1235)

    factory.assert_called_once_with(native_id, "127.0.0.1", 1234, 1235)
    self.assertIs(store._impl, factory.return_value)  # pylint: disable=protected-access

  def test_unwraps_python_wrapper_with_impl(self):
    native_id = kv_cache_store.RaidenId(job_name="test", data_name="store")
    wrapper = SimpleNamespace(_impl=native_id)

    with mock.patch.object(
        reshard_store._impl, "create_reshard_store"  # pylint: disable=protected-access
    ) as factory:
      reshard_store.ReshardStore(wrapper, "127.0.0.1")

    factory.assert_called_once_with(native_id, "127.0.0.1", 0, 0)

  def test_constructs_real_store_on_localhost(self):
    store = reshard_store.ReshardStore(
        kv_cache_store.RaidenId(job_name="runtime-test", data_name="store"),
        "127.0.0.1",
    )
    self.assertStartsWith(store.raiden_controller_address, "127.0.0.1:")
    self.assertGreater(store.reshard_service_port, 0)


if __name__ == "__main__":
  absltest.main()

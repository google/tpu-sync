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

from absl.testing import absltest
from tpu_sync.api.common import BlockStatus
from tpu_sync.api.common import RaidenId


class CommonApiTest(absltest.TestCase):

  def test_raiden_id_basic_instantiation(self):
    raiden_id = RaidenId(
        job_name="trainer",
        job_replica_id="1",
        data_name="kv_cache",
        data_replica_idx=0,
    )
    self.assertEqual(raiden_id.job_name, "trainer")
    self.assertEqual(raiden_id.job_replica_id, "1")
    self.assertEqual(raiden_id.data_name, "kv_cache")
    self.assertEqual(raiden_id.data_replica_idx, 0)

  def test_raiden_id_equality(self):
    id1 = RaidenId(
        job_name="sampler",
        job_replica_id="0",
        data_name="model.weights",
        data_replica_idx=2,
    )
    id2 = RaidenId(
        job_name="sampler",
        job_replica_id="0",
        data_name="model.weights",
        data_replica_idx=2,
    )
    id3 = RaidenId(
        job_name="sampler",
        job_replica_id="1",
        data_name="model.weights",
        data_replica_idx=2,
    )
    self.assertEqual(id1, id2)
    self.assertNotEqual(id1, id3)

  def test_raiden_id_hash(self):
    id1 = RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="weights",
        data_replica_idx=0,
    )
    id2 = RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="weights",
        data_replica_idx=0,
    )
    id_map = {id1: "value_trainer_0"}
    self.assertIn(id2, id_map)
    self.assertEqual(id_map[id2], "value_trainer_0")

    id_set = {id1}
    self.assertIn(id2, id_set)

  def test_raiden_id_empty(self):
    empty_id = RaidenId()
    self.assertTrue(empty_id.empty())
    non_empty = RaidenId(job_name="trainer")
    self.assertFalse(non_empty.empty())

  def test_raiden_id_repr_and_str(self):
    raiden_id = RaidenId(
        job_name="jobA",
        job_replica_id="replica1",
        data_name="kv",
        data_replica_idx=3,
    )
    self.assertIn("jobA", repr(raiden_id))
    self.assertIn("replica1", repr(raiden_id))
    self.assertIn("kv", repr(raiden_id))
    self.assertIn("3", repr(raiden_id))
    self.assertIn("jobA", str(raiden_id))

  def test_pure_python_raiden_id(self):
    from tpu_sync.api.common import _PurePythonRaidenId  # pylint: disable=g-import-not-at-top

    id1 = _PurePythonRaidenId("jobA", "0", "dataA", 0)
    id2 = _PurePythonRaidenId("jobA", "0", "dataA", 0)
    id3 = _PurePythonRaidenId("jobB", "0", "dataA", 0)
    self.assertEqual(id1, id2)
    self.assertNotEqual(id1, id3)
    self.assertEqual(hash(id1), hash(id2))
    self.assertFalse(id1.empty())
    self.assertTrue(_PurePythonRaidenId().empty())

  def test_block_status_enum(self):
    self.assertEqual(BlockStatus.INIT.value, 0)
    self.assertEqual(BlockStatus.REMOTE.value, 1)
    self.assertEqual(BlockStatus.HBM.value, 2)
    self.assertEqual(BlockStatus.HOST.value, 3)
    self.assertEqual(BlockStatus.HOST_AND_HBM.value, 4)
    self.assertEqual(BlockStatus.SHARED_STORAGE.value, 5)

  def test_telemetry_callbacks_registration_and_recording(self):
    from tpu_sync.api import common  # pylint: disable=g-import-not-at-top

    recorded_counters = []
    recorded_gauges = []
    recorded_histograms = []

    def mock_inc(name, val, labels):
      recorded_counters.append((name, val, labels))

    def mock_gauge(name, val, labels):
      recorded_gauges.append((name, val, labels))

    def mock_hist(name, val, labels):
      recorded_histograms.append((name, val, labels))

    common.register_telemetry_callbacks(
        increment_counter=mock_inc,
        set_gauge=mock_gauge,
        observe_histogram=mock_hist,
    )

    self.assertIsNotNone(common.get_telemetry_callbacks())

    common.record_counter("test_counter", 5, {"tag": "foo"})
    self.assertEqual(recorded_counters, [("test_counter", 5, {"tag": "foo"})])

    common.record_gauge("test_gauge", 42.5)
    self.assertEqual(recorded_gauges, [("test_gauge", 42.5, {})])

    common.record_histogram("test_hist", 123.4, {"layer": "0"})
    self.assertEqual(
        recorded_histograms, [("test_hist", 123.4, {"layer": "0"})]
    )


if __name__ == "__main__":
  absltest.main()

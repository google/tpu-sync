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

"""Tests for TPU Raiden Python telemetry bindings."""

import os
from unittest import mock
import urllib.request

from absl.testing import absltest
import portpicker

from tpu_sync.telemetry.python import _telemetry_binding_test_ext as telemetry_ext


class TelemetryBindingTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    telemetry_ext.configure_telemetry([])

  def test_configure_telemetry_enable_prometheus(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    self.assertEqual(
        telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
    )
    self.assertEqual(telemetry_ext.get_raiden_metrics_prometheus_text(), "")

  def test_configure_telemetry_case_insensitive(self):
    telemetry_ext.configure_telemetry(["Prometheus"])
    self.assertEqual(
        telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
    )

  def test_configure_telemetry_duplicate_backends_accepted(self):
    """Verifies passing duplicate backend names is safely accepted."""
    telemetry_ext.configure_telemetry(["prometheus", "prometheus"])
    self.assertEqual(
        telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
    )

  def test_configure_telemetry_empty_backends_clears_backends(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    self.assertEqual(
        telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
    )
    telemetry_ext.configure_telemetry([])
    self.assertEqual(telemetry_ext.get_metric_metadata(), [])
    self.assertEqual(telemetry_ext.get_raiden_metrics_prometheus_text(), "")

  def test_configure_telemetry_unknown_backend_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown telemetry backend: unknown_backend"
    ):
      telemetry_ext.configure_telemetry(["unknown_backend"])

  def test_configure_telemetry_tuple_sequence_supported(self):
    telemetry_ext.configure_telemetry(("prometheus",))
    self.assertEqual(
        telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
    )

  def test_configure_telemetry_set_raises_type_error(self):
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry({"prometheus"})

  def test_configure_telemetry_invalid_argument_type_raises_type_error(self):
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry(123)
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry([123])
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry(["prometheus", 123])

  def test_configure_telemetry_no_args_initializes_from_environment_unset(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      telemetry_ext.configure_telemetry()
      self.assertEqual(telemetry_ext.get_metric_metadata(), [])
      self.assertEqual(telemetry_ext.get_raiden_metrics_prometheus_text(), "")

  def test_configure_telemetry_no_args_initializes_from_environment_prometheus(
      self,
  ):
    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_TELEMETRY_BACKENDS": "prometheus"}
    ):
      telemetry_ext.configure_telemetry()
      self.assertEqual(
          telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
      )

  def test_configure_telemetry_none_initializes_from_environment_buffered(self):
    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_TELEMETRY_BACKENDS": "buffered"}
    ):
      telemetry_ext.configure_telemetry(None)
      self.assertEqual(
          telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
      )
      samples = telemetry_ext.get_and_reset_metric_samples()
      self.assertEqual(samples, {})

  def test_configure_telemetry_no_args_initializes_from_environment_multiple(
      self,
  ):
    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_TELEMETRY_BACKENDS": "prometheus,buffered"}
    ):
      telemetry_ext.configure_telemetry()
      self.assertEqual(
          telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
      )

  def test_configure_telemetry_environment_unknown_backend_raises_value_error(
      self,
  ):
    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_TELEMETRY_BACKENDS": "invalid_backend"}
    ):
      with self.assertRaisesRegex(
          ValueError,
          "Failed to initialize from environment: Unknown telemetry backend:"
          " invalid_backend",
      ):
        telemetry_ext.configure_telemetry()

  def test_configure_telemetry_environment_no_op_if_already_initialized(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_TELEMETRY_BACKENDS": "invalid_backend"}
    ):
      telemetry_ext.configure_telemetry()
      self.assertEqual(
          telemetry_ext.get_metric_metadata(), telemetry_ext.ALL_METRICS
      )

  def test_metric_type_enum(self):
    self.assertTrue(hasattr(telemetry_ext.MetricType, "COUNTER"))
    self.assertTrue(hasattr(telemetry_ext.MetricType, "GAUGE"))
    self.assertTrue(hasattr(telemetry_ext.MetricType, "HISTOGRAM"))
    self.assertNotEqual(
        telemetry_ext.MetricType.COUNTER, telemetry_ext.MetricType.GAUGE
    )
    self.assertNotEqual(
        telemetry_ext.MetricType.COUNTER, telemetry_ext.MetricType.HISTOGRAM
    )
    self.assertNotEqual(
        telemetry_ext.MetricType.GAUGE, telemetry_ext.MetricType.HISTOGRAM
    )

  def test_metric_metadata_properties_repr_equality(self):
    self.assertTrue(hasattr(telemetry_ext, "ALL_METRICS"))
    metrics = telemetry_ext.ALL_METRICS
    self.assertGreaterEqual(len(metrics), 3)

    sent_bytes_meta = metrics[0]
    self.assertEqual(sent_bytes_meta.name, "sent_bytes_total")
    self.assertIn("sent", sent_bytes_meta.description.lower())
    self.assertEqual(
        sent_bytes_meta.buckets,
        [
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            25.0,
            50.0,
            100.0,
            250.0,
            500.0,
            750.0,
            1000.0,
            2500.0,
            5000.0,
            7500.0,
            10000.0,
            25000.0,
            50000.0,
        ],
    )

    # Test __repr__
    repr_str = repr(sent_bytes_meta)
    self.assertIn("sent_bytes_total", repr_str)
    self.assertIn("MetricType.COUNTER", repr_str)

    # Test equality with same object / identical values
    self.assertEqual(sent_bytes_meta, metrics[0])
    self.assertNotEqual(sent_bytes_meta, metrics[1])

    # Test heterogeneous equality comparisons (must not raise TypeError)
    self.assertIsNotNone(sent_bytes_meta)
    self.assertNotEqual(sent_bytes_meta, "sent_bytes_total")
    self.assertNotEqual(sent_bytes_meta, 42)

  def test_get_metric_metadata_empty_when_no_backends(self):
    telemetry_ext.configure_telemetry([])
    metadata = telemetry_ext.get_metric_metadata()
    self.assertEqual(metadata, [])

  def test_get_and_reset_metric_samples_empty_when_no_backends(self):
    telemetry_ext.configure_telemetry([])
    samples = telemetry_ext.get_and_reset_metric_samples()
    self.assertEqual(samples, {})

  def test_get_and_reset_metric_samples_with_prometheus(self):
    """Verifies non-buffered Prometheus exporter safely returns empty sample dict."""
    telemetry_ext.configure_telemetry(["prometheus"])
    samples = telemetry_ext.get_and_reset_metric_samples()
    self.assertEqual(samples, {})

  def test_configure_telemetry_buffered(self):
    telemetry_ext.configure_telemetry(["buffered"])
    metadata = telemetry_ext.get_metric_metadata()
    self.assertEqual(metadata, telemetry_ext.ALL_METRICS)
    samples = telemetry_ext.get_and_reset_metric_samples()
    self.assertEqual(samples, {})

  def test_configure_telemetry_with_prometheus_port_env(self):
    port = str(portpicker.pick_unused_port())
    with absltest.mock.patch.dict(
        "os.environ", {"TPU_RAIDEN_PROMETHEUS_PORT": port}
    ):
      telemetry_ext.configure_telemetry(["prometheus"])
      with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as resp:
        self.assertEqual(resp.status, 200)


if __name__ == "__main__":
  absltest.main()

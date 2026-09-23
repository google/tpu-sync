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
from tpu_sync.fault_injection import fault_injection


class FaultInjectionBindingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    fault_injection.reset_faults()

  def tearDown(self):
    fault_injection.reset_faults()
    super().tearDown()

  def test_initial_and_reset_state(self):
    self.assertFalse(fault_injection.has_active_injections())
    self.assertFalse(fault_injection.is_hook_active("control.client.send"))
    self.assertEqual(fault_injection.get_hit_count(), 0)
    self.assertEqual(fault_injection.get_hit_count("control.client.send"), 0)
    self.assertEqual(fault_injection.get_fault_status(), {})

  def test_inject_and_reset_rules(self):
    fault_injection.inject_faults([
        {
            "hook": "control.client.send",
            "action": "fail",
            "probability": 1.0,
        },
        {
            "hook": "data.send.chunk",
            "action": "delay",
            "probability": 0.5,
            "min_delay_ms": 5,
            "max_delay_ms": 25,
        },
    ])
    self.assertTrue(fault_injection.has_active_injections())
    self.assertTrue(fault_injection.is_hook_active("control.client.send"))
    self.assertTrue(fault_injection.is_hook_active("data.send.chunk"))
    self.assertFalse(fault_injection.is_hook_active("data.recv.chunk"))

    fault_injection.reset_faults()
    self.assertFalse(fault_injection.has_active_injections())
    self.assertFalse(fault_injection.is_hook_active("control.client.send"))

  def test_empty_hook_installs_for_all_hooks(self):
    fault_injection.inject_faults([{"hook": "", "action": "fail", "probability": 1.0}])
    self.assertTrue(fault_injection.has_active_injections())
    self.assertTrue(fault_injection.is_hook_active("control.client.send"))
    self.assertTrue(fault_injection.is_hook_active("data.send.chunk"))

  def test_rejects_unknown_action_and_invalid_delay_range(self):
    with self.assertRaisesRegex(ValueError, "unknown action: bogus"):
      fault_injection.inject_faults(
          [{"hook": "control.client.send", "action": "bogus"}]
      )

    with self.assertRaisesRegex(
        ValueError, "min_delay_ms cannot be greater than max_delay_ms"
    ):
      fault_injection.inject_faults([
          {
              "hook": "data.send.chunk",
              "action": "delay",
              "min_delay_ms": 50,
              "max_delay_ms": 10,
          }
      ])


if __name__ == "__main__":
  absltest.main()

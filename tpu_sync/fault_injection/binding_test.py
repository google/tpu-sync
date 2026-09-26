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
    self.assertFalse(
        fault_injection.is_hook_active("transfer_recv_session.pull.request")
    )
    self.assertEqual(fault_injection.get_hit_count(), 0)
    self.assertEqual(
        fault_injection.get_hit_count("transfer_recv_session.pull.request"), 0
    )
    self.assertEqual(fault_injection.get_fault_status(), {})

  def test_inject_and_reset_rules(self):
    fault_injection.inject_faults([
        {
            "hook": "transfer_recv_session.pull.request",
            "action": "fail",
            "probability": 1.0,
        },
        {
            "hook": "socket_transport.push.send_payload",
            "action": "delay",
            "probability": 0.5,
            "min_delay_ms": 5,
            "max_delay_ms": 25,
        },
    ])
    self.assertTrue(fault_injection.has_active_injections())
    self.assertTrue(
        fault_injection.is_hook_active("transfer_recv_session.pull.request")
    )
    self.assertTrue(
        fault_injection.is_hook_active("socket_transport.push.send_payload")
    )
    self.assertFalse(
        fault_injection.is_hook_active("block_transport.recv.payload")
    )

    fault_injection.reset_faults()
    self.assertFalse(fault_injection.has_active_injections())
    self.assertFalse(
        fault_injection.is_hook_active("transfer_recv_session.pull.request")
    )

  def test_star_pattern_installs_for_all_hooks(self):
    fault_injection.inject_faults(
        [{"hook": "*", "action": "fail", "probability": 1.0}]
    )
    self.assertTrue(fault_injection.has_active_injections())
    self.assertTrue(
        fault_injection.is_hook_active("transfer_recv_session.pull.request")
    )
    self.assertTrue(
        fault_injection.is_hook_active("socket_transport.push.send_payload")
    )

  def test_prefix_pattern_installs_for_prefixed_hooks(self):
    fault_injection.inject_faults(
        [{"hook": "socket_transport.*", "action": "fail", "probability": 1.0}]
    )
    self.assertTrue(
        fault_injection.is_hook_active("socket_transport.push.send_payload")
    )
    self.assertFalse(
        fault_injection.is_hook_active("transfer_recv_session.pull.request")
    )

  def test_star_pattern_delay_installs_for_all_hooks(self):
    fault_injection.inject_faults([
        {"hook": "*", "action": "fail", "probability": 0.05},
        {
            "hook": "*",
            "action": "delay",
            "probability": 0.05,
            "min_delay_ms": 0,
            "max_delay_ms": 60_000,
        },
    ])
    self.assertTrue(fault_injection.has_active_injections())
    self.assertTrue(
        fault_injection.is_hook_active("block_transport.recv.payload")
    )
    self.assertTrue(
        fault_injection.is_hook_active("kv_cache_manager.pull.register_wait")
    )

  def test_rejects_unknown_action_and_invalid_delay_range(self):
    with self.assertRaisesRegex(ValueError, "unknown action: bogus"):
      fault_injection.inject_faults(
          [{"hook": "transfer_recv_session.pull.request", "action": "bogus"}]
      )

    with self.assertRaisesRegex(
        ValueError, "min_delay_ms cannot be greater than max_delay_ms"
    ):
      fault_injection.inject_faults([{
          "hook": "socket_transport.push.send_payload",
          "action": "delay",
          "min_delay_ms": 50,
          "max_delay_ms": 10,
      }])

    with self.assertRaisesRegex(
        ValueError,
        "delay is not permitted at hook 'transfer_recv_session.h2d.complete'",
    ):
      fault_injection.inject_faults([{
          "hook": "transfer_recv_session.h2d.complete",
          "action": "delay",
          "max_delay_ms": 10,
      }])

    with self.assertRaisesRegex(
        ValueError,
        "unknown hook 'nonexistent.hook'",
    ):
      fault_injection.inject_faults([{
          "hook": "nonexistent.hook",
          "action": "fail",
          "probability": 1.0,
      }])

    with self.assertRaisesRegex(
        ValueError,
        "fail is not permitted at hook 'kv_cache_manager.pull.register_wait'",
    ):
      fault_injection.inject_faults([{
          "hook": "kv_cache_manager.pull.register_wait",
          "action": "fail",
          "probability": 1.0,
      }])

    with self.assertRaisesRegex(
        ValueError,
        r"probability must be in \[0, 1\]",
    ):
      fault_injection.inject_faults([{
          "hook": "transfer_recv_session.pull.request",
          "action": "fail",
          "probability": 1.5,
      }])


if __name__ == "__main__":
  absltest.main()

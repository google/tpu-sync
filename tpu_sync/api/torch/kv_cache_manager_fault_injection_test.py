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

"""E2E fault injection tests over a live KVCacheManager transfer on TPU.

Note: blaze links one FaultInjector across all extensions, so the transfer
tests pass regardless of which extension the shim loads. The wheel links each
extension privately, so test_wrapper_resolves_to_torch_extension is what pins
the shim to _tpu_raiden_torch.
"""

import time
from typing import Sequence

from absl.testing import absltest
import torch

from tpu_sync.api.torch import fault_injection
from tpu_sync.api.torch import kv_cache_manager

KVCacheManager = kv_cache_manager.KVCacheManager

# Raised on the consumer's pull path before the control request goes out, so a
# failure here settles the request without depending on producer-side timing.
_PULL_REQUEST_HOOK = "transfer_recv_session.pull.request"


class KVCacheManagerFaultInjectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    self.num_layers = 2
    fault_injection.reset_faults()

  def tearDown(self):
    fault_injection.reset_faults()
    super().tearDown()

  def _setup_test_pair(
      self,
      num_blocks: int,
      seed: int = 123,
  ) -> tuple[
      kv_cache_manager.KVCacheManager,
      kv_cache_manager.KVCacheManager,
      list[torch.Tensor],
      list[torch.Tensor],
  ]:
    """Creates a producer/consumer pair over randomized source caches."""
    shape = (num_blocks, 128, 8, 8, 128)
    src_caches, src_refs = [], []
    for i in range(self.num_layers):
      torch.manual_seed(seed + i)
      host_ref = torch.randn(shape, dtype=torch.float32)
      src_caches.append(host_ref.to(self.device))
      src_refs.append(host_ref.cpu())

    dst_caches = [
        torch.zeros(shape, dtype=torch.float32, device=self.device)
        for _ in range(self.num_layers)
    ]

    producer = KVCacheManager(
        kv_caches=src_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=True,
    )
    consumer = KVCacheManager(
        kv_caches=dst_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=True,
    )
    self.assertGreater(producer.local_control_port, 0)
    return producer, consumer, src_refs, dst_caches

  def _to_loopback_endpoints(self, endpoints):
    res = []
    for ep in endpoints:
      d = dict(ep)
      port = d["endpoint"].split(":")[-1]
      d["endpoint"] = f"127.0.0.1:{port}"
      res.append(d)
    return res

  def _start_read(
      self,
      producer: kv_cache_manager.KVCacheManager,
      consumer: kv_cache_manager.KVCacheManager,
      req_id: str,
      uuid: int,
      block_ids: Sequence[int],
  ):
    """Registers and issues a read of `block_ids` from producer to consumer."""
    producer.register_read(req_id, uuid, list(block_ids))
    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertNotEmpty(endpoints)
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=list(block_ids),
        local_block_ids=list(block_ids),
    )

  def _await_recv_outcome(
      self,
      consumer: kv_cache_manager.KVCacheManager,
      req_id: str,
      timeout_s: float = 15.0,
      sleep_sec: float = 0.05,
  ) -> str:
    """Polls until `req_id` settles; returns "done" or "failed"."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        return "failed"
      if req_id in done_recving:
        return "done"
      time.sleep(sleep_sec)
    self.fail(f"Receiver did not settle {req_id} within {timeout_s}s")

  def test_wrapper_resolves_to_torch_extension(self):
    """The wrapper must resolve to the extension that carries the hooks.

    This is the structural guard for the RTLD_LOCAL singleton split. Unlike
    the behavioural tests below, it fails whenever the wrapper is pointed at
    any module other than the `fault_injection` submodule of
    `_tpu_raiden_torch` -- including the standalone extension, which is
    exactly the wiring that shipped broken.
    """
    # pylint: disable=g-import-not-at-top
    from tpu_sync.api.torch import torch_abi
    # pylint: enable=g-import-not-at-top
    extension = torch_abi.load_extension(
        "tpu_sync.frameworks.torch",
        "_tpu_raiden_torch",
    )
    self.assertTrue(
        hasattr(extension, "fault_injection"),
        "_tpu_raiden_torch does not expose a fault_injection submodule; the "
        "control surface is not linked into the extension that carries the "
        "hooks",
    )
    # pylint: disable=protected-access
    self.assertIs(fault_injection._impl(), extension.fault_injection)
    # pylint: enable=protected-access

  def test_injected_fault_fails_a_real_transfer(self):
    """A rule installed through the wrapper must fail a live transfer."""
    producer, consumer, _, dst_caches = self._setup_test_pair(
        num_blocks=2, seed=100
    )

    fault_injection.inject_faults([{
        "hook": _PULL_REQUEST_HOOK,
        "action": "fail",
        "probability": 1.0,
    }])
    self.assertTrue(fault_injection.has_active_injections())
    self.assertTrue(fault_injection.is_hook_active(_PULL_REQUEST_HOOK))

    self._start_read(producer, consumer, "req_injected_fail", 4242, [0, 1])
    outcome = self._await_recv_outcome(consumer, "req_injected_fail")

    self.assertEqual(
        outcome,
        "failed",
        "transfer succeeded despite a probability=1.0 fail rule at "
        f"{_PULL_REQUEST_HOOK}; the control surface and the transfer hooks "
        "are not sharing a FaultInjector",
    )
    # Read the counter back through the wrapper: this is what proves the
    # Python-visible injector is the one the C++ transfer code incremented.
    self.assertGreater(fault_injection.get_hit_count(_PULL_REQUEST_HOOK), 0)
    self.assertIn(_PULL_REQUEST_HOOK, fault_injection.get_fault_status())

    # The consumer's caches must be untouched by the aborted transfer.
    for t in dst_caches:
      self.assertTrue(torch.equal(t.cpu(), torch.zeros_like(t.cpu())))

  def test_reset_restores_a_healthy_transfer(self):
    """reset_faults must clear rules for the transfer path, not just Python."""
    producer, consumer, src_refs, dst_caches = self._setup_test_pair(
        num_blocks=2, seed=200
    )

    fault_injection.inject_faults([{
        "hook": _PULL_REQUEST_HOOK,
        "action": "fail",
        "probability": 1.0,
    }])
    fault_injection.reset_faults()
    self.assertFalse(fault_injection.has_active_injections())

    self._start_read(producer, consumer, "req_after_reset", 4343, [0, 1])
    outcome = self._await_recv_outcome(consumer, "req_after_reset")

    self.assertEqual(
        outcome,
        "done",
        f"transfer failed after reset_faults(); a stale rule at "
        f"{_PULL_REQUEST_HOOK} is still live in the transfer path",
    )
    for idx, t in enumerate(dst_caches):
      self.assertTrue(torch.equal(t.cpu(), src_refs[idx]))
    self.assertEqual(fault_injection.get_hit_count(), 0)


if __name__ == "__main__":
  absltest.main()

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

"""E2E physical unit tests for KVCacheManager transfer on XLA TPUs."""

import time
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import torch

from tpu_sync.api.torch import kv_cache_manager

KVCacheManager = kv_cache_manager.KVCacheManager

class KVCacheManagerTransferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 2
    self.skip_lock = True

  def _generate_random_cache(
      self,
      shape: Sequence[int],
      dtype: torch.dtype = torch.float32,
      seed: int = 123,
  ):
    torch.manual_seed(seed)
    if dtype == torch.bfloat16:
      host_data = torch.randn(shape, dtype=torch.bfloat16)
    else:
      host_data = torch.randn(shape, dtype=torch.float32)
    dev_arr = host_data.to(self.device)
    return dev_arr, host_data.cpu()

  def _wait_for_transfer(
      self,
      manager: kv_cache_manager.KVCacheManager,
      req_id: str,
      is_receiver: bool = True,
      timeout_s: float = 10.0,
      sleep_sec: float = 0.05,
  ):
    """Polls KVCacheManager until transfer finishes or times out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      done_sending, done_recving, failed_recving = manager.poll_stats()
      if is_receiver:
        if req_id in failed_recving:
          self.fail(f"Receiver transfer failed for request {req_id}")
        if req_id in done_recving:
          return
      else:
        if req_id in done_sending:
          return
      time.sleep(sleep_sec)
    role = "Receiver" if is_receiver else "Producer"
    self.fail(
        f"{role} did not finish transfer within {timeout_s}s for request"
        f" {req_id}"
    )

  def _to_loopback_endpoints(self, endpoints):
    res = []
    for ep in endpoints:
      d = dict(ep)
      endpoint_str = d["endpoint"]
      port = endpoint_str.split(":")[-1]
      d["endpoint"] = f"127.0.0.1:{port}"
      res.append(d)
    return res

  def _setup_test_pair(
      self,
      num_blocks: int,
      dtype: torch.dtype,
      seed: int = 123,
  ) -> tuple[
      kv_cache_manager.KVCacheManager,
      kv_cache_manager.KVCacheManager,
      list[torch.Tensor],
      list[torch.Tensor],
  ]:
    """Creates a Producer-Consumer pair with randomized source and zeroed destination caches."""
    shape = (num_blocks, 128, 8, 8, 128)
    src_caches, src_refs = [], []
    for i in range(self.num_layers):
      dev_arr, host_ref = self._generate_random_cache(
          shape, dtype=dtype, seed=seed + i
      )
      src_caches.append(dev_arr)
      src_refs.append(host_ref)

    dst_caches = [
        torch.zeros(shape, dtype=dtype, device=self.device)
        for _ in range(self.num_layers)
    ]

    producer = KVCacheManager(
        kv_caches=src_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    consumer = KVCacheManager(
        kv_caches=dst_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    self.assertGreater(producer.local_control_port, 0)
    return producer, consumer, src_refs, dst_caches

  def test_initialization(self):
    shape = (4, 128, 8, 8, 128)
    kv_caches = [torch.zeros(shape, device=self.device)]

    manager = KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=4,
        num_slots=2,
    )
    self.assertIsNotNone(manager)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_e2e_transfer_polling(self, dtype):
    producer, consumer, src_refs, dst_caches = self._setup_test_pair(
        num_blocks=2, dtype=dtype, seed=100
    )

    req_id = "test_req_poll"
    uuid = 12345
    producer.register_read(req_id, uuid, [0, 1])

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertNotEmpty(endpoints)
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
    )

    self._wait_for_transfer(consumer, req_id, is_receiver=True)

    # Check that consumer correctly loaded all layer values
    for idx, t in enumerate(dst_caches):
      self.assertTrue(torch.equal(t.cpu(), src_refs[idx]))

    # Poll producer until it's done sending
    self._wait_for_transfer(producer, req_id, is_receiver=False)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_non_contiguous_blocks(self, dtype):
    producer, consumer, src_refs, dst_caches = self._setup_test_pair(
        num_blocks=3, dtype=dtype, seed=200
    )

    req_id = "test_req_non_contig"
    uuid = 54321
    producer.register_read(req_id, uuid, [0, 2])

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertNotEmpty(endpoints)
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[0, 2],
        local_block_ids=[0, 1],
    )

    self._wait_for_transfer(consumer, req_id, is_receiver=True)

    for idx, t in enumerate(dst_caches):
      # local block 0 <- remote block 0
      self.assertTrue(torch.equal(t[0].cpu(), src_refs[idx][0]))
      # local block 1 <- remote block 2
      self.assertTrue(torch.equal(t[1].cpu(), src_refs[idx][2]))
      # local block 2 was not copied, should remain 0
      self.assertTrue(torch.equal(t[2].cpu(), torch.zeros_like(t[2].cpu())))

    # Poll producer until it's done sending
    self._wait_for_transfer(producer, req_id, is_receiver=False)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_host_reordering(self, dtype):
    producer, consumer, src_refs, dst_caches = self._setup_test_pair(
        num_blocks=2, dtype=dtype, seed=300
    )

    req_id = "test_req_reorder"
    uuid = 98765
    producer.register_read(req_id, uuid, [0, 1])

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertNotEmpty(endpoints)
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[1, 0],
        local_block_ids=[0, 1],
    )

    self._wait_for_transfer(consumer, req_id, is_receiver=True)

    for idx, t in enumerate(dst_caches):
      # local block 0 <- remote block 1
      self.assertTrue(torch.equal(t[0].cpu(), src_refs[idx][1]))
      # local block 1 <- remote block 0
      self.assertTrue(torch.equal(t[1].cpu(), src_refs[idx][0]))

    # Poll producer until it's done sending
    self._wait_for_transfer(producer, req_id, is_receiver=False)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_large_complex_non_contiguous_and_reorder(self, dtype):
    num_blocks = 16
    producer, consumer, src_refs, dst_caches = self._setup_test_pair(
        num_blocks=num_blocks, dtype=dtype, seed=400
    )

    req_id = "test_req_large_complex"
    uuid = 13579

    remote_blocks = [0, 2, 3, 5, 6, 7, 9, 11, 12, 14]
    requested_remote = list(reversed(remote_blocks))
    local_blocks = list(range(len(remote_blocks)))

    producer.register_read(req_id, uuid, remote_blocks)

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertNotEmpty(endpoints)
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=requested_remote,
        local_block_ids=local_blocks,
    )

    self._wait_for_transfer(consumer, req_id, is_receiver=True, timeout_s=15.0)

    for idx, t in enumerate(dst_caches):
      for local_idx, local_block in enumerate(local_blocks):
        remote_block = requested_remote[local_idx]
        self.assertTrue(
            torch.equal(t[local_block].cpu(), src_refs[idx][remote_block])
        )

      for local_block in range(len(local_blocks), num_blocks):
        self.assertTrue(
            torch.equal(
                t[local_block].cpu(), torch.zeros_like(t[local_block].cpu())
            )
        )

    # Poll producer until it's done sending
    self._wait_for_transfer(producer, req_id, is_receiver=False)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_parallel_pull(self, dtype):
    producer, consumer, src_refs, dst_caches = self._setup_test_pair(
        num_blocks=2, dtype=dtype, seed=500
    )

    req_id = "test_req_parallel"
    uuid = 99999
    producer.register_read(req_id, uuid, [0, 1])

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertNotEmpty(endpoints)
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
        parallelism=2,
    )

    self._wait_for_transfer(consumer, req_id, is_receiver=True)

    for idx, t in enumerate(dst_caches):
      self.assertTrue(torch.equal(t.cpu(), src_refs[idx]))

    # Poll producer until it's done sending
    self._wait_for_transfer(producer, req_id, is_receiver=False)


if __name__ == "__main__":
  absltest.main()

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

"""Real-TPU test: raw slot copies of packed fp4 (float4_e2m1fn_x2) tensors.

torch stores two fp4 values per byte of a float4_e2m1fn_x2 tensor; torch_tpu
backs such a tensor with an XLA F4E2M1FN array of twice the minor dimension,
accounted at one byte per element, so the storage counts twice the tensor's
nbytes. Raw transfer addresses the tensor by its on-device major-dimension
slices (`major_slice_bytes`), which is what a host buffer must hold per slot.

Never print a float4_e2m1fn_x2 tensor: torch_tpu aborts in its repr. Run with
`pytest --tb=short` (the long traceback reprs function arguments).
"""

from absl.testing import absltest
import torch

try:
  import torch_tpu  # noqa: F401  # pylint: disable=g-import-not-at-top,unused-import
except ImportError:  # pragma: no cover - chip-less hosts
  torch_tpu = None

from tpu_sync.api.torch import raw_transfer  # pylint: disable=g-import-not-at-top


def _tpu_available():
  try:
    return (
        torch_tpu is not None
        and torch.accelerator.is_available()
        and torch.accelerator.current_accelerator().type == "tpu"
    )
  except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
    return False


E, K, N = 6, 1024, 2560  # logical [E, K, N] fp4; x2 shape [E, K, N/2]


def _bytes(seed, rows, cols):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 255, (rows, cols), generator=g, dtype=torch.int32).to(
      torch.uint8
  )


class RawTransferFp4Test(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not _tpu_available():
      self.skipTest("needs a TPU")

  def _fp4_on_device(self, seed):
    """Returns an F4-backed x2 tensor filled via raw H2D of deterministic bytes."""
    t = torch.empty((E, K, N // 2), dtype=torch.float4_e2m1fn_x2, device="tpu")
    storage, nbytes = t.untyped_storage().nbytes(), t.nbytes
    if storage != 2 * nbytes:
      self.skipTest(
          "torch.empty(float4_e2m1fn_x2) is not F4-backed here (storage"
          f" {storage} for nbytes {nbytes}); the model's weights come from the"
          " Pallas quantizer"
      )
    slot = raw_transfer.major_slice_bytes(t)
    host = torch.empty((E, slot), dtype=torch.uint8, pin_memory=True)
    host.copy_(_bytes(seed, E, slot))
    raw_transfer.transfer_h2d_async(
        host,
        t,
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[E],
    ).wait()
    return t

  def _image(self, t):
    """Every slot's on-device bytes."""
    slot = raw_transfer.major_slice_bytes(t)
    host = torch.empty((t.shape[0], slot), dtype=torch.uint8, pin_memory=True)
    raw_transfer.transfer_d2h_async(
        t,
        host,
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[t.shape[0]],
    ).wait()
    return host.clone()

  def test_major_slice_is_the_on_device_slice(self):
    q = self._fp4_on_device(1)
    slot = raw_transfer.major_slice_bytes(q)
    # The device packs fp4: a slot is half of what the storage accounting
    # (one byte per F4 element) says. The code would also take an unpacked
    # device's slices (the on-device size over the major dimension).
    self.assertEqual(slot, K * N // 2)
    self.assertEqual(q.untyped_storage().nbytes(), 2 * slot * E)
    u8 = _bytes(2, E, K * N // 2).to("tpu")
    self.assertEqual(raw_transfer.major_slice_bytes(u8), K * N // 2)

  def test_a_uint8_view_of_an_fp4_tensor_is_not_its_storage(self):
    # `.view(float4_e2m1fn_x2)` of a uint8 device tensor keeps the uint8
    # buffer, whose size is half of what the x2 shape resolves to: not a
    # full reinterpret of an F4 storage, so raw transfer refuses it rather
    # than DMA half a slot.
    v = _bytes(3, E, K * N // 2).to("tpu").view(torch.float4_e2m1fn_x2)
    with self.assertRaisesRegex(ValueError, "span"):
      raw_transfer.major_slice_bytes(v)

  def test_slot_round_trips_byte_for_byte(self):
    src, dst = self._fp4_on_device(4), self._fp4_on_device(5)
    src_image, dst_before = self._image(src), self._image(dst)
    slot = raw_transfer.major_slice_bytes(src)
    host = torch.empty((1, slot), dtype=torch.uint8, pin_memory=True)
    raw_transfer.transfer_d2h_async(
        src,
        host,
        src_offsets_major_dim=[1],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[1],
    ).wait()
    self.assertTrue(torch.equal(host[0], src_image[1]))
    raw_transfer.transfer_h2d_async(
        host,
        dst,
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[4],
        copy_sizes_major_dim=[1],
    ).wait()
    got = self._image(dst)
    self.assertTrue(torch.equal(got[4], src_image[1]))
    self.assertFalse(torch.equal(got[4], dst_before[4]))
    self.assertTrue(torch.equal(got[:4], dst_before[:4]))
    self.assertTrue(torch.equal(got[5], dst_before[5]))


if __name__ == "__main__":
  absltest.main()

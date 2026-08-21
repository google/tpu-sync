// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace tpu_raiden {
namespace torch {

// A materialized PjRtBuffer together with the owning DeviceBufferRef that keeps
// it alive.
//
// IMPORTANT: the `buffer` pointer is only valid while `ref` is alive. Callers
// that retain `buffer` past this call MUST keep `ref` alive for as long as they
// use it (e.g. store it in a member / attach it to the transfer future).
struct UnpackedTensor {
  raiden::RaidenBufferHandle buffer;
  std::optional<torch_tpu::DeviceBufferRef> ref;
  std::vector<int64_t> logical_dimensions;
  size_t logical_slice_byte_size = 0;
  size_t logical_physical_size = 0;
};

// Unpacks a single PyTorch tensor into its materialized PjRtBuffer AND the
// owning DeviceBufferRef. Throws if validation or materialization fails.
UnpackedTensor UnpackTorchTensor(const at::Tensor& tensor,
                                 bool unsafe_skip_buffer_lock = false);

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_H_

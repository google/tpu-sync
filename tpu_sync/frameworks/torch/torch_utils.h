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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace tpu_raiden {
namespace torch {

// A 2D (layer x shard) batch of unpacked buffers together with the owning
// DeviceBufferRefs (flattened). Returned by value -- rather than via an
// optional out-param -- precisely so the caller cannot silently drop the refs:
// it MUST take `refs` and keep them alive for as long as it uses `buffers`
// (view-materialized buffers are owned solely by these refs).
//
// IMPORTANT: the `buffer` pointer is only valid while `ref` is alive. Callers
// that retain `buffer` past this call MUST keep `ref` alive for as long as they
// use it (e.g. store it in a member / attach it to the transfer future).
struct UnpackedTensors {
  std::vector<std::vector<raiden::RaidenBufferHandle>> buffers;
  std::vector<torch_tpu::DeviceBufferRef> refs;
  std::vector<int64_t> logical_dimensions;
  size_t logical_slice_byte_size = 0;
  size_t logical_physical_size = 0;
  bool has_logical_metadata = false;
};

// Unpacks a 2D vector of PyTorch sharded tensors (the batch form of
// UnpackTorchTensor). Throws if validation or materialization fails.

UnpackedTensors UnpackTorchTensors(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    bool unsafe_skip_buffer_lock = false);

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_UTILS_H_

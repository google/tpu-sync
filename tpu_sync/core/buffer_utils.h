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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_BUFFER_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_BUFFER_UTILS_H_

#include <vector>

#include "tpu_sync/core/raw_transfer_core.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

namespace tpu_raiden {

// Unpacks a 2D matrix of xla::PjRtBuffer pointers (layers x shards) into a 2D
// matrix of raiden::RaidenBufferHandle.
std::vector<std::vector<raiden::RaidenBufferHandle>> UnpackLayers(
    const std::vector<std::vector<xla::PjRtBuffer*>>& device_buffers,
    bool unsafe_skip_buffer_lock = false);

// Alias for UnpackLayers
inline std::vector<std::vector<raiden::RaidenBufferHandle>> UnpackPjRtBuffers(
    const std::vector<std::vector<xla::PjRtBuffer*>>& device_buffers,
    bool unsafe_skip_buffer_lock = false) {
  return UnpackLayers(device_buffers, unsafe_skip_buffer_lock);
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_BUFFER_UTILS_H_

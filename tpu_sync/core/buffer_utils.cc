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

#include "tpu_sync/core/buffer_utils.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace tpu_raiden {

std::vector<std::vector<raiden::RaidenBufferHandle>> UnpackLayers(
    const std::vector<std::vector<xla::PjRtBuffer*>>& device_buffers,
    bool unsafe_skip_buffer_lock) {
  std::vector<std::vector<raiden::RaidenBufferHandle>> handles;
  size_t num_layers = device_buffers.size();
  if (num_layers == 0) return handles;

  size_t num_shards = device_buffers[0].size();
  handles.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    if (device_buffers[l].size() != num_shards) {
      throw std::invalid_argument(
          "Symmetrical shards count mismatch across layers during PjRtBuffer "
          "unpack");
    }
    std::vector<raiden::RaidenBufferHandle> shard_buffers;
    shard_buffers.reserve(num_shards);
    for (size_t sh = 0; sh < num_shards; ++sh) {
      auto handle_or = raiden::RaidenBufferHandle::Acquire(
          device_buffers[l][sh], /*c_api=*/nullptr, /*extension=*/nullptr,
          unsafe_skip_buffer_lock);
      if (!handle_or.ok()) {
        throw std::runtime_error(absl::StrCat(
            "Failed to acquire RaidenBufferHandle: ",
            handle_or.status().message()));
      }
      shard_buffers.push_back(std::move(handle_or.value()));
    }
    handles.push_back(std::move(shard_buffers));
  }
  return handles;
}

}  // namespace tpu_raiden

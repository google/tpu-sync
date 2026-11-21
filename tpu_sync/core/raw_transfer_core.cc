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

#include "tpu_sync/core/raw_transfer_core.h"

#include <optional>
#include <vector>

#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace raiden {

std::optional<ExternalDmaInfo> RaidenBufferHandle::AcquireExternalDmaAccess()
    const {
  std::optional<ExternalDmaInfo> result;
  return result;
}

std::vector<int> DetectNumaNodes(
    const std::vector<std::vector<RaidenBufferHandle>>& layer_buffers) {
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (const auto& buf : layer) {
      if (buf.device) {
        int node = tpu_raiden::GetPjRtDeviceNumaNode(buf.device);
        if (node >= 0) {
          bool found = false;
          for (int n : unique_numa_nodes) {
            if (n == node) {
              found = true;
              break;
            }
          }
          if (!found) unique_numa_nodes.push_back(node);
        }
      }
    }
  }
  return unique_numa_nodes;
}

std::vector<tpu_raiden::transport::ShardInfo> DetectShards(
    const std::vector<std::vector<RaidenBufferHandle>>& layer_buffers) {
  std::vector<tpu_raiden::transport::ShardInfo> shards;
  if (layer_buffers.empty()) {
    return shards;
  }
  // Every layer places shard `i` on the same device, so layer 0 is
  // representative.
  shards.reserve(layer_buffers[0].size());
  for (size_t sh = 0; sh < layer_buffers[0].size(); ++sh) {
    const RaidenBufferHandle& buf = layer_buffers[0][sh];
    int node = -1;
    if (buf.device) {
      node = tpu_raiden::GetPjRtDeviceNumaNode(buf.device);
    }
    shards.push_back({.local_index = static_cast<int>(sh),
                      .numa_node = node >= 0 ? node : -1});
  }
  return shards;
}

}  // namespace raiden

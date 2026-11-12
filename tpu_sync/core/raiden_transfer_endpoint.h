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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_RAIDEN_TRANSFER_ENDPOINT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_RAIDEN_TRANSFER_ENDPOINT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/peer_endpoint.h"

namespace tpu_raiden {

struct RaidenTransferEndpoint {
  std::string endpoint;
  std::vector<int64_t> shards;
  std::vector<::tpu_sync::rpc::PoolHostAddrsProto> layer_host_addrs;
  std::string peregrine_endpoint;

  transport::PeerEndpoint peer_endpoint() const {
    return {endpoint, peregrine_endpoint};
  }

  ::tpu_sync::proto::RaidenTransferEndpointProto ToProto() const {
    ::tpu_sync::proto::RaidenTransferEndpointProto proto;
    proto.set_endpoint(endpoint);
    proto.mutable_shards()->Add(shards.begin(), shards.end());
    proto.mutable_layer_host_addrs()->Reserve(layer_host_addrs.size());
    for (const auto& addr : layer_host_addrs) {
      *proto.add_layer_host_addrs() = addr;
    }
    if (!peregrine_endpoint.empty()) {
      proto.set_peregrine_endpoint(peregrine_endpoint);
    }
    return proto;
  }

  static RaidenTransferEndpoint FromProto(
      const ::tpu_sync::proto::RaidenTransferEndpointProto& proto) {
    return {
        .endpoint = proto.endpoint(),
        .shards = {proto.shards().begin(), proto.shards().end()},
        .layer_host_addrs = {proto.layer_host_addrs().begin(),
                             proto.layer_host_addrs().end()},
        .peregrine_endpoint = proto.peregrine_endpoint(),
    };
  }

  bool operator==(const RaidenTransferEndpoint& other) const {
    if (endpoint != other.endpoint || shards != other.shards ||
        peregrine_endpoint != other.peregrine_endpoint ||
        layer_host_addrs.size() != other.layer_host_addrs.size()) {
      return false;
    }
    for (size_t i = 0; i < layer_host_addrs.size(); ++i) {
      const auto& a = layer_host_addrs[i];
      const auto& b = other.layer_host_addrs[i];
      if (a.block_stride_bytes() != b.block_stride_bytes() ||
          a.num_blocks() != b.num_blocks() ||
          !std::equal(a.host_base_addrs().begin(), a.host_base_addrs().end(),
                      b.host_base_addrs().begin(), b.host_base_addrs().end()) ||
          a.host_slot_by_block().size() != b.host_slot_by_block().size()) {
        return false;
      }
      for (const auto& [block, slot] : a.host_slot_by_block()) {
        auto it = b.host_slot_by_block().find(block);
        if (it == b.host_slot_by_block().end() || it->second != slot) {
          return false;
        }
      }
    }
    return true;
  }
};

// A group of transfer endpoints owned by a single peer worker, tagged with the
// worker's mesh node_id (used for source<->destination worker matching during
// ReadRemote) and worker_id (diagnostics/logging only).
struct RaidenWorkerEndpoints {
  int64_t node_id = 0;
  std::string worker_id;
  std::vector<RaidenTransferEndpoint> endpoints;

  bool operator==(const RaidenWorkerEndpoints& other) const {
    return node_id == other.node_id && worker_id == other.worker_id &&
           endpoints == other.endpoints;
  }
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_RAIDEN_TRANSFER_ENDPOINT_H_

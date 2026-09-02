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

#include "tpu_sync/core/pool_reshard_send_slots.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>

#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

int64_t CountPoolReshardSendSlots(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    const ::tpu_sync::rpc::ShardPushScheduleProto& schedule) {
  // Peers with at least one entry, per pool group.
  std::map<int32_t, std::set<std::string>> peers_by_group;
  for (const auto& entry : schedule.entries()) {
    peers_by_group[entry.pool_group()].insert(entry.dst_peer());
  }
  int64_t slots = 0;
  for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
    int32_t group_idx = -1;
    for (int g = 0; g < plan.pool_groups_size(); ++g) {
      const auto& indices = plan.pool_groups(g).pool_indices();
      if (std::find(indices.begin(), indices.end(), encoded_pool_idx) !=
          indices.end()) {
        group_idx = g;
        break;
      }
    }
    auto it = peers_by_group.find(group_idx);
    if (it == peers_by_group.end()) continue;
    slots += static_cast<int64_t>(it->second.size());
  }
  return slots;
}

}  // namespace tpu_raiden

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

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

void AddEntry(::tpu_sync::rpc::ShardPushScheduleProto* schedule,
              const std::string& peer, int32_t group) {
  auto* entry = schedule->add_entries();
  entry->set_dst_peer(peer);
  entry->set_pool_group(group);
  entry->set_size_bytes(1);
  entry->set_count(1);
}

::tpu_sync::rpc::StartTransferRequest PlanWithGroups(
    const std::vector<std::vector<int32_t>>& groups) {
  ::tpu_sync::rpc::StartTransferRequest plan;
  for (const auto& pools : groups) {
    auto* group = plan.add_pool_groups();
    for (int32_t pool : pools) {
      group->add_pool_indices(pool);
      plan.add_transfer_pool_indices(pool);
    }
  }
  return plan;
}

TEST(PoolReshardSendSlots, SingleDestinationIsPoolsTimesOnePeer) {
  auto plan = PlanWithGroups({{0, 1, 2}, {3}});
  ::tpu_sync::rpc::ShardPushScheduleProto schedule;
  AddEntry(&schedule, "10.0.0.2:9400", 0);
  AddEntry(&schedule, "10.0.0.2:9400", 1);
  EXPECT_EQ(CountPoolReshardSendSlots(plan, schedule), 4);
}

TEST(PoolReshardSendSlots, ReplicatedDestinationsIsPoolsTimesPeers) {
  auto plan = PlanWithGroups({{0, 1}, {2}});
  ::tpu_sync::rpc::ShardPushScheduleProto schedule;
  for (const std::string& peer : {"10.0.0.2:9400", "10.0.0.2:9401"}) {
    AddEntry(&schedule, peer, 0);
    AddEntry(&schedule, peer, 1);
  }
  EXPECT_EQ(CountPoolReshardSendSlots(plan, schedule), 6);
}

TEST(PoolReshardSendSlots, ShardedGroupCountsOnlyItsPeers) {
  // FA group (pools 0,1) reaches both peers; the state group (pools 2,3)
  // is routed to one peer only: 2*2 + 2*1 = 6, not 4*2 = 8.
  auto plan = PlanWithGroups({{0, 1}, {2, 3}});
  ::tpu_sync::rpc::ShardPushScheduleProto schedule;
  AddEntry(&schedule, "10.0.0.2:9400", 0);
  AddEntry(&schedule, "10.0.0.2:9401", 0);
  AddEntry(&schedule, "10.0.0.2:9401", 1);
  EXPECT_EQ(CountPoolReshardSendSlots(plan, schedule), 6);
}

TEST(PoolReshardSendSlots, GroupWithoutEntriesContributesNoSlot) {
  auto plan = PlanWithGroups({{0}, {1}});
  ::tpu_sync::rpc::ShardPushScheduleProto schedule;
  AddEntry(&schedule, "10.0.0.2:9400", 0);
  EXPECT_EQ(CountPoolReshardSendSlots(plan, schedule), 1);
}

TEST(PoolReshardSendSlots, EmptyScheduleHasNoSlots) {
  auto plan = PlanWithGroups({{0}});
  ::tpu_sync::rpc::ShardPushScheduleProto schedule;
  EXPECT_EQ(CountPoolReshardSendSlots(plan, schedule), 0);
}

}  // namespace
}  // namespace tpu_raiden

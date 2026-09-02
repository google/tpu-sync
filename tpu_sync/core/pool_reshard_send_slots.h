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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_POOL_RESHARD_SEND_SLOTS_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_POOL_RESHARD_SEND_SLOTS_H_

#include <cstdint>

#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

// Number of (pool, peer) push completions a sender-side pool-reshard plan
// produces: for every transfer pool, the distinct destination peers among
// the local schedule entries of that pool's group. A pool whose group has no
// entries for this sender (it owns none of the group's bytes, or routes the
// group to no destination) produces no push and contributes no slot. With
// multi-destination plans the count is NOT pools x peers: a sharded
// destination group (e.g. GDN state routed to one TP shard) pushes each of
// its pools to a single peer.
int64_t CountPoolReshardSendSlots(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    const ::tpu_sync::rpc::ShardPushScheduleProto& schedule);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_POOL_RESHARD_SEND_SLOTS_H_

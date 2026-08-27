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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_POOL_RESHARD_PLANNER_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_POOL_RESHARD_PLANNER_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// One executor schedule entry — the 12-tuple of
// rpc/raiden_controller.py's schedule lists, in wire order
// (ShardPushEntryProto fields 1-12).
struct ScheduleEntry {
  std::string dst_peer;
  int32_t dst_shard_idx = 0;
  int64_t dst_offset_bytes = 0;
  int64_t src_offset_bytes = 0;
  int64_t size_bytes = 0;
  int64_t src_block_id = 0;
  int64_t dst_block_id = 0;
  int64_t src_stride_bytes = 0;
  int64_t dst_stride_bytes = 0;
  int32_t count = 1;
  int32_t layer_idx = 0;
  int32_t pool_group = 0;
};

// One tag class's slice of the plan (TransferPlan.pool_groups element).
struct PlanPoolGroup {
  std::vector<int32_t> pool_indices;
  std::vector<int64_t> dst_device_block_ids;
  int32_t expected_pushes = 0;
  std::vector<int64_t> dst_expected_extent_bytes;
  int32_t order_rank = 0;
};

// The pool-path subset of TransferPlan that the encoder and coordinator
// consume. Field-for-field mirror of _build_byte_span_plan_claimed's
// return value.
struct PoolReshardPlan {
  std::vector<RaidenId> src_units;  // active source units, rank order
  RaidenId dst_unit;
  // Per source unit: shard 0's entry list (pool planning enforces one
  // endpoint per unit, so the inner Python dict always has the single key
  // 0). Keyed in src_units order.
  std::map<RaidenId, std::vector<ScheduleEntry>,
           RequestBlockRegistry::RaidenIdLess>
      schedules;
  std::map<RaidenId, std::string, RequestBlockRegistry::RaidenIdLess>
      worker_rpc_addresses;
  std::string dst_peer;  // worker_data_addresses[dst_unit][0]
  int64_t uuid = 0;
  std::string req_id;
  int64_t expected_block_count = 0;
  int32_t expected_pushes_per_pool = 0;
  std::vector<int32_t> transfer_pool_indices;
  std::vector<std::string> pool_dtype_tags;
  std::vector<int64_t> dst_device_block_ids;
  std::map<RaidenId, int32_t, RequestBlockRegistry::RaidenIdLess>
      src_schedule_keys;
  int32_t parallelism = 1;
  int64_t num_tokens = 0;
  std::map<std::string, int64_t> skipped_pool_counts;
  std::vector<int64_t> dst_expected_extent_bytes;
  std::vector<PlanPoolGroup> pool_groups;
  bool skip_d2h = false;
};

// Inputs to the plan build, mirroring _build_pool_reshard_plan's keyword
// arguments (metadata protos arrive exactly as fetched).
struct PlanRequest {
  std::vector<RaidenId> src_units;
  std::vector<RaidenId> dst_units;
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> src_metadata;
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> dst_metadata;
  std::string req_id;
  int64_t uuid = 0;
  std::vector<int64_t> dst_device_block_ids;
  int64_t num_tokens = 0;
  std::vector<std::string> transfer_pool_tags;
  std::optional<int32_t> parallelism;
  std::vector<int64_t> dst_block_counts;
  // Per-tag destination clip (empty = all zeros): plan only bytes >=
  // dst_skip_bytes[i] of tag i's global destination byte space, re-based
  // onto that tag's destination block list. Page-multiple, v1-only.
  std::vector<int64_t> dst_skip_bytes;
};

// _build_byte_span_plan_claimed: validates, claims the registration
// snapshot through `registry` under `claim_owner`, binds, and emits.
// Failure does NOT roll back the claim — the caller abandons it
// (_build_pool_reshard_plan's contract).
absl::StatusOr<PoolReshardPlan> BuildPoolReshardPlan(
    const PlanRequest& request, RequestBlockRegistry* registry,
    const void* claim_owner);

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_POOL_RESHARD_PLANNER_H_

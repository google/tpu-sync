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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_PLAN_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

// Encapsulates a validated StartTransferRequest and its precomputed block/shard
// lookup tables for chunk resolution, host-block mapping, and pool push
// progress expectations.
class TransferPlan {
 public:
  using EntryList = std::vector<const ::tpu_sync::rpc::ShardPushEntryProto*>;
  using BlockEntryMap = absl::flat_hash_map<int64_t, EntryList>;

  static absl::Status Validate(
      const ::tpu_sync::rpc::StartTransferRequest& request, size_t num_pools,
      bool explicit_pools, absl::Span<const kv_cache::PoolSpec> pools);

  static absl::StatusOr<TransferPlan> Create(
      ::tpu_sync::rpc::StartTransferRequest request, bool is_sender,
      size_t num_pools, bool explicit_pools,
      absl::Span<const kv_cache::PoolSpec> pools,
      absl::flat_hash_map<int64_t, int64_t> host_block_of = {});

  TransferPlan(const TransferPlan& other);
  TransferPlan& operator=(const TransferPlan& other);
  TransferPlan(TransferPlan&& other) noexcept;
  TransferPlan& operator=(TransferPlan&& other) noexcept;
  ~TransferPlan() = default;

  bool is_sender() const { return is_sender_; }
  const ::tpu_sync::rpc::StartTransferRequest& request() const {
    return request_;
  }
  const absl::flat_hash_map<int64_t, int64_t>* host_block_of() const {
    return &host_block_of_;
  }

  absl::StatusOr<std::optional<transport::PoolPushProgressSpec>>
  GetPoolPushProgressSpec(size_t pool_idx) const;

  std::optional<size_t> ExpectedPushSenders() const;

  absl::StatusOr<std::vector<int64_t>> PlanHostBlocks(
      uint64_t uuid, absl::Span<const int64_t> block_ids) const;

  std::vector<transport::BlockChunk> ResolvePlanBlockChunks(
      size_t layer_idx, size_t shard_idx, absl::Span<const int64_t> block_ids,
      size_t total_bytes, int64_t sender_node_id, absl::string_view peer,
      int64_t src_block_id, int64_t dst_block_id, bool explicit_pools,
      absl::FunctionRef<uint8_t*(int64_t block_id, int64_t host_block)>
          block_host_base) const;

 private:
  TransferPlan(::tpu_sync::rpc::StartTransferRequest request, bool is_sender,
               absl::flat_hash_map<int64_t, int64_t> host_block_of,
               absl::flat_hash_set<int64_t> staged_device_blocks);

  void RebuildScheduleIndex();

  ::tpu_sync::rpc::StartTransferRequest request_;
  bool is_sender_ = false;
  absl::flat_hash_map<int64_t, int64_t> host_block_of_;
  absl::flat_hash_set<int64_t> staged_device_blocks_;
  absl::flat_hash_map<int32_t, BlockEntryMap> schedule_by_block_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_PLAN_H_

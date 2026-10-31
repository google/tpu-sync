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

#include "tpu_sync/kv_cache/transfer_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

namespace {

const ::tpu_sync::rpc::PoolHostAddrsProto* FindReceiverAddrs(
    const ::tpu_sync::rpc::StartTransferRequest& request,
    absl::string_view peer, int32_t idx) {
  if (peer.empty()) return nullptr;
  auto receiver_it = request.receiver_addrs().find(std::string(peer));
  if (receiver_it == request.receiver_addrs().end()) return nullptr;
  const auto& pools = receiver_it->second.pools();
  auto pool_it = pools.find(idx);
  if (pool_it == pools.end() || pool_it->second.host_base_addrs().empty()) {
    return nullptr;
  }
  return &pool_it->second;
}

const ::tpu_sync::rpc::PoolHostAddrsProto* GetReceiverPoolAddrs(
    const ::tpu_sync::rpc::StartTransferRequest& request,
    absl::string_view peer, size_t pool_idx) {
  if (request.pool_groups_size() == 0) return nullptr;
  int32_t wire_pool_idx = static_cast<int32_t>(pool_idx);
  if (auto wire_it = request.wire_pool_indices().find(wire_pool_idx);
      wire_it != request.wire_pool_indices().end()) {
    wire_pool_idx = wire_it->second;
  }
  return FindReceiverAddrs(request, peer, wire_pool_idx);
}

const ::tpu_sync::rpc::PoolHostAddrsProto* GetReceiverLayerAddrs(
    const ::tpu_sync::rpc::StartTransferRequest& request,
    absl::string_view peer, size_t layer_idx) {
  if (request.pool_groups_size() > 0) return nullptr;
  return FindReceiverAddrs(request, peer, static_cast<int32_t>(layer_idx));
}

uint8_t* RemoteAddress(const ::tpu_sync::rpc::PoolHostAddrsProto* pool,
                       int64_t dst_shard_idx, int64_t dst_block_id,
                       int64_t dst_offset, int64_t size) {
  if (pool == nullptr) return nullptr;
  int64_t host_block = dst_block_id;
  if (!pool->host_slot_by_block().empty()) {
    auto it = pool->host_slot_by_block().find(host_block);
    if (it == pool->host_slot_by_block().end()) return nullptr;
    host_block = it->second;
  }
  if (dst_shard_idx < 0 || dst_shard_idx >= pool->host_base_addrs_size() ||
      host_block < 0 || host_block >= pool->num_blocks() ||
      pool->block_stride_bytes() <= 0 || dst_offset < 0 || size < 0 ||
      dst_offset > pool->block_stride_bytes() - size) {
    return nullptr;
  }
  const uint64_t addr = pool->host_base_addrs(dst_shard_idx) +
                        static_cast<uint64_t>(host_block) *
                            static_cast<uint64_t>(pool->block_stride_bytes()) +
                        static_cast<uint64_t>(dst_offset);
  return reinterpret_cast<uint8_t*>(addr);
}

}  // namespace

absl::Status TransferPlan::Validate(
    const ::tpu_sync::rpc::StartTransferRequest& request, size_t num_pools,
    bool explicit_pools, absl::Span<const kv_cache::PoolSpec> pools) {
  const bool has_transfer_pools = request.transfer_pool_indices_size() > 0;
  if (has_transfer_pools != (request.pool_groups_size() > 0)) {
    return absl::InvalidArgumentError(
        "pool-addressed plans must declare pool_groups partitioning their "
        "transfer_pool_indices");
  }
  if (has_transfer_pools) {
    for (const auto& group : request.pool_groups()) {
      if (group.expected_pushes() <= 0) {
        return absl::InvalidArgumentError(
            "every pool group must expect a positive push count");
      }
    }
    if (request.req_id().empty()) {
      return absl::InvalidArgumentError(
          "pool-keyed transfer plans require a non-empty req_id");
    }
    std::set<int32_t> seen_pool_indices;
    for (int32_t pool_idx : request.transfer_pool_indices()) {
      if (pool_idx < 0 || static_cast<size_t>(pool_idx) >= num_pools) {
        return absl::InvalidArgumentError(
            absl::StrCat("transfer pool index ", pool_idx,
                         " out of range: ", num_pools, " registered pools"));
      }
      if (!seen_pool_indices.insert(pool_idx).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("duplicate transfer pool index ", pool_idx));
      }
    }
  }
  if (request.pool_dtype_tags_size() > 0 && explicit_pools) {
    if (static_cast<size_t>(request.pool_dtype_tags_size()) != pools.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "plan pool_dtype_tags count ", request.pool_dtype_tags_size(),
          " does not match local pool count ", pools.size()));
    }
    for (size_t pool_idx = 0; pool_idx < pools.size(); ++pool_idx) {
      if (request.pool_dtype_tags(pool_idx) != pools[pool_idx].dtype_tag) {
        std::string local_pools;
        for (size_t i = 0; i < pools.size(); ++i) {
          absl::StrAppend(&local_pools, i ? "," : "", pools[i].tag, ":",
                          pools[i].dtype_tag);
        }
        return absl::InvalidArgumentError(absl::StrCat(
            "plan dtype tag mismatch for pool ", pool_idx, " (",
            pools[pool_idx].tag, "): plan=", request.pool_dtype_tags(pool_idx),
            " local=", pools[pool_idx].dtype_tag, "; plan dtype tags=[",
            absl::StrJoin(request.pool_dtype_tags(), ","), "] local pools=[",
            local_pools, "]"));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<TransferPlan> TransferPlan::Create(
    ::tpu_sync::rpc::StartTransferRequest request, bool is_sender,
    size_t num_pools, bool explicit_pools,
    absl::Span<const kv_cache::PoolSpec> pools,
    absl::flat_hash_map<int64_t, int64_t> host_block_of) {
  ABSL_RETURN_IF_ERROR(Validate(request, num_pools, explicit_pools, pools));
  absl::flat_hash_set<int64_t> staged_device_blocks;
  if (request.pool_groups_size() == 0) {
    for (const auto& [src_shard, schedule] : request.shard_push_schedules()) {
      (void)src_shard;
      for (const auto& entry : schedule.entries()) {
        staged_device_blocks.insert(is_sender ? entry.src_block_id()
                                              : entry.dst_block_id());
      }
    }
  }
  return TransferPlan(std::move(request), is_sender, std::move(host_block_of),
                      std::move(staged_device_blocks));
}

TransferPlan::TransferPlan(::tpu_sync::rpc::StartTransferRequest request,
                           bool is_sender,
                           absl::flat_hash_map<int64_t, int64_t> host_block_of,
                           absl::flat_hash_set<int64_t> staged_device_blocks)
    : request_(std::move(request)),
      is_sender_(is_sender),
      host_block_of_(std::move(host_block_of)),
      staged_device_blocks_(std::move(staged_device_blocks)) {
  RebuildScheduleIndex();
}

TransferPlan::TransferPlan(const TransferPlan& other)
    : request_(other.request_),
      is_sender_(other.is_sender_),
      host_block_of_(other.host_block_of_),
      staged_device_blocks_(other.staged_device_blocks_) {
  RebuildScheduleIndex();
}

TransferPlan& TransferPlan::operator=(const TransferPlan& other) {
  if (this != &other) {
    request_ = other.request_;
    is_sender_ = other.is_sender_;
    host_block_of_ = other.host_block_of_;
    staged_device_blocks_ = other.staged_device_blocks_;
    RebuildScheduleIndex();
  }
  return *this;
}

TransferPlan::TransferPlan(TransferPlan&& other) noexcept
    : request_(std::move(other.request_)),
      is_sender_(other.is_sender_),
      host_block_of_(std::move(other.host_block_of_)),
      staged_device_blocks_(std::move(other.staged_device_blocks_)) {
  RebuildScheduleIndex();
}

TransferPlan& TransferPlan::operator=(TransferPlan&& other) noexcept {
  if (this != &other) {
    request_ = std::move(other.request_);
    is_sender_ = other.is_sender_;
    host_block_of_ = std::move(other.host_block_of_);
    staged_device_blocks_ = std::move(other.staged_device_blocks_);
    RebuildScheduleIndex();
  }
  return *this;
}

void TransferPlan::RebuildScheduleIndex() {
  schedule_by_block_.clear();
  for (const auto& [src_shard, schedule] : request_.shard_push_schedules()) {
    auto& block_map = schedule_by_block_[src_shard];
    for (const auto& entry : schedule.entries()) {
      block_map[is_sender_ ? entry.src_block_id() : entry.dst_block_id()]
          .push_back(&entry);
    }
  }
}

absl::StatusOr<std::optional<transport::PoolPushProgressSpec>>
TransferPlan::GetPoolPushProgressSpec(size_t pool_idx) const {
  if (request_.pool_groups_size() == 0) {
    return std::nullopt;
  }
  bool pool_is_transferred = false;
  for (int32_t transferred_pool_idx : request_.transfer_pool_indices()) {
    if (transferred_pool_idx >= 0 &&
        static_cast<size_t>(transferred_pool_idx) == pool_idx) {
      pool_is_transferred = true;
      break;
    }
  }
  if (!pool_is_transferred) {
    return absl::InvalidArgumentError(absl::StrCat(
        "pool ", pool_idx, " is not in the active plan's transfer set"));
  }
  for (const auto& group : request_.pool_groups()) {
    const auto& indices = group.pool_indices();
    if (std::find(indices.begin(), indices.end(),
                  static_cast<int32_t>(pool_idx)) != indices.end()) {
      return transport::PoolPushProgressSpec{
          .expected_pushes = static_cast<size_t>(group.expected_pushes()),
          .expected_pools =
              static_cast<size_t>(request_.transfer_pool_indices_size()),
      };
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "pool ", pool_idx, " is not in any of the plan's pool groups"));
}

std::optional<size_t> TransferPlan::ExpectedPushSenders() const {
  if (is_sender_ || request_.pool_groups_size() > 0 ||
      request_.shard_push_schedules_size() == 0) {
    return std::nullopt;
  }
  return static_cast<size_t>(request_.shard_push_schedules_size());
}

absl::StatusOr<std::vector<int64_t>> TransferPlan::PlanHostBlocks(
    uint64_t uuid, absl::Span<const int64_t> block_ids) const {
  if (request_.pool_groups_size() > 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "transfer plan ", uuid, " is pool-addressed; its bytes are ",
        "staged at pool offsets, not per-block host blocks"));
  }
  std::vector<int64_t> out;
  out.reserve(block_ids.size());
  for (int64_t id : block_ids) {
    if (!staged_device_blocks_.contains(id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("block ", id, " is not staged by transfer plan ", uuid));
    }
    if (host_block_of_.empty()) {
      out.push_back(id);
      continue;
    }
    auto it = host_block_of_.find(id);
    if (it == host_block_of_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("block ", id, " is not staged by transfer plan ", uuid));
    }
    out.push_back(it->second);
  }
  return out;
}

std::vector<transport::BlockChunk> TransferPlan::ResolvePlanBlockChunks(
    size_t layer_idx, size_t shard_idx, absl::Span<const int64_t> block_ids,
    size_t total_bytes, int64_t sender_node_id, absl::string_view peer,
    int64_t src_block_id, int64_t dst_block_id, bool explicit_pools,
    absl::FunctionRef<uint8_t*(int64_t block_id, int64_t host_block)>
        block_host_base) const {
  const int32_t fixed_src_shard = is_sender_
                                      ? static_cast<int32_t>(shard_idx)
                                      : static_cast<int32_t>(sender_node_id);
  auto schedule_it = fixed_src_shard != -1
                         ? schedule_by_block_.find(fixed_src_shard)
                         : schedule_by_block_.end();
  const auto block_entries = [](const BlockEntryMap& block_map,
                                int64_t block_id)
      -> absl::Span<const ::tpu_sync::rpc::ShardPushEntryProto* const> {
    auto it = block_map.find(block_id);
    if (it == block_map.end()) return {};
    return it->second;
  };

  const auto entry_targets_pool = [this, layer_idx](const auto& entry) -> bool {
    if (request_.pool_groups_size() == 0) {
      return true;
    }
    const int32_t group_idx = entry.pool_group();
    if (group_idx < 0 || group_idx >= request_.pool_groups_size()) {
      return false;
    }
    const auto& indices = request_.pool_groups(group_idx).pool_indices();
    return std::find(indices.begin(), indices.end(),
                     static_cast<int32_t>(layer_idx)) != indices.end();
  };

  const ::tpu_sync::rpc::PoolHostAddrsProto* dst_addrs = nullptr;
  if (is_sender_) {
    dst_addrs = explicit_pools
                    ? GetReceiverPoolAddrs(request_, peer, layer_idx)
                    : GetReceiverLayerAddrs(request_, peer, layer_idx);
  }

  std::vector<transport::BlockChunk> chunks;
  std::vector<transport::BlockChunk> block_resolved_chunks;
  size_t accumulated_bytes = 0;

  for (int64_t block_id : block_ids) {
    if (accumulated_bytes >= total_bytes) break;
    int64_t host_block = block_id;
    if (!host_block_of_.empty()) {
      auto hb = host_block_of_.find(block_id);
      if (hb == host_block_of_.end()) {
        return {};
      }
      host_block = hb->second;
    }
    uint8_t* const block_base = block_host_base(block_id, host_block);
    if (block_base == nullptr) {
      return {};
    }
    block_resolved_chunks.clear();

    if (is_sender_) {
      if (schedule_it != schedule_by_block_.end()) {
        const auto& schedule = schedule_it->second;
        for (const ::tpu_sync::rpc::ShardPushEntryProto* entry_ptr :
             block_entries(schedule, block_id)) {
          const ::tpu_sync::rpc::ShardPushEntryProto& entry = *entry_ptr;
          if (!entry_targets_pool(entry)) {
            continue;
          }
          if (!peer.empty() && entry.dst_peer() != peer) {
            continue;
          }
          if (dst_block_id != -1 && entry.dst_block_id() != dst_block_id) {
            continue;
          }
          if (static_cast<size_t>(entry.src_block_id()) == block_id) {
            size_t src_base_offset = entry.src_offset_bytes();
            size_t size = entry.size_bytes();
            size_t src_stride = entry.src_stride_bytes();
            int count = entry.count();
            if (count <= 0) count = 1;

            for (int c = 0; c < count; ++c) {
              size_t src_offset = src_base_offset + c * src_stride;
              const int64_t dst_offset =
                  entry.dst_offset_bytes() + c * entry.dst_stride_bytes();
              block_resolved_chunks.push_back(
                  {.ptr = block_base + src_offset,
                   .size = size,
                   .raddr = RemoteAddress(dst_addrs, entry.dst_shard_idx(),
                                          entry.dst_block_id(), dst_offset,
                                          entry.size_bytes())});
            }
          }
        }
      }
    } else {
      int found_src_shard = -1;
      if (sender_node_id != -1) {
        found_src_shard = static_cast<int>(sender_node_id);
      } else {
        for (const auto& [src_shard, src_schedule] : schedule_by_block_) {
          for (const ::tpu_sync::rpc::ShardPushEntryProto* entry_ptr :
               block_entries(src_schedule, block_id)) {
            const ::tpu_sync::rpc::ShardPushEntryProto& entry = *entry_ptr;
            if (entry_targets_pool(entry) &&
                static_cast<size_t>(entry.dst_shard_idx()) == shard_idx &&
                static_cast<size_t>(entry.dst_block_id()) == block_id &&
                (src_block_id == -1 ||
                 static_cast<size_t>(entry.src_block_id()) == src_block_id)) {
              found_src_shard = src_shard;
              break;
            }
          }
          if (found_src_shard != -1) break;
        }
      }

      if (found_src_shard != -1) {
        auto schedule_found_it = sender_node_id != -1
                                     ? schedule_it
                                     : schedule_by_block_.find(found_src_shard);
        if (schedule_found_it != schedule_by_block_.end()) {
          const auto& schedule = schedule_found_it->second;
          for (const ::tpu_sync::rpc::ShardPushEntryProto* entry_ptr :
               block_entries(schedule, block_id)) {
            const ::tpu_sync::rpc::ShardPushEntryProto& entry = *entry_ptr;
            if (!entry_targets_pool(entry)) {
              continue;
            }
            if (static_cast<size_t>(entry.dst_shard_idx()) == shard_idx &&
                static_cast<size_t>(entry.dst_block_id()) == block_id &&
                (src_block_id == -1 ||
                 static_cast<size_t>(entry.src_block_id()) == src_block_id)) {
              if (!peer.empty() && entry.dst_peer() != peer) {
                continue;
              }
              size_t dst_base_offset = entry.dst_offset_bytes();
              size_t size = entry.size_bytes();
              size_t dst_stride = entry.dst_stride_bytes();
              int count = entry.count();
              if (count <= 0) count = 1;

              for (int c = 0; c < count; ++c) {
                size_t dst_offset = dst_base_offset + c * dst_stride;
                block_resolved_chunks.push_back(
                    {.ptr = block_base + dst_offset, .size = size});
              }
            }
          }
        }
      }
    }

    for (const auto& chunk : block_resolved_chunks) {
      if (accumulated_bytes >= total_bytes) break;
      size_t size_to_add =
          std::min(chunk.size, total_bytes - accumulated_bytes);
      if (size_to_add > 0) {
        chunks.push_back(
            {.ptr = chunk.ptr, .size = size_to_add, .raddr = chunk.raddr});
        accumulated_bytes += size_to_add;
      }
    }
  }

  if (!chunks.empty()) {
    return chunks;
  }
  return {};
}

}  // namespace tpu_raiden

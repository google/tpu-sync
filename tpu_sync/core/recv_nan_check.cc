// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0

#include "tpu_sync/core/recv_nan_check.h"

#include <cstdlib>
#include <utility>

#include "absl/log/log.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

// Pool tags remain opaque to the transfer protocol. Only this diagnostic
// interprets the vLLM full-attention vocabulary (fa or fa.l<layer index>).
bool IsFullAttention(std::string_view tag) {
  if (tag == "fa") return true;
  if (!tag.starts_with("fa.l") || tag.size() == 4) return false;
  return std::all_of(tag.begin() + 4, tag.end(),
                     [](char ch) { return ch >= '0' && ch <= '9'; });
}

}  // namespace

std::unique_ptr<RecvNanCheck> RecvNanCheck::CreateIfEnabled(
    kv_cache::KVCacheManagerBase* base,
    const ::tpu_sync::rpc::StartTransferRequest& plan) {
  const char* enabled = std::getenv("TPU_RAIDEN_CHECK_RECV_FA_NAN");
  if (enabled == nullptr || std::string_view(enabled) != "1") return nullptr;
  auto check = std::unique_ptr<RecvNanCheck>(
      new RecvNanCheck(base, plan.uuid(), plan.req_id()));
  for (const auto& group : plan.pool_groups()) {
    for (int32_t pool_idx : group.pool_indices()) {
      const kv_cache::PoolSpec& spec = *base->pool(pool_idx);
      if (!IsFullAttention(spec.tag)) continue;
      auto format = NanFormatForDtype(spec.dtype_tag);
      auto segments = kv_cache::ExpandPoolLiveSegments(spec);
      if (!format || !segments.ok()) {
        LOG(WARNING) << "Receive FA KV NaN check skipped: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " tag=" << spec.tag << " dtype=" << spec.dtype_tag;
        continue;
      }
      Pool pool{.format = *format, .block_extent = 0};
      bool aligned = true;
      for (const auto& segment : *segments) {
        pool.ranges.push_back({static_cast<size_t>(segment.physical_offset),
                               static_cast<size_t>(segment.size)});
        pool.block_extent = segment.physical_offset + segment.size;
        aligned &= segment.physical_offset % format->element_bytes == 0 &&
                   segment.size % format->element_bytes == 0;
      }
      pool.block_ids.assign(group.dst_device_block_ids().begin(),
                             group.dst_device_block_ids().end());
      for (int64_t extent : group.dst_expected_extent_bytes()) {
        aligned &= extent % format->element_bytes == 0;
        pool.live_bytes.push_back(static_cast<size_t>(extent));
      }
      if (!aligned) {
        LOG(WARNING) << "Receive FA KV NaN check skipped: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " live ranges split numeric elements";
        continue;
      }
      check->pools_.emplace(pool_idx, std::move(pool));
    }
  }
  return check->pools_.empty() ? nullptr : std::move(check);
}

void RecvNanCheck::BeforeH2d(size_t pool_idx) const {
  const auto it = pools_.find(pool_idx);
  if (it == pools_.end()) return;
  const Pool& pool = it->second;
  const auto& spec = *base_->pool(pool_idx);
  auto addrs = base_->PoolHostBaseAddrs(uuid_, pool_idx);
  if (!addrs.ok()) {
    LOG(WARNING) << "Receive FA KV NaN check unavailable: req_id=" << req_id_
                 << " pool=" << pool_idx << " addresses=" << addrs.status();
    return;
  }
  for (size_t ordinal = 0; ordinal < pool.block_ids.size(); ++ordinal) {
    const int64_t block_id = pool.block_ids[ordinal];
    const auto slot_it = addrs->host_slot_by_block().find(block_id);
    assert((addrs->host_slot_by_block().empty() ||
            slot_it != addrs->host_slot_by_block().end()) &&
           "received FA block must retain its staging lease until NaN scan ends");
    const int64_t slot = addrs->host_slot_by_block().empty()
                             ? block_id
                             : slot_it->second;
    for (int shard = 0; shard < addrs->host_base_addrs_size(); ++shard) {
      const auto* ptr = reinterpret_cast<const std::byte*>(
          addrs->host_base_addrs(shard) + slot * addrs->block_stride_bytes());
      if (auto offset = FindNanInLivePrefix(
              {ptr, pool.block_extent}, pool.ranges, pool.live_bytes[ordinal],
              pool.format)) {
        LOG(ERROR) << "NaN detected in received full-attention KV cache: req_id="
                   << req_id_ << " uuid=" << uuid_ << " pool=" << pool_idx
                   << " tag=" << spec.tag
                   << " block_id=" << block_id << " shard=" << shard
                   << " byte_offset=" << *offset << " source=host_staging";
        return;
      }
    }
  }
}

}  // namespace tpu_raiden

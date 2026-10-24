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

#include "tpu_sync/core/raiden_manager_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "peregrine/src/api/v1/rate_limiter.h"
#include "xla/future.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/fault_injection/hooks.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"

namespace tpu_raiden {

xla::Future<> ReturnFuture(const absl::Status& status) {
  return xla::Future<>(status);
}

RaidenManagerBase::RaidenManagerBase(
    size_t num_layers, std::vector<transport::ShardInfo> shards,
    size_t slice_byte_size, std::optional<int> local_port, int parallelism,
    std::optional<std::string> bind_ip, std::vector<int> numa_nodes,
    std::vector<HostNicAddress> host_nics, bool socket_only_transport)
    : num_layers_(num_layers),
      num_shards_(shards.size()),
      slice_byte_size_(slice_byte_size),
      parallelism_(parallelism),
      shards_(std::move(shards)),
      local_port_cfg_(local_port.value_or(0)),
      bind_ip_cfg_(std::move(bind_ip)),
      socket_only_transport_(socket_only_transport) {
  shard_factor_ = 1;
  (void)telemetry::RaidenMetricStore::GetGlobalMetricStore();
  const bool has_shard_placement = absl::c_any_of(
      shards_, [](const transport::ShardInfo& s) { return s.numa_node >= 0; });
  if (numa_nodes.empty()) {
    // Derive the distinct node list from the per-shard placement so the two
    // inputs stay interchangeable.
    for (const transport::ShardInfo& shard : shards_) {
      if (shard.numa_node >= 0 &&
          !absl::c_linear_search(numa_nodes, shard.numa_node)) {
        numa_nodes.push_back(shard.numa_node);
      }
    }
  }
  if (!numa_nodes.empty()) {
    assigned_numa_node_ = numa_nodes[0];
    if (numa_nodes.size() > 1) {
      if (!has_shard_placement) {
        LOG(WARNING) << "Incoming PJRT buffers are associated with more than "
                        "one NUMA node ("
                     << numa_nodes[0] << " vs " << numa_nodes[1]
                     << ") and no per-shard placement was supplied. Picking "
                        "the first detected NUMA node: "
                     << numa_nodes[0];
      } else {
        std::vector<int> shard_numa_nodes;
        shard_numa_nodes.reserve(shards_.size());
        for (const transport::ShardInfo& shard : shards_) {
          shard_numa_nodes.push_back(shard.numa_node);
        }
        LOG(INFO) << "Incoming PJRT buffers span NUMA nodes ["
                  << absl::StrJoin(numa_nodes, ",")
                  << "]; per-shard NUMA map: ["
                  << absl::StrJoin(shard_numa_nodes, ",") << "]";
      }
    }
  }
  if (!has_shard_placement) {
    // No placement supplied: every shard sits on the assigned node (or is
    // unknown) with an unknown global index.
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      shards_[sh].numa_node = assigned_numa_node_.value_or(-1);
    }
  }
  // `local_index` is the position in `shards_` by definition.
  for (size_t sh = 0; sh < shards_.size(); ++sh) {
    shards_[sh].local_index = static_cast<int>(sh);
  }
  InitTransportServer(std::move(host_nics));
}

RaidenManagerBase::~RaidenManagerBase() {
  StopTransportServer();
}

void RaidenManagerBase::StopTransportServer() {
  absl::MutexLock lock(server_init_mu_);
  if (server_) {
    server_.reset();
  }
}

xla::Future<> RaidenManagerBase::ForEachShardNumaLocal(
    tpu_raiden::NumaThreadPool& pool,
    absl::AnyInvocable<xla::Future<>(size_t shard_idx) const> fn) {
  std::shared_ptr<const absl::AnyInvocable<xla::Future<>(size_t) const>>
      shared_fn = std::make_shared<
          const absl::AnyInvocable<xla::Future<>(size_t) const>>(std::move(fn));
  std::vector<xla::Future<>> shard_futures;
  shard_futures.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    auto [promise, future] = xla::MakePromise<>();
    shard_futures.push_back(std::move(future));
    pool.Schedule(
        shard_numa_node(i),
        [shared_fn, i, promise = std::move(promise).ToShared()]() {
          xla::Future<> result = (*shared_fn)(i);
          if (!result.IsValid()) {
            promise->Set(absl::InternalError(
                "ForEachShardNumaLocal: shard task returned no future"));
            return;
          }
          std::move(result).OnReady(
              [promise](const absl::Status& status) { promise->Set(status); });
        });
  }
  return xla::JoinFutures(absl::MakeSpan(shard_futures));
}

void RaidenManagerBase::SetTestOnlyRateLimiters(
    std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> egress,
    std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> ingress) {
  server_->SetTestOnlyRateLimiters(std::move(egress), std::move(ingress));
}

tpu_raiden::transport::BlockTransport* RaidenManagerBase::InitTransportServer(
    std::vector<HostNicAddress> host_nics) {
  absl::MutexLock lock(server_init_mu_);
  if (server_) return server_.get();

  ip_numa_nodes_.clear();
  for (const auto& nic : host_nics) {
    if (nic.numa_node >= 0) {
      ip_numa_nodes_.emplace(nic.ip_address, nic.numa_node);
    }
  }

  std::vector<std::string> collected_ips;
  if (bind_ip_cfg_.has_value() && !bind_ip_cfg_->empty()) {
    collected_ips = {*bind_ip_cfg_};
  } else {
    std::vector<HostNicAddress> data_nics;
    std::vector<HostNicAddress> ctrl_nics;
    for (const auto& nic : host_nics) {
      if (nic.classification == NicClassification::kDataPlane) {
        data_nics.push_back(nic);
      } else if (nic.classification == NicClassification::kControlPlane) {
        ctrl_nics.push_back(nic);
      }
    }
    if (!data_nics.empty()) {
      host_nics = std::move(data_nics);
    } else if (!ctrl_nics.empty()) {
      host_nics = std::move(ctrl_nics);
    }

    if (!host_nics.empty()) {
      // NUMA nodes to serve, in first-seen shard order. Without a per-shard
      // map this degenerates to the single assigned node.
      std::vector<int> target_numas;
      for (const transport::ShardInfo& shard : shards_) {
        if (shard.numa_node >= 0 &&
            !absl::c_linear_search(target_numas, shard.numa_node)) {
          target_numas.push_back(shard.numa_node);
        }
      }
      if (target_numas.empty() && assigned_numa_node_.has_value() &&
          *assigned_numa_node_ >= 0) {
        target_numas.push_back(*assigned_numa_node_);
      }
      std::cerr << "InitTransportServer: target_numa=["
                << absl::StrJoin(target_numas, ",") << "]" << std::endl;

      auto collect = [&](bool data_plane_only) {
        for (int target_numa : target_numas) {
          for (const auto& nic : host_nics) {
            if (nic.numa_node != target_numa) continue;
            if (data_plane_only &&
                nic.classification != NicClassification::kDataPlane) {
              continue;
            }
            if (!absl::c_linear_search(collected_ips, nic.ip_address)) {
              collected_ips.push_back(nic.ip_address);
            }
          }
        }
      };

      // 1. Collect the Data NICs local to every shard NUMA node.
      collect(/*data_plane_only=*/true);

      // 2. Fallback: Collect all NUMA-local NICs.
      if (collected_ips.empty()) {
        collect(/*data_plane_only=*/false);
      }

      // 3. Ultimate Fallback: Use the first NIC on the host
      if (collected_ips.empty()) {
        collected_ips.push_back(host_nics[0].ip_address);
      }
    }
  }

  if (collected_ips.empty()) {
    collected_ips.push_back("127.0.0.1");
  }

  local_ips_ = std::move(collected_ips);

  for (const auto& ip : local_ips_) {
    std::cerr << "InitTransportServer: Local IP: " << ip
              << " numa=" << numa_node_for_ip(ip).value_or(-1) << std::endl;
  }

  server_ = std::make_unique<tpu_raiden::transport::BlockTransport>(
      this, local_port_cfg_, local_ips_, parallelism_, socket_only_transport_);
  return server_.get();
}

std::optional<int> RaidenManagerBase::local_port() const {
  return server_->local_port();
}

std::string RaidenManagerBase::local_ip() const { return server_->bound_ip(); }

std::vector<std::string> RaidenManagerBase::local_ips() const {
  if (local_ips_.empty()) {
    return {server_->bound_ip()};
  }
  return local_ips_;
}

std::optional<int> RaidenManagerBase::shard_numa_node(size_t shard_idx) const {
  if (shard_idx < shards_.size()) {
    const int node = shards_[shard_idx].numa_node;
    if (node >= 0) return node;
    return std::nullopt;
  }
  return assigned_numa_node_;
}

std::optional<int> RaidenManagerBase::numa_node_for_ip(
    absl::string_view ip) const {
  auto it = ip_numa_nodes_.find(ip);
  if (it == ip_numa_nodes_.end()) return std::nullopt;
  return it->second;
}

std::vector<std::string> RaidenManagerBase::local_ips_for_shard(
    size_t shard_idx) const {
  const std::vector<std::string> all_ips = local_ips();
  const std::optional<int> numa = shard_numa_node(shard_idx);
  if (!numa.has_value()) return all_ips;
  std::vector<std::string> ips;
  for (const std::string& ip : all_ips) {
    if (numa_node_for_ip(ip) == numa) ips.push_back(ip);
  }
  return ips.empty() ? all_ips : ips;
}

std::vector<int64_t> RaidenManagerBase::shards_for_local_ip(
    absl::string_view ip) const {
  std::vector<int64_t> all_shards(num_shards_);
  std::iota(all_shards.begin(), all_shards.end(), 0);
  const std::optional<int> numa = numa_node_for_ip(ip);
  if (!numa.has_value()) return all_shards;
  // A shard belongs to |ip| when it is NUMA-local to it, or when no local IP
  // is NUMA-local to the shard at all (e.g. one NIC serving shards on two
  // nodes): every shard must be advertised by at least one endpoint, so such
  // shards are claimed by every local IP (mirroring local_ips_for_shard).
  const std::vector<std::string> all_ips = local_ips();
  if (!absl::c_linear_search(all_ips, ip)) return all_shards;
  auto has_local_ip = [&](std::optional<int> shard_numa) {
    return absl::c_any_of(all_ips, [&](const std::string& local_ip) {
      return numa_node_for_ip(local_ip) == shard_numa;
    });
  };
  std::vector<int64_t> shards;
  for (size_t sh = 0; sh < num_shards_; ++sh) {
    const std::optional<int> shard_numa = shard_numa_node(sh);
    if (shard_numa == numa || !shard_numa.has_value() ||
        !has_local_ip(shard_numa)) {
      shards.push_back(static_cast<int64_t>(sh));
    }
  }
  return shards.empty() ? all_shards : shards;
}

// Resolves host memory pointer for a specific layer and shard.
// In multi-host distributed execution, shard_idx may represent the global shard
// ID across multiple worker nodes. Modulo indexing (`shard_idx %
// shards.size()`) ensures clean resolution to the local worker's assigned shard
// buffers.
// TODO: It might be clearer if the base manager doesn't have to
// deal with global shard idx.
uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return nullptr;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  return const_cast<uint8_t*>(layers_[layer_idx].shards[local_idx].host_ptr);
}

// Resolves host memory allocation size in bytes for a specific layer and shard
// using multi-host modulo indexing.
size_t RaidenManagerBase::GetHostSize(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return 0;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  return layers_[layer_idx].shards[local_idx].host_size;
}

size_t RaidenManagerBase::GetHostSize(size_t layer_idx,
                                      size_t shard_idx) const {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return 0;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  return layers_[layer_idx].shards[local_idx].host_size;
}

// Const overload resolving host memory pointer for a specific layer and shard
// using multi-host modulo indexing.
const uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx,
                                                 size_t shard_idx) const {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return nullptr;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  return layers_[layer_idx].shards[local_idx].host_ptr;
}

void RaidenManagerBase::SetExternalHostPointers(
    const std::vector<const uint8_t*>& host_ptrs,
    const std::vector<size_t>& host_sizes) {
  size_t idx = 0;
  for (size_t l = 0; l < layers_.size(); ++l) {
    for (size_t sh = 0; sh < layers_[l].shards.size(); ++sh) {
      if (idx < host_ptrs.size() && idx < host_sizes.size()) {
        layers_[l].shards[sh].host_ptr = host_ptrs[idx];
        layers_[l].shards[sh].host_size = host_sizes[idx];
        idx++;
      }
    }
  }
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hWriteDirect(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx) {
  RAIDEN_TRACE_FN("RaidenBase::H2hWriteDirect", [&]() {
    return absl::StrCat("blocks=", src_block_ids.size(),
                        " peers=", peers.size(), " uuid=", uuid);
  });
  return server_
      ->AsyncPush(peers, src_block_ids, dst_block_ids, parallelism_,
                  tpu_raiden::transport::MajorOrder::kLayerMajor, uuid,
                  layer_idx)
      .Await();
}

void RaidenManagerBase::H2hWriteDirectAsync(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx,
    std::function<void(absl::StatusOr<std::vector<int>>)> on_complete) {
  RAIDEN_TRACE_FN("RaidenBase::H2hWriteDirectAsync", [&]() {
    return absl::StrCat("blocks=", src_block_ids.size(),
                        " peers=", peers.size(), " uuid=", uuid);
  });
  absl::Status status = FaultInjectStatus(hooks::kRaidenManagerBaseH2hWrite);
  if (!status.ok()) {
    on_complete(status);
    return;
  }
  server_
      ->AsyncPush(peers, src_block_ids, dst_block_ids, parallelism_,
                  tpu_raiden::transport::MajorOrder::kLayerMajor, uuid,
                  layer_idx)
      .OnReady(std::move(on_complete));
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hReadDirect(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids) {
  RAIDEN_TRACE_FN("RaidenBase::H2hReadDirect", [&]() {
    return absl::StrCat("blocks=", src_block_ids.size(),
                        " peers=", peers.size());
  });
  return server_->SyncPull(peers, src_block_ids, {}, {}, parallelism_);
}

absl::Status RaidenManagerBase::PushWeightsChunk(
    absl::string_view peer, size_t dst_shard_idx, size_t dst_offset_bytes,
    const uint8_t* data_ptr, size_t size_bytes, uint64_t uuid,
    size_t layer_idx) {
  RAIDEN_TRACE_FN("RaidenBase::PushWeightsChunk", [&]() {
    return absl::StrCat("peer=", peer, " layer=", layer_idx,
                        " shard=", dst_shard_idx, " bytes=", size_bytes);
  });
  return server_->PushBuffer(peer, /*buffer_id=*/layer_idx, dst_shard_idx,
                             dst_offset_bytes, data_ptr, size_bytes, uuid);
}

absl::Status RaidenManagerBase::PullBuffer(
    absl::string_view peer, size_t buffer_id, size_t src_shard_idx,
    size_t src_offset_bytes, size_t dst_shard_idx, size_t dst_offset_bytes,
    size_t size_bytes) {
  RAIDEN_TRACE_FN("RaidenBase::PullBuffer", [&]() {
    return absl::StrCat("peer=", peer, " layer=", buffer_id,
                        " src_shard=", src_shard_idx,
                        " dst_shard=", dst_shard_idx, " bytes=", size_bytes);
  });
  return server_->PullBuffer(peer, buffer_id, src_shard_idx, src_offset_bytes,
                             dst_shard_idx, dst_offset_bytes, size_bytes);
}

absl::Status RaidenManagerBase::PushWeightsChunks(
    const std::vector<transport::BufferPushTask>& tasks, int parallelism,
    uint64_t uuid) {
  RAIDEN_TRACE_FN("RaidenBase::PushWeightsChunks", [&]() {
    return absl::StrCat("tasks=", tasks.size(), " uuid=", uuid);
  });
  return server_->PushBuffers(tasks, parallelism, uuid);
}

absl::Status RaidenManagerBase::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  return server_->RegisterExpectedChunks(uuid, expected_chunks);
}

absl::Status RaidenManagerBase::RegisterExpectedLayerChunks(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  return server_->RegisterExpectedLayerChunks(uuid, expected_layer_chunks);
}

void RaidenManagerBase::ForgetPushProgress(uint64_t uuid) {
  server_->ForgetPushProgress(uuid);
}

size_t RaidenManagerBase::block_bytes(size_t layer_idx) const {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return slice_byte_size_;
  }
  size_t dev_size = layers_[layer_idx].shards[0].device_size;
  return dev_size > 0 ? dev_size : slice_byte_size_;
}

}  // namespace tpu_raiden

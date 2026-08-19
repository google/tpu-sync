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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/buffer_push_task.h"

namespace tpu_raiden {

xla::Future<> ReturnFuture(const absl::Status& status) {
  return xla::Future<>(status);
}

void RaidenManagerBase::DetectAndAssignNumaNode(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers) {
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (const auto& buf : layer) {
      if (buf.device) {
        int node = GetPjRtDeviceNumaNode(buf.device);
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
  if (!unique_numa_nodes.empty()) {
    assigned_numa_node_ = unique_numa_nodes[0];
    if (unique_numa_nodes.size() > 1) {
      LOG(WARNING) << "Incoming PJRT buffers are associated with more than one "
                      "NUMA node ("
                   << unique_numa_nodes[0] << " vs " << unique_numa_nodes[1]
                   << "). Picking the first detected NUMA node: "
                   << unique_numa_nodes[0];
    }
  }
  InitTransportServer();
}

RaidenManagerBase::RaidenManagerBase(size_t num_layers, size_t num_shards,
                                     size_t slice_byte_size,
                                     std::optional<int> local_port,
                                     int parallelism,
                                     std::optional<std::string> bind_ip)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      parallelism_(parallelism),
      local_port_cfg_(local_port.value_or(0)),
      bind_ip_cfg_(bind_ip) {
  shard_factor_ = 1;
}

RaidenManagerBase::~RaidenManagerBase() {
  std::shared_ptr<tpu_raiden::transport::BlockTransport> transport;
  {
    absl::MutexLock lock(server_init_mu_);
    transport = std::move(server_);
  }
  if (transport) {
    transport->CancelPendingOperations();
    if (!transport->WaitForPendingOperations(std::chrono::seconds(30))) {
      LOG(FATAL) << "Transport operations did not drain before base manager "
                    "destruction";
    }
  }
}

std::vector<HostNicAddress> RaidenManagerBase::GetHostNics() const {
  return GetLocalHostNicAddresses();
}

std::shared_ptr<tpu_raiden::transport::BlockTransport>
RaidenManagerBase::InitTransportServer() {
  absl::MutexLock lock(server_init_mu_);
  if (server_) return server_;

  std::vector<std::string> collected_ips;
  if (bind_ip_cfg_.has_value() && !bind_ip_cfg_->empty()) {
    collected_ips = {*bind_ip_cfg_};
  } else {
    std::vector<HostNicAddress> host_nics = GetHostNics();
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
      int target_numa = assigned_numa_node_.value_or(-1);
      std::cerr << "InitTransportServer: target_numa=" << target_numa
                << std::endl;

      // 1. Collect all NUMA-local Data NICs
      if (target_numa >= 0) {
        for (const auto& nic : host_nics) {
          if (nic.numa_node == target_numa &&
              nic.classification == NicClassification::kDataPlane) {
            collected_ips.push_back(nic.ip_address);
          }
        }
      }

      // 2. Fallback: Collect all NUMA-local NICs
      if (collected_ips.empty() && target_numa >= 0) {
        for (const auto& nic : host_nics) {
          if (nic.numa_node == target_numa) {
            collected_ips.push_back(nic.ip_address);
          }
        }
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
    std::cerr << "InitTransportServer: Local IP: " << ip << std::endl;
  }

  server_ = std::make_shared<tpu_raiden::transport::BlockTransport>(
      this, local_port_cfg_, local_ips_, parallelism_);
  return server_;
}

std::shared_ptr<tpu_raiden::transport::BlockTransport>
RaidenManagerBase::GetTransportServer() const {
  absl::MutexLock lock(server_init_mu_);
  return server_;
}

void RaidenManagerBase::CancelTransportOperations() {
  std::shared_ptr<tpu_raiden::transport::BlockTransport> transport =
      GetTransportServer();
  if (!transport) return;
  transport->CancelPendingOperations();
  if (!transport->WaitForPendingOperations(std::chrono::seconds(30))) {
    LOG(FATAL) << "Transport operations did not drain while the derived "
                  "manager was alive";
  }
}

std::shared_ptr<void> RaidenManagerBase::TrackManagerCallback() {
  {
    std::lock_guard<std::mutex> lock(manager_callbacks_mu_);
    ++active_manager_callbacks_;
  }
  return std::shared_ptr<void>(this, [this](void*) {
    {
      std::lock_guard<std::mutex> lock(manager_callbacks_mu_);
      if (active_manager_callbacks_ == 0) {
        LOG(FATAL) << "Manager callback tracker underflow";
      }
      --active_manager_callbacks_;
    }
    manager_callbacks_cv_.notify_all();
  });
}

bool RaidenManagerBase::WaitForManagerCallbacks(
    std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(manager_callbacks_mu_);
  return manager_callbacks_cv_.wait_for(
      lock, timeout, [this]() { return active_manager_callbacks_ == 0; });
}

std::optional<int> RaidenManagerBase::local_port() const {
  auto transport = const_cast<RaidenManagerBase*>(this)->InitTransportServer();
  if (transport) return transport->local_port();
  return std::nullopt;
}

std::string RaidenManagerBase::local_ip() const {
  auto transport = const_cast<RaidenManagerBase*>(this)->InitTransportServer();
  if (transport) return transport->bound_ip();
  return "127.0.0.1";
}

std::vector<std::string> RaidenManagerBase::local_ips() const {
  auto transport = const_cast<RaidenManagerBase*>(this)->InitTransportServer();
  if (local_ips_.empty()) {
    return {transport ? transport->bound_ip() : "127.0.0.1"};
  }
  return local_ips_;
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
  auto transport = InitTransportServer();
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return transport->SyncPush(peers, src_block_ids, dst_block_ids, parallelism_,
                             tpu_raiden::transport::MajorOrder::kLayerMajor,
                             uuid, layer_idx);
}

void RaidenManagerBase::H2hWriteDirectAsync(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx,
    std::function<void(absl::StatusOr<std::vector<int>>)> on_complete) {
  auto transport = InitTransportServer();
  if (!transport) {
    on_complete(
        absl::FailedPreconditionError("Transport server is not running"));
    return;
  }
  transport->AsyncPush(peers, src_block_ids, dst_block_ids, parallelism_,
                       tpu_raiden::transport::MajorOrder::kLayerMajor, uuid,
                       layer_idx, std::move(on_complete));
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hReadDirect(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids) {
  auto transport = InitTransportServer();
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return transport->SyncPull(peers, src_block_ids, {}, {}, parallelism_);
}

absl::Status RaidenManagerBase::PushWeightsChunk(
    absl::string_view peer, size_t dst_shard_idx, size_t dst_offset_bytes,
    const uint8_t* data_ptr, size_t size_bytes, uint64_t uuid,
    size_t layer_idx) {
  auto transport = InitTransportServer();
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return transport->PushBuffer(peer, /*buffer_id=*/layer_idx, dst_shard_idx,
                               dst_offset_bytes, data_ptr, size_bytes, uuid);
}

absl::Status RaidenManagerBase::PushWeightsChunks(
    const std::vector<transport::BufferPushTask>& tasks, int parallelism,
    uint64_t uuid) {
  auto transport = InitTransportServer();
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return transport->PushBuffers(tasks, parallelism, uuid);
}

absl::Status RaidenManagerBase::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  auto transport = InitTransportServer();
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return transport->RegisterExpectedChunks(uuid, expected_chunks);
}

absl::Status RaidenManagerBase::RegisterExpectedLayerChunks(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  auto transport = InitTransportServer();
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return transport->RegisterExpectedLayerChunks(uuid, expected_layer_chunks);
}

void RaidenManagerBase::ForgetPushProgress(uint64_t uuid) {
  auto transport = InitTransportServer();
  if (transport) {
    transport->ForgetPushProgress(uuid);
  }
}

size_t RaidenManagerBase::bytes_per_block() const { return slice_byte_size_; }

size_t RaidenManagerBase::block_bytes(size_t layer_idx) const {
  if (layers_.empty() || layer_idx >= layers_.size() ||
      layers_[layer_idx].shards.empty()) {
    return bytes_per_block();
  }
  size_t dev_size = layers_[layer_idx].shards[0].device_size;
  return dev_size > 0 ? dev_size : bytes_per_block();
}

}  // namespace tpu_raiden

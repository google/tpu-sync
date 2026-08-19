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

#include "tpu_sync/frameworks/jax/kv_cache_manager.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/logging.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/metrics_collector.h"  // IWYU pragma: keep
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/core/utils.h"  // IWYU pragma: keep
#ifndef WITHOUT_PYTHON
#include "tpu_sync/frameworks/jax/utils.h"

namespace nb = nanobind;
#endif

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

namespace {
::tpu_raiden::HostBufferAllocator LocalCreateHostMemoryAllocator(
    xla::PjRtClient* client) {
  return [client](size_t size_bytes, const xla::PjRtDevice* device)
             -> absl::StatusOr<::tpu_raiden::HostBufferAllocation> {
    if (size_bytes == 0) {
      ::tpu_raiden::HostBufferAllocation alloc;
      alloc.ptr = nullptr;
      alloc.size = 0;
      return alloc;
    }

    // Allocate page-aligned memory
    size_t aligned_size = (size_bytes + 4095) & ~4095;

    void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      return absl::ResourceExhaustedError(
          "mmap failed in LocalCreateHostMemoryAllocator");
    }

    // Touch memory to force physical page allocation (NUMA first-touch)
    std::thread touch_thread([ptr, aligned_size, device]() {
      if (device != nullptr) {
        int node = ::tpu_raiden::GetPjRtDeviceNumaNode(device);
        if (node >= 0) {
          int rc = ::tpu_raiden::PinCurrentThreadToNumaNode(node);
          LOG(INFO) << "touch_thread: Pinned to NUMA node " << node
                    << ", status=" << rc
                    << ", thread_id=" << std::this_thread::get_id();
        } else {
          LOG(INFO) << "touch_thread: No NUMA node found for device, executing "
                       "without pinning. Thread_id="
                    << std::this_thread::get_id();
        }
      } else {
        LOG(INFO) << "touch_thread: Device is null, executing without pinning. "
                     "Thread_id="
                  << std::this_thread::get_id();
      }
      volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
      for (size_t i = 0; i < aligned_size; i += 4096) {
        p[i] = 0;
      }
    });
    touch_thread.join();

    // Register memory with the TPU device DMA engine
    if (client == nullptr) {
      munmap(ptr, aligned_size);
      return absl::InternalError(
          "PjRtClient is null in LocalCreateHostMemoryAllocator. Cannot "
          "perform DmaMap!");
    }

    bool should_dma_map = (client->platform_name() != "cpu");
    if (should_dma_map) {
      auto status = client->DmaMap(ptr, aligned_size);
      if (!status.ok()) {
        munmap(ptr, aligned_size);
        return absl::InternalError(
            "DmaMap failed in LocalCreateHostMemoryAllocator: " +
            std::string(status.message()));
      }
    }

    ::tpu_raiden::HostBufferAllocation alloc;
    alloc.ptr = static_cast<uint8_t*>(ptr);
    alloc.size = size_bytes;
    alloc.owner = std::shared_ptr<void>(
        ptr, [client, aligned_size, should_dma_map](void* p) {
          if (should_dma_map) {
            (void)client->DmaUnmap(p);
          }
          munmap(p, aligned_size);
        });

    return alloc;
  };
}
}  // namespace

#ifndef WITHOUT_PYTHON
namespace {
UnpackedCache UnpackAndMove(nanobind::list device_arrays,
                            bool unsafe_skip_buffer_lock = false) {
  auto layer_buffers = ::tpu_raiden::jax::UnpackJaxArrays(
      device_arrays, unsafe_skip_buffer_lock);
  return {std::move(layer_buffers), std::move(device_arrays)};
}
}  // namespace

NumaAwareKVCacheManager::NumaAwareKVCacheManager(
    nb::list device_arrays, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, int64_t node_id)
    : NumaAwareKVCacheManager(
          UnpackAndMove(std::move(device_arrays), unsafe_skip_buffer_lock),
          local_port, host_blocks_to_allocate, unsafe_skip_buffer_lock,
          parallelism, node_id) {}

NumaAwareKVCacheManager::NumaAwareKVCacheManager(
    UnpackedCache&& cache, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, int64_t node_id)
    : device_arrays_(std::move(cache.device_arrays)) {
  InitSubManagers(cache.layer_buffers, local_port, host_blocks_to_allocate,
                  unsafe_skip_buffer_lock, parallelism, node_id, -1, 0, 0,
                  120.0);
}

NumaAwareKVCacheManager::NumaAwareKVCacheManager(
    nanobind::list kv_caches, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    bool unsafe_skip_buffer_lock, int parallelism)
    : NumaAwareKVCacheManager(
          UnpackAndMove(std::move(kv_caches), unsafe_skip_buffer_lock), node_id,
          local_control_port, max_blocks, num_slots, timeout_s,
          unsafe_skip_buffer_lock, parallelism) {}

NumaAwareKVCacheManager::NumaAwareKVCacheManager(
    UnpackedCache&& cache, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    bool unsafe_skip_buffer_lock, int parallelism)
    : device_arrays_(std::move(cache.device_arrays)) {
  const char* enable_metrics_env = std::getenv("ENABLE_RAIDEN_METRICS");
  if (enable_metrics_env != nullptr &&
      std::strcmp(enable_metrics_env, "true") == 0) {
    metrics_collector_ = std::make_shared<MetricsCollector>();
  }
  InitSubManagers(cache.layer_buffers, std::nullopt, std::nullopt,
                  unsafe_skip_buffer_lock, parallelism, node_id,
                  local_control_port, max_blocks, num_slots, timeout_s);
}
#endif

NumaAwareKVCacheManager::NumaAwareKVCacheManager(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism) {
  auto sub_mgr = std::make_unique<KVCacheManagerWithTransfer>(
      num_layers, num_shards, slice_byte_size, local_port,
      host_blocks_to_allocate, parallelism);
  sub_managers_.push_back(std::move(sub_mgr));
  total_num_shards_ = num_shards;
  global_shard_to_submanager_.resize(total_num_shards_);
  for (size_t i = 0; i < total_num_shards_; ++i) {
    global_shard_to_submanager_[i] = {0, static_cast<int>(i)};
  }
}

NumaAwareKVCacheManager::NumaAwareKVCacheManager(
    std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers) {
  sub_managers_ = std::move(sub_managers);
  total_num_shards_ = sub_managers_.size();
  global_shard_to_submanager_.resize(total_num_shards_);
  for (size_t i = 0; i < sub_managers_.size(); ++i) {
    global_shard_to_submanager_[i] = {static_cast<int>(i), 0};
  }
}

NumaAwareKVCacheManager::~NumaAwareKVCacheManager() = default;

void NumaAwareKVCacheManager::InitSubManagers(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s) {
  if (layer_buffers.empty()) return;
  size_t num_layers = layer_buffers.size();
  total_num_shards_ = layer_buffers[0].size();
  global_shard_to_submanager_.resize(total_num_shards_);

  std::map<int, std::vector<int>> numa_to_shards;
  for (size_t sh = 0; sh < total_num_shards_; ++sh) {
    int numa = 0;
    if (layer_buffers[0][sh].device != nullptr) {
      numa = GetPjRtDeviceNumaNode(layer_buffers[0][sh].device);
    }
    if (numa < 0) numa = 0;
    numa_to_shards[numa].push_back(static_cast<int>(sh));
  }

  std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses();
  size_t num_ext_nics = 0;
  for (const auto& nic : host_nics) {
    if (nic.classification == NicClassification::kDataPlane) num_ext_nics++;
  }
  // Flag controlling whether multi-NUMA / multi-NIC sub-managers are enabled.
  // By default, we force a single NUMA 0 sub-manager to bypass JAX threading
  // overhead. Set ENABLE_MULTI_NUMA=1 or true to acknowledge multiple NUMA
  // nodes and spawn multiple Sub-Managers.
  bool force_single_numa = true;
  const char* enable_multi_numa_env = std::getenv("ENABLE_MULTI_NUMA");
  if (enable_multi_numa_env != nullptr) {
    if (absl::EqualsIgnoreCase(enable_multi_numa_env, "true") ||
        absl::EqualsIgnoreCase(enable_multi_numa_env, "1")) {
      force_single_numa = false;
    }
  }

  if (force_single_numa) {
    std::vector<int> all_shards;
    all_shards.reserve(total_num_shards_);
    for (int i = 0; i < static_cast<int>(total_num_shards_); ++i) {
      all_shards.push_back(i);
    }
    numa_to_shards.clear();
    numa_to_shards[0] = std::move(all_shards);
  } else if (numa_to_shards.size() == 1 && num_ext_nics > 1 &&
             total_num_shards_ > 1) {
    // If PJRT reports all devices on NUMA 0, but multiple external NICs exist,
    // distribute shards round-robin across NUMA nodes matching external NIC
    // count.
    numa_to_shards.clear();
    for (size_t sh = 0; sh < total_num_shards_; ++sh) {
      int target_numa = static_cast<int>(sh % num_ext_nics);
      numa_to_shards[target_numa].push_back(static_cast<int>(sh));
    }
  }

  // Allocate the per-NUMA sub-managers. With an ephemeral data-plane port
  // (local_port == 0), the first sub-manager binds a kernel-assigned base port
  // P and the rest bind the consecutive ports P+1, P+2, ... so a peer can
  // address every shard from a single "ip:base_port". Those consecutive ports
  // are bound explicitly and can collide with a port already held by another
  // manager alive in this process. If that happens, tear down the partially
  // built managers and retry the whole allocation from a fresh ephemeral base.
  const bool ephemeral_data_port = (local_port.value_or(-1) == 0);
  const int kMaxPortAttempts = ephemeral_data_port ? 64 : 1;
  for (int attempt = 0; attempt < kMaxPortAttempts; ++attempt) {
    sub_managers_.clear();
    submanager_to_global_shards_.clear();
    submanager_to_global_shards_.reserve(numa_to_shards.size());
    std::optional<int> bound_base_port = std::nullopt;
    bool bind_conflict = false;
    for (const auto& [numa, shards] : numa_to_shards) {
      int sub_idx = static_cast<int>(sub_managers_.size());
      std::vector<int64_t> gshards;
      for (size_t local_sh = 0; local_sh < shards.size(); ++local_sh) {
        global_shard_to_submanager_[shards[local_sh]] = {
            sub_idx, static_cast<int>(local_sh)};
        gshards.push_back(shards[local_sh]);
      }
      submanager_to_global_shards_.push_back(std::move(gshards));

      std::vector<std::vector<raiden::RaidenBufferHandle>> sub_buffers(
          num_layers);
      for (size_t l = 0; l < num_layers; ++l) {
        sub_buffers[l].reserve(shards.size());
        for (int sh : shards) {
          sub_buffers[l].push_back(layer_buffers[l][sh]);
        }
      }

      xla::PjRtClient* client = nullptr;
      if (!sub_buffers.empty() && !sub_buffers[0].empty()) {
        if (sub_buffers[0][0].device != nullptr) {
          client = sub_buffers[0][0].device->client();
        }
      }
      ::tpu_raiden::HostBufferAllocator host_alloc;
      const char* shm_key_env = std::getenv("RAIDEN_SHM_KEY");
      if (shm_key_env != nullptr && std::strlen(shm_key_env) > 0) {
        host_alloc = ::tpu_raiden::CreateHostMemoryAllocator(
            client, max_blocks,
            (sub_buffers.empty() || sub_buffers[0].empty())
                ? 0
                : sub_buffers[0][0].GetOnDeviceSizeInBytes());
      } else {
        host_alloc = LocalCreateHostMemoryAllocator(client);
      }

      // Bind sub-managers on CONSECUTIVE data-plane ports so a peer can address
      // every sub-manager from a single advertised base ("ip:base_port" +
      // sub_idx). This must hold even for an ephemeral base (local_port ==
      // nullopt/0): once sub-manager 0 binds a kernel-assigned base P (captured
      // into bound_base_port below), the rest must bind P+1, P+2, ...
      // explicitly — otherwise each gets a random ephemeral port and the
      // sender's base+idx assumption fails with "connection refused"
      // (multi-NUMA read_remote).
      // TOOD(jcgu): we will remove the sender's assumption soon.
      std::optional<int> sub_port = local_port;
      if (bound_base_port.has_value()) {
        sub_port = *bound_base_port + sub_idx;
      } else if (sub_port.has_value() && *sub_port > 0) {
        sub_port = *sub_port + sub_idx;
      }
      int64_t sub_ctrl_port = local_control_port > 0
                                  ? local_control_port + sub_idx
                                  : local_control_port;

      std::unique_ptr<KVCacheManagerWithTransfer> sub_mgr;
      try {
        sub_mgr = std::make_unique<KVCacheManagerWithTransfer>(
            sub_buffers, sub_port, host_blocks_to_allocate,
            unsafe_skip_buffer_lock, parallelism, host_alloc, node_id + sub_idx,
            sub_ctrl_port, max_blocks, num_slots, timeout_s);
        if (metrics_collector_) {
          sub_mgr->SetMetricsCollector(metrics_collector_);
        }
      } catch (const std::exception& e) {
        // A consecutive (or requested) port was unavailable. Retry the whole
        // allocation from a fresh ephemeral base; for fixed ports, propagate.
        if (!ephemeral_data_port || attempt + 1 >= kMaxPortAttempts) throw;
        bind_conflict = true;
        break;
      }
      if (local_port.has_value()) {
        (void)sub_mgr->local_port();
      }
      if (!bound_base_port.has_value() && sub_mgr->local_port().has_value()) {
        bound_base_port = sub_mgr->local_port().value();
      }

      sub_managers_.push_back(std::move(sub_mgr));
    }
    if (!bind_conflict) break;
  }
}

size_t NumaAwareKVCacheManager::num_layers() const {
  return sub_managers_.empty() ? 0 : sub_managers_[0]->num_layers();
}

size_t NumaAwareKVCacheManager::num_shards() const { return total_num_shards_; }

size_t NumaAwareKVCacheManager::slice_byte_size() const {
  return sub_managers_.empty() ? 0 : sub_managers_[0]->slice_byte_size();
}

size_t NumaAwareKVCacheManager::num_block_arrays() const {
  return sub_managers_.empty() ? 0 : sub_managers_[0]->num_block_arrays();
}

size_t NumaAwareKVCacheManager::block_bytes(size_t block_array_idx) const {
  return sub_managers_.empty() ? 0
                               : sub_managers_[0]->block_bytes(block_array_idx);
}

std::optional<int> NumaAwareKVCacheManager::local_port() const {
  return sub_managers_.empty() ? std::nullopt : sub_managers_[0]->local_port();
}

int NumaAwareKVCacheManager::local_control_port() const {
  return sub_managers_.empty() ? -1 : sub_managers_[0]->local_control_port();
}

int64_t NumaAwareKVCacheManager::node_id() const {
  return sub_managers_.empty() ? 0 : sub_managers_[0]->node_id();
}

uint8_t* NumaAwareKVCacheManager::GetHostPointer(size_t layer_idx,
                                                 size_t shard_idx) {
  if (shard_idx >= global_shard_to_submanager_.size()) return nullptr;
  auto [sub_idx, local_shard] = global_shard_to_submanager_[shard_idx];
  return sub_managers_[sub_idx]->GetHostPointer(layer_idx, local_shard);
}

const uint8_t* NumaAwareKVCacheManager::GetHostPointer(size_t layer_idx,
                                                       size_t shard_idx) const {
  if (shard_idx >= global_shard_to_submanager_.size()) return nullptr;
  auto [sub_idx, local_shard] = global_shard_to_submanager_[shard_idx];
  return sub_managers_[sub_idx]->GetHostPointer(layer_idx, local_shard);
}

size_t NumaAwareKVCacheManager::GetHostSize(size_t layer_idx,
                                            size_t shard_idx) {
  if (shard_idx >= global_shard_to_submanager_.size()) return 0;
  auto [sub_idx, local_shard] = global_shard_to_submanager_[shard_idx];
  return sub_managers_[sub_idx]->GetHostSize(layer_idx, local_shard);
}

int64_t NumaAwareKVCacheManager::NotifyForRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<int64_t>& block_ids) {
  int64_t res = 0;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    res = sub_managers_[s]->NotifyForRead(req_id, uuid, block_ids);
  }
  return res;
}

std::vector<int32_t> NumaAwareKVCacheManager::RenewRemoteLeases(
    const std::string& remote_endpoint, const std::vector<uint64_t>& uuids) {
  if (sub_managers_.empty()) {
    throw std::runtime_error("KVCacheManager has no transfer sub-managers");
  }
  // The endpoint identifies one producer control port. MSL performs any
  // required per-port fanout; fanning this request across local NUMA managers
  // would duplicate every renew/cancel operation.
  return sub_managers_.front()->RenewRemoteLeases(remote_endpoint, uuids);
}

std::vector<int32_t> NumaAwareKVCacheManager::CancelRemoteLeases(
    const std::string& remote_endpoint, const std::vector<uint64_t>& uuids) {
  if (sub_managers_.empty()) {
    throw std::runtime_error("KVCacheManager has no transfer sub-managers");
  }
  return sub_managers_.front()->CancelRemoteLeases(remote_endpoint, uuids);
}

std::vector<RaidenTransferEndpoint>
NumaAwareKVCacheManager::get_local_data_endpoints() const {
  std::vector<RaidenTransferEndpoint> res;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    auto sub_eps = sub_managers_[s]->get_local_data_endpoints();
    if (!sub_eps.empty()) {
      std::vector<int64_t> shards;
      if (s < submanager_to_global_shards_.size()) {
        shards = submanager_to_global_shards_[s];
      }
      for (const auto& sub_ep : sub_eps) {
        res.push_back({sub_ep.endpoint, shards});
      }
    }
  }
  return res;
}

std::vector<RaidenTransferEndpoint>
NumaAwareKVCacheManager::get_local_endpoints() const {
  std::vector<RaidenTransferEndpoint> res;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    auto sub_eps = sub_managers_[s]->get_local_endpoints();
    if (!sub_eps.empty()) {
      std::vector<int64_t> shards;
      if (s < submanager_to_global_shards_.size()) {
        shards = submanager_to_global_shards_[s];
      }
      for (const auto& sub_ep : sub_eps) {
        res.push_back({sub_ep.endpoint, shards});
      }
    }
  }
  return res;
}

void NumaAwareKVCacheManager::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  if (remote_descriptors.empty() || remote_block_ids.empty()) return;
  if (remote_block_ids.size() != local_block_ids.size()) return;
  if (local_host_block_ids.has_value() &&
      local_host_block_ids->size() != remote_block_ids.size())
    return;

  req_expected_counts_[req_id] = static_cast<int>(sub_managers_.size());

  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    absl::flat_hash_set<int64_t> sub_shards;
    if (s < submanager_to_global_shards_.size()) {
      sub_shards.insert(submanager_to_global_shards_[s].begin(),
                        submanager_to_global_shards_[s].end());
    }

    std::vector<RaidenTransferEndpoint> matched_descs;
    for (const auto& desc : remote_descriptors) {
      bool match = false;
      for (int64_t rsh : desc.shards) {
        if (sub_shards.find(rsh) != sub_shards.end()) {
          match = true;
          break;
        }
      }
      if (match) {
        matched_descs.push_back(desc);
      }
    }
    if (matched_descs.empty() && !remote_descriptors.empty()) {
      matched_descs.push_back(remote_descriptors[0]);
    }

    if (matched_descs.empty()) continue;

    sub_managers_[s]->StartRead(req_id, uuid, matched_descs, remote_block_ids,
                                local_block_ids, parallelism,
                                local_host_block_ids);
  }
}

void NumaAwareKVCacheManager::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  if (remote_block_ids.size() != local_block_ids.size()) return;
  if (local_host_block_ids.has_value() &&
      local_host_block_ids->size() != remote_block_ids.size())
    return;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    sub_managers_[s]->StartRead(req_id, uuid, remote_endpoint, remote_block_ids,
                                local_block_ids, parallelism,
                                local_host_block_ids);
  }
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
NumaAwareKVCacheManager::CompleteReadRaw() {
  std::vector<std::string> done_sending, done_recving, failed_recving;
  int num_subs = static_cast<int>(sub_managers_.size());
  if (num_subs == 0) return {};

  for (auto& sub : sub_managers_) {
    auto [s, r, f] = sub->CompleteReadRaw();
    for (const auto& req : s) {
      done_sending_counts_[req]++;
      if (done_sending_counts_[req] == num_subs) {
        done_sending.push_back(req);
        done_sending_counts_.erase(req);
      }
    }
    for (const auto& req : r) {
      auto [cnt_it, inserted] = done_recving_counts_.emplace(req, 0);
      cnt_it->second++;
      int expected = num_subs;
      if (auto exp_it = req_expected_counts_.find(req);
          exp_it != req_expected_counts_.end()) {
        expected = exp_it->second;
      }
      if (cnt_it->second >= expected) {
        done_recving_counts_.erase(cnt_it);
        req_expected_counts_.erase(req);
        if (failed_recving_set_.find(req) == failed_recving_set_.end()) {
          done_recving.push_back(req);
        }
        failed_recving_set_.erase(req);
      }
    }
    for (const auto& req : f) {
      if (failed_recving_set_.insert(req).second) {
        failed_recving.push_back(req);
      }
    }
  }
  return {std::move(done_sending), std::move(done_recving),
          std::move(failed_recving)};
}

absl::StatusOr<raiden::PjRtCopyFuture> NumaAwareKVCacheManager::H2d(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, std::optional<int64_t> slot_idx,
    std::optional<size_t> layer_idx, std::optional<size_t> shard_idx) {
  if (sub_managers_.empty()) {
    return raiden::PjRtCopyFuture();
  }
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  sub_copy_futures.reserve(sub_managers_.size());
  for (auto& sub : sub_managers_) {
    ASSIGN_OR_RETURN(auto f, sub->H2d(src_offsets, dst_offsets, copy_sizes,
                                      slot_idx, layer_idx, shard_idx));
    sub_copy_futures.push_back(std::move(f));
  }
  // Use the event-aware join. On TPU the per-shard copies complete via the
  // PJRT C-API event path (PjRtCopyFuture::FromEvents), which leaves
  // f.future invalid and signals through event_bundles. Joining the raw
  // f.future via xla::JoinFutures would CHECK-fail IsValid() and also drop the
  // event-based completion. JoinPjRtCopyFutures handles both paths.
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
}

absl::StatusOr<raiden::PjRtCopyFuture> NumaAwareKVCacheManager::D2h(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, std::optional<int64_t> slot_idx,
    std::optional<size_t> layer_idx, std::optional<size_t> shard_idx) {
  if (sub_managers_.empty()) {
    return raiden::PjRtCopyFuture();
  }
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  sub_copy_futures.reserve(sub_managers_.size());
  for (auto& sub : sub_managers_) {
    ASSIGN_OR_RETURN(auto f, sub->D2h(src_offsets, dst_offsets, copy_sizes,
                                      slot_idx, layer_idx, shard_idx));
    sub_copy_futures.push_back(std::move(f));
  }
  // Use the event-aware join (see H2d above): on TPU the per-shard copies
  // complete via the PJRT C-API event path, leaving f.future invalid and
  // signalling through event_bundles, so xla::JoinFutures on the raw future
  // would CHECK-fail IsValid().
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
NumaAwareKVCacheManager::D2hAutoAllocate(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& copy_sizes) {
  if (sub_managers_.empty()) {
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }
  std::vector<int> all_ids;
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    ASSIGN_OR_RETURN(
        auto res, sub_managers_[s]->D2hAutoAllocate(src_offsets, copy_sizes));
    if (s == 0) {
      all_ids = res.first;
    }
    sub_copy_futures.push_back(std::move(res.second));
  }
  // Event-aware join (see D2h/H2d): on TPU the per-shard copies use the PJRT
  // C-API event path, so f.future is invalid and completion is via
  // event_bundles; xla::JoinFutures on the raw future would CHECK-fail.
  raiden::PjRtCopyFuture composite =
      raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
  return std::make_pair(std::move(all_ids), std::move(composite));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
NumaAwareKVCacheManager::H2hWrite(std::string peer,
                                  const std::vector<int>& src_block_ids,
                                  const std::vector<int>& dst_block_ids,
                                  uint64_t uuid, int layer_idx) {
  if (sub_managers_.empty()) {
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }
  std::string host_prefix;
  int base_port = -1;
  size_t colon = peer.find_last_of(':');
  if (colon != std::string::npos) {
    host_prefix = peer.substr(0, colon + 1);
    try {
      base_port = std::stoi(peer.substr(colon + 1));
    } catch (...) {
      base_port = -1;
    }
  }

  std::vector<int> all_ids;
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    std::string sub_peer =
        (base_port >= 0) ? host_prefix + std::to_string(base_port + s) : peer;
    ASSIGN_OR_RETURN(
        auto res, sub_managers_[s]->H2hWrite(sub_peer, src_block_ids,
                                             dst_block_ids, uuid, layer_idx));
    if (s == 0) {
      all_ids = res.first;
    }
    sub_copy_futures.push_back(std::move(res.second));
  }
  // Event-aware join (see D2h/H2d): on TPU the per-shard transfers complete via
  // the PJRT C-API event path, leaving f.future invalid and signalling through
  // event_bundles; xla::JoinFutures on the raw future would CHECK-fail.
  raiden::PjRtCopyFuture composite =
      raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
  return std::make_pair(std::move(all_ids), std::move(composite));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
NumaAwareKVCacheManager::H2hRead(std::string peer,
                                 const std::vector<int>& src_block_ids) {
  if (sub_managers_.empty()) {
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }
  std::string host_prefix;
  int base_port = -1;
  size_t colon = peer.find_last_of(':');
  if (colon != std::string::npos) {
    host_prefix = peer.substr(0, colon + 1);
    try {
      base_port = std::stoi(peer.substr(colon + 1));
    } catch (...) {
      base_port = -1;
    }
  }

  std::vector<int> all_ids;
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    std::string sub_peer =
        (base_port >= 0) ? host_prefix + std::to_string(base_port + s) : peer;
    ASSIGN_OR_RETURN(auto res,
                     sub_managers_[s]->H2hRead(sub_peer, src_block_ids));
    if (s == 0) {
      all_ids = res.first;
    }
    sub_copy_futures.push_back(std::move(res.second));
  }
  // Event-aware join (see D2h/H2d): on TPU the per-shard transfers complete via
  // the PJRT C-API event path, leaving f.future invalid and signalling through
  // event_bundles; xla::JoinFutures on the raw future would CHECK-fail.
  raiden::PjRtCopyFuture composite =
      raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
  return std::make_pair(std::move(all_ids), std::move(composite));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
NumaAwareKVCacheManager::H2hWrite(
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx) {
  if (sub_managers_.empty()) {
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }
  std::vector<int> all_ids;
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;

  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    const auto& sub_shards = submanager_to_global_shards_[s];
    std::string matched_ep;
    for (const auto& desc : remote_descriptors) {
      for (int64_t gsh : sub_shards) {
        if (std::find(desc.shards.begin(), desc.shards.end(), gsh) !=
            desc.shards.end()) {
          matched_ep = desc.endpoint;
          break;
        }
      }
      if (!matched_ep.empty()) break;
    }
    if (matched_ep.empty() && !remote_descriptors.empty()) {
      matched_ep = remote_descriptors[0].endpoint;
    }

    ASSIGN_OR_RETURN(
        auto res, sub_managers_[s]->H2hWrite(matched_ep, src_block_ids,
                                             dst_block_ids, uuid, layer_idx));
    if (s == 0) {
      all_ids = res.first;
    }
    sub_copy_futures.push_back(std::move(res.second));
  }
  // Event-aware join (see D2h/H2d): on TPU the per-shard transfers complete via
  // the PJRT C-API event path, leaving f.future invalid and signalling through
  // event_bundles; xla::JoinFutures on the raw future would CHECK-fail.
  raiden::PjRtCopyFuture composite =
      raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
  return std::make_pair(std::move(all_ids), std::move(composite));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
NumaAwareKVCacheManager::H2hRead(
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int>& src_block_ids) {
  if (sub_managers_.empty()) {
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }
  std::vector<int> all_ids;
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;

  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    const auto& sub_shards = submanager_to_global_shards_[s];
    std::string matched_ep;
    for (const auto& desc : remote_descriptors) {
      for (int64_t gsh : sub_shards) {
        if (std::find(desc.shards.begin(), desc.shards.end(), gsh) !=
            desc.shards.end()) {
          matched_ep = desc.endpoint;
          break;
        }
      }
      if (!matched_ep.empty()) break;
    }
    if (matched_ep.empty() && !remote_descriptors.empty()) {
      matched_ep = remote_descriptors[0].endpoint;
    }

    ASSIGN_OR_RETURN(auto res,
                     sub_managers_[s]->H2hRead(matched_ep, src_block_ids));
    if (s == 0) {
      all_ids = res.first;
    }
    sub_copy_futures.push_back(std::move(res.second));
  }
  // Event-aware join (see D2h/H2d): on TPU the per-shard transfers complete via
  // the PJRT C-API event path, leaving f.future invalid and signalling through
  // event_bundles; xla::JoinFutures on the raw future would CHECK-fail.
  raiden::PjRtCopyFuture composite =
      raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
  return std::make_pair(std::move(all_ids), std::move(composite));
}

absl::StatusOr<raiden::PjRtCopyFuture> NumaAwareKVCacheManager::H2hReadExplicit(
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids) {
  if (sub_managers_.empty()) {
    return raiden::PjRtCopyFuture();
  }
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  sub_copy_futures.reserve(sub_managers_.size());

  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    // Same endpoint matching as the vector H2hRead/H2hWrite/H2dRead above.
    const auto& sub_shards = submanager_to_global_shards_[s];
    std::string matched_ep;
    for (const auto& desc : remote_descriptors) {
      for (int64_t gsh : sub_shards) {
        if (std::find(desc.shards.begin(), desc.shards.end(), gsh) !=
            desc.shards.end()) {
          matched_ep = desc.endpoint;
          break;
        }
      }
      if (!matched_ep.empty()) break;
    }
    if (matched_ep.empty() && !remote_descriptors.empty()) {
      matched_ep = remote_descriptors[0].endpoint;
    }

    ASSIGN_OR_RETURN(auto fut, sub_managers_[s]->H2hReadExplicit(
                                   matched_ep, src_block_ids, dst_block_ids,
                                   /*explicit_dst_ptrs=*/{}));
    sub_copy_futures.push_back(std::move(fut));
  }
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
}

absl::StatusOr<raiden::PjRtCopyFuture> NumaAwareKVCacheManager::H2dRead(
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int64_t>& src_host_offsets,
    const std::vector<int64_t>& dst_host_offsets,
    const std::vector<int64_t>& dst_device_offsets,
    const std::vector<int64_t>& copy_sizes) {
  if (sub_managers_.empty()) {
    return raiden::PjRtCopyFuture();
  }
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  sub_copy_futures.reserve(sub_managers_.size());

  for (size_t s = 0; s < sub_managers_.size(); ++s) {
    // Same endpoint matching as the vector H2hRead/H2hWrite above: pick the
    // remote descriptor that serves this sub-manager's global shards. Every
    // sub-manager owns a shard of every block, so the block offsets are
    // identical across sub-managers; only the peer differs.
    const auto& sub_shards = submanager_to_global_shards_[s];
    std::string matched_ep;
    for (const auto& desc : remote_descriptors) {
      for (int64_t gsh : sub_shards) {
        if (std::find(desc.shards.begin(), desc.shards.end(), gsh) !=
            desc.shards.end()) {
          matched_ep = desc.endpoint;
          break;
        }
      }
      if (!matched_ep.empty()) break;
    }
    if (matched_ep.empty() && !remote_descriptors.empty()) {
      matched_ep = remote_descriptors[0].endpoint;
    }

    ASSIGN_OR_RETURN(auto fut,
                     sub_managers_[s]->H2dRead(matched_ep, src_host_offsets,
                                               dst_host_offsets,
                                               dst_device_offsets, copy_sizes));
    sub_copy_futures.push_back(std::move(fut));
  }
  // Event-aware join (see D2h/H2d): on TPU the per-shard transfers complete via
  // the PJRT C-API event path, leaving f.future invalid and signalling through
  // event_bundles; xla::JoinFutures on the raw future would CHECK-fail.
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
}

absl::Status NumaAwareKVCacheManager::UnlockBlocks(
    const std::vector<int>& block_ids) {
  for (auto& sub : sub_managers_) {
    if (sub->host_block_manager() != nullptr) {
      auto status = sub->host_block_manager()->Unlock(block_ids);
      if (!status.ok()) return status;
    }
  }
  return absl::OkStatus();
}

std::string NumaAwareKVCacheManager::DumpMetricsToString() const {
  if (metrics_collector_) {
    return metrics_collector_->DumpMetricsToString();
  }
  return "[]";
}

#ifndef WITHOUT_PYTHON
KVCacheManager::KVCacheManager(
    nb::list device_arrays, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id, int64_t node_id)
    : numa_manager_(std::make_unique<NumaAwareKVCacheManager>(
          std::move(device_arrays), local_port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism, node_id)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::KVCacheManager(
    nanobind::list kv_caches, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    bool unsafe_skip_buffer_lock, int parallelism, int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id)
    : numa_manager_(std::make_unique<NumaAwareKVCacheManager>(
          std::move(kv_caches), node_id, local_control_port, max_blocks,
          num_slots, timeout_s, unsafe_skip_buffer_lock, parallelism)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}
#endif

KVCacheManager::KVCacheManager(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id)
    : numa_manager_(std::make_unique<NumaAwareKVCacheManager>(
          num_layers, num_shards, slice_byte_size, local_port,
          host_blocks_to_allocate, parallelism)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::KVCacheManager(
    std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers,
    int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id)
    : numa_manager_(
          std::make_unique<NumaAwareKVCacheManager>(std::move(sub_managers))) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::~KVCacheManager() {
  if (private_grpc_server_) {
    private_grpc_server_->SetTransferManager(nullptr);
  }
  controller::WorkerServiceServer::GetInstance().SetTransferManager(nullptr);
}

void KVCacheManager::StartGrpcServer(
    int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id) {
  if (!raiden_controller_address.has_value() ||
      raiden_controller_address->empty()) {
    return;
  }

  bool use_private_server = false;
  const char* disable_singleton =
      std::getenv("RAIDEN_DISABLE_SINGLETON_WORKER");
  if (disable_singleton != nullptr &&
      (std::strcmp(disable_singleton, "true") == 0 ||
       std::strcmp(disable_singleton, "1") == 0)) {
    use_private_server = true;
  }

  absl::Status status;
  if (use_private_server) {
    private_grpc_server_ = controller::WorkerServiceServer::Create();
    status = private_grpc_server_->StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(numa_manager_.get()),
        raiden_worker_port);
  } else {
    status = controller::WorkerServiceServer::GetInstance().StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(numa_manager_.get()),
        raiden_worker_port);
  }

  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to start gRPC server in KVCacheManager: ", status.message()));
  }

  if (raiden_controller_address.has_value() &&
      !raiden_controller_address->empty()) {
    int bound_port = GetRaidenWorkerPort();
    std::string w_id = worker_id.value_or("worker_0");

    std::string worker_ip = "127.0.0.1";
    auto ips = GetLocalHostIpAddresses();
    if (!ips.empty()) {
      worker_ip = ips[0];
    }
    std::string worker_endpoint;
    if (absl::StrContains(worker_ip, ":")) {
      worker_endpoint = absl::StrCat("[", worker_ip, "]:", bound_port);
    } else {
      worker_endpoint = absl::StrCat(worker_ip, ":", bound_port);
    }

    // Registry endpoints are what remote peers open DATA connections to (a
    // pull's H2hRead/H2dRead, a push's H2hWrite), so they must be data
    // endpoints even when this manager also runs a control server.
    auto local_eps = numa_manager_->get_local_data_endpoints();
    std::string transfer_endpoints_log = "";
    for (size_t i = 0; i < local_eps.size(); ++i) {
      if (i > 0) absl::StrAppend(&transfer_endpoints_log, ", ");
      absl::StrAppend(&transfer_endpoints_log, local_eps[i].endpoint);
    }

    std::vector<uint64_t> block_array_bytes;
    block_array_bytes.reserve(numa_manager_->num_block_arrays());
    for (size_t i = 0; i < numa_manager_->num_block_arrays(); ++i) {
      block_array_bytes.push_back(numa_manager_->block_bytes(i));
    }

    core::controller::RaidenControllerClient client(*raiden_controller_address);
    status = client.RegisterWorker(
        w_id, worker_endpoint, local_eps, numa_manager_->node_id(),
        block_array_bytes, static_cast<int32_t>(numa_manager_->num_shards()));
    if (!status.ok()) {
      LOG(ERROR) << "Failed to register worker with controller: "
                 << status.message();
    } else {
      LOG(INFO) << "Successfully registered worker " << w_id
                << " (worker_endpoint=" << worker_endpoint
                << ", transfer_endpoints=[" << transfer_endpoints_log
                << "]) with controller at " << *raiden_controller_address;
    }
  }
}

int KVCacheManager::GetRaidenWorkerPort() const {
  if (private_grpc_server_) {
    return private_grpc_server_->GetRaidenWorkerPort();
  }
  return controller::WorkerServiceServer::GetInstance().GetRaidenWorkerPort();
}

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

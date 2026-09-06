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

#include "tpu_sync/frameworks/jax/weight_synchronizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>  // NOLINT
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

#ifndef WITHOUT_PYTHON
#include "absl/strings/str_cat.h"
#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/utils.h"
#endif

namespace tpu_raiden {
namespace jax {

// ============================================================================
// NumaAwareWeightSynchronizer Implementation
// ============================================================================

#ifndef WITHOUT_PYTHON
NumaAwareWeightSynchronizer::NumaAwareWeightSynchronizer(
    nanobind::list jax_arrays, std::optional<int> local_port, int parallelism,
    bool unsafe_skip_buffer_lock, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, bool auto_h2d,
    std::optional<std::vector<int64_t>> global_shard_indices)
    : unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock),
      global_shard_indices_(
          global_shard_indices.value_or(std::vector<int64_t>{})) {
  auto layer_buffers =
      tpu_raiden::jax::UnpackJaxArrays(jax_arrays, unsafe_skip_buffer_lock);
  InitSubManagers(layer_buffers, local_port, unsafe_skip_buffer_lock,
                  parallelism, listener_port, bind_ip, auto_h2d,
                  global_shard_indices);
}

absl::Status NumaAwareWeightSynchronizer::BindWeights(
    nanobind::list jax_arrays) {
  try {
    auto layer_buffers =
        tpu_raiden::jax::UnpackJaxArrays(jax_arrays, unsafe_skip_buffer_lock_);
    if (layer_buffers.empty()) {
      return absl::InvalidArgumentError(
          "Empty layer buffers provided to BindWeights");
    }
    if (layer_buffers.size() != num_layers_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Layer count mismatch in BindWeights: expected ",
                       num_layers_, ", got ", layer_buffers.size()));
    }
    if (layer_buffers[0].size() != total_num_shards_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Shard count mismatch in BindWeights: expected ",
                       total_num_shards_, ", got ", layer_buffers[0].size()));
    }

    for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
      if (!sub_synchronizers_[s]) continue;
      const auto& local_shards = (s < submanager_to_local_shards_.size())
                                     ? submanager_to_local_shards_[s]
                                     : std::vector<int>{};
      std::vector<std::vector<raiden::RaidenBufferHandle>> sub_buffers(
          num_layers_);
      for (size_t l = 0; l < num_layers_; ++l) {
        sub_buffers[l].reserve(local_shards.size());
        for (int lsh : local_shards) {
          if (lsh < 0 || lsh >= static_cast<int>(layer_buffers[l].size())) {
            return absl::OutOfRangeError("Local shard index out of range");
          }
          sub_buffers[l].push_back(layer_buffers[l][lsh]);
        }
      }
      TF_RETURN_IF_ERROR(sub_synchronizers_[s]->BindWeights(sub_buffers));
    }
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
}
#endif

NumaAwareWeightSynchronizer::NumaAwareWeightSynchronizer(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    bool auto_h2d, std::optional<std::vector<int64_t>> global_shard_indices)
    : total_num_shards_(num_shards),
      num_layers_(num_layers),
      slice_byte_size_(slice_byte_size),
      global_shard_indices_(
          global_shard_indices.value_or(std::vector<int64_t>{})) {
  auto sub = std::make_unique<weight_sync::WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size, local_port,
      /*host_blocks_to_allocate=*/std::nullopt, parallelism, listener_port,
      bind_ip, /*layer_names=*/std::vector<std::string>{}, auto_h2d);
  sub_synchronizers_.push_back(std::move(sub));
  global_shard_to_submanager_.resize(total_num_shards_);
  submanager_to_global_shards_.resize(1);
  submanager_to_local_shards_.resize(1);
  for (size_t i = 0; i < total_num_shards_; ++i) {
    global_shard_to_submanager_[i] = {0, static_cast<int>(i)};
    int64_t gidx = (i < global_shard_indices_.size()) ? global_shard_indices_[i]
                                                      : static_cast<int64_t>(i);
    submanager_to_global_shards_[0].push_back(gidx);
    submanager_to_local_shards_[0].push_back(static_cast<int>(i));
  }
  if (!sub_synchronizers_.empty() && sub_synchronizers_[0]) {
    sub_synchronizers_[0]->SetGlobalShardIndices(
        submanager_to_global_shards_[0]);
    sub_synchronizers_[0]->SetLocalShardIndices(submanager_to_local_shards_[0]);
    sub_synchronizers_[0]->SetControlDelegate(this);
  }
}

NumaAwareWeightSynchronizer::NumaAwareWeightSynchronizer(
    std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>>
        sub_synchronizers) {
  sub_synchronizers_ = std::move(sub_synchronizers);
  total_num_shards_ = 0;
  for (const auto& sub : sub_synchronizers_) {
    if (sub) total_num_shards_ += sub->num_shards();
  }
  if (!sub_synchronizers_.empty() && sub_synchronizers_[0]) {
    num_layers_ = sub_synchronizers_[0]->num_layers();
    slice_byte_size_ = sub_synchronizers_[0]->slice_byte_size();
  }
  global_shard_to_submanager_.resize(total_num_shards_);
  submanager_to_global_shards_.resize(sub_synchronizers_.size());
  submanager_to_local_shards_.resize(sub_synchronizers_.size());
  global_shard_indices_.clear();
  global_shard_indices_.reserve(total_num_shards_);
  int global_idx = 0;
  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    size_t nsh =
        sub_synchronizers_[s] ? sub_synchronizers_[s]->num_shards() : 0;
    for (size_t l = 0; l < nsh; ++l) {
      global_shard_to_submanager_[global_idx] = {static_cast<int>(s),
                                                 static_cast<int>(l)};
      int64_t gidx = static_cast<int64_t>(global_idx);
      submanager_to_global_shards_[s].push_back(gidx);
      submanager_to_local_shards_[s].push_back(static_cast<int>(l));
      global_shard_indices_.push_back(gidx);
      global_idx++;
    }
  }
  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    if (sub_synchronizers_[s]) {
      sub_synchronizers_[s]->SetGlobalShardIndices(
          submanager_to_global_shards_[s]);
      sub_synchronizers_[s]->SetLocalShardIndices(
          submanager_to_local_shards_[s]);
      if (s == 0) {
        sub_synchronizers_[s]->SetControlDelegate(this);
      }
    }
  }
}

NumaAwareWeightSynchronizer::~NumaAwareWeightSynchronizer() = default;

void NumaAwareWeightSynchronizer::InitSubManagers(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, bool unsafe_skip_buffer_lock,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, bool auto_h2d,
    std::optional<std::vector<int64_t>> global_shard_indices) {
  if (layer_buffers.empty()) return;
  num_layers_ = layer_buffers.size();
  total_num_shards_ = layer_buffers[0].size();
  slice_byte_size_ = layer_buffers[0].empty()
                         ? 0
                         : layer_buffers[0][0].GetOnDeviceSizeInBytes();
  if (global_shard_indices.has_value()) {
    global_shard_indices_ = *global_shard_indices;
  }
  global_shard_to_submanager_.resize(total_num_shards_);

  absl::btree_map<int, std::vector<int>> numa_to_shards;
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
  std::vector<HostNicAddress> data_nics;
  for (const auto& nic : host_nics) {
    if (nic.classification == NicClassification::kDataPlane) {
      num_ext_nics++;
      data_nics.push_back(nic);
    }
  }

  bool force_single_numa = true;
  const char* enable_multi_numa_env = std::getenv("ENABLE_MULTI_NUMA");
  if (enable_multi_numa_env != nullptr) {
    if (absl::EqualsIgnoreCase(enable_multi_numa_env, "true") ||
        absl::EqualsIgnoreCase(enable_multi_numa_env, "1")) {
      force_single_numa = false;
    }
  }

  if (bind_ip.has_value() && !bind_ip->empty()) {
    force_single_numa = true;
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
    numa_to_shards.clear();
    size_t shards_per_numa =
        (total_num_shards_ + num_ext_nics - 1) / num_ext_nics;
    for (size_t sh = 0; sh < total_num_shards_; ++sh) {
      int target_numa = static_cast<int>(sh / shards_per_numa);
      numa_to_shards[target_numa].push_back(static_cast<int>(sh));
    }
  }

  const bool ephemeral_data_port = (local_port.value_or(-1) == 0);
  const int kMaxPortAttempts = ephemeral_data_port ? 64 : 1;
  for (int attempt = 0; attempt < kMaxPortAttempts; ++attempt) {
    sub_synchronizers_.clear();
    submanager_to_global_shards_.clear();
    submanager_to_global_shards_.reserve(numa_to_shards.size());
    submanager_to_local_shards_.clear();
    submanager_to_local_shards_.reserve(numa_to_shards.size());
    std::optional<int> bound_base_port = std::nullopt;
    std::optional<int> bound_base_listener_port = std::nullopt;
    bool bind_conflict = false;

    for (const auto& [numa, shards] : numa_to_shards) {
      int sub_idx = static_cast<int>(sub_synchronizers_.size());
      std::vector<int64_t> gshards;
      gshards.reserve(shards.size());
      for (size_t local_sh = 0; local_sh < shards.size(); ++local_sh) {
        int sh_idx = shards[local_sh];
        global_shard_to_submanager_[sh_idx] = {sub_idx,
                                               static_cast<int>(local_sh)};
        int64_t gidx =
            (static_cast<size_t>(sh_idx) < global_shard_indices_.size())
                ? global_shard_indices_[sh_idx]
                : static_cast<int64_t>(sh_idx);
        gshards.push_back(gidx);
      }
      submanager_to_global_shards_.push_back(std::move(gshards));
      submanager_to_local_shards_.push_back(shards);

      std::vector<std::vector<raiden::RaidenBufferHandle>> sub_buffers(
          num_layers_);
      for (size_t l = 0; l < num_layers_; ++l) {
        sub_buffers[l].reserve(shards.size());
        for (int sh : shards) {
          sub_buffers[l].push_back(layer_buffers[l][sh]);
        }
      }

      std::optional<int> sub_port = local_port;
      if (bound_base_port.has_value() && local_port.value_or(0) > 0) {
        sub_port = *bound_base_port + sub_idx;
      } else if (sub_port.has_value() && *sub_port > 0) {
        sub_port = *sub_port + sub_idx;
      }

      std::optional<int> sub_listener_port = listener_port;
      if (bound_base_listener_port.has_value() &&
          listener_port.value_or(0) > 0) {
        sub_listener_port = *bound_base_listener_port + sub_idx;
      } else if (sub_listener_port.has_value() && *sub_listener_port > 0) {
        sub_listener_port = *sub_listener_port + sub_idx;
      }

      std::optional<std::string> sub_bind_ip = bind_ip;
      if (!sub_bind_ip.has_value() || sub_bind_ip->empty()) {
        if (sub_idx < static_cast<int>(data_nics.size())) {
          sub_bind_ip = data_nics[sub_idx].ip_address;
        } else if (!data_nics.empty()) {
          sub_bind_ip = data_nics[sub_idx % data_nics.size()].ip_address;
        }
      }

      std::unique_ptr<weight_sync::WeightSynchronizerBase> sub_sync;
      try {
        sub_sync = std::make_unique<weight_sync::WeightSynchronizerBase>(
            sub_buffers, sub_port, /*external_host_ptrs=*/std::nullopt,
            unsafe_skip_buffer_lock, parallelism, sub_listener_port,
            sub_bind_ip, /*layer_names=*/std::vector<std::string>{}, auto_h2d);
      } catch (const std::exception& e) {
        if (!ephemeral_data_port || attempt + 1 >= kMaxPortAttempts) throw;
        bind_conflict = true;
        break;
      }

      if (local_port.has_value()) {
        (void)sub_sync->local_port();
      }
      if (!bound_base_port.has_value() && sub_sync->local_port().has_value()) {
        bound_base_port = sub_sync->local_port().value();
      }
      if (listener_port.has_value()) {
        (void)sub_sync->listener_port();
      }
      if (!bound_base_listener_port.has_value() &&
          sub_sync->listener_port().has_value()) {
        bound_base_listener_port = sub_sync->listener_port().value();
      }

      sub_sync->SetGlobalShardIndices(submanager_to_global_shards_[sub_idx]);
      sub_sync->SetLocalShardIndices(submanager_to_local_shards_[sub_idx]);
      if (sub_idx == 0) {
        sub_sync->SetControlDelegate(this);
      }

      sub_synchronizers_.push_back(std::move(sub_sync));
    }
    if (!bind_conflict) break;
  }
}

std::optional<int> NumaAwareWeightSynchronizer::local_port() const {
  return sub_synchronizers_.empty() ? std::nullopt
                                    : sub_synchronizers_[0]->local_port();
}

std::optional<int> NumaAwareWeightSynchronizer::listener_port() const {
  return sub_synchronizers_.empty() ? std::nullopt
                                    : sub_synchronizers_[0]->listener_port();
}

bool NumaAwareWeightSynchronizer::is_listener_active() const {
  return !sub_synchronizers_.empty() && sub_synchronizers_[0] &&
         sub_synchronizers_[0]->is_listener_active();
}

std::vector<std::string> NumaAwareWeightSynchronizer::local_ips() const {
  std::vector<std::string> ips;
  for (const auto& sub : sub_synchronizers_) {
    if (sub) {
      for (const auto& ip : sub->local_ips()) {
        if (std::find(ips.begin(), ips.end(), ip) == ips.end()) {
          ips.push_back(ip);
        }
      }
    }
  }
  return ips;
}

std::vector<RaidenTransferEndpoint>
NumaAwareWeightSynchronizer::get_local_endpoints() const {
  std::vector<RaidenTransferEndpoint> res;
  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    if (!sub_synchronizers_[s]) continue;
    auto sub_eps = sub_synchronizers_[s]->get_local_endpoints();
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

const uint8_t* NumaAwareWeightSynchronizer::GetHostBufferPtr(
    size_t layer_idx, size_t shard_idx) const {
  if (shard_idx >= global_shard_to_submanager_.size()) return nullptr;
  auto [sub_idx, local_shard] = global_shard_to_submanager_[shard_idx];
  if (sub_idx < 0 || sub_idx >= static_cast<int>(sub_synchronizers_.size()) ||
      !sub_synchronizers_[sub_idx]) {
    return nullptr;
  }
  return sub_synchronizers_[sub_idx]->GetHostBufferPtr(layer_idx, local_shard);
}

absl::StatusOr<raiden::PjRtCopyFuture> NumaAwareWeightSynchronizer::D2h(
    uint64_t uuid) {
  if (sub_synchronizers_.empty()) return raiden::PjRtCopyFuture();
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  sub_copy_futures.reserve(sub_synchronizers_.size());
  for (auto& sub : sub_synchronizers_) {
    if (sub) {
      ASSIGN_OR_RETURN(auto f, sub->D2h(uuid));
      sub_copy_futures.push_back(std::move(f));
    }
  }
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
}

absl::StatusOr<raiden::PjRtCopyFuture> NumaAwareWeightSynchronizer::H2d(
    uint64_t uuid) {
  if (sub_synchronizers_.empty()) return raiden::PjRtCopyFuture();
  std::vector<raiden::PjRtCopyFuture> sub_copy_futures;
  sub_copy_futures.reserve(sub_synchronizers_.size());
  for (auto& sub : sub_synchronizers_) {
    if (sub) {
      ASSIGN_OR_RETURN(auto f, sub->H2d(uuid));
      sub_copy_futures.push_back(std::move(f));
    }
  }
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(sub_copy_futures));
}

void NumaAwareWeightSynchronizer::SetSkipTiling(
    const std::vector<bool>& skip_tiling) {
  for (auto& sub : sub_synchronizers_) {
    if (sub) sub->SetSkipTiling(skip_tiling);
  }
}

void NumaAwareWeightSynchronizer::SetSkipTiling(bool skip_all) {
  for (auto& sub : sub_synchronizers_) {
    if (sub) sub->SetSkipTiling(skip_all);
  }
}

weight_sync::WeightSyncMetrics NumaAwareWeightSynchronizer::GetMetrics() const {
  weight_sync::WeightSyncMetrics aggregated;
  if (sub_synchronizers_.empty()) return aggregated;

  for (const auto& sub : sub_synchronizers_) {
    if (!sub) continue;
    weight_sync::WeightSyncMetrics m = sub->GetMetrics();
    aggregated.last_d2h_time_ms =
        std::max(aggregated.last_d2h_time_ms, m.last_d2h_time_ms);
    aggregated.last_h2d_time_ms =
        std::max(aggregated.last_h2d_time_ms, m.last_h2d_time_ms);
    aggregated.last_h2h_time_ms =
        std::max(aggregated.last_h2h_time_ms, m.last_h2h_time_ms);
    aggregated.last_staging_time_ms =
        std::max(aggregated.last_staging_time_ms, m.last_staging_time_ms);
    aggregated.last_tiling_time_ms =
        std::max(aggregated.last_tiling_time_ms, m.last_tiling_time_ms);
    aggregated.last_detiling_time_ms =
        std::max(aggregated.last_detiling_time_ms, m.last_detiling_time_ms);
    aggregated.last_total_push_resharded_time_ms =
        std::max(aggregated.last_total_push_resharded_time_ms,
                 m.last_total_push_resharded_time_ms);

    aggregated.last_d2h_bytes += m.last_d2h_bytes;
    aggregated.last_h2d_bytes += m.last_h2d_bytes;
    aggregated.last_h2h_bytes += m.last_h2h_bytes;
    aggregated.last_tiled_bytes += m.last_tiled_bytes;
    aggregated.last_detiled_bytes += m.last_detiled_bytes;

    aggregated.total_d2h_time_ms =
        std::max(aggregated.total_d2h_time_ms, m.total_d2h_time_ms);
    aggregated.total_h2d_time_ms =
        std::max(aggregated.total_h2d_time_ms, m.total_h2d_time_ms);
    aggregated.total_h2h_time_ms =
        std::max(aggregated.total_h2h_time_ms, m.total_h2h_time_ms);
    aggregated.total_staging_time_ms =
        std::max(aggregated.total_staging_time_ms, m.total_staging_time_ms);
    aggregated.total_tiling_time_ms =
        std::max(aggregated.total_tiling_time_ms, m.total_tiling_time_ms);
    aggregated.total_detiling_time_ms =
        std::max(aggregated.total_detiling_time_ms, m.total_detiling_time_ms);
    aggregated.total_push_resharded_time_ms =
        std::max(aggregated.total_push_resharded_time_ms,
                 m.total_push_resharded_time_ms);

    aggregated.total_d2h_bytes += m.total_d2h_bytes;
    aggregated.total_h2d_bytes += m.total_h2d_bytes;
    aggregated.total_h2h_bytes += m.total_h2h_bytes;
    aggregated.total_tiled_bytes += m.total_tiled_bytes;
    aggregated.total_detiled_bytes += m.total_detiled_bytes;

    aggregated.d2h_call_count =
        std::max(aggregated.d2h_call_count, m.d2h_call_count);
    aggregated.push_resharded_call_count = std::max(
        aggregated.push_resharded_call_count, m.push_resharded_call_count);
  }
  return aggregated;
}

void NumaAwareWeightSynchronizer::ResetMetrics() {
  for (auto& sub : sub_synchronizers_) {
    if (sub) sub->ResetMetrics();
  }
}

absl::Status NumaAwareWeightSynchronizer::PushWeights(
    const std::vector<std::string>& peers) {
  if (sub_synchronizers_.empty()) return absl::OkStatus();
  if (sub_synchronizers_.size() == 1) {
    return sub_synchronizers_[0]->PushWeightsLocal(peers);
  }
  if (!push_pool_) {
    push_pool_ =
        std::make_unique<tpu_raiden::NumaThreadPool>(sub_synchronizers_.size());
  }
  std::vector<std::future<absl::Status>> futures;
  futures.reserve(sub_synchronizers_.size());
  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    std::optional<int> numa_node =
        sub_synchronizers_[s] ? sub_synchronizers_[s]->assigned_numa_node()
                              : std::nullopt;
    futures.push_back(push_pool_->Schedule(numa_node, [this, s, &peers]() {
      return sub_synchronizers_[s]
                 ? sub_synchronizers_[s]->PushWeightsLocal(peers)
                 : absl::OkStatus();
    }));
  }
  absl::Status first_error = absl::OkStatus();
  for (auto& f : futures) {
    absl::Status status = f.get();
    if (!status.ok() && first_error.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

absl::Status NumaAwareWeightSynchronizer::PushWeightsResharded(
    const tpu_sync::rpc::StartTransferRequest& request) {
  if (sub_synchronizers_.empty()) return absl::OkStatus();
  if (sub_synchronizers_.size() == 1) {
    return sub_synchronizers_[0]->PushWeightsReshardedLocal(request);
  }
  if (!push_pool_) {
    push_pool_ =
        std::make_unique<tpu_raiden::NumaThreadPool>(sub_synchronizers_.size());
  }
  std::vector<std::future<absl::Status>> futures;
  futures.reserve(sub_synchronizers_.size());
  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    std::optional<int> numa_node =
        sub_synchronizers_[s] ? sub_synchronizers_[s]->assigned_numa_node()
                              : std::nullopt;
    futures.push_back(push_pool_->Schedule(numa_node, [this, s, &request]() {
      return sub_synchronizers_[s]
                 ? sub_synchronizers_[s]->PushWeightsReshardedLocal(request)
                 : absl::OkStatus();
    }));
  }
  absl::Status first_error = absl::OkStatus();
  for (auto& f : futures) {
    absl::Status status = f.get();
    if (!status.ok() && first_error.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

void NumaAwareWeightSynchronizer::StoreSkipTiling(
    uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request) {
  for (auto& sub : sub_synchronizers_) {
    if (sub) {
      sub->StoreSkipTilingLocal(uuid, request);
    }
  }

  const auto& schedules = request.shard_push_schedules();
  if (!schedules.empty()) {
    absl::MutexLock lock(expected_counts_mu_);
    auto& sub_layer_counts = uuid_to_sub_layer_counts_[uuid];
    auto& sub_total_counts = uuid_to_sub_total_counts_[uuid];
    sub_layer_counts.resize(sub_synchronizers_.size());
    sub_total_counts.resize(sub_synchronizers_.size(), 0);

    for (const auto& [src_shard, schedule] : schedules) {
      for (const auto& entry : schedule.entries()) {
        int dst_shard_idx = entry.dst_shard_idx();
        int target_sub = -1;
        auto it = std::find(global_shard_indices_.begin(),
                            global_shard_indices_.end(), dst_shard_idx);
        if (it != global_shard_indices_.end()) {
          size_t local_dst_shard =
              std::distance(global_shard_indices_.begin(), it);
          if (local_dst_shard < global_shard_to_submanager_.size()) {
            target_sub = global_shard_to_submanager_[local_dst_shard].first;
          }
        } else if (global_shard_indices_.empty() && dst_shard_idx >= 0 &&
                   static_cast<size_t>(dst_shard_idx) <
                       global_shard_to_submanager_.size()) {
          target_sub = global_shard_to_submanager_[dst_shard_idx].first;
        }
        if (target_sub < 0 ||
            static_cast<size_t>(target_sub) >= sub_synchronizers_.size()) {
          for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
            if (sub_synchronizers_[s]) {
              auto eps = sub_synchronizers_[s]->get_local_endpoints();
              for (const auto& ep : eps) {
                if (ep.endpoint == entry.dst_peer()) {
                  target_sub = static_cast<int>(s);
                  break;
                }
              }
              if (target_sub >= 0) break;
            }
          }
        }
        if (target_sub >= 0 &&
            static_cast<size_t>(target_sub) < sub_synchronizers_.size()) {
          bool is_contiguous =
              (entry.count() == 1) ||
              (entry.src_stride_bytes() == entry.size_bytes() &&
               entry.dst_stride_bytes() == entry.size_bytes());
          uint32_t tasks_count =
              is_contiguous ? 1 : (entry.count() > 0 ? entry.count() : 1);
          size_t layer_idx = entry.has_layer_idx()
                                 ? static_cast<size_t>(entry.layer_idx())
                                 : 0;
          sub_layer_counts[target_sub][layer_idx] += tasks_count;
          sub_total_counts[target_sub] += tasks_count;
        }
      }
    }
  }
}

absl::Status NumaAwareWeightSynchronizer::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  if (sub_synchronizers_.empty()) return absl::OkStatus();
  if (sub_synchronizers_.size() == 1 && sub_synchronizers_[0]) {
    return sub_synchronizers_[0]->RegisterExpectedChunksLocal(uuid,
                                                              expected_chunks);
  }

  std::vector<uint32_t> sub_totals;
  {
    absl::MutexLock lock(expected_counts_mu_);
    auto it = uuid_to_sub_total_counts_.find(uuid);
    if (it != uuid_to_sub_total_counts_.end() &&
        it->second.size() == sub_synchronizers_.size()) {
      sub_totals = it->second;
    }
  }

  if (sub_totals.empty()) {
    sub_totals.resize(sub_synchronizers_.size(), 0);
    for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
      size_t sub_shards = (s < submanager_to_global_shards_.size())
                              ? submanager_to_global_shards_[s].size()
                              : 0;
      if (total_num_shards_ > 0 && sub_shards > 0) {
        uint32_t base_count =
            (expected_chunks * sub_shards) / total_num_shards_;
        uint32_t remainder = (expected_chunks * sub_shards) % total_num_shards_;
        uint32_t remainder_subs = remainder / sub_shards;
        sub_totals[s] = base_count + (s < remainder_subs ? 1 : 0);
        if (sub_totals[s] == 0 && expected_chunks > 0) {
          sub_totals[s] = expected_chunks;
        }
      } else {
        sub_totals[s] = expected_chunks;
      }
    }
  }

  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    if (sub_synchronizers_[s]) {
      TF_RETURN_IF_ERROR(sub_synchronizers_[s]->RegisterExpectedChunksLocal(
          uuid, sub_totals[s]));
    }
  }
  return absl::OkStatus();
}

absl::Status NumaAwareWeightSynchronizer::RegisterExpectedLayerChunks(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  if (sub_synchronizers_.empty()) return absl::OkStatus();
  if (sub_synchronizers_.size() == 1 && sub_synchronizers_[0]) {
    return sub_synchronizers_[0]->RegisterExpectedLayerChunksLocal(
        uuid, expected_layer_chunks);
  }

  std::vector<absl::flat_hash_map<size_t, uint32_t>> sub_counts;
  {
    absl::MutexLock lock(expected_counts_mu_);
    auto it = uuid_to_sub_layer_counts_.find(uuid);
    if (it != uuid_to_sub_layer_counts_.end() &&
        it->second.size() == sub_synchronizers_.size()) {
      sub_counts = it->second;
    }
  }

  if (sub_counts.empty()) {
    sub_counts.resize(sub_synchronizers_.size());
    for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
      size_t sub_shards = (s < submanager_to_global_shards_.size())
                              ? submanager_to_global_shards_[s].size()
                              : 0;
      for (const auto& [l, count] : expected_layer_chunks) {
        if (count == 0) continue;
        uint32_t sub_count = count;
        if (total_num_shards_ > 0 && sub_shards > 0) {
          uint32_t base_count = (count * sub_shards) / total_num_shards_;
          uint32_t remainder = (count * sub_shards) % total_num_shards_;
          uint32_t remainder_subs = remainder / sub_shards;
          sub_count = base_count + (s < remainder_subs ? 1 : 0);
          if (sub_count == 0) {
            sub_count = count;
          }
        }
        if (sub_count > 0) {
          sub_counts[s][l] = sub_count;
        }
      }
    }
  }

  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    if (sub_synchronizers_[s]) {
      absl::flat_hash_map<size_t, uint32_t> non_zero_counts;
      for (const auto& [l, c] : sub_counts[s]) {
        if (c > 0) {
          non_zero_counts[l] = c;
        }
      }
      TF_RETURN_IF_ERROR(
          sub_synchronizers_[s]->RegisterExpectedLayerChunksLocal(
              uuid, non_zero_counts));
    }
  }
  return absl::OkStatus();
}

absl::Status NumaAwareWeightSynchronizer::WaitForTransferCompletion(
    uint64_t uuid) {
  if (sub_synchronizers_.empty()) return absl::OkStatus();
  for (auto& sub : sub_synchronizers_) {
    if (sub) {
      TF_RETURN_IF_ERROR(sub->WaitForTransferCompletion(uuid));
    }
  }
  return absl::OkStatus();
}

void NumaAwareWeightSynchronizer::ForgetPushProgress(uint64_t uuid) {
  {
    absl::MutexLock lock(expected_counts_mu_);
    uuid_to_sub_layer_counts_.erase(uuid);
    uuid_to_sub_total_counts_.erase(uuid);
  }
  for (auto& sub : sub_synchronizers_) {
    if (sub) {
      sub->ForgetPushProgress(uuid);
    }
  }
}

void NumaAwareWeightSynchronizer::DrainPendingH2d() {
  for (auto& sub : sub_synchronizers_) {
    if (sub) {
      sub->DrainPendingH2d();
    }
  }
}

void NumaAwareWeightSynchronizer::SetSubmanagerShardsForTesting(
    const std::vector<std::vector<int64_t>>& assignment) {
  submanager_to_global_shards_ = assignment;
  submanager_to_local_shards_.clear();
  submanager_to_local_shards_.resize(assignment.size());
  global_shard_to_submanager_.clear();
  global_shard_indices_.clear();
  total_num_shards_ = 0;
  for (size_t s = 0; s < assignment.size(); ++s) {
    total_num_shards_ += assignment[s].size();
  }
  global_shard_to_submanager_.resize(total_num_shards_);
  global_shard_indices_.reserve(total_num_shards_);
  int local_idx = 0;
  for (size_t s = 0; s < assignment.size(); ++s) {
    for (size_t l = 0; l < assignment[s].size(); ++l) {
      submanager_to_local_shards_[s].push_back(local_idx);
      global_shard_to_submanager_[local_idx] = {static_cast<int>(s),
                                                static_cast<int>(l)};
      global_shard_indices_.push_back(assignment[s][l]);
      local_idx++;
    }
  }
  for (size_t s = 0; s < sub_synchronizers_.size(); ++s) {
    if (sub_synchronizers_[s] && s < submanager_to_global_shards_.size()) {
      sub_synchronizers_[s]->SetGlobalShardIndices(
          submanager_to_global_shards_[s]);
    }
  }
}

// ============================================================================
// WeightSynchronizer (Top-Level Facade) Implementation
// ============================================================================

#ifndef WITHOUT_PYTHON
WeightSynchronizer::WeightSynchronizer(
    nanobind::list jax_arrays, std::optional<int> local_port, int parallelism,
    bool unsafe_skip_buffer_lock, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, bool auto_h2d,
    std::optional<std::vector<int64_t>> global_shard_indices) {
  numa_manager_ = std::make_unique<NumaAwareWeightSynchronizer>(
      jax_arrays, local_port, parallelism, unsafe_skip_buffer_lock,
      listener_port, bind_ip, auto_h2d, global_shard_indices);
}

absl::Status WeightSynchronizer::BindWeights(nanobind::list jax_arrays) {
  return numa_manager_->BindWeights(jax_arrays);
}
#endif

WeightSynchronizer::WeightSynchronizer(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    bool auto_h2d, std::optional<std::vector<int64_t>> global_shard_indices) {
  numa_manager_ = std::make_unique<NumaAwareWeightSynchronizer>(
      num_layers, num_shards, slice_byte_size, local_port, parallelism,
      listener_port, bind_ip, auto_h2d, global_shard_indices);
}

WeightSynchronizer::WeightSynchronizer(
    std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>>
        sub_synchronizers) {
  numa_manager_ = std::make_unique<NumaAwareWeightSynchronizer>(
      std::move(sub_synchronizers));
}

WeightSynchronizer::~WeightSynchronizer() = default;

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::D2h(uint64_t uuid) {
  return numa_manager_->D2h(uuid);
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::H2d(uint64_t uuid) {
  return numa_manager_->H2d(uuid);
}

void WeightSynchronizer::SetSkipTiling(const std::vector<bool>& skip_tiling) {
  numa_manager_->SetSkipTiling(skip_tiling);
}

void WeightSynchronizer::SetSkipTiling(bool skip_all) {
  numa_manager_->SetSkipTiling(skip_all);
}

weight_sync::WeightSyncMetrics WeightSynchronizer::GetMetrics() const {
  return numa_manager_->GetMetrics();
}

void WeightSynchronizer::ResetMetrics() { numa_manager_->ResetMetrics(); }

const uint8_t* WeightSynchronizer::GetHostBufferPtr(size_t layer_idx,
                                                    size_t shard_idx) const {
  return numa_manager_->GetHostBufferPtr(layer_idx, shard_idx);
}

std::optional<int> WeightSynchronizer::local_port() const {
  return numa_manager_->local_port();
}

std::optional<int> WeightSynchronizer::listener_port() const {
  return numa_manager_->listener_port();
}

bool WeightSynchronizer::is_listener_active() const {
  return numa_manager_->is_listener_active();
}

std::vector<std::string> WeightSynchronizer::local_ips() const {
  return numa_manager_->local_ips();
}

std::vector<RaidenTransferEndpoint> WeightSynchronizer::get_local_endpoints()
    const {
  return numa_manager_->get_local_endpoints();
}

size_t WeightSynchronizer::num_layers() const {
  return numa_manager_->num_layers();
}

size_t WeightSynchronizer::num_shards() const {
  return numa_manager_->num_shards();
}

size_t WeightSynchronizer::slice_byte_size() const {
  return numa_manager_->slice_byte_size();
}

}  // namespace jax
}  // namespace tpu_raiden

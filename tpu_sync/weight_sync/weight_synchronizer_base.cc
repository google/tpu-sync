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

#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/weight_sync/tiling_utils.h"
#include "tpu_sync/weight_sync/weight_synchronizer_listener.h"

ABSL_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size, 256 * 1024,
          "Amount of scratchpad to allocate to host buffers for resharding "
          "pulls.");

namespace tpu_raiden {
namespace weight_sync {
namespace {

// Allocates an uninitialized shared buffer, using
// std::make_shared_for_overwrite when supported (C++20), falling back to
// std::shared_ptr<uint8_t[]>(new uint8_t[size]).
inline std::shared_ptr<uint8_t[]> MakeSharedBufferForOverwrite(size_t size) {
#if defined(__cpp_lib_smart_ptr_for_overwrite) && \
    __cpp_lib_smart_ptr_for_overwrite >= 202002L
  return std::make_shared_for_overwrite<uint8_t[]>(size);
#else
  return std::shared_ptr<uint8_t[]>(new uint8_t[size]);
#endif
}

}  // namespace

WeightSynchronizerBase::WeightSynchronizerBase(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port,
    std::optional<std::vector<const uint8_t*>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    std::vector<std::string> layer_names, bool auto_h2d)
    : tpu_raiden::RaidenManagerBase(
          layer_buffers.size(),
          layer_buffers.empty() ? 0 : layer_buffers[0].size(),
          layer_buffers.empty() ? 0
                                : layer_buffers[0][0].GetOnDeviceSizeInBytes(),
          local_port, parallelism, bind_ip),
      auto_h2d_(auto_h2d) {
  if (layer_names.empty()) {
    layer_names_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
      layer_names_.push_back(absl::StrCat("weights_", i));
    }
  } else {
    layer_names_ = std::move(layer_names);
  }
  DetectAndAssignNumaNode(layer_buffers);

  if (num_layers_ == 0 || num_shards_ == 0) {
    return;
  }

  const auto& first_handle = layer_buffers[0][0];
  physical_size_ = first_handle.GetOnDeviceSizeInBytes();

  if (!first_handle.is_common_buffer && first_handle.c_hold) {
    c_api_ = first_handle.c_hold->c_api;
    extension_ = first_handle.c_hold->extension;
  }

  shard_factor_ = 1;
  major_dim_size_ = 1;

  xla::PjRtClient* client = nullptr;
  if (first_handle.buffer) {
    client = const_cast<xla::PjRtClient*>(first_handle.buffer->client());
  } else if (first_handle.device) {
    client = const_cast<xla::PjRtClient*>(first_handle.device->client());
  }

  std::unique_ptr<HostMemoryAllocator> host_allocator;
  if (client) {
    auto alloc = HostMemoryAllocator::Create(client);
    if (alloc.ok()) {
      host_allocator = *std::move(alloc);
    }
  }

  size_t shard_idx = 0;
  layers_.reserve(num_layers_);
  buffer_holds_.reserve(num_layers_);

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& dst_buffers = layer_buffers[layer_idx];
    if (dst_buffers.size() != num_shards_) {
      throw std::runtime_error(
          "Number of shards mismatch across layers during weight sync init");
    }

    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& dst_buffer = dst_buffers[i];
      ShardBufferInfoBase shard_info;

      shard_info.device_size = dst_buffer.GetOnDeviceSizeInBytes();

      size_t alloc_size =
          shard_info.device_size +
          absl::GetFlag(FLAGS_raiden_weight_sync_host_buffer_scratchpad_size);
      if (external_host_ptrs.has_value()) {
        if (shard_idx < external_host_ptrs->size()) {
          shard_info.host_ptr = (*external_host_ptrs)[shard_idx];
        } else {
          throw std::invalid_argument("External host pointers size mismatch");
        }
        shard_info.host_size = alloc_size;
        shard_idx++;
      } else {
        if (alloc_size > 0) {
          if (host_allocator && dst_buffer.device) {
            auto alloc = host_allocator->AllocateDmaMappedForDevice(
                alloc_size, dst_buffer.device);
            if (alloc.ok()) {
              shard_info.host_ptr = (*alloc).ptr;
              shard_info.host_size = alloc_size;
              shard_info.host_owner = (*alloc).owner;
            }
          }
          if (shard_info.host_ptr == nullptr) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, alloc_size) != 0) {
              throw std::runtime_error(
                  "Failed to allocate host weights buffer");
            }
            std::memset(ptr, 0, alloc_size);
            shard_info.owned_host_buffer =
                std::unique_ptr<uint8_t[], void (*)(void*)>(
                    static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
            shard_info.host_ptr = shard_info.owned_host_buffer.get();
            shard_info.host_size = alloc_size;
          }
        }
      }

      hold_info.push_back(dst_buffer);
      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
    buffer_holds_.push_back(std::move(hold_info));
  }

  if (listener_port) {
    listener_ =
        std::make_unique<WeightSynchronizerListener>(this, *listener_port);
  }
  if (auto_h2d_) {
    h2d_pool_ = std::make_unique<tpu_raiden::NumaThreadPool>(
        std::max(parallelism_, 4));
  }
  push_pool_ = std::make_unique<tpu_raiden::NumaThreadPool>(
      std::max(parallelism_, 4));
}

WeightSynchronizerBase::WeightSynchronizerBase(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d)
    : WeightSynchronizerBase(num_layers, num_shards,
                             std::vector<size_t>(num_layers, slice_byte_size),
                             local_port, host_blocks_to_allocate, parallelism,
                             listener_port, bind_ip, std::move(layer_names),
                             auto_h2d) {}

WeightSynchronizerBase::WeightSynchronizerBase(
    size_t num_layers, size_t num_shards, std::vector<size_t> slice_byte_sizes,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d)
    : tpu_raiden::RaidenManagerBase(
          num_layers, num_shards,
          slice_byte_sizes.empty() ? 0 : slice_byte_sizes[0], local_port,
          parallelism, bind_ip),
      auto_h2d_(auto_h2d) {
  if (layer_names.empty()) {
    layer_names_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
      layer_names_.push_back(absl::StrCat("weights_", i));
    }
  } else {
    layer_names_ = std::move(layer_names);
  }
  physical_size_ = slice_byte_size_;
  shard_factor_ = 1;
  major_dim_size_ = 1;

  layers_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    size_t current_slice_byte_size = (layer_idx < slice_byte_sizes.size())
                                         ? slice_byte_sizes[layer_idx]
                                         : slice_byte_size_;
    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      ShardBufferInfoBase shard_info;
      shard_info.device_size = current_slice_byte_size;

      size_t alloc_size =
          current_slice_byte_size +
          absl::GetFlag(FLAGS_raiden_weight_sync_host_buffer_scratchpad_size);
      void* ptr = nullptr;
      if (alloc_size > 0) {
        if (posix_memalign(&ptr, 64, alloc_size) != 0) {
          throw std::runtime_error("Failed to allocate host weights buffer");
        }
        std::memset(ptr, 0, alloc_size);
      }
      shard_info.owned_host_buffer =
          std::unique_ptr<uint8_t[], void (*)(void*)>(
              static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
      shard_info.host_ptr = shard_info.owned_host_buffer.get();
      shard_info.host_size = alloc_size;

      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
  }

  if (listener_port) {
    listener_ =
        std::make_unique<WeightSynchronizerListener>(this, *listener_port);
  }
  if (auto_h2d_) {
    h2d_pool_ = std::make_unique<tpu_raiden::NumaThreadPool>(
        std::max(parallelism_, 4));
  }
  push_pool_ = std::make_unique<tpu_raiden::NumaThreadPool>(
      std::max(parallelism_, 4));
}

std::optional<int> WeightSynchronizerBase::listener_port() const {
  if (listener_) {
    return listener_->listener_port();
  }
  return std::nullopt;
}

bool WeightSynchronizerBase::is_listener_active() const {
  if (listener_) {
    return listener_->is_active();
  }
  return false;
}

std::vector<RaidenTransferEndpoint>
WeightSynchronizerBase::get_local_endpoints() const {
  std::vector<int64_t> shards;
  shards.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    shards.push_back(static_cast<int64_t>(i));
  }
  const std::string ip = local_ip();
  const int port = local_port().value_or(0);
  std::string ep = absl::StrContains(ip, ':')
                       ? absl::StrCat("[", ip, "]:", port)
                       : absl::StrCat(ip, ":", port);
  return {RaidenTransferEndpoint{ep, std::move(shards)}};
}

WeightSynchronizerBase::~WeightSynchronizerBase() {
  StopTransportServer();
  listener_.reset();
  h2d_pool_.reset();
  push_pool_.reset();
}

size_t WeightSynchronizerBase::GetPipelineGroupSize() const {
  if (pipeline_group_size_override_.has_value()) {
    return *pipeline_group_size_override_;
  }
  const char* env = std::getenv("RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE");
  if (env != nullptr && *env != '\0') {
    size_t val = 0;
    if (absl::SimpleAtoi(env, &val)) {
      return val;
    }
  }
  return 1;
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2dLayer(
    size_t layer_idx, uint64_t uuid) {
  if (buffer_holds_.empty() || layer_idx >= num_layers_) {
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }
  std::vector<bool> active_skip;
  {
    absl::MutexLock lock(skip_tiling_mu_);
    if (uuid != 0) {
      auto it = uuid_to_skip_tiling_.find(uuid);
      if (it != uuid_to_skip_tiling_.end()) {
        active_skip = it->second;
      }
    }
    if (active_skip.empty()) {
      if (!latest_skip_tiling_.empty()) {
        active_skip = latest_skip_tiling_;
      } else {
        active_skip = std::vector<bool>(num_layers_, false);
      }
    }
  }

  const auto& layer_info = layers_[layer_idx];
  const auto& layer_holds = buffer_holds_[layer_idx];
  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;

  for (size_t i = 0; i < num_shards_; ++i) {
    const auto& shard_info = layer_info.shards[i];
    const auto& shard_hold = layer_holds[i];

    const xla::Layout* xla_layout = nullptr;
    if (shard_hold.shape.has_layout()) {
      xla_layout = &shard_hold.shape.layout();
    }
    bool is_tiled = xla_layout && !xla_layout->tiles().empty();
    bool skip_flag = layer_idx < active_skip.size() && active_skip[layer_idx];
    if (skip_flag) {
      is_tiled = false;
    }
    VLOG(1) << "[WeightSynchronizerBase] H2dLayer " << layer_idx << " shard "
            << i << " (layer: "
            << (layer_idx < layer_names_.size() ? layer_names_[layer_idx]
                                                : "unknown")
            << ", is_tiled=" << is_tiled << ", skip_flag=" << skip_flag
            << ", shape=" << shard_hold.shape.ToString()
            << ", size=" << shard_info.device_size << " bytes)";

    std::vector<xla::Future<>> shard_futures;
    if (is_tiled) {
      int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(
          shard_hold.shape.element_type());
      size_t physical_bytes =
          tpu_raiden::weight_sync::GetTiledBufferElements(shard_hold.shape) *
          itemsize;
      auto temp_buffer = MakeSharedBufferForOverwrite(physical_bytes);
      auto tile_start = absl::Now();
      auto status = tpu_raiden::weight_sync::TileBuffer(
          shard_info.host_ptr, temp_buffer.get(), shard_hold.shape,
          *xla_layout);
      if (!status.ok()) {
        return status;
      }
      double tile_time_ms =
          absl::ToDoubleMilliseconds(absl::Now() - tile_start);
      {
        absl::MutexLock lock(metrics_mu_);
        metrics_.last_tiling_time_ms =
            std::max(metrics_.last_tiling_time_ms, tile_time_ms);
        metrics_.total_tiling_time_ms += tile_time_ms;
        metrics_.last_tiled_bytes += physical_bytes;
        metrics_.total_tiled_bytes += physical_bytes;
      }

      xla::Future<> future = shard_hold.CopyRawHostToDevice(
          temp_buffer.get(), 0, physical_bytes);
      xla::Future<> mapped_future = future.Map([temp_buffer]() {});
      shard_futures.push_back(std::move(mapped_future));
    } else {
      xla::Future<> future = shard_hold.CopyRawHostToDevice(
          shard_info.host_ptr, 0, shard_info.device_size);
      shard_futures.push_back(std::move(future));
    }
    shard_futures_to_join.push_back(raiden::CreateBufferFuture(
        std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
  }
  return raiden::PjRtCopyFuture::FromFuture(
      xla::JoinFutures(absl::MakeSpan(shard_futures_to_join)));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2d(
    uint64_t uuid) {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }
  VLOG(1) << "Starting H2d across " << num_layers_ << " layers (uuid=" << uuid
          << ")...";
  {
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_tiling_time_ms = 0.0;
    metrics_.last_tiled_bytes = 0;
  }
  std::vector<raiden::PjRtCopyFuture> layer_futures;
  layer_futures.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture layer_future,
                        H2dLayer(layer_idx, uuid));
    layer_futures.push_back(std::move(layer_future));
  }
  VLOG(1) << "Done with scheduling H2d across " << num_layers_
          << " layers (uuid=" << uuid << ").";
  return raiden::JoinPjRtCopyFutures(layer_futures);
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::D2hLayer(
    size_t layer_idx, uint64_t uuid) {
  if (buffer_holds_.empty() || layer_idx >= num_layers_) {
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }
  std::vector<bool> active_skip;
  {
    absl::MutexLock lock(skip_tiling_mu_);
    if (uuid != 0) {
      auto it = uuid_to_skip_tiling_.find(uuid);
      if (it != uuid_to_skip_tiling_.end()) {
        active_skip = it->second;
      }
    }
    if (active_skip.empty()) {
      if (!latest_skip_tiling_.empty()) {
        active_skip = latest_skip_tiling_;
      } else {
        active_skip = std::vector<bool>(num_layers_, false);
      }
    }
  }

  const auto& layer_info = layers_[layer_idx];
  const auto& layer_holds = buffer_holds_[layer_idx];

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;
  for (size_t i = 0; i < num_shards_; ++i) {
    const auto& shard_info = layer_info.shards[i];
    const auto& shard_hold = layer_holds[i];
    uint8_t* dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

    const xla::Layout* xla_layout = nullptr;
    if (shard_hold.shape.has_layout()) {
      xla_layout = &shard_hold.shape.layout();
    }
    bool is_tiled = xla_layout && !xla_layout->tiles().empty();
    bool skip_flag = layer_idx < active_skip.size() && active_skip[layer_idx];
    if (skip_flag) {
      is_tiled = false;
    }
    VLOG(1) << "[WeightSynchronizerBase] D2hLayer " << layer_idx << " shard "
            << i << " (layer: "
            << (layer_idx < layer_names_.size() ? layer_names_[layer_idx]
                                                : "unknown")
            << ", is_tiled=" << is_tiled << ", skip_flag=" << skip_flag
            << ", shape=" << shard_hold.shape.ToString()
            << ", size=" << shard_info.device_size << " bytes)";

    std::vector<xla::Future<>> shard_futures;
    if (is_tiled) {
      int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(
          shard_hold.shape.element_type());
      size_t physical_bytes =
          tpu_raiden::weight_sync::GetTiledBufferElements(shard_hold.shape) *
          itemsize;
      auto temp_buffer = MakeSharedBufferForOverwrite(physical_bytes);
      uint8_t* temp_buffer_ptr = temp_buffer.get();

      xla::Future<> copy_future =
          shard_hold.CopyRawDeviceToHost(temp_buffer_ptr, 0, physical_bytes);

      xla::Future<> detile_future = copy_future.Map(
          [this, temp_buffer, dst_host_ptr, shape = shard_hold.shape,
           layout = *xla_layout, physical_bytes]() -> absl::Status {
            auto detile_start = absl::Now();
            absl::Status status = tpu_raiden::weight_sync::DetileBuffer(
                temp_buffer.get(), dst_host_ptr, shape, layout);
            double detile_time_ms =
                absl::ToDoubleMilliseconds(absl::Now() - detile_start);
            if (status.ok()) {
              absl::MutexLock lock(metrics_mu_);
              metrics_.last_detiling_time_ms =
                  std::max(metrics_.last_detiling_time_ms, detile_time_ms);
              metrics_.total_detiling_time_ms =
                  std::max(metrics_.total_detiling_time_ms, detile_time_ms);
              metrics_.last_detiled_bytes += physical_bytes;
              metrics_.total_detiled_bytes += physical_bytes;
            }
            return status;
          });

      shard_futures.push_back(std::move(detile_future));
    } else {
      xla::Future<> future = shard_hold.CopyRawDeviceToHost(
          dst_host_ptr, 0, shard_info.device_size);
      shard_futures.push_back(std::move(future));
    }
    shard_futures_to_join.push_back(raiden::CreateBufferFuture(
        std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
  }
  return raiden::PjRtCopyFuture::FromFuture(
      xla::JoinFutures(absl::MakeSpan(shard_futures_to_join)));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::D2h(
    uint64_t uuid) {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }
  VLOG(1) << "Starting D2h across " << num_layers_ << " layers (uuid=" << uuid
          << ")...";
  {
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_detiling_time_ms = 0.0;
    metrics_.last_detiled_bytes = 0;
  }
  std::vector<raiden::PjRtCopyFuture> layer_futures;
  layer_futures.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture f, D2hLayer(layer_idx, uuid));
    layer_futures.push_back(std::move(f));
  }
  VLOG(1) << "Done with scheduling D2h across " << num_layers_
          << " layers (uuid=" << uuid << ").";
  return raiden::JoinPjRtCopyFutures(layer_futures);
}

absl::Status WeightSynchronizerBase::PushWeights(
    const std::vector<std::string>& peers) {
  if (peers.empty()) {
    return absl::InvalidArgumentError(
        "Peer list cannot be empty for trainer weights sync");
  }

  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture d2h_future, D2h());
  TF_RETURN_IF_ERROR(d2h_future.Await());

  std::vector<int> weights_block_id = {0};
  for (const std::string& peer : peers) {
    TF_RETURN_IF_ERROR(H2hWriteDirect(peer, weights_block_id).status());
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::PushWeightsResharded(
    const tpu_sync::rpc::StartTransferRequest& request) {
  VLOG(1) << "Starting PushWeightsResharded for uuid=" << request.uuid()
          << " across " << num_layers_
          << " layers (skip_d2h=" << request.skip_d2h() << ")...";
  StoreSkipTiling(request.uuid(), request);
  int fallback_layer_idx = -1;
  bool checked_fallback = false;
  auto get_fallback_layer_idx = [&]() -> absl::StatusOr<int> {
    if (checked_fallback) return fallback_layer_idx;
    checked_fallback = true;
    if (request.src_units().empty()) {
      return absl::InvalidArgumentError("src_units list cannot be empty");
    }
    std::string data_name = request.src_units(0).data_name();
    for (size_t l = 0; l < layer_names_.size(); ++l) {
      if (layer_names_[l] == data_name) {
        fallback_layer_idx = static_cast<int>(l);
        break;
      }
    }
    if (fallback_layer_idx == -1) {
      return absl::NotFoundError(absl::StrCat(
          "Layer name not found in WeightSynchronizer: ", data_name));
    }
    return fallback_layer_idx;
  };

  std::vector<bool> skip_tiling(num_layers_, false);
  for (const auto& [layer_idx, skip_val] : request.skip_tiling()) {
    if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < num_layers_) {
      skip_tiling[layer_idx] = skip_val;
    }
  }

  auto push_start = absl::Now();

  auto staging_start = absl::Now();
  std::vector<std::vector<transport::BufferPushTask>> tasks_by_layer(
      num_layers_);
  const auto& schedules = request.shard_push_schedules();
  for (size_t i = 0; i < num_shards_; ++i) {
    auto it = schedules.find(static_cast<int32_t>(i));
    if (it == schedules.end()) {
      continue;
    }
    const auto& schedule = it->second;

    for (const auto& entry : schedule.entries()) {
      int layer_idx_to_use = -1;
      if (entry.has_layer_idx()) {
        layer_idx_to_use = entry.layer_idx();
      } else {
        TF_ASSIGN_OR_RETURN(layer_idx_to_use, get_fallback_layer_idx());
      }
      if (layer_idx_to_use < 0 ||
          static_cast<size_t>(layer_idx_to_use) >= num_layers_) {
        return absl::InvalidArgumentError(
            absl::StrCat("Layer index out of bounds: ", layer_idx_to_use));
      }
      const uint8_t* base_host_ptr = GetHostPointer(layer_idx_to_use, i);
      if (base_host_ptr == nullptr) {
        return absl::InternalError(
            "Host pointer is null during resharded push");
      }
      size_t shard_host_size = GetHostSize(layer_idx_to_use, i);

      const std::string& dst_peer = entry.dst_peer();
      size_t dst_shard_idx = entry.dst_shard_idx();
      size_t count = entry.count() > 0 ? entry.count() : 1;
      size_t src_stride = entry.src_stride_bytes();
      size_t dst_stride = entry.dst_stride_bytes();
      size_t dst_offset = entry.dst_offset_bytes();
      size_t src_offset = entry.src_offset_bytes();
      size_t size = entry.size_bytes();

      if (count == 1 || (src_stride == size && dst_stride == size)) {
        size_t total_payload_size = count * size;
        if (src_offset + total_payload_size > shard_host_size) {
          return absl::InvalidArgumentError("Push range out of bounds");
        }
        tasks_by_layer[layer_idx_to_use].push_back({
            .peer = dst_peer,
            .buffer_id = static_cast<size_t>(layer_idx_to_use),
            .dst_shard_idx = dst_shard_idx,
            .dst_offset_bytes = dst_offset,
            .data_ptr = base_host_ptr + src_offset,
            .size_bytes = total_payload_size,
        });
      } else {
        for (size_t c = 0; c < count; ++c) {
          size_t curr_src_offset = src_offset + c * src_stride;
          size_t curr_dst_offset = dst_offset + c * dst_stride;

          if (curr_src_offset + size > shard_host_size) {
            return absl::InvalidArgumentError("Push range out of bounds");
          }

          const uint8_t* data_ptr = base_host_ptr + curr_src_offset;
          tasks_by_layer[layer_idx_to_use].push_back({
              .peer = dst_peer,
              .buffer_id = static_cast<size_t>(layer_idx_to_use),
              .dst_shard_idx = dst_shard_idx,
              .dst_offset_bytes = curr_dst_offset,
              .data_ptr = data_ptr,
              .size_bytes = size,
          });
        }
      }
    }
  }
  double staging_time_ms =
      absl::ToDoubleMilliseconds(absl::Now() - staging_start);

  bool already_completed = false;
  uint64_t uuid = request.uuid();
  std::vector<raiden::PjRtCopyFuture> d2h_layer_futures;
  d2h_layer_futures.reserve(num_layers_);
  auto d2h_start = absl::Now();
  if (!request.skip_d2h()) {
    if (uuid != 0) {
      absl::MutexLock lock(d2h_mu_);
      already_completed = !completed_d2h_uuids_.insert(uuid).second;
    }
    if (!already_completed) {
      VLOG(1)
          << "PushWeightsResharded: Executing pipelined D2H copies for uuid "
          << uuid;
      for (size_t l = 0; l < num_layers_; ++l) {
        TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture f, D2hLayer(l, uuid));
        d2h_layer_futures.push_back(std::move(f));
      }
    } else {
      VLOG(1) << "PushWeightsResharded: Coalescing D2H copy (already completed "
                 "for uuid "
              << uuid << ")";
    }
  } else {
    VLOG(1) << "PushWeightsResharded: Skipping D2H copy.";
  }

  if (!push_pool_) {
    push_pool_ = std::make_unique<tpu_raiden::NumaThreadPool>(
        std::max(parallelism_, 4));
  }

  auto h2h_start = absl::Now();
  size_t total_h2h_bytes = 0;
  size_t total_d2h_bytes = 0;
  double first_d2h_time_ms = 0.0;

  size_t pipeline_group_size = GetPipelineGroupSize();
  size_t group_size =
      (pipeline_group_size == 0) ? num_layers_ : pipeline_group_size;
  if (group_size == 0) {
    group_size = 1;
  }

  std::vector<std::future<absl::Status>> push_futures;
  push_futures.reserve((num_layers_ + group_size - 1) / group_size);

  for (size_t group_start = 0; group_start < num_layers_;
       group_start += group_size) {
    size_t group_end = std::min(num_layers_, group_start + group_size);
    std::vector<transport::BufferPushTask> group_tasks;

    for (size_t l = group_start; l < group_end; ++l) {
      for (size_t s = 0; s < num_shards_; ++s) {
        total_d2h_bytes += GetHostSize(l, s);
      }
      if (!request.skip_d2h() && !already_completed) {
        TF_RETURN_IF_ERROR(d2h_layer_futures[l].Await());
        if (l == 0) {
          first_d2h_time_ms =
              absl::ToDoubleMilliseconds(absl::Now() - d2h_start);
        }
      }
      const auto& layer_tasks = tasks_by_layer[l];
      if (!layer_tasks.empty()) {
        for (const auto& t : layer_tasks) {
          total_h2h_bytes += t.size_bytes;
        }
        group_tasks.insert(group_tasks.end(), layer_tasks.begin(),
                           layer_tasks.end());
      }
    }

    if (!group_tasks.empty()) {
      int push_parallelism =
          request.parallelism() > 0 ? request.parallelism() : parallelism_;
      push_futures.push_back(
          push_pool_->Schedule([this, group_tasks = std::move(group_tasks),
                                push_parallelism, uuid = request.uuid()]() {
            return PushWeightsChunks(group_tasks, push_parallelism, uuid);
          }));
    }
  }

  for (auto& fut : push_futures) {
    TF_RETURN_IF_ERROR(fut.get());
  }
  double h2h_time_ms = absl::ToDoubleMilliseconds(absl::Now() - h2h_start);

  double total_push_time_ms =
      absl::ToDoubleMilliseconds(absl::Now() - push_start);
  {
    absl::MutexLock lock(metrics_mu_);
    if (!request.skip_d2h() && !already_completed) {
      metrics_.last_d2h_time_ms = first_d2h_time_ms;
      metrics_.last_d2h_bytes = total_d2h_bytes;
      metrics_.d2h_call_count++;
    }
    metrics_.last_staging_time_ms = staging_time_ms;
    metrics_.total_staging_time_ms += staging_time_ms;
    metrics_.last_h2h_time_ms = h2h_time_ms;
    metrics_.last_h2h_bytes = total_h2h_bytes;
    metrics_.last_total_push_resharded_time_ms = total_push_time_ms;
    metrics_.push_resharded_call_count++;
  }
  VLOG(1) << "Done with PushWeightsResharded (uuid=" << request.uuid()
          << ", total_push_time=" << total_push_time_ms
          << " ms, h2h_time=" << h2h_time_ms << " ms).";
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::BindWeights(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers) {
  if (layer_buffers.size() != num_layers_) {
    return absl::InvalidArgumentError("Number of layers mismatch");
  }
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    if (layer_buffers[layer_idx].size() != num_shards_) {
      return absl::InvalidArgumentError("Number of shards mismatch");
    }
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      const auto& new_buffer = layer_buffers[layer_idx][shard_idx];
      size_t new_size = new_buffer.GetOnDeviceSizeInBytes();
      if (new_size != layers_[layer_idx].shards[shard_idx].device_size) {
        return absl::InvalidArgumentError("Buffer size mismatch");
      }
    }
  }

  if (!layer_buffers.empty() && !layer_buffers[0].empty()) {
    const auto& first_handle = layer_buffers[0][0];
    physical_size_ = first_handle.GetOnDeviceSizeInBytes();
    if (!first_handle.is_common_buffer && first_handle.c_hold) {
      c_api_ = first_handle.c_hold->c_api;
      extension_ = first_handle.c_hold->extension;
    }
  }

  buffer_holds_.clear();
  buffer_holds_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      hold_info.push_back(layer_buffers[layer_idx][shard_idx]);
    }
    buffer_holds_.push_back(std::move(hold_info));
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  {
    absl::MutexLock lock(pending_h2d_mu_);
    if (pending_h2d_states_[uuid].expected_layers == 0) {
      pending_h2d_states_[uuid].expected_layers = num_layers_;
    }
  }
  return RaidenManagerBase::RegisterExpectedChunks(uuid, expected_chunks);
}

absl::Status WeightSynchronizerBase::RegisterExpectedLayerChunks(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  {
    absl::MutexLock lock(pending_h2d_mu_);
    pending_h2d_states_[uuid].expected_layers = expected_layer_chunks.size();
  }
  return RaidenManagerBase::RegisterExpectedLayerChunks(uuid,
                                                        expected_layer_chunks);
}

absl::Status WeightSynchronizerBase::OnLayerDataReceived(size_t layer_idx,
                                                         uint64_t uuid) {
  if (!auto_h2d_) {
    return absl::OkStatus();
  }
  if (!h2d_pool_) {
    h2d_pool_ = std::make_unique<tpu_raiden::NumaThreadPool>(
        std::max(parallelism_, 4));
  }
  {
    absl::MutexLock lock(pending_h2d_mu_);
    auto& state = pending_h2d_states_[uuid];
    if (state.expected_layers == 0) {
      state.expected_layers = num_layers_;
    }
    state.layer_futures[layer_idx] =
        h2d_pool_->Schedule([this, layer_idx, uuid]() {
          return H2dLayer(layer_idx, uuid);
        });
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::OnDataReceived(uint64_t uuid) {
  auto record_completion = [this, uuid]() {
    if (uuid > 0) {
      absl::MutexLock lock(completed_transfers_mu_);
      completed_transfers_.insert(uuid);
    }
  };

  if (!auto_h2d_) {
    record_completion();
    return absl::OkStatus();
  }
  VLOG(1) << "Starting OnDataReceived (auto_h2d for uuid=" << uuid << ")...";
  auto h2d_start = absl::Now();
  absl::flat_hash_map<size_t,
                      std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
      layer_futures_map;
  {
    absl::MutexLock lock(pending_h2d_mu_);
    auto it = pending_h2d_states_.find(uuid);
    if (it != pending_h2d_states_.end()) {
      auto condition_fn =
          +[](std::pair<WeightSynchronizerBase*, uint64_t>* p)
              ABSL_NO_THREAD_SAFETY_ANALYSIS {
                auto it = p->first->pending_h2d_states_.find(p->second);
                if (it == p->first->pending_h2d_states_.end()) return true;
                return it->second.layer_futures.size() >=
                       it->second.expected_layers;
              };
      std::pair<WeightSynchronizerBase*, uint64_t> ctx{this, uuid};
      pending_h2d_mu_.Await(absl::Condition(condition_fn, &ctx));

      it = pending_h2d_states_.find(uuid);
      if (it != pending_h2d_states_.end()) {
        layer_futures_map = std::move(it->second.layer_futures);
        pending_h2d_states_.erase(it);
      }
    }
  }

  if (!layer_futures_map.empty()) {
    std::vector<raiden::PjRtCopyFuture> futures_to_await;
    futures_to_await.reserve(layer_futures_map.size());
    for (auto& [layer_idx, f] : layer_futures_map) {
      TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture layer_future, f.get());
      futures_to_await.push_back(std::move(layer_future));
    }
    raiden::PjRtCopyFuture joined_future =
        raiden::JoinPjRtCopyFutures(futures_to_await);
    TF_RETURN_IF_ERROR(joined_future.Await());
  } else {
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d(uuid));
    TF_RETURN_IF_ERROR(h2d_future.Await());
  }

  double h2d_time_ms = absl::ToDoubleMilliseconds(absl::Now() - h2d_start);
  {
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_h2d_time_ms = h2d_time_ms;
    metrics_.total_h2d_time_ms += h2d_time_ms;
  }
  record_completion();
  VLOG(1) << "Done with OnDataReceived (auto_h2d for uuid=" << uuid
          << ", h2d_time=" << h2d_time_ms << " ms).";
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::WaitForTransferCompletion(uint64_t uuid) {
  absl::MutexLock lock(completed_transfers_mu_);
  auto condition_fn =
      +[](std::pair<absl::flat_hash_set<uint64_t>*, uint64_t>* p)
           ABSL_NO_THREAD_SAFETY_ANALYSIS {
             return p->first->contains(p->second);
           };
  std::pair<absl::flat_hash_set<uint64_t>*, uint64_t> ctx{&completed_transfers_,
                                                          uuid};
  completed_transfers_mu_.Await(absl::Condition(condition_fn, &ctx));
  return absl::OkStatus();
}

void WeightSynchronizerBase::StoreSkipTiling(
    uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request) {
  std::vector<bool> skip(num_layers_, false);
  for (const auto& [layer_idx, skip_val] : request.skip_tiling()) {
    if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < num_layers_) {
      skip[layer_idx] = skip_val;
    }
  }
  VLOG(1) << "[WeightSynchronizerBase] StoreSkipTiling uuid=" << uuid
          << " with " << request.skip_tiling_size()
          << " entries in StartTransferRequest";
  absl::MutexLock lock(skip_tiling_mu_);
  latest_skip_tiling_ = skip;
  uuid_to_skip_tiling_[uuid] = std::move(skip);
}

absl::Status WeightSynchronizerBase::OnBlocksReceived(
    const std::vector<int>& block_ids, uint64_t uuid) {
  if (!auto_h2d_) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d(uuid));
  return h2d_future.Await();
}

void WeightSynchronizerBase::ForgetPushProgress(uint64_t uuid) {
  RaidenManagerBase::ForgetPushProgress(uuid);
  {
    absl::MutexLock lock(completed_transfers_mu_);
    completed_transfers_.erase(uuid);
  }
  {
    absl::MutexLock lock(skip_tiling_mu_);
    uuid_to_skip_tiling_.erase(uuid);
  }
  {
    absl::MutexLock lock(d2h_mu_);
    completed_d2h_uuids_.erase(uuid);
  }
  {
    absl::MutexLock lock(pending_h2d_mu_);
    pending_h2d_states_.erase(uuid);
  }
}

}  // namespace weight_sync
}  // namespace tpu_raiden

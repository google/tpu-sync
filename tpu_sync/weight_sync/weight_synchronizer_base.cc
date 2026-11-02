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
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "peregrine/src/api/v1/rate_limiter.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/weight_sync/tiling_utils.h"
#include "tpu_sync/weight_sync/weight_synchronizer_listener.h"

ABSL_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size, 256 * 1024,
          "Amount of scratchpad to allocate to host buffers for resharding "
          "pulls.");

namespace tpu_raiden {
namespace weight_sync {

namespace {
static std::atomic<size_t> global_allocated_host_dram_bytes_{0};

std::vector<transport::ShardInfo> BuildDefaultShards(size_t num_shards) {
  std::vector<transport::ShardInfo> shards;
  shards.reserve(num_shards);
  for (size_t sh = 0; sh < num_shards; ++sh) {
    shards.push_back({.local_index = static_cast<int>(sh),
                      .global_index = static_cast<int64_t>(sh),
                      .numa_node = -1});
  }
  return shards;
}
}  // namespace

WeightSynchronizerBase::WeightSynchronizerBase(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, int parallelism,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d)
    : tpu_raiden::RaidenManagerBase(
          layer_buffers.size(), raiden::DetectShards(layer_buffers),
          layer_buffers.empty() ? 0
                                : layer_buffers[0][0].GetOnDeviceSizeInBytes(),
          local_port, parallelism, bind_ip,
          raiden::DetectNumaNodes(layer_buffers), GetLocalHostNicAddresses(),
          /*socket_only_transport=*/true),
      auto_h2d_(auto_h2d),
      push_pool_(std::make_unique<tpu_raiden::NumaThreadPool>(
          std::max(parallelism_, 4))),
      shard_pool_(std::make_unique<tpu_raiden::NumaThreadPool>(
          std::max<size_t>(num_shards_, 1))) {
  if (layer_names.empty()) {
    layer_names_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
      layer_names_.push_back(absl::StrCat("weights_", i));
    }
  } else {
    layer_names_ = std::move(layer_names);
  }
}

absl::Status WeightSynchronizerBase::InitDeviceBuffers(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    const std::optional<std::vector<const uint8_t*>>& external_host_ptrs,
    std::optional<int> listener_port) {
  if (num_layers_ == 0 || num_shards_ == 0) {
    return absl::OkStatus();
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

  if (client) {
    auto alloc = HostMemoryAllocator::Create(client);
    if (alloc.ok()) {
      host_allocator_ = *std::move(alloc);
    }
  }
  HostMemoryAllocator* host_allocator = host_allocator_.get();

  size_t shard_idx = 0;
  std::vector<LayerInfoBase> layers;
  layers.reserve(num_layers_);
  std::vector<std::vector<raiden::BufferHoldAndAlias>> holds;
  holds.reserve(num_layers_);

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& dst_buffers = layer_buffers[layer_idx];
    if (dst_buffers.size() != num_shards_) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of shards mismatch across layers during weight sync init: "
          "layer ",
          layer_idx, " has ", dst_buffers.size(), " shards, expected ",
          num_shards_));
    }

    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& dst_buffer = dst_buffers[i];
      ShardBufferInfoBase shard_info;
      shard_info.device = dst_buffer.device;

      shard_info.device_size = dst_buffer.GetOnDeviceSizeInBytes();

      size_t alloc_size =
          shard_info.device_size +
          absl::GetFlag(FLAGS_raiden_weight_sync_host_buffer_scratchpad_size);
      if (external_host_ptrs.has_value()) {
        if (shard_idx >= external_host_ptrs->size()) {
          return absl::InvalidArgumentError(
              absl::StrCat("External host pointers size mismatch: got ",
                           external_host_ptrs->size(), " pointers for ",
                           num_layers_ * num_shards_, " shards"));
        }
        shard_info.host_ptr = (*external_host_ptrs)[shard_idx];
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
              return absl::ResourceExhaustedError(
                  absl::StrCat("Failed to allocate ", alloc_size,
                               " byte host weights buffer for layer ",
                               layer_idx, " shard ", i));
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

      const xla::Layout* xla_layout = nullptr;
      if (dst_buffer.shape.has_layout()) {
        xla_layout = &dst_buffer.shape.layout();
      }
      if (xla_layout && !xla_layout->tiles().empty()) {
        int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(
            dst_buffer.shape.element_type());
        size_t physical_bytes =
            tpu_raiden::weight_sync::GetTiledBufferElements(dst_buffer.shape) *
            itemsize;
        if (physical_bytes > 0) {
          shard_info.tiled_size = physical_bytes;
        }
      }

      hold_info.push_back(dst_buffer);
      layer_info.shards.push_back(std::move(shard_info));
    }
    layers.push_back(std::move(layer_info));
    holds.push_back(std::move(hold_info));
  }
  {
    absl::MutexLock lock(buffer_holds_mu_);
    buffer_holds_ = std::move(holds);
  }
  layers_ =
      StagingArena(std::move(layers),
                   host_allocator ? MakeHostBufferAllocator(host_allocator)
                                  : MakeDefaultHostBufferAllocator());

  FinishInit(listener_port);
  return absl::OkStatus();
}

WeightSynchronizerBase::WeightSynchronizerBase(
    size_t num_layers, size_t num_shards,
    const std::vector<size_t>& slice_byte_sizes, std::optional<int> local_port,
    int parallelism, std::optional<std::string> bind_ip,
    std::vector<std::string> layer_names, bool auto_h2d,
    std::vector<int> numa_nodes, std::vector<HostNicAddress> host_nics,
    std::vector<transport::ShardInfo> shards)
    : tpu_raiden::RaidenManagerBase(
          num_layers,
          shards.size() == num_shards ? std::move(shards)
                                      : BuildDefaultShards(num_shards),
          slice_byte_sizes.empty() ? 0 : slice_byte_sizes[0], local_port,
          parallelism, bind_ip, std::move(numa_nodes), std::move(host_nics),
          /*socket_only_transport=*/true),
      auto_h2d_(auto_h2d),
      push_pool_(std::make_unique<tpu_raiden::NumaThreadPool>(
          std::max(parallelism_, 4))),
      shard_pool_(std::make_unique<tpu_raiden::NumaThreadPool>(
          std::max<size_t>(num_shards_, 1))) {
  if (layer_names.empty()) {
    layer_names_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
      layer_names_.push_back(absl::StrCat("weights_", i));
    }
  } else {
    layer_names_ = std::move(layer_names);
  }
}

absl::Status WeightSynchronizerBase::InitHostBuffers(
    const std::vector<size_t>& slice_byte_sizes,
    std::optional<int> listener_port) {
  physical_size_ = slice_byte_size_;
  shard_factor_ = 1;
  major_dim_size_ = 1;

  std::vector<LayerInfoBase> layers;
  layers.reserve(num_layers_);
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
          return absl::ResourceExhaustedError(absl::StrCat(
              "Failed to allocate ", alloc_size,
              " byte host weights buffer for layer ", layer_idx, " shard ", i));
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
    layers.push_back(std::move(layer_info));
  }
  layers_ = StagingArena(std::move(layers), MakeDefaultHostBufferAllocator());

  FinishInit(listener_port);
  return absl::OkStatus();
}

void WeightSynchronizerBase::FinishInit(std::optional<int> listener_port) {
  if (listener_port) {
    listener_ =
        std::make_unique<WeightSynchronizerListener>(this, *listener_port);
  }
  if (auto_h2d_) {
    h2d_pool_ =
        std::make_unique<tpu_raiden::NumaThreadPool>(std::max(parallelism_, 4));
  }

  tiled_scratchpads_.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    tiled_scratchpads_.push_back(std::make_unique<ShardScratchpad>());
  }
  ClearExplicitD2hState();

  size_t init_host_bytes = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    for (const auto& shard : layers_[i].shards) {
      init_host_bytes += shard.host_size;
    }
  }
  if (init_host_bytes > 0) {
    UpdateAllocatedOccupancyMetric(init_host_bytes);
  }
}

absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>>
WeightSynchronizerBase::Create(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port,
    std::optional<std::vector<const uint8_t*>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    std::vector<std::string> layer_names, bool auto_h2d) {
  // The constructor sizes the synchronizer from layer_buffers[0][0], so every
  // layer must have at least one shard before it runs.
  for (size_t layer_idx = 0; layer_idx < layer_buffers.size(); ++layer_idx) {
    if (layer_buffers[layer_idx].empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "WeightSynchronizerBase::Create: layer ", layer_idx,
          " has no shards; every layer needs at least one device buffer"));
    }
  }
  auto ws = absl::WrapUnique(new WeightSynchronizerBase(
      layer_buffers, local_port, parallelism, std::move(bind_ip),
      std::move(layer_names), auto_h2d));
  ABSL_RETURN_IF_ERROR(
      ws->InitDeviceBuffers(layer_buffers, external_host_ptrs, listener_port));
  return ws;
}

absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>>
WeightSynchronizerBase::Create(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d) {
  return Create(num_layers, num_shards,
                std::vector<size_t>(num_layers, slice_byte_size), local_port,
                host_blocks_to_allocate, parallelism, listener_port,
                std::move(bind_ip), std::move(layer_names), auto_h2d);
}

absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>>
WeightSynchronizerBase::Create(
    size_t num_layers, size_t num_shards, std::vector<size_t> slice_byte_sizes,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d, std::vector<int> numa_nodes,
    std::vector<HostNicAddress> host_nics,
    std::vector<transport::ShardInfo> shards) {
  auto ws = absl::WrapUnique(new WeightSynchronizerBase(
      num_layers, num_shards, slice_byte_sizes, local_port, parallelism,
      std::move(bind_ip), std::move(layer_names), auto_h2d,
      std::move(numa_nodes), std::move(host_nics), std::move(shards)));
  ABSL_RETURN_IF_ERROR(ws->InitHostBuffers(slice_byte_sizes, listener_port));
  return ws;
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
  const int port = local_port().value_or(0);
  std::vector<RaidenTransferEndpoint> eps;
  for (const std::string& ip : local_ips()) {
    std::string ep = absl::StrContains(ip, ':')
                         ? absl::StrCat("[", ip, "]:", port)
                         : absl::StrCat(ip, ":", port);
    std::vector<int64_t> local_shards = shards_for_local_ip(ip);
    std::vector<int64_t> shards;
    shards.reserve(local_shards.size());
    for (int64_t local_sh : local_shards) {
      if (local_sh >= 0 &&
          static_cast<size_t>(local_sh) < global_shard_indices_.size()) {
        shards.push_back(global_shard_indices_[local_sh]);
      } else {
        shards.push_back(local_sh);
      }
    }
    eps.push_back(RaidenTransferEndpoint{std::move(ep), std::move(shards)});
  }
  return eps;
}

WeightSynchronizerBase::~WeightSynchronizerBase() {
  StopTransportServer();
  listener_.reset();
  {
    absl::MutexLock op_lock(d2h_op_mu_);
    WaitForInFlightD2h();
  }
  h2d_pool_.reset();
  shard_pool_.reset();
  push_pool_.reset();
  for (auto& sp : tiled_scratchpads_) {
    if (sp) {
      absl::MutexLock lock(sp->mu);
      if (sp->in_flight_future.IsValid()) {
        (void)sp->in_flight_future.Await();
      }
    }
  }
  size_t instance_bytes =
      allocated_host_dram_bytes_.exchange(0, std::memory_order_relaxed);
  if (instance_bytes > 0) {
    size_t prev_total = global_allocated_host_dram_bytes_.fetch_sub(
        instance_bytes, std::memory_order_relaxed);
    size_t total =
        (prev_total >= instance_bytes) ? (prev_total - instance_bytes) : 0;
    telemetry::RaidenMetricStore::GetGlobalMetricStore().SetGauge(
        telemetry::metric_names::kWeightSyncBufferAllocatedBytes, {},
        static_cast<double>(total));
  }
}

size_t WeightSynchronizerBase::GetPipelineGroupSize() const {
  if (pipeline_group_size_override_.has_value()) {
    return *pipeline_group_size_override_;
  }
  const char* env_val = std::getenv("RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE");
  if (env_val != nullptr) {
    int parsed_val = 0;
    if (absl::SimpleAtoi(env_val, &parsed_val) && parsed_val >= 0) {
      return static_cast<size_t>(parsed_val);
    }
  }
  return 1;
}

absl::StatusOr<uint8_t*> WeightSynchronizerBase::AcquireTiledScratchpadLocked(
    ShardScratchpad& sp, size_t required_bytes, const xla::PjRtDevice* device) {
  if (sp.in_flight_future.IsValid()) {
    TF_RETURN_IF_ERROR(sp.in_flight_future.Await());
  }
  if (sp.capacity < required_bytes) {
    const size_t old_capacity = sp.capacity;
    sp.ptr = nullptr;
    sp.owner.reset();
    sp.owned_buffer.reset();
    if (host_allocator_ && device) {
      auto alloc =
          host_allocator_->AllocateDmaMappedForDevice(required_bytes, device);
      if (alloc.ok()) {
        sp.ptr = (*alloc).ptr;
        sp.owner = (*alloc).owner;
        sp.capacity = required_bytes;
      }
    }
    if (sp.ptr == nullptr) {
      void* raw_ptr = nullptr;
      if (posix_memalign(&raw_ptr, 64, required_bytes) != 0) {
        return absl::ResourceExhaustedError(
            "Failed to allocate host tiled scratch buffer");
      }
      std::memset(raw_ptr, 0, required_bytes);
      sp.owned_buffer = std::unique_ptr<uint8_t[], void (*)(void*)>(
          static_cast<uint8_t*>(raw_ptr), [](void* p) { free(p); });
      sp.ptr = sp.owned_buffer.get();
      sp.capacity = required_bytes;
    }
    if (required_bytes > old_capacity) {
      UpdateAllocatedOccupancyMetric(required_bytes - old_capacity);
    }
  }
  return sp.ptr;
}

void WeightSynchronizerBase::UpdateAllocatedOccupancyMetric(size_t delta) {
  allocated_host_dram_bytes_ += delta;
  size_t total = global_allocated_host_dram_bytes_.fetch_add(
                     delta, std::memory_order_relaxed) +
                 delta;
  telemetry::RaidenMetricStore::GetGlobalMetricStore().SetGauge(
      telemetry::metric_names::kWeightSyncBufferAllocatedBytes, {},
      static_cast<double>(total));
}

std::vector<bool> WeightSynchronizerBase::GetActiveSkipTiling(uint64_t uuid) {
  absl::MutexLock lock(skip_tiling_mu_);
  if (uuid != 0) {
    auto it = uuid_to_skip_tiling_.find(uuid);
    if (it != uuid_to_skip_tiling_.end() && !it->second.empty()) {
      return it->second;
    }
  }
  if (!latest_skip_tiling_.empty()) {
    return latest_skip_tiling_;
  }
  return std::vector<bool>(num_layers_, false);
}

namespace {

bool HasTiledLayout(const xla::Shape& shape) {
  return shape.has_layout() && !shape.layout().tiles().empty();
}

// Joins `futures` and keeps `hold` (including c_hold, common_hold, and
// common_raw_buffer) alive only until `futures` settle, dropping `hold`
// strictly before the returned status-only future resolves.
xla::Future<> ReleaseBufferHoldOnCompletion(std::vector<xla::Future<>> futures,
                                            raiden::BufferHoldAndAlias hold) {
  xla::Future<> join_future = xla::JoinFutures(absl::MakeSpan(futures));
  if (!join_future.IsValid()) {
    return xla::Future<>(absl::OkStatus());
  }
  auto [promise, done] = xla::MakePromise();
  auto shared_hold =
      std::make_shared<raiden::BufferHoldAndAlias>(std::move(hold));
  std::move(join_future)
      .OnReady(
          [promise = std::move(promise),
           shared_hold = std::move(shared_hold)](absl::Status status) mutable {
            shared_hold.reset();
            promise.Set(std::move(status));
          });
  return done;
}

std::vector<bool> ExtractRequestSkipTiling(
    size_t num_layers, const tpu_sync::rpc::StartTransferRequest& request) {
  std::vector<bool> skip(num_layers, false);
  for (const auto& [layer_idx, skip_val] : request.skip_tiling()) {
    if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < num_layers) {
      skip[layer_idx] = skip_val;
    }
  }
  return skip;
}

// Returns `copy` with `on_success` run once it completes successfully. When
// `copy` completes through its xla::Future (every copy issued by this file),
// `on_success` runs before the returned future becomes ready, so a caller that
// awaits the result observes everything `on_success` did. A failed copy skips
// `on_success`.
raiden::PjRtCopyFuture RunOnCopySuccess(raiden::PjRtCopyFuture copy,
                                        absl::AnyInvocable<void()> on_success) {
  if (!copy.IsValid()) {
    // Nothing in flight, e.g. no device buffers: the copy is already done.
    on_success();
    return copy;
  }
  if (copy.future.IsValid()) {
    auto [promise, done] = xla::MakePromise();
    std::move(copy.future)
        .OnReady([promise = std::move(promise),
                  on_success = std::move(on_success)](absl::Status s) mutable {
          if (s.ok()) on_success();
          promise.Set(std::move(s));
        });
    copy.future = std::move(done);
    return copy;
  }
  // PjRt-event-only completion: `on_success` runs from the event callback.
  copy.OnReady(
      [on_success = std::move(on_success)](
          const absl::StatusOr<raiden::BufferHolders>& result) mutable {
        if (result.ok()) on_success();
      });
  return copy;
}

}  // namespace

void WeightSynchronizerBase::ClearExplicitD2hState() {
  absl::MutexLock lock(explicit_d2h_mu_);
  explicit_d2h_ready_.assign(num_layers_, nullptr);
  explicit_d2h_skip_tiling_.assign(num_layers_, std::nullopt);
}

void WeightSynchronizerBase::ClearExplicitD2hLayer(size_t layer_idx) {
  absl::MutexLock lock(explicit_d2h_mu_);
  if (layer_idx < explicit_d2h_ready_.size()) {
    explicit_d2h_ready_[layer_idx] = nullptr;
  }
  if (layer_idx < explicit_d2h_skip_tiling_.size()) {
    explicit_d2h_skip_tiling_[layer_idx] = std::nullopt;
  }
}

void WeightSynchronizerBase::TrackInFlightD2h(xla::Future<> future) {
  if (!future.IsValid()) return;
  absl::MutexLock lock(explicit_d2h_mu_);
  if (in_flight_d2h_.IsValid() && !in_flight_d2h_.IsReady()) {
    std::array<xla::Future<>, 2> both = {std::move(in_flight_d2h_),
                                         std::move(future)};
    in_flight_d2h_ = xla::JoinFutures(absl::MakeSpan(both));
  } else {
    in_flight_d2h_ = std::move(future);
  }
}

void WeightSynchronizerBase::WaitForInFlightD2h() const {
  xla::Future<> in_flight;
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    in_flight = in_flight_d2h_;
  }
  if (in_flight.IsValid()) {
    (void)in_flight.Await();
  }
}

absl::Status WeightSynchronizerBase::CheckNoActiveH2hPushes(
    absl::string_view caller) const {
  absl::MutexLock lock(metrics_mu_);
  if (active_h2h_pushes_ == 0) return absl::OkStatus();
  return absl::FailedPreconditionError(absl::StrCat(
      "WeightSynchronizer cannot call ", caller, " while ", active_h2h_pushes_,
      " H2H transfer push(es) are reading host staging buffers "
      "(active_h2h_pushes=",
      active_h2h_pushes_,
      "); await the in-flight transfer before bind_weights()/d2h() for the "
      "next sync."));
}

absl::Status WeightSynchronizerBase::WaitForExplicitD2h(
    size_t layer_idx) const {
  std::shared_ptr<LayerTileReady> ready;
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    if (layer_idx < explicit_d2h_ready_.size()) {
      ready = explicit_d2h_ready_[layer_idx];
    }
  }
  if (ready != nullptr) {
    return ready->Wait();
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::WaitForAllExplicitD2h() const {
  std::vector<std::shared_ptr<LayerTileReady>> ready_list;
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    ready_list = explicit_d2h_ready_;
  }
  for (const std::shared_ptr<LayerTileReady>& ready : ready_list) {
    if (ready != nullptr) {
      ABSL_RETURN_IF_ERROR(ready->Wait());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<const uint8_t*> WeightSynchronizerBase::WaitAndGetHostBufferPtr(
    size_t layer_idx, size_t shard_idx) const {
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return absl::OutOfRangeError(
        absl::StrCat("Host buffer index out of range: layer_idx=", layer_idx,
                     " (num_layers=", num_layers_, "), shard_idx=", shard_idx,
                     " (num_shards=", num_shards_, ")"));
  }
  ABSL_RETURN_IF_ERROR(WaitForExplicitD2h(layer_idx));
  const uint8_t* host_ptr = layers_[layer_idx].shards[shard_idx].host_ptr;
  if (host_ptr == nullptr) {
    return absl::InternalError(
        absl::StrCat("Host buffer pointer is null for layer_idx=", layer_idx,
                     ", shard_idx=", shard_idx));
  }
  return host_ptr;
}

size_t WeightSynchronizerBase::LayerDeviceBytes(size_t layer_idx) const {
  if (layer_idx >= layers_.size()) return 0;
  size_t bytes = 0;
  for (const auto& shard : layers_[layer_idx].shards) {
    bytes += shard.device_size;
  }
  return bytes;
}

absl::StatusOr<WeightSynchronizerBase::ShardScratchpad*>
WeightSynchronizerBase::GetTiledScratchpad(size_t shard_idx) {
  if (shard_idx >= tiled_scratchpads_.size() ||
      !tiled_scratchpads_[shard_idx]) {
    return absl::InternalError("Shard index out of range for scratchpad");
  }
  return tiled_scratchpads_[shard_idx].get();
}

absl::StatusOr<absl::Span<const uint8_t>>
WeightSynchronizerBase::TileShardLocked(
    size_t layer_idx, size_t shard_idx,
    const raiden::BufferHoldAndAlias& shard_hold, ShardScratchpad& sp) {
  const auto& shard_info = layers_[layer_idx].shards[shard_idx];
  const size_t physical_bytes =
      tpu_raiden::weight_sync::GetTiledBufferElements(shard_hold.shape) *
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shard_hold.shape.element_type());
  ABSL_ASSIGN_OR_RETURN(
      uint8_t* tiled_buffer_ptr,
      AcquireTiledScratchpadLocked(sp, physical_bytes, shard_hold.device));
  if (tiled_buffer_ptr == nullptr) {
    return absl::InternalError("Tiled buffer pointer is null for tiled shape");
  }
  const absl::Time tile_start = absl::Now();
  ABSL_RETURN_IF_ERROR(tpu_raiden::weight_sync::TileBuffer(
      shard_info.host_ptr, tiled_buffer_ptr, shard_hold.shape,
      shard_hold.shape.layout(), shard_numa_node(shard_idx)));
  const double tile_time_ms =
      absl::ToDoubleMilliseconds(absl::Now() - tile_start);
  {
    absl::MutexLock metrics_lock(metrics_mu_);
    metrics_.last_tiling_time_ms =
        std::max(metrics_.last_tiling_time_ms, tile_time_ms);
    metrics_.total_tiling_time_ms += tile_time_ms;
    metrics_.last_tiled_bytes += physical_bytes;
    metrics_.total_tiled_bytes += physical_bytes;
  }
  return absl::Span<const uint8_t>(tiled_buffer_ptr, physical_bytes);
}

absl::StatusOr<xla::Future<raiden::BufferHolder>>
WeightSynchronizerBase::H2dShard(size_t layer_idx, size_t shard_idx,
                                 bool tile) {
  const auto& shard_info = layers_[layer_idx].shards[shard_idx];
  raiden::BufferHoldAndAlias shard_hold;
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::FailedPreconditionError(
          "WeightSynchronizer device weights are unbound; call bind_weights() "
          "before H2d.");
    }
    if (layer_idx >= buffer_holds_.size() ||
        shard_idx >= buffer_holds_[layer_idx].size()) {
      return absl::InternalError(absl::StrCat("H2dShard index out of bounds: (",
                                              layer_idx, ", ", shard_idx, ")"));
    }
    shard_hold = buffer_holds_[layer_idx][shard_idx];
  }
  const bool is_tiled = tile && HasTiledLayout(shard_hold.shape);
  VLOG(1) << "[WeightSynchronizerBase] H2dLayer " << layer_idx << " shard "
          << shard_idx << " (layer: "
          << (layer_idx < layer_names_.size() ? layer_names_[layer_idx]
                                              : "unknown")
          << ", is_tiled=" << is_tiled << ", tile=" << tile
          << ", shape=" << shard_hold.shape.ToString()
          << ", size=" << shard_info.device_size << " bytes)";

  std::vector<xla::Future<>> shard_futures;
  if (is_tiled) {
    ABSL_ASSIGN_OR_RETURN(ShardScratchpad * sp, GetTiledScratchpad(shard_idx));
    absl::MutexLock lock(sp->mu);
    ABSL_ASSIGN_OR_RETURN(
        absl::Span<const uint8_t> tiled,
        TileShardLocked(layer_idx, shard_idx, shard_hold, *sp));
    // The scratchpad is reused once this copy completes; see
    // AcquireTiledScratchpadLocked().
    xla::Future<> future =
        shard_hold.CopyRawHostToDevice(tiled.data(), 0, tiled.size());
    sp->in_flight_future = future;
    shard_futures.push_back(std::move(future));
  } else {
    shard_futures.push_back(shard_hold.CopyRawHostToDevice(
        shard_info.host_ptr, 0, shard_info.device_size));
  }
  return raiden::CreateBufferFuture(std::move(shard_futures), shard_hold.c_hold,
                                    shard_hold.common_hold);
}

bool WeightSynchronizerBase::IsTilingSkipped(size_t layer_idx, uint64_t uuid) {
  const std::vector<bool> skip = GetActiveSkipTiling(uuid);
  return layer_idx < skip.size() && skip[layer_idx];
}

tpu_sync::rpc::HostTilingMode WeightSynchronizerBase::GetReceiverTiling(
    uint64_t uuid) {
  absl::MutexLock lock(skip_tiling_mu_);
  auto it = receiver_tiling_.find(uuid);
  return it == receiver_tiling_.end() ? tpu_sync::rpc::HOST_TILING_MODE_DEFERRED
                                      : it->second;
}

absl::Status WeightSynchronizerBase::TileLayer(size_t layer_idx,
                                               uint64_t uuid) {
  if (layer_idx >= num_layers_ || IsTilingSkipped(layer_idx, uuid)) {
    return absl::OkStatus();
  }
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (buffer_holds_.empty()) {
      return absl::OkStatus();
    }
  }
  // Await: the receive coordinator issues this layer's H2D right after.
  return ForEachShardNumaLocal(
             *shard_pool_,
             [this, layer_idx](size_t i) -> xla::Future<> {
               raiden::BufferHoldAndAlias shard_hold;
               {
                 absl::MutexLock lock(buffer_holds_mu_);
                 if (weights_unbound_) {
                   return absl::FailedPreconditionError(
                       "WeightSynchronizer device weights are unbound; call "
                       "bind_weights() before TileLayer.");
                 }
                 if (layer_idx >= buffer_holds_.size() ||
                     i >= buffer_holds_[layer_idx].size()) {
                   return absl::OkStatus();
                 }
                 shard_hold = buffer_holds_[layer_idx][i];
               }
               if (!HasTiledLayout(shard_hold.shape)) return absl::OkStatus();
               ABSL_ASSIGN_OR_RETURN(ShardScratchpad * sp,
                                     GetTiledScratchpad(i));
               absl::MutexLock lock(sp->mu);
               ABSL_ASSIGN_OR_RETURN(
                   absl::Span<const uint8_t> tiled,
                   TileShardLocked(layer_idx, i, shard_hold, *sp));
               std::memcpy(
                   const_cast<uint8_t*>(layers_[layer_idx].shards[i].host_ptr),
                   tiled.data(), tiled.size());
               return absl::OkStatus();
             })
      .Await();
}

void WeightSynchronizerBase::LayerTileReady::Set(absl::Status status) {
  absl::MutexLock lock(mu_);
  if (done_) return;
  status_ = std::move(status);
  done_ = true;
}

absl::Status WeightSynchronizerBase::LayerTileReady::Wait() {
  absl::MutexLock lock(mu_);
  while (!mu_.AwaitWithTimeout(absl::Condition(&done_), absl::Seconds(60))) {
    LOG(WARNING) << "Still waiting for a received layer to be ready on host";
  }
  return status_;
}

bool WeightSynchronizerBase::LayerTileReady::IsSet() {
  absl::MutexLock lock(mu_);
  return done_;
}

void WeightSynchronizerBase::SetLayerTileReady(
    size_t layer_idx, std::shared_ptr<LayerTileReady> ready) {
  if (layer_idx >= num_layers_) return;
  absl::MutexLock lock(tile_ready_mu_);
  layer_tile_ready_.resize(num_layers_);
  layer_tile_ready_[layer_idx] = std::move(ready);
}

void WeightSynchronizerBase::MarkHostBufferUntiled() {
  absl::MutexLock lock(tile_ready_mu_);
  layer_tile_ready_.assign(num_layers_, nullptr);
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2dLayer(
    size_t layer_idx, bool tile) {
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::FailedPreconditionError(
          "WeightSynchronizer device weights are unbound; call bind_weights() "
          "before H2d.");
    }
    if (buffer_holds_.empty() || layer_idx >= num_layers_) {
      return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
    }
  }
  return raiden::PjRtCopyFuture::FromFuture(
      ForEachShardNumaLocal<raiden::BufferHolder>(
          *shard_pool_,
          [this, layer_idx,
           tile](size_t i) -> xla::Future<raiden::BufferHolder> {
            ABSL_ASSIGN_OR_RETURN(xla::Future<raiden::BufferHolder> future,
                                  H2dShard(layer_idx, i, tile));
            return future;
          }));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2d(
    uint64_t uuid) {
  RAIDEN_TRACE("WeightSynchronizerBase::H2d");
  ABSL_RETURN_IF_ERROR(RefreshDeviceBuffers());
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::FailedPreconditionError(
          "WeightSynchronizer device weights are unbound; call bind_weights() "
          "before H2d.");
    }
    if (buffer_holds_.empty()) {
      return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
    }
  }
  ABSL_RETURN_IF_ERROR(WaitForAllExplicitD2h());
  const auto start_time = absl::Now();
  std::vector<std::shared_ptr<LayerTileReady>> tile_ready;
  {
    absl::MutexLock lock(tile_ready_mu_);
    tile_ready = layer_tile_ready_;
  }
  tile_ready.resize(num_layers_);
  bool any_in_place = false;
  for (const std::shared_ptr<LayerTileReady>& ready : tile_ready) {
    any_in_place = any_in_place || ready != nullptr;
  }
  VLOG(1) << "Starting H2d across " << num_layers_ << " layers (uuid=" << uuid
          << ", any_in_place=" << any_in_place << ")...";
  if (!any_in_place) {
    // All layers are tiled below; otherwise keep the metrics recorded while
    // tiling in place on receive.
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_tiling_time_ms = 0.0;
    metrics_.last_tiled_bytes = 0;
  }
  // Layers whose host buffers are (or are about to be) in device layout are
  // copied as is; the rest are tiled into the scratchpads unless skipped.
  const std::vector<bool> active_skip = GetActiveSkipTiling(uuid);
  std::vector<bool> tile_layer(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    tile_layer[layer_idx] =
        tile_ready[layer_idx] == nullptr &&
        !(layer_idx < active_skip.size() && active_skip[layer_idx]);
  }
  // The transport acks before OnLayerDataReceived runs, so a layer's receive
  // callback (and in-place tiling) may still be in flight. Wait here on the
  // caller: that tiling runs on the shard pool used below, so a shard-pool
  // task must never block on it.
  for (const std::shared_ptr<LayerTileReady>& ready : tile_ready) {
    if (ready != nullptr) {
      TF_RETURN_IF_ERROR(ready->Wait());
    }
  }
  using LayerCopies = xla::Future<std::vector<raiden::BufferHolder>>;
  // Each shard has its own device and tiling scratchpad, so shards are tiled
  // and copied concurrently on their own NUMA node. Layers of a shard stay in
  // order because they share the scratchpad.
  raiden::PjRtCopyFuture joined = raiden::PjRtCopyFuture::FromFuture(
      ForEachShardNumaLocal<std::vector<raiden::BufferHolder>>(
          *shard_pool_,
          [this, tile_layer =
                     std::move(tile_layer)](size_t shard_idx) -> LayerCopies {
            std::vector<xla::Future<raiden::BufferHolder>> futures;
            futures.reserve(num_layers_);
            for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
              TF_ASSIGN_OR_RETURN(
                  xla::Future<raiden::BufferHolder> future,
                  H2dShard(layer_idx, shard_idx, tile_layer[layer_idx]));
              futures.push_back(std::move(future));
            }
            return xla::JoinFutures(absl::MakeSpan(futures));
          }));
  VLOG(1) << "Done with scheduling H2d across " << num_layers_ << " layers x "
          << num_shards_ << " shards (uuid=" << uuid << ") in "
          << absl::ToDoubleMilliseconds(absl::Now() - start_time) << " ms.";
  size_t h2d_bytes = 0;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    h2d_bytes += LayerDeviceBytes(layer_idx);
  }
  // `this` outlives in-flight copies; see the telemetry callback below.
  joined = RunOnCopySuccess(std::move(joined), [this, start_time, h2d_bytes]() {
    const double h2d_time_ms =
        absl::ToDoubleMilliseconds(absl::Now() - start_time);
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_h2d_time_ms = h2d_time_ms;
    metrics_.total_h2d_time_ms += h2d_time_ms;
    metrics_.last_h2d_bytes = h2d_bytes;
    metrics_.total_h2d_bytes += h2d_bytes;
  });
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (store.HasBackends()) {
    // Tiling runs on the shard pool after this returns, so sample its metric
    // once the copies are done. `this` outlives in-flight copies.
    joined.OnReady([this, start_time, uuid](const auto& result) {
      if (result.ok()) {
        auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
        store.ObserveHistogram(
            telemetry::metric_names::kWeightSyncH2dTransferTimeMs, {},
            absl::ToDoubleMilliseconds(absl::Now() - start_time));
        double tiling_time_ms = 0.0;
        {
          absl::MutexLock lock(metrics_mu_);
          tiling_time_ms = metrics_.last_tiling_time_ms;
        }
        if (tiling_time_ms > 0.0) {
          store.ObserveHistogram(
              telemetry::metric_names::kWeightSyncTilingTimeMs, {},
              tiling_time_ms);
        }
        store.FlushToCloudLogging("destination_h2d", uuid);
      }
    });
  }
  return joined;
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::D2hLayer(
    size_t layer_idx, uint64_t uuid,
    std::shared_ptr<std::atomic<double>> max_detile_ms) {
  const std::vector<bool> active_skip = GetActiveSkipTiling(uuid);
  const bool skip_tiling =
      layer_idx < active_skip.size() && active_skip[layer_idx];
  return D2hLayerWithSkipTiling(layer_idx, skip_tiling,
                                std::move(max_detile_ms));
}

absl::StatusOr<raiden::PjRtCopyFuture>
WeightSynchronizerBase::D2hLayerWithSkipTiling(
    size_t layer_idx, bool skip_tiling,
    std::shared_ptr<std::atomic<double>> max_detile_ms) {
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::FailedPreconditionError(
          "WeightSynchronizer device weights are unbound; call bind_weights() "
          "before D2h.");
    }
    if (buffer_holds_.empty() || layer_idx >= num_layers_) {
      return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
    }
  }
  // Each shard's D2H copy is issued from a thread pinned to that shard's NUMA
  // node, matching where its host staging buffer and scratchpad live. Each
  // shard releases its device BufferHolder as soon as its copy+detile settles,
  // so the returned PjRtCopyFuture carries no device holds.
  xla::Future<> shard_copies = ForEachShardNumaLocal(
      *shard_pool_,
      [this, layer_idx, skip_tiling,
       max_detile_ms = std::move(max_detile_ms)](size_t shard_idx) {
        return D2hShard(layer_idx, shard_idx, skip_tiling, max_detile_ms);
      });
  return raiden::PjRtCopyFuture(std::move(shard_copies),
                                raiden::BufferHolders{});
}

xla::Future<> WeightSynchronizerBase::D2hShard(
    size_t layer_idx, size_t shard_idx, bool skip_tiling,
    std::shared_ptr<std::atomic<double>> max_detile_ms) {
  auto& layer_info = layers_[layer_idx];
  auto& shard_info = layer_info.shards[shard_idx];
  raiden::BufferHoldAndAlias shard_hold;
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::FailedPreconditionError(
          "WeightSynchronizer device weights are unbound; call bind_weights() "
          "before D2h.");
    }
    if (layer_idx >= buffer_holds_.size() ||
        shard_idx >= buffer_holds_[layer_idx].size()) {
      return absl::InternalError(absl::StrCat("D2hShard index out of bounds: (",
                                              layer_idx, ", ", shard_idx, ")"));
    }
    shard_hold = buffer_holds_[layer_idx][shard_idx];
  }
  uint8_t* dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

  const xla::Layout* xla_layout = nullptr;
  if (shard_hold.shape.has_layout()) {
    xla_layout = &shard_hold.shape.layout();
  }
  bool is_tiled = xla_layout && !xla_layout->tiles().empty();
  if (skip_tiling) {
    is_tiled = false;
  }
  VLOG(1) << "[WeightSynchronizerBase] D2hShard " << layer_idx << " shard "
          << shard_idx << " (layer: "
          << (layer_idx < layer_names_.size() ? layer_names_[layer_idx]
                                              : "unknown")
          << ", is_tiled=" << is_tiled << ", skip_tiling=" << skip_tiling
          << ", shape=" << shard_hold.shape.ToString()
          << ", size=" << shard_info.device_size << " bytes)";

  std::vector<xla::Future<>> shard_futures;
  if (is_tiled) {
    int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(
        shard_hold.shape.element_type());
    size_t physical_bytes =
        tpu_raiden::weight_sync::GetTiledBufferElements(shard_hold.shape) *
        itemsize;

    if (shard_idx >= tiled_scratchpads_.size() ||
        !tiled_scratchpads_[shard_idx]) {
      return absl::InternalError("Shard index out of range for scratchpad");
    }
    auto& sp = *tiled_scratchpads_[shard_idx];
    absl::MutexLock lock(sp.mu);
    ABSL_ASSIGN_OR_RETURN(
        uint8_t* tiled_buffer_ptr,
        AcquireTiledScratchpadLocked(sp, physical_bytes, shard_hold.device));
    if (tiled_buffer_ptr == nullptr) {
      return absl::InternalError(
          "Tiled buffer pointer is null for tiled shape");
    }

    xla::Future<> copy_future =
        shard_hold.CopyRawDeviceToHost(tiled_buffer_ptr, 0, physical_bytes);

    xla::Future<> detile_future = copy_future.Map(
        [this, tiled_buffer_ptr, dst_host_ptr, shape = shard_hold.shape,
         layout = *xla_layout, physical_bytes, max_detile_ms,
         numa_node = shard_numa_node(shard_idx)]() -> absl::Status {
          auto detile_start = absl::Now();
          absl::Status status = tpu_raiden::weight_sync::DetileBuffer(
              tiled_buffer_ptr, dst_host_ptr, shape, layout, numa_node);
          double detile_time_ms =
              absl::ToDoubleMilliseconds(absl::Now() - detile_start);
          if (status.ok()) {
            if (max_detile_ms != nullptr) {
              double current = max_detile_ms->load(std::memory_order_relaxed);
              while (detile_time_ms > current &&
                     !max_detile_ms->compare_exchange_weak(
                         current, detile_time_ms, std::memory_order_relaxed)) {
              }
            }
            absl::MutexLock metrics_lock(metrics_mu_);
            metrics_.last_detiling_time_ms =
                std::max(metrics_.last_detiling_time_ms, detile_time_ms);
            metrics_.total_detiling_time_ms += detile_time_ms;
            metrics_.last_detiled_bytes += physical_bytes;
            metrics_.total_detiled_bytes += physical_bytes;
          }
          return status;
        });

    sp.in_flight_future = detile_future;
    shard_futures.push_back(std::move(detile_future));
  } else {
    xla::Future<> future =
        shard_hold.CopyRawDeviceToHost(dst_host_ptr, 0, shard_info.device_size);
    shard_futures.push_back(std::move(future));
  }
  return ReleaseBufferHoldOnCompletion(std::move(shard_futures),
                                       std::move(shard_hold));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::D2h(
    uint64_t uuid) {
  RAIDEN_TRACE("WeightSynchronizerBase::D2h");
  absl::MutexLock op_lock(d2h_op_mu_);
  WaitForInFlightD2h();
  ABSL_RETURN_IF_ERROR(CheckNoActiveH2hPushes("D2h()"));
  ABSL_RETURN_IF_ERROR(RefreshDeviceBuffers());
  {
    absl::MutexLock lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::FailedPreconditionError(
          "WeightSynchronizer device weights are unbound; call bind_weights() "
          "before D2h.");
    }
    if (buffer_holds_.empty()) {
      return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
    }
  }
  const auto start_time = absl::Now();
  VLOG(1) << "Starting D2h across " << num_layers_ << " layers (uuid=" << uuid
          << ")...";
  const std::vector<bool> active_skip = GetActiveSkipTiling(uuid);
  // D2h writes logical-layout (or skip_tiling raw) bytes into the host staging
  // buffers, replacing any receiver in-place tiled state.
  MarkHostBufferUntiled();
  {
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_detiling_time_ms = 0.0;
    metrics_.last_detiled_bytes = 0;
  }
  std::vector<std::shared_ptr<LayerTileReady>> layer_ready;
  layer_ready.reserve(num_layers_);
  for (size_t l = 0; l < num_layers_; ++l) {
    layer_ready.push_back(std::make_shared<LayerTileReady>());
  }
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    explicit_d2h_ready_ = layer_ready;
    explicit_d2h_skip_tiling_.resize(num_layers_);
    for (size_t l = 0; l < num_layers_; ++l) {
      explicit_d2h_skip_tiling_[l] = (l < active_skip.size() && active_skip[l]);
    }
  }
  auto max_detile_ms = std::make_shared<std::atomic<double>>(0.0);
  std::vector<raiden::PjRtCopyFuture> layer_futures;
  layer_futures.reserve(num_layers_);
  size_t scheduled_layers = 0;
  absl::Status schedule_error;
  auto partial_cleanup = absl::MakeCleanup([&]() {
    if (!schedule_error.ok()) {
      for (size_t l = scheduled_layers; l < layer_ready.size(); ++l) {
        layer_ready[l]->Set(schedule_error);
      }
      if (!layer_futures.empty()) {
        TrackInFlightD2h(raiden::JoinPjRtCopyFutures(layer_futures).future);
      }
    }
  });
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const bool layer_skip =
        layer_idx < active_skip.size() && active_skip[layer_idx];
    absl::StatusOr<raiden::PjRtCopyFuture> layer_copy =
        D2hLayerWithSkipTiling(layer_idx, layer_skip, max_detile_ms);
    if (!layer_copy.ok()) {
      schedule_error = layer_copy.status();
      return schedule_error;
    }
    ++scheduled_layers;
    std::shared_ptr<LayerTileReady> ready = layer_ready[layer_idx];
    if (layer_copy->future.IsValid()) {
      auto [promise, gated_future] = xla::MakePromise();
      std::move(layer_copy->future)
          .OnReady([ready = std::move(ready),
                    promise = std::move(promise)](absl::Status s) mutable {
            ready->Set(s);
            promise.Set(std::move(s));
          });
      layer_futures.emplace_back(std::move(gated_future),
                                 raiden::BufferHolders{});
    } else {
      ready->Set(absl::OkStatus());
      layer_futures.push_back(*std::move(layer_copy));
    }
  }
  VLOG(1) << "Done with scheduling D2h across " << num_layers_
          << " layers (uuid=" << uuid << ").";
  raiden::PjRtCopyFuture joined = raiden::JoinPjRtCopyFutures(layer_futures);
  size_t d2h_bytes = 0;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    d2h_bytes += LayerDeviceBytes(layer_idx);
  }
  // `this` outlives in-flight copies; ~WeightSynchronizerBase awaits
  // `in_flight_d2h_`.
  joined = RunOnCopySuccess(std::move(joined), [this, start_time, d2h_bytes]() {
    const double d2h_time_ms =
        absl::ToDoubleMilliseconds(absl::Now() - start_time);
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_d2h_time_ms = d2h_time_ms;
    metrics_.total_d2h_time_ms += d2h_time_ms;
    metrics_.last_d2h_bytes = d2h_bytes;
    metrics_.total_d2h_bytes += d2h_bytes;
    metrics_.d2h_call_count++;
  });
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (store.HasBackends()) {
    joined.OnReady([start_time, max_detile_ms](const auto& result) {
      if (result.ok()) {
        auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
        store.ObserveHistogram(
            telemetry::metric_names::kWeightSyncD2hTransferTimeMs, {},
            absl::ToDoubleMilliseconds(absl::Now() - start_time));
        double detiling_time_ms =
            max_detile_ms->load(std::memory_order_relaxed);
        if (detiling_time_ms > 0.0) {
          store.ObserveHistogram(
              telemetry::metric_names::kWeightSyncDetilingTimeMs, {},
              detiling_time_ms);
        }
      }
    });
  }
  TrackInFlightD2h(joined.future);
  return joined;
}

absl::Status WeightSynchronizerBase::PushWeights(
    const std::vector<std::string>& peers) {
  RAIDEN_TRACE("WeightSynchronizerBase::PushWeights");
  return PushWeightsLocal(peers);
}

absl::Status WeightSynchronizerBase::PushWeightsLocal(
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
  RAIDEN_TRACE("WeightSynchronizerBase::PushWeightsResharded");
  return PushWeightsReshardedLocal(request);
}

absl::StatusOr<std::vector<WeightSynchronizerBase::ShardPushSchedule>>
WeightSynchronizerBase::ResolveShardPushSchedules(
    const tpu_sync::rpc::StartTransferRequest& request) {
  const auto& schedules = request.shard_push_schedules();
  // Schedules are keyed by local shard index unless some key falls outside
  // the local index space, in which case they are keyed by global index. An
  // empty `local_shard_indices_` denotes the identity mapping [0, num_shards_).
  bool use_global_keys = false;
  if (!global_shard_indices_.empty()) {
    for (const auto& [sched_key, _] : schedules) {
      bool in_local;
      if (local_shard_indices_.empty()) {
        in_local =
            sched_key >= 0 && static_cast<size_t>(sched_key) < num_shards_;
      } else {
        in_local =
            std::find(local_shard_indices_.begin(), local_shard_indices_.end(),
                      static_cast<int>(sched_key)) !=
            local_shard_indices_.end();
      }
      if (!in_local) {
        use_global_keys = true;
        break;
      }
    }
  }
  std::vector<int32_t> sorted_sliced_keys;
  if (schedules.size() == num_shards_ && num_shards_ > 0) {
    bool is_default_zero_based = true;
    bool any_direct_match = false;
    for (size_t i = 0; i < num_shards_; ++i) {
      int64_t g_sh = global_shard_index(i);
      int64_t l_sh = local_shard_index(i);
      if (g_sh != static_cast<int64_t>(i) || l_sh != static_cast<int64_t>(i)) {
        is_default_zero_based = false;
        break;
      }
      if (schedules.contains(static_cast<int32_t>(l_sh)) ||
          schedules.contains(static_cast<int32_t>(g_sh))) {
        any_direct_match = true;
        break;
      }
    }
    if (is_default_zero_based && !any_direct_match) {
      sorted_sliced_keys.reserve(schedules.size());
      for (const auto& [sched_key, _] : schedules) {
        sorted_sliced_keys.push_back(sched_key);
      }
      std::sort(sorted_sliced_keys.begin(), sorted_sliced_keys.end());
    }
  }
  if (global_shard_indices_.empty() && sorted_sliced_keys.empty()) {
    std::vector<int32_t> offending_keys;
    for (const auto& [sched_key, _] : schedules) {
      bool in_local = false;
      if (!local_shard_indices_.empty()) {
        in_local =
            std::find(local_shard_indices_.begin(), local_shard_indices_.end(),
                      static_cast<int>(sched_key)) !=
            local_shard_indices_.end();
      } else {
        in_local =
            (sched_key >= 0 && static_cast<size_t>(sched_key) < num_shards_);
      }
      if (!in_local) {
        offending_keys.push_back(sched_key);
      }
    }
    if (!offending_keys.empty()) {
      std::sort(offending_keys.begin(), offending_keys.end());
      return absl::FailedPreconditionError(absl::StrCat(
          "Received schedule keys not among local shard indices while "
          "global_shard_indices_ is empty: [",
          absl::StrJoin(offending_keys, ", "), "], num_shards_=", num_shards_));
    }
  }
  std::vector<ShardPushSchedule> shard_schedules;
  shard_schedules.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    int64_t global_shard = global_shard_index(i);
    int64_t local_shard = local_shard_index(i);
    auto it = schedules.end();
    if (use_global_keys) {
      if (global_shard >= 0) {
        it = schedules.find(static_cast<int32_t>(global_shard));
      }
    } else {
      if (local_shard >= 0) {
        it = schedules.find(static_cast<int32_t>(local_shard));
      }
      if (it == schedules.end() && global_shard >= 0) {
        it = schedules.find(static_cast<int32_t>(global_shard));
      }
    }
    if (it == schedules.end() && i < sorted_sliced_keys.size()) {
      it = schedules.find(sorted_sliced_keys[i]);
    }
    if (it != schedules.end()) {
      shard_schedules.push_back({i, it->second});
    }
  }
  return shard_schedules;
}

absl::StatusOr<std::vector<std::vector<transport::BufferPushTask>>>
WeightSynchronizerBase::BuildLayerPushTasks(
    const tpu_sync::rpc::StartTransferRequest& request) {
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

  // A relay of host buffers that are (or are being tiled) in device layout.
  const bool pre_tiled =
      request.host_tiling_mode() == tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED;
  if (pre_tiled && !request.skip_d2h()) {
    return absl::InvalidArgumentError(
        "HOST_TILING_MODE_PRE_TILED pushes must skip D2H");
  }

  std::vector<std::vector<transport::BufferPushTask>> tasks_by_layer(
      num_layers_);
  TF_ASSIGN_OR_RETURN(const std::vector<ShardPushSchedule> shard_schedules,
                      ResolveShardPushSchedules(request));
  for (const ShardPushSchedule& shard_schedule : shard_schedules) {
    const size_t i = shard_schedule.shard_idx;
    const tpu_sync::rpc::ShardPushScheduleProto& schedule =
        shard_schedule.schedule;

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
      const uint8_t* base_host_ptr =
          (i < layers_[layer_idx_to_use].shards.size())
              ? layers_[layer_idx_to_use].shards[i].host_ptr
              : nullptr;
      if (base_host_ptr == nullptr) {
        return absl::InternalError(
            "Host pointer is null during resharded push");
      }
      size_t shard_host_size = layers_[layer_idx_to_use].shards[i].host_size;

      std::vector<std::string> peers;
      if (!entry.dst_peers().empty()) {
        peers.assign(entry.dst_peers().begin(), entry.dst_peers().end());
      } else if (!entry.dst_peer().empty()) {
        peers.push_back(entry.dst_peer());
      }
      size_t dst_shard_idx = entry.dst_shard_idx();
      size_t count = entry.count() > 0 ? entry.count() : 1;
      size_t src_stride = entry.src_stride_bytes();
      size_t dst_stride = entry.dst_stride_bytes();
      size_t dst_offset = entry.dst_offset_bytes();
      size_t src_offset = entry.src_offset_bytes();
      size_t size = entry.size_bytes();
      if (pre_tiled) {
        // The host buffer holds device-layout bytes and the receiver has the
        // same sharding, so relay exactly what H2D copies: the whole shard,
        // including any tiling padding past the logical span.
        if (entry.outer_counts_size() != 0) {
          return absl::InvalidArgumentError(
              "HOST_TILING_MODE_PRE_TILED entries must not be strided");
        }
        count = 1;
        src_offset = dst_offset = 0;
        size = layers_[layer_idx_to_use].shards[i].device_size;
      }

      const size_t effective_src_stride = src_stride > 0 ? src_stride : size;
      const size_t effective_dst_stride = dst_stride > 0 ? dst_stride : size;
      const bool is_inner_contiguous =
          (count == 1) ||
          (effective_src_stride == size && effective_dst_stride == size);

      if (!is_inner_contiguous && effective_src_stride < size) {
        return absl::InvalidArgumentError(
            "Source stride cannot be smaller than slice size");
      }

      const int num_outer_dims = entry.outer_counts_size();
      if (entry.outer_src_strides_bytes_size() != num_outer_dims ||
          entry.outer_dst_strides_bytes_size() != num_outer_dims) {
        return absl::InvalidArgumentError(
            "Mismatched outer_counts, outer_src_strides_bytes, and "
            "outer_dst_strides_bytes sizes in ShardPushEntryProto");
      }

      size_t num_outer = 1;
      for (int d = 0; d < num_outer_dims; ++d) {
        if (entry.outer_counts(d) <= 0) {
          return absl::InvalidArgumentError(
              "outer_counts entries must be positive");
        }
        num_outer *= static_cast<size_t>(entry.outer_counts(d));
      }

      const size_t inner_src_span =
          is_inner_contiguous ? (count * size)
                              : ((count - 1) * effective_src_stride + size);

      for (size_t outer_idx = 0; outer_idx < num_outer; ++outer_idx) {
        size_t curr_src_offset = src_offset;
        size_t curr_dst_offset = dst_offset;
        size_t temp = outer_idx;
        for (int d = num_outer_dims - 1; d >= 0; --d) {
          const size_t dim_size = static_cast<size_t>(entry.outer_counts(d));
          const size_t coord = temp % dim_size;
          temp /= dim_size;
          curr_src_offset +=
              coord * static_cast<size_t>(entry.outer_src_strides_bytes(d));
          curr_dst_offset +=
              coord * static_cast<size_t>(entry.outer_dst_strides_bytes(d));
        }

        if (curr_src_offset > shard_host_size ||
            inner_src_span > shard_host_size - curr_src_offset) {
          return absl::InvalidArgumentError("Push range out of bounds");
        }

        for (const auto& peer : peers) {
          if (is_inner_contiguous) {
            tasks_by_layer[layer_idx_to_use].push_back({
                .peer = peer,
                .buffer_id = static_cast<size_t>(layer_idx_to_use),
                .dst_shard_idx = dst_shard_idx,
                .dst_offset_bytes = curr_dst_offset,
                .data_ptr = base_host_ptr + curr_src_offset,
                .size_bytes = count * size,
                .dst_stride_bytes = 0,
                .count = 1,
                .src_stride_bytes = 0,
            });
          } else {
            tasks_by_layer[layer_idx_to_use].push_back({
                .peer = peer,
                .buffer_id = static_cast<size_t>(layer_idx_to_use),
                .dst_shard_idx = dst_shard_idx,
                .dst_offset_bytes = curr_dst_offset,
                .data_ptr = base_host_ptr + curr_src_offset,
                .size_bytes = size,
                .dst_stride_bytes = effective_dst_stride,
                .count = count,
                .src_stride_bytes = effective_src_stride,
            });
          }
        }
      }
    }
  }
  return tasks_by_layer;
}

absl::Status WeightSynchronizerBase::PushWeightsReshardedLocal(
    const tpu_sync::rpc::StartTransferRequest& request) {
  VLOG(1) << "Starting PushWeightsResharded for uuid=" << request.uuid()
          << " across " << num_layers_
          << " layers (skip_d2h=" << request.skip_d2h() << ")...";
  StoreSkipTilingLocal(request.uuid(), request);
  const std::vector<bool> planned_skip_tiling =
      ExtractRequestSkipTiling(num_layers_, request);

  auto push_start = absl::Now();

  auto staging_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<std::vector<transport::BufferPushTask>> tasks_by_layer,
      BuildLayerPushTasks(request));
  double staging_time_ms =
      absl::ToDoubleMilliseconds(absl::Now() - staging_start);

  bool already_completed = false;
  bool has_device_buffers = false;
  uint64_t uuid = request.uuid();
  std::shared_ptr<UuidD2hState> d2h_state;
  std::vector<raiden::PjRtCopyFuture> d2h_layer_futures;
  d2h_layer_futures.reserve(num_layers_);
  auto drain_d2h_futures = absl::MakeCleanup([&d2h_layer_futures]() {
    for (auto& fut : d2h_layer_futures) {
      if (fut.IsValid()) {
        (void)fut.Await();
      }
    }
  });
  auto d2h_start = absl::Now();
  if (!request.skip_d2h()) {
    if (uuid != 0) {
      absl::MutexLock lock(d2h_mu_);
      already_completed = !completed_d2h_uuids_.insert(uuid).second;
      auto& state_ref = d2h_states_[uuid];
      if (!state_ref) {
        state_ref = std::make_shared<UuidD2hState>(num_layers_);
      }
      d2h_state = state_ref;
    }
    if (!already_completed) {
      absl::MutexLock op_lock(d2h_op_mu_);
      auto track_scheduled_d2h =
          absl::MakeCleanup([this, &d2h_layer_futures]() {
            if (!d2h_layer_futures.empty()) {
              TrackInFlightD2h(
                  raiden::JoinPjRtCopyFutures(d2h_layer_futures).future);
            }
          });
      WaitForInFlightD2h();
      VLOG(1)
          << "PushWeightsResharded: Executing pipelined D2H copies for uuid "
          << uuid;
      absl::Status refresh_status = RefreshDeviceBuffers();
      if (!refresh_status.ok()) {
        if (d2h_state != nullptr) {
          absl::MutexLock lock(d2h_state->mu);
          d2h_state->status = refresh_status;
          d2h_state->all_ready = true;
        }
        return refresh_status;
      }
      {
        absl::MutexLock lock(buffer_holds_mu_);
        has_device_buffers = !weights_unbound_ && !buffer_holds_.empty();
      }
      ClearExplicitD2hState();
      for (size_t l = 0; l < num_layers_; ++l) {
        absl::StatusOr<raiden::PjRtCopyFuture> layer_copy =
            D2hLayerWithSkipTiling(l, planned_skip_tiling[l]);
        if (!layer_copy.ok()) {
          if (d2h_state != nullptr) {
            absl::MutexLock lock(d2h_state->mu);
            d2h_state->status = layer_copy.status();
            d2h_state->all_ready = true;
          }
          return layer_copy.status();
        }
        if (has_device_buffers) {
          absl::MutexLock lock(explicit_d2h_mu_);
          explicit_d2h_skip_tiling_[l] = planned_skip_tiling[l];
        }
        d2h_layer_futures.push_back(*std::move(layer_copy));
      }
    } else {
      VLOG(1) << "RAIDEN_DIAG push PushWeightsResharded: Coalescing D2H copy"
                 " (already completed for uuid "
              << uuid << " req_id=" << request.req_id() << ")";
    }
  } else {
    VLOG(1) << "PushWeightsResharded: Skipping D2H copy.";
  }

  auto d2h_state_cleanup = absl::MakeCleanup(
      [&d2h_state, already_completed, skip_d2h = request.skip_d2h()]() {
        if (!skip_d2h && !already_completed && d2h_state != nullptr) {
          absl::MutexLock lock(d2h_state->mu);
          if (!d2h_state->all_ready) {
            if (d2h_state->status.ok()) {
              d2h_state->status = absl::InternalError(
                  "D2H leader exited before completing all layers");
            }
            d2h_state->all_ready = true;
          }
        }
      });

  auto h2h_start = absl::Now();
  {
    absl::MutexLock lock(metrics_mu_);
    if (active_h2h_pushes_ == 0) {
      h2h_active_window_start_ = h2h_start;
    }
    active_h2h_pushes_++;
  }
  auto h2h_window_cleanup = absl::MakeCleanup([this]() {
    absl::MutexLock lock(metrics_mu_);
    absl::Time now = absl::Now();
    metrics_.total_h2h_time_ms +=
        absl::ToDoubleMilliseconds(now - h2h_active_window_start_);
    h2h_active_window_start_ = now;
    active_h2h_pushes_--;
  });

  size_t total_h2h_bytes = 0;
  size_t total_d2h_bytes = 0;
  absl::Time last_d2h_done_time = d2h_start;
  absl::Duration d2h_wait_duration = absl::ZeroDuration();

  const bool diag_vlog = VLOG_IS_ON(1);
  struct PushTaskTiming {
    absl::Mutex mu;
    std::vector<double> queue_waits ABSL_GUARDED_BY(mu);
  };
  auto task_timing = ABSL_PREDICT_FALSE(diag_vlog)
                         ? std::make_shared<PushTaskTiming>()
                         : nullptr;

  const bool sync_explicit_d2h =
      request.skip_d2h() &&
      request.host_tiling_mode() != tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED;
  std::vector<std::shared_ptr<LayerTileReady>> explicit_d2h_ready;
  std::vector<std::optional<bool>> staged_skip_tiling;
  bool needs_per_layer_staging_sync = false;
  if (sync_explicit_d2h) {
    {
      absl::MutexLock lock(explicit_d2h_mu_);
      explicit_d2h_ready = explicit_d2h_ready_;
      staged_skip_tiling = explicit_d2h_skip_tiling_;
    }
    explicit_d2h_ready.resize(num_layers_);
    staged_skip_tiling.resize(num_layers_);
    for (size_t l = 0; l < num_layers_; ++l) {
      if (tasks_by_layer[l].empty()) continue;
      if ((explicit_d2h_ready[l] != nullptr &&
           !explicit_d2h_ready[l]->IsSet()) ||
          (staged_skip_tiling[l].has_value() &&
           *staged_skip_tiling[l] != planned_skip_tiling[l])) {
        needs_per_layer_staging_sync = true;
        break;
      }
    }
  }

  size_t pipeline_group_size = GetPipelineGroupSize();
  if (((request.skip_d2h() && !needs_per_layer_staging_sync) ||
       already_completed) &&
      !pipeline_group_size_override_.has_value() &&
      std::getenv("RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE") == nullptr) {
    pipeline_group_size = 0;
  }
  size_t group_size =
      (pipeline_group_size == 0) ? num_layers_ : pipeline_group_size;
  if (group_size == 0) {
    group_size = 1;
  }

  std::vector<std::future<absl::Status>> push_futures;
  push_futures.reserve((num_layers_ + group_size - 1) / group_size);
  auto schedule_push = [&](std::vector<transport::BufferPushTask> tasks) {
    if (tasks.empty()) return;
    int push_parallelism =
        request.parallelism() > 0 ? request.parallelism() : parallelism_;
    absl::Time schedule_time =
        (task_timing != nullptr) ? absl::Now() : absl::InfinitePast();
    // A layer group's push tasks span every shard and are fanned out by the
    // transport's own pool, so the coordinator is not NUMA-pinned.
    push_futures.push_back(push_pool_->Schedule(
        [this, tasks = std::move(tasks), push_parallelism,
         uuid = request.uuid(), schedule_time, task_timing]() {
          if (task_timing != nullptr) {
            double wait_ms =
                absl::ToDoubleMilliseconds(absl::Now() - schedule_time);
            absl::MutexLock lock(task_timing->mu);
            task_timing->queue_waits.push_back(wait_ms);
          }
          return PushWeightsChunks(tasks, push_parallelism, uuid);
        }));
  };

  // A relay seed may still be tiling layers it received from the trainer.
  std::vector<std::shared_ptr<LayerTileReady>> tile_ready;
  if (request.host_tiling_mode() == tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED) {
    absl::MutexLock lock(tile_ready_mu_);
    tile_ready = layer_tile_ready_;
  }
  tile_ready.resize(num_layers_);

  bool refreshed_for_restage = false;
  for (size_t group_start = 0; group_start < num_layers_;
       group_start += group_size) {
    size_t group_end = std::min(num_layers_, group_start + group_size);
    std::vector<transport::BufferPushTask> group_tasks;

    for (size_t l = group_start; l < group_end; ++l) {
      total_d2h_bytes += LayerDeviceBytes(l);
      if (!request.skip_d2h() && !already_completed) {
        absl::Time d2h_wait_start =
            ABSL_PREDICT_FALSE(diag_vlog) ? absl::Now() : absl::InfinitePast();
        absl::Status await_status = d2h_layer_futures[l].Await();
        if (!await_status.ok()) {
          if (d2h_state != nullptr) {
            absl::MutexLock lock(d2h_state->mu);
            d2h_state->status = await_status;
            d2h_state->all_ready = true;
          }
          return await_status;
        }
        if (ABSL_PREDICT_FALSE(diag_vlog)) {
          d2h_wait_duration += (absl::Now() - d2h_wait_start);
        }
        last_d2h_done_time = absl::Now();
      }
      const auto& layer_tasks = tasks_by_layer[l];
      if (sync_explicit_d2h && !layer_tasks.empty()) {
        if (explicit_d2h_ready[l] != nullptr) {
          if (!explicit_d2h_ready[l]->IsSet()) {
            schedule_push(std::exchange(group_tasks, {}));
          }
          absl::Time d2h_wait_start = ABSL_PREDICT_FALSE(diag_vlog)
                                          ? absl::Now()
                                          : absl::InfinitePast();
          ABSL_RETURN_IF_ERROR(explicit_d2h_ready[l]->Wait());
          if (ABSL_PREDICT_FALSE(diag_vlog)) {
            d2h_wait_duration += (absl::Now() - d2h_wait_start);
          }
        }
        if (staged_skip_tiling[l].has_value() &&
            *staged_skip_tiling[l] != planned_skip_tiling[l]) {
          schedule_push(std::exchange(group_tasks, {}));
          absl::MutexLock restage_lock(restage_mu_);
          std::optional<bool> current_staged;
          {
            absl::MutexLock lock(explicit_d2h_mu_);
            if (l < explicit_d2h_skip_tiling_.size()) {
              current_staged = explicit_d2h_skip_tiling_[l];
            }
          }
          const bool planned_skip = planned_skip_tiling[l];
          if (current_staged.has_value() && *current_staged != planned_skip) {
            const bool staged_skip = *current_staged;
            const std::string layer_name = l < layer_names_.size()
                                               ? layer_names_[l]
                                               : absl::StrCat("weights_", l);
            auto make_unbound_mismatch_error = [&]() {
              return absl::FailedPreconditionError(absl::StrCat(
                  "PushWeightsResharded with skip_d2h=true cannot re-stage "
                  "layer ",
                  l, " ('", layer_name, "') for uuid=", uuid,
                  ": explicit d2h() staged skip_tiling=",
                  (staged_skip ? "true" : "false"),
                  " before the transfer plan set skip_tiling=",
                  (planned_skip ? "true" : "false"),
                  ", and device weights are unbound. Keep weights bound until "
                  "the first transfer finishes, or use skip_d2h=false."));
            };
            auto run_restage = [&]() -> absl::Status {
              raiden::PjRtCopyFuture restage_future;
              {
                absl::MutexLock op_lock(d2h_op_mu_);
                WaitForInFlightD2h();
                if (!refreshed_for_restage) {
                  ABSL_RETURN_IF_ERROR(RefreshDeviceBuffers());
                  refreshed_for_restage = true;
                }
                ABSL_ASSIGN_OR_RETURN(restage_future,
                                      D2hLayerWithSkipTiling(l, planned_skip));
                TrackInFlightD2h(restage_future.future);
              }
              return restage_future.Await();
            };
            if (absl::Status s = run_restage(); !s.ok()) {
              return absl::IsFailedPrecondition(s)
                         ? make_unbound_mismatch_error()
                         : s;
            }
            {
              absl::MutexLock lock(explicit_d2h_mu_);
              if (l < explicit_d2h_skip_tiling_.size()) {
                explicit_d2h_skip_tiling_[l] = planned_skip;
              }
            }
          }
        }
      }
      if (!layer_tasks.empty() && tile_ready[l] != nullptr) {
        if (!tile_ready[l]->IsSet()) {
          // Start sending the layers that are ready while this one tiles.
          schedule_push(std::exchange(group_tasks, {}));
        }
        ABSL_RETURN_IF_ERROR(tile_ready[l]->Wait());
      }
      if (!layer_tasks.empty()) {
        for (const auto& t : layer_tasks) {
          total_h2h_bytes += (t.count > 0 ? t.count : 1) * t.size_bytes;
        }
        group_tasks.insert(group_tasks.end(), layer_tasks.begin(),
                           layer_tasks.end());
      }
    }

    if (!request.skip_d2h() && d2h_state != nullptr && group_end > 0) {
      if (!already_completed) {
        absl::MutexLock lock(d2h_state->mu);
        for (size_t l = group_start; l < group_end; ++l) {
          d2h_state->layer_ready[l] = true;
        }
        if (group_end >= num_layers_) {
          d2h_state->all_ready = true;
        }
      } else {
        absl::Time d2h_wait_start =
            ABSL_PREDICT_FALSE(diag_vlog) ? absl::Now() : absl::InfinitePast();
        {
          absl::MutexLock lock(d2h_state->mu);
          struct WaitCtx {
            const UuidD2hState* state;
            size_t target_layer;
          } wait_ctx{d2h_state.get(), group_end - 1};
          auto cond_fn = +[](WaitCtx* c) ABSL_NO_THREAD_SAFETY_ANALYSIS {
            return c->state->all_ready ||
                   c->state->layer_ready[c->target_layer];
          };
          d2h_state->mu.Await(absl::Condition(cond_fn, &wait_ctx));
          TF_RETURN_IF_ERROR(d2h_state->status);
        }
        if (ABSL_PREDICT_FALSE(diag_vlog)) {
          d2h_wait_duration += (absl::Now() - d2h_wait_start);
        }
      }
    }

    schedule_push(std::move(group_tasks));
  }
  if (!request.skip_d2h() && !already_completed && d2h_state != nullptr &&
      num_layers_ == 0) {
    absl::MutexLock lock(d2h_state->mu);
    d2h_state->all_ready = true;
  }

  for (auto& fut : push_futures) {
    TF_RETURN_IF_ERROR(fut.get());
  }
  auto h2h_end = absl::Now();
  double h2h_time_ms = absl::ToDoubleMilliseconds(h2h_end - h2h_start);

  double total_push_time_ms =
      absl::ToDoubleMilliseconds(absl::Now() - push_start);
  // Every layer's D2H has completed by `last_d2h_done_time`.
  const double d2h_time_ms =
      absl::ToDoubleMilliseconds(last_d2h_done_time - d2h_start);
  double last_detiling_time_ms = 0.0;
  {
    absl::MutexLock lock(metrics_mu_);
    std::move(h2h_window_cleanup).Cancel();
    metrics_.total_h2h_time_ms +=
        absl::ToDoubleMilliseconds(h2h_end - h2h_active_window_start_);
    h2h_active_window_start_ = h2h_end;
    active_h2h_pushes_--;
    metrics_.total_h2h_bytes += total_h2h_bytes;
    metrics_.total_push_resharded_time_ms += total_push_time_ms;

    if (!request.skip_d2h() && !already_completed) {
      metrics_.last_d2h_time_ms = d2h_time_ms;
      metrics_.last_d2h_bytes = total_d2h_bytes;
      metrics_.total_d2h_time_ms += d2h_time_ms;
      metrics_.total_d2h_bytes += total_d2h_bytes;
      metrics_.d2h_call_count++;
    }
    metrics_.last_staging_time_ms = staging_time_ms;
    metrics_.total_staging_time_ms += staging_time_ms;
    metrics_.last_h2h_time_ms = h2h_time_ms;
    metrics_.last_h2h_bytes = total_h2h_bytes;
    metrics_.last_total_push_resharded_time_ms = total_push_time_ms;
    metrics_.push_resharded_call_count++;
    last_detiling_time_ms = metrics_.last_detiling_time_ms;
  }
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (store.HasBackends()) {
    if (!request.skip_d2h() && !already_completed && has_device_buffers) {
      store.ObserveHistogram(
          telemetry::metric_names::kWeightSyncD2hTransferTimeMs, {},
          d2h_time_ms);
      if (last_detiling_time_ms > 0.0) {
        store.ObserveHistogram(
            telemetry::metric_names::kWeightSyncDetilingTimeMs, {},
            last_detiling_time_ms);
      }
    }
    store.ObserveHistogram(telemetry::metric_names::kWeightSyncPushDurationMs,
                           {}, total_push_time_ms);
    store.FlushToCloudLogging("source_push", request.uuid(), request.req_id());
  }
  if (ABSL_PREDICT_FALSE(diag_vlog)) {
    double d2h_wait_ms = absl::ToDoubleMilliseconds(d2h_wait_duration);
    size_t num_push_tasks = 0;
    double max_queue_wait_ms = 0.0;
    double sum_queue_wait_ms = 0.0;
    if (task_timing != nullptr) {
      absl::MutexLock lock(task_timing->mu);
      num_push_tasks = task_timing->queue_waits.size();
      for (double w : task_timing->queue_waits) {
        max_queue_wait_ms = std::max(max_queue_wait_ms, w);
        sum_queue_wait_ms += w;
      }
    }
    double avg_queue_wait_ms =
        num_push_tasks > 0 ? (sum_queue_wait_ms / num_push_tasks) : 0.0;
    std::string d2h_status =
        request.skip_d2h() ? "skipped"
                           : (already_completed ? "coalesced" : "executed");
    VLOG(1) << "RAIDEN_DIAG push PushWeightsResharded complete"
            << " uuid=" << request.uuid() << " req_id=" << request.req_id()
            << " staging_time_ms=" << staging_time_ms
            << " d2h_wait_ms=" << d2h_wait_ms << " h2h_time_ms=" << h2h_time_ms
            << " total_push_time_ms=" << total_push_time_ms
            << " total_h2h_bytes=" << total_h2h_bytes
            << " push_tasks=" << num_push_tasks << " d2h_status=" << d2h_status
            << " queue_wait_avg_ms=" << avg_queue_wait_ms
            << " queue_wait_max_ms=" << max_queue_wait_ms;
  }
  VLOG(1) << "Done with PushWeightsResharded (uuid=" << request.uuid()
          << ", total_push_time=" << total_push_time_ms
          << " ms, h2h_time=" << h2h_time_ms << " ms).";
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::BindWeights(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    absl::AnyInvocable<void()> on_rebind) {
  RAIDEN_TRACE("WeightSynchronizerBase::BindWeights");
  if (layer_buffers.empty()) {
    UnbindWeights();
    return absl::OkStatus();
  }
  absl::MutexLock op_lock(d2h_op_mu_);
  WaitForInFlightD2h();
  ABSL_RETURN_IF_ERROR(CheckNoActiveH2hPushes("BindWeights()"));
  ABSL_RETURN_IF_ERROR(CheckDeviceBuffersMatch(layer_buffers));

  xla::PjRtClient* client = nullptr;
  if (!layer_buffers.empty() && !layer_buffers[0].empty()) {
    const auto& first_handle = layer_buffers[0][0];
    physical_size_ = first_handle.GetOnDeviceSizeInBytes();
    if (!first_handle.is_common_buffer && first_handle.c_hold) {
      c_api_ = first_handle.c_hold->c_api;
      extension_ = first_handle.c_hold->extension;
    }
    if (first_handle.buffer) {
      client = const_cast<xla::PjRtClient*>(first_handle.buffer->client());
    } else if (first_handle.device) {
      client = const_cast<xla::PjRtClient*>(first_handle.device->client());
    }
  }

  if (client && host_allocator_ == nullptr) {
    auto alloc = HostMemoryAllocator::Create(client);
    if (alloc.ok()) {
      host_allocator_ = *std::move(alloc);
    }
  }

  absl::MutexLock resolver_lock(device_buffer_resolver_mu_);
  ClearExplicitD2hState();
  ReplaceBufferHolds(layer_buffers, /*allow_from_unbound=*/true);
  if (on_rebind != nullptr) {
    on_rebind();
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::CheckDeviceBuffersMatch(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers)
    const {
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
  return absl::OkStatus();
}

void WeightSynchronizerBase::ReplaceBufferHolds(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    bool allow_from_unbound) {
  std::vector<std::vector<raiden::BufferHoldAndAlias>> new_holds;
  std::vector<std::vector<std::optional<size_t>>> new_tiled_sizes(
      num_layers_, std::vector<std::optional<size_t>>(num_shards_));
  new_holds.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      const auto& new_buffer = layer_buffers[layer_idx][shard_idx];
      const xla::Layout* xla_layout = nullptr;
      if (new_buffer.shape.has_layout()) {
        xla_layout = &new_buffer.shape.layout();
      }
      if (xla_layout && !xla_layout->tiles().empty()) {
        int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(
            new_buffer.shape.element_type());
        size_t physical_bytes =
            tpu_raiden::weight_sync::GetTiledBufferElements(new_buffer.shape) *
            itemsize;
        if (physical_bytes > 0) {
          new_tiled_sizes[layer_idx][shard_idx] = physical_bytes;
        }
      }
      hold_info.push_back(new_buffer);
    }
    new_holds.push_back(std::move(hold_info));
  }
  absl::MutexLock lock(buffer_holds_mu_);
  if (!allow_from_unbound && weights_unbound_) {
    return;
  }
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      if (new_tiled_sizes[layer_idx][shard_idx].has_value()) {
        layers_[layer_idx].shards[shard_idx].tiled_size =
            *new_tiled_sizes[layer_idx][shard_idx];
      }
    }
  }
  buffer_holds_ = std::move(new_holds);
  weights_unbound_ = false;
}

void WeightSynchronizerBase::SetDeviceBufferResolver(
    DeviceBufferResolver resolver) {
  absl::MutexLock lock(device_buffer_resolver_mu_);
  device_buffer_resolver_ = std::move(resolver);
}

absl::Status WeightSynchronizerBase::RefreshDeviceBuffers() {
  absl::MutexLock lock(device_buffer_resolver_mu_);
  if (device_buffer_resolver_ == nullptr) {
    return absl::OkStatus();
  }
  {
    absl::MutexLock holds_lock(buffer_holds_mu_);
    if (weights_unbound_) {
      return absl::OkStatus();
    }
  }
  RAIDEN_TRACE("WeightSynchronizerBase::RefreshDeviceBuffers");
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers,
      device_buffer_resolver_());
  ABSL_RETURN_IF_ERROR(CheckDeviceBuffersMatch(layer_buffers));
  ReplaceBufferHolds(layer_buffers, /*allow_from_unbound=*/false);
  return absl::OkStatus();
}

void WeightSynchronizerBase::UnbindWeights() {
  RAIDEN_TRACE("WeightSynchronizerBase::UnbindWeights");
  absl::MutexLock lock(buffer_holds_mu_);
  buffer_holds_.clear();
  weights_unbound_ = true;
}

absl::Status WeightSynchronizerBase::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  return RegisterExpectedChunksLocal(uuid, expected_chunks);
}

absl::Status WeightSynchronizerBase::RegisterExpectedChunksLocal(
    uint64_t uuid, uint32_t expected_chunks) {
  if (expected_chunks > 0) ClearExplicitD2hState();
  return RaidenManagerBase::RegisterExpectedChunks(uuid, expected_chunks);
}

absl::Status WeightSynchronizerBase::RegisterExpectedLayerChunks(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  return RegisterExpectedLayerChunksLocal(uuid, expected_layer_chunks);
}

absl::Status WeightSynchronizerBase::RegisterExpectedLayerChunksLocal(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  for (const auto& [layer_idx, count] : expected_layer_chunks) {
    if (count > 0) ClearExplicitD2hLayer(layer_idx);
  }
  // The transport fires OnLayerDataReceived only for layers with a non-zero
  // expected count.
  //
  // The transport acks the final chunk to the sender before it runs the
  // receive callbacks, so H2d() may be called before OnLayerDataReceived has
  // run. Create the per-layer readiness signals here, before any data can
  // arrive, so that H2d() waits for them.
  //
  // A tree broadcast re-arms each receiver with the same uuid and the full
  // counts before every pipeline stage, so layers that already have a signal
  // or have been received under `uuid` are left alone.
  const bool signal_layers =
      GetReceiverTiling(uuid) != tpu_sync::rpc::HOST_TILING_MODE_DEFERRED;
  if (auto_h2d_) {
    // Auto-H2D copies each layer as it arrives, so resolve the device buffers
    // once per uuid, before any of its data can arrive. A re-arm of the same
    // uuid may race that uuid's in-flight H2Ds and must not rebind.
    bool is_new_uuid;
    {
      absl::MutexLock lock(pending_h2d_mu_);
      is_new_uuid = !pending_h2d_states_.contains(uuid);
    }
    if (is_new_uuid) {
      ABSL_RETURN_IF_ERROR(RefreshDeviceBuffers());
    }
  }
  {
    absl::MutexLock lock(pending_h2d_mu_);
    bool is_new_uuid = !pending_h2d_states_.contains(uuid);
    auto& state = pending_h2d_states_[uuid];
    if (is_new_uuid) {
      absl::MutexLock metrics_lock(metrics_mu_);
      metrics_.last_tiling_time_ms = 0.0;
      metrics_.last_tiled_bytes = 0;
    }
    size_t expected_layers = 0;
    for (const auto& [layer_idx, count] : expected_layer_chunks) {
      if (count == 0) continue;
      ++expected_layers;
      if (state.tile_ready.contains(layer_idx) ||
          state.layer_futures.contains(layer_idx)) {
        continue;
      }
      std::shared_ptr<LayerTileReady> ready;
      if (signal_layers && layer_idx < num_layers_) {
        ready = std::make_shared<LayerTileReady>();
        state.tile_ready[layer_idx] = ready;
      }
      // Without a signal, H2d() treats the layer as logical layout.
      SetLayerTileReady(layer_idx, std::move(ready));
    }
    state.expected_layers = expected_layers;
  }
  return RaidenManagerBase::RegisterExpectedLayerChunks(uuid,
                                                        expected_layer_chunks);
}

absl::Status WeightSynchronizerBase::OnLayerDataReceived(size_t layer_idx,
                                                         uint64_t uuid) {
  ClearExplicitD2hLayer(layer_idx);
  const tpu_sync::rpc::HostTilingMode tiling = GetReceiverTiling(uuid);
  absl::MutexLock lock(pending_h2d_mu_);
  if (!h2d_pool_) {
    h2d_pool_ =
        std::make_unique<tpu_raiden::NumaThreadPool>(std::max(parallelism_, 4));
  }
  if (auto_h2d_) {
    active_h2d_uuids_.insert(uuid);
  }
  auto& state = pending_h2d_states_[uuid];
  std::shared_ptr<LayerTileReady> ready;
  if (auto it = state.tile_ready.find(layer_idx);
      it != state.tile_ready.end()) {
    ready = std::move(it->second);
    state.tile_ready.erase(it);
  } else {
    // Not armed via RegisterExpectedLayerChunksLocal().
    if (tiling != tpu_sync::rpc::HOST_TILING_MODE_DEFERRED) {
      ready = std::make_shared<LayerTileReady>();
    }
    SetLayerTileReady(layer_idx, ready);
  }
  const bool tile_in_place =
      tiling == tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL;
  // Not NUMA-pinned: TileLayer()/H2dLayer() run each shard on its own node.
  state.layer_futures[layer_idx] = h2d_pool_->Schedule(
      [this, layer_idx, uuid, ready,
       tile_in_place]() -> absl::StatusOr<raiden::PjRtCopyFuture> {
        if (ready == nullptr) {
          // Logical-layout host buffers; tile into a scratchpad.
          if (!auto_h2d_) {
            return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
          }
          return H2dLayer(layer_idx,
                          /*tile=*/!IsTilingSkipped(layer_idx, uuid));
        }
        // Pre-tiled data arrived in device layout and is used as is.
        absl::Status tile_status =
            tile_in_place ? TileLayer(layer_idx, uuid) : absl::OkStatus();
        ready->Set(tile_status);
        TF_RETURN_IF_ERROR(tile_status);
        if (!auto_h2d_) {
          return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
        }
        return H2dLayer(layer_idx, /*tile=*/false);
      });
  return absl::OkStatus();
}

absl::flat_hash_map<size_t, std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
WeightSynchronizerBase::TakeLayerFuturesWhenScheduled(uint64_t uuid) {
  absl::MutexLock lock(pending_h2d_mu_);
  // The transport releases its progress lock before invoking callbacks, so
  // another thread's OnLayerDataReceived may still be in flight when the
  // final chunk triggers OnDataReceived. Wait until every expected layer has
  // scheduled its future.
  auto all_layers_scheduled =
      [this, uuid]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pending_h2d_mu_) {
        auto it = pending_h2d_states_.find(uuid);
        return it == pending_h2d_states_.end() ||
               it->second.layer_futures.size() >= it->second.expected_layers;
      };
  pending_h2d_mu_.Await(absl::Condition(&all_layers_scheduled));
  absl::flat_hash_map<size_t,
                      std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
      layer_futures;
  auto it = pending_h2d_states_.find(uuid);
  if (it != pending_h2d_states_.end()) {
    layer_futures = std::move(it->second.layer_futures);
    pending_h2d_states_.erase(it);
  }
  return layer_futures;
}

absl::Status WeightSynchronizerBase::OnDataReceived(uint64_t uuid) {
  if (!auto_h2d_ && HasH2dHooks()) {
    // Copy to device (bracketed by the hooks) before recording completion
    // below, so a completed transfer implies updated device weights.
    ABSL_RETURN_IF_ERROR(HookedH2d(uuid));
  }
  auto record_completion = [this, uuid]() {
    if (uuid > 0) {
      absl::MutexLock lock(completed_transfers_mu_);
      completed_transfers_.insert(uuid);
    }
  };
  absl::flat_hash_map<size_t,
                      std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
      layer_futures_map = TakeLayerFuturesWhenScheduled(uuid);

  if (!auto_h2d_) {
    // Wait for the per-layer tasks scheduled by OnLayerDataReceived. For
    // in-place uuids these tile the host buffers so that a later H2d() can
    // perform a raw copy.
    for (auto& [layer_idx, tile_future] : layer_futures_map) {
      TF_RETURN_IF_ERROR(tile_future.get().status());
    }
    auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
    if (store.IsCloudLoggingEnabled()) {
      store.FlushToCloudLogging("destination_receive", uuid);
    }
    record_completion();
    if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
      std::string req_id;
      {
        absl::MutexLock lock(skip_tiling_mu_);
        auto it = uuid_to_req_id_.find(uuid);
        if (it != uuid_to_req_id_.end()) {
          req_id = std::move(it->second);
          uuid_to_req_id_.erase(it);
        }
      }
      VLOG(1) << "RAIDEN_DIAG recv OnDataReceived complete uuid=" << uuid
              << (req_id.empty() ? "" : absl::StrCat(" req_id=", req_id))
              << " h2d_time_ms=0 auto_h2d=false";
    }
    return absl::OkStatus();
  }
  VLOG(1) << "Starting OnDataReceived (auto_h2d for uuid=" << uuid << ")...";
  auto h2d_start = absl::Now();

  std::vector<raiden::PjRtCopyFuture> futures_to_await;
  futures_to_await.reserve(layer_futures_map.size());
  size_t h2d_bytes = 0;
  for (auto& [layer_idx, layer_future] : layer_futures_map) {
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture copy_future, layer_future.get());
    futures_to_await.push_back(std::move(copy_future));
    h2d_bytes += LayerDeviceBytes(layer_idx);
  }
  if (!futures_to_await.empty()) {
    raiden::PjRtCopyFuture joined_future =
        raiden::JoinPjRtCopyFutures(futures_to_await);
    TF_RETURN_IF_ERROR(joined_future.Await());
  }

  double h2d_time_ms = absl::ToDoubleMilliseconds(absl::Now() - h2d_start);
  double last_tiling_time_ms = 0.0;
  {
    absl::MutexLock lock(metrics_mu_);
    metrics_.last_h2d_time_ms = h2d_time_ms;
    metrics_.total_h2d_time_ms += h2d_time_ms;
    metrics_.last_h2d_bytes = h2d_bytes;
    metrics_.total_h2d_bytes += h2d_bytes;
    last_tiling_time_ms = metrics_.last_tiling_time_ms;
  }
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (store.HasBackends()) {
    store.ObserveHistogram(
        telemetry::metric_names::kWeightSyncH2dTransferTimeMs, {}, h2d_time_ms);
    if (last_tiling_time_ms > 0.0) {
      store.ObserveHistogram(telemetry::metric_names::kWeightSyncTilingTimeMs,
                             {}, last_tiling_time_ms);
    }
    store.FlushToCloudLogging("destination_on_data_received", uuid);
  }
  record_completion();
  {
    absl::MutexLock lock(pending_h2d_mu_);
    active_h2d_uuids_.erase(uuid);
  }
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
    std::string req_id;
    {
      absl::MutexLock lock(skip_tiling_mu_);
      auto it = uuid_to_req_id_.find(uuid);
      if (it != uuid_to_req_id_.end()) {
        req_id = std::move(it->second);
        uuid_to_req_id_.erase(it);
      }
    }
    VLOG(1) << "RAIDEN_DIAG recv OnDataReceived complete uuid=" << uuid
            << (req_id.empty() ? "" : absl::StrCat(" req_id=", req_id))
            << " h2d_time_ms=" << h2d_time_ms;
  }
  VLOG(1) << "Done with OnDataReceived (auto_h2d for uuid=" << uuid
          << ", h2d_time=" << h2d_time_ms << " ms).";
  return absl::OkStatus();
}

void WeightSynchronizerBase::DrainPendingH2d() {
  // Runs in both auto_h2d modes: a receiver armed with an in-place tiling mode
  // holds LayerTileReady signals that H2d() (and a PRE_TILED relay push) wait
  // on, and WaitForTransferCompletion() waits on completed_transfers_. Neither
  // is released by anything else once the transport stops delivering data, so
  // the drain must cancel the signals and record the pending uuids regardless
  // of who issues the H2D.
  absl::flat_hash_map<uint64_t, PendingH2dState> pending_states;
  {
    absl::MutexLock lock(pending_h2d_mu_);
    pending_states = std::move(pending_h2d_states_);
    pending_h2d_states_.clear();
    active_h2d_uuids_.clear();
  }
  // Layers whose data never arrived will not be tiled.
  for (auto& [uuid, state] : pending_states) {
    for (auto& [layer_idx, ready] : state.tile_ready) {
      ready->Set(absl::CancelledError("Pending H2D was drained"));
    }
  }

  if (pending_states.empty()) {
    // Legacy auto-H2D behavior: if no transfer ever completed, copy the host
    // buffers once so the device is not left stale. Without auto_h2d the
    // caller issues H2d() itself.
    if (!auto_h2d_) return;
    bool needs_h2d = false;
    {
      absl::MutexLock lock(completed_transfers_mu_);
      needs_h2d = completed_transfers_.empty();
    }
    if (needs_h2d) {
      auto status_or_future = H2d(0);
      if (status_or_future.ok()) {
        (void)status_or_future->Await();
      }
      absl::MutexLock lock(completed_transfers_mu_);
      completed_transfers_.insert(0);
    }
    return;
  }

  for (auto& [uuid, state] : pending_states) {
    std::vector<raiden::PjRtCopyFuture> futures_to_await;
    futures_to_await.reserve(state.layer_futures.size());
    for (auto& [layer_idx, layer_future] : state.layer_futures) {
      absl::StatusOr<raiden::PjRtCopyFuture> copy_future = layer_future.get();
      if (copy_future.ok()) {
        futures_to_await.push_back(*std::move(copy_future));
      }
    }
    if (!futures_to_await.empty()) {
      raiden::PjRtCopyFuture joined_future =
          raiden::JoinPjRtCopyFutures(futures_to_await);
      (void)joined_future.Await();
    }
    if (uuid > 0) {
      absl::MutexLock lock(completed_transfers_mu_);
      completed_transfers_.insert(uuid);
    }
  }
}

absl::Status WeightSynchronizerBase::WaitForTransferCompletion(uint64_t uuid) {
  absl::MutexLock lock(completed_transfers_mu_);
  auto condition_fn =
      +[](std::pair<absl::flat_hash_set<uint64_t>*, uint64_t>* p)
           ABSL_NO_THREAD_SAFETY_ANALYSIS {
             return p->second == 0 ? !p->first->empty()
                                   : p->first->contains(p->second);
           };
  std::pair<absl::flat_hash_set<uint64_t>*, uint64_t> ctx{&completed_transfers_,
                                                          uuid};
  completed_transfers_mu_.Await(absl::Condition(condition_fn, &ctx));
  return absl::OkStatus();
}

void WeightSynchronizerBase::StoreSkipTiling(
    uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request) {
  StoreSkipTilingLocal(uuid, request);
}

void WeightSynchronizerBase::StoreSkipTilingLocal(
    uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request) {
  std::vector<bool> skip = ExtractRequestSkipTiling(num_layers_, request);
  VLOG(1) << "[WeightSynchronizerBase] StoreSkipTiling uuid=" << uuid
          << " with " << request.skip_tiling_size()
          << " entries in StartTransferRequest";
  absl::MutexLock lock(skip_tiling_mu_);
  latest_skip_tiling_ = skip;
  uuid_to_skip_tiling_[uuid] = std::move(skip);
  if (!request.is_sender()) {
    const tpu_sync::rpc::HostTilingMode mode = request.host_tiling_mode();
    if (mode == tpu_sync::rpc::HOST_TILING_MODE_UNSPECIFIED ||
        mode == tpu_sync::rpc::HOST_TILING_MODE_DEFERRED) {
      receiver_tiling_.erase(uuid);
    } else {
      receiver_tiling_[uuid] = mode;
    }
  }
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
    if (!request.req_id().empty()) {
      uuid_to_req_id_[uuid] = request.req_id();
    }
  }
}

bool WeightSynchronizerBase::RecordBlockPushShards(
    absl::Span<const int> shard_ids, uint64_t uuid) {
  // Count the shards delivered for `uuid` and report completion only once all
  // of them are in; H2d(uuid) copies all layers and shards and would otherwise
  // upload host buffers whose other shards are still in flight. An unrouted
  // stream carries every shard, so it completes at once (one H2D per stream,
  // as before).
  absl::MutexLock lock(block_push_shards_mu_);
  const size_t finished = block_push_shards_[uuid] += shard_ids.size();
  if (finished < num_shards_) {
    VLOG(1) << "OnBlockShardsReceived: uuid " << uuid << " has " << finished
            << " / " << num_shards_ << " shards; deferring H2D";
    return false;
  }
  return true;
}

absl::Status WeightSynchronizerBase::OnBlockShardsReceived(
    const std::vector<int>& block_ids, absl::Span<const int> shard_ids,
    uint64_t uuid) {
  if (!auto_h2d_ && !HasH2dHooks()) {
    return absl::OkStatus();
  }
  // Copy to device once every shard of this push has landed. Streams that
  // carry all shards (the default) complete immediately, i.e. one H2D per
  // incoming push.
  if (!RecordBlockPushShards(shard_ids, uuid)) {
    return absl::OkStatus();
  }
  if (!auto_h2d_) {
    return HookedH2d(uuid);
  }
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d(uuid));
  return h2d_future.Await();
}

void WeightSynchronizerBase::RegisterH2dStartHook(
    absl::AnyInvocable<void()> hook) {
  absl::MutexLock lock(h2d_hook_mu_);
  h2d_start_hook_ = std::move(hook);
}

void WeightSynchronizerBase::RegisterH2dEndHook(
    absl::AnyInvocable<void(const absl::Status&)> hook) {
  absl::MutexLock lock(h2d_hook_mu_);
  h2d_end_hook_ = std::move(hook);
}

bool WeightSynchronizerBase::HasH2dHooks() const {
  absl::MutexLock lock(h2d_hook_mu_);
  return h2d_start_hook_ != nullptr || h2d_end_hook_ != nullptr;
}

absl::Status WeightSynchronizerBase::HookedH2d(uint64_t uuid) {
  absl::MutexLock lock(h2d_hook_mu_);
  const absl::Time start = absl::Now();
  if (h2d_start_hook_) {
    h2d_start_hook_();
  }
  const absl::Time h2d_start = absl::Now();
  absl::Status h2d_status = [&]() -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d(uuid));
    return h2d_future.Await();
  }();
  const absl::Time h2d_end = absl::Now();
  if (h2d_end_hook_) {
    h2d_end_hook_(h2d_status);
  }
  LOG(INFO) << "Hooked weight sync H2D uuid=" << uuid
            << " status=" << h2d_status << " start_hook_ms="
            << absl::ToDoubleMilliseconds(h2d_start - start)
            << " h2d_ms=" << absl::ToDoubleMilliseconds(h2d_end - h2d_start);
  return h2d_status;
}

void WeightSynchronizerBase::ForgetPushProgress(uint64_t uuid) {
  RaidenManagerBase::ForgetPushProgress(uuid);
  {
    absl::MutexLock lock(completed_transfers_mu_);
    completed_transfers_.erase(uuid);
  }
  {
    absl::MutexLock lock(block_push_shards_mu_);
    block_push_shards_.erase(uuid);
  }
  {
    absl::MutexLock lock(skip_tiling_mu_);
    uuid_to_skip_tiling_.erase(uuid);
    receiver_tiling_.erase(uuid);
    uuid_to_req_id_.erase(uuid);
  }
  {
    absl::MutexLock lock(d2h_mu_);
    completed_d2h_uuids_.erase(uuid);
    d2h_states_.erase(uuid);
  }
  absl::flat_hash_map<size_t, std::shared_ptr<LayerTileReady>> tile_ready;
  {
    absl::MutexLock lock(pending_h2d_mu_);
    auto it = pending_h2d_states_.find(uuid);
    if (it != pending_h2d_states_.end()) {
      tile_ready = std::move(it->second.tile_ready);
      pending_h2d_states_.erase(it);
    }
  }
  // Layers whose data never arrived will not be tiled.
  for (auto& [layer_idx, ready] : tile_ready) {
    ready->Set(absl::CancelledError("Receive was forgotten"));
  }
}

uint8_t* WeightSynchronizerBase::GetHostPointer(size_t layer_idx,
                                                size_t shard_idx) {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return nullptr;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  if (!local_shard_indices_.empty()) {
    auto it =
        std::find(local_shard_indices_.begin(), local_shard_indices_.end(),
                  static_cast<int>(shard_idx));
    if (it != local_shard_indices_.end()) {
      local_idx = std::distance(local_shard_indices_.begin(), it);
    } else if (!global_shard_indices_.empty()) {
      auto git =
          std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                    static_cast<int64_t>(shard_idx));
      if (git != global_shard_indices_.end()) {
        local_idx = std::distance(global_shard_indices_.begin(), git);
      }
    }
  } else if (!global_shard_indices_.empty()) {
    auto it =
        std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                  static_cast<int64_t>(shard_idx));
    if (it != global_shard_indices_.end()) {
      local_idx = std::distance(global_shard_indices_.begin(), it);
    }
  }
  return const_cast<uint8_t*>(layers_[layer_idx].shards[local_idx].host_ptr);
}

size_t WeightSynchronizerBase::GetHostSize(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return 0;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  if (!local_shard_indices_.empty()) {
    auto it =
        std::find(local_shard_indices_.begin(), local_shard_indices_.end(),
                  static_cast<int>(shard_idx));
    if (it != local_shard_indices_.end()) {
      local_idx = std::distance(local_shard_indices_.begin(), it);
    } else if (!global_shard_indices_.empty()) {
      auto git =
          std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                    static_cast<int64_t>(shard_idx));
      if (git != global_shard_indices_.end()) {
        local_idx = std::distance(global_shard_indices_.begin(), git);
      }
    }
  } else if (!global_shard_indices_.empty()) {
    auto it =
        std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                  static_cast<int64_t>(shard_idx));
    if (it != global_shard_indices_.end()) {
      local_idx = std::distance(global_shard_indices_.begin(), it);
    }
  }
  return layers_[layer_idx].shards[local_idx].host_size;
}

const uint8_t* WeightSynchronizerBase::GetHostPointer(size_t layer_idx,
                                                      size_t shard_idx) const {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return nullptr;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  if (!local_shard_indices_.empty()) {
    auto it =
        std::find(local_shard_indices_.begin(), local_shard_indices_.end(),
                  static_cast<int>(shard_idx));
    if (it != local_shard_indices_.end()) {
      local_idx = std::distance(local_shard_indices_.begin(), it);
    } else if (!global_shard_indices_.empty()) {
      auto git =
          std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                    static_cast<int64_t>(shard_idx));
      if (git != global_shard_indices_.end()) {
        local_idx = std::distance(global_shard_indices_.begin(), git);
      }
    }
  } else if (!global_shard_indices_.empty()) {
    auto it =
        std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                  static_cast<int64_t>(shard_idx));
    if (it != global_shard_indices_.end()) {
      local_idx = std::distance(global_shard_indices_.begin(), it);
    }
  }
  return layers_[layer_idx].shards[local_idx].host_ptr;
}

size_t WeightSynchronizerBase::GetHostSize(size_t layer_idx,
                                           size_t shard_idx) const {
  if (layer_idx >= layers_.size() || layers_[layer_idx].shards.empty()) {
    return 0;
  }
  size_t local_idx = shard_idx % layers_[layer_idx].shards.size();
  if (!local_shard_indices_.empty()) {
    auto it =
        std::find(local_shard_indices_.begin(), local_shard_indices_.end(),
                  static_cast<int>(shard_idx));
    if (it != local_shard_indices_.end()) {
      local_idx = std::distance(local_shard_indices_.begin(), it);
    } else if (!global_shard_indices_.empty()) {
      auto git =
          std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                    static_cast<int64_t>(shard_idx));
      if (git != global_shard_indices_.end()) {
        local_idx = std::distance(global_shard_indices_.begin(), git);
      }
    }
  } else if (!global_shard_indices_.empty()) {
    auto it =
        std::find(global_shard_indices_.begin(), global_shard_indices_.end(),
                  static_cast<int64_t>(shard_idx));
    if (it != global_shard_indices_.end()) {
      local_idx = std::distance(global_shard_indices_.begin(), it);
    }
  }
  return layers_[layer_idx].shards[local_idx].host_size;
}

void WeightSynchronizerBase::SetTestOnlyRateLimiters(
    std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> egress,
    std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> ingress) {
  RaidenManagerBase::SetTestOnlyRateLimiters(std::move(egress),
                                             std::move(ingress));
}

}  // namespace weight_sync
}  // namespace tpu_raiden

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

#include "tpu_sync/weight_sync/weight_buffers.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
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
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/weight_sync/tiling_utils.h"

namespace tpu_raiden {
namespace weight_sync {

namespace {

static std::atomic<size_t> global_allocated_host_dram_bytes_{0};

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

void LayerTileReady::Set(absl::Status status) {
  absl::MutexLock lock(mu_);
  if (done_) return;
  status_ = std::move(status);
  done_ = true;
}

absl::Status LayerTileReady::Wait() {
  absl::MutexLock lock(mu_);
  while (!mu_.AwaitWithTimeout(absl::Condition(&done_), absl::Seconds(60))) {
    LOG(WARNING) << "Still waiting for a received layer to be ready on host";
  }
  return status_;
}

bool LayerTileReady::IsSet() {
  absl::MutexLock lock(mu_);
  return done_;
}

WeightBuffers::WeightBuffers(RaidenManagerBase& manager,
                             std::vector<std::string> layer_names)
    : manager_(manager),
      num_layers_(manager.num_layers()),
      num_shards_(manager.num_shards()),
      layer_names_(std::move(layer_names)),
      shard_pool_(std::make_unique<tpu_raiden::NumaThreadPool>(
          std::max<size_t>(num_shards_, 1))) {
  tiled_scratchpads_.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    tiled_scratchpads_.push_back(std::make_unique<ShardScratchpad>());
  }
  ClearExplicitD2hState();
}

WeightBuffers::~WeightBuffers() {
  DrainInFlightD2h();
  shard_pool_.reset();
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

void WeightBuffers::AdoptDeviceInfo(
    const raiden::RaidenBufferHandle& first_handle) {
  physical_size_ = first_handle.GetOnDeviceSizeInBytes();
  if (!first_handle.is_common_buffer && first_handle.c_hold) {
    c_api_ = first_handle.c_hold->c_api;
    extension_ = first_handle.c_hold->extension;
  }
  xla::PjRtClient* client = nullptr;
  if (first_handle.buffer) {
    client = const_cast<xla::PjRtClient*>(first_handle.buffer->client());
  } else if (first_handle.device) {
    client = const_cast<xla::PjRtClient*>(first_handle.device->client());
  }
  if (client && host_allocator_ == nullptr) {
    auto alloc = HostMemoryAllocator::Create(client);
    if (alloc.ok()) {
      host_allocator_ = *std::move(alloc);
    }
  }
}

void WeightBuffers::SetPhysicalSize(size_t physical_size) {
  physical_size_ = physical_size;
}

void WeightBuffers::SetBufferHolds(
    std::vector<std::vector<raiden::BufferHoldAndAlias>> holds) {
  absl::MutexLock lock(buffer_holds_mu_);
  buffer_holds_ = std::move(holds);
}

absl::StatusOr<uint8_t*> WeightBuffers::AcquireTiledScratchpadLocked(
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

void WeightBuffers::UpdateAllocatedOccupancyMetric(size_t delta) {
  allocated_host_dram_bytes_ += delta;
  size_t total = global_allocated_host_dram_bytes_.fetch_add(
                     delta, std::memory_order_relaxed) +
                 delta;
  telemetry::RaidenMetricStore::GetGlobalMetricStore().SetGauge(
      telemetry::metric_names::kWeightSyncBufferAllocatedBytes, {},
      static_cast<double>(total));
}

void WeightBuffers::SetSkipTiling(const std::vector<bool>& skip_tiling) {
  absl::MutexLock lock(skip_tiling_mu_);
  latest_skip_tiling_ = skip_tiling;
}

void WeightBuffers::SetSkipTiling(bool skip_all) {
  absl::MutexLock lock(skip_tiling_mu_);
  latest_skip_tiling_.assign(num_layers_, skip_all);
}

void WeightBuffers::StoreSkipTiling(uint64_t uuid,
                                    std::vector<bool> skip_tiling) {
  absl::MutexLock lock(skip_tiling_mu_);
  latest_skip_tiling_ = skip_tiling;
  uuid_to_skip_tiling_[uuid] = std::move(skip_tiling);
}

void WeightBuffers::ForgetSkipTiling(uint64_t uuid) {
  absl::MutexLock lock(skip_tiling_mu_);
  uuid_to_skip_tiling_.erase(uuid);
}

std::vector<bool> WeightBuffers::GetActiveSkipTiling(uint64_t uuid) {
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

bool WeightBuffers::IsTilingSkipped(size_t layer_idx, uint64_t uuid) {
  const std::vector<bool> skip = GetActiveSkipTiling(uuid);
  return layer_idx < skip.size() && skip[layer_idx];
}

void WeightBuffers::ClearExplicitD2hState() {
  absl::MutexLock lock(explicit_d2h_mu_);
  explicit_d2h_ready_.assign(num_layers_, nullptr);
  explicit_d2h_skip_tiling_.assign(num_layers_, std::nullopt);
}

void WeightBuffers::ClearExplicitD2hLayer(size_t layer_idx) {
  absl::MutexLock lock(explicit_d2h_mu_);
  if (layer_idx < explicit_d2h_ready_.size()) {
    explicit_d2h_ready_[layer_idx] = nullptr;
  }
  if (layer_idx < explicit_d2h_skip_tiling_.size()) {
    explicit_d2h_skip_tiling_[layer_idx] = std::nullopt;
  }
}

WeightBuffers::ExplicitD2hSnapshot WeightBuffers::SnapshotExplicitD2h() const {
  ExplicitD2hSnapshot snapshot;
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    snapshot.ready = explicit_d2h_ready_;
    snapshot.skip_tiling = explicit_d2h_skip_tiling_;
  }
  snapshot.ready.resize(num_layers_);
  snapshot.skip_tiling.resize(num_layers_);
  return snapshot;
}

void WeightBuffers::TrackInFlightD2h(xla::Future<> future) {
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

void WeightBuffers::WaitForInFlightD2h() const {
  xla::Future<> in_flight;
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    in_flight = in_flight_d2h_;
  }
  if (in_flight.IsValid()) {
    (void)in_flight.Await();
  }
}

void WeightBuffers::DrainInFlightD2h() {
  absl::MutexLock op_lock(d2h_op_mu_);
  WaitForInFlightD2h();
}

absl::Status WeightBuffers::CheckNoActiveH2hPushes(
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

absl::Status WeightBuffers::WaitForExplicitD2h(size_t layer_idx) const {
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

absl::Status WeightBuffers::WaitForAllExplicitD2h() const {
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

absl::StatusOr<const uint8_t*> WeightBuffers::WaitAndGetHostBufferPtr(
    size_t layer_idx, size_t shard_idx) const {
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return absl::OutOfRangeError(
        absl::StrCat("Host buffer index out of range: layer_idx=", layer_idx,
                     " (num_layers=", num_layers_, "), shard_idx=", shard_idx,
                     " (num_shards=", num_shards_, ")"));
  }
  ABSL_RETURN_IF_ERROR(WaitForExplicitD2h(layer_idx));
  const uint8_t* host_ptr = layers()[layer_idx].shards[shard_idx].host_ptr;
  if (host_ptr == nullptr) {
    return absl::InternalError(
        absl::StrCat("Host buffer pointer is null for layer_idx=", layer_idx,
                     ", shard_idx=", shard_idx));
  }
  return host_ptr;
}

size_t WeightBuffers::LayerDeviceBytes(size_t layer_idx) const {
  if (layer_idx >= layers().size()) return 0;
  size_t bytes = 0;
  for (const auto& shard : layers()[layer_idx].shards) {
    bytes += shard.device_size;
  }
  return bytes;
}

uint8_t* WeightBuffers::GetTiledPointer(size_t shard_idx) {
  if (shard_idx < tiled_scratchpads_.size() && tiled_scratchpads_[shard_idx]) {
    absl::MutexLock lock(tiled_scratchpads_[shard_idx]->mu);
    return tiled_scratchpads_[shard_idx]->ptr;
  }
  return nullptr;
}

const uint8_t* WeightBuffers::GetTiledPointer(size_t shard_idx) const {
  if (shard_idx < tiled_scratchpads_.size() && tiled_scratchpads_[shard_idx]) {
    absl::MutexLock lock(tiled_scratchpads_[shard_idx]->mu);
    return tiled_scratchpads_[shard_idx]->ptr;
  }
  return nullptr;
}

absl::StatusOr<WeightBuffers::ShardScratchpad*>
WeightBuffers::GetTiledScratchpad(size_t shard_idx) {
  if (shard_idx >= tiled_scratchpads_.size() ||
      !tiled_scratchpads_[shard_idx]) {
    return absl::InternalError("Shard index out of range for scratchpad");
  }
  return tiled_scratchpads_[shard_idx].get();
}

absl::StatusOr<absl::Span<const uint8_t>> WeightBuffers::TileShardLocked(
    size_t layer_idx, size_t shard_idx,
    const raiden::BufferHoldAndAlias& shard_hold, ShardScratchpad& sp) {
  const auto& shard_info = layers()[layer_idx].shards[shard_idx];
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
      shard_hold.shape.layout(), manager_.shard_numa_node(shard_idx)));
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

absl::StatusOr<xla::Future<raiden::BufferHolder>> WeightBuffers::H2dShard(
    size_t layer_idx, size_t shard_idx, bool tile) {
  const auto& shard_info = layers()[layer_idx].shards[shard_idx];
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

absl::Status WeightBuffers::TileLayer(size_t layer_idx, uint64_t uuid) {
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
  return manager_
      .ForEachShardNumaLocal(
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
            ABSL_ASSIGN_OR_RETURN(ShardScratchpad * sp, GetTiledScratchpad(i));
            absl::MutexLock lock(sp->mu);
            ABSL_ASSIGN_OR_RETURN(
                absl::Span<const uint8_t> tiled,
                TileShardLocked(layer_idx, i, shard_hold, *sp));
            std::memcpy(
                const_cast<uint8_t*>(layers()[layer_idx].shards[i].host_ptr),
                tiled.data(), tiled.size());
            return absl::OkStatus();
          })
      .Await();
}

void WeightBuffers::SetLayerTileReady(size_t layer_idx,
                                      std::shared_ptr<LayerTileReady> ready) {
  if (layer_idx >= num_layers_) return;
  absl::MutexLock lock(tile_ready_mu_);
  layer_tile_ready_.resize(num_layers_);
  layer_tile_ready_[layer_idx] = std::move(ready);
}

void WeightBuffers::MarkHostBufferUntiled() {
  absl::MutexLock lock(tile_ready_mu_);
  layer_tile_ready_.assign(num_layers_, nullptr);
}

std::vector<std::shared_ptr<LayerTileReady>>
WeightBuffers::SnapshotLayerTileReady() {
  std::vector<std::shared_ptr<LayerTileReady>> tile_ready;
  {
    absl::MutexLock lock(tile_ready_mu_);
    tile_ready = layer_tile_ready_;
  }
  tile_ready.resize(num_layers_);
  return tile_ready;
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightBuffers::H2dLayer(size_t layer_idx,
                                                               bool tile) {
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
      manager_.ForEachShardNumaLocal<raiden::BufferHolder>(
          *shard_pool_,
          [this, layer_idx,
           tile](size_t i) -> xla::Future<raiden::BufferHolder> {
            ABSL_ASSIGN_OR_RETURN(xla::Future<raiden::BufferHolder> future,
                                  H2dShard(layer_idx, i, tile));
            return future;
          }));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightBuffers::H2d(uint64_t uuid) {
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
  std::vector<std::shared_ptr<LayerTileReady>> tile_ready =
      SnapshotLayerTileReady();
  bool any_in_place = false;
  for (const std::shared_ptr<LayerTileReady>& ready : tile_ready) {
    any_in_place = any_in_place || ready != nullptr;
  }
  VLOG(1) << "Starting H2d across " << num_layers_ << " layers (uuid=" << uuid
          << ", any_in_place=" << any_in_place << ")...";
  if (!any_in_place) {
    // All layers are tiled below; otherwise keep the metrics recorded while
    // tiling in place on receive.
    ResetLastTilingMetrics();
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
      manager_.ForEachShardNumaLocal<std::vector<raiden::BufferHolder>>(
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

absl::StatusOr<raiden::PjRtCopyFuture> WeightBuffers::D2hLayer(
    size_t layer_idx, uint64_t uuid,
    std::shared_ptr<std::atomic<double>> max_detile_ms) {
  const std::vector<bool> active_skip = GetActiveSkipTiling(uuid);
  const bool skip_tiling =
      layer_idx < active_skip.size() && active_skip[layer_idx];
  return D2hLayerWithSkipTiling(layer_idx, skip_tiling,
                                std::move(max_detile_ms));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightBuffers::D2hLayerWithSkipTiling(
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
  xla::Future<> shard_copies = manager_.ForEachShardNumaLocal(
      *shard_pool_,
      [this, layer_idx, skip_tiling,
       max_detile_ms = std::move(max_detile_ms)](size_t shard_idx) {
        return D2hShard(layer_idx, shard_idx, skip_tiling, max_detile_ms);
      });
  return raiden::PjRtCopyFuture(std::move(shard_copies),
                                raiden::BufferHolders{});
}

xla::Future<> WeightBuffers::D2hShard(
    size_t layer_idx, size_t shard_idx, bool skip_tiling,
    std::shared_ptr<std::atomic<double>> max_detile_ms) {
  auto& layer_info = layers()[layer_idx];
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
         numa_node = manager_.shard_numa_node(shard_idx)]() -> absl::Status {
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

absl::StatusOr<raiden::PjRtCopyFuture> WeightBuffers::D2h(uint64_t uuid) {
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
  // `this` outlives in-flight copies; ~WeightBuffers awaits `in_flight_d2h_`.
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

WeightBuffers::ScheduledD2h WeightBuffers::ScheduleAllLayerD2h(
    const std::vector<bool>& skip_tiling) {
  ScheduledD2h result;
  result.layer_futures.reserve(num_layers_);
  absl::MutexLock op_lock(d2h_op_mu_);
  auto track_scheduled_d2h = absl::MakeCleanup([this, &result]() {
    if (!result.layer_futures.empty()) {
      TrackInFlightD2h(
          raiden::JoinPjRtCopyFutures(result.layer_futures).future);
    }
  });
  WaitForInFlightD2h();
  result.status = RefreshDeviceBuffers();
  if (!result.status.ok()) {
    return result;
  }
  {
    absl::MutexLock lock(buffer_holds_mu_);
    result.has_device_buffers = !weights_unbound_ && !buffer_holds_.empty();
  }
  ClearExplicitD2hState();
  for (size_t l = 0; l < num_layers_; ++l) {
    const bool layer_skip = l < skip_tiling.size() && skip_tiling[l];
    absl::StatusOr<raiden::PjRtCopyFuture> layer_copy =
        D2hLayerWithSkipTiling(l, layer_skip);
    if (!layer_copy.ok()) {
      result.status = layer_copy.status();
      return result;
    }
    if (result.has_device_buffers) {
      absl::MutexLock lock(explicit_d2h_mu_);
      explicit_d2h_skip_tiling_[l] = layer_skip;
    }
    result.layer_futures.push_back(*std::move(layer_copy));
  }
  return result;
}

absl::Status WeightBuffers::RestageLayerIfMismatched(
    size_t l, bool planned_skip, uint64_t uuid,
    bool& refreshed_device_buffers) {
  absl::MutexLock restage_lock(restage_mu_);
  std::optional<bool> current_staged;
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    if (l < explicit_d2h_skip_tiling_.size()) {
      current_staged = explicit_d2h_skip_tiling_[l];
    }
  }
  if (!current_staged.has_value() || *current_staged == planned_skip) {
    return absl::OkStatus();
  }
  const bool staged_skip = *current_staged;
  const std::string layer_name =
      l < layer_names_.size() ? layer_names_[l] : absl::StrCat("weights_", l);
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
      if (!refreshed_device_buffers) {
        ABSL_RETURN_IF_ERROR(RefreshDeviceBuffers());
        refreshed_device_buffers = true;
      }
      ABSL_ASSIGN_OR_RETURN(restage_future,
                            D2hLayerWithSkipTiling(l, planned_skip));
      TrackInFlightD2h(restage_future.future);
    }
    return restage_future.Await();
  };
  if (absl::Status s = run_restage(); !s.ok()) {
    return absl::IsFailedPrecondition(s) ? make_unbound_mismatch_error() : s;
  }
  {
    absl::MutexLock lock(explicit_d2h_mu_);
    if (l < explicit_d2h_skip_tiling_.size()) {
      explicit_d2h_skip_tiling_[l] = planned_skip;
    }
  }
  return absl::OkStatus();
}

absl::Status WeightBuffers::BindWeights(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    absl::AnyInvocable<void()> on_rebind) {
  if (layer_buffers.empty()) {
    UnbindWeights();
    return absl::OkStatus();
  }
  absl::MutexLock op_lock(d2h_op_mu_);
  WaitForInFlightD2h();
  ABSL_RETURN_IF_ERROR(CheckNoActiveH2hPushes("BindWeights()"));
  ABSL_RETURN_IF_ERROR(CheckDeviceBuffersMatch(layer_buffers));

  if (!layer_buffers[0].empty()) {
    AdoptDeviceInfo(layer_buffers[0][0]);
  }

  absl::MutexLock resolver_lock(device_buffer_resolver_mu_);
  ClearExplicitD2hState();
  ReplaceBufferHolds(layer_buffers, /*allow_from_unbound=*/true);
  if (on_rebind != nullptr) {
    on_rebind();
  }
  return absl::OkStatus();
}

absl::Status WeightBuffers::CheckDeviceBuffersMatch(
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
      if (new_size != layers()[layer_idx].shards[shard_idx].device_size) {
        return absl::InvalidArgumentError("Buffer size mismatch");
      }
    }
  }
  return absl::OkStatus();
}

void WeightBuffers::ReplaceBufferHolds(
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
        layers()[layer_idx].shards[shard_idx].tiled_size =
            *new_tiled_sizes[layer_idx][shard_idx];
      }
    }
  }
  buffer_holds_ = std::move(new_holds);
  weights_unbound_ = false;
}

void WeightBuffers::SetDeviceBufferResolver(DeviceBufferResolver resolver) {
  absl::MutexLock lock(device_buffer_resolver_mu_);
  device_buffer_resolver_ = std::move(resolver);
}

absl::Status WeightBuffers::RefreshDeviceBuffers() {
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

void WeightBuffers::UnbindWeights() {
  absl::MutexLock lock(buffer_holds_mu_);
  buffer_holds_.clear();
  weights_unbound_ = true;
}

WeightSyncMetrics WeightBuffers::GetMetrics() const {
  absl::MutexLock lock(metrics_mu_);
  return metrics_;
}

void WeightBuffers::ResetMetrics() {
  absl::MutexLock lock(metrics_mu_);
  metrics_ = WeightSyncMetrics{};
  if (active_h2h_pushes_ > 0) {
    h2h_active_window_start_ = absl::Now();
  }
}

void WeightBuffers::SetMetricsForTesting(const WeightSyncMetrics& m) {
  absl::MutexLock lock(metrics_mu_);
  metrics_ = m;
}

void WeightBuffers::ResetLastTilingMetrics() {
  absl::MutexLock lock(metrics_mu_);
  metrics_.last_tiling_time_ms = 0.0;
  metrics_.last_tiled_bytes = 0;
}

double WeightBuffers::RecordH2d(double h2d_time_ms, size_t h2d_bytes) {
  absl::MutexLock lock(metrics_mu_);
  metrics_.last_h2d_time_ms = h2d_time_ms;
  metrics_.total_h2d_time_ms += h2d_time_ms;
  metrics_.last_h2d_bytes = h2d_bytes;
  metrics_.total_h2d_bytes += h2d_bytes;
  return metrics_.last_tiling_time_ms;
}

void WeightBuffers::BeginH2hPush(absl::Time start) {
  absl::MutexLock lock(metrics_mu_);
  if (active_h2h_pushes_ == 0) {
    h2h_active_window_start_ = start;
  }
  active_h2h_pushes_++;
}

void WeightBuffers::AbortH2hPush() {
  absl::MutexLock lock(metrics_mu_);
  absl::Time now = absl::Now();
  metrics_.total_h2h_time_ms +=
      absl::ToDoubleMilliseconds(now - h2h_active_window_start_);
  h2h_active_window_start_ = now;
  active_h2h_pushes_--;
}

double WeightBuffers::FinishH2hPush(const H2hPushStats& stats) {
  absl::MutexLock lock(metrics_mu_);
  metrics_.total_h2h_time_ms +=
      absl::ToDoubleMilliseconds(stats.h2h_end - h2h_active_window_start_);
  h2h_active_window_start_ = stats.h2h_end;
  active_h2h_pushes_--;
  metrics_.total_h2h_bytes += stats.h2h_bytes;
  metrics_.total_push_resharded_time_ms += stats.total_push_time_ms;

  if (stats.performed_d2h) {
    metrics_.last_d2h_time_ms = stats.d2h_time_ms;
    metrics_.last_d2h_bytes = stats.d2h_bytes;
    metrics_.total_d2h_time_ms += stats.d2h_time_ms;
    metrics_.total_d2h_bytes += stats.d2h_bytes;
    metrics_.d2h_call_count++;
  }
  metrics_.last_staging_time_ms = stats.staging_time_ms;
  metrics_.total_staging_time_ms += stats.staging_time_ms;
  metrics_.last_h2h_time_ms = stats.h2h_time_ms;
  metrics_.last_h2h_bytes = stats.h2h_bytes;
  metrics_.last_total_push_resharded_time_ms = stats.total_push_time_ms;
  metrics_.push_resharded_call_count++;
  return metrics_.last_detiling_time_ms;
}

void WeightBuffers::RegisterH2dStartHook(absl::AnyInvocable<void()> hook) {
  absl::MutexLock lock(h2d_hook_mu_);
  h2d_start_hook_ = std::move(hook);
}

void WeightBuffers::RegisterH2dEndHook(
    absl::AnyInvocable<void(const absl::Status&)> hook) {
  absl::MutexLock lock(h2d_hook_mu_);
  h2d_end_hook_ = std::move(hook);
}

bool WeightBuffers::HasH2dHooks() const {
  absl::MutexLock lock(h2d_hook_mu_);
  return h2d_start_hook_ != nullptr || h2d_end_hook_ != nullptr;
}

absl::Status WeightBuffers::HookedH2d(
    uint64_t uuid, absl::FunctionRef<absl::Status()> copy_to_device) {
  absl::MutexLock lock(h2d_hook_mu_);
  const absl::Time start = absl::Now();
  if (h2d_start_hook_) {
    h2d_start_hook_();
  }
  const absl::Time h2d_start = absl::Now();
  absl::Status h2d_status = copy_to_device();
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

}  // namespace weight_sync
}  // namespace tpu_raiden

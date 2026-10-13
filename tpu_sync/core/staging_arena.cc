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

#include "tpu_sync/core/staging_arena.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/logical_block_manager.h"

namespace tpu_raiden {

// ---------------------------------------------------------------------------
// StagingArena
// ---------------------------------------------------------------------------

void StagingArena::Allocation::Release() {
  if (arena_ != nullptr) {
    arena_->ReleaseSlot(index_);
  }
  arena_ = nullptr;
  index_ = 0;
}

void StagingArena::MoveSlotsFromLocked(StagingArena& other) {
  reuse_policy_ = other.reuse_policy_;
  shutting_down_ = other.shutting_down_;
  free_slots_ = std::move(other.free_slots_);
  in_use_ = std::move(other.in_use_);
  other.free_slots_.clear();
  other.in_use_.clear();
}

size_t StagingArena::AddSlotLocked() {
  const size_t index = in_use_.size();
  in_use_.push_back(false);
  free_slots_.push_back(index);
  return index;
}

bool StagingArena::CanAcquireLocked(int64_t /*num_blocks*/) const {
  return !free_slots_.empty() || CanGrowLocked();
}

absl::Status StagingArena::GrowLocked(int64_t /*num_blocks*/) {
  return absl::ResourceExhaustedError("Staging arena is out of capacity");
}

absl::StatusOr<StagingArena::Allocation> StagingArena::AcquireLocked(
    int64_t num_blocks) {
  if (shutting_down_) {
    return absl::CancelledError("Staging arena is shutting down");
  }
  if (free_slots_.empty()) {
    if (!CanGrowLocked()) {
      return absl::ResourceExhaustedError(
          absl::StrCat("Staging arena is out of capacity (", in_use_.size(),
                       " slots, all in use)"));
    }
    ABSL_RETURN_IF_ERROR(GrowLocked(num_blocks));
    if (free_slots_.empty()) {
      return absl::ResourceExhaustedError(
          "Staging arena could not grow to satisfy the request");
    }
  }
  const size_t index = free_slots_.front();
  ABSL_RETURN_IF_ERROR(PrepareSlotLocked(index, num_blocks));
  free_slots_.pop_front();
  in_use_[index] = true;
  return Allocation(this, index);
}

absl::StatusOr<StagingArena::Allocation> StagingArena::Acquire(
    int64_t num_blocks) {
  absl::MutexLock lock(mu_);
  if (shutting_down_) {
    return absl::CancelledError("Staging arena is shutting down");
  }
  ABSL_RETURN_IF_ERROR(ValidateRequestLocked(num_blocks));
  return AcquireLocked(num_blocks);
}

absl::StatusOr<StagingArena::Allocation> StagingArena::AcquireWithDeadline(
    int64_t num_blocks, absl::Time deadline) {
  absl::MutexLock lock(mu_);
  if (shutting_down_) {
    return absl::CancelledError("Staging arena is shutting down");
  }
  ABSL_RETURN_IF_ERROR(ValidateRequestLocked(num_blocks));
  if (!CanAcquireLocked(num_blocks) && absl::Now() < deadline) {
    auto can_proceed = [this, num_blocks]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return shutting_down_ || CanAcquireLocked(num_blocks);
    };
    mu_.AwaitWithDeadline(absl::Condition(&can_proceed), deadline);
  }
  return AcquireLocked(num_blocks);
}

absl::StatusOr<StagingArena::Allocation> StagingArena::AcquireWithTimeout(
    int64_t num_blocks, std::chrono::steady_clock::time_point deadline) {
  const auto remaining = deadline - std::chrono::steady_clock::now();
  const absl::Time absl_deadline =
      remaining.count() <= 0 ? absl::InfinitePast()
                             : absl::Now() + absl::FromChrono(remaining);
  return AcquireWithDeadline(num_blocks, absl_deadline);
}

void StagingArena::Shutdown() {
  absl::MutexLock lock(mu_);
  shutting_down_ = true;
}

void StagingArena::ReleaseSlot(size_t index) {
  absl::MutexLock lock(mu_);
  if (index >= in_use_.size() || !in_use_[index]) {
    return;  // Stale or duplicate release.
  }
  OnReleaseLocked(index);
  in_use_[index] = false;
  if (reuse_policy_ == ReusePolicy::kLifo) {
    free_slots_.push_front(index);
  } else {
    free_slots_.push_back(index);
  }
}

size_t StagingArena::size() const {
  absl::MutexLock lock(mu_);
  return in_use_.size();
}

size_t StagingArena::available_count() const {
  absl::MutexLock lock(mu_);
  return free_slots_.size();
}

size_t StagingArena::in_use_count() const {
  absl::MutexLock lock(mu_);
  return in_use_.size() - free_slots_.size();
}

bool StagingArena::is_in_use(size_t index) const {
  absl::MutexLock lock(mu_);
  return index < in_use_.size() && in_use_[index];
}

int64_t StagingArena::capacity() const { return static_cast<int64_t>(size()); }

// ---------------------------------------------------------------------------
// LayerArena
// ---------------------------------------------------------------------------

LayerInfoBase* LayerArena::Allocation::get() {
  if (!valid()) return nullptr;
  LayerArena* layer_arena = static_cast<LayerArena*>(arena());
  return &layer_arena->GetLayer(index());
}

const LayerInfoBase* LayerArena::Allocation::get() const {
  if (!valid()) return nullptr;
  const LayerArena* layer_arena = static_cast<const LayerArena*>(arena());
  return &layer_arena->GetLayer(index());
}

LayerArena::LayerArena() : StagingArena(ReusePolicy::kLifo) {}

LayerArena::LayerArena(Config config, HostBufferAllocator host_allocator)
    : StagingArena(ReusePolicy::kLifo) {
  absl::MutexLock lock(mu_);
  config_ = config;
  host_allocator_ = std::move(host_allocator);
  chunk_capacity_ = config.initial_capacity;
  if (chunk_capacity_ == 0) {
    return;
  }
  chunk_template_.assign(
      chunk_capacity_,
      std::vector<ShardTemplate>(
          config_.num_shards,
          ShardTemplate{.alloc_size = config_.shard_host_size,
                        .device_size = config_.shard_host_size}));
  auto chunk = std::make_unique<std::vector<LayerInfoBase>>(chunk_capacity_);
  for (size_t i = 0; i < chunk_capacity_; ++i) {
    (*chunk)[i].shards.resize(config_.num_shards);
    for (size_t s = 0; s < config_.num_shards; ++s) {
      (*chunk)[i].shards[s].device_size = config_.shard_host_size;
      if (config_.shard_host_size > 0 && host_allocator_) {
        // Best effort, as before: a failed shard is left unbacked and
        // surfaces when the slot is acquired.
        (void)AllocateShardLocked((*chunk)[i].shards[s], chunk_template_[i][s]);
      }
    }
  }
  chunks_.push_back(std::move(chunk));
  for (size_t i = 0; i < chunk_capacity_; ++i) {
    AddSlotLocked();
  }
}

LayerArena::LayerArena(std::vector<LayerInfoBase> layers,
                       HostBufferAllocator host_allocator)
    : LayerArena(std::move(layers), std::move(host_allocator), Config{}) {}

LayerArena::LayerArena(std::vector<LayerInfoBase> layers,
                       HostBufferAllocator host_allocator, Config config)
    : StagingArena(ReusePolicy::kLifo) {
  absl::MutexLock lock(mu_);
  config_ = config;
  host_allocator_ = std::move(host_allocator);
  if (!layers.empty()) {
    chunk_capacity_ = layers.size();
    config_.initial_capacity = chunk_capacity_;
    if (config_.num_shards == 0 && !layers[0].shards.empty()) {
      config_.num_shards = layers[0].shards.size();
    }
    chunk_template_.resize(chunk_capacity_);
    for (size_t i = 0; i < chunk_capacity_; ++i) {
      chunk_template_[i].resize(layers[i].shards.size());
      for (size_t s = 0; s < layers[i].shards.size(); ++s) {
        const ShardBufferInfoBase& shard = layers[i].shards[s];
        size_t alloc_size = shard.host_size > 0 ? shard.host_size
                                                : (config_.shard_host_size > 0
                                                       ? config_.shard_host_size
                                                       : shard.device_size);
        chunk_template_[i][s] = ShardTemplate{
            .alloc_size = alloc_size,
            .device_size = shard.device_size,
            .tiled_size = shard.tiled_size,
            .device = shard.device,
        };
      }
    }
    chunks_.push_back(
        std::make_unique<std::vector<LayerInfoBase>>(std::move(layers)));
    for (size_t i = 0; i < chunk_capacity_; ++i) {
      AddSlotLocked();
    }
  } else if (config_.initial_capacity > 0) {
    chunk_capacity_ = config_.initial_capacity;
    chunk_template_.assign(
        chunk_capacity_,
        std::vector<ShardTemplate>(
            config_.num_shards,
            ShardTemplate{.alloc_size = config_.shard_host_size,
                          .device_size = config_.shard_host_size}));
    (void)AppendChunkLocked();
  }
}

LayerArena::LayerArena(std::vector<LayerInfoBase> layers, Config config,
                       HostBufferAllocator host_allocator)
    : LayerArena(std::move(layers), std::move(host_allocator), config) {}

void LayerArena::MoveFromLocked(LayerArena& other) {
  MoveSlotsFromLocked(other);
  config_ = other.config_;
  host_allocator_ = std::move(other.host_allocator_);
  chunk_capacity_ = other.chunk_capacity_;
  chunk_template_ = std::move(other.chunk_template_);
  chunks_ = std::move(other.chunks_);
  other.chunk_capacity_ = 0;
  other.chunk_template_.clear();
  other.chunks_.clear();
}

LayerArena::LayerArena(LayerArena&& other) noexcept
    : StagingArena(ReusePolicy::kLifo) {
  absl::MutexLock lock_this(mu_);
  absl::MutexLock lock_other(other.mu_);
  MoveFromLocked(other);
}

LayerArena& LayerArena::operator=(LayerArena&& other) noexcept {
  if (this != &other) {
    absl::MutexLock lock_this(mu_);
    absl::MutexLock lock_other(other.mu_);
    MoveFromLocked(other);
  }
  return *this;
}

LayerArena::Config LayerArena::config() const {
  absl::MutexLock lock(mu_);
  return config_;
}

absl::Status LayerArena::AllocateShardLocked(ShardBufferInfoBase& shard,
                                             const ShardTemplate& tmpl) {
  shard.device_size = tmpl.device_size;
  shard.device = tmpl.device;
  shard.tiled_size = tmpl.tiled_size;
  if (tmpl.alloc_size == 0) {
    return absl::OkStatus();
  }
  if (!host_allocator_) {
    return absl::FailedPreconditionError(
        "LayerArena requires a host_allocator to allocate host memory");
  }
  ABSL_ASSIGN_OR_RETURN(HostBufferAllocation alloc,
                        host_allocator_(tmpl.alloc_size, tmpl.device));
  if (alloc.ptr == nullptr) {
    return absl::InternalError(
        "Host allocator returned null buffer for LayerArena");
  }
  if (alloc.size < tmpl.alloc_size) {
    return absl::InternalError(
        "Host allocator returned undersized buffer for LayerArena");
  }
  shard.host_ptr = alloc.ptr;
  shard.host_size = alloc.size;
  shard.host_owner = std::move(alloc.owner);
  return absl::OkStatus();
}

absl::Status LayerArena::AppendChunkLocked() {
  if (chunk_capacity_ == 0) {
    return absl::ResourceExhaustedError(
        "LayerArena has zero chunk capacity and cannot grow");
  }
  auto chunk = std::make_unique<std::vector<LayerInfoBase>>(chunk_capacity_);
  for (size_t i = 0; i < chunk_capacity_; ++i) {
    LayerInfoBase& layer = (*chunk)[i];
    size_t num_shards = i < chunk_template_.size() ? chunk_template_[i].size()
                                                   : config_.num_shards;
    layer.shards.resize(num_shards);
    for (size_t s = 0; s < num_shards; ++s) {
      ShardTemplate tmpl =
          (i < chunk_template_.size() && s < chunk_template_[i].size())
              ? chunk_template_[i][s]
              : ShardTemplate{.alloc_size = config_.shard_host_size,
                              .device_size = config_.shard_host_size};
      ABSL_RETURN_IF_ERROR(AllocateShardLocked(layer.shards[s], tmpl));
    }
  }
  chunks_.push_back(std::move(chunk));
  for (size_t i = 0; i < chunk_capacity_; ++i) {
    AddSlotLocked();
  }
  return absl::OkStatus();
}

bool LayerArena::CanGrowLocked() const { return config_.allow_dynamic_growth; }

absl::Status LayerArena::GrowLocked(int64_t /*num_blocks*/) {
  return AppendChunkLocked();
}

absl::Status LayerArena::PrepareSlotLocked(size_t index,
                                           int64_t /*num_blocks*/) {
  // Lazily back shards that are still missing host memory.
  if (!host_allocator_ || chunk_capacity_ == 0) {
    return absl::OkStatus();
  }
  const size_t tmpl_idx = index % chunk_capacity_;
  if (tmpl_idx >= chunk_template_.size()) {
    return absl::OkStatus();
  }
  LayerInfoBase& layer = GetLayerLocked(index);
  const std::vector<ShardTemplate>& shard_templates = chunk_template_[tmpl_idx];
  for (size_t s = 0; s < layer.shards.size() && s < shard_templates.size();
       ++s) {
    if (layer.shards[s].host_ptr == nullptr &&
        shard_templates[s].alloc_size > 0) {
      ABSL_RETURN_IF_ERROR(
          AllocateShardLocked(layer.shards[s], shard_templates[s]));
    }
  }
  return absl::OkStatus();
}

void LayerArena::OnReleaseLocked(size_t index) {
  if (config_.release_memory_on_free) {
    GetLayerLocked(index).ReleaseMemory();
  }
}

absl::StatusOr<LayerArena::Allocation> LayerArena::Acquire() {
  ABSL_ASSIGN_OR_RETURN(StagingArena::Allocation base,
                        StagingArena::Acquire(/*num_blocks=*/1));
  return Allocation(std::move(base));
}

LayerInfoBase& LayerArena::GetLayerLocked(size_t layer_idx) {
  return (*chunks_[layer_idx / chunk_capacity_])[layer_idx % chunk_capacity_];
}

const LayerInfoBase& LayerArena::GetLayerLocked(size_t layer_idx) const {
  return (*chunks_[layer_idx / chunk_capacity_])[layer_idx % chunk_capacity_];
}

LayerInfoBase& LayerArena::GetLayer(size_t layer_idx) {
  absl::MutexLock lock(mu_);
  return GetLayerLocked(layer_idx);
}

const LayerInfoBase& LayerArena::GetLayer(size_t layer_idx) const {
  absl::MutexLock lock(mu_);
  return GetLayerLocked(layer_idx);
}

size_t LayerArena::num_chunks() const {
  absl::MutexLock lock(mu_);
  return chunks_.size();
}

size_t LayerArena::chunk_size() const {
  absl::MutexLock lock(mu_);
  return chunk_capacity_;
}

// ---------------------------------------------------------------------------
// HostBlockArena
// ---------------------------------------------------------------------------

absl::Span<const int> HostBlockArena::Allocation::block_ids() const {
  if (!valid()) {
    return {};
  }
  return static_cast<const HostBlockArena*>(arena())->block_ids(index());
}

HostBlockArena::HostBlockArena(
    kv_cache::LogicalBlockManager* host_block_manager, int64_t num_slots,
    int64_t max_blocks, ReusePolicy reuse_policy)
    : StagingArena(reuse_policy),
      host_block_manager_(host_block_manager),
      num_slots_(num_slots),
      max_blocks_(max_blocks) {}

absl::Status HostBlockArena::ValidateRequestLocked(int64_t num_blocks) const {
  // The injected status is kInternal, which callers treat as non-retryable, so
  // this exercises the fail-fast abort path rather than the staging retry loop.
  ABSL_RETURN_IF_ERROR(FaultInjectStatus(hooks::kStagingAllocatorAcquire));
  const int64_t cap = capacity();
  if (num_blocks > cap) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested ", num_blocks, " blocks exceeds ",
                     dynamic_host_staging() ? "the host staging pool capacity ("
                                            : "a staging slot capacity (",
                     cap, ")"));
  }
  return absl::OkStatus();
}

absl::StatusOr<HostBlockArena::Allocation> HostBlockArena::Acquire(
    int64_t num_blocks) {
  if (num_blocks <= 0) {
    return Allocation();
  }
  ABSL_ASSIGN_OR_RETURN(StagingArena::Allocation base,
                        StagingArena::Acquire(num_blocks));
  return Allocation(std::move(base));
}

absl::StatusOr<HostBlockArena::Allocation> HostBlockArena::AcquireWithTimeout(
    int64_t num_blocks, std::chrono::steady_clock::time_point deadline) {
  if (num_blocks <= 0) {
    return Allocation();
  }
  ABSL_ASSIGN_OR_RETURN(StagingArena::Allocation base,
                        StagingArena::AcquireWithTimeout(num_blocks, deadline));
  return Allocation(std::move(base));
}

// ---------------------------------------------------------------------------
// SlotArena
// ---------------------------------------------------------------------------

absl::StatusOr<std::unique_ptr<SlotArena>> SlotArena::Create(
    kv_cache::LogicalBlockManager* host_block_manager, int64_t num_slots,
    int64_t max_blocks) {
  std::unique_ptr<SlotArena> arena(
      new SlotArena(host_block_manager, num_slots, max_blocks));
  ABSL_RETURN_IF_ERROR(arena->InitializeSlotPool());
  return arena;
}

SlotArena::SlotArena(kv_cache::LogicalBlockManager* host_block_manager,
                     int64_t num_slots, int64_t max_blocks)
    : HostBlockArena(host_block_manager, num_slots, max_blocks,
                     ReusePolicy::kFifo) {}

SlotArena::~SlotArena() {
  Shutdown();
  if (host_block_manager_ != nullptr && !slot_blocks_.empty()) {
    std::vector<int> blocks_to_unlock;
    blocks_to_unlock.reserve(slot_blocks_.size() * max_blocks_);
    for (const std::vector<int>& blocks : slot_blocks_) {
      blocks_to_unlock.insert(blocks_to_unlock.end(), blocks.begin(),
                              blocks.end());
    }
    (void)host_block_manager_->Unlock(blocks_to_unlock);
  }
}

absl::Status SlotArena::InitializeSlotPool() {
  if (num_slots_ <= 0 || max_blocks_ <= 0) {
    return absl::OkStatus();
  }
  if (host_block_manager_ == nullptr) {
    return absl::FailedPreconditionError(
        "Host block manager is null; cannot initialize staging slot pool");
  }
  if (host_block_manager_->num_free_blocks() < num_slots_ * max_blocks_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Insufficient free host blocks to initialize slot pool. Required: ",
        num_slots_ * max_blocks_,
        ", Available: ", host_block_manager_->num_free_blocks()));
  }
  absl::MutexLock lock(mu_);
  slot_blocks_.reserve(num_slots_);
  for (int64_t i = 0; i < num_slots_; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        std::vector<int> allocated_ids,
        host_block_manager_->Allocate(max_blocks_, /*lock=*/true));
    if (allocated_ids.size() != max_blocks_) {
      return absl::InternalError(absl::StrCat(
          "Slot pool allocation returned incorrect number of blocks: ",
          allocated_ids.size(), ", expected: ", max_blocks_));
    }
    slot_blocks_.push_back(std::move(allocated_ids));
    AddSlotLocked();
  }
  return absl::OkStatus();
}

absl::Span<const int> SlotArena::block_ids(size_t index) const {
  return slot_blocks_[index];
}

// ---------------------------------------------------------------------------
// BlockArena
// ---------------------------------------------------------------------------

BlockArena::BlockArena(kv_cache::LogicalBlockManager* host_block_manager,
                       int64_t num_slots, int64_t max_blocks)
    : HostBlockArena(host_block_manager, num_slots, max_blocks,
                     ReusePolicy::kLifo) {}

int64_t BlockArena::capacity() const {
  return host_block_manager_ != nullptr ? host_block_manager_->total_blocks()
                                        : 0;
}

bool BlockArena::CanAcquireLocked(int64_t num_blocks) const {
  return host_block_manager_ != nullptr &&
         host_block_manager_->num_free_blocks() >= num_blocks;
}

absl::Status BlockArena::GrowLocked(int64_t /*num_blocks*/) {
  slots_.emplace_back();
  AddSlotLocked();
  return absl::OkStatus();
}

absl::Status BlockArena::PrepareSlotLocked(size_t index, int64_t num_blocks) {
  if (host_block_manager_ == nullptr) {
    return absl::FailedPreconditionError("Host block manager is null");
  }
  // Lock the pages so the pool's LRU cannot evict staging that is mid-flight.
  // An allocation miss is only reported back; this helper neither retries
  // nor logs. The producer retries it until its deadline, the consumer
  // fails it at once, and each logs its own verdict.
  ABSL_ASSIGN_OR_RETURN(
      slots_[index], host_block_manager_->Allocate(static_cast<int>(num_blocks),
                                                   /*lock=*/true));
  return absl::OkStatus();
}

void BlockArena::OnReleaseLocked(size_t index) {
  std::vector<int>& blocks = slots_[index];
  if (!blocks.empty() && host_block_manager_ != nullptr) {
    (void)host_block_manager_->Unlock(blocks);
    (void)host_block_manager_->Deallocate(blocks);
  }
  blocks.clear();
}

absl::Span<const int> BlockArena::block_ids(size_t index) const {
  absl::MutexLock lock(mu_);
  return slots_[index];
}

}  // namespace tpu_raiden

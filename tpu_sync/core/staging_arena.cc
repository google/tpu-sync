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
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/logical_block_manager.h"

namespace tpu_raiden {
namespace {
thread_local bool in_staging_release_cb = false;

class ReleaseCallbackScope {
 public:
  ReleaseCallbackScope() { in_staging_release_cb = true; }
  ~ReleaseCallbackScope() { in_staging_release_cb = false; }
  ReleaseCallbackScope(const ReleaseCallbackScope&) = delete;
  ReleaseCallbackScope& operator=(const ReleaseCallbackScope&) = delete;
};
}  // namespace

void StagingArena::Allocation::Release() {
  if (arena_ != nullptr) {
    arena_->ReleaseSlot(layer_idx_);
    arena_ = nullptr;
    layer_idx_ = 0;
    layer_ = nullptr;
  }
}

StagingArena::StagingArena(Config config, HostBufferAllocator host_allocator)
    : config_(config),
      host_allocator_(std::move(host_allocator)),
      chunk_capacity_(config.initial_capacity) {
  if (chunk_capacity_ > 0) {
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
          (void)AllocateShardUnlocked((*chunk)[i].shards[s],
                                      chunk_template_[i][s]);
        }
      }
    }
    chunk_collections_.push_back(std::move(chunk));
    in_use_.assign(chunk_capacity_, false);
    for (size_t i = 0; i < chunk_capacity_; ++i) {
      free_slots_.push_back(i);
    }
  }
}

StagingArena::StagingArena(std::vector<LayerInfoBase> layers,
                           HostBufferAllocator host_allocator)
    : StagingArena(std::move(layers), std::move(host_allocator), Config{}) {}

StagingArena::StagingArena(std::vector<LayerInfoBase> layers,
                           HostBufferAllocator host_allocator, Config config)
    : config_(config), host_allocator_(std::move(host_allocator)) {
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
    chunk_collections_.push_back(
        std::make_unique<std::vector<LayerInfoBase>>(std::move(layers)));
    in_use_.assign(chunk_capacity_, false);
    for (size_t i = 0; i < chunk_capacity_; ++i) {
      free_slots_.push_back(i);
    }
  } else if (config_.initial_capacity > 0) {
    chunk_capacity_ = config_.initial_capacity;
    chunk_template_.assign(
        chunk_capacity_,
        std::vector<ShardTemplate>(
            config_.num_shards,
            ShardTemplate{.alloc_size = config_.shard_host_size,
                          .device_size = config_.shard_host_size}));
    (void)AppendChunkUnlocked();
  }
}

StagingArena::StagingArena(std::vector<LayerInfoBase> layers, Config config,
                           HostBufferAllocator host_allocator)
    : StagingArena(std::move(layers), std::move(host_allocator), config) {}

StagingArena::StagingArena(StagingArena&& other) noexcept {
  absl::MutexLock lock(other.mu_);
  config_ = other.config_;
  host_allocator_ = std::move(other.host_allocator_);
  chunk_capacity_ = other.chunk_capacity_;
  chunk_template_ = std::move(other.chunk_template_);
  chunk_collections_ = std::move(other.chunk_collections_);
  free_slots_ = std::move(other.free_slots_);
  in_use_ = std::move(other.in_use_);
  other.chunk_capacity_ = 0;
}

StagingArena& StagingArena::operator=(StagingArena&& other) noexcept {
  if (this != &other) {
    absl::MutexLock lock_this(mu_);
    absl::MutexLock lock_other(other.mu_);
    config_ = other.config_;
    host_allocator_ = std::move(other.host_allocator_);
    chunk_capacity_ = other.chunk_capacity_;
    chunk_template_ = std::move(other.chunk_template_);
    chunk_collections_ = std::move(other.chunk_collections_);
    free_slots_ = std::move(other.free_slots_);
    in_use_ = std::move(other.in_use_);
    other.chunk_capacity_ = 0;
  }
  return *this;
}

StagingArena::Config StagingArena::config() const {
  absl::MutexLock lock(mu_);
  return config_;
}

absl::Status StagingArena::AllocateShardUnlocked(ShardBufferInfoBase& shard,
                                                 const ShardTemplate& tmpl) {
  shard.device_size = tmpl.device_size;
  shard.device = tmpl.device;
  shard.tiled_size = tmpl.tiled_size;
  if (tmpl.alloc_size == 0) {
    return absl::OkStatus();
  }
  if (!host_allocator_) {
    return absl::FailedPreconditionError(
        "StagingArena requires a host_allocator to allocate host memory");
  }
  absl::StatusOr<HostBufferAllocation> alloc =
      host_allocator_(tmpl.alloc_size, tmpl.device);
  if (!alloc.ok()) {
    return alloc.status();
  }
  if (alloc->ptr == nullptr) {
    return absl::InternalError(
        "Host allocator returned null buffer for StagingArena");
  }
  if (alloc->size < tmpl.alloc_size) {
    return absl::InternalError(
        "Host allocator returned undersized buffer for StagingArena");
  }
  shard.host_ptr = alloc->ptr;
  shard.host_size = alloc->size;
  shard.host_owner = std::move(alloc->owner);
  return absl::OkStatus();
}

absl::Status StagingArena::EnsureLayerAllocatedUnlocked(size_t layer_idx) {
  if (!host_allocator_ || chunk_capacity_ == 0) {
    return absl::OkStatus();
  }
  size_t tmpl_idx = layer_idx % chunk_capacity_;
  if (tmpl_idx >= chunk_template_.size()) {
    return absl::OkStatus();
  }
  LayerInfoBase& layer = GetLayerUnlocked(layer_idx);
  const std::vector<ShardTemplate>& shard_templates = chunk_template_[tmpl_idx];
  for (size_t s = 0; s < layer.shards.size() && s < shard_templates.size();
       ++s) {
    if (layer.shards[s].host_ptr == nullptr &&
        shard_templates[s].alloc_size > 0) {
      absl::Status status =
          AllocateShardUnlocked(layer.shards[s], shard_templates[s]);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status StagingArena::AppendChunkUnlocked() {
  if (chunk_capacity_ == 0) {
    return absl::ResourceExhaustedError(
        "StagingArena has zero chunk capacity and cannot grow");
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
      absl::Status status = AllocateShardUnlocked(layer.shards[s], tmpl);
      if (!status.ok()) {
        return status;
      }
    }
  }
  size_t base_idx = in_use_.size();
  chunk_collections_.push_back(std::move(chunk));
  in_use_.resize(base_idx + chunk_capacity_, false);
  for (size_t i = 0; i < chunk_capacity_; ++i) {
    free_slots_.push_back(base_idx + i);
  }
  return absl::OkStatus();
}

absl::StatusOr<StagingArena::Allocation> StagingArena::Acquire() {
  absl::MutexLock lock(mu_);
  if (free_slots_.empty()) {
    if (!config_.allow_dynamic_growth) {
      return absl::ResourceExhaustedError(
          "StagingArena is out of capacity and dynamic capacity growth is "
          "disabled");
    }
    absl::Status status = AppendChunkUnlocked();
    if (!status.ok()) {
      return status;
    }
  }
  size_t slot_idx = free_slots_.front();
  absl::Status alloc_status = EnsureLayerAllocatedUnlocked(slot_idx);
  if (!alloc_status.ok()) {
    return alloc_status;
  }
  free_slots_.pop_front();
  in_use_[slot_idx] = true;
  return Allocation(this, slot_idx, &GetLayerUnlocked(slot_idx));
}

void StagingArena::ReleaseSlot(size_t layer_idx) {
  absl::MutexLock lock(mu_);
  if (layer_idx >= in_use_.size() || !in_use_[layer_idx]) {
    return;
  }
  if (config_.release_memory_on_free) {
    GetLayerUnlocked(layer_idx).ReleaseMemory();
  }
  in_use_[layer_idx] = false;
  free_slots_.push_front(layer_idx);
}

LayerInfoBase& StagingArena::GetLayerUnlocked(size_t layer_idx) {
  size_t chunk_idx = layer_idx / chunk_capacity_;
  size_t offset = layer_idx % chunk_capacity_;
  return (*chunk_collections_[chunk_idx])[offset];
}

const LayerInfoBase& StagingArena::GetLayerUnlocked(size_t layer_idx) const {
  size_t chunk_idx = layer_idx / chunk_capacity_;
  size_t offset = layer_idx % chunk_capacity_;
  return (*chunk_collections_[chunk_idx])[offset];
}

LayerInfoBase& StagingArena::GetLayer(size_t layer_idx) {
  absl::MutexLock lock(mu_);
  return GetLayerUnlocked(layer_idx);
}

const LayerInfoBase& StagingArena::GetLayer(size_t layer_idx) const {
  absl::MutexLock lock(mu_);
  return GetLayerUnlocked(layer_idx);
}

LayerInfoBase& StagingArena::operator[](size_t layer_idx) {
  absl::MutexLock lock(mu_);
  return GetLayerUnlocked(layer_idx);
}

const LayerInfoBase& StagingArena::operator[](size_t layer_idx) const {
  absl::MutexLock lock(mu_);
  return GetLayerUnlocked(layer_idx);
}

size_t StagingArena::SizeUnlocked() const {
  return chunk_collections_.size() * chunk_capacity_;
}

size_t StagingArena::size() const {
  absl::MutexLock lock(mu_);
  return SizeUnlocked();
}

size_t StagingArena::num_chunks() const {
  absl::MutexLock lock(mu_);
  return chunk_collections_.size();
}

size_t StagingArena::chunk_size() const {
  absl::MutexLock lock(mu_);
  return chunk_capacity_;
}

size_t StagingArena::in_use_count() const {
  absl::MutexLock lock(mu_);
  return SizeUnlocked() - free_slots_.size();
}

size_t StagingArena::available_count() const {
  absl::MutexLock lock(mu_);
  return free_slots_.size();
}

// ---------------------------------------------------------------------------
// StagingBlockAllocator
// ---------------------------------------------------------------------------

StagingBlockAllocator::Allocation::Allocation(StagingBlockAllocator* allocator,
                                              int64_t slot_idx,
                                              std::vector<int> block_ids)
    : allocator_(allocator),
      slot_idx_(slot_idx),
      block_ids_(std::move(block_ids)) {}

StagingBlockAllocator::Allocation::Allocation(StagingBlockAllocator* allocator,
                                              std::vector<int> dynamic_blocks)
    : allocator_(allocator), block_ids_(std::move(dynamic_blocks)) {}

StagingBlockAllocator::Allocation::Allocation(Allocation&& other) noexcept
    : allocator_(other.allocator_),
      slot_idx_(other.slot_idx_),
      block_ids_(std::move(other.block_ids_)) {
  other.allocator_ = nullptr;
  other.slot_idx_ = -1;
  other.block_ids_.clear();
}

StagingBlockAllocator::Allocation& StagingBlockAllocator::Allocation::operator=(
    Allocation&& other) noexcept {
  if (this != &other) {
    Reset();
    allocator_ = other.allocator_;
    slot_idx_ = other.slot_idx_;
    block_ids_ = std::move(other.block_ids_);
    other.allocator_ = nullptr;
    other.slot_idx_ = -1;
    other.block_ids_.clear();
  }
  return *this;
}

StagingBlockAllocator::Allocation::~Allocation() { Reset(); }

void StagingBlockAllocator::Allocation::Reset() {
  if (allocator_ != nullptr) {
    if (slot_idx_ >= 0) {
      allocator_->ReleaseSlot(slot_idx_);
    } else if (!block_ids_.empty()) {
      allocator_->ReleaseDynamicBlocks(block_ids_);
    }
  }
  allocator_ = nullptr;
  slot_idx_ = -1;
  block_ids_.clear();
}

absl::StatusOr<std::unique_ptr<StagingBlockAllocator>>
StagingBlockAllocator::Create(kv_cache::LogicalBlockManager* host_block_manager,
                              int64_t num_slots, int64_t max_blocks,
                              std::optional<bool> dynamic_host_staging) {
  const char* raw_env = std::getenv("TPU_RAIDEN_DYNAMIC_HOST_STAGING");
  const bool use_dynamic_staging = dynamic_host_staging.value_or(
      raw_env != nullptr && std::string(raw_env) == "1");
  std::unique_ptr<StagingBlockAllocator> allocator(new StagingBlockAllocator(
      host_block_manager, num_slots, max_blocks, use_dynamic_staging));
  if (num_slots > 0 && max_blocks > 0) {
    ABSL_RETURN_IF_ERROR(allocator->InitializeSlotPool());
  }
  return allocator;
}

StagingBlockAllocator::StagingBlockAllocator(
    kv_cache::LogicalBlockManager* host_block_manager, int64_t num_slots,
    int64_t max_blocks, bool dynamic_host_staging)
    : host_block_manager_(host_block_manager),
      num_slots_(num_slots),
      max_blocks_(max_blocks),
      dynamic_host_staging_(dynamic_host_staging) {}

StagingBlockAllocator::~StagingBlockAllocator() {
  {
    absl::MutexLock lock(mu_);
    shutting_down_ = true;
  }
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

absl::Status StagingBlockAllocator::InitializeSlotPool() {
  absl::MutexLock lock(mu_);
  free_slots_.clear();
  slot_blocks_.clear();
  if (dynamic_host_staging_ || num_slots_ <= 0) {
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
    free_slots_.push_back(i);
  }
  return absl::OkStatus();
}

void StagingBlockAllocator::Shutdown() {
  absl::MutexLock lock(mu_);
  shutting_down_ = true;
}

absl::StatusOr<StagingBlockAllocator::Allocation>
StagingBlockAllocator::AcquireLocked(int64_t num_blocks) {
  if (num_blocks <= 0) {
    return Allocation();
  }
  // The injected status is kInternal, which callers treat as non-retryable, so
  // this exercises the fail-fast abort path rather than the staging retry loop.
  ABSL_RETURN_IF_ERROR(FaultInjectStatus(hooks::kStagingAllocatorAcquire));
  const int64_t cap = capacity();
  if (num_blocks > cap) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested ", num_blocks, " blocks exceeds ",
                     dynamic_host_staging_ ? "the host staging pool capacity ("
                                           : "a staging slot capacity (",
                     cap, ")"));
  }
  if (!dynamic_host_staging_) {
    if (free_slots_.empty()) {
      return absl::ResourceExhaustedError("No free staging slots available");
    }
    int64_t slot_idx = free_slots_.front();
    free_slots_.pop_front();
    blocks_in_use_ += slot_blocks_[slot_idx].size();
    peak_blocks_in_use_ = std::max(peak_blocks_in_use_, blocks_in_use_);
    return Allocation(this, slot_idx, slot_blocks_[slot_idx]);
  }
  if (host_block_manager_ == nullptr) {
    return absl::FailedPreconditionError("Host block manager is null");
  }
  // Lock the pages so the pool's LRU cannot evict staging that is mid-flight.
  // An allocation miss is only reported back; this helper neither retries
  // nor logs. The producer retries it until its deadline, the consumer
  // fails it at once, and each logs its own verdict.
  ABSL_ASSIGN_OR_RETURN(std::vector<int> allocated,
                        host_block_manager_->Allocate(
                            static_cast<int>(num_blocks), /*lock=*/true));
  blocks_in_use_ += allocated.size();
  peak_blocks_in_use_ = std::max(peak_blocks_in_use_, blocks_in_use_);
  return Allocation(this, std::move(allocated));
}

absl::StatusOr<StagingBlockAllocator::Allocation>
StagingBlockAllocator::Acquire(int64_t num_blocks) {
  absl::MutexLock lock(mu_);
  return AcquireLocked(num_blocks);
}

absl::StatusOr<StagingBlockAllocator::Allocation>
StagingBlockAllocator::AcquireWithTimeout(
    int64_t num_blocks, std::chrono::steady_clock::time_point deadline) {
  absl::MutexLock lock(mu_);
  if (shutting_down_) {
    return absl::CancelledError("StagingBlockAllocator is shutting down");
  }
  absl::StatusOr<Allocation> acquired = AcquireLocked(num_blocks);
  if (acquired.ok() || !absl::IsResourceExhausted(acquired.status())) {
    return acquired;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    return acquired;
  }
  auto can_proceed = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    if (shutting_down_) return true;
    if (!dynamic_host_staging_) return !free_slots_.empty();
    return host_block_manager_ != nullptr &&
           host_block_manager_->num_free_blocks() >=
               static_cast<size_t>(num_blocks);
  };
  const auto remaining_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now)
          .count();
  mu_.AwaitWithTimeout(absl::Condition(&can_proceed),
                       absl::Nanoseconds(remaining_ns));
  if (shutting_down_) {
    return absl::CancelledError("StagingBlockAllocator is shutting down");
  }
  return AcquireLocked(num_blocks);
}

void StagingBlockAllocator::SetReleaseCallback(absl::AnyInvocable<void()> cb) {
  absl::MutexLock lock(mu_);
  release_cb_ = cb ? std::make_shared<absl::AnyInvocable<void()>>(std::move(cb))
                   : nullptr;
  if (release_cb_ == nullptr) {
    if (in_staging_release_cb) {
      mu_.Await(absl::Condition(
          +[](int* active) { return *active <= 1; }, &active_release_cbs_));
    } else {
      mu_.Await(absl::Condition(
          +[](int* active) { return *active == 0; }, &active_release_cbs_));
    }
  }
}

StagingStats StagingBlockAllocator::Stats() const {
  absl::MutexLock lock(mu_);
  StagingStats stats;
  stats.budget_blocks = capacity();
  stats.blocks_in_use = blocks_in_use_;
  stats.peak_blocks_in_use = peak_blocks_in_use_;
  return stats;
}

size_t StagingBlockAllocator::num_free_slots() const {
  absl::MutexLock lock(mu_);
  return free_slots_.size();
}

absl::Span<const int> StagingBlockAllocator::slot_blocks(
    int64_t slot_idx) const {
  return slot_blocks_[slot_idx];
}

int64_t StagingBlockAllocator::capacity() const {
  if (dynamic_host_staging_) {
    return host_block_manager_ != nullptr ? host_block_manager_->total_blocks()
                                          : 0;
  }
  return max_blocks_;
}

void StagingBlockAllocator::ReleaseSlot(int64_t slot_idx) {
  std::shared_ptr<absl::AnyInvocable<void()>> cb;
  {
    absl::MutexLock lock(mu_);
    if (slot_idx < 0 || slot_idx >= num_slots_) {
      return;
    }
    free_slots_.push_back(slot_idx);
    blocks_in_use_ -= slot_blocks_[slot_idx].size();
    if (release_cb_ != nullptr && *release_cb_ && !in_staging_release_cb) {
      cb = release_cb_;
      ++active_release_cbs_;
    }
  }
  if (cb != nullptr) {
    ReleaseCallbackScope scope;
    (*cb)();
    {
      absl::MutexLock lock(mu_);
      --active_release_cbs_;
    }
  }
}

void StagingBlockAllocator::ReleaseDynamicBlocks(absl::Span<const int> blocks) {
  if (blocks.empty() || host_block_manager_ == nullptr) {
    return;
  }
  std::shared_ptr<absl::AnyInvocable<void()>> cb;
  {
    absl::MutexLock lock(mu_);
    std::vector<int> block_vec(blocks.begin(), blocks.end());
    (void)host_block_manager_->Unlock(block_vec);
    (void)host_block_manager_->Deallocate(block_vec);
    blocks_in_use_ -= blocks.size();
    if (release_cb_ != nullptr && *release_cb_ && !in_staging_release_cb) {
      cb = release_cb_;
      ++active_release_cbs_;
    }
  }
  if (cb != nullptr) {
    ReleaseCallbackScope scope;
    (*cb)();
    {
      absl::MutexLock lock(mu_);
      --active_release_cbs_;
    }
  }
}

}  // namespace tpu_raiden

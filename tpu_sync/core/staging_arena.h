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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_STAGING_ARENA_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_STAGING_ARENA_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/kv_cache/logical_block_manager.h"

namespace tpu_raiden {

struct ShardBufferInfoBase {
  const uint8_t* host_ptr = nullptr;
  size_t host_size = 0;
  size_t device_size = 0;
  const xla::PjRtDevice* device = nullptr;
  std::unique_ptr<uint8_t[], void (*)(void*)> owned_host_buffer = {
      nullptr, [](void*) {}};
  std::shared_ptr<void> host_owner;
  std::unique_ptr<uint8_t[], void (*)(void*)> owned_tiled_buffer = {
      nullptr, [](void*) {}};
  std::shared_ptr<void> tiled_owner;
  uint8_t* tiled_ptr = nullptr;
  size_t tiled_size = 0;

  // Releases underlying host and tiled buffer allocations while preserving
  // metadata such as |device_size| and |device|.
  void ReleaseMemory() {
    host_ptr = nullptr;
    host_size = 0;
    owned_host_buffer.reset();
    host_owner.reset();
    owned_tiled_buffer.reset();
    tiled_owner.reset();
    tiled_ptr = nullptr;
    tiled_size = 0;
  }
};

struct LayerInfoBase {
  std::vector<ShardBufferInfoBase> shards;

  // Releases underlying host and tiled memory across all shards in this
  // layer while preserving the shard vector size.
  void ReleaseMemory() {
    for (ShardBufferInfoBase& shard : shards) {
      shard.ReleaseMemory();
    }
  }
};

// Adapts a non-owning |allocator| pointer into a `HostBufferAllocator`
// callback that invokes `AllocateDmaMappedForDevice`.
inline HostBufferAllocator MakeHostBufferAllocator(
    HostMemoryAllocator* allocator) {
  if (allocator == nullptr) {
    return nullptr;
  }
  return [allocator](size_t size_bytes, const xla::PjRtDevice* device)
             -> absl::StatusOr<HostBufferAllocation> {
    return allocator->AllocateDmaMappedForDevice(size_bytes, device);
  };
}

// Returns a default CPU `HostBufferAllocator` backed by
// `MallocHostMemoryAllocator` (`posix_memalign`).
inline HostBufferAllocator MakeDefaultHostBufferAllocator() {
  auto allocator = std::make_shared<MallocHostMemoryAllocator>();
  return [allocator](size_t size_bytes, const xla::PjRtDevice* /*device*/)
             -> absl::StatusOr<HostBufferAllocation> {
    return allocator->Allocate(size_bytes);
  };
}

// Manages layer staging metadata (`LayerInfoBase`) across one or more
// fixed-size `std::vector<LayerInfoBase>` chunk collections.
//
// Supports both:
// 1. Dynamic slot acquisition via `Acquire()` returning a move-only RAII
//    `Allocation` handle that returns the slot to the arena when destroyed.
//    When out of capacity and `Config::allow_dynamic_growth` is enabled,
//    the arena grows by appending a new `std::vector<LayerInfoBase>` with the
//    same size as the previous vector and allocating backing host buffers via
//    the configured `HostBufferAllocator`, preserving pointer/reference
//    stability for all existing `Allocation` handles.
// 2. Backward-compatible index-based access (`GetLayer(layer_idx)` and
//    `operator[](layer_idx)`).
//
// Thread-safe for concurrent `Acquire()` and `Allocation` release.
class StagingArena {
 public:
  struct Config {
    // Number of `LayerInfoBase` slots in each allocated chunk vector.
    size_t initial_capacity = 0;
    // Number of shards to pre-size on each `LayerInfoBase` slot when a chunk
    // is allocated.
    size_t num_shards = 0;
    // Per-shard host buffer size in bytes when allocating without an initial
    // `layers` template.
    size_t shard_host_size = 0;
    // If true, appends a new `std::vector<LayerInfoBase>` of the same size as
    // the previous chunk when `Acquire()` is called and no free slots remain.
    bool allow_dynamic_growth = false;
    // If true, calls `LayerInfoBase::ReleaseMemory()` when an `Allocation`
    // returns its slot to the arena.
    bool release_memory_on_free = false;
  };

  // Move-only RAII handle representing an acquired `LayerInfoBase` slot in
  // `StagingArena`. Automatically returns the slot to the arena on
  // destruction unless already released via `Release()`.
  class Allocation {
   public:
    Allocation() = default;
    Allocation(StagingArena* arena, size_t layer_idx, LayerInfoBase* layer)
        : arena_(arena), layer_idx_(layer_idx), layer_(layer) {}

    ~Allocation() { Release(); }

    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

    Allocation(Allocation&& other) noexcept
        : arena_(other.arena_),
          layer_idx_(other.layer_idx_),
          layer_(other.layer_) {
      other.arena_ = nullptr;
      other.layer_idx_ = 0;
      other.layer_ = nullptr;
    }

    Allocation& operator=(Allocation&& other) noexcept {
      if (this != &other) {
        Release();
        arena_ = other.arena_;
        layer_idx_ = other.layer_idx_;
        layer_ = other.layer_;
        other.arena_ = nullptr;
        other.layer_idx_ = 0;
        other.layer_ = nullptr;
      }
      return *this;
    }

    // Explicitly returns the acquired slot to the arena before destruction.
    void Release();

    // Frees underlying host/tiled memory on this slot's `LayerInfoBase`.
    void ReleaseMemory() {
      if (layer_ != nullptr) {
        layer_->ReleaseMemory();
      }
    }

    bool valid() const { return arena_ != nullptr && layer_ != nullptr; }
    explicit operator bool() const { return valid(); }

    size_t layer_idx() const { return layer_idx_; }
    size_t index() const { return layer_idx_; }

    LayerInfoBase& layer() { return *layer_; }
    const LayerInfoBase& layer() const { return *layer_; }
    LayerInfoBase* get() { return layer_; }
    const LayerInfoBase* get() const { return layer_; }
    LayerInfoBase& operator*() { return *layer_; }
    const LayerInfoBase& operator*() const { return *layer_; }
    LayerInfoBase* operator->() { return layer_; }
    const LayerInfoBase* operator->() const { return layer_; }

   private:
    StagingArena* arena_ = nullptr;
    size_t layer_idx_ = 0;
    LayerInfoBase* layer_ = nullptr;
  };

  StagingArena() = default;
  explicit StagingArena(Config config,
                        HostBufferAllocator host_allocator = nullptr);
  explicit StagingArena(std::vector<LayerInfoBase> layers,
                        HostBufferAllocator host_allocator = nullptr);
  StagingArena(std::vector<LayerInfoBase> layers,
               HostBufferAllocator host_allocator, Config config);
  StagingArena(std::vector<LayerInfoBase> layers, Config config,
               HostBufferAllocator host_allocator = nullptr);

  StagingArena(const StagingArena&) = delete;
  StagingArena& operator=(const StagingArena&) = delete;

  StagingArena(StagingArena&& other) noexcept;
  StagingArena& operator=(StagingArena&& other) noexcept;

  Config config() const;

  // Acquires a free `LayerInfoBase` slot from the arena. If all existing
  // slots are in use, grows the arena by adding another
  // `std::vector<LayerInfoBase>` of the same size and allocating its host
  // buffers via `host_allocator_` when `Config::allow_dynamic_growth` is
  // enabled, or returns `absl::ResourceExhaustedError` otherwise.
  absl::StatusOr<Allocation> Acquire();

  // Backward-compatible direct access to a specific layer by |layer_idx|.
  LayerInfoBase& GetLayer(size_t layer_idx);
  const LayerInfoBase& GetLayer(size_t layer_idx) const;
  LayerInfoBase& operator[](size_t layer_idx);
  const LayerInfoBase& operator[](size_t layer_idx) const;

  size_t size() const;
  size_t capacity() const { return size(); }

  // Arena inspection helpers.
  size_t num_chunks() const;
  size_t chunk_size() const;
  size_t in_use_count() const;
  size_t available_count() const;

 private:
  friend class Allocation;

  struct ShardTemplate {
    size_t alloc_size = 0;
    size_t device_size = 0;
    size_t tiled_size = 0;
    const xla::PjRtDevice* device = nullptr;
  };

  void ReleaseSlot(size_t layer_idx);
  absl::Status AllocateShardUnlocked(ShardBufferInfoBase& shard,
                                     const ShardTemplate& tmpl)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status EnsureLayerAllocatedUnlocked(size_t layer_idx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status AppendChunkUnlocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  LayerInfoBase& GetLayerUnlocked(size_t layer_idx)
      ABSL_SHARED_LOCKS_REQUIRED(mu_);
  const LayerInfoBase& GetLayerUnlocked(size_t layer_idx) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);
  size_t SizeUnlocked() const ABSL_SHARED_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  Config config_ ABSL_GUARDED_BY(mu_);
  HostBufferAllocator host_allocator_ ABSL_GUARDED_BY(mu_);
  size_t chunk_capacity_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<std::vector<ShardTemplate>> chunk_template_ ABSL_GUARDED_BY(mu_);
  std::vector<std::unique_ptr<std::vector<LayerInfoBase>>> chunk_collections_
      ABSL_GUARDED_BY(mu_);
  std::deque<size_t> free_slots_ ABSL_GUARDED_BY(mu_);
  std::vector<bool> in_use_ ABSL_GUARDED_BY(mu_);
};

// Counters and statistics for host staging allocations and admission.
struct StagingStats {
  int64_t budget_blocks = 0;
  int64_t blocks_in_use = 0;
  int64_t peak_blocks_in_use = 0;
  int64_t gate_waits_over_1ms = 0;
  int64_t rejected_streams = 0;
  int64_t admission_rejects = 0;
  int64_t admission_waits = 0;
  int64_t admission_requeues = 0;
  int64_t admission_timeouts = 0;
};

// Encapsulates host staging block allocation and release (both fixed slots
// and on-demand dynamic host blocks) behind an RAII |Allocation| handle that
// can be transferred into session classes.
//
// Fixed slots: `num_slots` slots of `max_blocks` host blocks each are
// allocated and locked in |host_block_manager| up front; `Acquire()` hands
// out whole slots.
// Dynamic staging (`dynamic_host_staging`): no slots are reserved; each
// `Acquire(num_blocks)` locks exactly |num_blocks| blocks in
// |host_block_manager| and releases them on `Allocation` destruction.
//
// Thread-safe. |host_block_manager| must outlive the allocator; it may be
// null, in which case only empty (`num_blocks <= 0`) acquisitions succeed.
class StagingBlockAllocator {
 public:
  // RAII handle owning either a fixed staging slot or a set of dynamically
  // allocated locked host blocks. Releases the resource automatically upon
  // destruction or Reset().
  class Allocation {
   public:
    Allocation() = default;
    Allocation(std::nullptr_t) {}  // NOLINT(runtime/explicit)
    Allocation(StagingBlockAllocator* allocator, int64_t slot_idx,
               std::vector<int> block_ids);
    Allocation(StagingBlockAllocator* allocator,
               std::vector<int> dynamic_blocks);
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    ~Allocation();

    // Releases any held slot or dynamic host blocks back to the allocator.
    // Safe to call multiple times.
    void Reset();

    bool empty() const { return slot_idx_ < 0 && block_ids_.empty(); }
    explicit operator bool() const { return !empty(); }
    bool is_dynamic() const { return slot_idx_ < 0 && !block_ids_.empty(); }
    int64_t slot_idx() const { return slot_idx_; }
    absl::Span<const int> block_ids() const { return block_ids_; }

   private:
    StagingBlockAllocator* allocator_ = nullptr;
    int64_t slot_idx_ = -1;
    std::vector<int> block_ids_;
  };

  // Creates an allocator over |host_block_manager|. With |num_slots| and
  // |max_blocks| both positive and fixed staging, reserves the slot pool
  // eagerly and fails with `kFailedPrecondition` when |host_block_manager|
  // lacks `num_slots * max_blocks` free blocks. |dynamic_host_staging|
  // defaults to the `TPU_RAIDEN_DYNAMIC_HOST_STAGING=1` environment switch.
  static absl::StatusOr<std::unique_ptr<StagingBlockAllocator>> Create(
      kv_cache::LogicalBlockManager* host_block_manager, int64_t num_slots,
      int64_t max_blocks,
      std::optional<bool> dynamic_host_staging = std::nullopt);
  ~StagingBlockAllocator();
  StagingBlockAllocator(const StagingBlockAllocator&) = delete;
  StagingBlockAllocator& operator=(const StagingBlockAllocator&) = delete;

  // Acquires staging for |num_blocks| blocks: a fixed slot when
  // !dynamic_host_staging_, or |num_blocks| locked dynamic host blocks when
  // dynamic_host_staging_ is enabled. Returns an error if staging capacity is
  // currently unavailable.
  absl::StatusOr<Allocation> Acquire(int64_t num_blocks);

  // Waits until |deadline| to acquire staging for |num_blocks| blocks. Returns
  // an error if |num_blocks| exceeds capacity(), if Shutdown() is called, or if
  // |deadline| expires before staging capacity becomes available.
  absl::StatusOr<Allocation> AcquireWithTimeout(
      int64_t num_blocks, std::chrono::steady_clock::time_point deadline);

  // Signals shutdown to wake any threads waiting in AcquireWithTimeout().
  void Shutdown();

  // Registers a release notification hook invoked after a staging slot or
  // dynamic blocks are released. The hook runs outside the allocator's mutex.
  // The callback must only post work; nested releases on the callback thread
  // are coalesced (the hook will not be re-invoked re-entrantly).
  // Passing nullptr disarms the hook and waits for any active callback to
  // finish.
  void SetReleaseCallback(absl::AnyInvocable<void()> cb);

  // Returns memory and allocation counters.
  StagingStats Stats() const;

  size_t num_free_slots() const;
  absl::Span<const int> slot_blocks(int64_t slot_idx) const;
  int64_t num_slots() const { return num_slots_; }
  int64_t max_blocks() const { return max_blocks_; }
  int64_t capacity() const;
  bool dynamic_host_staging() const { return dynamic_host_staging_; }

 private:
  StagingBlockAllocator(kv_cache::LogicalBlockManager* host_block_manager,
                        int64_t num_slots, int64_t max_blocks,
                        bool dynamic_host_staging);
  absl::Status InitializeSlotPool();
  absl::StatusOr<Allocation> AcquireLocked(int64_t num_blocks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseSlot(int64_t slot_idx);
  void ReleaseDynamicBlocks(absl::Span<const int> blocks);

  mutable absl::Mutex mu_;
  kv_cache::LogicalBlockManager* const host_block_manager_ = nullptr;
  bool shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  const int64_t num_slots_ = 0;
  const int64_t max_blocks_ = 0;
  const bool dynamic_host_staging_ = false;
  int active_release_cbs_ ABSL_GUARDED_BY(mu_) = 0;
  std::shared_ptr<absl::AnyInvocable<void()>> release_cb_ ABSL_GUARDED_BY(mu_);
  int64_t blocks_in_use_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t peak_blocks_in_use_ ABSL_GUARDED_BY(mu_) = 0;
  std::deque<int64_t> free_slots_ ABSL_GUARDED_BY(mu_);
  std::vector<std::vector<int>> slot_blocks_;
};
using StagingAllocation = StagingBlockAllocator::Allocation;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_STAGING_ARENA_H_

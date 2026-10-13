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
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
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

// Base class for the host staging pools in tpu_sync.
//
// A staging arena is a set of indexed slots that are handed out one at a
// time through a move-only RAII `Allocation` and returned when the handle is
// destroyed. The base class owns everything that is independent of what a
// slot holds:
//   * the mutex, free list and in-use bitmap,
//   * `Acquire()` / `AcquireWithDeadline()` / `AcquireWithTimeout()`,
//     including the deadline-bounded wait for a slot to be released,
//   * `Shutdown()`, which wakes waiters and fails subsequent acquires with
//     `kCancelled`,
//   * `size()` / `available_count()` / `in_use_count()`.
//
// Subclasses own the per-slot payload and customise behaviour through the
// protected `*Locked()` hooks, all of which run with `mu_` held:
//   * `LayerArena`  - `LayerInfoBase` slots (weight sync / KV cache layers).
//   * `SlotArena`   - fixed slots of pre-locked host blocks.
//   * `BlockArena`  - per-request host blocks (dynamic host staging).
//
// Thread-safe. Hooks must not call back into the arena's public API.
class StagingArena {
 public:
  // Order in which released slots are handed out again.
  enum class ReusePolicy {
    kLifo,  // most recently released slot first
    kFifo,  // release order (round-robin over the pool)
  };

  // Move-only RAII handle for an acquired slot. Returns the slot to the arena
  // on destruction unless already returned via `Release()`. A
  // default-constructed handle is empty.
  class Allocation {
   public:
    Allocation() = default;
    Allocation(std::nullptr_t) {}  // NOLINT(runtime/explicit)
    Allocation(StagingArena* arena, size_t index)
        : arena_(arena), index_(index) {}
    ~Allocation() { Release(); }

    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept
        : arena_(std::exchange(other.arena_, nullptr)),
          index_(std::exchange(other.index_, 0)) {}
    Allocation& operator=(Allocation&& other) noexcept {
      if (this != &other) {
        Release();
        arena_ = std::exchange(other.arena_, nullptr);
        index_ = std::exchange(other.index_, 0);
      }
      return *this;
    }

    // Returns the slot to the arena. Safe to call multiple times.
    void Release();
    // Alias of `Release()`.
    void Reset() { Release(); }

    bool valid() const { return arena_ != nullptr; }
    bool empty() const { return !valid(); }
    explicit operator bool() const { return valid(); }

    // Slot index within `arena()`; only meaningful when `valid()`.
    size_t index() const { return index_; }
    StagingArena* arena() const { return arena_; }

   private:
    StagingArena* arena_ = nullptr;
    size_t index_ = 0;
  };

  virtual ~StagingArena() = default;

  StagingArena(const StagingArena&) = delete;
  StagingArena& operator=(const StagingArena&) = delete;

  // Acquires a slot sized for |num_blocks| without waiting. Returns
  // `kResourceExhausted` when no slot is available, `kCancelled` after
  // `Shutdown()`, or whatever the subclass hooks report (e.g.
  // `kInvalidArgument` when |num_blocks| exceeds `capacity()`).
  // |num_blocks| is ignored by arenas whose slots have no size (LayerArena).
  absl::StatusOr<Allocation> Acquire(int64_t num_blocks = 1);

  // Like `Acquire()`, but waits until |deadline| for a slot to become
  // available when the arena is exhausted. Returns `kResourceExhausted` if
  // |deadline| expires first and `kCancelled` if `Shutdown()` is called while
  // waiting.
  absl::StatusOr<Allocation> AcquireWithDeadline(int64_t num_blocks,
                                                 absl::Time deadline);

  // `AcquireWithDeadline()` for callers that track steady_clock deadlines.
  absl::StatusOr<Allocation> AcquireWithTimeout(
      int64_t num_blocks, std::chrono::steady_clock::time_point deadline);

  // Wakes every waiter and fails all subsequent acquires with `kCancelled`.
  // Outstanding allocations can still be released.
  void Shutdown();

  // Total number of slots (free and in use).
  size_t size() const;
  size_t available_count() const;
  size_t in_use_count() const;
  // True iff |index| is currently handed out through an `Allocation`.
  bool is_in_use(size_t index) const;

  // Arena capacity in the subclass' natural unit (slots by default; host
  // blocks for the block arenas).
  virtual int64_t capacity() const;

 protected:
  explicit StagingArena(ReusePolicy reuse_policy = ReusePolicy::kLifo)
      : reuse_policy_(reuse_policy) {}

  // Transfers the slot bookkeeping of |other| into `*this`. Subclasses that
  // are movable call this from their move operations with both mutexes held.
  void MoveSlotsFromLocked(StagingArena& other)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_, other.mu_);

  // Registers a new free slot and returns its index. The subclass must have
  // stored the matching payload at that index before releasing `mu_`.
  size_t AddSlotLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // --- Hooks, all invoked with `mu_` held. ---------------------------------

  // Rejects a request up front (e.g. `num_blocks > capacity()`). Runs before
  // any waiting. Default: accept.
  virtual absl::Status ValidateRequestLocked(int64_t num_blocks) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return absl::OkStatus();
  }

  // Whether an acquire for |num_blocks| could succeed right now; the deadline
  // wait blocks until this (or shutdown) becomes true. Default: a free slot
  // exists or the arena can grow.
  virtual bool CanAcquireLocked(int64_t num_blocks) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Whether `GrowLocked()` may add slots. Default: false.
  virtual bool CanGrowLocked() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return false;
  }

  // Adds at least one free slot (payload + `AddSlotLocked()`). Called when the
  // free list is empty and `CanGrowLocked()`. Default: `kResourceExhausted`.
  virtual absl::Status GrowLocked(int64_t num_blocks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Prepares slot |index| for a request of |num_blocks| right before it is
  // handed out. On error the slot stays free. Default: no-op.
  virtual absl::Status PrepareSlotLocked(size_t index, int64_t num_blocks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return absl::OkStatus();
  }

  // Cleans up slot |index| right after it is returned. Default: no-op.
  virtual void OnReleaseLocked(size_t index)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {}

  mutable absl::Mutex mu_;

 private:
  friend class Allocation;

  absl::StatusOr<Allocation> AcquireLocked(int64_t num_blocks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseSlot(size_t index);

  ReusePolicy reuse_policy_;
  bool shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  std::deque<size_t> free_slots_ ABSL_GUARDED_BY(mu_);
  std::vector<bool> in_use_ ABSL_GUARDED_BY(mu_);
};

// `StagingArena` of layer staging metadata (`LayerInfoBase`), one slot per
// layer, stored in one or more fixed-size chunks so that slot references stay
// valid when the arena grows.
//
// Supports both:
// 1. Slot acquisition via `Acquire()`. When out of capacity and
//    `Config::allow_dynamic_growth` is enabled, the arena appends another
//    chunk with the same shard layout as the first one and allocates its
//    host buffers via the configured `HostBufferAllocator`. Shards whose host
//    buffer is missing are allocated lazily on acquire.
// 2. Index-based access (`GetLayer(layer_idx)` / `operator[]`) for callers
//    that use the arena as a plain layer container.
//
// Movable (`layers_ = LayerArena(...)`); moving with outstanding allocations
// is undefined.
class LayerArena : public StagingArena {
 public:
  struct Config {
    // Number of `LayerInfoBase` slots in each allocated chunk.
    size_t initial_capacity = 0;
    // Number of shards to pre-size on each `LayerInfoBase` slot when a chunk
    // is allocated.
    size_t num_shards = 0;
    // Per-shard host buffer size in bytes when allocating without an initial
    // `layers` template.
    size_t shard_host_size = 0;
    // If true, appends a chunk of `initial_capacity` slots when `Acquire()`
    // is called and no free slots remain.
    bool allow_dynamic_growth = false;
    // If true, calls `LayerInfoBase::ReleaseMemory()` when an `Allocation`
    // returns its slot to the arena.
    bool release_memory_on_free = false;
  };

  // `StagingArena::Allocation` with typed access to the layer.
  class Allocation : public StagingArena::Allocation {
   public:
    Allocation() = default;
    explicit Allocation(StagingArena::Allocation&& base)
        : StagingArena::Allocation(std::move(base)) {}

    // Frees underlying host/tiled memory on this slot's `LayerInfoBase`.
    void ReleaseMemory() {
      if (valid()) layer().ReleaseMemory();
    }

    size_t layer_idx() const { return index(); }
    LayerInfoBase& layer() { return *get(); }
    const LayerInfoBase& layer() const { return *get(); }
    LayerInfoBase* get();
    const LayerInfoBase* get() const;
    LayerInfoBase& operator*() { return layer(); }
    const LayerInfoBase& operator*() const { return layer(); }
    LayerInfoBase* operator->() { return get(); }
    const LayerInfoBase* operator->() const { return get(); }
  };

  LayerArena();
  explicit LayerArena(Config config,
                      HostBufferAllocator host_allocator = nullptr);
  explicit LayerArena(std::vector<LayerInfoBase> layers,
                      HostBufferAllocator host_allocator = nullptr);
  LayerArena(std::vector<LayerInfoBase> layers,
             HostBufferAllocator host_allocator, Config config);
  LayerArena(std::vector<LayerInfoBase> layers, Config config,
             HostBufferAllocator host_allocator = nullptr);
  ~LayerArena() override = default;

  LayerArena(LayerArena&& other) noexcept;
  LayerArena& operator=(LayerArena&& other) noexcept;

  Config config() const;

  // Acquires a free layer slot (see class comment for growth semantics).
  absl::StatusOr<Allocation> Acquire();

  // Direct access to a specific layer by |layer_idx|.
  LayerInfoBase& GetLayer(size_t layer_idx);
  const LayerInfoBase& GetLayer(size_t layer_idx) const;
  LayerInfoBase& operator[](size_t layer_idx) { return GetLayer(layer_idx); }
  const LayerInfoBase& operator[](size_t layer_idx) const {
    return GetLayer(layer_idx);
  }

  // Arena inspection helpers.
  size_t num_chunks() const;
  size_t chunk_size() const;

 protected:
  bool CanGrowLocked() const override ABSL_SHARED_LOCKS_REQUIRED(mu_);
  absl::Status GrowLocked(int64_t num_blocks) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status PrepareSlotLocked(size_t index, int64_t num_blocks) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void OnReleaseLocked(size_t index) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

 private:
  struct ShardTemplate {
    size_t alloc_size = 0;
    size_t device_size = 0;
    size_t tiled_size = 0;
    const xla::PjRtDevice* device = nullptr;
  };

  void MoveFromLocked(LayerArena& other)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_, other.mu_);
  absl::Status AllocateShardLocked(ShardBufferInfoBase& shard,
                                   const ShardTemplate& tmpl)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status AppendChunkLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  LayerInfoBase& GetLayerLocked(size_t layer_idx)
      ABSL_SHARED_LOCKS_REQUIRED(mu_);
  const LayerInfoBase& GetLayerLocked(size_t layer_idx) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  Config config_ ABSL_GUARDED_BY(mu_);
  HostBufferAllocator host_allocator_ ABSL_GUARDED_BY(mu_);
  size_t chunk_capacity_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<std::vector<ShardTemplate>> chunk_template_ ABSL_GUARDED_BY(mu_);
  std::vector<std::unique_ptr<std::vector<LayerInfoBase>>> chunks_
      ABSL_GUARDED_BY(mu_);
};

// `StagingArena` whose slots are sets of host block ids in a
// `kv_cache::LogicalBlockManager`. Shared contract of `SlotArena` and
// `BlockArena` so transfer sessions can stage into either.
//
// |host_block_manager| must outlive the arena; it may be null, in which case
// only empty (`num_blocks <= 0`) acquisitions succeed.
class HostBlockArena : public StagingArena {
 public:
  // `StagingArena::Allocation` with access to the staged host block ids.
  class Allocation : public StagingArena::Allocation {
   public:
    Allocation() = default;
    Allocation(std::nullptr_t) {}  // NOLINT(runtime/explicit)
    explicit Allocation(StagingArena::Allocation&& base)
        : StagingArena::Allocation(std::move(base)) {}

    // Host block ids backing this allocation; empty when `!valid()`.
    absl::Span<const int> block_ids() const;
    // Slot index, or -1 when `!valid()`.
    int64_t slot_idx() const {
      return valid() ? static_cast<int64_t>(index()) : -1;
    }
  };

  // Acquires staging for |num_blocks| blocks. |num_blocks <= 0| yields an
  // empty allocation.
  absl::StatusOr<Allocation> Acquire(int64_t num_blocks);
  absl::StatusOr<Allocation> AcquireWithTimeout(
      int64_t num_blocks, std::chrono::steady_clock::time_point deadline);

  // Host block ids backing slot |index|.
  virtual absl::Span<const int> block_ids(size_t index) const = 0;

  // True for arenas that lock blocks per request (`BlockArena`).
  virtual bool dynamic_host_staging() const = 0;

  // Configured slot layout (informational; `BlockArena` does not reserve
  // slots but still reports what it was configured with).
  int64_t num_slots() const { return num_slots_; }
  int64_t max_blocks() const { return max_blocks_; }
  size_t num_free_slots() const { return available_count(); }
  kv_cache::LogicalBlockManager* host_block_manager() const {
    return host_block_manager_;
  }

 protected:
  HostBlockArena(kv_cache::LogicalBlockManager* host_block_manager,
                 int64_t num_slots, int64_t max_blocks,
                 ReusePolicy reuse_policy);

  // Fault injection + `num_blocks > capacity()` check.
  absl::Status ValidateRequestLocked(int64_t num_blocks) const override
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  kv_cache::LogicalBlockManager* const host_block_manager_;
  const int64_t num_slots_;
  const int64_t max_blocks_;
};
using StagingAllocation = HostBlockArena::Allocation;

// `HostBlockArena` with `num_slots` fixed slots of `max_blocks` host blocks
// each, allocated and locked in the host block manager up front. `Acquire()`
// hands out whole slots in FIFO order; the slot pool is unlocked (but left
// allocated) on destruction.
class SlotArena : public HostBlockArena {
 public:
  // Reserves the slot pool. Fails with `kFailedPrecondition` when
  // |host_block_manager| is null or lacks `num_slots * max_blocks` free
  // blocks.
  static absl::StatusOr<std::unique_ptr<SlotArena>> Create(
      kv_cache::LogicalBlockManager* host_block_manager, int64_t num_slots,
      int64_t max_blocks);
  ~SlotArena() override;

  absl::Span<const int> block_ids(size_t index) const override;
  absl::Span<const int> slot_blocks(int64_t slot_idx) const {
    return block_ids(static_cast<size_t>(slot_idx));
  }
  bool dynamic_host_staging() const override { return false; }
  // Blocks per slot.
  int64_t capacity() const override { return max_blocks_; }

 private:
  SlotArena(kv_cache::LogicalBlockManager* host_block_manager,
            int64_t num_slots, int64_t max_blocks);
  absl::Status InitializeSlotPool();

  // Immutable after `InitializeSlotPool()`.
  std::vector<std::vector<int>> slot_blocks_;
};

// `HostBlockArena` that reserves nothing up front: every `Acquire(n)` locks
// exactly |n| blocks in the host block manager (so the pool's LRU cannot
// evict staging that is mid-flight) and releases them when the allocation is
// destroyed. Slots are created on demand and recycled.
class BlockArena : public HostBlockArena {
 public:
  // |num_slots| / |max_blocks| are informational only (see `HostBlockArena`).
  BlockArena(kv_cache::LogicalBlockManager* host_block_manager,
             int64_t num_slots, int64_t max_blocks);
  ~BlockArena() override = default;

  absl::Span<const int> block_ids(size_t index) const override;
  bool dynamic_host_staging() const override { return true; }
  // Total blocks in the host block manager.
  int64_t capacity() const override;

 protected:
  bool CanAcquireLocked(int64_t num_blocks) const override
      ABSL_SHARED_LOCKS_REQUIRED(mu_);
  bool CanGrowLocked() const override ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return true;
  }
  absl::Status GrowLocked(int64_t num_blocks) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status PrepareSlotLocked(size_t index, int64_t num_blocks) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void OnReleaseLocked(size_t index) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

 private:
  // Pointer-stable so `block_ids()` spans survive growth.
  std::deque<std::vector<int>> slots_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_STAGING_ARENA_H_

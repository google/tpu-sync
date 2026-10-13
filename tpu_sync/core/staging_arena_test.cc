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
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/kv_cache/logical_block_manager.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(LayerArenaTest, LayerOrderAcquireAndReleaseReuse) {
  LayerArena arena(LayerArena::Config{
      .initial_capacity = 2,
      .num_shards = 2,
      .allow_dynamic_growth = false,
  });

  EXPECT_EQ(arena.size(), 2);
  EXPECT_EQ(arena.num_chunks(), 1);
  EXPECT_EQ(arena.chunk_size(), 2);
  EXPECT_EQ(arena.available_count(), 2);
  EXPECT_EQ(arena.in_use_count(), 0);

  uint8_t dummy_buf_a = 0xAA;
  uint8_t dummy_buf_b = 0xBB;

  // Simulate processing 6 layers in layer order while only holding 1 slot at a
  // time. The arena should reuse slot 0 without growing.
  for (size_t step = 0; step < 6; ++step) {
    absl::StatusOr<LayerArena::Allocation> alloc = arena.Acquire();
    ASSERT_THAT(alloc, IsOk());
    ASSERT_TRUE(alloc->valid());
    EXPECT_EQ(alloc->layer_idx(), 0);
    EXPECT_EQ(arena.in_use_count(), 1);
    EXPECT_EQ(arena.available_count(), 1);
    ASSERT_EQ((*alloc)->shards.size(), 2);

    if (step == 0) {
      (*alloc)->shards[0].host_ptr = &dummy_buf_a;
      (*alloc)->shards[0].host_size = 64;
      (*alloc)->shards[1].host_ptr = &dummy_buf_b;
      (*alloc)->shards[1].host_size = 128;
    } else {
      // Buffer pointers remain intact across releases when
      // release_memory_on_free is false.
      EXPECT_EQ((*alloc)->shards[0].host_ptr, &dummy_buf_a);
      EXPECT_EQ((*alloc)->shards[0].host_size, 64);
      EXPECT_EQ((*alloc)->shards[1].host_ptr, &dummy_buf_b);
      EXPECT_EQ((*alloc)->shards[1].host_size, 128);
    }

    // Verify backward-compatible index access points to the same LayerInfoBase.
    EXPECT_EQ(&arena.GetLayer(alloc->layer_idx()), alloc->get());
    EXPECT_EQ(&arena[alloc->layer_idx()], alloc->get());
  }

  EXPECT_EQ(arena.in_use_count(), 0);
  EXPECT_EQ(arena.available_count(), 2);
  EXPECT_EQ(arena.num_chunks(), 1);
}

TEST(LayerArenaTest, OutOfCapacityWithoutDynamicGrowthReturnsError) {
  LayerArena arena(LayerArena::Config{
      .initial_capacity = 2,
      .num_shards = 1,
      .allow_dynamic_growth = false,
  });

  absl::StatusOr<LayerArena::Allocation> alloc0 = arena.Acquire();
  ASSERT_THAT(alloc0, IsOk());
  absl::StatusOr<LayerArena::Allocation> alloc1 = arena.Acquire();
  ASSERT_THAT(alloc1, IsOk());
  EXPECT_EQ(alloc0->layer_idx(), 0);
  EXPECT_EQ(alloc1->layer_idx(), 1);
  EXPECT_EQ(arena.in_use_count(), 2);
  EXPECT_EQ(arena.available_count(), 0);

  EXPECT_THAT(arena.Acquire(), StatusIs(absl::StatusCode::kResourceExhausted));

  // Explicitly releasing alloc0 returns slot 0 to the arena immediately.
  alloc0->Release();
  EXPECT_FALSE(alloc0->valid());
  EXPECT_EQ(arena.in_use_count(), 1);
  EXPECT_EQ(arena.available_count(), 1);

  absl::StatusOr<LayerArena::Allocation> alloc_reused = arena.Acquire();
  ASSERT_THAT(alloc_reused, IsOk());
  EXPECT_EQ(alloc_reused->layer_idx(), 0);
}

TEST(LayerArenaTest, DynamicCapacityGrowthPreservesPointerStability) {
  size_t allocator_calls = 0;
  HostBufferAllocator default_alloc = MakeDefaultHostBufferAllocator();
  HostBufferAllocator counting_alloc = [&](size_t size_bytes,
                                           const xla::PjRtDevice* device)
      -> absl::StatusOr<HostBufferAllocation> {
    ++allocator_calls;
    return default_alloc(size_bytes, device);
  };

  LayerArena arena(
      LayerArena::Config{
          .initial_capacity = 2,
          .num_shards = 3,
          .shard_host_size = 128,
          .allow_dynamic_growth = true,
      },
      counting_alloc);

  // Initial chunk of 2 layers * 3 shards = 6 allocations.
  EXPECT_EQ(allocator_calls, 6);

  absl::StatusOr<LayerArena::Allocation> alloc0 = arena.Acquire();
  ASSERT_THAT(alloc0, IsOk());
  absl::StatusOr<LayerArena::Allocation> alloc1 = arena.Acquire();
  ASSERT_THAT(alloc1, IsOk());
  LayerInfoBase* ptr0 = alloc0->get();
  LayerInfoBase* ptr1 = alloc1->get();
  EXPECT_EQ(arena.num_chunks(), 1);
  EXPECT_EQ(arena.size(), 2);

  // Acquiring a 3rd slot triggers adding a new std::vector<LayerInfoBase> of
  // the same size (2) and allocating 2 * 3 = 6 new shard buffers.
  absl::StatusOr<LayerArena::Allocation> alloc2 = arena.Acquire();
  ASSERT_THAT(alloc2, IsOk());
  EXPECT_EQ(allocator_calls, 12);
  EXPECT_EQ(arena.num_chunks(), 2);
  EXPECT_EQ(arena.chunk_size(), 2);
  EXPECT_EQ(arena.size(), 4);
  EXPECT_EQ(arena.in_use_count(), 3);
  EXPECT_EQ(arena.available_count(), 1);
  EXPECT_EQ(alloc2->layer_idx(), 2);
  ASSERT_EQ((*alloc2)->shards.size(), 3);
  for (size_t s = 0; s < 3; ++s) {
    EXPECT_NE((*alloc2)->shards[s].host_ptr, nullptr);
    EXPECT_EQ((*alloc2)->shards[s].host_size, 128);
  }

  // Existing allocations and layer references must remain pointer-stable.
  EXPECT_EQ(alloc0->get(), ptr0);
  EXPECT_EQ(alloc1->get(), ptr1);
  EXPECT_EQ(&arena.GetLayer(0), ptr0);
  EXPECT_EQ(&arena[1], ptr1);
  EXPECT_EQ(&arena.GetLayer(2), alloc2->get());

  // Acquire 2 more to trigger a 3rd chunk of size 2 (total capacity = 6).
  absl::StatusOr<LayerArena::Allocation> alloc3 = arena.Acquire();
  ASSERT_THAT(alloc3, IsOk());
  absl::StatusOr<LayerArena::Allocation> alloc4 = arena.Acquire();
  ASSERT_THAT(alloc4, IsOk());
  EXPECT_EQ(allocator_calls, 18);
  EXPECT_EQ(alloc3->layer_idx(), 3);
  EXPECT_EQ(alloc4->layer_idx(), 4);
  EXPECT_EQ(arena.num_chunks(), 3);
  EXPECT_EQ(arena.size(), 6);
  EXPECT_EQ(&arena[0], ptr0);
  EXPECT_EQ(&arena[1], ptr1);
  EXPECT_EQ(&arena[4], alloc4->get());
}

TEST(LayerArenaTest, DynamicGrowthWithoutAllocatorReturnsError) {
  std::vector<LayerInfoBase> initial_layers(1);
  initial_layers[0].shards.resize(1);
  initial_layers[0].shards[0].device_size = 256;
  initial_layers[0].shards[0].host_size = 256;

  LayerArena arena(std::move(initial_layers), /*host_allocator=*/nullptr,
                   LayerArena::Config{.allow_dynamic_growth = true});
  absl::StatusOr<LayerArena::Allocation> alloc0 = arena.Acquire();
  ASSERT_THAT(alloc0, IsOk());

  // Attempting to grow when shard memory is needed but no allocator was passed
  // must fail with kFailedPrecondition.
  EXPECT_THAT(arena.Acquire(), StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LayerArenaTest, ReleaseMemoryOnFreeResetsShardBuffers) {
  LayerArena arena(LayerArena::Config{
      .initial_capacity = 1,
      .num_shards = 1,
      .allow_dynamic_growth = false,
      .release_memory_on_free = true,
  });

  auto backing_memory = std::make_shared<std::vector<uint8_t>>(256, 0x7F);
  {
    absl::StatusOr<LayerArena::Allocation> alloc = arena.Acquire();
    ASSERT_THAT(alloc, IsOk());
    (*alloc)->shards[0].host_ptr = backing_memory->data();
    (*alloc)->shards[0].host_size = backing_memory->size();
    (*alloc)->shards[0].device_size = 512;
    (*alloc)->shards[0].host_owner = backing_memory;
    EXPECT_EQ(backing_memory.use_count(), 2);
  }

  // Destruction of `alloc` should have released `host_owner` and cleared host
  // pointers/sizes while preserving `shards.size()` and `device_size`.
  EXPECT_EQ(backing_memory.use_count(), 1);
  absl::StatusOr<LayerArena::Allocation> reacquired = arena.Acquire();
  ASSERT_THAT(reacquired, IsOk());
  ASSERT_EQ((*reacquired)->shards.size(), 1);
  EXPECT_EQ((*reacquired)->shards[0].host_ptr, nullptr);
  EXPECT_EQ((*reacquired)->shards[0].host_size, 0);
  EXPECT_EQ((*reacquired)->shards[0].host_owner, nullptr);
  EXPECT_EQ((*reacquired)->shards[0].device_size, 512);
}

TEST(LayerArenaTest, MoveAssignmentReleasesPreviousSlot) {
  LayerArena arena(LayerArena::Config{
      .initial_capacity = 2,
      .num_shards = 1,
      .allow_dynamic_growth = false,
  });

  absl::StatusOr<LayerArena::Allocation> alloc0 = arena.Acquire();
  ASSERT_THAT(alloc0, IsOk());
  absl::StatusOr<LayerArena::Allocation> alloc1 = arena.Acquire();
  ASSERT_THAT(alloc1, IsOk());
  EXPECT_EQ(arena.in_use_count(), 2);
  EXPECT_EQ(arena.available_count(), 0);

  // Move-assigning alloc1 into alloc0 must release slot 0 back to the arena.
  *alloc0 = *std::move(alloc1);
  EXPECT_EQ(arena.in_use_count(), 1);
  EXPECT_EQ(arena.available_count(), 1);
  EXPECT_EQ(alloc0->layer_idx(), 1);
}

TEST(LayerArenaTest, ConstructFromMovedVectorAndAccessByLayerIndex) {
  HostBufferAllocator alloc_fn = MakeDefaultHostBufferAllocator();
  std::vector<LayerInfoBase> initial_layers(3);
  for (size_t i = 0; i < initial_layers.size(); ++i) {
    initial_layers[i].shards.resize(2);
    for (size_t s = 0; s < 2; ++s) {
      size_t sz = (i + 1) * 1024;
      initial_layers[i].shards[s].device_size = sz;
      absl::StatusOr<HostBufferAllocation> buf = alloc_fn(sz, nullptr);
      ASSERT_THAT(buf, IsOk());
      initial_layers[i].shards[s].host_ptr = buf->ptr;
      initial_layers[i].shards[s].host_size = buf->size;
      initial_layers[i].shards[s].host_owner = std::move(buf->owner);
    }
  }

  LayerArena arena(std::move(initial_layers), alloc_fn,
                   LayerArena::Config{.allow_dynamic_growth = true});
  EXPECT_EQ(arena.size(), 3);
  EXPECT_EQ(arena.num_chunks(), 1);
  EXPECT_EQ(arena.chunk_size(), 3);
  EXPECT_EQ(arena[0].shards[0].device_size, 1024);
  EXPECT_EQ(arena.GetLayer(1).shards[0].device_size, 2048);
  EXPECT_EQ(arena[2].shards[0].device_size, 3072);

  // Acquiring 4 slots grows by adding another vector of size 3 (and 2 shards),
  // allocating new host memory matching each layer's template size.
  absl::StatusOr<LayerArena::Allocation> a0 = arena.Acquire();
  absl::StatusOr<LayerArena::Allocation> a1 = arena.Acquire();
  absl::StatusOr<LayerArena::Allocation> a2 = arena.Acquire();
  absl::StatusOr<LayerArena::Allocation> a3 = arena.Acquire();
  ASSERT_THAT(a0, IsOk());
  ASSERT_THAT(a1, IsOk());
  ASSERT_THAT(a2, IsOk());
  ASSERT_THAT(a3, IsOk());
  EXPECT_EQ(arena.num_chunks(), 2);
  EXPECT_EQ(arena.size(), 6);
  ASSERT_EQ((*a3)->shards.size(), 2);
  EXPECT_NE((*a3)->shards[0].host_ptr, nullptr);
  EXPECT_NE((*a3)->shards[0].host_ptr, arena[0].shards[0].host_ptr);
  EXPECT_EQ((*a3)->shards[0].host_size, 1024);
  EXPECT_EQ((*a3)->shards[0].device_size, 1024);
}

// ---------------------------------------------------------------------------
// SlotArena / BlockArena
// ---------------------------------------------------------------------------

TEST(SlotArenaTest, FixedSlotsReserveAndLockBlocksUpFront) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/8);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> allocator,
      SlotArena::Create(&blocks, /*num_slots=*/2, /*max_blocks=*/3));
  EXPECT_FALSE(allocator->dynamic_host_staging());
  EXPECT_EQ(allocator->num_slots(), 2);
  EXPECT_EQ(allocator->max_blocks(), 3);
  EXPECT_EQ(allocator->capacity(), 3);
  EXPECT_EQ(allocator->num_free_slots(), 2);
  // 2 slots * 3 blocks are allocated and locked in the host pool.
  EXPECT_EQ(blocks.num_free_blocks(), 2);
  EXPECT_EQ(blocks.num_locked_blocks(), 6);
  EXPECT_EQ(allocator->slot_blocks(0).size(), 3);
  EXPECT_EQ(allocator->slot_blocks(1).size(), 3);

  // Destroying the allocator unlocks (but does not deallocate) the slots.
  allocator.reset();
  EXPECT_EQ(blocks.num_locked_blocks(), 0);
  EXPECT_EQ(blocks.num_allocated_blocks(), 6);
}

TEST(SlotArenaTest, FixedSlotAcquireReleaseAndFifoReuse) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> allocator,
      SlotArena::Create(&blocks, /*num_slots=*/2, /*max_blocks=*/2));

  // num_blocks <= 0 yields an empty allocation without consuming a slot.
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation none, allocator->Acquire(0));
  EXPECT_TRUE(none.empty());
  EXPECT_FALSE(none);
  EXPECT_EQ(allocator->num_free_slots(), 2);

  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation a0, allocator->Acquire(1));
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation a1, allocator->Acquire(2));
  EXPECT_TRUE(a0);
  EXPECT_FALSE(allocator->dynamic_host_staging());
  EXPECT_EQ(a0.slot_idx(), 0);
  EXPECT_EQ(a1.slot_idx(), 1);
  // A partial request still hands out the whole slot.
  EXPECT_EQ(a0.block_ids().size(), 2);
  EXPECT_THAT(a0.block_ids(),
              ::testing::ElementsAreArray(allocator->slot_blocks(0)));
  EXPECT_EQ(allocator->num_free_slots(), 0);
  EXPECT_THAT(allocator->Acquire(1),
              StatusIs(absl::StatusCode::kResourceExhausted));

  // Slots are reused in release order (FIFO).
  a1.Reset();
  EXPECT_TRUE(a1.empty());
  a0.Reset();
  a0.Reset();  // idempotent
  EXPECT_EQ(allocator->num_free_slots(), 2);
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation first, allocator->Acquire(1));
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation second, allocator->Acquire(1));
  EXPECT_EQ(first.slot_idx(), 1);
  EXPECT_EQ(second.slot_idx(), 0);

  // Move semantics transfer ownership; the destination's slot is released.
  StagingAllocation moved = std::move(first);
  EXPECT_TRUE(first.empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved.slot_idx(), 1);
  EXPECT_EQ(allocator->num_free_slots(), 0);
  second = std::move(moved);
  EXPECT_EQ(second.slot_idx(), 1);
  EXPECT_EQ(allocator->num_free_slots(), 1);
}

TEST(SlotArenaTest, RequestExceedingSlotCapacityIsRejected) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> allocator,
      SlotArena::Create(&blocks, /*num_slots=*/1, /*max_blocks=*/2));
  EXPECT_THAT(allocator->Acquire(3),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      allocator->AcquireWithTimeout(
          3, std::chrono::steady_clock::now() + std::chrono::seconds(1)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator->num_free_slots(), 1);
}

TEST(SlotArenaTest, AcquireWithTimeoutExpiresThenWakesOnRelease) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> allocator,
      SlotArena::Create(&blocks, /*num_slots=*/1, /*max_blocks=*/1));
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation held, allocator->Acquire(1));

  EXPECT_THAT(
      allocator->AcquireWithTimeout(
          1, std::chrono::steady_clock::now() + std::chrono::milliseconds(20)),
      StatusIs(absl::StatusCode::kResourceExhausted));

  absl::Notification waiting;
  absl::StatusOr<StagingAllocation> acquired;
  std::thread waiter([&] {
    waiting.Notify();
    acquired = allocator->AcquireWithTimeout(
        1, std::chrono::steady_clock::now() + std::chrono::seconds(10));
  });
  waiting.WaitForNotification();
  absl::SleepFor(absl::Milliseconds(20));
  held.Reset();
  waiter.join();
  ABSL_ASSERT_OK(acquired);
  EXPECT_EQ(acquired->slot_idx(), 0);
}

TEST(SlotArenaTest, ShutdownWakesWaitersAndCancelsTimedAcquires) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> allocator,
      SlotArena::Create(&blocks, /*num_slots=*/1, /*max_blocks=*/1));
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation held, allocator->Acquire(1));

  absl::Notification waiting;
  absl::StatusOr<StagingAllocation> acquired;
  std::thread waiter([&] {
    waiting.Notify();
    acquired = allocator->AcquireWithTimeout(
        1, std::chrono::steady_clock::now() + std::chrono::seconds(10));
  });
  waiting.WaitForNotification();
  absl::SleepFor(absl::Milliseconds(20));
  allocator->Shutdown();
  waiter.join();
  EXPECT_THAT(acquired, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(
      allocator->AcquireWithTimeout(
          1, std::chrono::steady_clock::now() + std::chrono::seconds(1)),
      StatusIs(absl::StatusCode::kCancelled));

  // Held slots can still be returned after shutdown.
  held.Reset();
  EXPECT_EQ(allocator->num_free_slots(), 1);
}

TEST(BlockArenaTest, LocksAndReleasesHostBlocksPerRequest) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/4);
  BlockArena arena(&blocks, /*num_slots=*/2, /*max_blocks=*/2);
  EXPECT_TRUE(arena.dynamic_host_staging());
  // No slot pool is reserved; capacity is the whole host pool.
  EXPECT_EQ(arena.size(), 0);
  EXPECT_EQ(arena.capacity(), 4);
  EXPECT_EQ(blocks.num_allocated_blocks(), 0);

  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation a, arena.Acquire(3));
  EXPECT_TRUE(a);
  EXPECT_EQ(a.slot_idx(), 0);
  EXPECT_EQ(a.block_ids().size(), 3);
  EXPECT_EQ(blocks.num_locked_blocks(), 3);
  for (int block_id : a.block_ids()) {
    EXPECT_TRUE(blocks.IsLocked(block_id));
  }
  EXPECT_EQ(arena.size(), 1);
  EXPECT_EQ(arena.in_use_count(), 1);

  // Only one unlocked block remains, so a 2-block request is exhausted and
  // the on-demand slot it created stays free for reuse.
  EXPECT_THAT(arena.Acquire(2), StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(arena.available_count(), 1);
  EXPECT_THAT(arena.Acquire(5), StatusIs(absl::StatusCode::kInvalidArgument));

  // Release unlocks and deallocates the blocks.
  a.Reset();
  EXPECT_EQ(blocks.num_locked_blocks(), 0);
  EXPECT_EQ(blocks.num_allocated_blocks(), 0);
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation b, arena.Acquire(4));
  EXPECT_EQ(b.block_ids().size(), 4);
  // Slots are recycled rather than appended.
  EXPECT_EQ(arena.size(), 2);
}

TEST(BlockArenaTest, AcquireWithTimeoutWaitsForBlocksToBeReleased) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/2);
  BlockArena arena(&blocks, /*num_slots=*/1, /*max_blocks=*/2);
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation held, arena.Acquire(2));
  EXPECT_THAT(arena.AcquireWithTimeout(1, std::chrono::steady_clock::now() +
                                              std::chrono::milliseconds(20)),
              StatusIs(absl::StatusCode::kResourceExhausted));

  absl::Notification waiting;
  absl::StatusOr<StagingAllocation> acquired;
  std::thread waiter([&] {
    waiting.Notify();
    acquired = arena.AcquireWithTimeout(
        2, std::chrono::steady_clock::now() + std::chrono::seconds(10));
  });
  waiting.WaitForNotification();
  absl::SleepFor(absl::Milliseconds(20));
  held.Reset();
  waiter.join();
  ABSL_ASSERT_OK(acquired);
  EXPECT_EQ(acquired->block_ids().size(), 2);
}

TEST(SlotArenaTest, CreateFailsWithoutEnoughHostBlocks) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/3);
  EXPECT_THAT(SlotArena::Create(&blocks, /*num_slots=*/2, /*max_blocks=*/2),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // Nothing was reserved.
  EXPECT_EQ(blocks.num_allocated_blocks(), 0);
}

TEST(SlotArenaTest, NullHostBlockManager) {
  // Without a slot pool a null manager is accepted: this is the
  // "no shards" configuration where staging is never used.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> arena,
      SlotArena::Create(/*host_block_manager=*/nullptr, /*num_slots=*/0,
                        /*max_blocks=*/2));
  EXPECT_EQ(arena->num_free_slots(), 0);
  EXPECT_EQ(arena->capacity(), 2);
  TF_ASSERT_OK_AND_ASSIGN(StagingAllocation none, arena->Acquire(0));
  EXPECT_TRUE(none.empty());
  EXPECT_THAT(arena->Acquire(1),
              StatusIs(absl::StatusCode::kResourceExhausted));

  // Requesting a slot pool over a null manager fails at creation.
  EXPECT_THAT(SlotArena::Create(/*host_block_manager=*/nullptr,
                                /*num_slots=*/1, /*max_blocks=*/1),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(StagingArenaTest, ArenasShareTheBaseContract) {
  kv_cache::LogicalBlockManager blocks(/*num_blocks=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SlotArena> slot_arena,
      SlotArena::Create(&blocks, /*num_slots=*/1, /*max_blocks=*/1));
  BlockArena block_arena(&blocks, /*num_slots=*/1, /*max_blocks=*/1);
  LayerArena layer_arena(
      LayerArena::Config{.initial_capacity = 1, .num_shards = 1});

  for (StagingArena* arena : {static_cast<StagingArena*>(slot_arena.get()),
                              static_cast<StagingArena*>(&block_arena),
                              static_cast<StagingArena*>(&layer_arena)}) {
    EXPECT_EQ(arena->in_use_count(), 0);
    TF_ASSERT_OK_AND_ASSIGN(StagingArena::Allocation a, arena->Acquire(1));
    EXPECT_TRUE(a.valid());
    EXPECT_TRUE(arena->is_in_use(a.index()));
    EXPECT_EQ(arena->in_use_count(), 1);
    a.Release();
    EXPECT_EQ(arena->in_use_count(), 0);
    arena->Shutdown();
    EXPECT_THAT(arena->Acquire(1), StatusIs(absl::StatusCode::kCancelled));
  }
}

}  // namespace
}  // namespace tpu_raiden

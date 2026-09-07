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

#include "tpu_sync/kv_cache/logical_block_manager.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;

TEST(LogicalBlockManagerTest, InitialState) {
  LogicalBlockManager manager(5);
  EXPECT_EQ(manager.total_blocks(), 5);
  EXPECT_EQ(manager.num_free_blocks(), 5);
  EXPECT_EQ(manager.num_allocated_blocks(), 0);
  EXPECT_EQ(manager.num_locked_blocks(), 0);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(manager.IsAllocated(i));
    EXPECT_FALSE(manager.IsLocked(i));
  }
}

TEST(LogicalBlockManagerTest, BasicAllocation) {
  LogicalBlockManager manager(5);
  TF_ASSERT_OK_AND_ASSIGN(auto blocks, manager.Allocate(3, /*lock=*/false));
  EXPECT_THAT(blocks, ElementsAre(0, 1, 2));

  EXPECT_EQ(manager.num_free_blocks(), 2);
  EXPECT_EQ(manager.num_allocated_blocks(), 3);
  EXPECT_EQ(manager.num_locked_blocks(), 0);

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(manager.IsAllocated(i));
    EXPECT_FALSE(manager.IsLocked(i));
  }
}

TEST(LogicalBlockManagerTest, AllocationWithLocking) {
  LogicalBlockManager manager(5);
  TF_ASSERT_OK_AND_ASSIGN(auto blocks, manager.Allocate(2, /*lock=*/true));
  EXPECT_THAT(blocks, ElementsAre(0, 1));

  EXPECT_EQ(manager.num_locked_blocks(), 2);
  EXPECT_TRUE(manager.IsLocked(0));
  EXPECT_TRUE(manager.IsLocked(1));
}

TEST(LogicalBlockManagerTest, LruEvictionOrder) {
  LogicalBlockManager manager(4);

  // Allocate 2 blocks to entity 10 (unlocked).
  TF_ASSERT_OK_AND_ASSIGN(auto blocks1, manager.Allocate(2));
  EXPECT_THAT(blocks1, ElementsAre(0, 1));

  // Allocate 2 blocks to entity 20 (unlocked).
  TF_ASSERT_OK_AND_ASSIGN(auto blocks2, manager.Allocate(2));
  EXPECT_THAT(blocks2, ElementsAre(2, 3));

  EXPECT_EQ(manager.num_free_blocks(), 0);

  // Requesting 2 blocks for entity 30 should evict entity 10's blocks
  // because they were allocated earlier (LRU).
  TF_ASSERT_OK_AND_ASSIGN(auto blocks3, manager.Allocate(2));
  EXPECT_THAT(blocks3, ElementsAre(0, 1));
}

TEST(LogicalBlockManagerTest, AccessUpdatesLruOrder) {
  LogicalBlockManager manager(4);

  TF_ASSERT_OK_AND_ASSIGN(auto blocks1, manager.Allocate(2));
  TF_ASSERT_OK_AND_ASSIGN(auto blocks2, manager.Allocate(2));

  // Access entity 10's blocks, making entity 20's blocks the least recently
  // used.
  ABSL_EXPECT_OK(manager.AccessBlock(0));
  ABSL_EXPECT_OK(manager.AccessBlock(1));

  // Allocate 2 blocks for entity 30. Should evict entity 20's blocks (2 and 3).
  TF_ASSERT_OK_AND_ASSIGN(auto blocks3, manager.Allocate(2));
  EXPECT_THAT(blocks3, ElementsAre(2, 3));
}

TEST(LogicalBlockManagerTest, LockedBlocksPreventEviction) {
  LogicalBlockManager manager(4);

  // Allocate 2 locked blocks to entity 10.
  ABSL_ASSERT_OK(manager.Allocate(2, /*lock=*/true));
  // Allocate 2 unlocked blocks.
  ABSL_ASSERT_OK(manager.Allocate(2, /*lock=*/false));

  // Requesting 3 blocks should fail since only 2 blocks are evictable.
  EXPECT_THAT(manager.Allocate(3),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(LogicalBlockManagerTest, UnlockAllowsEviction) {
  LogicalBlockManager manager(4);

  ABSL_ASSERT_OK(manager.Allocate(2, /*lock=*/true));
  ABSL_ASSERT_OK(manager.Allocate(2, /*lock=*/false));

  // Unlock entity 10's blocks.
  std::vector<int> to_unlock = {0, 1};
  ABSL_EXPECT_OK(manager.Unlock(to_unlock));
  EXPECT_EQ(manager.num_locked_blocks(), 0);

  // Now requesting 3 blocks succeeds.
  TF_ASSERT_OK_AND_ASSIGN(auto blocks, manager.Allocate(3));
  EXPECT_EQ(blocks.size(), 3);
}

TEST(LogicalBlockManagerTest, InvalidArguments) {
  LogicalBlockManager manager(5);

  EXPECT_FALSE(manager.Allocate(0).ok());
  EXPECT_FALSE(manager.Allocate(-1).ok());
  EXPECT_FALSE(manager.Allocate(6).ok());

  EXPECT_FALSE(manager.AccessBlock(5).ok());
  EXPECT_FALSE(manager.AccessBlock(-1).ok());
  // Accessing unallocated block
  EXPECT_FALSE(manager.AccessBlock(0).ok());

  std::vector<int> invalid_unlock = {0};
  EXPECT_FALSE(manager.Unlock(invalid_unlock).ok());
}

TEST(LogicalBlockManagerTest, AllocateTargetMarksBlocksAllocatedAndLocked) {
  LogicalBlockManager manager(5);
  ABSL_ASSERT_OK(manager.AllocateTarget({1, 3}));
  EXPECT_TRUE(manager.IsAllocated(1));
  EXPECT_TRUE(manager.IsLocked(1));
  EXPECT_TRUE(manager.IsAllocated(3));
  EXPECT_TRUE(manager.IsLocked(3));
  EXPECT_EQ(manager.num_free_blocks(), 3);

  // Target-allocated blocks are never handed out by subsequent allocations.
  TF_ASSERT_OK_AND_ASSIGN(auto blocks, manager.Allocate(3, /*lock=*/true));
  EXPECT_THAT(blocks, ElementsAre(0, 2, 4));
  // Everything is locked now: further allocation must fail.
  EXPECT_FALSE(manager.Allocate(1).ok());
}

TEST(LogicalBlockManagerTest, AllocateTargetValidatesAtomically) {
  LogicalBlockManager manager(5);
  ABSL_ASSERT_OK(manager.Allocate(1));  // Block 0 becomes allocated.

  // Out of range.
  EXPECT_THAT(manager.AllocateTarget({1, 5}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Already allocated (even though unlocked).
  EXPECT_THAT(manager.AllocateTarget({1, 0}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // Duplicate ID within the batch.
  EXPECT_THAT(manager.AllocateTarget({2, 2}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Failed calls must not have modified any state.
  EXPECT_FALSE(manager.IsAllocated(1));
  EXPECT_FALSE(manager.IsAllocated(2));
  EXPECT_EQ(manager.num_allocated_blocks(), 1);
}

TEST(LogicalBlockManagerTest, DeallocateReturnsBlocksToFreePool) {
  LogicalBlockManager manager(4);
  ABSL_ASSERT_OK(manager.Allocate(2, /*lock=*/true));
  ABSL_ASSERT_OK(manager.Allocate(1, /*lock=*/false));
  EXPECT_EQ(manager.num_free_blocks(), 1);

  // Deallocation works on locked and unlocked blocks alike.
  ABSL_ASSERT_OK(manager.Deallocate({0, 2}));
  EXPECT_EQ(manager.num_free_blocks(), 3);
  EXPECT_EQ(manager.num_allocated_blocks(), 1);
  EXPECT_EQ(manager.num_locked_blocks(), 1);
  EXPECT_FALSE(manager.IsAllocated(0));
  EXPECT_FALSE(manager.IsLocked(0));
  EXPECT_FALSE(manager.IsAllocated(2));

  // Deallocated blocks are free again for both allocation paths.
  ABSL_ASSERT_OK(manager.AllocateTarget({0}));
  TF_ASSERT_OK_AND_ASSIGN(auto blocks, manager.Allocate(1));
  EXPECT_THAT(blocks, ElementsAre(2));
}

TEST(LogicalBlockManagerTest, DeallocateValidatesAtomically) {
  LogicalBlockManager manager(3);
  ABSL_ASSERT_OK(manager.Allocate(1, /*lock=*/true));  // Block 0.

  // Out of range.
  EXPECT_THAT(manager.Deallocate({0, 3}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Not allocated.
  EXPECT_THAT(manager.Deallocate({0, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Failed calls must not have modified any state.
  EXPECT_TRUE(manager.IsAllocated(0));
  EXPECT_TRUE(manager.IsLocked(0));

  ABSL_ASSERT_OK(manager.Deallocate({0}));
  // Double deallocation fails.
  EXPECT_THAT(manager.Deallocate({0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LogicalBlockManagerTest, TargetAllocatedBlocksReusableAfterUnlock) {
  LogicalBlockManager manager(3);
  ABSL_ASSERT_OK(manager.AllocateTarget({0, 1, 2}));
  ABSL_ASSERT_OK(manager.Unlock({1}));

  // The unlocked target-allocated block is evictable and gets reused.
  TF_ASSERT_OK_AND_ASSIGN(auto blocks, manager.Allocate(1));
  EXPECT_THAT(blocks, ElementsAre(1));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

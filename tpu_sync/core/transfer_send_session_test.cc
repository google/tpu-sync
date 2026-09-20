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

#include "tpu_sync/core/transfer_send_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;

kv_cache::KVCacheManagerBase MakeTestBase(size_t num_layers = 1) {
  return kv_cache::KVCacheManagerBase(
      num_layers, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
}

class FakeSendBase : public kv_cache::KVCacheManagerBase {
 public:
  explicit FakeSendBase(size_t num_layers = 1, int64_t host_blocks = 4)
      : kv_cache::KVCacheManagerBase(num_layers, /*num_shards=*/1,
                                     std::vector<size_t>(num_layers, 128),
                                     /*local_port=*/std::nullopt,
                                     /*host_blocks_to_allocate=*/host_blocks,
                                     /*parallelism=*/1, nullptr) {}

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    auto [promise, future] = xla::MakePromise<>();
    promise.Set(absl::OkStatus());
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

  void H2hWriteDirectAsync(const std::vector<std::string>& peers,
                           const std::vector<int>& src_block_ids,
                           const std::vector<int>& dst_block_ids, uint64_t uuid,
                           int layer_idx,
                           std::function<void(absl::StatusOr<std::vector<int>>)>
                               on_complete) override {
    on_complete(src_block_ids);
  }
};

TEST(TransferSendSessionTest, SendSessionImplementsTransferSessionInterface) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  auto now = std::chrono::steady_clock::now();
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req", /*uuid=*/30, /*block_ids=*/{},
      /*deadline=*/now + std::chrono::seconds(5),
      /*register_start=*/now, /*in_flight=*/1);
  TransferSession* base_session = session.get();

  EXPECT_FALSE(base_session->Done());
  EXPECT_FALSE(base_session->IsDraining());
  ABSL_EXPECT_OK(base_session->GetStatus());

  base_session->Finish(absl::UnavailableError("peer disconnected"));
  EXPECT_TRUE(base_session->IsDraining());
  EXPECT_FALSE(base_session->Done());
  EXPECT_THAT(base_session->GetStatus(),
              StatusIs(absl::StatusCode::kUnavailable));

  session->EndSendOp();
  EXPECT_TRUE(base_session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_THAT(base_session->AwaitForDone(),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST(TransferSendSessionTest,
     FinishingOneSendSessionDoesNotShutdownSharedStagingAllocator) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  // Session 1 completes a full send push and retires.
  std::shared_ptr<TransferSendSession> session1 = *TransferSendSession::Create(
      &base, allocator.get(), "req1", /*uuid=*/101, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(5), /*register_start=*/now);
  session1->ValidateAndBeginPull({0}, now);
  session1->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                      /*dst_block_ids=*/{0});
  ABSL_EXPECT_OK(session1->AwaitForDone());
  EXPECT_TRUE(session1->Done());
  EXPECT_FALSE(session1->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 2);

  // Session 2 shares the same StagingBlockAllocator and must still be able to
  // acquire staging and complete a push without CancelledError.
  std::shared_ptr<TransferSendSession> session2 = *TransferSendSession::Create(
      &base, allocator.get(), "req2", /*uuid=*/102, /*block_ids=*/{1},
      /*deadline=*/now + std::chrono::seconds(5), /*register_start=*/now);
  session2->ValidateAndBeginPull({1}, now);
  session2->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{1},
                      /*dst_block_ids=*/{0});
  ABSL_EXPECT_OK(session2->AwaitForDone());
  EXPECT_TRUE(session2->Done());
  EXPECT_FALSE(session2->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 2);
}

TEST(TransferSendSessionTest,
     CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/2);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/1, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  // Exhaust the only staging slot.
  absl::StatusOr<StagingAllocation> held_slot = allocator->Acquire(1);
  ABSL_ASSERT_OK(held_slot);
  ASSERT_EQ(allocator->num_free_slots(), 0);

  // Create a session with a 60-second deadline and block it in StartPush.
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_blocked", /*uuid=*/104, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(60), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);

  auto push_future = std::async(std::launch::async, [session]() {
    session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                       /*dst_block_ids=*/{0});
  });

  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_FALSE(session->Done());

  // Cancelling the session must wake AcquireWithTimeout promptly without
  // waiting for the 60-second deadline and without poisoning the allocator.
  const absl::Time cancel_start = absl::Now();
  session->Finish(absl::CancelledError("peer aborted"));
  ASSERT_EQ(push_future.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_LT(absl::Now() - cancel_start, absl::Seconds(1));

  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());

  // Returning the held slot leaves the shared allocator healthy.
  held_slot->Reset();
  EXPECT_EQ(allocator->num_free_slots(), 1);
  absl::StatusOr<StagingAllocation> reacquired =
      allocator->AcquireWithTimeout(1, now + std::chrono::seconds(1));
  ABSL_EXPECT_OK(reacquired);
}

}  // namespace
}  // namespace tpu_raiden

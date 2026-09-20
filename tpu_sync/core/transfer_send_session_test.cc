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
#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/transfer_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

kv_cache::KVCacheManagerBase MakeTestBase() {
  return kv_cache::KVCacheManagerBase(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
}

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
  EXPECT_TRUE(base_session->GetStatus().ok());

  base_session->Finish(absl::UnavailableError("peer disconnected"));
  EXPECT_TRUE(base_session->IsDraining());
  EXPECT_FALSE(base_session->Done());
  EXPECT_EQ(base_session->GetStatus().code(), absl::StatusCode::kUnavailable);

  session->EndSendOp();
  EXPECT_TRUE(base_session->Done());
  EXPECT_EQ(base_session->AwaitForDone().code(),
            absl::StatusCode::kUnavailable);
}

}  // namespace
}  // namespace tpu_raiden

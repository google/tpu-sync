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

#include "tpu_sync/transport/block_transport_session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {
namespace transport {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Eq;

class MockSessionDelegate : public BlockTransportDelegate {
 public:
  MockSessionDelegate(size_t slice_size = 1024, int max_blocks = 4,
                      size_t num_layers = 1, size_t num_shards = 1)
      : slice_size_(slice_size),
        max_blocks_(max_blocks),
        num_layers_(num_layers),
        num_shards_(num_shards) {
    buffers_.resize(num_layers_ * num_shards_);
    for (auto& buffer : buffers_) {
      buffer.resize(slice_size_ * max_blocks_, 0);
    }
  }

  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    std::vector<int> ids;
    ids.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
      ids.push_back(static_cast<int>(i % max_blocks_));
    }
    return ids;
  }

  absl::Status OnDataReceived(uint64_t uuid = 0) override {
    return absl::OkStatus();
  }

  absl::Status OnLayerReceived(size_t layer_idx, uint64_t uuid) override {
    ++layers_received_;
    return absl::OkStatus();
  }

  absl::StatusOr<std::optional<PoolPushProgressSpec>> GetPoolPushProgressSpec(
      size_t pool_idx, uint64_t uuid) const override {
    return std::nullopt;
  }

  bool AcceptsPlanlessExplicitPush(uint64_t uuid) const override {
    return true;
  }

  size_t num_layers() const override { return num_layers_; }
  size_t num_block_arrays() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_size_; }
  size_t block_bytes(size_t layer_idx = 0) const override {
    return slice_size_;
  }
  size_t shard_factor() const override { return 1; }
  int64_t node_id() const override { return 0; }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override {
    return buffers_[layer_idx * num_shards_ + shard_idx].data();
  }

  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override {
    return buffers_[layer_idx * num_shards_ + shard_idx].size();
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  absl::StatusOr<std::shared_ptr<TransferSession>> GetSession(
      uint64_t uuid) const override {
    return absl::NotFoundError("Session not found");
  }

  uint8_t* GetBlockArrayHostPointer(size_t layer, size_t shard) override {
    return buffers_[layer * num_shards_ + shard].data();
  }

  size_t GetBlockArrayHostSize(size_t layer, size_t shard) override {
    return buffers_[layer * num_shards_ + shard].size();
  }

  std::vector<BlockChunk> GetBlockChunks(
      size_t layer, size_t shard, absl::Span<const int64_t> block_ids,
      size_t expected_size = 0, uint64_t uuid = 0, int64_t remote_node_id = -1,
      absl::string_view peer = "", int64_t src_block_id = -1,
      int64_t dst_block_id = -1) override {
    std::vector<BlockChunk> chunks;
    chunks.reserve(block_ids.size());
    for (int64_t b : block_ids) {
      uint8_t* ptr = buffers_[layer * num_shards_ + shard].data() +
                     (b % max_blocks_) * slice_size_;
      chunks.push_back(BlockChunk{
          .ptr = ptr,
          .size = static_cast<uint32_t>(slice_size_),
          .raddr = nullptr,
      });
    }
    return chunks;
  }

  int layers_received() const { return layers_received_; }

 private:
  const size_t slice_size_;
  const int max_blocks_;
  const size_t num_layers_;
  const size_t num_shards_;
  std::vector<std::vector<uint8_t>> buffers_;
  int layers_received_ = 0;
};

TEST(BlockTransportServerPushSessionTest, LifecycleAndPostCompletion) {
  MockSessionDelegate delegate(/*slice_size=*/1024, /*max_blocks=*/4,
                               /*num_layers=*/2, /*num_shards=*/1);
  BlockTransportConfig config;

  auto unregistered_session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/42, &delegate, config);
  EXPECT_THAT(unregistered_session->uuid(), Eq(42u));
  EXPECT_THAT(
      unregistered_session->OnPeregrinePostComplete(
          lib::PostMetadata{
              .uuid = 42, .remote_id = 0, .local_id = 0, .shard_mask = 0},
          /*requests=*/{}),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_FALSE(unregistered_session->IsDone());
  EXPECT_THAT(delegate.layers_received(), Eq(0));

  auto session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/0, &delegate, config);

  EXPECT_THAT(session->uuid(), Eq(0u));
  EXPECT_THAT(session->kind(), Eq(BlockTransportServerSession::Kind::kPush));
  EXPECT_FALSE(session->IsDone());
  EXPECT_THAT(session->status(), IsOk());

  // First layer arrives via Peregrine post:
  lib::PostMetadata post0{
      .uuid = 0, .remote_id = 0, .local_id = 0, .shard_mask = 0};
  ABSL_EXPECT_OK(session->OnPeregrinePostComplete(post0, /*requests=*/{}));
  EXPECT_FALSE(session->IsDone());

  // Second layer arrives:
  lib::PostMetadata post1{
      .uuid = 0, .remote_id = 0, .local_id = 1, .shard_mask = 0};
  ABSL_EXPECT_OK(session->OnPeregrinePostComplete(post1, /*requests=*/{}));
  EXPECT_TRUE(session->IsDone());
  EXPECT_THAT(session->status(), IsOk());
  EXPECT_THAT(delegate.layers_received(), Eq(2));
}

TEST(BlockTransportServerPushSessionTest, Cancel) {
  MockSessionDelegate delegate;
  BlockTransportConfig config;

  auto session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/100, &delegate, config);

  EXPECT_FALSE(session->IsDone());
  session->Cancel();
  EXPECT_TRUE(session->IsDone());
  EXPECT_THAT(session->status(), StatusIs(absl::StatusCode::kCancelled));
}

TEST(BlockTransportServerPullSessionTest, InvalidArgumentsRejected) {
  MockSessionDelegate delegate;
  BlockTransportConfig config;
  lib::ChunkHeader header{};
  header.uuid = 77;
  header.count_or_size = 5;

  auto session = std::make_shared<BlockTransportServerPullSession>(
      /*uuid=*/77, &delegate, config,
      /*local_ips=*/std::vector<std::string>{"127.0.0.1"});

  EXPECT_THAT(session->uuid(), Eq(77u));
  EXPECT_THAT(session->kind(), Eq(BlockTransportServerSession::Kind::kPull));

  absl::Status start_status = session->StartStream(/*client_fd=*/-1, header);
  EXPECT_FALSE(start_status.ok());
}

TEST(BlockTransportServerPullSessionTest, UuidMismatchRejected) {
  MockSessionDelegate delegate;
  BlockTransportConfig config;
  lib::ChunkHeader header{};
  header.uuid = 99;

  auto session = std::make_shared<BlockTransportServerPullSession>(
      /*uuid=*/77, &delegate, config,
      /*local_ips=*/std::vector<std::string>{"127.0.0.1"});

  EXPECT_THAT(session->StartStream(/*client_fd=*/-1, header),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BlockTransportServerPullSessionTest, Cancel) {
  MockSessionDelegate delegate;
  BlockTransportConfig config;
  bool completed_called = false;

  auto session = std::make_shared<BlockTransportServerPullSession>(
      /*uuid=*/88, &delegate, config,
      /*local_ips=*/std::vector<std::string>{"127.0.0.1"},
      [&](uint64_t uuid) { completed_called = true; });

  EXPECT_FALSE(session->IsDone());
  session->Cancel();
  EXPECT_TRUE(session->IsDone());
  EXPECT_THAT(session->status(), StatusIs(absl::StatusCode::kCancelled));
  EXPECT_TRUE(completed_called);
}

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden

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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {
namespace transport {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

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
    return accepts_planless_explicit_push_;
  }
  void set_accepts_planless_explicit_push(bool accepts) {
    accepts_planless_explicit_push_ = accepts;
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

  using BlockTransportDelegate::GetHostBuffer;

  absl::StatusOr<absl::Span<const uint8_t>> GetHostBuffer(
      size_t layer_idx, size_t shard_idx) const override {
    if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
      return absl::OutOfRangeError(absl::StrCat(
          "layer_idx ", layer_idx, " shard_idx ", shard_idx,
          " out of range: ", num_layers_, " layers, ", num_shards_, " shards"));
    }
    const std::vector<uint8_t>& buf =
        buffers_[layer_idx * num_shards_ + shard_idx];
    if (buf.data() == nullptr) {
      return absl::FailedPreconditionError(absl::StrCat(
          "layer ", layer_idx, " shard ", shard_idx, " has no host buffer"));
    }
    return absl::Span<const uint8_t>(buf);
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  absl::StatusOr<std::shared_ptr<TransferSession>> GetSession(
      uint64_t uuid) const override {
    if (registered_session_ != nullptr && uuid == registered_uuid_) {
      return registered_session_;
    }
    return absl::NotFoundError("Session not found");
  }

  void RegisterSession(uint64_t uuid,
                       std::shared_ptr<TransferSession> session) {
    registered_uuid_ = uuid;
    registered_session_ = std::move(session);
  }

  absl::Status OnBlockShardsReceived(const std::vector<int>& block_ids,
                                     absl::Span<const int> shard_ids,
                                     uint64_t uuid) override {
    received_block_shards_.push_back(
        {block_ids, std::vector<int>(shard_ids.begin(), shard_ids.end()),
         uuid});
    return absl::OkStatus();
  }

  std::vector<BlockChunk> GetBlockChunks(
      size_t layer, size_t shard, absl::Span<const int64_t> block_ids,
      size_t expected_size = 0, uint64_t uuid = 0, int64_t remote_node_id = -1,
      absl::string_view peer = "", int64_t src_block_id = -1,
      int64_t dst_block_id = -1) override {
    last_src_block_id_ = src_block_id;
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

  struct BlockShardsReport {
    std::vector<int> block_ids;
    std::vector<int> shard_ids;
    uint64_t uuid = 0;
  };
  int layers_received() const { return layers_received_; }
  int64_t last_src_block_id() const { return last_src_block_id_; }
  const std::vector<BlockShardsReport>& received_block_shards() const {
    return received_block_shards_;
  }

 private:
  const size_t slice_size_;
  const int max_blocks_;
  const size_t num_layers_;
  const size_t num_shards_;
  std::vector<std::vector<uint8_t>> buffers_;
  int layers_received_ = 0;
  int64_t last_src_block_id_ = -1;
  bool accepts_planless_explicit_push_ = true;
  uint64_t registered_uuid_ = 0;
  std::shared_ptr<TransferSession> registered_session_;
  std::vector<BlockShardsReport> received_block_shards_;
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
      /*uuid=*/0, &delegate, config);

  EXPECT_FALSE(session->IsDone());
  session->Cancel();
  EXPECT_TRUE(session->IsDone());
  EXPECT_THAT(session->status(), StatusIs(absl::StatusCode::kCancelled));

  const lib::PostMetadata post{.uuid = 0, .remote_id = 0, .local_id = 0};
  EXPECT_THAT(
      session->ResolveRecvBuffer(
          post, {.shard = 0, .block_id = 0, .chunk_idx = 0, .len = 1024}),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(session->OnPeregrinePostComplete(post, /*requests=*/{}),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(delegate.layers_received(), Eq(0));
}

TEST(BlockTransportServerPushSessionTest,
     ResolveRecvBufferCancelledDuringAdmissionWithUuidZero) {
  struct CancellingDelegate : public MockSessionDelegate {
    BlockTransportServerPushSession* session = nullptr;
    absl::StatusOr<std::optional<PoolPushProgressSpec>> GetPoolPushProgressSpec(
        size_t pool_idx, uint64_t uuid) const override {
      if (session != nullptr) {
        session->Cancel();
      }
      return std::nullopt;
    }
  };
  CancellingDelegate delegate;
  BlockTransportConfig config;
  auto session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/0, &delegate, config);
  delegate.session = session.get();

  const lib::PostMetadata post{.uuid = 0, .remote_id = 0, .local_id = 0};
  EXPECT_THAT(
      session->ResolveRecvBuffer(
          post, {.shard = 0, .block_id = 0, .chunk_idx = 0, .len = 1024}),
      StatusIs(absl::StatusCode::kCancelled));
  EXPECT_FALSE(session->HasAdmittedPosts());
}

TEST(BlockTransportServerPushSessionTest,
     PeregrineBlockMajorResolveAndComplete) {
  constexpr size_t kSliceSize = 1024;
  constexpr size_t kMaxBlocks = 5;
  constexpr size_t kNumLayers = 2;
  constexpr size_t kNumShards = 1;
  MockSessionDelegate delegate(kSliceSize, kMaxBlocks, kNumLayers, kNumShards);
  BlockTransportConfig config;
  config.major_order = MajorOrder::kBlockMajor;

  auto session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/0, &delegate, config);
  const lib::PostMetadata post{
      .uuid = 0,
      .remote_id = 7,
      .local_id = 3,  // dst_block_id >= kNumLayers
      .shard_mask = 0,
      .op = lib::TransferOp::kBlockPushExplicit,
  };
  const lib::RequestMetadata req{
      .shard = 0,
      .block_id = 1,      // layer_id
      .src_block_id = 2,  // sender's source block_id
      .chunk_idx = 0,
      .len = kSliceSize,
  };

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host, delegate.GetHostBuffer(1, 0));
  EXPECT_THAT(session->ResolveRecvBuffer(post, req),
              IsOkAndHolds(absl::MakeSpan(host.data() + 3 * kSliceSize,
                                          kSliceSize)));
  EXPECT_THAT(delegate.last_src_block_id(), Eq(2));
  ABSL_EXPECT_OK(session->OnPeregrinePostComplete(post, {}, absl::OkStatus()));
  // Without a receive plan, completing one block does not retire the session,
  // so subsequent blocks on the same uuid can still be admitted and completed.
  EXPECT_FALSE(session->IsDone());
  EXPECT_THAT(delegate.layers_received(), Eq(0));

  lib::PostMetadata post2 = post;
  post2.local_id = 4;
  EXPECT_THAT(session->ResolveRecvBuffer(post2, req),
              IsOkAndHolds(absl::MakeSpan(host.data() + 4 * kSliceSize,
                                          kSliceSize)));
  ABSL_EXPECT_OK(session->OnPeregrinePostComplete(post2, {}, absl::OkStatus()));
  EXPECT_FALSE(session->IsDone());

  // Out-of-range layer_id is rejected under kBlockMajor.
  auto bad_session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/0, &delegate, config);
  lib::RequestMetadata bad_req = req;
  bad_req.block_id = kNumLayers;
  EXPECT_THAT(bad_session->ResolveRecvBuffer(post, bad_req),
              StatusIs(absl::StatusCode::kOutOfRange));

  // Pool-mode workers that reject planless explicit pushes reject kBlockMajor.
  delegate.set_accepts_planless_explicit_push(false);
  auto pool_mode_session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/0, &delegate, config);
  EXPECT_THAT(pool_mode_session->ResolveRecvBuffer(post, req),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BlockTransportServerPushSessionTest,
     PeregrineBlockMajorWithSessionReportsBlockShardsOnly) {
  class StubTransferSession final : public TransferSession {
   public:
    absl::Status Start() override { return absl::OkStatus(); }
    bool Done() const override { return false; }
    void Finish(const absl::Status& status) override { status_ = status; }
    bool TryBeginOp() override { return true; }
    void EndOp() override {}
    absl::Status GetStatus() const override { return status_; }
    absl::Status AwaitForDone() override { return status_; }
    bool IsDraining() const override { return false; }
    absl::StatusOr<size_t> ResolveStagingSlot(size_t) override {
      return absl::NotFoundError("no staging");
    }
    absl::Status OnDataReceived(size_t, absl::Span<const int>,
                                absl::Span<const int>) override {
      ++unit_completed_calls;
      return absl::FailedPreconditionError(
          "OnDataReceived must not be called with dst_block_id as unit");
    }
    tsl::Future<> GetDataReadyFuture(size_t) override {
      return tsl::Future<>(absl::OkStatus());
    }
    absl::optional<std::shared_ptr<const TransferPlan>> plan() const override {
      return absl::nullopt;
    }
    std::chrono::steady_clock::time_point deadline() const override {
      return std::chrono::steady_clock::time_point::max();
    }

    int unit_completed_calls = 0;
    absl::Status status_;
  };

  constexpr uint64_t kUuid = 99;
  constexpr size_t kSliceSize = 1024;
  constexpr size_t kMaxBlocks = 5;
  constexpr size_t kNumLayers = 2;
  constexpr size_t kNumShards = 1;
  MockSessionDelegate delegate(kSliceSize, kMaxBlocks, kNumLayers, kNumShards);
  auto transfer_session = std::make_shared<StubTransferSession>();
  delegate.RegisterSession(kUuid, transfer_session);

  BlockTransportConfig config;
  config.major_order = MajorOrder::kBlockMajor;
  auto session = std::make_shared<BlockTransportServerPushSession>(
      kUuid, &delegate, config);

  const lib::PostMetadata post{
      .uuid = kUuid,
      .remote_id = 7,
      .local_id = 3,  // dst_block_id >= kNumLayers
      .shard_mask = 0,
      .op = lib::TransferOp::kBlockPushExplicit,
  };
  const lib::RequestMetadata req{
      .shard = 0,
      .block_id = 1,
      .src_block_id = 2,
      .chunk_idx = 0,
      .len = kSliceSize,
  };

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host, delegate.GetHostBuffer(1, 0));
  EXPECT_THAT(session->ResolveRecvBuffer(post, req),
              IsOkAndHolds(absl::MakeSpan(host.data() + 3 * kSliceSize,
                                          kSliceSize)));
  ABSL_EXPECT_OK(session->OnPeregrinePostComplete(post, {}, absl::OkStatus()));
  EXPECT_THAT(transfer_session->unit_completed_calls, Eq(0));
  EXPECT_THAT(delegate.layers_received(), Eq(0));
  ASSERT_EQ(delegate.received_block_shards().size(), 1u);
  EXPECT_EQ(delegate.received_block_shards()[0].block_ids, std::vector<int>{3});
  EXPECT_EQ(delegate.received_block_shards()[0].shard_ids, std::vector<int>{0});
  EXPECT_EQ(delegate.received_block_shards()[0].uuid, kUuid);
}

TEST(BlockTransportServerPushSessionTest, ResolveRecvBufferReturnsChunkSpan) {
  MockSessionDelegate delegate(/*slice_size=*/1024, /*max_blocks=*/4);
  BlockTransportConfig config;
  auto session = std::make_shared<BlockTransportServerPushSession>(
      /*uuid=*/0, &delegate, config);

  const lib::PostMetadata post{.uuid = 0, .remote_id = 0, .local_id = 0};
  const absl::StatusOr<absl::Span<uint8_t>> resolved =
      session->ResolveRecvBuffer(
          post, {.shard = 0, .block_id = 2, .chunk_idx = 0, .len = 1024});
  ABSL_ASSERT_OK(resolved);
  const absl::StatusOr<absl::Span<const uint8_t>> host_buf =
      delegate.GetHostBuffer(0, 0);
  ABSL_ASSERT_OK(host_buf);
  EXPECT_EQ(resolved->data(), host_buf->data() + 2 * 1024);
  EXPECT_EQ(resolved->size(), 1024);

  EXPECT_THAT(
      session->ResolveRecvBuffer(
          post, {.shard = 0, .block_id = 2, .chunk_idx = 0, .len = 512}),
      StatusIs(absl::StatusCode::kInvalidArgument));
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

  EXPECT_DEBUG_DEATH(
      session->StartStream(/*client_fd=*/-1, header).IgnoreError(), "");
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

// Delegate whose GetBlockArrayHostSpan() returns a fixed result.
class FixedHostSpanDelegate : public MockSessionDelegate {
 public:
  explicit FixedHostSpanDelegate(absl::StatusOr<absl::Span<const uint8_t>> span)
      : span_(std::move(span)) {}

  absl::StatusOr<absl::Span<const uint8_t>> GetBlockArrayHostSpan(
      size_t block_array_idx, size_t shard_idx, uint64_t uuid) override {
    return span_;
  }

 private:
  const absl::StatusOr<absl::Span<const uint8_t>> span_;
};

TEST(ValidateChunksTest, BoundsChecksChunksAgainstHostSpan) {
  MockSessionDelegate delegate(/*slice_size=*/1024, /*max_blocks=*/4);
  const std::vector<BlockChunk> in_bounds = delegate.GetBlockChunks(
      /*layer=*/0, /*shard=*/0, std::vector<int64_t>{0, 3});
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host, delegate.GetHostBuffer(0, 0));
  const std::vector<BlockChunk> past_end = {
      {.ptr = host.data() + 3 * 1024 + 1, .size = 1024}};

  ABSL_EXPECT_OK(ValidateChunks(&delegate, 0, 0, /*uuid=*/7, in_bounds));
  EXPECT_THAT(ValidateChunks(&delegate, 0, 0, /*uuid=*/7, past_end),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("Block array size: 4096")));
}

TEST(ValidateChunksTest, PropagatesHostSpanError) {
  FixedHostSpanDelegate delegate(
      absl::FailedPreconditionError("no staging slot leased"));
  const std::vector<BlockChunk> chunks = delegate.GetBlockChunks(
      /*layer=*/0, /*shard=*/0, std::vector<int64_t>{0});

  EXPECT_THAT(ValidateChunks(&delegate, 0, 0, /*uuid=*/7, chunks),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("no staging slot leased")));
}

TEST(ValidateChunksTest, SkipsDelegateWithoutFlatHostBase) {
  FixedHostSpanDelegate delegate((absl::Span<const uint8_t>()));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host, delegate.GetHostBuffer(0, 0));
  const std::vector<BlockChunk> past_end = {
      {.ptr = host.data() + host.size(), .size = 1024}};

  ABSL_EXPECT_OK(ValidateChunks(&delegate, 0, 0, /*uuid=*/7, past_end));
}

TEST(ValidateChunksTest, RejectsExcessiveChunkCount) {
  MockSessionDelegate delegate(/*slice_size=*/1024, /*max_blocks=*/4);
  const std::vector<BlockChunk> too_many(
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1);

  EXPECT_THAT(ValidateChunks(&delegate, 0, 0, /*uuid=*/7, too_many),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("chunk_idx is 16-bit")));
}

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden

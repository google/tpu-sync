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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_service.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Gt;

struct PushRecord {
  uint64_t uuid = 0;
  uint64_t buffer_id = 0;
  uint32_t shard_id = 0;
  uint64_t chunk_batch_id = 0;
};

class FakeGrpcTransportDelegate final : public GrpcTransportDelegate {
 public:
  void AddBuffer(uint64_t buffer_id, uint32_t shard_id, size_t size_bytes,
                 uint8_t fill = 0) {
    absl::MutexLock lock(mu_);
    buffers_[{buffer_id, shard_id}] = std::vector<uint8_t>(size_bytes, fill);
  }

  std::vector<uint8_t> CopyBuffer(uint64_t buffer_id, uint32_t shard_id) const {
    absl::MutexLock lock(mu_);
    auto it = buffers_.find({buffer_id, shard_id});
    if (it == buffers_.end()) {
      return {};
    }
    return it->second;
  }

  void SetPullResponse(uint64_t uuid, uint64_t buffer_id, uint32_t shard_id,
                       uint64_t chunk_batch_id,
                       absl::StatusOr<PullChunksResponse> result) {
    absl::MutexLock lock(mu_);
    pull_results_[{uuid, buffer_id, shard_id, chunk_batch_id}] =
        std::move(result);
  }

  std::vector<PushRecord> push_records() const {
    absl::MutexLock lock(mu_);
    return push_records_;
  }

  absl::StatusOr<absl::Span<uint8_t>> GetHostBuffer(
      uint64_t buffer_id, uint32_t shard_id) override {
    absl::MutexLock lock(mu_);
    auto it = buffers_.find({buffer_id, shard_id});
    if (it == buffers_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Unknown buffer (", buffer_id, ", ", shard_id, ")"));
    }
    return absl::MakeSpan(it->second);
  }

  absl::Status OnPushChunks(uint64_t uuid, uint64_t buffer_id,
                            uint32_t shard_id,
                            uint64_t chunk_batch_id) override {
    absl::MutexLock lock(mu_);
    push_records_.push_back(PushRecord{
        .uuid = uuid,
        .buffer_id = buffer_id,
        .shard_id = shard_id,
        .chunk_batch_id = chunk_batch_id,
    });
    return absl::OkStatus();
  }

  absl::StatusOr<PullChunksResponse> OnPullChunks(
      const PullChunksRequest& request) override {
    absl::MutexLock lock(mu_);
    auto it = pull_results_.find({request.uuid, request.buffer_id,
                                  request.shard_id, request.chunk_batch_id});
    if (it == pull_results_.end()) {
      return absl::NotFoundError("No pull chunks registered");
    }
    return it->second;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::pair<uint64_t, uint32_t>, std::vector<uint8_t>>
      buffers_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::tuple<uint64_t, uint64_t, uint32_t, uint64_t>,
                      absl::StatusOr<PullChunksResponse>>
      pull_results_ ABSL_GUARDED_BY(mu_);
  std::vector<PushRecord> push_records_ ABSL_GUARDED_BY(mu_);
};

TEST(GrpcTransportTest, PushContiguousScatterChunkSucceeds) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/7, /*shard_id=*/2, /*size_bytes=*/12,
                     /*fill=*/0xAA);

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(/*requested_port=*/0), IsOkAndHolds(Gt(0)));

  GrpcTransportClient client;
  const std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6};
  PushChunksRequest request{
      .uuid = 101,
      .buffer_id = 7,
      .shard_id = 2,
      .chunk_batch_id = 3,
      .chunks = {ScatterChunkSpec{.payload = data, .byte_offset = 2}},
  };

  ABSL_ASSERT_OK(client.PushChunks(server.address(), request));

  EXPECT_THAT(
      delegate.CopyBuffer(7, 2),
      ElementsAre(0xAA, 0xAA, 1, 2, 3, 4, 5, 6, 0xAA, 0xAA, 0xAA, 0xAA));
  const std::vector<PushRecord> records = delegate.push_records();
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].uuid, 101);
  EXPECT_EQ(records[0].buffer_id, 7);
  EXPECT_EQ(records[0].shard_id, 2);
  EXPECT_EQ(records[0].chunk_batch_id, 3);
}

TEST(GrpcTransportTest, PushStridedScatterChunkPreservesGaps) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/1, /*shard_id=*/0, /*size_bytes=*/14,
                     /*fill=*/0);

  GrpcTransportServer server(&delegate);
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client;
  const std::vector<uint8_t> data = {10, 11, 20, 21, 30, 31};
  PushChunksRequest request{
      .uuid = 202,
      .buffer_id = 1,
      .shard_id = 0,
      .chunk_batch_id = 1,
      .chunks = {ScatterChunkSpec{
          .payload = data,
          .byte_offset = 1,
          .byte_size = 2,
          .byte_stride = 4,
      }},
  };

  ABSL_ASSERT_OK(client.PushChunks(server.address(), request));

  // Slices land at [1..2], [5..6], [9..10]; gaps remain 0.
  EXPECT_THAT(delegate.CopyBuffer(1, 0),
              ElementsAre(0, 10, 11, 0, 0, 20, 21, 0, 0, 30, 31, 0, 0, 0));
}

TEST(GrpcTransportTest, PushGatherChunkConcatenatesPayloadsAtOffset) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/3, /*shard_id=*/1, /*size_bytes=*/10,
                     /*fill=*/0xFF);

  GrpcTransportServer server(&delegate);
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client;
  const std::vector<uint8_t> p0 = {1, 2};
  const std::vector<uint8_t> p1 = {3, 4, 5};
  const std::vector<uint8_t> p2 = {6};
  PushChunksRequest request{
      .uuid = 303,
      .buffer_id = 3,
      .shard_id = 1,
      .chunk_batch_id = 9,
      .chunks = {GatherChunkSpec{.payloads = {p0, p1, p2}, .byte_offset = 2}},
  };

  ABSL_ASSERT_OK(client.PushChunks(server.address(), request));

  EXPECT_THAT(delegate.CopyBuffer(3, 1),
              ElementsAre(0xFF, 0xFF, 1, 2, 3, 4, 5, 6, 0xFF, 0xFF));
}

TEST(GrpcTransportTest, PushRejectsInvalidChunksAtomically) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/4, /*shard_id=*/0, /*size_bytes=*/8,
                     /*fill=*/0x55);

  GrpcTransportServer server(&delegate);
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client;

  // Chunk 0 is valid, chunk 1 overflows buffer -> buffer must stay untouched.
  const std::vector<uint8_t> valid_data = {1, 2, 3, 4};
  const std::vector<uint8_t> bad_data = {9, 9, 9};
  PushChunksRequest oob_req{
      .uuid = 1,
      .buffer_id = 4,
      .shard_id = 0,
      .chunks =
          {
              ScatterChunkSpec{.payload = valid_data, .byte_offset = 0},
              ScatterChunkSpec{.payload = bad_data, .byte_offset = 6},
          },
  };

  EXPECT_THAT(client.PushChunks(server.address(), oob_req),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(delegate.CopyBuffer(4, 0),
              ElementsAre(0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55));
  EXPECT_TRUE(delegate.push_records().empty());

  // Stride smaller than byte_size.
  const std::vector<uint8_t> stride_data = {1, 2, 3, 4, 5, 6, 7, 8};
  PushChunksRequest bad_stride_req{
      .buffer_id = 4,
      .shard_id = 0,
      .chunks = {ScatterChunkSpec{
          .payload = stride_data, .byte_size = 4, .byte_stride = 2}},
  };
  EXPECT_THAT(client.PushChunks(server.address(), bad_stride_req),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Payload size not divisible by byte_size.
  PushChunksRequest bad_div_req{
      .buffer_id = 4,
      .shard_id = 0,
      .chunks = {ScatterChunkSpec{.payload = valid_data, .byte_size = 3}},
  };
  EXPECT_THAT(client.PushChunks(server.address(), bad_div_req),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Unknown buffer returns NotFound via StatusProto.
  PushChunksRequest unknown_buf_req{.buffer_id = 999, .shard_id = 0};
  EXPECT_THAT(client.PushChunks(server.address(), unknown_buf_req),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GrpcTransportTest, PullChunksWritesIntoDstBufferAndPropagatesErrors) {
  FakeGrpcTransportDelegate delegate;

  auto backing = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{10, 11, 20, 21, 30, 31, 40, 41, 42});
  PullChunksResponse pull_resp{
      .chunks =
          {
              ScatterChunkSpec{
                  .payload = absl::MakeConstSpan(*backing).subspan(0, 4),
                  .byte_offset = 1,
                  .byte_size = 2,
                  .byte_stride = 4,
              },
              GatherChunkSpec{
                  .payloads =
                      {
                          absl::MakeConstSpan(*backing).subspan(4, 2),
                          absl::MakeConstSpan(*backing).subspan(6, 3),
                      },
                  .byte_offset = 10,
              },
          },
      .owner = backing,
  };
  delegate.SetPullResponse(/*uuid=*/50, /*buffer_id=*/2, /*shard_id=*/1,
                           /*chunk_batch_id=*/4, std::move(pull_resp));

  GrpcTransportServer server(&delegate);
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client;
  PullChunksRequest req{
      .uuid = 50,
      .buffer_id = 2,
      .shard_id = 1,
      .chunk_batch_id = 4,
  };

  std::vector<uint8_t> dst(16, 0);
  ABSL_ASSERT_OK(client.PullChunks(server.address(), req, absl::MakeSpan(dst)));
  EXPECT_THAT(dst, ElementsAre(0, 10, 11, 0, 0, 20, 21, 0, 0, 0, 30, 31, 40, 41,
                               42, 0));

  // Destination too small fails atomically without mutating dst.
  std::vector<uint8_t> small_dst(8, 0xCC);
  EXPECT_THAT(
      client.PullChunks(server.address(), req, absl::MakeSpan(small_dst)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(small_dst,
              ElementsAre(0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC));

  // Unregistered pull returns NotFound via StatusProto.
  PullChunksRequest missing_req{.uuid = 999};
  EXPECT_THAT(
      client.PullChunks(server.address(), missing_req, absl::MakeSpan(dst)),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(GrpcTransportTest,
     MultiChannelClientPoolsChannelsAndHandlesConcurrentPush) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/10, /*shard_id=*/0, /*size_bytes=*/16,
                     /*fill=*/0);

  GrpcTransportServer server(&delegate);
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client(
      GrpcTransportClientConfig{.num_channels_per_peer = 3});

  std::vector<std::thread> workers;
  workers.reserve(4);
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&, i]() {
      const std::vector<uint8_t> data(4, static_cast<uint8_t>(i + 1));
      PushChunksRequest req{
          .uuid = static_cast<uint64_t>(1000 + i),
          .buffer_id = 10,
          .shard_id = 0,
          .chunk_batch_id = static_cast<uint64_t>(i),
          .chunks = {ScatterChunkSpec{
              .payload = data,
              .byte_offset = static_cast<uint64_t>(i * 4),
          }},
      };
      ABSL_EXPECT_OK(client.PushChunks(server.address(), req));
    });
  }
  for (std::thread& t : workers) {
    t.join();
  }

  EXPECT_EQ(client.CachedChannelCount(server.address()), 3);
  EXPECT_THAT(delegate.CopyBuffer(10, 0),
              ElementsAre(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4));
  EXPECT_EQ(delegate.push_records().size(), 4);
}

TEST(GrpcTransportTest, ZeroCopyPushChunksScatterAndGatherSucceeds) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/20, /*shard_id=*/1, /*size_bytes=*/16,
                     /*fill=*/0);

  GrpcTransportServer server(&delegate);
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client;
  const std::vector<uint8_t> scatter_src = {10, 11, 20, 21};
  const std::vector<uint8_t> gather_part0 = {30, 31};
  const std::vector<uint8_t> gather_part1 = {40, 41, 42};

  PushChunksRequest req{
      .uuid = 777,
      .buffer_id = 20,
      .shard_id = 1,
      .chunk_batch_id = 5,
      .chunks =
          {
              ScatterChunkSpec{
                  .payload = scatter_src,
                  .byte_offset = 1,
                  .byte_size = 2,
                  .byte_stride = 4,
              },
              GatherChunkSpec{
                  .payloads = {gather_part0, gather_part1},
                  .byte_offset = 10,
              },
          },
  };

  ABSL_ASSERT_OK(client.PushChunks(server.address(), req));

  EXPECT_THAT(
      delegate.CopyBuffer(20, 1),
      ElementsAre(0, 10, 11, 0, 0, 20, 21, 0, 0, 0, 30, 31, 40, 41, 42, 0));
  const std::vector<PushRecord> records = delegate.push_records();
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].uuid, 777);
  EXPECT_EQ(records[0].chunk_batch_id, 5);
}

TEST(GrpcTransportTest, EnforcesMaxMessageSizeBound) {
  FakeGrpcTransportDelegate delegate;
  delegate.AddBuffer(/*buffer_id=*/1, /*shard_id=*/0, /*size_bytes=*/512,
                     /*fill=*/0);

  GrpcTransportServer server(
      &delegate, GrpcTransportServerConfig{.max_message_size_bytes = 64});
  ABSL_ASSERT_OK(server.Start());

  GrpcTransportClient client;
  const std::vector<uint8_t> oversized(256, 0xAB);
  PushChunksRequest req{
      .uuid = 1,
      .buffer_id = 1,
      .shard_id = 0,
      .chunks = {ScatterChunkSpec{.payload = oversized, .byte_offset = 0}},
  };
  EXPECT_THAT(client.PushChunks(server.address(), req),
              StatusIs(absl::StatusCode::kResourceExhausted));

  // Non-positive max_message_size_bytes is rejected.
  GrpcTransportServer invalid_server(
      &delegate, GrpcTransportServerConfig{.max_message_size_bytes = -1});
  EXPECT_THAT(invalid_server.Start(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

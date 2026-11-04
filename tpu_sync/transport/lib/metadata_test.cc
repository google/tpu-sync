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

#include "tpu_sync/transport/lib/metadata.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::HasSubstr;

constexpr PostMetadata kPostV1{.uuid = 1,
                               .remote_id = 2,
                               .local_id = 3,
                               .op = TransferOp::kBlockPushExplicit,
                               .version = kMetadataVersionV1};
constexpr PostMetadata kPostV2{.uuid = 1,
                               .remote_id = 2,
                               .local_id = 3,
                               .op = TransferOp::kBlockPushExplicit,
                               .version = 2};

constexpr RequestMetadata kRequest{
    .shard = 1, .block_id = 2, .src_block_id = 3, .chunk_idx = 4, .len = 5};

TEST(MetadataTest, PostMetadataV1RoundTripsAndPinsLayout) {
  const PostMetadata post{.uuid = 0x0123456789ABCDEFULL,
                          .remote_id = 0x11223344,
                          .local_id = 0x55667788,
                          .shard_mask = 0x9ABC,
                          .op = TransferOp::kBlockPull,
                          .version = kMetadataVersionV1};
  const MetadataBytes bytes = EncodePostMetadata(post);
  // version | rsvd | uuid | remote_id | local_id | shard_mask | op | rsvd
  EXPECT_THAT(bytes, ElementsAreArray<uint8_t>({
                         0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
                         0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,  //
                         0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55,  //
                         0xBC, 0x9A, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
                     }));

  EXPECT_THAT(DecodePostMetadata(bytes),
              IsOkAndHolds(FieldsAre(post.uuid, post.remote_id, post.local_id,
                                     post.shard_mask, post.op, /*version=*/1)));
}

TEST(MetadataTest, SendersEmitDefaultVersion) {
  const MetadataBytes bytes = EncodePostMetadata(PostMetadata{
      .uuid = 1, .remote_id = 2, .local_id = 3, .op = TransferOp::kBlockPush});
  EXPECT_EQ(bytes[0], kMetadataVersion);
  EXPECT_THAT(DecodePostMetadata(bytes),
              IsOkAndHolds(Field(&PostMetadata::version, kMetadataVersion)));
}

TEST(MetadataTest, PostMetadataRoundTripsEveryOp) {
  for (const TransferOp op : {TransferOp::kBlockPush, TransferOp::kBlockPull,
                              TransferOp::kBlockPushExplicit}) {
    EXPECT_THAT(DecodePostMetadata(EncodePostMetadata(PostMetadata{
                    .uuid = 1, .remote_id = 2, .local_id = 3, .op = op})),
                IsOkAndHolds(Field(&PostMetadata::op, op)));
  }
}

TEST(MetadataTest, RequestMetadataV1RoundTripsAndPinsLayout) {
  const RequestMetadata req{.shard = 0x11223344,
                            .block_id = -2,
                            .src_block_id = 0x55667788,
                            .chunk_idx = 0x9ABC,
                            .len = 0xDDEEFF00};
  const MetadataBytes bytes = EncodeRequestMetadata(kPostV1, req);
  // shard | block_id | src_block_id | chunk_idx | rsvd | len | reserved
  EXPECT_THAT(bytes, ElementsAreArray<uint8_t>({
                         0x44, 0x33, 0x22, 0x11, 0xFE, 0xFF, 0xFF, 0xFF,  //
                         0x88, 0x77, 0x66, 0x55, 0xBC, 0x9A, 0x00, 0x00,  //
                         0x00, 0xFF, 0xEE, 0xDD, 0x00, 0x00, 0x00, 0x00,  //
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     }));

  EXPECT_THAT(DecodeRequestMetadata(kPostV1, bytes),
              IsOkAndHolds(FieldsAre(req.shard, req.block_id, req.src_block_id,
                                     req.chunk_idx, req.len)));
}

TEST(MetadataTest, DecodePostMetadataRejectsUnknownVersion) {
  MetadataBytes post = EncodePostMetadata(PostMetadata{
      .uuid = 1, .remote_id = 2, .local_id = 3, .op = TransferOp::kBlockPush});
  post[0] = 2;  // version
  EXPECT_THAT(
      DecodePostMetadata(post),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("version: 2")));
}

TEST(MetadataTest, DecodePostMetadataPassesOpThrough) {
  MetadataBytes post = EncodePostMetadata(PostMetadata{
      .uuid = 1, .remote_id = 2, .local_id = 3, .op = TransferOp::kBlockPush});
  post[26] = 0x7F;  // op
  EXPECT_THAT(
      DecodePostMetadata(post),
      IsOkAndHolds(Field(&PostMetadata::op, static_cast<TransferOp>(0x7F))));
}

TEST(MetadataTest, EmptyBlobMeansNoMetadata) {
  EXPECT_TRUE(IsEmptyMetadata(MetadataBytes{}));
  MetadataBytes last_byte_set = {};
  last_byte_set[31] = 1;
  EXPECT_FALSE(IsEmptyMetadata(last_byte_set));
  EXPECT_FALSE(IsEmptyMetadata(
      EncodeRequestMetadata(kPostV1, RequestMetadata{.shard = 0,
                                                     .block_id = 0,
                                                     .src_block_id = 0,
                                                     .chunk_idx = 0,
                                                     .len = 1})));
  EXPECT_FALSE(IsEmptyMetadata(
      EncodePostMetadata(PostMetadata{.uuid = 0,
                                      .remote_id = 0,
                                      .local_id = 0,
                                      .op = TransferOp::kBlockPush})));
}

TEST(MetadataTest, DecodeRequestMetadataRejectsZeroLen) {
  MetadataBytes req = EncodeRequestMetadata(kPostV1, kRequest);
  for (int i = 16; i < 20; ++i) req[i] = 0;  // len
  EXPECT_THAT(
      DecodeRequestMetadata(kPostV1, req),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Zero len")));
  // An empty blob is rejected, not read as shard 0 / block 0.
  EXPECT_THAT(DecodeRequestMetadata(kPostV1, MetadataBytes{}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, DecodePostMetadataRejectsEmptyBlob) {
  EXPECT_THAT(
      DecodePostMetadata(MetadataBytes{}),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("version: 0")));
}

TEST(MetadataTest, DecodeRequestMetadataRejectsUnknownVersion) {
  const MetadataBytes req = EncodeRequestMetadata(kPostV1, kRequest);
  EXPECT_THAT(
      DecodeRequestMetadata(kPostV2, req),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("version: 2")));
}

TEST(MetadataTest, DecodePostMetadataRejectsNonZeroReserved) {
  for (const int offset : {1, 2, 3, 4, 5, 6, 7, 27, 28, 29, 30, 31}) {
    MetadataBytes post =
        EncodePostMetadata(PostMetadata{.uuid = 1,
                                        .remote_id = 2,
                                        .local_id = 3,
                                        .op = TransferOp::kBlockPush});
    post[offset] = 1;
    EXPECT_THAT(DecodePostMetadata(post),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Non-zero reserved field")))
        << "offset " << offset;
  }
}

TEST(MetadataTest, DecodeRequestMetadataRejectsNonZeroReserved) {
  for (const int offset :
       {14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}) {
    MetadataBytes req = EncodeRequestMetadata(kPostV1, kRequest);
    req[offset] = 1;
    EXPECT_THAT(DecodeRequestMetadata(kPostV1, req),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Non-zero reserved field")))
        << "offset " << offset;
  }
}

TEST(MetadataDeathTest, EncodePostMetadataRejectsUnknownVersion) {
  EXPECT_DEATH(EncodePostMetadata(PostMetadata{.uuid = 1,
                                               .remote_id = 2,
                                               .local_id = 3,
                                               .op = TransferOp::kBlockPush,
                                               .version = 2}),
               "Unsupported post metadata version: 2");
}

TEST(MetadataDeathTest, EncodeRequestMetadataRejectsUnknownVersion) {
  EXPECT_DEATH(EncodeRequestMetadata(kPostV2, kRequest),
               "Unsupported request metadata version: 2");
}

TEST(MetadataDeathTest, EncodeRequestMetadataRejectsZeroLen) {
  RequestMetadata req = kRequest;
  req.len = 0;
  EXPECT_DEATH(EncodeRequestMetadata(kPostV1, req), "non-zero len");
}

}  // namespace
}  // namespace tpu_raiden::transport::lib

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

#include "tpu_sync/weight_sync/shard_push_entry.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Property;
using ::tpu_sync::rpc::ShardPushEntryProto;

constexpr absl::string_view kEntryCtx = "push (uuid=7) shard 0 entry[1]";
// Four of these overflow uint64_t.
constexpr int64_t kTwoPow62Bytes = int64_t{1} << 62;

// Returns ShardPushEntry::FromProto(`proto`, `kEntryCtx`, `options`).
absl::StatusOr<ShardPushEntry> EntryFromProto(
    const ShardPushEntryProto& proto,
    const ShardPushEntryOptions& options = {}) {
  return ShardPushEntry::FromProto(proto, kEntryCtx, options);
}

// Matches the InvalidArgument error `kEntryCtx` followed by `detail`.
auto IsEntryError(absl::string_view detail) {
  return StatusIs(absl::StatusCode::kInvalidArgument,
                  absl::StrCat(kEntryCtx, detail));
}

TEST(ShardPushEntryTest, UnsetCountAndStridesMeanOneSlice) {
  ShardPushEntryProto proto;
  proto.set_dst_shard_idx(3);
  proto.set_src_offset_bytes(128);
  proto.set_dst_offset_bytes(256);
  proto.set_size_bytes(64);
  EXPECT_THAT(ShardPushEntry::FromProto(proto, kEntryCtx),
              IsOkAndHolds(
                  AllOf(Property(&ShardPushEntry::dst_shard_idx, 3),
                        Property(&ShardPushEntry::count, 1),
                        Property(&ShardPushEntry::src_offset, 128),
                        Property(&ShardPushEntry::dst_offset, 256),
                        Property(&ShardPushEntry::size, 64),
                        Property(&ShardPushEntry::effective_src_stride, 64),
                        Property(&ShardPushEntry::effective_dst_stride, 64),
                        Property(&ShardPushEntry::is_inner_contiguous, true),
                        Property(&ShardPushEntry::inner_src_span, 64),
                        Property(&ShardPushEntry::outer_counts, IsEmpty()),
                        Property(&ShardPushEntry::outer_src_strides, IsEmpty()),
                        Property(&ShardPushEntry::outer_dst_strides, IsEmpty()),
                        Property(&ShardPushEntry::num_outer, 1))));
}

TEST(ShardPushEntryTest, RowsAreContiguousOnlyWhenBothStridesEqualTheSlice) {
  ShardPushEntryProto proto;
  proto.set_count(3);
  proto.set_size_bytes(8);
  proto.set_src_stride_bytes(16);
  proto.set_dst_stride_bytes(32);
  EXPECT_THAT(EntryFromProto(proto),
              IsOkAndHolds(AllOf(
                  Property(&ShardPushEntry::count, 3),
                  Property(&ShardPushEntry::effective_src_stride, 16),
                  Property(&ShardPushEntry::effective_dst_stride, 32),
                  Property(&ShardPushEntry::is_inner_contiguous, false),
                  Property(&ShardPushEntry::inner_src_span, 2 * 16 + 8))));

  proto.set_src_stride_bytes(8);
  proto.clear_dst_stride_bytes();
  EXPECT_THAT(
      EntryFromProto(proto),
      IsOkAndHolds(AllOf(Property(&ShardPushEntry::is_inner_contiguous, true),
                         Property(&ShardPushEntry::inner_src_span, 3 * 8))));
}

TEST(ShardPushEntryTest, BoundsTheFarthestOuterRowByAKnownHostSize) {
  ShardPushEntryProto proto;
  proto.set_size_bytes(8);
  proto.add_outer_counts(2);
  proto.add_outer_counts(3);
  proto.add_outer_src_strides_bytes(48);
  proto.add_outer_src_strides_bytes(16);
  proto.add_outer_dst_strides_bytes(96);
  proto.add_outer_dst_strides_bytes(32);
  // The farthest source row starts at 1 * 48 + 2 * 16 = 80.
  EXPECT_THAT(
      EntryFromProto(proto, {.host_size = 88}),
      IsOkAndHolds(AllOf(
          Property(&ShardPushEntry::outer_counts, ElementsAre(2, 3)),
          Property(&ShardPushEntry::outer_src_strides, ElementsAre(48, 16)),
          Property(&ShardPushEntry::outer_dst_strides, ElementsAre(96, 32)),
          Property(&ShardPushEntry::num_outer, 6))));
  EXPECT_THAT(EntryFromProto(proto, {.host_size = 87}),
              IsEntryError(
                  " source push range [80, 80+8) exceeds host buffer size 87"));
  EXPECT_THAT(EntryFromProto(proto),
              IsOkAndHolds(Property(&ShardPushEntry::num_outer, 6)));
}

TEST(ShardPushEntryTest, PreTiledEntryRelaysAWholeShardFromOffsetZero) {
  ShardPushEntryProto proto;
  proto.set_count(4);
  proto.set_src_offset_bytes(128);
  proto.set_dst_offset_bytes(256);
  proto.set_size_bytes(64);
  proto.set_src_stride_bytes(128);
  proto.set_dst_stride_bytes(128);
  EXPECT_THAT(
      EntryFromProto(
          proto, {.pre_tiled = true, .device_size = 4096, .host_size = 4096}),
      IsOkAndHolds(AllOf(Property(&ShardPushEntry::count, 1),
                         Property(&ShardPushEntry::src_offset, 0),
                         Property(&ShardPushEntry::dst_offset, 0),
                         Property(&ShardPushEntry::size, 4096),
                         Property(&ShardPushEntry::effective_src_stride, 4096),
                         Property(&ShardPushEntry::effective_dst_stride, 4096),
                         Property(&ShardPushEntry::is_inner_contiguous, true),
                         Property(&ShardPushEntry::inner_src_span, 4096))));
  // Without a device size, the entry's own size is relayed from offset 0.
  EXPECT_THAT(EntryFromProto(proto, {.pre_tiled = true, .host_size = 64}),
              IsOkAndHolds(AllOf(Property(&ShardPushEntry::src_offset, 0),
                                 Property(&ShardPushEntry::size, 64))));
}

TEST(ShardPushEntryTest, BoundsTheOuterChunkCountByMaxPushEntryChunks) {
  ShardPushEntryProto proto;
  proto.set_size_bytes(8);
  // Dim 0 exercises MulCheckedU64(0, ...); 1 * 65535 * 65537 == 2^32 - 1.
  for (const int32_t count : {1, 65535, 65537}) {
    proto.add_outer_counts(count);
    proto.add_outer_src_strides_bytes(count == 1 ? 16 : 0);
    proto.add_outer_dst_strides_bytes(count == 1 ? 16 : 0);
  }
  EXPECT_THAT(
      EntryFromProto(proto, {.host_size = 8}),
      IsOkAndHolds(Property(&ShardPushEntry::num_outer, kMaxPushEntryChunks)));

  proto.set_outer_counts(1, 65536);
  proto.set_outer_counts(2, 65536);
  EXPECT_THAT(
      EntryFromProto(proto),
      IsEntryError(" outer_counts product exceeds uint32_t max at dim 2 "
                   "(outer_counts[2]=65536)"));
}

TEST(ShardPushEntryTest, RejectsANegativeScalarFieldByName) {
  struct NegativeFieldCase {
    void (*set_negative)(ShardPushEntryProto& proto);
    absl::string_view detail;
  };
  const NegativeFieldCase cases[] = {
      {[](ShardPushEntryProto& p) { p.set_dst_shard_idx(-1); },
       " sets negative dst_shard_idx=-1"},
      {[](ShardPushEntryProto& p) { p.set_src_offset_bytes(-2); },
       " sets negative src_offset_bytes=-2"},
      {[](ShardPushEntryProto& p) { p.set_dst_offset_bytes(-3); },
       " sets negative dst_offset_bytes=-3"},
      {[](ShardPushEntryProto& p) { p.set_size_bytes(-4); },
       " sets negative size_bytes=-4"},
      {[](ShardPushEntryProto& p) { p.set_src_stride_bytes(-5); },
       " sets negative src_stride_bytes=-5"},
      {[](ShardPushEntryProto& p) { p.set_dst_stride_bytes(-6); },
       " sets negative dst_stride_bytes=-6"},
      {[](ShardPushEntryProto& p) { p.set_count(-7); },
       " sets negative count=-7"},
  };
  for (const NegativeFieldCase& c : cases) {
    ShardPushEntryProto proto;
    c.set_negative(proto);
    EXPECT_THAT(EntryFromProto(proto), IsEntryError(c.detail));
  }
}

TEST(ShardPushEntryTest, RejectsOuterArraysOfDifferentLengths) {
  ShardPushEntryProto proto;
  proto.set_size_bytes(8);
  proto.add_outer_counts(2);
  proto.add_outer_counts(2);
  proto.add_outer_src_strides_bytes(8);
  proto.add_outer_dst_strides_bytes(8);
  proto.add_outer_dst_strides_bytes(16);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" has mismatched outer_counts (2), "
                           "outer_src_strides_bytes (1), and "
                           "outer_dst_strides_bytes (2) sizes"));

  proto.add_outer_src_strides_bytes(16);
  proto.mutable_outer_dst_strides_bytes()->RemoveLast();
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" has mismatched outer_counts (2), "
                           "outer_src_strides_bytes (2), and "
                           "outer_dst_strides_bytes (1) sizes"));
}

TEST(ShardPushEntryTest, RejectsAPreTiledEntryWithOuterDims) {
  ShardPushEntryProto proto;
  proto.set_size_bytes(64);
  proto.add_outer_counts(2);
  proto.add_outer_src_strides_bytes(64);
  proto.add_outer_dst_strides_bytes(64);
  EXPECT_THAT(EntryFromProto(proto, {.pre_tiled = true}),
              IsEntryError(" HOST_TILING_MODE_PRE_TILED entries must not be "
                           "strided (outer_counts_size=1)"));
}

TEST(ShardPushEntryTest, RejectsStridesSmallerThanTheSlice) {
  ShardPushEntryProto proto;
  proto.set_count(2);
  proto.set_size_bytes(8);
  proto.set_src_stride_bytes(4);
  proto.set_dst_stride_bytes(8);
  EXPECT_THAT(
      EntryFromProto(proto),
      IsEntryError(" source stride (4) cannot be smaller than slice size (8)"));

  proto.set_src_stride_bytes(8);
  proto.set_dst_stride_bytes(4);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" destination stride (4) cannot be smaller than "
                           "slice size (8)"));
}

TEST(ShardPushEntryTest, RejectsInnerSpansThatOverflow) {
  ShardPushEntryProto contiguous;
  contiguous.set_count(4);
  contiguous.set_size_bytes(kTwoPow62Bytes);
  EXPECT_THAT(EntryFromProto(contiguous),
              IsEntryError(" inner contiguous span overflows (count=4, "
                           "size_bytes=4611686018427387904)"));

  // Unchecked, this source span wraps to 8 bytes and passes any bounds check.
  ShardPushEntryProto strided;
  strided.set_count(5);
  strided.set_size_bytes(8);
  strided.set_src_stride_bytes(kTwoPow62Bytes);
  strided.set_dst_stride_bytes(8);
  EXPECT_THAT(EntryFromProto(strided),
              IsEntryError(" inner strided source span overflows (count=5, "
                           "src_stride_bytes=4611686018427387904, "
                           "size_bytes=8)"));

  strided.set_src_stride_bytes(8);
  strided.set_dst_stride_bytes(kTwoPow62Bytes);
  EXPECT_THAT(EntryFromProto(strided),
              IsEntryError(" inner strided destination span overflows "
                           "(count=5, dst_stride_bytes=4611686018427387904, "
                           "size_bytes=8)"));
}

TEST(ShardPushEntryTest, RejectsNonPositiveOuterCountsAndNegativeOuterStrides) {
  ShardPushEntryProto proto;
  proto.set_size_bytes(8);
  proto.add_outer_counts(2);
  proto.add_outer_counts(0);
  proto.add_outer_src_strides_bytes(16);
  proto.add_outer_src_strides_bytes(-8);
  proto.add_outer_dst_strides_bytes(16);
  proto.add_outer_dst_strides_bytes(-8);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" outer_counts[1] must be positive; got 0"));

  proto.set_outer_counts(1, -1);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" outer_counts[1] must be positive; got -1"));

  proto.set_outer_counts(1, 2);
  EXPECT_THAT(
      EntryFromProto(proto),
      IsEntryError(" outer_src_strides_bytes[1] must be non-negative; got -8"));

  proto.set_outer_src_strides_bytes(1, 8);
  EXPECT_THAT(
      EntryFromProto(proto),
      IsEntryError(" outer_dst_strides_bytes[1] must be non-negative; got -8"));
}

TEST(ShardPushEntryTest, RejectsOuterOffsetsThatOverflow) {
  ShardPushEntryProto proto;
  proto.set_size_bytes(8);
  proto.add_outer_counts(5);
  proto.add_outer_src_strides_bytes(kTwoPow62Bytes);
  proto.add_outer_dst_strides_bytes(0);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" outer source offset overflows at dim 0 "
                           "(outer_counts[0]=5, "
                           "outer_src_strides_bytes[0]=4611686018427387904)"));

  proto.set_outer_src_strides_bytes(0, 0);
  proto.set_outer_dst_strides_bytes(0, kTwoPow62Bytes);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" outer destination offset overflows at dim 0 "
                           "(outer_counts[0]=5, "
                           "outer_dst_strides_bytes[0]=4611686018427387904)"));
}

TEST(ShardPushEntryTest, RejectsRangeEndsThatOverflow) {
  // Two slices of INT64_MAX bytes span 2^64 - 2 bytes, so an offset of 2
  // overflows the end of the range.
  ShardPushEntryProto proto;
  proto.set_count(2);
  proto.set_size_bytes(std::numeric_limits<int64_t>::max());
  proto.set_src_offset_bytes(2);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" source span overflows (max_src_offset=2, "
                           "inner_src_span=18446744073709551614)"));

  proto.set_src_offset_bytes(0);
  proto.set_dst_offset_bytes(2);
  EXPECT_THAT(EntryFromProto(proto),
              IsEntryError(" destination span overflows (max_dst_offset=2, "
                           "inner_dst_span=18446744073709551614)"));
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

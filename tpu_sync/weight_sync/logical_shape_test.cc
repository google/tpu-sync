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

#include "tpu_sync/weight_sync/logical_shape.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Property;

tpu_sync::rpc::LogicalShapeProto MakeProto(xla::PrimitiveType type,
                                           absl::Span<const int64_t> dims) {
  tpu_sync::rpc::LogicalShapeProto proto;
  proto.set_element_type(static_cast<int32_t>(type));
  for (const int64_t d : dims) {
    proto.add_dims(d);
  }
  return proto;
}

TEST(LogicalShapeTest, FromProtoKeepsTheTypeAndDims) {
  const tpu_sync::rpc::LogicalShapeProto proto = MakeProto(xla::F32, {8, 128});
  TF_ASSERT_OK_AND_ASSIGN(const LogicalShape shape,
                       LogicalShape::FromProto(proto, "test"));
  EXPECT_EQ(shape.element_type(), xla::F32);
  EXPECT_THAT(shape.dims(), ElementsAre(8, 128));
  EXPECT_THAT(shape.ToProto(), EqualsProto(proto));
}

TEST(LogicalShapeTest, FromProtoComputesTheByteSizeOfWholeByteTypes) {
  const auto from_proto = [](xla::PrimitiveType type,
                             absl::Span<const int64_t> dims) {
    return LogicalShape::FromProto(MakeProto(type, dims), "test");
  };
  EXPECT_THAT(from_proto(xla::F32, {6, 10}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 240)));
  EXPECT_THAT(from_proto(xla::BF16, {8, 16}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 256)));
  EXPECT_THAT(from_proto(xla::U8, {12}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 12)));
  EXPECT_THAT(from_proto(xla::PRED, {5, 3}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 15)));
  EXPECT_THAT(from_proto(xla::C64, {2, 3}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 48)));
  EXPECT_THAT(from_proto(xla::C128, {4}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 64)));
  // Scalar (0 dims) and zero-sized dimension are valid.
  EXPECT_THAT(from_proto(xla::F32, {}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 4)));
  EXPECT_THAT(from_proto(xla::F32, {0, 16}),
              IsOkAndHolds(Property(&LogicalShape::byte_size, 0)));
}

TEST(LogicalShapeTest, FromProtoAcceptsOnlyValidXlaShapes) {
  // XLA accepts a huge dimension after a zero dimension, so FromProto() does
  // too, and the fingerprint check can build the XLA shape.
  TF_ASSERT_OK_AND_ASSIGN(const LogicalShape zero_then_huge,
                       LogicalShape::FromProto(
                           MakeProto(xla::F32, {0, int64_t{1} << 62}), "ctx"));
  EXPECT_EQ(zero_then_huge.byte_size(), 0);
  EXPECT_TRUE(zero_then_huge.IsRowMajorUntiledFingerprint(
      "f32[0,4611686018427387904]{1,0}"));

  // XLA rejects a byte count that overflows before the first zero dimension.
  EXPECT_THAT(LogicalShape::FromProto(
                  MakeProto(xla::F32, {int64_t{1} << 62, 0}), "ctx"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ctx byte size overflows before zero dims[1] for "
                       "f32[4611686018427387904,0]"));
}

TEST(LogicalShapeTest, FromProtoRejectsInvalidShapes) {
  // Invalid enum values.
  tpu_sync::rpc::LogicalShapeProto out_of_enum;
  out_of_enum.set_element_type(9999);
  EXPECT_THAT(LogicalShape::FromProto(out_of_enum, "ctx"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ctx has invalid element_type=9999"));

  // Non-array primitive types.
  for (xla::PrimitiveType non_array : {xla::PRIMITIVE_TYPE_INVALID, xla::TUPLE,
                                       xla::TOKEN, xla::OPAQUE_TYPE}) {
    EXPECT_THAT(
        LogicalShape::FromProto(MakeProto(non_array, {4}), "ctx"),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("non-array")));
  }

  // Sub-byte types.
  for (xla::PrimitiveType sub_byte :
       {xla::S1, xla::U1, xla::S2, xla::U2, xla::S4, xla::U4, xla::F4E2M1FN,
        xla::F6E3M2FN, xla::F6E2M3FN}) {
    EXPECT_THAT(
        LogicalShape::FromProto(MakeProto(sub_byte, {8}), "ctx"),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("sub-byte")));
  }

  // Negative dims (including after a 0 dim, and XLA's kUnboundedSize).
  EXPECT_THAT(LogicalShape::FromProto(MakeProto(xla::F32, {4, -1}), "ctx"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ctx has negative dims[1]=-1 in f32[4,-1]"));
  EXPECT_THAT(LogicalShape::FromProto(MakeProto(xla::F32, {0, -2}), "ctx"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("negative dims[1]=-2")));
  EXPECT_THAT(
      LogicalShape::FromProto(
          MakeProto(xla::F32, {std::numeric_limits<int64_t>::min(), 8}), "ctx"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("negative dims[0]")));

  // Overflowing element count.
  EXPECT_THAT(
      LogicalShape::FromProto(MakeProto(xla::U8, {int64_t{1} << 62, 4}), "ctx"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("element count overflows int64_t")));

  // Overflowing byte size.
  const int64_t huge = std::numeric_limits<int64_t>::max() / 2;
  EXPECT_THAT(LogicalShape::FromProto(MakeProto(xla::F32, {huge, 2}), "ctx"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("byte size overflows")));
}

TEST(LogicalShapeTest, FormatsAsLowercaseTypeAndDims) {
  TF_ASSERT_OK_AND_ASSIGN(
      const LogicalShape f32,
      LogicalShape::FromProto(MakeProto(xla::F32, {8, 128}), "test"));
  TF_ASSERT_OK_AND_ASSIGN(
      const LogicalShape pred,
      LogicalShape::FromProto(MakeProto(xla::PRED, {0, 3}), "test"));
  TF_ASSERT_OK_AND_ASSIGN(
      const LogicalShape bf16_scalar,
      LogicalShape::FromProto(MakeProto(xla::BF16, {}), "test"));

  EXPECT_EQ(absl::StrCat(f32), "f32[8,128]");
  EXPECT_EQ(absl::StrCat(pred), "pred[0,3]");
  EXPECT_EQ(absl::StrCat(bf16_scalar), "bf16[]");
}

TEST(LogicalShapeTest, ShapesAreEqualIffTypeAndDimsAre) {
  TF_ASSERT_OK_AND_ASSIGN(
      const LogicalShape f32_4x8,
      LogicalShape::FromProto(MakeProto(xla::F32, {4, 8}), "test"));
  TF_ASSERT_OK_AND_ASSIGN(const std::vector<LogicalShape> parsed,
                       LogicalShape::ParseAttr("f32[4,8];f32[8,4];s32[4,8]"));
  ASSERT_EQ(parsed.size(), 3);

  // Built independently, from a proto and from an attribute.
  EXPECT_EQ(parsed[0], f32_4x8);
  EXPECT_NE(parsed[1], f32_4x8);
  // Same byte size, different element type.
  EXPECT_NE(parsed[2], f32_4x8);
}

TEST(LogicalShapeTest, IsRowMajorUntiledFingerprintRejectsOtherLayouts) {
  TF_ASSERT_OK_AND_ASSIGN(
      const LogicalShape f32_4x8,
      LogicalShape::FromProto(MakeProto(xla::F32, {4, 8}), "test"));
  // Row-major untiled, without and with a memory space.
  EXPECT_TRUE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0}"));
  EXPECT_TRUE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0:S(1)}"));

  // Column-major, tiled (T), and metadata-prefixed (M) layouts, whose raw
  // device bytes are not the logical bytes.
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{0,1}"));
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0:T(8,128)}"));
  EXPECT_FALSE(
      f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0:T(8,128)S(1)}"));
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0:S(1)M(8)}"));

  // Memory spaces without a number.
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0:S()}"));
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("f32[4,8]{1,0:S(x)}"));

  // Fingerprints of other dimensions or element types, and no fingerprint.
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("f32[8,4]{1,0}"));
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint("bf16[4,8]{1,0}"));
  EXPECT_FALSE(f32_4x8.IsRowMajorUntiledFingerprint(""));
}

TEST(LogicalShapeTest, IsRowMajorUntiledFingerprintHandlesScalarLayouts) {
  TF_ASSERT_OK_AND_ASSIGN(
      const LogicalShape f32_scalar,
      LogicalShape::FromProto(MakeProto(xla::F32, {}), "test"));
  xla::Shape xla_scalar = xla::ShapeUtil::MakeShape(xla::F32, {});
  EXPECT_EQ(xla::ShapeUtil::HumanStringWithLayout(xla_scalar), "f32[]");
  xla_scalar.mutable_layout()->set_memory_space(1);
  EXPECT_EQ(xla::ShapeUtil::HumanStringWithLayout(xla_scalar), "f32[]{:S(1)}");

  EXPECT_TRUE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]"));
  EXPECT_TRUE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]{:S(1)}"));
  EXPECT_FALSE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]{}"));
  EXPECT_FALSE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]{:M(8)}"));
  EXPECT_FALSE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]{:S(1)M(8)}"));
  EXPECT_FALSE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]{:S()}"));
  EXPECT_FALSE(f32_scalar.IsRowMajorUntiledFingerprint("f32[]{:S(x)}"));
}

TEST(LogicalShapeTest, ParseAttrParsesEntriesInOrder) {
  TF_ASSERT_OK_AND_ASSIGN(const std::vector<LogicalShape> parsed,
                       LogicalShape::ParseAttr("bf16[16,64];f32[8,128];u8[]"));
  ASSERT_EQ(parsed.size(), 3);
  EXPECT_EQ(parsed[0].element_type(), xla::BF16);
  EXPECT_THAT(parsed[0].dims(), ElementsAre(16, 64));
  EXPECT_EQ(parsed[0].byte_size(), 2048);
  EXPECT_EQ(parsed[1].element_type(), xla::F32);
  EXPECT_THAT(parsed[1].dims(), ElementsAre(8, 128));
  EXPECT_EQ(parsed[1].byte_size(), 4096);
  EXPECT_EQ(parsed[2].element_type(), xla::U8);
  EXPECT_THAT(parsed[2].dims(), IsEmpty());
  EXPECT_EQ(parsed[2].byte_size(), 1);
}

TEST(LogicalShapeTest, ParseAttrRejectsUnparsableEntries) {
  EXPECT_THAT(LogicalShape::ParseAttr(""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute is empty")));
  EXPECT_THAT(LogicalShape::ParseAttr("f32[4];"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("logical_shapes[1] ('') is empty")));
  EXPECT_THAT(LogicalShape::ParseAttr("f32(4,4)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must have format")));
  EXPECT_THAT(LogicalShape::ParseAttr("bogus[4]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "logical_shapes[0] ('bogus[4]') has unknown XLA dtype "
                       "'bogus'"));
  EXPECT_THAT(LogicalShape::ParseAttr("f32[4,x]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid dimension 'x'")));
  EXPECT_THAT(LogicalShape::ParseAttr("f32[4,]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid dimension ''")));
  EXPECT_THAT(LogicalShape::ParseAttr("f32[99999999999999999999]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid dimension '99999999999999999999'")));
}

TEST(LogicalShapeTest, ParseAttrRejectsEntriesThatFailValidation) {
  EXPECT_THAT(LogicalShape::ParseAttr("u4[8]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported sub-byte element_type=u4")));
  EXPECT_THAT(LogicalShape::ParseAttr("token[]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-array element_type=TOKEN")));
  EXPECT_THAT(
      LogicalShape::ParseAttr("f32[-2,4]"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "logical_shapes[0] ('f32[-2,4]') has negative dims[0]=-2 "
               "in f32[-2,4]"));
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

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

#include "tpu_sync/core/layout_utils.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

// Parses a shape with its device layout, e.g. "f32[20,256]{1,0:T(8,128)}".
xla::Shape ParseShape(absl::string_view text) {
  absl::StatusOr<xla::Shape> shape = xla::ParseShape(text);
  CHECK_OK(shape.status()) << text;
  return *std::move(shape);
}

// Checks folding the leading dimensions of the `buffer` shape into those of the
// `folded` shape, both given with their device layouts.
absl::Status CheckFold(absl::string_view buffer, absl::string_view folded) {
  const xla::Shape folded_shape = ParseShape(folded);
  return CheckFoldedDeviceLayout(ParseShape(buffer), folded_shape.dimensions(),
                                 folded_shape.layout());
}

TEST(DimsToStringTest, FormatsDimensions) {
  EXPECT_EQ(DimsToString({4, 5, 256}), "[4, 5, 256]");
  EXPECT_EQ(DimsToString({}), "[]");
}

TEST(FoldsLeadingDimsTest, FoldsLeadingDimensionsIntoTheFirst) {
  EXPECT_TRUE(FoldsLeadingDims({8, 768, 2048}, {8 * 768, 2048}));
  EXPECT_TRUE(FoldsLeadingDims({2, 3, 4, 5}, {2 * 3 * 4, 5}));
  EXPECT_TRUE(FoldsLeadingDims({2, 3, 4, 5}, {2 * 3, 4, 5}));
  EXPECT_TRUE(FoldsLeadingDims({8, 768}, {8 * 768}));
  EXPECT_TRUE(FoldsLeadingDims({1, 5, 256}, {5, 256}));
}

TEST(FoldsLeadingDimsTest, RejectsOtherRegroupings) {
  // The same dimensions fold nothing.
  EXPECT_FALSE(FoldsLeadingDims({4, 5, 256}, {4, 5, 256}));
  // Folds trailing dimensions.
  EXPECT_FALSE(FoldsLeadingDims({8, 768, 2048}, {8, 768 * 2048}));
  // Regroups the dimensions it should keep.
  EXPECT_FALSE(FoldsLeadingDims({4, 8, 256}, {64, 128}));
  // The leading dimensions hold a different number of rows.
  EXPECT_FALSE(FoldsLeadingDims({4, 5, 256}, {21, 256}));
  // Splits instead of folding.
  EXPECT_FALSE(FoldsLeadingDims({20, 256}, {4, 5, 256}));
  EXPECT_FALSE(FoldsLeadingDims({4, 5}, {}));
}

TEST(CheckFoldedDeviceLayoutTest, AcceptsFoldedRowsThatFillWholeTiles) {
  // An [E, F, D] expert stack folded to [E * F, D].
  ABSL_EXPECT_OK(
      CheckFold("f32[8,64,256]{2,1,0:T(8,128)}", "f32[512,256]{1,0:T(8,128)}"));
  ABSL_EXPECT_OK(CheckFold("bf16[8,768,2048]{2,1,0:T(8,128)(2,1)}",
                           "bf16[6144,2048]{1,0:T(8,128)(2,1)}"));
  ABSL_EXPECT_OK(CheckFold("bf16[8,768,2048]{2,1,0:T(16,128)(2,1)}",
                           "bf16[6144,2048]{1,0:T(16,128)(2,1)}"));
  // Folds three dimensions; only the last of them is tiled.
  ABSL_EXPECT_OK(CheckFold("f32[2,3,16,256]{3,2,1,0:T(8,128)}",
                           "f32[96,256]{1,0:T(8,128)}"));
}

TEST(CheckFoldedDeviceLayoutTest, AcceptsFoldedDimensionsOutsideTheTiles) {
  // The tiles cover only the kept dimensions, so 5 rows need not fill a tile.
  ABSL_EXPECT_OK(CheckFold("f32[3,5,6,128]{3,2,1,0:T(8,128)}",
                           "f32[15,6,128]{2,1,0:T(8,128)}"));
  ABSL_EXPECT_OK(CheckFold("f32[4,5,7]{2,1,0}", "f32[20,7]{1,0}"));
}

TEST(CheckFoldedDeviceLayoutTest, AcceptsLeadingDimensionsOfSizeOne) {
  ABSL_EXPECT_OK(
      CheckFold("f32[1,5,256]{2,1,0:T(8,128)}", "f32[5,256]{1,0:T(8,128)}"));
}

TEST(CheckFoldedDeviceLayoutTest, IgnoresMemorySpace) {
  ABSL_EXPECT_OK(CheckFold("f32[8,64,256]{2,1,0:T(8,128)S(1)}",
                           "f32[512,256]{1,0:T(8,128)}"));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsFoldedRowsThatDoNotFillWholeTiles) {
  // Each group of 5 rows is padded to a tile of 8 rows, while a [20, 256]
  // buffer keeps its 20 rows in tiles one after another.
  EXPECT_THAT(
      CheckFold("f32[4,5,256]{2,1,0:T(8,128)}", "f32[20,256]{1,0:T(8,128)}"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("pads each group of 5 rows"),
                     HasSubstr("tiles of 8 rows"),
                     HasSubstr("a [20, 256] buffer"))));
  EXPECT_THAT(CheckFold("bf16[8,8,256]{2,1,0:T(16,128)(2,1)}",
                        "bf16[64,256]{1,0:T(16,128)(2,1)}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pads each group of 8 rows to whole tiles of "
                                 "16 rows")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsDifferentTiles) {
  // Whole tiles are not enough when TPU tiles the two shapes differently, as
  // it does for small dimensions.
  EXPECT_THAT(
      CheckFold("f32[8,2,256]{2,1,0:T(2,128)}", "f32[16,256]{1,0:T(8,128)}"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("differs from that of a [16, 256] buffer")));
  EXPECT_THAT(CheckFold("bf16[8,8,256]{2,1,0:T(8,128)(2,1)}",
                        "bf16[64,256]{1,0:T(16,128)(2,1)}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("differs from that of a [64, 256] buffer")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsLayoutsThatAreNotRowMajor) {
  EXPECT_THAT(
      CheckFold("f32[4,5,256]{2,0,1:T(4,128)}", "f32[20,256]{1,0:T(8,128)}"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("row-major")));
  EXPECT_THAT(
      CheckFold("f32[4,8,256]{2,1,0:T(8,128)}", "f32[32,256]{0,1:T(8,128)}"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("row-major")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsDimensionsThatDoNotFold) {
  EXPECT_THAT(
      CheckFold("f32[4,8,256]{2,1,0:T(8,128)}", "f32[64,128]{1,0:T(8,128)}"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("[64, 128] does not fold leading dimensions of "
                         "[4, 8, 256]")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsLayoutOfAnotherRank) {
  EXPECT_THAT(CheckFoldedDeviceLayout(
                  ParseShape("f32[4,8,256]{2,1,0:T(8,128)}"), {32, 256},
                  xla::Layout({2, 1, 0}, {xla::Tile({8, 128})})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not lay out [32, 256]")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsBufferWithoutLayout) {
  xla::Shape buffer = ParseShape("f32[8,64,256]{2,1,0:T(8,128)}");
  buffer.clear_layout();

  EXPECT_THAT(
      CheckFoldedDeviceLayout(buffer, {512, 256},
                              xla::Layout({1, 0}, {xla::Tile({8, 128})})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("the buffer has no device layout")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsTilesWithoutFixedSizes) {
  const xla::Tile tile({xla::Tile::kCombineDimension, 128});
  xla::Shape buffer = xla::ShapeUtil::MakeShape(xla::F32, {8, 64, 256});
  *buffer.mutable_layout() = xla::Layout({2, 1, 0}, {tile});

  EXPECT_THAT(
      CheckFoldedDeviceLayout(buffer, {512, 256}, xla::Layout({1, 0}, {tile})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("has a tile dimension without a fixed size")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsTilesSpanningSeveralFoldedDims) {
  const xla::Layout layout({3, 2, 1, 0}, {xla::Tile({4, 8, 128})});
  xla::Shape buffer = xla::ShapeUtil::MakeShape(xla::F32, {2, 4, 8, 128});
  *buffer.mutable_layout() = layout;

  EXPECT_THAT(
      CheckFoldedDeviceLayout(buffer, {64, 128},
                              xla::Layout({1, 0}, {xla::Tile({4, 8, 128})})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("span more than one of the folded dimensions")));
}

TEST(CheckFoldedDeviceLayoutTest, RejectsTilesInterleavingTheTileGrid) {
  const xla::Layout layout({2, 1, 0},
                           {xla::Tile({8, 128}), xla::Tile({2, 8, 128})});
  xla::Shape buffer = xla::ShapeUtil::MakeShape(xla::F32, {4, 8, 256});
  *buffer.mutable_layout() = layout;

  EXPECT_THAT(
      CheckFoldedDeviceLayout(
          buffer, {32, 256},
          xla::Layout({1, 0}, {xla::Tile({8, 128}), xla::Tile({2, 8, 128})})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("interleaves elements of different tiles")));
}

}  // namespace
}  // namespace tpu_raiden

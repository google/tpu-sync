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

#include <stdexcept>

#include <gtest/gtest.h>
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace raiden {
namespace {

TEST(GetMajorSliceByteSize, WholeByteTypesKeepTheShapeOnlySize) {
  const xla::Shape u8 = xla::ShapeUtil::MakeShape(xla::U8, {6, 1024, 2560});
  EXPECT_EQ(GetMajorSliceByteSize(u8), 1024 * 2560);
  EXPECT_EQ(GetMajorSliceByteSize(u8, /*physical_size=*/6 * 1024 * 2560),
            1024 * 2560);
  const xla::Shape f32 = xla::ShapeUtil::MakeShape(xla::F32, {192, 2560});
  EXPECT_EQ(GetMajorSliceByteSize(f32, /*physical_size=*/192 * 2560 * 4),
            2560 * 4);
}

TEST(GetMajorSliceByteSize, SubByteTypesComeFromTheOnDeviceSize) {
  // XLA reports one byte per F4 element from the shape; the device packs two.
  const xla::Shape f4 =
      xla::ShapeUtil::MakeShape(xla::F4E2M1FN, {6, 1024, 2560});
  EXPECT_EQ(GetMajorSliceByteSize(f4), 1024 * 2560);
  EXPECT_EQ(GetMajorSliceByteSize(f4, /*physical_size=*/6 * 1024 * 2560 / 2),
            1024 * 2560 / 2);
  // Without a physical size the shape-only answer stands.
  EXPECT_EQ(GetMajorSliceByteSize(f4, /*physical_size=*/0), 1024 * 2560);
}

TEST(GetMajorSliceByteSize, SubByteBufferMustSplitIntoWholeSlices) {
  const xla::Shape f4 = xla::ShapeUtil::MakeShape(xla::F4E2M1FN, {6, 8, 128});
  EXPECT_THROW(GetMajorSliceByteSize(f4, /*physical_size=*/6 * 8 * 128 / 2 + 1),
               std::invalid_argument);
}

TEST(GetMajorSliceByteSize, PredIsNotTreatedAsSubByte) {
  const xla::Shape pred = xla::ShapeUtil::MakeShape(xla::PRED, {4, 8, 128});
  EXPECT_EQ(GetMajorSliceByteSize(pred, /*physical_size=*/4 * 8 * 128),
            8 * 128);
}

}  // namespace
}  // namespace raiden

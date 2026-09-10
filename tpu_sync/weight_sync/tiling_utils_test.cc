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

#include "tpu_sync/weight_sync/tiling_utils.h"

#include <cstdint>
#include <vector>

#include "tpu_sync/core/numa_thread_pool.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace tpu_raiden::weight_sync {
namespace {

TEST(TilingUtilsTest, Standard2D) {
  // 2D matrix of shape 8x8, element type float (4 bytes).
  // Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 4x4.
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  // Tiled buffer size should be H * W * sizeof(float)
  const int64_t tiled_size_bytes = H * W * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure manually for a few values.
  // Original 8x8 matrix is divided into 2x2 grid of 4x4 tiles:
  // Tile(0, 0) covers rows 0-3, cols 0-3.
  // Tile(0, 1) covers rows 0-3, cols 4-7.
  // Tile(1, 0) covers rows 4-7, cols 0-3.
  // Tile(1, 1) covers rows 4-7, cols 4-7.
  //
  // Let's check row 0, col 4: linear index is 0*8 + 4 = 4. Value is 4.0.
  // In tiling: tile_h = 0, tile_w = 1. Tile index = 0 * 2 + 1 = 1.
  // Within tile: row_offset = 0, col_offset = 0. Offset within tile = 0*4 + 0 =
  // 0. Physical offset in tiled buffer: (tile_index * (tH * tW) +
  // offset_within_tile) * sizeof(float) = (1 * 16 + 0) * 4 = 64 bytes. Float
  // value at 64 bytes should be 4.0f.
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[16], 4.0f);

  // Let's check row 4, col 1: linear index is 4*8 + 1 = 33. Value is 33.0.
  // In tiling: tile_h = 1, tile_w = 0. Tile index = 1 * 2 + 0 = 2.
  // Within tile: row_offset = 0, col_offset = 1. Offset within tile = 0*4 + 1
  // = 1. Physical offset: (2 * 16 + 1) * 4 = 132 bytes (index 33 in float
  // array). Float value at index 33 in tiled array should be 33.0f.
  EXPECT_EQ(dst_tiled_float[33], 33.0f);

  // Now detile.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Standard2DLarge) {
  // 2D matrix of shape 256x512, element type float (4 bytes).
  // Total size: 256 * 512 * 4 = 512KB, which is > 128KB (triggers parallel
  // path). Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 128x128.
  const int64_t H = 256;
  const int64_t W = 512;
  const int64_t tH = 128;
  const int64_t tW = 128;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = H * W * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Now detile.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    ASSERT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Padding2DLarge) {
  // 2D matrix of shape 250x500, element type float (4 bytes).
  // Total size: 250 * 500 * 4 = 500KB, which is > 128KB (triggers parallel
  // path). Dimensions do not divide the tile size (128x128). Number of tiles in
  // H: ceil(250 / 128) = 2. Number of tiles in W: ceil(500 / 128) = 4. Total
  // tiles = 8. Total size in tiled buffer = 2 * 4 * 128 * 128 * 4 = 524288
  // bytes (512KB).
  const int64_t H = 250;
  const int64_t W = 500;
  const int64_t tH = 128;
  const int64_t tW = 128;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = 2 * 4 * tH * tW * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Now detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    ASSERT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Padding2D) {
  // 2D matrix of shape 6x6, element type float (4 bytes).
  // Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 4x4. The dimensions do not divide the tile size. Number of tiles
  // in W: ceil(6 / 4) = 2. Number of tiles in H: ceil(6 / 4) = 2. Total tiles
  // = 4. Total size in tiled buffer = 2 * 2 * 4 * 4 * 4 = 256 bytes (64
  // floats).
  const int64_t H = 6;
  const int64_t W = 6;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = 2 * 2 * tH * tW * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify that padding elements are 0.
  // E.g., Tile(1, 1) covers rows 4-7, cols 4-7. But matrix only goes up to
  // index 5. Row 5, col 5 is in matrix. Row 5, col 6 is padding. Tile(1, 1)
  // index is 1 * 2 + 1 = 3. Within Tile(1, 1), Row 5 (offset 1), Col 6 (offset
  // 2) -> offset = 1 * 4 + 2 = 6. Float index: 3 * 16 + 6 = 54. It should be
  // 0.0f.
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[54], 0.0f);

  // Now detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, PermutedOuterLayout) {
  // Shape: F32[2, 3, 8, 8]
  // Outer dimensions: 0 (size 2), 1 (size 3)
  // Tiled dimensions: 2 (size 8), 3 (size 8)
  // Tile dimensions: 4x4
  // Layout minor_to_major: {3, 2, 0, 1}
  const int64_t D0 = 2;
  const int64_t D1 = 3;
  const int64_t D2 = 8;
  const int64_t D3 = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, D1, D2, D3}, {3, 2, 0, 1},
      {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * D1 * D2 * D3;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const float* dst_tiled_float =
      reinterpret_cast<const float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[128], 64.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Bf16SubTiling) {
  // Shape: BF16[16, 256]
  // Tile 1: 8x128
  // Tile 2: 2x1
  const int64_t H = 16;
  const int64_t W = 256;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(uint16_t);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const uint16_t* dst_tiled_uint16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_uint16[1], 256);

  // Detile back.
  std::vector<uint16_t> dst_linear(num_elements, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Standard3D) {
  // 3D matrix of shape 2x8x8, element type float (4 bytes).
  // Layout has minor_to_major={2, 1, 0} (standard row-major), and tiling with
  // tile dimensions 4x4.
  const int64_t D0 = 2;
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, H, W}, {2, 1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure.
  // Block 0 (d0=0) covers indices 0-63.
  // Block 1 (d0=1) covers indices 64-127.
  // Within Block 1, the 2D matrix is 8x8 tiled with 4x4.
  // Let's check d0=1, row 0, col 4: linear index is 1*64 + 0*8 + 4 = 68. Value
  // is 68.0. Physically, Block 1 starts at 64 * 4 = 256 bytes. Within Block 1,
  // row 0, col 4 is in Tile(0,1), offset 0. Tile(0,1) index is 1. Offset within
  // Block 1: 1 * 16 * 4 = 64 bytes. Total physical offset: 256 + 64 = 320 bytes
  // (index 80 in float array).
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[80], 68.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Standard4D) {
  // 4D matrix of shape 2x3x8x8, element type float (4 bytes).
  // Layout has minor_to_major={3, 2, 1, 0} (standard row-major), and tiling
  // with tile dimensions 4x4.
  const int64_t D0 = 2;
  const int64_t D1 = 3;
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, D1, H, W}, {3, 2, 1, 0},
      {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * D1 * H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure.
  // We have D0 * D1 = 6 batches.
  // Each batch is 8x8 tiled with 4x4.
  // Let's check d0=1, d1=1, row 0, col 4:
  // Linear index: d0*(D1*H*W) + d1*(H*W) + row*W + col
  // = 1*(3*8*8) + 1*(8*8) + 0*8 + 4
  // = 192 + 64 + 4 = 260. Value is 260.0.
  // Physically, each batch has size H * W * sizeof(float) = 64 * 4 = 256 bytes.
  // Batch index is d0 * D1 + d1 = 1 * 3 + 1 = 4.
  // Batch offset: 4 * 256 = 1024 bytes.
  // Within batch 4, row 0, col 4 is in Tile(0,1), offset 0.
  // Tile(0,1) index is 1.
  // Offset within batch: 1 * 16 * 4 = 64 bytes.
  // Total physical offset: 1024 + 64 = 1088 bytes (index 272 in float array).
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[272], 260.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, UntiledLayout) {
  // Test that untiled buffers pass through directly via std::memcpy.
  const int64_t H = 16;
  const int64_t W = 32;
  xla::Shape shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {H, W});
  // Verify layout has no tiles
  ASSERT_TRUE(shape.layout().tiles().empty());

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i * 2 + 1);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(float));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Standard1DContiguous) {
  // 1D tensor with tile (1, 128) where W % 128 == 0 (identical contiguous
  // layout).
  const int64_t W = 4096;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0}, {xla::Tile({1, 128})});

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  std::vector<uint8_t> dst_tiled(W * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<uint16_t> dst_linear(W, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Standard1DWithTilePadding) {
  // 1D tensor with tile (8, 128) where W = 300 (has horizontal and vertical
  // tile padding).
  const int64_t W = 300;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0}, {xla::Tile({8, 128})});

  int64_t tiled_elements = GetTiledBufferElements(shape);
  // Ceil(1/8)=1, Ceil(300/128)=3 => 1 * 3 * 8 * 128 = 3072 elements.
  EXPECT_EQ(tiled_elements, 3072);

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i + 5);
  }

  std::vector<uint8_t> dst_tiled(tiled_elements * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<uint16_t> dst_linear(W, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, SingleColumn2DContiguous) {
  // 2D tensor where W == tile_W (128) and H % tile_H == 0 (256 % 8 == 0).
  const int64_t H = 256;
  const int64_t W = 128;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0}, {xla::Tile({8, 128})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<uint16_t> dst_linear(num_elements, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, SpecializedRowBytes_Int8_StandardTile) {
  // S8 dtype with tile (8, 128) => row_bytes = 128.
  const int64_t H = 64;
  const int64_t W = 256;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0}, {xla::Tile({8, 128})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(int8_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout()).ok());

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()),
                           shape, shape.layout()).ok());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, SpecializedRowBytes_Int8_TransposedTile) {
  // S8 dtype with tile (128, 8) => row_bytes = 8.
  const int64_t H = 256;
  const int64_t W = 64;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0}, {xla::Tile({128, 8})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(int8_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout()).ok());

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()),
                           shape, shape.layout()).ok());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, SpecializedRowBytes_BF16_StandardAndTransposed) {
  // BF16 dtype with tile (8, 128) => row_bytes = 256.
  // BF16 dtype with tile (128, 8) => row_bytes = 16.
  const int64_t H = 64;
  const int64_t W = 256;

  // Standard (8, 128) -> row_bytes = 256
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {H, W}, {1, 0}, {xla::Tile({8, 128})});
    const int64_t num_elements = H * W;
    std::vector<uint16_t> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i * 3 + 1);
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(uint16_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }

  // Transposed (128, 8) -> row_bytes = 16
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {256, 64}, {1, 0}, {xla::Tile({128, 8})});
    const int64_t num_elements = 256 * 64;
    std::vector<uint16_t> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i * 3 + 1);
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(uint16_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(TilingUtilsTest, SpecializedRowBytes_FP32_StandardAndTransposed) {
  // F32 dtype with tile (8, 128) => row_bytes = 512.
  // F32 dtype with tile (128, 8) => row_bytes = 32.
  const int64_t H = 64;
  const int64_t W = 256;

  // Standard (8, 128) -> row_bytes = 512
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({8, 128})});
    const int64_t num_elements = H * W;
    std::vector<float> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<float>(i) * 1.5f;
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(float));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<float> dst_linear(num_elements, 0.0f);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }

  // Transposed (128, 8) -> row_bytes = 32
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {256, 64}, {1, 0}, {xla::Tile({128, 8})});
    const int64_t num_elements = 256 * 64;
    std::vector<float> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<float>(i) * 1.5f;
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(float));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<float> dst_linear(num_elements, 0.0f);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(TilingUtilsTest, SpecializedRowBytes_64ByteRow) {
  // S8 with tile (8, 64) -> row_bytes = 64
  const int64_t H = 32;
  const int64_t W = 128;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0}, {xla::Tile({8, 64})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(int8_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout()).ok());

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()),
                           shape, shape.layout()).ok());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, SpecializedRowBytes_MultiBatch_AllTypes) {
  // Multi-batch 3D tensor: [4, 64, 256] with tile (8, 128)
  const int64_t B = 4;
  const int64_t H = 64;
  const int64_t W = 256;
  const int64_t total_elements = B * H * W;

  // 1. S8 (row_bytes = 128)
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::S8, {B, H, W}, {2, 1, 0}, {xla::Tile({8, 128})});
    std::vector<int8_t> src_linear(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      src_linear[i] = static_cast<int8_t>(i % 127);
    }
    std::vector<uint8_t> dst_tiled(total_elements * sizeof(int8_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<int8_t> dst_linear(total_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    EXPECT_EQ(dst_linear, src_linear);
  }

  // 2. BF16 (row_bytes = 256)
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {B, H, W}, {2, 1, 0}, {xla::Tile({8, 128})});
    std::vector<uint16_t> src_linear(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i);
    }
    std::vector<uint8_t> dst_tiled(total_elements * sizeof(uint16_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<uint16_t> dst_linear(total_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    EXPECT_EQ(dst_linear, src_linear);
  }

  // 3. F32 (row_bytes = 512)
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {B, H, W}, {2, 1, 0}, {xla::Tile({8, 128})});
    std::vector<float> src_linear(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      src_linear[i] = static_cast<float>(i) * 0.25f;
    }
    std::vector<uint8_t> dst_tiled(total_elements * sizeof(float));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout()).ok());
    std::vector<float> dst_linear(total_elements, 0.0f);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout()).ok());
    EXPECT_EQ(dst_linear, src_linear);
  }
}

TEST(TilingUtilsTest, NestedPacked_BF16_ExactTiles2D) {
  // Shape [64, 256] with nested layout {1, 0 : T(8, 128)(2, 1)}
  const int64_t H = 64;
  const int64_t W = 256;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i * 5 + 1);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  EXPECT_EQ(total_physical_elements, num_elements);

  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  // Verify against XLA canonical LinearIndexForNestedTiling
  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
            << "Mismatch at (" << indices[0] << ", " << indices[1] << ")";
        return true;
      });

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, NestedPacked_BF16_WithPadding2D) {
  // Shape [53, 175] with nested layout {1, 0 : T(8, 128)(2, 1)} (has padding in
  // H and W)
  const int64_t H = 53;
  const int64_t W = 175;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i * 3 + 7);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  // CeilOfRatio(53, 8) = 7, CeilOfRatio(175, 128) = 2 => 7 * 2 * 1024 = 14336
  EXPECT_EQ(total_physical_elements, 7 * 2 * 1024);

  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  // Verify against XLA canonical LinearIndexForNestedTiling for all valid
  // elements
  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
            << "Mismatch at (" << indices[0] << ", " << indices[1] << ")";
        return true;
      });

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, NestedPacked_BF16_1DTensor) {
  // Shape [2304] with nested layout {0 : T(8, 128)(2, 1)}
  const int64_t W = 2304;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i + 42);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
            << "Mismatch at index " << indices[0];
        return true;
      });

  std::vector<uint16_t> dst_linear(W, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, NestedPacked_BF16_MultiBatch3D) {
  // Shape [3, 40, 300] with nested layout {2, 1, 0 : T(8, 128)(2, 1)}
  const int64_t B = 3;
  const int64_t H = 40;
  const int64_t W = 300;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {B, H, W}, {2, 1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = B * H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 65535);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx]);
        return true;
      });

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, NestedPacked_S8_PackingFactor4) {
  // Shape [32, 256] with nested layout {1, 0 : T(8, 128)(4, 1)} (FP8/INT8 4-row
  // packing)
  const int64_t H = 32;
  const int64_t W = 256;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({4, 1})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(int8_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const int8_t* tiled_s8 = reinterpret_cast<const int8_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_s8[physical_idx], src_linear[linear_idx]);
        return true;
      });

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, ThreadPool_ExecuteOneTask) {
  tpu_raiden::NumaThreadPool pool(2);
  // On empty pool, ExecuteOneTask returns false
  EXPECT_FALSE(pool.ExecuteOneTask());

  std::atomic<bool> executed{false};
  // Schedule a task
  pool.Schedule(
      [&executed]() { executed.store(true, std::memory_order_release); });

  // Call ExecuteOneTask from current thread to help drain
  for (int i = 0; i < 100 && !executed.load(std::memory_order_acquire); ++i) {
    pool.ExecuteOneTask();
    std::this_thread::yield();
  }
  EXPECT_TRUE(executed.load(std::memory_order_acquire));
}

TEST(TilingUtilsTest, ThresholdGatedTiling_SmallTensorInline) {
  // Tensor of size 1 MB (512 x 1024 bfloat16 = 1,048,576 bytes)
  // This is < 4 MB threshold, so it executes inline on calling thread.
  const int64_t H = 512;
  const int64_t W = 1024;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i & 0x7FFF);
  }

  tpu_raiden::NumaThreadPool pool(4);
  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);

  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout(), &pool)
                  .ok());

  std::vector<uint16_t> dst_linear(num_elements, 0);
  EXPECT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout(), &pool)
                  .ok());

  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, ThresholdGatedTiling_LargeTensorParallel) {
  // Tensor of size 8 MB (2048 x 2048 bfloat16 = 8,388,608 bytes)
  // This is >= 4 MB threshold, so it exercises the parallel sub-tasking path.
  const int64_t H = 2048;
  const int64_t W = 2048;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled_seq(total_physical_elements * sizeof(uint16_t),
                                     0);
  std::vector<uint8_t> dst_tiled_par(total_physical_elements * sizeof(uint16_t),
                                     0);

  // 1. Sequential baseline (pool = nullptr)
  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled_seq.data(), shape, shape.layout(), nullptr)
                  .ok());

  // 2. Parallel path with injected 4-thread pool
  tpu_raiden::NumaThreadPool pool(4);
  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled_par.data(), shape, shape.layout(), &pool)
                  .ok());

  // Bit-for-bit parity between sequential and parallel tiling
  EXPECT_EQ(dst_tiled_par, dst_tiled_seq);

  // Detile in parallel and verify full round-trip accuracy
  std::vector<uint16_t> dst_linear(num_elements, 0);
  EXPECT_TRUE(DetileBuffer(dst_tiled_par.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout(), &pool)
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, ThresholdGatedTiling_InvokedFromWorkerThread_NoDeadlock) {
  // Test invoking TileBuffer on a large 8 MB tensor FROM WITHIN a worker thread
  // of the same thread pool. This specifically tests against thread starvation
  // deadlock.
  const int64_t H = 2048;
  const int64_t W = 2048;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);

  tpu_raiden::NumaThreadPool pool(4);

  // Dispatch from inside a NumaThreadPool worker thread
  auto future = pool.Schedule([&]() -> absl::Status {
    EXPECT_TRUE(tpu_raiden::NumaThreadPool::IsCurrentThreadWorker());
    return TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                      dst_tiled.data(), shape, shape.layout(), &pool);
  });

  absl::Status status = future.get();
  EXPECT_TRUE(status.ok()) << status.ToString();

  // Verify round-trip correctness
  std::vector<uint16_t> dst_linear(num_elements, 0);
  auto detile_future = pool.Schedule([&]() -> absl::Status {
    EXPECT_TRUE(tpu_raiden::NumaThreadPool::IsCurrentThreadWorker());
    return DetileBuffer(dst_tiled.data(),
                        reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                        shape.layout(), &pool);
  });

  absl::Status detile_status = detile_future.get();
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsTest, ThresholdGatedTiling_LargeTensorWithPadding) {
  // Test large tensor with dimensions not aligned to tile size:
  // H = 2050, W = 2050 (tile is 8, 128)
  // Total size ~8.4 MB (triggers parallel path and exercises padding logic)
  const int64_t H = 2050;
  const int64_t W = 2050;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);

  tpu_raiden::NumaThreadPool pool(4);
  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout(), &pool)
                  .ok());

  std::vector<uint16_t> dst_linear(num_elements, 0);
  EXPECT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout(), &pool)
                  .ok());

  EXPECT_EQ(dst_linear, src_linear);
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

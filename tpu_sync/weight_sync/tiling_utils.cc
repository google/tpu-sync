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

#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "hwy/highway.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tpu_sync/core/numa_thread_pool.h"

namespace tpu_raiden::weight_sync {

namespace {

// Tensors below this threshold are tiled inline directly on the calling thread
// to preserve private L2 cache locality and avoid thread pool scheduling
// overhead.
constexpr int64_t kParallelizationThresholdBytes = 4 * 1024 * 1024;  // 4 MB

// Maximum sub-task chunks per tensor when parallelizing across the thread pool,
// ensuring a single large tensor does not monopolize the pool during concurrent
// layer arrivals.
constexpr int64_t kMaxChunksPerTensor = 4;

constexpr int64_t kMaxNumThreads = 4;

tpu_raiden::NumaThreadPool* GetThreadPool() {
  static absl::NoDestructor<tpu_raiden::NumaThreadPool> global_pool(
      kMaxNumThreads);
  return global_pool.get();
}

bool IsStandardRowMajorTiled(const xla::Shape& shape,
                             const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  if (R < 1) return false;

  for (int i = 0; i < R; ++i) {
    if (layout.minor_to_major(i) != R - 1 - i) {
      return false;
    }
  }

  if (layout.tiles().size() == 1) {
    return layout.tiles(0).dimensions().size() == 2;
  }

  if (layout.tiles().size() == 2) {
    const auto& t0 = layout.tiles(0);
    const auto& t1 = layout.tiles(1);
    if (t0.dimensions().size() != 2 || t1.dimensions().size() != 2) {
      return false;
    }
    int64_t tile_H = t0.dimension(0);
    int64_t P = t1.dimension(0);
    int64_t sub_W = t1.dimension(1);
    return (sub_W == 1 && (P == 2 || P == 4) && tile_H % P == 0);
  }

  return false;
}

template <typename F>
decltype(auto) DispatchByPackingFactor(int64_t packing_factor, F&& f) {
  switch (packing_factor) {
    case 1:
      return f(std::integral_constant<int64_t, 1>{});
    case 2:
      return f(std::integral_constant<int64_t, 2>{});
    case 4:
      return f(std::integral_constant<int64_t, 4>{});
    default:
      return f(std::integral_constant<int64_t, 1>{});
  }
}

// Dispatches row copy operations to compile-time specialized fixed-width
// vector copy loops for common TPU tile row sizes (FP8, BF16, FP32 with tile_W
// 128 or 8), falling back to generic std::memcpy for arbitrary dimensions.
template <typename F>
decltype(auto) DispatchByRowBytes(int64_t row_bytes, F&& f) {
  switch (row_bytes) {
    case 8:
      return f(std::integral_constant<size_t, 8>{});
    case 16:
      return f(std::integral_constant<size_t, 16>{});
    case 32:
      return f(std::integral_constant<size_t, 32>{});
    case 64:
      return f(std::integral_constant<size_t, 64>{});
    case 128:
      return f(std::integral_constant<size_t, 128>{});
    case 256:
      return f(std::integral_constant<size_t, 256>{});
    case 512:
      return f(std::integral_constant<size_t, 512>{});
    default:
      return f(std::integral_constant<size_t, 0>{});
  }
}

namespace hn = hwy::HWY_NAMESPACE;

// Copies valid_bytes from src to dst and zero-fills the rest of total_row_bytes
// using portable Highway SIMD vector operations.
inline void CopyRowWithPaddingHighway(const uint8_t* src, uint8_t* dst,
                                      size_t valid_bytes,
                                      size_t total_row_bytes) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);
  size_t offset = 0;

  for (; offset + N <= valid_bytes; offset += N) {
    const auto v = hn::LoadU(d, src + offset);
    hn::StoreU(v, d, dst + offset);
  }

  if (offset < valid_bytes) {
    const size_t rem = valid_bytes - offset;
    const auto mask = hn::FirstN(d, rem);
    const auto v = hn::LoadU(d, src + offset);
    const auto zeros = hn::Zero(d);
    const auto blended = hn::IfThenElse(mask, v, zeros);
    hn::StoreU(blended, d, dst + offset);
    offset += N;
  }

  const auto zeros = hn::Zero(d);
  for (; offset + N <= total_row_bytes; offset += N) {
    hn::StoreU(zeros, d, dst + offset);
  }

  if (offset < total_row_bytes) {
    const auto mask = hn::FirstN(d, total_row_bytes - offset);
    hn::BlendedStore(zeros, mask, d, dst + offset);
  }
}

// Detiles valid_bytes from src to dst using Highway SIMD vectors.
inline void DetileRowWithPaddingHighway(const uint8_t* src, uint8_t* dst,
                                        size_t valid_bytes) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);
  size_t offset = 0;

  for (; offset + N <= valid_bytes; offset += N) {
    const auto v = hn::LoadU(d, src + offset);
    hn::StoreU(v, d, dst + offset);
  }

  if (offset < valid_bytes) {
    const auto mask = hn::FirstN(d, valid_bytes - offset);
    const auto v = hn::LoadU(d, src + offset);
    hn::BlendedStore(v, mask, d, dst + offset);
  }
}

// Zeroes total_row_bytes using Highway SIMD vectors.
inline void ZeroRowHighway(uint8_t* dst, size_t total_row_bytes) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);
  const auto zeros = hn::Zero(d);
  size_t offset = 0;
  for (; offset + N <= total_row_bytes; offset += N) {
    hn::StoreU(zeros, d, dst + offset);
  }
  if (offset < total_row_bytes) {
    const auto mask = hn::FirstN(d, total_row_bytes - offset);
    hn::BlendedStore(zeros, mask, d, dst + offset);
  }
}

// Copies a single tile from the source batch pointer to the destination tile
// pointer when no padding is needed. Template-specialized on compile-time
// packing factor and row byte width to enable inlined SIMD load/store vector
// instructions.
template <int64_t kPackingFactor, size_t kRowBytes>
void CopyTilePackedNoPadding(const uint8_t* src_batch_ptr,
                             uint8_t* dst_tile_ptr, int64_t tile_row,
                             int64_t tile_col, int64_t tile_H, int64_t tile_W,
                             int64_t W, int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  if constexpr (kPackingFactor == 1) {
    int64_t row_stride = (kRowBytes > 0) ? kRowBytes : (tile_W * itemsize);
    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      const uint8_t* src_row_ptr =
          src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
      uint8_t* dst_row_ptr = dst_tile_ptr + r * row_stride;

      if constexpr (kRowBytes > 0) {
        std::memcpy(dst_row_ptr, src_row_ptr, kRowBytes);
      } else {
        std::memcpy(dst_row_ptr, src_row_ptr, row_stride);
      }
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);
    const hn::ScalableTag<uint16_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      const uint16_t* src0 = reinterpret_cast<const uint16_t*>(
          src_batch_ptr + (r0 * W + logical_col_start) * sizeof(uint16_t));
      const uint16_t* src1 = reinterpret_cast<const uint16_t*>(
          src_batch_ptr + (r1 * W + logical_col_start) * sizeof(uint16_t));
      uint16_t* dst_group =
          reinterpret_cast<uint16_t*>(dst_tile_ptr + g * group_stride_bytes);

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        auto v0 = hn::LoadU(d, src0 + c);
        auto v1 = hn::LoadU(d, src1 + c);
        hn::StoreInterleaved2(v0, v1, d, dst_group + c * 2);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst_group[2 * c + 0] = src0[c];
        dst_group[2 * c + 1] = src1[c];
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      const uint8_t* src0 = src_batch_ptr + (r0 * W + logical_col_start);
      const uint8_t* src1 = src_batch_ptr + ((r0 + 1) * W + logical_col_start);
      const uint8_t* src2 = src_batch_ptr + ((r0 + 2) * W + logical_col_start);
      const uint8_t* src3 = src_batch_ptr + ((r0 + 3) * W + logical_col_start);
      uint8_t* dst_group = dst_tile_ptr + g * group_stride_bytes;

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        auto v0 = hn::LoadU(d, src0 + c);
        auto v1 = hn::LoadU(d, src1 + c);
        auto v2 = hn::LoadU(d, src2 + c);
        auto v3 = hn::LoadU(d, src3 + c);
        hn::StoreInterleaved4(v0, v1, v2, v3, d, dst_group + c * 4);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst_group[4 * c + 0] = src0[c];
        dst_group[4 * c + 1] = src1[c];
        dst_group[4 * c + 2] = src2[c];
        dst_group[4 * c + 3] = src3[c];
      }
    }
  }
}

// Copies a single tile from the source batch pointer to the destination tile
// pointer, handling padding if the tile is partially or fully out of bounds.
// Out-of-bounds elements in the destination tile are zero-initialized.
template <int64_t kPackingFactor>
void CopyTilePackedWithPadding(const uint8_t* src_batch_ptr,
                               uint8_t* dst_tile_ptr, int64_t tile_row,
                               int64_t tile_col, int64_t tile_H, int64_t tile_W,
                               int64_t H, int64_t W, int64_t itemsize,
                               int64_t tile_size_bytes) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t valid_elements = std::min(tile_W, W - logical_col_start);
  if (valid_elements <= 0) {
    ZeroRowHighway(dst_tile_ptr, static_cast<size_t>(tile_size_bytes));
    return;
  }

  if constexpr (kPackingFactor == 1) {
    size_t valid_bytes = static_cast<size_t>(valid_elements * itemsize);
    size_t total_row_bytes = static_cast<size_t>(tile_W * itemsize);

    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      uint8_t* dst_row_ptr = dst_tile_ptr + r * total_row_bytes;
      if (logical_row >= H) {
        ZeroRowHighway(dst_row_ptr, total_row_bytes);
        continue;
      }
      const uint8_t* src_row_ptr =
          src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;

      CopyRowWithPaddingHighway(src_row_ptr, dst_row_ptr, valid_bytes,
                                total_row_bytes);
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      uint16_t* dst_group =
          reinterpret_cast<uint16_t*>(dst_tile_ptr + g * group_stride_bytes);

      if (r0 >= H && r1 >= H) {
        ZeroRowHighway(reinterpret_cast<uint8_t*>(dst_group),
                       static_cast<size_t>(group_stride_bytes));
        continue;
      }

      const uint16_t* src0 =
          (r0 < H) ? reinterpret_cast<const uint16_t*>(
                         src_batch_ptr +
                         (r0 * W + logical_col_start) * sizeof(uint16_t))
                   : nullptr;
      const uint16_t* src1 =
          (r1 < H) ? reinterpret_cast<const uint16_t*>(
                         src_batch_ptr +
                         (r1 * W + logical_col_start) * sizeof(uint16_t))
                   : nullptr;

      for (int64_t c = 0; c < tile_W; ++c) {
        if (c < valid_elements) {
          dst_group[2 * c + 0] = src0 ? src0[c] : 0;
          dst_group[2 * c + 1] = src1 ? src1[c] : 0;
        } else {
          dst_group[2 * c + 0] = 0;
          dst_group[2 * c + 1] = 0;
        }
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      uint8_t* dst_group = dst_tile_ptr + g * group_stride_bytes;

      if (r0 >= H) {
        ZeroRowHighway(dst_group, static_cast<size_t>(group_stride_bytes));
        continue;
      }

      const uint8_t* src[4];
      for (int p = 0; p < 4; ++p) {
        src[p] = (r0 + p < H)
                     ? (src_batch_ptr + ((r0 + p) * W + logical_col_start))
                     : nullptr;
      }

      for (int64_t c = 0; c < tile_W; ++c) {
        if (c < valid_elements) {
          for (int p = 0; p < 4; ++p) {
            dst_group[4 * c + p] = src[p] ? src[p][c] : 0;
          }
        } else {
          for (int p = 0; p < 4; ++p) {
            dst_group[4 * c + p] = 0;
          }
        }
      }
    }
  }
}

template <int64_t kPackingFactor, size_t kRowBytes>
void DetileSingleTilePackedNoPadding(const uint8_t* src_tile_ptr,
                                     uint8_t* dst_batch_ptr, int64_t tile_row,
                                     int64_t tile_col, int64_t tile_H,
                                     int64_t tile_W, int64_t W,
                                     int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  if constexpr (kPackingFactor == 1) {
    int64_t row_stride = (kRowBytes > 0) ? kRowBytes : (tile_W * itemsize);
    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      uint8_t* dst_row_ptr =
          dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
      const uint8_t* src_row_ptr = src_tile_ptr + r * row_stride;

      if constexpr (kRowBytes > 0) {
        std::memcpy(dst_row_ptr, src_row_ptr, kRowBytes);
      } else {
        std::memcpy(dst_row_ptr, src_row_ptr, row_stride);
      }
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);
    const hn::ScalableTag<uint16_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      uint16_t* dst0 = reinterpret_cast<uint16_t*>(
          dst_batch_ptr + (r0 * W + logical_col_start) * sizeof(uint16_t));
      uint16_t* dst1 = reinterpret_cast<uint16_t*>(
          dst_batch_ptr + (r1 * W + logical_col_start) * sizeof(uint16_t));
      const uint16_t* src_group = reinterpret_cast<const uint16_t*>(
          src_tile_ptr + g * group_stride_bytes);

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        hn::Vec<decltype(d)> v0, v1;
        hn::LoadInterleaved2(d, src_group + c * 2, v0, v1);
        hn::StoreU(v0, d, dst0 + c);
        hn::StoreU(v1, d, dst1 + c);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst0[c] = src_group[2 * c + 0];
        dst1[c] = src_group[2 * c + 1];
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      uint8_t* dst0 = dst_batch_ptr + (r0 * W + logical_col_start);
      uint8_t* dst1 = dst_batch_ptr + ((r0 + 1) * W + logical_col_start);
      uint8_t* dst2 = dst_batch_ptr + ((r0 + 2) * W + logical_col_start);
      uint8_t* dst3 = dst_batch_ptr + ((r0 + 3) * W + logical_col_start);
      const uint8_t* src_group = src_tile_ptr + g * group_stride_bytes;

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        hn::Vec<decltype(d)> v0, v1, v2, v3;
        hn::LoadInterleaved4(d, src_group + c * 4, v0, v1, v2, v3);
        hn::StoreU(v0, d, dst0 + c);
        hn::StoreU(v1, d, dst1 + c);
        hn::StoreU(v2, d, dst2 + c);
        hn::StoreU(v3, d, dst3 + c);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst0[c] = src_group[4 * c + 0];
        dst1[c] = src_group[4 * c + 1];
        dst2[c] = src_group[4 * c + 2];
        dst3[c] = src_group[4 * c + 3];
      }
    }
  }
}

template <int64_t kPackingFactor>
void DetileSingleTilePackedWithPadding(const uint8_t* src_tile_ptr,
                                       uint8_t* dst_batch_ptr, int64_t tile_row,
                                       int64_t tile_col, int64_t tile_H,
                                       int64_t tile_W, int64_t H, int64_t W,
                                       int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t valid_elements = std::min(tile_W, W - logical_col_start);
  if (valid_elements <= 0) {
    return;
  }

  if constexpr (kPackingFactor == 1) {
    size_t valid_bytes = static_cast<size_t>(valid_elements * itemsize);
    size_t total_row_bytes = static_cast<size_t>(tile_W * itemsize);

    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      if (logical_row >= H) {
        continue;
      }
      uint8_t* dst_row_ptr =
          dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
      const uint8_t* src_row_ptr = src_tile_ptr + r * total_row_bytes;

      DetileRowWithPaddingHighway(src_row_ptr, dst_row_ptr, valid_bytes);
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      if (r0 >= H && r1 >= H) {
        continue;
      }
      uint16_t* dst0 = (r0 < H)
                           ? reinterpret_cast<uint16_t*>(
                                 dst_batch_ptr + (r0 * W + logical_col_start) *
                                                     sizeof(uint16_t))
                           : nullptr;
      uint16_t* dst1 = (r1 < H)
                           ? reinterpret_cast<uint16_t*>(
                                 dst_batch_ptr + (r1 * W + logical_col_start) *
                                                     sizeof(uint16_t))
                           : nullptr;
      const uint16_t* src_group = reinterpret_cast<const uint16_t*>(
          src_tile_ptr + g * group_stride_bytes);

      for (int64_t c = 0; c < valid_elements; ++c) {
        if (dst0) dst0[c] = src_group[2 * c + 0];
        if (dst1) dst1[c] = src_group[2 * c + 1];
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      if (r0 >= H) {
        continue;
      }
      uint8_t* dst[4];
      for (int p = 0; p < 4; ++p) {
        dst[p] = (r0 + p < H)
                     ? (dst_batch_ptr + ((r0 + p) * W + logical_col_start))
                     : nullptr;
      }
      const uint8_t* src_group = src_tile_ptr + g * group_stride_bytes;

      for (int64_t c = 0; c < valid_elements; ++c) {
        for (int p = 0; p < 4; ++p) {
          if (dst[p]) dst[p][c] = src_group[4 * c + p];
        }
      }
    }
  }
}

// Tiles a buffer using an optimized path for standard row-major layouts.
// It avoids global zero-initialization of the destination buffer to prevent
// CPU cache pollution, instead zeroing padding elements locally per tile.
absl::Status TileBufferNDOptimized(const uint8_t* src_linear,
                                   uint8_t* dst_tiled, const xla::Shape& shape,
                                   const xla::Layout& layout,
                                   tpu_raiden::NumaThreadPool* pool = nullptr) {
  const int R = shape.dimensions().size();
  int64_t H = (R == 1) ? 1 : shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);
  int64_t packing_factor = 1;
  if (layout.tiles().size() >= 2) {
    packing_factor = layout.tiles(1).dimension(0);
  }

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;

  bool has_padding = (H % tile_H != 0) || (W % tile_W != 0);

  // Fast-path: When the tiled layout is byte-for-byte identical to the linear
  // layout (e.g. single column of tiles W == tile_W with no vertical padding
  // H % tile_H == 0, or 1D tensor with tile_H == 1 and no padding) AND
  // packing_factor == 1.
  if (packing_factor == 1 && !has_padding &&
      (W == tile_W || (H == 1 && tile_H == 1))) {
    std::memcpy(dst_tiled, src_linear, batch_size * matrix_size_bytes);
    return absl::OkStatus();
  }

  // Fast-path for 1D tensors (H == 1) with packing_factor == 1.
  if (packing_factor == 1 && H == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
      uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t logical_col_start = tile_col * tile_W;
        int64_t valid_elements = std::min(tile_W, W - logical_col_start);
        uint8_t* dst_tile_ptr = dst_batch_ptr + tile_col * tile_size_bytes;
        if (valid_elements > 0) {
          std::memcpy(dst_tile_ptr,
                      src_batch_ptr + logical_col_start * itemsize,
                      valid_elements * itemsize);
          if (valid_elements < tile_W) {
            std::memset(dst_tile_ptr + valid_elements * itemsize, 0,
                        (tile_W - valid_elements) * itemsize);
          }
        } else {
          std::memset(dst_tile_ptr, 0, tile_W * itemsize);
        }
        if (tile_H > 1) {
          std::memset(dst_tile_ptr + tile_W * itemsize, 0,
                      (tile_H - 1) * tile_W * itemsize);
        }
      }
    }
    return absl::OkStatus();
  }

  int64_t total_tasks = batch_size * num_tiles_0;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t row_bytes = (packing_factor == 1) ? (tile_W * itemsize) : 0;

  DispatchByPackingFactor(packing_factor, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    if (total_bytes < kParallelizationThresholdBytes || total_tasks <= 1) {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        if (!has_padding) {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                uint8_t* dst_tile_ptr =
                    dst_batch_ptr + tile_index * tile_size_bytes;
                CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              }
            }
          }
        } else {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              bool is_row_interior = (tile_row * tile_H + tile_H <= H);
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                uint8_t* dst_tile_ptr =
                    dst_batch_ptr + tile_index * tile_size_bytes;
                bool is_col_interior = (tile_col * tile_W + tile_W <= W);
                if (is_row_interior && is_col_interior) {
                  CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                      src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                      tile_W, W, itemsize);
                } else {
                  CopyTilePackedWithPadding<kPackingFactor>(
                      src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                      tile_W, H, W, itemsize, tile_size_bytes);
                }
              }
            }
          }
        }
      });
    } else {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        auto run_task = [&](int64_t b, int64_t tile_row) {
          const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;

          if (!has_padding) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              uint8_t* dst_tile_ptr =
                  dst_batch_ptr + tile_index * tile_size_bytes;
              CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                  src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                  tile_W, W, itemsize);
            }
          } else {
            bool is_row_interior = (tile_row * tile_H + tile_H <= H);
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              uint8_t* dst_tile_ptr =
                  dst_batch_ptr + tile_index * tile_size_bytes;
              bool is_col_interior = (tile_col * tile_W + tile_W <= W);
              if (is_row_interior && is_col_interior) {
                CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              } else {
                CopyTilePackedWithPadding<kPackingFactor>(
                    src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                    tile_W, H, W, itemsize, tile_size_bytes);
              }
            }
          }
        };

        tpu_raiden::NumaThreadPool* target_pool =
            (pool != nullptr) ? pool : GetThreadPool();
        int64_t pool_threads =
            (target_pool != nullptr)
                ? static_cast<int64_t>(target_pool->num_threads())
                : 0;
        int64_t max_threads =
            std::min<int64_t>(kMaxChunksPerTensor, pool_threads);

        if (max_threads > 1 && total_tasks >= max_threads) {
          int64_t chunk_size = (total_tasks + max_threads - 1) / max_threads;
          int64_t num_chunks = (total_tasks + chunk_size - 1) / chunk_size;
          std::atomic<int64_t> remaining_tasks(num_chunks);

          // Schedule chunks 1 .. num_chunks - 1 to the pool
          for (int64_t t = 1; t < num_chunks; ++t) {
            int64_t begin = t * chunk_size;
            int64_t end = std::min(begin + chunk_size, total_tasks);
            target_pool->Schedule(
                [&run_task, &remaining_tasks, begin, end, num_tiles_0]() {
                  for (int64_t i = begin; i < end; ++i) {
                    int64_t b = i / num_tiles_0;
                    int64_t tile_row = i % num_tiles_0;
                    run_task(b, tile_row);
                  }
                  remaining_tasks.fetch_sub(1, std::memory_order_release);
                });
          }

          // Execute chunk 0 directly on the calling thread
          int64_t begin0 = 0;
          int64_t end0 = std::min(chunk_size, total_tasks);
          for (int64_t i = begin0; i < end0; ++i) {
            int64_t b = i / num_tiles_0;
            int64_t tile_row = i % num_tiles_0;
            run_task(b, tile_row);
          }
          remaining_tasks.fetch_sub(1, std::memory_order_release);

          // While waiting for other chunks to complete, help execute pending
          // tasks from the pool
          while (remaining_tasks.load(std::memory_order_acquire) > 0) {
            if (target_pool != nullptr && target_pool->ExecuteOneTask()) {
              continue;
            }
            sched_yield();
          }
        } else {
          for (int64_t i = 0; i < total_tasks; ++i) {
            int64_t b = i / num_tiles_0;
            int64_t tile_row = i % num_tiles_0;
            run_task(b, tile_row);
          }
        }
      });
    }
  });

  return absl::OkStatus();
}

absl::Status DetileBufferNDOptimized(
    const uint8_t* src_tiled, uint8_t* dst_linear, const xla::Shape& shape,
    const xla::Layout& layout, tpu_raiden::NumaThreadPool* pool = nullptr) {
  const int R = shape.dimensions().size();
  int64_t H = (R == 1) ? 1 : shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);
  int64_t packing_factor = 1;
  if (layout.tiles().size() >= 2) {
    packing_factor = layout.tiles(1).dimension(0);
  }

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;

  bool has_padding = (H % tile_H != 0) || (W % tile_W != 0);

  // Fast-path: When the tiled layout is byte-for-byte identical to the linear
  // layout (e.g. single column of tiles W == tile_W with no vertical padding
  // H % tile_H == 0, or 1D tensor with tile_H == 1 and no padding) AND
  // packing_factor == 1.
  if (packing_factor == 1 && !has_padding &&
      (W == tile_W || (H == 1 && tile_H == 1))) {
    std::memcpy(dst_linear, src_tiled, batch_size * matrix_size_bytes);
    return absl::OkStatus();
  }

  // Fast-path for 1D tensors (H == 1) with packing_factor == 1.
  if (packing_factor == 1 && H == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const uint8_t* src_batch_ptr = src_tiled + b * tiled_matrix_size_bytes;
      uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t logical_col_start = tile_col * tile_W;
        int64_t valid_elements = std::min(tile_W, W - logical_col_start);
        if (valid_elements <= 0) {
          break;
        }
        std::memcpy(dst_batch_ptr + logical_col_start * itemsize,
                    src_batch_ptr + tile_col * tile_size_bytes,
                    valid_elements * itemsize);
      }
    }
    return absl::OkStatus();
  }

  int64_t total_tasks = batch_size * num_tiles_0;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t row_bytes = (packing_factor == 1) ? (tile_W * itemsize) : 0;

  DispatchByPackingFactor(packing_factor, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    if (total_bytes < kParallelizationThresholdBytes || total_tasks <= 1) {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        if (!has_padding) {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr =
                src_tiled + b * tiled_matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                const uint8_t* src_tile_ptr =
                    src_batch_ptr + tile_index * tile_size_bytes;
                DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              }
            }
          }
        } else {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr =
                src_tiled + b * tiled_matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              bool is_row_interior = (tile_row * tile_H + tile_H <= H);
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                const uint8_t* src_tile_ptr =
                    src_batch_ptr + tile_index * tile_size_bytes;
                bool is_col_interior = (tile_col * tile_W + tile_W <= W);
                if (is_row_interior && is_col_interior) {
                  DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                      src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                      tile_W, W, itemsize);
                } else {
                  DetileSingleTilePackedWithPadding<kPackingFactor>(
                      src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                      tile_W, H, W, itemsize);
                }
              }
            }
          }
        }
      });
    } else {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        auto run_detile_task = [&](int64_t b, int64_t tile_row) {
          const uint8_t* src_batch_ptr =
              src_tiled + b * tiled_matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;

          if (!has_padding) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              const uint8_t* src_tile_ptr =
                  src_batch_ptr + tile_index * tile_size_bytes;
              DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                  src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                  tile_W, W, itemsize);
            }
          } else {
            bool is_row_interior = (tile_row * tile_H + tile_H <= H);
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              const uint8_t* src_tile_ptr =
                  src_batch_ptr + tile_index * tile_size_bytes;
              bool is_col_interior = (tile_col * tile_W + tile_W <= W);
              if (is_row_interior && is_col_interior) {
                DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              } else {
                DetileSingleTilePackedWithPadding<kPackingFactor>(
                    src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                    tile_W, H, W, itemsize);
              }
            }
          }
        };

        tpu_raiden::NumaThreadPool* target_pool =
            (pool != nullptr) ? pool : GetThreadPool();
        int64_t pool_threads =
            (target_pool != nullptr)
                ? static_cast<int64_t>(target_pool->num_threads())
                : 0;
        int64_t max_threads =
            std::min<int64_t>(kMaxChunksPerTensor, pool_threads);

        if (max_threads > 1 && total_tasks >= max_threads) {
          int64_t chunk_size = (total_tasks + max_threads - 1) / max_threads;
          int64_t num_chunks = (total_tasks + chunk_size - 1) / chunk_size;
          std::atomic<int64_t> remaining_tasks(num_chunks);

          // Schedule chunks 1 .. num_chunks - 1 to the pool
          for (int64_t t = 1; t < num_chunks; ++t) {
            int64_t begin = t * chunk_size;
            int64_t end = std::min(begin + chunk_size, total_tasks);
            target_pool->Schedule([&run_detile_task, &remaining_tasks, begin,
                                   end, num_tiles_0]() {
              for (int64_t i = begin; i < end; ++i) {
                int64_t b = i / num_tiles_0;
                int64_t tile_row = i % num_tiles_0;
                run_detile_task(b, tile_row);
              }
              remaining_tasks.fetch_sub(1, std::memory_order_release);
            });
          }

          // Execute chunk 0 directly on the calling thread
          int64_t begin0 = 0;
          int64_t end0 = std::min(chunk_size, total_tasks);
          for (int64_t i = begin0; i < end0; ++i) {
            int64_t b = i / num_tiles_0;
            int64_t tile_row = i % num_tiles_0;
            run_detile_task(b, tile_row);
          }
          remaining_tasks.fetch_sub(1, std::memory_order_release);

          // While waiting for other chunks to complete, help execute pending
          // tasks from the pool
          while (remaining_tasks.load(std::memory_order_acquire) > 0) {
            if (target_pool != nullptr && target_pool->ExecuteOneTask()) {
              continue;
            }
            sched_yield();
          }
        } else {
          for (int64_t i = 0; i < total_tasks; ++i) {
            int64_t b = i / num_tiles_0;
            int64_t tile_row = i % num_tiles_0;
            run_detile_task(b, tile_row);
          }
        }
      });
    }
  });

  return absl::OkStatus();
}

}  // namespace

int64_t GetTiledBufferElements(const xla::Shape& shape) {
  const int num_dims = shape.dimensions().size();
  if (num_dims == 0) {
    return 1;
  }

  std::vector<int64_t> current_shape;
  current_shape.reserve(std::max(num_dims, 2));
  if (num_dims == 1) {
    current_shape.push_back(1);
    current_shape.push_back(shape.dimensions(0));
  } else {
    for (int64_t i = num_dims - 1; i >= 0; --i) {
      int64_t logical_dim = shape.layout().minor_to_major(i);
      current_shape.push_back(shape.dimensions(logical_dim));
    }
  }

  for (const xla::Tile& tile : shape.layout().tiles()) {
    const int64_t tile_rank = tile.dimensions().size();
    if (tile_rank > current_shape.size()) {
      int64_t pad_size = tile_rank - current_shape.size();
      current_shape.insert(current_shape.begin(), pad_size, 1);
    }

    const int64_t suffix_start = current_shape.size() - tile_rank;
    std::vector<int64_t> next_shape;
    next_shape.reserve(current_shape.size() + tile_rank);

    for (int i = 0; i < suffix_start; ++i) {
      next_shape.push_back(current_shape[i]);
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t t = tile.dimension(i);
      next_shape.push_back(xla::CeilOfRatio(d, t));
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t t = tile.dimension(i);
      next_shape.push_back(t);
    }

    current_shape = std::move(next_shape);
  }

  int64_t total_elements = 1;
  for (int64_t dim_size : current_shape) {
    total_elements *= dim_size;
  }
  return total_elements;
}

absl::Status DetileBuffer(const uint8_t* src_tiled, uint8_t* dst_linear,
                          const xla::Shape& shape, const xla::Layout& layout,
                          tpu_raiden::NumaThreadPool* pool) {
  if (layout.tiles().empty()) {
    std::memcpy(dst_linear, src_tiled, xla::ShapeUtil::ByteSizeOf(shape));
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return DetileBufferNDOptimized(src_tiled, dst_linear, shape, layout, pool);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices) *
            itemsize;
        std::memcpy(dst_linear + linear_offset, src_tiled + physical_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

absl::Status TileBuffer(const uint8_t* src_linear, uint8_t* dst_tiled,
                        const xla::Shape& shape, const xla::Layout& layout,
                        tpu_raiden::NumaThreadPool* pool) {
  if (layout.tiles().empty()) {
    std::memcpy(dst_tiled, src_linear, xla::ShapeUtil::ByteSizeOf(shape));
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return TileBufferNDOptimized(src_linear, dst_tiled, shape, layout, pool);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  int64_t logical_elements = 1;
  for (int64_t dim : shape.dimensions()) {
    logical_elements *= dim;
  }
  if (total_physical_elements > logical_elements) {
    std::memset(dst_tiled, 0, total_physical_elements * itemsize);
  }

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices) *
            itemsize;
        std::memcpy(dst_tiled + physical_offset, src_linear + linear_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

}  // namespace tpu_raiden::weight_sync

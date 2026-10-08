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

#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

int64_t Product(absl::Span<const int64_t> dims) {
  int64_t prod = 1;
  for (int64_t d : dims) {
    prod *= d;
  }
  return prod;
}

std::vector<int64_t> ComputeStrides(absl::Span<const int64_t> sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  int64_t current = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = current;
    current *= sizes[i];
  }
  return strides;
}

std::vector<StridedCopyChunk> FoldInto2dStridedChunks(
    const std::vector<StridedCopyChunk>& raw_chunks) {
  if (raw_chunks.size() <= 1) return raw_chunks;
  std::vector<StridedCopyChunk> folded;
  folded.reserve(raw_chunks.size());

  size_t i = 0;
  while (i < raw_chunks.size()) {
    const StridedCopyChunk& base = raw_chunks[i];
    if (base.count > 1 || i + 1 >= raw_chunks.size()) {
      folded.push_back(base);
      ++i;
      continue;
    }
    const StridedCopyChunk& next = raw_chunks[i + 1];
    if (next.count > 1 || next.size_bytes != base.size_bytes) {
      folded.push_back(base);
      ++i;
      continue;
    }
    const int64_t src_stride = next.src_offset_bytes - base.src_offset_bytes;
    const int64_t dst_stride = next.dst_offset_bytes - base.dst_offset_bytes;
    if (src_stride <= 0 || dst_stride <= 0) {
      folded.push_back(base);
      ++i;
      continue;
    }
    int64_t count = 2;
    size_t j = i + 2;
    while (j < raw_chunks.size()) {
      const StridedCopyChunk& cand = raw_chunks[j];
      if (cand.count == 1 && cand.size_bytes == base.size_bytes &&
          cand.src_offset_bytes - raw_chunks[j - 1].src_offset_bytes ==
              src_stride &&
          cand.dst_offset_bytes - raw_chunks[j - 1].dst_offset_bytes ==
              dst_stride) {
        ++count;
        ++j;
      } else {
        break;
      }
    }
    if (src_stride == base.size_bytes && dst_stride == base.size_bytes) {
      folded.push_back(StridedCopyChunk{
          .src_offset_bytes = base.src_offset_bytes,
          .dst_offset_bytes = base.dst_offset_bytes,
          .size_bytes = base.size_bytes * count,
          .src_stride_bytes = 0,
          .dst_stride_bytes = 0,
          .count = 1,
      });
    } else {
      folded.push_back(StridedCopyChunk{
          .src_offset_bytes = base.src_offset_bytes,
          .dst_offset_bytes = base.dst_offset_bytes,
          .size_bytes = base.size_bytes,
          .src_stride_bytes = src_stride,
          .dst_stride_bytes = dst_stride,
          .count = count,
      });
    }
    i = j;
  }
  return folded;
}

struct PlanSignatureKey {
  std::vector<int64_t> src_global_shape;
  std::vector<int64_t> src_mesh_shape;
  std::vector<int64_t> src_layout;
  std::vector<int32_t> src_phys_to_logical;
  std::vector<int64_t> dst_global_shape;
  std::vector<int64_t> dst_mesh_shape;
  std::vector<int64_t> dst_layout;
  std::vector<int32_t> dst_phys_to_logical;
  int64_t itemsize = 0;
  bool skip_tiling = false;

  bool operator==(const PlanSignatureKey& o) const {
    return std::tie(src_global_shape, src_mesh_shape, src_layout,
                    src_phys_to_logical, dst_global_shape, dst_mesh_shape,
                    dst_layout, dst_phys_to_logical, itemsize, skip_tiling) ==
           std::tie(o.src_global_shape, o.src_mesh_shape, o.src_layout,
                    o.src_phys_to_logical, o.dst_global_shape, o.dst_mesh_shape,
                    o.dst_layout, o.dst_phys_to_logical, o.itemsize,
                    o.skip_tiling);
  }

  template <typename H>
  friend H AbslHashValue(H h, const PlanSignatureKey& k) {
    return H::combine(std::move(h), k.src_global_shape, k.src_mesh_shape,
                      k.src_layout, k.src_phys_to_logical, k.dst_global_shape,
                      k.dst_mesh_shape, k.dst_layout, k.dst_phys_to_logical,
                      k.itemsize, k.skip_tiling);
  }
};

// Maps each of the |num_physical_shards| shards holding |v| to the logical
// shard (row-major over `v.mesh_shape`) it holds, from the required
// `v.global_shard_indices`.
absl::StatusOr<std::vector<int32_t>> ResolvePhysicalToLogicalShardMapping(
    const VariableSpec& v, int32_t num_physical_shards,
    int32_t num_logical_slices) {
  if (v.global_shard_indices.size() !=
      static_cast<size_t>(std::max<int32_t>(0, num_physical_shards))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Variable '", v.name, "' has ", v.global_shard_indices.size(),
        " global_shard_indices for ", num_physical_shards,
        " shards; global_shard_indices is required with one entry per shard"));
  }
  std::vector<int32_t> mapping;
  mapping.reserve(v.global_shard_indices.size());
  for (int64_t g : v.global_shard_indices) {
    if (g < 0 || g >= num_logical_slices) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Variable '", v.name, "' has global shard index ", g, " outside [0, ",
          num_logical_slices, ") = [0, prod(mesh_shape))"));
    }
    mapping.push_back(static_cast<int32_t>(g));
  }
  return mapping;
}

}  // namespace

int32_t LogicalReshardPlanner::ComputeSeedSamplerCount(
    int32_t num_dst_replicas, double broadcast_host_ratio,
    int32_t trainer_hosts, int32_t sampler_hosts) {
  if (num_dst_replicas <= 1) return std::max<int32_t>(1, num_dst_replicas);
  if (broadcast_host_ratio <= 0.0) {
    return num_dst_replicas;
  }
  const int32_t h_train = std::max<int32_t>(1, trainer_hosts);
  const int32_t h_sample = std::max<int32_t>(1, sampler_hosts);
  const int32_t raw_seed = static_cast<int32_t>(std::floor(
      broadcast_host_ratio * static_cast<double>(h_train) / h_sample));
  return std::max<int32_t>(1, std::min<int32_t>(num_dst_replicas, raw_seed));
}

std::vector<int64_t> LogicalReshardPlanner::GetGlobalIndices(
    int64_t idx, absl::Span<const int64_t> shape) {
  std::vector<int64_t> coords(shape.size(), 0);
  int64_t rem = idx;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    coords[i] = rem % shape[i];
    rem /= shape[i];
  }
  return coords;
}

absl::StatusOr<std::vector<NdSlice>>
LogicalReshardPlanner::ComputeNdShardSlices(
    absl::Span<const int64_t> global_shape,
    absl::Span<const int64_t> mesh_shape, absl::Span<const int64_t> layout) {
  const int64_t num_shards = Product(mesh_shape);
  if (num_shards <= 0) {
    return absl::InvalidArgumentError("mesh_shape product must be positive");
  }
  std::vector<NdSlice> slices;
  slices.reserve(num_shards);

  const size_t rank = global_shape.size();
  std::vector<int64_t> shard_sizes(rank, 0);
  for (size_t d = 0; d < rank; ++d) {
    shard_sizes[d] = global_shape[d];
    if (d < mesh_shape.size() && mesh_shape[d] > 1) {
      if (global_shape[d] % mesh_shape[d] != 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "global_shape[", d, "]=", global_shape[d],
            " is not divisible by mesh_shape[", d, "]=", mesh_shape[d]));
      }
      shard_sizes[d] = global_shape[d] / mesh_shape[d];
    }
  }

  for (int64_t idx = 0; idx < num_shards; ++idx) {
    std::vector<int64_t> coords = GetGlobalIndices(idx, mesh_shape);
    NdSlice slice;
    slice.offsets.resize(rank, 0);
    slice.sizes = shard_sizes;
    for (size_t d = 0; d < rank && d < coords.size(); ++d) {
      slice.offsets[d] = coords[d] * shard_sizes[d];
    }
    slices.push_back(std::move(slice));
  }
  return slices;
}

std::optional<NdSlice> LogicalReshardPlanner::IntersectNdSlices(
    const NdSlice& slice_a, const NdSlice& slice_b) {
  if (slice_a.offsets.size() != slice_b.offsets.size() ||
      slice_a.sizes.size() != slice_b.sizes.size()) {
    return std::nullopt;
  }
  const size_t rank = slice_a.offsets.size();
  NdSlice inter;
  inter.offsets.resize(rank, 0);
  inter.sizes.resize(rank, 0);
  for (size_t d = 0; d < rank; ++d) {
    const int64_t start = std::max(slice_a.offsets[d], slice_b.offsets[d]);
    const int64_t end = std::min(slice_a.offsets[d] + slice_a.sizes[d],
                                 slice_b.offsets[d] + slice_b.sizes[d]);
    if (end <= start) {
      return std::nullopt;
    }
    inter.offsets[d] = start;
    inter.sizes[d] = end - start;
  }
  return inter;
}

bool IsValidLayoutPermutation(absl::Span<const int64_t> layout, size_t rank) {
  if (layout.size() != rank) {
    return false;
  }
  std::vector<bool> seen(rank, false);
  for (int64_t dim : layout) {
    if (dim < 0 || static_cast<size_t>(dim) >= rank ||
        seen[static_cast<size_t>(dim)]) {
      return false;
    }
    seen[static_cast<size_t>(dim)] = true;
  }
  return true;
}

std::vector<int64_t> NormalizeLayout(absl::Span<const int64_t> layout,
                                     size_t rank) {
  if (IsValidLayoutPermutation(layout, rank)) {
    return std::vector<int64_t>(layout.begin(), layout.end());
  }
  std::vector<int64_t> def(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    def[i] = static_cast<int64_t>(rank - 1 - i);
  }
  return def;
}

NdSlice LogicalReshardPlanner::ToPhysical(const NdSlice& slice,
                                          absl::Span<const int64_t> layout) {
  const size_t rank = slice.offsets.size();
  if (slice.sizes.size() != rank || !IsValidLayoutPermutation(layout, rank)) {
    return slice;
  }
  NdSlice phys;
  phys.offsets.reserve(layout.size());
  phys.sizes.reserve(layout.size());
  for (int i = static_cast<int>(layout.size()) - 1; i >= 0; --i) {
    const int64_t dim = layout[i];
    phys.offsets.push_back(slice.offsets[dim]);
    phys.sizes.push_back(slice.sizes[dim]);
  }
  return phys;
}

bool LogicalReshardPlanner::IsNdSliceTileAligned(
    const NdSlice& src_slice, const NdSlice& dst_slice,
    const NdSlice& intersection, absl::Span<const int64_t> src_layout,
    absl::Span<const int64_t> dst_layout) {
  if (src_slice.sizes.size() < 2) {
    return true;
  }
  const size_t rank = src_slice.sizes.size();
  const std::vector<int64_t> norm_src_layout =
      NormalizeLayout(src_layout, rank);
  const std::vector<int64_t> norm_dst_layout =
      NormalizeLayout(dst_layout, rank);
  if (norm_src_layout != norm_dst_layout) {
    return false;
  }
  const NdSlice p_src = ToPhysical(src_slice, norm_src_layout);
  const NdSlice p_dst = ToPhysical(dst_slice, norm_dst_layout);
  const NdSlice p_int = ToPhysical(intersection, norm_src_layout);
  const size_t n = p_src.sizes.size();

  const int64_t rel_src_row = p_int.offsets[n - 2] - p_src.offsets[n - 2];
  const int64_t rel_dst_row = p_int.offsets[n - 2] - p_dst.offsets[n - 2];
  if (rel_src_row % 8 != 0 || rel_dst_row % 8 != 0) {
    return false;
  }
  const int64_t int_rows = p_int.sizes[n - 2];
  if (int_rows % 8 != 0) {
    const bool touches_src_end = (rel_src_row + int_rows == p_src.sizes[n - 2]);
    const bool touches_dst_end = (rel_dst_row + int_rows == p_dst.sizes[n - 2]);
    if (!(touches_src_end && touches_dst_end)) {
      return false;
    }
  }

  const int64_t rel_src_col = p_int.offsets[n - 1] - p_src.offsets[n - 1];
  const int64_t rel_dst_col = p_int.offsets[n - 1] - p_dst.offsets[n - 1];
  if (rel_src_col % 128 != 0 || rel_dst_col % 128 != 0) {
    return false;
  }
  const int64_t int_cols = p_int.sizes[n - 1];
  if (int_cols % 128 != 0) {
    const bool touches_src_end = (rel_src_col + int_cols == p_src.sizes[n - 1]);
    const bool touches_dst_end = (rel_dst_col + int_cols == p_dst.sizes[n - 1]);
    if (!(touches_src_end && touches_dst_end)) {
      return false;
    }
  }
  return true;
}

std::vector<StridedCopyChunk> LogicalReshardPlanner::GenerateStridedCopyChunks(
    const NdSlice& src_slice, const NdSlice& dst_slice,
    const NdSlice& intersection, absl::Span<const int64_t> src_layout,
    absl::Span<const int64_t> dst_layout, int64_t itemsize) {
  if (intersection.IsEmpty()) return {};
  if (src_slice.sizes.empty()) {
    return {StridedCopyChunk{
        .src_offset_bytes = 0,
        .dst_offset_bytes = 0,
        .size_bytes = itemsize,
        .src_stride_bytes = 0,
        .dst_stride_bytes = 0,
        .count = 1,
    }};
  }

  const size_t rank = src_slice.sizes.size();
  const std::vector<int64_t> norm_src_layout =
      NormalizeLayout(src_layout, rank);
  const std::vector<int64_t> norm_dst_layout =
      NormalizeLayout(dst_layout, rank);

  if (norm_src_layout == norm_dst_layout) {
    const NdSlice p_src = ToPhysical(src_slice, norm_src_layout);
    const NdSlice p_dst = ToPhysical(dst_slice, norm_dst_layout);
    const NdSlice p_int = ToPhysical(intersection, norm_src_layout);

    const std::vector<int64_t> src_strides = ComputeStrides(p_src.sizes);
    const std::vector<int64_t> dst_strides = ComputeStrides(p_dst.sizes);
    const size_t ndim = p_int.sizes.size();

    size_t contig_dims = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 1; --i) {
      if (p_int.sizes[i] == p_src.sizes[i] &&
          p_int.sizes[i] == p_dst.sizes[i]) {
        ++contig_dims;
      } else {
        break;
      }
    }

    int64_t chunk_elements = 1;
    for (size_t i = ndim - contig_dims; i < ndim; ++i) {
      chunk_elements *= p_int.sizes[i];
    }
    const int64_t chunk_bytes = chunk_elements * itemsize;
    const size_t outer_dims = ndim - contig_dims;

    int64_t base_src_offset = 0;
    int64_t base_dst_offset = 0;
    for (size_t i = outer_dims; i < ndim; ++i) {
      base_src_offset += (p_int.offsets[i] - p_src.offsets[i]) * src_strides[i];
      base_dst_offset += (p_int.offsets[i] - p_dst.offsets[i]) * dst_strides[i];
    }

    if (outer_dims == 0) {
      return {StridedCopyChunk{
          .src_offset_bytes = base_src_offset * itemsize,
          .dst_offset_bytes = base_dst_offset * itemsize,
          .size_bytes = chunk_bytes,
          .src_stride_bytes = 0,
          .dst_stride_bytes = 0,
          .count = 1,
      }};
    }

    std::vector<StridedCopyChunk> raw_chunks;
    std::vector<int64_t> outer_coord(outer_dims, 0);
    std::function<void(size_t, int64_t, int64_t)> walk_outer =
        [&](size_t dim, int64_t cur_src, int64_t cur_dst) {
          if (dim == outer_dims) {
            const int64_t s_byte = (cur_src + base_src_offset) * itemsize;
            const int64_t d_byte = (cur_dst + base_dst_offset) * itemsize;
            if (!raw_chunks.empty() &&
                raw_chunks.back().src_offset_bytes +
                        raw_chunks.back().size_bytes ==
                    s_byte &&
                raw_chunks.back().dst_offset_bytes +
                        raw_chunks.back().size_bytes ==
                    d_byte) {
              raw_chunks.back().size_bytes += chunk_bytes;
            } else {
              raw_chunks.push_back(StridedCopyChunk{
                  .src_offset_bytes = s_byte,
                  .dst_offset_bytes = d_byte,
                  .size_bytes = chunk_bytes,
                  .src_stride_bytes = 0,
                  .dst_stride_bytes = 0,
                  .count = 1,
              });
            }
            return;
          }
          const int64_t rel_src_base = p_int.offsets[dim] - p_src.offsets[dim];
          const int64_t rel_dst_base = p_int.offsets[dim] - p_dst.offsets[dim];
          for (int64_t v = 0; v < p_int.sizes[dim]; ++v) {
            walk_outer(dim + 1, cur_src + (rel_src_base + v) * src_strides[dim],
                       cur_dst + (rel_dst_base + v) * dst_strides[dim]);
          }
        };
    walk_outer(0, 0, 0);
    return FoldInto2dStridedChunks(raw_chunks);
  }

  // Different layouts between src and dst: element-wise physical offset map
  // with contiguous and 2D-strided folding.
  const NdSlice p_src = ToPhysical(src_slice, norm_src_layout);
  const NdSlice p_dst = ToPhysical(dst_slice, norm_dst_layout);
  const std::vector<int64_t> src_phys_strides = ComputeStrides(p_src.sizes);
  const std::vector<int64_t> dst_phys_strides = ComputeStrides(p_dst.sizes);
  const size_t ndim = intersection.sizes.size();

  std::vector<int64_t> src_logical_strides(ndim, 0);
  std::vector<int64_t> dst_logical_strides(ndim, 0);
  for (size_t p_idx = 0; p_idx < ndim; ++p_idx) {
    const int64_t s_log_dim = norm_src_layout[ndim - 1 - p_idx];
    const int64_t d_log_dim = norm_dst_layout[ndim - 1 - p_idx];
    src_logical_strides[s_log_dim] = src_phys_strides[p_idx];
    dst_logical_strides[d_log_dim] = dst_phys_strides[p_idx];
  }

  std::vector<StridedCopyChunk> raw_chunks;
  std::function<void(size_t, int64_t, int64_t)> walk_logical =
      [&](size_t dim, int64_t cur_src, int64_t cur_dst) {
        if (dim == ndim) {
          const int64_t s_byte = cur_src * itemsize;
          const int64_t d_byte = cur_dst * itemsize;
          if (!raw_chunks.empty() &&
              raw_chunks.back().src_offset_bytes +
                      raw_chunks.back().size_bytes ==
                  s_byte &&
              raw_chunks.back().dst_offset_bytes +
                      raw_chunks.back().size_bytes ==
                  d_byte) {
            raw_chunks.back().size_bytes += itemsize;
          } else {
            raw_chunks.push_back(StridedCopyChunk{
                .src_offset_bytes = s_byte,
                .dst_offset_bytes = d_byte,
                .size_bytes = itemsize,
                .src_stride_bytes = 0,
                .dst_stride_bytes = 0,
                .count = 1,
            });
          }
          return;
        }
        const int64_t rel_s =
            intersection.offsets[dim] - src_slice.offsets[dim];
        const int64_t rel_d =
            intersection.offsets[dim] - dst_slice.offsets[dim];
        for (int64_t v = 0; v < intersection.sizes[dim]; ++v) {
          walk_logical(dim + 1,
                       cur_src + (rel_s + v) * src_logical_strides[dim],
                       cur_dst + (rel_d + v) * dst_logical_strides[dim]);
        }
      };
  walk_logical(0, 0, 0);
  return FoldInto2dStridedChunks(raw_chunks);
}

std::vector<StridedCopyChunk>
LogicalReshardPlanner::GenerateStridedCopyChunksTileAware(
    const NdSlice& src_slice, const NdSlice& dst_slice,
    const NdSlice& intersection, absl::Span<const int64_t> src_layout,
    absl::Span<const int64_t> dst_layout, int64_t itemsize) {
  if (intersection.IsEmpty()) return {};
  if (src_slice.sizes.size() < 2) {
    return GenerateStridedCopyChunks(src_slice, dst_slice, intersection,
                                     src_layout, dst_layout, itemsize);
  }
  if (!IsNdSliceTileAligned(src_slice, dst_slice, intersection, src_layout,
                            dst_layout)) {
    return GenerateStridedCopyChunks(src_slice, dst_slice, intersection,
                                     src_layout, dst_layout, itemsize);
  }
  const NdSlice p_src = ToPhysical(src_slice, src_layout);
  const NdSlice p_dst = ToPhysical(dst_slice, dst_layout);
  const NdSlice p_int = ToPhysical(intersection, src_layout);
  const size_t rank = p_src.sizes.size();

  constexpr int64_t kTileRow = 8;
  const int64_t w_src = p_src.sizes[rank - 1];
  const int64_t w_dst = p_dst.sizes[rank - 1];
  const int64_t w_int = p_int.sizes[rank - 1];
  const int64_t h_int = p_int.sizes[rank - 2];
  if (h_int < kTileRow || h_int % kTileRow != 0) {
    return GenerateStridedCopyChunks(src_slice, dst_slice, intersection,
                                     src_layout, dst_layout, itemsize);
  }

  const int64_t local_src_row =
      p_int.offsets[rank - 2] - p_src.offsets[rank - 2];
  const int64_t local_src_col =
      p_int.offsets[rank - 1] - p_src.offsets[rank - 1];
  const int64_t local_dst_row =
      p_int.offsets[rank - 2] - p_dst.offsets[rank - 2];
  const int64_t local_dst_col =
      p_int.offsets[rank - 1] - p_dst.offsets[rank - 1];

  const int64_t size_bytes = w_int * kTileRow * itemsize;
  const int64_t src_stride = w_src * kTileRow * itemsize;
  const int64_t dst_stride = w_dst * kTileRow * itemsize;
  const int64_t count = h_int / kTileRow;

  const int64_t src_offset =
      local_src_row * w_src * itemsize + local_src_col * kTileRow * itemsize;
  const int64_t dst_offset =
      local_dst_row * w_dst * itemsize + local_dst_col * kTileRow * itemsize;

  if (rank > 2) {
    const size_t num_outer_dims = rank - 2;
    std::vector<int64_t> outer_shape(num_outer_dims, 1);
    for (size_t d = 0; d < num_outer_dims; ++d) {
      outer_shape[d] = p_int.sizes[d];
    }
    std::vector<int64_t> src_outer_strides(num_outer_dims, 1);
    src_outer_strides[num_outer_dims - 1] =
        p_src.sizes[rank - 2] * p_src.sizes[rank - 1];
    for (int d = static_cast<int>(num_outer_dims) - 2; d >= 0; --d) {
      src_outer_strides[d] = src_outer_strides[d + 1] * p_src.sizes[d + 1];
    }

    std::vector<int64_t> dst_outer_strides(num_outer_dims, 1);
    dst_outer_strides[num_outer_dims - 1] =
        p_dst.sizes[rank - 2] * p_dst.sizes[rank - 1];
    for (int d = static_cast<int>(num_outer_dims) - 2; d >= 0; --d) {
      dst_outer_strides[d] = dst_outer_strides[d + 1] * p_dst.sizes[d + 1];
    }

    std::vector<int64_t> src_local_outer_start(num_outer_dims, 0);
    std::vector<int64_t> dst_local_outer_start(num_outer_dims, 0);
    for (size_t d = 0; d < num_outer_dims; ++d) {
      src_local_outer_start[d] = p_int.offsets[d] - p_src.offsets[d];
      dst_local_outer_start[d] = p_int.offsets[d] - p_dst.offsets[d];
    }

    const int64_t num_outer = Product(outer_shape);
    if ((count == 1 ||
         (size_bytes == src_stride && size_bytes == dst_stride)) &&
        num_outer_dims == 1) {
      const int64_t inner_size = size_bytes * count;
      const int64_t s_stride_outer = src_outer_strides.back() * itemsize;
      const int64_t d_stride_outer = dst_outer_strides.back() * itemsize;
      const int64_t base_src =
          src_offset + src_local_outer_start[0] * s_stride_outer;
      const int64_t base_dst =
          dst_offset + dst_local_outer_start[0] * d_stride_outer;
      if (inner_size == s_stride_outer && inner_size == d_stride_outer) {
        return {StridedCopyChunk{
            .src_offset_bytes = base_src,
            .dst_offset_bytes = base_dst,
            .size_bytes = inner_size * num_outer,
            .src_stride_bytes = 0,
            .dst_stride_bytes = 0,
            .count = 1,
        }};
      }
      return {StridedCopyChunk{
          .src_offset_bytes = base_src,
          .dst_offset_bytes = base_dst,
          .size_bytes = inner_size,
          .src_stride_bytes = s_stride_outer,
          .dst_stride_bytes = d_stride_outer,
          .count = num_outer,
      }};
    }

    std::vector<StridedCopyChunk> chunks;
    chunks.reserve(num_outer);
    for (int64_t i = 0; i < num_outer; ++i) {
      int64_t temp = i;
      int64_t outer_src_bytes = 0;
      int64_t outer_dst_bytes = 0;
      for (int d = static_cast<int>(num_outer_dims) - 1; d >= 0; --d) {
        const int64_t coord = temp % outer_shape[d];
        temp /= outer_shape[d];
        outer_src_bytes += (src_local_outer_start[d] + coord) *
                           src_outer_strides[d] * itemsize;
        outer_dst_bytes += (dst_local_outer_start[d] + coord) *
                           dst_outer_strides[d] * itemsize;
      }
      chunks.push_back(StridedCopyChunk{
          .src_offset_bytes = src_offset + outer_src_bytes,
          .dst_offset_bytes = dst_offset + outer_dst_bytes,
          .size_bytes = size_bytes,
          .src_stride_bytes = src_stride,
          .dst_stride_bytes = dst_stride,
          .count = count,
      });
    }
    return chunks;
  }

  if (size_bytes == src_stride && size_bytes == dst_stride) {
    return {StridedCopyChunk{
        .src_offset_bytes = src_offset,
        .dst_offset_bytes = dst_offset,
        .size_bytes = size_bytes * count,
        .src_stride_bytes = 0,
        .dst_stride_bytes = 0,
        .count = 1,
    }};
  }
  return {StridedCopyChunk{
      .src_offset_bytes = src_offset,
      .dst_offset_bytes = dst_offset,
      .size_bytes = size_bytes,
      .src_stride_bytes = src_stride,
      .dst_stride_bytes = dst_stride,
      .count = count,
  }};
}

std::vector<VariableBundleSpec> LogicalReshardPlanner::PartitionVariableBundles(
    int32_t num_layers, absl::Span<const int64_t> dst_layer_shard_bytes,
    int32_t num_bundle_groups) {
  if (num_layers <= 0) return {};
  const int32_t b_count =
      std::max<int32_t>(1, std::min<int32_t>(num_layers, num_bundle_groups));

  std::vector<int64_t> layer_weights(num_layers, 1);
  int64_t total_weight = 0;
  for (int32_t l = 0; l < num_layers; ++l) {
    int64_t sz = (static_cast<size_t>(l) < dst_layer_shard_bytes.size() &&
                  dst_layer_shard_bytes[l] > 0)
                     ? dst_layer_shard_bytes[l]
                     : 1;
    layer_weights[l] = sz;
    total_weight += sz;
  }

  std::vector<VariableBundleSpec> bundles;
  bundles.reserve(b_count);

  int32_t current_layer = 0;
  int64_t remaining_weight = total_weight;

  for (int32_t b = 0; b < b_count && current_layer < num_layers; ++b) {
    const int32_t remaining_bundles = b_count - b;
    const int32_t remaining_layers = num_layers - current_layer;

    VariableBundleSpec spec;
    spec.bundle_id = b;

    if (remaining_bundles == 1) {
      for (int32_t l = current_layer; l < num_layers; ++l) {
        spec.layer_indices.push_back(l);
        const int64_t b_sz =
            static_cast<size_t>(l) < dst_layer_shard_bytes.size()
                ? dst_layer_shard_bytes[l]
                : 0;
        spec.layer_byte_sizes.push_back(b_sz);
        spec.total_bytes += b_sz;
      }
      current_layer = num_layers;
    } else {
      const int64_t target_weight =
          std::max<int64_t>(1, remaining_weight / remaining_bundles);
      int64_t accumulated_weight = 0;
      // Ensure at least 1 layer per remaining bundle.
      const int32_t max_end_layer = num_layers - (remaining_bundles - 1);
      while (current_layer < max_end_layer) {
        const int32_t l = current_layer;
        const int64_t w = layer_weights[l];
        if (!spec.layer_indices.empty() &&
            accumulated_weight + w > target_weight &&
            (accumulated_weight + w - target_weight) >
                (target_weight - accumulated_weight)) {
          break;
        }
        spec.layer_indices.push_back(l);
        const int64_t b_sz =
            static_cast<size_t>(l) < dst_layer_shard_bytes.size()
                ? dst_layer_shard_bytes[l]
                : 0;
        spec.layer_byte_sizes.push_back(b_sz);
        spec.total_bytes += b_sz;
        accumulated_weight += w;
        remaining_weight -= w;
        ++current_layer;
      }
      if (remaining_layers > 0 && spec.layer_indices.empty() &&
          current_layer < num_layers) {
        const int32_t l = current_layer++;
        const int64_t b_sz =
            static_cast<size_t>(l) < dst_layer_shard_bytes.size()
                ? dst_layer_shard_bytes[l]
                : 0;
        spec.layer_indices.push_back(l);
        spec.layer_byte_sizes.push_back(b_sz);
        spec.total_bytes += b_sz;
        remaining_weight -= layer_weights[l];
      }
    }
    bundles.push_back(std::move(spec));
  }

  return bundles;
}

absl::StatusOr<SeedLayout> LogicalReshardPlanner::BuildSeedLayout(
    absl::Span<const int64_t> bundle_bytes, int32_t num_dst_replicas,
    const SeedingOptions& options, int32_t trainer_streams) {
  if (num_dst_replicas <= 0) {
    return absl::InvalidArgumentError("num_dst_replicas must be positive");
  }
  if (options.num_stripes < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_stripes must be >= 0 (0 = auto), got ", options.num_stripes));
  }
  if (options.replication <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "replication must be positive, got ", options.replication));
  }
  std::vector<int32_t> order;
  if (options.replica_order.empty()) {
    order.resize(num_dst_replicas);
    for (int32_t r = 0; r < num_dst_replicas; ++r) order[r] = r;
  } else {
    if (static_cast<int32_t>(options.replica_order.size()) !=
        num_dst_replicas) {
      return absl::InvalidArgumentError(absl::StrCat(
          "replica_order has ", options.replica_order.size(), " entries for ",
          num_dst_replicas, " destination replicas"));
    }
    std::vector<bool> seen(num_dst_replicas, false);
    for (int32_t r : options.replica_order) {
      if (r < 0 || r >= num_dst_replicas || seen[r]) {
        return absl::InvalidArgumentError(
            "replica_order must be a permutation of the replica indices");
      }
      seen[r] = true;
    }
    order = options.replica_order;
  }

  SeedLayout layout;
  layout.replication = std::min(options.replication, num_dst_replicas);
  const int32_t num_bundles = static_cast<int32_t>(bundle_bytes.size());
  if (num_bundles == 0) {
    layout.num_stripes = 0;
    return layout;
  }
  const int32_t max_stripes = std::max<int32_t>(
      1, std::min<int32_t>(num_bundles, num_dst_replicas / layout.replication));
  layout.num_stripes = options.num_stripes > 0
                           ? std::min(options.num_stripes, max_stripes)
                           : max_stripes;
  const int32_t num_stripes = layout.num_stripes;
  const int32_t replication = layout.replication;

  // Stripes are contiguous, byte-balanced bundle ranges.
  for (const VariableBundleSpec& range :
       PartitionVariableBundles(num_bundles, bundle_bytes, num_stripes)) {
    layout.stripe_bundles.push_back(range.layer_indices);
  }

  // Seed slots are spread evenly over the sorted replica order.
  const int64_t num_slots = static_cast<int64_t>(num_stripes) * replication;
  layout.stripe_seeds.assign(num_stripes, std::vector<int32_t>(replication));
  for (int32_t g = 0; g < num_stripes; ++g) {
    for (int32_t c = 0; c < replication; ++c) {
      const int64_t slot = static_cast<int64_t>(g) * replication + c;
      layout.stripe_seeds[g][c] = order[slot * num_dst_replicas / num_slots];
    }
  }

  // Replica-major waves: copy 0 of every stripe, then copy 1, and so on.
  const int32_t streams = std::max<int32_t>(1, trainer_streams);
  for (int32_t c = 0; c < replication; ++c) {
    for (int32_t g = 0; g < num_stripes; ++g) {
      if (layout.trainer_waves.empty() ||
          static_cast<int32_t>(layout.trainer_waves.back().size()) >= streams) {
        layout.trainer_waves.emplace_back();
      }
      layout.trainer_waves.back().emplace_back(layout.stripe_seeds[g][c], g);
    }
  }
  return layout;
}

absl::StatusOr<LogicalReshardSchedule>
LogicalReshardPlanner::ComputeLogicalSchedule(
    absl::Span<const VariableSpec> src_variables,
    absl::Span<const VariableSpec> dst_variables, int32_t num_src_shards,
    int32_t num_dst_shards, int32_t num_dst_replicas,
    double broadcast_host_ratio, int32_t trainer_hosts, int32_t sampler_hosts,
    int32_t num_bundle_groups, const SeedingOptions& seeding) {
  if (src_variables.size() != dst_variables.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Source variable count (", src_variables.size(),
                     ") does not match destination variable count (",
                     dst_variables.size(), ")"));
  }
  if (num_src_shards <= 0 || num_dst_shards <= 0 || num_dst_replicas <= 0) {
    return absl::InvalidArgumentError(
        "Shard and replica counts must be positive");
  }

  const int32_t num_layers = static_cast<int32_t>(src_variables.size());
  const int32_t trainer_streams =
      seeding.trainer_streams > 0
          ? seeding.trainer_streams
          : ComputeSeedSamplerCount(num_dst_replicas, broadcast_host_ratio,
                                    trainer_hosts, sampler_hosts);

  LogicalReshardSchedule schedule;
  schedule.num_src_shards = num_src_shards;
  schedule.num_dst_shards = num_dst_shards;
  schedule.num_dst_replicas = num_dst_replicas;
  schedule.variable_names.reserve(num_layers);
  schedule.variable_to_plan_id.resize(num_layers, 0);
  schedule.dst_layer_shard_bytes.resize(num_layers, 0);
  schedule.skip_tiling_by_layer.resize(num_layers, false);
  schedule.dst_shard_expected_layer_chunks.assign(
      num_dst_shards, std::vector<int32_t>(num_layers, 0));

  absl::flat_hash_map<PlanSignatureKey, int32_t> sig_to_plan_id;
  // Per-plan_id template actions for replica 0: `[plan_id][src_shard_idx]` ->
  // list of actions (with dst_replica_idx == 0).
  std::vector<std::vector<std::vector<ShardCopyAction>>>
      base_actions_by_plan_and_shard;
  // Per-plan_id chunk counts per dst_shard_idx: `[plan_id][dst_shard_idx]`.
  std::vector<std::vector<int32_t>> dst_shard_chunks_by_plan;

  for (int32_t l = 0; l < num_layers; ++l) {
    const VariableSpec& sv = src_variables[l];
    const VariableSpec& dv = dst_variables[l];
    if (sv.global_shape != dv.global_shape) {
      return absl::InvalidArgumentError(
          absl::StrCat("Global shape mismatch for variable '", sv.name, "'"));
    }
    if (sv.itemsize <= 0 || sv.itemsize != dv.itemsize) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid or mismatched itemsize for variable '", sv.name, "'"));
    }

    schedule.variable_names.push_back(
        !sv.name.empty() ? sv.name : absl::StrCat("var_", l));

    const int32_t num_src_logical =
        std::max<int32_t>(1, static_cast<int32_t>(Product(sv.mesh_shape)));
    const int32_t num_dst_logical =
        std::max<int32_t>(1, static_cast<int32_t>(Product(dv.mesh_shape)));
    ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> src_phys_to_logical,
                          ResolvePhysicalToLogicalShardMapping(
                              sv, num_src_shards, num_src_logical));
    ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> dst_phys_to_logical,
                          ResolvePhysicalToLogicalShardMapping(
                              dv, num_dst_shards, num_dst_logical));

    const bool skip_tiling = sv.skip_tiling || dv.skip_tiling;
    PlanSignatureKey key{
        .src_global_shape = sv.global_shape,
        .src_mesh_shape = sv.mesh_shape,
        .src_layout = sv.layout,
        .src_phys_to_logical = src_phys_to_logical,
        .dst_global_shape = dv.global_shape,
        .dst_mesh_shape = dv.mesh_shape,
        .dst_layout = dv.layout,
        .dst_phys_to_logical = dst_phys_to_logical,
        .itemsize = sv.itemsize,
        .skip_tiling = skip_tiling,
    };

    int32_t plan_id = 0;
    auto it = sig_to_plan_id.find(key);
    if (it != sig_to_plan_id.end()) {
      plan_id = it->second;
    } else {
      plan_id = static_cast<int32_t>(base_actions_by_plan_and_shard.size());
      sig_to_plan_id[key] = plan_id;

      ABSL_ASSIGN_OR_RETURN(
          std::vector<NdSlice> src_slices,
          ComputeNdShardSlices(sv.global_shape, sv.mesh_shape, sv.layout));
      ABSL_ASSIGN_OR_RETURN(
          std::vector<NdSlice> dst_slices,
          ComputeNdShardSlices(dv.global_shape, dv.mesh_shape, dv.layout));

      std::vector<std::vector<ShardCopyAction>> plan_shard_actions(
          num_src_shards);
      std::vector<int32_t> plan_dst_chunks(num_dst_shards, 0);

      // Group physical source shards by the logical slice they hold.
      std::vector<std::vector<int32_t>> src_shards_for_logical(
          src_slices.size());
      for (int32_t s_phys = 0; s_phys < num_src_shards; ++s_phys) {
        const int32_t s_log = src_phys_to_logical[s_phys];
        if (s_log >= 0 && static_cast<size_t>(s_log) < src_slices.size()) {
          src_shards_for_logical[s_log].push_back(s_phys);
        }
      }

      for (int32_t d_phys = 0; d_phys < num_dst_shards; ++d_phys) {
        const int32_t d_log = dst_phys_to_logical[d_phys];
        if (d_log < 0 || static_cast<size_t>(d_log) >= dst_slices.size()) {
          continue;
        }
        const NdSlice& d_slice = dst_slices[d_log];

        for (size_t s_log = 0; s_log < src_slices.size(); ++s_log) {
          const std::vector<int32_t>& candidates =
              src_shards_for_logical[s_log];
          if (candidates.empty()) continue;

          std::optional<NdSlice> inter =
              IntersectNdSlices(src_slices[s_log], d_slice);
          if (!inter.has_value() || inter->IsEmpty()) continue;

          const int32_t s_phys =
              candidates[static_cast<size_t>(d_phys) % candidates.size()];

          // With skip_tiling the workers copy the tiled bytes as they are (no
          // untile/retile), so the chunks must address whole tiles. The
          // tile-aware generator falls back to `GenerateStridedCopyChunks`
          // when the slices are not tile aligned.
          std::vector<StridedCopyChunk> chunks =
              skip_tiling ? GenerateStridedCopyChunksTileAware(
                                src_slices[s_log], d_slice, *inter, sv.layout,
                                dv.layout, sv.itemsize)
                          : GenerateStridedCopyChunks(
                                src_slices[s_log], d_slice, *inter, sv.layout,
                                dv.layout, sv.itemsize);

          for (const StridedCopyChunk& c : chunks) {
            plan_shard_actions[s_phys].push_back(ShardCopyAction{
                .dst_replica_idx = 0,
                .dst_shard_idx = d_phys,
                .src_offset_bytes = c.src_offset_bytes,
                .dst_offset_bytes = c.dst_offset_bytes,
                .size_bytes = c.size_bytes,
                .src_stride_bytes = c.src_stride_bytes,
                .dst_stride_bytes = c.dst_stride_bytes,
                .count = c.count,
                .plan_id = plan_id,
            });
            plan_dst_chunks[d_phys] += 1;
          }
        }
      }

      base_actions_by_plan_and_shard.push_back(std::move(plan_shard_actions));
      dst_shard_chunks_by_plan.push_back(std::move(plan_dst_chunks));
    }

    schedule.variable_to_plan_id[l] = plan_id;
    schedule.skip_tiling_by_layer[l] = skip_tiling;

    // Compute per-shard destination byte size for layer `l`.
    int64_t shard_elems = 1;
    for (size_t d = 0; d < dv.global_shape.size(); ++d) {
      const int64_t dim_sharding =
          (d < dv.mesh_shape.size() && dv.mesh_shape[d] > 1) ? dv.mesh_shape[d]
                                                             : 1;
      shard_elems *= (dv.global_shape[d] / dim_sharding);
    }
    schedule.dst_layer_shard_bytes[l] = shard_elems * dv.itemsize;

    for (int32_t d_idx = 0; d_idx < num_dst_shards; ++d_idx) {
      schedule.dst_shard_expected_layer_chunks[d_idx][l] =
          dst_shard_chunks_by_plan[plan_id][d_idx];
    }
  }

  schedule.num_unique_plans =
      static_cast<int32_t>(base_actions_by_plan_and_shard.size());
  schedule.trainer_actions_by_shard_and_plan.assign(
      num_src_shards,
      std::vector<std::vector<ShardCopyAction>>(schedule.num_unique_plans));

  const int32_t eff_trainer_hosts = std::max<int32_t>(1, trainer_hosts);
  const int32_t shards_per_trainer_host =
      std::max<int32_t>(1, num_src_shards / eff_trainer_hosts);
  const int32_t eff_sampler_hosts = std::max<int32_t>(1, sampler_hosts);
  const int32_t shards_per_sampler_host =
      std::max<int32_t>(1, num_dst_shards / eff_sampler_hosts);
  const int32_t eff_streams = std::max<int32_t>(1, trainer_streams);

  for (int32_t pid = 0; pid < schedule.num_unique_plans; ++pid) {
    for (int32_t s_idx = 0; s_idx < num_src_shards; ++s_idx) {
      const auto& base_vec = base_actions_by_plan_and_shard[pid][s_idx];
      auto& out_vec = schedule.trainer_actions_by_shard_and_plan[s_idx][pid];
      out_vec.reserve(base_vec.size());

      // Within a wave, trainer host `t` starts with push `t % wave_size` (see
      // `DynamicPullEngine::MaterializeTransferPlan`), so hosts `t` and
      // `t + wave_size` start on the same replica. Offset their destination
      // shard order by sampler host, and by local source shard, so that the
      // hosts of that replica receive about the same traffic concurrently.
      const int32_t trainer_host_idx = std::min<int32_t>(
          eff_trainer_hosts - 1, s_idx / shards_per_trainer_host);
      const int32_t local_src_shard = s_idx % shards_per_trainer_host;
      const int32_t cycle = trainer_host_idx / eff_streams;
      const int32_t dst_host_offset =
          (cycle % eff_sampler_hosts) * shards_per_sampler_host;
      const int32_t shard_offset =
          (dst_host_offset + local_src_shard) % num_dst_shards;

      std::vector<std::vector<ShardCopyAction>> actions_by_dst_shard(
          num_dst_shards);
      std::vector<ShardCopyAction> other_actions;
      for (const ShardCopyAction& act : base_vec) {
        if (act.dst_shard_idx >= 0 && act.dst_shard_idx < num_dst_shards) {
          actions_by_dst_shard[act.dst_shard_idx].push_back(act);
        } else {
          other_actions.push_back(act);
        }
      }
      for (int32_t d_step = 0; d_step < num_dst_shards; ++d_step) {
        const int32_t d_phys = (d_step + shard_offset) % num_dst_shards;
        out_vec.insert(out_vec.end(), actions_by_dst_shard[d_phys].begin(),
                       actions_by_dst_shard[d_phys].end());
      }
      out_vec.insert(out_vec.end(), other_actions.begin(), other_actions.end());
    }
  }

  // Bundles: at least one per requested stripe (capped by the variables).
  const int32_t eff_bundle_groups =
      std::max(num_bundle_groups, seeding.num_stripes);
  schedule.variable_bundles = PartitionVariableBundles(
      num_layers, schedule.dst_layer_shard_bytes, eff_bundle_groups);
  std::vector<int64_t> bundle_bytes;
  bundle_bytes.reserve(schedule.variable_bundles.size());
  for (const VariableBundleSpec& bundle : schedule.variable_bundles) {
    bundle_bytes.push_back(bundle.total_bytes);
  }
  ABSL_ASSIGN_OR_RETURN(schedule.seed_layout,
                        BuildSeedLayout(bundle_bytes, num_dst_replicas, seeding,
                                        trainer_streams));
  return schedule;
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

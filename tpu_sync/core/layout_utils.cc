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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"

namespace tpu_raiden {
namespace {

// Returns how many leading dimensions of `dims` `folded_dims` folds into its
// first dimension, or 0 unless that is two or more of them and the remaining
// dimensions of both are equal.
size_t NumFoldedLeadingDims(absl::Span<const int64_t> dims,
                            absl::Span<const int64_t> folded_dims) {
  if (folded_dims.empty() || dims.size() <= folded_dims.size()) {
    return 0;
  }
  const size_t num_folded = dims.size() - folded_dims.size() + 1;
  if (dims.subspan(num_folded) != folded_dims.subspan(1)) {
    return 0;
  }
  int64_t product = 1;
  for (const int64_t dim : dims.first(num_folded)) {
    // A product past folded_dims[0] can no longer equal it, and stopping there
    // keeps the multiplication from overflowing.
    if (dim < 0 || (dim > 0 && product > folded_dims[0] / dim)) {
      return 0;
    }
    product *= dim;
  }
  return product == folded_dims[0] ? num_folded : 0;
}

}  // namespace

std::string DimsToString(absl::Span<const int64_t> dims) {
  return absl::StrCat("[", absl::StrJoin(dims, ", "), "]");
}

bool FoldsLeadingDims(absl::Span<const int64_t> dims,
                      absl::Span<const int64_t> folded_dims) {
  return NumFoldedLeadingDims(dims, folded_dims) > 0;
}

absl::Status CheckFoldedDeviceLayout(const xla::Shape& buffer_shape,
                                     absl::Span<const int64_t> folded_dims,
                                     const xla::Layout& folded_layout) {
  const absl::Span<const int64_t> dims = buffer_shape.dimensions();
  const size_t num_folded = NumFoldedLeadingDims(dims, folded_dims);
  if (num_folded == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        DimsToString(folded_dims), " does not fold leading dimensions of ",
        DimsToString(dims)));
  }
  if (!buffer_shape.has_layout()) {
    return absl::InvalidArgumentError("the buffer has no device layout");
  }
  if (folded_layout.minor_to_major().size() != folded_dims.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("device layout ", folded_layout.ToString(),
                     " does not lay out ", DimsToString(folded_dims)));
  }
  const xla::Layout& layout = buffer_shape.layout();
  const std::string folded_buffer =
      absl::StrCat("a ", DimsToString(folded_dims), " buffer (device layout ",
                   folded_layout.ToString(), ")");

  // Folding regroups the major dimensions, so it keeps every element in place
  // only if both layouts put dimension 0 outermost and the others in order.
  if (!xla::LayoutUtil::IsMonotonicWithDim0Major(layout) ||
      !xla::LayoutUtil::IsMonotonicWithDim0Major(folded_layout)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "device layout ", layout.ToString(), " and ", folded_buffer,
        " do not both keep dimensions in row-major order"));
  }
  // Apart from their ranks, the layouts must be the same: same tiles, element
  // size and padding. Memory spaces do not place bytes, so they may differ.
  xla::Layout expected = folded_layout;
  expected.mutable_minor_to_major()->assign(layout.minor_to_major().begin(),
                                            layout.minor_to_major().end());
  if (!xla::Layout::Equal().IgnoreMemorySpace()(expected, layout)) {
    return absl::InvalidArgumentError(
        absl::StrCat("device layout ", layout.ToString(),
                     " differs from that of ", folded_buffer));
  }

  // Folded dimensions of size 1 before the last one do not regroup anything.
  const int64_t num_groups = absl::c_accumulate(
      dims.first(num_folded - 1), int64_t{1}, std::multiplies<>());
  const absl::Span<const xla::Tile> tiles = layout.tiles();
  if (num_groups == 1 || tiles.empty()) {
    return absl::OkStatus();
  }
  for (const xla::Tile& tile : tiles) {
    for (const int64_t tile_dim : tile.dimensions()) {
      if (tile_dim <= 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("device layout ", layout.ToString(),
                         " has a tile dimension without a fixed size"));
      }
    }
  }
  // Each later tile subdivides the tiles before it. One with more dimensions
  // than those reaches into the grid of tiles and can interleave the folded
  // dimensions.
  size_t num_tile_dims = tiles[0].dimensions().size();
  for (const xla::Tile& tile : tiles.subspan(1)) {
    if (tile.dimensions().size() > num_tile_dims) {
      return absl::InvalidArgumentError(
          absl::StrCat("device layout ", layout.ToString(),
                       " interleaves elements of different tiles"));
    }
    num_tile_dims += tile.dimensions().size();
  }

  // The first tile covers the same number of minor dimensions of both shapes.
  // If those are only dimensions the two share, the folded dimensions stay
  // untiled and major, where folding them moves nothing.
  const size_t num_kept = folded_dims.size() - 1;
  const absl::Span<const int64_t> tile = tiles[0].dimensions();
  if (tile.size() <= num_kept) {
    return absl::OkStatus();
  }
  if (tile.size() > num_kept + 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("the tiles of device layout ", layout.ToString(),
                     " span more than one of the folded dimensions"));
  }
  // Otherwise the tile also covers the last folded dimension, which must then
  // be a whole number of tiles: the buffer pads every group of rows to whole
  // tiles, where a folded buffer pads only its last tile.
  const int64_t rows = dims[num_folded - 1];
  if (rows % tile[0] != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "device layout ", layout.ToString(), " pads each group of ", rows,
        " rows to whole tiles of ", tile[0], " rows, unlike ", folded_buffer));
  }
  return absl::OkStatus();
}

}  // namespace tpu_raiden

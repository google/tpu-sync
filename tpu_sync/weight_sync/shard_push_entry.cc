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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

bool AddCheckedU64(uint64_t a, uint64_t b, uint64_t* out) {
  if (a > std::numeric_limits<uint64_t>::max() - b) {
    return false;
  }
  *out = a + b;
  return true;
}

bool MulCheckedU64(uint64_t a, uint64_t b, uint64_t* out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  *out = a * b;
  return true;
}

absl::Status CheckNonNegative(absl::string_view entry_ctx,
                              absl::string_view field, int64_t value) {
  if (value < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(entry_ctx, " sets negative ", field, "=", value));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ShardPushEntry> ShardPushEntry::FromProto(
    const tpu_sync::rpc::ShardPushEntryProto& proto,
    absl::string_view entry_ctx, const ShardPushEntryOptions& options) {
  ABSL_RETURN_IF_ERROR(
      CheckNonNegative(entry_ctx, "dst_shard_idx", proto.dst_shard_idx()));
  ABSL_RETURN_IF_ERROR(CheckNonNegative(entry_ctx, "src_offset_bytes",
                                        proto.src_offset_bytes()));
  ABSL_RETURN_IF_ERROR(CheckNonNegative(entry_ctx, "dst_offset_bytes",
                                        proto.dst_offset_bytes()));
  ABSL_RETURN_IF_ERROR(
      CheckNonNegative(entry_ctx, "size_bytes", proto.size_bytes()));
  ABSL_RETURN_IF_ERROR(CheckNonNegative(entry_ctx, "src_stride_bytes",
                                        proto.src_stride_bytes()));
  ABSL_RETURN_IF_ERROR(CheckNonNegative(entry_ctx, "dst_stride_bytes",
                                        proto.dst_stride_bytes()));
  ABSL_RETURN_IF_ERROR(CheckNonNegative(entry_ctx, "count", proto.count()));

  const int num_outer_dims = proto.outer_counts_size();
  if (proto.outer_src_strides_bytes_size() != num_outer_dims ||
      proto.outer_dst_strides_bytes_size() != num_outer_dims) {
    return absl::InvalidArgumentError(absl::StrCat(
        entry_ctx, " has mismatched outer_counts (", num_outer_dims,
        "), outer_src_strides_bytes (", proto.outer_src_strides_bytes_size(),
        "), and outer_dst_strides_bytes (",
        proto.outer_dst_strides_bytes_size(), ") sizes"));
  }

  uint64_t count =
      proto.count() > 0 ? static_cast<uint64_t>(proto.count()) : uint64_t{1};
  uint64_t src_offset = static_cast<uint64_t>(proto.src_offset_bytes());
  uint64_t dst_offset = static_cast<uint64_t>(proto.dst_offset_bytes());
  uint64_t size = static_cast<uint64_t>(proto.size_bytes());
  uint64_t src_stride = static_cast<uint64_t>(proto.src_stride_bytes());
  uint64_t dst_stride = static_cast<uint64_t>(proto.dst_stride_bytes());
  if (options.pre_tiled) {
    if (num_outer_dims != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx,
          " HOST_TILING_MODE_PRE_TILED entries must not be strided "
          "(outer_counts_size=",
          num_outer_dims, ")"));
    }
    count = 1;
    src_offset = 0;
    dst_offset = 0;
    src_stride = 0;
    dst_stride = 0;
    if (options.device_size.has_value()) {
      size = *options.device_size;
    }
  }

  const uint64_t effective_src_stride = src_stride > 0 ? src_stride : size;
  const uint64_t effective_dst_stride = dst_stride > 0 ? dst_stride : size;
  const bool is_inner_contiguous =
      (count == 1) ||
      (effective_src_stride == size && effective_dst_stride == size);
  if (!is_inner_contiguous && effective_src_stride < size) {
    return absl::InvalidArgumentError(
        absl::StrCat(entry_ctx, " source stride (", effective_src_stride,
                     ") cannot be smaller than slice size (", size, ")"));
  }
  if (!is_inner_contiguous && effective_dst_stride < size) {
    return absl::InvalidArgumentError(
        absl::StrCat(entry_ctx, " destination stride (", effective_dst_stride,
                     ") cannot be smaller than slice size (", size, ")"));
  }

  uint64_t inner_src_span = 0;
  uint64_t inner_dst_span = 0;
  if (is_inner_contiguous) {
    if (!MulCheckedU64(count, size, &inner_src_span)) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " inner contiguous span overflows (count=", count,
          ", size_bytes=", size, ")"));
    }
    inner_dst_span = inner_src_span;
  } else {
    uint64_t src_step_span = 0;
    if (!MulCheckedU64(count - 1, effective_src_stride, &src_step_span) ||
        !AddCheckedU64(src_step_span, size, &inner_src_span)) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " inner strided source span overflows (count=", count,
          ", src_stride_bytes=", effective_src_stride, ", size_bytes=", size,
          ")"));
    }
    uint64_t dst_step_span = 0;
    if (!MulCheckedU64(count - 1, effective_dst_stride, &dst_step_span) ||
        !AddCheckedU64(dst_step_span, size, &inner_dst_span)) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " inner strided destination span overflows (count=", count,
          ", dst_stride_bytes=", effective_dst_stride, ", size_bytes=", size,
          ")"));
    }
  }

  std::vector<uint64_t> outer_counts;
  std::vector<uint64_t> outer_src_strides;
  std::vector<uint64_t> outer_dst_strides;
  outer_counts.reserve(num_outer_dims);
  outer_src_strides.reserve(num_outer_dims);
  outer_dst_strides.reserve(num_outer_dims);
  uint64_t num_outer = 1;
  uint64_t max_src_offset = src_offset;
  uint64_t max_dst_offset = dst_offset;
  for (int d = 0; d < num_outer_dims; ++d) {
    if (proto.outer_counts(d) <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(entry_ctx, " outer_counts[", d,
                       "] must be positive; got ", proto.outer_counts(d)));
    }
    if (proto.outer_src_strides_bytes(d) < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " outer_src_strides_bytes[", d,
          "] must be non-negative; got ", proto.outer_src_strides_bytes(d)));
    }
    if (proto.outer_dst_strides_bytes(d) < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " outer_dst_strides_bytes[", d,
          "] must be non-negative; got ", proto.outer_dst_strides_bytes(d)));
    }
    const uint64_t dim_count = static_cast<uint64_t>(proto.outer_counts(d));
    const uint64_t src_dim_stride =
        static_cast<uint64_t>(proto.outer_src_strides_bytes(d));
    const uint64_t dst_dim_stride =
        static_cast<uint64_t>(proto.outer_dst_strides_bytes(d));
    if (!MulCheckedU64(num_outer, dim_count, &num_outer) ||
        num_outer > kMaxPushEntryChunks) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " outer_counts product exceeds uint32_t max at dim ", d,
          " (outer_counts[", d, "]=", dim_count, ")"));
    }
    uint64_t src_dim_span = 0;
    if (!MulCheckedU64(dim_count - 1, src_dim_stride, &src_dim_span) ||
        !AddCheckedU64(max_src_offset, src_dim_span, &max_src_offset)) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " outer source offset overflows at dim ", d,
          " (outer_counts[", d, "]=", dim_count, ", outer_src_strides_bytes[",
          d, "]=", src_dim_stride, ")"));
    }
    uint64_t dst_dim_span = 0;
    if (!MulCheckedU64(dim_count - 1, dst_dim_stride, &dst_dim_span) ||
        !AddCheckedU64(max_dst_offset, dst_dim_span, &max_dst_offset)) {
      return absl::InvalidArgumentError(absl::StrCat(
          entry_ctx, " outer destination offset overflows at dim ", d,
          " (outer_counts[", d, "]=", dim_count, ", outer_dst_strides_bytes[",
          d, "]=", dst_dim_stride, ")"));
    }
    outer_counts.push_back(dim_count);
    outer_src_strides.push_back(src_dim_stride);
    outer_dst_strides.push_back(dst_dim_stride);
  }

  uint64_t max_src_end = 0;
  if (!AddCheckedU64(max_src_offset, inner_src_span, &max_src_end)) {
    return absl::InvalidArgumentError(
        absl::StrCat(entry_ctx,
                     num_outer_dims > 0 ? " outer source span overflows"
                                        : " source span overflows",
                     " (max_src_offset=", max_src_offset,
                     ", inner_src_span=", inner_src_span, ")"));
  }
  uint64_t max_dst_end = 0;
  if (!AddCheckedU64(max_dst_offset, inner_dst_span, &max_dst_end)) {
    return absl::InvalidArgumentError(
        absl::StrCat(entry_ctx,
                     num_outer_dims > 0 ? " outer destination span overflows"
                                        : " destination span overflows",
                     " (max_dst_offset=", max_dst_offset,
                     ", inner_dst_span=", inner_dst_span, ")"));
  }

  if (options.host_size.has_value()) {
    const uint64_t host_size = *options.host_size;
    if (max_src_end > host_size) {
      return absl::InvalidArgumentError(
          absl::StrCat(entry_ctx, " source push range [", max_src_offset, ", ",
                       max_src_offset, "+", inner_src_span,
                       ") exceeds host buffer size ", host_size));
    }
  }

  ShardPushEntry entry;
  entry.dst_shard_idx_ = static_cast<uint64_t>(proto.dst_shard_idx());
  entry.count_ = count;
  entry.src_offset_ = src_offset;
  entry.dst_offset_ = dst_offset;
  entry.size_ = size;
  entry.effective_src_stride_ = effective_src_stride;
  entry.effective_dst_stride_ = effective_dst_stride;
  entry.inner_src_span_ = inner_src_span;
  entry.outer_counts_ = std::move(outer_counts);
  entry.outer_src_strides_ = std::move(outer_src_strides);
  entry.outer_dst_strides_ = std::move(outer_dst_strides);
  entry.num_outer_ = num_outer;
  entry.is_inner_contiguous_ = is_inner_contiguous;
  return entry;
}

}  // namespace tpu_raiden::weight_sync

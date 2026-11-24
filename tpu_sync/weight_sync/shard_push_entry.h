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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_SHARD_PUSH_ENTRY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_SHARD_PUSH_ENTRY_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden::weight_sync {

// Maximum number of chunks a single push entry or destination may emit; matches
// the 32-bit chunk counter in the transport wire format.
inline constexpr uint64_t kMaxPushEntryChunks =
    std::numeric_limits<uint32_t>::max();

// Options for ShardPushEntry::FromProto().
struct ShardPushEntryOptions {
  // Whether the push relays HOST_TILING_MODE_PRE_TILED host buffers, which
  // hold device-layout bytes: each entry then sends one whole shard from
  // offset 0, unstrided.
  bool pre_tiled = false;
  // With `pre_tiled`, the shard's device size, which replaces `size_bytes`.
  std::optional<uint64_t> device_size;
  // The size of the source host buffer, if known; every source row must fit in
  // it.
  std::optional<uint64_t> host_size;
};

// The destination shard and byte ranges of one ShardPushEntryProto, after any
// HOST_TILING_MODE_PRE_TILED overrides: count() inner slices of size() bytes,
// repeated over the outer dimensions. FromProto() validates, so every
// ShardPushEntry's source and destination ranges end within uint64_t and it has
// at most kMaxPushEntryChunks outer rows. Routing fields (peers, layer, and
// pool group) stay in the proto, since their checks depend on the kind of push.
class ShardPushEntry final {
 public:
  // Returns the entry that `proto` describes, or InvalidArgumentError starting
  // with `entry_ctx` and naming the offending field and value: a negative
  // signed field, outer arrays of different lengths, a non-positive outer
  // count, outer dimensions in a pre-tiled entry, a stride smaller than the
  // slice size when strided, a span or range end that overflows uint64_t, more
  // than kMaxPushEntryChunks outer rows, or a source row past
  // `options.host_size`. With `options.pre_tiled`, forces a count of 1, zero
  // offsets, unset strides (so both effective strides equal size()), and a
  // size of `*options.device_size` if set.
  static absl::StatusOr<ShardPushEntry> FromProto(
      const tpu_sync::rpc::ShardPushEntryProto& proto,
      absl::string_view entry_ctx, const ShardPushEntryOptions& options = {});

  uint64_t dst_shard_idx() const { return dst_shard_idx_; }
  // Number of inner slices: 1 if the proto leaves `count` unset.
  uint64_t count() const { return count_; }
  // Byte offsets of the first inner slice in the source host buffer and in the
  // destination shard.
  uint64_t src_offset() const { return src_offset_; }
  uint64_t dst_offset() const { return dst_offset_; }
  // Bytes per inner slice.
  uint64_t size() const { return size_; }
  // Byte steps between inner slices: size() if the proto leaves them unset.
  uint64_t effective_src_stride() const { return effective_src_stride_; }
  uint64_t effective_dst_stride() const { return effective_dst_stride_; }
  // True iff the inner slices form one contiguous range on both sides: count()
  // is 1 or both strides equal size().
  bool is_inner_contiguous() const { return is_inner_contiguous_; }
  // Bytes from the start of the first inner source slice to the end of the
  // last.
  uint64_t inner_src_span() const { return inner_src_span_; }

  // Outer dimensions, outermost first: dimension `d` repeats everything inside
  // it outer_counts()[d] times, stepping outer_src_strides()[d] bytes in the
  // source and outer_dst_strides()[d] bytes in the destination.
  absl::Span<const uint64_t> outer_counts() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return outer_counts_;
  }
  absl::Span<const uint64_t> outer_src_strides() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return outer_src_strides_;
  }
  absl::Span<const uint64_t> outer_dst_strides() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return outer_dst_strides_;
  }
  // Number of outer rows: the product of outer_counts(), or 1 without outer
  // dimensions.
  uint64_t num_outer() const { return num_outer_; }

 private:
  ShardPushEntry() = default;

  uint64_t dst_shard_idx_ = 0;
  uint64_t count_ = 1;
  uint64_t src_offset_ = 0;
  uint64_t dst_offset_ = 0;
  uint64_t size_ = 0;
  uint64_t effective_src_stride_ = 0;
  uint64_t effective_dst_stride_ = 0;
  uint64_t inner_src_span_ = 0;
  std::vector<uint64_t> outer_counts_;
  std::vector<uint64_t> outer_src_strides_;
  std::vector<uint64_t> outer_dst_strides_;
  uint64_t num_outer_ = 1;
  bool is_inner_contiguous_ = true;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_SHARD_PUSH_ENTRY_H_

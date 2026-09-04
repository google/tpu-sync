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

#ifndef THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_IMPL_H_
#define THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace raiden {

using ::xla::PjRtBuffer;
using ::xla::PjRtCApiBuffer;
using ::xla::Shape;

// Pure C++ implementation of D2H transfer core
inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_core(
    const std::vector<RaidenBufferHandle>& src_buffers,
    const std::vector<uint8_t*>& dst_ptrs, const std::vector<size_t>& dst_sizes,
    absl::Span<const int64_t> src_offsets_major_dim,
    absl::Span<const int64_t> dst_offsets_major_dim,
    absl::Span<const int64_t> copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock = false) {
  size_t num_shards = src_buffers.size();

  if (num_shards == 0) {
    return PjRtCopyFuture(std::vector<BufferHolder>{});
  }

  if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
      src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size lists must match");
  }

  const RaidenBufferHandle& first_buffer = src_buffers[0];
  const xla::Shape& shape = first_buffer.shape;

  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  if (is_partial) {
    if (shape.dimensions_size() < 3) {
      return absl::InvalidArgumentError(
          "Only support arrays with rank >= 3 for partial copies");
    }
  }

  if (num_shards != dst_ptrs.size() || num_shards != dst_sizes.size()) {
    return absl::InvalidArgumentError(
        "Number of shards in source and destination must match");
  }

  int64_t physical_size = first_buffer.GetOnDeviceSizeInBytes();

  int64_t slice_byte_size = GetMajorSliceByteSize(shape);

  if (is_partial && slice_byte_size % 4096 != 0) {
    return absl::InvalidArgumentError(
        "Unsupported shape: slice byte size must be a multiple "
        "of tile size (4KB) on device for partial copies");
  }

  std::vector<xla::Future<>> all_futures;
  BufferHolders holds;
  holds.reserve(num_shards);
  all_futures.reserve(num_shards);

  for (size_t i = 0; i < num_shards; ++i) {
    uint8_t* dst_data = dst_ptrs[i];
    size_t dst_size = dst_sizes[i];

    const RaidenBufferHandle& hold = src_buffers[i];

    if (!is_partial) {
      // Full copy.
      if (dst_size < physical_size) {
        return absl::InvalidArgumentError(
            "Destination buffer too small for raw tiled copy");
      }
      xla::Future<> future =
          hold.CopyRawDeviceToHost(dst_data, 0, physical_size);
      all_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        int64_t physical_offset = src_major_dim_offset * slice_byte_size;
        int64_t size_to_copy = major_dim_size * slice_byte_size;
        int64_t dst_offset = dst_major_dim_offset * slice_byte_size;

        if (physical_offset + size_to_copy > physical_size) {
          return absl::OutOfRangeError(
              "Copy range exceeds source buffer size.");
        }

        if (dst_offset + size_to_copy > dst_size) {
          return absl::OutOfRangeError(
              "Copy range exceeds destination buffer size.");
        }

        uint8_t* dst_ptr = dst_data + dst_offset;
        xla::Future<> future =
            hold.CopyRawDeviceToHost(dst_ptr, physical_offset, size_to_copy);
        all_futures.push_back(std::move(future));
      }
    }
    holds.push_back(BufferHolder{hold.c_hold, hold.common_hold,
                                 /*ext_hold=*/nullptr, /*user_hold=*/nullptr});
  }
  return PjRtCopyFuture(xla::JoinFutures(all_futures), std::move(holds));
}

// Pure H2D transfer core
inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_core(
    const std::vector<RaidenBufferHandle>& dst_buffers,
    const std::vector<const uint8_t*>& src_ptrs,
    const std::vector<size_t>& src_sizes,
    absl::Span<const int64_t> src_offsets_major_dim,
    absl::Span<const int64_t> dst_offsets_major_dim,
    absl::Span<const int64_t> copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock = false) {
  size_t num_shards = dst_buffers.size();

  if (num_shards == 0) {
    return PjRtCopyFuture(std::vector<BufferHolder>{});
  }

  if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
      src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size lists must match");
  }

  const RaidenBufferHandle& first_buffer = dst_buffers[0];
  const xla::Shape& shape = first_buffer.shape;

  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  if (is_partial) {
    if (shape.dimensions_size() < 3) {
      return absl::InvalidArgumentError(
          "Only support arrays with rank >= 3 for partial copies");
    }
  }

  if (num_shards != src_ptrs.size() || num_shards != src_sizes.size()) {
    return absl::InvalidArgumentError(
        "Number of shards in source and destination must match");
  }

  int64_t physical_size = first_buffer.GetOnDeviceSizeInBytes();

  int64_t slice_byte_size = GetMajorSliceByteSize(shape);

  if (is_partial && slice_byte_size % 4096 != 0) {
    return absl::InvalidArgumentError(
        "Unsupported shape: slice byte size must be a multiple "
        "of tile size (4KB) on device for partial copies");
  }

  std::vector<xla::Future<>> all_futures;
  BufferHolders holds;
  holds.reserve(num_shards);
  all_futures.reserve(num_shards);

  for (size_t i = 0; i < num_shards; ++i) {
    const uint8_t* src_data = src_ptrs[i];
    size_t src_size = src_sizes[i];

    const RaidenBufferHandle& hold = dst_buffers[i];

    if (!is_partial) {
      // Full copy.
      if (src_size < physical_size) {
        return absl::InvalidArgumentError(
            "Source buffer too small for raw tiled copy");
      }
      xla::Future<> future =
          hold.CopyRawHostToDevice(src_data, 0, physical_size);
      all_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        int64_t physical_offset = dst_major_dim_offset * slice_byte_size;
        int64_t size_to_copy = major_dim_size * slice_byte_size;
        int64_t src_offset = src_major_dim_offset * slice_byte_size;

        if (src_offset + size_to_copy > src_size) {
          return absl::OutOfRangeError("Copy range exceeds source buffer size");
        }
        if (physical_offset + size_to_copy > physical_size) {
          return absl::OutOfRangeError(
              "Copy range exceeds destination buffer size");
        }
        const uint8_t* src_ptr = src_data + src_offset;
        xla::Future<> future =
            hold.CopyRawHostToDevice(src_ptr, physical_offset, size_to_copy);
        all_futures.push_back(std::move(future));
      }
    }
    holds.push_back(BufferHolder{hold.c_hold, hold.common_hold,
                                 /*ext_hold=*/nullptr, /*user_hold=*/nullptr});
  }
  return PjRtCopyFuture(xla::JoinFutures(all_futures), std::move(holds));
}

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_IMPL_H_

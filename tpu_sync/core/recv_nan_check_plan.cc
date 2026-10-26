// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <tuple>

#include "tpu_sync/core/recv_nan_check.h"
namespace tpu_raiden {
recv_nan_internal::ScanBatch recv_nan_internal::BuildNanCheckBatch(
    std::span<const BatchInput> inputs) {
  using Storage = std::pair<size_t, size_t>;
  using Signature =
      std::tuple<size_t, size_t, size_t, NanScanKind, uint32_t, int64_t>;
  std::map<Storage, size_t> raw_arguments;
  std::map<Signature, std::vector<size_t>> groups;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& in = inputs[i];
    assert(in.bucket > 0 && (in.bucket & (in.bucket - 1)) == 0 &&
           "receive diagnostic chunk bucket must be a positive power of two");
    raw_arguments.try_emplace({in.storage, in.shard}, i);
    groups[{in.storage, in.shard, in.format.element_bytes,
            in.format.scan_kind, in.format.exponent_mask, in.bucket}]
        .push_back(i);
  }
  ScanBatch batch;
  for (auto& [storage, input] : raw_arguments) {
    batch.raw_inputs.push_back(input);
    input = batch.raw_inputs.size() - 1;
  }
  for (auto& [signature, indices] : groups) {
    const auto& in = inputs[indices.front()];
    batch.groups.push_back(
        {raw_arguments.at({in.storage, in.shard}), std::move(indices)});
  }
  return batch;
}
std::optional<NanFormat> NanFormatForPool(std::string_view tag,
                                       std::string_view dtype) {
  auto take_number = [](std::string_view* rest) {
    size_t n = 0;
    while (n < rest->size() && (*rest)[n] >= '0' && (*rest)[n] <= '9') ++n;
    rest->remove_prefix(n);
    return n != 0;
  };
  if (tag == "fa") return NanFormatForDtype(dtype);
  if (tag.starts_with("fa.l")) {
    tag.remove_prefix(4);
    return take_number(&tag) && tag.empty() ? NanFormatForDtype(dtype)
                                           : std::nullopt;
  }
  if (tag.starts_with("gdn.conv")) {
    tag.remove_prefix(8);
  } else if (tag.starts_with("gdn.ssm")) {
    tag.remove_prefix(7);
  } else {
    return std::nullopt;
  }
  if (tag.starts_with(".g")) {
    tag.remove_prefix(2);
    if (!take_number(&tag)) return std::nullopt;
  }
  if (tag.starts_with(".l")) {
    tag.remove_prefix(2);
    if (!take_number(&tag)) return std::nullopt;
  }
  return tag.empty() ? NanFormatForDtype(dtype) : std::nullopt;
}
std::vector<NanCheckChunk> BuildNanCheckChunks(
    int64_t block_base, std::span<const NanCheckRange> ranges,
    int64_t live_bytes) {
  std::vector<NanCheckChunk> chunks;
  assert(block_base >= 0 && live_bytes >= 0);
  for (const auto& range : ranges) {
    if (live_bytes == 0) break;
    const int64_t size = std::min(range.size, live_bytes);
    assert(range.offset >= 0 && size >= 0 &&
           block_base <= std::numeric_limits<int64_t>::max() - range.offset);
    int64_t begin = block_base + range.offset;
    assert(begin <= std::numeric_limits<int64_t>::max() - size);
    const int64_t end = begin + size;
    while (begin < end) {
      const int64_t word_begin = begin / 4;
      const int64_t count =
          std::min(end - begin, kNanCheckChunkBytes - (begin % 4));
      chunks.push_back({word_begin, begin, begin + count});
      begin += count;
    }
    live_bytes -= size;
  }
  assert(live_bytes == 0 && "received live prefix exceeds pool live regions");
  return chunks;
}
}  // namespace tpu_raiden

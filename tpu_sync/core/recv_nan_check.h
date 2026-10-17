// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0

#ifndef TPU_SYNC_CORE_RECV_NAN_CHECK_H_
#define TPU_SYNC_CORE_RECV_NAN_CHECK_H_

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tpu_sync::rpc {
class StartTransferRequest;
}

namespace tpu_raiden {
namespace kv_cache {
class KVCacheManagerBase;
}

struct NanFormat {
  size_t element_bytes;
  uint32_t exponent_mask;
  uint32_t fraction_mask;
};

inline std::optional<NanFormat> NanFormatForDtype(std::string_view dtype) {
  if (dtype == "bfloat16" || dtype == "bf16") {
    return NanFormat{2, 0x7f80, 0x007f};
  }
  if (dtype == "float16" || dtype == "f16") {
    return NanFormat{2, 0x7c00, 0x03ff};
  }
  if (dtype == "float32" || dtype == "f32") {
    return NanFormat{4, 0x7f800000, 0x007fffff};
  }
  return std::nullopt;
}

// Raw transfer bytes preserve the device's little-endian numeric bytes. Do not
// reinterpret a Python CPU carrier tensor: its logical lane order may differ.
inline std::optional<size_t> FindNanByteOffset(std::span<const std::byte> bytes,
                                            NanFormat format) {
  static_assert(std::endian::native == std::endian::little);
  assert(bytes.size() % format.element_bytes == 0 &&
         "received KV scan must contain whole numeric elements");
  for (size_t offset = 0; offset < bytes.size(); offset += format.element_bytes) {
    uint32_t bits = 0;
    std::memcpy(&bits, bytes.data() + offset, format.element_bytes);
    if ((bits & format.exponent_mask) == format.exponent_mask &&
        (bits & format.fraction_mask) != 0) {
      return offset;
    }
  }
  return std::nullopt;
}

struct NanCheckRange {
  size_t offset;
  size_t size;
};

// Ranges are sorted, disjoint live runs. live_bytes is a compact-live prefix,
// not a physical offset: padding consumes none of the prefix budget.
inline std::optional<size_t> FindNanInLivePrefix(
    std::span<const std::byte> block, std::span<const NanCheckRange> ranges,
    size_t live_bytes, NanFormat format) {
  for (const auto& range : ranges) {
    if (live_bytes == 0) break;
    const size_t size = std::min(range.size, live_bytes);
    assert(range.offset <= block.size() && size <= block.size() - range.offset &&
           "received KV live range must fit its pool block");
    if (auto offset = FindNanByteOffset(block.subspan(range.offset, size), format)) {
      return range.offset + *offset;
    }
    live_bytes -= size;
  }
  return std::nullopt;
}

// Opt-in diagnostic owned entirely by one receive session. It scans received
// host staging bytes before H2D while the session still owns its staging lease.
class RecvNanCheck {
 public:
  static std::unique_ptr<RecvNanCheck> CreateIfEnabled(
      kv_cache::KVCacheManagerBase* base,
      const ::tpu_sync::rpc::StartTransferRequest& plan);

  // Called only after this pool's wire transfer is complete and its H2D has
  // been claimed by the session's existing in-flight accounting. Read-only.
  void BeforeH2d(size_t pool_idx) const;

 private:
  struct Pool {
    NanFormat format;
    std::vector<int64_t> block_ids;
    std::vector<size_t> live_bytes;
    std::vector<NanCheckRange> ranges;
    size_t block_extent;
  };

  RecvNanCheck(kv_cache::KVCacheManagerBase* base, uint64_t uuid,
               std::string req_id)
      : base_(base), uuid_(uuid), req_id_(std::move(req_id)) {}

  kv_cache::KVCacheManagerBase* const base_;
  const uint64_t uuid_;
  const std::string req_id_;
  std::unordered_map<size_t, Pool> pools_;
};

}  // namespace tpu_raiden

#endif  // TPU_SYNC_CORE_RECV_NAN_CHECK_H_

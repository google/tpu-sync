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

#include "tpu_sync/kv_cache/host_meminfo.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {
constexpr absl::string_view kMemFreeKey = "MemFree";
constexpr absl::string_view kCachedKey = "Cached";
constexpr absl::string_view kAnonPagesKey = "AnonPages";
constexpr absl::string_view kBuffersKey = "Buffers";

// Parses meminfo lines of the form "<Key>: <value> kB" and returns each value
// in KiB keyed by field name. Lines with a malformed value are skipped.
absl::flat_hash_map<absl::string_view, uint64_t> ParseMeminfoKiB(
    absl::string_view meminfo_content) {
  absl::flat_hash_map<absl::string_view, uint64_t> fields;
  for (absl::string_view line :
       absl::StrSplit(meminfo_content, '\n', absl::SkipEmpty())) {
    const size_t colon_pos = line.find(':');
    if (colon_pos == absl::string_view::npos) {
      continue;
    }
    const absl::string_view key =
        absl::StripAsciiWhitespace(line.substr(0, colon_pos));
    absl::string_view value =
        absl::StripLeadingAsciiWhitespace(line.substr(colon_pos + 1));
    value = value.substr(0, value.find_first_of(" \t"));
    uint64_t kib = 0;
    if (!absl::SimpleAtoi(value, &kib)) {
      continue;
    }
    fields[key] = kib;
  }
  return fields;
}

// Converts KiB to bytes, saturating at the uint64_t maximum.
uint64_t KiBToBytes(uint64_t kib) {
  constexpr uint64_t kBytesPerKiB = 1024ULL;
  if (kib > std::numeric_limits<uint64_t>::max() / kBytesPerKiB) {
    return std::numeric_limits<uint64_t>::max();
  }
  return kib * kBytesPerKiB;
}

}  // namespace

std::optional<HostMemoryBreakdown> ParseHostMemoryBreakdown(
    absl::string_view meminfo_content, uint64_t kv_pool_bytes) {
  const absl::flat_hash_map<absl::string_view, uint64_t> fields_kib =
      ParseMeminfoKiB(meminfo_content);
  const auto mem_free_it = fields_kib.find(kMemFreeKey);
  const auto cached_it = fields_kib.find(kCachedKey);
  const auto anon_pages_it = fields_kib.find(kAnonPagesKey);
  if (mem_free_it == fields_kib.end() || cached_it == fields_kib.end() ||
      anon_pages_it == fields_kib.end()) {
    return std::nullopt;
  }
  const auto buffers_it = fields_kib.find(kBuffersKey);

  const uint64_t free_bytes = KiBToBytes(mem_free_it->second);
  const uint64_t buffers_bytes =
      buffers_it == fields_kib.end() ? 0ULL : KiBToBytes(buffers_it->second);
  const uint64_t cached_bytes = KiBToBytes(cached_it->second);
  const uint64_t page_cache_bytes =
      buffers_bytes > std::numeric_limits<uint64_t>::max() - cached_bytes
          ? std::numeric_limits<uint64_t>::max()
          : buffers_bytes + cached_bytes;
  const uint64_t anon_bytes = KiBToBytes(anon_pages_it->second);
  // Note: kv_pool_bytes currently reflects rank 0's anonymous staging
  // allocation. For POSIX shared-memory pools (/dev/shm via MapSharedMemory),
  // Linux accounts tmpfs pages under Cached/Shmem rather than AnonPages.
  const uint64_t process_heap_bytes =
      anon_bytes > kv_pool_bytes ? anon_bytes - kv_pool_bytes : 0ULL;

  return HostMemoryBreakdown{
      .kv_pool_bytes = kv_pool_bytes,
      .process_heap_bytes = process_heap_bytes,
      .page_cache_bytes = page_cache_bytes,
      .free_bytes = free_bytes,
  };
}

}  // namespace kv_cache
}  // namespace tpu_raiden

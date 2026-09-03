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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_

#include <dirent.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "tpu_sync/core/host_memory_allocator.h"

namespace tpu_raiden {

inline std::optional<std::vector<const uint8_t*>> CastExternalPointers(
    const std::optional<std::vector<uintptr_t>>& external_host_ptrs) {
  if (!external_host_ptrs.has_value()) return std::nullopt;
  std::vector<const uint8_t*> cast_ptrs;
  cast_ptrs.reserve(external_host_ptrs->size());
  for (uintptr_t addr : *external_host_ptrs) {
    cast_ptrs.push_back(reinterpret_cast<const uint8_t*>(addr));
  }
  return cast_ptrs;
}

// Scans /dev/shm for segments that carry `shm_key`'s prefix but are not in
// `expected_segment_names` (this process's own segments), and surfaces each
// one -- nothing is removed, because liveness cannot be proven from here: a
// crashed run's segment looks identical to a live sibling's. A name
// matching this job's per-device naming ("<base>_dev_<N>") is most likely a
// same-host sibling rank's segment, and a "<key>_metadata..." name is a
// store's metadata table (this process's own or a colocated store's); both
// are logged at INFO. Anything else does not belong to a running shape of
// this job and is logged at WARNING as a possibly stale leftover. ".lock"
// companion files are skipped. Returns the WARNING-class names (without the
// leading '/'), so tests can observe what was flagged.
inline std::vector<std::string> WarnAboutOrphanShmSegments(
    absl::string_view shm_key,
    const std::vector<std::string>& expected_segment_names) {
  std::vector<std::string> stale_suspects;
  const absl::string_view prefix = absl::StripPrefix(shm_key, "/");
  if (prefix.empty()) {
    return stale_suspects;
  }
  // Directory entries carry no leading '/'; compare stripped.
  std::vector<std::string> expected;
  expected.reserve(expected_segment_names.size());
  for (const std::string& name : expected_segment_names) {
    expected.emplace_back(absl::StripPrefix(name, "/"));
  }
  // This job's per-device names look like "<base>_dev_<N>"; an unexpected
  // one is most likely a same-host sibling rank's segment.
  const std::string sibling_prefix = absl::StrCat(
      absl::StripPrefix(
          SharedMemoryHostMemoryAllocator::ComposeSegmentName(shm_key,
                                                              std::nullopt),
          "/"),
      "_dev_");
  const std::string metadata_prefix = absl::StrCat(prefix, "_metadata");
  DIR* dir = opendir("/dev/shm");
  if (dir == nullptr) {
    // No scannable shm directory on this system; nothing to surface.
    return stale_suspects;
  }
  auto close_dir = absl::MakeCleanup([dir]() { closedir(dir); });
  while (const dirent* entry = readdir(dir)) {
    const absl::string_view name(entry->d_name);
    if (!absl::StartsWith(name, prefix)) continue;
    if (absl::EndsWith(name, ".lock")) continue;
    if (std::find(expected.begin(), expected.end(), name) != expected.end()) {
      continue;
    }
    const absl::string_view device_suffix =
        absl::StartsWith(name, sibling_prefix)
            ? name.substr(sibling_prefix.size())
            : absl::string_view();
    if (!device_suffix.empty() &&
        std::all_of(device_suffix.begin(), device_suffix.end(),
                    absl::ascii_isdigit)) {
      LOG(INFO) << "[SHM_ALLOCATOR] /dev/shm/" << name
                << " matches this job's per-device segment naming but is not "
                << "one of this manager's segments; likely a sibling rank's "
                << "segment on this host (stale only if no such rank is "
                << "running).";
      continue;
    }
    if (absl::StartsWith(name, metadata_prefix)) {
      LOG(INFO) << "[SHM_ALLOCATOR] /dev/shm/" << name
                << " is a KV metadata table under this key -- this store's "
                << "own or a colocated store's.";
      continue;
    }
    LOG(WARNING) << "[SHM_ALLOCATOR] /dev/shm/" << name << " carries the "
                 << "shm key prefix but does not match this job's segment "
                 << "naming; possibly a stale leftover. /dev/shm is never "
                 << "cleaned automatically.";
    stale_suspects.push_back(std::string(name));
  }
  return stale_suspects;
}

// Picks the host buffer allocator for a KVCacheManager. Shared memory is
// used only when the manager opted in (enable_shm) AND the deployment
// provides a segment namespace (RAIDEN_SHM_KEY); managers whose host buffers
// are transient staging (transfer engines) must not use shm.
//
// num_host_blocks is the resolved host pool block count -- the same
// host_blocks_to_allocate-or-num_slots*max_blocks resolution the manager
// applies -- and feeds the segment identity, so a pool resize is detected
// as an incompatible segment instead of warm-attaching at the wrong stride.
// total_payload_bytes only seeds the expected-schema field of that name,
// which the allocator owns (cumulative payload) and never validates.
inline HostBufferAllocator CreateHostMemoryAllocator(
    xla::PjRtClient* client, bool enable_shm, int64_t num_host_blocks = 0,
    size_t total_payload_bytes = 0) {
  const char* shm_key_env = std::getenv("RAIDEN_SHM_KEY");
  if (enable_shm && shm_key_env != nullptr && std::strlen(shm_key_env) > 0) {
    SharedMemoryHeader expected_schema;
    const char* model_uid_env = std::getenv("RAIDEN_SHM_MODEL_UID");
    if (model_uid_env != nullptr) {
      absl::SNPrintF(expected_schema.model_uid,
                     sizeof(expected_schema.model_uid), "%s", model_uid_env);
    } else {
      absl::SNPrintF(expected_schema.model_uid,
                     sizeof(expected_schema.model_uid), "default_model");
    }
    expected_schema.num_blocks = num_host_blocks;
    expected_schema.total_payload_bytes = total_payload_bytes;

    auto allocator_or = SharedMemoryHostMemoryAllocator::Create(
        client, shm_key_env, expected_schema);
    if (!allocator_or.ok()) {
      absl::Status status = allocator_or.status();
      return [status](size_t size_bytes, const xla::PjRtDevice* device)
                 -> absl::StatusOr<HostBufferAllocation> { return status; };
    }
    // The segment names this process can own are one per addressable device
    // (plus the no-device name used without a device context); classification
    // is by name alone, so scanning before any segment is created is safe.
    std::vector<std::string> expected_segments;
    if (client != nullptr) {
      for (const xla::PjRtDevice* device : client->addressable_devices()) {
        expected_segments.push_back(
            SharedMemoryHostMemoryAllocator::ComposeSegmentName(
                shm_key_env, device->global_device_id().value()));
      }
    }
    expected_segments.push_back(
        SharedMemoryHostMemoryAllocator::ComposeSegmentName(shm_key_env,
                                                            std::nullopt));
    (void)WarnAboutOrphanShmSegments(shm_key_env, expected_segments);
    std::shared_ptr<HostMemoryAllocator> allocator =
        std::move(allocator_or).value();
    return [allocator](size_t size_bytes, const xla::PjRtDevice* device)
               -> absl::StatusOr<HostBufferAllocation> {
      return allocator->AllocateDmaMappedForDevice(size_bytes, device);
    };
  }

  auto allocator_or = HostMemoryAllocator::Create(client);
  if (!allocator_or.ok()) {
    absl::Status status = allocator_or.status();
    return [status](size_t size_bytes, const xla::PjRtDevice* device)
               -> absl::StatusOr<HostBufferAllocation> { return status; };
  }
  std::shared_ptr<HostMemoryAllocator> allocator =
      std::move(allocator_or).value();
  return [allocator](size_t size_bytes, const xla::PjRtDevice* device)
             -> absl::StatusOr<HostBufferAllocation> {
    return allocator->AllocateDmaMappedForDevice(size_bytes, device);
  };
}

inline HostBufferAllocator CreateHostMemoryAllocator(xla::PjRtClient* client) {
  return CreateHostMemoryAllocator(client, /*enable_shm=*/false);
}

struct RawCopyChunk {
  int64_t src_offset;
  int64_t dst_offset;
  int64_t size_bytes;
};

inline void ValidatePartialSpec(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  bool present = !src_offsets_major_dim.empty() ||
                 !dst_offsets_major_dim.empty() ||
                 !copy_sizes_major_dim.empty();
  if (present &&
      (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
       src_offsets_major_dim.size() != copy_sizes_major_dim.size())) {
    throw std::invalid_argument(
        "src_offsets_major_dim, dst_offsets_major_dim, and "
        "copy_sizes_major_dim must have the same length");
  }
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] < 0 || dst_offsets_major_dim[i] < 0 ||
        copy_sizes_major_dim[i] < 0) {
      throw std::invalid_argument(
          "raw copy offsets and sizes must be non-negative");
    }
  }
}

inline bool IsPartialCopy(const xla::Shape& shape,
                          const std::vector<int64_t>& src_offsets_major_dim,
                          const std::vector<int64_t>& dst_offsets_major_dim,
                          const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_offsets_major_dim.empty()) return false;
  if (shape.dimensions().empty()) return true;
  const int64_t full_major_dim = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim) {
      return true;
    }
  }
  return false;
}

inline void ValidatePartialAlignment(const xla::Shape& shape,
                                     int64_t slice_byte_size) {
  if (shape.dimensions().size() < 3) {
    throw std::invalid_argument(
        "Only rank >= 3 TPU tensors support partial raw copies");
  }
  if (slice_byte_size % 4096 != 0) {
    throw std::invalid_argument(
        "Partial raw copies require a major-dimension slice size aligned to "
        "4096 bytes");
  }
}

inline std::vector<RawCopyChunk> ComputeAndValidateChunks(
    int64_t slice_byte_size, int64_t physical_size, int64_t max_cpu_size,
    bool is_partial, const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim, bool is_d2h) {
  std::vector<RawCopyChunk> chunks;
  if (!is_partial) {
    if (is_d2h) {
      if (max_cpu_size < physical_size) {
        throw std::invalid_argument("Destination CPU tensor is too small");
      }
    } else {
      if (max_cpu_size < physical_size) {
        throw std::invalid_argument("Source CPU tensor is too small");
      }
    }
    chunks.push_back({0, 0, physical_size});
  } else {
    chunks.reserve(src_offsets_major_dim.size());
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      const int64_t src_offset = src_offsets_major_dim[i] * slice_byte_size;
      const int64_t dst_offset = dst_offsets_major_dim[i] * slice_byte_size;
      const int64_t size_to_copy = copy_sizes_major_dim[i] * slice_byte_size;
      if (is_d2h) {
        if (src_offset + size_to_copy > physical_size) {
          throw std::invalid_argument("Copy range exceeds source TPU buffer");
        }
        if (dst_offset + size_to_copy > max_cpu_size) {
          throw std::invalid_argument(
              "Copy range exceeds destination CPU tensor");
        }
      } else {
        if (src_offset + size_to_copy > max_cpu_size) {
          throw std::invalid_argument("Copy range exceeds source CPU tensor");
        }
        if (dst_offset + size_to_copy > physical_size) {
          throw std::invalid_argument(
              "Copy range exceeds destination TPU buffer");
        }
      }
      chunks.push_back({src_offset, dst_offset, size_to_copy});
    }
  }
  return chunks;
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_

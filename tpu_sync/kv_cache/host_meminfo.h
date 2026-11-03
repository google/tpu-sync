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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_HOST_MEMINFO_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_HOST_MEMINFO_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace tpu_raiden {
namespace kv_cache {

// Host DRAM memory breakdown in bytes across the four components tracked by
// `tpu_raiden_host_memory_breakdown_bytes`:
//   - kv_pool_bytes: bytes allocated to the primary worker's (rank 0)
//     anonymous host DRAM staging allocation (`allocated_host_dram_bytes_`).
//     Note: for POSIX shared-memory pools (`/dev/shm`), backing pages are
//     accounted by Linux under `Cached`/`Shmem` rather than `AnonPages`, which
//     may be addressed in a follow-up.
//   - process_heap_bytes: non-KV-pool anonymous/heap memory (`AnonPages` minus
//     `kv_pool_bytes`, clamped to >= 0).
//   - page_cache_bytes: OS file-backed page cache (`Buffers + Cached`).
//   - free_bytes: free host DRAM (`MemFree`).
struct HostMemoryBreakdown {
  uint64_t kv_pool_bytes = 0;
  uint64_t process_heap_bytes = 0;
  uint64_t page_cache_bytes = 0;
  uint64_t free_bytes = 0;
};

// Parses `/proc/meminfo` text content and computes the 4-component host DRAM
// breakdown given `kv_pool_bytes`. Returns `std::nullopt` if required fields
// (`MemFree`, `Cached`, `AnonPages`) are missing or malformed.
std::optional<HostMemoryBreakdown> ParseHostMemoryBreakdown(
    absl::string_view meminfo_content, uint64_t kv_pool_bytes);

// Per-NUMA-node host DRAM memory in bytes across states (used, free, total)
// tracked by `tpu_raiden_host_numa_memory_bytes`.
struct NumaNodeMemory {
  int node_id = -1;
  uint64_t used_bytes = 0;
  uint64_t free_bytes = 0;
  uint64_t total_bytes = 0;
};

// Parses a NUMA node's meminfo text content (e.g. from
// `/sys/devices/system/node/node<N>/meminfo`) and computes used, free,
// and total bytes. If `MemUsed` is absent, falls back to `MemTotal - MemFree`
// (clamped to >= 0). Returns `std::nullopt` if required fields (`MemTotal`,
// `MemFree`) are missing or malformed.
std::optional<NumaNodeMemory> ParseNumaNodeMeminfo(
    int node_id, absl::string_view meminfo_content);

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_HOST_MEMINFO_H_

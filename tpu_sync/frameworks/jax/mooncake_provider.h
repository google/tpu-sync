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

// tpu_raiden's TPU buffer provider for Mooncake TENT.
//
// Mooncake stages TPU data through host DRAM. Its adapter cannot reach JAX
// device buffers itself, so it dlopen()s this library, fetches the C table in
// third_party/mooncake/tpu_buffer_provider.h through
// mc_tpu_buffer_provider_get(), and asks it to copy byte ranges between a
// buffer and host memory. The Python side
// (tpu_sync/api/jax/mooncake_provider.py) turns the shards of a jax.Array into
// the handles that table operates on.
//
// Lifetime. A handle owns one strong reference to the shard's PJRT raw buffer,
// and the raw buffer owns the device allocation. While the handle is live the
// HBM therefore stays allocated and every copy is memory-safe, even if Python
// drops or deletes the array. The handle holds no reference to the Python
// object and no PJRT usage hold, so releasing it never touches the GIL and
// never blocks JAX. The cost is that nothing orders the copies with JAX:
//   * The caller must keep JAX from writing the buffer while Mooncake copies
//     it, as for KVCacheManager's `unsafe_skip_buffer_lock`.
//   * Donating an exported array is unsupported: XLA may reuse the
//     allocation for the donated computation's output and race the copies.
//   * After the array is deleted, copies still reach the original
//     allocation, which JAX no longer sees.
//   * Handles must be released before the JAX backend that owns their
//     buffers is destroyed. Handles still live at process exit are leaked,
//     never released.
//
// Alignment. TPU DMA needs both the host pointer and the device offset to be
// 32-byte aligned; the length may be anything (measured on TPU v6e and v7x).
// The provider stages copies through misaligned host pointers via an aligned
// bounce buffer, at the cost of a memcpy. A misaligned device offset fails
// with the driver's error in both directions: a write would have to merge
// into a 32-byte granule that a concurrent copy may be writing, and the two
// ends of a transfer use the same offsets, so reads gain nothing from
// accepting what writes cannot. Callers must transfer ranges that start at a
// multiple of 32 bytes.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_MOONCAKE_PROVIDER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_MOONCAKE_PROVIDER_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "third_party/mooncake/tpu_buffer_provider.h"
#include "tpu_sync/core/raw_transfer_core.h"

// The entry point Mooncake looks up by name (MC_TPU_BUFFER_PROVIDER_ENTRY).
// Serves ABI version 1 only: fills `*out` and returns 0 for version 1, and
// returns non-zero for any other version or a null `out`.
extern "C" __attribute__((visibility("default"))) int
mc_tpu_buffer_provider_get(uint32_t requested_version,
                           mc_tpu_buffer_provider_t* out);

namespace raiden::mooncake {

// One device buffer to export, and the device ordinal its handle reports as
// `device_index` (TENT's "tpu:N").
struct ExportRequest {
  RaidenBufferHandle buffer;
  int device_index;
};

// Turns each request into a provider handle, in order. Each handle takes over
// its request's buffer reference and caches the buffer's size and NUMA node.
// The caller owns the returned handles until it passes them to Mooncake.
// All-or-nothing: returns InvalidArgument, and creates no handle, if any
// request holds no buffer or has a negative device index.
absl::StatusOr<std::vector<mc_tpu_buffer_t>> ExportBuffers(
    std::vector<ExportRequest> requests);

// Number of handles created by ExportBuffers and not yet released.
int64_t LiveBufferCount();

}  // namespace raiden::mooncake

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_MOONCAKE_PROVIDER_H_

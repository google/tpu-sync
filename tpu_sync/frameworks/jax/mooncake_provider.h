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
// device buffers itself, so it dlopen()s this library, fetches a C function
// table (ProviderTable below) through mc_tpu_buffer_provider_get(), and asks
// it to copy byte ranges between a buffer and host memory. The Python side
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
// DMA mapping. TENT registers each of its host staging pools with the TPU DMA
// engine through dma_map, so that copies into and out of the pool run as
// direct DMA. dma_map maps with the PJRT client of the most recently exported
// buffers (JAX has one TPU client per process), so it fails until something
// has been exported, and that JAX backend must be alive when Mooncake maps.
// Each mapping belongs to the client that made it and must be undone before
// that backend is destroyed; Mooncake undoes its mappings when its
// TransferEngine is destroyed. JAX's own atexit handler may destroy the
// backend, so once interpreter exit begins (MarkProcessExiting), dma_map fails
// and dma_unmap skips PJRT, leaking the mapping to the exiting process.
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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace raiden::mooncake {

// --- Mooncake's TPU buffer provider ABI, version 2 --------------------------
// Mooncake owns this contract: it is defined by
// mooncake-transfer-engine/tent/include/tent/platform/tpu_buffer_provider.h in
// https://github.com/kvcache-ai/Mooncake. These declarations match it field
// for field; the static_asserts in mooncake_provider.cc pin their binary
// layout. A new ABI version from Mooncake means updating both.
// TODO(wangziyi): Name the Mooncake commit that adds version 2 once it lands.

inline constexpr uint32_t kAbiVersion = 2;

// An opaque handle to one exported device buffer. 0 is never a valid handle.
using BufferHandle = uint64_t;

// The function table Mooncake calls. Every function may run concurrently on
// many threads, including for the same handle.
struct ProviderTable {
  uint32_t abi_version;
  const char* name;
  // Bytes in the buffer's on-device (possibly tiled) representation; 0 for
  // handle 0.
  size_t (*size)(BufferHandle handle);
  // The device_index given to ExportBuffers (TENT's "tpu:N"); -1 for handle 0.
  int (*device_index)(BufferHandle handle);
  // NUMA node of the buffer's device; -1 if unknown or for handle 0.
  int (*numa_node)(BufferHandle handle);
  // Blocking copies of bytes [offset, offset + len) of the buffer. Return 0 on
  // success and non-zero on failure.
  int (*copy_to_host)(BufferHandle handle, size_t offset, void* host_dst,
                      size_t len);
  int (*copy_from_host)(BufferHandle handle, size_t offset,
                        const void* host_src, size_t len);
  // Frees the handle. Mooncake calls it exactly once per handle.
  void (*release)(BufferHandle handle);

  // Version 2 appends these to the version 1 table.

  // Registers host memory [host, host + len) with the TPU DMA engine
  // (PjRtClient::DmaMap), so that copies into and out of it run as direct DMA.
  // TENT maps each host staging pool it allocates: `host` and `len` are
  // multiples of the page size, and the range overlaps no live mapping. Return
  // 0 on success and non-zero on failure, after which TENT uses the memory
  // unmapped, only with slower copies.
  int (*dma_map)(void* host, size_t len);
  // Undoes the dma_map that succeeded for the range starting at `host`. TENT
  // calls it when it releases the pool, after every copy into or out of it has
  // returned. Return 0 on success and non-zero on failure.
  int (*dma_unmap)(void* host);
};

// --- tpu_raiden side --------------------------------------------------------

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
// request holds no buffer or has a negative device index. A successful export
// also makes its buffers' PJRT client the one that dma_map maps with.
absl::StatusOr<std::vector<BufferHandle>> ExportBuffers(
    std::vector<ExportRequest> requests);

// Number of handles created by ExportBuffers and not yet released.
int64_t LiveBufferCount();

// Number of host ranges mapped by dma_map and not yet unmapped.
int64_t LiveDmaMappingCount();

// Declares that interpreter exit has begun, so the PJRT client may be
// destroyed at any moment: from now on dma_map fails, and dma_unmap forgets
// the mapping without calling PJRT. When this returns, no PJRT call from either
// is still running. tpu_sync/api/jax/mooncake_provider.py registers it with
// atexit, to run before JAX's own atexit handler.
void MarkProcessExiting();

}  // namespace raiden::mooncake

// The entry point Mooncake looks up by this exact name. Serves ABI versions 1
// and 2: writes the table of `requested_version` to `*out`, with `abi_version`
// set to it, and returns 0. A version 1 table is the part of ProviderTable
// before `dma_map`, and nothing past it is written, since a caller built
// against version 1 passes a table only that large. Returns non-zero, writing
// nothing, for any other version or a null `out`.
extern "C" __attribute__((visibility("default"))) int
mc_tpu_buffer_provider_get(uint32_t requested_version,
                           raiden::mooncake::ProviderTable* out);

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_MOONCAKE_PROVIDER_H_

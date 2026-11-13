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
// Mooncake stages TPU data through host DRAM. Its adapter cannot reach a
// framework's device buffers itself, so it dlopen()s the extension that
// exported them, fetches a C function table (ProviderTable below) through
// mc_tpu_buffer_provider_get(), and asks it to copy byte ranges between a
// buffer and host memory. A Python module per framework turns its arrays into
// the handles that table operates on:
// tpu_sync/solutions/mooncake/jax/mooncake_provider.py the shards of a
// jax.Array, tpu_sync/solutions/mooncake/torch/mooncake_provider.py torch
// tensors.
//
// Lifetime. A handle owns one strong reference to its buffer's PJRT raw
// buffer, which owns the device allocation, and its export request's
// keep_alive: for a torch tensor, TorchTPU's buffer, which also keeps
// TorchTPU's PJRT client alive. While the handle is live the HBM therefore
// stays allocated and every copy is memory-safe, even if Python drops or
// deletes the array. The handle holds no reference to a Python object and no
// PJRT usage hold, so releasing it never touches the GIL and never blocks the
// framework. The cost is that nothing orders the copies with the framework:
//   * The caller must keep the framework from writing the buffer while
//     Mooncake copies out of it, and from using it while Mooncake copies into
//     it, as for KVCacheManager's `unsafe_skip_buffer_lock`.
//   * Donating an exported jax.Array is unsupported: XLA may reuse the
//     allocation for the donated computation's output and race the copies.
//   * TorchTPU applies some in-place ops in a torch tensor's buffer, where the
//     handle sees their results, and for others moves the tensor to a new
//     buffer, which the handle does not see
//     (tpu_sync/solutions/mooncake/torch/mooncake_provider.py).
//   * After the array is deleted, copies still reach the original
//     allocation, which the framework no longer sees.
//   * Handles must be released before the PJRT client that owns their
//     buffers is destroyed. Handles still live at process exit are leaked,
//     never released.
//
// DMA mapping. TENT registers each of its host staging pools with the TPU DMA
// engine through dma_map, so that copies into and out of the pool run as
// direct DMA. dma_map maps with the PJRT client of the most recently exported
// buffers (JAX and TorchTPU each have one TPU client per process), so it fails
// until something has been exported, and that client must be alive when
// Mooncake maps. Each mapping belongs to the client that made it and must be
// undone before that client is destroyed; Mooncake undoes its mappings when
// its TransferEngine is destroyed. The framework's own atexit handler may
// destroy the client, so once interpreter exit begins (MarkProcessExiting),
// dma_map fails and dma_unmap skips PJRT, leaking the mapping to the exiting
// process.
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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_SOLUTIONS_MOONCAKE_MOONCAKE_PROVIDER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_SOLUTIONS_MOONCAKE_MOONCAKE_PROVIDER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace raiden::mooncake {

// --- Mooncake's TPU buffer provider ABI -------------------------------------
// Mooncake owns this contract: it is defined by
// mooncake-transfer-engine/tent/include/tent/platform/tpu_buffer_provider.h in
// Mooncake. These declarations must match it field for field, and the
// static_asserts in mooncake_provider.cc pin their binary layout.

// An opaque handle to one exported device buffer. 0 is never a valid handle.
using BufferHandle = uint64_t;

// The function table Mooncake calls. Every function may run concurrently on
// many threads, including for the same handle.
struct ProviderTable {
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

// One device buffer to export, the device ordinal its handle reports as
// `device_index` (TENT's "tpu:N"), and anything else that must stay alive while
// the handle uses the buffer, such as the framework object that owns it.
struct ExportRequest {
  RaidenBufferHandle buffer;
  int device_index;
  std::shared_ptr<void> keep_alive;
};

// Turns each request into a provider handle, in order. Each handle takes over
// its request's buffer reference and keep_alive, and caches the buffer's size
// and NUMA node. The caller owns the returned handles until it passes them to
// Mooncake. All-or-nothing: returns InvalidArgument, and creates no handle, if
// any request holds no buffer or has a negative device index. A successful
// export also makes its buffers' PJRT client the one that dma_map maps with.
absl::StatusOr<std::vector<BufferHandle>> ExportBuffers(
    std::vector<ExportRequest> requests);

// Number of handles created by ExportBuffers and not yet released.
int64_t LiveBufferCount();

// Number of host ranges mapped by dma_map and not yet unmapped.
int64_t LiveDmaMappingCount();

// Declares that interpreter exit has begun, so the PJRT client may be
// destroyed at any moment: from now on dma_map fails, and dma_unmap forgets
// the mapping without calling PJRT. When this returns, no PJRT call from either
// is still running. Each framework's Python module registers it with atexit,
// to run before the framework's own atexit handler.
void MarkProcessExiting();

// Writes the table to `*out` and returns 0, or returns non-zero if `out` is
// null. mc_tpu_buffer_provider_get() returns this.
int GetProviderTable(ProviderTable* out);

}  // namespace raiden::mooncake

// The entry point Mooncake looks up by this exact name, in the shared object
// of the extension that exported the handles. Each such extension defines it
// in its own sources, as `return raiden::mooncake::GetProviderTable(out);`.
// Defined in a dependency, it could be missing from that shared object:
// google3 links an extension's dependencies into the Python binary, and
// -Wl,--exclude-libs,ALL hides their symbols.
extern "C" __attribute__((visibility("default"))) int
mc_tpu_buffer_provider_get(raiden::mooncake::ProviderTable* out);

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_SOLUTIONS_MOONCAKE_MOONCAKE_PROVIDER_H_

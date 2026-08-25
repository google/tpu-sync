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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_DELEGATE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_DELEGATE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"

namespace tpu_raiden {
namespace transport {

// Represents a contiguous span of memory within a block transport operation.
struct BlockChunk {
  // Pointer to the local host memory address of this chunk.
  uint8_t* ptr;
  // Length of this memory chunk in bytes.
  size_t size;
};

// Receiver-side expectation contract for one plan-declared (pool-keyed)
// transfer. The transport keeps a single progress map for every push; a
// registered plan supplies this spec and switches the expectation source for
// its uuid from the wire header's per-push parallelism to the plan's global
// per-pool push count. expected_pools retires all progress records for the
// uuid once the final declared pool completes.
struct PoolPushProgressSpec {
  size_t expected_pushes = 0;
  size_t expected_pools = 0;
};

// Delegate interface for BlockTransport inheriting raw memory primitives.
class BlockTransportDelegate : public lib::RawBufferTransportDelegate {
 public:
  ~BlockTransportDelegate() override = default;

  // Explicit-destination (op=6) pushes normally ride a registered transfer
  // plan. Legacy uniform-layer delegates accept plan-less explicit pushes
  // (the ad-hoc H2H contract); pool-mode delegates reject them so that a
  // mis-addressed push cannot resolve chunks against a bystander pool's
  // host mirror.
  virtual bool AcceptsPlanlessExplicitPush(uint64_t uuid) const { return true; }

  // The transport address space is historically one block array per manager
  // layer. Explicit pool tables widen that address space to one block array
  // per pool without changing the wire's integer index.
  virtual size_t num_block_arrays() const { return num_layers(); }

  // Geometry and host bounds for one wire-addressable block array. Legacy
  // delegates inherit the uniform layer behavior; pool-aware delegates may
  // map an array to an interior range of a different backing storage.
  virtual size_t block_bytes(size_t block_array_idx) const {
    return bytes_per_block();
  }
  virtual uint8_t* GetBlockArrayHostPointer(size_t block_array_idx,
                                            size_t shard_idx) {
    return GetHostPointer(block_array_idx, shard_idx);
  }
  virtual size_t GetBlockArrayHostSize(size_t block_array_idx,
                                       size_t shard_idx) {
    return GetHostSize(block_array_idx, shard_idx);
  }

  // Returns the active node ID (rank) of the worker.
  virtual int64_t node_id() const { return -1; }

  // Returns the list of contiguous chunks that constitute a block range
  // for a specific transaction (identified by uuid).
  // Optionally accepts the sender_node_id to distinguish senders in many-to-one
  // transfers.
  // If `src_block_id` is provided (not -1), it is used to resolve correct chunk
  // offsets when multiple source blocks merge into a single destination block
  // (heterogeneous block sizes). If `dst_block_id` is provided (not -1), the
  // sender resolution is restricted to that destination block. This is needed
  // when one source block fans out to multiple blocks on the same peer.
  virtual std::vector<BlockChunk> GetBlockChunks(
      size_t layer_idx, size_t shard_idx, absl::Span<const int64_t> block_ids,
      size_t total_bytes, uint64_t uuid, int64_t sender_node_id = -1,
      absl::string_view peer = "", int64_t src_block_id = -1,
      int64_t dst_block_id = -1) {
    // Default implementation: blocks are contiguous and of uniform size.
    std::vector<BlockChunk> result;
    size_t accumulated_bytes = 0;
    for (int64_t block_id : block_ids) {
      if (accumulated_bytes >= total_bytes) break;
      size_t size =
          std::min(block_bytes(layer_idx), total_bytes - accumulated_bytes);
      result.push_back(
          {GetBlockHostPointer(layer_idx, shard_idx, block_id), size});
      accumulated_bytes += size;
    }
    return result;
  }

  // Brackets one incoming payload's use of the chunk pointers GetBlockChunks
  // resolves for it: Begin runs before resolution, End after the payload's
  // last byte has landed or its stream has failed. A delegate that recycles
  // the memory behind its chunks keeps it alive between the two. End receives
  // Begin's return value unchanged.
  virtual uint64_t BeginPayloadResolution(uint64_t uuid) { return 0; }
  virtual void EndPayloadResolution(uint64_t uuid, uint64_t token) {}

  virtual absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) = 0;

  virtual absl::Status OnLayerReceived(size_t layer_idx, uint64_t uuid) {
    return absl::OkStatus();
  }

  // Resolves the receive-progress expectation source for one wire-addressed
  // block array. A registered pool plan returns its contract (plan-declared
  // mode); nullopt selects the legacy header-declared mode; an error rejects
  // the push before any payload is read (e.g. a pool outside the plan's
  // transfer set).
  virtual absl::StatusOr<std::optional<PoolPushProgressSpec>>
  GetPoolPushProgressSpec(size_t pool_idx, uint64_t uuid) const {
    return std::nullopt;
  }

  // Number of distinct senders whose pushes assemble each block array of
  // `uuid` on this receiver, when a registered receive plan declares them
  // (e.g. several source ranks each writing a head slice of the same block).
  // A block array is complete, and OnLayerReceived fires, only once every
  // declared sender's streams for it have landed. nullopt selects the
  // header-declared contract: one push's own parallelism completes the array.
  virtual std::optional<size_t> ExpectedPushSenders(uint64_t uuid) const {
    return std::nullopt;
  }

  // Pool-keyed counterpart to OnLayerReceived, fired when a declared pool
  // reaches its plan-declared push count. An explicit-pool index is not
  // necessarily a constructor layer/storage index, so the default is a no-op.
  virtual absl::Status OnPoolReceived(size_t pool_idx, uint64_t uuid) {
    return absl::OkStatus();
  }

  virtual absl::Status OnBlocksReceived(const std::vector<int>& block_ids,
                                        uint64_t uuid = 0) {
    return OnDataReceived();
  }

  virtual absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) {
    return OnDataReceived();
  }

  using HostBlockReadyCallback = std::function<void(absl::Status)>;
  virtual void RegisterBlockReadinessCallback(size_t layer_idx,
                                              size_t shard_idx, int block_id,
                                              uint64_t uuid,
                                              HostBlockReadyCallback cb) {
    cb(absl::OkStatus());
  }

  virtual void ScheduleAsyncTask(std::function<void()> task) {
    std::thread(std::move(task)).detach();
  }

  virtual uint8_t* GetBlockHostPointer(size_t layer_idx, size_t shard_idx,
                                       int block_id) {
    uint8_t* base = GetBlockArrayHostPointer(layer_idx, shard_idx);
    if (base == nullptr || block_id < 0) {
      return nullptr;
    }
    return base + static_cast<size_t>(block_id) * block_bytes(layer_idx);
  }

  virtual size_t bytes_per_block() const { return slice_byte_size(); }

  virtual int GetRemoteReadBlockId(int base_remote_id, int chunk_k) = 0;

  virtual size_t num_layers() const = 0;
  virtual size_t num_shards() const = 0;
  virtual size_t slice_byte_size() const = 0;
  virtual size_t shard_factor() const = 0;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_DELEGATE_H_

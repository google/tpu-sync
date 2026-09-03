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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/peregrine_control_service.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden {
namespace transport {

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback = lib::BlockReceivedCallback;

// High-speed Key-Value block transport engine.
class BlockTransport final {
 public:
  // Constructor sets up a TCP listening socket on the given `local_port`. It
  // starts #`parallelism` worker threads to handle incoming `WriteTask`s.
  BlockTransport(BlockTransportDelegate* delegate, int local_port,
                 const std::vector<std::string>& local_ips = {},
                 int parallelism = 1);

  // Destructor closes all sockets and joins all threads.
  ~BlockTransport();

  // Return the TCP listening socket port.
  int local_port() const { return raw_transport_.local_port(); }

  // Return the bound IP address.
  // It is the first IP in `local_ips` if provided, otherwise "127.0.0.1".
  const std::string& bound_ip() const { return raw_transport_.bound_ip(); }

  // Returns PeregrineService to register onto the host gRPC server.
  ::peregrine::internal::control::PeregrineService::Service*
  peregrine_control_service() {
    return peregrine_control_.get();
  }

  // Asynchronous Scatter-Gather Push
  void AsyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, int parallelism,
      MajorOrder major_order, uint64_t uuid, int layer_idx,
      std::function<void(absl::StatusOr<std::vector<int>>)> raw_on_complete);

  // Synchronous Scatter-Gather Push (op = 1 / op = 6)
  absl::StatusOr<std::vector<int>> SyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0,
      int layer_idx = -1);

  // Synchronous Scatter-Gather Pull (op = 2)
  // When explicit_dst_ptrs is supplied it contains one base pointer per
  // (block array, shard), in block-array-major order.
  absl::StatusOr<std::vector<int>> SyncPull(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {}, uint64_t uuid = 0);

  // Drops receive-progress counters belonging to a finished, failed, or
  // timed-out plan so the `uuid` can be safely reused.
  void ForgetPushProgress(uint64_t uuid);

  // Synchronously pushes a buffer identified by `buffer_id` to the remote
  // `peer`, by sending out a `kOpBufferPush ChunkHeader` followed by the data.
  // It waits for a one-byte ack from the `peer` before it returns.
  absl::Status PushBuffer(absl::string_view peer, size_t buffer_id,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          const uint8_t* data_ptr, size_t size_bytes,
                          uint64_t uuid = 0);

  // Pushes a vector of buffers to multiple peers.
  absl::Status PushBuffers(const std::vector<BufferPushTask>& tasks,
                           int parallelism, uint64_t uuid);

  // Registers the expected number of chunks for the given `uuid`.
  // If the completed number of chunks is equal to the expected, it triggers
  // the delegate's `OnDataReceived()` H2D callback.
  absl::Status RegisterExpectedChunks(uint64_t uuid, uint32_t expected_chunks) {
    return raw_transport_.RegisterExpectedChunks(uuid, expected_chunks);
  }

  // Registers the per-layer expected number of chunks for the given `uuid`.
  // When all chunks for a layer arrive, it triggers
  // `OnLayerDataReceived(layer_idx, uuid)`.
  absl::Status RegisterExpectedLayerChunks(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
    return raw_transport_.RegisterExpectedLayerChunks(uuid,
                                                      expected_layer_chunks);
  }

 private:
  lib::Request BuildBlockRequest(
      uint8_t socket_opcode, uint8_t* laddr, size_t len, uint32_t count_or_size,
      int layer_idx, uint32_t request_id, uint64_t uuid, int parallelism,
      MajorOrder major_order, uint32_t remote_id, uint32_t local_id,
      int shard_idx, int stream_idx = 0,
      BlockReceivedCallback on_block_received = nullptr);

  // Builds a batch of Requests for block transfer.
  absl::StatusOr<std::vector<lib::Request>> BuildBlockRequests(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, MajorOrder major_order,
      uint64_t uuid = 0, int layer_idx = -1, int parallelism = 1);

  // Builds a batch of Requests for block pull transfer.
  absl::StatusOr<std::vector<lib::Request>> BuildBlockPullRequests(
      const std::vector<int>& src_block_ids,
      const std::vector<int>& allocated_ids,
      const std::vector<uint8_t*>& explicit_dst_ptrs, MajorOrder major_order,
      uint64_t uuid = 0, int parallelism = 1,
      BlockReceivedCallback on_block_received = nullptr);

  absl::Status HandleIncomingPush(int client_fd,
                                  const lib::ChunkHeader& header);
  absl::Status HandleIncomingPull(int client_fd,
                                  const lib::ChunkHeader& header);

  absl::StatusOr<std::vector<int>> SyncPullInternal(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids,
      const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
      MajorOrder major_order, BlockReceivedCallback on_block_received,
      uint64_t uuid);

  struct SendStreamState {
    int client_fd;
    uint64_t uuid;
    int remote_id;
    size_t count_or_size;
    MajorOrder major_order;
    size_t current_step = 0;
    size_t total_steps = 0;
  };

  struct LayerProgress {
    size_t completed_chunks = 0;
    // Streams landed per sender node id, with the stream count that sender
    // declared for this block array. Used when the receive plan declares how
    // many senders assemble the array.
    struct SenderStreams {
      size_t landed = 0;
      size_t declared = 0;
    };
    absl::flat_hash_map<uint32_t, SenderStreams> sender_streams;
    // Senders whose declared streams have all landed.
    size_t senders_complete = 0;
    // Sender count the receive plan declares for this block array, resolved
    // from the delegate on the array's first stream. nullopt once resolved
    // means the header-declared contract applies.
    bool expected_senders_resolved = false;
    std::optional<size_t> expected_senders;
    bool on_layer_received_called = false;
  };

  void TriggerNextSendStep(std::shared_ptr<SendStreamState> state);
  void ResolveStepCoordinates(const std::shared_ptr<SendStreamState>& state,
                              size_t* layer, size_t* shard, size_t* block_idx);
  uint32_t GetChunksTotalSize(const std::vector<BlockChunk>& chunks);
  absl::Status HandleCustomRequest(int client_fd,
                                   const lib::ChunkHeader& header);

  using SendMap =
      absl::flat_hash_map<uint64_t, std::shared_ptr<SendStreamState>>;
  using ProgressMap =
      absl::flat_hash_map<std::pair<uint64_t, int>, LayerProgress>;

 private:
  BlockTransportDelegate* const block_delegate_;
  const int parallelism_;

  absl::Mutex active_sends_mu_;
  SendMap active_sends_ ABSL_GUARDED_BY(active_sends_mu_);

  absl::Mutex progress_mu_;
  ProgressMap layer_progress_ ABSL_GUARDED_BY(progress_mu_);

  lib::RawBufferTransport raw_transport_;
  std::unique_ptr<lib::PeregrineControlServiceImpl> peregrine_control_;
  std::unique_ptr<lib::TransportAdapter> transport_adapter_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_

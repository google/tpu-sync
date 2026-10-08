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

#include <cstddef>
#include <cstdint>
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
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/peregrine_control_service.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/service.grpc.pb.h"
#include "tpu_sync/transport/lib/socket_transport_adapter.h"
#include "tpu_sync/transport/lib/transport_adapter.h"
#include "tpu_sync/transport/lib/transport_metrics_exporter.h"

namespace tpu_raiden {
namespace transport {

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback = lib::BlockReceivedCallback;

// Maximum shard index addressable by a shard-masked push stream; the
// per-stream shard subset travels as a 16-bit mask in `ChunkHeader::buffer_id`
// (0 = every shard, the legacy encoding).
inline constexpr int kMaxShardMaskBits = 16;

// High-speed Key-Value block transport engine.
class BlockTransport final {
 public:
  // Constructor sets up a TCP listening socket on the given `local_port`. It
  // starts #`parallelism` worker threads to handle incoming `WriteTask`s.
  //
  // When the delegate's shards (`delegate->shards()`) span more than one NUMA
  // node, push streams whose shards live on node N are sent by worker threads
  // pinned to N (see lib::SocketTransportAdapter). Otherwise every stream
  // uses the default (unpinned) send path exactly as before.
  BlockTransport(BlockTransportDelegate* delegate, int local_port,
                 const std::vector<std::string>& local_ips = {},
                 int parallelism = 1);

  // Destructor closes all sockets and joins all threads.
  ~BlockTransport();

  struct Config {
    std::optional<absl::Duration> handshake_read_timeout = std::nullopt;
    std::optional<absl::Duration> payload_read_timeout = std::nullopt;
    size_t coalesce_window_bytes = 0;
  };

  const Config& config() const { return config_; }

  std::optional<absl::Duration> handshake_read_timeout() const {
    return config_.handshake_read_timeout;
  }
  std::optional<absl::Duration> payload_read_timeout() const {
    return config_.payload_read_timeout;
  }

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

  // Asynchronous Scatter-Gather Push (op = 1 / op = 6). `layer_idx` selects the
  // local block array; `wire_layer_idx`, when set, is the index the receiver
  // resolves the pushed blocks against (a sender whose pool table is a subset
  // of the receiver's).
  //
  // The blocks are split evenly into `parallelism` streams and stream `i`
  // goes to `peers[i % n]`. When the delegate reports shards on more than one
  // NUMA node (`shards()`), the push is additionally split by NUMA node: each
  // node gets its own `parallelism` streams carrying only that node's shards
  // (`ChunkHeader::buffer_id` = shard mask), sent from the node's local NICs by
  // threads pinned to it. This requires explicit `dst_block_ids` (op 6) and at
  // most kMaxShardMaskBits shards; otherwise, and whenever every shard is on
  // one node, the legacy single-group layout is used unchanged.
  //
  // Returns a future that resolves with the destination block ids once every
  // peer has acknowledged, or with the first error. Callers that need a
  // blocking push call `Await()` on the result; asynchronous callers attach a
  // continuation with `OnReady()`, which runs on the transport thread that
  // completes the push (or inline if the push already failed validation).
  tsl::Future<std::vector<int>> AsyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0,
      int layer_idx = -1, std::optional<int> wire_layer_idx = std::nullopt);

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

  // Synchronously pulls |size_bytes| starting at |src_offset_bytes| of the
  // remote |peer|'s buffer |buffer_id| / |src_shard_idx| into the local
  // buffer |buffer_id| / |dst_shard_idx| at |dst_offset_bytes|. The transfer
  // is driven entirely by the caller; the source does not need to be armed.
  // Ranges larger than 1 GiB are split into sequential slices.
  // TODO(justinlu): Add a batched pull only if many small ranges make the
  // per-request round trip dominate. That needs a new wire op (one request
  // listing N ranges, streamed back with writev) analogous to
  // kOpBufferPushBatched; looping PullBuffer on a pool gains nothing here.
  absl::Status PullBuffer(absl::string_view peer, size_t buffer_id,
                          size_t src_shard_idx, size_t src_offset_bytes,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          size_t size_bytes);

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

  void SetTestOnlyRateLimiters(
      std::shared_ptr<lib::TestOnlyRateLimiter> egress,
      std::shared_ptr<lib::TestOnlyRateLimiter> ingress) {
    raw_transport_.SetTestOnlyRateLimiters(std::move(egress),
                                           std::move(ingress));
  }

 private:
  // One push of |shard_ids| only, through |adapter|. `AsyncPush` calls this
  // once per source NUMA node group and joins the futures; |num_groups| is the
  // number of such calls, so the header declares the stream count over all of
  // them and the receiver completes only once every group has landed.
  tsl::Future<std::vector<int>> AsyncPushShards(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, int parallelism,
      MajorOrder major_order, uint64_t uuid, int layer_idx,
      std::optional<int> wire_layer_idx, absl::Span<const int> shard_ids,
      size_t num_groups, lib::TransportAdapter* adapter);

  lib::Request BuildBlockRequest(
      uint8_t socket_opcode, uint8_t* laddr, uint8_t* raddr, size_t len,
      uint32_t count_or_size, uint32_t request_id, uint64_t uuid,
      uint64_t buffer_id, int parallelism, MajorOrder major_order,
      uint32_t remote_id, uint32_t local_id, int stream_idx = 0,
      BlockReceivedCallback on_block_received = nullptr,
      uint16_t shard_mask = 0, int total_streams = 0);

  // Builds the Requests of one block push for |shard_ids| only: `parallelism`
  // streams (stream_idx 0..parallelism-1) over an even block partition, each
  // carrying every layer of |shard_ids|. |total_streams| is what the receiver
  // completes the push on; pass the sum over all shard groups when a push is
  // split into several calls (one per NUMA node), 0 for `parallelism`. The
  // wire shard mask is 0 (legacy) when |shard_ids| is every shard.
  absl::StatusOr<std::vector<lib::Request>> BuildBlockRequests(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, MajorOrder major_order,
      uint64_t uuid, int layer_idx, int parallelism,
      std::optional<int> wire_layer_idx, absl::Span<const int> shard_ids,
      int total_streams = 0);

  // Adapter that sends streams reading from |numa_node|: the per-node adapter
  // when one exists, otherwise the default `transport_adapter_`.
  lib::TransportAdapter* AdapterFor(int numa_node);

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
      // Plan-declared mode: shards landed beyond the last full shard set; a
      // full set (one per split push) counts as one landed plan push.
      size_t landed_shards = 0;
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
  const Config config_;

  absl::Mutex active_sends_mu_;
  SendMap active_sends_ ABSL_GUARDED_BY(active_sends_mu_);

  absl::Mutex progress_mu_;
  ProgressMap layer_progress_ ABSL_GUARDED_BY(progress_mu_);
  lib::TransportMetricsExporter metrics_exporter_;
  absl::Notification stop_metrics_thread_;
  std::thread metrics_thread_;

  lib::RawBufferTransport raw_transport_;
  std::unique_ptr<lib::PeregrineControlServiceImpl> peregrine_control_;
  std::unique_ptr<lib::TransportAdapter> transport_adapter_;
  // One extra adapter per NUMA node the delegate's shards span (only when
  // they span more than one), with its send workers pinned to that node.
  // Push streams for shards on that node go through it; everything else
  // (pulls, single-group pushes) uses `transport_adapter_`.
  absl::flat_hash_map<int, std::unique_ptr<lib::SocketTransportAdapter>>
      numa_adapters_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_

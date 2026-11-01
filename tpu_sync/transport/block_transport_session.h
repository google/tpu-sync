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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_BLOCK_TRANSPORT_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_BLOCK_TRANSPORT_SESSION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "peregrine/src/internal/control/service.grpc.pb.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/chunk.h"

namespace tpu_raiden {
namespace transport {

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

// Data-plane adapter used by a BlockTransport block op.
enum class TransportBackend : uint8_t {
  kSocket = 0,     // lib::SocketTransportAdapter.
  kPeregrine = 1,  // lib::PeregrineTransportAdapter.
};

// Maximum shard index addressable by a shard-masked push stream; the
// per-stream shard subset travels as a 16-bit mask in `ChunkHeader::buffer_id`
// (0 = every shard, the legacy encoding).
inline constexpr int kMaxShardMaskBits = 16;

struct BlockTransportConfig {
  int handshake_read_timeout_ms = -1;
  int payload_read_timeout_ms = -1;
  int ack_write_timeout_ms = -1;
  int payload_write_timeout_ms = -1;
  size_t coalesce_window_bytes = 0;

  absl::flat_hash_map<lib::TransferOp, TransportBackend> op_backends;

  TransportBackend backend_for(lib::TransferOp op) const {
    auto it = op_backends.find(op);
    return it == op_backends.end() ? TransportBackend::kSocket : it->second;
  }
  bool uses_peregrine() const {
    for (const auto& [op, backend] : op_backends) {
      if (backend == TransportBackend::kPeregrine) return true;
    }
    return false;
  }
};

// Base class for transport-level transfer sessions.
class BlockTransportSession {
 public:
  virtual ~BlockTransportSession() = default;

  virtual uint64_t uuid() const = 0;
  virtual bool IsDone() const = 0;
  virtual absl::Status status() const = 0;
};

// Base class for server-side sessions handling incoming requests.
class BlockTransportServerSession : public BlockTransportSession {
 public:
  enum class Kind : uint8_t {
    kPush = 0,
    kPull = 1,
  };

  virtual ~BlockTransportServerSession() = default;

  virtual Kind kind() const = 0;
  virtual void Cancel() = 0;

  // Starts streaming for an incoming session over TCP socket |client_fd|.
  virtual absl::Status StartStream(int client_fd,
                                   const lib::ChunkHeader& header) = 0;
};

// Server-side session managing incoming push streams (op 1 / op 6 / Peregrine).
// Tracks multi-stream chunk progress and dispatches completion callbacks to
// the delegate.
class BlockTransportServerPushSession final
    : public BlockTransportServerSession {
 public:
  BlockTransportServerPushSession(uint64_t uuid,
                                  BlockTransportDelegate* delegate,
                                  const BlockTransportConfig& config);

  ~BlockTransportServerPushSession() override = default;

  Kind kind() const override { return Kind::kPush; }
  uint64_t uuid() const override { return uuid_; }
  bool IsDone() const override;
  absl::Status status() const override;
  void Cancel() override;

  // Processes an incoming stream push payload over TCP socket |client_fd|.
  absl::Status StartStream(int client_fd,
                           const lib::ChunkHeader& header) override;

  // Processes receiver-side completion of a Peregrine Post.
  absl::Status OnPeregrinePostComplete(
      const lib::PostMetadata& post,
      absl::Span<const lib::RequestMetadata> requests);

 private:
  // Accounting for stream and shard arrivals.
  absl::Status CompleteIncomingPush(
      uint32_t local_id, uint32_t remote_id, size_t num_streams,
      size_t landed_shards,
      const std::optional<PoolPushProgressSpec>& pool_progress_spec,
      bool is_peregrine = false);

  struct LayerProgress {
    size_t completed_chunks = 0;
    struct SenderStreams {
      size_t landed = 0;
      size_t declared = 0;
      size_t landed_shards = 0;
    };
    absl::flat_hash_map<uint32_t, SenderStreams> sender_streams;
    size_t senders_complete = 0;
    bool expected_senders_resolved = false;
    std::optional<size_t> expected_senders;
    bool on_layer_received_called = false;
  };

  static size_t AccumulateLandedShards(LayerProgress::SenderStreams& streams,
                                       size_t shards, size_t full_shards);

  const uint64_t uuid_;
  BlockTransportDelegate* const delegate_;
  const BlockTransportConfig config_;

  mutable absl::Mutex mu_;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int, LayerProgress> layer_progress_ ABSL_GUARDED_BY(mu_);
};

// Server-side session managing incoming pull requests (op 2).
// Streams chunks step-by-step to |client_fd| as blocks become ready on the
// host.
class BlockTransportServerPullSession final
    : public BlockTransportServerSession,
      public std::enable_shared_from_this<BlockTransportServerPullSession> {
 public:
  using OnCompletedCallback = std::function<void(uint64_t uuid)>;

  BlockTransportServerPullSession(uint64_t uuid,
                                  BlockTransportDelegate* delegate,
                                  const BlockTransportConfig& config,
                                  absl::Span<const std::string> local_ips,
                                  OnCompletedCallback on_completed = nullptr);

  ~BlockTransportServerPullSession() override;

  Kind kind() const override { return Kind::kPull; }
  uint64_t uuid() const override { return uuid_; }
  bool IsDone() const override;
  absl::Status status() const override;
  void Cancel() override;

  // Starts the pull response streaming state machine over TCP socket
  // |client_fd|.
  absl::Status StartStream(int client_fd,
                           const lib::ChunkHeader& header) override;

 private:
  void TriggerNextSendStep();
  void ResolveStepCoordinates(size_t* layer, size_t* shard, size_t* block_idx);
  uint32_t GetChunksTotalSize(const std::vector<BlockChunk>& chunks);
  void Finish(absl::Status s);

  int client_fd_ = -1;
  const uint64_t uuid_;
  int remote_id_ = 0;
  size_t count_or_size_ = 0;
  MajorOrder major_order_ = MajorOrder::kLayerMajor;
  BlockTransportDelegate* const delegate_;
  const BlockTransportConfig config_;
  const std::vector<std::string> local_ips_;
  OnCompletedCallback on_completed_;

  size_t total_steps_ = 0;
  mutable absl::Mutex mu_;
  size_t current_step_ ABSL_GUARDED_BY(mu_) = 0;
  bool done_ ABSL_GUARDED_BY(mu_) = false;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  bool cancelled_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_BLOCK_TRANSPORT_SESSION_H_

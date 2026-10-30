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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_ADAPTER_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_ADAPTER_TRANSPORT_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "peregrine/src/api/transport_metrics.h"
#include "peregrine/src/api/transport_types.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

// `Handle` uniquely identifies a transport request.
using Handle = uint32_t;

// Callback invoked when a single block chunk is received during Pull.
using BlockReceivedCallback = std::function<absl::Status(
    size_t layer_idx, size_t shard_idx, int block_id, size_t size_bytes)>;

enum class TransferOp : uint8_t {
  kBlockPush = 1,  // receiver allocates dst blocks
  kBlockPull = 2,
  kBlockPushExplicit = 6,  // explicit dst block ids
};

struct Request {
  TransferOp op;
  uint8_t* laddr;
  uint8_t* raddr;
  size_t len;
  // Opaque to the transport; carries what the peer handling the request needs.
  // Layout:
  //    63                   32 31                    0
  //   +-----------------------+-----------------------+
  //   |   layer_idx (32 bits) |   shard_idx (32 bits) |
  //   +-----------------------+-----------------------+
  uint64_t buffer_id;
  uint8_t major_order;
  int layer_idx;
  int parallelism;
  uint64_t remote_id;  // sender node_id (push) / remote block ID (pull)
  uint32_t local_id;   // layer_idx (push) / local block ID (pull)
  uint32_t count_or_size;
  uint64_t uuid;
  uint32_t request_id;
  int stream_idx;

  // Block push streams only (kBlockPush / kBlockPushExplicit); identical
  // across the requests of one Post().
  //
  // Bitmask of shard indices carried by these streams (bit `i` = shard `i`);
  // 0 = legacy, every shard. Sent as `ChunkHeader::buffer_id` so the receiver
  // iterates the same shard subset.
  uint16_t shard_mask = 0;
  // Total number of streams the receiver should expect for this push across
  // every Post() that contributes to it (one per shard group); 0 = this
  // Post()'s `parallelism`. Sent as `ChunkHeader::reserved`.
  int total_streams = 0;

  BlockReceivedCallback on_block_received = nullptr;
};

// Transport operation status, shared with Peregrine.
using Status = ::peregrine::Status;

// Callback invoked when a transport Post operation completes.
using CompletionCallback =
    absl::AnyInvocable<void(absl::StatusOr<std::vector<int>>)>;

// Application metadata attached to one Post. Identical across the Post's
// requests.
struct PostMetadata {
  uint64_t uuid;
  uint32_t remote_id;
  uint32_t local_id;
  uint16_t shard_mask = 0;
};

// Application metadata attached to one request of a Post. Expected to grow
// with what the receiver needs to resolve a request's destination buffer.
struct RequestMetadata {
  uint32_t shard;
};

// Abstract transport adapter interface for Post and Poll operations.
class TransportAdapter {
 public:
  virtual ~TransportAdapter() = default;

  virtual absl::StatusOr<Handle> Post(
      absl::Span<const std::string> peers, absl::Span<const Request> requests,
      absl::Span<const int> src_block_ids = {},
      absl::Span<const int> dst_block_ids = {},
      CompletionCallback on_complete = nullptr) = 0;

  virtual absl::StatusOr<Status> Poll(Handle handle) = 0;

  // Port peers use to reach this transport.
  virtual int local_port() const = 0;

  // Cleans up any cached per-session state (such as cancellation flags)
  // associated with uuid so that the uuid may be safely reused.
  virtual void ForgetUuid(uint64_t uuid) {}

  virtual ::peregrine::TransportMetrics GetTransportMetrics() const {
    return {};
  }
};

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_ADAPTER_TRANSPORT_ADAPTER_H_

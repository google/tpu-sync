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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TRANSPORT_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

// `Handle` uniquely identifies a transport request.
using Handle = uint32_t;

// Callback invoked when a single block chunk is received during Pull.
using BlockReceivedCallback = std::function<absl::Status(
    size_t layer_idx, size_t shard_idx, int block_id, size_t size_bytes)>;

struct Request {
  uint8_t socket_opcode;
  uint8_t* laddr;
  uint8_t* raddr;
  size_t len;

  uint8_t major_order;
  int layer_idx;
  int parallelism;
  uint64_t remote_id;  // Block: sender node_id / remote block ID | Buffer:
                       // dst/src byte offset.
  uint32_t local_id;   // Block: layer_idx / local block ID |
                       // Buffer: dst/src shard index.
  uint32_t count_or_size;
  uint64_t uuid;
  uint32_t request_id;
  int shard_idx;
  int stream_idx;

  BlockReceivedCallback on_block_received = nullptr;
};

// Transport operation status.
enum class Status : int {
  kNotFound = 2,
  kInProgress = 1,
  kSuccess = 0,
  kFailure = -1,
};

// Callback invoked when a transport Post operation completes.
using CompletionCallback =
    absl::AnyInvocable<void(absl::StatusOr<std::vector<int>>)>;

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
};

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TRANSPORT_ADAPTER_H_

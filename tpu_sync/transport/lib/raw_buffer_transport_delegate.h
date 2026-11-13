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

#ifndef TPU_SYNC_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_DELEGATE_H_
#define TPU_SYNC_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_DELEGATE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"

namespace tpu_raiden::transport::lib {

// Foundational delegate interface for RawBufferTransport to query base host
// memory.
class RawBufferTransportDelegate {
 public:
  virtual ~RawBufferTransportDelegate() = default;

  // Host staging buffer of (|buffer_id|, |shard_idx|). Returns
  // FailedPrecondition exactly when the target has no host buffer, and
  // OutOfRange when |buffer_id| or |shard_idx| is out of range; callers
  // rely on this distinction. An OK span has a non-null `data()` and may be
  // empty.
  virtual absl::StatusOr<absl::Span<const uint8_t>> GetHostBuffer(
      size_t buffer_id, size_t shard_idx) const = 0;

  // Mutable view of the const `GetHostBuffer()`, with the same error contract.
  virtual absl::StatusOr<absl::Span<uint8_t>> GetHostBuffer(size_t buffer_id,
                                                            size_t shard_idx) {
    ABSL_ASSIGN_OR_RETURN(
        absl::Span<const uint8_t> host_buffer,
        std::as_const(*this).GetHostBuffer(buffer_id, shard_idx));
    // The span views delegate-owned mutable staging memory.
    return absl::Span<uint8_t>(const_cast<uint8_t*>(host_buffer.data()),
                               host_buffer.size());
  }

  // Notification triggered upon verified data chunk arrival for a specific
  // layer/tensor.
  virtual absl::Status OnLayerDataReceived(size_t layer_idx,
                                           uint64_t uuid = 0) {
    return absl::OkStatus();
  }

  // Notification triggered upon verified data chunk arrival for the entire
  // transfer.
  virtual absl::Status OnDataReceived(uint64_t uuid = 0) {
    return absl::OkStatus();
  }

  // Returns the gRPC channel for PeregrineControlService for the given peer.
  // Required when FLAGS_require_psp_tcp is true.
  // Note the peer string is the same as the peer passed to BlockTransport
  // api, e.g. SyncPush(peer,...), which is the data-plane IP:port. The
  // delegate needs to resolve the peer to either a pre-existing or newly
  // created gRPC channel.
  virtual std::shared_ptr<grpc::Channel> GetPeregrineChannel(
      absl::string_view peer) {
    return nullptr;
  }
};

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_DELEGATE_H_

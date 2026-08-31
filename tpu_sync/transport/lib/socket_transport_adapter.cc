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

#include "tpu_sync/transport/lib/socket_transport_adapter.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

SocketTransportAdapter::SocketTransportAdapter(
    RawBufferTransport* raw_transport, int parallelism)
    : raw_transport_(raw_transport), parallelism_(parallelism) {}

SocketTransportAdapter::~SocketTransportAdapter() = default;

absl::StatusOr<Handle> SocketTransportAdapter::Post(
    absl::Span<const std::string> peers, absl::Span<const Request> requests,
    absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
    CompletionCallback on_complete) {
  return absl::UnimplementedError(
      "SocketTransportAdapter::Post not implemented yet");
}

absl::StatusOr<Status> SocketTransportAdapter::Poll(Handle handle) {
  return absl::UnimplementedError(
      "SocketTransportAdapter::Poll not implemented yet");
}

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

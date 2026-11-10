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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_ADAPTER_PEREGRINE_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_ADAPTER_PEREGRINE_TRANSPORT_ADAPTER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "peregrine/src/api/transport.h"
#include "peregrine/src/api/transport_types.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/metadata.h"

namespace tpu_raiden::transport::lib {

// Peregrine implementation of TransportAdapter.
class PeregrineTransportAdapter : public TransportAdapter {
 public:
  // Receiver-side hooks, invoked on Peregrine's receive threads.
  struct RecvHooks {
    using OnPostComplete = absl::AnyInvocable<absl::Status(
        const PostMetadata&, const absl::Status&) const>;
    using ResolveAddr = absl::AnyInvocable<uint8_t*(
        const PostMetadata&, const RequestMetadata&) const>;
    OnPostComplete on_post_complete;
    ResolveAddr resolve_addr;
  };

  // Creates a TCP Peregrine transport listening on `bind_ip`:`port` (0 =
  // kernel-assigned).
  static absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> Create(
      absl::string_view bind_ip, int port, int num_parallel_connections,
      RecvHooks hooks = {});

  ~PeregrineTransportAdapter() override = default;

  absl::StatusOr<Handle> Post(absl::Span<const std::string> peers,
                              absl::Span<const Request> requests,
                              absl::Span<const int> src_block_ids,
                              absl::Span<const int> dst_block_ids,
                              CompletionCallback on_complete) override;

  absl::StatusOr<Status> Poll(Handle handle) override;

  int local_port() const override { return local_port_; }

 private:
  PeregrineTransportAdapter(std::unique_ptr<::peregrine::Transport> transport,
                            int local_port);

  // Pull with sender-supplied `raddr` and no metadata.
  absl::StatusOr<Handle> PostPull(absl::string_view peer,
                                  absl::Span<const Request> requests,
                                  CompletionCallback on_complete);
  // Receiver-resolved explicit push: no `raddr`, Post and request metadata.
  absl::StatusOr<Handle> PostExplicitPush(absl::string_view peer,
                                          absl::Span<const Request> requests,
                                          CompletionCallback on_complete);
  // Posts `requests` to `peer` and adapts Peregrine's completion.
  absl::StatusOr<Handle> PostRequests(
      absl::string_view peer, absl::Span<const ::peregrine::Request> requests,
      const MetadataBytes& post_metadata, CompletionCallback on_complete);

  const std::unique_ptr<::peregrine::Transport> transport_;
  const int local_port_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_ADAPTER_PEREGRINE_TRANSPORT_ADAPTER_H_

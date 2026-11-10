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

#include "tpu_sync/transport/lib/adapter/peregrine_transport_adapter.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "peregrine/src/api/transport.h"
#include "peregrine/src/api/transport_types.h"
#include "peregrine/src/api/transport_util.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/metadata.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::peregrine::Op;
using ::peregrine::Options;
using ::peregrine::TransportType;

absl::Status ReportError(CompletionCallback& on_complete, absl::Status status) {
  if (!status.ok() && on_complete) {
    on_complete(status);
  }
  return status;
}

absl::Status ToStatus(::peregrine::Status s) {
  if (s == ::peregrine::Status::kSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat("Peregrine transport request failed with status: ",
                   ::peregrine::ToString(s)));
}

void Complete(CompletionCallback& callback, absl::Status status) {
  if (status.ok()) {
    callback(std::vector<int>{});
  } else {
    callback(std::move(status));
  }
}

// Port of an "ip:port" / "[ipv6]:port" endpoint, or -1 if it has none.
int PortOf(absl::string_view endpoint) {
  const size_t colon = endpoint.rfind(':');
  if (colon == absl::string_view::npos) return -1;
  int port;
  return absl::SimpleAtoi(endpoint.substr(colon + 1), &port) ? port : -1;
}

PostMetadata PostMetadataOf(const Request& request) {
  return PostMetadata{
      .uuid = request.uuid,
      .remote_id = static_cast<uint32_t>(request.remote_id),
      .local_id = request.local_id,
      .shard_mask = request.shard_mask,
      .op = request.op,
  };
}

absl::Status ValidateRequests(absl::Span<const Request> requests) {
  for (size_t i = 0; i < requests.size(); ++i) {
    const Request& req = requests[i];
    if (req.len == 0 || req.len > std::numeric_limits<uint32_t>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Peregrine request ", i, " of ", requests.size(),
                       ": len must be in [1, 2^32-1] (len=", req.len,
                       ", uuid=", req.uuid, ")"));
    }
    if (req.op == TransferOp::kBlockPushExplicit && req.dst_block_id < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Peregrine request ", i, " of ", requests.size(),
          ": explicit push needs a destination block (uuid=", req.uuid, ")"));
    }
  }
  return absl::OkStatus();
}

struct GroupCompletion {
  ~GroupCompletion() { Complete(on_complete, std::move(error)); }

  void Record(absl::Status status) {
    if (status.ok()) return;
    absl::MutexLock lock(mu);
    if (error.ok()) error = std::move(status);
  }

  CompletionCallback on_complete;
  absl::Mutex mu;
  absl::Status error ABSL_GUARDED_BY(mu);
};

using OnPostComplete = PeregrineTransportAdapter::RecvHooks::OnPostComplete;

::peregrine::RecvPostCallback MakeRecvPostCallback(
    std::shared_ptr<OnPostComplete> hook) {
  return [hook = std::move(hook)](
             ::peregrine::Status peregrine_status,
             const MetadataBytes& post_bytes,
             absl::Span<const MetadataBytes> /*req_bytes*/) {
    if (IsEmptyMetadata(post_bytes)) {
      VLOG(1) << "Ignoring Peregrine Post without application metadata: "
              << ::peregrine::ToString(peregrine_status);
      return;
    }
    absl::StatusOr<PostMetadata> post = DecodePostMetadata(post_bytes);
    if (!post.ok()) {
      LOG(ERROR) << "Ignoring Peregrine Post with undecodable metadata: "
                 << post.status();
      return;
    }
    const absl::Status status = (*hook)(*post, ToStatus(peregrine_status));
    if (!status.ok()) {
      LOG(ERROR) << "Peregrine post completion rejected: uuid=" << post->uuid
                 << " remote_id=" << post->remote_id
                 << " local_id=" << post->local_id << ": " << status;
    }
  };
}

::peregrine::ResolveAddrCallback MakeResolveAddrCallback(
    PeregrineTransportAdapter::RecvHooks::ResolveAddr hook,
    std::shared_ptr<OnPostComplete> done) {
  return [hook = std::move(hook), done = std::move(done)](
             const MetadataBytes& post_bytes,
             const MetadataBytes& req_bytes) -> uint8_t* {
    absl::StatusOr<PostMetadata> post = DecodePostMetadata(post_bytes);
    if (!post.ok()) {
      LOG(ERROR) << "Rejecting Peregrine Post with undecodable metadata: "
                 << post.status();
      return nullptr;
    }
    absl::StatusOr<RequestMetadata> req =
        DecodeRequestMetadata(*post, req_bytes);
    if (!req.ok()) {
      LOG(ERROR) << "Rejecting Peregrine Post with undecodable request "
                    "metadata: uuid="
                 << post->uuid << " remote_id=" << post->remote_id
                 << " local_id=" << post->local_id << ": " << req.status();
      // Peregrine drops the Post without a completion callback, so the
      // receiver must be told here to release the Post's lease.
      if (*done) {
        (*done)(*post,
                absl::InvalidArgumentError(absl::StrCat(
                    "Undecodable request metadata: ", req.status().message())))
            .IgnoreError();
      }
      return nullptr;
    }
    return hook(*post, *req);
  };
}

}  // namespace

absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>>
PeregrineTransportAdapter::Create(absl::string_view bind_ip, int port,
                                  int num_parallel_connections,
                                  RecvHooks hooks) {
  const std::string endpoint = absl::StrContains(bind_ip, ':')
                                   ? absl::StrCat("[", bind_ip, "]:", port)
                                   : absl::StrCat(bind_ip, ":", port);
  Options options{
      .transport_type = TransportType::kTcp,
      .num_parallel_connections = num_parallel_connections,
  };
  auto on_post_complete = std::make_shared<RecvHooks::OnPostComplete>(
      std::move(hooks.on_post_complete));
  if (*on_post_complete != nullptr) {
    options.recv_post_callback = MakeRecvPostCallback(on_post_complete);
  }
  if (hooks.resolve_addr != nullptr) {
    options.resolve_addr_callback = MakeResolveAddrCallback(
        std::move(hooks.resolve_addr), on_post_complete);
  }
  std::unique_ptr<::peregrine::Transport> transport =
      ::peregrine::CreateTransport(endpoint, std::move(options));
  if (transport == nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Failed to create Peregrine transport on ", endpoint));
  }
  const int local_port = PortOf(transport->GetControlEndpoint());
  if (local_port <= 0) {
    return absl::InternalError(
        absl::StrCat("Peregrine transport on ", endpoint,
                     " reported an invalid control endpoint: ",
                     transport->GetControlEndpoint()));
  }
  return absl::WrapUnique(
      new PeregrineTransportAdapter(std::move(transport), local_port));
}

PeregrineTransportAdapter::PeregrineTransportAdapter(
    std::unique_ptr<::peregrine::Transport> transport, int local_port)
    : transport_(std::move(transport)), local_port_(local_port) {}

absl::StatusOr<Handle> PeregrineTransportAdapter::Post(
    absl::Span<const std::string> peers, absl::Span<const Request> requests,
    absl::Span<const int> /*src_block_ids*/,
    absl::Span<const int> /*dst_block_ids*/, CompletionCallback on_complete) {
  if (peers.empty()) {
    return ReportError(on_complete,
                       absl::InvalidArgumentError("Peers cannot be empty"));
  }
  if (requests.empty()) {
    return ReportError(on_complete,
                       absl::InvalidArgumentError("Requests cannot be empty"));
  }
  ABSL_RETURN_IF_ERROR(ReportError(on_complete, ValidateRequests(requests)));

  // Every request of one Post carries the same op. Receiver-allocated pushes
  // have no remote address and cannot be expressed over Peregrine.
  //
  // Peregrine identifies a host by a single control-plane endpoint and handles
  // multi-NIC data transfer internally, so only the first peer is used.
  // TODO(swasthi): Weight sync broadcasts to multiple distinct hosts via
  // `ShardPushEntryProto.dst_peers`; that path is not supported here yet.
  switch (requests.front().op) {
    case TransferOp::kBlockPull:
      return PostPull(peers.front(), requests, std::move(on_complete));
    case TransferOp::kBlockPushExplicit:
      return PostExplicitPush(peers.front(), requests, std::move(on_complete));
    case TransferOp::kBlockPush:
      return ReportError(
          on_complete,
          absl::UnimplementedError(
              "Receiver-allocated block push is not supported over Peregrine; "
              "explicit destination block ids are required"));
    default:
      return ReportError(on_complete,
                         absl::InvalidArgumentError(absl::StrCat(
                             "Unsupported op over Peregrine: ",
                             static_cast<int>(requests.front().op))));
  }
}

absl::StatusOr<Handle> PeregrineTransportAdapter::PostPull(
    absl::string_view peer, absl::Span<const Request> requests,
    CompletionCallback on_complete) {
  // TODO(swasthi): Wire pull Post/Request metadata so the responder can
  // resolve `raddr` for `BlockTransport::SyncPull'.
  std::vector<::peregrine::Request> peregrine_requests;
  peregrine_requests.reserve(requests.size());
  for (const Request& req : requests) {
    peregrine_requests.push_back(::peregrine::Request{
        .op = Op::kRead,
        .laddr = req.laddr,
        .raddr = req.raddr,
        .len = req.len,
    });
  }
  return PostRequests(peer, peregrine_requests, /*post_metadata=*/{},
                      std::move(on_complete));
}

absl::StatusOr<Handle> PeregrineTransportAdapter::PostExplicitPush(
    absl::string_view peer, absl::Span<const Request> requests,
    CompletionCallback on_complete) {
  struct Group {
    PostMetadata post;
    std::vector<::peregrine::Request> reqs;
  };
  absl::btree_map<uint32_t, Group> groups;
  for (const Request& req : requests) {
    const uint32_t layer = static_cast<uint32_t>(req.buffer_id >> 32);
    auto [it, inserted] = groups.try_emplace(layer);
    Group& group = it->second;
    if (inserted) {
      group.post = PostMetadataOf(req);
      group.post.local_id = layer;
    }
    group.reqs.push_back(::peregrine::Request{
        .op = Op::kWrite,
        .laddr = req.laddr,
        .raddr = nullptr,
        .len = req.len,
        .metadata = EncodeRequestMetadata(
            group.post,
            RequestMetadata{
                .shard = static_cast<uint32_t>(req.buffer_id & 0xFFFFFFFF),
                .block_id = req.dst_block_id,
                .src_block_id = req.src_block_id,
                .chunk_idx = req.chunk_idx,
                .len = static_cast<uint32_t>(req.len),
            }),
    });
  }
  if (groups.size() == 1) {
    const Group& group = groups.begin()->second;
    return PostRequests(peer, group.reqs, EncodePostMetadata(group.post),
                        std::move(on_complete));
  }
  if (!on_complete) {
    return ReportError(
        on_complete,
        absl::InvalidArgumentError(absl::StrCat(
            "multi-layer push requires a completion callback (uuid=",
            requests.front().uuid, ")")));
  }

  auto state = std::make_shared<GroupCompletion>(std::move(on_complete));
  for (auto& [layer, group] : groups) {
    ABSL_RETURN_IF_ERROR(
        PostRequests(peer, group.reqs, EncodePostMetadata(group.post),
                     [state](absl::StatusOr<std::vector<int>> r) {
                       state->Record(r.status());
                     })
            .status());
  }
  return 0;
}

absl::StatusOr<Handle> PeregrineTransportAdapter::PostRequests(
    absl::string_view peer, absl::Span<const ::peregrine::Request> requests,
    const MetadataBytes& post_metadata, CompletionCallback on_complete) {
  auto callback = std::make_shared<CompletionCallback>(std::move(on_complete));
  absl::AnyInvocable<void(::peregrine::Status)> on_peregrine_complete = nullptr;
  if (*callback) {
    on_peregrine_complete = [callback](::peregrine::Status status) {
      Complete(*callback, ToStatus(status));
    };
  }
  // TODO(swasthi): Needs a Peregrine NACK before this path is enabled by
  // default.
  absl::StatusOr<::peregrine::Handle> handle = transport_->Post(
      peer, requests, std::move(on_peregrine_complete), post_metadata);
  if (!handle.ok()) return ReportError(*callback, handle.status());
  return handle->value();
}

absl::StatusOr<Status> PeregrineTransportAdapter::Poll(Handle handle) {
  return transport_->Poll(::peregrine::Handle(handle));
}

}  // namespace tpu_raiden::transport::lib

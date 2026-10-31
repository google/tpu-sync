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

#include "tpu_sync/transport/block_transport_session.h"

#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "peregrine/src/api/v1/ip_util.h"
#include "peregrine/src/api/v1/socket_util.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/fault_injection/hooks.h"
#include "tpu_sync/telemetry/label_util.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {
namespace transport {

namespace {

using ::peregrine::v1::GetAddrPortPair;
using ::peregrine::v1::ReadExact;
using ::peregrine::v1::ReadVExact;
using ::peregrine::v1::WriteExact;
using ::peregrine::v1::WriteVExact;
using ::tpu_raiden::telemetry::ExtractFirstEndpointIp;
using ::tpu_raiden::telemetry::ExtractIpFromEndpoint;
using ::tpu_raiden::telemetry::MetricLabel;
using ::tpu_raiden::telemetry::RaidenMetricStore;
namespace metric_labels = ::tpu_raiden::telemetry::metric_labels;
namespace metric_names = ::tpu_raiden::telemetry::metric_names;

void RecordTransferFailure(const absl::Status& status,
                           absl::string_view direction, uint64_t count = 1) {
  RaidenMetricStore& store = RaidenMetricStore::GetGlobalMetricStore();
  if (status.ok() || !store.HasBackends()) return;
  const absl::string_view error_code =
      absl::StatusCodeToStringView(status.code());
  const MetricLabel labels[] = {
      {metric_labels::kErrorCode, error_code},
      {metric_labels::kDirection, direction},
  };
  store.IncrementCounter(metric_names::kTransferFailuresTotal, labels, count);
}

constexpr MetricLabel kPushLabels[] = {
    {.key = metric_labels::kDirection, .value = metric_labels::kDirectionPush},
};

void RecordPullResponseSentBytes(int client_fd,
                                 absl::Span<const std::string> local_ips,
                                 uint64_t total_size) {
  RaidenMetricStore& store = RaidenMetricStore::GetGlobalMetricStore();
  if (!store.HasBackends()) return;
  const std::string addr_pair = GetAddrPortPair(client_fd);
  const std::pair<absl::string_view, absl::string_view> endpoints =
      absl::StrSplit(addr_pair, absl::MaxSplits("--", 1));
  const absl::string_view peer_endpoint = endpoints.second;
  const MetricLabel pull_response_labels[] = {
      {.key = metric_labels::kDirection,
       .value = metric_labels::kDirectionPullResponse},
      {.key = metric_labels::kSrcIp,
       .value = ExtractFirstEndpointIp(local_ips)},
      {.key = metric_labels::kDstIp,
       .value = ExtractIpFromEndpoint(peer_endpoint)},
  };
  store.IncrementCounter(metric_names::kSentBytesTotal, pull_response_labels,
                         total_size);
}

constexpr uint8_t kUseBlockChunksFlag = 0x80;

std::vector<struct iovec> ToIovec(const std::vector<BlockChunk>& chunks) {
  std::vector<struct iovec> iov;
  iov.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    if (chunk.size > 0) {
      iov.push_back({.iov_base = chunk.ptr, .iov_len = chunk.size});
    }
  }
  return iov;
}

absl::Status ValidateChunks(BlockTransportDelegate* delegate, size_t l,
                            size_t sh, const std::vector<BlockChunk>& chunks) {
  uint8_t* base = delegate->GetBlockArrayHostPointer(l, sh);
  // Some legacy delegates intentionally expose only scattered per-block
  // pointers and no flat array base. Preserve that contract; pool-aware
  // managers always expose their exact pool span here.
  if (base != nullptr) {
    const size_t host_size = delegate->GetBlockArrayHostSize(l, sh);
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
    for (const auto& chunk : chunks) {
      const uintptr_t chunk_addr = reinterpret_cast<uintptr_t>(chunk.ptr);
      if (chunk_addr < base_addr) {
        return absl::OutOfRangeError(absl::StrCat(
            "Chunk out of bounds. Chunk ptr: ", chunk_addr,
            ", size: ", chunk.size, ", Block array base: ", base_addr,
            ", Block array size: ", host_size));
      }
      const size_t offset = chunk_addr - base_addr;
      if (offset > host_size || chunk.size > host_size - offset) {
        return absl::OutOfRangeError(absl::StrCat(
            "Chunk out of bounds. Chunk ptr: ", chunk_addr,
            ", size: ", chunk.size, ", Block array base: ", base_addr,
            ", Block array size: ", host_size));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<MajorOrder> ParseMajorOrder(uint8_t value) {
  uint8_t order_val = value & ~kUseBlockChunksFlag;
  switch (order_val) {
    case static_cast<uint8_t>(MajorOrder::kLayerMajor):
      return MajorOrder::kLayerMajor;
    case static_cast<uint8_t>(MajorOrder::kBlockMajor):
      return MajorOrder::kBlockMajor;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown block transport major order: ", value));
  }
}

template <typename Fn>
absl::Status ForEachPayload(MajorOrder major_order,
                            const std::vector<int>& layer_ids,
                            absl::Span<const int> shard_ids, size_t num_blocks,
                            Fn fn) {
  switch (static_cast<int>(major_order)) {
    case static_cast<int>(MajorOrder::kLayerMajor):
      for (int l : layer_ids) {
        for (int sh : shard_ids) {
          for (size_t k = 0; k < num_blocks; ++k) {
            ABSL_RETURN_IF_ERROR(fn(l, static_cast<size_t>(sh), k));
          }
        }
      }
      return absl::OkStatus();
    case static_cast<int>(MajorOrder::kBlockMajor):
      for (size_t k = 0; k < num_blocks; ++k) {
        for (int l : layer_ids) {
          for (int sh : shard_ids) {
            ABSL_RETURN_IF_ERROR(fn(l, static_cast<size_t>(sh), k));
          }
        }
      }
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Unknown block transport major order");
}

std::vector<int> AllShards(size_t num_shards) {
  std::vector<int> shards(num_shards);
  std::iota(shards.begin(), shards.end(), 0);
  return shards;
}

// Shard subset addressed by a push stream's `ChunkHeader::buffer_id`; 0 is
// the legacy "every shard" encoding.
absl::StatusOr<std::vector<int>> ShardsFromMask(uint16_t mask,
                                                size_t num_shards) {
  if (mask == 0) return AllShards(num_shards);
  std::vector<int> shards;
  for (int sh = 0; sh < kMaxShardMaskBits; ++sh) {
    if ((mask >> sh) & 1u) {
      if (static_cast<size_t>(sh) >= num_shards) {
        return absl::OutOfRangeError(
            absl::StrCat("push stream shard mask ", mask, " addresses shard ",
                         sh, " but the receiver has ", num_shards));
      }
      shards.push_back(sh);
    }
  }
  return shards;
}

struct IncomingPushSpec {
  std::optional<PoolPushProgressSpec> pool_progress_spec;
  std::vector<int> target_layers;
};

absl::StatusOr<IncomingPushSpec> ResolveIncomingPushSpec(
    BlockTransportDelegate* delegate, uint32_t local_id,
    bool is_explicit_dst_push, uint64_t uuid) {
  std::vector<int> target_layers;
  if (local_id == 0xFFFFFFFF) {
    target_layers.resize(delegate->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    if (local_id >= delegate->num_block_arrays()) {
      return absl::OutOfRangeError(
          absl::StrCat("push block-array index ", local_id,
                       " out of range: ", delegate->num_block_arrays()));
    }
    target_layers = {static_cast<int>(local_id)};
  }

  // Resolve the expectation source before the explicit-destination handshake
  // or any payload write. A uuid whose registered receive plan carries pool
  // fields is plan-declared: it addresses exactly one declared pool and its
  // completion gate is the plan's global push count. nullopt keeps the
  // header-declared (legacy) contract.
  std::optional<PoolPushProgressSpec> pool_progress_spec;
  for (int target_layer : target_layers) {
    ABSL_ASSIGN_OR_RETURN(
        std::optional<PoolPushProgressSpec> candidate,
        delegate->GetPoolPushProgressSpec(target_layer, uuid));
    if (!candidate.has_value()) {
      if (pool_progress_spec.has_value()) {
        return absl::InvalidArgumentError(
            "push mixes pool-keyed and legacy block arrays");
      }
      continue;
    }
    if (target_layers.size() != 1) {
      return absl::InvalidArgumentError(
          "pool-keyed push must address exactly one transfer pool");
    }
    if (candidate->expected_pushes == 0 || candidate->expected_pools == 0) {
      return absl::InvalidArgumentError(
          "pool-keyed push progress counts must be positive");
    }
    pool_progress_spec = *candidate;
  }

  if (is_explicit_dst_push && !pool_progress_spec.has_value() &&
      !delegate->AcceptsPlanlessExplicitPush(uuid)) {
    return absl::FailedPreconditionError(
        absl::StrCat("explicit-destination push for uuid ", uuid,
                     " has no registered plan on this pool-mode worker"));
  }

  return IncomingPushSpec{
      .pool_progress_spec = pool_progress_spec,
      .target_layers = std::move(target_layers),
  };
}

absl::StatusOr<std::shared_ptr<TransferSession>> BeginIncomingPush(
    BlockTransportDelegate* delegate, uint64_t uuid) {
  if (uuid == 0) {
    return nullptr;
  }
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<TransferSession> session,
                        delegate->GetSession(uuid));
  FaultInjectDelay(hooks::kBlockTransportRecvBeginPush);
  if (!session->TryBeginOp()) {
    return absl::CancelledError(
        absl::StrCat("Session for uuid=", uuid,
                     " is not accepting pushes (draining or a send session)"));
  }
  return session;
}

absl::Status EndIncomingPush(const std::shared_ptr<TransferSession>& session,
                             const absl::Status& status = absl::OkStatus()) {
  if (session == nullptr) {
    return status;
  }
  if (!status.ok()) {
    session->Finish(status);
  }
  session->EndOp();
  if (!status.ok() || session->GetStatus().ok()) {
    return status;
  }
  return absl::CancelledError(session->GetStatus().message());
}

}  // namespace

// ===========================================================================
// BlockTransportServerPushSession
// ===========================================================================

BlockTransportServerPushSession::BlockTransportServerPushSession(
    uint64_t uuid, BlockTransportDelegate* delegate,
    const BlockTransportConfig& config)
    : uuid_(uuid), delegate_(delegate), config_(config) {}

bool BlockTransportServerPushSession::IsDone() const {
  absl::MutexLock lock(mu_);
  return done_;
}

absl::Status BlockTransportServerPushSession::status() const {
  absl::MutexLock lock(mu_);
  return status_;
}

void BlockTransportServerPushSession::Cancel() {
  absl::MutexLock lock(mu_);
  if (done_) return;
  done_ = true;
  status_ = absl::CancelledError("Push session cancelled");
}

absl::Status BlockTransportServerPushSession::StartStream(
    int client_fd, const lib::ChunkHeader& header) {
  ABSL_ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
  ABSL_ASSIGN_OR_RETURN(
      IncomingPushSpec push_spec,
      ResolveIncomingPushSpec(delegate_, header.local_id,
                              /*is_explicit_dst_push=*/header.op == 6,
                              header.uuid));

  std::vector<int> allocated_ids;
  std::vector<int> src_block_ids;
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<TransferSession> session,
                        BeginIncomingPush(delegate_, header.uuid));
  bool incoming_push_lease_held = true;
  absl::Cleanup end_incoming_push = [&]() {
    if (incoming_push_lease_held) {
      EndIncomingPush(session, absl::InternalError("Incoming push failed"))
          .IgnoreError();
    }
  };

  if (header.op == 1) {
    ABSL_ASSIGN_OR_RETURN(
        allocated_ids,
        delegate_->AllocateBlocks(header.count_or_size, header.uuid));
    const std::vector<uint8_t> s_ids = lib::SerializeBlockIds(allocated_ids);
    ABSL_RETURN_IF_ERROR(WriteExact(client_fd, s_ids.data(), s_ids.size(),
                                    config_.ack_write_timeout_ms));
  } else {
    std::vector<uint8_t> ids_buf(header.count_or_size * sizeof(uint32_t));
    FaultInjectSocket(hooks::kBlockTransportRecvBlockIds, client_fd);
    ABSL_RETURN_IF_ERROR(ReadExact(client_fd, ids_buf.data(), ids_buf.size(),
                                   config_.handshake_read_timeout_ms));
    allocated_ids = lib::DeserializeBlockIds(ids_buf);

    ABSL_RETURN_IF_ERROR(ReadExact(client_fd, ids_buf.data(), ids_buf.size(),
                                   config_.handshake_read_timeout_ms));
    src_block_ids = lib::DeserializeBlockIds(ids_buf);
    uint8_t ack = 1;
    FaultInjectSocket(hooks::kBlockTransportRecvSendHandshakeAck, client_fd);
    ABSL_RETURN_IF_ERROR(
        WriteExact(client_fd, &ack, 1, config_.ack_write_timeout_ms));
  }

  uint64_t total_received_bytes = 0;
  // Shards this stream carries: every shard for a legacy push, one NUMA
  // group's subset for a push split by source NUMA node.
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<int> stream_shards,
      ShardsFromMask(header.buffer_id, delegate_->num_shards()));

  // The payload arrives as one chunk per (layer, shard, block). The payload
  // fault injection hook fires once per push, right before the middle chunk,
  // so an injected failure or latency hits mid-transfer: half of the payload
  // has already been received and the rest is still in flight.
  const size_t fault_injection_payload_index = push_spec.target_layers.size() *
                                               stream_shards.size() *
                                               header.count_or_size / 2;
  size_t payload_index = 0;
  ABSL_RETURN_IF_ERROR(ForEachPayload(
      major_order, push_spec.target_layers, stream_shards, header.count_or_size,
      [&](size_t l, size_t sh, size_t k) -> absl::Status {
        ABSL_DCHECK_LT(k, allocated_ids.size());
        const int dst_id = allocated_ids[k];
        if (payload_index++ == fault_injection_payload_index) {
          FaultInjectSocket(hooks::kBlockTransportRecvPayload, client_fd);
        }
        uint8_t size_buf[lib::kChunkSizeFieldSize];
        ABSL_RETURN_IF_ERROR(ReadExact(client_fd, size_buf, sizeof(size_buf),
                                       config_.payload_read_timeout_ms));
        const uint32_t sender_size = lib::DeserializeChunkSize(size_buf);

        const int64_t block_id_val = dst_id;
        int64_t src_bid = -1;
        if (!src_block_ids.empty()) {
          ABSL_DCHECK_LT(k, src_block_ids.size());
          src_bid = src_block_ids[k];
        }
        std::vector<BlockChunk> chunks = delegate_->GetBlockChunks(
            l, sh, absl::MakeConstSpan(&block_id_val, 1), sender_size,
            header.uuid, static_cast<int64_t>(header.remote_id),
            /*peer=*/"", src_bid);
        if (chunks.empty()) {
          return absl::NotFoundError(
              absl::StrCat("No transfer chunks found for block ", dst_id,
                           " and uuid ", header.uuid));
        }
        ABSL_RETURN_IF_ERROR(ValidateChunks(delegate_, l, sh, chunks));

        uint32_t expected_size = 0;
        for (const auto& chunk : chunks) {
          expected_size += chunk.size;
        }
        if (sender_size != expected_size) {
          return absl::InternalError(absl::StrCat(
              "Incoming push size mismatch: sender offered ", sender_size,
              " bytes, receiver expected ", expected_size, " bytes (chunks=",
              chunks.size(), ") uuid=", header.uuid, " layer/pool=", l,
              " src_block=", src_bid, " dst_block=", dst_id));
        }

        if (expected_size > 0) {
          ABSL_RETURN_IF_ERROR(ReadVExact(client_fd, ToIovec(chunks),
                                          config_.payload_read_timeout_ms));
          total_received_bytes += expected_size;
        }
        return absl::OkStatus();
      }));

  if (total_received_bytes > 0) {
    // TODO: Add interface name (e.g. eth0, lo) using
    // GetSocketLocalNic(client_fd) as a label key.
    RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
        metric_names::kReceivedBytesTotal, kPushLabels, total_received_bytes);
  }

  ABSL_RETURN_IF_ERROR(
      CompleteIncomingPush(header.local_id, header.remote_id, header.reserved,
                           header.buffer_id == 0 ? 0 : stream_shards.size(),
                           push_spec.pool_progress_spec));

  // Report exactly the shards this stream carried: every shard for a legacy
  // stream, one route's subset for a routed stream. The delegate owns
  // per-block shard accounting across streams.
  ABSL_RETURN_IF_ERROR(delegate_->OnBlockShardsReceived(
      allocated_ids, stream_shards, header.uuid));
  incoming_push_lease_held = false;
  ABSL_RETURN_IF_ERROR(EndIncomingPush(session));
  uint8_t ack = 1;
  FaultInjectSocket(hooks::kBlockTransportRecvSendAck, client_fd);
  ABSL_RETURN_IF_ERROR(
      WriteExact(client_fd, &ack, 1, config_.ack_write_timeout_ms));
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Peregrine receive path.
// ---------------------------------------------------------------------------
absl::Status BlockTransportServerPushSession::OnPeregrinePostComplete(
    const lib::PostMetadata& post,
    absl::Span<const lib::RequestMetadata> requests) {
  // Today the sender issues one Post per block array and describes it entirely
  // in the post metadata; per-request metadata is not interpreted yet.
  if (!requests.empty()) {
    return absl::UnimplementedError(
        "OnRecvComplete does not interpret per-request metadata yet");
  }
  if (post.local_id == 0xFFFF'FFFF) {
    return absl::InvalidArgumentError(
        "OnRecvComplete requires an explicit layer/pool index in post "
        "metadata");
  }
  // Shards this Post carried: every shard for a legacy push, one NUMA group's
  // subset for a push split by source NUMA node.
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<int> shards,
      ShardsFromMask(post.shard_mask, delegate_->num_shards()));

  ABSL_ASSIGN_OR_RETURN(
      IncomingPushSpec push_spec,
      ResolveIncomingPushSpec(delegate_, post.local_id,
                              /*is_explicit_dst_push=*/true, uuid_));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<TransferSession> session,
                        BeginIncomingPush(delegate_, uuid_));
  absl::Status s = CompleteIncomingPush(
      post.local_id, post.remote_id, /*num_streams=*/1,
      post.shard_mask == 0 ? 0 : shards.size(), push_spec.pool_progress_spec,
      /*is_peregrine=*/true);
  return EndIncomingPush(session, s);
}

size_t BlockTransportServerPushSession::AccumulateLandedShards(
    LayerProgress::SenderStreams& streams, size_t shards, size_t full_shards) {
  streams.landed_shards += shards == 0 ? full_shards : shards;
  const size_t pushes = streams.landed_shards / full_shards;
  streams.landed += pushes;
  streams.landed_shards %= full_shards;
  return pushes;
}

absl::Status BlockTransportServerPushSession::CompleteIncomingPush(
    uint32_t local_id, uint32_t remote_id, size_t num_streams,
    size_t landed_shards,
    const std::optional<PoolPushProgressSpec>& pool_progress_spec,
    bool is_peregrine) {
  // Unified receive accounting: one progress map and one increment path for
  // both contracts; only the expectation source and the completion callback
  // differ. Plan-declared mode counts every sender's streams against the
  // plan's global per-pool total and retires the uuid when all declared pools
  // complete; header-declared (legacy) mode counts this push's parallelism
  // from num_streams and retires the uuid when all constructor layers
  // complete.
  const int l = (local_id == 0xFFFFFFFF) ? 0 : static_cast<int>(local_id);
  const bool plan_declared = pool_progress_spec.has_value();
  const size_t expected_chunks =
      plan_declared ? pool_progress_spec->expected_pushes : num_streams;

  std::optional<size_t> expected_senders;
  bool resolve_senders = false;
  if (!plan_declared) {
    absl::MutexLock lock(mu_);
    auto it = layer_progress_.find(l);
    if (it != layer_progress_.end() && it->second.expected_senders_resolved) {
      expected_senders = it->second.expected_senders;
    } else {
      resolve_senders = true;
    }
  }
  if (resolve_senders) {
    expected_senders = delegate_->ExpectedPushSenders(uuid_);
  }

  bool trigger_completion = false;
  {
    absl::MutexLock lock(mu_);
    auto& progress = layer_progress_[l];
    // A receive plan assembled from several senders completes a block array
    // only when every declared sender's streams for it have landed. The plan's
    // sender count is fixed for the life of the uuid, so it is resolved on the
    // array's first stream and kept with the array's progress.
    if (!plan_declared) {
      if (!progress.expected_senders_resolved) {
        progress.expected_senders = expected_senders;
        progress.expected_senders_resolved = true;
      }
      expected_senders = progress.expected_senders;
    }
    progress.completed_chunks++;
    bool array_complete = false;
    const size_t full_shards = std::max<size_t>(1, delegate_->num_shards());
    if (plan_declared) {
      // The plan counts pushes that each carry every shard. A sender that
      // split its push by source NUMA node spreads those shards over several
      // streams, so count landed shards per sender and fold every full shard
      // set back into one plan push. The sum reaches the plan's count only
      // once every sender's streams have all landed, so a partial group can
      // never complete the pool early.
      AccumulateLandedShards(progress.sender_streams[remote_id], landed_shards,
                             full_shards);
      size_t plan_pushes = 0;
      for (const auto& [sender, streams] : progress.sender_streams) {
        plan_pushes += streams.landed;
      }
      if (plan_pushes > expected_chunks) {
        return absl::AlreadyExistsError(
            absl::StrCat("pool ", l, " received more than ", expected_chunks,
                         " pushes for UUID ", uuid_));
      }
      array_complete = plan_pushes == expected_chunks;
    } else if (is_peregrine) {
      const size_t senders = expected_senders.value_or(1);
      if (senders == 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "block array ", l, " expected 0 senders for UUID ", uuid_));
      }
      auto& streams = progress.sender_streams[remote_id];
      if (progress.sender_streams.size() > senders) {
        return absl::AlreadyExistsError(
            absl::StrCat("block array ", l, " received pushes from more than ",
                         senders, " senders for UUID ", uuid_));
      }
      const size_t pushes =
          AccumulateLandedShards(streams, landed_shards, full_shards);
      if (streams.landed > 1 ||
          (streams.landed == 1 && streams.landed_shards > 0)) {
        return absl::AlreadyExistsError(absl::StrCat(
            "block array ", l, " received more than one push from sender ",
            remote_id, " for UUID ", uuid_));
      }
      if (pushes > 0 && streams.landed == pushes) {
        ++progress.senders_complete;
      }
      array_complete = progress.senders_complete >= senders;
    } else if (expected_senders.has_value()) {
      auto& streams = progress.sender_streams[remote_id];
      if (streams.landed == 0) {
        if (num_streams == 0) {
          return absl::InvalidArgumentError(
              absl::StrCat("block array ", l, " sender ", remote_id,
                           " declared no streams for UUID ", uuid_));
        }
        streams.declared = num_streams;
        if (progress.sender_streams.size() > *expected_senders) {
          return absl::AlreadyExistsError(absl::StrCat(
              "block array ", l, " received pushes from more than ",
              *expected_senders, " senders for UUID ", uuid_));
        }
      } else if (streams.declared != num_streams) {
        return absl::FailedPreconditionError(
            absl::StrCat("block array ", l, " sender ", remote_id, " declared ",
                         num_streams, " streams after declaring ",
                         streams.declared, " for UUID ", uuid_));
      }
      ++streams.landed;
      if (streams.landed == streams.declared) {
        ++progress.senders_complete;
      } else if (streams.landed > streams.declared) {
        return absl::AlreadyExistsError(
            absl::StrCat("block array ", l, " received ", streams.landed,
                         " streams from sender ", remote_id, " which declared ",
                         streams.declared, " for UUID ", uuid_));
      }
      array_complete = progress.senders_complete == *expected_senders;
    } else {
      array_complete = progress.completed_chunks == expected_chunks;
    }

    if (array_complete && !progress.on_layer_received_called) {
      progress.on_layer_received_called = true;
      trigger_completion = true;
      if (plan_declared) {
        size_t completed_pools = 0;
        for (const auto& [pool_id, pool_progress] : layer_progress_) {
          if (pool_progress.on_layer_received_called) {
            ++completed_pools;
          }
        }
        if (completed_pools == pool_progress_spec->expected_pools) {
          layer_progress_.clear();
          done_ = true;
        }
      } else {
        bool all_layers_called = true;
        for (size_t layer = 0; layer < delegate_->num_layers(); ++layer) {
          auto it = layer_progress_.find(layer);
          if (it == layer_progress_.end() ||
              !it->second.on_layer_received_called) {
            all_layers_called = false;
            break;
          }
        }
        if (all_layers_called) {
          layer_progress_.clear();
          done_ = true;
        }
      }
    }
  }

  // Telemetry-only hook: a failure must not block layer completion below.
  if (is_peregrine && !plan_declared) {
    if (absl::Status s = delegate_->OnRecvProgress(uuid_); !s.ok()) {
      LOG(WARNING) << "OnRecvProgress failed for uuid " << uuid_ << ": " << s;
    }
  }
  if (trigger_completion) {
    ABSL_RETURN_IF_ERROR(
        FaultInjectStatus(hooks::kBlockTransportRecvLayerDone));
    if (plan_declared) {
      ABSL_RETURN_IF_ERROR(delegate_->OnPoolReceived(l, uuid_));
    } else {
      absl::Status layer_status = delegate_->OnLayerReceived(l, uuid_);
      bool all_layers_dispatched = false;
      if (is_peregrine) {
        absl::MutexLock lock(mu_);
        if (++layers_dispatched_ >= delegate_->num_layers()) {
          all_layers_dispatched = true;
        }
      }
      ABSL_RETURN_IF_ERROR(layer_status);
      if (all_layers_dispatched) {
        ABSL_RETURN_IF_ERROR(delegate_->OnRecvNetworkComplete(uuid_));
      }
    }
  }
  return absl::OkStatus();
}

// ===========================================================================
// BlockTransportServerPullSession
// ===========================================================================

BlockTransportServerPullSession::BlockTransportServerPullSession(
    uint64_t uuid, BlockTransportDelegate* delegate,
    const BlockTransportConfig& config, absl::Span<const std::string> local_ips,
    OnCompletedCallback on_completed)
    : uuid_(uuid),
      delegate_(delegate),
      config_(config),
      local_ips_(local_ips.begin(), local_ips.end()),
      on_completed_(std::move(on_completed)) {}

BlockTransportServerPullSession::~BlockTransportServerPullSession() {
  Cancel();
}

bool BlockTransportServerPullSession::IsDone() const {
  absl::MutexLock lock(mu_);
  return done_;
}

absl::Status BlockTransportServerPullSession::status() const {
  absl::MutexLock lock(mu_);
  return status_;
}

void BlockTransportServerPullSession::Cancel() {
  OnCompletedCallback cb;
  {
    absl::MutexLock lock(mu_);
    if (done_) return;
    done_ = true;
    cancelled_ = true;
    status_ = absl::CancelledError("Pull session cancelled");
    if (client_fd_ >= 0) {
      shutdown(client_fd_, SHUT_RDWR);
      client_fd_ = -1;
    }
    cb = std::move(on_completed_);
  }
  if (cb) cb(uuid_);
}

void BlockTransportServerPullSession::Finish(absl::Status s) {
  OnCompletedCallback cb;
  {
    absl::MutexLock lock(mu_);
    if (done_) return;
    done_ = true;
    status_ = s;
    if (client_fd_ >= 0) {
      if (!s.ok()) {
        shutdown(client_fd_, SHUT_RDWR);
      }
      client_fd_ = -1;
    }
    cb = std::move(on_completed_);
  }
  if (cb) cb(uuid_);
}

absl::Status BlockTransportServerPullSession::StartStream(
    int client_fd, const lib::ChunkHeader& header) {
  if (header.uuid != uuid_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Header UUID ", header.uuid, " does not match session UUID ", uuid_));
  }
  {
    absl::MutexLock lock(mu_);
    if (cancelled_ || done_) {
      return absl::CancelledError("Pull session already cancelled");
    }
    if (client_fd_ >= 0) {
      return absl::AlreadyExistsError("Pull stream already started");
    }
  }

  ABSL_ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
  if (delegate_->shard_factor() == 0) {
    return absl::InvalidArgumentError("shard_factor must be positive");
  }
  if (header.count_or_size % delegate_->shard_factor() != 0) {
    return absl::InvalidArgumentError(
        "Requested remote block count is not divisible by shard_factor");
  }

  lib::ChunkHeader resp_header = {};
  resp_header.version = 1;
  resp_header.op = 2;
  resp_header.flags = header.flags;
  resp_header.remote_id = header.local_id;
  resp_header.local_id = 0;
  resp_header.count_or_size = header.count_or_size;
  const auto s = lib::SerializeChunkHeader(resp_header);
  ABSL_RETURN_IF_ERROR(
      WriteExact(client_fd, s.data(), s.size(), config_.ack_write_timeout_ms));

  size_t local_blocks = header.count_or_size / delegate_->shard_factor();
  if (header.remote_id >
          static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) -
                         static_cast<size_t>(header.remote_id)) {
    return absl::OutOfRangeError("Requested block range exceeds int range");
  }

  {
    absl::MutexLock lock(mu_);
    if (cancelled_ || done_) {
      return absl::CancelledError("Pull session cancelled");
    }
    client_fd_ = client_fd;
    remote_id_ = static_cast<int>(header.remote_id);
    count_or_size_ = header.count_or_size;
    major_order_ = major_order;
    total_steps_ =
        delegate_->num_block_arrays() * delegate_->num_shards() * local_blocks;
  }

  TriggerNextSendStep();
  return absl::OkStatus();
}

void BlockTransportServerPullSession::ResolveStepCoordinates(
    size_t* layer, size_t* shard, size_t* block_idx) {
  size_t step;
  {
    absl::MutexLock lock(mu_);
    step = current_step_;
  }
  const size_t num_layers = delegate_->num_block_arrays();
  const size_t num_shards = delegate_->num_shards();
  const size_t local_blocks = count_or_size_ / delegate_->shard_factor();

  if (major_order_ == MajorOrder::kLayerMajor) {
    // layer-major: block_idx innermost, then shard, then layer
    // s = ((l * Sh) + sh) * K + k
    *layer = step / (num_shards * local_blocks);
    size_t rem = step % (num_shards * local_blocks);
    *shard = rem / local_blocks;
    *block_idx = rem % local_blocks;
  } else {
    // block-major: shard innermost, then layer, then block_idx
    // s = ((k * L) + l) * Sh + sh
    *block_idx = step / (num_layers * num_shards);
    size_t rem = step % (num_layers * num_shards);
    *layer = rem / num_shards;
    *shard = rem % num_shards;
  }
}

uint32_t BlockTransportServerPullSession::GetChunksTotalSize(
    const std::vector<BlockChunk>& chunks) {
  uint32_t total_size = 0;
  for (const auto& chunk : chunks) total_size += chunk.size;
  return total_size;
}

void BlockTransportServerPullSession::TriggerNextSendStep() {
  size_t step;
  {
    absl::MutexLock lock(mu_);
    if (cancelled_ || done_) return;
    step = current_step_;
  }
  if (step >= total_steps_) {
    Finish(absl::OkStatus());
    return;
  }

  size_t l = 0, sh = 0, k = 0;
  ResolveStepCoordinates(&l, &sh, &k);
  int block_id = remote_id_ + static_cast<int>(k);

  auto self = shared_from_this();
  delegate_->RegisterBlockReadinessCallback(
      l, sh, block_id, uuid_,
      [self, l, sh, block_id, step](absl::Status ready_status) {
        if (!ready_status.ok()) {
          LOG(ERROR) << "Pull response failed at step " << step << " for uuid "
                     << self->uuid_ << ", status: " << ready_status;
          self->Finish(ready_status);
          return;
        }

        self->delegate_->ScheduleAsyncTask([self, l, sh, block_id]() {
          const int64_t block_id_val = block_id;
          std::vector<BlockChunk> chunks = self->delegate_->GetBlockChunks(
              l, sh, absl::MakeConstSpan(&block_id_val, 1),
              self->delegate_->block_bytes(l), self->uuid_);
          if (chunks.empty()) {
            LOG(ERROR) << "No transfer chunks found for block " << block_id
                       << " and uuid " << self->uuid_;
            self->Finish(absl::NotFoundError(
                absl::StrCat("No chunks for block ", block_id)));
            return;
          }
          absl::Status s = ValidateChunks(self->delegate_, l, sh, chunks);
          if (!s.ok()) {
            LOG(ERROR) << "Chunks validation failed: " << s;
            self->Finish(s);
            return;
          }

          uint32_t total_size = self->GetChunksTotalSize(chunks);
          const std::array<uint8_t, lib::kChunkSizeFieldSize> s_size =
              lib::SerializeChunkSize(total_size);
          s = WriteExact(self->client_fd_, s_size.data(), s_size.size(),
                         self->config_.payload_write_timeout_ms);
          if (!s.ok()) {
            LOG(ERROR) << "Write size failed: " << s;
            self->Finish(s);
            return;
          }
          if (total_size > 0) {
            s = WriteVExact(self->client_fd_, ToIovec(chunks),
                            self->config_.payload_write_timeout_ms);
          }
          if (!s.ok()) {
            LOG(ERROR) << "Write payload failed: " << s;
            self->Finish(s);
            return;
          }

          if (total_size > 0) {
            // TODO: Add interface name (e.g. eth0, lo) using
            // GetSocketLocalNic(client_fd_) as a label key.
            RecordPullResponseSentBytes(self->client_fd_, self->local_ips_,
                                        total_size);
          }

          {
            absl::MutexLock lock(self->mu_);
            self->current_step_++;
          }
          self->TriggerNextSendStep();
        });
      });
}

}  // namespace transport
}  // namespace tpu_raiden

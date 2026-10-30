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

#include "tpu_sync/transport/block_transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/block_transport_session.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/adapter/peregrine_transport_adapter.h"
#include "tpu_sync/transport/lib/adapter/socket_transport_adapter.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/psp/peregrine_control_service.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

namespace {

constexpr absl::Duration kTransportMetricsPollInterval = absl::Seconds(1);

void ApplyPeregrineEnv(BlockTransport::Config& config) {
  const char* raw = std::getenv("TPU_RAIDEN_ENABLE_PEREGRINE");
  if (raw == nullptr) return;
  const std::string val =
      absl::AsciiStrToLower(absl::StripAsciiWhitespace(raw));
  if (val.empty() || val == "false" || val == "0") return;
  if (val == "true" || val == "all") {
    for (lib::TransferOp op :
         {lib::TransferOp::kBlockPush, lib::TransferOp::kBlockPull,
          lib::TransferOp::kBlockPushExplicit}) {
      config.op_backends[op] = TransportBackend::kPeregrine;
    }
    return;
  }
  for (absl::string_view entry :
       absl::StrSplit(val, ',', absl::SkipWhitespace())) {
    int op = 0;
    if (absl::SimpleAtoi(absl::StripAsciiWhitespace(entry), &op) &&
        (op == 1 || op == 2 || op == 6)) {
      config.op_backends[static_cast<lib::TransferOp>(op)] =
          TransportBackend::kPeregrine;
    } else {
      LOG(WARNING) << "TPU_RAIDEN_ENABLE_PEREGRINE=\"" << raw
                   << "\": ignoring entry \"" << entry
                   << "\"; expected true|all, false|0, or a comma-separated "
                      "list of block ops in {1, 2, 6}";
    }
  }
}

BlockTransport::Config ReadConfigFromEnv(bool socket_only_transport) {
  BlockTransport::Config config;
  if (const char* val =
          std::getenv("TPU_RAIDEN_DECODE_HANDSHAKE_READ_TIMEOUT_S");
      val != nullptr && val[0] != '\0') {
    double parsed = 0.0;
    if (absl::SimpleAtod(val, &parsed) && parsed > 0.0) {
      config.handshake_read_timeout_ms =
          static_cast<int>(absl::ToInt64Milliseconds(absl::Seconds(parsed)));
    } else {
      LOG(WARNING) << "TPU_RAIDEN_DECODE_HANDSHAKE_READ_TIMEOUT_S=\"" << val
                   << "\" must be a positive number; using default (no "
                      "timeout)";
    }
  }
  if (const char* val =
          std::getenv("TPU_RAIDEN_DECODE_PAYLOAD_READ_TIMEOUT_S");
      val != nullptr && val[0] != '\0') {
    double parsed = 0.0;
    if (absl::SimpleAtod(val, &parsed) && parsed > 0.0) {
      config.payload_read_timeout_ms =
          static_cast<int>(absl::ToInt64Milliseconds(absl::Seconds(parsed)));
    } else {
      LOG(WARNING) << "TPU_RAIDEN_DECODE_PAYLOAD_READ_TIMEOUT_S=\"" << val
                   << "\" must be a positive number; using default (no "
                      "timeout)";
    }
  }
  if (const char* val = std::getenv("TPU_RAIDEN_DECODE_ACK_WRITE_TIMEOUT_S");
      val != nullptr && val[0] != '\0') {
    double parsed = 0.0;
    if (absl::SimpleAtod(val, &parsed) && parsed > 0.0) {
      config.ack_write_timeout_ms =
          static_cast<int>(absl::ToInt64Milliseconds(absl::Seconds(parsed)));
    } else {
      LOG(WARNING) << "TPU_RAIDEN_DECODE_ACK_WRITE_TIMEOUT_S=\"" << val
                   << "\" must be a positive number; using default (no "
                      "timeout)";
    }
  }
  if (const char* val =
          std::getenv("TPU_RAIDEN_DECODE_PAYLOAD_WRITE_TIMEOUT_S");
      val != nullptr && val[0] != '\0') {
    double parsed = 0.0;
    if (absl::SimpleAtod(val, &parsed) && parsed > 0.0) {
      config.payload_write_timeout_ms =
          static_cast<int>(absl::ToInt64Milliseconds(absl::Seconds(parsed)));
    } else {
      LOG(WARNING) << "TPU_RAIDEN_DECODE_PAYLOAD_WRITE_TIMEOUT_S=\"" << val
                   << "\" must be a positive number; using default (no "
                      "timeout)";
    }
  }
  if (const char* val = std::getenv("RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES");
      val != nullptr && val[0] != '\0') {
    size_t parsed = 0;
    if (absl::SimpleAtoi(val, &parsed)) {
      config.coalesce_window_bytes = parsed;
    }
  }
  if (!socket_only_transport) ApplyPeregrineEnv(config);
  return config;
}

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

constexpr uint8_t kUseBlockChunksFlag = 0x80;

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

// Shards of one push that share a NUMA node and therefore one set of streams.
struct ShardGroup {
  int numa_node;            // -1 = unknown / legacy.
  std::vector<int> shards;  // Ascending local shard indices.
  uint16_t shard_mask;      // 0 for the single legacy group.
};

// Groups `shards` by NUMA node, ordered by first appearance. Returns a single
// legacy group (every shard, mask 0, node -1) whenever a split is not possible
// or not useful: no placement info, every shard on one node, any shard on an
// unknown node, or more shards than fit in the wire mask.
std::vector<ShardGroup> ShardGroupsByNuma(absl::Span<const ShardInfo> shards,
                                          size_t num_shards) {
  std::vector<ShardGroup> groups;
  if (shards.size() == num_shards &&
      num_shards <= static_cast<size_t>(kMaxShardMaskBits)) {
    for (const ShardInfo& info : shards) {
      if (info.numa_node < 0 || info.local_index < 0 ||
          static_cast<size_t>(info.local_index) >= num_shards) {
        groups.clear();
        break;
      }
      auto it = std::find_if(
          groups.begin(), groups.end(),
          [&](const ShardGroup& g) { return g.numa_node == info.numa_node; });
      if (it == groups.end()) {
        groups.push_back(ShardGroup{info.numa_node, {}, 0});
        it = std::prev(groups.end());
      }
      it->shards.push_back(info.local_index);
      it->shard_mask |= static_cast<uint16_t>(1u << info.local_index);
    }
  }
  if (groups.size() <= 1) {
    return {ShardGroup{-1, AllShards(num_shards), 0}};
  }
  for (ShardGroup& g : groups) std::sort(g.shards.begin(), g.shards.end());
  return groups;
}

}  // namespace

BlockTransport::BlockTransport(BlockTransportDelegate* delegate, int local_port,
                               const std::vector<std::string>& local_ips,
                               int parallelism, bool socket_only_transport)
    : block_delegate_(delegate),
      parallelism_(parallelism),
      config_(ReadConfigFromEnv(socket_only_transport)),
      raw_transport_(
          delegate, local_port, local_ips,
          [this](int client_fd, const lib::ChunkHeader& header) {
            return HandleCustomRequest(client_fd, header);
          },
          config_.coalesce_window_bytes),
      peregrine_control_(
          std::make_unique<lib::PeregrineControlServiceImpl>(&raw_transport_)),
      transport_adapter_(std::make_unique<lib::SocketTransportAdapter>(
          &raw_transport_, parallelism_)) {
  // Pin a send adapter to each NUMA node the local shards span, but only when
  // they span more than one: that is the only case where pushes are split by
  // source node. Single-node managers keep the legacy send path untouched.
  std::vector<int> send_numa_nodes;
  for (const ShardInfo& shard : block_delegate_->shards()) {
    if (shard.numa_node >= 0 &&
        !absl::c_linear_search(send_numa_nodes, shard.numa_node)) {
      send_numa_nodes.push_back(shard.numa_node);
    }
  }
  if (send_numa_nodes.size() > 1) {
    for (const int node : send_numa_nodes) {
      numa_adapters_[node] = std::make_unique<lib::SocketTransportAdapter>(
          &raw_transport_, parallelism_, node);
    }
  }
  // The Peregrine transport is started only when a block op selects it; the
  // default socket configuration never touches Peregrine.
  if (config_.uses_peregrine()) {
    absl::StatusOr<std::unique_ptr<lib::PeregrineTransportAdapter>> adapter =
        lib::PeregrineTransportAdapter::Create(raw_transport_.bound_ip(),
                                               /*port=*/0, parallelism_);
    if (adapter.ok()) {
      peregrine_adapter_ = *std::move(adapter);
    } else {
      LOG(ERROR) << "Peregrine transport backend requested but unavailable; "
                    "block ops configured for it will fail: "
                 << adapter.status();
    }
  }
  LOG(INFO) << "local_port=" << local_port
            << ", local_ips=" << absl::StrJoin(local_ips, ",")
            << ", parallelism=" << parallelism_
            << ", numa_send_adapters=" << numa_adapters_.size();
  metrics_thread_ = std::thread([this]() {
    while (!stop_metrics_thread_.WaitForNotificationWithTimeout(
        kTransportMetricsPollInterval)) {
      metrics_exporter_.Export(transport_adapter_->GetTransportMetrics());
    }
  });
}

BlockTransport::~BlockTransport() {
  stop_metrics_thread_.Notify();
  if (metrics_thread_.joinable()) {
    metrics_thread_.join();
  }
  std::vector<std::shared_ptr<BlockTransportServerPullSession>> pull_sessions;
  std::vector<std::shared_ptr<BlockTransportServerPushSession>> push_sessions;
  {
    absl::MutexLock lock(server_sessions_mu_);
    for (const auto& [id, session] : active_pull_sessions_) {
      if (session) pull_sessions.push_back(session);
    }
    for (const auto& [uuid, session] : active_push_sessions_) {
      if (session) push_sessions.push_back(session);
    }
    active_push_sessions_.clear();
  }
  for (const auto& session : pull_sessions) {
    session->Cancel();
  }
  for (const auto& session : push_sessions) {
    session->Cancel();
  }
  {
    absl::MutexLock lock(server_sessions_mu_);
    server_sessions_mu_.Await(absl::Condition(
        +[](const absl::flat_hash_map<
             uint64_t, std::shared_ptr<BlockTransportServerPullSession>>* s) {
          return s->empty();
        },
        &active_pull_sessions_));
  }
}

absl::Status BlockTransport::HandleCustomRequest(
    int client_fd, const lib::ChunkHeader& header) {
  LOG(INFO) << "HandleCustomRequest (H2H read start): client_fd=" << client_fd
            << ", op=" << static_cast<int>(header.op)
            << ", uuid=" << header.uuid
            << ", numa=" << block_delegate_->node_id();

  if (header.op == 1 || header.op == 6) {
    absl::Status push_status = HandleIncomingPush(client_fd, header);
    if (!push_status.ok()) {
      // The connection drops after a failed push; without this the sender
      // only sees an anonymous "Push verification failed".
      LOG(ERROR) << "Incoming push rejected: uuid=" << header.uuid
                 << " op=" << static_cast<int>(header.op)
                 << " local_id=" << header.local_id << ": " << push_status;
    }
    return push_status;
  } else if (header.op == 2) {
    return HandleIncomingPull(client_fd, header);
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported block transport op: ", header.op));
  }
}

std::shared_ptr<BlockTransportServerPushSession>
BlockTransport::GetOrCreatePushSession(uint64_t uuid) {
  absl::MutexLock lock(server_sessions_mu_);
  auto& session = active_push_sessions_[uuid];
  if (session != nullptr) {
    return session;
  }
  session = std::make_shared<BlockTransportServerPushSession>(
      uuid, block_delegate_, config_);
  return session;
}

void BlockTransport::DeletePushSession(uint64_t uuid) {
  absl::MutexLock lock(server_sessions_mu_);
  active_push_sessions_.erase(uuid);
}

void BlockTransport::DeletePullSession(uint64_t session_id) {
  absl::MutexLock lock(server_sessions_mu_);
  active_pull_sessions_.erase(session_id);
}

absl::Status BlockTransport::HandleIncomingPush(
    int client_fd, const lib::ChunkHeader& header) {
  std::shared_ptr<BlockTransportServerPushSession> session =
      GetOrCreatePushSession(header.uuid);
  absl::Status status = session->StartStream(client_fd, header);
  if (session->IsDone()) {
    DeletePushSession(header.uuid);
  }
  return status;
}

// ---------------------------------------------------------------------------
// Peregrine receive path.
// ---------------------------------------------------------------------------
absl::Status BlockTransport::OnRecvComplete(
    const lib::PostMetadata& post,
    absl::Span<const lib::RequestMetadata> requests) {
  std::shared_ptr<BlockTransportServerPushSession> session =
      GetOrCreatePushSession(post.uuid);
  absl::Status status = session->OnPeregrinePostComplete(post, requests);
  if (session->IsDone()) {
    DeletePushSession(post.uuid);
  }
  return status;
}

absl::Status BlockTransport::HandleIncomingPull(
    int client_fd, const lib::ChunkHeader& header) {
  uint64_t session_id;
  std::shared_ptr<BlockTransportServerPullSession> session;
  {
    absl::MutexLock lock(server_sessions_mu_);
    session_id = ++next_pull_session_id_;
    session = std::make_shared<BlockTransportServerPullSession>(
        header.uuid, block_delegate_, config_, raw_transport_.local_ips(),
        [this, session_id](uint64_t /*uuid*/) {
          DeletePullSession(session_id);
        });
    active_pull_sessions_[session_id] = session;
  }
  absl::Status status = session->StartStream(client_fd, header);
  if (!status.ok()) {
    DeletePullSession(session_id);
  }
  return status;
}

tsl::Future<std::vector<int>> BlockTransport::AsyncPush(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx,
    std::optional<int> wire_layer_idx) {
  const size_t num_shards = block_delegate_->num_shards();
  const lib::TransferOp op = dst_block_ids.empty()
                                 ? lib::TransferOp::kBlockPush
                                 : lib::TransferOp::kBlockPushExplicit;
  if (config_.backend_for(op) == TransportBackend::kPeregrine) {
    absl::StatusOr<lib::TransportAdapter*> adapter = AdapterForOp(op);
    if (!adapter.ok()) {
      RecordTransferFailure(adapter.status(), metric_labels::kDirectionPush);
      return tsl::Future<std::vector<int>>(adapter.status());
    }
    const std::vector<int> all_shards = AllShards(num_shards);
    return AsyncPushShards(peers, src_block_ids, dst_block_ids, parallelism,
                           major_order, uuid, layer_idx, wire_layer_idx,
                           all_shards, /*num_groups=*/1, *adapter)
        .Map<std::vector<int>>([dst_block_ids](const std::vector<int>&) {
          return dst_block_ids;
        });
  }
  // Split the push by source NUMA node: every group of shards sharing a node
  // is pushed by the adapter pinned to that node. The receiver pairs the
  // per-group shard subsets of one block range by destination id, so a
  // multi-group push requires explicit destinations (op 6); otherwise fall
  // back to one legacy group.
  std::vector<ShardGroup> groups =
      ShardGroupsByNuma(block_delegate_->shards(), num_shards);
  if (groups.size() > 1 && dst_block_ids.empty()) {
    LOG_FIRST_N(WARNING, 1)
        << "Push without explicit destination block ids cannot be split by "
           "NUMA node; sending every shard on each stream.";
    groups = ShardGroupsByNuma({}, num_shards);
  }
  const size_t active_layers =
      std::max<size_t>(1, block_delegate_->max_concurrent_push_layers());
  const size_t num_blocks = std::max<size_t>(1, src_block_ids.size());
  if (!dst_block_ids.empty() &&
      num_shards <= static_cast<size_t>(kMaxShardMaskBits) &&
      static_cast<size_t>(parallelism) > num_blocks &&
      active_layers * groups.size() * num_blocks <
          static_cast<size_t>(parallelism)) {
    const size_t target_groups_per_layer =
        (static_cast<size_t>(parallelism) + active_layers * num_blocks - 1) /
        (active_layers * num_blocks);
    const size_t split_factor =
        (target_groups_per_layer + groups.size() - 1) / groups.size();
    if (split_factor > 1) {
      std::vector<ShardGroup> subdivided;
      for (const ShardGroup& g : groups) {
        if (g.shards.size() <= 1) {
          subdivided.push_back(g);
          continue;
        }
        const size_t sub_count = std::min(g.shards.size(), split_factor);
        const size_t base_sz = g.shards.size() / sub_count;
        const size_t rem = g.shards.size() % sub_count;
        size_t offset = 0;
        for (size_t i = 0; i < sub_count; ++i) {
          const size_t cnt = base_sz + (i < rem ? 1 : 0);
          uint16_t mask = 0;
          for (size_t k = 0; k < cnt; ++k) {
            int s = g.shards[offset + k];
            if (s >= 0 && s < kMaxShardMaskBits) {
              mask |= static_cast<uint16_t>(1u << s);
            }
          }
          subdivided.push_back(ShardGroup{
              .numa_node = g.numa_node,
              .shards = std::vector<int>(g.shards.begin() + offset,
                                         g.shards.begin() + offset + cnt),
              .shard_mask = mask,
          });
          offset += cnt;
        }
      }
      groups = std::move(subdivided);
    }
  }
  std::vector<tsl::Future<std::vector<int>>> futures;
  futures.reserve(groups.size());
  for (const ShardGroup& group : groups) {
    futures.push_back(AsyncPushShards(
        peers, src_block_ids, dst_block_ids, parallelism, major_order, uuid,
        layer_idx, wire_layer_idx, group.shards, groups.size(),
        AdapterFor(group.numa_node)));
  }
  // Every group returns the same destination ids; keep the first.
  return tsl::JoinFutures(absl::MakeSpan(futures))
      .Map<std::vector<int>>(
          [](const std::vector<std::vector<int>>& ids) { return ids.front(); });
}

tsl::Future<std::vector<int>> BlockTransport::AsyncPushShards(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx,
    std::optional<int> wire_layer_idx, absl::Span<const int> shard_ids,
    size_t num_groups, lib::TransportAdapter* adapter) {
  auto [promise, future] = tsl::MakePromise<std::vector<int>>();
  // The adapter's completion callback must be copyable, while tsl::Promise is
  // move-only.
  auto shared_promise =
      std::make_shared<tsl::Promise<std::vector<int>>>(std::move(promise));
  auto on_complete = [shared_promise](absl::StatusOr<std::vector<int>> res) {
    if (!res.ok()) {
      RecordTransferFailure(res.status(), metric_labels::kDirectionPush);
    }
    shared_promise->Set(std::move(res));
  };
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    on_complete(absl::InvalidArgumentError("Block list cannot be empty"));
    return std::move(future);
  }
  if (peers.empty()) {
    on_complete(absl::InvalidArgumentError("Peer list cannot be empty"));
    return std::move(future);
  }

  int P = parallelism;
  if (P <= 0) {
    on_complete(absl::InvalidArgumentError("parallelism must be positive"));
    return std::move(future);
  }
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  // The receiver completes a block array once it has seen the stream count
  // declared in the header, i.e. the streams of every shard group.
  const size_t total_streams = static_cast<size_t>(P) * num_groups;
  if (total_streams > std::numeric_limits<uint16_t>::max()) {
    on_complete(absl::InvalidArgumentError(
        absl::StrCat("push declares ", total_streams,
                     " streams, exceeding the header limit")));
    return std::move(future);
  }

  // In multi-NIC setups, `peers` contains all NIC rail endpoints for the
  // destination host. Request chunk resolution is identical across NICs, so
  // we pass `peers[0]` as the destination peer to build requests.
  auto requests = BuildBlockRequests(
      peers[0], src_block_ids, dst_block_ids, major_order, uuid, layer_idx, P,
      wire_layer_idx, shard_ids, static_cast<int>(total_streams));
  if (!requests.ok()) {
    on_complete(requests.status());
    return std::move(future);
  }

  // Post reports its own errors through `on_complete`, so the returned status
  // carries no additional information.
  adapter
      ->Post(peers, *requests, src_block_ids, dst_block_ids,
             std::move(on_complete))
      .status()
      .IgnoreError();
  return std::move(future);
}

absl::StatusOr<std::vector<int>> BlockTransport::SyncPull(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    MajorOrder major_order, BlockReceivedCallback on_block_received,
    uint64_t uuid) {
  auto res =
      SyncPullInternal(peers, src_block_ids, local_block_ids, explicit_dst_ptrs,
                       parallelism, major_order, on_block_received, uuid);
  if (!res.ok()) {
    RecordTransferFailure(res.status(), metric_labels::kDirectionPull);
  }
  return res;
}

absl::StatusOr<std::vector<int>> BlockTransport::SyncPullInternal(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    MajorOrder major_order, BlockReceivedCallback on_block_received,
    uint64_t uuid) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }
  if (peers.empty()) {
    return absl::InvalidArgumentError("Peer list cannot be empty");
  }

  if (block_delegate_->shard_factor() == 0) {
    return absl::InvalidArgumentError("shard_factor must be positive");
  }
  if (num_blocks % block_delegate_->shard_factor() != 0) {
    return absl::InvalidArgumentError(
        "Block count must be divisible by shard_factor");
  }
  size_t local_blocks = num_blocks / block_delegate_->shard_factor();
  if (local_blocks == 0) {
    return absl::InvalidArgumentError("Local block list cannot be empty");
  }
  if (!explicit_dst_ptrs.empty() &&
      explicit_dst_ptrs.size() !=
          block_delegate_->num_block_arrays() * block_delegate_->num_shards()) {
    return absl::InvalidArgumentError("explicit_dst_ptrs size mismatch");
  }

  std::vector<int> allocated_ids;
  if (!local_block_ids.empty()) {
    if (local_block_ids.size() != local_blocks) {
      return absl::InvalidArgumentError("local_block_ids size mismatch");
    }
    allocated_ids = local_block_ids;
  } else {
    ABSL_ASSIGN_OR_RETURN(allocated_ids,
                          block_delegate_->AllocateBlocks(local_blocks));
  }

  int P = parallelism;
  if (P <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  ABSL_ASSIGN_OR_RETURN(
      const auto requests,
      BuildBlockPullRequests(src_block_ids, allocated_ids, explicit_dst_ptrs,
                             major_order, uuid, P, on_block_received));

  ABSL_ASSIGN_OR_RETURN(lib::TransportAdapter* const adapter,
                        AdapterForOp(lib::TransferOp::kBlockPull));
  ABSL_RETURN_IF_ERROR(adapter->Post(peers, requests).status());
  return allocated_ids;
}

lib::Request BlockTransport::BuildBlockRequest(
    lib::TransferOp op, uint8_t* laddr, uint8_t* raddr, size_t len,
    uint32_t count_or_size, uint32_t request_id, uint64_t uuid,
    uint64_t buffer_id, int parallelism, MajorOrder major_order,
    uint32_t remote_id, uint32_t local_id, int stream_idx,
    BlockReceivedCallback on_block_received, uint16_t shard_mask,
    int total_streams) {
  return lib::Request{
      .op = op,
      .laddr = laddr,
      .raddr = raddr,
      .len = len,
      .buffer_id = buffer_id,
      .major_order = static_cast<uint8_t>(major_order),
      .parallelism = parallelism,
      .remote_id = remote_id,
      .local_id = local_id,
      .count_or_size = count_or_size,
      .uuid = uuid,
      .request_id = request_id,
      .stream_idx = stream_idx,
      .shard_mask = shard_mask,
      .total_streams = total_streams,
      .on_block_received = std::move(on_block_received),
  };
}

absl::StatusOr<std::vector<lib::Request>> BlockTransport::BuildBlockRequests(
    absl::string_view peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, MajorOrder major_order,
    uint64_t uuid, int layer_idx, int parallelism,
    std::optional<int> wire_layer_idx, absl::Span<const int> shard_ids,
    int total_streams) {
  const size_t num_blocks = src_block_ids.size();
  const lib::TransferOp op = dst_block_ids.empty()
                                 ? lib::TransferOp::kBlockPush
                                 : lib::TransferOp::kBlockPushExplicit;
  const uint32_t remote_id = static_cast<uint32_t>(block_delegate_->node_id());
  // Chunks are read from the local block array `layer_idx`; the request
  // header names the array the receiver writes into.
  const int header_layer_idx = wire_layer_idx.value_or(layer_idx);
  const uint32_t local_id = header_layer_idx == -1
                                ? 0xFFFF'FFFF
                                : static_cast<uint32_t>(header_layer_idx);
  if (num_blocks == 0) {
    return std::vector<lib::Request>{BuildBlockRequest(
        op, /*laddr=*/nullptr, /*raddr=*/nullptr, /*len=*/0,
        /*count_or_size=*/0, /*request_id=*/0, uuid, /*buffer_id=*/0,
        parallelism, major_order, remote_id, local_id)};
  }
  if (parallelism <= 0 || static_cast<size_t>(parallelism) > num_blocks) {
    return absl::InvalidArgumentError(absl::StrCat(
        "parallelism ", parallelism, " must be in [1, ", num_blocks, "]"));
  }

  // The wire shard mask stays 0 (legacy "every shard") unless this call
  // covers a strict subset, which the receiver must be able to decode.
  const size_t num_shards = block_delegate_->num_shards();
  uint16_t shard_mask = 0;
  if (shard_ids.size() != num_shards) {
    if (shard_ids.empty()) {
      return absl::InvalidArgumentError("shard_ids cannot be empty");
    }
    for (int sh : shard_ids) {
      if (sh < 0 || static_cast<size_t>(sh) >= num_shards ||
          sh >= kMaxShardMaskBits) {
        return absl::InvalidArgumentError(absl::StrCat(
            "shard ", sh, " cannot be addressed by a push shard mask (",
            num_shards, " shards, mask limit ", kMaxShardMaskBits, ")"));
      }
      shard_mask |= static_cast<uint16_t>(1u << sh);
    }
  }

  std::vector<int> target_layers;
  if (layer_idx == -1) {
    target_layers.resize(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    target_layers = {layer_idx};
  }

  std::vector<lib::Request> requests;
  requests.reserve(target_layers.size() * shard_ids.size() * num_blocks);

  // One stream per `stream_idx`, over an even block partition; the adapter
  // derives the same partition from `stream_idx` and `parallelism`.
  const size_t base_blocks_per_stream = num_blocks / parallelism;
  const size_t remainder = num_blocks % parallelism;
  for (int stream_idx = 0; stream_idx < parallelism; ++stream_idx) {
    const size_t block_count =
        base_blocks_per_stream +
        (static_cast<size_t>(stream_idx) < remainder ? 1 : 0);
    const size_t block_offset = stream_idx * base_blocks_per_stream +
                                std::min<size_t>(stream_idx, remainder);
    const uint32_t count_or_size = static_cast<uint32_t>(block_count);
    uint32_t request_id = 0;
    ABSL_RETURN_IF_ERROR(ForEachPayload(
        major_order, target_layers, shard_ids, block_count,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          ABSL_DCHECK_LT(block_offset + k, src_block_ids.size());
          const int src_id = src_block_ids[block_offset + k];

          const int64_t block_id_val = src_id;
          const int64_t dst_id_val =
              block_offset + k < dst_block_ids.size()
                  ? static_cast<int64_t>(dst_block_ids[block_offset + k])
                  : -1;
          std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
              l, sh, absl::MakeConstSpan(&block_id_val, 1),
              block_delegate_->block_bytes(l), uuid, -1, peer,
              /*src_block_id=*/-1, /*dst_block_id=*/dst_id_val);
          if (chunks.empty()) {
            return absl::NotFoundError(
                absl::StrCat("No transfer chunks found for block ", src_id,
                             " and uuid ", uuid));
          }
          ABSL_RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

          // What the receiver needs to handle these chunks. Uses
          // `wire_layer_idx` when set (the receiver's array index), otherwise
          // the iterated layer `l` (not `layer_idx`, which is -1 for all-layer
          // pushes).
          const size_t wire_layer =
              wire_layer_idx.has_value() ? static_cast<size_t>(*wire_layer_idx)
                                         : l;
          const uint64_t buffer_id =
              (static_cast<uint64_t>(wire_layer) << 32) |
              static_cast<uint32_t>(sh);
          for (const auto& chunk : chunks) {
            requests.push_back(BuildBlockRequest(
                op, chunk.ptr, chunk.raddr, chunk.size, count_or_size,
                request_id, uuid, buffer_id, parallelism, major_order,
                remote_id, local_id, stream_idx,
                /*on_block_received=*/nullptr, shard_mask, total_streams));
          }
          ++request_id;
          return absl::OkStatus();
        }));
  }
  return requests;
}

lib::TransportAdapter* BlockTransport::AdapterFor(int numa_node) {
  auto it = numa_adapters_.find(numa_node);
  return it != numa_adapters_.end() ? it->second.get()
                                    : transport_adapter_.get();
}

absl::StatusOr<lib::TransportAdapter*> BlockTransport::AdapterForOp(
    lib::TransferOp op) {
  switch (config_.backend_for(op)) {
    case TransportBackend::kSocket:
      return transport_adapter_.get();
    case TransportBackend::kPeregrine:
      if (peregrine_adapter_ == nullptr) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Block op ", static_cast<int>(op),
            " is configured for the Peregrine backend, which failed to start "
            "(see the BlockTransport startup log)"));
      }
      return peregrine_adapter_.get();
  }
  return absl::InternalError(absl::StrCat(
      "Unknown transport backend ", static_cast<int>(config_.backend_for(op)),
      " for block op ", static_cast<int>(op)));
}

absl::StatusOr<std::vector<lib::Request>>
BlockTransport::BuildBlockPullRequests(
    const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, MajorOrder major_order,
    uint64_t uuid, int parallelism, BlockReceivedCallback on_block_received) {
  if (parallelism <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }

  const size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return std::vector<lib::Request>{};
  }
  const size_t SF = block_delegate_->shard_factor();
  const size_t local_blocks = num_blocks / SF;
  const size_t base_blocks_per_stream = local_blocks / parallelism;
  const size_t remainder = local_blocks % parallelism;

  std::vector<lib::Request> requests;

  for (int i = 0; i < parallelism; ++i) {
    const size_t local_block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    const size_t local_block_offset =
        i * base_blocks_per_stream + std::min<size_t>(i, remainder);
    const size_t remote_block_count = local_block_count * SF;
    const size_t remote_block_offset = local_block_offset * SF;

    if (remote_block_offset > src_block_ids.size() ||
        remote_block_count > src_block_ids.size() - remote_block_offset) {
      return absl::OutOfRangeError(
          "Remote block range exceeds source block list");
    }

  struct PullChunk {
    size_t local_start_idx;
    size_t local_count;
    int base_remote_id;
    size_t remote_count;
  };
  std::vector<PullChunk> chunks;

  if (local_block_count > 0) {
    size_t curr_local_start = 0;
    size_t curr_local_count = 1;
    ABSL_DCHECK_LT(remote_block_offset, src_block_ids.size());
    int curr_base_remote_id = src_block_ids[remote_block_offset];

    for (size_t k = 1; k < local_block_count; ++k) {
      ABSL_DCHECK_LT(remote_block_offset + k * SF - 1, src_block_ids.size());
      int prev_last_remote = src_block_ids[remote_block_offset + k * SF - 1];
      ABSL_DCHECK_LT(remote_block_offset + k * SF, src_block_ids.size());
      int curr_first_remote = src_block_ids[remote_block_offset + k * SF];

      if (curr_first_remote == prev_last_remote + 1) {
        curr_local_count++;
      } else {
        chunks.push_back({curr_local_start, curr_local_count,
                          curr_base_remote_id, curr_local_count * SF});
        curr_local_start = k;
        curr_local_count = 1;
        curr_base_remote_id = curr_first_remote;
      }
    }
    chunks.push_back({curr_local_start, curr_local_count, curr_base_remote_id,
                      curr_local_count * SF});
  }

  uint32_t request_id = 0;

  for (const auto& chunk : chunks) {
    const int remote_read_block_id =
        block_delegate_->GetRemoteReadBlockId(chunk.base_remote_id, 0);
    if (remote_read_block_id < 0 ||
        static_cast<uint64_t>(remote_read_block_id) >
            std::numeric_limits<uint32_t>::max() ||
        chunk.remote_count > std::numeric_limits<uint32_t>::max()) {
      return absl::OutOfRangeError(
          "Remote block range exceeds transport header");
    }

    const uint32_t remote_id = static_cast<uint32_t>(remote_read_block_id);
    const uint32_t count_or_size = static_cast<uint32_t>(chunk.remote_count);

    std::vector<int> target_layers(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
    const std::vector<int> all_shards =
        AllShards(block_delegate_->num_shards());
    ABSL_RETURN_IF_ERROR(ForEachPayload(
        major_order, target_layers, all_shards, chunk.local_count,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          ABSL_DCHECK_LT(local_block_offset + chunk.local_start_idx + k,
                         allocated_ids.size());
          const int dst_id =
              allocated_ids[local_block_offset + chunk.local_start_idx + k];
          if (dst_id < 0) {
            return absl::InvalidArgumentError(
                "Destination block id is negative");
          }

          const int64_t block_id_val = dst_id;
          std::vector<BlockChunk> block_chunks =
              block_delegate_->GetBlockChunks(
                  l, sh, absl::MakeConstSpan(&block_id_val, 1),
                  block_delegate_->block_bytes(l), uuid);
          if (block_chunks.empty()) {
            return absl::NotFoundError(
                absl::StrCat("No transfer chunks found for block ", dst_id,
                             " and uuid ", uuid));
          }
          ABSL_RETURN_IF_ERROR(
              ValidateChunks(block_delegate_, l, sh, block_chunks));

          uint8_t* default_base = nullptr;
          uint8_t* explicit_base = nullptr;
          if (!explicit_dst_ptrs.empty()) {
            default_base = block_delegate_->GetBlockArrayHostPointer(l, sh);
            explicit_base =
                explicit_dst_ptrs[l * block_delegate_->num_shards() + sh];
          }

          const uint32_t local_id = static_cast<uint32_t>(dst_id);
          // What the source serving this pull needs to locate and bounds-check
          // the data.
          const uint64_t buffer_id = (static_cast<uint64_t>(l) << 32) |
                                     static_cast<uint32_t>(sh);

          for (auto& bc : block_chunks) {
            if (!explicit_dst_ptrs.empty()) {
              bc.ptr = (default_base != nullptr && explicit_base != nullptr)
                           ? explicit_base + (bc.ptr - default_base)
                           : nullptr;
            }
            requests.push_back(BuildBlockRequest(
                lib::TransferOp::kBlockPull, bc.ptr, /*raddr=*/nullptr, bc.size,
                count_or_size, request_id, uuid, buffer_id, parallelism,
                major_order, remote_id, local_id, /*stream_idx=*/i,
                on_block_received));
          }
          ++request_id;
          return absl::OkStatus();
        }));
  }
}

  return requests;
}

void BlockTransport::ForgetPushProgress(uint64_t uuid) {
  raw_transport_.ForgetPushProgress(uuid);
  if (transport_adapter_ != nullptr) {
    transport_adapter_->ForgetUuid(uuid);
  }
  for (auto& [_, adapter] : numa_adapters_) {
    if (adapter != nullptr) {
      adapter->ForgetUuid(uuid);
    }
  }
  std::shared_ptr<BlockTransportServerPushSession> session;
  {
    absl::MutexLock lock(server_sessions_mu_);
    auto it = active_push_sessions_.find(uuid);
    if (it != active_push_sessions_.end()) {
      session = std::move(it->second);
      active_push_sessions_.erase(it);
    }
  }
  if (session) {
    session->Cancel();
  }
}

absl::Status BlockTransport::PushBuffer(absl::string_view peer,
                                        size_t buffer_id, size_t dst_shard_idx,
                                        size_t dst_offset_bytes,
                                        const uint8_t* data_ptr,
                                        size_t size_bytes, uint64_t uuid) {
  ABSL_ASSIGN_OR_RETURN(
      const lib::BufferRequest req,
      lib::BuildBufferRequest(buffer_id, dst_shard_idx, dst_offset_bytes,
                              data_ptr, size_bytes, uuid, lib::kOpBufferPush));
  return raw_transport_.ProcessSocketBufferPush(peer, req);
}

absl::Status BlockTransport::PullBuffer(absl::string_view peer,
                                        size_t buffer_id, size_t src_shard_idx,
                                        size_t src_offset_bytes,
                                        size_t dst_shard_idx,
                                        size_t dst_offset_bytes,
                                        size_t size_bytes) {
  // The ChunkHeader wire format carries the pull size in a 32-bit field, so
  // large ranges are issued as sequential slices of at most 1 GiB each.
  // TODO(justinlu): The source offset is also a 32-bit wire field; ranges whose
  // source offset reaches 4 GiB are rejected by RawBufferTransport.
  constexpr size_t kMaxPullSliceBytes = 1ULL << 30;
  size_t remaining = size_bytes;
  size_t curr_src_offset = src_offset_bytes;
  size_t curr_dst_offset = dst_offset_bytes;
  while (remaining > 0) {
    const size_t chunk_bytes = std::min(remaining, kMaxPullSliceBytes);
    ABSL_RETURN_IF_ERROR(raw_transport_.PullBuffer(
        peer, buffer_id, src_shard_idx, curr_src_offset, dst_shard_idx,
        curr_dst_offset, chunk_bytes));
    curr_src_offset += chunk_bytes;
    curr_dst_offset += chunk_bytes;
    remaining -= chunk_bytes;
  }
  return absl::OkStatus();
}

absl::Status BlockTransport::PushBuffers(
    const std::vector<BufferPushTask>& tasks, int parallelism, uint64_t uuid) {
  return raw_transport_.PushBuffers(tasks, parallelism, uuid);
}

}  // namespace transport
}  // namespace tpu_raiden

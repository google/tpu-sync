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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>  // NOLINT
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/peregrine_control_service.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"
#include "tpu_sync/transport/peregrine/src/api/socket_util.h"

ABSL_FLAG(size_t, raiden_transport_coalesce_window_bytes, 0,
          "Maximum size in bytes of the host-side coalescing buffer used "
          "for network transfers. Set to 0 to disable coalescing.");

namespace tpu_raiden {
namespace transport {

namespace {

using ::peregrine::ReadExact;
using ::peregrine::ReadVExact;
using ::peregrine::WriteExact;
using ::peregrine::WriteVExact;
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

constexpr MetricLabel kPullResponseLabels[] = {
    {.key = metric_labels::kDirection,
     .value = metric_labels::kDirectionPullResponse},
};

constexpr uint8_t kUseBlockChunksFlag = 0x80;

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 1024
#endif

#ifndef IOV_MAX
#define IOV_MAX UIO_MAXIOV
#endif

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
                            size_t num_shards, size_t num_blocks, Fn fn) {
  switch (static_cast<int>(major_order)) {
    case static_cast<int>(MajorOrder::kLayerMajor):
      for (int l : layer_ids) {
        for (size_t sh = 0; sh < num_shards; ++sh) {
          for (size_t k = 0; k < num_blocks; ++k) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
    case static_cast<int>(MajorOrder::kBlockMajor):
      for (size_t k = 0; k < num_blocks; ++k) {
        for (int l : layer_ids) {
          for (size_t sh = 0; sh < num_shards; ++sh) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Unknown block transport major order");
}

}  // namespace

BlockTransport::BlockTransport(BlockTransportDelegate* delegate, int local_port,
                               const std::vector<std::string>& local_ips,
                               int parallelism)
    : block_delegate_(delegate),
      parallelism_(parallelism),
      rr_index_(0),
      scheduler_stopping_(false),
      raw_transport_(
          delegate, local_port, local_ips,
          [this](int client_fd, const lib::ChunkHeader& header) {
            return HandleCustomRequest(client_fd, header);
          },
          absl::GetFlag(FLAGS_raiden_transport_coalesce_window_bytes)),
      peregrine_control_(
          std::make_unique<lib::PeregrineControlServiceImpl>(&raw_transport_)) {
  socket_workers_.reserve(parallelism_);
  for (int i = 0; i < parallelism_; ++i) {
    socket_workers_.push_back(
        std::thread(&BlockTransport::SocketWorkerLoop, this));
  }
}

BlockTransport::~BlockTransport() {
  {
    absl::MutexLock lock(scheduler_mu_);
    scheduler_stopping_ = true;
  }
  scheduler_cv_.SignalAll();
  for (auto& t : socket_workers_) {
    if (t.joinable()) t.join();
  }
  {
    absl::MutexLock lock(active_sends_mu_);
    for (const auto& [uuid, state] : active_sends_) {
      if (state && state->client_fd >= 0) {
        shutdown(state->client_fd, SHUT_RDWR);
      }
    }
    active_sends_mu_.Await(absl::Condition(
        +[](const SendMap* s) { return s->empty(); }, &active_sends_));
  }
}

void BlockTransport::SocketWorkerLoop() {
  while (!scheduler_stopping_) {
    std::unique_ptr<WriteTask> task;
    {
      absl::MutexLock lock(scheduler_mu_);
      while (true) {
        if (scheduler_stopping_) return;
        task = SelectNextTask();
        if (task) break;
        scheduler_cv_.Wait(&scheduler_mu_);
      }
    }

    if (task) {
      task->run();
      {
        absl::MutexLock lock(scheduler_mu_);
        auto it = peer_queues_.find(task->peer);
        if (it != peer_queues_.end()) {
          it->second.active_streams--;
        }
      }
      scheduler_cv_.SignalAll();
    }
  }
}

std::unique_ptr<BlockTransport::WriteTask> BlockTransport::SelectNextTask() {
  if (active_peers_.empty()) return nullptr;

  size_t start_idx = rr_index_;
  do {
    const std::string& peer = active_peers_[rr_index_];
    rr_index_ = (rr_index_ + 1) % active_peers_.size();

    auto& q = peer_queues_[peer];
    if (!q.tasks.empty() && q.active_streams < parallelism_) {
      q.active_streams++;
      auto task = std::move(q.tasks.front());
      q.tasks.pop_front();
      return task;
    }
  } while (rr_index_ != start_idx);

  return nullptr;
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

absl::Status BlockTransport::HandleIncomingPush(
    int client_fd, const lib::ChunkHeader& header) {
  ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
  std::vector<int> target_layers;
  if (header.local_id == 0xFFFFFFFF) {
    target_layers.resize(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    if (header.local_id >= block_delegate_->num_block_arrays()) {
      return absl::OutOfRangeError(
          absl::StrCat("push block-array index ", header.local_id,
                       " out of range: ", block_delegate_->num_block_arrays()));
    }
    target_layers = {static_cast<int>(header.local_id)};
  }

  // Resolve the expectation source before the explicit-destination handshake
  // or any payload write. A uuid whose registered receive plan carries pool
  // fields is plan-declared: it addresses exactly one declared pool and its
  // completion gate is the plan's global push count. nullopt keeps the
  // header-declared (legacy) contract.
  std::optional<PoolPushProgressSpec> pool_progress_spec;
  for (int target_layer : target_layers) {
    ASSIGN_OR_RETURN(
        std::optional<PoolPushProgressSpec> candidate,
        block_delegate_->GetPoolPushProgressSpec(target_layer, header.uuid));
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

  if (header.op == 6 && !pool_progress_spec.has_value() &&
      !block_delegate_->AcceptsPlanlessExplicitPush(header.uuid)) {
    return absl::FailedPreconditionError(
        absl::StrCat("explicit-destination push for uuid ", header.uuid,
                     " has no registered plan on this pool-mode worker"));
  }

  std::vector<int> allocated_ids;

  std::vector<int> src_block_ids;
  if (header.op == 1) {
    ASSIGN_OR_RETURN(allocated_ids, block_delegate_->AllocateBlocks(
                                        header.count_or_size, header.uuid));
    const std::vector<uint8_t> s_ids = lib::SerializeBlockIds(allocated_ids);
    RETURN_IF_ERROR(WriteExact(client_fd, s_ids.data(), s_ids.size()));
  } else {
    std::vector<uint8_t> ids_buf(header.count_or_size * sizeof(uint32_t));
    RETURN_IF_ERROR(ReadExact(client_fd, ids_buf.data(), ids_buf.size()));
    allocated_ids = lib::DeserializeBlockIds(ids_buf);

    RETURN_IF_ERROR(ReadExact(client_fd, ids_buf.data(), ids_buf.size()));
    src_block_ids = lib::DeserializeBlockIds(ids_buf);
    uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  }

  uint64_t total_received_bytes = 0;
  RETURN_IF_ERROR(ForEachPayload(
      major_order, target_layers, block_delegate_->num_shards(),
      header.count_or_size, [&](size_t l, size_t sh, size_t k) -> absl::Status {
        ABSL_DCHECK_LT(k, allocated_ids.size());
        const int dst_id = allocated_ids[k];
        uint8_t size_buf[lib::kChunkSizeFieldSize];
        RETURN_IF_ERROR(ReadExact(client_fd, size_buf, sizeof(size_buf)));
        const uint32_t sender_size = lib::DeserializeChunkSize(size_buf);

        const int64_t block_id_val = dst_id;
        int64_t src_bid = -1;
        if (!src_block_ids.empty()) {
          ABSL_DCHECK_LT(k, src_block_ids.size());
          src_bid = src_block_ids[k];
        }
        std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
            l, sh, absl::MakeConstSpan(&block_id_val, 1), sender_size,
            header.uuid, static_cast<int64_t>(header.remote_id),
            /*peer=*/"", src_bid);
        if (chunks.empty()) {
          return absl::NotFoundError(
              absl::StrCat("No transfer chunks found for block ", dst_id,
                           " and uuid ", header.uuid));
        }
        RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

        uint32_t expected_size = 0;
        for (const auto& chunk : chunks) {
          expected_size += chunk.size;
        }
        if (sender_size != expected_size) {
          // The connection is dropped after this return; without a log the
          // sender only ever sees an anonymous "Push verification failed".
          LOG(ERROR) << "Incoming push size mismatch: sender offered "
                     << sender_size << " bytes, receiver expected "
                     << expected_size << " bytes (chunks=" << chunks.size()
                     << ") uuid=" << header.uuid << " layer/pool=" << l
                     << " src_block=" << src_bid << " dst_block=" << dst_id;
          return absl::InternalError(absl::StrCat(
              "Block transfer size mismatch! Sender offered: ", sender_size,
              " bytes, but Receiver expected: ", expected_size,
              " bytes for Block ID: ", dst_id));
        }

        if (expected_size > 0) {
          RETURN_IF_ERROR(ReadVExact(client_fd, ToIovec(chunks)));
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

  // Unified receive accounting: one progress map and one increment path for
  // both contracts; only the expectation source and the completion callback
  // differ. Plan-declared mode counts every sender's streams against the
  // plan's global per-pool total and retires the uuid when all declared pools
  // complete; header-declared (legacy) mode counts this push's parallelism
  // from header.reserved and retires the uuid when all constructor layers
  // complete.
  const int l =
      (header.local_id == 0xFFFFFFFF) ? 0 : static_cast<int>(header.local_id);
  const bool plan_declared = pool_progress_spec.has_value();
  const size_t expected_chunks =
      plan_declared ? pool_progress_spec->expected_pushes : header.reserved;
  // A receive plan assembled from several senders completes a block array
  // only when every declared sender's streams for it have landed. The plan's
  // sender count is fixed for the life of the uuid, so it is resolved on the
  // array's first stream and kept with the array's progress.
  std::optional<size_t> expected_senders;
  bool resolve_senders = false;
  if (!plan_declared) {
    absl::MutexLock lock(progress_mu_);
    auto it = layer_progress_.find({header.uuid, l});
    if (it != layer_progress_.end() && it->second.expected_senders_resolved) {
      expected_senders = it->second.expected_senders;
    } else {
      resolve_senders = true;
    }
  }
  if (resolve_senders) {
    expected_senders = block_delegate_->ExpectedPushSenders(header.uuid);
  }
  bool trigger_completion = false;
  {
    absl::MutexLock lock(progress_mu_);
    auto& progress = layer_progress_[{header.uuid, l}];
    if (!plan_declared) {
      if (!progress.expected_senders_resolved) {
        progress.expected_senders = expected_senders;
        progress.expected_senders_resolved = true;
      }
      expected_senders = progress.expected_senders;
    }
    progress.completed_chunks++;
    if (plan_declared && progress.completed_chunks > expected_chunks) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool ", l, " received more than ", expected_chunks,
                       " push streams for UUID ", header.uuid));
    }
    bool array_complete = false;
    if (expected_senders.has_value()) {
      auto& streams = progress.sender_streams[header.remote_id];
      if (streams.landed == 0) {
        if (header.reserved == 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "block array ", l, " sender ", header.remote_id,
              " declared no streams for UUID ", header.uuid));
        }
        streams.declared = header.reserved;
        if (progress.sender_streams.size() > *expected_senders) {
          return absl::AlreadyExistsError(absl::StrCat(
              "block array ", l, " received pushes from more than ",
              *expected_senders, " senders for UUID ", header.uuid));
        }
      } else if (streams.declared != header.reserved) {
        return absl::FailedPreconditionError(absl::StrCat(
            "block array ", l, " sender ", header.remote_id, " declared ",
            header.reserved, " streams after declaring ", streams.declared,
            " for UUID ", header.uuid));
      }
      ++streams.landed;
      if (streams.landed == streams.declared) {
        ++progress.senders_complete;
      } else if (streams.landed > streams.declared) {
        return absl::AlreadyExistsError(absl::StrCat(
            "block array ", l, " received ", streams.landed,
            " streams from sender ", header.remote_id, " which declared ",
            streams.declared, " for UUID ", header.uuid));
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
        for (const auto& [key, pool_progress] : layer_progress_) {
          if (key.first == header.uuid &&
              pool_progress.on_layer_received_called) {
            ++completed_pools;
          }
        }
        if (completed_pools == pool_progress_spec->expected_pools) {
          for (auto it = layer_progress_.begin();
               it != layer_progress_.end();) {
            if (it->first.first == header.uuid) {
              auto erase_it = it++;
              layer_progress_.erase(erase_it);
            } else {
              ++it;
            }
          }
        }
      } else {
        bool all_layers_called = true;
        for (size_t layer = 0; layer < block_delegate_->num_layers(); ++layer) {
          auto it = layer_progress_.find({header.uuid, layer});
          if (it == layer_progress_.end() ||
              !it->second.on_layer_received_called) {
            all_layers_called = false;
            break;
          }
        }
        if (all_layers_called) {
          for (size_t layer = 0; layer < block_delegate_->num_layers();
               ++layer) {
            layer_progress_.erase({header.uuid, layer});
          }
        }
      }
    }
  }

  if (trigger_completion) {
    if (plan_declared) {
      RETURN_IF_ERROR(block_delegate_->OnPoolReceived(l, header.uuid));
    } else {
      RETURN_IF_ERROR(block_delegate_->OnLayerReceived(l, header.uuid));
    }
  }

  LOG(INFO) << "HandleCustomRequest (H2H read complete): client_fd="
            << client_fd << ", uuid=" << header.uuid
            << ", numa=" << block_delegate_->node_id();
  RETURN_IF_ERROR(
      block_delegate_->OnBlocksReceived(allocated_ids, header.uuid));
  uint8_t ack = 1;
  RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  return absl::OkStatus();
}

absl::Status BlockTransport::HandleIncomingPull(
    int client_fd, const lib::ChunkHeader& header) {
  ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
  if (block_delegate_->shard_factor() == 0) {
    return absl::InvalidArgumentError("shard_factor must be positive");
  }
  if (header.count_or_size % block_delegate_->shard_factor() != 0) {
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
  RETURN_IF_ERROR(WriteExact(client_fd, s.data(), s.size()));

  size_t local_blocks = header.count_or_size / block_delegate_->shard_factor();
  if (header.remote_id >
          static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) -
                         static_cast<size_t>(header.remote_id)) {
    return absl::OutOfRangeError("Requested block range exceeds int range");
  }

  auto state = std::make_shared<SendStreamState>();
  state->client_fd = client_fd;
  state->uuid = header.uuid;
  state->remote_id = static_cast<int>(header.remote_id);
  state->count_or_size = header.count_or_size;
  state->major_order = major_order;
  state->current_step = 0;
  state->total_steps = block_delegate_->num_block_arrays() *
                       block_delegate_->num_shards() * local_blocks;

  {
    absl::MutexLock lock(active_sends_mu_);
    active_sends_[header.uuid] = state;
  }

  TriggerNextSendStep(state);
  return absl::OkStatus();
}

void BlockTransport::TriggerNextSendStep(
    std::shared_ptr<SendStreamState> state) {
  if (state->current_step >= state->total_steps) {
    {
      absl::MutexLock lock(active_sends_mu_);
      active_sends_.erase(state->uuid);
    }
    return;
  }

  size_t l, sh, k;
  ResolveStepCoordinates(state, &l, &sh, &k);
  int block_id = state->remote_id + k;

  block_delegate_->RegisterBlockReadinessCallback(
      l, sh, block_id, state->uuid,
      [this, state, l, sh, block_id](absl::Status status) {
        if (!status.ok()) {
          LOG(ERROR) << "Pull response failed at step " << state->current_step
                     << " for uuid " << state->uuid
                     << ", status: " << status.ToString();
          shutdown(state->client_fd, SHUT_RDWR);
          absl::MutexLock lock(active_sends_mu_);
          active_sends_.erase(state->uuid);
          return;
        }

        block_delegate_->ScheduleAsyncTask([this, state, l, sh, block_id]() {
          const int64_t block_id_val = block_id;
          std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
              l, sh, absl::MakeConstSpan(&block_id_val, 1),
              block_delegate_->block_bytes(l), state->uuid);
          if (chunks.empty()) {
            LOG(ERROR) << "No transfer chunks found for block " << block_id
                       << " and uuid " << state->uuid;
            shutdown(state->client_fd, SHUT_RDWR);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }
          absl::Status s = ValidateChunks(block_delegate_, l, sh, chunks);
          if (!s.ok()) {
            LOG(ERROR) << "Chunks validation failed: " << s.ToString();
            shutdown(state->client_fd, SHUT_RDWR);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }

          uint32_t total_size = GetChunksTotalSize(chunks);
          const std::array<uint8_t, lib::kChunkSizeFieldSize> s_size =
              lib::SerializeChunkSize(total_size);
          s = WriteExact(state->client_fd, s_size.data(), s_size.size());
          if (!s.ok()) {
            LOG(ERROR) << "Write size failed: " << s.ToString();
            shutdown(state->client_fd, SHUT_RDWR);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }
          if (total_size > 0) {
            s = WriteVExact(state->client_fd, ToIovec(chunks));
          }
          if (!s.ok()) {
            LOG(ERROR) << "Write payload failed: " << s.ToString();
            shutdown(state->client_fd, SHUT_RDWR);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }

          if (total_size > 0) {
            // TODO: Add interface name (e.g. eth0, lo) using
            // GetSocketLocalNic(state->client_fd) as a label key.
            RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
                metric_names::kSentBytesTotal, kPullResponseLabels, total_size);
          }

          state->current_step++;
          TriggerNextSendStep(state);
        });
      });
}

void BlockTransport::ResolveStepCoordinates(
    const std::shared_ptr<SendStreamState>& state, size_t* layer, size_t* shard,
    size_t* block_idx) {
  size_t L = block_delegate_->num_block_arrays();
  size_t Sh = block_delegate_->num_shards();
  size_t K = state->count_or_size / block_delegate_->shard_factor();
  size_t s = state->current_step;

  if (state->major_order == MajorOrder::kLayerMajor) {
    *block_idx = s % K;
    *shard = (s / K) % Sh;
    *layer = s / (K * Sh);
  } else {
    *shard = s % Sh;
    *layer = (s / Sh) % L;
    *block_idx = s / (Sh * L);
  }
}

uint32_t BlockTransport::GetChunksTotalSize(
    const std::vector<BlockChunk>& chunks) {
  uint32_t total = 0;
  for (const auto& chunk : chunks) {
    total += chunk.size;
  }
  return total;
}

absl::StatusOr<std::vector<int>> BlockTransport::SyncPush(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx) {
  auto promise =
      std::make_shared<std::promise<absl::StatusOr<std::vector<int>>>>();
  auto future = promise->get_future();
  AsyncPush(peers, src_block_ids, dst_block_ids, parallelism, major_order, uuid,
            layer_idx, [promise](absl::StatusOr<std::vector<int>> res) {
              promise->set_value(std::move(res));
            });
  return future.get();
}

void BlockTransport::AsyncPush(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx,
    std::function<void(absl::StatusOr<std::vector<int>>)> raw_on_complete) {
  auto on_complete = [raw_on_complete](absl::StatusOr<std::vector<int>> res) {
    if (!res.ok()) {
      RecordTransferFailure(res.status(), metric_labels::kDirectionPush);
    }
    raw_on_complete(std::move(res));
  };
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    on_complete(absl::InvalidArgumentError("Block list cannot be empty"));
    return;
  }
  if (peers.empty()) {
    on_complete(absl::InvalidArgumentError("Peer list cannot be empty"));
    return;
  }

  int P = parallelism;
  if (P <= 0) {
    on_complete(absl::InvalidArgumentError("parallelism must be positive"));
    return;
  }
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  auto shared_src_block_ids = std::make_shared<std::vector<int>>(src_block_ids);
  auto shared_dst_block_ids = std::make_shared<std::vector<int>>(dst_block_ids);
  auto allocated_ids = std::make_shared<std::vector<int>>(num_blocks, 0);
  auto statuses =
      std::make_shared<std::vector<absl::Status>>(P, absl::OkStatus());
  auto remaining_workers = std::make_shared<std::atomic<int>>(P);

  const size_t base_blocks_per_stream = num_blocks / P;
  const size_t remainder = num_blocks % P;
  size_t block_offset = 0;

  for (int i = 0; i < P; ++i) {
    const size_t block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    const auto local_ips = raw_transport_.local_ips();
    const size_t n = local_ips.size();
    const std::string local_ip = n >= 1 ? local_ips[i % n] : "";
    const std::string remote_peer = peers[i % peers.size()];

    auto task_run = [this, i, remote_peer, local_ip, block_offset, block_count,
                     shared_src_block_ids, shared_dst_block_ids, allocated_ids,
                     statuses, remaining_workers, major_order, uuid, layer_idx,
                     P, on_complete]() {
      H2hWriteWorker(i, remote_peer, local_ip, block_offset, block_count,
                     *shared_src_block_ids, *shared_dst_block_ids,
                     *allocated_ids, *statuses, major_order, uuid, layer_idx,
                     P);

      if (remaining_workers->fetch_sub(1) == 1) {
        absl::Status final_status = absl::OkStatus();
        for (const auto& s : *statuses) {
          if (!s.ok()) {
            final_status = s;
            break;
          }
        }
        if (!final_status.ok()) {
          on_complete(final_status);
        } else {
          on_complete(*allocated_ids);
        }
      }
    };

    auto task = std::make_unique<WriteTask>();
    task->uuid = uuid;
    task->layer_idx = layer_idx;
    task->stream_idx = i;
    task->peer = remote_peer;
    task->run = std::move(task_run);

    {
      absl::MutexLock lock(scheduler_mu_);
      auto& pq = peer_queues_[task->peer];
      pq.tasks.push_back(std::move(task));
      if (std::find(active_peers_.begin(), active_peers_.end(), remote_peer) ==
          active_peers_.end()) {
        active_peers_.push_back(remote_peer);
      }
    }
    scheduler_cv_.SignalAll();
    block_offset += block_count;
  }
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
    ASSIGN_OR_RETURN(allocated_ids,
                     block_delegate_->AllocateBlocks(local_blocks));
  }

  int P = parallelism;
  if (P <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  const size_t base_blocks_per_stream = local_blocks / P;
  const size_t remainder = local_blocks % P;
  size_t local_block_offset = 0;
  size_t remote_block_offset = 0;

  for (int i = 0; i < P; ++i) {
    const size_t local_block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    const size_t remote_block_count =
        local_block_count * block_delegate_->shard_factor();

    const auto local_ips = raw_transport_.local_ips();
    const size_t n = local_ips.size();
    const std::string local_ip = n >= 1 ? local_ips[i % n] : "";
    const std::string remote_peer = peers[i % peers.size()];

    threads.push_back(std::thread(
        &BlockTransport::H2hReadWorker, this, i, remote_peer, local_ip,
        local_block_offset, local_block_count, remote_block_offset,
        remote_block_count, std::ref(src_block_ids), std::ref(allocated_ids),
        std::ref(explicit_dst_ptrs), std::ref(statuses), major_order,
        on_block_received, uuid));

    local_block_offset += local_block_count;
    remote_block_offset += remote_block_count;
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

lib::Request BlockTransport::BuildBlockRequest(
    uint8_t socket_opcode, uint8_t* laddr, size_t len, uint32_t count_or_size,
    int layer_idx, uint32_t request_id, uint64_t uuid, int parallelism,
    MajorOrder major_order, uint32_t remote_id, uint32_t local_id,
    int shard_idx) {
  return lib::Request{
      .socket_opcode = socket_opcode,
      .laddr = laddr,
      .raddr = nullptr,
      .len = len,
      .major_order = static_cast<uint8_t>(major_order),
      .layer_idx = layer_idx,
      .parallelism = parallelism,
      .remote_id = remote_id,
      .local_id = local_id,
      .count_or_size = count_or_size,
      .uuid = uuid,
      .request_id = request_id,
      .shard_idx = shard_idx,
  };
}

absl::StatusOr<std::vector<lib::Request>> BlockTransport::BuildBlockRequests(
    absl::string_view peer, size_t block_offset, size_t block_count,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, MajorOrder major_order,
    uint64_t uuid, int layer_idx, int parallelism) {
  const uint8_t socket_opcode =
      static_cast<uint8_t>(dst_block_ids.empty() ? 1 : 6);
  const uint32_t remote_id = static_cast<uint32_t>(block_delegate_->node_id());
  const uint32_t local_id =
      layer_idx == -1 ? 0xFFFF'FFFF : static_cast<uint32_t>(layer_idx);
  const uint32_t count_or_size = static_cast<uint32_t>(block_count);

  if (block_count == 0) {
    return std::vector<lib::Request>{BuildBlockRequest(
        socket_opcode, /*laddr=*/nullptr, /*len=*/0, /*count_or_size=*/0,
        layer_idx, /*request_id=*/0, uuid, parallelism, major_order, remote_id,
        local_id, /*shard_idx=*/0)};
  }

  std::vector<int> target_layers;
  if (layer_idx == -1) {
    target_layers.resize(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    target_layers = {layer_idx};
  }

  std::vector<lib::Request> requests;
  requests.reserve(target_layers.size() * block_delegate_->num_shards() *
                   block_count);
  uint32_t request_id = 0;
  absl::Status s = ForEachPayload(
      major_order, target_layers, block_delegate_->num_shards(), block_count,
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
        RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

        const int shard_idx = static_cast<int>(sh);
        for (const auto& chunk : chunks) {
          requests.push_back(BuildBlockRequest(
              socket_opcode, chunk.ptr, chunk.size, count_or_size, layer_idx,
              request_id, uuid, parallelism, major_order, remote_id, local_id,
              shard_idx));
        }
        ++request_id;
        return absl::OkStatus();
      });

  if (!s.ok()) {
    return s;
  }
  return requests;
}

absl::Status BlockTransport::ProcessSocketPush(
    absl::string_view peer, absl::string_view local_ip,
    absl::Span<const lib::Request> requests,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, size_t block_offset,
    std::vector<int>& allocated_ids) {
  if (requests.empty()) {
    return absl::OkStatus();
  }

  const auto& first = requests.front();
  const uint8_t socket_opcode = first.socket_opcode;
  const uint64_t uuid = first.uuid;
  const uint32_t remote_id = first.remote_id;
  const uint32_t local_id = first.local_id;
  const uint32_t count_or_size = first.count_or_size;
  const int parallelism = first.parallelism;
  const uint8_t major_order = first.major_order;
  const size_t block_count = static_cast<size_t>(count_or_size);

  auto borrowed_fd = raw_transport_.BorrowConnection(peer, local_ip);
  if (!borrowed_fd.ok()) {
    return borrowed_fd.status();
  }

  const int fd = borrowed_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup([&] {
    raw_transport_.ReturnConnection(ok_to_pool, fd, peer, local_ip);
  });

  lib::ChunkHeader header = {};
  header.version = 1;
  header.op = socket_opcode;
  header.flags = major_order;
  header.buffer_id = 0;
  header.reserved = static_cast<uint16_t>(parallelism);
  header.remote_id = remote_id;
  header.local_id = local_id;
  header.count_or_size = count_or_size;
  header.uuid = uuid;
  const auto s_header = lib::SerializeChunkHeader(header);
  absl::Status s = WriteExact(fd, s_header.data(), s_header.size());
  if (!s.ok()) {
    return s;
  }

  if (socket_opcode == 6) {
    ABSL_DCHECK_LE(block_offset + block_count, dst_block_ids.size());
    const auto s_dst_ids = lib::SerializeBlockIds(
        {dst_block_ids.data() + block_offset, block_count});
    RETURN_IF_ERROR(WriteExact(fd, s_dst_ids.data(), s_dst_ids.size()));
    const auto s_src_ids = lib::SerializeBlockIds(
        {src_block_ids.data() + block_offset, block_count});
    RETURN_IF_ERROR(WriteExact(fd, s_src_ids.data(), s_src_ids.size()));
    uint8_t ack = 0;
    s = ReadExact(fd, &ack, 1);
    if (!s.ok() || ack != 1) {
      return absl::InternalError("Explicit push destination handshake failed");
    }
    for (size_t k = 0; k < block_count; ++k) {
      allocated_ids[block_offset + k] = dst_block_ids[block_offset + k];
    }
  } else {
    std::vector<uint8_t> ids_buf(block_count * sizeof(uint32_t));
    RETURN_IF_ERROR(ReadExact(fd, ids_buf.data(), ids_buf.size()));
    const std::vector<int> stream_allocated_ids =
        lib::DeserializeBlockIds(ids_buf);

    for (size_t k = 0; k < block_count; ++k) {
      ABSL_DCHECK_LT(block_offset + k, allocated_ids.size());
      ABSL_DCHECK_LT(k, stream_allocated_ids.size());
      allocated_ids[block_offset + k] = stream_allocated_ids[k];
    }
  }
  uint64_t stream_bytes_sent = 0;
  if (block_count > 0) {
    for (size_t i = 0; i < requests.size();) {
      size_t j = i;
      uint32_t total_size = 0;
      std::vector<struct iovec> iov;
      while (j < requests.size() &&
             requests[j].request_id == requests[i].request_id) {
        total_size += static_cast<uint32_t>(requests[j].len);
        if (requests[j].len > 0) {
          iov.push_back(
              {.iov_base = requests[j].laddr, .iov_len = requests[j].len});
        }
        ++j;
      }

      const std::array<uint8_t, lib::kChunkSizeFieldSize> s_size =
          lib::SerializeChunkSize(total_size);
      RETURN_IF_ERROR(WriteExact(fd, s_size.data(), s_size.size()));
      if (total_size > 0) {
        RETURN_IF_ERROR(WriteVExact(fd, absl::MakeSpan(iov)));
        stream_bytes_sent += total_size;
      }
      i = j;
    }
  }

  if (stream_bytes_sent > 0) {
    // TODO: Add interface name (e.g.
    // eth0, lo) using GetSocketLocalNic(fd) as a label key.
    RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
        metric_names::kSentBytesTotal, kPushLabels, stream_bytes_sent);
  }

  uint8_t ack = 0;
  s = ReadExact(fd, &ack, 1);
  if (!s.ok() || ack != 1) {
    return absl::InternalError("Push verification failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

void BlockTransport::H2hWriteWorker(int stream_idx, absl::string_view peer,
                                    absl::string_view local_ip,
                                    size_t block_offset, size_t block_count,
                                    const std::vector<int>& src_block_ids,
                                    const std::vector<int>& dst_block_ids,
                                    std::vector<int>& allocated_ids,
                                    std::vector<absl::Status>& statuses,
                                    MajorOrder major_order, uint64_t uuid,
                                    int layer_idx, int parallelism) {
  if (block_count > std::numeric_limits<uint32_t>::max()) {
    statuses[stream_idx] = absl::OutOfRangeError("Block count exceeds uint32");
    return;
  }

  auto requests_or = BuildBlockRequests(
      peer, block_offset, block_count, src_block_ids, dst_block_ids,
      major_order, uuid, layer_idx, parallelism);
  if (!requests_or.ok()) {
    statuses[stream_idx] = requests_or.status();
    return;
  }

  absl::Status s =
      ProcessSocketPush(peer, local_ip, *requests_or, src_block_ids,
                        dst_block_ids, block_offset, allocated_ids);
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }
}

absl::StatusOr<std::vector<lib::Request>>
BlockTransport::BuildBlockPullRequests(
    size_t local_block_offset, size_t local_block_count,
    size_t remote_block_offset, size_t remote_block_count,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, MajorOrder major_order,
    uint64_t uuid) {
  if (remote_block_offset > src_block_ids.size() ||
      remote_block_count > src_block_ids.size() - remote_block_offset) {
    return absl::OutOfRangeError(
        "Remote block range exceeds source block list");
  }

  size_t SF = block_delegate_->shard_factor();

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

  std::vector<lib::Request> requests;
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
    RETURN_IF_ERROR(ForEachPayload(
        major_order, target_layers, block_delegate_->num_shards(),
        chunk.local_count, [&](size_t l, size_t sh, size_t k) -> absl::Status {
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
          RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, block_chunks));

          uint8_t* default_base = nullptr;
          uint8_t* explicit_base = nullptr;
          if (!explicit_dst_ptrs.empty()) {
            default_base = block_delegate_->GetBlockArrayHostPointer(l, sh);
            explicit_base =
                explicit_dst_ptrs[l * block_delegate_->num_shards() + sh];
          }

          const int layer_id = static_cast<int>(l);
          const int shard_idx = static_cast<int>(sh);
          const uint32_t local_id = static_cast<uint32_t>(dst_id);

          for (auto& bc : block_chunks) {
            if (!explicit_dst_ptrs.empty()) {
              bc.ptr = (default_base != nullptr && explicit_base != nullptr)
                           ? explicit_base + (bc.ptr - default_base)
                           : nullptr;
            }
            requests.push_back(BuildBlockRequest(
                /*socket_opcode=*/2, bc.ptr, bc.size, count_or_size, layer_id,
                request_id, uuid, /*parallelism=*/1, major_order, remote_id,
                local_id, shard_idx));
          }
          ++request_id;
          return absl::OkStatus();
        }));
  }

  return requests;
}

absl::Status BlockTransport::ProcessSocketPull(
    absl::string_view peer, absl::string_view local_ip,
    absl::Span<const lib::Request> requests,
    BlockReceivedCallback on_block_received) {
  if (requests.empty()) {
    return absl::OkStatus();
  }

  const auto& first = requests.front();
  const uint64_t uuid = first.uuid;
  const uint8_t major_order = first.major_order;
  const uint8_t socket_opcode = first.socket_opcode;

  auto borrowed_fd = raw_transport_.BorrowConnection(peer, local_ip);
  if (!borrowed_fd.ok()) {
    return borrowed_fd.status();
  }

  const int fd = borrowed_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup(
      [&] { raw_transport_.ReturnConnection(ok_to_pool, fd, peer, local_ip); });

  uint64_t stream_bytes_received = 0;

  for (size_t i = 0; i < requests.size();) {
    const auto& seg_first = requests[i];
    const uint32_t remote_read_block_id = seg_first.remote_id;
    const uint32_t remote_count = seg_first.count_or_size;

    lib::ChunkHeader header = {};
    header.version = 1;
    header.op = socket_opcode;
    header.flags = major_order;
    header.remote_id = remote_read_block_id;
    header.count_or_size = remote_count;
    header.uuid = uuid;
    const auto s_header = lib::SerializeChunkHeader(header);
    RETURN_IF_ERROR(WriteExact(fd, s_header.data(), s_header.size()));

    char resp_buf[lib::kChunkHeaderSize];
    RETURN_IF_ERROR(ReadExact(fd, resp_buf, sizeof(resp_buf)));
    ASSIGN_OR_RETURN(const lib::ChunkHeader resp_header,
                     lib::DeserializeChunkHeader(resp_buf));
    if (resp_header.op != socket_opcode ||
        resp_header.count_or_size != remote_count) {
      return absl::InternalError("Unexpected block pull response header");
    }
    if (resp_header.flags != major_order) {
      return absl::InternalError("Unexpected block pull response major order");
    }

    while (i < requests.size() &&
           requests[i].remote_id == remote_read_block_id &&
           requests[i].count_or_size == remote_count) {
      const uint32_t cur_req_id = requests[i].request_id;
      const size_t l = static_cast<size_t>(requests[i].layer_idx);
      const size_t sh = static_cast<size_t>(requests[i].shard_idx);
      const int dst_id = static_cast<int>(requests[i].local_id);

      size_t j = i;
      uint32_t expected_size = 0;
      std::vector<struct iovec> iov;
      while (j < requests.size() && requests[j].request_id == cur_req_id) {
        expected_size += static_cast<uint32_t>(requests[j].len);
        if (requests[j].len > 0) {
          if (requests[j].laddr == nullptr) {
            return absl::FailedPreconditionError(
                "Destination host pointer is null");
          }
          iov.push_back(
              {.iov_base = requests[j].laddr, .iov_len = requests[j].len});
        }
        ++j;
      }

      uint8_t size_buf[lib::kChunkSizeFieldSize];
      RETURN_IF_ERROR(ReadExact(fd, size_buf, sizeof(size_buf)));
      const uint32_t sender_size = lib::DeserializeChunkSize(size_buf);

      if (sender_size != expected_size) {
        return absl::InternalError(absl::StrCat(
            "Block transfer size mismatch! Sender offered: ", sender_size,
            " bytes, but Receiver expected: ", expected_size,
            " bytes for Block ID: ", dst_id));
      }

      if (expected_size > 0) {
        RETURN_IF_ERROR(ReadVExact(fd, absl::MakeSpan(iov)));
        stream_bytes_received += expected_size;
      }

      if (on_block_received != nullptr) {
        RETURN_IF_ERROR(on_block_received(l, sh, dst_id, expected_size));
      }
      i = j;
    }
  }

  if (stream_bytes_received > 0) {
    // TODO: Add interface name (e.g. eth0, lo) using
    // GetSocketLocalNic(fd) as a label key.
    RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
        metric_names::kReceivedBytesTotal, kPullResponseLabels,
        stream_bytes_received);
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

void BlockTransport::H2hReadWorker(
    int stream_idx, absl::string_view peer, absl::string_view local_ip,
    size_t local_block_offset, size_t local_block_count,
    size_t remote_block_offset, size_t remote_block_count,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs,
    std::vector<absl::Status>& statuses, MajorOrder major_order,
    BlockReceivedCallback on_block_received, uint64_t uuid) {
  absl::StatusOr<std::vector<lib::Request>> requests = BuildBlockPullRequests(
      local_block_offset, local_block_count, remote_block_offset,
      remote_block_count, src_block_ids, allocated_ids, explicit_dst_ptrs,
      major_order, uuid);
  if (!requests.ok()) {
    statuses[stream_idx] = requests.status();
    return;
  }

  absl::Status s =
      ProcessSocketPull(peer, local_ip, *requests, on_block_received);
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }
}

void BlockTransport::ForgetPushProgress(uint64_t uuid) {
  raw_transport_.ForgetPushProgress(uuid);
  absl::MutexLock lock(progress_mu_);
  for (auto it = layer_progress_.begin(); it != layer_progress_.end();) {
    if (it->first.first == uuid) {
      // absl::flat_hash_map::erase(iterator) returns void.
      auto erase_it = it++;
      layer_progress_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

absl::Status BlockTransport::PushBuffer(absl::string_view peer,
                                        size_t buffer_id, size_t dst_shard_idx,
                                        size_t dst_offset_bytes,
                                        const uint8_t* data_ptr,
                                        size_t size_bytes, uint64_t uuid) {
  ASSIGN_OR_RETURN(
      const lib::Request req,
      lib::BuildBufferRequest(buffer_id, dst_shard_idx, dst_offset_bytes,
                              data_ptr, size_bytes, uuid, lib::kOpBufferPush));
  absl::Status status = raw_transport_.ProcessSocketBufferPush(peer, req);
  if (!status.ok()) {
    RecordTransferFailure(status, metric_labels::kDirectionPush);
  }
  return status;
}

absl::Status BlockTransport::PushBuffers(
    const std::vector<BufferPushTask>& tasks, int parallelism, uint64_t uuid) {
  absl::Status status = raw_transport_.PushBuffers(tasks, parallelism, uuid);
  if (!status.ok()) {
    RecordTransferFailure(status, metric_labels::kDirectionPush);
  }
  return status;
}

}  // namespace transport
}  // namespace tpu_raiden

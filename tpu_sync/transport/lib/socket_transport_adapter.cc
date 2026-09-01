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

#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"
#include "tpu_sync/transport/peregrine/src/api/socket_util.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

namespace {

using ::peregrine::ReadExact;
using ::peregrine::WriteExact;
using ::peregrine::WriteVExact;
using ::tpu_raiden::telemetry::MetricLabel;
using ::tpu_raiden::telemetry::RaidenMetricStore;
namespace metric_labels = ::tpu_raiden::telemetry::metric_labels;
namespace metric_names = ::tpu_raiden::telemetry::metric_names;

constexpr MetricLabel kPushLabels[] = {
    {.key = metric_labels::kDirection, .value = metric_labels::kDirectionPush},
};

}  // namespace

SocketTransportAdapter::SocketTransportAdapter(
    RawBufferTransport* raw_transport, int parallelism)
    : raw_transport_(raw_transport),
      parallelism_(parallelism),
      rr_index_(0),
      scheduler_stopping_(false) {
  ABSL_DCHECK(raw_transport_ != nullptr);
  socket_workers_.reserve(parallelism_);
  for (int i = 0; i < parallelism_; ++i) {
    socket_workers_.push_back(
        std::thread(&SocketTransportAdapter::SocketWorkerLoop, this));
  }
}

SocketTransportAdapter::~SocketTransportAdapter() {
  {
    absl::MutexLock lock(scheduler_mu_);
    scheduler_stopping_ = true;
  }
  scheduler_cv_.SignalAll();
  for (auto& t : socket_workers_) {
    if (t.joinable()) t.join();
  }
}

void SocketTransportAdapter::SocketWorkerLoop() {
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

std::unique_ptr<SocketTransportAdapter::WriteTask>
SocketTransportAdapter::SelectNextTask() {
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

absl::StatusOr<Handle> SocketTransportAdapter::Post(
    absl::Span<const std::string> peers, absl::Span<const Request> requests,
    absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
    CompletionCallback on_complete) {
  if (requests.empty()) {
    if (on_complete) {
      on_complete(absl::InvalidArgumentError("Requests cannot be empty"));
    }
    return absl::InvalidArgumentError("Requests cannot be empty");
  }

  const uint8_t opcode = requests.front().socket_opcode;
  if (opcode == 1 || opcode == 6) {
    return PostSocketPush(peers, requests, src_block_ids, dst_block_ids,
                          std::move(on_complete));
  }

  const auto status = absl::UnimplementedError(
      absl::StrCat("Unsupported socket opcode: ", opcode));
  if (on_complete) {
    on_complete(status);
  }
  return status;
}

absl::StatusOr<Handle> SocketTransportAdapter::PostSocketPush(
    absl::Span<const std::string> peers, absl::Span<const Request> requests,
    absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
    CompletionCallback on_complete) {
  const size_t num_blocks = src_block_ids.size();
  const auto& req = requests.front();
  const uint64_t uuid = req.uuid;
  const int layer_idx = req.layer_idx;
  const int P = req.parallelism;
  if (P <= 0) {
    const auto status =
        absl::InvalidArgumentError("parallelism must be positive");
    if (on_complete) {
      on_complete(status);
    }
    return status;
  }

  auto shared_requests =
      std::make_shared<std::vector<Request>>(requests.begin(), requests.end());
  auto shared_src_block_ids = std::make_shared<std::vector<int>>(
      src_block_ids.begin(), src_block_ids.end());
  auto shared_dst_block_ids = std::make_shared<std::vector<int>>(
      dst_block_ids.begin(), dst_block_ids.end());
  auto allocated_ids = std::make_shared<std::vector<int>>(num_blocks, 0);
  auto statuses =
      std::make_shared<std::vector<absl::Status>>(P, absl::OkStatus());
  auto remaining_workers = std::make_shared<std::atomic<int>>(P);
  auto shared_on_complete =
      std::make_shared<CompletionCallback>(std::move(on_complete));

  const size_t base_blocks_per_stream = num_blocks / P;
  const size_t remainder = num_blocks % P;
  size_t req_offset = 0;
  for (int i = 0; i < P; ++i) {
    const size_t block_offset =
        i * base_blocks_per_stream + std::min<size_t>(i, remainder);

    size_t req_end = req_offset;
    while (req_end < shared_requests->size() &&
           (*shared_requests)[req_end].stream_idx == i) {
      ++req_end;
    }

    const auto local_ips = raw_transport_->local_ips();
    const size_t n = local_ips.size();
    const std::string local_ip = n >= 1 ? local_ips[i % n] : "";
    const std::string remote_peer = peers[i % peers.size()];

    absl::Span<const Request> stream_requests =
        absl::MakeConstSpan(*shared_requests)
            .subspan(req_offset, req_end - req_offset);
    req_offset = req_end;

    auto task_run = [this, i, remote_peer, local_ip, block_offset,
                     shared_requests, stream_requests, shared_src_block_ids,
                     shared_dst_block_ids, allocated_ids, statuses,
                     remaining_workers, shared_on_complete]() {
      (*statuses)[i] = PostSocketPushInternal(
          remote_peer, local_ip, stream_requests, *shared_src_block_ids,
          *shared_dst_block_ids, block_offset, *allocated_ids);

      if (remaining_workers->fetch_sub(1) == 1) {
        absl::Status final_status = absl::OkStatus();
        for (const auto& s : *statuses) {
          if (!s.ok()) {
            final_status = s;
            break;
          }
        }
        if (*shared_on_complete) {
          if (!final_status.ok()) {
            (*shared_on_complete)(final_status);
          } else {
            (*shared_on_complete)(*allocated_ids);
          }
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
  }
  return 0;
}

absl::StatusOr<Status> SocketTransportAdapter::Poll(Handle handle) {
  return absl::UnimplementedError(
      "SocketTransportAdapter::Poll not implemented yet");
}

absl::Status SocketTransportAdapter::PostSocketPushInternal(
    absl::string_view peer, absl::string_view local_ip,
    absl::Span<const Request> requests, absl::Span<const int> src_block_ids,
    absl::Span<const int> dst_block_ids, size_t block_offset,
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

  auto borrowed_fd = raw_transport_->BorrowConnection(peer, local_ip);
  if (!borrowed_fd.ok()) {
    return borrowed_fd.status();
  }

  const int fd = borrowed_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup([&] {
    raw_transport_->ReturnConnection(ok_to_pool, fd, peer, local_ip);
  });

  ChunkHeader header = {};
  header.version = 1;
  header.op = socket_opcode;
  header.flags = major_order;
  header.buffer_id = 0;
  header.reserved = static_cast<uint16_t>(parallelism);
  header.remote_id = remote_id;
  header.local_id = local_id;
  header.count_or_size = count_or_size;
  header.uuid = uuid;
  const auto s_header = SerializeChunkHeader(header);
  absl::Status s = WriteExact(fd, s_header.data(), s_header.size());
  if (!s.ok()) {
    return s;
  }

  if (socket_opcode == 6) {
    ABSL_DCHECK_LE(block_offset + block_count, dst_block_ids.size());
    const auto s_dst_ids =
        SerializeBlockIds({dst_block_ids.data() + block_offset, block_count});
    RETURN_IF_ERROR(WriteExact(fd, s_dst_ids.data(), s_dst_ids.size()));
    const auto s_src_ids =
        SerializeBlockIds({src_block_ids.data() + block_offset, block_count});
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
    const std::vector<int> stream_allocated_ids = DeserializeBlockIds(ids_buf);

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

      const std::array<uint8_t, kChunkSizeFieldSize> s_size =
          SerializeChunkSize(total_size);
      RETURN_IF_ERROR(WriteExact(fd, s_size.data(), s_size.size()));
      if (total_size > 0) {
        RETURN_IF_ERROR(WriteVExact(fd, absl::MakeSpan(iov)));
        stream_bytes_sent += total_size;
      }
      i = j;
    }
  }

  if (stream_bytes_sent > 0) {
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

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

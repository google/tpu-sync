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

#include "tpu_sync/weight_sync/weight_send_session.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"
#include "tpu_sync/weight_sync/shard_push_entry.h"

namespace tpu_raiden::weight_sync {
namespace {

struct WeightChunkBatchKey {
  uint64_t buffer_id = 0;
  uint32_t shard_id = 0;
  uint64_t chunk_batch_id = 0;

  bool operator==(const WeightChunkBatchKey& other) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const WeightChunkBatchKey& key) {
    return H::combine(std::move(h), key.buffer_id, key.shard_id,
                      key.chunk_batch_id);
  }
};

std::vector<std::string> ExtractEntryPeers(
    const ::tpu_sync::rpc::ShardPushEntryProto& entry) {
  if (!entry.dst_peers().empty()) {
    return std::vector<std::string>(entry.dst_peers().begin(),
                                    entry.dst_peers().end());
  }
  if (!entry.dst_peer().empty()) {
    return {entry.dst_peer()};
  }
  return {};
}

// Upper bound on total ChunkSpec + gather-payload entries expanded per session
// to prevent memory/CPU amplification DoS and protobuf int Reserve() overflow.
constexpr size_t kMaxChunkItemsPerSession = 1'000'000;
// Upper bound on OS worker threads spawned by PushSendSession::Start().
constexpr int kMaxPushSendConcurrency = 64;

// Appends ChunkSpecs representing |push_entry| reading from |src_buf|,
// enforcing |kMaxChunkItemsPerSession| across |total_items|.
absl::Status AppendChunksForEntry(const ShardPushEntry& push_entry,
                                  absl::Span<const uint8_t> src_buf,
                                  std::vector<ChunkSpec>& chunks_out,
                                  size_t& total_items) {
  const uint64_t num_outer = push_entry.num_outer();
  const uint64_t count = push_entry.count();
  const bool inner_contiguous = push_entry.is_inner_contiguous();
  const bool is_gather = !inner_contiguous &&
                         push_entry.effective_dst_stride() == push_entry.size();
  const bool is_scatter =
      !inner_contiguous && !is_gather &&
      push_entry.effective_src_stride() == push_entry.size();

  uint64_t items_needed = num_outer;
  if (is_gather || (!inner_contiguous && !is_scatter)) {
    if (count > 0 && num_outer > std::numeric_limits<uint64_t>::max() / count) {
      return absl::InvalidArgumentError(
          "ShardPushEntry chunk expansion overflows uint64_t");
    }
    const uint64_t inner_items = num_outer * count;
    if (is_gather) {
      if (inner_items > std::numeric_limits<uint64_t>::max() - num_outer) {
        return absl::InvalidArgumentError(
            "ShardPushEntry gather expansion overflows uint64_t");
      }
      items_needed = num_outer + inner_items;
    } else {
      items_needed = inner_items;
    }
  }
  if (items_needed > kMaxChunkItemsPerSession - total_items) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShardPushEntry chunk expansion exceeds limit ",
                     kMaxChunkItemsPerSession));
  }
  total_items += static_cast<size_t>(items_needed);

  const absl::Span<const uint64_t> outer_counts = push_entry.outer_counts();
  const absl::Span<const uint64_t> outer_src_strides =
      push_entry.outer_src_strides();
  const absl::Span<const uint64_t> outer_dst_strides =
      push_entry.outer_dst_strides();

  for (uint64_t outer_idx = 0; outer_idx < num_outer; ++outer_idx) {
    uint64_t curr_src_offset = push_entry.src_offset();
    uint64_t curr_dst_offset = push_entry.dst_offset();
    uint64_t temp = outer_idx;
    for (int d = static_cast<int>(outer_counts.size()) - 1; d >= 0; --d) {
      const uint64_t coord = temp % outer_counts[d];
      temp /= outer_counts[d];
      curr_src_offset += coord * outer_src_strides[d];
      curr_dst_offset += coord * outer_dst_strides[d];
    }

    if (inner_contiguous) {
      chunks_out.push_back(ScatterChunkSpec{
          .payload =
              src_buf.subspan(static_cast<size_t>(curr_src_offset),
                              static_cast<size_t>(push_entry.inner_src_span())),
          .byte_offset = curr_dst_offset,
      });
    } else if (is_gather) {
      GatherChunkSpec gather;
      gather.byte_offset = curr_dst_offset;
      gather.payloads.reserve(static_cast<size_t>(count));
      for (uint64_t k = 0; k < count; ++k) {
        gather.payloads.push_back(src_buf.subspan(
            static_cast<size_t>(curr_src_offset +
                                k * push_entry.effective_src_stride()),
            static_cast<size_t>(push_entry.size())));
      }
      chunks_out.push_back(std::move(gather));
    } else if (is_scatter) {
      chunks_out.push_back(ScatterChunkSpec{
          .payload =
              src_buf.subspan(static_cast<size_t>(curr_src_offset),
                              static_cast<size_t>(count * push_entry.size())),
          .byte_offset = curr_dst_offset,
          .byte_size = push_entry.size(),
          .byte_stride = push_entry.effective_dst_stride(),
      });
    } else {
      for (uint64_t k = 0; k < count; ++k) {
        chunks_out.push_back(ScatterChunkSpec{
            .payload = src_buf.subspan(
                static_cast<size_t>(curr_src_offset +
                                    k * push_entry.effective_src_stride()),
                static_cast<size_t>(push_entry.size())),
            .byte_offset =
                curr_dst_offset + k * push_entry.effective_dst_stride(),
        });
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int32_t>> SortedSrcShards(
    const ::tpu_sync::rpc::StartTransferRequest& spec) {
  std::vector<int32_t> src_shards;
  src_shards.reserve(spec.shard_push_schedules_size());
  for (const auto& [src_shard_idx, unused] : spec.shard_push_schedules()) {
    if (src_shard_idx < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Negative src_shard_idx ", src_shard_idx));
    }
    src_shards.push_back(src_shard_idx);
  }
  std::sort(src_shards.begin(), src_shards.end());
  return src_shards;
}

// Active weight send session that pushes all model weight chunk batches
// described by StartTransferRequest::shard_push_schedules via
// GrpcTransportClient::PushChunks().
class PushSendSession final : public WeightSendSession {
 public:
  struct PushTask {
    std::string peer;
    PushChunksRequest request;
  };

  PushSendSession(GrpcTransportClient* client, uint64_t uuid,
                  int max_concurrency, std::vector<PushTask> tasks,
                  WeightSendSessionOptions options)
      : WeightSendSession(uuid, options.deadline,
                          std::move(options.host_buffer_owner)),
        client_(client),
        tasks_(std::move(tasks)),
        max_concurrency_(max_concurrency),
        remaining_tasks_(tasks_.size()) {}

  ~PushSendSession() override {
    Finish(absl::CancelledError("PushSendSession destroyed"));
    std::vector<std::thread> workers;
    {
      absl::MutexLock lock(workers_mu_);
      workers = std::move(workers_);
    }
    for (std::thread& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  absl::Status Start() override {
    absl::Status status = MarkStarted();
    if (!status.ok()) {
      return status;
    }
    if (tasks_.empty()) {
      Finish(absl::OkStatus());
      return absl::OkStatus();
    }

    const size_t num_workers =
        std::min<size_t>(static_cast<size_t>(max_concurrency_), tasks_.size());
    absl::MutexLock lock(workers_mu_);
    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
      if (!TryBeginInternalOp()) {
        break;
      }
      absl::Cleanup end_op = [this]() { EndInternalOp(); };
      workers_.emplace_back(&PushSendSession::WorkerLoop, this);
      std::move(end_op).Cancel();
    }
    return absl::OkStatus();
  }

 private:
  void WorkerLoop() {
    absl::Cleanup end_op = [this]() { EndInternalOp(); };
    while (!IsDraining()) {
      size_t task_idx = 0;
      {
        absl::MutexLock lock(task_mu_);
        if (next_task_idx_ >= tasks_.size()) {
          return;
        }
        task_idx = next_task_idx_++;
      }

      const PushTask& task = tasks_[task_idx];
      absl::Duration timeout = absl::ZeroDuration();
      if (deadline_ != std::chrono::steady_clock::time_point::max()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline_) {
          Finish(absl::DeadlineExceededError(
              "PushSendSession deadline exceeded before PushChunks"));
          return;
        }
        timeout = absl::FromChrono(deadline_ - now);
      }

      absl::Status push_status =
          client_->PushChunks(task.peer, task.request, timeout);
      if (!push_status.ok()) {
        Finish(push_status);
        return;
      }

      bool all_done = false;
      {
        absl::MutexLock lock(task_mu_);
        if (remaining_tasks_ > 0) {
          --remaining_tasks_;
        }
        all_done = (remaining_tasks_ == 0);
      }
      if (all_done) {
        Finish(absl::OkStatus());
        return;
      }
    }
  }

  GrpcTransportClient* const client_;
  std::vector<PushTask> tasks_;
  const int max_concurrency_;

  absl::Mutex task_mu_;
  size_t next_task_idx_ ABSL_GUARDED_BY(task_mu_) = 0;
  size_t remaining_tasks_ ABSL_GUARDED_BY(task_mu_) = 0;

  absl::Mutex workers_mu_;
  std::vector<std::thread> workers_ ABSL_GUARDED_BY(workers_mu_);
};

// Passive weight send session that serves incoming PullChunks RPCs for all
// model weight chunk batches described by
// StartTransferRequest::shard_push_schedules.
class PullSendSession final
    : public WeightSendSession,
      public std::enable_shared_from_this<PullSendSession> {
 public:
  struct ServeEntry {
    PullChunksResponse response;
    size_t expected_pulls = 1;
  };

  PullSendSession(uint64_t uuid,
                  absl::flat_hash_map<WeightChunkBatchKey, ServeEntry> entries,
                  size_t total_expected_pulls, WeightSendSessionOptions options)
      : WeightSendSession(uuid, options.deadline,
                          std::move(options.host_buffer_owner)),
        entries_(std::move(entries)),
        total_remaining_pulls_(total_expected_pulls) {}

  ~PullSendSession() override = default;

  absl::Status Start() override {
    absl::Status status = MarkStarted(/*allow_already_succeeded=*/true);
    if (!status.ok()) {
      return status;
    }
    bool empty = false;
    {
      absl::MutexLock lock(entries_mu_);
      empty = (total_remaining_pulls_ == 0);
    }
    if (empty) {
      Finish(absl::OkStatus());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<PullChunksResponse> HandlePullChunks(
      const PullChunksRequest& request) {
    if (request.uuid != uuid_) {
      return absl::InvalidArgumentError(
          absl::StrCat("PullSendSession uuid mismatch: expected ", uuid_,
                       ", got ", request.uuid));
    }
    if (!TryBeginInternalOp()) {
      return absl::FailedPreconditionError(
          "PullSendSession is draining or done");
    }
    absl::Cleanup end_op = [this]() { EndInternalOp(); };

    if (deadline_ != std::chrono::steady_clock::time_point::max() &&
        std::chrono::steady_clock::now() >= deadline_) {
      absl::Status err = absl::DeadlineExceededError(
          "PullSendSession deadline exceeded on PullChunks");
      Finish(err);
      return err;
    }

    WeightChunkBatchKey key{request.buffer_id, request.shard_id,
                            request.chunk_batch_id};
    PullChunksResponse response;
    bool all_done = false;
    {
      absl::MutexLock lock(entries_mu_);
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        return absl::NotFoundError(absl::StrCat(
            "No pull entry for buffer_id=", key.buffer_id, ", shard_id=",
            key.shard_id, ", chunk_batch_id=", key.chunk_batch_id));
      }
      if (it->second.expected_pulls == 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Expected pulls already exhausted for buffer_id=", key.buffer_id,
            ", shard_id=", key.shard_id,
            ", chunk_batch_id=", key.chunk_batch_id));
      }
      response = it->second.response;
      --it->second.expected_pulls;
      if (total_remaining_pulls_ > 0) {
        --total_remaining_pulls_;
      }
      all_done = (total_remaining_pulls_ == 0);
    }

    // Transfer the in-flight op lease to response.owner so Done() waits until
    // gRPC finishes writing the response on the wire.
    std::shared_ptr<PullSendSession> self = shared_from_this();
    std::shared_ptr<const void> inner_owner = std::move(response.owner);
    std::move(end_op).Cancel();
    response.owner = std::shared_ptr<const void>(
        nullptr, [self = std::move(self),
                  inner_owner = std::move(inner_owner)](const void*) mutable {
          inner_owner.reset();
          self->EndInternalOp();
        });

    if (all_done) {
      Finish(absl::OkStatus());
    }
    return response;
  }

 private:
  absl::Mutex entries_mu_;
  absl::flat_hash_map<WeightChunkBatchKey, ServeEntry> entries_
      ABSL_GUARDED_BY(entries_mu_);
  size_t total_remaining_pulls_ ABSL_GUARDED_BY(entries_mu_) = 0;
};

}  // namespace

WeightSendSession::WeightSendSession(
    uint64_t uuid, std::chrono::steady_clock::time_point deadline,
    std::shared_ptr<const void> host_buffer_owner)
    : uuid_(uuid),
      deadline_(deadline),
      host_buffer_owner_(std::move(host_buffer_owner)) {}

bool WeightSendSession::Done() const {
  absl::MutexLock lock(mu_);
  return done_;
}

void WeightSendSession::Finish(const absl::Status& status) {
  absl::MutexLock lock(mu_);
  if (!status.ok() && status_.ok()) {
    status_ = status;
  }
  draining_ = true;
  if (in_flight_ == 0) {
    done_ = true;
  }
}

void WeightSendSession::EndOp() {}

bool WeightSendSession::TryBeginInternalOp() {
  absl::MutexLock lock(mu_);
  if (draining_ || done_) {
    return false;
  }
  ++in_flight_;
  return true;
}

void WeightSendSession::EndInternalOp() {
  absl::MutexLock lock(mu_);
  if (in_flight_ > 0) {
    --in_flight_;
  }
  if (draining_ && in_flight_ == 0) {
    done_ = true;
  }
}

absl::Status WeightSendSession::GetStatus() const {
  absl::MutexLock lock(mu_);
  return status_;
}

absl::Status WeightSendSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

bool WeightSendSession::IsDraining() const {
  absl::MutexLock lock(mu_);
  return draining_;
}

absl::StatusOr<size_t> WeightSendSession::ResolveStagingSlot(size_t unit) {
  return absl::NotFoundError(
      "WeightSendSession addresses host buffers in place");
}

absl::Status WeightSendSession::OnDataReceived(
    size_t major_dimension_idx, absl::Span<const int> minor_dimension_indices,
    absl::Span<const int> shard_ids) {
  return absl::FailedPreconditionError(
      "WeightSendSession does not receive data");
}

tsl::Future<> WeightSendSession::GetDataReadyFuture(
    size_t major_dimension_idx) {
  return tsl::Future<>(absl::OkStatus());
}

absl::optional<std::shared_ptr<const ::tpu_raiden::TransferPlan>>
WeightSendSession::plan() const {
  return absl::nullopt;
}

std::chrono::steady_clock::time_point WeightSendSession::deadline() const {
  return deadline_;
}

absl::Status WeightSendSession::MarkStarted(bool allow_already_succeeded) {
  absl::MutexLock lock(mu_);
  if (started_) {
    return absl::FailedPreconditionError(
        "WeightSendSession::Start() already called");
  }
  if (draining_ || done_) {
    if (allow_already_succeeded && status_.ok()) {
      started_ = true;
      return absl::OkStatus();
    }
    return !status_.ok() ? status_
                         : absl::FailedPreconditionError(
                               "WeightSendSession is already draining or done");
  }
  started_ = true;
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
WeightSendSession::CreatePushSendSession(
    GrpcTransportClient* client,
    const ::tpu_sync::rpc::StartTransferRequest& spec,
    ConstHostBufferProvider host_buffer_provider,
    WeightSendSessionOptions options) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "PushSendSession requires a non-null GrpcTransportClient");
  }
  if (!host_buffer_provider) {
    return absl::InvalidArgumentError(
        "PushSendSession requires a non-empty ConstHostBufferProvider");
  }
  if (spec.uuid() < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "StartTransferRequest uuid must be non-negative, got ", spec.uuid()));
  }
  if (spec.parallelism() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("StartTransferRequest parallelism must be non-negative, "
                     "got ",
                     spec.parallelism()));
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> src_shards, SortedSrcShards(spec));
  const uint64_t uuid = static_cast<uint64_t>(spec.uuid());

  using BatchKey = std::tuple<std::string, uint64_t, uint32_t, uint64_t>;
  absl::flat_hash_map<BatchKey, size_t> task_index_by_key;
  std::vector<PushSendSession::PushTask> tasks;
  size_t total_chunk_items = 0;

  for (int32_t src_shard_idx : src_shards) {
    const ::tpu_sync::rpc::ShardPushScheduleProto& schedule =
        spec.shard_push_schedules().at(src_shard_idx);
    for (int i = 0; i < schedule.entries_size(); ++i) {
      const ::tpu_sync::rpc::ShardPushEntryProto& entry = schedule.entries(i);
      const int32_t layer_idx = entry.has_layer_idx() ? entry.layer_idx() : 0;
      if (layer_idx < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Negative layer_idx ", layer_idx));
      }
      const uint64_t buffer_id = static_cast<uint64_t>(layer_idx);
      const uint32_t src_shard = static_cast<uint32_t>(src_shard_idx);

      ABSL_ASSIGN_OR_RETURN(absl::Span<const uint8_t> src_buf,
                            host_buffer_provider(buffer_id, src_shard));
      const std::string entry_ctx =
          absl::StrCat("PushSendSession (uuid=", uuid, ") shard ",
                       src_shard_idx, " entry[", i, "]");
      ABSL_ASSIGN_OR_RETURN(
          ShardPushEntry push_entry,
          ShardPushEntry::FromProto(entry, entry_ctx,
                                    {.host_size = src_buf.size()}));

      std::vector<std::string> peers = ExtractEntryPeers(entry);
      if (peers.empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat(entry_ctx, " has no destination peers"));
      }
      const uint32_t dst_shard =
          static_cast<uint32_t>(push_entry.dst_shard_idx());
      const uint64_t chunk_batch_id = static_cast<uint64_t>(src_shard_idx);

      absl::flat_hash_set<std::string> seen_peers;
      for (const std::string& peer : peers) {
        if (peer.empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat(entry_ctx, " has empty peer"));
        }
        if (!seen_peers.insert(peer).second) {
          continue;
        }
        BatchKey key{peer, buffer_id, dst_shard, chunk_batch_id};
        auto [it, inserted] = task_index_by_key.emplace(key, tasks.size());
        if (inserted) {
          PushSendSession::PushTask task;
          task.peer = peer;
          task.request.uuid = uuid;
          task.request.buffer_id = buffer_id;
          task.request.shard_id = dst_shard;
          task.request.chunk_batch_id = chunk_batch_id;
          tasks.push_back(std::move(task));
        }
        ABSL_RETURN_IF_ERROR(AppendChunksForEntry(
            push_entry, src_buf, tasks[it->second].request.chunks,
            total_chunk_items));
      }
    }
  }

  const int max_concurrency =
      spec.parallelism() > 0
          ? std::min(spec.parallelism(), kMaxPushSendConcurrency)
          : 1;
  return std::make_shared<PushSendSession>(
      client, uuid, max_concurrency, std::move(tasks), std::move(options));
}

absl::StatusOr<PullSendSessionHandle> WeightSendSession::CreatePullSendSession(
    const ::tpu_sync::rpc::StartTransferRequest& spec,
    ConstHostBufferProvider host_buffer_provider,
    WeightSendSessionOptions options) {
  if (!host_buffer_provider) {
    return absl::InvalidArgumentError(
        "PullSendSession requires a non-empty ConstHostBufferProvider");
  }
  if (spec.uuid() < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "StartTransferRequest uuid must be non-negative, got ", spec.uuid()));
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> src_shards, SortedSrcShards(spec));
  const uint64_t uuid = static_cast<uint64_t>(spec.uuid());

  absl::flat_hash_map<WeightChunkBatchKey, PullSendSession::ServeEntry> entries;
  absl::flat_hash_map<WeightChunkBatchKey, absl::flat_hash_set<std::string>>
      peers_by_key;
  size_t total_chunk_items = 0;

  for (int32_t src_shard_idx : src_shards) {
    const ::tpu_sync::rpc::ShardPushScheduleProto& schedule =
        spec.shard_push_schedules().at(src_shard_idx);
    for (int i = 0; i < schedule.entries_size(); ++i) {
      const ::tpu_sync::rpc::ShardPushEntryProto& entry = schedule.entries(i);
      const int32_t layer_idx = entry.has_layer_idx() ? entry.layer_idx() : 0;
      if (layer_idx < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Negative layer_idx ", layer_idx));
      }
      const uint64_t buffer_id = static_cast<uint64_t>(layer_idx);
      const uint32_t src_shard = static_cast<uint32_t>(src_shard_idx);

      ABSL_ASSIGN_OR_RETURN(absl::Span<const uint8_t> src_buf,
                            host_buffer_provider(buffer_id, src_shard));
      const std::string entry_ctx =
          absl::StrCat("PullSendSession (uuid=", uuid, ") shard ",
                       src_shard_idx, " entry[", i, "]");
      ABSL_ASSIGN_OR_RETURN(
          ShardPushEntry push_entry,
          ShardPushEntry::FromProto(entry, entry_ctx,
                                    {.host_size = src_buf.size()}));

      WeightChunkBatchKey key{
          .buffer_id = buffer_id,
          .shard_id = static_cast<uint32_t>(push_entry.dst_shard_idx()),
          .chunk_batch_id = static_cast<uint64_t>(src_shard_idx),
      };
      PullSendSession::ServeEntry& serve_entry = entries[key];
      serve_entry.response.owner = options.host_buffer_owner;
      ABSL_RETURN_IF_ERROR(AppendChunksForEntry(
          push_entry, src_buf, serve_entry.response.chunks, total_chunk_items));

      for (const std::string& peer : ExtractEntryPeers(entry)) {
        if (!peer.empty()) {
          peers_by_key[key].insert(peer);
        }
      }
    }
  }

  size_t total_expected = 0;
  for (auto& [key, serve_entry] : entries) {
    auto peer_it = peers_by_key.find(key);
    serve_entry.expected_pulls =
        (peer_it != peers_by_key.end() && !peer_it->second.empty())
            ? peer_it->second.size()
            : 1;
    total_expected += serve_entry.expected_pulls;
  }

  auto session = std::make_shared<PullSendSession>(
      uuid, std::move(entries), total_expected, std::move(options));
  std::weak_ptr<PullSendSession> weak_session = session;
  PullChunksHandler on_pull =
      [weak_session = std::move(weak_session)](const PullChunksRequest& request)
      -> absl::StatusOr<PullChunksResponse> {
    std::shared_ptr<PullSendSession> live = weak_session.lock();
    if (live == nullptr) {
      return absl::FailedPreconditionError(
          "PullSendSession has been destroyed");
    }
    return live->HandlePullChunks(request);
  };
  return PullSendSessionHandle{.session = std::move(session),
                               .on_pull = std::move(on_pull)};
}

}  // namespace tpu_raiden::weight_sync

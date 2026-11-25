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

#include "tpu_sync/weight_sync/weight_receive_session.h"

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

namespace tpu_raiden::weight_sync {
namespace {

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

absl::Status RecordLayerBatchLocked(
    uint64_t layer_idx,
    absl::flat_hash_map<uint64_t, size_t>& remaining_per_layer,
    size_t& total_remaining, bool& layer_completed, bool& all_done) {
  auto it = remaining_per_layer.find(layer_idx);
  if (it == remaining_per_layer.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected data for layer_idx=", layer_idx));
  }
  if (it->second == 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("Excess data for layer_idx=", layer_idx));
  }
  --it->second;
  layer_completed = (it->second == 0);
  if (total_remaining > 0) {
    --total_remaining;
  }
  all_done = (total_remaining == 0);
  return absl::OkStatus();
}

// Active weight receive session that pulls all model weight chunk batches
// described by StartTransferRequest::shard_push_schedules via
// GrpcTransportClient::PullChunks().
class PullReceiveSession final : public WeightReceiveSession {
 public:
  struct PullTask {
    std::string peer;
    uint64_t buffer_id = 0;
    uint32_t shard_id = 0;
    uint64_t chunk_batch_id = 0;
    absl::Span<uint8_t> dst_buffer;
  };

  PullReceiveSession(GrpcTransportClient* client, uint64_t uuid,
                     int max_concurrency, std::vector<PullTask> tasks,
                     absl::flat_hash_map<uint64_t, size_t> expected_per_layer,
                     WeightReceiveSessionOptions options)
      : WeightReceiveSession(uuid, std::move(expected_per_layer), tasks.size(),
                             options.deadline,
                             std::move(options.on_layer_received)),
        client_(client),
        tasks_(std::move(tasks)),
        max_concurrency_(max_concurrency) {}

  ~PullReceiveSession() override {
    Finish(absl::CancelledError("PullReceiveSession destroyed"));
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
      workers_.emplace_back(&PullReceiveSession::WorkerLoop, this);
    }
    return absl::OkStatus();
  }

  bool TryBeginOp() override { return false; }

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

      const PullTask& task = tasks_[task_idx];
      absl::Duration timeout = absl::ZeroDuration();
      if (deadline_ != std::chrono::steady_clock::time_point::max()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline_) {
          Finish(absl::DeadlineExceededError(
              "PullReceiveSession deadline exceeded before PullChunks"));
          return;
        }
        timeout = absl::FromChrono(deadline_ - now);
      }

      PullChunksRequest request;
      request.uuid = uuid_;
      request.buffer_id = task.buffer_id;
      request.shard_id = task.shard_id;
      request.chunk_batch_id = task.chunk_batch_id;

      absl::Status pull_status =
          client_->PullChunks(task.peer, request, task.dst_buffer, timeout);
      if (!pull_status.ok()) {
        Finish(pull_status);
        return;
      }
      if (IsDraining()) {
        return;
      }

      const int shard_id = static_cast<int>(task.shard_id);
      absl::Status data_status =
          OnDataReceived(static_cast<size_t>(task.buffer_id), {},
                         absl::MakeConstSpan(&shard_id, 1));
      if (!data_status.ok()) {
        Finish(data_status);
        return;
      }
    }
  }

  GrpcTransportClient* const client_;
  std::vector<PullTask> tasks_;
  const int max_concurrency_;

  absl::Mutex task_mu_;
  size_t next_task_idx_ ABSL_GUARDED_BY(task_mu_) = 0;

  absl::Mutex workers_mu_;
  std::vector<std::thread> workers_ ABSL_GUARDED_BY(workers_mu_);
};

// Passive weight receive session that waits for incoming PushChunks RPCs
// matching StartTransferRequest::expected_layer_chunk_counts routed from
// GrpcTransportServer via OnDataReceived().
class PushReceiveSession final : public WeightReceiveSession {
 public:
  PushReceiveSession(uint64_t uuid,
                     absl::flat_hash_map<uint64_t, size_t> expected_per_layer,
                     size_t total_expected_batches,
                     WeightReceiveSessionOptions options)
      : WeightReceiveSession(uuid, std::move(expected_per_layer),
                             total_expected_batches, options.deadline,
                             std::move(options.on_layer_received)) {}

  ~PushReceiveSession() override = default;

  absl::Status Start() override {
    absl::Status status = MarkStarted();
    if (!status.ok()) {
      return status;
    }
    if (HasNoRemainingBatches()) {
      Finish(absl::OkStatus());
    }
    return absl::OkStatus();
  }
};

}  // namespace

WeightReceiveSession::WeightReceiveSession(
    uint64_t uuid, absl::flat_hash_map<uint64_t, size_t> expected_per_layer,
    size_t total_expected_batches,
    std::chrono::steady_clock::time_point deadline,
    OnLayerReceivedCallback on_layer_received)
    : uuid_(uuid),
      deadline_(deadline),
      on_layer_received_(std::move(on_layer_received)),
      remaining_per_layer_(std::move(expected_per_layer)),
      total_remaining_(total_expected_batches) {}

bool WeightReceiveSession::Done() const {
  absl::MutexLock lock(mu_);
  return done_;
}

void WeightReceiveSession::Finish(const absl::Status& status) {
  absl::MutexLock lock(mu_);
  if (!status.ok() && status_.ok()) {
    status_ = status;
  }
  draining_ = true;
  if (in_flight_ == 0) {
    done_ = true;
  }
}

bool WeightReceiveSession::TryBeginOp() { return TryBeginInternalOp(); }

void WeightReceiveSession::EndOp() { EndInternalOp(); }

bool WeightReceiveSession::TryBeginInternalOp() {
  absl::MutexLock lock(mu_);
  if (draining_ || done_) {
    return false;
  }
  ++in_flight_;
  return true;
}

void WeightReceiveSession::EndInternalOp() {
  absl::MutexLock lock(mu_);
  if (in_flight_ > 0) {
    --in_flight_;
  }
  if (draining_ && in_flight_ == 0) {
    done_ = true;
  }
}

absl::Status WeightReceiveSession::GetStatus() const {
  absl::MutexLock lock(mu_);
  return status_;
}

absl::Status WeightReceiveSession::AwaitForDone() {
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(&done_));
  return status_;
}

bool WeightReceiveSession::IsDraining() const {
  absl::MutexLock lock(mu_);
  return draining_;
}

absl::StatusOr<size_t> WeightReceiveSession::ResolveStagingSlot(size_t unit) {
  return absl::NotFoundError(
      "WeightReceiveSession addresses host buffers in place");
}

absl::Status WeightReceiveSession::OnDataReceived(
    size_t major_dimension_idx, absl::Span<const int> minor_dimension_indices,
    absl::Span<const int> shard_ids) {
  if (!minor_dimension_indices.empty()) {
    return absl::OkStatus();
  }
  if (!TryBeginInternalOp()) {
    absl::Status status = GetStatus();
    return !status.ok() ? status
                        : absl::FailedPreconditionError(
                              "WeightReceiveSession is draining or done");
  }
  absl::Cleanup end_op = [this]() { EndInternalOp(); };
  if (deadline_ != std::chrono::steady_clock::time_point::max() &&
      std::chrono::steady_clock::now() >= deadline_) {
    absl::Status err = absl::DeadlineExceededError(
        "WeightReceiveSession deadline exceeded on incoming data");
    Finish(err);
    return err;
  }

  const uint64_t layer_idx = static_cast<uint64_t>(major_dimension_idx);
  bool layer_completed = false;
  bool all_done = false;
  {
    absl::MutexLock lock(progress_mu_);
    ABSL_RETURN_IF_ERROR(RecordLayerBatchLocked(layer_idx, remaining_per_layer_,
                                                total_remaining_,
                                                layer_completed, all_done));
  }
  if (layer_completed && on_layer_received_) {
    absl::Status cb_status = on_layer_received_(layer_idx);
    if (!cb_status.ok()) {
      Finish(cb_status);
      return cb_status;
    }
  }
  if (all_done) {
    Finish(absl::OkStatus());
  }
  return absl::OkStatus();
}

tsl::Future<> WeightReceiveSession::GetDataReadyFuture(
    size_t major_dimension_idx) {
  return tsl::Future<>(absl::OkStatus());
}

absl::optional<std::shared_ptr<const ::tpu_raiden::TransferPlan>>
WeightReceiveSession::plan() const {
  return absl::nullopt;
}

std::chrono::steady_clock::time_point WeightReceiveSession::deadline() const {
  return deadline_;
}

absl::Status WeightReceiveSession::MarkStarted() {
  absl::MutexLock lock(mu_);
  if (started_) {
    return absl::FailedPreconditionError(
        "WeightReceiveSession::Start() already called");
  }
  if (draining_ || done_) {
    return absl::FailedPreconditionError(
        "WeightReceiveSession is already draining or done");
  }
  started_ = true;
  return absl::OkStatus();
}

bool WeightReceiveSession::HasNoRemainingBatches() const {
  absl::MutexLock lock(progress_mu_);
  return total_remaining_ == 0;
}

absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
WeightReceiveSession::CreatePullReceiveSession(
    GrpcTransportClient* client,
    const ::tpu_sync::rpc::StartTransferRequest& spec,
    HostBufferProvider host_buffer_provider,
    WeightReceiveSessionOptions options) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "PullReceiveSession requires a non-null GrpcTransportClient");
  }
  if (!host_buffer_provider) {
    return absl::InvalidArgumentError(
        "PullReceiveSession requires a non-empty HostBufferProvider");
  }
  if (spec.uuid() < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "StartTransferRequest uuid must be non-negative, got ", spec.uuid()));
  }

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

  using TaskKey = std::tuple<std::string, uint64_t, uint32_t, uint64_t>;
  absl::flat_hash_set<TaskKey> seen_tasks;
  std::vector<PullReceiveSession::PullTask> tasks;
  absl::flat_hash_map<uint64_t, size_t> expected_per_layer;

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
      if (entry.dst_shard_idx() < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Negative dst_shard_idx ", entry.dst_shard_idx()));
      }
      std::vector<std::string> peers = ExtractEntryPeers(entry);
      if (peers.empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("ShardPushEntryProto for src_shard=", src_shard_idx,
                         " entry[", i, "] has no peers"));
      }
      const uint64_t buffer_id = static_cast<uint64_t>(layer_idx);
      const uint32_t shard_id = static_cast<uint32_t>(entry.dst_shard_idx());
      const uint64_t chunk_batch_id = static_cast<uint64_t>(src_shard_idx);

      for (const std::string& peer : peers) {
        if (peer.empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat("ShardPushEntryProto for src_shard=", src_shard_idx,
                           " entry[", i, "] has empty peer"));
        }
        TaskKey key{peer, buffer_id, shard_id, chunk_batch_id};
        if (!seen_tasks.insert(key).second) {
          continue;
        }
        ABSL_ASSIGN_OR_RETURN(absl::Span<uint8_t> dst_buffer,
                              host_buffer_provider(buffer_id, shard_id));
        tasks.push_back(PullReceiveSession::PullTask{
            .peer = peer,
            .buffer_id = buffer_id,
            .shard_id = shard_id,
            .chunk_batch_id = chunk_batch_id,
            .dst_buffer = dst_buffer,
        });
        ++expected_per_layer[buffer_id];
      }
    }
  }

  const int max_concurrency = spec.parallelism() > 0 ? spec.parallelism() : 1;
  return std::make_shared<PullReceiveSession>(
      client, static_cast<uint64_t>(spec.uuid()), max_concurrency,
      std::move(tasks), std::move(expected_per_layer), std::move(options));
}

absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
WeightReceiveSession::CreatePushReceiveSession(
    const ::tpu_sync::rpc::StartTransferRequest& spec,
    WeightReceiveSessionOptions options) {
  if (spec.uuid() < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "StartTransferRequest uuid must be non-negative, got ", spec.uuid()));
  }
  absl::flat_hash_map<uint64_t, size_t> expected_per_layer;
  size_t total_expected = 0;
  for (const auto& [layer_idx, count] : spec.expected_layer_chunk_counts()) {
    if (layer_idx < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Negative layer_idx in expected_layer_chunk_counts: ", layer_idx));
    }
    if (count < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Negative expected chunk count for layer_idx=", layer_idx, ": ",
          count));
    }
    if (count == 0) {
      continue;
    }
    const uint64_t count_u64 = static_cast<uint64_t>(count);
    if (count_u64 > std::numeric_limits<size_t>::max() - total_expected) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Overflow accumulating expected_layer_chunk_counts at layer_idx=",
          layer_idx));
    }
    expected_per_layer[static_cast<uint64_t>(layer_idx)] =
        static_cast<size_t>(count_u64);
    total_expected += static_cast<size_t>(count_u64);
  }
  return std::make_shared<PushReceiveSession>(
      static_cast<uint64_t>(spec.uuid()), std::move(expected_per_layer),
      total_expected, std::move(options));
}

}  // namespace tpu_raiden::weight_sync

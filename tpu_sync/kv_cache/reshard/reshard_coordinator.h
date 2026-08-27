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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_COORDINATOR_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_COORDINATOR_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/kv_cache/reshard/pool_reshard_planner.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {
namespace reshard {

// The per-target StartTransferRequest (sender gets its own schedule under
// key 0; the receiver gets every source's schedule keyed by source
// ordinal). Split out of EncodeStartTransfer so controller delivery compiles
// the same proto object the framed wire would carry.
tpu_sync::rpc::StartTransferRequest BuildStartTransferForTarget(
    const PoolReshardPlan& plan, const RaidenId& target);

// Wire encoder: the pool-path port of WorkerRpcClient._encode_start_transfer.
// Builds the framed ControlRequest payload for one target unit.
std::string EncodeStartTransfer(const PoolReshardPlan& plan,
                                const RaidenId& target);

// Selects worker delivery. kFramed sends framed TCP requests directly to each
// worker. kController routes sender programs through the local
// RaidenController's persistent WorkerService channels and sends receiver-arm
// requests to the destination store's framed surface, where the destination
// dispatches them to its own workers through its own controller.
struct WorkerDelivery {
  enum class Mode { kFramed, kController };
  Mode mode = Mode::kFramed;
  // Borrowed; required (non-null) when mode == kController.
  tpu_raiden::controller::RaidenController* controller = nullptr;
};

// Decoded CoordinateTransferRequest arguments for the pool path.
struct PoolReshardArgs {
  std::vector<RaidenId> src_units;
  std::vector<RaidenId> dst_units;
  std::string req_id;
  std::vector<int64_t> dst_device_block_ids;
  bool has_dst_device_block_ids = false;
  bool use_block_chunks = false;
  int32_t dst_mem_type = 0;  // rpc::MemoryType value
  std::string dst_controller_address;
  int64_t uuid = 0;  // 0 = unset (Python None → random)
  bool is_sender = true;
  std::optional<int64_t> num_tokens;
  std::vector<std::string> transfer_pool_tags;
  bool has_transfer_pool_tags = false;
  std::optional<int32_t> parallelism;
  bool skip_d2h = false;
  std::vector<int64_t> dst_block_counts;
  // Per-tag destination clip (empty = all zeros); see PlanRequest.
  std::vector<int64_t> dst_skip_bytes;
};

// Port of _start_pool_reshard_transfer + _execute_pool_reshard, executed
// synchronously (the Python server awaits the coroutine before responding;
// C++ runs the same sequence inline): validate → metadata → plan (claim) →
// arm destination and await ack (abandon claim on failure) → dispatch the
// senders in parallel → milestone JSON on stderr.
class ReshardCoordinator {
 public:
  ReshardCoordinator(WorkUnitDirectory* directory,
                     RequestBlockRegistry* registry,
                     FramedTransport* transport,
                     WorkerDelivery delivery = WorkerDelivery{});

  // GetTransferStatusResponse::Status values.
  enum class Status : int {
    kNotStarted = 1,
    kInProgress = 2,
    kCompleted = 3,
    kFailed = 4,
  };

  absl::Status StartPoolReshard(const PoolReshardArgs& args);

  Status GetTransferStatus(const std::string& req_id) const;

 private:
  absl::Status ExecutePoolReshard(const PoolReshardArgs& args);

  // GET_METADATA against the destination controller (dst_controller_address
  // path), recorded shape-identical to Python's _query_remote_metadata.
  absl::StatusOr<std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>>
  QueryRemoteMetadata(const std::string& address);

  WorkUnitDirectory* directory_;
  RequestBlockRegistry* registry_;
  FramedTransport* transport_;
  WorkerDelivery delivery_;

  mutable absl::Mutex status_mu_;
  std::map<std::string, Status> transfer_status_ ABSL_GUARDED_BY(status_mu_);
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_COORDINATOR_H_

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

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/raiden_id.h"
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
// ordinal).
tpu_sync::rpc::StartTransferRequest BuildStartTransferForTarget(
    const PoolReshardPlan& plan, const RaidenId& target);

// Builds the ControlRequest proto for one target unit.
tpu_sync::rpc::ControlRequest BuildStartTransferControlRequest(
    const PoolReshardPlan& plan, const RaidenId& target);

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
  // Per-tag bounded-staging window (empty = unbounded); see
  // CoordinateTransferRequest.dst_window_blocks.
  std::vector<int64_t> dst_window_blocks;
};

// One tag's slice of one sub-transfer: destination blocks
// [offset, offset + count) of the tag's full destination list (`total`
// blocks; 0 = the tag is not windowed and rides whole).
struct SubTransferSlice {
  size_t tag_index = 0;
  int64_t offset = 0;
  int64_t count = 0;
  int64_t total = 0;
};

// How a request is split into sub-transfers by its per-tag windows: `count`
// sub-transfers, each carrying one slice per tag present in it (request tag
// order). A request without windows (or one that fits them) has count 1 and
// no slices, which plans exactly as before windows existed.
struct SubTransferChain {
  int32_t count = 1;
  std::vector<std::vector<SubTransferSlice>> sub_plans;
};

// Splits a request by `args.dst_window_blocks`: a tag with n destination
// blocks and window W (0 = unbounded) becomes ceil(n / W) windows in
// destination order; the chain is as long as the longest tag and shorter
// tags occupy its last sub-transfers, so a single-block state tag rides
// with the final paged window and lands after it (order_rank). Tags whose
// destination ids intersect (aliased arena pages) disable windowing for the
// request. InvalidArgument on malformed windows.
absl::StatusOr<SubTransferChain> PlanSubTransfers(const PoolReshardArgs& args);

// The engine plan uuid of sub-transfer `index` of a request coordinated
// under `uuid`: `uuid` itself for index 0, a positive mix otherwise.
int64_t SubPlanUuid(int64_t uuid, int32_t index);

// Port of _start_pool_reshard_transfer + _execute_pool_reshard, executed
// synchronously (the Python server awaits the coroutine before responding;
// C++ runs the same sequence inline): validate → metadata → plan (claim) →
// arm destination and await ack (abandon claim on failure) → dispatch the
// senders in parallel → milestone JSON on stderr.
//
// A request whose tags exceed their bounded-staging windows is executed as a
// chain of sub-transfers: the first one runs inline exactly like a whole
// request (the RPC returns once its senders are dispatched); the rest run in
// destination order on a coordinator thread, each armed (the receivers'
// staging leases pace the chain) and dispatched like the first, all under
// the request's single registry claim.
class ReshardCoordinator {
 public:
  ReshardCoordinator(WorkUnitDirectory* directory,
                     RequestBlockRegistry* registry, ControlPipeClient* client,
                     WorkerDelivery delivery = WorkerDelivery{});
  // Joins every sub-transfer chain still running (a chain checks for
  // shutdown between sub-transfers).
  ~ReshardCoordinator();

  // GetTransferStatusResponse::Status values.
  enum class Status : int {
    kNotStarted = 1,
    kInProgress = 2,
    kCompleted = 3,
    kFailed = 4,
  };

  absl::Status StartPoolReshard(const PoolReshardArgs& args);

  Status GetTransferStatus(const std::string& req_id) const;

  // Blocks until every sub-transfer chain started so far has finished
  // (tests; a chain ends by completing or by failing a sub-transfer).
  void WaitForSubTransferChains();

 private:
  struct ChainContext;

  absl::Status ExecutePoolReshard(const PoolReshardArgs& args);

  // Plans, arms and dispatches sub-transfer `index` of `ctx`'s request and
  // emits its milestone event. Sub-transfer 0 abandons the request's claim
  // on a planning or arm failure (nothing was dispatched yet); later ones
  // leave the claim to the dispatched senders' terminals.
  absl::Status RunSubPlan(const std::shared_ptr<ChainContext>& ctx,
                          int32_t index);

  // Runs sub-transfers 1..count-1 of `ctx` on a coordinator thread.
  void StartChain(std::shared_ptr<ChainContext> ctx);
  void ReapFinishedChainsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(chains_mu_);

  // GET_METADATA against the destination controller (dst_controller_address
  // path), recorded shape-identical to Python's _query_remote_metadata.
  absl::StatusOr<std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>>
  QueryRemoteMetadata(const std::string& address);

  // Sends one worker RPC via client_ and applies
  // WorkerRpcClient._verify_response. Returns the reply on success.
  absl::StatusOr<tpu_sync::rpc::ControlResponse> SendWorkerRpc(
      const std::string& address, const tpu_sync::rpc::ControlRequest& request);

  WorkUnitDirectory* directory_;
  RequestBlockRegistry* registry_;
  ControlPipeClient* client_;
  WorkerDelivery delivery_;

  mutable absl::Mutex status_mu_;
  std::map<std::string, Status> transfer_status_ ABSL_GUARDED_BY(status_mu_);

  std::atomic<bool> stopping_{false};
  absl::Mutex chains_mu_;
  std::vector<std::pair<std::shared_ptr<ChainContext>, std::thread>>
      chains_ ABSL_GUARDED_BY(chains_mu_);
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_COORDINATOR_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_MANAGER_WITH_TRANSFER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_MANAGER_WITH_TRANSFER_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {

class MetricsCollector;
class ReshardReceiveSession;
class ReshardSendSession;
class TransferSendSession;
class TransferReceiveSession;

// What is known about one settled transfer. Fields are unset when not
// recorded; new fields must be optional too.
struct TransferDetails {
  // When the transfer completed, in steady_clock nanoseconds: CLOCK_MONOTONIC,
  // the clock of Python's time.perf_counter_ns().
  std::optional<int64_t> completed_ns;
  // Duration from transfer registration until settlement, in nanoseconds.
  std::optional<int64_t> elapsed_ns;
  // Comma-separated remote peer endpoints.
  std::optional<std::string> peer;
  // Canonical absl::StatusCode name (e.g. "DEADLINE_EXCEEDED") and error
  // message when the transfer failed; unset if it succeeded.
  std::optional<std::string> code;
  std::optional<std::string> reason;
  // Execution phase in which a failed transfer failed: "await_pull", "staging",
  // "d2h", or "push" for a send; "connect", "receive", or "h2d" for a receive.
  std::optional<std::string> phase;
};

// Populates `TransferDetails` with the status code and error message if
// |status| is non-OK, or returns an empty `TransferDetails` on success.
TransferDetails DetailsFromStatus(const absl::Status& status);

// Result of CompleteReadWithDetails(): transfers settled since the previous
// poll.
struct CompleteReadResult {
  std::vector<std::string> sent;
  std::vector<std::string> received;
  std::vector<std::string> failed;
  // Details of settled transfers, by req_id: recorded for every failed transfer
  // and for every settled transfer send or receive session. Other settled
  // req_ids have no entry.
  absl::flat_hash_map<std::string, TransferDetails> details;
};

struct CopySpec {
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;
};

struct CopyPlan {
  int64_t num_blocks = 0;
  std::vector<int64_t> requested_remote_block_ids;
  std::vector<int64_t> requested_local_block_ids;
  std::vector<int64_t> producer_remote_block_ids;
  std::vector<int64_t> h2d_local_block_ids;
  std::vector<int64_t> h2d_host_block_ids;
  std::vector<int64_t> transport_host_block_ids;
  std::vector<size_t> host_dst_to_src;
  CopySpec d2h_copy;
  CopySpec h2d_copy;

  bool RequiresHostReorder() const { return !host_dst_to_src.empty(); }
};

// Builds a `StagingBlockAllocator` over |base|'s host block manager. When
// |num_slots| and |max_blocks| are both positive, also registers the slot
// layout with |base| via `ConfigureHostStagingSlots()` so `HostSpan()` can
// address the slots. Throws `std::runtime_error` if the slot pool cannot be
// reserved (mirrors the constructor contract of `KVCacheManagerWithTransfer`).
std::unique_ptr<StagingBlockAllocator> CreateStagingBlockAllocator(
    kv_cache::KVCacheManagerBase* base, int64_t num_slots, int64_t max_blocks,
    std::optional<bool> dynamic_host_staging = std::nullopt);

class KVCacheManagerWithTransfer {
 public:
  friend class TransferReceiveSession;
  friend class ReshardReceiveSession;

  KVCacheManagerWithTransfer(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      bool unsafe_skip_buffer_lock, int parallelism,
      HostBufferAllocator host_allocator, int64_t node_id = 0,
      int64_t local_control_port = -1, int64_t max_blocks = 0,
      int64_t num_slots = 0, double timeout_s = 120.0,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  KVCacheManagerWithTransfer(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      size_t slice_byte_size, const std::vector<int64_t>& dimensions,
      size_t physical_size, std::optional<int> local_port,
      std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
      int parallelism, HostBufferAllocator host_allocator, int64_t node_id = 0,
      int64_t local_control_port = -1, int64_t max_blocks = 0,
      int64_t num_slots = 0, double timeout_s = 120.0,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr,
      std::optional<int> num_staging_layers = std::nullopt);

  // Metadata-based constructor for a heterogeneous registration:
  // slice_byte_sizes carries one stride per block array, in registration
  // order.  Used by the host store node, which mirrors a serving host's
  // geometry CPU-side and must match it array for array.
  KVCacheManagerWithTransfer(
      size_t num_layers, size_t num_shards,
      std::vector<size_t> slice_byte_sizes, std::optional<int> local_port,
      std::optional<int> host_blocks_to_allocate, int parallelism = 1,
      int64_t node_id = 0, int64_t local_control_port = -1,
      int64_t max_blocks = 0, int64_t num_slots = 0, double timeout_s = 120.0,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  // Metadata-based constructor for FFI / CPU-only testing
  KVCacheManagerWithTransfer(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      int parallelism = 1, int64_t node_id = 0, int64_t local_control_port = -1,
      int64_t max_blocks = 0, int64_t num_slots = 0, double timeout_s = 120.0,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  void SetMetricsCollector(std::shared_ptr<MetricsCollector> collector) {
    metrics_collector_ = std::move(collector);
  }

  virtual ~KVCacheManagerWithTransfer();

  kv_cache::KVCacheManagerBase* base() { return base_.get(); }
  const kv_cache::KVCacheManagerBase* base() const { return base_.get(); }

  // Registers `block_ids` to be served when a consumer pulls `uuid`. Returns
  // `uuid`, or 0 when there is nothing to send (`block_ids` is empty, or the
  // consumer has already acknowledged the transfer). Throws
  // `std::invalid_argument` if `block_ids` contains duplicate block IDs, and
  // `std::runtime_error` if `uuid` already has an active registered read.
  virtual int64_t NotifyForRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<int64_t>& block_ids,
      std::optional<std::chrono::steady_clock::time_point> deadline =
          std::nullopt);

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt,
      std::optional<std::chrono::steady_clock::time_point> deadline =
          std::nullopt);

  // Reports each transfer settled since the previous poll, exactly once, with
  // its details.
  CompleteReadResult CompleteReadWithDetails();

  // CompleteReadWithDetails() without details: (sent, received, failed). Both
  // drain the same reports, so a caller uses one or the other.
  virtual std::tuple<std::vector<std::string>, std::vector<std::string>,
                     std::vector<std::string>>
  CompleteReadRaw();

  virtual absl::Status RegisterActivePlan(
      uint64_t uuid, const ::tpu_sync::rpc::StartTransferRequest& request,
      bool is_sender);
  virtual absl::Status UnregisterActivePlan(uint64_t uuid);

  virtual absl::Status RegisterRecv(
      uint64_t uuid, const std::string& req_id, int64_t expected_block_count,
      std::optional<std::chrono::steady_clock::time_point> deadline =
          std::nullopt);

  // Pool reshard executor. The controller owns topology and entry math; this
  // boundary validates the declared pool contract before any staging or
  // network side effect. Both methods are device-only: a manager without
  // device attachments fails closed.
  virtual absl::Status PoolReshardPush(
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> src_block_ids, int parallelism = 8);

  virtual absl::Status PoolReshardRegisterRecv(
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_block_ids);

  virtual absl::Status WaitForPendingWork();

  bool IsShuttingDown() const {
    return shutting_down_.load(std::memory_order_relaxed);
  }

  virtual std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  // Endpoints for the block-transport DATA protocol (pulls and pushes),
  // always the transport server's own port -- unlike get_local_endpoints(),
  // which prefers the CONTROL port when a control server is running because
  // StartRead speaks the control protocol.
  virtual std::vector<RaidenTransferEndpoint> get_local_data_endpoints() const;

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual int local_control_port() const { return local_control_port_; }
  virtual int64_t node_id() const { return node_id_; }

  // Bound on one control handshake, end to end, in seconds.
  double control_timeout_s() const { return control_timeout_s_; }

  // Returns the active transfer session registered under |uuid|, or kNotFound
  // when no session holds it. Thread-safe.
  absl::StatusOr<std::shared_ptr<TransferSession>> GetSession(
      uint64_t uuid) const;

 protected:
  KVCacheManagerWithTransfer(
      std::unique_ptr<kv_cache::KVCacheManagerBase> base, int64_t node_id = 0,
      int64_t local_control_port = -1, int64_t max_blocks = 0,
      int64_t num_slots = 0, double timeout_s = 120.0,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

  std::unique_ptr<kv_cache::KVCacheManagerBase> base_;

  std::vector<RaidenTransferEndpoint> BuildEndpoints(int64_t port) const;

  void AckSend(uint64_t uuid);
  void ConfigureDataPortFromKvTransfer();

  absl::Status EmplaceRecvSessionLocked(
      uint64_t uuid, const std::shared_ptr<TransferReceiveSession>& session)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename SessionT>
  std::shared_ptr<SessionT> FindSessionLocked(uint64_t uuid) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = sessions_.find(uuid);
    if (it == sessions_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<SessionT>(it->second);
  }

  // If |it->second| is already Done(), records its terminal completion into
  // |done_sending_| / |done_recving_| / |failed_recving_| /
  // |transfer_details_|, erases |it| from |sessions_|, and returns true.
  // Otherwise returns false. Never calls Finish(): a caller that wants a
  // ready-to-complete session retired must finish it first.
  bool RetireSettledSessionLocked(
      absl::flat_hash_map<uint64_t, std::shared_ptr<TransferSession>>::iterator
          it,
      bool* retired_push_progress = nullptr,
      bool* retired_pool_staging = nullptr) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void StartControlServer();
  void StopControlServer();

  absl::flat_hash_map<uint64_t, std::shared_ptr<TransferSession>> sessions_
      ABSL_GUARDED_BY(mu_);

  absl::flat_hash_set<uint64_t> pending_pool_reshard_recvs_
      ABSL_GUARDED_BY(mu_);

  std::chrono::steady_clock::time_point DeadlineFromNow() const;

  int64_t node_id_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;

  // Pull-serve workers launched by ProcessPullStream. The destructor waits
  // for them, and shutting_down_ ends a worker's staging wait early.
  std::atomic<bool> shutting_down_{false};
  absl::Mutex pull_workers_mu_;
  int active_pull_workers_ ABSL_GUARDED_BY(pull_workers_mu_) = 0;
  double timeout_s_ = 120.0;
  // Bound on one control handshake, end to end. Separate from timeout_s_,
  // which sizes a bulk KV transfer. Derived in the delegated-to constructor;
  // see DeriveControlTimeoutS.
  double control_timeout_s_ = 10.0;

  std::unique_ptr<StagingBlockAllocator> staging_allocator_;
  absl::flat_hash_set<uint64_t> pending_acks_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<std::string> done_sending_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<std::string> done_recving_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<std::string> failed_recving_ ABSL_GUARDED_BY(mu_);
  // CompleteReadResult::details not yet reported.
  absl::flat_hash_map<std::string, TransferDetails> transfer_details_
      ABSL_GUARDED_BY(mu_);
  // A request coordinated as a chain of pool-reshard sub-plans
  // (StartTransferRequest.sub_plan_count > 1) gets one terminal: reported
  // when its last sub-plan settles, or at its first sub-plan failure (later
  // sub-plans of a failed request are swallowed until the entry expires).
  // A chain whose next sub-plan never arrives is failed by the sweep once
  // it has had no session for timeout_s_.
  //
  // Registering sub-plan 0 of a chain starts a fresh entry for its req_id
  // (the coordinator arms sub-plan 0 before any other sub-plan of the
  // chain), so a retry under a reused req_id is tracked on its own; only
  // sessions registered into the current entry (`plan_uuids`) count toward
  // it, and late terminals of an earlier chain's sub-plans are ignored.
  struct SubPlanChainProgress {
    int32_t count = 0;
    absl::flat_hash_set<int32_t> settled;
    bool failed = false;
    std::chrono::steady_clock::time_point last_activity;
    // Set once sub-plan 0 of the chain has registered here; from then on
    // only sessions in `plan_uuids` belong to this chain.
    bool anchored = false;
    absl::flat_hash_set<uint64_t> plan_uuids;
  };
  absl::flat_hash_map<std::string, SubPlanChainProgress> recv_chains_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, SubPlanChainProgress> send_chains_
      ABSL_GUARDED_BY(mu_);
  // Records a newly registered pool-reshard session of plan `uuid` into its
  // request's sub-plan chain (no-op for single-plan requests).
  void NoteSubPlanRegisteredLocked(bool is_recv, const std::string& req_id,
                                   uint64_t uuid, int32_t sub_plan_index,
                                   int32_t sub_plan_count)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Records a settled pool-reshard session of plan `uuid` (`is_recv` selects
  // the role) into the terminal sets, aggregating per request across its
  // sub-plans.
  void RecordPoolReshardTerminalLocked(bool is_recv, const std::string& req_id,
                                       uint64_t uuid, int32_t sub_plan_index,
                                       int32_t sub_plan_count,
                                       const absl::Status& status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Fails stalled sub-plan chains and expires reported ones.
  void SweepSubPlanChainsLocked(std::chrono::steady_clock::time_point now)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  mutable absl::Mutex mu_;
  absl::CondVar cv_;
  std::atomic<bool> stopping_{false};
  std::unique_ptr<ControlPlaneHandler> control_handler_;
  std::unique_ptr<ControlPlaneBackend> control_backend_;
  std::shared_ptr<MetricsCollector> metrics_collector_ = nullptr;

 private:
  class ControlPlaneHandlerImpl;

  void InitializeBaseHooks();
  void InitializeControlPlane();
  absl::StatusOr<PullStreamResponseSpec> HandlePullStream(
      const PullStreamRequestSpec& req, absl::string_view fallback_peer_ip);
  absl::Status HandleAck(uint64_t uuid);
  uint64_t MaxPullStreamBlocks() const;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_CACHE_MANAGER_WITH_TRANSFER_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
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
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/transport/block_transport_delegate.h"

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

// Encapsulates host staging block allocation and release (both fixed slots
// and on-demand dynamic host blocks) behind an RAII |Allocation| handle that
// can be transferred into session classes.
class StagingBlockAllocator {
 public:
  // RAII handle owning either a fixed staging slot or a set of dynamically
  // allocated locked host blocks. Releases the resource automatically upon
  // destruction or Reset().
  class Allocation {
   public:
    Allocation() = default;
    Allocation(std::nullptr_t) {}  // NOLINT(runtime/explicit)
    Allocation(StagingBlockAllocator* allocator, int64_t slot_idx,
               std::vector<int> block_ids);
    Allocation(StagingBlockAllocator* allocator,
               std::vector<int> dynamic_blocks);
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    ~Allocation();

    // Releases any held slot or dynamic host blocks back to the allocator.
    // Safe to call multiple times.
    void Reset();

    bool empty() const { return slot_idx_ < 0 && block_ids_.empty(); }
    explicit operator bool() const { return !empty(); }
    bool is_dynamic() const { return slot_idx_ < 0 && !block_ids_.empty(); }
    int64_t slot_idx() const { return slot_idx_; }
    absl::Span<const int> block_ids() const { return block_ids_; }

   private:
    StagingBlockAllocator* allocator_ = nullptr;
    int64_t slot_idx_ = -1;
    std::vector<int> block_ids_;
  };

  static std::unique_ptr<StagingBlockAllocator> Create(
      kv_cache::KVCacheManagerBase* base, int64_t num_slots, int64_t max_blocks,
      std::optional<bool> dynamic_host_staging = std::nullopt);
  ~StagingBlockAllocator();
  StagingBlockAllocator(const StagingBlockAllocator&) = delete;
  StagingBlockAllocator& operator=(const StagingBlockAllocator&) = delete;

  // Acquires staging for |num_blocks| blocks: a fixed slot when
  // !dynamic_host_staging_, or |num_blocks| locked dynamic host blocks when
  // dynamic_host_staging_ is enabled. Returns an error if staging capacity is
  // currently unavailable.
  absl::StatusOr<Allocation> Acquire(int64_t num_blocks);

  // Waits until |deadline| to acquire staging for |num_blocks| blocks. Returns
  // an error if |num_blocks| exceeds capacity(), if Shutdown() is called, or if
  // |deadline| expires before staging capacity becomes available.
  absl::StatusOr<Allocation> AcquireWithTimeout(
      int64_t num_blocks, std::chrono::steady_clock::time_point deadline);

  // Signals shutdown to wake any threads waiting in AcquireWithTimeout().
  void Shutdown();

  // Leases bounded pool staging arena slots for |uuid| on |storage_index| for
  // |device_block_ids| via |base_|.
  absl::Status AcquirePoolStagingLease(
      uint64_t uuid, size_t storage_index,
      absl::Span<const int64_t> device_block_ids);

  // Releases all bounded pool staging leases held by |uuid|.
  void ReleasePoolStagingLeases(uint64_t uuid);

  size_t num_free_slots() const;
  absl::Span<const int> slot_blocks(int64_t slot_idx) const;
  int64_t num_slots() const { return num_slots_; }
  int64_t max_blocks() const { return max_blocks_; }
  int64_t capacity() const;
  bool dynamic_host_staging() const { return dynamic_host_staging_; }

 private:
  StagingBlockAllocator(kv_cache::KVCacheManagerBase* base, int64_t num_slots,
                        int64_t max_blocks, bool dynamic_host_staging);
  absl::Status Initialize();
  absl::Status InitializeSlotPool();
  absl::StatusOr<Allocation> AcquireLocked(int64_t num_blocks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ReleaseSlot(int64_t slot_idx);
  void ReleaseDynamicBlocks(absl::Span<const int> blocks);

  mutable absl::Mutex mu_;
  kv_cache::KVCacheManagerBase* const base_ = nullptr;
  bool shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  const int64_t num_slots_ = 0;
  const int64_t max_blocks_ = 0;
  const bool dynamic_host_staging_ = false;
  std::deque<int64_t> free_slots_ ABSL_GUARDED_BY(mu_);
  std::vector<std::vector<int>> slot_blocks_;
};
using StagingAllocation = StagingBlockAllocator::Allocation;

class KVCacheManagerWithTransfer {
 public:
  friend class TransferReceiveSession;
  friend class ReshardReceiveSession;
  friend class ReshardSendSession;

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
      std::optional<int> assigned_numa_node = std::nullopt,
      std::shared_ptr<MetricsCollector> metrics_collector = nullptr);

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

  virtual absl::Status OnBlocksReceived(const std::vector<int>& block_ids,
                                        uint64_t uuid = 0);

  virtual void OnReceiveFailed(uint64_t uuid, const absl::Status& status);

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

  // Serializes plan registration and unregistration, so a plan is never
  // published without its staging owner or torn down against a half-built
  // registration.
  absl::Mutex plan_lifecycle_mu_;
  // Source of plan generations: the uuid names a transfer, a generation
  // names one registration of it (the same uuid may come back, e.g. on a
  // retry of the same transfer).
  uint64_t plan_generation_counter_ ABSL_GUARDED_BY(plan_lifecycle_mu_) = 0;
  // Host staging held by a plan: a sender's, or a receiver's whose
  // destination is host memory. Released when the plan is unregistered.
  absl::flat_hash_map<uint64_t, StagingAllocation> plan_staging_
      ABSL_GUARDED_BY(mu_);
  absl::Status EmplaceRecvSessionLocked(
      uint64_t uuid, const std::shared_ptr<TransferReceiveSession>& session)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void StartControlServer();
  void StopControlServer();
  virtual void RegisterBlockReadinessCallback(
      size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
      transport::BlockTransportDelegate::HostBlockReadyCallback cb);

  absl::flat_hash_map<uint64_t, std::shared_ptr<TransferReceiveSession>>
      active_recv_sessions_;

  absl::flat_hash_map<uint64_t, std::shared_ptr<ReshardReceiveSession>>
      active_pool_reshard_recvs_;

  absl::flat_hash_map<uint64_t, std::shared_ptr<ReshardSendSession>>
      active_pool_reshard_sends_;

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
  // TransferSendSession is shared across threads: created/timed-out/cleaned-up
  // on the main thread, but accessed asynchronously in control worker threads
  // handling pull connections.
  absl::flat_hash_map<uint64_t, std::shared_ptr<TransferSendSession>>
      send_sessions_;
  absl::flat_hash_set<uint64_t> pending_acks_;
  absl::flat_hash_set<std::string> done_sending_;
  absl::flat_hash_set<std::string> done_recving_;
  absl::flat_hash_set<std::string> failed_recving_;
  absl::Mutex mu_;
  absl::CondVar cv_;
  std::atomic<bool> stopping_{false};
  std::unique_ptr<ControlPlaneHandler> control_handler_;
  std::unique_ptr<ControlPlaneBackend> control_backend_;

 private:
  class ControlPlaneHandlerImpl;

  void InitializeBaseHooks();
  void InitializeControlPlane();
  // Drops the plan of a receive that has settled; a plan already gone, or
  // a newer registration reusing the uuid, is left alone.
  void UnregisterSettledPlan(uint64_t uuid, uint64_t generation);
  void MaybeUnregisterSettledRecv(uint64_t uuid,
                                  TransferReceiveSession& session);
  // Drops the plan of a pool-reshard receive that has settled (its H2D
  // uploads and every push it admitted have ended); a no-op before then.
  void MaybeUnregisterSettledReshardRecv(uint64_t uuid,
                                         ReshardReceiveSession& session);
  absl::StatusOr<PullStreamResponseSpec> HandlePullStream(
      const PullStreamRequestSpec& req, absl::string_view fallback_peer_ip);
  absl::Status HandleAck(uint64_t uuid);
  uint64_t MaxPullStreamBlocks() const;

  std::shared_ptr<MetricsCollector> metrics_collector_ = nullptr;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_

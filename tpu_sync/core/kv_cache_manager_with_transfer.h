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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
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

class TransferFuture {
 public:
  TransferFuture() = default;

  void Add(raiden::PjRtCopyFuture future) {
    futures_.push_back(std::move(future));
  }

  void AddAll(const std::shared_ptr<TransferFuture>& other) {
    futures_.insert(futures_.end(), other->futures_.begin(),
                    other->futures_.end());
  }

  void Await() {
    for (auto& future : futures_) {
      if (future.IsValid()) {
        absl::Status status = future.Await();
        if (!status.ok()) {
          throw std::runtime_error("Async transfer failed: " +
                                   std::string(status.message()));
        }
      }
    }
    futures_.clear();
  }

  bool IsReady() const {
    for (const auto& future : futures_) {
      if (future.IsValid() && !future.IsReady()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<raiden::PjRtCopyFuture> futures_;
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

struct StageResult {
  // TransferFuture is shared because it is exported to Python via nanobind.
  // Python garbage collection and C++ pending operations list share ownership
  // of the future.
  std::shared_ptr<TransferFuture> future;
  std::vector<kv_cache::KVCacheHostSpan> host_spans;
  int64_t total_bytes = 0;
  int64_t copy_segments = 0;
};

struct CommitResult {
  double duration_issue_ms;
  double duration_wait_ms;
  double duration_total_ms;
  int64_t total_bytes;
};

struct PendingCopy {
  int64_t host_block_id;
  int64_t chip_block_id;
};

class KVCacheManagerWithTransfer : public kv_cache::KVCacheManagerBase {
 public:
  struct Slot {
    int64_t slot_idx;
    std::vector<int> block_ids;
  };

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

  virtual int64_t NotifyForRead(const std::string& req_id, uint64_t uuid,
                                const std::vector<int64_t>& block_ids);

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<std::string>& remote_endpoints,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual std::tuple<std::vector<std::string>, std::vector<std::string>,
                     std::vector<std::string>>
  CompleteReadRaw();

  absl::Status RegisterActivePlan(
      uint64_t uuid, const ::tpu_sync::rpc::StartTransferRequest& request,
      bool is_sender) override;

  absl::Status RegisterRecv(uint64_t uuid, const std::string& req_id,
                            int64_t expected_block_count) override;

  // Pool reshard executor. The controller owns topology and entry math; this
  // boundary validates the declared pool contract before any staging or
  // network side effect. Both methods are device-only: a manager without
  // device attachments fails closed.
  absl::Status PoolReshardPush(
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> src_block_ids, int parallelism = 8) override;

  absl::Status PoolReshardRegisterRecv(
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_block_ids) override;

  absl::Status OnBlocksReceived(const std::vector<int>& block_ids,
                                uint64_t uuid = 0) override;

  virtual std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  // Endpoints for the block-transport DATA protocol (pulls and pushes),
  // always the transport server's own port -- unlike get_local_endpoints(),
  // which prefers the CONTROL port when a control server is running because
  // StartRead speaks the control protocol. Use these anywhere a peer will
  // open a data connection, notably worker registration: the registry's
  // endpoints are what a remote ReadRemote pulls from, and pointing a pull
  // at a control port hangs both sides silently.
  virtual std::vector<RaidenTransferEndpoint> get_local_data_endpoints()
      const;

  static bool EncodeIpToIpv6Bytes(const std::string& ip, uint8_t out[16]);

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual int local_control_port() const { return local_control_port_; }
  virtual int64_t node_id() const { return node_id_; }

 protected:
  std::vector<RaidenTransferEndpoint> BuildEndpoints(int64_t port) const;

  struct SendEntry {
    std::string req_id;
    uint64_t uuid = 0;
    int64_t slot_idx = -1;
    // Host blocks held under demand staging, released on completion. Empty
    // when the transfer holds a fixed slot instead.
    std::vector<int> staged_host_blocks;
    int64_t num_blocks = 0;
    int64_t registered_num_blocks = 0;
    int64_t total_bytes = 0;
    int64_t copy_segments = 0;
    std::vector<int64_t> registered_block_ids;
    std::set<int64_t> registered_block_set;
    std::chrono::steady_clock::time_point register_start;
    std::chrono::steady_clock::time_point d2h_issue_start;
    std::chrono::steady_clock::time_point d2h_issue_done;
    std::chrono::steady_clock::time_point d2h_done;
    bool stage_done = false;
    bool failed = false;
    bool pull_started = false;
    bool slot_released = false;
    std::chrono::steady_clock::time_point deadline;
    std::vector<raiden::PjRtCopyFuture> d2h_layer_futures;
    std::vector<std::string> remote_data_endpoints;
    std::vector<int> src_ints;
    std::vector<int> dst_ints;
    std::atomic<size_t> remaining_h2h_layers{0};
  };

  struct StagingLayerReady {
    bool done = false;
    absl::Status status = absl::OkStatus();
  };

  struct StagingReadinessState {
    int64_t slot_idx = -1;
    int64_t num_blocks = 0;
    size_t num_layers = 0;
    size_t num_shards = 0;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<StagingLayerReady> layers;
  };

  struct PullBlockDescriptor {
    uint64_t remote_block_base = 0;
    uint64_t num_blocks = 0;
  };

  static constexpr int kMaxNics = 8;

  struct alignas(8) ControlRequestHeader {
    uint32_t magic = 0x52414944;  // "RAID"
    uint32_t op = 0;
    uint64_t uuid = 0;
    uint32_t ep_idx = 0;
    uint32_t consumer_data_port = 0;
    uint64_t num_blocks = 0;
    uint32_t num_ips = 0;
    uint8_t consumer_ips[kMaxNics][16] = {{0}};
    uint32_t padding = 0;
  };

  struct alignas(8) ControlResponseHeader {
    uint32_t magic = 0x44494152;  // "DIAR"
    int32_t status = 0;
    uint32_t num_layers = 0;
    uint32_t data_port = 0;
    uint64_t message_len = 0;
  };

  static constexpr uint32_t kControlMagic = 0x52414944;
  static constexpr uint32_t kResponseMagic = 0x44494152;
  static constexpr uint32_t kOpAck = 2;
  static constexpr uint32_t kOpPullStream = 3;

  std::string EndpointWithPort(const std::string& endpoint, int port) const;
  ControlResponseHeader ReadControlResponseHeader(int fd);
  void AckSend(uint64_t uuid);
  void ConfigureDataPortFromKvTransfer();

  std::vector<int> ContiguousBlockIds(uint64_t base, uint64_t count) const;
  static void ValidateRequestedBlocks(
      const SendEntry& entry, const std::vector<int64_t>& requested_block_ids);

  static bool DynamicHostStagingEnabled();
  absl::Status InitializeSlotPool(int64_t num_slots);
  Slot AcquireSlot();
  Slot AcquireSlotLocked();
  void ReleaseSlotLocked(int64_t slot_idx);
  struct RecvEntry;  // defined below; staging helpers take it by pointer
  // Host staging for one incoming read: exactly `num_blocks` blocks under
  // demand staging, a whole fixed slot otherwise. Returns nullopt when the
  // staging pool cannot seat the request.
  std::optional<std::vector<int64_t>> AcquireRecvStagingLocked(
      int64_t num_blocks, RecvEntry* entry);
  void ReleaseRecvStagingLocked(RecvEntry* entry);
  // Same, for paths that have already taken the entry's staging out of it.
  void ReleaseStagingLocked(int64_t slot_idx,
                            std::vector<int>* staged_host_blocks);
  void ReleaseEntrySlotLocked(const std::shared_ptr<SendEntry>& entry);

  void StartControlServer();
  void StopControlServer();
  void ControlServerLoop();
  void HandleControlConnection(int fd);
  void ProcessPullStream(int fd, const ControlRequestHeader& req);
  void AckRemote(const std::string& remote_endpoint, uint64_t uuid);
  absl::Status OnLayerReceived(size_t layer_idx, uint64_t uuid) override;
  absl::Status OnPoolReceived(size_t pool_idx, uint64_t uuid) override;
  void RegisterBlockReadinessCallback(
      size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
      transport::BlockTransportDelegate::HostBlockReadyCallback cb) override;
  void ScheduleAsyncTask(std::function<void()> task) override;
  std::shared_ptr<StagingReadinessState> CreateStagingReadiness(
      int64_t slot_idx, int64_t num_blocks);
  void MarkStagingLayerReady(
      const std::shared_ptr<StagingReadinessState>& state, size_t layer_idx,
      size_t shard_idx, absl::Status status);
  void RemoveStagingReadinessLocked(int64_t slot_idx);

  using H2dIssueFuture =
      std::shared_future<absl::StatusOr<raiden::PjRtCopyFuture>>;

  absl::Status WaitForPendingWork() override;

  struct RecvEntry {
    std::string req_id;
    int64_t slot_idx = -1;  // host staging slot to release on completion
    // Host blocks held under demand staging, released on completion. Empty
    // when the transfer holds a fixed slot instead.
    std::vector<int> staged_host_blocks;
    CopySpec h2d_copy;
    std::vector<int64_t> chip_block_ids;
    absl::flat_hash_map<int64_t, int64_t> host_to_chip;
    std::map<std::pair<size_t, size_t>, std::vector<PendingCopy>>
        pending_h2d_copies;
    std::vector<H2dIssueFuture> h2d_dispatch_futures;
    int32_t total_blocks = 0;
    int32_t num_completed_blocks = 0;
    int32_t num_completed_layers = 0;
    bool network_completed = false;
    bool h2d_started = false;
    bool reshard_finalizing = false;
    std::vector<int> accumulated_host_block_ids;
    std::chrono::steady_clock::time_point deadline;
    std::chrono::steady_clock::time_point start_time;
    std::vector<raiden::PjRtCopyFuture> h2d_futures;
    bool is_pool_reshard = false;
    std::set<size_t> expected_pool_indices;
    std::set<size_t> started_pool_indices;
    std::set<size_t> completed_pool_indices;
    // Multi-tag plans: per-pool H2D upload ordering. A pool's mirror is
    // uploaded only after every expected pool of a strictly lower order
    // rank has completed its upload (FA at rank 0, state classes at rank
    // 1, so state bytes land last on aliased arena pages). Single-tag
    // plans leave every rank 0 (upload immediately on wire completion).
    std::map<size_t, int> pool_order_ranks;
    std::set<size_t> h2d_launched_pools;
    // Multi-tag plans: each pool uploads only its own group's destination
    // block ids (the flat chip_block_ids list concatenates all groups).
    std::map<size_t, std::vector<int64_t>> pool_dst_block_ids;
  };
  absl::flat_hash_map<uint64_t, RecvEntry> active_recv_entries_;

  struct PoolReshardSendEntry {
    std::string req_id;
    uint64_t uuid = 0;
    int parallelism = 8;
    int remaining_pool_peer_pushes = 0;
    bool failed = false;
    bool finalizing = false;
    std::chrono::steady_clock::time_point deadline;
    ::tpu_sync::rpc::StartTransferRequest plan;
    std::vector<raiden::PjRtCopyFuture> d2h_futures;
  };
  absl::flat_hash_map<uint64_t, std::shared_ptr<PoolReshardSendEntry>>
      active_pool_reshard_sends_;

  // Acquires host staging for a pull-serve send, retrying while other
  // transfers hand blocks back, bounded by the entry's deadline. Returns
  // false when the transfer was failed or cancelled instead; the caller
  // has nothing left to do.
  bool AcquireSendStagingWithRetry(uint64_t uuid,
                                   const std::vector<int64_t>& src_block_ids,
                                   std::vector<int64_t>* host_block_ids);
  void StartPushInternal(uint64_t uuid,
                         const std::vector<std::string>& remote_data_endpoints,
                         const std::vector<int64_t>& src_block_ids,
                         const std::vector<int64_t>& dst_block_ids);
  void SendNextLayer(uint64_t uuid, size_t l);

  absl::Status ValidatePoolReshardPlan(
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> local_block_ids, bool is_sender);
  // Receiver-only byte accounting over the plan's group structure: group
  // pools must share one live-region map, declared extents must be
  // prefix-shaped, every destination block's compact-live bytes must be
  // covered exactly once up to its extent, and the group's expected push
  // count must equal the recomputation from the received schedules. This
  // is the trust boundary between a planner defect and silent decode
  // corruption; the arming worker validates for itself.
  absl::Status ValidatePoolReshardReceiverCoverage(
      const ::tpu_sync::rpc::StartTransferRequest& plan);
  void StartPoolReshardPush(uint64_t uuid, size_t pool_idx);
  void FinishPoolReshardSend(uint64_t uuid, const absl::Status& status);
  void FinishPoolReshardRecvPool(uint64_t uuid, size_t pool_idx,
                                 const absl::Status& status);
  // Launches H2D uploads for every wire-complete pool whose order-rank
  // prerequisites (all lower-rank pools uploaded) are satisfied.
  void LaunchEligiblePoolH2ds(uint64_t uuid);

  std::chrono::steady_clock::time_point DeadlineFromNow() const;

  static CopySpec Offsets(const std::vector<int64_t>& block_ids,
                          bool source_is_compact);
  static kv_cache::KVCacheCopySpec ToKVCacheCopySpec(const CopySpec& spec);

  int64_t node_id_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;
  int64_t max_blocks_ = 0;
  int64_t num_slots_ = 0;
  // Allocate host staging per request instead of carving the pool into
  // fixed slots. A short request then costs its own pages rather than a
  // whole slot, so the same pool seats more transfers at once.
  bool dynamic_host_staging_ = false;

  // Pull-serve workers launched by ProcessPullStream. The destructor waits
  // for them, and shutting_down_ ends a worker's staging wait early.
  std::atomic<bool> shutting_down_{false};
  absl::Mutex pull_workers_mu_;
  int active_pull_workers_ ABSL_GUARDED_BY(pull_workers_mu_) = 0;
  double timeout_s_ = 120.0;
  bool unsafe_skip_buffer_lock_ = true;

  std::deque<Slot> free_slots_;
  std::vector<Slot> all_slots_;
  // SendEntry is shared across threads: created/timed-out/cleaned-up on the
  // main thread, but accessed asynchronously in control worker threads
  // handling pull connections.
  std::map<uint64_t, std::shared_ptr<SendEntry>> send_entries_;
  std::set<uint64_t> pending_acks_;
  std::set<std::string> done_sending_;
  std::set<std::string> done_recving_;
  std::set<std::string> failed_recving_;
  // StagingReadinessState is shared because it is captured by value in the
  // async PjRt copy callbacks (e.g. OnReady). A shared_ptr is required here
  // to ensure the state stays alive even if the entry is removed from this
  // map (e.g. on timeout, cancellation, or slot release) before the async
  // callback runs.
  std::map<int64_t, std::shared_ptr<StagingReadinessState>> staging_readiness_;
  std::map<int64_t, std::shared_ptr<StagingReadinessState>>
      active_producer_blocks_;
  absl::Mutex mu_;
  absl::CondVar cv_;
  int control_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread control_thread_;

 private:
  std::optional<int> GetLocalTpuNumaNode(xla::PjRtBuffer* buf) const;

  StageResult IssueH2D(int64_t slot_idx, int64_t num_blocks,
                       const std::vector<int64_t>& local_block_ids);

  std::vector<kv_cache::KVCacheHostSpan> LayerSpans(int64_t slot_idx,
                                                    int64_t num_blocks);

  std::shared_ptr<MetricsCollector> metrics_collector_ = nullptr;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_

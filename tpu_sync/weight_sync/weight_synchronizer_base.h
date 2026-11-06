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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "peregrine/src/api/v1/rate_limiter.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/weight_sync/weight_buffers.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
namespace weight_sync {

class WeightSynchronizerListener;

class WeightSynchronizerBase : public tpu_raiden::RaidenManagerBase {
 public:
  // Creates a synchronizer wrapping the raw PJRT buffers in `layer_buffers`
  // (outer index = layer, inner index = shard) and allocates (or, when
  // `external_host_ptrs` is set, adopts) one host staging buffer per shard.
  // Returns InvalidArgument if the layers disagree on shard count or
  // `external_host_ptrs` has fewer entries than shards, and ResourceExhausted
  // if a host staging buffer cannot be allocated.
  static absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> Create(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      std::optional<int> local_port = std::nullopt,
      std::optional<std::vector<const uint8_t*>> external_host_ptrs =
          std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false);

  // Creates a CPU-only synchronizer (no device buffers) for remote workers and
  // mock E2E testing, with `slice_byte_size` bytes per layer. Returns
  // ResourceExhausted if a host staging buffer cannot be allocated.
  static absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> Create(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1, std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false);

  // As above, with heterogeneous per-layer `slice_byte_sizes` and custom
  // `layer_names`.
  static absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> Create(
      size_t num_layers, size_t num_shards,
      std::vector<size_t> slice_byte_sizes,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1, std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false,
      std::vector<int> numa_nodes = {},
      std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses(),
      std::vector<transport::ShardInfo> shards = {});

  std::optional<int> listener_port() const;
  bool is_listener_active() const;
  virtual std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  ~WeightSynchronizerBase() override;

  void SetGlobalShardIndices(std::vector<int64_t> indices) {
    global_shard_indices_ = std::move(indices);
  }
  int64_t global_shard_index(size_t local_shard_idx) const {
    if (local_shard_idx < global_shard_indices_.size()) {
      return global_shard_indices_[local_shard_idx];
    }
    return static_cast<int64_t>(local_shard_idx);
  }
  void SetLocalShardIndices(std::vector<int> indices) {
    local_shard_indices_ = std::move(indices);
  }
  int64_t local_shard_index(size_t local_shard_idx) const {
    if (local_shard_idx < local_shard_indices_.size()) {
      return local_shard_indices_[local_shard_idx];
    }
    return static_cast<int64_t>(local_shard_idx);
  }

  virtual absl::Status PushWeights(const std::vector<std::string>& peers);
  virtual absl::Status PushWeightsLocal(const std::vector<std::string>& peers);

  /**
   * Executes a distributed resharding push transfer based on precise
   * centralized Controller schedules.
   *
   * Automatically copies active local weight buffers from TPU device HBM to
   * Host staging memory (via D2H), iterates over all active local shards, and
   * pipelines non-contiguous byte chunks across persistent TCP connections to
   * target remote peer host buffers.
   *
   * @param request Demarshaled StartTransferRequest protobuf containing exact
   *                1D memory copy byte chunks, peer network coordinates, and
   *                offset schedules.
   * @return absl::OkStatus() upon complete, successfully ACK-handshaked
   * delivery to all remote peers.
   */
  virtual absl::Status PushWeightsResharded(
      const tpu_sync::rpc::StartTransferRequest& request);
  virtual absl::Status PushWeightsReshardedLocal(
      const tpu_sync::rpc::StartTransferRequest& request);

  const uint8_t* GetHostBufferPtr(size_t layer_idx, size_t shard_idx) const {
    if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
      return nullptr;
    }
    return layers_[layer_idx].shards[shard_idx].host_ptr;
  }

  // Waits for any in-flight explicit D2h() on `layer_idx` and returns the host
  // staging buffer pointer of `(layer_idx, shard_idx)`.
  absl::StatusOr<const uint8_t*> WaitAndGetHostBufferPtr(
      size_t layer_idx, size_t shard_idx) const {
    return buffers_->WaitAndGetHostBufferPtr(layer_idx, shard_idx);
  }
  absl::Status WaitForExplicitD2h(size_t layer_idx) const {
    return buffers_->WaitForExplicitD2h(layer_idx);
  }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override;
  const uint8_t* GetHostPointer(size_t layer_idx,
                                size_t shard_idx) const override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) const override;

  // Returns the tiling scratchpad of `shard_idx`, shared by all layers. The
  // pointer may be reallocated by a later H2D/D2H that needs more space.
  uint8_t* GetTiledPointer(size_t layer_idx, size_t shard_idx) {
    return buffers_->GetTiledPointer(shard_idx);
  }
  const uint8_t* GetTiledPointer(size_t layer_idx, size_t shard_idx) const {
    return const_cast<const WeightBuffers&>(*buffers_).GetTiledPointer(
        shard_idx);
  }

  // Returns the list of layer names associated with the weight synchronizer.
  const std::vector<std::string>& layer_names() const { return layer_names_; }

  // Copies the host staging buffers of all layers to device. For each layer:
  // - If the layer was armed with HOST_TILING_MODE_ON_ARRIVAL or
  //   HOST_TILING_MODE_PRE_TILED (see StartTransferRequest.host_tiling_mode),
  //   waits until the layer has been received (and, for ON_ARRIVAL, tiled
  //   in place by TileLayer()) and copies the host bytes as is. The
  //   transport acks the final chunk before running the receive callbacks,
  //   so this wait covers callers that start H2D as soon as the sender
  //   reports completion.
  // - Otherwise the host buffers hold logical-layout bytes; tiles them into
  //   the per-shard scratchpads and copies those.
  // Must not be called from the receive path (OnLayerDataReceived /
  // OnDataReceived) since it may block on tiling scheduled there.
  virtual absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid = 0);
  // Tiles the host staging buffers of `layer_idx` into device layout and
  // copies the tiled bytes back into the same host buffers. No-op for shards
  // without tiled layouts or when skip_tiling is set for `layer_idx` under
  // `uuid`.
  absl::Status TileLayer(size_t layer_idx, uint64_t uuid = 0);
  // Copies bound device buffers into host staging buffers asynchronously:
  // skip_d2h=true pushes and WaitAndGetHostBufferPtr() wait per layer; H2d()
  // waits for the whole in-flight D2h(). The future holds no device buffers;
  // once a shard's copy+detile completes it no longer holds them (bound weights
  // keep theirs until UnbindWeights()). Serializes behind in-flight D2H; fails
  // if unbound or H2H is sending.
  virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> D2hLayer(
      size_t layer_idx, uint64_t uuid = 0,
      std::shared_ptr<std::atomic<double>> max_detile_ms = nullptr);

  // Replaces bound device buffers in-place (matching initial layer/shard/size).
  // Drains in-flight D2H first, fails if H2H is sending (empty `layer_buffers`
  // aliases UnbindWeights() and stays allowed), and runs `on_rebind` atomically
  // under `device_buffer_resolver_mu_`.
  absl::Status BindWeights(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      absl::AnyInvocable<void()> on_rebind = nullptr);

  // Releases all device buffer holds while keeping host staging buffers,
  // metadata, and listeners intact for a later BindWeights(). Subsequent
  // D2h()/H2d() or device-reading pushes return FailedPrecondition until
  // rebound. Guarded by buffer_holds_mu_; safe once D2H completes even while a
  // skip_d2h=true H2H push streams (keep weights bound across the first
  // transfer to an uncached topology so layout mismatches can re-stage).
  virtual void UnbindWeights();

  virtual void StoreSkipTiling(
      uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request);
  virtual void StoreSkipTilingLocal(
      uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request);

  virtual void SetSkipTiling(const std::vector<bool>& skip_tiling) {
    buffers_->SetSkipTiling(skip_tiling);
  }
  virtual void SetSkipTiling(bool skip_all) {
    buffers_->SetSkipTiling(skip_all);
  }

  virtual WeightSyncMetrics GetMetrics() const {
    return buffers_->GetMetrics();
  }

  virtual void ResetMetrics() { buffers_->ResetMetrics(); }

  void SetMetricsForTesting(const WeightSyncMetrics& m) {
    buffers_->SetMetricsForTesting(m);
  }

  void SetTestOnlyRateLimiters(
      std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> egress,
      std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> ingress);

  void SetPipelineGroupSize(std::optional<size_t> group_size) {
    pipeline_group_size_override_ = group_size;
  }
  std::optional<size_t> pipeline_group_size_override() const {
    return pipeline_group_size_override_;
  }
  size_t GetPipelineGroupSize() const;

  // Block-push completion. Counts the shards reported for `uuid` and runs the
  // auto H2D (which copies every layer and shard) only once the count reaches
  // num_shards(); see block_push_shards_.
  absl::Status OnBlockShardsReceived(const std::vector<int>& block_ids,
                                     absl::Span<const int> shard_ids,
                                     uint64_t uuid = 0) override;

  absl::Status RegisterExpectedChunks(uint64_t uuid,
                                      uint32_t expected_chunks) override;
  virtual absl::Status RegisterExpectedChunksLocal(uint64_t uuid,
                                                   uint32_t expected_chunks);

  absl::Status RegisterExpectedLayerChunks(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks)
      override;
  virtual absl::Status RegisterExpectedLayerChunksLocal(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks);

  absl::Status OnLayerDataReceived(size_t layer_idx,
                                   uint64_t uuid = 0) override;
  absl::Status OnDataReceived(uint64_t uuid = 0) override;

  virtual absl::Status WaitForTransferCompletion(uint64_t uuid = 0);
  virtual void DrainPendingH2d();

  // Hooks around the device copy of a fully received transfer ("hooked H2D").
  //
  // TODO(justinlu): Remove these temporary hooks once H2D is invoked directly
  // by the controller instead of automatically upon transfer receipt.
  //
  // With auto_h2d=false the synchronizer normally only records a transfer as
  // complete once it has landed in host staging; nothing is written to device
  // until an explicit H2d(). Registering either hook changes that: once a whole
  // transfer (OnDataReceived) or every shard of a block push
  // (OnBlockShardsReceived) has landed, the synchronizer calls the start hook,
  // copies all layers to device and waits for the copy, calls the end hook with
  // the copy status, and only then records the transfer as complete, so
  // WaitForTransferCompletion() returning means device weights are updated.
  // Hooked H2Ds are serialized across concurrently completing transfers.
  //
  // The start hook must return only once it is safe to overwrite device
  // weights (e.g. no model execution in flight); the end hook is always called
  // if the start hook was, including on failure. Hooks have no effect with
  // auto_h2d=true and must be registered before any transfer starts.
  void RegisterH2dStartHook(absl::AnyInvocable<void()> hook);
  void RegisterH2dEndHook(absl::AnyInvocable<void(const absl::Status&)> hook);

  // Returns the device buffers that currently back the bound weights, in the
  // [layer][shard] form BindWeights() takes.
  using DeviceBufferResolver = WeightBuffers::DeviceBufferResolver;

  // Installs `resolver`, which is called to rebind the weights to the buffers
  // that currently back them before every D2h(), every controller-driven push
  // that runs a D2H, every H2d() and, with auto_h2d, whenever a receiver is
  // armed for a new uuid. Frameworks whose tensors move to a new device buffer
  // while keeping their identity (TorchTPU replaces a tensor's buffer on
  // in-place ops) install one, so that transfers never copy a buffer the
  // framework no longer uses. A resolver error fails the transfer, as does a
  // buffer whose size differs from the bound one. Must be called before any
  // transfer starts.
  void SetDeviceBufferResolver(DeviceBufferResolver resolver);

  void ForgetPushProgress(uint64_t uuid) override;

 protected:
  // Two-phase construction for Create() and subclasses: the constructors only
  // initialize the base class, thread pools and layer names and never fail;
  // the matching Init*() allocates host staging buffers and starts the
  // listener. An instance whose Init*() failed must be destroyed, not used.
  //
  // Device-buffer variant; follow with InitDeviceBuffers().
  WeightSynchronizerBase(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      std::optional<int> local_port, int parallelism,
      std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
      bool auto_h2d);
  // CPU-only variant; follow with InitHostBuffers().
  WeightSynchronizerBase(
      size_t num_layers, size_t num_shards,
      const std::vector<size_t>& slice_byte_sizes,
      std::optional<int> local_port, int parallelism,
      std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
      bool auto_h2d, std::vector<int> numa_nodes = {},
      std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses(),
      std::vector<transport::ShardInfo> shards = {});

  // Takes PJRT holds on `layer_buffers` and sets up one host staging buffer
  // per shard, adopting `external_host_ptrs` when provided. Starts the
  // listener on `listener_port` when set.
  absl::Status InitDeviceBuffers(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      const std::optional<std::vector<const uint8_t*>>& external_host_ptrs,
      std::optional<int> listener_port);
  // Allocates one host staging buffer per shard sized by `slice_byte_sizes`
  // (falling back to slice_byte_size() for missing entries). Starts the
  // listener on `listener_port` when set.
  absl::Status InitHostBuffers(const std::vector<size_t>& slice_byte_sizes,
                               std::optional<int> listener_port);

  // Records the shards of one block-push stream for `uuid` and returns true
  // once the shards reported so far reach num_shards(), i.e. the host buffers
  // of `uuid` are complete and may be copied to device. Subclasses that
  // override OnBlockShardsReceived() should gate their device copy on this.
  bool RecordBlockPushShards(absl::Span<const int> shard_ids, uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(block_push_shards_mu_);

  std::unique_ptr<WeightSynchronizerListener> listener_;
  // Opaque human-readable name labels mapped to logical layer index boundaries.
  std::vector<std::string> layer_names_;

  // Override parent AllocateBlocks as simple identity indices since we sync
  // entire buffers E2E!
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    std::vector<int> ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) ids[i] = static_cast<int>(i);
    return ids;
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

 private:
  // Shared tail of InitDeviceBuffers()/InitHostBuffers(): starts the listener
  // on `listener_port` when set, creates the auto-H2D pool, and records the
  // initial host-buffer occupancy metric.
  void FinishInit(std::optional<int> listener_port);

  // Runs the start hook, copies all layers of `uuid` to device via H2d() and
  // waits, then runs the end hook with the copy status, which is returned.
  // Hooked H2Ds never overlap; see WeightBuffers::HookedH2d().
  absl::Status HookedH2d(uint64_t uuid);

  std::vector<int64_t> global_shard_indices_;
  std::vector<int> local_shard_indices_;

  // When enabled, automatically schedules asynchronous device transfers (H2D)
  // upon complete host buffer writes.
  bool auto_h2d_ = false;

  struct PendingH2dState {
    // Number of layers with a non-zero expected chunk count, set by
    // RegisterExpectedLayerChunksLocal. OnDataReceived waits until
    // `layer_futures` holds this many entries.
    size_t expected_layers = 0;
    // Per-layer futures scheduled by OnLayerDataReceived, keyed by layer index.
    absl::flat_hash_map<size_t,
                        std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
        layer_futures;
    // Per-layer readiness signals created when the uuid was armed with a
    // non-deferred tiling mode, keyed by layer index. OnLayerDataReceived
    // takes each one and sets it once the layer is ready for a raw H2D;
    // signals still here when the uuid is dropped are cancelled.
    absl::flat_hash_map<size_t, std::shared_ptr<LayerTileReady>> tile_ready;
  };

  // Blocks until every expected layer of `uuid` has scheduled its future via
  // OnLayerDataReceived, then removes and returns those futures.
  absl::flat_hash_map<size_t,
                      std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
  TakeLayerFuturesWhenScheduled(uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(pending_h2d_mu_);

  // Returns the StartTransferRequest.host_tiling_mode the receiver was armed
  // with for `uuid`, normalized so that UNSPECIFIED becomes
  // HOST_TILING_MODE_DEFERRED.
  tpu_sync::rpc::HostTilingMode GetReceiverTiling(uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(receiver_tiling_mu_);

  std::unique_ptr<tpu_raiden::NumaThreadPool> h2d_pool_;
  absl_nonnull std::unique_ptr<tpu_raiden::NumaThreadPool> push_pool_;

  // A local shard and its schedule in a StartTransferRequest.
  struct ShardPushSchedule {
    size_t shard_idx;
    const tpu_sync::rpc::ShardPushScheduleProto& schedule;
  };
  // Returns the local shards that have an entry in `request`'s
  // shard_push_schedules (keyed by global or local shard index, or by sorted
  // key order for zero-based slicing), in ascending shard order. The
  // references point into `request`.
  absl::StatusOr<std::vector<ShardPushSchedule>> ResolveShardPushSchedules(
      const tpu_sync::rpc::StartTransferRequest& request);
  // Expands `request`'s shard_push_schedules into per-layer push tasks.
  absl::StatusOr<std::vector<std::vector<transport::BufferPushTask>>>
  BuildLayerPushTasks(const tpu_sync::rpc::StartTransferRequest& request);

  // Guards the receiver-side bookkeeping armed by StoreSkipTilingLocal().
  mutable absl::Mutex receiver_tiling_mu_;
  absl::flat_hash_map<uint64_t, std::string> uuid_to_req_id_
      ABSL_GUARDED_BY(receiver_tiling_mu_);
  // Receiver uuids armed with a non-default
  // StartTransferRequest.host_tiling_mode.
  absl::flat_hash_map<uint64_t, tpu_sync::rpc::HostTilingMode> receiver_tiling_
      ABSL_GUARDED_BY(receiver_tiling_mu_);

  struct UuidD2hState {
    explicit UuidD2hState(size_t num_layers) : layer_ready(num_layers, false) {}
    absl::Mutex mu;
    std::vector<bool> layer_ready ABSL_GUARDED_BY(mu);
    bool all_ready ABSL_GUARDED_BY(mu) = false;
    absl::Status status ABSL_GUARDED_BY(mu);
  };

  mutable absl::Mutex d2h_mu_;
  absl::flat_hash_set<uint64_t> completed_d2h_uuids_ ABSL_GUARDED_BY(d2h_mu_);
  absl::flat_hash_map<uint64_t, std::shared_ptr<UuidD2hState>> d2h_states_
      ABSL_GUARDED_BY(d2h_mu_);

  // Lock order: pending_h2d_mu_ before WeightBuffers' tile_ready_mu_ and
  // metrics_mu_ (SetLayerTileReady() / ResetLastTilingMetrics() are called
  // with it held).
  mutable absl::Mutex pending_h2d_mu_;
  absl::flat_hash_map<uint64_t, PendingH2dState> pending_h2d_states_
      ABSL_GUARDED_BY(pending_h2d_mu_);
  absl::flat_hash_set<uint64_t> active_h2d_uuids_
      ABSL_GUARDED_BY(pending_h2d_mu_);

  mutable absl::Mutex completed_transfers_mu_;
  absl::flat_hash_set<uint64_t> completed_transfers_
      ABSL_GUARDED_BY(completed_transfers_mu_);

  // Number of shards reported by OnBlockShardsReceived per block-push uuid. A
  // routed push delivers a block range over several streams that each carry a
  // shard subset; H2d(uuid) copies all shards, so it waits until the count
  // reaches num_shards(). Cleared by ForgetPushProgress.
  mutable absl::Mutex block_push_shards_mu_;
  absl::flat_hash_map<uint64_t, size_t> block_push_shards_
      ABSL_GUARDED_BY(block_push_shards_mu_);

  std::optional<size_t> pipeline_group_size_override_;

  // Memory and copy plane: device buffer holds, scratchpads, shard pool,
  // host-buffer layout state, copies and metrics. Never null; destroyed
  // explicitly by ~WeightSynchronizerBase() after `h2d_pool_` is joined.
  std::unique_ptr<WeightBuffers> buffers_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_

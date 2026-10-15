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

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
class HostMemoryAllocator;

namespace weight_sync {

struct WeightSyncMetrics {
  double last_d2h_time_ms = 0.0;
  double last_h2d_time_ms = 0.0;
  double last_h2h_time_ms = 0.0;
  double last_staging_time_ms = 0.0;
  double last_tiling_time_ms = 0.0;
  double last_detiling_time_ms = 0.0;
  double last_total_push_resharded_time_ms = 0.0;

  size_t last_d2h_bytes = 0;
  size_t last_h2d_bytes = 0;
  size_t last_h2h_bytes = 0;
  size_t last_tiled_bytes = 0;
  size_t last_detiled_bytes = 0;

  double total_d2h_time_ms = 0.0;
  double total_h2d_time_ms = 0.0;
  double total_h2h_time_ms = 0.0;
  double total_staging_time_ms = 0.0;
  double total_tiling_time_ms = 0.0;
  double total_detiling_time_ms = 0.0;
  double total_push_resharded_time_ms = 0.0;

  size_t total_d2h_bytes = 0;
  size_t total_h2d_bytes = 0;
  size_t total_h2h_bytes = 0;
  size_t total_tiled_bytes = 0;
  size_t total_detiled_bytes = 0;

  uint64_t d2h_call_count = 0;
  uint64_t push_resharded_call_count = 0;
};

class WeightSynchronizerListener;

class WeightSynchronizerBase : public tpu_raiden::RaidenManagerBase {
 public:
  // Creates a synchronizer wrapping the raw PJRT buffers in |layer_buffers|
  // (outer index = layer, inner index = shard) and allocates (or, when
  // |external_host_ptrs| is set, adopts) one host staging buffer per shard.
  // Returns InvalidArgument if the layers disagree on shard count or
  // |external_host_ptrs| has fewer entries than shards, and ResourceExhausted
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
  // mock E2E testing, with |slice_byte_size| bytes per layer. Returns
  // ResourceExhausted if a host staging buffer cannot be allocated.
  static absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> Create(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1, std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false);

  // As above, with heterogeneous per-layer |slice_byte_sizes| and custom
  // |layer_names|.
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

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override;
  const uint8_t* GetHostPointer(size_t layer_idx,
                                size_t shard_idx) const override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) const override;

  // Returns the tiling scratchpad of `shard_idx`, shared by all layers. The
  // pointer may be reallocated by a later H2D/D2H that needs more space.
  uint8_t* GetTiledPointer(size_t layer_idx, size_t shard_idx) {
    if (shard_idx < tiled_scratchpads_.size() &&
        tiled_scratchpads_[shard_idx]) {
      absl::MutexLock lock(tiled_scratchpads_[shard_idx]->mu);
      return tiled_scratchpads_[shard_idx]->ptr;
    }
    return nullptr;
  }
  const uint8_t* GetTiledPointer(size_t layer_idx, size_t shard_idx) const {
    if (shard_idx < tiled_scratchpads_.size() &&
        tiled_scratchpads_[shard_idx]) {
      absl::MutexLock lock(tiled_scratchpads_[shard_idx]->mu);
      return tiled_scratchpads_[shard_idx]->ptr;
    }
    return nullptr;
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
  // Tiles the host staging buffers of |layer_idx| into device layout and
  // copies the tiled bytes back into the same host buffers. No-op for shards
  // without tiled layouts or when skip_tiling is set for |layer_idx| under
  // |uuid|.
  absl::Status TileLayer(size_t layer_idx, uint64_t uuid = 0);
  virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> D2hLayer(
      size_t layer_idx, uint64_t uuid = 0,
      std::shared_ptr<std::atomic<double>> max_detile_ms = nullptr);

  // Binds new device buffers to the weight synchronizer in-place.
  //
  // This replaces the existing bound buffers with the new ones. The new buffers
  // must match the shape and layout configuration (number of layers, shards,
  // and size per shard) established at initialization.
  //
  // Calling this releases the holds on the previously bound buffers and
  // acquires holds on the new ones.
  //
  // Returns InvalidArgumentError if the number of layers, shards, or buffer
  // sizes do not match the initialized configuration.
  absl::Status BindWeights(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>&
          layer_buffers);

  // Unbinds all device buffers and releases all TPU HBM buffer holds while
  // keeping allocated host memory staging buffers, layer/shard metadata, and
  // listeners intact. A subsequent BindWeights() re-arms the synchronizer
  // without re-allocating host memory or re-registering with the controller.
  //
  // After this call, D2h()/H2d() (and any controller-driven push with
  // skip_d2h=false) return FailedPrecondition until BindWeights() is called
  // again. BindWeights({}) is an alias for UnbindWeights().
  //
  // Thread-safety: buffer_holds_ is not mutex-protected. Callers must not
  // invoke this while a D2H/H2D/push on this synchronizer is in flight (e.g.
  // call it only after the controller's start_transfer future has resolved).
  // DMA copies already scheduled remain safe: their futures own their own
  // buffer holds.
  virtual void UnbindWeights();

  virtual void StoreSkipTiling(
      uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request);
  virtual void StoreSkipTilingLocal(
      uint64_t uuid, const tpu_sync::rpc::StartTransferRequest& request);

  virtual void SetSkipTiling(const std::vector<bool>& skip_tiling) {
    absl::MutexLock lock(skip_tiling_mu_);
    latest_skip_tiling_ = skip_tiling;
  }
  virtual void SetSkipTiling(bool skip_all) {
    absl::MutexLock lock(skip_tiling_mu_);
    latest_skip_tiling_.assign(num_layers_, skip_all);
  }

  virtual WeightSyncMetrics GetMetrics() const {
    absl::MutexLock lock(metrics_mu_);
    return metrics_;
  }

  virtual void ResetMetrics() {
    absl::MutexLock lock(metrics_mu_);
    metrics_ = WeightSyncMetrics{};
    if (active_h2h_pushes_ > 0) {
      h2h_active_window_start_ = absl::Now();
    }
  }

  void SetMetricsForTesting(const WeightSyncMetrics& m) {
    absl::MutexLock lock(metrics_mu_);
    metrics_ = m;
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

  // Block-push completion. Counts the shards reported for |uuid| and runs the
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
  void RegisterH2dStartHook(absl::AnyInvocable<void()> hook)
      ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);
  void RegisterH2dEndHook(absl::AnyInvocable<void(const absl::Status&)> hook)
      ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);

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

  // Takes PJRT holds on |layer_buffers| and sets up one host staging buffer
  // per shard, adopting |external_host_ptrs| when provided. Starts the
  // listener on |listener_port| when set.
  absl::Status InitDeviceBuffers(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      const std::optional<std::vector<const uint8_t*>>& external_host_ptrs,
      std::optional<int> listener_port);
  // Allocates one host staging buffer per shard sized by |slice_byte_sizes|
  // (falling back to slice_byte_size() for missing entries). Starts the
  // listener on |listener_port| when set.
  absl::Status InitHostBuffers(const std::vector<size_t>& slice_byte_sizes,
                               std::optional<int> listener_port);

  // Records the shards of one block-push stream for |uuid| and returns true
  // once the shards reported so far reach num_shards(), i.e. the host buffers
  // of |uuid| are complete and may be copied to device. Subclasses that
  // override OnBlockShardsReceived() should gate their device copy on this.
  bool RecordBlockPushShards(absl::Span<const int> shard_ids, uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(block_push_shards_mu_);

  std::unique_ptr<WeightSynchronizerListener> listener_;
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;
  // Opaque human-readable name labels mapped to logical layer index boundaries.
  std::vector<std::string> layer_names_;

  // Separate PJRT active holds matrix E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_;

  // Set by UnbindWeights() and cleared by BindWeights(). While set, D2h()/H2d()
  // return FailedPrecondition instead of silently no-op'ing on the (now empty)
  // buffer_holds_, so a missing re-bind cannot ship stale host staging data.
  // CPU-only instances never set this flag and keep their no-op behavior.
  bool weights_unbound_ = false;

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
  // on |listener_port| when set, creates the auto-H2D pool and per-shard
  // scratchpads, and records the initial host-buffer occupancy metric.
  void FinishInit(std::optional<int> listener_port);

  // True if RegisterH2dStartHook() or RegisterH2dEndHook() has been called.
  bool HasH2dHooks() const ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);
  // Runs the start hook, copies all layers of |uuid| to device and waits,
  // then runs the end hook with the copy status, which is returned. Holds
  // |h2d_hook_mu_| throughout so hooked H2Ds never overlap.
  absl::Status HookedH2d(uint64_t uuid) ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);

  // Guards the hooks and serializes hooked H2Ds. The destructor stops the
  // listener and transport server (waiting for in-flight callbacks) before any
  // member is destroyed, so a running HookedH2d() always finishes first.
  mutable absl::Mutex h2d_hook_mu_;
  absl::AnyInvocable<void()> h2d_start_hook_ ABSL_GUARDED_BY(h2d_hook_mu_);
  absl::AnyInvocable<void(const absl::Status&)> h2d_end_hook_
      ABSL_GUARDED_BY(h2d_hook_mu_);
  std::vector<int64_t> global_shard_indices_;
  std::vector<int> local_shard_indices_;

  // When enabled, automatically schedules asynchronous device transfers (H2D)
  // upon complete host buffer writes.
  bool auto_h2d_ = false;

  // Signals that a received layer's host buffers hold device-layout bytes,
  // i.e. the layer has arrived and, if needed, been tiled in place.
  // Thread-safe.
  class LayerTileReady {
   public:
    // Records |status| and wakes all waiters. Calls after the first are
    // ignored.
    void Set(absl::Status status) ABSL_LOCKS_EXCLUDED(mu_);
    // Blocks until Set() has been called and returns the recorded status.
    absl::Status Wait() ABSL_LOCKS_EXCLUDED(mu_);
    // Returns true if Set() has been called.
    bool IsSet() ABSL_LOCKS_EXCLUDED(mu_);

   private:
    absl::Mutex mu_;
    bool done_ ABSL_GUARDED_BY(mu_) = false;
    absl::Status status_ ABSL_GUARDED_BY(mu_);
  };

  struct PendingH2dState {
    // Number of layers with a non-zero expected chunk count, set by
    // RegisterExpectedLayerChunksLocal. OnDataReceived waits until
    // |layer_futures| holds this many entries.
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

  // Blocks until every expected layer of |uuid| has scheduled its future via
  // OnLayerDataReceived, then removes and returns those futures.
  absl::flat_hash_map<size_t,
                      std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
  TakeLayerFuturesWhenScheduled(uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(pending_h2d_mu_);

  // Returns true if skip_tiling is set for |layer_idx| under |uuid|, falling
  // back to the latest stored skip_tiling when |uuid| is unknown.
  bool IsTilingSkipped(size_t layer_idx, uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Returns the StartTransferRequest.host_tiling_mode the receiver was armed
  // with for |uuid|, normalized so that UNSPECIFIED becomes
  // HOST_TILING_MODE_DEFERRED.
  tpu_sync::rpc::HostTilingMode GetReceiverTiling(uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Sets the in-place tiling signal of |layer_idx|; nullptr marks the layer's
  // host buffers as holding logical-layout bytes.
  void SetLayerTileReady(size_t layer_idx,
                         std::shared_ptr<LayerTileReady> ready)
      ABSL_LOCKS_EXCLUDED(tile_ready_mu_);
  // Marks the host staging buffers of all layers as holding logical-layout
  // bytes, so that the next H2d() tiles them.
  void MarkHostBufferUntiled() ABSL_LOCKS_EXCLUDED(tile_ready_mu_);

  std::unique_ptr<tpu_raiden::NumaThreadPool> h2d_pool_;
  absl_nonnull std::unique_ptr<tpu_raiden::NumaThreadPool> push_pool_;
  // Runs ForEachShardNumaLocal()'s per-shard tasks (tiling, H2D/D2H issue).
  // Separate from |h2d_pool_| so that H2d() never waits on tasks queued behind
  // its own caller. It has one thread per shard, so a task on it must never
  // block on another task of this pool (e.g. H2d() waiting for in-place tiling
  // done by TileLayer()).
  absl_nonnull std::unique_ptr<tpu_raiden::NumaThreadPool> shard_pool_;
  std::unique_ptr<HostMemoryAllocator> host_allocator_;

  // Shared reusable scratchpad per shard (one per local device/chip) to avoid
  // allocating redundant tiled staging buffers across all layers.
  struct ShardScratchpad {
    absl::Mutex mu;
    uint8_t* ptr ABSL_GUARDED_BY(mu) = nullptr;
    size_t capacity ABSL_GUARDED_BY(mu) = 0;
    std::shared_ptr<void> owner ABSL_GUARDED_BY(mu);
    std::unique_ptr<uint8_t[], void (*)(void*)> owned_buffer ABSL_GUARDED_BY(
        mu) = {nullptr, [](void*) {}};
    // Copy that still reads (H2D) or writes (D2H) |ptr|. Awaited before the
    // buffer is reused or reallocated.
    xla::Future<> in_flight_future ABSL_GUARDED_BY(mu);
  };
  std::vector<std::unique_ptr<ShardScratchpad>> tiled_scratchpads_;

  absl::StatusOr<uint8_t*> AcquireTiledScratchpadLocked(
      ShardScratchpad& sp, size_t required_bytes,
      const xla::PjRtDevice* device) ABSL_EXCLUSIVE_LOCKS_REQUIRED(sp.mu);

  // Returns the per-layer skip-tiling mask for |uuid|, falling back to the
  // latest mask set via SetSkipTiling() and then to all-false.
  std::vector<bool> GetActiveSkipTiling(uint64_t uuid);

  // Starts the host-to-device copy of shard |shard_idx| of layer |layer_idx|.
  // If |tile| is true and the shard has a tiled layout, first tiles the host
  // bytes into the shard's scratchpad and copies that; otherwise copies the
  // host bytes as is. Tiling runs on the calling thread before this returns;
  // the copy is asynchronous. Safe to call concurrently for distinct shards;
  // calls for the same shard serialize on its scratchpad.
  absl::StatusOr<xla::Future<raiden::BufferHolder>> H2dShard(size_t layer_idx,
                                                             size_t shard_idx,
                                                             bool tile);
  // Copies all shards of |layer_idx| to device via H2dShard().
  absl::StatusOr<raiden::PjRtCopyFuture> H2dLayer(size_t layer_idx, bool tile);
  // Starts the device-to-host copy of shard |shard_idx| of layer |layer_idx|
  // into its host staging buffer. If the shard has a tiled layout and
  // |skip_tiling| is false, copies into the shard's scratchpad and detiles into
  // the staging buffer once the copy completes, recording the detile time in
  // |max_detile_ms| when non-null. Errors are returned as a failed future.
  xla::Future<raiden::BufferHolder> D2hShard(
      size_t layer_idx, size_t shard_idx, bool skip_tiling,
      std::shared_ptr<std::atomic<double>> max_detile_ms);
  // A local shard and its schedule in a StartTransferRequest.
  struct ShardPushSchedule {
    size_t shard_idx;
    const tpu_sync::rpc::ShardPushScheduleProto& schedule;
  };
  // Returns the local shards that have an entry in |request|'s
  // shard_push_schedules (keyed by global or local shard index, or by sorted
  // key order for zero-based slicing), in ascending shard order. The
  // references point into |request|.
  absl::StatusOr<std::vector<ShardPushSchedule>> ResolveShardPushSchedules(
      const tpu_sync::rpc::StartTransferRequest& request);
  // Expands |request|'s shard_push_schedules into per-layer push tasks.
  absl::StatusOr<std::vector<std::vector<transport::BufferPushTask>>>
  BuildLayerPushTasks(const tpu_sync::rpc::StartTransferRequest& request);
  // Returns the scratchpad of shard |shard_idx|.
  absl::StatusOr<ShardScratchpad*> GetTiledScratchpad(size_t shard_idx);
  // Tiles the host bytes of shard |shard_idx| of layer |layer_idx| into |sp|
  // and records tiling metrics. Returns the tiled bytes, valid while |sp.mu|
  // is held.
  absl::StatusOr<absl::Span<const uint8_t>> TileShardLocked(size_t layer_idx,
                                                            size_t shard_idx,
                                                            ShardScratchpad& sp)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(sp.mu);

  // Per-layer in-place tiling state of the host staging buffers, indexed by
  // layer. A non-null entry means the layer's host buffers hold device-layout
  // bytes once the signal is set; null means they hold logical-layout bytes.
  // Lock order: pending_h2d_mu_ before tile_ready_mu_.
  absl::Mutex tile_ready_mu_;
  std::vector<std::shared_ptr<LayerTileReady>> layer_tile_ready_
      ABSL_GUARDED_BY(tile_ready_mu_);

  mutable absl::Mutex skip_tiling_mu_;
  absl::flat_hash_map<uint64_t, std::vector<bool>> uuid_to_skip_tiling_
      ABSL_GUARDED_BY(skip_tiling_mu_);
  std::vector<bool> latest_skip_tiling_ ABSL_GUARDED_BY(skip_tiling_mu_);
  absl::flat_hash_map<uint64_t, std::string> uuid_to_req_id_
      ABSL_GUARDED_BY(skip_tiling_mu_);
  // Receiver uuids armed with a non-default
  // StartTransferRequest.host_tiling_mode.
  absl::flat_hash_map<uint64_t, tpu_sync::rpc::HostTilingMode> receiver_tiling_
      ABSL_GUARDED_BY(skip_tiling_mu_);

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

  mutable absl::Mutex pending_h2d_mu_;
  absl::flat_hash_map<uint64_t, PendingH2dState> pending_h2d_states_
      ABSL_GUARDED_BY(pending_h2d_mu_);
  absl::flat_hash_set<uint64_t> active_h2d_uuids_
      ABSL_GUARDED_BY(pending_h2d_mu_);

  mutable absl::Mutex metrics_mu_;
  WeightSyncMetrics metrics_ ABSL_GUARDED_BY(metrics_mu_);
  size_t active_h2h_pushes_ ABSL_GUARDED_BY(metrics_mu_) = 0;
  absl::Time h2h_active_window_start_ ABSL_GUARDED_BY(metrics_mu_) =
      absl::InfinitePast();

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
  void UpdateAllocatedOccupancyMetric(size_t delta = 0);
  std::atomic<size_t> allocated_host_dram_bytes_{0};
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_BUFFERS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_BUFFERS_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/core/tpu_utils.h"

namespace xla {
class PjRtDevice;
}  // namespace xla

namespace tpu_raiden {
class HostMemoryAllocator;

namespace weight_sync {

struct WeightSyncMetrics {
  // `last_*` fields describe the most recently completed operation and are
  // overwritten by the next one; `total_*` fields accumulate over all
  // operations since the last ResetMetrics().

  // Duration of the most recent completed D2H: from issuing the copy of the
  // first layer until every layer of every shard is in host staging (including
  // host detiling). Recorded by D2h() and by PushWeightsResharded() when it
  // performs the D2H.
  double last_d2h_time_ms = 0.0;
  // Duration of the most recent completed H2D: from H2d() (or the auto-H2D of
  // a received transfer) until every copied layer is on device (including host
  // tiling).
  double last_h2d_time_ms = 0.0;
  double last_h2h_time_ms = 0.0;
  double last_staging_time_ms = 0.0;
  double last_tiling_time_ms = 0.0;
  double last_detiling_time_ms = 0.0;
  double last_total_push_resharded_time_ms = 0.0;

  // D2H/H2D bytes are the on-device bytes copied, not the (larger) capacity of
  // the host staging buffers.
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

// Signals that a layer's host buffers hold the bytes a reader expects, e.g. a
// received layer has arrived and, if needed, been tiled in place, or an
// explicit D2H of the layer has landed. Thread-safe.
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

// Memory and copy plane of a weight synchronizer: owns the PJRT device buffer
// holds of the bound weights, the per-shard tiling scratchpads, the shard
// thread pool, and the state that tracks the layout and readiness of the host
// staging buffers, and performs the host<->device copies (with host tiling)
// between them. The host staging buffers themselves live in the
// RaidenManagerBase's StagingArena passed to the constructor.
//
// Thread-safe. Every method acquires only WeightBuffers' own mutexes, which
// are ordered: h2d_hook_mu_ (held by HookedH2d() for the whole copy) before
// restage_mu_ before d2h_op_mu_ before device_buffer_resolver_mu_ and the
// leaf mutexes (explicit_d2h_mu_, buffer_holds_mu_, tile_ready_mu_,
// metrics_mu_, skip_tiling_mu_ and the per-shard scratchpad mutexes). Callers
// may hold their own mutexes across any method that does not block on a copy
// (e.g. SetLayerTileReady(), ResetLastTilingMetrics()), as long as they never
// acquire them from a callback run by WeightBuffers.
class WeightBuffers {
 public:
  // Returns the device buffers that currently back the bound weights, in the
  // [layer][shard] form BindWeights() takes.
  using DeviceBufferResolver = absl::AnyInvocable<
      absl::StatusOr<std::vector<std::vector<raiden::RaidenBufferHandle>>>()>;

  // Result of ScheduleAllLayerD2h().
  struct ScheduledD2h {
    // First error hit while scheduling; the layers after the failing one were
    // not scheduled.
    absl::Status status;
    // Copies of the layers scheduled before any error, indexed by layer.
    std::vector<raiden::PjRtCopyFuture> layer_futures;
    // Whether bound device buffers existed when the copies were scheduled.
    bool has_device_buffers = false;
  };

  // Per-layer state recorded by the last explicit D2h() (readiness) and by
  // every D2H that staged a layer (skip_tiling layout), indexed by layer. A
  // null `ready` entry or an empty `skip_tiling` entry means no D2H has
  // staged the layer since it was last cleared.
  struct ExplicitD2hSnapshot {
    std::vector<std::shared_ptr<LayerTileReady>> ready;
    std::vector<std::optional<bool>> skip_tiling;
  };

  // Statistics of one completed resharded push, see FinishH2hPush().
  struct H2hPushStats {
    absl::Time h2h_end;
    size_t h2h_bytes = 0;
    double h2h_time_ms = 0.0;
    double staging_time_ms = 0.0;
    double total_push_time_ms = 0.0;
    // Whether the push performed the D2H, i.e. `d2h_time_ms` and `d2h_bytes`
    // are recorded as the last D2H.
    bool performed_d2h = false;
    double d2h_time_ms = 0.0;
    size_t d2h_bytes = 0;
  };

  // Creates the buffers of |manager|'s layers and shards, with one tiling
  // scratchpad per shard and no device buffers bound. |manager| owns the host
  // staging buffers and must outlive this object; |layer_names| label layers
  // in logs.
  WeightBuffers(RaidenManagerBase& manager,
                std::vector<std::string> layer_names);

  // Waits for in-flight D2H copies, joins the shard pool, waits for copies
  // still using the scratchpads, and removes this object's host allocations
  // from the allocated-host-DRAM metric.
  ~WeightBuffers();

  WeightBuffers(const WeightBuffers&) = delete;
  WeightBuffers& operator=(const WeightBuffers&) = delete;

  // ---------------------------------------------------------------------------
  // Initialization. Not thread-safe; call before any copy or transfer starts.
  // ---------------------------------------------------------------------------

  // Records the PJRT API and device size of |first_handle| and, when it has a
  // PJRT client and no host allocator exists yet, creates the DMA-mapped host
  // allocator used for scratchpads (and by the owner for staging buffers).
  void AdoptDeviceInfo(const raiden::RaidenBufferHandle& first_handle);
  // Records |physical_size| for CPU-only instances without device buffers.
  void SetPhysicalSize(size_t physical_size);
  // Returns the DMA-mapped host allocator, or null if none could be created.
  HostMemoryAllocator* absl_nullable host_allocator() const {
    return host_allocator_.get();
  }
  // Takes the initial device buffer |holds| ([layer][shard]).
  void SetBufferHolds(
      std::vector<std::vector<raiden::BufferHoldAndAlias>> holds)
      ABSL_LOCKS_EXCLUDED(buffer_holds_mu_);
  // Adds |delta| bytes to this object's and the process-wide allocated host
  // DRAM and publishes the process-wide total. Thread-safe.
  void UpdateAllocatedOccupancyMetric(size_t delta = 0);

  // ---------------------------------------------------------------------------
  // Device buffers.
  // ---------------------------------------------------------------------------

  // Replaces bound device buffers in-place (matching initial layer/shard/size)
  // with |layer_buffers|. Drains in-flight D2H first, fails if H2H is sending,
  // and runs |on_rebind| atomically under `device_buffer_resolver_mu_`. Empty
  // |layer_buffers| aliases UnbindWeights().
  absl::Status BindWeights(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      absl::AnyInvocable<void()> on_rebind = nullptr)
      ABSL_LOCKS_EXCLUDED(d2h_op_mu_, device_buffer_resolver_mu_);
  // Releases all device buffer holds; copies fail with FailedPrecondition
  // until the next BindWeights().
  void UnbindWeights() ABSL_LOCKS_EXCLUDED(buffer_holds_mu_);
  // Installs |resolver|, see WeightSynchronizerBase::SetDeviceBufferResolver.
  void SetDeviceBufferResolver(DeviceBufferResolver resolver)
      ABSL_LOCKS_EXCLUDED(device_buffer_resolver_mu_);
  // Rebinds the weights to the buffers the resolver returns. A no-op without a
  // resolver or while the weights are unbound.
  absl::Status RefreshDeviceBuffers()
      ABSL_LOCKS_EXCLUDED(device_buffer_resolver_mu_);

  // ---------------------------------------------------------------------------
  // Copies and tiling.
  // ---------------------------------------------------------------------------

  // Copies the host staging buffers of all layers to device, see
  // WeightSynchronizerBase::H2d(). |uuid| selects the skip-tiling mask. Must
  // not be called from a shard pool task.
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid);
  // Copies all shards of |layer_idx| to device, tiling them into the
  // scratchpads first when |tile| is true and the shard has a tiled layout.
  absl::StatusOr<raiden::PjRtCopyFuture> H2dLayer(size_t layer_idx, bool tile)
      ABSL_LOCKS_EXCLUDED(buffer_holds_mu_);
  // Tiles the host staging buffers of |layer_idx| into device layout in
  // place, see WeightSynchronizerBase::TileLayer(). Blocks until done; must
  // not be called from a shard pool task.
  absl::Status TileLayer(size_t layer_idx, uint64_t uuid);
  // Copies bound device buffers into host staging buffers asynchronously and
  // records the per-layer explicit-D2H readiness and layout, see
  // WeightSynchronizerBase::D2h(). Serializes behind in-flight D2H on
  // `d2h_op_mu_`.
  absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(d2h_op_mu_);
  // Copies all shards of |layer_idx| to host staging using the skip-tiling
  // mask of |uuid|, recording the detile time in |max_detile_ms| when
  // non-null. Does not record explicit-D2H state.
  absl::StatusOr<raiden::PjRtCopyFuture> D2hLayer(
      size_t layer_idx, uint64_t uuid,
      std::shared_ptr<std::atomic<double>> max_detile_ms = nullptr);
  // Returns the on-device bytes of all shards of |layer_idx|, i.e. what one
  // D2H or H2D of the layer copies.
  size_t LayerDeviceBytes(size_t layer_idx) const;
  // Returns the tiling scratchpad of |shard_idx|, shared by all layers, or
  // null. The pointer may be reallocated by a later copy that needs more
  // space.
  uint8_t* GetTiledPointer(size_t shard_idx);
  const uint8_t* GetTiledPointer(size_t shard_idx) const;

  // ---------------------------------------------------------------------------
  // Skip tiling.
  // ---------------------------------------------------------------------------

  // Sets the latest skip-tiling mask to |skip_tiling| (indexed by layer).
  void SetSkipTiling(const std::vector<bool>& skip_tiling)
      ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Sets the latest skip-tiling mask of every layer to |skip_all|.
  void SetSkipTiling(bool skip_all) ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Records |skip_tiling| as the mask of |uuid| and as the latest mask.
  void StoreSkipTiling(uint64_t uuid, std::vector<bool> skip_tiling)
      ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Drops the mask recorded for |uuid|.
  void ForgetSkipTiling(uint64_t uuid) ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Returns the per-layer skip-tiling mask for |uuid|, falling back to the
  // latest mask and then to all-false.
  std::vector<bool> GetActiveSkipTiling(uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);
  // Returns true if skip_tiling is set for |layer_idx| under |uuid|, falling
  // back to the latest stored mask when |uuid| is unknown.
  bool IsTilingSkipped(size_t layer_idx, uint64_t uuid)
      ABSL_LOCKS_EXCLUDED(skip_tiling_mu_);

  // ---------------------------------------------------------------------------
  // Host layer readiness (in-place tiling state of the staging buffers).
  // ---------------------------------------------------------------------------

  // Sets the in-place tiling signal of |layer_idx| to |ready|; nullptr marks
  // the layer's host buffers as holding logical-layout bytes. Acquires only
  // `tile_ready_mu_`.
  void SetLayerTileReady(size_t layer_idx,
                         std::shared_ptr<LayerTileReady> ready)
      ABSL_LOCKS_EXCLUDED(tile_ready_mu_);
  // Marks the host staging buffers of all layers as holding logical-layout
  // bytes, so that the next H2d() tiles them.
  void MarkHostBufferUntiled() ABSL_LOCKS_EXCLUDED(tile_ready_mu_);
  // Returns the in-place tiling signals of all layers, indexed by layer
  // (num_layers() entries; null for logical-layout layers).
  std::vector<std::shared_ptr<LayerTileReady>> SnapshotLayerTileReady()
      ABSL_LOCKS_EXCLUDED(tile_ready_mu_);

  // ---------------------------------------------------------------------------
  // Explicit D2H state.
  // ---------------------------------------------------------------------------

  // Waits for any in-flight explicit D2h() on |layer_idx| and returns the host
  // staging buffer pointer of (|layer_idx|, |shard_idx|).
  absl::StatusOr<const uint8_t*> WaitAndGetHostBufferPtr(size_t layer_idx,
                                                         size_t shard_idx) const
      ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  // Waits for any in-flight explicit D2h() on |layer_idx| and returns its
  // status.
  absl::Status WaitForExplicitD2h(size_t layer_idx) const
      ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  // Serializes behind D2H scheduling on `d2h_op_mu_` and waits for every
  // in-flight D2H copy to settle.
  void DrainInFlightD2h() ABSL_LOCKS_EXCLUDED(d2h_op_mu_, explicit_d2h_mu_);
  // Forgets the explicit-D2H readiness and layout of all layers.
  void ClearExplicitD2hState() ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  // Forgets the explicit-D2H readiness and layout of |layer_idx|.
  void ClearExplicitD2hLayer(size_t layer_idx)
      ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  // Returns a copy of the explicit-D2H state, resized to num_layers().
  ExplicitD2hSnapshot SnapshotExplicitD2h() const
      ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  // Schedules the D2H of every layer for a push, with |skip_tiling| (indexed
  // by layer) selecting each layer's layout. Under `d2h_op_mu_`: waits for
  // in-flight D2H, refreshes the device buffers, forgets the explicit-D2H
  // state, schedules each layer (recording its layout when device buffers are
  // bound), and tracks the scheduled copies as in flight, also when a layer
  // fails to schedule. Does not wait for the copies.
  ScheduledD2h ScheduleAllLayerD2h(const std::vector<bool>& skip_tiling)
      ABSL_LOCKS_EXCLUDED(restage_mu_, d2h_op_mu_);
  // Re-stages |layer_idx| with |planned_skip| when the layout recorded for it
  // by an earlier D2H differs, and waits for the copy. Concurrent calls
  // serialize on `restage_mu_`. Refreshes the device buffers first unless
  // |refreshed_device_buffers| is set, and sets it once they were refreshed.
  // Returns FailedPrecondition naming |uuid| if the layer cannot be re-staged
  // because the device weights are unbound.
  absl::Status RestageLayerIfMismatched(size_t layer_idx, bool planned_skip,
                                        uint64_t uuid,
                                        bool& refreshed_device_buffers)
      ABSL_LOCKS_EXCLUDED(restage_mu_, d2h_op_mu_);

  // ---------------------------------------------------------------------------
  // Metrics.
  // ---------------------------------------------------------------------------

  WeightSyncMetrics GetMetrics() const ABSL_LOCKS_EXCLUDED(metrics_mu_);
  // Resets all metrics; restarts the H2H window if a push is active.
  void ResetMetrics() ABSL_LOCKS_EXCLUDED(metrics_mu_);
  void SetMetricsForTesting(const WeightSyncMetrics& m)
      ABSL_LOCKS_EXCLUDED(metrics_mu_);
  // Resets the last tiling time and bytes, e.g. when a new transfer starts.
  // Acquires only `metrics_mu_`.
  void ResetLastTilingMetrics() ABSL_LOCKS_EXCLUDED(metrics_mu_);
  // Records an H2D of |h2d_bytes| that took |h2d_time_ms| and returns the
  // last tiling time.
  double RecordH2d(double h2d_time_ms, size_t h2d_bytes)
      ABSL_LOCKS_EXCLUDED(metrics_mu_);
  // Opens an H2H push window at |start|: while any push is open, D2h() and
  // BindWeights() fail since the push reads the host staging buffers. Every
  // call must be followed by exactly one AbortH2hPush() or FinishH2hPush().
  void BeginH2hPush(absl::Time start) ABSL_LOCKS_EXCLUDED(metrics_mu_);
  // Closes a push window opened by BeginH2hPush() without recording stats.
  void AbortH2hPush() ABSL_LOCKS_EXCLUDED(metrics_mu_);
  // Closes a push window opened by BeginH2hPush() and records |stats|.
  // Returns the last detiling time.
  double FinishH2hPush(const H2hPushStats& stats)
      ABSL_LOCKS_EXCLUDED(metrics_mu_);

  // ---------------------------------------------------------------------------
  // H2D hooks, see WeightSynchronizerBase::RegisterH2dStartHook().
  // ---------------------------------------------------------------------------

  void RegisterH2dStartHook(absl::AnyInvocable<void()> hook)
      ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);
  void RegisterH2dEndHook(absl::AnyInvocable<void(const absl::Status&)> hook)
      ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);
  // True if RegisterH2dStartHook() or RegisterH2dEndHook() has been called.
  bool HasH2dHooks() const ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);
  // Runs the start hook, then |copy_to_device| (which copies all layers of
  // |uuid| to device and waits), then the end hook with the copy status,
  // which is returned. Holds `h2d_hook_mu_` throughout so hooked H2Ds never
  // overlap.
  absl::Status HookedH2d(uint64_t uuid,
                         absl::FunctionRef<absl::Status()> copy_to_device)
      ABSL_LOCKS_EXCLUDED(h2d_hook_mu_);

 private:
  // Shared reusable scratchpad per shard (one per local device/chip) to avoid
  // allocating redundant tiled staging buffers across all layers.
  struct ShardScratchpad {
    absl::Mutex mu;
    uint8_t* ptr ABSL_GUARDED_BY(mu) = nullptr;
    size_t capacity ABSL_GUARDED_BY(mu) = 0;
    std::shared_ptr<void> owner ABSL_GUARDED_BY(mu);
    std::unique_ptr<uint8_t[], void (*)(void*)> owned_buffer
        ABSL_GUARDED_BY(mu) = {nullptr, [](void*) {}};
    // Copy that still reads (H2D) or writes (D2H) `ptr`. Awaited before the
    // buffer is reused or reallocated.
    xla::Future<> in_flight_future ABSL_GUARDED_BY(mu);
  };

  StagingArena& layers() const { return manager_.staging_arena(); }

  // Returns InvalidArgument unless `layer_buffers` has the bound number of
  // layers and shards and every buffer has the bound shard's device size.
  absl::Status CheckDeviceBuffersMatch(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers)
      const;
  // Replaces buffer_holds_ with `layer_buffers`, which must have passed
  // CheckDeviceBuffersMatch(). When `allow_from_unbound` is false (refreshing
  // already-bound weights), leaves an intervening UnbindWeights() untouched.
  void ReplaceBufferHolds(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      bool allow_from_unbound) ABSL_LOCKS_EXCLUDED(buffer_holds_mu_);

  absl::StatusOr<uint8_t*> AcquireTiledScratchpadLocked(
      ShardScratchpad& sp, size_t required_bytes, const xla::PjRtDevice* device)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(sp.mu);
  // Returns the scratchpad of shard `shard_idx`.
  absl::StatusOr<ShardScratchpad*> GetTiledScratchpad(size_t shard_idx);
  // Tiles the host bytes of shard `shard_idx` of layer `layer_idx` into `sp`
  // using `shard_hold` and records tiling metrics. Returns the tiled bytes,
  // valid while `sp.mu` is held.
  absl::StatusOr<absl::Span<const uint8_t>> TileShardLocked(
      size_t layer_idx, size_t shard_idx,
      const raiden::BufferHoldAndAlias& shard_hold, ShardScratchpad& sp)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(sp.mu);

  // Starts the host-to-device copy of shard `shard_idx` of layer `layer_idx`.
  // If `tile` is true and the shard has a tiled layout, first tiles the host
  // bytes into the shard's scratchpad and copies that; otherwise copies the
  // host bytes as is. Tiling runs on the calling thread before this returns;
  // the copy is asynchronous. Safe to call concurrently for distinct shards;
  // calls for the same shard serialize on its scratchpad.
  absl::StatusOr<xla::Future<raiden::BufferHolder>> H2dShard(size_t layer_idx,
                                                             size_t shard_idx,
                                                             bool tile);
  // Starts the device-to-host copy of shard `shard_idx` of layer `layer_idx`
  // into its host staging buffer. If the shard has a tiled layout and
  // `skip_tiling` is false, copies into the shard's scratchpad and detiles into
  // the staging buffer once the copy completes, recording the detile time in
  // `max_detile_ms` when non-null. Releases the shard's device buffer hold as
  // soon as the copy+detile completes. Errors are returned as a failed future.
  xla::Future<> D2hShard(size_t layer_idx, size_t shard_idx, bool skip_tiling,
                         std::shared_ptr<std::atomic<double>> max_detile_ms)
      ABSL_LOCKS_EXCLUDED(buffer_holds_mu_);
  absl::StatusOr<raiden::PjRtCopyFuture> D2hLayerWithSkipTiling(
      size_t layer_idx, bool skip_tiling,
      std::shared_ptr<std::atomic<double>> max_detile_ms = nullptr)
      ABSL_LOCKS_EXCLUDED(buffer_holds_mu_);
  void TrackInFlightD2h(xla::Future<> future)
      ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  void WaitForInFlightD2h() const ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  absl::Status WaitForAllExplicitD2h() const
      ABSL_LOCKS_EXCLUDED(explicit_d2h_mu_);
  absl::Status CheckNoActiveH2hPushes(absl::string_view caller) const
      ABSL_LOCKS_EXCLUDED(metrics_mu_);

  // Non-owning; owns the host staging buffers and outlives this object.
  RaidenManagerBase& manager_;
  const size_t num_layers_;
  const size_t num_shards_;
  // Opaque human-readable name labels mapped to logical layer indices, used
  // in logs.
  const std::vector<std::string> layer_names_;

  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;

  // Leaf mutex guarding `buffer_holds_` and `weights_unbound_`.
  mutable absl::Mutex buffer_holds_mu_;
  // Separate PJRT active holds matrix E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_
      ABSL_GUARDED_BY(buffer_holds_mu_);

  // Set by UnbindWeights() and cleared by BindWeights(). While set, D2h()/H2d()
  // return FailedPrecondition instead of silently no-op'ing on the (now empty)
  // buffer_holds_, so a missing re-bind cannot ship stale host staging data.
  // CPU-only instances never set this flag and keep their no-op behavior.
  bool weights_unbound_ ABSL_GUARDED_BY(buffer_holds_mu_) = false;

  // Guards the resolver and serializes RefreshDeviceBuffers() calls.
  absl::Mutex device_buffer_resolver_mu_;
  DeviceBufferResolver device_buffer_resolver_
      ABSL_GUARDED_BY(device_buffer_resolver_mu_);

  // Guards the hooks and serializes hooked H2Ds. The owner stops the
  // listener and transport server (waiting for in-flight callbacks) before
  // destroying this object, so a running HookedH2d() always finishes first.
  mutable absl::Mutex h2d_hook_mu_;
  absl::AnyInvocable<void()> h2d_start_hook_ ABSL_GUARDED_BY(h2d_hook_mu_);
  absl::AnyInvocable<void(const absl::Status&)> h2d_end_hook_
      ABSL_GUARDED_BY(h2d_hook_mu_);

  std::unique_ptr<HostMemoryAllocator> host_allocator_;
  // Runs ForEachShardNumaLocal()'s per-shard tasks (tiling, H2D/D2H issue).
  // Separate from the owner's receive pool so that H2d() never waits on tasks
  // queued behind its own caller. It has one thread per shard, so a task on it
  // must never block on another task of this pool (e.g. H2d() waiting for
  // in-place tiling done by TileLayer()).
  absl_nonnull std::unique_ptr<tpu_raiden::NumaThreadPool> shard_pool_;
  std::vector<std::unique_ptr<ShardScratchpad>> tiled_scratchpads_;

  // Serializes on-demand per-layer re-staging when concurrent skip_d2h=true
  // pushes encounter a staged-vs-planned skip_tiling mismatch.
  absl::Mutex restage_mu_;

  // Serializes D2H scheduling (D2h(), in-push D2H, re-staging) and non-empty
  // BindWeights(). Ordered after restage_mu_ and before
  // device_buffer_resolver_mu_, explicit_d2h_mu_, buffer_holds_mu_,
  // tile_ready_mu_, metrics_mu_, and skip_tiling_mu_.
  mutable absl::Mutex d2h_op_mu_;

  // Per-layer readiness and layout recorded by explicit D2h() (and layout
  // recorded by in-push D2H / re-staging), plus combined in-flight D2H future.
  // Cleared by BindWeights(), receiver chunk registration, and per-layer by
  // OnLayerDataReceived(); preserved across UnbindWeights() so a subsequent
  // skip_d2h=true push can detect layout mismatches.
  mutable absl::Mutex explicit_d2h_mu_;
  std::vector<std::shared_ptr<LayerTileReady>> explicit_d2h_ready_
      ABSL_GUARDED_BY(explicit_d2h_mu_);
  std::vector<std::optional<bool>> explicit_d2h_skip_tiling_
      ABSL_GUARDED_BY(explicit_d2h_mu_);
  xla::Future<> in_flight_d2h_ ABSL_GUARDED_BY(explicit_d2h_mu_);

  // Per-layer in-place tiling state of the host staging buffers, indexed by
  // layer. A non-null entry means the layer's host buffers hold device-layout
  // bytes once the signal is set; null means they hold logical-layout bytes.
  absl::Mutex tile_ready_mu_;
  std::vector<std::shared_ptr<LayerTileReady>> layer_tile_ready_
      ABSL_GUARDED_BY(tile_ready_mu_);

  mutable absl::Mutex skip_tiling_mu_;
  absl::flat_hash_map<uint64_t, std::vector<bool>> uuid_to_skip_tiling_
      ABSL_GUARDED_BY(skip_tiling_mu_);
  std::vector<bool> latest_skip_tiling_ ABSL_GUARDED_BY(skip_tiling_mu_);

  mutable absl::Mutex metrics_mu_;
  WeightSyncMetrics metrics_ ABSL_GUARDED_BY(metrics_mu_);
  size_t active_h2h_pushes_ ABSL_GUARDED_BY(metrics_mu_) = 0;
  absl::Time h2h_active_window_start_ ABSL_GUARDED_BY(metrics_mu_) =
      absl::InfinitePast();

  std::atomic<size_t> allocated_host_dram_bytes_{0};
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_BUFFERS_H_

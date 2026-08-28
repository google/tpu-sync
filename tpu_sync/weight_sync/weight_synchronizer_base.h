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

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
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
  // Symmetrical core constructor wrapping raw PJRT buffers directly E2E
  WeightSynchronizerBase(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      std::optional<int> local_port = std::nullopt,
      std::optional<std::vector<const uint8_t*>> external_host_ptrs =
          std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false);

  // CPU-only constructor for remote workers and mock E2E testing
  WeightSynchronizerBase(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1, std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false);

  // CPU-only constructor for remote workers and mock E2E testing supporting
  // heterogeneous slice sizes and custom layer names.
  WeightSynchronizerBase(
      size_t num_layers, size_t num_shards,
      std::vector<size_t> slice_byte_sizes,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1, std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      std::vector<std::string> layer_names = {}, bool auto_h2d = false);

  std::optional<int> listener_port() const;
  bool is_listener_active() const;
  virtual std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  ~WeightSynchronizerBase() override;

  // Trainer pushes current weights to all inference server peers E2E (D2H +
  // network push)
  absl::Status PushWeights(const std::vector<std::string>& peers);

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
  absl::Status PushWeightsResharded(
      const tpu_sync::rpc::StartTransferRequest& request);

  const uint8_t* GetHostBufferPtr(size_t layer_idx, size_t shard_idx) const {
    if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
      return nullptr;
    }
    return layers_[layer_idx].shards[shard_idx].host_ptr;
  }

  // Returns the list of layer names associated with the weight synchronizer.
  const std::vector<std::string>& layer_names() const { return layer_names_; }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> H2dLayer(size_t layer_idx,
                                                  uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> D2hLayer(size_t layer_idx,
                                                  uint64_t uuid = 0);

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

  void StoreSkipTiling(uint64_t uuid,
                       const tpu_sync::rpc::StartTransferRequest& request);

  void SetSkipTiling(const std::vector<bool>& skip_tiling) {
    absl::MutexLock lock(skip_tiling_mu_);
    latest_skip_tiling_ = skip_tiling;
  }
  void SetSkipTiling(bool skip_all) {
    absl::MutexLock lock(skip_tiling_mu_);
    latest_skip_tiling_.assign(num_layers_, skip_all);
  }

  WeightSyncMetrics GetMetrics() const {
    absl::MutexLock lock(metrics_mu_);
    return metrics_;
  }

  void ResetMetrics() {
    absl::MutexLock lock(metrics_mu_);
    metrics_ = WeightSyncMetrics{};
  }

  void SetPipelineGroupSize(std::optional<size_t> group_size) {
    pipeline_group_size_override_ = group_size;
  }
  std::optional<size_t> pipeline_group_size_override() const {
    return pipeline_group_size_override_;
  }
  size_t GetPipelineGroupSize() const;

  absl::Status OnBlocksReceived(const std::vector<int>& block_ids,
                                uint64_t uuid = 0) override;

  absl::Status RegisterExpectedChunks(uint64_t uuid,
                                      uint32_t expected_chunks) override;
  absl::Status RegisterExpectedLayerChunks(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks)
      override;

  absl::Status OnLayerDataReceived(size_t layer_idx,
                                   uint64_t uuid = 0) override;
  absl::Status OnDataReceived(uint64_t uuid = 0) override;

  absl::Status WaitForTransferCompletion(uint64_t uuid);

  void ForgetPushProgress(uint64_t uuid) override;

 protected:
  std::unique_ptr<WeightSynchronizerListener> listener_;
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;
  // Opaque human-readable name labels mapped to logical layer index boundaries.
  std::vector<std::string> layer_names_;

  // Separate PJRT active holds matrix E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_;

  // Override parent AllocateBlocks as simple identity indices since we sync
  // entire buffers E2E!
  absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) override {
    std::vector<int> ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) ids[i] = static_cast<int>(i);
    return ids;
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

 private:
  // When enabled, automatically schedules asynchronous device transfers (H2D)
  // upon complete host buffer writes.
  bool auto_h2d_ = false;

  struct PendingH2dState {
    size_t expected_layers = 0;
    absl::flat_hash_map<size_t,
                        std::future<absl::StatusOr<raiden::PjRtCopyFuture>>>
        layer_futures;
  };

  std::unique_ptr<tpu_raiden::NumaThreadPool> h2d_pool_;
  std::unique_ptr<tpu_raiden::NumaThreadPool> push_pool_;

  mutable absl::Mutex skip_tiling_mu_;
  absl::flat_hash_map<uint64_t, std::vector<bool>> uuid_to_skip_tiling_
      ABSL_GUARDED_BY(skip_tiling_mu_);
  std::vector<bool> latest_skip_tiling_ ABSL_GUARDED_BY(skip_tiling_mu_);

  mutable absl::Mutex d2h_mu_;
  absl::flat_hash_set<uint64_t> completed_d2h_uuids_ ABSL_GUARDED_BY(d2h_mu_);

  mutable absl::Mutex pending_h2d_mu_;
  absl::flat_hash_map<uint64_t, PendingH2dState> pending_h2d_states_
      ABSL_GUARDED_BY(pending_h2d_mu_);

  mutable absl::Mutex metrics_mu_;
  WeightSyncMetrics metrics_ ABSL_GUARDED_BY(metrics_mu_);

  mutable absl::Mutex completed_transfers_mu_;
  absl::flat_hash_set<uint64_t> completed_transfers_
      ABSL_GUARDED_BY(completed_transfers_mu_);

  std::optional<size_t> pipeline_group_size_override_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_

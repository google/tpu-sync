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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raiden_future.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/jax_utils.h"
#endif
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
namespace jax {

// TODO(raiden-dev): Refactor weight synchronizer hierarchy so that it inherits
// from a common NumaAware manager base abstraction.
class NumaAwareWeightSynchronizer
    : public weight_sync::WeightSynchronizerControlDelegate {
 public:
  NumaAwareWeightSynchronizer(const NumaAwareWeightSynchronizer&) = delete;
  NumaAwareWeightSynchronizer& operator=(const NumaAwareWeightSynchronizer&) =
      delete;
  NumaAwareWeightSynchronizer(NumaAwareWeightSynchronizer&&) = default;
  NumaAwareWeightSynchronizer& operator=(NumaAwareWeightSynchronizer&&) =
      default;

#ifndef WITHOUT_PYTHON
  NumaAwareWeightSynchronizer(
      nanobind::list jax_arrays, std::optional<int> local_port = std::nullopt,
      int parallelism = 1, bool unsafe_skip_buffer_lock = false,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt, bool auto_h2d = false,
      std::optional<int> global_shard_offset = std::nullopt);

  absl::Status BindWeights(nanobind::list jax_arrays);
#endif

  // CPU / Mock metadata constructor for tests without PJRT TPU devices
  NumaAwareWeightSynchronizer(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt, bool auto_h2d = false,
      std::optional<int> global_shard_offset = std::nullopt);

  // Test-only constructor for injecting mock sub-synchronizers
  explicit NumaAwareWeightSynchronizer(
      std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>>
          sub_synchronizers);

  virtual ~NumaAwareWeightSynchronizer();

  size_t num_layers() const { return num_layers_; }
  size_t num_shards() const { return total_num_shards_; }
  size_t slice_byte_size() const { return slice_byte_size_; }

  std::optional<int> local_port() const;
  std::optional<int> listener_port() const;
  bool is_listener_active() const;

  std::vector<std::string> local_ips() const;
  std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  const uint8_t* GetHostBufferPtr(size_t layer_idx, size_t shard_idx) const;

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid = 0);

  void SetSkipTiling(const std::vector<bool>& skip_tiling);
  void SetSkipTiling(bool skip_all);

  weight_sync::WeightSyncMetrics GetMetrics() const;
  void ResetMetrics();

  absl::Status PushWeights(const std::vector<std::string>& peers) override;
  absl::Status PushWeightsResharded(
      const tpu_sync::rpc::StartTransferRequest& request) override;
  void StoreSkipTiling(
      uint64_t uuid,
      const tpu_sync::rpc::StartTransferRequest& request) override;
  absl::Status RegisterExpectedChunks(uint64_t uuid,
                                      uint32_t expected_chunks) override;
  absl::Status RegisterExpectedLayerChunks(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks)
      override;
  absl::Status WaitForTransferCompletion(uint64_t uuid) override;
  void ForgetPushProgress(uint64_t uuid) override;
  void DrainPendingH2d() override;

  const std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>>&
  sub_synchronizers() const {
    return sub_synchronizers_;
  }

  void SetSubmanagerShardsForTesting(
      const std::vector<std::vector<int64_t>>& assignment);

 private:
  void InitSubManagers(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      std::optional<int> local_port, bool unsafe_skip_buffer_lock,
      int parallelism, std::optional<int> listener_port,
      std::optional<std::string> bind_ip, bool auto_h2d,
      std::optional<int> global_shard_offset = std::nullopt);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>>
      sub_synchronizers_;
  std::vector<std::pair<int, int>> global_shard_to_submanager_;
  std::vector<std::vector<int64_t>> submanager_to_global_shards_;
  std::vector<std::vector<int>> submanager_to_local_shards_;
  size_t total_num_shards_ = 0;
  size_t num_layers_ = 0;
  size_t slice_byte_size_ = 0;
  bool unsafe_skip_buffer_lock_ = false;
  int global_shard_offset_ = 0;
  std::unique_ptr<tpu_raiden::NumaThreadPool> push_pool_;

  absl::Mutex expected_counts_mu_;
  absl::flat_hash_map<uint64_t,
                      std::vector<absl::flat_hash_map<size_t, uint32_t>>>
      uuid_to_sub_layer_counts_ ABSL_GUARDED_BY(expected_counts_mu_);
  absl::flat_hash_map<uint64_t, std::vector<uint32_t>> uuid_to_sub_total_counts_
      ABSL_GUARDED_BY(expected_counts_mu_);
};

class WeightSynchronizer {
 public:
  WeightSynchronizer(const WeightSynchronizer&) = delete;
  WeightSynchronizer& operator=(const WeightSynchronizer&) = delete;
  WeightSynchronizer(WeightSynchronizer&&) = default;
  WeightSynchronizer& operator=(WeightSynchronizer&&) = default;

#ifndef WITHOUT_PYTHON
  WeightSynchronizer(nanobind::list jax_arrays,
                     std::optional<int> local_port = std::nullopt,
                     int parallelism = 1, bool unsafe_skip_buffer_lock = false,
                     std::optional<int> listener_port = std::nullopt,
                     std::optional<std::string> bind_ip = std::nullopt,
                     bool auto_h2d = false,
                     std::optional<int> global_shard_offset = std::nullopt);
  absl::Status BindWeights(nanobind::list jax_arrays);
#endif

  // CPU / Mock metadata constructor for tests without PJRT TPU devices
  WeightSynchronizer(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size,
                     std::optional<int> local_port = std::nullopt,
                     int parallelism = 1,
                     std::optional<int> listener_port = std::nullopt,
                     std::optional<std::string> bind_ip = std::nullopt,
                     bool auto_h2d = false,
                     std::optional<int> global_shard_offset = std::nullopt);

  // Test-only constructor for injecting mock sub-synchronizers
  explicit WeightSynchronizer(
      std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>>
          sub_synchronizers);

  ~WeightSynchronizer();

  NumaAwareWeightSynchronizer* numa_manager() const {
    return numa_manager_.get();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid = 0);
  void SetSkipTiling(const std::vector<bool>& skip_tiling);
  void SetSkipTiling(bool skip_all);

  weight_sync::WeightSyncMetrics GetMetrics() const;
  void ResetMetrics();

  const uint8_t* GetHostBufferPtr(size_t layer_idx, size_t shard_idx) const;
  std::optional<int> local_port() const;
  std::optional<int> listener_port() const;
  bool is_listener_active() const;

  std::vector<std::string> local_ips() const;
  std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  size_t num_layers() const;
  size_t num_shards() const;
  size_t slice_byte_size() const;

 private:
  std::unique_ptr<NumaAwareWeightSynchronizer> numa_manager_;
};

}  // namespace jax
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_

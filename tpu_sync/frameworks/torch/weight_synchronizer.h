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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_WEIGHT_SYNCHRONIZER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_WEIGHT_SYNCHRONIZER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#ifndef WITHOUT_PYTHON
#include "ATen/core/TensorBody.h"
#endif
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_sync {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc
}  // namespace tpu_sync

namespace tpu_raiden {
namespace torch {

// PyTorch facade over weight_sync::WeightSynchronizerBase. Unpacks at::Tensor
// shards into PJRT buffer handles and keeps the tensor buffer references alive
// for the lifetime of the binding. TorchTPU moves a tensor to a new device
// buffer on in-place ops, so the bound tensors' buffers are looked up again
// before every D2H and H2D. NUMA placement of host tiling, H2D and transport
// listeners is handled by the base class per shard.
class WeightSynchronizer {
 public:
  WeightSynchronizer(const WeightSynchronizer&) = delete;
  WeightSynchronizer& operator=(const WeightSynchronizer&) = delete;
  WeightSynchronizer(WeightSynchronizer&&) = default;
  WeightSynchronizer& operator=(WeightSynchronizer&&) = default;

#ifndef WITHOUT_PYTHON
  // Creates a synchronizer bound to `device_tensors` (outer index = layer,
  // inner index = shard) and keeps their TPU buffer handles alive. Returns
  // InvalidArgument if `device_tensors` yields no layer buffers and Internal
  // if unpacking the tensors or constructing the underlying
  // WeightSynchronizerBase throws. `global_shard_indices`, when set, overrides
  // the default identity local->global shard mapping.
  static absl::StatusOr<std::unique_ptr<WeightSynchronizer>> Create(
      const std::vector<std::vector<at::Tensor>>& device_tensors,
      std::optional<int> local_port = std::nullopt, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt,
      bool unsafe_skip_buffer_lock = true, bool auto_h2d = false,
      std::optional<std::vector<int64_t>> global_shard_indices = std::nullopt);

  absl::Status BindWeights(
      const std::vector<std::vector<at::Tensor>>& device_tensors);
#endif
  // Releases all bound device tensors and TPU HBM buffer holds while keeping
  // the synchronizer (host staging memory, listeners, controller registration)
  // alive for a later BindWeights(). D2h()/H2d() and pushes that must read
  // device buffers fail with FailedPrecondition until weights are re-bound.
  // Once D2h() has completed, UnbindWeights() may be called even while a
  // skip_d2h=true H2H push is still sending from host staging buffers (for the
  // first transfer to a destination topology whose skip_tiling plan is not yet
  // cached, keep weights bound until the transfer finishes so any layout
  // mismatch can be re-staged).
  void UnbindWeights();

  ~WeightSynchronizer();

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0);
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid = 0);

  absl::Status PushWeights(const std::vector<std::string>& peers);
  absl::Status PushWeightsResharded(
      const tpu_sync::rpc::StartTransferRequest& request);

  void SetSkipTiling(const std::vector<bool>& skip_tiling);
  void SetSkipTiling(bool skip_all);
  void DrainPendingH2d();
  absl::Status WaitForTransferCompletion(uint64_t uuid = 0);

  weight_sync::WeightSyncMetrics GetMetrics() const;
  void ResetMetrics();

  // Returns the local host staging buffer future of (|layer_idx|,
  // |local_shard_idx|), with the contract of
  // `WeightSynchronizerBase::GetHostBufferWithLocalIndex()`.
  xla::Future<absl::Span<const uint8_t>> GetHostBufferWithLocalIndex(
      size_t layer_idx, size_t local_shard_idx) const;
  xla::Future<absl::Span<uint8_t>> GetHostBufferWithLocalIndex(
      size_t layer_idx, size_t local_shard_idx);
  std::optional<int> local_port() const;
  std::optional<int> listener_port() const;
  bool is_listener_active() const;

  std::vector<std::string> local_ips() const;
  std::vector<RaidenTransferEndpoint> get_local_endpoints() const;

  size_t num_layers() const;
  size_t num_shards() const;
  size_t slice_byte_size() const;

  void test_only_set_bandwidth_limit(double test_only_simulated_egress_gbps,
                                     double test_only_simulated_ingress_gbps);

  // Creates a CPU-only (no PJRT device buffers) synchronizer for tests and
  // remote workers; see WeightSynchronizerBase::Create for failure modes.
  static absl::StatusOr<std::unique_ptr<WeightSynchronizer>>
  test_only_create_cpu_instance(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt, bool auto_h2d = false,
      std::optional<std::vector<int64_t>> global_shard_indices = std::nullopt,
      double test_only_simulated_egress_gbps = 0.0,
      double test_only_simulated_ingress_gbps = 0.0);

  static absl::StatusOr<std::unique_ptr<WeightSynchronizer>>
  test_only_create_cpu_instance(
      size_t num_layers, size_t num_shards,
      std::vector<size_t> slice_byte_sizes,
      std::optional<int> local_port = std::nullopt, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt, bool auto_h2d = false,
      std::optional<std::vector<int64_t>> global_shard_indices = std::nullopt,
      double test_only_simulated_egress_gbps = 0.0,
      double test_only_simulated_ingress_gbps = 0.0);

 private:
  // Used by test_only_create_cpu_instance(); adopts an already constructed
  // `base`.
  explicit WeightSynchronizer(
      std::unique_ptr<weight_sync::WeightSynchronizerBase> base);
#ifndef WITHOUT_PYTHON
  // The bound tensors and the handles that pin their device buffers; defined
  // in the .cc file.
  struct BoundTensors;

  // Used by Create(); adopts an already constructed `base` and the `bound`
  // tensors, whose buffers `base` resolves before every transfer.
  WeightSynchronizer(std::unique_ptr<weight_sync::WeightSynchronizerBase> base,
                     std::shared_ptr<BoundTensors> bound);

  // Returns the device buffers that currently back `bound`'s tensors, once
  // pending TorchTPU writes to them have completed, and keeps them pinned.
  static absl::StatusOr<std::vector<std::vector<raiden::RaidenBufferHandle>>>
  ResolveDeviceBuffers(BoundTensors& bound);
#endif

  std::unique_ptr<weight_sync::WeightSynchronizerBase> base_;
#ifndef WITHOUT_PYTHON
  // Null for CPU-only instances.
  std::shared_ptr<BoundTensors> bound_;
#endif
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_WEIGHT_SYNCHRONIZER_H_

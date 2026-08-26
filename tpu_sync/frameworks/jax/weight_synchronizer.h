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

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/core/raiden_future.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#ifndef WITHOUT_PYTHON
#include "tpu_sync/frameworks/jax/jax_utils.h"
#endif
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

#ifndef WITHOUT_PYTHON
namespace nb = nanobind;
#endif

namespace tpu_raiden {
namespace jax {



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
                     bool auto_h2d = false);
  absl::Status BindWeights(nanobind::list jax_arrays);
#endif

  ~WeightSynchronizer();

  absl::StatusOr<raiden::PjRtCopyFuture> D2h();
  absl::StatusOr<raiden::PjRtCopyFuture> H2d();
  void SetSkipTiling(const std::vector<bool>& skip_tiling);
  void SetSkipTiling(bool skip_all);

  weight_sync::WeightSyncMetrics GetMetrics() const;

  const uint8_t* GetHostBufferPtr(size_t layer_idx, size_t shard_idx) const;
  std::optional<int> local_port() const;
  std::optional<int> listener_port() const;
  bool is_listener_active() const;
  std::vector<RaidenTransferEndpoint> get_local_endpoints() const;
  size_t num_layers() const;
  size_t num_shards() const;
  size_t slice_byte_size() const;

 private:
#ifndef WITHOUT_PYTHON
  bool unsafe_skip_buffer_lock_ = false;
#endif
  std::unique_ptr<weight_sync::WeightSynchronizerBase> impl_;
};

}  // namespace jax
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_

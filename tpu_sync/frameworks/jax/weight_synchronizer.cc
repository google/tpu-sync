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

#include "tpu_sync/frameworks/jax/weight_synchronizer.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"
namespace tpu_raiden {
namespace jax {

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/utils.h"

WeightSynchronizer::WeightSynchronizer(nanobind::list jax_arrays,
                                       std::optional<int> local_port,
                                       int parallelism,
                                       bool unsafe_skip_buffer_lock,
                                       std::optional<int> listener_port,
                                       std::optional<std::string> bind_ip,
                                       bool auto_h2d)
    : unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
  auto layer_buffers =
      tpu_raiden::jax::UnpackJaxArrays(jax_arrays, unsafe_skip_buffer_lock);
  impl_ = std::make_unique<weight_sync::WeightSynchronizerBase>(
      layer_buffers, local_port,
      /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock, parallelism,
      listener_port, bind_ip, /*layer_names=*/std::vector<std::string>{},
      auto_h2d);
}

absl::Status WeightSynchronizer::BindWeights(nanobind::list jax_arrays) {
  try {
    auto layer_buffers =
        tpu_raiden::jax::UnpackJaxArrays(jax_arrays, unsafe_skip_buffer_lock_);
    TF_RETURN_IF_ERROR(impl_->BindWeights(layer_buffers));
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
}

#endif  // WITHOUT_PYTHON

WeightSynchronizer::~WeightSynchronizer() = default;

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::D2h() {
  return impl_->D2h();
}
absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::H2d() {
  return impl_->H2d();
}
void WeightSynchronizer::SetSkipTiling(const std::vector<bool>& skip_tiling) {
  impl_->SetSkipTiling(skip_tiling);
}
void WeightSynchronizer::SetSkipTiling(bool skip_all) {
  impl_->SetSkipTiling(skip_all);
}

weight_sync::WeightSyncMetrics WeightSynchronizer::GetMetrics() const {
  return impl_->GetMetrics();
}

const uint8_t* WeightSynchronizer::GetHostBufferPtr(size_t layer_idx,
                                                    size_t shard_idx) const {
  return impl_->GetHostBufferPtr(layer_idx, shard_idx);
}
std::optional<int> WeightSynchronizer::local_port() const {
  return impl_->local_port();
}
std::optional<int> WeightSynchronizer::listener_port() const {
  return impl_->listener_port();
}
bool WeightSynchronizer::is_listener_active() const {
  return impl_->is_listener_active();
}
std::vector<RaidenTransferEndpoint> WeightSynchronizer::get_local_endpoints()
    const {
  return impl_->get_local_endpoints();
}
size_t WeightSynchronizer::num_layers() const { return impl_->num_layers(); }
size_t WeightSynchronizer::num_shards() const { return impl_->num_shards(); }
size_t WeightSynchronizer::slice_byte_size() const {
  return impl_->slice_byte_size();
}

}  // namespace jax
}  // namespace tpu_raiden

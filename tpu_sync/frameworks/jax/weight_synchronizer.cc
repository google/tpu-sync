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

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "peregrine/src/api/v1/rate_limiter.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

#ifndef WITHOUT_PYTHON
#include "absl/strings/str_cat.h"
#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/utils.h"
#endif

namespace tpu_raiden {
namespace jax {

#ifndef WITHOUT_PYTHON
WeightSynchronizer::WeightSynchronizer(
    std::unique_ptr<weight_sync::WeightSynchronizerBase> base,
    bool unsafe_skip_buffer_lock)
    : base_(std::move(base)),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {}

absl::StatusOr<std::unique_ptr<WeightSynchronizer>> WeightSynchronizer::Create(
    nanobind::list jax_arrays, std::optional<int> local_port, int parallelism,
    bool unsafe_skip_buffer_lock, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, bool auto_h2d,
    std::optional<std::vector<int64_t>> global_shard_indices) {
  try {
    std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers =
        tpu_raiden::jax::UnpackJaxArrays(jax_arrays, unsafe_skip_buffer_lock);
    if (layer_buffers.empty()) {
      return absl::InvalidArgumentError(
          "WeightSynchronizer::Create: jax_arrays yielded no layer buffers");
    }
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<weight_sync::WeightSynchronizerBase> base,
        weight_sync::WeightSynchronizerBase::Create(
            layer_buffers, local_port, /*external_host_ptrs=*/std::nullopt,
            unsafe_skip_buffer_lock, parallelism, listener_port,
            std::move(bind_ip), /*layer_names=*/std::vector<std::string>{},
            auto_h2d));
    if (global_shard_indices.has_value()) {
      base->SetGlobalShardIndices(*std::move(global_shard_indices));
    }
    return absl::WrapUnique(
        new WeightSynchronizer(std::move(base), unsafe_skip_buffer_lock));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("WeightSynchronizer::Create failed: ", e.what()));
  }
}

absl::Status WeightSynchronizer::BindWeights(nanobind::list jax_arrays) {
  try {
    std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers =
        tpu_raiden::jax::UnpackJaxArrays(jax_arrays, unsafe_skip_buffer_lock_);
    if (layer_buffers.empty()) {
      return absl::InvalidArgumentError(
          "Empty layer buffers provided to BindWeights");
    }
    if (layer_buffers.size() != num_layers()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Layer count mismatch in BindWeights: expected ",
                       num_layers(), ", got ", layer_buffers.size()));
    }
    if (layer_buffers[0].size() != num_shards()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Shard count mismatch in BindWeights: expected ",
                       num_shards(), ", got ", layer_buffers[0].size()));
    }
    return base_->BindWeights(layer_buffers);
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
}
#endif

WeightSynchronizer::WeightSynchronizer(
    std::unique_ptr<weight_sync::WeightSynchronizerBase> base)
    : base_(std::move(base)) {}

WeightSynchronizer::~WeightSynchronizer() = default;

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::D2h(uint64_t uuid) {
  return base_->D2h(uuid);
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::H2d(uint64_t uuid) {
  return base_->H2d(uuid);
}

absl::Status WeightSynchronizer::WaitForTransferCompletion(uint64_t uuid) {
  return base_->WaitForTransferCompletion(uuid);
}

void WeightSynchronizer::SetSkipTiling(const std::vector<bool>& skip_tiling) {
  base_->SetSkipTiling(skip_tiling);
}

void WeightSynchronizer::SetSkipTiling(bool skip_all) {
  base_->SetSkipTiling(skip_all);
}

weight_sync::WeightSyncMetrics WeightSynchronizer::GetMetrics() const {
  return base_->GetMetrics();
}

void WeightSynchronizer::ResetMetrics() { base_->ResetMetrics(); }

const uint8_t* WeightSynchronizer::GetHostBufferPtr(size_t layer_idx,
                                                    size_t shard_idx) const {
  return base_->GetHostBufferPtr(layer_idx, shard_idx);
}

size_t WeightSynchronizer::GetHostBufferSize(size_t layer_idx,
                                             size_t shard_idx) const {
  return base_->GetHostSize(layer_idx, shard_idx);
}

std::optional<int> WeightSynchronizer::local_port() const {
  return base_->local_port();
}

std::optional<int> WeightSynchronizer::listener_port() const {
  return base_->listener_port();
}

bool WeightSynchronizer::is_listener_active() const {
  return base_->is_listener_active();
}

std::vector<std::string> WeightSynchronizer::local_ips() const {
  return base_->local_ips();
}

std::vector<RaidenTransferEndpoint> WeightSynchronizer::get_local_endpoints()
    const {
  return base_->get_local_endpoints();
}

size_t WeightSynchronizer::num_layers() const { return base_->num_layers(); }

size_t WeightSynchronizer::num_shards() const { return base_->num_shards(); }

size_t WeightSynchronizer::slice_byte_size() const {
  return base_->slice_byte_size();
}

void WeightSynchronizer::test_only_set_bandwidth_limit(
    double test_only_simulated_egress_gbps,
    double test_only_simulated_ingress_gbps) {
  std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> egress_limiter;
  if (test_only_simulated_egress_gbps > 0.0) {
    egress_limiter = std::make_shared<peregrine::v1::TestOnlyRateLimiter>(
        static_cast<uint64_t>(test_only_simulated_egress_gbps * 1e9 / 8.0));
  }
  std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> ingress_limiter;
  if (test_only_simulated_ingress_gbps > 0.0) {
    ingress_limiter = std::make_shared<peregrine::v1::TestOnlyRateLimiter>(
        static_cast<uint64_t>(test_only_simulated_ingress_gbps * 1e9 / 8.0));
  }
  base_->SetTestOnlyRateLimiters(std::move(egress_limiter),
                                 std::move(ingress_limiter));
}

absl::StatusOr<std::unique_ptr<WeightSynchronizer>>
WeightSynchronizer::test_only_create_cpu_instance(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    bool auto_h2d, std::optional<std::vector<int64_t>> global_shard_indices,
    double test_only_simulated_egress_gbps,
    double test_only_simulated_ingress_gbps) {
  return test_only_create_cpu_instance(
      num_layers, num_shards, std::vector<size_t>(num_layers, slice_byte_size),
      local_port, parallelism, listener_port, std::move(bind_ip), auto_h2d,
      std::move(global_shard_indices), test_only_simulated_egress_gbps,
      test_only_simulated_ingress_gbps);
}

absl::StatusOr<std::unique_ptr<WeightSynchronizer>>
WeightSynchronizer::test_only_create_cpu_instance(
    size_t num_layers, size_t num_shards, std::vector<size_t> slice_byte_sizes,
    std::optional<int> local_port, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    bool auto_h2d, std::optional<std::vector<int64_t>> global_shard_indices,
    double test_only_simulated_egress_gbps,
    double test_only_simulated_ingress_gbps) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<weight_sync::WeightSynchronizerBase> base,
      weight_sync::WeightSynchronizerBase::Create(
          num_layers, num_shards, std::move(slice_byte_sizes), local_port,
          /*host_blocks_to_allocate=*/std::nullopt, parallelism, listener_port,
          std::move(bind_ip), /*layer_names=*/std::vector<std::string>{},
          auto_h2d));
  if (global_shard_indices.has_value()) {
    base->SetGlobalShardIndices(*std::move(global_shard_indices));
  }
  auto ws = absl::WrapUnique(new WeightSynchronizer(std::move(base)));
  if (test_only_simulated_egress_gbps > 0.0 ||
      test_only_simulated_ingress_gbps > 0.0) {
    ws->test_only_set_bandwidth_limit(test_only_simulated_egress_gbps,
                                      test_only_simulated_ingress_gbps);
  }
  return ws;
}

}  // namespace jax
}  // namespace tpu_raiden

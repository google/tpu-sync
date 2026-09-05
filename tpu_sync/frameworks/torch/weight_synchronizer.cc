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

#include "tpu_sync/frameworks/torch/weight_synchronizer.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "tpu_sync/frameworks/torch/torch_utils.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace torch {

WeightSynchronizer::WeightSynchronizer(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    bool unsafe_skip_buffer_lock, bool auto_h2d)
    : WeightSynchronizer(
          UnpackTorchTensors(device_tensors, unsafe_skip_buffer_lock),
          local_port, parallelism, listener_port, bind_ip,
          unsafe_skip_buffer_lock, auto_h2d) {}

WeightSynchronizer::WeightSynchronizer(
    UnpackedTensors unpacked, std::optional<int> local_port, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    bool unsafe_skip_buffer_lock, bool auto_h2d)
    : weight_sync::WeightSynchronizerBase(
          std::move(unpacked.buffers), local_port,
          /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock,
          parallelism, listener_port, bind_ip,
          /*layer_names=*/{}, auto_h2d),
      buffer_refs_(std::move(unpacked.refs)),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {}

absl::Status WeightSynchronizer::BindWeights(
    const std::vector<std::vector<at::Tensor>>& device_tensors) {
  UnpackedTensors unpacked =
      UnpackTorchTensors(device_tensors, unsafe_skip_buffer_lock_);
  buffer_refs_ = std::move(unpacked.refs);
  return weight_sync::WeightSynchronizerBase::BindWeights(
      std::move(unpacked.buffers));
}

WeightSynchronizer::~WeightSynchronizer() = default;

}  // namespace torch
}  // namespace tpu_raiden

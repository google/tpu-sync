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

#include <optional>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "tpu_sync/frameworks/torch/torch_utils.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace torch {

class WeightSynchronizer : public weight_sync::WeightSynchronizerBase {
 public:
  // PyTorch sharded constructor E2E
  WeightSynchronizer(const std::vector<std::vector<at::Tensor>>& device_tensors,
                     std::optional<int> local_port = std::nullopt,
                     int parallelism = 1,
                     std::optional<int> listener_port = std::nullopt,
                     std::optional<std::string> bind_ip = std::nullopt,
                     bool unsafe_skip_buffer_lock = true,
                     bool auto_h2d = false);

  ~WeightSynchronizer() override;

 private:
  // Delegated-to ctor: moves the keep-alive refs into buffer_refs_ so the
  // materialized (possibly view) device buffers survive for our lifetime.
  WeightSynchronizer(UnpackedTensors unpacked, std::optional<int> local_port,
                     int parallelism, std::optional<int> listener_port,
                     std::optional<std::string> bind_ip,
                     bool unsafe_skip_buffer_lock, bool auto_h2d);

  std::vector<torch_tpu::DeviceBufferRef> buffer_refs_;
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_WEIGHT_SYNCHRONIZER_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_BUFFER_PUSH_TASK_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_BUFFER_PUSH_TASK_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace tpu_raiden {
namespace transport {

// TODO(yongx): add a description.
struct BufferPushTask {
  std::string peer;
  size_t buffer_id = 0;
  size_t dst_shard_idx = 0;
  size_t dst_offset_bytes = 0;
  const uint8_t* data_ptr = nullptr;
  size_t size_bytes = 0;
  size_t dst_stride_bytes = 0;
  size_t count = 1;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_BUFFER_PUSH_TASK_H_

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_TILING_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_TILING_UTILS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/layout.h"
#include "xla/shape.h"

namespace tpu_raiden {
class NumaThreadPool;
}  // namespace tpu_raiden

namespace tpu_raiden::weight_sync {

// Calculates the total number of physical elements required for a tiled buffer.
int64_t GetTiledBufferElements(const xla::Shape& shape);

// Reconstructs a linear buffer from a tiled buffer based on shape and layout.
absl::Status DetileBuffer(const uint8_t* src_tiled, uint8_t* dst_linear,
                          const xla::Shape& shape, const xla::Layout& layout,
                          tpu_raiden::NumaThreadPool* pool = nullptr);

// Tiles a linear buffer based on shape and layout.
absl::Status TileBuffer(const uint8_t* src_linear, uint8_t* dst_tiled,
                        const xla::Shape& shape, const xla::Layout& layout,
                        tpu_raiden::NumaThreadPool* pool = nullptr);

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_TILING_UTILS_H_

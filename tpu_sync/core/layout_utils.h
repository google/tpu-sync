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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_LAYOUT_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_LAYOUT_UTILS_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/shape.h"

namespace tpu_raiden {

// Returns `dims` formatted as "[d0, d1, ...]".
std::string DimsToString(absl::Span<const int64_t> dims);

// Returns whether `folded_dims` folds two or more leading dimensions of `dims`
// into its first dimension and keeps the remaining dimensions, e.g.
// [8 * 768, 2048] for [8, 768, 2048].
bool FoldsLeadingDims(absl::Span<const int64_t> dims,
                      absl::Span<const int64_t> folded_dims);

// Returns OK if a buffer of shape `buffer_shape` keeps every element at the
// device byte offset where a buffer of the same element type, dimensions
// `folded_dims` and device layout `folded_layout` keeps it, so that raw device
// bytes can be copied between the two. `folded_dims` must fold leading
// dimensions of `buffer_shape` (see FoldsLeadingDims), and `buffer_shape` must
// carry the buffer's device layout. Otherwise returns an InvalidArgumentError
// saying why the device bytes differ.
absl::Status CheckFoldedDeviceLayout(const xla::Shape& buffer_shape,
                                     absl::Span<const int64_t> folded_dims,
                                     const xla::Layout& folded_layout);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_LAYOUT_UTILS_H_

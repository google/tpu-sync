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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_STORAGE_BACKEND_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_STORAGE_BACKEND_UTILS_H_

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden::kv_cache::backends::storage {

size_t GetStorageDirectIOAlignment();

bool AreStorageSlicesDirectIOAligned(
    absl::Span<const HostBufferDescriptor> slices);

std::string SanitizeStorageModelName(absl::string_view model_name);

}  // namespace tpu_raiden::kv_cache::backends::storage

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_STORAGE_BACKEND_UTILS_H_

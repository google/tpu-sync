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

#include "tpu_sync/kv_cache/backends/storage/storage_backend_utils.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden::kv_cache::backends::storage {

size_t GetStorageDirectIOAlignment() {
  static const size_t kAlign = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return kAlign;
}

bool AreStorageSlicesDirectIOAligned(
    absl::Span<const HostBufferDescriptor> slices) {
  const uintptr_t mask = GetStorageDirectIOAlignment() - 1;
  for (const auto& s : slices) {
    if (((reinterpret_cast<uintptr_t>(s.ptr) | s.size) & mask) != 0)
      return false;
  }
  return true;
}

std::string SanitizeStorageModelName(absl::string_view model_name) {
  std::string out;
  for (const char c : model_name) {
    out.push_back((absl::ascii_isalnum(static_cast<unsigned char>(c)) ||
                   c == '.' || c == '_' || c == '-')
                      ? c
                      : '_');
  }
  return out.empty() ? "unknown" : out;
}

}  // namespace tpu_raiden::kv_cache::backends::storage

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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_HOST_TILING_VALIDATION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_HOST_TILING_VALIDATION_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {

// Validates that `mode` is a known HostTilingMode enum value so callers never
// silently fall through to PRE_TILED or DEFERRED behavior. Returns
// InvalidArgumentError for unknown or unhandled values, using an explicit
// switch rather than HostTilingMode_IsValid so newly added proto enum values
// fail closed until consumers handle them.
inline absl::Status CheckKnownHostTilingMode(
    absl::string_view context, absl::string_view field,
    tpu_sync::rpc::HostTilingMode mode) {
  switch (mode) {
    case tpu_sync::rpc::HOST_TILING_MODE_UNSPECIFIED:
    case tpu_sync::rpc::HOST_TILING_MODE_DEFERRED:
    case tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL:
    case tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          context, " sets unknown ", field, "=", static_cast<int>(mode)));
  }
}

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_HOST_TILING_VALIDATION_H_

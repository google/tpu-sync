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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_VERSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_VERSION_H_

#include <cstddef>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/rpc/weight_transfer_common.pb.h"

namespace tpu_raiden::weight_sync {

// The longest version (`WeightVersion.tag`), in bytes.
// TODO(raiden-eng): Confirm with the first clients.
inline constexpr size_t kMaxWeightVersionBytes = 256;

// The version of a model's weights (see tpu_sync.rpc.WeightVersion): opaque
// bytes, 1 to kMaxWeightVersionBytes long, that name one set of weights and are
// never reused for different weights. Versions are compared bytewise for
// equality and never parsed, ordered or hashed. Only FromBytes() and
// FromProto() build one, so every WeightVersion is valid; callers use
// std::nullopt for "no version".
class WeightVersion final {
 public:
  WeightVersion() = delete;  // "No version" is std::nullopt, not an empty one.

  // Returns the version `bytes`, or InvalidArgument naming `field` (for
  // example "PrepareWeightReceive (uuid=3) version") if `bytes` is empty, which
  // names no version, or longer than kMaxWeightVersionBytes, naming both sizes.
  static absl::StatusOr<WeightVersion> FromBytes(absl::string_view bytes,
                                                 absl::string_view field);

  // FromBytes() of `proto.tag()`; an unset message has an empty tag.
  static absl::StatusOr<WeightVersion> FromProto(
      const tpu_sync::rpc::WeightVersion& proto, absl::string_view field);

  tpu_sync::rpc::WeightVersion ToProto() const;

  friend bool operator==(const WeightVersion&, const WeightVersion&) = default;

  // Formats `version` as logs and errors show it: version(len=N, "E"), where E
  // is absl::CHexEscape of at most the first 64 bytes, followed by "…" only
  // when longer.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WeightVersion& version) {
    sink.Append(version.FormatVersionBytes());
  }

 private:
  explicit WeightVersion(std::string bytes) : bytes_(std::move(bytes)) {}

  std::string FormatVersionBytes() const;

  std::string bytes_;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_VERSION_H_

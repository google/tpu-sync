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

// Maximum length of `WeightVersion.tag`, in bytes.
// TODO(raiden-eng): Confirm with the first clients.
inline constexpr size_t kMaxWeightVersionBytes = 256;

// Opaque version identifier for a set of weights (0 to kMaxWeightVersionBytes
// bytes), compared bytewise for equality. Empty bytes denote the default
// version (version 0).
class WeightVersion final {
 public:
  // Constructs the default version (version 0, empty tag).
  WeightVersion() = default;

  static WeightVersion Default() { return WeightVersion(); }

  // Returns the version for `bytes` (empty denotes Default()), or
  // InvalidArgument if longer than kMaxWeightVersionBytes.
  static absl::StatusOr<WeightVersion> FromBytes(absl::string_view bytes,
                                                 absl::string_view field);

  // Parses `proto.tag()`; an unset proto maps to Default().
  static absl::StatusOr<WeightVersion> FromProto(
      const tpu_sync::rpc::WeightVersion& proto, absl::string_view field);

  tpu_sync::rpc::WeightVersion ToProto() const;

  bool is_default() const { return bytes_.empty(); }

  friend bool operator==(const WeightVersion&, const WeightVersion&) = default;

  // Formats as "version(default)" or version(len=N, "E") with at most 64
  // escaped bytes.
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

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

#include "tpu_sync/weight_sync/weight_version.h"

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/rpc/weight_transfer_common.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

// Maximum bytes of a version shown in logs and errors.
constexpr size_t kMaxShownVersionBytes = 64;

}  // namespace

std::string WeightVersion::FormatVersionBytes() const {
  if (bytes_.empty()) {
    return "version(default)";
  }
  const absl::string_view bytes = bytes_;
  return absl::StrCat("version(len=", bytes.size(), ", \"",
                      absl::CHexEscape(bytes.substr(0, kMaxShownVersionBytes)),
                      bytes.size() > kMaxShownVersionBytes ? "\u2026" : "",
                      "\")");
}

absl::StatusOr<WeightVersion> WeightVersion::FromBytes(
    absl::string_view bytes, absl::string_view field) {
  if (bytes.size() > kMaxWeightVersionBytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        field, " has ", bytes.size(),
        " bytes, more than kMaxWeightVersionBytes=", kMaxWeightVersionBytes));
  }
  return WeightVersion(std::string(bytes));
}

absl::StatusOr<WeightVersion> WeightVersion::FromProto(
    const tpu_sync::rpc::WeightVersion& proto, absl::string_view field) {
  return FromBytes(proto.tag(), field);
}

tpu_sync::rpc::WeightVersion WeightVersion::ToProto() const {
  tpu_sync::rpc::WeightVersion proto;
  proto.set_tag(bytes_);
  return proto;
}

}  // namespace tpu_raiden::weight_sync

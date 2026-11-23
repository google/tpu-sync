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

#include "tpu_sync/weight_sync/host_tiling_validation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

using ::absl_testing::StatusIs;

TEST(CheckKnownHostTilingModeTest, AcceptsEveryKnownMode) {
  for (const tpu_sync::rpc::HostTilingMode mode :
       {tpu_sync::rpc::HOST_TILING_MODE_UNSPECIFIED,
        tpu_sync::rpc::HOST_TILING_MODE_DEFERRED,
        tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL,
        tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED}) {
    ABSL_EXPECT_OK(
        CheckKnownHostTilingMode("push (uuid=5)", "host_tiling_mode", mode))
        << tpu_sync::rpc::HostTilingMode_Name(mode);
  }
}

TEST(CheckKnownHostTilingModeTest, RejectsAnUnknownModeNamingItsValue) {
  for (const int unknown : {-1, 4, 7}) {
    EXPECT_THAT(
        CheckKnownHostTilingMode(
            "push (uuid=5)", "host_tiling_mode",
            static_cast<tpu_sync::rpc::HostTilingMode>(unknown)),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 absl::StrCat("push (uuid=5) sets unknown host_tiling_mode=",
                              unknown)));
  }
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden

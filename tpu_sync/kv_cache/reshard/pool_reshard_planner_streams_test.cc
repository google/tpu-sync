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

#include <cstdlib>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tpu_sync/kv_cache/reshard/pool_reshard_planner.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

class ScopedStreamsEnv {
 public:
  explicit ScopedStreamsEnv(const char* value) {
    if (const char* prev = std::getenv(kTransferStreamsEnvVar)) {
      previous_ = prev;
    }
    if (value == nullptr) {
      unsetenv(kTransferStreamsEnvVar);
    } else {
      setenv(kTransferStreamsEnvVar, value, 1);
    }
  }
  ~ScopedStreamsEnv() {
    if (previous_.has_value()) {
      setenv(kTransferStreamsEnvVar, previous_->c_str(), 1);
    } else {
      unsetenv(kTransferStreamsEnvVar);
    }
  }

 private:
  std::optional<std::string> previous_;
};

TEST(DefaultTransferStreamsTest, UnsetEmptyOrZeroKeepsOneStreamPerSourceRank) {
  {
    ScopedStreamsEnv env(nullptr);
    EXPECT_THAT(DefaultTransferStreams(8), IsOkAndHolds(8));
  }
  {
    ScopedStreamsEnv env("");
    EXPECT_THAT(DefaultTransferStreams(8), IsOkAndHolds(8));
  }
  {
    ScopedStreamsEnv env("0");
    EXPECT_THAT(DefaultTransferStreams(8), IsOkAndHolds(8));
  }
}

TEST(DefaultTransferStreamsTest, PositiveValueCapsStreamsWithinAdmission) {
  {
    ScopedStreamsEnv env("1");
    EXPECT_THAT(DefaultTransferStreams(8), IsOkAndHolds(1));
  }
  {
    ScopedStreamsEnv env("4");
    EXPECT_THAT(DefaultTransferStreams(8), IsOkAndHolds(4));
  }
  {
    ScopedStreamsEnv env("16");
    EXPECT_THAT(DefaultTransferStreams(8), IsOkAndHolds(8));
  }
}

TEST(DefaultTransferStreamsTest, RejectsUnparsableValueNamingVariableAndValue) {
  ScopedStreamsEnv env("many");
  EXPECT_THAT(DefaultTransferStreams(8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("TPU_RAIDEN_TRANSFER_STREAMS"),
                             HasSubstr("many"))));
}

TEST(DefaultTransferStreamsTest, RejectsNegativeValueNamingVariableAndValue) {
  ScopedStreamsEnv env("-2");
  EXPECT_THAT(DefaultTransferStreams(8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("TPU_RAIDEN_TRANSFER_STREAMS"),
                             HasSubstr("-2"))));
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

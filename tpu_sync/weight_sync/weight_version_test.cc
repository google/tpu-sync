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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/rpc/weight_transfer_common.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::StatusIs;

TEST(WeightVersionTest, FromBytesTakesOneToMaxBytes) {
  for (const std::string& bytes :
       {std::string("v"), std::string(kMaxWeightVersionBytes, 'x')}) {
    TF_ASSERT_OK_AND_ASSIGN(const WeightVersion version,
                         WeightVersion::FromBytes(bytes, "version"));
    EXPECT_EQ(version.ToProto().tag(), bytes);
  }
}

TEST(WeightVersionTest, FromBytesRejectsEmptyAndOverlongBytes) {
  EXPECT_THAT(
      WeightVersion::FromBytes("", "PrepareWeightReceive (uuid=3) version"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "PrepareWeightReceive (uuid=3) version is empty: the request "
               "carries no version"));
  EXPECT_THAT(
      WeightVersion::FromBytes(std::string(kMaxWeightVersionBytes + 1, 'x'),
                               "PrepareWeightReceive (uuid=3) version"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "PrepareWeightReceive (uuid=3) version has 257 bytes, more than "
               "kMaxWeightVersionBytes=256"));
}

TEST(WeightVersionTest, FromProtoRejectsAnUnsetAndAnOverlongTag) {
  EXPECT_THAT(WeightVersion::FromProto(tpu_sync::rpc::WeightVersion(),
                                       "PrepareWeightReceive version"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "PrepareWeightReceive version is empty: the request "
                       "carries no version"));
  tpu_sync::rpc::WeightVersion overlong;
  overlong.set_tag(std::string(kMaxWeightVersionBytes + 1, 'x'));
  EXPECT_THAT(
      WeightVersion::FromProto(overlong, "PrepareWeightReceive version"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "PrepareWeightReceive version has 257 bytes, more than "
               "kMaxWeightVersionBytes=256"));
}

TEST(WeightVersionTest, FromProtoReadsTheTagAndToProtoWritesIt) {
  tpu_sync::rpc::WeightVersion proto;
  proto.set_tag("step-7");
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion version,
                       WeightVersion::FromProto(proto, "version"));
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion step_7,
                       WeightVersion::FromBytes("step-7", "version"));

  EXPECT_EQ(version, step_7);
  EXPECT_EQ(version.ToProto().tag(), "step-7");
}

TEST(WeightVersionTest, EqualityIsBytewise) {
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion v7,
                       WeightVersion::FromBytes("v7", "version"));
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion also_v7,
                       WeightVersion::FromBytes("v7", "version"));
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion upper_v7,
                       WeightVersion::FromBytes("V7", "version"));
  TF_ASSERT_OK_AND_ASSIGN(
      const WeightVersion v7_nul,
      WeightVersion::FromBytes(std::string("v7\0", 3), "version"));

  EXPECT_EQ(v7, also_v7);
  EXPECT_NE(v7, upper_v7);
  EXPECT_NE(v7, v7_nul);
}

TEST(WeightVersionTest, PrintsTheLengthAndAtMost64EscapedBytes) {
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion short_version,
                       WeightVersion::FromBytes("v7", "version"));
  EXPECT_EQ(absl::StrCat(short_version), "version(len=2, \"v7\")");

  const std::string exactly_64(64, 'a');
  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion version_64,
                       WeightVersion::FromBytes(exactly_64, "version"));
  EXPECT_EQ(absl::StrCat(version_64),
            absl::StrCat("version(len=64, \"", exactly_64, "\")"));

  TF_ASSERT_OK_AND_ASSIGN(const WeightVersion long_version,
                       WeightVersion::FromBytes(exactly_64 + "bcd", "version"));
  EXPECT_EQ(absl::StrCat(long_version),
            absl::StrCat("version(len=67, \"", exactly_64, "\u2026\")"));

  TF_ASSERT_OK_AND_ASSIGN(
      const WeightVersion max_version,
      WeightVersion::FromBytes(std::string(kMaxWeightVersionBytes, 'x'),
                               "version"));
  EXPECT_EQ(
      absl::StrCat(max_version),
      absl::StrCat("version(len=256, \"", std::string(64, 'x'), "\u2026\")"));

  TF_ASSERT_OK_AND_ASSIGN(
      const WeightVersion binary,
      WeightVersion::FromBytes(std::string("\x00\xff\n", 3), "version"));
  EXPECT_EQ(absl::StrCat(binary), "version(len=3, \"\\x00\\xff\\n\")");
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

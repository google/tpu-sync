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

#include "tpu_sync/fault_injection/fault_injector.h"

#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::status::StatusIs;

constexpr std::string_view kTestHookAlpha = "test.hook.alpha";
constexpr std::string_view kTestHookBeta = "test.hook.beta";

class FaultInjectorTest : public ::testing::Test {
 protected:
  void SetUp() override { GetFaultInjector().Reset(); }
  void TearDown() override { GetFaultInjector().Reset(); }
};

TEST_F(FaultInjectorTest, DefaultStateIsInactive) {
  EXPECT_FALSE(FaultInjector::HasActiveInjections());
  EXPECT_FALSE(GetFaultInjector().IsHookActive(kTestHookAlpha));
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 0);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 0);
  EXPECT_THAT(GetFaultInjector().GetHitCounts(), IsEmpty());

  FaultInjectionAction action = GetFaultInjector().Evaluate(kTestHookAlpha);
  EXPECT_EQ(action.type, FaultInjectionType::kNone);
  EXPECT_EQ(action.delay_ms, 0);
}

TEST_F(FaultInjectorTest, EmptyHookNameInstallsForAllHooks) {
  ABSL_ASSERT_OK(GetFaultInjector().Install(
      {FaultInjectionRule{.hook = "",
                          .action = FaultInjectionType::kFail,
                          .probability = 1.0}}));
  EXPECT_TRUE(FaultInjector::HasActiveInjections());
  EXPECT_TRUE(GetFaultInjector().IsHookActive(kTestHookAlpha));
  EXPECT_TRUE(GetFaultInjector().IsHookActive(kTestHookBeta));

  EXPECT_EQ(GetFaultInjector().Evaluate(kTestHookAlpha).type,
            FaultInjectionType::kFail);
  EXPECT_EQ(GetFaultInjector().Evaluate(kTestHookBeta).type,
            FaultInjectionType::kFail);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 1);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookBeta), 1);
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 2);
}

TEST_F(FaultInjectorTest, DelayEligibilityRejectsEmptyAndFailOnlyHooks) {
  EXPECT_THAT(
      GetFaultInjector().Install({FaultInjectionRule{
          .hook = "",
          .action = FaultInjectionType::kDelay,
          .probability = 1.0,
          .max_delay_ms = 10,
      }}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("delay is not permitted at hook ''")));

  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = std::string(hooks::kTransferRecvSessionH2dComplete),
                  .action = FaultInjectionType::kDelay,
                  .probability = 1.0,
                  .max_delay_ms = 10,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("delay is not permitted at hook "
                                 "'transfer_recv_session.h2d.complete'")));
}

TEST_F(FaultInjectorTest, DefaultProbabilityZeroDoesNotTrigger) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
      },
  }));

  EXPECT_TRUE(FaultInjector::HasActiveInjections());
  EXPECT_TRUE(GetFaultInjector().IsHookActive(kTestHookAlpha));
  EXPECT_FALSE(GetFaultInjector().IsHookActive(kTestHookBeta));

  FaultInjectionAction action = GetFaultInjector().Evaluate(kTestHookAlpha);
  EXPECT_EQ(action.type, FaultInjectionType::kNone);
  EXPECT_EQ(action.delay_ms, 0);
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 0);
}

TEST_F(FaultInjectorTest, EvaluateSetsDelayMsOnlyForDelayAction) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
          .min_delay_ms = 100,
          .max_delay_ms = 100,
      },
      FaultInjectionRule{
          .hook = std::string(kTestHookBeta),
          .action = FaultInjectionType::kDelay,
          .probability = 1.0,
          .min_delay_ms = 50,
          .max_delay_ms = 50,
      },
  }));

  FaultInjectionAction fail_action =
      GetFaultInjector().Evaluate(kTestHookAlpha);
  EXPECT_EQ(fail_action.type, FaultInjectionType::kFail);
  EXPECT_EQ(fail_action.delay_ms, 0);

  FaultInjectionAction delay_action =
      GetFaultInjector().Evaluate(kTestHookBeta);
  EXPECT_EQ(delay_action.type, FaultInjectionType::kDelay);
  EXPECT_EQ(delay_action.delay_ms, 50);

  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 1);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookBeta), 1);
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 2);
  EXPECT_THAT(GetFaultInjector().GetHitCounts(),
              UnorderedElementsAre(Pair(std::string(kTestHookAlpha), 1),
                                   Pair(std::string(kTestHookBeta), 1)));
}

TEST_F(FaultInjectorTest, MinMaxDelayRangeAndValidation) {
  EXPECT_THAT(
      GetFaultInjector().Install({FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kDelay,
          .probability = 1.0,
          .min_delay_ms = 50,
          .max_delay_ms = 10,
      }}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("min_delay_ms cannot be greater than max_delay_ms")));

  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kDelay,
          .probability = 1.0,
          .min_delay_ms = 10,
          .max_delay_ms = 30,
      },
  }));

  for (int i = 0; i < 50; ++i) {
    FaultInjectionAction action = GetFaultInjector().Evaluate(kTestHookAlpha);
    EXPECT_EQ(action.type, FaultInjectionType::kDelay);
    EXPECT_GE(action.delay_ms, 10u);
    EXPECT_LE(action.delay_ms, 30u);
  }
}

TEST_F(FaultInjectorTest, MultipleRulesPerHookFallthrough) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kDelay,
          .probability = 0.0,
          .max_delay_ms = 25,
      },
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
  }));

  FaultInjectionAction action = GetFaultInjector().Evaluate(kTestHookAlpha);
  EXPECT_EQ(action.type, FaultInjectionType::kFail);
  EXPECT_EQ(action.delay_ms, 0);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 1);
}

TEST_F(FaultInjectorTest, FractionalProbabilitySampling) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 0.5,
      },
  }));

  for (int i = 0; i < 1000; ++i) {
    GetFaultInjector().Evaluate(kTestHookAlpha);
  }

  uint64_t hits = GetFaultInjector().GetHitCount(kTestHookAlpha);
  EXPECT_THAT(hits, Ge(350));
  EXPECT_THAT(hits, Le(650));
}

TEST_F(FaultInjectorTest, FaultInjectDelaySleepsOnlyOnDelayRule) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kDelay,
          .probability = 1.0,
          .min_delay_ms = 20,
          .max_delay_ms = 20,
      },
  }));

  absl::Time start = absl::Now();
  FaultInjectDelay(kTestHookAlpha);
  absl::Duration elapsed = absl::Now() - start;
  EXPECT_GE(elapsed, absl::Milliseconds(15));
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 1);

  // Calling FaultInjectStatus on a kDelay rule must NOT fail.
  ABSL_EXPECT_OK(FaultInjectStatus(kTestHookAlpha));
}

TEST_F(FaultInjectorTest, FaultInjectThrowThrowsOnlyOnFailRule) {
  EXPECT_NO_THROW(FaultInjectThrow(kTestHookAlpha));

  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
  }));

  EXPECT_THROW(FaultInjectThrow(kTestHookAlpha), std::runtime_error);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 1);
}

TEST_F(FaultInjectorTest, FaultInjectStatusReturnsInternalErrorOnFailRule) {
  ABSL_EXPECT_OK(FaultInjectStatus(kTestHookAlpha));

  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
  }));

  EXPECT_THAT(FaultInjectStatus(kTestHookAlpha),
              StatusIs(absl::StatusCode::kInternal, HasSubstr(kTestHookAlpha)));
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookAlpha), 1);
}

TEST_F(FaultInjectorTest, FaultInjectErrnoSetsCustomErrnoOnFailRule) {
  errno = 0;
  EXPECT_FALSE(FaultInjectErrno(kTestHookAlpha, ENOMEM));
  EXPECT_EQ(errno, 0);

  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
  }));

  errno = 0;
  EXPECT_TRUE(FaultInjectErrno(kTestHookAlpha));
  EXPECT_EQ(errno, EIO);

  errno = 0;
  EXPECT_TRUE(FaultInjectErrno(kTestHookAlpha, ENOMEM));
  EXPECT_EQ(errno, ENOMEM);
}

TEST_F(FaultInjectorTest, FastPathOverheadIsSubNanosecond) {
  constexpr int kIterations = 10'000'000;
  absl::Time start = absl::Now();
  int ok_count = 0;
  for (int i = 0; i < kIterations; ++i) {
    FaultInjectDelay(kTestHookAlpha);
    FaultInjectThrow(kTestHookBeta);
    if (FaultInjectStatus(kTestHookAlpha).ok() &&
        !FaultInjectErrno(kTestHookBeta, ENOMEM)) {
      ++ok_count;
    }
  }
  absl::Duration elapsed = absl::Now() - start;
  EXPECT_EQ(ok_count, kIterations);
  double ns_per_check =
      absl::ToDoubleNanoseconds(elapsed) / (4.0 * kIterations);
  EXPECT_LT(ns_per_check, 5.0) << "ns_per_check=" << ns_per_check;
}

}  // namespace
}  // namespace tpu_raiden

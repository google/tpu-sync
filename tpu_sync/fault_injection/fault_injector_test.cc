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

#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
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

constexpr std::string_view kTestHookAlpha =
    hooks::kTransferRecvSessionPullRequest;
constexpr std::string_view kTestHookBeta =
    hooks::kSocketTransportPushSendPayload;

// Long enough that only an interruption can end it within the test budget.
constexpr uint32_t kLongDelayMs = 60'000;
// Upper bound for an interrupted delay to return.
constexpr absl::Duration kPromptWake = absl::Seconds(1);

// Waits until `hook` has been hit once, i.e. a caller has entered its delay.
void WaitForHit(std::string_view hook) {
  const absl::Time deadline = absl::Now() + absl::Seconds(2);
  while (GetFaultInjector().GetHitCount(hook) == 0) {
    ASSERT_LT(absl::Now(), deadline) << "hook never hit: " << hook;
    absl::SleepFor(absl::Milliseconds(1));
  }
}

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

TEST_F(FaultInjectorTest, StarPatternInstallsForAllHooks) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({FaultInjectionRule{
      .hook = "*", .action = FaultInjectionType::kFail, .probability = 1.0}}));
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

TEST_F(FaultInjectorTest, PrefixPatternMatchesOnlyPrefixedHooks) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({FaultInjectionRule{
      .hook = "grpc_control_plane.*",
      .action = FaultInjectionType::kFail,
      .probability = 1.0,
  }}));

  EXPECT_EQ(GetFaultInjector()
                .Evaluate(hooks::kGrpcControlPlanePullStreamSendRequest)
                .type,
            FaultInjectionType::kFail);
  EXPECT_EQ(GetFaultInjector()
                .Evaluate(hooks::kGrpcControlPlanePullStreamSendReply)
                .type,
            FaultInjectionType::kFail);
  EXPECT_EQ(GetFaultInjector().Evaluate(hooks::kConnPoolBorrowConnect).type,
            FaultInjectionType::kNone);
}

TEST_F(FaultInjectorTest, PatternWithoutCapableHookIsRejected) {
  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = "nonexistent.*",
                  .action = FaultInjectionType::kFail,
                  .probability = 1.0,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pattern 'nonexistent.*' matches no hook")));
  // A delay pattern over fail-only hooks matches nothing.
  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = "staging_allocator.*",
                  .action = FaultInjectionType::kDelay,
                  .probability = 1.0,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = "",
                  .action = FaultInjectionType::kFail,
                  .probability = 1.0,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown hook ''")));
}

TEST_F(FaultInjectorTest, WildcardDelayFiresOnlyAtDelayCapableHooks) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({FaultInjectionRule{
      .hook = "*",
      .action = FaultInjectionType::kDelay,
      .probability = 1.0,
      .min_delay_ms = 20,
      .max_delay_ms = 20,
  }}));

  FaultInjectionAction action = GetFaultInjector().Evaluate(kTestHookBeta);
  EXPECT_EQ(action.type, FaultInjectionType::kDelay);
  EXPECT_EQ(action.delay_ms, 20);

  // Fail-only hook: the wildcard delay does not apply and is not counted.
  action = GetFaultInjector().Evaluate(hooks::kTransferRecvSessionH2dComplete);
  EXPECT_EQ(action.type, FaultInjectionType::kNone);
  EXPECT_EQ(
      GetFaultInjector().GetHitCount(hooks::kTransferRecvSessionH2dComplete),
      0);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookBeta), 1);
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 1);
}

TEST_F(FaultInjectorTest, WildcardFailSkipsDelayOnlyHook) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({FaultInjectionRule{
      .hook = "*",
      .action = FaultInjectionType::kFail,
      .probability = 1.0,
  }}));

  FaultInjectionAction action =
      GetFaultInjector().Evaluate(hooks::kKvCacheManagerPullRegisterWait);
  EXPECT_EQ(action.type, FaultInjectionType::kNone);
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 0);
}

TEST_F(FaultInjectorTest, WildcardFallsThroughToCapableRule) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = "*",
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
      FaultInjectionRule{
          .hook = "*",
          .action = FaultInjectionType::kDelay,
          .probability = 1.0,
          .min_delay_ms = 5,
          .max_delay_ms = 5,
      },
  }));

  EXPECT_EQ(
      GetFaultInjector().Evaluate(hooks::kKvCacheManagerPullRegisterWait).type,
      FaultInjectionType::kDelay);
  EXPECT_EQ(GetFaultInjector().Evaluate(hooks::kConnPoolBorrowReuse).type,
            FaultInjectionType::kFail);
  EXPECT_EQ(GetFaultInjector().GetHitCount(), 2);
}

TEST_F(FaultInjectorTest, DelayEligibilityRejectsFailOnlyHooks) {
  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = std::string(hooks::kTransferRecvSessionH2dComplete),
                  .action = FaultInjectionType::kDelay,
                  .probability = 1.0,
                  .max_delay_ms = 10,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("delay is not permitted at hook "
                                 "'transfer_recv_session.h2d.complete'")));

  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = "unknown.hook",
                  .action = FaultInjectionType::kFail,
                  .probability = 1.0,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown hook 'unknown.hook'")));

  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = std::string(hooks::kKvCacheManagerPullRegisterWait),
                  .action = FaultInjectionType::kFail,
                  .probability = 1.0,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("fail is not permitted at hook "
                                 "'kv_cache_manager.pull.register_wait'")));

  EXPECT_THAT(GetFaultInjector().Install({FaultInjectionRule{
                  .hook = std::string(kTestHookAlpha),
                  .action = FaultInjectionType::kFail,
                  .probability = -0.1,
              }}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("probability must be in [0, 1]")));
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

TEST_F(FaultInjectorTest, FaultInjectStatusUsesCallerStatusCode) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
  }));

  EXPECT_THAT(
      FaultInjectStatus(kTestHookAlpha, absl::StatusCode::kUnavailable),
      StatusIs(absl::StatusCode::kUnavailable, HasSubstr(kTestHookAlpha)));
}

TEST_F(FaultInjectorTest, FaultInjectStatusRejectsOkCode) {
  ABSL_ASSERT_OK(GetFaultInjector().Install({
      FaultInjectionRule{
          .hook = std::string(kTestHookAlpha),
          .action = FaultInjectionType::kFail,
          .probability = 1.0,
      },
  }));

  EXPECT_DEATH(
      FaultInjectStatus(kTestHookAlpha, absl::StatusCode::kOk).IgnoreError(),
      "must be an error");
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

class FaultInjectSocketTest : public FaultInjectorTest {
 protected:
  void SetUp() override {
    FaultInjectorTest::SetUp();
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds_), 0);
  }
  void TearDown() override {
    FaultInjectorTest::TearDown();
    for (int fd : fds_) {
      if (fd >= 0) ::close(fd);
    }
  }

  void InstallSocketRule(FaultInjectionType action, uint32_t delay_ms = 0) {
    ABSL_ASSERT_OK(GetFaultInjector().Install({FaultInjectionRule{
        .hook = std::string(kTestHookBeta),
        .action = action,
        .probability = 1.0,
        .min_delay_ms = delay_ms,
        .max_delay_ms = delay_ms,
    }}));
  }

  // fds_[0] is the injected side; fds_[1] is the peer.
  int fds_[2] = {-1, -1};
};

TEST_F(FaultInjectSocketTest, FailShutsDownConnection) {
  InstallSocketRule(FaultInjectionType::kFail);

  FaultInjectSocket(kTestHookBeta, fds_[0]);

  char c = 0;
  EXPECT_EQ(::read(fds_[1], &c, 1), 0);
  errno = 0;
  EXPECT_EQ(::send(fds_[0], "x", 1, MSG_NOSIGNAL), -1);
  EXPECT_EQ(errno, EPIPE);
  EXPECT_EQ(GetFaultInjector().GetHitCount(kTestHookBeta), 1);
}

TEST_F(FaultInjectSocketTest, DelayEndsOnPeerClose) {
  InstallSocketRule(FaultInjectionType::kDelay, kLongDelayMs);

  absl::Notification done;
  std::thread waiter([this, &done] {
    FaultInjectSocket(kTestHookBeta, fds_[0]);
    done.Notify();
  });
  WaitForHit(kTestHookBeta);

  ::close(fds_[1]);
  fds_[1] = -1;
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(kPromptWake));
  GetFaultInjector().Reset();
  waiter.join();
}

TEST_F(FaultInjectSocketTest, DelayEndsOnLocalShutdown) {
  InstallSocketRule(FaultInjectionType::kDelay, kLongDelayMs);

  absl::Notification done;
  std::thread waiter([this, &done] {
    FaultInjectSocket(kTestHookBeta, fds_[0]);
    done.Notify();
  });
  WaitForHit(kTestHookBeta);

  ::shutdown(fds_[0], SHUT_RDWR);
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(kPromptWake));
  GetFaultInjector().Reset();
  waiter.join();
}

TEST_F(FaultInjectSocketTest, DelayIgnoresReadableData) {
  constexpr uint32_t kDelayMs = 200;
  InstallSocketRule(FaultInjectionType::kDelay, kDelayMs);
  ASSERT_EQ(::send(fds_[1], "x", 1, MSG_NOSIGNAL), 1);

  const absl::Time start = absl::Now();
  FaultInjectSocket(kTestHookBeta, fds_[0]);
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(kDelayMs - 20));
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

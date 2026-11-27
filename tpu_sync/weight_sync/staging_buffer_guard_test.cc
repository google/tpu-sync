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

#include "tpu_sync/weight_sync/staging_buffer_guard.h"

#include <cstdint>
#include <optional>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Optional;
using ReadPin = StagingBufferGuard::ReadPin;

template <typename... S>
auto Aborted(const S&... substrs) {
  return StatusIs(absl::StatusCode::kAborted, AllOf(HasSubstr(substrs)...));
}
template <typename... S>
auto FailedPrecondition(const S&... substrs) {
  return StatusIs(absl::StatusCode::kFailedPrecondition,
                  AllOf(HasSubstr(substrs)...));
}
template <typename... S>
auto DeadlineExceeded(const S&... substrs) {
  return StatusIs(absl::StatusCode::kDeadlineExceeded,
                  AllOf(HasSubstr(substrs)...));
}
template <typename... S>
auto NotFound(const S&... substrs) {
  return StatusIs(absl::StatusCode::kNotFound, AllOf(HasSubstr(substrs)...));
}

WeightVersion MakeVersion(absl::string_view tag) {
  absl::StatusOr<WeightVersion> version =
      WeightVersion::FromBytes(tag, "version");
  CHECK_OK(version);
  return *std::move(version);
}

tpu_sync::rpc::ShardExtent MakeExtent(int32_t layer_idx, int64_t global_shard,
                                      int64_t device_bytes) {
  tpu_sync::rpc::ShardExtent extent;
  extent.set_layer_idx(layer_idx);
  extent.set_global_shard(global_shard);
  extent.set_device_bytes(device_bytes);
  return extent;
}

struct BackgroundStatus {
  absl::Notification done;
  absl::Status status;
  std::thread thread;
  ~BackgroundStatus() {
    if (thread.joinable()) thread.join();
  }
  bool IsBlocked() {
    return !done.WaitForNotificationWithTimeout(absl::Milliseconds(10));
  }
  absl::Status Join() {
    thread.join();
    return status;
  }
};

struct AwaitHandle : BackgroundStatus {
  AwaitHandle(StagingBufferGuard& guard, uint64_t uuid, absl::Time deadline) {
    absl::Notification started;
    thread = std::thread([this, &guard, &started, uuid, deadline] {
      started.Notify();
      status = guard.AwaitReady(uuid, deadline);
      done.Notify();
    });
    started.WaitForNotification();
  }
};

// Blocks until staged_ is dropped, proving BeginLegacyWrite entered its wait.
struct LegacyWaitHandle : BackgroundStatus {
  LegacyWaitHandle(StagingBufferGuard& guard, uint64_t uuid, uint64_t epoch) {
    thread = std::thread([this, &guard, uuid, epoch] {
      status = guard.BeginLegacyWrite(uuid, epoch);
      done.Notify();
    });
    while (guard.staged_version().has_value() && !done.HasBeenNotified()) {
      std::this_thread::yield();
    }
  }
};

TEST(StagingBufferGuardTest, ClaimSupersedesAndAdvancesEpoch) {
  StagingBufferGuard guard;
  EXPECT_EQ(guard.claim_epoch(), 0);
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_EQ(guard.claim_epoch(), 1);
  // Pre-wait order: ready claim returns OK even at a past deadline.
  ABSL_EXPECT_OK(guard.AwaitReady(1, absl::InfinitePast()));

  EXPECT_EQ(guard.Claim(2), 1);
  EXPECT_EQ(guard.claim_epoch(), 2);
  // AbortClaim ignores non-owners, keeps claim_epoch_, and is idempotent.
  guard.AbortClaim(1);
  ABSL_EXPECT_OK(guard.AwaitReady(2, absl::InfinitePast()));
  guard.AbortClaim(2);
  EXPECT_EQ(guard.claim_epoch(), 2);
  guard.AbortClaim(2);
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_EQ(guard.claim_epoch(), 3);
}

TEST(StagingBufferGuardTest, AwaitReadyPreWaitChecksOwnerBeforeDeadline) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.AwaitReady(1, absl::InfinitePast()),
              Aborted("uuid=1", "no active claim"));
  EXPECT_EQ(guard.Claim(2), 0);
  EXPECT_THAT(guard.AwaitReady(1, absl::InfinitePast()),
              Aborted("uuid=1", "uuid=2"));
  // Aborting a ready claim causes AwaitReady to return Aborted.
  guard.AbortClaim(2);
  EXPECT_THAT(guard.AwaitReady(2, absl::InfiniteFuture()), Aborted("uuid=2"));
}

TEST(StagingBufferGuardTest, AwaitReadyWaitsForLegacyWritesAndWakesJoiners) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginLegacyWrite(0, 0));
  ABSL_ASSERT_OK(guard.BeginLegacyWrite(9, 0));
  EXPECT_EQ(guard.Claim(1), 0);

  AwaitHandle joiner1(guard, 1, absl::InfiniteFuture());
  AwaitHandle joiner2(guard, 1, absl::InfiniteFuture());
  guard.EndLegacyWrite();
  EXPECT_TRUE(joiner1.IsBlocked());
  EXPECT_FALSE(joiner2.done.HasBeenNotified());
  EXPECT_THAT(guard.AwaitReady(1, absl::InfinitePast()),
              DeadlineExceeded("legacy writes in flight: 1"));

  guard.EndLegacyWrite();
  ABSL_EXPECT_OK(joiner1.Join());
  ABSL_EXPECT_OK(joiner2.Join());
}

TEST(StagingBufferGuardTest,
     AwaitReadyWakesJoinersOnAbortSupersedeAndFailedCommit) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginLegacyWrite(0, 0));

  EXPECT_EQ(guard.Claim(1), 0);
  AwaitHandle w1(guard, 1, absl::InfiniteFuture());
  AwaitHandle w2(guard, 1, absl::InfiniteFuture());
  EXPECT_TRUE(w1.IsBlocked());
  EXPECT_TRUE(w2.IsBlocked());
  guard.AbortClaim(1);
  EXPECT_THAT(w1.Join(), Aborted("uuid=1"));
  EXPECT_THAT(w2.Join(), Aborted("uuid=1"));

  EXPECT_EQ(guard.Claim(2), 0);
  AwaitHandle w3(guard, 2, absl::InfiniteFuture());
  EXPECT_TRUE(w3.IsBlocked());
  EXPECT_EQ(guard.Claim(3), 2);
  EXPECT_THAT(w3.Join(), Aborted("uuid=2", "uuid=3"));

  AwaitHandle w4(guard, 3, absl::InfiniteFuture());
  EXPECT_TRUE(w4.IsBlocked());
  EXPECT_THAT(guard.CommitAndPin(3, MakeVersion("v1"), {}),
              FailedPrecondition("uuid=3", "legacy writes in flight: 1"));
  EXPECT_THAT(w4.Join(), Aborted("uuid=3"));
  guard.EndLegacyWrite();
}

TEST(StagingBufferGuardTest, AwaitReadyTimeoutDoesNotMutateClaim) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginLegacyWrite(0, 0));
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_THAT(guard.AwaitReady(1, absl::Now() + absl::Milliseconds(10)),
              DeadlineExceeded("uuid=1", "legacy writes in flight: 1"));
  // Claim 1 remains active and draining after timeout.
  EXPECT_THAT(guard.AwaitReady(1, absl::InfinitePast()),
              DeadlineExceeded("uuid=1", "legacy writes in flight: 1"));
  EXPECT_THAT(guard.BeginLegacyWrite(0, guard.claim_epoch()),
              FailedPrecondition("uuid=1"));
  guard.EndLegacyWrite();
  ABSL_EXPECT_OK(guard.AwaitReady(1, absl::InfinitePast()));
  ABSL_EXPECT_OK(guard.CommitAndPin(1, MakeVersion("v1"), {}));
}

TEST(StagingBufferGuardTest, AwaitReadyTimeoutNamesLegacyWritesAndPins) {
  StagingBufferGuard guard;
  const WeightVersion v1 = MakeVersion("v1");
  ABSL_ASSERT_OK(guard.BeginLegacyWrite(0, 0));
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_THAT(guard.AwaitReady(1, absl::InfinitePast()),
              DeadlineExceeded("uuid=1", "legacy writes in flight: 1",
                               "read pins: none"));
  guard.EndLegacyWrite();

  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin, guard.CommitAndPin(1, v1, {}));
  EXPECT_EQ(guard.Claim(2), 0);
  EXPECT_THAT(guard.AwaitReady(2, absl::InfinitePast()),
              DeadlineExceeded(
                  "uuid=2", "legacy writes in flight: 0",
                  absl::StrCat("read pins: pin ", pin.pin_id, " of ", v1)));
  guard.ReleaseReadPin(pin.pin_id);
}

TEST(StagingBufferGuardTest, CommitAndPinStagesVersionAndManagesReadPins) {
  StagingBufferGuard guard;
  const WeightVersion v1 = MakeVersion("v1");
  const WeightVersion v2 = MakeVersion("v2");
  EXPECT_THAT(guard.TryAcquireReadPin(v1), NotFound("v1", "no version"));

  EXPECT_EQ(guard.Claim(1), 0);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin1,
                       guard.CommitAndPin(1, v1, {MakeExtent(0, 3, 4096)}));
  EXPECT_GT(pin1.pin_id, 0);
  ASSERT_NE(pin1.commit, nullptr);
  EXPECT_EQ(pin1.commit->version, v1);
  ASSERT_EQ(pin1.commit->extents.size(), 1);
  EXPECT_EQ(pin1.commit->extents[0].layer_idx(), 0);
  EXPECT_EQ(pin1.commit->extents[0].global_shard(), 3);
  EXPECT_EQ(pin1.commit->extents[0].device_bytes(), 4096);
  EXPECT_THAT(guard.staged_version(), Optional(v1));
  EXPECT_THAT(guard.AwaitReady(1, absl::InfinitePast()),
              Aborted("uuid=1", "no active claim"));

  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin2, guard.TryAcquireReadPin(v1));
  EXPECT_GT(pin2.pin_id, pin1.pin_id);
  EXPECT_EQ(pin2.commit, pin1.commit);
  EXPECT_THAT(guard.TryAcquireReadPin(v2), NotFound("v2", "v1"));

  // Next Claim() drops staged_ and drains existing pins.
  EXPECT_EQ(guard.Claim(2), 0);
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  EXPECT_THAT(guard.TryAcquireReadPin(v1), NotFound("no version"));
  EXPECT_THAT(guard.AwaitReady(2, absl::InfinitePast()),
              DeadlineExceeded(absl::StrCat("pin ", pin1.pin_id),
                               absl::StrCat("pin ", pin2.pin_id)));
  // ReleaseReadPin ignores unknown IDs and is idempotent.
  guard.ReleaseReadPin(0);
  guard.ReleaseReadPin(999);
  guard.ReleaseReadPin(pin1.pin_id);
  guard.ReleaseReadPin(pin1.pin_id);
  EXPECT_THAT(guard.AwaitReady(2, absl::InfinitePast()),
              DeadlineExceeded(absl::StrCat("pin ", pin2.pin_id)));
  guard.ReleaseReadPin(pin2.pin_id);
  ABSL_EXPECT_OK(guard.AwaitReady(2, absl::InfinitePast()));
}

TEST(StagingBufferGuardTest, CommitAndPinRefusesInvalidStates) {
  StagingBufferGuard guard;
  const WeightVersion v1 = MakeVersion("v1");
  EXPECT_THAT(guard.CommitAndPin(1, v1, {}),
              FailedPrecondition("uuid=1", "no active claim"));
  // Non-owner does not mutate the active claim.
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_THAT(guard.CommitAndPin(2, v1, {}),
              FailedPrecondition("uuid=2", "uuid=1"));
  ABSL_EXPECT_OK(guard.AwaitReady(1, absl::InfinitePast()));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  guard.AbortClaim(1);

  // Unready claim (legacy write in flight) clears the claim, stages nothing,
  // and unblocks legacy writes.
  ABSL_ASSERT_OK(guard.BeginLegacyWrite(0, guard.claim_epoch()));
  EXPECT_EQ(guard.Claim(2), 0);
  EXPECT_THAT(guard.CommitAndPin(2, v1, {}),
              FailedPrecondition("uuid=2", "legacy writes in flight: 1"));
  EXPECT_THAT(guard.AwaitReady(2, absl::InfinitePast()),
              Aborted("uuid=2", "no active claim"));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  ABSL_EXPECT_OK(guard.BeginLegacyWrite(0, guard.claim_epoch()));
  guard.EndLegacyWrite();
  guard.EndLegacyWrite();

  // Unready solely due to a held ReadPin clears the claim and refuses.
  EXPECT_EQ(guard.Claim(3), 0);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin, guard.CommitAndPin(3, v1, {}));
  EXPECT_EQ(guard.Claim(4), 0);
  EXPECT_THAT(guard.CommitAndPin(4, v1, {}),
              FailedPrecondition("uuid=4", "legacy writes in flight: 0",
                                 absl::StrCat("pin ", pin.pin_id)));
  EXPECT_THAT(guard.AwaitReady(4, absl::InfinitePast()),
              Aborted("uuid=4", "no active claim"));
  guard.ReleaseReadPin(pin.pin_id);
}

TEST(StagingBufferGuardTest, BeginLegacyWriteRefusesClaimAndStaleEpoch) {
  StagingBufferGuard guard;
  const WeightVersion v1 = MakeVersion("v1");
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_THAT(guard.BeginLegacyWrite(0, guard.claim_epoch()),
              FailedPrecondition("uuid=0", "uuid=1"));
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin, guard.CommitAndPin(1, v1, {}));
  guard.ReleaseReadPin(pin.pin_id);
  // Stale epoch refuses without dropping staged_.
  EXPECT_THAT(guard.BeginLegacyWrite(7, /*expected_claim_epoch=*/0),
              FailedPrecondition("uuid=7", "0", "1"));
  EXPECT_THAT(guard.staged_version(), Optional(v1));

  const uint64_t epoch = guard.claim_epoch();
  ABSL_EXPECT_OK(guard.BeginLegacyWrite(7, epoch));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  ABSL_EXPECT_OK(guard.BeginLegacyWrite(0, epoch));
  guard.EndLegacyWrite();
  guard.EndLegacyWrite();
}

TEST(StagingBufferGuardTest, BeginLegacyWriteWaitsForPinAndTimesOut) {
  StagingBufferGuard zero_wait(absl::ZeroDuration());
  ABSL_EXPECT_OK(zero_wait.BeginLegacyWrite(0, 0));
  zero_wait.EndLegacyWrite();

  StagingBufferGuard guard(/*legacy_write_pin_wait=*/absl::Milliseconds(20));
  const WeightVersion v1 = MakeVersion("v1");
  EXPECT_EQ(guard.Claim(1), 0);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin1, guard.CommitAndPin(1, v1, {}));
  EXPECT_THAT(
      guard.BeginLegacyWrite(5, guard.claim_epoch()),
      DeadlineExceeded("uuid=5", absl::StrCat("pin ", pin1.pin_id), "v1"));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  guard.ReleaseReadPin(pin1.pin_id);
  EXPECT_EQ(guard.Claim(2), 0);
  ABSL_EXPECT_OK(guard.AwaitReady(2, absl::InfinitePast()));

  // Drops staged_ before waiting (no pin starvation) and admits on release.
  StagingBufferGuard wait_guard;
  EXPECT_EQ(wait_guard.Claim(2), 0);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin, wait_guard.CommitAndPin(2, v1, {}));
  LegacyWaitHandle writer(wait_guard, 0, wait_guard.claim_epoch());
  EXPECT_THAT(wait_guard.TryAcquireReadPin(v1), NotFound("no version"));
  EXPECT_TRUE(writer.IsBlocked());
  wait_guard.ReleaseReadPin(pin.pin_id);
  ABSL_EXPECT_OK(writer.Join());
  wait_guard.EndLegacyWrite();
}

TEST(StagingBufferGuardTest, BeginLegacyWriteWakesOnMidWaitClaimAndAba) {
  StagingBufferGuard guard;
  const WeightVersion v1 = MakeVersion("v1");
  const WeightVersion v2 = MakeVersion("v2");
  EXPECT_EQ(guard.Claim(1), 0);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin1, guard.CommitAndPin(1, v1, {}));
  LegacyWaitHandle writer1(guard, 0, guard.claim_epoch());
  EXPECT_EQ(guard.Claim(2), 0);
  EXPECT_THAT(writer1.Join(), FailedPrecondition("uuid=0", "uuid=2"));
  guard.AbortClaim(2);
  guard.ReleaseReadPin(pin1.pin_id);
  // ABA mid-wait: Claim(4) advances claim_epoch_ and CommitAndPin(v2) clears
  // the claim again, so writer2 refuses (claimed or epoch mismatch) without
  // dropping v2 or leaking a write.
  EXPECT_EQ(guard.Claim(3), 0);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin2, guard.CommitAndPin(3, v1, {}));
  const uint64_t epoch2 = guard.claim_epoch();
  LegacyWaitHandle writer2(guard, 0, epoch2);
  EXPECT_EQ(guard.Claim(4), 0);
  guard.ReleaseReadPin(pin2.pin_id);
  TF_ASSERT_OK_AND_ASSIGN(ReadPin pin3, guard.CommitAndPin(4, v2, {}));
  guard.ReleaseReadPin(pin3.pin_id);
  EXPECT_THAT(writer2.Join(),
              FailedPrecondition("uuid=0", absl::StrCat(epoch2),
                                 absl::StrCat(guard.claim_epoch())));
  EXPECT_THAT(guard.staged_version(), Optional(v2));
  EXPECT_EQ(guard.Claim(5), 0);
  ABSL_EXPECT_OK(guard.AwaitReady(5, absl::InfinitePast()));
}

TEST(StagingBufferGuardDeathTest, InvalidCallsDie) {
  EXPECT_DEATH(StagingBufferGuard(absl::Milliseconds(-1)),
               "legacy_write_pin_wait must be non-negative");
  StagingBufferGuard guard;
  EXPECT_DEATH(guard.Claim(0), "non-zero uuid");
  EXPECT_EQ(guard.Claim(1), 0);
  EXPECT_DEATH(guard.Claim(1), "already holds the active claim");
  EXPECT_DEATH(
      guard.EndLegacyWrite(),
      "EndLegacyWrite\\(\\) without a matching BeginLegacyWrite\\(\\)");
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

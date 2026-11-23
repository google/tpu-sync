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
#include <memory>
#include <optional>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;
using EndedClaim = StagingBufferGuard::EndedClaim;
using ReadPin = StagingBufferGuard::ReadPin;

auto EndedWith(uint64_t uuid, absl::StatusCode code) {
  return AllOf(Field(&EndedClaim::uuid, uuid),
               Field(&EndedClaim::reason, StatusIs(code)));
}

class StagingBufferGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(v1_, WeightVersion::FromBytes("v1", "version"));
    TF_ASSERT_OK_AND_ASSIGN(v2_, WeightVersion::FromBytes("v2", "version"));
  }

  WeightVersion v1_;
  WeightVersion v2_;
};

// Claims `uuid` of `version` and waits until the claim is ready.
absl::Status ClaimAndAwait(StagingBufferGuard& guard, uint64_t uuid,
                           const WeightVersion& version) {
  ABSL_RETURN_IF_ERROR(guard.Claim(uuid, version).status());
  return guard.AwaitReady(uuid);
}

// Returns once the guard refuses the writes of `uuid`, ending each write it
// admitted before.
void AwaitWritesRefused(StagingBufferGuard& guard, uint64_t uuid) {
  while (guard.BeginWrite(uuid).ok()) {
    guard.EndWrite(uuid);
    std::this_thread::yield();
  }
}

TEST_F(StagingBufferGuardTest, CommitStagesTheVersionUntilALegacyWrite) {
  StagingBufferGuard guard;
  EXPECT_EQ(guard.staged_version(), std::nullopt);

  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  ABSL_ASSERT_OK(guard.BeginWrite(1));
  guard.EndWrite(1);
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin pin, guard.CommitReceiveAndPin(1, {}));
  EXPECT_THAT(guard.staged_version(), Optional(v1_));

  // The commit ended the claim, so the next write is a legacy write.
  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("uuid=1 refused: it holds no claim")));
  guard.ReleaseReadPin(pin.pin_id);
  ABSL_ASSERT_OK(guard.BeginWrite(0));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  guard.EndWrite(0);
}

TEST_F(StagingBufferGuardTest, NewClaimDropsTheStagedVersion) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin pin, guard.CommitReceiveAndPin(1, {}));
  ASSERT_THAT(guard.staged_version(), Optional(v1_));
  guard.ReleaseReadPin(pin.pin_id);

  ABSL_ASSERT_OK(ClaimAndAwait(guard, 2, v2_));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  ABSL_ASSERT_OK(guard.BeginWrite(2));
  guard.EndWrite(2);
  ABSL_ASSERT_OK(guard.CommitReceiveAndPin(2, {}));
  EXPECT_THAT(guard.staged_version(), Optional(v2_));
}

TEST_F(StagingBufferGuardTest, PinsGetTheStagedCommitOfTheirVersion) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.TryAcquireReadPin(v1_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("the staging buffer holds no version")));
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin first, guard.CommitReceiveAndPin(1, {}));
  guard.ReleaseReadPin(first.pin_id);
  // A receive of the same bytes, held mid-way with its write in flight: no
  // read pin returns the old commit, which it overwrites, or its own.
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 2, v1_));
  ABSL_ASSERT_OK(guard.BeginWrite(2));
  EXPECT_THAT(guard.TryAcquireReadPin(v1_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("the staging buffer holds no version")));
  guard.EndWrite(2);
  tpu_sync::rpc::ShardExtent extent;
  extent.set_layer_idx(3);
  extent.set_layout(tpu_sync::rpc::STAGING_LAYOUT_DEVICE);
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin second,
                       guard.CommitReceiveAndPin(2, {extent}));
  guard.ReleaseReadPin(second.pin_id);

  ASSERT_NE(second.commit, nullptr);
  EXPECT_EQ(second.commit->version, v1_);
  EXPECT_THAT(second.commit->extents, ElementsAre(EqualsProto(extent)));
  EXPECT_NE(second.commit, first.commit);
  EXPECT_THAT(guard.TryAcquireReadPin(v2_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Read pin of version(len=2, \"v2\") refused: "
                                 "the staging buffer holds version(len=2, "
                                 "\"v1\")")));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin by_version, guard.TryAcquireReadPin(v1_));
  EXPECT_EQ(by_version.commit, second.commit);
  guard.ReleaseReadPin(by_version.pin_id);
}

TEST_F(StagingBufferGuardTest, LegacyWriteWaitsForEveryReadPin) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin committed,
                       guard.CommitReceiveAndPin(1, {}));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin copy, guard.TryAcquireReadPin(v1_));
  absl::StatusOr<StagingBufferGuard::WriteAdmission> written;
  absl::Notification returned;
  std::thread writer([&] {
    written = guard.BeginWrite(0);
    returned.Notify();
  });
  // Waiting, the write dropped the staged version, so no new pins are taken.
  while (guard.staged_version().has_value()) std::this_thread::yield();
  EXPECT_THAT(guard.TryAcquireReadPin(v1_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("the staging buffer holds no version")));

  guard.ReleaseReadPin(committed.pin_id);
  EXPECT_FALSE(returned.HasBeenNotified());  // It still waits for the copy.
  guard.ReleaseReadPin(copy.pin_id);
  returned.WaitForNotification();
  writer.join();
  ASSERT_THAT(written,
              IsOkAndHolds(StagingBufferGuard::WriteAdmission::kUnclaimed));
  guard.EndWrite(0);
}

TEST_F(StagingBufferGuardTest, LegacyWriteFailsIfPinsOutlastItsWait) {
  StagingBufferGuard guard(kVersionedReceiveClaimTimeout,
                           /*legacy_write_pin_wait=*/absl::Milliseconds(10));
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin copy, guard.CommitReceiveAndPin(1, {}));

  EXPECT_THAT(guard.BeginWrite(0),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Write of uuid=0 refused: read pins still "
                                 "held after 10ms: pin 1 on "
                                 "version(len=2, \"v1\")")));
  // It dropped the version, so no new pins were taken.
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  guard.ReleaseReadPin(copy.pin_id);
  ABSL_ASSERT_OK(guard.BeginWrite(0));
  guard.EndWrite(0);
}

TEST_F(StagingBufferGuardTest, ClaimRefusesALegacyWriteWaitingForReadPins) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin copy, guard.CommitReceiveAndPin(1, {}));
  absl::Status written;
  std::thread writer([&] { written = guard.BeginWrite(0).status(); });
  // Waiting, the write dropped the staged version.
  while (guard.staged_version().has_value()) std::this_thread::yield();

  EXPECT_THAT(guard.Claim(2, v2_),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  writer.join();
  EXPECT_THAT(written,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Write of uuid=0 refused: receive uuid=2 of "
                                 "version(len=2, \"v2\") claims")));
  // The claim gets ready once the pin ends.
  guard.ReleaseReadPin(copy.pin_id);
  ABSL_EXPECT_OK(guard.AwaitReady(2));
}

TEST_F(StagingBufferGuardTest, ClaimWaitsForReadPins) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin committed,
                       guard.CommitReceiveAndPin(1, {}));
  guard.ReleaseReadPin(committed.pin_id);
  guard.ReleaseReadPin(committed.pin_id);  // Does nothing.
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin copy, guard.TryAcquireReadPin(v1_));
  absl::Status claimed;
  absl::Notification returned;
  std::thread receiver([&] {
    claimed = ClaimAndAwait(guard, 2, v2_);
    returned.Notify();
  });
  // Installed, the claim drops the staged version and waits for the copy.
  while (guard.staged_version().has_value()) std::this_thread::yield();

  EXPECT_THAT(guard.BeginWrite(2),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("(writes: none; read pins: pin 2 on "
                                 "version(len=2, \"v1\"))")));
  EXPECT_FALSE(returned.HasBeenNotified());
  guard.ReleaseReadPin(copy.pin_id);
  returned.WaitForNotification();
  receiver.join();
  ABSL_EXPECT_OK(claimed);
}

TEST_F(StagingBufferGuardTest, CommitBeforeTheClaimIsReadyAbortsIt) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin copy, guard.CommitReceiveAndPin(1, {}));
  absl::Status claimed;
  std::thread receiver([&] { claimed = ClaimAndAwait(guard, 2, v2_); });
  // Installed, the claim drops the staged version and waits for the copy.
  while (guard.staged_version().has_value()) std::this_thread::yield();

  EXPECT_THAT(guard.CommitReceiveAndPin(2, {}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("(writes: none; read pins: pin 1 on "
                                 "version(len=2, \"v1\"))")));
  receiver.join();
  EXPECT_THAT(claimed,
              StatusIs(absl::StatusCode::kAborted,
                       AllOf(HasSubstr("receive uuid=2 of version(len=2, "
                                       "\"v2\") was aborted: "),
                             HasSubstr("FAILED_PRECONDITION: Commit of "
                                       "uuid=2"))));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  guard.ReleaseReadPin(copy.pin_id);
}

TEST_F(StagingBufferGuardTest, ClaimAdmitsOnlyItsOwnWrites) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));

  EXPECT_THAT(guard.BeginWrite(2),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Write of uuid=2 refused: receive uuid=1 of "
                                 "version(len=2, \"v1\") claims")));
  EXPECT_THAT(guard.BeginWrite(0),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("receive uuid=1 of")));
  ABSL_EXPECT_OK(guard.BeginWrite(1));
  guard.EndWrite(1);
}

TEST_F(StagingBufferGuardTest, RepeatedClaimJoinsOnlyWithTheSameVersion) {
  // A new claim would time out at once on the write in flight.
  StagingBufferGuard guard(/*claim_timeout=*/absl::ZeroDuration());
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  ABSL_ASSERT_OK(guard.BeginWrite(1));

  EXPECT_THAT(guard.Claim(1, v1_),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  ABSL_EXPECT_OK(guard.AwaitReady(1));
  EXPECT_THAT(guard.Claim(1, v2_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uuid=1 repeated with version(len=2, "
                                       "\"v2\")"),
                             HasSubstr("for version(len=2, \"v1\")"))));
  EXPECT_THAT(guard.Claim(0, v1_), StatusIs(absl::StatusCode::kInvalidArgument,
                                            HasSubstr("got uuid=0")));
  // The join kept the claim ready despite the write in flight, which a new
  // claim would wait for.
  EXPECT_THAT(guard.BeginWrite(1),
              IsOkAndHolds(StagingBufferGuard::WriteAdmission::kClaimed));
  guard.EndWrite(1);
  guard.EndWrite(1);
  ABSL_EXPECT_OK(guard.CommitReceiveAndPin(1, {}));
  EXPECT_THAT(guard.staged_version(), Optional(v1_));
}

TEST_F(StagingBufferGuardTest, CommitWithAWriteInFlightAbortsTheClaim) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  ABSL_ASSERT_OK(guard.BeginWrite(1));

  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("still in flight (writes: 1 of uuid=1;")));
  guard.EndWrite(1);
  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("receive uuid=1 of version(len=2, \"v1\") was "
                                 "aborted: FAILED_PRECONDITION: Commit")));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
}

TEST_F(StagingBufferGuardTest, AbortEndsTheClaimWithItsReason) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  // Aborting a receive that holds no claim does nothing.
  guard.AbortReceive(2, absl::InternalError("tiling failed"));
  EXPECT_THAT(guard.BeginWrite(3),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("receive uuid=1 of")));

  guard.AbortReceive(1, absl::InternalError("tiling failed"));
  const auto aborted =
      StatusIs(absl::StatusCode::kAborted,
               HasSubstr("receive uuid=1 of version(len=2, \"v1\") was "
                         "aborted: INTERNAL: tiling failed"));
  EXPECT_THAT(guard.BeginWrite(1), aborted);
  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}), aborted);
  EXPECT_THAT(ClaimAndAwait(guard, 1, v1_), aborted);
  ABSL_EXPECT_OK(guard.BeginWrite(3));
  guard.EndWrite(3);
}

TEST_F(StagingBufferGuardTest, NewerReceiveReplacesTheClaim) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 2, v2_));

  const auto superseded =
      StatusIs(absl::StatusCode::kAborted,
               HasSubstr("receive uuid=1 of version(len=2, \"v1\") was "
                         "aborted: ABORTED: superseded by receive uuid=2 of "
                         "version(len=2, \"v2\")"));
  EXPECT_THAT(guard.BeginWrite(1), superseded);
  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}), superseded);
  ABSL_EXPECT_OK(guard.BeginWrite(2));
  guard.EndWrite(2);
  ABSL_ASSERT_OK(guard.CommitReceiveAndPin(2, {}));
  // A late write of the replaced receive cannot drop the newer version.
  EXPECT_THAT(guard.BeginWrite(1), superseded);
  EXPECT_THAT(guard.staged_version(), Optional(v2_));
}

TEST_F(StagingBufferGuardTest, ReceiveAbortedLastCannotClaimAgain) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 2, v2_));

  EXPECT_THAT(ClaimAndAwait(guard, 1, v1_),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("Claim of uuid=1 refused: receive uuid=1 of "
                                 "version(len=2, \"v1\") was aborted: ABORTED: "
                                 "superseded by receive uuid=2")));
  // The refused claim left the claim of uuid 2 in place.
  ABSL_EXPECT_OK(guard.BeginWrite(2));
  guard.EndWrite(2);
  ABSL_EXPECT_OK(guard.CommitReceiveAndPin(2, {}));
  EXPECT_THAT(guard.staged_version(), Optional(v2_));
}

TEST_F(StagingBufferGuardTest, ClaimDrainsTheWritesOfTheReplacedClaim) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  ABSL_ASSERT_OK(guard.BeginWrite(1));
  absl::Status claimed;
  absl::Notification returned;
  std::thread receiver([&] {
    claimed = ClaimAndAwait(guard, 2, v2_);
    returned.Notify();
  });
  AwaitWritesRefused(guard, 1);  // Superseded by uuid 2.

  // Until the earlier write ends, the claim admits not even its own writes.
  EXPECT_THAT(guard.BeginWrite(2),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("in flight before it (writes: 1 of uuid=1;")));
  EXPECT_FALSE(returned.HasBeenNotified());
  guard.EndWrite(1);
  returned.WaitForNotification();
  receiver.join();
  ABSL_EXPECT_OK(claimed);
  ABSL_EXPECT_OK(guard.BeginWrite(2));
  guard.EndWrite(2);
}

TEST_F(StagingBufferGuardTest, ClaimIsReadyOnlyOnceEveryEarlierWriteEnded) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginWrite(7));  // Two legacy writes.
  ABSL_ASSERT_OK(guard.BeginWrite(8));
  absl::Status claimed;
  absl::Notification returned;
  std::thread receiver([&] {
    claimed = ClaimAndAwait(guard, 1, v1_);
    returned.Notify();
  });
  AwaitWritesRefused(guard, 9);  // Claimed by uuid 1.

  guard.EndWrite(7);
  EXPECT_THAT(
      guard.BeginWrite(1),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               AllOf(HasSubstr("in flight before it (writes: 1 of uuid=8;"),
                     Not(HasSubstr("uuid=7")))));
  EXPECT_FALSE(returned.HasBeenNotified());
  guard.EndWrite(8);
  returned.WaitForNotification();
  receiver.join();
  ABSL_EXPECT_OK(claimed);
}

TEST_F(StagingBufferGuardTest, JoinDuringTheDrainReturnsOnceTheClaimIsReady) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginWrite(7));  // A legacy write.
  absl::Status claimed;
  absl::Status joined;
  absl::Notification claim_returned;
  absl::Notification joining;
  absl::Notification join_returned;
  std::thread claimer([&] {
    claimed = ClaimAndAwait(guard, 1, v1_);
    claim_returned.Notify();
  });
  AwaitWritesRefused(guard, 8);  // Claimed by uuid 1.
  std::thread joiner([&] {
    joining.Notify();
    joined = ClaimAndAwait(guard, 1, v1_);
    join_returned.Notify();
  });
  joining.WaitForNotification();

  // Neither can return before the write ends. The join normally waits by
  // now; one that comes after EndWrite() returns OK at once.
  EXPECT_FALSE(claim_returned.HasBeenNotified());
  EXPECT_FALSE(join_returned.HasBeenNotified());
  guard.EndWrite(7);
  claimer.join();
  joiner.join();
  ABSL_EXPECT_OK(claimed);
  ABSL_EXPECT_OK(joined);
}

TEST_F(StagingBufferGuardTest, JoinDuringTheDrainWaitsWithTheClaim) {
  StagingBufferGuard guard(/*claim_timeout=*/absl::Milliseconds(100));
  ABSL_ASSERT_OK(guard.BeginWrite(7));  // A legacy write outlasting both waits.
  absl::Status claimed;
  std::thread claimer([&] { claimed = ClaimAndAwait(guard, 1, v1_); });
  AwaitWritesRefused(guard, 8);  // Claimed by uuid 1.

  // A join that returned before the claim is ready would return OK. Whichever
  // wait ends first times out and aborts the claim, which fails the other.
  const absl::Status joined = ClaimAndAwait(guard, 1, v1_);
  claimer.join();
  EXPECT_THAT((std::vector<absl::StatusCode>{joined.code(), claimed.code()}),
              UnorderedElementsAre(absl::StatusCode::kAborted,
                                   absl::StatusCode::kDeadlineExceeded));
  guard.EndWrite(7);
}

TEST_F(StagingBufferGuardTest, AbortDuringTheDrainFailsTheClaimAndItsJoin) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginWrite(7));  // A legacy write.
  absl::Status claimed;
  absl::Status joined;
  absl::Notification joining;
  std::thread claimer([&] { claimed = ClaimAndAwait(guard, 1, v1_); });
  AwaitWritesRefused(guard, 8);  // Claimed by uuid 1.
  std::thread joiner([&] {
    joining.Notify();
    joined = ClaimAndAwait(guard, 1, v1_);
  });
  joining.WaitForNotification();

  guard.AbortReceive(1, absl::InternalError("registration failed"));
  claimer.join();
  joiner.join();
  // A join that comes after the abort is refused with the same reason.
  const auto aborted =
      StatusIs(absl::StatusCode::kAborted,
               AllOf(HasSubstr("uuid=1 of version(len=2, \"v1\") was aborted"),
                     HasSubstr(": INTERNAL: registration failed")));
  EXPECT_THAT(claimed, aborted);
  EXPECT_THAT(joined, aborted);
  ABSL_EXPECT_OK(guard.BeginWrite(9));  // No claim blocks legacy writes.
  guard.EndWrite(9);
  guard.EndWrite(7);
}

TEST_F(StagingBufferGuardTest, ClaimSupersededWhileDrainingIsAborted) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(guard.BeginWrite(7));  // A legacy write.
  absl::Status first;
  absl::Notification first_returned;
  std::thread first_receiver([&] {
    first = ClaimAndAwait(guard, 1, v1_);
    first_returned.Notify();
  });
  AwaitWritesRefused(guard, 8);  // Claimed by uuid 1.
  absl::Status second;
  std::thread second_receiver([&] { second = ClaimAndAwait(guard, 2, v2_); });

  first_returned.WaitForNotification();
  EXPECT_THAT(first,
              StatusIs(absl::StatusCode::kAborted,
                       AllOf(HasSubstr("receive uuid=1 of version(len=2, "
                                       "\"v1\") was aborted: "),
                             HasSubstr("ABORTED: superseded by receive uuid=2 "
                                       "of version(len=2, \"v2\")"))));
  guard.EndWrite(7);
  first_receiver.join();
  second_receiver.join();
  ABSL_EXPECT_OK(second);
}

TEST_F(StagingBufferGuardTest, ClaimNotReadyInTimeIsAborted) {
  StagingBufferGuard guard(/*claim_timeout=*/absl::Milliseconds(10));
  ABSL_ASSERT_OK(guard.BeginWrite(7));

  EXPECT_THAT(
      ClaimAndAwait(guard, 1, v1_),
      StatusIs(absl::StatusCode::kDeadlineExceeded,
               AllOf(HasSubstr("Receive uuid=1 of version(len=2, \"v1\") timed "
                               "out after 10ms"),
                     HasSubstr("before its claim (writes: 1 of uuid=7;"))));
  // The aborted claim no longer blocks legacy writes.
  ABSL_ASSERT_OK(guard.BeginWrite(8));
  guard.EndWrite(8);
  EXPECT_THAT(guard.BeginWrite(1),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("receive uuid=1 of version(len=2, \"v1\") was "
                                 "aborted: DEADLINE_EXCEEDED")));
  guard.EndWrite(7);
}

TEST_F(StagingBufferGuardTest, ClaimReturnsTheReplacedUuidWithoutWaiting) {
  StagingBufferGuard guard;
  EXPECT_FALSE(guard.Claims(2));
  ABSL_ASSERT_OK(guard.BeginWrite(7));  // A legacy write that claims wait for.
  EXPECT_THAT(guard.Claim(2, v2_),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  EXPECT_TRUE(guard.Claims(2));

  // Joining returns 0; superseding returns the replaced uuid immediately.
  EXPECT_THAT(guard.Claim(2, v2_),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  EXPECT_THAT(guard.Claim(3, v2_),
              IsOkAndHolds(EndedWith(2, absl::StatusCode::kAborted)));
  EXPECT_FALSE(guard.Claims(2));
  EXPECT_TRUE(guard.Claims(3));

  // AwaitReady() on the superseded uuid fails without blocking.
  EXPECT_THAT(guard.AwaitReady(2),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("superseded by receive uuid=3")));
  guard.EndWrite(7);
  ABSL_EXPECT_OK(guard.AwaitReady(3));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin committed,
                       guard.CommitReceiveAndPin(3, {}));
  EXPECT_FALSE(guard.Claims(3));
  guard.ReleaseReadPin(committed.pin_id);
}

TEST_F(StagingBufferGuardTest, UnawaitedClaimPastItsDrainDoesNotWedgeBuffer) {
  {
    StagingBufferGuard guard(/*claim_timeout=*/absl::Milliseconds(10));
    ABSL_ASSERT_OK(guard.BeginWrite(7));
    EXPECT_THAT(guard.Claim(1, v1_),
                IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
    absl::SleepFor(absl::Milliseconds(25));
    // Next BeginWrite() aborts the timed-out unawaited claim and proceeds.
    ABSL_ASSERT_OK(guard.BeginWrite(8));
    guard.EndWrite(8);
    EXPECT_THAT(guard.AwaitReady(1), StatusIs(absl::StatusCode::kAborted,
                                              HasSubstr("DEADLINE_EXCEEDED")));
    guard.EndWrite(7);
  }
  {
    StagingBufferGuard guard(/*claim_timeout=*/absl::Milliseconds(10));
    ABSL_ASSERT_OK(guard.BeginWrite(7));
    EXPECT_THAT(guard.Claim(1, v1_),
                IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
    absl::SleepFor(absl::Milliseconds(25));
    // Re-claiming the timed-out claim aborts it and refuses with Aborted.
    EXPECT_THAT(guard.Claim(1, v1_), StatusIs(absl::StatusCode::kAborted,
                                              HasSubstr("DEADLINE_EXCEEDED")));
    guard.EndWrite(7);
  }
}

TEST_F(StagingBufferGuardTest, CheckClaimAndCheckUnclaimedReflectClaimState) {
  StagingBufferGuard guard;
  ABSL_EXPECT_OK(guard.CheckUnclaimed("Legacy registration", 9));
  EXPECT_THAT(guard.CheckClaim("Commit", 1),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("holds no claim")));

  ABSL_ASSERT_OK(guard.BeginWrite(7));
  EXPECT_THAT(guard.Claim(1, v1_),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  // Active even while still draining.
  ABSL_EXPECT_OK(guard.CheckClaim("Commit", 1));
  EXPECT_THAT(guard.CheckUnclaimed("Legacy registration", 9),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("Legacy registration of uuid=9 refused"),
                             HasSubstr("receive uuid=1 of version(len=2, "
                                       "\"v1\") claims"))));

  EXPECT_THAT(guard.Claim(2, v2_),
              IsOkAndHolds(EndedWith(1, absl::StatusCode::kAborted)));
  EXPECT_THAT(guard.CheckClaim("Commit", 1),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("superseded by receive uuid=2")));
  guard.EndWrite(7);
  ABSL_EXPECT_OK(guard.AwaitReady(2));
}

TEST_F(StagingBufferGuardTest,
       ExpiredClaimUnblocksLegacyWritesAndRefusesOwnWritesAndCommit) {
  StagingBufferGuard guard;
  const absl::Time expiry = absl::Now() + absl::Milliseconds(20);
  EXPECT_THAT(guard.Claim(1, v1_, expiry),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  ABSL_ASSERT_OK(guard.AwaitReady(1));
  EXPECT_EQ(guard.ClaimExpiry(1), expiry);

  absl::SleepFor(absl::Milliseconds(35));
  EXPECT_FALSE(guard.Claims(1));
  EXPECT_THAT(guard.CheckClaim("Commit", 1),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("Receive uuid=1")));
  ABSL_EXPECT_OK(guard.CheckUnclaimed("Legacy", 0));

  // Legacy write expires the claim and proceeds; ExpireIfPastDeadline() drains
  // the expired claim record.
  ABSL_ASSERT_OK(guard.BeginWrite(0));
  guard.EndWrite(0);
  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());

  // The expired receive's own late writes, commit, and re-claim are refused.
  const auto expired_aborted = StatusIs(
      absl::StatusCode::kAborted,
      AllOf(HasSubstr("receive uuid=1"), HasSubstr("DEADLINE_EXCEEDED")));
  EXPECT_THAT(guard.BeginWrite(1), expired_aborted);
  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}), expired_aborted);
  EXPECT_THAT(guard.Claim(1, v1_), expired_aborted);
}

TEST_F(StagingBufferGuardTest, NextClaimReportsExpiredClaimAsDeadlineExceeded) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() + absl::Milliseconds(20)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  ABSL_ASSERT_OK(guard.AwaitReady(1));

  absl::SleepFor(absl::Milliseconds(35));
  EXPECT_THAT(guard.Claim(2, v2_),
              IsOkAndHolds(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());
  ABSL_EXPECT_OK(guard.AwaitReady(2));
}

TEST_F(StagingBufferGuardTest, CommitOfExpiredClaimReturnsDeadlineExceeded) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() + absl::Milliseconds(20)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  ABSL_ASSERT_OK(guard.AwaitReady(1));

  absl::SleepFor(absl::Milliseconds(35));
  EXPECT_THAT(guard.CommitReceiveAndPin(1, {}),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("Receive uuid=1")));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());
}

TEST_F(StagingBufferGuardTest, ExpireIfPastDeadlineDirectlyExpiresActiveClaim) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() - absl::Milliseconds(1)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));

  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());
  EXPECT_THAT(guard.BeginWrite(1), StatusIs(absl::StatusCode::kAborted,
                                            HasSubstr("DEADLINE_EXCEEDED")));
  ASSERT_THAT(guard.BeginWrite(0),
              IsOkAndHolds(StagingBufferGuard::WriteAdmission::kUnclaimed));
  guard.EndWrite(0);
}

TEST_F(StagingBufferGuardTest, AwaitReadyExpiresAlreadyReadyPastDeadlineClaim) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() - absl::Milliseconds(1)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));

  EXPECT_THAT(guard.AwaitReady(1),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("expired before it committed")));
  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());
}

TEST_F(StagingBufferGuardTest,
       ReclaimOfUnobservedExpiredClaimReturnsAbortedAndQueuesExpiry) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() - absl::Milliseconds(1)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));

  EXPECT_THAT(guard.Claim(1, v1_), StatusIs(absl::StatusCode::kAborted,
                                            HasSubstr("DEADLINE_EXCEEDED")));
  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());
}

TEST_F(StagingBufferGuardTest,
       CommitOfOtherUuidExpiresUnobservedClaimAndQueuesExpiry) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() - absl::Milliseconds(1)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));

  EXPECT_THAT(guard.CommitReceiveAndPin(2, {}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("holds no claim")));
  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  EXPECT_THAT(guard.ExpireIfPastDeadline(), IsEmpty());
}

TEST_F(StagingBufferGuardTest,
       AwaitReadyUsesEarlierOfDrainDeadlineAndReceiveExpiry) {
  StagingBufferGuard guard(/*claim_timeout=*/absl::Seconds(10));
  ABSL_ASSERT_OK(guard.BeginWrite(7));

  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() + absl::Milliseconds(20)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  const absl::Time start = absl::Now();
  EXPECT_THAT(guard.AwaitReady(1),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("expired before it committed")));
  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  guard.EndWrite(7);
}

TEST_F(StagingBufferGuardTest,
       AwaitReadyWakesWhenConcurrentJoinTightensExpiry) {
  StagingBufferGuard guard(/*claim_timeout=*/absl::Seconds(4));
  ABSL_ASSERT_OK(guard.BeginWrite(7));
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() + absl::Seconds(10)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));

  absl::Notification entering_wait;
  absl::Status wait_status;
  std::thread waiter([&] {
    entering_wait.Notify();
    wait_status = guard.AwaitReady(1);
  });
  entering_wait.WaitForNotification();
  absl::SleepFor(absl::Milliseconds(20));

  const absl::Time start = absl::Now();
  EXPECT_THAT(guard.Claim(1, v1_, absl::Now() + absl::Milliseconds(20)),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  waiter.join();

  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
  EXPECT_THAT(wait_status, StatusIs(absl::StatusCode::kDeadlineExceeded,
                                    HasSubstr("expired before it committed")));
  EXPECT_THAT(guard.ExpireIfPastDeadline(),
              ElementsAre(EndedWith(1, absl::StatusCode::kDeadlineExceeded)));
  guard.EndWrite(7);
}

TEST_F(StagingBufferGuardTest, JoinTightensExpiryNeverExtends) {
  StagingBufferGuard guard;
  const absl::Time initial = absl::Now() + absl::Seconds(10);
  const absl::Time later = initial + absl::Seconds(10);
  const absl::Time earlier = absl::Now() + absl::Seconds(5);

  EXPECT_EQ(guard.ClaimExpiry(1), absl::InfiniteFuture());
  EXPECT_THAT(guard.Claim(1, v1_, initial),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  EXPECT_EQ(guard.ClaimExpiry(2), absl::InfiniteFuture());
  EXPECT_THAT(guard.Claim(1, v1_, later),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  EXPECT_EQ(guard.ClaimExpiry(1), initial);
  EXPECT_THAT(guard.Claim(1, v1_, earlier),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  EXPECT_EQ(guard.ClaimExpiry(1), earlier);
}

TEST_F(StagingBufferGuardTest, ForgetAbortedAllowsOnlyForgottenUuidToReclaim) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  guard.AbortReceive(1, absl::InternalError("evicted receive"));
  EXPECT_THAT(guard.Claim(1, v1_),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("INTERNAL: evicted receive")));

  // Forgetting another uuid keeps the aborted record for uuid 1 intact.
  guard.ForgetAborted(2);
  EXPECT_THAT(guard.Claim(1, v1_),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("INTERNAL: evicted receive")));

  // Forgetting uuid 1 lets it claim afresh; a repeated call does nothing.
  guard.ForgetAborted(1);
  guard.ForgetAborted(1);
  EXPECT_THAT(guard.Claim(1, v1_),
              IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
  ABSL_EXPECT_OK(guard.AwaitReady(1));
  EXPECT_TRUE(guard.Claims(1));
}

TEST_F(StagingBufferGuardTest,
       LegacyWriteNeverAdmitsOverLivePinAfterInterveningClaimExpires) {
  // Subcase (a): intervening claim expires while a read pin stays held;
  // legacy write must wait and fail rather than admitting over the live pin.
  {
    StagingBufferGuard guard(kVersionedReceiveClaimTimeout,
                             /*legacy_write_pin_wait=*/absl::Milliseconds(120));
    ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
    TF_ASSERT_OK_AND_ASSIGN(const ReadPin pin, guard.CommitReceiveAndPin(1, {}));

    absl::StatusOr<StagingBufferGuard::WriteAdmission> admission;
    std::thread writer([&] {
      admission = guard.BeginWrite(
          /*uuid=*/0, StagingBufferGuard::WritePolicy::kAllowLegacy);
    });
    // Once the staged version is dropped, the writer is waiting on `pin`.
    while (guard.staged_version().has_value()) std::this_thread::yield();

    // Install an already-expired claim to wake the writer while `pin` stays
    // held; BeginWrite() expires it and must re-wait on `pin`.
    EXPECT_THAT(
        guard.Claim(/*uuid=*/2, v2_, absl::Now() - absl::Milliseconds(1)),
        IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
    writer.join();

    EXPECT_THAT(
        guard.CheckClaim("Write", 2),
        StatusIs(absl::StatusCode::kAborted, HasSubstr("DEADLINE_EXCEEDED")));
    ASSERT_THAT(admission, StatusIs(absl::StatusCode::kFailedPrecondition,
                                    AllOf(HasSubstr("read pins still held"),
                                          HasSubstr("pin 1"))));
    guard.ReleaseReadPin(pin.pin_id);
    EXPECT_THAT(guard.ExpireIfPastDeadline(),
                ElementsAre(EndedWith(2, absl::StatusCode::kDeadlineExceeded)));
  }

  // Subcase (b): positive control — releasing the read pin after the
  // intervening claim expires unblocks the legacy write cleanly.
  {
    StagingBufferGuard guard(kVersionedReceiveClaimTimeout,
                             /*legacy_write_pin_wait=*/absl::Milliseconds(500));
    ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
    TF_ASSERT_OK_AND_ASSIGN(const ReadPin pin, guard.CommitReceiveAndPin(1, {}));

    absl::Notification writer_done;
    absl::StatusOr<StagingBufferGuard::WriteAdmission> admission;
    std::thread writer([&] {
      admission = guard.BeginWrite(
          /*uuid=*/0, StagingBufferGuard::WritePolicy::kAllowLegacy);
      writer_done.Notify();
    });
    while (guard.staged_version().has_value()) std::this_thread::yield();

    EXPECT_THAT(
        guard.Claim(/*uuid=*/2, v2_, absl::Now() - absl::Milliseconds(1)),
        IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));

    // Writer must remain blocked while `pin` is still held.
    EXPECT_FALSE(
        writer_done.WaitForNotificationWithTimeout(absl::Milliseconds(30)));
    guard.ReleaseReadPin(pin.pin_id);
    writer.join();

    ASSERT_THAT(admission,
                IsOkAndHolds(StagingBufferGuard::WriteAdmission::kUnclaimed));
    guard.EndWrite(0);
    EXPECT_THAT(guard.ExpireIfPastDeadline(),
                ElementsAre(EndedWith(2, absl::StatusCode::kDeadlineExceeded)));
  }

  // Subcase (c): intervening claim with no AwaitReady() waiter misses its drain
  // deadline while `pin` stays held; BeginWrite() aborts the stalled claim and
  // still refuses to admit over the live pin.
  {
    StagingBufferGuard guard(/*claim_timeout=*/absl::ZeroDuration(),
                             /*legacy_write_pin_wait=*/absl::Milliseconds(120));
    ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
    TF_ASSERT_OK_AND_ASSIGN(const ReadPin pin, guard.CommitReceiveAndPin(1, {}));

    absl::StatusOr<StagingBufferGuard::WriteAdmission> admission;
    std::thread writer([&] {
      admission = guard.BeginWrite(
          /*uuid=*/0, StagingBufferGuard::WritePolicy::kAllowLegacy);
    });
    while (guard.staged_version().has_value()) std::this_thread::yield();

    EXPECT_THAT(guard.Claim(/*uuid=*/3, v2_, absl::InfiniteFuture()),
                IsOkAndHolds(EndedWith(0, absl::StatusCode::kOk)));
    writer.join();

    ASSERT_THAT(admission, StatusIs(absl::StatusCode::kFailedPrecondition,
                                    AllOf(HasSubstr("read pins still held"),
                                          HasSubstr("pin 1"))));
    EXPECT_FALSE(guard.Claims(3));
    EXPECT_THAT(guard.BeginWrite(3),
                StatusIs(absl::StatusCode::kAborted, HasSubstr("timed out")));
    guard.ReleaseReadPin(pin.pin_id);
  }
}

TEST_F(StagingBufferGuardTest, AwaitReadyWithoutAClaimFailsAtOnce) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.AwaitReady(1),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("Wait of uuid=1 refused: it holds no claim on "
                                 "the staging buffer")));
}

TEST_F(StagingBufferGuardTest, BeginWriteReportsClaimedVsUnclaimedAdmission) {
  StagingBufferGuard guard;
  EXPECT_THAT(guard.BeginWrite(0),
              IsOkAndHolds(StagingBufferGuard::WriteAdmission::kUnclaimed));
  guard.EndWrite(0);

  ABSL_ASSERT_OK(ClaimAndAwait(guard, 7, v1_));
  EXPECT_THAT(guard.BeginWrite(7),
              IsOkAndHolds(StagingBufferGuard::WriteAdmission::kClaimed));
  guard.EndWrite(7);
  ASSERT_THAT(
      guard.BeginWrite(7, StagingBufferGuard::WritePolicy::kClaimedOnly),
      IsOkAndHolds(StagingBufferGuard::WriteAdmission::kClaimed));
  guard.EndWrite(7);
}

TEST_F(StagingBufferGuardTest,
       BeginWriteClaimedOnlyRefusesCommittedUuidWithoutDroppingStaged) {
  StagingBufferGuard guard;
  ABSL_ASSERT_OK(ClaimAndAwait(guard, 1, v1_));
  TF_ASSERT_OK_AND_ASSIGN(const ReadPin pin, guard.CommitReceiveAndPin(1, {}));

  // Even while a read pin is held, kClaimedOnly refuses immediately without
  // waiting on pins or dropping the staged commit.
  EXPECT_THAT(
      guard.BeginWrite(1, StagingBufferGuard::WritePolicy::kClaimedOnly),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("holds no claim")));
  EXPECT_THAT(guard.staged_version(), Optional(v1_));

  guard.ReleaseReadPin(pin.pin_id);
  EXPECT_THAT(
      guard.BeginWrite(1, StagingBufferGuard::WritePolicy::kClaimedOnly),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("holds no claim")));
  EXPECT_THAT(guard.staged_version(), Optional(v1_));

  // Positive pairing: kAllowLegacy still admits an unclaimed write and drops
  // the staged version.
  ASSERT_THAT(
      guard.BeginWrite(0, StagingBufferGuard::WritePolicy::kAllowLegacy),
      IsOkAndHolds(StagingBufferGuard::WriteAdmission::kUnclaimed));
  EXPECT_EQ(guard.staged_version(), std::nullopt);
  guard.EndWrite(0);
}

TEST(StagingBufferGuardDeathTest, EndWriteWithoutBeginWriteFails) {
  StagingBufferGuard guard;
  EXPECT_DEATH(guard.EndWrite(5),
               "EndWrite\\(uuid=5\\) without a matching BeginWrite");
}

TEST(StagingBufferGuardDeathTest, AbortReceiveWithAnOkReasonFailsInDebug) {
  StagingBufferGuard guard;
  EXPECT_DEBUG_DEATH(guard.AbortReceive(1, absl::OkStatus()),
                     "AbortReceive\\(uuid=1\\) needs a non-OK reason");
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

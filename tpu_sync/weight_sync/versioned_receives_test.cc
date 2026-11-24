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

#include "tpu_sync/weight_sync/versioned_receives.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

ReceiveSpec SpecFor(const WeightVersion& version) {
  ReceiveSpec spec;
  spec.version = version;
  return spec;
}

// Starts receive `uuid` at `at` and commits it at once; returns the uuids that
// StartActive() evicted.
std::vector<uint64_t> StartAndCommit(VersionedReceives& records, uint64_t uuid,
                                     const WeightVersion& version,
                                     absl::Duration receive_timeout,
                                     absl::Time at) {
  std::vector<uint64_t> evicted = records.StartActive(
      uuid, SpecFor(version), receive_timeout, at + receive_timeout, at);
  records.MarkCommitted(uuid, at);
  return evicted;
}

class VersionedReceivesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(v1_, WeightVersion::FromBytes("v1", "version"));
    TF_ASSERT_OK_AND_ASSIGN(v2_, WeightVersion::FromBytes("v2", "version"));
  }

  WeightVersion v1_;
  WeightVersion v2_;
  const absl::Time t0_ = absl::UnixEpoch() + absl::Hours(1);
};

TEST_F(VersionedReceivesTest, ClaimJoinAndMismatchDecisions) {
  VersionedReceives records;
  const auto first = records.LookupForPrepare(1, SpecFor(v1_), t0_);
  EXPECT_EQ(first.decision, VersionedReceives::PrepareDecision::kClaim);
  ABSL_EXPECT_OK(first.status);

  EXPECT_THAT(records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  EXPECT_EQ(records.active_claim_uuid(), uint64_t{1});
  EXPECT_FALSE(records.IsReadyActive(1));

  // Repeat with the same version joins (whether draining or ready).
  const auto join_draining = records.LookupForPrepare(1, SpecFor(v1_), t0_);
  EXPECT_EQ(join_draining.decision, VersionedReceives::PrepareDecision::kJoin);

  records.MarkReady(1);
  EXPECT_TRUE(records.IsReadyActive(1));
  const auto join_ready = records.LookupForPrepare(1, SpecFor(v1_), t0_);
  EXPECT_EQ(join_ready.decision, VersionedReceives::PrepareDecision::kJoin);

  // Repeat with another version fails InvalidArgument naming both versions.
  const auto mismatch = records.LookupForPrepare(1, SpecFor(v2_), t0_);
  EXPECT_EQ(mismatch.decision, VersionedReceives::PrepareDecision::kError);
  EXPECT_THAT(mismatch.status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uuid=1"), HasSubstr("\"v2\""),
                             HasSubstr("\"v1\""))));
}

TEST_F(VersionedReceivesTest, FinishedRecordsReturnOutcomesAndBlockWrites) {
  VersionedReceives records;

  // Committed receive.
  EXPECT_THAT(records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.MarkReady(1);
  records.MarkCommitted(1, t0_ + absl::Seconds(1));
  EXPECT_EQ(records.active_claim_uuid(), uint64_t{0});
  // A repeated commit keeps the first commit's retention.
  records.MarkCommitted(1, t0_ + absl::Seconds(2));
  ASSERT_NE(records.Find(1), nullptr);
  EXPECT_EQ(records.Find(1)->retain_until,
            t0_ + absl::Seconds(1) + kMinReceiveRecordRetention);

  const auto prep_committed =
      records.LookupForPrepare(1, SpecFor(v1_), t0_ + absl::Seconds(5));
  EXPECT_EQ(prep_committed.decision,
            VersionedReceives::PrepareDecision::kFinished);
  EXPECT_THAT(
      prep_committed.status,
      StatusIs(absl::StatusCode::kFailedPrecondition,
               AllOf(HasSubstr("uuid=1"), HasSubstr("already committed"))));
  std::vector<uint64_t> evicted;
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(1, t0_ + absl::Seconds(5), &evicted),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("already committed")));
  EXPECT_THAT(evicted, IsEmpty());

  // Superseded (aborted) receive.
  EXPECT_THAT(records.StartActive(2, SpecFor(v2_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.FinishWithFailure(2, absl::AbortedError("superseded by uuid=3"),
                            t0_ + absl::Seconds(1));
  // First terminal state wins.
  records.FinishWithFailure(2, absl::DeadlineExceededError("late expiry"),
                            t0_ + absl::Seconds(2));
  const auto prep_aborted =
      records.LookupForPrepare(2, SpecFor(v2_), t0_ + absl::Seconds(5));
  EXPECT_EQ(prep_aborted.decision,
            VersionedReceives::PrepareDecision::kFinished);
  EXPECT_THAT(prep_aborted.status,
              StatusIs(absl::StatusCode::kAborted, HasSubstr("superseded")));
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(2, t0_ + absl::Seconds(5), &evicted),
      StatusIs(absl::StatusCode::kAborted, HasSubstr("superseded")));

  // Expired receive.
  EXPECT_THAT(records.StartActive(3, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.FinishWithFailure(3, absl::DeadlineExceededError("expired"),
                            t0_ + absl::Seconds(30));
  ASSERT_NE(records.Find(3), nullptr);
  EXPECT_EQ(records.Find(3)->state, ReceiveState::kExpired);
  const auto prep_expired =
      records.LookupForPrepare(3, SpecFor(v1_), t0_ + absl::Seconds(35));
  EXPECT_EQ(prep_expired.decision,
            VersionedReceives::PrepareDecision::kFinished);
  EXPECT_THAT(prep_expired.status, StatusIs(absl::StatusCode::kDeadlineExceeded,
                                            HasSubstr("expired")));
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(3, t0_ + absl::Seconds(35), &evicted),
      StatusIs(absl::StatusCode::kAborted, HasSubstr("DEADLINE_EXCEEDED")));

  // Internal failure.
  EXPECT_THAT(records.StartActive(4, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.FinishWithFailure(4, absl::InternalError("tiling failed"),
                            t0_ + absl::Seconds(1));
  ASSERT_NE(records.Find(4), nullptr);
  EXPECT_EQ(records.Find(4)->state, ReceiveState::kFailed);
  const auto prep_failed =
      records.LookupForPrepare(4, SpecFor(v1_), t0_ + absl::Seconds(5));
  EXPECT_EQ(prep_failed.decision,
            VersionedReceives::PrepareDecision::kFinished);
  EXPECT_THAT(prep_failed.status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("tiling")));
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(4, t0_ + absl::Seconds(5), &evicted),
      StatusIs(absl::StatusCode::kAborted, HasSubstr("INTERNAL")));
}

TEST_F(VersionedReceivesTest, RetentionUsesMaxOfTenMinutesAndTwiceTimeout) {
  VersionedReceives records;
  // Short timeout (30 s): retained for 10 min after finish.
  EXPECT_THAT(records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  const absl::Time finish1 = t0_ + absl::Seconds(10);
  records.MarkCommitted(1, finish1);
  EXPECT_EQ(
      records.LookupForPrepare(1, SpecFor(v1_), finish1 + absl::Minutes(9))
          .decision,
      VersionedReceives::PrepareDecision::kFinished);
  const auto expired1 =
      records.LookupForPrepare(1, SpecFor(v1_), finish1 + absl::Minutes(10));
  EXPECT_EQ(expired1.decision, VersionedReceives::PrepareDecision::kClaim);
  EXPECT_THAT(expired1.evicted_uuids, ElementsAre(1));

  // Long timeout (20 min): retained for 2 * 20 min = 40 min after finish.
  EXPECT_THAT(records.StartActive(2, SpecFor(v2_), absl::Minutes(20),
                                  t0_ + absl::Minutes(20), t0_),
              IsEmpty());
  const absl::Time finish2 = t0_ + absl::Minutes(1);
  records.MarkCommitted(2, finish2);
  EXPECT_EQ(
      records.LookupForPrepare(2, SpecFor(v2_), finish2 + absl::Minutes(39))
          .decision,
      VersionedReceives::PrepareDecision::kFinished);
  std::vector<uint64_t> evicted;
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(2, finish2 + absl::Minutes(40), &evicted),
      IsOkAndHolds(false));
  EXPECT_THAT(evicted, ElementsAre(2));
}

TEST_F(VersionedReceivesTest, CapacityEvictsOldestFinishedRecordFirst) {
  VersionedReceives records(/*max_records=*/3,
                            /*min_retention=*/absl::Minutes(10));
  // Fill 3 finished records: uuids 1, 2, 3.
  for (uint64_t u : {1, 2, 3}) {
    EXPECT_THAT(records.StartActive(u, SpecFor(v1_), absl::Seconds(10),
                                    t0_ + absl::Seconds(10), t0_),
                IsEmpty());
    records.MarkCommitted(u, t0_ + absl::Seconds(u));
  }
  EXPECT_EQ(records.size(), size_t{3});

  // Starting uuid 4 at capacity evicts the oldest finished record (uuid 1),
  // never refusing the prepare.
  EXPECT_THAT(
      records.StartActive(4, SpecFor(v2_), absl::Seconds(10),
                          t0_ + absl::Seconds(10), t0_ + absl::Seconds(5)),
      ElementsAre(1));
  EXPECT_EQ(records.size(), size_t{3});
  EXPECT_EQ(records.Find(1), nullptr);
  EXPECT_NE(records.Find(2), nullptr);
  EXPECT_NE(records.Find(3), nullptr);
  EXPECT_NE(records.Find(4), nullptr);

  // Evicted uuid 1 can now claim afresh.
  EXPECT_EQ(records.LookupForPrepare(1, SpecFor(v1_), t0_ + absl::Seconds(6))
                .decision,
            VersionedReceives::PrepareDecision::kClaim);
}

TEST_F(VersionedReceivesTest, SweepEvictsExpiredRecordThatIsNotTheOldest) {
  VersionedReceives records(/*max_records=*/3,
                            /*min_retention=*/absl::Minutes(1));
  // Retained until t0+20m, t0+1s+1m, and t0+2s+20m.
  EXPECT_THAT(StartAndCommit(records, 1, v1_, absl::Minutes(10), t0_),
              IsEmpty());
  EXPECT_THAT(StartAndCommit(records, 2, v1_, absl::ZeroDuration(),
                             t0_ + absl::Seconds(1)),
              IsEmpty());
  EXPECT_THAT(StartAndCommit(records, 3, v1_, absl::Minutes(10),
                             t0_ + absl::Seconds(2)),
              IsEmpty());

  // At capacity, the sweep evicts expired uuid 2, so the oldest (uuid 1) stays.
  EXPECT_THAT(
      records.StartActive(4, SpecFor(v2_), absl::Minutes(10),
                          t0_ + absl::Minutes(12), t0_ + absl::Minutes(2)),
      ElementsAre(2));
  EXPECT_NE(records.Find(1), nullptr);
  EXPECT_EQ(records.Find(2), nullptr);

  // The sweep kept the finish order intact: later starts evict uuid 1 and then
  // uuid 3, the oldest survivors.
  records.MarkCommitted(4, t0_ + absl::Minutes(2));
  EXPECT_THAT(StartAndCommit(records, 5, v2_, absl::Minutes(10),
                             t0_ + absl::Minutes(3)),
              ElementsAre(1));
  EXPECT_THAT(StartAndCommit(records, 6, v2_, absl::Minutes(10),
                             t0_ + absl::Minutes(4)),
              ElementsAre(3));
}

TEST_F(VersionedReceivesTest,
       ReusedUuidAfterRetentionExpiryEvictsOldestNotYoungest) {
  VersionedReceives records(/*max_records=*/3,
                            /*min_retention=*/absl::Minutes(1));
  // Uuid 1 is retained until t0+1m and uuid 2 until t0+1s+20m.
  EXPECT_THAT(StartAndCommit(records, 1, v1_, absl::ZeroDuration(), t0_),
              IsEmpty());
  EXPECT_THAT(StartAndCommit(records, 2, v1_, absl::Minutes(10),
                             t0_ + absl::Seconds(1)),
              IsEmpty());
  // A prepare evicts expired uuid 1, which is then reused: uuid 2 becomes the
  // oldest finished record.
  EXPECT_THAT(records.LookupForPrepare(1, SpecFor(v1_), t0_ + absl::Minutes(2))
                  .evicted_uuids,
              ElementsAre(1));
  EXPECT_THAT(StartAndCommit(records, 1, v2_, absl::ZeroDuration(),
                             t0_ + absl::Minutes(2)),
              IsEmpty());
  EXPECT_THAT(StartAndCommit(records, 3, v2_, absl::Minutes(10),
                             t0_ + absl::Minutes(2) + absl::Seconds(1)),
              IsEmpty());

  EXPECT_THAT(records.StartActive(4, SpecFor(v2_), absl::Minutes(10),
                                  t0_ + absl::Minutes(12),
                                  t0_ + absl::Minutes(2) + absl::Seconds(2)),
              ElementsAre(2));
  EXPECT_NE(records.Find(1), nullptr);
  EXPECT_EQ(records.Find(2), nullptr);
}

TEST_F(VersionedReceivesTest, CommittingReleasesActiveSlotForNextClaim) {
  VersionedReceives records;
  EXPECT_THAT(records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.MarkReady(1);
  records.MarkCommitting(1);
  EXPECT_EQ(records.active_claim_uuid(), uint64_t{0});
  EXPECT_FALSE(records.IsReadyActive(1));

  // Late writes and duplicate prepares for committing uuid 1 are refused.
  EXPECT_EQ(records.LookupForPrepare(1, SpecFor(v1_), t0_).decision,
            VersionedReceives::PrepareDecision::kFinished);
  std::vector<uint64_t> evicted;
  EXPECT_THAT(records.CheckWriteNotTombstoned(1, t0_, &evicted),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("already committed")));

  // Meanwhile uuid 2 can install the next active claim while uuid 1's hooked
  // copy finishes, and uuid 1 can still transition to kCommitted or kFailed
  // without disturbing uuid 2.
  EXPECT_THAT(records.StartActive(2, SpecFor(v2_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  EXPECT_EQ(records.active_claim_uuid(), uint64_t{2});
  records.MarkCommitted(1, t0_ + absl::Seconds(1));
  EXPECT_EQ(records.active_claim_uuid(), uint64_t{2});
  ASSERT_NE(records.Find(1), nullptr);
  EXPECT_EQ(records.Find(1)->state, ReceiveState::kCommitted);
}

TEST_F(VersionedReceivesTest, FirstTerminalTransitionWins) {
  VersionedReceives records;
  // A committed record ignores later failures, even from the hooked H2D path.
  EXPECT_THAT(StartAndCommit(records, 1, v1_, absl::Seconds(30), t0_),
              IsEmpty());
  records.FinishWithFailure(1, absl::AbortedError("late abort"),
                            t0_ + absl::Seconds(1));
  records.Finish(1, ReceiveState::kFailed, absl::InternalError("late H2D"),
                 t0_ + absl::Seconds(2), /*allow_committing=*/true);
  ASSERT_NE(records.Find(1), nullptr);
  EXPECT_EQ(records.Find(1)->state, ReceiveState::kCommitted);
  ABSL_EXPECT_OK(records.Find(1)->outcome);
  EXPECT_EQ(records.Find(1)->retain_until, t0_ + kMinReceiveRecordRetention);

  // A failed record ignores a later commit.
  EXPECT_THAT(records.StartActive(2, SpecFor(v2_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.FinishWithFailure(2, absl::InternalError("tiling failed"), t0_);
  records.MarkCommitted(2, t0_ + absl::Seconds(1));
  ASSERT_NE(records.Find(2), nullptr);
  EXPECT_EQ(records.Find(2)->state, ReceiveState::kFailed);
  EXPECT_THAT(records.Find(2)->outcome, StatusIs(absl::StatusCode::kInternal,
                                                 HasSubstr("tiling failed")));
  EXPECT_EQ(records.Find(2)->retain_until, t0_ + kMinReceiveRecordRetention);
}

TEST_F(VersionedReceivesTest, CancelledFailureFinishesAsAborted) {
  VersionedReceives records;
  EXPECT_THAT(records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.FinishWithFailure(
      1, absl::CancelledError("Receive uuid=1 was drained before it committed"),
      t0_);
  ASSERT_NE(records.Find(1), nullptr);
  EXPECT_EQ(records.Find(1)->state, ReceiveState::kAborted);
  EXPECT_THAT(records.Find(1)->outcome,
              StatusIs(absl::StatusCode::kCancelled, HasSubstr("drained")));
}

TEST_F(VersionedReceivesTest, JoinChecksTiling) {
  VersionedReceives records;
  ReceiveSpec base;
  base.version = v1_;
  base.host_tiling_mode = tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL;
  base.skip_tiling = {{0, true}};

  EXPECT_THAT(records.StartActive(10, base, absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());

  ReceiveSpec bad_tiling = base;
  bad_tiling.host_tiling_mode = tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED;
  const auto tiling_mismatch = records.LookupForPrepare(10, bad_tiling, t0_);
  EXPECT_EQ(tiling_mismatch.decision,
            VersionedReceives::PrepareDecision::kError);
  EXPECT_THAT(tiling_mismatch.status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("HOST_TILING_MODE_PRE_TILED"),
                             HasSubstr("HOST_TILING_MODE_ON_ARRIVAL"))));

  ReceiveSpec bad_skip = base;
  bad_skip.skip_tiling = {{1, true}};
  const auto skip_mismatch = records.LookupForPrepare(10, bad_skip, t0_);
  EXPECT_EQ(skip_mismatch.decision, VersionedReceives::PrepareDecision::kError);
  EXPECT_THAT(skip_mismatch.status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("{1:true}"), HasSubstr("{0:true}"))));
}

TEST_F(VersionedReceivesTest, ExternalFinishDoesNotOverwriteCommittingRecord) {
  VersionedReceives records(/*max_records=*/4,
                            /*min_retention=*/absl::Minutes(1));
  std::vector<uint64_t> evicted;

  // Unknown uuid reports false (not active versioned).
  EXPECT_THAT(records.CheckWriteNotTombstoned(20, t0_, &evicted),
              IsOkAndHolds(false));
  EXPECT_THAT(evicted, IsEmpty());

  // Active non-committing uuid reports true.
  EXPECT_THAT(records.StartActive(20, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  records.MarkReady(20);
  EXPECT_THAT(records.CheckWriteNotTombstoned(20, t0_, &evicted),
              IsOkAndHolds(true));

  // Committing uuid refuses writes and ignores external FinishWithFailure.
  records.MarkCommitting(20);
  EXPECT_THAT(records.CheckWriteNotTombstoned(20, t0_, &evicted),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("already committed")));
  records.FinishWithFailure(20, absl::AbortedError("external abort"),
                            t0_ + absl::Seconds(1));
  ASSERT_NE(records.Find(20), nullptr);
  EXPECT_EQ(records.Find(20)->state, ReceiveState::kActive);
  EXPECT_TRUE(records.Find(20)->committing);

  // Explicit allow_committing=true transitions committing -> kFailed.
  records.Finish(20, ReceiveState::kFailed,
                 absl::InternalError("hooked H2D failed"),
                 t0_ + absl::Seconds(2), /*allow_committing=*/true);
  ASSERT_NE(records.Find(20), nullptr);
  EXPECT_EQ(records.Find(20)->state, ReceiveState::kFailed);
  EXPECT_FALSE(records.Find(20)->committing);
  EXPECT_THAT(records.Find(20)->outcome,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("hooked H2D")));
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(20, t0_ + absl::Seconds(3), &evicted),
      StatusIs(absl::StatusCode::kAborted, HasSubstr("hooked H2D")));

  // Once evicted past retention, CheckWriteNotTombstoned returns false.
  EXPECT_THAT(
      records.CheckWriteNotTombstoned(20, t0_ + absl::Minutes(5), &evicted),
      IsOkAndHolds(false));
  EXPECT_THAT(evicted, ElementsAre(20));
}

TEST_F(VersionedReceivesTest, UnclaimedWriteIsTrackedUntilCleared) {
  VersionedReceives records;
  // Uuid 0 tags untagged pushes, whose writes are never tracked.
  records.RecordUnclaimedWrite(0);
  records.RecordUnclaimedWrite(30);
  EXPECT_FALSE(records.HasUnclaimedWrite(0));
  EXPECT_TRUE(records.HasUnclaimedWrite(30));
  EXPECT_FALSE(records.Contains(30));

  records.ClearUnclaimedWrite(30);
  EXPECT_FALSE(records.HasUnclaimedWrite(30));
}

TEST_F(VersionedReceivesTest, FinishForgetsUnclaimedWriteAndKeepsTombstone) {
  VersionedReceives records;
  records.RecordUnclaimedWrite(31);
  EXPECT_THAT(records.StartActive(31, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  EXPECT_TRUE(records.Contains(31));
  EXPECT_TRUE(records.HasUnclaimedWrite(31));

  records.FinishWithFailure(
      31, absl::FailedPreconditionError("chunks were admitted before claim"),
      t0_ + absl::Seconds(1));
  EXPECT_TRUE(records.Contains(31));
  EXPECT_FALSE(records.HasUnclaimedWrite(31));
}

TEST_F(VersionedReceivesTest, UnclaimedWritesIsBoundedLru) {
  VersionedReceives records(kMaxReceiveRecords, kMinReceiveRecordRetention,
                            /*max_unclaimed_writes=*/4);
  for (const uint64_t uuid : {1, 2, 3, 4}) records.RecordUnclaimedWrite(uuid);
  records.RecordUnclaimedWrite(1);  // Refreshes uuid 1.
  records.RecordUnclaimedWrite(5);  // Evicts uuid 2, the least recent.
  EXPECT_FALSE(records.HasUnclaimedWrite(2));
  for (const uint64_t uuid : {1, 3, 4, 5}) {
    EXPECT_TRUE(records.HasUnclaimedWrite(uuid)) << uuid;
  }

  records.ClearUnclaimedWrite(3);
  records.RecordUnclaimedWrite(6);  // Under the bound again: evicts nothing.
  for (const uint64_t uuid : {1, 4, 5, 6}) {
    EXPECT_TRUE(records.HasUnclaimedWrite(uuid)) << uuid;
  }

  // Committing uuid 4, and evicting its record after a late write, drop its
  // entry without a stale LRU node: the next write past the bound evicts
  // uuid 1.
  EXPECT_THAT(StartAndCommit(records, 4, v1_, absl::Seconds(30), t0_),
              IsEmpty());
  EXPECT_FALSE(records.HasUnclaimedWrite(4));
  records.RecordUnclaimedWrite(4);
  std::vector<uint64_t> evicted;
  EXPECT_EQ(records.FindLive(4, t0_ + absl::Hours(1), &evicted), nullptr);
  EXPECT_THAT(evicted, ElementsAre(4));
  EXPECT_FALSE(records.HasUnclaimedWrite(4));
  records.RecordUnclaimedWrite(7);
  records.RecordUnclaimedWrite(8);
  EXPECT_FALSE(records.HasUnclaimedWrite(1));
  for (const uint64_t uuid : {5, 6, 7, 8}) {
    EXPECT_TRUE(records.HasUnclaimedWrite(uuid)) << uuid;
  }
}

using VersionedReceivesDeathTest = VersionedReceivesTest;

TEST_F(VersionedReceivesDeathTest, ConstructorRejectsInvalidBounds) {
  EXPECT_DEATH(
      { VersionedReceives records(/*max_records=*/2); },
      "at least one finished record; got 2");
  EXPECT_DEATH(
      { VersionedReceives records(kMaxReceiveRecords, absl::Seconds(-1)); },
      "min_retention must not be negative");
  EXPECT_DEATH(
      {
        VersionedReceives records(kMaxReceiveRecords,
                                  kMinReceiveRecordRetention,
                                  /*max_unclaimed_writes=*/0);
      },
      "max_unclaimed_writes must be at least 1");
}

TEST_F(VersionedReceivesDeathTest, StartActiveRejectsInvalidClaims) {
  VersionedReceives records;
  EXPECT_DEATH(records.StartActive(0, SpecFor(v1_), absl::Seconds(30),
                                   t0_ + absl::Seconds(30), t0_),
               "non-zero uuid");
  EXPECT_THAT(records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                                  t0_ + absl::Seconds(30), t0_),
              IsEmpty());
  EXPECT_DEATH(records.StartActive(2, SpecFor(v2_), absl::Seconds(30),
                                   t0_ + absl::Seconds(30), t0_),
               "receive uuid=1 still holds the active claim");
  records.MarkCommitted(1, t0_ + absl::Seconds(1));
  EXPECT_DEATH(
      records.StartActive(1, SpecFor(v1_), absl::Seconds(30),
                          t0_ + absl::Minutes(1), t0_ + absl::Seconds(2)),
      "called for an existing record");
  // Past its retention, the tombstone is evicted and reported instead.
  EXPECT_THAT(
      records.StartActive(1, SpecFor(v2_), absl::Seconds(30),
                          t0_ + absl::Hours(1), t0_ + absl::Minutes(11)),
      ElementsAre(1));
}

TEST_F(VersionedReceivesDeathTest, FinishRejectsNonTerminalStateOrOkOutcome) {
  VersionedReceives records;
  EXPECT_DEATH(records.FinishWithFailure(1, absl::OkStatus(), t0_),
               "needs a non-OK outcome");
  EXPECT_DEATH(records.Finish(1, ReceiveState::kCommitted,
                              absl::InternalError("tiling failed"), t0_),
               "needs kAborted/kExpired/kFailed, got 1");
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

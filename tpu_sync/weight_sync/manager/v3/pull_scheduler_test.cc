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

#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/weight_sync/manager/v3/fake_sampler_fleet.h"
#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Lt;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// Drives an inline `PullScheduler` on a fake clock and records the replies of
// every host.
class Harness {
 public:
  Harness(int32_t replicas, int32_t hosts, int32_t bundles,
          std::vector<std::vector<int32_t>> seeds,
          PullSchedulerOptions options = PullSchedulerOptions()) {
    options.inline_execution = true;
    options.clock = [this] { return now_; };
    absl::StatusOr<std::unique_ptr<PullScheduler>> scheduler =
        PullScheduler::Create(replicas, hosts, bundles, std::move(seeds),
                              std::move(options));
    CHECK_OK(scheduler.status());
    scheduler_ = *std::move(scheduler);
  }

  PullScheduler& scheduler() { return *scheduler_; }

  // Sends a request of host (`replica`, `host`).
  void Ask(int32_t replica, int32_t host, int32_t max_grants,
           std::vector<uint64_t> completed = {},
           std::vector<uint64_t> failed = {}, bool seeded = false,
           bool lost_data = false) {
    PullRequest req;
    req.replica = replica;
    req.host = host;
    req.completed.assign(completed.begin(), completed.end());
    req.failed.assign(failed.begin(), failed.end());
    req.max_grants = max_grants;
    req.seeded = seeded;
    req.lost_data = lost_data;
    scheduler_->Submit(std::move(req), [this, replica, host](PullReply reply) {
      replies_[{replica, host}].push_back(std::move(reply));
    });
  }

  // Removes and returns the replies host (`replica`, `host`) received.
  std::vector<PullReply> Take(int32_t replica, int32_t host = 0) {
    std::vector<PullReply> out;
    out.swap(replies_[{replica, host}]);
    return out;
  }

  // The single reply of a host (fails if there are none or several).
  PullReply TakeOne(int32_t replica, int32_t host = 0) {
    std::vector<PullReply> replies = Take(replica, host);
    CHECK_EQ(replies.size(), 1u) << "replica " << replica << " host " << host;
    return std::move(replies[0]);
  }

  void Advance(absl::Duration d) {
    now_ += d;
    scheduler_->AdvanceTime();
  }

  int64_t Stat(int64_t PullShardStats::* field, int32_t host = 0) {
    return scheduler_->GetStats()[host].*field;
  }

 private:
  absl::Time now_ = absl::UnixEpoch();
  // Declared before `scheduler_`, whose destructor answers parked requests.
  absl::flat_hash_map<std::pair<int32_t, int32_t>, std::vector<PullReply>>
      replies_;
  std::unique_ptr<PullScheduler> scheduler_;
};

std::vector<int32_t> Sources(const PullReply& reply) {
  std::vector<int32_t> out;
  for (const PullGrant& g : reply.grants) out.push_back(g.source_replica);
  return out;
}

std::vector<int32_t> Bundles(const PullReply& reply) {
  std::vector<int32_t> out;
  for (const PullGrant& g : reply.grants) out.push_back(g.bundle_id);
  return out;
}

TEST(PullSchedulerTest, CreateRejectsInvalidArguments) {
  EXPECT_THAT(PullScheduler::Create(0, 1, 1, {}, {}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PullScheduler::Create(2, 1, 1, {{0}}, {}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Seeded bundles given for 1")));
  EXPECT_THAT(PullScheduler::Create(2, 1, 2, {{0}, {}}, {}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Bundle 1 has no seed")));
  EXPECT_THAT(PullScheduler::Create(2, 1, 1, {{0}, {3}}, {}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("out of range")));
}

TEST(PullSchedulerTest, EffectiveGrantBatchSizeIsClampedToUploadCap) {
  PullSchedulerOptions options;
  options.grant_batch_size = 8;
  options.max_concurrent_uploads_per_source = 1;
  EXPECT_EQ(EffectiveGrantBatchSize(options), 1);
  options.allow_grant_batch_above_upload_cap = true;
  EXPECT_EQ(EffectiveGrantBatchSize(options), 8);
  options.grant_batch_size = 0;
  EXPECT_EQ(EffectiveGrantBatchSize(options), 1);
}

TEST(PullSchedulerTest, SeededBundlesFollowSeedLayout) {
  SeedLayout layout;
  layout.num_stripes = 2;
  layout.replication = 2;
  layout.stripe_bundles = {{0, 1}, {2}};
  layout.stripe_seeds = {{0, 2}, {1, 3}};
  EXPECT_THAT(PullScheduler::SeededBundles(layout, 5),
              ElementsAre(ElementsAre(0, 1), ElementsAre(2), ElementsAre(0, 1),
                          ElementsAre(2), IsEmpty()));
}

TEST(PullSchedulerTest, GrantsFromSeedOnceItReportsSeeded) {
  Harness h(/*replicas=*/2, /*hosts=*/1, /*bundles=*/1, {{0}, {}});
  h.Ask(1, 0, /*max_grants=*/1);
  EXPECT_THAT(h.Take(1), IsEmpty());  // Parked: the seed has nothing yet.

  h.Ask(0, 0, 1, {}, {}, /*seeded=*/true);
  EXPECT_EQ(h.TakeOne(0).kind, PullReplyKind::kDone);
  const PullReply grant = h.TakeOne(1);
  ASSERT_EQ(grant.kind, PullReplyKind::kGrants);
  ASSERT_THAT(grant.grants, SizeIs(1));
  EXPECT_EQ(grant.grants[0].bundle_id, 0);
  EXPECT_EQ(grant.grants[0].source_replica, 0);
  EXPECT_EQ(PullScheduler::LeaseBundle(grant.grants[0].lease_id), 0);
  EXPECT_FALSE(h.scheduler().complete());

  h.Ask(1, 0, 1, {grant.grants[0].lease_id});
  EXPECT_EQ(h.TakeOne(1).kind, PullReplyKind::kDone);
  EXPECT_TRUE(h.scheduler().complete());
  EXPECT_OK(h.scheduler().WaitForCompletion(absl::InfinitePast()));
}

TEST(PullSchedulerTest, MarkSeededMakesSeedASource) {
  Harness h(2, 2, 1, {{0}, {}});
  h.Ask(1, 0, 1);
  h.Ask(1, 1, 1);
  h.scheduler().MarkSeeded(0);
  EXPECT_THAT(Sources(h.TakeOne(1, 0)), ElementsAre(0));
  EXPECT_THAT(Sources(h.TakeOne(1, 1)), ElementsAre(0));
}

TEST(PullSchedulerTest, GrantsRarestBundleFirst) {
  // Bundle 0 has two holders, bundle 1 one.
  Harness h(4, 1, 2, {{0, 1}, {0}, {}, {}});
  h.scheduler().MarkSeeded(0);
  h.scheduler().MarkSeeded(1);
  h.Ask(2, 0, 1);
  const PullReply reply = h.TakeOne(2);
  EXPECT_THAT(Bundles(reply), ElementsAre(1));
  EXPECT_THAT(Sources(reply), ElementsAre(0));
}

TEST(PullSchedulerTest, PrefersLeastLoadedSource) {
  PullSchedulerOptions options;
  options.grant_batch_size = 2;
  options.max_concurrent_uploads_per_source = 2;
  Harness h(4, 1, 1, {{0}, {0}, {}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.scheduler().MarkSeeded(1);
  h.Ask(2, 0, 1);
  h.Ask(3, 0, 1);
  const std::vector<int32_t> first = Sources(h.TakeOne(2));
  const std::vector<int32_t> second = Sources(h.TakeOne(3));
  ASSERT_THAT(first, SizeIs(1));
  ASSERT_THAT(second, SizeIs(1));
  EXPECT_NE(first[0], second[0]);
}

TEST(PullSchedulerTest, UploadCapLimitsConcurrentUploads) {
  PullSchedulerOptions options;
  options.grant_batch_size = 1;
  options.max_concurrent_uploads_per_source = 1;
  Harness h(3, 1, 1, {{0}, {}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.Ask(1, 0, 1);
  h.Ask(2, 0, 1);
  const PullReply granted = h.TakeOne(1);
  ASSERT_THAT(granted.grants, SizeIs(1));
  EXPECT_THAT(h.Take(2), IsEmpty());  // The seed's only upload slot is busy.

  // Replica 1 completes: the seed is free again and replica 1 is a source.
  h.Ask(1, 0, 1, {granted.grants[0].lease_id});
  EXPECT_EQ(h.TakeOne(1).kind, PullReplyKind::kDone);
  EXPECT_THAT(h.TakeOne(2).grants, SizeIs(1));
}

TEST(PullSchedulerTest, NewRequestAnswersParkedOne) {
  Harness h(2, 1, 1, {{0}, {}});
  h.Ask(1, 0, 1);
  EXPECT_THAT(h.Take(1), IsEmpty());
  h.Ask(1, 0, 1);
  const PullReply reply = h.TakeOne(1);
  EXPECT_EQ(reply.kind, PullReplyKind::kGrants);
  EXPECT_THAT(reply.grants, IsEmpty());
}

TEST(PullSchedulerTest, LongPollTimesOutWithNoGrants) {
  PullSchedulerOptions options;
  options.long_poll_timeout = absl::Seconds(5);
  Harness h(2, 1, 1, {{0}, {}}, options);
  h.Ask(1, 0, 1);
  h.Advance(absl::Seconds(4));
  EXPECT_THAT(h.Take(1), IsEmpty());
  h.Advance(absl::Seconds(1));
  const PullReply reply = h.TakeOne(1);
  EXPECT_EQ(reply.kind, PullReplyKind::kGrants);
  EXPECT_THAT(reply.grants, IsEmpty());
  EXPECT_EQ(h.Stat(&PullShardStats::long_poll_timeouts), 1);
}

TEST(PullSchedulerTest, FailedPullExcludesSourceForThatPuller) {
  PullSchedulerOptions options;
  options.grant_batch_size = 1;
  options.max_concurrent_uploads_per_source = 1;
  Harness h(3, 1, 1, {{0}, {0}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.scheduler().MarkSeeded(1);
  h.Ask(2, 0, 1);
  const PullReply first = h.TakeOne(2);
  ASSERT_THAT(first.grants, SizeIs(1));
  const int32_t bad = first.grants[0].source_replica;
  h.Ask(2, 0, 1, {}, {first.grants[0].lease_id});
  const PullReply second = h.TakeOne(2);
  ASSERT_THAT(second.grants, SizeIs(1));
  EXPECT_NE(second.grants[0].source_replica, bad);
  EXPECT_EQ(h.Stat(&PullShardStats::failures), 1);
}

TEST(PullSchedulerTest, SourceReportedByKPullersBecomesUnhealthy) {
  PullSchedulerOptions options;
  options.unhealthy_source_reports = 2;
  options.grant_batch_size = 1;
  options.max_concurrent_uploads_per_source = 2;
  // Replica 1 is the second seed of the bundle but has not landed yet.
  Harness h(5, 1, 1, {{0}, {0}, {}, {}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.Ask(2, 0, 1);
  h.Ask(3, 0, 1);
  const PullReply a = h.TakeOne(2);
  const PullReply b = h.TakeOne(3);
  ASSERT_THAT(Sources(a), ElementsAre(0));
  ASSERT_THAT(Sources(b), ElementsAre(0));
  h.Ask(2, 0, 1, {}, {a.grants[0].lease_id});
  h.Ask(3, 0, 1, {}, {b.grants[0].lease_id});
  EXPECT_EQ(h.Stat(&PullShardStats::unhealthy_sources), 1);
  // Replica 4 never failed with replica 0 but must not be sent to it either.
  h.Ask(4, 0, 1);
  EXPECT_THAT(h.Take(4), IsEmpty());
  // The pending seed keeps the bundle alive: no abort.
  EXPECT_OK(h.scheduler().status());
  h.scheduler().MarkSeeded(1);
  EXPECT_THAT(Sources(h.TakeOne(2)), ElementsAre(1));
  EXPECT_THAT(Sources(h.TakeOne(3)), ElementsAre(1));
}

TEST(PullSchedulerTest, ExpiredLeaseOfParkedPullerIsReissued) {
  PullSchedulerOptions options;
  options.lease_timeout = absl::Seconds(10);
  options.long_poll_timeout = absl::Seconds(100);
  options.grant_batch_size = 2;
  options.max_concurrent_uploads_per_source = 2;
  Harness h(2, 1, 2, {{0, 1}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.Ask(1, 0, 2);
  const PullReply grants = h.TakeOne(1);
  ASSERT_THAT(grants.grants, SizeIs(2));
  // One pull completes; the other one hangs while the host stays parked.
  h.Ask(1, 0, 1, {grants.grants[0].lease_id});
  EXPECT_THAT(h.Take(1), IsEmpty());
  h.Advance(absl::Seconds(10));
  EXPECT_EQ(h.Stat(&PullShardStats::expired_leases), 1);
  EXPECT_EQ(h.Stat(&PullShardStats::suspended_hosts), 0);
  const PullReply again = h.TakeOne(1);
  EXPECT_THAT(Bundles(again), ElementsAre(grants.grants[1].bundle_id));
  EXPECT_NE(again.grants[0].lease_id, grants.grants[1].lease_id);
}

TEST(PullSchedulerTest, SilentPullerIsSuspendedAndResumesOnContact) {
  PullSchedulerOptions options;
  options.lease_timeout = absl::Seconds(10);
  options.long_poll_timeout = absl::Seconds(100);
  options.grant_batch_size = 1;
  options.max_concurrent_uploads_per_source = 1;
  Harness h(3, 1, 1, {{0}, {}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.Ask(1, 0, 1);
  const PullReply lost = h.TakeOne(1);
  ASSERT_THAT(lost.grants, SizeIs(1));
  h.Ask(2, 0, 1);
  EXPECT_THAT(h.Take(2), IsEmpty());  // The seed's upload slot is busy.

  // Replica 1 goes silent: its lease is reclaimed and the seed serves 2.
  h.Advance(absl::Seconds(10));
  EXPECT_EQ(h.Stat(&PullShardStats::suspended_hosts), 1);
  EXPECT_THAT(Sources(h.TakeOne(2)), ElementsAre(0));

  // Replica 1 comes back and reports the pull late: it still counts.
  h.Ask(1, 0, 1, {lost.grants[0].lease_id});
  EXPECT_EQ(h.TakeOne(1).kind, PullReplyKind::kDone);
}

TEST(PullSchedulerTest, LostDataPullsEverythingAgain) {
  PullSchedulerOptions options;
  options.grant_batch_size = 2;
  options.max_concurrent_uploads_per_source = 2;
  Harness h(2, 1, 2, {{0, 1}, {}}, options);
  h.scheduler().MarkSeeded(0);
  h.Ask(1, 0, 2);
  PullReply grants = h.TakeOne(1);
  ASSERT_THAT(grants.grants, SizeIs(2));
  h.Ask(1, 0, 2, {grants.grants[0].lease_id, grants.grants[1].lease_id});
  EXPECT_EQ(h.TakeOne(1).kind, PullReplyKind::kDone);

  h.Ask(1, 0, 2, {}, {}, /*seeded=*/false, /*lost_data=*/true);
  grants = h.TakeOne(1);
  EXPECT_THAT(Bundles(grants), UnorderedElementsAre(0, 1));
}

TEST(PullSchedulerTest, AbortsWhenTheLastSourceOfABundleDies) {
  Harness h(3, 1, 2, {{0}, {1}, {}});
  h.scheduler().MarkSeeded(0);
  h.Ask(2, 0, 2);
  const PullReply first = h.TakeOne(2);
  EXPECT_THAT(Bundles(first), ElementsAre(0));
  h.Ask(2, 0, 1);  // Parked: waits for bundle 1.
  EXPECT_THAT(h.Take(2), IsEmpty());

  h.scheduler().MarkReplicaDead(1);
  EXPECT_THAT(h.scheduler().status(),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable,
                                     HasSubstr("bundle 1")));
  const PullReply aborted = h.TakeOne(2);
  EXPECT_EQ(aborted.kind, PullReplyKind::kAborted);
  EXPECT_THAT(aborted.status,
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(h.scheduler().WaitForCompletion(absl::InfiniteFuture()),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));
}

TEST(PullSchedulerTest, DeadReplicaDoesNotBlockCompletion) {
  Harness h(3, 1, 1, {{0}, {}, {}});
  h.scheduler().MarkSeeded(0);
  h.Ask(1, 0, 1);
  const PullReply grant = h.TakeOne(1);
  h.Ask(1, 0, 1, {grant.grants[0].lease_id});
  EXPECT_EQ(h.TakeOne(1).kind, PullReplyKind::kDone);
  EXPECT_FALSE(h.scheduler().complete());
  h.scheduler().MarkReplicaDead(2);
  EXPECT_TRUE(h.scheduler().complete());
  // The dead replica is refused.
  h.Ask(2, 0, 1);
  EXPECT_EQ(h.TakeOne(2).kind, PullReplyKind::kAborted);
}

TEST(PullSchedulerTest, AbortAnswersParkedRequests) {
  Harness h(2, 1, 1, {{0}, {}});
  h.Ask(1, 0, 1);
  h.scheduler().Abort(absl::CancelledError("cancelled by test"));
  const PullReply reply = h.TakeOne(1);
  EXPECT_EQ(reply.kind, PullReplyKind::kAborted);
  EXPECT_THAT(reply.status,
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(h.scheduler().status(),
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
}

TEST(PullSchedulerTest, ThreadedWaitForCompletionHonorsDeadline) {
  absl::StatusOr<std::unique_ptr<PullScheduler>> scheduler =
      PullScheduler::Create(2, 2, 1, {{0}, {}}, PullSchedulerOptions());
  ASSERT_OK(scheduler);
  EXPECT_THAT(
      (*scheduler)->WaitForCompletion(absl::Now() + absl::Milliseconds(20)),
      absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(PullSchedulerTest, ThreadedDestructorCancelsParkedRequests) {
  absl::StatusOr<std::unique_ptr<PullScheduler>> scheduler =
      PullScheduler::Create(2, 1, 1, {{0}, {}}, PullSchedulerOptions());
  ASSERT_OK(scheduler);
  absl::Mutex mu;
  std::optional<PullReply> got;
  PullRequest req;
  req.replica = 1;
  req.max_grants = 1;
  (*scheduler)->Submit(std::move(req), [&](PullReply reply) {
    absl::MutexLock lock(mu);
    got = std::move(reply);
  });
  scheduler->reset();
  absl::MutexLock lock(mu);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(got->kind, PullReplyKind::kAborted);
  EXPECT_THAT(got->status,
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
}

// Hosts that pull every grant instantly, from the reply callbacks, against
// the threaded scheduler.
TEST(PullSchedulerTest, ThreadedTransferCompletes) {
  constexpr int32_t kReplicas = 16;
  constexpr int32_t kHosts = 2;
  constexpr int32_t kBundles = 8;
  std::vector<std::vector<int32_t>> seeds(kReplicas);
  seeds[0] = {0, 1, 2, 3};
  seeds[1] = {4, 5, 6, 7};
  PullSchedulerOptions options;
  options.grant_batch_size = 2;
  options.max_concurrent_uploads_per_source = 2;
  absl::StatusOr<std::unique_ptr<PullScheduler>> scheduler =
      PullScheduler::Create(kReplicas, kHosts, kBundles, seeds, options);
  ASSERT_OK(scheduler);
  PullScheduler* s = scheduler->get();
  // Re-submits with the grants of each reply reported as completed.
  struct Driver {
    PullScheduler* scheduler;
    void Send(int32_t replica, int32_t host, std::vector<uint64_t> completed,
              bool seeded) {
      PullRequest req;
      req.replica = replica;
      req.host = host;
      req.completed.assign(completed.begin(), completed.end());
      req.max_grants = 2;
      req.seeded = seeded;
      scheduler->Submit(std::move(req), [this, replica, host](PullReply r) {
        if (r.kind != PullReplyKind::kGrants) return;
        std::vector<uint64_t> done;
        for (const PullGrant& g : r.grants) done.push_back(g.lease_id);
        if (!done.empty()) Send(replica, host, std::move(done), false);
      });
    }
  };
  Driver driver{s};
  for (int32_t r = 0; r < kReplicas; ++r) {
    for (int32_t h = 0; h < kHosts; ++h) {
      driver.Send(r, h, {}, /*seeded=*/!seeds[r].empty());
    }
  }
  EXPECT_OK(s->WaitForCompletion(absl::Now() + absl::Seconds(30)));
  int64_t grants = 0;
  for (const PullShardStats& stats : s->GetStats()) grants += stats.grants;
  EXPECT_EQ(grants, kHosts * (kReplicas * kBundles - 8));
}

// ---- Fleet simulations ------------------------------------------------------

struct FleetSetup {
  FakeFleetTopology topology;
  double lower_bound = 0;
};

FleetSetup MakeFleet(int32_t replicas, int32_t hosts, int32_t bundles) {
  FleetSetup setup;
  setup.topology.num_replicas = replicas;
  setup.topology.num_hosts = hosts;
  setup.topology.bundle_bytes.assign(bundles, 1.0);
  SeedingOptions seeding;
  seeding.replication = 2;
  absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
      std::vector<int64_t>(bundles, 1), replicas, seeding,
      /*trainer_streams=*/replicas);
  CHECK_OK(layout.status());
  setup.topology.seed_layout = *layout;
  size_t min_seeded = bundles;
  for (const auto& s : PullScheduler::SeededBundles(*layout, replicas)) {
    min_seeded = std::min(min_seeded, s.size());
  }
  setup.lower_bound = bundles - static_cast<double>(min_seeded);
  return setup;
}

PullSchedulerOptions FleetOptions(int32_t grant_batch) {
  PullSchedulerOptions options;
  options.grant_batch_size = grant_batch;
  options.max_concurrent_uploads_per_source = grant_batch;
  // Virtual seconds: one bundle at full ingress takes 1 s.
  options.lease_timeout = absl::Seconds(1000);
  options.long_poll_timeout = absl::Seconds(1000);
  return options;
}

class FleetHappyPathTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(FleetHappyPathTest, FinishesNearTheIngressBound) {
  const int32_t grant_batch = GetParam();
  const FleetSetup setup = MakeFleet(64, 2, 64);
  absl::StatusOr<FakeFleetResult> result = SimulateScheduledPulls(
      setup.topology, FleetOptions(grant_batch), FakeSamplerFleetOptions());
  ASSERT_OK(result);
  ASSERT_OK(result->status);
  EXPECT_TRUE(result->scheduler_complete);
  EXPECT_EQ(result->completed_replicas, 64);
  // Within 5% of the bound plus the k_b-deep start-up.
  EXPECT_THAT(result->finish_time, Le(1.05 * setup.lower_bound + grant_batch));
  EXPECT_THAT(result->max_concurrent_uploads, Le(grant_batch));
  EXPECT_EQ(result->failed_pulls, 0);
}

INSTANTIATE_TEST_SUITE_P(GrantBatch, FleetHappyPathTest,
                         ::testing::Values(1, 4, 8));

TEST(PullSchedulerFleetTest, SlowSourceIsAvoided) {
  const FleetSetup setup = MakeFleet(32, 1, 32);
  FakeSamplerFleetOptions fleet;
  fleet.upload_bandwidth.assign(32, 1.0);
  fleet.upload_bandwidth[3] = 0.01;  // 100x slower uploads.
  fleet.pull_timeout = 4;
  PullSchedulerOptions options = FleetOptions(1);
  options.unhealthy_source_reports = 2;
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(setup.topology, options, fleet);
  ASSERT_OK(result);
  ASSERT_OK(result->status);
  EXPECT_THAT(result->failed_pulls, Gt(0));
  EXPECT_THAT(result->failed_pulls, Le(2));
  // Without avoidance every pull from replica 3 would take 100 s.
  EXPECT_THAT(result->finish_time, Lt(1.5 * setup.lower_bound + 8));
  EXPECT_EQ(result->scheduler_stats[0].unhealthy_sources, 1);
}

TEST(PullSchedulerFleetTest, DeadSourceIsReclaimed) {
  const FleetSetup setup = MakeFleet(32, 2, 32);
  FakeSamplerFleetOptions fleet;
  fleet.dead_at[5] = 3.5;
  fleet.death_detection_delay = 1;
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(setup.topology, FleetOptions(4), fleet);
  ASSERT_OK(result);
  ASSERT_OK(result->status);
  EXPECT_EQ(result->completed_replicas, 31);
  EXPECT_EQ(result->replica_finish_time[5], -1);
  EXPECT_TRUE(result->scheduler_complete);
  EXPECT_THAT(result->failed_pulls, Gt(0));
}

TEST(PullSchedulerFleetTest, UndetectedDeathIsHandledByLeaseExpiry) {
  const FleetSetup setup = MakeFleet(32, 2, 32);
  FakeSamplerFleetOptions fleet;
  fleet.dead_at[5] = 3.5;  // Never reported to the scheduler.
  PullSchedulerOptions options = FleetOptions(4);
  options.lease_timeout = absl::Seconds(20);
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(setup.topology, options, fleet);
  ASSERT_OK(result);
  ASSERT_OK(result->status);
  EXPECT_EQ(result->completed_replicas, 31);
  EXPECT_THAT(result->scheduler_stats[0].suspended_hosts, Gt(0));
}

TEST(PullSchedulerFleetTest, RestartedReplicaPullsAgain) {
  const FleetSetup setup = MakeFleet(32, 2, 32);
  FakeSamplerFleetOptions fleet;
  fleet.restart_at[7] = 10.5;
  fleet.restart_downtime = 2;
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(setup.topology, FleetOptions(4), fleet);
  ASSERT_OK(result);
  ASSERT_OK(result->status);
  EXPECT_EQ(result->rejoins, 1);
  EXPECT_EQ(result->completed_replicas, 32);
  EXPECT_THAT(result->replica_finish_time[7], Gt(12.5));
  EXPECT_TRUE(result->scheduler_complete);
}

TEST(PullSchedulerFleetTest, LosingEverySeedOfAStripeAborts) {
  const FleetSetup setup = MakeFleet(32, 1, 32);
  FakeSamplerFleetOptions fleet;
  for (int32_t seed : setup.topology.seed_layout.stripe_seeds[0]) {
    fleet.dead_at[seed] = 0.5;  // Before any pull completes.
  }
  fleet.death_detection_delay = 0;
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(setup.topology, FleetOptions(1), fleet);
  ASSERT_OK(result);
  EXPECT_THAT(result->status,
              absl_testing::StatusIs(absl::StatusCode::kUnavailable,
                                     HasSubstr("No live replica holds")));
  EXPECT_THAT(result->scheduler_status,
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));
}

TEST(PullSchedulerFleetTest, TrainerWavesDelaySeedsButNotCorrectness) {
  const FleetSetup setup = MakeFleet(32, 2, 32);
  FakeSamplerFleetOptions fleet;
  fleet.trainer_stream_bandwidth = 0.5;
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(setup.topology, FleetOptions(4), fleet);
  ASSERT_OK(result);
  ASSERT_OK(result->status);
  EXPECT_EQ(result->completed_replicas, 32);
  EXPECT_THAT(result->finish_time, Gt(setup.lower_bound));
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

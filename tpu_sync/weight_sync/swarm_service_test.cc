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

#include "tpu_sync/weight_sync/swarm_service.h"

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::tpu_sync::rpc::AcquireBundlePullTokenRequest;
using ::tpu_sync::rpc::AcquireBundlePullTokenResponse;
using ::tpu_sync::rpc::RaidenIdProto;
using ::tpu_sync::rpc::RegisterBundleAvailabilityRequest;
using ::tpu_sync::rpc::RegisterBundleAvailabilityResponse;

RaidenIdProto Sampler(int i) {
  RaidenIdProto unit;
  unit.set_job_name("sampler");
  unit.set_job_replica_id(absl::StrCat(i));
  unit.set_data_name("weights");
  unit.set_data_replica_idx(i);
  return unit;
}

// Single-host samplers 0..n-1 with data endpoint `<prefix><i>`.
std::vector<SwarmService::Participant> Samplers(int n,
                                                absl::string_view prefix) {
  std::vector<SwarmService::Participant> participants(n);
  for (int i = 0; i < n; ++i) {
    participants[i].unit = Sampler(i);
    participants[i].host_data_endpoints = {absl::StrCat(prefix, i)};
  }
  return participants;
}

SwarmService::SessionConfig Config(int32_t max_uploads, absl::Duration timeout,
                                   int32_t target_transfers = 2) {
  return SwarmService::SessionConfig{
      .max_concurrent_uploads_per_source = max_uploads,
      .target_transfers_per_bundle = target_transfers,
      .queue_timeout = timeout,
  };
}

RegisterBundleAvailabilityRequest Register(absl::string_view req_id,
                                           int64_t uuid, int sampler,
                                           int32_t bundle_index) {
  RegisterBundleAvailabilityRequest req;
  req.set_req_id(req_id);
  req.set_uuid(uuid);
  *req.mutable_unit() = Sampler(sampler);
  req.set_host_idx(0);
  req.set_bundle_index(bundle_index);
  return req;
}

AcquireBundlePullTokenRequest Acquire(absl::string_view req_id, int64_t uuid,
                                      int sampler,
                                      std::initializer_list<int32_t> needed) {
  AcquireBundlePullTokenRequest req;
  req.set_req_id(req_id);
  req.set_uuid(uuid);
  *req.mutable_dst_unit() = Sampler(sampler);
  req.set_host_idx(0);
  for (int32_t b : needed) req.add_needed_bundle_indices(b);
  return req;
}

// Returns the snapshot of host 0 of `sampler` (all test samplers have one
// host, so hosts are indexed by sampler).
SwarmService::HostSnapshot HostOf(const SwarmService& swarm,
                                  absl::string_view req_id, int sampler) {
  std::optional<SwarmService::SessionSnapshot> snapshot =
      swarm.GetSessionSnapshot(req_id);
  CHECK(snapshot.has_value());
  return snapshot->hosts[sampler];
}

int32_t ServedCount(const SwarmService& swarm, absl::string_view req_id,
                    int sampler, int32_t bundle_index) {
  const SwarmService::HostSnapshot host = HostOf(swarm, req_id, sampler);
  auto it = host.served_by_bundle.find(bundle_index);
  return it != host.served_by_bundle.end() ? it->second : 0;
}

int32_t QueuedRequests(const SwarmService& swarm, absl::string_view req_id) {
  std::optional<SwarmService::SessionSnapshot> snapshot =
      swarm.GetSessionSnapshot(req_id);
  return snapshot.has_value() ? snapshot->queued_requests : 0;
}

TEST(SwarmServiceTest, TracksAvailabilityEnforcesEgressCapAndPromotesPullers) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession(
      "req_1", /*uuid=*/100, /*num_bundles=*/1, Samplers(3, "127.0.0.1:800"),
      Config(/*max_uploads=*/1, absl::ZeroDuration(), /*target_transfers=*/1)));

  // Before the seed registers, pullers cannot acquire a token.
  const AcquireBundlePullTokenRequest acquire_s1 =
      Acquire("req_1", 100, /*sampler=*/1, {0});
  absl::StatusOr<AcquireBundlePullTokenResponse> empty =
      swarm.AcquireBundlePullToken(acquire_s1);
  ASSERT_OK(empty);
  EXPECT_FALSE(empty->granted());

  // Seed s0 receives bundle 0 from the trainer and registers it.
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_1", 100, 0, 0)));
  EXPECT_THAT(HostOf(swarm, "req_1", 0).available_bundles, ElementsAre(0));

  // s1 acquires a token from s0.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s1 =
      swarm.AcquireBundlePullToken(acquire_s1);
  ASSERT_OK(token_s1);
  EXPECT_TRUE(token_s1->granted());
  EXPECT_EQ(token_s1->assigned_bundle_index(), 0);
  EXPECT_EQ(token_s1->source_unit().job_replica_id(), "0");
  EXPECT_EQ(token_s1->source_data_endpoint(), "127.0.0.1:8000");
  EXPECT_EQ(HostOf(swarm, "req_1", 0).active_uploads, 1);

  // s2 requests bundle 0 while s0 is busy serving s1: throttled.
  const AcquireBundlePullTokenRequest acquire_s2 =
      Acquire("req_1", 100, /*sampler=*/2, {0});
  absl::StatusOr<AcquireBundlePullTokenResponse> congested =
      swarm.AcquireBundlePullToken(acquire_s2);
  ASSERT_OK(congested);
  EXPECT_FALSE(congested->granted());

  // s1 registers bundle 0: releases s0's upload slot and becomes a source.
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_1", 100, 1, 0)));
  EXPECT_EQ(HostOf(swarm, "req_1", 0).active_uploads, 0);
  EXPECT_THAT(HostOf(swarm, "req_1", 1).available_bundles, ElementsAre(0));

  // s1 has served fewer tokens than s0, so s2 pulls from s1.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s2 =
      swarm.AcquireBundlePullToken(acquire_s2);
  ASSERT_OK(token_s2);
  EXPECT_TRUE(token_s2->granted());
  EXPECT_EQ(token_s2->source_unit().job_replica_id(), "1");
  EXPECT_EQ(token_s2->source_data_endpoint(), "127.0.0.1:8001");

  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_1", 100, 2, 0)));
  std::optional<SwarmService::SessionSnapshot> snapshot =
      swarm.GetSessionSnapshot("req_1");
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_TRUE(snapshot->complete);
  EXPECT_TRUE(snapshot->has_worker_activity);
  EXPECT_OK(swarm.WaitForSessionComplete("req_1", absl::ZeroDuration()));
}

TEST(SwarmServiceTest, SupportsConcurrentPipelinedTokensAndIndependentRelease) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession("req_pipe", /*uuid=*/200, /*num_bundles=*/2,
                               Samplers(2, "127.0.0.1:800"),
                               Config(/*max_uploads=*/2, absl::Seconds(20))));
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_pipe", 200, 0, 0)));
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_pipe", 200, 0, 1)));

  absl::StatusOr<AcquireBundlePullTokenResponse> token_1 =
      swarm.AcquireBundlePullToken(Acquire("req_pipe", 200, 1, {0, 1}));
  ASSERT_OK(token_1);
  EXPECT_TRUE(token_1->granted());
  EXPECT_EQ(token_1->assigned_bundle_index(), 0);
  EXPECT_EQ(HostOf(swarm, "req_pipe", 0).active_uploads, 1);

  // s1 prefetches bundle 1 while bundle 0 is in flight.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_2 =
      swarm.AcquireBundlePullToken(Acquire("req_pipe", 200, 1, {1}));
  ASSERT_OK(token_2);
  EXPECT_TRUE(token_2->granted());
  EXPECT_EQ(token_2->assigned_bundle_index(), 1);
  EXPECT_EQ(HostOf(swarm, "req_pipe", 0).active_uploads, 2);

  // Registering bundle 0 releases only bundle 0's token.
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_pipe", 200, 1, 0)));
  EXPECT_EQ(HostOf(swarm, "req_pipe", 0).active_uploads, 1);
  EXPECT_THAT(HostOf(swarm, "req_pipe", 1).available_bundles, ElementsAre(0));

  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_pipe", 200, 1, 1)));
  EXPECT_EQ(HostOf(swarm, "req_pipe", 0).active_uploads, 0);
  EXPECT_THAT(HostOf(swarm, "req_pipe", 1).available_bundles,
              ElementsAre(0, 1));
  EXPECT_TRUE(swarm.GetSessionSnapshot("req_pipe")->complete);
}

TEST(SwarmServiceTest, QueuesConcurrentIncomingRequestsAndDrainsFairly) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession("req_queue", /*uuid=*/101, /*num_bundles=*/1,
                               Samplers(3, "127.0.0.1:800"),
                               Config(/*max_uploads=*/1, absl::Seconds(5))));
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_queue", 101, 0, 0)));

  // s1 saturates s0's single upload slot.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s1 =
      swarm.AcquireBundlePullToken(Acquire("req_queue", 101, 1, {0}));
  ASSERT_OK(token_s1);
  EXPECT_TRUE(token_s1->granted());
  EXPECT_EQ(token_s1->source_unit().job_replica_id(), "0");

  // s2 must queue until a source frees up.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s2;
  std::atomic<bool> s2_finished{false};
  std::thread t_s2([&]() {
    token_s2 = swarm.AcquireBundlePullToken(Acquire("req_queue", 101, 2, {0}));
    s2_finished.store(true);
  });
  while (QueuedRequests(swarm, "req_queue") == 0) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_EQ(QueuedRequests(swarm, "req_queue"), 1);
  EXPECT_FALSE(s2_finished.load());

  // s1 registers bundle 0, which frees s0 and makes s1 a source.
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_queue", 101, 1, 0)));
  t_s2.join();
  ASSERT_OK(token_s2);
  EXPECT_TRUE(token_s2->granted());
  EXPECT_EQ(token_s2->assigned_bundle_index(), 0);
  EXPECT_EQ(QueuedRequests(swarm, "req_queue"), 0);
}

TEST(SwarmServiceTest, QueuedRequestTimesOutCleanlyWhenNoSourceAvailable) {
  SwarmService swarm;
  ASSERT_OK(
      swarm.StartSession("req_timeout", /*uuid=*/102, /*num_bundles=*/1,
                         Samplers(2, "127.0.0.1:800"),
                         Config(/*max_uploads=*/1, absl::Milliseconds(50))));

  const absl::Time start = absl::Now();
  absl::StatusOr<AcquireBundlePullTokenResponse> resp =
      swarm.AcquireBundlePullToken(Acquire("req_timeout", 102, 1, {0}));
  const absl::Duration elapsed = absl::Now() - start;

  ASSERT_OK(resp);
  EXPECT_FALSE(resp->granted());
  EXPECT_GE(elapsed, absl::Milliseconds(40));
  EXPECT_EQ(QueuedRequests(swarm, "req_timeout"), 0);
}

TEST(SwarmServiceTest, PrefetchFromHostWithActiveTokenDoesNotQueue) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession("req_prefetch", /*uuid=*/104, /*num_bundles=*/2,
                               Samplers(2, "127.0.0.1:800"),
                               Config(/*max_uploads=*/1, absl::Seconds(10))));
  ASSERT_OK(
      swarm.RegisterBundleAvailability(Register("req_prefetch", 104, 0, 0)));
  ASSERT_OK(
      swarm.RegisterBundleAvailability(Register("req_prefetch", 104, 0, 1)));

  // s1 gets bundle 0 from s0, saturating s0's single upload slot.
  absl::StatusOr<AcquireBundlePullTokenResponse> first =
      swarm.AcquireBundlePullToken(Acquire("req_prefetch", 104, 1, {0, 1}));
  ASSERT_OK(first);
  EXPECT_TRUE(first->granted());
  EXPECT_EQ(first->assigned_bundle_index(), 0);

  // A prefetch while holding a token returns immediately instead of queueing.
  const absl::Time start = absl::Now();
  absl::StatusOr<AcquireBundlePullTokenResponse> second =
      swarm.AcquireBundlePullToken(Acquire("req_prefetch", 104, 1, {1}));
  ASSERT_OK(second);
  EXPECT_FALSE(second->granted());
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
  EXPECT_EQ(QueuedRequests(swarm, "req_prefetch"), 0);
}

TEST(SwarmServiceTest, EndSessionUnblocksWaitingRequestsImmediately) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession("req_cancel", /*uuid=*/103, /*num_bundles=*/1,
                               Samplers(2, "127.0.0.1:800"),
                               Config(/*max_uploads=*/1, absl::Seconds(10))));

  absl::StatusOr<AcquireBundlePullTokenResponse> resp;
  std::thread t([&]() {
    resp = swarm.AcquireBundlePullToken(Acquire("req_cancel", 103, 1, {0}));
  });
  while (QueuedRequests(swarm, "req_cancel") == 0) {
    absl::SleepFor(absl::Milliseconds(1));
  }

  const absl::Time cancel_start = absl::Now();
  swarm.EndSession("req_cancel");
  t.join();
  EXPECT_LT(absl::Now() - cancel_start, absl::Seconds(2));
  ASSERT_OK(resp);
  EXPECT_FALSE(resp->granted());
  EXPECT_FALSE(swarm.GetSessionSnapshot("req_cancel").has_value());
}

TEST(SwarmServiceTest,
     PrioritizesNextBundleAfterReachingTransferQuotaAndProvidesFallback) {
  SwarmService swarm;
  ASSERT_OK(
      swarm.StartSession("req_quota", /*uuid=*/200, /*num_bundles=*/2,
                         Samplers(6, "127.0.0.1:900"),
                         Config(/*max_uploads=*/1, absl::Milliseconds(100),
                                /*target_transfers=*/2)));
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_quota", 200, 0, 0)));

  // 1. s1 pulls bundle 0 from s0 (first transfer of b0 from s0).
  const AcquireBundlePullTokenRequest acquire_s1 =
      Acquire("req_quota", 200, 1, {0});
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s1 =
      swarm.AcquireBundlePullToken(acquire_s1);
  ASSERT_OK(token_s1);
  EXPECT_TRUE(token_s1->granted());
  EXPECT_EQ(token_s1->source_unit().job_replica_id(), "0");
  EXPECT_EQ(token_s1->assigned_bundle_index(), 0);

  // 2. s0 is busy, so s2 cannot be granted concurrently.
  const AcquireBundlePullTokenRequest acquire_s2 =
      Acquire("req_quota", 200, 2, {0});
  absl::StatusOr<AcquireBundlePullTokenResponse> busy =
      swarm.AcquireBundlePullToken(acquire_s2);
  ASSERT_OK(busy);
  EXPECT_FALSE(busy->granted());

  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_quota", 200, 1, 0)));
  EXPECT_EQ(ServedCount(swarm, "req_quota", 0, 0), 1);

  // 3. s2 pulls bundle 0 from s0 (second transfer of b0 from s0): s0 is in
  // tier 0 (quota started but not met), ahead of the fresh source s1.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s2 =
      swarm.AcquireBundlePullToken(acquire_s2);
  ASSERT_OK(token_s2);
  EXPECT_TRUE(token_s2->granted());
  EXPECT_EQ(token_s2->source_unit().job_replica_id(), "0");
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_quota", 200, 2, 0)));
  EXPECT_EQ(ServedCount(swarm, "req_quota", 0, 0), 2);

  // s0 receives bundle 1 from the trainer.
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_quota", 200, 0, 1)));

  // 4. s3 and s4 pull b0 from the under-quota sources s1 and s2.
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s3 =
      swarm.AcquireBundlePullToken(Acquire("req_quota", 200, 3, {0}));
  ASSERT_OK(token_s3);
  EXPECT_TRUE(token_s3->granted());
  EXPECT_EQ(token_s3->source_unit().job_replica_id(), "1");
  absl::StatusOr<AcquireBundlePullTokenResponse> token_s4 =
      swarm.AcquireBundlePullToken(Acquire("req_quota", 200, 4, {0}));
  ASSERT_OK(token_s4);
  EXPECT_TRUE(token_s4->granted());
  EXPECT_EQ(token_s4->source_unit().job_replica_id(), "2");

  // s1 and s2 are busy; s0 is the only idle source. s0 has met its quota for
  // b0 (tier 2) but not served b1 (tier 1), so s5 gets b1 even though it
  // prefers b0.
  absl::StatusOr<AcquireBundlePullTokenResponse> prioritized =
      swarm.AcquireBundlePullToken(Acquire("req_quota", 200, 5, {0, 1}));
  ASSERT_OK(prioritized);
  EXPECT_TRUE(prioritized->granted());
  EXPECT_EQ(prioritized->source_unit().job_replica_id(), "0");
  EXPECT_EQ(prioritized->assigned_bundle_index(), 1);
  EXPECT_EQ(ServedCount(swarm, "req_quota", 0, 1), 1);
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_quota", 200, 5, 1)));

  // 5. s0 still serves b0 as a fallback once its quota is met.
  absl::StatusOr<AcquireBundlePullTokenResponse> fallback =
      swarm.AcquireBundlePullToken(Acquire("req_quota", 200, 5, {0}));
  ASSERT_OK(fallback);
  EXPECT_TRUE(fallback->granted());
  EXPECT_EQ(fallback->source_unit().job_replica_id(), "0");
  EXPECT_EQ(fallback->assigned_bundle_index(), 0);
  EXPECT_EQ(ServedCount(swarm, "req_quota", 0, 0), 3);
}

TEST(SwarmServiceTest, RegisterGrantsNextTokenUnlessHostHoldsAllBundles) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession(
      "req_auto", /*uuid=*/300, /*num_bundles=*/2, Samplers(2, "127.0.0.1:900"),
      Config(/*max_uploads=*/1, absl::Milliseconds(100),
             /*target_transfers=*/2)));

  // Seeds only register; they get bundles pushed from the trainer.
  absl::StatusOr<RegisterBundleAvailabilityResponse> seed_resp =
      swarm.RegisterBundleAvailability(Register("req_auto", 300, 0, 0));
  ASSERT_OK(seed_resp);
  EXPECT_TRUE(seed_resp->acknowledged());
  EXPECT_FALSE(seed_resp->has_next_pull_token());
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_auto", 300, 0, 1)));

  absl::StatusOr<AcquireBundlePullTokenResponse> token_s1 =
      swarm.AcquireBundlePullToken(Acquire("req_auto", 300, 1, {0, 1}));
  ASSERT_OK(token_s1);
  EXPECT_TRUE(token_s1->granted());
  EXPECT_EQ(token_s1->assigned_bundle_index(), 0);
  EXPECT_EQ(token_s1->source_unit().job_replica_id(), "0");

  // s1 still lacks bundle 1, so registering bundle 0 piggybacks a token.
  RegisterBundleAvailabilityRequest reg_b0 = Register("req_auto", 300, 1, 0);
  reg_b0.set_request_next_bundle(true);
  reg_b0.add_needed_bundle_indices(1);
  absl::StatusOr<RegisterBundleAvailabilityResponse> resp_b0 =
      swarm.RegisterBundleAvailability(reg_b0);
  ASSERT_OK(resp_b0);
  ASSERT_TRUE(resp_b0->has_next_pull_token());
  EXPECT_TRUE(resp_b0->next_pull_token().granted());
  EXPECT_EQ(resp_b0->next_pull_token().assigned_bundle_index(), 1);
  EXPECT_EQ(resp_b0->next_pull_token().source_unit().job_replica_id(), "0");

  // s1 now holds every bundle, so no next token is returned.
  RegisterBundleAvailabilityRequest reg_b1 = Register("req_auto", 300, 1, 1);
  reg_b1.set_request_next_bundle(true);
  absl::StatusOr<RegisterBundleAvailabilityResponse> resp_b1 =
      swarm.RegisterBundleAvailability(reg_b1);
  ASSERT_OK(resp_b1);
  EXPECT_FALSE(resp_b1->has_next_pull_token());
}

TEST(SwarmServiceTest, FailedSourceIsReleasedAndDeprioritized) {
  SwarmService swarm;
  ASSERT_OK(swarm.StartSession("req_fail", /*uuid=*/400, /*num_bundles=*/1,
                               Samplers(3, "127.0.0.1:800"),
                               Config(/*max_uploads=*/1, absl::ZeroDuration(),
                                      /*target_transfers=*/1)));
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_fail", 400, 0, 0)));
  ASSERT_OK(swarm.RegisterBundleAvailability(Register("req_fail", 400, 1, 0)));

  // s2 is assigned s0 (s0 and s1 tie; s0 registered first).
  absl::StatusOr<AcquireBundlePullTokenResponse> first =
      swarm.AcquireBundlePullToken(Acquire("req_fail", 400, 2, {0}));
  ASSERT_OK(first);
  ASSERT_TRUE(first->granted());
  EXPECT_EQ(first->source_unit().job_replica_id(), "0");

  // s0 fails to serve; s2 retries and is moved to s1.
  AcquireBundlePullTokenRequest retry = Acquire("req_fail", 400, 2, {0});
  retry.set_failed_source_replica_id("sampler:0:weights:0");
  absl::StatusOr<AcquireBundlePullTokenResponse> second =
      swarm.AcquireBundlePullToken(retry);
  ASSERT_OK(second);
  ASSERT_TRUE(second->granted());
  EXPECT_EQ(second->source_unit().job_replica_id(), "1");
  EXPECT_EQ(HostOf(swarm, "req_fail", 0).active_uploads, 0);
}

TEST(SwarmServiceTest, RejectsUnknownSessionAndUnit) {
  SwarmService swarm;
  EXPECT_EQ(swarm.StartSession("", 1, 1, {}, {}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      swarm.AcquireBundlePullToken(Acquire("nope", 0, 0, {0})).status().code(),
      absl::StatusCode::kNotFound);

  ASSERT_OK(swarm.StartSession("req", /*uuid=*/500, /*num_bundles=*/1,
                               Samplers(1, "127.0.0.1:800"), {}));
  RegisterBundleAvailabilityRequest unknown_unit = Register("req", 500, 7, 0);
  unknown_unit.mutable_unit()->set_job_name("other");
  EXPECT_EQ(swarm.RegisterBundleAvailability(unknown_unit).status().code(),
            absl::StatusCode::kNotFound);
  // Lookup by uuid alone works.
  EXPECT_OK(swarm.RegisterBundleAvailability(Register("", 500, 0, 0)));
  EXPECT_THAT(HostOf(swarm, "req", 0).available_bundles, ElementsAre(0));
  EXPECT_THAT(HostOf(swarm, "req", 0).served_by_bundle, IsEmpty());
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden

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

#include "tpu_sync/kv_cache/store_monitor.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using global_registry::CreateTestGlobalRegistryServer;
using global_registry::GlobalRegistryClient;
using global_registry::StoreStatus;
using global_registry::TestGlobalRegistryServer;

StoreMonitor::StatusFn ReportFreeBlocks(int64_t free_blocks) {
  return [free_blocks] {
    StoreStatus status;
    status.set_free_blocks(free_blocks);
    return status;
  };
}

class StoreMonitorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_ = CreateTestGlobalRegistryServer();
    client_ = std::make_shared<GlobalRegistryClient>(registry_->channel);
  }

  std::unique_ptr<TestGlobalRegistryServer> registry_;
  std::shared_ptr<GlobalRegistryClient> client_;
};

TEST_F(StoreMonitorTest, HeartbeatsKeepTheRegistrationAlive) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.7:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::Seconds(2), "groupA",
                                  /*evict_tier=*/1)
                  .ok());

  std::atomic<int> reregister_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Milliseconds(500)},
      client_, id, ReportFreeBlocks(7),
      /*reregister_fn=*/[&reregister_calls] {
        ++reregister_calls;
        return absl::OkStatus();
      });
  monitor.Start();

  // Well past the original 2s TTL: only the heartbeats keep it alive.
  absl::SleepFor(absl::Seconds(3));
  EXPECT_TRUE(client_->ResolveStore(id).ok());
  // Alive the whole time, so the heartbeats never needed the re-register
  // fallback.
  EXPECT_EQ(reregister_calls.load(), 0);

  monitor.Stop();
  monitor.Stop();  // Idempotent.

  // Silence past the refreshed deadline: the registration expires.
  absl::SleepFor(absl::Milliseconds(2500));
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

TEST_F(StoreMonitorTest, ReregistersWhenTheRegistrationLapses) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  // No prior registration: the registry answers the first heartbeat with
  // NotFound, standing in for a registry restart or an expired TTL.
  std::atomic<int> reregister_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Milliseconds(300)},
      client_, id, ReportFreeBlocks(7),
      /*reregister_fn=*/[this, &id, &reregister_calls] {
        ++reregister_calls;
        return client_->RegisterStore(id, "10.0.0.7:1111",
                                      /*controller_address=*/"",
                                      /*ttl=*/absl::Seconds(2), "groupA",
                                      /*evict_tier=*/1);
      });
  monitor.Start();

  absl::SleepFor(absl::Seconds(1));
  EXPECT_GE(reregister_calls.load(), 1);
  EXPECT_TRUE(client_->ResolveStore(id).ok());
}

TEST_F(StoreMonitorTest, ReportedStatusFeedsThePlacementRanking) {
  auto register_store = [&](absl::string_view job, int32_t tier) {
    RaidenId id = {std::string(job), "r0", "dataS", 0};
    EXPECT_TRUE(client_
                    ->RegisterStore(id, absl::StrCat(job, ":1111"),
                                    /*controller_address=*/"",
                                    /*ttl=*/absl::ZeroDuration(), "groupA",
                                    tier)
                    .ok());
    return id;
  };
  RaidenId caller = register_store("caller", 0);
  RaidenId crowded = register_store("crowded", 1);
  RaidenId roomy = register_store("roomy", 1);

  StoreStatus crowded_status;
  crowded_status.set_free_blocks(10);
  ASSERT_TRUE(client_->Heartbeat(crowded, crowded_status).ok());

  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Milliseconds(300)},
      client_, roomy, ReportFreeBlocks(50),
      /*reregister_fn=*/[] { return absl::OkStatus(); });
  monitor.Start();
  absl::SleepFor(absl::Seconds(1));  // At least one heartbeat.

  auto targets = client_->GetPlacementTargets(caller, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok());
  ASSERT_EQ(targets->size(), 2);
  EXPECT_EQ((*targets)[0].raiden_id().job_name(), "roomy");
  EXPECT_EQ((*targets)[1].raiden_id().job_name(), "crowded");
}

TEST_F(StoreMonitorTest, SweepRunsOnItsPeriod) {
  RaidenId id = {"sweeping", "r0", "dataS", 0};
  std::atomic<int> sweep_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Hours(1),
                            .sweep_period = absl::Milliseconds(200)},
      client_, id, ReportFreeBlocks(1),
      /*reregister_fn=*/[] { return absl::OkStatus(); },
      /*sweep_fn=*/[&sweep_calls] {
        ++sweep_calls;
        return false;
      });
  monitor.Start();
  absl::SleepFor(absl::Seconds(1));
  EXPECT_GE(sweep_calls.load(), 2);
  monitor.Stop();
}

TEST_F(StoreMonitorTest, ARequestWakesTheSweepBeforeItsPeriod) {
  RaidenId id = {"sweeping", "r0", "dataS", 0};
  std::atomic<int> sweep_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Hours(1),
                            .sweep_period = absl::Hours(1)},
      client_, id, ReportFreeBlocks(1),
      /*reregister_fn=*/[] { return absl::OkStatus(); },
      /*sweep_fn=*/[&sweep_calls] {
        ++sweep_calls;
        return false;
      });
  monitor.Start();
  absl::SleepFor(absl::Milliseconds(200));
  ASSERT_EQ(sweep_calls.load(), 0);  // A period away from the first sweep.

  monitor.RequestSweep();
  for (int i = 0; i < 100 && sweep_calls.load() == 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(sweep_calls.load(), 1);
  monitor.Stop();
}

TEST_F(StoreMonitorTest, SweepKeepsSteppingWhileItReportsMoreWork) {
  RaidenId id = {"sweeping", "r0", "dataS", 0};
  std::atomic<int> sweep_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Hours(1),
                            .sweep_period = absl::Hours(1)},
      client_, id, ReportFreeBlocks(1),
      /*reregister_fn=*/[] { return absl::OkStatus(); },
      // Three steps of pending work, then done: one wake-up must run all
      // four calls back to back.
      /*sweep_fn=*/[&sweep_calls] { return ++sweep_calls < 4; });
  monitor.Start();
  monitor.RequestSweep();
  for (int i = 0; i < 100 && sweep_calls.load() < 4; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(sweep_calls.load(), 4);
  monitor.Stop();
}

TEST_F(StoreMonitorTest, ARequestWithoutASweepIsANoOp) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Hours(1)}, client_, id,
      ReportFreeBlocks(1),
      /*reregister_fn=*/[] { return absl::OkStatus(); });
  monitor.Start();
  monitor.RequestSweep();
  monitor.Stop();
}

TEST_F(StoreMonitorTest, ASuccessfulReregisterTriggersTheRepublish) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  // No prior registration: the first heartbeat lapses, re-registers, and
  // must then drive the republish step by step while it reports more work.
  std::atomic<int> republish_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Milliseconds(300)},
      client_, id, ReportFreeBlocks(7),
      /*reregister_fn=*/
      [this, &id] {
        return client_->RegisterStore(id, "10.0.0.7:1111",
                                      /*controller_address=*/"",
                                      /*ttl=*/absl::Seconds(5), "groupA",
                                      /*evict_tier=*/1);
      },
      /*sweep_fn=*/nullptr,
      /*republish_fn=*/[&republish_calls] { return ++republish_calls < 3; });
  monitor.Start();

  absl::SleepFor(absl::Seconds(1));
  // Exactly three: the steps chain while the callback reports more work,
  // and the healthy heartbeats after the re-register start no new report.
  EXPECT_EQ(republish_calls.load(), 3);
  monitor.Stop();
}

TEST_F(StoreMonitorTest, ARequestStartsARepublishWithoutALapse) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.7:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::ZeroDuration())
                  .ok());
  std::atomic<int> republish_calls{0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Hours(1)}, client_, id,
      ReportFreeBlocks(7),
      /*reregister_fn=*/[] { return absl::OkStatus(); },
      /*sweep_fn=*/nullptr,
      /*republish_fn=*/[&republish_calls] {
        ++republish_calls;
        return false;
      });
  monitor.Start();
  monitor.RequestRepublish();

  absl::SleepFor(absl::Milliseconds(500));
  EXPECT_EQ(republish_calls.load(), 1);
  monitor.Stop();
}

TEST_F(StoreMonitorTest, ARequestWithoutARepublishFnIsANoOp) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  StoreMonitor monitor(
      StoreMonitor::Options{.heartbeat_period = absl::Hours(1)}, client_, id,
      ReportFreeBlocks(1),
      /*reregister_fn=*/[] { return absl::OkStatus(); });
  monitor.Start();
  monitor.RequestRepublish();
  monitor.Stop();
}

TEST_F(StoreMonitorTest, DestructorStopsAnUnstoppedMonitor) {
  RaidenId id = {"monitored", "r0", "dataS", 0};
  {
    StoreMonitor monitor(
        StoreMonitor::Options{.heartbeat_period = absl::Milliseconds(100)},
        client_, id, ReportFreeBlocks(1),
        /*reregister_fn=*/[] { return absl::OkStatus(); });
    monitor.Start();
    absl::SleepFor(absl::Milliseconds(250));
  }  // No Stop(): the destructor joins the thread.

  StoreMonitor never_started(StoreMonitor::Options{}, client_, id,
                             ReportFreeBlocks(1),
                             /*reregister_fn=*/[] { return absl::OkStatus(); });
  never_started.Stop();  // Fine without Start().
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

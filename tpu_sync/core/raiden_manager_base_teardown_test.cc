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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {
namespace {

using ::testing::Each;
using ::testing::Eq;

constexpr absl::string_view kLoopbackIp = "127.0.0.1";
constexpr size_t kNumLayers = 1;
constexpr size_t kNumShards = 1;
constexpr size_t kSliceByteSize = 1024;
constexpr uint64_t kUuid = 4242;
constexpr uint8_t kPayloadByte = 0x5A;
constexpr int kForgetWorkerCount = 2;
// Bounds every wait, so that a regression fails the test instead of hanging.
constexpr absl::Duration kWaitTimeout = absl::Seconds(30);
constexpr absl::Duration kPollInterval = absl::Milliseconds(1);

// One-layer, one-shard manager whose socket transport serves loopback. Its
// receive callback stands for one that is still running when
// StopTransportServer() starts: it waits until teardown has detached the
// transport, then calls ForgetPushProgress().
class TeardownTestManager : public RaidenManagerBase {
 public:
  TeardownTestManager()
      : RaidenManagerBase(
            kNumLayers, std::vector<transport::ShardInfo>(kNumShards),
            kSliceByteSize, /*local_port=*/std::nullopt,
            /*parallelism=*/1, /*bind_ip=*/std::string(kLoopbackIp),
            /*numa_nodes=*/{}, /*host_nics=*/{},
            /*socket_only_transport=*/true),
        host_buffer_(kSliceByteSize) {
    std::vector<LayerInfoBase> layers(kNumLayers);
    for (LayerInfoBase& layer : layers) {
      layer.shards.resize(num_shards_);
    }
    layers_ = StagingArena(std::move(layers));
    SetExternalHostPointers({host_buffer_.data()}, {host_buffer_.size()});
  }

  // Receive callbacks write `host_buffer_`, so the transport stops first.
  ~TeardownTestManager() override { StopTransportServer(); }

  std::string Address() const {
    return absl::StrCat(kLoopbackIp, ":", local_port().value());
  }

  bool IsTransportRunning() const {
    absl::ReaderMutexLock lock(server_init_mu_);
    return server_ != nullptr;
  }

  const std::vector<uint8_t>& host_buffer() const { return host_buffer_; }
  const absl::Notification& callback_started() const {
    return callback_started_;
  }
  const absl::Notification& callback_forgot_progress() const {
    return callback_forgot_progress_;
  }

 protected:
  absl::Status OnDataReceived(uint64_t uuid) override {
    callback_started_.Notify();
    if (!AwaitTransportDetached()) {
      return absl::DeadlineExceededError(
          absl::StrCat("transport of the receive callback for uuid ", uuid,
                       " still attached after ", kWaitTimeout));
    }
    ForgetPushProgress(uuid);
    callback_forgot_progress_.Notify();
    return absl::OkStatus();
  }

 private:
  // Returns whether StopTransportServer() moved `server_` out within
  // kWaitTimeout. Uses try_lock_shared() so that a teardown holding
  // `server_init_mu_` while it joins this callback's thread times out here
  // instead of deadlocking.
  bool AwaitTransportDetached() const {
    const absl::Time deadline = absl::Now() + kWaitTimeout;
    while (absl::Now() < deadline) {
      if (server_init_mu_.try_lock_shared()) {
        const bool detached = server_ == nullptr;
        server_init_mu_.unlock_shared();
        if (detached) return true;
      }
      absl::SleepFor(kPollInterval);
    }
    return false;
  }

  std::vector<uint8_t> host_buffer_;
  absl::Notification callback_started_;
  absl::Notification callback_forgot_progress_;
};

// ForgetPushProgress() used to dereference the null `server_` once
// StopTransportServer() had destroyed the transport.
TEST(RaidenManagerBaseTeardownTest,
     ForgetPushProgressAfterStopTransportServerIsNoOp) {
  TeardownTestManager manager;
  ASSERT_TRUE(manager.IsTransportRunning());
  manager.StopTransportServer();

  manager.ForgetPushProgress(kUuid);

  EXPECT_FALSE(manager.IsTransportRunning());
}

// StopTransportServer() used to hold `server_init_mu_` while it joined the
// transport's threads, so a receive callback taking that lock (here through
// ForgetPushProgress()) deadlocked teardown.
TEST(RaidenManagerBaseTeardownTest,
     ReceiveCallbackCanForgetPushProgressWhileTransportStops) {
  TeardownTestManager receiver;
  TeardownTestManager sender;  // Only pushes.
  ABSL_ASSERT_OK(receiver.RegisterExpectedChunks(kUuid, /*expected_chunks=*/1));
  const std::vector<uint8_t> payload(kSliceByteSize, kPayloadByte);
  ABSL_ASSERT_OK(sender.PushWeightsChunk(
      receiver.Address(), /*dst_shard_idx=*/0, /*dst_offset_bytes=*/0,
      payload.data(), payload.size(), kUuid, /*layer_idx=*/0));
  ASSERT_TRUE(
      receiver.callback_started().WaitForNotificationWithTimeout(kWaitTimeout));

  receiver.StopTransportServer();

  EXPECT_TRUE(receiver.callback_forgot_progress().HasBeenNotified())
      << "StopTransportServer() kept server_init_mu_ locked while joining "
         "the receive callback for uuid "
      << kUuid;
  EXPECT_THAT(receiver.host_buffer(), Each(Eq(kPayloadByte)));
}

// Exercises ForgetPushProgress() concurrently with StopTransportServer(),
// covering the non-null forwarding branch, shared `server_init_mu_`
// synchronization across readers and teardown, and post-stop no-op calls.
TEST(RaidenManagerBaseTeardownTest,
     ConcurrentForgetPushProgressAndStopTransportServer) {
  TeardownTestManager manager;
  ASSERT_TRUE(manager.IsTransportRunning());
  // Seed a progress entry so the first ForgetPushProgress() while running
  // exercises non-empty forwarding into BlockTransport.
  ABSL_ASSERT_OK(manager.RegisterExpectedChunks(kUuid, /*expected_chunks=*/2));

  std::atomic<bool> stop{false};
  std::array<absl::Notification, kForgetWorkerCount> workers_started;
  std::vector<std::thread> workers;
  workers.reserve(kForgetWorkerCount);
  for (int i = 0; i < kForgetWorkerCount; ++i) {
    workers.emplace_back([&, i]() {
      manager.ForgetPushProgress(kUuid);
      workers_started[i].Notify();
      // Guarantee at least one call after Notify() with no happens-before edge
      // against StopTransportServer() unless ForgetPushProgress() locks
      // `server_init_mu_`.
      do {
        manager.ForgetPushProgress(kUuid);
      } while (!stop.load(std::memory_order_relaxed));
    });
  }

  bool all_workers_started = true;
  for (absl::Notification& started : workers_started) {
    if (!started.WaitForNotificationWithTimeout(kWaitTimeout)) {
      all_workers_started = false;
    }
  }

  manager.StopTransportServer();
  stop.store(true, std::memory_order_relaxed);
  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_TRUE(all_workers_started);
  EXPECT_FALSE(manager.IsTransportRunning());
}

}  // namespace
}  // namespace tpu_raiden

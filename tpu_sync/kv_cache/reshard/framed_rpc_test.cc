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

#include "tpu_sync/kv_cache/reshard/framed_rpc.h"

#include <fstream>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

using ::absl_testing::IsOkAndHolds;

// Number of virtual memory areas mapped into this process. Every live thread
// contributes two of them (an 8 MB `rw-p` stack plus a 4 KB `---p` guard
// page), so a thread whose stack is never unmapped is directly visible here.
int CountVirtualMemoryAreas() {
  std::ifstream maps("/proc/self/maps");
  int count = 0;
  std::string line;
  while (std::getline(maps, line)) {
    ++count;
  }
  return count;
}

std::string EchoHandler(const std::string& request) {
  return absl::StrCat("echo:", request);
}

TEST(FramedServerTest, LoopbackRoundTrip) {
  FramedServer server(0, EchoHandler);
  ABSL_ASSERT_OK(server.Bind());
  server.Start();

  SocketFramedTransport transport;
  EXPECT_THAT(transport.Call(absl::StrCat("127.0.0.1:", server.port()), "hello",
                             absl::Seconds(10)),
              IsOkAndHolds("echo:hello"));

  server.Stop();
}

// FramedServer used to retain every connection's std::thread in a vector that
// was only joined in Stop(). The threads exited after their single RPC, but
// their handles stayed joinable, so glibc never unmapped the stacks: 2 leaked
// VMAs per RPC until vm.max_map_count was exhausted.
TEST(FramedServerTest, ConnectionThreadsDoNotLeakVirtualMemoryAreas) {
  constexpr int kWarmupRequests = 16;
  constexpr int kMeasuredRequests = 256;
  // Generous slack: unrelated allocator arenas may map a handful of regions
  // while the test runs. The pre-fix leak was 2 VMAs per request, i.e. ~512.
  constexpr int kAllowedGrowth = 32;

  FramedServer server(0, EchoHandler);
  ABSL_ASSERT_OK(server.Bind());
  server.Start();
  const std::string address = absl::StrCat("127.0.0.1:", server.port());
  SocketFramedTransport transport;

  // Warm up so lazily initialized allocator and TLS mappings are not counted
  // against the measured window.
  for (int i = 0; i < kWarmupRequests; ++i) {
    ABSL_ASSERT_OK(transport.Call(address, "warmup", absl::Seconds(10)));
  }
  const int baseline = CountVirtualMemoryAreas();

  for (int i = 0; i < kMeasuredRequests; ++i) {
    ABSL_ASSERT_OK(
        transport.Call(address, absl::StrCat("req-", i), absl::Seconds(10)));
  }

  // Connection threads are detached, so stack teardown happens shortly after
  // the client observes the response. Poll until the mappings are reclaimed.
  const absl::Time deadline = absl::Now() + absl::Seconds(60);
  int mappings = CountVirtualMemoryAreas();
  while (mappings > baseline + kAllowedGrowth && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(50));
    mappings = CountVirtualMemoryAreas();
  }
  EXPECT_LE(mappings, baseline + kAllowedGrowth)
      << "Leaked " << mappings - baseline << " VMAs over " << kMeasuredRequests
      << " requests; connection thread stacks are not being unmapped.";

  server.Stop();
}

// Stop() must not return while a handler is still running: connection threads
// are detached, so the drain is the only thing keeping them from outliving the
// server object and its handler.
TEST(FramedServerTest, StopWaitsForInFlightConnections) {
  absl::Mutex mu;
  bool handler_entered = false;
  bool handler_finished = false;

  FramedServer server(0, [&](const std::string& request) {
    {
      absl::MutexLock lock(mu);
      handler_entered = true;
    }
    absl::SleepFor(absl::Milliseconds(500));
    {
      absl::MutexLock lock(mu);
      handler_finished = true;
    }
    return std::string("done");
  });
  ABSL_ASSERT_OK(server.Bind());
  server.Start();

  const std::string address = absl::StrCat("127.0.0.1:", server.port());
  std::thread client([&address] {
    SocketFramedTransport transport;
    EXPECT_THAT(transport.Call(address, "slow", absl::Seconds(30)),
                IsOkAndHolds("done"));
  });

  // Block until the handler is mid-flight, then stop the server underneath it.
  {
    absl::MutexLock lock(mu);
    mu.Await(absl::Condition(&handler_entered));
    EXPECT_FALSE(handler_finished);
  }
  server.Stop();

  // Stop() returned, so the connection thread must already have finished.
  {
    absl::MutexLock lock(mu);
    EXPECT_TRUE(handler_finished);
  }
  client.join();
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

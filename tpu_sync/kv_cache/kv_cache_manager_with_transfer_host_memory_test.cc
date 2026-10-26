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

// Host-memory transfers between two managers over loopback without a device.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/core/metrics_collector.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace {

using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::ThrowsMessage;

constexpr size_t kBlockBytes = 128;
constexpr auto kHost = BlockMemory::kHost;
enum class Settled { kNone, kSent, kReceived, kFailed };

class HostPoolManager : public KVCacheManagerWithTransfer {
 public:
  explicit HostPoolManager(size_t layers = 2, size_t shards = 1, int blocks = 4,
                           int parallelism = 1, double timeout_s = 5.0)
      : KVCacheManagerWithTransfer(layers, shards, kBlockBytes, std::nullopt,
                                   blocks, parallelism, 0, 0, 0, 0, timeout_s),
        host_blocks_(blocks) {}
  std::string control_endpoint() const {
    return absl::StrCat("127.0.0.1:", local_control_port());
  }
  int data_port() const { return local_data_port_; }
  static std::string Pattern(char tag, size_t l, size_t s, int b) {
    std::string out(kBlockBytes, '\0');
    for (size_t i = 0; i < kBlockBytes; ++i)
      out[i] = static_cast<char>(tag + l * 7 + s * 29 + b * 13 + i);
    return out;
  }
  void Fill(char tag) {
    for (size_t l = 0; l < base()->num_layers(); ++l)
      for (size_t s = 0; s < base()->num_shards(); ++s)
        for (int b = 0; b < host_blocks_; ++b)
          std::memcpy(base()->GetHostPointer(l, s) + b * kBlockBytes,
                      Pattern(tag, l, s, b).data(), kBlockBytes);
  }
  void ExpectBlocks(
      char tag,
      std::vector<std::pair<int, std::pair<char, int>>> from = {}) const {
    for (size_t l = 0; l < base()->num_layers(); ++l)
      for (size_t s = 0; s < base()->num_shards(); ++s)
        for (int b = 0; b < host_blocks_; ++b) {
          std::pair<char, int> w = {tag, b};
          for (auto [dst, src] : from)
            if (dst == b) w = src;
          auto* p = reinterpret_cast<const char*>(base()->GetHostPointer(l, s));
          EXPECT_EQ(absl::string_view(p + b * kBlockBytes, kBlockBytes),
                    Pattern(w.first, l, s, w.second));
        }
  }
  struct Outcome {
    Settled settled = Settled::kNone;
    TransferDetails details;
  };
  Outcome AwaitOutcome(absl::string_view req_id) {
    std::string id(req_id);
    for (auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
         std::chrono::steady_clock::now() < end;
         std::this_thread::sleep_for(std::chrono::milliseconds(10))) {
      auto r = CompleteReadWithDetails();
      auto in = [&](const auto& v) { return absl::c_linear_search(v, id); };
      Settled s = in(r.failed)     ? Settled::kFailed
                  : in(r.received) ? Settled::kReceived
                  : in(r.sent)     ? Settled::kSent
                                   : Settled::kNone;
      if (s != Settled::kNone) return {s, r.details[id]};
    }
    return {};
  }
  Settled AwaitSettled(absl::string_view r) { return AwaitOutcome(r).settled; }
  const int host_blocks_;
};

auto TransferHostBlocks(HostPoolManager& p, HostPoolManager& c,
                        const std::string& req, uint64_t uuid,
                        const std::vector<int64_t>& src,
                        const std::vector<int64_t>& dst) {
  EXPECT_GT(p.NotifyForRead(req, uuid, src, {}, kHost), 0);
  c.StartRead(req, uuid, p.control_endpoint(), src, dst, 1, dst, {}, kHost);
  auto received = c.AwaitOutcome(req), sent = p.AwaitOutcome(req);
  EXPECT_EQ(received.settled, Settled::kReceived);
  EXPECT_EQ(sent.settled, Settled::kSent);
  return std::pair(received.details, sent.details);
}

std::optional<int64_t> MetricField(absl::string_view dump,
                                   absl::string_view field) {
  size_t at = dump.find(absl::StrCat("\"", field, "\": "));
  if (at == absl::string_view::npos) return std::nullopt;
  auto rest = dump.substr(at + field.size() + 4);
  int64_t v = 0;
  return absl::SimpleAtoi(rest.substr(0, rest.find_first_of(",\n")), &v)
             ? std::optional(v)
             : std::nullopt;
}

TEST(HostMemoryTransferTest, HostBlocksLandAcrossShardsAndParallelStreams) {
  HostPoolManager p(3, 2, 6, 2), c(3, 2, 6, 2);
  p.Fill('p');
  c.Fill('c');
  TransferHostBlocks(p, c, "sharded_req", 72, {5, 1, 3, 0}, {2, 4, 0, 5});
  c.ExpectBlocks('c',
                 {{2, {'p', 5}}, {4, {'p', 1}}, {0, {'p', 3}}, {5, {'p', 0}}});
  p.ExpectBlocks('p');
}

TEST(HostMemoryTransferTest, LocalBlockIdsMayBeOmittedAndUuidReused) {
  HostPoolManager p, c;
  p.Fill('p');
  c.Fill('c');
  constexpr uint64_t kUuid = 73;
  ASSERT_GT(p.NotifyForRead("first", kUuid, {2, 0}, {}, kHost), 0);
  c.StartRead("first", kUuid, p.control_endpoint(), {2, 0}, {}, 1, {{1, 3}}, {},
              kHost);
  EXPECT_EQ(c.AwaitSettled("first"), Settled::kReceived);
  EXPECT_EQ(p.AwaitSettled("first"), Settled::kSent);
  p.Fill('q');
  TransferHostBlocks(p, c, "second", kUuid, {2}, {0});
  c.ExpectBlocks('c', {{1, {'p', 2}}, {3, {'p', 0}}, {0, {'q', 2}}});
}

TEST(HostMemoryTransferTest, DetailsAndMetricsDescribeASettledHostTransfer) {
  HostPoolManager p, c;
  auto metrics = std::make_shared<MetricsCollector>("/nonexistent/sysfs/net");
  c.SetMetricsCollector(metrics);
  p.Fill('p');
  c.Fill('c');
  auto [received, sent] =
      TransferHostBlocks(p, c, "detailed", 75, {0, 1}, {2, 3});
  EXPECT_EQ(received.peer, p.control_endpoint());
  EXPECT_THAT(received.elapsed_ns, Optional(Ge(0)));
  EXPECT_FALSE(received.code.has_value());
  EXPECT_FALSE(received.phase.has_value());
  EXPECT_THAT(sent.peer, Optional(HasSubstr(absl::StrCat(":", c.data_port()))));
  EXPECT_THAT(sent.elapsed_ns, Optional(Ge(0)));
  EXPECT_FALSE(sent.code.has_value());
  const std::string dump = metrics->DumpMetricsToString();
  ASSERT_THAT(dump, HasSubstr("\"req_id\": \"detailed\""));
  EXPECT_THAT(MetricField(dump, "connection_setup_us"), Optional(Ge(0)));
  EXPECT_THAT(MetricField(dump, "network_transit_us"), Optional(Ge(0)));
  EXPECT_THAT(MetricField(dump, "e2e_duration_us"), Optional(Ge(0)));
}

TEST(HostMemoryTransferTest, InvalidHostBlockArgumentsAreRejected) {
  HostPoolManager m(1, 1, 2);
  auto check = [](auto fn, absl::string_view msg) {
    EXPECT_THAT(fn, ThrowsMessage<std::invalid_argument>(HasSubstr(msg)));
  };
  check([&] { m.StartRead("r0", 77, "h:1", {0}, {0}, 1, {}, {}, kHost); },
        "local_host_block_ids is required");
  check([&] { m.NotifyForRead("r1", 78, {0, 2}, {}, kHost); }, "host block 2");
  check([&] { m.StartRead("r2", 79, "h:1", {0}, {5}, 1, {{5}}, {}, kHost); },
        "host block 5");
  check([&] { m.StartRead("r3", 80, "h:1", {0}, {0}, 1, {{7}}); },
        "host block 7");
  check(
      [&] { m.StartRead("4", 0, "h", {0, 1}, {1, 1}, 1, {{1, 1}}, {}, kHost); },
      "host block 1 is named more than once");
  auto [sent, received, failed] = m.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && received.empty() && failed.empty());
}

TEST(HostMemoryTransferTest, DeviceReadsAreRejectedOnAHostPoolServer) {
  HostPoolManager p;
  auto bad = ThrowsMessage<std::invalid_argument>(HasSubstr("source=kHost"));
  EXPECT_THAT([&] { p.NotifyForRead("dev", 82, {0}); }, bad);
  EXPECT_THAT(
      [&] { p.NotifyForRead("dev", 82, {0}, {}, BlockMemory::kDevice); }, bad);
  auto [sent, received, failed] = p.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && failed.empty());
}

TEST(HostMemoryTransferTest, AReadNobodyPullsExpiresAwaitingThePull) {
  HostPoolManager p;
  auto past = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  ASSERT_GT(p.NotifyForRead("unpulled", 83, {0}, past, kHost), 0);
  auto outcome = p.AwaitOutcome("unpulled");
  EXPECT_EQ(outcome.settled, Settled::kFailed);
  EXPECT_EQ(outcome.details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(outcome.details.phase, "await_pull");
}

TEST(HostMemoryTransferTest, TheConsumerFailsWhenTheProducerNeverRegisters) {
  HostPoolManager p(2, 1, 4, 1, 7.0), c(2, 1, 4, 1, 7.0);
  c.StartRead("unregistered", 84, p.control_endpoint(), {0}, {1}, 1, {{1}}, {},
              kHost);
  auto outcome = c.AwaitOutcome("unregistered");
  EXPECT_EQ(outcome.settled, Settled::kFailed);
  EXPECT_EQ(outcome.details.code, "FAILED_PRECONDITION");
  EXPECT_THAT(outcome.details.reason,
              Optional(HasSubstr("no read registered")));
  EXPECT_EQ(outcome.details.phase, "connect");
  EXPECT_EQ(outcome.details.peer, p.control_endpoint());
}

TEST(HostMemoryTransferTest, AFailedPushFailsTheSendAndExpiresTheReceive) {
  GetFaultInjector().Reset();
  absl::Cleanup reset_faults = [] { GetFaultInjector().Reset(); };
  ABSL_ASSERT_OK(GetFaultInjector().Install(
      {{.hook = std::string(hooks::kRaidenManagerBaseH2hWrite),
        .probability = 1.0}}));
  HostPoolManager p(2, 1, 4, 1, 1.0), c(2, 1, 4, 1, 1.0);
  p.Fill('p');
  c.Fill('c');
  ASSERT_GT(p.NotifyForRead("broken_push", 85, {0, 1}, {}, kHost), 0);
  c.StartRead("broken_push", 85, p.control_endpoint(), {0, 1}, {2, 3}, 1,
              {{2, 3}}, {}, kHost);
  auto sent = p.AwaitOutcome("broken_push");
  EXPECT_EQ(sent.settled, Settled::kFailed);
  EXPECT_EQ(sent.details.code, "INTERNAL");
  EXPECT_THAT(sent.details.reason, Optional(HasSubstr("injected fault")));
  EXPECT_EQ(sent.details.phase, "push");
  auto received = c.AwaitOutcome("broken_push");
  EXPECT_EQ(received.settled, Settled::kFailed);
  EXPECT_EQ(received.details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(received.details.phase, "receive");
  c.ExpectBlocks('c');
}

}  // namespace
}  // namespace tpu_raiden

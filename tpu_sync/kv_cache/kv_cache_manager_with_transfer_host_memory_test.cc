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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/metrics_collector.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace {

using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::ThrowsMessage;

constexpr size_t kBlockBytes = 128;
constexpr auto kHost = BlockLocation::kHost;
enum class Settled { kNone, kSent, kReceived, kFailed };

// How a test manager is built. Defaults give a two-layer, single-shard host
// pool of four blocks that serves host reads on a free control port.
struct HostPoolConfig {
  size_t layers = 2;
  size_t shards = 1;
  int blocks = 4;
  int parallelism = 1;
  double timeout_s = 5.0;
  int control_port = 0;  // -1 opens no control server.
  // Staging slots carved out of the host pool, as a device-transfer manager
  // has; a host pool that holds host transfers has none.
  int64_t staging_slots = 0;
  int64_t staging_slot_blocks = 0;
  // Leases staging dynamically from the host pool
  // (TPU_RAIDEN_DYNAMIC_HOST_STAGING).
  bool dynamic_host_staging = false;
  // Bounded host staging: fewer host layer buffers than layers.
  std::optional<int> staging_layers = std::nullopt;
};

class HostPoolManager : public KVCacheManagerWithTransfer {
 public:
  explicit HostPoolManager(const HostPoolConfig& config = {})
      : KVCacheManagerWithTransfer(
            std::make_unique<kv_cache::KVCacheManagerBase>(
                config.layers, config.shards,
                std::vector<size_t>(config.layers, kBlockBytes),
                /*local_port=*/std::nullopt,
                /*host_blocks_to_allocate=*/config.blocks, config.parallelism,
                /*host_allocator=*/nullptr, /*bind_ip=*/std::nullopt,
                config.staging_layers),
            /*node_id=*/0, config.control_port,
            /*max_blocks=*/config.staging_slot_blocks,
            /*num_slots=*/config.staging_slots, config.timeout_s),
        host_blocks_(config.blocks) {
    if (config.dynamic_host_staging) {
      staging_allocator_ = CreateStagingBlockAllocator(
          base(), config.staging_slots, config.staging_slot_blocks,
          /*dynamic_host_staging=*/true);
    }
  }
  std::string control_endpoint() const {
    return absl::StrCat("127.0.0.1:", local_control_port());
  }
  int data_port() const { return local_data_port_; }

  // The bytes block `block` of (`layer`, `shard`) holds on a manager filled
  // with `tag`: distinct per block, so a misplaced block is caught.
  static std::string Pattern(char tag, size_t layer, size_t shard, int block) {
    std::string pattern(kBlockBytes, '\0');
    for (size_t i = 0; i < kBlockBytes; ++i) {
      pattern[i] =
          static_cast<char>(tag + layer * 7 + shard * 29 + block * 13 + i);
    }
    return pattern;
  }

  void Fill(char tag) {
    for (size_t layer = 0; layer < base()->num_layers(); ++layer) {
      for (size_t shard = 0; shard < base()->num_shards(); ++shard) {
        for (int block = 0; block < host_blocks_; ++block) {
          TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buffer,
                               base()->GetHostBuffer(layer, shard));
          std::memcpy(host_buffer.data() + block * kBlockBytes,
                      Pattern(tag, layer, shard, block).data(), kBlockBytes);
        }
      }
    }
  }

  // A host block expected to hold another block's pattern after a transfer:
  // `block` of this manager holds what block `source_block` of the manager
  // filled with `source_tag` held.
  struct ExpectedBlock {
    int block;
    char source_tag;
    int source_block;
  };

  // Expects every host block to hold its own `own_tag` pattern, except the
  // blocks `overrides` name, which hold the pattern transferred into them.
  void ExpectBlocks(char own_tag,
                    const std::vector<ExpectedBlock>& overrides = {}) const {
    for (size_t layer = 0; layer < base()->num_layers(); ++layer) {
      for (size_t shard = 0; shard < base()->num_shards(); ++shard) {
        for (int block = 0; block < host_blocks_; ++block) {
          char expected_tag = own_tag;
          int expected_block = block;
          for (const ExpectedBlock& expected : overrides) {
            if (expected.block == block) {
              expected_tag = expected.source_tag;
              expected_block = expected.source_block;
            }
          }
          TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> host_buffer,
                               base()->GetHostBuffer(layer, shard));
          const char* host_layer =
              reinterpret_cast<const char*>(host_buffer.data());
          EXPECT_EQ(
              absl::string_view(host_layer + block * kBlockBytes, kBlockBytes),
              Pattern(expected_tag, layer, shard, expected_block))
              << "layer " << layer << " shard " << shard << " block " << block;
        }
      }
    }
  }

  struct Outcome {
    Settled settled = Settled::kNone;
    TransferDetails details;
  };

  // Polls poll_stats until `req_id` settles (up to a minute). Every poll
  // reports each settled transfer once, so other requests settled meanwhile
  // are kept for their own AwaitOutcome.
  Outcome AwaitOutcome(absl::string_view req_id) {
    const std::string id(req_id);
    const auto give_up =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < give_up) {
      CompleteReadResult result = CompleteReadWithDetails();
      for (const std::string& sent : result.sent) {
        settled_[sent] = {Settled::kSent, result.details[sent]};
      }
      for (const std::string& received : result.received) {
        settled_[received] = {Settled::kReceived, result.details[received]};
      }
      for (const std::string& failed : result.failed) {
        settled_[failed] = {Settled::kFailed, result.details[failed]};
      }
      if (auto it = settled_.find(id); it != settled_.end()) {
        const Outcome outcome = it->second;
        settled_.erase(it);
        return outcome;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return {};
  }

  const int host_blocks_;
  // Settled transfers no AwaitOutcome has claimed yet, by req_id.
  absl::flat_hash_map<std::string, Outcome> settled_;
};

struct TransferOutcome {
  TransferDetails received;
  TransferDetails sent;
};

// Pulls `producer_blocks` of `producer`'s host pool into `consumer_blocks` of
// `consumer`'s, and expects both ends to settle successfully.
TransferOutcome TransferHostBlocks(
    HostPoolManager& producer, HostPoolManager& consumer,
    const std::string& req_id, uint64_t uuid,
    const std::vector<int64_t>& producer_blocks,
    const std::vector<int64_t>& consumer_blocks) {
  EXPECT_GT(producer.NotifyForRead(req_id, uuid, producer_blocks, {}, kHost),
            0);
  consumer.StartRead(req_id, uuid, producer.control_endpoint(), producer_blocks,
                     consumer_blocks, /*parallelism=*/1, {}, {}, kHost);
  const HostPoolManager::Outcome received = consumer.AwaitOutcome(req_id);
  const HostPoolManager::Outcome sent = producer.AwaitOutcome(req_id);
  EXPECT_EQ(received.settled, Settled::kReceived);
  EXPECT_EQ(sent.settled, Settled::kSent);
  return {received.details, sent.details};
}

// Expects the metrics of transfer `uuid` to describe `req_id` with its
// timestamps in order: start, first packet, last packet, end.
void ExpectOrderedMetrics(const MetricsCollector& metrics, uint64_t uuid,
                          absl::string_view req_id) {
  const std::optional<TransferMetrics> recorded =
      metrics.GetTransferMetrics(uuid);
  ASSERT_TRUE(recorded.has_value());
  EXPECT_EQ(recorded->req_id, req_id);
  EXPECT_LE(recorded->start_time, recorded->first_packet_time);
  EXPECT_LE(recorded->first_packet_time, recorded->last_packet_time);
  EXPECT_LE(recorded->last_packet_time, recorded->end_time);
}

TEST(HostMemoryTransferTest, HostBlocksLandAcrossShardsAndParallelStreams) {
  const HostPoolConfig sharded = {
      .layers = 3, .shards = 2, .blocks = 6, .parallelism = 2};
  HostPoolManager p(sharded), c(sharded);
  p.Fill('p');
  c.Fill('c');
  TransferHostBlocks(p, c, "sharded_req", 72, {5, 1, 3, 0}, {2, 4, 0, 5});
  c.ExpectBlocks('c', {{2, 'p', 5}, {4, 'p', 1}, {0, 'p', 3}, {5, 'p', 0}});
  p.ExpectBlocks('p');
}

TEST(HostMemoryTransferTest, AUuidIsReusableOnceItsTransferSettles) {
  HostPoolManager p, c;
  p.Fill('p');
  c.Fill('c');
  constexpr uint64_t kUuid = 73;
  TransferHostBlocks(p, c, "first", kUuid, {2, 0}, {1, 3});
  p.Fill('q');
  TransferHostBlocks(p, c, "second", kUuid, {2}, {0});
  c.ExpectBlocks('c', {{1, 'p', 2}, {3, 'p', 0}, {0, 'q', 2}});
}

TEST(HostMemoryTransferTest, AConsumerWithoutAControlServerCanPull) {
  HostPoolManager p, c({.control_port = -1});
  p.Fill('p');
  c.Fill('c');
  // The consumer advertises its data port in the pull request; a port of 0
  // would send the producer's push nowhere and expire the read.
  EXPECT_GT(c.data_port(), 0);
  TransferHostBlocks(p, c, "pull_only", 74, {1, 2}, {3, 0});
  c.ExpectBlocks('c', {{3, 'p', 1}, {0, 'p', 2}});
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
  ExpectOrderedMetrics(*metrics, 75, "detailed");
  EXPECT_THAT(metrics->DumpMetricsToString(),
              HasSubstr("\"req_id\": \"detailed\""));
}

TEST(HostMemoryTransferTest, ASingleLayerHostTransferRecordsItsFirstPacket) {
  // The transport reports a layer before the layer's blocks, so a one-layer
  // receive settles on its layer report: the first packet must be recorded
  // there, or the metrics describe a transfer that began at the epoch.
  HostPoolManager p({.layers = 1}), c({.layers = 1});
  auto metrics = std::make_shared<MetricsCollector>("/nonexistent/sysfs/net");
  c.SetMetricsCollector(metrics);
  p.Fill('p');
  c.Fill('c');
  TransferHostBlocks(p, c, "one_layer", 76, {0, 1}, {2, 3});
  c.ExpectBlocks('c', {{2, 'p', 0}, {3, 'p', 1}});
  ExpectOrderedMetrics(*metrics, 76, "one_layer");
}

TEST(HostMemoryTransferTest, InvalidHostBlockArgumentsAreRejected) {
  HostPoolManager m({.layers = 1, .blocks = 2});
  auto check = [](auto fn, absl::string_view msg) {
    EXPECT_THAT(fn, ThrowsMessage<std::invalid_argument>(HasSubstr(msg)));
  };
  check([&] { m.NotifyForRead("r1", 78, {0, 2}, {}, kHost); }, "host block 2");
  check([&] { m.StartRead("r2", 79, "h:1", {0}, {5}, 1, {}, {}, kHost); },
        "host block 5");
  check([&] { m.StartRead("r3", 80, "h:1", {0}, {0}, 1, {{7}}); },
        "host block 7");
  check([&] { m.StartRead("r4", 81, "h:1", {0, 1}, {1, 1}, 1, {}, {}, kHost); },
        "host block 1 is named more than once");
  // A host destination names its blocks in local_block_ids alone.
  check([&] { m.StartRead("r5", 82, "h:1", {0}, {0}, 1, {{1}}, {}, kHost); },
        "local_host_block_ids must not be given");
  auto [sent, received, failed] = m.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && received.empty() && failed.empty());
}

TEST(HostMemoryTransferTest, DeviceReadsAreRejectedOnAHostPoolServer) {
  HostPoolManager p;
  auto bad = ThrowsMessage<std::invalid_argument>(HasSubstr("host reads only"));
  EXPECT_THAT([&] { p.NotifyForRead("dev", 82, {0}); }, bad);
  EXPECT_THAT(
      [&] { p.NotifyForRead("dev", 82, {0}, {}, BlockLocation::kDevice); },
      bad);
  auto [sent, received, failed] = p.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && failed.empty());
}

TEST(HostMemoryTransferTest,
     HostTransfersAreRejectedOnAManagerWithStagingSlots) {
  // Staging slots are carved out of the same host pool a host transfer
  // addresses by block id, so a host block could alias a live device transfer.
  HostPoolManager m({.staging_slots = 1, .staging_slot_blocks = 2});
  auto bad = ThrowsMessage<std::invalid_argument>(HasSubstr("num_slots == 0"));
  EXPECT_THAT([&] { m.NotifyForRead("send", 86, {2}, {}, kHost); }, bad);
  EXPECT_THAT(
      [&] { m.StartRead("recv", 87, "h:1", {0}, {2}, 1, {}, {}, kHost); }, bad);
  auto [sent, received, failed] = m.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && received.empty() && failed.empty());
}

TEST(HostMemoryTransferTest, HostTransfersAreRejectedWithDynamicHostStaging) {
  // Dynamic staging reserves no slots but leases blocks of the same host pool
  // on demand, so a named host block could still alias a device transfer.
  HostPoolManager m({.dynamic_host_staging = true});
  auto bad = ThrowsMessage<std::invalid_argument>(HasSubstr("dynamic host"));
  EXPECT_THAT([&] { m.NotifyForRead("send", 90, {2}, {}, kHost); }, bad);
  EXPECT_THAT(
      [&] { m.StartRead("recv", 91, "h:1", {0}, {2}, 1, {}, {}, kHost); }, bad);
  auto [sent, received, failed] = m.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && received.empty() && failed.empty());
}

TEST(HostMemoryTransferTest, HostTransfersAreRejectedWithBoundedHostStaging) {
  // With fewer host layer buffers than layers, "host block b of layer l" does
  // not exist for every layer, and the transport would wait for a leased
  // layer buffer that a host transfer never frees.
  HostPoolManager m({.layers = 2, .staging_layers = 1});
  auto bad = ThrowsMessage<std::invalid_argument>(
      HasSubstr("num_staging_layers == num_layers"));
  EXPECT_THAT([&] { m.NotifyForRead("send", 88, {0}, {}, kHost); }, bad);
  EXPECT_THAT(
      [&] { m.StartRead("recv", 89, "h:1", {0}, {1}, 1, {}, {}, kHost); }, bad);
  auto [sent, received, failed] = m.CompleteReadRaw();
  EXPECT_TRUE(sent.empty() && received.empty() && failed.empty());
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
  // Above kPullRegistrationGrace (5 s): the producer's grace must lapse before
  // the consumer's control deadline, or the outcome is a dead heat with
  // DEADLINE_EXCEEDED. Costs ~5 s.
  const HostPoolConfig patient = {.timeout_s = 7.0};
  HostPoolManager p(patient), c(patient);
  c.StartRead("unregistered", 84, p.control_endpoint(), {0}, {1}, 1, {}, {},
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
  const HostPoolConfig impatient = {.timeout_s = 1.0};
  HostPoolManager p(impatient), c(impatient);
  p.Fill('p');
  c.Fill('c');
  ASSERT_GT(p.NotifyForRead("broken_push", 85, {0, 1}, {}, kHost), 0);
  c.StartRead("broken_push", 85, p.control_endpoint(), {0, 1}, {2, 3}, 1, {},
              {}, kHost);
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

TEST(HostMemoryTransferTest,
     TransferEndpointProtoRoundTripPreservesPeregrineEndpoint) {
  RaidenTransferEndpoint ep{
      .endpoint = "10.0.0.1:10000",
      .shards = {0, 1},
      .layer_host_addrs = {},
      .peregrine_endpoint = "10.0.0.1:20000",
  };
  const ::tpu_sync::proto::RaidenTransferEndpointProto proto = ep.ToProto();
  EXPECT_EQ(proto.peregrine_endpoint(), "10.0.0.1:20000");

  const RaidenTransferEndpoint round_tripped =
      RaidenTransferEndpoint::FromProto(proto);
  EXPECT_EQ(round_tripped, ep);
  EXPECT_EQ(round_tripped.peregrine_endpoint, "10.0.0.1:20000");

  RaidenTransferEndpoint different_pg = ep;
  different_pg.peregrine_endpoint = "10.0.0.1:20001";
  EXPECT_FALSE(ep == different_pg);
}

TEST(HostMemoryTransferTest,
     BuildEndpointsPopulatesPeregrineEndpointWhenEnabled) {
  auto cleanup =
      absl::MakeCleanup([] { unsetenv("TPU_RAIDEN_ENABLE_PEREGRINE"); });
  setenv("TPU_RAIDEN_ENABLE_PEREGRINE", "6", 1);

  HostPoolManager sender({.control_port = -1});
  HostPoolManager receiver({.control_port = -1});
  sender.Fill('s');
  receiver.Fill('r');

  ASSERT_GT(receiver.base()->peregrine_local_port(), 0);
  const std::string expected_pg_ep =
      absl::StrCat(receiver.base()->local_ip(), ":",
                   receiver.base()->peregrine_local_port());

  const std::vector<RaidenTransferEndpoint> data_eps =
      receiver.get_local_data_endpoints();
  ASSERT_FALSE(data_eps.empty());
  for (const auto& ep : data_eps) {
    EXPECT_EQ(ep.peregrine_endpoint, expected_pg_ep);
  }
  const std::vector<RaidenTransferEndpoint> local_eps =
      receiver.get_local_endpoints();
  ASSERT_FALSE(local_eps.empty());
  for (const auto& ep : local_eps) {
    EXPECT_EQ(ep.peregrine_endpoint, expected_pg_ep);
  }

  // KVManagerHolder H2hWrite with RaidenTransferEndpoint descriptors registers
  // the peer's Peregrine endpoint on the sender and completes the explicit push
  // over Peregrine.
  KVManagerHolder sender_holder(&sender);
  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture write_fut,
                       sender_holder.H2hWrite(data_eps, /*src_offsets=*/{1, 3},
                                              /*dst_offsets=*/{2, 0}));
  ABSL_ASSERT_OK(write_fut.Await());
  receiver.ExpectBlocks('r', {{2, 's', 1}, {0, 's', 3}});
}

}  // namespace
}  // namespace tpu_raiden

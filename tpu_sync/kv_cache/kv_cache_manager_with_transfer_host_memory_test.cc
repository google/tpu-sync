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

// Host-memory transfers between two managers over loopback, without a
// device: a producer pushes blocks of its host pool as they are (no staging,
// no device-to-host copy), and a consumer lands them in named blocks of its
// own host pool and settles there (no host-to-device copy). This is the path
// a hierarchical cache takes to hand host-resident KV to a peer's host cache.

#include <algorithm>
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
#include "absl/cleanup/cleanup.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/metrics_collector.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace {

using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Optional;

constexpr size_t kBlockBytes = 128;
constexpr double kTimeoutS = 5.0;
// Long enough for the slowest sanitizer run; the sessions under test settle
// (or expire) well before it.
constexpr absl::Duration kAwait = absl::Seconds(60);

// How a manager over a host pool is shaped; defaults are the smallest shape
// that exercises every layer.
struct HostPoolOptions {
  size_t num_layers = 2;
  size_t num_shards = 1;
  int host_blocks = 4;
  int parallelism = 1;
  double timeout_s = kTimeoutS;
};

// A manager over a host pool with no staging slots and no device attached:
// the shape of a host cache. Its control server takes an ephemeral port.
class HostPoolManager : public KVCacheManagerWithTransfer {
 public:
  explicit HostPoolManager(const HostPoolOptions& options = {})
      : KVCacheManagerWithTransfer(
            options.num_layers, options.num_shards, kBlockBytes,
            /*local_port=*/std::nullopt, options.host_blocks,
            options.parallelism, /*node_id=*/0, /*local_control_port=*/0,
            /*max_blocks=*/0, /*num_slots=*/0, options.timeout_s),
        num_shards_(options.num_shards),
        host_blocks_(options.host_blocks) {}

  std::string control_endpoint() const {
    return absl::StrCat("127.0.0.1:", local_control_port());
  }
  int data_port() const { return local_data_port_; }

  // One host block of one layer and shard, filled with a byte pattern unique
  // to (tag, layer, shard, block) so a misrouted block is told apart from a
  // missing one.
  static std::string Pattern(char tag, size_t layer, size_t shard, int block) {
    std::string bytes(kBlockBytes, '\0');
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<char>(tag + layer * 7 + shard * 29 + block * 13 +
                                   i);
    }
    return bytes;
  }

  void WriteBlock(size_t layer, size_t shard, int block,
                  absl::string_view bytes) {
    ASSERT_EQ(bytes.size(), kBlockBytes);
    std::memcpy(base()->GetHostPointer(layer, shard) + block * kBlockBytes,
                bytes.data(), bytes.size());
  }

  std::string ReadBlock(size_t layer, size_t shard, int block) const {
    const uint8_t* ptr =
        base()->GetHostPointer(layer, shard) + block * kBlockBytes;
    return std::string(reinterpret_cast<const char*>(ptr), kBlockBytes);
  }

  // Fills every block of every layer and shard with its `tag` pattern.
  void Fill(char tag) {
    for (size_t layer = 0; layer < base()->num_layers(); ++layer) {
      for (size_t shard = 0; shard < num_shards_; ++shard) {
        for (int block = 0; block < host_blocks_; ++block) {
          WriteBlock(layer, shard, block, Pattern(tag, layer, shard, block));
        }
      }
    }
  }

  // Expects every block of every layer and shard to hold its `tag` pattern,
  // except `expected_from`: (block -> (source tag, source block)) pairs that
  // hold the named source block's pattern instead.
  void ExpectBlocks(char tag,
                    const std::vector<std::pair<int, std::pair<char, int>>>&
                        expected_from = {}) const {
    for (size_t layer = 0; layer < base()->num_layers(); ++layer) {
      for (size_t shard = 0; shard < num_shards_; ++shard) {
        for (int block = 0; block < host_blocks_; ++block) {
          char want_tag = tag;
          int want_block = block;
          for (const auto& [dst, src] : expected_from) {
            if (dst == block) {
              want_tag = src.first;
              want_block = src.second;
            }
          }
          EXPECT_EQ(ReadBlock(layer, shard, block),
                    Pattern(want_tag, layer, shard, want_block))
              << "layer " << layer << " shard " << shard << " block " << block;
        }
      }
    }
  }

  enum class Settled { kNone, kSent, kReceived, kFailed };
  struct Outcome {
    Settled settled = Settled::kNone;
    TransferDetails details;
  };

  // Polls completion reports until `req_id` settles as sent, received, or
  // failed, or `timeout` passes, and returns how it settled with its details.
  Outcome AwaitOutcome(absl::string_view req_id,
                       absl::Duration timeout = kAwait) {
    const absl::Time give_up = absl::Now() + timeout;
    const std::string wanted(req_id);
    while (true) {
      CompleteReadResult completions = CompleteReadWithDetails();
      auto has = [&](const std::vector<std::string>& ids) {
        return std::find(ids.begin(), ids.end(), wanted) != ids.end();
      };
      Outcome outcome;
      if (has(completions.failed)) {
        outcome.settled = Settled::kFailed;
      } else if (has(completions.received)) {
        outcome.settled = Settled::kReceived;
      } else if (has(completions.sent)) {
        outcome.settled = Settled::kSent;
      }
      if (outcome.settled != Settled::kNone) {
        auto it = completions.details.find(wanted);
        if (it != completions.details.end()) {
          outcome.details = it->second;
        }
        return outcome;
      }
      if (absl::Now() >= give_up) {
        return outcome;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  Settled AwaitSettled(absl::string_view req_id,
                       absl::Duration timeout = kAwait) {
    return AwaitOutcome(req_id, timeout).settled;
  }

 private:
  const size_t num_shards_;
  const int host_blocks_;
};

// Runs one host-to-host transfer of `src_blocks` of `producer` into
// `dst_blocks` of `consumer`, registered under `req_id` / `uuid`, and
// expects both ends to settle successfully.
void TransferHostBlocks(HostPoolManager& producer, HostPoolManager& consumer,
                        absl::string_view req_id, uint64_t uuid,
                        const std::vector<int64_t>& src_blocks,
                        const std::vector<int64_t>& dst_blocks) {
  const std::string req(req_id);
  ASSERT_GT(producer.NotifyForRead(req, uuid, src_blocks,
                                   /*deadline=*/std::nullopt,
                                   BlockMemory::kHost),
            0);
  consumer.StartRead(req, uuid, producer.control_endpoint(), src_blocks,
                     dst_blocks, /*parallelism=*/1, dst_blocks,
                     /*deadline=*/std::nullopt, BlockMemory::kHost);
  EXPECT_EQ(consumer.AwaitSettled(req), HostPoolManager::Settled::kReceived);
  EXPECT_EQ(producer.AwaitSettled(req), HostPoolManager::Settled::kSent);
}

// The integer `field` of the first transfer in a MetricsCollector JSON dump.
std::optional<int64_t> MetricField(absl::string_view dump,
                                   absl::string_view field) {
  const std::string key = absl::StrCat("\"", field, "\": ");
  const size_t at = dump.find(key);
  if (at == absl::string_view::npos) return std::nullopt;
  absl::string_view rest = dump.substr(at + key.size());
  rest = rest.substr(0, rest.find_first_of(",\n"));
  int64_t value = 0;
  if (!absl::SimpleAtoi(rest, &value)) return std::nullopt;
  return value;
}

TEST(HostMemoryTransferTest, HostBlocksLandInThePeersHostPool) {
  HostPoolManager producer;
  HostPoolManager consumer;
  producer.Fill('p');
  consumer.Fill('c');

  // Blocks land at the ids the consumer names, crossed on purpose.
  TransferHostBlocks(producer, consumer, "host_req", /*uuid=*/71,
                     /*src_blocks=*/{1, 3}, /*dst_blocks=*/{2, 0});

  // The named blocks hold the producer's bytes; the others keep the
  // consumer's own, and the producer's blocks are read, not moved.
  consumer.ExpectBlocks('c', {{2, {'p', 1}}, {0, {'p', 3}}});
  producer.ExpectBlocks('p');
}

TEST(HostMemoryTransferTest, HostBlocksLandAcrossShardsAndParallelStreams) {
  HostPoolOptions options;
  options.num_layers = 3;
  options.num_shards = 2;
  options.host_blocks = 6;
  options.parallelism = 2;
  HostPoolManager producer(options);
  HostPoolManager consumer(options);
  producer.Fill('p');
  consumer.Fill('c');

  TransferHostBlocks(producer, consumer, "sharded_req", /*uuid=*/72,
                     /*src_blocks=*/{5, 1, 3, 0}, /*dst_blocks=*/{2, 4, 0, 5});

  consumer.ExpectBlocks(
      'c', {{2, {'p', 5}}, {4, {'p', 1}}, {0, {'p', 3}}, {5, {'p', 0}}});
  producer.ExpectBlocks('p');
}

TEST(HostMemoryTransferTest, AUuidIsReusableOnceItsTransferSettles) {
  HostPoolManager producer;
  HostPoolManager consumer;
  producer.Fill('p');
  consumer.Fill('c');
  constexpr uint64_t kUuid = 73;

  TransferHostBlocks(producer, consumer, "first", kUuid, /*src_blocks=*/{0},
                     /*dst_blocks=*/{1});
  // The second transfer under the same uuid starts from clean accounting on
  // both ends: it lands, rather than being taken for a repeat of the first.
  producer.Fill('q');
  TransferHostBlocks(producer, consumer, "second", kUuid, /*src_blocks=*/{2},
                     /*dst_blocks=*/{3});

  consumer.ExpectBlocks('c', {{1, {'p', 0}}, {3, {'q', 2}}});
}

TEST(HostMemoryTransferTest, LocalBlockIdsMayBeOmittedForAHostDestination) {
  HostPoolManager producer;
  HostPoolManager consumer;
  producer.Fill('p');
  consumer.Fill('c');
  const std::vector<int64_t> src_blocks = {2, 0};
  const std::vector<int64_t> dst_blocks = {1, 3};

  ASSERT_GT(producer.NotifyForRead("no_device_blocks", /*uuid=*/74, src_blocks,
                                   /*deadline=*/std::nullopt,
                                   BlockMemory::kHost),
            0);
  // A host destination has no device blocks to name.
  consumer.StartRead("no_device_blocks", /*uuid=*/74,
                     producer.control_endpoint(), src_blocks,
                     /*local_block_ids=*/{}, /*parallelism=*/1, dst_blocks,
                     /*deadline=*/std::nullopt, BlockMemory::kHost);

  EXPECT_EQ(consumer.AwaitSettled("no_device_blocks"),
            HostPoolManager::Settled::kReceived);
  EXPECT_EQ(producer.AwaitSettled("no_device_blocks"),
            HostPoolManager::Settled::kSent);
  consumer.ExpectBlocks('c', {{1, {'p', 2}}, {3, {'p', 0}}});
}

TEST(HostMemoryTransferTest, DetailsDescribeASettledHostTransfer) {
  HostPoolManager producer;
  HostPoolManager consumer;
  producer.Fill('p');
  consumer.Fill('c');

  ASSERT_GT(producer.NotifyForRead("detailed", /*uuid=*/75, {1},
                                   /*deadline=*/std::nullopt,
                                   BlockMemory::kHost),
            0);
  consumer.StartRead("detailed", /*uuid=*/75, producer.control_endpoint(), {1},
                     {2}, /*parallelism=*/1, {{2}},
                     /*deadline=*/std::nullopt, BlockMemory::kHost);

  HostPoolManager::Outcome received = consumer.AwaitOutcome("detailed");
  EXPECT_EQ(received.settled, HostPoolManager::Settled::kReceived);
  EXPECT_EQ(received.details.peer, producer.control_endpoint());
  EXPECT_THAT(received.details.elapsed_ns, Optional(Ge(0)));
  EXPECT_FALSE(received.details.code.has_value());
  EXPECT_FALSE(received.details.phase.has_value());

  HostPoolManager::Outcome sent = producer.AwaitOutcome("detailed");
  EXPECT_EQ(sent.settled, HostPoolManager::Settled::kSent);
  // The producer pushed to the consumer's data port, not its control port.
  EXPECT_THAT(sent.details.peer,
              Optional(HasSubstr(absl::StrCat(":", consumer.data_port()))));
  EXPECT_THAT(sent.details.elapsed_ns, Optional(Ge(0)));
  EXPECT_FALSE(sent.details.code.has_value());
}

TEST(HostMemoryTransferTest, AHostReceiveReportsItsNetworkPhaseToMetrics) {
  HostPoolManager producer;
  HostPoolManager consumer;
  // No NIC directory: byte counters stay empty and only timings are recorded.
  auto metrics = std::make_shared<MetricsCollector>("/nonexistent/sysfs/net");
  consumer.SetMetricsCollector(metrics);
  producer.Fill('p');
  consumer.Fill('c');

  TransferHostBlocks(producer, consumer, "metered", /*uuid=*/76,
                     /*src_blocks=*/{0, 1}, /*dst_blocks=*/{2, 3});

  // A host receive ends with the network, so its last packet is the last
  // layer: the transit is measured from the first packet, not from the epoch.
  const std::string dump = metrics->DumpMetricsToString();
  ASSERT_THAT(dump, HasSubstr("\"req_id\": \"metered\""));
  EXPECT_THAT(MetricField(dump, "connection_setup_us"), Optional(Ge(0)));
  EXPECT_THAT(MetricField(dump, "network_transit_us"), Optional(Ge(0)));
  EXPECT_THAT(MetricField(dump, "e2e_duration_us"), Optional(Ge(0)));
}

TEST(HostMemoryTransferTest, HostReadWithoutHostBlocksIsRejected) {
  HostPoolManager consumer;
  EXPECT_THROW(
      consumer.StartRead("no_blocks", /*uuid=*/77, "127.0.0.1:1",
                         /*remote_block_ids=*/{0},
                         /*local_block_ids=*/{0}, /*parallelism=*/1,
                         /*local_host_block_ids=*/std::nullopt,
                         /*deadline=*/std::nullopt, BlockMemory::kHost),
      std::invalid_argument);
}

TEST(HostMemoryTransferTest, HostBlocksOutsideThePoolAreRejected) {
  HostPoolOptions options;
  options.num_layers = 1;
  options.host_blocks = 2;
  HostPoolManager manager(options);
  try {
    manager.NotifyForRead("send_outside", /*uuid=*/78, /*block_ids=*/{0, 2},
                          /*deadline=*/std::nullopt, BlockMemory::kHost);
    FAIL() << "a host block past the pool was registered";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), HasSubstr("host block 2"));
  }
  try {
    manager.StartRead("read_outside", /*uuid=*/79, "127.0.0.1:1",
                      /*remote_block_ids=*/{0}, /*local_block_ids=*/{5},
                      /*parallelism=*/1, /*local_host_block_ids=*/{{5}},
                      /*deadline=*/std::nullopt, BlockMemory::kHost);
    FAIL() << "a host block past the pool was read into";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), HasSubstr("host block 5"));
  }
  // Explicit staging for a device destination names host blocks too.
  try {
    manager.StartRead("stage_outside", /*uuid=*/80, "127.0.0.1:1",
                      /*remote_block_ids=*/{0}, /*local_block_ids=*/{0},
                      /*parallelism=*/1, /*local_host_block_ids=*/{{7}},
                      /*deadline=*/std::nullopt, BlockMemory::kDevice);
    FAIL() << "a host block past the pool was staged into";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), HasSubstr("host block 7"));
  }
  // None of the registrations took.
  auto [sent, received, failed] = manager.CompleteReadRaw();
  EXPECT_THAT(sent, IsEmpty());
  EXPECT_THAT(received, IsEmpty());
  EXPECT_THAT(failed, IsEmpty());
}

TEST(HostMemoryTransferTest, ARepeatedHostDestinationBlockIsRejected) {
  HostPoolManager consumer;
  try {
    consumer.StartRead("twice", /*uuid=*/81, "127.0.0.1:1",
                       /*remote_block_ids=*/{1, 3}, /*local_block_ids=*/{2, 2},
                       /*parallelism=*/1, /*local_host_block_ids=*/{{2, 2}},
                       /*deadline=*/std::nullopt, BlockMemory::kHost);
    FAIL() << "two blocks were read into one host block";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), HasSubstr("host block 2 is named more than once"));
  }
  auto [sent, received, failed] = consumer.CompleteReadRaw();
  EXPECT_THAT(received, IsEmpty());
  EXPECT_THAT(failed, IsEmpty());
}

TEST(HostMemoryTransferTest, DeviceReadsAreRejectedOnAHostPoolServer) {
  HostPoolManager producer;
  // The default source is the device: a caller that forgets source=kHost on a
  // host cache is told so at registration, not through a staged copy into
  // the pool it addresses.
  try {
    producer.NotifyForRead("device_read", /*uuid=*/82, /*block_ids=*/{0});
    FAIL() << "a device read was registered on a host pool server";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), HasSubstr("source=kHost"));
  }
  EXPECT_THROW(
      producer.NotifyForRead("device_read", /*uuid=*/82, /*block_ids=*/{0},
                             /*deadline=*/std::nullopt, BlockMemory::kDevice),
      std::invalid_argument);
  auto [sent, received, failed] = producer.CompleteReadRaw();
  EXPECT_THAT(sent, IsEmpty());
  EXPECT_THAT(failed, IsEmpty());
}

TEST(HostMemoryTransferTest, AReadNobodyPullsExpiresAwaitingThePull) {
  HostPoolManager producer;
  ASSERT_GT(producer.NotifyForRead("unpulled", /*uuid=*/83, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1),
                                   BlockMemory::kHost),
            0);

  HostPoolManager::Outcome outcome = producer.AwaitOutcome("unpulled");

  EXPECT_EQ(outcome.settled, HostPoolManager::Settled::kFailed);
  EXPECT_EQ(outcome.details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(outcome.details.phase, "await_pull");
}

TEST(HostMemoryTransferTest, TheConsumerFailsWhenTheProducerNeverRegisters) {
  // The producer waits out a 5s registration grace before it turns a pull
  // away, and the consumer's handshake deadline is the transfer timeout: the
  // timeout must clear the grace, or the two would be a dead heat.
  HostPoolOptions options;
  options.timeout_s = 7.0;
  HostPoolManager producer(options);
  HostPoolManager consumer(options);

  consumer.StartRead("unregistered", /*uuid=*/84, producer.control_endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{1},
                     /*parallelism=*/1, /*local_host_block_ids=*/{{1}},
                     /*deadline=*/std::nullopt, BlockMemory::kHost);

  // The producer turns the pull away once its registration grace lapses; the
  // consumer fails in the handshake, naming the producer.
  HostPoolManager::Outcome outcome = consumer.AwaitOutcome("unregistered");
  EXPECT_EQ(outcome.settled, HostPoolManager::Settled::kFailed);
  EXPECT_EQ(outcome.details.code, "FAILED_PRECONDITION");
  EXPECT_THAT(outcome.details.reason, Optional(HasSubstr("no read registered")));
  EXPECT_EQ(outcome.details.phase, "connect");
  EXPECT_EQ(outcome.details.peer, producer.control_endpoint());
}

TEST(HostMemoryTransferTest, AFailedPushFailsTheSendAndExpiresTheReceive) {
  GetFaultInjector().Reset();
  absl::Cleanup reset_faults = [] { GetFaultInjector().Reset(); };
  FaultInjectionRule failed_push;
  failed_push.hook = std::string(hooks::kRaidenManagerBaseH2hWrite);
  failed_push.probability = 1.0;
  ABSL_ASSERT_OK(GetFaultInjector().Install({failed_push}));
  HostPoolOptions options;
  options.timeout_s = 1.0;
  HostPoolManager producer(options);
  HostPoolManager consumer(options);
  producer.Fill('p');
  consumer.Fill('c');

  ASSERT_GT(producer.NotifyForRead("broken_push", /*uuid=*/85, {0, 1},
                                   /*deadline=*/std::nullopt,
                                   BlockMemory::kHost),
            0);
  consumer.StartRead("broken_push", /*uuid=*/85, producer.control_endpoint(),
                     {0, 1}, {2, 3}, /*parallelism=*/1, {{2, 3}},
                     /*deadline=*/std::nullopt, BlockMemory::kHost);

  // The send fails in its push phase with the injected error.
  HostPoolManager::Outcome sent = producer.AwaitOutcome("broken_push");
  EXPECT_EQ(sent.settled, HostPoolManager::Settled::kFailed);
  EXPECT_EQ(sent.details.code, "INTERNAL");
  EXPECT_THAT(sent.details.reason, Optional(HasSubstr("injected fault")));
  EXPECT_EQ(sent.details.phase, "push");

  // Nothing reaches the consumer, which times out waiting for data rather
  // than hanging, and its blocks are untouched.
  HostPoolManager::Outcome received = consumer.AwaitOutcome("broken_push");
  EXPECT_EQ(received.settled, HostPoolManager::Settled::kFailed);
  EXPECT_EQ(received.details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(received.details.phase, "receive");
  consumer.ExpectBlocks('c');
}

}  // namespace
}  // namespace tpu_raiden

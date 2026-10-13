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

#include "tpu_sync/core/raiden_manager_base.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {
namespace {

// Test subclass to populate protected layers_ and implement AllocateBlocks
// One ShardInfo per entry of |numa_nodes|, shard `i` on `numa_nodes[i]`.
std::vector<transport::ShardInfo> ShardsOnNuma(std::vector<int> numa_nodes) {
  std::vector<transport::ShardInfo> shards;
  shards.reserve(numa_nodes.size());
  for (size_t sh = 0; sh < numa_nodes.size(); ++sh) {
    shards.push_back(
        {.local_index = static_cast<int>(sh), .numa_node = numa_nodes[sh]});
  }
  return shards;
}

class TestRaidenManager : public RaidenManagerBase {
 public:
  TestRaidenManager(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt, int parallelism = 1,
      std::vector<int> numa_nodes = {},
      std::vector<HostNicAddress> mock_nics = GetLocalHostNicAddresses(),
      std::vector<transport::ShardInfo> shards = {})
      : RaidenManagerBase(num_layers, num_shards, slice_byte_size, local_port,
                          parallelism, /*bind_ip=*/std::nullopt,
                          std::move(numa_nodes), std::move(mock_nics),
                          std::move(shards)) {
    std::vector<LayerInfoBase> layers(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      layers[l].shards.resize(num_shards);
    }
    layers_ = LayerArena(std::move(layers));
  }

  absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) override {
    std::vector<int> ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
      ids[i] = i;
    }
    return ids;
  }
};

TEST(RaidenManagerBaseTest, LifecycleAndConfig) {
  TestRaidenManager manager(/*num_layers=*/2, /*num_shards=*/4,
                            /*slice_byte_size=*/1024);

  EXPECT_EQ(manager.num_layers(), 2);
  EXPECT_EQ(manager.num_shards(), 4);
  EXPECT_EQ(manager.slice_byte_size(), 1024);
  EXPECT_TRUE(manager.local_port().has_value());
  EXPECT_GT(manager.local_port().value(), 0);
}

TEST(RaidenManagerBaseTest, SetExternalHostPointersWithPinnedAllocator) {
  // Setup TPU PJRT and Pinned Host Allocator
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(pjrt_manager->client()));

  // Setup Manager
  size_t num_layers = 2;
  size_t num_shards = 2;
  size_t slice_size = 1024;
  TestRaidenManager manager(num_layers, num_shards, slice_size);

  // Allocate and bind host pointers
  std::vector<HostBufferAllocation> allocations;
  std::vector<const uint8_t*> host_ptrs;
  std::vector<size_t> host_sizes;

  size_t total_buffers = num_layers * num_shards;
  allocations.reserve(total_buffers);
  host_ptrs.reserve(total_buffers);
  host_sizes.reserve(total_buffers);

  for (size_t i = 0; i < total_buffers; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(slice_size));
    allocations.push_back(std::move(alloc));
    host_ptrs.push_back(allocations.back().ptr);
    host_sizes.push_back(slice_size);
  }

  manager.SetExternalHostPointers(host_ptrs, host_sizes);

  // Verify
  size_t idx = 0;
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t sh = 0; sh < num_shards; ++sh) {
      EXPECT_EQ(manager.GetHostPointer(l, sh), host_ptrs[idx]);
      EXPECT_EQ(manager.GetHostSize(l, sh), slice_size);
      idx++;
    }
  }
}

TEST(RaidenManagerBaseTest, E2eLoopbackTransferH2h) {
  // Setup TPU PJRT and Allocator
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(pjrt_manager->client()));

  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_size = 1024;  // 1KB

  // Instantiate Sender and Receiver on ephemeral ports
  TestRaidenManager sender(num_layers, num_shards, slice_size);
  TestRaidenManager receiver(num_layers, num_shards, slice_size);

  ASSERT_TRUE(sender.local_port().has_value());
  ASSERT_TRUE(receiver.local_port().has_value());

  // Allocate pinned buffers for sender and receiver
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation send_alloc,
                          allocator->Allocate(slice_size));
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation recv_alloc,
                          allocator->Allocate(slice_size));

  sender.SetExternalHostPointers({send_alloc.ptr}, {slice_size});
  receiver.SetExternalHostPointers({recv_alloc.ptr}, {slice_size});

  // Write a test pattern to sender's buffer
  std::memset(send_alloc.ptr, 0x5A, slice_size);
  // Zero out receiver's buffer
  std::memset(recv_alloc.ptr, 0, slice_size);

  // H2H Write (Push) from Sender to Receiver
  std::string peer_address =
      absl::StrCat("127.0.0.1:", receiver.local_port().value());

  // Sender pushes block 0 to Receiver. Receiver will store it using the
  // allocated block ID (0).
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> pushed_ids,
                          sender.H2hWriteDirect(peer_address, {0}));
  EXPECT_EQ(pushed_ids.size(), 1);
  EXPECT_EQ(pushed_ids[0], 0);

  // Give a brief moment for socket transfer to complete and trigger ACKs
  absl::SleepFor(absl::Milliseconds(50));

  // Verify Receiver's buffer got populated!
  for (size_t i = 0; i < slice_size; ++i) {
    ASSERT_EQ(recv_alloc.ptr[i], 0x5A)
        << "Mismatch at index " << i << ", expected 0x5A, got "
        << static_cast<int>(recv_alloc.ptr[i]);
  }

  // Now let's test Pull. Zero out Sender's buffer, and fill Receiver's buffer
  // with a new pattern.
  std::memset(send_alloc.ptr, 0, slice_size);
  std::memset(recv_alloc.ptr, 0x3C, slice_size);

  // Sender pulls block 0 from Receiver
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> pulled_ids,
                          sender.H2hReadDirect(peer_address, {0}));
  EXPECT_EQ(pulled_ids.size(), 1);
  EXPECT_EQ(pulled_ids[0], 0);

  absl::SleepFor(absl::Milliseconds(50));

  for (size_t i = 0; i < slice_size; ++i) {
    ASSERT_EQ(send_alloc.ptr[i], 0x3C)
        << "Mismatch at index " << i << ", expected 0x3C, got "
        << static_cast<int>(send_alloc.ptr[i]);
  }
}

TEST(RaidenManagerBaseTest, IpCollectionNumaLocalData) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth0", "10.0.0.1", 0, NicClassification::kControlPlane},
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
      {"eth2", "10.0.0.3", 1, NicClassification::kControlPlane},
      {"eth3", "10.0.0.4", 1, NicClassification::kDataPlane},
      {"eth4", "10.0.0.5", 1, NicClassification::kDataPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{1}, mock_nics);

  auto ips = manager.local_ips();

  ASSERT_EQ(ips.size(), 2);
  EXPECT_EQ(ips[0], "10.0.0.4");
  EXPECT_EQ(ips[1], "10.0.0.5");
}

TEST(RaidenManagerBaseTest, IpCollectionFallbackToNumaLocalAny) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth0", "10.0.0.1", 0, NicClassification::kControlPlane},
      {"eth2", "10.0.0.3", 1, NicClassification::kControlPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{1}, mock_nics);

  auto ips = manager.local_ips();

  ASSERT_EQ(ips.size(), 1);
  EXPECT_EQ(ips[0], "10.0.0.3");
}

TEST(RaidenManagerBaseTest, IpCollectionFallbackToFirstNic) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth0", "10.0.0.1", 0, NicClassification::kControlPlane},
      {"eth1", "10.0.0.2", 0, NicClassification::kControlPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{1}, mock_nics);

  auto ips = manager.local_ips();

  ASSERT_EQ(ips.size(), 1);
  EXPECT_EQ(ips[0], "10.0.0.1");
}

TEST(RaidenManagerBaseTest, IpCollectionFallbackToLoopback) {
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{}, /*mock_nics=*/{});

  auto ips = manager.local_ips();

  ASSERT_EQ(ips.size(), 1);
  EXPECT_EQ(ips[0], "127.0.0.1");
}

TEST(RaidenManagerBaseTest, IpCollectionPrioritizeControlOverLoopback) {
  std::vector<HostNicAddress> mock_nics = {
      {"lo", "127.0.0.1", -1, NicClassification::kUnknown},
      {"eth0", "10.0.0.1", 0, NicClassification::kControlPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{1}, mock_nics);

  auto ips = manager.local_ips();

  ASSERT_EQ(ips.size(), 1);
  EXPECT_EQ(ips[0], "10.0.0.1");
}

TEST(RaidenManagerBaseTest, IpCollectionSpansEveryShardNumaNode) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth0", "10.0.0.1", 0, NicClassification::kControlPlane},
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
      {"eth2", "10.0.0.3", 1, NicClassification::kControlPlane},
      {"eth3", "10.0.0.4", 1, NicClassification::kDataPlane},
      {"eth4", "10.0.0.5", 1, NicClassification::kDataPlane},
  };
  // Shards 0-1 live on NUMA 1, shards 2-3 on NUMA 0: NUMA 1 NICs come first.
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/4,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{}, mock_nics,
                            ShardsOnNuma({1, 1, 0, 0}));

  EXPECT_THAT(manager.local_ips(),
              ::testing::ElementsAre("10.0.0.4", "10.0.0.5", "10.0.0.2"));
  // assigned_numa_node derives from the shard map when numa_nodes is empty.
  EXPECT_THAT(manager.assigned_numa_node(), ::testing::Optional(1));

  EXPECT_THAT(manager.shard_numa_node(0), ::testing::Optional(1));
  EXPECT_THAT(manager.shard_numa_node(3), ::testing::Optional(0));
  EXPECT_THAT(manager.numa_node_for_ip("10.0.0.2"), ::testing::Optional(0));
  EXPECT_THAT(manager.numa_node_for_ip("10.0.0.3"), ::testing::Optional(1));
  EXPECT_EQ(manager.numa_node_for_ip("192.168.0.1"), std::nullopt);

  EXPECT_THAT(manager.local_ips_for_shard(1),
              ::testing::ElementsAre("10.0.0.4", "10.0.0.5"));
  EXPECT_THAT(manager.local_ips_for_shard(2),
              ::testing::ElementsAre("10.0.0.2"));
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.2"),
              ::testing::ElementsAre(2, 3));
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.5"),
              ::testing::ElementsAre(0, 1));
  // Unknown IP: every shard, so the endpoint is never advertised as empty.
  EXPECT_THAT(manager.shards_for_local_ip("192.168.0.1"),
              ::testing::ElementsAre(0, 1, 2, 3));
}

TEST(RaidenManagerBaseTest, UniformShardNumaMapMatchesLegacySelection) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
      {"eth3", "10.0.0.4", 1, NicClassification::kDataPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/2,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{1}, mock_nics,
                            ShardsOnNuma({1, 1}));

  EXPECT_THAT(manager.local_ips(), ::testing::ElementsAre("10.0.0.4"));
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.4"),
              ::testing::ElementsAre(0, 1));
  EXPECT_THAT(manager.local_ips_for_shard(0),
              ::testing::ElementsAre("10.0.0.4"));
}

TEST(RaidenManagerBaseTest, UnknownShardNumaFallsBackToAllLocalIps) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
      {"eth3", "10.0.0.4", 1, NicClassification::kDataPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/2,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{}, mock_nics,
                            ShardsOnNuma({0, -1}));

  EXPECT_THAT(manager.local_ips(), ::testing::ElementsAre("10.0.0.2"));
  EXPECT_EQ(manager.shard_numa_node(1), std::nullopt);
  EXPECT_THAT(manager.local_ips_for_shard(1),
              ::testing::ElementsAre("10.0.0.2"));
  // NUMA 1 NIC is not a local IP, but its shard query still degrades to all.
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.4"),
              ::testing::ElementsAre(0, 1));
}

TEST(RaidenManagerBaseTest, SingleNicAdvertisesShardsOfEveryNumaNode) {
  // One data NIC on NUMA 0 while shards span NUMA 0 and 1: shards 0-1 have no
  // NUMA-local NIC, so the only endpoint must still advertise them.
  std::vector<HostNicAddress> mock_nics = {
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/4,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{}, mock_nics,
                            ShardsOnNuma({1, 1, 0, 0}));

  EXPECT_THAT(manager.local_ips(), ::testing::ElementsAre("10.0.0.2"));
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.2"),
              ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(manager.local_ips_for_shard(0),
              ::testing::ElementsAre("10.0.0.2"));
  EXPECT_THAT(manager.local_ips_for_shard(2),
              ::testing::ElementsAre("10.0.0.2"));
}

TEST(RaidenManagerBaseTest, OrphanShardsAreAdvertisedByEveryLocalIp) {
  // NICs on NUMA 0 and 1, shards on NUMA 0, 1 and 2: shard 3 (NUMA 2) has no
  // local NIC and is claimed by both endpoints; the others stay partitioned.
  std::vector<HostNicAddress> mock_nics = {
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
      {"eth3", "10.0.0.4", 1, NicClassification::kDataPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/4,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{}, mock_nics,
                            ShardsOnNuma({0, 0, 1, 2}));

  EXPECT_THAT(manager.local_ips(),
              ::testing::ElementsAre("10.0.0.2", "10.0.0.4"));
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.2"),
              ::testing::ElementsAre(0, 1, 3));
  EXPECT_THAT(manager.shards_for_local_ip("10.0.0.4"),
              ::testing::ElementsAre(2, 3));
  EXPECT_THAT(manager.local_ips_for_shard(3),
              ::testing::ElementsAre("10.0.0.2", "10.0.0.4"));
}

TEST(RaidenManagerBaseTest, ShardsDefaultToAssignedNumaAndUnknownGlobal) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth1", "10.0.0.2", 1, NicClassification::kDataPlane},
  };
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/3,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{1}, mock_nics);

  const absl::Span<const transport::ShardInfo> shards = manager.shards();
  ASSERT_EQ(shards.size(), 3);
  for (size_t sh = 0; sh < shards.size(); ++sh) {
    EXPECT_EQ(shards[sh].local_index, static_cast<int>(sh));
    EXPECT_EQ(shards[sh].global_index, -1);
    EXPECT_EQ(shards[sh].numa_node, 1);
  }
  EXPECT_EQ(manager.shard_numa_node(2), 1);
}

TEST(RaidenManagerBaseTest, ShardsWithoutAnyNumaInfoAreUnknown) {
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/2,
                            /*slice_byte_size=*/1024);

  ASSERT_EQ(manager.shards().size(), 2);
  EXPECT_EQ(manager.shards()[0].numa_node, -1);
  EXPECT_EQ(manager.shards()[1].numa_node, -1);
  EXPECT_EQ(manager.shard_numa_node(0), std::nullopt);
}

TEST(RaidenManagerBaseTest, ShardsCarryPerShardPlacement) {
  std::vector<HostNicAddress> mock_nics = {
      {"eth1", "10.0.0.2", 0, NicClassification::kDataPlane},
      {"eth3", "10.0.0.4", 1, NicClassification::kDataPlane},
  };
  std::vector<transport::ShardInfo> placement = ShardsOnNuma({0, 0, 1, -1});
  placement[0].global_index = 8;
  placement[2].global_index = 10;
  // local_index is normalised to the position regardless of the input.
  placement[3].local_index = 99;
  TestRaidenManager manager(/*num_layers=*/1, /*num_shards=*/4,
                            /*slice_byte_size=*/1024,
                            /*local_port=*/std::nullopt, /*parallelism=*/1,
                            /*numa_nodes=*/{}, mock_nics, placement);

  const absl::Span<const transport::ShardInfo> shards = manager.shards();
  ASSERT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].numa_node, 0);
  EXPECT_EQ(shards[1].numa_node, 0);
  EXPECT_EQ(shards[2].numa_node, 1);
  EXPECT_EQ(shards[3].numa_node, -1);
  EXPECT_EQ(shards[0].global_index, 8);
  EXPECT_EQ(shards[2].global_index, 10);
  EXPECT_EQ(shards[1].global_index, -1);
  EXPECT_EQ(shards[3].global_index, -1);
  EXPECT_EQ(shards[3].local_index, 3);
  // A mismatched per-shard placement is ignored, not partially applied.
  TestRaidenManager mismatched(/*num_layers=*/1, /*num_shards=*/4,
                               /*slice_byte_size=*/1024,
                               /*local_port=*/std::nullopt, /*parallelism=*/1,
                               /*numa_nodes=*/{0}, mock_nics,
                               ShardsOnNuma({0, 1}));
  ASSERT_EQ(mismatched.shards().size(), 4);
  EXPECT_EQ(mismatched.shards()[1].numa_node, 0);
  EXPECT_THAT(mismatched.local_ips(), ::testing::ElementsAre("10.0.0.2"));
}

}  // namespace
}  // namespace tpu_raiden

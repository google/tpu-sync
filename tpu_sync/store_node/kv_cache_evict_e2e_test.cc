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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/store_node/kv_cache_host_store_node.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"

namespace tpu_raiden {
namespace store_node {
namespace {

// One worker, one shard, two block arrays of 256 bytes: a block is 512 bytes.
constexpr size_t kNumBlockArrays = 2;
constexpr size_t kArrayBytes = 256;
constexpr size_t kBlockBytes = kNumBlockArrays * kArrayBytes;

// Small enough that a handful of inserts crosses the sweep watermarks.
constexpr size_t kSourceCapacity = 8;

KVTransferSpec TestSpec() {
  return KVTransferSpec{/*block_array_bytes=*/{kArrayBytes, kArrayBytes},
                        /*num_kv_shards=*/1, /*num_workers=*/1};
}

// The deterministic contents of block-array `array` of the block hashed
// `hash_idx`, byte by byte.
uint8_t PatternByte(int hash_idx, size_t array, size_t offset) {
  return static_cast<uint8_t>((hash_idx * 131 + array * 31 + offset) % 251);
}

// A serving host's assembly, mirrored from KVCacheHostStoreNode's Phase B
// with the sweep turned on: the manager owns the real DRAM pool and data
// endpoint, the store owns the coordination plane, and the worker
// registration completes the transfer path.
struct SourceAssembly {
  std::unique_ptr<KVCacheManagerWithTransfer> manager;
  std::unique_ptr<kv_cache::KVCacheStore> store;
  std::unique_ptr<controller::WorkerServiceServer> worker_server;
};

class EvictE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", /*overwrite=*/1);
    registry_ = kv_cache::global_registry::CreateTestGlobalRegistryServer();
  }
  void TearDown() override { unsetenv("RAIDEN_DISABLE_SINGLETON_WORKER"); }

  absl::StatusOr<SourceAssembly> MakeSource(
      const kv_cache::RaidenId& id, absl::string_view kv_pool_group,
      bool enable_evict_sweep = true, double evict_low_watermark = 0.5,
      double evict_high_watermark = 0.75) {
    SourceAssembly src;
    src.manager = std::make_unique<KVCacheManagerWithTransfer>(
        kNumBlockArrays, /*num_shards=*/1, kArrayBytes,
        /*local_port=*/0, /*host_blocks_to_allocate=*/kSourceCapacity,
        /*parallelism=*/1, /*node_id=*/0, /*local_control_port=*/-1);

    kv_cache::BackendConfig config;
    config.type = "HostOffloadBackend";
    config.capacity = kSourceCapacity;
    config.raiden_id = id;
    config.global_registry_address = registry_->server_address;
    config.kv_pool_group = std::string(kv_pool_group);
    config.monitor_config.enable = true;
    config.monitor_config.enable_evict_sweep = enable_evict_sweep;
    config.monitor_config.evict_sweep_period = absl::Milliseconds(200);
    config.monitor_config.evict_low_watermark = evict_low_watermark;
    config.monitor_config.evict_high_watermark = evict_high_watermark;
    ABSL_ASSIGN_OR_RETURN(
        src.store, kv_cache::KVCacheStore::Create(
                       config, kSourceCapacity, registry_->server_address, id,
                       /*num_shards=*/1,
                       /*shard_size_bytes=*/kArrayBytes,
                       /*store_server_ip=*/"localhost",
                       /*raiden_controller_port=*/0));

    src.worker_server = controller::WorkerServiceServer::Create();
    ABSL_RETURN_IF_ERROR(src.worker_server->StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(src.manager.get()),
        /*port=*/0));
    core::controller::RaidenControllerClient controller_client(
        src.store->raiden_controller_address());
    ABSL_RETURN_IF_ERROR(controller_client.RegisterWorker(
        "worker_0",
        absl::StrCat("localhost:", src.worker_server->GetRaidenWorkerPort()),
        src.manager->get_local_data_endpoints(), /*node_id=*/0));
    return src;
  }

  // Fills block `block_id` of every block array with `hash_idx`'s pattern.
  static void FillBlock(KVCacheManagerWithTransfer& manager, int block_id,
                        int hash_idx) {
    for (size_t array = 0; array < kNumBlockArrays; ++array) {
      uint8_t* base = manager.base()->GetHostPointer(array, /*shard_idx=*/0);
      ASSERT_NE(base, nullptr);
      for (size_t b = 0; b < kArrayBytes; ++b) {
        base[block_id * kArrayBytes + b] = PatternByte(hash_idx, array, b);
      }
    }
  }

  // Expects block `block_id` of every block array to hold `hash_idx`'s
  // pattern.
  static void ExpectBlockBytes(KVCacheManagerWithTransfer& manager,
                               int block_id, int hash_idx) {
    for (size_t array = 0; array < kNumBlockArrays; ++array) {
      const uint8_t* base =
          manager.base()->GetHostPointer(array, /*shard_idx=*/0);
      ASSERT_NE(base, nullptr);
      for (size_t b = 0; b < kArrayBytes; ++b) {
        ASSERT_EQ(base[block_id * kArrayBytes + b],
                  PatternByte(hash_idx, array, b))
            << "array " << array << " byte " << b << " of hash " << hash_idx;
      }
    }
  }

  std::unique_ptr<kv_cache::global_registry::TestGlobalRegistryServer>
      registry_;
};

TEST_F(EvictE2ETest, SweepDemotesToTheStoreNodeAndReadsBack) {
  kv_cache::RaidenId src_id{"e2e_serving", "0", "kv_pool", 0};
  auto src_or = MakeSource(src_id, "e2e_pool");
  ASSERT_TRUE(src_or.ok()) << src_or.status().ToString();
  SourceAssembly& src = *src_or;

  KVCacheHostStoreNode::Options options;
  options.raiden_id = kv_cache::RaidenId{"e2e_store_node", "0", "kv_pool", 0};
  options.store_server_ip = "localhost";
  options.global_registry_address = registry_->server_address;
  options.kv_pool_group = "e2e_pool";
  options.dram_budget_bytes = 16 * kBlockBytes;
  StaticKVTransferSpecSource source(TestSpec());
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &source);
  ASSERT_TRUE(node.ok()) << node.status().ToString();

  // 6 of 8 blocks in use: free ratio 0.25, below the 0.5 low watermark.
  // Every block carries its own byte pattern -- what the demotion must move
  // and the read-back must return.
  constexpr int kNumBlocks = 6;
  std::vector<std::string> hashes;
  std::vector<kv_cache::RaidenBlockId> slices;
  auto ids_or = src.store->raiden_controller()->AllocateBlockIds(kNumBlocks);
  ASSERT_TRUE(ids_or.ok()) << ids_or.status().ToString();
  for (int i = 0; i < kNumBlocks; ++i) {
    hashes.push_back(absl::StrCat("e2e_blk_", i));
    slices.push_back(kv_cache::RaidenBlockId(src_id, (*ids_or)[i],
                                             kv_cache::BlockStatus::HOST));
    FillBlock(*src.manager, (*ids_or)[i], i);
  }
  // Insert() pins; the sweep only demotes unpinned blocks, so release.
  ASSERT_TRUE(src.store->Insert(hashes, slices, /*on_host=*/true).ok());
  src.store->Release(hashes);

  // The sweep must raise free blocks from 2 to the 0.75 watermark's 6,
  // demoting 4 cold blocks to the node -- the only same-group higher-tier
  // store the registry knows.
  auto* controller = src.store->raiden_controller();
  for (int i = 0; i < 1000 && controller->num_free_blocks() != 6; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(controller->num_free_blocks(), 6)
      << "the sweep never demoted down to the high watermark";

  // Wipe the freed source blocks: a demotion is a copy, so without this the
  // read-back below could pass on bytes left over from before the demotion
  // whenever the landing allocation happens to reuse the block's old id.
  for (int blk = 0; blk < static_cast<int>(kSourceCapacity); ++blk) {
    bool resident = false;
    for (int i = 0; i < kNumBlocks; ++i) {
      auto l = src.store->backend()->Lookup(
          {hashes[i]}, kv_cache::LookupOptions{.enable_global = false});
      if (l.ok() && l->size() == 1 && l->front().second.host_block_id == blk) {
        resident = true;
      }
    }
    if (!resident) {
      for (size_t array = 0; array < kNumBlockArrays; ++array) {
        std::memset(
            src.manager->base()->GetHostPointer(array, 0) + blk * kArrayBytes,
            0xEE, kArrayBytes);
      }
    }
  }

  // Every demoted block must round-trip: gone locally, resolved through the
  // registry to the node, pulled back over the real data plane, byte-equal.
  // The pull is the controller's host-landing ReadRemote: the landing block
  // is chosen up front and the bytes must arrive exactly there, which is the
  // contract the KVManagerHolder fix restores. One block at a time, so the
  // reads never dip free blocks below the low watermark and restart the
  // sweep mid-verification.
  // Local-only lookups: a global lookup falls back to the registry, where
  // the demoted blocks now resolve to the node -- which is exactly what must
  // NOT count as locally resident here.
  const kv_cache::LookupOptions local_only{.enable_global = false};
  int demoted = 0;
  for (int i = 0; i < kNumBlocks; ++i) {
    auto local = src.store->backend()->Lookup({hashes[i]}, local_only);
    if (local.ok() && local->size() == 1) {
      continue;  // Survived the sweep locally; nothing to read back.
    }
    ++demoted;
    // The registry is what turns the local miss into the owner's
    // coordinates: a global lookup must name the node as the owner and the
    // block's id over there.
    auto resolved = src.store->Lookup({hashes[i]}, /*enable_global=*/true);
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    ASSERT_EQ(resolved->size(), 1u)
        << hashes[i] << " is neither local nor in the registry";
    const kv_cache::RaidenBlockId& remote = (*resolved)[0].second;
    ASSERT_EQ(remote.status, kv_cache::BlockStatus::REMOTE);
    ASSERT_EQ(remote.raiden_id, options.raiden_id)
        << hashes[i] << " is owned by someone other than the store node";

    // An explicitly chosen, wiped landing block: the pattern showing up in
    // it proves the transfer honored the requested destination instead of
    // landing wherever the manager's own allocation pointed.
    auto dst_or = src.store->raiden_controller()->AllocateBlockIds(1);
    ASSERT_TRUE(dst_or.ok()) << dst_or.status().ToString();
    const int dst_block = (*dst_or)[0];
    absl::Status read =
        src.store->raiden_controller()
            ->ReadRemote((*node)->raiden_controller_address(),
                         {remote.host_block_id}, {dst_block}, {hashes[i]})
            .Await();
    ASSERT_TRUE(read.ok()) << "reading " << hashes[i]
                           << " back from the store node: " << read;
    ExpectBlockBytes(*src.manager, dst_block, i);
    // Hand the landing block back so every iteration sees the same pool.
    ASSERT_TRUE(
        src.store->raiden_controller()->DeallocateBlockIds({dst_block}).ok());
  }
  EXPECT_EQ(demoted, 4);
}

// Tests that when a tier-1 host store node fills up, its own background evict
// sweep proactively reclaims headroom down to its high watermark so that a new
// demotion from tier 0 can land via FREE-only WriteRemote allocation, while
// the oldest unpinned LRU block is evicted locally and from the registry.
TEST_F(EvictE2ETest, FullHostStoreNodeEvictsLruBlockOnNewDemotion) {
  constexpr size_t kNodeCapacity = 4;
  const kv_cache::RaidenId src_id{"e2e_serving", "0", "kv_pool", 0};
  // Tier-0 source has its background sweep disabled; driven via
  // SweepOnceForTesting. Low watermark 0.7 and high 0.75 ensure:
  // - Phase 1: 6 blocks used (free ratio 2/8 = 0.25 <= 0.7), deficit 4 ->
  // demotes 4 blocks.
  // - Phase 2: 3 blocks used (free ratio 5/8 = 0.625 <= 0.7), deficit 1 ->
  // demotes 1 block.
  TF_ASSERT_OK_AND_ASSIGN(
      SourceAssembly src,
      MakeSource(src_id, "e2e_pool", /*enable_evict_sweep=*/false,
                 /*evict_low_watermark=*/0.7,
                 /*evict_high_watermark=*/0.75));

  KVCacheHostStoreNode::Options options;
  options.raiden_id = kv_cache::RaidenId{"e2e_store_node", "0", "kv_pool", 0};
  options.store_server_ip = "localhost";
  options.global_registry_address = registry_->server_address;
  options.kv_pool_group = "e2e_pool";
  options.evict_tier = 1;
  options.dram_budget_bytes = kNodeCapacity * kBlockBytes;  // 4 blocks capacity
  options.evict_sweep_period = absl::Milliseconds(100);
  // Watermarks chosen so when the 4-block node is full (free ratio 0.0 < 0.5),
  // evict sweep triggers and frees headroom up to high watermark 0.75 (deficit
  // 3). When node has 2+ free blocks (ratio >= 0.5), sweep stays idle.
  options.evict_low_watermark = 0.5;
  options.evict_high_watermark = 0.75;
  StaticKVTransferSpecSource source(TestSpec());
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &source);
  ASSERT_TRUE(node.ok()) << node.status().ToString();
  ASSERT_EQ((*node)->num_host_blocks(), kNodeCapacity);

  // -------------------------------------------------------------------------
  // Phase 1: Fill the tier-1 host store node via demotions from tier-0 source.
  // -------------------------------------------------------------------------
  // Insert blocks one-by-one into tier-0 so that e2e_blk_0 is the oldest (LRU
  // tail) and e2e_blk_5 is the newest (MRU head). Batch insert reverses LRU
  // order (host_offload_backend.cc:401).
  constexpr int kPhase1Blocks = 6;
  std::vector<std::string> hashes;
  std::vector<kv_cache::RaidenBlockId> slices;
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<int> ids,
      src.store->raiden_controller()->AllocateBlockIds(kPhase1Blocks));
  for (int i = 0; i < kPhase1Blocks; ++i) {
    const std::string hash = absl::StrCat("e2e_blk_", i);
    hashes.push_back(hash);
    slices.push_back(
        kv_cache::RaidenBlockId(src_id, ids[i], kv_cache::BlockStatus::HOST));
    FillBlock(*src.manager, ids[i], i);
    ASSERT_TRUE(
        src.store->Insert({hash}, {slices.back()}, /*on_host=*/true).ok());
    src.store->Release({hash});
  }

  // Synchronously drive the sweep on tier 0. It picks the 4 oldest blocks from
  // the LRU tail (e2e_blk_0 .. e2e_blk_3) and demotes them to the host store
  // node.
  EXPECT_TRUE(src.store->SweepOnceForTesting());

  // Phase 1 guarantees: tier 0 successfully demoted blk_0..3 to the node and
  // evicted them locally. We check tier 0 local eviction here. On the node,
  // because the background sweep is active from boot, the node transitions to
  // full and begins freeing headroom; therefore, we avoid a racy
  // "num_free_blocks == 0" assertion on the node and instead poll for the
  // sweep to free headroom below.
  const kv_cache::LookupOptions local_only{.enable_global = false};
  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto local_src,
                            src.store->backend()->Lookup(
                                {absl::StrCat("e2e_blk_", i)}, local_only));
    EXPECT_TRUE(local_src.empty())
        << "Demoted block e2e_blk_" << i << " should be evicted from tier 0";
  }

  // Bounded poll (≤5s, 10ms steps) for the node's background sweep to free
  // headroom. No ASSERT hard stop here: on unpatched base, sweep is disabled
  // so free blocks stay 0, and continuing allows the test to reach the 4
  // assertions below and fail cleanly as the reproducer.
  auto* node_controller = (*node)->store()->raiden_controller();
  for (int i = 0; i < 500 && node_controller->num_free_blocks() == 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }

  // -------------------------------------------------------------------------
  // Phase 2: Add 1 new block to tier-0, dropping its free ratio below 0.7.
  // Free blocks on source becomes 5 / 8 = 0.625 < 0.7.
  // Deficit to reach high watermark (6 free) is 6 - 5 = 1 block.
  // The oldest evictable block remaining on tier-0 is e2e_blk_4.
  // -------------------------------------------------------------------------
  TF_ASSERT_OK_AND_ASSIGN(const std::vector<int> new_id,
                          src.store->raiden_controller()->AllocateBlockIds(1));
  const std::string new_hash = "e2e_blk_6";
  FillBlock(*src.manager, new_id[0], 6);
  ASSERT_TRUE(src.store
                  ->Insert({new_hash},
                           {kv_cache::RaidenBlockId(
                               src_id, new_id[0], kv_cache::BlockStatus::HOST)},
                           /*on_host=*/true)
                  .ok());
  src.store->Release({new_hash});

  // Drive SweepOnce on tier 0 to demote e2e_blk_4 to the store node.
  // On base (without host store node evict sweep enabled): the node stays full,
  // destination WriteRemote allocates via AllocateBlockIds (FREE only) and
  // refuses with RESOURCE_EXHAUSTED; e2e_blk_4 is dropped locally on tier-0
  // and never lands on the store node.
  // In CL 1: the host store node's own background evict sweep has freed
  // headroom, so the new demotion lands, leaving the previously sweep-evicted
  // oldest block (e2e_blk_0) gone locally and unregistered from the global
  // registry.
  src.store->SweepOnceForTesting();

  // -------------------------------------------------------------------------
  // Invariant checks (fails on base, passes after fix):
  // -------------------------------------------------------------------------
  // 1. The new demotion (e2e_blk_4) must have successfully landed on the node.
  TF_ASSERT_OK_AND_ASSIGN(auto local_4, (*node)->store()->backend()->Lookup(
                                            {"e2e_blk_4"}, local_only));
  EXPECT_EQ(local_4.size(), 1u)
      << "New demotion e2e_blk_4 should land on the full host store node";

  // 2. Positive retention: the newest block from Phase 1 (e2e_blk_3) was not
  // reclaimed by the node's sweep (which reclaimed 3 blocks down to high
  // watermark) and remains on the node.
  TF_ASSERT_OK_AND_ASSIGN(auto local_3, (*node)->store()->backend()->Lookup(
                                            {"e2e_blk_3"}, local_only));
  EXPECT_EQ(local_3.size(), 1u)
      << "Newest Phase-1 block e2e_blk_3 must be retained on the store node";

  // 3. The oldest unpinned LRU block on the node (e2e_blk_0) must be evicted
  // locally.
  TF_ASSERT_OK_AND_ASSIGN(auto local_0, (*node)->store()->backend()->Lookup(
                                            {"e2e_blk_0"}, local_only));
  EXPECT_TRUE(local_0.empty())
      << "Oldest block e2e_blk_0 should be evicted from the store node";

  // 4. The oldest block (e2e_blk_0) must be unregistered from the global
  // registry.
  TF_ASSERT_OK_AND_ASSIGN(auto registry_lookup,
                          registry_->client->Lookup({"e2e_blk_0"}));
  EXPECT_TRUE(registry_lookup.empty())
      << "Oldest block e2e_blk_0 should no longer be present in global "
         "registry";

  // 5. Global lookup from peer store must no longer resolve e2e_blk_0.
  TF_ASSERT_OK_AND_ASSIGN(
      auto global_0, src.store->Lookup({"e2e_blk_0"}, /*enable_global=*/true));
  EXPECT_TRUE(global_0.empty())
      << "Oldest block e2e_blk_0 should no longer be discoverable by peers";
}

// Tests that demotions filling a host store node wake its evict sweep right
// away instead of on its period, so the next demotion lands.
TEST_F(EvictE2ETest, RemoteWriteWakesTheStoreNodeSweepBeforeItsPeriod) {
  constexpr size_t kNodeCapacity = 4;
  const kv_cache::RaidenId src_id{"e2e_serving", "0", "kv_pool", 0};
  TF_ASSERT_OK_AND_ASSIGN(
      SourceAssembly src,
      MakeSource(src_id, "e2e_pool", /*enable_evict_sweep=*/false,
                 /*evict_low_watermark=*/0.7,
                 /*evict_high_watermark=*/0.75));

  KVCacheHostStoreNode::Options options;
  options.raiden_id = kv_cache::RaidenId{"e2e_store_node", "0", "kv_pool", 0};
  options.store_server_ip = "localhost";
  options.global_registry_address = registry_->server_address;
  options.kv_pool_group = "e2e_pool";
  options.evict_tier = 1;
  options.dram_budget_bytes = kNodeCapacity * kBlockBytes;
  // The period outlasts the test, so only a wake can run the node's sweep.
  options.evict_sweep_period = absl::Hours(1);
  options.evict_low_watermark = 0.5;
  options.evict_high_watermark = 0.75;
  StaticKVTransferSpecSource source(TestSpec());
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &source);
  ASSERT_TRUE(node.ok()) << node.status().ToString();
  ASSERT_EQ((*node)->num_host_blocks(), kNodeCapacity);

  // Phase 1: Fill the tier-1 host store node via demotions from tier-0 source.
  constexpr int kPhase1Blocks = 6;
  std::vector<std::string> hashes;
  std::vector<kv_cache::RaidenBlockId> slices;
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<int> ids,
      src.store->raiden_controller()->AllocateBlockIds(kPhase1Blocks));
  for (int i = 0; i < kPhase1Blocks; ++i) {
    const std::string hash = absl::StrCat("e2e_blk_", i);
    hashes.push_back(hash);
    slices.push_back(
        kv_cache::RaidenBlockId(src_id, ids[i], kv_cache::BlockStatus::HOST));
    FillBlock(*src.manager, ids[i], i);
    ASSERT_TRUE(
        src.store->Insert({hash}, {slices.back()}, /*on_host=*/true).ok());
    src.store->Release({hash});
  }

  EXPECT_TRUE(src.store->SweepOnceForTesting());

  const kv_cache::LookupOptions local_only{.enable_global = false};
  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto local_src,
                            src.store->backend()->Lookup(
                                {absl::StrCat("e2e_blk_", i)}, local_only));
    EXPECT_TRUE(local_src.empty())
        << "Demoted block e2e_blk_" << i << " should be evicted from tier 0";
  }

  // Bounded poll (<=5s, 10ms steps) for the node's free blocks to become
  // non-zero.
  auto* node_controller = (*node)->store()->raiden_controller();
  for (int i = 0; i < 500 && node_controller->num_free_blocks() == 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }

  // Phase 2: Add 1 new block to tier-0, dropping its free ratio below 0.7.
  TF_ASSERT_OK_AND_ASSIGN(const std::vector<int> new_id,
                          src.store->raiden_controller()->AllocateBlockIds(1));
  const std::string new_hash = "e2e_blk_6";
  FillBlock(*src.manager, new_id[0], 6);
  ASSERT_TRUE(src.store
                  ->Insert({new_hash},
                           {kv_cache::RaidenBlockId(
                               src_id, new_id[0], kv_cache::BlockStatus::HOST)},
                           /*on_host=*/true)
                  .ok());
  src.store->Release({new_hash});

  src.store->SweepOnceForTesting();

  // Assertions: e2e_blk_4 landed on the node; e2e_blk_3 retained; e2e_blk_0
  // evicted locally.
  TF_ASSERT_OK_AND_ASSIGN(auto local_4, (*node)->store()->backend()->Lookup(
                                            {"e2e_blk_4"}, local_only));
  EXPECT_EQ(local_4.size(), 1u)
      << "New demotion e2e_blk_4 should land on the store node";

  TF_ASSERT_OK_AND_ASSIGN(auto local_3, (*node)->store()->backend()->Lookup(
                                            {"e2e_blk_3"}, local_only));
  EXPECT_EQ(local_3.size(), 1u)
      << "Phase-1 block e2e_blk_3 must be retained on the store node";

  TF_ASSERT_OK_AND_ASSIGN(auto local_0, (*node)->store()->backend()->Lookup(
                                            {"e2e_blk_0"}, local_only));
  EXPECT_TRUE(local_0.empty())
      << "Oldest block e2e_blk_0 should be evicted from the store node";
}

}  // namespace
}  // namespace store_node
}  // namespace tpu_raiden

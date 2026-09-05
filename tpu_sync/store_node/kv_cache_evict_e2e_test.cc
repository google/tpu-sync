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

#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
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

  absl::StatusOr<SourceAssembly> MakeSource(const kv_cache::RaidenId& id,
                                            absl::string_view kv_pool_group) {
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
    config.monitor_config.enable_evict_sweep = true;
    config.monitor_config.evict_sweep_period = absl::Milliseconds(200);
    config.monitor_config.evict_low_watermark = 0.5;
    config.monitor_config.evict_high_watermark = 0.75;
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
      uint8_t* base = manager.GetHostPointer(array, /*shard_idx=*/0);
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
      const uint8_t* base = manager.GetHostPointer(array, /*shard_idx=*/0);
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
  auto* block_manager = src.store->raiden_controller()->block_manager();
  for (int i = 0; i < 1000 && block_manager->num_free_blocks() != 6; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(block_manager->num_free_blocks(), 6)
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
        std::memset(src.manager->GetHostPointer(array, 0) + blk * kArrayBytes,
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

}  // namespace
}  // namespace store_node
}  // namespace tpu_raiden

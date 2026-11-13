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

#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_cache_store_backend.h"

#include <unistd.h>

#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/mock_storage_plugin_loader.h"
#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_backend.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace fs = std::filesystem;

constexpr int kTpSize = 2;

LookupOptions PrefixOptions() {
  return LookupOptions{.enable_global = false,
                       .match_mode = LookupMatchMode::kPrefix};
}

LookupOptions SubsetOptions() {
  return LookupOptions{.enable_global = false,
                       .match_mode = LookupMatchMode::kSubset};
}

// Runs the coordinator backend and TP workers on the mock plugin, which keeps
// one file per shard under `root_dir_`. A test evicts a shard by removing its
// file, the way a storage system evicts data without telling Raiden.
class OutofTreePluginKVCacheStoreBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    vtable_ = LoadMockPlugin();
    ASSERT_NE(vtable_, nullptr);

    const ::testing::TestInfo* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    root_dir_ = absl::StrCat(::testing::TempDir(), "/plugin_store_", getpid(),
                             "_", test_info->name());
    fs::remove_all(root_dir_);
    for (int rank = 0; rank < kTpSize; ++rank) {
      workers_.push_back(CreatePluginBackend(vtable_, rank));
    }
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(root_dir_, ec);
  }

  // Creates the plugin instance of TP rank `tp_rank`. The coordinator's
  // instance has rank 0 and maps the shard keys of all ranks.
  std::shared_ptr<PluginKVBackend> CreatePluginBackend(
      const RaidenStorageOutofTreePluginV1* vtable, int tp_rank) {
    absl::StatusOr<std::shared_ptr<PluginKVBackend>> backend =
        PluginKVBackend::Create(vtable, {{"root_dir", root_dir_}},
                                {.tp_size = kTpSize, .tp_rank = tp_rank});
    EXPECT_TRUE(backend.ok()) << backend.status();
    return backend.ok() ? *backend : nullptr;
  }

  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> CreateStore(
      OutofTreePluginKVCacheStoreBackend::Options options = {}) {
    return std::make_unique<OutofTreePluginKVCacheStoreBackend>(
        CreatePluginBackend(vtable_, /*tp_rank=*/0), "mock_plugin",
        std::move(options));
  }

  // Writes the shard of `block_hash` that TP rank `tp_rank` owns.
  void WriteShard(const std::string& block_hash, int tp_rank) {
    KVBackend& worker = *workers_[tp_rank];
    TF_ASSERT_OK_AND_ASSIGN(BlockKey key, worker.mapper()->MapKey(block_hash));
    std::string payload = absl::StrCat(block_hash, "_rank", tp_rank);
    absl::Status status;
    absl::Notification done;
    worker.WriteAsync(
        key,
        {HostBufferDescriptor{.ptr = payload.data(), .size = payload.size()}},
        payload.size(), [&](absl::Status write_status) {
          status = std::move(write_status);
          done.Notify();
        });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  void WriteBlock(const std::string& block_hash) {
    for (int rank = 0; rank < kTpSize; ++rank) WriteShard(block_hash, rank);
  }

  // Removes the file of one shard behind the backend's back.
  void EvictShard(const std::string& block_hash, int tp_rank) {
    TF_ASSERT_OK_AND_ASSIGN(BlockKey key,
                         workers_[tp_rank]->mapper()->MapKey(block_hash));
    ASSERT_TRUE(
        fs::remove(absl::StrCat(root_dir_, "/", key.resolved_key, ".bin")))
        << block_hash << " rank " << tp_rank;
  }

  absl::Status ReadShard(const std::string& block_hash, int tp_rank) {
    KVBackend& worker = *workers_[tp_rank];
    absl::StatusOr<BlockKey> key = worker.mapper()->MapKey(block_hash);
    if (!key.ok()) return key.status();
    std::string buffer(absl::StrCat(block_hash, "_rank", tp_rank).size(), '\0');
    absl::Status status;
    absl::Notification done;
    worker.ReadAsync(
        *key,
        {HostBufferDescriptor{.ptr = buffer.data(), .size = buffer.size()}},
        buffer.size(), [&](absl::Status read_status) {
          status = std::move(read_status);
          done.Notify();
        });
    done.WaitForNotification();
    return status;
  }

  // Block hashes that `store` reports as available.
  static std::vector<std::string> LookupHashes(
      OutofTreePluginKVCacheStoreBackend& store,
      std::vector<std::string> block_hashes,
      const LookupOptions& options = PrefixOptions()) {
    absl::StatusOr<BlockSliceList> found = store.Lookup(block_hashes, options);
    EXPECT_TRUE(found.ok()) << found.status();
    std::vector<std::string> hashes;
    if (!found.ok()) return hashes;
    for (const auto& [hash, block] : *found) hashes.push_back(hash);
    return hashes;
  }

  const RaidenStorageOutofTreePluginV1* vtable_ = nullptr;
  std::string root_dir_;
  std::vector<std::shared_ptr<PluginKVBackend>> workers_;
};

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       ReportsBlockOnceEveryRankWroteItsShard) {
  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> store = CreateStore();
  WriteShard("a", /*tp_rank=*/0);
  EXPECT_THAT(LookupHashes(*store, {"a"}), IsEmpty());

  WriteShard("a", /*tp_rank=*/1);
  TF_ASSERT_OK_AND_ASSIGN(BlockSliceList found,
                       store->Lookup({"a"}, PrefixOptions()));

  ASSERT_EQ(found.size(), 1);
  EXPECT_EQ(found[0].first, "a");
  EXPECT_EQ(found[0].second.status, BlockStatus::SHARED_STORAGE);
  EXPECT_EQ(found[0].second.raiden_id.data_name, "mock_plugin");
}

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       EvictedShardIsAMissOnTheNextLookup) {
  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> store = CreateStore();
  for (const char* hash : {"a", "b", "c"}) WriteBlock(hash);
  ASSERT_THAT(LookupHashes(*store, {"a", "b", "c"}),
              ElementsAre("a", "b", "c"));

  // Only rank 1 loses its shard of "b"; rank 0 still holds its shard.
  EvictShard("b", /*tp_rank=*/1);

  EXPECT_THAT(LookupHashes(*store, {"a", "b", "c"}), ElementsAre("a"));
  EXPECT_THAT(LookupHashes(*store, {"a", "b", "c"}, SubsetOptions()),
              ElementsAre("a", "c"));
  EXPECT_THAT(ReadShard("b", /*tp_rank=*/1),
              StatusIs(absl::StatusCode::kNotFound));
  ABSL_EXPECT_OK(ReadShard("b", /*tp_rank=*/0));
  // KVCacheStore reacts to a failed recall with Delete. It drops nothing.
  EXPECT_EQ(store->Delete({"b"}), 0);
}

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       ReportsBlockAgainAfterItIsRewritten) {
  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> store = CreateStore();
  WriteBlock("a");
  ASSERT_THAT(LookupHashes(*store, {"a"}), ElementsAre("a"));
  for (int rank = 0; rank < kTpSize; ++rank) EvictShard("a", rank);
  ASSERT_THAT(LookupHashes(*store, {"a"}), IsEmpty());

  WriteBlock("a");

  EXPECT_THAT(LookupHashes(*store, {"a"}), ElementsAre("a"));
  for (int rank = 0; rank < kTpSize; ++rank) {
    ABSL_EXPECT_OK(ReadShard("a", rank));
  }
}

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       DeleteDropsNothingAndClearsItsOutputs) {
  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> store = CreateStore();
  WriteBlock("a");
  ASSERT_THAT(LookupHashes(*store, {"a"}), ElementsAre("a"));

  std::vector<int> host_block_ids = {7};
  std::vector<std::string> deleted_hashes = {"stale"};
  EXPECT_EQ(store->Delete({"a"}, DeleteOptions::EvictHostResident(
                                     &host_block_ids, &deleted_hashes)),
            0);
  EXPECT_THAT(host_block_ids, IsEmpty());
  EXPECT_THAT(deleted_hashes, IsEmpty());

  deleted_hashes = {"stale"};
  EXPECT_EQ(store->Delete({"a"}, {.deleted_hashes = &deleted_hashes}), 0);
  EXPECT_THAT(deleted_hashes, IsEmpty());

  // The block is still on storage, so Lookup still reports it.
  EXPECT_THAT(LookupHashes(*store, {"a"}), ElementsAre("a"));
}

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       StorageTierHoldsNoPinsOrEntries) {
  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> store =
      CreateStore({.capacity_bytes = 4096});

  EXPECT_EQ(store->name(), "mock_plugin");
  EXPECT_TRUE(store->Insert({"a"}, {RaidenBlockId()}));
  EXPECT_FALSE(
      store->Insert({"a"}, {RaidenBlockId()}, InsertOption::kAdmitStrict));
  EXPECT_TRUE(store->Pin({"a"}));
  store->Release({"a"});
  EXPECT_EQ(store->GetPinCount("a"), 0);
  EXPECT_EQ(store->GetSize(), 0);
  EXPECT_EQ(store->GetCapacity(), 4096);
  EXPECT_EQ(store->GetAvailableSpace(), 4096);
}

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       ZeroBatchSizeAndEmptyDpRankUseDefaults) {
  std::unique_ptr<OutofTreePluginKVCacheStoreBackend> store =
      CreateStore({.lookup_batch_size = 0, .dp_rank = ""});

  EXPECT_EQ(store->options().lookup_batch_size, kDefaultPluginLookupBatchSize);
  EXPECT_EQ(store->options().dp_rank, "0");
}

// Block-level mode: a plugin without map_key answers for whole blocks.

const RaidenStorageOutofTreePluginV1* g_mock_vtable = nullptr;
size_t g_probed_keys = 0;

void CountingBatchExists(void* ctx, const RaidenBlockKey* keys, size_t num_keys,
                         RaidenBatchExistsCallback callback, void* user_data) {
  g_probed_keys += num_keys;
  g_mock_vtable->batch_exists_async(ctx, keys, num_keys, callback, user_data);
}

TEST_F(OutofTreePluginKVCacheStoreBackendTest,
       ProbesOneKeyPerBlockWithoutMapKey) {
  g_mock_vtable = vtable_;
  g_probed_keys = 0;
  RaidenStorageOutofTreePluginV1 block_level_vtable = *vtable_;
  block_level_vtable.map_key = nullptr;
  block_level_vtable.batch_exists_async = &CountingBatchExists;
  std::shared_ptr<PluginKVBackend> worker =
      CreatePluginBackend(&block_level_vtable, /*tp_rank=*/0);
  ASSERT_NE(worker, nullptr);
  OutofTreePluginKVCacheStoreBackend store(
      CreatePluginBackend(&block_level_vtable, /*tp_rank=*/0), "block_level",
      OutofTreePluginKVCacheStoreBackend::Options());

  // The raw block hash is the storage key, so one write stores the block.
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, worker->mapper()->MapKey("blk_a"));
  EXPECT_EQ(key.resolved_key, "blk_a");
  std::string payload = "block_level_payload";
  absl::Status status;
  absl::Notification done;
  worker->WriteAsync(
      key,
      {HostBufferDescriptor{.ptr = payload.data(), .size = payload.size()}},
      payload.size(), [&](absl::Status write_status) {
        status = std::move(write_status);
        done.Notify();
      });
  done.WaitForNotification();
  ABSL_ASSERT_OK(status);

  EXPECT_THAT(LookupHashes(store, {"blk_a", "blk_missing"}, SubsetOptions()),
              ElementsAre("blk_a"));
  // 2 blocks at TP=2 cost 2 probes, not 4.
  EXPECT_EQ(g_probed_keys, 2);

  // Nothing is cached, so the same lookup probes both keys again.
  EXPECT_THAT(LookupHashes(store, {"blk_a", "blk_missing"}, SubsetOptions()),
              ElementsAre("blk_a"));
  EXPECT_EQ(g_probed_keys, 4);

  // An empty hash cannot be a key. It is skipped like any unmappable hash.
  EXPECT_THAT(LookupHashes(store, {"", "blk_a"}, SubsetOptions()),
              ElementsAre("blk_a"));
  EXPECT_THAT(LookupHashes(store, {"", "blk_a"}), IsEmpty());
}

}  // namespace
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

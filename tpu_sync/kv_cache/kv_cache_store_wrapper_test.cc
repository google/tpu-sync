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

#include "tpu_sync/kv_cache/kv_cache_store_wrapper.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::IsEmpty;

// Wrapper construction reads the monitor env contract, so every fixture
// here starts and ends with it cleared -- ambient configuration on the
// machine running the tests must not leak in.
void ClearMonitorEnv() {
  unsetenv("RAIDEN_ENABLE_STORE_MONITOR");
  unsetenv("RAIDEN_ENABLE_EVICT_SWEEP");
  unsetenv("RAIDEN_STORE_MONITOR_HEARTBEAT_S");
  unsetenv("RAIDEN_EVICT_SWEEP_PERIOD_S");
  unsetenv("RAIDEN_EVICT_LOW_WATERMARK");
  unsetenv("RAIDEN_EVICT_HIGH_WATERMARK");
}

// Drives the wrapper through the same environment contract the serving stack
// uses: RAIDEN_SHM_KEY turns the metadata table on, RAIDEN_SHM_MODEL_UID
// names its owner, and RAIDEN_DISABLE_SINGLETON_WORKER isolates each
// wrapper's embedded controller server so tests do not share state.
class KVCacheStoreWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_key_ = absl::StrCat("raiden_store_wrapper_test_", getpid());
    UnlinkSegments();
    setenv("RAIDEN_SHM_KEY", shm_key_.c_str(), /*overwrite=*/1);
    setenv("RAIDEN_SHM_MODEL_UID", "model_a", /*overwrite=*/1);
    setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", /*overwrite=*/1);
    unsetenv("RAIDEN_SHM_SERVER_NAME");
    ClearMonitorEnv();
  }

  void TearDown() override {
    UnlinkSegments();
    unsetenv("RAIDEN_SHM_KEY");
    unsetenv("RAIDEN_SHM_MODEL_UID");
    unsetenv("RAIDEN_DISABLE_SINGLETON_WORKER");
    unsetenv("RAIDEN_SHM_SERVER_NAME");
    ClearMonitorEnv();
  }

  static RaidenId TestRaidenId() {
    return RaidenId{"wrapper_test_job", "0", "wrapper_test_cache", 0};
  }

  // TestRaidenId()'s contribution to the metadata segment name.
  static constexpr absl::string_view kIdentitySuffix =
      "_wrapper_test_job_0_wrapper_test_cache_0";

  std::unique_ptr<KVCacheStoreWrapper> MakeWrapper(
      size_t capacity, int num_shards,
      const std::string& global_registry_address = "") {
    return std::make_unique<KVCacheStoreWrapper>(
        capacity, global_registry_address, TestRaidenId(), num_shards,
        /*shard_size_bytes=*/512,
        /*store_server_ip=*/"127.0.0.1");
  }

  bool MetadataSegmentExists(const std::string& suffix) {
    std::string name = absl::StrCat("/", shm_key_, suffix);
    int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
      return false;
    }
    close(fd);
    return true;
  }

  void UnlinkSegments() {
    // The serving stack never unlinks a segment or its ".lock" companion;
    // decommissioning is the operator's job -- here, the test's.
    for (const std::string& name :
         {absl::StrCat("/", shm_key_, "_metadata", kIdentitySuffix),
          absl::StrCat("/", shm_key_, "_metadata_test_server",
                       kIdentitySuffix)}) {
      shm_unlink(name.c_str());
      shm_unlink(absl::StrCat(name, ".lock").c_str());
    }
  }

  // Inserts `hashes` as host-resident blocks 0..n-1, mirroring them into the
  // metadata table.
  void InsertHostBlocks(KVCacheStoreWrapper& wrapper,
                        const std::vector<std::string>& hashes) {
    RaidenId rid = TestRaidenId();
    std::vector<RaidenBlockId> slices;
    slices.reserve(hashes.size());
    for (int i = 0; i < static_cast<int>(hashes.size()); ++i) {
      slices.push_back(RaidenBlockId(rid, i, BlockStatus::HOST));
    }
    ASSERT_TRUE(wrapper->Insert(hashes, slices, /*on_host=*/true).ok());
    // Insert pins what it takes; these cases only want the blocks resident,
    // and a held pin would make them unevictable.
    wrapper->Release(hashes);
  }

  std::string shm_key_;
};

TEST_F(KVCacheStoreWrapperTest, NoShmKeySkipsMetadataTable) {
  unsetenv("RAIDEN_SHM_KEY");
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_FALSE(
      MetadataSegmentExists(absl::StrCat("_metadata", kIdentitySuffix)));
  EXPECT_EQ((*wrapper)->capacity(), 4);
}

// num_shards == 0 (the old pure-LRU configuration) is no longer supported.
// The wrapper is routed through KVCacheStore::Create(), so a misconfigured
// caller gets a catchable exception, not a process abort.
TEST_F(KVCacheStoreWrapperTest, NumShardsZeroThrows) {
  EXPECT_THROW(
      { MakeWrapper(/*capacity=*/4, /*num_shards=*/0); },
      std::invalid_argument);
}

TEST_F(KVCacheStoreWrapperTest, ColdStartCreatesMetadataTable) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_TRUE(
      MetadataSegmentExists(absl::StrCat("_metadata", kIdentitySuffix)));
  auto lookup_or = (*wrapper)->Lookup({"host_1"}, LookupOptions{});
  ASSERT_TRUE(lookup_or.ok());
  EXPECT_THAT(*lookup_or, IsEmpty());
}

TEST_F(KVCacheStoreWrapperTest, ServerNameSuffixesMetadataSegment) {
  setenv("RAIDEN_SHM_SERVER_NAME", "test_server", /*overwrite=*/1);
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_TRUE(MetadataSegmentExists(
      absl::StrCat("_metadata_test_server", kIdentitySuffix)));
  EXPECT_FALSE(
      MetadataSegmentExists(absl::StrCat("_metadata", kIdentitySuffix)));
}

TEST_F(KVCacheStoreWrapperTest, MetadataShmKeyAppendsStoreIdentity) {
  EXPECT_EQ(MetadataShmKey(RaidenId{"job_a", "1", "kv_cache", 2}),
            absl::StrCat(shm_key_, "_metadata_job_a_1_kv_cache_2"));
  setenv("RAIDEN_SHM_SERVER_NAME", "replica0", /*overwrite=*/1);
  EXPECT_EQ(MetadataShmKey(RaidenId{"job_a", "1", "kv_cache", 2}),
            absl::StrCat(shm_key_, "_metadata_replica0_job_a_1_kv_cache_2"));
}

// Colocated stores share the host's env block, so the RaidenId must be what
// tells their tables apart -- and it may hold characters (path separators,
// borg cell prefixes) that are illegal inside a shm name.
TEST_F(KVCacheStoreWrapperTest, MetadataShmKeySanitizesTheIdentity) {
  EXPECT_EQ(MetadataShmKey(RaidenId{"cell/job:a", "0", "kv cache", 0}),
            absl::StrCat(shm_key_, "_metadata_cell_job_a_0_kv_cache_0"));
  EXPECT_NE(MetadataShmKey(RaidenId{"job_a", "0", "kv_cache", 0}),
            MetadataShmKey(RaidenId{"job_a", "1", "kv_cache", 0}));
}

TEST_F(KVCacheStoreWrapperTest, MetadataShmKeyEmptyIdentityAddsNoSuffix) {
  EXPECT_EQ(MetadataShmKey(RaidenId{}),
            absl::StrCat(shm_key_, "_metadata"));
}

// A malformed shm env value is a configuration error: it must throw (a
// Python ValueError), not degrade to silently serving without recovery.
TEST_F(KVCacheStoreWrapperTest, MalformedShmKeyThrows) {
  setenv("RAIDEN_SHM_KEY", "tmp/raiden", /*overwrite=*/1);
  EXPECT_THROW(
      { MakeWrapper(/*capacity=*/4, /*num_shards=*/1); },
      std::invalid_argument);
}

TEST_F(KVCacheStoreWrapperTest, MalformedShmServerNameThrows) {
  setenv("RAIDEN_SHM_SERVER_NAME", "rep lica", /*overwrite=*/1);
  EXPECT_THROW(
      { MakeWrapper(/*capacity=*/4, /*num_shards=*/1); },
      std::invalid_argument);
}

TEST_F(KVCacheStoreWrapperTest, RecoversHostBlocksAfterRestart) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  InsertHostBlocks(*wrapper, {"host_1", "host_2"});

  // Destroying the wrapper simulates the engine dying: the mapping goes
  // away, the segment survives. The next incarnation must rebuild the LRU
  // cache from the surviving table during construction.
  wrapper.reset();
  wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);

  auto lookup_or = (*wrapper)->Lookup({"host_1", "host_2"}, LookupOptions{});
  ASSERT_TRUE(lookup_or.ok());
  ASSERT_EQ(lookup_or->size(), 2);
  EXPECT_EQ((*lookup_or)[0].first, "host_1");
  EXPECT_EQ((*lookup_or)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_or)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_or)[1].first, "host_2");
  EXPECT_EQ((*lookup_or)[1].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_or)[1].second.host_block_id, 1);
}

// Exercises the env contract StoreMonitorConfigFromEnv documents; the
// wrapper feeds its result straight into BackendConfig.monitor_config.
class StoreMonitorConfigFromEnvTest : public ::testing::Test {
 protected:
  void SetUp() override { ClearMonitorEnv(); }
  void TearDown() override { ClearMonitorEnv(); }
};

TEST_F(StoreMonitorConfigFromEnvTest, UnsetEnvironmentIsAllDefaults) {
  StoreMonitorConfig config = StoreMonitorConfigFromEnv();
  EXPECT_FALSE(config.enable);
  EXPECT_FALSE(config.enable_evict_sweep);
  EXPECT_EQ(config.heartbeat_period, absl::ZeroDuration());
  EXPECT_EQ(config.evict_sweep_period, absl::ZeroDuration());
  EXPECT_EQ(config.evict_low_watermark, 0.0);
  EXPECT_EQ(config.evict_high_watermark, 0.0);
}

TEST_F(StoreMonitorConfigFromEnvTest, ReadsTheFullConfiguration) {
  setenv("RAIDEN_ENABLE_STORE_MONITOR", "true", /*overwrite=*/1);
  setenv("RAIDEN_ENABLE_EVICT_SWEEP", "1", /*overwrite=*/1);
  setenv("RAIDEN_STORE_MONITOR_HEARTBEAT_S", "7", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_SWEEP_PERIOD_S", "45", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_LOW_WATERMARK", "0.1", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_HIGH_WATERMARK", "0.25", /*overwrite=*/1);

  StoreMonitorConfig config = StoreMonitorConfigFromEnv();
  EXPECT_TRUE(config.enable);
  EXPECT_TRUE(config.enable_evict_sweep);
  EXPECT_EQ(config.heartbeat_period, absl::Seconds(7));
  EXPECT_EQ(config.evict_sweep_period, absl::Seconds(45));
  EXPECT_EQ(config.evict_low_watermark, 0.1);
  EXPECT_EQ(config.evict_high_watermark, 0.25);
}

TEST_F(StoreMonitorConfigFromEnvTest, InvalidValuesFallBackToTheDefaults) {
  // Enables are "true"/"1" only -- "0" and other strings do not enable.
  setenv("RAIDEN_ENABLE_STORE_MONITOR", "0", /*overwrite=*/1);
  setenv("RAIDEN_ENABLE_EVICT_SWEEP", "yes", /*overwrite=*/1);
  setenv("RAIDEN_STORE_MONITOR_HEARTBEAT_S", "soon", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_SWEEP_PERIOD_S", "-3", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_LOW_WATERMARK", "1.5", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_HIGH_WATERMARK", "-0.1", /*overwrite=*/1);

  StoreMonitorConfig config = StoreMonitorConfigFromEnv();
  EXPECT_FALSE(config.enable);
  EXPECT_FALSE(config.enable_evict_sweep);
  EXPECT_EQ(config.heartbeat_period, absl::ZeroDuration());
  EXPECT_EQ(config.evict_sweep_period, absl::ZeroDuration());
  EXPECT_EQ(config.evict_low_watermark, 0.0);
  EXPECT_EQ(config.evict_high_watermark, 0.0);

  // NaN parses but is not in (0, 1); it must not leak past the range check
  // (every NaN comparison downstream would be false).
  setenv("RAIDEN_EVICT_LOW_WATERMARK", "nan", /*overwrite=*/1);
  EXPECT_EQ(StoreMonitorConfigFromEnv().evict_low_watermark, 0.0);
}

TEST_F(KVCacheStoreWrapperTest, MonitorEnvWithoutRegistryIsDropped) {
  // The env block is shared fleet-wide, so the switches must not break a
  // registry-less (local-only) store: KVCacheStore::Create rejects the
  // monitor without a registry, so the wrapper drops the request instead.
  setenv("RAIDEN_ENABLE_STORE_MONITOR", "true", /*overwrite=*/1);
  setenv("RAIDEN_ENABLE_EVICT_SWEEP", "true", /*overwrite=*/1);
  EXPECT_NE(MakeWrapper(/*capacity=*/4, /*num_shards=*/1), nullptr);
}

TEST_F(KVCacheStoreWrapperTest, MonitorEnvReachesTheStore) {
  // Watermarks the store rejects (low > high) prove the parsed environment
  // lands in BackendConfig.monitor_config rather than on the floor.
  auto registry = global_registry::CreateTestGlobalRegistryServer();
  setenv("RAIDEN_ENABLE_STORE_MONITOR", "true", /*overwrite=*/1);
  setenv("RAIDEN_ENABLE_EVICT_SWEEP", "true", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_LOW_WATERMARK", "0.9", /*overwrite=*/1);
  setenv("RAIDEN_EVICT_HIGH_WATERMARK", "0.5", /*overwrite=*/1);
  try {
    MakeWrapper(/*capacity=*/4, /*num_shards=*/1, registry->server_address);
    FAIL() << "expected the watermark rejection to reach construction";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("watermarks"));
  }
}

TEST_F(KVCacheStoreWrapperTest, ModelUidMismatchColdStarts) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  InsertHostBlocks(*wrapper, {"host_1"});
  wrapper.reset();

  // A restart under another model must not resurrect the surviving table.
  setenv("RAIDEN_SHM_MODEL_UID", "model_b", /*overwrite=*/1);
  wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);

  auto lookup_or = (*wrapper)->Lookup({"host_1"}, LookupOptions{});
  ASSERT_TRUE(lookup_or.ok());
  EXPECT_THAT(*lookup_or, IsEmpty());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

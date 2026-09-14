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

#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {
namespace {

namespace fs = std::filesystem;

class StorageDriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scratch_dir_ = absl::StrCat(
        testing::TempDir(), "/storage_driver_test_", getpid(), "_",
        std::chrono::system_clock::now().time_since_epoch().count());
    fs::create_directories(scratch_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(scratch_dir_, ec);
  }

  void CreateFile(const std::string& path) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream file(path, std::ios::binary);
    file << "data";
  }

  std::vector<absl::StatusOr<bool>> RunBatchExists(
      PosixKVBackend& backend, absl::Span<const BlockKey> keys) {
    absl::Notification done;
    std::vector<absl::StatusOr<bool>> results;
    backend.BatchExistsAsync(keys, [&](std::vector<absl::StatusOr<bool>> r) {
      results = std::move(r);
      done.Notify();
    });
    done.WaitForNotification();
    return results;
  }

  struct StoreTestEnv {
    std::shared_ptr<PosixKVBackend> backend;
    std::shared_ptr<PosixPathMapper> mapper;
    std::unique_ptr<PosixKVCacheStoreBackend> store;

    void CreateBlock(const std::string& hash) {
      std::string path = mapper->MapKey(hash, {.rank = 0}).resolved_key;
      fs::create_directories(fs::path(path).parent_path());
      std::ofstream file(path, std::ios::binary);
      file << "data";
    }
  };

  StoreTestEnv CreateStoreBackend(size_t batch_size = 16) {
    auto backend = std::make_shared<PosixKVBackend>("posix_disk");
    auto mapper =
        std::make_shared<PosixPathMapper>(scratch_dir_, "model_test", 1, 0);
    backend->set_mapper(mapper);
    auto store = std::make_unique<PosixKVCacheStoreBackend>(
        backend, "posix_disk", 0, batch_size);
    return StoreTestEnv{std::move(backend), std::move(mapper),
                        std::move(store)};
  }

  void VerifyScatterGatherFidelity(PosixKVBackend& backend, const BlockKey& key,
                                   size_t num_slices, size_t slice_bytes) {
    const size_t total_bytes = num_slices * slice_bytes;
    std::vector<std::vector<uint8_t>> src_buffers(
        num_slices, std::vector<uint8_t>(slice_bytes));
    std::vector<BackendBufferDescriptor> write_slices;
    write_slices.reserve(num_slices);

    for (size_t i = 0; i < num_slices; ++i) {
      for (size_t b = 0; b < slice_bytes; ++b) {
        src_buffers[i][b] = static_cast<uint8_t>((i * 31 + b) % 256);
      }
      write_slices.push_back(BackendBufferDescriptor{
          .ptr = src_buffers[i].data(), .size = slice_bytes});
    }

    absl::Notification write_done;
    absl::Status write_status;
    backend.WriteAsync(key, write_slices, total_bytes, [&](absl::Status s) {
      write_status = s;
      write_done.Notify();
    });
    write_done.WaitForNotification();
    ASSERT_TRUE(write_status.ok()) << write_status;

    std::error_code ec;
    uintmax_t file_size = fs::file_size(key.resolved_key, ec);
    ASSERT_FALSE(ec) << ec.message();
    EXPECT_EQ(file_size, total_bytes);

    std::vector<std::vector<uint8_t>> dst_buffers(
        num_slices, std::vector<uint8_t>(slice_bytes, 0));
    std::vector<BackendBufferDescriptor> read_slices;
    read_slices.reserve(num_slices);
    for (size_t i = 0; i < num_slices; ++i) {
      read_slices.push_back(BackendBufferDescriptor{
          .ptr = dst_buffers[i].data(), .size = slice_bytes});
    }

    absl::Notification read_done;
    absl::Status read_status;
    backend.ReadAsync(key, read_slices, total_bytes, [&](absl::Status s) {
      read_status = s;
      read_done.Notify();
    });
    read_done.WaitForNotification();
    ASSERT_TRUE(read_status.ok()) << read_status;

    for (size_t i = 0; i < num_slices; ++i) {
      EXPECT_EQ(std::memcmp(src_buffers[i].data(), dst_buffers[i].data(),
                            slice_bytes),
                0)
          << "Slice " << i << " mismatch";
    }
  }

  std::string scratch_dir_;
};

using PosixBackendTest = StorageDriverTest;

// ===========================================================================
// Theme 1: Path Mapping & Key Resolution
// ===========================================================================

TEST_F(StorageDriverTest, PosixPathMapperResolution) {
  PosixPathMapper mapper(scratch_dir_, "llama3", /*tp_size=*/4, /*rank=*/2);
  EXPECT_EQ(mapper.tp_size(), 4);

  // Default rank (-1 uses configured rank 2)
  BlockKey key_default = mapper.MapKey("hash_12345");
  EXPECT_EQ(key_default.block_hash, "hash_12345");
  EXPECT_EQ(key_default.resolved_key,
            absl::StrCat(scratch_dir_, "/llama3_tp4_r2/has/h_/hash_12345.bin"));

  // Explicit rank override via KeyMappingOptions
  BlockKey key_rank0 = mapper.MapKey("hash_12345", {.rank = 0});
  EXPECT_EQ(key_rank0.block_hash, "hash_12345");
  EXPECT_EQ(key_rank0.resolved_key,
            absl::StrCat(scratch_dir_, "/llama3_tp4_r0/has/h_/hash_12345.bin"));

  // Explicit rank and tp_size override via KeyMappingOptions
  BlockKey key_override =
      mapper.MapKey("hash_12345", {.rank = 3, .tp_size = 8});
  EXPECT_EQ(key_override.resolved_key,
            absl::StrCat(scratch_dir_, "/llama3_tp8_r3/has/h_/hash_12345.bin"));
}

TEST_F(PosixBackendTest, PosixPathMapperShortAndEmptyHashHandling) {
  PosixPathMapper mapper(scratch_dir_, "llama3", /*tp_size=*/1, /*rank=*/0);
  std::string base = absl::StrCat(scratch_dir_, "/llama3_tp1_r0");

  EXPECT_TRUE(mapper.MapKey("", {.rank = 0}).resolved_key.empty());
  EXPECT_EQ(mapper.MapKey("a", {.rank = 0}).resolved_key,
            absl::StrCat(base, "/a/00/a.bin"));
  EXPECT_EQ(mapper.MapKey("abc", {.rank = 0}).resolved_key,
            absl::StrCat(base, "/abc/00/abc.bin"));
  EXPECT_EQ(mapper.MapKey("abcd", {.rank = 0}).resolved_key,
            absl::StrCat(base, "/abc/d/abcd.bin"));
}

// ===========================================================================
// Theme 2: Factory Construction & Auto-Discovery
// ===========================================================================

TEST_F(StorageDriverTest, FactoryAutoDiscoversTpSize) {
  ::tpu_sync::rpc::RaidenIdProto unit;
  unit.set_job_name("test_job");
  unit.set_job_replica_id("0");
  unit.set_data_name("test_data");
  unit.set_data_replica_idx(0);

  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "PosixKVCacheStoreBackend";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");

  // 1. Auto-discovery from RaidenController shards (num_shards = 4)
  auto c1_or = ::tpu_raiden::controller::RaidenController::Create(
      unit, /*num_blocks=*/16, /*num_shards=*/4, /*shard_size_bytes=*/512, "");
  ASSERT_TRUE(c1_or.ok());
  auto b1_or = ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                   .CreateBackend(config, c1_or->get());
  ASSERT_TRUE(b1_or.ok());
  auto b1 = std::dynamic_pointer_cast<PosixKVCacheStoreBackend>(*b1_or);
  ASSERT_NE(b1, nullptr);
  EXPECT_EQ(b1->storage_backend()->mapper()->tp_size(), 4);

  // 2. Auto-discovery from registered workers (2 registered workers)
  auto c2_or = ::tpu_raiden::controller::RaidenController::Create(
      unit, /*num_blocks=*/16, /*num_shards=*/1, /*shard_size_bytes=*/512, "");
  ASSERT_TRUE(c2_or.ok());
  ::tpu_raiden::core::controller::WorkerRegistration w0;
  w0.worker_id = "worker_0";
  w0.raiden_worker_endpoint = "localhost:10001";
  ::tpu_raiden::core::controller::WorkerRegistration w1;
  w1.worker_id = "worker_1";
  w1.raiden_worker_endpoint = "localhost:10002";
  (*c2_or)->worker_registry()->SetOnRegisterCallback(nullptr);
  ASSERT_TRUE((*c2_or)->worker_registry()->RegisterWorker(w0).ok());
  ASSERT_TRUE((*c2_or)->worker_registry()->RegisterWorker(w1).ok());
  auto b2_or = ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                   .CreateBackend(config, c2_or->get());
  ASSERT_TRUE(b2_or.ok());
  auto b2 = std::dynamic_pointer_cast<PosixKVCacheStoreBackend>(*b2_or);
  ASSERT_NE(b2, nullptr);
  EXPECT_EQ(b2->storage_backend()->mapper()->tp_size(), 2);
}

// ===========================================================================
// Theme 3: Speculative Batch Existence Probing & Lookup
// ===========================================================================

TEST_F(StorageDriverTest, BatchExistsMixedHitsAndMisses) {
  PosixKVBackend backend("posix_disk");
  EXPECT_TRUE(RunBatchExists(backend, {}).empty());

  PosixPathMapper mapper(scratch_dir_, "model_test", /*tp_size=*/1, /*rank=*/0);

  const int kNumFiles = 32;
  std::vector<BlockKey> keys;
  keys.reserve(kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    BlockKey key = mapper.MapKey(absl::StrCat("block_mixed_", i), {.rank = 0});
    if (i % 2 == 0) {
      CreateFile(key.resolved_key);
    }
    keys.push_back(std::move(key));
  }

  auto results = RunBatchExists(backend, keys);
  ASSERT_EQ(results.size(), kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    ASSERT_TRUE(results[i].ok());
    EXPECT_EQ(*results[i], (i % 2 == 0));
  }

  // Single item hit and miss check
  auto single_hit = RunBatchExists(backend, {keys[0]});
  ASSERT_EQ(single_hit.size(), 1);
  ASSERT_TRUE(single_hit[0].ok());
  EXPECT_TRUE(*single_hit[0]);

  auto single_miss = RunBatchExists(backend, {keys[1]});
  ASSERT_EQ(single_miss.size(), 1);
  ASSERT_TRUE(single_miss[0].ok());
  EXPECT_FALSE(*single_miss[0]);
}

TEST_F(StorageDriverTest, BatchExistsAsyncAndWorkerThreads) {
  PosixKVBackend backend("posix_disk", {{"lookup_threads", "1"}});
  PosixPathMapper mapper(scratch_dir_, "model_test", /*tp_size=*/1, /*rank=*/0);

  const int kNumFiles = 16;
  std::vector<BlockKey> keys;
  keys.reserve(kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    BlockKey key =
        mapper.MapKey(absl::StrCat("block_async_worker_", i), {.rank = 0});
    if (i % 2 == 0) {
      CreateFile(key.resolved_key);
    }
    keys.push_back(std::move(key));
  }

  absl::Notification done;
  std::vector<absl::StatusOr<bool>> async_results;
  backend.BatchExistsAsync(keys,
                           [&](std::vector<absl::StatusOr<bool>> results) {
                             async_results = std::move(results);
                             done.Notify();
                           });
  done.WaitForNotification();

  ASSERT_EQ(async_results.size(), kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    ASSERT_TRUE(async_results[i].ok());
    EXPECT_EQ(*async_results[i], (i % 2 == 0));
  }
}

TEST_F(StorageDriverTest, LookupLeadingMissTerminatesEarly) {
  auto env = CreateStoreBackend(/*batch_size=*/16);

  // Block 0 missing, blocks 1 and 2 exist
  env.CreateBlock("block_1");
  env.CreateBlock("block_2");

  std::vector<std::string> hashes = {"block_0", "block_1", "block_2"};
  auto lookup_or = env.store->Lookup(hashes);
  ASSERT_TRUE(lookup_or.ok());
  EXPECT_TRUE(lookup_or->empty());
}

TEST_F(StorageDriverTest, LookupWindowedPrefixHitsAndChunkBoundaries) {
  // 1. Windowed prefix hits across chunks (48 blocks with batch_size=16)
  auto env = CreateStoreBackend(/*batch_size=*/16);
  const int kTotal = 48;
  std::vector<std::string> hashes;
  hashes.reserve(kTotal);
  for (int i = 0; i < kTotal; ++i) {
    std::string hash = absl::StrCat("window_block_", i);
    env.CreateBlock(hash);
    hashes.push_back(std::move(hash));
  }

  auto lookup_or = env.store->Lookup(hashes);
  ASSERT_TRUE(lookup_or.ok());
  ASSERT_EQ(lookup_or->size(), kTotal);
  for (int i = 0; i < kTotal; ++i) {
    EXPECT_EQ((*lookup_or)[i].first, absl::StrCat("window_block_", i));
    EXPECT_EQ((*lookup_or)[i].second.status, BlockStatus::SHARED_STORAGE);
    EXPECT_EQ((*lookup_or)[i].second.raiden_id.job_replica_id, "shared");
    EXPECT_EQ((*lookup_or)[i].second.raiden_id.data_name, "posix_disk");
  }

  // 2. Mid-stream miss stops at chunk boundary (create 24, query 40)
  for (int i = 0; i < 24; ++i) {
    env.CreateBlock(absl::StrCat("mid_block_", i));
  }
  std::vector<std::string> mid_hashes;
  mid_hashes.reserve(40);
  for (int i = 0; i < 40; ++i) {
    mid_hashes.push_back(absl::StrCat("mid_block_", i));
  }
  auto mid_lookup_or = env.store->Lookup(mid_hashes);
  ASSERT_TRUE(mid_lookup_or.ok());
  ASSERT_EQ(mid_lookup_or->size(), 24);

  // 3. Configurable chunk size (batch_size=4)
  auto small_chunk_env = CreateStoreBackend(/*batch_size=*/4);
  EXPECT_EQ(small_chunk_env.store->lookup_batch_size(), 4);
  std::vector<std::string> chunk_hashes;
  chunk_hashes.reserve(10);
  for (int i = 0; i < 10; ++i) {
    std::string hash = absl::StrCat("chunk_block_", i);
    chunk_hashes.push_back(hash);
    if (i < 6) {
      small_chunk_env.CreateBlock(hash);
    }
  }
  auto small_lookup_or = small_chunk_env.store->Lookup(chunk_hashes);
  ASSERT_TRUE(small_lookup_or.ok());
  EXPECT_EQ(small_lookup_or->size(), 6);
}

// ===========================================================================
// Theme 4: Atomic Commit & Filesystem Fault Recovery
// ===========================================================================

TEST_F(PosixBackendTest, PosixKVBackendAtomicRenameCommit) {
  PosixKVBackend backend("posix_disk");
  PosixPathMapper mapper(scratch_dir_, "model_atomic", /*tp_size=*/1,
                         /*rank=*/0);
  BlockKey key = mapper.MapKey("block_atomic_commit", {.rank = 0});

  const size_t kSize = 4096;
  std::vector<uint8_t> write_data(kSize, 0xCD);
  BackendBufferDescriptor src_desc;
  src_desc.ptr = write_data.data();
  src_desc.size = kSize;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, {src_desc}, kSize, [&](absl::Status status) {
    write_status = status;
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ASSERT_TRUE(write_status.ok()) << write_status;

  // Verify resolved file exists
  EXPECT_TRUE(fs::exists(key.resolved_key));

  // Verify file content matches
  std::vector<uint8_t> read_data(kSize, 0);
  BackendBufferDescriptor dst_desc;
  dst_desc.ptr = read_data.data();
  dst_desc.size = kSize;

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, {dst_desc}, kSize, [&](absl::Status status) {
    read_status = status;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ASSERT_TRUE(read_status.ok()) << read_status;
  EXPECT_EQ(std::memcmp(write_data.data(), read_data.data(), kSize), 0);

  // Verify no leftover .tmp_* files in parent directory
  fs::path parent_dir = fs::path(key.resolved_key).parent_path();
  ASSERT_TRUE(fs::exists(parent_dir));
  for (const auto& entry : fs::directory_iterator(parent_dir)) {
    std::string filename = entry.path().filename().string();
    EXPECT_FALSE(absl::StrContains(filename, ".tmp_"))
        << "Found leftover temp file: " << entry.path();
  }
}

TEST_F(PosixBackendTest, PosixKVBackendAtomicRenameCleansUpOnFailure) {
  PosixKVBackend backend("posix_disk");
  std::string read_only_dir = absl::StrCat(scratch_dir_, "/read_only_dir");
  fs::create_directories(read_only_dir);
  fs::permissions(read_only_dir, fs::perms::owner_read | fs::perms::owner_exec,
                  fs::perm_options::replace);

  BlockKey key{"fail_block", absl::StrCat(read_only_dir, "/fail.bin"), 0, 0};
  const size_t kSize = 1024;
  std::vector<uint8_t> write_data(kSize, 0xFF);
  BackendBufferDescriptor src_desc;
  src_desc.ptr = write_data.data();
  src_desc.size = kSize;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, {src_desc}, kSize, [&](absl::Status status) {
    write_status = status;
    write_done.Notify();
  });
  write_done.WaitForNotification();

  EXPECT_FALSE(write_status.ok());

  // Restore permissions before inspecting and cleaning up
  fs::permissions(read_only_dir, fs::perms::owner_all,
                  fs::perm_options::replace);

  for (const auto& entry : fs::directory_iterator(read_only_dir)) {
    std::string filename = entry.path().filename().string();
    EXPECT_FALSE(absl::StrContains(filename, ".tmp_"))
        << "Found orphan temp file: " << entry.path();
  }
}

TEST_F(PosixBackendTest, PosixKVBackendPreadPrematureEOFReturnsDataLossError) {
  PosixKVBackend backend("posix_disk");
  PosixPathMapper mapper(scratch_dir_, "model_trunc", /*tp_size=*/1,
                         /*rank=*/0);
  BlockKey key = mapper.MapKey("block_truncated", {.rank = 0});

  // Write a truncated file of 1024 bytes directly to disk
  fs::create_directories(fs::path(key.resolved_key).parent_path());
  {
    std::ofstream file(key.resolved_key, std::ios::binary);
    std::vector<char> buf(1024, 'x');
    file.write(buf.data(), buf.size());
  }

  // Request 4096 bytes via ReadAsync
  const size_t kRequestedSize = 4096;
  std::vector<uint8_t> read_buf(kRequestedSize, 0);
  BackendBufferDescriptor dst_desc;
  dst_desc.ptr = read_buf.data();
  dst_desc.size = kRequestedSize;

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, {dst_desc}, kRequestedSize, [&](absl::Status status) {
    read_status = status;
    read_done.Notify();
  });
  read_done.WaitForNotification();

  EXPECT_TRUE(absl::IsDataLoss(read_status)) << read_status;
}

TEST_F(StorageDriverTest, StorageErrorHandlingAndRecovery) {
  PosixKVBackend backend("posix_disk", {{"root_dir", scratch_dir_}});

  // Non-existent file read returns NotFoundError
  BlockKey missing_key =
      backend.mapper()->MapKey("missing_key_xyz", {.rank = 0});
  std::vector<uint8_t> read_buf(4096, 0);
  BackendBufferDescriptor valid_desc{.ptr = read_buf.data(), .size = 4096};

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(missing_key, {valid_desc}, 4096, [&](absl::Status s) {
    read_status = s;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  EXPECT_TRUE(absl::IsNotFound(read_status)) << read_status;

  // Null pointer slice on write returns InvalidArgumentError
  BlockKey write_key =
      backend.mapper()->MapKey("null_slice_write", {.rank = 0});
  BackendBufferDescriptor null_desc{.ptr = nullptr, .size = 4096};

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(write_key, {null_desc}, 4096, [&](absl::Status s) {
    write_status = s;
    write_done.Notify();
  });
  write_done.WaitForNotification();
  EXPECT_TRUE(absl::IsInvalidArgument(write_status)) << write_status;

  // Null pointer slice on read returns InvalidArgumentError
  std::vector<uint8_t> valid_data(4096, 0x5A);
  BackendBufferDescriptor valid_write_desc{.ptr = valid_data.data(),
                                           .size = 4096};
  absl::Notification prep_done;
  absl::Status prep_status;
  backend.WriteAsync(write_key, {valid_write_desc}, 4096, [&](absl::Status s) {
    prep_status = s;
    prep_done.Notify();
  });
  prep_done.WaitForNotification();
  ASSERT_TRUE(prep_status.ok()) << prep_status;

  absl::Notification read_null_done;
  absl::Status read_null_status;
  backend.ReadAsync(write_key, {null_desc}, 4096, [&](absl::Status s) {
    read_null_status = s;
    read_null_done.Notify();
  });
  read_null_done.WaitForNotification();
  EXPECT_TRUE(absl::IsInvalidArgument(read_null_status)) << read_null_status;
}

// ===========================================================================
// Theme 5: Multi-Slice Scatter/Gather I/O & Partition Isolation
// ===========================================================================

TEST_F(StorageDriverTest, MultiSliceScatterGatherFidelityAndStress) {
  PosixKVBackend backend("posix_disk", {{"root_dir", scratch_dir_}});
  // 1. Multi-layer 8 slices (4 KB per slice = 32 KB total)
  VerifyScatterGatherFidelity(
      backend, backend.mapper()->MapKey("slice_8", {.rank = 0}), 8, 4096);
  // 2. Distributed serving parity 32 slices (4 KB per slice = 128 KB total)
  VerifyScatterGatherFidelity(
      backend, backend.mapper()->MapKey("slice_32", {.rank = 0}), 32, 4096);
  // 3. Stress 2048 fine-grained slices (256 B per slice = 512 KB total)
  VerifyScatterGatherFidelity(
      backend, backend.mapper()->MapKey("slice_2048", {.rank = 0}), 2048, 256);
}

TEST_F(StorageDriverTest, MultiTpPartitionedIsolation) {
  const int tp_size = 2;
  PosixKVBackend backend("posix_disk");
  PosixPathMapper mapper(scratch_dir_, "model_tp_test", tp_size, /*rank=*/0);

  const size_t kBlockSize = 4096;
  std::vector<std::vector<uint8_t>> written(tp_size,
                                            std::vector<uint8_t>(kBlockSize));
  for (int r = 0; r < tp_size; ++r) {
    std::fill(written[r].begin(), written[r].end(),
              static_cast<uint8_t>(0x30 * (r + 1)));
    BlockKey key = mapper.MapKey("tp_block", {.rank = r});
    EXPECT_EQ(key.resolved_key,
              absl::StrCat(scratch_dir_, "/model_tp_test_tp2_r", r,
                           "/tp_/bl/tp_block.bin"));
    BackendBufferDescriptor desc{.ptr = written[r].data(), .size = kBlockSize};

    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(key, {desc}, kBlockSize, [&](absl::Status s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    ASSERT_TRUE(status.ok()) << status;
  }

  for (int r = 0; r < tp_size; ++r) {
    std::vector<uint8_t> read_buf(kBlockSize, 0);
    BackendBufferDescriptor desc{.ptr = read_buf.data(), .size = kBlockSize};
    BlockKey key = mapper.MapKey("tp_block", {.rank = r});

    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(key, {desc}, kBlockSize, [&](absl::Status s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(std::memcmp(written[r].data(), read_buf.data(), kBlockSize), 0);
  }
  EXPECT_NE(std::memcmp(written[0].data(), written[1].data(), kBlockSize), 0);
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

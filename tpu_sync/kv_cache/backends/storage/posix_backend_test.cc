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
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"
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

using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;

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
      TF_ASSERT_OK_AND_ASSIGN(
          BlockKey key, mapper->MapKey(hash, {.parallelism = {.tp_rank = 0}}));
      const std::string& path = key.resolved_key;
      fs::create_directories(fs::path(path).parent_path());
      std::ofstream file(path, std::ios::binary);
      file << "data";
    }
  };

  StoreTestEnv CreateStoreBackend(size_t batch_size = 16) {
    auto backend = std::make_shared<PosixKVBackend>(
        "posix_disk",
        absl::flat_hash_map<std::string, std::string>{{"tp_rank", "0"}});
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
    std::vector<HostBufferDescriptor> write_slices;
    write_slices.reserve(num_slices);

    for (size_t i = 0; i < num_slices; ++i) {
      for (size_t b = 0; b < slice_bytes; ++b) {
        src_buffers[i][b] = static_cast<uint8_t>((i * 31 + b) % 256);
      }
      write_slices.push_back(HostBufferDescriptor{.ptr = src_buffers[i].data(),
                                                  .size = slice_bytes});
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
    std::vector<HostBufferDescriptor> read_slices;
    read_slices.reserve(num_slices);
    for (size_t i = 0; i < num_slices; ++i) {
      read_slices.push_back(HostBufferDescriptor{.ptr = dst_buffers[i].data(),
                                                 .size = slice_bytes});
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
  PosixPathMapper mapper(scratch_dir_, "llama3", /*tp_size=*/4, /*tp_rank=*/2);
  EXPECT_EQ(mapper.tp_size(), 4);

  // "hash_12345" hex-encodes to 686173685f3132333435.
  // Default rank (-1 uses configured rank 2)
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key_default, mapper.MapKey("hash_12345"));
  EXPECT_EQ(key_default.block_hash, "hash_12345");
  EXPECT_EQ(key_default.resolved_key,
            absl::StrCat(scratch_dir_,
                         "/llama3/tp4_r2/686/17/686173685f3132333435.bin"));

  // Explicit rank override via KeyMappingOptions
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_rank0,
      mapper.MapKey("hash_12345", {.parallelism = {.tp_rank = 0}}));
  EXPECT_EQ(key_rank0.block_hash, "hash_12345");
  EXPECT_EQ(key_rank0.resolved_key,
            absl::StrCat(scratch_dir_,
                         "/llama3/tp4_r0/686/17/686173685f3132333435.bin"));

  // Explicit rank and tp_size override via KeyMappingOptions
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_override,
      mapper.MapKey("hash_12345",
                    {.parallelism = {.tp_size = 8, .tp_rank = 3}}));
  EXPECT_EQ(key_override.resolved_key,
            absl::StrCat(scratch_dir_,
                         "/llama3/tp8_r3/686/17/686173685f3132333435.bin"));
}

// The prefix directories are right-padded, so every input length yields the
// same directory depth and the same component widths.
TEST_F(PosixBackendTest, PosixPathMapperFixedWidthAndEmptyHashHandling) {
  PosixPathMapper mapper(scratch_dir_, "llama3", /*tp_size=*/1, /*tp_rank=*/0);
  const std::string base = absl::StrCat(scratch_dir_, "/llama3/tp1_r0");

  EXPECT_THAT(mapper.MapKey("", {.parallelism = {.tp_rank = 0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must not be empty")));

  TF_ASSERT_OK_AND_ASSIGN(BlockKey one_byte,
                          mapper.MapKey("a", {.parallelism = {.tp_rank = 0}}));
  EXPECT_EQ(one_byte.resolved_key, absl::StrCat(base, "/610/00/61.bin"));

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey three_bytes,
      mapper.MapKey("abc", {.parallelism = {.tp_rank = 0}}));
  EXPECT_EQ(three_bytes.resolved_key, absl::StrCat(base, "/616/26/616263.bin"));

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey four_bytes,
      mapper.MapKey("abcd", {.parallelism = {.tp_rank = 0}}));
  EXPECT_EQ(four_bytes.resolved_key,
            absl::StrCat(base, "/616/26/61626364.bin"));
}

// Real block hashes are raw digests, not printable text.
TEST_F(PosixBackendTest, PosixPathMapperHexEncodesArbitraryBytes) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);

  const std::string binary_hash("\xff\xff\xff\xff\xff\xff\xff\xff", 8);
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey(binary_hash));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama_70b/tp8_r2/fff/ff/ffffffffffffffff.bin");
  // Encoding is a storage concern: block identity stays the raw bytes.
  EXPECT_EQ(key.block_hash, binary_hash);
}

// 0x2F is a path separator: unencoded, it escapes the intended directory.
TEST_F(PosixBackendTest, PosixPathMapperHashSeparatorBytesDoNotInjectDirs) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);

  const std::string injecting_hash("\x2f\x2e\x2e\x2f", 4);  // "/../"
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey(injecting_hash));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama_70b/tp8_r2/2f2/e2/2f2e2e2f.bin");

  const std::vector<absl::string_view> components =
      absl::StrSplit(key.resolved_key, '/', absl::SkipEmpty());
  // tmp, kv_cache, llama_70b, tp8_r2, 2f2, e2, 2f2e2e2f.bin
  EXPECT_EQ(components.size(), 7);
  EXPECT_THAT(components, Not(Contains("..")));
  EXPECT_THAT(key.resolved_key, Not(HasSubstr("//")));
}

// 0x00 truncates at open(2), so unencoded prefixes alias onto one file.
TEST_F(PosixBackendTest, PosixPathMapperNulBytesDoNotAlias) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);

  TF_ASSERT_OK_AND_ASSIGN(BlockKey shorter,
                          mapper.MapKey(std::string("\x01\x00", 2)));
  TF_ASSERT_OK_AND_ASSIGN(BlockKey longer,
                          mapper.MapKey(std::string("\x01\x00\x02", 3)));

  // Same leaf directory, but the NUL is preserved in distinct filenames.
  EXPECT_EQ(shorter.resolved_key,
            "/tmp/kv_cache/llama_70b/tp8_r2/010/00/0100.bin");
  EXPECT_EQ(longer.resolved_key,
            "/tmp/kv_cache/llama_70b/tp8_r2/010/00/010002.bin");
  EXPECT_NE(shorter.resolved_key, longer.resolved_key);
  EXPECT_EQ(shorter.block_hash.size(), 2);
  EXPECT_EQ(longer.block_hash.size(), 3);
}

// Hex encoding doubles the filename length, so the cap is half of NAME_MAX.
TEST_F(PosixBackendTest, PosixPathMapperRejectsOverlongHash) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);

  ABSL_EXPECT_OK(mapper.MapKey(std::string(kMaxBlockHashBytes, 'a')));
  EXPECT_THAT(
      mapper.MapKey(std::string(kMaxBlockHashBytes + 1, 'a')),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("NAME_MAX")));
}

// HuggingFace-style IDs must not inject an extra directory level.
TEST_F(PosixBackendTest, PosixPathMapperSanitizesModelName) {
  PosixPathMapper mapper("/tmp/kv_cache", "meta-llama/Llama-3.1-70B",
                         /*tp_size=*/8, /*tp_rank=*/2);

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/meta-llama_Llama-3.1-70B/tp8_r2/610/00/61.bin");

  const std::vector<absl::string_view> components =
      absl::StrSplit(key.resolved_key, '/', absl::SkipEmpty());
  // tmp, kv_cache, meta-llama_Llama-3.1-70B, tp8_r2, 610, 00, 61.bin
  EXPECT_EQ(components.size(), 7);
}

// End-to-end: a raw binary hash survives a real write/read round trip.
TEST_F(PosixBackendTest, BinaryHashWriteReadRoundTrip) {
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_binary", /*tp_size=*/1,
                         /*tp_rank=*/0);

  const std::string binary_hash("\x00\x2f\xff\x41", 4);
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey(binary_hash, {.parallelism = {.tp_rank = 0}}));
  EXPECT_EQ(
      key.resolved_key,
      absl::StrCat(scratch_dir_, "/model_binary/tp1_r0/002/ff/002fff41.bin"));

  const size_t kSize = 4096;
  std::vector<uint8_t> write_data(kSize, 0xAB);
  HostBufferDescriptor src_desc{.ptr = write_data.data(), .size = kSize};

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, {src_desc}, kSize, [&](absl::Status s) {
    write_status = s;
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ASSERT_TRUE(write_status.ok()) << write_status;
  ASSERT_TRUE(fs::exists(key.resolved_key));

  std::vector<uint8_t> read_data(kSize, 0);
  HostBufferDescriptor dst_desc{.ptr = read_data.data(), .size = kSize};

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, {dst_desc}, kSize, [&](absl::Status s) {
    read_status = s;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ASSERT_TRUE(read_status.ok()) << read_status;
  EXPECT_EQ(std::memcmp(write_data.data(), read_data.data(), kSize), 0);
}

// End-to-end: NUL-containing hashes must not read back each other's data.
TEST_F(PosixBackendTest, NulContainingHashesDoNotAliasOnDisk) {
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_nul", /*tp_size=*/1,
                         /*tp_rank=*/0);

  TF_ASSERT_OK_AND_ASSIGN(BlockKey shorter,
                          mapper.MapKey(std::string("\x01\x00", 2),
                                        {.parallelism = {.tp_rank = 0}}));
  TF_ASSERT_OK_AND_ASSIGN(BlockKey longer,
                          mapper.MapKey(std::string("\x01\x00\x02", 3),
                                        {.parallelism = {.tp_rank = 0}}));
  ASSERT_NE(shorter.resolved_key, longer.resolved_key);

  const size_t kSize = 1024;
  std::vector<uint8_t> shorter_data(kSize, 0x11);
  std::vector<uint8_t> longer_data(kSize, 0x22);

  const std::vector<std::pair<BlockKey, std::vector<uint8_t>*>> writes = {
      {shorter, &shorter_data}, {longer, &longer_data}};
  for (const auto& [key, data] : writes) {
    HostBufferDescriptor desc{.ptr = data->data(), .size = kSize};
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(key, {desc}, kSize, [&](absl::Status s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    ASSERT_TRUE(status.ok()) << status;
  }

  std::vector<uint8_t> read_buf(kSize, 0);
  HostBufferDescriptor read_desc{.ptr = read_buf.data(), .size = kSize};
  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(shorter, {read_desc}, kSize, [&](absl::Status s) {
    read_status = s;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ASSERT_TRUE(read_status.ok()) << read_status;
  EXPECT_EQ(std::memcmp(shorter_data.data(), read_buf.data(), kSize), 0);
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
  config.type = "posix";
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
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  EXPECT_TRUE(RunBatchExists(backend, {}).empty());

  PosixPathMapper mapper(scratch_dir_, "model_test", /*tp_size=*/1,
                         /*tp_rank=*/0);

  const int kNumFiles = 32;
  std::vector<BlockKey> keys;
  keys.reserve(kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(BlockKey key,
                            mapper.MapKey(absl::StrCat("block_mixed_", i),
                                          {.parallelism = {.tp_rank = 0}}));
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
  PosixKVBackend backend(
      "posix_disk", {{"storage_io_thread_pool_size", "1"}, {"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_test", /*tp_size=*/1,
                         /*tp_rank=*/0);

  const int kNumFiles = 16;
  std::vector<BlockKey> keys;
  keys.reserve(kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        BlockKey key, mapper.MapKey(absl::StrCat("block_async_worker_", i),
                                    {.parallelism = {.tp_rank = 0}}));
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
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_atomic", /*tp_size=*/1,
                         /*tp_rank=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey("block_atomic_commit", {.parallelism = {.tp_rank = 0}}));

  const size_t kSize = 4096;
  std::vector<uint8_t> write_data(kSize, 0xCD);
  HostBufferDescriptor src_desc;
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
  HostBufferDescriptor dst_desc;
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
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  std::string read_only_dir = absl::StrCat(scratch_dir_, "/read_only_dir");
  fs::create_directories(read_only_dir);
  fs::permissions(read_only_dir, fs::perms::owner_read | fs::perms::owner_exec,
                  fs::perm_options::replace);

  BlockKey key{"fail_block", absl::StrCat(read_only_dir, "/fail.bin"), 0, 0};
  const size_t kSize = 1024;
  std::vector<uint8_t> write_data(kSize, 0xFF);
  HostBufferDescriptor src_desc;
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

// A write that dies part-way through must publish nothing: the final path has
// to stay absent so the next Lookup reports a miss and the block is simply
// re-fetched. Before the temp-file+rename rewrite the partial bytes landed
// directly on the final path, which Lookup (a bare existence check) then
// reported as a hit forever, failing every subsequent recall with DataLoss.
TEST_F(PosixBackendTest, PosixKVBackendWriteFailureLeavesNoVisibleFile) {
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_partial", /*tp_size=*/1,
                         /*tp_rank=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey("block_partial_write", {.parallelism = {.tp_rank = 0}}));

  // First slice is valid so bytes actually reach the temp file; the second is
  // a null pointer, which aborts the write after it has begun.
  const size_t kSliceSize = 2048;
  std::vector<uint8_t> good_data(kSliceSize, 0xAB);
  std::vector<HostBufferDescriptor> slices = {
      HostBufferDescriptor{.ptr = good_data.data(), .size = kSliceSize},
      HostBufferDescriptor{.ptr = nullptr, .size = kSliceSize},
  };

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, slices, 2 * kSliceSize, [&](absl::Status status) {
    write_status = status;
    write_done.Notify();
  });
  write_done.WaitForNotification();
  EXPECT_THAT(write_status, StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Null slice pointer")));

  EXPECT_FALSE(fs::exists(key.resolved_key))
      << "Partial write became visible at " << key.resolved_key;

  // The backend's own existence probe must agree: this block is a miss.
  std::vector<absl::StatusOr<bool>> exists = RunBatchExists(backend, {key});
  ASSERT_EQ(exists.size(), 1);
  EXPECT_THAT(exists[0], absl_testing::IsOkAndHolds(false));

  // The aborted temp file must be unlinked; nothing else ever collects it
  // because this backend's Delete is a no-op stub.
  fs::path parent_dir = fs::path(key.resolved_key).parent_path();
  if (fs::exists(parent_dir)) {
    for (const auto& entry : fs::directory_iterator(parent_dir)) {
      EXPECT_FALSE(absl::StrContains(entry.path().filename().string(), ".tmp_"))
          << "Found orphan temp file: " << entry.path();
    }
  }
}

// Publishing by whole-file rename is incompatible with a partial-file write,
// so a non-zero offset must be refused rather than silently truncating the
// other writers' slices away.
TEST_F(PosixBackendTest, PosixKVBackendWriteRejectsNonZeroOffset) {
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_offset", /*tp_size=*/1,
                         /*tp_rank=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey("block_offset_write", {.parallelism = {.tp_rank = 0}}));
  key.offset = 4096;

  const size_t kSize = 1024;
  std::vector<uint8_t> write_data(kSize, 0x5A);
  HostBufferDescriptor src_desc;
  src_desc.ptr = write_data.data();
  src_desc.size = kSize;

  absl::Notification write_done;
  absl::Status write_status;
  std::thread::id callback_thread;
  const std::thread::id caller_thread = std::this_thread::get_id();
  backend.WriteAsync(key, {src_desc}, kSize, [&](absl::Status status) {
    write_status = status;
    callback_thread = std::this_thread::get_id();
    write_done.Notify();
  });
  write_done.WaitForNotification();

  EXPECT_THAT(write_status, StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("offset")));
  EXPECT_FALSE(fs::exists(key.resolved_key));

  // A rejected precondition must report on the pool like every other outcome
  // of WriteAsync. Reporting inline would re-enter the caller before
  // WriteAsync returned, deadlocking anyone holding a lock across the call --
  // and it would do so on this error path only, which is the worst way to
  // break a contract. Asserting on the thread rather than on
  // `!write_done.HasBeenNotified()` immediately after the call keeps this
  // deterministic: the pool thread is free to finish before the main thread
  // gets that far, so the notification check could pass by luck.
  EXPECT_NE(callback_thread, caller_thread);
}

TEST_F(PosixBackendTest, PosixKVBackendPreadPrematureEOFReturnsDataLossError) {
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_trunc", /*tp_size=*/1,
                         /*tp_rank=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey("block_truncated", {.parallelism = {.tp_rank = 0}}));

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
  HostBufferDescriptor dst_desc;
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
  PosixKVBackend backend("posix_disk",
                         {{"root_dir", scratch_dir_}, {"tp_rank", "0"}});

  // Non-existent file read returns NotFoundError
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey missing_key,
      backend.mapper()->MapKey("missing_key_xyz",
                               {.parallelism = {.tp_rank = 0}}));
  std::vector<uint8_t> read_buf(4096, 0);
  HostBufferDescriptor valid_desc{.ptr = read_buf.data(), .size = 4096};

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(missing_key, {valid_desc}, 4096, [&](absl::Status s) {
    read_status = s;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  EXPECT_TRUE(absl::IsNotFound(read_status)) << read_status;

  // Null pointer slice on write returns InvalidArgumentError
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey write_key,
      backend.mapper()->MapKey("null_slice_write",
                               {.parallelism = {.tp_rank = 0}}));
  HostBufferDescriptor null_desc{.ptr = nullptr, .size = 4096};

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
  HostBufferDescriptor valid_write_desc{.ptr = valid_data.data(), .size = 4096};
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
  PosixKVBackend backend("posix_disk",
                         {{"root_dir", scratch_dir_}, {"tp_rank", "0"}});
  // 1. Multi-layer 8 slices (4 KB per slice = 32 KB total)
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_8,
      backend.mapper()->MapKey("slice_8", {.parallelism = {.tp_rank = 0}}));
  VerifyScatterGatherFidelity(backend, key_8, 8, 4096);
  // 2. Distributed serving parity 32 slices (4 KB per slice = 128 KB total)
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_32,
      backend.mapper()->MapKey("slice_32", {.parallelism = {.tp_rank = 0}}));
  VerifyScatterGatherFidelity(backend, key_32, 32, 4096);
  // 3. Stress 2048 fine-grained slices (256 B per slice = 512 KB total)
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_2048,
      backend.mapper()->MapKey("slice_2048", {.parallelism = {.tp_rank = 0}}));
  VerifyScatterGatherFidelity(backend, key_2048, 2048, 256);
}

TEST_F(StorageDriverTest, MultiTpPartitionedIsolation) {
  const int tp_size = 2;
  PosixKVBackend backend("posix_disk", {{"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "model_tp_test", tp_size, /*tp_rank=*/0);

  const size_t kBlockSize = 4096;
  std::vector<std::vector<uint8_t>> written(tp_size,
                                            std::vector<uint8_t>(kBlockSize));
  for (int r = 0; r < tp_size; ++r) {
    std::fill(written[r].begin(), written[r].end(),
              static_cast<uint8_t>(0x30 * (r + 1)));
    TF_ASSERT_OK_AND_ASSIGN(
        BlockKey key,
        mapper.MapKey("tp_block", {.parallelism = {.tp_rank = r}}));
    // "tp_block" hex-encodes to 74705f626c6f636b.
    EXPECT_EQ(key.resolved_key,
              absl::StrCat(scratch_dir_, "/model_tp_test/tp2_r", r,
                           "/747/05/74705f626c6f636b.bin"));
    HostBufferDescriptor desc{.ptr = written[r].data(), .size = kBlockSize};

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
    HostBufferDescriptor desc{.ptr = read_buf.data(), .size = kBlockSize};
    TF_ASSERT_OK_AND_ASSIGN(
        BlockKey key,
        mapper.MapKey("tp_block", {.parallelism = {.tp_rank = r}}));

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

TEST_F(StorageDriverTest, RealFilesystemFaccessatExistsAndBatchExists) {
  PosixKVBackend backend("posix_disk",
                         {{"root_dir", scratch_dir_}, {"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "llama_70b", /*tp_size=*/1,
                         /*tp_rank=*/0);

  // Real 32-byte binary SHA-256 digests.
  const std::string hash_present =
      "\x1a\x2b\x3c\x4d\x5e\x6f\x70\x81\x92\xa3\xb4\xc5\xd6\xe7\xf8\x09"
      "\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00";
  const std::string hash_missing =
      "\xfe\xdc\xba\x98\x76\x54\x32\x10\x01\x23\x45\x67\x89\xab\xcd\xef"
      "\xaa\xbb\xcc\xdd\xee\xff\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99";

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_present,
      mapper.MapKey(hash_present, {.parallelism = {.tp_rank = 0}}));
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_missing,
      mapper.MapKey(hash_missing, {.parallelism = {.tp_rank = 0}}));

  // Write a real file to the scratch filesystem for key_present.
  CreateFile(key_present.resolved_key);
  ASSERT_TRUE(fs::exists(key_present.resolved_key));
  ASSERT_FALSE(fs::exists(key_missing.resolved_key));

  // Test single-key existence via RunBatchExists against real filesystem:
  auto present_res = RunBatchExists(backend, {key_present});
  ASSERT_EQ(present_res.size(), 1);
  ASSERT_TRUE(present_res[0].ok());
  EXPECT_TRUE(*present_res[0]);

  auto missing_res = RunBatchExists(backend, {key_missing});
  ASSERT_EQ(missing_res.size(), 1);
  ASSERT_TRUE(missing_res[0].ok());
  EXPECT_FALSE(*missing_res[0]);

  // Mixed batch existence probing:
  auto batch_res = RunBatchExists(backend, {key_present, key_missing});
  ASSERT_EQ(batch_res.size(), 2);
  ASSERT_TRUE(batch_res[0].ok());
  EXPECT_TRUE(*batch_res[0]);
  ASSERT_TRUE(batch_res[1].ok());
  EXPECT_FALSE(*batch_res[1]);
}

TEST_F(StorageDriverTest, OptimisticOpenCreatesDirectoriesOnEnoent) {
  PosixKVBackend backend("posix_disk",
                         {{"root_dir", scratch_dir_}, {"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "llama_70b", /*tp_size=*/1,
                         /*tp_rank=*/0);

  const std::string hash_new_dir =
      "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
      "\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x20";
  const std::string hash_second =
      "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
      "\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30";

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_new_dir,
      mapper.MapKey(hash_new_dir, {.parallelism = {.tp_rank = 0}}));

  // Confirm that the parent directory does not exist prior to WriteAsync.
  fs::path parent_dir = fs::path(key_new_dir.resolved_key).parent_path();
  ASSERT_FALSE(fs::exists(parent_dir));

  const size_t kBlockSize = 4096;
  std::vector<uint8_t> payload(kBlockSize, 0xAB);
  HostBufferDescriptor desc{.ptr = payload.data(), .size = kBlockSize};

  // First write: triggers ENOENT on open(), falls back to create_directories(),
  // and succeeds.
  absl::Notification done_1;
  absl::Status status_1;
  backend.WriteAsync(key_new_dir, {desc}, kBlockSize, [&](absl::Status s) {
    status_1 = s;
    done_1.Notify();
  });
  done_1.WaitForNotification();
  ASSERT_TRUE(status_1.ok()) << status_1;
  EXPECT_TRUE(fs::exists(key_new_dir.resolved_key));

  // Second write to same directory prefix: optimistic open() succeeds
  // immediately on first try.
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key_second,
      mapper.MapKey(hash_second, {.parallelism = {.tp_rank = 0}}));
  absl::Notification done_2;
  absl::Status status_2;
  backend.WriteAsync(key_second, {desc}, kBlockSize, [&](absl::Status s) {
    status_2 = s;
    done_2.Notify();
  });
  done_2.WaitForNotification();
  ASSERT_TRUE(status_2.ok()) << status_2;
  EXPECT_TRUE(fs::exists(key_second.resolved_key));
}

TEST_F(StorageDriverTest, VectoredIovMaxChunkingOver1024Slices) {
  PosixKVBackend backend("posix_disk",
                         {{"root_dir", scratch_dir_}, {"tp_rank", "0"}});
  PosixPathMapper mapper(scratch_dir_, "llama_70b", /*tp_size=*/1,
                         /*tp_rank=*/0);

  const std::string hash_2048 =
      "\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00\x11\x22\x33"
      "\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00\x11\x22\x33\x44";

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key, mapper.MapKey(hash_2048, {.parallelism = {.tp_rank = 0}}));

  // 2048 slices > UIO_MAXIOV (1024). Verifies loop chunking behavior.
  VerifyScatterGatherFidelity(backend, key, /*num_slices=*/2048,
                              /*slice_bytes=*/256);
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

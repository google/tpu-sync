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
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <limits.h>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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
        "posix_disk", absl::flat_hash_map<std::string, std::string>{});
    auto mapper = std::make_shared<PosixPathMapper>(
        scratch_dir_, "model_test",
        ParallelismConfig{.tp_size = 1, .tp_rank = 0});
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
  PosixPathMapper mapper(scratch_dir_, "llama3",
                         ParallelismConfig{.tp_size = 4, .tp_rank = 2});
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
  PosixPathMapper mapper(scratch_dir_, "llama3",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
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
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b",
                         ParallelismConfig{.tp_size = 8, .tp_rank = 2});

  const std::string binary_hash("\xff\xff\xff\xff\xff\xff\xff\xff", 8);
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey(binary_hash));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama_70b/tp8_r2/fff/ff/ffffffffffffffff.bin");
  // Encoding is a storage concern: block identity stays the raw bytes.
  EXPECT_EQ(key.block_hash, binary_hash);
}

// 0x2F is a path separator: unencoded, it escapes the intended directory.
TEST_F(PosixBackendTest, PosixPathMapperHashSeparatorBytesDoNotInjectDirs) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b",
                         ParallelismConfig{.tp_size = 8, .tp_rank = 2});

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
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b",
                         ParallelismConfig{.tp_size = 8, .tp_rank = 2});

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
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b",
                         ParallelismConfig{.tp_size = 8, .tp_rank = 2});

  ABSL_EXPECT_OK(mapper.MapKey(std::string(kMaxBlockHashBytes, 'a')));
  EXPECT_THAT(
      mapper.MapKey(std::string(kMaxBlockHashBytes + 1, 'a')),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("NAME_MAX")));
}

// HuggingFace-style IDs must not inject an extra directory level.
TEST_F(PosixBackendTest, PosixPathMapperSanitizesModelName) {
  PosixPathMapper mapper("/tmp/kv_cache", "meta-llama/Llama-3.1-70B",
                         ParallelismConfig{.tp_size = 8, .tp_rank = 2});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/meta-llama_Llama-3.1-70B/tp8_r2/610/00/61.bin");

  const std::vector<absl::string_view> components =
      absl::StrSplit(key.resolved_key, '/', absl::SkipEmpty());
  // tmp, kv_cache, meta-llama_Llama-3.1-70B, tp8_r2, 610, 00, 61.bin
  EXPECT_EQ(components.size(), 7);
}

// "a" hex-encodes to 61, so every key below lands in .../610/00/61.bin.
TEST_F(PosixBackendTest, PosixPathMapperPcpAndTpTopologyDir) {
  PosixPathMapper mapper(
      "/tmp/kv_cache", "llama3",
      ParallelismConfig{
          .tp_size = 1, .tp_rank = 0, .pcp_size = 8, .pcp_rank = 3});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama3/pcp8_r3_tp1_r0/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperTpOnlyTopologyDir) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3",
                         ParallelismConfig{.tp_size = 8, .tp_rank = 5});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key, "/tmp/kv_cache/llama3/tp8_r5/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperPcpOnlyTopologyDir) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3",
                         ParallelismConfig{.pcp_size = 8, .pcp_rank = 3});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key, "/tmp/kv_cache/llama3/pcp8_r3/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperNoAxesOmitsTopologyDir) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3", ParallelismConfig{});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key, "/tmp/kv_cache/llama3/610/00/61.bin");
}

// A declared size-1 axis still adds a segment, so `tp8_r3`, `pp1_r0_tp8_r3`
// and `pp1_r0_pcp1_r0_tp8_r3` are three different directories.
TEST(FormatTopologyDirTest, OneSegmentPerDeclaredAxisInPpPcpTpOrder) {
  struct Case {
    ParallelismConfig parallelism;
    std::string expected;
  };
  const Case cases[] = {
      {ParallelismConfig{}, ""},
      {ParallelismConfig{.tp_size = 8, .tp_rank = 3}, "tp8_r3"},
      {ParallelismConfig{.tp_size = 1, .tp_rank = 0}, "tp1_r0"},
      {ParallelismConfig{.pcp_size = 8, .pcp_rank = 3}, "pcp8_r3"},
      {ParallelismConfig{.pp_size = 2, .pp_rank = 1}, "pp2_r1"},
      {ParallelismConfig{
           .tp_size = 8, .tp_rank = 3, .pp_size = 1, .pp_rank = 0},
       "pp1_r0_tp8_r3"},
      {ParallelismConfig{.tp_size = 8,
                         .tp_rank = 3,
                         .pcp_size = 1,
                         .pcp_rank = 0,
                         .pp_size = 1,
                         .pp_rank = 0},
       "pp1_r0_pcp1_r0_tp8_r3"},
      {ParallelismConfig{.tp_size = 4,
                         .tp_rank = 3,
                         .pcp_size = 2,
                         .pcp_rank = 0,
                         .pp_size = 2,
                         .pp_rank = 1},
       "pp2_r1_pcp2_r0_tp4_r3"},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(FormatTopologyDir(c.parallelism), c.expected);
  }
}

TEST_F(PosixBackendTest, PosixPathMapperPerCallPcpRankOverride) {
  PosixPathMapper mapper(
      "/tmp/kv_cache", "llama3",
      ParallelismConfig{
          .tp_size = 1, .tp_rank = 0, .pcp_size = 8, .pcp_rank = 3});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key,
                          mapper.MapKey("a", {.parallelism = {.pcp_rank = 5}}));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama3/pcp8_r5_tp1_r0/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperRejectsZeroSize) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3", ParallelismConfig{});

  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.tp_size = 0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tp_size must be >= 1")));
  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.pcp_size = 0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pcp_size must be >= 1")));
}

TEST_F(PosixBackendTest, PosixPathMapperRejectsRankOutOfRange) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3", ParallelismConfig{});

  EXPECT_THAT(
      mapper.MapKey("a", {.parallelism = {.tp_size = 2, .tp_rank = 2}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("tp_rank must be in [0, 2) for tp_size 2; got 2")));
  EXPECT_THAT(
      mapper.MapKey("a", {.parallelism = {.pcp_size = 4, .pcp_rank = 4}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("pcp_rank must be in [0, 4) for pcp_size 4; got 4")));
}

TEST_F(PosixBackendTest, PosixPathMapperRejectsRankWithoutSize) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3", ParallelismConfig{});

  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.tp_rank = 0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tp_rank 0 was given without tp_size")));
  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.pcp_rank = 1}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pcp_rank 1 was given without pcp_size")));
}

// Only -1 means "use the configured value"; other negative per-call values
// are rejected rather than replaced by the mapper's valid topology.
TEST_F(PosixBackendTest, PosixPathMapperRejectsInvalidNegativeOverride) {
  PosixPathMapper mapper(
      "/tmp/kv_cache", "llama3",
      ParallelismConfig{
          .tp_size = 2, .tp_rank = 1, .pcp_size = 4, .pcp_rank = 3});

  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.tp_size = -2}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tp_size must be >= 1")));
  EXPECT_THAT(
      mapper.MapKey("a", {.parallelism = {.pcp_rank = -3}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("pcp_rank must be in [0, 4) for pcp_size 4; got -3")));
}

TEST_F(PosixBackendTest, PosixBackendOptions_ParsesPcpTopology) {
  TF_ASSERT_OK_AND_ASSIGN(
      PosixBackendOptions opts,
      PosixBackendOptions::FromProperties({{"pcp_size", "4"},
                                           {"pcp_rank", "3"},
                                           {"tp_size", "2"},
                                           {"tp_rank", "1"}}));
  EXPECT_EQ(opts.pcp_size, 4);
  EXPECT_EQ(opts.pcp_rank, 3);
  EXPECT_EQ(opts.tp_size, 2);
  EXPECT_EQ(opts.tp_rank, 1);
}

TEST_F(PosixBackendTest, PosixBackendOptions_LeavesUndeclaredAxes) {
  TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions opts,
                          PosixBackendOptions::FromProperties({}));
  EXPECT_EQ(opts.pcp_size, kAxisUndeclared);
  EXPECT_EQ(opts.pcp_rank, kAxisUndeclared);
  EXPECT_EQ(opts.tp_size, kAxisUndeclared);
  EXPECT_EQ(opts.tp_rank, kAxisUndeclared);
}

TEST_F(PosixBackendTest, PosixBackendOptions_RejectsZeroSize) {
  EXPECT_THAT(PosixBackendOptions::FromProperties({{"tp_size", "0"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tp_size must be >= 1")));
  EXPECT_THAT(PosixBackendOptions::FromProperties({{"pcp_size", "0"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pcp_size must be >= 1")));
}

TEST_F(PosixBackendTest, PosixBackendOptions_RejectsRankOutOfRange) {
  EXPECT_THAT(
      PosixBackendOptions::FromProperties({{"tp_size", "2"}, {"tp_rank", "2"}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("tp_rank must be in [0, 2) for tp_size 2; got 2")));
  EXPECT_THAT(
      PosixBackendOptions::FromProperties(
          {{"pcp_size", "4"}, {"pcp_rank", "-3"}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("pcp_rank must be in [0, 4) for pcp_size 4; got -3")));
}

TEST_F(PosixBackendTest, PosixBackendOptions_RejectsRankWithoutSize) {
  EXPECT_THAT(PosixBackendOptions::FromProperties({{"tp_rank", "0"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tp_rank 0 was given without tp_size")));
  EXPECT_THAT(PosixBackendOptions::FromProperties({{"pcp_rank", "2"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pcp_rank 2 was given without pcp_size")));
}

TEST_F(PosixBackendTest, PosixPathMapperPpPcpTpTopologyDir) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3",
                         ParallelismConfig{.tp_size = 4,
                                           .tp_rank = 3,
                                           .pcp_size = 2,
                                           .pcp_rank = 0,
                                           .pp_size = 2,
                                           .pp_rank = 1});
  EXPECT_EQ(mapper.pp_size(), 2);
  EXPECT_EQ(mapper.shards_per_block(), 16);

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama3/pp2_r1_pcp2_r0_tp4_r3/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperPpOnlyTopologyDir) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3",
                         ParallelismConfig{.pp_size = 4, .pp_rank = 2});
  EXPECT_EQ(mapper.tp_size(), kAxisUndeclared);
  EXPECT_EQ(mapper.pcp_size(), kAxisUndeclared);
  EXPECT_EQ(mapper.shards_per_block(), 4);

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("a"));
  EXPECT_EQ(key.resolved_key, "/tmp/kv_cache/llama3/pp4_r2/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperUnsetPpAxisIsUndeclared) {
  PosixPathMapper mapper(
      "/tmp/kv_cache", "llama3",
      ParallelismConfig{
          .tp_size = 2, .tp_rank = 1, .pcp_size = 2, .pcp_rank = 0});
  EXPECT_EQ(mapper.pp_size(), kAxisUndeclared);
  EXPECT_EQ(mapper.shards_per_block(), 4);
}

TEST_F(PosixBackendTest, PosixPathMapperPerCallPpRankOverride) {
  PosixPathMapper mapper(
      "/tmp/kv_cache", "llama3",
      ParallelismConfig{
          .tp_size = 2, .tp_rank = 0, .pp_size = 4, .pp_rank = 0});

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key,
                          mapper.MapKey("a", {.parallelism = {.pp_rank = 3}}));
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama3/pp4_r3_tp2_r0/610/00/61.bin");
}

TEST_F(PosixBackendTest, PosixPathMapperRejectsInvalidPpAxis) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama3", ParallelismConfig{});

  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.pp_size = 0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pp_size must be >= 1")));
  EXPECT_THAT(
      mapper.MapKey("a", {.parallelism = {.pp_size = 2, .pp_rank = 2}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("pp_rank must be in [0, 2) for pp_size 2; got 2")));
  EXPECT_THAT(mapper.MapKey("a", {.parallelism = {.pp_rank = 1}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pp_rank 1 was given without pp_size")));
}

TEST_F(PosixBackendTest, PosixBackendOptions_ParsesPpTopology) {
  TF_ASSERT_OK_AND_ASSIGN(
      PosixBackendOptions opts,
      PosixBackendOptions::FromProperties({{"pp_size", "2"},
                                           {"pp_rank", "1"},
                                           {"pcp_size", "4"},
                                           {"pcp_rank", "3"},
                                           {"tp_size", "2"},
                                           {"tp_rank", "0"}}));
  EXPECT_EQ(opts.pp_size, 2);
  EXPECT_EQ(opts.pp_rank, 1);
  EXPECT_EQ(opts.pcp_size, 4);
  EXPECT_EQ(opts.pcp_rank, 3);
  EXPECT_EQ(opts.tp_size, 2);
  EXPECT_EQ(opts.tp_rank, 0);

  TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions empty,
                          PosixBackendOptions::FromProperties({}));
  EXPECT_EQ(empty.pp_size, kAxisUndeclared);
  EXPECT_EQ(empty.pp_rank, kAxisUndeclared);
}

TEST_F(PosixBackendTest, PosixBackendOptions_RejectsInvalidPpAxis) {
  EXPECT_THAT(PosixBackendOptions::FromProperties({{"pp_size", "0"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pp_size must be >= 1")));
  EXPECT_THAT(
      PosixBackendOptions::FromProperties({{"pp_size", "2"}, {"pp_rank", "2"}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("pp_rank must be in [0, 2) for pp_size 2; got 2")));
  EXPECT_THAT(PosixBackendOptions::FromProperties({{"pp_rank", "0"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pp_rank 0 was given without pp_size")));
}

// End-to-end: a raw binary hash survives a real write/read round trip.
TEST_F(PosixBackendTest, BinaryHashWriteReadRoundTrip) {
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_binary",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

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
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_nul",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

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
// Theme 2: Factory Construction
// ===========================================================================

std::shared_ptr<PosixKVCacheStoreBackend> AsPosixStoreBackend(
    const std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>& b) {
  return std::dynamic_pointer_cast<PosixKVCacheStoreBackend>(b);
}

TEST_F(StorageDriverTest, FactoryTopologyIgnoresControllerShardsAndWorkers) {
  ::tpu_sync::rpc::RaidenIdProto unit;
  unit.set_job_name("test_job");
  unit.set_job_replica_id("0");
  unit.set_data_name("test_data");
  unit.set_data_replica_idx(0);

  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");
  config.parallelism = {.tp_size = 1, .tp_rank = 0};

  // num_shards counts controller buffer shards (e.g. chips), not storage
  // writers, so it must not leak into tp_size.
  auto c1_or = ::tpu_raiden::controller::RaidenController::Create(
      unit, /*num_blocks=*/16, /*num_shards=*/4, /*shard_size_bytes=*/512, "");
  ASSERT_TRUE(c1_or.ok());
  auto b1_or = ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                   .CreateBackend(config, c1_or->get());
  ASSERT_TRUE(b1_or.ok()) << b1_or.status();
  auto b1 = AsPosixStoreBackend(*b1_or);
  ASSERT_NE(b1, nullptr);
  EXPECT_EQ(b1->storage_backend()->mapper()->tp_size(), 1);

  // Neither does the registered worker count.
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
  ASSERT_TRUE(b2_or.ok()) << b2_or.status();
  auto b2 = AsPosixStoreBackend(*b2_or);
  ASSERT_NE(b2, nullptr);
  EXPECT_EQ(b2->storage_backend()->mapper()->tp_size(), 1);

  // And without a controller.
  auto b3_or = ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                   .CreateBackend(config, /*controller=*/nullptr);
  ASSERT_TRUE(b3_or.ok()) << b3_or.status();
  auto b3 = AsPosixStoreBackend(*b3_or);
  ASSERT_NE(b3, nullptr);
  EXPECT_EQ(b3->storage_backend()->mapper()->tp_size(), 1);
  EXPECT_EQ(b3->storage_backend()->mapper()->pcp_size(), kAxisUndeclared);
  EXPECT_EQ(b3->storage_backend()->mapper()->shards_per_block(), 1);
}

// With no axis declared every worker would write the same shard files.
TEST_F(StorageDriverTest, FactoryRejectsConfigWithNoAxisDeclared) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");

  EXPECT_THAT(::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                  .CreateBackend(config, /*controller=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("no parallelism axis is declared (received "
                                 "topology: pp=undeclared pcp=undeclared "
                                 "tp=undeclared)")));
}

TEST_F(StorageDriverTest, FactoryUsesParallelismTpSize) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");
  config.parallelism = {.tp_size = 3, .tp_rank = 2};

  auto b_or = ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                  .CreateBackend(config, /*controller=*/nullptr);
  ASSERT_TRUE(b_or.ok()) << b_or.status();
  auto b = AsPosixStoreBackend(*b_or);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->storage_backend()->mapper()->tp_size(), 3);
  // The coordinator is pinned to rank 0; parallelism.tp_rank is worker-only.
  auto posix = std::dynamic_pointer_cast<PosixKVBackend>(b->storage_backend());
  ASSERT_NE(posix, nullptr);
  EXPECT_EQ(posix->options().tp_rank, 0);
}

TEST_F(StorageDriverTest, FactoryIgnoresTpProperties) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");
  config.SetProperty("tp_size", "8");
  config.SetProperty("tp_rank", "5");
  config.parallelism = {.tp_size = 2, .tp_rank = 1};

  auto b_or = ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                  .CreateBackend(config, /*controller=*/nullptr);
  ASSERT_TRUE(b_or.ok()) << b_or.status();
  auto b = AsPosixStoreBackend(*b_or);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->storage_backend()->mapper()->tp_size(), 2);
  auto posix = std::dynamic_pointer_cast<PosixKVBackend>(b->storage_backend());
  ASSERT_NE(posix, nullptr);
  EXPECT_EQ(posix->options().tp_rank, 0);
}

TEST_F(StorageDriverTest, FactoryReturnsErrorForInvalidOptions) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");
  config.SetProperty("metadata_cache_ttl_secs", "0");
  config.parallelism = {.tp_size = 1, .tp_rank = 0};

  EXPECT_THAT(::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                  .CreateBackend(config, /*controller=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("metadata_cache_ttl_secs")));
}

TEST_F(StorageDriverTest, FactoryUsesParallelismPcpSize) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");
  config.parallelism = {
      .tp_size = 2, .tp_rank = 1, .pcp_size = 4, .pcp_rank = 3};

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend> created,
      ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
          .CreateBackend(config, /*controller=*/nullptr));
  auto b = AsPosixStoreBackend(created);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->storage_backend()->mapper()->pcp_size(), 4);
  EXPECT_EQ(b->storage_backend()->mapper()->tp_size(), 2);
  EXPECT_EQ(b->storage_backend()->mapper()->shards_per_block(), 8);
  // The coordinator is pinned to rank 0 on every declared axis.
  auto posix = std::dynamic_pointer_cast<PosixKVBackend>(b->storage_backend());
  ASSERT_NE(posix, nullptr);
  EXPECT_EQ(posix->options().pcp_rank, 0);
  EXPECT_EQ(posix->options().tp_rank, 0);
}

TEST_F(StorageDriverTest, FactoryRejectsZeroPcpSize) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.parallelism.pcp_size = 0;

  EXPECT_THAT(::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                  .CreateBackend(config, /*controller=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pcp_size must be >= 1")));
}

TEST_F(StorageDriverTest, FactoryUsesParallelismPpSize) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("model_name", "auto_model");
  config.parallelism = {.tp_size = 2,
                        .tp_rank = 1,
                        .pcp_size = 2,
                        .pcp_rank = 1,
                        .pp_size = 2,
                        .pp_rank = 1};

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend> created,
      ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
          .CreateBackend(config, /*controller=*/nullptr));
  auto b = AsPosixStoreBackend(created);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->storage_backend()->mapper()->pp_size(), 2);
  EXPECT_EQ(b->storage_backend()->mapper()->shards_per_block(), 8);
  // The coordinator is pinned to rank 0 on every declared axis.
  auto posix = std::dynamic_pointer_cast<PosixKVBackend>(b->storage_backend());
  ASSERT_NE(posix, nullptr);
  EXPECT_EQ(posix->options().pp_rank, 0);
  EXPECT_EQ(posix->options().pcp_rank, 0);
  EXPECT_EQ(posix->options().tp_rank, 0);
}

TEST_F(StorageDriverTest, FactoryRejectsPpRankWithoutPpSize) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.parallelism.pp_rank = 0;

  EXPECT_THAT(::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
                  .CreateBackend(config, /*controller=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pp_rank 0 was given without pp_size")));
}

// ===========================================================================
// Theme 3: Speculative Batch Existence Probing & Lookup
// ===========================================================================

TEST_F(StorageDriverTest, BatchExistsMixedHitsAndMisses) {
  PosixKVBackend backend("posix_disk", {});
  EXPECT_TRUE(RunBatchExists(backend, {}).empty());

  PosixPathMapper mapper(scratch_dir_, "model_test",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

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
  PosixKVBackend backend("posix_disk", {{"storage_io_thread_pool_size", "1"}});
  PosixPathMapper mapper(scratch_dir_, "model_test",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

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

TEST_F(StorageDriverTest,
       LookupSubsetContinuesPastMissingBlocksAcrossChunkBoundaries) {
  auto env = CreateStoreBackend(/*batch_size=*/2);
  EXPECT_EQ(env.store->lookup_batch_size(), 2);

  env.CreateBlock("present_0");
  env.CreateBlock("present_2");
  env.CreateBlock("present_3");

  // With batch_size=2, chunk 0 probes ["present_0", "missing_1"] (which has an
  // absent key), and chunk 1 probes ["present_2", "present_3"]. In kSubset mode
  // (plus an unmappable empty hash ""), Lookup must not stop at "missing_1" or
  // "".
  std::vector<std::string> hashes = {"present_0", "missing_1", "", "present_2",
                                     "present_3"};
  TF_ASSERT_OK_AND_ASSIGN(
      auto subset_res,
      env.store->Lookup(
          hashes,
          ::tpu_raiden::kv_cache::LookupOptions{
              .enable_global = false,
              .match_mode = ::tpu_raiden::kv_cache::LookupMatchMode::kSubset}));
  ASSERT_EQ(subset_res.size(), 3);
  EXPECT_EQ(subset_res[0].first, "present_0");
  EXPECT_EQ(subset_res[0].second.status, BlockStatus::SHARED_STORAGE);
  EXPECT_EQ(subset_res[1].first, "present_2");
  EXPECT_EQ(subset_res[1].second.status, BlockStatus::SHARED_STORAGE);
  EXPECT_EQ(subset_res[2].first, "present_3");
  EXPECT_EQ(subset_res[2].second.status, BlockStatus::SHARED_STORAGE);
}

TEST_F(StorageDriverTest,
       LookupHostResidentAndEmptySnapshotReturnEmptyOnPosixBackend) {
  auto env = CreateStoreBackend(/*batch_size=*/4);
  env.CreateBlock("present_0");
  env.CreateBlock("present_1");

  // Positive control: standard local prefix lookup finds both blocks.
  TF_ASSERT_OK_AND_ASSIGN(
      auto prefix_res, env.store->Lookup({"present_0", "present_1"},
                                         ::tpu_raiden::kv_cache::LookupOptions{
                                             .enable_global = false}));
  EXPECT_EQ(prefix_res.size(), 2);

  // Host-resident queries and empty-span snapshots return empty on secondary
  // storage.
  TF_ASSERT_OK_AND_ASSIGN(
      auto host_subset,
      env.store->Lookup(
          {"present_0", "present_1"},
          ::tpu_raiden::kv_cache::LookupOptions::HostResidentSubset()));
  EXPECT_TRUE(host_subset.empty());

  TF_ASSERT_OK_AND_ASSIGN(
      auto snapshot,
      env.store->Lookup(
          /*block_hashes=*/{},
          ::tpu_raiden::kv_cache::LookupOptions::SnapshotHostResident()));
  EXPECT_TRUE(snapshot.empty());

  // Invalid option combinations return InvalidArgumentError.
  EXPECT_THAT(
      env.store->Lookup(
          {"present_0"},
          ::tpu_raiden::kv_cache::LookupOptions{
              .enable_global = true,
              .match_mode = ::tpu_raiden::kv_cache::LookupMatchMode::kSubset}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      env.store->Lookup({"present_0"},
                        ::tpu_raiden::kv_cache::LookupOptions{
                            .enable_global = true, .host_resident_only = true}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(env.store->Lookup({"present_0"},
                                ::tpu_raiden::kv_cache::LookupOptions{
                                    .enable_global = false,
                                    .pin_found = true,
                                    .include_eviction_candidates = true}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// ===========================================================================
// Theme 4: Atomic Commit & Filesystem Fault Recovery
// ===========================================================================

TEST_F(PosixBackendTest, PosixKVBackendAtomicRenameCommit) {
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_atomic",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
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
  PosixKVBackend backend("posix_disk", {});
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
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_partial",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
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
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_offset",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
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

// ---------------------------------------------------------------------------
// Temp-file naming: bounded length and uniqueness across hosts.
// ---------------------------------------------------------------------------

namespace {

// Runs one WriteAsync of `data` to `key` and returns its status.
absl::Status WriteBlock(PosixKVBackend& backend, const BlockKey& key,
                        absl::Span<const uint8_t> data) {
  // HostBufferDescriptor::ptr is void*; WriteAsync only reads through it.
  HostBufferDescriptor src{.ptr = const_cast<uint8_t*>(data.data()),
                           .size = data.size()};
  absl::Notification done;
  absl::Status status;
  backend.WriteAsync(key, {src}, data.size(), [&](absl::Status s) {
    status = s;
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

// Names under `dir` that contain `marker`.
std::vector<std::string> EntriesContaining(const fs::path& dir,
                                           absl::string_view marker) {
  std::vector<std::string> found;
  if (!fs::exists(dir)) return found;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    if (absl::StrContains(name, marker)) found.push_back(name);
  }
  return found;
}

}  // namespace

TEST(MakeTempPathTest, SameDirectoryShortAndMarked) {
  // A final name at the hash cap is 254 characters; the temp name must not
  // grow with it, or it crosses NAME_MAX.
  const std::string final_path = absl::StrCat(
      "/some/root/model/tp1_r0/abc/de/", std::string(2 * kMaxBlockHashBytes, 'f'),
      ".bin");
  ASSERT_EQ(fs::path(final_path).filename().string().size(), NAME_MAX - 1);

  const std::string tmp = MakeTempPath(final_path);
  EXPECT_EQ(fs::path(tmp).parent_path(), fs::path(final_path).parent_path());
  const std::string base = fs::path(tmp).filename().string();
  EXPECT_TRUE(absl::StartsWith(base, ".tmp_")) << base;
  EXPECT_LT(base.size(), 60) << base;
  // "<pid>_<16 hex nonce>_<seq>" after the marker.
  const std::vector<std::string> parts =
      absl::StrSplit(absl::StripPrefix(base, ".tmp_"), '_');
  ASSERT_EQ(parts.size(), 3) << base;
  EXPECT_EQ(parts[0], absl::StrCat(getpid()));
  EXPECT_EQ(parts[1].size(), 16) << base;
}

TEST(MakeTempPathTest, UniqueAcrossThreads) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 10000;
  std::vector<std::vector<std::string>> per_thread(kThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      per_thread[t].reserve(kPerThread);
      for (int i = 0; i < kPerThread; ++i) {
        per_thread[t].push_back(MakeTempPath("/r/m/a/b/c.bin"));
      }
    });
  }
  for (auto& th : threads) th.join();
  std::vector<std::string> all;
  for (auto& v : per_thread) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  EXPECT_EQ(std::adjacent_find(all.begin(), all.end()), all.end())
      << "duplicate temp path";
}

// The temp name is unique by construction, so a file already at that name
// belongs to another writer. O_EXCL makes the open fail instead of sharing
// (and truncating) that writer's inode, and the foreign file is left alone.
TEST_F(PosixBackendTest, PosixKVBackendTempOpenEexistIsReportedAndFileKept) {
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_eexist",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey("block_eexist", {.parallelism = {.tp_rank = 0}}));

  // Predict the name WriteAsync will draw next: same pid and nonce, and the
  // process-wide counter advances by one per call. Nothing else in this test
  // calls MakeTempPath between here and the write.
  const std::string probe = MakeTempPath(key.resolved_key);
  const size_t sep = probe.rfind('_');
  ASSERT_NE(sep, std::string::npos);
  uint64_t seq = 0;
  ASSERT_TRUE(absl::SimpleAtoi(probe.substr(sep + 1), &seq));
  const std::string next_tmp = absl::StrCat(probe.substr(0, sep + 1), seq + 1);
  CreateFile(next_tmp);
  const uintmax_t foreign_size = fs::file_size(next_tmp);

  const std::vector<uint8_t> data(1024, 0x77);
  EXPECT_THAT(WriteBlock(backend, key, data),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("Failed to open file for write")));
  EXPECT_FALSE(fs::exists(key.resolved_key));
  ASSERT_TRUE(fs::exists(next_tmp)) << "foreign temp file was removed";
  EXPECT_EQ(fs::file_size(next_tmp), foreign_size)
      << "foreign temp file was truncated";
}

// Several writers racing on one key -- the situation two replicas sharing a
// directory are in -- each publish a complete file by rename; the last rename
// wins and every writer reports success.
TEST_F(PosixBackendTest, PosixKVBackendConcurrentSameKeyWritesLastRenameWins) {
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_race",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper.MapKey("block_race", {.parallelism = {.tp_rank = 0}}));

  constexpr int kWriters = 16;
  std::vector<uint8_t> data(64 * 1024, 0xC3);
  absl::Mutex mu;
  std::vector<absl::Status> statuses;  // Guarded by mu.
  absl::Notification all_done;
  for (int i = 0; i < kWriters; ++i) {
    HostBufferDescriptor src{.ptr = data.data(), .size = data.size()};
    backend.WriteAsync(key, {src}, data.size(), [&](absl::Status s) {
      absl::MutexLock lock(mu);
      statuses.push_back(s);
      if (statuses.size() == kWriters) all_done.Notify();
    });
  }
  all_done.WaitForNotification();
  {
    absl::MutexLock lock(mu);
    for (const absl::Status& s : statuses) ABSL_EXPECT_OK(s);
  }

  ASSERT_TRUE(fs::exists(key.resolved_key));
  EXPECT_EQ(fs::file_size(key.resolved_key), data.size());
  std::vector<uint8_t> read_back(data.size(), 0);
  HostBufferDescriptor dst{.ptr = read_back.data(), .size = read_back.size()};
  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, {dst}, data.size(), [&](absl::Status s) {
    read_status = s;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ABSL_ASSERT_OK(read_status);
  EXPECT_EQ(read_back, data);
  EXPECT_THAT(EntriesContaining(fs::path(key.resolved_key).parent_path(),
                                ".tmp_"),
              ::testing::IsEmpty());
}

// Sanitizing only replaces characters; a model name that is still longer
// than NAME_MAX afterwards would make every write fail with ENAMETOOLONG, so
// it is refused when the backend is configured.
TEST_F(PosixBackendTest, PosixBackendOptions_RejectsOverlongModelName) {
  EXPECT_THAT(PosixBackendOptions::FromProperties(
                  {{"model_name", std::string(NAME_MAX + 1, 'm')}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("model_name sanitizes to 256")));
  // Sanitization does not shorten: 300 '/' become 300 '_'.
  EXPECT_THAT(PosixBackendOptions::FromProperties(
                  {{"model_name", std::string(300, '/')}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("the limit is 255")));
  // Exactly NAME_MAX is fine.
  TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions opts,
                          PosixBackendOptions::FromProperties(
                              {{"model_name", std::string(NAME_MAX, 'm')}}));
  EXPECT_EQ(opts.model_name.size(), NAME_MAX);
}

TEST_F(PosixBackendTest, PosixKVBackendPreadPrematureEOFReturnsDataLossError) {
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_trunc",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});
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
  PosixKVBackend backend(
      "posix_disk",
      {{"root_dir", scratch_dir_}, {"tp_size", "1"}, {"tp_rank", "0"}});

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
  PosixKVBackend backend(
      "posix_disk",
      {{"root_dir", scratch_dir_}, {"tp_size", "1"}, {"tp_rank", "0"}});
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
  PosixKVBackend backend("posix_disk", {});
  PosixPathMapper mapper(scratch_dir_, "model_tp_test",
                         ParallelismConfig{.tp_size = tp_size, .tp_rank = 0});

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
  PosixKVBackend backend("posix_disk", {{"root_dir", scratch_dir_}});
  PosixPathMapper mapper(scratch_dir_, "llama_70b",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

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
  PosixKVBackend backend("posix_disk", {{"root_dir", scratch_dir_}});
  PosixPathMapper mapper(scratch_dir_, "llama_70b",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

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
  PosixKVBackend backend("posix_disk", {{"root_dir", scratch_dir_}});
  PosixPathMapper mapper(scratch_dir_, "llama_70b",
                         ParallelismConfig{.tp_size = 1, .tp_rank = 0});

  const std::string hash_2048 =
      "\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00\x11\x22\x33"
      "\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00\x11\x22\x33\x44";

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key, mapper.MapKey(hash_2048, {.parallelism = {.tp_rank = 0}}));

  // 2048 slices > UIO_MAXIOV (1024). Verifies loop chunking behavior.
  VerifyScatterGatherFidelity(backend, key, /*num_slices=*/2048,
                              /*slice_bytes=*/256);
}

// ===========================================================================
// Theme: Direct I/O (O_DIRECT) Tests
// ===========================================================================

// Helper function to read current Dirty memory in kB from /proc/meminfo.
static int64_t ReadDirtyPageCacheKb() {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (absl::StartsWith(line, "Dirty:")) {
      std::vector<std::string> parts =
          absl::StrSplit(line, ' ', absl::SkipWhitespace());
      int64_t val = 0;
      if (parts.size() >= 2 && absl::SimpleAtoi(parts[1], &val)) {
        return val;
      }
    }
  }
  return -1;
}

TEST_F(PosixBackendTest, PosixBackendOptions_DirectIOPropertyParsing) {
  // 1. Defaults to false when omitted
  {
    TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions opts,
                            PosixBackendOptions::FromProperties({}));
    EXPECT_FALSE(opts.direct_io);
  }

  // 2. Strict true: "true", "TRUE", "True"
  for (const char* true_val : {"true", "TRUE", "True"}) {
    TF_ASSERT_OK_AND_ASSIGN(
        PosixBackendOptions opts,
        PosixBackendOptions::FromProperties({{"direct_io", true_val}}));
    EXPECT_TRUE(opts.direct_io) << "Failed for: " << true_val;
  }

  // 3. Strict false: "false", "FALSE", "False"
  for (const char* false_val : {"false", "FALSE", "False"}) {
    TF_ASSERT_OK_AND_ASSIGN(
        PosixBackendOptions opts,
        PosixBackendOptions::FromProperties({{"direct_io", false_val}}));
    EXPECT_FALSE(opts.direct_io) << "Failed for: " << false_val;
  }

  // 4. Invalid strings return InvalidArgument
  for (const char* invalid_val : {"1", "yes", "0", "no", "invalid"}) {
    EXPECT_THAT(
        PosixBackendOptions::FromProperties({{"direct_io", invalid_val}}),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("expected 'true' or 'false'")));
  }
}

TEST_F(PosixBackendTest, PosixKVBackend_ProbeDirectIOLifecycleAndCleanup) {
  bool supported = PosixKVBackend::ProbeDirectIO(scratch_dir_);
  (void)supported;
  for (const auto& entry : fs::directory_iterator(scratch_dir_)) {
    EXPECT_THAT(entry.path().filename().string(),
                testing::Not(HasSubstr(".o_direct_probe_")));
  }
}

// Owns one mmap'd, page-aligned region (the XlaHostMemoryAllocator pattern).
class AlignedRegion {
 public:
  explicit AlignedRegion(size_t bytes) : bytes_(bytes) {
    void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ptr_ = (p == MAP_FAILED) ? nullptr : static_cast<uint8_t*>(p);
  }
  ~AlignedRegion() {
    if (ptr_ != nullptr) munmap(ptr_, bytes_);
  }
  AlignedRegion(const AlignedRegion&) = delete;
  AlignedRegion& operator=(const AlignedRegion&) = delete;
  uint8_t* data() const { return ptr_; }

 private:
  size_t bytes_;
  uint8_t* ptr_ = nullptr;
};

absl::Status RunWrite(PosixKVBackend& backend, const BlockKey& key,
                      absl::Span<const HostBufferDescriptor> slices,
                      size_t total_bytes) {
  absl::Notification done;
  absl::Status status;
  backend.WriteAsync(key, slices, total_bytes, [&](absl::Status s) {
    status = std::move(s);
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

absl::Status RunRead(PosixKVBackend& backend, const BlockKey& key,
                     absl::Span<const HostBufferDescriptor> slices,
                     size_t total_bytes) {
  absl::Notification done;
  absl::Status status;
  backend.ReadAsync(key, slices, total_bytes, [&](absl::Status s) {
    status = std::move(s);
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

TEST(DirectIOAlignmentTest, MatchesPageSizeAndIsPowerOfTwo) {
  const size_t align = DirectIOAlignment();
  EXPECT_EQ(align, static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  EXPECT_GT(align, 0u);
  EXPECT_EQ(align & (align - 1), 0u);
}

TEST(DirectIOAlignmentTest, SlicesAreDirectIOAligned) {
  const size_t align = DirectIOAlignment();
  AlignedRegion region(2 * align);
  ASSERT_NE(region.data(), nullptr);
  uint8_t* base = region.data();

  EXPECT_TRUE(SlicesAreDirectIOAligned({}));
  EXPECT_TRUE(SlicesAreDirectIOAligned(
      {HostBufferDescriptor{.ptr = base, .size = align},
       HostBufferDescriptor{.ptr = base + align, .size = align}}));
  // Misaligned pointer.
  EXPECT_FALSE(SlicesAreDirectIOAligned(
      {HostBufferDescriptor{.ptr = base + 1, .size = align}}));
  // Misaligned size.
  EXPECT_FALSE(SlicesAreDirectIOAligned(
      {HostBufferDescriptor{.ptr = base, .size = align - 1}}));
  // One bad slice among good ones.
  EXPECT_FALSE(SlicesAreDirectIOAligned(
      {HostBufferDescriptor{.ptr = base, .size = align},
       HostBufferDescriptor{.ptr = base + align, .size = 1234}}));
}

TEST_F(PosixBackendTest, PosixKVBackend_DirectIO_AlignedWriteAndReadFidelity) {
  if (!PosixKVBackend::ProbeDirectIO(scratch_dir_)) {
    GTEST_SKIP() << "O_DIRECT is not supported on " << scratch_dir_;
  }
  auto backend = std::make_shared<PosixKVBackend>(
      "posix_directio",
      absl::flat_hash_map<std::string, std::string>{
          {"root_dir", scratch_dir_},
          {"direct_io", "true"}});
  ASSERT_TRUE(backend->is_direct_io_supported());

  auto mapper = std::make_shared<PosixPathMapper>(
      scratch_dir_, "model_test",
      ParallelismConfig{.tp_size = 1, .tp_rank = 0});
  backend->set_mapper(mapper);

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper->MapKey("directio_hash", {.parallelism = {.tp_rank = 0}}));

  const size_t align = DirectIOAlignment();
  AlignedRegion src(align);
  AlignedRegion dst(align);
  ASSERT_NE(src.data(), nullptr);
  ASSERT_NE(dst.data(), nullptr);
  for (size_t i = 0; i < align; ++i) {
    src.data()[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(dst.data(), 0, align);

  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = src.data(), .size = align}};

  int64_t dirty_before = ReadDirtyPageCacheKb();
  ASSERT_OK(RunWrite(*backend, key, write_slices, align));

  // Direct I/O verification via Linux page cache dirty page tracking:
  // Standard buffered I/O writes data into the kernel page cache first, which
  // immediately increments the OS-wide 'Dirty:' counter in /proc/meminfo by
  // the number of newly dirtied pages until flushed by writeback threads.
  // In contrast, O_DIRECT writes bypass the page cache entirely and transfer
  // directly between user-space buffers and disk, so 'Dirty:' must not increase
  // (delta <= 0) as a result of this write.
  int64_t dirty_after = ReadDirtyPageCacheKb();
  if (dirty_before >= 0 && dirty_after >= 0) {
    EXPECT_LE(dirty_after - dirty_before, 0);
  }

  std::vector<HostBufferDescriptor> read_slices = {
      HostBufferDescriptor{.ptr = dst.data(), .size = align}};
  ASSERT_OK(RunRead(*backend, key, read_slices, align));

  EXPECT_EQ(std::memcmp(src.data(), dst.data(), align), 0);
}

TEST_F(PosixBackendTest, PosixKVBackend_DirectIO_UnsupportedFilesystemIsFatal) {
  // The probe's open(2) cannot succeed under /proc, so construction with
  // direct_io=true must fail explicitly instead of falling back.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        PosixKVBackend backend(
            "posix_unsupported",
            absl::flat_hash_map<std::string, std::string>{
                {"root_dir", "/proc/raiden_no_o_direct"},
                {"direct_io", "true"}});
      },
      "O_DIRECT is not supported");
}

TEST_F(PosixBackendTest, PosixKVBackend_DirectIO_UnalignedBufferRejected) {
  if (!PosixKVBackend::ProbeDirectIO(scratch_dir_)) {
    GTEST_SKIP() << "O_DIRECT is not supported on " << scratch_dir_;
  }
  auto backend = std::make_shared<PosixKVBackend>(
      "posix_unaligned",
      absl::flat_hash_map<std::string, std::string>{
          {"root_dir", scratch_dir_},
          {"direct_io", "true"}});
  auto mapper = std::make_shared<PosixPathMapper>(
      scratch_dir_, "model_test",
      ParallelismConfig{.tp_size = 1, .tp_rank = 0});
  backend->set_mapper(mapper);

  TF_ASSERT_OK_AND_ASSIGN(
      BlockKey key,
      mapper->MapKey("unaligned_hash", {.parallelism = {.tp_rank = 0}}));

  const size_t align = DirectIOAlignment();
  AlignedRegion region(2 * align);
  ASSERT_NE(region.data(), nullptr);
  uint8_t* base = region.data();

  const std::vector<HostBufferDescriptor> misaligned_ptr = {
      HostBufferDescriptor{.ptr = base + 1, .size = align}};
  const std::vector<HostBufferDescriptor> misaligned_size = {
      HostBufferDescriptor{.ptr = base, .size = 1234}};
  const auto kNotAligned = StatusIs(absl::StatusCode::kInvalidArgument,
                                    HasSubstr("not aligned to"));

  // Writes are rejected before any file (temp or final) is created.
  EXPECT_THAT(RunWrite(*backend, key, misaligned_ptr, align), kNotAligned);
  EXPECT_THAT(RunWrite(*backend, key, misaligned_size, 1234), kNotAligned);
  const fs::path model_dir = fs::path(scratch_dir_) / "model_test";
  if (fs::exists(model_dir)) {
    for (const auto& entry : fs::recursive_directory_iterator(model_dir)) {
      EXPECT_FALSE(entry.is_regular_file()) << entry.path();
    }
  }

  // Reads: misaligned pointer, size, or file offset.
  EXPECT_THAT(RunRead(*backend, key, misaligned_ptr, align), kNotAligned);
  EXPECT_THAT(RunRead(*backend, key, misaligned_size, 1234), kNotAligned);
  BlockKey offset_key = key;
  offset_key.offset = 512;
  const std::vector<HostBufferDescriptor> aligned = {
      HostBufferDescriptor{.ptr = base, .size = align}};
  EXPECT_THAT(RunRead(*backend, offset_key, aligned, align), kNotAligned);
}

// ===========================================================================
// Theme 8: Metadata Cache
// ===========================================================================

// Counts existence probes so tests can observe which lookups reach storage.
class CountingPosixKVBackend : public PosixKVBackend {
 public:
  using PosixKVBackend::PosixKVBackend;

  void BatchExistsAsync(absl::Span<const BlockKey> keys,
                        std::function<void(std::vector<absl::StatusOr<bool>>)>
                            callback) override {
    {
      absl::MutexLock lock(mu_);
      ++batch_calls_;
      probed_keys_ += keys.size();
    }
    PosixKVBackend::BatchExistsAsync(keys, std::move(callback));
  }

  int batch_calls() const {
    absl::MutexLock lock(mu_);
    return batch_calls_;
  }
  size_t probed_keys() const {
    absl::MutexLock lock(mu_);
    return probed_keys_;
  }
  void ResetCounts() {
    absl::MutexLock lock(mu_);
    batch_calls_ = 0;
    probed_keys_ = 0;
  }

 private:
  mutable absl::Mutex mu_;
  int batch_calls_ ABSL_GUARDED_BY(mu_) = 0;
  size_t probed_keys_ ABSL_GUARDED_BY(mu_) = 0;
};

class MetadataCacheTest : public StorageDriverTest {
 protected:
  // Short enough to wait out in a test, long enough that a hit checked right
  // after insertion is still fresh on a loaded machine.
  static constexpr absl::Duration kShortTtl = absl::Milliseconds(200);

  struct Env {
    std::shared_ptr<CountingPosixKVBackend> backend;
    std::shared_ptr<PosixPathMapper> mapper;
    std::unique_ptr<PosixKVCacheStoreBackend> store;

    // A rank is ignored when the mapper leaves its axis undeclared.
    std::string PathOf(const std::string& hash, int tp_rank = 0,
                       int pcp_rank = 0, int pp_rank = 0) {
      KeyMappingOptions options;
      if (mapper->tp_size() != kAxisUndeclared) {
        options.parallelism.tp_rank = tp_rank;
      }
      if (mapper->pcp_size() != kAxisUndeclared) {
        options.parallelism.pcp_rank = pcp_rank;
      }
      if (mapper->pp_size() != kAxisUndeclared) {
        options.parallelism.pp_rank = pp_rank;
      }
      absl::StatusOr<BlockKey> key = mapper->MapKey(hash, options);
      EXPECT_TRUE(key.ok()) << key.status();
      return key.ok() ? key->resolved_key : "";
    }
    void CreateBlock(const std::string& hash, int tp_rank = 0, int pcp_rank = 0,
                     int pp_rank = 0) {
      const std::string path = PathOf(hash, tp_rank, pcp_rank, pp_rank);
      fs::create_directories(fs::path(path).parent_path());
      std::ofstream file(path, std::ios::binary);
      file << "data";
    }
    void RemoveBlock(const std::string& hash, int tp_rank = 0, int pcp_rank = 0,
                     int pp_rank = 0) {
      ASSERT_TRUE(fs::remove(PathOf(hash, tp_rank, pcp_rank, pp_rank)))
          << hash << " pp_r" << pp_rank << " pcp_r" << pcp_rank << " tp_r"
          << tp_rank;
    }
    // Writes the shard of every (pp_rank, pcp_rank, tp_rank) triple.
    void CreateAllShards(const std::string& hash) {
      for (int q = 0; q < std::max(1, mapper->pp_size()); ++q) {
        for (int p = 0; p < std::max(1, mapper->pcp_size()); ++p) {
          for (int r = 0; r < std::max(1, mapper->tp_size()); ++r) {
            CreateBlock(hash, r, p, q);
          }
        }
      }
    }
    void RemoveAllShards(const std::string& hash) {
      for (int q = 0; q < std::max(1, mapper->pp_size()); ++q) {
        for (int p = 0; p < std::max(1, mapper->pcp_size()); ++p) {
          for (int r = 0; r < std::max(1, mapper->tp_size()); ++r) {
            RemoveBlock(hash, r, p, q);
          }
        }
      }
    }
    size_t LookupCount(const std::vector<std::string>& hashes) {
      absl::StatusOr<BlockSliceList> result = store->Lookup(hashes);
      EXPECT_TRUE(result.ok()) << result.status();
      return result.ok() ? result->size() : 0;
    }
  };

  static MetadataCacheOptions Enabled(size_t max_entries = 1000,
                                      absl::Duration ttl = absl::Seconds(60)) {
    return MetadataCacheOptions{.max_entries = max_entries, .ttl = ttl};
  }

  // Each axis is declared (with rank 0) only when its size != -1.
  Env CreateEnv(MetadataCacheOptions cache_options, size_t batch_size = 4,
                int tp_size = 1, int pcp_size = kAxisUndeclared,
                int pp_size = kAxisUndeclared) {
    auto backend = std::make_shared<CountingPosixKVBackend>(
        "posix", absl::flat_hash_map<std::string, std::string>{});
    auto rank_for = [](int size) {
      return size == kAxisUndeclared ? kAxisUndeclared : 0;
    };
    auto mapper = std::make_shared<PosixPathMapper>(
        scratch_dir_, "model_test",
        ParallelismConfig{.tp_size = tp_size,
                          .tp_rank = rank_for(tp_size),
                          .pcp_size = pcp_size,
                          .pcp_rank = rank_for(pcp_size),
                          .pp_size = pp_size,
                          .pp_rank = rank_for(pp_size)});
    backend->set_mapper(mapper);
    auto store = std::make_unique<PosixKVCacheStoreBackend>(
        backend, "posix", 0, batch_size, cache_options);
    return Env{std::move(backend), std::move(mapper), std::move(store)};
  }
};

TEST_F(PosixBackendTest, PosixBackendOptions_MetadataCachePropertyParsing) {
  {
    TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions opts,
                            PosixBackendOptions::FromProperties({}));
    EXPECT_EQ(opts.metadata_cache_max_entries, 0);
    EXPECT_EQ(opts.metadata_cache_ttl_secs, 60);
    const MetadataCacheOptions cache =
        MetadataCacheOptions::FromPosixOptions(opts);
    EXPECT_FALSE(cache.enabled());
    EXPECT_EQ(cache.max_entries, 0);
    EXPECT_EQ(cache.ttl, absl::Seconds(60));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions opts,
                            PosixBackendOptions::FromProperties(
                                {{"metadata_cache_max_entries", "-1"},
                                 {"metadata_cache_ttl_secs", "-1"}}));
    const MetadataCacheOptions cache =
        MetadataCacheOptions::FromPosixOptions(opts);
    EXPECT_TRUE(cache.enabled());
    EXPECT_EQ(cache.max_entries, std::numeric_limits<size_t>::max());
    EXPECT_EQ(cache.ttl, absl::InfiniteDuration());
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(PosixBackendOptions opts,
                            PosixBackendOptions::FromProperties(
                                {{"metadata_cache_max_entries", "7"},
                                 {"metadata_cache_ttl_secs", "5"}}));
    const MetadataCacheOptions cache =
        MetadataCacheOptions::FromPosixOptions(opts);
    EXPECT_TRUE(cache.enabled());
    EXPECT_EQ(cache.max_entries, 7);
    EXPECT_EQ(cache.ttl, absl::Seconds(5));
  }
  for (const char* invalid_val : {"-2", "abc"}) {
    EXPECT_THAT(PosixBackendOptions::FromProperties(
                    {{"metadata_cache_max_entries", invalid_val}}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("metadata_cache_max_entries")))
        << invalid_val;
  }
  for (const char* invalid_val : {"0", "-2", "abc"}) {
    EXPECT_THAT(PosixBackendOptions::FromProperties(
                    {{"metadata_cache_ttl_secs", invalid_val}}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("metadata_cache_ttl_secs")))
        << invalid_val;
  }
}

TEST_F(StorageDriverTest, FactoryWiresMetadataCacheOptions) {
  ::tpu_raiden::kv_cache::BackendConfig config;
  config.type = "posix";
  config.SetProperty("root_dir", scratch_dir_);
  config.SetProperty("metadata_cache_max_entries", "10");
  config.SetProperty("metadata_cache_ttl_secs", "5");
  config.parallelism = {.tp_size = 1, .tp_rank = 0};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend> created,
      ::tpu_raiden::kv_cache::KVCacheStoreBackendFactory::Instance()
          .CreateBackend(config, /*controller=*/nullptr));
  auto store = std::dynamic_pointer_cast<PosixKVCacheStoreBackend>(created);
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(store->metadata_cache_options().enabled());
  EXPECT_EQ(store->metadata_cache_options().max_entries, 10);
  EXPECT_EQ(store->metadata_cache_options().ttl, absl::Seconds(5));
}

TEST_F(MetadataCacheTest, DisabledProbesStorageEveryTime) {
  Env env = CreateEnv(MetadataCacheOptions{});
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->batch_calls(), 2);
  EXPECT_EQ(env.store->metadata_cache_size(), 0);

  env.RemoveBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.store->Delete({"a"}), 0);  // No-op when disabled.
}

TEST_F(MetadataCacheTest, CachedHitSkipsStorageProbe) {
  Env env = CreateEnv(Enabled());
  for (const char* hash : {"a", "b", "c"}) env.CreateBlock(hash);

  EXPECT_EQ(env.LookupCount({"a", "b", "c"}), 3);
  EXPECT_EQ(env.backend->batch_calls(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 3);

  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a", "b", "c"}), 3);
  EXPECT_EQ(env.backend->batch_calls(), 0);
}

TEST_F(MetadataCacheTest, NoSingleRankShardIsAWitness) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/4, /*tp_size=*/2);
  env.CreateBlock("a", /*tp_rank=*/1);
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  env.RemoveBlock("a", /*tp_rank=*/1);
  env.CreateBlock("b", /*tp_rank=*/0);
  EXPECT_EQ(env.LookupCount({"b"}), 0);
}

TEST_F(MetadataCacheTest, AllRankShardsAreRequiredAndCached) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, /*tp_size=*/4);
  for (int r : {0, 1, 2}) env.CreateBlock("a", r);
  // Rank 3 has not landed yet: miss, but the 3 confirmed shards are cached.
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.store->metadata_cache_size(), 3);

  // Once rank 3 lands, only that shard is probed.
  env.CreateBlock("a", /*tp_rank=*/3);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->probed_keys(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 4);

  // Every shard is now cached.
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->batch_calls(), 0);
}

TEST_F(MetadataCacheTest, DirectStorageLookupRequiresAllShards) {
  Env env = CreateEnv(MetadataCacheOptions{}, /*batch_size=*/16,
                      /*tp_size=*/4);
  env.CreateAllShards("a");
  for (int r : {0, 1, 3}) env.CreateBlock("b", r);  // Rank 2's write failed.
  env.CreateAllShards("c");
  TF_ASSERT_OK_AND_ASSIGN(BlockSliceList result,
                          env.store->Lookup({"a", "b", "c"}));
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].first, "a");
  EXPECT_EQ(env.store->metadata_cache_size(), 0);
}

TEST_F(MetadataCacheTest, ShardsSplitAcrossChunksStopAtMissingShard) {
  // 4 shards per block in chunks of 3: block boundaries do not align with
  // chunk boundaries.
  Env env = CreateEnv(MetadataCacheOptions{}, /*batch_size=*/3,
                      /*tp_size=*/4);
  env.CreateAllShards("a");
  env.CreateAllShards("b");
  for (int r : {0, 2, 3}) env.CreateBlock("c", r);  // c is missing rank 1.
  env.CreateAllShards("d");
  TF_ASSERT_OK_AND_ASSIGN(BlockSliceList result,
                          env.store->Lookup({"a", "b", "c", "d"}));
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].first, "a");
  EXPECT_EQ(result[1].first, "b");
  // Flat shard list: a0 a1 a2 | a3 b0 b1 | b2 b3 c0 | c1(absent) c2 c3 | ...
  // The 4th chunk holds the absent c1, so d's shards are never probed.
  EXPECT_EQ(env.backend->batch_calls(), 4);
  EXPECT_EQ(env.backend->probed_keys(), 12);
}

TEST_F(MetadataCacheTest, PartialCacheHitProbesOnlyMissingShards) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/4, /*tp_size=*/2);
  env.CreateBlock("a", /*tp_rank=*/0);
  EXPECT_EQ(env.LookupCount({"a"}), 0);  // Caches r0.
  ASSERT_EQ(env.store->metadata_cache_size(), 1);

  // r0 is still fresh in the cache even after the file is gone; r1 only on
  // storage. The block is answered from both.
  env.RemoveBlock("a", /*tp_rank=*/0);
  env.CreateBlock("a", /*tp_rank=*/1);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->probed_keys(), 1);
}

TEST_F(MetadataCacheTest, DeleteInvalidatesAllRankShards) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, /*tp_size=*/3);
  env.CreateAllShards("a");
  env.CreateAllShards("b");
  EXPECT_EQ(env.LookupCount({"a", "b"}), 2);
  EXPECT_EQ(env.store->metadata_cache_size(), 6);

  EXPECT_EQ(env.store->Delete({"a"}), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 3);

  // a's rank-2 shard is gone; the re-probe sees it.
  env.RemoveBlock("a", /*tp_rank=*/2);
  EXPECT_EQ(env.LookupCount({"a", "b"}), 0);
}

TEST_F(MetadataCacheTest, PcpOnlyTpUndeclaredHitRequiresEveryPcpShard) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, kAxisUndeclared,
                      /*pcp_size=*/4);
  EXPECT_EQ(env.mapper->shards_per_block(), 4);
  // "a" hex-encodes to 61.
  EXPECT_EQ(env.PathOf("a", /*tp_rank=*/0, /*pcp_rank=*/3),
            absl::StrCat(scratch_dir_, "/model_test/pcp4_r3/610/00/61.bin"));
  for (int p : {0, 1, 2}) env.CreateBlock("a", /*tp_rank=*/0, p);
  // pcp rank 3 has not landed yet: miss, but the 3 confirmed shards are cached.
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.store->metadata_cache_size(), 3);

  // Once pcp rank 3 lands, only that shard is probed.
  env.CreateBlock("a", /*tp_rank=*/0, /*pcp_rank=*/3);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->probed_keys(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 4);
}

TEST_F(MetadataCacheTest, NoAxesDeclaredSingleShard) {
  Env env =
      CreateEnv(Enabled(), /*batch_size=*/4, kAxisUndeclared, kAxisUndeclared);
  EXPECT_EQ(env.mapper->shards_per_block(), 1);
  EXPECT_EQ(env.PathOf("a"),
            absl::StrCat(scratch_dir_, "/model_test/610/00/61.bin"));
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 1);

  EXPECT_EQ(env.store->Delete({"a"}), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 0);
  env.RemoveBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 0);
}

TEST_F(MetadataCacheTest, PcpAndTp1HitRequiresEveryPcpShard) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, /*tp_size=*/1,
                      /*pcp_size=*/4);
  for (int p : {0, 1, 2}) env.CreateBlock("a", /*tp_rank=*/0, p);
  // pcp rank 3 has not landed yet: miss, but the 3 confirmed shards are cached.
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.store->metadata_cache_size(), 3);

  // Once pcp rank 3 lands, only that shard is probed.
  env.CreateBlock("a", /*tp_rank=*/0, /*pcp_rank=*/3);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->probed_keys(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 4);

  // Every shard is now cached.
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->batch_calls(), 0);
}

TEST_F(MetadataCacheTest, PcpTimesTpHitRequiresEveryShard) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, /*tp_size=*/2,
                      /*pcp_size=*/2);
  env.CreateBlock("a", /*tp_rank=*/0, /*pcp_rank=*/0);
  env.CreateBlock("a", /*tp_rank=*/1, /*pcp_rank=*/0);
  env.CreateBlock("a", /*tp_rank=*/0, /*pcp_rank=*/1);
  // Shard (pcp 1, tp 1) is missing: miss, the other 3 are cached.
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.store->metadata_cache_size(), 3);

  env.CreateBlock("a", /*tp_rank=*/1, /*pcp_rank=*/1);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->probed_keys(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 4);
}

TEST_F(MetadataCacheTest, DeleteInvalidatesAllPcpTimesTpShards) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, /*tp_size=*/2,
                      /*pcp_size=*/2);
  env.CreateAllShards("a");
  env.CreateAllShards("b");
  EXPECT_EQ(env.LookupCount({"a", "b"}), 2);
  EXPECT_EQ(env.store->metadata_cache_size(), 8);

  EXPECT_EQ(env.store->Delete({"a"}), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 4);

  // a's (pcp 1, tp 1) shard is gone; the re-probe sees it.
  env.RemoveBlock("a", /*tp_rank=*/1, /*pcp_rank=*/1);
  EXPECT_EQ(env.LookupCount({"a", "b"}), 0);
}

TEST_F(MetadataCacheTest, PpTimesPcpTimesTpHitRequiresEveryShard) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/16, /*tp_size=*/2,
                      /*pcp_size=*/2, /*pp_size=*/2);
  EXPECT_EQ(env.mapper->shards_per_block(), 8);
  // "a" hex-encodes to 61.
  EXPECT_EQ(env.PathOf("a", /*tp_rank=*/1, /*pcp_rank=*/0, /*pp_rank=*/1),
            absl::StrCat(scratch_dir_,
                         "/model_test/pp2_r1_pcp2_r0_tp2_r1/610/00/61.bin"));
  env.CreateAllShards("a");
  // Shard (pp 1, pcp 1, tp 1) is missing: miss, the other 7 are cached.
  env.RemoveBlock("a", /*tp_rank=*/1, /*pcp_rank=*/1, /*pp_rank=*/1);
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.store->metadata_cache_size(), 7);

  // Once it lands, only that shard is probed.
  env.CreateBlock("a", /*tp_rank=*/1, /*pcp_rank=*/1, /*pp_rank=*/1);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->probed_keys(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 8);
}

TEST_F(MetadataCacheTest, PpOnlyHitRequiresEveryPpShard) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/8, kAxisUndeclared,
                      kAxisUndeclared, /*pp_size=*/4);
  EXPECT_EQ(env.mapper->shards_per_block(), 4);
  EXPECT_EQ(env.PathOf("a", /*tp_rank=*/0, /*pcp_rank=*/0, /*pp_rank=*/3),
            absl::StrCat(scratch_dir_, "/model_test/pp4_r3/610/00/61.bin"));
  for (int q : {0, 1, 2})
    env.CreateBlock("a", /*tp_rank=*/0, /*pcp_rank=*/0, q);
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  env.CreateBlock("a", /*tp_rank=*/0, /*pcp_rank=*/0, /*pp_rank=*/3);
  EXPECT_EQ(env.LookupCount({"a"}), 1);
}

TEST_F(MetadataCacheTest, DeleteInvalidatesAllPpPcpTpShards) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/16, /*tp_size=*/2,
                      /*pcp_size=*/2, /*pp_size=*/2);
  env.CreateAllShards("a");
  env.CreateAllShards("b");
  EXPECT_EQ(env.LookupCount({"a", "b"}), 2);
  EXPECT_EQ(env.store->metadata_cache_size(), 16);

  EXPECT_EQ(env.store->Delete({"a"}), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 8);

  // a's (pp 1, pcp 0, tp 0) shard is gone; the re-probe sees it.
  env.RemoveBlock("a", /*tp_rank=*/0, /*pcp_rank=*/0, /*pp_rank=*/1);
  EXPECT_EQ(env.LookupCount({"a", "b"}), 0);
}

TEST_F(MetadataCacheTest, StaleEntryServedWithinTtlThenExpires) {
  Env env = CreateEnv(Enabled(/*max_entries=*/1000, kShortTtl));
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);

  // Removed out of band: the cache still vouches for it until the TTL passes.
  env.RemoveBlock("a");
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->batch_calls(), 0);

  // No hits for a full TTL after the last one: the entry expires.
  absl::SleepFor(kShortTtl + absl::Milliseconds(50));
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.backend->batch_calls(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 0);
}

TEST_F(MetadataCacheTest, CacheHitsExtendTheTtl) {
  Env env = CreateEnv(Enabled(/*max_entries=*/1000, kShortTtl));
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  const absl::Time inserted = absl::Now();
  // Removed out of band, so any storage probe would report "a" absent: every
  // positive answer below must come from the cache.
  env.RemoveBlock("a");
  env.backend->ResetCounts();

  // Keep hitting the entry for twice its TTL. Each hit refreshes last_used,
  // so it stays fresh well past `inserted + kShortTtl`.
  while (absl::Now() - inserted < 2 * kShortTtl) {
    EXPECT_EQ(env.LookupCount({"a"}), 1);
    absl::SleepFor(absl::Milliseconds(20));
  }
  EXPECT_EQ(env.backend->batch_calls(), 0);

  // Once idle for a full TTL it expires and storage is probed again.
  absl::SleepFor(kShortTtl + absl::Milliseconds(50));
  EXPECT_EQ(env.LookupCount({"a"}), 0);
  EXPECT_EQ(env.backend->batch_calls(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 0);
}

TEST_F(MetadataCacheTest, InfiniteTtlNeverExpires) {
  Env env = CreateEnv(Enabled(/*max_entries=*/1000, absl::InfiniteDuration()));
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  env.RemoveBlock("a");
  absl::SleepFor(kShortTtl);
  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.backend->batch_calls(), 0);
}

TEST_F(MetadataCacheTest, DeleteInvalidatesSoNextLookupReprobes) {
  Env env = CreateEnv(Enabled());
  env.CreateBlock("a");
  env.CreateBlock("b");
  EXPECT_EQ(env.LookupCount({"a", "b"}), 2);

  env.RemoveBlock("a");
  std::vector<int> deallocated_ids = {99};
  std::vector<std::string> deleted_hashes = {"stale"};
  EXPECT_EQ(env.store->Delete(
                {"a", "uncached", std::string(kMaxBlockHashBytes + 1, 'x')},
                DeleteOptions{.deallocated_host_block_ids = &deallocated_ids,
                              .deleted_hashes = &deleted_hashes}),
            1);
  EXPECT_TRUE(deallocated_ids.empty());
  EXPECT_THAT(deleted_hashes, ::testing::ElementsAre("a"));
  EXPECT_EQ(env.store->metadata_cache_size(), 1);

  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"a", "b"}), 0);
  EXPECT_EQ(env.backend->probed_keys(), 1);  // Only "a"; "b" still cached.
}

TEST_F(MetadataCacheTest, DeleteHostResidentOnlyIsNoOp) {
  Env env = CreateEnv(Enabled());
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 1);

  std::vector<int> deallocated_ids = {99};
  std::vector<std::string> evicted_hashes = {"stale"};
  EXPECT_EQ(env.store->Delete({"a"}, DeleteOptions::EvictHostResident(
                                         &deallocated_ids, &evicted_hashes)),
            0);
  EXPECT_TRUE(deallocated_ids.empty());
  EXPECT_TRUE(evicted_hashes.empty());
  EXPECT_EQ(env.store->metadata_cache_size(), 1);
}

TEST_F(MetadataCacheTest, MaxEntriesEvictsLeastRecentlyUsed) {
  Env env = CreateEnv(Enabled(/*max_entries=*/2));
  for (const char* hash : {"a", "b", "c"}) env.CreateBlock(hash);

  EXPECT_EQ(env.LookupCount({"a", "b", "c"}), 3);
  EXPECT_EQ(env.store->metadata_cache_size(), 2);

  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"b", "c"}), 2);
  EXPECT_EQ(env.backend->batch_calls(), 0);
  EXPECT_EQ(env.LookupCount({"a"}), 1);  // Evicted: probed again.
  EXPECT_EQ(env.backend->probed_keys(), 1);
  EXPECT_EQ(env.store->metadata_cache_size(), 2);
}

TEST_F(MetadataCacheTest, UnboundedCacheHoldsEverything) {
  Env env = CreateEnv(Enabled(std::numeric_limits<size_t>::max()),
                      /*batch_size=*/16);
  std::vector<std::string> hashes;
  for (int i = 0; i < 100; ++i) {
    hashes.push_back(absl::StrCat("block_", i));
    env.CreateBlock(hashes.back());
  }
  EXPECT_EQ(env.LookupCount(hashes), 100);
  EXPECT_EQ(env.store->metadata_cache_size(), 100);
}

TEST_F(MetadataCacheTest, PrefixMergesCacheAndStorage) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/4);
  env.CreateBlock("a");
  env.CreateBlock("c");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  EXPECT_EQ(env.LookupCount({"c"}), 1);
  // Only the cache still knows "a" and "c" (within TTL); "b" is only on disk.
  env.RemoveBlock("a");
  env.RemoveBlock("c");
  env.CreateBlock("b");

  env.backend->ResetCounts();
  TF_ASSERT_OK_AND_ASSIGN(BlockSliceList result,
                          env.store->Lookup({"a", "b", "c", "d"}));
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0].first, "a");
  EXPECT_EQ(result[1].first, "b");
  EXPECT_EQ(result[2].first, "c");
  for (const auto& [hash, block] : result) {
    EXPECT_EQ(block.status, BlockStatus::SHARED_STORAGE);
    EXPECT_EQ(block.raiden_id.job_replica_id, "shared");
    EXPECT_EQ(block.raiden_id.data_name, "posix");
  }
  // Only the cache misses ("b", "d") were probed, in one batch.
  EXPECT_EQ(env.backend->batch_calls(), 1);
  EXPECT_EQ(env.backend->probed_keys(), 2);
}

TEST_F(MetadataCacheTest, StopsAtFirstBlockAbsentEverywhere) {
  Env env = CreateEnv(Enabled());
  env.CreateBlock("a");
  EXPECT_EQ(env.LookupCount({"a"}), 1);
  env.CreateBlock("c");
  EXPECT_EQ(env.LookupCount({"a", "b", "c"}), 1);
}

TEST_F(MetadataCacheTest, ColdLookupProbesOnlyUntilFirstMissingChunk) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/4);
  std::vector<std::string> hashes;
  for (int i = 0; i < 12; ++i) hashes.push_back(absl::StrCat("block_", i));
  env.CreateBlock(hashes[0]);
  env.CreateBlock(hashes[1]);

  EXPECT_EQ(env.LookupCount(hashes), 2);
  EXPECT_EQ(env.backend->batch_calls(), 1);
  EXPECT_EQ(env.backend->probed_keys(), 4);
}

TEST_F(MetadataCacheTest, ConfirmedBlocksPastThePrefixAreCached) {
  Env env = CreateEnv(Enabled(), /*batch_size=*/4);
  env.CreateBlock("b");
  env.CreateBlock("c");
  EXPECT_EQ(env.LookupCount({"a", "b", "c"}), 0);
  EXPECT_EQ(env.store->metadata_cache_size(), 2);

  env.backend->ResetCounts();
  EXPECT_EQ(env.LookupCount({"b", "c"}), 2);
  EXPECT_EQ(env.backend->batch_calls(), 0);
}

TEST_F(MetadataCacheTest, UnmappableHashTruncatesPrefix) {
  Env env = CreateEnv(Enabled());
  env.CreateBlock("a");
  env.CreateBlock("c");
  EXPECT_EQ(env.LookupCount({"a", std::string(kMaxBlockHashBytes + 1, 'x'),
                             "c"}),
            1);
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

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

#include "tpu_sync/kv_cache/backends/storage/tds_backend.h"

#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/storage_backend_utils.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

static_assert(std::is_base_of_v<tkv::backends::KVBackend,
                                tkv::backends::storage::TdsKVBackend>);

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size, size_t alignment = GetStorageDirectIOAlignment())
      : size_(size) {
    void* raw = nullptr;
    if (posix_memalign(&raw, alignment, size) != 0) {
      raw = nullptr;
    }
    ptr_ = static_cast<uint8_t*>(raw);
  }

  ~AlignedBuffer() { std::free(ptr_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  uint8_t* data() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  uint8_t* ptr_ = nullptr;
  size_t size_ = 0;
};

std::vector<uint8_t> MakeDeterministicPayload(size_t size, uint8_t seed) {
  std::vector<uint8_t> payload(size);
  for (size_t i = 0; i < size; ++i) {
    payload[i] =
        static_cast<uint8_t>((i * 37 + (i >> 8) * 101 + seed * 13 + 7) & 0xFF);
  }
  return payload;
}

int64_t FileSizeOrNegative(const std::string& path) {
  struct stat st = {};
  if (::stat(path.c_str(), &st) != 0) return -1;
  return st.st_size;
}

class TdsKVBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = info != nullptr ? info->name() : "test";
    test_dir_ = absl::StrCat(::testing::TempDir(), "/tds_kv_", test_name);
    std::filesystem::remove_all(test_dir_);
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  std::string test_dir_;
};

// Verifies that page-aligned multi-slice buffers are written and read back
// end-to-end through `TdsKVBackend` (`tds_writev` / `tds_readv` with raw
// buffer descriptors, using `O_DIRECT` when supported by the test filesystem).
TEST_F(TdsKVBackendTest, DirectZeroCopyMultiSliceReadWrite) {
  TdsKVBackend backend("tds_direct", {{"root_dir", test_dir_},
                                      {"storage_io_thread_pool_size", "4"},
                                      {"direct_io", "true"}});

  const size_t align = GetStorageDirectIOAlignment();
  const size_t kSlice0Size = align;
  const size_t kSlice1Size = 2 * align;
  const size_t kSlice2Size = align;
  const size_t kTotalBytes = kSlice0Size + kSlice1Size + kSlice2Size;

  AlignedBuffer src0(kSlice0Size);
  AlignedBuffer src1(kSlice1Size);
  AlignedBuffer src2(kSlice2Size);
  ASSERT_NE(src0.data(), nullptr);
  ASSERT_NE(src1.data(), nullptr);
  ASSERT_NE(src2.data(), nullptr);

  std::vector<uint8_t> expected0 = MakeDeterministicPayload(kSlice0Size, 1);
  std::vector<uint8_t> expected1 = MakeDeterministicPayload(kSlice1Size, 2);
  std::vector<uint8_t> expected2 = MakeDeterministicPayload(kSlice2Size, 3);
  std::memcpy(src0.data(), expected0.data(), kSlice0Size);
  std::memcpy(src1.data(), expected1.data(), kSlice1Size);
  std::memcpy(src2.data(), expected2.data(), kSlice2Size);

  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = src0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = src1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = src2.data(), .size = kSlice2Size},
  };

  BlockKey key;
  key.block_hash = "direct_block_01";
  key.resolved_key = absl::StrCat(test_dir_, "/direct_block_01.bin");
  key.offset = 0;
  key.size = static_cast<int64_t>(kTotalBytes);

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, write_slices, kTotalBytes, [&](absl::Status status) {
    write_status = std::move(status);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  AlignedBuffer dst0(kSlice0Size);
  AlignedBuffer dst1(kSlice1Size);
  AlignedBuffer dst2(kSlice2Size);
  std::memset(dst0.data(), 0, kSlice0Size);
  std::memset(dst1.data(), 0, kSlice1Size);
  std::memset(dst2.data(), 0, kSlice2Size);

  std::vector<HostBufferDescriptor> read_slices = {
      HostBufferDescriptor{.ptr = dst0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = dst1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = dst2.data(), .size = kSlice2Size},
  };

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, read_slices, kTotalBytes, [&](absl::Status status) {
    read_status = std::move(status);
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ABSL_ASSERT_OK(read_status);

  EXPECT_EQ(std::memcmp(dst0.data(), expected0.data(), kSlice0Size), 0);
  EXPECT_EQ(std::memcmp(dst1.data(), expected1.data(), kSlice1Size), 0);
  EXPECT_EQ(std::memcmp(dst2.data(), expected2.data(), kSlice2Size), 0);
}

// Verifies that unaligned slice pointers/sizes automatically omit `O_DIRECT`
// when opening the file and complete multi-slice writes and reads via buffered
// `libtdsul` I/O without corrupting file size or payload bytes.
TEST_F(TdsKVBackendTest, UnalignedBufferedFallbackMultiSliceReadWrite) {
  TdsKVBackend backend("tds_unaligned", {{"root_dir", test_dir_},
                                         {"storage_io_thread_pool_size", "4"},
                                         {"direct_io", "true"}});

  constexpr size_t kSlice0Size = 1000;
  constexpr size_t kSlice1Size = 513;
  constexpr size_t kSlice2Size = 2000;
  constexpr size_t kTotalBytes = kSlice0Size + kSlice1Size + kSlice2Size;
  static_assert(kTotalBytes == 3513);

  std::vector<uint8_t> src0 = MakeDeterministicPayload(kSlice0Size, 11);
  std::vector<uint8_t> src1 = MakeDeterministicPayload(kSlice1Size, 22);
  std::vector<uint8_t> src2 = MakeDeterministicPayload(kSlice2Size, 33);

  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = src0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = src1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = src2.data(), .size = kSlice2Size},
  };

  BlockKey key;
  key.block_hash = "bounced_block_01";
  key.resolved_key = absl::StrCat(test_dir_, "/bounced_block_01.bin");
  key.offset = 0;
  key.size = kTotalBytes;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, write_slices, kTotalBytes, [&](absl::Status status) {
    write_status = std::move(status);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  ASSERT_TRUE(std::filesystem::exists(key.resolved_key));
  EXPECT_EQ(std::filesystem::file_size(key.resolved_key), kTotalBytes);

  std::vector<uint8_t> dst0(kSlice0Size, 0);
  std::vector<uint8_t> dst1(kSlice1Size, 0);
  std::vector<uint8_t> dst2(kSlice2Size, 0);

  std::vector<HostBufferDescriptor> read_slices = {
      HostBufferDescriptor{.ptr = dst0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = dst1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = dst2.data(), .size = kSlice2Size},
  };

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, read_slices, kTotalBytes, [&](absl::Status status) {
    read_status = std::move(status);
    read_done.Notify();
  });
  read_done.WaitForNotification();
  ABSL_ASSERT_OK(read_status);

  EXPECT_EQ(dst0, src0);
  EXPECT_EQ(dst1, src1);
  EXPECT_EQ(dst2, src2);
}

// Verifies that `BatchExistsAsync` concurrently checks file existence across
// present and missing block keys and preserves input ordering in results.
TEST_F(TdsKVBackendTest, BatchExistsAsync) {
  TdsKVBackend backend(
      "tds_exists",
      {{"root_dir", test_dir_}, {"storage_io_thread_pool_size", "8"}});

  constexpr int kNumKeys = 16;
  std::vector<uint8_t> payload = MakeDeterministicPayload(2048, 7);
  HostBufferDescriptor slice{.ptr = payload.data(), .size = payload.size()};

  std::vector<BlockKey> keys;
  for (int i = 0; i < kNumKeys; ++i) {
    BlockKey key;
    key.block_hash = absl::StrCat("block_", i);
    key.resolved_key = absl::StrCat(test_dir_, "/exists_", i, ".bin");
    key.offset = 0;
    key.size = static_cast<int64_t>(payload.size());
    keys.push_back(key);

    if (i % 2 == 0) {
      absl::Notification write_done;
      absl::Status write_status;
      backend.WriteAsync(key, {slice}, payload.size(), [&](absl::Status s) {
        write_status = std::move(s);
        write_done.Notify();
      });
      write_done.WaitForNotification();
      ABSL_ASSERT_OK(write_status);
    }
  }

  absl::Notification batch_done;
  std::vector<absl::StatusOr<bool>> results;
  backend.BatchExistsAsync(keys, [&](std::vector<absl::StatusOr<bool>> res) {
    results = std::move(res);
    batch_done.Notify();
  });
  batch_done.WaitForNotification();

  ASSERT_EQ(results.size(), kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    EXPECT_THAT(results[i], IsOkAndHolds(i % 2 == 0)) << "Failed at key " << i;
  }
}

// Verifies that non-contiguous page-aligned sub-slices carved out of a larger
// contiguous host buffer pool are gathered on write and read back into a single
// contiguous destination buffer in slice order.
TEST_F(TdsKVBackendTest, SubSlicesOfAlignedPoolReadWrite) {
  TdsKVBackend backend("tds_pool", {{"root_dir", test_dir_}});

  const size_t align = GetStorageDirectIOAlignment();
  const size_t kPoolSize = 16 * align;
  AlignedBuffer pool(kPoolSize);
  ASSERT_NE(pool.data(), nullptr);
  std::vector<uint8_t> fill = MakeDeterministicPayload(kPoolSize, 55);
  std::memcpy(pool.data(), fill.data(), kPoolSize);

  std::vector<HostBufferDescriptor> slices = {
      HostBufferDescriptor{.ptr = pool.data() + align, .size = 2 * align},
      HostBufferDescriptor{.ptr = pool.data() + 8 * align, .size = align},
  };
  const size_t kTotalBytes = 3 * align;
  BlockKey key;
  key.block_hash = "pool_block";
  key.resolved_key = absl::StrCat(test_dir_, "/pool_block.bin");
  key.offset = 0;
  key.size = static_cast<int64_t>(kTotalBytes);

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, slices, kTotalBytes, [&](absl::Status s) {
    write_status = std::move(s);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  AlignedBuffer read_buf(kTotalBytes);
  std::memset(read_buf.data(), 0, kTotalBytes);
  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(
      key, {HostBufferDescriptor{.ptr = read_buf.data(), .size = kTotalBytes}},
      kTotalBytes, [&](absl::Status s) {
        read_status = std::move(s);
        read_done.Notify();
      });
  read_done.WaitForNotification();
  ABSL_ASSERT_OK(read_status);
  EXPECT_EQ(std::memcmp(read_buf.data(), pool.data() + align, 2 * align), 0);
  EXPECT_EQ(std::memcmp(read_buf.data() + 2 * align, pool.data() + 8 * align,
                        align),
            0);
}

// Verifies that `TdsKVBackend` uses `StoragePathMapper` for shard-aware path
// mapping and is registered under `"tds"` in `KVCacheStoreBackendFactory`.
TEST_F(TdsKVBackendTest,
       StoragePathMapperAndKVCacheStoreBackendFactoryIntegration) {
  absl::flat_hash_map<std::string, std::string> props = {
      {"root_dir", test_dir_},
      {"model_name", "gemini_ultra"},
      {"tp_size", "8"},
      {"tp_rank", "3"},
  };
  TdsKVBackend backend("tds_mapper", props);
  ASSERT_NE(backend.mapper(), nullptr);
  EXPECT_EQ(backend.mapper()->tp_size(), 8);

  const std::string binary_hash("\xab\xcd\x00\x2f\xef\x01", 6);
  absl::StatusOr<BlockKey> mapped = backend.mapper()->MapKey(binary_hash);
  ABSL_ASSERT_OK(mapped);
  EXPECT_EQ(mapped->block_hash, binary_hash);
  EXPECT_EQ(
      mapped->resolved_key,
      absl::StrCat(test_dir_, "/gemini_ultra/tp8_r3/abc/d0/abcd002fef01.bin"));

  BackendConfig store_cfg;
  store_cfg.type = std::string(kTdsBackendName);
  store_cfg.properties = {
      {"root_dir", test_dir_},
      {"model_name", "gemini_ultra"},
      {"tp_size", "1"},
      {"capacity_bytes", "1048576"},
  };
  absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> store_backend =
      KVCacheStoreBackendFactory::Instance().Create(store_cfg,
                                                    /*controller=*/nullptr);
  ABSL_ASSERT_OK(store_backend);
  ASSERT_NE(*store_backend, nullptr);
  EXPECT_EQ((*store_backend)->name(), kTdsBackendName);
  EXPECT_EQ((*store_backend)->GetCapacity(), 1048576);

  TdsKVBackend rank0_writer("tds_rank0", {{"root_dir", test_dir_},
                                          {"model_name", "gemini_ultra"},
                                          {"tp_size", "1"},
                                          {"tp_rank", "0"}});
  absl::StatusOr<BlockKey> rank0_key =
      rank0_writer.mapper()->MapKey(binary_hash);
  ABSL_ASSERT_OK(rank0_key);

  const size_t align = GetStorageDirectIOAlignment();
  AlignedBuffer buf(align);
  std::memset(buf.data(), 0x5A, align);
  absl::Notification write_done;
  absl::Status write_status;
  rank0_writer.WriteAsync(
      *rank0_key, {HostBufferDescriptor{.ptr = buf.data(), .size = align}},
      align, [&](absl::Status s) {
        write_status = std::move(s);
        write_done.Notify();
      });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  absl::StatusOr<BlockSliceList> lookup_hits =
      (*store_backend)->Lookup({binary_hash, "missing_hash"});
  ABSL_ASSERT_OK(lookup_hits);
  ASSERT_EQ(lookup_hits->size(), 1);
  EXPECT_EQ((*lookup_hits)[0].first, binary_hash);
  EXPECT_EQ((*lookup_hits)[0].second.status, BlockStatus::SHARED_STORAGE);
}

// Verifies concurrent multi-block, multi-slice writes followed by controller
// store lookup and concurrent reads across worker threads.
TEST_F(TdsKVBackendTest, ConcurrentMultiSliceWriteLookupAndRead) {
  constexpr int kNumBlocks = 12;
  const size_t kSliceBytes = GetStorageDirectIOAlignment();
  constexpr size_t kSlicesPerBlock = 3;
  const size_t kBlockBytes = kSliceBytes * kSlicesPerBlock;

  BackendConfig store_cfg;
  store_cfg.type = std::string(kTdsBackendName);
  store_cfg.properties = {
      {"root_dir", test_dir_},
      {"model_name", "concurrent_model"},
      {"tp_size", "1"},
      {"storage_io_thread_pool_size", "8"},
  };
  absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> store_backend =
      KVCacheStoreBackendFactory::Instance().Create(store_cfg,
                                                    /*controller=*/nullptr);
  ABSL_ASSERT_OK(store_backend);

  TdsKVBackend worker_backend("tds_worker", store_cfg.properties);

  std::vector<std::unique_ptr<AlignedBuffer>> write_arena;
  std::vector<std::string> block_hashes;
  std::vector<BlockKey> mapped_keys;
  for (int b = 0; b < kNumBlocks; ++b) {
    block_hashes.push_back(absl::StrCat("concurrent_block_hash_", b));
    absl::StatusOr<BlockKey> key =
        worker_backend.mapper()->MapKey(block_hashes.back());
    ABSL_ASSERT_OK(key);
    mapped_keys.push_back(*std::move(key));
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      auto buf = std::make_unique<AlignedBuffer>(kSliceBytes);
      std::vector<uint8_t> payload = MakeDeterministicPayload(
          kSliceBytes, static_cast<uint8_t>(b * 7 + s + 1));
      std::memcpy(buf->data(), payload.data(), kSliceBytes);
      write_arena.push_back(std::move(buf));
    }
  }

  std::atomic<int> remaining_writes{kNumBlocks};
  absl::Notification all_writes_done;
  std::vector<absl::Status> write_statuses(kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    std::vector<HostBufferDescriptor> slices;
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      slices.push_back(HostBufferDescriptor{
          .ptr = write_arena[b * kSlicesPerBlock + s]->data(),
          .size = kSliceBytes});
    }
    worker_backend.WriteAsync(
        mapped_keys[b], slices, kBlockBytes, [&, b](absl::Status status) {
          write_statuses[b] = std::move(status);
          if (remaining_writes.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            all_writes_done.Notify();
          }
        });
  }
  all_writes_done.WaitForNotification();
  for (int b = 0; b < kNumBlocks; ++b) {
    ABSL_ASSERT_OK(write_statuses[b]);
  }

  absl::StatusOr<BlockSliceList> hits = (*store_backend)->Lookup(block_hashes);
  ABSL_ASSERT_OK(hits);
  ASSERT_EQ(hits->size(), kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    EXPECT_EQ((*hits)[b].first, block_hashes[b]);
    EXPECT_EQ((*hits)[b].second.status, BlockStatus::SHARED_STORAGE);
  }

  std::vector<std::unique_ptr<AlignedBuffer>> read_arena;
  for (size_t i = 0; i < kNumBlocks * kSlicesPerBlock; ++i) {
    auto buf = std::make_unique<AlignedBuffer>(kSliceBytes);
    std::memset(buf->data(), 0, kSliceBytes);
    read_arena.push_back(std::move(buf));
  }

  std::atomic<int> remaining_reads{kNumBlocks};
  absl::Notification all_reads_done;
  std::vector<absl::Status> read_statuses(kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    std::vector<HostBufferDescriptor> slices;
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      slices.push_back(HostBufferDescriptor{
          .ptr = read_arena[b * kSlicesPerBlock + s]->data(),
          .size = kSliceBytes});
    }
    worker_backend.ReadAsync(
        mapped_keys[b], slices, kBlockBytes, [&, b](absl::Status status) {
          read_statuses[b] = std::move(status);
          if (remaining_reads.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            all_reads_done.Notify();
          }
        });
  }
  all_reads_done.WaitForNotification();
  for (int b = 0; b < kNumBlocks; ++b) {
    ABSL_ASSERT_OK(read_statuses[b]);
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      const size_t idx = b * kSlicesPerBlock + s;
      EXPECT_EQ(std::memcmp(read_arena[idx]->data(), write_arena[idx]->data(),
                            kSliceBytes),
                0);
    }
  }
}

// Verifies error status reporting for missing files, slice size mismatches,
// non-zero write offsets, and null slice pointers.
TEST_F(TdsKVBackendTest, ErrorHandling) {
  TdsKVBackend backend("tds_errors", {{"root_dir", test_dir_}});

  const size_t align = GetStorageDirectIOAlignment();
  AlignedBuffer buf(align);

  // 1. Reading a non-existent file returns kNotFound.
  {
    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(
        BlockKey{.block_hash = "non_existent",
                 .resolved_key = absl::StrCat(test_dir_, "/missing.bin"),
                 .offset = 0,
                 .size = static_cast<int64_t>(align)},
        {HostBufferDescriptor{.ptr = buf.data(), .size = align}}, align,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
  }

  // 2. Writing with mismatched total_bytes returns kInvalidArgument.
  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        BlockKey{.block_hash = "mismatch",
                 .resolved_key = absl::StrCat(test_dir_, "/mismatch.bin"),
                 .offset = 0,
                 .size = static_cast<int64_t>(2 * align)},
        {HostBufferDescriptor{.ptr = buf.data(), .size = align}}, 2 * align,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3. Writing with non-zero offset returns kInvalidArgument.
  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        BlockKey{.block_hash = "nonzero_offset",
                 .resolved_key = absl::StrCat(test_dir_, "/nonzero.bin"),
                 .offset = static_cast<int64_t>(align),
                 .size = static_cast<int64_t>(align)},
        {HostBufferDescriptor{.ptr = buf.data(), .size = align}}, align,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 4. Null slice pointer with non-zero size returns kInvalidArgument.
  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        BlockKey{.block_hash = "null_slice",
                 .resolved_key = absl::StrCat(test_dir_, "/null_slice.bin"),
                 .offset = 0,
                 .size = static_cast<int64_t>(align)},
        {HostBufferDescriptor{.ptr = nullptr, .size = align}}, align,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

// Verifies that reading from an unaligned file offset falls back to buffered
// I/O (omitting `O_DIRECT`) and returns the exact sub-range of bytes at
// `key.offset`.
TEST_F(TdsKVBackendTest, UnalignedOffsetReadReturnsCorrectBytes) {
  TdsKVBackend backend("tds_unaligned_read",
                       {{"root_dir", test_dir_},
                        {"storage_io_thread_pool_size", "2"},
                        {"direct_io", "true"}});

  const size_t align = GetStorageDirectIOAlignment();
  const size_t kBaselineBytes = 3 * align;
  AlignedBuffer src(kBaselineBytes);
  ASSERT_NE(src.data(), nullptr);
  const std::vector<uint8_t> baseline =
      MakeDeterministicPayload(kBaselineBytes, 97);
  std::memcpy(src.data(), baseline.data(), kBaselineBytes);

  BlockKey write_key;
  write_key.block_hash = "bounce_read_baseline";
  write_key.resolved_key = absl::StrCat(test_dir_, "/bounce_read.bin");
  write_key.offset = 0;
  write_key.size = static_cast<int64_t>(kBaselineBytes);

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        write_key,
        {HostBufferDescriptor{.ptr = src.data(), .size = kBaselineBytes}},
        kBaselineBytes, [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  const int64_t kReadOffset = static_cast<int64_t>(align) + 904;
  constexpr size_t kReadBytes = 300;
  std::vector<uint8_t> dst(kReadBytes, 0);

  BlockKey read_key;
  read_key.block_hash = "bounce_read_baseline";
  read_key.resolved_key = write_key.resolved_key;
  read_key.offset = kReadOffset;
  read_key.size = kReadBytes;

  {
    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(
        read_key, {HostBufferDescriptor{.ptr = dst.data(), .size = kReadBytes}},
        kReadBytes, [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(std::memcmp(dst.data(), baseline.data() + kReadOffset, kReadBytes),
            0);
}

// Verifies that scatter-gather transfers with more than `UIO_MAXIOV` (1024)
// slices are split into multiple `tds_writev` / `tds_readv` batches with
// advancing file offsets.
TEST_F(TdsKVBackendTest, ScatterGatherCrossesIovLimitInMultipleBatches) {
  TdsKVBackend backend("tds_iov",
                       {{"root_dir", test_dir_},
                        {"storage_io_thread_pool_size", "2"},
                        {"direct_io", "true"}});

  const size_t align = GetStorageDirectIOAlignment();
  constexpr size_t kPages = static_cast<size_t>(UIO_MAXIOV) + 1;
  const size_t kTotalBytes = kPages * align;

  AlignedBuffer src(kTotalBytes);
  AlignedBuffer dst(kTotalBytes);
  ASSERT_NE(src.data(), nullptr);
  ASSERT_NE(dst.data(), nullptr);

  const std::vector<uint8_t> payload =
      MakeDeterministicPayload(kTotalBytes, 53);
  std::memcpy(src.data(), payload.data(), kTotalBytes);
  std::memset(dst.data(), 0, kTotalBytes);

  std::vector<HostBufferDescriptor> write_slices;
  std::vector<HostBufferDescriptor> read_slices;
  write_slices.reserve(kPages);
  read_slices.reserve(kPages);
  for (size_t i = 0; i < kPages; ++i) {
    write_slices.push_back(
        HostBufferDescriptor{.ptr = src.data() + i * align, .size = align});
    read_slices.push_back(
        HostBufferDescriptor{.ptr = dst.data() + i * align, .size = align});
  }

  BlockKey key;
  key.block_hash = "iov_block";
  key.resolved_key = absl::StrCat(test_dir_, "/iov_block.bin");
  key.offset = 0;
  key.size = static_cast<int64_t>(kTotalBytes);

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(key, write_slices, kTotalBytes, [&](absl::Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(FileSizeOrNegative(key.resolved_key),
            static_cast<int64_t>(kTotalBytes));

  {
    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(key, read_slices, kTotalBytes, [&](absl::Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(std::memcmp(dst.data(), src.data(), kTotalBytes), 0);
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

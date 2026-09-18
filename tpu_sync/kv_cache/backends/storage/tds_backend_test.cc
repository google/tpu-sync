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

#include <fcntl.h>
#include <sys/mman.h>
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
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

// Verify class inheritance hierarchy and namespace alias.
static_assert(std::is_base_of_v<tkv::backends::KVBackend,
                                tkv::backends::storage::TdsKVBackend>);

constexpr size_t kAlignment = 4096;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size, size_t alignment = kAlignment)
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

// Builds a reproducible byte pattern for round-trip comparisons.
//
// The `i >> 8` term is load-bearing. A plain `i * 37` repeats every 256 bytes
// (`37 * 256 == 0 (mod 256)`), and since 4096 is a multiple of 256 that would
// make every 4KB page byte-identical. Any test that read the wrong page would
// then still compare equal, hiding precisely the offset bugs these tests exist
// to catch. Mixing in the high bits of `i` gives each page a distinct pattern.
std::vector<uint8_t> MakeDeterministicPayload(size_t size, uint8_t seed) {
  std::vector<uint8_t> payload(size);
  for (size_t i = 0; i < size; ++i) {
    payload[i] =
        static_cast<uint8_t>((i * 37 + (i >> 8) * 101 + seed * 13 + 7) & 0xFF);
  }
  return payload;
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

// =============================================================================
// Test Suite Overview (17 tests, numbered in file order; tests 1-11 also have
// a "Test N" header comment)
// =============================================================================
//
// Group A: bounce-free transfers (`direct_ops`) via `libtdsul`
//   1. DirectZeroCopyMultiSliceReadWrite
//      3 aligned, registered slices -> `tds_writev` / `tds_readv`;
//      `direct_ops == 2`, `bounced_ops == 0`.
//   6. SubSliceOfRegisteredPoolUsesDirectZeroCopy
//      Slices inside one registered 64KB arena use the arena's
//      `tds_buffer_handle_t` at non-zero offsets.
//   7. UnregisteredAlignedSlicesZeroCopyOrBouncePerRequireReg
//      Unregistered aligned slice: bounce-free with
//      `require_registration = false`, bounced with `true`.
//  11. LibtdsulDmaBufRegistrationAndSubSliceReadWrite
//      `memfd` registered via `tds_buffer_register_dmabuf(..., TDS_MEM_HOST)`;
//      reads use `{.fd = memfd, .offset}` descriptors.
//  12. ODirectIsMeasuredAndReportedSeparatelyFromBouncing
//      `o_direct_ops` reflects `fcntl(fd, F_GETFL) & O_DIRECT`, independent of
//      `direct_ops` / `bounced_ops`.
//  17. ScatterGatherCrossesIovLimitInMultipleBatches
//      `UIO_MAXIOV + 1` aligned pages force `ExecuteTdsIo()` to issue more
//      than one `tds_writev` / `tds_readv` batch.
//
// Group B: `BounceWindow` and read-modify-write (`bounced_ops`)
//   2. UnalignedBounceBufferMultiSliceReadWrite
//      1000B + 513B + 2000B = 3513B at offset 0; file is truncated to 3513B.
//   5. UnalignedNonZeroOffsetPreservesAdjacentBytes
//      6000B baseline, then 777B patch at offset 500 (window [0, 4096)).
//  13. BounceWriteRmwPreservesSurroundingBytes
//      8192B aligned baseline, then 100B patch at offset 5000
//      (window [4096, 8192)); padding bytes keep the baseline data.
//  14. BounceWriteTruncatesAlignmentPaddingFromFileLength
//      200B at offset 100 into a new file (window [0, 4096)); file size is
//      300B and [0, 100) reads as zeros.
//  15. BounceWriteBeyondEofPreservesExistingBytesAndZeroFill
//      5000B baseline, then 100B at offset 6000 (window [4096, 8192)); the
//      pre-read is clamped to [4096, 5000), [5000, 6000) reads as zeros, file
//      size is 6100B.
//  16. BounceReadAtNonZeroAlignedStartReturnsCorrectBytes
//      3 aligned pages, then a 300B read at offset 5000 (window starts at
//      4096, head_pad 904).
//
// Group C: URI schemes, Coordinator catalog, concurrency, errors
//   3. LustreAndNfsSchemeConfiguration
//   4. BatchExistsAsync
//   8. PosixPathMapperAndKVCacheStoreBackendFactoryIntegration
//   9. ConcurrentMultiSliceWriteLookupAndRead
//  10. ErrorHandlingAndStatsInvariant
// =============================================================================

// -----------------------------------------------------------------------------
// Test 1: DirectZeroCopyMultiSliceReadWrite
//
// Scenario:
//   Writes and reads back 3 contiguous 4KB-aligned slices (4KB + 8KB + 4KB =
//   16KB total) that are pre-registered in [dram: User host_buf] via
//   `RegisterBuffer` (`tds_buffer_register_vaddr(..., TDS_MEM_HOST)`).
//
// Data Path:
//   [dram: User host_buf] <--(libtdsul tds_writev / tds_readv)-->
//   file under `test_dir_` (opened with `O_DIRECT` if the filesystem allows).
//   Because `key.offset` (0), every slice address, and every slice length are
//   multiples of 4096B, `PrepareTransfer()` sets `can_bypass_bounce = true`
//   and no `BounceWindow` is allocated.
//
// Expected Outcome:
//   - `IsDmaAlignedAndRegistered()` returns true for each slice.
//   - Read-back bytes equal the written bytes.
//   - `ops == 2`, `direct_ops == 2`, `bounced_ops == 0`.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, DirectZeroCopyMultiSliceReadWrite) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 4;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_direct", options);

  constexpr size_t kSlice0Size = 4096;
  constexpr size_t kSlice1Size = 8192;
  constexpr size_t kSlice2Size = 4096;
  constexpr size_t kTotalBytes = kSlice0Size + kSlice1Size + kSlice2Size;

  // Step 1: Allocate 4KB-aligned source buffers in [dram: User host_buf] and
  // register each with `libtdsul` (`tds_buffer_register_vaddr`).
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

  ABSL_ASSERT_OK(backend.RegisterBuffer(src0.data(), kSlice0Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(src1.data(), kSlice1Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(src2.data(), kSlice2Size));

  EXPECT_TRUE(backend.IsDmaAlignedAndRegistered(src0.data(), kSlice0Size, 0));
  EXPECT_TRUE(
      backend.IsDmaAlignedAndRegistered(src1.data(), kSlice1Size, kSlice0Size));
  EXPECT_TRUE(backend.IsDmaAlignedAndRegistered(src2.data(), kSlice2Size,
                                                kSlice0Size + kSlice1Size));

  // Step 2: Issue a 3-slice scatter-gather write (`tds_writev`) to storage.
  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = src0.data(), .size = kSlice0Size},
      HostBufferDescriptor{.ptr = src1.data(), .size = kSlice1Size},
      HostBufferDescriptor{.ptr = src2.data(), .size = kSlice2Size},
  };

  BlockKey key;
  key.block_hash = "direct_block_01";
  key.resolved_key = absl::StrCat(test_dir_, "/direct_block_01.bin");
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

  // Step 3: Allocate and register 4KB-aligned destination buffers in
  // [dram: User host_buf], then issue a 3-slice gather read (`tds_readv`).
  AlignedBuffer dst0(kSlice0Size);
  AlignedBuffer dst1(kSlice1Size);
  AlignedBuffer dst2(kSlice2Size);
  std::memset(dst0.data(), 0, kSlice0Size);
  std::memset(dst1.data(), 0, kSlice1Size);
  std::memset(dst2.data(), 0, kSlice2Size);

  ABSL_ASSERT_OK(backend.RegisterBuffer(dst0.data(), kSlice0Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(dst1.data(), kSlice1Size));
  ABSL_ASSERT_OK(backend.RegisterBuffer(dst2.data(), kSlice2Size));

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

  // Step 4: Verify payload integrity and confirm both ops took `direct_ops`.
  EXPECT_EQ(std::memcmp(dst0.data(), expected0.data(), kSlice0Size), 0);
  EXPECT_EQ(std::memcmp(dst1.data(), expected1.data(), kSlice1Size), 0);
  EXPECT_EQ(std::memcmp(dst2.data(), expected2.data(), kSlice2Size), 0);

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.direct_ops, 2);
  EXPECT_EQ(stats.bounced_ops, 0);
  EXPECT_EQ(stats.failed_ops, 0);
  EXPECT_EQ(stats.bytes, 2 * kTotalBytes);

  ABSL_EXPECT_OK(backend.UnregisterBuffer(src0.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(src1.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(src2.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst0.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst1.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst2.data()));
}

// -----------------------------------------------------------------------------
// Test 2: UnalignedBounceBufferMultiSliceReadWrite
//
// Scenario:
//   Writes and reads back 3 unaligned slices (1000B + 513B + 2000B = 3513B)
//   held in ordinary `std::vector<uint8_t>` buffers in [dram: User host_buf].
//
// Data Path:
//   The slice sizes are not multiples of `alignment` (4096), so with
//   `use_direct_io = true` both the write and the read go through a 4096B
//   `BounceWindow` registered via `tds_buffer_register_vaddr(...,
//   TDS_MEM_HOST)`:
//   - Write: copies the 3 slices into the window, `tds_write`s the whole
//     4096B window (to a `.tmp_` file, since this is a whole-file write), then
//     `::ftruncate(fd, 3513)` and renames it into place.
//   - Read: `tds_read`s the window (a short 3513B read at EOF is accepted) and
//     copies the bytes back into the 3 slices.
//
// Expected Outcome:
//   - File size on disk is 3513B, not 4096B.
//   - All 3 slices round-trip identically.
//   - `ops == 2`, `bounced_ops == 2`, `direct_ops == 0`.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, UnalignedBounceBufferMultiSliceReadWrite) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 4;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_bounce", options);

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

  // Verify logical file size matches exact written bytes (not padded 4096).
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

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.bounced_ops, 2);
  EXPECT_EQ(stats.direct_ops, 0);
  EXPECT_EQ(stats.failed_ops, 0);
  EXPECT_EQ(stats.bytes, 2 * kTotalBytes);
}

// -----------------------------------------------------------------------------
// Test 3: LustreAndNfsSchemeConfiguration
//
// Scenario:
//   Verifies property-map parsing (`TdsBackendOptions::FromProperties`),
//   [shared-storage: Lustre/NFS] striping options (`lustre_stripe_size`,
//   `lustre_stripe_count`), URI scheme stripping (`lustre://` and `nfs://`
//   prefixes on `BlockKey::resolved_key`), and rejection of invalid properties
//   (non-power-of-two alignment or negative thread pool size).
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, LustreAndNfsSchemeConfiguration) {
  absl::flat_hash_map<std::string, std::string> lustre_props = {
      {"root_dir", test_dir_},
      {"storage_io_thread_pool_size", "8"},
      {"alignment", "4096"},
      {"use_direct_io", "true"},
      {"lustre_stripe_size", "1048576"},
      {"lustre_stripe_count", "4"},
  };

  absl::StatusOr<TdsBackendOptions> parsed_lustre_opts =
      TdsBackendOptions::FromProperties(lustre_props);
  ABSL_ASSERT_OK(parsed_lustre_opts);
  EXPECT_EQ(parsed_lustre_opts->root_dir, test_dir_);
  EXPECT_EQ(parsed_lustre_opts->storage_io_thread_pool_size, 8);
  EXPECT_EQ(parsed_lustre_opts->alignment, 4096);
  EXPECT_TRUE(parsed_lustre_opts->use_direct_io);
  EXPECT_EQ(parsed_lustre_opts->lustre_stripe_size, 1048576);
  EXPECT_EQ(parsed_lustre_opts->lustre_stripe_count, 4);

  TdsKVBackend lustre_backend("lustre", lustre_props);
  EXPECT_EQ(lustre_backend.name(), "lustre");
  EXPECT_EQ(lustre_backend.alignment(), 4096);
  EXPECT_EQ(lustre_backend.options().lustre_stripe_count, 4);

  absl::flat_hash_map<std::string, std::string> nfs_props = {
      {"root_dir", test_dir_},
      {"storage_io_thread_pool_size", "4"},
      {"alignment", "4096"},
      {"use_direct_io", "false"},
  };
  absl::StatusOr<TdsBackendOptions> parsed_nfs_opts =
      TdsBackendOptions::FromProperties(nfs_props);
  ABSL_ASSERT_OK(parsed_nfs_opts);
  EXPECT_FALSE(parsed_nfs_opts->use_direct_io);

  TdsKVBackend nfs_backend("nfs", *parsed_nfs_opts, nfs_props);
  EXPECT_EQ(nfs_backend.name(), "nfs");

  // Verify URI scheme resolution (writing via `lustre://...` and reading the
  // same underlying file back via `nfs://...`).
  std::vector<uint8_t> data = MakeDeterministicPayload(1024, 99);
  HostBufferDescriptor desc{.ptr = data.data(), .size = data.size()};

  BlockKey lustre_key;
  lustre_key.block_hash = "lustre_hash";
  lustre_key.resolved_key =
      absl::StrCat("lustre://", test_dir_, "/lustre_obj.bin");
  lustre_key.size = data.size();

  absl::Notification done1;
  absl::Status status1;
  lustre_backend.WriteAsync(lustre_key, {desc}, data.size(),
                            [&](absl::Status s) {
                              status1 = std::move(s);
                              done1.Notify();
                            });
  done1.WaitForNotification();
  ABSL_ASSERT_OK(status1);

  std::vector<uint8_t> read_back(1024, 0);
  HostBufferDescriptor read_desc{.ptr = read_back.data(),
                                 .size = read_back.size()};
  absl::Notification done2;
  absl::Status status2;
  nfs_backend.ReadAsync(BlockKey{.block_hash = "lustre_hash",
                                 .resolved_key = absl::StrCat(
                                     "nfs://", test_dir_, "/lustre_obj.bin"),
                                 .size = static_cast<int64_t>(data.size())},
                        {read_desc}, read_back.size(), [&](absl::Status s) {
                          status2 = std::move(s);
                          done2.Notify();
                        });
  done2.WaitForNotification();
  ABSL_ASSERT_OK(status2);
  EXPECT_EQ(read_back, data);

  // Verify invalid property error handling.
  EXPECT_THAT(TdsBackendOptions::FromProperties({{"alignment", "3000"}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TdsBackendOptions::FromProperties(
                  {{"storage_io_thread_pool_size", "-2"}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// -----------------------------------------------------------------------------
// Test 4: BatchExistsAsync
//
// Scenario:
//   Creates 16 `BlockKey` entries, writes only the even-indexed blocks to disk,
//   and calls `BatchExistsAsync()` across all 16 keys.
//
// Expected Outcome:
//   `BatchExistsAsync()` checks both file existence (`::stat`) and minimum byte
//   coverage (`st_size >= key.offset + key.size`), returning `true` for even
//   indices and `false` for odd indices.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, BatchExistsAsync) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 8;
  TdsKVBackend backend("tds_exists", options);

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

// -----------------------------------------------------------------------------
// Test 5: UnalignedNonZeroOffsetPreservesAdjacentBytes
//
// Scenario:
//   Writes a 6000B baseline file at offset 0, then overwrites 777B in place at
//   `offset = 500` (inside the first page, so the bounce window is [0, 4096)).
//   Test 13 `BounceWriteRmwPreservesSurroundingBytes` covers a window that
//   starts at 4096.
//
// Expected Outcome:
//   - The patch write pre-reads [0, 4096) via `tds_read`, copies the 777B
//     patch to [500, 1277) of the window, and writes [0, 4096) back.
//   - Reading the whole file returns the baseline with only [500, 1277)
//     replaced, and the file size stays 6000B.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, UnalignedNonZeroOffsetPreservesAdjacentBytes) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.alignment = kAlignment;
  TdsKVBackend backend("tds_offset", options);

  // First write a 6000-byte baseline block at offset 0.
  std::vector<uint8_t> base_payload = MakeDeterministicPayload(6000, 42);
  HostBufferDescriptor base_desc{.ptr = base_payload.data(),
                                 .size = base_payload.size()};
  BlockKey base_key;
  base_key.block_hash = "offset_block";
  base_key.resolved_key = absl::StrCat(test_dir_, "/offset_block.bin");
  base_key.offset = 0;
  base_key.size = 6000;

  absl::Notification done1;
  absl::Status status1;
  backend.WriteAsync(base_key, {base_desc}, base_payload.size(),
                     [&](absl::Status s) {
                       status1 = std::move(s);
                       done1.Notify();
                     });
  done1.WaitForNotification();
  ABSL_ASSERT_OK(status1);

  // Overwrite 777 bytes at unaligned offset 500 inside the existing file.
  std::vector<uint8_t> patch_payload = MakeDeterministicPayload(777, 84);
  HostBufferDescriptor patch_desc{.ptr = patch_payload.data(),
                                  .size = patch_payload.size()};
  BlockKey patch_key = base_key;
  patch_key.offset = 500;
  patch_key.size = 777;

  absl::Notification done2;
  absl::Status status2;
  backend.WriteAsync(patch_key, {patch_desc}, patch_payload.size(),
                     [&](absl::Status s) {
                       status2 = std::move(s);
                       done2.Notify();
                     });
  done2.WaitForNotification();
  ABSL_ASSERT_OK(status2);

  // File size must remain 6000.
  EXPECT_EQ(std::filesystem::file_size(base_key.resolved_key), 6000);

  // Read back the entire 6000 bytes and verify head, patch, and tail bytes.
  std::vector<uint8_t> read_full(6000, 0);
  HostBufferDescriptor read_desc{.ptr = read_full.data(),
                                 .size = read_full.size()};
  absl::Notification done3;
  absl::Status status3;
  backend.ReadAsync(base_key, {read_desc}, read_full.size(),
                    [&](absl::Status s) {
                      status3 = std::move(s);
                      done3.Notify();
                    });
  done3.WaitForNotification();
  ABSL_ASSERT_OK(status3);

  std::vector<uint8_t> expected = base_payload;
  std::memcpy(expected.data() + 500, patch_payload.data(), 777);
  EXPECT_EQ(read_full, expected);
}

// -----------------------------------------------------------------------------
// Test 6: SubSliceOfRegisteredPoolUsesDirectZeroCopy
//
// Scenario:
//   Registers one 64KB arena in [dram: User host_buf] with `RegisterBuffer`
//   (like a pre-registered KV block pool), then writes two slices located
//   inside it (`pool + 4096`, 8192B and `pool + 32768`, 4096B).
//
// Expected Outcome:
//   - Registering a sub-range of the arena again returns `kAlreadyExists`.
//   - Both slices are found inside the arena's registration, so
//     `PrepareTransfer()` uses the arena's `tds_buffer_handle_t` at offsets
//     4096 and 32768 and the write is bounce-free (`direct_ops == 1`,
//     `bounced_ops == 0`).
//   - `ResetStats()` zeroes the counters.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, SubSliceOfRegisteredPoolUsesDirectZeroCopy) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.alignment = kAlignment;
  TdsKVBackend backend("tds_pool", options);

  constexpr size_t kPoolSize = 64 * 1024;
  AlignedBuffer pool(kPoolSize);
  ASSERT_NE(pool.data(), nullptr);
  std::vector<uint8_t> fill = MakeDeterministicPayload(kPoolSize, 55);
  std::memcpy(pool.data(), fill.data(), kPoolSize);

  ABSL_ASSERT_OK(backend.RegisterBuffer(pool.data(), kPoolSize));

  // Duplicate/overlapping registration should fail with AlreadyExists.
  EXPECT_THAT(backend.RegisterBuffer(pool.data() + 4096, 4096),
              StatusIs(absl::StatusCode::kAlreadyExists));

  // Write two slices that are sub-regions of the single registered pool.
  std::vector<HostBufferDescriptor> slices = {
      HostBufferDescriptor{.ptr = pool.data() + 4096, .size = 8192},
      HostBufferDescriptor{.ptr = pool.data() + 32768, .size = 4096},
  };
  BlockKey key;
  key.block_hash = "pool_block";
  key.resolved_key = absl::StrCat(test_dir_, "/pool_block.bin");
  key.offset = 0;
  key.size = 12288;

  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, slices, 12288, [&](absl::Status s) {
    write_status = std::move(s);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.direct_ops, 1);
  EXPECT_EQ(stats.bounced_ops, 0);

  backend.ResetStats();
  EXPECT_EQ(backend.stats().ops, 0);
  EXPECT_EQ(backend.stats().direct_ops, 0);

  ABSL_EXPECT_OK(backend.UnregisterBuffer(pool.data()));
}

// -----------------------------------------------------------------------------
// Test 7: UnregisteredAlignedSlicesZeroCopyOrBouncePerRequireReg
//
// Scenario:
//   Compares two configurations when given a 4KB-aligned slice in
//   [dram: User host_buf] that was NOT pre-registered with `RegisterBuffer()`:
//   - Case 1 (`require_registration = false`, default): `PrepareTransfer()`
//     wraps the raw pointer in `tds_create_raw_buffer_io` and dispatches on
//     the zero-bounce fast path (`direct_ops == 1`, `bounced_ops == 0`).
//   - Case 2 (`require_registration = true`): `PrepareTransfer()` sets
//     `reg_ok = false`, forcing unregistered slices through `BounceWindow`
//     (`direct_ops == 0`, `bounced_ops == 1`).
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest,
       UnregisteredAlignedSlicesZeroCopyOrBouncePerRequireReg) {
  constexpr size_t kSliceSize = 8192;
  AlignedBuffer src(kSliceSize);
  AlignedBuffer dst(kSliceSize);
  ASSERT_NE(src.data(), nullptr);
  ASSERT_NE(dst.data(), nullptr);
  std::vector<uint8_t> expected = MakeDeterministicPayload(kSliceSize, 77);
  std::memcpy(src.data(), expected.data(), kSliceSize);

  // Case 1: Default `require_registration = false` uses O_DIRECT zero-copy
  // (`direct_ops`) for 4KB-aligned [dram: User host_buf] slices without needing
  // RegisterBuffer().
  {
    TdsBackendOptions default_opts;
    default_opts.root_dir = test_dir_;
    default_opts.alignment = kAlignment;
    default_opts.use_direct_io = true;
    EXPECT_FALSE(default_opts.require_registration);

    TdsKVBackend default_backend("tds_default_unregistered", default_opts);
    BlockKey key{
        .block_hash = "unreg_zero_copy",
        .resolved_key = absl::StrCat(test_dir_, "/unreg_zero_copy.bin"),
        .offset = 0,
        .size = static_cast<int64_t>(kSliceSize),
    };
    HostBufferDescriptor write_slice{.ptr = src.data(), .size = kSliceSize};
    absl::Notification write_done;
    absl::Status write_status;
    default_backend.WriteAsync(key, {write_slice}, kSliceSize,
                               [&](absl::Status s) {
                                 write_status = std::move(s);
                                 write_done.Notify();
                               });
    write_done.WaitForNotification();
    ABSL_ASSERT_OK(write_status);
    EXPECT_EQ(default_backend.stats().direct_ops, 1);
    EXPECT_EQ(default_backend.stats().bounced_ops, 0);
  }

  // Case 2: When `require_registration = true` is configured, unregistered
  // 4KB-aligned slices take the 4KB bounce-buffer fallback (`bounced_ops`)
  // until registered.
  {
    absl::flat_hash_map<std::string, std::string> strict_props = {
        {"root_dir", test_dir_},
        {"require_registration", "true"},
    };
    TdsKVBackend strict_backend("tds_strict_registration", strict_props);
    EXPECT_TRUE(strict_backend.options().require_registration);

    BlockKey key{
        .block_hash = "strict_bounced",
        .resolved_key = absl::StrCat(test_dir_, "/strict_bounced.bin"),
        .offset = 0,
        .size = static_cast<int64_t>(kSliceSize),
    };
    HostBufferDescriptor write_slice{.ptr = src.data(), .size = kSliceSize};
    absl::Notification write_done;
    absl::Status write_status;
    strict_backend.WriteAsync(key, {write_slice}, kSliceSize,
                              [&](absl::Status s) {
                                write_status = std::move(s);
                                write_done.Notify();
                              });
    write_done.WaitForNotification();
    ABSL_ASSERT_OK(write_status);
    EXPECT_EQ(strict_backend.stats().direct_ops, 0);
    EXPECT_EQ(strict_backend.stats().bounced_ops, 1);
  }
}

// -----------------------------------------------------------------------------
// Test 8: PosixPathMapperAndKVCacheStoreBackendFactoryIntegration
//
// Scenario:
//   Integration between the Worker data-plane backend (`TdsKVBackend`) and the
//   Coordinator catalog backend (`PosixKVCacheStoreBackend` created via
//   `KVCacheStoreBackendFactory` for type `"tds"`):
//   1. `PosixPathMapper` hex-encodes a binary `block_hash` (containing NUL
//      `0x00` and `/` `0x2F`) into
//      `<root>/<model>/tp<size>_r<rank>/<l1>/<l2>/<hex>.bin`, here
//      `gemini_ultra/tp8_r3/abc/d0/abcd002fef01.bin`.
//   2. The factory creates a `"tds"` store backend with the configured
//      capacity.
//   3. A rank-0 `TdsKVBackend` writes the block, and the store backend's
//      `Lookup()` reports it as `BlockStatus::SHARED_STORAGE` while a missing
//      hash is not returned.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest,
       PosixPathMapperAndKVCacheStoreBackendFactoryIntegration) {
  absl::flat_hash_map<std::string, std::string> props = {
      {"root_dir", test_dir_},
      {"model_name", "gemini_ultra"},
      {"tp_size", "8"},
      {"tp_rank", "3"},
  };
  TdsKVBackend backend("tds_mapper", props);
  ASSERT_NE(backend.mapper(), nullptr);
  EXPECT_EQ(backend.mapper()->tp_size(), 8);

  // Construct a binary block_hash containing embedded NUL (0x00) and '/' (0x2F)
  // bytes to verify filesystem-safe hex encoding via PosixPathMapper.
  const std::string binary_hash("\xab\xcd\x00\x2f\xef\x01", 6);
  absl::StatusOr<BlockKey> mapped = backend.mapper()->MapKey(binary_hash);
  ABSL_ASSERT_OK(mapped);
  EXPECT_EQ(mapped->block_hash, binary_hash);
  EXPECT_EQ(
      mapped->resolved_key,
      absl::StrCat(test_dir_, "/gemini_ultra/tp8_r3/abc/d0/abcd002fef01.bin"));

  // Verify that "tds" (kTdsBackendName) is registered in
  // KVCacheStoreBackendFactory and creates a working PosixKVCacheStoreBackend
  // backed by TdsKVBackend.
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

  // Write a block at rank 0 via TdsKVBackend and verify that
  // PosixKVCacheStoreBackend::Lookup discovers it via TdsKVBackend's
  // BatchExistsAsync and returns BlockStatus::SHARED_STORAGE.
  TdsKVBackend rank0_writer("tds_rank0", {{"root_dir", test_dir_},
                                          {"model_name", "gemini_ultra"},
                                          {"tp_size", "1"},
                                          {"tp_rank", "0"}});
  AlignedBuffer buf(4096);
  std::memset(buf.data(), 0x5A, 4096);
  absl::Notification write_done;
  absl::Status write_status;
  rank0_writer.WriteAsync(
      BlockKey{.block_hash = binary_hash,
               .resolved_key = "",
               .offset = 0,
               .size = 4096},
      {HostBufferDescriptor{.ptr = buf.data(), .size = 4096}}, 4096,
      [&](absl::Status s) {
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

// -----------------------------------------------------------------------------
// Test 9: ConcurrentMultiSliceWriteLookupAndRead
//
// Scenario:
//   Issues 12 concurrent 3-slice writes (3 x 4KB = 12KB each, aligned) from
//   [dram: User host_buf] on an 8-thread pool, checks that the Coordinator's
//   `Lookup()` reports all 12 blocks as `SHARED_STORAGE`, then issues 12
//   concurrent 3-slice reads back into [dram: User host_buf].
//
// Expected Outcome:
//   - All writes and reads succeed and every read slice matches the written
//     slice.
//   - `direct_ops == 24` (`2 * kNumBlocks`) and `bounced_ops == 0`.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, ConcurrentMultiSliceWriteLookupAndRead) {
  constexpr int kNumBlocks = 12;
  constexpr size_t kSliceBytes = 4096;
  constexpr size_t kSlicesPerBlock = 3;
  constexpr size_t kBlockBytes = kSliceBytes * kSlicesPerBlock;

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
  for (int b = 0; b < kNumBlocks; ++b) {
    block_hashes.push_back(absl::StrCat("concurrent_block_hash_", b));
    for (size_t s = 0; s < kSlicesPerBlock; ++s) {
      auto buf = std::make_unique<AlignedBuffer>(kSliceBytes);
      std::vector<uint8_t> payload = MakeDeterministicPayload(
          kSliceBytes, static_cast<uint8_t>(b * 7 + s + 1));
      std::memcpy(buf->data(), payload.data(), kSliceBytes);
      write_arena.push_back(std::move(buf));
    }
  }

  // Issue all 12 multi-slice block writes concurrently.
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
    BlockKey key{.block_hash = block_hashes[b],
                 .resolved_key = "",
                 .offset = 0,
                 .size = static_cast<int64_t>(kBlockBytes)};
    worker_backend.WriteAsync(
        key, slices, kBlockBytes, [&, b](absl::Status status) {
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

  // Verify Coordinator Lookup finds all 12 blocks in SHARED_STORAGE.
  absl::StatusOr<BlockSliceList> hits = (*store_backend)->Lookup(block_hashes);
  ABSL_ASSERT_OK(hits);
  ASSERT_EQ(hits->size(), kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    EXPECT_EQ((*hits)[b].first, block_hashes[b]);
    EXPECT_EQ((*hits)[b].second.status, BlockStatus::SHARED_STORAGE);
  }

  // Issue all 12 multi-slice block reads concurrently and verify payloads.
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
    BlockKey key{.block_hash = block_hashes[b],
                 .resolved_key = "",
                 .offset = 0,
                 .size = static_cast<int64_t>(kBlockBytes)};
    worker_backend.ReadAsync(
        key, slices, kBlockBytes, [&, b](absl::Status status) {
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

  EXPECT_EQ(worker_backend.stats().direct_ops, 2 * kNumBlocks);
  EXPECT_EQ(worker_backend.stats().bounced_ops, 0);
}

// -----------------------------------------------------------------------------
// Test 10: ErrorHandlingAndStatsInvariant
//
// Scenario:
//   Triggers failure paths (reading a missing file -> `kNotFound`; writing with
//   `total_bytes != sum(slice.size)` -> `kInvalidArgument`; registering invalid
//   null/zero-size buffers) and verifies the telemetry invariant:
//     `stats.ops == stats.direct_ops + stats.bounced_ops + stats.failed_ops`
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, ErrorHandlingAndStatsInvariant) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  TdsKVBackend backend("tds_errors", options);

  AlignedBuffer buf(4096);

  // 1. Reading a non-existent file returns kNotFound and increments failed_ops.
  {
    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(
        BlockKey{.block_hash = "non_existent",
                 .resolved_key = absl::StrCat(test_dir_, "/missing.bin"),
                 .offset = 0,
                 .size = 4096},
        {HostBufferDescriptor{.ptr = buf.data(), .size = 4096}}, 4096,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
  }

  // 2. Writing with mismatched total_bytes returns kInvalidArgument and
  // increments failed_ops.
  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        BlockKey{.block_hash = "mismatch",
                 .resolved_key = absl::StrCat(test_dir_, "/mismatch.bin"),
                 .offset = 0,
                 .size = 8192},
        {HostBufferDescriptor{.ptr = buf.data(), .size = 4096}}, 8192,
        [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3. Register/Unregister invalid arguments.
  EXPECT_THAT(backend.RegisterBuffer(nullptr, 4096),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(backend.RegisterBuffer(buf.data(), 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(backend.UnregisterBuffer(buf.data()),
              StatusIs(absl::StatusCode::kNotFound));

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.failed_ops, 2);
  EXPECT_EQ(stats.direct_ops, 0);
  EXPECT_EQ(stats.bounced_ops, 0);
  EXPECT_EQ(stats.ops, stats.direct_ops + stats.bounced_ops + stats.failed_ops);
}

// -----------------------------------------------------------------------------
// Test 11: LibtdsulDmaBufRegistrationAndSubSliceReadWrite
//
// Scenario (`require_registration = true`):
//   Exercises `RegisterDmaBuf()`, which calls
//   `tds_buffer_register_dmabuf(fd, offset, TDS_MEM_HOST, size, &handle)`.
//   A `memfd_create` fd stands in for a dma_buf fd backed by [dram].
//   1. `RegisterDmaBuf(memfd, offset=128, ...)` is rejected with
//      `kInvalidArgument` because 128 is not a multiple of `alignment`.
//   2. `RegisterDmaBuf(memfd, 0, 16KB)` succeeds; `libtdsul` `mmap`s the fd
//      and `TdsKVBackend` records the mapped `vaddr` range plus the
//      `fd -> base offset` mapping. Registering it again returns
//      `kAlreadyExists`.
//   3. Writes two 4KB slices at `base` and `base + 4096` (by pointer).
//   4. Reads the 8KB block back into `[8192, 16384)` of the same mapping using
//      fd-based descriptors (`{.ptr = nullptr, .fd = memfd, .offset = 8192}`
//      and `.offset = 12288`).
//
// Expected Outcome:
//   - `ResolveSlicePtrLocked()` maps the fd-based descriptors to
//     `base + 8192` / `base + 12288`, and the read-back bytes equal the
//     written bytes.
//   - Both write and read run bounce-free (`direct_ops == 2`,
//     `bounced_ops == 0`).
//   - `UnregisterDmaBuf()` succeeds once, then returns `kNotFound`.
// -----------------------------------------------------------------------------
TEST_F(TdsKVBackendTest, LibtdsulDmaBufRegistrationAndSubSliceReadWrite) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.require_registration = true;
  TdsKVBackend backend("tds_dmabuf", options);

  constexpr size_t kPoolSize = 16384;
  int memfd = ::memfd_create("raiden_tds_dmabuf_test", 0);
  ASSERT_GE(memfd, 0);
  ASSERT_EQ(::ftruncate(memfd, kPoolSize), 0);

  EXPECT_THAT(backend.RegisterDmaBuf(memfd, /*offset=*/128, kPoolSize),
              StatusIs(absl::StatusCode::kInvalidArgument));

  absl::StatusOr<void*> mapped_ptr =
      backend.RegisterDmaBuf(memfd, /*offset=*/0, kPoolSize);
  ABSL_ASSERT_OK(mapped_ptr);
  EXPECT_THAT(backend.RegisterDmaBuf(memfd, /*offset=*/0, kPoolSize),
              StatusIs(absl::StatusCode::kAlreadyExists));
  uint8_t* base = static_cast<uint8_t*>(*mapped_ptr);

  // Sub-slice 0 [0, 4096) and Sub-slice 1 [4096, 8192) hold write payload;
  // Sub-slice 2 [8192, 12288) and Sub-slice 3 [12288, 16384) receive readback.
  for (size_t i = 0; i < 8192; ++i) {
    base[i] = static_cast<uint8_t>((i * 17 + 5) & 0xFF);
  }
  std::memset(base + 8192, 0, 8192);

  EXPECT_TRUE(backend.IsDmaAlignedAndRegistered(base, 4096, 0));
  EXPECT_TRUE(backend.IsDmaAlignedAndRegistered(base + 4096, 4096, 4096));

  BlockKey key{.block_hash = "dmabuf_block_01", .offset = 0, .size = 8192};
  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{.ptr = base, .size = 4096},
      HostBufferDescriptor{.ptr = base + 4096, .size = 4096},
  };
  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(key, write_slices, 8192, [&](absl::Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  // Read back using HostBufferDescriptor{.ptr = nullptr, .fd = memfd,
  // .offset = ...} to verify that TdsKVBackend resolves dma_buf_fd slices
  // directly to the registered TDS_MEM_HOST handle.
  std::vector<HostBufferDescriptor> read_slices = {
      HostBufferDescriptor{
          .ptr = nullptr, .size = 4096, .fd = memfd, .offset = 8192},
      HostBufferDescriptor{
          .ptr = nullptr, .size = 4096, .fd = memfd, .offset = 12288},
  };
  {
    absl::Notification done;
    absl::Status status;
    backend.ReadAsync(key, read_slices, 8192, [&](absl::Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(std::memcmp(base, base + 8192, 8192), 0);
  EXPECT_EQ(backend.stats().direct_ops, 2);
  EXPECT_EQ(backend.stats().bounced_ops, 0);

  ABSL_EXPECT_OK(backend.UnregisterDmaBuf(memfd));
  EXPECT_THAT(backend.UnregisterDmaBuf(memfd),
              StatusIs(absl::StatusCode::kNotFound));
  ::close(memfd);
}

// Probes whether the filesystem backing `dir` accepts `O_DIRECT` at open time.
//
// The test mount varies (a `tmpfs` /tmp accepts `O_DIRECT` on some kernels and
// rejects it on others), so the expectation below is derived from the actual
// mount instead of being hard-coded. That keeps the assertion exact rather than
// merely permissive.
bool FilesystemSupportsDirectIo(const std::string& dir) {
  const std::string probe_path = absl::StrCat(dir, "/o_direct_probe.bin");
  const int fd =
      ::open(probe_path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);
  const bool supported = (fd >= 0);
  if (fd >= 0) ::close(fd);
  ::unlink(probe_path.c_str());
  return supported;
}

// Checks that `TdsStats::o_direct_ops` reports whether [dram: OS Page Cache]
// was actually bypassed, independently of `direct_ops`/`bounced_ops`.
//
// `TdsKVBackend` opens the storage file on [local-storage: NVMe SSD] /
// [shared-storage: Lustre/NFS] with `O_DIRECT` and passes that fd to `libtdsul`
// as `fd://<fd>`. `O_DIRECT` is a property of the open file description, so
// `libtdsul`'s `pread`/`pwrite` on that fd are direct I/O.
//
// Two fallbacks can remove `O_DIRECT` while the transfer still succeeds: the
// buffered re-open in `OpenFile()` and the `EINVAL`-triggered `fcntl(F_SETFL)`
// in `ExecuteTdsIo()`. The data then goes through [dram: OS Page Cache] and a
// kernel memcpy. `o_direct_ops` samples the fd after the transfer so this is
// visible. The expected value depends on whether the test filesystem accepts
// `O_DIRECT` (`FilesystemSupportsDirectIo()`).
TEST_F(TdsKVBackendTest, ODirectIsMeasuredAndReportedSeparatelyFromBouncing) {
  const bool fs_supports_direct_io = FilesystemSupportsDirectIo(test_dir_);

  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 2;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_o_direct", options);

  // A whole number of 4KB pages at file offset 0: eligible for the zero-bounce
  // fast path, so `direct_ops` and `o_direct_ops` should agree.
  constexpr size_t kAlignedBytes = 2 * kAlignment;
  AlignedBuffer src(kAlignedBytes);
  ASSERT_NE(src.data(), nullptr);
  const std::vector<uint8_t> payload =
      MakeDeterministicPayload(kAlignedBytes, 11);
  std::memcpy(src.data(), payload.data(), kAlignedBytes);
  ABSL_ASSERT_OK(backend.RegisterBuffer(src.data(), kAlignedBytes));

  BlockKey aligned_key;
  aligned_key.block_hash = "o_direct_aligned";
  aligned_key.resolved_key = absl::StrCat(test_dir_, "/o_direct_aligned.bin");
  aligned_key.offset = 0;
  aligned_key.size = kAlignedBytes;

  const std::vector<HostBufferDescriptor> aligned_slices = {
      HostBufferDescriptor{.ptr = src.data(), .size = kAlignedBytes},
  };

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(aligned_key, aligned_slices, kAlignedBytes,
                       [&](absl::Status s) {
                         status = std::move(s);
                         done.Notify();
                       });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 1);
  // Bounce dimension: the caller's aligned, registered [dram: User host_buf]
  // slice went straight to `tds_write` with no scratch copy.
  EXPECT_EQ(stats.direct_ops, 1);
  EXPECT_EQ(stats.bounced_ops, 0);
  // Page-cache dimension: independent of the above. On an `O_DIRECT`-capable
  // mount the bypass must be real; on a mount that rejects `O_DIRECT` the
  // backend must admit the fallback rather than over-report a bypass.
  EXPECT_EQ(stats.o_direct_ops, fs_supports_direct_io ? 1 : 0);

  // An unaligned size at an unaligned file offset forces the `BounceWindow`
  // in [dram: User host_buf]. The bounce dimension flips, but the transfer
  // still runs over a descriptor opened the same way, so the page-cache
  // dimension must not flip with it.
  constexpr size_t kUnalignedBytes = 100;
  std::vector<uint8_t> unaligned_src(kUnalignedBytes, 0xAB);

  BlockKey unaligned_key;
  unaligned_key.block_hash = "o_direct_unaligned";
  unaligned_key.resolved_key =
      absl::StrCat(test_dir_, "/o_direct_unaligned.bin");
  unaligned_key.offset = 17;  // Not a multiple of 4096.
  unaligned_key.size = kUnalignedBytes;

  const std::vector<HostBufferDescriptor> unaligned_slices = {
      HostBufferDescriptor{.ptr = unaligned_src.data(),
                           .size = kUnalignedBytes},
  };

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(unaligned_key, unaligned_slices, kUnalignedBytes,
                       [&](absl::Status s) {
                         status = std::move(s);
                         done.Notify();
                       });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.direct_ops, 1);
  EXPECT_EQ(stats.bounced_ops, 1);
  EXPECT_EQ(stats.o_direct_ops, fs_supports_direct_io ? 2 : 0);

  // Telemetry invariant: every operation lands in exactly one bucket of the
  // bounce dimension. `o_direct_ops` is orthogonal and deliberately excluded.
  EXPECT_EQ(stats.ops, stats.direct_ops + stats.bounced_ops + stats.failed_ops);

  // With Direct I/O disabled, nothing bypasses [dram: OS Page Cache], so
  // `o_direct_ops` must stay at zero even though the transfer is bounce-free.
  TdsBackendOptions buffered_options = options;
  buffered_options.use_direct_io = false;
  TdsKVBackend buffered_backend("tds_buffered", buffered_options);
  ABSL_ASSERT_OK(buffered_backend.RegisterBuffer(src.data(), kAlignedBytes));

  BlockKey buffered_key = aligned_key;
  buffered_key.block_hash = "buffered_aligned";
  buffered_key.resolved_key = absl::StrCat(test_dir_, "/buffered_aligned.bin");

  {
    absl::Notification done;
    absl::Status status;
    buffered_backend.WriteAsync(buffered_key, aligned_slices, kAlignedBytes,
                                [&](absl::Status s) {
                                  status = std::move(s);
                                  done.Notify();
                                });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  const TdsStats buffered_stats = buffered_backend.stats();
  EXPECT_EQ(buffered_stats.ops, 1);
  // Bounce dimension: buffered mode allocates no scratch, so it is bounce-free
  // exactly like the Direct I/O fast path.
  EXPECT_EQ(buffered_stats.direct_ops, 1);
  EXPECT_EQ(buffered_stats.bounced_ops, 0);
  // Page-cache dimension: buffered mode is precisely the case that does NOT
  // bypass [dram: OS Page Cache], and must never be reported as if it did.
  EXPECT_EQ(buffered_stats.o_direct_ops, 0);

  ABSL_EXPECT_OK(buffered_backend.UnregisterBuffer(src.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(src.data()));
}

// Reads `size` bytes at `offset` straight off [local-storage: NVMe SSD] /
// [shared-storage: Lustre/NFS] with a plain buffered descriptor.
//
// Ground truth for the tests below is deliberately taken with raw POSIX
// `::pread` rather than `TdsKVBackend::ReadAsync`: the read path shares the
// same `BounceWindow` machinery as the write path, so verifying a write with a
// read could let a symmetric bug cancel itself out and still pass.
std::vector<uint8_t> ReadFileBytes(const std::string& path, int64_t offset,
                                   size_t size) {
  std::vector<uint8_t> out(size, 0);
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return {};
  size_t done = 0;
  while (done < size) {
    const ssize_t n = ::pread(fd, out.data() + done, size - done,
                              offset + static_cast<int64_t>(done));
    if (n <= 0) break;
    done += static_cast<size_t>(n);
  }
  ::close(fd);
  out.resize(done);
  return out;
}

int64_t FileSizeOrNegative(const std::string& path) {
  struct stat st = {};
  if (::stat(path.c_str(), &st) != 0) return -1;
  return st.st_size;
}

// Regression test for data loss during an unaligned in-place partial write.
//
// `O_DIRECT` requires the file offset and length to be multiples of the
// configured alignment (4096 here), so a write that starts or ends mid-page
// cannot be handed to `libtdsul` as-is. `ExecuteWrite()` instead allocates an
// aligned `BounceWindow` in [dram: User host_buf], and that window is *wider*
// than the caller's payload: it covers padding bytes at the head and tail that
// the caller never supplied.
//
// Those padding bytes already hold committed data on storage. If the window is
// written out without first reading them back, the write zero-fills its own
// alignment padding and overwrites neighbouring bytes of the same file. The
// read-modify-write pre-read in `ExecuteWrite()` prevents that, and this test
// checks it.
//
// This overlaps with `UnalignedNonZeroOffsetPreservesAdjacentBytes` above but
// is deliberately kept separate, because it pins down three things that test
// cannot:
//   1. The bounce window here starts at 4096, not 0. A pre-read that passed a
//      hardcoded offset of 0 instead of `win.aligned_start` would restore the
//      wrong page and still satisfy a window anchored at 0.
//   2. It asserts `direct_ops`/`bounced_ops` directly, so the test fails loudly
//      if a future alignment change quietly routes this case onto the
//      zero-bounce fast path and stops covering the RMW code at all.
//   3. It verifies with raw `::pread` rather than `ReadAsync`.
TEST_F(TdsKVBackendTest, BounceWriteRmwPreservesSurroundingBytes) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 2;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_rmw", options);

  const std::string path = absl::StrCat(test_dir_, "/rmw_in_place.bin");

  // Step 1: lay down a fully aligned 2-page baseline via the zero-bounce fast
  // path, so every byte on storage has a known value.
  constexpr size_t kBaselineBytes = 2 * kAlignment;  // 8192
  AlignedBuffer baseline_buf(kBaselineBytes);
  ASSERT_NE(baseline_buf.data(), nullptr);
  const std::vector<uint8_t> baseline =
      MakeDeterministicPayload(kBaselineBytes, 23);
  std::memcpy(baseline_buf.data(), baseline.data(), kBaselineBytes);
  ABSL_ASSERT_OK(backend.RegisterBuffer(baseline_buf.data(), kBaselineBytes));

  BlockKey baseline_key;
  baseline_key.block_hash = "rmw_baseline";
  baseline_key.resolved_key = path;
  baseline_key.offset = 0;
  baseline_key.size = kBaselineBytes;

  const std::vector<HostBufferDescriptor> baseline_slices = {
      HostBufferDescriptor{.ptr = baseline_buf.data(), .size = kBaselineBytes},
  };

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(baseline_key, baseline_slices, kBaselineBytes,
                       [&](absl::Status s) {
                         status = std::move(s);
                         done.Notify();
                       });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  ASSERT_EQ(FileSizeOrNegative(path), static_cast<int64_t>(kBaselineBytes));
  EXPECT_EQ(backend.stats().direct_ops, 1);
  EXPECT_EQ(backend.stats().bounced_ops, 0);

  // Step 2: overwrite a small unaligned range sitting entirely inside the
  // second page. The patch buffer is neither 4KB-aligned nor registered, and
  // the file offset is not a multiple of 4096, so this is forced onto the
  // `BounceWindow` path.
  //
  // The resulting window is [4096, 8192): 904 bytes of head padding before the
  // patch and 3092 bytes of tail padding after it. All 3996 padding bytes hold
  // baseline data that must survive.
  constexpr int64_t kPatchOffset = 5000;
  constexpr size_t kPatchBytes = 100;
  static_assert(kPatchOffset % kAlignment != 0,
                "Patch offset must be unaligned to exercise the bounce path");
  static_assert(kPatchOffset + kPatchBytes < kBaselineBytes,
                "Patch must land strictly inside the existing file");

  std::vector<uint8_t> patch(kPatchBytes, 0);
  for (size_t i = 0; i < kPatchBytes; ++i) {
    patch[i] = static_cast<uint8_t>(0xC0 + (i % 16));
  }

  BlockKey patch_key;
  patch_key.block_hash = "rmw_patch";
  patch_key.resolved_key = path;
  patch_key.offset = kPatchOffset;
  patch_key.size = kPatchBytes;

  const std::vector<HostBufferDescriptor> patch_slices = {
      HostBufferDescriptor{.ptr = patch.data(), .size = kPatchBytes},
  };

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(patch_key, patch_slices, kPatchBytes,
                       [&](absl::Status s) {
                         status = std::move(s);
                         done.Notify();
                       });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(backend.stats().ops, 2);
  EXPECT_EQ(backend.stats().direct_ops, 1);
  EXPECT_EQ(backend.stats().bounced_ops, 1);

  // An in-place patch must never resize the file. A missing `target_file_size`
  // clamp would grow it to the 8192 window end; a spurious `::ftruncate()`
  // would shrink it to 5100.
  EXPECT_EQ(FileSizeOrNegative(path), static_cast<int64_t>(kBaselineBytes));

  const std::vector<uint8_t> actual = ReadFileBytes(path, 0, kBaselineBytes);
  ASSERT_EQ(actual.size(), kBaselineBytes);

  // Bytes before the bounce window entirely (page 0) were never touched.
  EXPECT_EQ(std::memcmp(actual.data(), baseline.data(), kAlignment), 0)
      << "Page 0 lies outside the bounce window and must be untouched";

  // Head padding inside the window: [4096, 5000). Only the RMW pre-read can
  // keep these alive.
  EXPECT_EQ(
      std::memcmp(actual.data() + kAlignment, baseline.data() + kAlignment,
                  kPatchOffset - kAlignment),
      0)
      << "Bounce-window head padding was clobbered; RMW pre-read is missing";

  // The patched range itself.
  EXPECT_EQ(
      std::memcmp(actual.data() + kPatchOffset, patch.data(), kPatchBytes), 0);

  // Tail padding inside the window: [5100, 8192).
  EXPECT_EQ(std::memcmp(actual.data() + kPatchOffset + kPatchBytes,
                        baseline.data() + kPatchOffset + kPatchBytes,
                        kBaselineBytes - kPatchOffset - kPatchBytes),
            0)
      << "Bounce-window tail padding was clobbered; RMW pre-read is missing";

  ABSL_EXPECT_OK(backend.UnregisterBuffer(baseline_buf.data()));
}

// Regression test for alignment padding leaking into the published file length.
//
// The `BounceWindow` always writes whole `alignment` units (4096 here), so a
// 200-byte payload at offset 100 issues a 4096-byte `tds_write`. Without the
// `::ftruncate()` in `ExecuteWrite()` the file would be 4096 bytes long, with
// 3796 trailing zero bytes the caller never wrote, even though the payload
// bytes themselves are correct.
TEST_F(TdsKVBackendTest, BounceWriteTruncatesAlignmentPaddingFromFileLength) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 2;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_truncate", options);

  const std::string path = absl::StrCat(test_dir_, "/rmw_padding.bin");

  // Unaligned offset + unaligned size against a file that does not yet exist.
  // The window is [0, 4096) and there is nothing to pre-read, so this isolates
  // the truncate behaviour from the RMW behaviour tested above.
  constexpr int64_t kOffset = 100;
  constexpr size_t kBytes = 200;
  std::vector<uint8_t> src(kBytes, 0x5A);

  BlockKey key;
  key.block_hash = "padding_block";
  key.resolved_key = path;
  key.offset = kOffset;
  key.size = kBytes;

  const std::vector<HostBufferDescriptor> slices = {
      HostBufferDescriptor{.ptr = src.data(), .size = kBytes},
  };

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(key, slices, kBytes, [&](absl::Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(backend.stats().bounced_ops, 1);
  EXPECT_EQ(backend.stats().direct_ops, 0);

  // The file must end exactly at offset+size, not at the 4096 window end.
  EXPECT_EQ(FileSizeOrNegative(path), kOffset + static_cast<int64_t>(kBytes))
      << "Bounce-window alignment padding leaked into the published file size";

  // The head gap [0, 100) was never written by the caller and is defined to
  // read back as zeros (the `BounceWindow` is memset before use).
  const std::vector<uint8_t> head = ReadFileBytes(path, 0, kOffset);
  ASSERT_EQ(head.size(), static_cast<size_t>(kOffset));
  EXPECT_EQ(head, std::vector<uint8_t>(kOffset, 0));

  const std::vector<uint8_t> body = ReadFileBytes(path, kOffset, kBytes);
  ASSERT_EQ(body.size(), kBytes);
  EXPECT_EQ(body, src);
}

// Regression test for a bounce write that starts past the current end of file.
//
// This is the case where the RMW pre-read can only recover *part* of its
// window. The file ends at 5000 but the window spans [4096, 8192), so the
// pre-read is clamped to the 904 bytes that actually exist. Everything after
// that is a hole the caller never wrote and must read back as zeros, while the
// 904 recovered bytes must still match the baseline on
// [local-storage: NVMe SSD] / [shared-storage: Lustre/NFS].
//
// This pins down the `existing_in_window` clamp in `ExecuteWrite()`. A clamp
// that asked for the full `aligned_size` would turn the normal short read at
// EOF into a spurious failure; a clamp that asked for nothing would drop the
// 904 live bytes and corrupt them.
TEST_F(TdsKVBackendTest,
       BounceWriteBeyondEofPreservesExistingBytesAndZeroFill) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 2;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_eof", options);

  const std::string path = absl::StrCat(test_dir_, "/beyond_eof.bin");

  // Baseline of 5000 bytes: an unaligned length, so this write also lands on
  // the bounce path and is published via the `.tmp` + `::rename()` route.
  constexpr size_t kBaselineBytes = 5000;
  std::vector<uint8_t> baseline = MakeDeterministicPayload(kBaselineBytes, 71);

  BlockKey baseline_key;
  baseline_key.block_hash = "eof_baseline";
  baseline_key.resolved_key = path;
  baseline_key.offset = 0;
  baseline_key.size = kBaselineBytes;

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        baseline_key,
        {HostBufferDescriptor{.ptr = baseline.data(), .size = kBaselineBytes}},
        kBaselineBytes, [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  ASSERT_EQ(FileSizeOrNegative(path), static_cast<int64_t>(kBaselineBytes));

  // Now write 100 bytes at offset 6000 -- 1000 bytes past EOF, and unaligned.
  // Window is [4096, 8192) with head_pad 1904; only [4096, 5000) exists.
  constexpr int64_t kPatchOffset = 6000;
  constexpr size_t kPatchBytes = 100;
  std::vector<uint8_t> patch(kPatchBytes, 0x3C);

  BlockKey patch_key;
  patch_key.block_hash = "eof_patch";
  patch_key.resolved_key = path;
  patch_key.offset = kPatchOffset;
  patch_key.size = kPatchBytes;

  {
    absl::Notification done;
    absl::Status status;
    backend.WriteAsync(
        patch_key,
        {HostBufferDescriptor{.ptr = patch.data(), .size = kPatchBytes}},
        kPatchBytes, [&](absl::Status s) {
          status = std::move(s);
          done.Notify();
        });
    done.WaitForNotification();
    ABSL_ASSERT_OK(status);
  }

  EXPECT_EQ(backend.stats().bounced_ops, 2);
  EXPECT_EQ(backend.stats().direct_ops, 0);

  // The file must now end exactly at 6100, not at the 8192 window end.
  EXPECT_EQ(FileSizeOrNegative(path),
            kPatchOffset + static_cast<int64_t>(kPatchBytes));

  const std::vector<uint8_t> actual =
      ReadFileBytes(path, 0, static_cast<size_t>(kPatchOffset) + kPatchBytes);
  ASSERT_EQ(actual.size(), static_cast<size_t>(kPatchOffset) + kPatchBytes);

  // [0, 5000): the entire original baseline survives. Bytes [4096, 5000) are
  // the ones that had to come back through the partial pre-read.
  EXPECT_EQ(std::memcmp(actual.data(), baseline.data(), kBaselineBytes), 0)
      << "Baseline bytes were lost across the beyond-EOF bounce write";

  // [5000, 6000): the hole between old EOF and the new payload reads as zeros.
  const std::vector<uint8_t> gap(actual.begin() + kBaselineBytes,
                                 actual.begin() + kPatchOffset);
  EXPECT_EQ(gap, std::vector<uint8_t>(kPatchOffset - kBaselineBytes, 0))
      << "Gap between old EOF and the new payload must be zero-filled";

  // [6000, 6100): the payload itself.
  EXPECT_EQ(
      std::memcmp(actual.data() + kPatchOffset, patch.data(), kPatchBytes), 0);
}

// Regression test for the read-side bounce window anchor.
//
// `ExecuteRead()` issues its `tds_read` at `win.aligned_start` and then copies
// out of the window starting at `win.head_pad`. If `aligned_start` were wrong
// (e.g. always 0), the copy offset would still look self-consistent but the
// bytes would come from the wrong page.
//
// Every other unaligned read in this file is at `key.offset == 0`, where
// `aligned_start` is 0 and such a bug is invisible. This test reads 300B at
// offset 5000 (`aligned_start == 4096`, `head_pad == 904`), so the read must
// start at the second page of the file for the assertion to hold.
TEST_F(TdsKVBackendTest, BounceReadAtNonZeroAlignedStartReturnsCorrectBytes) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 2;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_bounce_read", options);

  // Three aligned pages written through the zero-bounce fast path, so the
  // on-storage contents are known exactly.
  constexpr size_t kBaselineBytes = 3 * kAlignment;  // 12288
  AlignedBuffer src(kBaselineBytes);
  ASSERT_NE(src.data(), nullptr);
  const std::vector<uint8_t> baseline =
      MakeDeterministicPayload(kBaselineBytes, 97);
  std::memcpy(src.data(), baseline.data(), kBaselineBytes);
  ABSL_ASSERT_OK(backend.RegisterBuffer(src.data(), kBaselineBytes));

  BlockKey write_key;
  write_key.block_hash = "bounce_read_baseline";
  write_key.resolved_key = absl::StrCat(test_dir_, "/bounce_read.bin");
  write_key.offset = 0;
  write_key.size = kBaselineBytes;

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
  ASSERT_EQ(backend.stats().direct_ops, 1);

  // Read 300 bytes starting at unaligned offset 5000. The destination is a
  // plain [dram: User host_buf] vector -- neither 4KB-aligned nor registered --
  // which together with the unaligned offset forces the `BounceWindow` path.
  constexpr int64_t kReadOffset = 5000;
  constexpr size_t kReadBytes = 300;
  static_assert(kReadOffset % kAlignment != 0,
                "Read offset must be unaligned to exercise the bounce path");
  static_assert(kReadOffset / kAlignment == 1,
                "Read must land on the second page so aligned_start is 4096");

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

  EXPECT_EQ(backend.stats().bounced_ops, 1);
  EXPECT_EQ(backend.stats().direct_ops, 1);

  // The bytes must come from file offset 5000, not from offset 904 (which is
  // what a window anchored at 0 would hand back after applying head_pad).
  EXPECT_EQ(std::memcmp(dst.data(), baseline.data() + kReadOffset, kReadBytes),
            0)
      << "Bounce read returned the wrong region of the file";

  ABSL_EXPECT_OK(backend.UnregisterBuffer(src.data()));
}

// Regression test for scatter-gather transfers that exceed the kernel's iovec
// limit.
//
// `tds_readv` / `tds_writev` bottom out in `preadv` / `pwritev`, which accept
// at most `UIO_MAXIOV` (1024) iovecs per call. `ExecuteTdsIo()` therefore
// batches: it submits up to 1024 descriptors, then advances `current_offset` by
// the bytes transferred and submits the next group.
//
// Every other test in this file uses at most a handful of slices, so that loop
// has only ever run a single iteration and the offset advance between batches
// was never exercised. A failure to carry `current_offset` forward would
// rewrite the first batch instead of appending the second -- silent corruption
// that appears only once a caller passes more than 1024 slices.
//
// Using `UIO_MAXIOV + 1` pages keeps the test exactly one page past the
// boundary, which is the cheapest input that forces a second batch.
TEST_F(TdsKVBackendTest, ScatterGatherCrossesIovLimitInMultipleBatches) {
  TdsBackendOptions options;
  options.root_dir = test_dir_;
  options.storage_io_thread_pool_size = 2;
  options.alignment = kAlignment;
  options.use_direct_io = true;

  TdsKVBackend backend("tds_iov", options);

  constexpr size_t kPages = static_cast<size_t>(UIO_MAXIOV) + 1;
  constexpr size_t kTotalBytes = kPages * kAlignment;

  // One contiguous, page-aligned pool in [dram: User host_buf], registered once
  // with `tds_buffer_register_vaddr(..., TDS_MEM_HOST)`. Each slice below is a
  // distinct page of this pool, so every slice is 4KB-aligned in both address
  // and length and the whole transfer stays on the zero-bounce fast path.
  AlignedBuffer src(kTotalBytes);
  AlignedBuffer dst(kTotalBytes);
  ASSERT_NE(src.data(), nullptr);
  ASSERT_NE(dst.data(), nullptr);

  // The payload varies per page, so a batch written at the wrong file offset
  // produces a detectable mismatch rather than coincidentally matching.
  const std::vector<uint8_t> payload =
      MakeDeterministicPayload(kTotalBytes, 53);
  std::memcpy(src.data(), payload.data(), kTotalBytes);
  std::memset(dst.data(), 0, kTotalBytes);

  ABSL_ASSERT_OK(backend.RegisterBuffer(src.data(), kTotalBytes));
  ABSL_ASSERT_OK(backend.RegisterBuffer(dst.data(), kTotalBytes));

  std::vector<HostBufferDescriptor> write_slices;
  std::vector<HostBufferDescriptor> read_slices;
  write_slices.reserve(kPages);
  read_slices.reserve(kPages);
  for (size_t i = 0; i < kPages; ++i) {
    write_slices.push_back(HostBufferDescriptor{
        .ptr = src.data() + i * kAlignment, .size = kAlignment});
    read_slices.push_back(HostBufferDescriptor{
        .ptr = dst.data() + i * kAlignment, .size = kAlignment});
  }
  ASSERT_GT(write_slices.size(), static_cast<size_t>(UIO_MAXIOV))
      << "Test must exceed UIO_MAXIOV to force a second batch";

  BlockKey key;
  key.block_hash = "iov_block";
  key.resolved_key = absl::StrCat(test_dir_, "/iov_block.bin");
  key.offset = 0;
  key.size = kTotalBytes;

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

  // Every page must have reached [local-storage: NVMe SSD] /
  // [shared-storage: Lustre/NFS]; a dropped or rewritten batch shows up here.
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

  // Both the write and the read stayed bounce-free despite spanning batches.
  const TdsStats stats = backend.stats();
  EXPECT_EQ(stats.ops, 2);
  EXPECT_EQ(stats.direct_ops, 2);
  EXPECT_EQ(stats.bounced_ops, 0);
  EXPECT_EQ(stats.failed_ops, 0);

  EXPECT_EQ(std::memcmp(dst.data(), payload.data(), kTotalBytes), 0)
      << "Round trip across the UIO_MAXIOV batch boundary lost or misplaced "
         "data";

  // Pinpoint the boundary itself so a regression reports the interesting page
  // rather than just "the 4MB blob differs".
  const size_t boundary = static_cast<size_t>(UIO_MAXIOV) * kAlignment;
  EXPECT_EQ(
      std::memcmp(dst.data() + boundary, payload.data() + boundary, kAlignment),
      0)
      << "The page immediately after the first batch is wrong";

  ABSL_EXPECT_OK(backend.UnregisterBuffer(src.data()));
  ABSL_EXPECT_OK(backend.UnregisterBuffer(dst.data()));
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

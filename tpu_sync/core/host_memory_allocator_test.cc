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

#include "tpu_sync/core/host_memory_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"
#include "tpu_sync/core/utils.h"

namespace tpu_raiden {
namespace {

// The payload mappings are detached from the header page, so the tests read
// the header straight from the segment file.
SharedMemoryHeader ReadSegmentHeader(const std::string& shm_key) {
  const std::string name = SharedMemoryHostMemoryAllocator::ComposeSegmentName(
      shm_key, std::nullopt);
  int fd = shm_open(name.c_str(), O_RDONLY, 0666);
  CHECK_GE(fd, 0) << "shm_open failed for " << name;
  void* ptr =
      mmap(nullptr, sizeof(SharedMemoryHeader), PROT_READ, MAP_SHARED, fd, 0);
  CHECK_NE(ptr, MAP_FAILED);
  SharedMemoryHeader header = *static_cast<SharedMemoryHeader*>(ptr);
  munmap(ptr, sizeof(SharedMemoryHeader));
  close(fd);
  return header;
}

// Creates a segment file holding only the given header, as a predecessor
// that died before its first allocation (or an older-version run) leaves it.
void WriteBareSegment(const std::string& shm_key,
                      const SharedMemoryHeader& header) {
  const std::string name = SharedMemoryHostMemoryAllocator::ComposeSegmentName(
      shm_key, std::nullopt);
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
  CHECK_GE(fd, 0) << "shm_open failed for " << name;
  CHECK_EQ(
      posix_fallocate(fd, 0, SharedMemoryHostMemoryAllocator::kPayloadOffset),
      0);
  void* ptr = mmap(nullptr, sizeof(SharedMemoryHeader), PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
  CHECK_NE(ptr, MAP_FAILED);
  *static_cast<SharedMemoryHeader*>(ptr) = header;
  munmap(ptr, sizeof(SharedMemoryHeader));
  close(fd);
}

TEST(HostMemoryAllocatorTest, FallbackAllocationWithoutClient) {
  TF_ASSERT_OK_AND_ASSIGN(auto allocator, HostMemoryAllocator::Create(nullptr));

  // Allocate 1024 bytes
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(1024));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, 1024);
  EXPECT_NE(alloc.owner, nullptr);

  // Verify 64-byte alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(alloc.ptr) % 64, 0);

  // Write and read back to verify it's usable memory
  std::memset(alloc.ptr, 0xAB, 1024);
  for (size_t i = 0; i < 1024; ++i) {
    EXPECT_EQ(alloc.ptr[i], 0xAB);
  }

  // Zero allocation should work and return a nullptr or empty alloc safely
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation zero_alloc,
                          allocator->Allocate(0));
  EXPECT_EQ(zero_alloc.ptr, nullptr);
  EXPECT_EQ(zero_alloc.size, 0);
}

TEST(HostMemoryAllocatorTest, AllocationWithTpuClient) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * manager,
                          TpuPjrtManager::GetDefault());
  ASSERT_NE(manager->client(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(manager->client()));

  // Allocate 4096 bytes
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(4096));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, 4096);
  EXPECT_NE(alloc.owner, nullptr);

  // Verify 64-byte alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(alloc.ptr) % 64, 0);

  // Write and read back to verify it's usable memory
  std::memset(alloc.ptr, 0xCD, 4096);
  for (size_t i = 0; i < 4096; ++i) {
    EXPECT_EQ(alloc.ptr[i], 0xCD);
  }
}

TEST(HostMemoryAllocatorTest, SharedMemoryColdAndWarmBoot) {
  std::string shm_key = "/test_raiden_shm_key_" + std::to_string(getpid());

  SharedMemoryHeader schema1 = {};
  schema1.magic = 0x52414944454E;
  schema1.version = 1;
  absl::SNPrintF(schema1.model_uid, sizeof(schema1.model_uid), "test_model_v1");
  schema1.num_blocks = 128;
  schema1.block_size = 4096;
  schema1.num_heads = 32;
  schema1.head_dim = 128;
  schema1.itemsize = 2;

  {
    shm_unlink(shm_key.c_str());

    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator1,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema1));

    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc1,
                            allocator1->Allocate(1024));
    EXPECT_NE(alloc1.ptr, nullptr);
    EXPECT_EQ(alloc1.size, 1024);

    std::memset(alloc1.ptr, 0x55, 1024);

    SharedMemoryHeader header1 = ReadSegmentHeader(shm_key);
    EXPECT_EQ(header1.reference_count, 1);
    EXPECT_EQ(header1.version, 1);
    EXPECT_STREQ(header1.model_uid, "test_model_v1");

    {
      TF_ASSERT_OK_AND_ASSIGN(
          auto allocator2,
          SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema1));

      TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc2,
                              allocator2->Allocate(1024));
      EXPECT_NE(alloc2.ptr, nullptr);
      EXPECT_EQ(alloc2.size, 1024);

      for (size_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(alloc2.ptr[i], 0x55);
      }

      EXPECT_EQ(ReadSegmentHeader(shm_key).reference_count, 2);
    }

    EXPECT_EQ(ReadSegmentHeader(shm_key).reference_count, 1);

    {
      SharedMemoryHeader schema2 = schema1;
      schema2.version = 2;
      absl::SNPrintF(schema2.model_uid, sizeof(schema2.model_uid),
                     "test_model_v2");

      TF_ASSERT_OK_AND_ASSIGN(
          auto allocator3,
          SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema2));

      TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc3,
                              allocator3->Allocate(1024));
      EXPECT_NE(alloc3.ptr, nullptr);

      SharedMemoryHeader header3 = ReadSegmentHeader(shm_key);
      EXPECT_EQ(header3.reference_count, 1);
      EXPECT_EQ(header3.version, 2);
      EXPECT_STREQ(header3.model_uid, "test_model_v2");

      for (size_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(alloc3.ptr[i], 0);
      }
    }
  }

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryAllocationsAreDistinctRegions) {
  std::string shm_key = "/test_raiden_shm_multi_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "multi_model");

  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));

  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc1,
                          allocator->Allocate(1024));
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc2,
                          allocator->Allocate(1024));
  ASSERT_NE(alloc1.ptr, nullptr);
  ASSERT_NE(alloc2.ptr, nullptr);
  EXPECT_NE(alloc1.ptr, alloc2.ptr);

  // Writing one allocation must not show through the other.
  std::memset(alloc1.ptr, 0x11, 1024);
  std::memset(alloc2.ptr, 0x22, 1024);
  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(alloc1.ptr[i], 0x11);
    ASSERT_EQ(alloc2.ptr[i], 0x22);
  }

  // Each allocation advanced the recorded payload by a full page.
  EXPECT_EQ(ReadSegmentHeader(shm_key).total_payload_bytes, 2 * 4096);

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryWarmBootPreservesEachAllocation) {
  std::string shm_key = "/test_raiden_shm_warm_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "warm_model");

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc1,
                            allocator->Allocate(1024));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc2,
                            allocator->Allocate(1024));
    std::memset(alloc1.ptr, 0x11, 1024);
    std::memset(alloc2.ptr, 0x22, 1024);
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc1,
                            allocator->Allocate(1024));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc2,
                            allocator->Allocate(1024));
    // The allocations replay in order, each recovering its own bytes.
    for (size_t i = 0; i < 1024; ++i) {
      ASSERT_EQ(alloc1.ptr[i], 0x11);
      ASSERT_EQ(alloc2.ptr[i], 0x22);
    }
  }

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryWarmBootBeyondRecordedPayload) {
  std::string shm_key = "/test_raiden_shm_beyond_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "beyond_model");

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(1024));
    std::memset(alloc.ptr, 0x77, 1024);
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc1,
                            allocator->Allocate(1024));
    // Allocating past what the previous run recorded degrades that region
    // to zeroed memory but keeps the earlier region's recovered bytes.
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc2,
                            allocator->Allocate(1024));
    for (size_t i = 0; i < 1024; ++i) {
      ASSERT_EQ(alloc1.ptr[i], 0x77);
      ASSERT_EQ(alloc2.ptr[i], 0);
    }
    EXPECT_EQ(ReadSegmentHeader(shm_key).total_payload_bytes, 2 * 4096);
  }

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryBlockSizeChangeColdStarts) {
  std::string shm_key = "/test_raiden_shm_bpb_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "block_model");
  schema.num_blocks = 4;

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(8192));
    std::memset(alloc.ptr, 0x77, 8192);
  }
  EXPECT_EQ(ReadSegmentHeader(shm_key).block_size, 8192 / 4);

  {
    // Same block count and identity, but halved bytes per block: the halved
    // request divides the recorded payload evenly, so only the recorded
    // block size tells the layouts apart. The segment must cold-start.
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(4096));
    for (size_t i = 0; i < 4096; ++i) {
      ASSERT_EQ(alloc.ptr[i], 0);
    }
    SharedMemoryHeader header = ReadSegmentHeader(shm_key);
    EXPECT_EQ(header.block_size, 4096 / 4);
    EXPECT_EQ(header.total_payload_bytes, 4096);
  }

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryHeaderOnlySegmentServesZeroed) {
  std::string shm_key = "/test_raiden_shm_bare_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "crashed_model");
  // A predecessor that died after creating the header but before its first
  // allocation: zero recorded payload.
  WriteBareSegment(shm_key, schema);

  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(1024));
  ASSERT_NE(alloc.ptr, nullptr);
  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(alloc.ptr[i], 0);
  }
  SharedMemoryHeader header = ReadSegmentHeader(shm_key);
  EXPECT_EQ(header.total_payload_bytes, 4096);
  EXPECT_EQ(header.reference_count, 1);

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryVersionOneSegmentColdStarts) {
  std::string shm_key = "/test_raiden_shm_v1_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader v1 = {};
  v1.version = 1;
  absl::SNPrintF(v1.model_uid, sizeof(v1.model_uid), "upgrade_model");
  v1.total_payload_bytes = 512;
  WriteBareSegment(shm_key, v1);

  // The production attach side: a default-constructed schema carries the
  // current version, so a version-1 segment reformats once on upgrade.
  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "upgrade_model");
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(1024));
  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(alloc.ptr[i], 0);
  }
  SharedMemoryHeader header = ReadSegmentHeader(shm_key);
  EXPECT_EQ(header.version, 2);
  EXPECT_EQ(header.reference_count, 1);
  EXPECT_EQ(header.total_payload_bytes, 4096);

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryRequestSizeChangeColdStarts) {
  std::string shm_key = "/test_raiden_shm_resize_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "resize_model");

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(1024));
    std::memset(alloc.ptr, 0x77, 1024);
  }

  {
    // A different request size means a different region layout; the recorded
    // payload no longer divides into it, so the segment cold-starts.
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(8192));
    for (size_t i = 0; i < 8192; ++i) {
      ASSERT_EQ(alloc.ptr[i], 0);
    }
    EXPECT_EQ(ReadSegmentHeader(shm_key).total_payload_bytes, 8192);
  }

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, ScopedShmLockSerializesHolders) {
  const std::string name = "/test_raiden_shm_lock_" + std::to_string(getpid());

  std::optional<ScopedShmLock> first;
  {
    TF_ASSERT_OK_AND_ASSIGN(ScopedShmLock lock, ScopedShmLock::Acquire(name));
    first.emplace(std::move(lock));
  }

  std::atomic<bool> second_acquired{false};
  std::thread contender([&] {
    TF_ASSERT_OK_AND_ASSIGN(ScopedShmLock lock, ScopedShmLock::Acquire(name));
    second_acquired.store(true);
  });

  // The contender must still be blocked while the first hold is live. A
  // sleep cannot prove blocking, but a broken lock makes this fail with high
  // probability rather than ever failing a correct implementation.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(second_acquired.load());

  first.reset();
  contender.join();
  EXPECT_TRUE(second_acquired.load());

  shm_unlink((name + ".lock").c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryCreationLockOutlivesSegment) {
  std::string shm_key = "/test_raiden_shm_lockfile_" + std::to_string(getpid());
  const std::string lock_file =
      std::string("/dev/shm") + shm_key + ".lock";
  shm_unlink(shm_key.c_str());
  shm_unlink((shm_key + ".lock").c_str());

  SharedMemoryHeader schema = {};
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "lock_model");

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(1024));
    std::memset(alloc.ptr, 0x11, 1024);
    EXPECT_EQ(access(lock_file.c_str(), F_OK), 0);
  }

  // The lock file survives the allocator (it is never unlinked) and a warm
  // boot next to it still recovers the data.
  EXPECT_EQ(access(lock_file.c_str(), F_OK), 0);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema));
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(1024));
    for (size_t i = 0; i < 1024; ++i) {
      ASSERT_EQ(alloc.ptr[i], 0x11);
    }
  }

  shm_unlink(shm_key.c_str());
  shm_unlink((shm_key + ".lock").c_str());
}

TEST(HostMemoryAllocatorTest, WarnAboutOrphanShmSegmentsReportsUnattached) {
  const std::string shm_key =
      "/test_raiden_shm_orphan_" + std::to_string(getpid());
  const std::string own = shm_key + "_dev_1";
  const std::string sibling = shm_key + "_dev_99";
  const std::string metadata = shm_key + "_metadata_other";
  const std::string leftover = shm_key + "_old_run";
  const std::string lock_file = shm_key + ".lock";
  auto cleanup = absl::MakeCleanup([&]() {
    for (const std::string& name :
         {own, sibling, metadata, leftover, lock_file}) {
      shm_unlink(name.c_str());
    }
  });

  for (const std::string& name :
       {own, sibling, metadata, leftover, lock_file}) {
    shm_unlink(name.c_str());
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_GE(fd, 0);
    close(fd);
  }

  // The expected own segment is silent, the sibling-shaped and metadata
  // names surface at INFO only, ".lock" companions are skipped; the
  // unclassifiable leftover is the one returned finding.
  std::vector<std::string> reported =
      WarnAboutOrphanShmSegments(shm_key, {own});
  ASSERT_EQ(reported.size(), 1u);
  EXPECT_EQ(reported[0], leftover.substr(1));
}

}  // namespace
}  // namespace tpu_raiden

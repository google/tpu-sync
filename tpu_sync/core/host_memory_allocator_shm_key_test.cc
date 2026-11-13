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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/host_memory_allocator.h"

namespace tpu_raiden {
namespace {

class ShmSegmentNameTest : public ::testing::Test {
 protected:
  void SetUp() override { unsetenv("RAIDEN_SHM_SERVER_NAME"); }
  void TearDown() override { unsetenv("RAIDEN_SHM_SERVER_NAME"); }
};

TEST_F(ShmSegmentNameTest, BareKeyGetsSlashPrefix) {
  EXPECT_EQ(SharedMemoryHostMemoryAllocator::ComposeSegmentName(
                "raiden_pool", std::nullopt),
            "/raiden_pool");
  EXPECT_EQ(SharedMemoryHostMemoryAllocator::ComposeSegmentName(
                "/raiden_pool", std::nullopt),
            "/raiden_pool");
}

TEST_F(ShmSegmentNameTest, ServerNameAndDeviceIdSuffixes) {
  setenv("RAIDEN_SHM_SERVER_NAME", "replica0", /*overwrite=*/1);
  EXPECT_EQ(SharedMemoryHostMemoryAllocator::ComposeSegmentName("raiden_pool",
                                                                7),
            "/raiden_pool_replica0_dev_7");
}

// The segment name must differ across the ranks of one host. Ranks are told
// apart by the GLOBAL device id (the physical chip); the per-process local
// ordinal is 0 in every rank of a multi-process serving job, and a name
// derived from it would make all ranks silently share one segment.
TEST_F(ShmSegmentNameTest, DistinctGlobalDeviceIdsComposeDistinctNames) {
  EXPECT_NE(
      SharedMemoryHostMemoryAllocator::ComposeSegmentName("raiden_pool", 0),
      SharedMemoryHostMemoryAllocator::ComposeSegmentName("raiden_pool", 1));
}

TEST_F(ShmSegmentNameTest, ValidatePassesWellFormedParts) {
  EXPECT_TRUE(SharedMemoryHostMemoryAllocator::ValidateShmNameParts(
                  "raiden_pool-v2.1")
                  .ok());
  // One leading '/' is the shm_open prefix, not a violation.
  EXPECT_TRUE(
      SharedMemoryHostMemoryAllocator::ValidateShmNameParts("/raiden_pool")
          .ok());
  setenv("RAIDEN_SHM_SERVER_NAME", "replica0", /*overwrite=*/1);
  EXPECT_TRUE(
      SharedMemoryHostMemoryAllocator::ValidateShmNameParts("raiden_pool")
          .ok());
}

TEST_F(ShmSegmentNameTest, ValidateRejectsMalformedKey) {
  absl::Status status =
      SharedMemoryHostMemoryAllocator::ValidateShmNameParts("tmp/raiden");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("RAIDEN_SHM_KEY"));
  EXPECT_THAT(status.message(), ::testing::HasSubstr("'/'"));
  EXPECT_FALSE(
      SharedMemoryHostMemoryAllocator::ValidateShmNameParts("raiden pool")
          .ok());
  EXPECT_FALSE(SharedMemoryHostMemoryAllocator::ValidateShmNameParts(
                   std::string(201, 'a'))
                   .ok());
  EXPECT_FALSE(SharedMemoryHostMemoryAllocator::ValidateShmNameParts("").ok());
}

TEST_F(ShmSegmentNameTest, ValidateRejectsMalformedServerName) {
  setenv("RAIDEN_SHM_SERVER_NAME", "rep/lica", /*overwrite=*/1);
  absl::Status status =
      SharedMemoryHostMemoryAllocator::ValidateShmNameParts("raiden_pool");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("RAIDEN_SHM_SERVER_NAME"));
}

// Create is where every consumer (the serving-host allocator factory and any
// future C++ node) first sees the key, so the misconfiguration must fail
// there, at boot, not as a bare shm_open errno at the first allocation.
TEST_F(ShmSegmentNameTest, CreateRejectsMalformedKey) {
  EXPECT_FALSE(SharedMemoryHostMemoryAllocator::Create(
                   /*client=*/nullptr, "tmp/raiden", SharedMemoryHeader{})
                   .ok());
  EXPECT_TRUE(SharedMemoryHostMemoryAllocator::Create(
                  /*client=*/nullptr, "raiden_pool", SharedMemoryHeader{})
                  .ok());
}

TEST_F(ShmSegmentNameTest, AllocatePopulatesShmFdAndPageAlignedOffset) {
  const std::string shm_key = "test_shm_fd_offset_" + std::to_string(getpid());
  const std::string seg_name =
      SharedMemoryHostMemoryAllocator::ComposeSegmentName(shm_key,
                                                          std::nullopt);
  shm_unlink(seg_name.c_str());

  SharedMemoryHeader schema = {};
  std::snprintf(schema.model_uid, sizeof(schema.model_uid), "fd_offset_model");
  auto allocator_or = SharedMemoryHostMemoryAllocator::Create(
      /*client=*/nullptr, shm_key, schema);
  ASSERT_TRUE(allocator_or.ok());

  auto alloc1_or = (*allocator_or)->Allocate(1024);
  ASSERT_TRUE(alloc1_or.ok());
  auto alloc2_or = (*allocator_or)->Allocate(1024);
  ASSERT_TRUE(alloc2_or.ok());

  EXPECT_GE(alloc1_or->fd, 0);
  EXPECT_EQ(alloc1_or->fd, alloc2_or->fd);
  EXPECT_EQ(
      alloc1_or->fd_offset,
      static_cast<int64_t>(SharedMemoryHostMemoryAllocator::kPayloadOffset));
  EXPECT_EQ(alloc2_or->fd_offset,
            static_cast<int64_t>(
                SharedMemoryHostMemoryAllocator::kPayloadOffset + 4096));
  EXPECT_EQ(alloc1_or->shm_name, seg_name);

  std::memset(alloc2_or->ptr, 0x5A, 1024);
  std::vector<uint8_t> readback(1024, 0);
  ASSERT_EQ(pread(alloc2_or->fd, readback.data(), readback.size(),
                  alloc2_or->fd_offset),
            1024);
  for (uint8_t b : readback) {
    EXPECT_EQ(b, 0x5A);
  }

  shm_unlink(seg_name.c_str());
}

}  // namespace
}  // namespace tpu_raiden

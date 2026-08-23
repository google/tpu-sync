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
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

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

    SharedMemoryHeader* header1 = reinterpret_cast<SharedMemoryHeader*>(
        static_cast<uint8_t*>(alloc1.ptr) - sizeof(SharedMemoryHeader));
    EXPECT_EQ(header1->reference_count, 1);
    EXPECT_EQ(header1->version, 1);
    EXPECT_STREQ(header1->model_uid, "test_model_v1");

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

      SharedMemoryHeader* header2 = reinterpret_cast<SharedMemoryHeader*>(
          static_cast<uint8_t*>(alloc2.ptr) - sizeof(SharedMemoryHeader));
      EXPECT_EQ(header2->reference_count, 2);
    }

    EXPECT_EQ(header1->reference_count, 1);

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

      SharedMemoryHeader* header3 = reinterpret_cast<SharedMemoryHeader*>(
          static_cast<uint8_t*>(alloc3.ptr) - sizeof(SharedMemoryHeader));
      EXPECT_EQ(header3->reference_count, 1);
      EXPECT_EQ(header3->version, 2);
      EXPECT_STREQ(header3->model_uid, "test_model_v2");

      for (size_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(alloc3.ptr[i], 0);
      }
    }
  }

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, SharedMemoryInfoAndHostBufferAllocator) {
  std::string shm_key = "/test_raiden_shm_info_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  schema.magic = 0x52414944454E;
  schema.version = 1;
  schema.num_blocks = 64;
  schema.block_size = 4096;

  auto allocator_or =
      SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema);
  ASSERT_TRUE(allocator_or.ok());
  std::shared_ptr<SharedMemoryHostMemoryAllocator> shm_alloc =
      std::move(allocator_or).value();

  // Test HostBufferAllocator wrapping shared_ptr
  HostBufferAllocator wrapper(shm_alloc);
  EXPECT_TRUE(static_cast<bool>(wrapper));
  EXPECT_EQ(wrapper.host_memory_allocator(), shm_alloc);

  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc, wrapper(4096, nullptr));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, 4096);

  // Test GetSharedMemoryInfo
  auto info_or = shm_alloc->GetSharedMemoryInfo(alloc.ptr);
  ASSERT_TRUE(info_or.ok());
  SharedMemoryInfo info = info_or.value();
  EXPECT_EQ(info.shm_key, shm_key);
  EXPECT_GT(info.size, 0);
  EXPECT_EQ(info.offset, sizeof(SharedMemoryHeader));
  EXPECT_EQ(info.base_ptr, static_cast<const uint8_t*>(alloc.ptr) -
                               sizeof(SharedMemoryHeader));
  EXPECT_GE(info.fd, 0);

  // Invalid pointer test
  uint8_t dummy_ptr[16];
  EXPECT_FALSE(shm_alloc->GetSharedMemoryInfo(dummy_ptr).ok());

  shm_unlink(shm_key.c_str());
}

TEST(HostMemoryAllocatorTest, CreateHostMemoryAllocatorFactory) {
  std::string shm_key = "/test_create_factory_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());
  setenv("RAIDEN_SHM_KEY", shm_key.c_str(), 1);

  auto host_alloc = CreateHostMemoryAllocator(nullptr, /*num_blocks=*/4,
                                              /*block_size=*/4096);
  ASSERT_TRUE(static_cast<bool>(host_alloc));
  ASSERT_NE(host_alloc.host_memory_allocator(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          host_alloc(4 * 4096, nullptr));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, 4 * 4096);

  auto info_or =
      host_alloc.host_memory_allocator()->GetSharedMemoryInfo(alloc.ptr);
  ASSERT_TRUE(info_or.ok());
  EXPECT_EQ(info_or->shm_key, shm_key);
  EXPECT_GE(info_or->fd, 0);
  EXPECT_EQ(info_or->offset, sizeof(SharedMemoryHeader));

  unsetenv("RAIDEN_SHM_KEY");
  shm_unlink(shm_key.c_str());
}

namespace {
bool SendFdOverUds(int sock, int fd, void* buf, size_t buflen) {
  struct msghdr msg = {0};
  struct iovec iov[1];
  iov[0].iov_base = buf;
  iov[0].iov_len = buflen;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  *(reinterpret_cast<int*>(CMSG_DATA(cmsg))) = fd;

  return sendmsg(sock, &msg, 0) == static_cast<ssize_t>(buflen);
}

int RecvFdOverUds(int sock, void* buf, size_t buflen) {
  struct msghdr msg = {0};
  struct iovec iov[1];
  iov[0].iov_base = buf;
  iov[0].iov_len = buflen;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  if (recvmsg(sock, &msg, 0) < 0) return -1;

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    return *(reinterpret_cast<int*>(CMSG_DATA(cmsg)));
  }
  return -1;
}
}  // namespace

TEST(HostMemoryAllocatorTest, CrossProcessSharedMemoryIpcWithFdTransfer) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);

  std::string shm_key = "/test_raiden_shm_uds_" + std::to_string(getpid());
  shm_unlink(shm_key.c_str());

  SharedMemoryHeader schema = {};
  schema.magic = 0x52414944454E;
  schema.version = 1;
  absl::SNPrintF(schema.model_uid, sizeof(schema.model_uid), "uds_ipc_model");
  schema.num_blocks = 4;
  schema.block_size = 4096;

  auto allocator_or =
      SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema);
  ASSERT_TRUE(allocator_or.ok());
  auto allocator = std::move(allocator_or).value();

  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(4096));
  std::memset(alloc.ptr, 0x42, 4096);

  auto info_or = allocator->GetSharedMemoryInfo(alloc.ptr);
  ASSERT_TRUE(info_or.ok());
  SharedMemoryInfo info = info_or.value();
  ASSERT_GE(info.fd, 0);

  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {  // Child: Receiver daemon
    close(sv[0]);

    struct {
      size_t offset;
      size_t size;
    } meta;

    int recved_fd = RecvFdOverUds(sv[1], &meta, sizeof(meta));
    if (recved_fd < 0) _exit(1);

    void* mapped = mmap(nullptr, meta.offset + meta.size,
                        PROT_READ | PROT_WRITE, MAP_SHARED, recved_fd, 0);
    if (mapped == MAP_FAILED) _exit(2);

    uint8_t* payload = static_cast<uint8_t*>(mapped) + meta.offset;
    for (size_t i = 0; i < meta.size; ++i) {
      if (payload[i] != 0x42) _exit(3);
    }

    // Modify buffer from child
    std::memset(payload, 0x77, meta.size);

    char ack = 'K';
    (void)send(sv[1], &ack, 1, 0);

    munmap(mapped, meta.offset + meta.size);
    close(recved_fd);
    close(sv[1]);
    _exit(0);
  } else {  // Parent: Sender
    close(sv[1]);

    struct {
      size_t offset;
      size_t size;
    } meta = {info.offset, alloc.size};

    ASSERT_TRUE(SendFdOverUds(sv[0], info.fd, &meta, sizeof(meta)));

    char ack = 0;
    (void)recv(sv[0], &ack, 1, 0);
    EXPECT_EQ(ack, 'K');

    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);

    // Verify parent sees child's zero-copy modification
    for (size_t i = 0; i < 4096; ++i) {
      EXPECT_EQ(alloc.ptr[i], 0x77);
    }

    close(sv[0]);
    shm_unlink(shm_key.c_str());
  }
}

TEST(HostMemoryAllocatorTest, FileStorageOffloadAndRecall) {
  TF_ASSERT_OK_AND_ASSIGN(auto allocator, HostMemoryAllocator::Create(nullptr));

  const size_t kSize = 1024 * 1024;  // 1MB
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(kSize));
  ASSERT_NE(alloc.ptr, nullptr);

  for (size_t i = 0; i < kSize; ++i) {
    alloc.ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  std::string temp_file =
      "/tmp/raiden_allocator_test_" + std::to_string(getpid()) + ".bin";
  {
    std::ofstream ofs(temp_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ofs.write(reinterpret_cast<const char*>(alloc.ptr), kSize);
    ASSERT_TRUE(ofs.good());
  }

  // Corrupt original buffer
  std::memset(alloc.ptr, 0, kSize);

  // Recall back from disk
  {
    std::ifstream ifs(temp_file, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    ifs.read(reinterpret_cast<char*>(alloc.ptr), kSize);
    ASSERT_TRUE(ifs.good());
  }

  // Verify contents
  for (size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(alloc.ptr[i], static_cast<uint8_t>(i & 0xFF));
  }

  std::remove(temp_file.c_str());
}

}  // namespace
}  // namespace tpu_raiden

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

#include "tdsul/tdsul.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/syscall_internal.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tdsul/def.h"

namespace {

class MockSyscall : public tdsul::Syscall {
 public:
  MOCK_METHOD(ssize_t, pread, (int fd, void* buf, size_t count, off_t offset),
              (override));
  MOCK_METHOD(ssize_t, pwrite,
              (int fd, const void* buf, size_t count, off_t offset),
              (override));
  MOCK_METHOD(ssize_t, preadv,
              (int fd, const struct iovec* iov, int iovcnt, off_t offset),
              (override));
  MOCK_METHOD(ssize_t, pwritev,
              (int fd, const struct iovec* iov, int iovcnt, off_t offset),
              (override));
};

class TdsulTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {}
};
// todo: add test for virtual pointer registration
TEST_F(TdsulTest, TestDramBufferRegistration) {
  // Allocate host DRAM memory
  const std::size_t kSize = 4096;
  void* host_ptr = std::malloc(kSize);
  ASSERT_NE(host_ptr, nullptr);

  tds_buffer_handle_t* handle = nullptr;

  // 1. Success case: Register DRAM buffer
  tds_result_t res =
      tds_buffer_register_vaddr(host_ptr, TDS_MEM_HOST, kSize, &handle);
  EXPECT_EQ(res, TDS_SUCCESS);
  EXPECT_NE(handle, nullptr);

  // 2. Deregister DRAM buffer
  res = tds_buffer_deregister(handle);
  EXPECT_EQ(res, TDS_SUCCESS);

  // 3. Error case: Register with null pointer
  handle = nullptr;
  res = tds_buffer_register_vaddr(nullptr, TDS_MEM_HOST, kSize, &handle);
  EXPECT_EQ(res, TDS_ERROR_INVALID_PARAMETER);
  EXPECT_EQ(handle, nullptr);

  std::free(host_ptr);
}

TEST_F(TdsulTest, TestDeviceMemoryRegistrationDefault) {
  tds_init(nullptr);

  const std::size_t kSize = 4096;
  void* dummy_ptr = reinterpret_cast<void*>(0x12340000);
  tds_buffer_handle_t* handle = nullptr;

  tds_result_t res =
      tds_buffer_register_vaddr(dummy_ptr, TDS_MEM_DEVICE, kSize, &handle);
  EXPECT_EQ(res, TDS_ERROR_P2P_UNSUPPORTED);
  EXPECT_EQ(handle, nullptr);
  tds_shutdown();
}

TEST_F(TdsulTest, TestDeviceMemoryRegistrationEnabled) {
  tds_config_t* config = tds_config_create();
  tds_config_set_int(config, "enable_p2p", 1);
  tds_init(config);

  const std::size_t kSize = 4096;
  void* dummy_ptr = reinterpret_cast<void*>(0x12340000);
  tds_buffer_handle_t* handle = nullptr;

  tds_result_t res =
      tds_buffer_register_vaddr(dummy_ptr, TDS_MEM_DEVICE, kSize, &handle);
  EXPECT_EQ(res, TDS_SUCCESS);
  EXPECT_NE(handle, nullptr);

  if (handle) {
    EXPECT_EQ(tds_buffer_deregister(handle), TDS_SUCCESS);
  }

  // cleanup
  tds_config_destroy(config);
  tds_shutdown();
}

TEST_F(TdsulTest, TestDeviceDmaBufRegistrationDefault) {
  tds_init(nullptr);

  const std::size_t kSize = 4096;
  int dummy_fd = 42;
  tds_buffer_handle_t* handle = nullptr;

  tds_result_t res =
      tds_buffer_register_dmabuf(dummy_fd, 0, TDS_MEM_DEVICE, kSize, &handle);
  EXPECT_EQ(res, TDS_ERROR_P2P_UNSUPPORTED);
  EXPECT_EQ(handle, nullptr);

  tds_shutdown();
}

TEST_F(TdsulTest, TestDeviceDmaBufRegistrationEnabled) {
  tds_config_t* config = tds_config_create();
  tds_config_set_int(config, "enable_p2p", 1);
  tds_init(config);

  // Note: we can't fully mock mmap here easily for success case without custom
  // mocking, but if we pass a valid file descriptor or just mock the sys call,
  // wait. The dmabuf registration actually calls mmap on the fd! Let's pass
  // /dev/zero so mmap succeeds.
  int fd = open("/dev/zero", O_RDWR);
  ASSERT_GE(fd, 0);

  const std::size_t kSize = 4096;
  tds_buffer_handle_t* handle = nullptr;

  tds_result_t res =
      tds_buffer_register_dmabuf(fd, 0, TDS_MEM_DEVICE, kSize, &handle);
  EXPECT_EQ(res, TDS_SUCCESS);
  EXPECT_NE(handle, nullptr);

  if (handle) {
    EXPECT_EQ(tds_buffer_deregister(handle), TDS_SUCCESS);
  }

  close(fd);
  tds_config_destroy(config);
  tds_shutdown();
}

TEST_F(TdsulTest, TestReadInterruptedBySignalEINTR) {
  MockSyscall mock;
  tdsul::Syscall* original_syscall = tdsul::g_syscall;
  tdsul::g_syscall = &mock;

  const std::size_t kSize = 1024;
  void* buf = std::malloc(kSize);
  tds_buffer_handle_t* buf_handle = nullptr;
  ASSERT_EQ(tds_buffer_register_vaddr(buf, TDS_MEM_HOST, kSize, &buf_handle),
            TDS_SUCCESS);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, kSize);
  tds_buffer_io_t buffer_io =
      tds_create_registered_buffer_io(buf_handle, 0, kSize);

  using ::testing::_;
  using ::testing::Return;

  // Expect pread to be called. First time returns -1 with EINTR, second time
  // returns kSize.
  EXPECT_CALL(mock, pread(42, _, kSize, 0))
      .WillOnce([](int, void*, size_t, off_t) {
        errno = EINTR;
        return -1;
      })
      .WillOnce(Return(kSize));

  ssize_t res = tds_read(&storage_io, &buffer_io);
  EXPECT_EQ(res, static_cast<ssize_t>(kSize));

  // Cleanup
  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  EXPECT_EQ(tds_buffer_deregister(buf_handle), TDS_SUCCESS);
  std::free(buf);
  tdsul::g_syscall = original_syscall;
}

TEST_F(TdsulTest, TestWriteInterruptedBySignalEINTR) {
  MockSyscall mock;
  tdsul::Syscall* original_syscall = tdsul::g_syscall;
  tdsul::g_syscall = &mock;

  const std::size_t kSize = 1024;
  void* buf = std::malloc(kSize);
  tds_buffer_handle_t* buf_handle = nullptr;
  ASSERT_EQ(tds_buffer_register_vaddr(buf, TDS_MEM_HOST, kSize, &buf_handle),
            TDS_SUCCESS);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, kSize);
  tds_buffer_io_t buffer_io =
      tds_create_registered_buffer_io(buf_handle, 0, kSize);

  using ::testing::_;
  using ::testing::Return;

  // Expect pwrite to be called. First time returns -1 with EINTR, second time
  // returns kSize.
  EXPECT_CALL(mock, pwrite(42, _, kSize, 0))
      .WillOnce([](int, const void*, size_t, off_t) {
        errno = EINTR;
        return -1;
      })
      .WillOnce(Return(kSize));

  ssize_t res = tds_write(&storage_io, &buffer_io);
  EXPECT_EQ(res, static_cast<ssize_t>(kSize));

  // Cleanup
  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  EXPECT_EQ(tds_buffer_deregister(buf_handle), TDS_SUCCESS);
  std::free(buf);
  tdsul::g_syscall = original_syscall;
}

TEST_F(TdsulTest, ReadvEINTRRetry) {
  MockSyscall mock;
  tdsul::Syscall* original_syscall = tdsul::g_syscall;
  tdsul::g_syscall = &mock;

  const std::size_t kSize = 1024;
  void* buf = std::malloc(kSize);
  tds_buffer_io_t buffer_io = tds_create_raw_buffer_io(buf, kSize);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, kSize);

  using ::testing::_;
  using ::testing::Return;

  EXPECT_CALL(mock, preadv(42, _, 1, 0))
      .WillOnce([](int, const struct iovec*, int, off_t) {
        errno = EINTR;
        return -1;
      })
      .WillOnce(Return(kSize));

  ssize_t res = tds_readv(&storage_io, &buffer_io, 1);
  EXPECT_EQ(res, static_cast<ssize_t>(kSize));

  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
  tdsul::g_syscall = original_syscall;
}

TEST_F(TdsulTest, WritevEINTRRetry) {
  MockSyscall mock;
  tdsul::Syscall* original_syscall = tdsul::g_syscall;
  tdsul::g_syscall = &mock;

  const std::size_t kSize = 1024;
  void* buf = std::malloc(kSize);
  tds_buffer_io_t buffer_io = tds_create_raw_buffer_io(buf, kSize);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, kSize);

  using ::testing::_;
  using ::testing::Return;

  EXPECT_CALL(mock, pwritev(42, _, 1, 0))
      .WillOnce([](int, const struct iovec*, int, off_t) {
        errno = EINTR;
        return -1;
      })
      .WillOnce(Return(kSize));

  ssize_t res = tds_writev(&storage_io, &buffer_io, 1);
  EXPECT_EQ(res, static_cast<ssize_t>(kSize));

  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
  tdsul::g_syscall = original_syscall;
}

TEST_F(TdsulTest, VectoredIoSizeOverflow) {
  void* buf = std::malloc(1024);

  tds_buffer_io_t buffer_iov[2];
  buffer_iov[0] = tds_create_raw_buffer_io(buf, SSIZE_MAX - 100);
  buffer_iov[1] = tds_create_raw_buffer_io(buf, 200);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, 1024);

  errno = 0;
  ssize_t res = tds_readv(&storage_io, buffer_iov, 2);
  EXPECT_EQ(res, -1);
  EXPECT_EQ(errno, EINVAL);

  errno = 0;
  res = tds_writev(&storage_io, buffer_iov, 2);
  EXPECT_EQ(res, -1);
  EXPECT_EQ(errno, EINVAL);

  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
}

TEST_F(TdsulTest, RegisteredBufferBoundsCheck) {
  void* buf = std::malloc(1024);

  tds_buffer_handle_t* handle = nullptr;
  ASSERT_EQ(tds_buffer_register_vaddr(buf, TDS_MEM_HOST, 1024, &handle),
            TDS_SUCCESS);

  // offset + size > 1024
  tds_buffer_io_t buffer_io = tds_create_registered_buffer_io(handle, 500, 600);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, 1024);

  errno = 0;
  ssize_t res = tds_read(&storage_io, &buffer_io);
  EXPECT_EQ(res, -1);
  EXPECT_EQ(errno, EINVAL);

  EXPECT_EQ(tds_buffer_deregister(handle), TDS_SUCCESS);
  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
}

TEST_F(TdsulTest, VectoredIoInvalidArgs) {
  void* buf = std::malloc(1024);
  tds_buffer_io_t bio = tds_create_raw_buffer_io(buf, 1024);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, 1024);

  // Null storage_io
  EXPECT_EQ(tds_readv(nullptr, &bio, 1), -1);
  // Null storage_io->handle
  tds_storage_io_t null_handle_io = storage_io;
  null_handle_io.handle = nullptr;
  EXPECT_EQ(tds_readv(&null_handle_io, &bio, 1), -1);
  // Null buffer_iov
  EXPECT_EQ(tds_readv(&storage_io, nullptr, 1), -1);
  // iovcnt <= 0
  EXPECT_EQ(tds_readv(&storage_io, &bio, 0), -1);
  // iovcnt > IOV_MAX
  std::vector<tds_buffer_io_t> huge_iov(IOV_MAX + 1, bio);
  EXPECT_EQ(tds_readv(&storage_io, huge_iov.data(), IOV_MAX + 1), -1);

  // Same for writev
  EXPECT_EQ(tds_writev(nullptr, &bio, 1), -1);
  EXPECT_EQ(tds_writev(&null_handle_io, &bio, 1), -1);
  EXPECT_EQ(tds_writev(&storage_io, nullptr, 1), -1);
  EXPECT_EQ(tds_writev(&storage_io, &bio, -1), -1);
  EXPECT_EQ(tds_writev(&storage_io, huge_iov.data(), IOV_MAX + 1), -1);

  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
}

TEST_F(TdsulTest, VectoredIoZeroSize) {
  MockSyscall mock;
  tdsul::Syscall* original_syscall = tdsul::g_syscall;
  tdsul::g_syscall = &mock;

  void* buf = std::malloc(1024);
  tds_buffer_io_t bio = tds_create_raw_buffer_io(buf, 0);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, 1024);

  // Expect 0 without calling preadv/pwritev
  EXPECT_CALL(mock,
              preadv(::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .Times(0);
  EXPECT_CALL(mock,
              pwritev(::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .Times(0);

  EXPECT_EQ(tds_readv(&storage_io, &bio, 1), 0);
  EXPECT_EQ(tds_writev(&storage_io, &bio, 1), 0);

  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
  tdsul::g_syscall = original_syscall;
}

TEST_F(TdsulTest, VectoredIoSuccess) {
  MockSyscall mock;
  tdsul::Syscall* original_syscall = tdsul::g_syscall;
  tdsul::g_syscall = &mock;

  const std::size_t kSize = 1024;
  void* buf = std::malloc(kSize);
  tds_buffer_io_t buffer_io[2];
  buffer_io[0] = tds_create_raw_buffer_io(buf, kSize / 2);
  buffer_io[1] =
      tds_create_raw_buffer_io(static_cast<char*>(buf) + kSize / 2, kSize / 2);

  tds_storage_handle_t* storage_handle = nullptr;
  tds_storage_descr_t storage_descr;
  storage_descr.uri = "fd://42";
  storage_descr.options = nullptr;
  ASSERT_EQ(tds_storage_handle_register(&storage_descr, &storage_handle),
            TDS_SUCCESS);

  tds_storage_io_t storage_io = tds_create_file_io(storage_handle, 0, kSize);

  using ::testing::_;
  using ::testing::Return;

  EXPECT_CALL(mock, preadv(42, _, 2, 0)).WillOnce(Return(kSize));
  ssize_t res = tds_readv(&storage_io, buffer_io, 2);
  EXPECT_EQ(res, static_cast<ssize_t>(kSize));

  EXPECT_CALL(mock, pwritev(42, _, 2, 0)).WillOnce(Return(kSize));
  res = tds_writev(&storage_io, buffer_io, 2);
  EXPECT_EQ(res, static_cast<ssize_t>(kSize));

  EXPECT_EQ(tds_storage_handle_deregister(storage_handle), TDS_SUCCESS);
  std::free(buf);
  tdsul::g_syscall = original_syscall;
}
// TODO: add tests for tdsul async IO after the logic is finished. We shall
// specically check:
//  (1) whether the FIFO order is guaranteed, w/ or w/o batched IO.
//  (2) whether concurrency submission is properly handled.
}  // namespace

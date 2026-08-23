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

#include "tpu_sync/kv_cache/storage/storage.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "tpu_sync/kv_cache/storage/k5_backend.h"

namespace tpu_raiden {
namespace kv_cache {
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

  std::string scratch_dir_;
};

TEST_F(StorageDriverTest, PosixPathMapperResolution) {
  PosixPathMapper mapper(scratch_dir_, "llama3", /*tp_size=*/4, /*rank=*/2);
  EXPECT_EQ(mapper.tp_size(), 4);

  // Default rank (-1 uses rank 2)
  BlockKey key_default = mapper.MapKey("hash_12345", -1);
  EXPECT_EQ(key_default.block_hash, "hash_12345");
  EXPECT_EQ(key_default.resolved_key,
            absl::StrCat(scratch_dir_, "/llama3_tp4_r2/hash_12345.bin"));

  // Explicit rank override
  BlockKey key_rank0 = mapper.MapKey("hash_12345", 0);
  EXPECT_EQ(key_rank0.block_hash, "hash_12345");
  EXPECT_EQ(key_rank0.resolved_key,
            absl::StrCat(scratch_dir_, "/llama3_tp4_r0/hash_12345.bin"));
}

TEST_F(StorageDriverTest, PosixBackendWriteReadAndExists) {
  PosixBackend backend("posix_disk");
  EXPECT_EQ(backend.scheme(), "posix_disk");

  PosixPathMapper mapper(scratch_dir_, "model_v1", /*tp_size=*/1, /*rank=*/0);
  BlockKey key = mapper.MapKey("block_alpha", 0);

  auto exists_or = backend.Exists(key);
  ASSERT_TRUE(exists_or.ok());
  EXPECT_FALSE(exists_or.value());

  // Prepare write buffer
  const size_t kSize = 4096;
  std::vector<uint8_t> write_data(kSize, 0xAB);
  StorageBufferDescriptor src_desc;
  src_desc.ptr = write_data.data();

  // Async write
  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, src_desc, kSize, [&](const absl::Status& status) {
    write_status = status;
    write_done.Notify();
  });
  write_done.WaitForNotification();
  EXPECT_TRUE(write_status.ok());

  // Verify file exists
  exists_or = backend.Exists(key);
  ASSERT_TRUE(exists_or.ok());
  EXPECT_TRUE(exists_or.value());

  // Async read
  std::vector<uint8_t> read_data(kSize, 0);
  StorageBufferDescriptor dst_desc;
  dst_desc.ptr = read_data.data();

  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, dst_desc, kSize, [&](const absl::Status& status) {
    read_status = status;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  EXPECT_TRUE(read_status.ok());

  // Verify read content matches written content
  EXPECT_EQ(std::memcmp(write_data.data(), read_data.data(), kSize), 0);

  // Read non-existent key returns error
  BlockKey missing_key = mapper.MapKey("non_existent_hash", 0);
  absl::Notification missing_done;
  absl::Status missing_status;
  backend.ReadAsync(missing_key, dst_desc, kSize,
                    [&](const absl::Status& status) {
                      missing_status = status;
                      missing_done.Notify();
                    });
  missing_done.WaitForNotification();
  EXPECT_FALSE(missing_status.ok());
}

TEST_F(StorageDriverTest, K5BlockNameMapperResolution) {
  K5BlockNameMapper mapper(scratch_dir_, "gemma2", /*tp_size=*/8, /*rank=*/3);

  BlockKey key = mapper.MapKey("chunk_999", -1);
  EXPECT_EQ(key.block_hash, "chunk_999");
  EXPECT_EQ(key.resolved_key,
            absl::StrCat(scratch_dir_, "/k5_gemma2_tp8_r3_chunk_999.bin"));
}

// Simple test daemon for testing K5 IPC over UDS
class TestUdsDaemon {
 public:
  TestUdsDaemon(std::string uds_path, std::string storage_dir)
      : uds_path_(std::move(uds_path)), storage_dir_(std::move(storage_dir)) {}

  ~TestUdsDaemon() { Stop(); }

  absl::Status Start() {
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
      return absl::InternalError(
          absl::StrCat("socket failed: ", std::strerror(errno)));
    }

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, uds_path_.c_str(), sizeof(addr.sun_path) - 1);
    unlink(uds_path_.c_str());

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      close(server_fd_);
      server_fd_ = -1;
      return absl::InternalError(
          absl::StrCat("bind failed on UDS: ", std::strerror(errno)));
    }

    if (listen(server_fd_, 5) < 0) {
      close(server_fd_);
      server_fd_ = -1;
      return absl::InternalError(
          absl::StrCat("listen failed on UDS: ", std::strerror(errno)));
    }

    running_ = true;
    thread_ = std::thread([this]() { Run(); });
    return absl::OkStatus();
  }

  void Stop() {
    running_ = false;
    if (server_fd_ >= 0) {
      close(server_fd_);
      server_fd_ = -1;
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    unlink(uds_path_.c_str());
  }

 private:
  void Run() {
    while (running_) {
      struct timeval tv;
      tv.tv_sec = 0;
      tv.tv_usec = 50000;
      fd_set readfds;
      FD_ZERO(&readfds);
      FD_SET(server_fd_, &readfds);

      int activity = select(server_fd_ + 1, &readfds, nullptr, nullptr, &tv);
      if (activity <= 0) continue;

      int client_sock = accept(server_fd_, nullptr, nullptr);
      if (client_sock < 0) continue;

      char control[CMSG_SPACE(sizeof(int))];
      char metadata[1024] = {0};

      struct iovec iov[1];
      iov[0].iov_base = metadata;
      iov[0].iov_len = sizeof(metadata) - 1;

      struct msghdr msg;
      std::memset(&msg, 0, sizeof(msg));
      msg.msg_iov = iov;
      msg.msg_iovlen = 1;
      msg.msg_control = control;
      msg.msg_controllen = sizeof(control);

      ssize_t recved = recvmsg(client_sock, &msg, 0);
      if (recved <= 0) {
        close(client_sock);
        continue;
      }

      struct cmsghdr* cmptr = CMSG_FIRSTHDR(&msg);
      if (cmptr == nullptr || cmptr->cmsg_type != SCM_RIGHTS) {
        (void)send(client_sock, "ERROR", 5, 0);
        close(client_sock);
        continue;
      }

      int fd = *(reinterpret_cast<int*>(CMSG_DATA(cmptr)));
      std::string op, key;
      size_t offset = 0, size = 0;

      for (const auto& token : absl::StrSplit(metadata, ' ')) {
        std::vector<std::string> kv = absl::StrSplit(token, '=');
        if (kv.size() == 2) {
          if (kv[0] == "op") op = kv[1];
          if (kv[0] == "offset") (void)absl::SimpleAtoi(kv[1], &offset);
          if (kv[0] == "size") (void)absl::SimpleAtoi(kv[1], &size);
          if (kv[0] == "key") key = kv[1];
        }
      }

      if (op == "write") {
        void* mapped =
            mmap(nullptr, offset + size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped != MAP_FAILED) {
          fs::create_directories(fs::path(key).parent_path());
          std::ofstream out(key, std::ios::binary);
          out.write(static_cast<const char*>(mapped) + offset, size);
          out.close();
          munmap(mapped, offset + size);
          (void)send(client_sock, "OK", 2, 0);
        } else {
          (void)send(client_sock, "ERROR", 5, 0);
        }
      } else if (op == "read") {
        void* mapped = mmap(nullptr, offset + size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
        if (mapped != MAP_FAILED) {
          std::ifstream in(key, std::ios::binary);
          if (in) {
            in.read(static_cast<char*>(mapped) + offset, size);
            in.close();
            (void)send(client_sock, "OK", 2, 0);
          } else {
            (void)send(client_sock, "ERROR", 5, 0);
          }
          munmap(mapped, offset + size);
        } else {
          (void)send(client_sock, "ERROR", 5, 0);
        }
      }
      close(fd);
      close(client_sock);
    }
  }

  std::string uds_path_;
  std::string storage_dir_;
  int server_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread thread_;
};

TEST_F(StorageDriverTest, K5BackendUdsTransfer) {
  std::string uds_path = absl::StrCat(scratch_dir_, "/k5_test.sock");
  std::string storage_dir = absl::StrCat(scratch_dir_, "/k5_storage");
  TestUdsDaemon daemon(uds_path, storage_dir);
  ASSERT_TRUE(daemon.Start().ok());

  K5Backend backend(uds_path);
  EXPECT_EQ(backend.scheme(), "k5");

  K5BlockNameMapper mapper(storage_dir, "model_test", /*tp_size=*/1,
                           /*rank=*/0);
  BlockKey key = mapper.MapKey("hash_xyz", 0);

  const size_t kSize = 4096;
  int shm_fd = memfd_create("test_k5_shm", 0);
  ASSERT_GE(shm_fd, 0);
  ASSERT_EQ(ftruncate(shm_fd, kSize), 0);

  void* ptr =
      mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);
  std::memset(ptr, 0x7E, kSize);

  StorageBufferDescriptor desc;
  desc.ptr = static_cast<uint8_t*>(ptr);
  desc.fd = shm_fd;
  desc.offset = 0;

  // Test WriteAsync
  absl::Notification write_done;
  absl::Status write_status;
  backend.WriteAsync(key, desc, kSize, [&](const absl::Status& status) {
    write_status = status;
    write_done.Notify();
  });
  write_done.WaitForNotification();
  EXPECT_TRUE(write_status.ok());

  // Test Exists
  auto exists_or = backend.Exists(key);
  ASSERT_TRUE(exists_or.ok());
  EXPECT_TRUE(exists_or.value());

  // Zero buffer and test ReadAsync
  std::memset(ptr, 0, kSize);
  absl::Notification read_done;
  absl::Status read_status;
  backend.ReadAsync(key, desc, kSize, [&](const absl::Status& status) {
    read_status = status;
    read_done.Notify();
  });
  read_done.WaitForNotification();
  EXPECT_TRUE(read_status.ok());

  // Verify data
  uint8_t* p = static_cast<uint8_t*>(ptr);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(p[i], 0x7E);
  }

  munmap(ptr, kSize);
  close(shm_fd);
}

}  // namespace
}  // namespace storage
}  // namespace kv_cache
}  // namespace tpu_raiden

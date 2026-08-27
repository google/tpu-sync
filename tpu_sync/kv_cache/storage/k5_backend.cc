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

#include "tpu_sync/kv_cache/storage/k5_backend.h"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/kv_cache/storage/storage.h"

namespace tpu_raiden {
namespace kv_cache {
namespace storage {

K5BackendMock::K5BackendMock(std::string uds_socket_path)
    : uds_socket_path_(std::move(uds_socket_path)) {}

absl::Status K5BackendMock::SendFdAndMetadata(int fd, size_t offset,
                                              size_t size,
                                              const std::string& resolved_key,
                                              bool is_write) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) {
    return absl::InternalError(
        absl::StrCat("socket failed: ", std::strerror(errno)));
  }

  struct sockaddr_un addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, uds_socket_path_.c_str(),
               sizeof(addr.sun_path) - 1);

  if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    close(sock);
    return absl::InternalError(
        absl::StrCat("connect failed on UDS: ", std::strerror(errno)));
  }

  std::string metadata =
      absl::StrCat("op=", is_write ? "write" : "read", " offset=", offset,
                   " size=", size, " key=", resolved_key);

  struct iovec iov[1];
  iov[0].iov_base = const_cast<char*>(metadata.data());
  iov[0].iov_len = metadata.size();

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct msghdr msg;
  std::memset(&msg, 0, sizeof(msg));
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  struct cmsghdr* cmptr = CMSG_FIRSTHDR(&msg);
  cmptr->cmsg_len = CMSG_LEN(sizeof(int));
  cmptr->cmsg_level = SOL_SOCKET;
  cmptr->cmsg_type = SCM_RIGHTS;
  *(reinterpret_cast<int*>(CMSG_DATA(cmptr))) = fd;

  ssize_t sent = sendmsg(sock, &msg, 0);
  if (sent < 0) {
    close(sock);
    return absl::InternalError(
        absl::StrCat("sendmsg failed: ", std::strerror(errno)));
  }

  char reply[16];
  std::memset(reply, 0, sizeof(reply));
  ssize_t recved = recv(sock, reply, sizeof(reply) - 1, 0);
  close(sock);

  if (recved <= 0 || absl::string_view(reply) != "OK") {
    return absl::InternalError(
        absl::StrCat("daemon verification failed: ", reply));
  }

  return absl::OkStatus();
}

void K5BackendMock::WriteAsync(
    const BlockKey& key, StorageBufferDescriptor src_buffer, size_t size,
    std::function<void(const absl::Status&)> callback) {
  absl::Status status =
      SendFdAndMetadata(src_buffer.fd, src_buffer.offset, size,
                        key.resolved_key, /*is_write=*/true);
  callback(status);
}

void K5BackendMock::ReadAsync(
    const BlockKey& key, StorageBufferDescriptor dst_buffer, size_t size,
    std::function<void(const absl::Status&)> callback) {
  absl::Status status =
      SendFdAndMetadata(dst_buffer.fd, dst_buffer.offset, size,
                        key.resolved_key, /*is_write=*/false);
  callback(status);
}

absl::StatusOr<bool> K5BackendMock::Exists(const BlockKey& key) {
  // Check if file exists inside scratch directory by matching hash path
  struct stat st;
  if (stat(key.resolved_key.c_str(), &st) == 0) {
    return true;
  }
  return false;
}

// --- K5BlockNameMapper Implementation ---

K5BlockNameMapper::K5BlockNameMapper(absl::string_view root_dir,
                                     absl::string_view model_name, int tp_size,
                                     int rank)
    : root_dir_(root_dir),
      model_name_(model_name),
      tp_size_(tp_size),
      rank_(rank) {}

BlockKey K5BlockNameMapper::MapKey(const std::string& block_hash,
                                   int rank) const {
  int target_rank = (rank == -1) ? rank_ : rank;
  std::string k5_block_name = absl::StrCat("k5_", model_name_, "_tp", tp_size_,
                                           "_r", target_rank, "_", block_hash);
  std::string resolved_path =
      absl::StrCat(root_dir_, "/", k5_block_name, ".bin");
  return BlockKey{block_hash, resolved_path};
}

}  // namespace storage
}  // namespace kv_cache
}  // namespace tpu_raiden

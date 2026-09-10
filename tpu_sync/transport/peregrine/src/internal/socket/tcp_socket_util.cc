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

#include "tpu_sync/transport/peregrine/src/internal/socket/tcp_socket_util.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/peregrine/src/internal/base/types.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/socket_util.h"
#include "tpu_sync/transport/peregrine/src/internal/util/util.h"

namespace peregrine::internal {

namespace {
inline std::string ErrMsg(std::string_view what, fd_t fd, int last_errno) {
  return absl::StrFormat("tcp socket %s failed: fd=%d %s errno=%d (%s)", what,
                         fd.value(), AddrPortPair(fd), last_errno,
                         std::strerror(last_errno));
}
}  // namespace

absl::Status TcpSocketUtil::Send(const fd_t fd, const Byte* const buf,
                                 const size_t len) {
  DCHECK(IsValidSocket(fd));
  DCHECK(IsBlockingMode(fd));

  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  if (len == 0) return absl::OkStatus();

  const Byte* ptr = buf;
  size_t sent = 0;
  ssize_t left = len;
  while (left > 0) {
    const ssize_t bytes = ::send(fd.value(), ptr, left, MSG_NOSIGNAL);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      DCHECK_LE(bytes, left);
      ptr += bytes;
      left -= bytes;
      sent += bytes;
      DCHECK_EQ(buf + len, ptr + left);
    } else {
      if ABSL_PREDICT_TRUE (bytes < 0) {
        const auto last_errno = errno;
        if (Interrupted(last_errno)) continue;
        DCHECK(!WouldBlock(last_errno));
        return absl::InternalError(ErrMsg("send", fd, last_errno));
      } else {  // rarely happens
        DCHECK_EQ(bytes, 0);
        return absl::InternalError("send zero");
      }
    }
  }
  DCHECK_EQ(left, 0);
  DCHECK_EQ(sent, len);
  return absl::OkStatus();
}

absl::Status TcpSocketUtil::Recv(const fd_t fd, Byte* const buf,
                                 const size_t len) {
  DCHECK(IsValidSocket(fd));
  DCHECK(IsBlockingMode(fd));

  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  if (len == 0) return absl::OkStatus();

  Byte* ptr = buf;
  size_t rcvd = 0;
  ssize_t left = len;
  while (left > 0) {
    const ssize_t bytes = ::recv(fd.value(), ptr, left, /*flags=*/0);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      DCHECK_LE(bytes, left);
      ptr += bytes;
      left -= bytes;
      rcvd += bytes;
      DCHECK_EQ(buf + len, ptr + left);
    } else if (bytes == 0) {  // peer closed connection
      return absl::InternalError("recv eof");
    } else {
      const auto last_errno = errno;
      if (Interrupted(last_errno)) continue;
      DCHECK(!WouldBlock(last_errno));
      return absl::InternalError(ErrMsg("recv", fd, last_errno));
    }
  }
  DCHECK_EQ(left, 0);
  DCHECK_EQ(rcvd, len);
  return absl::OkStatus();
}

absl::Status TcpSocketUtil::SendV(const fd_t fd,
                                  const absl::Span<const IoVec> iovecs) {
  DCHECK(IsValidSocket(fd));
  DCHECK(IsBlockingMode(fd));
  DCHECK_LE(iovecs.size(), IOV_MAX);

  const size_t len = TotalLength(iovecs);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  if (len == 0) return absl::OkStatus();

  std::vector<struct iovec> vecs{iovecs.begin(), iovecs.end()};
  const size_t n = vecs.size();
  size_t sent = 0;
  size_t i = 0;
  struct msghdr msg = {};
  while (i < n) {
    msg.msg_iov = &vecs[i];
    msg.msg_iovlen = n - i;
    const ssize_t bytes = ::sendmsg(fd.value(), &msg, MSG_NOSIGNAL);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      sent += bytes;
      if ABSL_PREDICT_TRUE (sent >= len) break;
      size_t b = static_cast<size_t>(bytes);
      while (i < n && vecs[i].iov_len <= b) {  // advance iov index
        b -= vecs[i].iov_len;
        ++i;
      }
      if (i >= n) break;
      if (b > 0) {  // adjust iov ptr/len
        vecs[i].iov_base = static_cast<Byte*>(vecs[i].iov_base) + b;
        vecs[i].iov_len -= b;
      }
    } else {
      if ABSL_PREDICT_TRUE (bytes < 0) {
        const auto last_errno = errno;
        if (Interrupted(last_errno)) continue;
        DCHECK(!WouldBlock(last_errno));
        return absl::InternalError(ErrMsg("sendmsg", fd, last_errno));
      } else {  // rarely happens
        DCHECK_EQ(bytes, 0);
        return absl::InternalError("sendmsg zero");
      }
    }
  }
  DCHECK_EQ(sent, len);
  return absl::OkStatus();
}

absl::Status TcpSocketUtil::RecvV(const fd_t fd,
                                  const absl::Span<const IoVec> iovecs) {
  DCHECK(IsValidSocket(fd));
  DCHECK(IsBlockingMode(fd));
  DCHECK_LE(iovecs.size(), IOV_MAX);

  const size_t len = TotalLength(iovecs);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  if (len == 0) return absl::OkStatus();

  std::vector<struct iovec> vecs{iovecs.begin(), iovecs.end()};
  const int n = vecs.size();
  size_t rcvd = 0;
  int i = 0;
  while (i < n) {
    const ssize_t bytes = ::readv(fd.value(), &vecs[i], n - i);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      rcvd += bytes;
      if ABSL_PREDICT_TRUE (rcvd >= len) break;
      size_t b = static_cast<size_t>(bytes);
      while (i < n && vecs[i].iov_len <= b) {  // advance iov index
        b -= vecs[i].iov_len;
        ++i;
      }
      if (i >= n) break;
      if (b > 0) {  // adjust iov ptr/len
        vecs[i].iov_base = static_cast<Byte*>(vecs[i].iov_base) + b;
        vecs[i].iov_len -= b;
      }
    } else if (bytes == 0) {  // peer closed connection
      return absl::InternalError("readv eof");
    } else {
      const auto last_errno = errno;
      if (Interrupted(last_errno)) continue;
      DCHECK(!WouldBlock(last_errno));
      return absl::InternalError(ErrMsg("readv", fd, last_errno));
    }
  }
  DCHECK_EQ(rcvd, len);
  return absl::OkStatus();
}

}  // namespace peregrine::internal

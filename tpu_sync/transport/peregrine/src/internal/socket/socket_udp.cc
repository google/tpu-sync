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

#include "tpu_sync/transport/peregrine/src/internal/socket/socket_udp.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_sync/transport/peregrine/src/internal/base/types.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/socket_base.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/socket_util.h"
#include "tpu_sync/transport/peregrine/src/internal/util/util.h"

namespace peregrine::internal {

std::unique_ptr<UdpSocket> UdpSocket::Create(int family) {
  const fd_t fd = CreateSocket(family, SOCK_DGRAM, /*blocking=*/true);
  if ABSL_PREDICT_FALSE (fd.value() < 0) {
    return nullptr;
  } else {
    LOG(INFO) << okMsg("created", fd);
    return absl::WrapUnique(new UdpSocket(fd, family));
  }
}

UdpSocket::~UdpSocket() {
  DCHECK(invariant());
  if (connected_) Shutdown();
  DCHECK(!connected_);
  LOG(INFO) << okMsg("closing");
  DCHECK(invariant());
  ::close(fd_.value());
  fd_ = fd_t(-1);
}

void UdpSocket::Shutdown() {
  DCHECK(invariant());
  LOG(INFO) << okMsg("shutdown");
  ::shutdown(fd_.value(), SHUT_RDWR);
  connected_ = false;
  DCHECK(invariant());
}

bool UdpSocket::Bind(const Endpoint& local) const {
  DCHECK(invariant());
  if ABSL_PREDICT_FALSE (SocketBase::Bind(fd_, local) < 0) {
    const auto last_errno = errno;
    LOG(WARNING) << errMsg("bind", last_errno);
    return false;
  } else {
    LOG(INFO) << okMsg("bound");
    return true;
  }
}

bool UdpSocket::Connect(const Endpoint& peer) {
  DCHECK(invariant());
  if ABSL_PREDICT_FALSE (SocketBase::Connect(fd_, peer) < 0) {
    const auto last_errno = errno;
    LOG(WARNING) << errMsg("connect", last_errno);
    return false;
  } else {
    LOG(INFO) << okMsg("connected");
    connected_ = true;
    return true;
  }
}

ssize_t UdpSocket::Send(const Byte* const buf, const size_t len) const {
  DCHECK(invariant());
  DCHECK(IsBlocking());
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());

  while (true) {
    const ssize_t bytes = ::send(fd_.value(), buf, len, MSG_NOSIGNAL);
    DCHECK(bytes == len || bytes < 0);
    if ABSL_PREDICT_TRUE (bytes == len) {
      VLOG(1) << ioMsg("send", bytes);
      return bytes;
    }
    const auto last_errno = errno;
    if (Interrupted(last_errno)) continue;
    DCHECK(!WouldBlock(last_errno));
    LOG(WARNING) << errMsg("send", last_errno);
    return -1;
  }
}

ssize_t UdpSocket::Recv(Byte* const buf, const size_t len) const {
  DCHECK(invariant());
  DCHECK(IsBlocking());
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());

  while (true) {
    const ssize_t bytes = ::recv(fd_.value(), buf, len, /*flags=*/0);
    DCHECK_LE(bytes, len);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      VLOG(1) << ioMsg("recv", bytes);
      return bytes;
    } else if (bytes < 0) {
      const auto last_errno = errno;
      if (Interrupted(last_errno)) continue;
      DCHECK(!WouldBlock(last_errno));
      LOG(WARNING) << errMsg("recv", last_errno);
      return -1;
    } else {
      DCHECK_EQ(bytes, 0);
      LOG(INFO) << ioMsg("recv no payload", 0);
      return 0;
    }
  }
}

ssize_t UdpSocket::SendV(const absl::Span<const IoVec> iovecs) const {
  DCHECK(invariant());
  DCHECK(IsBlocking());
  DCHECK_LE(iovecs.size(), IOV_MAX);

  const size_t len = TotalLength(iovecs);
  DCHECK_EQ(len, TotalLength(iovecs));
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());

  while (true) {
    const ssize_t bytes = ::writev(fd_.value(), iovecs.data(), iovecs.size());
    DCHECK(bytes == len || bytes < 0);
    if ABSL_PREDICT_TRUE (bytes == len) {
      VLOG(1) << ioMsg("writev", bytes);
      return bytes;
    }
    const auto last_errno = errno;
    if (Interrupted(last_errno)) continue;
    DCHECK(!WouldBlock(last_errno));
    LOG(WARNING) << errMsg("writev", last_errno);
    return -1;
  }
}

ssize_t UdpSocket::RecvV(const absl::Span<const IoVec> iovecs) const {
  DCHECK(invariant());
  DCHECK(IsBlocking());
  DCHECK_LE(iovecs.size(), IOV_MAX);

  const size_t len = TotalLength(iovecs);
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());

  while (true) {
    const ssize_t bytes = ::readv(fd_.value(), iovecs.data(), iovecs.size());
    DCHECK_LE(bytes, len);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      VLOG(1) << ioMsg("readv", bytes);
      return bytes;
    } else if (bytes < 0) {
      const auto last_errno = errno;
      if (Interrupted(last_errno)) continue;
      DCHECK(!WouldBlock(last_errno));
      LOG(WARNING) << errMsg("readv", last_errno);
      return -1;
    } else {
      DCHECK_EQ(bytes, 0);
      LOG(INFO) << ioMsg("readv no payload", 0);
      return 0;
    }
  }
}

std::string UdpSocket::ToString() const {
  return absl::StrCat("udp socket: ", AddrPortPair(fd_));
}

}  // namespace peregrine::internal

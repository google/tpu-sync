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

#ifndef THIRD_PARTY_PEREGRINE_SRC_API_SOCKET_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_API_SOCKET_UTIL_H_

#include <sys/uio.h>

#include <algorithm>
#include <cstddef>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/peregrine/src/api/transport_types.h"
#include "tpu_sync/transport/peregrine/src/internal/base/types.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/socket_util.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/tcp_socket_util.h"
#include "tpu_sync/transport/peregrine/src/internal/util/util.h"

namespace peregrine {

// Writes exactly `len` bytes of data from the `buf` to the socket `fd`.
// Returns OK if all the bytes are sent successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status WriteExact(int fd, const void* buf, size_t len) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  const Byte* const buffer = static_cast<const Byte*>(buf);
  return internal::TcpSocketUtil::Send(internal::fd_t(fd), buffer, len);
}

// Writes all the bytes from the `iovs` to the socket `fd`.
// Returns OK if all the bytes are sent successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status WriteVExact(int fd, absl::Span<const struct iovec> iovs) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  DCHECK(internal::IsValid(iovs));
  const size_t n = iovs.size();
  if ABSL_PREDICT_TRUE (1 <= n && n <= IOV_MAX) {
    return internal::TcpSocketUtil::SendV(internal::fd_t(fd), iovs);
  }
  if (n > IOV_MAX) {
    // IOV_MAX limits one syscall, not the logical byte stream. Pool
    // resharding can expose many small, non-contiguous state fragments.
    while (!iovs.empty()) {
      const size_t count = std::min(iovs.size(), size_t{IOV_MAX});
      const absl::Status status = internal::TcpSocketUtil::SendV(
          internal::fd_t(fd), iovs.subspan(0, count));
      if (!status.ok()) return status;
      iovs = iovs.subspan(count);
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("#iovs=", n));
}

// Reads exactly `len` bytes of data from the socket `fd` into the `buf`.
// Returns OK if all the bytes are received successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status ReadExact(int fd, void* buf, size_t len) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  Byte* const buffer = static_cast<Byte*>(buf);
  return internal::TcpSocketUtil::Recv(internal::fd_t(fd), buffer, len);
}

// Reads from the socket `fd` into the `iovs`.
// Returns OK if all the bytes are received successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status ReadVExact(int fd, absl::Span<const struct iovec> iovs) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  DCHECK(internal::IsValid(iovs));
  const size_t n = iovs.size();
  if ABSL_PREDICT_TRUE (1 <= n && n <= IOV_MAX) {
    return internal::TcpSocketUtil::RecvV(internal::fd_t(fd), iovs);
  }
  if (n > IOV_MAX) {
    // Receive batch boundaries need not match the sender's boundaries.
    while (!iovs.empty()) {
      const size_t count = std::min(iovs.size(), size_t{IOV_MAX});
      const absl::Status status = internal::TcpSocketUtil::RecvV(
          internal::fd_t(fd), iovs.subspan(0, count));
      if (!status.ok()) return status;
      iovs = iovs.subspan(count);
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("#iovs=", n));
}

}  // namespace peregrine

#endif  // THIRD_PARTY_PEREGRINE_SRC_API_SOCKET_UTIL_H_

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

#ifndef TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_
#define TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_

#include <sys/uio.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"

namespace tpu_raiden::transport::lib {

struct ConnectTiming {
  double connect_ms = 0.0;
  double psp_key_exchange_ms = 0.0;
};

// Connects to remote TCP peer with optional local IP binding and optional
// gRPC channel for TCP-over-PSP out-of-band key exchange. The connect times
// out after a fixed bound.
absl::StatusOr<int> ConnectToPeer(
    absl::string_view peer, absl::string_view local_ip = "",
    bool require_psp = false, std::shared_ptr<grpc::Channel> channel = nullptr,
    ConnectTiming* timing = nullptr);

// Reads exactly `len` bytes from `fd` into `buf`. When `timeout` is `nullopt`,
// delegates to `::peregrine::ReadExact`; otherwise bounds the total wait by
// `*timeout` and returns DeadlineExceededError on timeout.
absl::Status ReadExactWithTimeout(int fd, void* buf, size_t len,
                                  std::optional<absl::Duration> timeout);

// Reads all bytes described by `iovs` from `fd`. When `timeout` is `nullopt`,
// delegates to `::peregrine::ReadVExact`; otherwise bounds the total wait by
// `*timeout` and returns DeadlineExceededError on timeout.
absl::Status ReadVExactWithTimeout(int fd, absl::Span<const struct iovec> iovs,
                                   std::optional<absl::Duration> timeout);

// Writes exactly `len` bytes from `buf` to `fd`. When `timeout` is `nullopt`,
// delegates to `::peregrine::WriteExact`; otherwise bounds the total wait by
// `*timeout` and returns DeadlineExceededError on timeout.
absl::Status WriteExactWithTimeout(int fd, const void* buf, size_t len,
                                   std::optional<absl::Duration> timeout);

// Writes all bytes described by `iovs` to `fd`. When `timeout` is `nullopt`,
// delegates to `::peregrine::WriteVExact`; otherwise bounds the total wait by
// `*timeout` and returns DeadlineExceededError on timeout.
absl::Status WriteVExactWithTimeout(int fd, absl::Span<const struct iovec> iovs,
                                    std::optional<absl::Duration> timeout);

// Returns the local endpoint ("ip:port") for the socket `fd`.
std::string GetLocalEndpoint(int fd);

// Returns a string of self/peer ip:port pair ("self_ip:port <> peer_ip:port")
// for the socket `fd`.
std::string GetAddrPortPair(int fd);

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_

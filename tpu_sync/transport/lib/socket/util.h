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

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"

namespace tpu_raiden::transport::lib {

// Bounds blocking I/O on a connected or accepted data plane socket `fd`.
//
// Two kernel mechanisms cover two different ways a peer stops answering.
// SO_SNDTIMEO and SO_RCVTIMEO bound one blocking send() or recv(), which
// covers a peer that holds the connection open but stops reading. TCP
// keepalive covers the opposite case: an idle pooled connection whose peer
// vanished without sending FIN, where no data is in flight for a send
// deadline to expire on. TCP_USER_TIMEOUT bounds how long the kernel
// retransmits unacknowledged data before failing the connection, which is
// otherwise governed by tcp_retries2 and takes on the order of fifteen
// minutes.
//
// Best effort. An option the kernel rejects is logged and skipped, leaving
// that socket with whatever bounds did apply. Does nothing when
// --raiden_transport_io_timeout is zero.
void ApplyDataPlaneSocketOptions(int fd);

// Connects to remote TCP peer with optional local IP binding and optional
// gRPC channel for TCP-over-PSP out-of-band key exchange.
//
// The connect attempt is bounded by --raiden_transport_connect_timeout, and
// the returned socket carries the options ApplyDataPlaneSocketOptions sets.
// The socket is returned in blocking mode, which the send and recv paths
// assert on.
absl::StatusOr<int> ConnectToPeer(
    absl::string_view peer, absl::string_view local_ip = "",
    bool require_psp = false,
    std::shared_ptr<grpc::Channel> channel = nullptr);

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_

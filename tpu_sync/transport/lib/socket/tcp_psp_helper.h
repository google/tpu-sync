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

#ifndef TPU_SYNC_TRANSPORT_LIB_SOCKET_TCP_PSP_HELPER_H_
#define TPU_SYNC_TRANSPORT_LIB_SOCKET_TCP_PSP_HELPER_H_

#include <sys/socket.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/declare.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"

ABSL_DECLARE_FLAG(bool, require_psp_tcp);

namespace tpu_raiden::transport::lib {

namespace internal {
void SetPspTcpSyscallsForTesting(void* psp_sys);
}  // namespace internal

// Structure representing an exchanged PSP key.
struct PspPeerKey {
  uint32_t spi = 0;
  std::string key;
};

// Returns true if PSP-TCP is supported.
bool IsPspSupported();

// Registers a client's PSP key on server_fd and returns the allocated server
// RX key.
absl::StatusOr<PspPeerKey> RegisterPspPeerKey(
    int server_fd, uint32_t client_spi, absl::string_view client_key);

// Returns true if the accepted socket has valid PSP encryption active
// according to GetInitialRxSpi.
bool PspEnabled(int client_fd);

// Performs out-of-band PSP key exchange on sock_fd using the provided gRPC
// channel and connects to addr.
absl::Status TcpPspConnect(
    int sock_fd, const struct sockaddr* addr, socklen_t addrlen,
    std::shared_ptr<grpc::Channel> channel);

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_SOCKET_TCP_PSP_HELPER_H_

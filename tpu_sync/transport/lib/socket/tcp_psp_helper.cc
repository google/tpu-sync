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

#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

#include <sys/socket.h>

#include <cstdint>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"

ABSL_FLAG(bool, require_psp_tcp, false,
          "Whether to require PSP-TCP encryption for TCP transport connections."
          " This flag is not needed in Google Cloud since the traffic is "
          "automatically encrypted by default(https://cloud.google.com/docs/"
          "security/encryption-in-transit).");

namespace tpu_raiden::transport::lib {

bool IsPspSupported() { return false; }

absl::StatusOr<PspPeerKey> RegisterPspPeerKey(
    int server_fd, uint32_t client_spi, absl::string_view client_key) {
  return absl::UnimplementedError("PSP-TCP is unimplemented.");
}

bool PspEnabled(int client_fd) { return false; }

absl::Status TcpPspConnect(
    int sock_fd, const struct sockaddr* addr, socklen_t addrlen,
    std::shared_ptr<grpc::Channel> channel) {
  return absl::UnimplementedError("PSP-TCP is unimplemented.");
}

}  // namespace tpu_raiden::transport::lib

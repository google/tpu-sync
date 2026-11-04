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

#include "tpu_sync/transport/lib/psp/tcp_psp_helper.h"

#include <sys/socket.h>

#include <cstdint>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"
#include "peregrine/src/api/v1/psp/psp.h"

ABSL_FLAG(bool, require_psp_tcp, false,
          "Whether to require PSP-TCP encryption for TCP transport connections."
          " This flag is not needed in Google Cloud since the traffic is "
          "automatically encrypted by default(https://cloud.google.com/docs/"
          "security/encryption-in-transit).");

namespace tpu_raiden::transport::lib {

bool IsPspSupported() { return false; }

bool IsPspEnabled(int client_fd) { return false; }

absl::StatusOr<::peregrine::v1::PspToken> RegisterPeerPspToken(
    int server_fd, uint32_t client_spi, absl::string_view client_key,
    absl::string_view peer_target) {
  return absl::UnimplementedError("PSP-TCP is unimplemented.");
}

absl::Status TcpPspConnect(
    int sock_fd, const struct sockaddr* addr, socklen_t addrlen,
    std::shared_ptr<grpc::Channel> channel, double* psp_kex_ms,
    double* tcp_connect_ms) {
  return absl::UnimplementedError("PSP-TCP is unimplemented.");
}

}  // namespace tpu_raiden::transport::lib

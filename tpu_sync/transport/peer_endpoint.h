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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_PEER_ENDPOINT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_PEER_ENDPOINT_H_

#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden {
namespace transport {

// Formats an IPv4 or IPv6 literal and port into `ip:port` or `[ip]:port`.
inline std::string FormatIpPort(absl::string_view ip, int port) {
  if (absl::StrContains(ip, ':') && !absl::StartsWith(ip, "[")) {
    return absl::StrCat("[", ip, "]:", port);
  }
  return absl::StrCat(ip, ":", port);
}

// Identifies a remote peer's block-transport endpoints across backends:
// `data` is the canonical socket data endpoint (`ip:data_port`), and
// `peregrine` is the Peregrine control endpoint (`ip:peregrine_port`).
//
// TODO(swasthi): this pair exists only for the transition period in which the
// socket and Peregrine backends coexist per op (TPU_RAIDEN_ENABLE_PEREGRINE).
// Once every transfer runs on Peregrine, collapse it back to a single
// endpoint string so that one listener port (`local_port`) is the sole source
// of truth for all transfers.
struct PeerEndpoint {
  std::string data;       // ip:data_port
  std::string peregrine;  // ip:peregrine_port, empty when not advertised
  PeerEndpoint(std::string data_ep = {}, std::string peregrine_ep = {})
      : data(std::move(data_ep)), peregrine(std::move(peregrine_ep)) {}
  PeerEndpoint(const char* data_ep) : data(data_ep) {}

  friend bool operator==(const PeerEndpoint& a, const PeerEndpoint& b) {
    return a.data == b.data && a.peregrine == b.peregrine;
  }
  friend bool operator<(const PeerEndpoint& a, const PeerEndpoint& b) {
    return std::tie(a.data, a.peregrine) < std::tie(b.data, b.peregrine);
  }
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_PEER_ENDPOINT_H_

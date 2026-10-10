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
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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

// True when a socket bound to `source_ip` and connected to `peer`
// ("host:port" or "[v6]:port") would egress through the interface that owns
// `source_ip`, per the kernel's route lookup (RTM_GETROUTE with RTA_DST and
// RTA_SRC, honouring policy-routing rules). False when the kernel would route
// the peer via a different interface -- binding `source_ip` would then send
// packets with a source address foreign to the egress link, which cloud
// anti-spoofing drops. Loopback source and loopback peer are trivially true.
// Errors: `peer` unparsable / unresolvable, `source_ip` not an address of any
// local interface, or netlink failure.
//
// Results are memoized for the lifetime of the process keyed by (source_ip,
// peer) exactly as passed; host routing is assumed static for the job. Errors
// are not memoized, so a transient failure is retried on the next call.
absl::StatusOr<bool> SourceIpRoutesToPeer(absl::string_view source_ip,
                                          absl::string_view peer);

// Test-only: drops every memoized SourceIpRoutesToPeer result.
void ClearSourceIpRouteCacheForTesting();

// Test-only: number of memoized SourceIpRoutesToPeer results.
size_t SourceIpRouteCacheSizeForTesting();

// Whether to pin the client-side source address of outbound data connections
// (env TPU_RAIDEN_ENABLE_SOURCE_IP_BIND is "1", "true", "yes" or "on").
//
// Binding a source IP only steers egress onto a particular NIC when the host
// has a policy routing rule for that address (`ip rule from <src> ...`). That
// setup exists on multi-NIC deployments where peer addresses are not covered
// by specific routes and would otherwise all leave via the control-plane NIC.
//
// Without such a rule the kernel still routes by destination, so the bound
// address disagrees with the interface the packet leaves on, and cloud
// anti-spoofing filters drop it. bind() itself succeeds and only the data path
// breaks, so the failure surfaces as a hang rather than an error. Callers must
// therefore only bind a source when the kernel's own route lookup confirms
// that the source's NIC is the egress for that peer (SourceIpRoutesToPeer).
// Binding stays opt-in because the policy routing that makes it useful is a
// property of the host.
bool SourceBindEnabled();

// Returns the local endpoint ("ip:port") for the socket `fd`.
std::string GetLocalEndpoint(int fd);

// Returns a string of self/peer ip:port pair ("self_ip:port <> peer_ip:port")
// for the socket `fd`.
std::string GetAddrPortPair(int fd);

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_

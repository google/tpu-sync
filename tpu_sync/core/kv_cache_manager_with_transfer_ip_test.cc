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

#include <arpa/inet.h>

#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "tpu_sync/core/tcp_control_plane_backend.h"

namespace tpu_raiden {
namespace {

// Mirrors the producer side (ProcessPullStream), which decodes the 16-byte
// consumer_ip back to text with inet_ntop(AF_INET6, ...).
std::string DecodeIpv6Bytes(const uint8_t in[16]) {
  char buf[INET6_ADDRSTRLEN];
  if (inet_ntop(AF_INET6, in, buf, sizeof(buf)) == nullptr) return "";
  return std::string(buf);
}

bool IsAllZero(const uint8_t b[16]) {
  for (int i = 0; i < 16; ++i) {
    if (b[i] != 0) return false;
  }
  return true;
}

// The bug this fix addresses: a bare IPv4 string used to fail AF_INET6 parsing,
// leaving consumer_ip zeroed, which forced the producer onto its getpeername()
// fallback (the control NIC, != the data NIC on a multi-NIC host).
TEST(ConsumerIpEncodingTest, Ipv4IsMappedAndRoundTrips) {
  uint8_t buf[16];
  EXPECT_TRUE(TcpControlPlaneBackend::EncodeIpToIpv6Bytes("10.210.0.4", buf));
  EXPECT_FALSE(IsAllZero(buf));
  // The producer decodes it back as IPv4-mapped IPv6 -- not zeroed.
  EXPECT_EQ(DecodeIpv6Bytes(buf), "::ffff:10.210.0.4");
}

TEST(ConsumerIpEncodingTest, Ipv4MappedInputPassesThrough) {
  uint8_t buf[16];
  EXPECT_TRUE(
      TcpControlPlaneBackend::EncodeIpToIpv6Bytes("::ffff:10.210.0.4", buf));
  EXPECT_EQ(DecodeIpv6Bytes(buf), "::ffff:10.210.0.4");
}

TEST(ConsumerIpEncodingTest, Ipv6PassesThrough) {
  uint8_t buf[16];
  EXPECT_TRUE(TcpControlPlaneBackend::EncodeIpToIpv6Bytes("fd00::1", buf));
  EXPECT_EQ(DecodeIpv6Bytes(buf), "fd00::1");
}

TEST(ConsumerIpEncodingTest, UnparseableIsZeroedAndReturnsFalse) {
  uint8_t buf[16];
  std::memset(buf, 0xAB, sizeof(buf));  // poison, to prove it gets zeroed
  EXPECT_FALSE(TcpControlPlaneBackend::EncodeIpToIpv6Bytes("not-an-ip", buf));
  EXPECT_TRUE(IsAllZero(buf));
}

}  // namespace
}  // namespace tpu_raiden
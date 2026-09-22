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

#include "tpu_sync/core/control_plane_backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/grpc_control_plane_backend.h"
#include "tpu_sync/core/tcp_control_plane_backend.h"

namespace tpu_raiden {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class MockControlPlaneHandler : public ControlPlaneHandler {
 public:
  using PullStreamCallback =
      std::function<absl::StatusOr<PullStreamResponseSpec>(
          const PullStreamRequestSpec&, absl::string_view)>;
  using AckCallback = std::function<absl::Status(uint64_t)>;

  void SetPullStreamCallback(PullStreamCallback cb) {
    absl::MutexLock lock(mu_);
    pull_stream_cb_ = std::move(cb);
  }

  void SetAckCallback(AckCallback cb) {
    absl::MutexLock lock(mu_);
    ack_cb_ = std::move(cb);
  }

  absl::StatusOr<PullStreamResponseSpec> OnPullStream(
      const PullStreamRequestSpec& req,
      absl::string_view fallback_peer_ip) override {
    PullStreamCallback cb;
    {
      absl::MutexLock lock(mu_);
      cb = pull_stream_cb_;
    }
    if (cb) {
      return cb(req, fallback_peer_ip);
    }
    return PullStreamResponseSpec{
        .status = 0,
        .num_layers = 8,
        .data_port = 50000,
        .message = "",
    };
  }

  absl::Status OnAck(uint64_t uuid) override {
    AckCallback cb;
    {
      absl::MutexLock lock(mu_);
      cb = ack_cb_;
    }
    if (cb) {
      return cb(uuid);
    }
    return absl::OkStatus();
  }

 private:
  absl::Mutex mu_;
  PullStreamCallback pull_stream_cb_ ABSL_GUARDED_BY(mu_);
  AckCallback ack_cb_ ABSL_GUARDED_BY(mu_);
};

// Simple thread-spawning executor for tests so TCP handlers can run
// concurrently or sleep without blocking the accept loop.
ControlPlaneBackend::TaskExecutor AsyncTestExecutor() {
  return
      [](std::function<void()> task) { std::thread(std::move(task)).detach(); };
}

class ControlPlaneBackendTest
    : public ::testing::TestWithParam<ControlPlaneBackendType> {};

TEST_P(ControlPlaneBackendTest, EphemeralPortBindingAndResolution) {
  MockControlPlaneHandler handler;
  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();
  EXPECT_GT(*bound_port, 0);
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, SendPullRequestWithExplicitConsumerIps) {
  MockControlPlaneHandler handler;
  PullStreamRequestSpec received_req;
  std::string received_fallback_ip;
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        received_req = req;
        received_fallback_ip = std::string(fallback_ip);
        return PullStreamResponseSpec{
            .status = 0,
            .num_layers = 16,
            .data_port = 55000,
            .message = "",
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  PullStreamRequestSpec req;
  req.uuid = 123456;
  req.ep_idx = 2;
  req.consumer_data_port = 60000;
  req.consumer_ips = {"10.0.0.1", "10.0.0.2"};
  req.src_block_ids = {10, 20, 30};
  req.dst_block_ids = {100, 200, 300};

  absl::StatusOr<PullStreamResponseSpec> response =
      client->SendPullRequest(endpoint, req, absl::Seconds(5));
  ASSERT_TRUE(response.ok()) << response.status();
  EXPECT_EQ(response->status, 0);
  EXPECT_EQ(response->num_layers, 16u);
  EXPECT_EQ(response->data_port, 55000u);

  EXPECT_EQ(received_req.uuid, 123456u);
  EXPECT_EQ(received_req.ep_idx, 2u);
  EXPECT_EQ(received_req.consumer_data_port, 60000u);
  EXPECT_THAT(received_req.consumer_ips, ElementsAre("10.0.0.1", "10.0.0.2"));
  EXPECT_THAT(received_req.src_block_ids, ElementsAre(10, 20, 30));
  EXPECT_THAT(received_req.dst_block_ids, ElementsAre(100, 200, 300));
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, FallbackPeerIpExtractionWhenConsumerIpsEmpty) {
  MockControlPlaneHandler handler;
  std::string received_fallback_ip;
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        received_fallback_ip = std::string(fallback_ip);
        return PullStreamResponseSpec{
            .status = 0,
            .num_layers = 4,
            .data_port = 50001,
            .message = "",
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  PullStreamRequestSpec req;
  req.uuid = 42;
  req.consumer_data_port = 61000;
  req.consumer_ips = {};
  req.src_block_ids = {1};
  req.dst_block_ids = {2};

  absl::StatusOr<PullStreamResponseSpec> response =
      client->SendPullRequest(endpoint, req, absl::Seconds(5));
  ASSERT_TRUE(response.ok()) << response.status();
  EXPECT_EQ(response->status, 0);
  EXPECT_FALSE(received_fallback_ip.empty());
  EXPECT_TRUE(received_fallback_ip == "127.0.0.1" ||
              received_fallback_ip == "::1")
      << "Unexpected fallback IP: " << received_fallback_ip;
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, ApplicationErrorPropagation) {
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        return PullStreamResponseSpec{
            .status = -1,
            .num_layers = 0,
            .data_port = 0,
            .message = "no read registered for uuid 999",
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  PullStreamRequestSpec req;
  req.uuid = 999;
  req.src_block_ids = {1};
  req.dst_block_ids = {1};

  absl::StatusOr<PullStreamResponseSpec> response =
      client->SendPullRequest(endpoint, req, absl::Seconds(5));
  ASSERT_TRUE(response.ok()) << response.status();
  EXPECT_EQ(response->status, -1);
  EXPECT_THAT(response->message, HasSubstr("no read registered for uuid 999"));
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, SendAckRoundTrip) {
  MockControlPlaneHandler handler;
  std::atomic<uint64_t> acked_uuid{0};
  handler.SetAckCallback([&](uint64_t uuid) {
    acked_uuid.store(uuid);
    return absl::OkStatus();
  });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  absl::Status status = client->SendAck(endpoint, 777888, absl::Seconds(5));
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(acked_uuid.load(), 777888u);

  if (GetParam() == ControlPlaneBackendType::kTcp) {
    // Verify raw TCP kOpAck header routes to OnAck on TCP backend.
    absl::StatusOr<int> fd = TcpControlPlaneBackend::ConnectTcp(endpoint, 5.0);
    ASSERT_TRUE(fd.ok()) << fd.status();
    TcpControlPlaneBackend::ControlRequestHeader req;
    req.magic = TcpControlPlaneBackend::kControlMagic;
    req.op = TcpControlPlaneBackend::kOpAck;
    req.uuid = 999111;
    req.num_blocks = 0;
    const absl::Time deadline = absl::Now() + absl::Seconds(5);
    ASSERT_TRUE(TcpControlPlaneBackend::WriteExact(*fd, &req, sizeof(req),
                                                   deadline)
                    .ok());
    TcpControlPlaneBackend::ControlResponseHeader resp;
    ASSERT_TRUE(TcpControlPlaneBackend::ReadExact(*fd, &resp, sizeof(resp),
                                                  deadline)
                    .ok());
    close(*fd);
    EXPECT_EQ(resp.status, 0);
    EXPECT_EQ(acked_uuid.load(), 999111u);
  }
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, TimeoutWhenServerDelaysBeyondClientTimeout) {
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        absl::SleepFor(absl::Milliseconds(400));
        return PullStreamResponseSpec{
            .status = 0,
            .num_layers = 4,
            .data_port = 50000,
            .message = "",
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  PullStreamRequestSpec req;
  req.uuid = 111;
  req.src_block_ids = {1};
  req.dst_block_ids = {1};

  absl::StatusOr<PullStreamResponseSpec> response =
      client->SendPullRequest(endpoint, req, absl::Milliseconds(80));
  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(absl::IsDeadlineExceeded(response.status())) << response.status();
  server->StopServer();
}

// The `timeout` argument to SendPullRequest is a bound on the whole call, not
// a hint: however the peer behaves, the caller gets its worker back within it.
// Asserted against both backends because the promise belongs to the
// ControlPlaneBackend interface, not to either transport -- gRPC has always
// honoured it via ClientContext::set_deadline, and the TCP backend used to
// apply it per syscall instead, which is not the same bound.
TEST_P(ControlPlaneBackendTest, PullRequestDeadlineBoundsTheWholeCall) {
  constexpr absl::Duration kDeadline = absl::Milliseconds(300);
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        absl::SleepFor(absl::Seconds(3));
        return PullStreamResponseSpec{
            .status = 0,
            .num_layers = 4,
            .data_port = 50000,
            .message = "",
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  PullStreamRequestSpec req;
  req.uuid = 222;
  req.src_block_ids = {1};
  req.dst_block_ids = {1};

  const absl::Time start = absl::Now();
  absl::StatusOr<PullStreamResponseSpec> response =
      client->SendPullRequest(endpoint, req, kDeadline);
  const absl::Duration elapsed = absl::Now() - start;

  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(absl::IsDeadlineExceeded(response.status())) << response.status();
  EXPECT_LT(elapsed, 5 * kDeadline)
      << "SendPullRequest held its caller for " << elapsed
      << " against a deadline of " << kDeadline;
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, ConcurrentRequestsOverSharedBackend) {
  MockControlPlaneHandler handler;
  std::atomic<int> total_pulls{0};
  std::atomic<int> total_acks{0};
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        total_pulls.fetch_add(1);
        return PullStreamResponseSpec{
            .status = 0,
            .num_layers = static_cast<uint32_t>(req.uuid % 100),
            .data_port = 50000,
            .message = "",
        };
      });
  handler.SetAckCallback([&](uint64_t uuid) {
    total_acks.fetch_add(1);
    return absl::OkStatus();
  });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string endpoint = absl::StrCat("127.0.0.1:", *bound_port);

  constexpr int kNumThreads = 8;
  constexpr int kRequestsPerThread = 10;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < kRequestsPerThread; ++i) {
        uint64_t uuid = static_cast<uint64_t>(t * 1000 + i + 1);
        PullStreamRequestSpec req;
        req.uuid = uuid;
        req.src_block_ids = {static_cast<int64_t>(i)};
        req.dst_block_ids = {static_cast<int64_t>(i)};
        absl::StatusOr<PullStreamResponseSpec> resp =
            client->SendPullRequest(endpoint, req, absl::Seconds(5));
        EXPECT_TRUE(resp.ok()) << resp.status();
        if (resp.ok()) {
          EXPECT_EQ(resp->status, 0);
          EXPECT_EQ(resp->num_layers, static_cast<uint32_t>(uuid % 100));
        }

        absl::Status ack_status =
            client->SendAck(endpoint, uuid, absl::Seconds(5));
        EXPECT_TRUE(ack_status.ok()) << ack_status;
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(total_pulls.load(), kNumThreads * kRequestsPerThread);
  EXPECT_EQ(total_acks.load(), kNumThreads * kRequestsPerThread);
  server->StopServer();
}

INSTANTIATE_TEST_SUITE_P(TcpAndGrpc, ControlPlaneBackendTest,
                         ::testing::Values(ControlPlaneBackendType::kTcp,
                                           ControlPlaneBackendType::kGrpc));

// A peer that accepts the connection, swallows the request, and then answers
// one byte per `gap`. With `gap` inside the socket timeout every recv()
// succeeds, so SO_RCVTIMEO never fires however long the answer takes -- only a
// deadline spanning the whole handshake can end the call. TCP-only by nature:
// it speaks the wire protocol rather than the interface.
class DribblingPeer {
 public:
  DribblingPeer(std::vector<uint8_t> payload, size_t immediate,
                absl::Duration gap)
      : payload_(std::move(payload)), immediate_(immediate), gap_(gap) {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_GE(listen_fd_, 0);
    int on = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(
        bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(
        getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    port_ = ntohs(addr.sin_port);
    EXPECT_EQ(listen(listen_fd_, 8), 0);
    thread_ = std::thread([this] { Serve(); });
  }

  ~DribblingPeer() {
    {
      absl::MutexLock lock(mu_);
      stopping_ = true;
      if (client_fd_ >= 0) shutdown(client_fd_, SHUT_RDWR);
    }
    shutdown(listen_fd_, SHUT_RDWR);
    close(listen_fd_);
    thread_.join();
    absl::MutexLock lock(mu_);
    if (client_fd_ >= 0) close(client_fd_);
  }

  std::string endpoint() const { return absl::StrCat("127.0.0.1:", port_); }

  int bytes_sent() {
    absl::MutexLock lock(mu_);
    return bytes_sent_;
  }

 private:
  bool stopping() {
    absl::MutexLock lock(mu_);
    return stopping_;
  }

  void Serve() {
    int client = accept(listen_fd_, nullptr, nullptr);
    if (client < 0) return;
    {
      absl::MutexLock lock(mu_);
      client_fd_ = client;
      if (stopping_) return;
    }
    // Swallow the request; these tests send no block ids.
    TcpControlPlaneBackend::ControlRequestHeader req;
    size_t got = 0;
    while (got < sizeof(req) && !stopping()) {
      ssize_t n =
          read(client, reinterpret_cast<uint8_t*>(&req) + got, sizeof(req) - got);
      if (n <= 0) return;
      got += static_cast<size_t>(n);
    }

    size_t sent = 0;
    if (immediate_ > 0 && !payload_.empty()) {
      sent = std::min(immediate_, payload_.size());
      if (send(client, payload_.data(), sent, MSG_NOSIGNAL) !=
          static_cast<ssize_t>(sent)) {
        return;
      }
      absl::MutexLock lock(mu_);
      bytes_sent_ += static_cast<int>(sent);
    }
    for (; sent < payload_.size() && !stopping(); ++sent) {
      absl::SleepFor(gap_);
      if (send(client, &payload_[sent], 1, MSG_NOSIGNAL) != 1) return;
      absl::MutexLock lock(mu_);
      ++bytes_sent_;
    }
    // Stay connected: the consumer must not be let off by an EOF.
    while (!stopping()) absl::SleepFor(absl::Milliseconds(20));
  }

  const std::vector<uint8_t> payload_;
  const size_t immediate_;
  const absl::Duration gap_;
  int listen_fd_ = -1;
  int port_ = 0;
  int client_fd_ ABSL_GUARDED_BY(mu_) = -1;
  int bytes_sent_ ABSL_GUARDED_BY(mu_) = 0;
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  absl::Mutex mu_;
  std::thread thread_;
};

std::vector<uint8_t> ResponseHeaderBytes(int32_t status, uint64_t message_len) {
  TcpControlPlaneBackend::ControlResponseHeader hdr;
  hdr.magic = TcpControlPlaneBackend::kResponseMagic;
  hdr.status = status;
  hdr.num_layers = 1;
  hdr.data_port = 50000;
  hdr.message_len = message_len;
  std::vector<uint8_t> out(sizeof(hdr));
  std::memcpy(out.data(), &hdr, sizeof(hdr));
  return out;
}

constexpr absl::Duration kDribbleDeadline = absl::Milliseconds(400);
// A quarter of kDribbleDeadline, not a half: at a half a single scheduling
// hiccup pushes one gap past the socket timeout, and the handshake then fails
// early for the wrong reason -- passing the test even against a backend with
// no total deadline at all.
constexpr absl::Duration kDribbleGap = absl::Milliseconds(100);

TEST(TcpControlPlaneDeadlineTest, DeadlineBoundsDribbledResponseHeader) {
  DribblingPeer peer(ResponseHeaderBytes(/*status=*/0, /*message_len=*/0),
                     /*immediate=*/0, kDribbleGap);
  TcpControlPlaneBackend client;
  PullStreamRequestSpec req;
  req.uuid = 4242;

  const absl::Time start = absl::Now();
  absl::StatusOr<PullStreamResponseSpec> response =
      client.SendPullRequest(peer.endpoint(), req, kDribbleDeadline);
  const absl::Duration elapsed = absl::Now() - start;

  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(absl::IsDeadlineExceeded(response.status())) << response.status();
  EXPECT_LT(elapsed, 3 * kDribbleDeadline)
      << "handshake ran " << elapsed << " against a " << kDribbleDeadline
      << " deadline; the peer had sent " << peer.bytes_sent() << " of "
      << sizeof(TcpControlPlaneBackend::ControlResponseHeader)
      << " header bytes, each one arriving inside SO_RCVTIMEO";
}

TEST(TcpControlPlaneDeadlineTest, DeadlineBoundsDribbledErrorBody) {
  // status = 0 with a non-zero message_len. SendPullRequest enters the body
  // read on message_len alone, with no status check, so a peer reporting
  // success reaches it. The body is capped at kMaxControlErrorMessageBytes
  // rather than at the 24-byte header, which is what makes this the dominant
  // term in the unbounded hold time rather than a footnote to the case above.
  constexpr uint64_t kBodyBytes = 64;
  std::vector<uint8_t> payload =
      ResponseHeaderBytes(/*status=*/0, /*message_len=*/kBodyBytes);
  const size_t header_bytes = payload.size();
  payload.insert(payload.end(), kBodyBytes, 'x');

  DribblingPeer peer(std::move(payload), /*immediate=*/header_bytes,
                     kDribbleGap);
  TcpControlPlaneBackend client;
  PullStreamRequestSpec req;
  req.uuid = 4243;

  const absl::Time start = absl::Now();
  absl::StatusOr<PullStreamResponseSpec> response =
      client.SendPullRequest(peer.endpoint(), req, kDribbleDeadline);
  const absl::Duration elapsed = absl::Now() - start;

  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(absl::IsDeadlineExceeded(response.status())) << response.status();
  EXPECT_LT(elapsed, 3 * kDribbleDeadline)
      << "error-body read ran " << elapsed << " against a " << kDribbleDeadline
      << " deadline after the peer declared " << kBodyBytes
      << " bytes and dribbled them";
}

TEST(TcpControlPlaneDeadlineTest, HealthyPeerIsUnaffectedByTheDeadline) {
  // The bound must not cost anything when the peer answers promptly: the whole
  // payload arrives at once, well inside the deadline.
  DribblingPeer peer(ResponseHeaderBytes(/*status=*/0, /*message_len=*/0),
                     /*immediate=*/sizeof(
                         TcpControlPlaneBackend::ControlResponseHeader),
                     kDribbleGap);
  TcpControlPlaneBackend client;
  PullStreamRequestSpec req;
  req.uuid = 4244;

  const absl::Time start = absl::Now();
  absl::StatusOr<PullStreamResponseSpec> response =
      client.SendPullRequest(peer.endpoint(), req, kDribbleDeadline);
  const absl::Duration elapsed = absl::Now() - start;

  ASSERT_TRUE(response.ok()) << response.status();
  EXPECT_EQ(response->status, 0);
  EXPECT_LT(elapsed, kDribbleDeadline);
}

TEST(ExtractIpFromGrpcPeerTest, HandlesAllFormats) {
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv4:10.210.0.4:54321"), "10.210.0.4");
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv6:[::1]:54321"), "::1");
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv6:[2001:db8::1]:54321"), "2001:db8::1");
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv6:%5B::1%5D:54321"), "::1");
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv6:%5b2001:db8::2%5d:54321"),
            "2001:db8::2");
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv6:[::ffff:10.0.0.1]:54321"), "10.0.0.1");
  EXPECT_EQ(ExtractIpFromGrpcPeer("ipv6:%5B::ffff:10.0.0.1%5D:54321"),
            "10.0.0.1");
}

TEST(ResolveControlPlaneBackendTypeTest, EnvVarAndOverridePrecedence) {
  unsetenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND");
  unsetenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE");

  // Default is kTcp
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kTcp);

  // Explicit override takes precedence
  EXPECT_EQ(ResolveControlPlaneBackendType(ControlPlaneBackendType::kGrpc),
            ControlPlaneBackendType::kGrpc);

  // TPU_RAIDEN_CONTROL_PLANE_BACKEND=grpc
  setenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "grpc", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);
  EXPECT_EQ(ResolveControlPlaneBackendType(ControlPlaneBackendType::kTcp),
            ControlPlaneBackendType::kTcp);

  // TPU_RAIDEN_CONTROL_PLANE_BACKEND=tcp
  setenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "tcp", 1);
  setenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kTcp);

  // Fallback to TPU_RAIDEN_USE_GRPC_CONTROL_PLANE=1 when primary env unset
  unsetenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND");
  setenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);

  setenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "true", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);

  unsetenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE");
}

TEST(GrpcControlPlaneBackendLruTest,
     EvictsLeastRecentlyUsedStubWhenCapacityExceeded) {
  MockControlPlaneHandler handler;
  GrpcControlPlaneBackend server1;
  GrpcControlPlaneBackend server2;
  GrpcControlPlaneBackend server3;

  absl::StatusOr<int> port1 = server1.StartServer(0, &handler);
  absl::StatusOr<int> port2 = server2.StartServer(0, &handler);
  absl::StatusOr<int> port3 = server3.StartServer(0, &handler);
  ASSERT_TRUE(port1.ok()) << port1.status();
  ASSERT_TRUE(port2.ok()) << port2.status();
  ASSERT_TRUE(port3.ok()) << port3.status();

  std::string ep1 = absl::StrCat("127.0.0.1:", *port1);
  std::string ep2 = absl::StrCat("127.0.0.1:", *port2);
  std::string ep3 = absl::StrCat("127.0.0.1:", *port3);

  GrpcControlPlaneBackend client(/*max_cached_stubs=*/2);
  EXPECT_EQ(client.TEST_CachedStubCount(), 0u);

  // 1. Send request to ep1, then ep2 -> both cached.
  ASSERT_TRUE(client.SendAck(ep1, 1, absl::Seconds(5)).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 1u);
  EXPECT_TRUE(client.TEST_HasCachedStub(ep1));

  ASSERT_TRUE(client.SendAck(ep2, 2, absl::Seconds(5)).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 2u);
  EXPECT_TRUE(client.TEST_HasCachedStub(ep1));
  EXPECT_TRUE(client.TEST_HasCachedStub(ep2));

  // 2. Access ep1 again (promotes ep1 to MRU, making ep2 LRU).
  ASSERT_TRUE(client.SendAck(ep1, 3, absl::Seconds(5)).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 2u);

  // 3. Send request to ep3 -> ep2 is evicted, ep1 and ep3 remain cached.
  ASSERT_TRUE(client.SendAck(ep3, 4, absl::Seconds(5)).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 2u);
  EXPECT_FALSE(client.TEST_HasCachedStub(ep2));
  EXPECT_TRUE(client.TEST_HasCachedStub(ep1));
  EXPECT_TRUE(client.TEST_HasCachedStub(ep3));

  server1.StopServer();
  server2.StopServer();
  server3.StopServer();
}

TEST(GrpcControlPlaneBackendLruTest, ZeroCapacityBypassesCache) {
  MockControlPlaneHandler handler;
  GrpcControlPlaneBackend server;
  absl::StatusOr<int> port = server.StartServer(0, &handler);
  ASSERT_TRUE(port.ok()) << port.status();

  std::string ep = absl::StrCat("127.0.0.1:", *port);
  GrpcControlPlaneBackend client(/*max_cached_stubs=*/0);

  ASSERT_TRUE(client.SendAck(ep, 100, absl::Seconds(5)).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 0u);
  EXPECT_FALSE(client.TEST_HasCachedStub(ep));

  server.StopServer();
}

}  // namespace
}  // namespace tpu_raiden

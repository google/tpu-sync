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
#include <functional>
#include <memory>
#include <optional>
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
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/grpc_control_plane_backend.h"
#include "tpu_sync/core/tcp_control_plane_backend.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

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

  using CapabilitiesCallback = std::function<ControlPlaneCapabilities()>;

  void SetCapabilitiesCallback(CapabilitiesCallback cb) {
    absl::MutexLock lock(mu_);
    capabilities_cb_ = std::move(cb);
  }

  ControlPlaneCapabilities GetCapabilities() const override {
    CapabilitiesCallback cb;
    {
      absl::MutexLock lock(mu_);
      cb = capabilities_cb_;
    }
    if (cb) {
      return cb();
    }
    return ControlPlaneCapabilities{};
  }

 private:
  mutable absl::Mutex mu_;
  PullStreamCallback pull_stream_cb_ ABSL_GUARDED_BY(mu_);
  AckCallback ack_cb_ ABSL_GUARDED_BY(mu_);
  CapabilitiesCallback capabilities_cb_ ABSL_GUARDED_BY(mu_);
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
      client->SendPullRequest(endpoint, req, absl::Seconds(5)).Await();
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

TEST_P(ControlPlaneBackendTest, SendPullRequestCarriesConsumerHostLayout) {
  MockControlPlaneHandler handler;
  PullStreamRequestSpec received_req;
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        received_req = req;
        return PullStreamResponseSpec{.status = 0};
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> bound_port = server->StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  PullStreamRequestSpec req;
  req.uuid = 7;
  req.consumer_ips = {"10.0.0.1"};
  req.src_block_ids = {1};
  req.dst_block_ids = {2};
  req.layer_host_addrs.resize(2);
  req.layer_host_addrs[0].add_host_base_addrs(0x1000);
  req.layer_host_addrs[0].add_host_base_addrs(0x2000);
  req.layer_host_addrs[0].set_block_stride_bytes(128);
  req.layer_host_addrs[0].set_num_blocks(4);
  req.layer_host_addrs[1].add_host_base_addrs(0x3000);
  req.layer_host_addrs[1].set_block_stride_bytes(256);
  (*req.layer_host_addrs[1].mutable_host_slot_by_block())[5] = 1;

  absl::StatusOr<PullStreamResponseSpec> response =
      client
          ->SendPullRequest(absl::StrCat("127.0.0.1:", *bound_port), req,
                            absl::Seconds(5))
          .Await();
  ASSERT_TRUE(response.ok()) << response.status();
  if (GetParam() == ControlPlaneBackendType::kTcp) {
    EXPECT_TRUE(received_req.layer_host_addrs.empty());
    server->StopServer();
    return;
  }
  ASSERT_EQ(received_req.layer_host_addrs.size(), 2u);
  EXPECT_THAT(received_req.layer_host_addrs[0].host_base_addrs(),
              ElementsAre(0x1000, 0x2000));
  EXPECT_EQ(received_req.layer_host_addrs[0].block_stride_bytes(), 128);
  EXPECT_EQ(received_req.layer_host_addrs[0].num_blocks(), 4);
  EXPECT_THAT(received_req.layer_host_addrs[1].host_base_addrs(),
              ElementsAre(0x3000));
  EXPECT_EQ(received_req.layer_host_addrs[1].block_stride_bytes(), 256);
  EXPECT_EQ(received_req.layer_host_addrs[1].host_slot_by_block().at(5), 1);
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
      client->SendPullRequest(endpoint, req, absl::Seconds(5)).Await();
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
      client->SendPullRequest(endpoint, req, absl::Seconds(5)).Await();
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
    ASSERT_TRUE(
        TcpControlPlaneBackend::WriteExact(*fd, &req, sizeof(req)).ok());
    TcpControlPlaneBackend::ControlResponseHeader resp;
    ASSERT_TRUE(
        TcpControlPlaneBackend::ReadExact(*fd, &resp, sizeof(resp)).ok());
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
      client->SendPullRequest(endpoint, req, absl::Milliseconds(80)).Await();
  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(absl::IsDeadlineExceeded(response.status())) << response.status();
  server->StopServer();
}

// The `timeout` argument to SendPullRequest bounds the whole call, not each
// step inside it: however the peer behaves, the caller gets its worker back
// within it. Asserted against both backends because the promise belongs to the
// ControlPlaneBackend interface rather than to either transport, so a backend
// swap cannot quietly drop it.
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
      client->SendPullRequest(endpoint, req, kDeadline).Await();
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
            client->SendPullRequest(endpoint, req, absl::Seconds(5)).Await();
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

TEST_P(ControlPlaneBackendTest, PullStreamWindowParametersAndAckEcho) {
  MockControlPlaneHandler handler;
  PullStreamRequestSpec received_req;
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view) {
        received_req = req;
        std::string msg;
        if (req.chunk_blocks > 0 && req.window_chunks > 0) {
          msg = FormatStagingWindowAck(req.chunk_blocks, req.window_chunks);
        }
        return PullStreamResponseSpec{
            .status = 0,
            .message = std::move(msg),
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> port = server->StartServer(0, &handler);
  ASSERT_TRUE(port.ok()) << port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  PullStreamRequestSpec req{
      .uuid = 777, .chunk_blocks = 64, .window_chunks = 2};
  auto resp = client
                  ->SendPullRequest(absl::StrCat("127.0.0.1:", *port), req,
                                    absl::Seconds(5))
                  .Await();
  ASSERT_TRUE(resp.ok()) << resp.status();
  EXPECT_EQ(resp->status, 0);
  EXPECT_EQ(resp->message, "staging_window=64,2");
  EXPECT_EQ(received_req.chunk_blocks, 64);
  EXPECT_EQ(received_req.window_chunks, 2);
  EXPECT_EQ(ParseStagingWindowAck(resp->message), std::make_pair(64, 2));

  PullStreamRequestSpec legacy_req{
      .uuid = 778, .chunk_blocks = 0, .window_chunks = 0};
  auto legacy_resp = client
                         ->SendPullRequest(absl::StrCat("127.0.0.1:", *port),
                                           legacy_req, absl::Seconds(5))
                         .Await();
  ASSERT_TRUE(legacy_resp.ok()) << legacy_resp.status();
  EXPECT_EQ(legacy_resp->status, 0);
  EXPECT_TRUE(legacy_resp->message.empty());
  EXPECT_EQ(received_req.chunk_blocks, 0);
  EXPECT_EQ(received_req.window_chunks, 0);
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, StagingWindowRejectedDelivery) {
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [](const PullStreamRequestSpec&, absl::string_view) {
        return PullStreamResponseSpec{
            .status = -1,
            .message = FormatStagingWindowRejected("serve_disabled"),
        };
      });

  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> port = server->StartServer(0, &handler);
  ASSERT_TRUE(port.ok()) << port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  PullStreamRequestSpec req{
      .uuid = 888, .chunk_blocks = 64, .window_chunks = 2};
  auto resp = client
                  ->SendPullRequest(absl::StrCat("127.0.0.1:", *port), req,
                                    absl::Seconds(5))
                  .Await();
  ASSERT_TRUE(resp.ok()) << resp.status();
  EXPECT_NE(resp->status, 0);
  EXPECT_EQ(ParseStagingWindowRejectedReason(resp->message), "serve_disabled");
  server->StopServer();
}

TEST_P(ControlPlaneBackendTest, CapabilitiesDefaultAndCustomHandler) {
  MockControlPlaneHandler handler;
  auto server = CreateControlPlaneBackend(GetParam(), AsyncTestExecutor());
  absl::StatusOr<int> port = server->StartServer(0, &handler);
  ASSERT_TRUE(port.ok()) << port.status();

  auto client = CreateControlPlaneBackend(GetParam());
  std::string ep = absl::StrCat("127.0.0.1:", *port);

  auto def_caps = client->GetCapabilities(ep, absl::Seconds(5)).Await();
  ASSERT_TRUE(def_caps.ok()) << def_caps.status();
  EXPECT_FALSE(def_caps->serve_staging_window);
  EXPECT_EQ(def_caps->max_ring_blocks, 0);

  handler.SetCapabilitiesCallback([] {
    return ControlPlaneCapabilities{.serve_staging_window = true,
                                    .max_ring_blocks = 512};
  });
  auto custom_caps = client->GetCapabilities(ep, absl::Seconds(5)).Await();
  ASSERT_TRUE(custom_caps.ok()) << custom_caps.status();
  EXPECT_TRUE(custom_caps->serve_staging_window);
  EXPECT_EQ(custom_caps->max_ring_blocks, 512);
  server->StopServer();
}

INSTANTIATE_TEST_SUITE_P(TcpAndGrpc, ControlPlaneBackendTest,
                         ::testing::Values(ControlPlaneBackendType::kTcp,
                                           ControlPlaneBackendType::kGrpc));

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

  // Default is kGrpc
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);

  // Explicit override takes precedence
  EXPECT_EQ(ResolveControlPlaneBackendType(ControlPlaneBackendType::kTcp),
            ControlPlaneBackendType::kTcp);

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

// The point of the async path: SendPullRequest returns while the peer is
// still sitting on the handshake, so the caller's thread is not held.
TEST(GrpcControlPlaneBackendAsyncTest, SendPullRequestReturnsBeforeResponse) {
  absl::Notification release;
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        release.WaitForNotificationWithTimeout(absl::Seconds(10));
        return PullStreamResponseSpec{};
      });
  GrpcControlPlaneBackend server;
  absl::StatusOr<int> bound_port = server.StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  GrpcControlPlaneBackend client;
  PullStreamRequestSpec req;
  req.uuid = 9;
  const absl::Time start = absl::Now();
  tsl::Future<PullStreamResponseSpec> response = client.SendPullRequest(
      absl::StrCat("127.0.0.1:", *bound_port), req, absl::Seconds(10));
  EXPECT_LT(absl::Now() - start, absl::Seconds(1))
      << "SendPullRequest blocked its caller";
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_FALSE(response.IsReady());

  release.Notify();
  EXPECT_TRUE(response.Await().ok()) << response.Await().status();
  server.StopServer();
}

// Destroying the backend must not leave a callback to run against freed
// state: it cancels outstanding pulls and waits for their callbacks.
TEST(GrpcControlPlaneBackendAsyncTest, DestructionCancelsPendingPull) {
  absl::Notification release;
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [&](const PullStreamRequestSpec& req, absl::string_view fallback_ip) {
        release.WaitForNotificationWithTimeout(absl::Seconds(10));
        return PullStreamResponseSpec{};
      });
  GrpcControlPlaneBackend server;
  absl::StatusOr<int> bound_port = server.StartServer(0, &handler);
  ASSERT_TRUE(bound_port.ok()) << bound_port.status();

  auto client = std::make_unique<GrpcControlPlaneBackend>();
  PullStreamRequestSpec req;
  req.uuid = 10;
  tsl::Future<PullStreamResponseSpec> response = client->SendPullRequest(
      absl::StrCat("127.0.0.1:", *bound_port), req, absl::Seconds(10));
  std::atomic<bool> callback_ran{false};
  response.OnReady([&](const absl::StatusOr<PullStreamResponseSpec>&) {
    callback_ran = true;
  });
  absl::SleepFor(absl::Milliseconds(100));

  const absl::Time start = absl::Now();
  client.reset();
  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
  EXPECT_TRUE(callback_ran) << "destructor returned before the callback ran";
  ASSERT_TRUE(response.IsReady());
  EXPECT_TRUE(absl::IsCancelled(response.Await().status()))
      << response.Await().status();

  release.Notify();
  server.StopServer();
}

TEST(TcpControlPlaneWireTest, PaddingPackUnpackRoundTrip) {
  int32_t c = 0, w = 0;
  uint32_t pad = TcpControlPlaneBackend::PackPullStreamPadding(64, 2);
  EXPECT_NE(pad & (1u << 24), 0u);
  TcpControlPlaneBackend::UnpackPullStreamPadding(pad, &c, &w);
  EXPECT_EQ(c, 64);
  EXPECT_EQ(w, 2);

  uint32_t max_pad = TcpControlPlaneBackend::PackPullStreamPadding(65535, 255);
  TcpControlPlaneBackend::UnpackPullStreamPadding(max_pad, &c, &w);
  EXPECT_EQ(c, 65535);
  EXPECT_EQ(w, 255);

  EXPECT_EQ(TcpControlPlaneBackend::PackPullStreamPadding(64, 0), 0u);
  EXPECT_EQ(TcpControlPlaneBackend::PackPullStreamPadding(64, -1), 0u);
  EXPECT_EQ(TcpControlPlaneBackend::PackPullStreamPadding(0, 2), 0u);
  EXPECT_EQ(TcpControlPlaneBackend::PackPullStreamPadding(-1, 2), 0u);
  EXPECT_EQ(TcpControlPlaneBackend::PackPullStreamPadding(65536, 2), 0u);
  EXPECT_EQ(TcpControlPlaneBackend::PackPullStreamPadding(64, 256), 0u);
  TcpControlPlaneBackend::UnpackPullStreamPadding(0, &c, &w);
  EXPECT_EQ(c, 0);
  EXPECT_EQ(w, 0);
  TcpControlPlaneBackend::UnpackPullStreamPadding(0x00020040u, &c, &w);
  EXPECT_EQ(c, 0);
  EXPECT_EQ(w, 0);
  TcpControlPlaneBackend::UnpackPullStreamPadding((1u << 24) | 64u, &c, &w);
  EXPECT_EQ(c, 0);
  EXPECT_EQ(w, 0);
  TcpControlPlaneBackend::UnpackPullStreamPadding((1u << 24) | (2u << 16), &c,
                                                  &w);
  EXPECT_EQ(c, 0);
  EXPECT_EQ(w, 0);
}

TEST(ControlPlaneMarkerAndAckHelpersTest, FormatAndParseHelpers) {
  EXPECT_EQ(FormatStagingWindowRejected("serve_disabled"),
            "staging_window_rejected:serve_disabled");
  EXPECT_EQ(ParseStagingWindowRejectedReason(
                "staging_window_rejected:serve_disabled"),
            "serve_disabled");
  EXPECT_EQ(ParseStagingWindowRejectedReason(
                "staging_window_rejected:ring_exceeds_capacity"),
            "ring_exceeds_capacity");
  EXPECT_EQ(ParseStagingWindowRejectedReason("staging_window_rejected:"),
            std::nullopt);
  EXPECT_EQ(ParseStagingWindowRejectedReason("unrelated error"), std::nullopt);

  EXPECT_EQ(FormatStagingWindowAck(64, 2), "staging_window=64,2");
  EXPECT_EQ(ParseStagingWindowAck("staging_window=64,2"),
            std::make_pair(64, 2));
  EXPECT_EQ(ParseStagingWindowAck("staging_window=0,2"), std::nullopt);
  EXPECT_EQ(ParseStagingWindowAck("staging_window=64,0"), std::nullopt);
  EXPECT_EQ(ParseStagingWindowAck("not_ack_64,2"), std::nullopt);

  EXPECT_EQ(FormatCapabilitiesReply({.serve_staging_window = false}), "caps=");
  EXPECT_EQ(FormatCapabilitiesReply(
                {.serve_staging_window = true, .max_ring_blocks = 512}),
            "caps=staging_window:serve;max_ring_blocks=512");
  auto parsed =
      ParseCapabilitiesReply("caps=staging_window:serve;max_ring_blocks=512");
  ASSERT_TRUE(parsed.ok());
  EXPECT_TRUE(parsed->serve_staging_window);
  EXPECT_EQ(parsed->max_ring_blocks, 512);
  EXPECT_FALSE(ParseCapabilitiesReply("no_prefix").ok());
  EXPECT_FALSE(ParseCapabilitiesReply("caps=max_ring_blocks=bad").ok());
  EXPECT_FALSE(ParseCapabilitiesReply("caps=max_ring_blocks=-1").ok());

  EXPECT_TRUE(IsCapabilitiesUnsupportedStatus(
      absl::UnimplementedError("unimplemented")));
  EXPECT_TRUE(IsCapabilitiesUnsupportedStatus(
      absl::InternalError("unknown control op code: 4")));
  EXPECT_FALSE(IsCapabilitiesUnsupportedStatus(absl::OkStatus()));
}

TEST(TcpControlPlaneBackendTest, OldClientIgnoresSuccessMessageAndNoDesync) {
  MockControlPlaneHandler handler;
  handler.SetPullStreamCallback(
      [](const PullStreamRequestSpec&, absl::string_view) {
        return PullStreamResponseSpec{.status = 0,
                                      .message = "staging_window=64,2"};
      });
  TcpControlPlaneBackend server(AsyncTestExecutor());
  absl::StatusOr<int> port = server.StartServer(0, &handler);
  ASSERT_TRUE(port.ok()) << port.status();

  TcpControlPlaneBackend client;
  std::string ep = absl::StrCat("127.0.0.1:", *port);
  PullStreamRequestSpec req{.uuid = 42};
  auto resp = client.SendPullRequest(ep, req, absl::Seconds(5)).Await();
  ASSERT_TRUE(resp.ok()) << resp.status();
  EXPECT_EQ(resp->status, 0);
  EXPECT_EQ(resp->message, "staging_window=64,2");
  EXPECT_TRUE(client.SendAck(ep, 42, absl::Seconds(5)).ok());

  absl::StatusOr<int> fd = TcpControlPlaneBackend::ConnectTcp(ep, 5.0);
  ASSERT_TRUE(fd.ok()) << fd.status();
  TcpControlPlaneBackend::ControlRequestHeader legacy_req{
      .magic = TcpControlPlaneBackend::kControlMagic,
      .op = TcpControlPlaneBackend::kOpPullStream,
      .uuid = 43,
  };
  ASSERT_TRUE(
      TcpControlPlaneBackend::WriteExact(*fd, &legacy_req, sizeof(legacy_req))
          .ok());
  TcpControlPlaneBackend::ControlResponseHeader legacy_resp_hdr;
  ASSERT_TRUE(TcpControlPlaneBackend::ReadExact(*fd, &legacy_resp_hdr,
                                                sizeof(legacy_resp_hdr))
                  .ok());
  EXPECT_EQ(legacy_resp_hdr.status, 0);
  EXPECT_EQ(legacy_resp_hdr.message_len, 19);
  close(*fd);

  EXPECT_TRUE(client.SendAck(ep, 43, absl::Seconds(5)).ok());
  server.StopServer();
}

TEST(TcpControlPlaneBackendTest, UnknownOpReturnsUnimplemented) {
  MockControlPlaneHandler handler;
  TcpControlPlaneBackend server(AsyncTestExecutor());
  absl::StatusOr<int> port = server.StartServer(0, &handler);
  ASSERT_TRUE(port.ok()) << port.status();

  absl::StatusOr<int> fd = TcpControlPlaneBackend::ConnectTcp(
      absl::StrCat("127.0.0.1:", *port), 5.0);
  ASSERT_TRUE(fd.ok()) << fd.status();

  TcpControlPlaneBackend::ControlRequestHeader req{
      .magic = TcpControlPlaneBackend::kControlMagic, .op = 99};
  ASSERT_TRUE(TcpControlPlaneBackend::WriteExact(*fd, &req, sizeof(req)).ok());

  TcpControlPlaneBackend::ControlResponseHeader resp;
  ASSERT_TRUE(TcpControlPlaneBackend::ReadExact(*fd, &resp, sizeof(resp)).ok());
  EXPECT_EQ(resp.status, -1);
  std::string msg(resp.message_len, '\0');
  ASSERT_TRUE(
      TcpControlPlaneBackend::ReadExact(*fd, msg.data(), msg.size()).ok());
  EXPECT_THAT(msg, HasSubstr("unknown control op code: 99"));
  close(*fd);
  server.StopServer();
}

TEST(CapabilitiesUnsupportedFallbackTest, TcpOldServerReturnsUnsupported) {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(listen_fd, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ASSERT_EQ(bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)),
            0);
  ASSERT_EQ(listen(listen_fd, 1), 0);
  socklen_t addr_len = sizeof(addr);
  ASSERT_EQ(
      getsockname(listen_fd, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  int port = ntohs(addr.sin_port);

  std::thread server_thread([listen_fd]() {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr),
                           &client_len);
    if (client_fd >= 0) {
      TcpControlPlaneBackend::ControlRequestHeader req{};
      if (TcpControlPlaneBackend::ReadExact(client_fd, &req, sizeof(req))
              .ok()) {
        std::string err_msg = absl::StrCat("unknown control op code: ", req.op);
        TcpControlPlaneBackend::ControlResponseHeader resp{
            .magic = TcpControlPlaneBackend::kResponseMagic,
            .status = -1,
            .message_len = static_cast<uint32_t>(err_msg.size()),
        };
        if (TcpControlPlaneBackend::WriteExact(client_fd, &resp, sizeof(resp))
                .ok()) {
          (void)TcpControlPlaneBackend::WriteExact(client_fd, err_msg.data(),
                                                   err_msg.size());
        }
      }
      close(client_fd);
    }
    close(listen_fd);
  });

  TcpControlPlaneBackend client;
  auto caps =
      client.GetCapabilities(absl::StrCat("127.0.0.1:", port), absl::Seconds(5))
          .Await();
  EXPECT_FALSE(caps.ok());
  EXPECT_TRUE(absl::IsUnimplemented(caps.status())) << caps.status();
  EXPECT_TRUE(IsCapabilitiesUnsupportedStatus(caps.status()));

  server_thread.join();
}

TEST(CapabilitiesUnsupportedFallbackTest,
     GrpcMissingServiceReturnsUnsupported) {
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                           &port);
  control_plane::proto::KVCacheControlPlaneService::Service service;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);

  GrpcControlPlaneBackend client;
  auto caps =
      client.GetCapabilities(absl::StrCat("127.0.0.1:", port), absl::Seconds(5))
          .Await();
  EXPECT_FALSE(caps.ok());
  EXPECT_TRUE(absl::IsUnimplemented(caps.status())) << caps.status();
  EXPECT_TRUE(IsCapabilitiesUnsupportedStatus(caps.status()));

  server->Shutdown();
}

TEST(CapabilitiesUnsupportedFallbackTest,
     GetCapabilitiesBlockingTranslatesUnsupported) {
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                           &port);
  control_plane::proto::KVCacheControlPlaneService::Service service;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);

  auto caps = GrpcControlPlaneBackend().GetCapabilitiesBlocking(
      absl::StrCat("127.0.0.1:", port), absl::Seconds(5));
  ASSERT_TRUE(caps.ok()) << caps.status();
  EXPECT_FALSE(caps->supported);
  EXPECT_FALSE(caps->serve_staging_window);
  EXPECT_EQ(caps->max_ring_blocks, 0);
  server->Shutdown();
}

}  // namespace
}  // namespace tpu_raiden

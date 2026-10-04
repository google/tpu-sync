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

#include "tpu_sync/weight_sync/manager/v3/async_control_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using control_pipe::proto::ControlEnvelope;
using control_pipe::proto::ControlResponseEnvelope;

ControlResponseEnvelope Echo(const ControlEnvelope& request) {
  ControlResponseEnvelope response;
  response.set_payload(absl::StrCat("echo:", request.payload()));
  return response;
}

ControlEnvelope Request(uint64_t request_id, std::string payload) {
  ControlEnvelope env;
  env.set_message_type("test.Echo");
  env.set_request_id(request_id);
  env.set_payload(std::move(payload));
  return env;
}

std::unique_ptr<ControlPipeClient> TcpClient() {
  ControlPipeConfig config;
  config.backend_type = ControlPipeBackendType::kTcp;
  return CreateControlPipeClient(config);
}

// A blocking client socket speaking the envelope framing directly.
class RawConnection {
 public:
  explicit RawConnection(int port) : fd_(socket(AF_INET, SOCK_STREAM, 0)) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    connected_ =
        connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
  }
  ~RawConnection() { close(fd_); }

  bool connected() const { return connected_; }

  void SendBytes(const std::string& bytes) {
    ASSERT_EQ(send(fd_, bytes.data(), bytes.size(), MSG_NOSIGNAL),
              static_cast<ssize_t>(bytes.size()));
  }

  static std::string Frame(const ControlEnvelope& env) {
    const std::string bytes = env.SerializeAsString();
    const uint32_t net_len = htonl(static_cast<uint32_t>(bytes.size()));
    std::string out("CPIP");
    out.append(reinterpret_cast<const char*>(&net_len), sizeof(net_len));
    return out + bytes;
  }

  // Reads one response frame, or fails on EOF.
  absl::StatusOr<ControlResponseEnvelope> Read() {
    char header[8];
    if (!ReadExact(header, sizeof(header))) {
      return absl::UnavailableError("connection closed");
    }
    if (std::memcmp(header, "PIPC", 4) != 0) {
      return absl::InternalError("bad magic");
    }
    uint32_t net_len = 0;
    std::memcpy(&net_len, header + 4, sizeof(net_len));
    std::string body(ntohl(net_len), '\0');
    if (!ReadExact(body.data(), body.size())) {
      return absl::UnavailableError("connection closed");
    }
    ControlResponseEnvelope response;
    if (!response.ParseFromString(body)) {
      return absl::InternalError("bad envelope");
    }
    return response;
  }

 private:
  bool ReadExact(char* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
      const ssize_t n = recv(fd_, buf + done, len - done, 0);
      if (n <= 0) return false;
      done += static_cast<size_t>(n);
    }
    return true;
  }

  int fd_;
  bool connected_ = false;
};

TEST(EpollControlServerTest, ServesControlPipeClients) {
  std::unique_ptr<AsyncControlServer> server = CreateEpollControlServer(
      {.num_threads = 2}, [](const ControlContext& ctx, ControlEnvelope request,
                             AsyncControlReply reply) {
        EXPECT_EQ(ctx.request_id, request.request_id());
        std::move(reply)(Echo(request));
      });
  absl::StatusOr<int> port = server->Start(0);
  ASSERT_OK(port);
  EXPECT_EQ(server->bound_port(), *port);
  EXPECT_EQ(server->Start(0).status().code(),
            absl::StatusCode::kFailedPrecondition);

  std::unique_ptr<ControlPipeClient> client = TcpClient();
  const std::string endpoint = absl::StrCat("127.0.0.1:", *port);
  for (int i = 0; i < 20; ++i) {
    absl::StatusOr<ControlResponseEnvelope> response = client->SendRaw(
        endpoint, Request(i + 1, absl::StrCat("m", i)), absl::Seconds(10));
    ASSERT_OK(response);
    EXPECT_EQ(response->request_id(), i + 1);
    EXPECT_EQ(response->payload(), absl::StrCat("echo:m", i));
  }
  const AsyncControlServer::Stats stats = server->GetStats();
  EXPECT_EQ(stats.requests, 20);
  EXPECT_EQ(stats.responses, 20);
  EXPECT_GE(stats.accepted_connections, 1);
  server->Stop();
}

// Requests wait without holding a thread: one event-loop thread keeps many
// requests pending and answers them later, from another thread, in any
// order.
TEST(EpollControlServerTest, ParkedRequestsHoldNoThread) {
  absl::Mutex mu;
  std::vector<std::pair<ControlEnvelope, AsyncControlReply>> parked;
  std::unique_ptr<AsyncControlServer> server = CreateEpollControlServer(
      {.num_threads = 1}, [&](const ControlContext&, ControlEnvelope request,
                              AsyncControlReply reply) {
        absl::MutexLock lock(mu);
        parked.emplace_back(std::move(request), std::move(reply));
      });
  absl::StatusOr<int> port = server->Start(0);
  ASSERT_OK(port);

  constexpr int kClients = 64;
  std::vector<std::thread> clients;
  absl::Mutex results_mu;
  int ok = 0;
  for (int i = 0; i < kClients; ++i) {
    clients.emplace_back([&, i] {
      std::unique_ptr<ControlPipeClient> client = TcpClient();
      absl::StatusOr<ControlResponseEnvelope> response =
          client->SendRaw(absl::StrCat("127.0.0.1:", *port),
                          Request(i + 1, absl::StrCat(i)), absl::Seconds(30));
      if (response.ok() && response->payload() == absl::StrCat("echo:", i)) {
        absl::MutexLock lock(results_mu);
        ++ok;
      }
    });
  }
  {
    absl::MutexLock lock(mu);
    auto all_parked = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu) {
      return parked.size() == kClients;
    };
    ASSERT_TRUE(
        mu.AwaitWithTimeout(absl::Condition(&all_parked), absl::Seconds(30)));
  }
  // Answer from this thread, newest first.
  std::vector<std::pair<ControlEnvelope, AsyncControlReply>> to_answer;
  {
    absl::MutexLock lock(mu);
    to_answer.swap(parked);
  }
  for (auto it = to_answer.rbegin(); it != to_answer.rend(); ++it) {
    std::move(it->second)(Echo(it->first));
  }
  for (std::thread& t : clients) t.join();
  EXPECT_EQ(ok, kClients);
  EXPECT_GE(server->GetStats().accepted_connections, kClients);
  server->Stop();
  EXPECT_EQ(server->GetStats().responses, kClients);
}

TEST(EpollControlServerTest, PipelinedRequestsAreAnsweredOutOfOrder) {
  absl::Mutex mu;
  std::vector<std::pair<ControlEnvelope, AsyncControlReply>> parked;
  std::unique_ptr<AsyncControlServer> server = CreateEpollControlServer(
      {}, [&](const ControlContext&, ControlEnvelope request,
              AsyncControlReply reply) {
        absl::MutexLock lock(mu);
        parked.emplace_back(std::move(request), std::move(reply));
      });
  absl::StatusOr<int> port = server->Start(0);
  ASSERT_OK(port);
  RawConnection conn(*port);
  ASSERT_TRUE(conn.connected());
  // Two frames in one write, the second split across two writes.
  const std::string second = RawConnection::Frame(Request(2, "b"));
  conn.SendBytes(RawConnection::Frame(Request(1, "a")) + second.substr(0, 5));
  conn.SendBytes(second.substr(5));
  {
    absl::MutexLock lock(mu);
    auto both = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu) {
      return parked.size() == 2;
    };
    ASSERT_TRUE(mu.AwaitWithTimeout(absl::Condition(&both), absl::Seconds(10)));
  }
  std::move(parked[1].second)(Echo(parked[1].first));
  absl::StatusOr<ControlResponseEnvelope> first = conn.Read();
  ASSERT_OK(first);
  EXPECT_EQ(first->request_id(), 2);
  EXPECT_EQ(first->payload(), "echo:b");
  std::move(parked[0].second)(Echo(parked[0].first));
  absl::StatusOr<ControlResponseEnvelope> next = conn.Read();
  ASSERT_OK(next);
  EXPECT_EQ(next->request_id(), 1);
  server->Stop();
}

TEST(EpollControlServerTest, ClosesConnectionsWithMalformedFrames) {
  std::unique_ptr<AsyncControlServer> server = CreateEpollControlServer(
      {.max_frame_bytes = 1024},
      [](const ControlContext&, ControlEnvelope request,
         AsyncControlReply reply) { std::move(reply)(Echo(request)); });
  absl::StatusOr<int> port = server->Start(0);
  ASSERT_OK(port);
  {
    RawConnection conn(*port);
    conn.SendBytes("JUNKJUNK");
    EXPECT_EQ(conn.Read().status().code(), absl::StatusCode::kUnavailable);
  }
  {
    RawConnection conn(*port);
    conn.SendBytes(RawConnection::Frame(Request(1, std::string(2000, 'x'))));
    EXPECT_EQ(conn.Read().status().code(), absl::StatusCode::kUnavailable);
  }
  // The server still serves new connections.
  RawConnection conn(*port);
  conn.SendBytes(RawConnection::Frame(Request(7, "ok")));
  absl::StatusOr<ControlResponseEnvelope> response = conn.Read();
  ASSERT_OK(response);
  EXPECT_EQ(response->payload(), "echo:ok");
  server->Stop();
}

TEST(EpollControlServerTest, RepliesAfterStopAreDropped) {
  absl::Mutex mu;
  std::vector<AsyncControlReply> parked;
  std::unique_ptr<AsyncControlServer> server = CreateEpollControlServer(
      {}, [&](const ControlContext&, ControlEnvelope, AsyncControlReply reply) {
        absl::MutexLock lock(mu);
        parked.push_back(std::move(reply));
      });
  absl::StatusOr<int> port = server->Start(0);
  ASSERT_OK(port);
  RawConnection conn(*port);
  conn.SendBytes(RawConnection::Frame(Request(1, "late")));
  {
    absl::MutexLock lock(mu);
    auto one = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu) {
      return parked.size() == 1;
    };
    ASSERT_TRUE(mu.AwaitWithTimeout(absl::Condition(&one), absl::Seconds(10)));
  }
  server->Stop();
  EXPECT_EQ(conn.Read().status().code(), absl::StatusCode::kUnavailable);
  std::move(parked[0])(ControlResponseEnvelope());
  EXPECT_EQ(server->GetStats().responses, 0);
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

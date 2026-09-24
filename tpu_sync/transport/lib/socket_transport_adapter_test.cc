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

#include "tpu_sync/transport/lib/socket_transport_adapter.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"
#include "tpu_sync/transport/peregrine/src/api/socket_util.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Gt;

auto MatchDirectionLabel(absl::string_view dir) {
  return ElementsAre(telemetry::MetricLabel{
      .key = telemetry::metric_labels::kDirection, .value = dir});
}

auto MatchBytesLabels(absl::string_view direction, absl::string_view src_ip,
                      absl::string_view dst_ip) {
  return ElementsAre(
      telemetry::MetricLabel{.key = telemetry::metric_labels::kDirection,
                             .value = direction},
      telemetry::MetricLabel{.key = telemetry::metric_labels::kSrcIp,
                             .value = src_ip},
      telemetry::MetricLabel{.key = telemetry::metric_labels::kDstIp,
                             .value = dst_ip});
}

auto MatchP2pTimeLabels(absl::string_view src_ip, absl::string_view dst_ip) {
  return ElementsAre(
      telemetry::MetricLabel{.key = telemetry::metric_labels::kSrcIp,
                             .value = src_ip},
      telemetry::MetricLabel{.key = telemetry::metric_labels::kDstIp,
                             .value = dst_ip});
}

std::string GetIpPort(const RawBufferTransport& transport) {
  return absl::StrCat("127.0.0.1:", transport.local_port());
}

TEST(SocketTransportAdapterTest, PostWithEmptyRequestsFails) {
  RawBufferTransport raw_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter adapter(&raw_transport, /*parallelism=*/1);

  auto result = adapter.Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/{},
      /*src_block_ids=*/{}, /*dst_block_ids=*/{}, /*on_complete=*/nullptr);

  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SocketTransportAdapterTest, PostWithInvalidParallelismFails) {
  RawBufferTransport raw_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter adapter(&raw_transport, /*parallelism=*/1);

  Request req = {};
  req.socket_opcode = 1;
  req.parallelism = -1;

  auto result = adapter.Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{}, /*on_complete=*/nullptr);

  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SocketTransportAdapterTest, PostWithUnsupportedOpcodeFails) {
  RawBufferTransport raw_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter adapter(&raw_transport, /*parallelism=*/1);

  Request req = {};
  req.socket_opcode = 99;  // Unsupported opcode.

  auto result = adapter.Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{}, /*on_complete=*/nullptr);

  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(SocketTransportAdapterTest, PollReturnsUnimplemented) {
  RawBufferTransport raw_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter adapter(&raw_transport, /*parallelism=*/1);
  EXPECT_THAT(adapter.Poll(0).status(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(SocketTransportAdapterTest, PostSocketPushOp1Success) {
  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    if (header.op != 1) {
      return absl::InvalidArgumentError("Expected op 1");
    }
    std::vector<int> allocated_ids(header.count_or_size);
    for (size_t i = 0; i < header.count_or_size; ++i) {
      allocated_ids[i] = static_cast<int>(100 + i);
    }
    const std::vector<uint8_t> s_ids = SerializeBlockIds(allocated_ids);
    if (auto s = ::peregrine::WriteExact(client_fd, s_ids.data(), s_ids.size());
        !s.ok()) {
      return s;
    }

    for (size_t i = 0; i < header.count_or_size; ++i) {
      uint8_t size_buf[kChunkSizeFieldSize];
      if (auto s =
              ::peregrine::ReadExact(client_fd, size_buf, sizeof(size_buf));
          !s.ok()) {
        return s;
      }
      const uint32_t chunk_size = DeserializeChunkSize(size_buf);
      std::vector<uint8_t> payload(chunk_size);
      if (chunk_size > 0) {
        if (auto s = ::peregrine::ReadExact(client_fd, payload.data(),
                                            payload.size());
            !s.ok()) {
          return s;
        }
      }
    }

    uint8_t ack = 1;
    return ::peregrine::WriteExact(client_fd, &ack, 1);
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);

  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> test_data = {1, 2, 3, 4, 5, 6, 7, 8};
  Request req = {};
  req.socket_opcode = 1;
  req.laddr = test_data.data();
  req.len = test_data.size();
  req.count_or_size = 1;
  req.uuid = 42;
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;

  absl::Notification done;
  absl::StatusOr<std::vector<int>> push_result;
  const int src_bid = 10;
  auto handle = client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_bid, 1),
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> res) {
        push_result = std::move(res);
        done.Notify();
      });

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  done.WaitForNotification();
  ASSERT_THAT(push_result, IsOkAndHolds(::testing::ElementsAre(100)));
}

TEST(SocketTransportAdapterTest, PostSocketPushOp6ExplicitPushSuccess) {
  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    if (header.op != 6) {
      return absl::InvalidArgumentError("Expected op 6");
    }
    const size_t count = header.count_or_size;
    std::vector<uint8_t> ids_buf(count * sizeof(uint32_t));
    if (auto s =
            ::peregrine::ReadExact(client_fd, ids_buf.data(), ids_buf.size());
        !s.ok()) {
      return s;
    }
    if (auto s =
            ::peregrine::ReadExact(client_fd, ids_buf.data(), ids_buf.size());
        !s.ok()) {
      return s;
    }
    uint8_t ack = 1;
    if (auto s = ::peregrine::WriteExact(client_fd, &ack, 1); !s.ok()) {
      return s;
    }

    for (size_t i = 0; i < count; ++i) {
      uint8_t size_buf[kChunkSizeFieldSize];
      if (auto s =
              ::peregrine::ReadExact(client_fd, size_buf, sizeof(size_buf));
          !s.ok()) {
        return s;
      }
      const uint32_t chunk_size = DeserializeChunkSize(size_buf);
      std::vector<uint8_t> payload(chunk_size);
      if (chunk_size > 0) {
        if (auto s = ::peregrine::ReadExact(client_fd, payload.data(),
                                            payload.size());
            !s.ok()) {
          return s;
        }
      }
    }

    return ::peregrine::WriteExact(client_fd, &ack, 1);
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);

  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> test_data = {10, 20, 30, 40};
  Request req = {};
  req.socket_opcode = 6;
  req.laddr = test_data.data();
  req.len = test_data.size();
  req.count_or_size = 1;
  req.uuid = 84;
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;

  absl::Notification done;
  absl::StatusOr<std::vector<int>> push_result;
  const int src_bid = 5;
  const int dst_bid = 55;
  auto handle = client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_bid, 1),
      /*dst_block_ids=*/absl::MakeConstSpan(&dst_bid, 1),
      [&](absl::StatusOr<std::vector<int>> res) {
        push_result = std::move(res);
        done.Notify();
      });

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  done.WaitForNotification();
  ASSERT_THAT(push_result, IsOkAndHolds(::testing::ElementsAre(55)));
}

TEST(SocketTransportAdapterTest, PostSocketPullOp2Success) {
  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    if (header.op != 2) {
      return absl::InvalidArgumentError("Expected op 2");
    }
    // Echo back pull response header with identical count_or_size and flags.
    ChunkHeader resp = header;
    const auto s_resp = SerializeChunkHeader(resp);
    if (auto s =
            ::peregrine::WriteExact(client_fd, s_resp.data(), s_resp.size());
        !s.ok()) {
      return s;
    }

    std::vector<uint8_t> payload = {1, 2, 3, 4};
    const auto s_size = SerializeChunkSize(payload.size());
    if (auto s =
            ::peregrine::WriteExact(client_fd, s_size.data(), s_size.size());
        !s.ok()) {
      return s;
    }
    return ::peregrine::WriteExact(client_fd, payload.data(), payload.size());
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> recv_buf(4, 0);
  Request req = {};
  req.socket_opcode = 2;
  req.laddr = recv_buf.data();
  req.len = recv_buf.size();
  req.count_or_size = 1;
  req.remote_id = 10;
  req.local_id = 20;
  req.uuid = 99;
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;

  auto handle = client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&req, 1));

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  EXPECT_THAT(recv_buf, ::testing::ElementsAre(1, 2, 3, 4));
}

TEST(SocketTransportAdapterTest,
     PostSocketPushMultiStreamBatchBarrierTelemetry) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  telemetry::ScopedMetricsBackendReset reset(std::move(mock_backend));

  // Expect exactly 1 observation at the barrier for all streams in the pair.
  EXPECT_CALL(
      *raw_mock,
      ObserveHistogram(telemetry::metric_names::kP2pTransferTimeMs,
                       MatchP2pTimeLabels("127.0.0.2", "127.0.0.1"), Gt(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              IncrementCounter(
                  telemetry::metric_names::kSentBytesTotal,
                  MatchBytesLabels(telemetry::metric_labels::kDirectionPush,
                                   "127.0.0.2", "127.0.0.1"),
                  8))
      .Times(2);

  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    if (header.op != 1) {
      return absl::InvalidArgumentError("Expected op 1");
    }
    std::vector<int> allocated_ids(header.count_or_size, 200);
    const std::vector<uint8_t> serialized_ids =
        SerializeBlockIds(allocated_ids);
    if (absl::Status status = peregrine::WriteExact(
            client_fd, serialized_ids.data(), serialized_ids.size());
        !status.ok()) {
      return status;
    }

    for (size_t i = 0; i < header.count_or_size; ++i) {
      uint8_t size_buf[kChunkSizeFieldSize];
      if (absl::Status status =
              peregrine::ReadExact(client_fd, size_buf, sizeof(size_buf));
          !status.ok()) {
        return status;
      }
      const uint32_t chunk_size = DeserializeChunkSize(size_buf);
      std::vector<uint8_t> payload(chunk_size);
      if (chunk_size > 0) {
        if (absl::Status status =
                peregrine::ReadExact(client_fd, payload.data(), payload.size());
            !status.ok()) {
          return status;
        }
      }
    }

    uint8_t ack = 1;
    return peregrine::WriteExact(client_fd, &ack, 1);
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{"127.0.0.2"});
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/2);

  std::vector<uint8_t> test_data(8, 1);
  std::vector<Request> requests(2);
  for (int i = 0; i < 2; ++i) {
    requests[i].socket_opcode = 1;
    requests[i].laddr = test_data.data();
    requests[i].len = test_data.size();
    requests[i].count_or_size = 1;
    requests[i].uuid = 500;
    requests[i].parallelism = 2;
    requests[i].request_id = i;
    requests[i].stream_idx = i;
  }

  absl::Notification done;
  absl::StatusOr<std::vector<int>> push_result;
  std::vector<int> src_block_ids = {10, 11};
  ABSL_ASSERT_OK(client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/requests,
      /*src_block_ids=*/src_block_ids,
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> result) {
        push_result = std::move(result);
        done.Notify();
      }));

  done.WaitForNotification();
  EXPECT_THAT(push_result, IsOkAndHolds(ElementsAre(200, 200)));
}

TEST(SocketTransportAdapterTest,
     PostSocketPullMultiStreamBatchBarrierTelemetry) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  telemetry::ScopedMetricsBackendReset reset(std::move(mock_backend));

  // Expect exactly 1 observation at the barrier for all streams in the pair.
  EXPECT_CALL(
      *raw_mock,
      ObserveHistogram(telemetry::metric_names::kP2pTransferTimeMs,
                       MatchP2pTimeLabels("127.0.0.1", "127.0.0.2"), Gt(0.0)))
      .Times(1);
  EXPECT_CALL(
      *raw_mock,
      IncrementCounter(
          telemetry::metric_names::kReceivedBytesTotal,
          MatchDirectionLabel(telemetry::metric_labels::kDirectionPullResponse),
          4))
      .Times(2);

  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    if (header.op != 2) {
      return absl::InvalidArgumentError("Expected op 2");
    }
    const ChunkHeader response = header;
    const absl::InlinedVector<char, kChunkHeaderSize> serialized_header =
        SerializeChunkHeader(response);
    if (absl::Status status = peregrine::WriteExact(
            client_fd, serialized_header.data(), serialized_header.size());
        !status.ok()) {
      return status;
    }

    std::vector<uint8_t> payload = {1, 2, 3, 4};
    const std::array<uint8_t, kChunkSizeFieldSize> serialized_size =
        SerializeChunkSize(payload.size());
    if (absl::Status status = peregrine::WriteExact(
            client_fd, serialized_size.data(), serialized_size.size());
        !status.ok()) {
      return status;
    }
    return peregrine::WriteExact(client_fd, payload.data(), payload.size());
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{"127.0.0.2"});
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/2);

  std::vector<uint8_t> recv_buf0(4, 0);
  std::vector<uint8_t> recv_buf1(4, 0);
  std::vector<Request> requests(2);
  for (int i = 0; i < 2; ++i) {
    requests[i].socket_opcode = 2;
    requests[i].laddr = (i == 0) ? recv_buf0.data() : recv_buf1.data();
    requests[i].len = 4;
    requests[i].count_or_size = 1;
    requests[i].remote_id = 10 + i;
    requests[i].local_id = 20 + i;
    requests[i].uuid = 501;
    requests[i].parallelism = 2;
    requests[i].request_id = i;
    requests[i].stream_idx = i;
  }

  ABSL_ASSERT_OK(client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/requests));
  EXPECT_THAT(recv_buf0, ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(recv_buf1, ElementsAre(1, 2, 3, 4));
}

std::string GetPeerIp(int client_fd) {
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (getpeername(client_fd, reinterpret_cast<struct sockaddr*>(&addr),
                  &addr_len) != 0) {
    return "";
  }
  char ip_str[INET6_ADDRSTRLEN] = {0};
  if (addr.ss_family == AF_INET) {
    auto* sin = reinterpret_cast<struct sockaddr_in*>(&addr);
    inet_ntop(AF_INET, &sin->sin_addr, ip_str, sizeof(ip_str));
  } else if (addr.ss_family == AF_INET6) {
    auto* sin6 = reinterpret_cast<struct sockaddr_in6*>(&addr);
    inet_ntop(AF_INET6, &sin6->sin6_addr, ip_str, sizeof(ip_str));
  }
  return std::string(ip_str);
}

TEST(SocketTransportAdapterTest, SourceBindDisabledWhenEnvUnsetOrZero) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND"); });

  unsetenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND");
  EXPECT_FALSE(SourceBindEnabled());
  EXPECT_EQ(SelectSourceIp({"10.0.0.1", "10.0.0.2"}, 0), "");

  setenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND", "0", 1);
  EXPECT_FALSE(SourceBindEnabled());
  EXPECT_EQ(SelectSourceIp({"10.0.0.1", "10.0.0.2"}, 0), "");

  std::string observed_peer_ip;
  auto server_handler = [&](int client_fd,
                            const ChunkHeader& header) -> absl::Status {
    observed_peer_ip = GetPeerIp(client_fd);
    ChunkHeader resp = header;
    const auto s_resp = SerializeChunkHeader(resp);
    if (auto s =
            ::peregrine::WriteExact(client_fd, s_resp.data(), s_resp.size());
        !s.ok()) {
      return s;
    }
    std::vector<uint8_t> payload = {9, 8, 7, 6};
    const auto s_size = SerializeChunkSize(payload.size());
    if (auto s =
            ::peregrine::WriteExact(client_fd, s_size.data(), s_size.size());
        !s.ok()) {
      return s;
    }
    return ::peregrine::WriteExact(client_fd, payload.data(), payload.size());
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  // Provide "127.0.0.2" in local_ips. Since source IP binding is disabled
  // ("0"), SelectSourceIp returns "", so ConnectToPeer does NOT bind to
  // 127.0.0.2 and the kernel uses 127.0.0.1 instead.
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{"127.0.0.2"});
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> recv_buf(4, 0);
  Request req = {};
  req.socket_opcode = 2;
  req.laddr = recv_buf.data();
  req.len = recv_buf.size();
  req.count_or_size = 1;
  req.remote_id = 10;
  req.local_id = 20;
  req.uuid = 101;
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;

  auto handle = client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&req, 1));

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  EXPECT_THAT(recv_buf, ::testing::ElementsAre(9, 8, 7, 6));
  EXPECT_THAT(observed_peer_ip, ::testing::HasSubstr("127.0.0.1"));
}

TEST(SocketTransportAdapterTest, SourceBindEnabledWithEnableSourceIpBindEnv) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND"); });

  setenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND", "1", 1);
  EXPECT_TRUE(SourceBindEnabled());
  EXPECT_EQ(SelectSourceIp({"10.0.0.1", "10.0.0.2"}, 0), "10.0.0.1");
  EXPECT_EQ(SelectSourceIp({"10.0.0.1", "10.0.0.2"}, 1), "10.0.0.2");

  setenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND", "true", 1);
  EXPECT_TRUE(SourceBindEnabled());
  EXPECT_EQ(SelectSourceIp({"10.0.0.1", "10.0.0.2"}, 0), "10.0.0.1");

  std::string observed_peer_ip;
  auto server_handler = [&](int client_fd,
                            const ChunkHeader& header) -> absl::Status {
    observed_peer_ip = GetPeerIp(client_fd);
    ChunkHeader resp = header;
    const auto s_resp = SerializeChunkHeader(resp);
    if (auto s =
            ::peregrine::WriteExact(client_fd, s_resp.data(), s_resp.size());
        !s.ok()) {
      return s;
    }
    std::vector<uint8_t> payload = {5, 6, 7, 8};
    const auto s_size = SerializeChunkSize(payload.size());
    if (auto s =
            ::peregrine::WriteExact(client_fd, s_size.data(), s_size.size());
        !s.ok()) {
      return s;
    }
    return ::peregrine::WriteExact(client_fd, payload.data(), payload.size());
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);

  // 1. When local_ips contains "127.0.0.2" and
  // TPU_RAIDEN_ENABLE_SOURCE_IP_BIND is "1", ConnectToPeer binds to
  // "127.0.0.2" and the server sees peer IP "127.0.0.2".
  {
    setenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND", "1", 1);
    RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                        /*local_ips=*/{"127.0.0.2"});
    SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

    std::vector<uint8_t> recv_buf(4, 0);
    Request req = {};
    req.socket_opcode = 2;
    req.laddr = recv_buf.data();
    req.len = recv_buf.size();
    req.count_or_size = 1;
    req.remote_id = 10;
    req.local_id = 20;
    req.uuid = 102;
    req.parallelism = 1;
    req.request_id = 0;
    req.stream_idx = 0;

    auto handle = client_adapter.Post(
        /*peers=*/{GetIpPort(server_transport)},
        /*requests=*/absl::MakeConstSpan(&req, 1));

    ASSERT_THAT(handle.status(), absl_testing::IsOk());
    EXPECT_THAT(recv_buf, ::testing::ElementsAre(5, 6, 7, 8));
    EXPECT_THAT(observed_peer_ip, ::testing::HasSubstr("127.0.0.2"));
  }

  // 2. When local_ips contains "127.0.0.3" and TPU_RAIDEN_ENABLE_SOURCE_IP_BIND
  // is "true", ConnectToPeer binds to "127.0.0.3" and the server sees
  // "127.0.0.3".
  {
    setenv("TPU_RAIDEN_ENABLE_SOURCE_IP_BIND", "true", 1);
    RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                        /*local_ips=*/{"127.0.0.3"});
    SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

    std::vector<uint8_t> recv_buf(4, 0);
    Request req = {};
    req.socket_opcode = 2;
    req.laddr = recv_buf.data();
    req.len = recv_buf.size();
    req.count_or_size = 1;
    req.remote_id = 10;
    req.local_id = 20;
    req.uuid = 103;
    req.parallelism = 1;
    req.request_id = 0;
    req.stream_idx = 0;

    auto handle = client_adapter.Post(
        /*peers=*/{GetIpPort(server_transport)},
        /*requests=*/absl::MakeConstSpan(&req, 1));

    ASSERT_THAT(handle.status(), absl_testing::IsOk());
    EXPECT_THAT(recv_buf, ::testing::ElementsAre(5, 6, 7, 8));
    EXPECT_THAT(observed_peer_ip, ::testing::HasSubstr("127.0.0.3"));
  }
}

}  // namespace
}  // namespace tpu_raiden::transport::lib

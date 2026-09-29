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
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>  // NOLINT
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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "peregrine/src/api/socket_util.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
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

auto MatchP2pTimeLabels(absl::string_view src_ip, absl::string_view dst_ip,
                        absl::string_view local_rank = "0") {
  return ElementsAre(
      telemetry::MetricLabel{.key = telemetry::metric_labels::kSrcIp,
                             .value = src_ip},
      telemetry::MetricLabel{.key = telemetry::metric_labels::kDstIp,
                             .value = dst_ip},
      telemetry::MetricLabel{.key = telemetry::metric_labels::kLocalRank,
                             .value = local_rank});
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
  req.buffer_id = (uint64_t{3} << 32) | 5;  // layer 3, shard 5.
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;
  size_t got_layer = 0, got_shard = 0, got_size = 0;
  int got_block = -1;
  req.on_block_received = [&](size_t layer, size_t shard, int block_id,
                              size_t size) {
    got_layer = layer;
    got_shard = shard;
    got_block = block_id;
    got_size = size;
    return absl::OkStatus();
  };

  auto handle = client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&req, 1));

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  EXPECT_THAT(recv_buf, ::testing::ElementsAre(1, 2, 3, 4));
  EXPECT_EQ(got_layer, 3);
  EXPECT_EQ(got_shard, 5);
  EXPECT_EQ(got_block, 20);
  EXPECT_EQ(got_size, 4);
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

TEST(SocketTransportAdapterTest,
     PostSocketPushOverlappingConcurrentTransfersClampDuration) {
  setenv("LOCAL_RANK", "2", 1);
  absl::Cleanup unset_local_rank = [] { unsetenv("LOCAL_RANK"); };

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  telemetry::ScopedMetricsBackendReset reset(std::move(mock_backend));

  std::vector<double> observed_durations_ms;
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kP2pTransferTimeMs,
                               MatchP2pTimeLabels("127.0.0.9", "127.0.0.1",
                                                  /*local_rank=*/"2"),
                               _))
      .Times(2)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const telemetry::MetricLabel>,
              double value) { observed_durations_ms.push_back(value); });
  EXPECT_CALL(*raw_mock,
              IncrementCounter(
                  telemetry::metric_names::kSentBytesTotal,
                  MatchBytesLabels(telemetry::metric_labels::kDirectionPush,
                                   "127.0.0.9", "127.0.0.1"),
                  8))
      .Times(2);

  absl::Notification push1_in_flight;
  absl::Notification push2_in_flight;
  absl::Notification release_push1;
  absl::Notification release_push2;

  auto server_handler = [&](int client_fd,
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

    if (header.uuid == 601) {
      push1_in_flight.Notify();
      release_push1.WaitForNotification();
    } else if (header.uuid == 602) {
      push2_in_flight.Notify();
      release_push2.WaitForNotification();
    }

    uint8_t ack = 1;
    return peregrine::WriteExact(client_fd, &ack, 1);
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{"127.0.0.9"});
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/2);

  std::vector<uint8_t> test_data(8, 1);
  auto make_push_request = [&](uint64_t uuid) {
    Request request = {};
    request.socket_opcode = 1;
    request.laddr = test_data.data();
    request.len = test_data.size();
    request.count_or_size = 1;
    request.uuid = uuid;
    request.parallelism = 1;
    request.request_id = 0;
    request.stream_idx = 0;
    return request;
  };

  const Request push1_request = make_push_request(601);
  const Request push2_request = make_push_request(602);
  const int src_block_id = 10;

  absl::Notification push1_done;
  absl::Notification push2_done;
  absl::StatusOr<std::vector<int>> push1_result;
  absl::StatusOr<std::vector<int>> push2_result;

  const std::chrono::steady_clock::time_point wall_start =
      std::chrono::steady_clock::now();
  ABSL_ASSERT_OK(client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&push1_request, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_block_id, 1),
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> result) {
        push1_result = std::move(result);
        push1_done.Notify();
      }));
  push1_in_flight.WaitForNotification();

  ABSL_ASSERT_OK(client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&push2_request, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_block_id, 1),
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> result) {
        push2_result = std::move(result);
        push2_done.Notify();
      }));
  push2_in_flight.WaitForNotification();

  // Keep both transfers concurrently in flight for 50ms so their intervals
  // overlap by at least 50ms.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Complete Push 1 first (setting busy_until = end_ts_1), then complete
  // Push 2 immediately afterward.
  release_push1.Notify();
  push1_done.WaitForNotification();

  release_push2.Notify();
  push2_done.WaitForNotification();
  const std::chrono::steady_clock::time_point wall_end =
      std::chrono::steady_clock::now();
  const double total_wall_ms =
      absl::ToDoubleMilliseconds(absl::FromChrono(wall_end - wall_start));

  EXPECT_THAT(push1_result, IsOkAndHolds(ElementsAre(200)));
  EXPECT_THAT(push2_result, IsOkAndHolds(ElementsAre(200)));
  ASSERT_EQ(observed_durations_ms.size(), 2);
  EXPECT_GE(observed_durations_ms[0], 45.0);
  // Because Push 2's effective start is clamped to Push 1's end timestamp,
  // the sum of both recorded durations cannot exceed total elapsed wall time.
  EXPECT_LE(observed_durations_ms[0] + observed_durations_ms[1], total_wall_ms);
  EXPECT_LT(observed_durations_ms[1], observed_durations_ms[0]);
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

TEST(SocketTransportAdapterTest, DefaultTimeoutsWhenEnvUnset) {
  auto cleanup = absl::MakeCleanup([] {
    unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S");
    unsetenv("TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S");
  });
  unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S");
  unsetenv("TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S");

  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);
  EXPECT_EQ(client_adapter.handshake_ack_read_timeout(), absl::Seconds(5));
  EXPECT_EQ(client_adapter.final_ack_read_timeout(), absl::Seconds(60));
}

TEST(SocketTransportAdapterTest, InvalidEnvFallsBackToDefault) {
  auto cleanup = absl::MakeCleanup([] {
    unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S");
    unsetenv("TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S");
  });
  setenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S", "invalid_number",
         1);
  setenv("TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S", "-10.0", 1);

  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);
  EXPECT_EQ(client_adapter.handshake_ack_read_timeout(), absl::Seconds(5));
  EXPECT_EQ(client_adapter.final_ack_read_timeout(), absl::Seconds(60));
}

TEST(SocketTransportAdapterTest, PushTimesOutWhenHandshakeNeverResponds) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S"); });
  setenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S", "0.1", 1);

  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    absl::SleepFor(absl::Milliseconds(500));
    return absl::OkStatus();
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> test_data = {1, 2, 3, 4};
  Request req = {};
  req.socket_opcode = 6;
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
  const int dst_bid = 20;
  const absl::Time start = absl::Now();
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
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_FALSE(push_result.ok());
  EXPECT_THAT(push_result.status(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       ::testing::HasSubstr("timed out")));
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
}

TEST(SocketTransportAdapterTest, PushTimesOutWhenFinalAckNeverResponds) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S"); });
  setenv("TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S", "0.1", 1);

  // Server receives handshake and payload properly, but never sends final ack.
  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    const size_t count = header.count_or_size;
    std::vector<uint8_t> ids_buf(count * sizeof(uint32_t));
    ABSL_RETURN_IF_ERROR(
        ::peregrine::ReadExact(client_fd, ids_buf.data(), ids_buf.size()));
    ABSL_RETURN_IF_ERROR(
        ::peregrine::ReadExact(client_fd, ids_buf.data(), ids_buf.size()));
    uint8_t ack = 1;
    ABSL_RETURN_IF_ERROR(::peregrine::WriteExact(client_fd, &ack, 1));

    uint8_t size_buf[kChunkSizeFieldSize];
    ABSL_RETURN_IF_ERROR(
        ::peregrine::ReadExact(client_fd, size_buf, sizeof(size_buf)));
    const uint32_t chunk_size = DeserializeChunkSize(size_buf);
    std::vector<uint8_t> payload(chunk_size);
    if (chunk_size > 0) {
      ABSL_RETURN_IF_ERROR(
          ::peregrine::ReadExact(client_fd, payload.data(), payload.size()));
    }

    // Hang before sending final ack.
    absl::SleepFor(absl::Milliseconds(500));
    return absl::OkStatus();
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> test_data = {1, 2, 3, 4};
  Request req = {};
  req.socket_opcode = 6;
  req.laddr = test_data.data();
  req.len = test_data.size();
  req.count_or_size = 1;
  req.uuid = 43;
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;

  absl::Notification done;
  absl::StatusOr<std::vector<int>> push_result;
  const int src_bid = 10;
  const int dst_bid = 20;
  const absl::Time start = absl::Now();
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
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_FALSE(push_result.ok());
  EXPECT_THAT(push_result.status(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       ::testing::HasSubstr("timed out")));
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
}

TEST(SocketTransportAdapterTest, PushTimesOutOnPartialBlockIdsStall) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S"); });
  setenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S", "0.1", 1);

  // Server sends partial block IDs (e.g. 2 bytes out of 4) and then stalls.
  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    uint16_t partial_id = 100;
    ABSL_RETURN_IF_ERROR(
        ::peregrine::WriteExact(client_fd, &partial_id, sizeof(partial_id)));
    absl::SleepFor(absl::Milliseconds(500));
    return absl::OkStatus();
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> test_data = {1, 2, 3, 4};
  Request req = {};
  req.socket_opcode = 1;
  req.laddr = test_data.data();
  req.len = test_data.size();
  req.count_or_size = 1;
  req.uuid = 44;
  req.parallelism = 1;
  req.request_id = 0;
  req.stream_idx = 0;

  absl::Notification done;
  absl::StatusOr<std::vector<int>> push_result;
  const int src_bid = 10;
  const absl::Time start = absl::Now();
  auto handle = client_adapter.Post(
      /*peers=*/{GetIpPort(server_transport)},
      /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_bid, 1),
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> res) {
        push_result = std::move(res);
        done.Notify();
      });

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_FALSE(push_result.ok());
  EXPECT_THAT(push_result.status(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       ::testing::HasSubstr("timed out")));
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
}

TEST(SocketTransportAdapterTest,
     SickDecodeDoesNotFreezePrefillWorkersForHealthyDecode) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S"); });
  setenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S", "0.2", 1);

  // Sick server that hangs without responding.
  auto sick_server_handler = [](int client_fd,
                                const ChunkHeader& header) -> absl::Status {
    absl::SleepFor(absl::Milliseconds(500));
    return absl::OkStatus();
  };
  RawBufferTransport sick_server(/*delegate=*/nullptr, /*local_port=*/0,
                                 /*local_ips=*/{}, sick_server_handler);

  // Healthy server that responds to op 1.
  auto healthy_server_handler = [](int client_fd,
                                   const ChunkHeader& header) -> absl::Status {
    std::vector<int> allocated_ids = {100};
    const std::vector<uint8_t> s_ids = SerializeBlockIds(allocated_ids);
    ABSL_RETURN_IF_ERROR(
        ::peregrine::WriteExact(client_fd, s_ids.data(), s_ids.size()));

    uint8_t size_buf[kChunkSizeFieldSize];
    ABSL_RETURN_IF_ERROR(
        ::peregrine::ReadExact(client_fd, size_buf, sizeof(size_buf)));
    const uint32_t chunk_size = DeserializeChunkSize(size_buf);
    std::vector<uint8_t> payload(chunk_size);
    if (chunk_size > 0) {
      ABSL_RETURN_IF_ERROR(
          ::peregrine::ReadExact(client_fd, payload.data(), payload.size()));
    }
    uint8_t ack = 1;
    return ::peregrine::WriteExact(client_fd, &ack, 1);
  };
  RawBufferTransport healthy_server(/*delegate=*/nullptr, /*local_port=*/0,
                                    /*local_ips=*/{}, healthy_server_handler);

  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  // Prefill has 4 socket workers.
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/4);

  // 1. Post to sick server using 4 streams (exhausting all 4 workers if it
  // hung).
  std::vector<uint8_t> test_data = {1, 2, 3, 4};
  std::vector<Request> sick_reqs(4);
  std::vector<int> sick_src_bids = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) {
    sick_reqs[i].socket_opcode = 1;
    sick_reqs[i].laddr = test_data.data();
    sick_reqs[i].len = test_data.size();
    sick_reqs[i].count_or_size = 1;
    sick_reqs[i].uuid = 10;
    sick_reqs[i].parallelism = 4;
    sick_reqs[i].stream_idx = i;
    sick_reqs[i].request_id = i;
  }

  const absl::Time start = absl::Now();
  absl::Notification sick_done;
  absl::StatusOr<std::vector<int>> sick_result;
  std::vector<std::string> sick_peers(4, GetIpPort(sick_server));
  ASSERT_THAT(
      client_adapter
          .Post(sick_peers, absl::MakeConstSpan(sick_reqs),
                absl::MakeConstSpan(sick_src_bids), /*dst_block_ids=*/{},
                [&](absl::StatusOr<std::vector<int>> res) {
                  sick_result = std::move(res);
                  sick_done.Notify();
                })
          .status(),
      absl_testing::IsOk());

  // 2. Post to healthy server.
  Request healthy_req = {};
  healthy_req.socket_opcode = 1;
  healthy_req.laddr = test_data.data();
  healthy_req.len = test_data.size();
  healthy_req.count_or_size = 1;
  healthy_req.uuid = 20;
  healthy_req.parallelism = 1;
  healthy_req.stream_idx = 0;
  healthy_req.request_id = 0;
  const int healthy_src_bid = 99;

  absl::Notification healthy_done;
  absl::StatusOr<std::vector<int>> healthy_result;
  ASSERT_THAT(client_adapter
                  .Post({GetIpPort(healthy_server)},
                        absl::MakeConstSpan(&healthy_req, 1),
                        absl::MakeConstSpan(&healthy_src_bid, 1),
                        /*dst_block_ids=*/{},
                        [&](absl::StatusOr<std::vector<int>> res) {
                          healthy_result = std::move(res);
                          healthy_done.Notify();
                        })
                  .status(),
              absl_testing::IsOk());

  // Healthy push must succeed quickly and not be starved by the sick decode.
  EXPECT_TRUE(healthy_done.WaitForNotificationWithTimeout(absl::Seconds(3)));
  EXPECT_THAT(healthy_result, IsOkAndHolds(::testing::ElementsAre(100)));

  // Sick push must fail due to timeout.
  EXPECT_TRUE(sick_done.WaitForNotificationWithTimeout(absl::Seconds(3)));
  EXPECT_FALSE(sick_result.ok());
  EXPECT_THAT(sick_result.status(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       ::testing::HasSubstr("timed out")));
  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
}

TEST(SocketTransportAdapterTest, WorkerCountGrowsPerPeer) {
  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    return absl::OkStatus();
  };
  RawBufferTransport server1(/*delegate=*/nullptr, /*local_port=*/0, {},
                             server_handler);
  RawBufferTransport server2(/*delegate=*/nullptr, /*local_port=*/0, {},
                             server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/4);

  // Initially 0 workers.
  EXPECT_EQ(client_adapter.worker_count(), 0);

  // Post to peer 1 spawns 4 workers.
  std::vector<uint8_t> test_data = {1, 2};
  Request req = {};
  req.socket_opcode = 1;
  req.laddr = test_data.data();
  req.len = test_data.size();
  req.count_or_size = 1;
  req.uuid = 1;
  req.parallelism = 1;
  const int bid = 10;
  auto handle1 =
      client_adapter.Post({GetIpPort(server1)}, absl::MakeConstSpan(&req, 1),
                          absl::MakeConstSpan(&bid, 1), {}, nullptr);
  ASSERT_THAT(handle1.status(), absl_testing::IsOk());
  EXPECT_EQ(client_adapter.worker_count(), 4);

  // Another request to peer 1 does NOT spawn additional workers.
  req.uuid = 2;
  auto handle2 =
      client_adapter.Post({GetIpPort(server1)}, absl::MakeConstSpan(&req, 1),
                          absl::MakeConstSpan(&bid, 1), {}, nullptr);
  ASSERT_THAT(handle2.status(), absl_testing::IsOk());
  EXPECT_EQ(client_adapter.worker_count(), 4);

  // Request to peer 2 spawns 4 more workers (total 8).
  req.uuid = 3;
  auto handle3 =
      client_adapter.Post({GetIpPort(server2)}, absl::MakeConstSpan(&req, 1),
                          absl::MakeConstSpan(&bid, 1), {}, nullptr);
  ASSERT_THAT(handle3.status(), absl_testing::IsOk());
  EXPECT_EQ(client_adapter.worker_count(), 8);
}

TEST(SocketTransportAdapterTest, WorkerCountCappedAtMaxSocketWorkers) {
  auto cleanup =
      absl::MakeCleanup([] { unsetenv("TPU_RAIDEN_MAX_SOCKET_WORKERS"); });
  setenv("TPU_RAIDEN_MAX_SOCKET_WORKERS", "6", 1);

  auto server_handler = [](int client_fd,
                           const ChunkHeader& header) -> absl::Status {
    return absl::OkStatus();
  };
  RawBufferTransport server1(/*delegate=*/nullptr, /*local_port=*/0, {},
                             server_handler);
  RawBufferTransport server2(/*delegate=*/nullptr, /*local_port=*/0, {},
                             server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/4);

  EXPECT_EQ(client_adapter.max_socket_workers(), 6);
  EXPECT_EQ(client_adapter.worker_count(), 0);

  // Peer 1 spawns 4 workers.
  std::vector<uint8_t> test_data = {1, 2};
  Request req = {};
  req.socket_opcode = 1;
  req.laddr = test_data.data();
  req.len = test_data.size();
  req.count_or_size = 1;
  req.uuid = 1;
  req.parallelism = 1;
  const int bid = 10;
  auto handle1 =
      client_adapter.Post({GetIpPort(server1)}, absl::MakeConstSpan(&req, 1),
                          absl::MakeConstSpan(&bid, 1), {}, nullptr);
  ASSERT_THAT(handle1.status(), absl_testing::IsOk());
  EXPECT_EQ(client_adapter.worker_count(), 4);

  // Peer 2 would want 4 more workers (total 8), but is capped at max 6.
  req.uuid = 2;
  auto handle2 =
      client_adapter.Post({GetIpPort(server2)}, absl::MakeConstSpan(&req, 1),
                          absl::MakeConstSpan(&bid, 1), {}, nullptr);
  ASSERT_THAT(handle2.status(), absl_testing::IsOk());
  EXPECT_EQ(client_adapter.worker_count(), 6);
}

TEST(SocketTransportAdapterTest,
     RequestStreamFailureCancelsPendingTasksForSameRequest) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S"); });
  setenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S", "0.1", 1);

  std::atomic<int> server_requests_received = 0;
  auto server_handler = [&](int client_fd,
                            const ChunkHeader& header) -> absl::Status {
    server_requests_received++;
    absl::SleepFor(absl::Milliseconds(300));
    return absl::OkStatus();
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  // Single stream cap per peer so stream 0 runs while stream 1 remains pending
  // in queue.
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);

  std::vector<uint8_t> test_data = {1, 2, 3, 4};
  std::vector<Request> reqs(2);
  std::vector<int> src_bids = {1, 2};
  for (int i = 0; i < 2; ++i) {
    reqs[i].socket_opcode = 1;
    reqs[i].laddr = test_data.data();
    reqs[i].len = test_data.size();
    reqs[i].count_or_size = 1;
    reqs[i].uuid = 12345;
    reqs[i].parallelism = 2;
    reqs[i].stream_idx = i;
    reqs[i].request_id = i;
  }

  absl::Notification done;
  absl::StatusOr<std::vector<int>> push_result;
  const std::string peer = GetIpPort(server_transport);
  auto handle = client_adapter.Post(
      {peer, peer}, absl::MakeConstSpan(reqs), absl::MakeConstSpan(src_bids),
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> res) {
        push_result = std::move(res);
        done.Notify();
      });

  ASSERT_THAT(handle.status(), absl_testing::IsOk());
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_FALSE(push_result.ok());
  EXPECT_THAT(push_result.status(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       ::testing::HasSubstr("timed out")));

  // Stream 0 was received and timed out; stream 1 was cancelled in queue and
  // never executed.
  EXPECT_EQ(server_requests_received.load(), 1);
}

TEST(SocketTransportAdapterTest,
     RequestFailureCancelsSubsequentLayerPostsForSameUuid) {
  auto cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S"); });
  setenv("TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S", "0.1", 1);

  std::atomic<int> server_requests_received = 0;
  auto server_handler = [&](int client_fd,
                            const ChunkHeader& header) -> absl::Status {
    server_requests_received++;
    absl::SleepFor(absl::Milliseconds(300));
    return absl::OkStatus();
  };

  RawBufferTransport server_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{}, server_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0);
  SocketTransportAdapter client_adapter(&client_transport, /*parallelism=*/1);
  const std::string peer = GetIpPort(server_transport);

  const uint64_t session_uuid = 99999;
  std::vector<uint8_t> test_data = {1, 2};
  Request layer0_req = {};
  layer0_req.socket_opcode = 1;
  layer0_req.laddr = test_data.data();
  layer0_req.len = test_data.size();
  layer0_req.count_or_size = 1;
  layer0_req.uuid = session_uuid;
  layer0_req.parallelism = 1;
  const int bid = 1;

  // Layer 0 is posted and times out on handshake.
  absl::Notification done0;
  absl::StatusOr<std::vector<int>> res0;
  auto h0 = client_adapter.Post({peer}, absl::MakeConstSpan(&layer0_req, 1),
                                absl::MakeConstSpan(&bid, 1), {}, [&](auto r) {
                                  res0 = std::move(r);
                                  done0.Notify();
                                });
  ASSERT_THAT(h0.status(), absl_testing::IsOk());
  EXPECT_TRUE(done0.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_FALSE(res0.ok());
  EXPECT_THAT(res0.status(), StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_EQ(server_requests_received.load(), 1);

  // Layer 1 is posted with the SAME session UUID.
  // It must immediately fail with CancelledError without sending any RPC to the
  // server.
  Request layer1_req = layer0_req;
  absl::Notification done1;
  absl::StatusOr<std::vector<int>> res1;
  auto h1 = client_adapter.Post({peer}, absl::MakeConstSpan(&layer1_req, 1),
                                absl::MakeConstSpan(&bid, 1), {}, [&](auto r) {
                                  res1 = std::move(r);
                                  done1.Notify();
                                });
  // Post itself immediately returned CancelledError or called the callback with
  // CancelledError.
  if (!h1.ok()) {
    EXPECT_THAT(h1.status(), StatusIs(absl::StatusCode::kCancelled));
  } else {
    EXPECT_TRUE(done1.WaitForNotificationWithTimeout(absl::Seconds(1)));
    EXPECT_FALSE(res1.ok());
    EXPECT_THAT(res1.status(), StatusIs(absl::StatusCode::kCancelled));
  }

  // Server was never contacted for Layer 1.
  EXPECT_EQ(server_requests_received.load(), 1);
}

}  // namespace
}  // namespace tpu_raiden::transport::lib

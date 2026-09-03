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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"
#include "tpu_sync/transport/peregrine/src/api/socket_util.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

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

}  // namespace
}  // namespace tpu_raiden::transport::lib

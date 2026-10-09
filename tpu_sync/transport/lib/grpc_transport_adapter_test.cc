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

#include "tpu_sync/transport/lib/grpc_transport_adapter.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

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

auto MatchP2pTimeLabels(absl::string_view src_ip, absl::string_view dst_ip) {
  return ElementsAre(
      telemetry::MetricLabel{.key = telemetry::metric_labels::kSrcIp,
                             .value = src_ip},
      telemetry::MetricLabel{.key = telemetry::metric_labels::kDstIp,
                             .value = dst_ip});
}

TEST(GrpcTransportAdapterTest, PostWithEmptyRequestsFails) {
  GrpcTransportAdapter adapter(/*raw_transport=*/nullptr, /*parallelism=*/1);

  absl::Status callback_status = absl::OkStatus();
  auto result = adapter.Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/{},
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> res) {
        callback_status = res.status();
      });

  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GrpcTransportAdapterTest, PostWithInvalidParallelismFails) {
  GrpcTransportAdapter adapter(/*raw_transport=*/nullptr, /*parallelism=*/1);

  Request req = {};
  req.socket_opcode = 1;
  req.parallelism = -1;

  auto result = adapter.Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{}, /*on_complete=*/nullptr);

  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GrpcTransportAdapterTest, PostWithUnsupportedOpcodeFails) {
  GrpcTransportAdapter adapter(/*raw_transport=*/nullptr, /*parallelism=*/1);

  Request req = {};
  req.socket_opcode = 99;  // Unsupported opcode.

  auto result = adapter.Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{}, /*on_complete=*/nullptr);

  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(GrpcTransportAdapterTest, PollReturnsUnimplemented) {
  GrpcTransportAdapter adapter(/*raw_transport=*/nullptr, /*parallelism=*/1);
  EXPECT_THAT(adapter.Poll(0).status(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(GrpcTransportAdapterTest, PostGrpcPushOp1Success) {
  std::vector<uint8_t> received_payload;
  auto push_handler =
      [&](const ChunkHeader& header, absl::Span<const int> dst_block_ids,
          absl::Span<const int> src_block_ids,
          std::function<absl::Status(absl::Span<const int>)> send_handshake,
          std::function<absl::StatusOr<absl::Cord>()> read_next_payload)
      -> absl::Status {
    if (header.op != 1) {
      return absl::InvalidArgumentError("Expected op 1");
    }
    std::vector<int> allocated_ids(header.count_or_size);
    for (size_t i = 0; i < header.count_or_size; ++i) {
      allocated_ids[i] = static_cast<int>(100 + i);
    }
    ABSL_RETURN_IF_ERROR(send_handshake(allocated_ids));

    for (size_t i = 0; i < header.count_or_size; ++i) {
      ABSL_ASSIGN_OR_RETURN(absl::Cord payload, read_next_payload());
      const std::string payload_str(payload);
      received_payload.assign(payload_str.begin(), payload_str.end());
    }
    return absl::OkStatus();
  };

  GrpcTransportServer server(push_handler);
  GrpcTransportAdapter client_adapter(/*raw_transport=*/nullptr,
                                      /*parallelism=*/1);

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
      /*peers=*/{server.address()},
      /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_bid, 1),
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> res) {
        push_result = std::move(res);
        done.Notify();
      });

  ABSL_ASSERT_OK(handle);
  done.WaitForNotification();
  EXPECT_THAT(push_result, IsOkAndHolds(ElementsAre(100)));
  EXPECT_THAT(received_payload, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
}

TEST(GrpcTransportAdapterTest, PostGrpcPushOp6ExplicitPushSuccess) {
  std::vector<int> observed_dst_ids;
  std::vector<int> observed_src_ids;
  std::vector<uint8_t> received_payload;

  auto push_handler =
      [&](const ChunkHeader& header, absl::Span<const int> dst_block_ids,
          absl::Span<const int> src_block_ids,
          std::function<absl::Status(absl::Span<const int>)> send_handshake,
          std::function<absl::StatusOr<absl::Cord>()> read_next_payload)
      -> absl::Status {
    if (header.op != 6) {
      return absl::InvalidArgumentError("Expected op 6");
    }
    observed_dst_ids.assign(dst_block_ids.begin(), dst_block_ids.end());
    observed_src_ids.assign(src_block_ids.begin(), src_block_ids.end());
    ABSL_RETURN_IF_ERROR(send_handshake({}));

    for (size_t i = 0; i < header.count_or_size; ++i) {
      ABSL_ASSIGN_OR_RETURN(absl::Cord payload, read_next_payload());
      const std::string payload_str(payload);
      received_payload.assign(payload_str.begin(), payload_str.end());
    }
    return absl::OkStatus();
  };

  GrpcTransportServer server(push_handler);
  GrpcTransportAdapter client_adapter(/*raw_transport=*/nullptr,
                                      /*parallelism=*/1);

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
      /*peers=*/{server.address()},
      /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/absl::MakeConstSpan(&src_bid, 1),
      /*dst_block_ids=*/absl::MakeConstSpan(&dst_bid, 1),
      [&](absl::StatusOr<std::vector<int>> res) {
        push_result = std::move(res);
        done.Notify();
      });

  ABSL_ASSERT_OK(handle);
  done.WaitForNotification();
  EXPECT_THAT(push_result, IsOkAndHolds(ElementsAre(55)));
  EXPECT_THAT(observed_dst_ids, ElementsAre(55));
  EXPECT_THAT(observed_src_ids, ElementsAre(5));
  EXPECT_THAT(received_payload, ElementsAre(10, 20, 30, 40));
}

TEST(GrpcTransportAdapterTest, PostGrpcPullOp2ReturnsUnimplemented) {
  GrpcTransportAdapter client_adapter(/*raw_transport=*/nullptr,
                                      /*parallelism=*/1);
  Request req = {};
  req.socket_opcode = 2;
  req.parallelism = 1;

  auto handle = client_adapter.Post(
      /*peers=*/{"127.0.0.1:1234"},
      /*requests=*/absl::MakeConstSpan(&req, 1));
  EXPECT_THAT(handle.status(), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(GrpcTransportAdapterTest, PostGrpcPushMultiStreamParallelismSuccess) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  telemetry::ScopedMetricsBackendReset reset(std::move(mock_backend));

  EXPECT_CALL(
      *raw_mock,
      ObserveHistogram(telemetry::metric_names::kP2pTransferTimeMs,
                       MatchP2pTimeLabels("127.0.0.2", "127.0.0.1"), Gt(0.0)))
      .Times(1);
  EXPECT_CALL(
      *raw_mock,
      IncrementCounter(
          telemetry::metric_names::kSentBytesTotal,
          MatchDirectionLabel(telemetry::metric_labels::kDirectionPush), 8))
      .Times(2);

  absl::Mutex mu;
  int next_alloc_id = 200;
  auto push_handler =
      [&](const ChunkHeader& header, absl::Span<const int> dst_block_ids,
          absl::Span<const int> src_block_ids,
          std::function<absl::Status(absl::Span<const int>)> send_handshake,
          std::function<absl::StatusOr<absl::Cord>()> read_next_payload)
      -> absl::Status {
    if (header.op != 1) {
      return absl::InvalidArgumentError("Expected op 1");
    }
    std::vector<int> allocated_ids(header.count_or_size);
    {
      absl::MutexLock lock(mu);
      for (size_t i = 0; i < header.count_or_size; ++i) {
        allocated_ids[i] = next_alloc_id++;
      }
    }
    ABSL_RETURN_IF_ERROR(send_handshake(allocated_ids));
    for (size_t i = 0; i < header.count_or_size; ++i) {
      ABSL_ASSIGN_OR_RETURN(absl::Cord payload, read_next_payload());
      (void)payload;
    }
    return absl::OkStatus();
  };

  GrpcTransportServer server(push_handler);
  RawBufferTransport client_transport(/*delegate=*/nullptr, /*local_port=*/0,
                                      /*local_ips=*/{"127.0.0.2"});
  GrpcTransportAdapter client_adapter(&client_transport, /*parallelism=*/2);

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
      /*peers=*/{server.address()},
      /*requests=*/requests,
      /*src_block_ids=*/src_block_ids,
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> result) {
        push_result = std::move(result);
        done.Notify();
      }));

  done.WaitForNotification();
  ABSL_ASSERT_OK(push_result);
  EXPECT_THAT(*push_result, ::testing::UnorderedElementsAre(200, 201));
}

TEST(GrpcTransportAdapterTest, UseGrpcTransportAdapterEnvParsing) {
  auto cleanup =
      absl::MakeCleanup([] { unsetenv("TPU_RAIDEN_DATA_TRANSPORT"); });

  unsetenv("TPU_RAIDEN_DATA_TRANSPORT");
  EXPECT_FALSE(UseGrpcTransportAdapter());

  setenv("TPU_RAIDEN_DATA_TRANSPORT", "socket", 1);
  EXPECT_FALSE(UseGrpcTransportAdapter());

  setenv("TPU_RAIDEN_DATA_TRANSPORT", "grpc", 1);
  EXPECT_TRUE(UseGrpcTransportAdapter());

  setenv("TPU_RAIDEN_DATA_TRANSPORT", "GRPC", 1);
  EXPECT_TRUE(UseGrpcTransportAdapter());

  setenv("TPU_RAIDEN_DATA_TRANSPORT", "1", 1);
  EXPECT_TRUE(UseGrpcTransportAdapter());
}

}  // namespace
}  // namespace tpu_raiden::transport::lib

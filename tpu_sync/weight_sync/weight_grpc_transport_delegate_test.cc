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

#include "tpu_sync/weight_sync/weight_grpc_transport_delegate.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_service.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"
#include "tpu_sync/weight_sync/weight_receive_session.h"
#include "tpu_sync/weight_sync/weight_send_session.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::UnorderedElementsAre;

using BufferShardKey = std::pair<uint64_t, uint32_t>;

TEST(WeightGrpcTransportDelegateTest, PushEndToEndWithServerAndSessions) {
  absl::Mutex buf_mu;
  absl::flat_hash_map<BufferShardKey, std::vector<uint8_t>> dst_buffers;
  dst_buffers[BufferShardKey{0, 0}] = std::vector<uint8_t>(8, 0);
  dst_buffers[BufferShardKey{1, 0}] = std::vector<uint8_t>(6, 0);

  WeightGrpcTransportDelegate delegate(
      [&](uint64_t buffer_id,
          uint32_t shard_id) -> absl::StatusOr<absl::Span<uint8_t>> {
        absl::MutexLock lock(buf_mu);
        auto it = dst_buffers.find(BufferShardKey{buffer_id, shard_id});
        if (it == dst_buffers.end()) {
          return absl::NotFoundError(
              absl::StrCat("Missing buffer ", buffer_id, " shard ", shard_id));
        }
        return absl::MakeSpan(it->second);
      });

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  absl::Mutex cb_mu;
  std::vector<uint64_t> completed_layers;

  ::tpu_sync::rpc::StartTransferRequest recv_spec;
  recv_spec.set_uuid(101);
  (*recv_spec.mutable_expected_layer_chunk_counts())[0] = 2;
  (*recv_spec.mutable_expected_layer_chunk_counts())[1] = 1;

  WeightReceiveSessionOptions recv_options;
  recv_options.on_layer_received = [&](uint64_t layer_idx) {
    absl::MutexLock lock(cb_mu);
    completed_layers.push_back(layer_idx);
    return absl::OkStatus();
  };
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_or =
      WeightReceiveSession::CreatePushReceiveSession(recv_spec,
                                                     std::move(recv_options));
  ASSERT_THAT(recv_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> recv_session =
      *std::move(recv_or);

  ASSERT_THAT(delegate.RegisterSession(101, recv_session), IsOk());
  ASSERT_THAT(recv_session->Start(), IsOk());

  const std::vector<uint8_t> src_l0_s0 = {1, 2, 3, 4};
  const std::vector<uint8_t> src_l0_s1 = {5, 6, 7, 8};
  const std::vector<uint8_t> src_l1_s0 = {10, 20, 30, 40};

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(101);
  send_spec.set_is_sender(true);
  send_spec.set_parallelism(2);
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[1].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(4);
    entry->set_size_bytes(4);
  }
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(1);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(1);
    entry->set_size_bytes(4);
  }

  auto src_provider =
      [&](uint64_t buffer_id,
          uint32_t shard_id) -> absl::StatusOr<absl::Span<const uint8_t>> {
    if (buffer_id == 0 && shard_id == 0) return absl::MakeConstSpan(src_l0_s0);
    if (buffer_id == 0 && shard_id == 1) return absl::MakeConstSpan(src_l0_s1);
    if (buffer_id == 1 && shard_id == 0) return absl::MakeConstSpan(src_l1_s0);
    return absl::NotFoundError("Unknown source buffer");
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePushSendSession(&client, send_spec,
                                               src_provider);
  ASSERT_THAT(send_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_or);

  ASSERT_THAT(send_session->Start(), IsOk());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(recv_session->AwaitForDone(), IsOk());

  EXPECT_THAT((dst_buffers[BufferShardKey{0, 0}]),
              ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT((dst_buffers[BufferShardKey{1, 0}]),
              ElementsAre(0, 10, 20, 30, 40, 0));
  {
    absl::MutexLock lock(cb_mu);
    EXPECT_THAT(completed_layers, UnorderedElementsAre(0, 1));
  }

  EXPECT_THAT(delegate.UnregisterSession(101), IsOk());
  EXPECT_THAT(delegate.UnregisterSession(101),
              StatusIs(absl::StatusCode::kNotFound));
  server.Stop();
}

TEST(WeightGrpcTransportDelegateTest, PullEndToEndWithServerAndSessions) {
  WeightGrpcTransportDelegate delegate(
      [](uint64_t, uint32_t) -> absl::StatusOr<absl::Span<uint8_t>> {
        return absl::NotFoundError("unused");
      });

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  auto src = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{11, 22, 33, 44});

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(202);
  send_spec.set_is_sender(true);
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePullSendSession(
          send_spec,
          [&](uint64_t, uint32_t) { return absl::MakeConstSpan(*src); },
          WeightSendSessionOptions{.host_buffer_owner = src});
  ASSERT_THAT(send_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_or);

  ASSERT_THAT(delegate.RegisterSession(202, send_session), IsOk());
  ASSERT_THAT(send_session->Start(), IsOk());

  std::vector<uint8_t> dst(4, 0);
  ::tpu_sync::rpc::StartTransferRequest recv_spec = send_spec;
  recv_spec.set_is_sender(false);
  (*recv_spec.mutable_shard_push_schedules())[0]
      .mutable_entries(0)
      ->add_dst_peers(server.address());

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_or =
      WeightReceiveSession::CreatePullReceiveSession(
          &client, recv_spec,
          [&](uint64_t, uint32_t) { return absl::MakeSpan(dst); });
  ASSERT_THAT(recv_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> recv_session =
      *std::move(recv_or);

  ASSERT_THAT(recv_session->Start(), IsOk());
  EXPECT_THAT(recv_session->AwaitForDone(), IsOk());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(dst, ElementsAre(11, 22, 33, 44));

  EXPECT_THAT(delegate.UnregisterSession(202), IsOk());
  EXPECT_THAT(delegate.GetSession(202), StatusIs(absl::StatusCode::kNotFound));
  server.Stop();
}

TEST(WeightGrpcTransportDelegateTest, RegistrationAndErrorHandling) {
  WeightGrpcTransportDelegate empty_provider_delegate(nullptr);
  EXPECT_THAT(empty_provider_delegate.GetHostBuffer(0, 0),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(empty_provider_delegate.GetSession(1),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(empty_provider_delegate.OnPushChunks(1, 0, 0, 0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(empty_provider_delegate.OnPushChunks(/*uuid=*/0, 0, 0, 0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(empty_provider_delegate.UnregisterSession(1),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      empty_provider_delegate.OnPullChunks(PullChunksRequest{.uuid = 1}),
      StatusIs(absl::StatusCode::kNotFound));

  std::vector<uint8_t> dst_buf(4, 0xEE);
  WeightGrpcTransportDelegate delegate(
      [&](uint64_t, uint32_t) -> absl::StatusOr<absl::Span<uint8_t>> {
        return absl::MakeSpan(dst_buf);
      });

  // Invalid registration rejected.
  EXPECT_THAT(delegate.RegisterSession(1, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Duplicate uuid registration returns kAlreadyExists; OnPullChunks against a
  // receive session returns kFailedPrecondition.
  ::tpu_sync::rpc::StartTransferRequest fail_spec;
  fail_spec.set_uuid(303);
  (*fail_spec.mutable_expected_layer_chunk_counts())[0] = 1;
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> fail_or =
      WeightReceiveSession::CreatePushReceiveSession(fail_spec);
  ASSERT_THAT(fail_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> fail_session =
      *std::move(fail_or);

  ASSERT_THAT(delegate.RegisterSession(303, fail_session), IsOk());
  EXPECT_THAT(delegate.RegisterSession(303, fail_session),
              StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_THAT(delegate.OnPullChunks(PullChunksRequest{.uuid = 303}),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Through GrpcTransportServer, OnDataReceived failure settles the session via
  // Finish(status) + EndOp(), and subsequent pushes are rejected before writing
  // to host memory.
  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  const std::vector<uint8_t> payload = {1, 2, 3, 4};
  EXPECT_THAT(client.PushChunks(server.address(),
                                PushChunksRequest{
                                    .uuid = 303,
                                    .buffer_id = 99,
                                    .shard_id = 0,
                                    .chunks = {ScatterChunkSpec{
                                        .payload = payload, .byte_offset = 0}},
                                }),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(fail_session->Done());
  EXPECT_THAT(fail_session->GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  dst_buf.assign(4, 0xDD);
  EXPECT_THAT(client.PushChunks(server.address(),
                                PushChunksRequest{
                                    .uuid = 303,
                                    .buffer_id = 0,
                                    .shard_id = 0,
                                    .chunks = {ScatterChunkSpec{
                                        .payload = payload, .byte_offset = 0}},
                                }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(dst_buf, ElementsAre(0xDD, 0xDD, 0xDD, 0xDD));

  EXPECT_THAT(delegate.UnregisterSession(303), IsOk());
  EXPECT_THAT(delegate.UnregisterSession(303),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(delegate.GetSession(303), StatusIs(absl::StatusCode::kNotFound));
  server.Stop();
}

TEST(WeightGrpcTransportDelegateTest,
     OnPushChunksReturnsCancelledWhenSessionCancelledMidCallback) {
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(404);
  (*spec.mutable_expected_layer_chunk_counts())[0] = 1;

  std::shared_ptr<::tpu_raiden::TransferSession> session;
  WeightReceiveSessionOptions options;
  options.on_layer_received = [&](uint64_t) {
    session->Finish(absl::DeadlineExceededError("timed out"));
    return absl::OkStatus();
  };
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> session_or =
      WeightReceiveSession::CreatePushReceiveSession(spec, std::move(options));
  ASSERT_THAT(session_or, IsOk());
  session = *std::move(session_or);

  std::vector<uint8_t> dst_buf(4, 0);
  WeightGrpcTransportDelegate delegate(
      [&](uint64_t, uint32_t) -> absl::StatusOr<absl::Span<uint8_t>> {
        return absl::MakeSpan(dst_buf);
      });
  ASSERT_THAT(delegate.RegisterSession(404, session), IsOk());

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;
  const std::vector<uint8_t> payload = {5, 6, 7, 8};

  EXPECT_THAT(client.PushChunks(server.address(),
                                PushChunksRequest{
                                    .uuid = 404,
                                    .buffer_id = 0,
                                    .shard_id = 0,
                                    .chunks = {ScatterChunkSpec{
                                        .payload = payload, .byte_offset = 0}},
                                }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->GetStatus(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  server.Stop();
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

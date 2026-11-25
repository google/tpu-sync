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

#include "tpu_sync/weight_sync/weight_send_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_service.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"
#include "tpu_sync/weight_sync/weight_receive_session.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::UnorderedElementsAre;

using BufferShardKey = std::pair<uint64_t, uint32_t>;

class TestSendTransportRouter final : public GrpcTransportDelegate {
 public:
  void AddHostBuffer(uint64_t buffer_id, uint32_t shard_id, size_t size_bytes,
                     uint8_t fill = 0) {
    absl::MutexLock lock(mu_);
    buffers_[BufferShardKey{buffer_id, shard_id}] =
        std::vector<uint8_t>(size_bytes, fill);
  }

  std::vector<uint8_t> ReadHostBuffer(uint64_t buffer_id,
                                      uint32_t shard_id) const {
    absl::MutexLock lock(mu_);
    auto it = buffers_.find(BufferShardKey{buffer_id, shard_id});
    if (it == buffers_.end()) {
      return {};
    }
    return it->second;
  }

  void RegisterSession(uint64_t uuid,
                       std::shared_ptr<::tpu_raiden::TransferSession> session) {
    absl::MutexLock lock(mu_);
    sessions_[uuid] = std::move(session);
  }

  void RegisterPushReceiveSession(
      uint64_t uuid, std::shared_ptr<::tpu_raiden::TransferSession> session) {
    RegisterSession(uuid, std::move(session));
  }

  absl::StatusOr<absl::Span<uint8_t>> GetHostBuffer(
      uint64_t buffer_id, uint32_t shard_id) override {
    absl::MutexLock lock(mu_);
    auto it = buffers_.find(BufferShardKey{buffer_id, shard_id});
    if (it == buffers_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No host buffer for buffer_id=", buffer_id, ", shard_id=", shard_id));
    }
    return absl::MakeSpan(it->second);
  }

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> GetSession(
      uint64_t uuid) const override {
    absl::MutexLock lock(mu_);
    auto it = sessions_.find(uuid);
    if (it == sessions_.end()) {
      return absl::NotFoundError(absl::StrCat("No session for uuid=", uuid));
    }
    return it->second;
  }

  absl::Status OnPushChunks(uint64_t uuid, uint64_t buffer_id,
                            uint32_t shard_id,
                            uint64_t chunk_batch_id) override {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<::tpu_raiden::TransferSession> session,
        GetSession(uuid));
    const int shard = static_cast<int>(shard_id);
    return session->OnDataReceived(static_cast<size_t>(buffer_id), {},
                                   absl::MakeConstSpan(&shard, 1));
  }

  absl::StatusOr<PullChunksResponse> OnPullChunks(
      const PullChunksRequest& request) override {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<::tpu_raiden::TransferSession> session,
        GetSession(request.uuid));
    auto send_session = std::dynamic_pointer_cast<WeightSendSession>(session);
    if (send_session == nullptr) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Session for uuid=", request.uuid, " is not a WeightSendSession"));
    }
    return send_session->OnPullChunks(request);
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<BufferShardKey, std::vector<uint8_t>> buffers_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<uint64_t, std::shared_ptr<::tpu_raiden::TransferSession>>
      sessions_ ABSL_GUARDED_BY(mu_);
};

TEST(WeightSendSessionTest, PushSendAndPushReceiveWholeModelEndToEnd) {
  TestSendTransportRouter router;
  // Layer 0 dst_shard 0 receives 2 contiguous batches (from src_shard 0 and 1).
  router.AddHostBuffer(/*buffer_id=*/0, /*shard_id=*/0, /*size_bytes=*/8, 0);
  // Layer 1 dst_shard 0 receives a strided gather (src_stride=4, dst_stride=2).
  router.AddHostBuffer(/*buffer_id=*/1, /*shard_id=*/0, /*size_bytes=*/6, 0);

  absl::Mutex cb_mu;
  std::vector<uint64_t> completed_layers;

  ::tpu_sync::rpc::StartTransferRequest recv_spec;
  recv_spec.set_uuid(1001);
  (*recv_spec.mutable_expected_layer_chunk_counts())[0] = 2;
  (*recv_spec.mutable_expected_layer_chunk_counts())[1] = 1;

  WeightReceiveSessionOptions recv_options;
  recv_options.on_layer_received = [&](uint64_t layer_idx) {
    absl::MutexLock lock(cb_mu);
    completed_layers.push_back(layer_idx);
    return absl::OkStatus();
  };
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
      recv_session_or = WeightReceiveSession::CreatePushReceiveSession(
          recv_spec, std::move(recv_options));
  ASSERT_THAT(recv_session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> recv_session =
      *std::move(recv_session_or);
  router.RegisterPushReceiveSession(/*uuid=*/1001, recv_session);

  GrpcTransportServer server(&router);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  ASSERT_THAT(recv_session->Start(), IsOk());

  // Source host buffers for 2 layers.
  const std::vector<uint8_t> src_l0_s0 = {1, 2, 3, 4};
  const std::vector<uint8_t> src_l0_s1 = {5, 6, 7, 8};
  // Strided source: pick [10, 20] at offset 0 and [30, 40] at offset 4.
  const std::vector<uint8_t> src_l1_s0 = {10, 20, 99, 99, 30, 40, 99, 99};

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(1001);
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
    entry->set_size_bytes(2);
    entry->set_count(2);
    entry->set_src_stride_bytes(4);
    entry->set_dst_stride_bytes(2);
  }

  auto src_provider =
      [&](uint64_t buffer_id,
          uint32_t shard_id) -> absl::StatusOr<absl::Span<const uint8_t>> {
    if (buffer_id == 0 && shard_id == 0) {
      return absl::MakeConstSpan(src_l0_s0);
    }
    if (buffer_id == 0 && shard_id == 1) {
      return absl::MakeConstSpan(src_l0_s1);
    }
    if (buffer_id == 1 && shard_id == 0) {
      return absl::MakeConstSpan(src_l1_s0);
    }
    return absl::NotFoundError("Unknown source buffer");
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
      send_session_or = WeightSendSession::CreatePushSendSession(
          &client, send_spec, src_provider);
  ASSERT_THAT(send_session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_session_or);

  ASSERT_THAT(send_session->Start(), IsOk());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(recv_session->AwaitForDone(), IsOk());
  EXPECT_TRUE(send_session->Done());
  EXPECT_TRUE(recv_session->Done());

  EXPECT_THAT(router.ReadHostBuffer(0, 0), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(router.ReadHostBuffer(1, 0), ElementsAre(0, 10, 20, 30, 40, 0));
  {
    absl::MutexLock lock(cb_mu);
    EXPECT_THAT(completed_layers, UnorderedElementsAre(0, 1));
  }

  server.Stop();
}

TEST(WeightSendSessionTest, PullSendAndPullReceiveWholeModelEndToEnd) {
  TestSendTransportRouter router;

  auto src_l0_s0 = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{11, 22, 33, 44});
  auto src_l0_s1 = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{55, 66, 77, 88});
  // Contiguous source, strided destination (scatter).
  auto src_l1_s0 =
      std::make_shared<std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3, 4});

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(2002);
  send_spec.set_is_sender(true);
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[1].add_entries();
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(4);
    entry->set_size_bytes(4);
  }
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->set_layer_idx(1);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(2);
    entry->set_count(2);
    entry->set_src_stride_bytes(2);
    entry->set_dst_stride_bytes(4);
  }

  auto src_provider =
      [&](uint64_t buffer_id,
          uint32_t shard_id) -> absl::StatusOr<absl::Span<const uint8_t>> {
    if (buffer_id == 0 && shard_id == 0) {
      return absl::MakeConstSpan(*src_l0_s0);
    }
    if (buffer_id == 0 && shard_id == 1) {
      return absl::MakeConstSpan(*src_l0_s1);
    }
    if (buffer_id == 1 && shard_id == 0) {
      return absl::MakeConstSpan(*src_l1_s0);
    }
    return absl::NotFoundError("Unknown source buffer");
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
      send_session_or =
          WeightSendSession::CreatePullSendSession(send_spec, src_provider);
  ASSERT_THAT(send_session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_session_or);
  router.RegisterSession(/*uuid=*/2002, send_session);

  GrpcTransportServer server(&router);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  ASSERT_THAT(send_session->Start(), IsOk());

  // Build receiver pull spec pointing to server.address().
  ::tpu_sync::rpc::StartTransferRequest recv_spec = send_spec;
  recv_spec.set_is_sender(false);
  recv_spec.set_parallelism(2);
  for (auto& [src_shard, schedule] :
       *recv_spec.mutable_shard_push_schedules()) {
    for (auto& entry : *schedule.mutable_entries()) {
      entry.add_dst_peers(server.address());
    }
  }

  std::vector<uint8_t> dst_l0_s0(8, 0);
  std::vector<uint8_t> dst_l1_s0(6, 0);
  auto dst_provider =
      [&](uint64_t buffer_id,
          uint32_t shard_id) -> absl::StatusOr<absl::Span<uint8_t>> {
    if (buffer_id == 0 && shard_id == 0) {
      return absl::MakeSpan(dst_l0_s0);
    }
    if (buffer_id == 1 && shard_id == 0) {
      return absl::MakeSpan(dst_l1_s0);
    }
    return absl::NotFoundError("Unknown dst buffer");
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
      recv_session_or = WeightReceiveSession::CreatePullReceiveSession(
          &client, recv_spec, dst_provider);
  ASSERT_THAT(recv_session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> recv_session =
      *std::move(recv_session_or);

  ASSERT_THAT(recv_session->Start(), IsOk());
  EXPECT_THAT(recv_session->AwaitForDone(), IsOk());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(dst_l0_s0, ElementsAre(11, 22, 33, 44, 55, 66, 77, 88));
  EXPECT_THAT(dst_l1_s0, ElementsAre(1, 2, 0, 0, 3, 4));

  server.Stop();
}

TEST(WeightSendSessionTest, PullSendFanoutServesMultipleReceivers) {
  TestSendTransportRouter router;
  auto src = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{42, 43, 44, 45});

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(3003);
  send_spec.set_is_sender(true);
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers("receiver_a");
    entry->add_dst_peers("receiver_b");
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
      send_session_or = WeightSendSession::CreatePullSendSession(
          send_spec,
          [&](uint64_t, uint32_t) { return absl::MakeConstSpan(*src); },
          WeightSendSessionOptions{.host_buffer_owner = src});
  ASSERT_THAT(send_session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_session_or);
  router.RegisterSession(/*uuid=*/3003, send_session);

  GrpcTransportServer server(&router);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  ASSERT_THAT(send_session->Start(), IsOk());

  std::vector<uint8_t> dst_a(4, 0);
  std::vector<uint8_t> dst_b(4, 0);

  auto make_recv = [&](absl::Span<uint8_t> dst) {
    ::tpu_sync::rpc::StartTransferRequest recv_spec;
    recv_spec.set_uuid(3003);
    auto* entry = (*recv_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    return WeightReceiveSession::CreatePullReceiveSession(
        &client, recv_spec, [dst](uint64_t, uint32_t) { return dst; });
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_a_or =
      make_recv(absl::MakeSpan(dst_a));
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_b_or =
      make_recv(absl::MakeSpan(dst_b));
  ASSERT_THAT(recv_a_or, IsOk());
  ASSERT_THAT(recv_b_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> recv_a = *std::move(recv_a_or);
  std::shared_ptr<::tpu_raiden::TransferSession> recv_b = *std::move(recv_b_or);

  ASSERT_THAT(recv_a->Start(), IsOk());
  ASSERT_THAT(recv_b->Start(), IsOk());

  EXPECT_THAT(recv_a->AwaitForDone(), IsOk());
  EXPECT_THAT(recv_b->AwaitForDone(), IsOk());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(dst_a, ElementsAre(42, 43, 44, 45));
  EXPECT_THAT(dst_b, ElementsAre(42, 43, 44, 45));

  server.Stop();
}

TEST(WeightSendSessionTest, PullSendWaitsForResponseOwnerReleaseToDrain) {
  TestSendTransportRouter router;
  auto src = std::make_shared<std::vector<uint8_t>>(4, 9);
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(6006);
  auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
  entry->set_layer_idx(1);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(4);

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePullSendSession(
          spec, [&](uint64_t, uint32_t) { return absl::MakeConstSpan(*src); },
          WeightSendSessionOptions{.host_buffer_owner = src});
  ASSERT_THAT(send_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_or);
  router.RegisterSession(/*uuid=*/6006, send_session);
  ASSERT_THAT(send_session->Start(), IsOk());

  PullChunksRequest req{
      .uuid = 6006, .buffer_id = 1, .shard_id = 0, .chunk_batch_id = 0};
  absl::StatusOr<PullChunksResponse> resp_or = router.OnPullChunks(req);
  ASSERT_THAT(resp_or, IsOk());

  EXPECT_TRUE(send_session->IsDraining());
  EXPECT_FALSE(send_session->Done());

  // External EndOp() must be a no-op and not prematurely drain internal leases.
  send_session->EndOp();
  EXPECT_FALSE(send_session->Done());

  resp_or->owner.reset();
  EXPECT_TRUE(send_session->Done());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
}

TEST(WeightSendSessionTest, PullSendStartSucceedsAfterEarlyPullCompletion) {
  TestSendTransportRouter router;
  auto src = std::make_shared<std::vector<uint8_t>>(4, 9);
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(6007);
  auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
  entry->set_layer_idx(0);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(4);

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePullSendSession(
          spec, [&](uint64_t, uint32_t) { return absl::MakeConstSpan(*src); },
          WeightSendSessionOptions{.host_buffer_owner = src});
  ASSERT_THAT(send_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_or);
  router.RegisterSession(/*uuid=*/6007, send_session);

  // Pull arrives and completes before Start() is called.
  PullChunksRequest req{
      .uuid = 6007, .buffer_id = 0, .shard_id = 0, .chunk_batch_id = 0};
  absl::StatusOr<PullChunksResponse> resp_or = router.OnPullChunks(req);
  ASSERT_THAT(resp_or, IsOk());
  resp_or->owner.reset();
  EXPECT_TRUE(send_session->Done());

  // Start() succeeds once, and rejects a duplicate call.
  EXPECT_THAT(send_session->Start(), IsOk());
  EXPECT_THAT(send_session->Start(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(WeightSendSessionTest, RejectsNegativeParallelismAndExcessiveExpansion) {
  GrpcTransportClient client;
  const std::vector<uint8_t> src(4, 1);
  auto provider = [&](uint64_t, uint32_t) { return absl::MakeConstSpan(src); };

  ::tpu_sync::rpc::StartTransferRequest neg_spec;
  neg_spec.set_uuid(7001);
  neg_spec.set_parallelism(-1);
  EXPECT_THAT(
      WeightSendSession::CreatePushSendSession(&client, neg_spec, provider),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Zero-stride outer expansion exceeding kMaxChunkItemsPerSession is rejected.
  ::tpu_sync::rpc::StartTransferRequest huge_spec;
  huge_spec.set_uuid(7002);
  auto* entry = (*huge_spec.mutable_shard_push_schedules())[0].add_entries();
  entry->add_dst_peers("127.0.0.1:1234");
  entry->set_layer_idx(0);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(0);
  entry->add_outer_counts(2000);
  entry->add_outer_counts(2000);
  entry->add_outer_src_strides_bytes(0);
  entry->add_outer_src_strides_bytes(0);
  entry->add_outer_dst_strides_bytes(0);
  entry->add_outer_dst_strides_bytes(0);
  EXPECT_THAT(
      WeightSendSession::CreatePushSendSession(&client, huge_spec, provider),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WeightSendSession::CreatePullSendSession(huge_spec, provider),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WeightSendSessionTest, EmptySpecsCompleteImmediatelyOnStart) {
  GrpcTransportClient client;
  auto unused_provider =
      [](uint64_t, uint32_t) -> absl::StatusOr<absl::Span<const uint8_t>> {
    return absl::NotFoundError("unused");
  };

  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(3);
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> push_send_or =
      WeightSendSession::CreatePushSendSession(&client, spec, unused_provider);
  ASSERT_THAT(push_send_or, IsOk());
  ASSERT_THAT((*push_send_or)->Start(), IsOk());
  EXPECT_TRUE((*push_send_or)->Done());
  EXPECT_THAT((*push_send_or)->AwaitForDone(), IsOk());

  spec.set_uuid(4);
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> pull_send_or =
      WeightSendSession::CreatePullSendSession(spec, unused_provider);
  ASSERT_THAT(pull_send_or, IsOk());
  ASSERT_THAT((*pull_send_or)->Start(), IsOk());
  EXPECT_TRUE((*pull_send_or)->Done());
  EXPECT_THAT((*pull_send_or)->AwaitForDone(), IsOk());
}

TEST(WeightSendSessionTest, TransferSessionInterfaceContracts) {
  GrpcTransportClient client;
  auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(10);

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePushSendSession(
          &client, spec,
          [](uint64_t, uint32_t) -> absl::StatusOr<absl::Span<const uint8_t>> {
            return absl::NotFoundError("unused");
          },
          WeightSendSessionOptions{.deadline = past});
  ASSERT_THAT(send_or, IsOk());
  ::tpu_raiden::TransferSession* send_iface = send_or->get();
  EXPECT_EQ(send_iface->plan(), absl::nullopt);
  EXPECT_EQ(send_iface->deadline(), past);
  EXPECT_FALSE(send_iface->TryBeginOp());
  EXPECT_THAT(send_iface->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(send_iface->OnDataReceived(0, {}, {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(send_iface->GetDataReadyFuture(0).Await(), IsOk());
}

TEST(WeightSendSessionTest, PushSendWaitsForDataReadyFutureBeforeSending) {
  TestSendTransportRouter router;
  router.AddHostBuffer(/*buffer_id=*/0, /*shard_id=*/0, /*size_bytes=*/4, 0);

  GrpcTransportServer server(&router);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  ::tpu_sync::rpc::StartTransferRequest recv_spec;
  recv_spec.set_uuid(8001);
  (*recv_spec.mutable_expected_layer_chunk_counts())[0] = 1;
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_or =
      WeightReceiveSession::CreatePushReceiveSession(recv_spec);
  ASSERT_THAT(recv_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> recv_session =
      *std::move(recv_or);
  router.RegisterSession(8001, recv_session);
  ASSERT_THAT(recv_session->Start(), IsOk());

  std::vector<uint8_t> src(4, 0);
  auto [promise, future] = tsl::MakePromise<>();

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(8001);
  send_spec.set_is_sender(true);
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }

  WeightSendSessionOptions options;
  options.data_ready_future_provider = [f = future](size_t) { return f; };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePushSendSession(
          &client, send_spec,
          [&](uint64_t, uint32_t) { return absl::MakeConstSpan(src); },
          std::move(options));
  ASSERT_THAT(send_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_or);

  EXPECT_FALSE(send_session->GetDataReadyFuture(0).IsReady());
  ASSERT_THAT(send_session->Start(), IsOk());
  EXPECT_FALSE(send_session->Done());

  // Populate host memory (simulating async D2H) and resolve the ready future.
  src = {9, 8, 7, 6};
  promise.Set(absl::OkStatus());

  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(recv_session->AwaitForDone(), IsOk());
  EXPECT_THAT(router.ReadHostBuffer(0, 0), ElementsAre(9, 8, 7, 6));
  server.Stop();
}

TEST(WeightSendSessionTest, PushSendFailsOnDataReadyError) {
  GrpcTransportClient client;
  const std::vector<uint8_t> src(4, 1);
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(8002);
  {
    auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers("127.0.0.1:1");
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }

  auto [promise, future] = tsl::MakePromise<>();
  WeightSendSessionOptions options;
  options.data_ready_future_provider = [f = future](size_t) { return f; };
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePushSendSession(
          &client, spec,
          [&](uint64_t, uint32_t) { return absl::MakeConstSpan(src); },
          std::move(options));
  ASSERT_THAT(send_or, IsOk());
  ASSERT_THAT((*send_or)->Start(), IsOk());
  promise.Set(absl::InternalError("d2h failed"));
  EXPECT_THAT((*send_or)->AwaitForDone(),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(WeightSendSessionTest, PullSendWaitsForDataReadyFutureBeforeServing) {
  TestSendTransportRouter router;
  auto src = std::make_shared<std::vector<uint8_t>>(4, 0);
  auto [promise, future] = tsl::MakePromise<>();

  ::tpu_sync::rpc::StartTransferRequest send_spec;
  send_spec.set_uuid(8003);
  send_spec.set_is_sender(true);
  {
    auto* entry = (*send_spec.mutable_shard_push_schedules())[0].add_entries();
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }

  WeightSendSessionOptions options;
  options.host_buffer_owner = src;
  options.data_ready_future_provider = [f = future](size_t) { return f; };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePullSendSession(
          send_spec,
          [&](uint64_t, uint32_t) { return absl::MakeConstSpan(*src); },
          std::move(options));
  ASSERT_THAT(send_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> send_session =
      *std::move(send_or);
  router.RegisterSession(8003, send_session);

  GrpcTransportServer server(&router);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;
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
  EXPECT_FALSE(recv_session->Done());

  // Populate host memory and resolve D2H future; pull completes with new bytes.
  *src = {12, 34, 56, 78};
  promise.Set(absl::OkStatus());

  EXPECT_THAT(recv_session->AwaitForDone(), IsOk());
  EXPECT_THAT(send_session->AwaitForDone(), IsOk());
  EXPECT_THAT(dst, ElementsAre(12, 34, 56, 78));
  server.Stop();
}

TEST(WeightSendSessionTest, PullSendFailsOnDataReadyError) {
  TestSendTransportRouter router;
  auto src = std::make_shared<std::vector<uint8_t>>(4, 1);

  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(8004);
  {
    auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(4);
  }

  auto [promise, future] = tsl::MakePromise<>();
  WeightSendSessionOptions options;
  options.host_buffer_owner = src;
  options.data_ready_future_provider = [f = future](size_t) { return f; };
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> send_or =
      WeightSendSession::CreatePullSendSession(
          spec, [&](uint64_t, uint32_t) { return absl::MakeConstSpan(*src); },
          std::move(options));
  ASSERT_THAT(send_or, IsOk());
  router.RegisterSession(8004, *send_or);
  ASSERT_THAT((*send_or)->Start(), IsOk());

  promise.Set(absl::InternalError("d2h failed"));
  PullChunksRequest req{
      .uuid = 8004, .buffer_id = 0, .shard_id = 0, .chunk_batch_id = 0};
  EXPECT_THAT(router.OnPullChunks(req), StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT((*send_or)->AwaitForDone(),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

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

#include "tpu_sync/weight_sync/weight_receive_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_client.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_service.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

namespace tpu_raiden::weight_sync {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::UnorderedElementsAre;

using BufferShardKey = std::pair<uint64_t, uint32_t>;
using PullKey = std::tuple<uint64_t, uint64_t, uint32_t, uint64_t>;

class FakeReceiveTransportDelegate final : public GrpcTransportDelegate {
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

  void RegisterPushReceiveSession(
      uint64_t uuid, std::shared_ptr<::tpu_raiden::TransferSession> session) {
    absl::MutexLock lock(mu_);
    push_receive_sessions_[uuid] = std::move(session);
  }

  void SetPullResponse(uint64_t uuid, uint64_t buffer_id, uint32_t shard_id,
                       uint64_t chunk_batch_id,
                       absl::StatusOr<PullChunksResponse> response) {
    absl::MutexLock lock(mu_);
    pull_responses_[std::make_tuple(uuid, buffer_id, shard_id,
                                    chunk_batch_id)] = std::move(response);
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

  absl::Status OnPushChunks(uint64_t uuid, uint64_t buffer_id,
                            uint32_t shard_id,
                            uint64_t chunk_batch_id) override {
    std::shared_ptr<::tpu_raiden::TransferSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = push_receive_sessions_.find(uuid);
      if (it == push_receive_sessions_.end()) {
        return absl::NotFoundError(
            absl::StrCat("No PushReceiveSession for uuid=", uuid));
      }
      session = it->second;
    }
    if (!session->TryBeginOp()) {
      return absl::FailedPreconditionError("Session is draining or done");
    }
    absl::Cleanup end_op = [&]() { session->EndOp(); };
    const int shard = static_cast<int>(shard_id);
    return session->OnDataReceived(static_cast<size_t>(buffer_id), {},
                                   absl::MakeConstSpan(&shard, 1));
  }

  absl::StatusOr<PullChunksResponse> OnPullChunks(
      const PullChunksRequest& request) override {
    absl::MutexLock lock(mu_);
    auto it = pull_responses_.find(
        std::make_tuple(request.uuid, request.buffer_id, request.shard_id,
                        request.chunk_batch_id));
    if (it == pull_responses_.end()) {
      return absl::NotFoundError("Pull response not configured");
    }
    return it->second;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<BufferShardKey, std::vector<uint8_t>> buffers_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<uint64_t, std::shared_ptr<::tpu_raiden::TransferSession>>
      push_receive_sessions_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<PullKey, absl::StatusOr<PullChunksResponse>>
      pull_responses_ ABSL_GUARDED_BY(mu_);
};

TEST(WeightReceiveSessionTest, PullReceiveSessionPullsWholeModelFromSpec) {
  FakeReceiveTransportDelegate delegate;
  auto src_l0_s0 = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{11, 22, 33, 44});
  auto src_l0_s1 = std::make_shared<std::vector<uint8_t>>(
      std::vector<uint8_t>{55, 66, 77, 88});
  auto src_l1_s0 =
      std::make_shared<std::vector<uint8_t>>(std::vector<uint8_t>{99, 100});

  // Layer 0 receives two chunk batches (src_shard 0 and 1) into dst_shard 0;
  // Layer 1 receives one chunk batch (src_shard 0) into dst_shard 1.
  delegate.SetPullResponse(
      2002, /*buffer_id=*/0, /*shard_id=*/0, /*chunk_batch_id=*/0,
      PullChunksResponse{
          .chunks = {ScatterChunkSpec{
              .payload = absl::MakeConstSpan(*src_l0_s0), .byte_offset = 0}},
          .owner = src_l0_s0});
  delegate.SetPullResponse(
      2002, /*buffer_id=*/0, /*shard_id=*/0, /*chunk_batch_id=*/1,
      PullChunksResponse{
          .chunks = {ScatterChunkSpec{
              .payload = absl::MakeConstSpan(*src_l0_s1), .byte_offset = 4}},
          .owner = src_l0_s1});
  delegate.SetPullResponse(
      2002, /*buffer_id=*/1, /*shard_id=*/1, /*chunk_batch_id=*/0,
      PullChunksResponse{
          .chunks = {GatherChunkSpec{
              .payloads = {absl::MakeConstSpan(*src_l1_s0)}, .byte_offset = 2}},
          .owner = src_l1_s0});

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  std::vector<uint8_t> dst_l0_shard0(8, 0);
  std::vector<uint8_t> dst_l1_shard1(4, 0);

  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(2002);
  spec.set_parallelism(2);
  {
    auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
  }
  {
    auto* entry = (*spec.mutable_shard_push_schedules())[1].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(0);
    entry->set_dst_shard_idx(0);
  }
  {
    auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
    entry->add_dst_peers(server.address());
    entry->set_layer_idx(1);
    entry->set_dst_shard_idx(1);
  }

  absl::Mutex cb_mu;
  std::vector<uint64_t> completed_layers;
  WeightReceiveSessionOptions options;
  options.on_layer_received = [&](uint64_t layer_idx) {
    absl::MutexLock lock(cb_mu);
    completed_layers.push_back(layer_idx);
    return absl::OkStatus();
  };

  auto buffer_provider =
      [&](uint64_t buffer_id,
          uint32_t shard_id) -> absl::StatusOr<absl::Span<uint8_t>> {
    if (buffer_id == 0 && shard_id == 0) {
      return absl::MakeSpan(dst_l0_shard0);
    }
    if (buffer_id == 1 && shard_id == 1) {
      return absl::MakeSpan(dst_l1_shard1);
    }
    return absl::NotFoundError("Unknown buffer");
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> session_or =
      WeightReceiveSession::CreatePullReceiveSession(
          &client, spec, buffer_provider, std::move(options));
  ASSERT_THAT(session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> session =
      *std::move(session_or);

  ASSERT_THAT(session->Start(), IsOk());
  EXPECT_THAT(session->AwaitForDone(), IsOk());
  EXPECT_TRUE(session->Done());

  EXPECT_THAT(dst_l0_shard0, ElementsAre(11, 22, 33, 44, 55, 66, 77, 88));
  EXPECT_THAT(dst_l1_shard1, ElementsAre(0, 0, 99, 100));
  {
    absl::MutexLock lock(cb_mu);
    EXPECT_THAT(completed_layers, UnorderedElementsAre(0, 1));
  }

  server.Stop();
}

TEST(WeightReceiveSessionTest, PullReceiveSessionPropagatesServerError) {
  FakeReceiveTransportDelegate delegate;
  delegate.SetPullResponse(2003, 0, 0, 0,
                           absl::InternalError("upstream failure"));

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  std::vector<uint8_t> dst(4, 0);
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(2003);
  auto* entry = (*spec.mutable_shard_push_schedules())[0].add_entries();
  entry->add_dst_peers(server.address());
  entry->set_layer_idx(0);
  entry->set_dst_shard_idx(0);

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> session_or =
      WeightReceiveSession::CreatePullReceiveSession(
          &client, spec,
          [&](uint64_t, uint32_t) { return absl::MakeSpan(dst); });
  ASSERT_THAT(session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> session =
      *std::move(session_or);

  ASSERT_THAT(session->Start(), IsOk());
  EXPECT_THAT(session->AwaitForDone(), StatusIs(absl::StatusCode::kInternal));

  server.Stop();
}

TEST(WeightReceiveSessionTest, PushReceiveSessionReceivesWholeModelFromSpec) {
  FakeReceiveTransportDelegate delegate;
  delegate.AddHostBuffer(/*buffer_id=*/0, /*shard_id=*/0, /*size_bytes=*/8, 0);
  delegate.AddHostBuffer(/*buffer_id=*/1, /*shard_id=*/1, /*size_bytes=*/6, 0);

  absl::Mutex cb_mu;
  std::vector<uint64_t> completed_layers;

  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(1001);
  (*spec.mutable_expected_layer_chunk_counts())[0] = 2;
  (*spec.mutable_expected_layer_chunk_counts())[1] = 1;

  WeightReceiveSessionOptions options;
  options.on_layer_received = [&](uint64_t layer_idx) {
    absl::MutexLock lock(cb_mu);
    completed_layers.push_back(layer_idx);
    return absl::OkStatus();
  };

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> session_or =
      WeightReceiveSession::CreatePushReceiveSession(spec, std::move(options));
  ASSERT_THAT(session_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> session =
      *std::move(session_or);
  delegate.RegisterPushReceiveSession(/*uuid=*/1001, session);

  GrpcTransportServer server(&delegate);
  ASSERT_THAT(server.Start(0), IsOkAndHolds(Gt(0)));
  GrpcTransportClient client;

  ASSERT_THAT(session->Start(), IsOk());

  const std::vector<uint8_t> part0 = {1, 2, 3, 4};
  const std::vector<uint8_t> part1 = {5, 6, 7, 8};
  const std::vector<uint8_t> g0 = {10, 20};
  const std::vector<uint8_t> g1 = {30, 40, 50};

  PushChunksRequest req0;
  req0.uuid = 1001;
  req0.buffer_id = 0;
  req0.shard_id = 0;
  req0.chunk_batch_id = 0;
  req0.chunks.push_back(ScatterChunkSpec{.payload = absl::MakeConstSpan(part0),
                                         .byte_offset = 0});
  ASSERT_THAT(client.PushChunks(server.address(), req0), IsOk());

  PushChunksRequest req1;
  req1.uuid = 1001;
  req1.buffer_id = 0;
  req1.shard_id = 0;
  req1.chunk_batch_id = 1;
  req1.chunks.push_back(ScatterChunkSpec{.payload = absl::MakeConstSpan(part1),
                                         .byte_offset = 4});
  ASSERT_THAT(client.PushChunks(server.address(), req1), IsOk());

  PushChunksRequest req2;
  req2.uuid = 1001;
  req2.buffer_id = 1;
  req2.shard_id = 1;
  req2.chunk_batch_id = 0;
  req2.chunks.push_back(GatherChunkSpec{
      .payloads = {absl::MakeConstSpan(g0), absl::MakeConstSpan(g1)},
      .byte_offset = 1});
  ASSERT_THAT(client.PushChunks(server.address(), req2), IsOk());

  EXPECT_THAT(session->AwaitForDone(), IsOk());
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(delegate.ReadHostBuffer(0, 0),
              ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(delegate.ReadHostBuffer(1, 1),
              ElementsAre(0, 10, 20, 30, 40, 50));
  {
    absl::MutexLock lock(cb_mu);
    EXPECT_THAT(completed_layers, UnorderedElementsAre(0, 1));
  }

  server.Stop();
}

TEST(WeightReceiveSessionTest, PushReceiveDrainAndStickyFailureSemantics) {
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(7007);
  (*spec.mutable_expected_layer_chunk_counts())[0] = 1;

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_or =
      WeightReceiveSession::CreatePushReceiveSession(spec);
  ASSERT_THAT(recv_or, IsOk());
  std::shared_ptr<::tpu_raiden::TransferSession> session = *std::move(recv_or);
  ASSERT_THAT(session->Start(), IsOk());

  EXPECT_THAT(session->Start(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // Block reports (non-empty minor_dimension_indices) must complete no unit.
  const int minor_idx = 0;
  EXPECT_THAT(
      session->OnDataReceived(0, absl::MakeConstSpan(&minor_idx, 1), {}),
      IsOk());
  EXPECT_FALSE(session->Done());
  EXPECT_THAT(session->OnDataReceived(99, {}, {}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_TRUE(session->TryBeginOp());
  session->Finish(absl::CancelledError("first failure"));
  EXPECT_TRUE(session->IsDraining());
  EXPECT_FALSE(session->Done());
  EXPECT_FALSE(session->TryBeginOp());

  session->Finish(absl::OkStatus());
  session->Finish(absl::InternalError("second failure"));
  EXPECT_THAT(session->GetStatus(), StatusIs(absl::StatusCode::kCancelled));
  // OnDataReceived on a draining/cancelled session is rejected without firing
  // callbacks.
  EXPECT_THAT(session->OnDataReceived(0, {}, {}),
              StatusIs(absl::StatusCode::kCancelled));

  session->EndOp();
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->AwaitForDone(), StatusIs(absl::StatusCode::kCancelled));
}

TEST(WeightReceiveSessionTest, EmptySpecsCompleteImmediatelyOnStart) {
  GrpcTransportClient client;
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(1);

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> pull_recv_or =
      WeightReceiveSession::CreatePullReceiveSession(
          &client, spec,
          [](uint64_t, uint32_t) -> absl::StatusOr<absl::Span<uint8_t>> {
            return absl::NotFoundError("unused");
          });
  ASSERT_THAT(pull_recv_or, IsOk());
  ASSERT_THAT((*pull_recv_or)->Start(), IsOk());
  EXPECT_TRUE((*pull_recv_or)->Done());
  EXPECT_THAT((*pull_recv_or)->AwaitForDone(), IsOk());

  spec.set_uuid(2);
  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> push_recv_or =
      WeightReceiveSession::CreatePushReceiveSession(spec);
  ASSERT_THAT(push_recv_or, IsOk());
  ASSERT_THAT((*push_recv_or)->Start(), IsOk());
  EXPECT_TRUE((*push_recv_or)->Done());
  EXPECT_THAT((*push_recv_or)->AwaitForDone(), IsOk());
}

TEST(WeightReceiveSessionTest, TransferSessionInterfaceContracts) {
  auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(11);
  (*spec.mutable_expected_layer_chunk_counts())[0] = 1;

  absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>> recv_or =
      WeightReceiveSession::CreatePushReceiveSession(
          spec, WeightReceiveSessionOptions{.deadline = past});
  ASSERT_THAT(recv_or, IsOk());
  ::tpu_raiden::TransferSession* recv_iface = recv_or->get();
  EXPECT_EQ(recv_iface->plan(), absl::nullopt);
  EXPECT_EQ(recv_iface->deadline(), past);
  EXPECT_THAT(recv_iface->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(recv_iface->GetDataReadyFuture(0).Await(), IsOk());
  ASSERT_THAT(recv_iface->Start(), IsOk());
  EXPECT_THAT(recv_iface->OnDataReceived(0, {}, {}),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(recv_iface->AwaitForDone(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(WeightReceiveSessionTest, PushReceiveRejectsTotalExpectedOverflow) {
  ::tpu_sync::rpc::StartTransferRequest spec;
  spec.set_uuid(12);
  (*spec.mutable_expected_layer_chunk_counts())[0] =
      std::numeric_limits<int64_t>::max();
  (*spec.mutable_expected_layer_chunk_counts())[1] =
      std::numeric_limits<int64_t>::max();
  (*spec.mutable_expected_layer_chunk_counts())[2] =
      std::numeric_limits<int64_t>::max();

  EXPECT_THAT(WeightReceiveSession::CreatePushReceiveSession(spec),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace tpu_raiden::weight_sync

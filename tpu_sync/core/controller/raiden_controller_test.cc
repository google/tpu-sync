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

#include "tpu_sync/core/controller/raiden_controller.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/controller_server.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/kv_cache/storage/storage.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;

class RaidenControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestWorkerServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");
  }

  void RegisterAndInitWorker(RaidenController& controller,
                             const std::string& worker_id,
                             const std::string& worker_address,
                             int64_t node_id = 0) {
    core::controller::RaidenControllerClient client(
        controller.controller_address());
    auto status = client.RegisterWorker(
        worker_id, worker_address,
        {::tpu_raiden::RaidenTransferEndpoint{worker_address, {}}}, node_id);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  ::tpu_sync::rpc::RaidenIdProto unit_;
  std::unique_ptr<TestWorkerServer> test_server_;
};

TEST_F(RaidenControllerTest, AllocateAndDeallocateSuccess) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto controller,
        RaidenController::Create(
            unit_, std::vector<std::string>{test_server_->server_address},
            /*num_blocks=*/10, /*num_shards=*/2,
            /*shard_size_bytes=*/1024, ""));

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 0);

    TF_ASSERT_OK_AND_ASSIGN(const auto& allocated_buffers,
                            controller->Allocate(/*num_blocks=*/3));
    ASSERT_EQ(allocated_buffers.size(), 3);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 3);

    EXPECT_EQ(allocated_buffers[0].buffer_handles_size(), 2);
    EXPECT_NE(allocated_buffers[0].buffer_handles(0).handle(),
              allocated_buffers[0].buffer_handles(1).handle());

    ASSERT_TRUE(controller->Deallocate(allocated_buffers).ok());
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 0);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, RegisterWorkerSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));

  // Initial state shouldn't have the worker in registry
  EXPECT_FALSE(controller->worker_registry()->GetWorker("worker_0").ok());

  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // Worker should be registered
  TF_ASSERT_OK_AND_ASSIGN(auto worker,
                          controller->worker_registry()->GetWorker("worker_0"));
  EXPECT_NE(worker.worker_service_client, nullptr);

  // Registration should have synchronously created the buffers
  EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
}

TEST_F(RaidenControllerTest, ConstructWithServerAddressWorks) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto controller,
        RaidenController::Create(
            unit_, std::vector<std::string>{test_server_->server_address},
            /*num_blocks=*/5, /*num_shards=*/2,
            /*shard_size_bytes=*/512, ""));
    EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, InitWithIPv6BracketedAddress) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, "[::1]:12345"));

  EXPECT_TRUE(absl::StartsWith(controller->controller_address(), "[::1]:"));
}

TEST_F(RaidenControllerTest, AllocateExceedingCapacityFails) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  EXPECT_THAT(controller->Allocate(/*num_blocks=*/10), Not(IsOk()));
  EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 0);
}

TEST_F(RaidenControllerTest, DeallocateInvalidIndexFails) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  // 1. Missing index
  ::tpu_sync::proto::BufferProto missing_index_buf;
  EXPECT_TRUE(
      absl::IsInvalidArgument(controller->Deallocate({missing_index_buf})));

  // 2. Negative index
  ::tpu_sync::proto::BufferProto negative_index_buf;
  negative_index_buf.set_index(-1);
  EXPECT_TRUE(
      absl::IsInvalidArgument(controller->Deallocate({negative_index_buf})));

  // 3. Out-of-bounds index (e.g. index 10 when num_blocks is 5)
  ::tpu_sync::proto::BufferProto oob_buf;
  oob_buf.set_index(10);
  EXPECT_TRUE(absl::IsInvalidArgument(controller->Deallocate({oob_buf})));
}

TEST_F(RaidenControllerTest, AllocateAndDeallocateBlockIdsSuccess) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto controller,
        RaidenController::Create(
            unit_, std::vector<std::string>{test_server_->server_address},
            /*num_blocks=*/10, /*num_shards=*/2,
            /*shard_size_bytes=*/1024, ""));

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 0);

    TF_ASSERT_OK_AND_ASSIGN(const auto& block_ids,
                            controller->AllocateBlockIds(/*num_blocks=*/3));
    ASSERT_EQ(block_ids.size(), 3);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 3);

    for (int block_id : block_ids) {
      EXPECT_GE(block_id, 0);
      EXPECT_LT(block_id, 10);
    }

    ASSERT_TRUE(controller->DeallocateBlockIds(block_ids).ok());
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 0);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, DeallocateNonExistentBlockIdFails) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  std::vector<int> to_delete = {9999};
  EXPECT_FALSE(controller->DeallocateBlockIds(to_delete).ok());
}

TEST_F(RaidenControllerTest, CreateFailsOnBufferCreationFailure) {
  EXPECT_THAT(
      RaidenController::Create(unit_, std::vector<std::string>{"localhost:1"},
                               /*num_blocks=*/5,
                               /*num_shards=*/1, /*shard_size_bytes=*/512, ""),
      Not(IsOk()));
}

TEST_F(RaidenControllerTest, ExpectedWorkerCountSatisfiedByStaticWorkers) {
  // Statically supplied workers register during Init, so the barrier is
  // already met when the wait runs and construction returns promptly.
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/2,
          /*shard_size_bytes=*/512, "",
          /*preprovision_worker_buffers=*/true, /*expected_worker_count=*/1));
  EXPECT_EQ(controller->worker_registry()->GetRegisteredWorkers().size(), 1);
}

TEST_F(RaidenControllerTest, ExpectedWorkerCountTimesOut) {
  setenv("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S", "1", /*overwrite=*/1);
  EXPECT_THAT(
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/2,
          /*shard_size_bytes=*/512, "",
          /*preprovision_worker_buffers=*/true, /*expected_worker_count=*/2),
      StatusIs(absl::StatusCode::kDeadlineExceeded,
               testing::AllOf(HasSubstr("expected 2 worker(s)"),
                              HasSubstr("got 1"))));
  unsetenv("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S");
}

TEST_F(RaidenControllerTest,
       ExpectedWorkerCountTimeoutCleansUpRegistryAndCallbacks) {
  setenv("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S", "1", /*overwrite=*/1);
  EXPECT_THAT(
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/2,
          /*shard_size_bytes=*/512, "",
          /*preprovision_worker_buffers=*/true, /*expected_worker_count=*/2),
      Not(IsOk()));
  unsetenv("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S");

  // After the constructor threw due to timeout, the callback capturing the
  // destroyed controller must be detached, and the singleton server must not
  // invoke callbacks on dangling memory when a late worker registers.
  auto& server = core::controller::ControllerServer::GetInstance();
  core::controller::RaidenControllerClient client(
      absl::StrCat("localhost:", server.GetGrpcPort()));
  auto status = client.RegisterWorker(
      "late_worker", test_server_->server_address,
      {::tpu_raiden::RaidenTransferEndpoint{test_server_->server_address, {}}},
      /*node_id=*/0);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(RaidenControllerTest, RegisterWorkerFailsOnBufferCreationFailure) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5,
                               /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  core::controller::RaidenControllerClient client(
      controller->controller_address());
  auto status = client.RegisterWorker(
      "worker_bad", "localhost:1",
      {::tpu_raiden::RaidenTransferEndpoint{"localhost:1", {}}});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("CreateBuffers RPC failed: failed to connect"));
}

TEST_F(RaidenControllerTest, TransferBuffersDelegatesToWorkerService) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  Buffer src_buf(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller->TransferBuffers("worker_0", {src_buf}, {dst_buf}).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer manager is not configured on WorkerService"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedOffsets) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  Buffer src_buf1(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer src_buf2(30, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf1(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller->TransferBuffers("worker_0", {src_buf1, src_buf2}, {dst_buf1})
          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Source and destination buffers must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedCopySizes) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  Buffer src_buf1(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer src_buf2(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf1(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf2(30, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  std::vector<int64_t> copy_sizes = {1};

  auto status =
      controller
          ->TransferBuffers({src_buf1, src_buf2}, {dst_buf1, dst_buf2},
                            /*staging_host_buffers=*/{}, copy_sizes)
          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "copy_sizes, if provided, must match the length of src_buffers"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationEmptyOffsets) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  auto status = controller->TransferBuffers({}, {}).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Source and destination buffers must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationInvalidIndex) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512, ""));

  // 1. Source buffer has invalid negative index
  Buffer src_buf_invalid(-1, {}, std::nullopt,
                         ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf_valid(2, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status1 =
      controller
          ->TransferBuffers("worker_0", {src_buf_invalid}, {dst_buf_valid})
          .Await();
  EXPECT_FALSE(status1.ok());
  EXPECT_EQ(status1.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status1.message(),
              HasSubstr("Source buffer has invalid negative index: -1"));

  // 2. Destination buffer has invalid negative index
  Buffer src_buf_valid(1, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf_invalid(-2, {}, std::nullopt,
                         ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status2 =
      controller
          ->TransferBuffers("worker_0", {src_buf_valid}, {dst_buf_invalid})
          .Await();
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status2.message(),
              HasSubstr("Destination buffer has invalid negative index: -2"));
}

TEST_F(RaidenControllerTest, MultiWorkerBroadcastSupport) {
  auto test_server2 = CreateTestWorkerServer();
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  EXPECT_EQ(test_server2->service->GetBufferCount(), 0);

  std::vector<std::string> addresses = {test_server_->server_address,
                                        test_server2->server_address};

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto controller, RaidenController::Create(
                             unit_, addresses, /*num_blocks=*/5,
                             /*num_shards=*/2, /*shard_size_bytes=*/512, ""));

    // Buffers created on both worker servers.
    EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
    EXPECT_EQ(test_server2->service->GetBufferCount(), 10);

    // Broadcast TransferBuffers.
    Buffer src_buf(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
    Buffer dst_buf(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
    auto status = controller->TransferBuffers({src_buf}, {dst_buf}).Await();
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(
        status.message(),
        HasSubstr("Transfer manager is not configured on WorkerService"));
  }

  // Buffers cleaned up on both worker servers on destructor.
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  EXPECT_EQ(test_server2->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, TransferBuffersD2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  Buffer src_buf1(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer src_buf2(30, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf1(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf2(40, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  std::vector<int64_t> copy_sizes = {1, 2};

  auto status = controller
                    ->TransferBuffers("worker_0", {src_buf1, src_buf2},
                                      {dst_buf1, dst_buf2},
                                      /*staging_host_buffers=*/{}, copy_sizes)
                    .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10, 30));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20, 40));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1, 2));
}

TEST_F(RaidenControllerTest, TransferBuffersH2DSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  Buffer src_buf(100, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(200, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto status =
      controller->TransferBuffers("worker_0", {src_buf}, {dst_buf}).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersH2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  Buffer src_buf(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(20, {}, "localhost:8080", ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller->TransferBuffers("worker_0", {src_buf}, {dst_buf}).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20));
}

// ===========================================================================
// ReadRemote: receiver-initiated pull with a source-side lease.
//
// These run against a REAL in-process source RaidenControllerServiceImpl with
// its lease handlers live, so the acquire/release wire protocol is exercised
// for real on CPU; only the transfer manager is mocked. That is what lets the
// SUCCESS paths run here at all -- a real PJRT transfer cannot execute on CPU,
// and the commit-side logic that only runs after a successful transfer is
// exactly where the last cycle's bug hid.
// ===========================================================================

namespace {

// Stand-in for the source store: pins by hash and hands back deterministic
// authoritative ids that differ from whatever the destination asked for, so a
// test can prove the destination used the SOURCE's ids.
class FakeSourceStore {
 public:
  void SetHostResident(const std::vector<std::string>& hashes) {
    for (const auto& h : hashes) host_resident_.insert(h);
  }
  static int32_t IdFor(const std::string& hash) {
    return static_cast<int32_t>(500 + hash.back());
  }
  int pins(const std::string& h) const {
    auto it = pins_.find(h);
    return it == pins_.end() ? 0 : it->second;
  }
  absl::StatusOr<std::vector<int32_t>> ValidateAndPin(
      absl::Span<const std::string> hashes) {
    absl::MutexLock lock(&mu_);
    for (const auto& h : hashes) {
      if (!host_resident_.contains(h)) {
        return absl::NotFoundError(absl::StrCat("no such hash: ", h));
      }
    }
    std::vector<int32_t> ids;
    for (const auto& h : hashes) {
      pins_[h]++;
      ids.push_back(IdFor(h));
    }
    return ids;
  }
  void Unpin(absl::Span<const std::string> hashes) {
    absl::MutexLock lock(&mu_);
    for (const auto& h : hashes) pins_[h]--;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_set<std::string> host_resident_;
  absl::flat_hash_map<std::string, int> pins_;
};

}  // namespace

class ReadRemotePullTest : public RaidenControllerTest {
 protected:
  void SetUp() override {
    RaidenControllerTest::SetUp();
    src_server_ = core::controller::CreateTestControllerServer();
    src_store_ = std::make_unique<FakeSourceStore>();
    src_store_->SetHostResident({"h0", "h1"});
    src_server_->service->SetReadRemoteHooks(
        [this](absl::Span<const std::string> h) {
          return src_store_->ValidateAndPin(h);
        },
        [this](absl::Span<const std::string> h) { src_store_->Unpin(h); });

    // The source advertises one worker group whose node_id matches the
    // destination worker's, so strict node_id matching pairs them.
    ABSL_ASSERT_OK(src_server_->client->RegisterWorker(
        "src_worker_0", "src_worker_0_addr",
        {::tpu_raiden::RaidenTransferEndpoint{"src_data_ep:41000", {0, 1}}},
        /*node_id=*/0));
  }

  void TearDown() override {
    if (src_server_ && src_server_->service) {
      src_server_->service->ClearReadRemoteHooks();
    }
    RaidenControllerTest::TearDown();
  }

  std::unique_ptr<core::controller::TestControllerServer> src_server_;
  std::unique_ptr<FakeSourceStore> src_store_;
};

TEST_F(ReadRemotePullTest, FullSuccessPathUsesAuthoritativeIdsAndSrcEndpoints) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st =
      dest->ReadRemote(src_server_->server_address,
                       /*src_host_block_ids=*/{10, 11},
                       /*dst_host_block_ids=*/{20, 21}, {"h0", "h1"})
          .Await();
  ABSL_EXPECT_OK(st);

  // The pull ran against the SOURCE's ids, not the advisory ones we sent...
  EXPECT_EQ(mock.vector_h2h_read_calls, 1);
  EXPECT_THAT(mock.last_src_offsets, ElementsAre(FakeSourceStore::IdFor("h0"),
                                                 FakeSourceStore::IdFor("h1")));
  EXPECT_THAT(mock.last_dst_offsets, ElementsAre(20, 21));
  // ...routed to the source's shard-tagged data endpoint...
  ASSERT_EQ(mock.last_read_descriptors.size(), 1u);
  EXPECT_EQ(mock.last_read_descriptors[0].endpoint, "src_data_ep:41000");
  // ...and the lease was released, so the source holds no pins.
  EXPECT_EQ(src_store_->pins("h0"), 0);
  EXPECT_EQ(src_store_->pins("h1"), 0);
}

TEST_F(ReadRemotePullTest, HbmModeBuildsStagingPlusDeviceDst) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st = dest->ReadRemote(src_server_->server_address, {10, 11},
                                     {20, 21}, {"h0", "h1"},
                                     /*dst_device_block_ids=*/{7, 8})
                        .Await();
  ABSL_EXPECT_OK(st);

  // Flow order: remote host src -> local host staging -> local device dst.
  // The device destination must be the CALLER's device ids, never the host
  // landing ids reused as device ids.
  EXPECT_EQ(mock.vector_h2d_read_calls, 1);
  EXPECT_THAT(mock.last_src_offsets, ElementsAre(FakeSourceStore::IdFor("h0"),
                                                 FakeSourceStore::IdFor("h1")));
  EXPECT_THAT(mock.last_staging_offsets, ElementsAre(20, 21));
  EXPECT_THAT(mock.last_dst_offsets, ElementsAre(7, 8));
}

TEST_F(ReadRemotePullTest, HostModeBuildsDramDstWithNoStaging) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  ABSL_EXPECT_OK(
      dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h0"})
          .Await());
  EXPECT_EQ(mock.vector_h2h_read_calls, 1);
  EXPECT_EQ(mock.vector_h2d_read_calls, 0);
  EXPECT_TRUE(mock.last_staging_offsets.empty());
}

TEST_F(ReadRemotePullTest, DeviceIdSizeMismatchRejectedBeforeAcquire) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st =
      dest->ReadRemote(src_server_->server_address, {10, 11}, {20, 21},
                       {"h0", "h1"}, /*dst_device_block_ids=*/{7})
          .Await();
  EXPECT_EQ(st.code(), absl::StatusCode::kInvalidArgument);
  // Nothing was pinned at the source: a caller error must not cost the peer
  // any capacity.
  EXPECT_EQ(src_store_->pins("h0"), 0);
  EXPECT_EQ(src_server_->service->LeaseCountForTest(), 0u);
  EXPECT_EQ(mock.vector_h2h_read_calls, 0);
}

// Every peer restart resolves to a new address, and the old one is never
// dialled again. While the caller cached addresses forever a restarted peer
// was simply unreachable, which incidentally kept this map at one entry per
// peer; resolving per read removed that accidental bound, so the cache has to
// carry its own.
TEST_F(ReadRemotePullTest, PeerChurnDoesNotGrowTheStubCacheWithoutBound) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  // Dial far more distinct addresses than the bound. They need not answer: the
  // stub (and its cache entry) is created before the acquire is attempted.
  const size_t churn = RaidenController::MaxCachedStubsForTest() + 50;
  for (size_t i = 0; i < churn; ++i) {
    (void)dest
        ->ReadRemote(absl::StrCat("127.0.0.1:", 40000 + i), {10}, {20}, {"h0"})
        .Await();
  }
  EXPECT_LE(dest->CachedStubCountForTest(),
            RaidenController::MaxCachedStubsForTest());

  // The bound must not cost the live peer its channel: the real source is the
  // most recent address, so it is still cached and still works.
  ABSL_EXPECT_OK(
      dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h0"})
          .Await());
  ABSL_EXPECT_OK(
      dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h0"})
          .Await());
  EXPECT_LE(dest->CachedStubCountForTest(),
            RaidenController::MaxCachedStubsForTest());
}

// The controller resolves nothing: the caller supplies the source's address.
// An empty one is a caller bug and must be named as such, before any RPC --
// dialling "" instead surfaces as a gRPC parse error naming no peer.
TEST_F(ReadRemotePullTest, EmptyControllerAddressRejectedBeforeAcquire) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st =
      dest->ReadRemote(/*src_controller_address=*/"", {10}, {20}, {"h0"})
          .Await();
  EXPECT_EQ(st.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(st.message()), HasSubstr("controller address"));
  EXPECT_EQ(src_store_->pins("h0"), 0);
  EXPECT_EQ(src_server_->service->LeaseCountForTest(), 0u);
  EXPECT_EQ(mock.vector_h2h_read_calls, 0);
}

TEST_F(ReadRemotePullTest, EmptyHashesRejected) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);
  absl::Status st = dest->ReadRemote(src_server_->server_address, {10}, {20},
                                     /*block_hashes=*/{})
                        .Await();
  EXPECT_EQ(st.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ReadRemotePullTest, AcquireNotFoundFailsFastWithoutTransfer) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st =
      dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h_missing"})
          .Await();
  // The source's verify-hook distinction survives the RPC boundary, and there
  // is no retry: NOT_FOUND is terminal against this peer.
  EXPECT_EQ(st.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(mock.vector_h2h_read_calls, 0);
  EXPECT_EQ(mock.h2h_read_calls, 0);
  EXPECT_EQ(src_server_->service->LeaseCountForTest(), 0u);
}

TEST_F(ReadRemotePullTest, TransferFailureStillReleasesTheLease) {
  ShardAwareMockTransferManager mock;
  mock.fail_transfers = true;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st =
      dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h0"}).Await();
  EXPECT_FALSE(st.ok());
  // Release runs on EVERY path: a failed read must not leave the source
  // holding pins until the TTL expires.
  EXPECT_EQ(src_store_->pins("h0"), 0);
}

TEST_F(ReadRemotePullTest, RevokedVerdictFailsTheRead) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  // Revoke every lease the moment it is granted, so the transfer succeeds but
  // the verdict does not. This is the case the TTL alone could not catch: the
  // bytes may come from blocks the source already reused.
  src_server_->service->SetLeaseGrantedHookForTest([this](uint64_t lease_id) {
    src_server_->service->ForceExpireForTest(lease_id);
  });

  TF_ASSERT_OK_AND_ASSIGN(
      auto dest,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);

  absl::Status st =
      dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h0"}).Await();
  EXPECT_EQ(st.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(st.message()), HasSubstr("revoked"));
}

TEST_F(ReadRemotePullTest, ControllerTeardownMidReadIsSafe) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  tsl::Future<> read;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto dest,
        RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                                 /*shard_size_bytes=*/512, ""));
    RegisterAndInitWorker(*dest, "worker_0", test_server_->server_address);
    read = dest->ReadRemote(src_server_->server_address, {10}, {20}, {"h0"});
    // dest is destroyed here, possibly with the read still in flight. The
    // continuations hold only shared state plus a lifetime handle, so this
    // must not be a use-after-free.
  }
  absl::Status st = read.Await();
  // Either it completed before teardown or it was cancelled -- both fine; a
  // crash is not.
  EXPECT_TRUE(st.ok() || st.code() == absl::StatusCode::kCancelled)
      << st.ToString();
  // Whatever happened, the source must not be left holding pins.
  for (int i = 0; i < 200 && src_store_->pins("h0") != 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(src_store_->pins("h0"), 0);
}

// End-to-end at the source worker: when the source worker's transfer manager
// exposes the vector (shard-matching) H2h overloads, the controller fan-out ->
// WorkerServiceImpl -> KVManagerHolder must dispatch to that SHARD-MATCHING
// path (not the single-endpoint string fallback) and hand it the full
// shard-tagged descriptor list. This is the path that was previously untested
// because MockTransferManager only has the string overloads.
TEST_F(RaidenControllerTest,
       TransferBuffersTriggersShardMatchingVectorPathAtWorker) {
  ShardAwareMockTransferManager shard_mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&shard_mock));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // Source block -> destination host block. The destination peer worker exposes
  // two shard-tagged sub-manager endpoints; both must reach the worker's vector
  // H2hWrite intact so its NUMA sub-managers can shard-match against them.
  std::vector<Buffer> src_buffers;
  src_buffers.emplace_back(/*index=*/10, std::vector<BufferShard>{},
                           std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  // One destination peer worker group. Its node_id (0) matches the source
  // worker registered above (RegisterAndInitWorker defaults node_id to 0), so
  // strict node_id matching selects it.
  ::tpu_raiden::RaidenWorkerEndpoints peer_group{
      /*node_id=*/0,
      /*worker_id=*/"worker_0",
      {{"10.0.0.9:41000", {0, 1, 2, 3}}, {"10.0.0.9:41001", {4, 5, 6, 7}}}};
  Buffer dst_buf(/*index=*/20, std::vector<BufferShard>{}, std::nullopt,
                 ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dst_buf.set_remote_worker_endpoints({peer_group});
  std::vector<Buffer> dst_buffers;
  dst_buffers.push_back(std::move(dst_buf));

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok()) << status.message();

  // Shard-matching (vector) path was taken, NOT the single-endpoint string
  // fallback...
  EXPECT_EQ(shard_mock.vector_h2h_write_calls, 1);
  EXPECT_EQ(shard_mock.h2h_write_calls, 0);
  // ...and the worker received the full shard-tagged descriptor list intact.
  ASSERT_EQ(shard_mock.last_write_descriptors.size(), 2u);
  EXPECT_EQ(shard_mock.last_write_descriptors[0].endpoint, "10.0.0.9:41000");
  EXPECT_EQ(shard_mock.last_write_descriptors[0].shards,
            (std::vector<int64_t>{0, 1, 2, 3}));
  EXPECT_EQ(shard_mock.last_write_descriptors[1].endpoint, "10.0.0.9:41001");
  EXPECT_EQ(shard_mock.last_write_descriptors[1].shards,
            (std::vector<int64_t>{4, 5, 6, 7}));
}

// Source workers are matched to destination peer groups strictly by node_id,
// not by sorted-worker-id index. Here the worker_id sort order is the REVERSE
// of the node_id order, so index-based matching would pair them backwards.
TEST_F(RaidenControllerTest,
       TransferBuffersMatchesWorkersByNodeIdNotSortOrder) {
  auto test_server2 = CreateTestWorkerServer();
  MockTransferManager mock_lo;  // node_id 10
  MockTransferManager mock_hi;  // node_id 20
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_lo));
  test_server2->service->SetTransferManager(KVManagerHolder(&mock_hi));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  // worker_id "z_lo" (node 10) sorts AFTER "a_hi" (node 20): sorted order is
  // [a_hi(node20), z_lo(node10)] -- reverse of node order.
  RegisterAndInitWorker(*controller, "z_lo", test_server_->server_address,
                        /*node_id=*/10);
  RegisterAndInitWorker(*controller, "a_hi", test_server2->server_address,
                        /*node_id=*/20);

  std::vector<Buffer> src_buffers;
  src_buffers.emplace_back(/*index=*/1, std::vector<BufferShard>{},
                           std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  // Groups listed node-10 first, node-20 second.
  Buffer dst_buf(/*index=*/2, std::vector<BufferShard>{}, std::nullopt,
                 ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dst_buf.set_remote_worker_endpoints({
      ::tpu_raiden::RaidenWorkerEndpoints{10, "z_lo", {{"ep_lo:1", {}}}},
      ::tpu_raiden::RaidenWorkerEndpoints{20, "a_hi", {{"ep_hi:1", {}}}},
  });
  std::vector<Buffer> dst_buffers;
  dst_buffers.push_back(std::move(dst_buf));

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok()) << status.message();

  // node-10 worker got node-10's endpoints; node-20 worker got node-20's --
  // NOT what sorted index (a_hi first -> group[0]=node10) would produce.
  EXPECT_EQ(mock_lo.last_peer, "ep_lo:1");
  EXPECT_EQ(mock_hi.last_peer, "ep_hi:1");
}

// Strict matching: a source worker whose node_id has no destination group is an
// error (no silent fallback).
TEST_F(RaidenControllerTest, TransferBuffersStrictErrorsOnUnmatchedNodeId) {
  MockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address,
                        /*node_id=*/99);

  std::vector<Buffer> src_buffers;
  src_buffers.emplace_back(/*index=*/1, std::vector<BufferShard>{},
                           std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(/*index=*/2, std::vector<BufferShard>{}, std::nullopt,
                 ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dst_buf.set_remote_worker_endpoints(
      {::tpu_raiden::RaidenWorkerEndpoints{5, "other", {{"ep:1", {}}}}});
  std::vector<Buffer> dst_buffers;
  dst_buffers.push_back(std::move(dst_buf));

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()), HasSubstr("node_id"));
}

// Pull direction: the peer's worker endpoint groups arrive on the SRC buffer.
// Each local worker must receive exactly its own node_id's group, converted
// from remote_worker_endpoints into remote_descriptors before dispatch.
TEST_F(RaidenControllerTest, TransferBuffersMatchesSrcEndpointGroupsByNodeId) {
  auto test_server2 = CreateTestWorkerServer();
  ShardAwareMockTransferManager mock_lo;  // node_id 10
  ShardAwareMockTransferManager mock_hi;  // node_id 20
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_lo));
  test_server2->service->SetTransferManager(KVManagerHolder(&mock_hi));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  // Same reverse sort-order trap as the dst-side test above.
  RegisterAndInitWorker(*controller, "z_lo", test_server_->server_address,
                        /*node_id=*/10);
  RegisterAndInitWorker(*controller, "a_hi", test_server2->server_address,
                        /*node_id=*/20);

  Buffer src_buf(/*index=*/1, std::vector<BufferShard>{}, std::nullopt,
                 ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  src_buf.set_remote_worker_endpoints({
      ::tpu_raiden::RaidenWorkerEndpoints{
          10, "z_lo", {{"src_lo:1", {0, 1}}, {"src_lo:2", {2, 3}}}},
      ::tpu_raiden::RaidenWorkerEndpoints{20, "a_hi", {{"src_hi:1", {4, 5}}}},
  });
  std::vector<Buffer> src_buffers;
  src_buffers.push_back(std::move(src_buf));
  std::vector<Buffer> dst_buffers;
  dst_buffers.emplace_back(/*index=*/2, std::vector<BufferShard>{},
                           std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok()) << status.message();

  // Each worker pulled from its own node_id's group, with the shard tags
  // intact.
  EXPECT_EQ(mock_lo.vector_h2h_read_calls, 1);
  ASSERT_EQ(mock_lo.last_read_descriptors.size(), 2u);
  EXPECT_EQ(mock_lo.last_read_descriptors[0].endpoint, "src_lo:1");
  EXPECT_EQ(mock_lo.last_read_descriptors[0].shards,
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(mock_lo.last_read_descriptors[1].endpoint, "src_lo:2");

  EXPECT_EQ(mock_hi.vector_h2h_read_calls, 1);
  ASSERT_EQ(mock_hi.last_read_descriptors.size(), 1u);
  EXPECT_EQ(mock_hi.last_read_descriptors[0].endpoint, "src_hi:1");
}

// Strict matching on the src side too: a local worker whose node_id has no
// source group is a hard error, never a broadcast to some other node's shards.
// (Asserted with a SINGLE registered worker: the per-worker loop dispatches
// each RPC before validating the next, so with >1 worker an earlier RPC would
// already be in flight when the mismatch is detected.)
TEST_F(RaidenControllerTest, TransferBuffersUnmatchedSrcNodeIdFails) {
  ShardAwareMockTransferManager mock;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock));
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address,
                        /*node_id=*/99);

  Buffer src_buf(/*index=*/1, std::vector<BufferShard>{}, std::nullopt,
                 ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  src_buf.set_remote_worker_endpoints(
      {::tpu_raiden::RaidenWorkerEndpoints{5, "other", {{"ep:1", {}}}}});
  std::vector<Buffer> src_buffers;
  src_buffers.push_back(std::move(src_buf));
  std::vector<Buffer> dst_buffers;
  dst_buffers.emplace_back(/*index=*/2, std::vector<BufferShard>{},
                           std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()), HasSubstr("node_id"));
  // Never broadcast: no worker RPC was issued at all.
  EXPECT_EQ(mock.vector_h2h_read_calls, 0);
  EXPECT_EQ(mock.h2h_read_calls, 0);
}

// The controller rejects a second worker that registers a duplicate non-zero
// node_id; re-registration under the same worker_id is an allowed update.
TEST_F(RaidenControllerTest, RegisterWorkerRejectsDuplicateNodeId) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                               /*shard_size_bytes=*/512, ""));
  core::controller::RaidenControllerClient client(
      controller->controller_address());

  const std::string& addr = test_server_->server_address;
  ASSERT_TRUE(
      client.RegisterWorker("worker_0", addr, {{addr, {}}}, /*node_id=*/7)
          .ok());
  // Distinct worker, same node_id -> rejected (node_id check runs before any
  // worker-side buffer creation, so the address is irrelevant here).
  auto dup = client.RegisterWorker("worker_1", addr, {{addr, {}}},
                                   /*node_id=*/7);
  EXPECT_FALSE(dup.ok());
  // Same worker_id re-registering the same node_id -> allowed (update).
  auto same =
      client.RegisterWorker("worker_0", addr, {{addr, {}}}, /*node_id=*/7);
  EXPECT_TRUE(same.ok()) << same.message();
}

TEST_F(RaidenControllerTest, PreprovisionDisabledSkipsPhysicalBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/4, /*num_shards=*/2,
                               /*shard_size_bytes=*/256, "",
                               /*preprovision_worker_buffers=*/false));
  // Registration still succeeds (and still probes the worker's WorkerService
  // with an empty CreateBuffers)...
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // ...the logical-block ledger is fully usable...
  TF_ASSERT_OK_AND_ASSIGN(auto ids, controller->AllocateBlockIds(4));
  EXPECT_EQ(ids.size(), 4);
  ASSERT_TRUE(controller->DeallocateBlockIds(ids).ok());

  // ...but no physical buffers exist, so the Legacy Physical/BufferProto
  // mode reports the precondition instead of indexing an empty pool.
  EXPECT_THAT(controller->AllocateBuffers(1),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(RaidenControllerTest, AllocateBuffersAndDeallocateBuffersSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(
          unit_, std::vector<std::string>{test_server_->server_address},
          /*num_blocks=*/10, /*num_shards=*/2,
          /*shard_size_bytes=*/1024, ""));

  TF_ASSERT_OK_AND_ASSIGN(const auto& buffers,
                          controller->AllocateBuffers(/*num_blocks=*/3));
  ASSERT_EQ(buffers.size(), 3);
  EXPECT_FALSE(buffers[0].empty());
  EXPECT_EQ(buffers[0].index(), 0);
  EXPECT_EQ(buffers[1].index(), 1);
  EXPECT_EQ(buffers[2].index(), 2);
  EXPECT_EQ(buffers[0].shards().size(), 2);

  EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 3);
  ASSERT_TRUE(controller->DeallocateBuffers(buffers).ok());
  EXPECT_EQ(controller->block_manager()->num_locked_blocks(), 0);
}

TEST_F(RaidenControllerTest, TransferBuffersSingleBufferSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto buffers, controller->AllocateBuffers(2));
  buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto status =
      controller->TransferBuffers("worker_0", {buffers[0]}, {buffers[1]})
          .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersSpanBufferSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(2));
  src_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  src_buffers[1].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(2));

  auto status =
      controller->TransferBuffers("worker_0", src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(2, 3));
}

TEST_F(RaidenControllerTest, TransferBuffersSimplifiedH2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(1));

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(1));
  dst_buffers[0].set_remote_address("localhost:8080");

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersBufferProtoSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(const auto& protos, controller->Allocate(2));
  std::vector<Buffer> src_buffers;
  for (const auto& proto : protos) {
    src_buffers.push_back(Buffer::FromProto(proto, std::nullopt));
    src_buffers.back().set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  }
  std::vector<Buffer> dst_buffers;
  for (const auto& proto : protos) {
    dst_buffers.push_back(Buffer::FromProto(proto, std::nullopt));
    dst_buffers.back().set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  }

  auto status =
      controller->TransferBuffers("worker_0", src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(0, 1));
}

TEST_F(RaidenControllerTest, TransferBuffersBroadcastSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  Buffer src_buf(100, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(200, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto status = controller->TransferBuffers({src_buf}, {dst_buf}).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
}

TEST_F(RaidenControllerTest, TransferBuffersBroadcastH2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  Buffer src_buf(5, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(6, {}, "localhost:8080", ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status = controller->TransferBuffers({src_buf}, {dst_buf}).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2h_write_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(5));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(6));
}

TEST_F(RaidenControllerTest, TransferBuffersLocalDramToRemoteHbmSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(1));
  src_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(1));
  dst_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  dst_buffers[0].set_remote_address("remote_host:9090");

  TF_ASSERT_OK_AND_ASSIGN(auto staging_buffers, controller->AllocateBuffers(1));
  staging_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller->TransferBuffers(src_buffers, dst_buffers, staging_buffers)
          .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2d_write_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:9090");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersLocalHbmToRemoteDramSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(1));
  src_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(1));
  dst_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dst_buffers[0].set_remote_address("remote_host:9090");

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_write_calls, 1);
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:9090");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersRemoteHbmToLocalDramSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(1));
  src_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  src_buffers[0].set_remote_address("remote_host:9090");

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(1));
  dst_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  TF_ASSERT_OK_AND_ASSIGN(auto staging_buffers, controller->AllocateBuffers(1));
  staging_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller->TransferBuffers(src_buffers, dst_buffers, staging_buffers)
          .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_read_calls, 1);
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_write_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:9090");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersRemoteDramToLocalHbmSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(1));
  src_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  src_buffers[0].set_remote_address("remote_host:9090");

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(1));
  dst_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto status = controller->TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2d_read_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_write_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:9090");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
}

TEST_F(RaidenControllerTest, MapperAndRegisterBackends) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  EXPECT_EQ(controller->mapper(), nullptr);
  auto mapper = std::make_shared<kv_cache::storage::PosixPathMapper>(
      "/tmp/storage", "test_model", 1, 0);
  controller->SetMapper(mapper);
  EXPECT_EQ(controller->mapper(), mapper);

  std::vector<::tpu_sync::proto::BackendConfig> configs;
  ::tpu_sync::proto::BackendConfig config;
  config.set_name("PosixBackend");
  config.set_scheme("posix_test");
  configs.push_back(config);

  auto status = controller->RegisterBackends(configs);
  EXPECT_TRUE(status.ok());
}

TEST_F(RaidenControllerTest, ExecuteTransferBuffersgRPCSyncSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      RaidenController::Create(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                               /*shard_size_bytes=*/512, ""));
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(auto src_buffers, controller->AllocateBuffers(1));
  src_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  TF_ASSERT_OK_AND_ASSIGN(auto dst_buffers, controller->AllocateBuffers(1));
  dst_buffers[0].set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller->ExecuteTransferBuffersgRPCSync(src_buffers, dst_buffers);
  EXPECT_TRUE(status.ok());
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden

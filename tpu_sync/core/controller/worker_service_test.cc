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

#include <stdint.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/core/controller/worker_service_impl.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/proto/worker_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;



class WorkerServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestWorkerServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");
  }

  ::tpu_sync::rpc::RaidenIdProto unit_;
  std::unique_ptr<TestWorkerServer> test_server_;
};

TEST_F(WorkerServiceTest, CreateAndDeleteBuffersSuccess) {
  ::tpu_sync::proto::CreateBuffersRequest create_req;
  *create_req.mutable_unit() = unit_;
  auto* spec1 = create_req.add_buffers();
  spec1->set_num_shards(2);
  spec1->set_size_bytes(1024);
  auto* spec2 = create_req.add_buffers();
  spec2->set_num_shards(2);
  spec2->set_size_bytes(1024);

  auto create_resp_or = test_server_->client->CreateBuffers(create_req).Await();
  ASSERT_TRUE(create_resp_or.ok());
  EXPECT_TRUE(create_resp_or->success());
  ASSERT_EQ(create_resp_or->buffers_size(), 2);
  EXPECT_EQ(test_server_->service->GetBufferCount(), 4);

  const auto& buf1 = create_resp_or->buffers(0);
  const auto& buf2 = create_resp_or->buffers(1);
  ASSERT_EQ(buf1.buffer_handles_size(), 2);
  ASSERT_EQ(buf2.buffer_handles_size(), 2);

  uint64_t handle1 = buf1.buffer_handles(0).handle();
  uint64_t handle2 = buf1.buffer_handles(1).handle();
  EXPECT_NE(handle1, handle2);

  auto alloc_or = test_server_->service->GetBuffer(BufferHandle(handle1));
  ASSERT_TRUE(alloc_or.ok());
  EXPECT_EQ(alloc_or->size, 1024);
  EXPECT_NE(alloc_or->ptr, nullptr);

  ::tpu_sync::proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = unit_;
  *delete_req.add_sharded_buffers() = buf1;
  *delete_req.add_sharded_buffers() = buf2;

  auto delete_resp_or = test_server_->client->DeleteBuffers(delete_req).Await();
  ASSERT_TRUE(delete_resp_or.ok());
  EXPECT_TRUE(delete_resp_or->success());
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(WorkerServiceTest, CreateBuffersWithInvalidSpecFails) {
  ::tpu_sync::proto::CreateBuffersRequest create_req;
  *create_req.mutable_unit() = unit_;
  auto* spec = create_req.add_buffers();
  spec->set_num_shards(0);
  spec->set_size_bytes(512);

  auto create_resp_or = test_server_->client->CreateBuffers(create_req).Await();
  ASSERT_TRUE(create_resp_or.ok());
  EXPECT_FALSE(create_resp_or->success());
  EXPECT_THAT(create_resp_or->message(), HasSubstr("must be positive"));
}

TEST_F(WorkerServiceTest, DeleteNonExistentBufferFails) {
  ::tpu_sync::proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = unit_;
  auto* sharded_buf = delete_req.add_sharded_buffers();
  sharded_buf->add_buffer_handles()->set_handle(9999);

  auto delete_resp_or = test_server_->client->DeleteBuffers(delete_req).Await();
  ASSERT_TRUE(delete_resp_or.ok());
  EXPECT_FALSE(delete_resp_or->success());
  EXPECT_THAT(delete_resp_or->message(), HasSubstr("not found"));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_dst_buffers()->set_remote_address("localhost:8080");

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2h_read_calls, 0);
  EXPECT_EQ(mock_mgr.h2h_write_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hMissingPeerFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Peer address must be provided for H2H transfers"));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hInvalidCopySizeFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_copy_sizes(2);
  transfer->add_dst_buffers()->set_remote_address("localhost:8080");

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("H2H transfers only support copy size of 1"));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hOverflowFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(2147483648L);
  transfer->add_dst_offsets(20);
  transfer->add_dst_buffers()->set_remote_address("localhost:8080");

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "H2H transfer dispatch failed: Offset 2147483648 overflows int"));
}

TEST_F(WorkerServiceTest, TransferBuffersWithoutTransferManagerFails) {
  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer manager is not configured on WorkerService"));
}

TEST_F(WorkerServiceTest, TransferBuffersOffsetsMismatchFails) {
  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_dst_offsets(30);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Source and destination offsets must have the same "
                        "non-zero length"));
}

TEST_F(WorkerServiceTest, TransferBuffersCopySizesMismatchFails) {
  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_copy_sizes(1);
  transfer->add_copy_sizes(2);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "copy_sizes, if provided, must match the length of src_offsets"));
}

TEST_F(WorkerServiceTest, TransferBuffersD2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_src_offsets(30);
  transfer->add_dst_offsets(20);
  transfer->add_dst_offsets(40);
  transfer->add_copy_sizes(1);
  transfer->add_copy_sizes(2);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10, 30));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20, 40));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1, 2));
}

TEST_F(WorkerServiceTest, TransferBuffersRemoteD2hWithPeerSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(100);
  transfer->add_dst_offsets(200);
  transfer->add_dst_buffers()->set_remote_address("remote_host:1234");

  auto* staging_buf = transfer->add_staging_host_buffers();
  staging_buf->set_index(300);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_write_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:1234");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_staging_offsets, ElementsAre(300));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest,
       TransferBuffersRemoteD2hWithDstBufferRemoteAddressSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(100);
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(200);
  dst_buf->set_remote_address("remote_host:5678");

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_write_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:5678");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest,
       TransferBuffersRemoteD2hWithSrcBufferRemoteAddressSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(100);
  src_buf->set_remote_address("remote_host:5678");
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(200);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_write_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_read_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:5678");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest, TransferBuffersLocalD2hFallbackSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(100);
  transfer->add_dst_offsets(200);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.d2h_write_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest, TransferBuffersH2DSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->add_src_offsets(100);
  transfer->add_dst_offsets(200);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest, TransferBuffersRemoteH2dWithPeerSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->add_src_offsets(100);
  transfer->add_dst_offsets(200);
  transfer->add_dst_buffers()->set_remote_address("remote_host:1234");
  transfer->add_staging_host_buffers()->set_index(300);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_write_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:1234");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_staging_offsets, ElementsAre(300));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest,
       TransferBuffersRemoteH2dWithDstBufferRemoteAddressSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(100);
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(200);
  dst_buf->set_remote_address("remote_host:5678");

  auto* staging_buf = transfer->add_staging_host_buffers();
  staging_buf->set_index(300);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_write_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:5678");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_staging_offsets, ElementsAre(300));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest,
       TransferBuffersRemoteH2dWithSrcBufferRemoteAddressSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(100);
  src_buf->set_remote_address("remote_host:5678");
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(200);

  auto* staging_buf = transfer->add_staging_host_buffers();
  staging_buf->set_index(300);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_write_calls, 0);
  EXPECT_EQ(mock_mgr.d2h_read_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_read_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:5678");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_staging_offsets, ElementsAre(300));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

// Receiver-initiated pull to HBM: per-shard remote descriptors on the SRC
// buffer route to the vector H2dRead overload, and take precedence over the
// single-peer remote_address fallback.
TEST_F(WorkerServiceTest, TransferBuffersRemoteSrcHbmDstRunsVectorH2dRead) {
  ShardAwareMockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(100);
  // A stale single-peer address that must lose to the descriptors.
  src_buf->set_remote_address("ignored_host:1111");
  auto* ep0 = src_buf->add_remote_descriptors();
  ep0->set_endpoint("shard_host_a:2222");
  ep0->add_shards(0);
  auto* ep1 = src_buf->add_remote_descriptors();
  ep1->set_endpoint("shard_host_b:3333");
  ep1->add_shards(1);

  transfer->add_dst_buffers()->set_index(200);
  transfer->add_staging_host_buffers()->set_index(300);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.vector_h2d_read_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_read_calls, 0) << "string overload must not be used";
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_write_calls, 0);
  ASSERT_EQ(mock_mgr.last_h2d_read_descriptors.size(), 2u);
  EXPECT_EQ(mock_mgr.last_h2d_read_descriptors[0].endpoint,
            "shard_host_a:2222");
  EXPECT_EQ(mock_mgr.last_h2d_read_descriptors[1].endpoint,
            "shard_host_b:3333");
  // Flow order: remote host src -> local host staging -> local device dst.
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_staging_offsets, ElementsAre(300));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest, TransferBuffersLocalH2dFallbackSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->add_src_offsets(100);
  transfer->add_dst_offsets(200);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_write_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(WorkerServiceTest, TransferBuffersWithBufferProtosSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(10);
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(20);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20));
}

TEST_F(WorkerServiceTest,
       TransferBuffersInfersMemoryTypeAndPeerFromBufferProtos) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(10);
  src_buf->set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(20);
  dst_buf->set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dst_buf->set_remote_address("localhost:8080");

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2h_read_calls, 0);
  EXPECT_EQ(mock_mgr.h2h_write_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hReadRemoteSrcSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  auto* src_buf = transfer->add_src_buffers();
  src_buf->set_index(10);
  src_buf->set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  src_buf->set_remote_address("localhost:8080");
  auto* dst_buf = transfer->add_dst_buffers();
  dst_buf->set_index(20);
  dst_buf->set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2h_read_calls, 1);
  EXPECT_EQ(mock_mgr.h2h_write_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, testing::IsEmpty());
}

TEST_F(WorkerServiceTest, TransferBuffersWithInvalidBufferProtoFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  ::tpu_sync::proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_buffers();  // index not set
  transfer->add_dst_buffers();  // index not set

  auto status = test_server_->client->TransferBuffers(transfer_req).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("BufferProto missing valid index"));
}

TEST_F(WorkerServiceTest, TransferBackendBuffersPropagatesNotFoundErrorCode) {
  // 0. Unconfigured transfer_manager_ returns FAILED_PRECONDITION.
  ::tpu_sync::proto::TransferBackendBuffersRequest req;
  req.set_direction(::tpu_sync::proto::TRANSFER_DIR_RECALL);
  req.add_block_hashes("missing_block");
  req.add_hbm_block_ids(4);
  req.add_host_block_ids(1);
  req.add_backend_specs()->set_name("posix");
  EXPECT_THAT(
      test_server_->client->TransferBackendBuffers(req).Await(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Transfer manager is not configured")));

  MockTransferManager mock_mgr;
  kv_cache::BackendConfig cfg;
  cfg.type = "posix";
  cfg.parallelism.tp_rank = 0;
  cfg.parallelism.tp_size = 1;
  mock_mgr.RegisterKVBackends({cfg});
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  // 1. Pipelined completion failure (copy_future.value().Await() returns
  //    NOT_FOUND, as produced by PosixKVBackend::ReadAsync on ENOENT).
  mock_mgr.backend_future_status =
      absl::NotFoundError("Failed to open file for reading");
  EXPECT_THAT(test_server_->client->TransferBackendBuffers(req).Await(),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Failed to open file for reading")));
  mock_mgr.backend_future_status = absl::OkStatus();

  // 2. Synchronous dispatch failure (H2dReadFromBackend returns NOT_FOUND).
  mock_mgr.backend_transfer_status =
      absl::NotFoundError("Dispatch file not found");
  EXPECT_THAT(test_server_->client->TransferBackendBuffers(req).Await(),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Dispatch file not found")));
  mock_mgr.backend_transfer_status = absl::OkStatus();

  // 3. Key-mapping failure (MapKey returns INVALID_ARGUMENT on empty hash).
  ::tpu_sync::proto::TransferBackendBuffersRequest bad_key_req;
  bad_key_req.set_direction(::tpu_sync::proto::TRANSFER_DIR_RECALL);
  bad_key_req.add_block_hashes("");
  bad_key_req.add_hbm_block_ids(4);
  bad_key_req.add_host_block_ids(1);
  bad_key_req.add_backend_specs()->set_name("posix");
  EXPECT_THAT(test_server_->client->TransferBackendBuffers(bad_key_req).Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to map block hash")));

  // 4. Server-side request & backend precondition validation codes.
  ::tpu_sync::proto::TransferBackendBuffersRequest empty_req;
  empty_req.set_direction(::tpu_sync::proto::TRANSFER_DIR_RECALL);
  empty_req.add_backend_specs()->set_name("posix");
  EXPECT_THAT(test_server_->client->TransferBackendBuffers(empty_req).Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must specify at least one block hash")));

  ::tpu_sync::proto::TransferBackendBuffersRequest unknown_backend_req = req;
  unknown_backend_req.mutable_backend_specs(0)->set_name("unregistered");
  EXPECT_THAT(
      test_server_->client->TransferBackendBuffers(unknown_backend_req).Await(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("No backend registered for backend_name")));

  ::tpu_sync::proto::TransferBackendBuffersRequest bad_dir_req = req;
  bad_dir_req.set_direction(::tpu_sync::proto::TRANSFER_DIR_UNSPECIFIED);
  EXPECT_THAT(test_server_->client->TransferBackendBuffers(bad_dir_req).Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported transfer direction")));
}

TEST_F(WorkerServiceTest,
       TransferBackendBuffersDefaultsUnsetErrorCodeToInternal) {
  // Worker stub returning raw wire error_code values (0 / unset from older
  // workers, < 0, or > kUnauthenticated) so client-side ParseWorkerErrorCode
  // boundary handling is exercised directly over the wire without absl::Status
  // clamping out-of-range codes to kUnknown on the server first.
  class RawErrorCodeWorkerService final
      : public ::tpu_sync::proto::WorkerService::Service {
   public:
    int32_t raw_error_code = 0;
    std::string message = "Legacy worker failure without error_code";

    grpc::Status TransferBackendBuffers(
        grpc::ServerContext* /*context*/,
        const ::tpu_sync::proto::TransferBackendBuffersRequest* /*request*/,
        ::tpu_sync::proto::TransferBackendBuffersResponse* response) override {
      response->set_success(false);
      if (raw_error_code != 0) {
        response->set_error_code(raw_error_code);
      }
      response->set_message(message);
      return grpc::Status::OK;
    }
  };
  RawErrorCodeWorkerService raw_service;
  int port = 0;
  std::unique_ptr<grpc::Server> raw_server =
      test_util_internal::StartOnPrivateEphemeralPort(&raw_service, &port);
  ASSERT_NE(raw_server, nullptr);
  WorkerServiceClient raw_client(
      CreateWorkerServiceChannel(absl::StrCat("localhost:", port)));

  ::tpu_sync::proto::TransferBackendBuffersRequest req;
  req.set_direction(::tpu_sync::proto::TRANSFER_DIR_RECALL);
  req.add_block_hashes("block_0");
  req.add_hbm_block_ids(4);
  req.add_host_block_ids(1);
  req.add_backend_specs()->set_name("posix");

  // 1. Unset error_code (0 on the wire) defaults to kInternal.
  EXPECT_THAT(raw_client.TransferBackendBuffers(req).Await(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Legacy worker failure without error_code")));

  // 2. Negative error_code (< 0) defaults to kInternal.
  raw_service.raw_error_code = -1;
  raw_service.message = "Negative error code";
  EXPECT_THAT(
      raw_client.TransferBackendBuffers(req).Await(),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Negative error code")));

  // 3. Out-of-upper-bound error_code (> kUnauthenticated) defaults to
  //    kInternal.
  raw_service.raw_error_code = 999;
  raw_service.message = "Out-of-upper-bound error code";
  EXPECT_THAT(raw_client.TransferBackendBuffers(req).Await(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Out-of-upper-bound error code")));

  raw_server->Shutdown();
}
}  // namespace
}  // namespace controller
}  // namespace tpu_raiden

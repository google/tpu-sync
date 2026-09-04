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

#include "tpu_sync/frameworks/torch/kv_cache_manager.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/controller/controller_service.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"
#include "torch/torch.h"

namespace tpu_raiden {
namespace torch {
namespace {

class KVCacheManagerTorchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  std::unique_ptr<xla::PjRtClient> client_;
};

TEST_F(KVCacheManagerTorchTest, ConstructorSucceedsWithMocks) {
  // Create a real CPU PJRT buffer
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            data.data(), xla::F32, {8, 1024},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  // Create a CPU PyTorch tensor
  at::Tensor tensor = ::torch::zeros({8, 1024}, ::torch::kFloat32);

  // Register the mapping
  RegisterMockTensor(tensor, pjrt_buffer.get());

  // Prepare input for KVCacheManager
  std::vector<std::vector<at::Tensor>> device_tensors = {{tensor}};

  // Construct KVCacheManager (should use mocks internally)
  KVCacheManager manager(device_tensors,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8);

  EXPECT_EQ(manager.slice_byte_size(), 1024 * sizeof(float));
}

TEST_F(KVCacheManagerTorchTest, ConstructorWithRawPjRtBuffersSucceeds) {
  // Create a real CPU PJRT buffer
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            data.data(), xla::F32, {8, 1024},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  // Prepare 2D vector of raw PjRtBuffer pointers
  std::vector<std::vector<xla::PjRtBuffer*>> device_buffers = {
      {pjrt_buffer.get()}};

  // Construct KVCacheManager directly from raw PjRtBuffers
  KVCacheManager manager(device_buffers,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8,
                         /*unsafe_skip_buffer_lock=*/true);

  EXPECT_EQ(manager.num_layers(), 1);
  EXPECT_EQ(manager.num_shards(), 1);
  EXPECT_EQ(manager.slice_byte_size(), 1024 * sizeof(float));
}

TEST_F(KVCacheManagerTorchTest, GrpcServerOptionalAndOffByDefault) {
  KVCacheManager mgr_default(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/1024, /*node_id=*/0,
                             /*local_port=*/std::nullopt,
                             /*host_blocks_to_allocate=*/8);
  EXPECT_EQ(mgr_default.GetRaidenWorkerPort(), 0);

  KVCacheManager mgr_explicit_off(/*num_layers=*/1, /*num_shards=*/1,
                                  /*slice_byte_size=*/1024, /*node_id=*/0,
                                  /*local_port=*/std::nullopt,
                                  /*host_blocks_to_allocate=*/8,
                                  /*parallelism=*/1, /*raiden_worker_port=*/0,
                                  /*raiden_controller_address=*/std::nullopt);
  EXPECT_EQ(mgr_explicit_off.GetRaidenWorkerPort(), 0);

  KVCacheManager mgr_started(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/1024, /*node_id=*/0,
                             /*local_port=*/std::nullopt,
                             /*host_blocks_to_allocate=*/8,
                             /*parallelism=*/1, /*raiden_worker_port=*/0,
                             /*raiden_controller_address=*/"localhost:12345");
  EXPECT_GT(mgr_started.GetRaidenWorkerPort(), 0);
}

TEST_F(KVCacheManagerTorchTest, WorkerSelfRegistrationWithControllerSuccess) {
  auto test_server = core::controller::CreateTestControllerServer();
  ASSERT_NE(test_server, nullptr);

  std::string raiden_controller_address = test_server->server_address;

  KVCacheManager mgr(/*num_layers=*/1, /*num_shards=*/1,
                     /*slice_byte_size=*/1024, /*node_id=*/0,
                     /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/8,
                     /*parallelism=*/1, /*raiden_worker_port=*/0,
                     raiden_controller_address, "torch_worker_0");

  auto workers =
      test_server->service->worker_registry()->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "torch_worker_0");
  EXPECT_TRUE(absl::StrContains(workers[0].raiden_worker_endpoint,
                                std::to_string(mgr.GetRaidenWorkerPort())));
}

TEST_F(KVCacheManagerTorchTest,
       WorkerSelfRegistrationUsesDataEndpointsWhenControlPortActive) {
  auto test_server = core::controller::CreateTestControllerServer();
  ASSERT_NE(test_server, nullptr);

  std::string raiden_controller_address = test_server->server_address;

  // Create CPU tensor and register mock buffer
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            data.data(), xla::F32, {8, 1024},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));
  at::Tensor tensor = ::torch::zeros({8, 1024}, ::torch::kFloat32);
  RegisterMockTensor(tensor, pjrt_buffer.get());

  std::vector<std::vector<at::Tensor>> device_tensors = {{tensor}};

  // Construct KVCacheManager with local_control_port=50051 (> 0).
  KVCacheManager mgr(
      {tensor}, /*node_id=*/0,
      /*local_control_port=*/50051,
      /*max_blocks=*/8, /*num_slots=*/8,
      /*timeout_s=*/120.0, /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*raiden_worker_port=*/0,
      raiden_controller_address,
      "torch_worker_ctrl");

  // Verify that get_local_endpoints() returns the CONTROL port (50051)
  // while get_local_data_endpoints() returns the DATA transport port.
  auto local_control_eps = mgr.torch_manager()->get_local_endpoints();
  auto local_data_eps = mgr.torch_manager()->get_local_data_endpoints();

  ASSERT_FALSE(local_control_eps.empty());
  ASSERT_FALSE(local_data_eps.empty());

  EXPECT_TRUE(absl::StrContains(local_control_eps[0].endpoint, "50051"));
  EXPECT_NE(local_control_eps[0].endpoint, local_data_eps[0].endpoint);

  auto workers =
      test_server->service->worker_registry()->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "torch_worker_ctrl");

  // Verify registered transfer endpoints carry the DATA transport port (local_data_eps), NOT the control port (local_control_eps).
  ASSERT_FALSE(workers[0].raiden_transfer_endpoints.empty());
  EXPECT_EQ(workers[0].raiden_transfer_endpoints[0].endpoint, local_data_eps[0].endpoint);
  EXPECT_NE(workers[0].raiden_transfer_endpoints[0].endpoint, local_control_eps[0].endpoint);
}

}  // namespace
}  // namespace torch
}  // namespace tpu_raiden

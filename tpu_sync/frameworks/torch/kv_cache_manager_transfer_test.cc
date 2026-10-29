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

#include <memory>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/frameworks/torch/kv_cache_manager.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"

namespace tpu_raiden::torch {
namespace {

class KVCacheManagerTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  std::unique_ptr<xla::PjRtClient> client_;
};

TEST_F(KVCacheManagerTransferTest, Basic) {
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

  // Create a CPU PyTorch tensor.
  at::Tensor tensor = at::zeros({8, 1024}, at::kFloat);

  // Register the mapping
  RegisterMockTensor(tensor, pjrt_buffer.get());

  std::vector<at::Tensor> kv_caches = {tensor};

  // Construct KVCacheManager with transfer enabled
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KVCacheManager> engine,
                          KVCacheManager::CreateWithNodeId(
                              kv_caches, /*node_id=*/0,
                              /*local_control_port=*/0, /*max_blocks=*/2,
                              /*num_slots=*/2, /*timeout_s=*/10.0,
                              /*unsafe_skip_buffer_lock=*/true));

  EXPECT_EQ(engine->kv_caches().size(), 1);
}

TEST_F(KVCacheManagerTransferTest,
       CreateWithNodeIdConfiguresBoundedStagingWindow) {
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

  at::Tensor tensor = at::zeros({8, 1024}, at::kFloat);
  RegisterMockTensor(tensor, pjrt_buffer.get());
  std::vector<at::Tensor> kv_caches = {tensor};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KVCacheManager> engine,
      KVCacheManager::CreateWithNodeId(
          kv_caches, /*node_id=*/0, /*local_control_port=*/0, /*max_blocks=*/8,
          /*num_slots=*/2, /*timeout_s=*/10.0, /*unsafe_skip_buffer_lock=*/true,
          /*parallelism=*/4, /*listener_port=*/std::nullopt,
          /*raiden_worker_port=*/0, /*raiden_controller_address=*/std::nullopt,
          /*worker_id=*/std::nullopt, /*enable_shm=*/false,
          /*host_blocks_to_allocate=*/std::nullopt, /*window_blocks=*/2,
          /*window_depth=*/2));

  ASSERT_NE(engine->torch_manager(), nullptr);
  EXPECT_EQ(engine->torch_manager()->window_blocks(), 2);
  EXPECT_EQ(engine->torch_manager()->window_depth(), 2);
  ASSERT_NE(engine->torch_manager()->staging_allocator(), nullptr);
  EXPECT_TRUE(
      engine->torch_manager()->staging_allocator()->dynamic_host_staging());
  EXPECT_EQ(engine->torch_manager()->base()->GetHostSize(0, 0),
            4 * engine->slice_byte_size());
}

}  // namespace
}  // namespace tpu_raiden::torch

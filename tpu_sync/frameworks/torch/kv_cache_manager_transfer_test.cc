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

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/from_blob.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/frameworks/torch/kv_cache_manager.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"

namespace tpu_raiden::torch {
namespace {

using ::tsl::testing::IsOk;

class KVCacheManagerTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  struct DeviceKVCache {
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer;
    at::Tensor tensor;
  };

  absl::StatusOr<DeviceKVCache> CreateMockKVCache(
      int num_blocks, int block_elements, const std::vector<float>& init_data) {
    TF_ASSIGN_OR_RETURN(
        xla::PjRtMemorySpace * memory_space,
        client_->addressable_devices()[0]->default_memory_space());
    TF_ASSIGN_OR_RETURN(
        auto pjrt_buffer,
        client_->BufferFromHostBuffer(
            init_data.data(), xla::F32, {num_blocks, block_elements},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));

    at::Tensor tensor = at::zeros({num_blocks, block_elements}, at::kFloat);
    RegisterMockTensor(tensor, pjrt_buffer.get());
    return DeviceKVCache{std::move(pjrt_buffer), tensor};
  }

  std::unique_ptr<xla::PjRtClient> client_;
};

TEST_F(KVCacheManagerTransferTest, Basic) {
  constexpr int kNumBlocks = 8;
  constexpr int kBlockElements = 1024;
  std::vector<float> data(kNumBlocks * kBlockElements, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(auto cache,
                          CreateMockKVCache(kNumBlocks, kBlockElements, data));

  std::vector<at::Tensor> kv_caches = {cache.tensor};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KVCacheManager> engine,
                          KVCacheManager::CreateWithNodeId(
                              kv_caches, /*node_id=*/0,
                              /*local_control_port=*/0, /*max_blocks=*/2,
                              /*num_slots=*/2, /*timeout_s=*/10.0,
                              /*unsafe_skip_buffer_lock=*/true));

  EXPECT_EQ(engine->kv_caches().size(), 1);
  EXPECT_EQ(engine->num_layers(), 1);
  EXPECT_EQ(engine->num_shards(), 1);
  EXPECT_EQ(engine->slice_byte_size(), kBlockElements * sizeof(float));
}

// Functional test: Offloads device KV blocks to host DRAM (D2H), verifies host
// DRAM content, reloads to different device blocks (H2D), and verifies exact
// numerical round-trip accuracy on device memory.
TEST_F(KVCacheManagerTransferTest, FunctionalD2hOffloadAndH2dReloadRoundTrip) {
  constexpr int kNumBlocks = 4;
  constexpr int kBlockElements = 1024;
  constexpr size_t kBlockBytes = kBlockElements * sizeof(float);

  // Initialize 4 distinct numerical patterns for each block
  std::vector<float> init_data(kNumBlocks * kBlockElements);
  for (int b = 0; b < kNumBlocks; ++b) {
    for (int i = 0; i < kBlockElements; ++i) {
      init_data[b * kBlockElements + i] = static_cast<float>(b * 1000 + i);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, CreateMockKVCache(kNumBlocks, kBlockElements, init_data));

  std::vector<at::Tensor> kv_caches = {cache.tensor};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KVCacheManager> engine,
      KVCacheManager::CreateWithNodeId(kv_caches, /*node_id=*/0,
                                       /*local_control_port=*/0,
                                       /*max_blocks=*/kNumBlocks,
                                       /*num_slots=*/2, /*timeout_s=*/10.0,
                                       /*unsafe_skip_buffer_lock=*/true));

  // 1. Offload device blocks {0, 2} into host DRAM staging blocks {0, 1}
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_future,
                          engine->D2h(/*src_offsets_major_dim=*/{0, 2},
                                      /*dst_offsets_major_dim=*/{0, 1},
                                      /*copy_sizes_major_dim=*/{1, 1}));
  ASSERT_THAT(d2h_future.Await(), IsOk());

  // 2. Verify functional data integrity directly on host DRAM
  TF_ASSERT_OK_AND_ASSIGN(std::string host_blk0, engine->ReadBlockBytes(0, 0));
  TF_ASSERT_OK_AND_ASSIGN(std::string host_blk1, engine->ReadBlockBytes(0, 1));
  ASSERT_EQ(host_blk0.size(), kBlockBytes);
  ASSERT_EQ(host_blk1.size(), kBlockBytes);

  const auto* host_floats_0 = reinterpret_cast<const float*>(host_blk0.data());
  const auto* host_floats_1 = reinterpret_cast<const float*>(host_blk1.data());
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(host_floats_0[i], static_cast<float>(0 * 1000 + i));
    EXPECT_EQ(host_floats_1[i], static_cast<float>(2 * 1000 + i));
  }

  // 3. Reload from host DRAM into device blocks {1, 3}:
  //    host block 0 (original block 0) -> device block 1
  //    host block 1 (original block 2) -> device block 3
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_future,
                          engine->H2d(/*src_offsets_major_dim=*/{0, 1},
                                      /*dst_offsets_major_dim=*/{1, 3},
                                      /*copy_sizes_major_dim=*/{1, 1}));
  ASSERT_THAT(h2d_future.Await(), IsOk());

  // 4. Read back device memory to verify numerical round-trip accuracy
  TF_ASSERT_OK_AND_ASSIGN(auto literal, cache.pjrt_buffer->ToLiteralSync());
  auto device_floats = literal->data<float>();
  ASSERT_EQ(device_floats.size(), kNumBlocks * kBlockElements);

  // Device block 1 must now contain original block 0
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[1 * kBlockElements + i],
              static_cast<float>(0 * 1000 + i));
  }
  // Device block 3 must now contain original block 2
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[3 * kBlockElements + i],
              static_cast<float>(2 * 1000 + i));
  }
  // Untouched device blocks 0 and 2 must remain intact
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[0 * kBlockElements + i],
              static_cast<float>(0 * 1000 + i));
    EXPECT_EQ(device_floats[2 * kBlockElements + i],
              static_cast<float>(2 * 1000 + i));
  }
}

// Functional test: Verifies dynamic host DRAM allocation on offload
// (D2hAutoAllocate) followed by reloading to target device blocks.
TEST_F(KVCacheManagerTransferTest, FunctionalD2hAutoAllocateAndReload) {
  constexpr int kNumBlocks = 4;
  constexpr int kBlockElements = 1024;

  std::vector<float> init_data(kNumBlocks * kBlockElements);
  for (int b = 0; b < kNumBlocks; ++b) {
    for (int i = 0; i < kBlockElements; ++i) {
      init_data[b * kBlockElements + i] = static_cast<float>(b * 500 + i);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, CreateMockKVCache(kNumBlocks, kBlockElements, init_data));

  std::vector<std::vector<at::Tensor>> device_tensors = {{cache.tensor}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KVCacheManager> engine,
      KVCacheManager::Create(device_tensors, /*local_port=*/std::nullopt,
                             /*host_blocks_to_allocate=*/kNumBlocks,
                             /*unsafe_skip_buffer_lock=*/true));

  // Offload device blocks {1, 3} using dynamic host DRAM allocation
  TF_ASSERT_OK_AND_ASSIGN(
      auto d2h_res, engine->D2hAutoAllocate(/*src_offsets_major_dim=*/{1, 3},
                                            /*copy_sizes_major_dim=*/{1, 1}));
  const std::vector<int>& allocated_host_ids = d2h_res.first;
  ASSERT_EQ(allocated_host_ids.size(), 2);
  EXPECT_NE(allocated_host_ids[0], allocated_host_ids[1]);

  ASSERT_THAT(d2h_res.second.Await(), IsOk());

  // Reload the dynamically allocated host blocks into device blocks {0, 2}
  std::vector<int64_t> host_src = {allocated_host_ids[0],
                                   allocated_host_ids[1]};
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_future,
                          engine->H2d(/*src_offsets_major_dim=*/host_src,
                                      /*dst_offsets_major_dim=*/{0, 2},
                                      /*copy_sizes_major_dim=*/{1, 1}));
  ASSERT_THAT(h2d_future.Await(), IsOk());

  // Verify device blocks {0, 2} now match original blocks {1, 3}
  TF_ASSERT_OK_AND_ASSIGN(auto literal, cache.pjrt_buffer->ToLiteralSync());
  auto device_floats = literal->data<float>();

  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[0 * kBlockElements + i],
              static_cast<float>(1 * 500 + i));
    EXPECT_EQ(device_floats[2 * kBlockElements + i],
              static_cast<float>(3 * 500 + i));
  }
}

// Functional test: Tests direct host DRAM manipulation via WriteBlockBytes
// followed by H2D reloading into device memory.
TEST_F(KVCacheManagerTransferTest, FunctionalHostWriteAndReloadIntoDevice) {
  constexpr int kNumBlocks = 4;
  constexpr int kBlockElements = 1024;
  constexpr size_t kBlockBytes = kBlockElements * sizeof(float);

  std::vector<float> init_data(kNumBlocks * kBlockElements, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, CreateMockKVCache(kNumBlocks, kBlockElements, init_data));

  std::vector<at::Tensor> kv_caches = {cache.tensor};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KVCacheManager> engine,
      KVCacheManager::CreateWithNodeId(kv_caches, /*node_id=*/0,
                                       /*local_control_port=*/0,
                                       /*max_blocks=*/kNumBlocks,
                                       /*num_slots=*/2, /*timeout_s=*/10.0,
                                       /*unsafe_skip_buffer_lock=*/true));

  // Synthesize new KV data for host DRAM block 1
  std::vector<float> custom_pattern(kBlockElements);
  for (int i = 0; i < kBlockElements; ++i) {
    custom_pattern[i] = 42.0f + static_cast<float>(i);
  }
  std::string payload(reinterpret_cast<const char*>(custom_pattern.data()),
                      kBlockBytes);

  ASSERT_THAT(engine->WriteBlockBytes(/*layer_idx=*/0, /*block_id=*/1, payload),
              IsOk());

  // Reload host block 1 into device block 2
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_future,
                          engine->H2d(/*src_offsets_major_dim=*/{1},
                                      /*dst_offsets_major_dim=*/{2},
                                      /*copy_sizes_major_dim=*/{1}));
  ASSERT_THAT(h2d_future.Await(), IsOk());

  // Verify device block 2 accurately contains the custom pattern
  TF_ASSERT_OK_AND_ASSIGN(auto literal, cache.pjrt_buffer->ToLiteralSync());
  auto device_floats = literal->data<float>();

  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[2 * kBlockElements + i], custom_pattern[i]);
  }
  // Other blocks must remain 0.0f
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[0 * kBlockElements + i], 0.0f);
    EXPECT_EQ(device_floats[1 * kBlockElements + i], 0.0f);
    EXPECT_EQ(device_floats[3 * kBlockElements + i], 0.0f);
  }
}

// Functional test: Offloads and reloads using PyTorch at::Tensor object tensors
// directly, verifying host PyTorch tensor integration and numerical equality.
TEST_F(KVCacheManagerTransferTest, FunctionalObjectTensorOffloadAndReload) {
  constexpr int kNumBlocks = 4;
  constexpr int kBlockElements = 1024;
  constexpr size_t kBlockBytes = kBlockElements * sizeof(float);

  std::vector<float> init_data(kNumBlocks * kBlockElements);
  for (int b = 0; b < kNumBlocks; ++b) {
    for (int i = 0; i < kBlockElements; ++i) {
      init_data[b * kBlockElements + i] = static_cast<float>(b * 250 + i);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, CreateMockKVCache(kNumBlocks, kBlockElements, init_data));

  std::vector<at::Tensor> kv_caches = {cache.tensor};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KVCacheManager> engine,
      KVCacheManager::CreateWithNodeId(kv_caches, /*node_id=*/0,
                                       /*local_control_port=*/0,
                                       /*max_blocks=*/kNumBlocks,
                                       /*num_slots=*/2, /*timeout_s=*/10.0,
                                       /*unsafe_skip_buffer_lock=*/true));

  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pool_size =
      ((kBlockBytes + page_size - 1) / page_size) * page_size;
  void* pool = nullptr;
  ASSERT_EQ(posix_memalign(&pool, page_size, pool_size), 0);
  std::memset(pool, 0, pool_size);

  engine->SetSharedMemoryMappedForTest(reinterpret_cast<uintptr_t>(pool),
                                       pool_size);

  // Create PyTorch CPU tensor [num_ranks=1, num_layers=1, page_nbytes]
  const c10::TensorOptions obj_options = c10::TensorOptions().dtype(at::kChar);
  at::Tensor obj =
      at::from_blob(static_cast<uint8_t*>(pool),
                    {1, 1, static_cast<int64_t>(kBlockBytes)}, obj_options);

  // 1. Offload device block 1 into obj PyTorch tensor
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_obj, engine->D2h({1}, {obj}, /*rank_id=*/0));
  ASSERT_THAT(d2h_obj.Await(), IsOk());

  // 2. Verify obj PyTorch tensor contains block 1 data
  const auto* obj_floats = reinterpret_cast<const float*>(obj.data_ptr());
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(obj_floats[i], static_cast<float>(1 * 250 + i));
  }

  // 3. Reload obj tensor into device block 3
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_obj, engine->H2d({3}, {obj}, /*rank_id=*/0));
  ASSERT_THAT(h2d_obj.Await(), IsOk());

  // 4. Verify device block 3 now contains block 1 data
  TF_ASSERT_OK_AND_ASSIGN(auto literal, cache.pjrt_buffer->ToLiteralSync());
  auto device_floats = literal->data<float>();
  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(device_floats[3 * kBlockElements + i],
              static_cast<float>(1 * 250 + i));
  }

  engine->ResetSharedMemoryMappedForTest();
  free(pool);
}

// Functional test: Verifies multi-layer KV cache offloading (D2H) and reloading
// (H2D) across multiple transformer layers simultaneously.
TEST_F(KVCacheManagerTransferTest,
       FunctionalMultiLayerOffloadAndReloadRoundTrip) {
  constexpr int kNumLayers = 2;
  constexpr int kNumBlocks = 4;
  constexpr int kBlockElements = 1024;
  constexpr size_t kBlockBytes = kBlockElements * sizeof(float);

  std::vector<float> data_l0(kNumBlocks * kBlockElements);
  std::vector<float> data_l1(kNumBlocks * kBlockElements);
  for (int b = 0; b < kNumBlocks; ++b) {
    for (int i = 0; i < kBlockElements; ++i) {
      data_l0[b * kBlockElements + i] = static_cast<float>(1000 * b + i);
      data_l1[b * kBlockElements + i] = static_cast<float>(2000 * b + i);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto cache_l0, CreateMockKVCache(kNumBlocks, kBlockElements, data_l0));
  TF_ASSERT_OK_AND_ASSIGN(
      auto cache_l1, CreateMockKVCache(kNumBlocks, kBlockElements, data_l1));

  std::vector<at::Tensor> kv_caches = {cache_l0.tensor, cache_l1.tensor};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KVCacheManager> engine,
      KVCacheManager::CreateWithNodeId(kv_caches, /*node_id=*/0,
                                       /*local_control_port=*/0,
                                       /*max_blocks=*/kNumBlocks,
                                       /*num_slots=*/2, /*timeout_s=*/10.0,
                                       /*unsafe_skip_buffer_lock=*/true));

  EXPECT_EQ(engine->num_layers(), kNumLayers);

  // 1. Offload blocks {0, 1} across both layers into host DRAM blocks {0, 1}
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_future,
                          engine->D2h(/*src_offsets_major_dim=*/{0, 1},
                                      /*dst_offsets_major_dim=*/{0, 1},
                                      /*copy_sizes_major_dim=*/{1, 1}));
  ASSERT_THAT(d2h_future.Await(), IsOk());

  // 2. Verify both layers independently in host DRAM
  TF_ASSERT_OK_AND_ASSIGN(std::string host_l0_b0, engine->ReadBlockBytes(0, 0));
  TF_ASSERT_OK_AND_ASSIGN(std::string host_l0_b1, engine->ReadBlockBytes(0, 1));
  TF_ASSERT_OK_AND_ASSIGN(std::string host_l1_b0, engine->ReadBlockBytes(1, 0));
  TF_ASSERT_OK_AND_ASSIGN(std::string host_l1_b1, engine->ReadBlockBytes(1, 1));

  ASSERT_EQ(host_l0_b0.size(), kBlockBytes);
  ASSERT_EQ(host_l1_b0.size(), kBlockBytes);

  const auto* l0_f0 = reinterpret_cast<const float*>(host_l0_b0.data());
  const auto* l0_f1 = reinterpret_cast<const float*>(host_l0_b1.data());
  const auto* l1_f0 = reinterpret_cast<const float*>(host_l1_b0.data());
  const auto* l1_f1 = reinterpret_cast<const float*>(host_l1_b1.data());

  for (int i = 0; i < kBlockElements; ++i) {
    EXPECT_EQ(l0_f0[i], static_cast<float>(1000 * 0 + i));
    EXPECT_EQ(l0_f1[i], static_cast<float>(1000 * 1 + i));
    EXPECT_EQ(l1_f0[i], static_cast<float>(2000 * 0 + i));
    EXPECT_EQ(l1_f1[i], static_cast<float>(2000 * 1 + i));
  }

  // 3. Reload host blocks {0, 1} into device blocks {2, 3} across both layers
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_future,
                          engine->H2d(/*src_offsets_major_dim=*/{0, 1},
                                      /*dst_offsets_major_dim=*/{2, 3},
                                      /*copy_sizes_major_dim=*/{1, 1}));
  ASSERT_THAT(h2d_future.Await(), IsOk());

  // 4. Verify round-trip accuracy on both layer device buffers
  TF_ASSERT_OK_AND_ASSIGN(auto literal_l0,
                          cache_l0.pjrt_buffer->ToLiteralSync());
  TF_ASSERT_OK_AND_ASSIGN(auto literal_l1,
                          cache_l1.pjrt_buffer->ToLiteralSync());

  auto dev_l0 = literal_l0->data<float>();
  auto dev_l1 = literal_l1->data<float>();

  for (int i = 0; i < kBlockElements; ++i) {
    // Layer 0: block 2 == original block 0, block 3 == original block 1
    EXPECT_EQ(dev_l0[2 * kBlockElements + i], static_cast<float>(1000 * 0 + i));
    EXPECT_EQ(dev_l0[3 * kBlockElements + i], static_cast<float>(1000 * 1 + i));
    // Layer 1: block 2 == original block 0, block 3 == original block 1
    EXPECT_EQ(dev_l1[2 * kBlockElements + i], static_cast<float>(2000 * 0 + i));
    EXPECT_EQ(dev_l1[3 * kBlockElements + i], static_cast<float>(2000 * 1 + i));
  }
}

}  // namespace
}  // namespace tpu_raiden::torch

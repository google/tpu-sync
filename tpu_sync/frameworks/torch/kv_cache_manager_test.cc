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

#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/from_blob.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/controller/controller_service.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"

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
  at::Tensor tensor = at::zeros({8, 1024}, at::kFloat);

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
  at::Tensor tensor = at::zeros({8, 1024}, at::kFloat);
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

  // Verify registered transfer endpoints carry the DATA transport port
  // (local_data_eps), NOT the control port (local_control_eps).
  ASSERT_FALSE(workers[0].raiden_transfer_endpoints.empty());
  EXPECT_EQ(workers[0].raiden_transfer_endpoints[0].endpoint,
            local_data_eps[0].endpoint);
  EXPECT_NE(workers[0].raiden_transfer_endpoints[0].endpoint,
            local_control_eps[0].endpoint);
}

TEST_F(KVCacheManagerTorchTest, MapSharedMemoryValidation) {
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
  std::vector<std::vector<at::Tensor>> device_tensors = {{tensor}};

  KVCacheManager manager(device_tensors,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8);

  EXPECT_FALSE(manager.is_shared_memory_mapped());

  const int64_t page_size_val = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size_val, 0);
  const size_t page_size = static_cast<size_t>(page_size_val);

  void* aligned_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&aligned_ptr, page_size, page_size * 2), 0);
  ASSERT_NE(aligned_ptr, nullptr);

  // Null pointer check
  EXPECT_FALSE(manager.MapSharedMemory(0, page_size).ok());

  // Zero size check
  EXPECT_FALSE(manager.MapSharedMemory(
      reinterpret_cast<uintptr_t>(aligned_ptr), 0).ok());

  // Unaligned address check
  EXPECT_FALSE(manager.MapSharedMemory(
      reinterpret_cast<uintptr_t>(aligned_ptr) + 1, page_size).ok());

  // Unaligned size check
  EXPECT_FALSE(manager.MapSharedMemory(
      reinterpret_cast<uintptr_t>(aligned_ptr), page_size + 1).ok());

  // Address overflow check
  EXPECT_FALSE(manager.MapSharedMemory(
      std::numeric_limits<uintptr_t>::max() - page_size + 1,
      page_size * 2).ok());

  // Unmap when not mapped
  EXPECT_FALSE(manager.UnmapSharedMemory().ok());

  // Calling MapSharedMemory delegates to client->DmaMap, which fails with
  // Unimplemented on CPU PJRT client, leaving mapping state unmapped.
  absl::Status status = manager.MapSharedMemory(
      reinterpret_cast<uintptr_t>(aligned_ptr), page_size);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(manager.is_shared_memory_mapped());

  free(aligned_ptr);
}

TEST_F(KVCacheManagerTorchTest, H2dD2hWithObjectTensorsValidation) {
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
  std::vector<std::vector<at::Tensor>> device_tensors = {{tensor}};

  KVCacheManager manager(device_tensors,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8);

  const int64_t page_nbytes = manager.slice_byte_size();
  EXPECT_EQ(page_nbytes, 1024 * sizeof(float));

  // Valid 3D object tensor: [num_ranks=2, num_layers=1, page_nbytes=4096]
  at::Tensor valid_obj = at::zeros({2, 1, page_nbytes}, at::kChar);

  // Empty block_ids
  EXPECT_FALSE(manager.H2d({}, {}, 0).ok());

  // Size mismatch
  EXPECT_FALSE(manager.H2d({0}, {}, 0).ok());

  // Calling H2d/D2h while unmapped fails with FailedPrecondition
  EXPECT_FALSE(manager.H2d({0}, {valid_obj}, 0).ok());
  EXPECT_FALSE(manager.D2h({0}, {valid_obj}, 0).ok());

  // Simulate mapped shared memory via test helper
  manager.SetSharedMemoryMappedForTest(0x1000000, 1024 * 1024);
  EXPECT_TRUE(manager.is_shared_memory_mapped());

  // Duplicate mapping attempt fails
  EXPECT_FALSE(manager.MapSharedMemory(0x1000000, 4096).ok());

  // Out-of-bounds block_id (max_blocks is 8)
  EXPECT_FALSE(manager.H2d({8}, {valid_obj}, 0).ok());
  EXPECT_FALSE(manager.H2d({-1}, {valid_obj}, 0).ok());

  // Duplicate block_id
  at::Tensor valid_obj2 = at::zeros({2, 1, page_nbytes}, at::kChar);
  EXPECT_FALSE(manager.H2d({0, 0}, {valid_obj, valid_obj2}, 0).ok());

  // Non-3D tensor
  at::Tensor non_3d = at::zeros({2, page_nbytes}, at::kChar);
  EXPECT_FALSE(manager.H2d({0}, {non_3d}, 0).ok());

  // Non-int8 dtype
  at::Tensor float_obj = at::zeros({2, 1, page_nbytes}, at::kFloat);
  EXPECT_FALSE(manager.H2d({0}, {float_obj}, 0).ok());

  // num_layers mismatch (2 layers vs 1 layer)
  at::Tensor wrong_layers = at::zeros({2, 2, page_nbytes}, at::kChar);
  EXPECT_FALSE(manager.H2d({0}, {wrong_layers}, 0).ok());

  // Slice size mismatch (2048 vs 4096)
  at::Tensor wrong_slice = at::zeros({2, 1, 2048}, at::kChar);
  EXPECT_FALSE(manager.H2d({0}, {wrong_slice}, 0).ok());

  // rank_id out of range (rank_id=2 >= num_ranks=2)
  EXPECT_FALSE(manager.H2d({0}, {valid_obj}, /*rank_id=*/2).ok());
  EXPECT_FALSE(manager.H2d({0}, {valid_obj}, /*rank_id=*/-1).ok());

  // Reset test mapping state
  manager.ResetSharedMemoryMappedForTest();
  EXPECT_FALSE(manager.is_shared_memory_mapped());
}

TEST_F(KVCacheManagerTorchTest, H2dD2hWithObjectTensorsEndToEnd) {
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 0.0f);
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
  std::vector<std::vector<at::Tensor>> device_tensors = {{tensor}};

  KVCacheManager manager(device_tensors,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8);

  const int64_t page_nbytes = manager.slice_byte_size();

  // Model the production layout: the object tensors are carved out of the
  // registered shared memory pool, so their addresses genuinely fall inside
  // the mapped range.  Registering an address the tensors do not live at
  // would not exercise the DMA path at all.
  const int64_t obj_nbytes = 2 * 1 * page_nbytes;
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pool_size =
      ((2 * static_cast<size_t>(obj_nbytes) + page_size - 1) / page_size) *
      page_size;
  void* pool = nullptr;
  ASSERT_EQ(posix_memalign(&pool, page_size, pool_size), 0);
  std::memset(pool, 0, pool_size);
  auto* pool_bytes = static_cast<uint8_t*>(pool);

  const c10::TensorOptions obj_options = c10::TensorOptions().dtype(at::kChar);
  at::Tensor src_obj =
      at::from_blob(pool_bytes, {2, 1, page_nbytes}, obj_options);
  at::Tensor dst_obj =
      at::from_blob(pool_bytes + obj_nbytes, {2, 1, page_nbytes}, obj_options);

  // Fill src_obj rank 0 with pattern 0x42
  uint8_t* src_ptr = static_cast<uint8_t*>(src_obj.data_ptr());
  std::memset(src_ptr, 0x42, page_nbytes);

  // Fill src_obj rank 1 with pattern 0x77
  std::memset(src_ptr + page_nbytes, 0x77, page_nbytes);

  manager.SetSharedMemoryMappedForTest(reinterpret_cast<uintptr_t>(pool),
                                       pool_size);

  // H2D: copy src_obj rank 0 into block 3
  auto h2d_future_or = manager.H2d({3}, {src_obj}, /*rank_id=*/0);
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_future, h2d_future_or);
  EXPECT_TRUE(h2d_future.Await().ok());

  // D2H: copy block 3 back into dst_obj rank 0
  auto d2h_future_or = manager.D2h({3}, {dst_obj}, /*rank_id=*/0);
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_future, d2h_future_or);
  EXPECT_TRUE(d2h_future.Await().ok());

  // Verify dst_obj rank 0 matches pattern 0x42
  const uint8_t* dst_ptr = static_cast<const uint8_t*>(dst_obj.data_ptr());
  for (int64_t i = 0; i < page_nbytes; ++i) {
    ASSERT_EQ(dst_ptr[i], 0x42);
  }

  // H2D: copy src_obj rank 1 into block 5
  auto h2d_rank1_or = manager.H2d({5}, {src_obj}, /*rank_id=*/1);
  TF_ASSERT_OK_AND_ASSIGN(auto h2d_rank1, h2d_rank1_or);
  EXPECT_TRUE(h2d_rank1.Await().ok());

  // D2H: copy block 5 into dst_obj rank 1
  auto d2h_rank1_or = manager.D2h({5}, {dst_obj}, /*rank_id=*/1);
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_rank1, d2h_rank1_or);
  EXPECT_TRUE(d2h_rank1.Await().ok());

  // Verify dst_obj rank 1 matches pattern 0x77
  for (int64_t i = 0; i < page_nbytes; ++i) {
    ASSERT_EQ(dst_ptr[page_nbytes + i], 0x77);
  }

  // A tensor that lives outside the registered pool must be rejected instead
  // of being handed to the DMA engine.
  at::Tensor outside_obj = at::zeros({2, 1, page_nbytes}, at::kChar);
  EXPECT_FALSE(manager.H2d({1}, {outside_obj}, /*rank_id=*/0).ok());
  EXPECT_FALSE(manager.D2h({1}, {outside_obj}, /*rank_id=*/0).ok());

  manager.ResetSharedMemoryMappedForTest();
  free(pool);
}

// Submissions racing an unmap must never tear down the registration out from
// under an in-flight copy: each call either succeeds or is refused outright.
TEST_F(KVCacheManagerTorchTest, ConcurrentCopiesRacingUnmapAreSafe) {
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 0.0f);
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
  std::vector<std::vector<at::Tensor>> device_tensors = {{tensor}};

  KVCacheManager manager(device_tensors,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8);

  const int64_t page_nbytes = manager.slice_byte_size();
  const int64_t obj_nbytes = 1 * 1 * page_nbytes;
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  constexpr int kNumThreads = 4;
  const size_t pool_size = ((kNumThreads * static_cast<size_t>(obj_nbytes) +
                             page_size - 1) /
                            page_size) *
                           page_size;
  void* pool = nullptr;
  ASSERT_EQ(posix_memalign(&pool, page_size, pool_size), 0);
  std::memset(pool, 0, pool_size);
  auto* pool_bytes = static_cast<uint8_t*>(pool);

  const c10::TensorOptions obj_options = c10::TensorOptions().dtype(at::kChar);
  std::vector<at::Tensor> per_thread_obj;
  per_thread_obj.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    per_thread_obj.push_back(at::from_blob(pool_bytes + t * obj_nbytes,
                                           {1, 1, page_nbytes}, obj_options));
  }

  manager.SetSharedMemoryMappedForTest(reinterpret_cast<uintptr_t>(pool),
                                       pool_size);

  std::atomic<bool> stop = false;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&manager, &per_thread_obj, &stop, t]() {
      while (!stop.load(std::memory_order_relaxed)) {
        // Each thread owns a distinct block so the copies never collide.
        auto result = manager.H2d({t}, {per_thread_obj[t]}, /*rank_id=*/0);
        if (!result.ok()) {
          // The only legal refusal is "the pool is no longer mapped".
          EXPECT_EQ(result.status().code(),
                    absl::StatusCode::kFailedPrecondition)
              << result.status();
        }
      }
    });
  }

  // Racing unmap.  DmaUnmap is unimplemented on the CPU client so this is
  // expected to fail, but it still drives the drain path that must wait for
  // every in-flight submission to register.
  absl::SleepFor(absl::Milliseconds(20));
  manager.UnmapSharedMemory().IgnoreError();

  stop.store(true, std::memory_order_relaxed);
  for (std::thread& thread : threads) {
    thread.join();
  }

  manager.ResetSharedMemoryMappedForTest();
  free(pool);
}

// The pool is DMA mapped against exactly one PJRT client.  A layer whose
// buffer belongs to a second client would be handed host memory that client
// never registered, so such a manager must refuse object tensor transfers
// instead of issuing the copy.
TEST_F(KVCacheManagerTorchTest, LayersOnDifferentPjRtClientsAreRejected) {
  // A second, independent CPU client: its devices report a different
  // PjRtClient* than client_, which is what the check keys on.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> other_client,
                          xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  ASSERT_NE(other_client.get(), client_.get());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space_a,
      client_->addressable_devices()[0]->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space_b,
      other_client->addressable_devices()[0]->default_memory_space());

  std::vector<float> data_a(8 * 1024, 1.0f);
  std::vector<float> data_b(8 * 1024, 2.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_a,
      client_->BufferFromHostBuffer(
          data_a.data(), xla::F32, {8, 1024},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space_a,
          /*device_layout=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_b,
      other_client->BufferFromHostBuffer(
          data_b.data(), xla::F32, {8, 1024},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space_b,
          /*device_layout=*/nullptr));

  // Layer 0 on client_, layer 1 on other_client.
  std::vector<std::vector<xla::PjRtBuffer*>> device_buffers = {
      {buffer_a.get()}, {buffer_b.get()}};
  KVCacheManager manager(device_buffers,
                         /*local_port=*/std::nullopt,
                         /*host_blocks_to_allocate=*/8,
                         /*unsafe_skip_buffer_lock=*/true);
  ASSERT_EQ(manager.num_layers(), 2);

  const int64_t page_nbytes = manager.slice_byte_size();
  const int64_t obj_nbytes = 2 * 2 * page_nbytes;
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pool_size =
      ((static_cast<size_t>(obj_nbytes) + page_size - 1) / page_size) *
      page_size;
  void* pool = nullptr;
  ASSERT_EQ(posix_memalign(&pool, page_size, pool_size), 0);
  std::memset(pool, 0, pool_size);

  const c10::TensorOptions obj_options = c10::TensorOptions().dtype(at::kChar);
  at::Tensor obj = at::from_blob(static_cast<uint8_t*>(pool),
                                 {2, 2, page_nbytes}, obj_options);

  // Map the pool so the failure below is attributable to the split clients
  // and not to a missing registration.
  manager.SetSharedMemoryMappedForTest(reinterpret_cast<uintptr_t>(pool),
                                       pool_size);

  const absl::Status h2d = manager.H2d({1}, {obj}, /*rank_id=*/0).status();
  EXPECT_EQ(h2d.code(), absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(absl::StrContains(h2d.message(), "share one PJRT client"))
      << h2d.message();

  const absl::Status d2h = manager.D2h({1}, {obj}, /*rank_id=*/0).status();
  EXPECT_EQ(d2h.code(), absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(absl::StrContains(d2h.message(), "share one PJRT client"))
      << d2h.message();

  manager.ResetSharedMemoryMappedForTest();
  free(pool);
}

}  // namespace
}  // namespace torch
}  // namespace tpu_raiden

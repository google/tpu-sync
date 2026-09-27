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

#include "tpu_sync/frameworks/torch/torch_raw_transfer.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status_matchers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/raw_dma_pacer.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"
#include "torch/torch.h"

namespace raiden {
namespace {

using ::tpu_raiden::torch::RegisterMockTensor;

class TorchRawTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    device_ = client_->addressable_devices()[0];
  }

  std::unique_ptr<xla::PjRtClient> client_;
  xla::PjRtDevice* device_;
};

TEST_F(TorchRawTransferTest, RawHostBufferAllocation) {
  RawHostBuffer buffer(1024);
  EXPECT_EQ(buffer.SizeBytes(), 1024);
  EXPECT_FALSE(buffer.IsPjRtBacked());

  buffer.EnsureBoundToDevice(device_);
  EXPECT_TRUE(buffer.IsPjRtBacked() || buffer.DataPtr() != 0);
  EXPECT_NE(buffer.DataPtr(), 0);
}

TEST_F(TorchRawTransferTest, PreparedTransferD2HAndH2D) {
  // Create a real CPU PJRT buffer to mock TPU buffer
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  std::vector<float> initial_data(256, 42.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            initial_data.data(), xla::F32, {256},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  // Create a CPU PyTorch tensor to represent the TPU tensor
  at::Tensor tpu_tensor = ::torch::zeros({256}, ::torch::kFloat32);
  RegisterMockTensor(tpu_tensor, pjrt_buffer.get());

  // Create Host buffer
  auto host_buffer = std::make_shared<RawHostBuffer>(256 * sizeof(float));
  host_buffer->EnsureBoundToDevice(device_);

  // Prepare transfer
  auto transfer = std::make_shared<PreparedTorchRawTransfer>(
      tpu_tensor, host_buffer, /*unsafe_skip_buffer_lock=*/true);

  // Test D2H (Device to Host)
  transfer->D2H();

  // Verify host buffer has the data
  float* host_data = reinterpret_cast<float*>(host_buffer->MutableData());
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(host_data[i], 42.0f);
  }

  // Modify host buffer data
  for (int i = 0; i < 256; ++i) {
    host_data[i] = 24.0f;
  }

  // Test H2D (Host to Device)
  transfer->H2D();

  // To verify H2D, we would need to read back from pjrt_buffer.
  // We can do this by creating another tensor/buffer and doing D2H again,
  // or just use PJRT API to read from pjrt_buffer.
  // Let's use PJRT API to check.
  std::vector<float> readback_data(256);
  ASSERT_OK(
      pjrt_buffer->CopyRawToHost(readback_data.data(), 0, 256 * sizeof(float))
          .Await());

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(readback_data[i], 24.0f);
  }
}

TEST_F(TorchRawTransferTest, BatchTransferD2HAndH2D) {
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  std::vector<float> initial_data(256, 42.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            initial_data.data(), xla::F32, {256},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  at::Tensor tpu_tensor = ::torch::zeros({256}, ::torch::kFloat32);
  RegisterMockTensor(tpu_tensor, pjrt_buffer.get());

  at::Tensor host_tensor = ::torch::zeros({256}, ::torch::kFloat32);

  // D2H Batch Async
  auto future = TransferD2HBatchAsync({tpu_tensor}, {host_tensor}, {}, {}, {});
  ASSERT_OK(future.Await());

  float* host_data = reinterpret_cast<float*>(host_tensor.data_ptr());
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(host_data[i], 42.0f);
  }

  // Modify host data
  for (int i = 0; i < 256; ++i) {
    host_data[i] = 24.0f;
  }

  // H2D Batch Async
  auto future2 = TransferH2DBatchAsync({host_tensor}, {tpu_tensor}, {}, {}, {});
  ASSERT_OK(future2.Await());

  // Verify
  std::vector<float> readback_data(256);
  ASSERT_OK(
      pjrt_buffer->CopyRawToHost(readback_data.data(), 0, 256 * sizeof(float))
          .Await());

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(readback_data[i], 24.0f);
  }
}

// Shrinks the process-wide DMA window so a small CPU buffer is split into
// many pieces, and restores the previous limits afterwards.
class ScopedTinyDmaWindow {
 public:
  explicit ScopedTinyDmaWindow(RawDmaPacer& pacer)
      : pacer_(pacer), saved_(pacer.options()) {
    pacer_.SetOptions(RawDmaPacerOptions{kWindowBytes, kChunkBytes});
    pacer_.ResetPeak();
  }
  ~ScopedTinyDmaWindow() { pacer_.SetOptions(saved_); }

  static constexpr int64_t kChunkBytes = 4096;
  static constexpr int64_t kWindowBytes = 4 * kChunkBytes;

 private:
  RawDmaPacer& pacer_;
  RawDmaPacerOptions saved_;
};

TEST_F(TorchRawTransferTest, PacedBatchTransferIsByteExact) {
  TF_ASSERT_OK_AND_ASSIGN(RawDmaPacer * pacer, RawDmaPacer::Global());
  ScopedTinyDmaWindow window(*pacer);
  // 1 MiB + 3 KiB: many full pieces plus an unaligned tail.
  constexpr int64_t kCount = (1 << 18) + 768;
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  std::vector<float> zeros(kCount, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            zeros.data(), xla::F32, {kCount},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));
  at::Tensor tpu_tensor = ::torch::zeros({kCount}, ::torch::kFloat32);
  RegisterMockTensor(tpu_tensor, pjrt_buffer.get());

  at::Tensor source = ::torch::arange(kCount, ::torch::kFloat32);
  ABSL_ASSERT_OK(
      TransferH2DBatchAsync({source}, {tpu_tensor}, {}, {}, {}).Await());
  std::vector<float> readback(kCount);
  ABSL_ASSERT_OK(
      pjrt_buffer->CopyRawToHost(readback.data(), 0, kCount * sizeof(float))
          .Await());
  for (int64_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(readback[i], static_cast<float>(i)) << "at element " << i;
  }

  at::Tensor sink = ::torch::zeros({kCount}, ::torch::kFloat32);
  ABSL_ASSERT_OK(
      TransferD2HBatchAsync({tpu_tensor}, {sink}, {}, {}, {}).Await());
  EXPECT_TRUE(::torch::equal(sink, source));

  const RawDmaPacerStats stats = pacer->stats();
  EXPECT_LE(stats.peak_inflight_bytes, ScopedTinyDmaWindow::kWindowBytes);
  EXPECT_EQ(stats.inflight_bytes, 0);
}

TEST_F(TorchRawTransferTest, PacedPreparedTransferIsByteExact) {
  TF_ASSERT_OK_AND_ASSIGN(RawDmaPacer * pacer, RawDmaPacer::Global());
  ScopedTinyDmaWindow window(*pacer);
  constexpr int64_t kCount = 3 * 4096 + 5;  // Three full pieces plus a tail.
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * memory_space,
                          device_->default_memory_space());
  std::vector<float> initial(kCount);
  for (int64_t i = 0; i < kCount; ++i) {
    initial[i] = static_cast<float>(i) * 0.5f;
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            initial.data(), xla::F32, {kCount},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));
  at::Tensor tpu_tensor = ::torch::zeros({kCount}, ::torch::kFloat32);
  RegisterMockTensor(tpu_tensor, pjrt_buffer.get());
  auto host_buffer = std::make_shared<RawHostBuffer>(kCount * sizeof(float));
  host_buffer->EnsureBoundToDevice(device_);
  auto transfer = std::make_shared<PreparedTorchRawTransfer>(
      tpu_tensor, host_buffer, /*unsafe_skip_buffer_lock=*/true);

  transfer->D2H();
  float* host = reinterpret_cast<float*>(host_buffer->MutableData());
  for (int64_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(host[i], initial[i]) << "at element " << i;
    host[i] = -host[i];
  }
  transfer->H2D();
  std::vector<float> readback(kCount);
  ABSL_ASSERT_OK(
      pjrt_buffer->CopyRawToHost(readback.data(), 0, kCount * sizeof(float))
          .Await());
  for (int64_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(readback[i], -initial[i]) << "at element " << i;
  }
  EXPECT_LE(pacer->stats().peak_inflight_bytes,
            ScopedTinyDmaWindow::kWindowBytes);
}

}  // namespace
}  // namespace raiden

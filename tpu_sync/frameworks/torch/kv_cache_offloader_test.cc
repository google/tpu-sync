// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/frameworks/torch/kv_cache_offloader.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "absl/status/status.h"
#include "c10/util/intrusive_ptr.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils_mock.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace tpu_raiden::torch {
namespace offloader_internal {

class KVCacheOffloaderTestPeer {
 public:
  static std::unique_ptr<KVCacheOffloader> Create(
      std::shared_ptr<OffloaderPlatform> platform) {
    return std::unique_ptr<KVCacheOffloader>(
        new KVCacheOffloader(std::move(platform)));
  }

  static std::unique_ptr<KVCacheOffloader> CreateWithTensors(
      const std::vector<at::Tensor>& kv_cache_tensors, size_t page_nbytes,
      std::shared_ptr<OffloaderPlatform> platform) {
    return std::unique_ptr<KVCacheOffloader>(new KVCacheOffloader(
        kv_cache_tensors, page_nbytes, std::move(platform)));
  }
};

}  // namespace offloader_internal
namespace {

using ::tpu_raiden::torch::RegisterMockTensor;
using ::tpu_raiden::torch::offloader_internal::KVCacheOffloaderTestPeer;
using ::tpu_raiden::torch::offloader_internal::OffloaderPlatform;

constexpr size_t kPageSize = 4096;
constexpr size_t kPoolSize = 2 * kPageSize;
void* const kMappedAddress = reinterpret_cast<void*>(0x40000000);

struct MappingRecorder {
  std::vector<std::string> calls;
  absl::Status dma_map_status = absl::OkStatus();
  absl::Status dma_unmap_status = absl::OkStatus();
  void* observed_dma_map_address = nullptr;
  void* observed_dma_unmap_address = nullptr;
};

std::shared_ptr<OffloaderPlatform> RecordingPlatform(
    std::shared_ptr<MappingRecorder> recorder) {
  auto platform = std::make_shared<OffloaderPlatform>();
  platform->page_size = [recorder]() {
    recorder->calls.push_back("page_size");
    return absl::StatusOr<size_t>(kPageSize);
  };
  platform->dma_map = [recorder](void* address, size_t size_bytes) {
    recorder->calls.push_back("DmaMap");
    recorder->observed_dma_map_address = address;
    EXPECT_EQ(size_bytes, kPoolSize);
    return recorder->dma_map_status;
  };
  platform->dma_unmap = [recorder](void* address) {
    recorder->calls.push_back("DmaUnmap");
    recorder->observed_dma_unmap_address = address;
    return recorder->dma_unmap_status;
  };
  return platform;
}

TEST(KVCacheOffloaderMappingTest, RegistersAndUnregistersCallerMappingInOrder) {
  auto recorder = std::make_shared<MappingRecorder>();
  auto offloader =
      KVCacheOffloaderTestPeer::Create(RecordingPlatform(recorder));

  EXPECT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());
  EXPECT_TRUE(offloader->is_shared_memory_mapped());

  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
  EXPECT_FALSE(offloader->is_shared_memory_mapped());
  EXPECT_EQ(recorder->observed_dma_unmap_address,
            recorder->observed_dma_map_address);
  EXPECT_EQ(recorder->calls,
            (std::vector<std::string>{"page_size", "DmaMap", "DmaUnmap"}));
}

TEST(KVCacheOffloaderMappingTest,
     RejectsInvalidAddressAndSizeBeforeDmaMap) {
  auto recorder = std::make_shared<MappingRecorder>();
  auto offloader =
      KVCacheOffloaderTestPeer::Create(RecordingPlatform(recorder));

  EXPECT_EQ(offloader->MapSharedMemory(nullptr, kPageSize).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(offloader->MapSharedMemory(kMappedAddress, 0).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(offloader->MapSharedMemory(kMappedAddress, kPageSize + 1).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(offloader
                ->MapSharedMemory(
                    reinterpret_cast<void*>(
                        reinterpret_cast<uintptr_t>(kMappedAddress) + 1),
                    kPageSize)
                .code(),
            absl::StatusCode::kInvalidArgument);
  const uintptr_t last_page = std::numeric_limits<uintptr_t>::max() -
                              std::numeric_limits<uintptr_t>::max() % kPageSize;
  EXPECT_EQ(
      offloader->MapSharedMemory(reinterpret_cast<void*>(last_page), kPageSize)
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(recorder->calls,
            (std::vector<std::string>{"page_size", "page_size", "page_size"}));
}

TEST(KVCacheOffloaderMappingTest, RejectsSecondActiveMapping) {
  auto recorder = std::make_shared<MappingRecorder>();
  auto offloader =
      KVCacheOffloaderTestPeer::Create(RecordingPlatform(recorder));
  ASSERT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());

  EXPECT_EQ(offloader->MapSharedMemory(kMappedAddress, kPoolSize).code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST(KVCacheOffloaderMappingTest,
     DmaMapFailureReleasesRegistrationReservation) {
  auto recorder = std::make_shared<MappingRecorder>();
  recorder->dma_map_status = absl::InternalError("injected DmaMap failure");
  auto offloader =
      KVCacheOffloaderTestPeer::Create(RecordingPlatform(recorder));

  EXPECT_EQ(offloader->MapSharedMemory(kMappedAddress, kPoolSize).code(),
            absl::StatusCode::kInternal);
  EXPECT_FALSE(offloader->is_shared_memory_mapped());
  EXPECT_EQ(recorder->calls,
            (std::vector<std::string>{"page_size", "DmaMap"}));

  recorder->dma_map_status = absl::OkStatus();
  EXPECT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST(KVCacheOffloaderMappingTest, DmaUnmapFailureKeepsRegistrationForRetry) {
  auto recorder = std::make_shared<MappingRecorder>();
  auto offloader =
      KVCacheOffloaderTestPeer::Create(RecordingPlatform(recorder));
  ASSERT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());
  recorder->dma_unmap_status = absl::InternalError("injected DmaUnmap failure");

  EXPECT_EQ(offloader->UnmapSharedMemory().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(offloader->is_shared_memory_mapped());
  EXPECT_EQ(recorder->calls.back(), "DmaUnmap");

  recorder->dma_unmap_status = absl::OkStatus();
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
  EXPECT_FALSE(offloader->is_shared_memory_mapped());
}

TEST(KVCacheOffloaderMappingTest, DestructorCleansUpActiveMapping) {
  auto recorder = std::make_shared<MappingRecorder>();
  {
    auto offloader =
        KVCacheOffloaderTestPeer::Create(RecordingPlatform(recorder));
    ASSERT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());
  }

  EXPECT_EQ(recorder->calls.back(), "DmaUnmap");
  EXPECT_EQ(recorder->observed_dma_unmap_address,
            recorder->observed_dma_map_address);
}

TEST(KVCacheOffloaderMappingTest, ConcurrentMapReservesStateBeforeDmaMap) {
  auto platform = std::make_shared<OffloaderPlatform>();
  std::mutex gate_mutex;
  std::condition_variable gate_cv;
  bool first_in_dma_map = false;
  bool release_first = false;
  int dma_map_calls = 0;

  platform->page_size = [] { return absl::StatusOr<size_t>(kPageSize); };
  platform->dma_map = [&](void*, size_t) {
    std::unique_lock<std::mutex> lock(gate_mutex);
    ++dma_map_calls;
    first_in_dma_map = true;
    gate_cv.notify_all();
    gate_cv.wait(lock, [&] { return release_first; });
    return absl::OkStatus();
  };
  platform->dma_unmap = [](void*) { return absl::OkStatus(); };

  auto offloader = KVCacheOffloaderTestPeer::Create(platform);
  absl::Status first_status;
  std::thread first([&] {
    first_status = offloader->MapSharedMemory(kMappedAddress, kPoolSize);
  });
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate_cv.wait(lock, [&] { return first_in_dma_map; });
  }

  const absl::Status second_status =
      offloader->MapSharedMemory(kMappedAddress, kPoolSize);
  EXPECT_EQ(second_status.code(), absl::StatusCode::kFailedPrecondition);
  {
    std::lock_guard<std::mutex> lock(gate_mutex);
    release_first = true;
  }
  gate_cv.notify_all();
  first.join();

  EXPECT_TRUE(first_status.ok()) << first_status;
  EXPECT_EQ(dma_map_calls, 1);
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST(KVCacheOffloaderMappingTest, UnmapRejectsMappingInProgress) {
  auto platform = std::make_shared<OffloaderPlatform>();
  std::mutex gate_mutex;
  std::condition_variable gate_cv;
  bool map_in_dma_map = false;
  bool release_map = false;

  platform->page_size = [] { return absl::StatusOr<size_t>(kPageSize); };
  platform->dma_map = [&](void*, size_t) {
    std::unique_lock<std::mutex> lock(gate_mutex);
    map_in_dma_map = true;
    gate_cv.notify_all();
    gate_cv.wait(lock, [&] { return release_map; });
    return absl::OkStatus();
  };
  platform->dma_unmap = [](void*) { return absl::OkStatus(); };

  auto offloader = KVCacheOffloaderTestPeer::Create(platform);
  absl::Status map_status;
  std::thread map_thread([&] {
    map_status = offloader->MapSharedMemory(kMappedAddress, kPoolSize);
  });
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate_cv.wait(lock, [&] { return map_in_dma_map; });
  }

  EXPECT_EQ(offloader->UnmapSharedMemory().code(),
            absl::StatusCode::kFailedPrecondition);
  {
    std::lock_guard<std::mutex> lock(gate_mutex);
    release_map = true;
  }
  gate_cv.notify_all();
  map_thread.join();

  EXPECT_TRUE(map_status.ok()) << map_status;
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

class KVCacheOffloaderCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    TF_ASSERT_OK_AND_ASSIGN(
        memory_space_,
        client_->addressable_devices()[0]->default_memory_space());
  }

  at::Tensor AddLayer(int64_t kernel_blocks, int64_t elements_per_block,
                      float initial_value = 0.0f) {
    std::vector<float> initial(
        static_cast<size_t>(kernel_blocks * elements_per_block), initial_value);
    auto buffer_or = client_->BufferFromHostBuffer(
        initial.data(), xla::F32, {kernel_blocks, elements_per_block},
        /*byte_strides=*/std::nullopt,
        xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
        /*on_done_with_host_buffer=*/nullptr, memory_space_,
        /*device_layout=*/nullptr);
    EXPECT_TRUE(buffer_or.ok()) << buffer_or.status();
    if (!buffer_or.ok()) return at::Tensor();
    EXPECT_TRUE(buffer_or.value()->GetReadyFuture().Await().ok());

    at::Tensor tensor = at::zeros({kernel_blocks, elements_per_block});
    RegisterMockTensor(tensor, buffer_or.value().get());
    buffers_.push_back(std::move(buffer_or.value()));
    return tensor;
  }

  at::Tensor ObjectView(size_t prefix_bytes, int64_t num_ranks,
                        int64_t num_layers, int64_t page_nbytes) {
    const int64_t object_bytes = num_ranks * num_layers * page_nbytes;
    at::Tensor backing = at::zeros(
        {static_cast<int64_t>(prefix_bytes) + object_bytes + 7}, at::kChar);
    backings_.push_back(backing);
    return backing.narrow(0, static_cast<int64_t>(prefix_bytes), object_bytes)
        .view({num_ranks, num_layers, page_nbytes});
  }

  std::unique_ptr<KVCacheOffloader> CreateMappedOffloader(
      const std::vector<at::Tensor>& layers, size_t page_nbytes,
      std::shared_ptr<MappingRecorder> recorder =
          std::make_shared<MappingRecorder>()) {
    auto offloader = KVCacheOffloaderTestPeer::CreateWithTensors(
        layers, page_nbytes, RecordingPlatform(std::move(recorder)));
    EXPECT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());
    return offloader;
  }

  void ExpectLayerBytes(xla::PjRtBuffer* buffer, size_t offset,
                        const int8_t* expected, size_t size) {
    std::vector<int8_t> actual(size);
    EXPECT_TRUE(
        buffer->CopyRawToHost(actual.data(), offset, size).Await().ok());
    EXPECT_EQ(actual, std::vector<int8_t>(expected, expected + size));
  }

  std::unique_ptr<xla::PjRtClient> client_;
  xla::PjRtMemorySpace* memory_space_ = nullptr;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers_;
  std::vector<at::Tensor> backings_;
};

TEST_F(KVCacheOffloaderCopyTest,
       CopiesFullNonZeroOffsetObjectViewsInBothDirections) {
  constexpr size_t kPageBytes = 2 * 4 * sizeof(float);
  std::vector<at::Tensor> layers = {AddLayer(4, 4), AddLayer(4, 4)};
  auto offloader = CreateMappedOffloader(layers, kPageBytes);

  at::Tensor source0 = ObjectView(/*prefix_bytes=*/11, /*num_ranks=*/2,
                                  /*num_layers=*/2, kPageBytes);
  at::Tensor source1 = ObjectView(/*prefix_bytes=*/19, /*num_ranks=*/2,
                                  /*num_layers=*/2, kPageBytes);
  ASSERT_GT(source0.storage_offset(), 0);
  ASSERT_GT(source1.storage_offset(), 0);
  auto fill_rank = [](at::Tensor tensor, int64_t rank, int8_t seed) {
    auto* bytes = static_cast<int8_t*>(tensor.data_ptr());
    const size_t rank_offset =
        static_cast<size_t>(rank * tensor.size(1) * tensor.size(2));
    for (size_t i = 0; i < static_cast<size_t>(tensor.size(1) * tensor.size(2));
         ++i) {
      bytes[rank_offset + i] = static_cast<int8_t>(seed + i);
    }
  };
  fill_rank(source0, /*rank=*/1, /*seed=*/10);
  fill_rank(source1, /*rank=*/1, /*seed=*/70);

  auto h2d = offloader->H2d(/*block_ids=*/{1, 0}, {source0, source1},
                            /*rank_id=*/1);
  ASSERT_TRUE(h2d.ok()) << h2d.status();
  ASSERT_TRUE(h2d->Await().ok());

  at::Tensor destination0 = ObjectView(/*prefix_bytes=*/13, /*num_ranks=*/2,
                                       /*num_layers=*/2, kPageBytes);
  at::Tensor destination1 = ObjectView(/*prefix_bytes=*/23, /*num_ranks=*/2,
                                       /*num_layers=*/2, kPageBytes);
  auto d2h = offloader->D2h(/*block_ids=*/{1, 0}, {destination0, destination1},
                            /*rank_id=*/0);
  ASSERT_TRUE(d2h.ok()) << d2h.status();
  ASSERT_TRUE(d2h->Await().ok());

  const size_t rank_bytes = 2 * kPageBytes;
  EXPECT_EQ(std::memcmp(destination0.data_ptr(),
                        static_cast<int8_t*>(source0.data_ptr()) + rank_bytes,
                        rank_bytes),
            0);
  EXPECT_EQ(std::memcmp(destination1.data_ptr(),
                        static_cast<int8_t*>(source1.data_ptr()) + rank_bytes,
                        rank_bytes),
            0);
  const auto* untouched =
      static_cast<const int8_t*>(destination0.data_ptr()) + rank_bytes;
  EXPECT_TRUE(std::all_of(untouched, untouched + rank_bytes,
                          [](int8_t value) { return value == 0; }));

  EXPECT_EQ(offloader->num_layers(), 2);
  EXPECT_EQ(offloader->num_blocks(), 2);
  EXPECT_EQ(offloader->page_nbytes(), kPageBytes);
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST_F(KVCacheOffloaderCopyTest,
       TreatsSameSizedDifferentShapeLayersAsRawBytePages) {
  constexpr size_t kPageBytes = 32;
  std::vector<at::Tensor> layers = {AddLayer(4, 4), AddLayer(2, 8)};
  auto offloader = CreateMappedOffloader(layers, kPageBytes);

  at::Tensor source = ObjectView(/*prefix_bytes=*/5, /*num_ranks=*/1,
                                 /*num_layers=*/2, kPageBytes);
  auto* source_bytes = static_cast<int8_t*>(source.data_ptr());
  for (size_t byte = 0; byte < source.nbytes(); ++byte) {
    source_bytes[byte] = static_cast<int8_t>(byte + 1);
  }
  auto h2d = offloader->H2d(/*block_ids=*/{1}, {source}, /*rank_id=*/0);
  ASSERT_TRUE(h2d.ok()) << h2d.status();
  ASSERT_TRUE(h2d->Await().ok());

  at::Tensor destination = ObjectView(/*prefix_bytes=*/7, /*num_ranks=*/1,
                                      /*num_layers=*/2, kPageBytes);
  auto d2h = offloader->D2h(/*block_ids=*/{1}, {destination}, /*rank_id=*/0);
  ASSERT_TRUE(d2h.ok()) << d2h.status();
  ASSERT_TRUE(d2h->Await().ok());

  EXPECT_EQ(std::memcmp(destination.data_ptr(), source.data_ptr(),
                        source.nbytes()),
            0);
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST_F(KVCacheOffloaderCopyTest,
       ReleasesCompletedObjectViewsBeforeSubmittingTheNextCopy) {
  constexpr size_t kPageBytes = 32;
  std::vector<at::Tensor> layers = {AddLayer(4, 4)};
  auto offloader = CreateMappedOffloader(layers, kPageBytes);

  at::Tensor first = ObjectView(11, 1, 1, kPageBytes);
  c10::weak_intrusive_ptr<c10::TensorImpl> first_view(first.getIntrusivePtr());
  {
    auto copy = offloader->H2d({0}, {first}, /*rank_id=*/0);
    ASSERT_TRUE(copy.ok()) << copy.status();
    ASSERT_TRUE(copy->Await().ok());
  }
  first = at::Tensor();
  ASSERT_FALSE(first_view.expired());

  at::Tensor second = ObjectView(13, 1, 1, kPageBytes);
  auto copy = offloader->H2d({1}, {second}, /*rank_id=*/0);
  ASSERT_TRUE(copy.ok()) << copy.status();
  EXPECT_TRUE(first_view.expired());
  EXPECT_TRUE(copy->Await().ok());
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST_F(KVCacheOffloaderCopyTest, RejectsInvalidDeviceGeometry) {
  at::Tensor four_blocks = AddLayer(4, 4);
  EXPECT_THROW(KVCacheOffloaderTestPeer::CreateWithTensors(
                   {four_blocks}, /*page_nbytes=*/24,
                   RecordingPlatform(std::make_shared<MappingRecorder>())),
               std::invalid_argument);

  at::Tensor three_blocks = AddLayer(3, 4);
  EXPECT_THROW(KVCacheOffloaderTestPeer::CreateWithTensors(
                   {three_blocks}, /*page_nbytes=*/32,
                   RecordingPlatform(std::make_shared<MappingRecorder>())),
               std::invalid_argument);

  at::Tensor different_slice = AddLayer(4, 8);
  EXPECT_THROW(KVCacheOffloaderTestPeer::CreateWithTensors(
                   {four_blocks, different_slice}, /*page_nbytes=*/32,
                   RecordingPlatform(std::make_shared<MappingRecorder>())),
               std::invalid_argument);

  EXPECT_THROW(
      KVCacheOffloaderTestPeer::CreateWithTensors(
          {four_blocks},
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) + size_t{1},
          RecordingPlatform(std::make_shared<MappingRecorder>())),
      std::overflow_error);
}

TEST_F(KVCacheOffloaderCopyTest, ValidatesEveryArgumentBeforeIssuingCopies) {
  constexpr size_t kPageBytes = 32;
  std::vector<at::Tensor> layers = {AddLayer(4, 4), AddLayer(4, 4)};
  auto offloader = CreateMappedOffloader(layers, kPageBytes);
  at::Tensor valid = ObjectView(5, 2, 2, kPageBytes);
  std::memset(valid.data_ptr(), 0x5a, valid.nbytes());
  at::Tensor invalid =
      at::zeros({2, 2, static_cast<int64_t>(kPageBytes)}, at::kFloat);

  auto copy = offloader->H2d({0, 1}, {valid, invalid}, /*rank_id=*/0);
  ASSERT_FALSE(copy.ok());
  EXPECT_EQ(copy.status().code(), absl::StatusCode::kInvalidArgument);

  std::vector<int8_t> expected_zero(kPageBytes, 0);
  ExpectLayerBytes(buffers_[0].get(), /*offset=*/0, expected_zero.data(),
                   kPageBytes);
  ExpectLayerBytes(buffers_[1].get(), /*offset=*/0, expected_zero.data(),
                   kPageBytes);
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST_F(KVCacheOffloaderCopyTest, RejectsBlockRankAndTensorLayoutErrors) {
  constexpr size_t kPageBytes = 32;
  std::vector<at::Tensor> layers = {AddLayer(4, 4), AddLayer(4, 4)};
  auto offloader = CreateMappedOffloader(layers, kPageBytes);
  at::Tensor object0 = ObjectView(3, 2, 2, kPageBytes);
  at::Tensor object1 = ObjectView(7, 2, 2, kPageBytes);

  EXPECT_EQ(offloader->H2d({-1}, {object0}, 0).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(offloader->H2d({2}, {object0}, 0).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(offloader->H2d({0}, {object0}, 2).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(offloader->H2d({0, 0}, {object0, object1}, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(offloader->H2d({0}, {object0.transpose(1, 2)}, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
}

TEST_F(KVCacheOffloaderCopyTest, RequiresMappingAndUnmapDrainsDroppedFuture) {
  constexpr size_t kPageBytes = 32;
  constexpr float kInitialValue = 3.25f;
  std::vector<at::Tensor> layers = {AddLayer(4, 4, kInitialValue)};
  auto recorder = std::make_shared<MappingRecorder>();
  auto platform = RecordingPlatform(recorder);
  auto offloader =
      KVCacheOffloaderTestPeer::CreateWithTensors(layers, kPageBytes, platform);
  at::Tensor destination = ObjectView(9, 1, 1, kPageBytes);

  EXPECT_EQ(offloader->D2h({0}, {destination}, 0).status().code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(offloader->MapSharedMemory(kMappedAddress, kPoolSize).ok());

  std::vector<float> source_values(kPageBytes / sizeof(float), kInitialValue);
  const auto* source = reinterpret_cast<const int8_t*>(source_values.data());
  {
    auto dropped = offloader->D2h({0}, {destination}, 0);
    ASSERT_TRUE(dropped.ok()) << dropped.status();
  }

  bool observed_copy_before_dma_unmap = false;
  platform->dma_unmap = [&](void* address) {
    EXPECT_EQ(address, kMappedAddress);
    observed_copy_before_dma_unmap =
        std::memcmp(destination.data_ptr(), source, kPageBytes) == 0;
    recorder->calls.push_back("DmaUnmap");
    return absl::OkStatus();
  };
  EXPECT_TRUE(offloader->UnmapSharedMemory().ok());
  EXPECT_TRUE(observed_copy_before_dma_unmap);
}

}  // namespace
}  // namespace tpu_raiden::torch

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

#include "tpu_sync/frameworks/jax/mooncake_provider.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "third_party/mooncake/tpu_buffer_provider.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"

namespace raiden::mooncake {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::NotNull;

constexpr int kDeviceIndex = 3;

mc_tpu_buffer_provider_t GetTable() {
  mc_tpu_buffer_provider_t table{};
  EXPECT_EQ(
      mc_tpu_buffer_provider_get(MC_TPU_BUFFER_PROVIDER_ABI_VERSION, &table),
      0);
  return table;
}

// Acquires `buffer` as jax_compat.cc does for jax >= 0.11.1: a reference to a
// raw alias of the buffer, with no usage hold.
absl::StatusOr<RaidenBufferHandle> AcquireRawAlias(xla::PjRtBuffer* buffer) {
  ABSL_ASSIGN_OR_RETURN(xla::PjRtRawBufferRef raw,
                        xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buffer));
  return RaidenBufferHandle::AcquireFromRaw(raw.get(),
                                            buffer->on_device_shape());
}

// Exercises the provider through the C table, as Mooncake's adapter does, on
// buffers of the XLA CPU client: their raw aliases take the same
// RaidenBufferHandle path as TPU ones.
class MooncakeProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  // A device buffer holding bytes 0, 1, 2, ... (mod 256).
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> MakeBuffer(size_t size) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint8_t>(i);
    ABSL_ASSIGN_OR_RETURN(
        xla::PjRtMemorySpace * memory_space,
        client_->addressable_devices()[0]->default_memory_space());
    return client_->BufferFromHostBuffer(
        data.data(), xla::U8, {static_cast<int64_t>(size)},
        /*byte_strides=*/std::nullopt,
        xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
        /*on_done_with_host_buffer=*/nullptr, memory_space,
        /*device_layout=*/nullptr);
  }

  absl::StatusOr<mc_tpu_buffer_t> Export(xla::PjRtBuffer* buffer) {
    ABSL_ASSIGN_OR_RETURN(RaidenBufferHandle raw_alias,
                          AcquireRawAlias(buffer));
    std::vector<ExportRequest> requests;
    requests.push_back({std::move(raw_alias), kDeviceIndex});
    ABSL_ASSIGN_OR_RETURN(std::vector<mc_tpu_buffer_t> handles,
                          ExportBuffers(std::move(requests)));
    return handles.front();
  }

  std::unique_ptr<xla::PjRtClient> client_;
  const mc_tpu_buffer_provider_t table_ = GetTable();
};

TEST(MooncakeProviderAbiTest, ServesVersionOneWithCompleteTable) {
  mc_tpu_buffer_provider_t table{};
  ASSERT_EQ(mc_tpu_buffer_provider_get(1, &table), 0);
  EXPECT_EQ(table.abi_version, 1u);
  EXPECT_STREQ(table.name, "tpu-sync");
  EXPECT_THAT(table.size, NotNull());
  EXPECT_THAT(table.device_index, NotNull());
  EXPECT_THAT(table.numa_node, NotNull());
  EXPECT_THAT(table.copy_to_host, NotNull());
  EXPECT_THAT(table.copy_from_host, NotNull());
  EXPECT_THAT(table.release, NotNull());
}

TEST(MooncakeProviderAbiTest, RejectsOtherVersionsAndNullTable) {
  mc_tpu_buffer_provider_t table{};
  EXPECT_NE(mc_tpu_buffer_provider_get(0, &table), 0);
  EXPECT_NE(mc_tpu_buffer_provider_get(2, &table), 0);
  EXPECT_NE(mc_tpu_buffer_provider_get(1, nullptr), 0);
}

TEST_F(MooncakeProviderTest, InvalidHandleFailsEverywhere) {
  uint8_t host[4] = {};
  EXPECT_EQ(table_.size(0), 0u);
  EXPECT_EQ(table_.device_index(0), -1);
  EXPECT_EQ(table_.numa_node(0), -1);
  EXPECT_NE(table_.copy_to_host(0, 0, host, sizeof(host)), 0);
  EXPECT_NE(table_.copy_from_host(0, 0, host, sizeof(host)), 0);
  table_.release(0);  // A no-op, not a crash.
}

TEST_F(MooncakeProviderTest, ReportsBufferMetadata) {
  constexpr size_t kSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(kSize));
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));

  EXPECT_NE(handle, 0);
  EXPECT_EQ(table_.size(handle), buffer->GetOnDeviceSizeInBytes().value());
  EXPECT_EQ(table_.device_index(handle), kDeviceIndex);
  EXPECT_EQ(table_.numa_node(handle),
            tpu_raiden::GetPjRtDeviceNumaNode(buffer->device()));
  table_.release(handle);
}

TEST_F(MooncakeProviderTest, CopiesSubRangesBothWays) {
  constexpr size_t kSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(kSize));
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));

  std::vector<uint8_t> patch(100, 0xAB);
  ASSERT_EQ(table_.copy_from_host(handle, 200, patch.data(), patch.size()), 0);

  std::vector<uint8_t> expected(kSize);
  for (size_t i = 0; i < kSize; ++i) expected[i] = static_cast<uint8_t>(i);
  std::fill(expected.begin() + 200, expected.begin() + 300, 0xAB);
  std::vector<uint8_t> whole(kSize);
  ASSERT_EQ(table_.copy_to_host(handle, 0, whole.data(), kSize), 0);
  EXPECT_THAT(whole, ElementsAreArray(expected));

  std::vector<uint8_t> tail(24);
  ASSERT_EQ(table_.copy_to_host(handle, kSize - 24, tail.data(), tail.size()),
            0);
  EXPECT_THAT(tail, ElementsAreArray(expected.end() - 24, expected.end()));
  table_.release(handle);
}

TEST_F(MooncakeProviderTest, CopiesThroughMisalignedHostPointers) {
  constexpr size_t kSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(kSize));
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));

  // Host pointers one byte past an aligned allocation take the bounce path.
  alignas(64) uint8_t storage[kSize + 1];
  uint8_t* host = storage + 1;
  ASSERT_EQ(table_.copy_to_host(handle, 0, host, kSize), 0);
  for (size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(host[i], static_cast<uint8_t>(i)) << "at byte " << i;
  }

  std::fill(host, host + 64, 0xCD);
  ASSERT_EQ(table_.copy_from_host(handle, 64, host, 64), 0);
  std::vector<uint8_t> window(64);
  ASSERT_EQ(table_.copy_to_host(handle, 64, window.data(), 64), 0);
  EXPECT_THAT(window, ::testing::Each(0xCD));
  table_.release(handle);
}

TEST_F(MooncakeProviderTest, RejectsOutOfRangeCopies) {
  constexpr size_t kSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(kSize));
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));
  std::vector<uint8_t> host(kSize + 1);

  EXPECT_NE(table_.copy_to_host(handle, 0, host.data(), kSize + 1), 0);
  EXPECT_NE(table_.copy_to_host(handle, kSize, host.data(), 1), 0);
  EXPECT_NE(table_.copy_to_host(handle, kSize + 1, host.data(), 0), 0);
  EXPECT_NE(table_.copy_from_host(handle, 1000, host.data(), 25), 0);
  // offset + len would wrap around to a small number.
  EXPECT_NE(table_.copy_to_host(handle, SIZE_MAX, host.data(), 2), 0);
  EXPECT_NE(table_.copy_to_host(handle, 0, nullptr, 1), 0);
  // An empty range at the very end is in range.
  EXPECT_EQ(table_.copy_to_host(handle, kSize, host.data(), 0), 0);
  table_.release(handle);
}

TEST_F(MooncakeProviderTest, ReleaseFreesTheHandle) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(64));
  const int64_t before = LiveBufferCount();
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));
  EXPECT_EQ(LiveBufferCount(), before + 1);
  table_.release(handle);
  EXPECT_EQ(LiveBufferCount(), before);
}

TEST_F(MooncakeProviderTest, HandleKeepsDeviceMemoryAliveAfterBufferIsGone) {
  constexpr size_t kSize = 4096;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(kSize));
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));
  buffer->Delete();
  buffer.reset();

  std::vector<uint8_t> host(kSize);
  ASSERT_EQ(table_.copy_to_host(handle, 0, host.data(), kSize), 0);
  for (size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(host[i], static_cast<uint8_t>(i)) << "at byte " << i;
  }
  table_.release(handle);
}

TEST_F(MooncakeProviderTest, ExportIsAllOrNothing) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(64));
  TF_ASSERT_OK_AND_ASSIGN(RaidenBufferHandle valid,
                          AcquireRawAlias(buffer.get()));
  const int64_t before = LiveBufferCount();

  std::vector<ExportRequest> requests;
  requests.push_back({valid, kDeviceIndex});
  requests.push_back({RaidenBufferHandle(), kDeviceIndex});
  EXPECT_THAT(ExportBuffers(std::move(requests)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(LiveBufferCount(), before);

  requests.clear();
  requests.push_back({valid, kDeviceIndex});
  requests.push_back({valid, -1});
  EXPECT_THAT(ExportBuffers(std::move(requests)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(LiveBufferCount(), before);
}

// TENT stages several chunks of one buffer at once. Each thread owns a
// disjoint slice, writes a thread-specific pattern into it and reads it back.
TEST_F(MooncakeProviderTest, ConcurrentCopiesOnOneHandle) {
  constexpr int kThreads = 8;
  constexpr int kIterations = 50;
  constexpr size_t kSlice = 4096;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtBuffer> buffer,
                          MakeBuffer(kThreads * kSlice));
  TF_ASSERT_OK_AND_ASSIGN(mc_tpu_buffer_t handle, Export(buffer.get()));

  std::vector<int> failures(kThreads, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const size_t offset = t * kSlice;
      std::vector<uint8_t> out(kSlice);
      std::vector<uint8_t> in(kSlice);
      for (int i = 0; i < kIterations; ++i) {
        std::iota(out.begin(), out.end(), static_cast<uint8_t>(t * 31 + i));
        if (table_.copy_from_host(handle, offset, out.data(), kSlice) != 0 ||
            table_.copy_to_host(handle, offset, in.data(), kSlice) != 0 ||
            in != out) {
          ++failures[t];
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  EXPECT_THAT(failures, ::testing::Each(0));
  table_.release(handle);
}

}  // namespace
}  // namespace raiden::mooncake

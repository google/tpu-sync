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

#include "tpu_sync/core/buffer_utils.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"

namespace tpu_raiden {
namespace {

TEST(BufferUtilsTest, UnpackLayersEmptyReturnsEmpty) {
  std::vector<std::vector<xla::PjRtBuffer*>> empty;
  auto handles = UnpackLayers(empty);
  EXPECT_TRUE(handles.empty());
}

TEST(BufferUtilsTest, UnpackLayersSucceedsWithCpuBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client->BufferFromHostBuffer(
                            data.data(), xla::F32, {8, 1024},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  std::vector<std::vector<xla::PjRtBuffer*>> device_buffers = {
      {pjrt_buffer.get()}};

  auto handles = UnpackLayers(device_buffers, /*unsafe_skip_buffer_lock=*/true);
  ASSERT_EQ(handles.size(), 1);
  ASSERT_EQ(handles[0].size(), 1);
  EXPECT_NE(handles[0][0].device, nullptr);
}

}  // namespace
}  // namespace tpu_raiden

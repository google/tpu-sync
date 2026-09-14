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

#include "tpu_sync/core/kv_manager_holder.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden {
namespace {

using ::testing::ElementsAre;
using ::tpu_raiden::controller::MockTransferManager;
using ::tpu_raiden::controller::ShardAwareMockTransferManager;

std::vector<RaidenTransferEndpoint> TwoShardEndpoints() {
  return {{"10.0.0.9:41000", {0, 1, 2, 3}}, {"10.0.0.9:41001", {4, 5, 6, 7}}};
}

// An impl exposing the vector H2dRead overload receives the descriptors and
// all four offset vectors verbatim; the string fallback is not used.
TEST(KVManagerHolderTest, VectorH2dReadDispatchesToVectorImpl) {
  ShardAwareMockTransferManager mock;
  KVManagerHolder holder(&mock);

  auto result = holder.H2dRead(TwoShardEndpoints(), /*src_host_offsets=*/{10},
                               /*dst_host_offsets=*/{20},
                               /*dst_device_offsets=*/{30}, /*copy_sizes=*/{64});
  ASSERT_TRUE(result.ok()) << result.status().message();

  EXPECT_EQ(mock.vector_h2d_read_calls, 1);
  EXPECT_EQ(mock.h2d_read_calls, 0);  // string fallback NOT taken
  ASSERT_EQ(mock.last_h2d_read_descriptors.size(), 2u);
  EXPECT_EQ(mock.last_h2d_read_descriptors[0].endpoint, "10.0.0.9:41000");
  EXPECT_EQ(mock.last_h2d_read_descriptors[0].shards,
            (std::vector<int64_t>{0, 1, 2, 3}));
  EXPECT_EQ(mock.last_h2d_read_descriptors[1].endpoint, "10.0.0.9:41001");
  EXPECT_EQ(mock.last_h2d_read_descriptors[1].shards,
            (std::vector<int64_t>{4, 5, 6, 7}));
  EXPECT_THAT(mock.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock.last_staging_offsets, ElementsAre(20));
  EXPECT_THAT(mock.last_dst_offsets, ElementsAre(30));
  EXPECT_THAT(mock.last_copy_sizes, ElementsAre(64));
}

// An impl without the vector overload falls back to the single-peer string
// H2dRead with descriptors[0].endpoint, mirroring the vector H2hRead/H2hWrite
// fallback behaviour.
TEST(KVManagerHolderTest, VectorH2dReadFallsBackToFirstEndpoint) {
  MockTransferManager mock;  // string overloads only
  KVManagerHolder holder(&mock);

  auto result = holder.H2dRead(TwoShardEndpoints(), /*src_host_offsets=*/{10},
                               /*dst_host_offsets=*/{20},
                               /*dst_device_offsets=*/{30}, /*copy_sizes=*/{64});
  ASSERT_TRUE(result.ok()) << result.status().message();

  EXPECT_EQ(mock.h2d_read_calls, 1);
  EXPECT_EQ(mock.last_peer, "10.0.0.9:41000");
  EXPECT_THAT(mock.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock.last_staging_offsets, ElementsAre(20));
  EXPECT_THAT(mock.last_dst_offsets, ElementsAre(30));
  EXPECT_THAT(mock.last_copy_sizes, ElementsAre(64));
}

TEST(KVManagerHolderTest, VectorH2dReadEmptyDescriptorsFallBackToEmptyPeer) {
  MockTransferManager mock;
  KVManagerHolder holder(&mock);

  auto result =
      holder.H2dRead(std::vector<RaidenTransferEndpoint>{},
                     /*src_host_offsets=*/{10}, /*dst_host_offsets=*/{20},
                     /*dst_device_offsets=*/{30}, /*copy_sizes=*/{64});
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(mock.h2d_read_calls, 1);
  EXPECT_EQ(mock.last_peer, "");
}

TEST(KVManagerHolderTest, VectorH2dReadOnNullHolderFails) {
  KVManagerHolder holder;
  auto result = holder.H2dRead(TwoShardEndpoints(), {10}, {20}, {30}, {64});
  EXPECT_FALSE(result.ok());
}

struct MockManagerWithSlices : public MockTransferManager {
  std::vector<kv_cache::backends::BackendBufferDescriptor> ResolveBlockSlices(
      int staging_block_id) const {
    kv_cache::backends::BackendBufferDescriptor desc;
    desc.ptr = reinterpret_cast<void*>(0x1234);
    desc.size = 1024;
    desc.fd = 42;
    desc.offset = staging_block_id * 1024;
    return {desc};
  }
};

TEST(KVManagerHolderTest, ResolveBlockSlicesDelegatesCorrectly) {
  MockManagerWithSlices mock;
  KVManagerHolder holder(&mock);

  auto slices = holder.ResolveBlockSlices(5);
  ASSERT_EQ(slices.size(), 1u);
  EXPECT_EQ(slices[0].ptr, reinterpret_cast<void*>(0x1234));
  EXPECT_EQ(slices[0].size, 1024u);
  EXPECT_EQ(slices[0].fd, 42);
  EXPECT_EQ(slices[0].offset, 5120);

  // An impl without ResolveBlockSlices returns empty slices.
  MockTransferManager mock_without;
  KVManagerHolder holder_without(&mock_without);
  EXPECT_TRUE(holder_without.ResolveBlockSlices(5).empty());

  // Null holder returns empty slices.
  KVManagerHolder null_holder;
  EXPECT_TRUE(null_holder.ResolveBlockSlices(5).empty());
}

}  // namespace
}  // namespace tpu_raiden

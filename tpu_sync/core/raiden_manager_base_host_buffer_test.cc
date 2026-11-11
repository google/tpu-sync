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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr size_t kNumLayers = 2;
constexpr size_t kNumShards = 2;
constexpr size_t kSliceByteSize = 1024;

std::vector<transport::ShardInfo> MakeShards(size_t num_shards) {
  std::vector<transport::ShardInfo> shards;
  for (size_t sh = 0; sh < num_shards; ++sh) {
    shards.push_back({.local_index = static_cast<int>(sh),
                      .global_index = static_cast<int64_t>(sh)});
  }
  return shards;
}

// Populates the protected `layers_` with `num_shards` empty shards per layer.
class HostBufferTestManager : public RaidenManagerBase {
 public:
  HostBufferTestManager(size_t num_layers, size_t num_shards)
      : RaidenManagerBase(num_layers, MakeShards(num_shards), kSliceByteSize) {
    std::vector<LayerInfoBase> layers(num_layers);
    for (LayerInfoBase& layer : layers) {
      layer.shards.resize(num_shards_);
    }
    layers_ = StagingArena(std::move(layers));
  }

  void ClearLayerShards(size_t layer_idx) { layers_[layer_idx].shards.clear(); }
};

TEST(RaidenManagerBaseHostBufferTest, MutableAndConstReturnExternalBuffers) {
  HostBufferTestManager manager(kNumLayers, kNumShards);
  // storage[layer][shard], with a distinct size per buffer.
  std::vector<std::vector<std::vector<uint8_t>>> storage = {
      {std::vector<uint8_t>(64), std::vector<uint8_t>(128)},
      {std::vector<uint8_t>(192), std::vector<uint8_t>(256)}};
  std::vector<const uint8_t*> host_ptrs;
  std::vector<size_t> host_sizes;
  for (const std::vector<std::vector<uint8_t>>& layer : storage) {
    for (const std::vector<uint8_t>& shard : layer) {
      host_ptrs.push_back(shard.data());
      host_sizes.push_back(shard.size());
    }
  }
  manager.SetExternalHostPointers(host_ptrs, host_sizes);
  const RaidenManagerBase& const_manager = manager;
  transport::BlockTransportDelegate& delegate = manager;
  const transport::BlockTransportDelegate& const_delegate = manager;

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t s = 0; s < kNumShards; ++s) {
      const std::vector<uint8_t>& expected = storage[l][s];
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> mutable_buffer,
                           manager.GetHostBuffer(l, s));
      EXPECT_EQ(mutable_buffer.data(), expected.data());
      EXPECT_EQ(mutable_buffer.size(), expected.size());
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_buffer,
                           const_manager.GetHostBuffer(l, s));
      EXPECT_EQ(const_buffer.data(), expected.data());
      EXPECT_EQ(const_buffer.size(), expected.size());
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> delegate_buffer,
                           delegate.GetHostBuffer(l, s));
      EXPECT_EQ(delegate_buffer.data(), mutable_buffer.data());
      EXPECT_EQ(delegate_buffer.size(), mutable_buffer.size());
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_delegate_buffer,
                           const_delegate.GetHostBuffer(l, s));
      EXPECT_EQ(const_delegate_buffer.data(), expected.data());
      EXPECT_EQ(const_delegate_buffer.size(), expected.size());
    }
  }
}

TEST(RaidenManagerBaseHostBufferTest, ShardIndexWrapsModuloShardCount) {
  HostBufferTestManager manager(kNumLayers, kNumShards);
  std::vector<uint8_t> shard0(64);
  std::vector<uint8_t> shard1(128);
  manager.SetExternalHostPointers({shard0.data(), shard1.data()},
                                  {shard0.size(), shard1.size()});

  // Global shard 3 on a 2-shard manager resolves to local shard 1.
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> buffer, manager.GetHostBuffer(0, 3));

  EXPECT_EQ(buffer.data(), shard1.data());
  EXPECT_EQ(buffer.size(), shard1.size());
}

TEST(RaidenManagerBaseHostBufferTest, LayerPastHostLayersIsOutOfRange) {
  HostBufferTestManager manager(kNumLayers, kNumShards);
  std::vector<uint8_t> storage(64);
  manager.SetExternalHostPointers(
      {storage.data(), storage.data(), storage.data(), storage.data()},
      {storage.size(), storage.size(), storage.size(), storage.size()});
  const RaidenManagerBase& const_manager = manager;

  ABSL_EXPECT_OK(manager.GetHostBuffer(1, 0));
  EXPECT_THAT(manager.GetHostBuffer(2, 0),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("layer_idx 2 out of range: 2 host layers")));
  EXPECT_THAT(const_manager.GetHostBuffer(2, 0),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("layer_idx 2 out of range: 2 host layers")));
}

TEST(RaidenManagerBaseHostBufferTest, LayerWithNoShardsIsOutOfRange) {
  HostBufferTestManager manager(kNumLayers, kNumShards);
  std::vector<uint8_t> storage(64);
  manager.SetExternalHostPointers({storage.data(), storage.data()},
                                  {storage.size(), storage.size()});
  manager.ClearLayerShards(1);

  ABSL_EXPECT_OK(manager.GetHostBuffer(0, 1));
  EXPECT_THAT(manager.GetHostBuffer(1, 1),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("shard_idx 1 out of range: layer 1 has no "
                                 "shards")));
}

TEST(RaidenManagerBaseHostBufferTest, NullHostPointerIsFailedPrecondition) {
  HostBufferTestManager manager(kNumLayers, kNumShards);
  std::vector<uint8_t> storage(64);
  // (0, 0) has a size but no pointer; (0, 1) is a real buffer.
  manager.SetExternalHostPointers({nullptr, storage.data()},
                                  {storage.size(), storage.size()});
  const RaidenManagerBase& const_manager = manager;

  ABSL_EXPECT_OK(manager.GetHostBuffer(0, 1));
  EXPECT_THAT(manager.GetHostBuffer(0, 0),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("layer 0 shard 0 has no host buffer")));
  EXPECT_THAT(const_manager.GetHostBuffer(0, 0),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("layer 0 shard 0 has no host buffer")));
}

TEST(RaidenManagerBaseHostBufferTest, NonNullPointerWithZeroSizeIsOkEmpty) {
  HostBufferTestManager manager(kNumLayers, kNumShards);
  std::vector<uint8_t> storage(64);
  manager.SetExternalHostPointers({storage.data()}, {0});

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> buffer, manager.GetHostBuffer(0, 0));

  EXPECT_EQ(buffer.data(), storage.data());
  EXPECT_TRUE(buffer.empty());
}

}  // namespace
}  // namespace tpu_raiden

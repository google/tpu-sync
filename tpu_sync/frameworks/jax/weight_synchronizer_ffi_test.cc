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

#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

// Keep ffi buffers alive during test scope
struct FfiBufferFixture {
  XLA_FFI_Buffer ffi_buf;
  std::vector<int64_t> dims;

  FfiBufferFixture(XLA_FFI_DataType dtype, void* data,
                   std::vector<int64_t> dimensions)
      : dims(dimensions) {
    ffi_buf.struct_size = sizeof(XLA_FFI_Buffer);
    ffi_buf.extension_start = nullptr;
    ffi_buf.dtype = dtype;
    ffi_buf.data = data;
    ffi_buf.rank = dims.size();
    ffi_buf.dims = dims.data();
  }

  xla::ffi::AnyBuffer AsAnyBuffer() { return xla::ffi::AnyBuffer(&ffi_buf); }
};

// Returns the standard-FFI instance registered for |shard_idx|.
WeightSynchronizerBase* StandardWs(int32_t shard_idx) {
  return WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
      shard_idx);
}

class WeightSynchronizerFfiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure Host platform is registered and active
    auto platform = stream_executor::PlatformManager::PlatformWithName("Host");
    ASSERT_TRUE(platform.ok())
        << "Host Platform must be available on CPU sandboxes";
  }

  void TearDown() override {
    // Clean up global registry between tests
    WeightSynchronizerFfiManager::Default().Clear(/*delete_instances=*/true);
    ClearSharedWsMap();
  }
};

enum class FfiType { kInit, kInitAndD2h };

class WeightSynchronizerFfiParamTest : public WeightSynchronizerFfiTest,
                                       public ::testing::WithParamInterface<FfiType> {
 protected:
  xla::ffi::Error CallInit(xla::ffi::AnyBuffer x,
                           xla::ffi::AnyBuffer shard_idx_buf,
                           xla::ffi::AnyBuffer slice_byte_sizes_buf,
                           int32_t local_port, int32_t parallelism,
                           int32_t num_layers,
                           xla::ffi::Result<xla::ffi::AnyBuffer> out,
                           int32_t listener_port = 0, int32_t num_shards = 1) {
    if (GetParam() == FfiType::kInit) {
      return TriggerWeightSynchronizerInitImpl(
          x, shard_idx_buf, slice_byte_sizes_buf, local_port, parallelism,
          num_layers, listener_port, num_shards, out);
    } else {
      std::vector<xla::ffi::AnyBuffer> jax_arrays = {x};
      return TriggerWeightSynchronizerInitAndD2hHelper(
          shard_idx_buf, slice_byte_sizes_buf, jax_arrays, local_port,
          parallelism, num_layers, listener_port, num_shards, out);
    }
  }
};

TEST_P(WeightSynchronizerFfiParamTest, TriggerWSInitSucceeds) {
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});

  std::vector<int32_t> anchor(256);
  for (int i = 0; i < 256; ++i) anchor[i] = i + 1;
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});

  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();
  xla::ffi::AnyBuffer shard_idx_buf = shard_idx_fixture.AsAnyBuffer();

  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  int32_t local_port = 0;  // Allocate dynamic free port
  int32_t parallelism = 1;
  int32_t num_layers = 1;

  // Output buffer for IP (16 bytes) and port (4 bytes) -> 20 bytes -> 5 int32
  std::vector<int32_t> out_data(5, 0);
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {5});
  xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

  xla::ffi::Error err = CallInit(x, shard_idx_buf, slice_byte_sizes_buf,
                                 local_port, parallelism, num_layers, out);

  EXPECT_TRUE(err.success()) << "WS Init failed: " << err.message();
  EXPECT_NE(StandardWs(0), nullptr);
  EXPECT_NE(g_streams[0], nullptr);

  // Verify output contains some IP (can be all zeros if no external IP, but
  // port should be > 0)
  int32_t port = out_data[4];
  EXPECT_GT(port, 0) << "Assigned port should be positive";
  VLOG(1) << "Assigned port: " << port;
}

TEST_F(WeightSynchronizerFfiTest, TriggerMultiH2DSucceeds) {
  // 1. Initialize WeightSynchronizer
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});
  xla::ffi::AnyBuffer shard_idx_buf = shard_idx_fixture.AsAnyBuffer();

  // We will have 2 layers, each of size 1024 bytes (256 int32)
  std::vector<int32_t> slice_byte_sizes = {1024, 1024};
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, slice_byte_sizes.data(),
                                 {2});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  // Anchor is not really used in Init (Host mode) except to pass to
  // TriggerWSInit We can just pass dummy anchor
  std::vector<int32_t> dummy_anchor(1);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, dummy_anchor.data(),
                                  {1});
  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();

  // Output buffer for Init
  std::vector<int32_t> out_data(6, 0);  // Use 6 to get listener_port too
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
  xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

  xla::ffi::Error err = TriggerWeightSynchronizerInitImpl(
      x, shard_idx_buf, slice_byte_sizes_buf,
      /*local_port=*/0, /*parallelism=*/1, /*num_layers=*/2,
      /*listener_port=*/0, /*num_shards=*/1, out);
  ASSERT_TRUE(err.success()) << "Init failed: " << err.message();

  WeightSynchronizerBase* ws = StandardWs(shard_idx);
  ASSERT_NE(ws, nullptr);

  // 2. Populate Host Staging Buffers with dummy data
  // Layer 0: [0, 1, 2, ... 255]
  // Layer 1: [256, 257, ... 511]
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf_0,
                       ws->GetHostBufferWithLocalIndex(0, 0).Await());
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf_1,
                       ws->GetHostBufferWithLocalIndex(1, 0).Await());
  ASSERT_GE(host_buf_0.size(), 1024);
  ASSERT_GE(host_buf_1.size(), 1024);
  uint8_t* host_ptr_0 = host_buf_0.data();
  uint8_t* host_ptr_1 = host_buf_1.data();

  std::vector<int32_t> src_data_0(256);
  std::vector<int32_t> src_data_1(256);
  for (int i = 0; i < 256; ++i) {
    src_data_0[i] = i;
    src_data_1[i] = i + 256;
  }
  std::memcpy(host_ptr_0, src_data_0.data(), 1024);
  std::memcpy(host_ptr_1, src_data_1.data(), 1024);

  // 3. Prepare Output Buffers for H2D
  std::vector<int32_t> dst_data_0(256, 0);
  std::vector<int32_t> dst_data_1(256, 0);
  FfiBufferFixture dst_fixture_0(XLA_FFI_DataType_S32, dst_data_0.data(),
                                 {256});
  FfiBufferFixture dst_fixture_1(XLA_FFI_DataType_S32, dst_data_1.data(),
                                 {256});

  XLA_FFI_Buffer ffi_buf_0 = dst_fixture_0.ffi_buf;
  XLA_FFI_Buffer ffi_buf_1 = dst_fixture_1.ffi_buf;

  XLA_FFI_RetType types[] = {XLA_FFI_RetType_BUFFER, XLA_FFI_RetType_BUFFER};
  void* rets_ptrs[] = {&ffi_buf_0, &ffi_buf_1};

  XLA_FFI_Rets rets;
  rets.struct_size = sizeof(XLA_FFI_Rets);
  rets.extension_start = nullptr;
  rets.size = 2;
  rets.types = types;
  rets.rets = rets_ptrs;

  xla::ffi::RemainingRets remaining_rets(&rets, 0);

  // 4. Call TriggerMultiH2DImpl
  err = TriggerMultiH2DImpl(shard_idx_buf, remaining_rets);
  EXPECT_TRUE(err.success()) << "MultiH2D failed: " << err.message();

  // 5. Verify data was copied
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(dst_data_0[i], src_data_0[i])
        << "Mismatch at layer 0 index " << i;
    EXPECT_EQ(dst_data_1[i], src_data_1[i])
        << "Mismatch at layer 1 index " << i;
  }
}

TEST_F(WeightSynchronizerFfiTest, MultiShardGetLocalSlotAndGlobalShardIndices) {
  // Test scenario simulating Host 1 with global shards 8 and 9 sharing
  // listener_port=10019
  int32_t shard_idx_8 = 8;
  int32_t shard_idx_9 = 9;
  FfiBufferFixture shard_8_fixture(XLA_FFI_DataType_S32, &shard_idx_8, {1});
  FfiBufferFixture shard_9_fixture(XLA_FFI_DataType_S32, &shard_idx_9, {1});
  xla::ffi::AnyBuffer shard_8_buf = shard_8_fixture.AsAnyBuffer();
  xla::ffi::AnyBuffer shard_9_buf = shard_9_fixture.AsAnyBuffer();

  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  std::vector<int32_t> dummy_anchor(1);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, dummy_anchor.data(),
                                  {1});
  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();

  std::vector<int32_t> out_data_8(6, 0);
  std::vector<int32_t> out_data_9(6, 0);
  FfiBufferFixture out_fixture_8(XLA_FFI_DataType_S32, out_data_8.data(), {6});
  FfiBufferFixture out_fixture_9(XLA_FFI_DataType_S32, out_data_9.data(), {6});
  xla::ffi::Result<xla::ffi::AnyBuffer> out_8 = out_fixture_8.AsAnyBuffer();
  xla::ffi::Result<xla::ffi::AnyBuffer> out_9 = out_fixture_9.AsAnyBuffer();

  int32_t listener_port = 10019;
  int32_t num_shards = 2;

  // 1. Initialize Shard 8
  xla::ffi::Error err8 = TriggerWeightSynchronizerInitImpl(
      x, shard_8_buf, slice_byte_sizes_buf,
      /*local_port=*/0, /*parallelism=*/1, /*num_layers=*/1, listener_port,
      num_shards, out_8);
  ASSERT_TRUE(err8.success()) << err8.message();

  // 2. Initialize Shard 9
  xla::ffi::Error err9 = TriggerWeightSynchronizerInitImpl(
      x, shard_9_buf, slice_byte_sizes_buf,
      /*local_port=*/0, /*parallelism=*/1, /*num_layers=*/1, listener_port,
      num_shards, out_9);
  ASSERT_TRUE(err9.success()) << err9.message();

  // Both shards should share the exact same underlying WeightSynchronizerBase
  // instance
  WeightSynchronizerBase* ws8 = StandardWs(8);
  WeightSynchronizerBase* ws9 = StandardWs(9);
  ASSERT_NE(ws8, nullptr);
  ASSERT_EQ(ws8, ws9);

  // Global shard indices should be 8 and 9
  EXPECT_EQ(ws8->global_shard_index(0), 8);
  EXPECT_EQ(ws8->global_shard_index(1), 9);

  // Staging buffer pointers for shard 8 (slot 0) and shard 9 (slot 1) must be
  // distinct
  TF_ASSERT_OK_AND_ASSIGN(
      absl::Span<const uint8_t> buf8,
      std::as_const(*ws8).GetHostBufferWithLocalIndex(0, 0).Await());
  TF_ASSERT_OK_AND_ASSIGN(
      absl::Span<const uint8_t> buf9,
      std::as_const(*ws9).GetHostBufferWithLocalIndex(0, 1).Await());
  EXPECT_NE(buf8.data(), buf9.data());
}

TEST_F(WeightSynchronizerFfiTest, MultiNumaFourShardsInitAndD2hTest) {
  // Simulating Worker 1 in a multi-NUMA setup owning global shards 4, 5, 6, 7
  // with num_shards = 4 sharing listener_port = 10020
  const int32_t kNumShards = 4;
  const int32_t kBaseShard = 4;
  const int32_t kNumLayers = 2;
  const int32_t kSliceElements = 256;
  const int32_t kSliceByteSize = kSliceElements * sizeof(int32_t);
  const int32_t kListenerPort = 10020;

  std::vector<int32_t> slice_byte_sizes = {kSliceByteSize, kSliceByteSize};
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, slice_byte_sizes.data(),
                                 {kNumLayers});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  // Create unique source data for each shard and layer
  std::vector<std::vector<std::vector<int32_t>>> shard_layer_data(
      kNumShards, std::vector<std::vector<int32_t>>(
                      kNumLayers, std::vector<int32_t>(kSliceElements)));

  for (int s = 0; s < kNumShards; ++s) {
    for (int l = 0; l < kNumLayers; ++l) {
      for (int i = 0; i < kSliceElements; ++i) {
        shard_layer_data[s][l][i] = (s + kBaseShard) * 1000 + l * 100 + i;
      }
    }
  }

  // Execute InitAndD2h for each of the 4 shards
  for (int s = 0; s < kNumShards; ++s) {
    int32_t global_shard = kBaseShard + s;
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, &global_shard, {1});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<FfiBufferFixture> anchor_fixtures;
    std::vector<xla::ffi::AnyBuffer> jax_arrays;
    anchor_fixtures.reserve(kNumLayers);
    jax_arrays.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      anchor_fixtures.emplace_back(XLA_FFI_DataType_S32,
                                   shard_layer_data[s][l].data(),
                                   std::vector<int64_t>{kSliceElements});
      jax_arrays.push_back(anchor_fixtures.back().AsAnyBuffer());
    }

    std::vector<int32_t> out_data(6, 0);
    FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
    xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

    xla::ffi::Error err = TriggerWeightSynchronizerInitAndD2hHelper(
        shard_buf, slice_byte_sizes_buf, jax_arrays,
        /*local_port=*/0, /*parallelism=*/1, kNumLayers, kListenerPort,
        kNumShards, out);
    ASSERT_TRUE(err.success()) << "InitAndD2h failed for shard " << global_shard
                               << ": " << err.message();
  }

  // Verify all 4 global shards share the same synchronizer instance
  WeightSynchronizerBase* ws = StandardWs(kBaseShard);
  ASSERT_NE(ws, nullptr);
  for (int s = 1; s < kNumShards; ++s) {
    EXPECT_EQ(StandardWs(kBaseShard + s), ws);
  }

  // Verify global shard indices are mapped to slots 0, 1, 2, 3
  for (int s = 0; s < kNumShards; ++s) {
    EXPECT_EQ(ws->global_shard_index(s), kBaseShard + s);
  }

  // Verify that D2H wrote the exact shard data into the corresponding host slot
  for (int s = 0; s < kNumShards; ++s) {
    for (int l = 0; l < kNumLayers; ++l) {
      TF_ASSERT_OK_AND_ASSIGN(
          absl::Span<const uint8_t> host_buf,
          std::as_const(*ws).GetHostBufferWithLocalIndex(l, s).Await());
      ASSERT_GE(host_buf.size(), kSliceElements * sizeof(int32_t));
      const int32_t* host_data =
          reinterpret_cast<const int32_t*>(host_buf.data());
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(host_data[i], shard_layer_data[s][l][i])
            << "Mismatch at shard " << (kBaseShard + s) << " (slot " << s
            << "), layer " << l << ", index " << i;
      }
    }
  }

  // Verify H2D roundtrip for each shard
  for (int s = 0; s < kNumShards; ++s) {
    int32_t global_shard = kBaseShard + s;
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, &global_shard, {1});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<std::vector<int32_t>> dst_data(
        kNumLayers, std::vector<int32_t>(kSliceElements, 0));
    std::vector<FfiBufferFixture> dst_fixtures;
    dst_fixtures.reserve(kNumLayers);
    std::vector<XLA_FFI_Buffer> ffi_bufs;
    ffi_bufs.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      dst_fixtures.emplace_back(XLA_FFI_DataType_S32, dst_data[l].data(),
                                std::vector<int64_t>{kSliceElements});
      ffi_bufs.push_back(dst_fixtures.back().ffi_buf);
    }

    std::vector<XLA_FFI_RetType> types(kNumLayers, XLA_FFI_RetType_BUFFER);
    std::vector<void*> rets_ptrs(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      rets_ptrs[l] = &ffi_bufs[l];
    }

    XLA_FFI_Rets rets;
    rets.struct_size = sizeof(XLA_FFI_Rets);
    rets.extension_start = nullptr;
    rets.size = kNumLayers;
    rets.types = types.data();
    rets.rets = rets_ptrs.data();

    xla::ffi::RemainingRets remaining_rets(&rets, 0);
    xla::ffi::Error err = TriggerMultiH2DImpl(shard_buf, remaining_rets);
    EXPECT_TRUE(err.success()) << "MultiH2D failed for shard " << global_shard
                               << ": " << err.message();

    for (int l = 0; l < kNumLayers; ++l) {
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(dst_data[l][i], shard_layer_data[s][l][i])
            << "H2D mismatch at shard " << global_shard << ", layer " << l
            << ", index " << i;
      }
    }
  }
}

TEST_F(WeightSynchronizerFfiTest,
       TriggerMultiNumaSubmanagersDistinctAndNoCollision) {
  constexpr int kTotalShards = 8;
  constexpr int kShardsPerNuma = 4;
  constexpr int kNumLayers = 2;
  constexpr int kSliceElements = 256;
  constexpr int32_t kSliceByteSize = kSliceElements * sizeof(int32_t);
  constexpr int32_t kListenerPort = 0;  // Ephemeral port allocation

  std::vector<int32_t> slice_byte_sizes(kNumLayers, kSliceByteSize);
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, slice_byte_sizes.data(),
                                 {kNumLayers});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  // Create unique source data for each (shard, layer)
  std::vector<std::vector<std::vector<int32_t>>> shard_layer_data(
      kTotalShards, std::vector<std::vector<int32_t>>(
                        kNumLayers, std::vector<int32_t>(kSliceElements)));

  for (int s = 0; s < kTotalShards; ++s) {
    for (int l = 0; l < kNumLayers; ++l) {
      for (int i = 0; i < kSliceElements; ++i) {
        shard_layer_data[s][l][i] = s * 1000 + l * 100 + i;
      }
    }
  }

  // Execute InitAndD2h for all 8 shards with num_shards = 4
  for (int s = 0; s < kTotalShards; ++s) {
    int32_t global_shard = s;
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, &global_shard, {1});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<FfiBufferFixture> anchor_fixtures;
    std::vector<xla::ffi::AnyBuffer> jax_arrays;
    anchor_fixtures.reserve(kNumLayers);
    jax_arrays.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      anchor_fixtures.emplace_back(XLA_FFI_DataType_S32,
                                   shard_layer_data[s][l].data(),
                                   std::vector<int64_t>{kSliceElements});
      jax_arrays.push_back(anchor_fixtures.back().AsAnyBuffer());
    }

    std::vector<int32_t> out_data(6, 0);
    FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
    xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

    xla::ffi::Error err = TriggerWeightSynchronizerInitAndD2hHelper(
        shard_buf, slice_byte_sizes_buf, jax_arrays,
        /*local_port=*/0, /*parallelism=*/1, kNumLayers, kListenerPort,
        kShardsPerNuma, out);
    ASSERT_TRUE(err.success()) << "InitAndD2h failed for shard " << global_shard
                               << ": " << err.message();
  }

  // Submanager 0 (shards 0..3)
  WeightSynchronizerBase* ws_0 = StandardWs(0);
  ASSERT_NE(ws_0, nullptr);
  for (int s = 1; s < 4; ++s) {
    EXPECT_EQ(StandardWs(s), ws_0)
        << "Shard " << s << " should share submanager 0";
  }

  // Submanager 1 (shards 4..7)
  WeightSynchronizerBase* ws_1 = StandardWs(4);
  ASSERT_NE(ws_1, nullptr);
  for (int s = 5; s < 8; ++s) {
    EXPECT_EQ(StandardWs(s), ws_1)
        << "Shard " << s << " should share submanager 1";
  }

  // Verify that ws_0 and ws_1 are distinct submanager instances
  EXPECT_NE(ws_0, ws_1)
      << "NUMA submanager 0 and submanager 1 must be distinct instances";

  // Verify global shard indices for each submanager
  for (int s = 0; s < kShardsPerNuma; ++s) {
    EXPECT_EQ(ws_0->global_shard_index(s), s);
    EXPECT_EQ(ws_1->global_shard_index(s), s + kShardsPerNuma);
  }

  // Verify host buffer data in ws_0 (shards 0..3)
  for (int s = 0; s < 4; ++s) {
    for (int l = 0; l < kNumLayers; ++l) {
      TF_ASSERT_OK_AND_ASSIGN(
          absl::Span<const uint8_t> host_buf,
          std::as_const(*ws_0).GetHostBufferWithLocalIndex(l, s).Await());
      ASSERT_GE(host_buf.size(), kSliceElements * sizeof(int32_t));
      const int32_t* host_data =
          reinterpret_cast<const int32_t*>(host_buf.data());
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(host_data[i], shard_layer_data[s][l][i])
            << "Mismatch in ws_0 at shard " << s << ", layer " << l
            << ", index " << i;
      }
    }
  }

  // Verify host buffer data in ws_1 (shards 4..7)
  for (int s = 4; s < 8; ++s) {
    int slot = s - 4;
    for (int l = 0; l < kNumLayers; ++l) {
      TF_ASSERT_OK_AND_ASSIGN(
          absl::Span<const uint8_t> host_buf,
          std::as_const(*ws_1).GetHostBufferWithLocalIndex(l, slot).Await());
      ASSERT_GE(host_buf.size(), kSliceElements * sizeof(int32_t));
      const int32_t* host_data =
          reinterpret_cast<const int32_t*>(host_buf.data());
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(host_data[i], shard_layer_data[s][l][i])
            << "Mismatch in ws_1 at shard " << s << " (slot " << slot
            << "), layer " << l << ", index " << i;
      }
    }
  }

  // Verify H2D roundtrip for all 8 shards
  for (int s = 0; s < kTotalShards; ++s) {
    int32_t global_shard = s;
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, &global_shard, {1});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<std::vector<int32_t>> dst_data(
        kNumLayers, std::vector<int32_t>(kSliceElements, 0));
    std::vector<FfiBufferFixture> dst_fixtures;
    dst_fixtures.reserve(kNumLayers);
    std::vector<XLA_FFI_Buffer> ffi_bufs;
    ffi_bufs.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      dst_fixtures.emplace_back(XLA_FFI_DataType_S32, dst_data[l].data(),
                                std::vector<int64_t>{kSliceElements});
      ffi_bufs.push_back(dst_fixtures.back().ffi_buf);
    }

    std::vector<XLA_FFI_RetType> types(kNumLayers, XLA_FFI_RetType_BUFFER);
    std::vector<void*> rets_ptrs(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      rets_ptrs[l] = &ffi_bufs[l];
    }

    XLA_FFI_Rets rets;
    rets.struct_size = sizeof(XLA_FFI_Rets);
    rets.extension_start = nullptr;
    rets.size = kNumLayers;
    rets.types = types.data();
    rets.rets = rets_ptrs.data();

    xla::ffi::RemainingRets remaining_rets(&rets, 0);
    xla::ffi::Error err = TriggerMultiH2DImpl(shard_buf, remaining_rets);
    EXPECT_TRUE(err.success()) << "MultiH2D failed for shard " << global_shard
                               << ": " << err.message();

    for (int l = 0; l < kNumLayers; ++l) {
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(dst_data[l][i], shard_layer_data[s][l][i])
            << "H2D mismatch at shard " << global_shard << ", layer " << l
            << ", index " << i;
      }
    }
  }
}

TEST_F(WeightSynchronizerFfiTest, NonContiguousShardsInitAndD2hTest) {
  constexpr int kNumLayers = 3;
  constexpr int kSliceElements = 256;
  constexpr int kShardsPerHost = 4;
  constexpr int32_t kListenerPort = 61000;

  // 2 hosts with non-contiguous 2D subgrid shards:
  // Host 0: global shards [0, 1, 4, 5], local slots [0, 1, 2, 3]
  // Host 1: global shards [2, 3, 6, 7], local slots [0, 1, 2, 3]
  struct ShardAssignment {
    int32_t global_shard;
    int32_t local_slot;
    int32_t host_idx;
  };
  std::vector<ShardAssignment> assignments = {
      {0, 0, 0}, {1, 1, 0}, {4, 2, 0}, {5, 3, 0},
      {2, 0, 1}, {3, 1, 1}, {6, 2, 1}, {7, 3, 1},
  };

  std::vector<int32_t> slice_byte_sizes(kNumLayers,
                                        kSliceElements * sizeof(int32_t));
  FfiBufferFixture slice_sizes_fixture(XLA_FFI_DataType_S32,
                                       slice_byte_sizes.data(), {kNumLayers});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = slice_sizes_fixture.AsAnyBuffer();

  // Distinct test data per shard and per layer
  absl::flat_hash_map<int32_t, std::vector<std::vector<int32_t>>>
      shard_layer_data;
  for (const auto& a : assignments) {
    auto& layers = shard_layer_data[a.global_shard];
    layers.resize(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      layers[l].resize(kSliceElements);
      for (int i = 0; i < kSliceElements; ++i) {
        layers[l][i] = (a.global_shard + 1) * 100000 + (l + 1) * 1000 + i;
      }
    }
  }

  for (const auto& a : assignments) {
    int32_t shard_info[3] = {a.global_shard, a.local_slot, a.host_idx};
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, shard_info, {3});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<FfiBufferFixture> anchor_fixtures;
    std::vector<xla::ffi::AnyBuffer> jax_arrays;
    anchor_fixtures.reserve(kNumLayers);
    jax_arrays.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      anchor_fixtures.emplace_back(XLA_FFI_DataType_S32,
                                   shard_layer_data[a.global_shard][l].data(),
                                   std::vector<int64_t>{kSliceElements});
      jax_arrays.push_back(anchor_fixtures.back().AsAnyBuffer());
    }

    std::vector<int32_t> out_data(6, 0);
    FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
    xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

    xla::ffi::Error err = TriggerWeightSynchronizerInitAndD2hHelper(
        shard_buf, slice_byte_sizes_buf, jax_arrays,
        /*local_port=*/0, /*parallelism=*/1, kNumLayers, kListenerPort,
        kShardsPerHost, out);
    ASSERT_TRUE(err.success()) << "InitAndD2h failed for shard "
                               << a.global_shard << ": " << err.message();
  }

  // Host 0: shards 0, 1, 4, 5
  WeightSynchronizerBase* ws_0 = StandardWs(0);
  ASSERT_NE(ws_0, nullptr);
  EXPECT_EQ(StandardWs(1), ws_0);
  EXPECT_EQ(StandardWs(4), ws_0);
  EXPECT_EQ(StandardWs(5), ws_0);

  // Host 1: shards 2, 3, 6, 7
  WeightSynchronizerBase* ws_1 = StandardWs(2);
  ASSERT_NE(ws_1, nullptr);
  EXPECT_EQ(StandardWs(3), ws_1);
  EXPECT_EQ(StandardWs(6), ws_1);
  EXPECT_EQ(StandardWs(7), ws_1);

  EXPECT_NE(ws_0, ws_1);

  // Verify global shard indices for host 0 and host 1
  EXPECT_EQ(ws_0->global_shard_index(0), 0);
  EXPECT_EQ(ws_0->global_shard_index(1), 1);
  EXPECT_EQ(ws_0->global_shard_index(2), 4);
  EXPECT_EQ(ws_0->global_shard_index(3), 5);

  EXPECT_EQ(ws_1->global_shard_index(0), 2);
  EXPECT_EQ(ws_1->global_shard_index(1), 3);
  EXPECT_EQ(ws_1->global_shard_index(2), 6);
  EXPECT_EQ(ws_1->global_shard_index(3), 7);

  // Verify host buffer data in ws_0 and ws_1
  for (const auto& a : assignments) {
    WeightSynchronizerBase* ws = (a.host_idx == 0) ? ws_0 : ws_1;
    for (int l = 0; l < kNumLayers; ++l) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> host_buf,
                           std::as_const(*ws)
                               .GetHostBufferWithLocalIndex(l, a.local_slot)
                               .Await());
      ASSERT_GE(host_buf.size(), kSliceElements * sizeof(int32_t));
      const int32_t* host_data =
          reinterpret_cast<const int32_t*>(host_buf.data());
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(host_data[i], shard_layer_data[a.global_shard][l][i])
            << "Mismatch at shard " << a.global_shard << " (slot "
            << a.local_slot << "), layer " << l << ", index " << i;
      }
    }
  }

  // Verify H2D roundtrip for all shards
  for (const auto& a : assignments) {
    int32_t shard_info[3] = {a.global_shard, a.local_slot, a.host_idx};
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, shard_info, {3});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<std::vector<int32_t>> dst_data(
        kNumLayers, std::vector<int32_t>(kSliceElements, 0));
    std::vector<FfiBufferFixture> dst_fixtures;
    dst_fixtures.reserve(kNumLayers);
    std::vector<XLA_FFI_Buffer> ffi_bufs;
    ffi_bufs.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      dst_fixtures.emplace_back(XLA_FFI_DataType_S32, dst_data[l].data(),
                                std::vector<int64_t>{kSliceElements});
      ffi_bufs.push_back(dst_fixtures.back().ffi_buf);
    }

    std::vector<XLA_FFI_RetType> types(kNumLayers, XLA_FFI_RetType_BUFFER);
    std::vector<void*> rets_ptrs(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      rets_ptrs[l] = &ffi_bufs[l];
    }

    XLA_FFI_Rets rets;
    rets.struct_size = sizeof(XLA_FFI_Rets);
    rets.extension_start = nullptr;
    rets.size = kNumLayers;
    rets.types = types.data();
    rets.rets = rets_ptrs.data();

    xla::ffi::RemainingRets remaining_rets(&rets, 0);
    xla::ffi::Error err = TriggerMultiH2DImpl(shard_buf, remaining_rets);
    EXPECT_TRUE(err.success()) << "MultiH2D failed for shard " << a.global_shard
                               << ": " << err.message();

    for (int l = 0; l < kNumLayers; ++l) {
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(dst_data[l][i], shard_layer_data[a.global_shard][l][i])
            << "H2D mismatch at shard " << a.global_shard << ", layer " << l
            << ", index " << i;
      }
    }
  }
}

TEST_F(WeightSynchronizerFfiTest, TorusPermutedDeviceIdsInitAndD2hTest) {
  constexpr int kNumLayers = 2;
  constexpr int kSliceElements = 128;
  constexpr int kShardsPerHost = 4;
  constexpr int32_t kListenerPort = 61010;

  // Simulates a 2x2x2 torus slice where physical device IDs (d.id) are ordered
  // along a 4-chip ICI ring:
  // Host 0: physical d.id = [0, 1, 3, 2], slots = [0, 1, 2, 3], global = [0..3]
  // Host 1: physical d.id = [6, 7, 5, 4], slots = [0, 1, 2, 3], global = [4..7]
  struct ShardAssignment {
    int32_t device_id;
    int32_t local_slot;
    int32_t host_idx;
    int32_t global_mesh_idx;
  };
  std::vector<ShardAssignment> assignments = {
      {0, 0, 0, 0}, {1, 1, 0, 1}, {3, 2, 0, 2}, {2, 3, 0, 3},
      {6, 0, 1, 4}, {7, 1, 1, 5}, {5, 2, 1, 6}, {4, 3, 1, 7},
  };

  std::vector<int32_t> slice_byte_sizes(kNumLayers,
                                        kSliceElements * sizeof(int32_t));
  FfiBufferFixture slice_sizes_fixture(XLA_FFI_DataType_S32,
                                       slice_byte_sizes.data(), {kNumLayers});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = slice_sizes_fixture.AsAnyBuffer();

  absl::flat_hash_map<int32_t, std::vector<std::vector<int32_t>>>
      device_layer_data;
  for (const auto& a : assignments) {
    auto& layers = device_layer_data[a.device_id];
    layers.resize(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      layers[l].resize(kSliceElements);
      for (int i = 0; i < kSliceElements; ++i) {
        layers[l][i] = (a.global_mesh_idx + 1) * 100000 + (l + 1) * 1000 + i;
      }
    }
  }

  for (const auto& a : assignments) {
    int32_t shard_info[4] = {a.device_id, a.local_slot, a.host_idx,
                             a.global_mesh_idx};
    FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, shard_info, {4});
    xla::ffi::AnyBuffer shard_buf = shard_fixture.AsAnyBuffer();

    std::vector<FfiBufferFixture> anchor_fixtures;
    std::vector<xla::ffi::AnyBuffer> jax_arrays;
    anchor_fixtures.reserve(kNumLayers);
    jax_arrays.reserve(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
      anchor_fixtures.emplace_back(XLA_FFI_DataType_S32,
                                   device_layer_data[a.device_id][l].data(),
                                   std::vector<int64_t>{kSliceElements});
      jax_arrays.push_back(anchor_fixtures.back().AsAnyBuffer());
    }

    std::vector<int32_t> out_data(6, 0);
    FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
    xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

    xla::ffi::Error err = TriggerWeightSynchronizerInitAndD2hHelper(
        shard_buf, slice_byte_sizes_buf, jax_arrays,
        /*local_port=*/0, /*parallelism=*/1, kNumLayers, kListenerPort,
        kShardsPerHost, out);
    ASSERT_TRUE(err.success()) << "InitAndD2h failed for device " << a.device_id
                               << ": " << err.message();
  }

  WeightSynchronizerBase* ws_0 = StandardWs(0);
  WeightSynchronizerBase* ws_1 = StandardWs(6);
  ASSERT_NE(ws_0, nullptr);
  ASSERT_NE(ws_1, nullptr);
  EXPECT_NE(ws_0, ws_1);

  // Verify global shard indices match logical mesh order [0, 1, 2, 3] and
  // [4, 5, 6, 7], NOT physical device IDs [0, 1, 3, 2] or [6, 7, 5, 4].
  EXPECT_EQ(ws_0->global_shard_index(0), 0);
  EXPECT_EQ(ws_0->global_shard_index(1), 1);
  EXPECT_EQ(ws_0->global_shard_index(2), 2);
  EXPECT_EQ(ws_0->global_shard_index(3), 3);

  EXPECT_EQ(ws_1->global_shard_index(0), 4);
  EXPECT_EQ(ws_1->global_shard_index(1), 5);
  EXPECT_EQ(ws_1->global_shard_index(2), 6);
  EXPECT_EQ(ws_1->global_shard_index(3), 7);

  // Verify slot 2 holds device 3's data (tp=2) and slot 3 holds device 2's
  // data (tp=3).
  for (const auto& a : assignments) {
    WeightSynchronizerBase* ws = (a.host_idx == 0) ? ws_0 : ws_1;
    for (int l = 0; l < kNumLayers; ++l) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> host_buf,
                           std::as_const(*ws)
                               .GetHostBufferWithLocalIndex(l, a.local_slot)
                               .Await());
      ASSERT_GE(host_buf.size(), kSliceElements * sizeof(int32_t));
      const int32_t* host_data =
          reinterpret_cast<const int32_t*>(host_buf.data());
      for (int i = 0; i < kSliceElements; ++i) {
        EXPECT_EQ(host_data[i], device_layer_data[a.device_id][l][i])
            << "Mismatch at device " << a.device_id << " (slot " << a.local_slot
            << "), layer " << l << ", index " << i;
      }
    }
  }
}

TEST_F(WeightSynchronizerFfiTest,
       InitAndD2hAndMultiH2DRecordsTelemetryMetrics) {
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  ASSERT_TRUE(store.InitializeFromBackendNames({"cloud_logging"}).ok());
  store.GetAndResetMetricSamples();

  int32_t shard_idx_0 = 0;
  int32_t shard_idx_1 = 1;
  FfiBufferFixture shard_0_fixture(XLA_FFI_DataType_S32, &shard_idx_0, {1});
  FfiBufferFixture shard_1_fixture(XLA_FFI_DataType_S32, &shard_idx_1, {1});
  xla::ffi::AnyBuffer shard_0_buf = shard_0_fixture.AsAnyBuffer();
  xla::ffi::AnyBuffer shard_1_buf = shard_1_fixture.AsAnyBuffer();

  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  std::vector<int32_t> src_data(256, 42);
  FfiBufferFixture src_fixture(XLA_FFI_DataType_S32, src_data.data(), {256});
  std::vector<xla::ffi::AnyBuffer> jax_arrays = {src_fixture.AsAnyBuffer()};

  std::vector<int32_t> out_data_0(6, 0);
  std::vector<int32_t> out_data_1(6, 0);
  FfiBufferFixture out_fixture_0(XLA_FFI_DataType_S32, out_data_0.data(), {6});
  FfiBufferFixture out_fixture_1(XLA_FFI_DataType_S32, out_data_1.data(), {6});
  xla::ffi::Result<xla::ffi::AnyBuffer> out_0 = out_fixture_0.AsAnyBuffer();
  xla::ffi::Result<xla::ffi::AnyBuffer> out_1 = out_fixture_1.AsAnyBuffer();

  xla::ffi::Error err0 = TriggerWeightSynchronizerInitAndD2hHelper(
      shard_0_buf, slice_byte_sizes_buf, jax_arrays,
      /*local_port=*/0, /*parallelism=*/1, /*num_layers=*/1,
      /*listener_port=*/10031, /*num_shards=*/2, out_0);
  ASSERT_TRUE(err0.success()) << err0.message();

  xla::ffi::Error err1 = TriggerWeightSynchronizerInitAndD2hHelper(
      shard_1_buf, slice_byte_sizes_buf, jax_arrays,
      /*local_port=*/0, /*parallelism=*/1, /*num_layers=*/1,
      /*listener_port=*/10031, /*num_shards=*/2, out_1);
  ASSERT_TRUE(err1.success()) << err1.message();

  auto d2h_samples = store.GetAndResetMetricSamples();
  ASSERT_EQ(d2h_samples.count("tpu_raiden_weight_sync_d2h_transfer_time_ms"),
            1u);
  EXPECT_EQ(d2h_samples["tpu_raiden_weight_sync_d2h_transfer_time_ms"].size(),
            2u);

  std::vector<int32_t> dst_data(256, 0);
  FfiBufferFixture dst_fixture(XLA_FFI_DataType_S32, dst_data.data(), {256});
  XLA_FFI_Buffer ffi_buf = dst_fixture.ffi_buf;
  XLA_FFI_RetType types[] = {XLA_FFI_RetType_BUFFER};
  void* rets_ptrs[] = {&ffi_buf};
  XLA_FFI_Rets rets;
  rets.struct_size = sizeof(XLA_FFI_Rets);
  rets.extension_start = nullptr;
  rets.size = 1;
  rets.types = types;
  rets.rets = rets_ptrs;
  xla::ffi::RemainingRets remaining_rets(&rets, 0);

  // Shard 0 (slot 0) finishes first: should record sample and NOT flush yet.
  testing::internal::CaptureStdout();
  err0 = TriggerMultiH2DImpl(shard_0_buf, remaining_rets);
  std::string stdout_after_shard0 = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(err0.success()) << err0.message();
  EXPECT_EQ(stdout_after_shard0, "");

  // Shard 1 (slot 1) finishes last: should record second sample and flush both.
  testing::internal::CaptureStdout();
  err1 = TriggerMultiH2DImpl(shard_1_buf, remaining_rets);
  std::string stdout_after_shard1 = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(err1.success()) << err1.message();
  EXPECT_NE(stdout_after_shard1.find("\"phase\":\"destination_h2d\""),
            std::string::npos);
  EXPECT_NE(stdout_after_shard1.find(
                "\"tpu_raiden_weight_sync_h2d_transfer_time_ms\":{\"count\":2"),
            std::string::npos);

  // Because the last finishing shard flushed to Cloud Logging, the buffer is
  // now drained.
  auto h2d_samples = store.GetAndResetMetricSamples();
  EXPECT_EQ(h2d_samples.count("tpu_raiden_weight_sync_h2d_transfer_time_ms"),
            0u);
  store.SetBackends({});
}

std::string UnpackIpFromInt32s(const int32_t* words) {
  uint8_t raw[16];
  std::memcpy(raw, words, 16);
  bool is_v4_mapped = (raw[10] == 0xff && raw[11] == 0xff);
  for (int i = 0; i < 10; ++i) {
    if (raw[i] != 0) is_v4_mapped = false;
  }
  char buf[INET6_ADDRSTRLEN] = {0};
  if (is_v4_mapped) {
    inet_ntop(AF_INET, raw + 12, buf, sizeof(buf));
  } else {
    inet_ntop(AF_INET6, raw, buf, sizeof(buf));
  }
  return std::string(buf);
}

TEST_F(WeightSynchronizerFfiTest,
       ResolveControlPlaneIpPrefersEth0WhenClassifiedAsDataPlane) {
  // Simulates TPU_RAIDEN_DATA_NICS="eth0,eth2" where eth0 (10.11.3.60) and
  // eth2 (10.34.3.60) are classified as kDataPlane while eth1 (10.35.3.60) and
  // docker0 (172.17.0.1) are classified as kControlPlane.
  // Even for the NUMA-1 submanager whose data IP is 10.34.3.60 (eth2),
  // ResolveControlPlaneIp must return eth0's IP (10.11.3.60) so cpu-np
  // orchestrators can reach the listener.
  std::vector<HostNicAddress> nics = {
      {"docker0", "172.17.0.1", -1, NicClassification::kControlPlane},
      {"eth0", "10.11.3.60", 0, NicClassification::kDataPlane},
      {"eth1", "10.35.3.60", 0, NicClassification::kControlPlane},
      {"eth2", "10.34.3.60", 1, NicClassification::kDataPlane},
  };
  EXPECT_EQ(ResolveControlPlaneIp(nics, /*fallback_ip=*/"10.34.3.60",
                                  /*use_default_route_probe=*/false),
            "10.11.3.60");
}

TEST_F(WeightSynchronizerFfiTest,
       ResolveControlPlaneIpSkipsVirtualNicsAndFallsBackToPhysicalControlNic) {
  std::vector<HostNicAddress> nics = {
      {"lo", "127.0.0.1", -1, NicClassification::kControlPlane},
      {"docker0", "172.17.0.1", -1, NicClassification::kControlPlane},
      {"cilium_host", "10.0.1.1", -1, NicClassification::kControlPlane},
      {"enp0s1", "10.11.5.10", 0, NicClassification::kControlPlane},
      {"enp129s0", "10.34.5.10", 1, NicClassification::kDataPlane},
  };
  EXPECT_EQ(ResolveControlPlaneIp(nics, /*fallback_ip=*/"10.34.5.10",
                                  /*use_default_route_probe=*/false),
            "10.11.5.10");

  std::vector<HostNicAddress> only_virtual = {
      {"docker0", "172.17.0.1", -1, NicClassification::kControlPlane},
  };
  EXPECT_EQ(ResolveControlPlaneIp(only_virtual, /*fallback_ip=*/"10.34.5.10",
                                  /*use_default_route_probe=*/false),
            "10.34.5.10");
}

TEST_P(WeightSynchronizerFfiParamTest,
       TenElementOutputBufferPopulatesControlPlaneIp) {
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});
  std::vector<int32_t> anchor(256, 7);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});
  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});

  std::vector<int32_t> out_data(10, 0);
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {10});
  xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

  xla::ffi::Error err =
      CallInit(anchor_fixture.AsAnyBuffer(), shard_idx_fixture.AsAnyBuffer(),
               sizes_fixture.AsAnyBuffer(), /*local_port=*/0, /*parallelism=*/1,
               /*num_layers=*/1, out, /*listener_port=*/0, /*num_shards=*/1);
  ASSERT_TRUE(err.success()) << err.message();

  EXPECT_GT(out_data[4], 0);
  EXPECT_GT(out_data[5], 0);
  std::string data_ip = UnpackIpFromInt32s(out_data.data());
  std::string ctrl_ip = UnpackIpFromInt32s(out_data.data() + 6);
  EXPECT_FALSE(data_ip.empty());
  EXPECT_FALSE(ctrl_ip.empty());
  EXPECT_NE(ctrl_ip, "0.0.0.0");
}

TEST_P(WeightSynchronizerFfiParamTest, DescriptorNumaNodesSetShardPlacement) {
  std::vector<int32_t> anchor(256, 7);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});
  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  std::vector<int32_t> out_data(5, 0);
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {5});

  // num_shards = 2: [shard_idx, local_slot, submanager_idx, global_shard_idx,
  // numa_0, numa_1].
  std::vector<int32_t> desc = {0, 0, 0, 0, 1, 0};
  FfiBufferFixture desc_fixture(XLA_FFI_DataType_S32, desc.data(), {6});
  xla::ffi::Error err = CallInit(
      anchor_fixture.AsAnyBuffer(), desc_fixture.AsAnyBuffer(),
      sizes_fixture.AsAnyBuffer(), /*local_port=*/0, /*parallelism=*/1,
      /*num_layers=*/1, out_fixture.AsAnyBuffer(), /*listener_port=*/-1,
      /*num_shards=*/2);
  ASSERT_TRUE(err.success()) << err.message();
  ASSERT_NE(StandardWs(0), nullptr);
  EXPECT_THAT(StandardWs(0)->shards(),
              ::testing::ElementsAre(
                  ::testing::Field(&transport::ShardInfo::numa_node, 1),
                  ::testing::Field(&transport::ShardInfo::numa_node, 0)));

  // A NUMA list shorter than num_shards is rejected.
  std::vector<int32_t> partial = {1, 1, 0, 1, 1};
  FfiBufferFixture partial_fixture(XLA_FFI_DataType_S32, partial.data(), {5});
  err = CallInit(anchor_fixture.AsAnyBuffer(), partial_fixture.AsAnyBuffer(),
                 sizes_fixture.AsAnyBuffer(), /*local_port=*/0,
                 /*parallelism=*/1, /*num_layers=*/1, out_fixture.AsAnyBuffer(),
                 /*listener_port=*/-1, /*num_shards=*/2);
  EXPECT_EQ(err.errc(), xla::ffi::ErrorCode::kInvalidArgument);
  EXPECT_THAT(err.message(), ::testing::HasSubstr("4 + num_shards"));
  EXPECT_EQ(StandardWs(1), nullptr);
}

TEST_P(WeightSynchronizerFfiParamTest, ReinitIsIdempotentAndRejectsConflicts) {
  std::vector<int32_t> anchor(256, 7);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});
  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  int32_t shard_idx = 1;
  FfiBufferFixture shard_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});
  auto init = [&](std::vector<int32_t>& row, int32_t num_layers) {
    FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, row.data(), {6});
    return CallInit(anchor_fixture.AsAnyBuffer(), shard_fixture.AsAnyBuffer(),
                    sizes_fixture.AsAnyBuffer(), /*local_port=*/0,
                    /*parallelism=*/1, num_layers, out_fixture.AsAnyBuffer(),
                    /*listener_port=*/0, /*num_shards=*/2);
  };
  std::vector<int32_t> first(6, 0);
  xla::ffi::Error err = init(first, /*num_layers=*/1);
  ASSERT_TRUE(err.success()) << err.message();
  WeightSynchronizerBase* ws = StandardWs(1);
  ASSERT_NE(ws, nullptr);

  // Tunix re-runs init with identical arguments on every rebind.
  std::vector<int32_t> second(6, 0);
  err = init(second, /*num_layers=*/1);
  ASSERT_TRUE(err.success()) << err.message();
  EXPECT_EQ(StandardWs(1), ws);
  EXPECT_EQ(WeightSynchronizerFfiManager::Default().GetLocalSlot(1), 1);
  EXPECT_EQ(second, first);

  std::vector<int32_t> conflicting(6, 0);
  err = init(conflicting, /*num_layers=*/2);
  EXPECT_EQ(err.errc(), xla::ffi::ErrorCode::kFailedPrecondition);
  EXPECT_THAT(err.message(), ::testing::HasSubstr("num_layers 2"));
  EXPECT_EQ(StandardWs(1), ws);
}

TEST_P(WeightSynchronizerFfiParamTest, OutOfRangeLocalSlotIsInvalidArgument) {
  std::vector<int32_t> anchor(256, 7);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});
  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  std::vector<int32_t> desc = {0, /*local_slot=*/2};
  FfiBufferFixture desc_fixture(XLA_FFI_DataType_S32, desc.data(), {2});
  std::vector<int32_t> out_data(6, 0);
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
  xla::ffi::Error err = CallInit(
      anchor_fixture.AsAnyBuffer(), desc_fixture.AsAnyBuffer(),
      sizes_fixture.AsAnyBuffer(), /*local_port=*/0, /*parallelism=*/1,
      /*num_layers=*/1, out_fixture.AsAnyBuffer(), /*listener_port=*/-1,
      /*num_shards=*/2);
  EXPECT_EQ(err.errc(), xla::ffi::ErrorCode::kInvalidArgument);
  EXPECT_THAT(err.message(), ::testing::HasSubstr("local_slot 2"));
  EXPECT_EQ(StandardWs(0), nullptr);
}

INSTANTIATE_TEST_SUITE_P(FfiTypeTests, WeightSynchronizerFfiParamTest,
                         ::testing::Values(FfiType::kInit,
                                           FfiType::kInitAndD2h));

TEST(FfiShardDescriptorTest, FromSpanFillsMissingTrailingFieldsWithDefaults) {
  const std::vector<int32_t> full = {9, 1, 2, 40};
  for (size_t len = 1; len <= full.size(); ++len) {
    SCOPED_TRACE(len);
    TF_ASSERT_OK_AND_ASSIGN(
        FfiShardDescriptor desc,
        FfiShardDescriptor::FromSpan(absl::MakeConstSpan(full.data(), len),
                                     /*num_shards=*/4));
    EXPECT_EQ(desc.shard_idx, 9);
    EXPECT_EQ(desc.local_slot, len >= 2 ? 1 : -1);
    EXPECT_EQ(desc.submanager_idx, len >= 3 ? 2 : -1);
    EXPECT_EQ(desc.global_shard_idx, len >= 4 ? 40 : -1);
    EXPECT_TRUE(desc.slot_numa_nodes.empty());
  }
}

TEST(FfiShardDescriptorTest, FromSpanParsesSlotNumaNodes) {
  const std::vector<int32_t> raw = {5, 1, 1, 5, 0, 0, 1, -1};
  TF_ASSERT_OK_AND_ASSIGN(FfiShardDescriptor desc,
                          FfiShardDescriptor::FromSpan(raw, /*num_shards=*/4));
  EXPECT_EQ(desc.global_shard_idx, 5);
  EXPECT_THAT(desc.slot_numa_nodes, ::testing::ElementsAre(0, 0, 1, -1));
}

TEST(FfiShardDescriptorTest, FromSpanRejectsPartialOrOverlongNumaLists) {
  for (size_t len : {5, 7, 9}) {
    SCOPED_TRACE(len);
    const std::vector<int32_t> raw(len, 0);
    EXPECT_THAT(FfiShardDescriptor::FromSpan(raw, /*num_shards=*/4),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       ::testing::HasSubstr("4 + num_shards")));
  }
  const std::vector<int32_t> bad_node = {0, 0, 0, 0, 0, -2};
  EXPECT_THAT(FfiShardDescriptor::FromSpan(bad_node, /*num_shards=*/2),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("NUMA node -2")));
}

TEST(FfiShardDescriptorTest, EffectiveFieldsFallBackToShardIdx) {
  FfiShardDescriptor desc{.shard_idx = 9};
  EXPECT_EQ(desc.effective_submanager_idx(/*num_shards=*/4), 2);
  EXPECT_EQ(desc.effective_submanager_idx(/*num_shards=*/0), 0);
  EXPECT_EQ(desc.effective_global_shard(), 9);
  EXPECT_FALSE(desc.has_explicit_global());

  desc.submanager_idx = 3;
  desc.global_shard_idx = 40;
  EXPECT_EQ(desc.effective_submanager_idx(/*num_shards=*/4), 3);
  EXPECT_EQ(desc.effective_global_shard(), 40);
  EXPECT_TRUE(desc.has_explicit_global());
}

TEST(FfiShardDescriptorTest, FromSpanRejectsEmptyAndOutOfBoundsShardIdx) {
  EXPECT_THAT(FfiShardDescriptor::FromSpan({}, /*num_shards=*/1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  const std::vector<int32_t> negative = {-1, 0};
  EXPECT_THAT(FfiShardDescriptor::FromSpan(negative, /*num_shards=*/1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("out of bounds")));
  const std::vector<int32_t> too_large = {static_cast<int32_t>(kMaxShards)};
  EXPECT_THAT(FfiShardDescriptor::FromSpan(too_large, /*num_shards=*/1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("out of bounds")));
}

TEST(FfiShardDescriptorTest, FromSpanRejectsOptionalFieldsBelowMinusOne) {
  const std::vector<std::string> names = {"local_slot", "submanager_idx",
                                          "global_shard_idx"};
  for (size_t i = 0; i < names.size(); ++i) {
    SCOPED_TRACE(names[i]);
    std::vector<int32_t> raw = {0, -1, -1, -1};
    raw[i + 1] = -2;
    EXPECT_THAT(FfiShardDescriptor::FromSpan(raw, /*num_shards=*/1),
                absl_testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    ::testing::HasSubstr(absl::StrCat(names[i], " -2"))));
  }
}

using IpBytes = std::array<uint8_t, 16>;

IpBytes BytesOf(const int32_t* words) {
  IpBytes bytes;
  std::memcpy(bytes.data(), words, bytes.size());
  return bytes;
}

TEST(ShardEndpointsTest, PackToSpanWritesWireLayoutForEveryWidth) {
  struct IpCase {
    std::string ip;
    IpBytes bytes;
  };
  // IPv4 is written IPv4-mapped (::ffff:a.b.c.d); IPv6 is written verbatim; an
  // empty IP is all zeros.
  const IpCase v4_data = {
      "10.1.2.3", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 1, 2, 3}};
  const IpCase v4_ctrl = {
      "10.9.8.7", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 9, 8, 7}};
  const IpCase v6_data = {
      "2001:0db8:0:0:0:0:0:1",
      {0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01}};
  const IpCase v6_ctrl = {
      "fe80::2", {0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02}};
  const IpCase empty = {"", {}};
  const std::vector<std::pair<IpCase, IpCase>> ip_pairs = {
      {v4_data, v4_ctrl}, {v6_data, v6_ctrl}, {empty, empty}};

  constexpr int32_t kSentinel = -7;
  for (size_t width : {5, 6, 10}) {
    for (const auto& [data, ctrl] : ip_pairs) {
      SCOPED_TRACE(absl::StrCat("width=", width, " data_ip=", data.ip));
      ShardEndpoints endpoints{.data_ip = data.ip,
                               .data_port = 1234,
                               .listener_port = 5678,
                               .control_ip = ctrl.ip};
      std::vector<int32_t> row(width, kSentinel);
      ABSL_ASSERT_OK(endpoints.PackToSpan(absl::MakeSpan(row)));
      EXPECT_EQ(BytesOf(row.data()), data.bytes);
      EXPECT_EQ(row[4], 1234);
      if (width >= 6) {
        EXPECT_EQ(row[5], 5678);
      }
      if (width >= 10) {
        EXPECT_EQ(BytesOf(row.data() + 6), ctrl.bytes);
      }
    }
  }
}

TEST(ShardEndpointsTest, RejectsRowsNarrowerThanFiveInts) {
  std::vector<int32_t> row(4, -7);
  ShardEndpoints endpoints{.data_ip = "10.1.2.3", .data_port = 1};
  EXPECT_THAT(endpoints.PackToSpan(absl::MakeSpan(row)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("got 4")));
  EXPECT_EQ(row, std::vector<int32_t>(4, -7));
}

TEST_F(WeightSynchronizerFfiTest,
       DefaultManagerSharesInstancePerSubmanagerAndRegistersShards) {
  WeightSynchronizerFfiManager& manager =
      WeightSynchronizerFfiManager::Default();
  const std::vector<size_t> sizes = {1024};
  auto create = [&](int32_t shard_idx) {
    return manager.GetOrCreateSynchronizer(
        FfiShardDescriptor{.shard_idx = shard_idx}, /*listener_port=*/-1,
        /*num_layers=*/1, /*parallelism=*/1, sizes, /*local_port=*/0,
        /*num_shards=*/2);
  };
  // With num_shards = 2, shards 0 and 1 form submanager 0; shard 2 starts
  // submanager 1.
  TF_ASSERT_OK_AND_ASSIGN(WeightSynchronizerBase * ws0, create(0));
  TF_ASSERT_OK_AND_ASSIGN(WeightSynchronizerBase * ws1, create(1));
  TF_ASSERT_OK_AND_ASSIGN(WeightSynchronizerBase * ws2, create(2));
  EXPECT_EQ(ws1, ws0);
  EXPECT_NE(ws2, ws0);
  EXPECT_EQ(StandardWs(0), ws0);
  EXPECT_EQ(StandardWs(1), ws0);
  EXPECT_EQ(StandardWs(2), ws2);
  EXPECT_EQ(manager.GetSynchronizerForShard(-1), nullptr);
  EXPECT_EQ(manager.GetSynchronizerForShard(kMaxShards), nullptr);
  EXPECT_EQ(manager.GetLocalSlot(0), 0);
  EXPECT_EQ(manager.GetLocalSlot(1), 1);
  EXPECT_EQ(manager.GetLocalSlot(2), 0);

  // ClearSharedWsMap unregisters every shard but keeps instances alive.
  ClearSharedWsMap();
  EXPECT_EQ(StandardWs(0), nullptr);
  EXPECT_EQ(StandardWs(2), nullptr);
  EXPECT_EQ(ws0->num_shards(), 2);
  EXPECT_EQ(ws2->num_shards(), 2);
  delete ws0;
  delete ws2;
}

TEST_F(WeightSynchronizerFfiTest, ManagerClearWithDeleteResetsRegistration) {
  auto manager = std::make_unique<WeightSynchronizerFfiManager>();
  const std::vector<size_t> sizes = {1024};
  for (int32_t shard_idx : {0, 1}) {
    ABSL_ASSERT_OK(manager
                       ->GetOrCreateSynchronizer(
                           FfiShardDescriptor{.shard_idx = shard_idx},
                           /*listener_port=*/-1, /*num_layers=*/1,
                           /*parallelism=*/1, sizes,
                           /*local_port=*/0, /*num_shards=*/2)
                       .status());
  }
  ASSERT_NE(manager->GetSynchronizerForShard(0), nullptr);
  EXPECT_EQ(manager->GetSynchronizerForShard(1),
            manager->GetSynchronizerForShard(0));
  EXPECT_FALSE(manager->HasActiveListener());

  // The shared instance is deleted once even though two shards reference it.
  manager->Clear(/*delete_instances=*/true);
  EXPECT_EQ(manager->GetSynchronizerForShard(0), nullptr);
  EXPECT_EQ(manager->GetSynchronizerForShard(1), nullptr);
  std::vector<int32_t> row(10, 0);
  EXPECT_THAT(manager->WriteShardEndpoints(0, absl::MakeSpan(row)),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));

  // Re-registration after Clear creates a new instance; a descriptor without
  // local_slot takes slot shard_idx % num_shards.
  ABSL_ASSERT_OK(
      manager
          ->GetOrCreateSynchronizer(FfiShardDescriptor{.shard_idx = 1},
                                    /*listener_port=*/0, /*num_layers=*/1,
                                    /*parallelism=*/1, sizes, /*local_port=*/0,
                                    /*num_shards=*/2)
          .status());
  EXPECT_EQ(manager->GetLocalSlot(1), 1);
  EXPECT_TRUE(manager->HasActiveListener());
  manager->Clear(/*delete_instances=*/true);
  EXPECT_FALSE(manager->HasActiveListener());
}

// Data NICs on bindable loopback addresses: card A on NUMA node 0 and card B
// on NUMA node 1. None is a control-plane NIC.
constexpr char kCardA[] = "127.0.0.2";
constexpr char kCardB[] = "127.0.0.3";

std::vector<HostNicAddress> TwoSocketNics() {
  return {{"eth1", kCardA, 0, NicClassification::kDataPlane},
          {"eth2", kCardB, 1, NicClassification::kDataPlane}};
}

// Registers every local slot of submanager |submanager_idx| with |manager|,
// each slot passing the host's |slot_numa_nodes|, and returns the 10-int
// endpoint row of each slot.
absl::StatusOr<std::vector<std::vector<int32_t>>> RegisterHost(
    WeightSynchronizerFfiManager& manager, int32_t submanager_idx,
    const std::vector<int32_t>& slot_numa_nodes) {
  const int32_t num_shards = static_cast<int32_t>(slot_numa_nodes.size());
  std::vector<std::vector<int32_t>> rows;
  for (int32_t slot = 0; slot < num_shards; ++slot) {
    const int32_t shard_idx = submanager_idx * num_shards + slot;
    const FfiShardDescriptor desc{.shard_idx = shard_idx,
                                  .local_slot = slot,
                                  .submanager_idx = submanager_idx,
                                  .global_shard_idx = shard_idx,
                                  .slot_numa_nodes = slot_numa_nodes};
    ABSL_RETURN_IF_ERROR(manager
                             .GetOrCreateSynchronizer(
                                 desc, /*listener_port=*/-1, /*num_layers=*/1,
                                 /*parallelism=*/1, {1024}, /*local_port=*/0,
                                 num_shards)
                             .status());
    std::vector<int32_t>& row = rows.emplace_back(10, 0);
    ABSL_RETURN_IF_ERROR(
        manager.WriteShardEndpoints(shard_idx, absl::MakeSpan(row)));
  }
  return rows;
}

// Data IP (words 0..3) of every endpoint row.
std::vector<std::string> DataIps(
    const std::vector<std::vector<int32_t>>& rows) {
  std::vector<std::string> ips;
  for (const std::vector<int32_t>& row : rows) {
    ips.push_back(UnpackIpFromInt32s(row.data()));
  }
  return ips;
}

TEST(WeightSynchronizerFfiManagerNumaTest, SlotsReportTheDataNicOfTheirSocket) {
  struct Case {
    std::vector<int32_t> slot_numa_nodes;
    int32_t submanager_idx;
    std::vector<std::string> expected_data_ips;
  };
  const std::vector<Case> cases = {
      {{0, 0, 1, 1}, 0, {kCardA, kCardA, kCardB, kCardB}},
      {{1, 1, 0, 0}, 0, {kCardB, kCardB, kCardA, kCardA}},
      // Chips 4-7 of an 8-chip host whose upper half sits on socket 1.
      {{1, 1, 1, 1}, 1, {kCardB, kCardB, kCardB, kCardB}},
      // Unknown sockets fall back to the first data NIC.
      {{-1, -1, -1, -1}, 0, {kCardA, kCardA, kCardA, kCardA}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(absl::StrJoin(c.slot_numa_nodes, ","));
    WeightSynchronizerFfiManager manager(TwoSocketNics);
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<int32_t>> rows,
        RegisterHost(manager, c.submanager_idx, c.slot_numa_nodes));
    EXPECT_EQ(DataIps(rows), c.expected_data_ips);
    const WeightSynchronizerBase* ws = manager.GetSynchronizerForShard(
        c.submanager_idx * static_cast<int32_t>(c.slot_numa_nodes.size()));
    ASSERT_NE(ws, nullptr);
    for (size_t slot = 0; slot < c.slot_numa_nodes.size(); ++slot) {
      EXPECT_EQ(ws->shards()[slot].numa_node, c.slot_numa_nodes[slot]);
    }
    manager.Clear(/*delete_instances=*/true);
  }
}

TEST(WeightSynchronizerFfiManagerNumaTest,
     SlotsOnDifferentSocketsReportTheSameControlIp) {
  // No NIC is a control-plane NIC, so the control IP falls back to an
  // instance-wide address rather than each slot's data NIC.
  WeightSynchronizerFfiManager manager(TwoSocketNics);
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<int32_t>> rows,
                          RegisterHost(manager, /*submanager_idx=*/0,
                                       /*slot_numa_nodes=*/{0, 1}));
  ASSERT_THAT(DataIps(rows), ::testing::ElementsAre(kCardA, kCardB));
  EXPECT_EQ(UnpackIpFromInt32s(rows[0].data() + 6),
            UnpackIpFromInt32s(rows[1].data() + 6));
  manager.Clear(/*delete_instances=*/true);
}

// `GetOrCreateSynchronizer` arguments besides the descriptor.
struct InitArgs {
  int32_t num_shards = 4;
  int32_t num_layers = 1;
  int32_t parallelism = 1;
  std::vector<size_t> sizes = {1024};
  int32_t listener_port = -1;
  int32_t local_port = 0;
};

absl::StatusOr<WeightSynchronizerBase*> Register(
    WeightSynchronizerFfiManager& manager, const FfiShardDescriptor& desc,
    const InitArgs& args = {}) {
  return manager.GetOrCreateSynchronizer(
      desc, args.listener_port, args.num_layers, args.parallelism, args.sizes,
      args.local_port, args.num_shards);
}

TEST(WeightSynchronizerFfiManagerReuseTest,
     RecordsCallerShardIndicesForOutOfOrderArrivals) {
  // Host 0 with 4 chips that arrive in the order 2, 0, 3, 1. Per shard_idx,
  // each case lists the local slot and global shard index its descriptor
  // carries (only the first |width| descriptor elements are passed).
  struct Case {
    size_t width;
    std::array<int32_t, 4> slot;
    std::array<int32_t, 4> global;
  };
  const std::vector<Case> cases = {
      // Tunix destination: mesh position only, so slot = shard_idx % 4.
      {1, {0, 1, 2, 3}, {0, 1, 2, 3}},
      {3, {0, 1, 2, 3}, {0, 1, 2, 3}},
      // Torus ring order: mesh positions 2 and 3 sit in slots 3 and 2.
      {3, {0, 1, 3, 2}, {0, 1, 2, 3}},
      // Explicit global shard indices in permuted order.
      {4, {0, 1, 2, 3}, {0, 2, 1, 3}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(
        absl::StrCat("width=", c.width, " slots=", absl::StrJoin(c.slot, ",")));
    WeightSynchronizerFfiManager manager(TwoSocketNics);
    for (int32_t shard_idx : {2, 0, 3, 1}) {
      const std::vector<int32_t> raw = {shard_idx, c.slot[shard_idx], 0,
                                        c.global[shard_idx]};
      TF_ASSERT_OK_AND_ASSIGN(
          FfiShardDescriptor desc,
          FfiShardDescriptor::FromSpan(absl::MakeConstSpan(raw.data(), c.width),
                                       /*num_shards=*/4));
      ABSL_ASSERT_OK(Register(manager, desc).status());
    }
    const WeightSynchronizerBase* ws = manager.GetSynchronizerForShard(0);
    ASSERT_NE(ws, nullptr);
    for (int32_t shard_idx = 0; shard_idx < 4; ++shard_idx) {
      EXPECT_EQ(manager.GetSynchronizerForShard(shard_idx), ws);
      EXPECT_EQ(manager.GetLocalSlot(shard_idx),
                static_cast<size_t>(c.slot[shard_idx]));
      EXPECT_EQ(ws->global_shard_index(c.slot[shard_idx]), c.global[shard_idx]);
    }
    manager.Clear(/*delete_instances=*/true);
  }
}

TEST(WeightSynchronizerFfiManagerReuseTest,
     ConflictingReinitFailsAndLeavesRegistryUnchanged) {
  WeightSynchronizerFfiManager manager(TwoSocketNics);
  const FfiShardDescriptor shard0{.shard_idx = 0, .slot_numa_nodes = {0, 1}};
  TF_ASSERT_OK_AND_ASSIGN(WeightSynchronizerBase * ws,
                          Register(manager, shard0, {.num_shards = 2}));
  // Identical arguments, with or without the NUMA nodes, are accepted.
  EXPECT_THAT(Register(manager, shard0, {.num_shards = 2}),
              absl_testing::IsOkAndHolds(ws));
  EXPECT_THAT(Register(manager, {.shard_idx = 0}, {.num_shards = 2}),
              absl_testing::IsOkAndHolds(ws));

  struct Case {
    std::string message;
    FfiShardDescriptor desc;
    InitArgs args;
  };
  const std::vector<Case> cases = {
      {"bound to the instance of",
       {.shard_idx = 0},
       {.num_shards = 2, .local_port = -1}},
      {"Cannot re-initialize shard_idx 0 with num_layers 2",
       {.shard_idx = 0},
       {.num_shards = 2, .num_layers = 2}},
      // A new shard joining the instance with mismatched arguments.
      {"Cannot register shard_idx 1 with num_layers 2",
       {.shard_idx = 1},
       {.num_shards = 2, .num_layers = 2}},
      {"num_shards 4", {.shard_idx = 0}, {.num_shards = 4}},
      {"parallelism 2", {.shard_idx = 0}, {.num_shards = 2, .parallelism = 2}},
      {"slice_byte_sizes 2048",
       {.shard_idx = 0},
       {.num_shards = 2, .sizes = {2048}}},
      {"slot NUMA nodes 1,0",
       {.shard_idx = 0, .slot_numa_nodes = {1, 0}},
       {.num_shards = 2}},
      {"local slot 1", {.shard_idx = 0, .local_slot = 1}, {.num_shards = 2}},
      {"global shard index 5",
       {.shard_idx = 0, .global_shard_idx = 5},
       {.num_shards = 2}},
      // Shard 2 of submanager 0 maps to slot 2 % 2 = 0, which shard 0 holds.
      {"local slot 0 of the instance",
       {.shard_idx = 2, .submanager_idx = 0},
       {.num_shards = 2}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.message);
    EXPECT_THAT(Register(manager, c.desc, c.args),
                absl_testing::StatusIs(
                    absl::StatusCode::kFailedPrecondition,
                    ::testing::AllOf(::testing::HasSubstr(c.message),
                                     ::testing::HasSubstr(
                                         "destroy_weight_synchronizer()"))));
  }
  EXPECT_EQ(manager.GetSynchronizerForShard(0), ws);
  EXPECT_EQ(manager.GetSynchronizerForShard(1), nullptr);
  EXPECT_EQ(manager.GetSynchronizerForShard(2), nullptr);
  EXPECT_EQ(manager.GetLocalSlot(0), 0);
  EXPECT_EQ(ws->global_shard_index(0), 0);
  manager.Clear(/*delete_instances=*/true);
}

TEST(WeightSynchronizerFfiManagerReuseTest,
     RejectsOutOfRangeLocalSlotAndNumShards) {
  WeightSynchronizerFfiManager manager(TwoSocketNics);
  EXPECT_THAT(
      Register(manager, {.shard_idx = 0, .local_slot = 2}, {.num_shards = 2}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("local_slot 2")));
  EXPECT_THAT(Register(manager, {.shard_idx = 0}, {.num_shards = 0}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("num_shards")));
  EXPECT_EQ(manager.GetSynchronizerForShard(0), nullptr);
}

TEST(WeightSynchronizerFfiManagerReuseTest,
     InstanceKeyDistinguishesSubmanagersAndPorts) {
  WeightSynchronizerFfiManager manager(TwoSocketNics);
  // An arithmetic key of listener_port and submanager_idx would map these two
  // to the same instance.
  TF_ASSERT_OK_AND_ASSIGN(
      WeightSynchronizerBase * with_listener,
      Register(manager, {.shard_idx = 0, .submanager_idx = 999},
               {.num_shards = 1, .listener_port = 0}));
  TF_ASSERT_OK_AND_ASSIGN(
      WeightSynchronizerBase * without_listener,
      Register(manager, {.shard_idx = 1, .submanager_idx = 0},
               {.num_shards = 1}));
  EXPECT_NE(without_listener, with_listener);
  EXPECT_TRUE(with_listener->is_listener_active());
  EXPECT_FALSE(without_listener->is_listener_active());
  // A different local_port selects a different instance.
  TF_ASSERT_OK_AND_ASSIGN(
      WeightSynchronizerBase * other_local_port,
      Register(manager, {.shard_idx = 2, .submanager_idx = 0},
               {.num_shards = 1, .local_port = -1}));
  EXPECT_NE(other_local_port, without_listener);
  // Every negative port means "none", so -7 selects the same instance as -1.
  EXPECT_THAT(Register(manager, {.shard_idx = 2, .submanager_idx = 0},
                       {.num_shards = 1, .local_port = -7}),
              absl_testing::IsOkAndHolds(other_local_port));

  // Positive ports are offset by the submanager index, so listener_port P - 1
  // of submanager 1 and listener_port P of submanager 0 both derive port P.
  const int32_t port = tsl::testing::PickUnusedPortOrDie();
  ABSL_ASSERT_OK(Register(manager, {.shard_idx = 3, .submanager_idx = 1},
                          {.num_shards = 1, .listener_port = port - 1})
                     .status());
  EXPECT_THAT(Register(manager, {.shard_idx = 4, .submanager_idx = 0},
                       {.num_shards = 1, .listener_port = port}),
              absl_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr("same listener or data port")));
  EXPECT_EQ(manager.GetSynchronizerForShard(4), nullptr);
  manager.Clear(/*delete_instances=*/true);
}

TEST(WeightSynchronizerFfiManagerReuseTest,
     MultiH2dCountsEachShardOnceAndClearResetsCounts) {
  WeightSynchronizerFfiManager manager(TwoSocketNics);
  for (int32_t shard_idx : {0, 1, 0}) {
    ABSL_ASSERT_OK(
        Register(manager, {.shard_idx = shard_idx}, {.num_shards = 2})
            .status());
  }
  // A round is two completions; counting the re-initialized shard 0 twice
  // would make it three.
  EXPECT_FALSE(manager.RecordMultiH2dCompletion(/*num_shards=*/2));
  EXPECT_TRUE(manager.RecordMultiH2dCompletion(/*num_shards=*/2));
  EXPECT_FALSE(manager.RecordMultiH2dCompletion(/*num_shards=*/2));

  manager.Clear(/*delete_instances=*/true);
  ABSL_ASSERT_OK(
      Register(manager, {.shard_idx = 0}, {.num_shards = 2}).status());
  EXPECT_FALSE(manager.RecordMultiH2dCompletion(/*num_shards=*/2));
  EXPECT_TRUE(manager.RecordMultiH2dCompletion(/*num_shards=*/2));
  manager.Clear(/*delete_instances=*/true);
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden

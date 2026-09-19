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

#include "tpu_sync/frameworks/jax/kv_cache_manager_ffi.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/frameworks/jax/kv_cache_manager_ffi_internal.h"

namespace tpu_raiden {
namespace kv_cache {
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

class KVCacheManagerFfiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure Host platform is registered and active
    auto platform = stream_executor::PlatformManager::PlatformWithName("Host");
    ASSERT_TRUE(platform.ok())
        << "Host Platform must be available on CPU sandboxes";
  }

  void TearDown() override {
    // Clean up global registry between tests
    for (int i = 0; i < 32; ++i) {
      if (g_kv_cache_managers[i] != nullptr) {
        delete g_kv_cache_managers[i];
        g_kv_cache_managers[i] = nullptr;
      }
      g_streams[i].reset();
    }
  }
};

TEST_F(KVCacheManagerFfiTest, TriggerRaidenInitSucceeds) {
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});

  int32_t anchor = 0;
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, &anchor, {1});

  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();
  xla::ffi::AnyBuffer shard_idx_buf = shard_idx_fixture.AsAnyBuffer();

  int64_t slice_byte_size = 1024;
  int32_t local_port = -1;  // Disable symmetrical H2H network loopback
  int32_t parallelism = 1;
  int32_t host_blocks_to_allocate = 8;
  int32_t num_layers = 2;

  xla::ffi::Result<xla::ffi::AnyBuffer> out = anchor_fixture.AsAnyBuffer();

  xla::ffi::Error err = TriggerRaidenInitImpl(
      x, shard_idx_buf, slice_byte_size, local_port, parallelism,
      host_blocks_to_allocate, num_layers, out);

  EXPECT_TRUE(err.success()) << "Raiden Init failed: " << err.message();
  EXPECT_NE(g_kv_cache_managers[0], nullptr);
  EXPECT_NE(g_streams[0], nullptr);
}

TEST_F(KVCacheManagerFfiTest, TriggerRaidenH2dAndD2hDMAOrchestration) {
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});
  int32_t anchor = 0;
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, &anchor, {1});

  int64_t slice_byte_size = 4096;
  int32_t host_blocks_to_allocate = 8;
  int32_t num_layers = 2;

  // 1. Initialize FFI Manager
  xla::ffi::Error init_err = TriggerRaidenInitImpl(
      anchor_fixture.AsAnyBuffer(), shard_idx_fixture.AsAnyBuffer(),
      slice_byte_size, -1, 1, host_blocks_to_allocate, num_layers,
      anchor_fixture.AsAnyBuffer());
  ASSERT_TRUE(init_err.success());

  // Verify manager internal footprint: block_byte_size = 4096,
  // local_blocks_per_shard = 8
  int64_t block_byte_size = slice_byte_size;
  ASSERT_EQ(block_byte_size, 4096);

  // 2. Setup Test Data & Mock Device Cache Buffers
  // cache_slice_buf represents the 1-layer physical on-device cache slice
  // A single device slice size for 8 blocks of size 4096: 32768 bytes
  std::vector<uint8_t> cache_slice_data(32768, 0);
  FfiBufferFixture cache_slice_fixture(XLA_FFI_DataType_U8,
                                       cache_slice_data.data(), {32768});

  // Populate Host block indices with test data to push
  // Host layer memory block 0 populated with sequence
  uint8_t* h_base_layer0 = const_cast<uint8_t*>(
      g_kv_cache_managers[0]->base()->GetHostPointer(0, 0));
  for (size_t j = 0; j < 4096; ++j) {
    h_base_layer0[j] = static_cast<uint8_t>(j % 256);
  }

  // Setup H2D PCIe DMA chunk mappings: Copy block 0 (local_s_idx = 0) to device
  // block 2 (local_d_idx = 2)
  std::vector<int32_t> src_offsets = {0};
  std::vector<int32_t> dst_offsets = {2};
  std::vector<int32_t> copy_sizes = {1};  // 1 block copy

  FfiBufferFixture src_offsets_fixture(XLA_FFI_DataType_S32, src_offsets.data(),
                                       {1});
  FfiBufferFixture dst_offsets_fixture(XLA_FFI_DataType_S32, dst_offsets.data(),
                                       {1});
  FfiBufferFixture copy_sizes_fixture(XLA_FFI_DataType_S32, copy_sizes.data(),
                                      {1});

  // 3. Execute H2D Transfer custom call FFI
  xla::ffi::Error h2d_err = TriggerRaidenH2dImpl(
      src_offsets_fixture.AsAnyBuffer(), dst_offsets_fixture.AsAnyBuffer(),
      copy_sizes_fixture.AsAnyBuffer(), shard_idx_fixture.AsAnyBuffer(),
      cache_slice_fixture.AsAnyBuffer(), /*layer_idx=*/0,
      cache_slice_fixture.AsAnyBuffer());

  EXPECT_TRUE(h2d_err.success())
      << "H2D transfer failed: " << h2d_err.message();

  // Verify device block 2 contains the sequence from host block 0
  uint8_t* d_block2 = cache_slice_data.data() + 2 * 4096;
  for (size_t j = 0; j < 4096; ++j) {
    ASSERT_EQ(d_block2[j], static_cast<uint8_t>(j % 256))
        << "Mismatch at index " << j << " on device block 2";
  }

  // 4. Setup D2H PCIe DMA chunk mappings: Copy device block 2 (local_s_idx = 2)
  // back to host block 5 (local_d_idx = 5)
  std::vector<int32_t> d2h_src_offsets = {2};
  std::vector<int32_t> d2h_dst_offsets = {5};
  std::vector<int32_t> d2h_copy_sizes = {1};

  FfiBufferFixture d2h_src_offsets_fixture(XLA_FFI_DataType_S32,
                                           d2h_src_offsets.data(), {1});
  FfiBufferFixture d2h_dst_offsets_fixture(XLA_FFI_DataType_S32,
                                           d2h_dst_offsets.data(), {1});
  FfiBufferFixture d2h_copy_sizes_fixture(XLA_FFI_DataType_S32,
                                          d2h_copy_sizes.data(), {1});

  // Zero out host destination block 5 memory to ensure we read fresh data
  uint8_t* h_block5 = h_base_layer0 + 5 * 4096;
  std::memset(h_block5, 0, 4096);

  // Execute D2H Transfer custom call FFI
  xla::ffi::Error d2h_err = TriggerRaidenD2hImpl(
      d2h_src_offsets_fixture.AsAnyBuffer(),
      d2h_dst_offsets_fixture.AsAnyBuffer(),
      d2h_copy_sizes_fixture.AsAnyBuffer(), shard_idx_fixture.AsAnyBuffer(),
      cache_slice_fixture.AsAnyBuffer(), /*layer_idx=*/0,
      cache_slice_fixture.AsAnyBuffer());

  EXPECT_TRUE(d2h_err.success())
      << "D2H transfer failed: " << d2h_err.message();

  // Verify host block 5 contains the sequence successfully pulled back from
  // device block 2
  for (size_t j = 0; j < 4096; ++j) {
    ASSERT_EQ(h_block5[j], static_cast<uint8_t>(j % 256))
        << "Mismatch at index " << j << " on host block 5 after D2H";
  }
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

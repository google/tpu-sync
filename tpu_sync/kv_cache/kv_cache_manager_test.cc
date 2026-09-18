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

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

class TestKVCacheManager : public KVCacheManagerBase {
 public:
  TestKVCacheManager(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int host_blocks = 0)
      : KVCacheManagerBase(
            num_layers, num_shards, slice_byte_size,
            /*local_port=*/std::nullopt,
            host_blocks > 0 ? std::make_optional(host_blocks) : std::nullopt) {
    buffer_holds_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      buffer_holds_[l].holds.resize(num_shards);
      for (size_t sh = 0; sh < num_shards; ++sh) {
        layers_[l].shards[sh].device_size = host_blocks > 0
                                                ? host_blocks * slice_byte_size
                                                : num_layers * slice_byte_size;
      }
    }
  }

  void SetLayerPhysicalSizeForTest(size_t layer_idx, size_t physical_size,
                                   int64_t major_dim_size) {
    buffer_holds_[layer_idx].physical_size = physical_size;
    major_dim_size_ = major_dim_size;
  }

  using KVCacheManagerBase::layers_;
  using KVCacheManagerBase::UpdateAllocatedOccupancyMetric;
};

// Pool with one strided live region per block: live [0, 32) and [64, 96)
// within each block, everything else non-live.
PoolSpec StridedPool(const std::string& tag, size_t storage_index,
                     int64_t base_offset, int64_t stride, int64_t num_blocks) {
  return PoolSpec{
      .tag = tag,
      .storage_index = storage_index,
      .base_offset_bytes = base_offset,
      .block_stride_bytes = stride,
      .num_blocks = num_blocks,
      .regions = {RegionSpec{
          .name = "payload",
          .offset_bytes = 0,
          .stride_bytes = 64,
          .unit_bytes = 32,
          .num_units = 2,
          .units_per_stride = 1,
      }},
      .dtype_tag = "dtype_a",
  };
}

// Pool whose single region covers every byte of the block.
PoolSpec DensePool(const std::string& tag, size_t storage_index,
                   int64_t base_offset, int64_t stride, int64_t num_blocks) {
  return PoolSpec{
      .tag = tag,
      .storage_index = storage_index,
      .base_offset_bytes = base_offset,
      .block_stride_bytes = stride,
      .num_blocks = num_blocks,
      .regions = {RegionSpec{
          .name = "block",
          .offset_bytes = 0,
          .stride_bytes = stride,
          .unit_bytes = stride,
          .num_units = 1,
          .units_per_stride = 1,
      }},
      .dtype_tag = "dtype_a",
  };
}

TEST(KVCacheManagerTest, CompilesAndLinksSuccessfully) { EXPECT_TRUE(true); }

TEST(KVCacheManagerTest, RegisterPoolsValidatesAgainstStorage) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);

  // Overlapping pools on one storage are allowed (aliased-raw pattern).
  ABSL_ASSERT_OK(manager.RegisterPools({
      DensePool("kind_a", 0, 0, 128, 2),
      StridedPool("kind_b", 0, 0, 128, 2),
      DensePool("kind_a", 1, 64, 64, 3),
  }));
  EXPECT_TRUE(manager.has_explicit_pools());
  EXPECT_EQ(manager.num_pools(), 3);

  // storage_index out of range.
  absl::Status status =
      manager.RegisterPools({DensePool("kind_a", 5, 0, 128, 2)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("storage_index"));

  // Pool exceeding the storage: 2 blocks x 128 + base 64 > 256.
  status = manager.RegisterPools({DensePool("kind_a", 0, 64, 128, 2)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("exceeds storage bytes"));

  // The device physical size takes precedence over the host mirror size.
  TestKVCacheManager device_backed(/*num_layers=*/1, /*num_shards=*/1,
                                   /*slice_byte_size=*/256, /*host_blocks=*/2);
  device_backed.SetLayerPhysicalSizeForTest(/*layer_idx=*/0,
                                            /*physical_size=*/128,
                                            /*major_dim_size=*/1);
  ABSL_EXPECT_OK(
      device_backed.RegisterPools({DensePool("kind_a", 0, 0, 128, 1)}));
  status = device_backed.RegisterPools({DensePool("kind_a", 0, 0, 128, 2)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("exceeds storage bytes"));
}

// Pool block pointer math with interior base offsets and per-pool strides.
TEST(KVCacheManagerTest, GetPoolBlockRefPointerMath) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/4);
  ABSL_ASSERT_OK(manager.RegisterPools({
      DensePool("kind_a", 0, 0, 128, 2),
      DensePool("kind_b", 0, 256, 64, 4),
  }));

  uint8_t* base = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto ref,
      manager.GetPoolBlockRef(/*pool_idx=*/0, /*shard_idx=*/0, /*block_id=*/1));
  EXPECT_EQ(ref.ptr, base + 128);
  EXPECT_EQ(ref.block_stride_bytes, 128);
  EXPECT_EQ(ref.pool->tag, "kind_a");
  EXPECT_EQ(ref.pool_idx, 0);
  EXPECT_EQ(ref.block_id, 1);

  TF_ASSERT_OK_AND_ASSIGN(
      ref,
      manager.GetPoolBlockRef(/*pool_idx=*/1, /*shard_idx=*/0, /*block_id=*/0));
  EXPECT_EQ(ref.ptr, base + 256);
  TF_ASSERT_OK_AND_ASSIGN(
      ref,
      manager.GetPoolBlockRef(/*pool_idx=*/1, /*shard_idx=*/0, /*block_id=*/3));
  EXPECT_EQ(ref.ptr, base + 256 + 3 * 64);

  EXPECT_EQ(manager.GetPoolBlockRef(0, 0, 2).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(manager.GetPoolBlockRef(1, 0, 4).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(manager.GetPoolBlockRef(2, 0, 0).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(manager.GetPoolBlockRef(0, 1, 0).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(KVCacheManagerTest, RegisterPoolsGrowsDeviceBackedHostMirror) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/64, /*host_blocks=*/1);
  uint8_t* original = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(original, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    original[i] = static_cast<uint8_t>(i);
  }

  manager.SetLayerPhysicalSizeForTest(/*layer_idx=*/0,
                                      /*physical_size=*/256,
                                      /*major_dim_size=*/1);
  ABSL_ASSERT_OK(manager.RegisterPools({DensePool("kind_a", 0, 0, 64, 4)}));
  EXPECT_GE(manager.GetHostSize(/*layer_idx=*/0, /*shard_idx=*/0), 256);

  uint8_t* grown = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(grown, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(grown[i], static_cast<uint8_t>(i));
  }
  TF_ASSERT_OK_AND_ASSIGN(auto last_ref, manager.GetPoolBlockRef(
                                             /*pool_idx=*/0, /*shard_idx=*/0,
                                             /*block_id=*/3));
  EXPECT_EQ(last_ref.ptr, grown + 3 * 64);
}

// The pool table is frozen while plans are active.
TEST(KVCacheManagerTest, RegisterPoolsFailsAfterActivePlanRegistered) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/1);
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(445566);
  request.set_is_sender(true);

  ABSL_ASSERT_OK(
      manager.RegisterActivePlan(445566, request, /*is_sender=*/true));

  absl::Status status =
      manager.RegisterPools({DensePool("kind_a", 0, 0, 128, 1)});
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), testing::HasSubstr("active plans"));
}

TEST(KVCacheManagerTest, PoolIndicesWithTag) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  PoolSpec pool_b = DensePool("kind_b", 1, 0, 128, 2);
  pool_b.dtype_tag = "dtype_b";
  ABSL_ASSERT_OK(manager.RegisterPools({
      DensePool("kind_a", 0, 0, 128, 2),
      pool_b,
      StridedPool("kind_a", 1, 0, 128, 2),
  }));

  EXPECT_THAT(manager.PoolIndicesWithTag("kind_a"), testing::ElementsAre(0, 2));
  EXPECT_THAT(manager.PoolIndicesWithTag("kind_b"), testing::ElementsAre(1));
  EXPECT_THAT(manager.PoolIndicesWithTag("missing"), testing::IsEmpty());
}

TEST(KVCacheManagerTest, RegisterActivePlanChecksPoolDtypeTags) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  PoolSpec pool_b = DensePool("kind_b", 1, 0, 128, 2);
  pool_b.dtype_tag = "dtype_b";
  ABSL_ASSERT_OK(manager.RegisterPools({
      DensePool("kind_a", 0, 0, 128, 2),
      pool_b,
      StridedPool("kind_a", 1, 0, 128, 2),
  }));

  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(777);
  request.add_pool_dtype_tags("dtype_a");
  request.add_pool_dtype_tags("dtype_b");
  absl::Status status =
      manager.RegisterActivePlan(777, request, /*is_sender=*/true);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("pool_dtype_tags count"));

  request.add_pool_dtype_tags("dtype_a");
  request.set_pool_dtype_tags(1, "dtype_wrong");
  status = manager.RegisterActivePlan(777, request, /*is_sender=*/true);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("dtype tag mismatch"));

  request.set_pool_dtype_tags(1, "dtype_b");
  ABSL_ASSERT_OK(manager.RegisterActivePlan(777, request, /*is_sender=*/true));
  ABSL_EXPECT_OK(manager.UnregisterActivePlan(777));
}

TEST(KVCacheManagerTest, ExplicitPoolAddressingUsesPoolBaseAndStride) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/4);
  ABSL_ASSERT_OK(manager.RegisterPools({
      DensePool("kind_a", /*storage_index=*/1,
                /*base_offset=*/128, /*stride=*/64,
                /*num_blocks=*/4),
  }));

  uint8_t* storage_base =
      manager.GetHostPointer(/*layer_idx=*/1, /*shard_idx=*/0);
  ASSERT_NE(storage_base, nullptr);
  EXPECT_EQ(manager.GetBlockHostPointer(/*pool_idx=*/0, /*shard_idx=*/0,
                                        /*block_id=*/2),
            storage_base + 128 + 2 * 64);

  std::vector<int64_t> block_ids = {0, 2, 3};
  std::vector<transport::BlockChunk> chunks = manager.GetBlockChunks(
      /*pool_idx=*/0, /*shard_idx=*/0, block_ids, /*total_bytes=*/150,
      /*uuid=*/0);
  ASSERT_EQ(chunks.size(), 3);
  EXPECT_EQ(chunks[0].ptr, storage_base + 128);
  EXPECT_EQ(chunks[0].size, 64);
  EXPECT_EQ(chunks[1].ptr, storage_base + 128 + 2 * 64);
  EXPECT_EQ(chunks[1].size, 64);
  EXPECT_EQ(chunks[2].ptr, storage_base + 128 + 3 * 64);
  EXPECT_EQ(chunks[2].size, 22);
}

TEST(KVCacheManagerTest,
     ExplicitPoolTransportUsesBackingStorageBaseAndPoolStride) {
  TestKVCacheManager sender(/*num_layers=*/2, /*num_shards=*/1,
                            /*slice_byte_size=*/64, /*host_blocks=*/8);
  TestKVCacheManager receiver(/*num_layers=*/2, /*num_shards=*/1,
                              /*slice_byte_size=*/64, /*host_blocks=*/8);
  const std::vector<PoolSpec> pools = {
      DensePool("kind_a", /*storage_index=*/1, /*base_offset=*/64,
                /*stride=*/128, /*num_blocks=*/2),
  };
  ABSL_ASSERT_OK(sender.RegisterPools(pools));
  ABSL_ASSERT_OK(receiver.RegisterPools(pools));

  TF_ASSERT_OK_AND_ASSIGN(
      auto src_ref,
      sender.GetPoolBlockRef(/*pool_idx=*/0, /*shard_idx=*/0, /*block_id=*/0));
  TF_ASSERT_OK_AND_ASSIGN(auto dst_ref, receiver.GetPoolBlockRef(
                                            /*pool_idx=*/0, /*shard_idx=*/0,
                                            /*block_id=*/1));
  std::vector<uint8_t> pattern(128);
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>((i * 17 + 3) % 251);
  }
  std::memcpy(src_ref.ptr, pattern.data(), pattern.size());

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  tpu_sync::rpc::StartTransferRequest dummy_plan;
  ABSL_ASSERT_OK(
      receiver.RegisterActivePlan(0, dummy_plan, /*is_sender=*/false));
  ABSL_ASSERT_OK(sender.H2hWriteDirect(
      absl::StrCat(receiver.local_ip(), ":", *receiver_port),
      /*src_block_ids=*/{0}, /*dst_block_ids=*/{1}, /*uuid=*/0,
      /*layer_idx=*/0));
  EXPECT_EQ(std::memcmp(dst_ref.ptr, pattern.data(), pattern.size()), 0);

  const uint8_t* receiver_storage0 =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(receiver_storage0, nullptr);
  EXPECT_TRUE(std::all_of(receiver_storage0, receiver_storage0 + 8 * 64,
                          [](uint8_t value) { return value == 0; }));

  // Block 2 is outside the pool but still falls inside storage 1. It must not
  // expose or overwrite the neighboring storage bytes.
  uint8_t* sender_storage1 =
      sender.GetHostPointer(/*layer_idx=*/1, /*shard_idx=*/0);
  uint8_t* receiver_storage1 =
      receiver.GetHostPointer(/*layer_idx=*/1, /*shard_idx=*/0);
  ASSERT_NE(sender_storage1, nullptr);
  ASSERT_NE(receiver_storage1, nullptr);
  std::memset(sender_storage1 + 64 + 2 * 128, 0xD4, 128);
  auto rejected = sender.H2hWriteDirect(
      absl::StrCat(receiver.local_ip(), ":", *receiver_port),
      /*src_block_ids=*/{2}, /*dst_block_ids=*/{2}, /*uuid=*/0,
      /*layer_idx=*/0);
  ASSERT_FALSE(rejected.ok());
  EXPECT_TRUE(std::all_of(receiver_storage1 + 64 + 2 * 128,
                          receiver_storage1 + 64 + 3 * 128,
                          [](uint8_t value) { return value == 0; }));
}

TEST(KVCacheManagerTest,
     AliasedPoolTransportCopiesOnlyLiveRegionsAndPreservesPadding) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);
  const std::vector<PoolSpec> pools = {
      StridedPool("aliased", /*storage_index=*/0, /*base_offset=*/32,
                  /*stride=*/128, /*num_blocks=*/2),
  };
  ABSL_ASSERT_OK(sender.RegisterPools(pools));
  ABSL_ASSERT_OK(receiver.RegisterPools(pools));

  uint8_t* src = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* dst = receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);
  std::memset(src, 0xA5, 256);
  std::memset(dst, 0xEE, 256);
  std::memset(src + 32, 0x11, 32);
  std::memset(src + 96, 0x22, 32);

  // The logical two-stride array would end at byte 288. Its last live byte is
  // exactly byte 256, so admission and the last block reference are valid.
  TF_ASSERT_OK_AND_ASSIGN(auto last_ref, sender.GetPoolBlockRef(
                                             /*pool_idx=*/0, /*shard_idx=*/0,
                                             /*block_id=*/1));
  EXPECT_EQ(last_ref.ptr, src + 160);
  EXPECT_EQ(sender.GetBlockArrayHostSize(/*pool_idx=*/0, /*shard_idx=*/0), 224);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  tpu_sync::rpc::StartTransferRequest dummy_plan;
  ABSL_ASSERT_OK(
      receiver.RegisterActivePlan(0, dummy_plan, /*is_sender=*/false));
  ABSL_ASSERT_OK(sender.H2hWriteDirect(
      absl::StrCat(receiver.local_ip(), ":", *receiver_port),
      /*src_block_ids=*/{0}, /*dst_block_ids=*/{1}, /*uuid=*/0,
      /*layer_idx=*/0));
  EXPECT_TRUE(std::all_of(dst + 160, dst + 192,
                          [](uint8_t value) { return value == 0x11; }));
  EXPECT_TRUE(std::all_of(dst + 192, dst + 224,
                          [](uint8_t value) { return value == 0xEE; }));
  EXPECT_TRUE(std::all_of(dst + 224, dst + 256,
                          [](uint8_t value) { return value == 0x22; }));
}

TEST(KVCacheManagerTest, ExplicitPoolTransportEnumeratesAllPools) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/64, /*host_blocks=*/8);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/64, /*host_blocks=*/8);
  const std::vector<PoolSpec> pools = {
      DensePool("kind_a", /*storage_index=*/0, /*base_offset=*/0,
                /*stride=*/64, /*num_blocks=*/2),
      DensePool("kind_b", /*storage_index=*/0, /*base_offset=*/256,
                /*stride=*/64, /*num_blocks=*/2),
  };
  ABSL_ASSERT_OK(sender.RegisterPools(pools));
  ABSL_ASSERT_OK(receiver.RegisterPools(pools));
  ASSERT_EQ(sender.num_block_arrays(), 2);
  ASSERT_EQ(receiver.num_block_arrays(), 2);

  TF_ASSERT_OK_AND_ASSIGN(auto sender_a, sender.GetPoolBlockRef(0, 0, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto sender_b, sender.GetPoolBlockRef(1, 0, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto receiver_a, receiver.GetPoolBlockRef(0, 0, 1));
  TF_ASSERT_OK_AND_ASSIGN(auto receiver_b, receiver.GetPoolBlockRef(1, 0, 1));
  std::memset(sender_a.ptr, 0xA1, 64);
  std::memset(sender_b.ptr, 0xB2, 64);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  tpu_sync::rpc::StartTransferRequest dummy_plan;
  ABSL_ASSERT_OK(
      receiver.RegisterActivePlan(0, dummy_plan, /*is_sender=*/false));
  ABSL_ASSERT_OK(sender.H2hWriteDirect(
      absl::StrCat(receiver.local_ip(), ":", *receiver_port),
      /*src_block_ids=*/{0}, /*dst_block_ids=*/{1}, /*uuid=*/0,
      /*layer_idx=*/-1));
  EXPECT_TRUE(std::all_of(receiver_a.ptr, receiver_a.ptr + 64,
                          [](uint8_t value) { return value == 0xA1; }));
  EXPECT_TRUE(std::all_of(receiver_b.ptr, receiver_b.ptr + 64,
                          [](uint8_t value) { return value == 0xB2; }));

  TF_ASSERT_OK_AND_ASSIGN(auto receiver_a0, receiver.GetPoolBlockRef(0, 0, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto receiver_b0, receiver.GetPoolBlockRef(1, 0, 0));
  std::memset(receiver_a0.ptr, 0, 64);
  std::memset(receiver_b0.ptr, 0, 64);
  const std::optional<int> sender_port = sender.local_port();
  ASSERT_TRUE(sender_port.has_value());
  TF_ASSERT_OK_AND_ASSIGN(
      auto pulled,
      receiver.H2hReadDirect(absl::StrCat(sender.local_ip(), ":", *sender_port),
                             /*src_block_ids=*/{0}));
  ASSERT_EQ(pulled, std::vector<int>({0}));
  EXPECT_TRUE(std::all_of(receiver_a0.ptr, receiver_a0.ptr + 64,
                          [](uint8_t value) { return value == 0xA1; }));
  EXPECT_TRUE(std::all_of(receiver_b0.ptr, receiver_b0.ptr + 64,
                          [](uint8_t value) { return value == 0xB2; }));

  std::vector<uint8_t> external_a(2 * 64, 0);
  std::vector<uint8_t> external_b(2 * 64, 0);
  std::vector<uint8_t*> explicit_pool_bases = {external_a.data(),
                                               external_b.data()};
  TF_ASSERT_OK_AND_ASSIGN(
      auto explicit_pull,
      receiver.H2hReadExplicit(
          absl::StrCat(sender.local_ip(), ":", *sender_port),
          /*src_block_ids=*/{0}, /*local_block_ids=*/{1}, explicit_pool_bases,
          /*parallelism=*/1, transport::MajorOrder::kLayerMajor,
          /*on_block_received=*/nullptr));
  ABSL_ASSERT_OK(explicit_pull.Await());
  EXPECT_TRUE(std::all_of(external_a.begin(), external_a.begin() + 64,
                          [](uint8_t value) { return value == 0; }));
  EXPECT_TRUE(std::all_of(external_a.begin() + 64, external_a.end(),
                          [](uint8_t value) { return value == 0xA1; }));
  EXPECT_TRUE(std::all_of(external_b.begin(), external_b.begin() + 64,
                          [](uint8_t value) { return value == 0; }));
  EXPECT_TRUE(std::all_of(external_b.begin() + 64, external_b.end(),
                          [](uint8_t value) { return value == 0xB2; }));
}

TEST(KVCacheManagerTest, PoolBlockCopiesRejectHostOnlyManager) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/64,
                             /*local_port=*/std::nullopt,
                             /*host_blocks_to_allocate=*/2);
  ABSL_ASSERT_OK(manager.RegisterPools({DensePool("kind_a", 0, 0, 64, 2)}));

  auto d2h = manager.D2hPoolBlocks(/*pool_idx=*/0, /*block_ids=*/{0});
  ASSERT_FALSE(d2h.ok());
  EXPECT_EQ(d2h.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(d2h.status().message(), testing::HasSubstr("host-only"));

  auto h2d = manager.H2dPoolBlocks(/*pool_idx=*/0, /*block_ids=*/{0});
  ASSERT_FALSE(h2d.ok());
  EXPECT_EQ(h2d.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(h2d.status().message(), testing::HasSubstr("host-only"));
}

// Without RegisterPools the manager exposes one implicit Opaque pool per
// storage and reports no explicit pools.
TEST(KVCacheManagerTest, ImplicitPoolsMirrorStorages) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  EXPECT_FALSE(manager.has_explicit_pools());
  EXPECT_EQ(manager.num_pools(), 2);

  const PoolSpec* pool = manager.pool(0);
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->tag, "opaque");
  EXPECT_EQ(pool->storage_index, 0);
  EXPECT_EQ(pool->base_offset_bytes, 0);
  EXPECT_EQ(pool->block_stride_bytes, 128);
  EXPECT_EQ(pool->num_blocks, 2);

  TF_ASSERT_OK_AND_ASSIGN(
      auto ref, manager.GetPoolBlockRef(/*pool_idx=*/1, /*shard_idx=*/0,
                                        /*block_id=*/1));
  EXPECT_EQ(ref.ptr, manager.GetHostPointer(/*layer_idx=*/1,
                                            /*shard_idx=*/0) +
                         128);
}

TEST(KVCacheManagerTest, UnregisterActivePlanAllowsUuidReuse) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(112233);
  request.set_is_sender(true);

  absl::Status status = manager.UnregisterActivePlan(112233);
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);

  ABSL_EXPECT_OK(
      manager.RegisterActivePlan(112233, request, /*is_sender=*/true));

  status = manager.RegisterActivePlan(112233, request, /*is_sender=*/true);
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);

  ABSL_EXPECT_OK(manager.UnregisterActivePlan(112233));

  ABSL_EXPECT_OK(
      manager.RegisterActivePlan(112233, request, /*is_sender=*/true));
}

TEST(KVCacheManagerTest, D2hFailsWithMismatchedCopySpecLengths) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0, 1};  // Mismatch
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2h(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must have the same length"));
}

TEST(KVCacheManagerTest, H2dFailsWithNegativeOffsets) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {-1};  // Negative
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.H2d(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must be non-negative"));
}

TEST(KVCacheManagerTest, D2hFailsWithCpuOnlyManager) {
  // Use the base class directly to test CPU-only behavior
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2h(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("requires a device-backed"));
}

TEST(KVCacheManagerTest, H2dFailsWithOutOfRangeLayer) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;

  auto status = manager.H2d(src_offsets, dst_offsets, sizes,
                            /*slot_idx=*/std::nullopt,
                            /*layer_idx=*/1, /*shard_idx=*/0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("layer or shard index out of range"));
}

TEST(KVCacheManagerTest, H2hReadExplicitAcceptsParallelism) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<uint8_t*> ptrs = {nullptr};
  auto status = manager.H2hReadExplicit("127.0.0.1:8080", {0}, {0}, ptrs,
                                        /*parallelism=*/2);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("Failed to connect to peer"));
}

TEST(KVCacheManagerTest, AsymmetricBlockSizesGetBlockChunks) {
  // 1. Sender (Block size: 256 bytes)
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/256);
  std::vector<uint8_t> sender_buffer(256 * 10, 0);
  std::vector<const uint8_t*> sender_ptrs = {sender_buffer.data()};
  std::vector<size_t> sender_sizes = {sender_buffer.size()};
  sender.SetExternalHostPointers(sender_ptrs, sender_sizes);

  // 2. Receiver (Block size: 512 bytes)
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/512);
  std::vector<uint8_t> receiver_buffer(512 * 10, 0);
  std::vector<const uint8_t*> receiver_ptrs = {receiver_buffer.data()};
  std::vector<size_t> receiver_sizes = {receiver_buffer.size()};
  receiver.SetExternalHostPointers(receiver_ptrs, receiver_sizes);

  // Set up StartTransferRequest with schedules
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(112233);
  request.set_is_sender(true);

  // Schedule:
  // Sender (256B block 0) has shard entry:
  //   - Pushes src_block_id=0, src_offset_bytes=64, size_bytes=128
  //   - To dst_shard_idx=0, dst_block_id=0, dst_offset_bytes=192 on receiver
  auto* schedules = request.mutable_shard_push_schedules();
  auto* src_schedule = &(*schedules)[0];  // shard 0
  auto* entry = src_schedule->add_entries();
  entry->set_dst_peer("127.0.0.1:20025");
  entry->set_dst_shard_idx(0);
  entry->set_dst_offset_bytes(192);
  entry->set_src_offset_bytes(64);
  entry->set_size_bytes(128);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);

  // Register active plan on both sides
  ABSL_ASSERT_OK(
      sender.RegisterActivePlan(112233, request, /*is_sender=*/true));

  // Receiver schedule should be the same request but marked as is_sender=false
  request.set_is_sender(false);
  ABSL_ASSERT_OK(
      receiver.RegisterActivePlan(112233, request, /*is_sender=*/false));

  // 3. Resolve chunks on Sender (should return offset 64 from block 0 base)
  std::vector<int64_t> src_block_ids = {0};
  std::vector<transport::BlockChunk> sender_chunks = sender.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, src_block_ids, /*total_bytes=*/128,
      /*uuid=*/112233, /*sender_node_id=*/-1, /*peer=*/"127.0.0.1:20025");

  ASSERT_EQ(sender_chunks.size(), 1);
  EXPECT_EQ(sender_chunks[0].size, 128);
  // Verify pointer offset
  uint8_t* sender_base =
      sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  EXPECT_EQ(sender_chunks[0].ptr, sender_base + 64);

  // 4. Resolve chunks on Receiver (should return offset 192 from block 0 base)
  std::vector<int64_t> dst_block_ids = {0};
  std::vector<transport::BlockChunk> receiver_chunks = receiver.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, dst_block_ids, /*total_bytes=*/128,
      /*uuid=*/112233, /*sender_node_id=*/0, /*peer=*/"127.0.0.1:20025");

  ASSERT_EQ(receiver_chunks.size(), 1);
  EXPECT_EQ(receiver_chunks[0].size, 128);
  // Verify pointer offset
  uint8_t* receiver_base =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  EXPECT_EQ(receiver_chunks[0].ptr, receiver_base + 192);
}

class TestD2hKVCacheManager : public TestKVCacheManager {
 public:
  TestD2hKVCacheManager(size_t num_layers, size_t num_shards,
                        size_t slice_byte_size, int host_blocks = 0)
      : TestKVCacheManager(num_layers, num_shards, slice_byte_size,
                           host_blocks) {}

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) override {
    d2h_called_ = true;
    last_src_offsets_ = src_offsets_major_dim;
    last_dst_offsets_ = dst_offsets_major_dim;
    last_copy_sizes_ = copy_sizes_major_dim;
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }

  absl::StatusOr<std::vector<raiden::PjRtCopyFuture>> DispatchD2hChunks(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes,
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt,
      int64_t device_id = -1) override {
    dispatch_d2h_chunks_called_ = true;
    dispatched_src_offsets_.push_back(src_offsets);
    dispatched_dst_offsets_.push_back(dst_offsets);
    dispatched_copy_sizes_.push_back(copy_sizes);
    std::vector<raiden::PjRtCopyFuture> futures;
    futures.push_back(
        raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
    return futures;
  }

  bool d2h_called_ = false;
  bool dispatch_d2h_chunks_called_ = false;
  std::vector<int64_t> last_src_offsets_;
  std::vector<int64_t> last_dst_offsets_;
  std::vector<int64_t> last_copy_sizes_;

  std::vector<std::vector<int64_t>> dispatched_src_offsets_;
  std::vector<std::vector<int64_t>> dispatched_dst_offsets_;
  std::vector<std::vector<int64_t>> dispatched_copy_sizes_;
};

class TestH2dKVCacheManager : public TestKVCacheManager {
 public:
  TestH2dKVCacheManager(size_t num_layers, size_t num_shards,
                        size_t slice_byte_size, int host_blocks = 0)
      : TestKVCacheManager(num_layers, num_shards, slice_byte_size,
                           host_blocks) {}

  absl::StatusOr<raiden::PjRtCopyFuture> H2dSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) override {
    h2d_called_ = true;
    last_h2d_src_offsets_ = src_offsets_major_dim;
    last_h2d_dst_offsets_ = dst_offsets_major_dim;
    last_h2d_copy_sizes_ = copy_sizes_major_dim;
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }

  bool h2d_called_ = false;
  std::vector<int64_t> last_h2d_src_offsets_;
  std::vector<int64_t> last_h2d_dst_offsets_;
  std::vector<int64_t> last_h2d_copy_sizes_;
};

TEST(KVCacheManagerTest, D2hWriteFailsWithCpuOnlyManager) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  auto res = manager.D2hWrite("127.0.0.1:8080", /*src_device=*/{0},
                              /*src_host=*/{0}, /*dst_host=*/{0}, {1});
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(res.status().message(),
              testing::HasSubstr("requires a device-backed"));
}

TEST(KVCacheManagerTest, D2hWriteFailsWithInvalidHostBlockId) {
  TestD2hKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                                /*slice_byte_size=*/128, /*host_blocks=*/2);
  auto res = manager.D2hWrite("127.0.0.1:8080", /*src_device=*/{0},
                              /*src_host=*/{-1}, /*dst_host=*/{0}, {1});
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(res.status().message(),
              testing::HasSubstr("Invalid host block ID"));
}

TEST(KVCacheManagerTest, D2hWriteSuccessWithMockD2h) {
  TestD2hKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  std::string receiver_peer =
      absl::StrCat(receiver.local_ip(), ":", *receiver_port);

  std::vector<int64_t> src_device_offsets = {0};
  std::vector<int64_t> src_host_offsets = {0};  // local staging (bridge)
  std::vector<int64_t> dst_host_offsets = {1};  // remote destination
  std::vector<int64_t> copy_sizes = {1};

  ABSL_ASSERT_OK(sender.D2hWrite(receiver_peer, src_device_offsets,
                                 src_host_offsets, dst_host_offsets,
                                 copy_sizes));
  EXPECT_TRUE(sender.d2h_called_);
  EXPECT_EQ(sender.last_src_offsets_, src_device_offsets);
  // The D2H stage lands in the EXPLICIT local staging blocks, not in a local
  // alias of the remote destination id.
  EXPECT_EQ(sender.last_dst_offsets_, src_host_offsets);
  EXPECT_EQ(sender.last_copy_sizes_, copy_sizes);
}

TEST(KVCacheManagerTest, D2hWritePipelinedSuccess) {
  TestD2hKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  std::string receiver_peer =
      absl::StrCat(receiver.local_ip(), ":", *receiver_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  ASSERT_NE(receiver_buf, nullptr);

  std::memset(sender_buf, 0xAB, 128);
  std::memset(sender_buf + 128, 0xCD, 128);
  std::memset(receiver_buf, 0, 256);

  std::vector<int64_t> src_device_offsets = {0, 1};
  std::vector<int64_t> src_host_offsets = {0, 1};  // local staging (bridge)
  std::vector<int64_t> dst_host_offsets = {0, 1};  // remote destination
  std::vector<int64_t> copy_sizes = {1, 1};

  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      sender.D2hWrite(receiver_peer, src_device_offsets, src_host_offsets,
                      dst_host_offsets, copy_sizes));
  ABSL_EXPECT_OK(res.Await());

  EXPECT_TRUE(sender.dispatch_d2h_chunks_called_);
  EXPECT_EQ(sender.dispatched_src_offsets_.size(), 2);
  EXPECT_EQ(sender.dispatched_src_offsets_[0], std::vector<int64_t>{0});
  EXPECT_EQ(sender.dispatched_src_offsets_[1], std::vector<int64_t>{1});

  EXPECT_TRUE(std::all_of(receiver_buf, receiver_buf + 128,
                          [](uint8_t v) { return v == 0xAB; }));
  EXPECT_TRUE(std::all_of(receiver_buf + 128, receiver_buf + 256,
                          [](uint8_t v) { return v == 0xCD; }));
}

TEST(KVCacheManagerTest, H2dReadSuccess) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> sender_port = sender.local_port();
  ASSERT_TRUE(sender_port.has_value());
  std::string sender_peer = absl::StrCat(sender.local_ip(), ":", *sender_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  std::memset(sender_buf, 0xEF, 128);

  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(receiver_buf, nullptr);
  std::memset(receiver_buf, 0, 256);

  // Test empty src_offsets returns OK empty future
  TF_ASSERT_OK_AND_ASSIGN(auto empty_res,
                          receiver.H2dRead(sender_peer, {}, {}, {}, {}));
  ABSL_EXPECT_OK(empty_res.Await());

  // Test H2dRead reading sender block 0 via local staging block 0 into
  // receiver device block 0.
  TF_ASSERT_OK_AND_ASSIGN(
      auto res, receiver.H2dRead(sender_peer, /*src_host=*/{0},
                                 /*dst_host(staging)=*/{0}, /*dst_device=*/{0},
                                 /*copy_sizes=*/{1}));
  ABSL_EXPECT_OK(res.Await());

  EXPECT_TRUE(std::all_of(receiver_buf, receiver_buf + 128,
                          [](uint8_t v) { return v == 0xEF; }));
}

TEST(KVCacheManagerTest, H2dReadPipelinedSuccess) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> sender_port = sender.local_port();
  ASSERT_TRUE(sender_port.has_value());
  std::string sender_peer = absl::StrCat(sender.local_ip(), ":", *sender_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  ASSERT_NE(receiver_buf, nullptr);

  std::memset(sender_buf, 0x55, 128);
  std::memset(sender_buf + 128, 0x66, 128);
  std::memset(receiver_buf, 0, 256);

  std::vector<int64_t> src_host_offsets = {0, 1};
  std::vector<int64_t> dst_host_offsets = {0, 1};  // local staging (bridge)
  std::vector<int64_t> dst_device_offsets = {0, 1};
  std::vector<int64_t> copy_sizes = {1, 1};

  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      receiver.H2dRead(sender_peer, src_host_offsets, dst_host_offsets,
                       dst_device_offsets, copy_sizes));
  ABSL_EXPECT_OK(res.Await());

  EXPECT_TRUE(std::all_of(receiver_buf, receiver_buf + 128,
                          [](uint8_t v) { return v == 0x55; }));
  EXPECT_TRUE(std::all_of(receiver_buf + 128, receiver_buf + 256,
                          [](uint8_t v) { return v == 0x66; }));
}

TEST(KVCacheManagerTest, H2dReadCallsH2dForTpuHbmDestination) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestH2dKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                                 /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> sender_port = sender.local_port();
  ASSERT_TRUE(sender_port.has_value());
  std::string sender_peer = absl::StrCat(sender.local_ip(), ":", *sender_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  std::memset(sender_buf, 0x77, 128);

  TF_ASSERT_OK_AND_ASSIGN(
      auto res, receiver.H2dRead(sender_peer, /*src_host=*/{0},
                                 /*dst_host(staging)=*/{0}, /*dst_device=*/{1},
                                 /*copy_sizes=*/{1}));
  ABSL_EXPECT_OK(res.Await());

  // H2dRead MUST trigger Stage 2 H2d DMA from the explicit staging block {0}
  // into TPU HBM destination offset {1}.
  EXPECT_TRUE(receiver.h2d_called_);
  EXPECT_EQ(receiver.last_h2d_src_offsets_, std::vector<int64_t>{0});
  EXPECT_EQ(receiver.last_h2d_dst_offsets_, std::vector<int64_t>{1});
  EXPECT_EQ(receiver.last_h2d_copy_sizes_, std::vector<int64_t>{1});
}

TEST(KVCacheManagerTest, H2dWriteSuccess) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  std::string receiver_peer =
      absl::StrCat(receiver.local_ip(), ":", *receiver_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  ASSERT_NE(receiver_buf, nullptr);

  std::memset(sender_buf, 0xCD, 128);
  std::memset(receiver_buf, 0, 256);

  std::vector<int64_t> src_host_offsets = {0};
  std::vector<int64_t> dst_host_offsets = {1};  // remote staging (bridge)
  std::vector<int64_t> dst_device_offsets = {0};
  std::vector<int64_t> copy_sizes = {1};

  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      sender.H2dWrite(receiver_peer, src_host_offsets, dst_host_offsets,
                      dst_device_offsets, copy_sizes));
  ABSL_EXPECT_OK(res.Await());

  EXPECT_TRUE(std::all_of(receiver_buf, receiver_buf + 128,
                          [](uint8_t v) { return v == 0; }));
  EXPECT_TRUE(std::all_of(receiver_buf + 128, receiver_buf + 256,
                          [](uint8_t v) { return v == 0xCD; }));
}

TEST(KVCacheManagerTest, H2dWritePipelinedSuccess) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  std::string receiver_peer =
      absl::StrCat(receiver.local_ip(), ":", *receiver_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  ASSERT_NE(receiver_buf, nullptr);

  std::memset(sender_buf, 0x33, 128);
  std::memset(sender_buf + 128, 0x44, 128);
  std::memset(receiver_buf, 0, 256);

  std::vector<int64_t> src_host_offsets = {0, 1};
  std::vector<int64_t> dst_host_offsets = {0, 1};  // remote staging (bridge)
  std::vector<int64_t> dst_device_offsets = {0, 1};
  std::vector<int64_t> copy_sizes = {1, 1};

  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      sender.H2dWrite(receiver_peer, src_host_offsets, dst_host_offsets,
                      dst_device_offsets, copy_sizes));
  ABSL_EXPECT_OK(res.Await());

  EXPECT_TRUE(std::all_of(receiver_buf, receiver_buf + 128,
                          [](uint8_t v) { return v == 0x33; }));
  EXPECT_TRUE(std::all_of(receiver_buf + 128, receiver_buf + 256,
                          [](uint8_t v) { return v == 0x44; }));
}

// Anti-clobber regression: the local staging block for H2dRead is the
// EXPLICIT dst_host block. A local block whose id happens to equal the remote
// src id must NOT be touched (the pre-fix code aliased the remote src id as
// the local staging id and destroyed that block's contents).
TEST(KVCacheManagerTest, H2dReadExplicitStagingDoesNotClobberAliasedBlock) {
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> sender_port = sender.local_port();
  ASSERT_TRUE(sender_port.has_value());
  std::string sender_peer = absl::StrCat(sender.local_ip(), ":", *sender_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  ASSERT_NE(receiver_buf, nullptr);

  std::memset(sender_buf, 0xEF, 128);
  // Sentinel in receiver's local block 0 -- same id as the REMOTE src block.
  // The pre-fix code staged into local block 0 and destroyed this.
  std::memset(receiver_buf, 0x99, 128);
  std::memset(receiver_buf + 128, 0, 128);

  TF_ASSERT_OK_AND_ASSIGN(
      auto res, receiver.H2dRead(sender_peer, /*src_host=*/{0},
                                 /*dst_host(staging)=*/{1}, /*dst_device=*/{0},
                                 /*copy_sizes=*/{1}));
  ABSL_EXPECT_OK(res.Await());

  // Data staged into the explicit staging block 1.
  EXPECT_TRUE(std::all_of(receiver_buf + 128, receiver_buf + 256,
                          [](uint8_t v) { return v == 0xEF; }));
  // The would-be-aliased local block 0 is untouched.
  EXPECT_TRUE(std::all_of(receiver_buf, receiver_buf + 128,
                          [](uint8_t v) { return v == 0x99; }));
}

// Anti-clobber regression: D2hWrite stages through the EXPLICIT src_host
// block and pushes THAT block to the peer. A local block whose id happens to
// equal the remote dst id must NOT be used (the pre-fix code staged into and
// pushed from the local alias of the remote dst id).
TEST(KVCacheManagerTest, D2hWriteExplicitStagingIsPushedNotAliasedBlock) {
  TestD2hKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  std::string receiver_peer =
      absl::StrCat(receiver.local_ip(), ":", *receiver_port);

  uint8_t* sender_buf = sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  uint8_t* receiver_buf =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(sender_buf, nullptr);
  ASSERT_NE(receiver_buf, nullptr);

  // Staging block 0 holds the payload (the mocked D2h stage is a no-op, so
  // the pre-seeded content is what gets pushed). Local block 1 -- same id as
  // the REMOTE dst block -- holds a sentinel the pre-fix code would have
  // staged into and pushed.
  std::memset(sender_buf, 0xAB, 128);
  std::memset(sender_buf + 128, 0x99, 128);
  std::memset(receiver_buf, 0, 256);

  TF_ASSERT_OK_AND_ASSIGN(
      auto res, sender.D2hWrite(receiver_peer, /*src_device=*/{0},
                                /*src_host(staging)=*/{0}, /*dst_host=*/{1},
                                /*copy_sizes=*/{1}));
  ABSL_EXPECT_OK(res.Await());

  // The peer received the STAGING block's payload, not the sentinel from the
  // sender's local block 1 (the would-be alias of the remote dst id).
  EXPECT_TRUE(std::all_of(receiver_buf + 128, receiver_buf + 256,
                          [](uint8_t v) { return v == 0xAB; }));
  // The sender's local block 1 is untouched.
  EXPECT_TRUE(std::all_of(sender_buf + 128, sender_buf + 256,
                          [](uint8_t v) { return v == 0x99; }));
}

// The 2-stage remote APIs require the explicit staging list (same length as
// the other offset lists) -- no silent alias fallback.
TEST(KVCacheManagerTest, RemoteTwoStageApisRequireExplicitStaging) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);

  auto h2d_read = manager.H2dRead("localhost:1", /*src_host=*/{0},
                                  /*dst_host(staging)=*/{}, /*dst_device=*/{0},
                                  /*copy_sizes=*/{1});
  EXPECT_EQ(h2d_read.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(h2d_read.status().message(), testing::HasSubstr("same length"));

  auto d2h_write = manager.D2hWrite("localhost:1", /*src_device=*/{0},
                                    /*src_host(staging)=*/{}, /*dst_host=*/{0},
                                    /*copy_sizes=*/{1});
  EXPECT_EQ(d2h_write.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(d2h_write.status().message(), testing::HasSubstr("same length"));

  auto h2d_write =
      manager.H2dWrite("localhost:1", /*src_host=*/{0},
                       /*dst_host(staging)=*/{}, /*dst_device=*/{0},
                       /*copy_sizes=*/{1});
  EXPECT_EQ(h2d_write.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(h2d_write.status().message(), testing::HasSubstr("same length"));
}

class TestBackgroundKVCacheManager : public TestKVCacheManager {
 public:
  TestBackgroundKVCacheManager(size_t num_layers, size_t num_shards,
                               size_t slice_byte_size, int host_blocks = 0)
      : TestKVCacheManager(num_layers, num_shards, slice_byte_size,
                           host_blocks) {}

  absl::StatusOr<raiden::PjRtCopyFuture> H2dSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    absl::MutexLock lock(mu_);
    execution_order_.push_back("H2dSyncDispatch");
    h2d_count_++;
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    absl::MutexLock lock(mu_);
    execution_order_.push_back("D2hSyncDispatch");
    d2h_count_++;
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }

  absl::Mutex mu_;
  std::vector<std::string> execution_order_ ABSL_GUARDED_BY(mu_);
  int h2d_count_ ABSL_GUARDED_BY(mu_) = 0;
  int d2h_count_ ABSL_GUARDED_BY(mu_) = 0;
};

TEST(KVCacheManagerTest, BackgroundWorkerThreadExecutesInFifoOrder) {
  setenv("RAIDEN_ENABLE_ASYNC_DISPATCH", "1", 1);
  TestBackgroundKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                                       /*slice_byte_size=*/128,
                                       /*host_blocks=*/2);
  // Queue H2D, D2H, H2D sequentially
  TF_ASSERT_OK_AND_ASSIGN(auto f1, manager.H2d({0}, {0}, {1}));
  TF_ASSERT_OK_AND_ASSIGN(auto f2, manager.D2h({0}, {0}, {1}));
  TF_ASSERT_OK_AND_ASSIGN(auto f3, manager.H2d({0}, {0}, {1}));

  // Await all futures
  ABSL_EXPECT_OK(f1.Await());
  ABSL_EXPECT_OK(f2.Await());
  ABSL_EXPECT_OK(f3.Await());

  absl::MutexLock lock(manager.mu_);
  EXPECT_EQ(manager.h2d_count_, 2);
  EXPECT_EQ(manager.d2h_count_, 1);
  ASSERT_EQ(manager.execution_order_.size(), 3);
  EXPECT_EQ(manager.execution_order_[0], "H2dSyncDispatch");
  EXPECT_EQ(manager.execution_order_[1], "D2hSyncDispatch");
  EXPECT_EQ(manager.execution_order_[2], "H2dSyncDispatch");
  unsetenv("RAIDEN_ENABLE_ASYNC_DISPATCH");
}

TEST(KVCacheManagerTest, BackgroundWorkerThreadDisabledByDefault) {
  unsetenv("RAIDEN_ENABLE_ASYNC_DISPATCH");
  TestBackgroundKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                                       /*slice_byte_size=*/128,
                                       /*host_blocks=*/2);
  ABSL_ASSERT_OK(manager.H2d({0}, {0}, {1}));
  absl::MutexLock lock(manager.mu_);
  EXPECT_EQ(manager.h2d_count_, 1);
}

TEST(KVCacheManagerTest, TelemetryMetricsObservedWhenEnabled) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  auto* raw_backend = mock_backend.get();

  EXPECT_CALL(
      *raw_backend,
      ObserveHistogram(testing::Eq(telemetry::metric_names::kH2dTransferTimeMs),
                       testing::_, testing::Ge(0.0)))
      .Times(testing::AtLeast(1));
  EXPECT_CALL(
      *raw_backend,
      ObserveHistogram(testing::Eq(telemetry::metric_names::kD2hTransferTimeMs),
                       testing::_, testing::Ge(0.0)))
      .Times(testing::AtLeast(1));

  telemetry::ScopedMetricsBackendReset scoped_reset(std::move(mock_backend));

  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  std::vector<int64_t> offsets = {0};
  std::vector<int64_t> sizes = {1};

  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture h2d_res,
                          manager.H2d(offsets, offsets, sizes));
  ABSL_EXPECT_OK(h2d_res.Await());

  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture d2h_res,
                          manager.D2h(offsets, offsets, sizes));
  ABSL_EXPECT_OK(d2h_res.Await());
}

TEST(KVCacheManagerTest, TelemetryMetricsSkippedWhenDisabled) {
  telemetry::ScopedMetricsBackendReset scoped_reset;
  telemetry::RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  EXPECT_FALSE(
      telemetry::RaidenMetricStore::GetGlobalMetricStore().HasBackends());

  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  std::vector<int64_t> offsets = {0};
  std::vector<int64_t> sizes = {1};

  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture h2d_res,
                          manager.H2d(offsets, offsets, sizes));
  ABSL_EXPECT_OK(h2d_res.Await());

  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture d2h_res,
                          manager.D2h(offsets, offsets, sizes));
  ABSL_EXPECT_OK(d2h_res.Await());
}

TEST(KVCacheManagerTest, D2hWritePipelinedTelemetryBatchObservation) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  auto* raw_backend = mock_backend.get();

  // Exactly 1 observation for the entire batch of chunks, not 1 per chunk.
  EXPECT_CALL(
      *raw_backend,
      ObserveHistogram(testing::Eq(telemetry::metric_names::kD2hTransferTimeMs),
                       testing::_, testing::Ge(0.0)))
      .Times(1);

  telemetry::ScopedMetricsBackendReset scoped_reset(std::move(mock_backend));

  TestD2hKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/128, /*host_blocks=*/2);
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/128, /*host_blocks=*/2);

  const std::optional<int> receiver_port = receiver.local_port();
  ASSERT_TRUE(receiver_port.has_value());
  std::string receiver_peer =
      absl::StrCat(receiver.local_ip(), ":", *receiver_port);

  std::vector<int64_t> src_device_offsets = {0, 1};
  std::vector<int64_t> src_host_offsets = {0, 1};
  std::vector<int64_t> dst_host_offsets = {0, 1};
  std::vector<int64_t> copy_sizes = {1, 1};

  TF_ASSERT_OK_AND_ASSIGN(
      raiden::PjRtCopyFuture res,
      sender.D2hWrite(receiver_peer, src_device_offsets, src_host_offsets,
                      dst_host_offsets, copy_sizes));
  ABSL_EXPECT_OK(res.Await());
}

TEST(KVCacheManagerTest, BufferAllocatedHostDramTelemetry) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  telemetry::ScopedMetricsBackendReset scoped_reset(std::move(mock_backend));

  // Expect initial gauge sets in constructor:
  // num_layers = 2, num_shards = 2, slice_byte_size = 128, host_blocks = 4
  // total allocated host dram = 2 * 2 * (4 * 128) = 2048 bytes
  EXPECT_CALL(
      *raw_mock,
      SetGauge(testing::Eq(telemetry::metric_names::kBufferAllocatedBytes),
               testing::IsEmpty(), testing::DoubleEq(2048.0)))
      .Times(testing::AtLeast(1));

  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/2,
                             /*slice_byte_size=*/128,
                             /*host_blocks=*/4);

  EXPECT_EQ(manager.GetAllocatedHostDramBytes(), 2048);
}

TEST(KVCacheManagerTest, BufferAllocatedHostDramMaintenance) {
  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  telemetry::ScopedMetricsBackendReset scoped_reset(std::move(mock_backend));

  // Initial gauge set in constructor (2 * 2 * 4 * 128 = 2048 bytes).
  EXPECT_CALL(
      *raw_mock,
      SetGauge(testing::Eq(telemetry::metric_names::kBufferAllocatedBytes),
               testing::IsEmpty(), testing::DoubleEq(2048.0)))
      .Times(1);

  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/2,
                             /*slice_byte_size=*/128,
                             /*host_blocks=*/4);

  EXPECT_EQ(manager.GetAllocatedHostDramBytes(), 2048);

  // Calling RegisterPools grows layer 0 host mirror from 4*128 to 8*128 bytes
  // per shard (2 shards = +1024 bytes -> 3072 bytes total).
  EXPECT_CALL(
      *raw_mock,
      SetGauge(testing::Eq(telemetry::metric_names::kBufferAllocatedBytes),
               testing::IsEmpty(), testing::DoubleEq(3072.0)))
      .Times(1);
  manager.SetLayerPhysicalSizeForTest(/*layer_idx=*/0,
                                      /*physical_size=*/1024,
                                      /*major_dim_size=*/1);
  ABSL_ASSERT_OK(manager.RegisterPools(
      {DensePool("kind_a", /*storage_index=*/0, /*base_offset=*/0,
                 /*stride=*/128, /*num_blocks=*/8)}));
  EXPECT_EQ(manager.GetAllocatedHostDramBytes(), 3072);

  // Subsequent call to UpdateAllocatedOccupancyMetric maintains and reports
  // the running value (3072).
  EXPECT_CALL(
      *raw_mock,
      SetGauge(testing::Eq(telemetry::metric_names::kBufferAllocatedBytes),
               testing::IsEmpty(), testing::DoubleEq(3072.0)))
      .Times(1);
  manager.UpdateAllocatedOccupancyMetric();
}

// ===========================================================================
// Pipelined Storage Offload & Recall Engine
// ===========================================================================

TEST(KVCacheManagerTest, MultiSliceOffloadAndRecallContentValidity) {
  const std::string scratch_dir =
      absl::StrCat(testing::TempDir(), "/multi_rank_e2e_", getpid(), "_",
                   std::chrono::system_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(scratch_dir);

  auto cleanup = absl::MakeCleanup([&scratch_dir]() {
    std::error_code ec;
    std::filesystem::remove_all(scratch_dir, ec);
  });

  const int kTpSize = 2;
  const size_t kBlockSize = 128;
  const int kNumBlocks = 4;
  const size_t kNumLayers = 2;

  LOG(INFO)
      << "[E2E Test] Starting MultiSliceOffloadAndRecallContentValidity with "
      << "kTpSize=" << kTpSize << ", kNumLayers=" << kNumLayers
      << ", kNumBlocks=" << kNumBlocks << ", kBlockSize=" << kBlockSize
      << ", scratch_dir=" << scratch_dir;

  // Realistic 32-byte cryptographic token prefix hashes (SHA-256 digests).
  const std::vector<std::string> kRealisticBlockHashes = {
      absl::HexStringToBytes(
          "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"),
      absl::HexStringToBytes(
          "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"),
      absl::HexStringToBytes(
          "9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca7"),
      absl::HexStringToBytes(
          "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"),
  };
  ASSERT_EQ(kRealisticBlockHashes.size(), kNumBlocks);

  // 1. Initialize real PJRT device memory environment.
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  // Rank-specific resources for TP=2 (Rank 0 and Rank 1).
  struct WorkerRankEnv {
    int rank = 0;
    std::shared_ptr<backends::storage::PosixKVBackend> backend;
    std::unique_ptr<backends::storage::PosixPathMapper> mapper;
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> pjrt_buffers;
    std::vector<std::vector<raiden::RaidenBufferHandle>> layer_handles;
    std::unique_ptr<KVCacheManagerBase> manager;
    std::vector<backends::BlockKey> block_keys;
    std::vector<int64_t> dev_block_ids;
    std::vector<int64_t> host_block_ids;
  };

  std::vector<WorkerRankEnv> workers(kTpSize);
  const int64_t total_shard_bytes = kNumBlocks * kBlockSize;

  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    w.rank = r;
    w.backend = std::make_shared<backends::storage::PosixKVBackend>(
        "posix",
        absl::flat_hash_map<std::string, std::string>{
            {"tp_rank", absl::StrCat(r)}, {"tp_size", absl::StrCat(kTpSize)}});
    w.mapper = std::make_unique<backends::storage::PosixPathMapper>(
        scratch_dir, "test_model", /*tp_size=*/kTpSize, /*tp_rank=*/r);

    w.pjrt_buffers.resize(kNumLayers);
    w.layer_handles.resize(kNumLayers);

    for (size_t l = 0; l < kNumLayers; ++l) {
      w.pjrt_buffers[l].resize(1);
      w.layer_handles[l].resize(1);
      std::vector<uint8_t> init_data(total_shard_bytes, 0);
      TF_ASSERT_OK_AND_ASSIGN(
          w.pjrt_buffers[l][0],
          client->BufferFromHostBuffer(init_data.data(), xla::U8,
                                       {static_cast<int64_t>(kNumBlocks),
                                        static_cast<int64_t>(kBlockSize)},
                                       /*byte_strides=*/std::nullopt,
                                       xla::PjRtClient::HostBufferSemantics::
                                           kImmutableUntilTransferCompletes,
                                       /*on_done_with_host_buffer=*/nullptr,
                                       memory_space,
                                       /*device_layout=*/nullptr));

      TF_ASSERT_OK_AND_ASSIGN(
          w.layer_handles[l][0],
          raiden::RaidenBufferHandle::Acquire(w.pjrt_buffers[l][0].get()));
    }

    w.manager = std::make_unique<KVCacheManagerBase>(
        w.layer_handles, /*local_port=*/std::nullopt,
        /*host_blocks_to_allocate=*/kNumBlocks,
        /*unsafe_skip_buffer_lock=*/true, /*parallelism=*/1,
        /*host_allocator=*/nullptr, /*bind_ip=*/std::nullopt,
        /*logical_slice_byte_size=*/kBlockSize);

    w.block_keys.reserve(kNumBlocks);
    w.dev_block_ids.reserve(kNumBlocks);
    w.host_block_ids.reserve(kNumBlocks);
    for (int b = 0; b < kNumBlocks; ++b) {
      TF_ASSERT_OK_AND_ASSIGN(
          backends::BlockKey key,
          w.mapper->MapKey(kRealisticBlockHashes[b],
                           {.parallelism = {.tp_rank = r}}));
      LOG(INFO)
          << "[Setup] Rank " << r << " Block " << b << " (hash_hex="
          << absl::BytesToHexString(kRealisticBlockHashes[b]).substr(0, 16)
          << "...)"
          << " mapped to path: " << key.resolved_key;
      w.block_keys.push_back(std::move(key));
      w.dev_block_ids.push_back(b);
      w.host_block_ids.push_back(b);
    }
  }

  // Unique data pattern function parameterized by (layer, rank/shard, block):
  auto get_pattern = [](size_t l, int r, int b) -> uint8_t {
    return static_cast<uint8_t>(0x40 * l + 0x10 * r + b * 2 + 1);
  };

  // Populate device buffers with known patterns for both ranks:
  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    for (size_t l = 0; l < kNumLayers; ++l) {
      for (int b = 0; b < kNumBlocks; ++b) {
        std::vector<uint8_t> block_data(kBlockSize, get_pattern(l, r, b));
        auto fut = w.layer_handles[l][0].CopyRawHostToDevice(
            block_data.data(), b * kBlockSize, kBlockSize);
        ASSERT_OK(fut.Await());
      }
    }
  }

  // 2. OFFLOAD: Device HBM -> Host Staging DRAM -> Posix Storage File for ALL
  // ranks.
  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    LOG(INFO) << "[Offload] Initiating D2hWriteToBackend for Rank " << r << " ("
              << w.block_keys.size() << " blocks)...";
    auto d2h_or = w.manager->D2hWriteToBackend(
        {w.backend}, w.block_keys, w.dev_block_ids, w.host_block_ids);
    ASSERT_TRUE(d2h_or.ok()) << d2h_or.status().ToString();
    ASSERT_OK(d2h_or->Await());
    LOG(INFO) << "[Offload] Completed D2hWriteToBackend for Rank " << r;
  }

  // 3. VERIFY ON DISK: Assert files exist under each rank's namespace and
  // verify bytes.
  const size_t bytes_per_rank_file = kNumLayers * kBlockSize;
  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    for (int b = 0; b < kNumBlocks; ++b) {
      const std::string& filepath = w.block_keys[b].resolved_key;
      // Explicitly verify path structure contains tp2_r<r>
      EXPECT_THAT(filepath, testing::HasSubstr(absl::StrCat("tp2_r", r)));

      std::ifstream file(filepath, std::ios::binary);
      ASSERT_TRUE(file.is_open())
          << "File not found for Rank " << r << ": " << filepath;
      std::vector<uint8_t> file_data(bytes_per_rank_file);
      file.read(reinterpret_cast<char*>(file_data.data()), bytes_per_rank_file);
      EXPECT_EQ(file.gcount(), bytes_per_rank_file);

      for (size_t l = 0; l < kNumLayers; ++l) {
        size_t offset = l * kBlockSize;
        uint8_t expected_val = get_pattern(l, r, b);
        EXPECT_TRUE(std::all_of(file_data.begin() + offset,
                                file_data.begin() + offset + kBlockSize,
                                [&](uint8_t v) { return v == expected_val; }))
            << "Disk mismatch: rank " << r << ", block " << b << ", layer "
            << l;
      }
      LOG(INFO) << "[Disk Audit] Verified Rank " << r << " Block " << b
                << " at " << filepath << " (bytes: " << file_data.size()
                << ", pattern OK)";
    }
  }

  // 4. ZERO OUT DEVICE & HOST MEMORY across ALL ranks:
  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    for (size_t l = 0; l < kNumLayers; ++l) {
      for (int b = 0; b < kNumBlocks; ++b) {
        std::vector<uint8_t> zeros(kBlockSize, 0x00);
        auto fut = w.layer_handles[l][0].CopyRawHostToDevice(
            zeros.data(), b * kBlockSize, kBlockSize);
        ASSERT_OK(fut.Await());
        // Rank r explicitly addresses Shard r in host memory
        std::memset(w.manager->GetBlockHostPointer(l, /*shard_idx=*/r, b), 0x00,
                    kBlockSize);
      }
      TF_ASSERT_OK_AND_ASSIGN(auto literal,
                              w.pjrt_buffers[l][0]->ToLiteral().Await());
      for (uint8_t val : literal->data<uint8_t>()) {
        ASSERT_EQ(val, 0x00);
      }
    }
    LOG(INFO) << "[Zero Memory] Zeroed device HBM and host DRAM for Rank " << r;
  }

  // 5. RECALL: Posix Storage File -> Host Staging DRAM -> Device HBM for ALL
  // ranks.
  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    LOG(INFO) << "[Recall] Initiating H2dReadFromBackend for Rank " << r << " ("
              << w.block_keys.size() << " blocks)...";
    auto h2d_or = w.manager->H2dReadFromBackend(
        {w.backend}, w.block_keys, w.host_block_ids, w.dev_block_ids);
    ASSERT_TRUE(h2d_or.ok()) << h2d_or.status().ToString();
    ASSERT_OK(h2d_or->Await());
    LOG(INFO) << "[Recall] Completed H2dReadFromBackend for Rank " << r;
  }

  // 6. VERIFY RECALLED CONTENTS ON DEVICE & HOST for ALL ranks:
  for (int r = 0; r < kTpSize; ++r) {
    auto& w = workers[r];
    for (size_t l = 0; l < kNumLayers; ++l) {
      TF_ASSERT_OK_AND_ASSIGN(auto literal,
                              w.pjrt_buffers[l][0]->ToLiteral().Await());
      auto dev_data = literal->data<uint8_t>();
      for (int b = 0; b < kNumBlocks; ++b) {
        // Rank r explicitly addresses Shard r in host memory
        uint8_t* host_ptr =
            w.manager->GetBlockHostPointer(l, /*shard_idx=*/r, b);
        uint8_t expected_val = get_pattern(l, r, b);
        size_t block_offset = b * kBlockSize;

        EXPECT_TRUE(std::all_of(host_ptr, host_ptr + kBlockSize,
                                [&](uint8_t v) { return v == expected_val; }))
            << "Host staging mismatch: rank " << r << ", block " << b
            << ", layer " << l;

        EXPECT_TRUE(std::all_of(dev_data.begin() + block_offset,
                                dev_data.begin() + block_offset + kBlockSize,
                                [&](uint8_t v) { return v == expected_val; }))
            << "Device HBM mismatch: rank " << r << ", block " << b
            << ", layer " << l;
      }
    }
    LOG(INFO)
        << "[Verification] Rank " << r
        << " verified bit-for-bit in both Host DRAM and Device HBM across all "
        << kNumLayers << " layers and " << kNumBlocks << " blocks.";
  }
}

TEST(KVCacheManagerTest, RegisterKVBackendsPropagatesParallelismTpSize) {
  std::string scratch_dir =
      absl::StrCat(testing::TempDir(), "/test_manager_tp_size_reg_", getpid());
  std::filesystem::create_directories(scratch_dir);
  auto cleanup = absl::MakeCleanup([&scratch_dir]() {
    std::error_code ec;
    std::filesystem::remove_all(scratch_dir, ec);
  });

  KVCacheManagerBase manager(/*buffers=*/{}, /*local_port=*/std::nullopt,
                             /*host_blocks_to_allocate=*/4,
                             /*unsafe_skip_buffer_lock=*/true,
                             /*parallelism=*/1, /*host_allocator=*/nullptr,
                             /*bind_ip=*/std::nullopt,
                             /*logical_slice_byte_size=*/512);

  BackendConfig cfg;
  cfg.type = "posix";
  cfg.parallelism.tp_rank = 2;
  cfg.parallelism.tp_size = 4;
  cfg.SetProperty("root_dir", scratch_dir);
  cfg.SetProperty("model_name", "test_model_tp");

  manager.RegisterKVBackends({cfg});

  auto backend = manager.GetKVBackend("posix");
  ASSERT_NE(backend, nullptr);
  auto mapper = backend->mapper();
  ASSERT_NE(mapper, nullptr);
  EXPECT_EQ(mapper->tp_size(), 4);

  // Verify that MapKey resolves to tp4_r2 without options override
  TF_ASSERT_OK_AND_ASSIGN(auto key, mapper->MapKey("block_hash_1"));
  EXPECT_THAT(key.resolved_key, ::testing::HasSubstr("/tp4_r2/"));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

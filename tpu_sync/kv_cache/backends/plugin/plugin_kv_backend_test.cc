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

#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_backend.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/mock_storage_plugin_loader.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// Runs PluginKVBackend on the function table of the mock plugin.
class PluginKVBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    vtable_ = LoadMockPlugin();
    ASSERT_NE(vtable_, nullptr);
  }

  static HostBufferDescriptor SliceOf(std::string& bytes) {
    return HostBufferDescriptor{.ptr = bytes.data(), .size = bytes.size()};
  }

  static size_t TotalBytes(absl::Span<const HostBufferDescriptor> slices) {
    size_t total = 0;
    for (const HostBufferDescriptor& slice : slices) total += slice.size;
    return total;
  }

  // Writes `slices` under `key` and waits for the answer of the plugin.
  static absl::Status Write(KVBackend& backend, const BlockKey& key,
                            absl::Span<const HostBufferDescriptor> slices) {
    absl::Status status;
    absl::Notification done;
    backend.WriteAsync(key, slices, TotalBytes(slices),
                       [&](absl::Status write_status) {
                         status = std::move(write_status);
                         done.Notify();
                       });
    done.WaitForNotification();
    return status;
  }

  // Reads `key` into `slices` and waits for the answer of the plugin.
  static absl::Status Read(KVBackend& backend, const BlockKey& key,
                           absl::Span<const HostBufferDescriptor> slices) {
    absl::Status status;
    absl::Notification done;
    backend.ReadAsync(key, slices, TotalBytes(slices),
                      [&](absl::Status read_status) {
                        status = std::move(read_status);
                        done.Notify();
                      });
    done.WaitForNotification();
    return status;
  }

  // Probes `keys` and waits for the answer of the plugin.
  static std::vector<absl::StatusOr<bool>> BatchExists(
      KVBackend& backend, absl::Span<const BlockKey> keys) {
    std::vector<absl::StatusOr<bool>> answers;
    absl::Notification done;
    backend.BatchExistsAsync(keys,
                             [&](std::vector<absl::StatusOr<bool>> result) {
                               answers = std::move(result);
                               done.Notify();
                             });
    done.WaitForNotification();
    return answers;
  }

  const RaidenStorageOutofTreePluginV1* vtable_ = nullptr;
};

TEST_F(PluginKVBackendTest, WritesReadsAndProbesBlocks) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<PluginKVBackend> backend,
      PluginKVBackend::Create(
          vtable_,
          {{"prefix", "round_trip"}, {"clear_store_on_create", "true"}},
          {.tp_size = 1, .tp_rank = 0}));
  EXPECT_EQ(backend->name(), "mock_plugin");
  EXPECT_TRUE(backend->HasMapKey());

  TF_ASSERT_OK_AND_ASSIGN(BlockKey stored, backend->mapper()->MapKey("stored"));
  TF_ASSERT_OK_AND_ASSIGN(BlockKey absent, backend->mapper()->MapKey("absent"));

  // A block arrives as several host slices, one per layer.
  std::string first = "first_layer_";
  std::string second = "second_layer";
  ABSL_ASSERT_OK(Write(*backend, stored, {SliceOf(first), SliceOf(second)}));

  EXPECT_THAT(BatchExists(*backend, {stored, absent}),
              ElementsAre(IsOkAndHolds(true), IsOkAndHolds(false)));

  std::string out_first(first.size(), '\0');
  std::string out_second(second.size(), '\0');
  ABSL_ASSERT_OK(
      Read(*backend, stored, {SliceOf(out_first), SliceOf(out_second)}));
  EXPECT_EQ(out_first, first);
  EXPECT_EQ(out_second, second);

  // RAIDEN_STATUS_NOT_FOUND from the plugin arrives as absl's kNotFound.
  EXPECT_THAT(Read(*backend, absent, {SliceOf(out_first)}),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(PluginKVBackendTest, MapKeyUsesTheInstanceRankUnlessTheCallNamesOne) {
  auto create_rank = [this](int tp_rank) {
    return PluginKVBackend::Create(vtable_, {{"prefix", "ranks"}},
                                   {.tp_size = 2, .tp_rank = tp_rank});
  };
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PluginKVBackend> rank0, create_rank(0));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PluginKVBackend> rank1, create_rank(1));
  EXPECT_EQ(rank0->mapper()->tp_size(), 2);

  // The mock names a shard after the hex block hash and the (pp, pcp, tp)
  // ranks. It prints -1 for an axis that is not declared.
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key0, rank0->mapper()->MapKey("h"));
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key1, rank1->mapper()->MapKey("h"));
  EXPECT_EQ(key0.block_hash, "h");
  EXPECT_EQ(key0.resolved_key, "ranks/68_pp_-1_pcp_-1_tp_0");
  EXPECT_EQ(key1.resolved_key, "ranks/68_pp_-1_pcp_-1_tp_1");

  // The coordinator has one instance and maps the key of every rank with it.
  KeyMappingOptions as_rank1;
  as_rank1.parallelism.tp_size = 2;
  as_rank1.parallelism.tp_rank = 1;
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key1_from_rank0,
                       rank0->mapper()->MapKey("h", as_rank1));
  EXPECT_EQ(key1_from_rank0.resolved_key, key1.resolved_key);

  EXPECT_THAT(rank0->mapper()->MapKey(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PluginKVBackendTest, UsesTheRawHashAsKeyWithoutMapKey) {
  RaidenStorageOutofTreePluginV1 block_level_vtable = *vtable_;
  block_level_vtable.map_key = nullptr;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PluginKVBackend> backend,
                       PluginKVBackend::Create(&block_level_vtable, {},
                                               {.tp_size = 4, .tp_rank = 2}));
  EXPECT_FALSE(backend->HasMapKey());

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, backend->mapper()->MapKey("raw_hash"));
  EXPECT_EQ(key.block_hash, "raw_hash");
  EXPECT_EQ(key.resolved_key, "raw_hash");
}

TEST_F(PluginKVBackendTest, ReturnsTheErrorOfAFailedCreate) {
  EXPECT_THAT(PluginKVBackend::Create(vtable_, {{"fail_create", "true"}},
                                      {.tp_size = 1, .tp_rank = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("simulated create failure")));
  EXPECT_THAT(PluginKVBackend::Create(nullptr, {}, {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PluginKVBackendTest, ShmArenaRegistrationAndZeroCopyPreadWrite) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<PluginKVBackend> worker,
      PluginKVBackend::Create(vtable_,
                              {{"prefix", "shm_test"},
                               {"use_pread_on_write", "true"},
                               {"clear_store_on_create", "true"}},
                              {.tp_size = 1, .tp_rank = 0}));

  const std::string shm_key = absl::StrCat("raiden_test_plugin_shm_", getpid());
  const std::string seg_name =
      SharedMemoryHostMemoryAllocator::ComposeSegmentName(shm_key,
                                                          std::nullopt);
  shm_unlink(seg_name.c_str());

  SharedMemoryHeader schema{};
  std::snprintf(schema.model_uid, sizeof(schema.model_uid), "plugin_shm_model");
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SharedMemoryHostMemoryAllocator> shm_allocator,
      SharedMemoryHostMemoryAllocator::Create(/*client=*/nullptr, shm_key,
                                              schema));
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                       shm_allocator->Allocate(8192));
  ASSERT_GE(alloc.fd, 0);

  HostBufferDescriptor arena_region{alloc.ptr, alloc.size, alloc.fd,
                                    alloc.fd_offset};
  ABSL_ASSERT_OK(worker->OnHostMemoryInitialized({arena_region, arena_region}));

  auto* bytes = static_cast<uint8_t*>(alloc.ptr);
  const std::string slice_a = "first_layer_slice_in_shm_";
  const std::string slice_b = "second_layer_slice_in_shm";
  std::memcpy(bytes + 128, slice_a.data(), slice_a.size());
  std::memcpy(bytes + 1024, slice_b.data(), slice_b.size());

  std::vector<HostBufferDescriptor> write_slices = {
      HostBufferDescriptor{bytes + 128, slice_a.size(), alloc.fd,
                           alloc.fd_offset + 128},
      HostBufferDescriptor{bytes + 1024, slice_b.size(), alloc.fd,
                           alloc.fd_offset + 1024},
  };

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key,
                       worker->mapper()->MapKey("shm_block_hash"));
  ABSL_ASSERT_OK(Write(*worker, key, write_slices));

  std::string out_a(slice_a.size(), '\0');
  std::string out_b(slice_b.size(), '\0');
  ABSL_ASSERT_OK(Read(*worker, key, {SliceOf(out_a), SliceOf(out_b)}));
  EXPECT_EQ(out_a, slice_a);
  EXPECT_EQ(out_b, slice_b);
  shm_unlink(seg_name.c_str());
}

}  // namespace
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

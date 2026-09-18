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

#include "tpu_sync/kv_cache/backends/backend.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace {

using ::absl_testing::StatusIs;
using ::tpu_raiden::kv_cache::backends::storage::PosixPathMapper;

TEST(HostBufferDescriptorTest, DefaultValues) {
  HostBufferDescriptor desc;
  EXPECT_EQ(desc.ptr, nullptr);
  EXPECT_EQ(desc.size, 0);
  EXPECT_EQ(desc.fd, -1);
  EXPECT_EQ(desc.offset, 0);
}

TEST(PosixPathMapperTest, MapKeyDefaultOptions) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);
  EXPECT_EQ(mapper.tp_size(), 8);

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("abc123hash"));
  EXPECT_EQ(key.block_hash, "abc123hash");
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama_70b/tp8_r2/616/26/61626331323368617368.bin");
  EXPECT_EQ(key.offset, 0);
  EXPECT_EQ(key.size, 0);
}

TEST(PosixPathMapperTest, MapKeyCustomOptions) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);

  KeyMappingOptions options;
  options.parallelism.tp_size = 16;
  options.parallelism.tp_rank = 5;

  TF_ASSERT_OK_AND_ASSIGN(BlockKey key, mapper.MapKey("custom_hash", options));
  EXPECT_EQ(key.block_hash, "custom_hash");
  EXPECT_EQ(
      key.resolved_key,
      "/tmp/kv_cache/llama_70b/tp16_r5/637/57/637573746f6d5f68617368.bin");
  EXPECT_EQ(key.offset, 0);
  EXPECT_EQ(key.size, 0);
}

TEST(PosixPathMapperTest, MapKeyRejectsEmptyHash) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);

  EXPECT_THAT(mapper.MapKey(""), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PosixPathMapperTest, GetParentDir) {
  EXPECT_EQ(PosixPathMapper::GetParentDir("/a/b/c/d.bin"), "/a/b/c");
  EXPECT_EQ(PosixPathMapper::GetParentDir("/root_dir"), "");
  EXPECT_EQ(PosixPathMapper::GetParentDir("relative_no_slash"), "");
}

}  // namespace
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

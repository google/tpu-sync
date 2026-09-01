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

#include <stdlib.h>

#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace {

using ::tpu_raiden::kv_cache::backends::storage::PosixPathMapper;

TEST(BackendBufferDescriptorTest, DefaultValues) {
  BackendBufferDescriptor desc;
  EXPECT_EQ(desc.ptr, nullptr);
  EXPECT_EQ(desc.size, 0);
  EXPECT_EQ(desc.fd, -1);
  EXPECT_EQ(desc.offset, 0);
}

TEST(ResolveBackendPropertiesTest, BasePropertiesPrecedence) {
  setenv("RAIDEN_BACKEND_EXTRA_POSIX_IO_THREADS", "16", 1);
  setenv("RAIDEN_BACKEND_EXTRA_IO_THREADS", "8", 1);

  absl::flat_hash_map<std::string, std::string> base_properties = {
      {"io_threads", "4"},
  };

  auto resolved = ResolveBackendProperties("posix", base_properties);
  EXPECT_EQ(resolved["io_threads"], "4");

  unsetenv("RAIDEN_BACKEND_EXTRA_POSIX_IO_THREADS");
  unsetenv("RAIDEN_BACKEND_EXTRA_IO_THREADS");
}

TEST(ResolveBackendPropertiesTest, ScopedEnvOverridesGlobalEnv) {
  setenv("RAIDEN_BACKEND_EXTRA_POSIX_IO_THREADS", "16", 1);
  setenv("RAIDEN_BACKEND_EXTRA_IO_THREADS", "8", 1);
  setenv("RAIDEN_BACKEND_EXTRA_ASYNC_DEPTH", "32", 1);

  auto resolved = ResolveBackendProperties("posix", /*base_properties=*/{});
  EXPECT_EQ(resolved["io_threads"], "16");
  EXPECT_EQ(resolved["async_depth"], "32");

  unsetenv("RAIDEN_BACKEND_EXTRA_POSIX_IO_THREADS");
  unsetenv("RAIDEN_BACKEND_EXTRA_IO_THREADS");
  unsetenv("RAIDEN_BACKEND_EXTRA_ASYNC_DEPTH");
}

TEST(ResolveBackendPropertiesTest, GlobalEnvFallback) {
  setenv("RAIDEN_BACKEND_EXTRA_IO_THREADS", "8", 1);

  auto resolved = ResolveBackendProperties("posix", /*base_properties=*/{});
  EXPECT_EQ(resolved["io_threads"], "8");

  unsetenv("RAIDEN_BACKEND_EXTRA_IO_THREADS");
}

TEST(PosixPathMapperTest, MapKeyDefaultOptions) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*rank=*/2);
  EXPECT_EQ(mapper.tp_size(), 8);

  BlockKey key = mapper.MapKey("abc123hash");
  EXPECT_EQ(key.block_hash, "abc123hash");
  EXPECT_EQ(key.resolved_key, "/tmp/kv_cache/llama_70b_tp8_r2/abc123hash.bin");
  EXPECT_EQ(key.offset, 0);
  EXPECT_EQ(key.size, 0);
}

TEST(PosixPathMapperTest, MapKeyCustomOptions) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*rank=*/2);

  KeyMappingOptions options;
  options.tp_size = 16;
  options.rank = 5;

  BlockKey key = mapper.MapKey("custom_hash", options);
  EXPECT_EQ(key.block_hash, "custom_hash");
  EXPECT_EQ(key.resolved_key,
            "/tmp/kv_cache/llama_70b_tp16_r5/custom_hash.bin");
  EXPECT_EQ(key.offset, 0);
  EXPECT_EQ(key.size, 0);
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

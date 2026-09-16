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

TEST(PosixPathMapperTest, MapKeyIsUnimplementedInPart1) {
  PosixPathMapper mapper("/tmp/kv_cache", "llama_70b", /*tp_size=*/8,
                         /*tp_rank=*/2);
  EXPECT_THAT(mapper.MapKey("abc123hash"),
              StatusIs(absl::StatusCode::kUnimplemented));
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

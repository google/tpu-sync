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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer.h"

ABSL_DECLARE_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size);

namespace tpu_raiden {
namespace jax {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::SizeIs;

TEST(WeightSynchronizerTest,
     GetHostBufferWithLocalIndexReportsMissingBufferAndBadIndices) {
  absl::FlagSaver flag_saver;
  // With no scratchpad, a zero-byte layer gets no host buffer (null pointer).
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size, 0);
  constexpr size_t kNumLayers = 2;
  constexpr size_t kSliceByteSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizer> ws,
      WeightSynchronizer::test_only_create_cpu_instance(
          kNumLayers, /*num_shards=*/1,
          std::vector<size_t>{kSliceByteSize, /*no host buffer=*/0}));

  EXPECT_THAT(ws->GetHostBufferWithLocalIndex(/*layer_idx=*/0,
                                              /*local_shard_idx=*/0)
                  .Await(),
              IsOkAndHolds(SizeIs(kSliceByteSize)));
  EXPECT_THAT(ws->GetHostBufferWithLocalIndex(/*layer_idx=*/1,
                                              /*local_shard_idx=*/0)
                  .Await(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ws->GetHostBufferWithLocalIndex(/*layer_idx=*/kNumLayers,
                                              /*local_shard_idx=*/0)
                  .Await(),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(ws->GetHostBufferWithLocalIndex(/*layer_idx=*/0,
                                              /*local_shard_idx=*/1)
                  .Await(),
              StatusIs(absl::StatusCode::kOutOfRange));
}

}  // namespace
}  // namespace jax
}  // namespace tpu_raiden

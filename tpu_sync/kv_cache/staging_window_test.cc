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

#include "tpu_sync/kv_cache/staging_window.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* prev = std::getenv(name)) previous_ = prev;
    if (value == nullptr) {
      unsetenv(name);
    } else {
      setenv(name, value, 1);
    }
  }
  ~ScopedEnv() {
    if (previous_.has_value()) {
      setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

TEST(StagingWindowConfigTest, DefaultsWhenUnset) {
  TF_ASSERT_OK_AND_ASSIGN(
      const auto config,
      StagingWindowConfigFromEnv([](const char*) { return nullptr; }));

  EXPECT_EQ(config.window_chunks, 0);
  EXPECT_FALSE(config.enabled());
  EXPECT_TRUE(config.serve_windowed_pulls);
  EXPECT_EQ(config.chunk_blocks, 0);
  EXPECT_EQ(config.chunk_bytes_target, 8 << 20);
  EXPECT_EQ(config.stream_bytes_target, 4 << 20);
  EXPECT_EQ(config.admission, StagingWindowConfig::Admission::kFailFast);
  EXPECT_EQ(config.admission_timeout_ms, 0);
  EXPECT_FALSE(config.assume_peer_support);
  EXPECT_EQ(config.order, StagingWindowConfig::UnitOrder::kChunkMajor);
}

TEST(StagingWindowConfigTest, EnvOverridesDefaults) {
  const absl::flat_hash_map<std::string, std::string> env = {
      {"TPU_RAIDEN_STAGING_WINDOW_CHUNKS", "2"},
      {"TPU_RAIDEN_STAGING_CHUNK_BLOCKS", "64"},
      {"TPU_RAIDEN_STAGING_CHUNK_BYTES", "16777216"},
      {"TPU_RAIDEN_STAGING_STREAM_BYTES", "8388608"},
      {"TPU_RAIDEN_STAGING_ADMISSION", "wait"},
      {"TPU_RAIDEN_STAGING_ADMISSION_TIMEOUT_MS", "5000"},
      {"TPU_RAIDEN_STAGING_WINDOW_SERVE", "0"},
      {"TPU_RAIDEN_STAGING_WINDOW_ASSUME_PEER", "1"},
  };

  TF_ASSERT_OK_AND_ASSIGN(
      const auto config,
      StagingWindowConfigFromEnv([&](const char* name) -> const char* {
        const auto it = env.find(name);
        return it != env.end() ? it->second.c_str() : nullptr;
      }));

  EXPECT_EQ(config.window_chunks, 2);
  EXPECT_TRUE(config.enabled());
  EXPECT_EQ(config.chunk_blocks, 64);
  EXPECT_EQ(config.chunk_bytes_target, 16777216);
  EXPECT_EQ(config.stream_bytes_target, 8388608);
  EXPECT_EQ(config.admission, StagingWindowConfig::Admission::kWait);
  EXPECT_EQ(config.admission_timeout_ms, 5000);
  EXPECT_FALSE(config.serve_windowed_pulls);
  EXPECT_TRUE(config.assume_peer_support);
}

TEST(StagingWindowConfigTest, ZeroArgOverloadSmoke) {
  const auto config = StagingWindowConfigFromEnv();
  ABSL_EXPECT_OK(config);
}

TEST(StagingWindowConfigTest, EnvStringLiteralsParsed) {
  const absl::flat_hash_map<std::string, std::string> env = {
      {"TPU_RAIDEN_STAGING_WINDOW_SERVE", "true"},
      {"TPU_RAIDEN_STAGING_WINDOW_ASSUME_PEER", "false"},
      {"TPU_RAIDEN_STAGING_ADMISSION", "fail_fast"},
  };

  TF_ASSERT_OK_AND_ASSIGN(
      const auto config,
      StagingWindowConfigFromEnv([&](const char* name) -> const char* {
        const auto it = env.find(name);
        return it != env.end() ? it->second.c_str() : nullptr;
      }));

  EXPECT_TRUE(config.serve_windowed_pulls);
  EXPECT_FALSE(config.assume_peer_support);
  EXPECT_EQ(config.admission, StagingWindowConfig::Admission::kFailFast);
}

TEST(StagingWindowConfigTest, EnvInvalidValuesFail) {
  for (const std::string var : {
           "TPU_RAIDEN_STAGING_WINDOW_CHUNKS",
           "TPU_RAIDEN_STAGING_CHUNK_BLOCKS",
           "TPU_RAIDEN_STAGING_CHUNK_BYTES",
           "TPU_RAIDEN_STAGING_STREAM_BYTES",
           "TPU_RAIDEN_STAGING_ADMISSION",
           "TPU_RAIDEN_STAGING_ADMISSION_TIMEOUT_MS",
           "TPU_RAIDEN_STAGING_WINDOW_SERVE",
           "TPU_RAIDEN_STAGING_WINDOW_ASSUME_PEER",
       }) {
    EXPECT_THAT(
        StagingWindowConfigFromEnv([&](const char* name) -> const char* {
          return var == name ? "bad_val" : nullptr;
        }),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(var)));
  }
}

TEST(ValidateStagingWindowConfigTest, OffConfigAlwaysOk) {
  StagingWindowConfig config;
  config.window_chunks = 0;
  ABSL_EXPECT_OK(ValidateStagingWindowConfig(config, /*pool_blocks=*/0));
}

TEST(ValidateStagingWindowConfigTest, EnabledValidationBounds) {
  StagingWindowConfig config;
  config.window_chunks = 2;
  config.chunk_blocks = 64;

  ABSL_EXPECT_OK(ValidateStagingWindowConfig(config, /*pool_blocks=*/128));
  EXPECT_THAT(ValidateStagingWindowConfig(config, /*pool_blocks=*/127),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds pool_blocks")));

  config.window_chunks = 256;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 65536),
              StatusIs(absl::StatusCode::kInvalidArgument));

  config.window_chunks = 2;
  config.chunk_blocks = 65536;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 200000),
              StatusIs(absl::StatusCode::kInvalidArgument));

  config.chunk_blocks = 0;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds pool_blocks")));

  config.chunk_blocks = -1;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 100),
              StatusIs(absl::StatusCode::kInvalidArgument));

  config.chunk_blocks = 64;
  config.chunk_bytes_target = 0;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 1000),
              StatusIs(absl::StatusCode::kInvalidArgument));

  config.chunk_bytes_target = 8 << 20;
  config.stream_bytes_target = 0;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 1000),
              StatusIs(absl::StatusCode::kInvalidArgument));

  config.stream_bytes_target = 4 << 20;
  config.admission_timeout_ms = -1;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 1000),
              StatusIs(absl::StatusCode::kInvalidArgument));

  config.admission_timeout_ms = 0;
  config.window_chunks = -1;
  EXPECT_THAT(ValidateStagingWindowConfig(config, 1000),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateStagingWindowConfigTest, AutoChunkValidationWithBlockBytes) {
  StagingWindowConfig config;
  config.window_chunks = 2;
  config.chunk_blocks = 0;
  const int64_t bytes_per_block = 1 << 20;

  // 8 MiB target / 1 MiB per block = 8 blocks. W=2 => req = 16 blocks.
  config.chunk_bytes_target = 8 << 20;
  ABSL_EXPECT_OK(
      ValidateStagingWindowConfig(config, /*pool_blocks=*/16, bytes_per_block));
  EXPECT_THAT(
      ValidateStagingWindowConfig(config, /*pool_blocks=*/15, bytes_per_block),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds pool_blocks")));

  // Re-validate when chunk_bytes_target changes: 16 MiB target => 16 blocks.
  // W=2 => req = 32 blocks.
  config.chunk_bytes_target = 16 << 20;
  ABSL_EXPECT_OK(
      ValidateStagingWindowConfig(config, /*pool_blocks=*/32, bytes_per_block));
  EXPECT_THAT(
      ValidateStagingWindowConfig(config, /*pool_blocks=*/31, bytes_per_block),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds pool_blocks")));
}

TEST(EffectiveChunkBlocksTest, ReturnsExplicitOrAutoComputed) {
  StagingWindowConfig config;
  config.chunk_blocks = 12;
  config.chunk_bytes_target = 8 << 20;
  EXPECT_EQ(
      EffectiveChunkBlocks(config, /*bytes_per_block_all_shards=*/1 << 20), 12);

  config.chunk_blocks = 0;
  EXPECT_EQ(
      EffectiveChunkBlocks(config, /*bytes_per_block_all_shards=*/1 << 20), 8);
  EXPECT_EQ(EffectiveChunkBlocks(config, /*bytes_per_block_all_shards=*/0), 1);
}

TEST(AutoChunkBlocksTest, GeometryCalculationsAndClamping) {
  EXPECT_EQ(AutoChunkBlocks(8 << 20, 8 * (16 << 10)), 64);
  EXPECT_EQ(AutoChunkBlocks(8 << 20, 8 * (1 << 20)), 1);
  EXPECT_EQ(AutoChunkBlocks(8 << 20, 8 * (512 << 10)), 2);
  EXPECT_EQ(AutoChunkBlocks(0, 1024), 1);
  EXPECT_EQ(AutoChunkBlocks(100, 10000), 1);
  EXPECT_EQ(AutoChunkBlocks(1001, 500), 3);
  EXPECT_EQ(AutoChunkBlocks(std::numeric_limits<int64_t>::max(), 1LL << 40),
            65535);
  EXPECT_EQ(AutoChunkBlocks(1LL << 40, 1), 65535);
}

TEST(StreamsPerUnitTest, ClampsCorrectly) {
  // parallelism < floor (floor = 4, clamped to parallelism = 2)
  EXPECT_EQ(StreamsPerUnit(16 << 20, 4 << 20, /*parallelism=*/2, 8), 2);
  // C_c < parallelism (floor = 4, clamped to chunk_blocks_c = 2)
  EXPECT_EQ(StreamsPerUnit(16 << 20, 4 << 20, /*parallelism=*/8, 2), 2);
  // floor < 1 (floor = 0, clamped to 1)
  EXPECT_EQ(StreamsPerUnit(1 << 20, 4 << 20, /*parallelism=*/4, 4), 1);
  // Unclamped case: 15 MiB / 4 MiB = 3 streams (upper bound = min(8, 8) = 8).
  EXPECT_EQ(StreamsPerUnit(15 << 20, 4 << 20, /*parallelism=*/8,
                           /*chunk_blocks_c=*/8),
            3);
}

TEST(ChunkPlanTest, BasicGeometriesAndHelpers) {
  // k <= C: n=1, C_last=k
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p, BuildChunkPlan(3, 5, 2, true));
    EXPECT_EQ(p.n_chunks, 1);
    EXPECT_EQ(p.ChunkLen(0), 3);
    EXPECT_EQ(p.ChunkRange(0), (ChunkRange{0, 3}));
    EXPECT_EQ(p.ring_blocks, 3);
  }
  // k = C + 1: n=2, C_last=1
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p, BuildChunkPlan(5, 4, 2, true));
    EXPECT_EQ(p.n_chunks, 2);
    EXPECT_EQ(p.ChunkLen(0), 4);
    EXPECT_EQ(p.ChunkLen(1), 1);
    EXPECT_EQ(p.ChunkRange(0), (ChunkRange{0, 4}));
    EXPECT_EQ(p.ChunkRange(1), (ChunkRange{4, 5}));
  }
  // k mod C != 0
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p, BuildChunkPlan(10, 3, 2, true));
    EXPECT_EQ(p.n_chunks, 4);
    EXPECT_EQ(p.ChunkLen(0), 3);
    EXPECT_EQ(p.ChunkLen(3), 1);
  }
  // W = 1 with reuse (SlotOf all 0)
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p, BuildChunkPlan(10, 2, 1, true));
    for (int32_t c = 0; c < p.n_chunks; ++c) EXPECT_EQ(p.SlotOf(c), 0);
  }
  // W_eff = max(W, n) without reuse (W=2, C=2, k=7 -> n=4, W_eff=4,
  // ring_blocks=7)
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p, BuildChunkPlan(7, 2, 2, false));
    EXPECT_EQ(p.n_chunks, 4);
    EXPECT_EQ(p.effective_window, 4);
    EXPECT_EQ(p.ring_blocks, 7);
    EXPECT_EQ(p.TotalUnits(32), 32 * 4);
  }
}

TEST(ChunkPlanTest, ExpandRingAndPositions) {
  // ExpandRing mapping for (k=7, C=2, W=2, reuse)
  TF_ASSERT_OK_AND_ASSIGN(const auto p, BuildChunkPlan(7, 2, 2, true));
  EXPECT_EQ(p.ring_blocks, 4);

  const std::vector<int64_t> ring_ids = {10, 11, 20, 21};
  EXPECT_THAT(p.ExpandRing(ring_ids),
              IsOkAndHolds(ElementsAre(10, 11, 20, 21, 10, 11, 20)));

  EXPECT_THAT(p.ExpandRing({10, 11, 20}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ring_ids size mismatch")));

  EXPECT_THAT(p.ChunkOfPosition(0), IsOkAndHolds(0));
  EXPECT_THAT(p.ChunkOfPosition(1), IsOkAndHolds(0));
  EXPECT_THAT(p.ChunkOfPosition(2), IsOkAndHolds(1));
  EXPECT_THAT(p.ChunkOfPosition(6), IsOkAndHolds(3));
  EXPECT_THAT(p.ChunkOfPosition(-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(p.ChunkOfPosition(7),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ChunkPlanTest, HasDuplicates) {
  EXPECT_FALSE(ChunkPlan::HasDuplicates({}));
  EXPECT_FALSE(ChunkPlan::HasDuplicates({1, 2, 3, 4}));
  EXPECT_TRUE(ChunkPlan::HasDuplicates({1, 2, 3, 1}));
}

TEST(ChunkPlanTest, SlotOfMultiWindow) {
  // reuse_slots = true (W=2, C=2, k=7)
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p,
                            BuildChunkPlan(7, 2, 2, /*reuse_slots=*/true));
    EXPECT_EQ(p.SlotOf(0), 0);
    EXPECT_EQ(p.SlotOf(1), 1);
    EXPECT_EQ(p.SlotOf(2), 0);
    EXPECT_EQ(p.SlotOf(3), 1);
  }
  // reuse_slots = false (W=2, C=2, k=7 -> effective_window = max(2, 4) = 4)
  {
    TF_ASSERT_OK_AND_ASSIGN(const auto p,
                            BuildChunkPlan(7, 2, 2, /*reuse_slots=*/false));
    for (int32_t c = 0; c < p.n_chunks; ++c) {
      EXPECT_EQ(p.SlotOf(c), c);
    }
  }
}

TEST(ChunkPlanTest, ExpandRingSubWindow) {
  // k < W * C: k=3, C=2, W=2, reuse=true -> ring_blocks = min(3, 4) = 3
  TF_ASSERT_OK_AND_ASSIGN(const auto p,
                          BuildChunkPlan(3, 2, 2, /*reuse_slots=*/true));
  EXPECT_EQ(p.ring_blocks, 3);
  EXPECT_THAT(p.ExpandRing({10, 20, 30}),
              IsOkAndHolds(ElementsAre(10, 20, 30)));
}

TEST(ChunkPlanTest, DefaultConstructedGuards) {
  ChunkPlan p;
  EXPECT_THAT(p.ExpandRing({}), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(p.ChunkOfPosition(0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(p.SlotOf(0), 0);
  EXPECT_EQ(p.SlotOf(-1), 0);
}

TEST(ChunkPlanTest, LargeKChunkPlanOverflow) {
  EXPECT_THAT(BuildChunkPlan(std::numeric_limits<int64_t>::max(), 1, 1, true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ChunkPlanTest, BuildChunkPlanInvalidInputs) {
  EXPECT_THAT(BuildChunkPlan(0, 2, 2, true),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildChunkPlan(7, 0, 2, true),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildChunkPlan(7, 2, 0, true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BuildStagingWindowConfigTest, AllNoneReturnsNullopt) {
  TF_ASSERT_OK_AND_ASSIGN(auto res, BuildStagingWindowConfig());
  EXPECT_FALSE(res.has_value());

  TF_ASSERT_OK_AND_ASSIGN(
      auto res_explicit,
      BuildStagingWindowConfig(
          /*window_chunks=*/std::nullopt, /*chunk_blocks=*/std::nullopt,
          /*chunk_bytes_target=*/std::nullopt,
          /*stream_bytes_target=*/std::nullopt, /*admission=*/std::nullopt,
          /*admission_timeout_ms=*/std::nullopt,
          /*assume_peer_support=*/std::nullopt));
  EXPECT_FALSE(res_explicit.has_value());
}

TEST(BuildStagingWindowConfigTest,
     ExplicitWindowChunksZeroWithEnvOverridesToWindowOff) {
  ScopedEnv env("TPU_RAIDEN_STAGING_WINDOW_CHUNKS", "2");
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          BuildStagingWindowConfig(/*window_chunks=*/0));
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res->window_chunks, 0);
  EXPECT_FALSE(res->enabled());
}

TEST(BuildStagingWindowConfigTest,
     OnlyAdmissionWaitWithEnvWindowKeepsEnvWindow) {
  ScopedEnv env("TPU_RAIDEN_STAGING_WINDOW_CHUNKS", "2");
  TF_ASSERT_OK_AND_ASSIGN(auto res, BuildStagingWindowConfig(
                                        /*window_chunks=*/std::nullopt,
                                        /*chunk_blocks=*/std::nullopt,
                                        /*chunk_bytes_target=*/std::nullopt,
                                        /*stream_bytes_target=*/std::nullopt,
                                        /*admission=*/"wait"));
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res->window_chunks, 2);
  EXPECT_TRUE(res->enabled());
  EXPECT_EQ(res->admission, StagingWindowConfig::Admission::kWait);
}

TEST(BuildStagingWindowConfigTest, CustomValuesBuildsConfig) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto res, BuildStagingWindowConfig(
                    /*window_chunks=*/4, /*chunk_blocks=*/16,
                    /*chunk_bytes=*/16 << 20, /*stream_bytes=*/8 << 20,
                    /*admission=*/"wait", /*admission_timeout_ms=*/500,
                    /*assume_peer_support=*/true));
  ASSERT_TRUE(res.has_value());
  const auto& cfg = *res;
  EXPECT_EQ(cfg.window_chunks, 4);
  EXPECT_EQ(cfg.chunk_blocks, 16);
  EXPECT_EQ(cfg.chunk_bytes_target, 16 << 20);
  EXPECT_EQ(cfg.stream_bytes_target, 8 << 20);
  EXPECT_EQ(cfg.admission, StagingWindowConfig::Admission::kWait);
  EXPECT_EQ(cfg.admission_timeout_ms, 500);
  EXPECT_TRUE(cfg.assume_peer_support);
}

TEST(BuildStagingWindowConfigTest, InvalidAdmissionReturnsError) {
  auto res = BuildStagingWindowConfig(
      /*window_chunks=*/4, /*chunk_blocks=*/16,
      /*chunk_bytes=*/16 << 20, /*stream_bytes=*/8 << 20,
      /*admission=*/"invalid_policy", /*admission_timeout_ms=*/500,
      /*assume_peer_support=*/true);
  EXPECT_THAT(res.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace tpu_raiden

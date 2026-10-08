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

#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::SizeIs;

TEST(LogicalReshardPlannerTest, ComputeSeedSamplerCountMatchesHostRatio) {
  // Equal host counts (4 trainer hosts, 4 sampler hosts), ratio 1.0 -> 1.
  EXPECT_EQ(LogicalReshardPlanner::ComputeSeedSamplerCount(
                /*num_dst_replicas=*/8, /*broadcast_host_ratio=*/1.0,
                /*trainer_hosts=*/4, /*sampler_hosts=*/4),
            1);

  // 16 trainer hosts, 4 sampler hosts, ratio 1.0 -> 4.
  EXPECT_EQ(LogicalReshardPlanner::ComputeSeedSamplerCount(
                /*num_dst_replicas=*/8, /*broadcast_host_ratio=*/1.0,
                /*trainer_hosts=*/16, /*sampler_hosts=*/4),
            4);

  // 16 trainer hosts, 4 sampler hosts, ratio 0.5 -> 2.
  EXPECT_EQ(LogicalReshardPlanner::ComputeSeedSamplerCount(
                /*num_dst_replicas=*/8, /*broadcast_host_ratio=*/0.5,
                /*trainer_hosts=*/16, /*sampler_hosts=*/4),
            2);

  // Ratio <= 0.0 -> push to all samplers at once (D = 8).
  EXPECT_EQ(LogicalReshardPlanner::ComputeSeedSamplerCount(
                /*num_dst_replicas=*/8, /*broadcast_host_ratio=*/0.0,
                /*trainer_hosts=*/4, /*sampler_hosts=*/4),
            8);

  // Smaller trainer than sampler clamps to at least 1.
  EXPECT_EQ(LogicalReshardPlanner::ComputeSeedSamplerCount(
                /*num_dst_replicas=*/8, /*broadcast_host_ratio=*/1.0,
                /*trainer_hosts=*/1, /*sampler_hosts=*/8),
            1);
}

TEST(LogicalReshardPlannerTest, PartitionVariableBundlesRespectsGroupCount) {
  std::vector<int64_t> layer_bytes(32, 1024 * 1024);

  // Default 8 bundle groups across 32 uniform layers -> 8 bundles of 4 layers.
  std::vector<VariableBundleSpec> bundles_8 =
      LogicalReshardPlanner::PartitionVariableBundles(32, layer_bytes, 8);
  ASSERT_THAT(bundles_8, SizeIs(8));
  for (int32_t b = 0; b < 8; ++b) {
    EXPECT_EQ(bundles_8[b].bundle_id, b);
    EXPECT_THAT(bundles_8[b].layer_indices,
                ElementsAre(b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3));
    EXPECT_EQ(bundles_8[b].total_bytes, 4 * 1024 * 1024);
  }

  // Configurable group count = 4 -> 4 bundles of 8 layers.
  std::vector<VariableBundleSpec> bundles_4 =
      LogicalReshardPlanner::PartitionVariableBundles(32, layer_bytes, 4);
  ASSERT_THAT(bundles_4, SizeIs(4));
  for (int32_t b = 0; b < 4; ++b) {
    EXPECT_THAT(bundles_4[b].layer_indices, SizeIs(8));
  }
}

TEST(LogicalReshardPlannerTest, DeduplicatesPlansAndBuildsStripedSeeding) {
  std::vector<VariableSpec> src_vars;
  std::vector<VariableSpec> dst_vars;
  for (int i = 0; i < 16; ++i) {
    VariableSpec sv;
    sv.name = absl::StrCat("weights_", i);
    sv.global_shape = (i % 2 == 0) ? std::vector<int64_t>{64, 128}
                                   : std::vector<int64_t>{128, 64};
    sv.mesh_shape = {2, 1};
    sv.layout = {1, 0};
    sv.itemsize = 2;
    sv.global_shard_indices = {0, 1};
    src_vars.push_back(sv);

    VariableSpec dv = sv;
    dv.mesh_shape = {1, 2};
    dst_vars.push_back(dv);
  }

  absl::StatusOr<LogicalReshardSchedule> schedule =
      LogicalReshardPlanner::ComputeLogicalSchedule(
          src_vars, dst_vars, /*num_src_shards=*/2, /*num_dst_shards=*/2,
          /*num_dst_replicas=*/4, /*broadcast_host_ratio=*/1.0,
          /*trainer_hosts=*/1, /*sampler_hosts=*/1, /*num_bundle_groups=*/8);
  ASSERT_OK(schedule);
  EXPECT_EQ(schedule->num_unique_plans, 2);
  EXPECT_THAT(schedule->variable_bundles, SizeIs(8));

  // Default seeding: R = 2 and G = min(8 bundles, 4 / 2) = 2 stripes of 4
  // bundles, seeded on replicas {0, 1} and {2, 3}.
  const SeedLayout& layout = schedule->seed_layout;
  EXPECT_EQ(layout.replication, 2);
  EXPECT_EQ(layout.num_stripes, 2);
  EXPECT_THAT(layout.stripe_bundles,
              ElementsAre(ElementsAre(0, 1, 2, 3), ElementsAre(4, 5, 6, 7)));
  EXPECT_THAT(layout.stripe_seeds,
              ElementsAre(ElementsAre(0, 1), ElementsAre(2, 3)));
  // One Trainer stream (1 trainer host / 1 sampler host): one push per wave,
  // copy 0 of both stripes first.
  EXPECT_THAT(layout.trainer_waves,
              ElementsAre(ElementsAre(Pair(0, 0)), ElementsAre(Pair(2, 1)),
                          ElementsAre(Pair(1, 0)), ElementsAre(Pair(3, 1))));

  // Trainer actions are replica independent; the materializer emits them for
  // the seeds of each layer's stripe only.
  for (int32_t s = 0; s < 2; ++s) {
    for (int32_t pid = 0; pid < schedule->num_unique_plans; ++pid) {
      for (const ShardCopyAction& act :
           schedule->trainer_actions_by_shard_and_plan[s][pid]) {
        EXPECT_EQ(act.dst_replica_idx, 0);
      }
    }
  }
  ASSERT_THAT(schedule->dst_shard_expected_layer_chunks, SizeIs(2));
  for (const std::vector<int32_t>& per_layer :
       schedule->dst_shard_expected_layer_chunks) {
    ASSERT_THAT(per_layer, SizeIs(16));
    for (int32_t chunks : per_layer) EXPECT_GT(chunks, 0);
  }
}

TEST(LogicalReshardPlannerTest, RaisesBundleCountToRequestedStripes) {
  VariableSpec v;
  v.global_shape = {16, 16};
  v.mesh_shape = {1, 1};
  v.layout = {1, 0};
  v.itemsize = 4;
  v.global_shard_indices = {0};
  std::vector<VariableSpec> vars(8, v);
  SeedingOptions seeding;
  seeding.num_stripes = 4;
  absl::StatusOr<LogicalReshardSchedule> schedule =
      LogicalReshardPlanner::ComputeLogicalSchedule(
          vars, vars, /*num_src_shards=*/1, /*num_dst_shards=*/1,
          /*num_dst_replicas=*/8, /*broadcast_host_ratio=*/1.0,
          /*trainer_hosts=*/1, /*sampler_hosts=*/1, /*num_bundle_groups=*/2,
          seeding);
  ASSERT_OK(schedule);
  EXPECT_THAT(schedule->variable_bundles, SizeIs(4));
  EXPECT_EQ(schedule->seed_layout.num_stripes, 4);

  // Never more bundles than variables.
  seeding.num_stripes = 64;
  schedule = LogicalReshardPlanner::ComputeLogicalSchedule(
      vars, vars, 1, 1, /*num_dst_replicas=*/64, 1.0, 1, 1,
      /*num_bundle_groups=*/2, seeding);
  ASSERT_OK(schedule);
  EXPECT_THAT(schedule->variable_bundles, SizeIs(8));
  EXPECT_EQ(schedule->seed_layout.num_stripes, 8);
}

TEST(NdSliceAndLogicalGeometryTest,
     ComputesSlicesIntersectionsPhysicalLayoutsAndTiling) {
  NdSlice s1{{0, 0}, {64, 256}};
  EXPECT_EQ(s1.sizes.size(), 2);
  EXPECT_FALSE(s1.IsEmpty());
  EXPECT_EQ(s1.NumElements(), 64 * 256);

  NdSlice s2{{32, 128}, {64, 256}};
  std::optional<NdSlice> inter =
      LogicalReshardPlanner::IntersectNdSlices(s1, s2);
  ASSERT_TRUE(inter.has_value());
  EXPECT_THAT(inter->offsets, ElementsAre(32, 128));
  EXPECT_THAT(inter->sizes, ElementsAre(32, 128));

  NdSlice disjoint{{64, 0}, {32, 256}};
  EXPECT_FALSE(
      LogicalReshardPlanner::IntersectNdSlices(s1, disjoint).has_value());

  EXPECT_THAT(LogicalReshardPlanner::GetGlobalIndices(3, {2, 2}),
              ElementsAre(1, 1));

  NdSlice phys = LogicalReshardPlanner::ToPhysical(s2, {0, 1});
  EXPECT_THAT(phys.offsets, ElementsAre(128, 32));
  EXPECT_THAT(phys.sizes, ElementsAre(256, 64));

  EXPECT_TRUE(LogicalReshardPlanner::IsNdSliceTileAligned(s1, s1, *inter,
                                                          {1, 0}, {1, 0}));

  std::vector<StridedCopyChunk> chunks =
      LogicalReshardPlanner::GenerateStridedCopyChunks(
          s1, s1, *inter, /*src_layout=*/{1, 0}, /*dst_layout=*/{1, 0},
          /*itemsize=*/2);
  EXPECT_THAT(chunks, SizeIs(Gt(0)));

  std::vector<StridedCopyChunk> tile_chunks =
      LogicalReshardPlanner::GenerateStridedCopyChunksTileAware(
          s1, s1, *inter, /*src_layout=*/{1, 0}, /*dst_layout=*/{1, 0},
          /*itemsize=*/2);
  EXPECT_THAT(tile_chunks, SizeIs(Gt(0)));
}

TEST(LogicalReshardPlannerTest,
     OffsetsDestinationShardsAcrossTrainerHostsForIngressBalancing) {
  VariableSpec sv;
  sv.name = "weight_dense";
  sv.global_shape = {128, 128};
  sv.mesh_shape = {8, 1};
  sv.layout = {0, 1};
  sv.itemsize = 2;
  sv.global_shard_indices = {0, 1, 2, 3, 4, 5, 6, 7};

  VariableSpec dv = sv;
  dv.mesh_shape = {4, 1};
  dv.global_shard_indices = {0, 1, 2, 3};

  // 4 trainer hosts (2 shards each = 8 source shards), 2 sampler hosts per
  // replica (2 shards each = 4 destination shards) and ratio 1.0 -> 2 Trainer
  // streams per wave.
  absl::StatusOr<LogicalReshardSchedule> sched_or =
      LogicalReshardPlanner::ComputeLogicalSchedule(
          {sv}, {dv}, /*num_src_shards=*/8, /*num_dst_shards=*/4,
          /*num_dst_replicas=*/4, /*broadcast_host_ratio=*/1.0,
          /*trainer_hosts=*/4, /*sampler_hosts=*/2,
          /*num_bundle_groups=*/1);
  ASSERT_OK(sched_or);
  const LogicalReshardSchedule& sched = *sched_or;
  for (const auto& wave : sched.seed_layout.trainer_waves) {
    EXPECT_THAT(wave, SizeIs(testing::Le(2)));
  }

  // Trainer hosts 0 and 1 (source shards 0 and 2) start on different pushes
  // of a wave and both start at sampler host 0 (shards 0, 1).
  const auto& actions_th0 = sched.trainer_actions_by_shard_and_plan[0][0];
  ASSERT_FALSE(actions_th0.empty());
  EXPECT_LT(actions_th0.front().dst_shard_idx, 2);
  const auto& actions_th1 = sched.trainer_actions_by_shard_and_plan[2][0];
  ASSERT_FALSE(actions_th1.empty());
  EXPECT_LT(actions_th1.front().dst_shard_idx, 2);

  // Trainer host 2 (source shard 4) starts on the same push as host 0, so it
  // starts at sampler host 1 (shards 2, 3).
  const auto& actions_th2 = sched.trainer_actions_by_shard_and_plan[4][0];
  ASSERT_FALSE(actions_th2.empty());
  EXPECT_GE(actions_th2.front().dst_shard_idx, 2);
  for (const auto& per_shard : sched.trainer_actions_by_shard_and_plan) {
    for (const ShardCopyAction& act : per_shard[0]) {
      EXPECT_EQ(act.dst_replica_idx, 0);
    }
  }
}

TEST(LogicalReshardPlannerTest,
     NormalizesLegacyNegativeLayoutsWithoutOutOfBoundsAccess) {
  VariableSpec sv;
  sv.name = "legacy_var";
  sv.global_shape = {32, 64};
  sv.mesh_shape = {2, 1};
  sv.layout = {-1, 0};
  sv.itemsize = 4;
  sv.global_shard_indices = {0, 1};

  VariableSpec dv = sv;
  dv.mesh_shape = {1, 2};
  dv.layout = {-1, -1};

  absl::StatusOr<LogicalReshardSchedule> sched_or =
      LogicalReshardPlanner::ComputeLogicalSchedule(
          {sv}, {dv}, /*num_src_shards=*/2, /*num_dst_shards=*/2,
          /*num_dst_replicas=*/1, /*broadcast_host_ratio=*/1.0,
          /*trainer_hosts=*/1, /*sampler_hosts=*/1,
          /*num_bundle_groups=*/1);
  ASSERT_OK(sched_or);
  // One replica: the replication is clamped to 1 and nothing is pulled.
  EXPECT_EQ(sched_or->seed_layout.replication, 1);
  EXPECT_THAT(sched_or->seed_layout.stripe_seeds, ElementsAre(ElementsAre(0)));
}

TEST(LogicalReshardPlannerTest, RequiresInRangeGlobalShardIndices) {
  VariableSpec v;
  v.name = "w";
  v.global_shape = {32, 64};
  v.mesh_shape = {2, 1};
  v.layout = {1, 0};
  v.itemsize = 4;
  v.global_shard_indices = {0, 1};

  VariableSpec missing = v;
  missing.global_shard_indices.clear();
  EXPECT_THAT(
      LogicalReshardPlanner::ComputeLogicalSchedule(
          {missing}, {v}, /*num_src_shards=*/2, /*num_dst_shards=*/2,
          /*num_dst_replicas=*/1, /*broadcast_host_ratio=*/1.0,
          /*trainer_hosts=*/1, /*sampler_hosts=*/1,
          /*num_bundle_groups=*/1),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("global_shard_indices is required")));

  VariableSpec out_of_range = v;
  out_of_range.global_shard_indices = {0, 2};
  EXPECT_THAT(LogicalReshardPlanner::ComputeLogicalSchedule(
                  {v}, {out_of_range}, /*num_src_shards=*/2,
                  /*num_dst_shards=*/2, /*num_dst_replicas=*/1,
                  /*broadcast_host_ratio=*/1.0, /*trainer_hosts=*/1,
                  /*sampler_hosts=*/1, /*num_bundle_groups=*/1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("outside [0, 2)")));
}

std::vector<int64_t> EqualBundles(int32_t num_bundles) {
  return std::vector<int64_t>(num_bundles, 1 << 20);
}

SeedingOptions Seeding(int32_t num_stripes, int32_t replication) {
  SeedingOptions options;
  options.num_stripes = num_stripes;
  options.replication = replication;
  return options;
}

TEST(SeedLayoutTest, BuildsStripesSeedsAndReplicaMajorWaves) {
  // D = 10, R = 2, auto G = min(8, 10 / 2) = 5: every replica is a seed.
  absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
      EqualBundles(8), /*num_dst_replicas=*/10, Seeding(0, 2),
      /*trainer_streams=*/3);
  ASSERT_OK(layout);
  EXPECT_EQ(layout->num_stripes, 5);
  EXPECT_THAT(layout->stripe_bundles,
              ElementsAre(ElementsAre(0, 1), ElementsAre(2, 3), ElementsAre(4),
                          ElementsAre(5, 6), ElementsAre(7)));
  EXPECT_THAT(
      layout->stripe_seeds,
      ElementsAre(ElementsAre(0, 1), ElementsAre(2, 3), ElementsAre(4, 5),
                  ElementsAre(6, 7), ElementsAre(8, 9)));
  EXPECT_THAT(layout->trainer_waves,
              ElementsAre(ElementsAre(Pair(0, 0), Pair(2, 1), Pair(4, 2)),
                          ElementsAre(Pair(6, 3), Pair(8, 4), Pair(1, 0)),
                          ElementsAre(Pair(3, 1), Pair(5, 2), Pair(7, 3)),
                          ElementsAre(Pair(9, 4))));

  // Fewer stripes than replicas: seeds are spread over the replicas.
  layout = LogicalReshardPlanner::BuildSeedLayout(
      EqualBundles(8), /*num_dst_replicas=*/12, Seeding(2, 2),
      /*trainer_streams=*/16);
  ASSERT_OK(layout);
  EXPECT_THAT(layout->stripe_bundles,
              ElementsAre(ElementsAre(0, 1, 2, 3), ElementsAre(4, 5, 6, 7)));
  EXPECT_THAT(layout->stripe_seeds,
              ElementsAre(ElementsAre(0, 3), ElementsAre(6, 9)));
  EXPECT_THAT(
      layout->trainer_waves,
      ElementsAre(ElementsAre(Pair(0, 0), Pair(6, 1), Pair(3, 0), Pair(9, 1))));
}

TEST(SeedLayoutTest, ClampsAndValidatesOptions) {
  // R is clamped to D, and G to min(B, D / R).
  absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
      EqualBundles(4), /*num_dst_replicas=*/3, Seeding(100, 5), 1);
  ASSERT_OK(layout);
  EXPECT_EQ(layout->replication, 3);
  EXPECT_EQ(layout->num_stripes, 1);
  EXPECT_THAT(layout->stripe_seeds, ElementsAre(ElementsAre(0, 1, 2)));

  layout = LogicalReshardPlanner::BuildSeedLayout(EqualBundles(3), 16,
                                                  Seeding(100, 2), 1);
  ASSERT_OK(layout);
  EXPECT_EQ(layout->num_stripes, 3);

  EXPECT_EQ(LogicalReshardPlanner::BuildSeedLayout(EqualBundles(4), 4,
                                                   Seeding(-1, 2), 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(LogicalReshardPlanner::BuildSeedLayout(EqualBundles(4), 4,
                                                   Seeding(0, 0), 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  SeedingOptions bad_order = Seeding(0, 2);
  bad_order.replica_order = {0, 1, 1, 3};
  EXPECT_EQ(
      LogicalReshardPlanner::BuildSeedLayout(EqualBundles(4), 4, bad_order, 1)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(SeedLayoutTest, SingleStripeIsFullSeeding) {
  // G = 1, R = D: every replica gets the whole model from the Trainer.
  absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
      EqualBundles(3), /*num_dst_replicas=*/4, Seeding(1, 4), 4);
  ASSERT_OK(layout);
  EXPECT_THAT(layout->stripe_bundles, ElementsAre(ElementsAre(0, 1, 2)));
  EXPECT_THAT(layout->stripe_seeds, ElementsAre(ElementsAre(0, 1, 2, 3)));

  // G = 1, R = 2: two full seeds spread over the replicas; everybody else
  // pulls every bundle at runtime.
  layout = LogicalReshardPlanner::BuildSeedLayout(EqualBundles(3), 6,
                                                  Seeding(1, 2), 4);
  ASSERT_OK(layout);
  EXPECT_THAT(layout->stripe_seeds, ElementsAre(ElementsAre(0, 3)));
}

TEST(SeedLayoutTest, StableUnderUnchangedMembership) {
  // The same units listed in two different orders. `replica_order` is the
  // sorted unit order, as computed by the controller.
  const std::vector<std::string> listing_a = {"u0", "u1", "u2", "u3",
                                              "u4", "u5", "u6", "u7"};
  const std::vector<std::string> listing_b = {"u5", "u2", "u7", "u0",
                                              "u3", "u6", "u1", "u4"};
  auto seeds_by_unit = [](const std::vector<std::string>& units) {
    SeedingOptions options = Seeding(0, 2);
    options.replica_order.resize(units.size());
    for (int32_t i = 0; i < static_cast<int32_t>(units.size()); ++i) {
      options.replica_order[i] = i;
    }
    std::sort(options.replica_order.begin(), options.replica_order.end(),
              [&](int32_t a, int32_t b) { return units[a] < units[b]; });
    absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
        EqualBundles(4), static_cast<int32_t>(units.size()), options, 2);
    EXPECT_OK(layout);
    std::vector<std::vector<std::string>> out;
    for (const std::vector<int32_t>& seeds : layout->stripe_seeds) {
      out.emplace_back();
      for (int32_t r : seeds) out.back().push_back(units[r]);
    }
    return out;
  };
  EXPECT_EQ(seeds_by_unit(listing_a), seeds_by_unit(listing_b));
  EXPECT_EQ(seeds_by_unit(listing_a),
            (std::vector<std::vector<std::string>>{
                {"u0", "u1"}, {"u2", "u3"}, {"u4", "u5"}, {"u6", "u7"}}));

  // Rebuilding with unchanged inputs yields the identical layout.
  absl::StatusOr<SeedLayout> l1 = LogicalReshardPlanner::BuildSeedLayout(
      EqualBundles(16), 24, Seeding(0, 2), 4);
  absl::StatusOr<SeedLayout> l2 = LogicalReshardPlanner::BuildSeedLayout(
      EqualBundles(16), 24, Seeding(0, 2), 4);
  ASSERT_OK(l1);
  ASSERT_OK(l2);
  EXPECT_EQ(*l1, *l2);
}

TEST(SeedLayoutTest, BalancesUnevenBundleSizes) {
  const std::vector<int64_t> bytes = {8, 1, 1, 1, 4, 2, 1, 6, 3, 3, 1, 5};
  absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
      bytes, /*num_dst_replicas=*/8, Seeding(0, 2), 4);
  ASSERT_OK(layout);
  ASSERT_EQ(layout->num_stripes, 4);
  // Stripes are contiguous and byte balanced (36 bytes total, 9 per stripe).
  int32_t next_bundle = 0;
  for (const std::vector<int32_t>& stripe : layout->stripe_bundles) {
    int64_t stripe_bytes = 0;
    for (int32_t b : stripe) {
      EXPECT_EQ(b, next_bundle++);
      stripe_bytes += bytes[b];
    }
    EXPECT_GE(stripe_bytes, 6);
    EXPECT_LE(stripe_bytes, 12);
  }
  EXPECT_EQ(next_bundle, static_cast<int32_t>(bytes.size()));
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

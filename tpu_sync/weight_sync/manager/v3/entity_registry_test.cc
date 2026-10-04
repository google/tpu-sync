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

#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;

TEST(EntityRegistryTest,
     BuildsCompositeEntitiesAndGroupsMultiHostDestinationReplicas) {
  EntityRegistry reg;

  // Register a 2-host trainer (t_h0, t_h1) and 2x 2-host samplers
  // (sampler_0: h0/h1, sampler_1: h0/h1).
  for (int h = 0; h < 2; ++h) {
    tpu_sync::rpc::RegisterWorkUnitRequest treq;
    treq.mutable_unit()->set_job_name("trainer");
    treq.mutable_unit()->set_job_replica_id(absl::StrCat(h));
    treq.mutable_unit()->set_data_name("weights");
    treq.mutable_unit()->set_data_replica_idx(0);
    treq.add_shards(absl::StrCat("10.0.0.", h + 1, ":8000"));
    treq.add_shards(absl::StrCat("10.0.0.", h + 1, ":8000"));
    treq.set_control_plane_rpc_address(absl::StrCat("10.0.0.", h + 1, ":9000"));
    auto* v = treq.add_variables();
    v->set_name("w0");
    v->add_shape(64);
    v->add_shape(128);
    v->add_mesh_shape(2);
    v->add_mesh_shape(2);
    v->add_layout(1);
    v->add_layout(0);
    v->set_item_size(2);
    v->set_layer_idx(0);
    v->add_global_shard_indices(h * 2);
    v->add_global_shard_indices(h * 2 + 1);
    ASSERT_OK(reg.RegisterWorkUnit(treq));
  }

  std::vector<RaidenId> dst_host_units;
  for (int r = 0; r < 2; ++r) {
    for (int h = 0; h < 2; ++h) {
      RaidenId u{absl::StrCat("sampler_", r), absl::StrCat(h), "weights", r};
      dst_host_units.push_back(u);
      tpu_sync::rpc::RegisterWorkUnitRequest sreq;
      *sreq.mutable_unit() = RaidenIdToProto(u);
      sreq.add_shards(absl::StrCat("10.0.", r + 1, ".", h + 1, ":8100"));
      sreq.add_shards(absl::StrCat("10.0.", r + 1, ".", h + 1, ":8100"));
      sreq.set_control_plane_rpc_address(
          absl::StrCat("10.0.", r + 1, ".", h + 1, ":9100"));
      auto* v = sreq.add_variables();
      v->set_name("w0");
      v->add_shape(64);
      v->add_shape(128);
      v->add_mesh_shape(1);
      v->add_mesh_shape(4);
      v->add_layout(1);
      v->add_layout(0);
      v->set_item_size(2);
      v->set_layer_idx(0);
      v->add_global_shard_indices(h * 2);
      v->add_global_shard_indices(h * 2 + 1);
      ASSERT_OK(reg.RegisterWorkUnit(sreq));
    }
  }

  std::vector<RaidenId> trainer_units = {
      RaidenId{"trainer", "1", "weights", 0},
      RaidenId{"trainer", "0", "weights", 0},
  };
  absl::StatusOr<WorkUnitEntity> comp_trainer =
      reg.BuildCompositeEntity(trainer_units);
  ASSERT_OK(comp_trainer);
  EXPECT_THAT(comp_trainer->shards, SizeIs(4));
  EXPECT_THAT(comp_trainer->hosts, SizeIs(2));
  // Sorted by numeric job_replica_id (0 then 1) so shard order matches global
  // shard indices {0, 1, 2, 3}.
  EXPECT_EQ(comp_trainer->hosts[0].unit.job_replica_id, "0");
  EXPECT_EQ(comp_trainer->hosts[1].unit.job_replica_id, "1");

  absl::StatusOr<std::vector<WorkUnitEntity>> dst_replicas =
      reg.BuildDestinationReplicaEntities(dst_host_units);
  ASSERT_OK(dst_replicas);
  ASSERT_THAT(*dst_replicas, SizeIs(2));
  EXPECT_EQ((*dst_replicas)[0].unit.job_name, "sampler_0");
  EXPECT_THAT((*dst_replicas)[0].shards, SizeIs(4));
  EXPECT_THAT((*dst_replicas)[0].hosts, SizeIs(2));
  EXPECT_EQ((*dst_replicas)[1].unit.job_name, "sampler_1");
  EXPECT_THAT((*dst_replicas)[1].shards, SizeIs(4));
  EXPECT_THAT((*dst_replicas)[1].hosts, SizeIs(2));
}

tpu_sync::rpc::RegisterWorkUnitRequest MakeIpv6TwoHostRequest() {
  tpu_sync::rpc::RegisterWorkUnitRequest req;
  req.mutable_unit()->set_job_name("sampler");
  req.mutable_unit()->set_job_replica_id("0");
  req.mutable_unit()->set_data_name("weights");
  // Four distinct data endpoints on two hosts, so hosts are matched by IP.
  req.add_shards("[fd00::1]:8000");
  req.add_shards("[fd00::1]:8001");
  req.add_shards("[fd00::2]:8000");
  req.add_shards("[fd00::2]:8001");
  req.set_control_plane_rpc_address("[fd00::2]:9000,[fd00::1]:9000");
  return req;
}

TEST(EntityRegistryTest, MatchesBracketedIpv6HostsByAddress) {
  EntityRegistry reg;
  tpu_sync::rpc::RegisterWorkUnitRequest req = MakeIpv6TwoHostRequest();
  ASSERT_OK(reg.RegisterWorkUnit(req));

  absl::StatusOr<WorkUnitEntity> entity =
      reg.GetEntity(RaidenIdFromProto(req.unit()));
  ASSERT_OK(entity);
  ASSERT_THAT(entity->hosts, SizeIs(2));
  EXPECT_EQ(entity->hosts[0].control_address, "[fd00::2]:9000");
  EXPECT_THAT(entity->hosts[0].owned_global_shard_indices, ElementsAre(2, 3));
  EXPECT_EQ(entity->hosts[1].control_address, "[fd00::1]:9000");
  EXPECT_THAT(entity->hosts[1].owned_global_shard_indices, ElementsAre(0, 1));
}

TEST(EntityRegistryTest, OnlyShardEndpointOrVariableChangesAreTopologyChanges) {
  EntityRegistry reg;
  tpu_sync::rpc::RegisterWorkUnitRequest req = MakeIpv6TwoHostRequest();
  ASSERT_OK(reg.RegisterWorkUnit(req));

  absl::StatusOr<bool> changed = reg.RegisterWorkUnit(req);
  ASSERT_OK(changed);
  EXPECT_FALSE(*changed);

  req.add_shards("[fd00::2]:8002");
  changed = reg.RegisterWorkUnit(req);
  ASSERT_OK(changed);
  EXPECT_TRUE(*changed);
}

tpu_sync::rpc::RegisterWorkUnitRequest MakeOneVariableRequest() {
  tpu_sync::rpc::RegisterWorkUnitRequest req;
  req.mutable_unit()->set_job_name("trainer");
  req.mutable_unit()->set_job_replica_id("0");
  req.add_shards("10.0.0.1:8000");
  req.add_shards("10.0.0.1:8000");
  auto* v = req.add_variables();
  v->set_name("w0");
  v->add_shape(64);
  v->add_mesh_shape(2);
  v->add_layout(0);
  v->set_item_size(2);
  return req;
}

TEST(EntityRegistryTest, RejectsVariablesWithoutGlobalShardIndices) {
  EntityRegistry reg;
  tpu_sync::rpc::RegisterWorkUnitRequest req = MakeOneVariableRequest();
  EXPECT_THAT(reg.RegisterWorkUnit(req),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("'w0'"),
                        HasSubstr("global_shard_indices is required"))));
  EXPECT_FALSE(reg.HasUnit(RaidenIdFromProto(req.unit())));
}

TEST(EntityRegistryTest, RejectsGlobalShardIndicesOfTheWrongLength) {
  EntityRegistry reg;
  tpu_sync::rpc::RegisterWorkUnitRequest req = MakeOneVariableRequest();
  req.mutable_variables(0)->add_global_shard_indices(0);
  EXPECT_THAT(reg.RegisterWorkUnit(req),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("1 global_shard_indices for 2 "
                                               "shards")));

  req.mutable_variables(0)->add_global_shard_indices(1);
  ASSERT_OK(reg.RegisterWorkUnit(req));
}

TEST(EntityRegistryTest, RejectsUnitLevelGeometryWithoutVariables) {
  EntityRegistry reg;
  tpu_sync::rpc::RegisterWorkUnitRequest req = MakeOneVariableRequest();
  req.clear_variables();
  req.add_global_shape(64);
  req.add_mesh_shape(2);
  EXPECT_THAT(reg.RegisterWorkUnit(req),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("`variables`")));
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

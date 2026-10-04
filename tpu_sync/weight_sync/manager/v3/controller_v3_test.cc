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

#include "tpu_sync/weight_sync/manager/v3/controller_v3.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/async_control_server.h"
#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"
#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"
#include "tpu_sync/weight_sync/manager/v3/test_pull_wire_codec.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

ABSL_DECLARE_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size);

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::SizeIs;

// Expects that the pull scheduler of the last execution of |req_id|
// completed: every host of every replica holds every bundle. Returns the
// number of completed pulls.
int64_t ExpectPullPhaseComplete(const RaidenControllerV3& controller,
                                absl::string_view req_id) {
  std::shared_ptr<TransferPullService> service =
      controller.GetPullServiceForTest(req_id);
  EXPECT_NE(service, nullptr);
  if (service == nullptr) return 0;
  EXPECT_TRUE(service->scheduler().complete());
  EXPECT_OK(service->scheduler().status());
  int64_t completions = 0;
  for (const PullShardStats& shard : service->scheduler().GetStats()) {
    completions += shard.completions;
  }
  return completions;
}

TEST(RaidenControllerV3E2eTest,
     HybridTrainerSeedPushAndDynamicSamplerBundlePull) {
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size, 0);

  constexpr size_t kNumLayers = 8;
  constexpr size_t kNumShards = 2;
  constexpr size_t kSliceByteSize = 16384;
  constexpr int kNumSamplers = 4;

  // Start V3 controller server with 4 bundle groups (2 layers per bundle) and
  // broadcast_host_ratio = 1.0 (so the Trainer pushes to 1 sampler at a time).
  RaidenControllerV3::Options opts;
  opts.port = 0;
  opts.broadcast_host_ratio = 1.0;
  opts.num_bundle_groups = 4;
  opts.max_concurrent_uploads_per_source = 1;
  RaidenControllerV3 controller(opts);
  absl::StatusOr<int> ctrl_port = controller.StartServer();
  ASSERT_OK(ctrl_port);
  ASSERT_THAT(*ctrl_port, Gt(0));

  // Create 1 Trainer WeightSynchronizerBase with listener enabled.
  auto trainer_ws = std::make_unique<WeightSynchronizerBase>(
      kNumLayers, kNumShards, kSliceByteSize,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/2, /*listener_port=*/0,
      /*bind_ip=*/"127.0.0.1", /*layer_names=*/std::vector<std::string>{},
      /*auto_h2d=*/false);
  ASSERT_TRUE(trainer_ws->local_port().has_value());
  ASSERT_TRUE(trainer_ws->listener_port().has_value());

  // Populate Trainer host buffers with a unique pattern per (layer, shard, i).
  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t s = 0; s < kNumShards; ++s) {
      uint8_t* ptr = const_cast<uint8_t*>(trainer_ws->GetHostPointer(l, s));
      ASSERT_NE(ptr, nullptr);
      for (size_t i = 0; i < kSliceByteSize; ++i) {
        ptr[i] = static_cast<uint8_t>((l * 37 + s * 13 + i + 1) & 0xFF);
      }
    }
  }

  // Create 4 Sampler WeightSynchronizerBase instances with listeners enabled.
  std::vector<std::unique_ptr<WeightSynchronizerBase>> sampler_ws(kNumSamplers);
  for (int r = 0; r < kNumSamplers; ++r) {
    sampler_ws[r] = std::make_unique<WeightSynchronizerBase>(
        kNumLayers, kNumShards, kSliceByteSize,
        /*local_port=*/0, /*host_blocks_to_allocate=*/1,
        /*parallelism=*/2, /*listener_port=*/0,
        /*bind_ip=*/"127.0.0.1", /*layer_names=*/std::vector<std::string>{},
        /*auto_h2d=*/false);
    ASSERT_TRUE(sampler_ws[r]->local_port().has_value());
    ASSERT_TRUE(sampler_ws[r]->listener_port().has_value());
    for (size_t l = 0; l < kNumLayers; ++l) {
      for (size_t s = 0; s < kNumShards; ++s) {
        uint8_t* ptr =
            const_cast<uint8_t*>(sampler_ws[r]->GetHostPointer(l, s));
        ASSERT_NE(ptr, nullptr);
        std::memset(ptr, 0, kSliceByteSize);
      }
    }
  }

  // Register Trainer work unit with the V3 controller.
  RaidenId trainer_unit{"trainer", "0", "weights", 0};
  tpu_sync::rpc::RegisterWorkUnitRequest trainer_reg;
  *trainer_reg.mutable_unit() = RaidenIdToProto(trainer_unit);
  std::string trainer_data_ep =
      absl::StrCat("127.0.0.1:", *trainer_ws->local_port());
  for (size_t s = 0; s < kNumShards; ++s) {
    trainer_reg.add_shards(trainer_data_ep);
  }
  trainer_reg.set_control_plane_rpc_address(
      absl::StrCat("127.0.0.1:", *trainer_ws->listener_port()));
  for (size_t l = 0; l < kNumLayers; ++l) {
    auto* v = trainer_reg.add_variables();
    v->set_name(absl::StrCat("weights_", l));
    v->add_shape(2);
    v->add_shape(kSliceByteSize / 2);
    v->add_mesh_shape(2);
    v->add_mesh_shape(1);
    v->add_layout(1);
    v->add_layout(0);
    v->set_item_size(2);
    v->set_layer_idx(static_cast<int32_t>(l));
    for (size_t s = 0; s < kNumShards; ++s) {
      v->add_global_shard_indices(static_cast<int64_t>(s));
    }
  }
  ASSERT_OK(controller.RegisterWorkUnit(trainer_reg));

  // Register all 4 Sampler work units with the V3 controller.
  std::vector<RaidenId> sampler_units;
  for (int r = 0; r < kNumSamplers; ++r) {
    RaidenId u{"sampler", absl::StrCat(r), "weights", r};
    sampler_units.push_back(u);
    tpu_sync::rpc::RegisterWorkUnitRequest s_reg = trainer_reg;
    *s_reg.mutable_unit() = RaidenIdToProto(u);
    s_reg.clear_shards();
    std::string s_data_ep =
        absl::StrCat("127.0.0.1:", *sampler_ws[r]->local_port());
    for (size_t s = 0; s < kNumShards; ++s) {
      s_reg.add_shards(s_data_ep);
    }
    s_reg.set_control_plane_rpc_address(
        absl::StrCat("127.0.0.1:", *sampler_ws[r]->listener_port()));
    ASSERT_OK(controller.RegisterWorkUnit(s_reg));
  }

  // Verify materialized plan structure before executing:
  // - 4 variable bundles (2 layers each) in 2 stripes (D / R = 4 / 2), each
  //   pushed to 2 of the 4 samplers;
  // - one Trainer stream, so 4 waves of one push each;
  // - one command per sampler host.
  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan(
          "e2e_v3_sync", /*uuid=*/7777, {trainer_unit}, sampler_units,
          tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true);
  ASSERT_OK(plan);
  EXPECT_THAT(plan->variable_bundles, SizeIs(4));
  EXPECT_EQ(plan->seed_layout.num_stripes, 2);
  EXPECT_THAT(plan->seed_layout.stripe_seeds,
              ElementsAre(SizeIs(2), SizeIs(2)));
  EXPECT_THAT(plan->sampler_commands, SizeIs(kNumSamplers));
  EXPECT_THAT(plan->trainer_wave_commands, SizeIs(4));

  // Execute the full striped seeding + scheduled pull transfer E2E, reusing
  // the stored plan (no re-materialization).
  ASSERT_OK(controller.ExecuteTransferSync(
      "e2e_v3_sync", /*uuid=*/7777, {trainer_unit}, sampler_units,
      tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true));
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);

  // Every sampler holds all 4 bundles: 2 from the Trainer, 2 pulled.
  EXPECT_EQ(ExpectPullPhaseComplete(controller, "e2e_v3_sync"),
            kNumSamplers * 2);

  // Verify byte-for-byte weight equality across all 4 samplers, 8 layers, and
  // 2 shards!
  for (int r = 0; r < kNumSamplers; ++r) {
    for (size_t l = 0; l < kNumLayers; ++l) {
      for (size_t s = 0; s < kNumShards; ++s) {
        const uint8_t* src_ptr = trainer_ws->GetHostPointer(l, s);
        const uint8_t* dst_ptr = sampler_ws[r]->GetHostPointer(l, s);
        ASSERT_NE(dst_ptr, nullptr);
        EXPECT_EQ(std::memcmp(src_ptr, dst_ptr, kSliceByteSize), 0)
            << "Weight mismatch on sampler " << r << " layer " << l << " shard "
            << s;
      }
    }
  }

  controller.StopServer();
}

TEST(DynamicPullEngineAndControllerRpcTest,
     MaterializesMultiHostPlanAndHandlesControlPlaneRpcs) {
  RaidenControllerV3::Options opts;
  RaidenControllerV3 controller(opts);
  tpu_sync::rpc::RegisterWorkUnitRequest reg_req;
  reg_req.mutable_unit()->set_job_name("trainer");
  reg_req.mutable_unit()->set_job_replica_id("0");
  reg_req.mutable_unit()->set_data_name("weights");
  reg_req.add_shards("127.0.0.1:8000");
  reg_req.set_control_plane_rpc_address("127.0.0.1:9000");
  auto* v = reg_req.add_variables();
  v->set_name("w0");
  v->add_shape(16);
  v->add_shape(16);
  v->add_mesh_shape(1);
  v->add_mesh_shape(1);
  v->add_layout(1);
  v->add_layout(0);
  v->set_item_size(2);
  v->set_layer_idx(0);
  v->add_global_shard_indices(0);

  tpu_sync::rpc::ControlRequest ctrl_reg;
  ctrl_reg.set_command(
      tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
  *ctrl_reg.mutable_register_work_unit_request() = reg_req;
  tpu_sync::rpc::ControlResponse ctrl_resp =
      controller.HandleControlRequest(ctrl_reg);
  EXPECT_TRUE(ctrl_resp.success());

  tpu_sync::rpc::ControlRequest ctrl_meta;
  ctrl_meta.set_command(tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA);
  tpu_sync::rpc::ControlResponse meta_resp =
      controller.HandleControlRequest(ctrl_meta);
  EXPECT_TRUE(meta_resp.success());
  EXPECT_THAT(meta_resp.get_metadata_response().metadata(), SizeIs(1));

  tpu_sync::rpc::ControllerRequest c_req;
  c_req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS);
  c_req.mutable_get_transfer_status_request()->set_req_id("unknown_req");
  tpu_sync::rpc::ControllerResponse c_resp =
      controller.HandleControllerRequest(c_req);
  EXPECT_TRUE(c_resp.success());
  EXPECT_EQ(c_resp.get_transfer_status_response().status(),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED);
}

TEST(ControllerV3CacheAndOverridesTest,
     ScheduleCacheKeyDistinguishesDestinationSubsetsAndInvalidatesAnyUnit) {
  RaidenControllerV3::Options opts;
  opts.enable_plan_cache = true;
  RaidenControllerV3 controller(opts);

  auto make_reg = [](const RaidenId& unit, const std::string& shard_ep,
                     const std::string& ctrl_ep) {
    tpu_sync::rpc::RegisterWorkUnitRequest req;
    *req.mutable_unit() = RaidenIdToProto(unit);
    req.add_shards(shard_ep);
    req.set_control_plane_rpc_address(ctrl_ep);
    auto* v = req.add_variables();
    v->set_name("w0");
    v->add_shape(32);
    v->add_shape(64);
    v->add_mesh_shape(1);
    v->add_mesh_shape(1);
    v->add_layout(1);
    v->add_layout(0);
    v->set_item_size(2);
    v->set_layer_idx(0);
    v->add_global_shard_indices(0);
    return req;
  };

  RaidenId trainer{"trainer", "0", "weights", 0};
  RaidenId dst0{"sampler", "0", "weights", 0};
  RaidenId dst1{"sampler", "1", "weights", 0};
  RaidenId dst2{"sampler", "2", "weights", 0};

  ASSERT_OK(controller.RegisterWorkUnit(
      make_reg(trainer, "127.0.0.1:8000", "127.0.0.1:9000")));
  ASSERT_OK(controller.RegisterWorkUnit(
      make_reg(dst0, "127.0.0.1:8100", "127.0.0.1:9100")));
  ASSERT_OK(controller.RegisterWorkUnit(
      make_reg(dst1, "127.0.0.1:8101", "127.0.0.1:9101")));
  ASSERT_OK(controller.RegisterWorkUnit(
      make_reg(dst2, "127.0.0.1:8102", "127.0.0.1:9102")));

  ASSERT_OK(controller.BuildMaterializedPlan(
      "req_sub_01", /*uuid=*/1, {trainer}, {dst0, dst1},
      tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true, /*parallelism=*/1));
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);

  // Same num_dst_replicas (2) and same first dst_unit (dst0), but second unit
  // is dst2 instead of dst1 -> must produce a distinct ScheduleCacheKey!
  ASSERT_OK(controller.BuildMaterializedPlan(
      "req_sub_02", /*uuid=*/2, {trainer}, {dst0, dst2},
      tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true, /*parallelism=*/1));
  EXPECT_EQ(controller.GetPlanCacheSize(), 2);

  // Re-registering non-first destination unit dst2 must invalidate {dst0, dst2}
  // while keeping {dst0, dst1} in the cache.
  ASSERT_OK(controller.RegisterWorkUnit(
      make_reg(dst2, "127.0.0.1:8202", "127.0.0.1:9202")));
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);

  // Calling AttachHost on non-first destination unit dst1 must invalidate
  // {dst0, dst1}.
  ASSERT_OK(controller.AttachHost(dst1, "127.0.0.1:9301", {"127.0.0.1:8301"}));
  EXPECT_EQ(controller.GetPlanCacheSize(), 0);
}

TEST(ControllerV3CacheAndOverridesTest,
     SinglePlanMaterializationAndTtlTransferRecordEviction) {
  RaidenControllerV3::Options opts;
  opts.request_registry_ttl_s = 0.02;
  RaidenControllerV3 controller(opts);

  tpu_sync::rpc::RegisterWorkUnitRequest treq;
  treq.mutable_unit()->set_job_name("trainer");
  treq.mutable_unit()->set_job_replica_id("0");
  treq.mutable_unit()->set_data_name("weights");
  treq.add_shards("127.0.0.1:8000");
  auto* tv = treq.add_variables();
  tv->set_name("w0");
  tv->add_shape(16);
  tv->add_shape(16);
  tv->add_mesh_shape(1);
  tv->add_mesh_shape(1);
  tv->add_layout(1);
  tv->add_layout(0);
  tv->set_item_size(2);
  tv->set_layer_idx(0);
  tv->add_global_shard_indices(0);
  ASSERT_OK(controller.RegisterWorkUnit(treq));

  tpu_sync::rpc::RegisterWorkUnitRequest sreq = treq;
  sreq.mutable_unit()->set_job_name("sampler");
  sreq.clear_shards();
  sreq.add_shards("127.0.0.1:8100");
  ASSERT_OK(controller.RegisterWorkUnit(sreq));

  RaidenId trainer{"trainer", "0", "weights", 0};
  RaidenId sampler{"sampler", "0", "weights", 0};

  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 0);
  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan("req_single_mat", /*uuid=*/99, {trainer},
                                       {sampler},
                                       tpu_sync::rpc::MEMORY_TYPE_DRAM,
                                       /*skip_d2h=*/true, /*parallelism=*/1);
  ASSERT_OK(plan);
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);

  // The pull service is created when the plan is executed.
  EXPECT_EQ(controller.GetPullServiceForTest("req_single_mat"), nullptr);
  ASSERT_OK(controller.ExecuteMaterializedTransferSync("req_single_mat"));
  EXPECT_NE(controller.GetPullServiceForTest("req_single_mat"), nullptr);
  // Materialization count must remain 1 (no second materialization!).
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);
  EXPECT_EQ(controller.GetTransferRecordCount(), 1);

  absl::SleepFor(absl::Milliseconds(35));
  // Triggering a status check after TTL evicts the expired completed transfer
  // and its pull service.
  EXPECT_EQ(controller.GetTransferStatus("req_single_mat"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED);
  EXPECT_EQ(controller.GetTransferRecordCount(), 0);
  EXPECT_EQ(controller.GetPullServiceForTest("req_single_mat"), nullptr);
}

TEST(ControllerV3EndToEndTest, ExecutesSingleCopyStripesWithPromotedSources) {
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size, 0);

  constexpr size_t kNumLayers = 4;
  constexpr size_t kNumShards = 2;
  constexpr size_t kSliceByteSize = 16384;
  constexpr int kNumSamplers = 4;

  // Start V3 controller with 2 stripes of one copy each: the Trainer pushes
  // stripe 0 (bundle 0) to sampler_0 and stripe 1 (bundle 1) to sampler_2,
  // and every other copy comes from the samplers.
  RaidenControllerV3::Options opts;
  opts.port = 0;
  opts.num_bundle_groups = 2;
  opts.num_stripes = 2;
  opts.seed_replication = 1;
  opts.max_concurrent_uploads_per_source = 1;
  RaidenControllerV3 controller(opts);
  absl::StatusOr<int> ctrl_port = controller.StartServer();
  ASSERT_OK(ctrl_port);
  ASSERT_THAT(*ctrl_port, Gt(0));

  auto trainer_ws = std::make_unique<WeightSynchronizerBase>(
      kNumLayers, kNumShards, kSliceByteSize,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/2, /*listener_port=*/0,
      /*bind_ip=*/"127.0.0.1", /*layer_names=*/std::vector<std::string>{},
      /*auto_h2d=*/false);
  ASSERT_TRUE(trainer_ws->local_port().has_value());
  ASSERT_TRUE(trainer_ws->listener_port().has_value());

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t s = 0; s < kNumShards; ++s) {
      uint8_t* ptr = const_cast<uint8_t*>(trainer_ws->GetHostPointer(l, s));
      ASSERT_NE(ptr, nullptr);
      for (size_t i = 0; i < kSliceByteSize; ++i) {
        ptr[i] = static_cast<uint8_t>((l * 41 + s * 17 + i + 3) & 0xFF);
      }
    }
  }

  std::vector<std::unique_ptr<WeightSynchronizerBase>> sampler_ws(kNumSamplers);
  for (int r = 0; r < kNumSamplers; ++r) {
    sampler_ws[r] = std::make_unique<WeightSynchronizerBase>(
        kNumLayers, kNumShards, kSliceByteSize,
        /*local_port=*/0, /*host_blocks_to_allocate=*/1,
        /*parallelism=*/2, /*listener_port=*/0,
        /*bind_ip=*/"127.0.0.1", /*layer_names=*/std::vector<std::string>{},
        /*auto_h2d=*/false);
    ASSERT_TRUE(sampler_ws[r]->local_port().has_value());
    ASSERT_TRUE(sampler_ws[r]->listener_port().has_value());
    for (size_t l = 0; l < kNumLayers; ++l) {
      for (size_t s = 0; s < kNumShards; ++s) {
        uint8_t* ptr =
            const_cast<uint8_t*>(sampler_ws[r]->GetHostPointer(l, s));
        ASSERT_NE(ptr, nullptr);
        std::memset(ptr, 0, kSliceByteSize);
      }
    }
  }

  RaidenId trainer_unit{"trainer", "0", "weights", 0};
  tpu_sync::rpc::RegisterWorkUnitRequest trainer_reg;
  *trainer_reg.mutable_unit() = RaidenIdToProto(trainer_unit);
  std::string trainer_data_ep =
      absl::StrCat("127.0.0.1:", *trainer_ws->local_port());
  for (size_t s = 0; s < kNumShards; ++s) {
    trainer_reg.add_shards(trainer_data_ep);
  }
  trainer_reg.set_control_plane_rpc_address(
      absl::StrCat("127.0.0.1:", *trainer_ws->listener_port()));
  for (size_t l = 0; l < kNumLayers; ++l) {
    auto* v = trainer_reg.add_variables();
    v->set_name(absl::StrCat("weights_", l));
    v->add_shape(2);
    v->add_shape(kSliceByteSize / 2);
    v->add_mesh_shape(2);
    v->add_mesh_shape(1);
    v->add_layout(1);
    v->add_layout(0);
    v->set_item_size(2);
    v->set_layer_idx(static_cast<int32_t>(l));
    for (size_t s = 0; s < kNumShards; ++s) {
      v->add_global_shard_indices(static_cast<int64_t>(s));
    }
  }
  ASSERT_OK(controller.RegisterWorkUnit(trainer_reg));

  std::vector<RaidenId> sampler_units;
  for (int r = 0; r < kNumSamplers; ++r) {
    RaidenId u{"sampler", absl::StrCat(r), "weights", r};
    sampler_units.push_back(u);
    tpu_sync::rpc::RegisterWorkUnitRequest s_reg = trainer_reg;
    *s_reg.mutable_unit() = RaidenIdToProto(u);
    s_reg.clear_shards();
    std::string s_data_ep =
        absl::StrCat("127.0.0.1:", *sampler_ws[r]->local_port());
    for (size_t s = 0; s < kNumShards; ++s) {
      s_reg.add_shards(s_data_ep);
    }
    s_reg.set_control_plane_rpc_address(
        absl::StrCat("127.0.0.1:", *sampler_ws[r]->listener_port()));
    ASSERT_OK(controller.RegisterWorkUnit(s_reg));
  }

  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan(
          "e2e_v3_stripes", /*uuid=*/8888, {trainer_unit}, sampler_units,
          tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true);
  ASSERT_OK(plan);
  EXPECT_THAT(plan->seed_layout.stripe_seeds,
              ElementsAre(ElementsAre(0), ElementsAre(2)));

  ASSERT_OK(controller.ExecuteTransferSync(
      "e2e_v3_stripes", /*uuid=*/8888, {trainer_unit}, sampler_units,
      tpu_sync::rpc::MEMORY_TYPE_DRAM, /*skip_d2h=*/true));
  // Samplers 0 and 2 pull one bundle each, samplers 1 and 3 both.
  EXPECT_EQ(ExpectPullPhaseComplete(controller, "e2e_v3_stripes"), 6);

  for (int r = 0; r < kNumSamplers; ++r) {
    for (size_t l = 0; l < kNumLayers; ++l) {
      for (size_t s = 0; s < kNumShards; ++s) {
        const uint8_t* src_ptr = trainer_ws->GetHostPointer(l, s);
        const uint8_t* dst_ptr = sampler_ws[r]->GetHostPointer(l, s);
        ASSERT_NE(dst_ptr, nullptr);
        EXPECT_EQ(std::memcmp(src_ptr, dst_ptr, kSliceByteSize), 0)
            << "Weight mismatch on sampler " << r << " layer " << l << " shard "
            << s;
      }
    }
  }

  controller.StopServer();
}

TEST(ControllerV3CacheAndOverridesTest,
     HandlesControlRequestStartTransferCommand) {
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [](absl::string_view /*ep*/,
                              const tpu_sync::rpc::ControlRequest& /*req*/) {
    tpu_sync::rpc::ControlResponse resp;
    resp.set_success(true);
    return resp;
  };
  RaidenControllerV3 controller(opts);

  tpu_sync::rpc::RegisterWorkUnitRequest treq;
  treq.mutable_unit()->set_job_name("trainer");
  treq.mutable_unit()->set_job_replica_id("0");
  treq.mutable_unit()->set_data_name("weights");
  treq.add_shards("127.0.0.1:8000");
  treq.set_control_plane_rpc_address("127.0.0.1:9000");
  auto* tv = treq.add_variables();
  tv->set_name("w0");
  tv->add_shape(16);
  tv->add_shape(16);
  tv->add_mesh_shape(1);
  tv->add_mesh_shape(1);
  tv->add_layout(1);
  tv->add_layout(0);
  tv->set_item_size(2);
  tv->set_layer_idx(0);
  tv->add_global_shard_indices(0);
  ASSERT_OK(controller.RegisterWorkUnit(treq));

  tpu_sync::rpc::RegisterWorkUnitRequest sreq = treq;
  sreq.mutable_unit()->set_job_name("sampler");
  sreq.clear_shards();
  sreq.add_shards("127.0.0.1:8100");
  sreq.set_control_plane_rpc_address("127.0.0.1:9100");
  ASSERT_OK(controller.RegisterWorkUnit(sreq));

  tpu_sync::rpc::ControlRequest start_ctrl_req;
  start_ctrl_req.set_command(
      tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER);
  auto* st = start_ctrl_req.mutable_start_transfer_request();
  *st->add_src_units() = treq.unit();
  *st->add_dst_units() = sreq.unit();
  st->set_req_id("ctrl_req_start_1");
  st->set_uuid(777);
  st->set_skip_d2h(true);

  tpu_sync::rpc::ControlResponse resp =
      controller.HandleControlRequest(start_ctrl_req);
  EXPECT_TRUE(resp.success()) << resp.message();
  ASSERT_OK(controller.WaitForTransfer("ctrl_req_start_1", absl::Seconds(5)));
  EXPECT_EQ(controller.GetTransferStatus("ctrl_req_start_1"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED);
}

TEST(ControllerV3CacheAndOverridesTest,
     StartTransferAsyncMarksRecordFailedWhenPlanningFailsEarly) {
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [](absl::string_view /*ep*/,
                              const tpu_sync::rpc::ControlRequest& /*req*/) {
    tpu_sync::rpc::ControlResponse resp;
    resp.set_success(true);
    return resp;
  };
  RaidenControllerV3 controller(opts);

  // Nothing is registered: ExecuteTransferSync fails before it ever records a
  // terminal status itself. The async wrapper must still mark the pre-created
  // IN_PROGRESS record as FAILED so pollers do not hang.
  RaidenId ghost_src{"trainer", "ghost", "weights", 0};
  RaidenId ghost_dst{"sampler", "ghost", "weights", 0};
  ASSERT_OK(controller.StartTransferAsync("async_early_fail", /*uuid=*/5,
                                          {ghost_src}, {ghost_dst}));
  absl::Status waited =
      controller.WaitForTransfer("async_early_fail", absl::Seconds(5));
  EXPECT_FALSE(waited.ok());
  EXPECT_NE(waited.code(), absl::StatusCode::kDeadlineExceeded) << waited;
  EXPECT_EQ(controller.GetTransferStatus("async_early_fail"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED);
}

// Registers a single-host unit with one 32x64 variable. An empty |ctrl_ep|
// registers no control endpoint (the unit then receives no commands).
tpu_sync::rpc::RegisterWorkUnitRequest MakeSimpleReg(const RaidenId& unit,
                                                     int idx,
                                                     bool with_ctrl = true) {
  tpu_sync::rpc::RegisterWorkUnitRequest req;
  *req.mutable_unit() = RaidenIdToProto(unit);
  req.add_shards(absl::StrCat("127.0.0.1:", 8100 + idx));
  if (with_ctrl) {
    req.set_control_plane_rpc_address(absl::StrCat("127.0.0.1:", 9100 + idx));
  }
  auto* v = req.add_variables();
  v->set_name("w0");
  v->add_shape(32);
  v->add_shape(64);
  v->add_mesh_shape(1);
  v->add_mesh_shape(1);
  v->add_layout(1);
  v->add_layout(0);
  v->set_item_size(2);
  v->set_layer_idx(0);
  v->add_global_shard_indices(0);
  return req;
}

absl::StatusOr<tpu_sync::rpc::ControlResponse> OkResponse() {
  tpu_sync::rpc::ControlResponse resp;
  resp.set_success(true);
  return resp;
}

// Registers trainer (idx 0) and |num_samplers| samplers (idx 1..n).
std::vector<RaidenId> RegisterSimpleTopology(RaidenControllerV3& controller,
                                             int num_samplers,
                                             bool with_ctrl = true) {
  RaidenId trainer{"trainer", "0", "weights", 0};
  EXPECT_OK(controller.RegisterWorkUnit(MakeSimpleReg(trainer, 0, with_ctrl)));
  std::vector<RaidenId> samplers;
  for (int i = 0; i < num_samplers; ++i) {
    RaidenId s{"sampler", absl::StrCat(i), "weights", 0};
    EXPECT_OK(controller.RegisterWorkUnit(MakeSimpleReg(s, i + 1, with_ctrl)));
    samplers.push_back(s);
  }
  return samplers;
}

// Calls `ReportPulls` and waits for the reply (long poll 0 unless set).
PullServiceReply CallController(RaidenControllerV3& controller,
                                PullServiceRequest request) {
  absl::Notification done;
  PullServiceReply out;
  controller.ReportPulls(std::move(request), [&](PullServiceReply reply) {
    out = std::move(reply);
    done.Notify();
  });
  done.WaitForNotification();
  return out;
}

// Waits until the current execution of |req_id| has a pull service.
bool WaitForPullService(const RaidenControllerV3& controller,
                        absl::string_view req_id) {
  const absl::Time deadline = absl::Now() + absl::Seconds(30);
  while (controller.GetPullServiceForTest(req_id) == nullptr) {
    if (absl::Now() > deadline) return false;
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

// Replica indices the Trainer seeds in |plan|.
absl::flat_hash_set<int32_t> SeedReplicas(
    const MaterializedTransferPlan& plan) {
  absl::flat_hash_set<int32_t> seeds;
  for (const std::vector<int32_t>& stripe : plan.seed_layout.stripe_seeds) {
    seeds.insert(stripe.begin(), stripe.end());
  }
  return seeds;
}

TEST(ControllerV3RegressionTest, EvictionDropsPullServices) {
  RaidenControllerV3::Options opts;
  opts.request_registry_ttl_s = 0.2;
  opts.min_unobserved_ttl_s = 1.0;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  // Terminal record: pull service dropped together with the record.
  ASSERT_OK(controller.ExecuteTransferSync("b1_done", /*uuid=*/1, {trainer},
                                           samplers));
  std::weak_ptr<TransferPullService> done =
      controller.GetPullServiceForTest("b1_done");
  EXPECT_FALSE(done.expired());
  absl::SleepFor(absl::Milliseconds(300));
  EXPECT_EQ(controller.GetTransferRecordCount(), 0);
  EXPECT_TRUE(done.expired());
  PullServiceReply reply = CallController(
      controller, {.req_id = "b1_done", .uuid = 1, .unit = samplers[0]});
  EXPECT_EQ(reply.kind, PullReplyKind::kAborted);
  EXPECT_EQ(reply.status.code(), absl::StatusCode::kNotFound);

  // Never-executed plan (no pull service): kept for max(ttl,
  // min_unobserved_ttl_s), then evicted.
  ASSERT_OK(controller.BuildMaterializedPlan("b1_pending", /*uuid=*/2,
                                             {trainer}, samplers));
  EXPECT_EQ(controller.GetPullServiceForTest("b1_pending"), nullptr);
  absl::SleepFor(absl::Milliseconds(300));
  EXPECT_EQ(controller.GetTransferRecordCount(), 1)
      << "pending plan evicted by the terminal-record TTL";
  absl::SleepFor(absl::Milliseconds(900));
  EXPECT_EQ(controller.GetTransferRecordCount(), 0);
  EXPECT_EQ(controller.ExecuteMaterializedTransferSync("b1_pending").code(),
            absl::StatusCode::kNotFound);
}

TEST(ControllerV3RegressionTest, ZeroTtlStillReportsTransferResults) {
  RaidenControllerV3::Options opts;
  opts.request_registry_ttl_s = 0.0;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  EXPECT_OK(controller.ExecuteTransferSync("zero_sync", /*uuid=*/1, {trainer},
                                           samplers));

  // The pending plan survives a zero TTL (min_unobserved_ttl_s default).
  ASSERT_OK(controller.BuildMaterializedPlan("zero_mat", /*uuid=*/2, {trainer},
                                             samplers));
  EXPECT_EQ(controller.GetRetainedPlanUuids().count("zero_mat"), 1);
  EXPECT_OK(controller.ExecuteMaterializedTransferSync("zero_mat"));

  ASSERT_OK(controller.StartTransferAsync("zero_async", /*uuid=*/3, {trainer},
                                          samplers));
  EXPECT_OK(controller.WaitForTransfer("zero_async", absl::Seconds(10)));

  // Finished records are gone at the next eviction pass, pull services too.
  EXPECT_EQ(controller.GetTransferStatus("zero_async"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED);
  EXPECT_EQ(controller.GetTransferRecordCount(), 0);
  EXPECT_EQ(controller.GetPullServiceForTest("zero_async"), nullptr);
  EXPECT_EQ(controller.WaitForTransfer("zero_async", absl::Seconds(1)).code(),
            absl::StatusCode::kNotFound);
}

TEST(ControllerV3RegressionTest, PullOptionsAreExecutionOptions) {
  RaidenControllerV3::Options opts;
  opts.grant_batch_size = 0;                   // Clamped to 1.
  opts.max_concurrent_uploads_per_source = 0;  // Clamped to 1.
  opts.lease_timeout_ms = 0;                   // Clamped to 1.
  opts.long_poll_timeout_ms = -5;              // Clamped to 1.
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  EXPECT_EQ(controller.grant_batch_size(), 1);
  EXPECT_EQ(controller.max_concurrent_uploads_per_source(), 1);
  EXPECT_EQ(controller.lease_timeout_ms(), 1);
  EXPECT_EQ(controller.long_poll_timeout_ms(), 1);
  // A 1ms lease may expire before its pull lands.
  controller.set_lease_timeout_ms(30000);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 4);
  ASSERT_OK(
      controller.ExecuteTransferSync("b3", /*uuid=*/1, {trainer}, samplers));

  // The options apply to the next execution without re-materializing.
  controller.set_grant_batch_size(3);
  controller.set_max_concurrent_uploads_per_source(5);
  controller.set_lease_timeout_ms(1234);
  controller.set_long_poll_timeout_ms(777);
  EXPECT_EQ(controller.grant_batch_size(), 3);
  EXPECT_EQ(controller.max_concurrent_uploads_per_source(), 5);
  EXPECT_EQ(controller.lease_timeout_ms(), 1234);
  EXPECT_EQ(controller.long_poll_timeout_ms(), 777);
  ASSERT_OK(
      controller.ExecuteTransferSync("b3", /*uuid=*/1, {trainer}, samplers));
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);
  std::shared_ptr<TransferPullService> service =
      controller.GetPullServiceForTest("b3");
  ASSERT_NE(service, nullptr);
  const PullSchedulerOptions& applied = service->scheduler().options();
  EXPECT_EQ(applied.grant_batch_size, 3);
  EXPECT_EQ(applied.max_concurrent_uploads_per_source, 5);
  EXPECT_EQ(applied.lease_timeout, absl::Milliseconds(1234));
  EXPECT_EQ(applied.long_poll_timeout, absl::Milliseconds(777));
}

TEST(ControllerV3RegressionTest, ReExecutingStoredPlanPullsAgain) {
  std::atomic<int> donor_pushes{0};
  RaidenControllerV3::Options opts;
  // One copy of the single stripe: sampler 0 is the only seed.
  opts.seed_replication = 1;
  opts.custom_rpc_sender = [&](absl::string_view ep,
                               const tpu_sync::rpc::ControlRequest& req) {
    // Sampler 0 (control port 9101) is the seed and the only donor.
    if (req.start_transfer_request().is_sender() && ep == "127.0.0.1:9101") {
      ++donor_pushes;
    }
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan("rerun", /*uuid=*/1, {trainer},
                                       samplers);
  ASSERT_OK(plan);
  ASSERT_THAT(plan->seed_layout.stripe_seeds, ElementsAre(ElementsAre(0)));
  const int num_bundles = static_cast<int>(plan->variable_bundles.size());
  ASSERT_GT(num_bundles, 0);

  ASSERT_OK(controller.ExecuteMaterializedTransferSync("rerun"));
  EXPECT_EQ(donor_pushes.load(), num_bundles);
  std::shared_ptr<TransferPullService> first =
      controller.GetPullServiceForTest("rerun");
  ASSERT_NE(first, nullptr);
  // The second run gets a fresh scheduler (the completed one would treat
  // every bundle as held and skip moving any data).
  ASSERT_OK(controller.ExecuteMaterializedTransferSync("rerun"));
  EXPECT_EQ(donor_pushes.load(), 2 * num_bundles);
  EXPECT_NE(controller.GetPullServiceForTest("rerun"), first);
  // Same for the ExecuteTransferSync plan-reuse path.
  ASSERT_OK(
      controller.ExecuteTransferSync("rerun", /*uuid=*/1, {trainer}, samplers));
  EXPECT_EQ(donor_pushes.load(), 3 * num_bundles);
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);
  EXPECT_EQ(ExpectPullPhaseComplete(controller, "rerun"), num_bundles);
}

TEST(ControllerV3RegressionTest, RejectsReplanningAndReexecutingInProgress) {
  absl::Notification entered;
  absl::Notification release;
  std::atomic<bool> first_rpc{false};
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [&](absl::string_view,
                               const tpu_sync::rpc::ControlRequest&) {
    if (!first_rpc.exchange(true)) entered.Notify();
    release.WaitForNotification();
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  ASSERT_OK(
      controller.StartTransferAsync("busy", /*uuid=*/1, {trainer}, samplers));
  entered.WaitForNotification();
  EXPECT_EQ(controller.GetTransferStatus("busy"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS);
  EXPECT_EQ(controller.BuildMaterializedPlan("busy", 2, {trainer}, samplers)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(
      controller.ExecuteTransferSync("busy", 1, {trainer}, samplers).code(),
      absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(controller.ExecuteMaterializedTransferSync("busy").code(),
            absl::StatusCode::kFailedPrecondition);
  // A duplicate async start is a no-op.
  EXPECT_OK(controller.StartTransferAsync("busy", 1, {trainer}, samplers));
  release.Notify();
  EXPECT_OK(controller.WaitForTransfer("busy", absl::Seconds(10)));
  EXPECT_EQ(controller.GetTransferStatus("busy"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED);
}

TEST(ControllerV3RegressionTest, ZeroTtlKeepsUnobservedResultsUntilPolled) {
  RaidenControllerV3::Options opts;
  opts.request_registry_ttl_s = 0.0;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  ASSERT_OK(controller.StartTransferAsync("unobserved", /*uuid=*/1, {trainer},
                                          samplers));
  controller.StopServer();  // Joins the async transfer without observing it.
  // Another transfer finishing runs eviction passes; the unobserved result of
  // "unobserved" must survive them.
  ASSERT_OK(
      controller.ExecuteTransferSync("other", /*uuid=*/2, {trainer}, samplers));
  EXPECT_EQ(controller.GetTransferRecordCount(), 1);
  EXPECT_EQ(controller.GetTransferStatus("unobserved"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED);
  // Observed now: evicted (with its pull service) at the next pass.
  EXPECT_EQ(controller.GetTransferStatus("unobserved"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED);
  EXPECT_EQ(controller.GetPullServiceForTest("unobserved"), nullptr);

  // Results nobody ever observes are still bounded by min_unobserved_ttl_s.
  RaidenControllerV3::Options bounded_opts = opts;
  bounded_opts.min_unobserved_ttl_s = 0.1;
  RaidenControllerV3 bounded(bounded_opts);
  samplers = RegisterSimpleTopology(bounded, 2);
  ASSERT_OK(
      bounded.StartTransferAsync("forgotten", /*uuid=*/3, {trainer}, samplers));
  bounded.StopServer();
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_EQ(bounded.GetTransferRecordCount(), 0);
  EXPECT_EQ(bounded.GetPullServiceForTest("forgotten"), nullptr);
}

TEST(ControllerV3RegressionTest, GeneratedIdsAreUniqueAcrossEntryPoints) {
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan("", /*uuid=*/0, {trainer}, samplers);
  ASSERT_OK(plan);
  EXPECT_GT(plan->uuid, 0);
  EXPECT_EQ(plan->req_id, absl::StrCat("req_", plan->uuid));
  // A server-side transfer with generated ids to a different destination set
  // must not replace (or reuse) the stored plan.
  ASSERT_OK(
      controller.StartTransferAsync("", /*uuid=*/0, {trainer}, {samplers[0]}));
  controller.StopServer();
  absl::flat_hash_map<std::string, uint64_t> retained =
      controller.GetRetainedPlanUuids();
  EXPECT_EQ(retained.size(), 2);
  ASSERT_TRUE(retained.contains(plan->req_id));
  EXPECT_EQ(retained.at(plan->req_id), plan->uuid);
}

TEST(ControllerV3RegressionTest, ExecuteMaterializedChecksExpectedUuid) {
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  ASSERT_OK(controller.BuildMaterializedPlan("pinned", /*uuid=*/1, {trainer},
                                             samplers));
  ASSERT_OK(controller.BuildMaterializedPlan("pinned", /*uuid=*/2, {trainer},
                                             samplers));
  EXPECT_EQ(
      controller.ExecuteMaterializedTransferSync("pinned", /*expected_uuid=*/1)
          .code(),
      absl::StatusCode::kFailedPrecondition);
  // The rejected call did not claim or touch the record.
  EXPECT_EQ(controller.GetTransferStatus("pinned"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_NOT_STARTED);
  EXPECT_OK(controller.ExecuteMaterializedTransferSync("pinned",
                                                       /*expected_uuid=*/2));
}

// Records the receiver (`is_sender == false`) `StartTransferRequest`s sent to
// workers and answers every RPC with success.
class ReceiverRecorder {
 public:
  DynamicPullEngine::RpcSenderFn Sender() {
    return [this](absl::string_view, const tpu_sync::rpc::ControlRequest& req) {
      if (req.command() ==
              tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER &&
          !req.start_transfer_request().is_sender()) {
        absl::MutexLock lock(mu_);
        receivers_.push_back(req.start_transfer_request());
      }
      return OkResponse();
    };
  }

  // Returns and clears the recorded requests.
  std::vector<tpu_sync::rpc::StartTransferRequest> Take() {
    absl::MutexLock lock(mu_);
    return std::exchange(receivers_, {});
  }

 private:
  absl::Mutex mu_;
  std::vector<tpu_sync::rpc::StartTransferRequest> receivers_;
};

TEST(ControllerV3RegressionTest, StoredPlanIsReusedOnlyForMatchingOptions) {
  ReceiverRecorder recorder;
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = recorder.Sender();
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);
  constexpr auto kDram = tpu_sync::rpc::MEMORY_TYPE_DRAM;
  constexpr auto kHbm = tpu_sync::rpc::MEMORY_TYPE_HBM;

  // Checks the receiver commands (seed + puller) of the last execution.
  auto expect_receivers = [&](tpu_sync::rpc::MemoryType mem_type, bool skip_d2h,
                              int32_t parallelism, bool skip_tiling_layer0) {
    std::vector<tpu_sync::rpc::StartTransferRequest> reqs = recorder.Take();
    ASSERT_THAT(reqs, SizeIs(2));
    for (const tpu_sync::rpc::StartTransferRequest& r : reqs) {
      EXPECT_EQ(r.dst_mem_type(), mem_type);
      EXPECT_EQ(r.skip_d2h(), skip_d2h);
      EXPECT_EQ(r.parallelism(), parallelism);
      ASSERT_TRUE(r.skip_tiling().contains(0));
      EXPECT_EQ(r.skip_tiling().at(0), skip_tiling_layer0);
    }
  };
  auto materializations = [&] {
    return controller.GetPlanMaterializationCountForTest();
  };

  ASSERT_OK(controller.BuildMaterializedPlan("opts", /*uuid=*/1, {trainer},
                                             samplers, kDram,
                                             /*skip_d2h=*/false,
                                             /*parallelism=*/1));
  EXPECT_EQ(materializations(), 1);
  // Same options: the stored plan is reused.
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers,
                                           kDram, false, 1));
  EXPECT_EQ(materializations(), 1);
  expect_receivers(kDram, false, 1, false);

  // Each plan-affecting option forces a new plan that carries the new value.
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           false, 1));
  EXPECT_EQ(materializations(), 2);
  expect_receivers(kHbm, false, 1, false);
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           false, 4));
  EXPECT_EQ(materializations(), 3);
  expect_receivers(kHbm, false, 4, false);
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 4));
  EXPECT_EQ(materializations(), 4);
  expect_receivers(kHbm, true, 4, false);
  // Memory type, D2H and parallelism do not change the logical schedule, so
  // the schedule cache still has a single entry; skip_tiling does.
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 4, {{0, true}}));
  EXPECT_EQ(materializations(), 5);
  expect_receivers(kHbm, true, 4, true);
  EXPECT_EQ(controller.GetPlanCacheSize(), 2);

  // Unchanged options reuse the replacement plan, which is also what
  // `ExecuteMaterializedTransferSync` now runs.
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 4, {{0, true}}));
  EXPECT_EQ(materializations(), 5);
  expect_receivers(kHbm, true, 4, true);
  ASSERT_OK(controller.ExecuteMaterializedTransferSync("opts"));
  EXPECT_EQ(materializations(), 5);
  expect_receivers(kHbm, true, 4, true);

  // `false` skip_tiling entries are equivalent to absent ones.
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 4, {{0, false}}));
  EXPECT_EQ(materializations(), 6);
  expect_receivers(kHbm, true, 4, false);
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 4, {}));
  EXPECT_EQ(materializations(), 6);
  expect_receivers(kHbm, true, 4, false);

  // Parallelism is compared after clamping to >= 1, like the commands.
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 1));
  EXPECT_EQ(materializations(), 7);
  recorder.Take();
  ASSERT_OK(controller.ExecuteTransferSync("opts", 1, {trainer}, samplers, kHbm,
                                           true, 0));
  EXPECT_EQ(materializations(), 7);
  expect_receivers(kHbm, true, 1, false);

  // Server-initiated (async) transfers follow the same rule.
  ASSERT_OK(controller.BuildMaterializedPlan("opts_async", /*uuid=*/2,
                                             {trainer}, samplers, kDram));
  EXPECT_EQ(materializations(), 8);
  ASSERT_OK(controller.StartTransferAsync("opts_async", 2, {trainer}, samplers,
                                          kHbm));
  ASSERT_OK(controller.WaitForTransfer("opts_async", absl::Seconds(10)));
  EXPECT_EQ(materializations(), 9);
  expect_receivers(kHbm, false, 1, false);
}

TEST(ControllerV3RegressionTest,
     StoredPlanIsNotReusedAfterRegistrationOrSettingsChange) {
  absl::Mutex mu;
  std::vector<std::string> receiver_endpoints;
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [&](absl::string_view ep,
                               const tpu_sync::rpc::ControlRequest& req) {
    if (req.command() ==
            tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER &&
        !req.start_transfer_request().is_sender()) {
      absl::MutexLock lock(mu);
      receiver_endpoints.push_back(std::string(ep));
    }
    return OkResponse();
  };
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);
  auto take_endpoints = [&] {
    absl::MutexLock lock(mu);
    std::vector<std::string> out = std::exchange(receiver_endpoints, {});
    std::sort(out.begin(), out.end());
    return out;
  };
  auto materializations = [&] {
    return controller.GetPlanMaterializationCountForTest();
  };
  auto execute = [&] {
    return controller.ExecuteTransferSync("inputs", /*uuid=*/1, {trainer},
                                          samplers);
  };

  ASSERT_OK(controller.BuildMaterializedPlan("inputs", /*uuid=*/1, {trainer},
                                             samplers));
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 1);
  EXPECT_THAT(take_endpoints(),
              ElementsAre("127.0.0.1:9101", "127.0.0.1:9102"));

  // Sampler 1 restarts with a new control endpoint. The stored plan still
  // targets the old endpoint and must not be reused.
  tpu_sync::rpc::RegisterWorkUnitRequest moved = MakeSimpleReg(samplers[1], 2);
  moved.set_control_plane_rpc_address("127.0.0.1:9200");
  ASSERT_OK(controller.RegisterWorkUnit(moved));
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 2);
  EXPECT_THAT(take_endpoints(),
              ElementsAre("127.0.0.1:9101", "127.0.0.1:9200"));
  // An identical re-registration changes nothing: the plan is reused.
  ASSERT_OK(controller.RegisterWorkUnit(moved));
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 2);
  EXPECT_THAT(take_endpoints(),
              ElementsAre("127.0.0.1:9101", "127.0.0.1:9200"));

  // So do changes of the controller settings the plan is built from.
  controller.set_num_stripes(1);
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 3);
  controller.set_seed_replication(1);
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 4);
  controller.set_num_bundle_groups(controller.num_bundle_groups() + 1);
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 5);
  controller.set_broadcast_host_ratio(0.5);
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 6);
  // Pull options are execution options: the plan is reused.
  controller.set_grant_batch_size(2);
  controller.set_max_concurrent_uploads_per_source(2);
  controller.set_lease_timeout_ms(1000);
  controller.set_long_poll_timeout_ms(1000);
  ASSERT_OK(execute());
  EXPECT_EQ(materializations(), 6);
}

// Registers trainer (idx 0) and |num_samplers| samplers (idx 1..n), each
// with |num_layers| 32x64 variables.
std::vector<RaidenId> RegisterLayeredTopology(RaidenControllerV3& controller,
                                              int num_samplers,
                                              int num_layers) {
  auto make_reg = [num_layers](const RaidenId& unit, int idx) {
    tpu_sync::rpc::RegisterWorkUnitRequest req = MakeSimpleReg(unit, idx);
    req.clear_variables();
    for (int l = 0; l < num_layers; ++l) {
      auto* v = req.add_variables();
      v->set_name(absl::StrCat("w", l));
      v->add_shape(32);
      v->add_shape(64);
      v->add_mesh_shape(1);
      v->add_mesh_shape(1);
      v->add_layout(1);
      v->add_layout(0);
      v->set_item_size(2);
      v->set_layer_idx(l);
      v->add_global_shard_indices(0);
    }
    return req;
  };
  RaidenId trainer{"trainer", "0", "weights", 0};
  EXPECT_OK(controller.RegisterWorkUnit(make_reg(trainer, 0)));
  std::vector<RaidenId> samplers;
  for (int i = 0; i < num_samplers; ++i) {
    RaidenId s{"sampler", absl::StrCat(i), "weights", 0};
    EXPECT_OK(controller.RegisterWorkUnit(make_reg(s, i + 1)));
    samplers.push_back(s);
  }
  return samplers;
}

TEST(ControllerV3SeedingTest, SeedingSettingsChangeTheScheduleCacheKey) {
  RaidenControllerV3::Options opts;
  opts.num_bundle_groups = 4;
  RaidenControllerV3 controller(opts);
  EXPECT_EQ(controller.num_stripes(), 0);
  EXPECT_EQ(controller.seed_replication(), 2);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterLayeredTopology(controller, 4, 4);

  auto seed_layout = [&](absl::string_view req_id) {
    absl::StatusOr<MaterializedTransferPlan> plan =
        controller.BuildMaterializedPlan(req_id, /*uuid=*/0, {trainer},
                                         samplers);
    EXPECT_OK(plan);
    return plan.ok() ? plan->seed_layout : SeedLayout();
  };

  // Auto: G = min(B, D / R) = 2 stripes of 2 copies.
  SeedLayout layout = seed_layout("auto");
  EXPECT_EQ(layout.num_stripes, 2);
  EXPECT_EQ(layout.replication, 2);
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);
  layout = seed_layout("auto");
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);

  // One copy per stripe: G = min(4, 4 / 1) = 4.
  controller.set_seed_replication(1);
  EXPECT_EQ(controller.GetPlanCacheSize(), 0);
  layout = seed_layout("r1");
  EXPECT_EQ(layout.num_stripes, 4);
  EXPECT_EQ(layout.replication, 1);
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);

  // An explicit stripe count.
  controller.set_num_stripes(1);
  EXPECT_EQ(controller.GetPlanCacheSize(), 0);
  layout = seed_layout("g1");
  EXPECT_EQ(layout.num_stripes, 1);
  EXPECT_THAT(layout.stripe_bundles, ElementsAre(ElementsAre(0, 1, 2, 3)));
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);
}

TEST(ControllerV3SeedingTest, SeedsFollowTheUnitsNotTheirOrder) {
  RaidenControllerV3::Options opts;
  opts.num_bundle_groups = 4;
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterLayeredTopology(controller, 8, 4);
  std::vector<RaidenId> reversed(samplers.rbegin(), samplers.rend());

  // Units seeded with each stripe.
  auto seeded_units = [&](absl::string_view req_id,
                          const std::vector<RaidenId>& dst) {
    std::vector<std::vector<std::string>> out;
    absl::StatusOr<MaterializedTransferPlan> plan =
        controller.BuildMaterializedPlan(req_id, /*uuid=*/0, {trainer}, dst);
    EXPECT_OK(plan);
    if (!plan.ok()) return out;
    for (const std::vector<int32_t>& seeds : plan->seed_layout.stripe_seeds) {
      std::vector<std::string> units;
      for (int32_t r : seeds) units.push_back(dst[r].job_replica_id);
      std::sort(units.begin(), units.end());
      out.push_back(std::move(units));
    }
    return out;
  };
  const std::vector<std::vector<std::string>> forward =
      seeded_units("forward", samplers);
  EXPECT_THAT(forward, SizeIs(4));
  EXPECT_EQ(seeded_units("reversed", reversed), forward);
}

// Options whose samplers never ask for pulls (scheduled pull phase), so
// every transfer runs until its deadline.
RaidenControllerV3::Options SilentSamplerOptions(int64_t transfer_timeout_ms) {
  RaidenControllerV3::Options opts;
  opts.transfer_timeout_ms = transfer_timeout_ms;
  opts.pull_phase = RaidenControllerV3::PullPhaseKind::kScheduled;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  return opts;
}

TEST(ControllerV3TransferTimeoutTest,
     ExpiredTransferFailsWithDeadlineExceeded) {
  RaidenControllerV3 controller(SilentSamplerOptions(200));
  EXPECT_EQ(controller.transfer_timeout_ms(), 200);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  absl::Status status = controller.ExecuteTransferSync("expired", /*uuid=*/1,
                                                       {trainer}, samplers);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer expired passed its deadline"));
  EXPECT_EQ(controller.GetTransferStatus("expired"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED);
  // Samplers still asking learn that the transfer failed.
  std::shared_ptr<TransferPullService> service =
      controller.GetPullServiceForTest("expired");
  ASSERT_NE(service, nullptr);
  EXPECT_EQ(service->scheduler().status().code(),
            absl::StatusCode::kDeadlineExceeded);
  PullServiceReply reply = CallController(
      controller,
      {.req_id = "expired", .uuid = 1, .unit = samplers[1], .max_grants = 1});
  EXPECT_EQ(reply.kind, PullReplyKind::kAborted);
  EXPECT_EQ(reply.status.code(), absl::StatusCode::kDeadlineExceeded);

  // An RPC error after the deadline (e.g. an RPC cut short by it) is reported
  // as the deadline.
  RaidenControllerV3::Options opts;
  opts.transfer_timeout_ms = 100;
  opts.custom_rpc_sender = [](absl::string_view ep,
                              const tpu_sync::rpc::ControlRequest& req)
      -> absl::StatusOr<tpu_sync::rpc::ControlResponse> {
    if (req.start_transfer_request().is_sender() && ep == "127.0.0.1:9100") {
      absl::SleepFor(absl::Milliseconds(300));
      return absl::UnavailableError("Trainer push cut short");
    }
    return OkResponse();
  };
  RaidenControllerV3 slow(opts);
  samplers = RegisterSimpleTopology(slow, 2);
  status =
      slow.ExecuteTransferSync("cut_short", /*uuid=*/1, {trainer}, samplers);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(), HasSubstr("Trainer push cut short"));
  EXPECT_EQ(slow.GetTransferStatus("cut_short"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED);
}

TEST(ControllerV3TransferTimeoutTest, PerTransferTimeoutOverridesTheDefault) {
  RaidenControllerV3 controller(SilentSamplerOptions(/*transfer_timeout_ms=*/
                                                     600000));
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);
  ASSERT_OK(controller.BuildMaterializedPlan("override", /*uuid=*/1, {trainer},
                                             samplers));

  // A shorter override ends the transfer long before the 10 min default.
  const absl::Time start = absl::Now();
  EXPECT_EQ(controller
                .ExecuteMaterializedTransferSync(
                    "override", /*expected_uuid=*/1,
                    /*rpc_sender_override=*/nullptr, absl::Milliseconds(200))
                .code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(
      controller
          .ExecuteTransferSync("override", /*uuid=*/1, {trainer}, samplers,
                               tpu_sync::rpc::MEMORY_TYPE_DRAM,
                               /*skip_d2h=*/false, /*parallelism=*/1,
                               /*skip_tiling=*/{}, absl::Milliseconds(200))
          .code(),
      absl::StatusCode::kDeadlineExceeded);
  EXPECT_LT(absl::Now() - start, absl::Seconds(30));
  // The timeout is an execution option: neither the override nor the
  // setter forces a new plan.
  controller.set_transfer_timeout_ms(100);
  EXPECT_EQ(controller.transfer_timeout_ms(), 100);
  EXPECT_EQ(
      controller
          .ExecuteTransferSync("override", /*uuid=*/1, {trainer}, samplers)
          .code(),
      absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);
  EXPECT_EQ(controller.GetPlanCacheSize(), 1);

  // A non-positive override is rejected without claiming the transfer.
  EXPECT_EQ(controller
                .ExecuteMaterializedTransferSync(
                    "override", /*expected_uuid=*/0,
                    /*rpc_sender_override=*/nullptr, absl::ZeroDuration())
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(controller
                .StartTransferAsync("override", /*uuid=*/1, {trainer}, samplers,
                                    tpu_sync::rpc::MEMORY_TYPE_DRAM,
                                    /*skip_d2h=*/false, /*parallelism=*/1,
                                    /*skip_tiling=*/{}, absl::Seconds(-1))
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(controller.GetTransferStatus("override"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED);
  controller.set_transfer_timeout_ms(0);  // Clamped to 1 ms.
  EXPECT_EQ(controller.transfer_timeout_ms(), 1);

  // A longer override lets a transfer outlive the default.
  RaidenControllerV3::Options opts;
  opts.transfer_timeout_ms = 100;
  opts.custom_rpc_sender = [](absl::string_view ep,
                              const tpu_sync::rpc::ControlRequest& req) {
    if (req.start_transfer_request().is_sender() && ep == "127.0.0.1:9100") {
      absl::SleepFor(absl::Milliseconds(300));
    }
    return OkResponse();
  };
  RaidenControllerV3 slow(opts);
  samplers = RegisterSimpleTopology(slow, 2);
  EXPECT_EQ(
      slow.ExecuteTransferSync("slow", /*uuid=*/1, {trainer}, samplers).code(),
      absl::StatusCode::kDeadlineExceeded);
  EXPECT_OK(slow.ExecuteTransferSync("slow", /*uuid=*/1, {trainer}, samplers,
                                     tpu_sync::rpc::MEMORY_TYPE_DRAM,
                                     /*skip_d2h=*/false, /*parallelism=*/1,
                                     /*skip_tiling=*/{}, absl::Seconds(30)));
}

TEST(ControllerV3TransferTimeoutTest, WaitForTransferFollowsTheDeadline) {
  RaidenControllerV3 controller(SilentSamplerOptions(/*transfer_timeout_ms=*/
                                                     600000));
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);

  // By default the waiter gets the transfer's own DEADLINE_EXCEEDED failure.
  ASSERT_OK(controller.StartTransferAsync(
      "async", /*uuid=*/1, {trainer}, samplers, tpu_sync::rpc::MEMORY_TYPE_DRAM,
      /*skip_d2h=*/false, /*parallelism=*/1, /*skip_tiling=*/{},
      absl::Milliseconds(300)));
  // An explicit timeout still works, and giving up does not fail the
  // transfer.
  absl::Status gave_up =
      controller.WaitForTransfer("async", absl::Milliseconds(10));
  EXPECT_EQ(gave_up.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(gave_up.message(), HasSubstr("Timed out waiting"));
  EXPECT_EQ(controller.GetTransferStatus("async"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS);
  absl::Status status = controller.WaitForTransfer("async");
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer async passed its deadline"));
  EXPECT_EQ(controller.GetTransferStatus("async"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_FAILED);

  // A plan that starts while the waiter waits: the waiter follows it.
  RaidenControllerV3::Options opts;
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    return OkResponse();
  };
  RaidenControllerV3 plain(opts);
  samplers = RegisterSimpleTopology(plain, 2);
  ASSERT_OK(
      plain.BuildMaterializedPlan("pending", /*uuid=*/1, {trainer}, samplers));
  std::thread executor([&plain] {
    absl::SleepFor(absl::Milliseconds(100));
    EXPECT_OK(plain.ExecuteMaterializedTransferSync("pending"));
  });
  EXPECT_OK(plain.WaitForTransfer("pending"));
  executor.join();

  // A transfer that overruns its deadline (here a worker RPC that ignores
  // it): the default wait gives up shortly after the deadline, while the
  // transfer is still running.
  absl::Notification release;
  RaidenControllerV3::Options stuck_opts;
  stuck_opts.transfer_timeout_ms = 100;
  stuck_opts.custom_rpc_sender = [&release](
                                     absl::string_view ep,
                                     const tpu_sync::rpc::ControlRequest& req) {
    if (req.start_transfer_request().is_sender() && ep == "127.0.0.1:9100") {
      release.WaitForNotification();
    }
    return OkResponse();
  };
  RaidenControllerV3 stuck(stuck_opts);
  samplers = RegisterSimpleTopology(stuck, 2);
  ASSERT_OK(stuck.StartTransferAsync("stuck", /*uuid=*/1, {trainer}, samplers));
  const absl::Time start = absl::Now();
  status = stuck.WaitForTransfer("stuck");
  const absl::Duration waited = absl::Now() - start;
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(), HasSubstr("past its deadline"));
  EXPECT_GE(waited, RaidenControllerV3::kWaitGraceAfterDeadline);
  EXPECT_LT(waited, absl::Seconds(60));
  EXPECT_EQ(stuck.GetTransferStatus("stuck"),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_IN_PROGRESS);
  // Its own failure follows once the RPC returns; the default wait would end
  // right away now that the grace period is over.
  release.Notify();
  status = stuck.WaitForTransfer("stuck", absl::Seconds(10));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer stuck passed its deadline"));
}

// A sampler host running the pull loop against the controller: it pulls
// every grant at once and reports it in its next request.
class LoopingSampler {
 public:
  using CallFn = std::function<void(PullServiceRequest, PullServiceCallback)>;

  LoopingSampler(CallFn call, std::string req_id, uint64_t uuid, RaidenId unit,
                 bool seed)
      : call_(std::move(call)),
        req_id_(std::move(req_id)),
        uuid_(uuid),
        unit_(std::move(unit)),
        seed_(seed) {}

  void Start() { Ask({}, seed_); }

  // Waits for the final (`kDone` or `kAborted`) reply.
  PullServiceReply Wait() {
    absl::MutexLock lock(mu_);
    mu_.Await(absl::Condition(&finished_));
    return final_;
  }

 private:
  void Ask(std::vector<uint64_t> completed, bool seeded) {
    call_({.req_id = req_id_,
           .uuid = uuid_,
           .unit = unit_,
           .max_grants = 8,
           .long_poll = absl::Seconds(10),
           .seeded = seeded,
           .completed = std::move(completed)},
          [this](PullServiceReply reply) {
            if (reply.kind != PullReplyKind::kGrants) {
              absl::MutexLock lock(mu_);
              final_ = std::move(reply);
              finished_ = true;
              return;
            }
            std::vector<uint64_t> done;
            for (const PhysicalPullGrant& grant : reply.grants) {
              done.push_back(grant.lease_id);
            }
            Ask(std::move(done), /*seeded=*/false);
          });
  }

  const CallFn call_;
  const std::string req_id_;
  const uint64_t uuid_;
  const RaidenId unit_;
  const bool seed_;
  absl::Mutex mu_;
  bool finished_ ABSL_GUARDED_BY(mu_) = false;
  PullServiceReply final_ ABSL_GUARDED_BY(mu_);
};

TEST(ControllerV3PullServiceTest, ScheduledPullPhaseCompletesThroughPullRpcs) {
  RaidenControllerV3::Options opts = SilentSamplerOptions(600000);
  opts.num_bundle_groups = 4;
  RaidenControllerV3 controller(opts);
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterLayeredTopology(controller, 8, 4);
  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan("sched", /*uuid=*/1, {trainer},
                                       samplers);
  ASSERT_OK(plan);
  ASSERT_OK(
      controller.StartTransferAsync("sched", /*uuid=*/1, {trainer}, samplers));
  ASSERT_TRUE(WaitForPullService(controller, "sched"));
  EXPECT_EQ(controller.GetPlanMaterializationCountForTest(), 1);

  // `AcquirePulls` carries no reports; unknown transfers are refused.
  absl::Notification acquired;
  controller.AcquirePulls(
      {.req_id = "sched", .uuid = 1, .unit = samplers[0], .completed = {7}},
      [&](PullServiceReply reply) {
        EXPECT_EQ(reply.kind, PullReplyKind::kAborted);
        EXPECT_EQ(reply.status.code(), absl::StatusCode::kInvalidArgument);
        acquired.Notify();
      });
  acquired.WaitForNotification();
  PullServiceReply unknown = CallController(
      controller, {.req_id = "unknown", .uuid = 1, .unit = samplers[0]});
  EXPECT_EQ(unknown.status.code(), absl::StatusCode::kNotFound);

  const absl::flat_hash_set<int32_t> seeds = SeedReplicas(*plan);
  std::vector<std::unique_ptr<LoopingSampler>> hosts;
  for (size_t r = 0; r < samplers.size(); ++r) {
    hosts.push_back(std::make_unique<LoopingSampler>(
        [&controller](PullServiceRequest req, PullServiceCallback done) {
          controller.ReportPulls(std::move(req), std::move(done));
        },
        "sched", 1, samplers[r], seeds.contains(static_cast<int32_t>(r))));
  }
  for (auto& host : hosts) host->Start();
  for (auto& host : hosts) EXPECT_EQ(host->Wait().kind, PullReplyKind::kDone);
  ASSERT_OK(controller.WaitForTransfer("sched", absl::Seconds(30)));
  // 8 replicas x 4 bundles, 2 copies of each come from the Trainer.
  EXPECT_EQ(ExpectPullPhaseComplete(controller, "sched"), 8 * 4 - 2 * 4);
  absl::StatusOr<std::vector<PullShardStats>> stats =
      controller.GetPullStats("sched");
  ASSERT_OK(stats);
  ASSERT_THAT(*stats, SizeIs(1));
  EXPECT_EQ((*stats)[0].done_hosts, 8);
  EXPECT_EQ(controller.GetPullStats("unknown").status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ControllerV3PullServiceTest, SchedulerAbortFailsTheTransfer) {
  RaidenControllerV3 controller(SilentSamplerOptions(600000));
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterSimpleTopology(controller, 2);
  ASSERT_OK(
      controller.StartTransferAsync("silent", /*uuid=*/1, {trainer}, samplers));
  ASSERT_TRUE(WaitForPullService(controller, "silent"));
  controller.GetPullServiceForTest("silent")->scheduler().Abort(
      absl::DataLossError("disk on fire"));
  absl::Status status = controller.WaitForTransfer("silent", absl::Seconds(30));
  EXPECT_EQ(status.code(), absl::StatusCode::kDataLoss);
  EXPECT_THAT(status.message(), HasSubstr("disk on fire"));
}

TEST(ControllerV3PullServerTest, ServesPullRpcsWithTheTestEncoding) {
  RaidenControllerV3::Options opts = SilentSamplerOptions(600000);
  opts.num_bundle_groups = 4;
  opts.pull_server_port = 0;
  opts.pull_wire_codec = std::make_shared<TestPullWireCodec>();
  RaidenControllerV3 controller(opts);
  EXPECT_EQ(controller.pull_server_port(), 0);
  ASSERT_OK(controller.StartServer());
  ASSERT_GT(controller.pull_server_port(), 0);
  const std::string endpoint =
      absl::StrCat("127.0.0.1:", controller.pull_server_port());
  RaidenId trainer{"trainer", "0", "weights", 0};
  std::vector<RaidenId> samplers = RegisterLayeredTopology(controller, 4, 4);
  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan("wire", /*uuid=*/1, {trainer}, samplers);
  ASSERT_OK(plan);
  ASSERT_OK(
      controller.StartTransferAsync("wire", /*uuid=*/1, {trainer}, samplers));
  ASSERT_TRUE(WaitForPullService(controller, "wire"));

  ControlPipeConfig config;
  config.backend_type = ControlPipeBackendType::kTcp;
  // The pull server serves nothing else.
  std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(config);
  {
    // A `ControlRequest` sent with CPIP framing (not the legacy framing).
    ControlPipeConfig cpip_config = config;
    cpip_config.allow_legacy_framing = false;
    std::unique_ptr<ControlPipeClient> cpip_client =
        CreateControlPipeClient(cpip_config);
    tpu_sync::rpc::ControlRequest other;
    other.set_command(tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA);
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type("tpu_sync.rpc.ControlRequest");
    env.set_request_id(1);
    env.set_payload(other.SerializeAsString());
    absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> resp =
        cpip_client->SendRaw(endpoint, env, absl::Seconds(10));
    ASSERT_OK(resp);
    EXPECT_EQ(resp->status_code(),
              static_cast<int32_t>(absl::StatusCode::kUnimplemented));
  }

  // Every sampler runs a blocking pull loop on its own connection.
  const absl::flat_hash_set<int32_t> seeds = SeedReplicas(*plan);
  std::atomic<int> done_hosts{0};
  std::vector<std::thread> threads;
  for (size_t r = 0; r < samplers.size(); ++r) {
    threads.emplace_back([&, r] {
      std::unique_ptr<ControlPipeClient> c = CreateControlPipeClient(config);
      PullServiceRequest req{.req_id = "wire",
                             .uuid = 1,
                             .unit = samplers[r],
                             .max_grants = 8,
                             .long_poll = absl::Seconds(10),
                             .seeded = seeds.contains(static_cast<int32_t>(r))};
      for (int i = 0; i < 1000; ++i) {
        control_pipe::proto::ControlEnvelope env;
        env.set_message_type(std::string(TestPullWireCodec::kMessageType));
        env.set_request_id(i + 1);
        env.set_payload(TestPullWireCodec::EncodeRequest(req));
        absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> resp =
            c->SendRaw(endpoint, env, absl::Seconds(30));
        ASSERT_OK(resp);
        ASSERT_EQ(resp->status_code(), 0) << resp->error_message();
        absl::StatusOr<PullServiceReply> reply =
            TestPullWireCodec::DecodeReply(resp->payload());
        ASSERT_OK(reply);
        if (reply->kind == PullReplyKind::kDone) {
          ++done_hosts;
          return;
        }
        ASSERT_EQ(reply->kind, PullReplyKind::kGrants) << reply->status;
        req.seeded = false;
        req.completed.clear();
        for (const PhysicalPullGrant& grant : reply->grants) {
          EXPECT_FALSE(grant.source_data_endpoint.empty());
          req.completed.push_back(grant.lease_id);
        }
      }
    });
  }
  for (std::thread& t : threads) t.join();
  EXPECT_EQ(done_hosts.load(), 4);
  ASSERT_OK(controller.WaitForTransfer("wire", absl::Seconds(30)));
  // 4 replicas x 4 bundles, 2 copies of each come from the Trainer.
  EXPECT_EQ(ExpectPullPhaseComplete(controller, "wire"), 4 * 4 - 2 * 4);
  std::optional<AsyncControlServer::Stats> stats =
      controller.GetPullServerStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_GE(stats->requests, 5);
  EXPECT_EQ(stats->requests, stats->responses);
  controller.StopServer();
  EXPECT_FALSE(controller.GetPullServerStats().has_value());

  // Without a codec the pull RPCs are not served yet.
  RaidenControllerV3::Options bare_opts;
  bare_opts.pull_server_port = 0;
  RaidenControllerV3 bare(bare_opts);
  ASSERT_OK(bare.StartServer());
  control_pipe::proto::ControlEnvelope env;
  env.set_message_type(std::string(TestPullWireCodec::kMessageType));
  env.set_request_id(1);
  env.set_payload(TestPullWireCodec::EncodeRequest({.req_id = "wire"}));
  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> resp =
      client->SendRaw(absl::StrCat("127.0.0.1:", bare.pull_server_port()), env,
                      absl::Seconds(10));
  ASSERT_OK(resp);
  EXPECT_EQ(resp->status_code(),
            static_cast<int32_t>(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(resp->error_message(), HasSubstr("UNIMPLEMENTED"));
  bare.StopServer();
}

TEST(ControllerV3PullServerTest, PullCommandsAreNotOnTheWireYet) {
  RaidenControllerV3 controller(RaidenControllerV3::Options{});
  tpu_sync::rpc::ControlRequest req;
  req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_UNSPECIFIED);
  tpu_sync::rpc::ControlResponse resp = controller.HandleControlRequest(req);
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("UNIMPLEMENTED"));
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
